// Numerical coherence test for the Kimi K3 kernels.
//
// Each kernel is checked against an INDEPENDENT float64 CPU implementation of the
// same formula, written from the reference semantics rather than from the CUDA code,
// on pseudo-random inputs. That independence is the point: if both sides were derived
// from the same transcription, a misreading of the reference would agree with itself.
//
// Why this shape of test. Every K3-specific op is a silent-wrong-output risk, not a
// crash risk — collapse KDA's per-channel decay to a scalar, or use normalised values
// where the reference uses raw ones, and the model still emits fluent text. There is
// no assertion available at the token level that catches that. A float64 reference on
// random inputs does.
//
// Each case also asserts a WRONG-BUT-PLAUSIBLE variant is measurably different, so a
// tolerance that happens to be loose enough to hide the bug fails the test instead.
//
// Build/run (needs a GPU):
//   nvcc -std=c++17 -arch=sm_90 -I kernels/include \
//     kernels/tests/kimi_k3_numeric_test.cu kernels/csrc/cuda/kimi_k3/k3_kernels.cu \
//     -o /tmp/k3test && /tmp/k3test

#include "sparkinfer/kernels/kimi_k3.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace sparkinfer::kernels::k3;

static int g_fail = 0;
static int g_case = 0;

#define CU(expr)                                                                 \
    do {                                                                         \
        cudaError_t e_ = (expr);                                                  \
        if (e_ != cudaSuccess) {                                                 \
            std::printf("  CUDA FAIL %s:%d %s: %s\n", __FILE__, __LINE__, #expr,  \
                        cudaGetErrorString(e_));                                 \
            std::exit(2);                                                        \
        }                                                                        \
    } while (0)

// Relative-L2 against a float64 reference. f32 kernels using __expf/rsqrtf will not
// match to 1e-7; 2e-5 is tight enough that any structural error (wrong index, wrong
// activation, missing term) fails by orders of magnitude, and loose enough that fast
// intrinsics do not.
static bool close_enough(const std::vector<float>& got, const std::vector<double>& want,
                         const char* what, double tol = 2e-5) {
    ++g_case;
    if (got.size() != want.size()) {
        std::printf("  FAIL %-34s size %zu vs %zu\n", what, got.size(), want.size());
        ++g_fail;
        return false;
    }
    double num = 0.0, den = 0.0, worst = 0.0;
    size_t worst_i = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = (double)got[i] - want[i];
        num += d * d;
        den += want[i] * want[i];
        const double rel = std::fabs(d) / (std::fabs(want[i]) + 1e-12);
        if (rel > worst) { worst = rel; worst_i = i; }
    }
    const double rl2 = std::sqrt(num / (den + 1e-30));
    const bool ok = rl2 <= tol;
    std::printf("  %-34s relL2=%.3e  worst_rel=%.3e @%zu  %s\n", what, rl2, worst, worst_i,
                ok ? "OK" : "FAIL");
    if (!ok) {
        ++g_fail;
        for (size_t i = 0; i < got.size() && i < 4; ++i)
            std::printf("        [%zu] got %.9g want %.9g\n", i, (double)got[i], want[i]);
    }
    return ok;
}

// Assert two references genuinely differ, i.e. the tolerance cannot hide the bug the
// test claims to catch.
static void assert_variant_differs(const std::vector<double>& a,
                                   const std::vector<double>& b, const char* what) {
    ++g_case;
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        num += d * d;
        den += a[i] * a[i];
    }
    const double rl2 = std::sqrt(num / (den + 1e-30));
    const bool ok = rl2 > 1e-3;
    std::printf("  %-34s relL2=%.3e vs wrong variant  %s\n", what, rl2,
                ok ? "OK (distinguishable)" : "FAIL (tolerance would hide it)");
    if (!ok) ++g_fail;
}

static std::vector<float> rnd(size_t n, std::mt19937& rng, float lo = -1.0f, float hi = 1.0f) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

template <class T>
static T* to_dev(const std::vector<T>& h) {
    T* p = nullptr;
    CU(cudaMalloc(&p, h.size() * sizeof(T)));
    CU(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

static std::vector<float> from_dev(const float* p, size_t n) {
    std::vector<float> h(n);
    CU(cudaMemcpy(h.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

// ---------------------------------------------------------------------------
// 1. situ
// ---------------------------------------------------------------------------

static void test_situ() {
    std::printf("situ activation (beta=4.0, linear_beta=25.0 — K3's values)\n");
    std::mt19937 rng(1234);
    const int64_t n = 100000;
    // Wide range so tanh saturation and sigmoid tails are both exercised.
    auto gate = rnd(n, rng, -30.f, 30.f);
    auto up = rnd(n, rng, -60.f, 60.f);
    const float beta = 4.0f, lb = 25.0f;

    std::vector<double> ref(n), wrong_symmetric(n);
    for (int64_t i = 0; i < n; ++i) {
        const double g = gate[i], u = up[i];
        // reference: beta*tanh(g/beta)*sigmoid(g) * lb*tanh(u/lb)
        const double a = beta * std::tanh(g / beta) * (1.0 / (1.0 + std::exp(-g)));
        ref[i] = a * (lb * std::tanh(u / lb));
        // the plausible wrong version: sigmoid on BOTH branches (SwiGLU-ish)
        wrong_symmetric[i] = a * (lb * std::tanh(u / lb)) * (1.0 / (1.0 + std::exp(-u)));
    }
    assert_variant_differs(ref, wrong_symmetric, "vs sigmoid-on-both-branches");

    float *dg = to_dev(gate), *du = to_dev(up), *dout = nullptr;
    CU(cudaMalloc(&dout, n * sizeof(float)));
    situ_f32(dout, dg, du, n, beta, lb, 0);
    CU(cudaDeviceSynchronize());
    close_enough(from_dev(dout, n), ref, "situ vs float64 reference");

    // linear_beta <= 0 must use the up value RAW.
    std::vector<double> ref_nolb(n);
    for (int64_t i = 0; i < n; ++i) {
        const double g = gate[i];
        ref_nolb[i] = beta * std::tanh(g / beta) * (1.0 / (1.0 + std::exp(-g))) * up[i];
    }
    situ_f32(dout, dg, du, n, beta, 0.0f, 0);
    CU(cudaDeviceSynchronize());
    close_enough(from_dev(dout, n), ref_nolb, "situ linear_beta<=0 (up raw)");

    CU(cudaFree(dg)); CU(cudaFree(du)); CU(cudaFree(dout));
}

// ---------------------------------------------------------------------------
// 2. KDA decode step
// ---------------------------------------------------------------------------

static void test_kda_decode_step() {
    std::printf("KDA decode step (gated delta rule, per-channel decay)\n");
    std::mt19937 rng(99);
    const int D = 128;      // K3's kda head_dim
    const int H = 12;       // a few heads; K3 has 96
    auto q = rnd((size_t)D * H, rng);
    auto k = rnd((size_t)D * H, rng);
    auto v = rnd((size_t)D * H, rng);
    // g is the log-decay: K3 produces lb*sigmoid(...) with lb=-5, so g in (-5, 0).
    auto g = rnd((size_t)D * H, rng, -5.0f, 0.0f);
    auto beta = rnd((size_t)H, rng, 0.0f, 1.0f);
    auto s0 = rnd((size_t)D * D * H, rng, -0.5f, 0.5f);

    // L2-normalise q,k per head and scale q by 1/sqrt(D) — the caller's job in
    // kimi-k3.cpp, replicated here so the kernel sees what it will really see.
    for (int h = 0; h < H; ++h) {
        double nq = 0, nk = 0;
        for (int d = 0; d < D; ++d) { nq += (double)q[h*D+d]*q[h*D+d]; nk += (double)k[h*D+d]*k[h*D+d]; }
        nq = 1.0/std::sqrt(nq + 1e-6); nk = 1.0/std::sqrt(nk + 1e-6);
        const double qs = nq / std::sqrt((double)D);
        for (int d = 0; d < D; ++d) { q[h*D+d] = (float)(q[h*D+d]*qs); k[h*D+d] = (float)(k[h*D+d]*nk); }
    }

    // --- float64 reference, written from the formula, not from the kernel ---
    // S[i][j] at s[j*D + i]
    std::vector<double> S(s0.begin(), s0.end());
    std::vector<double> ref_out((size_t)D * H, 0.0);
    std::vector<double> ref_state = S;
    for (int h = 0; h < H; ++h) {
        double* Sh = ref_state.data() + (size_t)h * D * D;
        const float* qh = &q[h*D]; const float* kh = &k[h*D];
        const float* vh = &v[h*D]; const float* gh = &g[h*D];
        const double b = beta[h];
        // 1. per-channel decay
        for (int j = 0; j < D; ++j) {
            const double dec = std::exp((double)gh[j]);
            for (int i = 0; i < D; ++i) Sh[(size_t)j*D + i] *= dec;
        }
        // 2. sk[j] = sum_i S[i][j]*k[i]
        std::vector<double> sk(D, 0.0);
        for (int j = 0; j < D; ++j)
            for (int i = 0; i < D; ++i) sk[j] += Sh[(size_t)j*D + i] * (double)kh[i];
        // 3. d[m] = b*(v[m]-sk[m])
        std::vector<double> dd(D);
        for (int m = 0; m < D; ++m) dd[m] = b * ((double)vh[m] - sk[m]);
        // 4. S[i][j] += k[i]*d[j]
        for (int j = 0; j < D; ++j)
            for (int i = 0; i < D; ++i) Sh[(size_t)j*D + i] += (double)kh[i] * dd[j];
        // 5. o[j] = sum_i S[i][j]*q[i]
        for (int j = 0; j < D; ++j) {
            double o = 0.0;
            for (int i = 0; i < D; ++i) o += Sh[(size_t)j*D + i] * (double)qh[i];
            ref_out[(size_t)h*D + j] = o;
        }
    }

    // The wrong-but-runnable variant: collapse the per-channel decay to a scalar
    // (i.e. treat KDA as GDA). Confirm the test can tell them apart.
    {
        std::vector<double> Sw(s0.begin(), s0.end());
        std::vector<double> out_w((size_t)D * H, 0.0);
        for (int h = 0; h < H; ++h) {
            double* Sh = Sw.data() + (size_t)h * D * D;
            const float* qh = &q[h*D]; const float* kh = &k[h*D];
            const float* vh = &v[h*D]; const float* gh = &g[h*D];
            const double b = beta[h];
            const double dec = std::exp((double)gh[0]);   // scalar decay — the bug
            for (size_t t = 0; t < (size_t)D*D; ++t) Sh[t] *= dec;
            std::vector<double> sk(D, 0.0);
            for (int j = 0; j < D; ++j)
                for (int i = 0; i < D; ++i) sk[j] += Sh[(size_t)j*D + i] * (double)kh[i];
            std::vector<double> dd(D);
            for (int m = 0; m < D; ++m) dd[m] = b * ((double)vh[m] - sk[m]);
            for (int j = 0; j < D; ++j)
                for (int i = 0; i < D; ++i) Sh[(size_t)j*D + i] += (double)kh[i] * dd[j];
            for (int j = 0; j < D; ++j) {
                double o = 0.0;
                for (int i = 0; i < D; ++i) o += Sh[(size_t)j*D + i] * (double)qh[i];
                out_w[(size_t)h*D + j] = o;
            }
        }
        assert_variant_differs(ref_out, out_w, "vs scalar-decay (GDA not KDA)");
    }

    float *dq = to_dev(q), *dk = to_dev(k), *dv = to_dev(v), *dg = to_dev(g),
          *db = to_dev(beta), *ds = to_dev(s0), *dout = nullptr;
    CU(cudaMalloc(&dout, (size_t)D * H * sizeof(float)));
    kda_decode_step_f32(dout, ds, dq, dk, dv, dg, db, D, H, 0);
    CU(cudaDeviceSynchronize());
    CU(cudaGetLastError());

    close_enough(from_dev(dout, (size_t)D * H), ref_out, "kda out vs float64 reference");
    close_enough(from_dev(ds, (size_t)D * D * H), ref_state, "kda state vs float64 reference");

    CU(cudaFree(dq)); CU(cudaFree(dk)); CU(cudaFree(dv)); CU(cudaFree(dg));
    CU(cudaFree(db)); CU(cudaFree(ds)); CU(cudaFree(dout));
}

// ---------------------------------------------------------------------------
// 3. KDA output gating
// ---------------------------------------------------------------------------

static void test_kda_gate_out() {
    std::printf("KDA output gating (rms_norm * sigmoid(full-rank gate))\n");
    std::mt19937 rng(7);
    const int D = 128, H = 16;
    auto o = rnd((size_t)D * H, rng, -3.f, 3.f);
    auto nw = rnd((size_t)D, rng, 0.5f, 1.5f);
    auto g2 = rnd((size_t)D * H, rng, -6.f, 6.f);
    const float eps = 1e-5f;

    std::vector<double> ref((size_t)D * H);
    for (int h = 0; h < H; ++h) {
        double ss = 0.0;
        for (int d = 0; d < D; ++d) ss += (double)o[h*D+d] * o[h*D+d];
        const double inv = 1.0 / std::sqrt(ss / (double)D + eps);
        for (int d = 0; d < D; ++d) {
            const double normed = (double)o[h*D+d] * inv * (double)nw[d];
            ref[(size_t)h*D+d] = normed * (1.0 / (1.0 + std::exp(-(double)g2[h*D+d])));
        }
    }

    float *dO = to_dev(o), *dnw = to_dev(nw), *dg2 = to_dev(g2), *dout = nullptr;
    CU(cudaMalloc(&dout, (size_t)D * H * sizeof(float)));
    kda_gate_out_f32(dout, dO, dnw, dg2, D, H, eps, 0);
    CU(cudaDeviceSynchronize());
    CU(cudaGetLastError());
    close_enough(from_dev(dout, (size_t)D * H), ref, "kda gate vs float64 reference");
    CU(cudaFree(dO)); CU(cudaFree(dnw)); CU(cudaFree(dg2)); CU(cudaFree(dout));
}

// ---------------------------------------------------------------------------
// 4. cross-layer attention residual mix
// ---------------------------------------------------------------------------

static void test_attn_res_mix() {
    std::printf("cross-layer attn residual mix (scores normalised, sum over RAW)\n");
    std::mt19937 rng(4242);
    const int E = 7168;    // K3 hidden
    const int C = 7;       // banked checkpoints
    auto ck = rnd((size_t)E * C, rng, -2.f, 2.f);
    auto cur = rnd((size_t)E, rng, -2.f, 2.f);
    auto sw = rnd((size_t)E, rng, -0.05f, 0.05f);
    const float eps = 1e-5f;

    auto score_of = [&](const float* x) {
        double ss = 0.0;
        for (int d = 0; d < E; ++d) ss += (double)x[d] * x[d];
        const double inv = 1.0 / std::sqrt(ss / (double)E + eps);
        double dot = 0.0;
        for (int d = 0; d < E; ++d) dot += ((double)x[d] * inv) * (double)sw[d];
        return dot;
    };

    std::vector<double> sc(C + 1);
    for (int c = 0; c < C; ++c) sc[c] = score_of(&ck[(size_t)c * E]);
    sc[C] = score_of(cur.data());
    double mx = sc[0];
    for (int c = 1; c <= C; ++c) mx = std::max(mx, sc[c]);
    double sum = 0.0;
    std::vector<double> p(C + 1);
    for (int c = 0; c <= C; ++c) { p[c] = std::exp(sc[c] - mx); sum += p[c]; }
    for (int c = 0; c <= C; ++c) p[c] /= sum;

    std::vector<double> ref(E), wrong_normalised(E);
    for (int d = 0; d < E; ++d) {
        double acc = 0.0;
        for (int c = 0; c < C; ++c) acc += p[c] * (double)ck[(size_t)c*E + d];
        ref[d] = acc + p[C] * (double)cur[d];
    }
    // The plausible wrong version: weight the NORMALISED values instead of the raw
    // ones. This is the natural thing to write and the reference explicitly is not it.
    {
        std::vector<double> invs(C + 1);
        for (int c = 0; c < C; ++c) {
            double ss = 0.0;
            for (int d = 0; d < E; ++d) ss += (double)ck[(size_t)c*E+d]*ck[(size_t)c*E+d];
            invs[c] = 1.0 / std::sqrt(ss / (double)E + eps);
        }
        double ss = 0.0;
        for (int d = 0; d < E; ++d) ss += (double)cur[d]*cur[d];
        invs[C] = 1.0 / std::sqrt(ss / (double)E + eps);
        for (int d = 0; d < E; ++d) {
            double acc = 0.0;
            for (int c = 0; c < C; ++c) acc += p[c] * (double)ck[(size_t)c*E+d] * invs[c];
            wrong_normalised[d] = acc + p[C] * (double)cur[d] * invs[C];
        }
    }
    assert_variant_differs(ref, wrong_normalised, "vs weighting normalised values");

    float *dck = to_dev(ck), *dcur = to_dev(cur), *dsw = to_dev(sw), *dout = nullptr;
    CU(cudaMalloc(&dout, (size_t)E * sizeof(float)));
    attn_res_mix_f32(dout, dck, dcur, dsw, E, C, eps, 0);
    CU(cudaDeviceSynchronize());
    CU(cudaGetLastError());
    close_enough(from_dev(dout, E), ref, "attn_res_mix vs float64 reference");

    // n_ckpt == 0 must pass cur through unchanged (layer 0, nothing banked).
    attn_res_mix_f32(dout, dck, dcur, dsw, E, 0, eps, 0);
    CU(cudaDeviceSynchronize());
    std::vector<double> ref_pass(cur.begin(), cur.end());
    close_enough(from_dev(dout, E), ref_pass, "attn_res_mix n_ckpt=0 passthrough");

    CU(cudaFree(dck)); CU(cudaFree(dcur)); CU(cudaFree(dsw)); CU(cudaFree(dout));
}

// ---------------------------------------------------------------------------
// 5. MLA output gate
// ---------------------------------------------------------------------------

static void test_mla_gate_out() {
    std::printf("MLA output gate (sigmoid before o_proj)\n");
    std::mt19937 rng(55);
    const int64_t n = 96 * 128;
    auto a = rnd(n, rng, -4.f, 4.f);
    auto gp = rnd(n, rng, -8.f, 8.f);
    std::vector<double> ref(n);
    for (int64_t i = 0; i < n; ++i)
        ref[i] = (double)a[i] * (1.0 / (1.0 + std::exp(-(double)gp[i])));

    float *da = to_dev(a), *dg = to_dev(gp), *dout = nullptr;
    CU(cudaMalloc(&dout, n * sizeof(float)));
    mla_gate_out_f32(dout, da, dg, n, 0);
    CU(cudaDeviceSynchronize());
    CU(cudaGetLastError());
    close_enough(from_dev(dout, n), ref, "mla gate vs float64 reference");
    CU(cudaFree(da)); CU(cudaFree(dg)); CU(cudaFree(dout));
}

// ---------------------------------------------------------------------------

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no CUDA device — skipping (not a failure)\n");
        return 0;
    }
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, 0);
    std::printf("device: %s sm_%d%d\n\n", p.name, p.major, p.minor);

    test_situ();               std::printf("\n");
    test_kda_decode_step();    std::printf("\n");
    test_kda_gate_out();       std::printf("\n");
    test_attn_res_mix();       std::printf("\n");
    test_mla_gate_out();       std::printf("\n");

    std::printf("%d cases, %d failure(s)\n", g_case, g_fail);
    return g_fail == 0 ? 0 : 1;
}
