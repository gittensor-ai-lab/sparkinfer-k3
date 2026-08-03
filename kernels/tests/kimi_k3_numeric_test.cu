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

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstring>

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
        // 1. per-channel decay, indexed by i (the CONTRACTION index), matching
        // ggml_compute_forward_gated_delta_net_one_chunk: "S[i][:] *= exp(g[i])".
        // This reference previously used exp(g[j]) — the same wrong axis as the
        // kernel it was supposed to check independently, because both were
        // transcribed from the same (dead) llama.cpp path. It therefore agreed with
        // the bug. An "independent" reference derived from the same source as the
        // implementation is not independent.
        for (int j = 0; j < D; ++j) {
            for (int i = 0; i < D; ++i) Sh[(size_t)j*D + i] *= std::exp((double)gh[i]);
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
// 6. KDA causal short conv, decode step
// ---------------------------------------------------------------------------

static void test_kda_conv_step() {
    std::printf("KDA causal short conv, decode step (d_conv=4, depthwise, silu after)\n");
    std::mt19937 rng(31337);
    const int d_conv = 4;                  // K3's short_conv_kernel_size
    const int d_inner = 96 * 128;          // n_head_kda * head_dim
    auto x = rnd((size_t)d_inner, rng, -2.f, 2.f);
    auto w = rnd((size_t)d_conv * d_inner, rng, -1.f, 1.f);
    auto st0 = rnd((size_t)(d_conv - 1) * d_inner, rng, -2.f, 2.f);

    // --- float64 reference ---
    std::vector<double> ref_out(d_inner), ref_state((size_t)(d_conv - 1) * d_inner);
    for (int c = 0; c < d_inner; ++c) {
        const float* stc = &st0[(size_t)c * (d_conv - 1)];
        const float* wc = &w[(size_t)c * d_conv];
        double acc = 0.0;
        for (int tt = 0; tt < d_conv - 1; ++tt) acc += (double)stc[tt] * (double)wc[tt];
        acc += (double)x[c] * (double)wc[d_conv - 1];       // current token is LAST
        ref_out[c] = acc * (1.0 / (1.0 + std::exp(-acc)));  // silu AFTER the conv
        for (int tt = 0; tt < d_conv - 2; ++tt)
            ref_state[(size_t)c * (d_conv - 1) + tt] = (double)stc[tt + 1];
        ref_state[(size_t)c * (d_conv - 1) + (d_conv - 2)] = (double)x[c];
    }

    // The plausible wrong version: reversed window, i.e. w[0] weights the current
    // token. Still a valid depthwise conv, still fluent output, different filter.
    {
        std::vector<double> wrong(d_inner);
        for (int c = 0; c < d_inner; ++c) {
            const float* stc = &st0[(size_t)c * (d_conv - 1)];
            const float* wc = &w[(size_t)c * d_conv];
            double acc = (double)x[c] * (double)wc[0];
            for (int tt = 0; tt < d_conv - 1; ++tt) acc += (double)stc[tt] * (double)wc[tt + 1];
            wrong[c] = acc * (1.0 / (1.0 + std::exp(-acc)));
        }
        assert_variant_differs(ref_out, wrong, "vs reversed conv window");
    }
    // And: silu BEFORE the conv instead of after.
    {
        std::vector<double> wrong(d_inner);
        for (int c = 0; c < d_inner; ++c) {
            const float* stc = &st0[(size_t)c * (d_conv - 1)];
            const float* wc = &w[(size_t)c * d_conv];
            auto silu = [](double z) { return z * (1.0 / (1.0 + std::exp(-z))); };
            double acc = 0.0;
            for (int tt = 0; tt < d_conv - 1; ++tt) acc += silu((double)stc[tt]) * (double)wc[tt];
            acc += silu((double)x[c]) * (double)wc[d_conv - 1];
            wrong[c] = acc;
        }
        assert_variant_differs(ref_out, wrong, "vs silu before the conv");
    }

    float *dx = to_dev(x), *dw = to_dev(w), *dst = to_dev(st0), *dout = nullptr;
    CU(cudaMalloc(&dout, (size_t)d_inner * sizeof(float)));
    kda_conv_step_f32(dout, dst, dx, dw, d_conv, d_inner, 0);
    CU(cudaDeviceSynchronize());
    CU(cudaGetLastError());
    close_enough(from_dev(dout, d_inner), ref_out, "kda conv out vs float64 ref");
    close_enough(from_dev(dst, (size_t)(d_conv - 1) * d_inner), ref_state,
                 "kda conv state shift vs float64 ref");

    CU(cudaFree(dx)); CU(cudaFree(dw)); CU(cudaFree(dst)); CU(cudaFree(dout));
}

// ---------------------------------------------------------------------------
// 7. KDA decay gate (lower_bound form)
// ---------------------------------------------------------------------------

static void test_kda_decay_gate() {
    std::printf("KDA decay gate (lower_bound*sigmoid — K3 form, NOT softplus)\n");
    std::mt19937 rng(2026);
    const int D = 128, H = 16;
    const float lb = -5.0f;   // K3's kda.gate_lower_bound
    auto g_raw = rnd((size_t)D * H, rng, -4.f, 4.f);
    // A = -exp(A_log); typically negative and small-magnitude after folding.
    auto A = rnd((size_t)H, rng, -2.0f, -0.05f);

    std::vector<double> ref((size_t)D * H), wrong_softplus((size_t)D * H);
    for (int h = 0; h < H; ++h) {
        const double Ah = A[h];
        for (int d = 0; d < D; ++d) {
            const double gr = g_raw[(size_t)h * D + d];
            // K3: lb * sigmoid(-(A * g_raw))
            ref[(size_t)h * D + d] =
                (double)lb * (1.0 / (1.0 + std::exp(Ah * gr)));
            // kimi-linear (unset lower_bound): A * softplus(g_raw)
            const double sp = std::log1p(std::exp(gr));
            wrong_softplus[(size_t)h * D + d] = Ah * sp;
        }
    }
    assert_variant_differs(ref, wrong_softplus, "vs softplus (kimi-linear form)");

    float *dg = to_dev(g_raw), *dA = to_dev(A), *dout = nullptr;
    CU(cudaMalloc(&dout, (size_t)D * H * sizeof(float)));
    kda_decay_gate_f32(dout, dg, dA, D, H, lb, 0);
    CU(cudaDeviceSynchronize());
    CU(cudaGetLastError());
    close_enough(from_dev(dout, (size_t)D * H), ref, "kda decay gate vs float64 ref");
    CU(cudaFree(dg)); CU(cudaFree(dA)); CU(cudaFree(dout));
}

// ---------------------------------------------------------------------------
// 8. L2-norm heads (+ optional scale)
// ---------------------------------------------------------------------------

static void test_l2_norm_heads() {
    std::printf("L2-norm heads (scale=1 and scale=1/sqrt(D) for KDA Q)\n");
    std::mt19937 rng(11);
    const int D = 128, H = 24;
    const float eps = 1e-6f;
    auto x = rnd((size_t)D * H, rng, -3.f, 3.f);

    auto make_ref = [&](double scale) {
        std::vector<double> ref((size_t)D * H);
        for (int h = 0; h < H; ++h) {
            double ss = 0.0;
            for (int d = 0; d < D; ++d) ss += (double)x[h * D + d] * x[h * D + d];
            const double inv = scale / std::sqrt(ss + eps);
            for (int d = 0; d < D; ++d) ref[(size_t)h * D + d] = (double)x[h * D + d] * inv;
        }
        return ref;
    };

    float *dx = to_dev(x), *dout = nullptr;
    CU(cudaMalloc(&dout, (size_t)D * H * sizeof(float)));

    l2_norm_heads_f32(dout, dx, D, H, 1.0f, eps, 0);
    CU(cudaDeviceSynchronize());
    CU(cudaGetLastError());
    close_enough(from_dev(dout, (size_t)D * H), make_ref(1.0), "l2_norm scale=1");

    const float qscale = 1.0f / std::sqrt((float)D);
    l2_norm_heads_f32(dout, dx, D, H, qscale, eps, 0);
    CU(cudaDeviceSynchronize());
    close_enough(from_dev(dout, (size_t)D * H), make_ref(qscale), "l2_norm scale=1/sqrt(D)");

    CU(cudaFree(dx)); CU(cudaFree(dout));
}

// ---------------------------------------------------------------------------
// 9. MLA absorb Q
// ---------------------------------------------------------------------------

static void test_mla_absorb_q() {
    std::printf("MLA absorb Q (wk_b @ q_nope, concat q_pe)\n");
    std::mt19937 rng(42);
    // K3 dims, fewer heads for the micro-test.
    const int qk_nope = 128, kv_lora = 512, rope = 64, H = 8;
    const int key_length = kv_lora + rope;   // 576
    auto q_nope = rnd((size_t)qk_nope * H, rng);
    auto q_pe = rnd((size_t)rope * H, rng);
    auto wk_b = rnd((size_t)qk_nope * kv_lora * H, rng, -0.1f, 0.1f);

    std::vector<double> ref((size_t)key_length * H, 0.0);
    for (int h = 0; h < H; ++h) {
        for (int r = 0; r < kv_lora; ++r) {
            double acc = 0.0;
            for (int d = 0; d < qk_nope; ++d) {
                const size_t wi = (size_t)h * qk_nope * kv_lora + (size_t)r * qk_nope + d;
                acc += (double)wk_b[wi] * (double)q_nope[(size_t)h * qk_nope + d];
            }
            ref[(size_t)h * key_length + r] = acc;
        }
        for (int d = 0; d < rope; ++d)
            ref[(size_t)h * key_length + kv_lora + d] = q_pe[(size_t)h * rope + d];
    }

    float *dn = to_dev(q_nope), *dp = to_dev(q_pe), *dw = to_dev(wk_b), *dout = nullptr;
    CU(cudaMalloc(&dout, (size_t)key_length * H * sizeof(float)));
    mla_absorb_q_f32(dout, dn, dp, dw, qk_nope, kv_lora, rope, H, 0);
    CU(cudaDeviceSynchronize());
    CU(cudaGetLastError());
    close_enough(from_dev(dout, (size_t)key_length * H), ref, "mla absorb Q vs float64 ref");
    CU(cudaFree(dn)); CU(cudaFree(dp)); CU(cudaFree(dw)); CU(cudaFree(dout));
}

// ---------------------------------------------------------------------------
// 10. MLA NoPE decode attention
// ---------------------------------------------------------------------------

// n_ctx is a parameter because the shared-memory footprint used to scale with it.
// The kernel staged one float per cached token, so the launch crossed the 48 KB
// dynamic-shared limit at n_ctx = (49152/4) - kv_lora - 9 and returned
// cudaErrorInvalidValue — silently, since the launcher returns void and the caller
// never polls. At the 48-token shape this test used to run at, that is unreachable.
// The long case below is above the old ceiling for these dims (12,215) and is the
// regression guard: it fails at the LAUNCH, not on a tolerance, if the score vector
// ever goes back into shared memory.
static void test_mla_decode_attn(int n_ctx, int H) {
    std::printf("MLA NoPE decode attn (MQA K-cache + wv_b; scale=1/sqrt(192)), "
                "n_ctx=%d heads=%d\n", n_ctx, H);
    std::mt19937 rng(77);
    const int kv_lora = 64;     // shrunk from 512 so the float64 ref stays cheap
    const int rope = 16;        // shrunk from 64; ratio preserved
    const int v_dim = 32;       // shrunk from 128
    const int key_length = kv_lora + rope;
    // Correct K3 scale uses n_embd_head_k_mla = qk_nope+rope, NOT key_length.
    const int qk_nope = 32;     // stand-in for 128
    const float scale = 1.0f / std::sqrt((float)(qk_nope + rope));
    const float wrong_scale = 1.0f / std::sqrt((float)key_length);

    auto q = rnd((size_t)key_length * H, rng);
    auto k_cache = rnd((size_t)key_length * n_ctx, rng);
    auto wv_b = rnd((size_t)kv_lora * v_dim * H, rng, -0.2f, 0.2f);

    auto run_ref = [&](double sc) {
        std::vector<double> out((size_t)v_dim * H, 0.0);
        for (int h = 0; h < H; ++h) {
            std::vector<double> scores(n_ctx);
            double mx = -1e300;
            for (int t = 0; t < n_ctx; ++t) {
                double s = 0.0;
                for (int d = 0; d < key_length; ++d)
                    s += (double)q[(size_t)h * key_length + d] *
                         (double)k_cache[(size_t)t * key_length + d];
                scores[t] = s * sc;
                mx = std::max(mx, scores[t]);
            }
            double sum = 0.0;
            for (int t = 0; t < n_ctx; ++t) {
                scores[t] = std::exp(scores[t] - mx);
                sum += scores[t];
            }
            for (int t = 0; t < n_ctx; ++t) scores[t] /= sum;

            std::vector<double> latent(kv_lora, 0.0);
            for (int r = 0; r < kv_lora; ++r)
                for (int t = 0; t < n_ctx; ++t)
                    latent[r] += scores[t] * (double)k_cache[(size_t)t * key_length + r];

            for (int v = 0; v < v_dim; ++v) {
                double acc = 0.0;
                for (int r = 0; r < kv_lora; ++r) {
                    const size_t wi = (size_t)h * kv_lora * v_dim + (size_t)v * kv_lora + r;
                    acc += (double)wv_b[wi] * latent[r];
                }
                out[(size_t)h * v_dim + v] = acc;
            }
        }
        return out;
    };

    auto ref = run_ref(scale);
    auto wrong = run_ref(wrong_scale);
    assert_variant_differs(ref, wrong, "vs scale=1/sqrt(key_length=80)");

    float *dq = to_dev(q), *dk = to_dev(k_cache), *dw = to_dev(wv_b), *dout = nullptr;
    CU(cudaMalloc(&dout, (size_t)v_dim * H * sizeof(float)));

    // The kernel now reads the ROW INDEX from device memory and lengths its loop as
    // *d_pos + 1, so the test has to supply n_ctx the same way the runtime does. Passing
    // the length where the index belongs is exactly the off-by-one this test should catch,
    // so it is spelled out rather than folded into the call.
    const int pos_host = n_ctx - 1;
    int* d_pos = nullptr;
    CU(cudaMalloc(&d_pos, sizeof(int)));
    CU(cudaMemcpy(d_pos, &pos_host, sizeof(int), cudaMemcpyHostToDevice));

    mla_decode_attn_f32(dout, dq, dk, dw, key_length, kv_lora, v_dim, H,
                        n_ctx, d_pos, scale, 0);
    CU(cudaDeviceSynchronize());
    CU(cudaGetLastError());
    close_enough(from_dev(dout, (size_t)v_dim * H), ref, "mla decode attn vs float64 ref");

    // The device length must be what the kernel actually obeys, not decoration. Re-run
    // with d_pos one SHORT and require the result to change: if the kernel were still
    // using the host n_ctx, this would silently pass and the whole device-position
    // change would be unverified.
    // Collapse to a SINGLE position rather than shortening by one. One token out of a
    // long context can move the output by less than the tolerance — measured 6.5e-04 at
    // one of these shapes — so "shorter by one" is not a reliable discriminator and a
    // kernel that ignored d_pos entirely could pass it. Attending over position 0 alone
    // cannot coincide with attending over all of them.
    if (n_ctx >= 2) {
        const int pos_short = 0;
        CU(cudaMemcpy(d_pos, &pos_short, sizeof(int), cudaMemcpyHostToDevice));
        mla_decode_attn_f32(dout, dq, dk, dw, key_length, kv_lora, v_dim, H,
                            n_ctx, d_pos, scale, 0);
        CU(cudaDeviceSynchronize());
        CU(cudaGetLastError());
        const std::vector<float> got_f = from_dev(dout, (size_t)v_dim * H);
        const std::vector<double> got(got_f.begin(), got_f.end());
        assert_variant_differs(got, ref,
                               "mla decode attn ignores d_pos (device length not obeyed)");
    }

    // ---- F16-cache arm, same shapes so it takes the same kernel path ----
    // The device cache is built by mla_kv_store_row_f16 itself (cmpr/k_pe fed
    // from slices of the f32 rows), then pinned BYTE-FOR-BYTE against host
    // round-to-nearest — so the store kernel is verified, not just used.
    __half* dk16 = nullptr;
    CU(cudaMalloc(&dk16, k_cache.size() * sizeof(__half)));
    for (int t = 0; t < n_ctx; ++t) {
        CU(cudaMemcpy(d_pos, &t, sizeof(int), cudaMemcpyHostToDevice));
        k3_mla_kv_store_f16(dk16, dk + (size_t)t * key_length,
                            dk + (size_t)t * key_length, d_pos,
                            kv_lora, key_length - kv_lora, key_length, 0);
    }
    CU(cudaMemcpy(d_pos, &pos_host, sizeof(int), cudaMemcpyHostToDevice));
    CU(cudaDeviceSynchronize());
    CU(cudaGetLastError());
    std::vector<__half> hk(k_cache.size()), hk_dev(k_cache.size());
    for (size_t i = 0; i < k_cache.size(); ++i) hk[i] = __float2half_rn(k_cache[i]);
    CU(cudaMemcpy(hk_dev.data(), dk16, k_cache.size() * sizeof(__half),
                  cudaMemcpyDeviceToHost));
    ++g_case;
    if (std::memcmp(hk.data(), hk_dev.data(), k_cache.size() * sizeof(__half)) != 0) {
        ++g_fail;
        std::printf("  FAIL  f16 store rows != host round-to-nearest\n");
    } else {
        std::printf("  ok    f16 store rows == host round-to-nearest (bitwise)\n");
    }

    // The reference for this arm is rebuilt FROM THE ROUNDED cache, so the
    // kernel is held to the same float64 bar as the f32 arm above and the
    // storage rounding is not laundered into the kernel's error budget.
    for (size_t i = 0; i < k_cache.size(); ++i) k_cache[i] = __half2float(hk[i]);
    auto ref16 = run_ref(scale);
    mla_decode_attn_kvf16(dout, dq, dk16, dw, key_length, kv_lora, v_dim, H,
                          n_ctx, d_pos, scale, 0);
    CU(cudaDeviceSynchronize());
    CU(cudaGetLastError());
    close_enough(from_dev(dout, (size_t)v_dim * H), ref16,
                 "mla decode attn (f16 cache) vs float64 ref on rounded cache");

    CU(cudaFree(dk16));
    CU(cudaFree(dq)); CU(cudaFree(dk)); CU(cudaFree(dw)); CU(cudaFree(dout));
    CU(cudaFree(d_pos));
}

// ---------------------------------------------------------------------------
// 12. noaux_tc MoE router
// ---------------------------------------------------------------------------

static void test_moe_router_noaux_tc() {
    std::printf("noaux_tc router (sigmoid + bias for SELECTION only, then renormalise)\n");
    std::mt19937 rng(20260730);
    const int E = 896, K = 16, TOK = 8;          // K3's real routing shape
    auto logits = rnd((size_t)E * TOK, rng, -6.f, 6.f);
    auto bias = rnd((size_t)E, rng, -0.35f, 0.35f);

    // --- float64 reference, straight from build_moe_ffn's semantics ---
    std::vector<double> ref_w((size_t)K * TOK);
    std::vector<int> ref_id((size_t)K * TOK);
    std::vector<double> wrong_biased_w((size_t)K * TOK);   // weights from BIASED probs
    std::vector<int> wrong_unbiased_sel((size_t)K * TOK);  // select on UNBIASED probs

    for (int t0 = 0; t0 < TOK; ++t0) {
        std::vector<double> p(E), sel(E);
        for (int e = 0; e < E; ++e) {
            p[e] = 1.0 / (1.0 + std::exp(-(double)logits[(size_t)t0 * E + e]));
            sel[e] = p[e] + (double)bias[e];
        }
        auto topk = [&](const std::vector<double>& score) {
            std::vector<double> s = score;
            std::vector<int> out;
            for (int k = 0; k < K; ++k) {
                int bi = -1; double bv = -1e300;
                for (int e = 0; e < E; ++e)
                    if (s[e] > bv) { bv = s[e]; bi = e; }   // lower index wins ties
                out.push_back(bi);
                s[bi] = -1e300;
            }
            return out;
        };
        // reference: select on biased, weight from UNBIASED, then renormalise
        std::vector<int> sel_ids = topk(sel);
        double sum = 0.0;
        for (int k = 0; k < K; ++k) { ref_id[(size_t)t0*K+k] = sel_ids[k];
                                      ref_w[(size_t)t0*K+k] = p[sel_ids[k]]; sum += p[sel_ids[k]]; }
        sum = std::max(sum, 6.103515625e-5);
        for (int k = 0; k < K; ++k) ref_w[(size_t)t0*K+k] /= sum;

        // wrong variant A: weights taken from the BIASED probs (the natural bug)
        double s2 = 0.0;
        for (int k = 0; k < K; ++k) { wrong_biased_w[(size_t)t0*K+k] = sel[sel_ids[k]]; s2 += sel[sel_ids[k]]; }
        s2 = std::max(s2, 6.103515625e-5);
        for (int k = 0; k < K; ++k) wrong_biased_w[(size_t)t0*K+k] /= s2;

        // wrong variant B: select on unbiased probs (bias ignored entirely)
        std::vector<int> u = topk(p);
        for (int k = 0; k < K; ++k) wrong_unbiased_sel[(size_t)t0*K+k] = u[k];
    }
    assert_variant_differs(ref_w, wrong_biased_w, "vs weights from BIASED probs");
    {   // selection must actually differ when the bias is applied, or the test is vacuous
        ++g_case;
        int diff = 0;
        for (size_t i = 0; i < ref_id.size(); ++i) if (ref_id[i] != wrong_unbiased_sel[i]) ++diff;
        const bool ok = diff > 0;
        std::printf("  %-34s %d/%zu ids differ  %s\n", "bias changes the SELECTION",
                    diff, ref_id.size(), ok ? "OK" : "FAIL (bias had no effect)");
        if (!ok) ++g_fail;
    }

    float *dl = to_dev(logits), *db = to_dev(bias), *dw = nullptr; int* di = nullptr;
    CU(cudaMalloc(&dw, (size_t)K * TOK * sizeof(float)));
    CU(cudaMalloc(&di, (size_t)K * TOK * sizeof(int)));
    moe_router_noaux_tc_f32(dw, di, dl, db, E, K, TOK, /*norm_w=*/true, /*w_scale=*/1.0f, 0);
    CU(cudaDeviceSynchronize());
    CU(cudaGetLastError());

    close_enough(from_dev(dw, (size_t)K * TOK), ref_w, "router weights vs float64 ref");

    std::vector<int> got_id((size_t)K * TOK);
    CU(cudaMemcpy(got_id.data(), di, got_id.size() * sizeof(int), cudaMemcpyDeviceToHost));
    ++g_case;
    int bad = 0;
    for (size_t i = 0; i < got_id.size(); ++i) if (got_id[i] != ref_id[i]) ++bad;
    std::printf("  %-34s %d/%zu wrong  %s\n", "selected expert ids", bad, got_id.size(),
                bad ? "FAIL" : "OK");
    if (bad) { ++g_fail;
        for (size_t i = 0; i < got_id.size() && i < 6; ++i)
            std::printf("        [%zu] got %d want %d\n", i, got_id[i], ref_id[i]);
    }

    // weights must sum to 1 per token after renormalisation
    auto gw = from_dev(dw, (size_t)K * TOK);
    ++g_case;
    double worst = 0;
    for (int t0 = 0; t0 < TOK; ++t0) {
        double s = 0; for (int k = 0; k < K; ++k) s += gw[(size_t)t0*K+k];
        worst = std::fmax(worst, std::fabs(s - 1.0));
    }
    std::printf("  %-34s max |sum-1| = %.3e  %s\n", "renormalised to 1 per token", worst,
                worst < 1e-5 ? "OK" : "FAIL");
    if (worst >= 1e-5) ++g_fail;

    CU(cudaFree(dl)); CU(cudaFree(db)); CU(cudaFree(dw)); CU(cudaFree(di));
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
    test_kda_conv_step();      std::printf("\n");
    test_mla_gate_out();       std::printf("\n");
    test_kda_decay_gate();     std::printf("\n");
    test_l2_norm_heads();      std::printf("\n");
    test_mla_absorb_q();       std::printf("\n");
    test_mla_decode_attn(48, 4);       std::printf("\n");
    // Above the 48 KB shared-memory ceiling the pre-online-softmax kernel had at
    // these dims (12,215 tokens). It launched with cudaErrorInvalidValue and left
    // the output buffer untouched — which, on the real forward, means the previous
    // layer's attention output survives into this one.
    test_mla_decode_attn(20000, 2);    std::printf("\n");
    // Head count divisible by kMlaHeadsPerBlock and a context past kMlaSplitMinCtx:
    // the only shape that reaches mla_decode_attn_hbatch_kernel, where one block owns
    // 12 heads and the per-tile softmax is reduced over a warp instead of the block.
    // The two cases above take the per-head kernels (H=2 does not divide), so without
    // this one the batched path ships unexercised on a device.
    test_mla_decode_attn(20000, 12);   std::printf("\n");
    // Ragged final slice: 12,289 is one token past a multiple of the tile, so the last
    // slice ends mid-tile and the online rescale has to fold a short tail.
    test_mla_decode_attn(12289, 12);   std::printf("\n");
    // 8 does not divide 12, so k3_mla_heads_per_block falls back to the 8-head
    // instantiation — the branch a non-K3 head count would take.
    test_mla_decode_attn(20000, 8);    std::printf("\n");
    // A WIDE COMBINE, which nothing above reaches.
    //
    // The split count is min(fill target, n_ctx / k3_mla_min_slice_len) capped by
    // the budget, so at n_ctx 20,000 it lands on 39 whatever the cap is — every case
    // above therefore merges a few dozen slices and would pass identically with the
    // cap at 64 or at 512. Head-sharding MLA to 12 heads raises the budget to 512
    // slices, and 128k of context is what actually asks for them: 256 partials
    // folded by one combine, each carrying its own max and sum-of-exp.
    //
    // Both of those counts were WRONG from the day they were written until the slice
    // floor was derived: the flat kMlaMinSliceLen = 1024 halved them to 19 and 128.
    // The comment described the coverage that was intended; the constant silently
    // delivered half of it. They agree now, and k3_mla_min_slice_len's static_assert
    // is what keeps them agreeing.
    //
    // This is the case that fails if the online-softmax rescale does not hold at
    // that width, and it exists because the end-to-end accuracy gate CANNOT see
    // it: the eval scores on a short reference prompt while timing at 128k, so it
    // runs splits=1 and reports a byte-identical KLD however wrong this path is.
    test_mla_decode_attn(131072, 12);  std::printf("\n");
    test_moe_router_noaux_tc(); std::printf("\n");

    std::printf("%d cases, %d failure(s)\n", g_case, g_fail);
    return g_fail == 0 ? 0 : 1;
}
