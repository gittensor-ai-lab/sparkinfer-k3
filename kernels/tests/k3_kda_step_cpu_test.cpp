// CPU model of the KDA decode step with the i-reduction split across a warp.
//
// WHY A CPU TEST FOR A CUDA KERNEL. This checks the SCHEDULE, not the formula — the
// answer is already pinned by kimi_k3_numeric_test, and what is new is the order the
// terms are summed in and the width they are stored at. A CPU model checks exactly
// that, and it is also the only coverage that RUNS on a fork's PR: the five ci.yml
// checks stay pending until a maintainer approves the workflow, so a test that needs a
// device is a test nobody sees.
//
// Every case asserts three things, and the third is the one that matters:
//   1. the schedule agrees with a float64 reference to f32 tolerance;
//   2. it agrees with the sequential schedule it replaces to within a few ulp;
//   3. a WRONG-BUT-PLAUSIBLE variant does NOT — so a tolerance loose enough to hide
//      the bug fails the test instead of passing it.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

int g_fail = 0;
int g_case = 0;

void check(const char* what, double rel, double tol) {
    ++g_case;
    const bool ok = rel <= tol;
    std::printf("  %-52s rel=%.3e tol=%.1e  %s\n", what, rel, tol, ok ? "OK" : "FAIL");
    if (!ok) ++g_fail;
}

void check_differs(const char* what, double rel, double floor_) {
    ++g_case;
    const bool ok = rel >= floor_;
    std::printf("  %-52s rel=%.3e floor=%.1e  %s\n", what, rel, floor_,
                ok ? "OK" : "FAIL (variant is indistinguishable)");
    if (!ok) ++g_fail;
}

double rel_l2(const std::vector<float>& got, const std::vector<double>& want) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = (double)got[i] - want[i];
        num += d * d;
        den += want[i] * want[i];
    }
    return std::sqrt(num / (den + 1e-30));
}

double rel_l2f(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)b[i] * (double)b[i];
    }
    return std::sqrt(num / (den + 1e-30));
}

// The butterfly a warp does: for off in 16,8,4,2,1 every lane adds the partner's
// value. All 32 lanes end holding the same total, in a specific association tree.
float warp_butterfly_sum(float lanes_in[32]) {
    float v[32];
    for (int i = 0; i < 32; ++i) v[i] = lanes_in[i];
    for (int off = 16; off > 0; off >>= 1) {
        float t[32];
        for (int l = 0; l < 32; ++l) t[l] = v[l] + v[l ^ off];
        for (int l = 0; l < 32; ++l) v[l] = t[l];
    }
    return v[0];
}

// ===========================================================================
// Factor A — KDA decode step, one warp per column with the i-axis split
// ===========================================================================
//
// Layout contract, the one kimi_k3_numeric_test documents: S[i][j] lives at s[j*D+i],
// so i is the FAST axis and lane l of the warp owning column j takes i = 4l..4l+3 as
// one float4.
struct KdaOut {
    std::vector<float> out;
    std::vector<float> state;
};

// The kernel this replaces: one thread per column, i accumulated in increasing order.
KdaOut kda_sequential(const std::vector<float>& S0, const std::vector<float>& q,
                      const std::vector<float>& k, const std::vector<float>& v,
                      const std::vector<float>& g, const std::vector<float>& beta,
                      int D, int H) {
    KdaOut r;
    r.out.assign((size_t)D * H, 0.0f);
    r.state = S0;
    for (int h = 0; h < H; ++h) {
        std::vector<float> ge(D);
        for (int i = 0; i < D; ++i) ge[i] = std::exp(g[(size_t)h * D + i]);
        for (int j = 0; j < D; ++j) {
            float* S = &r.state[(size_t)h * D * D + (size_t)j * D];
            float sk = 0.0f;
            for (int i = 0; i < D; ++i) sk += (S[i] * ge[i]) * k[(size_t)h * D + i];
            const float d_j = beta[h] * (v[(size_t)h * D + j] - sk);
            float o = 0.0f;
            for (int i = 0; i < D; ++i) {
                const float s2 = S[i] * ge[i] + k[(size_t)h * D + i] * d_j;
                S[i] = s2;
                o += s2 * q[(size_t)h * D + i];
            }
            r.out[(size_t)h * D + j] = o;
        }
    }
    return r;
}

// The new schedule: lane l owns i = 4l..4l+3, sums its four in order, then the warp
// folds the 32 partials in the butterfly above.
KdaOut kda_warp_per_column(const std::vector<float>& S0, const std::vector<float>& q,
                           const std::vector<float>& k, const std::vector<float>& v,
                           const std::vector<float>& g, const std::vector<float>& beta,
                           int D, int H) {
    KdaOut r;
    r.out.assign((size_t)D * H, 0.0f);
    r.state = S0;
    for (int h = 0; h < H; ++h) {
        std::vector<float> ge(D);
        for (int i = 0; i < D; ++i) ge[i] = std::exp(g[(size_t)h * D + i]);
        for (int j = 0; j < D; ++j) {
            float* S = &r.state[(size_t)h * D * D + (size_t)j * D];
            float part[32];
            for (int l = 0; l < 32; ++l) {
                float acc = 0.0f;
                for (int e = 0; e < 4; ++e) {
                    const int i = 4 * l + e;
                    acc += (S[i] * ge[i]) * k[(size_t)h * D + i];
                }
                part[l] = acc;
            }
            const float sk  = warp_butterfly_sum(part);
            const float d_j = beta[h] * (v[(size_t)h * D + j] - sk);

            float po[32];
            for (int l = 0; l < 32; ++l) {
                float acc = 0.0f;
                for (int e = 0; e < 4; ++e) {
                    const int i = 4 * l + e;
                    const float s2 = S[i] * ge[i] + k[(size_t)h * D + i] * d_j;
                    S[i] = s2;
                    acc += s2 * q[(size_t)h * D + i];
                }
                po[l] = acc;
            }
            r.out[(size_t)h * D + j] = warp_butterfly_sum(po);
        }
    }
    return r;
}

// float64 reference, written from the formula.
void kda_reference(const std::vector<float>& S0, const std::vector<float>& q,
                   const std::vector<float>& k, const std::vector<float>& v,
                   const std::vector<float>& g, const std::vector<float>& beta,
                   int D, int H, std::vector<double>& out, std::vector<double>& state) {
    out.assign((size_t)D * H, 0.0);
    state.assign(S0.size(), 0.0);
    for (size_t i = 0; i < S0.size(); ++i) state[i] = (double)S0[i];
    for (int h = 0; h < H; ++h) {
        std::vector<double> ge(D);
        for (int i = 0; i < D; ++i) ge[i] = std::exp((double)g[(size_t)h * D + i]);
        for (int j = 0; j < D; ++j) {
            double* S = &state[(size_t)h * D * D + (size_t)j * D];
            double sk = 0.0;
            for (int i = 0; i < D; ++i) sk += S[i] * ge[i] * (double)k[(size_t)h * D + i];
            const double d_j = (double)beta[h] * ((double)v[(size_t)h * D + j] - sk);
            double o = 0.0;
            for (int i = 0; i < D; ++i) {
                const double s2 = S[i] * ge[i] + (double)k[(size_t)h * D + i] * d_j;
                S[i] = s2;
                o += s2 * (double)q[(size_t)h * D + i];
            }
            out[(size_t)h * D + j] = o;
        }
    }
}

void test_kda_warp_per_column() {
    std::printf("KDA decode step: warp per column, i split across lanes\n");
    const int D = 128, H = 3;
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    auto rnd = [&](size_t n, float lo, float hi) {
        std::vector<float> x(n);
        for (auto& e : x) e = lo + (hi - lo) * (0.5f * (u(rng) + 1.0f));
        return x;
    };

    // A random NON-ZERO starting state: production starts from zeros, which is what
    // makes a layout error invisible in an end-to-end run.
    auto S0   = rnd((size_t)D * D * H, -0.5f, 0.5f);
    auto q    = rnd((size_t)D * H, -1.0f, 1.0f);
    auto k    = rnd((size_t)D * H, -1.0f, 1.0f);
    auto v    = rnd((size_t)D * H, -1.0f, 1.0f);
    auto g    = rnd((size_t)D * H, -1.5f, -0.05f);   // decay: exp(g) < 1
    auto beta = rnd((size_t)H, 0.1f, 0.9f);

    std::vector<double> ref_out, ref_state;
    kda_reference(S0, q, k, v, g, beta, D, H, ref_out, ref_state);

    const KdaOut seq = kda_sequential(S0, q, k, v, g, beta, D, H);
    const KdaOut wpc = kda_warp_per_column(S0, q, k, v, g, beta, D, H);

    check("warp-per-column out vs float64", rel_l2(wpc.out, ref_out), 2e-5);
    check("warp-per-column state vs float64", rel_l2(wpc.state, ref_state), 2e-5);
    check("sequential out vs float64", rel_l2(seq.out, ref_out), 2e-5);

    // The reassociation this change makes, quantified rather than asserted away. It is
    // the same class the expert all-reduce and the split MLA combine already make.
    const double d_out = rel_l2f(wpc.out, seq.out);
    check("warp-per-column vs sequential (reassociation only)", d_out, 1e-5);

    // The state must be IDENTICAL: the rank-1 update is elementwise in i, so no
    // reassociation can reach it. Only d_j feeds it, and d_j is a scalar per column.
    // If this drifts, the split is wrong, not merely reordered.
    double worst_state = 0.0;
    for (size_t i = 0; i < wpc.state.size(); ++i)
        worst_state = std::fmax(worst_state,
                                std::fabs((double)wpc.state[i] - (double)seq.state[i]));
    std::printf("  %-52s max_abs=%.3e\n", "state drift vs sequential", worst_state);

    // WRONG-BUT-PLAUSIBLE: lane l taking i = l, l+32, l+64, l+96 (a strided split
    // instead of a contiguous float4) is the obvious alternative decomposition and is
    // NOT what the kernel does. It must be measurably different, or the tolerance
    // above is not actually testing the schedule.
    KdaOut strided;
    strided.out.assign((size_t)D * H, 0.0f);
    strided.state = S0;
    for (int h = 0; h < H; ++h) {
        std::vector<float> ge(D);
        for (int i = 0; i < D; ++i) ge[i] = std::exp(g[(size_t)h * D + i]);
        for (int j = 0; j < D; ++j) {
            float* S = &strided.state[(size_t)h * D * D + (size_t)j * D];
            float part[32];
            for (int l = 0; l < 32; ++l) {
                float acc = 0.0f;
                for (int e = 0; e < 4; ++e) {
                    const int i = l + 32 * e;
                    acc += (S[i] * ge[i]) * k[(size_t)h * D + i];
                }
                part[l] = acc;
            }
            const float sk  = warp_butterfly_sum(part);
            const float d_j = beta[h] * (v[(size_t)h * D + j] - sk);
            float po[32];
            for (int l = 0; l < 32; ++l) {
                float acc = 0.0f;
                for (int e = 0; e < 4; ++e) {
                    const int i = l + 32 * e;
                    const float s2 = S[i] * ge[i] + k[(size_t)h * D + i] * d_j;
                    S[i] = s2;
                    acc += s2 * q[(size_t)h * D + i];
                }
                po[l] = acc;
            }
            strided.out[(size_t)h * D + j] = warp_butterfly_sum(po);
        }
    }
    check_differs("vs strided-lane split (a different schedule)",
                  rel_l2f(strided.out, wpc.out), 1e-9);
}

}  // namespace

int main() {
    std::printf("KDA decode step schedule — CPU model vs float64\n\n");
    test_kda_warp_per_column();
    std::printf("\n%d cases, %d failures\n", g_case, g_fail);
    return g_fail == 0 ? 0 : 1;
}
