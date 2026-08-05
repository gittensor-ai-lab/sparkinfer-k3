// k3_kda_chunk_prefill at K3's real KDA dims, graded against float64.
//
// The ALGEBRA is already proven on the host (k3_kda_chunk_prefill_cpu_test.cpp:
// the chunk form equals the token-by-token recurrence to 1e-15, over multi-chunk,
// ragged and T=1 shapes, with the axis-of-decay swap as a negative control). What
// that test structurally cannot reach is everything that only exists on a device:
//
//   - the two-kernel split. K1 writes a workspace that K2 reads back; a tile-index
//     mismatch between the two (they compute the offset independently, one as
//     h*gridDim.x+chunk, the other as h*n_chunks+c) silently feeds K2 another
//     head's chunk and still produces a well-formed, wrong sequence.
//   - the thread-to-channel and thread-to-column mappings, including the phase
//     alternation between channel-parallel (thread = d) and token-parallel
//     (thread = t) work across __syncthreads().
//   - the in-place rewrite of s_qd/s_kd in K1 phase 1c (staged raw, then
//     overwritten decayed by the same thread).
//   - the ragged tail's zero-padding actually reaching the kernel, rather than
//     reading uninitialised shared memory past actual_len.
//
// The reference here is float64 over the SAME raw inputs, applying the same three
// folded activations (op 7 gate, op 8 L2 norm + q scale, beta sigmoid) and then
// the token-by-token recurrence — i.e. it re-derives the answer from the raw
// tensors independently of anything the kernel does, rather than comparing the
// kernel to a transcription of itself.

#include "sparkinfer/kernels/kimi_k3.h"
#include "k3_fast_test_util.h"

#include <cmath>
#include <vector>

using namespace sparkinfer::kernels::k3;
using namespace k3test;
#define CU K3T_CU

namespace {

constexpr int kD = 128;   // K3's kda_head_dim — the only width the kernel accepts

// float64 ground truth: activations, then the pinned per-token recurrence.
void reference(const std::vector<float>& S0, const std::vector<float>& q_raw,
               const std::vector<float>& k_raw, const std::vector<float>& v,
               const std::vector<float>& g_raw, const std::vector<float>& beta_raw,
               const std::vector<float>& A, int T, int H,
               double lower_bound, double l2_eps,
               std::vector<double>& out, std::vector<double>& state) {
    out.assign((size_t)T * H * kD, 0.0);
    state.assign(S0.size(), 0.0);
    for (size_t i = 0; i < S0.size(); ++i) state[i] = (double)S0[i];

    const double q_scale = 1.0 / std::sqrt((double)kD);

    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < H; ++h) {
            const size_t base = ((size_t)t * H + h) * kD;

            // op 8: L2 norm over head_dim, q additionally scaled by 1/sqrt(D).
            double qs = 0.0, ks = 0.0;
            for (int d = 0; d < kD; ++d) {
                qs += (double)q_raw[base + d] * (double)q_raw[base + d];
                ks += (double)k_raw[base + d] * (double)k_raw[base + d];
            }
            const double qn = q_scale / std::sqrt(qs + l2_eps);
            const double kn = 1.0 / std::sqrt(ks + l2_eps);

            std::vector<double> qa(kD), ka(kD), ge(kD);
            for (int d = 0; d < kD; ++d) {
                qa[d] = (double)q_raw[base + d] * qn;
                ka[d] = (double)k_raw[base + d] * kn;
                // op 7: g = lower_bound * sigmoid(-(A[h] * g_raw))
                const double gv = lower_bound /
                    (1.0 + std::exp((double)A[h] * (double)g_raw[base + d]));
                ge[d] = std::exp(gv);
            }
            const double bh = 1.0 / (1.0 + std::exp(-(double)beta_raw[(size_t)t * H + h]));

            double* Sh = &state[(size_t)h * kD * kD];
            for (int j = 0; j < kD; ++j) {
                double* S = &Sh[(size_t)j * kD];
                double sk = 0.0;
                for (int i = 0; i < kD; ++i) sk += S[i] * ge[i] * ka[i];
                const double d_j = bh * ((double)v[base + j] - sk);
                double o = 0.0;
                for (int i = 0; i < kD; ++i) {
                    const double s2 = S[i] * ge[i] + ka[i] * d_j;
                    S[i] = s2;
                    o += s2 * qa[i];
                }
                out[base + j] = o;
            }
        }
    }
}

void run_case(const char* what, int T, int H, double tol) {
    const float lower_bound = -5.0f, l2_eps = 1e-6f;
    std::mt19937 rng(0x5EED + T * 131 + H);

    auto S0       = rnd((size_t)kD * kD * H, rng, -0.4f, 0.4f);
    auto q_raw    = rnd((size_t)T * H * kD, rng, -2.0f, 2.0f);
    auto k_raw    = rnd((size_t)T * H * kD, rng, -2.0f, 2.0f);
    auto v        = rnd((size_t)T * H * kD, rng, -1.0f, 1.0f);
    auto g_raw    = rnd((size_t)T * H * kD, rng, -3.0f, 3.0f);
    auto beta_raw = rnd((size_t)T * H, rng, -3.0f, 3.0f);
    auto A        = rnd((size_t)H, rng, -4.0f, -0.05f);   // ssm_a = -exp(A_log), negative

    std::vector<double> ref_out, ref_state;
    reference(S0, q_raw, k_raw, v, g_raw, beta_raw, A, T, H,
              lower_bound, l2_eps, ref_out, ref_state);

    float* d_q     = to_dev(q_raw);
    float* d_k     = to_dev(k_raw);
    float* d_v     = to_dev(v);
    float* d_g     = to_dev(g_raw);
    float* d_beta  = to_dev(beta_raw);
    float* d_A     = to_dev(A);
    float* d_state = to_dev(S0);
    float* d_out   = nullptr;
    CU(cudaMalloc(&d_out, (size_t)T * H * kD * sizeof(float)));
    // Poison: a row the kernel never writes must fail loudly, not read as a lucky 0.
    CU(cudaMemset(d_out, 0x7f, (size_t)T * H * kD * sizeof(float)));

    const bool ok = k3_kda_chunk_prefill(d_out, d_state, d_q, d_k, d_v, d_g, d_beta,
                                         d_A, T, kD, H, lower_bound, l2_eps, nullptr);
    note("launched", ok);
    if (ok) {
        CU(cudaDeviceSynchronize());
        check(what, from_dev(d_out, (size_t)T * H * kD), ref_out, tol);
        char sbuf[128];
        std::snprintf(sbuf, sizeof(sbuf), "%s  [final state]", what);
        check(sbuf, from_dev(d_state, S0.size()), ref_state, tol);
    }

    CU(cudaFree(d_q)); CU(cudaFree(d_k)); CU(cudaFree(d_v)); CU(cudaFree(d_g));
    CU(cudaFree(d_beta)); CU(cudaFree(d_A)); CU(cudaFree(d_state)); CU(cudaFree(d_out));
}

}  // namespace

int main() {
    std::printf("=== K3 KDA chunk-parallel prefill (GPU) — head_dim %d ===\n\n", kD);
    if (!have_device()) return 0;

    // Exact multiples of CHUNK=16: the cross-chunk state carry, which a single
    // chunk cannot exercise at all.
    std::printf("exact chunk multiples:\n");
    run_case("T=16  H=2  (one chunk)",       16, 2, 2e-5);
    run_case("T=64  H=2  (four chunks)",     64, 2, 2e-5);

    // Ragged tails — the common case for a real prompt, not a corner.
    std::printf("\nragged final chunk:\n");
    run_case("T=1   H=1  (1 of 16)",          1, 1, 2e-5);
    run_case("T=17  H=2  (16 + 1)",          17, 2, 2e-5);
    run_case("T=37  H=3  (32 + 5)",          37, 3, 2e-5);

    // K3's real head count per rank under the sharded policy (n_head 96 / tp 8),
    // and a longer prompt so K2 runs a meaningful number of sequential steps.
    std::printf("\nK3 shapes:\n");
    run_case("T=256 H=12 (sharded n_head)", 256, 12, 2e-5);

    // The width guard: anything but 128 must be REFUSED, not silently mis-tiled,
    // because every shared-memory extent in the kernel is sized against it.
    {
        float* p = nullptr;
        CU(cudaMalloc(&p, 1024));
        note("refuses head_dim != 128",
             !k3_kda_chunk_prefill(p, p, p, p, p, p, p, p, 16, 64, 1, -5.0f, 1e-6f, nullptr));
        note("refuses T <= 0",
             !k3_kda_chunk_prefill(p, p, p, p, p, p, p, p, 0, kD, 1, -5.0f, 1e-6f, nullptr));
        CU(cudaFree(p));
    }

    return report();
}
