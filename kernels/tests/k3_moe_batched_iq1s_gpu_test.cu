// Batched IQ1_S MoE expert FFN vs the shipped per-token scalar op.
//
// The reference here is not float64 — it is `moe_expert_ffn_iq1s_f32` itself, called T
// times over the same tokens with the same routing. That is the right reference because
// the question this file has to answer is not "is the formula right" (the scalar op
// already owns that, and kimi_k3_numeric_test grades it against float64) but "does
// bucketing T tokens by expert and running each expert once compute the same thing as
// running each token separately".
//
// Everything that can go wrong here is a PERMUTATION bug, and a float64 reference would
// be a weaker detector of those than the scalar op is:
//
//   - the counting sort places a pair under the wrong expert
//   - `rows` gathers the wrong token into the GEMM
//   - `slot` loses which of the token's top_k a row came from, so the combine applies
//     another expert's routing weight
//   - the band test disagrees with the scalar op's, so a rank sums a different subset
//   - a token that won NO local expert is left holding stale memory instead of zero
//
// Every one of those produces a well-formed tensor of the right shape and magnitude.
//
// AGREEMENT IS NOT EXACT AND MUST NOT BE ASSERTED AS SUCH. The tolerance is a
// relative-L2 band, and the CONTROLS below are what stop that band from being vacuous.
//
// THE BAND IS 2.5e-2, AND THAT NUMBER IS MEASURED, NOT PICKED. Two effects sit under it
// and both were isolated on the node rather than assumed:
//
//   1. weps. The scalar op DROPS every expert whose routing weight is below
//      SPARKINFER_K3_MOE_WEPS, which defaults to 0.08 and is live here because
//      k3_weps_depth_live() is true when no position pointer is bound. With random
//      weights in [0.01, 1] that silently deletes ~7% of the selections from the
//      REFERENCE only. Measured: 2.8e-2 with it on, 1.7e-2 with it off. This test sets
//      it to 0 in main() so the comparison is of the arithmetic, not of a heuristic the
//      batched path deliberately does not implement.
//
//   2. int8 activation quantization, applied twice — once to x, once to h before the
//      down projection. The batched path scales per ROW; the scalar path uses
//      block_q8_K's per-256 scale. Residual after weps is off: 1.7e-2.
//
// THE EVIDENCE THAT (2) IS NUMERICS AND NOT A PERMUTATION BUG IS THE T=1 CASE. At T=1
// there is no batching and no cross-token gather at all, and it measures 1.700e-2 —
// the same as T=256's 1.667e-2. A gather or bucket defect cannot be invariant to T.
// The token-rotation control below is the other half: it takes a CORRECT result and
// permutes it, and must blow the band by 10x.
//
// 1.7e-2 PER MoE LAYER IS NOT OBVIOUSLY INSIDE THE PARITY GATE over 92 layers, and this
// test does not claim it is — it grades this op against the shipped op, not the model
// against llama.cpp. The likely fix is to carry the activation scale per 32 values
// rather than per row: the GEMM already drains its int32 accumulator every 32 k for the
// WEIGHT scale, so a matching activation scale costs one extra multiply per drain and
// nothing else. That is the next change here and it is not made in this commit.

#include "sparkinfer/kernels/kimi_k3.h"
#include "k3_fast_test_util.h"

#include <cstdlib>
#include <cstring>
#include <vector>

using namespace sparkinfer::kernels::k3;
using namespace k3test;
#define CU K3T_CU

namespace {

constexpr int kQK = 256, kBlkBytes = 50;

// Random IQ1_S bytes. Every bit pattern is a valid block for reconstruction (11-bit
// grid index, 3-bit scale, 1 sign bit), so random bits sweep the lattice; only `d` is
// drawn deliberately, to keep fp16 corner cases out of the comparison.
std::vector<uint8_t> random_iq1s(size_t n_blocks, std::mt19937& rng) {
    std::vector<uint8_t> v(n_blocks * kBlkBytes);
    std::uniform_int_distribution<int> byte(0, 255);
    for (size_t b = 0; b < n_blocks; ++b) {
        uint8_t* p = v.data() + b * kBlkBytes;
        const uint16_t d = 0x2800;                       // ~0.031 in fp16
        std::memcpy(p, &d, 2);
        for (int i = 2; i < kBlkBytes; ++i) p[i] = (uint8_t)byte(rng);
    }
    return v;
}

void run_case(const char* what, int T, int latent, int ffn, int top_k,
              int n_expert, int expert_begin, int n_local, double tol) {
    std::mt19937 rng(0xB0A7Du + T * 31 + n_local);
    const float beta = 4.0f, lb = 25.0f;

    const size_t gu_blocks = (size_t)n_local * ffn    * (latent / kQK);
    const size_t d_blocks  = (size_t)n_local * latent * (ffn    / kQK);
    auto hgate = random_iq1s(gu_blocks, rng);
    auto hup   = random_iq1s(gu_blocks, rng);
    auto hdown = random_iq1s(d_blocks,  rng);

    auto hx = rnd((size_t)T * latent, rng, -0.6f, 0.6f);

    // Routing: distinct experts per token (a real top-k has no duplicates), drawn from
    // the GLOBAL range so out-of-band selections are exercised on every rank but one.
    std::vector<int>   hids((size_t)T * top_k);
    std::vector<float> hw((size_t)T * top_k);
    std::uniform_real_distribution<float> wd(0.01f, 1.0f);
    for (int t = 0; t < T; ++t) {
        std::vector<int> pool(n_expert);
        for (int i = 0; i < n_expert; ++i) pool[i] = i;
        for (int k = 0; k < top_k; ++k) {
            std::uniform_int_distribution<int> pick(k, n_expert - 1);
            std::swap(pool[k], pool[pick(rng)]);
            hids[(size_t)t * top_k + k] = pool[k];
            hw[(size_t)t * top_k + k]   = wd(rng);
        }
    }

    void *dg = nullptr, *du = nullptr, *dd = nullptr;
    CU(cudaMalloc(&dg, hgate.size())); CU(cudaMemcpy(dg, hgate.data(), hgate.size(), cudaMemcpyHostToDevice));
    CU(cudaMalloc(&du, hup.size()));   CU(cudaMemcpy(du, hup.data(),   hup.size(),   cudaMemcpyHostToDevice));
    CU(cudaMalloc(&dd, hdown.size())); CU(cudaMemcpy(dd, hdown.data(), hdown.size(), cudaMemcpyHostToDevice));

    float* dx = to_dev(hx);
    int*   dids = nullptr; float* dw = nullptr;
    CU(cudaMalloc(&dids, hids.size() * sizeof(int)));
    CU(cudaMemcpy(dids, hids.data(), hids.size() * sizeof(int), cudaMemcpyHostToDevice));
    CU(cudaMalloc(&dw, hw.size() * sizeof(float)));
    CU(cudaMemcpy(dw, hw.data(), hw.size() * sizeof(float), cudaMemcpyHostToDevice));

    float *d_batched = nullptr, *d_ref = nullptr, *d_scratch = nullptr;
    CU(cudaMalloc(&d_batched, (size_t)T * latent * sizeof(float)));
    CU(cudaMalloc(&d_ref,     (size_t)T * latent * sizeof(float)));
    CU(cudaMalloc(&d_scratch, (size_t)top_k * ffn * sizeof(float)));
    // Poison both: a row the path never writes must fail, not read as a lucky zero.
    CU(cudaMemset(d_batched, 0x7f, (size_t)T * latent * sizeof(float)));
    CU(cudaMemset(d_ref,     0x7f, (size_t)T * latent * sizeof(float)));

    // Reference: the shipped op, once per token.
    for (int t = 0; t < T; ++t)
        moe_expert_ffn_iq1s_f32(d_ref + (size_t)t * latent, d_scratch,
                                dx + (size_t)t * latent,
                                dids + (size_t)t * top_k, dw + (size_t)t * top_k,
                                dg, du, dd, latent, ffn, top_k, beta, lb,
                                nullptr, expert_begin, n_local);
    CU(cudaDeviceSynchronize());

    const bool ok = k3_moe_expert_ffn_batched_iq1s(
        d_batched, dx, dids, dw, dg, du, dd, T, latent, ffn, top_k,
        beta, lb, expert_begin, n_local, nullptr);
    note("batched launched", ok);
    if (ok) {
        CU(cudaDeviceSynchronize());
        const auto got  = from_dev(d_batched, (size_t)T * latent);
        const auto want = from_dev(d_ref,     (size_t)T * latent);
        std::vector<double> wantd(want.begin(), want.end());
        check(what, got, wantd, tol);

        // CONTROL 1 — the comparison must be able to FAIL. Rotating the batched result
        // by one token is the exact shape a `rows` gather bug produces, and it must
        // blow the same tolerance the real comparison just passed.
        if (T > 1) {
            std::vector<float> rot((size_t)T * latent);
            for (int t = 0; t < T; ++t)
                std::memcpy(&rot[(size_t)t * latent],
                            &got[(size_t)((t + 1) % T) * latent], latent * sizeof(float));
            double num = 0, den = 0;
            for (size_t i = 0; i < rot.size(); ++i) {
                const double d = rot[i] - wantd[i];
                num += d * d; den += wantd[i] * wantd[i];
            }
            note("  control: token-rotated result fails the same tolerance",
                 std::sqrt(num / (den + 1e-30)) > tol * 10);
        }

        // CONTROL 2 — the output is not trivially zero (which would pass a relative-L2
        // against a zero reference if the band logic silently dropped everything).
        double mag = 0; for (float v : got) mag += (double)v * v;
        note("  control: output is non-trivial", mag > 1e-6);
    }

    CU(cudaFree(dg)); CU(cudaFree(du)); CU(cudaFree(dd)); CU(cudaFree(dx));
    CU(cudaFree(dids)); CU(cudaFree(dw));
    CU(cudaFree(d_batched)); CU(cudaFree(d_ref)); CU(cudaFree(d_scratch));
}

}  // namespace

int main() {
    // Disable the reference's routing-weight skip BEFORE any CUDA call: k3_moe_weps_host
    // caches it in a function-local static on first use. Left at its 0.08 default the
    // scalar op deletes ~7% of the selections and this test would be grading a heuristic
    // instead of the arithmetic. See the header note.
    setenv("SPARKINFER_K3_MOE_WEPS", "0", 1);
    std::printf("=== K3 batched IQ1_S MoE expert FFN vs the per-token scalar op ===\n");
    std::printf("    (SPARKINFER_K3_MOE_WEPS=0 — grading arithmetic, not the weight-skip)\n\n");
    if (!have_device()) return 0;

    // Shrunk dims: latent/ffn must be whole 256-blocks, and the reference costs T
    // launches, so K3's real 3584/3072 at a useful T would be a very long test. The
    // PERMUTATION logic under test does not depend on the dims.
    std::printf("all experts local (tp=1 shape):\n");
    run_case("T=64  lat=512 ffn=512 k=4  E=32",  64, 512, 512, 4, 32, 0, 32, 2.5e-2);
    run_case("T=128 lat=512 ffn=768 k=8  E=64", 128, 512, 768, 8, 64, 0, 64, 2.5e-2);

    std::printf("\nragged / degenerate M:\n");
    run_case("T=1   lat=512 ffn=512 k=4  E=32",   1, 512, 512, 4, 32, 0, 32, 2.5e-2);
    run_case("T=3   lat=512 ffn=512 k=2  E=64",   3, 512, 512, 2, 64, 0, 64, 2.5e-2);
    // More tokens than experts: every expert gets several rows, which is the shape a
    // real prefill chunk produces.
    run_case("T=256 lat=512 ffn=512 k=4  E=16", 256, 512, 512, 4, 16, 0, 16, 2.5e-2);

    std::printf("\nsharded band (out-of-band ids must contribute exactly zero):\n");
    run_case("E=64 band [0,16)",  128, 512, 512, 8, 64,  0, 16, 2.5e-2);
    run_case("E=64 band [16,32)", 128, 512, 512, 8, 64, 16, 16, 2.5e-2);
    run_case("E=64 band [48,64)", 128, 512, 512, 8, 64, 48, 16, 2.5e-2);

    std::printf("\nrefusals:\n");
    {
        float* p = nullptr; int* i = nullptr;
        CU(cudaMalloc(&p, 1024)); CU(cudaMalloc(&i, 64));
        note("refuses latent not a multiple of 256",
             !k3_moe_expert_ffn_batched_iq1s(p, p, i, p, p, p, p, 4, 512 + 32, 512, 2,
                                             4.f, 25.f, 0, 8, nullptr));
        note("refuses n_local_experts <= 0",
             !k3_moe_expert_ffn_batched_iq1s(p, p, i, p, p, p, p, 4, 512, 512, 2,
                                             4.f, 25.f, 0, 0, nullptr));
        CU(cudaFree(p)); CU(cudaFree(i));
    }

    return report();
}
