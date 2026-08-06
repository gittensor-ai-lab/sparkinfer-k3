// k3_proj_q8_mma_gemm graded against the projection it is meant to replace.
//
// THE REFERENCE IS k3_proj_ggml_f32, WHICH IS THE PATH THE RUNTIME ACTUALLY TAKES.
//
// qact_proj() defaults TRUE (kimi_k3.cpp: `!env_zero("SPARKINFER_K3_GGML_QACT_PROJ")`),
// so every dense projection already quantises its activation to Q8_0 before
// multiplying. This kernel's per-32 int8 activation is therefore the SAME
// approximation the engine ships, not a new one, and grading against the
// f32-activation k3_proj_f32 would charge this GEMM for a difference that exists
// in main too.
//
// That distinction was measured, not assumed. Against k3_proj_f32 every shape here
// lands at relL2 3.7e-3 — identical for M=1 and M=256, both tiles, every K — which
// is exactly what per-32 int8 activation quantisation predicts and nothing to do
// with the GEMM:
//
//     x ~ U(-1,1): amax(32) ~ 0.97, d = amax/127 = 0.0076
//     RMS(quant err) = d/sqrt(12) = 0.0022, RMS(x) = 0.577  ->  0.0038
//
// K-independent, which is the giveaway: a structural fault (wrong index, wrong
// stride, transposed codes) fails by orders of magnitude and varies with shape.
// Both arms are kept below — the ggml one tight, the f32 one loose and labelled —
// so a regression in the quantiser cannot hide inside a tolerance chosen for the
// GEMM, and neither can a regression in the GEMM hide inside one chosen for the
// quantiser.
//
// What only a device can catch here, and what the shapes below are chosen for:
//   - the uint16 staging. b.qs sits at +2 in a 34-byte block, so it is 2-byte but
//     never 16-byte aligned; a uint4 read faults on most blocks. The lane->uint16
//     mapping is new code and a wrong h2 stride silently transposes 32 codes.
//   - the BM 32/64 switch at kMSwitch=48. Two different tile shapes, two
//     different warp counts, one contract — M=32 and M=71 straddle it.
//   - ragged M and N. The mma computes rows past M and columns past N regardless;
//     only the store is guarded, so a stale or NaN operand would leak sideways
//     through the accumulator rather than staying in its own row.
//   - K spanning many BK tiles, which is where a wrong blk_per_row shows up.

#include "sparkinfer/kernels/kimi_k3.h"
#include "k3_fast_test_util.h"

#include <cuda_fp16.h>

#include <cmath>
#include <cstring>
#include <vector>

using namespace sparkinfer::kernels::k3;
using namespace k3test;
#define CU K3T_CU

namespace {

// Q8_0: per 32 values, one f16 scale then 32 int8 codes. Built here rather than
// borrowed so the test does not depend on the loader agreeing with the kernel.
struct BlockQ80 { uint16_t d; int8_t qs[32]; };

uint16_t f32_to_f16(float f) {
    // Round-to-nearest-even via the hardware path the kernel reads back with.
    __half h = __float2half(f);
    uint16_t out;
    std::memcpy(&out, &h, sizeof(out));
    return out;
}
float f16_to_f32(uint16_t u) {
    __half h;
    std::memcpy(&h, &u, sizeof(h));
    return __half2float(h);
}

// Quantise a [N,K] f32 matrix to Q8_0 and also return the DEQUANTISED values, so
// the reference multiplies exactly the weights the kernel sees — otherwise the
// comparison would charge the GEMM for the weight quantisation too.
std::vector<uint8_t> make_q80(const std::vector<float>& w, int N, int K,
                              std::vector<double>& deq) {
    const int nb = K / 32;
    std::vector<uint8_t> out((size_t)N * nb * sizeof(BlockQ80));
    deq.assign((size_t)N * K, 0.0);
    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < nb; ++b) {
            const float* src = &w[(size_t)n * K + b * 32];
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(src[i]));
            const float d = amax / 127.0f;
            const uint16_t dh = f32_to_f16(d);
            const float dd = f16_to_f32(dh);      // what the KERNEL will use
            BlockQ80 blk{};
            blk.d = dh;
            for (int i = 0; i < 32; ++i) {
                const int q = (amax == 0.0f) ? 0 : (int)std::lrint(src[i] / d);
                blk.qs[i] = (int8_t)(q < -127 ? -127 : (q > 127 ? 127 : q));
                deq[(size_t)n * K + b * 32 + i] = (double)dd * (double)blk.qs[i];
            }
            std::memcpy(&out[((size_t)n * nb + b) * sizeof(BlockQ80)], &blk,
                        sizeof(BlockQ80));
        }
    }
    return out;
}

void run_case(const char* what, int M, int N, int K, double tol) {
    std::mt19937 rng(0xC0FFEE + M * 7919 + N * 131 + K);
    auto x = rnd((size_t)M * K, rng, -1.0f, 1.0f);
    auto w = rnd((size_t)N * K, rng, -0.5f, 0.5f);

    std::vector<double> wdeq;
    auto q80 = make_q80(w, N, K, wdeq);

    // --- the arm under test -------------------------------------------------
    float* dx = to_dev(x);
    void*  dW = nullptr;
    CU(cudaMalloc(&dW, q80.size()));
    CU(cudaMemcpy(dW, q80.data(), q80.size(), cudaMemcpyHostToDevice));

    signed char* dA = nullptr;
    float* dsa = nullptr;
    float* dC  = nullptr;
    CU(cudaMalloc(&dA, (size_t)M * K));
    CU(cudaMalloc(&dsa, (size_t)M * (K / 32) * sizeof(float)));
    CU(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));
    // Poison: a row the kernel never writes must fail loudly, not read as a lucky 0.
    CU(cudaMemset(dC, 0x7f, (size_t)M * N * sizeof(float)));

    note("quantize_rows", k3_moe_iq1s_mma_quantize_rows(dA, dsa, dx, M, K, nullptr));
    const bool ok = k3_proj_q8_mma_gemm(dC, dA, dsa, dW, M, N, K, nullptr);
    note("gemm launched", ok);
    if (!ok) return;
    CU(cudaDeviceSynchronize());
    auto got = from_dev(dC, (size_t)M * N);

    // --- the arm it must match: M separate calls to the RUNTIME's projection --
    // wtype 8 == GGML_TYPE_Q8_0, the id the runtime dispatches on.
    std::vector<float> ref((size_t)M * N, 0.0f);
    std::vector<float> ref_f32((size_t)M * N, 0.0f);
    {
        float* dy = nullptr;
        void*  dq8 = nullptr;
        CU(cudaMalloc(&dy, (size_t)N * sizeof(float)));
        CU(cudaMalloc(&dq8, k3_q8_0_bytes(K)));
        for (int m = 0; m < M; ++m) {
            const bool okp = k3_proj_ggml_f32(dy, dx + (size_t)m * K, dW, 8, N, K,
                                              dq8, nullptr);
            if (!okp) { note("k3_proj_ggml_f32 reference", false); return; }
            CU(cudaDeviceSynchronize());
            CU(cudaMemcpy(&ref[(size_t)m * N], dy, (size_t)N * sizeof(float),
                          cudaMemcpyDeviceToHost));
            if (!k3_proj_f32(dy, dx + (size_t)m * K, dW, 8, N, K, nullptr)) {
                note("k3_proj_f32 reference", false); return;
            }
            CU(cudaDeviceSynchronize());
            CU(cudaMemcpy(&ref_f32[(size_t)m * N], dy, (size_t)N * sizeof(float),
                          cudaMemcpyDeviceToHost));
        }
        CU(cudaFree(dy)); CU(cudaFree(dq8));
    }
    ++g_case;
    const double rl2 = rel_l2f(got, ref);
    const bool pass = rl2 <= tol;
    std::printf("  %-46s relL2=%.3e tol=%.1e %s\n", what, rl2, tol,
                pass ? "OK" : "FAIL");
    if (!pass) ++g_fail;
    // The f32-activation arm, LOOSE AND LABELLED: this is the cost of quantising
    // the activation at all, which main pays too. It must stay near 3.8e-3 — a jump
    // means the quantiser moved, a collapse toward 0 means qact stopped being on.
    std::printf("  %-46s relL2=%.3e   (vs f32-act; expect ~3.8e-3)\n",
                "  ... same rows vs k3_proj_f32", rel_l2f(got, ref_f32));

    // --- float64 over the DEQUANTISED weights, row 0 only -------------------
    // Bounds what the int8 activation costs. If this drifts while the arm above
    // still passes, the quantiser moved and the tolerance above absorbed it.
    {
        std::vector<double> exact((size_t)N, 0.0);
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) acc += (double)x[k] * wdeq[(size_t)n * K + k];
            exact[n] = acc;
        }
        std::vector<float> row0(got.begin(), got.begin() + N);
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%s  [row 0 vs f64]", what);
        check(buf, row0, exact, 6e-3);   // int8 activation, not the GEMM
    }

    CU(cudaFree(dx)); CU(cudaFree(dW)); CU(cudaFree(dA));
    CU(cudaFree(dsa)); CU(cudaFree(dC));
}

}  // namespace

int main() {
    std::printf("=== K3 batched Q8_0 projection on int8 tensor cores ===\n\n");
    if (!have_device()) return 0;

    // Every shape here must clear the crossover guard (M >= 32 and M*N >= 131072),
    // because below it the kernel DECLINES by design — those cases are exercised
    // separately at the bottom.
    std::printf("small M (BM=32 tile):\n");
    run_case("M=32   N=4096 K=1024",  32, 4096, 1024, 3e-4);
    run_case("M=32   N=7168 K=7168",  32, 7168, 7168, 3e-4);
    run_case("M=48   N=2816 K=512",   48, 2816,  512, 3e-4);

    // Straddle kMSwitch=48 — a different tile and a different warp count.
    std::printf("\nlarge M (BM=64 tile):\n");
    run_case("M=71   N=2048 K=1024",  71, 2048, 1024, 3e-4);
    run_case("M=128  N=1536 K=7168", 128, 1536, 7168, 3e-4);
    run_case("M=256  N=3584 K=7168", 256, 3584, 7168, 3e-4);

    // Ragged M and N: rows and columns the mma computes and the store discards.
    // N deliberately odd/prime-ish so the last kBN tile is partial.
    std::printf("\nragged tiles:\n");
    run_case("M=77   N=2049 K=2048",  77, 2049, 2048, 3e-4);
    run_case("M=130  N=4097 K=576",  130, 4097,  576, 3e-4);
    run_case("M=71   N=3585 K=1024",  71, 3585, 1024, 3e-4);
    run_case("M=33   N=4001 K=64",    33, 4001,   64, 3e-4);

    // K3's real projection shapes, per rank under the sharded policy.
    std::printf("\nK3 shapes:\n");
    run_case("attn_output  M=64 N=7168 K=1536", 64, 7168, 1536, 3e-4);
    run_case("routed_down  M=64 N=3584 K=7168", 64, 3584, 7168, 3e-4);
    run_case("routed_up    M=256 N=7168 K=3584", 256, 7168, 3584, 3e-4);

    {
        float* p = nullptr;
        CU(cudaMalloc(&p, 4096));
        // Shape guard: K not a multiple of the BK tile must be REFUSED, not silently
        // mis-tiled into a wrong-valued result.
        note("refuses K % 64 != 0",
             !k3_proj_q8_mma_gemm(p, (const signed char*)p, p, p, 64, 4096, 96, nullptr));
        note("refuses M <= 0",
             !k3_proj_q8_mma_gemm(p, (const signed char*)p, p, p, 0, 4096, 64, nullptr));
        // CROSSOVER GUARD. Below it this kernel is 8-50x SLOWER than the GEMV it
        // replaces (see the measured table in k3_proj_q8_mma.cu), so it must DECLINE
        // and leave the caller on its existing path. A silent regression of that size
        // is worse than not having the kernel.
        note("declines M=1 (GEMV territory)",
             !k3_proj_q8_mma_gemm(p, (const signed char*)p, p, p, 1, 7168, 1536, nullptr));
        note("declines M=8",
             !k3_proj_q8_mma_gemm(p, (const signed char*)p, p, p, 8, 7168, 1536, nullptr));
        note("declines narrow N at M=32 (M*N under the bar)",
             !k3_proj_q8_mma_gemm(p, (const signed char*)p, p, p, 32, 1536, 7168, nullptr));
        note("accepts M=32 at wide N",
             k3_proj_q8_mma_gemm(p, (const signed char*)p, p, p, 32, 4096, 64, nullptr));
        CU(cudaFree(p));
        CU(cudaDeviceSynchronize());
    }

    return report();
}
