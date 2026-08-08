// The batched prefill projection against the single-token path it must replace, on device.
//
// The CPU test (k3_prefill_order_cpu_test) proves the two loop nests are the same float
// PROGRAM. This proves the two implementations agree on hardware, which is a different
// claim: it exercises dp4a, __float2int_rn, the real block_for tier, the launch geometry
// and the epilogue, none of which the host simulation can reach.
//
// The comparison is bit-for-bit, not tolerance-based, and it has to be. Prefill and decode
// SHARE the KV cache: a prompt ingested through the batched path leaves a cache the decode
// step then continues from, so a one-ULP disagreement would not fail cleanly here -- it
// would compound through the generation.
//
// Shapes are K3's real ones, and the token counts straddle every tile boundary the
// launcher can pick (1, 2, 4, 8) plus ragged tails, because the tail is where a batched
// kernel is most likely to be wrong and least likely to be noticed.

#include "sparkinfer/kernels/kimi_k3.h"
#include "sparkinfer/kernels/kimi_k3_prefill.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

using namespace sparkinfer::kernels::k3;

#define CU(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) {                     \
    std::printf("CUDA FAIL %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));  \
    return 2; } } while (0)

struct BlockQ8_0 { uint16_t d; int8_t qs[32]; };

int main() {
    std::mt19937 rng(20260805);
    std::uniform_int_distribution<int> qd(-127, 127);

    // N, K, name. N is kept modest where the real tensor is enormous -- the kernel's
    // behaviour depends on K (which picks BLOCK) and on the token tile, not on how many
    // output rows follow, and a 163840-row LM head would only slow the test down.
    const struct { int N, K; const char* name; } cases[] = {
        { 2048, 7168, "ffn_routed_down / MLA kv_a  (nb 224, BLOCK 128)" },
        { 2048, 3584, "ffn_routed_up               (nb 112, BLOCK 128)" },
        { 2048, 1536, "KDA attn_output             (nb  48, BLOCK  64)" },
        { 1024,  768, "shexp gate/up               (nb  24, BLOCK  32)" },
    };
    const int tok_counts[] = { 1, 2, 3, 4, 5, 7, 8, 9, 16, 33 };

    long total_cells = 0, total_bad = 0;

    for (const auto& c : cases) {
        const int N = c.N, K = c.K, nb = K / 32;

        // Random Q8_0 weights, and activations in a range that exercises the quantiser
        // rather than saturating it.
        std::vector<BlockQ8_0> hW((size_t)N * nb);
        for (auto& b : hW) {
            const uint16_t e = (uint16_t)(8 + (rng() % 15));
            b.d = (uint16_t)(((rng() & 1) << 15) | (e << 10) | (rng() % 1024));
            for (int j = 0; j < 32; ++j) b.qs[j] = (int8_t)qd(rng);
        }
        void* dW = nullptr;
        CU(cudaMalloc(&dW, hW.size() * sizeof(BlockQ8_0)));
        CU(cudaMemcpy(dW, hW.data(), hW.size() * sizeof(BlockQ8_0), cudaMemcpyHostToDevice));

        for (int T : tok_counts) {
            std::vector<float> hX((size_t)T * K);
            for (auto& v : hX) v = (float)((int)(rng() % 2001) - 1000) / 500.0f;

            float* dX = nullptr;
            CU(cudaMalloc(&dX, hX.size() * sizeof(float)));
            CU(cudaMemcpy(dX, hX.data(), hX.size() * sizeof(float), cudaMemcpyHostToDevice));

            // ---- reference: the shipping single-token path, one token at a time ----
            void*  dq1 = nullptr;
            float* dy1 = nullptr;
            CU(cudaMalloc(&dq1, k3_q8_0_bytes(K)));
            CU(cudaMalloc(&dy1, (size_t)T * N * sizeof(float)));
            for (int t = 0; t < T; ++t) {
                if (!k3_quantize_act_f32(dq1, dX + (size_t)t * K, K, nullptr)) {
                    std::printf("FAIL: k3_quantize_act_f32 declined K=%d\n", K); return 1;
                }
                if (!k3_proj_q8act_f32(dy1 + (size_t)t * N, dq1, dW, 8, N, K, nullptr)) {
                    std::printf("FAIL: k3_proj_q8act_f32 declined N=%d K=%d\n", N, K); return 1;
                }
            }
            CU(cudaDeviceSynchronize());

            // ---- batched: one quantise, one projection, the whole tile ----
            void*  dqB = nullptr;
            float* dyB = nullptr;
            CU(cudaMalloc(&dqB, k3_prefill_act_q8_bytes(K, T)));
            CU(cudaMalloc(&dyB, (size_t)T * N * sizeof(float)));
            if (!k3_prefill_quantize_act(dqB, dX, K, T, nullptr)) {
                std::printf("FAIL: k3_prefill_quantize_act declined K=%d T=%d\n", K, T);
                return 1;
            }
            if (!k3_prefill_proj_q8act(dyB, dqB, dW, 8, N, K, T, nullptr)) {
                std::printf("FAIL: k3_prefill_proj_q8act declined N=%d K=%d T=%d\n", N, K, T);
                return 1;
            }
            CU(cudaDeviceSynchronize());
            CU(cudaGetLastError());

            std::vector<float> h1((size_t)T * N), hB((size_t)T * N);
            CU(cudaMemcpy(h1.data(), dy1, h1.size() * sizeof(float), cudaMemcpyDeviceToHost));
            CU(cudaMemcpy(hB.data(), dyB, hB.size() * sizeof(float), cudaMemcpyDeviceToHost));

            long bad = 0;
            for (size_t i = 0; i < h1.size(); ++i)
                if (std::memcmp(&h1[i], &hB[i], sizeof(float)) != 0) {
                    if (bad < 3)
                        std::printf("  MISMATCH N=%d K=%d T=%d tok=%d row=%d  %.9g vs %.9g\n",
                                    N, K, T, (int)(i / N), (int)(i % N), h1[i], hB[i]);
                    ++bad;
                }
            total_cells += (long)h1.size();
            total_bad   += bad;
            std::printf("  N=%-5d K=%-5d tile=%-3d cells=%-8zu %s (tile width %d)\n",
                        N, K, T, h1.size(), bad ? "MISMATCH" : "exact",
                        k3_prefill_proj_token_tile(N, K, T));

            CU(cudaFree(dq1)); CU(cudaFree(dy1));
            CU(cudaFree(dqB)); CU(cudaFree(dyB)); CU(cudaFree(dX));
        }
        std::printf("%s\n", c.name);
        CU(cudaFree(dW));
    }

    std::printf("\ncells compared: %ld   mismatches: %ld\n", total_cells, total_bad);
    std::printf("%s\n", total_bad == 0
        ? "PASS: batched prefill projection is bit-identical to the single-token path"
        : "FAIL: batched prefill projection diverges from the single-token path");
    return total_bad == 0 ? 0 : 1;
}
