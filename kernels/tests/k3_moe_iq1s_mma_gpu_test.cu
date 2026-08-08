// IQ1_S expert GEMM on the int8 tensor cores, at K3's real expert dims.
//
// The ARITHMETIC is proved on the CPU (k3_moe_iq1s_mma_cpu_test.cpp): IQ1_S is exactly
// an int8 format, the 32-value sub-block coincides with mma.m16n8k32's k-extent, and
// the drain-per-sub-block reduction beats the shipped f32 scalar path. None of that
// needs a device and it is not re-litigated here.
//
// What this binary covers is everything the CPU test structurally cannot:
//
//   [1] the mma fragment layout — that ldmatrix.x4 plus the m16n8k32 accumulator
//       element->(row,col) mapping actually place values where the epilogue assumes.
//       An off-by-one here is invisible to any scalar port and produces a transposed
//       or shuffled tile that still has the right norm.
//   [2] the shared-memory swizzle — decode stores and ldmatrix loads go through one
//       swz(), so a wrong swizzle is self-consistent and silently wrong.
//   [3] the M tail — 896 experts at top_k 16 give a ragged, small M (~36 at a 2048
//       token chunk), so a partial BM=32 tile is the COMMON case, not the corner.
//   [4] the row gather — `rows` indexes A and `sa` by source token while C stays
//       compact. Two different indices into one loop is exactly where a scatter path
//       goes wrong, and the identity case cannot see it.
//
// Graded against float64 over the same reconstructed weights and the same int8
// activations, so the only difference under test is the schedule.

#include "sparkinfer/kernels/kimi_k3.h"
#include "sparkinfer/kernels/iq1s_tables.h"
#include "k3_fast_test_util.h"

#include <cstring>

using namespace sparkinfer::kernels::k3;
using namespace k3test;
#define CU K3T_CU

namespace {

constexpr int kQK = 256, kSub = 32;

struct BlockIQ1S { uint16_t d; uint8_t qs[32]; uint16_t qh[8]; };
static_assert(sizeof(BlockIQ1S) == 50, "IQ1_S block must be 50 bytes");

float h2f_host(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const uint32_t exp  = (h >> 10) & 0x1f;
    const uint32_t man  = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else { int e = -1; uint32_t m = man;
               do { m <<= 1; ++e; } while (!(m & 0x400));
               bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3ff) << 13); }
    } else if (exp == 31) { bits = sign | 0x7f800000u | (man << 13); }
    else { bits = sign | ((exp - 15 + 127) << 23) | (man << 13); }
    float f; std::memcpy(&f, &bits, 4); return f;
}

uint16_t f2h_host(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (exp <= 0)  return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
    const uint32_t rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) ++h;
    return h;
}

// ggml's own reconstruction order — the reference the kernel must reproduce.
void ref_dequant_block(const BlockIQ1S& b, float* out) {
    const float d = h2f_host(b.d);
    for (int ib32 = 0; ib32 < kQK / kSub; ++ib32) {
        const uint16_t h = b.qh[ib32];
        const float dl    = d * (float)(2 * ((h >> 12) & 7) + 1);
        const float delta = (h & 0x8000) ? -SPARKINFER_IQ1S_DELTA : SPARKINFER_IQ1S_DELTA;
        for (int l = 0; l < 4; ++l) {
            const uint32_t idx =
                (uint32_t)b.qs[4 * ib32 + l] | (((uint32_t)(h >> (3 * l)) & 7u) << 8);
            const int8_t* grid = (const int8_t*)&iq1s_grid_host[idx];
            for (int j = 0; j < 8; ++j)
                out[ib32 * kSub + l * 8 + j] = dl * ((float)grid[j] + delta);
        }
    }
}

BlockIQ1S random_block(std::mt19937& rng) {
    BlockIQ1S b{};
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> dd(0.001f, 0.5f);
    b.d = f2h_host(dd(rng));
    for (int i = 0; i < 32; ++i) b.qs[i] = (uint8_t)byte(rng);
    for (int i = 0; i < 8; ++i)  b.qh[i] = (uint16_t)(byte(rng) | (byte(rng) << 8));
    return b;
}

// One case: M rows (optionally gathered out of `pool` tokens) x N cols x K.
void run_case(const char* what, int M, int N, int K, bool gather) {
    std::mt19937 rng(0xC0FFEEu + M * 131 + N * 17 + K + (gather ? 7 : 0));
    const int blk_per_row = K / kQK;
    const int pool = gather ? (M * 3 + 5) : M;      // a buffer this expert is a subset of

    std::vector<BlockIQ1S> W((size_t)N * blk_per_row);
    for (auto& b : W) b = random_block(rng);

    // int8 activations over the whole pool, with PER-32 scales: sa is [pool][K/32], the
    // granularity the GEMM drains at. Deliberately varied across blocks within a row —
    // a constant-per-row sa would pass even if the kernel ignored the block index.
    const int nb = K / kSub;
    std::vector<int8_t> A((size_t)pool * K);
    std::vector<float>  sa((size_t)pool * nb);
    std::uniform_int_distribution<int> q(-127, 127);
    std::uniform_real_distribution<float> sd(0.002f, 0.02f);
    for (int r = 0; r < pool; ++r) {
        for (int b = 0; b < nb; ++b) sa[(size_t)r * nb + b] = sd(rng);
        for (int k = 0; k < K; ++k) A[(size_t)r * K + k] = (int8_t)q(rng);
    }
    // A deliberately non-monotonic, non-identity row list: a gather bug that merely
    // shifts rows would survive a sorted one.
    std::vector<int> rows(M);
    for (int i = 0; i < M; ++i) rows[i] = (i * 7 + 3) % pool;

    // float64 reference over the reconstructed weights and the same int8 activations,
    // applying each 32-block's own activation scale — the schedule the kernel now runs.
    std::vector<double> ref((size_t)M * N);
    std::vector<float>  wrow(K);
    for (int n = 0; n < N; ++n) {
        for (int ib = 0; ib < blk_per_row; ++ib)
            ref_dequant_block(W[(size_t)n * blk_per_row + ib], &wrow[ib * kQK]);
        for (int m = 0; m < M; ++m) {
            const int src = gather ? rows[m] : m;
            double acc = 0.0;
            for (int b = 0; b < nb; ++b) {
                double blk = 0.0;
                for (int j = 0; j < kSub; ++j) {
                    const int k = b * kSub + j;
                    blk += (double)wrow[k] * (double)A[(size_t)src * K + k];
                }
                acc += blk * (double)sa[(size_t)src * nb + b];
            }
            ref[(size_t)m * N + n] = acc;
        }
    }

    signed char* dA = nullptr; float* dsa = nullptr; void* dW = nullptr;
    int* drows = nullptr; float* dC = nullptr;
    CU(cudaMalloc(&dA, A.size()));
    CU(cudaMalloc(&dsa, sa.size() * sizeof(float)));
    CU(cudaMalloc(&dW, W.size() * sizeof(BlockIQ1S)));
    CU(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));
    CU(cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dsa, sa.data(), sa.size() * sizeof(float), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dW, W.data(), W.size() * sizeof(BlockIQ1S), cudaMemcpyHostToDevice));
    if (gather) {
        CU(cudaMalloc(&drows, rows.size() * sizeof(int)));
        CU(cudaMemcpy(drows, rows.data(), rows.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    // Poison C so a tile the kernel never writes fails loudly instead of reading as 0.
    CU(cudaMemset(dC, 0x7f, (size_t)M * N * sizeof(float)));

    const bool launched =
        k3_moe_iq1s_mma_gemm(dC, dA, dsa, dW, drows, M, N, K, nullptr);
    note(launched ? "launched" : "k3_moe_iq1s_mma_gemm launched", launched);
    if (launched) {
        CU(cudaDeviceSynchronize());
        const auto got = from_dev(dC, (size_t)M * N);
        // Tight: the int32 dot is exact and only the K/32 f32 accumulations round, so
        // this should sit well below the 2e-5 the fast-intrinsic kernels need.
        check(what, got, ref, 1e-6);
    }

    CU(cudaFree(dA)); CU(cudaFree(dsa)); CU(cudaFree(dW)); CU(cudaFree(dC));
    if (drows) CU(cudaFree(drows));
}

}  // namespace

int main() {
    std::printf("=== K3 MoE IQ1_S int8 tensor-core GEMM (GPU) ===\n\n");
    if (!have_device()) return 0;

    // K3's expert dims. expert_latent 3584 and moe_ffn 3072 are the two K values the
    // routed experts actually reduce over; both are whole IQ1_S blocks.
    std::printf("K3 expert shapes (expert_latent 3584, moe_ffn 3072):\n");
    run_case("M=36  N=3072 K=3584  (T=2048 chunk)",   36, 3072, 3584, false);
    run_case("M=73  N=3584 K=3072  (T=4096 chunk)",   73, 3584, 3072, false);

    std::printf("\nragged M — the common case, not the corner:\n");
    run_case("M=1   N=256  K=3584  (degenerate)",      1,  256, 3584, false);
    run_case("M=31  N=256  K=3584  (BM tail, 32-1)",  31,  256, 3584, false);
    run_case("M=32  N=256  K=3584  (BM exact)",       32,  256, 3584, false);
    run_case("M=33  N=256  K=3584  (BM+1)",           33,  256, 3584, false);

    std::printf("\nragged N — BN=128 tail:\n");
    run_case("M=36  N=127  K=3584  (BN tail)",        36,  127, 3584, false);
    run_case("M=36  N=129  K=3584  (BN+1)",           36,  129, 3584, false);

    std::printf("\nrow gather (rows indexes A and sa; C stays compact):\n");
    run_case("M=36  N=256  K=3584  gathered",         36,  256, 3584, true);
    run_case("M=17  N=384  K=3072  gathered, ragged", 17,  384, 3072, true);

    // The guard the header promises: a K that is not whole IQ1_S blocks must be
    // REFUSED, not silently mis-tiled. The caller falls back on false.
    {
        float* c = nullptr; signed char* a = nullptr; float* s = nullptr; void* w = nullptr;
        CU(cudaMalloc(&c, 16 * sizeof(float))); CU(cudaMalloc(&a, 16));
        CU(cudaMalloc(&s, 4 * sizeof(float)));  CU(cudaMalloc(&w, 64));
        note("refuses K not a multiple of 256",
             !k3_moe_iq1s_mma_gemm(c, a, s, w, nullptr, 4, 4, 3584 + 32, nullptr));
        note("refuses M <= 0", !k3_moe_iq1s_mma_gemm(c, a, s, w, nullptr, 0, 4, 3584, nullptr));
        CU(cudaFree(c)); CU(cudaFree(a)); CU(cudaFree(s)); CU(cudaFree(w));
    }

    return report();
}
