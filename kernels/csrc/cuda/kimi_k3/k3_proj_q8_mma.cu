// ===========================================================================
// BATCHED Q8_0 PROJECTION ON THE INT8 TENSOR CORES.
// ===========================================================================
// Every dense projection in K3 — attn_q/k/v, ssm_g, attn_output, ffn_routed_down,
// ffn_routed_up, the router, the shared expert, and all of MLA's — goes through
// k3_proj_f32, which takes ONE activation vector and is therefore a GEMV. That is
// the right shape for decode and the wrong one for prefill: at M=1 a projection
// reads a multi-megabyte weight matrix to produce a single 7168-float row, so its
// arithmetic intensity is ~2 FLOP per weight byte and it is bandwidth-bound by
// construction. Fusing epilogues does not help — the activation round trip is
// 28 KB against megabytes of weights — and neither does a faster GEMV.
//
// The only thing that helps is more rows per weight read, which is what this is.
//
// WHY THIS IS THE 70% AND THE MERGED EXPERT GEMM IS THE 30%.
//   routed experts   531 GiB of UD-IQ1_S's 553, but SPARSE: top_k 16 of 896, so a
//                    token touches 16/896 * 531 = 9.5 GiB of it.
//   everything else   22 GiB, and DENSE: every token reads all of it.
// So the dense side is ~70% of per-token weight traffic, and unlike the expert
// side it amortises at ANY batch size — every token in the chunk multiplies the
// identical matrix, so M rows cost one weight read instead of M. The expert GEMM
// (k3_moe_iq1s_mma.cu) only amortises when two tokens pick the same expert, which
// at 8-way sharding needs B in the hundreds before a tile fills.
//
// Q8_0 IS ALREADY INT8, WHICH IS WHY THERE IS NO CONVERSION HERE. A block is
// 34 bytes: one f16 scale then 32 int8 codes, w = d * q. That is exactly
// mma.m16n8k32.s8's operand with a scale applied at the k=32 drain boundary — the
// same drain the sibling IQ1_S kernel already uses, where it applies dl/8. No
// dequantisation pass, no fp8 round trip, no accuracy loss: the int8 the tensor
// core multiplies IS the stored weight, not an approximation of it.
//
// (FP8 was considered and is strictly worse on this model. The experts are IQ1_S
// at ~1.6 bits/weight, so an FP8 copy of the same weights is ~2.6 TB against the
// node's ~1120 GiB — it does not load. And on Hopper FP8 and INT8 run at the same
// tensor-core rate, so it would buy nothing even if it fit.)
//
// STAGING IS uint16, NOT uint4, AND THAT IS FORCED BY THE FORMAT. Block i starts
// at byte 34*i and its codes start at 34*i+2, so the codes are 2-byte aligned and
// never 16-byte aligned — a uint4 load off b.qs faults on most blocks rather than
// being merely slow. 34*i+2 IS always even, so uint16 is safe, and assigning 16
// consecutive lanes to the 16 uint16s of one sub-block keeps the read coalesced.
// Repacking Q8_0 to a 16-byte-aligned layout at load time would allow uint4 and is
// the obvious next step; it is a loader change, not a kernel one, so it is not
// bundled here.
//
// The mma loop, the XOR swizzle, the ldmatrix fragment mapping and the per-32
// activation scales are all the sibling kernel's, unchanged. The two files differ
// only in how a weight sub-block is turned into 32 int8 and one scale.

#include "sparkinfer/kernels/kimi_k3.h"
#include "k3_pdl.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace sparkinfer {
namespace kernels {
namespace k3 {

namespace {

constexpr int kSub = 32;    // values per Q8_0 block == mma.m16n8k32's k-extent
constexpr int kBN  = 128;
constexpr int kBK  = 64;    // two sub-blocks
// Crossover between the two tiles: below it the wasted mma rows of the larger tile
// cost more than the extra staging the small one pays, above it the reverse. A
// starting value, not a measured one — the sibling's 70 was measured against a
// 128-row large tile, and this file's is 64 (see below), so that number does not
// transfer.
constexpr int kMSwitch = 48;

// THE LARGE TILE IS 64 ROWS, NOT 128, AND ptxas DECIDED THAT.
//
// At BM=128 the accumulator is kMFrag*kNFrag*4 = 8*2*4 = 64 floats before a single
// fragment or address register, and __launch_bounds__'s second argument asks for 2
// resident blocks of 256 threads — 128 registers each. Measured: 56 bytes of stack
// frame, 156 bytes of spill stores, 104 of spill loads, all inside the mma loop.
// The two ways out are fewer resident blocks (1 block/SM at 256 registers, which
// gives the tensor pipe nothing to interleave with) or a smaller accumulator.
// BM=64 halves kMFrag to 4, so accf is 32 floats and af is 16, and it fits 128
// registers with room — zero spill at 2 blocks/SM.
//
// The sibling IQ1_S kernel keeps 128 because its staging is a decode, not a copy:
// it has fewer live address registers at the point the accumulator peaks. Same
// bound, different pressure, so the tiles legitimately differ.
template <int BM> struct QCfg;
template <> struct QCfg<32> { static constexpr int warps = 4; };
template <> struct QCfg<64> { static constexpr int warps = 8; };

// 34 bytes: f16 scale, then 32 int8 codes. Packed, because the GGUF layout is.
#pragma pack(push, 1)
struct BlockQ80 { uint16_t d; int8_t qs[32]; };
#pragma pack(pop)
static_assert(sizeof(BlockQ80) == 34, "Q8_0 block must be 34 bytes");

// XOR swizzle at 16 B granularity — the sibling kernel's, byte for byte, so the
// two layouts cannot drift. The low 4 bits pass through, which is what makes a
// 2-byte store at an even offset well defined: c and c+1 are always in the same
// 16-byte chunk when c is even.
__device__ __forceinline__ int qswz(int k, int row) {
    return (((k >> 4) ^ (row & 3)) << 4) | (k & 15);
}

__device__ __forceinline__ void qldm_x4(unsigned& r0, unsigned& r1, unsigned& r2,
                                        unsigned& r3, const signed char* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

// ---------------------------------------------------------------------------
// C[M,N] = A[M,K] @ W[N,K]^T, W in Q8_0.
// ---------------------------------------------------------------------------
// A is int8 with a per-32 scale in `sa` ([M][K/32]) — the same activation format
// k3_moe_iq1s_mma_quantize_rows produces, and this kernel reuses that quantiser
// rather than shipping a second one that could drift from it.
template <int BM>
__global__ __launch_bounds__(QCfg<BM>::warps * 32,
                             65536 / (128 * QCfg<BM>::warps * 32))
void proj_q8_mma_kernel(float* __restrict__ C,
                        const signed char* __restrict__ A,
                        const float* __restrict__ sa,
                        const BlockQ80* __restrict__ W,
                        int M, int N, int K) {
    k3_pdl_sync();
    constexpr int kBM      = BM;
    constexpr int kWarps   = QCfg<BM>::warps;
    constexpr int kThreads = kWarps * 32;
    constexpr int kMFrag   = kBM / 16;
    constexpr int kNFrag   = (kBN / kWarps) / 8;

    __shared__ __align__(16) signed char As[2][kBM][kBK];
    __shared__ __align__(16) signed char Bs[2][kBN][kBK];
    __shared__ float Bsc[2][2][kBN];    // [buf][sub-block][row] = weight scale d
    __shared__ float Asc[2][2][kBM];    // [buf][sub-block][row] = activation scale
    // No lattice table here — Q8_0 needs no decode — so this is 4 KB smaller than
    // the IQ1_S sibling at the same tile.
    static_assert(sizeof(As) + sizeof(Bs) + sizeof(Bsc) + sizeof(Asc) <= 48 * 1024,
                  "static shared exceeds the 48 KB per-block limit");

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int grp  = lane >> 2;
    const int tig  = lane & 3;
    const int sub  = lane >> 3;
    const int lrow = lane & 7;
    const int m0   = blockIdx.y * kBM;
    const int n0   = blockIdx.x * kBN;
    const int blk_per_row = K / kSub;
    const int nk   = K / kBK;

    float accf[kMFrag][kNFrag][4];
#pragma unroll
    for (int i = 0; i < kMFrag; ++i)
#pragma unroll
        for (int j = 0; j < kNFrag; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) accf[i][j][e] = 0.0f;

    auto stage = [&](int buf, int k0) {
        // A: 16 B chunks, exactly as the sibling stages it. A row past M is zeroed
        // rather than left stale — the mma computes it regardless and only the
        // store is guarded, so a NaN there would not stay in its own row.
        for (int s = tid; s < kBM * (kBK / 16); s += kThreads) {
            const int r = s / (kBK / 16), c = (s % (kBK / 16)) << 4;
            const int gk = k0 + c;
            signed char* dst = &As[buf][r][qswz(c, r)];
            if (m0 + r < M && gk < K)
                *reinterpret_cast<uint4*>(dst) =
                    *reinterpret_cast<const uint4*>(A + (size_t)(m0 + r) * K + gk);
            else
                *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
        }
        for (int u = tid; u < kBM * (kBK / kSub); u += kThreads) {
            const int r = u >> 1, half = u & 1;
            const int kb = (k0 + half * kSub) / kSub;
            Asc[buf][half][r] =
                (m0 + r < M) ? sa[(size_t)(m0 + r) * (K / kSub) + kb] : 0.0f;
        }
        // W codes, 2 bytes per thread. `h2` is the uint16 index inside the 32-byte
        // sub-block, so 16 consecutive lanes cover one sub-block contiguously.
        for (int u = tid; u < kBN * (kBK / kSub) * (kSub / 2); u += kThreads) {
            const int h2   = u & 15;
            const int unit = u >> 4;
            const int r    = unit >> 1;
            const int half = unit & 1;
            const int gn   = n0 + r;
            uint16_t v = 0;
            if (gn < N) {
                const BlockQ80& b = W[(size_t)gn * blk_per_row + (k0 + half * kSub) / kSub];
                // b.qs is at +2 from a 34-byte stride: always 2-byte aligned, never
                // 16-byte aligned. This read is the reason the staging is uint16.
                v = reinterpret_cast<const uint16_t*>(b.qs)[h2];
            }
            *reinterpret_cast<uint16_t*>(&Bs[buf][r][qswz(half * kSub + h2 * 2, r)]) = v;
        }
        // W scales, one per (row, sub-block). Zero past N so the columns the mma
        // computes and the store discards contribute nothing.
        for (int u = tid; u < kBN * (kBK / kSub); u += kThreads) {
            const int r = u >> 1, half = u & 1;
            const int gn = n0 + r;
            Bsc[buf][half][r] =
                (gn < N)
                    ? __half2float(__ushort_as_half(
                          W[(size_t)gn * blk_per_row + (k0 + half * kSub) / kSub].d))
                    : 0.0f;
        }
    };

    stage(0, 0);
    int buf = 0;
    for (int t = 0; t < nk; ++t) {
        __syncthreads();
        if (t + 1 < nk) stage(buf ^ 1, (t + 1) * kBK);

        // One mma per 32 k, drained immediately, because 32 is both the Q8_0 block
        // and the instruction's k-extent. That coincidence is the whole design.
#pragma unroll
        for (int half = 0; half < kBK / kSub; ++half) {
            const int kk = half * kSub;
            unsigned af[kMFrag][4], bf[kNFrag][2];
#pragma unroll
            for (int i = 0; i < kMFrag; ++i) {
                const int row = i * 16 + (sub & 1) * 8 + lrow;
                qldm_x4(af[i][0], af[i][1], af[i][2], af[i][3],
                        &As[buf][row][qswz(kk + (sub >> 1) * 16, row)]);
            }
#pragma unroll
            for (int jp = 0; jp < kNFrag; jp += 2) {
                const int col = warp * (kBN / kWarps) + (jp + (sub >> 1)) * 8 + lrow;
                qldm_x4(bf[jp][0], bf[jp][1], bf[jp + 1][0], bf[jp + 1][1],
                        &Bs[buf][col][qswz(kk + (sub & 1) * 16, col)]);
            }
            float saA[kMFrag], saB[kMFrag];
#pragma unroll
            for (int i = 0; i < kMFrag; ++i) {
                saA[i] = Asc[buf][half][i * 16 + grp];
                saB[i] = Asc[buf][half][i * 16 + grp + 8];
            }
#pragma unroll
            for (int j = 0; j < kNFrag; ++j) {
                const int c0 = warp * (kBN / kWarps) + j * 8 + tig * 2;
                const float sc0 = Bsc[buf][half][c0];
                const float sc1 = Bsc[buf][half][c0 + 1];
#pragma unroll
                for (int i = 0; i < kMFrag; ++i) {
                    int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
                        : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]),
                          "r"(bf[j][0]), "r"(bf[j][1]));
                    accf[i][j][0] += (float)a0 * sc0 * saA[i];
                    accf[i][j][1] += (float)a1 * sc1 * saA[i];
                    accf[i][j][2] += (float)a2 * sc0 * saB[i];
                    accf[i][j][3] += (float)a3 * sc1 * saB[i];
                }
            }
        }
        buf ^= 1;
    }

#pragma unroll
    for (int i = 0; i < kMFrag; ++i) {
#pragma unroll
        for (int j = 0; j < kNFrag; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int gm = m0 + i * 16 + grp + (e >> 1) * 8;
                const int gn = n0 + warp * (kBN / kWarps) + j * 8 + tig * 2 + (e & 1);
                if (gm < M && gn < N) C[(size_t)gm * N + gn] = accf[i][j][e];
            }
        }
    }
}

}  // namespace

bool k3_proj_q8_mma_gemm(float* C, const signed char* A, const float* sa,
                         const void* W, int M, int N, int K, cudaStream_t stream) {
    // K must be a whole number of BK tiles: the sub-block index is derived by shift
    // and mask and there is no partial-tile form. K3's projection K values (hidden
    // 7168, expert_latent 3584, and the per-rank qkv bands) are all multiples of 64.
    if (!C || !A || !sa || !W) return false;
    if (M <= 0 || N <= 0 || K <= 0 || (K % kBK) != 0) return false;

    // DECLINE BELOW THE CROSSOVER, because this kernel is 8-50x SLOWER than the GEMV
    // at M=1 and a caller that reached for it unconditionally would be worse off than
    // main. Measured on one H200, M sequential k3_proj_ggml_f32 calls against one
    // batched call (the quantise charged to the batched arm), ms:
    //
    //   N=7168 K=1536   M=1 0.12x   32 3.01x   64  7.97x  128 15.89x  256 24.40x
    //   N=7168 K=3584   M=1 0.07x   32 1.86x   64  5.08x  128  7.93x  256 13.89x
    //   N=3584 K=7168   M=1 0.03x   32 0.87x   64  2.34x  128  4.43x  256  7.13x
    //   N=1536 K=7168   M=1 0.02x   32 0.48x   64  1.27x  128  2.50x  256  5.08x
    //
    // The GEMM's time is nearly FLAT in M (0.093 -> 0.095 ms across that whole range
    // on the first shape) because it is bound by the single weight pass, while the
    // GEMV pays that pass M times. So the crossover is where one weight pass equals M
    // GEMV rows, and it moves with N rather than with K: the grid is N/kBN wide, so a
    // narrow N starves the device and needs a larger M to pay for itself. The four
    // measured crossovers (~12, ~20, ~40, ~90) all satisfy M*N ~ 1.4e5, which is what
    // this guard encodes, with M >= 32 as a floor because below that the largest tile
    // cannot fill even one mma.
    //
    // Returning false rather than running slowly is the same contract every other
    // declining kernel in this tree uses: the caller keeps its existing path.
    if (M < 32 || (long)M * (long)N < 131072L) return false;

    const auto go = [&](auto tag) {
        constexpr int BM = decltype(tag)::value;
        dim3 grid((unsigned)((N + kBN - 1) / kBN), (unsigned)((M + BM - 1) / BM));
        k3_pdl_launch(grid, QCfg<BM>::warps * 32, 0, stream, proj_q8_mma_kernel<BM>,
                      C, A, sa, reinterpret_cast<const BlockQ80*>(W), M, N, K);
    };
    if (M <= kMSwitch) go(std::integral_constant<int, 32>{});
    else               go(std::integral_constant<int, 64>{});
    return true;
}

}  // namespace k3
}  // namespace kernels
}  // namespace sparkinfer
