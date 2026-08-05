// ===========================================================================
// IQ1_S EXPERT GEMM ON THE INT8 TENSOR CORES.
// ===========================================================================
// The routed experts are ~531 of UD-IQ1_S's 553 GiB. Every other tensor-core
// opportunity in K3 is rounding error next to this one, and this is the only
// kernel in the tree where the weights that dominate the model meet an
// instruction that can consume them at full rate.
//
// ---------------------------------------------------------------------------
// WHY IQ1_S IS AN INT8 FORMAT, EXACTLY
// ---------------------------------------------------------------------------
// IQ1_S reconstructs a weight as
//
//     w = dl * (g + delta),   g in {-1,0,+1},   delta = +/-0.125,   dl = d * (2*sc + 1)
//
// with dl and delta CONSTANT across each 32-value sub-block. Multiply through by 8:
//
//     w = (dl/8) * (8g + s),   s = +/-1,   8g + s in {-9,-7,-1,+1,+7,+9}
//
// so a sub-block is an int8 vector times ONE scalar. This is not the argument
// prefill_gemm_i8.cu makes for Q4_K/Q6_K ("quantizing to int8 is strictly higher
// precision than what is stored", an inequality) — it is an EQUALITY. Both sides are
// a single rounding of the same exact real product, so the conversion is bit-for-bit
// lossless and there is no accuracy trade to argue about. See
// kernels/tests/k3_moe_iq1s_mma_cpu_test.cpp, which checks it over a million values
// and runs without a GPU.
//
// THE SECOND HALF OF THE FIT IS THE INSTRUCTION. mma.sync.aligned.m16n8k32.s8 reduces
// exactly 32 values of k per issue, and an IQ1_S sub-block is exactly 32 values under
// one scale. The scale domain and the instruction's k-extent COINCIDE: the int32
// accumulator is drained and scaled once per mma, and no scale is ever applied inside
// a reduction. A 16- or 64-value sub-block would have forced either a partial-k mma or
// a scale inside the reduction; at 32 it is exact by construction.
//
// A consequence worth stating because it inverts the usual expectation: this path is
// MORE accurate than the scalar one it replaces, not less. The 32-term int32 dot is
// exact — integers, no rounding — so a K=3584 reduction carries 112 rounded adds
// against the scalar path's 3584. The CPU test measures 3.3e-08 against 1.7e-07 on the
// well-conditioned scale, a 5x improvement.
//
// ---------------------------------------------------------------------------
// THE SHAPE THIS IS TILED FOR, AND WHY IT IS NOT prefill_gemm_i8's SHAPE
// ---------------------------------------------------------------------------
// 896 experts at top_k 16 means a T-token chunk sends only T*16/896 rows to each
// expert: ~36 rows at T=2048, ~73 at T=4096. The expert GEMM is STRUCTURALLY SKINNY in
// M and no chunk size available in a 32k prefill changes that — reaching M=128 would
// need T=7168. So BM is 32, not prefill_gemm_i8.cu's 128: a 128-row tile would run
// three quarters empty on the shape this kernel actually sees, and the M dimension is
// the one that cannot be widened by asking for a bigger chunk.
//
// BN stays at 128 and BK at 64. BK=64 is two IQ1_S sub-blocks, and because k0 is a
// multiple of 64 and 256 % 64 == 0 a tile can never straddle a 256-value block
// boundary — which is what keeps the (block, sub-block) split a shift and a mask
// rather than a division.
//
// ---------------------------------------------------------------------------
// WHAT THIS DOES NOT DO
// ---------------------------------------------------------------------------
// Nothing routes to it yet. At M=1 — which is every path K3 runs today, decode and
// prefill alike, because prompt ingestion is a forward_token loop — an m16 tile is
// 1/16 occupied and this kernel is strictly worse than the shipped GEMV. It becomes
// the right kernel only once a batched prefill exists to give it M>1, and it is
// landed first because it is the piece that decides whether that batching is worth
// building: if IQ1_S could not reach the tensor cores losslessly, the rest of the
// batched path would be arguing over a much smaller prize.

#include "sparkinfer/kernels/kimi_k3.h"
#include "sparkinfer/kernels/iq1s_tables.h"
#include "k3_pdl.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdlib>
#include <mutex>

namespace sparkinfer {
namespace kernels {
namespace k3 {
namespace {

constexpr int kQK   = 256;    // values per IQ1_S block
constexpr int kSub  = 32;     // values per sub-block == mma.m16n8k32's k-extent
constexpr int kBM   = 32;     // see the shape note above
constexpr int kBN   = 128;
constexpr int kBK   = 64;     // two sub-blocks
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kMFrag = kBM / 16;              // 2
constexpr int kNFrag = (kBN / kWarps) / 8;    // 4

// 50 bytes, matching ggml block_iq1_s. Redeclared here rather than shared with
// k3_kernels.cu because that struct is private to that TU; the static_assert is what
// keeps the two from drifting silently.
struct BlockIQ1S { uint16_t d; uint8_t qs[32]; uint16_t qh[8]; };
static_assert(sizeof(BlockIQ1S) == 50, "IQ1_S block must be 50 bytes");

// The packed 2-bit lattice: field j holds (grid[j] & 3), sign-extended on read. Packed
// host-side from the SAME iq1s_grid_host every other consumer uses. This TU carries its
// own copy because a __device__ symbol does not cross translation units; the packing
// rule is duplicated from k3_kernels.cu's ensure_iq1s_tables and the CPU test rebuilds
// it independently, so a divergence fails a test rather than biasing a tensor.
__device__ static uint16_t g_iq1s_grid_p[SPARKINFER_IQ1S_NGRID];

// Upload once per device. Must not run inside a stream capture — same constraint, and
// the same reason, as ensure_iq1s_tables: cudaMemcpyToSymbol is illegal there.
bool ensure_grid_uploaded() {
    static std::mutex mu;
    static bool done[64] = {false};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    std::lock_guard<std::mutex> lk(mu);
    if (done[dev]) return true;

    static uint16_t packed[SPARKINFER_IQ1S_NGRID];
    static bool packed_ready = false;
    if (!packed_ready) {
        for (int i = 0; i < SPARKINFER_IQ1S_NGRID; ++i) {
            const int8_t* g = (const int8_t*)&iq1s_grid_host[i];
            uint16_t v = 0;
            for (int j = 0; j < 8; ++j) v |= (uint16_t)((uint16_t)g[j] & 3u) << (2 * j);
            packed[i] = v;
        }
        packed_ready = true;
    }
    if (cudaMemcpyToSymbol(g_iq1s_grid_p, packed, sizeof(packed)) != cudaSuccess) return false;
    done[dev] = true;
    return true;
}

// XOR swizzle at 16B granularity, identical to prefill_gemm_i8.cu's pf_swz: chunk c of
// row r lives at chunk (c ^ (r & 3)). Rows 0..3 — the stride the operand loads walk —
// land on disjoint banks. Both the decode's stores and the ldmatrix loads go through
// this one function, so the layout cannot disagree with itself.
__device__ __forceinline__ int swz(int k, int row) {
    return (((k >> 4) ^ (row & 3)) << 4) | (k & 15);
}

__device__ __forceinline__ void ldm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                       const signed char* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

__device__ __forceinline__ int unpack2(uint16_t pw, int j) {
    return ((int)((uint32_t)pw << (30 - 2 * j))) >> 30;   // sign-extend 2-bit field j
}

// ---------------------------------------------------------------------------
// Per-row symmetric int8 quantization of the f32 activations.
// ---------------------------------------------------------------------------
// K3 is f32 end to end, so this is the f32 twin of launch_prefill_quantize_rows_i8
// (which takes bf16). Per-ROW scales: the scale is constant in k, which is exactly
// what lets the GEMM hoist it out of the reduction and apply it once in the epilogue.
//
// Quantizing the activation to int8 is not a new approximation being introduced here:
// llama.cpp's own vec_dot contract for IQ1_S converts the activation to block_q8_K
// first, and k3_kernels.cu already carries that path (block_dot_q8k) as the pinned
// reference. This is the same int8 activation, with a per-row rather than per-256
// scale.
__global__ void quantize_rows_i8_f32_kernel(const float* __restrict__ x,
                                            signed char* __restrict__ q,
                                            float* __restrict__ scale,
                                            int rows, int cols) {
    k3_pdl_sync();
    const int r = blockIdx.x, lane = threadIdx.x;
    if (r >= rows) return;
    const float* xr = x + (size_t)r * cols;
    float amax = 0.0f;
    for (int c = lane; c < cols; c += 32) amax = fmaxf(amax, fabsf(xr[c]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.0f;
    if (lane == 0) scale[r] = d;
    signed char* qr = q + (size_t)r * cols;
    for (int c = lane; c < cols; c += 32)
        qr[c] = (signed char)((amax == 0.0f) ? 0 : (int)rintf(xr[c] / d));
}

// ---------------------------------------------------------------------------
// C[M,N] = (A[M,K] @ W[N,K]^T) with W in IQ1_S.
// ---------------------------------------------------------------------------
// `rows` optionally gathers A's M rows out of a larger token buffer (the MoE dispatch's
// per-expert row list). Null means identity, which is what the standalone test uses.
//
// The launch_bounds second argument is load-bearing, not decorative: the accumulator
// block is kMFrag*kNFrag*4 floats plus the fragment registers, and letting ptxas spend
// freely here costs the second resident block on a kernel whose whole purpose is to
// keep the tensor pipe fed.
__global__ __launch_bounds__(kThreads, 4)
void moe_iq1s_mma_kernel(float* __restrict__ C,
                         const signed char* __restrict__ A,
                         const float* __restrict__ sa,
                         const BlockIQ1S* __restrict__ W,
                         const int* __restrict__ rows,
                         int M, int N, int K) {
    k3_pdl_sync();
    // __align__(16) is REQUIRED, not defensive: ldmatrix.sync.aligned takes a shared
    // address that must be 16 B aligned, and swz() only ever returns multiples of 16
    // WITHIN a row — which buys nothing unless the array base is aligned too. A char
    // array carries alignment 1 by type, so without this the guarantee has a hole that
    // happens to be filled by whatever nvcc chose to do that day.
    __shared__ __align__(16) signed char As[2][kBM][kBK];
    __shared__ __align__(16) signed char Bs[2][kBN][kBK];
    __shared__ float       Bsc[2][2][kBN];        // [buf][sub-block][row] = dl/8
    __shared__ uint16_t    Sgrid[SPARKINFER_IQ1S_NGRID];
    // 4 + 16 + 2 + 4 = 26 KB, inside the 48 KB static-shared ceiling, and 4 resident
    // blocks fit the SM's 228 KB. Both halves of __launch_bounds__ depend on that.
    static_assert(sizeof(As) + sizeof(Bs) + sizeof(Bsc) + sizeof(Sgrid) <= 48 * 1024,
                  "static shared exceeds the 48 KB per-block limit");

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int grp  = lane >> 2;          // accumulator row within the 8-row half
    const int tig  = lane & 3;           // accumulator column pair
    const int sub  = lane >> 3;          // which ldmatrix tile this thread addresses
    const int lrow = lane & 7;
    const int m0   = blockIdx.y * kBM;
    const int n0   = blockIdx.x * kBN;
    const int blk_per_row = K / kQK;
    const int nk   = K / kBK;

    // The lattice gather is divergent across the warp, which is exactly the access the
    // constant unit serialises one address per cycle. 4 KB in shared costs one staging
    // pass and turns it into a bank-conflict problem instead.
    for (int i = tid; i < SPARKINFER_IQ1S_NGRID; i += kThreads) Sgrid[i] = g_iq1s_grid_p[i];

    float accf[kMFrag][kNFrag][4];
#pragma unroll
    for (int i = 0; i < kMFrag; ++i)
#pragma unroll
        for (int j = 0; j < kNFrag; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) accf[i][j][e] = 0.0f;

    // Stage one BK=64 tile of A (gathered) and of W (decoded from IQ1_S).
    auto stage = [&](int buf, int k0) {
        // A: kBM rows x 64 B = 128 chunks of 16 B over 128 threads, one each.
        {
            const int r = tid >> 2, c = (tid & 3) << 4;
            const int gk = k0 + c;
            signed char* dst = &As[buf][r][swz(c, r)];
            const int src_row = (m0 + r < M) ? (rows ? rows[m0 + r] : m0 + r) : -1;
            if (src_row >= 0 && gk < K)
                *reinterpret_cast<uint4*>(dst) =
                    *reinterpret_cast<const uint4*>(A + (size_t)src_row * K + gk);
            else
                *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
        }
        // W: 128 rows x 2 sub-blocks = 256 decode units over 128 threads, two each.
        // One unit is one (row, 32 values) — one qh, four qs bytes, four lattice
        // lookups — and it emits 32 int8 plus the single scale that covers them.
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const int unit = tid + u * kThreads;
            const int r    = unit >> 1;                 // row within the BN tile
            const int half = unit & 1;                  // which sub-block of the BK tile
            const int kk   = k0 + half * kSub;
            const int gn   = n0 + r;
            if (gn >= N) {
                // Out-of-range N rows must be ZERO, not stale: the mma computes them
                // regardless and they are simply not stored, but a NaN left in shared
                // would not stay in its own column.
                *reinterpret_cast<uint4*>(&Bs[buf][r][swz(half * kSub, r)])      = make_uint4(0u,0u,0u,0u);
                *reinterpret_cast<uint4*>(&Bs[buf][r][swz(half * kSub + 16, r)]) = make_uint4(0u,0u,0u,0u);
                Bsc[buf][half][r] = 0.0f;
                continue;
            }
            const int ib   = kk / kQK;                  // 256-value block; kBK|kQK so no straddle
            const int ib32 = (kk % kQK) / kSub;         // sub-block inside it
            const BlockIQ1S& b = W[(size_t)gn * blk_per_row + ib];
            const uint16_t h = b.qh[ib32];
            const float dl = __half2float(__ushort_as_half(b.d)) * (float)(2 * ((h >> 12) & 7) + 1);
            const int s = (h & 0x8000) ? -1 : 1;
            Bsc[buf][half][r] = dl * 0.125f;            // dl/8, exact: 0.125 is a power of two

            // 16 B alignment is required by the two uint4 stores below and is NOT
            // implied for a local array of char.
            __align__(16) signed char m8[kSub];
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const uint32_t idx =
                    (uint32_t)b.qs[4 * ib32 + l] | (((uint32_t)(h >> (3 * l)) & 7u) << 8);
                const uint16_t pw = Sgrid[idx];
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    m8[l * 8 + j] = (signed char)(8 * unpack2(pw, j) + s);
            }
            // Two 16 B stores, because the swizzle is defined at 16 B granularity.
            *reinterpret_cast<uint4*>(&Bs[buf][r][swz(half * kSub, r)]) =
                *reinterpret_cast<const uint4*>(m8);
            *reinterpret_cast<uint4*>(&Bs[buf][r][swz(half * kSub + 16, r)]) =
                *reinterpret_cast<const uint4*>(m8 + 16);
        }
    };

    __syncthreads();                 // Sgrid must be complete before the first decode
    stage(0, 0);
    int buf = 0;
    for (int t = 0; t < nk; ++t) {
        __syncthreads();
        if (t + 1 < nk) stage(buf ^ 1, (t + 1) * kBK);

        // Each half is one IQ1_S sub-block and one mma: the accumulator starts at zero,
        // takes exactly the 32 k it is scaled for, and is drained immediately. This is
        // the whole reason the format and the instruction fit.
#pragma unroll
        for (int half = 0; half < kBK / kSub; ++half) {
            const int kk = half * kSub;
            unsigned af[kMFrag][4], bf[kNFrag][2];
#pragma unroll
            for (int i = 0; i < kMFrag; ++i) {
                const int row = i * 16 + (sub & 1) * 8 + lrow;
                ldm_x4(af[i][0], af[i][1], af[i][2], af[i][3],
                       &As[buf][row][swz(kk + (sub >> 1) * 16, row)]);
            }
#pragma unroll
            for (int jp = 0; jp < kNFrag; jp += 2) {
                const int col = warp * (kBN / kWarps) + (jp + (sub >> 1)) * 8 + lrow;
                ldm_x4(bf[jp][0], bf[jp][1], bf[jp + 1][0], bf[jp + 1][1],
                       &Bs[buf][col][swz(kk + (sub & 1) * 16, col)]);
            }
#pragma unroll
            for (int j = 0; j < kNFrag; ++j) {
                // dl depends on (n, sub-block) and NOT on m, so the two column scales
                // are loaded once and reused across every M fragment.
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
                    // Elements 0,1 are row grp; 2,3 are row grp+8. Columns alternate.
                    accf[i][j][0] += (float)a0 * sc0;
                    accf[i][j][1] += (float)a1 * sc1;
                    accf[i][j][2] += (float)a2 * sc0;
                    accf[i][j][3] += (float)a3 * sc1;
                }
            }
        }
        buf ^= 1;
    }

    // The per-token activation scale is constant in k, so it never entered the loop and
    // is applied exactly once, here.
    //
    // `sa` is indexed by the SOURCE token, not by the compact output row: when `rows`
    // gathers this expert's tokens out of a larger buffer, row gm of C came from token
    // rows[gm] and carries that token's scale. C itself stays compact — the caller
    // scatters it — so the two indices are deliberately different and must not be
    // collapsed into one.
#pragma unroll
    for (int i = 0; i < kMFrag; ++i) {
#pragma unroll
        for (int j = 0; j < kNFrag; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int gm = m0 + i * 16 + grp + (e >> 1) * 8;
                const int gn = n0 + warp * (kBN / kWarps) + j * 8 + tig * 2 + (e & 1);
                if (gm < M && gn < N)
                    C[(size_t)gm * N + gn] = accf[i][j][e] * sa[rows ? rows[gm] : gm];
            }
        }
    }
}

}  // namespace

bool k3_moe_iq1s_mma_quantize_rows(signed char* q, float* scale, const float* x,
                                   int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || (cols & 31) != 0) return false;
    k3_pdl_launch((unsigned)rows, 32, 0, stream, quantize_rows_i8_f32_kernel,
                  x, q, scale, rows, cols);
    return true;
}

bool k3_moe_iq1s_mma_gemm(float* C, const signed char* A, const float* sa,
                          const void* W, const int* rows,
                          int M, int N, int K, cudaStream_t stream) {
    // K must be a whole number of IQ1_S blocks — the tile arithmetic derives the
    // sub-block index by shift and mask and has no partial-block form. K3's expert
    // dims (3584, 3072, 7168) are all multiples of 256.
    if (M <= 0 || N <= 0 || K <= 0 || (K % kQK) != 0) return false;
    if (!ensure_grid_uploaded()) return false;
    dim3 grid((unsigned)((N + kBN - 1) / kBN), (unsigned)((M + kBM - 1) / kBM));
    k3_pdl_launch(grid, kThreads, 0, stream, moe_iq1s_mma_kernel,
                  C, A, sa, reinterpret_cast<const BlockIQ1S*>(W), rows, M, N, K);
    return true;
}

}  // namespace k3
}  // namespace kernels
}  // namespace sparkinfer
