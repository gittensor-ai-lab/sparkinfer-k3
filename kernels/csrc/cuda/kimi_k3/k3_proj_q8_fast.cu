// Q8_0 x Q8_0 multi-row GEMV with ONE barrier instead of two per output row.
//
// ---------------------------------------------------------------------------
// WHAT THE ROW LOOP COSTS, AND WHY IT IS NOT THE ARITHMETIC
// ---------------------------------------------------------------------------
// proj_q8_0_q8_0_multirow_kernel accumulates ROWS output rows per block and then
// finishes like this:
//
//     for (int r = 0; r < ROWS; ++r) {
//         const float v = block_sum<BLOCK>(acc[r], shm);
//         if (threadIdx.x == 0 && n0 + r < n_rows) y[n0 + r] = v;
//     }
//
// block_sum contains TWO __syncthreads(). So a ROWS=16 launch pays **32 block-wide
// barriers**, strictly serialised, one dependent reduction after another — and it
// pays them after each thread has done almost no work at all, because K3's shapes
// put blocks_per_row at or below the block width:
//
//     ffn_routed_up      N 7168  K 3584  nb 112  BLOCK 128  ROWS 16  -> 1 iteration
//     attn_output (KDA)  N 7168  K 1536  nb  48  BLOCK  64  ROWS 16  -> 1 iteration
//     ffn_down_shexp     N 7168  K  768  nb  24  BLOCK  32  ROWS 16  -> 1 iteration
//     ffn_routed_down    N 3584  K 7168  nb 224  BLOCK 128  ROWS  8  -> 2 iterations
//
// One dp4a pass, then thirty-two barriers. That is the shape of a kernel whose cost
// is its epilogue, and it is the reason this path sits far below what the same block
// layout reaches elsewhere: proj_q8_0_q8_0_fused4_kernel measures 1.33 TB/s on the
// IDENTICAL 34-byte stride, which is the evidence that the stride, the 2-byte loads
// and the block format are not what is holding the multi-row kernel back.
//
// ---------------------------------------------------------------------------
// ONE BARRIER, AND THE SAME FLOAT
// ---------------------------------------------------------------------------
// The ROWS reductions are independent, so they do not need to be serialised — only
// separated from the shared write. Reduce all ROWS within each warp using shuffles
// alone (no barrier at all), stage NWARP x ROWS partials, cross ONE __syncthreads(),
// and let thread r fold its own row's warp partials.
//
// BIT-IDENTICAL, and by construction rather than by tolerance. block_sum does:
// a shuffle-down butterfly over the warp, lane 0 writes shm[warp], barrier, thread 0
// sums shm[0..NWARP-1] in INCREASING w. This does the same butterfly over the same
// lanes, and thread r sums shm[w * ROWS + r] over the same increasing w. Same
// operands, same order, same roundings — the only thing that changes is that the
// sixteen of them no longer wait for each other.
//
// The activation quantisation is duplicated here rather than reached for, because
// k3_proj_ggml_f32 is the only exported entry and it does quantise+project as one
// call. The arithmetic below is copied line for line — the serial amax scan included,
// which is what preserves ggml's first-maximum sign rule — so both paths emit the
// same Q8_0 bytes and the A/B compares the epilogue and nothing else.
//
// Both the multi-row and the FOUR-TENSOR fused kernel are covered: the fused one calls
// block_sum 4 * ROWS times, so at K3's ROWS 4 it pays 32 barriers on every one of the
// 69 KDA layers.
//
// SPARKINFER_K3_PROJ_1BAR=0 declines both, and every projection goes back through
// k3_proj_ggml_f32 / k3_proj_ggml_f32_x4 exactly as main runs them.

#include "sparkinfer/kernels/k3_proj_rowbudget.h"
#include "sparkinfer/kernels/k3_kda_qkvg_rows.h"
#include "sparkinfer/kernels/kimi_k3_fast.h"
#include "k3_pdl.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>

namespace sparkinfer {
namespace kernels {
namespace k3 {

namespace {

struct BlockQ8_0 {   // 34 bytes, matches ggml block_q8_0 exactly
    uint16_t d;      // fp16 scale
    int8_t   qs[32];
};
static_assert(sizeof(BlockQ8_0) == 34, "bad block_q8_0 layout");
static_assert(alignof(BlockQ8_0) == 2, "get_int_b2 assumes 2-byte alignment");

// Four int8 weights as one packed int from a 2-byte-aligned source. BlockQ8_0 is 34
// bytes with alignment 2, so qs is 4-byte aligned only for odd blocks and nvcc cannot
// legally widen a scalar loop; two aligned uint16 loads rebuild the identical four
// bytes in the identical lane order.
__device__ __forceinline__ int get_int_b2(const int8_t* __restrict__ qs, int i32) {
    const uint16_t* x16 = (const uint16_t*)qs;
    int x32 = (int)x16[2 * i32 + 0];
    x32 |= (int)x16[2 * i32 + 1] << 16;
    return x32;
}

// ONE WARP PER QUANT BLOCK, not one thread.
//
// k3_kernels.cu gives one CUDA THREAD to each 32-value quant block, so the launch is
// ceil(nb/128) blocks of 128 — and nb is K/32, which at K3's shapes is tiny:
//
//     K 7168 -> nb 224 -> 2 blocks (224 useful threads)
//     K 3584 -> nb 112 -> 1 block  (112)
//     K 1536 -> nb  48 -> 1 block  ( 48)
//     K  128 -> nb   4 -> 1 block  (  4 threads on a 132-SM part)
//
// It is the single most frequently launched kernel in the decode — 859 launches per
// token per rank, 3.60 ms, 8.8% of GPU time at 4.19 us each to move ~28 KB, i.e.
// 6.7 GB/s. Nothing about that is arithmetic.
//
// A warp per block is 32x the threads: nb 224 becomes 224 warps = 28 blocks of 256.
//
// BIT-IDENTICAL, and the in-tree comment this replaces is the reason to say why
// carefully. It reads: "the serial max scan preserves ggml's first-maximum sign rule,
// and a parallel reduction would let equal-magnitude +/- values pick a
// schedule-dependent sign." That concern is real for quantisers that select a
// REPRESENTATIVE value, but q8_0's scan is `amax = fmaxf(amax, fabsf(x[j]))` — a max
// over MAGNITUDES. No sign is carried, fmaxf over a fixed set is order-independent for
// non-NaN, and d = amax/127 is therefore the same float whichever order it is reduced
// in. Every qs[j] is an independent per-element rounding. The test compares the
// emitted bytes against the reference kernel rather than trusting this paragraph.
__global__ void quantize_q8_0_warp_kernel(BlockQ8_0* __restrict__ out,
                                          const float* __restrict__ x, int n_blocks) {
    k3_pdl_sync();
    const int b = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (b >= n_blocks) return;
    const int lane = threadIdx.x & 31;
    const float v = x[(size_t)b * 32 + lane];
    float amax = fabsf(v);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
    const float d  = amax / 127.0f;
    const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
    if (lane == 0) out[b].d = __half_as_ushort(__float2half_rn(d));
    out[b].qs[lane] = (int8_t)__float2int_rn(v * id);
}

// The reference shape, kept so the toggle is an exact A/B on one binary.
__global__ void quantize_q8_0_same_kernel(BlockQ8_0* __restrict__ out,
                                          const float* __restrict__ x, int n_blocks) {
    k3_pdl_sync();   // no-op unless launched programmatically
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const float* xb = x + (size_t)b * 32;
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(xb[j]));
    const float d = amax / 127.0f;
    const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
    out[b].d = __half_as_ushort(__float2half_rn(d));
#pragma unroll
    for (int j = 0; j < 32; ++j)
        out[b].qs[j] = (int8_t)__float2int_rn(xb[j] * id);
}

template <int BLOCK, int ROWS>
__global__ void proj_q8_multirow_1bar_kernel(float* __restrict__ y,
                                             const BlockQ8_0* __restrict__ x,
                                             const BlockQ8_0* __restrict__ W,
                                             int blocks_per_row, int n_rows) {
    k3_pdl_sync();   // no-op unless launched programmatically
    constexpr int NWARP = BLOCK / 32;
    const int n0   = blockIdx.x * ROWS;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    __shared__ float shm[NWARP * ROWS];

    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;

    for (int b = threadIdx.x; b < blocks_per_row; b += BLOCK) {
        // ONE activation block, staged once and reused by every row in this block.
        int xq[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) xq[i] = get_int_b2(x[b].qs, i);
        const float dx = __half2float(__ushort_as_half(x[b].d));

#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (n0 + r >= n_rows) continue;
            const BlockQ8_0* row = W + (size_t)(n0 + r) * blocks_per_row;
            int sumi = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i)
                sumi = __dp4a(get_int_b2(row[b].qs, i), xq[i], sumi);
            const float dw = __half2float(__ushort_as_half(row[b].d));
            acc[r] += (float)sumi * (dw * dx);
        }
    }

    // Every row's warp reduction, with no barrier between them. Same butterfly over
    // the same lanes block_sum uses, so each warp's partial is the same float it
    // would have produced there.
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc[r] += __shfl_down_sync(0xffffffff, acc[r], off);
        if (lane == 0) shm[warp * ROWS + r] = acc[r];
    }

    __syncthreads();   // the only one

    // Thread r folds ITS row across the warps, in increasing w — block_sum's order.
    if (threadIdx.x < ROWS) {
        const int r = threadIdx.x;
        if (n0 + r < n_rows) {
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < NWARP; ++w) s += shm[w * ROWS + r];
            y[n0 + r] = s;
        }
    }
}

// The SAME epilogue defect, one level worse: the fused four-tensor kernel calls
// block_sum `4 * ROWS` times, so at K3's ROWS 4 that is sixteen reductions and
// **32 barriers** on every one of the 69 KDA layers — for the q/k/v/g group, which is
// the second-largest weight read in the token after the latent-MoE brackets.
//
// Identical fix, identical bit-identity argument: the sixteen reductions are
// independent, so they need to be separated from the shared write, not from each other.
template <int BLOCK, int ROWS>
__global__ void proj_q8_fused4_1bar_kernel(float* __restrict__ y0, float* __restrict__ y1,
                                           float* __restrict__ y2, float* __restrict__ y3,
                                           const BlockQ8_0* __restrict__ x,
                                           const BlockQ8_0* __restrict__ W0,
                                           const BlockQ8_0* __restrict__ W1,
                                           const BlockQ8_0* __restrict__ W2,
                                           const BlockQ8_0* __restrict__ W3,
                                           int blocks_per_row, int n_rows) {
    k3_pdl_sync();   // no-op unless launched programmatically
    constexpr int NWARP = BLOCK / 32;
    constexpr int SLOTS = 4 * ROWS;
    const int n0   = blockIdx.x * ROWS;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    __shared__ float shm[NWARP * SLOTS];
    const BlockQ8_0* Wt[4] = {W0, W1, W2, W3};
    float* Yt[4] = {y0, y1, y2, y3};

    float acc[4][ROWS];
#pragma unroll
    for (int t = 0; t < 4; ++t)
#pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[t][r] = 0.0f;

    for (int b = threadIdx.x; b < blocks_per_row; b += BLOCK) {
        int xq[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) xq[i] = get_int_b2(x[b].qs, i);
        const float dx = __half2float(__ushort_as_half(x[b].d));
#pragma unroll
        for (int t = 0; t < 4; ++t) {
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                if (n0 + r >= n_rows) continue;
                const BlockQ8_0* row = Wt[t] + (size_t)(n0 + r) * blocks_per_row;
                int sumi = 0;
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    sumi = __dp4a(get_int_b2(row[b].qs, i), xq[i], sumi);
                const float dw = __half2float(__ushort_as_half(row[b].d));
                acc[t][r] += (float)sumi * (dw * dx);
            }
        }
    }

#pragma unroll
    for (int t = 0; t < 4; ++t)
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            float v = acc[t][r];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_down_sync(0xffffffff, v, off);
            if (lane == 0) shm[warp * SLOTS + t * ROWS + r] = v;
        }

    __syncthreads();   // the only one

    if (threadIdx.x < SLOTS) {
        const int t = threadIdx.x / ROWS, r = threadIdx.x - t * ROWS;
        if (n0 + r < n_rows) {
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < NWARP; ++w) s += shm[w * SLOTS + t * ROWS + r];
            Yt[t][n0 + r] = s;
        }
    }
}

inline int block_for(int blocks_per_row) {
    if (blocks_per_row <= 32) return 32;
    if (blocks_per_row <= 64) return 64;
    return 128;
}

}  // namespace

void k3_quantize_q8_0(void* out, const float* x, int n_blocks, cudaStream_t stream) {
    if (n_blocks <= 0) return;
    static const bool warp = [] {
        const char* e = std::getenv("SPARKINFER_K3_QUANT_WARP");
        return !(e && e[0] == '0');
    }();
    constexpr int QT = 128;
    if (warp) {
        const int threads = 256;
        const int blocks = (n_blocks * 32 + threads - 1) / threads;
        k3_pdl_launch(dim3(blocks), dim3(threads), 0, stream, quantize_q8_0_warp_kernel,
                      (BlockQ8_0*)out, x, n_blocks);
    } else {
        k3_pdl_launch(dim3((n_blocks + QT - 1) / QT), dim3(QT), 0, stream,
                      quantize_q8_0_same_kernel, (BlockQ8_0*)out, x, n_blocks);
    }
}

bool k3_proj_q8_multirow_1bar(float* y, const float* x, const void* W, int wtype,
                              int N, int K, void* q8_scratch, cudaStream_t stream,
                              bool x_pre_q8) {
    static const bool want = [] {
        const char* e = std::getenv("SPARKINFER_K3_PROJ_1BAR");
        return !(e && e[0] == '0');
    }();
    if (!want) return false;
    if (N <= 0 || K <= 0 || wtype != 8 || !q8_scratch || K % 32 != 0) return false;

    // Only the multi-row shapes. Below these thresholds k3_proj_ggml_f32 takes the
    // single-row kernel, which pays ONE block_sum and is also the one the numeric test
    // pins bit-for-bit — there is nothing to win there and a fallback is free.
    constexpr int MIN_N_W16 = 4096, MIN_N_W8 = 2048, MIN_N_W4 = 1024;
    if (N < MIN_N_W4) return false;

    const int nb = K / 32;
    const int TB = block_for(nb);
    // The N-only tier, kept as the input to the warp budget rather than replaced
    // outright: the budget may only ever LOWER it (widen the grid), never raise it.
    // See sparkinfer/kernels/k3_proj_rowbudget.h — TB is decided from K above and
    // stays exactly as it was, which is what keeps every accumulation order intact.
    const int legacy_rows = (N >= MIN_N_W16) ? 16 : (N >= MIN_N_W8) ? 8 : 4;
    const int rows = k3_proj_rows_for_budget(N, TB, legacy_rows);

    // Validate the geometry BEFORE quantising. Declining after a launch would leave
    // the caller's fallback re-quantising into the same scratch — the same bytes, so
    // harmless, but a decline should cost nothing at all.
    const bool shape_ok = (TB == 32 || TB == 64 || TB == 128) &&
                          (rows == 1 || rows == 2 || rows == 4 ||
                           rows == 8 || rows == 16);
    if (!shape_ok) return false;

    constexpr int QT = 128;
    if (!x_pre_q8) k3_quantize_q8_0(q8_scratch, x, nb, stream);

#define K3_1BAR_LAUNCH(B, R)                                                        \
    k3_pdl_launch(dim3((unsigned)((N + (R) - 1) / (R))), dim3(B), 0, stream,        \
                  proj_q8_multirow_1bar_kernel<B, R>,                               \
                  y, (const BlockQ8_0*)q8_scratch, (const BlockQ8_0*)W, nb, N)

    switch (TB * 100 + rows) {
        case 32 * 100 + 16: K3_1BAR_LAUNCH(32, 16); break;
        case 32 * 100 +  8: K3_1BAR_LAUNCH(32,  8); break;
        case 32 * 100 +  4: K3_1BAR_LAUNCH(32,  4); break;
        case 32 * 100 +  2: K3_1BAR_LAUNCH(32,  2); break;
        case 32 * 100 +  1: K3_1BAR_LAUNCH(32,  1); break;
        case 64 * 100 + 16: K3_1BAR_LAUNCH(64, 16); break;
        case 64 * 100 +  8: K3_1BAR_LAUNCH(64,  8); break;
        case 64 * 100 +  4: K3_1BAR_LAUNCH(64,  4); break;
        case 64 * 100 +  2: K3_1BAR_LAUNCH(64,  2); break;
        case 64 * 100 +  1: K3_1BAR_LAUNCH(64,  1); break;
        case 128 * 100 + 16: K3_1BAR_LAUNCH(128, 16); break;
        case 128 * 100 +  8: K3_1BAR_LAUNCH(128,  8); break;
        case 128 * 100 +  4: K3_1BAR_LAUNCH(128,  4); break;
        case 128 * 100 +  2: K3_1BAR_LAUNCH(128,  2); break;
        case 128 * 100 +  1: K3_1BAR_LAUNCH(128,  1); break;
        default: return false;
    }
#undef K3_1BAR_LAUNCH
    return true;
}

bool k3_proj_q8_fused4_1bar(float* y0, float* y1, float* y2, float* y3, const float* x,
                            const void* W0, const void* W1, const void* W2, const void* W3,
                            int wtype, int N, int K, void* q8_scratch,
                            cudaStream_t stream, bool x_pre_q8) {
    static const bool want = [] {
        const char* e = std::getenv("SPARKINFER_K3_PROJ_1BAR");
        return !(e && e[0] == '0');
    }();
    if (!want) return false;
    if (N <= 0 || K <= 0 || wtype != 8 || K % 32 != 0) return false;
    if (!y0 || !y1 || !y2 || !y3 || !W0 || !W1 || !W2 || !W3 || !q8_scratch) return false;

    constexpr int LEGACY_ROWS = 4;   // k3_proj_ggml_f32_x4's, so the two arms differ
    if (N < LEGACY_ROWS) return false;  // only in the epilogue

    const int nb = K / 32;
    const int TB = block_for(nb);
    if (TB != 32 && TB != 64 && TB != 128) return false;

    // The one shape #107's warp budget skipped — see k3_kda_qkvg_rows.h. At K3's KDA
    // group this walks ROWS 4 -> 2, doubling the grid to 3072 warps and paying an
    // extra 11 MB of L2-resident activation re-read for it. Returns 4 unchanged under
    // SPARKINFER_K3_KDA_QKVG_ROWS=0, which is the shipped geometry on the same binary.
    const int rows = k3_kda_qkvg_rows_for_budget(N, TB, LEGACY_ROWS);
    if (rows != 1 && rows != 2 && rows != 4) return false;

    constexpr int QT = 128;
    if (!x_pre_q8) k3_quantize_q8_0(q8_scratch, x, nb, stream);

    const BlockQ8_0* xq = (const BlockQ8_0*)q8_scratch;
#define K3_1BAR4_LAUNCH(BS, R)                                                   \
    k3_pdl_launch(dim3((unsigned)((N + (R) - 1) / (R))), dim3(BS), 0, stream,     \
                  proj_q8_fused4_1bar_kernel<BS, R>,                              \
                  y0, y1, y2, y3, xq, (const BlockQ8_0*)W0, (const BlockQ8_0*)W1, \
                  (const BlockQ8_0*)W2, (const BlockQ8_0*)W3, nb, N)
    switch (TB * 10 + rows) {
        case 32 * 10 + 4:  K3_1BAR4_LAUNCH(32,  4); break;
        case 32 * 10 + 2:  K3_1BAR4_LAUNCH(32,  2); break;
        case 32 * 10 + 1:  K3_1BAR4_LAUNCH(32,  1); break;
        case 64 * 10 + 4:  K3_1BAR4_LAUNCH(64,  4); break;
        case 64 * 10 + 2:  K3_1BAR4_LAUNCH(64,  2); break;
        case 64 * 10 + 1:  K3_1BAR4_LAUNCH(64,  1); break;
        case 128 * 10 + 4: K3_1BAR4_LAUNCH(128, 4); break;
        case 128 * 10 + 2: K3_1BAR4_LAUNCH(128, 2); break;
        case 128 * 10 + 1: K3_1BAR4_LAUNCH(128, 1); break;
        default: return false;
    }
#undef K3_1BAR4_LAUNCH
    return true;
}

}  // namespace k3
}  // namespace kernels
}  // namespace sparkinfer
