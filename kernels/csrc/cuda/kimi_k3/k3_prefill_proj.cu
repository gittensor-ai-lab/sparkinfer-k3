// Batched Q8_0 projection for prompt prefill. See kimi_k3_prefill.h for why.
//
// The single-token kernel (proj_q8_multirow_1bar_kernel) stages ONE activation block and
// reuses it across ROWS weight rows. That is the right trade when there is one token: the
// activation is the thing you can amortise, because the weight is streamed once anyway.
//
// At prefill the scarcity inverts. There are T activations and still one weight, so the
// WEIGHT is what must be held while several activations pass over it. This kernel keeps
// both amortisations at once -- a tile of ROWS output rows by TOKS tokens -- so one
// 34-byte weight block feeds TOKS dot products and one activation block feeds ROWS.
//
// WHAT THIS DOES NOT DO. It does not touch the loop order ABOVE it. Calling this once per
// token would read every weight T times exactly as today; the amortisation only exists if
// the caller has T tokens of activation in hand at the same layer, which means the driver
// must run `for layer: for token-tile` rather than `for token: for layer`. This kernel is
// the half of that change that lives in CUDA.

#include "sparkinfer/kernels/kimi_k3_prefill.h"
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
    uint16_t d;
    int8_t   qs[32];
};
static_assert(sizeof(BlockQ8_0) == 34, "bad block_q8_0 layout");
static_assert(alignof(BlockQ8_0) == 2, "get_int_b2 assumes 2-byte alignment");

// The block is 2-byte aligned, so a 4-byte load off qs is not guaranteed aligned.
// Rebuild the int from two uint16 loads -- identical bytes, no misaligned access.
// Duplicated from k3_proj_q8_fast.cu deliberately: this file must not depend on that
// one's internal anonymous namespace, and the function is four lines whose contract is
// pinned by the block layout above.
__device__ __forceinline__ int get_int_b2(const int8_t* __restrict__ qs, int i32) {
    const uint16_t* x16 = (const uint16_t*)qs;
    int x32 = (int)x16[2 * i32 + 0];
    x32 |= (int)x16[2 * i32 + 1] << 16;
    return x32;
}

// One thread per (token, 32-value group). Byte-for-byte what the decode path's
// quantize_q8_0 writes for a single row: amax/127, __float2int_rn, and no sign rule --
// which is also why the warp-vs-serial distinction that makes Q8_K order-dependent does
// not arise here.
__global__ void prefill_quant_q8_kernel(BlockQ8_0* __restrict__ out,
                                        const float* __restrict__ x,
                                        int n_blocks, int n_tok) {
    k3_pdl_sync();
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)n_blocks * n_tok;
    if (i >= total) return;

    const int t = (int)(i / n_blocks);
    const int b = (int)(i % n_blocks);
    const float* xb = x + (size_t)t * ((size_t)n_blocks * 32) + (size_t)b * 32;

    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(xb[j]));
    const float d  = amax / 127.0f;
    const float id = amax != 0.0f ? 127.0f / amax : 0.0f;

    BlockQ8_0* o = out + i;
    o->d = __half_as_ushort(__float2half_rn(d));
#pragma unroll
    for (int j = 0; j < 32; ++j) o->qs[j] = (int8_t)__float2int_rn(xb[j] * id);
}

// Y[t][n] = sum_b dp4a(W[n][b], X[t][b]) * (dw * dx)
//
// BIT-IDENTICAL to proj_q8_multirow_1bar_kernel at the same BLOCK and ROWS, for every
// token independently. The three things that decide that are all preserved verbatim:
//   * b strides `threadIdx.x, += BLOCK`, so each (t, n) accumulator visits the same
//     blocks in the same order and lands the same partial in the same lane;
//   * the dp4a operand order is (weight, activation) and the contraction is
//     `(float)sumi * (dw * dx)` -- one multiply of the two scales, then one FMA, not
//     `sumi * dw * dx` which would round twice;
//   * the epilogue is the same shuffle butterfly followed by a fold over INCREASING w.
template <int BLOCK, int ROWS, int TOKS>
__global__ void prefill_proj_q8_kernel(float* __restrict__ y,
                                       const BlockQ8_0* __restrict__ x,
                                       const BlockQ8_0* __restrict__ W,
                                       int blocks_per_row, int n_rows, int n_tok,
                                       int64_t y_row_stride) {
    k3_pdl_sync();
    constexpr int NWARP = BLOCK / 32;
    const int n0   = blockIdx.x * ROWS;
    const int t0   = blockIdx.y * TOKS;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    __shared__ float shm[NWARP * ROWS * TOKS];

    float acc[ROWS][TOKS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int t = 0; t < TOKS; ++t) acc[r][t] = 0.0f;

    // The token tail duplicates the last valid row rather than branching inside the hot
    // loop: the surplus lanes compute a dot product that the epilogue never stores, and
    // every load stays in bounds. Same idiom the MLA decode kernel uses for its token
    // tail, and for the same reason -- a predicate here would sit on the dp4a chain.
    const int t_last = n_tok - 1;

    for (int b = threadIdx.x; b < blocks_per_row; b += BLOCK) {
        // TOKS activation blocks, staged once and reused by every row below.
        int   xq[TOKS][8];
        float dx[TOKS];
#pragma unroll
        for (int t = 0; t < TOKS; ++t) {
            const int tt = min(t0 + t, t_last);
            const BlockQ8_0* xr = x + (size_t)tt * blocks_per_row;
#pragma unroll
            for (int i = 0; i < 8; ++i) xq[t][i] = get_int_b2(xr[b].qs, i);
            dx[t] = __half2float(__ushort_as_half(xr[b].d));
        }

#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (n0 + r >= n_rows) continue;
            const BlockQ8_0* row = W + (size_t)(n0 + r) * blocks_per_row;
            // THE WEIGHT BLOCK IS READ ONCE HERE AND USED TOKS TIMES. This single hoist
            // is the entire point of the file: at TOKS=1 it degenerates to the decode
            // kernel's traffic, and the prefill win is exactly the factor TOKS.
            int wq[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) wq[i] = get_int_b2(row[b].qs, i);
            const float dw = __half2float(__ushort_as_half(row[b].d));

#pragma unroll
            for (int t = 0; t < TOKS; ++t) {
                int sumi = 0;
#pragma unroll
                for (int i = 0; i < 8; ++i) sumi = __dp4a(wq[i], xq[t][i], sumi);
                acc[r][t] += (float)sumi * (dw * dx[t]);
            }
        }
    }

    // Every (row, token) reduction is independent, so none of them need a barrier --
    // only separation from the shared write. One __syncthreads() for the whole tile.
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int t = 0; t < TOKS; ++t) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                acc[r][t] += __shfl_down_sync(0xffffffff, acc[r][t], off);
            if (lane == 0) shm[(warp * ROWS + r) * TOKS + t] = acc[r][t];
        }

    __syncthreads();

    // Thread (r, t) folds ITS OWN cell across the warps in increasing w -- block_sum's
    // order, which is what keeps the sum bit-identical to the single-token path.
    if ((int)threadIdx.x < ROWS * TOKS) {
        const int r  = (int)threadIdx.x / TOKS;
        const int t  = (int)threadIdx.x % TOKS;
        const int nn = n0 + r;
        const int tt = t0 + t;
        if (nn < n_rows && tt < n_tok) {
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < NWARP; ++w) s += shm[(w * ROWS + r) * TOKS + t];
            y[(int64_t)tt * y_row_stride + nn] = s;
        }
    }
}

// COPIED VERBATIM from k3_proj_q8_fast.cu's block_for, and it MUST stay verbatim.
//
// BLOCK is not a tuning knob here, it is part of the answer: it sets how many partials
// the warp butterfly folds and therefore the exact order the floats are summed in. A
// prefilled token and a decoded token agree only if both pick the same width for the
// same K. The first version of this function was written from memory as
// `nb >= 128 ? 128 : nb >= 64 ? 64 : 32`, which disagrees with the real tier at every
// shape between the boundaries -- nb = 48 (KDA attn_output, K = 1536) and nb = 112
// (ffn_routed_up, K = 3584) both landed one width too narrow. Same arithmetic, different
// reduction tree, silently different logits.
//
// The duplication is deliberate: the original lives in another translation unit's
// anonymous namespace. k3_prefill_block_for_matches_decode() in the CPU test pins the two
// against each other across every nb K3 can produce, so a future edit to either one fails
// a test rather than drifting the numerics.
inline int block_for(int blocks_per_row) {
    if (blocks_per_row <= 32) return 32;
    if (blocks_per_row <= 64) return 64;
    return 128;
}

}  // namespace

size_t k3_prefill_act_q8_bytes(int K, int n_tok) {
    if (K <= 0 || n_tok <= 0 || K % 32 != 0) return 0;
    return (size_t)(K / 32) * (size_t)n_tok * sizeof(BlockQ8_0);
}

bool k3_prefill_quantize_act(void* q8_out, const float* x, int K, int n_tok,
                             cudaStream_t stream) {
    if (!q8_out || !x || K <= 0 || n_tok <= 0 || K % 32 != 0) return false;
    const long long total = (long long)(K / 32) * n_tok;
    const int T = 256;
    const long long grid = (total + T - 1) / T;
    if (grid > 2147483647LL) return false;
    prefill_quant_q8_kernel<<<(unsigned)grid, T, 0, stream>>>(
        (BlockQ8_0*)q8_out, x, K / 32, n_tok);
    return cudaGetLastError() == cudaSuccess;
}

// ROWS x TOKS is bounded by registers, not by shared: the tile holds ROWS*TOKS
// accumulators plus TOKS*8 staged activation ints per thread. 4x4 measured as the
// balanced point on paper (56 live values before the compiler's own temporaries); the
// wider token tiles exist for the tall-N shapes where ROWS can shrink to pay for them.
int k3_prefill_proj_token_tile(int N, int K, int n_tok) {
    if (N <= 0 || K <= 0 || n_tok <= 0 || K % 32 != 0) return 0;
    if (n_tok >= 8) return 8;
    if (n_tok >= 4) return 4;
    if (n_tok >= 2) return 2;
    return 1;
}

bool k3_prefill_proj_q8act(float* y, const void* q8_acts, const void* W, int wtype,
                           int N, int K, int n_tok, cudaStream_t stream,
                           int64_t y_row_stride) {
    static const bool want = [] {
        const char* e = std::getenv("SPARKINFER_K3_PREFILL_PROJ");
        return !(e && e[0] == '0');
    }();
    if (!want) return false;
    if (!y || !q8_acts || !W || wtype != 8) return false;
    if (N <= 0 || K <= 0 || n_tok <= 0 || K % 32 != 0) return false;
    if (y_row_stride == 0) y_row_stride = N;
    if (y_row_stride < N) return false;

    const int nb   = K / 32;
    const int TB   = block_for(nb);
    const int toks = k3_prefill_proj_token_tile(N, K, n_tok);
    if (toks <= 0) return false;

    // ROWS pays for TOKS. The product is what costs registers, so the wider the token
    // tile the narrower the row tile -- and rows are the cheaper of the two to give up,
    // because the activation they amortise is L2-resident while the weight the tokens
    // amortise comes from HBM.
    const int rows = (toks >= 8) ? 2 : 4;

    const dim3 grid((unsigned)((N + rows - 1) / rows),
                    (unsigned)((n_tok + toks - 1) / toks));

#define K3_PF_LAUNCH(BS, R, TK)                                                       \
    prefill_proj_q8_kernel<BS, R, TK><<<grid, BS, 0, stream>>>(                       \
        y, (const BlockQ8_0*)q8_acts, (const BlockQ8_0*)W, nb, N, n_tok, y_row_stride)

    switch (TB * 100 + rows * 10 + (toks == 8 ? 8 : toks)) {
        case 32 * 100 + 4 * 10 + 1: K3_PF_LAUNCH(32,  4, 1); break;
        case 32 * 100 + 4 * 10 + 2: K3_PF_LAUNCH(32,  4, 2); break;
        case 32 * 100 + 4 * 10 + 4: K3_PF_LAUNCH(32,  4, 4); break;
        case 32 * 100 + 2 * 10 + 8: K3_PF_LAUNCH(32,  2, 8); break;
        case 64 * 100 + 4 * 10 + 1: K3_PF_LAUNCH(64,  4, 1); break;
        case 64 * 100 + 4 * 10 + 2: K3_PF_LAUNCH(64,  4, 2); break;
        case 64 * 100 + 4 * 10 + 4: K3_PF_LAUNCH(64,  4, 4); break;
        case 64 * 100 + 2 * 10 + 8: K3_PF_LAUNCH(64,  2, 8); break;
        case 128 * 100 + 4 * 10 + 1: K3_PF_LAUNCH(128, 4, 1); break;
        case 128 * 100 + 4 * 10 + 2: K3_PF_LAUNCH(128, 4, 2); break;
        case 128 * 100 + 4 * 10 + 4: K3_PF_LAUNCH(128, 4, 4); break;
        case 128 * 100 + 2 * 10 + 8: K3_PF_LAUNCH(128, 2, 8); break;
        default: return false;
    }
#undef K3_PF_LAUNCH

    return cudaGetLastError() == cudaSuccess;
}

}  // namespace k3
}  // namespace kernels
}  // namespace sparkinfer
