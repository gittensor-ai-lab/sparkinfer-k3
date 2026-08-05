// Token-batched Q8_0 projection — see sparkinfer/kernels/k3_batch_proj.h.
//
// A transpose of proj_q8_multirow_1bar_kernel, not a rewrite: everything that decides a
// float (block width, b-stride, dp4a argument order, accumulator expression, butterfly,
// single barrier, increasing-warp fold) is copied unchanged. Only the reuse axis moves.
//
// The helpers below mirror k3_proj_q8_fast.cu's anonymous namespace rather than being
// hoisted into a shared header, for the reason that file gives for its own duplication:
// the two arms must stay comparable on one binary.

#include "sparkinfer/kernels/k3_batch_proj.h"
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

// Four int8 weights as one packed int from a 2-byte-aligned source. Identical to
// k3_proj_q8_fast.cu's: BlockQ8_0 is 34 bytes with alignment 2, so qs is 4-byte
// aligned only for odd blocks and nvcc cannot legally widen a scalar loop.
__device__ __forceinline__ int get_int_b2(const int8_t* __restrict__ qs, int i32) {
    const uint16_t* x16 = (const uint16_t*)qs;
    int x32 = (int)x16[2 * i32 + 0];
    x32 |= (int)x16[2 * i32 + 1] << 16;
    return x32;
}

// THE BLOCK WIDTH IS THE BIT-IDENTITY CONTRACT, so it is derived by the same rule the
// single-token launcher uses and never tuned here. Changing it would change which
// quant blocks a thread accumulates and therefore the summation order.
inline int block_for(int blocks_per_row) {
    if (blocks_per_row <= 32) return 32;
    if (blocks_per_row <= 64) return 64;
    return 128;
}

// ---------------------------------------------------------------------------
// THE KERNEL
// ---------------------------------------------------------------------------
// One CTA owns ROWS output rows and all NT tokens. Per quant block b it reads the ROWS
// weight blocks once into registers and drives them against NT activations.
//
// Loop order matters and is not arbitrary: the WEIGHT is the outer staged operand
// because the weight is the DRAM traffic being amortised (27 MB per projection per
// layer) and the activation is the L2 traffic being spent (3.8 KB per token). Staging
// the activations instead would hold NT x 8 ints live across the row loop and buy
// nothing — the weight would then be re-read per token, which is the cost this whole
// file exists to remove.
template <int BLOCK, int ROWS, int NT>
__global__ void proj_q8_batch_1bar_kernel(float* __restrict__ y, long y_stride,
                                          const BlockQ8_0* __restrict__ x,
                                          const BlockQ8_0* __restrict__ W,
                                          int blocks_per_row, int n_rows, int n_tok) {
    k3_pdl_sync();   // no-op unless launched programmatically
    constexpr int NWARP = BLOCK / 32;
    constexpr int SLOTS = ROWS * NT;
    const int n0   = blockIdx.x * ROWS;
    const int lane = threadIdx.x & 31;
    const int tx   = (int)threadIdx.x;
    const int warp = tx >> 5;

    __shared__ float shm[NWARP * SLOTS];

    float acc[ROWS][NT];
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[r][t] = 0.0f;

    for (int b = tx; b < blocks_per_row; b += BLOCK) {
        // ONE weight block per row, staged once and reused by every token in this CTA.
        int   wq[ROWS][8];
        float dw[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (n0 + r >= n_rows) { dw[r] = 0.0f; continue; }
            const BlockQ8_0* row = W + (size_t)(n0 + r) * blocks_per_row;
#pragma unroll
            for (int i = 0; i < 8; ++i) wq[r][i] = get_int_b2(row[b].qs, i);
            dw[r] = __half2float(__ushort_as_half(row[b].d));
        }

#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t >= n_tok) continue;
            const BlockQ8_0* xt = x + (size_t)t * blocks_per_row;
            int xq[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) xq[i] = get_int_b2(xt[b].qs, i);
            const float dx = __half2float(__ushort_as_half(xt[b].d));

#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                if (n0 + r >= n_rows) continue;
                int sumi = 0;
                // SAME ARGUMENT ORDER as the single-token kernel — weight first,
                // activation second. dp4a is exact integer arithmetic so the order is
                // not numerically load-bearing, but keeping it identical is what makes
                // the two kernels diffable line for line.
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    sumi = __dp4a(wq[r][i], xq[i], sumi);
                acc[r][t] += (float)sumi * (dw[r] * dx);
            }
        }
    }

    // Every (row, token) slot's warp reduction, with no barrier between them. Same
    // butterfly over the same lanes block_sum uses, so each warp's partial is the same
    // float it would have produced there.
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                acc[r][t] += __shfl_down_sync(0xffffffff, acc[r][t], off);
            if (lane == 0) shm[warp * SLOTS + r * NT + t] = acc[r][t];
        }
    }

    __syncthreads();   // the only one

    // One thread per slot folds ITS (row, token) across the warps, in increasing w —
    // block_sum's order. SLOTS <= BLOCK is enforced by the launcher's table.
    if (tx < SLOTS) {
        const int r = tx / NT;
        const int t = tx % NT;
        if (n0 + r < n_rows && t < n_tok) {
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < NWARP; ++w) s += shm[w * SLOTS + r * NT + t];
            y[(long)t * y_stride + n0 + r] = s;
        }
    }
}

// ROWS for the batched kernel is a FREE parameter, unlike BLOCK.
//
// It decides only which output rows share a CTA — the accumulation order over b is set
// by BLOCK — so it can be chosen for registers and grid width without touching a
// single float. Two constraints bound it:
//   ROWS * NT <= BLOCK   the epilogue gives one thread to each slot
//   ROWS * NT accumulators + ROWS * 8 staged weight ints stay under the register file
// At ROWS 4 / NT 8 that is 32 accumulators and 32 staged ints; at ROWS 2 / NT 8, 16 and
// 16. SPARKINFER_K3_BATCH_ROWS overrides for the node sweep.
int batch_rows_for(int block_threads, int nt) {
    static const int env = [] {
        const char* e = std::getenv("SPARKINFER_K3_BATCH_ROWS");
        const int v = e ? atoi(e) : 0;
        return (v == 1 || v == 2 || v == 4) ? v : 0;
    }();
    int r = env ? env : 4;
    while (r > 1 && r * nt > block_threads) r >>= 1;
    return r;
}

}  // namespace

bool k3_batch_proj_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("SPARKINFER_K3_BATCH_PROJ");
        return !(e && e[0] == '0');
    }();
    return on;
}

int k3_batch_proj_max_nt() { return 8; }

void k3_quantize_q8_0_batch(void* out, const float* x, int K, int n_tok,
                            cudaStream_t stream) {
    // ONE call over n_tok * nb linear blocks. k3_quantize_q8_0's kernels index quant
    // blocks linearly and each block's scale is a max over its own 32 floats, so this
    // emits byte-for-byte what n_tok separate single-token calls emit — which is the
    // premise the header's bit-identity argument rests on. It also removes n_tok - 1
    // launches from every projection group, which at 69 KDA layers is not incidental.
    if (K <= 0 || n_tok <= 0 || (K % 32) != 0) return;
    k3_quantize_q8_0(out, x, (K / 32) * n_tok, stream);
}

bool k3_proj_q8_batch_1bar(float* y, long y_stride,
                           const void* xq, const void* W, int wtype,
                           int N, int K, int n_tok, cudaStream_t stream) {
    if (!k3_batch_proj_enabled()) return false;
    if (wtype != 8) return false;                       // Q8_0 only; caller falls back
    if (N <= 0 || K <= 0 || (K % 32) != 0) return false;
    if (!y || !xq || !W) return false;
    if (n_tok < 1 || n_tok > k3_batch_proj_max_nt()) return false;
    if (y_stride < N) return false;

    const int nb    = K / 32;
    const int BLOCK = block_for(nb);
    if (BLOCK != 32 && BLOCK != 64 && BLOCK != 128) return false;

    // Round the live token count UP to an instantiated NT and let the kernel's
    // `t >= n_tok` guards drop the slack. Rounding down would need two launches for a
    // tail chunk; the guarded slots cost their registers and nothing else, because the
    // continue is uniform across the CTA.
    const int NT = (n_tok <= 2) ? 2 : (n_tok <= 4) ? 4 : 8;
    const int ROWS = batch_rows_for(BLOCK, NT);
    if (ROWS * NT > BLOCK) return false;

    const unsigned grid = (unsigned)((N + ROWS - 1) / ROWS);
    const BlockQ8_0* xb = (const BlockQ8_0*)xq;
    const BlockQ8_0* wb = (const BlockQ8_0*)W;

#define K3_BATCH_LAUNCH(B, R, T)                                                   \
    k3_pdl_launch(dim3(grid), dim3(B), 0, stream,                                  \
                  proj_q8_batch_1bar_kernel<B, R, T>,                              \
                  y, y_stride, xb, wb, nb, N, n_tok)

    switch (BLOCK * 10000 + ROWS * 100 + NT) {
        case 32 * 10000 + 1 * 100 + 2: K3_BATCH_LAUNCH(32, 1, 2); break;
        case 32 * 10000 + 2 * 100 + 2: K3_BATCH_LAUNCH(32, 2, 2); break;
        case 32 * 10000 + 4 * 100 + 2: K3_BATCH_LAUNCH(32, 4, 2); break;
        case 32 * 10000 + 1 * 100 + 4: K3_BATCH_LAUNCH(32, 1, 4); break;
        case 32 * 10000 + 2 * 100 + 4: K3_BATCH_LAUNCH(32, 2, 4); break;
        case 32 * 10000 + 4 * 100 + 4: K3_BATCH_LAUNCH(32, 4, 4); break;
        case 32 * 10000 + 1 * 100 + 8: K3_BATCH_LAUNCH(32, 1, 8); break;
        case 32 * 10000 + 2 * 100 + 8: K3_BATCH_LAUNCH(32, 2, 8); break;
        case 32 * 10000 + 4 * 100 + 8: K3_BATCH_LAUNCH(32, 4, 8); break;

        case 64 * 10000 + 1 * 100 + 2: K3_BATCH_LAUNCH(64, 1, 2); break;
        case 64 * 10000 + 2 * 100 + 2: K3_BATCH_LAUNCH(64, 2, 2); break;
        case 64 * 10000 + 4 * 100 + 2: K3_BATCH_LAUNCH(64, 4, 2); break;
        case 64 * 10000 + 1 * 100 + 4: K3_BATCH_LAUNCH(64, 1, 4); break;
        case 64 * 10000 + 2 * 100 + 4: K3_BATCH_LAUNCH(64, 2, 4); break;
        case 64 * 10000 + 4 * 100 + 4: K3_BATCH_LAUNCH(64, 4, 4); break;
        case 64 * 10000 + 1 * 100 + 8: K3_BATCH_LAUNCH(64, 1, 8); break;
        case 64 * 10000 + 2 * 100 + 8: K3_BATCH_LAUNCH(64, 2, 8); break;
        case 64 * 10000 + 4 * 100 + 8: K3_BATCH_LAUNCH(64, 4, 8); break;

        case 128 * 10000 + 1 * 100 + 2: K3_BATCH_LAUNCH(128, 1, 2); break;
        case 128 * 10000 + 2 * 100 + 2: K3_BATCH_LAUNCH(128, 2, 2); break;
        case 128 * 10000 + 4 * 100 + 2: K3_BATCH_LAUNCH(128, 4, 2); break;
        case 128 * 10000 + 1 * 100 + 4: K3_BATCH_LAUNCH(128, 1, 4); break;
        case 128 * 10000 + 2 * 100 + 4: K3_BATCH_LAUNCH(128, 2, 4); break;
        case 128 * 10000 + 4 * 100 + 4: K3_BATCH_LAUNCH(128, 4, 4); break;
        case 128 * 10000 + 1 * 100 + 8: K3_BATCH_LAUNCH(128, 1, 8); break;
        case 128 * 10000 + 2 * 100 + 8: K3_BATCH_LAUNCH(128, 2, 8); break;
        case 128 * 10000 + 4 * 100 + 8: K3_BATCH_LAUNCH(128, 4, 8); break;
        default: return false;
    }
#undef K3_BATCH_LAUNCH
    return true;
}

}  // namespace k3
}  // namespace kernels
}  // namespace sparkinfer
