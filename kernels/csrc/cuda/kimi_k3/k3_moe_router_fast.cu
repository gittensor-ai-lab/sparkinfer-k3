// Factor — MoE router: keep the selection in shared memory.
//
// ===========================================================================
// WHAT THE ORIGINAL DOES, AND WHERE THE TIME GOES
// ===========================================================================
// moe_router_noaux_tc_kernel picks top_k = 16 of K3's 896 experts, and at decode it is
// ONE BLOCK — grid is n_tokens, and n_tokens is 1. It runs 92 times per token per rank
// and measures 17.8 us a call at K3's shape, which is 1.6 ms of a ~30 ms token for a
// top-16 of 896 floats.
//
// Three things cost, and none of them is the arithmetic:
//
//   1. THE WINNER IS PUBLISHED THROUGH GLOBAL MEMORY. Each of the 16 passes ends with
//      `ids[k] = mi; w[k] = s_p[mi];` writing straight to the caller's global buffers.
//      Then the normalisation re-READS all 16 back — `for k: sum += w[k]` — from a
//      single thread, as sixteen DEPENDENT global loads. On Hopper that alone is
//      thousands of cycles of pure latency at the end of every router call.
//   2. THE CROSS-WARP REDUCTION IS A SERIAL LOOP. After each pass's warp reduce, thread
//      0 alone walks the BLOCK/32 per-warp candidates while 255 threads wait at the
//      barrier that follows.
//   3. Both happen 16 times, sequentially, because pass k+1 cannot start until pass k
//      has masked its winner out of s_sel.
//
// This changes only 1 and 2. The 16 passes stay: they are what makes the selection
// deterministic, and the previous attempt to collapse them into a single warp pass was
// measured at +0.57 ms and dropped.
//
// ===========================================================================
// WHY IT IS BIT-IDENTICAL, INCLUDING THE TIE-BREAK
// ===========================================================================
// The selection compares (value descending, index ascending) — `v > bv || (v == bv &&
// e < bi)`. That is a TOTAL ORDER on the candidates, so "take the maximum" under it is
// associative and commutative, and the result does not depend on the order the
// candidates are folded in. Replacing thread 0's serial walk over the per-warp winners
// with a warp reduce over the same set therefore returns the same (value, index) pair,
// not merely an equivalent one.
//
// Everything else is arithmetic-for-arithmetic identical: the same sigmoid, the same
// bias add, the same masking with -INFINITY, the same normalisation with the same
// 6.103515625e-5f floor (the smallest normal f16, per the reference), the same w_scale.
// The only difference is that w[] and ids[] live in shared until the end and are written
// to global once.
//
// ===========================================================================
// WHAT IT IS NOT
// ===========================================================================
// NOT a different selection algorithm — same 16 sequential exact argmax passes.
// NOT a widening of a single-block launch in the sense #71 proposes for rms_norm: the
// block count is unchanged at 1, because the passes are inherently sequential; what
// changes is where the intermediate results live.
// NOT related to expert SHARDING (#96, merged) — that decides which experts a rank
// owns; this decides nothing about placement and runs before any of it.
// NOT a change to any caller: same entry point, same signature, same outputs.

#include "sparkinfer/kernels/kimi_k3.h"
#include "sparkinfer/kernels/kimi_k3_fast.h"
#include "k3_pdl.cuh"

#include <cuda_runtime.h>

#include <cstdlib>

namespace sparkinfer {
namespace kernels {
namespace k3 {
namespace {

constexpr int kRouterMaxTopK = 32;   // K3 uses 16; the shared arrays are sized for this

__device__ __forceinline__ void router_better(float ov, int oi, float& bv, int& bi) {
    // The original's comparison, verbatim: larger value wins; on an exact tie the
    // SMALLER expert index wins. oi >= 0 guards the -1 that an empty lane carries.
    if (ov > bv || (ov == bv && oi >= 0 && oi < bi)) { bv = ov; bi = oi; }
}

template <int BLOCK>
__global__ void moe_router_shared_kernel(float* __restrict__ out_w,
                                         int* __restrict__ out_ids,
                                         const float* __restrict__ logits,
                                         const float* __restrict__ bias,
                                         int n_expert, int top_k,
                                         bool norm_w, float w_scale) {
    k3_pdl_sync();
    const int tok = blockIdx.x;
    const float* lg = logits + (size_t)tok * n_expert;
    float* w   = out_w   + (size_t)tok * top_k;
    int*   ids = out_ids + (size_t)tok * top_k;

    extern __shared__ float smem_rf[];
    float* s_sel = smem_rf;                  // biased scores, mutated during selection
    float* s_p   = s_sel + n_expert;         // UNBIASED probs, never mutated
    float* s_w   = s_p + n_expert;           // THE CHANGE: the winners stay here...
    int*   s_ids = (int*)(s_w + kRouterMaxTopK);   // ...and so do their indices
    __shared__ float s_bestv[BLOCK / 32];
    __shared__ int   s_besti[BLOCK / 32];

    for (int e = threadIdx.x; e < n_expert; e += BLOCK) {
        const float p = 1.0f / (1.0f + __expf(-lg[e]));   // sigmoid
        s_p[e]   = p;
        s_sel[e] = bias ? (p + bias[e]) : p;              // bias: selection only
    }
    __syncthreads();

    constexpr int NWARP = BLOCK / 32;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;

    for (int k = 0; k < top_k; ++k) {
        float bv = -INFINITY; int bi = -1;
        for (int e = threadIdx.x; e < n_expert; e += BLOCK)
            router_better(s_sel[e], e, bv, bi);

        // Butterfly, not shfl_down: every lane ends with the warp's winner, so no
        // broadcast is needed and the warp's own reduction is one fewer step of
        // bookkeeping. Order-independent by the total order above.
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            router_better(__shfl_xor_sync(0xffffffff, bv, off),
                          __shfl_xor_sync(0xffffffff, bi, off), bv, bi);
        if (lane == 0) { s_bestv[warp] = bv; s_besti[warp] = bi; }
        __syncthreads();

        // The cross-warp fold, by ONE WARP instead of one thread. NWARP is 8 at
        // BLOCK 256, so this is three shuffle steps against a serial loop of eight
        // that 255 threads were waiting on.
        if (warp == 0) {
            float mv = (lane < NWARP) ? s_bestv[lane] : -INFINITY;
            int   mi = (lane < NWARP) ? s_besti[lane] : -1;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                router_better(__shfl_xor_sync(0xffffffff, mv, off),
                              __shfl_xor_sync(0xffffffff, mi, off), mv, mi);
            if (lane == 0) {
                s_ids[k] = mi;
                s_w[k]   = s_p[mi];        // UNBIASED prob — the whole point
                s_sel[mi] = -INFINITY;     // remove from further rounds
            }
        }
        __syncthreads();
    }

    // The normalisation reads SHARED, not the sixteen dependent global loads the
    // original paid here. Still one thread: it is 16 values, and a reduction would cost
    // more in barriers than it saves.
    if (threadIdx.x == 0) {
        if (norm_w) {
            float sum = 0.0f;
            for (int k = 0; k < top_k; ++k) sum += s_w[k];
            // smallest normal F16, per the reference — not an arbitrary epsilon
            sum = fmaxf(sum, 6.103515625e-5f);
            for (int k = 0; k < top_k; ++k) s_w[k] /= sum;
        }
        if (w_scale != 0.0f && w_scale != 1.0f)
            for (int k = 0; k < top_k; ++k) s_w[k] *= w_scale;
    }
    __syncthreads();

    // ONE coalesced write of each output, at the end.
    for (int k = threadIdx.x; k < top_k; k += BLOCK) { w[k] = s_w[k]; ids[k] = s_ids[k]; }
}

}  // namespace

bool k3_moe_router_fast(float* out_w, int* out_ids, const float* logits,
                        const float* bias, int n_expert, int top_k, int n_tokens,
                        bool norm_w, float w_scale, cudaStream_t stream) {
    static const bool want = [] {
        const char* e = std::getenv("SPARKINFER_K3_ROUTER_FAST");
        return !(e && e[0] == '0');
    }();
    if (!want) return false;
    if (!out_w || !out_ids || !logits) return false;
    if (n_expert <= 0 || top_k <= 0 || n_tokens <= 0) return false;
    if (top_k > kRouterMaxTopK) return false;

    constexpr int BLOCK = 256;   // the original's, and what the sweep picked
    // s_sel + s_p over the experts, plus the top_k winners and their indices.
    const size_t shm = ((size_t)2 * n_expert + kRouterMaxTopK) * sizeof(float) +
                       (size_t)kRouterMaxTopK * sizeof(int);
    if (shm > 48u * 1024u) return false;   // decline rather than fail a launch

    k3_pdl_launch((unsigned)n_tokens, BLOCK, shm, stream,
                  moe_router_shared_kernel<BLOCK>,
                  out_w, out_ids, logits, bias, n_expert, top_k, norm_w, w_scale);
    return true;
}

}  // namespace k3
}  // namespace kernels
}  // namespace sparkinfer
