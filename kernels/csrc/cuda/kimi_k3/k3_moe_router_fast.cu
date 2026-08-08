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
            // START AT NWARP/2, NOT 16. Only lanes [0, NWARP) carry a candidate; the
            // rest hold (-INFINITY, -1), against which router_better is a PROVABLE
            // no-op — `ov > bv` is false for -INFINITY, and the tie arm is gated by
            // `oi >= 0`, which -1 fails. At NWARP=8 the old bound ran five rounds
            // where three suffice, so two of every five dependent shuffle rounds were
            // dead. This loop is the kernel's critical path (gridDim is 1 for decode,
            // so nothing else is resident to hide it) and it runs top_k=16 times per
            // call, 92 times per token. The comment above already claimed three steps;
            // the code did five.
            static_assert(NWARP > 0 && (NWARP & (NWARP - 1)) == 0,
                          "butterfly fold needs a power-of-two warp count");
#pragma unroll
            for (int off = NWARP / 2; off > 0; off >>= 1)
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

// ===========================================================================
// THE DECODE SPECIALISATION
// ===========================================================================
// Same algorithm, same 16 sequential exact-argmax passes, same total order. Three
// things are compile-time here that are runtime above, and one array moves:
//
//   NV   — the per-thread candidate count, so the biased scores live in REGISTERS
//          instead of being re-read from shared on every one of the top_k passes.
//          The general kernel re-reads s_sel[e] each pass: ceil(896/256)=4 loads x
//          256 threads x 16 passes = 14,336 dependent shared loads per call, on a
//          kernel whose runtime IS its dependence chain. s_sel then disappears from
//          shared entirely (-3584 B).
//   TOPK — so nvcc can unroll the two normalisation loops and the masking test.
//          As a runtime argument it could unroll none of them.
//   n_tokens == 1 — which is what makes __launch_bounds__(BLOCK, 1) correct rather
//          than merely harmless: at gridDim 1 there is exactly one CTA on one SM,
//          so occupancy is worth nothing and capping registers to buy it is pure
//          loss. nvcc cannot know that on its own.
//
// BIT-IDENTICAL, by the same total-order argument as the general kernel:
//   * the initial scan visits the same experts in the same ascending order;
//   * masking a winner sets the SAME candidate to -INFINITY, just in the register
//     that holds it instead of in shared — the owner of expert e is thread
//     e % BLOCK at slot e / BLOCK, and the test is unrolled over compile-time
//     slots so `v` never spills to local memory (a runtime index would);
//   * the normalisation keeps the serial ascending FADD chain, the same
//     6.103515625e-5f floor, and the same per-element divide. Unrolling a chain of
//     dependent += does not reassociate it.
template <int BLOCK, int NV, int TOPK>
__global__ __launch_bounds__(BLOCK, 1)
void moe_router_reg_kernel(float* __restrict__ out_w,
                           int* __restrict__ out_ids,
                           const float* __restrict__ logits,
                           const float* __restrict__ bias,
                           int n_expert, bool norm_w, float w_scale) {
    k3_pdl_sync();
    const float* lg = logits;            // n_tokens == 1, so tok is 0
    extern __shared__ float smem_rr[];
    float* s_p   = smem_rr;              // UNBIASED probs, never mutated
    float* s_w   = s_p + n_expert;
    int*   s_ids = (int*)(s_w + TOPK);
    __shared__ float s_bestv[BLOCK / 32];
    __shared__ int   s_besti[BLOCK / 32];

    constexpr int NWARP = BLOCK / 32;
    static_assert(NWARP > 0 && (NWARP & (NWARP - 1)) == 0,
                  "butterfly fold needs a power-of-two warp count");
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;

    // The biased scores, held for the whole kernel. THIS is the change.
    float v[NV];
#pragma unroll
    for (int i = 0; i < NV; ++i) {
        const int e = (int)threadIdx.x + i * BLOCK;
        if (e < n_expert) {
            const float p = 1.0f / (1.0f + __expf(-lg[e]));   // sigmoid
            s_p[e] = p;
            v[i] = bias ? (p + bias[e]) : p;                  // bias: selection only
        } else {
            v[i] = -INFINITY;
        }
    }
    __syncthreads();

    for (int k = 0; k < TOPK; ++k) {
        float bv = -INFINITY; int bi = -1;
#pragma unroll
        for (int i = 0; i < NV; ++i) {
            const int e = (int)threadIdx.x + i * BLOCK;
            if (e < n_expert) router_better(v[i], e, bv, bi);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)     // full warp: every lane is live
            router_better(__shfl_xor_sync(0xffffffff, bv, off),
                          __shfl_xor_sync(0xffffffff, bi, off), bv, bi);
        if (lane == 0) { s_bestv[warp] = bv; s_besti[warp] = bi; }
        __syncthreads();

        if (warp == 0) {
            float mv = (lane < NWARP) ? s_bestv[lane] : -INFINITY;
            int   mi = (lane < NWARP) ? s_besti[lane] : -1;
#pragma unroll
            for (int off = NWARP / 2; off > 0; off >>= 1)   // see the general kernel
                router_better(__shfl_xor_sync(0xffffffff, mv, off),
                              __shfl_xor_sync(0xffffffff, mi, off), mv, mi);
            if (lane == 0) { s_ids[k] = mi; s_w[k] = s_p[mi]; }
        }
        __syncthreads();

        // Mask the winner out of the OWNER's register. Unrolled over compile-time
        // slots: a runtime `v[slot]` would push the whole array to local memory and
        // cost far more than the shared read it replaced.
        const int won = s_ids[k];
#pragma unroll
        for (int i = 0; i < NV; ++i)
            if (won == (int)threadIdx.x + i * BLOCK) v[i] = -INFINITY;
    }

    if (threadIdx.x == 0) {
        if (norm_w) {
            float sum = 0.0f;
#pragma unroll
            for (int k = 0; k < TOPK; ++k) sum += s_w[k];     // ascending, unchanged
            sum = fmaxf(sum, 6.103515625e-5f);
#pragma unroll
            for (int k = 0; k < TOPK; ++k) s_w[k] /= sum;
        }
        if (w_scale != 0.0f && w_scale != 1.0f) {
#pragma unroll
            for (int k = 0; k < TOPK; ++k) s_w[k] *= w_scale;
        }
    }
    __syncthreads();

    for (int k = threadIdx.x; k < TOPK; k += BLOCK) { out_w[k] = s_w[k]; out_ids[k] = s_ids[k]; }
}

// ===========================================================================
// THE TWO-PHASE DECODE SPECIALISATION
// ===========================================================================
// The register kernel above pays TWO block-wide barriers per selected expert —
// 32 __syncthreads on a single-CTA kernel whose runtime is its dependence chain,
// with seven of eight warps parked at every one of them. This form replaces the
// per-pass cross-warp rendezvous with the standard top-k decomposition:
//
//   phase A — each warp extracts the top-TOPK of ITS OWN candidates using only
//             warp shuffles: zero block barriers, warps run fully independent;
//   phase B — warp 0 merges the NWARP x TOPK survivors, again shuffle-only.
//
// Three __syncthreads total (survivor publish, epilogue in, epilogue out).
//
// BIT-IDENTICAL, and the argument is the same total order as always:
//   * (value desc, index asc) is a total order, so each warp's k-th extraction is
//     exactly its rank-k candidate, and a warp whose candidate belongs to the
//     global top-TOPK necessarily holds it in its own top-TOPK — the union of the
//     per-warp lists contains the global list;
//   * phase B extracts from that union in rank order, so its k-th winner IS the
//     global rank-k expert — the same id sequence the 16 global passes produce;
//   * s_w[k] = s_p[winner], the normalisation chain, the floor and the scale are
//     copied unchanged from the kernel above.
template <int BLOCK, int NV, int TOPK>
__global__ __launch_bounds__(BLOCK, 1)
void moe_router_2p_kernel(float* __restrict__ out_w,
                          int* __restrict__ out_ids,
                          const float* __restrict__ logits,
                          const float* __restrict__ bias,
                          int n_expert, bool norm_w, float w_scale) {
    k3_pdl_sync();
    const float* lg = logits;
    extern __shared__ float smem_rr[];
    float* s_p   = smem_rr;              // UNBIASED probs, never mutated
    float* s_w   = s_p + n_expert;
    int*   s_ids = (int*)(s_w + TOPK);

    constexpr int NWARP = BLOCK / 32;
    static_assert(NWARP > 0 && NWARP <= 32, "phase B is one warp over NWARP lists");
    __shared__ float s_pv[NWARP][TOPK];
    __shared__ int   s_pi[NWARP][TOPK];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;

    float v[NV];
#pragma unroll
    for (int i = 0; i < NV; ++i) {
        const int e = (int)threadIdx.x + i * BLOCK;
        if (e < n_expert) {
            const float p = 1.0f / (1.0f + __expf(-lg[e]));   // sigmoid
            s_p[e] = p;
            v[i] = bias ? (p + bias[e]) : p;                  // bias: selection only
        } else {
            v[i] = -INFINITY;
        }
    }

    // Phase A: this warp's top-TOPK of its own NV*32 candidates. No barriers —
    // every step is warp-synchronous.
    for (int k = 0; k < TOPK; ++k) {
        float bv = -INFINITY; int bi = -1;
#pragma unroll
        for (int i = 0; i < NV; ++i) {
            const int e = (int)threadIdx.x + i * BLOCK;
            if (e < n_expert) router_better(v[i], e, bv, bi);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            router_better(__shfl_xor_sync(0xffffffff, bv, off),
                          __shfl_xor_sync(0xffffffff, bi, off), bv, bi);
        // Every lane already holds the warp winner after the butterfly; lane 0's
        // copy is canonical. Publish, then the OWNER lane masks it.
        if (lane == 0) { s_pv[warp][k] = bv; s_pi[warp][k] = bi; }
        const int won = __shfl_sync(0xffffffff, bi, 0);
#pragma unroll
        for (int i = 0; i < NV; ++i)
            if (won == (int)threadIdx.x + i * BLOCK) v[i] = -INFINITY;
    }
    __syncthreads();                     // survivors visible to warp 0

    // Phase B: warp 0 merges NWARP lists of TOPK. Lane l holds survivors
    // l, l+32, l+64, ... — survivor s is s_p{v,i}[s / TOPK][s % TOPK].
    if (warp == 0) {
        constexpr int NSURV = NWARP * TOPK;
        constexpr int SLOTS = (NSURV + 31) / 32;
        float mv[SLOTS]; int mi[SLOTS];
#pragma unroll
        for (int j = 0; j < SLOTS; ++j) {
            const int s = lane + 32 * j;
            if (s < NSURV) { mv[j] = s_pv[s / TOPK][s % TOPK]; mi[j] = s_pi[s / TOPK][s % TOPK]; }
            else           { mv[j] = -INFINITY;                mi[j] = -1; }
        }
        for (int k = 0; k < TOPK; ++k) {
            float bv = -INFINITY; int bi = -1;
#pragma unroll
            for (int j = 0; j < SLOTS; ++j) router_better(mv[j], mi[j], bv, bi);
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                router_better(__shfl_xor_sync(0xffffffff, bv, off),
                              __shfl_xor_sync(0xffffffff, bi, off), bv, bi);
            if (lane == 0) { s_ids[k] = bi; s_w[k] = s_p[bi]; }
            const int won = __shfl_sync(0xffffffff, bi, 0);
#pragma unroll
            for (int j = 0; j < SLOTS; ++j)
                if (mi[j] == won) mv[j] = -INFINITY;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        if (norm_w) {
            float sum = 0.0f;
#pragma unroll
            for (int k = 0; k < TOPK; ++k) sum += s_w[k];     // ascending, unchanged
            sum = fmaxf(sum, 6.103515625e-5f);
#pragma unroll
            for (int k = 0; k < TOPK; ++k) s_w[k] /= sum;
        }
        if (w_scale != 0.0f && w_scale != 1.0f) {
#pragma unroll
            for (int k = 0; k < TOPK; ++k) s_w[k] *= w_scale;
        }
    }
    __syncthreads();

    for (int k = threadIdx.x; k < TOPK; k += BLOCK) { out_w[k] = s_w[k]; out_ids[k] = s_ids[k]; }
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

    // The decode specialisation, taken only when every assumption it is built on
    // holds. SPARKINFER_K3_ROUTER_REG=0 restores the general kernel on the SAME
    // binary, so the A/B needs no second build.
    //   n_tokens == 1     -> gridDim 1, which is what justifies __launch_bounds__(,1)
    //   top_k   == kTopK  -> the unrolled normalisation
    //   n_expert <= NV*BLOCK -> every candidate fits in registers
    // K3 decode is n_expert 896, top_k 16, n_tokens 1, so it qualifies; anything
    // else falls through to the general kernel unchanged.
    constexpr int kNV = 4, kTopK = 16;
    static const bool want_reg = [] {
        const char* e = std::getenv("SPARKINFER_K3_ROUTER_REG");
        return !(e && e[0] == '0');
    }();
    // The two-phase form drops the register kernel's 32 per-pass block barriers to
    // three. Same guards, same shapes; =0 keeps the per-pass-rendezvous kernel.
    static const bool want_2p = [] {
        const char* e = std::getenv("SPARKINFER_K3_ROUTER_2P");
        return e && e[0] == '1';
    }();
    if (want_reg && n_tokens == 1 && top_k == kTopK && n_expert <= kNV * BLOCK) {
        // No s_sel: the biased scores live in registers.
        const size_t shm = ((size_t)n_expert + kTopK) * sizeof(float) +
                           (size_t)kTopK * sizeof(int);
        if (shm <= 48u * 1024u) {
            if (want_2p)
                k3_pdl_launch(1u, BLOCK, shm, stream,
                              moe_router_2p_kernel<BLOCK, kNV, kTopK>,
                              out_w, out_ids, logits, bias, n_expert, norm_w, w_scale);
            else
                k3_pdl_launch(1u, BLOCK, shm, stream,
                              moe_router_reg_kernel<BLOCK, kNV, kTopK>,
                              out_w, out_ids, logits, bias, n_expert, norm_w, w_scale);
            return true;
        }
    }

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
