// KDA decode step — one WARP per state column, with the i-axis split across its lanes.
//
// ---------------------------------------------------------------------------
// WHAT IS WRONG WITH THE KERNEL THIS REPLACES
// ---------------------------------------------------------------------------
// kda_decode_step_vt_kernel (k3_kernels.cu) gives one THREAD to each state column and
// tiles the columns BV=32 at a time, so its launch is (n_head, head_dim/BV) blocks of
// 32 threads. #63 head-sharded KDA, taking n_head from 96 to 12 at tp=8, so at the
// scored shape that is:
//
//     grid (12, 4) x 32 threads = 48 warps, on a part with 132 SMs x 64 warp slots
//
// 0.57% occupancy, and 84 SMs with nothing to do. The kernel moves 2.25 MiB per
// launch (read S, re-read S, write S) and runs 69 times per token; measured against
// #77's own +6.7% that is ~2.5-3.0 ms per token per rank at roughly 55-65 GB/s on a
// 4.8 TB/s part — a 75-90x gap to its own bandwidth floor.
//
// THE PARALLELISM IS BOUNDED BY THE COLUMN COUNT, AND NO CHOICE OF BV CHANGES THAT.
// One thread per column means the whole launch has n_head * head_dim = 12 * 128 =
// 1536 threads. BV moves those threads between blocks; it cannot create more of them.
// The work has to be divided along i or it does not divide.
//
// ---------------------------------------------------------------------------
// WHY DIVIDING ALONG i IS LEGAL
// ---------------------------------------------------------------------------
// The recurrence runs over TOKENS, not over i. Within one step, for state column j:
//
//     sk[j] = sum_i S[i][j] * exp(g[i]) * k[i]          <- a plain reduction over i
//     d[j]  = beta * (v[j] - sk[j])                     <- column-local scalar
//     S'[i][j] = S[i][j] * exp(g[i]) + k[i] * d[j]      <- elementwise in i
//     o[j]  = sum_i S'[i][j] * q[i]                     <- a plain reduction over i
//
// so a warp can own column j, spread i across its 32 lanes, and pay exactly two
// shuffle reductions. The stored layout is untouched: S[i][j] lives at s[j*D+i], the
// contract kimi_k3_numeric_test documents, which is also what makes i the FAST axis
// and the vectorisation below possible.
//
// ---------------------------------------------------------------------------
// WHAT IT BUYS BEYOND THE 32x IN THREADS
// ---------------------------------------------------------------------------
//   1. THE STATE IS READ ONCE, NOT TWICE. The tiled kernel stages its slab of S into
//      shared for pass 1, drops it, and reads the same rows from global again in
//      pass 2. A lane's share of a column is four floats, so it simply keeps them in
//      registers across both passes: global traffic per launch goes 2R + W -> R + W,
//      a third less, on top of the occupancy.
//   2. ONE 512-BYTE TRANSACTION PER WARP PER ARRAY. Lane l takes i = 4l..4l+3 as a
//      single float4, so a warp's read of a column is exactly one aligned 512-byte
//      line. k, q and g are read the same way. The tiled kernel issues 384 scalar
//      memory instructions per block instead.
//   3. NO SHARED MEMORY AND NO __syncthreads. Everything a column needs is inside its
//      own warp, so the block is a container rather than a synchronisation domain,
//      and 31 registers per thread leaves the SM free to hold the maximum.
//
// ---------------------------------------------------------------------------
// WHAT IT COSTS
// ---------------------------------------------------------------------------
// The two sums become butterfly trees instead of sequential chains, so this is NOT
// bit-identical — the same reassociation the expert all-reduce, the split MLA combine
// and #74's peer-oneshot collective already make. It is bounded at ~1 ulp per column
// and it cannot accumulate across tokens: the recurrence is contracting, every step
// multiplies the state by exp(g) < 1.
//
// SPARKINFER_K3_KDA_IP=0 declines here and leaves the tiled kernel exactly as main
// runs it, so both arms live in one binary.

#include "sparkinfer/kernels/kimi_k3_fast.h"
#include "k3_pdl.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>

namespace sparkinfer {
namespace kernels {
namespace k3 {

namespace {

// CPB = columns (warps) per block. VPL = float4s per lane = head_dim / 128.
//
// DEFER moves the recurrent-state write-back BELOW the output write and signals
// programmatic completion between them. See the note above the write for why that is
// sound here and is not sound for most kernels.
template <int CPB, int VPL, bool DEFER>
__global__ void kda_step_ip_kernel(float* __restrict__ out,
                                   float* __restrict__ state,
                                   const float* __restrict__ q,
                                   const float* __restrict__ k,
                                   const float* __restrict__ v,
                                   const float* __restrict__ g,
                                   const float* __restrict__ beta,
                                   int head_dim, bool beta_sigmoid) {
    k3_pdl_sync();   // no-op unless launched programmatically
    const int h    = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int j    = blockIdx.y * CPB + warp;
    // UNDER DEFER THIS RETURN MUST BE UNREACHABLE, and the launcher is what guarantees
    // it: the programmatic-completion port fires when every BLOCK has triggered or
    // terminated, so a block where some warps exited before the trigger and others had
    // not yet reached it is exactly the race this whole factor must not introduce.
    // Rather than rewrite the return as a predicate and pay for it on every shape,
    // k3_kda_decode_step_ip declines DEFER unless head_dim % CPB == 0 — at which point
    // j < head_dim for every warp of every block and this line is dead code.
    if (j >= head_dim) return;

    // Row j of this head's state. i is the fast axis, so the warp's float4 reads are
    // one contiguous 512-byte line. beta and v are per-column scalars from L1.
    float4* __restrict__ Srow =
        (float4*)(state + ((size_t)h * head_dim + j) * head_dim);
    const float4* __restrict__ k4 = (const float4*)(k + (size_t)h * head_dim);
    const float4* __restrict__ q4 = (const float4*)(q + (size_t)h * head_dim);
    const float4* __restrict__ g4 = (const float4*)(g + (size_t)h * head_dim);
    // The standalone sigmoid_inplace launch is skipped when this is set; the
    // expression is copied verbatim from sigmoid_inplace_f32_kernel so the value the
    // recurrence sees is the same float either way.
    const float braw = beta[h];
    const float b    = beta_sigmoid ? (1.0f / (1.0f + __expf(-braw))) : braw;
    const float v_j = v[(size_t)h * head_dim + j];

    float4 sv[VPL], kv[VPL], qv[VPL], ge[VPL];

    // --- pass 1: decay, and sk = sum_i S[i][j] * exp(g[i]) * k[i] ---
    //
    // __fmul_rn, not `*`, for the reason the tiled kernel gives: the decayed value is
    // reused in pass 2, so it has to be a materialised, rounded f32 in both places.
    // Left as `*`, nvcc (--fmad=true) contracts it into the surrounding fma in one
    // pass and not the other, and the state and the output would then disagree about
    // what S*exp(g) is.
    float sk = 0.0f;
#pragma unroll
    for (int u = 0; u < VPL; ++u) {
        const int idx = u * 32 + lane;
        sv[u] = Srow[idx];
        kv[u] = k4[idx];
        qv[u] = q4[idx];
        const float4 gr = g4[idx];
        ge[u] = make_float4(__expf(gr.x), __expf(gr.y), __expf(gr.z), __expf(gr.w));
        sk += __fmul_rn(sv[u].x, ge[u].x) * kv[u].x;
        sk += __fmul_rn(sv[u].y, ge[u].y) * kv[u].y;
        sk += __fmul_rn(sv[u].z, ge[u].z) * kv[u].z;
        sk += __fmul_rn(sv[u].w, ge[u].w) * kv[u].w;
    }
    // Butterfly rather than shfl_down plus a broadcast: every lane needs d_j, and xor
    // leaves the total in all 32 of them for the same five instructions.
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sk += __shfl_xor_sync(0xffffffff, sk, off);

    const float d_j = b * (v_j - sk);

    // --- pass 2: rank-1 update in place, and o = sum_i S'[i][j] * q[i] ---
    float o = 0.0f;
    float4 s2v[VPL];
#pragma unroll
    for (int u = 0; u < VPL; ++u) {
        float4 s2;
        s2.x = __fmul_rn(sv[u].x, ge[u].x) + kv[u].x * d_j;
        s2.y = __fmul_rn(sv[u].y, ge[u].y) + kv[u].y * d_j;
        s2.z = __fmul_rn(sv[u].z, ge[u].z) + kv[u].z * d_j;
        s2.w = __fmul_rn(sv[u].w, ge[u].w) + kv[u].w * d_j;
        // Same value, same rounding, either way — only WHEN it reaches memory moves.
        if constexpr (DEFER) s2v[u] = s2; else Srow[u * 32 + lane] = s2;
        o += s2.x * qv[u].x;
        o += s2.y * qv[u].y;
        o += s2.z * qv[u].z;
        o += s2.w * qv[u].w;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) o += __shfl_xor_sync(0xffffffff, o, off);

    if (lane == 0) out[(size_t)h * head_dim + j] = o;

    if constexpr (DEFER) {
        // ---------------------------------------------------------------------
        // THE ONE PLACE IN THIS DECODE WHERE EARLY COMPLETION IS PROVABLY SOUND
        // ---------------------------------------------------------------------
        // This kernel writes two things, and they are wildly different sizes and have
        // wildly different readers:
        //
        //   out    n_head * head_dim floats     6 KiB    read by the very next kernel
        //   state  n_head * head_dim^2 floats 768 KiB    read by NOTHING until the
        //                                                SAME kernel, one token later
        //
        // So 99% of the store traffic is not on any successor's critical path, and
        // above it was being issued BEFORE the 6 KiB the successor is actually waiting
        // for. Swapping the order and signalling completion in between hands the whole
        // state write-back to the successor as overlap.
        //
        // WHY THE DEFERRED WRITE CANNOT RACE, which is the question that killed the
        // same idea on the TP collective. There, the deferred-past buffer was read by
        // the seven PEER ranks inside the same token, and a programmatic edge is
        // pairwise — releasing one successor releases the whole chain behind it, so
        // nothing downstream re-established the order. Here the deferred buffer is
        // st.delta_state, which is rank-private and is touched by exactly two call
        // sites in the entire forward: this kernel and the tiled kernel it replaces
        // (kimi_k3.cpp:1350 and :1357), plus the zero-fill at reset. The next read is
        // the next token's launch, and tokens are separated by a graph boundary and a
        // stream synchronise. Unbounded run-ahead inside this token therefore cannot
        // reach it — there is no reader to reach.
        //
        // The fence publishes `out`, not the state: the state is written after, and no
        // one is waiting for it. __syncthreads() is what makes the trigger a whole-block
        // event, since the port needs every BLOCK and a block that triggered from one
        // warp while another was still storing `out` would be a torn read.
        __threadfence();
        __syncthreads();
        k3_pdl_launch_complete();
#pragma unroll
        for (int u = 0; u < VPL; ++u) Srow[u * 32 + lane] = s2v[u];
    }
}

inline bool aligned16(const void* p) { return ((uintptr_t)p & 15u) == 0; }

}  // namespace

bool k3_kda_decode_step_ip(float* out, float* state,
                           const float* q, const float* k, const float* v,
                           const float* g, const float* beta,
                           int head_dim, int n_head, bool beta_sigmoid,
                           cudaStream_t stream) {
    static const bool want = [] {
        const char* e = std::getenv("SPARKINFER_K3_KDA_IP");
        return !(e && e[0] == '0');
    }();
    if (!want || head_dim <= 0 || n_head <= 0) return false;

    // K3 runs head_dim 128 and VPL 1. Anything else declines rather than growing a
    // register array it cannot index at compile time.
    if (head_dim != 128) return false;

    // The float4 path needs 16-byte alignment on all four vectors and on each state
    // row. Rows are head_dim floats apart and head_dim is 128, so the row stride is
    // aligned by construction; the bases are CHECKED rather than assumed, because a
    // scratch arena is free to hand out any offset and a misaligned float4 load is a
    // fault, not a slowdown.
    if (!aligned16(state) || !aligned16(q) || !aligned16(k) || !aligned16(g)) return false;

    // CPB 4 puts four columns (four warps, 128 threads) in a block, so the grid is
    // (n_head, head_dim/4) = (12, 32) = 384 blocks at the sharded shape — 8x the
    // tiled kernel's 48, and every block full.
    // Columns (warps) per block, MEASURED on the node rather than picked: 8x H200,
    // D=128 H=12, 30 reps — CPB 1/2/4/8 gave 7.55 / 7.55 / 5.57 / 5.70 us. 4 wins, and
    // the shape of the curve says why: below 4 the block cannot cover the memory
    // latency of its own column, above it the grid stops covering the device.
    // SPARKINFER_K3_KDA_CPB re-runs that sweep on one binary.
    static const int cpb = [] {
        const char* e = std::getenv("SPARKINFER_K3_KDA_CPB");
        const int v = e ? std::atoi(e) : 0;
        return (v == 1 || v == 2 || v == 4 || v == 8) ? v : 4;
    }();

    // Defer the 768 KiB state write-back past the 6 KiB output write and signal
    // programmatic completion between them. SPARKINFER_K3_KDA_DEFER=0 restores the
    // original write order on the same binary, which is what makes this an A/B of one
    // process rather than of two builds.
    //
    // The divisibility test is a CORRECTNESS gate, not a tuning one: the deferring
    // kernel's trigger sits below the `j >= head_dim` return, and a block that split
    // between exited warps and triggering warps is undefined. head_dim % cpb == 0 makes
    // that return unreachable. At K3's shape (128 % 4) it always holds; a shape where
    // it does not simply keeps the original order.
    static const bool want_defer = [] {
        const char* e = std::getenv("SPARKINFER_K3_KDA_DEFER");
        return !(e && e[0] == '0');
    }();
    const bool defer = want_defer && (head_dim % cpb == 0);

#define K3_KDA_LAUNCH(C, D)                                                         \
    do {                                                                            \
        const dim3 g_((unsigned)n_head, (unsigned)((head_dim + (C) - 1) / (C)));     \
        k3_pdl_launch(g_, dim3((C) * 32), 0, stream, kda_step_ip_kernel<(C), 1, D>,  \
                      out, state, q, k, v, g, beta, head_dim, beta_sigmoid);        \
    } while (0)
#define K3_KDA_DISPATCH(C) \
    do { if (defer) K3_KDA_LAUNCH(C, true); else K3_KDA_LAUNCH(C, false); } while (0)
    switch (cpb) {
        case 1: K3_KDA_DISPATCH(1); break;
        case 2: K3_KDA_DISPATCH(2); break;
        case 8: K3_KDA_DISPATCH(8); break;
        default: K3_KDA_DISPATCH(4); break;
    }
#undef K3_KDA_DISPATCH
#undef K3_KDA_LAUNCH
    return true;
}

}  // namespace k3
}  // namespace kernels
}  // namespace sparkinfer
