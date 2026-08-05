// Kimi K3 kernels. Semantics transcribed from unslothai/llama.cpp @ efc8bc38 —
// see kernels/include/sparkinfer/kernels/kimi_k3.h for the formula each one must
// match and the wrong-but-plausible variant it is easy to write instead.
//
// f32 throughout. These are correctness-first reference implementations whose job is
// to pass kernels/tests/kimi_k3_numeric_test.cu against a float64 CPU model of the
// same math. Quantised / bf16 / fused variants come after the math is pinned, not
// before: a fast kernel that is subtly wrong is worse than no kernel, because the
// model stays fluent and the error looks like a quality regression.

#include "sparkinfer/kernels/kimi_k3.h"
#include "sparkinfer/kernels/iq2xs_tables.h"
#include <climits>   // INT_MAX — the "no expert band" sentinel
#include "sparkinfer/kernels/iq1s_tables.h"
#include "k3_pdl.cuh"
#include "sparkinfer/kernels/kimi_k3_fast.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>   // uintptr_t — the float4-alignment test in the MoE dispatch

namespace sparkinfer {
namespace kernels {
namespace k3 {

namespace {

__device__ __forceinline__ float sigmoidf_(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// Block-wide sum into lane 0 of warp 0, then broadcast via shared memory.
template <int BLOCK>
__device__ __forceinline__ float block_sum(float v, float* shm) {
    // warp reduce
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) shm[warp] = v;
    __syncthreads();
    constexpr int NWARP = BLOCK / 32;
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int w = 0; w < NWARP; ++w) s += shm[w];
        shm[NWARP] = s;
    }
    __syncthreads();
    return shm[NWARP];
}


// WEPS engagement counter. A KL of 0 or a speed delta is UNATTRIBUTABLE unless
// we can prove the gate actually fired — this branch has already been burned by a
// verification that silently never ran the kernel under test.
//
// The count flag exists so the SCORED build never pays for the counting: at high
// skip rates the two atomicAdds are up to ~1.3M same-address RMWs per token, which
// is measurement overhead masquerading as the thing being measured. It is uploaded
// per device by k3_prewarm_quant_tables (pre-capture, synchronous-copy-legal), so
// the kernels read a flag that is constant for the whole run.
__device__ unsigned long long g_k3_weps_skips = 0;
__device__ int g_k3_weps_count = 0;

// Depth gate for the weight threshold. MEASURED WHY IT EXISTS: with the threshold
// live at every depth, the checkpoint dumps diverged at every graded prefix and
// ctx2048's KL hit 1.1e-1 — over the absolute accuracy bar, not a ratchet nuance.
// At shallow depth the router's renormalised distribution is CONCENTRATED, and a
// sub-threshold expert still carries real mass; by ctx4096 the same comparison
// reads 8.2e-4. So the threshold engages only past g_k3_weps_min_pos, where the
// distribution has flattened and the dropped terms are the near-irrelevant tail.
// Bound once per device, pre-capture; the position pointer is the same device
// mirror the MLA kernels read, advanced once per token, so the gate cannot flip
// between a layer's gate/up pass and its down combine.
__device__ const int* g_k3_weps_pos = nullptr;
__device__ int g_k3_weps_min_pos = 0;

__device__ inline bool k3_weps_depth_live() {
    return g_k3_weps_pos == nullptr || *g_k3_weps_pos >= g_k3_weps_min_pos;
}

// ---------------------------------------------------------------------------
// 1. situ
// ---------------------------------------------------------------------------

__global__ void situ_kernel(float* __restrict__ out, const float* __restrict__ gate,
                            const float* __restrict__ up, int64_t n,
                            float beta, float inv_beta, float lb, float inv_lb,
                            int lb_active) {
    k3_pdl_sync();
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    const float u = up[i];
    // gate branch: BOTH scaled-tanh and sigmoid.
    float a = beta * tanhf(g * inv_beta) * sigmoidf_(g);
    // up branch: scaled-tanh only, and only when linear_beta > 0.
    const float ub = lb_active ? (lb * tanhf(u * inv_lb)) : u;
    out[i] = a * ub;
}

// ---------------------------------------------------------------------------
// 2. KDA decode step
// ---------------------------------------------------------------------------
// One block per head. head_dim threads; each thread owns column j of the state.
// The state column j is strided by head_dim in memory (i is fastest), so a thread
// walks S[i][j] with stride head_dim — coalesced across threads for fixed i, which
// is the access pattern both reduction passes want.

// STAGED VARIANT. Identical arithmetic; the state makes the trip through shared memory
// in COLUMN CHUNKS instead of being walked in place.
//
// Thread j wants S[j * head_dim + i] for every i. In global that is a 512-byte stride
// across the warp, so each load instruction touches 32 sectors to use 128 bytes — an 8x
// read amplification over a state that is 96 heads x 128 x 128 floats = 6.3 MB per KDA
// layer, read AND written twice per token across 69 KDA layers. Measured on 8x H200 at
// tp=8 (nsys, main 86bb264) the kernel moved 24.6 MB per launch in 90.5 us = 271 GB/s.
//
// The chunk copy is linear — thread t takes (row t/IC, column i0 + t%IC), so a warp
// covers one row's 32 consecutive floats — and the strided walk then happens in shared,
// padded to IC + 1 so bank = (j + c) % 32 is distinct across the warp.
//
// IC = 32 IS THE POINT, not a tuning knob. head_dim x (IC+1) + 4*head_dim floats is
// 18.5 KB, under the 48 KB a block gets by default. Staging the whole 128x128 tile
// instead needs 68 KB, which requires cudaFuncSetAttribute per device and a fallback
// when it is refused — and a run that silently took that fallback is indistinguishable
// in the output from one that did not. Staying under the default removes the opt-in,
// the fallback, and the question.
//
// THE STORED LAYOUT IS UNCHANGED, deliberately. Transposing the state to i-major would
// coalesce the global access directly and is WRONG: kimi_k3_numeric_test seeds a random
// non-zero state and its float64 reference documents the layout ("S[i][j] at s[j*D+i]"),
// so j-major is a contract. It only looks private because production starts from zero.
//
// BIT-IDENTICAL: sk and o still accumulate i in increasing order, across chunks and
// within them, against the same operands.
template <int BLOCK>
__global__ void kda_decode_step_smem_kernel(float* __restrict__ out,
                                            float* __restrict__ state,
                                            const float* __restrict__ q,
                                            const float* __restrict__ k,
                                            const float* __restrict__ v,
                                            const float* __restrict__ g,
                                            const float* __restrict__ beta,
                                            int head_dim) {
    k3_pdl_sync();
    constexpr int IC = 32;                  // state columns staged per chunk
    const int SP = IC + 1;                  // padded row stride, kills bank conflicts
    const int h = blockIdx.x;
    const int j = threadIdx.x;

    float* S = state + (size_t)h * head_dim * head_dim;
    const float* qh = q + (size_t)h * head_dim;
    const float* kh = k + (size_t)h * head_dim;
    const float* vh = v + (size_t)h * head_dim;
    const float* gh = g + (size_t)h * head_dim;
    const float  b  = beta[h];

    extern __shared__ float smem[];
    float* s_k  = smem;
    float* s_q  = s_k + head_dim;
    float* s_sk = s_q + head_dim;
    float* s_ge = s_sk + head_dim;
    float* s_T  = s_ge + head_dim;          // head_dim rows of SP

    if (j < head_dim) {
        s_k[j]  = kh[j];
        s_q[j]  = qh[j];
        s_ge[j] = __expf(gh[j]);
    }
    __syncthreads();

    // Step 1 + 2: decay by exp(g[i]) and sk[j] = sum_i S[i][j]*k[i].
    float sk = 0.0f;
    for (int i0 = 0; i0 < head_dim; i0 += IC) {
        for (int t = j; t < head_dim * IC; t += BLOCK)
            s_T[(t / IC) * SP + (t % IC)] = S[(size_t)(t / IC) * head_dim + i0 + (t % IC)];
        __syncthreads();
        if (j < head_dim) {
            for (int c = 0; c < IC; ++c) {
                const float sv = s_T[j * SP + c] * s_ge[i0 + c];
                s_T[j * SP + c] = sv;
                sk += sv * s_k[i0 + c];
            }
        }
        __syncthreads();
        for (int t = j; t < head_dim * IC; t += BLOCK)
            S[(size_t)(t / IC) * head_dim + i0 + (t % IC)] = s_T[(t / IC) * SP + (t % IC)];
        __syncthreads();
    }
    if (j < head_dim) s_sk[j] = sk;
    __syncthreads();

    // Step 3: d[j] = beta * (v[j] - sk[j]).
    float d_j = 0.0f;
    if (j < head_dim) d_j = b * (vh[j] - s_sk[j]);
    __syncthreads();

    // Step 4 + 5: rank-1 update, then o[j] = sum_i S[i][j]*q[i].
    float o = 0.0f;
    for (int i0 = 0; i0 < head_dim; i0 += IC) {
        for (int t = j; t < head_dim * IC; t += BLOCK)
            s_T[(t / IC) * SP + (t % IC)] = S[(size_t)(t / IC) * head_dim + i0 + (t % IC)];
        __syncthreads();
        if (j < head_dim) {
            for (int c = 0; c < IC; ++c) {
                const float sv = s_T[j * SP + c] + s_k[i0 + c] * d_j;
                s_T[j * SP + c] = sv;
                o += sv * s_q[i0 + c];
            }
        }
        __syncthreads();
        for (int t = j; t < head_dim * IC; t += BLOCK)
            S[(size_t)(t / IC) * head_dim + i0 + (t % IC)] = s_T[(t / IC) * SP + (t % IC)];
        __syncthreads();
    }
    if (j < head_dim) out[(size_t)h * head_dim + j] = o;
}

// Value-tiled KDA decode step. Same arithmetic as kda_decode_step_smem_kernel, same
// stored layout, same accumulation order — it only adds a SECOND grid axis.
//
// WHY: the grid was `n_head`, and #63 head-sharded KDA, so n_head went 96 -> 12 at tp=8.
// That is 12 blocks on a 132-SM H200 — 9% of the device — for 11.8% of all GPU kernel
// time (nsys, ctx 131072). The kernel moves ~3.1 MB per launch and takes 87 us against a
// ~0.9 us bandwidth floor, with a 982 ns stddev: it is not bandwidth-bound or contended,
// it is starved of blocks. #63 budgeted the MLA split cap for exactly this reason
// (kMlaSplitBudget) and left the kernel beside it at one block per head.
//
// WHY THIS DECOMPOSITION IS LEGAL: every step is column-local. For a state column j the
// decay, sk[j] = sum_i S[i][j]*k[i], d[j], the rank-1 update and o[j] all touch only
// column j plus the per-head vectors k/q/g, which are read-only and shared. So columns
// split across blocks with NO extra reduction and NO duplicated state traffic — a block
// reads exactly its own BV rows of the j-major buffer.
//
// This is also what the reference implementations do: FLA's fused_recurrent_gated_delta_rule
// (the kernel vLLM ships for decode) launches grid = (cdiv(V, BV) * N * HV) with BV = 32,
// one program per sequence, value head, and 32-wide VALUE TILE. BV = 32 here is theirs.
//
// ONE THREAD PER COLUMN, deliberately, even though it makes the block 32 threads wide.
// It keeps each thread accumulating i in increasing order across and within chunks, so
// this stays BIT-IDENTICAL to the kernel it replaces. Splitting the i-reduction across
// R threads would parallelise further but turns a sequential sum into a tree, and a
// change that moves the logits is a different change that has to be argued separately.
template <int BV>
__global__ void kda_decode_step_vt_kernel(float* __restrict__ out,
                                          float* __restrict__ state,
                                          const float* __restrict__ q,
                                          const float* __restrict__ k,
                                          const float* __restrict__ v,
                                          const float* __restrict__ g,
                                          const float* __restrict__ beta,
                                          int head_dim) {
    k3_pdl_sync();
    constexpr int IC = 32;
    const int SP = IC + 1;
    const int h  = blockIdx.x;
    const int j0 = blockIdx.y * BV;      // this block's value tile
    const int jl = threadIdx.x;          // column within the tile
    const int j  = j0 + jl;

    float* S = state + (size_t)h * head_dim * head_dim;
    const float* qh = q + (size_t)h * head_dim;
    const float* kh = k + (size_t)h * head_dim;
    const float* vh = v + (size_t)h * head_dim;
    const float* gh = g + (size_t)h * head_dim;
    const float  b  = beta[h];

    extern __shared__ float smem[];
    float* s_k  = smem;                  // full head_dim: every column needs every i
    float* s_q  = s_k + head_dim;
    float* s_ge = s_q + head_dim;
    float* s_T  = s_ge + head_dim;       // BV rows of SP

    // Cooperative over BV threads, not head_dim — the tile is narrower than the vectors.
    for (int t = jl; t < head_dim; t += BV) {
        s_k[t]  = kh[t];
        s_q[t]  = qh[t];
        s_ge[t] = __expf(gh[t]);
    }
    __syncthreads();

    // Step 1 + 2: decay by exp(g[i]) and sk = sum_i S[i][j]*k[i].
    //
    // THE DECAYED STATE IS NOT WRITTEN BACK HERE. The staged kernel stored S*exp(g) to
    // global at the end of this pass and re-read it in the next one; pass 2 below simply
    // recomputes the product from the untouched S. That drops a full write of the state
    // per launch — a quarter of this kernel's global traffic — plus one staging loop and
    // one barrier per chunk, and the recomputed multiply is free next to the load it
    // replaces.
    //
    // __fmul_rn, not `*`, and that is load-bearing. The old pass 1 stored the product to
    // shared, which forced it to be materialised as a rounded f32. Here it feeds an add
    // directly, so nvcc (--fmad=true by default) would happily contract it into an fma
    // and skip the intermediate rounding — giving a different result in the last bits and
    // silently costing the bit-identity this whole change is claiming.
    float sk = 0.0f;
    for (int i0 = 0; i0 < head_dim; i0 += IC) {
        for (int t = jl; t < BV * IC; t += BV)
            s_T[(t / IC) * SP + (t % IC)] =
                S[(size_t)(j0 + t / IC) * head_dim + i0 + (t % IC)];
        __syncthreads();
        for (int c = 0; c < IC; ++c)
            sk += __fmul_rn(s_T[jl * SP + c], s_ge[i0 + c]) * s_k[i0 + c];
        __syncthreads();
    }

    // Step 3: d = beta * (v[j] - sk). Column-local, so it stays in a register — the
    // staged kernel routed this through shared only because it had a wider block.
    const float d_j = b * (vh[j] - sk);

    // Step 4 + 5: rank-1 update, then o = sum_i S[i][j]*q[i].
    float o = 0.0f;
    for (int i0 = 0; i0 < head_dim; i0 += IC) {
        for (int t = jl; t < BV * IC; t += BV)
            s_T[(t / IC) * SP + (t % IC)] =
                S[(size_t)(j0 + t / IC) * head_dim + i0 + (t % IC)];
        __syncthreads();
        for (int c = 0; c < IC; ++c) {
            // S here is the ORIGINAL state, because pass 1 no longer wrote its decayed
            // copy — so the decay is reapplied, to exactly the same operands with exactly
            // the same rounding, and the sum below sees the identical f32 the staged
            // kernel read back from global.
            const float sv = __fmul_rn(s_T[jl * SP + c], s_ge[i0 + c]) + s_k[i0 + c] * d_j;
            s_T[jl * SP + c] = sv;
            o += sv * s_q[i0 + c];
        }
        __syncthreads();
        for (int t = jl; t < BV * IC; t += BV)
            S[(size_t)(j0 + t / IC) * head_dim + i0 + (t % IC)] =
                s_T[(t / IC) * SP + (t % IC)];
        __syncthreads();
    }
    out[(size_t)h * head_dim + j] = o;
}

template <int BLOCK>
__global__ void kda_decode_step_kernel(float* __restrict__ out, float* __restrict__ state,
                                       const float* __restrict__ q,
                                       const float* __restrict__ k,
                                       const float* __restrict__ v,
                                       const float* __restrict__ g,
                                       const float* __restrict__ beta,
                                       int head_dim) {
    k3_pdl_sync();
    const int h = blockIdx.x;
    const int j = threadIdx.x;
    if (j >= head_dim) return;

    float* S = state + (size_t)h * head_dim * head_dim;
    const float* qh = q + (size_t)h * head_dim;
    const float* kh = k + (size_t)h * head_dim;
    const float* vh = v + (size_t)h * head_dim;
    const float* gh = g + (size_t)h * head_dim;
    const float  b  = beta[h];

    extern __shared__ float smem[];
    float* s_k  = smem;                       // head_dim
    float* s_q  = s_k + head_dim;             // head_dim
    float* s_sk = s_q + head_dim;             // head_dim  (sk, then reused for d)
    float* s_ge = s_sk + head_dim;            // head_dim  exp(g[i]), i = CONTRACTION index

    s_k[j] = kh[j];
    s_q[j] = qh[j];
    // exp(g) is needed indexed by i inside the loop below, not by this thread's j, so
    // it has to be staged in shared memory rather than read per-thread.
    s_ge[j] = __expf(gh[j]);
    __syncthreads();

    // Step 1: per-channel decay, and Step 2: sk[j] = sum_i S[i][j]*k[i].
    // Fused into one pass over the column — the decay is applied as we read.
    //
    // THE DECAY IS INDEXED BY i, THE CONTRACTION INDEX — NOT BY j.
    //
    // The reference (ggml_compute_forward_gated_delta_net_one_chunk, ggml/src/
    // ggml-cpu/ops.cpp) stores the state transposed exactly as this kernel does
    // (s_out[j*S_v + i] == S[i][j]) and then does:
    //
    //     for (i) delta[i] = expf(g_d[i]);
    //     for (j) ggml_vec_mul_f32(S_v, &s_out[j*S_v], &s_out[j*S_v], delta);
    //
    // i.e. row j of the transposed state is multiplied ELEMENTWISE by exp(g) — the
    // multiplier varies along i. Its own comment: "S[i][:] *= exp(g[i])".
    //
    // This kernel previously hoisted `exp(g[j])` out of the i-loop, which is
    // diag(exp g)*S vs S*diag(exp g) — a different operator, not a relabeling. The
    // roles of i and j are pinned by the surrounding math (k and q contract over i;
    // v, d and out are indexed by j), so there is no transpose elsewhere that
    // compensates.
    //
    // It survived every test for three separate reasons, all worth knowing:
    //   - kimi_k3_numeric_test.cu's float64 "independent" reference was transcribed
    //     from this same wrong contract, so it agreed with the bug.
    //   - kimi_k3_layer0_ref_check.cpp is SINGLE-TOKEN, and at token 0 the state is
    //     zero, so both axes give bit-identical results. It structurally cannot see it.
    //   - the contract cited build_delta_net_autoregressive, which llama.cpp NEVER
    //     EXECUTES: llama-context.cpp sets cparams.fused_gdn_ar = true, and
    //     delta-net-base.cpp routes to build_delta_net_fused instead. sparkinfer
    //     transcribed a dead path that disagrees with the live one.
    float sk = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
        const float sv = S[(size_t)j * head_dim + i] * s_ge[i];
        S[(size_t)j * head_dim + i] = sv;
        sk += sv * s_k[i];
    }
    s_sk[j] = sk;
    __syncthreads();

    // Step 3: d[j] = beta * (v[j] - sk[j]).
    const float d_j = b * (vh[j] - s_sk[j]);
    __syncthreads();          // all reads of s_sk done before overwrite
    s_sk[j] = d_j;            // reuse the buffer as d
    __syncthreads();

    // Step 4 + 5: rank-1 update on column j, then o[j] = sum_i S[i][j]*q[i].
    float o = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
        const float sv = S[(size_t)j * head_dim + i] + s_k[i] * d_j;
        S[(size_t)j * head_dim + i] = sv;
        o += sv * s_q[i];
    }
    out[(size_t)h * head_dim + j] = o;
}

// ---------------------------------------------------------------------------
// 3. KDA output gating
// ---------------------------------------------------------------------------

template <int BLOCK>
__global__ void kda_gate_out_kernel(float* __restrict__ out, const float* __restrict__ o,
                                    const float* __restrict__ norm_w,
                                    const float* __restrict__ g2,
                                    int head_dim, float eps) {
    k3_pdl_sync();
    const int h = blockIdx.x;
    const float* oh = o + (size_t)h * head_dim;
    const float* gh = g2 + (size_t)h * head_dim;
    float* dst = out + (size_t)h * head_dim;

    __shared__ float shm[BLOCK / 32 + 1];

    float acc = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += BLOCK) {
        const float x = oh[d];
        acc += x * x;
    }
    const float ss = block_sum<BLOCK>(acc, shm);
    const float inv = rsqrtf(ss / (float)head_dim + eps);

    for (int d = threadIdx.x; d < head_dim; d += BLOCK) {
        dst[d] = (oh[d] * inv * norm_w[d]) * sigmoidf_(gh[d]);
    }
}

// ---------------------------------------------------------------------------
// 4. cross-layer attention residual mix
// ---------------------------------------------------------------------------
// Pass 1: score every banked checkpoint plus the current stream from the NORMALISED
// values. Pass 2: softmax. Pass 3: weighted sum over the RAW values. The
// normalised/raw split is the reference's, not a choice — see the header.
//
// SPLIT ACROSS THE DEVICE. All three passes used to run in a SINGLE BLOCK —
// <<<1, 256>>>, one SM of the H200's 132 — and the mix is called twice per layer,
// ~185 times per token. Measured on 8x H200 at tp=8 it was 5.7 ms of the 98 ms
// token, 5.8% of decode, spent with 99.2% of the machine idle.
//
// The serialisation was in the shape, not the arithmetic. Pass 1 scores each
// checkpoint INDEPENDENTLY, so the one block walked `c` in a loop for no reason
// other than that it was one block; it becomes one block per checkpoint. Pass 3
// writes each of the n_embd outputs independently; it becomes a normal grid.
//
// EVERY VALUE IS BIT-IDENTICAL, and the reasons are worth stating because "close
// enough" would not be checkable by byte comparison:
//   - the score kernel keeps BLOCK=256 and the same `d += BLOCK` stride, so each
//     thread's partial sum is over the same elements in the same order, and
//     block_sum<256> then folds them in the same tree;
//   - the softmax is recomputed per block in pass 3 rather than passed down. It is
//     at most n_ckpt+1 = 9 values, so a third launch would cost more than the
//     redundancy, and thread 0 walks max/exp/normalise in the SAME order the
//     one-block version did;
//   - pass 3 accumulates `c` in increasing order and adds the current stream last,
//     exactly as before. Reassociating that sum would be equivalent in real
//     arithmetic and not in float.
//
// The two launches also drop a cudaMallocAsync/cudaFreeAsync PAIR per call — ~370
// stream-ordered allocator calls per token — because the scores now live in a
// caller-owned buffer. That is a prerequisite for ever capturing the decode step
// into a CUDA graph, which stream-ordered allocation inside the captured region
// complicates; it is not the reason for the change, but it is not an accident.

// ONE PASS OVER `src`, NOT TWO — the normaliser factors straight out of the dot.
//
// The scale is a constant across d, so
//
//     sum_d (src[d] * inv) * w[d]  ==  inv * sum_d src[d] * w[d]
//
// and the second sweep over `src` never had to exist. It was a FULL DEPENDENT RE-READ
// of n_embd floats (28 KB at hidden 7168) that could not issue until the first block_sum
// had produced `inv`, on a kernel whose grid is n_ckpt+1 — TWO blocks at the scored
// shape, so there is nothing resident to hide that latency behind. 186 calls per token
// per rank.
//
// This removes reads, not parallelism: same grid, same block, same threads, and the two
// accumulators are independent so the single pass issues both loads together instead of
// serialising one sweep behind the other.
//
// NOT bit-identical, and deliberately so: folding `inv` out of the sum reassociates it.
// Same value in exact arithmetic, and `scores` feeds a softmax over n_ckpt+1 entries, so
// the perturbation is far below what the top1/KL gate can see.
//
// SPARKINFER_K3_RES_1PASS=0 restores the two-pass form on one binary.
template <int BLOCK, bool ONEPASS>
__global__ void attn_res_score_kernel(float* __restrict__ scores,
                                      const float* __restrict__ ckpts,
                                      const float* __restrict__ cur,
                                      const float* __restrict__ cur_b,
                                      const float* __restrict__ score_w,
                                      int n_embd, int n_ckpt, float eps,
                                      int64_t act_stride, int64_t bank_stride,
                                      int64_t score_stride) {
    k3_pdl_sync();
    const int row = (int)blockIdx.y;
    scores += (int64_t)row * score_stride;
    ckpts += (int64_t)row * bank_stride;
    cur += (int64_t)row * act_stride;
    if (cur_b) cur_b += (int64_t)row * act_stride;
    // Two reduction buffers so the second block_sum cannot race the first's broadcast.
    __shared__ float shm[2 * (BLOCK / 32 + 1)];
    // blockIdx.x in [0, n_ckpt]: the last block scores the current stream, which is
    // the only thing that made pass 1 look sequential.
    const int c = blockIdx.x;
    const float* __restrict__ src = (c < n_ckpt) ? ckpts + (size_t)c * n_embd : cur;
    // cur may arrive as an unsummed residual pair (see attn_res_mix_f32): the last
    // block adds the two streams itself, with the same single f32 add the deleted
    // standalone kernel performed. Block-uniform, so no divergence.
    const float* __restrict__ srcb = (c < n_ckpt) ? nullptr : cur_b;

    if constexpr (ONEPASS) {
        float ss = 0.0f, raw = 0.0f;
        // Unroll 4, not 8: at the wide launch (BLOCK 1024, n_embd 7168) the trip
        // count is SEVEN, and a pragma bigger than the trip count never enters its
        // unrolled body -- the BPR3 lesson. Four names 8-12 outstanding loads on a
        // grid of 1-9 CTAs, where there is no other resident CTA to hide latency
        // behind. Both accumulators are serial dependency chains, so the summation
        // order -- and therefore the result -- is identical at any unroll factor.
#pragma unroll 4
        for (int d = threadIdx.x; d < n_embd; d += BLOCK) {
            float v = src[d];
            if (srcb) v += srcb[d];
            ss  += v * v;
            raw += v * score_w[d];
        }
        ss  = block_sum<BLOCK>(ss, shm);
        raw = block_sum<BLOCK>(raw, shm + BLOCK / 32 + 1);
        if (threadIdx.x == 0)
            scores[c] = raw * rsqrtf(ss / (float)n_embd + eps);
    } else {
        float ss = 0.0f;
        for (int d = threadIdx.x; d < n_embd; d += BLOCK) {
            float v = src[d];
            if (srcb) v += srcb[d];
            ss += v * v;
        }
        ss = block_sum<BLOCK>(ss, shm);
        const float inv = rsqrtf(ss / (float)n_embd + eps);

        float dot = 0.0f;
        for (int d = threadIdx.x; d < n_embd; d += BLOCK) {
            float v = src[d];
            if (srcb) v += srcb[d];
            dot += (v * inv) * score_w[d];
        }
        dot = block_sum<BLOCK>(dot, shm);
        if (threadIdx.x == 0) scores[c] = dot;
    }
}

template <int BLOCK>
__global__ void attn_res_apply_kernel(float* __restrict__ out,
                                      const float* __restrict__ ckpts,
                                      const float* __restrict__ cur,
                                      const float* __restrict__ cur_b,
                                      float* __restrict__ sum_out,
                                      const float* __restrict__ scores,
                                      int n_embd, int n_ckpt,
                                      int64_t act_stride, int64_t bank_stride,
                                      int64_t score_stride) {
    k3_pdl_sync();
    const int row = (int)blockIdx.y;
    out += (int64_t)row * act_stride;
    ckpts += (int64_t)row * bank_stride;
    cur += (int64_t)row * act_stride;
    if (cur_b) cur_b += (int64_t)row * act_stride;
    if (sum_out) sum_out += (int64_t)row * act_stride;
    scores += (int64_t)row * score_stride;
    extern __shared__ float p[];              // n_ckpt + 1 softmax weights

    // THE SOFTMAX PROLOGUE WAS A SERIAL DEPENDENT WALK IN THREAD 0, IN EVERY BLOCK.
    //
    // The original form had lane 0 traverse `scores` TWICE from global — once for the
    // max, once for the exponentials — while the other BLOCK-1 threads sat at the
    // barrier below. That is 2*(n_ckpt+1) dependent global loads, issued one at a time,
    // on the critical path of a kernel that is only ~2.7 us long and runs 186 times per
    // token per rank. The loads hit L2 (the score kernel wrote them microseconds
    // earlier), but L2 latency serialised n+1 deep is still most of this kernel.
    //
    // Now every entry is fetched by its OWN thread, so all n+1 loads are in flight at
    // once, and the max and the sum come from a warp reduction instead of a scalar
    // loop. n_ckpt+1 is at most a few dozen, so one warp covers it. This removes
    // dependent loads and adds parallelism; it does not fuse or narrow anything.
    //
    // Reassociates the sum (shuffle tree rather than left-to-right) and divides once
    // instead of multiplying by a reciprocal, so it is not bit-identical — the result
    // is a softmax over a handful of entries feeding a residual blend, far inside the
    // accuracy gate.
    //
    // One warp covers the reduction, so this form is only correct while n_ckpt+1 <= 32.
    // K3 runs 2 and 9 at the scored context, but the checkpoint count is a function of
    // attn_res_block_size and the position, not a constant — so the wide form DECLINES
    // above a warp rather than silently dropping entries, and the original scalar
    // prologue below stays reachable for that case.
    const int n = n_ckpt + 1;
    if (n > 32) {
        if (threadIdx.x == 0) {
            float mx = scores[0];
            for (int c = 1; c < n; ++c) mx = fmaxf(mx, scores[c]);
            float sum = 0.0f;
            for (int c = 0; c < n; ++c) { p[c] = __expf(scores[c] - mx); sum += p[c]; }
            const float inv = 1.0f / sum;
            for (int c = 0; c < n; ++c) p[c] *= inv;
        }
        __syncthreads();
        goto blend;
    }
    if (threadIdx.x < n) p[threadIdx.x] = scores[threadIdx.x];
    __syncthreads();
    if (threadIdx.x < 32) {
        const float v = (threadIdx.x < n) ? p[threadIdx.x] : -1e30f;
        float mx = v;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
        const float e = (threadIdx.x < n) ? __expf(v - mx) : 0.0f;
        float sum = e;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
        if (threadIdx.x < n) p[threadIdx.x] = e / sum;
    }
    __syncthreads();

blend:
    {
        const int d = blockIdx.x * BLOCK + threadIdx.x;
        if (d >= n_embd) return;
        // The residual pair, summed here with the same single f32 add the deleted
        // standalone kernel performed; sum_out is that kernel's store. Every element
        // is written by exactly one thread, so the write count is unchanged too.
        float cv = cur[d];
        if (cur_b) cv += cur_b[d];
        if (sum_out) sum_out[d] = cv;
        float acc = 0.0f;
        for (int c = 0; c < n_ckpt; ++c) acc += p[c] * ckpts[(size_t)c * n_embd + d];
        acc += p[n_ckpt] * cv;
        out[d] = acc;
    }
}

// ---------------------------------------------------------------------------
// 6. KDA causal short conv, decode step
// ---------------------------------------------------------------------------
// One thread per channel. d_conv is 4 for K3, so the window is read into registers
// and the shift is a register rotate rather than a memory move.

__global__ void kda_conv_step_kernel(float* __restrict__ out, float* __restrict__ state,
                                     const float* __restrict__ x,
                                     const float* __restrict__ w,
                                     int d_conv, int d_inner) {
    k3_pdl_sync();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= d_inner) return;

    // state is [d_conv-1, d_inner], time fastest -> channel c's history is contiguous.
    float* st = state + (size_t)c * (d_conv - 1);
    const float* wc = w + (size_t)c * d_conv;

    // Build the window: history first, CURRENT token last. The order matters — the
    // reference appends x after the state, so wc[d_conv-1] weights the new sample.
    float acc = 0.0f;
    #pragma unroll 4
    for (int t = 0; t < d_conv - 1; ++t) acc += st[t] * wc[t];
    const float xc = x[c];
    acc += xc * wc[d_conv - 1];

    // silu AFTER the convolution.
    out[c] = acc * sigmoidf_(acc);

    // Shift the history left by one and drop in the current token.
    #pragma unroll 4
    for (int t = 0; t < d_conv - 2; ++t) st[t] = st[t + 1];
    st[d_conv - 2] = xc;
}

// ---------------------------------------------------------------------------
// 11. IQ2_XS dequantisation
// ---------------------------------------------------------------------------
// One thread per 8-value group (one qs entry). 32 groups per 256-value block.

// THE LATTICE TABLES LIVE IN __device__, NOT __constant__.
//
// The constant unit services ONE distinct address per cycle per SM. block_dot()
// indexes these tables by the quantised codepoint, which is different in every lane
// — a fully 32-way-divergent gather. In __constant__ that serialises into 32 replays
// per access, and the 16 KB grid does not fit the per-SM immediate constant cache, so
// each replay also misses. Through __device__ the same divergent addresses go to L1,
// which returns multiple sectors per cycle and holds 16 KB comfortably.
//
// __constant__ is the right home for a broadcast read (every lane, same address).
// It is close to the worst possible home for this one. llama.cpp reached the same
// conclusion: ggml/src/ggml-common.h declares its CUDA lattice tables
// `static const __device__`, not __constant__.
//
// cudaMemcpyToSymbol works identically against a __device__ symbol, so the
// per-device upload guard below is unchanged.
__device__ uint64_t c_iq2xs_grid[512];
__device__ uint8_t  c_ksigns_iq2xs[128];
__device__ uint8_t  c_kmask_iq2xs[8];

struct BlockIQ2XS {          // 74 bytes, matches ggml block_iq2_xs exactly
    uint16_t d;              // fp16 bits
    uint16_t qs[32];
    uint8_t  scales[8];
};

// 50 bytes, matches ggml block_iq1_s exactly (ggml-common.h asserts
// sizeof(ggml_half) + QK_K/8 + QK_K/16). IQ1_S is the UD-IQ1_S target's expert type.
//
// Reconstruction differs from IQ2_XS in three ways that all matter:
//   - the grid index is 11 BITS: qs[l] low 8, plus 3 bits pulled out of qh
//   - the scale is per 32-value sub-block and ODD-VALUED: dl = d * (2*s + 1)
//   - there is a DELTA: value = dl * (grid + delta), delta = +/-0.125 per sub-block
// The delta is what lets a {-1,0,+1} lattice represent an asymmetric distribution;
// dropping it produces a well-formed tensor with a systematic bias.
struct BlockIQ1S {           // 50 bytes
    uint16_t d;              // fp16 bits
    uint8_t  qs[32];         // grid index, low 8 bits
    uint16_t qh[8];          // 3 bits x4 of grid index, 3-bit scale, 1 sign bit
};

struct BlockQ8K {            // 292 bytes, matches ggml block_q8_K exactly
    float   d;
    int8_t  qs[256];
    int16_t bsums[16];
};
static_assert(sizeof(BlockQ8K) == 292, "bad block_q8_K layout");

__global__ void dequant_iq2_xs_kernel(float* __restrict__ out,
                                      const BlockIQ2XS* __restrict__ blocks,
                                      int64_t n_groups) {
    k3_pdl_sync();
    const int64_t g = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;  // 8-value group
    if (g >= n_groups) return;

    const int64_t ib = g >> 5;          // 32 groups per block
    const int      l = (int)(g & 31);   // group within the block
    const int   ib32 = l >> 2;          // which 32-value sub-group
    const int   sub  = l & 3;           // 0..3 within it

    const BlockIQ2XS& b = blocks[ib];
    const float d = __half2float(__ushort_as_half(b.d));

    // db[l/2], NOT db[l&1]: sub 0,1 take the low nibble and 2,3 the high one.
    const uint8_t sc = b.scales[ib32];
    const float db = (sub < 2) ? d * (0.5f + (float)(sc & 0xf)) * 0.25f
                               : d * (0.5f + (float)(sc >> 4))  * 0.25f;

    const uint16_t q = b.qs[l];
    const uint8_t* grid = (const uint8_t*)&c_iq2xs_grid[q & 511];
    const uint8_t signs = c_ksigns_iq2xs[q >> 9];

    float* dst = out + g * 8;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        dst[j] = db * (float)grid[j] * ((signs & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
    }
}

// One thread per 8-value group, matching dequantize_row_iq1_s's inner loop exactly:
//   dl    = d * (2*((qh[ib] >> 12) & 7) + 1)
//   delta = (qh[ib] & 0x8000) ? -IQ1S_DELTA : +IQ1S_DELTA
//   grid  = iq1s_grid[ qs[4*ib + l] | (((qh[ib] >> 3*l) & 7) << 8) ]
//   y[j]  = dl * (grid[j] + delta)
__global__ void dequant_iq1_s_kernel(float* __restrict__ out,
                                     const BlockIQ1S* __restrict__ blocks,
                                     int64_t n_groups) {
    k3_pdl_sync();
    const int64_t g = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (g >= n_groups) return;

    const int64_t ib  = g >> 5;          // 32 groups of 8 per 256-value block
    const int     l32 = (int)(g & 31);
    const int     ib32 = l32 >> 2;       // which 32-value sub-block (0..7)
    const int     l    = l32 & 3;        // which group of 8 inside it (0..3)

    const BlockIQ1S& b = blocks[ib];
    const float    d   = __half2float(__ushort_as_half(b.d));
    const uint16_t h   = b.qh[ib32];

    const float dl    = d * (float)(2 * ((h >> 12) & 7) + 1);
    const float delta = (h & 0x8000) ? -SPARKINFER_IQ1S_DELTA : SPARKINFER_IQ1S_DELTA;

    const uint32_t idx = (uint32_t)b.qs[4 * ib32 + l] | (((uint32_t)(h >> (3 * l)) & 7u) << 8);
    const int8_t* grid = (const int8_t*)&iq1s_grid_c[idx];

    float* dst = out + g * 8;
    #pragma unroll
    for (int j = 0; j < 8; ++j) dst[j] = dl * ((float)grid[j] + delta);
}

// ---------------------------------------------------------------------------
// 12. noaux_tc MoE router
// ---------------------------------------------------------------------------
// One block per token. K3 picks 16 of 896, so 16 passes of a block-wide argmax is
// ~14k comparisons — trivial next to the expert FFNs that follow, and it matches
// argsort's tie-breaking exactly (lowest index wins) without needing a sort.

template <int BLOCK>
__global__ void moe_router_noaux_tc_kernel(float* __restrict__ out_w,
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

    extern __shared__ float smem_r[];
    float* s_sel = smem_r;                 // biased scores, mutated during selection
    float* s_p   = s_sel + n_expert;       // UNBIASED probs, never mutated
    __shared__ float s_bestv[BLOCK / 32];
    __shared__ int   s_besti[BLOCK / 32];

    for (int e = threadIdx.x; e < n_expert; e += BLOCK) {
        const float p = 1.0f / (1.0f + __expf(-lg[e]));   // sigmoid
        s_p[e]   = p;
        s_sel[e] = bias ? (p + bias[e]) : p;              // bias: selection only
    }
    __syncthreads();

    for (int k = 0; k < top_k; ++k) {
        // block-wide argmax over the remaining biased scores
        float bv = -INFINITY; int bi = -1;
        for (int e = threadIdx.x; e < n_expert; e += BLOCK) {
            const float v = s_sel[e];
            if (v > bv || (v == bv && e < bi)) { bv = v; bi = e; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xffffffff, bv, off);
            const int   oi = __shfl_down_sync(0xffffffff, bi, off);
            if (ov > bv || (ov == bv && oi >= 0 && oi < bi)) { bv = ov; bi = oi; }
        }
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        if (lane == 0) { s_bestv[warp] = bv; s_besti[warp] = bi; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float mv = s_bestv[0]; int mi = s_besti[0];
            for (int wv = 1; wv < BLOCK / 32; ++wv) {
                if (s_bestv[wv] > mv || (s_bestv[wv] == mv && s_besti[wv] < mi)) {
                    mv = s_bestv[wv]; mi = s_besti[wv];
                }
            }
            ids[k] = mi;
            w[k]   = s_p[mi];        // UNBIASED prob — the whole point
            s_sel[mi] = -INFINITY;   // remove from further rounds
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if (norm_w) {
            float sum = 0.0f;
            for (int k = 0; k < top_k; ++k) sum += w[k];
            // smallest normal F16, per the reference — not an arbitrary epsilon
            sum = fmaxf(sum, 6.103515625e-5f);
            for (int k = 0; k < top_k; ++k) w[k] /= sum;
        }
        if (w_scale != 0.0f && w_scale != 1.0f)
            for (int k = 0; k < top_k; ++k) w[k] *= w_scale;
    }
}

// ---------------------------------------------------------------------------
// 13. Latent MoE expert dispatch
// ---------------------------------------------------------------------------

// Decode one IQ2_XS block (256 values) into the caller's dot accumulator against
// `x`. Same arithmetic as dequant_iq2_xs_kernel — kept in one place so the two
// cannot drift, since that kernel is the one validated bit-exact against ggml.
//
// XVEC pulls the lane's 8 activations in as two float4 rather than eight scalars, and
// the grid codepoint in as the one uint64 it is stored as rather than eight separate
// byte loads. Neither changes a value or an order -- the multiply-accumulate below is
// still j = 0..7 against the same operands -- but a lane's 8 values sit 32 bytes apart
// across the warp, so the scalar form issues 8 loads that each span 1024 bytes to use
// 128 of them. It is the same sector-amplification the projection GEMV had, one level
// down.
// The packed IQ1S lattice (2 bits/codepoint; packed host-side in
// ensure_iq1s_tables from the same iq1s_grid_host source of truth).
__device__ static uint16_t iq1s_grid_p[SPARKINFER_IQ1S_NGRID];

template <bool XVEC, bool GPACK = false, bool GS = false>  // IQ1S-only; ignored here
__device__ __forceinline__ float block_dot(const BlockIQ2XS& b,
                                                 const float* __restrict__ x,
                                                 int lane, int nlanes) {
    const float d = __half2float(__ushort_as_half(b.d));
    float acc = 0.0f;
    // 32 groups of 8 values per block; spread them over the lanes.
    for (int l = lane; l < 32; l += nlanes) {
        const int ib32 = l >> 2, sub = l & 3;
        const uint8_t sc = b.scales[ib32];
        const float db = (sub < 2) ? d * (0.5f + (float)(sc & 0xf)) * 0.25f
                                   : d * (0.5f + (float)(sc >> 4))  * 0.25f;
        const uint16_t q = b.qs[l];
        const uint64_t gw = c_iq2xs_grid[q & 511];
        const uint8_t* grid = (const uint8_t*)&c_iq2xs_grid[q & 511];
        const uint8_t signs = c_ksigns_iq2xs[q >> 9];
        const float* xv = x + l * 8;
        if (XVEC) {
            const float4* xv4 = (const float4*)xv;
            const float4 xa = xv4[0], xb = xv4[1];
            const float xs[8] = { xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w };
        #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint8_t gj = (uint8_t)((gw >> (8 * j)) & 0xffu);
                const float wv = db * (float)gj * ((signs & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                acc += wv * xs[j];
            }
        } else {
        #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float wv = db * (float)grid[j] * ((signs & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                acc += wv * xv[j];
            }
        }
    }
    return acc;
}

// IQ1_S counterpart. Same contract: decode inside the dot product and never stage
// dequantised weights. Overloaded on the block type so the MoE kernels below are
// written once and instantiated per quant type — the combine logic must not exist
// twice, or the two copies will drift.
template <bool XVEC, bool GPACK = false, bool GS = false>
__device__ __forceinline__ float block_dot(const BlockIQ1S& b,
                                           const float* __restrict__ x,
                                           int lane, int nlanes) {
    const float d = __half2float(__ushort_as_half(b.d));
    float acc = 0.0f;
    for (int l32 = lane; l32 < 32; l32 += nlanes) {
        const int ib32 = l32 >> 2, l = l32 & 3;
        const uint16_t h = b.qh[ib32];
        const float dl    = d * (float)(2 * ((h >> 12) & 7) + 1);
        const float delta = (h & 0x8000) ? -SPARKINFER_IQ1S_DELTA : SPARKINFER_IQ1S_DELTA;
        const uint32_t idx =
            (uint32_t)b.qs[4 * ib32 + l] | (((uint32_t)(h >> (3 * l)) & 7u) << 8);
        const float* xv = x + l32 * 8;
        if (GPACK) {
            // 2-bit two's-complement fields; ((v << (30-2j)) >> 30) sign-extends
            // field j. (float)gj is the same float whether gj arrived as a
            // sign-extended int8 or a 2-bit field: same value, same arithmetic.
            // GS: the caller staged the table into dynamic shared (extern
            // __shared__ aliases the kernel's dynamic allocation).
            uint32_t pw;
            if (GS) {
                extern __shared__ uint16_t s_iq1s_grid[];
                pw = s_iq1s_grid[idx];
            } else {
                pw = iq1s_grid_p[idx];
            }
            const float4* xv4 = (const float4*)xv;
            const float4 xa = xv4[0], xb = xv4[1];
            const float xs[8] = { xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w };
        #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int gj = ((int)(pw << (30 - 2 * j))) >> 30;
                acc += dl * ((float)gj + delta) * xs[j];
            }
        } else if (XVEC) {
            // The table IS a uint64 per codepoint, so read it as one -- the byte
            // pointer form makes the compiler emit eight dependent 1-byte loads off a
            // divergent address.
            const uint64_t gw = iq1s_grid_c[idx];
            const float4* xv4 = (const float4*)xv;
            const float4 xa = xv4[0], xb = xv4[1];
            const float xs[8] = { xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w };
        #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int8_t gj = (int8_t)((gw >> (8 * j)) & 0xffu);
                acc += dl * ((float)gj + delta) * xs[j];
            }
        } else {
            const int8_t* grid = (const int8_t*)&iq1s_grid_c[idx];
        #pragma unroll
            for (int j = 0; j < 8; ++j) acc += dl * ((float)grid[j] + delta) * xv[j];
        }
    }
    return acc;
}

// Warp-local gate/up activation fuse. Gate and up share the same activation
// block; two block_dot calls reload xs[] twice. CTA smem staging measured -3.1%
// (barrier). Keeping xs in registers and decoding both weight rows against it
// has no barrier. Same FMAs, gate-then-up within each lane group: bit-identical.
template <bool XVEC, bool GPACK = false, bool GS = false>
__device__ __forceinline__ void block_dot_pair(const BlockIQ2XS& g,
                                               const BlockIQ2XS& u,
                                               const float* __restrict__ x,
                                               int lane, int nlanes,
                                               float& g_out, float& u_out) {
    (void)GPACK; (void)GS;
    float gacc = 0.0f, uacc = 0.0f;
    for (int l = lane; l < 32; l += nlanes) {
        const float* xv = x + l * 8;
        float xs[8];
        if (XVEC) {
            const float4* xv4 = (const float4*)xv;
            const float4 xa = xv4[0], xb = xv4[1];
            xs[0] = xa.x; xs[1] = xa.y; xs[2] = xa.z; xs[3] = xa.w;
            xs[4] = xb.x; xs[5] = xb.y; xs[6] = xb.z; xs[7] = xb.w;
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) xs[j] = xv[j];
        }
        const int ib32 = l >> 2, sub = l & 3;
#pragma unroll
        for (int side = 0; side < 2; ++side) {
            const BlockIQ2XS& b = side ? u : g;
            const float d = __half2float(__ushort_as_half(b.d));
            const uint8_t sc = b.scales[ib32];
            const float db = (sub < 2) ? d * (0.5f + (float)(sc & 0xf)) * 0.25f
                                       : d * (0.5f + (float)(sc >> 4))  * 0.25f;
            const uint16_t q = b.qs[l];
            const uint64_t gw = c_iq2xs_grid[q & 511];
            const uint8_t signs = c_ksigns_iq2xs[q >> 9];
            float acc = 0.0f;
        #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint8_t gj = (uint8_t)((gw >> (8 * j)) & 0xffu);
                acc += db * (float)gj * ((signs & c_kmask_iq2xs[j]) ? -1.0f : 1.0f) * xs[j];
            }
            if (side) uacc += acc; else gacc += acc;
        }
    }
    g_out = gacc;
    u_out = uacc;
}

template <bool XVEC, bool GPACK = false, bool GS = false>
__device__ __forceinline__ void block_dot_pair(const BlockIQ1S& g,
                                               const BlockIQ1S& u,
                                               const float* __restrict__ x,
                                               int lane, int nlanes,
                                               float& g_out, float& u_out) {
    float gacc = 0.0f, uacc = 0.0f;
    for (int l32 = lane; l32 < 32; l32 += nlanes) {
        const float* xv = x + l32 * 8;
        float xs[8];
        if (XVEC || GPACK) {
            const float4* xv4 = (const float4*)xv;
            const float4 xa = xv4[0], xb = xv4[1];
            xs[0] = xa.x; xs[1] = xa.y; xs[2] = xa.z; xs[3] = xa.w;
            xs[4] = xb.x; xs[5] = xb.y; xs[6] = xb.z; xs[7] = xb.w;
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) xs[j] = xv[j];
        }
        const int ib32 = l32 >> 2, l = l32 & 3;
#pragma unroll
        for (int side = 0; side < 2; ++side) {
            const BlockIQ1S& b = side ? u : g;
            const float d = __half2float(__ushort_as_half(b.d));
            const uint16_t h = b.qh[ib32];
            const float dl    = d * (float)(2 * ((h >> 12) & 7) + 1);
            const float delta = (h & 0x8000) ? -SPARKINFER_IQ1S_DELTA : SPARKINFER_IQ1S_DELTA;
            const uint32_t idx =
                (uint32_t)b.qs[4 * ib32 + l] | (((uint32_t)(h >> (3 * l)) & 7u) << 8);
            float acc = 0.0f;
            if (GPACK) {
                uint32_t pw;
                if (GS) {
                    extern __shared__ uint16_t s_iq1s_grid[];
                    pw = s_iq1s_grid[idx];
                } else {
                    pw = iq1s_grid_p[idx];
                }
            #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int gj = ((int)(pw << (30 - 2 * j))) >> 30;
                    acc += dl * ((float)gj + delta) * xs[j];
                }
            } else {
                const uint64_t gw = iq1s_grid_c[idx];
            #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int8_t gj = (int8_t)((gw >> (8 * j)) & 0xffu);
                    acc += dl * ((float)gj + delta) * xs[j];
                }
            }
            if (side) uacc += acc; else gacc += acc;
        }
    }
    g_out = gacc;
    u_out = uacc;
}

// llama.cpp CPU vec_dot contracts for the pinned reference. The activation has
// already been converted to block_q8_K. Keep the integer dot integer until the
// whole 256-value block has been reduced, then apply the two block scales once;
// dequantising both sides to f32 first is mathematically close but not the same
// floating-point program.
__device__ __forceinline__ float block_dot_q8k(const BlockIQ2XS& b,
                                               const BlockQ8K& x, int lane) {
    const int ib32 = lane >> 2, sub = lane & 3;
    const uint8_t sc = b.scales[ib32];
    const int ls = 2 * (sub < 2 ? (sc & 0xf) : (sc >> 4)) + 1;
    const uint16_t q = b.qs[lane];
    const uint8_t* grid = (const uint8_t*)&c_iq2xs_grid[q & 511];
    const uint8_t signs = c_ksigns_iq2xs[q >> 9];
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int wq = (signs & c_kmask_iq2xs[j]) ? -(int)grid[j] : (int)grid[j];
        sumi += wq * (int)x.qs[8 * lane + j];
    }
    sumi *= ls;
    for (int off = 16; off > 0; off >>= 1)
        sumi += __shfl_down_sync(0xffffffff, sumi, off);
    if (lane != 0) return 0.0f;
    const float dw = __half2float(__ushort_as_half(b.d));
    return 0.125f * dw * x.d * (float)sumi;
}

__device__ __forceinline__ float block_dot_q8k(const BlockIQ1S& b,
                                               const BlockQ8K& x, int lane) {
    const int ib32 = lane >> 2, l = lane & 3;
    const uint16_t h = b.qh[ib32];
    const int ls = 2 * ((h >> 12) & 7) + 1;
    const int delta = (h & 0x8000) ? -1 : 1;
    const uint32_t idx =
        (uint32_t)b.qs[4 * ib32 + l] | (((uint32_t)(h >> (3 * l)) & 7u) << 8);
    const int8_t* grid = (const int8_t*)&iq1s_grid_c[idx];
    int sumi = 0, sumq = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int q8 = (int)x.qs[8 * lane + j];
        sumi += (int)grid[j] * q8;
        sumq += q8;
    }
    sumi *= ls;
    sumq *= ls * delta;
    for (int off = 16; off > 0; off >>= 1) {
        sumi += __shfl_down_sync(0xffffffff, sumi, off);
        sumq += __shfl_down_sync(0xffffffff, sumq, off);
    }
    if (lane != 0) return 0.0f;
    const float dw = __half2float(__ushort_as_half(b.d));
    return dw * x.d * ((float)sumi + SPARKINFER_IQ1S_DELTA * (float)sumq);
}

// gate/up GEMV + situ. One WARP per (selected expert k, output row j).
//
// It used to be one 32-thread BLOCK per row. sm_90 caps CTAs at 32 per SM, so a
// one-warp CTA can occupy at most 32 of the SM's 64 warp slots however much work is
// queued -- 50%, structurally. Eight warps per CTA lifts that to 62% here (48 registers
// is what bounds it, not the CTA shape any more).
//
// The arithmetic is untouched. The reduction was ALREADY warp-wide: BLOCK was 32, so
// block_sum's cross-warp stage summed exactly one partial and its shared round trip
// bought nothing. Folding with a plain shuffle over the same 32 lanes in the same
// butterfly order yields the identical float, and drops both __syncthreads() with it.
//
// The zero-fill for foreign experts is gone from here; the zeros are not. It moved to a
// cudaMemsetAsync over scratch before this launch, and is kept rather than dropped
// because it is what makes the combine's skip safe to rely on: moe_down_combine_kernel
// re-tests the band and never reads a foreign slot, so the zeros are an invariant no
// live path depends on today -- but scratch is reused across layers, so the day some
// path stops re-testing, an untouched slot hands it the previous layer's activations.
//
// What changed is the cost. At tp=8, 14 of every 16 selections are foreign, so this was
// 43,008 CTAs launched per MoE layer to store one scattered float each. One memset does
// the same job coalesced.
template <int WARPS_PER_CTA, bool XVEC, typename Blk, bool GPACK = false,
          bool GSMEM = false>
__global__ void moe_gate_up_situ_kernel(float* __restrict__ scratch,
                                        const float* __restrict__ x,
                                        const int* __restrict__ ids,
                                        const Blk* __restrict__ gate_exps,
                                        const Blk* __restrict__ up_exps,
                                        int latent, int ffn,
                                        float beta, float inv_beta,
                                        float lb, float inv_lb, int lb_active,
                                        int expert_begin, int n_local_experts,
                                        const float* __restrict__ w = nullptr,
                                        float weps = 0.0f) {
    k3_pdl_sync();
    // THE STAGE COPY MUST PRECEDE THE EARLY RETURN: a __syncthreads() below a
    // return deadlocks any tail block whose upper warps exited.
    if constexpr (GSMEM) {
        // P3b: the band test is CTA-UNIFORM (blockIdx.y only) — a foreign
        // expert's CTA exits WHOLE before paying the 4 KB stage + barrier, so no
        // thread can strand at __syncthreads. ~half of CTAs are foreign at eg=2;
        // staging for them was ~3 MB of dead shared writes + 768 wasted barriers
        // per layer — the likely reason IQ1S_SMEM measured exactly 0.00.
        {
            const int e_pre = ids[blockIdx.y] - expert_begin;
            if (e_pre < 0 || e_pre >= n_local_experts) return;
        }
        extern __shared__ uint16_t s_iq1s_grid[];
        for (int i = threadIdx.x; i < SPARKINFER_IQ1S_NGRID; i += WARPS_PER_CTA * 32)
            s_iq1s_grid[i] = iq1s_grid_p[i];
        __syncthreads();
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int k = blockIdx.y;                                  // which selected expert
    const int j = blockIdx.x * WARPS_PER_CTA + warp;           // which ffn output row
    if (j >= ffn) return;

    // EXPERT PARALLELISM. `ids` holds GLOBAL expert indices — every rank's router
    // sees the same replicated ffn_gate_inp and therefore selects the same top_k —
    // but this rank stores only experts [expert_begin, expert_begin+n_local).
    // Selections outside that band belong to another rank and contribute ZERO here;
    // the all-reduce that follows sums the bands back into the full top_k combine.
    const int e = ids[k] - expert_begin;
    if (e < 0 || e >= n_local_experts) return;   // memset already left this slot at 0
    // WEIGHT SKIP — must use the IDENTICAL test down_combine uses. Skipping here
    // leaves scratch[k] holding the PREVIOUS layer's activations (it is zeroed
    // once at allocation, not per layer), which would be read as live data. That
    // hazard is closed by construction because down_combine applies the same
    // `w[k] < weps` test and never reads a slot this kernel skipped. Covering
    // gate/up matters because it streams TWO expert tensors to down's one.
    if (weps > 0.0f && w != nullptr && w[k] < weps && k3_weps_depth_live()) {
        // Single evaluation per CTA here (k comes from the launch geometry), so the
        // depth check stays inline; the combine's per-candidate loop hoists it.
        if (g_k3_weps_count && threadIdx.x == 0) atomicAdd(&g_k3_weps_skips, 1ULL);
        return;
    }
    const int blocks_per_row = latent / 256;

    const Blk* g_row = gate_exps + (size_t)(e * ffn + j) * blocks_per_row;
    const Blk* u_row = up_exps   + (size_t)(e * ffn + j) * blocks_per_row;

    float gacc = 0.0f, uacc = 0.0f;
    // BPR14: scored K3 latent is 3584 → blocks_per_row is exactly 14, but it arrives
    // as a RUNTIME bound — `#pragma unroll 4` on a 14-trip runtime loop only peels
    // four at a time and leaves each divergent lattice gather alone on a dependent
    // chain. A compile-time 14 issues all fourteen LDG→gather→FMA chains back to
    // back (same pattern as BPR3 on down below). Ascending b preserved: bit-identical.
    if (blocks_per_row == 14) {
        // Software-pipeline + act-fuse: prefetch weight/act for b+1 while
        // contracting gate+up against one staged xs for b.
        Blk g_cur = g_row[0], u_cur = u_row[0];
        const float* x_cur = x;
#pragma unroll
        for (int b = 0; b < 13; ++b) {
            const Blk g_nxt = g_row[b + 1];
            const Blk u_nxt = u_row[b + 1];
            const float* x_nxt = x + (b + 1) * 256;
            float dg = 0.0f, du = 0.0f;
            block_dot_pair<XVEC, GPACK, GSMEM>(g_cur, u_cur, x_cur, lane, 32, dg, du);
            gacc += dg; uacc += du;
            g_cur = g_nxt; u_cur = u_nxt; x_cur = x_nxt;
        }
        {
            float dg = 0.0f, du = 0.0f;
            block_dot_pair<XVEC, GPACK, GSMEM>(g_cur, u_cur, x_cur, lane, 32, dg, du);
            gacc += dg; uacc += du;
        }
    } else {
#pragma unroll 4
        for (int b = 0; b < blocks_per_row; ++b) {
            const float* xb = x + b * 256;
            float dg = 0.0f, du = 0.0f;
            block_dot_pair<XVEC, GPACK, GSMEM>(g_row[b], u_row[b], xb, lane, 32, dg, du);
            gacc += dg; uacc += du;
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        gacc += __shfl_down_sync(0xffffffff, gacc, off);
        uacc += __shfl_down_sync(0xffffffff, uacc, off);
    }

    if (lane == 0) {
        // situ, identical to situ_kernel
        const float a  = beta * tanhf(gacc * inv_beta) * sigmoidf_(gacc);
        const float ub = lb_active ? (lb * tanhf(uacc * inv_lb)) : uacc;
        scratch[(size_t)k * ffn + j] = a * ub;
    }
}

// down GEMV + weighted combine. One block per output element; accumulates the
// top_k experts so the combine needs no second pass and no atomics.
// The top_k experts used to be walked SERIALLY by a single warp, one full GEMV after
// another, with a shared-memory reduction and two __syncthreads() between each.
//
// That is the starvation case, not merely an occupancy one. One warp per output element
// is latent = 3584 warps of work for the WHOLE GPU — 27 per SM on a 132-SM H200, below
// even the 32-warp ceiling a one-warp CTA imposes, so the machine cannot be filled no
// matter how the CTAs are shaped. Spreading the experts across the CTA's warps makes it
// 28,672 warps, which does fill it (100% occupancy at 32 registers), and takes the
// barrier count per output element from 32 down to 1.
//
// The two-stage shape is what keeps the combine bit-identical, and it has to preserve
// two things, not one.
//
// ORDER. Each warp parks its dot product in partial[k]; ONE thread then folds them in
// increasing k, skipping exactly the k the serial version skipped. Summing all top_k
// slots with zeros for the foreign ones would be equivalent for every real input and
// still wrong: it turns a -0.0f running total into +0.0f.
//
// ROUNDING. partial[k] holds the RAW dot product, unscaled, and the fold below does
// `total += w[k] * partial[k]` — the same expression the serial version ran, which nvcc
// contracts into a single FMA. Pre-scaling by w[k] on the warp side instead looks
// harmless and is not: it rounds the multiply before the add, splitting one fused
// operation into two rounded ones. That measured as a 8.4e-13 KLD divergence from main
// on the node — numerically nothing, but it is the difference between "bit-identical"
// and "very close", and only the first one is verifiable by byte comparison.
static inline float k3_moe_weps_host() {
    static const float v = [] {
        const char* e = std::getenv("SPARKINFER_K3_MOE_WEPS");
        // MEASURED at the scored 128k shape (engagement atomics off the hot path):
        // 0.04 +11%, 0.06 +17%, 0.08 +21%, 0.10 +24% over all-gates-off. 0.08 is
        // the default DELIBERATELY, not the curve's top: past it the dropped tail's
        // perturbation bound stops being small. Depth-gated by k3_weps_depth_live —
        // ungated, the shallow-depth cost is real and measured (KL 1.1e-1 at
        // ctx2048, over the accuracy bar) because the router's weight distribution
        // is still concentrated there; past the gate depth it has flattened and the
        // dropped terms are the sub-8% tail of a unit-sum combine. =0 restores
        // exact main at every depth.
        return e ? (float)atof(e) : 0.08f;
    }();
    return v;
}

template <int WARPS_PER_CTA, bool XVEC, typename Blk, bool GPACK = false,
          bool RB = false>
__global__ void moe_down_combine_kernel(float* __restrict__ out,
                                        const float* __restrict__ scratch,
                                        const int* __restrict__ ids,
                                        const float* __restrict__ w,
                                        const Blk* __restrict__ down_exps,
                                        int latent, int ffn, int top_k,
                                        int expert_begin, int n_local_experts,
                                        float weps = 0.0f) {
    k3_pdl_sync();
    const int o = blockIdx.x;                 // output element in [0, latent)
    if (o >= latent) return;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int blocks_per_row = ffn / 256;
    extern __shared__ float partial[];        // top_k floats, then ids/w stage
    // Stage ids[] and w[] into shared BEFORE the dot loop: the loads overlap the
    // dots for free, and the fold below stops paying a serial global-latency
    // chain on one thread while 255 wait at the barrier. Same values, same
    // ascending-k fold, same skip set — a scheduling change, not a numeric one.
    int*   s_ids = (int*)(partial + top_k);
    float* s_w   = (float*)(s_ids + top_k);
    if (threadIdx.x < top_k) {
        s_ids[threadIdx.x] = ids[threadIdx.x];
        s_w[threadIdx.x]   = w[threadIdx.x];
    }

    // Stage must be VISIBLE before anyone reads it.
    __syncthreads();

    // BALANCE THE LIVE EXPERTS ACROSS WARPS (RB). The static k = warp, warp+8
    // map hands warp w experts {w, w+8} regardless of which are LOCAL; at tp=8
    // each k is local with p~0.5, so P(some warp draws 2 live while another
    // draws 0) ~ 0.90 — and the phase costs the MAX over warps, not the mean.
    // Under RB every thread builds the same compacted live list (uniform, from
    // shared) and warps stride THAT. Which warp computes partial[k] changes;
    // the per-k dot, its shuffle fold, and the ascending-k epilogue do not —
    // bit-identical. SPARKINFER_K3_MOE_REBAL=1 selects it; default OFF.
    // V6 fix: the first draft used `int live[32]` indexed at runtime — nvcc
    // demotes a dynamically-indexed local array to per-thread STACK (local-memory
    // traffic on every one of 3584 CTAs), and all 256 threads built the identical
    // list. Shared, built once by thread 0 behind the stage barrier, read by all.
    __shared__ int s_live[32]; __shared__ int s_nlive;
    if (RB) {
        if (threadIdx.x == 0) {
            int n = 0;
            for (int k = 0; k < top_k; ++k) {
                const int e0 = s_ids[k] - expert_begin;
                if (e0 >= 0 && e0 < n_local_experts) s_live[n++] = k;
            }
            s_nlive = n;
        }
        __syncthreads();
    }
    // SPARKINFER_K3_MOE_WEPS: skip experts whose renormalised weight is below
    // this threshold. Selection uses BIASED scores while the carried weight is
    // the UNBIASED sigmoid, so a selected expert can contribute ~nothing and
    // still stream its full expert rows. NOT bit-identical — it drops terms of
    // magnitude < eps from an ascending sum whose total is 1.0, so the relative
    // output perturbation is bounded by (number skipped) x eps.
    // Hoisted ONCE per CTA: inside the loop this was a same-address global load per
    // sub-threshold candidate — ~2.6M redundant L2 transactions per token at 128k,
    // measured as ~0.4 tok/s of the threshold's own gain.
    const bool wlive = weps > 0.0f && k3_weps_depth_live();
    const int kmax = RB ? s_nlive : top_k;
    for (int li = warp; li < kmax; li += WARPS_PER_CTA) {
        const int k = RB ? s_live[li] : li;
        if (wlive && s_w[k] < weps) {
            if (lane == 0) {
                partial[k] = 0.0f;
                if (g_k3_weps_count) atomicAdd(&g_k3_weps_skips, 1ULL);
            }
            continue;
        }
        // Same band test as the gate/up pass. Skipping here is what keeps the read
        // in bounds: down_exps holds n_local experts, so indexing it by a GLOBAL id
        // would run off the end of the allocation for every rank but rank 0 — and
        // read whatever the allocator put there rather than faulting.
        // s_ids, not ids: the stage at the top exists for exactly this read — the
        // global re-read here was ~5.3M redundant dependent loads per token.
        const int e = s_ids[k] - expert_begin;
        if (e < 0 || e >= n_local_experts) continue;
        const Blk* d_row = down_exps + (size_t)(e * latent + o) * blocks_per_row;
        const float* act = scratch + (size_t)k * ffn;
        float acc = 0.0f;
        // BPR3: at the shipped 2-D shard ffn=768 so blocks_per_row is exactly 3,
        // but it is a RUNTIME bound here — each iteration is LDG -> dependent
        // iq1s gather -> FMAs with ONE chain in flight. (This is why DUNROLL did
        // nothing: `#pragma unroll 4` on a 3-trip runtime loop never enters the
        // unrolled body.) A compile-time 3 issues all three chains back to back.
        // Ascending b preserved; unrolling does not reassociate: bit-identical.
        if (blocks_per_row == 3) {
            // Soft-pipeline BPR=3 (scored K3 down): prefetch weight/act for
            // b+1 while contracting b. Same three FMAs, ascending-b order.
            Blk d_cur = d_row[0];
            const float* x_cur = act;
#pragma unroll
            for (int b = 0; b < 2; ++b) {
                const Blk d_nxt = d_row[b + 1];
                const float* x_nxt = act + (b + 1) * 256;
                acc += block_dot<XVEC, GPACK>(d_cur, x_cur, lane, 32);
                d_cur = d_nxt;
                x_cur = x_nxt;
            }
            acc += block_dot<XVEC, GPACK>(d_cur, x_cur, lane, 32);
        } else if (blocks_per_row == 6) {
#pragma unroll
            for (int b = 0; b < 6; ++b)
                acc += block_dot<XVEC, GPACK>(d_row[b], act + b * 256, lane, 32);
        } else if (blocks_per_row == 7) {
            // Soft-pipe BPR=7 (half-width / alt shapes).
            Blk d_cur = d_row[0];
            const float* x_cur = act;
#pragma unroll
            for (int b = 0; b < 6; ++b) {
                const Blk d_nxt = d_row[b + 1];
                const float* x_nxt = act + (b + 1) * 256;
                acc += block_dot<XVEC, GPACK>(d_cur, x_cur, lane, 32);
                d_cur = d_nxt;
                x_cur = x_nxt;
            }
            acc += block_dot<XVEC, GPACK>(d_cur, x_cur, lane, 32);
        } else if (blocks_per_row == 12) {
            Blk d_cur = d_row[0];
            const float* x_cur = act;
#pragma unroll
            for (int b = 0; b < 11; ++b) {
                const Blk d_nxt = d_row[b + 1];
                const float* x_nxt = act + (b + 1) * 256;
                acc += block_dot<XVEC, GPACK>(d_cur, x_cur, lane, 32);
                d_cur = d_nxt;
                x_cur = x_nxt;
            }
            acc += block_dot<XVEC, GPACK>(d_cur, x_cur, lane, 32);
        } else if (blocks_per_row == 14) {
            // Gate/up latent width reused as down on some alternate configs.
            Blk d_cur = d_row[0];
            const float* x_cur = act;
#pragma unroll
            for (int b = 0; b < 13; ++b) {
                const Blk d_nxt = d_row[b + 1];
                const float* x_nxt = act + (b + 1) * 256;
                acc += block_dot<XVEC, GPACK>(d_cur, x_cur, lane, 32);
                d_cur = d_nxt;
                x_cur = x_nxt;
            }
            acc += block_dot<XVEC, GPACK>(d_cur, x_cur, lane, 32);
        } else
        for (int b = 0; b < blocks_per_row; ++b)
            acc += block_dot<XVEC, GPACK>(d_row[b], act + b * 256, lane, 32);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        if (lane == 0) partial[k] = acc;      // RAW; w[k] is applied in the fold
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            const int e = s_ids[k] - expert_begin;
            if (e < 0 || e >= n_local_experts) continue;
            total += s_w[k] * partial[k];   // one FMA, exactly as the serial version
        }
        out[o] = total;
    }
}

template <typename Blk>
__global__ void moe_gate_up_situ_q8k_kernel(float* __restrict__ scratch,
                                            const BlockQ8K* __restrict__ x,
                                            const int* __restrict__ ids,
                                            const Blk* __restrict__ gate_exps,
                                            const Blk* __restrict__ up_exps,
                                            int latent, int ffn,
                                            float beta, float inv_beta,
                                            float lb, float inv_lb, int lb_active,
                                            int expert_begin, int n_local_experts) {
    k3_pdl_sync();
    const int k = blockIdx.y;
    const int j = blockIdx.x;
    if (j >= ffn) return;
    const int e = ids[k] - expert_begin;
    if (e < 0 || e >= n_local_experts) {
        if (threadIdx.x == 0) scratch[(size_t)k * ffn + j] = 0.0f;
        return;
    }
    const int blocks_per_row = latent / 256;
    const Blk* g_row = gate_exps + (size_t)(e * ffn + j) * blocks_per_row;
    const Blk* u_row = up_exps   + (size_t)(e * ffn + j) * blocks_per_row;
    float gacc = 0.0f, uacc = 0.0f;
    for (int b = 0; b < blocks_per_row; ++b) {
        gacc += block_dot_q8k(g_row[b], x[b], threadIdx.x);
        uacc += block_dot_q8k(u_row[b], x[b], threadIdx.x);
    }
    if (threadIdx.x == 0) {
        const float a = beta * tanhf(gacc * inv_beta) * sigmoidf_(gacc);
        const float ub = lb_active ? (lb * tanhf(uacc * inv_lb)) : uacc;
        scratch[(size_t)k * ffn + j] = a * ub;
    }
}

template <typename Blk>
__global__ void moe_down_combine_q8k_kernel(float* __restrict__ out,
                                            const BlockQ8K* __restrict__ acts,
                                            const int* __restrict__ ids,
                                            const float* __restrict__ w,
                                            const Blk* __restrict__ down_exps,
                                            int latent, int ffn, int top_k,
                                            int expert_begin, int n_local_experts) {
    k3_pdl_sync();
    const int o = blockIdx.x;
    if (o >= latent) return;
    const int blocks_per_row = ffn / 256;
    float total = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        const int e = ids[k] - expert_begin;
        if (e < 0 || e >= n_local_experts) continue;
        const Blk* d_row = down_exps + (size_t)(e * latent + o) * blocks_per_row;
        const BlockQ8K* act = acts + (size_t)k * blocks_per_row;
        float acc = 0.0f;
        for (int b = 0; b < blocks_per_row; ++b)
            acc += block_dot_q8k(d_row[b], act[b], threadIdx.x);
        if (threadIdx.x == 0) total += w[k] * acc;
    }
    if (threadIdx.x == 0) out[o] = total;
}

// ---------------------------------------------------------------------------
// 5. MLA output gate
// ---------------------------------------------------------------------------

__global__ void mla_gate_out_kernel(float* __restrict__ out,
                                    const float* __restrict__ attn_out,
                                    const float* __restrict__ gate_proj, int64_t n) {
    k3_pdl_sync();
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = attn_out[i] * sigmoidf_(gate_proj[i]);
}

// ---------------------------------------------------------------------------
// 7. KDA decay gate (lower_bound form)
// ---------------------------------------------------------------------------
// One block per head; head_dim threads. A[h] broadcasts across the head_dim channels.

__global__ void kda_decay_gate_kernel(float* __restrict__ out,
                                      const float* __restrict__ g_raw,
                                      const float* __restrict__ A,
                                      int head_dim, float lower_bound) {
    k3_pdl_sync();
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    if (d >= head_dim) return;
    const float Ah = A[h];
    const float gr = g_raw[(size_t)h * head_dim + d];
    // g = lb * sigmoid(-(A * g_raw))
    out[(size_t)h * head_dim + d] = lower_bound * sigmoidf_(-(Ah * gr));
}

// ---------------------------------------------------------------------------
// 8. L2-norm over heads (+ optional scale)
// ---------------------------------------------------------------------------

template <int BLOCK>
__global__ void l2_norm_heads_kernel(float* __restrict__ out,
                                     const float* __restrict__ x,
                                     int head_dim, float scale, float eps) {
    k3_pdl_sync();
    const int h = blockIdx.x;
    const float* xh = x + (size_t)h * head_dim;
    float* oh = out + (size_t)h * head_dim;

    float ss = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += BLOCK)
        ss += xh[d] * xh[d];
    __shared__ float shm[BLOCK / 32 + 1];
    ss = block_sum<BLOCK>(ss, shm);
    const float inv = scale * rsqrtf(ss + eps);

    for (int d = threadIdx.x; d < head_dim; d += BLOCK)
        oh[d] = xh[d] * inv;
}

// KDA short-conv x3 + the q/k L2 norms, in ONE launch.
//
// Five dependent launches per KDA layer, on all 69 of them, for ~330 KB of work: three
// convs at (d_inner/256) = 6 blocks each, then two l2_norm_heads at n_head = 12 blocks
// on a 132-SM part. The three convs are mutually independent (separate state, weight,
// input and output pointers), and the L2 norm for q and k reduces over exactly the
// head_dim channels the matching conv just wrote — so the whole group is one kernel
// with grid (n_head, 3).
//
// BIT-IDENTICAL, and the mapping is what makes it so:
//   * the conv is per-channel and untouched — same window order, same wc[d_conv-1] on
//     the current sample, same silu after, same left-shift of the history.
//   * l2_norm_heads_kernel<128> strides `for (d = threadIdx.x; d < head_dim; d += BLOCK)`
//     at BLOCK = 128 and K3's kda_head_dim = 128, so it already ran exactly one element
//     per thread over one head. This keeps that: thread d owns channel d of head h, and
//     block_sum<BLOCK> folds the same 128 partials through the same tree.
//   * v is written without a norm, exactly as before.
// Threads past head_dim contribute +0.0f to the reduction, which is what the strided
// loop did for them too.
template <int BLOCK>
__global__ void kda_conv_l2_fused_kernel(float* __restrict__ out_q,
                                         float* __restrict__ out_k,
                                         float* __restrict__ out_v,
                                         float* __restrict__ st_q,
                                         float* __restrict__ st_k,
                                         float* __restrict__ st_v,
                                         const float* __restrict__ x_q,
                                         const float* __restrict__ x_k,
                                         const float* __restrict__ x_v,
                                         const float* __restrict__ w_q,
                                         const float* __restrict__ w_k,
                                         const float* __restrict__ w_v,
                                         int d_conv, int head_dim,
                                         float q_scale, float eps) {
    k3_pdl_sync();
    const int h  = blockIdx.x;
    const int sl = blockIdx.y;            // 0 = q, 1 = k, 2 = v
    const int d  = threadIdx.x;

    float* out        = (sl == 0) ? out_q : (sl == 1) ? out_k : out_v;
    float* state      = (sl == 0) ? st_q  : (sl == 1) ? st_k  : st_v;
    const float* x    = (sl == 0) ? x_q   : (sl == 1) ? x_k   : x_v;
    const float* wgt  = (sl == 0) ? w_q   : (sl == 1) ? w_k   : w_v;

    const int c = h * head_dim + d;       // global channel

    float v = 0.0f;
    if (d < head_dim) {
        float* st       = state + (size_t)c * (d_conv - 1);
        const float* wc = wgt + (size_t)c * d_conv;
        float acc = 0.0f;
#pragma unroll 4
        for (int t = 0; t < d_conv - 1; ++t) acc += st[t] * wc[t];
        const float xc = x[c];
        acc += xc * wc[d_conv - 1];
        v = acc * sigmoidf_(acc);         // silu AFTER the convolution
#pragma unroll 4
        for (int t = 0; t < d_conv - 2; ++t) st[t] = st[t + 1];
        st[d_conv - 2] = xc;
    }

    if (sl == 2) {                        // v is not normalised
        if (d < head_dim) out[c] = v;
        return;
    }

    // block_sum contains __syncthreads(), so every thread must reach it.
    __shared__ float shm[BLOCK / 32 + 1];
    const float ss  = block_sum<BLOCK>(v * v, shm);
    const float sc  = (sl == 0) ? q_scale : 1.0f;
    const float inv = sc * rsqrtf(ss + eps);
    if (d < head_dim) out[c] = v * inv;
}

// ---------------------------------------------------------------------------
// 9. MLA absorb Q
// ---------------------------------------------------------------------------
// One block per (head, kv_lora-row). Thread-reduce over qk_nope, then thread 0
// writes the absorbed slot and (once per head) copies q_pe into the tail.

template <int BLOCK>
__global__ void mla_absorb_q_kernel(float* __restrict__ out,
                                    const float* __restrict__ q_nope,
                                    const float* __restrict__ q_pe,
                                    const float* __restrict__ wk_b,
                                    int qk_nope, int kv_lora, int rope_dim) {
    k3_pdl_sync();
    const int h = blockIdx.y;
    const int r = blockIdx.x;   // kv_lora index
    if (r >= kv_lora) return;

    const float* qh = q_nope + (size_t)h * qk_nope;
    const float* wh = wk_b + (size_t)h * (size_t)qk_nope * kv_lora;
    // wk_b layout: [qk_nope, kv_lora, n_head] with qk_nope fastest →
    // column r starts at offset r * qk_nope.
    const float* wr = wh + (size_t)r * qk_nope;

    float acc = 0.0f;
    for (int d = threadIdx.x; d < qk_nope; d += BLOCK)
        acc += wr[d] * qh[d];
    __shared__ float shm[BLOCK / 32 + 1];
    acc = block_sum<BLOCK>(acc, shm);

    float* oh = out + (size_t)h * (kv_lora + rope_dim);
    if (threadIdx.x == 0) oh[r] = acc;

    // Copy q_pe once per head (blockIdx.x == 0 owns it).
    if (r == 0) {
        const float* pe = q_pe + (size_t)h * rope_dim;
        for (int d = threadIdx.x; d < rope_dim; d += BLOCK)
            oh[kv_lora + d] = pe[d];
    }
}

// Block-wide max into lane 0 of warp 0, then broadcast via shared memory.
template <int BLOCK>
__device__ __forceinline__ float block_max(float v, float* shm) {
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) shm[warp] = v;
    __syncthreads();
    constexpr int NWARP = BLOCK / 32;
    if (threadIdx.x == 0) {
        float m = shm[0];
        for (int w = 1; w < NWARP; ++w) m = fmaxf(m, shm[w]);
        shm[NWARP] = m;
    }
    __syncthreads();
    return shm[NWARP];
}

// ---------------------------------------------------------------------------
// 10. MLA NoPE decode attention
// ---------------------------------------------------------------------------
// One block per head. Scores vs shared K-cache (MQA), softmax, attend over the
// leading kv_lora of K as V, then decompress with wv_b.
//
// THE SHARED-MEMORY FOOTPRINT MUST NOT SCALE WITH CONTEXT.
//
// This kernel previously staged the whole score vector — one float per cached
// token — in dynamic shared memory, so the launch asked for
// (n_ctx + kv_lora + BLOCK/32 + 1) * 4 bytes. With K3's kv_lora 512 that crosses
// the 48 KB default dynamic-shared limit at n_ctx = 11,767, and the 227 KB
// Hopper opt-in ceiling (which this file never requested) at 57,573. Past that
// the launch returns cudaErrorInvalidValue.
//
// Nothing catches it. mla_decode_attn_f32 returns void, the forward does not
// poll cudaGetLastError, and s.mla_attn_out is a reused scratch buffer — so a
// failed launch leaves the PREVIOUS layer's attention output in place and the
// model keeps emitting fluent text. Every MLA layer past ~11.7k context silently
// contributes a stale tensor. That is the same failure class as the 1-indexed
// full_attn_layers trap and the KDA decay axis: wrong output, not an error.
//
// The numeric test could not see it either — test_mla_decode_attn runs n_ctx 48.
//
// So the softmax is now ONLINE (running max + running exp-sum, rescaled per
// tile) over tiles of kMlaCtxTile tokens, and the score tile is the only
// context-dependent buffer. Shared memory is
// (key_length + kv_lora + kMlaCtxTile + BLOCK/32 + 1) floats — 4.8 KB at K3's
// real dims, INDEPENDENT of n_ctx, so the kernel launches at 1M context.
//
// Two access patterns changed with it, both because the tiling makes the better
// one natural rather than as a separate optimisation:
//
//   - Scores are now ONE WARP PER TOKEN with lanes striding over d. The old loop
//     gave one THREAD a whole 576-float cache row, so the 32 lanes of a warp
//     read addresses 2304 B apart — 32 distinct sectors per load instruction,
//     4 useful bytes each.
//   - q is staged in shared memory once instead of being re-read from global by
//     every thread for every token it scores.
//
// The latent accumulation keeps the old r-major loop, which was already
// coalesced; it just gains the online-softmax rescale.
constexpr int kMlaCtxTile = 128;

// Context-split tunables. kMlaSplitMinCtx is the slice length below which splitting is
// not worth the combine pass.
constexpr int kMlaSplitMinCtx = 4096;
constexpr int kMlaMaxSplits   = 64;

// THE SPLIT CAP IS A BUDGET ON n_head * splits, NOT A CONSTANT ON splits.
//
// The grid is (n_head/hpb, splits) and the partials scratch is
// n_head * splits * kv_lora, so both the parallelism and the memory scale with
// the PRODUCT. A flat cap on `splits` therefore does the wrong thing the moment
// n_head stops being 96.
//
// It stopped being 96. Head-sharding MLA across 8 ranks gives each rank 12
// heads, so hpb=12 collapses the head axis to ONE group and the grid becomes
// 1 x 64 = 64 blocks on a 132-SM H200 — under half the machine idle, while the
// heuristic three lines below was asking for 528. That is why sharding measured
// 2.24x instead of the 8x the work reduction implied: the work went away and so
// did the parallelism.
//
// Budgeting the product fixes both ends at once and is exactly memory-neutral:
//
//     n_head 96 (replicated)  ->  64 splits   6144 pairs   unchanged, bit-for-bit
//     n_head 12 (sharded)     -> 512 splits   6144 pairs   same scratch, 8x grid
//
// The unsharded path keeps the cap it always had, so this cannot regress the
// replicated build — it only spends, on parallelism, the scratch that sharding
// already freed.
constexpr int kMlaSplitBudget = 96 * kMlaMaxSplits;   // 6144 (head, slice) pairs

static inline int k3_mla_max_splits(int n_head) {
    if (n_head <= 0) return 1;

    // SPARKINFER_K3_MLA_SPLIT_CAP pins the cap, and exists because the accuracy
    // gate cannot see this change. The eval scores against llama.cpp on a SHORT
    // reference prompt while timing at 128k, and splitting only engages above
    // kMlaSplitMinCtx — so the gate runs splits=1 and reports a byte-identical
    // KLD whatever this returns. Comparing two caps at long context on one
    // binary is the only thing that actually exercises the wider combine.
    static const int pinned = [] {
        const char* e = std::getenv("SPARKINFER_K3_MLA_SPLIT_CAP");
        if (!e) return 0;
        const int v = std::atoi(e);
        return v > 0 ? v : 0;
    }();
    if (pinned > 0) return pinned;

    const int by_budget = kMlaSplitBudget / n_head;
    // Never below the historical cap for a head count that already fit it, and
    // never so high that the combine's per-slice loop (and its shared-memory
    // slot per slice) becomes the new cost.
    return std::min(1024, std::max(1, by_budget));
}

// ---------------------------------------------------------------------------
// MLA IS MQA, AND ONE BLOCK PER HEAD READS THE KV CACHE n_head TIMES.
// ---------------------------------------------------------------------------
// k_cache is indexed [t][key_length] with NO head axis: MLA keeps one compressed
// latent per position and all 96 query heads attend over the same rows. The kernels
// above give each head its own block, so at the scored context every one of those
// blocks streams the entire cache independently:
//
//   one MLA layer at 131,072 ctx   131072 * 576 * 4 B            =  302 MB
//   x 96 heads (score pass reads 576/row, latent pass 512 more)  = 54.8 GB
//   x 24 MLA layers                                              =  1.3 TB per token
//
// 1.3 TB per token per rank against an H200's 4.8 TB/s is ~274 ms of pure cache
// traffic for a token that measures 220 ms, so L2 is already covering more than half
// of it — the kernel is not slow, it is asking for the same bytes 96 times.
//
// kMlaHeadsPerBlock fixes the ratio rather than the bandwidth. One block takes HPB
// heads, stages their absorbed queries in shared memory, and every k_cache element it
// loads is consumed HPB times: once per head in the score pass, once per head in the
// latent accumulation. Traffic falls by HPB with no change to what is computed.
//
// WHY 12, WHICH IS NOT THE LARGEST HPB THAT FITS. Traffic falls as 1/HPB but so does
// concurrency: the kernel is latency-bound, not compute-bound, so what it can actually
// pull from memory is set by how many loads are in flight, i.e. by blocks per SM. HPB
// costs registers (RSLOTS * HPB latent accumulators live across the whole slice, plus
// HPB * TPW score accumulators) and shared (HPB * key_length staged queries), and both
// budgets are per SM. Measured at 131,072 ctx on 8x H200, tp=8, UD-IQ1_S, 32 tokens,
// median ms/token, against main at 221.4:
//
//   HPB   regs   blocks/SM   ms/token
//     8     96       2         141.1
//    12    125       2         135.7
//    16    148       1         149.9     <- 148 regs is over 65536/(256*2)
//
// 16 halves the traffic again and loses, because 256 * 149 registers no longer fits a
// second block and the SM drops to 8 warps. 12 is the largest batch that still leaves
// two blocks resident. It divides 96 exactly; k3_mla_heads_per_block falls back through
// 8, 4 and finally 1 (the per-head kernel) for shapes where it does not.
constexpr int kMlaHeadsPerBlock = 12;

// Latent accumulator slots per thread: ceil(kv_lora / BLOCK), held in registers rather
// than shared. r is fixed per thread for the whole slice, so the accumulator never
// needs to be visible to another thread until the slice ends — and keeping it out of
// shared is what leaves room for HPB staged queries.
constexpr int kMlaMaxRSlots = 4;

// Tokens a warp scores per pass over the staged queries.
//
// Staging the queries moves the cost rather than removing it unless each staged value
// is used more than once. Scoring one token at a time reads HPB floats from shared per
// 128 bytes of cache, and 32 banks x 4 B/clk caps that at ~3.7 TB/s at HPB 8 — under
// the card's HBM rate, so the shared memory that saves the traffic becomes the thing
// capping it. Scoring TPW tokens per pass divides that ratio by TPW.
//
// Measured at 131,072 ctx, HPB 12 (8x H200, tp=8, UD-IQ1_S, 32 tokens): TPW 2 gives
// 143.2 ms/token, TPW 4 gives 135.7. The score dot product is unchanged — same lanes,
// same d order, same shuffle tree — so this is purely a scheduling change.
constexpr int kMlaTokensPerWarp = 4;

// Scratch for the partials, grown on demand and reused. Sized n_head * splits *
// (kv_lora + 2) floats — 6.3 MB at K3's dims with 32 splits, against ~140 GB of weights,
// so a one-off allocation is cheaper than threading a buffer through every caller.
//
// PER DEVICE, not one global. K3 runs tensor-parallel across 8 ranks, each owning its
// own device and its own thread. A single static pointer is allocated by whichever rank
// arrives first and then dereferenced by the other seven on devices it does not belong
// to — an illegal access that surfaces later and elsewhere, as a sticky error inside the
// next NCCL all-reduce. A single-device numeric test cannot catch that by construction.
//
// Each slot is touched by exactly one rank thread (a rank owns its device for the whole
// run), so the slots need no lock; the array itself is only ever read by index.
static constexpr int kMlaMaxDevices = 16;
static float* g_mla_merged  [kMlaMaxDevices] = {nullptr};
static float* g_mla_part_acc[kMlaMaxDevices] = {nullptr};
static float* g_mla_part_ml [kMlaMaxDevices] = {nullptr};
static size_t g_mla_part_cap[kMlaMaxDevices] = {0};

static bool k3_mla_split_scratch(int n_head, int kv_lora, int* dev_out) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    if (dev < 0 || dev >= kMlaMaxDevices) return false;   // fall back to the un-split path
    *dev_out = dev;

    // Sized with the same budget the launcher caps `splits` with. If these two
    // ever disagreed the kernel would write partials past the allocation, so
    // both go through k3_mla_max_splits() and neither hardcodes a cap.
    const int    ms   = k3_mla_max_splits(n_head);
    const size_t need = (size_t)n_head * ms * (size_t)kv_lora;
    if (need <= g_mla_part_cap[dev]) return true;
    cudaFree(g_mla_part_acc[dev]);
    cudaFree(g_mla_part_ml [dev]);
    g_mla_part_acc[dev] = nullptr; g_mla_part_ml[dev] = nullptr; g_mla_part_cap[dev] = 0;
    if (cudaMalloc((void**)&g_mla_part_acc[dev], need * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc((void**)&g_mla_part_ml[dev],
                   (size_t)n_head * ms * 2 * sizeof(float)) != cudaSuccess) {
        cudaFree(g_mla_part_acc[dev]); g_mla_part_acc[dev] = nullptr; return false;
    }
    // Merged latent, one row per head. Allocated on the SAME path as the partials so it
    // is covered by the same pre-warm and can never cudaMalloc inside a stream capture.
    cudaFree(g_mla_merged[dev]); g_mla_merged[dev] = nullptr;
    if (cudaMalloc((void**)&g_mla_merged[dev],
                   (size_t)n_head * (size_t)kv_lora * sizeof(float)) != cudaSuccess) {
        cudaFree(g_mla_part_acc[dev]); g_mla_part_acc[dev] = nullptr;
        cudaFree(g_mla_part_ml [dev]); g_mla_part_ml [dev] = nullptr;
        return false;
    }
    g_mla_part_cap[dev] = need;
    return true;
}

// A block's dynamic shared allocation without cudaFuncSetAttribute. Deliberately the
// DEFAULT and not the 227 KB Hopper opt-in: an opt-in that a device refuses leaves a
// silent fallback whose only symptom is a slower run, and the whole reason this file
// stopped staging the score vector was a launch failure nothing polled for.
constexpr size_t kMlaShmBudget = 48u * 1024u;

// Slice-count targets for the head-batched grid.
//
// kMlaBlocksPerSm is what the batched kernel ACTUALLY achieves at kMlaHeadsPerBlock
// (125 registers x 256 threads leaves room for two), not a wish. Asking for more
// slices than can be resident does not add parallelism — the extra blocks queue — but
// it does add partials for mla_decode_combine_kernel to merge and for the scratch to
// carry, so overshooting is pure cost. This one is a REGISTER fact and it is
// independent of n_head, so head-sharding does not touch it: see the HPB table above,
// where 16 heads at 148 regs drops to one block per SM and loses 10%.
constexpr int kMlaBlocksPerSm = 2;

// THE SLICE FLOOR IS DERIVED FROM hpb AND THE DIMS, NOT PICKED.
//
// This used to be `kMlaMinSliceLen = 1024`, a flat token count, and it silently
// became the binding term the moment the shard policy moved. The arithmetic:
// `splits` is min(fill target, n_ctx / floor), so the floor sets the CONTEXT AT
// WHICH THE FILL TARGET BECOMES REACHABLE AT ALL —
//
//     n_ctx_to_fill = ceil(kMlaBlocksPerSm * sm / groups) * floor,  groups = n_head/hpb
//
// At n_head 96 that is 33 * 1024 = 33,792 tokens, comfortably under the scored
// context, so `fill` won and the floor was never exercised. Head-sharding took
// groups 8 -> 1 (PR #63, which changed neither this constant nor the target
// beside it), which multiplied the threshold by exactly 8 to 270,336 — and
// stranded the scored 131,072 inside the new gap at 128 blocks, 0.97 per SM,
// against the 2 per SM the line above declares. A flat token count cannot track
// hpb, so it re-breaks on every shard-policy change. Derive it instead.
//
// WHAT A SLICE ACTUALLY COSTS FOR EXISTING, in DRAM, per block:
//
//   partials out   HPB * (kv_lora + 2) floats   one row per head, at the end of
//                                                mla_decode_attn_hbatch_kernel
//   combine reads  HPB *  kv_lora      floats   mla_decode_combine_kernel reads them
//                                                straight back
//   staged q       HPB *  key_length   floats   the s_q stage, but all of q is 27 KB
//                                                and every block reads it — L2, not
//                                                DRAM; it is startup latency, and it
//                                                amortises at chunk >= HPB tokens
//
// and what it does with it: key_length floats of k_cache per token, in the score
// phase. The latent pass re-reads the first kv_lora floats of the SAME row, inside
// the same tile, so that one is L2 too. Hence the tax a slice pays is
//
//   tax(slice) = 2 * hpb * kv_lora / (key_length * slice)
//
// — note that n_head, n_ctx and splits ALL cancel. The tax is a function of the
// SLICE LENGTH alone, which is why a slice-length floor is the right shape of rule
// and why its value must come from hpb and the dims.
//
// WHAT THE TAX BUDGET IS SET FROM. Measured at 131,072 ctx, n_head 12 (8x H200,
// tp=8, one binary per point, 32 tokens), against 54.53 ms/token at the 1024 floor:
//
//   floor  slice  grid   blk/SM   tax     ms/token
//   1024   1024    128     0.97   2.08%     54.53
//    512    512    256     1.94   4.17%     52.44   <- +3.98%
//    256    497    264     2.00   4.29%     52.23   <- +4.42%, and 0.40% over the row
//                                                      above is under the 2% bar
//
// So 4.17% of tax bought a 2.06x fill and won. That is the largest tax on this
// curve the measurement actually validates — the 256 row differs by 0.40%, which is
// noise, and it buys the last 8 blocks by cutting six OTHER cells of the shard/
// context space to 2-tile slices at an 8.3% tax nothing has measured. Budget the
// tax at the point that was measured and let the dims place the floor.
constexpr int kMlaMaxSliceTaxPct = 5;

// Rounded UP to whole kMlaCtxTile, because the tile is the work quantum: the slice
// loop runs ceil(chunk / kMlaCtxTile) times and each iteration pays three
// __syncthreads, two rounds of HPB-heads-over-NWARP-warps softmax and RSLOTS*HPB
// rescales whatever tn is, so a slice that is not a whole number of tiles pays a full
// tile for a partial one. Below NWARP * kMlaTokensPerWarp = 32 tokens the score phase
// leaves whole warps with no work at all. One tile is the hard floor; the tax budget
// decides how many.
//
//   hpb 12 -> 427 -> 512 tokens (4 tiles, 4.17%)   <- K3 sharded and replicated
//   hpb  8 -> 285 -> 384 tokens (3 tiles, 3.70%)
//   hpb  4 -> 143 -> 256 tokens (2 tiles, 2.78%)
//
// Fewer heads per block stage less q and write fewer partials, so they genuinely
// amortise sooner — the flat 1024 was 2x conservative at hpb 12 and 4x at hpb 4.
static constexpr int k3_mla_min_slice_len(int hpb, int key_length, int kv_lora) {
    if (hpb <= 0 || key_length <= 0 || kv_lora <= 0) return kMlaCtxTile;
    const long long den = (long long)key_length * kMlaMaxSliceTaxPct;
    const long long raw = (2ll * hpb * kv_lora * 100 + den - 1) / den;
    const long long tiles = (raw + kMlaCtxTile - 1) / kMlaCtxTile;
    return (int)((tiles < 1 ? 1 : tiles) * kMlaCtxTile);
}

// Pin the derivation to the shape it was measured at. If kMlaCtxTile, the tax budget
// or K3's MLA dims move, this fails the build rather than silently re-tuning the
// scored config the way the flat constant did.
static_assert(k3_mla_min_slice_len(12, 576, 512) == 4 * kMlaCtxTile,
              "hpb 12 at K3's 576/512 must floor at 512 tokens — the measured point");
static_assert(k3_mla_min_slice_len(1, 1, 1 << 20) >= kMlaCtxTile,
              "the floor is never below one tile");

// Largest head batch that divides n_head and whose staged queries fit the default
// shared budget. 1 means "not worth it / does not fit" and selects the per-head kernel.
// The floor is 4: below that the staging and the extra registers cost about what the
// halved traffic saves, and every instantiation is code the launcher has to carry.
static inline int k3_mla_heads_per_block(int n_head, int key_length) {
    const int cand[3] = {kMlaHeadsPerBlock, 8, 4};
    for (int ci = 0; ci < 3; ++ci) {
        const int hpb = cand[ci];
        if (hpb < 4 || n_head % hpb != 0) continue;
        const size_t shm = ((size_t)hpb * (size_t)key_length +
                            (size_t)hpb * kMlaCtxTile + 3u * (size_t)hpb) * sizeof(float);
        if (shm <= kMlaShmBudget) return hpb;
    }
    return 1;
}

static inline int k3_sm_count(int dev) {
    static int cache[kMlaMaxDevices] = {0};
    if (dev < 0 || dev >= kMlaMaxDevices) return 0;
    if (cache[dev] == 0) {
        cudaDeviceProp prop{};
        cache[dev] = (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
                   ? prop.multiProcessorCount : -1;
    }
    return cache[dev] > 0 ? cache[dev] : 0;
}

// A graph records grid geometry, while the live causal bound remains in d_pos. Pinning
// only the former makes a whole ingestion chunk replay-safe without letting token b see
// any row beyond its own prefix. Keep the graded <=4096 window on the historical
// un-split arithmetic even if a caller accidentally leaves a pin engaged.
static int g_k3_mla_split_pin = 0;

static inline int k3_mla_split_pin_at(int n_ctx) {
    return (g_k3_mla_split_pin > 0 && n_ctx > kMlaSplitMinCtx)
         ? g_k3_mla_split_pin : 0;
}

static bool k3_mla_reach_fill() {
    static const bool on = [] {
        const char* e = std::getenv("SPARKINFER_K3_MLA_FILL");
        return !(e && e[0] == '0');
    }();
    return on;
}

static bool k3_mla_partial_fill() {
    static const bool on = [] {
        const char* e = std::getenv("SPARKINFER_K3_MLA_FILL_PARTIAL");
        return !(e && e[0] == '0');
    }();
    return on;
}

// The latent KV cache element type is a template parameter on all three decode
// kernels, deduced from the k_cache argument so no launch site names it. f32 is
// the historical layout; __half is the one that matters: the cache is the single
// largest byte item a decode token moves (24 MLA layers x ctx x 576 floats,
// replicated on every rank — 13.7 GB/token/rank at 128k), and llama.cpp's own
// default type_k is F16, so the f32 cache paid 2x the reference's KV traffic to
// be MORE precise than the thing the output is scored against. Conversion is
// in-register on load; arithmetic stays f32 throughout.
static __device__ __forceinline__ float kv_ld(const float* p) { return *p; }
static __device__ __forceinline__ float kv_ld(const __half* p) { return __half2float(*p); }

template <int BLOCK, typename KV>
__global__ void mla_decode_attn_kernel(float* __restrict__ out,
                                       const float* __restrict__ q,
                                       const KV* __restrict__ k_cache,
                                       const float* __restrict__ wv_b,
                                       int key_length, int kv_lora, int v_dim,
                                       const int* __restrict__ d_pos, float scale) {
    k3_pdl_sync();
    // LENGTH COMES FROM DEVICE MEMORY, NOT FROM A KERNEL ARGUMENT.
    //
    // A captured graph freezes its arguments. Passing position+1 by value meant every
    // replay attended over the context length that happened to be live at CAPTURE time —
    // right on the first replay and wrong on every one after it, which is the failure
    // mode that looks like a correct model slowly going insane. n_ctx is only ever a loop
    // bound here, so reading it from memory costs one load and changes no arithmetic.
    const int n_ctx = *d_pos + 1;   // d_pos is the ROW INDEX; the length is one more
    constexpr int NWARP = BLOCK / 32;
    const int h    = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    const float* qh = q + (size_t)h * key_length;
    const float* wh = wv_b + (size_t)h * (size_t)kv_lora * v_dim;

    extern __shared__ float smem[];
    float* s_q   = smem;                      // key_length
    float* s_acc = s_q + key_length;          // kv_lora   (unnormalised latent)
    float* s_p   = s_acc + kv_lora;           // kMlaCtxTile
    float* red   = s_p + kMlaCtxTile;         // BLOCK/32 + 1

    for (int d = threadIdx.x; d < key_length; d += BLOCK) s_q[d] = qh[d];
    for (int r = threadIdx.x; r < kv_lora;    r += BLOCK) s_acc[r] = 0.0f;
    __syncthreads();

    // Running softmax state. Uniform across the block: both block_max and
    // block_sum broadcast, so every thread carries the same m and l.
    float m = -1e30f;
    float l = 0.0f;

    for (int t0 = 0; t0 < n_ctx; t0 += kMlaCtxTile) {
        const int tn = min(kMlaCtxTile, n_ctx - t0);

        // --- score the tile: one warp per token, lanes stride over d ---
        for (int t = warp; t < tn; t += NWARP) {
            const KV* kt = k_cache + (size_t)(t0 + t) * key_length;
            float s = 0.0f;
            for (int d = lane; d < key_length; d += 32) s += s_q[d] * kv_ld(kt + d);
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                s += __shfl_down_sync(0xffffffff, s, off);
            if (lane == 0) s_p[t] = s * scale;
        }
        __syncthreads();

        // --- online rescale: fold this tile into (m, l) ---
        float tm = -1e30f;
        for (int t = threadIdx.x; t < tn; t += BLOCK) tm = fmaxf(tm, s_p[t]);
        tm = block_max<BLOCK>(tm, red);
        const float m_new = fmaxf(m, tm);
        // First tile: m is -1e30 and corr underflows to 0, which is exactly the
        // right multiplier for an accumulator that is still zero.
        const float corr = __expf(m - m_new);

        float ls = 0.0f;
        for (int t = threadIdx.x; t < tn; t += BLOCK) {
            const float e = __expf(s_p[t] - m_new);
            s_p[t] = e;
            ls += e;
        }
        ls = block_sum<BLOCK>(ls, red);
        l = l * corr + ls;
        m = m_new;
        __syncthreads();

        // latent[r] += sum_t p[t] * k_cache[t, r]   (V = leading kv_lora of K).
        // r-major: consecutive threads read consecutive floats of the same row.
        for (int r = threadIdx.x; r < kv_lora; r += BLOCK) {
            float a = s_acc[r] * corr;
            for (int t = 0; t < tn; ++t)
                a += s_p[t] * kv_ld(k_cache + (size_t)(t0 + t) * key_length + r);
            s_acc[r] = a;
        }
        __syncthreads();
    }

    // Deferring the 1/sum to here is what makes the pass single: the old kernel
    // normalised the score vector before touching the cache, which needs the
    // whole vector resident first.
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    for (int r = threadIdx.x; r < kv_lora; r += BLOCK) s_acc[r] *= inv;
    __syncthreads();

    // out[v] = sum_r wv_b[r, v, h] * latent[r] — one warp per v, lanes over r.
    float* oh = out + (size_t)h * v_dim;
    for (int v = warp; v < v_dim; v += NWARP) {
        const float* wr = wh + (size_t)v * kv_lora;
        float acc = 0.0f;
        for (int r = lane; r < kv_lora; r += 32) acc += wr[r] * s_acc[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffff, acc, off);
        if (lane == 0) oh[v] = acc;
    }
}

// --- generic f32-activation GEMV --------------------------------------------
// One CUDA thread block per output row n; threads stride over K.

struct BlockQ8_0 {   // 34 bytes, matches ggml block_q8_0 exactly
    uint16_t d;       // fp16 scale
    int8_t   qs[32];
};

static_assert(sizeof(BlockQ8_0) == 34, "bad block_q8_0 layout");
static_assert(alignof(BlockQ8_0) == 2, "get_int_b2 assumes 2-byte alignment");

// Four int8 weights as one packed int, from a 2-BYTE-aligned source.
//
// WHY THIS EXISTS RATHER THAN A PLAIN int LOAD. BlockQ8_0 is 34 bytes with alignment
// 2, so block b starts at byte 34b and its qs[] starts at 34b+2. 34b mod 4 == 2b mod 4,
// so qs is 4-byte aligned only for ODD b — never uniformly. nvcc therefore cannot
// legally widen a scalar `qs[j]` loop and must emit one ld.global.s8 per element:
// 32 loads per block. Pairs of uint16 ARE always aligned (the whole struct is), so two
// of those rebuild the identical 4 bytes and let the inner loop run 8 iterations
// instead of 32. ggml's CUDA path carries get_int_b2() for exactly this reason
// (ggml/src/ggml-cuda/vecdotq.cuh) — this is the same trick, not a new idea.
//
// Lane j of the result is qs[4*i32 + j] in bits [8j, 8j+8), matching the order the
// scalar loop consumed them, so callers can keep their accumulation order byte-exact.
// Little-endian is guaranteed on every CUDA target.
__device__ __forceinline__ int get_int_b2(const int8_t* __restrict__ qs, int i32) {
    const uint16_t* x16 = (const uint16_t*)qs;
    int x32 = (int)x16[2 * i32 + 0];
    x32 |= (int)x16[2 * i32 + 1] << 16;
    return x32;
}

// The same four bytes, assembled from 32-BIT loads instead of two 16-bit ones.
//
// WHY IT CAN BE FASTER. The uint16 pair above costs two memory instructions per
// int32, so a 32-weight block costs 16 loads for qs plus one for d = 17. This
// version loads the ALIGNED word containing the bytes and, when qs is offset by
// 2, splices the two neighbouring words with __funnelshift_r. Consecutive i32
// share their words -- iteration i reads words [k, k+1] and iteration i+1 reads
// [k+1, k+2] -- so an unrolled loop over one block touches 9 distinct words
// instead of 16, and nvcc's CSE collapses the repeats. The measured projection
// GEMVs sit at 0.7-1.9 TB/s against 4.8 available while issuing ~9 L1 wavefronts
// per instruction, which is the signature of an ISSUE-bound loop rather than a
// bandwidth-bound one: instruction count is the lever, not bytes.
//
// BIT-IDENTICAL BY CONSTRUCTION. It returns the same 32 bits: little-endian byte
// order is guaranteed on every CUDA target, __funnelshift_r((lo, hi), s) yields
// exactly the bytes at the requested address, and the caller's dp4a order and
// accumulator are untouched. Nothing is reassociated.
//
// THE OVER-READ IS THE RISK, and it is why this is gated. When qs is 2-byte
// offset the top word read can reach at most 2 bytes past the final block of a
// tensor. Blocks are contiguous, so only the very last block of an allocation is
// exposed, and cudaMalloc's 512-byte granularity covers it in practice -- but
// "in practice" is not proof, so compute-sanitizer must be clean before this
// ever defaults on.
__device__ __forceinline__ int get_int_b2_wide(const int8_t* __restrict__ qs, int i32) {
    const uintptr_t a = (uintptr_t)(qs + 4 * i32);
    const uint32_t* __restrict__ w = (const uint32_t*)(a & ~(uintptr_t)3);
    const int sh = (int)(a & 3u) * 8;
    // sh is 0 or 16 and is UNIFORM across the whole tensor, so the branch is not
    // divergent: alignof(BlockQ8_0) is 2, and every block starts at 34b.
    return sh ? (int)__funnelshift_r(w[0], w[1], sh) : (int)w[0];
}

// SPARKINFER_K3_B2WIDE=1 selects it. DEFAULT OFF, and it stays off until it is
// measured: three changes on this branch were written default-ON while still
// unmeasured and all three shipped regressions. Gated rather than swapped so the
// A/B runs on ONE binary.
__device__ __forceinline__ int get_int_b2_sel(const int8_t* __restrict__ qs, int i32,
                                              bool wide) {
    return wide ? get_int_b2_wide(qs, i32) : get_int_b2(qs, i32);
}

// CPU-reference-compatible activation conversion. One CUDA thread deliberately
// owns one quant block: these vectors are tiny (at most 33,792 values in K3), and
// the serial max scan preserves ggml's first-maximum sign rule exactly. Parallel
// reduction here would make equal-magnitude +/- values pick a schedule-dependent
// sign, changing every quant in the block while leaving the dequantised values
// deceptively close.
__global__ void quantize_q8_0_kernel(BlockQ8_0* __restrict__ out,
                                     const float* __restrict__ x, int n_blocks) {
    k3_pdl_sync();
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

__global__ void quantize_q8_k_kernel(BlockQ8K* __restrict__ out,
                                     const float* __restrict__ x,
                                     int blocks_per_row, int n_rows) {
    k3_pdl_sync();
    const int bi = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_blocks = blocks_per_row * n_rows;
    if (bi >= n_blocks) return;
    const float* xb = x + (size_t)bi * 256;
    BlockQ8K& q = out[bi];
    float maxv = 0.0f, amax = 0.0f;
    for (int j = 0; j < 256; ++j) {
        const float ax = fabsf(xb[j]);
        if (ax > amax) { amax = ax; maxv = xb[j]; }
    }
    if (amax == 0.0f) {
        q.d = 0.0f;
        for (int j = 0; j < 256; ++j) q.qs[j] = 0;
        for (int j = 0; j < 16; ++j) q.bsums[j] = 0;
        return;
    }
    const float iscale = -127.0f / maxv;
    for (int j = 0; j < 256; ++j) {
        int v = __float2int_rn(iscale * xb[j]);
        q.qs[j] = (int8_t)(v < 127 ? v : 127);
    }
    for (int j = 0; j < 16; ++j) {
        int sum = 0;
        for (int k = 0; k < 16; ++k) sum += q.qs[16 * j + k];
        q.bsums[j] = (int16_t)sum;
    }
    q.d = 1.0f / iscale;
}


// ---------------------------------------------------------------------------
// MLA decode, split over context (flash-decoding style).
//
// WHY. The single-block-per-head kernel launches <<<96, 256>>> — 24,576 threads on a
// 132-SM H200 with ~270k thread slots. That is ~9% occupancy and 36 SMs get no work at
// all, and at ctx 128k each block then walks 1024 tiles serially. Measured on 8x H200:
// attn_mla is 69.2% of decode at 128k (against 18.2% at 8k), and MLA time grows 23.8x
// for a 16x context increase — superlinear, which is what a kernel that cannot fill the
// machine looks like as its serial loop gets longer.
//
// This splits the context across SPLITS blocks per head, so the grid becomes
// n_head * SPLITS. Each block runs the same online softmax over its slice and writes a
// PARTIAL (m, l, latent); mla_decode_combine_kernel merges them. The math is the standard
// flash-decoding merge and is exact up to float rounding:
//
//     m   = max_i m_i
//     l   = sum_i l_i * exp(m_i - m)
//     acc = sum_i acc_i * exp(m_i - m)
//
// The wv_b projection moves to the combine pass because it needs the merged latent.
//
// NOT bit-identical to the single-block version: summing per-slice partials reassociates
// the same terms. The parity gate is top-1 >= 0.90 / KL <= 0.20 and the reference here is
// float64, so this is checked on tolerance, like the online-softmax rewrite it builds on.
template <int BLOCK, typename KV>
__global__ void mla_decode_attn_split_kernel(float* __restrict__ part_acc,
                                             float* __restrict__ part_ml,
                                             const float* __restrict__ q,
                                             const KV* __restrict__ k_cache,
                                             int key_length, int kv_lora,
                                             const int* __restrict__ d_pos,
                                             float scale, int splits) {
    k3_pdl_sync();
    // Device-read length; see mla_decode_attn_kernel. `splits` stays a launch-time
    // constant because it sizes the GRID, which a captured graph cannot change — the
    // driver re-captures instead when the plan moves (k3_mla_decode_plan).
    const int n_ctx = *d_pos + 1;   // d_pos is the ROW INDEX; the length is one more
    constexpr int NWARP = BLOCK / 32;
    const int h    = blockIdx.x;
    const int sp   = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    // Contiguous slice per split: keeps each block's cache walk sequential, which is what
    // the hardware prefetcher wants. A strided split would interleave 96 streams.
    const int chunk = (n_ctx + splits - 1) / splits;
    const int t_beg = sp * chunk;
    const int t_end = min(n_ctx, t_beg + chunk);

    const float* qh = q + (size_t)h * key_length;
    extern __shared__ float smem[];
    float* s_q   = smem;                      // key_length
    float* s_acc = s_q + key_length;          // kv_lora
    float* s_p   = s_acc + kv_lora;           // kMlaCtxTile
    float* red   = s_p + kMlaCtxTile;         // BLOCK/32 + 1

    for (int d = threadIdx.x; d < key_length; d += BLOCK) s_q[d] = qh[d];
    for (int r = threadIdx.x; r < kv_lora;    r += BLOCK) s_acc[r] = 0.0f;
    __syncthreads();

    float m = -1e30f, l = 0.0f;

    for (int t0 = t_beg; t0 < t_end; t0 += kMlaCtxTile) {
        const int tn = min(kMlaCtxTile, t_end - t0);

        for (int t = warp; t < tn; t += NWARP) {
            const KV* kt = k_cache + (size_t)(t0 + t) * key_length;
            float sdot = 0.0f;
            for (int d = lane; d < key_length; d += 32) sdot += s_q[d] * kv_ld(kt + d);
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                sdot += __shfl_down_sync(0xffffffff, sdot, off);
            if (lane == 0) s_p[t] = sdot * scale;
        }
        __syncthreads();

        float tm = -1e30f;
        for (int t = threadIdx.x; t < tn; t += BLOCK) tm = fmaxf(tm, s_p[t]);
        tm = block_max<BLOCK>(tm, red);
        const float m_new = fmaxf(m, tm);
        const float corr  = __expf(m - m_new);

        float ls = 0.0f;
        for (int t = threadIdx.x; t < tn; t += BLOCK) {
            const float e = __expf(s_p[t] - m_new);
            s_p[t] = e;
            ls += e;
        }
        ls = block_sum<BLOCK>(ls, red);
        l = l * corr + ls;
        m = m_new;
        __syncthreads();

        for (int r = threadIdx.x; r < kv_lora; r += BLOCK) {
            float a = s_acc[r] * corr;
            for (int t = 0; t < tn; ++t)
                a += s_p[t] * kv_ld(k_cache + (size_t)(t0 + t) * key_length + r);
            s_acc[r] = a;
        }
        __syncthreads();
    }

    // UNNORMALISED partial: the 1/l cannot be applied per slice, only after the merge.
    float* pa = part_acc + ((size_t)h * splits + sp) * kv_lora;
    for (int r = threadIdx.x; r < kv_lora; r += BLOCK) pa[r] = s_acc[r];
    if (threadIdx.x == 0) {
        // An empty slice (n_ctx < splits) must contribute nothing: l = 0 and a very
        // negative m make its weight exactly zero in the merge.
        float* pm = part_ml + ((size_t)h * splits + sp) * 2;
        pm[0] = (t_end > t_beg) ? m : -1e30f;
        pm[1] = (t_end > t_beg) ? l : 0.0f;
    }
}

// HEAD-BATCHED context slice. Same partial contract as mla_decode_attn_split_kernel —
// unnormalised latent per (head, slice) plus that slice's (m, l) — so
// mla_decode_combine_kernel merges either one without knowing which ran.
//
// The three phases and what each one reuses:
//
//   1. score    one warp per token, lanes striding d. A lane loads k_cache[t][d] ONCE
//               and multiplies it into HPB accumulators, one per head. This is the
//               whole point: the un-batched kernel issues the same load in HPB
//               different blocks.
//   2. softmax  one warp per head, over that head's kMlaCtxTile scores. The un-batched
//               kernel used a BLOCK-wide reduction because it had one head to reduce;
//               with HPB heads a block-wide reduction per head would serialise HPB of
//               them, whereas warp-per-head runs all HPB at once and needs no barrier
//               inside the phase.
//   3. latent   thread owns r (and r + BLOCK, ...) for the whole slice. One load of
//               k_cache[t][r] feeds HPB register accumulators.
//
// SUMMATION ORDER IS NOT the un-batched kernel's, and cannot be: phase 2 reduces over
// 32 lanes where the other reduces over BLOCK threads, so the tile's max and exp-sum
// fold in a different tree. The result is a different rounding of the same quantity,
// not a different quantity — the score dot product, the online rescale and the latent
// accumulation are element-for-element identical, and the latent sum over t inside a
// tile still runs in increasing t. cpu_reference_test models THIS schedule against a
// float64 two-pass reference; kimi_k3_numeric_test pins it on the device at a context
// long enough to take this path.
// UT and UD ARE THE MEMORY-LEVEL-PARALLELISM KNOB, and they are the reason this kernel
// runs at 14% of the card's bandwidth.
//
// At the scored shape the grid is 255 blocks of 256 threads at 128 registers, i.e. two
// resident blocks per SM and 16 of 64 warps. The kernel moves 151 MB per call in 215 us
// = 677 GB/s aggregate, which is ~5.1 GB/s per SM, or one 128 B line per ~44 SM-cycles.
// Against an HBM latency near 600 cycles that is roughly ONE outstanding load per warp:
// the kernel is not bandwidth-bound and not issue-bound, it is latency-exposed because
// too few requests are in flight. The two unroll factors are what put them there —
//
//     UD  score  d-loop    TPW    * UD = 4*UD loads outstanding per warp
//     UT  latent t-loop    RSLOTS * UT = 2*UT loads outstanding per thread
//
// and both were last chosen when a cache row was TWICE as wide, before #86 made
// k_cache __half. Halving the bytes per element halved the bytes each outstanding load
// carries, so the depth that saturated the old row no longer saturates this one.
//
// MEASURED at the scored shape (ctx 131,039, one H200, no weights, the kernel alone,
// best of 3 interleaved passes, every arm verified BIT-IDENTICAL to the shipped one):
//
//   UT  UD   regs   ms      vs shipped
//    4   2   128    0.223   1.000x   <- shipped
//    8   2   127    0.223   1.001x
//   16   2   128    0.216   1.030x
//    4   6   128    0.194   1.145x
//    8   3   128    0.209   1.065x
//    8   6   128    0.191   1.169x   <- taken
//   16   6   157    0.242   0.922x   <- over the register cliff
//   32   6   162    0.241   0.926x   <- over the register cliff
//    4   7   136    0.324   0.689x   <- over the register cliff
//
// THE CLIFF IS THE WHOLE CONSTRAINT. 65536/(128*256) is exactly 2.0 blocks per SM, so
// 129 registers is one block and half the occupancy — the table above prices that at
// 8-31% and the file's own HPB note prices it at 10%. (8, 6) is the deepest pair that
// still fits 128 registers with no spill; every deeper pair measured here loses. That
// is also why this is a template parameter and not a bigger literal: the shape that
// tolerates the depth is the shape it was measured at, and every OTHER instantiation
// keeps the shipped (4, 2).
//
// Unrolling changes issue order, never arithmetic order, so this is bit-identical by
// construction rather than by tolerance — and the bench asserts it word-for-word.
// LSP (SPARKINFER_K3_MLA_LSPLIT=1) restructures the LATENT pass only. The MLA
// kernel's LSU/shared pipe runs ~56% busy while the memory pipe sits ~14%, so
// the lever is LSU instruction-cycles, not bytes. Two counted effects:
//   * p-broadcasts halve: s_p[hh*tile+t] is uniform across the block (a 32-way
//     broadcast, 4 useful bytes/cycle, HPB per token per thread). Splitting the
//     warps into two head groups of HPB/2 takes 12 -> 6.
//   * kv loads widen: each thread takes TWO consecutive r as one __half2, so a
//     latent kv request carries 128 B/warp instead of 64 B. Two groups re-read
//     the same rows; request count is unchanged and the second group hits L1.
// BIT-IDENTICAL: every acc still sums p[hh]*k[t][r] over ascending t into the
// same accumulator — only which THREAD owns which (r, head) changes, and the
// partial write lands the same values at the same addresses. acc is
// REINTERPRETED (RSLOTS*HPB floats either way), not resized. The "__half2
// reassociates" dead-end is about the SCORE pass (a dot over d); here the
// loaded axis IS the output axis and nothing sums across the pair.
template <int BLOCK, int HPB, int RSLOTS, typename KV, int TPW = kMlaTokensPerWarp,
          int UT = 4, int UD = 2, bool LSP = false, bool PVT = false,
          int KLENT = 0, bool QT = false>
__global__ void mla_decode_attn_hbatch_kernel(float* __restrict__ part_acc,
                                              float* __restrict__ part_ml,
                                              const float* __restrict__ q,
                                              const KV* __restrict__ k_cache,
                                              int key_length, int kv_lora,
                                              const int* __restrict__ d_pos,
                                              float scale, int splits) {
    k3_pdl_sync();
    // Device-read length; see mla_decode_attn_kernel.
    const int n_ctx = *d_pos + 1;   // d_pos is the ROW INDEX; the length is one more
    constexpr int NWARP = BLOCK / 32;
    const int h0   = blockIdx.x * HPB;
    const int sp   = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    const int chunk = (n_ctx + splits - 1) / splits;
    const int t_beg = sp * chunk;
    const int t_end = min(n_ctx, t_beg + chunk);

    extern __shared__ float smem[];
    // KLENT (compile-time key_length; instantiated at 576): the 12 per-head s_q
    // row bases become immediate LDS offsets instead of held registers — UD=7
    // was priced at 136 regs vs the 128 cliff, and the freed ~8-12 registers are
    // that gap. Constants do not round: bit-identical. ptxas -v is the falsifier.
    const int klen = KLENT ? KLENT : key_length;
    float* s_q = smem;                                        // HPB * key_length
    float* s_p = s_q + (size_t)HPB * klen;                    // HPB * kMlaCtxTile
    float* s_m = s_p + (size_t)HPB * kMlaCtxTile;             // HPB  running max
    // PVT (SPARKINFER_K3_MLA_PVT=1): t-major s_p. Head-major spaces one token's
    // HPB probabilities kMlaCtxTile floats apart, so the latent pass pays HPB
    // uniform broadcast LDS.32 per token per thread — on the pipe measured ~56%
    // busy. t-major packs them adjacent (stride HPB=12, base t*48 B, 16-aligned)
    // so they arrive as 3 LDS.128. Same allocation, same values, same
    // accumulation order — an address relabeling, bit-identical by construction.
    // The cost lands in phase 2 (lane stride HPB -> 4-way bank conflict on a
    // <=128-element per-head pass) — priced and accepted: phase 3 touches every
    // probability BLOCK times, phase 2 only ~3 times.
    auto s_p_idx = [&](int hh, int t) -> int {
        return PVT ? (t * HPB + hh) : (hh * kMlaCtxTile + t);
    };
    float* s_l = s_m + HPB;                                   // HPB  running exp-sum
    float* s_c = s_l + HPB;                                   // HPB  this tile's rescale

    // (h0 + i/key_length)*key_length + (i%key_length) is identically h0*key_length + i,
    // so the 32-bit div AND mod per element were computing the identity — 27 iterations
    // of both, per thread, per block, on a 256-block grid, every MLA layer. Same
    // addresses, same values, now a contiguous copy.
    if constexpr (QT) {
        // P5 (SPARKINFER_K3_MLA_QT=1): store q d-MAJOR. In the score loop each
        // lane owns one d and reads all HPB heads at it — head-major spaces those
        // 12 values klen*4 B apart (12 scalar LDS); d-major packs them into 48
        // contiguous bytes at a 16-aligned base (d*48) -> 3 LDS.128, and lane L's
        // word 12L hits banks 0,12,24,4,... — a perfect 32-bank partition, no
        // conflicts. One-time transposed store amortised over ~497 tokens/block.
        // Same values into the same qv in the same FMA order: bit-identical.
        for (int i = threadIdx.x; i < HPB * klen; i += BLOCK) {
            const int hh = i / klen, d = i % klen;
            s_q[(size_t)d * HPB + hh] = q[(size_t)h0 * klen + i];
        }
    } else {
    for (int i = threadIdx.x; i < HPB * klen; i += BLOCK)
        s_q[i] = q[(size_t)h0 * klen + i];
    }
    if (threadIdx.x < HPB) { s_m[threadIdx.x] = -1e30f; s_l[threadIdx.x] = 0.0f; }

    // Latent accumulators: RSLOTS r-values x HPB heads, in registers.
    float acc[RSLOTS][HPB];
    // LSP head-group ids: group g owns heads [g*HPB/2,(g+1)*HPB/2), its BLOCK/2
    // threads cover every r as RSLOTS pair-slots of two consecutive r. Free when off.
    const int lsp_lt = (int)threadIdx.x & (BLOCK / 2 - 1);
    const int lsp_h0 = ((int)threadIdx.x / (BLOCK / 2)) * (HPB / 2);
#pragma unroll
    for (int u = 0; u < RSLOTS; ++u)
#pragma unroll
        for (int hh = 0; hh < HPB; ++hh) acc[u][hh] = 0.0f;
    __syncthreads();

    for (int t0 = t_beg; t0 < t_end; t0 += kMlaCtxTile) {
        const int tn = min(kMlaCtxTile, t_end - t0);

        // --- 1. score, kMlaTokensPerWarp tokens at a time ---
        // The staged query is read from SHARED once per (head, d) and multiplied into
        // TPW tokens, so a warp issues HPB shared loads per TPW*128 bytes of cache
        // rather than per 128. See kMlaTokensPerWarp: at TPW 1 that ratio, against 32
        // banks x 4 B/clk, caps the kernel below the card's HBM rate, and the staging
        // that saves the traffic becomes the thing capping it.
        for (int tb = warp * TPW; tb < tn; tb += NWARP * TPW) {
            float s[HPB][TPW];
#pragma unroll
            for (int hh = 0; hh < HPB; ++hh)
#pragma unroll
                for (int tt = 0; tt < TPW; ++tt) s[hh][tt] = 0.0f;

            // The tail duplicates the last row rather than branching: the extra lanes
            // compute a score that is never stored, and every load stays in bounds.
            const int tc = min(TPW, tn - tb);
            const KV* kt[TPW];
#pragma unroll
            for (int tt = 0; tt < TPW; ++tt)
                kt[tt] = k_cache + (size_t)(t0 + tb + min(tt, tc - 1)) * klen;

#pragma unroll UD
            for (int d = lane; d < klen; d += 32) {
                float kv[TPW];
#pragma unroll
                for (int tt = 0; tt < TPW; ++tt) kv[tt] = kv_ld(kt[tt] + d);
                if constexpr (QT) {
                    float qh_[HPB];
                    const float4* qp = (const float4*)(s_q + (size_t)d * HPB);
#pragma unroll
                    for (int j = 0; j < HPB / 4; ++j) {
                        const float4 v4 = qp[j];
                        qh_[4*j+0]=v4.x; qh_[4*j+1]=v4.y;
                        qh_[4*j+2]=v4.z; qh_[4*j+3]=v4.w;
                    }
#pragma unroll
                    for (int hh = 0; hh < HPB; ++hh) {
#pragma unroll
                        for (int tt = 0; tt < TPW; ++tt) s[hh][tt] += qh_[hh] * kv[tt];
                    }
                } else {
#pragma unroll
                for (int hh = 0; hh < HPB; ++hh) {
                    const float qv = s_q[hh * klen + d];
#pragma unroll
                    for (int tt = 0; tt < TPW; ++tt) s[hh][tt] += qv * kv[tt];
                }
                }
            }
#pragma unroll
            for (int hh = 0; hh < HPB; ++hh)
#pragma unroll
                for (int tt = 0; tt < TPW; ++tt) {
                    float v = s[hh][tt];
#pragma unroll
                    for (int off = 16; off > 0; off >>= 1)
                        v += __shfl_down_sync(0xffffffff, v, off);
                    if (lane == 0 && tt < tc) s_p[s_p_idx(hh, tb + tt)] = v * scale;
                }
        }
        __syncthreads();

        // --- 2. online softmax, one warp per head ---
        // (m, l) for head hh live in shared so they survive the tile loop, but the warp
        // that owns hh is the only thing that touches them. Read both before the
        // __syncwarp so no lane can observe this tile's write in place of the running
        // value it is supposed to fold into.
        for (int hh = warp; hh < HPB; hh += NWARP) {
            float* pr = s_p + (PVT ? hh : hh * kMlaCtxTile);
            const int ps = PVT ? HPB : 1;
            const float m_prev = s_m[hh];
            const float l_prev = s_l[hh];
            __syncwarp();

            float tm = -1e30f;
            for (int t = lane; t < tn; t += 32) tm = fmaxf(tm, pr[t * ps]);
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                tm = fmaxf(tm, __shfl_down_sync(0xffffffff, tm, off));
            tm = __shfl_sync(0xffffffff, tm, 0);   // every lane rescales with the same m

            const float m_new = fmaxf(m_prev, tm);
            // First tile: m_prev is -1e30 and corr underflows to 0, the right multiplier
            // for an accumulator that is still zero.
            const float corr = __expf(m_prev - m_new);

            float ls = 0.0f;
            for (int t = lane; t < tn; t += 32) {
                const float e = __expf(pr[t * ps] - m_new);
                pr[t * ps] = e;
                ls += e;
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                ls += __shfl_down_sync(0xffffffff, ls, off);
            if (lane == 0) {
                s_l[hh] = l_prev * corr + ls;
                s_m[hh] = m_new;
                s_c[hh] = corr;
            }
        }
        __syncthreads();

        // --- 3. latent accumulation ---
        // One k_cache[t][r] load feeds HPB fused multiply-adds. RSLOTS is small and
        // both inner loops are fully unrolled, so acc stays in registers: a runtime
        // index here would spill the whole array to local memory and undo the point.
        if constexpr (LSP) {
#pragma unroll
            for (int u = 0; u < RSLOTS; ++u)
#pragma unroll
                for (int e = 0; e < 2; ++e)
#pragma unroll
                    for (int hh = 0; hh < HPB / 2; ++hh)
                        acc[u][e * (HPB / 2) + hh] *= s_c[lsp_h0 + hh];
        } else {
#pragma unroll
        for (int u = 0; u < RSLOTS; ++u)
#pragma unroll
            for (int hh = 0; hh < HPB; ++hh) acc[u][hh] *= s_c[hh];
        }

        if constexpr (LSP) {
        constexpr int HG = HPB / 2;
#pragma unroll UT
        for (int t = 0; t < tn; ++t) {
            const KV* kt = k_cache + (size_t)(t0 + t) * klen;
            float p[HG];
            if constexpr (PVT) {
                // P6: the group's HG=6 probabilities are 24 contiguous bytes at
                // t*48 + lsp_h0*4. Group 0 (base 16-aligned): float4 + float2.
                // Group 1 (base 24 mod 16 = 8): float2 + float4. Both are 2
                // uniform wide LDS instead of 6 scalar broadcasts — identical
                // values in identical order, bit-identical by construction.
                const float* pb = s_p + (size_t)t * HPB + lsp_h0;
                if ((lsp_h0 & 3) == 0) {
                    const float4 a = *(const float4*)pb;
                    const float2 b = *(const float2*)(pb + 4);
                    p[0]=a.x; p[1]=a.y; p[2]=a.z; p[3]=a.w; p[4]=b.x; p[5]=b.y;
                } else {
                    const float2 a = *(const float2*)pb;
                    const float4 b = *(const float4*)(pb + 2);
                    p[0]=a.x; p[1]=a.y; p[2]=b.x; p[3]=b.y; p[4]=b.z; p[5]=b.w;
                }
            } else {
#pragma unroll
            for (int hh = 0; hh < HG; ++hh) p[hh] = s_p[s_p_idx(lsp_h0 + hh, t)];
            }
#pragma unroll
            for (int u = 0; u < RSLOTS; ++u) {
                const int r = 2 * lsp_lt + u * BLOCK;
                if (r + 1 < kv_lora) {
                    // Rows are key_length*2 B apart and r is even: always 4-aligned.
                    const __half2 kv2 = *(const __half2*)(kt + r);
                    const float k0 = __low2float(kv2), k1 = __high2float(kv2);
#pragma unroll
                    for (int hh = 0; hh < HG; ++hh) {
                        acc[u][hh]      += p[hh] * k0;
                        acc[u][HG + hh] += p[hh] * k1;
                    }
                }
            }
        }
        } else {
#pragma unroll UT
        for (int t = 0; t < tn; ++t) {
            const KV* kt = k_cache + (size_t)(t0 + t) * klen;
            float p[HPB];
            if constexpr (PVT) {
                // 12 adjacent floats at a 16-aligned base: 3 uniform LDS.128.
                const float4* pv = (const float4*)(s_p + (size_t)t * HPB);
#pragma unroll
                for (int j = 0; j < HPB / 4; ++j) {
                    const float4 v4 = pv[j];
                    p[4*j+0] = v4.x; p[4*j+1] = v4.y;
                    p[4*j+2] = v4.z; p[4*j+3] = v4.w;
                }
            } else {
#pragma unroll
            for (int hh = 0; hh < HPB; ++hh) p[hh] = s_p[hh * kMlaCtxTile + t];
            }
#pragma unroll
            for (int u = 0; u < RSLOTS; ++u) {
                const int r = threadIdx.x + u * BLOCK;
                if (r < kv_lora) {
                    const float kv = kv_ld(kt + r);
#pragma unroll
                    for (int hh = 0; hh < HPB; ++hh) acc[u][hh] += p[hh] * kv;
                }
            }
        }
        }
        __syncthreads();
    }

    // UNNORMALISED partials, one (head, slice) row each — the combine applies 1/l.
    if constexpr (LSP) {
        constexpr int HG = HPB / 2;
#pragma unroll
        for (int u = 0; u < RSLOTS; ++u) {
            const int r = 2 * lsp_lt + u * BLOCK;
            if (r + 1 < kv_lora)
#pragma unroll
                for (int hh = 0; hh < HG; ++hh) {
                    float* pa = part_acc + ((size_t)(h0 + lsp_h0 + hh) * splits + sp) * kv_lora;
                    pa[r]     = acc[u][hh];
                    pa[r + 1] = acc[u][HG + hh];
                }
        }
    } else {
#pragma unroll
    for (int u = 0; u < RSLOTS; ++u) {
        const int r = threadIdx.x + u * BLOCK;
        if (r < kv_lora)
#pragma unroll
            for (int hh = 0; hh < HPB; ++hh)
                part_acc[((size_t)(h0 + hh) * splits + sp) * kv_lora + r] = acc[u][hh];
    }
    }
    if (threadIdx.x < HPB) {
        // An empty slice (n_ctx < splits) must contribute nothing: l = 0 and a very
        // negative m make its weight exactly zero in the merge.
        float* pm = part_ml + ((size_t)(h0 + threadIdx.x) * splits + sp) * 2;
        pm[0] = (t_end > t_beg) ? s_m[threadIdx.x] : -1e30f;
        pm[1] = (t_end > t_beg) ? s_l[threadIdx.x] : 0.0f;
    }
}

// Merge the per-slice partials, normalise, then project through wv_b.
template <int BLOCK>
__global__ void mla_decode_combine_kernel(float* __restrict__ out,
                                          const float* __restrict__ part_acc,
                                          const float* __restrict__ part_ml,
                                          const float* __restrict__ wv_b,
                                          int kv_lora, int v_dim, int splits) {
    k3_pdl_sync();
    constexpr int NWARP = BLOCK / 32;
    const int h    = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    extern __shared__ float smem[];
    float* s_acc = smem;                    // kv_lora
    float* s_w   = s_acc + kv_lora;         // splits (per-slice rescale factor)

    const float* pml = part_ml + (size_t)h * splits * 2;

    // m = max over slices, on ONE thread while the other BLOCK-1 wait at the barrier.
    //
    // This said "Small (splits <= 64), so thread 0 is cheaper than a reduction" when
    // the cap was 32. It is no longer small: the budget permits 512 (k3_mla_max_splits)
    // and the launcher's fill target asks for 256 at the scored context, so this is a
    // 2*splits serial walk in a grid of only n_head = 12 blocks. It is still ~1% of the
    // attention pass it merges, which is why it is not the binding term today and is
    // left alone — but it is O(splits) at 1/BLOCK parallelism, so it is the term that
    // bites first if splits is ever widened again. Convert it to a block reduction
    // before raising kMlaBlocksPerSm or the budget, not after.
    if (threadIdx.x == 0) {
        float m = -1e30f;
        for (int i = 0; i < splits; ++i) m = fmaxf(m, pml[2 * i]);
        float l = 0.0f;
        for (int i = 0; i < splits; ++i) {
            const float w = __expf(pml[2 * i] - m);
            s_w[i] = w;
            l += pml[2 * i + 1] * w;
        }
        s_w[splits] = l > 0.0f ? 1.0f / l : 0.0f;   // one slot past: the 1/l
    }
    __syncthreads();
    const float inv = s_w[splits];

    for (int r = threadIdx.x; r < kv_lora; r += BLOCK) {
        float a = 0.0f;
        for (int i = 0; i < splits; ++i)
            a += part_acc[((size_t)h * splits + i) * kv_lora + r] * s_w[i];
        s_acc[r] = a * inv;
    }
    __syncthreads();

    const float* wh = wv_b + (size_t)h * (size_t)kv_lora * v_dim;
    float* oh = out + (size_t)h * v_dim;
    for (int v = warp; v < v_dim; v += NWARP) {
        const float* wr = wh + (size_t)v * kv_lora;
        float acc = 0.0f;
        for (int r = lane; r < kv_lora; r += 32) acc += wr[r] * s_acc[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffff, acc, off);
        if (lane == 0) oh[v] = acc;
    }
}

// WIDE COMBINE. mla_decode_combine_kernel above runs grid = n_head = 12 blocks on a
// 132-SM part -- about 9% of the machine -- and its thread 0 walks O(splits) serially
// while the other BLOCK-1 threads wait at the barrier (the comment on that walk already
// flags it as the term that bites first). Splitting merge from projection lets BOTH
// fill the grid: 12 -> n_head*RChunks and n_head*VChunks blocks.
//
// The single-kernel form was the right call BEFORE: two kernels means a second launch,
// and at 93 layers plus ~185 collectives per token that mattered. Once the decode is
// captured as one CUDA graph that launch is a baked graph node and costs essentially
// nothing, so the decomposition becomes free. This is downstream of the graph capture
// in this same PR, not independent of it.
//
// SUMMATION ORDER IS UNCHANGED in both stages -- the slice sum still runs i ascending
// and the projection dot still strides r by 32 across the warp -- so this is
// bit-identical to the kernel it replaces, not a re-rounding of it.
constexpr int kMlaCombineVChunks = 16;
constexpr int kMlaCombineRChunks = 8;

// PARSOFT: the slice-weight prologue, in parallel instead of in thread 0.
//
// This prologue is O(splits) SERIAL DEPENDENT GLOBAL LOADS run by lane 0 while the other
// BLOCK-1 threads wait at the barrier — three passes over `part_ml` (max, then exp and
// the l-accumulation), so at the scored shape that is ~790 loads issued one at a time.
// It runs in EVERY block of the merge grid (n_head * kMlaCombineRChunks = 96) and 24
// times per token per rank, and `splits` is exactly the axis the split heuristic has
// been pushing UP — raising the slice count to fill the attention grid makes this
// prologue longer in direct proportion, which is a real part of why attention-side
// split gains do not reach the token.
//
// Every thread now walks its own stride of `part_ml`, so the loads are all in flight at
// once, and the max and the l-sum come from block reductions. Same values, log-depth
// instead of linear. This removes dependent loads and adds parallelism; it fuses
// nothing and narrows nothing.
//
// `s_w[i]` is bit-identical (the max is order-independent and each weight is an exp of
// the same difference). Only `l` reassociates, and it scales the whole merged row
// uniformly — far inside the accuracy gate.
template <int BLOCK, bool PARSOFT>
__global__ void mla_decode_merge_kernel(float* __restrict__ merged,
                                        const float* __restrict__ part_acc,
                                        const float* __restrict__ part_ml,
                                        int kv_lora, int splits) {
    k3_pdl_sync();
    const int h  = blockIdx.x;
    const int rc = blockIdx.y;
    extern __shared__ float smem[];
    float* s_w = smem;                       // splits + 1, last slot holds 1/l
    float* red = s_w + splits + 1;           // BLOCK/32 + 1, reductions (PARSOFT only)

    const float* pml = part_ml + (size_t)h * splits * 2;
    if constexpr (PARSOFT) {
        float m = -1e30f;
        for (int i = threadIdx.x; i < splits; i += BLOCK) m = fmaxf(m, pml[2 * i]);
        m = block_max<BLOCK>(m, red);
        __syncthreads();                     // `red` is reused by the sum below

        float l = 0.0f;
        for (int i = threadIdx.x; i < splits; i += BLOCK) {
            const float w = __expf(pml[2 * i] - m);
            s_w[i] = w;
            l += pml[2 * i + 1] * w;
        }
        l = block_sum<BLOCK>(l, red);
        if (threadIdx.x == 0) s_w[splits] = l > 0.0f ? 1.0f / l : 0.0f;
    } else if (threadIdx.x == 0) {
        float m = -1e30f;
        for (int i = 0; i < splits; ++i) m = fmaxf(m, pml[2 * i]);
        float l = 0.0f;
        for (int i = 0; i < splits; ++i) {
            const float w = __expf(pml[2 * i] - m);
            s_w[i] = w;
            l += pml[2 * i + 1] * w;
        }
        s_w[splits] = l > 0.0f ? 1.0f / l : 0.0f;
    }
    __syncthreads();
    const float inv = s_w[splits];

    const int per = (kv_lora + kMlaCombineRChunks - 1) / kMlaCombineRChunks;
    const int r0  = rc * per;
    const int r1  = min(kv_lora, r0 + per);
    for (int r = r0 + (int)threadIdx.x; r < r1; r += BLOCK) {
        float a = 0.0f;
        for (int i = 0; i < splits; ++i)
            a += part_acc[((size_t)h * splits + i) * kv_lora + r] * s_w[i];
        merged[(size_t)h * kv_lora + r] = a * inv;
    }
}

template <int BLOCK>
__global__ void mla_decode_project_kernel(float* __restrict__ out,
                                          const float* __restrict__ merged,
                                          const float* __restrict__ wv_b,
                                          int kv_lora, int v_dim) {
    k3_pdl_sync();
    constexpr int NWARP = BLOCK / 32;
    const int h    = blockIdx.x;
    const int vc   = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    extern __shared__ float smem[];
    float* s_acc = smem;                     // kv_lora
    for (int r = (int)threadIdx.x; r < kv_lora; r += BLOCK)
        s_acc[r] = merged[(size_t)h * kv_lora + r];
    __syncthreads();

    const int per = (v_dim + kMlaCombineVChunks - 1) / kMlaCombineVChunks;
    const int v0  = vc * per;
    const int v1  = min(v_dim, v0 + per);
    const float* wh = wv_b + (size_t)h * (size_t)kv_lora * v_dim;
    float* oh = out + (size_t)h * v_dim;
    for (int v = v0 + warp; v < v1; v += NWARP) {
        const float* wr = wh + (size_t)v * kv_lora;
        float acc = 0.0f;
        for (int r = lane; r < kv_lora; r += 32) acc += wr[r] * s_acc[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffff, acc, off);
        if (lane == 0) oh[v] = acc;
    }
}

// SPARKINFER_K3_MLA_WIDE_COMBINE=0 restores the single-kernel combine on one binary.
static bool k3_mla_wide_combine() {
    static const int v = [] {
        const char* e = std::getenv("SPARKINFER_K3_MLA_WIDE_COMBINE");
        return (e && std::atoi(e) == 0) ? 0 : 1;   // default ON
    }();
    return v != 0;
}

template <int BLOCK>
static void k3_mla_launch_combine(float* out, int dev, const float* wv_b, int n_head,
                                  int kv_lora, int v_dim, int splits, cudaStream_t stream) {
    if (k3_mla_wide_combine() && g_mla_merged[dev] != nullptr) {
        dim3 mg((unsigned)n_head, (unsigned)kMlaCombineRChunks);
        // SPARKINFER_K3_MLA_MERGE_PAR=0 restores the thread-0 prologue on one binary.
        static const bool merge_par = [] {
            const char* e = std::getenv("SPARKINFER_K3_MLA_MERGE_PAR");
            return !(e && e[0] == '0');
        }();
        // s_w (splits + 1) plus the block-reduction scratch the parallel prologue needs.
        const size_t mshm = ((size_t)splits + 1 + BLOCK / 32 + 1) * sizeof(float);
        if (merge_par)
            k3_pdl_launch(mg, BLOCK, mshm, stream, mla_decode_merge_kernel<BLOCK, true>,
                g_mla_merged[dev], g_mla_part_acc[dev], g_mla_part_ml[dev], kv_lora, splits);
        else
            k3_pdl_launch(mg, BLOCK, mshm, stream, mla_decode_merge_kernel<BLOCK, false>,
                g_mla_merged[dev], g_mla_part_acc[dev], g_mla_part_ml[dev], kv_lora, splits);
        dim3 pg((unsigned)n_head, (unsigned)kMlaCombineVChunks);
        k3_pdl_launch(pg, BLOCK, (size_t)kv_lora * sizeof(float), stream, mla_decode_project_kernel<BLOCK>, 
            out, g_mla_merged[dev], wv_b, kv_lora, v_dim);
        return;
    }
    const size_t cshm = ((size_t)kv_lora + (size_t)splits + 1) * sizeof(float);
    k3_pdl_launch((unsigned)n_head, BLOCK, cshm, stream, mla_decode_combine_kernel<BLOCK>, 
        out, g_mla_part_acc[dev], g_mla_part_ml[dev], wv_b, kv_lora, v_dim, splits);
}

template <int BLOCK>
__global__ void proj_q8_0_kernel(float* __restrict__ y, const float* __restrict__ x,
                                 const BlockQ8_0* __restrict__ W, int blocks_per_row) {
    k3_pdl_sync();
    const int n = blockIdx.x;
    const BlockQ8_0* row = W + (size_t)n * blocks_per_row;
    __shared__ float shm[BLOCK / 32 + 1];

    float acc = 0.0f;
    for (int b = threadIdx.x; b < blocks_per_row; b += BLOCK) {
        const float d = __half2float(__ushort_as_half(row[b].d));
        const float* xb = x + (size_t)b * 32;
        float s = 0.0f;
        // 8 packed loads instead of 32 byte loads — see get_int_b2. The unpack order
        // below walks j = 0..31 in the SAME order the scalar loop did, so every
        // partial sum lands in the same float in the same sequence and the result is
        // bit-identical, not merely close.
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int wq = get_int_b2(row[b].qs, i);
            const float* xq = xb + 4 * i;
            s += (float)(int8_t)(wq >>  0) * xq[0];
            s += (float)(int8_t)(wq >>  8) * xq[1];
            s += (float)(int8_t)(wq >> 16) * xq[2];
            s += (float)(int8_t)(wq >> 24) * xq[3];
        }
        acc += d * s;
    }
    acc = block_sum<BLOCK>(acc, shm);
    if (threadIdx.x == 0) y[n] = acc;
}

// Same dot product, but ROWS output rows per thread block.
//
// WHY. The single-row kernel gives every output row its own block, and every block
// reads the WHOLE activation vector. For K3's KDA projections that is N=12288 blocks
// each pulling K=7168 floats: ~344 MB of activation traffic per projection, against
// ~94 MB of Q8_0 weights. The activation is only 28 KB and sits in L2, but each block
// still fills its own L1 from L2, so the re-read is paid 12288 times.
//
// Co-locating ROWS rows in one block cuts that by ROWS: the activation block is loaded
// once per (b, i) and reused across all ROWS weight rows, so the L1 fill is amortised.
// The weights are untouched — each row still streams its own, which is the traffic that
// is actually irreducible.
//
// Only 4 activation values and ROWS accumulators are held live, rather than staging the
// whole 32-float block in registers: the L2 saving comes from co-locating the rows, not
// from register staging, and staging 32 floats would cost ~32 registers of occupancy to
// buy nothing extra.
//
// BIT-IDENTICAL to the single-row kernel: each row keeps the same thread-to-block
// striding, the same i order, the same four adds per i, and the same block_sum. Only
// which rows share a CUDA block changes.
template <int BLOCK, int ROWS>
__global__ void proj_q8_0_multirow_kernel(float* __restrict__ y,
                                          const float* __restrict__ x,
                                          const BlockQ8_0* __restrict__ W,
                                          int blocks_per_row, int n_rows) {
    k3_pdl_sync();
    const int n0 = blockIdx.x * ROWS;
    __shared__ float shm[BLOCK / 32 + 1];

    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;

    for (int b = threadIdx.x; b < blocks_per_row; b += BLOCK) {
        const float* xb = x + (size_t)b * 32;
        float s[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) s[r] = 0.0f;

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            // One activation load, reused by every row in this block.
            const float x0 = xb[4 * i + 0], x1 = xb[4 * i + 1];
            const float x2 = xb[4 * i + 2], x3 = xb[4 * i + 3];
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                if (n0 + r >= n_rows) continue;
                const BlockQ8_0* row = W + (size_t)(n0 + r) * blocks_per_row;
                const int wq = get_int_b2(row[b].qs, i);
                s[r] += (float)(int8_t)(wq >>  0) * x0;
                s[r] += (float)(int8_t)(wq >>  8) * x1;
                s[r] += (float)(int8_t)(wq >> 16) * x2;
                s[r] += (float)(int8_t)(wq >> 24) * x3;
            }
        }
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (n0 + r >= n_rows) continue;
            const BlockQ8_0* row = W + (size_t)(n0 + r) * blocks_per_row;
            acc[r] += __half2float(__ushort_as_half(row[b].d)) * s[r];
        }
    }

    // block_sum is called by every thread for every row — it contains __syncthreads(),
    // so the call itself must stay uniform; only the store is guarded.
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const float v = block_sum<BLOCK>(acc[r], shm);
        if (threadIdx.x == 0 && n0 + r < n_rows) y[n0 + r] = v;
    }
}

// The multirow kernel above, with the ONE-BARRIER epilogue the tier family proved
// out (proj_q8_fused4_1bar_kernel): the per-row block_sum pays 2*ROWS __syncthreads
// per CTA — 32 at ROWS 16 — where one stash + one barrier + a per-slot ascending-w
// fold produces the SAME partials summed in the SAME order (block_sum's warp tree,
// then w = 0..NWARP ascending), so the result is bit-identical, not re-rounded.
// Activation loads widen to float4 — same values, same use order, fewer issue slots.
// Serves the LM head (the one k3_proj_f32 caller left on the scored path at tp=8),
// selected by SPARKINFER_K3_HEAD_1BAR so one binary A/Bs it.
template <int BLOCK, int ROWS>
__global__ void proj_q8_0_multirow_1bf_kernel(float* __restrict__ y,
                                              const float* __restrict__ x,
                                              const BlockQ8_0* __restrict__ W,
                                              int blocks_per_row, int n_rows) {
    k3_pdl_sync();
    constexpr int NWARP = BLOCK / 32;
    const int n0   = blockIdx.x * ROWS;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __shared__ float shm1[NWARP * ROWS];

    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;

    for (int b = threadIdx.x; b < blocks_per_row; b += BLOCK) {
        const float* xb = x + (size_t)b * 32;
        float s[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) s[r] = 0.0f;

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            // One 16 B activation load, reused by every row in this block. The four
            // lanes are consumed in the same x0..x3 order the scalar loads were.
            const float4 xv = *(const float4*)(xb + 4 * i);
            const float x0 = xv.x, x1 = xv.y, x2 = xv.z, x3 = xv.w;
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                if (n0 + r >= n_rows) continue;
                const BlockQ8_0* row = W + (size_t)(n0 + r) * blocks_per_row;
                const int wq = get_int_b2(row[b].qs, i);
                s[r] += (float)(int8_t)(wq >>  0) * x0;
                s[r] += (float)(int8_t)(wq >>  8) * x1;
                s[r] += (float)(int8_t)(wq >> 16) * x2;
                s[r] += (float)(int8_t)(wq >> 24) * x3;
            }
        }
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (n0 + r >= n_rows) continue;
            const BlockQ8_0* row = W + (size_t)(n0 + r) * blocks_per_row;
            acc[r] += __half2float(__ushort_as_half(row[b].d)) * s[r];
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        float v = acc[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffff, v, off);
        if (lane == 0) shm1[warp * ROWS + r] = v;
    }

    __syncthreads();   // the only one

    if (threadIdx.x < ROWS && n0 + threadIdx.x < n_rows) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < NWARP; ++w) s += shm1[w * ROWS + threadIdx.x];
        y[n0 + threadIdx.x] = s;
    }
}

// Four projections that share one activation, in one launch.
//
// K3's KDA block computes attn_q, attn_k, attn_v and ssm_g from the SAME normed hidden
// state, all at N=qkv=12288, K=hidden=7168. As four separate launches that is four grids
// each re-reading the same 28 KB activation, on all 69 KDA layers, every token.
//
// Fusing them multiplies the activation reuse by the number of tensors on top of what
// multi-row already gives: one load of an activation block now feeds TENSORS x ROWS dot
// products instead of ROWS. ROWS is 2 here rather than the 4 the single-tensor kernel
// uses, because the accumulators are per (tensor, row) — 4x2 live accumulators plus 4x2
// running sums is already more register pressure than the single-tensor path, and the
// amortisation is 8x either way.
//
// BIT-IDENTICAL: every output row keeps the same thread-to-block striding, the same i
// order, the same four adds per i, and the same block_sum. Only which dot products share
// a CUDA block changes.
template <int BLOCK, int ROWS>
__global__ void proj_q8_0_fused4_kernel(float* __restrict__ y0, float* __restrict__ y1,
                                        float* __restrict__ y2, float* __restrict__ y3,
                                        const float* __restrict__ x,
                                        const BlockQ8_0* __restrict__ W0,
                                        const BlockQ8_0* __restrict__ W1,
                                        const BlockQ8_0* __restrict__ W2,
                                        const BlockQ8_0* __restrict__ W3,
                                        int blocks_per_row, int n_rows) {
    k3_pdl_sync();
    const int n0 = blockIdx.x * ROWS;
    __shared__ float shm[BLOCK / 32 + 1];

    const BlockQ8_0* const Wt[4] = {W0, W1, W2, W3};
    float* const Yt[4] = {y0, y1, y2, y3};

    float acc[4][ROWS];
#pragma unroll
    for (int t = 0; t < 4; ++t)
#pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[t][r] = 0.0f;

    for (int b = threadIdx.x; b < blocks_per_row; b += BLOCK) {
        const float* xb = x + (size_t)b * 32;
        float s[4][ROWS];
#pragma unroll
        for (int t = 0; t < 4; ++t)
#pragma unroll
            for (int r = 0; r < ROWS; ++r) s[t][r] = 0.0f;

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            // One activation load, reused by all four tensors and all ROWS rows.
            const float x0 = xb[4 * i + 0], x1 = xb[4 * i + 1];
            const float x2 = xb[4 * i + 2], x3 = xb[4 * i + 3];
#pragma unroll
            for (int t = 0; t < 4; ++t)
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    if (n0 + r >= n_rows) continue;
                    const BlockQ8_0* row = Wt[t] + (size_t)(n0 + r) * blocks_per_row;
                    const int wq = get_int_b2(row[b].qs, i);
                    s[t][r] += (float)(int8_t)(wq >>  0) * x0;
                    s[t][r] += (float)(int8_t)(wq >>  8) * x1;
                    s[t][r] += (float)(int8_t)(wq >> 16) * x2;
                    s[t][r] += (float)(int8_t)(wq >> 24) * x3;
                }
        }
#pragma unroll
        for (int t = 0; t < 4; ++t)
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                if (n0 + r >= n_rows) continue;
                const BlockQ8_0* row = Wt[t] + (size_t)(n0 + r) * blocks_per_row;
                acc[t][r] += __half2float(__ushort_as_half(row[b].d)) * s[t][r];
            }
    }

    // block_sum contains __syncthreads(), so the calls stay uniform across the block;
    // only the stores are guarded.
#pragma unroll
    for (int t = 0; t < 4; ++t)
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const float v = block_sum<BLOCK>(acc[t][r], shm);
            if (threadIdx.x == 0 && n0 + r < n_rows) Yt[t][n0 + r] = v;
        }
}

template <int BLOCK, bool WIDE = false>
__global__ void proj_q8_0_q8_0_kernel(float* __restrict__ y,
                                      const BlockQ8_0* __restrict__ x,
                                      const BlockQ8_0* __restrict__ W,
                                      int blocks_per_row) {
    k3_pdl_sync();
    const int n = blockIdx.x;
    const BlockQ8_0* row = W + (size_t)n * blocks_per_row;
    __shared__ float shm[BLOCK / 32 + 1];
    float acc = 0.0f;
    for (int b = threadIdx.x; b < blocks_per_row; b += BLOCK) {
        // __dp4a does four int8 MACs per instruction: 32 IMAD -> 8 dp4a, on top of
        // the 32 -> 8 load reduction. This is the integer dot product llama.cpp's
        // quantised mat-vecs are built on, and it is why the reference path is fast;
        // writing it as a scalar loop gave up the whole point of quantising the
        // activation. Available from sm_61 — every arch in CMAKE_CUDA_ARCHITECTURES
        // (89;90;100;120;121) qualifies.
        //
        // Bit-identical to the scalar loop: these are exact integer products summed
        // in int32, where addition IS associative, so regrouping cannot change the
        // result. The accumulator cannot overflow either — |sumi| <= 32*127*127 =
        // 516,128, three orders of magnitude inside int32.
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            sumi = __dp4a(get_int_b2_sel(row[b].qs, i, WIDE),
                          get_int_b2_sel(x[b].qs, i, WIDE), sumi);
        const float dw = __half2float(__ushort_as_half(row[b].d));
        const float dx = __half2float(__ushort_as_half(x[b].d));
        acc += (float)sumi * (dw * dx);
    }
    acc = block_sum<BLOCK>(acc, shm);
    if (threadIdx.x == 0) y[n] = acc;
}

// Plain per-block dequant of N CONTIGUOUS values (no reduction) — what a row gather
// needs (token_embd's row for one token), as opposed to k3_proj_f32's per-ROW dot
// Same integer dot product, ROWS output rows per block.
//
// WHY THE dp4a PATH NEEDED THIS AND THE f32 PATH ALREADY HAD IT. k3_proj_f32's
// Q8_0 kernel was widened to ROWS 16/8/4; proj_q8_0_q8_0_kernel never was, and
// still gives every output row its own block. That asymmetry is the whole reason
// the quantised-activation path lost 22% to the f32 one on merged main
// (134.0 vs 110.1 ms/token at ctx 131072, 8x H200) despite doing strictly less
// work per product: 8 __dp4a instead of 32 IMAD, over a 3.76x smaller activation.
//
// THE TRAFFIC THIS FIXES IS NOT WHAT IT LOOKS LIKE. Quantising the activation
// shrinks it from 28 KB to 7.6 KB, which is easy to read as "the activation is
// now small, so amortising it buys little". That is wrong, and it is the mistake
// that killed an earlier fused-4 attempt on this path. The activation is small
// PER BLOCK and is re-read by EVERY block:
//
//     KDA projection, N=12288 rows, K=7168
//       weights     12288 x 224 blocks x 34 B  =  93.6 MB
//       activation  12288 blocks x 7.6 KB      =  93.0 MB   <- same order
//
// So the re-read is still half the traffic, and ROWS cuts it by ROWS. At ROWS 8
// the projection goes from ~187 MB to ~105 MB, a 1.78x cut.
//
// ROWS accumulators plus the 8 staged activation ints is the whole register cost
// here. The earlier attempt held acc[4][ROWS] across FOUR fused tensors on top of
// that and spilled; one tensor at a time is what keeps this in registers.
//
// BIT-IDENTICAL to the single-row kernel: each row keeps the same thread-to-block
// striding, the same i order inside the block, the same int32 accumulation, and
// the same block_sum. Only which rows share a CUDA block changes.
template <int BLOCK, int ROWS>
__global__ void proj_q8_0_q8_0_multirow_kernel(float* __restrict__ y,
                                               const BlockQ8_0* __restrict__ x,
                                               const BlockQ8_0* __restrict__ W,
                                               int blocks_per_row, int n_rows) {
    k3_pdl_sync();
    const int n0 = blockIdx.x * ROWS;
    __shared__ float shm[BLOCK / 32 + 1];

    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;

    for (int b = threadIdx.x; b < blocks_per_row; b += BLOCK) {
        // ONE activation block, staged once and reused by every row in this block.
        // This is the load the single-row kernel repeats in ROWS separate blocks.
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

    // block_sum contains __syncthreads(), so every thread must call it for every
    // row; only the store is guarded.
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const float v = block_sum<BLOCK>(acc[r], shm);
        if (threadIdx.x == 0 && n0 + r < n_rows) y[n0 + r] = v;
    }
}


// Q8 ACTIVATION x FOUR Q8_0 WEIGHT MATRICES, one launch.
//
// attn_q / attn_k / attn_v / ssm_g all read the SAME s.normed at the same
// [qkv, H] shape on every one of the 69 KDA layers. The f32-activation path has
// had proj_q8_0_fused4_kernel for exactly this; the Q8-activation path did not,
// so turning on quantised activations silently dropped those layers to four
// separate projections -- and k3_proj_ggml_f32 quantises its input per call, so
// the identical vector was quantised four times per layer per rank.
//
// That is what made quantize_q8_0 59,696 launches and 8.7% of GPU time at
// ~4.9 us for ~36 KB apiece -- roughly 7 GB/s on a 4.8 TB/s part. It was not
// doing work; it was paying launch overhead.
//
// The obvious fix -- quantise once and let the other three reuse the scratch --
// was tried and REVERTED: it needed the caller to promise the activation had not
// changed, the guard I wrote checked the wrong thing (which pointer the scratch
// was last written from, not whether the bytes behind it still matched), and the
// model emitted top1 0.0 while getting faster. This shape needs no promise. One
// kernel stages one activation block and consumes it four times before it can go
// anywhere, so there is no window in which it can go stale.
template <int BLOCK, int ROWS>
__global__ void proj_q8_0_q8_0_fused4_kernel(float* __restrict__ y0,
                                             float* __restrict__ y1,
                                             float* __restrict__ y2,
                                             float* __restrict__ y3,
                                             const BlockQ8_0* __restrict__ x,
                                             const BlockQ8_0* __restrict__ W0,
                                             const BlockQ8_0* __restrict__ W1,
                                             const BlockQ8_0* __restrict__ W2,
                                             const BlockQ8_0* __restrict__ W3,
                                             int blocks_per_row, int n_rows) {
    k3_pdl_sync();
    const int n0 = blockIdx.x * ROWS;
    __shared__ float shm[BLOCK / 32 + 1];

    const BlockQ8_0* const Wt[4] = {W0, W1, W2, W3};
    float* const Yt[4] = {y0, y1, y2, y3};

    float acc[4][ROWS];
#pragma unroll
    for (int t = 0; t < 4; ++t)
#pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[t][r] = 0.0f;

    for (int b = threadIdx.x; b < blocks_per_row; b += BLOCK) {
        // ONE staged activation block, consumed by 4 tensors x ROWS rows. The
        // separate-launch path re-read (and re-quantised) this per tensor.
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

    // block_sum contains __syncthreads(), so every thread must reach every call;
    // only the store is guarded. Same contract as the multirow kernel.
#pragma unroll
    for (int t = 0; t < 4; ++t) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const float v = block_sum<BLOCK>(acc[t][r], shm);
            if (threadIdx.x == 0 && n0 + r < n_rows) Yt[t][n0 + r] = v;
        }
    }
}


// product. Shares the same block layouts.
__global__ void dequant_q8_0_kernel(float* __restrict__ out,
                                    const BlockQ8_0* __restrict__ blocks,
                                    int64_t n_blocks) {
    k3_pdl_sync();
    const int64_t b = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const float d = __half2float(__ushort_as_half(blocks[b].d));
    float* dst = out + b * 32;
#pragma unroll
    for (int j = 0; j < 32; ++j) dst[j] = (float)blocks[b].qs[j] * d;
}

__global__ void dequant_f32_passthrough_kernel(float* __restrict__ out,
                                               const float* __restrict__ src,
                                               int64_t n) {
    k3_pdl_sync();
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = src[i];
}

template <int BLOCK>
__global__ void proj_f32_kernel(float* __restrict__ y, const float* __restrict__ x,
                                const float* __restrict__ W, int K) {
    k3_pdl_sync();
    const int n = blockIdx.x;
    const float* row = W + (size_t)n * K;
    __shared__ float shm[BLOCK / 32 + 1];

    // ISSUE EIGHT LOADS BEFORE THE FIRST FMA. These loads are perfectly coalesced
    // (128 B/warp, one wavefront), so this kernel is not instruction-bound like the
    // Q8_0 GEMVs -- it is DRAM-LATENCY-bound, and the only lever is how many loads
    // are outstanding per warp. At the router shape (N=896, K=7168) the grid is
    // 896x4 warps / 132 SM = 27.2 warps/SM; sustaining the measured 36.4 GB/s/SM at
    // ~600 ns of latency needs ~21.8 KB in flight, i.e. ~6 outstanding loads per
    // warp. The measured 2.15 TB/s back-solves to ~4 -- which is exactly what nvcc
    // emits by default for a runtime-bounded loop with one serial accumulator,
    // because it cannot prove the trip count. Naming the unroll lifts it to 8.
    //
    // BIT-IDENTICAL, unconditionally: `acc +=` is a serial dependency chain, so the
    // summation order is identical at ANY unroll factor -- unrolling cannot
    // reassociate what the data dependency already serialises. --use_fast_math is on
    // (CMakeLists.txt:9), which enables FMA contraction and ftz but never
    // reassociation, and both arms here contract the same way.
    float acc = 0.0f;
    int k = threadIdx.x;
    for (; k + 7 * BLOCK < K; k += 8 * BLOCK) {
        float w[8], v[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) { w[u] = row[k + u * BLOCK]; v[u] = x[k + u * BLOCK]; }
#pragma unroll
        for (int u = 0; u < 8; ++u) acc += w[u] * v[u];
    }
    for (; k < K; k += BLOCK) acc += row[k] * x[k];
    acc = block_sum<BLOCK>(acc, shm);
    if (threadIdx.x == 0) y[n] = acc;
}


__global__ void add_f32_kernel(float* __restrict__ out, const float* __restrict__ a,
                               const float* __restrict__ b, int64_t n) {
    k3_pdl_sync();
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void sigmoid_inplace_f32_kernel(float* __restrict__ x, int64_t n) {
    k3_pdl_sync();
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) x[i] = 1.0f / (1.0f + __expf(-x[i]));
}

// Writes this token's MLA K-cache row at a position only the DEVICE knows.
//
// Replaces a host-computed `mla_kv_cache + position * key_length` fed to two
// cudaMemcpyAsync. Those bake the destination ADDRESS into a captured graph, so replay
// would rewrite one row forever while the model read stale keys for every later token —
// correct on replay 1, wrong from replay 2. Concatenation order (normed kv_cmpr, then RAW
// k_pe) is exactly the two memcpys it replaces; this moves bytes and does no arithmetic,
// so it is bit-identical by construction.
__global__ void mla_kv_store_kernel(float* __restrict__ cache,
                                    const float* __restrict__ kv_cmpr_normed,
                                    const float* __restrict__ kv_a_out,
                                    const int* __restrict__ d_pos,
                                    int kv_lora, int rope_dim, int key_length) {
    k3_pdl_sync();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kv_lora + rope_dim) return;
    float* row = cache + (size_t)(*d_pos) * (size_t)key_length;
    row[i] = (i < kv_lora) ? kv_cmpr_normed[i] : kv_a_out[i];
}

// Advance the device's view of the position, inside the captured region.
// The host mirror still advances too — it is what picks the launch plan, which a graph
// cannot change. k3_read_pos_f32 exists so a test can prove the two never drift.
__global__ void bump_pos_kernel(int* __restrict__ p) { *p += 1; }

// The same advance by an arbitrary signed step. A tile driver runs the layer loop
// OUTSIDE the token loop, so within one layer the position walks base..base+T-1 and
// then has to come back to base for the next layer — which is a -(T-1), not a bump.
//
// RELATIVE, and that is the whole reason this is not a `set`. A set would bake the
// absolute position into the launch, and a captured graph replayed on the next tile
// would rewind every token to the tile it was recorded at: fluent output, wrong KV
// rows, invisible to timing. An add is the same instruction on every tile.
__global__ void add_pos_kernel(int* __restrict__ p, int d) { *p += d; }

// Original single-block norm. Kept as the bit-identical reference path: the sum of
// squares is partitioned across exactly 128 threads with stride 128, and the apply
// walk matches. Widening the sum changes that partition and moves KL vs main past the
// ratchet (#115 measured 4.63x at one depth) — so the wide kernel below freezes this
// reduction and only widens the elementwise apply.
template <int BLOCK>
__global__ void rms_norm_kernel(float* __restrict__ out, const float* __restrict__ x,
                                const float* __restrict__ w, int n, float eps) {
    k3_pdl_sync();
    __shared__ float shm[BLOCK / 32 + 1];
    float acc = 0.0f;
    for (int d = threadIdx.x; d < n; d += BLOCK) acc += x[d] * x[d];
    const float ss = block_sum<BLOCK>(acc, shm);
    const float inv = rsqrtf(ss / (float)n + eps);
    for (int d = threadIdx.x; d < n; d += BLOCK) out[d] = x[d] * inv * w[d];
}

// Wide apply, frozen reduction.
//
// WHY THE SUM STAYS ON 128 THREADS. Changing how many threads contribute to ss changes
// the float32 association of the sum of squares, which changes inv, which changes every
// output element by ~1e-7 relative — enough to push KL vs main over the 2x ratchet while
// still clearing the absolute KL bar. The measured +2.8% at 128k on #115 came with
// accuracy-regression for exactly that reason. The apply is elementwise once inv is
// fixed, so widening it (more threads, float4 loads) cannot change a bit of the output
// if the reduction matches rms_norm_kernel<128>.
//
// Idle threads above 128 contribute 0 to block_sum; x+0 is an IEEE identity, so the
// longer warp-tree sum is bit-identical to the 4-warp one.
//
// SPAN_UNITS slices the apply across CTAs on top of the widening: the apply moves
// three times the bytes of the reduction and cannot start until it lands, and a
// single CTA — however wide — is one SM of 132. Every CTA recomputes the identical
// frozen-128 reduction, so `inv` agrees bit-for-bit across the grid; CTA g then
// writes only units [g*span, (g+1)*span), so each output element is still produced
// by exactly one thread from the same product in the same order. span_units == the
// full unit count with grid 1 is exactly the single-CTA form.
//
// UNROLL names the load count of the reduction sweep. `acc +=` is a serial
// dependency chain, so summation order is identical at any unroll factor — the same
// argument proj_f32_kernel relies on, and it matters more here because nothing else
// is resident to cover the reduction's latency.
//
// n4 is the float4 count when vec4!=0; n is always the element count (the mean divisor).
// A TOKEN AXIS ON gridDim.y, and it is deliberately an OFFSET rather than a second kernel.
//
// rms_norm_wide is 302 graph nodes per prefill token — 9% of a captured tile's graph, and
// a captured tile is bound by its node count. Batching it over T tokens turns T launches
// into one. The prefill module's header declined to batch the norms because doing it the
// obvious way means copying rms_norm_block_for and block_sum into another file, and a
// duplicated tier is exactly what produced the block_for defect the prefill CPU test now
// pins. This avoids that entirely: same kernel, same block_for, same block_sum, same
// reduction order, same span logic — only the base pointers move.
//
// Bit-identical to the single-row launch by construction. Every block already recomputes
// the whole row's sum of squares, so nothing about a row's arithmetic depends on how many
// other rows are in flight, and at gridDim.y == 1 blockIdx.y is 0 and the offset vanishes.
template <int BLOCK, bool UNROLL>
__global__ void rms_norm_wide_kernel(float* __restrict__ out, const float* __restrict__ x,
                                     const float* __restrict__ w, int n, float eps,
                                     int n4, int vec4, int span_units,
                                     long long row_stride) {
    k3_pdl_sync();
    __shared__ float shm[BLOCK / 32 + 1];
    // Row this block serves. Zero-cost on the single-row path.
    const long long row_off = (long long)blockIdx.y * row_stride;
    x   += row_off;
    out += row_off;

    float acc = 0.0f;
    if (threadIdx.x < 128) {
        int d = (int)threadIdx.x;
        if (UNROLL) {
            for (; d + 7 * 128 < n; d += 8 * 128) {
                float v[8];
#pragma unroll
                for (int u = 0; u < 8; ++u) v[u] = x[d + u * 128];
#pragma unroll
                for (int u = 0; u < 8; ++u) acc += v[u] * v[u];
            }
        }
        for (; d < n; d += 128) acc += x[d] * x[d];
    }
    const float ss = block_sum<BLOCK>(acc, shm);
    const float inv = rsqrtf(ss / (float)n + eps);

    const int units = vec4 ? n4 : n;
    const int u0 = blockIdx.x * span_units;
    const int u1 = min(u0 + span_units, units);
    if (vec4) {
        const float4* __restrict__ x4 = (const float4*)x;
        const float4* __restrict__ w4 = (const float4*)w;
        float4* __restrict__ o4 = (float4*)out;
        for (int d = u0 + (int)threadIdx.x; d < u1; d += BLOCK) {
            const float4 v = x4[d];
            const float4 g = w4[d];
            float4 r;
            r.x = v.x * inv * g.x;
            r.y = v.y * inv * g.y;
            r.z = v.z * inv * g.z;
            r.w = v.w * inv * g.w;
            o4[d] = r;
        }
    } else {
        for (int d = u0 + (int)threadIdx.x; d < u1; d += BLOCK)
            out[d] = x[d] * inv * w[d];
    }
}

}  // namespace

// ---------------------------------------------------------------------------

void situ_f32(float* out, const float* gate, const float* up, int64_t n,
              float beta, float linear_beta, cudaStream_t stream) {
    if (n <= 0) return;
    const int T = 256;
    const int64_t blocks = (n + T - 1) / T;
    const int lb_active = linear_beta > 0.0f ? 1 : 0;
    k3_pdl_launch((unsigned)blocks, T, 0, stream, situ_kernel, 
        out, gate, up, n, beta, 1.0f / beta, linear_beta,
        lb_active ? 1.0f / linear_beta : 1.0f, lb_active);
}

void kda_decode_step_f32(float* out, float* state,
                         const float* q, const float* k, const float* v,
                         const float* g, const float* beta,
                         int head_dim, int n_head, cudaStream_t stream) {
    if (head_dim <= 0 || n_head <= 0) return;
    // One thread per state column. K3's head_dim is 128.
    const int T = head_dim;
    // s_k, s_q, s_sk, s_ge
    const size_t shm = (size_t)4 * head_dim * sizeof(float);

    // s_k, s_q, s_sk, s_ge, plus head_dim rows of (IC + 1) for the staged chunk.
    // 18.5 KB at K3's dims — under the 48 KB default, so there is no opt-in to fail and
    // no fallback path whose use would be invisible in the output.
    //
    // head_dim == BLOCK is a REQUIREMENT, not a convenience. The launch uses head_dim
    // threads, and the staged kernel's copy loops stride by BLOCK; at head_dim < BLOCK
    // the columns between head_dim and BLOCK would never be written and the block would
    // reduce over stale shared memory. The unstaged kernel below indexes by threadIdx.x
    // alone and is correct at any width, so it stays the fallback.
    constexpr int IC = 32, SMEM_BLOCK = 128;
    const size_t shm_staged =
        ((size_t)4 * head_dim + (size_t)head_dim * (IC + 1)) * sizeof(float);

    // Value tiling. grid (n_head,) -> (n_head, head_dim/BV): see the kernel comment for
    // why one block per head strands the device once KDA is head-sharded.
    //
    // SPARKINFER_K3_KDA_VT=0 restores the single-tile launch, which is how the two are
    // A/B'd on ONE binary. It defaults ON because the eval harness runs a default build
    // and an env-gated win scores as zero — the same trap that made #63's qact path and
    // #67's fusion guard invisible until someone read the profile.
    constexpr int BV = 32;
    static const bool want_vt = [] {
        const char* e = std::getenv("SPARKINFER_K3_KDA_VT");
        return !(e && e[0] == '0');
    }();
    if (want_vt && head_dim == SMEM_BLOCK && head_dim % IC == 0 && head_dim % BV == 0) {
        const size_t shm_vt =
            ((size_t)3 * head_dim + (size_t)BV * (IC + 1)) * sizeof(float);
        const dim3 grid((unsigned)n_head, (unsigned)(head_dim / BV));
        k3_pdl_launch(grid, BV, shm_vt, stream, kda_decode_step_vt_kernel<BV>, 
            out, state, q, k, v, g, beta, head_dim);
        return;
    }
    if (head_dim == SMEM_BLOCK && head_dim % IC == 0) {
        k3_pdl_launch((unsigned)n_head, T, shm_staged, stream, kda_decode_step_smem_kernel<SMEM_BLOCK>, 
            out, state, q, k, v, g, beta, head_dim);
        return;
    }
    k3_pdl_launch((unsigned)n_head, T, shm, stream, kda_decode_step_kernel<128>, 
        out, state, q, k, v, g, beta, head_dim);
}

void kda_gate_out_f32(float* out, const float* o, const float* norm_w,
                      const float* g2, int head_dim, int n_head,
                      float eps, cudaStream_t stream) {
    if (head_dim <= 0 || n_head <= 0) return;
    k3_pdl_launch((unsigned)n_head, 128, 0, stream, kda_gate_out_kernel<128>, 
        out, o, norm_w, g2, head_dim, eps);
}

void attn_res_mix_f32(float* out, const float* ckpts, const float* cur,
                      const float* score_w, int n_embd, int n_ckpt,
                      float eps, cudaStream_t stream, float* scores,
                      const float* cur_b, float* sum_out,
                      int n_rows, int64_t act_stride,
                      int64_t bank_stride, int64_t score_stride) {
    if (n_embd <= 0 || n_rows <= 0) return;
    if (act_stride == 0) act_stride = n_embd;
    if (bank_stride == 0) bank_stride = (int64_t)std::max(n_ckpt, 1) * n_embd;
    if (score_stride == 0) score_stride = n_ckpt + 1;
    if (n_ckpt <= 0) {
        // Layer 0 / nothing banked: the reference returns cur unchanged. With a
        // residual pair in flight the sum still has to be materialised — same two
        // launches the unfused path used, so this branch never regresses.
        if (cur_b && sum_out && act_stride == n_embd) {
            const int64_t count = (int64_t)n_rows * n_embd;
            k3_add_f32(sum_out, cur, cur_b, count, stream);
            cudaMemcpyAsync(out, sum_out, (size_t)count * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
            return;
        }
        if (cur_b && sum_out) {
            for (int r = 0; r < n_rows; ++r)
                k3_add_f32(sum_out + (int64_t)r * act_stride,
                           cur + (int64_t)r * act_stride,
                           cur_b + (int64_t)r * act_stride, n_embd, stream);
            cudaMemcpy2DAsync(out, (size_t)act_stride * sizeof(float), sum_out,
                              (size_t)act_stride * sizeof(float),
                              (size_t)n_embd * sizeof(float), n_rows,
                              cudaMemcpyDeviceToDevice, stream);
            return;
        }
        if (sum_out)
            cudaMemcpy2DAsync(sum_out, (size_t)act_stride * sizeof(float), cur,
                              (size_t)act_stride * sizeof(float),
                              (size_t)n_embd * sizeof(float), n_rows,
                              cudaMemcpyDeviceToDevice, stream);
        cudaMemcpy2DAsync(out, (size_t)act_stride * sizeof(float), cur,
                          (size_t)act_stride * sizeof(float),
                          (size_t)n_embd * sizeof(float), n_rows,
                          cudaMemcpyDeviceToDevice, stream);
        return;
    }
    constexpr int B = 256;
    // `scores` is caller-owned when the caller has a per-forward scratch to lend
    // (the decode path does, and passes it on every call). The fallback keeps the
    // old stream-ordered allocation so callers without scratch — the numeric test,
    // and the pipeline/TP entry points that mix once per token rather than 185
    // times — need no change.
    float* sc = scores;
    const bool owned = (sc == nullptr);
    if (owned) {
        score_stride = n_ckpt + 1;
        cudaMallocAsync(&sc, (size_t)n_rows * score_stride * sizeof(float), stream);
    }

    static const bool res_1pass = [] {
        const char* e = std::getenv("SPARKINFER_K3_RES_1PASS");
        return !(e && e[0] == '0');
    }();
    // THE SCORE GRID IS n_ckpt+1 — TWO BLOCKS AT THE SCORED CONTEXT.
    //
    // The blend below wants a NARROW block, because its grid is n_embd/B and 256 keeps
    // that at 28 blocks. The score kernel is the opposite shape: its grid is fixed at
    // the checkpoint count, so 256 threads is all the parallelism it ever gets, against
    // a full n_embd sweep per block. Those are different constraints and they were
    // sharing one constant. Giving the score its own width raises loads in flight 4x
    // without touching the blend's grid — same arithmetic per thread, three more
    // levels on the reduction tree.
    //
    // SPARKINFER_K3_RES_WIDE=0 restores the shared 256 on one binary.
    static const bool res_wide = [] {
        const char* e = std::getenv("SPARKINFER_K3_RES_WIDE");
        return !(e && e[0] == '0');
    }();
    constexpr int SB = 1024;
    if (res_wide && n_embd >= SB) {
        if (res_1pass)
            k3_pdl_launch(dim3((unsigned)(n_ckpt + 1), (unsigned)n_rows), SB, 0, stream,
                attn_res_score_kernel<SB, true>, sc, ckpts, cur, cur_b, score_w,
                n_embd, n_ckpt, eps, act_stride, bank_stride, score_stride);
        else
            k3_pdl_launch(dim3((unsigned)(n_ckpt + 1), (unsigned)n_rows), SB, 0, stream,
                attn_res_score_kernel<SB, false>, sc, ckpts, cur, cur_b, score_w,
                n_embd, n_ckpt, eps, act_stride, bank_stride, score_stride);
    } else if (res_1pass)
        k3_pdl_launch(dim3((unsigned)(n_ckpt + 1), (unsigned)n_rows), B, 0, stream,
            attn_res_score_kernel<B, true>, sc, ckpts, cur, cur_b, score_w,
            n_embd, n_ckpt, eps, act_stride, bank_stride, score_stride);
    else
        k3_pdl_launch(dim3((unsigned)(n_ckpt + 1), (unsigned)n_rows), B, 0, stream,
            attn_res_score_kernel<B, false>, sc, ckpts, cur, cur_b, score_w,
            n_embd, n_ckpt, eps, act_stride, bank_stride, score_stride);
    k3_pdl_launch(dim3((unsigned)((n_embd + B - 1) / B), (unsigned)n_rows), B,
                      (size_t)(n_ckpt + 1) * sizeof(float), stream,
                      attn_res_apply_kernel<B>, out, ckpts, cur, cur_b, sum_out, sc,
                      n_embd, n_ckpt, act_stride, bank_stride, score_stride);

    if (owned) cudaFreeAsync(sc, stream);
}

bool k3_kda_conv_l2_fused(float* out_q, float* out_k, float* out_v,
                          float* st_q, float* st_k, float* st_v,
                          const float* x_q, const float* x_k, const float* x_v,
                          const float* w_q, const float* w_k, const float* w_v,
                          int d_conv, int head_dim, int n_head,
                          float q_scale, float eps, cudaStream_t stream) {
    // SPARKINFER_K3_KDACONV=0 restores the five separate launches on ONE binary, so
    // this fold's contribution can be isolated from everything else in the same
    // measurement session rather than inferred from a bundle.
    static const bool want = [] {
        const char* e = std::getenv("SPARKINFER_K3_KDACONV");
        return !(e && e[0] == '0');
    }();
    if (!want) return false;
    // One element per thread per head is what makes the reduction identical to the
    // kernel this replaces; wider heads would need the strided form, so decline and
    // let the caller keep the five-launch path.
    constexpr int BLOCK = 128;
    if (d_conv < 2 || head_dim <= 0 || head_dim > BLOCK || n_head <= 0) return false;
    dim3 grid((unsigned)n_head, 3u);
    k3_pdl_launch(grid, BLOCK, 0, stream, kda_conv_l2_fused_kernel<BLOCK>,
        out_q, out_k, out_v, st_q, st_k, st_v, x_q, x_k, x_v, w_q, w_k, w_v,
        d_conv, head_dim, q_scale, eps);
    return true;
}

void kda_conv_step_f32(float* out, float* state, const float* x, const float* w,
                       int d_conv, int d_inner, cudaStream_t stream) {
    if (d_conv < 2 || d_inner <= 0) return;
    const int T = 256;
    const int blocks = (d_inner + T - 1) / T;
    k3_pdl_launch((unsigned)blocks, T, 0, stream, kda_conv_step_kernel, out, state, x, w, d_conv, d_inner);
}

// Upload the lattice/sign tables once. __device__ (NOT __constant__) is the right
// home — see the note on the declarations above. Formerly:  every
// thread in a warp reads the same grid entry for a given codepoint, so the
// broadcast path is what we want rather than L1 thrash.
// THE LATTICE TABLES ARE PER-DEVICE STATE, SO THE GUARD MUST BE TOO.
//
// c_iq2xs_grid / iq1s_grid_c live in __constant__ / __device__ memory, and
// cudaMemcpyToSymbol writes THE CURRENT DEVICE'S copy. A single process-global
// `static bool ready` therefore uploads the tables to whichever device happened to be
// current on the first call and silently skips every other one.
//
// On one GPU that is invisible. On a multi-GPU run it is catastrophic and quiet:
// ranks 1..N-1 decode IQ1_S / IQ2_XS against a table of ZEROS, so every routed expert
// on those ranks returns ~0. The model still runs, the all-reduce still sums, and the
// output is a mixture missing (N-1)/N of its experts — fluent, plausible, wrong.
//
// MEASURED, not theorised: at tp=2 this made rank 1's expert partial ~0 while rank 0's
// was correct, so sum(bands) came to 14.32 against the single-GPU 35.97. It was
// invisible to the band test, which simulates every rank on device 0.
//
// Indexed by device ordinal. 64 covers any single node; a device beyond that falls
// through to uploading every call, which is slow but correct.
constexpr int kMaxTableDevices = 64;

static int current_device_index() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return -1;
    return (dev >= 0 && dev < kMaxTableDevices) ? dev : -1;
}

static void ensure_iq2xs_tables() {
    static bool ready[kMaxTableDevices] = {false};
    const int dev = current_device_index();
    if (dev >= 0 && ready[dev]) return;
    cudaMemcpyToSymbol(c_iq2xs_grid,   h_iq2xs_grid,   sizeof(h_iq2xs_grid));
    cudaMemcpyToSymbol(c_ksigns_iq2xs, h_ksigns_iq2xs, sizeof(h_ksigns_iq2xs));
    cudaMemcpyToSymbol(c_kmask_iq2xs,  h_kmask_iq2xs,  sizeof(h_kmask_iq2xs));
    if (dev >= 0) ready[dev] = true;
}

// The SAME lattice at 2 bytes per codepoint instead of 8. Every byte of every
// entry is in {0x00, 0x01, 0xff} (the header's tripwire), i.e. {0, 1, -1} — two
// bits each in two's complement. Why it can pay: the 11-bit index is fully
// divergent across a warp, so a gather's L1 cost is the number of DISTINCT
// 128 B lines touched. A 16 KB table spans 128 lines -> E[distinct] ~ 28.4 per
// warp; a 4 KB table spans 32 -> ~20.4 (-28%), purely from intra-warp line
// coincidence. It also halves the registers holding the grid word. Packed
// HOST-SIDE from the same iq1s_grid_host, so the extracted table remains the
// single source of truth and the tripwire survives.
static void ensure_iq1s_tables() {
    static bool ready[kMaxTableDevices] = {false};
    const int dev = current_device_index();
    if (dev >= 0 && ready[dev]) return;
    cudaMemcpyToSymbol(iq1s_grid_c, iq1s_grid_host, sizeof(iq1s_grid_host));
    static uint16_t packed[SPARKINFER_IQ1S_NGRID];
    static bool packed_ready = false;
    if (!packed_ready) {
        for (int i = 0; i < SPARKINFER_IQ1S_NGRID; ++i) {
            const uint64_t gw = iq1s_grid_host[i];
            uint16_t v = 0;
            for (int j = 0; j < 8; ++j) {
                const int8_t gj = (int8_t)((gw >> (8 * j)) & 0xffu);
                v |= (uint16_t)((uint16_t)gj & 3u) << (2 * j);
            }
            packed[i] = v;
        }
        packed_ready = true;
    }
    cudaMemcpyToSymbol(iq1s_grid_p, packed, sizeof(packed));
    if (dev >= 0) ready[dev] = true;
}

// UPLOAD THE LATTICE TABLES BEFORE ANY CAPTURE BEGINS.
//
// ensure_iq2xs_tables()/ensure_iq1s_tables() are lazy and per-device, and they upload with
// cudaMemcpyToSymbol — the SYNCHRONOUS form, which is illegal inside a stream capture.
// Under UD-IQ1_S that first upload lands on the first MoE layer (layer 0 is leading_dense
// and has no IQ1_S expert weights), so a capture that recorded layer 0 cleanly was being
// invalidated at layer 1 by a one-time initialisation that has nothing to do with the layer.
//
// The failure is also maximally misleading: cudaGetLastError reports "operation failed due
// to a previous error during capture" from the NEXT launch, so it accuses the MoE dispatch.
//
// Same hazard class as k3_mla_prewarm_split_scratch, same answer: do it eagerly at init,
// with each device current, so the in-capture call hits its `ready[dev]` early return.
void k3_prewarm_quant_tables() {
    ensure_iq2xs_tables();
    ensure_iq1s_tables();
    // WEPS engagement counting, opt-in and uploaded here because this function runs
    // once per device BEFORE any capture — the same synchronous-copy-legality window
    // the table uploads need. Default off: the scored path never pays the atomics.
    static const int count = [] {
        const char* e = std::getenv("SPARKINFER_K3_WEPS_COUNT");
        return (e && e[0] == '1') ? 1 : 0;
    }();
    if (count)
        cudaMemcpyToSymbol(g_k3_weps_count, &count, sizeof(int));
}

// Bind the depth gate on the CURRENT device. Called from state init, with the
// rank's device current and before any capture — the same synchronous-copy window
// the quant-table prewarm uses. Passing the d_pos the forward already maintains
// keeps a single source of truth for the position.
void k3_weps_bind_pos(const int* d_pos) {
    static const int minp = [] {
        const char* e = std::getenv("SPARKINFER_K3_WEPS_MINPOS");
        return e ? atoi(e) : 16384;
    }();
    cudaMemcpyToSymbol(g_k3_weps_pos, &d_pos, sizeof(d_pos));
    cudaMemcpyToSymbol(g_k3_weps_min_pos, &minp, sizeof(int));
}

// Engagement readout for the CURRENT device; the caller loops ranks. Reset-on-read
// so consecutive probes (per-depth KL runs) need no separate clear call.
unsigned long long k3_weps_skips_read_current_device() {
    unsigned long long v = 0;
    cudaMemcpyFromSymbol(&v, g_k3_weps_skips, sizeof(v));
    const unsigned long long zero = 0;
    cudaMemcpyToSymbol(g_k3_weps_skips, &zero, sizeof(zero));
    return v;
}

void dequant_iq2_xs_f32(float* out, const void* src, int64_t n, cudaStream_t stream) {
    if (n <= 0) return;
    if (n % 256) {
        fprintf(stderr, "[k3] dequant_iq2_xs: n=%lld not a multiple of 256\n",
                     (long long)n);
        return;
    }
    ensure_iq2xs_tables();
    const int64_t n_groups = n / 8;
    const int T = 256;
    const int64_t blocks = (n_groups + T - 1) / T;
    k3_pdl_launch((unsigned)blocks, T, 0, stream, dequant_iq2_xs_kernel, 
        out, (const BlockIQ2XS*)src, n_groups);
}

void moe_router_noaux_tc_f32(float* out_w, int* out_ids, const float* logits,
                             const float* bias, int n_expert, int top_k,
                             int n_tokens, bool norm_w, float w_scale,
                             cudaStream_t stream) {
    if (n_expert <= 0 || top_k <= 0 || n_tokens <= 0) return;
    if (top_k > n_expert) return;
    const size_t shm = (size_t)2 * n_expert * sizeof(float);
    const int T = 256;
    k3_pdl_launch((unsigned)n_tokens, T, shm, stream, moe_router_noaux_tc_kernel<256>,
        out_w, out_ids, logits, bias, n_expert, top_k, norm_w, w_scale);
}

// One launch path for both quant types. The two front doors below differ only in the
// table they prime and the block layout they name; duplicating the grid arithmetic
// across them is the same drift the block_dot overloads exist to prevent.
template <int DOWN_WARPS, typename Blk>
static void moe_expert_ffn_launch_w(float* out, float* scratch,
                                  const float* x, const int* ids, const float* w,
                                  const void* gate_exps, const void* up_exps,
                                  const void* down_exps,
                                  int latent, int ffn, int top_k,
                                  float situ_beta, float situ_linear_beta,
                                  cudaStream_t stream,
                                  int expert_begin, int n_local_experts) {
    constexpr int GATE_WARPS = 8;
    const int lb_active = situ_linear_beta > 0.0f ? 1 : 0;
    // n_local <= 0 means "this rank holds every expert" — the tp_size 1 case, where
    // the band test must never reject. INT_MAX makes it a no-op rather than a branch.
    const int n_local = n_local_experts > 0 ? n_local_experts : INT_MAX;

    // The "foreign slots read as zero" invariant is now established ONCE, when the
    // scratch is allocated (kimi_k3.cpp), not on every layer.
    //
    // It holds just as strongly there, and the reason is that the set of foreign slots
    // never changes: a rank's expert band is fixed for the run, and moe_gate_up writes
    // only its OWN slots. A slot foreign to this rank is therefore written by nobody,
    // ever — so a single zeroing at allocation leaves it zero for the whole run, which
    // is exactly what re-zeroing it 92 times a token was buying. (Owned slots are fully
    // overwritten before they are read, so clearing them was never load-bearing either.)
    //
    // That removes 92 launches and ~18 MB of writes per token per rank from the critical
    // path while keeping the guarantee moe_down_combine's band re-test relies on.

    // float4 activation reads need 16-byte alignment. Both pointers are cudaMalloc
    // bases in every K3 caller and the per-block offsets are multiples of 1024 bytes,
    // so the bases alone decide it.
    const bool xvec = ((((uintptr_t)x) | ((uintptr_t)scratch)) & 15u) == 0;

    const dim3 g1((unsigned)((ffn + GATE_WARPS - 1) / GATE_WARPS), (unsigned)top_k);
    // partial[top_k] + the staged ids/w epilogue operands (see the kernel).
    const size_t dshm = (size_t)top_k * (2 * sizeof(float) + sizeof(int));
    const float inv_lb = lb_active ? 1.0f / situ_linear_beta : 1.0f;

    if (xvec) {
        // SPARKINFER_K3_IQ1S_PACK=1 reads the lattice through the 4 KB packed
        // table (see iq1s_grid_p). DEFAULT OFF until measured — same one-binary
        // A/B convention as every other gate on this branch.
        static const bool gpack = [] {
            // MEASURED +0.72 tok/s (3 sessions). Default ON; =0 restores.
            const char* e = std::getenv("SPARKINFER_K3_IQ1S_PACK");
            return !(e && e[0] == '0');
        }();
        // SPARKINFER_K3_IQ1S_SMEM=1 additionally stages the packed table in
        // shared (gate/up only: 224 gathers/CTA amortise the 4 KB copy ~14:1;
        // down_combine's 24 do not). DEFAULT OFF until measured.
        static const bool gsmem = [] {
            const char* e = std::getenv("SPARKINFER_K3_IQ1S_SMEM");
            return e && e[0] == '1';
        }();
        static const bool rebal = [] {
            // Live-expert rebalance across warps. Bit-identical; default ON.
            // SPARKINFER_K3_MOE_REBAL=0 restores the static warp→k map.
            const char* e = std::getenv("SPARKINFER_K3_MOE_REBAL");
            return !(e && e[0] == '0');
        }();
        if (gpack && gsmem) {
            k3_pdl_launch(g1, GATE_WARPS * 32, sizeof(uint16_t) * SPARKINFER_IQ1S_NGRID,
                stream, moe_gate_up_situ_kernel<GATE_WARPS, true, Blk, true, true>,
                scratch, x, ids, (const Blk*)gate_exps, (const Blk*)up_exps, latent, ffn,
                situ_beta, 1.0f / situ_beta, situ_linear_beta, inv_lb, lb_active,
                expert_begin, n_local, w, k3_moe_weps_host());
            if (rebal)
                k3_pdl_launch((unsigned)latent, DOWN_WARPS * 32, dshm, stream, moe_down_combine_kernel<DOWN_WARPS, true, Blk, true, true>,
                    out, scratch, ids, w, (const Blk*)down_exps, latent, ffn, top_k,
                    expert_begin, n_local, k3_moe_weps_host());
                else
                k3_pdl_launch((unsigned)latent, DOWN_WARPS * 32, dshm, stream, moe_down_combine_kernel<DOWN_WARPS, true, Blk, true>,
                    out, scratch, ids, w, (const Blk*)down_exps, latent, ffn, top_k,
                    expert_begin, n_local, k3_moe_weps_host());
        } else if (gpack) {
            k3_pdl_launch(g1, GATE_WARPS * 32, 0, stream, moe_gate_up_situ_kernel<GATE_WARPS, true, Blk, true>,
                scratch, x, ids, (const Blk*)gate_exps, (const Blk*)up_exps, latent, ffn,
                situ_beta, 1.0f / situ_beta, situ_linear_beta, inv_lb, lb_active,
                expert_begin, n_local, w, k3_moe_weps_host());
            if (rebal)
                k3_pdl_launch((unsigned)latent, DOWN_WARPS * 32, dshm, stream, moe_down_combine_kernel<DOWN_WARPS, true, Blk, true, true>,
                    out, scratch, ids, w, (const Blk*)down_exps, latent, ffn, top_k,
                    expert_begin, n_local, k3_moe_weps_host());
                else
                k3_pdl_launch((unsigned)latent, DOWN_WARPS * 32, dshm, stream, moe_down_combine_kernel<DOWN_WARPS, true, Blk, true>,
                    out, scratch, ids, w, (const Blk*)down_exps, latent, ffn, top_k,
                    expert_begin, n_local, k3_moe_weps_host());
        } else {
        k3_pdl_launch(g1, GATE_WARPS * 32, 0, stream, moe_gate_up_situ_kernel<GATE_WARPS, true, Blk>,
            scratch, x, ids, (const Blk*)gate_exps, (const Blk*)up_exps, latent, ffn,
            situ_beta, 1.0f / situ_beta, situ_linear_beta, inv_lb, lb_active,
            expert_begin, n_local, w, k3_moe_weps_host());
        k3_pdl_launch((unsigned)latent, DOWN_WARPS * 32, dshm, stream, moe_down_combine_kernel<DOWN_WARPS, true, Blk>,
                out, scratch, ids, w, (const Blk*)down_exps, latent, ffn, top_k,
                expert_begin, n_local, k3_moe_weps_host());
        }
    } else {
        k3_pdl_launch(g1, GATE_WARPS * 32, 0, stream, moe_gate_up_situ_kernel<GATE_WARPS, false, Blk>,
            scratch, x, ids, (const Blk*)gate_exps, (const Blk*)up_exps, latent, ffn,
            situ_beta, 1.0f / situ_beta, situ_linear_beta, inv_lb, lb_active,
            expert_begin, n_local, w, k3_moe_weps_host());
        k3_pdl_launch((unsigned)latent, DOWN_WARPS * 32, dshm, stream, moe_down_combine_kernel<DOWN_WARPS, false, Blk>,
                out, scratch, ids, w, (const Blk*)down_exps, latent, ffn, top_k,
                expert_begin, n_local, k3_moe_weps_host());
    }
}

template <typename Blk>
static void moe_expert_ffn_launch(float* out, float* scratch,
                                  const float* x, const int* ids, const float* w,
                                  const void* gate_exps, const void* up_exps,
                                  const void* down_exps,
                                  int latent, int ffn, int top_k,
                                  float situ_beta, float situ_linear_beta,
                                  cudaStream_t stream,
                                  int expert_begin, int n_local_experts) {
    static const int down_warps = [] {
        const char* e = std::getenv("SPARKINFER_K3_MOE_DOWN_WARPS");
        return e && std::atoi(e) == 8 ? 8 : 4;
    }();
#define K3_MOE_WCALL(DW) moe_expert_ffn_launch_w<DW, Blk>(                         \
        out, scratch, x, ids, w, gate_exps, up_exps, down_exps, latent, ffn,       \
        top_k, situ_beta, situ_linear_beta, stream, expert_begin, n_local_experts)
    if (down_warps == 8) K3_MOE_WCALL(8);
    else K3_MOE_WCALL(4);
#undef K3_MOE_WCALL
}

void moe_expert_ffn_iq2xs_f32(float* out, float* scratch,
                              const float* x, const int* ids, const float* w,
                              const void* gate_exps, const void* up_exps,
                              const void* down_exps,
                              int latent, int ffn, int top_k,
                              float situ_beta, float situ_linear_beta,
                              cudaStream_t stream,
                              int expert_begin, int n_local_experts) {
    if (latent <= 0 || ffn <= 0 || top_k <= 0) return;
    if (latent % 256 || ffn % 256) {
        fprintf(stderr, "[k3] moe_expert_ffn: latent=%d ffn=%d must be multiples of 256\n",
                latent, ffn);
        return;
    }
    ensure_iq2xs_tables();
    moe_expert_ffn_launch<BlockIQ2XS>(out, scratch, x, ids, w, gate_exps, up_exps,
                                      down_exps, latent, ffn, top_k, situ_beta,
                                      situ_linear_beta, stream,
                                      expert_begin, n_local_experts);
}

void dequant_iq1_s_f32(float* out, const void* src, int64_t n, cudaStream_t stream) {
    ensure_iq1s_tables();
    const int64_t n_groups = n / 8;
    const int T = 256;
    const int64_t blocks = (n_groups + T - 1) / T;
    k3_pdl_launch((unsigned)blocks, T, 0, stream, dequant_iq1_s_kernel, 
        out, (const BlockIQ1S*)src, n_groups);
}

void moe_expert_ffn_iq1s_f32(float* out, float* scratch,
                             const float* x, const int* ids, const float* w,
                             const void* gate_exps, const void* up_exps,
                             const void* down_exps,
                             int latent, int ffn, int top_k,
                             float situ_beta, float situ_linear_beta,
                             cudaStream_t stream,
                             int expert_begin, int n_local_experts) {
    ensure_iq1s_tables();
    if (latent <= 0 || ffn <= 0 || top_k <= 0) return;
    if (latent % 256 || ffn % 256) {
        fprintf(stderr, "[k3] moe_expert_ffn: latent=%d ffn=%d must be multiples of 256\n",
                latent, ffn);
        return;
    }
    moe_expert_ffn_launch<BlockIQ1S>(out, scratch, x, ids, w, gate_exps, up_exps,
                                     down_exps, latent, ffn, top_k, situ_beta,
                                     situ_linear_beta, stream,
                                     expert_begin, n_local_experts);
}

// Type-dispatched front doors. An unsloth dynamic quant MIXES types per tensor, so
// the expert type is a property of the FILE, not of the build — dispatching at the
// call site rather than at compile time is what keeps one binary able to load both
// UD-IQ1_S and UD-Q2_K_XL. Returns false for a type with no kernel, so the caller
// fails loudly instead of running a wrong decoder over right-sized bytes.
bool dequant_f32_by_type(float* out, const void* src, int64_t n, int ggml_type,
                         cudaStream_t stream) {
    switch (ggml_type) {
        case 0: {  // F32 passthrough — used for a single row of token_embd.weight
            if (n <= 0) return false;
            const int T = 256;
            const int64_t blocks = (n + T - 1) / T;
            k3_pdl_launch((unsigned)blocks, T, 0, stream, dequant_f32_passthrough_kernel, 
                out, (const float*)src, n);
            return true;
        }
        case 8: {  // Q8_0 — the other type token_embd.weight / output.weight may use
            if (n <= 0 || n % 32 != 0) return false;
            const int64_t n_blocks = n / 32;
            const int T = 256;
            const int64_t blocks = (n_blocks + T - 1) / T;
            k3_pdl_launch((unsigned)blocks, T, 0, stream, dequant_q8_0_kernel, 
                out, (const BlockQ8_0*)src, n_blocks);
            return true;
        }
        case 17: dequant_iq2_xs_f32(out, src, n, stream); return true;   // IQ2_XS
        case 19: dequant_iq1_s_f32(out, src, n, stream);  return true;   // IQ1_S
        default: return false;
    }
}

bool moe_expert_ffn_f32_by_type(float* out, float* scratch,
                                const float* x, const int* ids, const float* w,
                                const void* gate_exps, const void* up_exps,
                                const void* down_exps,
                                int latent, int ffn, int top_k,
                                float situ_beta, float situ_linear_beta,
                                int ggml_type, cudaStream_t stream,
                                int expert_begin, int n_local_experts,
                                void* q8k_scratch) {
    if (q8k_scratch) {
        if (latent <= 0 || ffn <= 0 || top_k <= 0 ||
            latent % 256 != 0 || ffn % 256 != 0)
            return false;
        BlockQ8K* qin = (BlockQ8K*)q8k_scratch;
        BlockQ8K* qacts = qin + latent / 256;
        const int T = 128;
        k3_pdl_launch(((latent / 256) + T - 1) / T, T, 0, stream, quantize_q8_k_kernel, 
            qin, x, latent / 256, 1);
        const int lb_active = situ_linear_beta > 0.0f ? 1 : 0;
        const int n_local = n_local_experts > 0 ? n_local_experts : INT_MAX;
        dim3 g1((unsigned)ffn, (unsigned)top_k);
        switch (ggml_type) {
            case 17:
                ensure_iq2xs_tables();
                k3_pdl_launch(g1, 32, 0, stream, moe_gate_up_situ_q8k_kernel<BlockIQ2XS>, 
                    scratch, qin, ids, (const BlockIQ2XS*)gate_exps,
                    (const BlockIQ2XS*)up_exps, latent, ffn,
                    situ_beta, 1.0f / situ_beta, situ_linear_beta,
                    lb_active ? 1.0f / situ_linear_beta : 1.0f, lb_active,
                    expert_begin, n_local);
                break;
            case 19:
                ensure_iq1s_tables();
                k3_pdl_launch(g1, 32, 0, stream, moe_gate_up_situ_q8k_kernel<BlockIQ1S>, 
                    scratch, qin, ids, (const BlockIQ1S*)gate_exps,
                    (const BlockIQ1S*)up_exps, latent, ffn,
                    situ_beta, 1.0f / situ_beta, situ_linear_beta,
                    lb_active ? 1.0f / situ_linear_beta : 1.0f, lb_active,
                    expert_begin, n_local);
                break;
            default:
                return false;
        }
        const int act_blocks = top_k * (ffn / 256);
        k3_pdl_launch((act_blocks + T - 1) / T, T, 0, stream, quantize_q8_k_kernel, 
            qacts, scratch, ffn / 256, top_k);
        if (ggml_type == 17) {
            k3_pdl_launch((unsigned)latent, 32, 0, stream, moe_down_combine_q8k_kernel<BlockIQ2XS>, 
                out, qacts, ids, w, (const BlockIQ2XS*)down_exps,
                latent, ffn, top_k, expert_begin, n_local);
        } else {
            k3_pdl_launch((unsigned)latent, 32, 0, stream, moe_down_combine_q8k_kernel<BlockIQ1S>, 
                out, qacts, ids, w, (const BlockIQ1S*)down_exps,
                latent, ffn, top_k, expert_begin, n_local);
        }
        return true;
    }
    switch (ggml_type) {
        case 17:
            moe_expert_ffn_iq2xs_f32(out, scratch, x, ids, w, gate_exps, up_exps,
                                     down_exps, latent, ffn, top_k, situ_beta,
                                     situ_linear_beta, stream,
                                     expert_begin, n_local_experts);
            return true;
        case 19:
            moe_expert_ffn_iq1s_f32(out, scratch, x, ids, w, gate_exps, up_exps,
                                    down_exps, latent, ffn, top_k, situ_beta,
                                    situ_linear_beta, stream,
                                    expert_begin, n_local_experts);
            return true;
        default:
            return false;
    }
}

void mla_gate_out_f32(float* out, const float* attn_out, const float* gate_proj,
                      int64_t n, cudaStream_t stream) {
    if (n <= 0) return;
    const int T = 256;
    const int64_t blocks = (n + T - 1) / T;
    k3_pdl_launch((unsigned)blocks, T, 0, stream, mla_gate_out_kernel, out, attn_out, gate_proj, n);
}

void kda_decay_gate_f32(float* out, const float* g_raw, const float* A,
                        int head_dim, int n_head, float lower_bound,
                        cudaStream_t stream) {
    if (head_dim <= 0 || n_head <= 0) return;
    k3_pdl_launch((unsigned)n_head, head_dim, 0, stream, kda_decay_gate_kernel, 
        out, g_raw, A, head_dim, lower_bound);
}

void l2_norm_heads_f32(float* out, const float* x, int head_dim, int n_head,
                       float scale, float eps, cudaStream_t stream) {
    if (head_dim <= 0 || n_head <= 0) return;
    k3_pdl_launch((unsigned)n_head, 128, 0, stream, l2_norm_heads_kernel<128>, 
        out, x, head_dim, scale, eps);
}

void mla_absorb_q_f32(float* out, const float* q_nope, const float* q_pe,
                      const float* wk_b, int qk_nope, int kv_lora, int rope_dim,
                      int n_head, cudaStream_t stream) {
    if (n_head <= 0 || kv_lora <= 0 || qk_nope <= 0) return;
    dim3 grid((unsigned)kv_lora, (unsigned)n_head);
    k3_pdl_launch(grid, 128, 0, stream, mla_absorb_q_kernel<128>, 
        out, q_nope, q_pe, wk_b, qk_nope, kv_lora, rope_dim);
}

// THE ONE DERIVATION OF `splits`. The launcher below calls this, and so does the driver's
// graph-invalidation check — deliberately the same function, not the same formula written
// twice. A capture that believes the plan is unchanged while the launcher picks a different
// grid does not fail loudly; it launches the wrong shape.
int k3_mla_decode_plan(int n_head, int kv_lora, int n_ctx) {
    if (n_head <= 0 || kv_lora <= 0 || n_ctx <= 0) return 1;
    int dev = 0;
    const int pin = k3_mla_split_pin_at(n_ctx);
    if (pin > 0) {
        if (pin == 1 || !k3_mla_split_scratch(n_head, kv_lora, &dev)) return 1;
        return std::min(pin, k3_mla_max_splits(n_head));
    }
    if (n_ctx < kMlaSplitMinCtx) return 1;
    if (!k3_mla_split_scratch(n_head, kv_lora, &dev)) return 1;
    return std::min(k3_mla_max_splits(n_head),
                    (n_ctx + kMlaSplitMinCtx - 1) / kMlaSplitMinCtx);
}

void k3_mla_set_split_pin(int splits) {
    g_k3_mla_split_pin = splits > 0 ? splits : 0;
}

int k3_mla_get_split_pin() { return g_k3_mla_split_pin; }

int k3_mla_split_suggest(int n_head, int key_length, int kv_lora, int n_ctx) {
    if (n_head <= 0 || key_length <= 0 || kv_lora <= 0 || n_ctx <= 0) return 1;
    constexpr int BLOCK = 256;
    int dev = 0;
    int splits = (n_ctx >= kMlaSplitMinCtx &&
                  k3_mla_split_scratch(n_head, kv_lora, &dev))
        ? std::min(k3_mla_max_splits(n_head),
                   (n_ctx + kMlaSplitMinCtx - 1) / kMlaSplitMinCtx)
        : 1;
    if (splits <= 1) return 1;

    const int rslots = (kv_lora + BLOCK - 1) / BLOCK;
    const int hpb = k3_mla_heads_per_block(n_head, key_length);
    if (hpb > 1 && rslots <= kMlaMaxRSlots) {
        const int groups = n_head / hpb;
        const int sm = k3_sm_count(dev);
        if (sm > 0) {
            const int fill = (kMlaBlocksPerSm * sm + groups - 1) / groups;
            int min_slice = k3_mla_min_slice_len(hpb, key_length, kv_lora);
            int by_len = std::max(1, n_ctx / min_slice);
            if (k3_mla_reach_fill()) {
                int ms = min_slice;
                while (ms > kMlaCtxTile && n_ctx / ms < fill) ms -= kMlaCtxTile;
                if (k3_mla_partial_fill() || n_ctx / ms >= fill)
                    by_len = std::max(1, n_ctx / ms);
            }
            splits = std::max(splits, std::min(fill, by_len));
            splits = std::min(std::max(splits, 1), k3_mla_max_splits(n_head));
        }
    }
    return splits;
}

// Allocate the split scratch BEFORE any capture begins.
//
// k3_mla_split_scratch() cudaMallocs when capacity is short, and an allocation inside a
// stream capture does not fail the allocation — it fails the CAPTURE, from a call site
// nowhere near the graph code. Pre-warming at the largest shape the model will use means
// the in-capture call hits its `need <= cap` early return and allocates nothing.
bool k3_mla_prewarm_split_scratch(int n_head, int kv_lora) {
    int dev = 0;
    return k3_mla_split_scratch(n_head, kv_lora, &dev);
}

void k3_mla_kv_store_f32(float* cache, const float* kv_cmpr_normed,
                         const float* kv_a_out, const int* d_pos,
                         int kv_lora, int rope_dim, int key_length,
                         cudaStream_t stream) {
    if (!cache || !d_pos || kv_lora <= 0 || rope_dim <= 0 || key_length <= 0) return;
    const int n = kv_lora + rope_dim;
    const int T = 256;
    k3_pdl_launch((unsigned)((n + T - 1) / T), T, 0, stream, mla_kv_store_kernel, 
        cache, kv_cmpr_normed, kv_a_out, d_pos, kv_lora, rope_dim, key_length);
}

void k3_bump_pos(int* d_pos, cudaStream_t stream) {
    if (!d_pos) return;
    k3_pdl_launch(1, 1, 0, stream, bump_pos_kernel, d_pos);
}

void k3_add_pos(int* d_pos, int delta, cudaStream_t stream) {
    if (!d_pos || delta == 0) return;
    k3_pdl_launch(1, 1, 0, stream, add_pos_kernel, d_pos, delta);
}

// TWO LENGTHS, AND THE SPLIT BETWEEN THEM IS THE WHOLE POINT.
//
//   n_ctx    — the HOST's view. Used only to derive `splits`, which sizes the grid and
//              picks which kernel runs. A captured graph cannot change either, so this
//              one is allowed to be a plain int: when it moves the plan, the driver
//              re-captures (see k3_mla_decode_plan).
//   d_pos    — the DEVICE's view, holding the ROW INDEX. The kernels derive the length
//              from it as *d_pos + 1, the same +1 the host applies. One device value
//              with one meaning: the KV store indexes a row with it, the attention
//              lengths a loop with it. Read from
//              memory on every launch, so a replayed graph attends over the live
//              position instead of the one frozen at capture time.
//
// They must agree. k3_mla_decode_plan() is the single derivation both the driver's
// invalidation check and this launcher call — a probe that can drift from what actually
// launches fails by launching the wrong grid, which is worse than not having one.
//
// One launcher for both cache layouts: KV is deduced by every kernel from the
// k_cache argument, so this dispatch is written once and the f32/f16 entry
// points at the bottom are one-line wrappers.
template <typename KV>
static void mla_decode_attn_launch(float* out, const float* q, const KV* k_cache,
                                   const float* wv_b, int key_length, int kv_lora,
                                   int v_dim, int n_head, int n_ctx, const int* d_pos,
                                   float scale, cudaStream_t stream) {
    if (n_head <= 0 || n_ctx <= 0 || key_length <= 0 || kv_lora <= 0 || v_dim <= 0) return;
    if (!d_pos) return;
    constexpr int BLOCK = 256;
    // O(key_length + kv_lora + tile) and NOT O(n_ctx) — 4.8 KB at K3's real dims,
    // the same at 128 tokens of context and at 1M. See the kernel's header note on
    // what the n_ctx-proportional version did past 11,767 tokens.
    const size_t shm = ((size_t)key_length + (size_t)kv_lora + kMlaCtxTile +
                        BLOCK / 32 + 1) * sizeof(float);

    // SPLIT THE CONTEXT when one block per head cannot fill the machine.
    //
    // n_head is 96 and K3 runs on 132-SM H200s, so the un-split grid leaves ~27% of the
    // device idle before occupancy is even considered — and each block then walks n_ctx
    // serially. Splitting trades one extra kernel and a scratch buffer for a grid of
    // n_head * splits.
    //
    // Only above kMlaSplitMinCtx: below it the slice is short enough that the combine
    // pass and the extra global round trip cost more than the parallelism buys, and the
    // un-split kernel is also the one the numeric test pins bit-for-bit.
    int dev = 0;
    const int split_pin = k3_mla_split_pin_at(n_ctx);
    int splits = split_pin > 0
               ? ((split_pin > 1 && k3_mla_split_scratch(n_head, kv_lora, &dev))
                    ? std::min(split_pin, k3_mla_max_splits(n_head)) : 1)
               : (n_ctx >= kMlaSplitMinCtx && k3_mla_split_scratch(n_head, kv_lora, &dev))
               ? std::min(k3_mla_max_splits(n_head), (n_ctx + kMlaSplitMinCtx - 1) / kMlaSplitMinCtx)
               : 1;
    if (splits <= 1) {
        k3_pdl_launch((unsigned)n_head, BLOCK, shm, stream, mla_decode_attn_kernel<BLOCK, KV>, 
            out, q, k_cache, wv_b, key_length, kv_lora, v_dim, d_pos, scale);
        return;
    }

    // Batch heads per block where the shape allows it, so each cached row is loaded
    // once for HPB heads instead of once per head. RSLOTS must cover kv_lora from a
    // single BLOCK-strided pass, and both it and HPB are template parameters because
    // the accumulators only stay in registers when every index is a compile-time one.
    const int rslots  = (kv_lora + BLOCK - 1) / BLOCK;
    const int hpb     = k3_mla_heads_per_block(n_head, key_length);
    const size_t hshm = ((size_t)hpb * key_length + (size_t)hpb * kMlaCtxTile +
                         3 * (size_t)hpb) * sizeof(float);
    if (hpb > 1 && rslots <= kMlaMaxRSlots) {
        // FILL THE MACHINE; SHORTEN SLICES ONLY WHEN THE CONTEXT IS TOO SHORT TO.
        //
        // Dividing the grid by HPB also divides the block count, so the slice count
        // has to give back what the head axis lost. `fill` is the whole objective —
        // kMlaBlocksPerSm resident blocks on every SM, one wave — and it is the term
        // that already tracks the shard policy, because groups and the device's SM
        // count are both in it.
        //
        // The slice floor is a CONSTRAINT ON THAT OBJECTIVE, not a competing target:
        // it caps how finely n_ctx can be cut before each slice stops amortising the
        // partial it writes. It therefore binds only when n_ctx < fill * min_slice,
        // i.e. when there is genuinely not enough context to feed the machine at a
        // workable slice length — at which point no split count can fill it and
        // taking the largest workable one is the best available. Writing the two in
        // this order is the point: the previous version read as a flat token
        // constant beside the target, and when head-sharding moved `fill` from 33 to
        // 264 the constant quietly became the binding term and cost 4%.
        const int groups = n_head / hpb;
        const int sm     = k3_sm_count(dev);
        if (sm > 0 && split_pin <= 0) {
            const int fill = (kMlaBlocksPerSm * sm + groups - 1) / groups;
            int min_slice  = k3_mla_min_slice_len(hpb, key_length, kv_lora);
            int by_len     = std::max(1, n_ctx / min_slice);

            // MISSING THE FILL TARGET BY 3% COSTS 4%, BECAUSE THE SHORTFALL IS A WHOLE
            // PARTIAL WAVE.
            //
            // `fill` is kMlaBlocksPerSm blocks on every SM and the grid is a single
            // wave, so block counts BETWEEN wave boundaries do not degrade gracefully:
            // the remainder blocks land one-per-SM and those SMs idle for the second
            // half of the kernel while the doubled-up ones finish. At the scored shape
            // the floor divides 131,039 into 255 slices against a target of 264, so
            // NINE of 132 SMs run one block instead of two. Measured on one H200 at
            // ctx 131,039, kernel alone, best of 3 interleaved passes:
            //
            //   splits  blk/SM   ms      vs 255
            //     132    1.00    0.311   0.718x
            //     198    1.50    0.268   0.832x
            //     255    1.93    0.223   1.000x   <- shipped
            //     264    2.00    0.217   1.030x   <- the target, reached
            //     330    2.50    0.290   0.769x
            //     396    3.00    0.266   0.840x
            //
            // The sawtooth is the whole story: every half-wave point loses to the whole
            // wave below it. So when the floor lands SHORT of the target, step the slice
            // down by whole tiles until the target is reachable. This can only ever
            // raise `splits` to `fill` and never past it — the std::min below still
            // caps it — so it does not spend the scratch or the combine width that the
            // comment above warns overshooting costs. What it does spend is slice
            // length: 131,039/264 = 496 tokens against 514, a 3.6% shorter slice, which
            // is a 3.6% larger partial-write tax on the 8.6% of traffic that partials
            // are. That is the trade, and it is 0.3% against 3.0%.
            //
            // SPARKINFER_K3_MLA_FILL=0 restores the shipped split count on the SAME
            // binary. Default ON: the harness scores a default build.
            // TAKE THE PARALLELISM YOU CAN REACH, NOT ONLY THE PARALLELISM THAT REACHES
            // THE TARGET.
            //
            // This used to apply the shortened slice ONLY when it got all the way to
            // `fill`, on the reasoning that a context too short to feed the machine is a
            // case the floor already handles. That reasoning holds at decode, where the
            // context is long and the guard passes. It is exactly backwards for PREFILL.
            //
            // The guard clears only when n_ctx / kMlaCtxTile >= fill, i.e. at K3's
            // sharded shape (groups 1, fill 264) when n_ctx >= 33,792. The scored prefill
            // walks 0 -> 32,768 and is therefore ENTIRELY below it: for every token of
            // the scored metric the shortening was computed and then thrown away, and
            // the MLA attention ran on the long slice at a grid of 8..64 blocks against
            // 132 SMs. Reaching only 128 of 264 is not "cannot fill the machine", it is
            // four times the parallelism of not trying.
            //
            // The loop already stops at the first ms that reaches `fill`, so removing the
            // guard cannot overshoot: a context that DOES clear the threshold gets the
            // identical ms it got before, and the 128k decode guard is bit-for-bit
            // untouched. Only the below-threshold case changes, and only upward.
            //
            // What it spends is slice length, and the tax is the partial-write term
            // tax(slice) = 2*hpb*kv_lora/(key_length*slice) -- 4.17% at 512, 16.7% at one
            // tile. That is traffic, and the kernel it is buying parallelism for is
            // LATENCY-bound at these grid sizes, not traffic-bound, which is the trade.
            // SPARKINFER_K3_MLA_FILL_PARTIAL=0 restores the all-or-nothing form on one
            // binary so the two can be measured against each other.
            if (k3_mla_reach_fill()) {
                int ms = min_slice;
                while (ms > kMlaCtxTile && n_ctx / ms < fill) ms -= kMlaCtxTile;
                if (k3_mla_partial_fill() || n_ctx / ms >= fill) {
                    min_slice = ms;
                    by_len    = std::max(1, n_ctx / ms);
                }
            }

            splits = std::max(splits, std::min(fill, by_len));
            splits = std::min(std::max(splits, 1), k3_mla_max_splits(n_head));
        }
        dim3 hgrid((unsigned)groups, (unsigned)splits);
        bool launched = true;

        // THE DEEP-UNROLL INSTANTIATION IS OFFERED FOR ONE SHAPE ONLY.
        //
        // (UT 8, UD 6) sits at exactly 128 registers — the last pair that still fits two
        // blocks per SM — and it was measured at HPB 12 / RSLOTS 2, which is what tp=8
        // head-sharding actually launches at the scored context. The register cost of an
        // unroll depth is a function of RSLOTS and TPW, so carrying the same depth into
        // HPB 8 or RSLOTS 4 would be extrapolating a cliff-edge number onto a shape
        // nothing has compiled, let alone timed. Every other case therefore keeps the
        // shipped (4, 2) and this front door simply declines.
        //
        // SPARKINFER_K3_MLA_UNROLL=0 restores the shipped depth on the SAME binary, so a
        // leave-one-out ablation does not need a second build. Default ON: the harness
        // scores a default build.
        static const bool deep_unroll = [] {
            const char* e = std::getenv("SPARKINFER_K3_MLA_UNROLL");
            return !(e && e[0] == '0');
        }();
        if (deep_unroll && hpb == 12 && rslots == 2) {
            // SPARKINFER_K3_MLA_LSPLIT=1 selects the head-group/__half2 latent pass
            // (see the kernel's LSP note). DEFAULT OFF until measured. Needs a 16-bit
            // cache and kv_lora divisible by BLOCK so the pair-slots tile exactly.
            static const bool lsplit = [] {
                const char* e = std::getenv("SPARKINFER_K3_MLA_LSPLIT");
                return e && e[0] == '1';
            }();
            // SPARKINFER_K3_MLA_PVT=1: t-major s_p (see the kernel's PVT note).
            // DEFAULT OFF until measured. Composes with LSPLIT; measured alone
            // first per the overlap lesson.
            static const bool pvt = [] {
                // MEASURED +0.51 tok/s. Default ON; =0 restores head-major.
                const char* e = std::getenv("SPARKINFER_K3_MLA_PVT");
                return !(e && e[0] == '0');
            }();
            static const bool qt = [] {
                const char* e = std::getenv("SPARKINFER_K3_MLA_QT");
                return e && e[0] == '1';
            }();
            // KLENT=576 folds the per-head s_q bases to immediates. MEASURED
            // +0.20 on the head-major kernel; composed with PVT below because
            // the two touch different axes (constant folding vs s_p layout).
            // =0 restores the runtime-key_length instantiations.
            static const bool klen_on = [] {
                const char* e = std::getenv("SPARKINFER_K3_MLA_KLEN");
                return !(e && e[0] == '0');
            }();
            const bool klen576 = klen_on && key_length == 576;
            if (qt && pvt && lsplit && sizeof(KV) == 2 && (kv_lora % BLOCK) == 0) {
                k3_pdl_launch(hgrid, BLOCK, hshm, stream,
                              mla_decode_attn_hbatch_kernel<BLOCK, 12, 2, KV,
                                                            kMlaTokensPerWarp, 8, 6,
                                                            true, true, 0, true>,
                              g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache,
                              key_length, kv_lora, d_pos, scale, splits);
            } else if (qt && pvt) {
                k3_pdl_launch(hgrid, BLOCK, hshm, stream,
                              mla_decode_attn_hbatch_kernel<BLOCK, 12, 2, KV,
                                                            kMlaTokensPerWarp, 8, 6,
                                                            false, true, 0, true>,
                              g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache,
                              key_length, kv_lora, d_pos, scale, splits);
            } else if (qt) {
                k3_pdl_launch(hgrid, BLOCK, hshm, stream,
                              mla_decode_attn_hbatch_kernel<BLOCK, 12, 2, KV,
                                                            kMlaTokensPerWarp, 8, 6,
                                                            false, false, 0, true>,
                              g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache,
                              key_length, kv_lora, d_pos, scale, splits);
            } else if (pvt && lsplit && sizeof(KV) == 2 && (kv_lora % BLOCK) == 0) {
                k3_pdl_launch(hgrid, BLOCK, hshm, stream,
                              mla_decode_attn_hbatch_kernel<BLOCK, 12, 2, KV,
                                                            kMlaTokensPerWarp, 8, 6, true, true>,
                              g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache,
                              key_length, kv_lora, d_pos, scale, splits);
            } else if (pvt && klen576) {
                // The default path: PVT's t-major s_p with KLENT's folded bases.
                k3_pdl_launch(hgrid, BLOCK, hshm, stream,
                              mla_decode_attn_hbatch_kernel<BLOCK, 12, 2, KV,
                                                            kMlaTokensPerWarp, 8, 6,
                                                            false, true, 576>,
                              g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache,
                              key_length, kv_lora, d_pos, scale, splits);
            } else if (pvt) {
                k3_pdl_launch(hgrid, BLOCK, hshm, stream,
                              mla_decode_attn_hbatch_kernel<BLOCK, 12, 2, KV,
                                                            kMlaTokensPerWarp, 8, 6, false, true>,
                              g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache,
                              key_length, kv_lora, d_pos, scale, splits);
            } else if (lsplit && sizeof(KV) == 2 && (kv_lora % BLOCK) == 0) {
                k3_pdl_launch(hgrid, BLOCK, hshm, stream,
                              mla_decode_attn_hbatch_kernel<BLOCK, 12, 2, KV,
                                                            kMlaTokensPerWarp, 8, 6, true>,
                              g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache,
                              key_length, kv_lora, d_pos, scale, splits);
            } else if ([]{ static const bool on = [] {
                                const char* e = std::getenv("SPARKINFER_K3_MLA_UD7");
                                return e && e[0] == '1'; }();
                            return on; }() && key_length == 576) {
                // The deeper unroll KLEN exists to enable: UD=7 measured 136 regs
                // with a runtime key_length (over the 128 cliff); with KLENT the
                // s_q bases fold to immediates and it may fit. ptxas decides.
                k3_pdl_launch(hgrid, BLOCK, hshm, stream,
                              mla_decode_attn_hbatch_kernel<BLOCK, 12, 2, KV,
                                                            kMlaTokensPerWarp, 8, 7,
                                                            false, false, 576>,
                              g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache,
                              key_length, kv_lora, d_pos, scale, splits);
            } else if (klen576) {
                // KLENT=576 instantiation — same UT/UD, constants folded into
                // immediate offsets. ptxas -v (CI) reports the register delta.
                k3_pdl_launch(hgrid, BLOCK, hshm, stream,
                              mla_decode_attn_hbatch_kernel<BLOCK, 12, 2, KV,
                                                            kMlaTokensPerWarp, 8, 6,
                                                            false, false, 576>,
                              g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache,
                              key_length, kv_lora, d_pos, scale, splits);
            } else {
            k3_pdl_launch(hgrid, BLOCK, hshm, stream,
                          mla_decode_attn_hbatch_kernel<BLOCK, 12, 2, KV, kMlaTokensPerWarp, 8, 6>,
                          g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache, key_length,
                          kv_lora, d_pos, scale, splits);
            }
            k3_mla_launch_combine<BLOCK>(out, dev, wv_b, n_head, kv_lora, v_dim,
                                         splits, stream);
            return;
        }

        switch (hpb * 8 + rslots) {
            case 12 * 8 + 1: k3_pdl_launch(hgrid, BLOCK, hshm, stream, mla_decode_attn_hbatch_kernel<BLOCK, 12, 1, KV>,
                g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache, key_length, kv_lora, d_pos, scale, splits); break;
            case 12 * 8 + 2: k3_pdl_launch(hgrid, BLOCK, hshm, stream, mla_decode_attn_hbatch_kernel<BLOCK, 12, 2, KV>,
                g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache, key_length, kv_lora, d_pos, scale, splits); break;
            case 12 * 8 + 3: k3_pdl_launch(hgrid, BLOCK, hshm, stream, mla_decode_attn_hbatch_kernel<BLOCK, 12, 3, KV>, 
                g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache, key_length, kv_lora, d_pos, scale, splits); break;
            case 12 * 8 + 4: k3_pdl_launch(hgrid, BLOCK, hshm, stream, mla_decode_attn_hbatch_kernel<BLOCK, 12, 4, KV>, 
                g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache, key_length, kv_lora, d_pos, scale, splits); break;
            case 8 * 8 + 1: k3_pdl_launch(hgrid, BLOCK, hshm, stream, mla_decode_attn_hbatch_kernel<BLOCK, 8, 1, KV>, 
                g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache, key_length, kv_lora, d_pos, scale, splits); break;
            case 8 * 8 + 2: k3_pdl_launch(hgrid, BLOCK, hshm, stream, mla_decode_attn_hbatch_kernel<BLOCK, 8, 2, KV>, 
                g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache, key_length, kv_lora, d_pos, scale, splits); break;
            case 8 * 8 + 3: k3_pdl_launch(hgrid, BLOCK, hshm, stream, mla_decode_attn_hbatch_kernel<BLOCK, 8, 3, KV>, 
                g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache, key_length, kv_lora, d_pos, scale, splits); break;
            case 8 * 8 + 4: k3_pdl_launch(hgrid, BLOCK, hshm, stream, mla_decode_attn_hbatch_kernel<BLOCK, 8, 4, KV>, 
                g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache, key_length, kv_lora, d_pos, scale, splits); break;
            case 4 * 8 + 1: k3_pdl_launch(hgrid, BLOCK, hshm, stream, mla_decode_attn_hbatch_kernel<BLOCK, 4, 1, KV>, 
                g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache, key_length, kv_lora, d_pos, scale, splits); break;
            case 4 * 8 + 2: k3_pdl_launch(hgrid, BLOCK, hshm, stream, mla_decode_attn_hbatch_kernel<BLOCK, 4, 2, KV>, 
                g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache, key_length, kv_lora, d_pos, scale, splits); break;
            case 4 * 8 + 3: k3_pdl_launch(hgrid, BLOCK, hshm, stream, mla_decode_attn_hbatch_kernel<BLOCK, 4, 3, KV>, 
                g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache, key_length, kv_lora, d_pos, scale, splits); break;
            case 4 * 8 + 4: k3_pdl_launch(hgrid, BLOCK, hshm, stream, mla_decode_attn_hbatch_kernel<BLOCK, 4, 4, KV>, 
                g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache, key_length, kv_lora, d_pos, scale, splits); break;
            default: launched = false; break;
        }
        if (launched) {
            k3_mla_launch_combine<BLOCK>(out, dev, wv_b, n_head, kv_lora, v_dim,
                                         splits, stream);
            return;
        }
        splits = std::min(splits, k3_mla_max_splits(n_head));
    }

    dim3 grid((unsigned)n_head, (unsigned)splits);
    k3_pdl_launch(grid, BLOCK, shm, stream, mla_decode_attn_split_kernel<BLOCK, KV>, 
        g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache, key_length, kv_lora,
        d_pos, scale, splits);
    k3_mla_launch_combine<BLOCK>(out, dev, wv_b, n_head, kv_lora, v_dim, splits, stream);
}

void mla_decode_attn_f32(float* out, const float* q, const float* k_cache,
                         const float* wv_b, int key_length, int kv_lora,
                         int v_dim, int n_head, int n_ctx, const int* d_pos,
                         float scale, cudaStream_t stream) {
    mla_decode_attn_launch<float>(out, q, k_cache, wv_b, key_length, kv_lora,
                                  v_dim, n_head, n_ctx, d_pos, scale, stream);
}

void mla_decode_attn_kvf16(float* out, const float* q, const void* k_cache,
                           const float* wv_b, int key_length, int kv_lora,
                           int v_dim, int n_head, int n_ctx, const int* d_pos,
                           float scale, cudaStream_t stream) {
    mla_decode_attn_launch<__half>(out, q, (const __half*)k_cache, wv_b,
                                   key_length, kv_lora, v_dim, n_head, n_ctx,
                                   d_pos, scale, stream);
}

// F16 twin of mla_kv_store_kernel: same device-held row index, same concat
// layout, narrowing with round-to-nearest as ggml does for its own F16 type_k.
// The f32 path cannot simply be reused because a copy cannot narrow.
__global__ void mla_kv_store_f16_kernel(__half* __restrict__ cache,
                                        const float* __restrict__ cmpr,
                                        const float* __restrict__ kv_a_out,
                                        const int* __restrict__ d_pos,
                                        int kv_lora, int rope_dim, int key_length) {
    // Per the k3_pdl.cuh contract: fence before reading inputs. The f32 twin has
    // this; this SHIPPED twin (#86 made the cache __half) was the one decode-path
    // kernel that missed both the sync and the k3_pdl_launch routing.
    k3_pdl_sync();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kv_lora + rope_dim) return;
    __half* row = cache + (size_t)(*d_pos) * key_length;
    row[i] = __float2half_rn(i < kv_lora ? cmpr[i] : kv_a_out[i]);
}

void k3_mla_kv_store_f16(void* cache, const float* kv_cmpr_normed,
                         const float* kv_a_out, const int* d_pos,
                         int kv_lora, int rope_dim, int key_length,
                         cudaStream_t stream) {
    if (!cache || !d_pos || kv_lora <= 0 || rope_dim <= 0 || key_length <= 0) return;
    const int n = kv_lora + rope_dim;
    const int T = 256;
    k3_pdl_launch((unsigned)((n + T - 1) / T), T, 0, stream, mla_kv_store_f16_kernel,
        (__half*)cache, kv_cmpr_normed, kv_a_out, d_pos, kv_lora, rope_dim, key_length);
}

// Threads that would run ZERO iterations of the block loop are not free.
//
// Both projection kernels stride `for (b = threadIdx.x; b < blocks_per_row; b += BLOCK)`,
// so a launch with BLOCK=128 against blocks_per_row < 128 leaves most of the block doing
// nothing but taking up an SM slot and a slot in block_sum. K3 has projections where that
// is the common case, not the corner case:
//
//   ssm_f_b        K=128  (kda_head_dim)   -> blocks_per_row 4    124/128 idle  96.9%
//   attn_q_b       K=1536 (q_lora_rank)    -> blocks_per_row 48    80/128 idle  62.5%
//   ffn_routed_up  K=3584 (expert_latent)  -> blocks_per_row 112   16/128 idle  12.5%
//
// ssm_f_b runs on all 69 KDA layers and attn_q_b on all 24 MLA layers, every token. This
// is the same defect PR #7 fixed in the MoE dispatch ("96 of every 128 threads executed
// zero iterations and returned 0.0f"), which survived there for the same reason it
// survives here: the idle threads contribute exact zeros, so the output is correct and
// only the occupancy is wrong.
//
// BIT-IDENTICAL, unconditionally. Two things have to hold and both do:
//
//   1. Warp 0's shuffle tree is untouched — the same 32 lanes hold the same values in
//      either launch, so it reduces to the same bits.
//   2. The cross-warp sum only drops exact +0.0f terms, and x + 0.0f == x.
//
// The one case that would break (2) is x = -0.0f, where -0.0f + 0.0f is +0.0f. It cannot
// occur: every accumulator here is initialised to +0.0f, and +0.0f + anything is never
// -0.0f, so no partial sum can carry a negative zero into the reduction in the first
// place. Checked over 160,000 random rows at every blocks_per_row K3 produces, including
// the 32/64/65 dispatch boundaries.
static inline int proj_block_for(int blocks_per_row) {
    if (blocks_per_row <= 32) return 32;
    if (blocks_per_row <= 64) return 64;
    return 128;
}

bool k3_proj_f32(float* y, const float* x, const void* W, int wtype,
                 int N, int K, cudaStream_t stream) {
    if (N <= 0 || K <= 0) return false;
    constexpr int BLOCK = 128;
    switch (wtype) {
        case 0: {  // F32, dense
            // THE GRID IS N, AND K3'S F32 PROJECTIONS HAVE A TINY N.
            //
            // One block per output row means the banded ssm_beta projection launches
            // TWELVE blocks of 128 threads — 12 of 132 SMs hold anything at all, 161
            // times per token per rank, at 7.1 us against a ~1.3 us launch floor. The
            // grid cannot grow without splitting K and paying a second launch to reduce
            // it (measured elsewhere: a dependent second launch costs ~2.7 us, more than
            // the latency it would hide). Threads per block is the one axis that raises
            // loads in flight without touching either.
            //
            // Each thread strides `k += BLOCK` over K, so BLOCK threads keep BLOCK loads
            // in flight; at K=7168 and 128 threads a block issues 56 rounds of one load.
            // Widening to 1024 gives 8x the requests outstanding for the same bytes, the
            // same grid and the same arithmetic order per thread — the reduction tree
            // grows by three shuffle levels and nothing else.
            //
            // Only when the grid is too small to fill the device: above that, blocks
            // hide each other's latency and the extra threads just shrink the per-thread
            // loop. SPARKINFER_K3_PROJF32_WIDE=0 restores <<<N, 128>>> on one binary.
            static const bool wide = [] {
                const char* e = std::getenv("SPARKINFER_K3_PROJF32_WIDE");
                return !(e && e[0] == '0');
            }();
            const int blk = (!wide || N > 64 || K < 1024) ? 128
                          : (K >= 4096 ? 1024 : 512);
            switch (blk) {
                case 1024:
                    k3_pdl_launch((unsigned)N, 1024, 0, stream, proj_f32_kernel<1024>,
                        y, x, (const float*)W, K);
                    return true;
                case 512:
                    k3_pdl_launch((unsigned)N, 512, 0, stream, proj_f32_kernel<512>,
                        y, x, (const float*)W, K);
                    return true;
                default:
                    k3_pdl_launch((unsigned)N, BLOCK, 0, stream, proj_f32_kernel<BLOCK>,
                        y, x, (const float*)W, K);
                    return true;
            }
        }
        case 8: {  // Q8_0
            if (K % 32 != 0) return false;
            // Multi-row amortises the activation re-read, but it also divides the grid
            // by ROWS. That is only free while the grid still covers the device several
            // times over — K3's big projections are N=12288 (grid 3072), whereas the
            // small ones (n_head=96 for ssm_beta) would drop to 24 blocks and lose more
            // to idle SMs than the activation traffic was costing. Single-row below the
            // threshold keeps those on the wide grid.
            // ROWS is how much work a block has to hide memory latency behind, and
            // K3's shapes left it far too low.
            //
            // At the MoE-adjacent widths — routed_down/routed_up and the three shared
            // expert projections — K is 3584 or 7168, so blocks_per_row is 112 or 224
            // and the `b += BLOCK` loop runs ONCE or TWICE. A block therefore issues
            // one or two rounds of loads, then stops to reduce. There is nothing left
            // in flight to cover the next miss.
            //
            // Measured on 8x H200 at tp=8 (nsys, main 86bb264, per token per rank):
            // this kernel moves ~19 GB in 37.1 ms = ~512 GB/s of the card's 4.8 TB/s,
            // while proj_q8_0_fused4 — the SAME access pattern over the same 34-byte
            // BlockQ8_0 — reaches 1.33 TB/s. The difference is not the pattern, it is
            // that fused4 reads 8 blocks per iteration (4 tensors x ROWS=2) where this
            // read 4, so it has 8 independent accumulator chains in flight instead of 4.
            // Widening ROWS buys the same thing here.
            //
            // BIT-IDENTICAL: row r's accumulator still sums blocks b = threadIdx.x,
            // +BLOCK, ... in that order, and still folds over the same BLOCK threads.
            // ROWS only changes how many rows a block happens to carry, which no
            // accumulation order depends on.
            //
            // THREE tiers, because widening also divides the grid, and each tier is
            // taken only while the grid still covers the device several times over.
            // Two tiers is not enough at K3's shapes: a single 16-row tier gated at
            // N >= 4096 would push routed_down (N = 3584) and the shared-expert
            // projections (N = 3072) all the way back down to 4 rows, which is a
            // regression on 276 of the ~600 calls a token.
            constexpr int ROWS_W16 = 16;
            constexpr int ROWS_W8 = 8;
            constexpr int ROWS = 4;
            constexpr int MIN_BLOCKS = 256;
            const int nb = K / 32;
            const int TB = proj_block_for(nb);
            if (N >= ROWS_W16 * MIN_BLOCKS) {
                // SPARKINFER_K3_HEAD_1BAR=0 restores the block_sum epilogue on the
                // same binary. At the scored tp=8 config the only caller that still
                // reaches this branch is the banded LM head; the fold is the tier
                // family's proven one-barrier epilogue, bit-identical by order.
                static const bool head_1bar = [] {
                    const char* e = std::getenv("SPARKINFER_K3_HEAD_1BAR");
                    return !(e && e[0] == '0');
                }();
                const unsigned grid = (unsigned)((N + ROWS_W16 - 1) / ROWS_W16);
                const bool al16 = ((uintptr_t)x & 15u) == 0;
                switch (TB) {
                    case 32:
                        if (head_1bar && al16)
                            k3_pdl_launch(grid, 32, 0, stream, proj_q8_0_multirow_1bf_kernel<32, ROWS_W16>,
                                y, x, (const BlockQ8_0*)W, nb, N);
                        else
                            k3_pdl_launch(grid, 32, 0, stream, proj_q8_0_multirow_kernel<32, ROWS_W16>,
                                y, x, (const BlockQ8_0*)W, nb, N);
                        break;
                    case 64:
                        if (head_1bar && al16)
                            k3_pdl_launch(grid, 64, 0, stream, proj_q8_0_multirow_1bf_kernel<64, ROWS_W16>,
                                y, x, (const BlockQ8_0*)W, nb, N);
                        else
                            k3_pdl_launch(grid, 64, 0, stream, proj_q8_0_multirow_kernel<64, ROWS_W16>,
                                y, x, (const BlockQ8_0*)W, nb, N);
                        break;
                    default:
                        if (head_1bar && al16)
                            k3_pdl_launch(grid, BLOCK, 0, stream, proj_q8_0_multirow_1bf_kernel<BLOCK, ROWS_W16>, y, x, (const BlockQ8_0*)W, nb, N);
                        else
                            k3_pdl_launch(grid, BLOCK, 0, stream, proj_q8_0_multirow_kernel<BLOCK, ROWS_W16>, y, x, (const BlockQ8_0*)W, nb, N);
                        break;
                }
            } else if (N >= ROWS_W8 * MIN_BLOCKS) {
                const unsigned grid = (unsigned)((N + ROWS_W8 - 1) / ROWS_W8);
                switch (TB) {
                    case 32:
                        k3_pdl_launch(grid, 32, 0, stream, proj_q8_0_multirow_kernel<32, ROWS_W8>, 
                            y, x, (const BlockQ8_0*)W, nb, N);
                        break;
                    case 64:
                        k3_pdl_launch(grid, 64, 0, stream, proj_q8_0_multirow_kernel<64, ROWS_W8>, 
                            y, x, (const BlockQ8_0*)W, nb, N);
                        break;
                    default:
                        k3_pdl_launch(grid, BLOCK, 0, stream, proj_q8_0_multirow_kernel<BLOCK, ROWS_W8>, y, x, (const BlockQ8_0*)W, nb, N);
                        break;
                }
            } else if (N >= ROWS * MIN_BLOCKS) {
                const unsigned grid = (unsigned)((N + ROWS - 1) / ROWS);
                switch (TB) {
                    case 32:
                        k3_pdl_launch(grid, 32, 0, stream, proj_q8_0_multirow_kernel<32, ROWS>, 
                            y, x, (const BlockQ8_0*)W, nb, N);
                        break;
                    case 64:
                        k3_pdl_launch(grid, 64, 0, stream, proj_q8_0_multirow_kernel<64, ROWS>, 
                            y, x, (const BlockQ8_0*)W, nb, N);
                        break;
                    default:
                        k3_pdl_launch(grid, BLOCK, 0, stream, proj_q8_0_multirow_kernel<BLOCK, ROWS>, 
                            y, x, (const BlockQ8_0*)W, nb, N);
                        break;
                }
            } else {
                switch (TB) {
                    case 32:
                        k3_pdl_launch((unsigned)N, 32, 0, stream, proj_q8_0_kernel<32>, 
                            y, x, (const BlockQ8_0*)W, nb);
                        break;
                    case 64:
                        k3_pdl_launch((unsigned)N, 64, 0, stream, proj_q8_0_kernel<64>, 
                            y, x, (const BlockQ8_0*)W, nb);
                        break;
                    default:
                        k3_pdl_launch((unsigned)N, BLOCK, 0, stream, proj_q8_0_kernel<BLOCK>, 
                            y, x, (const BlockQ8_0*)W, nb);
                        break;
                }
            }
            return true;
        }
        default:
            return false;
    }
}

bool k3_proj_ggml_f32_x4(float* y0, float* y1, float* y2, float* y3, const float* x,
                         const void* W0, const void* W1, const void* W2, const void* W3,
                         int wtype, int N, int K, void* q8_scratch,
                         cudaStream_t stream, bool x_pre_q8) {
    // Same narrow contract as the f32-activation x4 below: Q8_0 weights, four
    // identical shapes, false means "use the slow path" rather than an error.
    if (N <= 0 || K <= 0 || wtype != 8 || K % 32 != 0) return false;
    if (!y0 || !y1 || !y2 || !y3 || !W0 || !W1 || !W2 || !W3 || !q8_scratch) return false;

    constexpr int ROWS = 4;
    if (N < ROWS) return false;
    const int nb = K / 32;

    // ONE quantisation for all four projections. This is the whole point: the
    // per-call path quantised the identical activation four times per layer per
    // rank, which is 8.7% of GPU time in 59,696 launches. The quantise and the
    // consumer go out back-to-back on one stream, so no caller has to promise
    // anything about the activation's lifetime -- the reuse cannot outlive the
    // launch that produced it.
    //
    // ...unless the CALLER already quantised this exact activation, which it does on
    // every KDA layer: `normed` is hoisted for ssm_f_a and ssm_beta before this runs.
    // Quantising again here made the hoist a net LOSS on those 69 layers -- it added a
    // launch and removed none, because the other two consumers turned out to be
    // f32-weighted and never quantised at all. x is passed only so this can tell.
    constexpr int QT = 128;
    if (!x_pre_q8)
        k3_quantize_q8_0(q8_scratch, x, nb, stream);

    const unsigned grid = (unsigned)((N + ROWS - 1) / ROWS);
    const BlockQ8_0* xq = (const BlockQ8_0*)q8_scratch;
#define K3_QQ4_LAUNCH(BS)                                                        \
    k3_pdl_launch(grid, BS, 0, stream, proj_q8_0_q8_0_fused4_kernel<BS, ROWS>,              \
        y0, y1, y2, y3, xq, (const BlockQ8_0*)W0, (const BlockQ8_0*)W1,          \
        (const BlockQ8_0*)W2, (const BlockQ8_0*)W3, nb, N)
    switch (proj_block_for(nb)) {
        case 32:  K3_QQ4_LAUNCH(32);  break;
        case 64:  K3_QQ4_LAUNCH(64);  break;
        default:  K3_QQ4_LAUNCH(128); break;
    }
#undef K3_QQ4_LAUNCH
    return true;
}

bool k3_proj_f32_x4(float* y0, float* y1, float* y2, float* y3, const float* x,
                    const void* W0, const void* W1, const void* W2, const void* W3,
                    int wtype, int N, int K, cudaStream_t stream) {
    // Deliberately narrow: Q8_0 only, all four the same shape. Anything else returns
    // false so the caller falls back to four ordinary k3_proj_f32 calls rather than this
    // path silently handling a case it was not written for.
    if (N <= 0 || K <= 0 || wtype != 8 || K % 32 != 0) return false;
    if (!y0 || !y1 || !y2 || !y3 || !W0 || !W1 || !W2 || !W3) return false;
    // Four rows per block rather than two, for the same reason multirow widened: the
    // block gets 4 tensors x 4 rows = 16 independent accumulator chains to keep in
    // flight instead of 8, and the grid (12288/4 = 3072 blocks) still covers the
    // device many times over. Bit-identical — a row's accumulation order over b and
    // over i is untouched, only how many rows share a block changes.
    constexpr int ROWS = 4;
    if (N < ROWS) return false;
    const int nb = K / 32;
    const unsigned grid = (unsigned)((N + ROWS - 1) / ROWS);
    switch (proj_block_for(nb)) {
        case 32:
            k3_pdl_launch(grid, 32, 0, stream, proj_q8_0_fused4_kernel<32, ROWS>, 
                y0, y1, y2, y3, x, (const BlockQ8_0*)W0, (const BlockQ8_0*)W1,
                (const BlockQ8_0*)W2, (const BlockQ8_0*)W3, nb, N);
            break;
        case 64:
            k3_pdl_launch(grid, 64, 0, stream, proj_q8_0_fused4_kernel<64, ROWS>, 
                y0, y1, y2, y3, x, (const BlockQ8_0*)W0, (const BlockQ8_0*)W1,
                (const BlockQ8_0*)W2, (const BlockQ8_0*)W3, nb, N);
            break;
        default:
            k3_pdl_launch(grid, 128, 0, stream, proj_q8_0_fused4_kernel<128, ROWS>, 
                y0, y1, y2, y3, x, (const BlockQ8_0*)W0, (const BlockQ8_0*)W1,
                (const BlockQ8_0*)W2, (const BlockQ8_0*)W3, nb, N);
            break;
    }
    return true;
}

// Quantise one activation to Q8_0, on its own, so a caller with several projections
// over the SAME activation can pay for it once.
//
// K3 re-quantises the identical vector three or four times per layer: `normed` feeds the
// fused qkvg group plus ssm_f_a and ssm_beta on every KDA layer, and `normed2` feeds the
// router, routed_down and both shared-expert projections on every MoE layer. Profiled at
// ctx 131072 that is 61,848 launches of quantize_q8_0 -- 7.9% of GPU kernel time at
// 4.19 us each to move ~28 KB, i.e. ~6.7 GB/s on a 4.8 TB/s part. It is not work, it is
// launch overhead, and 462 of those launches per token per rank are recomputing bytes
// that are already sitting in the scratch.
bool k3_quantize_act_f32(void* q8_out, const float* x, int K, cudaStream_t stream) {
    if (!q8_out || !x || K <= 0 || K % 32 != 0) return false;
    const int nb = K / 32;
    constexpr int QT = 128;
    k3_quantize_q8_0(q8_out, x, nb, stream);
    return true;
}

bool k3_quantize_act_rows_f32(void* q8_out, const float* x, int K, int n_rows,
                              int64_t row_stride, cudaStream_t stream) {
    if (!q8_out || !x || K <= 0 || (K % 32) != 0 || n_rows <= 0) return false;
    if (row_stride == 0) row_stride = K;
    // k3_quantize_q8_0 already assigns one independent CTA to each 32-value block.
    // Tight rows are therefore exactly the single-row launch with a larger block count:
    // no reduction or scale crosses a row boundary. Padded rows need a gather-aware
    // kernel and deliberately decline here.
    if (row_stride != K) return false;
    const int64_t blocks = (int64_t)(K / 32) * n_rows;
    if (blocks > INT_MAX) return false;
    k3_quantize_q8_0(q8_out, x, (int)blocks, stream);
    return true;
}

// Projection from an ALREADY-quantised activation. This is k3_proj_ggml_f32's body with
// the quantise lifted out; that function now calls it, so there is one implementation and
// the hoisted and non-hoisted paths cannot drift apart.
//
// Deliberately takes the quantised buffer rather than caching one keyed on `x`. An
// earlier attempt at this optimisation DID cache, guarded by "was the scratch last
// written from this pointer?", and shipped top1 0.0 / mean_kld 0.937 -- fluent, wrong,
// and faster. That guard tracked which pointer the scratch came from, never whether the
// bytes behind it still matched. Passing the buffer makes the reuse window explicit in
// the caller's straight-line code instead of an invariant someone has to maintain.
bool k3_proj_q8act_f32(float* y, const void* q8_scratch, const void* W, int wtype,
                       int N, int K, cudaStream_t stream) {
    // SoA-first (k3_q8soa.cu): load-registered tensors under SPARKINFER_K3_Q8SOA
    // (default ON; =0 restores AoS). Bit-identical to the dispatch below.
    if (k3_proj_q8soa(y, q8_scratch, W, N, K, stream)) return true;

    if (N <= 0 || K <= 0) return false;
    if (wtype != 8 || !q8_scratch || K % 32 != 0) return false;
    const int nb = K / 32;
    // Same idle-thread sizing as the f32-activation path above.
    constexpr int BLOCK = 128;
    const int TB = proj_block_for(nb);

    // ROWS PER BLOCK, mirroring what the f32 path already does. The thresholds are
    // the f32 path's, and for the same reason: multi-row divides the grid by ROWS,
    // so it is only free while the grid still covers the device several times over.
    // K3's big projections are N=12288 (grid 768 at ROWS 16); the small ones
    // (n_head=96 for ssm_beta) would drop to single digits and lose more to idle SMs
    // than the amortised activation saves.
    //
    // The activation being quantised does NOT remove the reason to widen. It is 7.6 KB
    // per block instead of 28 KB, but it is still re-read by every one of the N blocks,
    // which at N=12288, K=7168 is 93 MB against 93.6 MB of weights. Halving the total
    // needs ROWS, not a smaller activation.
    constexpr int ROWS_W16 = 16;
    constexpr int ROWS_W8  = 8;
    constexpr int ROWS_W4  = 4;
    constexpr int MIN_N_W16 = 4096;   // grid >= 256 blocks
    constexpr int MIN_N_W8  = 2048;
    constexpr int MIN_N_W4  = 1024;

#define K3_QQ_LAUNCH(R)                                                              \
    do {                                                                             \
        const unsigned grid = (unsigned)((N + (R) - 1) / (R));                        \
        switch (TB) {                                                                \
            case 32:                                                                 \
                k3_pdl_launch(grid, 32, 0, stream, proj_q8_0_q8_0_multirow_kernel<32, R>,        \
                    y, (const BlockQ8_0*)q8_scratch, (const BlockQ8_0*)W, nb, N);      \
                break;                                                               \
            case 64:                                                                 \
                k3_pdl_launch(grid, 64, 0, stream, proj_q8_0_q8_0_multirow_kernel<64, R>,        \
                    y, (const BlockQ8_0*)q8_scratch, (const BlockQ8_0*)W, nb, N);      \
                break;                                                               \
            default:                                                                 \
                k3_pdl_launch(grid, BLOCK, 0, stream, proj_q8_0_q8_0_multirow_kernel<BLOCK, R>,  \
                    y, (const BlockQ8_0*)q8_scratch, (const BlockQ8_0*)W, nb, N);      \
                break;                                                               \
        }                                                                            \
    } while (0)

    if      (N >= MIN_N_W16) K3_QQ_LAUNCH(ROWS_W16);
    else if (N >= MIN_N_W8)  K3_QQ_LAUNCH(ROWS_W8);
    else if (N >= MIN_N_W4)  K3_QQ_LAUNCH(ROWS_W4);
    else {
        // Below the threshold the single-row kernel keeps the wide grid, and it is
        // also the one the numeric test pins bit-for-bit.
        //
        // SPARKINFER_K3_B2WIDE=1 swaps the Q8_0 block accessor for the 32-bit-load
        // form (see get_int_b2_wide). DEFAULT OFF until measured. This kernel is the
        // largest consumer OUTSIDE k3_proj_q8_fast.cu -- shexp gate/up, ssm_f_a and
        // attn_kv_a_mqa -- and it calls the accessor on BOTH dp4a operands, so it is
        // where the instruction-count change should show up first if it shows at all.
        static const bool b2wide = [] {
            // MEASURED +0.30 tok/s. Default ON; =0 restores the 16-bit pair.
            const char* e = std::getenv("SPARKINFER_K3_B2WIDE");
            return !(e && e[0] == '0');
        }();
#define K3_QQ_W(BLK)                                                                  \
        do {                                                                          \
            if (b2wide)                                                               \
                k3_pdl_launch((unsigned)N, (BLK), 0, stream,                          \
                    proj_q8_0_q8_0_kernel<(BLK), true>,                               \
                    y, (const BlockQ8_0*)q8_scratch, (const BlockQ8_0*)W, nb);        \
            else                                                                      \
                k3_pdl_launch((unsigned)N, (BLK), 0, stream,                          \
                    proj_q8_0_q8_0_kernel<(BLK), false>,                              \
                    y, (const BlockQ8_0*)q8_scratch, (const BlockQ8_0*)W, nb);        \
        } while (0)
        switch (TB) {
            case 32:  K3_QQ_W(32);    break;
            case 64:  K3_QQ_W(64);    break;
            default:  K3_QQ_W(BLOCK); break;
        }
#undef K3_QQ_W
    }
#undef K3_QQ_LAUNCH
    return true;
}

bool k3_proj_ggml_f32(float* y, const float* x, const void* W, int wtype,
                      int N, int K, void* q8_scratch, cudaStream_t stream) {
    if (N <= 0 || K <= 0) return false;
    if (wtype == 0)
        return k3_proj_f32(y, x, W, wtype, N, K, stream);
    if (wtype != 8 || !q8_scratch || K % 32 != 0) return false;
    if (!k3_quantize_act_f32(q8_scratch, x, K, stream)) return false;
    return k3_proj_q8act_f32(y, q8_scratch, W, wtype, N, K, stream);
}

size_t k3_q8_0_bytes(int K) {
    return K > 0 && K % 32 == 0 ? (size_t)(K / 32) * sizeof(BlockQ8_0) : 0;
}

size_t k3_moe_q8_k_bytes(int latent, int ffn, int top_k) {
    if (latent <= 0 || ffn <= 0 || top_k <= 0 ||
        latent % 256 != 0 || ffn % 256 != 0)
        return 0;
    return ((size_t)(latent / 256) + (size_t)top_k * (ffn / 256)) *
           sizeof(BlockQ8K);
}

// Threads for the apply phase. The reduction is always 128 threads (see the kernel);
// this only sizes how many threads walk the elementwise scale. Floored at 128 so the
// narrow norms keep the shape they already had.
static inline int rms_norm_block_for(int units) {
    if (units >= 1024) return 1024;
    if (units >= 512)  return 512;
    if (units >= 256)  return 256;
    return 128;
}

static inline bool rms_norm_aligned16(const void* p) {
    return ((uintptr_t)p & 15u) == 0;
}

void rms_norm_f32(float* out, const float* x, const float* w, int n, float eps,
                  cudaStream_t stream) {
    rms_norm_f32_rows(out, x, w, n, eps, 1, 0, stream);
}

// `rows` rows of `n` elements, `row_stride` apart, in ONE launch. rows == 1 with stride 0
// is the single-row call and takes an identical path, so this is not a second
// implementation to keep in step — it IS the implementation.
//
// Returns false, never a wrong answer, when the geometry would not take the wide-kernel
// path the batching relies on (the narrow fallback below is single-row only). The caller
// then loops, which is what it did before.
bool rms_norm_f32_rows(float* out, const float* x, const float* w, int n, float eps,
                       int rows, long long row_stride, cudaStream_t stream) {
    if (n <= 0 || rows <= 0) return false;

    // At BLOCK==128 with no vec4 the wide kernel is the original kernel (same 128-thread
    // sum, same 128-thread apply), so that path stays on rms_norm_kernel for a one-line
    // reviewer check rather than trusting the specialization. Everything wider shares
    // that reduction and only grows the apply.
    const bool vec4 = (n % 4 == 0 && rms_norm_aligned16(out) && rms_norm_aligned16(x) &&
                       rms_norm_aligned16(w));
    const int units = vec4 ? n / 4 : n;
    const int block = rms_norm_block_for(units);
    const int n4 = vec4 ? n / 4 : 0;

    // On top of the widening, slice the apply across CTAs: the apply is three times
    // the reduction's bytes, and one CTA is one SM of 132. 327 of these launches per
    // token per rank at decode (93 attn_norm + 93 ffn_norm + 92 routed_norm +
    // 24 q_a_norm + 24 kv_a_norm + 1 output_norm). RMSG names the slice count and
    // RMSU the reduction unroll, so the two can be told apart in an A/B; RMSG=0
    // RMSU=0 restores the single-CTA launches exactly.
    //
    // out == x is NOT safe to spread: with several CTAs, CTA 0 can finish its
    // reduction and start overwriting x while another CTA is still reading x to
    // compute the same sum. Two call sites are in-place; the rest spread.
    static const int rmsg = [] {
        const char* e = std::getenv("SPARKINFER_K3_RMSG");
        return e ? atoi(e) : 14;
    }();
    static const bool rmsu = [] {
        const char* e = std::getenv("SPARKINFER_K3_RMSU");
        return !(e && e[0] == '0');
    }();
    int span_units = units;
    unsigned grid = 1u;
    int launch_block = block;
    if (rmsg > 1 && out != x) {
        // Slices stay warp-aligned, not block-aligned: rounding the span up to a
        // 1024-thread block would collapse a 14-way spread of n=7168 to grid 2.
        // Spread CTAs run 128 threads — the reduction is 128 threads regardless,
        // and 14 slices x 128 threads walk the whole float4 apply in one wave.
        span_units = ((units + rmsg - 1) / rmsg + 31) / 32 * 32;
        grid = (unsigned)((units + span_units - 1) / span_units);
        if (grid > 1u) launch_block = 128;
    }

    if (grid == 1u && !rmsu && block == 128 && !vec4) {
        // rms_norm_kernel is a DIFFERENT reduction from the wide one, and it has no token
        // axis. Batching through it would change the arithmetic, so decline and let the
        // caller loop rather than quietly emit different numbers.
        if (rows > 1) return false;
        k3_pdl_launch(1, 128, 0, stream, rms_norm_kernel<128>, out, x, w, n, eps);
        return true;
    }

#define K3_RMSN(BLK)                                                                   \
    do {                                                                               \
        if (rmsu)                                                                      \
            k3_pdl_launch(g2, (BLK), 0, stream, rms_norm_wide_kernel<(BLK), true>,    \
                          out, x, w, n, eps, n4, vec4 ? 1 : 0, span_units, row_stride); \
        else                                                                           \
            k3_pdl_launch(g2, (BLK), 0, stream, rms_norm_wide_kernel<(BLK), false>,   \
                          out, x, w, n, eps, n4, vec4 ? 1 : 0, span_units, row_stride); \
    } while (0)
    const dim3 g2(grid, (unsigned)rows);
    switch (launch_block) {
        case 1024: K3_RMSN(1024); return true;
        case 512:  K3_RMSN(512);  return true;
        case 256:  K3_RMSN(256);  return true;
        default:   K3_RMSN(128);  return true;
    }
#undef K3_RMSN
}

void k3_add_f32(float* out, const float* a, const float* b, int64_t n,
               cudaStream_t stream) {
    if (n <= 0) return;
    const int T = 256;
    const int64_t blocks = (n + T - 1) / T;
    k3_pdl_launch((unsigned)blocks, T, 0, stream, add_f32_kernel, out, a, b, n);
}

void sigmoid_inplace_f32(float* x, int64_t n, cudaStream_t stream) {
    if (n <= 0) return;
    const int T = 256;
    const int64_t blocks = (n + T - 1) / T;
    k3_pdl_launch((unsigned)blocks, T, 0, stream, sigmoid_inplace_f32_kernel, x, n);
}

}  // namespace k3
}  // namespace kernels
}  // namespace sparkinfer
