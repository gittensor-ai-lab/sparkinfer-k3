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


// ---------------------------------------------------------------------------
// 1. situ
// ---------------------------------------------------------------------------

__global__ void situ_kernel(float* __restrict__ out, const float* __restrict__ gate,
                            const float* __restrict__ up, int64_t n,
                            float beta, float inv_beta, float lb, float inv_lb,
                            int lb_active) {
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

template <int BLOCK>
__global__ void kda_decode_step_kernel(float* __restrict__ out, float* __restrict__ state,
                                       const float* __restrict__ q,
                                       const float* __restrict__ k,
                                       const float* __restrict__ v,
                                       const float* __restrict__ g,
                                       const float* __restrict__ beta,
                                       int head_dim) {
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

template <int BLOCK>
__global__ void attn_res_score_kernel(float* __restrict__ scores,
                                      const float* __restrict__ ckpts,
                                      const float* __restrict__ cur,
                                      const float* __restrict__ score_w,
                                      int n_embd, int n_ckpt, float eps) {
    __shared__ float shm[BLOCK / 32 + 1];
    // blockIdx.x in [0, n_ckpt]: the last block scores the current stream, which is
    // the only thing that made pass 1 look sequential.
    const int c = blockIdx.x;
    const float* __restrict__ src = (c < n_ckpt) ? ckpts + (size_t)c * n_embd : cur;

    float ss = 0.0f;
    for (int d = threadIdx.x; d < n_embd; d += BLOCK) ss += src[d] * src[d];
    ss = block_sum<BLOCK>(ss, shm);
    const float inv = rsqrtf(ss / (float)n_embd + eps);

    float dot = 0.0f;
    for (int d = threadIdx.x; d < n_embd; d += BLOCK) dot += (src[d] * inv) * score_w[d];
    dot = block_sum<BLOCK>(dot, shm);
    if (threadIdx.x == 0) scores[c] = dot;
}

template <int BLOCK>
__global__ void attn_res_apply_kernel(float* __restrict__ out,
                                      const float* __restrict__ ckpts,
                                      const float* __restrict__ cur,
                                      const float* __restrict__ scores,
                                      int n_embd, int n_ckpt) {
    extern __shared__ float p[];              // n_ckpt + 1 softmax weights

    if (threadIdx.x == 0) {
        const int n = n_ckpt + 1;
        float mx = scores[0];
        for (int c = 1; c < n; ++c) mx = fmaxf(mx, scores[c]);
        float sum = 0.0f;
        for (int c = 0; c < n; ++c) { p[c] = __expf(scores[c] - mx); sum += p[c]; }
        const float inv = 1.0f / sum;
        for (int c = 0; c < n; ++c) p[c] *= inv;
    }
    __syncthreads();

    const int d = blockIdx.x * BLOCK + threadIdx.x;
    if (d >= n_embd) return;
    float acc = 0.0f;
    for (int c = 0; c < n_ckpt; ++c) acc += p[c] * ckpts[(size_t)c * n_embd + d];
    acc += p[n_ckpt] * cur[d];
    out[d] = acc;
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
template <bool XVEC>
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
template <bool XVEC>
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
        if (XVEC) {
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
template <int WARPS_PER_CTA, bool XVEC, typename Blk>
__global__ void moe_gate_up_situ_kernel(float* __restrict__ scratch,
                                        const float* __restrict__ x,
                                        const int* __restrict__ ids,
                                        const Blk* __restrict__ gate_exps,
                                        const Blk* __restrict__ up_exps,
                                        int latent, int ffn,
                                        float beta, float inv_beta,
                                        float lb, float inv_lb, int lb_active,
                                        int expert_begin, int n_local_experts) {
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
    const int blocks_per_row = latent / 256;

    const Blk* g_row = gate_exps + (size_t)(e * ffn + j) * blocks_per_row;
    const Blk* u_row = up_exps   + (size_t)(e * ffn + j) * blocks_per_row;

    float gacc = 0.0f, uacc = 0.0f;
    for (int b = 0; b < blocks_per_row; ++b) {
        const float* xb = x + b * 256;
        gacc += block_dot<XVEC>(g_row[b], xb, lane, 32);
        uacc += block_dot<XVEC>(u_row[b], xb, lane, 32);
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
template <int WARPS_PER_CTA, bool XVEC, typename Blk>
__global__ void moe_down_combine_kernel(float* __restrict__ out,
                                        const float* __restrict__ scratch,
                                        const int* __restrict__ ids,
                                        const float* __restrict__ w,
                                        const Blk* __restrict__ down_exps,
                                        int latent, int ffn, int top_k,
                                        int expert_begin, int n_local_experts) {
    const int o = blockIdx.x;                 // output element in [0, latent)
    if (o >= latent) return;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int blocks_per_row = ffn / 256;
    extern __shared__ float partial[];        // top_k floats

    for (int k = warp; k < top_k; k += WARPS_PER_CTA) {
        // Same band test as the gate/up pass. Skipping here is what keeps the read
        // in bounds: down_exps holds n_local experts, so indexing it by a GLOBAL id
        // would run off the end of the allocation for every rank but rank 0 — and
        // read whatever the allocator put there rather than faulting.
        const int e = ids[k] - expert_begin;
        if (e < 0 || e >= n_local_experts) continue;
        const Blk* d_row = down_exps + (size_t)(e * latent + o) * blocks_per_row;
        const float* act = scratch + (size_t)k * ffn;
        float acc = 0.0f;
        for (int b = 0; b < blocks_per_row; ++b)
            acc += block_dot<XVEC>(d_row[b], act + b * 256, lane, 32);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        if (lane == 0) partial[k] = acc;      // RAW; w[k] is applied in the fold
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            const int e = ids[k] - expert_begin;
            if (e < 0 || e >= n_local_experts) continue;
            total += w[k] * partial[k];   // one FMA, exactly as the serial version
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

template <int BLOCK>
__global__ void mla_decode_attn_kernel(float* __restrict__ out,
                                       const float* __restrict__ q,
                                       const float* __restrict__ k_cache,
                                       const float* __restrict__ wv_b,
                                       int key_length, int kv_lora, int v_dim,
                                       int n_ctx, float scale) {
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
            const float* kt = k_cache + (size_t)(t0 + t) * key_length;
            float s = 0.0f;
            for (int d = lane; d < key_length; d += 32) s += s_q[d] * kt[d];
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
                a += s_p[t] * k_cache[(size_t)(t0 + t) * key_length + r];
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

// CPU-reference-compatible activation conversion. One CUDA thread deliberately
// owns one quant block: these vectors are tiny (at most 33,792 values in K3), and
// the serial max scan preserves ggml's first-maximum sign rule exactly. Parallel
// reduction here would make equal-magnitude +/- values pick a schedule-dependent
// sign, changing every quant in the block while leaving the dequantised values
// deceptively close.
__global__ void quantize_q8_0_kernel(BlockQ8_0* __restrict__ out,
                                     const float* __restrict__ x, int n_blocks) {
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
template <int BLOCK>
__global__ void mla_decode_attn_split_kernel(float* __restrict__ part_acc,
                                             float* __restrict__ part_ml,
                                             const float* __restrict__ q,
                                             const float* __restrict__ k_cache,
                                             int key_length, int kv_lora,
                                             int n_ctx, float scale, int splits) {
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
            const float* kt = k_cache + (size_t)(t0 + t) * key_length;
            float sdot = 0.0f;
            for (int d = lane; d < key_length; d += 32) sdot += s_q[d] * kt[d];
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
                a += s_p[t] * k_cache[(size_t)(t0 + t) * key_length + r];
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
template <int BLOCK, int HPB, int RSLOTS, int TPW = kMlaTokensPerWarp>
__global__ void mla_decode_attn_hbatch_kernel(float* __restrict__ part_acc,
                                              float* __restrict__ part_ml,
                                              const float* __restrict__ q,
                                              const float* __restrict__ k_cache,
                                              int key_length, int kv_lora,
                                              int n_ctx, float scale, int splits) {
    constexpr int NWARP = BLOCK / 32;
    const int h0   = blockIdx.x * HPB;
    const int sp   = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    const int chunk = (n_ctx + splits - 1) / splits;
    const int t_beg = sp * chunk;
    const int t_end = min(n_ctx, t_beg + chunk);

    extern __shared__ float smem[];
    float* s_q = smem;                                        // HPB * key_length
    float* s_p = s_q + (size_t)HPB * key_length;              // HPB * kMlaCtxTile
    float* s_m = s_p + (size_t)HPB * kMlaCtxTile;             // HPB  running max
    float* s_l = s_m + HPB;                                   // HPB  running exp-sum
    float* s_c = s_l + HPB;                                   // HPB  this tile's rescale

    for (int i = threadIdx.x; i < HPB * key_length; i += BLOCK)
        s_q[i] = q[(size_t)(h0 + i / key_length) * key_length + (i % key_length)];
    if (threadIdx.x < HPB) { s_m[threadIdx.x] = -1e30f; s_l[threadIdx.x] = 0.0f; }

    // Latent accumulators: RSLOTS r-values x HPB heads, in registers.
    float acc[RSLOTS][HPB];
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
            const float* kt[TPW];
#pragma unroll
            for (int tt = 0; tt < TPW; ++tt)
                kt[tt] = k_cache + (size_t)(t0 + tb + min(tt, tc - 1)) * key_length;

#pragma unroll 2
            for (int d = lane; d < key_length; d += 32) {
                float kv[TPW];
#pragma unroll
                for (int tt = 0; tt < TPW; ++tt) kv[tt] = kt[tt][d];
#pragma unroll
                for (int hh = 0; hh < HPB; ++hh) {
                    const float qv = s_q[hh * key_length + d];
#pragma unroll
                    for (int tt = 0; tt < TPW; ++tt) s[hh][tt] += qv * kv[tt];
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
                    if (lane == 0 && tt < tc) s_p[hh * kMlaCtxTile + tb + tt] = v * scale;
                }
        }
        __syncthreads();

        // --- 2. online softmax, one warp per head ---
        // (m, l) for head hh live in shared so they survive the tile loop, but the warp
        // that owns hh is the only thing that touches them. Read both before the
        // __syncwarp so no lane can observe this tile's write in place of the running
        // value it is supposed to fold into.
        for (int hh = warp; hh < HPB; hh += NWARP) {
            float* pr = s_p + hh * kMlaCtxTile;
            const float m_prev = s_m[hh];
            const float l_prev = s_l[hh];
            __syncwarp();

            float tm = -1e30f;
            for (int t = lane; t < tn; t += 32) tm = fmaxf(tm, pr[t]);
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
                const float e = __expf(pr[t] - m_new);
                pr[t] = e;
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
#pragma unroll
        for (int u = 0; u < RSLOTS; ++u)
#pragma unroll
            for (int hh = 0; hh < HPB; ++hh) acc[u][hh] *= s_c[hh];

#pragma unroll 4
        for (int t = 0; t < tn; ++t) {
            const float* kt = k_cache + (size_t)(t0 + t) * key_length;
            float p[HPB];
#pragma unroll
            for (int hh = 0; hh < HPB; ++hh) p[hh] = s_p[hh * kMlaCtxTile + t];
#pragma unroll
            for (int u = 0; u < RSLOTS; ++u) {
                const int r = threadIdx.x + u * BLOCK;
                if (r < kv_lora) {
                    const float kv = kt[r];
#pragma unroll
                    for (int hh = 0; hh < HPB; ++hh) acc[u][hh] += p[hh] * kv;
                }
            }
        }
        __syncthreads();
    }

    // UNNORMALISED partials, one (head, slice) row each — the combine applies 1/l.
#pragma unroll
    for (int u = 0; u < RSLOTS; ++u) {
        const int r = threadIdx.x + u * BLOCK;
        if (r < kv_lora)
#pragma unroll
            for (int hh = 0; hh < HPB; ++hh)
                part_acc[((size_t)(h0 + hh) * splits + sp) * kv_lora + r] = acc[u][hh];
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

template <int BLOCK>
__global__ void proj_q8_0_kernel(float* __restrict__ y, const float* __restrict__ x,
                                 const BlockQ8_0* __restrict__ W, int blocks_per_row) {
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

template <int BLOCK>
__global__ void proj_q8_0_q8_0_kernel(float* __restrict__ y,
                                      const BlockQ8_0* __restrict__ x,
                                      const BlockQ8_0* __restrict__ W,
                                      int blocks_per_row) {
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
            sumi = __dp4a(get_int_b2(row[b].qs, i), get_int_b2(x[b].qs, i), sumi);
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
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = src[i];
}

template <int BLOCK>
__global__ void proj_f32_kernel(float* __restrict__ y, const float* __restrict__ x,
                                const float* __restrict__ W, int K) {
    const int n = blockIdx.x;
    const float* row = W + (size_t)n * K;
    __shared__ float shm[BLOCK / 32 + 1];

    float acc = 0.0f;
    for (int k = threadIdx.x; k < K; k += BLOCK) acc += row[k] * x[k];
    acc = block_sum<BLOCK>(acc, shm);
    if (threadIdx.x == 0) y[n] = acc;
}


__global__ void add_f32_kernel(float* __restrict__ out, const float* __restrict__ a,
                               const float* __restrict__ b, int64_t n) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void sigmoid_inplace_f32_kernel(float* __restrict__ x, int64_t n) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) x[i] = 1.0f / (1.0f + __expf(-x[i]));
}

template <int BLOCK>
__global__ void rms_norm_kernel(float* __restrict__ out, const float* __restrict__ x,
                                const float* __restrict__ w, int n, float eps) {
    __shared__ float shm[BLOCK / 32 + 1];
    float acc = 0.0f;
    for (int d = threadIdx.x; d < n; d += BLOCK) acc += x[d] * x[d];
    const float ss = block_sum<BLOCK>(acc, shm);
    const float inv = rsqrtf(ss / (float)n + eps);
    for (int d = threadIdx.x; d < n; d += BLOCK) out[d] = x[d] * inv * w[d];
}

}  // namespace

// ---------------------------------------------------------------------------

void situ_f32(float* out, const float* gate, const float* up, int64_t n,
              float beta, float linear_beta, cudaStream_t stream) {
    if (n <= 0) return;
    const int T = 256;
    const int64_t blocks = (n + T - 1) / T;
    const int lb_active = linear_beta > 0.0f ? 1 : 0;
    situ_kernel<<<(unsigned)blocks, T, 0, stream>>>(
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
    if (head_dim == SMEM_BLOCK && head_dim % IC == 0) {
        kda_decode_step_smem_kernel<SMEM_BLOCK><<<(unsigned)n_head, T, shm_staged, stream>>>(
            out, state, q, k, v, g, beta, head_dim);
        return;
    }
    kda_decode_step_kernel<128><<<(unsigned)n_head, T, shm, stream>>>(
        out, state, q, k, v, g, beta, head_dim);
}

void kda_gate_out_f32(float* out, const float* o, const float* norm_w,
                      const float* g2, int head_dim, int n_head,
                      float eps, cudaStream_t stream) {
    if (head_dim <= 0 || n_head <= 0) return;
    kda_gate_out_kernel<128><<<(unsigned)n_head, 128, 0, stream>>>(
        out, o, norm_w, g2, head_dim, eps);
}

void attn_res_mix_f32(float* out, const float* ckpts, const float* cur,
                      const float* score_w, int n_embd, int n_ckpt,
                      float eps, cudaStream_t stream, float* scores) {
    if (n_embd <= 0) return;
    if (n_ckpt <= 0) {
        // Layer 0 / nothing banked: the reference returns cur unchanged.
        cudaMemcpyAsync(out, cur, (size_t)n_embd * sizeof(float),
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
    if (owned) cudaMallocAsync(&sc, (size_t)(n_ckpt + 1) * sizeof(float), stream);

    attn_res_score_kernel<B><<<(unsigned)(n_ckpt + 1), B, 0, stream>>>(
        sc, ckpts, cur, score_w, n_embd, n_ckpt, eps);
    attn_res_apply_kernel<B><<<(unsigned)((n_embd + B - 1) / B), B,
                              (size_t)(n_ckpt + 1) * sizeof(float), stream>>>(
        out, ckpts, cur, sc, n_embd, n_ckpt);

    if (owned) cudaFreeAsync(sc, stream);
}

void kda_conv_step_f32(float* out, float* state, const float* x, const float* w,
                       int d_conv, int d_inner, cudaStream_t stream) {
    if (d_conv < 2 || d_inner <= 0) return;
    const int T = 256;
    const int blocks = (d_inner + T - 1) / T;
    kda_conv_step_kernel<<<(unsigned)blocks, T, 0, stream>>>(out, state, x, w, d_conv, d_inner);
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

static void ensure_iq1s_tables() {
    static bool ready[kMaxTableDevices] = {false};
    const int dev = current_device_index();
    if (dev >= 0 && ready[dev]) return;
    cudaMemcpyToSymbol(iq1s_grid_c, iq1s_grid_host, sizeof(iq1s_grid_host));
    if (dev >= 0) ready[dev] = true;
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
    dequant_iq2_xs_kernel<<<(unsigned)blocks, T, 0, stream>>>(
        out, (const BlockIQ2XS*)src, n_groups);
}

void moe_router_noaux_tc_f32(float* out_w, int* out_ids, const float* logits,
                             const float* bias, int n_expert, int top_k,
                             int n_tokens, bool norm_w, float w_scale,
                             cudaStream_t stream) {
    if (n_expert <= 0 || top_k <= 0 || n_tokens <= 0) return;
    if (top_k > n_expert) return;
    const int T = 256;
    const size_t shm = (size_t)2 * n_expert * sizeof(float);
    moe_router_noaux_tc_kernel<256><<<(unsigned)n_tokens, T, shm, stream>>>(
        out_w, out_ids, logits, bias, n_expert, top_k, norm_w, w_scale);
}

// One launch path for both quant types. The two front doors below differ only in the
// table they prime and the block layout they name; duplicating the grid arithmetic
// across them is the same drift the block_dot overloads exist to prevent.
template <typename Blk>
static void moe_expert_ffn_launch(float* out, float* scratch,
                                  const float* x, const int* ids, const float* w,
                                  const void* gate_exps, const void* up_exps,
                                  const void* down_exps,
                                  int latent, int ffn, int top_k,
                                  float situ_beta, float situ_linear_beta,
                                  cudaStream_t stream,
                                  int expert_begin, int n_local_experts) {
    constexpr int WARPS = 8;                       // 256-thread CTAs
    const int lb_active = situ_linear_beta > 0.0f ? 1 : 0;
    // n_local <= 0 means "this rank holds every expert" — the tp_size 1 case, where
    // the band test must never reject. INT_MAX makes it a no-op rather than a branch.
    const int n_local = n_local_experts > 0 ? n_local_experts : INT_MAX;

    // Holds the "foreign slots read as zero" invariant the gate/up kernel used to
    // maintain per-CTA. Only a banded rank can leave a slot untouched; at tp_size 1
    // every slot is written, so memsetting there would be pure added work.
    if (n_local_experts > 0 || expert_begin != 0)
        cudaMemsetAsync(scratch, 0, (size_t)top_k * ffn * sizeof(float), stream);

    // float4 activation reads need 16-byte alignment. Both pointers are cudaMalloc
    // bases in every K3 caller and the per-block offsets are multiples of 1024 bytes,
    // so the bases alone decide it.
    const bool xvec = ((((uintptr_t)x) | ((uintptr_t)scratch)) & 15u) == 0;

    const dim3 g1((unsigned)((ffn + WARPS - 1) / WARPS), (unsigned)top_k);
    const size_t dshm = (size_t)top_k * sizeof(float);
    const float inv_lb = lb_active ? 1.0f / situ_linear_beta : 1.0f;

    if (xvec) {
        moe_gate_up_situ_kernel<WARPS, true, Blk><<<g1, WARPS * 32, 0, stream>>>(
            scratch, x, ids, (const Blk*)gate_exps, (const Blk*)up_exps, latent, ffn,
            situ_beta, 1.0f / situ_beta, situ_linear_beta, inv_lb, lb_active,
            expert_begin, n_local);
        moe_down_combine_kernel<WARPS, true, Blk>
            <<<(unsigned)latent, WARPS * 32, dshm, stream>>>(
                out, scratch, ids, w, (const Blk*)down_exps, latent, ffn, top_k,
                expert_begin, n_local);
    } else {
        moe_gate_up_situ_kernel<WARPS, false, Blk><<<g1, WARPS * 32, 0, stream>>>(
            scratch, x, ids, (const Blk*)gate_exps, (const Blk*)up_exps, latent, ffn,
            situ_beta, 1.0f / situ_beta, situ_linear_beta, inv_lb, lb_active,
            expert_begin, n_local);
        moe_down_combine_kernel<WARPS, false, Blk>
            <<<(unsigned)latent, WARPS * 32, dshm, stream>>>(
                out, scratch, ids, w, (const Blk*)down_exps, latent, ffn, top_k,
                expert_begin, n_local);
    }
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
    dequant_iq1_s_kernel<<<(unsigned)blocks, T, 0, stream>>>(
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
            dequant_f32_passthrough_kernel<<<(unsigned)blocks, T, 0, stream>>>(
                out, (const float*)src, n);
            return true;
        }
        case 8: {  // Q8_0 — the other type token_embd.weight / output.weight may use
            if (n <= 0 || n % 32 != 0) return false;
            const int64_t n_blocks = n / 32;
            const int T = 256;
            const int64_t blocks = (n_blocks + T - 1) / T;
            dequant_q8_0_kernel<<<(unsigned)blocks, T, 0, stream>>>(
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
        quantize_q8_k_kernel<<<((latent / 256) + T - 1) / T, T, 0, stream>>>(
            qin, x, latent / 256, 1);
        const int lb_active = situ_linear_beta > 0.0f ? 1 : 0;
        const int n_local = n_local_experts > 0 ? n_local_experts : INT_MAX;
        dim3 g1((unsigned)ffn, (unsigned)top_k);
        switch (ggml_type) {
            case 17:
                ensure_iq2xs_tables();
                moe_gate_up_situ_q8k_kernel<BlockIQ2XS><<<g1, 32, 0, stream>>>(
                    scratch, qin, ids, (const BlockIQ2XS*)gate_exps,
                    (const BlockIQ2XS*)up_exps, latent, ffn,
                    situ_beta, 1.0f / situ_beta, situ_linear_beta,
                    lb_active ? 1.0f / situ_linear_beta : 1.0f, lb_active,
                    expert_begin, n_local);
                break;
            case 19:
                ensure_iq1s_tables();
                moe_gate_up_situ_q8k_kernel<BlockIQ1S><<<g1, 32, 0, stream>>>(
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
        quantize_q8_k_kernel<<<(act_blocks + T - 1) / T, T, 0, stream>>>(
            qacts, scratch, ffn / 256, top_k);
        if (ggml_type == 17) {
            moe_down_combine_q8k_kernel<BlockIQ2XS><<<(unsigned)latent, 32, 0, stream>>>(
                out, qacts, ids, w, (const BlockIQ2XS*)down_exps,
                latent, ffn, top_k, expert_begin, n_local);
        } else {
            moe_down_combine_q8k_kernel<BlockIQ1S><<<(unsigned)latent, 32, 0, stream>>>(
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
    mla_gate_out_kernel<<<(unsigned)blocks, T, 0, stream>>>(out, attn_out, gate_proj, n);
}

void kda_decay_gate_f32(float* out, const float* g_raw, const float* A,
                        int head_dim, int n_head, float lower_bound,
                        cudaStream_t stream) {
    if (head_dim <= 0 || n_head <= 0) return;
    kda_decay_gate_kernel<<<(unsigned)n_head, head_dim, 0, stream>>>(
        out, g_raw, A, head_dim, lower_bound);
}

void l2_norm_heads_f32(float* out, const float* x, int head_dim, int n_head,
                       float scale, float eps, cudaStream_t stream) {
    if (head_dim <= 0 || n_head <= 0) return;
    l2_norm_heads_kernel<128><<<(unsigned)n_head, 128, 0, stream>>>(
        out, x, head_dim, scale, eps);
}

void mla_absorb_q_f32(float* out, const float* q_nope, const float* q_pe,
                      const float* wk_b, int qk_nope, int kv_lora, int rope_dim,
                      int n_head, cudaStream_t stream) {
    if (n_head <= 0 || kv_lora <= 0 || qk_nope <= 0) return;
    dim3 grid((unsigned)kv_lora, (unsigned)n_head);
    mla_absorb_q_kernel<128><<<grid, 128, 0, stream>>>(
        out, q_nope, q_pe, wk_b, qk_nope, kv_lora, rope_dim);
}

void mla_decode_attn_f32(float* out, const float* q, const float* k_cache,
                         const float* wv_b, int key_length, int kv_lora,
                         int v_dim, int n_head, int n_ctx, float scale,
                         cudaStream_t stream) {
    if (n_head <= 0 || n_ctx <= 0 || key_length <= 0 || kv_lora <= 0 || v_dim <= 0) return;
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
    int splits = (n_ctx >= kMlaSplitMinCtx && k3_mla_split_scratch(n_head, kv_lora, &dev))
               ? std::min(k3_mla_max_splits(n_head), (n_ctx + kMlaSplitMinCtx - 1) / kMlaSplitMinCtx)
               : 1;
    if (splits <= 1) {
        mla_decode_attn_kernel<BLOCK><<<(unsigned)n_head, BLOCK, shm, stream>>>(
            out, q, k_cache, wv_b, key_length, kv_lora, v_dim, n_ctx, scale);
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
        if (sm > 0) {
            const int fill      = (kMlaBlocksPerSm * sm + groups - 1) / groups;
            const int min_slice = k3_mla_min_slice_len(hpb, key_length, kv_lora);
            const int by_len    = std::max(1, n_ctx / min_slice);
            splits = std::max(splits, std::min(fill, by_len));
            splits = std::min(std::max(splits, 1), k3_mla_max_splits(n_head));
        }
        dim3 hgrid((unsigned)groups, (unsigned)splits);
        bool launched = true;
        switch (hpb * 8 + rslots) {
            case 12 * 8 + 1: mla_decode_attn_hbatch_kernel<BLOCK, 12, 1><<<hgrid, BLOCK, hshm, stream>>>(
                g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache, key_length, kv_lora, n_ctx, scale, splits); break;
            case 12 * 8 + 2: mla_decode_attn_hbatch_kernel<BLOCK, 12, 2><<<hgrid, BLOCK, hshm, stream>>>(
                g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache, key_length, kv_lora, n_ctx, scale, splits); break;
            case 12 * 8 + 3: mla_decode_attn_hbatch_kernel<BLOCK, 12, 3><<<hgrid, BLOCK, hshm, stream>>>(
                g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache, key_length, kv_lora, n_ctx, scale, splits); break;
            case 12 * 8 + 4: mla_decode_attn_hbatch_kernel<BLOCK, 12, 4><<<hgrid, BLOCK, hshm, stream>>>(
                g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache, key_length, kv_lora, n_ctx, scale, splits); break;
            case 8 * 8 + 1: mla_decode_attn_hbatch_kernel<BLOCK, 8, 1><<<hgrid, BLOCK, hshm, stream>>>(
                g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache, key_length, kv_lora, n_ctx, scale, splits); break;
            case 8 * 8 + 2: mla_decode_attn_hbatch_kernel<BLOCK, 8, 2><<<hgrid, BLOCK, hshm, stream>>>(
                g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache, key_length, kv_lora, n_ctx, scale, splits); break;
            case 8 * 8 + 3: mla_decode_attn_hbatch_kernel<BLOCK, 8, 3><<<hgrid, BLOCK, hshm, stream>>>(
                g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache, key_length, kv_lora, n_ctx, scale, splits); break;
            case 8 * 8 + 4: mla_decode_attn_hbatch_kernel<BLOCK, 8, 4><<<hgrid, BLOCK, hshm, stream>>>(
                g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache, key_length, kv_lora, n_ctx, scale, splits); break;
            case 4 * 8 + 1: mla_decode_attn_hbatch_kernel<BLOCK, 4, 1><<<hgrid, BLOCK, hshm, stream>>>(
                g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache, key_length, kv_lora, n_ctx, scale, splits); break;
            case 4 * 8 + 2: mla_decode_attn_hbatch_kernel<BLOCK, 4, 2><<<hgrid, BLOCK, hshm, stream>>>(
                g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache, key_length, kv_lora, n_ctx, scale, splits); break;
            case 4 * 8 + 3: mla_decode_attn_hbatch_kernel<BLOCK, 4, 3><<<hgrid, BLOCK, hshm, stream>>>(
                g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache, key_length, kv_lora, n_ctx, scale, splits); break;
            case 4 * 8 + 4: mla_decode_attn_hbatch_kernel<BLOCK, 4, 4><<<hgrid, BLOCK, hshm, stream>>>(
                g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache, key_length, kv_lora, n_ctx, scale, splits); break;
            default: launched = false; break;
        }
        if (launched) {
            const size_t cshm = ((size_t)kv_lora + (size_t)splits + 1) * sizeof(float);
            mla_decode_combine_kernel<BLOCK><<<(unsigned)n_head, BLOCK, cshm, stream>>>(
                out, g_mla_part_acc[dev], g_mla_part_ml[dev], wv_b, kv_lora, v_dim, splits);
            return;
        }
        splits = std::min(splits, k3_mla_max_splits(n_head));
    }

    dim3 grid((unsigned)n_head, (unsigned)splits);
    mla_decode_attn_split_kernel<BLOCK><<<grid, BLOCK, shm, stream>>>(
        g_mla_part_acc[dev], g_mla_part_ml[dev], q, k_cache, key_length, kv_lora,
        n_ctx, scale, splits);
    // s_acc + s_w, where s_w needs splits + 1 slots (the last holds 1/l).
    const size_t cshm = ((size_t)kv_lora + (size_t)splits + 1) * sizeof(float);
    mla_decode_combine_kernel<BLOCK><<<(unsigned)n_head, BLOCK, cshm, stream>>>(
        out, g_mla_part_acc[dev], g_mla_part_ml[dev], wv_b, kv_lora, v_dim, splits);
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
        case 0:   // F32, dense
            proj_f32_kernel<BLOCK><<<(unsigned)N, BLOCK, 0, stream>>>(
                y, x, (const float*)W, K);
            return true;
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
                const unsigned grid = (unsigned)((N + ROWS_W16 - 1) / ROWS_W16);
                switch (TB) {
                    case 32:
                        proj_q8_0_multirow_kernel<32, ROWS_W16><<<grid, 32, 0, stream>>>(
                            y, x, (const BlockQ8_0*)W, nb, N);
                        break;
                    case 64:
                        proj_q8_0_multirow_kernel<64, ROWS_W16><<<grid, 64, 0, stream>>>(
                            y, x, (const BlockQ8_0*)W, nb, N);
                        break;
                    default:
                        proj_q8_0_multirow_kernel<BLOCK, ROWS_W16>
                            <<<grid, BLOCK, 0, stream>>>(y, x, (const BlockQ8_0*)W, nb, N);
                        break;
                }
            } else if (N >= ROWS_W8 * MIN_BLOCKS) {
                const unsigned grid = (unsigned)((N + ROWS_W8 - 1) / ROWS_W8);
                switch (TB) {
                    case 32:
                        proj_q8_0_multirow_kernel<32, ROWS_W8><<<grid, 32, 0, stream>>>(
                            y, x, (const BlockQ8_0*)W, nb, N);
                        break;
                    case 64:
                        proj_q8_0_multirow_kernel<64, ROWS_W8><<<grid, 64, 0, stream>>>(
                            y, x, (const BlockQ8_0*)W, nb, N);
                        break;
                    default:
                        proj_q8_0_multirow_kernel<BLOCK, ROWS_W8>
                            <<<grid, BLOCK, 0, stream>>>(y, x, (const BlockQ8_0*)W, nb, N);
                        break;
                }
            } else if (N >= ROWS * MIN_BLOCKS) {
                const unsigned grid = (unsigned)((N + ROWS - 1) / ROWS);
                switch (TB) {
                    case 32:
                        proj_q8_0_multirow_kernel<32, ROWS><<<grid, 32, 0, stream>>>(
                            y, x, (const BlockQ8_0*)W, nb, N);
                        break;
                    case 64:
                        proj_q8_0_multirow_kernel<64, ROWS><<<grid, 64, 0, stream>>>(
                            y, x, (const BlockQ8_0*)W, nb, N);
                        break;
                    default:
                        proj_q8_0_multirow_kernel<BLOCK, ROWS><<<grid, BLOCK, 0, stream>>>(
                            y, x, (const BlockQ8_0*)W, nb, N);
                        break;
                }
            } else {
                switch (TB) {
                    case 32:
                        proj_q8_0_kernel<32><<<(unsigned)N, 32, 0, stream>>>(
                            y, x, (const BlockQ8_0*)W, nb);
                        break;
                    case 64:
                        proj_q8_0_kernel<64><<<(unsigned)N, 64, 0, stream>>>(
                            y, x, (const BlockQ8_0*)W, nb);
                        break;
                    default:
                        proj_q8_0_kernel<BLOCK><<<(unsigned)N, BLOCK, 0, stream>>>(
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
                         cudaStream_t stream) {
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
    constexpr int QT = 128;
    quantize_q8_0_kernel<<<(nb + QT - 1) / QT, QT, 0, stream>>>(
        (BlockQ8_0*)q8_scratch, x, nb);

    const unsigned grid = (unsigned)((N + ROWS - 1) / ROWS);
    const BlockQ8_0* xq = (const BlockQ8_0*)q8_scratch;
#define K3_QQ4_LAUNCH(BS)                                                        \
    proj_q8_0_q8_0_fused4_kernel<BS, ROWS><<<grid, BS, 0, stream>>>(             \
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
            proj_q8_0_fused4_kernel<32, ROWS><<<grid, 32, 0, stream>>>(
                y0, y1, y2, y3, x, (const BlockQ8_0*)W0, (const BlockQ8_0*)W1,
                (const BlockQ8_0*)W2, (const BlockQ8_0*)W3, nb, N);
            break;
        case 64:
            proj_q8_0_fused4_kernel<64, ROWS><<<grid, 64, 0, stream>>>(
                y0, y1, y2, y3, x, (const BlockQ8_0*)W0, (const BlockQ8_0*)W1,
                (const BlockQ8_0*)W2, (const BlockQ8_0*)W3, nb, N);
            break;
        default:
            proj_q8_0_fused4_kernel<128, ROWS><<<grid, 128, 0, stream>>>(
                y0, y1, y2, y3, x, (const BlockQ8_0*)W0, (const BlockQ8_0*)W1,
                (const BlockQ8_0*)W2, (const BlockQ8_0*)W3, nb, N);
            break;
    }
    return true;
}

bool k3_proj_ggml_f32(float* y, const float* x, const void* W, int wtype,
                      int N, int K, void* q8_scratch, cudaStream_t stream) {
    if (N <= 0 || K <= 0) return false;
    if (wtype == 0)
        return k3_proj_f32(y, x, W, wtype, N, K, stream);
    if (wtype != 8 || !q8_scratch || K % 32 != 0) return false;
    const int nb = K / 32;
    constexpr int QT = 128;
    quantize_q8_0_kernel<<<(nb + QT - 1) / QT, QT, 0, stream>>>(
        (BlockQ8_0*)q8_scratch, x, nb);
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
                proj_q8_0_q8_0_multirow_kernel<32, R><<<grid, 32, 0, stream>>>(       \
                    y, (const BlockQ8_0*)q8_scratch, (const BlockQ8_0*)W, nb, N);      \
                break;                                                               \
            case 64:                                                                 \
                proj_q8_0_q8_0_multirow_kernel<64, R><<<grid, 64, 0, stream>>>(       \
                    y, (const BlockQ8_0*)q8_scratch, (const BlockQ8_0*)W, nb, N);      \
                break;                                                               \
            default:                                                                 \
                proj_q8_0_q8_0_multirow_kernel<BLOCK, R><<<grid, BLOCK, 0, stream>>>( \
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
        switch (TB) {
            case 32:
                proj_q8_0_q8_0_kernel<32><<<(unsigned)N, 32, 0, stream>>>(
                    y, (const BlockQ8_0*)q8_scratch, (const BlockQ8_0*)W, nb);
                break;
            case 64:
                proj_q8_0_q8_0_kernel<64><<<(unsigned)N, 64, 0, stream>>>(
                    y, (const BlockQ8_0*)q8_scratch, (const BlockQ8_0*)W, nb);
                break;
            default:
                proj_q8_0_q8_0_kernel<BLOCK><<<(unsigned)N, BLOCK, 0, stream>>>(
                    y, (const BlockQ8_0*)q8_scratch, (const BlockQ8_0*)W, nb);
                break;
        }
    }
#undef K3_QQ_LAUNCH
    return true;
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

void rms_norm_f32(float* out, const float* x, const float* w, int n, float eps,
                  cudaStream_t stream) {
    if (n <= 0) return;
    constexpr int BLOCK = 128;
    rms_norm_kernel<BLOCK><<<1, BLOCK, 0, stream>>>(out, x, w, n, eps);
}

void k3_add_f32(float* out, const float* a, const float* b, int64_t n,
               cudaStream_t stream) {
    if (n <= 0) return;
    const int T = 256;
    const int64_t blocks = (n + T - 1) / T;
    add_f32_kernel<<<(unsigned)blocks, T, 0, stream>>>(out, a, b, n);
}

void sigmoid_inplace_f32(float* x, int64_t n, cudaStream_t stream) {
    if (n <= 0) return;
    const int T = 256;
    const int64_t blocks = (n + T - 1) / T;
    sigmoid_inplace_f32_kernel<<<(unsigned)blocks, T, 0, stream>>>(x, n);
}

}  // namespace k3
}  // namespace kernels
}  // namespace sparkinfer
