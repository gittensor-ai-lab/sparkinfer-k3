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
// One block. Pass 1: score every banked checkpoint plus the current stream from the
// NORMALISED values. Pass 2: softmax. Pass 3: weighted sum over the RAW values.
// The normalised/raw split is the reference's, not a choice — see the header.

template <int BLOCK>
__global__ void attn_res_mix_kernel(float* __restrict__ out,
                                    const float* __restrict__ ckpts,
                                    const float* __restrict__ cur,
                                    const float* __restrict__ score_w,
                                    int n_embd, int n_ckpt, float eps,
                                    float* __restrict__ scratch) {
    __shared__ float shm[BLOCK / 32 + 1];
    float* scores = scratch;              // n_ckpt + 1

    // --- score the banked checkpoints ---
    for (int c = 0; c < n_ckpt; ++c) {
        const float* src = ckpts + (size_t)c * n_embd;
        float ss = 0.0f;
        for (int d = threadIdx.x; d < n_embd; d += BLOCK) ss += src[d] * src[d];
        ss = block_sum<BLOCK>(ss, shm);
        const float inv = rsqrtf(ss / (float)n_embd + eps);
        float dot = 0.0f;
        for (int d = threadIdx.x; d < n_embd; d += BLOCK) dot += (src[d] * inv) * score_w[d];
        dot = block_sum<BLOCK>(dot, shm);
        if (threadIdx.x == 0) scores[c] = dot;
        __syncthreads();
    }

    // --- score the current stream ---
    {
        float ss = 0.0f;
        for (int d = threadIdx.x; d < n_embd; d += BLOCK) ss += cur[d] * cur[d];
        ss = block_sum<BLOCK>(ss, shm);
        const float inv = rsqrtf(ss / (float)n_embd + eps);
        float dot = 0.0f;
        for (int d = threadIdx.x; d < n_embd; d += BLOCK) dot += (cur[d] * inv) * score_w[d];
        dot = block_sum<BLOCK>(dot, shm);
        if (threadIdx.x == 0) scores[n_ckpt] = dot;
        __syncthreads();
    }

    // --- softmax over n_ckpt + 1, computed once by thread 0 (tiny) ---
    if (threadIdx.x == 0) {
        const int n = n_ckpt + 1;
        float mx = scores[0];
        for (int c = 1; c < n; ++c) mx = fmaxf(mx, scores[c]);
        float sum = 0.0f;
        for (int c = 0; c < n; ++c) { scores[c] = __expf(scores[c] - mx); sum += scores[c]; }
        const float inv = 1.0f / sum;
        for (int c = 0; c < n; ++c) scores[c] *= inv;
    }
    __syncthreads();

    // --- weighted sum over the RAW values ---
    for (int d = threadIdx.x; d < n_embd; d += BLOCK) {
        float acc = 0.0f;
        for (int c = 0; c < n_ckpt; ++c) acc += scores[c] * ckpts[(size_t)c * n_embd + d];
        acc += scores[n_ckpt] * cur[d];
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
// not worth the combine pass; kMlaMaxSplits caps the scratch and keeps the combine's
// per-slice loop short.
constexpr int kMlaSplitMinCtx = 4096;
constexpr int kMlaMaxSplits   = 32;

// ...and the two the OCCUPANCY floor needs. n_head is how many heads THIS CALL runs,
// which under tensor parallelism is a band rather than all 96, and the grid is
// n_head * splits. A slice count fixed against context alone therefore shrinks the
// grid by exactly the factor the band shrank the work, and hands the saving back as
// idle SMs: at 128k a 12-head rank would launch 384 blocks onto 132 SMs where the
// unbanded call launches 3072.
//
// kMlaMinSlice is the floor on slice LENGTH — the point past which a block spends
// more on its own prologue and its combine partial than on the cache walk. It binds
// before kMlaMaxSplitsAbs at every context below 128k.
constexpr int kMlaMinSlice     = 1024;
constexpr int kMlaMaxSplitsAbs = 128;

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

static bool k3_mla_split_scratch(int n_head, int splits, int kv_lora, int* dev_out) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    if (dev < 0 || dev >= kMlaMaxDevices) return false;   // fall back to the un-split path
    *dev_out = dev;

    const size_t need = (size_t)n_head * (size_t)splits * (size_t)kv_lora;
    if (need <= g_mla_part_cap[dev]) return true;
    cudaFree(g_mla_part_acc[dev]);
    cudaFree(g_mla_part_ml [dev]);
    g_mla_part_acc[dev] = nullptr; g_mla_part_ml[dev] = nullptr; g_mla_part_cap[dev] = 0;
    if (cudaMalloc((void**)&g_mla_part_acc[dev], need * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc((void**)&g_mla_part_ml[dev],
                   (size_t)n_head * (size_t)splits * 2 * sizeof(float)) != cudaSuccess) {
        cudaFree(g_mla_part_acc[dev]); g_mla_part_acc[dev] = nullptr; return false;
    }
    g_mla_part_cap[dev] = need;
    return true;
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

    // m = max over slices. Small (splits <= 64), so thread 0 is cheaper than a reduction.
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

// `blocks_per_row` is how many blocks this call REDUCES; `row_stride` is how far apart
// two rows of W are. They differ only when the caller is reducing a column slice of a
// wider tensor (see k3_proj_cols_f32) — for a whole-row projection they are equal and
// the indexing is byte-for-byte what it was before the parameter existed.
template <int BLOCK>
__global__ void proj_q8_0_kernel(float* __restrict__ y, const float* __restrict__ x,
                                 const BlockQ8_0* __restrict__ W, int blocks_per_row,
                                 int row_stride) {
    const int n = blockIdx.x;
    const BlockQ8_0* row = W + (size_t)n * row_stride;
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
                                          int blocks_per_row, int n_rows,
                                          int row_stride) {
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
                const BlockQ8_0* row = W + (size_t)(n0 + r) * row_stride;
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
            const BlockQ8_0* row = W + (size_t)(n0 + r) * row_stride;
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
                                      int blocks_per_row, int row_stride) {
    const int n = blockIdx.x;
    const BlockQ8_0* row = W + (size_t)n * row_stride;
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
                                const float* __restrict__ W, int K, int row_stride) {
    const int n = blockIdx.x;
    const float* row = W + (size_t)n * row_stride;
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
                      float eps, cudaStream_t stream) {
    if (n_embd <= 0) return;
    if (n_ckpt <= 0) {
        // Layer 0 / nothing banked: the reference returns cur unchanged.
        cudaMemcpyAsync(out, cur, (size_t)n_embd * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        return;
    }
    float* scratch = nullptr;
    cudaMallocAsync(&scratch, (size_t)(n_ckpt + 1) * sizeof(float), stream);
    attn_res_mix_kernel<256><<<1, 256, 0, stream>>>(
        out, ckpts, cur, score_w, n_embd, n_ckpt, eps, scratch);
    cudaFreeAsync(scratch, stream);
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

// Blocks of this kernel that fit on one SM, asked of the driver rather than guessed,
// and cached per device because it is a property of the compiled kernel and the
// launch shape, both fixed for the life of the process.
//
// Used only to decide how many slices keep the machine busy — a wrong answer costs
// occupancy, never correctness — so a failed query degrades to 1 rather than failing.
template <int BLOCK>
static int k3_mla_blocks_per_sm(size_t shm) {
    static int cached[kMlaMaxDevices] = {0};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kMlaMaxDevices) return 1;
    if (cached[dev] > 0) return cached[dev];
    int n = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &n, mla_decode_attn_split_kernel<BLOCK>, BLOCK, shm) != cudaSuccess || n < 1)
        n = 1;
    cached[dev] = n;
    return n;
}

// How many context slices to cut, given the heads this call actually runs.
//
// Two rules, and the second only ever RAISES the count:
//   1. context: one slice per kMlaSplitMinCtx tokens, capped at kMlaMaxSplits. This
//      is unchanged, and at the full 96 heads it is still what decides — rule 2 asks
//      for 11 slices there, well under the 32 this gives at 128k.
//   2. occupancy: enough slices that n_head * splits covers the resident blocks the
//      device can hold, bounded by kMlaMinSlice tokens per slice.
//
// Rule 2 only ever RESCALES a launch rule 1 already chose to split. Where rule 1 says
// one block per head, that stands — the un-split kernel is the one kimi_k3_numeric_test
// pins bit-for-bit, and moving that boundary would move what the test is testing.
static int k3_mla_split_count(int n_head, int n_ctx, size_t shm, int blocks_per_sm) {
    if (n_ctx < kMlaSplitMinCtx || n_head <= 0) return 1;
    (void)shm;
    int splits = std::min(kMlaMaxSplits, (n_ctx + kMlaSplitMinCtx - 1) / kMlaSplitMinCtx);
    if (splits <= 1) return 1;

    int sms = 0, dev = 0;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess &&
        sms > 0) {
        const int want = (sms * blocks_per_sm + n_head - 1) / n_head;
        if (want > splits) splits = want;
    }
    const int by_length = n_ctx / kMlaMinSlice;
    if (splits > by_length) splits = by_length;
    if (splits > kMlaMaxSplitsAbs) splits = kMlaMaxSplitsAbs;
    return splits < 1 ? 1 : splits;
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
    //
    // n_head is whatever the CALLER runs, which under tensor parallelism is a band, so
    // the slice count is derived from it — see k3_mla_split_count.
    int dev = 0;
    int splits = k3_mla_split_count(n_head, n_ctx, shm, k3_mla_blocks_per_sm<BLOCK>(shm));
    if (splits > 1 && !k3_mla_split_scratch(n_head, splits, kv_lora, &dev)) splits = 1;
    if (splits <= 1) {
        mla_decode_attn_kernel<BLOCK><<<(unsigned)n_head, BLOCK, shm, stream>>>(
            out, q, k_cache, wv_b, key_length, kv_lora, v_dim, n_ctx, scale);
        return;
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

// The one dispatch both k3_proj_f32 and k3_proj_cols_f32 go through. `W` already
// points at the first element (F32) or block (Q8_0) of THIS call's slice of row 0;
// `K` is the slice width and `K_stride` the distance between rows of the underlying
// tensor. K == K_stride is the whole-row case and reproduces the previous code
// exactly — same kernels, same block sizes, same thread-to-block striding, so the
// result is bit-identical, not merely close.
static bool proj_launch(float* y, const float* x, const void* W, int wtype,
                        int N, int K, int K_stride, cudaStream_t stream) {
    if (N <= 0 || K <= 0 || K_stride < K) return false;
    constexpr int BLOCK = 128;
    switch (wtype) {
        case 0:   // F32, dense
            proj_f32_kernel<BLOCK><<<(unsigned)N, BLOCK, 0, stream>>>(
                y, x, (const float*)W, K, K_stride);
            return true;
        case 8: {  // Q8_0
            if (K % 32 != 0 || K_stride % 32 != 0) return false;
            const int stride_nb = K_stride / 32;
            // Multi-row amortises the activation re-read, but it also divides the grid
            // by ROWS. That is only free while the grid still covers the device several
            // times over — K3's big projections are N=12288 (grid 3072), whereas the
            // small ones (n_head=96 for ssm_beta) would drop to 24 blocks and lose more
            // to idle SMs than the activation traffic was costing. Single-row below the
            // threshold keeps those on the wide grid.
            constexpr int ROWS = 4;
            constexpr int MULTIROW_MIN_N = 1024;   // grid >= 256 blocks
            const int nb = K / 32;
            const int TB = proj_block_for(nb);
            if (N >= MULTIROW_MIN_N) {
                const unsigned grid = (unsigned)((N + ROWS - 1) / ROWS);
                switch (TB) {
                    case 32:
                        proj_q8_0_multirow_kernel<32, ROWS><<<grid, 32, 0, stream>>>(
                            y, x, (const BlockQ8_0*)W, nb, N, stride_nb);
                        break;
                    case 64:
                        proj_q8_0_multirow_kernel<64, ROWS><<<grid, 64, 0, stream>>>(
                            y, x, (const BlockQ8_0*)W, nb, N, stride_nb);
                        break;
                    default:
                        proj_q8_0_multirow_kernel<BLOCK, ROWS><<<grid, BLOCK, 0, stream>>>(
                            y, x, (const BlockQ8_0*)W, nb, N, stride_nb);
                        break;
                }
            } else {
                switch (TB) {
                    case 32:
                        proj_q8_0_kernel<32><<<(unsigned)N, 32, 0, stream>>>(
                            y, x, (const BlockQ8_0*)W, nb, stride_nb);
                        break;
                    case 64:
                        proj_q8_0_kernel<64><<<(unsigned)N, 64, 0, stream>>>(
                            y, x, (const BlockQ8_0*)W, nb, stride_nb);
                        break;
                    default:
                        proj_q8_0_kernel<BLOCK><<<(unsigned)N, BLOCK, 0, stream>>>(
                            y, x, (const BlockQ8_0*)W, nb, stride_nb);
                        break;
                }
            }
            return true;
        }
        default:
            return false;
    }
}

bool k3_proj_f32(float* y, const float* x, const void* W, int wtype,
                 int N, int K, cudaStream_t stream) {
    return proj_launch(y, x, W, wtype, N, K, K, stream);
}

// Byte offset of column k_off in a row of a [*, K] tensor of type `wtype`. Q8_0 rows
// are 34-byte blocks of 32 weights, so a slice can only start on a block boundary —
// which every K3 head band does (all head widths here are multiples of 32).
static bool proj_col_offset(int wtype, int k_off, size_t* out_bytes) {
    if (k_off < 0) return false;
    if (wtype == 0) { *out_bytes = (size_t)k_off * sizeof(float); return true; }
    if (wtype == 8) {
        if (k_off % 32 != 0) return false;
        *out_bytes = (size_t)(k_off / 32) * sizeof(BlockQ8_0);
        return true;
    }
    return false;
}

bool k3_proj_cols_f32(float* y, const float* x, const void* W, int wtype,
                      int N, int k_off, int k_len, int K, cudaStream_t stream) {
    if (k_len <= 0 || k_off + k_len > K) return false;
    size_t off = 0;
    if (!proj_col_offset(wtype, k_off, &off)) return false;
    return proj_launch(y, x + k_off, (const char*)W + off, wtype, N, k_len, K, stream);
}

bool k3_proj_f32_x4(float* y0, float* y1, float* y2, float* y3, const float* x,
                    const void* W0, const void* W1, const void* W2, const void* W3,
                    int wtype, int N, int K, cudaStream_t stream) {
    // Deliberately narrow: Q8_0 only, all four the same shape. Anything else returns
    // false so the caller falls back to four ordinary k3_proj_f32 calls rather than this
    // path silently handling a case it was not written for.
    if (N <= 0 || K <= 0 || wtype != 8 || K % 32 != 0) return false;
    if (!y0 || !y1 || !y2 || !y3 || !W0 || !W1 || !W2 || !W3) return false;
    constexpr int ROWS = 2;
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

// Shared by k3_proj_ggml_f32 and its column-sliced form, on the same terms as
// proj_launch: `W` points at row 0 of the slice, `K` is the slice width, `K_stride`
// the row pitch of the underlying tensor.
//
// Quantising only the slice is not an approximation of quantising the whole
// activation and taking part of it. Q8_0 scales are per 32-value block and the
// slice is block-aligned, so the blocks this call produces are the same blocks,
// with the same scales, that the full-width call would have produced.
static bool proj_q8act_launch(float* y, const float* x, const void* W, int wtype,
                              int N, int K, int K_stride, void* q8_scratch,
                              cudaStream_t stream) {
    if (N <= 0 || K <= 0 || K_stride < K) return false;
    if (wtype == 0)
        return proj_launch(y, x, W, wtype, N, K, K_stride, stream);
    if (wtype != 8 || !q8_scratch || K % 32 != 0 || K_stride % 32 != 0) return false;
    const int nb = K / 32;
    const int stride_nb = K_stride / 32;
    constexpr int QT = 128;
    quantize_q8_0_kernel<<<(nb + QT - 1) / QT, QT, 0, stream>>>(
        (BlockQ8_0*)q8_scratch, x, nb);
    // Same idle-thread sizing as the f32-activation path above.
    constexpr int BLOCK = 128;
    switch (proj_block_for(nb)) {
        case 32:
            proj_q8_0_q8_0_kernel<32><<<(unsigned)N, 32, 0, stream>>>(
                y, (const BlockQ8_0*)q8_scratch, (const BlockQ8_0*)W, nb, stride_nb);
            break;
        case 64:
            proj_q8_0_q8_0_kernel<64><<<(unsigned)N, 64, 0, stream>>>(
                y, (const BlockQ8_0*)q8_scratch, (const BlockQ8_0*)W, nb, stride_nb);
            break;
        default:
            proj_q8_0_q8_0_kernel<BLOCK><<<(unsigned)N, BLOCK, 0, stream>>>(
                y, (const BlockQ8_0*)q8_scratch, (const BlockQ8_0*)W, nb, stride_nb);
            break;
    }
    return true;
}

bool k3_proj_ggml_f32(float* y, const float* x, const void* W, int wtype,
                      int N, int K, void* q8_scratch, cudaStream_t stream) {
    return proj_q8act_launch(y, x, W, wtype, N, K, K, q8_scratch, stream);
}

bool k3_proj_cols_ggml_f32(float* y, const float* x, const void* W, int wtype,
                           int N, int k_off, int k_len, int K, void* q8_scratch,
                           cudaStream_t stream) {
    if (k_len <= 0 || k_off + k_len > K) return false;
    size_t off = 0;
    if (!proj_col_offset(wtype, k_off, &off)) return false;
    return proj_q8act_launch(y, x + k_off, (const char*)W + off, wtype, N, k_len, K,
                             q8_scratch, stream);
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
