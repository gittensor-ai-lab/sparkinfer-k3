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
#include "sparkinfer/kernels/iq1s_tables.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <cstdio>

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

    s_k[j] = kh[j];
    s_q[j] = qh[j];
    __syncthreads();

    // Step 1: per-channel decay on column j, and Step 2: sk[j] = sum_i S[i][j]*k[i].
    // Fused into one pass over the column — the decay is applied as we read.
    const float decay = __expf(gh[j]);
    float sk = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
        const float sv = S[(size_t)j * head_dim + i] * decay;
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

__constant__ uint64_t c_iq2xs_grid[512];
__constant__ uint8_t  c_ksigns_iq2xs[128];
__constant__ uint8_t  c_kmask_iq2xs[8];

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
        const uint8_t* grid = (const uint8_t*)&c_iq2xs_grid[q & 511];
        const uint8_t signs = c_ksigns_iq2xs[q >> 9];
        const float* xv = x + l * 8;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float wv = db * (float)grid[j] * ((signs & c_kmask_iq2xs[j]) ? -1.0f : 1.0f);
            acc += wv * xv[j];
        }
    }
    return acc;
}

// IQ1_S counterpart. Same contract: decode inside the dot product and never stage
// dequantised weights. Overloaded on the block type so the MoE kernels below are
// written once and instantiated per quant type — the combine logic must not exist
// twice, or the two copies will drift.
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
        const int8_t* grid = (const int8_t*)&iq1s_grid_c[idx];
        const float* xv = x + l32 * 8;
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc += dl * ((float)grid[j] + delta) * xv[j];
    }
    return acc;
}

// gate/up GEMV + situ. One block per (selected expert k, output chunk).
template <int BLOCK, typename Blk>
__global__ void moe_gate_up_situ_kernel(float* __restrict__ scratch,
                                        const float* __restrict__ x,
                                        const int* __restrict__ ids,
                                        const Blk* __restrict__ gate_exps,
                                        const Blk* __restrict__ up_exps,
                                        int latent, int ffn,
                                        float beta, float inv_beta,
                                        float lb, float inv_lb, int lb_active) {
    const int k = blockIdx.y;                 // which selected expert
    const int j = blockIdx.x;                 // which ffn output row
    if (j >= ffn) return;
    const int e = ids[k];
    const int blocks_per_row = latent / 256;

    const Blk* g_row = gate_exps + (size_t)(e * ffn + j) * blocks_per_row;
    const Blk* u_row = up_exps   + (size_t)(e * ffn + j) * blocks_per_row;

    float gacc = 0.0f, uacc = 0.0f;
    for (int b = 0; b < blocks_per_row; ++b) {
        const float* xb = x + b * 256;
        gacc += block_dot(g_row[b], xb, threadIdx.x, BLOCK);
        uacc += block_dot(u_row[b], xb, threadIdx.x, BLOCK);
    }
    __shared__ float shm[BLOCK / 32 + 1];
    gacc = block_sum<BLOCK>(gacc, shm);
    __syncthreads();
    uacc = block_sum<BLOCK>(uacc, shm);

    if (threadIdx.x == 0) {
        // situ, identical to situ_kernel
        const float a  = beta * tanhf(gacc * inv_beta) * sigmoidf_(gacc);
        const float ub = lb_active ? (lb * tanhf(uacc * inv_lb)) : uacc;
        scratch[(size_t)k * ffn + j] = a * ub;
    }
}

// down GEMV + weighted combine. One block per output element; accumulates the
// top_k experts so the combine needs no second pass and no atomics.
template <int BLOCK, typename Blk>
__global__ void moe_down_combine_kernel(float* __restrict__ out,
                                        const float* __restrict__ scratch,
                                        const int* __restrict__ ids,
                                        const float* __restrict__ w,
                                        const Blk* __restrict__ down_exps,
                                        int latent, int ffn, int top_k) {
    const int o = blockIdx.x;                 // output element in [0, latent)
    if (o >= latent) return;
    const int blocks_per_row = ffn / 256;
    __shared__ float shm[BLOCK / 32 + 1];

    float total = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        const int e = ids[k];
        const Blk* d_row = down_exps + (size_t)(e * latent + o) * blocks_per_row;
        const float* act = scratch + (size_t)k * ffn;
        float acc = 0.0f;
        for (int b = 0; b < blocks_per_row; ++b)
            acc += block_dot(d_row[b], act + b * 256, threadIdx.x, BLOCK);
        acc = block_sum<BLOCK>(acc, shm);
        __syncthreads();
        if (threadIdx.x == 0) total += w[k] * acc;
        __syncthreads();
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

template <int BLOCK>
__global__ void mla_decode_attn_kernel(float* __restrict__ out,
                                       const float* __restrict__ q,
                                       const float* __restrict__ k_cache,
                                       const float* __restrict__ wv_b,
                                       int key_length, int kv_lora, int v_dim,
                                       int n_ctx, float scale) {
    const int h = blockIdx.x;
    const float* qh = q + (size_t)h * key_length;
    const float* wh = wv_b + (size_t)h * (size_t)kv_lora * v_dim;

    extern __shared__ float smem[];
    float* scores = smem;                     // n_ctx
    float* latent = scores + n_ctx;           // kv_lora
    float* red    = latent + kv_lora;         // BLOCK/32 + 1

    for (int t = threadIdx.x; t < n_ctx; t += BLOCK) {
        const float* kt = k_cache + (size_t)t * key_length;
        float s = 0.0f;
#pragma unroll 4
        for (int d = 0; d < key_length; ++d) s += qh[d] * kt[d];
        scores[t] = s * scale;
    }
    __syncthreads();

    float mx = -1e30f;
    for (int t = threadIdx.x; t < n_ctx; t += BLOCK)
        mx = fmaxf(mx, scores[t]);
    mx = block_max<BLOCK>(mx, red);

    float sum = 0.0f;
    for (int t = threadIdx.x; t < n_ctx; t += BLOCK) {
        const float e = __expf(scores[t] - mx);
        scores[t] = e;
        sum += e;
    }
    sum = block_sum<BLOCK>(sum, red);
    const float inv = 1.0f / sum;
    for (int t = threadIdx.x; t < n_ctx; t += BLOCK)
        scores[t] *= inv;
    __syncthreads();

    // latent[r] = sum_t p[t] * k_cache[t, r]   (V = leading kv_lora of K)
    for (int r = threadIdx.x; r < kv_lora; r += BLOCK) {
        float acc = 0.0f;
        for (int t = 0; t < n_ctx; ++t)
            acc += scores[t] * k_cache[(size_t)t * key_length + r];
        latent[r] = acc;
    }
    __syncthreads();

    // out[v] = sum_r wv_b[r, v, h] * latent[r]
    float* oh = out + (size_t)h * v_dim;
    for (int v = threadIdx.x; v < v_dim; v += BLOCK) {
        const float* wr = wh + (size_t)v * kv_lora;
        float acc = 0.0f;
#pragma unroll 4
        for (int r = 0; r < kv_lora; ++r) acc += wr[r] * latent[r];
        oh[v] = acc;
    }
}

// --- generic f32-activation GEMV --------------------------------------------
// One CUDA thread block per output row n; threads stride over K.

struct BlockQ8_0 {   // 34 bytes, matches ggml block_q8_0 exactly
    uint16_t d;       // fp16 scale
    int8_t   qs[32];
};

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
#pragma unroll
        for (int j = 0; j < 32; ++j) s += (float)row[b].qs[j] * xb[j];
        acc += d * s;
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
    const size_t shm = (size_t)3 * head_dim * sizeof(float);
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

// Upload the lattice/sign tables once. __constant__ is the right home: every
// thread in a warp reads the same grid entry for a given codepoint, so the
// broadcast path is what we want rather than L1 thrash.
static void ensure_iq2xs_tables() {
    static bool ready = false;
    if (ready) return;
    cudaMemcpyToSymbol(c_iq2xs_grid,   h_iq2xs_grid,   sizeof(h_iq2xs_grid));
    cudaMemcpyToSymbol(c_ksigns_iq2xs, h_ksigns_iq2xs, sizeof(h_ksigns_iq2xs));
    cudaMemcpyToSymbol(c_kmask_iq2xs,  h_kmask_iq2xs,  sizeof(h_kmask_iq2xs));
    ready = true;
}

static void ensure_iq1s_tables() {
    static bool ready = false;
    if (ready) return;
    cudaMemcpyToSymbol(iq1s_grid_c, iq1s_grid_host, sizeof(iq1s_grid_host));
    ready = true;
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

void moe_expert_ffn_iq2xs_f32(float* out, float* scratch,
                              const float* x, const int* ids, const float* w,
                              const void* gate_exps, const void* up_exps,
                              const void* down_exps,
                              int latent, int ffn, int top_k,
                              float situ_beta, float situ_linear_beta,
                              cudaStream_t stream) {
    if (latent <= 0 || ffn <= 0 || top_k <= 0) return;
    if (latent % 256 || ffn % 256) {
        fprintf(stderr, "[k3] moe_expert_ffn: latent=%d ffn=%d must be multiples of 256\n",
                latent, ffn);
        return;
    }
    ensure_iq2xs_tables();
    const int lb_active = situ_linear_beta > 0.0f ? 1 : 0;
    dim3 g1((unsigned)ffn, (unsigned)top_k);
    moe_gate_up_situ_kernel<128, BlockIQ2XS><<<g1, 128, 0, stream>>>(
        scratch, x, ids, (const BlockIQ2XS*)gate_exps, (const BlockIQ2XS*)up_exps,
        latent, ffn, situ_beta, 1.0f / situ_beta, situ_linear_beta,
        lb_active ? 1.0f / situ_linear_beta : 1.0f, lb_active);
    moe_down_combine_kernel<128, BlockIQ2XS><<<(unsigned)latent, 128, 0, stream>>>(
        out, scratch, ids, w, (const BlockIQ2XS*)down_exps, latent, ffn, top_k);
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
                             cudaStream_t stream) {
    ensure_iq1s_tables();
    const int lb_active = situ_linear_beta > 0.0f ? 1 : 0;
    dim3 g1((unsigned)ffn, (unsigned)top_k);
    moe_gate_up_situ_kernel<128, BlockIQ1S><<<g1, 128, 0, stream>>>(
        scratch, x, ids, (const BlockIQ1S*)gate_exps, (const BlockIQ1S*)up_exps,
        latent, ffn, situ_beta, 1.0f / situ_beta, situ_linear_beta,
        lb_active ? 1.0f / situ_linear_beta : 1.0f, lb_active);
    moe_down_combine_kernel<128, BlockIQ1S><<<(unsigned)latent, 128, 0, stream>>>(
        out, scratch, ids, w, (const BlockIQ1S*)down_exps, latent, ffn, top_k);
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
                                int ggml_type, cudaStream_t stream) {
    switch (ggml_type) {
        case 17:
            moe_expert_ffn_iq2xs_f32(out, scratch, x, ids, w, gate_exps, up_exps,
                                     down_exps, latent, ffn, top_k, situ_beta,
                                     situ_linear_beta, stream);
            return true;
        case 19:
            moe_expert_ffn_iq1s_f32(out, scratch, x, ids, w, gate_exps, up_exps,
                                    down_exps, latent, ffn, top_k, situ_beta,
                                    situ_linear_beta, stream);
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
    const size_t shm = ((size_t)n_ctx + (size_t)kv_lora + BLOCK / 32 + 1) * sizeof(float);
    mla_decode_attn_kernel<BLOCK><<<(unsigned)n_head, BLOCK, shm, stream>>>(
        out, q, k_cache, wv_b, key_length, kv_lora, v_dim, n_ctx, scale);
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
            proj_q8_0_kernel<BLOCK><<<(unsigned)N, BLOCK, 0, stream>>>(
                y, x, (const BlockQ8_0*)W, K / 32);
            return true;
        }
        default:
            return false;
    }
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

}  // namespace k3
}  // namespace kernels
}  // namespace sparkinfer
