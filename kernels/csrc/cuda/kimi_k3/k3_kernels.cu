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

#include <cuda_runtime.h>
#include <math.h>

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
// 5. MLA output gate
// ---------------------------------------------------------------------------

__global__ void mla_gate_out_kernel(float* __restrict__ out,
                                    const float* __restrict__ attn_out,
                                    const float* __restrict__ gate_proj, int64_t n) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = attn_out[i] * sigmoidf_(gate_proj[i]);
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

void mla_gate_out_f32(float* out, const float* attn_out, const float* gate_proj,
                      int64_t n, cudaStream_t stream) {
    if (n <= 0) return;
    const int T = 256;
    const int64_t blocks = (n + T - 1) / T;
    mla_gate_out_kernel<<<(unsigned)blocks, T, 0, stream>>>(out, attn_out, gate_proj, n);
}

}  // namespace k3
}  // namespace kernels
}  // namespace sparkinfer
