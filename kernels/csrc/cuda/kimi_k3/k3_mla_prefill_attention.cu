#include "sparkinfer/kernels/k3_mla_prefill_attention.h"

#include <cuda_fp16.h>
#include <cmath>

namespace sparkinfer::kernels::k3 {
namespace {

constexpr int kBlock = 256;
constexpr int kHeadsPerBlock = 4;
constexpr int kSplitTokensPerBlock = 2;
constexpr int kSplitHeadsPerBlock = 4;
constexpr int kSplitPairs = kSplitTokensPerBlock * kSplitHeadsPerBlock;
constexpr int kMaxKey = 576;
constexpr int kMaxLatent = 512;
constexpr int kMaxSplits = 32;

__device__ __forceinline__ float block_sum(float x, float* warp_sums) {
    for (int d = 16; d; d >>= 1) x += __shfl_down_sync(0xffffffffu, x, d);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = x;
    __syncthreads();
    x = threadIdx.x < blockDim.x / 32 ? warp_sums[lane] : 0.0f;
    if (warp == 0)
        for (int d = 16; d; d >>= 1) x += __shfl_down_sync(0xffffffffu, x, d);
    if (threadIdx.x == 0) warp_sums[0] = x;
    __syncthreads();
    return warp_sums[0];
}

template <typename KV>
__device__ __forceinline__ float kv_load(const KV* p) { return (float)*p; }

template <>
__device__ __forceinline__ float kv_load<__half>(const __half* p) { return __half2float(*p); }

template <typename KV>
__global__ void mla_prefill_attention_kernel(
    float* __restrict__ output, const float* __restrict__ query,
    const KV* __restrict__ cache, const float* __restrict__ wv_b,
    const float* __restrict__ gate, int start_pos, int tokens, int context,
    int heads, int key_length, int kv_lora, int value_dim, float scale) {
    const int token = blockIdx.x;
    const int head0 = blockIdx.y * kHeadsPerBlock;
    const int live_heads = min(kHeadsPerBlock, heads - head0);
    if (token >= tokens || live_heads <= 0) return;

    extern __shared__ float smem[];
    float* row_s = smem;
    float* q = row_s + key_length;
    float* acc = q + kHeadsPerBlock * key_length;
    float* warp_sums = acc + kHeadsPerBlock * kv_lora;
    float* ml = warp_sums + kBlock / 32;

    for (int i = threadIdx.x; i < live_heads * key_length; i += blockDim.x) {
        const int h = i / key_length;
        const int d = i - h * key_length;
        q[i] = query[((size_t)token * heads + head0 + h) * key_length + d];
    }
    for (int i = threadIdx.x; i < live_heads * kv_lora; i += blockDim.x) acc[i] = 0.0f;
    if (threadIdx.x < live_heads) {
        ml[threadIdx.x * 3] = -CUDART_INF_F;
        ml[threadIdx.x * 3 + 1] = 0.0f;
        ml[threadIdx.x * 3 + 2] = 0.0f;
    }
    __syncthreads();

    const int causal_end = min(context, start_pos + token + 1);
    for (int pos = 0; pos < causal_end; ++pos) {
        const KV* row = cache + (size_t)pos * key_length;
        for (int d = threadIdx.x; d < key_length; d += blockDim.x)
            row_s[d] = kv_load(row + d);
        __syncthreads();

        // All heads in the band reuse the one staged MQA row.  The previous
        // one-head grid fetched the same 576-value row once per head; K3 has 12
        // local heads, so this removes 75% of that global traffic at HPB=4.
        for (int h = 0; h < live_heads; ++h) {
            float dot = 0.0f;
            for (int d = threadIdx.x; d < key_length; d += blockDim.x)
                dot += q[h * key_length + d] * row_s[d];
            const float score = block_sum(dot, warp_sums) * scale;
            if (threadIdx.x == 0) {
                float* state = ml + h * 3;
                const float old_m = state[0];
                const float new_m = fmaxf(old_m, score);
                state[2] = isfinite(old_m) ? expf(old_m - new_m) : 0.0f;
                state[1] = state[1] * state[2] + expf(score - new_m);
                state[0] = new_m;
            }
            __syncthreads();
            const float old_scale = ml[h * 3 + 2];
            const float weight = expf(score - ml[h * 3]);
            for (int d = threadIdx.x; d < kv_lora; d += blockDim.x) {
                const int i = h * kv_lora + d;
                acc[i] = acc[i] * old_scale + row_s[d] * weight;
            }
            __syncthreads();
        }
    }

    for (int hv = threadIdx.x; hv < live_heads * value_dim; hv += blockDim.x) {
        const int h = hv / value_dim;
        const int v = hv - h * value_dim;
        const int head = head0 + h;
        const size_t out_base = ((size_t)token * heads + head) * value_dim;
        const float inv_l = ml[h * 3 + 1] > 0.0f ? 1.0f / ml[h * 3 + 1] : 0.0f;
        const float* w = wv_b + ((size_t)head * value_dim + v) * kv_lora;
        float y = 0.0f;
        for (int d = 0; d < kv_lora; ++d) y += acc[h * kv_lora + d] * w[d];
        y *= inv_l;
        if (gate) y *= 1.0f / (1.0f + expf(-gate[out_base + v]));
        output[out_base + v] = y;
    }
}

template <typename KV>
__global__ void mla_prefill_attention_partial_kernel(
    float* __restrict__ partial, const float* __restrict__ query,
    const KV* __restrict__ cache, int start_pos, int tokens, int context,
    int heads, int key_length, int kv_lora, int splits, float scale) {
    const int token0 = blockIdx.x * kSplitTokensPerBlock;
    const int head0 = blockIdx.y * kSplitHeadsPerBlock;
    const int split = blockIdx.z;
    const int live_tokens = min(kSplitTokensPerBlock, tokens - token0);
    const int live_heads = min(kSplitHeadsPerBlock, heads - head0);
    if (live_tokens <= 0 || live_heads <= 0 || split >= splits) return;

    extern __shared__ float smem[];
    float* row_s = smem;
    float* q = row_s + key_length;                       // [8][key]
    float* acc = q + kSplitPairs * key_length;           // [8][latent]
    float* state = acc + kSplitPairs * kv_lora;          // [8][m,l,a,b]
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int local_t = warp / kSplitHeadsPerBlock;
    const int local_h = warp - local_t * kSplitHeadsPerBlock;
    const bool live = local_t < live_tokens && local_h < live_heads;
    const int token = token0 + local_t;
    const int head = head0 + local_h;

    if (live) {
        const float* src = query + ((size_t)token * heads + head) * key_length;
        for (int d = lane; d < key_length; d += 32)
            q[warp * key_length + d] = src[d];
        for (int d = lane; d < kv_lora; d += 32)
            acc[warp * kv_lora + d] = 0.0f;
        if (lane == 0) {
            state[warp * 4] = -CUDART_INF_F;
            state[warp * 4 + 1] = 0.0f;
        }
    }
    __syncthreads();

    // The block walks the union of its two queries' split ranges. Each KV row is
    // fetched once and consumed by up to eight (query, head) warps. The old split
    // grid fetched it once per query/head band, so at K3's T=16,H=12 this halves
    // the dominant global cache traffic while retaining 96 context-split blocks.
    const int first_end = min(context, start_pos + token0 + 1);
    const int last_end = min(context, start_pos + token0 + live_tokens);
    const int begin = (int)(((long long)first_end * split) / splits);
    const int end = (int)(((long long)last_end * (split + 1)) / splits);
    for (int pos = begin; pos < end; ++pos) {
        const KV* row = cache + (size_t)pos * key_length;
        for (int d = threadIdx.x; d < key_length; d += blockDim.x)
            row_s[d] = kv_load(row + d);
        __syncthreads();

        const int causal_end = min(context, start_pos + token + 1);
        const int pair_begin = (int)(((long long)causal_end * split) / splits);
        const int pair_end = (int)(((long long)causal_end * (split + 1)) / splits);
        const bool active = live && pos >= pair_begin && pos < pair_end;
        float dot = 0.0f;
        if (active)
            for (int d = lane; d < key_length; d += 32)
                dot += q[warp * key_length + d] * row_s[d];
        for (int d = 16; d; d >>= 1)
            dot += __shfl_down_sync(0xffffffffu, dot, d);
        if (lane == 0) {
            float* st = state + warp * 4;
            if (active) {
                const float score = dot * scale;
                const float nm = fmaxf(st[0], score);
                st[2] = isfinite(st[0]) ? expf(st[0] - nm) : 0.0f;
                st[3] = expf(score - nm);
                st[1] = st[1] * st[2] + st[3];
                st[0] = nm;
            } else {
                st[2] = 1.0f; st[3] = 0.0f;
            }
        }
        __syncthreads();
        if (live)
            for (int d = lane; d < kv_lora; d += 32) {
                const int i = warp * kv_lora + d;
                acc[i] = acc[i] * state[warp * 4 + 2] + row_s[d] * state[warp * 4 + 3];
            }
        __syncthreads();
    }

    const int stride = kv_lora + 2;
    if (live) {
        float* dst = partial + (((size_t)token * heads + head) * splits + split) * stride;
        if (lane == 0) { dst[0] = state[warp * 4]; dst[1] = state[warp * 4 + 1]; }
        for (int d = lane; d < kv_lora; d += 32)
            dst[2 + d] = acc[warp * kv_lora + d];
    }
}

__global__ void mla_prefill_attention_merge_kernel(
    float* __restrict__ output, const float* __restrict__ partial,
    const float* __restrict__ wv_b, const float* __restrict__ gate,
    int tokens, int heads, int kv_lora, int value_dim, int splits) {
    const int token = blockIdx.x, head = blockIdx.y;
    if (token >= tokens || head >= heads) return;
    extern __shared__ float smem[];
    float* acc = smem;
    float* state = acc + kv_lora;  // [global max, global denominator]
    const int stride = kv_lora + 2;
    const float* base = partial + ((size_t)token * heads + head) * splits * stride;
    if (threadIdx.x == 0) {
        float m = -CUDART_INF_F;
        for (int s = 0; s < splits; ++s) m = fmaxf(m, base[(size_t)s * stride]);
        float l = 0.0f;
        for (int s = 0; s < splits; ++s) {
            const float sm = base[(size_t)s * stride];
            if (isfinite(sm)) l += base[(size_t)s * stride + 1] * expf(sm - m);
        }
        state[0] = m; state[1] = l;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < kv_lora; d += blockDim.x) {
        float a = 0.0f;
        for (int s = 0; s < splits; ++s) {
            const float* p = base + (size_t)s * stride;
            if (isfinite(p[0])) a += p[2 + d] * expf(p[0] - state[0]);
        }
        acc[d] = a;
    }
    __syncthreads();
    const size_t out_base = ((size_t)token * heads + head) * value_dim;
    const float inv_l = state[1] > 0.0f ? 1.0f / state[1] : 0.0f;
    for (int v = threadIdx.x; v < value_dim; v += blockDim.x) {
        const float* w = wv_b + ((size_t)head * value_dim + v) * kv_lora;
        float y = 0.0f;
        for (int d = 0; d < kv_lora; ++d) y += acc[d] * w[d];
        y *= inv_l;
        if (gate) y *= 1.0f / (1.0f + expf(-gate[out_base + v]));
        output[out_base + v] = y;
    }
}

}  // namespace

bool k3_mla_prefill_attention_supported(int tokens, int context, int heads,
                                        int key_length, int kv_lora, int value_dim) {
    return tokens > 0 && context > 0 && heads > 0 && value_dim > 0 &&
           kv_lora > 0 && kv_lora <= kMaxLatent &&
           key_length >= kv_lora && key_length <= kMaxKey;
}

bool k3_mla_prefill_attention_f32(float* output, const float* query,
                                  const float* kv_cache, const float* wv_b,
                                  const float* gate, int start_pos, int tokens,
                                  int context, int heads, int key_length,
                                  int kv_lora, int value_dim, float scale,
                                  cudaStream_t stream) {
    if (!output || !query || !kv_cache || !wv_b || start_pos < 0 ||
        start_pos >= context || tokens > context - start_pos ||
        !k3_mla_prefill_attention_supported(tokens, context, heads, key_length,
                                             kv_lora, value_dim))
        return false;
    const size_t shmem = (size_t)(key_length + kHeadsPerBlock * (key_length + kv_lora) +
                                  kBlock / 32 + kHeadsPerBlock * 3) * sizeof(float);
    mla_prefill_attention_kernel<float><<<
        dim3(tokens, (heads + kHeadsPerBlock - 1) / kHeadsPerBlock), kBlock, shmem, stream>>>(
        output, query, kv_cache, wv_b, gate, start_pos, tokens, context, heads,
        key_length, kv_lora, value_dim, scale);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool k3_mla_prefill_attention_kvf16(float* output, const float* query,
                                    const void* kv_cache, const float* wv_b,
                                    const float* gate, int start_pos, int tokens,
                                    int context, int heads, int key_length,
                                    int kv_lora, int value_dim, float scale,
                                    cudaStream_t stream) {
    if (!output || !query || !kv_cache || !wv_b || start_pos < 0 ||
        start_pos >= context || tokens > context - start_pos ||
        !k3_mla_prefill_attention_supported(tokens, context, heads, key_length,
                                             kv_lora, value_dim))
        return false;
    const size_t shmem = (size_t)(key_length + kHeadsPerBlock * (key_length + kv_lora) +
                                  kBlock / 32 + kHeadsPerBlock * 3) * sizeof(float);
    mla_prefill_attention_kernel<__half><<<
        dim3(tokens, (heads + kHeadsPerBlock - 1) / kHeadsPerBlock), kBlock, shmem, stream>>>(
        output, query, (const __half*)kv_cache, wv_b, gate, start_pos, tokens, context,
        heads, key_length, kv_lora, value_dim, scale);
    return cudaPeekAtLastError() == cudaSuccess;
}

size_t k3_mla_prefill_attention_workspace_bytes(int tokens, int heads,
                                                int kv_lora, int splits) {
    if (tokens <= 0 || heads <= 0 || kv_lora <= 0 || splits <= 0 || splits > kMaxSplits)
        return 0;
    return (size_t)tokens * heads * splits * (kv_lora + 2) * sizeof(float);
}

template <typename KV>
static bool launch_split(float* output, void* workspace, size_t workspace_bytes,
                         const float* query, const KV* cache, const float* wv_b,
                         const float* gate, int start_pos, int tokens, int context,
                         int heads, int key_length, int kv_lora, int value_dim,
                         int splits, float scale, cudaStream_t stream) {
    const size_t need = k3_mla_prefill_attention_workspace_bytes(tokens, heads,
                                                                 kv_lora, splits);
    if (!output || !workspace || workspace_bytes < need || !query || !cache || !wv_b ||
        need == 0 || start_pos < 0 || start_pos >= context ||
        tokens > context - start_pos ||
        !k3_mla_prefill_attention_supported(tokens, context, heads, key_length,
                                             kv_lora, value_dim)) return false;
    const size_t pshm = (size_t)(key_length + kSplitPairs * (key_length + kv_lora + 4)) *
                         sizeof(float);
    mla_prefill_attention_partial_kernel<KV><<<
        dim3((tokens + kSplitTokensPerBlock - 1) / kSplitTokensPerBlock,
             (heads + kSplitHeadsPerBlock - 1) / kSplitHeadsPerBlock, splits),
        kBlock, pshm, stream>>>((float*)workspace, query, cache, start_pos, tokens,
                               context, heads, key_length, kv_lora, splits, scale);
    if (cudaPeekAtLastError() != cudaSuccess) return false;
    mla_prefill_attention_merge_kernel<<<dim3(tokens, heads), kBlock,
        (size_t)(kv_lora + 2) * sizeof(float), stream>>>(
        output, (const float*)workspace, wv_b, gate, tokens, heads, kv_lora,
        value_dim, splits);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool k3_mla_prefill_attention_split_f32(
    float* output, void* workspace, size_t workspace_bytes, const float* query,
    const float* kv_cache, const float* wv_b, const float* gate, int start_pos,
    int tokens, int context, int heads, int key_length, int kv_lora,
    int value_dim, int splits, float scale, cudaStream_t stream) {
    return launch_split(output, workspace, workspace_bytes, query, kv_cache, wv_b, gate,
                        start_pos, tokens, context, heads, key_length, kv_lora,
                        value_dim, splits, scale, stream);
}

bool k3_mla_prefill_attention_split_kvf16(
    float* output, void* workspace, size_t workspace_bytes, const float* query,
    const void* kv_cache, const float* wv_b, const float* gate, int start_pos,
    int tokens, int context, int heads, int key_length, int kv_lora,
    int value_dim, int splits, float scale, cudaStream_t stream) {
    return launch_split(output, workspace, workspace_bytes, query,
                        (const __half*)kv_cache, wv_b, gate, start_pos, tokens,
                        context, heads, key_length, kv_lora, value_dim, splits,
                        scale, stream);
}

}  // namespace sparkinfer::kernels::k3
