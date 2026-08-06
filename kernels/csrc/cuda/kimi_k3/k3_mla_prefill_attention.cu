#include "sparkinfer/kernels/k3_mla_prefill_attention.h"

#include <cuda_fp16.h>
#include <cmath>

namespace sparkinfer::kernels::k3 {
namespace {

constexpr int kBlock = 256;
constexpr int kMaxKey = 576;
constexpr int kMaxLatent = 512;

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
    const int head = blockIdx.y;
    if (token >= tokens || head >= heads) return;

    extern __shared__ float smem[];
    float* q = smem;
    float* acc = q + key_length;
    float* warp_sums = acc + kv_lora;
    float* ml = warp_sums + kBlock / 32;

    const float* q_src = query + ((size_t)token * heads + head) * key_length;
    for (int d = threadIdx.x; d < key_length; d += blockDim.x) q[d] = q_src[d];
    for (int d = threadIdx.x; d < kv_lora; d += blockDim.x) acc[d] = 0.0f;
    if (threadIdx.x == 0) { ml[0] = -CUDART_INF_F; ml[1] = 0.0f; ml[2] = 0.0f; }
    __syncthreads();

    const int causal_end = min(context, start_pos + token + 1);
    for (int pos = 0; pos < causal_end; ++pos) {
        const KV* row = cache + (size_t)pos * key_length;
        float dot = 0.0f;
        for (int d = threadIdx.x; d < key_length; d += blockDim.x)
            dot += q[d] * kv_load(row + d);
        const float score = block_sum(dot, warp_sums) * scale;

        if (threadIdx.x == 0) {
            const float old_m = ml[0];
            const float new_m = fmaxf(old_m, score);
            ml[2] = isfinite(old_m) ? expf(old_m - new_m) : 0.0f;
            ml[1] = ml[1] * ml[2] + expf(score - new_m);
            ml[0] = new_m;
        }
        __syncthreads();
        const float old_scale = ml[2];
        const float weight = expf(score - ml[0]);
        for (int d = threadIdx.x; d < kv_lora; d += blockDim.x)
            acc[d] = acc[d] * old_scale + kv_load(row + d) * weight;
        __syncthreads();
    }

    const size_t out_base = ((size_t)token * heads + head) * value_dim;
    const float inv_l = ml[1] > 0.0f ? 1.0f / ml[1] : 0.0f;
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
    const size_t shmem = (size_t)(key_length + kv_lora + kBlock / 32 + 3) * sizeof(float);
    mla_prefill_attention_kernel<float><<<dim3(tokens, heads), kBlock, shmem, stream>>>(
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
    const size_t shmem = (size_t)(key_length + kv_lora + kBlock / 32 + 3) * sizeof(float);
    mla_prefill_attention_kernel<__half><<<dim3(tokens, heads), kBlock, shmem, stream>>>(
        output, query, (const __half*)kv_cache, wv_b, gate, start_pos, tokens, context,
        heads, key_length, kv_lora, value_dim, scale);
    return cudaPeekAtLastError() == cudaSuccess;
}

}  // namespace sparkinfer::kernels::k3
