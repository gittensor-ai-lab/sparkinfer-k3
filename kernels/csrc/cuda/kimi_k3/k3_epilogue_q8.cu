// Factor — fold a one-op producer into the Q8_0 quantise its consumer would issue.
//
// ===========================================================================
// THE SHAPE
// ===========================================================================
// The decode graph's most common node is still quantize_q8_0_warp_kernel (557 of
// ~3,449 nodes per rank per token on a recent census). hoist_act already removes the
// re-quantises of the shared attn/ffn norms; what remains are activations that exist
// only to feed ONE projection a few microseconds later:
//
//   kda_gate_out  -> attn_output     69 layers
//   mla_gate_out  -> attn_output     24 layers
//   situ          -> shexp/dense down  1+92 layers
//
// Each pair is an elementwise (or per-head RMS) launch followed by a quantise of the
// vector that launch just wrote. This does both in one kernel and hands the Q8_0
// straight to k3_proj_q8act_f32, so the standalone producer and the standalone
// quantise both leave the graph.
//
// ===========================================================================
// WHY IT IS BIT-IDENTICAL
// ===========================================================================
//   * situ and mla_gate are pure elementwise; the fused path evaluates the same
//     expression the standalone kernels use, then runs the shipped warp quantiser
//     (amax over magnitudes via shfl_xor, d = amax/127, per-element rn) on those
//     values. Order-independent amax, independent qs — same bytes as
//     situ/mla_gate followed by k3_quantize_q8_0.
//   * kda_gate keeps the frozen 128-lane block_sum over head_dim that
//     kda_gate_out_kernel<128> uses; only the apply is then quantised. Idle lanes
//     above head_dim are unreachable at K3's head_dim == 128.
//
// The float buffers (gate_out / dense_situ / mla_attn_out) are still written when
// the caller passes them, so debug tags and any later reader see what the split
// path wrote.
//
// ===========================================================================
// WHAT IT DECLINES
// ===========================================================================
// SPARKINFER_K3_EPILOGUE_Q8=0, a missing pointer, n not a multiple of 32, or
// (for the KDA gate) head_dim != 128. The caller keeps the standalone pair.

#include "sparkinfer/kernels/kimi_k3_fast.h"
#include "k3_pdl.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdlib>

namespace sparkinfer {
namespace kernels {
namespace k3 {
namespace {

struct BlockQ8_0 {
    uint16_t d;
    int8_t   qs[32];
};
static_assert(sizeof(BlockQ8_0) == 34, "bad block_q8_0 layout");

__device__ __forceinline__ float sigmoidf_ep(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// Same tree as k3_kernels.cu — duplicated so this TU stays independent.
template <int BLOCK>
__device__ __forceinline__ float block_sum(float v, float* shm) {
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

__device__ __forceinline__ void quant_warp_lane(BlockQ8_0* __restrict__ out,
                                               int b, int lane, float v) {
    float amax = fabsf(v);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
    const float d  = amax / 127.0f;
    const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
    if (lane == 0) out[b].d = __half_as_ushort(__float2half_rn(d));
    out[b].qs[lane] = (int8_t)__float2int_rn(v * id);
}

__device__ __forceinline__ float situ_one(float g, float u, float beta, float inv_beta,
                                          float lb, float inv_lb, int lb_active) {
    const float a = beta * tanhf(g * inv_beta) * sigmoidf_ep(g);
    const float ub = lb_active ? (lb * tanhf(u * inv_lb)) : u;
    return a * ub;
}

// One warp per 32-value quant block. Each lane evaluates situ for its element, then
// the warp emits one BlockQ8_0 — same shape as quantize_q8_0_warp_kernel.
__global__ void situ_q8_warp_kernel(BlockQ8_0* __restrict__ q8,
                                    float* situ_out,
                                    const float* __restrict__ gate,
                                    const float* __restrict__ up,
                                    int n_blocks, float beta, float inv_beta,
                                    float lb, float inv_lb, int lb_active) {
    k3_pdl_sync();
    const int b = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (b >= n_blocks) return;
    const int lane = threadIdx.x & 31;
    const int i = b * 32 + lane;
    const float v = situ_one(gate[i], up[i], beta, inv_beta, lb, inv_lb, lb_active);
    if (situ_out) situ_out[i] = v;
    quant_warp_lane(q8, b, lane, v);
}

__global__ void mla_gate_q8_warp_kernel(BlockQ8_0* __restrict__ q8,
                                        float* gated_out,
                                        const float* __restrict__ attn_out,
                                        const float* __restrict__ gate_proj,
                                        int n_blocks) {
    k3_pdl_sync();
    const int b = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (b >= n_blocks) return;
    const int lane = threadIdx.x & 31;
    const int i = b * 32 + lane;
    const float v = attn_out[i] * sigmoidf_ep(gate_proj[i]);
    if (gated_out) gated_out[i] = v;
    quant_warp_lane(q8, b, lane, v);
}

// One block per head, 128 threads. RMS matches kda_gate_out_kernel<128>; each of the
// four warps then emits one Q8_0 block covering its 32 lanes of the head.
template <int BLOCK>
__global__ void kda_gate_q8_kernel(BlockQ8_0* __restrict__ q8,
                                   float* gate_out,
                                   const float* __restrict__ o,
                                   const float* __restrict__ norm_w,
                                   const float* __restrict__ g2,
                                   int head_dim, float eps) {
    k3_pdl_sync();
    const int h = blockIdx.x;
    const float* oh = o + (size_t)h * head_dim;
    const float* gh = g2 + (size_t)h * head_dim;
    float* dst = gate_out ? gate_out + (size_t)h * head_dim : nullptr;

    __shared__ float shm[BLOCK / 32 + 1];

    float acc = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += BLOCK) {
        const float x = oh[d];
        acc += x * x;
    }
    const float ss = block_sum<BLOCK>(acc, shm);
    const float inv = rsqrtf(ss / (float)head_dim + eps);

    // head_dim == BLOCK == 128: one element per thread, four warps = four Q8 blocks.
    const int d = threadIdx.x;
    float v = 0.0f;
    if (d < head_dim)
        v = (oh[d] * inv * norm_w[d]) * sigmoidf_ep(gh[d]);
    if (dst && d < head_dim) dst[d] = v;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int b = h * (head_dim / 32) + warp;
    if (d < head_dim) quant_warp_lane(q8, b, lane, v);
}

bool epilogue_q8_on() {
    static const bool on = [] {
        const char* e = std::getenv("SPARKINFER_K3_EPILOGUE_Q8");
        return !(e && e[0] == '0');
    }();
    return on;
}

}  // namespace

bool k3_situ_q8(void* q8_out, float* situ_out, const float* gate, const float* up,
                int64_t n, float beta, float linear_beta, cudaStream_t stream) {
    if (!epilogue_q8_on()) return false;
    if (!q8_out || !gate || !up || n <= 0 || n % 32 != 0) return false;
    const int n_blocks = (int)(n / 32);
    const int lb_active = linear_beta > 0.0f ? 1 : 0;
    const float inv_beta = 1.0f / beta;
    const float inv_lb = lb_active ? 1.0f / linear_beta : 1.0f;
    constexpr int threads = 256;
    const int blocks = (n_blocks * 32 + threads - 1) / threads;
    k3_pdl_launch(dim3(blocks), dim3(threads), 0, stream, situ_q8_warp_kernel,
                  (BlockQ8_0*)q8_out, situ_out, gate, up, n_blocks, beta, inv_beta,
                  linear_beta, inv_lb, lb_active);
    return true;
}

bool k3_mla_gate_q8(void* q8_out, float* gated_out, const float* attn_out,
                    const float* gate_proj, int64_t n, cudaStream_t stream) {
    if (!epilogue_q8_on()) return false;
    if (!q8_out || !attn_out || !gate_proj || n <= 0 || n % 32 != 0) return false;
    const int n_blocks = (int)(n / 32);
    constexpr int threads = 256;
    const int blocks = (n_blocks * 32 + threads - 1) / threads;
    k3_pdl_launch(dim3(blocks), dim3(threads), 0, stream, mla_gate_q8_warp_kernel,
                  (BlockQ8_0*)q8_out, gated_out, attn_out, gate_proj, n_blocks);
    return true;
}

bool k3_kda_gate_q8(void* q8_out, float* gate_out, const float* o,
                    const float* norm_w, const float* g2, int head_dim, int n_head,
                    float eps, cudaStream_t stream) {
    if (!epilogue_q8_on()) return false;
    if (!q8_out || !o || !norm_w || !g2 || head_dim != 128 || n_head <= 0)
        return false;
    k3_pdl_launch((unsigned)n_head, 128, 0, stream, kda_gate_q8_kernel<128>,
                  (BlockQ8_0*)q8_out, gate_out, o, norm_w, g2, head_dim, eps);
    return true;
}

}  // namespace k3
}  // namespace kernels
}  // namespace sparkinfer
