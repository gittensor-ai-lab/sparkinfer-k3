// Factor — pre-attention identity mix + RMS, in one launch.
//
// ===========================================================================
// THE SHAPE
// ===========================================================================
// When there is no residual mix (res_bs==0 / n_ckpt stays 0), every layer's
// Attn phase opens with the same two host-issued steps:
//
//     cudaMemcpyAsync(mixed, hidden_in, ...)     // identity "mix"
//     rms_norm_f32(normed, mixed, attn_norm, ...)
//
// That is a memcpy plus a kernel over the same 28 KiB vector before any
// attention work begins — 93 times per token per rank. The productive path is
// a single launch: RMS hidden_in straight into normed, and optionally mirror
// the identity store into mixed so debug / any later reader of mixed still
// sees what the separate path wrote.
//
// ===========================================================================
// WHY IT IS BIT-IDENTICAL
// ===========================================================================
// RMS of hidden_in equals RMS of mixed after the identity memcpy. The
// reduction freezes the same partition rms_norm_wide_kernel uses: only the
// first 128 threads accumulate with stride 128, idle lanes above 128
// contribute 0 to block_sum, and the apply widens with the same
// rms_norm_block_for selection (and the same float4 path when every pointer is
// 16-byte aligned and n % 4 == 0). Writing mixed is a plain store of the same
// values the memcpy would have left.
//
// ===========================================================================
// WHAT IT DECLINES
// ===========================================================================
// SPARKINFER_K3_ATTN_PREP=0, a missing required pointer, or n <= 0. The caller
// keeps the separate memcpy / rms path — same binary A/B as k3_ffn_prep_f32.
// Residual-mix layers (res_bs > 0) never call this; they need attn_res_mix_f32.

#include "sparkinfer/kernels/kimi_k3.h"
#include "k3_pdl.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>

namespace sparkinfer {
namespace kernels {
namespace k3 {
namespace {

// Same tree as k3_ffn_prep.cu / k3_kernels.cu — duplicated so this TU stays
// independent. Bit-identical because the shuffle/shared schedule matches.
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

// Sum of squares on 128 threads / stride 128 — see rms_norm_wide_kernel.
// write_mixed: store the identity mix into mixed_out so callers that still
// read it (debug tags, anything that expected the memcpy) stay correct.
template <int BLOCK>
__global__ void attn_prep_kernel(float* __restrict__ normed_out,
                                 float* mixed_out,
                                 const float* __restrict__ hidden_in,
                                 const float* __restrict__ rms_w,
                                 int n, float eps, int write_mixed,
                                 int n4, int vec4) {
    k3_pdl_sync();
    __shared__ float shm[BLOCK / 32 + 1];

    float acc = 0.0f;
    if (threadIdx.x < 128) {
        for (int d = (int)threadIdx.x; d < n; d += 128) {
            const float v = hidden_in[d];
            if (write_mixed) mixed_out[d] = v;
            acc += v * v;
        }
    }
    const float ss = block_sum<BLOCK>(acc, shm);
    const float inv = rsqrtf(ss / (float)n + eps);

    // Apply from hidden_in (not mixed_out): when write_mixed is false there is
    // no mixed buffer, and when it is true the values are identical anyway —
    // reading the source avoids a RAW hazard inside the same kernel.
    if (vec4) {
        const float4* __restrict__ x4 = (const float4*)hidden_in;
        const float4* __restrict__ w4 = (const float4*)rms_w;
        float4* __restrict__ o4 = (float4*)normed_out;
        for (int d = (int)threadIdx.x; d < n4; d += BLOCK) {
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
        for (int d = (int)threadIdx.x; d < n; d += BLOCK)
            normed_out[d] = hidden_in[d] * inv * rms_w[d];
    }
}

static inline int rms_norm_block_for(int units) {
    if (units >= 1024) return 1024;
    if (units >= 512)  return 512;
    if (units >= 256)  return 256;
    return 128;
}

static inline bool aligned16(const void* p) {
    return ((uintptr_t)p & 15u) == 0;
}

}  // namespace

bool k3_attn_prep_f32(float* normed_out, float* mixed_out,
                      const float* hidden_in, const float* rms_w,
                      int n, float eps, cudaStream_t stream) {
    static const bool want = [] {
        const char* e = std::getenv("SPARKINFER_K3_ATTN_PREP");
        return !(e && e[0] == '0');
    }();
    if (!want) return false;
    if (!normed_out || !hidden_in || !rms_w || n <= 0) return false;

    // Match rms_norm_f32's block / vec4 selection so the reduction tree and the
    // apply schedule are the same specialization the separate path would pick.
    // mixed_out alignment only matters when we write it; null skips the store.
    const int write_mixed = (mixed_out != nullptr) ? 1 : 0;
    const bool vec4 = (n % 4 == 0 && aligned16(normed_out) && aligned16(hidden_in) &&
                       aligned16(rms_w) &&
                       (!write_mixed || aligned16(mixed_out)));
    const int units = vec4 ? n / 4 : n;
    const int block = rms_norm_block_for(units);
    const int n4 = vec4 ? n / 4 : 0;

    switch (block) {
        case 1024:
            k3_pdl_launch(1, 1024, 0, stream, attn_prep_kernel<1024>,
                          normed_out, mixed_out, hidden_in, rms_w,
                          n, eps, write_mixed, n4, vec4 ? 1 : 0);
            return true;
        case 512:
            k3_pdl_launch(1, 512, 0, stream, attn_prep_kernel<512>,
                          normed_out, mixed_out, hidden_in, rms_w,
                          n, eps, write_mixed, n4, vec4 ? 1 : 0);
            return true;
        case 256:
            k3_pdl_launch(1, 256, 0, stream, attn_prep_kernel<256>,
                          normed_out, mixed_out, hidden_in, rms_w,
                          n, eps, write_mixed, n4, vec4 ? 1 : 0);
            return true;
        default:
            k3_pdl_launch(1, 128, 0, stream, attn_prep_kernel<128>,
                          normed_out, mixed_out, hidden_in, rms_w,
                          n, eps, write_mixed, n4, vec4 ? 1 : 0);
            return true;
    }
}

}  // namespace k3
}  // namespace kernels
}  // namespace sparkinfer
