// Factor — pre-FFN residual combine + RMS, in one launch.
//
// ===========================================================================
// THE SHAPE
// ===========================================================================
// Every layer's FfnPartial opens with the same three host-issued steps when there
// is no residual mix (res_bs==0):
//
//     k3_add_f32(hidden_out, hidden_in, attn_out)   // or memcpy when banked
//     cudaMemcpyAsync(mixed2, hidden_out, ...)      // identity mix
//     rms_norm_f32(normed2, mixed2, ffn_norm, ...)
//
// That is three launches (or two memcpy + one kernel) over the same 28 KiB vector
// before any FFN work begins. At 92 MoE layers the host pays them 92 times per
// token per rank for arithmetic that is already bandwidth-bound inside a single
// SM. This collapses the productive path into one launch: write the residual that
// later layers need as hidden_out, then RMS it into normed2.
//
// ===========================================================================
// WHY IT IS BIT-IDENTICAL
// ===========================================================================
// The residual write is the same f32 add (or copy) the separate path stores.
// The RMS reduction freezes the same partition rms_norm_wide_kernel uses: only
// the first 128 threads accumulate with stride 128, idle lanes above 128
// contribute 0 to block_sum, and the apply widens with the same
// rms_norm_block_for selection (and the same float4 path when every pointer is
// 16-byte aligned and n % 4 == 0). Reading residual_out for the apply matches
// rms_norm reading mixed2 after the identity memcpy.
//
// ===========================================================================
// WHAT IT DECLINES
// ===========================================================================
// SPARKINFER_K3_FFN_PREP=0, a missing pointer, or n <= 0. The caller keeps the
// separate add / memcpy / rms path — same binary A/B as every other factor here.

#include "sparkinfer/kernels/kimi_k3.h"
#include "k3_pdl.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>

namespace sparkinfer {
namespace kernels {
namespace k3 {
namespace {

// Same tree as k3_kernels.cu — duplicated so this TU stays independent of that
// anonymous namespace. Bit-identical because the shuffle/shared schedule matches.
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

// do_add: residual = hidden_in + attn_out; else residual = attn_out (cur).
// Sum of squares stays on 128 threads / stride 128 — see rms_norm_wide_kernel.
template <int BLOCK>
__global__ void ffn_prep_kernel(float* __restrict__ normed_out,
                                float* residual_out,
                                const float* hidden_in,
                                const float* __restrict__ attn_out,
                                const float* __restrict__ rms_w,
                                int n, float eps, int do_add, int n4, int vec4) {
    k3_pdl_sync();
    __shared__ float shm[BLOCK / 32 + 1];

    float acc = 0.0f;
    if (threadIdx.x < 128) {
        for (int d = (int)threadIdx.x; d < n; d += 128) {
            const float v = do_add ? (hidden_in[d] + attn_out[d]) : attn_out[d];
            // Skip the store when residual already is cur (RMS-only call site).
            if (residual_out != attn_out || do_add) residual_out[d] = v;
            acc += v * v;
        }
    }
    const float ss = block_sum<BLOCK>(acc, shm);
    const float inv = rsqrtf(ss / (float)n + eps);

    if (vec4) {
        const float4* __restrict__ x4 = (const float4*)residual_out;
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
            normed_out[d] = residual_out[d] * inv * rms_w[d];
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

bool k3_ffn_prep_f32(float* normed_out, float* residual_out,
                     const float* hidden_in, const float* attn_out,
                     const float* rms_w, int n, float eps, bool do_add,
                     cudaStream_t stream) {
    static const bool want = [] {
        const char* e = std::getenv("SPARKINFER_K3_FFN_PREP");
        return !(e && e[0] == '0');
    }();
    if (!want) return false;
    if (!normed_out || !residual_out || !attn_out || !rms_w || n <= 0) return false;
    if (do_add && !hidden_in) return false;

    // Match rms_norm_f32's block / vec4 selection so the reduction tree and the
    // apply schedule are the same specialization the separate path would pick.
    const bool vec4 = (n % 4 == 0 && aligned16(normed_out) && aligned16(residual_out) &&
                       aligned16(rms_w));
    const int units = vec4 ? n / 4 : n;
    const int block = rms_norm_block_for(units);
    const int n4 = vec4 ? n / 4 : 0;
    const int add_i = do_add ? 1 : 0;

    switch (block) {
        case 1024:
            k3_pdl_launch(1, 1024, 0, stream, ffn_prep_kernel<1024>,
                          normed_out, residual_out, hidden_in, attn_out, rms_w,
                          n, eps, add_i, n4, vec4 ? 1 : 0);
            return true;
        case 512:
            k3_pdl_launch(1, 512, 0, stream, ffn_prep_kernel<512>,
                          normed_out, residual_out, hidden_in, attn_out, rms_w,
                          n, eps, add_i, n4, vec4 ? 1 : 0);
            return true;
        case 256:
            k3_pdl_launch(1, 256, 0, stream, ffn_prep_kernel<256>,
                          normed_out, residual_out, hidden_in, attn_out, rms_w,
                          n, eps, add_i, n4, vec4 ? 1 : 0);
            return true;
        default:
            k3_pdl_launch(1, 128, 0, stream, ffn_prep_kernel<128>,
                          normed_out, residual_out, hidden_in, attn_out, rms_w,
                          n, eps, add_i, n4, vec4 ? 1 : 0);
            return true;
    }
}

}  // namespace k3
}  // namespace kernels
}  // namespace sparkinfer
