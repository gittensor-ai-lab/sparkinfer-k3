// Token-batched residual add and RMS norm — see sparkinfer/kernels/k3_batch_elem.h.
//
// The add is elementwise, so a token axis on blockIdx.y cannot change a bit.
//
// The norm is the delicate one. It reproduces rms_norm_wide_kernel exactly: the reduction
// runs on threads < 128 only with the same optional 8-way unroll, block_sum is the same
// butterfly and the same fold over increasing warp index, idle threads contribute 0.0f, and
// every CTA recomputes the identical reduction so inv agrees across the grid. blockIdx.y
// only selects which token's slice is reduced and applied. Do not widen the reduction —
// #115 measured +2.8% from doing so and it came with an accuracy regression.

#include "sparkinfer/kernels/k3_batch_elem.h"
#include "k3_pdl.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>

namespace sparkinfer {
namespace kernels {
namespace k3 {

namespace {

// Copied from k3_kernels.cu rather than shared, for the reason that file gives for its
// own duplications: the two arms have to be comparable on one binary, and a shared
// definition someone later "improves" would move both sides of the A/B at once.
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

// out[t][i] = a[t][i] + b[t][i]. blockIdx.y is the token.
//
// Strided rather than packed so the caller can hand this the same [B][n] scratch slabs
// kimi_k3_forward_alloc_scratch_n already allocates, with no gather and no copy.
__global__ void add_batch_kernel(float* __restrict__ out, const float* __restrict__ a,
                                 const float* __restrict__ b, int n,
                                 long so, long sa, long sb) {
    k3_pdl_sync();
    const long t = blockIdx.y;
    const int  i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[t * so + i] = a[t * sa + i] + b[t * sb + i];
}

// The token-batched twin of rms_norm_wide_kernel. blockIdx.x slices the apply exactly as
// it does there; blockIdx.y selects the token.
template <int BLOCK, bool UNROLL>
__global__ void rms_norm_batch_kernel(float* __restrict__ out, const float* __restrict__ x,
                                      const float* __restrict__ w, int n, float eps,
                                      int n4, int vec4, int span_units,
                                      long so, long sx) {
    k3_pdl_sync();
    __shared__ float shm[BLOCK / 32 + 1];

    const long t = blockIdx.y;
    const float* __restrict__ xt = x + t * sx;
    float* __restrict__ ot = out + t * so;

    float acc = 0.0f;
    if (threadIdx.x < 128) {
        int d = (int)threadIdx.x;
        if (UNROLL) {
            for (; d + 7 * 128 < n; d += 8 * 128) {
                float v[8];
#pragma unroll
                for (int u = 0; u < 8; ++u) v[u] = xt[d + u * 128];
#pragma unroll
                for (int u = 0; u < 8; ++u) acc += v[u] * v[u];
            }
        }
        for (; d < n; d += 128) acc += xt[d] * xt[d];
    }
    const float ss = block_sum<BLOCK>(acc, shm);
    const float inv = rsqrtf(ss / (float)n + eps);

    const int units = vec4 ? n4 : n;
    const int u0 = blockIdx.x * span_units;
    const int u1 = min(u0 + span_units, units);
    if (vec4) {
        const float4* __restrict__ x4 = (const float4*)xt;
        const float4* __restrict__ w4 = (const float4*)w;
        float4* __restrict__ o4 = (float4*)ot;
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
            ot[d] = xt[d] * inv * w[d];
    }
}

inline bool aligned16(const void* p) { return ((uintptr_t)p & 15u) == 0; }

}  // namespace

bool k3_batch_elem_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("SPARKINFER_K3_BATCH_ELEM");
        return !(e && e[0] == '0');
    }();
    return on;
}

bool k3_add_f32_batch(float* out, const float* a, const float* b, int n, int n_tok,
                      long stride_out, long stride_a, long stride_b,
                      cudaStream_t stream) {
    if (!k3_batch_elem_enabled()) return false;
    if (!out || !a || !b || n <= 0 || n_tok <= 0) return false;
    constexpr int BLK = 256;
    const dim3 grid((unsigned)((n + BLK - 1) / BLK), (unsigned)n_tok);
    k3_pdl_launch(grid, dim3(BLK), 0, stream, add_batch_kernel,
                  out, a, b, n, stride_out, stride_a, stride_b);
    return true;
}

bool k3_rms_norm_f32_batch(float* out, const float* x, const float* w, int n, float eps,
                           int n_tok, long stride_out, long stride_x,
                           cudaStream_t stream) {
    if (!k3_batch_elem_enabled()) return false;
    if (!out || !x || !w || n <= 0 || n_tok <= 0) return false;

    // IN-PLACE IS REFUSED, NOT HANDLED. With the apply spread over several CTAs, CTA 0
    // can finish its reduction and begin overwriting x while another CTA is still
    // reading x to compute the same sum. The shipped launcher takes the same position
    // (`out == x` is the one case it does not spread); here it is simpler and safer to
    // decline and let the caller run the per-token path, because a batched in-place norm
    // would be wrong only sometimes and only under a race.
    if (out == x) return false;

    const bool vec4 = (n % 4 == 0) && aligned16(out) && aligned16(x) && aligned16(w) &&
                      (stride_out % 4 == 0) && (stride_x % 4 == 0);
    const int units = vec4 ? n / 4 : n;
    const int n4 = vec4 ? n / 4 : 0;

    // Mirror the shipped RMSG default so a batched norm covers the machine the same way
    // the per-token one does, then divide the slice count by the token axis: the grid is
    // already n_tok times taller, and 14 slices x B tokens would overshoot a wave for no
    // gain. Slices stay warp-aligned for the same reason the shipped launcher does it.
    static const int rmsg = [] {
        const char* e = std::getenv("SPARKINFER_K3_RMSG");
        return e ? atoi(e) : 14;
    }();
    static const bool rmsu = [] {
        const char* e = std::getenv("SPARKINFER_K3_RMSU");
        return !(e && e[0] == '0');
    }();
    int slices = rmsg / (n_tok < rmsg ? n_tok : rmsg);
    if (slices < 1) slices = 1;
    int span_units = units;
    unsigned gx = 1u;
    if (slices > 1) {
        span_units = ((units + slices - 1) / slices + 31) / 32 * 32;
        gx = (unsigned)((units + span_units - 1) / span_units);
    }
    const dim3 grid(gx, (unsigned)n_tok);

    // BLOCK 128 always. The reduction is frozen at 128 threads whatever the launch
    // width, so a wider block would only widen the apply — and the token axis has
    // already multiplied the resident work by B, which is what the width was buying.
    if (rmsu)
        k3_pdl_launch(grid, dim3(128), 0, stream, rms_norm_batch_kernel<128, true>,
                      out, x, w, n, eps, n4, vec4 ? 1 : 0, span_units, stride_out, stride_x);
    else
        k3_pdl_launch(grid, dim3(128), 0, stream, rms_norm_batch_kernel<128, false>,
                      out, x, w, n, eps, n4, vec4 ? 1 : 0, span_units, stride_out, stride_x);
    return true;
}

}  // namespace k3
}  // namespace kernels
}  // namespace sparkinfer
