// The KDA layer's elementwise tails, folded into the kernels that consume them.
//
// ---------------------------------------------------------------------------
// TWO LAUNCHES PER KDA LAYER THAT EXIST ONLY TO WRITE A VALUE SOMEBODY ELSE READS
// ---------------------------------------------------------------------------
// Between the ssm_f_b projection and the decode step, a KDA layer issues:
//
//     k3_add_f32(g_raw, g_raw, ssm_dt.bias, qkv)     6 blocks of 256
//     kda_decay_gate_f32(decay_g, g_raw, ssm_a, ..)  12 blocks of 128
//     sigmoid_inplace_f32(beta_out, n_head)          1 block, 96 useful lanes
//
// and 69 layers of that is 207 launches per token per rank whose entire content is one
// arithmetic operation per element. On this path a launch is ~1.5-1.9 us however small
// it is — the profile has `kda_conv_step` moving 110 KB in 1.9 us, i.e. 23 ns of actual
// bandwidth work — so the cost is the launch, not the arithmetic.
//
// Both folds are into a CONSUMER that already reads the same memory, so nothing is
// recomputed and no launch is added:
//
//   dt_bias   the decay gate reads g_raw anyway, so it can add the bias on the way in.
//             The standalone add wrote a ROUNDED f32 back to g_raw, and this must round
//             identically — which it does, because `A[h] * (g_raw + dt)` is a multiply
//             of a sum and nvcc has no fma to contract it into. The summed g_raw is no
//             longer written back; nothing reads it after the gate, and the next layer's
//             ssm_f_b projection overwrites the scratch.
//
//   sigmoid   the decode step reads beta[h] once per head, so it can apply the sigmoid
//             there. `1.0f / (1.0f + __expf(-x))` is copied verbatim from
//             sigmoid_inplace_f32_kernel, which is the same expression sigmoidf_ uses.
//             That fold lives in k3_kda_step_ip.cu, next to the kernel that does it, and
//             is gated on this same toggle — see k3_kda_decode_step_ip's `beta_sigmoid`.
//
// SPARKINFER_K3_KDA_FUSE=0 declines, and the forward issues the add, the gate and the
// sigmoid exactly as main does.

#include "sparkinfer/kernels/kimi_k3_fast.h"
#include "k3_pdl.cuh"

#include <cuda_runtime.h>

#include <cstdlib>

namespace sparkinfer {
namespace kernels {
namespace k3 {

namespace {

__device__ __forceinline__ float sigmoidf_fused(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// Same grid, same block, same arithmetic as kda_decay_gate_kernel — it just reads the
// bias itself instead of being handed a buffer somebody wrote one launch earlier.
__global__ void kda_decay_gate_dt_kernel(float* __restrict__ out,
                                         const float* __restrict__ g_raw,
                                         const float* __restrict__ dt_bias,
                                         const float* __restrict__ A,
                                         int head_dim, float lower_bound) {
    k3_pdl_sync();   // no-op unless launched programmatically
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    if (d >= head_dim) return;
    const size_t i = (size_t)h * head_dim + d;
    const float Ah = A[h];
    // One f32 add, exactly what add_f32_kernel stored — then the gate reads it as the
    // standalone kernel would have.
    const float gr = g_raw[i] + dt_bias[i];
    // g = lb * sigmoid(-(A * g_raw))
    out[i] = lower_bound * sigmoidf_fused(-(Ah * gr));
}

}  // namespace

bool k3_kda_fuse_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("SPARKINFER_K3_KDA_FUSE");
        return !(e && e[0] == '0');
    }();
    return on;
}

bool k3_kda_decay_gate_dt(float* out, const float* g_raw, const float* dt_bias,
                          const float* A, int head_dim, int n_head, float lower_bound,
                          cudaStream_t stream) {
    if (!k3_kda_fuse_enabled()) return false;
    if (head_dim <= 0 || n_head <= 0 || !dt_bias) return false;
    // kda_decay_gate_f32 launches head_dim threads per head and guards d >= head_dim;
    // above the hardware block limit there is no equivalent launch, so decline.
    if (head_dim > 1024) return false;
    k3_pdl_launch(dim3((unsigned)n_head), dim3((unsigned)head_dim), 0, stream,
                  kda_decay_gate_dt_kernel, out, g_raw, dt_bias, A, head_dim,
                  lower_bound);
    return true;
}

}  // namespace k3
}  // namespace kernels
}  // namespace sparkinfer
