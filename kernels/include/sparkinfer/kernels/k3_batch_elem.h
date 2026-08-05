// Token-batched residual add and RMS norm for prompt prefill.
//
// Both move ~28 KB per launch and run 617 times per token per rank. That cost is launch and
// memory latency rather than work, so the only thing that removes it is fewer, bigger
// launches: these take a token axis so B prompt tokens cost one launch instead of B.
//
// Strides are in ELEMENTS and address the [B][n] slabs kimi_k3_forward_alloc_scratch_n
// allocates, so a caller passes the same pointers the per-token path uses.
//
// SPARKINFER_K3_BATCH_ELEM=0 declines both; the caller falls back to B separate calls.
#pragma once

#include <cuda_runtime.h>

namespace sparkinfer {
namespace kernels {
namespace k3 {

bool k3_batch_elem_enabled();

// out[t][i] = a[t][i] + b[t][i]. Elementwise, so bit-identity is trivial.
bool k3_add_f32_batch(float* out, const float* a, const float* b, int n, int n_tok,
                      long stride_out, long stride_a, long stride_b,
                      cudaStream_t stream);

// out[t] = rms_norm(x[t]) * w. `w` is shared across tokens — it is the layer weight.
//
// Reproduces rms_norm_wide_kernel exactly, and the reduction must stay frozen at 128
// threads: widening it changes the float32 association of the sum of squares, which moves
// every output element by ~1e-7 relative. See #115.
//
// REFUSES in-place (out == x): with the apply spread over CTAs, one can overwrite x while
// another still reads it for the same sum. Returns false; the caller runs the per-token path.
bool k3_rms_norm_f32_batch(float* out, const float* x, const float* w, int n, float eps,
                           int n_tok, long stride_out, long stride_x,
                           cudaStream_t stream);

}  // namespace k3
}  // namespace kernels
}  // namespace sparkinfer
