// Token-batched Q8_0 projection for prompt prefill.
//
// Prompt ingestion is one forward_token call per prompt token, so every token re-reads the
// whole weight set. This reads a weight ONCE and drives it against NT tokens, dividing both
// the weight traffic and the launch count by NT.
//
// It is proj_q8_multirow_1bar_kernel with the reuse axis transposed: that kernel stages one
// activation block and reuses it across ROWS output rows; this stages ROWS weight blocks and
// reuses each across NT tokens. Same grid width, so no launch is merged and no block
// narrowed.
//
// BIT-IDENTICAL BY CONSTRUCTION. BLOCK is chosen from K by the same block_for() rule, so a
// thread accumulates the same quant blocks in the same order; the dp4a chain, the
// accumulator expression, the butterfly and the single-barrier fold are copied unchanged.
// Only which outputs share a CTA differs.
//
// SPARKINFER_K3_BATCH_PROJ=0 declines; the caller falls back to a per-token loop.
#pragma once

#include <cuda_runtime.h>

namespace sparkinfer {
namespace kernels {
namespace k3 {

bool k3_batch_proj_enabled();

// Largest NT this TU instantiates. Exposed so a caller can size its staging without
// guessing.
int  k3_batch_proj_max_nt();

// Quantise n_tok contiguous [K] activations into n_tok contiguous Q8_0 rows.
//
// `x` is [n_tok][K] f32, DENSE with stride exactly K. The stride is not a parameter on
// purpose: k3_quantize_q8_0 walks quant blocks linearly, and this call being
// indistinguishable from n_tok separate ones is what the bit-identity argument rests on.
void k3_quantize_q8_0_batch(void* out, const float* x, int K, int n_tok,
                            cudaStream_t stream);

// y[t * y_stride + n] = dot(W[n, :], x[t, :]).
//
//   y       [n_tok][y_stride] f32, y_stride >= N — strided so the caller can pass the same
//           per-token buffers the per-token path writes, and A/B the two without a copy.
//   xq      n_tok contiguous Q8_0 rows, from k3_quantize_q8_0_batch. Pre-quantised because
//           a chunk projects one activation through several weights.
//   W       [N][K] Q8_0; wtype must be 8.
//
// Returns false and launches NOTHING when it declines (toggle off, wrong type, K not a
// multiple of 32, n_tok out of range, geometry not instantiated).
bool k3_proj_q8_batch_1bar(float* y, long y_stride,
                           const void* xq, const void* W, int wtype,
                           int N, int K, int n_tok, cudaStream_t stream);

}  // namespace k3
}  // namespace kernels
}  // namespace sparkinfer
