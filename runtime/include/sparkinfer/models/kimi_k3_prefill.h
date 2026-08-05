#pragma once
// Batched-prompt prefill driver for Kimi K3.
//
// SCOPE, STATED PLAINLY. This is NOT the full batched-GEMM prefill described in
// docs/tensor-parallel.md item 2 (tensor-core GEMMs replacing per-token GEMVs for
// the weight-bound projections). That needs new dequant+GEMM kernels for K3's
// native IQ1_S/IQ2_XS quant formats, which do not exist yet — see the roadmap.
//
// What this DOES do, and why it is still a real win:
//   1. Hoists the x/x_next device buffers out of the per-token call. The existing
//      token loop (kimi_k3_forward_token) cudaMalloc's and cudaFree's both on
//      EVERY token; over an N-token prompt that is 2N alloc/free pairs for two
//      H-wide buffers that never change size.
//   2. Skips the output_norm + lm_head (output.weight, [vocab, hidden]) projection
//      for every prompt token except the last. kimi_k3_generate.cpp's own comment
//      states only the final prompt position's logits are ever used (they seed the
//      first decode step) — every earlier token's lm_head pass is fully wasted
//      work today. vocab is one of the single largest GEMMs in the model, so this
//      is the dominant saving here, not the allocation hoist.
//
// Everything else is byte-for-byte the existing per-layer call sequence
// (kimi_k3_forward_layer, K3LayerPhase::All) — no new kernels, no changed
// numerics, no new TP behavior. This function fills the same KV cache / KDA
// recurrent state / residual bank a run of kimi_k3_forward_token would, so a
// subsequent decode step is bit-identical to today's naive prompt loop.
//
// UNVERIFIED ON HARDWARE. Written and reasoned against the actual per-layer
// source (kimi_k3_forward_layer_phase), but not yet run — no GPU was available
// at write time. Needs kimi_k3_eval.sh / a correctness check before it can be
// trusted, same as any other change here.

#include "sparkinfer/models/kimi_k3.h"

namespace sparkinfer {

// Run the whole prompt through the layer stack, one token per iteration, filling
// KV cache / KDA state / residual bank exactly as kimi_k3_forward_token would.
// Only the LAST token's logits are computed and written to out_logits
// (cfg.vocab floats) — earlier tokens skip output_norm + lm_head entirely, since
// nothing downstream reads their logits.
//
// fwd must already be allocated (kimi_k3_forward_alloc_scratch) and fwd.state
// must be freshly reset (kimi_k3_reset_state) if this is a new sequence — same
// preconditions kimi_k3_forward_token has today. Advances fwd.state->position by
// n on success, exactly as n calls to kimi_k3_forward_token would.
//
// Returns false on any per-layer failure (same failure contract as
// kimi_k3_forward_token) — caller should treat this as "prefill failed", not
// partially applied; state may be partially advanced on failure, same as today's
// token loop, since layer failure part-way through has always left state exactly
// as far as it got.
bool kimi_k3_prefill_tokens(KimiK3Forward& fwd, const int* prompt_ids, int n,
                            float* out_logits);

} // namespace sparkinfer
