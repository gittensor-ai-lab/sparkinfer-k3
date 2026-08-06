#pragma once

#include <cuda_runtime.h>

namespace sparkinfer::kernels::k3 {

// Causal multi-query attention for a tile of already-projected prompt positions.
//
// This kernel does not implement MLA projection or weight absorption. It consumes
// absorbed queries and compressed KV rows and evaluates causal attention for several
// prompt positions.  Layouts are:
//
//   query    [tokens][heads][key_length]
//   kv_cache [context][key_length]                 (shared by all query heads)
//   wv_b     [heads][value_dim][kv_lora]
//   gate     [tokens][heads][value_dim], optional
//   output   [tokens][heads][value_dim]
//
// Query t may attend through cache row start_pos + t, inclusive.  The first kv_lora
// entries of each cache row are both the compressed NoPE key and latent value; the
// remaining entries are the RoPE key.  When gate is non-null the result is multiplied
// by sigmoid(gate), matching mla_gate_out_f32.
bool k3_mla_prefill_attention_supported(int tokens, int context, int heads,
                                        int key_length, int kv_lora, int value_dim);

bool k3_mla_prefill_attention_f32(float* output, const float* query,
                                  const float* kv_cache, const float* wv_b,
                                  const float* gate, int start_pos, int tokens,
                                  int context, int heads, int key_length,
                                  int kv_lora, int value_dim, float scale,
                                  cudaStream_t stream);

bool k3_mla_prefill_attention_kvf16(float* output, const float* query,
                                    const void* kv_cache, const float* wv_b,
                                    const float* gate, int start_pos, int tokens,
                                    int context, int heads, int key_length,
                                    int kv_lora, int value_dim, float scale,
                                    cudaStream_t stream);

}  // namespace sparkinfer::kernels::k3
