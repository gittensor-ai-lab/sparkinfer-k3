#pragma once

#include <vector>

namespace sparkinfer {

// Kimi K3 (Moonshot) decode configuration.
//
// Defaults match moonshotai/Kimi-K3 / unsloth UD-Q2_K_XL as typed by
// unslothai/llama.cpp @ efc8bc38 (LLM_TYPE_2_8T_A50B). Required GGUF keys
// (expert_latent_length, attn_res.block_size, both situ betas) must be present
// in the file — kimi_k3_config_from_gguf refuses silent defaults for those,
// because a defaulted load emits fluent garbage rather than failing.
struct KimiK3Config {
    int   vocab       = 163840;
    int   hidden      = 7168;
    int   n_layers    = 93;
    int   n_q_heads   = 96;
    int   head_dim    = 128;     // KDA head_dim; MLA uses separate dims below
    int   n_experts   = 896;
    int   top_k       = 16;
    int   n_shared    = 2;
    int   moe_ffn     = 3072;
    int   expert_latent = 3584;  // REQUIRED — routed experts live here, not at hidden
    int   dense_ffn   = 33792;   // leading dense block only
    int   leading_dense = 1;
    float rms_eps     = 1e-5f;
    int   eos_id      = 163586;  // READ from the real UD-IQ1_S metadata shard
    int   max_seq     = 1048576;

    // situ — both REQUIRED in GGUF
    float situ_beta         = 4.0f;
    float situ_linear_beta  = 25.0f;

    // cross-layer residual attention — REQUIRED
    int   attn_res_block_size = 12;

    // KDA (linear / recurrent)
    int   kda_head_dim      = 128;
    int   kda_conv_kernel   = 4;
    float kda_gate_lower_bound = -5.0f;

    // MLA (full attention). GGUF encodes MLA as MQA with key_length = 576.
    int   q_lora_rank       = 1536;
    int   kv_lora_rank      = 512;
    int   key_length        = 576;   // kv_lora + qk_rope
    int   key_length_mla    = 192;   // qk_nope + qk_rope
    int   value_length_mla  = 128;
    int   rope_dim          = 64;    // present in GGUF; arch is NoPE-only at runtime

    // Per-layer type map. layer_is_kda[i] == true when head_count_kv[i] == 0.
    // Length must equal n_layers. Derived from the GGUF int array — never baked.
    //
    // The real pattern, read off the file, is period 4 with the full-attention layer
    // LAST in each group: k k k M | k k k M | …  So layer 0 is KDA and layer 3 is the
    // first MLA. Recorded as an observation only — the map is still read from the GGUF
    // every time, because baking a period-4 assumption would silently mistype every
    // layer in any model that interleaves differently.
    std::vector<char> layer_is_kda;

    // Both counts are bounded by n_layers, NOT by layer_is_kda.size().
    //
    // They must agree on the same denominator or n_mla_layers() goes NEGATIVE: it is
    // defined as n_layers - n_kda_layers(), so if n_kda_layers() counted the whole
    // vector while n_layers was smaller, the difference underflows. That is not
    // hypothetical — a caller that legitimately reduces n_layers (a pipeline test
    // capping depth to what fits in VRAM, a partial-model bring-up) leaves the map
    // longer than the layer count, and the negative then propagates into a
    // vector::resize() as a huge size_t and throws std::length_error far from the
    // cause. Clamping the loop here keeps the two consistent by construction.
    int n_kda_layers() const {
        const int n_map = (int)layer_is_kda.size();
        const int lim = n_layers < n_map ? n_layers : n_map;
        int n = 0;
        for (int i = 0; i < lim; ++i) n += (layer_is_kda[i] != 0);
        return n;
    }
    int n_mla_layers() const { return n_layers - n_kda_layers(); }

    bool is_kda_layer(int layer) const {
        return layer >= 0 && layer < (int)layer_is_kda.size() && layer_is_kda[layer] != 0;
    }
};

} // namespace sparkinfer
