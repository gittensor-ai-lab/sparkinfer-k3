#pragma once

#include "sparkinfer/gguf.h"
#include "sparkinfer/models/kimi_k3_config.h"

#include <cstdio>
#include <limits>
#include <string>

namespace sparkinfer {

inline bool kimi_k3_is_arch(const GGUF& g) {
    return g.meta_str("general.architecture") == "kimi-k3";
}

inline long kimi_k3_meta_int(const GGUF& g, const std::string& key, long def) {
    return g.meta_int("kimi-k3." + key, def);
}

inline double kimi_k3_meta_float(const GGUF& g, const std::string& key, double def) {
    return g.meta_float("kimi-k3." + key, def);
}

// True when the key is present as a scalar int (not merely defaulted).
inline bool kimi_k3_has_int(const GGUF& g, const std::string& key) {
    const long missing = std::numeric_limits<long>::min();
    return g.meta_int("kimi-k3." + key, missing) != missing;
}

inline bool kimi_k3_has_float(const GGUF& g, const std::string& key) {
    const double missing = -std::numeric_limits<double>::infinity();
    return g.meta_float("kimi-k3." + key, missing) != missing;
}

// Parse Kimi K3 hparams from a GGUF. Returns false (and leaves cfg partially
// filled) when the architecture is wrong or a REQUIRED key is missing.
//
// Required keys match llama_model_kimi_k3::load_arch_hparams asserts:
//   expert_latent_length, attn_res.block_size, situ_beta, situ_linear_beta
// plus the per-layer attention.head_count_kv array (length == block_count).
inline bool kimi_k3_config_from_gguf(const GGUF& g, KimiK3Config& cfg) {
    if (!kimi_k3_is_arch(g)) {
        std::fprintf(stderr, "[kimi-k3] general.architecture is '%s', want 'kimi-k3'\n",
                     g.meta_str("general.architecture").c_str());
        return false;
    }

    cfg.n_layers  = (int)kimi_k3_meta_int(g, "block_count", cfg.n_layers);
    cfg.hidden    = (int)kimi_k3_meta_int(g, "embedding_length", cfg.hidden);
    cfg.vocab     = (int)kimi_k3_meta_int(g, "vocab_size", cfg.vocab);
    cfg.n_q_heads = (int)kimi_k3_meta_int(g, "attention.head_count", cfg.n_q_heads);
    cfg.key_length = (int)kimi_k3_meta_int(g, "attention.key_length", cfg.key_length);
    cfg.key_length_mla = (int)kimi_k3_meta_int(g, "attention.key_length_mla", cfg.key_length_mla);
    cfg.value_length_mla = (int)kimi_k3_meta_int(g, "attention.value_length_mla",
                                                 cfg.value_length_mla);
    cfg.q_lora_rank  = (int)kimi_k3_meta_int(g, "attention.q_lora_rank", cfg.q_lora_rank);
    cfg.kv_lora_rank = (int)kimi_k3_meta_int(g, "attention.kv_lora_rank", cfg.kv_lora_rank);
    cfg.rope_dim     = (int)kimi_k3_meta_int(g, "rope.dimension_count", cfg.rope_dim);
    cfg.rms_eps      = (float)kimi_k3_meta_float(g, "attention.layer_norm_rms_epsilon", cfg.rms_eps);

    cfg.n_experts = (int)kimi_k3_meta_int(g, "expert_count", cfg.n_experts);
    cfg.top_k     = (int)kimi_k3_meta_int(g, "expert_used_count", cfg.top_k);
    cfg.n_shared  = (int)kimi_k3_meta_int(g, "expert_shared_count", cfg.n_shared);
    cfg.moe_ffn   = (int)kimi_k3_meta_int(g, "expert_feed_forward_length", cfg.moe_ffn);
    cfg.leading_dense = (int)kimi_k3_meta_int(g, "leading_dense_block_count", cfg.leading_dense);
    cfg.dense_ffn = (int)kimi_k3_meta_int(g, "feed_forward_length", cfg.dense_ffn);

    cfg.kda_head_dim = (int)kimi_k3_meta_int(g, "kda.head_dim", cfg.kda_head_dim);
    cfg.head_dim = cfg.kda_head_dim;
    cfg.kda_conv_kernel = (int)kimi_k3_meta_int(g, "ssm.conv_kernel", cfg.kda_conv_kernel);
    cfg.kda_gate_lower_bound =
        (float)kimi_k3_meta_float(g, "kda.gate_lower_bound", cfg.kda_gate_lower_bound);

    cfg.eos_id = (int)g.meta_int("tokenizer.ggml.eos_token_id", cfg.eos_id);
    if (const GGUFTensor* emb = g.tensor("token_embd.weight")) {
        if (emb->n_dims >= 2) cfg.vocab = (int)emb->dims[1];
    }

    // ---- REQUIRED keys (silent default = fluent garbage) --------------------
    if (!kimi_k3_has_int(g, "expert_latent_length")) {
        std::fprintf(stderr, "[kimi-k3] missing REQUIRED key kimi-k3.expert_latent_length\n");
        return false;
    }
    if (!kimi_k3_has_int(g, "attn_res.block_size")) {
        std::fprintf(stderr, "[kimi-k3] missing REQUIRED key kimi-k3.attn_res.block_size\n");
        return false;
    }
    if (!kimi_k3_has_float(g, "activation.situ_beta")) {
        std::fprintf(stderr, "[kimi-k3] missing REQUIRED key kimi-k3.activation.situ_beta\n");
        return false;
    }
    if (!kimi_k3_has_float(g, "activation.situ_linear_beta")) {
        std::fprintf(stderr, "[kimi-k3] missing REQUIRED key kimi-k3.activation.situ_linear_beta\n");
        return false;
    }

    cfg.expert_latent = (int)kimi_k3_meta_int(g, "expert_latent_length", 0);
    cfg.attn_res_block_size = (int)kimi_k3_meta_int(g, "attn_res.block_size", 0);
    cfg.situ_beta = (float)kimi_k3_meta_float(g, "activation.situ_beta", 0.0);
    cfg.situ_linear_beta = (float)kimi_k3_meta_float(g, "activation.situ_linear_beta", 0.0);

    if (cfg.expert_latent <= 0) {
        std::fprintf(stderr, "[kimi-k3] expert_latent_length must be > 0\n");
        return false;
    }
    if (cfg.attn_res_block_size <= 0) {
        std::fprintf(stderr, "[kimi-k3] attn_res.block_size must be > 0\n");
        return false;
    }

    // Per-layer KDA/MLA map from head_count_kv array (0 = KDA, 1 = MLA).
    const std::vector<long>* hckv = g.meta_int_array("kimi-k3.attention.head_count_kv");
    if (!hckv) {
        std::fprintf(stderr, "[kimi-k3] missing REQUIRED array kimi-k3.attention.head_count_kv\n");
        return false;
    }
    if ((int)hckv->size() != cfg.n_layers) {
        std::fprintf(stderr,
                     "[kimi-k3] head_count_kv length %zu != block_count %d\n",
                     hckv->size(), cfg.n_layers);
        return false;
    }
    cfg.layer_is_kda.assign(cfg.n_layers, 0);
    for (int i = 0; i < cfg.n_layers; ++i) {
        const long v = (*hckv)[i];
        if (v != 0 && v != 1) {
            std::fprintf(stderr, "[kimi-k3] head_count_kv[%d]=%ld (want 0=KDA or 1=MLA)\n",
                         i, v);
            return false;
        }
        cfg.layer_is_kda[i] = (v == 0) ? 1 : 0;
    }

    return true;
}

} // namespace sparkinfer
