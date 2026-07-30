#pragma once

#include "sparkinfer/gguf.h"
#include "sparkinfer/models/kimi_k3_config.h"
#include "sparkinfer/models/kimi_k3_gguf_config.h"

#include <cstdio>
#include <string>
#include <vector>

namespace sparkinfer {

// Result of a Kimi K3 tensor-manifest check against a (possibly multi-shard) GGUF.
struct KimiK3ManifestReport {
    int missing = 0;
    int checked = 0;
    std::vector<std::string> missing_names;  // capped for readability
};

inline bool kimi_k3_has_tensor(const GGUF& g, const std::string& name) {
    return g.tensor(name) != nullptr;
}

inline bool kimi_k3_require(const GGUF& g, const std::string& name,
                            KimiK3ManifestReport& r) {
    ++r.checked;
    if (kimi_k3_has_tensor(g, name)) return true;
    ++r.missing;
    if (r.missing_names.size() < 48) r.missing_names.push_back(name);
    return false;
}

// Validate that every tensor the K3 graph needs is present in the GGUF index.
// Presence only — dtype/shape checks come later at upload time.
// Mirrors llama_model_kimi_k3::load_arch_tensors required set.
inline bool kimi_k3_validate_tensors(const GGUF& g, const KimiK3Config& cfg,
                                     KimiK3ManifestReport* report = nullptr) {
    KimiK3ManifestReport local;
    KimiK3ManifestReport& r = report ? *report : local;

    auto blk = [](int i, const char* suffix) {
        char buf[96];
        snprintf(buf, sizeof(buf), "blk.%d.%s", i, suffix);
        return std::string(buf);
    };

    kimi_k3_require(g, "token_embd.weight", r);
    kimi_k3_require(g, "output_norm.weight", r);
    kimi_k3_require(g, "output.weight", r);
    if (cfg.attn_res_block_size > 0)
        kimi_k3_require(g, "output_res_score.weight", r);

    for (int i = 0; i < cfg.n_layers; ++i) {
        kimi_k3_require(g, blk(i, "attn_norm.weight"), r);
        kimi_k3_require(g, blk(i, "ffn_norm.weight"), r);
        if (cfg.attn_res_block_size > 0) {
            kimi_k3_require(g, blk(i, "attn_res_score.weight"), r);
            kimi_k3_require(g, blk(i, "ffn_res_score.weight"), r);
        }

        if (cfg.is_kda_layer(i)) {
            kimi_k3_require(g, blk(i, "ssm_conv1d_q.weight"), r);
            kimi_k3_require(g, blk(i, "ssm_conv1d_k.weight"), r);
            kimi_k3_require(g, blk(i, "ssm_conv1d_v.weight"), r);
            kimi_k3_require(g, blk(i, "attn_q.weight"), r);
            kimi_k3_require(g, blk(i, "attn_k.weight"), r);
            kimi_k3_require(g, blk(i, "attn_v.weight"), r);
            kimi_k3_require(g, blk(i, "ssm_f_a.weight"), r);
            kimi_k3_require(g, blk(i, "ssm_f_b.weight"), r);
            kimi_k3_require(g, blk(i, "ssm_beta.weight"), r);
            // TENSOR_NAMES maps SSM_A to "blk.{bid}.ssm_a" — no ".weight" suffix.
            kimi_k3_require(g, blk(i, "ssm_a"), r);
            kimi_k3_require(g, blk(i, "ssm_dt.bias"), r);
            kimi_k3_require(g, blk(i, "ssm_g.weight"), r);
            kimi_k3_require(g, blk(i, "ssm_norm.weight"), r);
            kimi_k3_require(g, blk(i, "attn_output.weight"), r);
        } else {
            kimi_k3_require(g, blk(i, "attn_kv_a_norm.weight"), r);
            // LoRA-Q path (q_a + q_b) OR dense q — exactly one layout.
            if (kimi_k3_has_tensor(g, blk(i, "attn_q_a.weight"))) {
                kimi_k3_require(g, blk(i, "attn_q_a.weight"), r);
                kimi_k3_require(g, blk(i, "attn_q_b.weight"), r);
            } else {
                kimi_k3_require(g, blk(i, "attn_q.weight"), r);
            }
            kimi_k3_require(g, blk(i, "attn_kv_a_mqa.weight"), r);
            if (kimi_k3_has_tensor(g, blk(i, "attn_kv_b.weight"))) {
                kimi_k3_require(g, blk(i, "attn_kv_b.weight"), r);
            } else {
                kimi_k3_require(g, blk(i, "attn_k_b.weight"), r);
                kimi_k3_require(g, blk(i, "attn_v_b.weight"), r);
            }
            // attn_gate is TENSOR_NOT_REQUIRED in the reference — not checked.
            kimi_k3_require(g, blk(i, "attn_output.weight"), r);
        }

        if (i < cfg.leading_dense) {
            kimi_k3_require(g, blk(i, "ffn_gate.weight"), r);
            kimi_k3_require(g, blk(i, "ffn_up.weight"), r);
            kimi_k3_require(g, blk(i, "ffn_down.weight"), r);
        } else {
            kimi_k3_require(g, blk(i, "ffn_gate_inp.weight"), r);
            kimi_k3_require(g, blk(i, "exp_probs_b.bias"), r);
            kimi_k3_require(g, blk(i, "ffn_gate_exps.weight"), r);
            kimi_k3_require(g, blk(i, "ffn_up_exps.weight"), r);
            kimi_k3_require(g, blk(i, "ffn_down_exps.weight"), r);
            if (cfg.expert_latent > 0) {
                kimi_k3_require(g, blk(i, "ffn_routed_down.weight"), r);
                kimi_k3_require(g, blk(i, "ffn_routed_up.weight"), r);
                // ffn_routed_norm is optional
            }
            // shared experts optional in the reference
        }
    }

    if (r.missing > 0) {
        std::fprintf(stderr, "[kimi-k3] manifest: %d missing of %d required checks\n",
                     r.missing, r.checked);
        for (const auto& n : r.missing_names)
            std::fprintf(stderr, "  missing: %s\n", n.c_str());
        if (r.missing > (int)r.missing_names.size())
            std::fprintf(stderr, "  … %d more\n", r.missing - (int)r.missing_names.size());
        return false;
    }
    return true;
}

// Convenience: open (multi-shard) + config + manifest in one shot.
inline bool kimi_k3_load_and_validate(const std::string& first_shard_path,
                                      KimiK3Config& cfg,
                                      GGUF& g,
                                      KimiK3ManifestReport* report = nullptr) {
    if (!g.open(first_shard_path)) return false;
    if (!kimi_k3_config_from_gguf(g, cfg)) return false;
    return kimi_k3_validate_tensors(g, cfg, report);
}

} // namespace sparkinfer
