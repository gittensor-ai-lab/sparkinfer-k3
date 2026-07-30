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

// Which layers are FULLY present. The question a partial model actually has to answer.
//
// is_partial() tells you shards are missing; it cannot tell you what you can still do.
// A layer whose every required tensor is present can be validated numerically against
// llama.cpp even when 13 of 14 shards are absent — and that is the whole development
// workflow on a single GPU. A layer missing one tensor can not, and must not be tried:
// the absent tensor reads as a null pointer.
struct KimiK3LayerCoverage {
    std::vector<char> layer_complete;   // per layer, 1 when every required tensor exists
    int n_complete = 0;
    int first_complete_kda = -1;        // -1 when none
    int first_complete_mla = -1;
    bool embed_present = false;
    bool head_present  = false;
};

inline KimiK3LayerCoverage kimi_k3_layer_coverage(const GGUF& g, const KimiK3Config& cfg) {
    KimiK3LayerCoverage cov;
    cov.layer_complete.assign(cfg.n_layers, 0);
    cov.embed_present = kimi_k3_has_tensor(g, "token_embd.weight");
    cov.head_present  = kimi_k3_has_tensor(g, "output.weight");

    for (int i = 0; i < cfg.n_layers; ++i) {
        // Reuse the required-set walk for ONE layer by running the manifest against a
        // single-layer view: same predicate, so coverage cannot drift from validation.
        KimiK3Config one = cfg;
        one.n_layers = 1;
        one.layer_is_kda.assign(1, cfg.is_kda_layer(i) ? 1 : 0);
        // leading_dense is an absolute layer index, so re-express it for the shifted view.
        one.leading_dense = (i < cfg.leading_dense) ? 1 : 0;

        KimiK3ManifestReport r;
        bool complete = true;
        auto blk = [i](const char* suffix) {
            char buf[96];
            snprintf(buf, sizeof(buf), "blk.%d.%s", i, suffix);
            return std::string(buf);
        };
        // Mirror of the per-layer block in kimi_k3_validate_tensors, indexed at i.
        auto need = [&](const char* suffix) {
            if (!kimi_k3_has_tensor(g, blk(suffix))) complete = false;
        };
        need("attn_norm.weight");
        need("ffn_norm.weight");
        if (cfg.attn_res_block_size > 0) { need("attn_res_score.weight"); need("ffn_res_score.weight"); }
        if (cfg.is_kda_layer(i)) {
            for (const char* t : {"ssm_conv1d_q.weight","ssm_conv1d_k.weight","ssm_conv1d_v.weight",
                                  "attn_q.weight","attn_k.weight","attn_v.weight",
                                  "ssm_f_a.weight","ssm_f_b.weight","ssm_beta.weight","ssm_a",
                                  "ssm_dt.bias","ssm_g.weight","ssm_norm.weight","attn_output.weight"})
                need(t);
        } else {
            need("attn_kv_a_norm.weight");
            if (kimi_k3_has_tensor(g, blk("attn_q_a.weight"))) { need("attn_q_a.weight"); need("attn_q_b.weight"); }
            else need("attn_q.weight");
            need("attn_kv_a_mqa.weight");
            if (!kimi_k3_has_tensor(g, blk("attn_kv_b.weight"))) { need("attn_k_b.weight"); need("attn_v_b.weight"); }
            need("attn_output.weight");
        }
        if (i < cfg.leading_dense) {
            need("ffn_gate.weight"); need("ffn_up.weight"); need("ffn_down.weight");
        } else {
            need("ffn_gate_inp.weight"); need("exp_probs_b.bias");
            need("ffn_gate_exps.weight"); need("ffn_up_exps.weight"); need("ffn_down_exps.weight");
            if (cfg.expert_latent > 0) { need("ffn_routed_down.weight"); need("ffn_routed_up.weight"); }
        }
        (void)r;

        cov.layer_complete[i] = complete ? 1 : 0;
        if (complete) {
            ++cov.n_complete;
            if (cfg.is_kda_layer(i)) { if (cov.first_complete_kda < 0) cov.first_complete_kda = i; }
            else                     { if (cov.first_complete_mla < 0) cov.first_complete_mla = i; }
        }
    }
    return cov;
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

// Partial-model load: metadata and config are still REQUIRED to be complete (they live
// in shard 1), but a missing tensor is reported as coverage rather than as failure.
//
// Returns false only when the config itself cannot be read — that is a real error even
// in development, because every downstream shape derives from it. Tensor coverage comes
// back in `cov` for the caller to act on.
inline bool kimi_k3_load_partial(const std::string& first_shard_path,
                                 KimiK3Config& cfg,
                                 GGUF& g,
                                 KimiK3LayerCoverage* cov = nullptr) {
    GGUFOpenOptions opt;
    opt.allow_missing_shards = true;
    if (!g.open(first_shard_path, opt)) return false;
    if (!kimi_k3_config_from_gguf(g, cfg)) return false;
    if (cov) *cov = kimi_k3_layer_coverage(g, cfg);
    return true;
}

} // namespace sparkinfer
