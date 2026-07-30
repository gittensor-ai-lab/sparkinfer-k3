// Validate the K3 decode plan against real weights, and report per-rank residency.
//
// Three things no CPU test can answer, all answerable in one pass over the GGUF index
// (no tensor data is read, so this is fast even on 802 GiB):
//
//   1. Do the plan's declared widths match the file? A wrong contracted dim in a GEMV
//      reads adjacent weights instead of faulting, so this is the check that turns an
//      inferred width into a known one.
//   2. Which widths are still UNPINNED? Their actual dims are printed so they can be
//      hard-coded — that is the intended workflow, not a warning to ignore.
//   3. What does one rank actually have to hold? The number that decides whether a
//      given context length fits, since TP replicates more than layer-splitting does.
//
// Usage: kimi_k3_plan_check <first-shard.gguf> [tp_size]

#include "sparkinfer/gguf.h"
#include "sparkinfer/models/kimi_k3_config.h"
#include "sparkinfer/models/kimi_k3_decode_plan.h"
#include "sparkinfer/models/kimi_k3_gguf_config.h"
#include "sparkinfer/models/kimi_k3_gguf_manifest.h"
#include "sparkinfer/tp/weight_residency.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace sparkinfer;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <first-shard.gguf> [tp_size]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const int tp_size = argc > 2 ? std::atoi(argv[2]) : 8;

    GGUF g;
    KimiK3Config cfg;
    KimiK3ManifestReport man;
    if (!kimi_k3_load_and_validate(path, cfg, g, &man)) {
        std::fprintf(stderr, "load/manifest failed\n");
        return 1;
    }
    std::printf("model: %zu tensors across %zu shard(s)\n", g.n_tensors(), g.n_shards());
    std::printf("config: %d layers (%d KDA / %d MLA), hidden %d, %d experts top-%d, "
                "latent %d, moe_ffn %d\n",
                cfg.n_layers, cfg.n_kda_layers(), cfg.n_mla_layers(), cfg.hidden,
                cfg.n_experts, cfg.top_k, cfg.expert_latent, cfg.moe_ffn);
    std::printf("manifest: %d checks, %d missing\n\n", man.checked, man.missing);

    // ---- plan layout options come from the file, not from assumptions ----
    K3PlanOptions opt;
    int probe = -1;
    for (int i = 0; i < cfg.n_layers && probe < 0; ++i)
        if (!cfg.is_kda_layer(i)) probe = i;
    if (probe >= 0) {
        const std::string p = "blk." + std::to_string(probe) + ".";
        opt.has_q_lora     = g.tensor((p + "attn_q_a.weight").c_str()) != nullptr;
        opt.has_fused_kv_b = g.tensor((p + "attn_kv_b.weight").c_str()) != nullptr;
        opt.has_attn_gate  = g.tensor((p + "attn_gate.weight").c_str()) != nullptr;
    }
    const int moe0 = cfg.leading_dense;
    if (moe0 < cfg.n_layers) {
        const std::string p = "blk." + std::to_string(moe0) + ".";
        opt.has_shared_experts = g.tensor((p + "ffn_gate_shexp.weight").c_str()) != nullptr;
        opt.has_routed_norm    = g.tensor((p + "ffn_routed_norm.weight").c_str()) != nullptr;
    }
    std::printf("layout: q_lora=%d fused_kv_b=%d attn_gate=%d shared_experts=%d "
                "routed_norm=%d\n",
                opt.has_q_lora, opt.has_fused_kv_b, opt.has_attn_gate,
                opt.has_shared_experts, opt.has_routed_norm);

    tp::ShardDims d;
    d.tp_size = tp_size;
    d.rank = 0;
    d.hidden = cfg.hidden;
    d.n_experts_total = cfg.n_experts;
    d.experts_sharded = (cfg.n_experts % tp_size) == 0;
    d.n_experts = d.experts_sharded ? cfg.n_experts / tp_size : cfg.n_experts;
    d.expert_band = d.experts_sharded ? tp::even_band(cfg.n_experts, tp_size, 0)
                                      : tp::Band{0, cfg.n_experts};

    const K3DecodePlan plan = build_k3_decode_plan(cfg, d, opt);
    std::printf("plan: %zu steps, %d all-reduces (%d per layer)\n\n",
                plan.steps.size(), plan.n_reduces,
                cfg.n_layers ? plan.n_reduces / cfg.n_layers : 0);

    // ---- 1 + 2: shapes ----
    K3PlanShapeReport shp;
    const bool shapes_ok = k3_validate_plan_shapes(plan, g, &shp);
    std::printf("shapes: %d checked, %d mismatched, %d missing, %d unpinned\n",
                shp.checked, shp.mismatched, shp.missing, shp.unpinned);
    if (!shp.unpinned_shapes.empty()) {
        std::printf("\nUNPINNED — actual dims, ready to hard-code:\n");
        for (const auto& u : shp.unpinned_shapes) std::printf("  %s\n", u.c_str());
    }

    // ---- 3: residency ----
    std::vector<tp::TensorDesc> descs;
    descs.reserve(g.n_tensors());
    for (const auto& name : g.tensor_names()) {
        const GGUFTensor* t = g.tensor(name);
        if (!t) continue;
        tp::TensorDesc td;
        td.name = name;
        td.n_dims = t->n_dims;
        td.ggml_type = t->ggml_type;
        for (int i = 0; i < 4; ++i) td.ne[i] = t->dims[i];
        descs.push_back(td);
    }
    const tp::ResidencyReport res = tp::plan_model_residency(descs, d);
    const double GiB = 1024.0 * 1024.0 * 1024.0;
    std::printf("\nresidency at tp=%d:\n", tp_size);
    std::printf("  model total       %10.2f GiB  (%ld tensors)\n",
                res.bytes_model / GiB, res.tensors);
    std::printf("  per rank          %10.2f GiB  (%ld sharded / %ld replicated)\n",
                res.bytes_rank / GiB, res.sharded, res.replicated);
    std::printf("  of which replica  %10.2f GiB  <- the TP memory premium\n",
                res.bytes_replicated / GiB);
    std::printf("  all ranks summed  %10.2f GiB  (%.2fx the model)\n",
                res.bytes_all_ranks / GiB,
                res.bytes_model ? (double)res.bytes_all_ranks / res.bytes_model : 0.0);
    if (res.uneven)
        std::printf("  %ld tensor(s) use a block-aligned UNEVEN split — per-rank shapes "
                    "differ\n", res.uneven);
    if (res.failed) {
        std::printf("\n%ld tensor(s) FAILED to plan:\n", res.failed);
        for (const auto& e : res.errors) std::printf("  %s\n", e.c_str());
    }

    const bool ok = shapes_ok && res.failed == 0;
    std::printf("\n%s\n", ok ? "OK: plan and residency agree with the real weights"
                             : "FAILED: see above");
    return ok ? 0 : 1;
}
