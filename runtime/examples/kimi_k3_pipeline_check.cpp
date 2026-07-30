// Layer-split pipeline equivalence: the SAME token through a 1-stage pipeline and
// through an N-stage one must produce BIT-IDENTICAL logits.
//
// Why bit-identical rather than "close": every stage runs the same kernels in the
// same order on the same device, and the only thing a stage boundary adds is a
// device->host->device round-trip of f32 values, which is exact. So there is no
// legitimate source of difference at all. That makes this a far sharper test than
// a tolerance check — a missing or truncated handoff cannot hide inside a
// tolerance, it shows up as a nonzero diff immediately.
//
// THE THING THIS IS REALLY TESTING is the residual-bank handoff. The cross-layer
// residual bank is read at EVERY layer and written at every attn_res.block_size-th
// one, so it spans the whole forward pass. Put a stage boundary anywhere after the
// first checkpoint layer and the next stage's res_mix must score against
// checkpoints banked on the PREVIOUS stage — a different GPU in production. An
// implementation that hands off only the hidden state (the obvious one) still
// runs, still produces plausible logits, and is wrong from the first boundary
// onward. Splitting at several different points is what catches that.
//
// Runs against however many layers the GGUF currently has, by capping n_layers —
// the pipeline's split/handoff mechanics are identical regardless of depth, so
// this is testable long before all 594 GB has downloaded.

#include "sparkinfer/models/kimi_k3.h"
#include "sparkinfer/models/kimi_k3_gguf_manifest.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace sparkinfer;

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1]
        : "/workspace/models_k3/UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf";
    const int token = argc > 2 ? std::atoi(argv[2]) : 1000;
    // Depth is bounded by VRAM, not by what has downloaded: a MoE layer is ~6 GiB
    // of weights, so ~13 layers is the ceiling on a 96 GiB card. Default is chosen
    // to span at least two checkpoint layers (0 and attn_res_block_size) so a stage
    // boundary genuinely lands after a bank push — which is the whole point.
    const int max_layers = argc > 3 ? std::atoi(argv[3]) : 13;

    GGUF g; KimiK3Config cfg; KimiK3LayerCoverage cov;
    if (!kimi_k3_load_partial(path, cfg, g, &cov)) { std::printf("load failed\n"); return 1; }

    // Cap to the contiguous run of complete layers starting at 0.
    int n_avail = 0;
    while (n_avail < cfg.n_layers && cov.layer_complete[n_avail]) ++n_avail;
    if (n_avail < 2) { std::printf("need >=2 complete layers, have %d\n", n_avail); return 1; }
    const int n_full = cfg.n_layers;
    cfg.n_layers = n_avail < max_layers ? n_avail : max_layers;
    std::printf("pipeline equivalence over %d layers (%d downloaded, %d in the full model; "
                "VRAM caps depth at ~13 MoE layers on a 96 GiB card)\n",
                cfg.n_layers, n_avail, n_full);
    if (cfg.attn_res_block_size > 0)
        std::printf("checkpoint layers within this depth: every %d (so at least layers 0"
                    "%s)\n", cfg.attn_res_block_size,
                    cfg.n_layers > cfg.attn_res_block_size ? " and 12" : "");

    K3PlanOptions opt;
    {
        const int probe = n_avail > 3 ? 3 : 0;
        const std::string p = "blk." + std::to_string(probe) + ".";
        opt.has_q_lora      = g.tensor((p + "attn_q_a.weight").c_str()) != nullptr;
        opt.has_attn_gate   = g.tensor((p + "attn_gate.weight").c_str()) != nullptr;
        opt.has_fused_kv_b  = false;
        opt.has_shared_experts = g.tensor((p + "ffn_gate_shexp.weight").c_str()) != nullptr;
        opt.has_routed_norm    = g.tensor((p + "ffn_routed_norm.weight").c_str()) != nullptr;
    }

    auto run = [&](const std::vector<int>& devices, std::vector<float>& logits) -> bool {
        KimiK3Pipeline p;
        if (!kimi_k3_pipeline_init(g, cfg, opt, devices, /*max_ctx=*/16, p)) {
            std::printf("pipeline init failed for %zu stage(s)\n", devices.size());
            return false;
        }
        logits.assign(cfg.vocab, 0.0f);
        const bool ok = kimi_k3_pipeline_forward_token(p, token, logits.data());
        kimi_k3_pipeline_free(p);
        return ok;
    };

    std::vector<float> ref;
    if (!run({0}, ref)) { std::printf("1-stage run failed\n"); return 1; }
    double ss = 0; for (float v : ref) ss += (double)v * v;
    std::printf("\n1-stage (reference): logits rms=%.6g  first4=%.6g,%.6g,%.6g,%.6g\n",
                std::sqrt(ss / cfg.vocab), ref[0], ref[1], ref[2], ref[3]);

    int argmax_ref = 0;
    for (int i = 1; i < cfg.vocab; ++i) if (ref[i] > ref[argmax_ref]) argmax_ref = i;
    std::printf("1-stage argmax token = %d (logit %.6g)\n", argmax_ref, ref[argmax_ref]);

    int failures = 0;
    // Several split counts: the boundary lands in a different place each time, so
    // between them they put a boundary both before and after checkpoint layers.
    for (int nstage : {2, 3, 4}) {
        std::vector<int> devices((size_t)nstage, 0);   // same GPU: tests handoff, not transport
        std::vector<float> got;
        if (!run(devices, got)) { std::printf("%d-stage run failed\n", nstage); ++failures; continue; }

        long ndiff = 0; double worst = 0;
        for (int i = 0; i < cfg.vocab; ++i) {
            if (std::memcmp(&ref[i], &got[i], sizeof(float)) != 0) {
                ++ndiff;
                worst = std::fmax(worst, std::fabs((double)ref[i] - (double)got[i]));
            }
        }
        int argmax_got = 0;
        for (int i = 1; i < cfg.vocab; ++i) if (got[i] > got[argmax_got]) argmax_got = i;

        const bool ok = (ndiff == 0);
        if (!ok) ++failures;
        std::printf("%d-stage: %ld/%d logits differ from 1-stage", nstage, ndiff, cfg.vocab);
        if (ndiff) std::printf(" (worst abs %.3e)", worst);
        std::printf("   argmax=%d %s\n", argmax_got,
                    ok ? "BIT-IDENTICAL" : (argmax_got == argmax_ref ? "DIFFER (argmax same)"
                                                                     : "DIFFER (argmax CHANGED)"));
    }

    std::printf("\n%s\n", failures == 0
        ? "PASS: N-stage pipeline is bit-identical to 1-stage (hidden + residual bank "
          "handoff correct)"
        : "FAIL");
    return failures == 0 ? 0 : 1;
}
