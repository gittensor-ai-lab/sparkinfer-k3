// What does tensor parallelism actually buy on Kimi K3 decode?
//
//   kimi_k3_tp_bench <first-shard.gguf> [tp_size] [n_layers] [n_tokens]
//
// Reports ms/token and ms/layer, and extrapolates ms/layer to the full 93. The
// extrapolation is the useful number: no single GPU can hold 93 layers of this model,
// so a measured 93-layer single-GPU baseline does not exist and cannot be produced.
// Per-layer cost is the only quantity that compares across configurations.
//
// WHAT TP DOES AND DOES NOT SPEED UP HERE. Under ShardPolicy::ExpertsOnly the routed
// experts are banded across ranks, so the MoE dispatch — 69.7% of decode by the
// measured profile — is divided by tp_size. Attention is REPLICATED and therefore
// runs redundantly on every rank at full cost. Amdahl puts the ceiling at
//
//     1 / (0.30 + 0.70/tp)   ->  ~2.6x at tp=8, not 8x
//
// against which the measured number should be read. Beating it would mean something
// is wrong with the measurement; falling far short of it means the collective or the
// launch overhead is eating the win, which is the thing worth knowing.
//
// The first token is discarded: it pays one-time CUDA context, module load and the
// per-device lattice-table upload, none of which recur.

#include "sparkinfer/gguf.h"
#include "sparkinfer/models/kimi_k3.h"
#include "sparkinfer/models/kimi_k3_config.h"
#include "sparkinfer/models/kimi_k3_gguf_manifest.h"
#include "sparkinfer/models/kimi_k3_tp.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace sparkinfer;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <first-shard.gguf> [tp_size] [n_layers] [n_tokens]\n",
                     argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const int tp_size  = argc > 2 ? std::atoi(argv[2]) : 8;
    const int want_lyr = argc > 3 ? std::atoi(argv[3]) : 16;
    const int n_tokens = argc > 4 ? std::atoi(argv[4]) : 8;

    GGUF g; KimiK3Config cfg; KimiK3LayerCoverage cov;
    if (!kimi_k3_load_partial(path.c_str(), cfg, g, &cov)) { std::printf("load failed\n"); return 1; }
    int n_avail = 0;
    while (n_avail < cfg.n_layers && cov.layer_complete[n_avail]) ++n_avail;
    const int n_full = cfg.n_layers;

    int mla_probe = -1, moe_probe = -1;
    for (int i = 0; i < n_full && (mla_probe < 0 || moe_probe < 0); ++i) {
        if (!cov.layer_complete[i]) continue;
        if (mla_probe < 0 && !cfg.is_kda_layer(i)) mla_probe = i;
        if (moe_probe < 0 && i >= cfg.leading_dense) moe_probe = i;
    }
    if (mla_probe < 0) { std::printf("no complete MLA layer\n"); return 1; }

    cfg.n_layers = std::min(n_avail, want_lyr);
    K3PlanOptions opt;
    {
        const std::string p = "blk." + std::to_string(mla_probe) + ".";
        opt.has_q_lora     = g.tensor((p + "attn_q_a.weight").c_str()) != nullptr;
        opt.has_fused_kv_b = g.tensor((p + "attn_kv_b.weight").c_str()) != nullptr;
        opt.has_attn_gate  = g.tensor((p + "attn_gate.weight").c_str()) != nullptr;
    }
    if (moe_probe >= 0) {
        const std::string p = "blk." + std::to_string(moe_probe) + ".";
        opt.has_shared_experts = g.tensor((p + "ffn_gate_shexp.weight").c_str()) != nullptr;
        opt.has_routed_norm    = g.tensor((p + "ffn_routed_norm.weight").c_str()) != nullptr;
    }

    int ndev = 0; cudaGetDeviceCount(&ndev);
    if (tp_size > ndev) { std::printf("tp_size %d > %d GPUs\n", tp_size, ndev); return 1; }

    std::vector<int> devs;
    for (int i = 0; i < tp_size; ++i) devs.push_back(i);

    const auto t_load0 = std::chrono::steady_clock::now();
    KimiK3TP p;
    if (!kimi_k3_tp_init(g, cfg, opt, devs, /*max_ctx=*/64, p)) {
        std::printf("init failed at tp=%d, %d layers\n", tp_size, cfg.n_layers);
        return 1;
    }
    const auto t_load1 = std::chrono::steady_clock::now();
    const double load_s = std::chrono::duration<double>(t_load1 - t_load0).count();

    std::vector<float> logits((size_t)cfg.vocab);
    // Warm-up token, discarded: one-time context/module/lattice-table costs.
    if (!kimi_k3_tp_forward_token(p, 1000, logits.data())) { std::printf("warmup failed\n"); return 1; }

    const auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < n_tokens; ++t) {
        if (!kimi_k3_tp_forward_token(p, 1000 + t, logits.data())) {
            std::printf("token %d failed\n", t); return 1;
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / n_tokens;

    size_t free_b = 0, total_b = 0;
    cudaSetDevice(devs[0]);
    cudaMemGetInfo(&free_b, &total_b);
    const double GiB = 1024.0 * 1024.0 * 1024.0;

    std::printf("\n=== Kimi K3 UD-IQ1_S decode, tp=%d, ExpertsOnly ===\n", tp_size);
    std::printf("  layers measured    %d of %d\n", cfg.n_layers, n_full);
    std::printf("  load               %.1f s\n", load_s);
    std::printf("  rank0 VRAM in use  %.2f GiB of %.2f\n",
                (total_b - free_b) / GiB, total_b / GiB);
    std::printf("  ms/token           %.2f  (%.2f tok/s)\n", ms, 1000.0 / ms);
    std::printf("  ms/layer           %.3f\n", ms / cfg.n_layers);
    std::printf("  collectives/token  %ld\n", p.n_collectives / (n_tokens + 1));
    std::printf("  --> extrapolated to %d layers: %.1f ms/token (%.2f tok/s)\n",
                n_full, ms / cfg.n_layers * n_full,
                1000.0 / (ms / cfg.n_layers * n_full));

    // SPARKINFER_K3_PROFILE=1 prints the per-phase split. Note it adds cudaEvent
    // pairs around every branch, which roughly doubles ms/token — read the SHARES,
    // never the absolute time, from a profiled run.
    kimi_k3_profile_report();

    kimi_k3_tp_free(p);
    return 0;
}
