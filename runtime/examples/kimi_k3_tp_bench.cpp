// What does tensor parallelism actually buy on Kimi K3 decode?
//
//   kimi_k3_tp_bench <first-shard.gguf> [tp_size] [n_layers] [n_tokens] [--ctx N] [--seek]
//                    [--ids a,b,c] [--logits FILE]
//
// --ids feeds an exact prompt (ids, not text) instead of the synthetic sequence, and
// --logits dumps the final step's logits in .spkl. Together they are what compares a
// full 93-layer TP run against the captured llama.cpp reference in bench/refdata/ —
// the only correctness check available at full depth, since no single GPU can hold
// 553 GiB to serve as a tp=1 baseline.
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
#include <cstring>
#include <string>
#include <vector>

using namespace sparkinfer;

namespace {

// .spkl: magic | u32 version | u32 n_tokens | u32 n_vocab | f32[n_tokens][n_vocab].
// Byte-identical to kimi_k3_generate's writer so compare_logits.py reads both.
bool write_spkl(const char* path, const std::vector<float>& logits, int n_vocab) {
    std::FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    const uint32_t ver = 1, ntok = 1, nv = (uint32_t)n_vocab;
    const bool ok = std::fwrite("SPKL", 1, 4, f) == 4 &&
                    std::fwrite(&ver, 4, 1, f) == 1 &&
                    std::fwrite(&ntok, 4, 1, f) == 1 &&
                    std::fwrite(&nv, 4, 1, f) == 1 &&
                    std::fwrite(logits.data(), sizeof(float), logits.size(), f) == logits.size();
    std::fclose(f);
    return ok;
}

std::vector<int> parse_ids(const char* s) {
    std::vector<int> out;
    const char* p = s;
    while (*p) {
        out.push_back(std::atoi(p));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <first-shard.gguf> [tp_size] [n_layers] [n_tokens] "
                     "[--ctx N] [--seek]\n",
                     argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const int tp_size  = argc > 2 ? std::atoi(argv[2]) : 8;
    const int want_lyr = argc > 3 ? std::atoi(argv[3]) : 16;
    int n_tokens = argc > 4 ? std::atoi(argv[4]) : 8;
    const char* logits_path = nullptr;
    // --ctx N allocates the KV cache for N positions; --seek jumps position to near the end
    // of it so the MLA attention actually reduces over the whole thing.
    //
    // WHY SEEK RATHER THAN FILL. K3 has no prefill path, so genuinely reaching 128k means
    // 131,072 sequential forward_token calls -- about 10 hours at main's speed, per
    // measurement. The decode cost at a given context is DATA-INDEPENDENT: MLA attention
    // runs the same dense reduction over the KV cache whether the entries are real
    // activations or zeros, the KDA recurrent state is fixed-size, and the projections do
    // not depend on context at all. Allocating the cache at the target size, leaving it
    // zeroed, and seeking position therefore measures the same per-token cost the real fill
    // would, in minutes.
    //
    // WHAT IT CANNOT DO: establish correctness at that context. It attends over zeros, so
    // the logits are meaningless. Correctness is established separately at short context
    // against the llama.cpp capture in bench/refdata/ -- which is why kimi_k3_eval.sh runs
    // the accuracy pass WITHOUT these flags.
    int  max_ctx = 64;
    bool do_seek = false;
    std::vector<int> ids;
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--ids") && i + 1 < argc) ids = parse_ids(argv[++i]);
        else if (!std::strcmp(argv[i], "--logits") && i + 1 < argc) logits_path = argv[++i];
        else if (!std::strcmp(argv[i], "--ctx") && i + 1 < argc) max_ctx = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--seek")) do_seek = true;
    }

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
    if (!kimi_k3_tp_init(g, cfg, opt, devs, max_ctx, p)) {
        std::printf("init failed at tp=%d, %d layers\n", tp_size, cfg.n_layers);
        return 1;
    }
    const auto t_load1 = std::chrono::steady_clock::now();
    const double load_s = std::chrono::duration<double>(t_load1 - t_load0).count();

    // --seek: place every rank near the END of the allocated KV cache so the MLA attention
    // reduces over the full context. Leaves room for the warm-up token plus n_tokens more,
    // so no step runs past max_ctx.
    if (do_seek) {
        const int target = max_ctx - n_tokens - 2;
        if (target <= 0) {
            std::printf("--seek: ctx %d too small for %d tokens\n", max_ctx, n_tokens);
            return 1;
        }
        for (auto& r : p.ranks) {
            // BOTH copies. Setting only r.state.position leaves the device at 0 while the
            // host plans for 131040 — the kernels then attend over a single position and
            // the benchmark measures a token that is not doing the work it claims to.
            cudaSetDevice(r.device);
            if (!kimi_k3_set_position(r.state, target)) {
                std::printf("seek: failed to set device position on rank %d\n", r.rank);
                return 1;
            }
        }
        std::printf("seek: position -> %d (ctx alloc %d)\n", target, max_ctx);
    }

    std::vector<float> logits((size_t)cfg.vocab);

    // --ids: feed the exact prompt, no warm-up, no synthetic tokens. The recurrent
    // state and KV cache must see the reference's ids in the reference's order, so a
    // discarded warm-up token would advance position and corrupt the comparison.
    if (!ids.empty()) {
        std::printf("prompt: %zu ids from --ids\n", ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            if (!kimi_k3_tp_forward_token(p, ids[i], logits.data())) {
                std::printf("prompt token %zu (id %d) failed\n", i, ids[i]); return 1;
            }
        }
        int best = 0;
        for (int i = 1; i < cfg.vocab; ++i)
            if (logits[(size_t)i] > logits[(size_t)best]) best = i;
        std::printf("argmax next-token id: %d  logit: %.6f\n", best, logits[(size_t)best]);
        if (logits_path) {
            std::printf("%s %s\n",
                        write_spkl(logits_path, logits, cfg.vocab) ? "wrote" : "FAILED to write",
                        logits_path);
        }
        kimi_k3_tp_free(p);
        return 0;
    }

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
