// End-to-end tensor-parallel decode: TP=N logits must match the single-GPU logits.
//
//   kimi_k3_tp_check <first-shard.gguf> [tp_size] [n_layers] [n_tokens]
//
// Everything upstream of this validated one piece each: the band arithmetic tiles
// (kimi_k3_tp_load_check), the phase split is a pure refactor (kimi_k3_tp_phase_check),
// the expert bands sum back to the unsharded dispatch (kimi_k3_tp_moe_check), and the
// f32 collective reduces exactly (tp_allreduce_check). This runs all of them together
// against real weights and asks the only question that matters: does an N-GPU decode
// produce the same distribution as a 1-GPU decode.
//
// MULTI-TOKEN, deliberately. A single token would not exercise the state that
// persists ACROSS tokens and is the easiest thing to get wrong per rank: the KDA
// conv window and delta matrix, the MLA KV cache, and the position counter. A rank
// whose position drifted, or whose recurrent state was allocated but never advanced,
// is exactly right on token 1 and wrong from token 2 onward.
//
// The comparison is normalised, not bitwise: the collective reassociates the expert
// partial sums, so ~1 ulp of drift per MoE layer is expected and correct. What is not
// expected is a changed argmax — if the top token moves, the model has changed.

#include "sparkinfer/gguf.h"
#include "sparkinfer/models/kimi_k3.h"
#include "sparkinfer/models/kimi_k3_config.h"
#include "sparkinfer/models/kimi_k3_gguf_manifest.h"
#include "sparkinfer/models/kimi_k3_tp.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace sparkinfer;

namespace {

int argmax(const std::vector<float>& v) {
    int best = 0;
    for (int i = 1; i < (int)v.size(); ++i) if (v[(size_t)i] > v[(size_t)best]) best = i;
    return best;
}

// A trace of every debug checkpoint the forward emits, as (name, L1 norm). Comparing
// two traces localises a divergence to the exact tag+layer where it first appears,
// which is far cheaper than reasoning backwards from the logits.
struct Trace {
    std::vector<std::string> name;
    std::vector<double> l1;
};

KimiK3DebugFn make_tracer(Trace* tr) {
    return [tr](const char* tag, int layer, const float* dev, int64_t n) {
        if (!dev || n <= 0) return;
        std::vector<float> host((size_t)n);
        if (cudaMemcpy(host.data(), dev, (size_t)n * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess) return;
        double s = 0;
        for (int64_t i = 0; i < n; ++i) s += std::fabs((double)host[(size_t)i]);
        tr->name.push_back(std::string(tag) + "@" + std::to_string(layer));
        tr->l1.push_back(s);
    };
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <first-shard.gguf> [tp_size] [n_layers] [n_tokens]\n",
                     argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const int tp_size  = argc > 2 ? std::atoi(argv[2]) : 8;
    const int want_lyr = argc > 3 ? std::atoi(argv[3]) : 8;
    const int n_tokens = argc > 4 ? std::atoi(argv[4]) : 3;

    GGUF g; KimiK3Config cfg; KimiK3LayerCoverage cov;
    if (!kimi_k3_load_partial(path.c_str(), cfg, g, &cov)) { std::printf("load failed\n"); return 1; }
    int n_avail = 0;
    while (n_avail < cfg.n_layers && cov.layer_complete[n_avail]) ++n_avail;
    const int n_full = cfg.n_layers;

    // Probe the layout BEFORE trimming n_layers. These flags describe the MODEL, not
    // the slice under test, so they must come from a layer of the right kind wherever
    // it lives — otherwise a short bisecting run (2-3 layers, all KDA) cannot resolve
    // the MLA options at all.
    int mla_probe = -1, moe_probe = -1;
    for (int i = 0; i < n_full && (mla_probe < 0 || moe_probe < 0); ++i) {
        if (!cov.layer_complete[i]) continue;
        if (mla_probe < 0 && !cfg.is_kda_layer(i)) mla_probe = i;
        if (moe_probe < 0 && i >= cfg.leading_dense) moe_probe = i;
    }

    cfg.n_layers = std::min(n_avail, want_lyr);
    if (cfg.n_layers < 1) { std::printf("need >=1 complete layer, have %d\n", n_avail); return 1; }

    // Probe the layout from the RIGHT KIND OF LAYER for each question. The MLA
    // options only exist on a full-attention layer: a KDA layer has attn_q.weight and
    // no attn_q_a/attn_q_b, so probing q_lora on one reports "no q_lora" and the
    // loader then demands attn_q.weight from every MLA layer — which does not have it.
    // (That is exactly how this check failed first time: leading_dense is 1, and
    // layer 1 is KDA.)
    K3PlanOptions opt;
    if (mla_probe < 0) {
        std::printf("no complete MLA layer in the downloaded shards\n");
        return 1;
    }
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
    std::printf("layout (mla probe blk.%d, moe probe blk.%d): q_lora=%d attn_gate=%d "
                "shexp=%d routed_norm=%d\n", mla_probe, moe_probe,
                opt.has_q_lora, opt.has_attn_gate, opt.has_shared_experts,
                opt.has_routed_norm);

    int ndev = 0;
    cudaGetDeviceCount(&ndev);
    std::printf("TP decode check: %d layers (of %d in the model, %d downloaded), "
                "tp_size %d, %d visible GPU(s), %d tokens\n",
                cfg.n_layers, n_full, n_avail, tp_size, ndev, n_tokens);
    if (tp_size > ndev) { std::printf("tp_size %d > %d GPUs\n", tp_size, ndev); return 1; }

    const std::vector<int> prompt = {1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000};
    const int max_ctx = 16;

    // SPARKINFER_K3_TP_TRACE=1 records rank 0's debug checkpoints in both runs and
    // reports the FIRST one that differs — the divergence point, not its consequences.
    Trace tr_ref, tr_got;
    const bool trace_on = [] {
        const char* e = std::getenv("SPARKINFER_K3_TP_TRACE");
        return e && e[0] == '1';
    }();

    // ---- reference: tp_size 1 ----
    std::vector<std::vector<float>> ref(n_tokens);
    {
        std::vector<int> devs = {0};
        KimiK3TP one;
        if (!kimi_k3_tp_init(g, cfg, opt, devs, max_ctx, one)) {
            std::printf("tp=1 init failed\n"); return 1;
        }
        if (trace_on) one.ranks[0].fwd.debug = make_tracer(&tr_ref);
        for (int t = 0; t < n_tokens; ++t) {
            ref[(size_t)t].assign((size_t)cfg.vocab, 0.0f);
            if (!kimi_k3_tp_forward_token(one, prompt[(size_t)t], ref[(size_t)t].data())) {
                std::printf("tp=1 token %d failed\n", t); return 1;
            }
        }
        std::printf("reference (tp=1): %ld collectives\n", one.n_collectives);
        kimi_k3_tp_free(one);
    }

    // ---- candidate: tp_size N ----
    std::vector<std::vector<float>> got(n_tokens);
    long collectives = 0;
    {
        std::vector<int> devs;
        for (int i = 0; i < tp_size; ++i) devs.push_back(i);
        KimiK3TP many;
        if (!kimi_k3_tp_init(g, cfg, opt, devs, max_ctx, many)) {
            std::printf("tp=%d init failed\n", tp_size); return 1;
        }
        if (trace_on) many.ranks[0].fwd.debug = make_tracer(&tr_got);
        for (int t = 0; t < n_tokens; ++t) {
            got[(size_t)t].assign((size_t)cfg.vocab, 0.0f);
            if (!kimi_k3_tp_forward_token(many, prompt[(size_t)t], got[(size_t)t].data())) {
                std::printf("tp=%d token %d failed\n", tp_size, t); return 1;
            }
        }
        collectives = many.n_collectives;
        kimi_k3_tp_free(many);
    }

    // One collective per MoE layer per token; the leading dense layer has none.
    const long want_coll = (long)(cfg.n_layers - cfg.leading_dense) * n_tokens;
    std::printf("candidate (tp=%d): %ld collectives (expected %ld = %d MoE layers x %d tokens)\n",
                tp_size, collectives, want_coll, cfg.n_layers - cfg.leading_dense, n_tokens);

    int failures = 0;
    if (collectives != want_coll) {
        std::printf("  FAIL: collective count is wrong — a missing reduce leaves a partial "
                    "expert sum, an extra one multiplies a complete tensor\n");
        ++failures;
    }

    for (int t = 0; t < n_tokens; ++t) {
        double max_abs = 0, inf = 0;
        int worst = -1;
        for (int i = 0; i < cfg.vocab; ++i) {
            const double a = ref[(size_t)t][(size_t)i], b = got[(size_t)t][(size_t)i];
            const double d = std::fabs(a - b);
            inf = std::fabs(a) > inf ? std::fabs(a) : inf;
            if (d > max_abs) { max_abs = d; worst = i; }
        }
        const double norm = inf > 0 ? max_abs / inf : 0.0;
        const int a_ref = argmax(ref[(size_t)t]), a_got = argmax(got[(size_t)t]);
        const bool ok = (a_ref == a_got) && norm < 1e-4;
        std::printf("  token %d: argmax ref=%d tp=%d  max_abs %.3e  /|inf| %.3e  %s\n",
                    t, a_ref, a_got, max_abs, norm, ok ? "OK" : "MISMATCH");
        if (!ok) {
            ++failures;
            if (worst >= 0)
                std::printf("    worst logit %d: ref %.9g  tp %.9g\n", worst,
                            ref[(size_t)t][(size_t)worst], got[(size_t)t][(size_t)worst]);
        }
    }

    if (trace_on) {
        std::printf("\n[trace] %zu ref checkpoints, %zu tp checkpoints\n",
                    tr_ref.name.size(), tr_got.name.size());
        const size_t n = std::min(tr_ref.name.size(), tr_got.name.size());
        int shown = 0;
        for (size_t i = 0; i < n && shown < 12; ++i) {
            if (tr_ref.name[i] != tr_got.name[i]) {
                std::printf("  [%zu] TAG MISMATCH ref=%s tp=%s\n", i,
                            tr_ref.name[i].c_str(), tr_got.name[i].c_str());
                break;
            }
            const double a = tr_ref.l1[i], b = tr_got.l1[i];
            const double rel = std::fabs(a) > 1e-9 ? std::fabs(a - b) / std::fabs(a) : 0.0;
            if (rel > 1e-6) {
                std::printf("  [%3zu] %-22s ref_l1 %14.6f  tp_l1 %14.6f  rel %.3e  <-- DIVERGES\n",
                            i, tr_ref.name[i].c_str(), a, b, rel);
                ++shown;
            }
        }
        if (shown == 0) std::printf("  every checkpoint agrees to 1e-6 relative\n");
    }

    std::printf("\n%s: tp=%d %s the single-GPU decode over %d token(s)\n",
                failures ? "FAIL" : "PASS", tp_size,
                failures ? "DIVERGES from" : "reproduces", n_tokens);
    return failures ? 1 : 0;
}
