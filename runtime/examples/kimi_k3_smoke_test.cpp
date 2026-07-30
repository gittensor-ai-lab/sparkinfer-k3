// First real run of the K3 executor against REAL weights (whatever shards are
// present). Smoke test only: does it load, run one layer without crashing, and
// produce finite numbers? Numerical comparison against llama.cpp is the next step,
// once this confirms the plumbing itself is sound.
#include "sparkinfer/models/kimi_k3.h"
#include "sparkinfer/models/kimi_k3_gguf_manifest.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace sparkinfer;

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1]
        : "/workspace/models_k3/UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf";

    GGUF g;
    KimiK3Config cfg;
    KimiK3LayerCoverage cov;
    if (!kimi_k3_load_partial(path, cfg, g, &cov)) {
        std::printf("FAILED to load config/partial GGUF\n");
        return 1;
    }
    std::printf("model: %d layers (%d KDA/%d MLA), hidden %d, %d/%d shards\n",
                cfg.n_layers, cfg.n_kda_layers(), cfg.n_mla_layers(), cfg.hidden,
                g.shards_loaded(), g.split_count());
    std::printf("coverage: %d/%d layers complete; first complete KDA=%d MLA=%d\n",
                cov.n_complete, cfg.n_layers, cov.first_complete_kda, cov.first_complete_mla);

    int layer = cov.first_complete_kda >= 0 ? cov.first_complete_kda : cov.first_complete_mla;
    if (argc > 2) layer = std::atoi(argv[2]);
    if (layer < 0 || !cov.layer_complete[layer]) {
        std::printf("no complete layer available to test (asked for layer %d)\n", layer);
        return 1;
    }
    std::printf("testing layer %d (%s)\n", layer, cfg.is_kda_layer(layer) ? "KDA" : "MLA");

    // Auto-detect layout options from tensor presence, same convention as
    // kimi_k3_plan_check.cpp.
    K3PlanOptions opt;
    {
        const std::string p = "blk." + std::to_string(layer) + ".";
        opt.has_q_lora     = g.tensor((p + "attn_q_a.weight").c_str()) != nullptr;
        opt.has_fused_kv_b = g.tensor((p + "attn_kv_b.weight").c_str()) != nullptr;
        opt.has_attn_gate  = g.tensor((p + "attn_gate.weight").c_str()) != nullptr;
        opt.has_shared_experts = g.tensor((p + "ffn_gate_shexp.weight").c_str()) != nullptr;
        opt.has_routed_norm    = g.tensor((p + "ffn_routed_norm.weight").c_str()) != nullptr;
    }
    std::printf("layout: q_lora=%d fused_kv_b=%d attn_gate=%d shared_experts=%d routed_norm=%d\n",
                opt.has_q_lora, opt.has_fused_kv_b, opt.has_attn_gate,
                opt.has_shared_experts, opt.has_routed_norm);

    KimiK3Weights w;
    if (!kimi_k3_load_weights(g, cfg, opt, w, layer, layer)) {
        std::printf("FAILED to load layer %d weights\n", layer);
        return 1;
    }
    std::printf("weights loaded OK\n");

    KimiK3RuntimeState state;
    if (!kimi_k3_alloc_state(cfg, /*max_ctx=*/64, state)) {
        std::printf("FAILED to alloc state\n");
        return 1;
    }

    KimiK3Forward fwd;
    fwd.cfg = &cfg;
    fwd.w = &w;
    fwd.state = &state;
    fwd.opt = opt;
    fwd.stream = 0;
    if (!kimi_k3_forward_alloc_scratch(cfg, fwd)) {
        std::printf("FAILED to alloc scratch\n");
        return 1;
    }

    // Random hidden_in — real activation stats plausible for a residual stream.
    std::mt19937 rng(20260730);
    std::normal_distribution<float> N(0.f, 1.f);
    std::vector<float> h_in(cfg.hidden);
    for (auto& v : h_in) v = N(rng);

    float *d_in, *d_out;
    cudaMalloc(&d_in, cfg.hidden * sizeof(float));
    cudaMalloc(&d_out, cfg.hidden * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), cfg.hidden * sizeof(float), cudaMemcpyHostToDevice);

    const bool ok = kimi_k3_forward_layer(fwd, layer, d_in, d_out);
    cudaError_t err = cudaGetLastError();
    cudaDeviceSynchronize();
    cudaError_t err2 = cudaGetLastError();
    std::printf("forward_layer returned %s, cuda errors: launch=%s sync=%s\n",
                ok ? "true" : "false", cudaGetErrorString(err), cudaGetErrorString(err2));
    if (!ok) return 1;

    std::vector<float> h_out(cfg.hidden);
    cudaMemcpy(h_out.data(), d_out, cfg.hidden * sizeof(float), cudaMemcpyDeviceToHost);

    double mn = h_out[0], mx = h_out[0], sum = 0, sumsq = 0;
    int n_nan = 0, n_inf = 0;
    for (float v : h_out) {
        if (std::isnan(v)) ++n_nan;
        if (std::isinf(v)) ++n_inf;
        mn = std::fmin(mn, (double)v);
        mx = std::fmax(mx, (double)v);
        sum += v;
        sumsq += (double)v * v;
    }
    const double mean = sum / cfg.hidden;
    const double rms = std::sqrt(sumsq / cfg.hidden);
    std::printf("\nlayer %d output stats: min=%.4f max=%.4f mean=%.4f rms=%.4f nan=%d inf=%d\n",
                layer, mn, mx, mean, rms, n_nan, n_inf);
    std::printf("first 8: ");
    for (int i = 0; i < 8; ++i) std::printf("%.4f ", h_out[i]);
    std::printf("\n\n%s\n", (n_nan == 0 && n_inf == 0 && rms > 0) ?
                "PASS: layer forward ran, output is finite and non-degenerate" : "FAIL");
    return (n_nan == 0 && n_inf == 0 && rms > 0) ? 0 : 1;
}
