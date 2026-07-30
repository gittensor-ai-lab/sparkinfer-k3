// Expert parallelism: 8 disjoint expert bands, summed, must equal the unsharded MoE.
//
//   kimi_k3_tp_moe_check <first-shard.gguf> [tp_size]
//
// This is the correctness proof for the whole expert-parallel scheme. Rank r loads
// only experts [r*896/tp, (r+1)*896/tp) and evaluates only the selected ids inside
// that band, so what it leaves in the dispatch accumulator is a PARTIAL sum. The
// all-reduce is supposed to turn those partials back into exactly what one GPU
// holding all 896 experts would have computed. Either that identity holds or the
// model is quietly wrong — a dropped band loses a fraction of every MoE layer's
// output, which reads as mild quality loss rather than as a bug.
//
// Everything is simulated on ONE device: rank r's weights are loaded, its partial is
// read back to the host, and the host sums them. That is the same arithmetic the
// collective does, without needing 8 GPUs to test the band logic.
//
// TOLERANCE, NOT BITWISE — and this one genuinely is floating point, not sloppiness.
// The reference accumulates the top_k in k order (k = 0..15). The banded run
// accumulates each rank's subset in k order and then sums across ranks, which is a
// different association of the same 16 addends. Real TP has exactly this property;
// llama.cpp's single-GPU answer and an 8-way TP answer differ in the last bits for
// the same reason. What must NOT differ is the magnitude.

#include "sparkinfer/gguf.h"
#include "sparkinfer/models/kimi_k3.h"
#include "sparkinfer/models/kimi_k3_config.h"
#include "sparkinfer/models/kimi_k3_gguf_manifest.h"
#include "sparkinfer/tp/shard.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace sparkinfer;

namespace {

// Run Attn + FfnPartial for `layer` and return this rank's expert-dispatch partial.
bool run_partial(const GGUF& g, const KimiK3Config& cfg, const K3PlanOptions& opt,
                 int layer, int n_layers, int tp_size, int rank,
                 const std::vector<float>& host_in, std::vector<float>& out_partial) {
    KimiK3Weights w;
    w.policy = KimiK3Weights::ShardPolicy::ExpertsOnly;
    w.shard.tp_size = tp_size;
    w.shard.rank = rank;
    w.shard.hidden = cfg.hidden;
    w.shard.n_experts_total = cfg.n_experts;
    w.shard.experts_sharded = tp_size > 1 && (cfg.n_experts % tp_size) == 0;
    w.shard.n_experts = w.shard.experts_sharded ? cfg.n_experts / tp_size : cfg.n_experts;
    w.shard.expert_band = w.shard.experts_sharded
                              ? tp::even_band(cfg.n_experts, tp_size, rank)
                              : tp::Band{0, cfg.n_experts};
    if (tp_size > 1 && !w.shard.experts_sharded) {
        std::printf("  %d experts do not divide %d ranks\n", cfg.n_experts, tp_size);
        return false;
    }

    if (!kimi_k3_load_weights(g, cfg, opt, w, 0, n_layers - 1)) return false;

    KimiK3RuntimeState st;
    if (!kimi_k3_alloc_state(cfg, 8, st, 0, n_layers - 1)) return false;
    KimiK3Forward fwd;
    fwd.cfg = &cfg; fwd.w = &w; fwd.state = &st; fwd.opt = opt; fwd.stream = nullptr;
    if (!kimi_k3_forward_alloc_scratch(cfg, fwd)) return false;

    const int H = cfg.hidden;
    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, (size_t)H * sizeof(float));
    cudaMalloc(&d_out, (size_t)H * sizeof(float));
    cudaMemcpy(d_in, host_in.data(), (size_t)H * sizeof(float), cudaMemcpyHostToDevice);

    kimi_k3_reset_state(st);
    bool ok = kimi_k3_forward_layer_phase(fwd, layer, K3LayerPhase::Attn, d_in, d_out) &&
              kimi_k3_forward_layer_phase(fwd, layer, K3LayerPhase::FfnPartial, d_in, d_out);
    cudaDeviceSynchronize();

    int n = 0;
    float* partial = kimi_k3_partial_buffer(fwd, layer, K3LayerPhase::FfnPartial, &n);
    if (ok && partial && n > 0) {
        out_partial.assign((size_t)n, 0.0f);
        cudaMemcpy(out_partial.data(), partial, (size_t)n * sizeof(float),
                   cudaMemcpyDeviceToHost);
    } else {
        ok = false;
    }

    cudaFree(d_in); cudaFree(d_out);
    kimi_k3_forward_free_scratch(fwd);
    kimi_k3_free_state(st);
    kimi_k3_free_weights(w);
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <first-shard.gguf> [tp_size]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const int tp_size = argc > 2 ? std::atoi(argv[2]) : 8;

    GGUF g; KimiK3Config cfg; KimiK3LayerCoverage cov;
    if (!kimi_k3_load_partial(path.c_str(), cfg, g, &cov)) { std::printf("load failed\n"); return 1; }
    int n_avail = 0;
    while (n_avail < cfg.n_layers && cov.layer_complete[n_avail]) ++n_avail;

    // The first MoE layer: layer 0 is the leading DENSE block and has no experts.
    const int layer = cfg.leading_dense;
    if (n_avail <= layer) { std::printf("need layer %d complete, have %d\n", layer, n_avail); return 1; }
    const int n_layers = layer + 1;

    K3PlanOptions opt;
    {
        const std::string p = "blk." + std::to_string(layer) + ".";
        opt.has_q_lora     = g.tensor((p + "attn_q_a.weight").c_str()) != nullptr;
        opt.has_fused_kv_b = g.tensor((p + "attn_kv_b.weight").c_str()) != nullptr;
        opt.has_attn_gate  = g.tensor((p + "attn_gate.weight").c_str()) != nullptr;
        opt.has_shared_experts = g.tensor((p + "ffn_gate_shexp.weight").c_str()) != nullptr;
        opt.has_routed_norm    = g.tensor((p + "ffn_routed_norm.weight").c_str()) != nullptr;
    }
    std::printf("expert-parallel check: layer %d, %d experts, top_k %d, tp_size %d "
                "(%d experts/rank)\n",
                layer, cfg.n_experts, cfg.top_k, tp_size, cfg.n_experts / tp_size);

    const int H = cfg.hidden;
    std::vector<float> host_in((size_t)H);
    for (int i = 0; i < H; ++i)
        host_in[(size_t)i] = 0.05f * std::sin(0.017f * (float)i) + 0.01f * (float)((i % 7) - 3);

    // Reference: every expert on one rank.
    std::vector<float> ref;
    if (!run_partial(g, cfg, opt, layer, n_layers, 1, 0, host_in, ref)) {
        std::printf("reference run failed\n"); return 1;
    }
    std::printf("reference (tp=1): %zu elems\n", ref.size());

    // Banded: sum the partials.
    std::vector<double> acc(ref.size(), 0.0);
    for (int r = 0; r < tp_size; ++r) {
        std::vector<float> part;
        if (!run_partial(g, cfg, opt, layer, n_layers, tp_size, r, host_in, part)) {
            std::printf("rank %d run failed\n", r); return 1;
        }
        if (part.size() != ref.size()) {
            std::printf("rank %d produced %zu elems, want %zu\n", r, part.size(), ref.size());
            return 1;
        }
        double nz = 0;
        for (size_t i = 0; i < part.size(); ++i) { acc[i] += part[i]; nz += std::fabs(part[i]); }
        std::printf("  rank %d: experts [%4d,%4d)  |partial|_1 = %.4f\n",
                    r, tp::even_band(cfg.n_experts, tp_size, r).offset,
                    tp::even_band(cfg.n_experts, tp_size, r).end(), nz);
    }

    // Compare, NORMALISED BY THE VECTOR'S OWN SCALE.
    //
    // Not per-element relative error: this is an activation, so a good fraction of
    // its 3584 elements sit near zero, and dividing a ~1-ulp absolute difference by a
    // near-zero reference manufactures a huge "relative" error out of nothing. (An
    // earlier version of this check used a 1e-6 floor and reported 3.3e-4 for a
    // result whose worst element was 0.0201550107 vs 0.0201550078 — a 1-ulp
    // difference dressed up as a failure.) The meaningful question for a summation-
    // order change is how the largest deviation compares to the largest value.
    double max_abs = 0, ref_inf = 0, ref_l1 = 0;
    int worst = -1;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double a = ref[i], b = acc[i];
        const double d = std::fabs(a - b);
        ref_l1  += std::fabs(a);
        ref_inf  = std::fabs(a) > ref_inf ? std::fabs(a) : ref_inf;
        if (d > max_abs) { max_abs = d; worst = (int)i; }
    }
    const double norm_err = ref_inf > 0 ? max_abs / ref_inf : 0.0;
    std::printf("\nsum(bands) vs unsharded:\n");
    std::printf("  max_abs   %.3e\n", max_abs);
    std::printf("  |ref|_inf %.6f   |ref|_1 %.4f\n", ref_inf, ref_l1);
    std::printf("  max_abs / |ref|_inf = %.3e   (float32 eps = 1.19e-07)\n", norm_err);
    if (worst >= 0)
        std::printf("  worst elem %d: ref %.9g  banded %.9g\n", worst, ref[worst], acc[worst]);

    // A DROPPED BAND is not a rounding error — it removes whole experts, so the
    // deviation lands at percent scale, four orders of magnitude above this bound.
    // 1e-5 of the peak is ~100x float32 eps: loose enough for any reassociation of
    // 16 addends, far too tight for a missing eighth of the mixture.
    const bool pass = norm_err < 1e-5 && ref_l1 > 0.0;
    std::printf("\n%s: expert bands %s the unsharded MoE output\n",
                pass ? "PASS" : "FAIL",
                pass ? "reconstruct (to float32 summation-order noise)"
                     : "DO NOT reconstruct");
    return pass ? 0 : 1;
}
