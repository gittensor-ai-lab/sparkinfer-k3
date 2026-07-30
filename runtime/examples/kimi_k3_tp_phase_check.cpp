// Phase-split equivalence: running a layer as ONE call must equal running it as
// THREE, bit for bit.
//
//   kimi_k3_tp_phase_check <first-shard.gguf> [n_layers]
//
// The TP forward cannot call kimi_k3_forward_layer — it has to stop at each
// collective, let every rank catch up, reduce, and resume. So the layer was split
// into Attn / FfnPartial / FfnFinish. At tp_size 1 no collective is inserted between
// them, which means the split is a pure refactor and ANY difference is a bug in the
// split itself: a buffer that was live across a boundary and got clobbered, a
// bookkeeping side effect (the residual bank push, the KV cache write, position) run
// twice or not at all.
//
// BITWISE, not approximate. The same kernels run in the same order on the same
// stream, so the only correct answer is an exact match — a tolerance here would hide
// exactly the class of bug this exists to catch.
//
// The state is reset between the two runs because a K3 layer MUTATES state: the KDA
// conv window and delta matrix, the MLA KV cache row, the cross-layer residual bank,
// and the position counter. Replaying without a reset compares the first token
// against the second.

#include "sparkinfer/gguf.h"
#include "sparkinfer/models/kimi_k3.h"
#include "sparkinfer/models/kimi_k3_config.h"
#include "sparkinfer/models/kimi_k3_gguf_manifest.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace sparkinfer;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <first-shard.gguf> [n_layers]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const int want_layers = argc > 2 ? std::atoi(argv[2]) : 4;

    GGUF g;
    KimiK3Config cfg;
    KimiK3LayerCoverage cov;
    if (!kimi_k3_load_partial(path.c_str(), cfg, g, &cov)) {
        std::printf("load failed\n");
        return 1;
    }
    int n_avail = 0;
    while (n_avail < cfg.n_layers && cov.layer_complete[n_avail]) ++n_avail;
    if (n_avail < 1) { std::printf("need >=1 complete layer, have %d\n", n_avail); return 1; }
    const int n_layers = n_avail < want_layers ? n_avail : want_layers;
    std::printf("phase equivalence over %d layer(s) (%d complete in the downloaded shards)\n",
                n_layers, n_avail);

    K3PlanOptions opt;
    {
        const int probe = n_avail > 3 ? 3 : 0;
        const std::string p = "blk." + std::to_string(probe) + ".";
        opt.has_q_lora     = g.tensor((p + "attn_q_a.weight").c_str()) != nullptr;
        opt.has_fused_kv_b = g.tensor((p + "attn_kv_b.weight").c_str()) != nullptr;
        opt.has_attn_gate  = g.tensor((p + "attn_gate.weight").c_str()) != nullptr;
    }
    const int moe0 = cfg.leading_dense;
    if (moe0 < n_avail) {
        const std::string p = "blk." + std::to_string(moe0) + ".";
        opt.has_shared_experts = g.tensor((p + "ffn_gate_shexp.weight").c_str()) != nullptr;
        opt.has_routed_norm    = g.tensor((p + "ffn_routed_norm.weight").c_str()) != nullptr;
    }

    const int H = cfg.hidden;
    const int max_ctx = 8;

    KimiK3Weights w;
    if (!kimi_k3_load_weights(g, cfg, opt, w, 0, n_layers - 1)) {
        std::printf("weight load failed\n"); return 1;
    }
    KimiK3RuntimeState st;
    if (!kimi_k3_alloc_state(cfg, max_ctx, st, 0, n_layers - 1)) {
        std::printf("state alloc failed\n"); return 1;
    }
    KimiK3Forward fwd;
    fwd.cfg = &cfg; fwd.w = &w; fwd.state = &st; fwd.opt = opt; fwd.stream = nullptr;
    if (!kimi_k3_forward_alloc_scratch(cfg, fwd)) {
        std::printf("scratch alloc failed\n"); return 1;
    }

    // A fixed, non-trivial input. Constant or zero input would let a dropped phase
    // pass by symmetry.
    std::vector<float> host_in((size_t)H);
    for (int i = 0; i < H; ++i)
        host_in[(size_t)i] = 0.05f * std::sin(0.017f * (float)i) + 0.01f * (float)((i % 7) - 3);

    float *d_in = nullptr, *d_out_all = nullptr, *d_out_phased = nullptr;
    cudaMalloc(&d_in, (size_t)H * sizeof(float));
    cudaMalloc(&d_out_all, (size_t)H * sizeof(float));
    cudaMalloc(&d_out_phased, (size_t)H * sizeof(float));
    cudaMemcpy(d_in, host_in.data(), (size_t)H * sizeof(float), cudaMemcpyHostToDevice);

    int failures = 0;
    for (int layer = 0; layer < n_layers; ++layer) {
        // ---- run A: one call ----
        kimi_k3_reset_state(st);
        if (!kimi_k3_forward_layer(fwd, layer, d_in, d_out_all)) {
            std::printf("  layer %d: All failed\n", layer); return 1;
        }
        cudaDeviceSynchronize();
        std::vector<float> a((size_t)H);
        cudaMemcpy(a.data(), d_out_all, (size_t)H * sizeof(float), cudaMemcpyDeviceToHost);

        // ---- run B: three calls, exactly where the collectives go ----
        kimi_k3_reset_state(st);
        bool ok = kimi_k3_forward_layer_phase(fwd, layer, K3LayerPhase::Attn,
                                              d_in, d_out_phased);
        // At tp_size 1 the collective is the identity, so nothing happens here — but
        // this is the point at which the driver would reduce, and the partial buffer
        // it would hand the collective is queried the same way.
        int n_partial = 0;
        float* partial = kimi_k3_partial_buffer(fwd, layer, K3LayerPhase::Attn, &n_partial);
        if (!partial || n_partial != H) {
            std::printf("  layer %d: attn partial buffer is %p/%d, want non-null/%d\n",
                        layer, (void*)partial, n_partial, H);
            ++failures;
        }
        ok = ok && kimi_k3_forward_layer_phase(fwd, layer, K3LayerPhase::FfnPartial,
                                               d_in, d_out_phased);
        partial = kimi_k3_partial_buffer(fwd, layer, K3LayerPhase::FfnPartial, &n_partial);
        const int want_w = (layer < cfg.leading_dense) ? H : cfg.expert_latent;
        if (!partial || n_partial != want_w) {
            std::printf("  layer %d: ffn partial buffer is %p/%d, want non-null/%d\n",
                        layer, (void*)partial, n_partial, want_w);
            ++failures;
        }
        ok = ok && kimi_k3_forward_layer_phase(fwd, layer, K3LayerPhase::FfnFinish,
                                               d_in, d_out_phased);
        if (!ok) { std::printf("  layer %d: phased run failed\n", layer); return 1; }
        cudaDeviceSynchronize();
        std::vector<float> b((size_t)H);
        cudaMemcpy(b.data(), d_out_phased, (size_t)H * sizeof(float), cudaMemcpyDeviceToHost);

        // ---- compare, bitwise ----
        int diff = 0; int first = -1; float fa = 0, fb = 0;
        for (int i = 0; i < H; ++i) {
            if (std::memcmp(&a[(size_t)i], &b[(size_t)i], sizeof(float)) != 0) {
                if (first < 0) { first = i; fa = a[(size_t)i]; fb = b[(size_t)i]; }
                ++diff;
            }
        }
        const char* kind = (layer < cfg.leading_dense) ? "dense" : "moe";
        if (diff == 0) {
            std::printf("  layer %-3d (%s, %s)  IDENTICAL\n", layer,
                        cfg.is_kda_layer(layer) ? "kda" : "mla", kind);
        } else {
            std::printf("  layer %-3d (%s, %s)  DIFFERS in %d/%d elems; first [%d] "
                        "All=%.9g phased=%.9g\n", layer,
                        cfg.is_kda_layer(layer) ? "kda" : "mla", kind, diff, H, first, fa, fb);
            ++failures;
        }
    }

    cudaFree(d_in); cudaFree(d_out_all); cudaFree(d_out_phased);
    kimi_k3_forward_free_scratch(fwd);
    kimi_k3_free_state(st);
    kimi_k3_free_weights(w);

    std::printf("\n%s: %d layer(s), %d failure(s)\n",
                failures ? "FAIL" : "PASS", n_layers, failures);
    return failures ? 1 : 0;
}
