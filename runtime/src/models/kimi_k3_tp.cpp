// Tensor-parallel Kimi K3 decode. See kimi_k3_tp.h for the scope and the reasoning.

#include "sparkinfer/models/kimi_k3_tp.h"

#include "sparkinfer/kernels/kimi_k3.h"
#include "sparkinfer/tp/shard.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <utility>

namespace k3k = sparkinfer::kernels::k3;

namespace sparkinfer {

namespace {

// Token row gather. Same contract as the single-GPU path: token_embd is replicated,
// so every rank does this identically and no broadcast is needed.
bool embed_token(const KimiK3Weights& w, const KimiK3Config& cfg, int token_id,
                 float* x, cudaStream_t stream) {
    if (!w.token_embd.ok()) return false;
    long row_bytes = 0;
    if (w.token_embd.type == 0)      row_bytes = (long)cfg.hidden * sizeof(float);
    else if (w.token_embd.type == 8) row_bytes = (long)(cfg.hidden / 32) * 34;
    else return false;
    const char* base = (const char*)w.token_embd.data + (size_t)token_id * row_bytes;
    return k3k::dequant_f32_by_type(x, base, cfg.hidden, w.token_embd.type, stream);
}

}  // namespace

bool kimi_k3_tp_init(const GGUF& g, const KimiK3Config& cfg, const K3PlanOptions& opt,
                    const std::vector<int>& devices, int max_ctx, KimiK3TP& out) {
    const int tp_size = (int)devices.size();
    if (tp_size < 1) return false;
    if (tp_size > 1 && (cfg.n_experts % tp_size) != 0) {
        std::fprintf(stderr, "[k3-tp] %d experts do not divide %d ranks — a ragged "
                             "expert band would give every rank a different launch "
                             "geometry\n", cfg.n_experts, tp_size);
        return false;
    }

    out.cfg = cfg;
    out.opt = opt;
    out.ranks.clear();
    out.ranks.resize((size_t)tp_size);
    out.streams.assign((size_t)tp_size, nullptr);
    out.reduce_bufs.assign((size_t)tp_size, nullptr);
    out.n_collectives = 0;

    for (int r = 0; r < tp_size; ++r) {
        KimiK3TPRank& R = out.ranks[(size_t)r];
        R.device = devices[(size_t)r];
        R.rank = r;
        if (cudaSetDevice(R.device) != cudaSuccess) return false;
        if (cudaStreamCreate(&R.stream) != cudaSuccess) return false;
        out.streams[(size_t)r] = R.stream;

        R.weights.policy = KimiK3Weights::ShardPolicy::ExpertsOnly;
        R.weights.shard.tp_size = tp_size;
        R.weights.shard.rank = r;
        R.weights.shard.hidden = cfg.hidden;
        R.weights.shard.n_experts_total = cfg.n_experts;
        R.weights.shard.experts_sharded = tp_size > 1;
        R.weights.shard.n_experts = cfg.n_experts / tp_size;
        R.weights.shard.expert_band = tp_size > 1
            ? tp::even_band(cfg.n_experts, tp_size, r)
            : tp::Band{0, cfg.n_experts};

        if (!kimi_k3_load_weights(g, cfg, opt, R.weights, 0, cfg.n_layers - 1)) {
            std::fprintf(stderr, "[k3-tp] rank %d: weight load failed\n", r);
            return false;
        }
        if (!kimi_k3_alloc_state(cfg, max_ctx, R.state)) return false;

        R.fwd.cfg = &out.cfg;
        R.fwd.w = &R.weights;
        R.fwd.state = &R.state;
        R.fwd.opt = out.opt;
        R.fwd.stream = R.stream;
        if (!kimi_k3_forward_alloc_scratch(cfg, R.fwd)) return false;

        if (cudaMalloc(&R.x, (size_t)cfg.hidden * sizeof(float)) != cudaSuccess) return false;
        if (cudaMalloc(&R.x_next, (size_t)cfg.hidden * sizeof(float)) != cudaSuccess) return false;
        if (r == 0 &&
            cudaMalloc(&R.logits, (size_t)cfg.vocab * sizeof(float)) != cudaSuccess) return false;

        std::fprintf(stderr, "[k3-tp] rank %d: device %d, experts [%d,%d)\n",
                     r, R.device, R.weights.shard.expert_band.offset,
                     R.weights.shard.expert_band.end());
    }

    // The collective. need_f32 because K3's residual stream is f32; max_count is the
    // widest payload the forward will reduce, which is the expert_latent partial.
    std::string requested_env;
    if (const char* e = std::getenv("SPARKINFER_TP_BACKEND")) requested_env = e;
    std::string why;
    const tp::Backend requested = tp::backend_from_string(requested_env, &why);
    if (!why.empty()) std::fprintf(stderr, "[k3-tp] %s\n", why.c_str());

    std::string err;
    out.coll = tp::make_collective(devices, requested, &err,
                                   /*max_count=*/(size_t)cfg.expert_latent,
                                   /*need_f32=*/true);
    if (!out.coll) {
        std::fprintf(stderr, "[k3-tp] no collective: %s\n", err.c_str());
        return false;
    }
    if (tp_size > 1 && out.coll->backend() == tp::Backend::None) {
        // A no-op collective at tp>1 leaves every rank holding its own expert band's
        // partial sum and calls it the answer. Refuse rather than emit fluent noise.
        std::fprintf(stderr, "[k3-tp] FATAL: fell back to the no-op collective at "
                             "tp_size %d: %s\n", tp_size, err.c_str());
        return false;
    }
    if (!out.coll->supports_f32()) {
        std::fprintf(stderr, "[k3-tp] FATAL: backend %s cannot reduce f32\n",
                     tp::backend_name(out.coll->backend()));
        return false;
    }
    if (out.coll->owns_buffers()) {
        // Mode B needs the caller to stage through reduce_in/reduce_out. The forward
        // reduces a buffer the executor owns, so that path would need a copy in and
        // out per collective — not wired, and silently staging it would erase the
        // reason to use those backends. need_f32 already steers away from them.
        std::fprintf(stderr, "[k3-tp] FATAL: %s uses collective-owned buffers, which "
                             "this forward does not stage through\n",
                     tp::backend_name(out.coll->backend()));
        return false;
    }
    return true;
}

bool kimi_k3_tp_forward_token(KimiK3TP& p, int token_id, float* out_logits) {
    if (p.ranks.empty() || !p.coll) return false;
    const KimiK3Config& cfg = p.cfg;
    const int H = cfg.hidden;
    const int tp_size = (int)p.ranks.size();

    // The cross-layer residual bank is PER TOKEN on every rank — same lifetime rule
    // as the single-GPU path, and forgetting it here fails on token 2 rather than
    // token 1, because max_ckpt is exactly one token's worth of checkpoints.
    for (auto& R : p.ranks) R.state.n_ckpt = 0;

    for (auto& R : p.ranks) {
        if (cudaSetDevice(R.device) != cudaSuccess) return false;
        if (!embed_token(R.weights, cfg, token_id, R.x, R.stream)) return false;
    }

    for (int layer = 0; layer < cfg.n_layers; ++layer) {
        const bool is_moe = layer >= cfg.leading_dense;

        // --- phase 1 + 2 on every rank -------------------------------------
        // Under ExpertsOnly, attention is replicated, so no collective separates
        // these two phases. They are still issued as distinct calls because the
        // phase boundary is where a fully-sharded build WOULD reduce, and keeping
        // the call sites is what makes that a one-line change rather than a
        // re-split of the forward.
        for (auto& R : p.ranks) {
            if (cudaSetDevice(R.device) != cudaSuccess) return false;
            if (!kimi_k3_forward_layer_phase(R.fwd, layer, K3LayerPhase::Attn,
                                             R.x, R.x_next)) return false;
            if (!kimi_k3_forward_layer_phase(R.fwd, layer, K3LayerPhase::FfnPartial,
                                             R.x, R.x_next)) return false;
        }

        // --- the collective -------------------------------------------------
        // MoE layers only. The leading dense layer's FFN is replicated under this
        // policy, so it holds a complete result already and reducing it would
        // multiply it by tp_size.
        if (is_moe && tp_size > 1) {
            int count = 0;
            for (int r = 0; r < tp_size; ++r) {
                int n = 0;
                float* buf = kimi_k3_partial_buffer(p.ranks[(size_t)r].fwd, layer,
                                                    K3LayerPhase::FfnPartial, &n);
                if (!buf || n <= 0) return false;
                if (r == 0) count = n;
                else if (n != count) return false;   // ranks must agree on the width
                p.reduce_bufs[(size_t)r] = buf;
            }
            // Enqueued on each rank's own stream, after that rank's dispatch, so the
            // ordering is stream-ordered — no host barrier, and no cudaDeviceSynchronize
            // on the critical path. The GROUP form is mandatory: a single host thread
            // looping per-rank allreduce deadlocks (see collective.h).
            // SPARKINFER_K3_TP_HOST_REDUCE=1 replaces the collective with an explicit
            // device->host->sum->device round trip. Slow and only for bisecting: it
            // isolates "the reduce is wrong" from "the reduce is right but something
            // around it is", which are otherwise indistinguishable from the logits.
            static const bool host_reduce = [] {
                const char* e = std::getenv("SPARKINFER_K3_TP_HOST_REDUCE");
                return e && e[0] == '1';
            }();
            if (host_reduce) {
                std::vector<float> sum((size_t)count, 0.0f), tmp((size_t)count);
                for (int r = 0; r < tp_size; ++r) {
                    cudaSetDevice(p.ranks[(size_t)r].device);
                    cudaStreamSynchronize(p.ranks[(size_t)r].stream);
                    cudaMemcpy(tmp.data(), p.reduce_bufs[(size_t)r],
                               (size_t)count * sizeof(float), cudaMemcpyDeviceToHost);
                    for (int i = 0; i < count; ++i) sum[(size_t)i] += tmp[(size_t)i];
                }
                for (int r = 0; r < tp_size; ++r) {
                    cudaSetDevice(p.ranks[(size_t)r].device);
                    cudaMemcpy(p.reduce_bufs[(size_t)r], sum.data(),
                               (size_t)count * sizeof(float), cudaMemcpyHostToDevice);
                }
            } else if (!p.coll->allreduce_f32_group(p.reduce_bufs, (size_t)count, p.streams)) {
                std::fprintf(stderr, "[k3-tp] all-reduce failed at layer %d\n", layer);
                return false;
            }
            ++p.n_collectives;
        }

        // --- phase 3 on every rank ------------------------------------------
        for (auto& R : p.ranks) {
            if (cudaSetDevice(R.device) != cudaSuccess) return false;
            if (!kimi_k3_forward_layer_phase(R.fwd, layer, K3LayerPhase::FfnFinish,
                                             R.x, R.x_next)) return false;
            std::swap(R.x, R.x_next);
        }
    }

    // --- head: every rank holds the identical hidden state, so rank 0 suffices ---
    KimiK3TPRank& R0 = p.ranks[0];
    if (cudaSetDevice(R0.device) != cudaSuccess) return false;
    const KimiK3Weights& w = R0.weights;

    if (cfg.attn_res_block_size > 0) {
        if (!w.has_output_res_score || !w.output_res_score.ok()) return false;
        k3k::attn_res_mix_f32(R0.x_next, R0.state.res_bank, R0.x,
                              (const float*)w.output_res_score.data, H,
                              R0.state.n_ckpt, cfg.rms_eps, R0.stream);
        std::swap(R0.x, R0.x_next);
    }
    if (!w.output_norm.ok()) return false;
    k3k::rms_norm_f32(R0.x_next, R0.x, (const float*)w.output_norm.data, H,
                      cfg.rms_eps, R0.stream);
    if (!w.output.ok()) return false;
    if (!k3k::k3_proj_f32(R0.logits, R0.x_next, w.output.data, w.output.type,
                          cfg.vocab, H, R0.stream))
        return false;

    if (cudaStreamSynchronize(R0.stream) != cudaSuccess) return false;
    if (cudaMemcpy(out_logits, R0.logits, (size_t)cfg.vocab * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return false;

    // Advance every rank's position together. They run the same attention, so a rank
    // whose position drifted would index a different KV row for the same token.
    for (auto& R : p.ranks) ++R.state.position;
    return true;
}

void kimi_k3_tp_free(KimiK3TP& p) {
    for (auto& R : p.ranks) {
        cudaSetDevice(R.device);
        if (R.x) cudaFree(R.x);
        if (R.x_next) cudaFree(R.x_next);
        if (R.logits) cudaFree(R.logits);
        kimi_k3_forward_free_scratch(R.fwd);
        kimi_k3_free_state(R.state);
        kimi_k3_free_weights(R.weights);
        if (R.stream) cudaStreamDestroy(R.stream);
    }
    p.ranks.clear();
    p.streams.clear();
    p.reduce_bufs.clear();
    p.coll.reset();
}

}  // namespace sparkinfer
