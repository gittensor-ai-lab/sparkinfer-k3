#include "sparkinfer/models/kimi_k3.h"

#include "sparkinfer/kernels/kimi_k3.h"
#include "sparkinfer/models/kimi_k3_gguf_manifest.h"
#include "sparkinfer/tp/weight_residency.h"   // plan_tensor_residency for the sharded loader

#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <utility>

namespace sparkinfer {

namespace k3k = sparkinfer::kernels::k3;

namespace {

std::string blk(int i, const char* suffix) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "blk.%d.%s", i, suffix);
    return std::string(buf);
}

bool env_one(const char* name) {
    const char* value = std::getenv(name);
    return value && value[0] == '1';
}

bool env_zero(const char* name) {
    const char* value = std::getenv(name);
    return value && value[0] == '0';
}

// Diagnostic compatibility controls. The original all-in-one switch remains as a
// convenient umbrella, while the component switches let the validation harness
// distinguish projection, expert-dispatch, and LM-head quantization drift.
bool qact_all()    { return env_one("SPARKINFER_K3_GGML_QACT"); }

// PROJECTIONS ARE ON BY DEFAULT; the other two are not, and the asymmetry is
// measured rather than stylistic.
//
// Q8 activations let the projection GEMV run llama.cpp's __dp4a integer path
// instead of decoding weights to f32, and projections are 35.7% of decode GPU
// time. Measured at the scored 128k context, tp=8, interleaved against its own
// control on one binary: 109.43 -> 97.10 ms/token, -11.3%. Accuracy holds by a
// wide margin -- top1 1.0, mean_kld 0.006433 against a 0.20 gate.
//
// The MoE and LM-head variants stay OFF: _MOE measured a STABLE -8.6%
// REGRESSION (118.0/117.7/118.8/120.0 vs 109.0/108.8/108.7/109.3) and _OUTPUT
// measured no effect at all, being one launch per token. Enabling all three
// under one umbrella switch is what hid that for so long.
//
// SPARKINFER_K3_GGML_QACT_PROJ=0 restores f32 projections. It has to be an
// explicit 0 rather than an unset: an env-gated default-off improvement is one
// the scoring harness never runs, which is exactly the trap this comment exists
// to stop the next person walking into.
bool qact_proj()   { return !env_zero("SPARKINFER_K3_GGML_QACT_PROJ"); }
bool qact_moe()    { return qact_all() || env_one("SPARKINFER_K3_GGML_QACT_MOE"); }
bool qact_output() { return qact_all() || env_one("SPARKINFER_K3_GGML_QACT_OUTPUT"); }

// Upload one GGUF tensor's raw bytes to the device, native quant format preserved.
// This is the WHOLE loader primitive — every K3 weight is read natively at forward
// time (k3_proj_f32 / the MoE dispatch kernels decode Q8_0 / IQ1_S / IQ2_XS / F32
// directly), so there is no equivalent of a load-time dequantize-to-bf16 pass.
// Upload this rank's SLICE of a tensor, honouring w.shard. At tp_size 1 the plan
// degenerates to Replicate and this is byte-for-byte the old whole-tensor upload.
//
// The slicing decisions are NOT made here — plan_tensor_residency() owns them, and
// is unit-tested without a GPU (tp_weight_residency_cpu_test). This function is the
// copy plumbing for the StridedCopy that planner returns, and nothing more. That
// split matters: ColShard cuts WITHIN each memory row, and a cut that lands
// mid-quantisation-block is unrecoverable and undetectable — right shape, right byte
// count, noise for values. The planner bands over BLOCKS so every cut is legal by
// construction; re-deriving offsets here would put that guarantee back at risk.
bool upload_sliced(const GGUF& g, const std::string& name, KimiK3Weights& w,
                   KimiK3Tensor& out, bool required, bool kda_layer) {
    const GGUFTensor* t = g.tensor(name);
    if (!t) {
        if (required)
            std::fprintf(stderr, "[k3] missing required tensor: %s\n", name.c_str());
        return !required;   // absent-and-optional is not an error
    }

    // --- tp_size 1: the original path, unchanged ---
    if (w.shard.tp_size <= 1) {
        void* d = nullptr;
        if (cudaMalloc(&d, (size_t)t->n_bytes) != cudaSuccess) {
            std::fprintf(stderr, "[k3] cudaMalloc failed for %s (%ld bytes)\n",
                         name.c_str(), t->n_bytes);
            return false;
        }
        if (cudaMemcpy(d, t->data, (size_t)t->n_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            std::fprintf(stderr, "[k3] cudaMemcpy failed for %s\n", name.c_str());
            cudaFree(d);
            return false;
        }
        w.owned.push_back(d);
        out.data = d;
        out.type = t->ggml_type;
        out.n_bytes = t->n_bytes;
        for (int k = 0; k < 4; ++k) out.rank_ne[k] = t->dims[k];
        return true;
    }

    // --- tp_size > 1 ---
    // Under ExpertsOnly, only the three routed-expert stacks are banded. Everything
    // else takes the replicate path above, at full width, so the forward's cfg-derived
    // shapes stay correct on every rank. Matching on the tensor SUFFIX rather than on
    // the rule is deliberate: attn_k_b/attn_v_b also carry Rule::ExpertShard (they
    // band the head axis), and banding those without teaching the MLA kernels their
    // per-rank head count would silently drop 7/8 of the attention heads.
    if (w.policy != KimiK3Weights::ShardPolicy::Full) {
        auto ends_with = [&](const char* suf) {
            const std::size_t n = std::strlen(suf);
            return name.size() >= n && name.compare(name.size() - n, n, suf) == 0;
        };
        const bool routed_expert_stack = ends_with("ffn_gate_exps.weight") ||
                                         ends_with("ffn_up_exps.weight")   ||
                                         ends_with("ffn_down_exps.weight");

        // ExpertsAndMla bands the 24 MLA layers' PER-HEAD tensors and col-shards
        // their output. `!kda_layer` carries exactly the same weight here that
        // `kda_layer` does below, and for the same reason: attn_output.weight and
        // attn_q.weight are written by BOTH branches. Banding either by name alone
        // would shard the wrong two-thirds of the model and leave every KDA layer
        // contracting over an eighth of its input, which stays fluent and is wrong.
        //
        // attn_kv_a_mqa / attn_kv_a_norm are NOT here: MQA means one latent KV
        // shared by all 96 heads, so every rank needs that projection — and the
        // cache it fills — whole. Banding it is the one mistake that would make
        // this look like it works while each rank attended to an eighth of the
        // context.
        const bool mla_sharded =
            KimiK3Weights::shards_mla(w.policy) && !kda_layer &&
            (ends_with("attn_q_b.weight")  || ends_with("attn_k_b.weight") ||
             ends_with("attn_v_b.weight")  || ends_with("attn_gate.weight") ||
             ends_with("attn_q.weight")    || ends_with("attn_output.weight"));

        // ExpertsAndKda additionally bands the KDA attention projections.
        //
        // `kda_layer` IS NOT REDUNDANT WITH THE SUFFIX, and leaving it out is the
        // bug this whole branch exists to avoid. attn_output.weight is written by
        // BOTH branches — it is KDA's wo on 69 layers and MLA's wo on the other 24
        // — and MLA is deliberately not sharded. Banding it by name alone would
        // col-shard MLA's wo while the MLA kernels still hand it the full
        // qh*value_length_mla activation, so every MLA layer would contract over an
        // eighth of its input and the all-reduce would sum eight of those into a
        // fluent wrong answer. Same trap for attn_q.weight, which MLA reuses as its
        // dense-q fallback.
        const bool kda_sharded =
            KimiK3Weights::shards_kda(w.policy) && kda_layer &&
            (ends_with("attn_q.weight")  || ends_with("attn_k.weight") ||
             ends_with("attn_v.weight")  || ends_with("ssm_g.weight")  ||
             ends_with("ssm_f_b.weight") || ends_with("attn_output.weight"));

        // The shared expert is banded too (weight_plan.cpp: RowShard/RowShard/
        // ColShard over the ffn width). It has to be listed HERE as well: this
        // short-circuit is what decides whether the rule table is consulted at all,
        // so a rule added there and not here is silently inert — the tensor loads
        // full width, rank_ne reports the full width, and every rank computes a
        // complete shared-expert output that the fused reduce then sums tp_size
        // times. That is not hypothetical; it shipped once and was caught only by
        // reading the loader rather than the plan.
        const bool shared_expert_stack = ends_with("ffn_gate_shexp.weight") ||
                                         ends_with("ffn_up_shexp.weight")   ||
                                         ends_with("ffn_down_shexp.weight");

        if (!routed_expert_stack && !shared_expert_stack &&
            !kda_sharded && !mla_sharded) {
            void* d = nullptr;
            if (cudaMalloc(&d, (size_t)t->n_bytes) != cudaSuccess) {
                std::fprintf(stderr, "[k3] cudaMalloc failed for %s (%ld bytes)\n",
                             name.c_str(), t->n_bytes);
                return false;
            }
            if (cudaMemcpy(d, t->data, (size_t)t->n_bytes, cudaMemcpyHostToDevice)
                    != cudaSuccess) {
                std::fprintf(stderr, "[k3] cudaMemcpy failed for %s\n", name.c_str());
                cudaFree(d);
                return false;
            }
            w.owned.push_back(d);
            out.data = d;
            out.type = t->ggml_type;
            out.n_bytes = t->n_bytes;
            for (int k = 0; k < 4; ++k) out.rank_ne[k] = t->dims[k];
            return true;
        }
    }

    // A KDA LAYER'S attn_k/attn_v ARE NOT A GQA KV GROUP.
    //
    // weight_plan downgrades attn_k.weight / attn_v.weight to Replicate whenever
    // ShardDims::kv_replicated is set, because a kv group must be visible whole to
    // every query head attending through it. That rule is right for MLA — K3's GGUF
    // stores it as MQA with ONE kv head, so at tp=8 it must replicate — and wrong
    // for KDA, where the same two tensor NAMES are per-head input projections at
    // [hidden, qkv] with no kv group anywhere near them.
    //
    // The downgrade does not fire today only because the K3 TP driver never sets
    // kv_replicated, so it defaults to false. That is luck, not design: the moment
    // anyone populates `shard` through tp::shard_dims() — which DOES set it for K3 —
    // every rank would silently hold a full-width attn_k while its attn_q was banded,
    // and the layer would run at full speed over mismatched slices. So the KDA path
    // states the exemption rather than depending on a field being unset.
    tp::ShardDims sd = w.shard;
    if (kda_layer) sd.kv_replicated = false;
    const tp::TensorResidency r =
        tp::plan_tensor_residency(name, t->dims, t->n_dims, t->ggml_type, sd);

    // A tensor with no rule, an unknown quant type, or a mid-block cut STOPS THE
    // LOAD. The alternative — fall back to replicating it — silently changes the
    // model: a weight that should have been sharded but was replicated makes every
    // rank compute the full contraction, and the following all-reduce then sums
    // tp_size copies of the same complete answer.
    if (!r.ok()) {
        std::fprintf(stderr, "[k3] TP: %s — %s (%s)%s%s\n", name.c_str(),
                     tp::residency_error_name(r.error), tp::rule_name(r.rule),
                     r.note.empty() ? "" : ": ", r.note.c_str());
        return false;
    }

    void* d = nullptr;
    if (cudaMalloc(&d, r.rank_bytes) != cudaSuccess) {
        std::fprintf(stderr, "[k3] cudaMalloc failed for %s rank slice (%zu bytes)\n",
                     name.c_str(), r.rank_bytes);
        return false;
    }

    const char* src = static_cast<const char*>(t->data);
    char* dst = static_cast<char*>(d);
    const tp::StridedCopy& c = r.copy;
    cudaError_t e;
    if (c.contiguous()) {
        // RowShard / ExpertShard / Replicate: one band of whole memory rows.
        e = cudaMemcpy(dst + c.dst_offset, src + c.src_offset, c.total_bytes(),
                       cudaMemcpyHostToDevice);
    } else {
        // ColShard: a sub-range within EVERY row, so ne1 separate runs. cudaMemcpy2D
        // rather than a loop — for a 3-D expert tensor the loop would be millions of
        // tiny transfers.
        e = cudaMemcpy2D(dst + c.dst_offset, c.dst_stride,
                         src + c.src_offset, c.src_stride,
                         c.row_bytes, (size_t)c.n_rows,
                         cudaMemcpyHostToDevice);
    }
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[k3] upload failed for %s rank slice: %s\n",
                     name.c_str(), cudaGetErrorString(e));
        cudaFree(d);
        return false;
    }

    w.owned.push_back(d);
    out.data = d;
    out.type = t->ggml_type;
    out.n_bytes = (long)r.rank_bytes;
    for (int k = 0; k < 4; ++k) out.rank_ne[k] = r.rank_ne[k];
    return true;
}

// Kept as the name the 49 load sites use.
bool upload_raw(const GGUF& g, const std::string& name, KimiK3Weights& w,
                KimiK3Tensor& out, bool required, bool kda_layer = false) {
    return upload_sliced(g, name, w, out, required, kda_layer);
}

// wk_b / wv_b feed mla_absorb_q_f32 / mla_decode_attn_f32, which take them as plain
// `const float*` — those two kernels have no quantized-input variant, unlike every
// other weight in the model, because the per-head contraction they do is not an
// ordinary GEMV. So if the file stores them quantized (expected: Q8_0, same tier as
// the rest of MLA's attention weights), this uploads native bytes AND THEN
// dequantises once into a fresh F32 buffer — a one-time cost at load, not per token
// — rather than writing a second quantized-aware kernel variant for a tensor this
// small (about 25 MB each at F32, per K3's dims).
bool upload_force_f32(const GGUF& g, const std::string& name, KimiK3Weights& w,
                     KimiK3Tensor& out, bool required, long n_values) {
    const GGUFTensor* t = g.tensor(name);
    if (!t) {
        if (required)
            std::fprintf(stderr, "[k3] missing required tensor: %s\n", name.c_str());
        return !required;
    }
    if (t->ggml_type == 0) {
        return upload_raw(g, name, w, out, required, /*kda_layer=*/false);
    }

    // Stage the rank's QUANTISED slice through upload_sliced (so the ColShard /
    // ExpertShard band arithmetic has exactly one implementation), then dequantise
    // that slice. wk_b/wv_b are ExpertShard on the HEAD axis — ne2 = 96 heads, banded
    // 12-per-rank at tp 8 — so the rank's element count is NOT the caller's
    // full-tensor n_values, and dequantising n_values elements out of a buffer
    // holding an eighth of them reads past the end.
    KimiK3Tensor staged;
    const std::size_t owned_before = w.owned.size();
    if (!upload_sliced(g, name, w, staged, required, /*kda_layer=*/false)) return false;
    if (!staged.ok()) return !required;

    long rank_values = 1;
    for (int k = 0; k < 4; ++k) rank_values *= (staged.rank_ne[k] > 0 ? staged.rank_ne[k] : 1);
    if (w.shard.tp_size <= 1 && rank_values != n_values) {
        std::fprintf(stderr, "[k3] %s: shape gives %ld values, caller expected %ld\n",
                     name.c_str(), rank_values, n_values);
        return false;
    }

    void* f32 = nullptr;
    if (cudaMalloc(&f32, (size_t)rank_values * sizeof(float)) != cudaSuccess) return false;
    const bool ok = k3k::dequant_f32_by_type((float*)f32, staged.data, rank_values,
                                             staged.type, 0);
    cudaDeviceSynchronize();

    // Drop the staged quantised buffer: only the f32 expansion is kept live.
    if (w.owned.size() > owned_before) {
        cudaFree(w.owned[owned_before]);
        w.owned.erase(w.owned.begin() + (long)owned_before);
    }
    if (!ok) {
        std::fprintf(stderr, "[k3] %s: ggml type %d has no dequant path (needed as f32 "
                             "for the absorbed-MLA kernels)\n", name.c_str(), staged.type);
        cudaFree(f32);
        return false;
    }
    w.owned.push_back(f32);
    out.data = f32;
    out.type = 0;   // now F32, regardless of the file's original type
    out.n_bytes = rank_values * (long)sizeof(float);
    for (int k = 0; k < 4; ++k) out.rank_ne[k] = staged.rank_ne[k];
    return true;
}


// Opt-in phase timing, SPARKINFER_K3_PROFILE=1. Records cudaEvent pairs around the
// attention and FFN branches and accumulates GPU time per tag, reading the elapsed
// time back only when the tag is next reused (by which point the events are long
// since complete), so the hot loop takes no extra sync. Off by default and behind a
// single env check, so the production path is untouched — this exists to find the
// bottleneck for the H200 optimization pass, not to run in it.
// SLOTS ARE KEYED BY (TAG, DEVICE), NOT BY TAG ALONE.
//
// cudaEventElapsedTime is defined only for two events recorded on the SAME device.
// This profiler is a process-global singleton, so under tensor parallelism every rank
// reaches the same tag ("ffn_moe") from a different device, and rank 1's start would be
// paired against rank 0's stop — across devices, where the driver returns garbage.
//
// MEASURED before this fix, at tp=8: ffn_dense -9557 ms, ffn_moe -1265 ms, total
// -8564 ms. Negative elapsed times are obvious once printed, but the same defect with a
// smaller cross-device skew would have produced plausible-looking SHARES instead — and
// deciding where the optimisation effort goes is the only thing this profiler is for.
struct K3Profiler {
    struct Slot { cudaEvent_t a = nullptr, b = nullptr; bool pending = false; double ms = 0; long n = 0; };
    std::unordered_map<std::string, Slot> slots;   // key: "tag@device"
    bool on = false;
    K3Profiler() { const char* e = std::getenv("SPARKINFER_K3_PROFILE"); on = e && e[0] == '1'; }

    static std::string key_for(const std::string& tag) {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) dev = -1;
        return tag + "@" + std::to_string(dev);
    }
    void start(const std::string& tag, cudaStream_t st) {
        if (!on) return;
        auto& sl = slots[key_for(tag)];
        if (!sl.a) { cudaEventCreate(&sl.a); cudaEventCreate(&sl.b); }
        if (sl.pending) {   // drain the previous pair for this tag+device first
            cudaEventSynchronize(sl.b);
            float ms = 0;
            if (cudaEventElapsedTime(&ms, sl.a, sl.b) == cudaSuccess) { sl.ms += ms; ++sl.n; }
        }
        cudaEventRecord(sl.a, st);
        sl.pending = true;
    }
    void stop(const std::string& tag, cudaStream_t st) {
        if (!on) return;
        auto it = slots.find(key_for(tag));
        if (it != slots.end() && it->second.b) cudaEventRecord(it->second.b, st);
    }
    void report() {
        if (!on) return;
        // Aggregate per TAG across devices. Every rank runs the same phases, so the
        // per-tag sum is total GPU time in that phase across the node, and the shares
        // are what the optimisation decision needs.
        std::unordered_map<std::string, double> by_tag;
        std::unordered_map<std::string, int> devs_seen;
        for (auto& kv : slots) {
            auto& sl = kv.second;
            if (sl.pending) {
                cudaEventSynchronize(sl.b);
                float ms = 0;
                if (cudaEventElapsedTime(&ms, sl.a, sl.b) == cudaSuccess) { sl.ms += ms; ++sl.n; }
                sl.pending = false;
            }
            const std::size_t at = kv.first.rfind('@');
            const std::string tag = at == std::string::npos ? kv.first : kv.first.substr(0, at);
            by_tag[tag] += sl.ms;
            devs_seen[tag] += 1;
        }
        double total = 0;
        std::vector<std::pair<std::string, double>> rows;
        for (auto& kv : by_tag) { rows.emplace_back(kv.first, kv.second); total += kv.second; }
        std::sort(rows.begin(), rows.end(), [](auto& a, auto& b){ return a.second > b.second; });
        std::fprintf(stderr, "\n[k3 profile] %.1f ms of GPU time across timed phases "
                             "(summed over ranks):\n", total);
        for (auto& r : rows)
            std::fprintf(stderr, "  %-18s %9.2f ms  (%5.1f%%)  over %d device(s)\n",
                         r.first.c_str(), r.second,
                         total > 0 ? 100.0 * r.second / total : 0.0, devs_seen[r.first]);
    }
};
K3Profiler& k3_profiler() { static K3Profiler p; return p; }

}  // namespace

void kimi_k3_profile_report() { k3_profiler().report(); }

bool kimi_k3_load_weights(const GGUF& g, const KimiK3Config& cfg,
                         const K3PlanOptions& opt, KimiK3Weights& out,
                         int first_layer, int last_layer) {
    return kimi_k3_load_weights_scoped(g, cfg, opt, out, first_layer, last_layer,
                                      /*load_embed=*/true, /*load_head=*/true);
}

bool kimi_k3_load_weights_scoped(const GGUF& g, const KimiK3Config& cfg,
                                const K3PlanOptions& opt, KimiK3Weights& out,
                                int first_layer, int last_layer,
                                bool load_embed, bool load_head) {
    // THIS RANK's KDA slice, derived once and carried on the weights so the loader
    // and the forward cannot disagree about it. Under any policy but ExpertsAndKda
    // it stays the tp=1 identity: full width, zero offsets.
    {
        tp::Config tcfg;
        tcfg.tp_size = KimiK3Weights::shards_kda(out.policy)
                     ? (out.shard.tp_size > 0 ? out.shard.tp_size : 1) : 1;
        tp::KdaShape ks;
        ks.hidden = cfg.hidden;
        ks.n_heads = cfg.n_q_heads;
        ks.head_dim = cfg.kda_head_dim;
        ks.conv_kernel = cfg.kda_conv_kernel;
        const int r = tcfg.tp_size > 1 ? out.shard.rank : 0;
        tp::ShardError e = tp::kda_shard_dims(ks, tcfg, r, &out.kda);
        if (!e.ok()) {
            std::fprintf(stderr, "[k3] KDA shard: %s\n", e.message.c_str());
            return false;
        }
    }

    // THIS RANK's MLA head slice, on the same discipline: derived once, carried on
    // the weights, tp=1 identity under any policy but ExpertsAndMla. The forward
    // reads md.n_heads unconditionally, so a policy that does not shard MLA gets
    // the full 96 from the same field rather than from a different code path.
    {
        tp::Config tcfg;
        tcfg.tp_size = KimiK3Weights::shards_mla(out.policy)
                     ? (out.shard.tp_size > 0 ? out.shard.tp_size : 1) : 1;
        tp::MlaShape ms;
        ms.hidden = cfg.hidden;
        ms.n_q_heads = cfg.n_q_heads;
        ms.key_length_mla = cfg.key_length_mla;
        ms.value_length_mla = cfg.value_length_mla;
        ms.qk_nope = cfg.key_length_mla - cfg.rope_dim;
        ms.kv_lora_rank = cfg.kv_lora_rank;
        const int r = tcfg.tp_size > 1 ? out.shard.rank : 0;
        tp::ShardError e = tp::mla_shard_dims(ms, tcfg, r, &out.mla);
        if (!e.ok()) {
            std::fprintf(stderr, "[k3] MLA shard: %s\n", e.message.c_str());
            return false;
        }
    }

    out.layers.assign(cfg.n_layers, KimiK3LayerWeights{});

    bool ok = true;
    if (load_embed)
        ok &= upload_raw(g, "token_embd.weight", out, out.token_embd, true);
    if (load_head) {
        ok &= upload_raw(g, "output_norm.weight", out, out.output_norm, true);
        ok &= upload_raw(g, "output.weight", out, out.output, true);
        if (cfg.attn_res_block_size > 0) {
            out.has_output_res_score = true;
            ok &= upload_raw(g, "output_res_score.weight", out, out.output_res_score, true);
        }
    }
    if (!ok) return false;

    const int qh = cfg.n_q_heads;
    const int qk_nope = cfg.key_length_mla - cfg.rope_dim;

    for (int i = first_layer; i <= last_layer && i < cfg.n_layers; ++i) {
        if (i < 0) continue;
        KimiK3LayerWeights& L = out.layers[i];
        L.is_kda = cfg.is_kda_layer(i);

        ok &= upload_raw(g, blk(i, "attn_norm.weight"), out, L.attn_norm, true);
        ok &= upload_raw(g, blk(i, "ffn_norm.weight"), out, L.ffn_norm, true);
        if (cfg.attn_res_block_size > 0) {
            ok &= upload_raw(g, blk(i, "attn_res_score.weight"), out, L.attn_res_score, true);
            ok &= upload_raw(g, blk(i, "ffn_res_score.weight"), out, L.ffn_res_score, true);
        }

        if (L.is_kda) {
            ok &= upload_raw(g, blk(i, "attn_q.weight"), out, L.attn_q, true, /*kda_layer=*/true);
            ok &= upload_raw(g, blk(i, "attn_k.weight"), out, L.attn_k, true, /*kda_layer=*/true);
            ok &= upload_raw(g, blk(i, "attn_v.weight"), out, L.attn_v, true, /*kda_layer=*/true);
            ok &= upload_raw(g, blk(i, "ssm_conv1d_q.weight"), out, L.ssm_conv1d_q, true);
            ok &= upload_raw(g, blk(i, "ssm_conv1d_k.weight"), out, L.ssm_conv1d_k, true);
            ok &= upload_raw(g, blk(i, "ssm_conv1d_v.weight"), out, L.ssm_conv1d_v, true);
            if (L.ssm_conv1d_q.ok() && L.ssm_conv1d_q.type != 0) {
                std::fprintf(stderr, "[k3] blk.%d.ssm_conv1d_q.weight is ggml type %d, "
                                     "not F32 — kda_conv_step_f32 has no quantized path\n",
                             i, L.ssm_conv1d_q.type);
                ok = false;
            }
            ok &= upload_raw(g, blk(i, "ssm_f_a.weight"), out, L.ssm_f_a, true);
            ok &= upload_raw(g, blk(i, "ssm_f_b.weight"), out, L.ssm_f_b, true, /*kda_layer=*/true);
            ok &= upload_raw(g, blk(i, "ssm_beta.weight"), out, L.ssm_beta, true);
            ok &= upload_raw(g, blk(i, "ssm_dt.bias"), out, L.ssm_dt_bias, true);
            ok &= upload_raw(g, blk(i, "ssm_a"), out, L.ssm_a, true);
            ok &= upload_raw(g, blk(i, "ssm_g.weight"), out, L.ssm_g, true, /*kda_layer=*/true);
            ok &= upload_raw(g, blk(i, "ssm_norm.weight"), out, L.ssm_norm, true);
            ok &= upload_raw(g, blk(i, "attn_output.weight"), out, L.attn_output, true, /*kda_layer=*/true);
        } else {
            L.has_q_lora = opt.has_q_lora;
            if (L.has_q_lora) {
                ok &= upload_raw(g, blk(i, "attn_q_a.weight"), out, L.attn_q_a, true);
                ok &= upload_raw(g, blk(i, "attn_q_a_norm.weight"), out, L.attn_q_a_norm, true);
                ok &= upload_raw(g, blk(i, "attn_q_b.weight"), out, L.attn_q_b, true);
            } else {
                ok &= upload_raw(g, blk(i, "attn_q.weight"), out, L.attn_q_dense, true);
            }
            ok &= upload_raw(g, blk(i, "attn_kv_a_mqa.weight"), out, L.attn_kv_a_mqa, true);
            ok &= upload_raw(g, blk(i, "attn_kv_a_norm.weight"), out, L.attn_kv_a_norm, true);

            L.has_fused_kv_b = opt.has_fused_kv_b;
            if (L.has_fused_kv_b) {
                std::fprintf(stderr, "[k3] blk.%d: fused attn_kv_b is not numerically "
                                     "wired (the split this needs is not derived) — "
                                     "refusing rather than running untested code\n", i);
                ok = false;
            } else {
                const long kb_n = (long)qk_nope * cfg.kv_lora_rank * qh;
                const long vb_n = (long)cfg.kv_lora_rank * cfg.value_length_mla * qh;
                ok &= upload_force_f32(g, blk(i, "attn_k_b.weight"), out, L.attn_k_b,
                                       true, kb_n);
                ok &= upload_force_f32(g, blk(i, "attn_v_b.weight"), out, L.attn_v_b,
                                       true, vb_n);
            }
            L.has_attn_gate = opt.has_attn_gate;
            if (L.has_attn_gate)
                ok &= upload_raw(g, blk(i, "attn_gate.weight"), out, L.attn_gate, true);
            ok &= upload_raw(g, blk(i, "attn_output.weight"), out, L.attn_output, true);
        }

        if (i < cfg.leading_dense) {
            ok &= upload_raw(g, blk(i, "ffn_gate.weight"), out, L.ffn_gate, true);
            ok &= upload_raw(g, blk(i, "ffn_up.weight"), out, L.ffn_up, true);
            ok &= upload_raw(g, blk(i, "ffn_down.weight"), out, L.ffn_down, true);
        } else {
            ok &= upload_raw(g, blk(i, "ffn_gate_inp.weight"), out, L.ffn_gate_inp, true);
            ok &= upload_raw(g, blk(i, "exp_probs_b.bias"), out, L.exp_probs_b, true);
            ok &= upload_raw(g, blk(i, "ffn_gate_exps.weight"), out, L.ffn_gate_exps, true);
            ok &= upload_raw(g, blk(i, "ffn_up_exps.weight"), out, L.ffn_up_exps, true);
            ok &= upload_raw(g, blk(i, "ffn_down_exps.weight"), out, L.ffn_down_exps, true);
            ok &= upload_raw(g, blk(i, "ffn_routed_down.weight"), out, L.ffn_routed_down, true);
            ok &= upload_raw(g, blk(i, "ffn_routed_up.weight"), out, L.ffn_routed_up, true);
            L.has_routed_norm = opt.has_routed_norm;
            if (L.has_routed_norm)
                ok &= upload_raw(g, blk(i, "ffn_routed_norm.weight"), out, L.ffn_routed_norm, true);
            L.has_shared_experts = opt.has_shared_experts;
            if (L.has_shared_experts) {
                ok &= upload_raw(g, blk(i, "ffn_gate_shexp.weight"), out, L.ffn_gate_shexp, true);
                ok &= upload_raw(g, blk(i, "ffn_up_shexp.weight"), out, L.ffn_up_shexp, true);
                ok &= upload_raw(g, blk(i, "ffn_down_shexp.weight"), out, L.ffn_down_shexp, true);
            }
        }
    }
    return ok;
}

void kimi_k3_free_weights(KimiK3Weights& w) {
    for (void* p : w.owned) cudaFree(p);
    w.owned.clear();
    w.layers.clear();
}

int kimi_k3_kda_ordinal(const KimiK3Config& cfg, int layer) {
    if (layer < 0 || layer >= cfg.n_layers || !cfg.is_kda_layer(layer)) return -1;
    int ord = 0;
    for (int i = 0; i < layer; ++i) if (cfg.is_kda_layer(i)) ++ord;
    return ord;
}

int kimi_k3_mla_ordinal(const KimiK3Config& cfg, int layer) {
    if (layer < 0 || layer >= cfg.n_layers || cfg.is_kda_layer(layer)) return -1;
    int ord = 0;
    for (int i = 0; i < layer; ++i) if (!cfg.is_kda_layer(i)) ++ord;
    return ord;
}

bool kimi_k3_alloc_state(const KimiK3Config& cfg, int max_ctx, KimiK3RuntimeState& out,
                        int first_layer, int last_layer, const tp::KdaShardDims* kda) {
    // THIS RANK's KDA width. Null kda == full width, which is what tp=1 and every
    // unsharded policy must keep getting, byte for byte.
    const int qkv    = kda ? kda->qkv     : cfg.n_q_heads * cfg.kda_head_dim;
    const int n_head = kda ? kda->n_heads : cfg.n_q_heads;
    const int n_kda = cfg.n_kda_layers();
    const int n_mla = cfg.n_mla_layers();
    const int lo = first_layer < 0 ? 0 : first_layer;
    const int hi = last_layer < 0 ? cfg.n_layers - 1 : last_layer;
    out.max_ctx = max_ctx;
    out.max_ckpt = cfg.attn_res_block_size > 0
        ? (cfg.n_layers + cfg.attn_res_block_size - 1) / cfg.attn_res_block_size
        : 0;

    auto alloc = [&](size_t n_floats) -> float* {
        void* p = nullptr;
        if (cudaMalloc(&p, n_floats * sizeof(float)) != cudaSuccess) return nullptr;
        out.owned.push_back(p);
        return (float*)p;
    };

    // The state vectors stay indexed by GLOBAL ordinal (kimi_k3_kda_ordinal /
    // kimi_k3_mla_ordinal), so they keep their full length — but only the slots for
    // layers this stage actually owns are allocated; the rest stay null and are never
    // indexed. This is what makes the pipeline fit: without it, EVERY stage would
    // allocate all 24 MLA layers' KV cache, and at 1M context that is ~58 GB per
    // stage of KV that mostly belongs to other GPUs' layers. Sized to what a stage
    // owns, it is ~58 GB / 8 instead.
    out.conv_state_q.assign(n_kda, nullptr);
    out.conv_state_k.assign(n_kda, nullptr);
    out.conv_state_v.assign(n_kda, nullptr);
    out.delta_state.assign(n_kda, nullptr);
    out.mla_kv_cache.assign(n_mla, nullptr);

    for (int layer = lo; layer <= hi && layer < cfg.n_layers; ++layer) {
        if (cfg.is_kda_layer(layer)) {
            const int k = kimi_k3_kda_ordinal(cfg, layer);
            out.conv_state_q[k] = alloc((size_t)(cfg.kda_conv_kernel - 1) * qkv);
            out.conv_state_k[k] = alloc((size_t)(cfg.kda_conv_kernel - 1) * qkv);
            out.conv_state_v[k] = alloc((size_t)(cfg.kda_conv_kernel - 1) * qkv);
            out.delta_state[k]  = alloc((size_t)cfg.kda_head_dim * cfg.kda_head_dim * n_head);
            if (!out.conv_state_q[k] || !out.conv_state_k[k] || !out.conv_state_v[k] ||
                !out.delta_state[k])
                return false;
        } else {
            const int k = kimi_k3_mla_ordinal(cfg, layer);
            out.mla_kv_cache[k] = alloc((size_t)cfg.key_length * max_ctx);
            if (!out.mla_kv_cache[k]) return false;
        }
    }

    if (out.max_ckpt > 0) {
        out.res_bank = alloc((size_t)cfg.hidden * out.max_ckpt);
        if (!out.res_bank) return false;
    }

    out.conv_state_elems = (cfg.kda_conv_kernel - 1) * qkv;
    out.delta_state_elems = cfg.kda_head_dim * cfg.kda_head_dim * n_head;
    out.kv_cache_elems = cfg.key_length * max_ctx;
    out.res_bank_row_elems = cfg.hidden;

    kimi_k3_reset_state(out);
    return true;
}

void kimi_k3_reset_state(KimiK3RuntimeState& s) {
    s.position = 0;
    s.n_ckpt = 0;
    auto z = [](float* p, size_t n_elems) {
        if (p && n_elems > 0) cudaMemset(p, 0, n_elems * sizeof(float));
    };
    for (float* p : s.conv_state_q) z(p, (size_t)s.conv_state_elems);
    for (float* p : s.conv_state_k) z(p, (size_t)s.conv_state_elems);
    for (float* p : s.conv_state_v) z(p, (size_t)s.conv_state_elems);
    for (float* p : s.delta_state)  z(p, (size_t)s.delta_state_elems);
    for (float* p : s.mla_kv_cache) z(p, (size_t)s.kv_cache_elems);
    // res_bank is not zeroed — n_ckpt=0 means no row is read until pushed, and a
    // push always writes the full row before n_ckpt is incremented, so stale bytes
    // in unused rows are never observed.
}

void kimi_k3_free_state(KimiK3RuntimeState& s) {
    for (void* p : s.owned) cudaFree(p);
    s.owned.clear();
    s.conv_state_q.clear(); s.conv_state_k.clear(); s.conv_state_v.clear();
    s.delta_state.clear(); s.mla_kv_cache.clear();
    s.res_bank = nullptr;
}

// ---------------------------------------------------------------------------
// Scratch
// ---------------------------------------------------------------------------

struct KimiK3Forward::Scratch {
    float* mixed = nullptr;        // [H]
    float* mixed2 = nullptr;       // [H]
    float* normed = nullptr;       // [H]
    float* normed2 = nullptr;      // [H]
    float* attn_out = nullptr;     // [H]
    float* ffn_out = nullptr;      // [H]

    // KDA
    float* qkv_q = nullptr, *qkv_k = nullptr, *qkv_v = nullptr;   // [qkv]
    float* conv_q = nullptr, *conv_k = nullptr, *conv_v = nullptr; // [qkv]
    float* f_a_out = nullptr;      // [head_dim]
    float* g_raw = nullptr;        // [qkv]
    float* decay_g = nullptr;      // [qkv]
    float* beta_out = nullptr;     // [n_head]
    float* delta_out = nullptr;    // [qkv]
    float* g_proj_out = nullptr;   // [qkv]
    float* gate_out = nullptr;     // [qkv]

    // MLA
    float* q_lora_out = nullptr;   // [q_lora_rank]
    float* q_proj_out = nullptr;   // [qh*key_length_mla]
    float* q_nope = nullptr;       // [qh*qk_nope]
    float* q_pe = nullptr;         // [qh*rope_dim]
    float* kv_a_out = nullptr;     // [key_length]
    float* kv_cmpr_normed = nullptr;  // [kv_lora_rank]
    float* absorbed_q = nullptr;   // [qh*key_length]
    float* mla_attn_out = nullptr; // [qh*value_length_mla]
    float* gate_proj_out = nullptr;   // [qh*value_length_mla]

    // FFN / MoE
    float* router_logits = nullptr;   // [n_experts]
    float* router_w = nullptr;        // [top_k]
    int*   router_ids = nullptr;      // [top_k]
    float* routed_down_out = nullptr; // [expert_latent]
    float* moe_scratch = nullptr;     // [top_k*moe_ffn]
    // [expert_latent + H], the only owned pointer of the three. moe_out and
    // shexp_out are views into it so one collective reduces both — see the
    // allocation in kimi_k3_forward_alloc_scratch.
    float* moe_fused = nullptr;
    float* moe_out = nullptr;         // = moe_fused,                 [expert_latent]
    float* dense_gate = nullptr, *dense_up = nullptr, *dense_situ = nullptr; // [dense_ffn]
    float* shexp_out = nullptr;        // = moe_fused + expert_latent, [H]
    // Softmax scratch for attn_res_mix_f32, [max_ckpt + 1]. Persistent because the
    // mix runs twice per layer: allocating it per call cost a cudaMallocAsync /
    // cudaFreeAsync pair ~185 times per token.
    float* res_scores = nullptr;       // [ceil(n_layers / attn_res_block_size) + 1]
    void* proj_q8 = nullptr;           // optional llama CPU-compat block_q8_0 scratch
    void* moe_q8 = nullptr;            // optional llama CPU-compat block_q8_K scratch

    std::vector<void*> owned;
};

bool kimi_k3_forward_alloc_scratch(const KimiK3Config& cfg, KimiK3Forward& fwd) {
    fwd.s = new KimiK3Forward::Scratch();
    auto& s = *fwd.s;
    const int H = cfg.hidden;
    const int qkv = cfg.n_q_heads * cfg.kda_head_dim;
    const int qh = cfg.n_q_heads;
    const int qk_nope = cfg.key_length_mla - cfg.rope_dim;

    auto alloc_f = [&](float*& ptr, size_t n) {
        void* p = nullptr;
        if (cudaMalloc(&p, n * sizeof(float)) != cudaSuccess) return false;
        s.owned.push_back(p);
        ptr = (float*)p;
        return true;
    };
    auto alloc_i = [&](int*& ptr, size_t n) {
        void* p = nullptr;
        if (cudaMalloc(&p, n * sizeof(int)) != cudaSuccess) return false;
        s.owned.push_back(p);
        ptr = (int*)p;
        return true;
    };
    auto alloc_bytes = [&](void*& ptr, size_t n) {
        if (n == 0 || cudaMalloc(&ptr, n) != cudaSuccess) return false;
        s.owned.push_back(ptr);
        return true;
    };

    bool ok = true;
    ok &= alloc_f(s.mixed, H);
    ok &= alloc_f(s.mixed2, H);
    ok &= alloc_f(s.normed, H);
    ok &= alloc_f(s.normed2, H);
    ok &= alloc_f(s.attn_out, H);
    ok &= alloc_f(s.ffn_out, H);

    ok &= alloc_f(s.qkv_q, qkv);
    ok &= alloc_f(s.qkv_k, qkv);
    ok &= alloc_f(s.qkv_v, qkv);
    ok &= alloc_f(s.conv_q, qkv);
    ok &= alloc_f(s.conv_k, qkv);
    ok &= alloc_f(s.conv_v, qkv);
    ok &= alloc_f(s.f_a_out, cfg.kda_head_dim);
    ok &= alloc_f(s.g_raw, qkv);
    ok &= alloc_f(s.decay_g, qkv);
    ok &= alloc_f(s.beta_out, cfg.n_q_heads);
    ok &= alloc_f(s.delta_out, qkv);
    ok &= alloc_f(s.g_proj_out, qkv);
    ok &= alloc_f(s.gate_out, qkv);

    ok &= alloc_f(s.q_lora_out, cfg.q_lora_rank);
    ok &= alloc_f(s.q_proj_out, (size_t)qh * cfg.key_length_mla);
    ok &= alloc_f(s.q_nope, (size_t)qh * qk_nope);
    ok &= alloc_f(s.q_pe, (size_t)qh * cfg.rope_dim);
    ok &= alloc_f(s.kv_a_out, cfg.key_length);
    ok &= alloc_f(s.kv_cmpr_normed, cfg.kv_lora_rank);
    ok &= alloc_f(s.absorbed_q, (size_t)qh * cfg.key_length);
    ok &= alloc_f(s.mla_attn_out, (size_t)qh * cfg.value_length_mla);
    ok &= alloc_f(s.gate_proj_out, (size_t)qh * cfg.value_length_mla);

    ok &= alloc_f(s.router_logits, cfg.n_experts);
    ok &= alloc_f(s.router_w, cfg.top_k);
    ok &= alloc_i(s.router_ids, cfg.top_k);
    ok &= alloc_f(s.routed_down_out, cfg.expert_latent);
    ok &= alloc_f(s.moe_scratch, (size_t)cfg.top_k * cfg.moe_ffn);
    // ONE ALLOCATION, TWO TENSORS, BECAUSE ONE COLLECTIVE COVERS BOTH.
    //
    // The expert accumulator (expert_latent) and the shared-expert partial (hidden)
    // are both partial sums over the SAME rank band, produced in the same phase and
    // reduced at the same point. Placing them adjacently lets the driver reduce
    // expert_latent + hidden in a single call instead of issuing a second collective
    // for the shared expert — see kimi_k3_partial_buffer, which reports the fused
    // width, and weight_plan.cpp on why the shared expert is banded at all.
    //
    // moe_out MUST come first: the fused buffer's base is what the collective
    // reduces from, and routed_norm/routed_up read the latent prefix in place.
    ok &= alloc_f(s.moe_fused, (size_t)cfg.expert_latent + (size_t)H);
    if (ok) {
        s.moe_out   = s.moe_fused;
        s.shexp_out = s.moe_fused + cfg.expert_latent;
    }
    ok &= alloc_f(s.dense_gate, cfg.dense_ffn);
    ok &= alloc_f(s.dense_up, cfg.dense_ffn);
    ok &= alloc_f(s.dense_situ, cfg.dense_ffn);
    // Same bound kimi_k3_alloc_state uses for max_ckpt, +1 for the current stream.
    // Sized from cfg rather than from the state so the scratch stays allocatable
    // without one; a mix that somehow saw more checkpoints than the bank can hold
    // would already have failed the `n_ckpt >= max_ckpt` guard in the forward.
    ok &= alloc_f(s.res_scores,
                  cfg.attn_res_block_size > 0
                      ? (size_t)((cfg.n_layers + cfg.attn_res_block_size - 1) /
                                 cfg.attn_res_block_size) + 1
                      : 1);
    ok &= alloc_bytes(s.proj_q8, k3k::k3_q8_0_bytes(cfg.dense_ffn));
    ok &= alloc_bytes(s.moe_q8,
                      k3k::k3_moe_q8_k_bytes(cfg.expert_latent, cfg.moe_ffn, cfg.top_k));

    if (!ok) { kimi_k3_forward_free_scratch(fwd); return false; }
    return true;
}

void kimi_k3_forward_free_scratch(KimiK3Forward& fwd) {
    if (!fwd.s) return;
    for (void* p : fwd.s->owned) cudaFree(p);
    delete fwd.s;
    fwd.s = nullptr;
}

// ---------------------------------------------------------------------------
// Forward, one layer
// ---------------------------------------------------------------------------

bool kimi_k3_forward_layer(KimiK3Forward& fwd, int layer, const float* hidden_in,
                          float* hidden_out) {
    return kimi_k3_forward_layer_phase(fwd, layer, K3LayerPhase::All,
                                      hidden_in, hidden_out);
}

float* kimi_k3_partial_buffer(KimiK3Forward& fwd, int layer, K3LayerPhase phase,
                             int* count) {
    const KimiK3Config& cfg = *fwd.cfg;
    auto& s = *fwd.s;
    if (phase == K3LayerPhase::Attn) {
        if (count) *count = cfg.hidden;
        return s.attn_out;
    }
    if (phase == K3LayerPhase::FfnPartial) {
        // The leading dense layer reduces ffn_down's full-width partial; every MoE
        // layer reduces the expert accumulator, which is LATENT wide. Two different
        // buffers and two different widths, decided by the same branch the forward
        // took — not by the caller guessing.
        if (layer < cfg.leading_dense) {
            if (count) *count = cfg.hidden;
            return s.ffn_out;
        }
        // FUSED: the expert accumulator AND the shared-expert partial, adjacent by
        // construction (kimi_k3_forward_alloc_scratch). Both are partial sums over
        // this rank's bands and both are consumed after the same reduce, so they are
        // reduced as one payload rather than two collectives. The width is what makes
        // that true — a caller that reduced only expert_latent here would leave the
        // shared expert unreduced and every rank would add its own eighth of it.
        if (count) *count = cfg.expert_latent + cfg.hidden;
        return s.moe_fused;
    }
    if (count) *count = 0;
    return nullptr;
}

float* kimi_k3_swap_partial_buffer(KimiK3Forward& fwd, K3LayerPhase phase,
                                  float* buf) {
    auto& s = *fwd.s;
    float* old = nullptr;
    if (phase == K3LayerPhase::Attn) {
        old = s.attn_out;
        s.attn_out = buf;
    } else if (phase == K3LayerPhase::FfnPartial) {
        // The FUSED buffer moves, and both views move with it — they are offsets into
        // one reduced payload, so repointing the base alone would leave the shared
        // expert writing the old allocation while the collective reduced the new one.
        old = s.moe_fused;
        s.moe_fused = buf;
        s.moe_out   = buf;
        s.shexp_out = buf + fwd.cfg->expert_latent;
    }
    return old;
}


// ---------------------------------------------------------------------------
// Launch-failure guard.
//
// 18 of the k3 kernel launchers return void, and nothing on this path polls
// cudaGetLastError. A launch that fails its CONFIGURATION — too much dynamic shared
// memory, a bad grid — never runs, returns no status anybody reads, and leaves the
// destination buffer holding whatever was in it before. Every scratch buffer here is
// reused across layers, so that is the PREVIOUS layer's tensor: the model stays
// fluent and the output is quietly wrong.
//
// That is not hypothetical. mla_decode_attn_kernel sized its dynamic shared memory by
// context length and stopped launching at n_ctx = 11,767; past that, all 24 MLA layers
// silently reused stale scratch, and a 32k decode measured FASTER than a 4k one because
// it was skipping a quarter of the model. Confirmed on 8x H200:
//
//     [MLA-LAUNCH-FAIL] layer 3 n_ctx=32763 : invalid argument
//
// gittensor-ai-lab/sparkinfer-k3#33 fixes that kernel's sizing, and this does NOT
// duplicate it — the sizing fix is that PR's, and it is the right fix. This closes the
// other half: the reason the failure was invisible for as long as it was. With this in
// place a launch-config regression in ANY of the 18 launchers fails the forward loudly
// instead of degrading the output.
//
// Polled once per phase rather than after every launch: three calls per layer, ~279 per
// token, against ~2300 kernel launches. cudaGetLastError is host-side and does not
// synchronise, so it costs no device time and cannot hide a stall.
//
// It reports the LAST error in the phase, not the first. That is enough to fail the run
// and name the layer and phase; bisecting to the exact kernel is what the debug callback
// is for.
static bool k3_check_launch(int layer, K3LayerPhase phase) {
    const cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return true;
    const char* pn = phase == K3LayerPhase::All        ? "All"
                   : phase == K3LayerPhase::Attn       ? "Attn"
                   : phase == K3LayerPhase::FfnPartial ? "FfnPartial" : "FfnFinish";
    std::fprintf(stderr,
                 "[k3] LAUNCH FAILED at layer %d, phase %s: %s\n"
                 "     A kernel did not launch. Its output buffer still holds the "
                 "previous layer's data,\n"
                 "     so continuing would produce fluent, wrong output. Failing the "
                 "forward instead.\n",
                 layer, pn, cudaGetErrorString(e));
    return false;
}

bool kimi_k3_forward_layer_phase(KimiK3Forward& fwd, int layer, K3LayerPhase phase,
                                const float* hidden_in, float* hidden_out) {
    const KimiK3Config& cfg = *fwd.cfg;
    const KimiK3LayerWeights& L = fwd.w->layers[layer];
    KimiK3RuntimeState& st = *fwd.state;
    auto& s = *fwd.s;
    cudaStream_t stream = fwd.stream;
    const float eps = cfg.rms_eps;

    const bool do_attn   = (phase == K3LayerPhase::All || phase == K3LayerPhase::Attn);
    const bool do_ffn_p  = (phase == K3LayerPhase::All || phase == K3LayerPhase::FfnPartial);
    const bool do_ffn_f  = (phase == K3LayerPhase::All || phase == K3LayerPhase::FfnFinish);

    const int H = cfg.hidden;
    // THIS RANK's KDA widths. Under every policy but ExpertsAndKda these are the
    // full dimensions, because KimiK3Weights::kda is left at the tp=1 identity --
    // so the unsharded path is the same code with the same values, not a branch.
    //
    // qh IS this rank's MLA head count, and the note that used to sit here said
    // the opposite: that head-sharding MLA "divides the FLOPs but not the bytes",
    // because MQA means every rank walks the whole latent cache. That is wrong for
    // this kernel and the profile is what corrected it. mla_decode_attn_hbatch
    // stages 12 heads per block, so 96 heads is EIGHT passes over the 302 MB
    // latent cache per layer; a 12-head band is ONE pass. The bytes divide by 8
    // because the unsharded kernel was already reading the cache eight times.
    //
    // Under every other policy `mla` holds the tp=1 identity, so this is the same
    // value the replicated path always had rather than a branch that can drift.
    const tp::KdaShardDims& kd = fwd.w->kda;
    const tp::MlaShardDims& md = fwd.w->mla;
    const int qkv = kd.qkv;
    const int qh = md.n_heads;
    const int qk_nope = cfg.key_length_mla - cfg.rope_dim;
    const int res_bs = cfg.attn_res_block_size;
    const bool banked = res_bs > 0 && (layer % res_bs == 0);
    const bool ggml_qact_proj = qact_proj();
    const bool ggml_qact_moe = qact_moe();

    auto proj = [&](float* y, const float* x, const KimiK3Tensor& W, int N, int K) {
        if (!W.ok()) return false;
        if (ggml_qact_proj)
            return k3k::k3_proj_ggml_f32(y, x, W.data, W.type, N, K,
                                         s.proj_q8, stream);
        return k3k::k3_proj_f32(y, x, W.data, W.type, N, K, stream);
    };

    // ---- phase 1: attention. Ends holding a FULL-WIDTH PARTIAL SUM,
    // because attn_output is ColShard. The driver reduces it before phase 2. ----
    if (do_attn) {
        // --- pre-attention mix, then bank push (raw pre-mix value) ---
        if (res_bs > 0) {
            if (!L.attn_res_score.ok()) return false;
            k3k::attn_res_mix_f32(s.mixed, st.res_bank, hidden_in,
                                  (const float*)L.attn_res_score.data, H, st.n_ckpt,
                                  eps, stream, s.res_scores);
            if (banked) {
                if (st.n_ckpt >= st.max_ckpt) return false;
                cudaMemcpyAsync(st.res_bank + (size_t)st.n_ckpt * H, hidden_in,
                                (size_t)H * sizeof(float), cudaMemcpyDeviceToDevice, stream);
                ++st.n_ckpt;
            }
        } else {
            cudaMemcpyAsync(s.mixed, hidden_in, (size_t)H * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        }
        if (fwd.debug) fwd.debug("attn_res_mix", layer, s.mixed, H);

        if (!L.attn_norm.ok()) return false;
        k3k::rms_norm_f32(s.normed, s.mixed, (const float*)L.attn_norm.data, H, eps, stream);
        if (fwd.debug) fwd.debug("attn_norm", layer, s.normed, H);

        k3_profiler().start(L.is_kda ? "attn_kda" : "attn_mla", stream);
        if (L.is_kda) {
            const int kda_ord = kimi_k3_kda_ordinal(cfg, layer);
            if (kda_ord < 0) return false;
            const int head_dim = cfg.kda_head_dim;
            const int n_head = kd.n_heads;              // heads THIS rank owns

            // The replicated per-channel vectors, read at this rank's band.
            // ssm_conv1d_* is [conv_kernel, 1, qkv] and CHANNEL-MAJOR -- the kernel
            // reads w + c*d_conv -- so the band is a pointer offset, not a copy.
            // At tp=1 and under ExpertsOnly every offset here is 0, so these are the
            // same pointers the unsharded path always used.
            const int    ch_off = kd.qkv_band.offset;
            const int    hd_off = kd.head_band.offset;
            const float* w_cq = (const float*)L.ssm_conv1d_q.data + (size_t)ch_off * cfg.kda_conv_kernel;
            const float* w_ck = (const float*)L.ssm_conv1d_k.data + (size_t)ch_off * cfg.kda_conv_kernel;
            const float* w_cv = (const float*)L.ssm_conv1d_v.data + (size_t)ch_off * cfg.kda_conv_kernel;
            const float* w_dt = (const float*)L.ssm_dt_bias.data + ch_off;
            const float* w_a  = (const float*)L.ssm_a.data + hd_off;

            // attn_q / attn_k / attn_v / ssm_g all read the SAME s.normed at the same
            // [qkv, H] shape, so they go out as one launch instead of four, and one
            // activation load feeds all four. ssm_g is hoisted up here from below to
            // join them: s.normed is written once at attn_norm above and is not touched
            // again anywhere in this block, so computing g early is the same value at a
            // different point in the stream order. Its consumer (kda_gate_out_f32) still
            // runs after, on the same stream.
            //
            // k3_proj_f32_x4 returns false for anything outside its narrow contract
            // (non-Q8_0, mismatched shape, missing tensor) — that is "use the slow path",
            // not an error, so the fallback below is the general case, not dead code.
            const bool fused_qkvg =
                !ggml_qact_proj &&
                L.attn_q.ok() && L.attn_k.ok() && L.attn_v.ok() && L.ssm_g.ok() &&
                L.attn_q.type == 8 && L.attn_k.type == 8 &&
                L.attn_v.type == 8 && L.ssm_g.type == 8 &&
                k3k::k3_proj_f32_x4(s.qkv_q, s.qkv_k, s.qkv_v, s.g_proj_out, s.normed,
                                    L.attn_q.data, L.attn_k.data, L.attn_v.data,
                                    L.ssm_g.data, L.attn_q.type, qkv, H, stream);
            if (!fused_qkvg) {
                if (!proj(s.qkv_q, s.normed, L.attn_q, qkv, H)) return false;
                if (!proj(s.qkv_k, s.normed, L.attn_k, qkv, H)) return false;
                if (!proj(s.qkv_v, s.normed, L.attn_v, qkv, H)) return false;
            }

            if (!L.ssm_conv1d_q.ok() || !L.ssm_conv1d_k.ok() || !L.ssm_conv1d_v.ok())
                return false;
            k3k::kda_conv_step_f32(s.conv_q, st.conv_state_q[kda_ord], s.qkv_q,
                                   w_cq, cfg.kda_conv_kernel,
                                   qkv, stream);
            k3k::kda_conv_step_f32(s.conv_k, st.conv_state_k[kda_ord], s.qkv_k,
                                   w_ck, cfg.kda_conv_kernel,
                                   qkv, stream);
            k3k::kda_conv_step_f32(s.conv_v, st.conv_state_v[kda_ord], s.qkv_v,
                                   w_cv, cfg.kda_conv_kernel,
                                   qkv, stream);
            if (fwd.debug) fwd.debug("dbg_conv_q", layer, s.conv_q, qkv);
            if (fwd.debug) fwd.debug("dbg_conv_v", layer, s.conv_v, qkv);

            // q gets the extra 1/sqrt(head_dim) scale the reference applies before the
            // scan; k does not (see kda_decode_step_f32's contract on pre-scaled q).
            k3k::l2_norm_heads_f32(s.conv_q, s.conv_q, head_dim, n_head,
                                   1.0f / std::sqrt((float)head_dim), eps, stream);
            k3k::l2_norm_heads_f32(s.conv_k, s.conv_k, head_dim, n_head, 1.0f, eps, stream);
            if (fwd.debug) fwd.debug("dbg_l2_q", layer, s.conv_q, qkv);

            if (!proj(s.f_a_out, s.normed, L.ssm_f_a, head_dim, H)) return false;
            if (!proj(s.g_raw, s.f_a_out, L.ssm_f_b, qkv, head_dim)) return false;
            if (!L.ssm_dt_bias.ok()) return false;
            k3k::k3_add_f32(s.g_raw, s.g_raw, w_dt, qkv, stream);

            if (!L.ssm_a.ok()) return false;
            k3k::kda_decay_gate_f32(s.decay_g, s.g_raw, w_a,
                                    head_dim, n_head, cfg.kda_gate_lower_bound, stream);

            // ssm_beta is [n_heads, hidden] and stays REPLICATED: its suffix is shared
            // with Qwen's GDN, so one weight_plan entry governs both models. Project
            // all cfg.n_q_heads rows and read this rank's band -- 96 rows of a
            // 7168-wide Q8_0 matrix is 0.7 MB against 468 MB of projections, so the
            // duplicated work is not worth a rule that would change another model.
            if (!proj(s.beta_out, s.normed, L.ssm_beta, cfg.n_q_heads, H)) return false;
            // beta is the delta-rule update rate: llama.cpp's build_kda_layer applies
            // a sigmoid to the projection before the scan consumes it. Omitting this
            // left beta unbounded (rms ~1.7 vs the correct ~0.75) and scaled the whole
            // KDA output — the layer-0 divergence vs llama that this restores.
            k3k::sigmoid_inplace_f32(s.beta_out, cfg.n_q_heads, stream);

            if (fwd.debug) fwd.debug("dbg_decay_g", layer, s.decay_g, qkv);
            if (fwd.debug) fwd.debug("dbg_beta", layer, s.beta_out, cfg.n_q_heads);
            k3k::kda_decode_step_f32(s.delta_out, st.delta_state[kda_ord],
                                     s.conv_q, s.conv_k, s.conv_v, s.decay_g,
                                     s.beta_out + hd_off,
                                     head_dim, n_head, stream);
            if (fwd.debug) fwd.debug("dbg_delta_out", layer, s.delta_out, qkv);

            // Already computed above when the q/k/v/g fusion took the fast path.
            if (!fused_qkvg && !proj(s.g_proj_out, s.normed, L.ssm_g, qkv, H)) return false;
            if (!L.ssm_norm.ok()) return false;
            k3k::kda_gate_out_f32(s.gate_out, s.delta_out, (const float*)L.ssm_norm.data,
                                  s.g_proj_out, head_dim, n_head, eps, stream);
            if (fwd.debug) fwd.debug("dbg_gate_out", layer, s.gate_out, qkv);

            if (!proj(s.attn_out, s.gate_out, L.attn_output, H, qkv)) return false;
            if (fwd.debug) fwd.debug("kda_out", layer, s.attn_out, H);
        } else {
            const int mla_ord = kimi_k3_mla_ordinal(cfg, layer);
            if (mla_ord < 0) return false;

            if (L.has_q_lora) {
                if (!proj(s.q_lora_out, s.normed, L.attn_q_a, cfg.q_lora_rank, H)) return false;
                if (!L.attn_q_a_norm.ok()) return false;
                k3k::rms_norm_f32(s.q_lora_out, s.q_lora_out,
                                  (const float*)L.attn_q_a_norm.data, cfg.q_lora_rank, eps,
                                  stream);
                if (!proj(s.q_proj_out, s.q_lora_out, L.attn_q_b, qh * cfg.key_length_mla,
                         cfg.q_lora_rank))
                    return false;
            } else {
                if (!proj(s.q_proj_out, s.normed, L.attn_q_dense, qh * cfg.key_length_mla, H))
                    return false;
            }

            // De-interleave [qh, key_length_mla] into q_nope [qh, qk_nope] and
            // q_pe [qh, rope_dim] — each head's 192 values are qk_nope(128) then
            // rope_dim(64) concatenated, so this is a strided 2-D copy, not a plain split.
            cudaMemcpy2DAsync(s.q_nope, (size_t)qk_nope * sizeof(float),
                              s.q_proj_out, (size_t)cfg.key_length_mla * sizeof(float),
                              (size_t)qk_nope * sizeof(float), qh,
                              cudaMemcpyDeviceToDevice, stream);
            cudaMemcpy2DAsync(s.q_pe, (size_t)cfg.rope_dim * sizeof(float),
                              (const char*)s.q_proj_out + (size_t)qk_nope * sizeof(float),
                              (size_t)cfg.key_length_mla * sizeof(float),
                              (size_t)cfg.rope_dim * sizeof(float), qh,
                              cudaMemcpyDeviceToDevice, stream);

            if (!proj(s.kv_a_out, s.normed, L.attn_kv_a_mqa, cfg.key_length, H)) return false;
            if (!L.attn_kv_a_norm.ok()) return false;
            k3k::rms_norm_f32(s.kv_cmpr_normed, s.kv_a_out,
                              (const float*)L.attn_kv_a_norm.data, cfg.kv_lora_rank, eps,
                              stream);
            if (fwd.debug) fwd.debug("dbg_kvcmpr", layer, s.kv_cmpr_normed, cfg.kv_lora_rank);

            // K-cache row for this position: concat(normed kv_cmpr, RAW k_pe).
            float* row = st.mla_kv_cache[mla_ord] + (size_t)st.position * cfg.key_length;
            cudaMemcpyAsync(row, s.kv_cmpr_normed, (size_t)cfg.kv_lora_rank * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(row + cfg.kv_lora_rank, s.kv_a_out + cfg.kv_lora_rank,
                            (size_t)cfg.rope_dim * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);

            if (!L.attn_k_b.ok() || !L.attn_v_b.ok()) return false;
            k3k::mla_absorb_q_f32(s.absorbed_q, s.q_nope, s.q_pe,
                                  (const float*)L.attn_k_b.data, qk_nope, cfg.kv_lora_rank,
                                  cfg.rope_dim, qh, stream);
            if (fwd.debug) fwd.debug("mla_absorb_q", layer, s.absorbed_q, qh * cfg.key_length);

            const float mla_scale = 1.0f / std::sqrt((float)cfg.key_length_mla);
            k3k::mla_decode_attn_f32(s.mla_attn_out, s.absorbed_q, st.mla_kv_cache[mla_ord],
                                     (const float*)L.attn_v_b.data, cfg.key_length,
                                     cfg.kv_lora_rank, cfg.value_length_mla, qh,
                                     st.position + 1, mla_scale, stream);
            if (fwd.debug) fwd.debug("dbg_preattn", layer, s.mla_attn_out, qh * cfg.value_length_mla);

            if (L.has_attn_gate) {
                if (!proj(s.gate_proj_out, s.normed, L.attn_gate, qh * cfg.value_length_mla, H))
                    return false;
                if (fwd.debug) fwd.debug("dbg_gateproj", layer, s.gate_proj_out, qh * cfg.value_length_mla);
                k3k::mla_gate_out_f32(s.mla_attn_out, s.mla_attn_out, s.gate_proj_out,
                                      (int64_t)qh * cfg.value_length_mla, stream);
                if (fwd.debug) fwd.debug("dbg_postgate", layer, s.mla_attn_out, qh * cfg.value_length_mla);
            }

            if (!proj(s.attn_out, s.mla_attn_out, L.attn_output, H, qh * cfg.value_length_mla))
                return false;
            if (fwd.debug) fwd.debug("mla_out", layer, s.attn_out, H);
        }

        k3_profiler().stop(L.is_kda ? "attn_kda" : "attn_mla", stream);
    }

    // ---- phase 2: attention residual, FFN norm, and the FFN's PARTIAL. ----
    if (do_ffn_p) {
        // --- combine: replace on a checkpoint layer, add otherwise. Uses hidden_in
        // (the RAW pre-mix value), not s.mixed — the reference's residual add is
        // against the unmixed prefix_sum, only the norm/attention input was mixed. ---
        if (banked) {
            cudaMemcpyAsync(hidden_out, s.attn_out, (size_t)H * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        } else {
            k3k::k3_add_f32(hidden_out, hidden_in, s.attn_out, H, stream);
        }

        // --- pre-FFN mix, no bank push ---
        if (res_bs > 0) {
            if (!L.ffn_res_score.ok()) return false;
            k3k::attn_res_mix_f32(s.mixed2, st.res_bank, hidden_out,
                                  (const float*)L.ffn_res_score.data, H, st.n_ckpt, eps,
                                  stream, s.res_scores);
        } else {
            cudaMemcpyAsync(s.mixed2, hidden_out, (size_t)H * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        }
        if (!L.ffn_norm.ok()) return false;
        k3k::rms_norm_f32(s.normed2, s.mixed2, (const float*)L.ffn_norm.data, H, eps, stream);
        if (fwd.debug) fwd.debug("ffn_norm", layer, s.normed2, H);

        k3_profiler().start(layer < cfg.leading_dense ? "ffn_dense" : "ffn_moe", stream);
        if (layer < cfg.leading_dense) {
            if (!proj(s.dense_gate, s.normed2, L.ffn_gate, cfg.dense_ffn, H)) return false;
            if (!proj(s.dense_up, s.normed2, L.ffn_up, cfg.dense_ffn, H)) return false;
            if (fwd.debug) fwd.debug("dbg_dense_gate", layer, s.dense_gate, cfg.dense_ffn);
            if (fwd.debug) fwd.debug("dbg_dense_up", layer, s.dense_up, cfg.dense_ffn);
            k3k::situ_f32(s.dense_situ, s.dense_gate, s.dense_up, cfg.dense_ffn,
                         cfg.situ_beta, cfg.situ_linear_beta, stream);
            if (fwd.debug) fwd.debug("dbg_dense_situ", layer, s.dense_situ, cfg.dense_ffn);
            if (!proj(s.ffn_out, s.dense_situ, L.ffn_down, H, cfg.dense_ffn)) return false;
        } else {
            if (!proj(s.router_logits, s.normed2, L.ffn_gate_inp, cfg.n_experts, H))
                return false;
            if (fwd.debug) fwd.debug("dbg_router_logits", layer, s.router_logits, cfg.n_experts);
            if (!L.exp_probs_b.ok()) return false;
            k3k::moe_router_noaux_tc_f32(s.router_w, s.router_ids, s.router_logits,
                                         (const float*)L.exp_probs_b.data, cfg.n_experts,
                                         cfg.top_k, /*n_tokens=*/1, /*norm_w=*/true,
                                         /*w_scale=*/1.0f, stream);
            if (fwd.debug) fwd.debug("dbg_router_w", layer, s.router_w, cfg.top_k);

            // routed_down feeds the dispatch UNNORMALISED — routed_norm (if present)
            // normalises the dispatch's OUTPUT, not this. See build_latent_moe in the
            // reference: build_moe_ffn runs first, then "if (layer.ffn_routed_norm)
            // moe_out = build_norm(moe_out, ...)", and only after that does
            // ffn_routed_up run. Getting this backwards (norm between routed_down and
            // the dispatch) was an earlier version's bug, same class as the ssm_norm fix.
            if (!proj(s.routed_down_out, s.normed2, L.ffn_routed_down, cfg.expert_latent, H))
                return false;

            if (!L.ffn_gate_exps.ok() || !L.ffn_up_exps.ok() || !L.ffn_down_exps.ok())
                return false;
            // THE EXPERT BAND. The router above ran on the replicated ffn_gate_inp,
            // so every rank picked the SAME top_k global ids; this rank evaluates only
            // the ones whose weights it holds, leaving a partial sum in s.moe_out that
            // the driver reduces at expert_latent before phase 3. At tp_size 1 the
            // band covers every expert and the call is unchanged.
            const int expert_begin  = fwd.w->shard.expert_band.offset;
            const int n_local_exp   = fwd.w->shard.tp_size > 1
                                        ? fwd.w->shard.expert_band.extent : 0;
            const bool moe_ok = k3k::moe_expert_ffn_f32_by_type(
                s.moe_out, s.moe_scratch, s.routed_down_out, s.router_ids, s.router_w,
                L.ffn_gate_exps.data, L.ffn_up_exps.data, L.ffn_down_exps.data,
                cfg.expert_latent, cfg.moe_ffn, cfg.top_k, cfg.situ_beta,
                cfg.situ_linear_beta, L.ffn_gate_exps.type, stream,
                expert_begin, n_local_exp, ggml_qact_moe ? s.moe_q8 : nullptr);
            if (!moe_ok) return false;
            // This rank's PARTIAL expert sum: it legitimately differs per rank and
            // from the tp=1 value. dbg_moe_reduced below is the one that must match.
            if (fwd.debug) fwd.debug("dbg_moe_partial", layer, s.moe_out, cfg.expert_latent);

            // THE SHARED EXPERT, IN THIS PHASE ON PURPOSE.
            //
            // It reads `normed2` and nothing downstream of the collective, so it can
            // run here — and running here is what lets its partial ride the expert
            // reduce instead of needing one of its own (weight_plan.cpp). gate/up are
            // RowShard so this rank owns a band of the ffn width; situ is elementwise
            // and preserves the band; down is ColShard over the same band, so what
            // lands in shexp_out is a full-width PARTIAL, exactly like moe_out.
            //
            // Widths come from rank_ne, never from cfg: block_aligned_band hands a
            // remainder to the low ranks, so a rank that derived n_ff_shexp/tp_size
            // would be wrong on precisely the ranks that differ.
            if (L.has_shared_experts) {
                if (!L.ffn_gate_shexp.ok() || !L.ffn_up_shexp.ok() ||
                    !L.ffn_down_shexp.ok()) return false;
                const int shexp_band = (int)L.ffn_gate_shexp.rank_ne[1];
                if (shexp_band <= 0 || (int)L.ffn_down_shexp.rank_ne[0] != shexp_band)
                    return false;
                // THE BAND MUST BE REAL, AND THIS IS WHERE THAT IS PROVEN.
                //
                // shexp_out is a view into the buffer the driver reduces, so a rank
                // holding the FULL tensor here computes a complete shared-expert
                // output and the collective sums tp_size copies of it — fluent output,
                // silently wrong, invisible to any timing benchmark. That is exactly
                // what happens if the loader's ExpertsOnly short-circuit
                // (upload_sliced, this file) is not kept in step with weight_plan.cpp.
                // Refuse instead: a load that cannot band is a bug, not a fallback.
                const int tp_size = fwd.w->shard.tp_size;
                if (tp_size > 1 && shexp_band * tp_size != cfg.moe_ffn * cfg.n_shared) {
                    std::fprintf(stderr,
                                 "[k3] layer %d: shared expert not banded — rank width "
                                 "%d x tp_size %d != %d. The reduce would sum %d copies "
                                 "of a complete shared-expert output.\n",
                                 layer, shexp_band, tp_size,
                                 cfg.moe_ffn * cfg.n_shared, tp_size);
                    return false;
                }
                if (!proj(s.dense_gate, s.normed2, L.ffn_gate_shexp, shexp_band, H))
                    return false;
                if (!proj(s.dense_up, s.normed2, L.ffn_up_shexp, shexp_band, H))
                    return false;
                k3k::situ_f32(s.dense_situ, s.dense_gate, s.dense_up, shexp_band,
                             cfg.situ_beta, cfg.situ_linear_beta, stream);
                if (!proj(s.shexp_out, s.dense_situ, L.ffn_down_shexp, H, shexp_band))
                    return false;
                if (fwd.debug) fwd.debug("dbg_shexp_partial", layer, s.shexp_out, H);
            } else {
                // The fused buffer is reduced whole, so a layer without a shared
                // expert must still present a well-defined summand there.
                cudaMemsetAsync(s.shexp_out, 0, (size_t)H * sizeof(float), stream);
            }
        }
    }

    // ---- phase 3: everything downstream of the FFN collective. Every weight
    // here is Replicate and reads the ALREADY-REDUCED value — routed_norm is an
    // rms_norm, so the sum cannot be deferred past it. ----
    if (do_ffn_f) {
        if (layer >= cfg.leading_dense) {
            // Post-collective: every rank must now hold the SAME complete expert sum,
            // equal to what tp=1 computed. If this tag differs, the dispatch or the
            // reduce is wrong; if it matches and ffn_out still differs, the fault is
            // downstream in routed_norm / routed_up / shexp.
            if (fwd.debug) fwd.debug("dbg_moe_reduced", layer, s.moe_out, cfg.expert_latent);
            if (L.has_routed_norm) {
                if (!L.ffn_routed_norm.ok()) return false;
                k3k::rms_norm_f32(s.moe_out, s.moe_out,
                                  (const float*)L.ffn_routed_norm.data, cfg.expert_latent,
                                  eps, stream);
            }

            if (fwd.debug) fwd.debug("dbg_moe_normed", layer, s.moe_out, cfg.expert_latent);
            if (!proj(s.ffn_out, s.moe_out, L.ffn_routed_up, H, cfg.expert_latent))
                return false;
            if (fwd.debug) fwd.debug("dbg_routed_up", layer, s.ffn_out, H);

            // The shared expert was computed in phase FfnPartial and its partial has
            // been reduced along with the expert accumulator, so what is left here is
            // the add. Both contributions are at hidden width by this point:
            // ffn_routed_up lifted the latent, shexp_out was already hidden-wide.
            if (L.has_shared_experts) {
                if (fwd.debug) fwd.debug("dbg_shexp", layer, s.shexp_out, H);
                k3k::k3_add_f32(s.ffn_out, s.ffn_out, s.shexp_out, H, stream);
            }
        }
        k3_profiler().stop(layer < cfg.leading_dense ? "ffn_dense" : "ffn_moe", stream);
        if (fwd.debug) fwd.debug("ffn_out", layer, s.ffn_out, H);

        // FFN residual is ALWAYS an add, never a replace.
        k3k::k3_add_f32(hidden_out, hidden_out, s.ffn_out, H, stream);
        if (fwd.debug) fwd.debug("l_out", layer, hidden_out, H);
    }
    return k3_check_launch(layer, phase);
}

// ---------------------------------------------------------------------------
// Forward, one token
// ---------------------------------------------------------------------------

bool kimi_k3_forward_token(KimiK3Forward& fwd, int token_id, float* out_logits) {
    const KimiK3Config& cfg = *fwd.cfg;
    const KimiK3Weights& w = *fwd.w;
    cudaStream_t stream = fwd.stream;
    const int H = cfg.hidden;

    // THE CROSS-LAYER RESIDUAL BANK IS PER-TOKEN, NOT PERSISTENT. In the reference
    // it is `ckpts`, a member of the per-forward-pass `graph` object, so it is
    // constructed empty on every forward pass and discarded at the end of it — the
    // checkpoints a token banks are visible only to LATER LAYERS OF THAT SAME TOKEN,
    // never to the next token.
    //
    // This is the opposite lifetime from the KDA recurrent state and the MLA KV
    // cache in the same struct, which DO persist across tokens (that is the entire
    // point of them). Three kinds of state, two different lifetimes, one struct —
    // so resetting it is easy to forget, and forgetting it is not subtle: with
    // max_ckpt = ceil(93/12) = 8 exactly filled by one token's eight checkpoint
    // layers, token 2's first push hits the `n_ckpt >= max_ckpt` guard in
    // forward_layer and the whole call fails. Caught by reasoning through a
    // multi-token decode before ever running one.
    //
    // Deliberately NOT reset inside forward_layer: that entry point is also used
    // standalone for per-layer validation, where the caller owns bank lifetime.
    fwd.state->n_ckpt = 0;

    float* x = nullptr;
    float* x_next = nullptr;
    if (cudaMalloc(&x, (size_t)H * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&x_next, (size_t)H * sizeof(float)) != cudaSuccess) {
        cudaFree(x);
        return false;
    }

    bool ok = true;
    if (!w.token_embd.ok()) ok = false;
    if (ok) {
        // Row gather: token_id's row is hidden contiguous values starting at
        // token_id * row_bytes(hidden, wtype).
        long row_bytes = 0;
        if (w.token_embd.type == 0) row_bytes = (long)H * sizeof(float);
        else if (w.token_embd.type == 8) row_bytes = (long)(H / 32) * 34;
        else ok = false;
        if (ok) {
            const char* base = (const char*)w.token_embd.data + (size_t)token_id * row_bytes;
            ok = k3k::dequant_f32_by_type(x, base, H, w.token_embd.type, stream);
        }
    }
    if (ok && fwd.debug) fwd.debug("inp_embd", -1, x, H);

    for (int layer = 0; ok && layer < cfg.n_layers; ++layer) {
        ok = kimi_k3_forward_layer(fwd, layer, x, x_next);
        std::swap(x, x_next);
    }

    if (ok) {
        if (cfg.attn_res_block_size > 0) {
            if (!w.has_output_res_score || !w.output_res_score.ok()) {
                ok = false;
            } else {
                k3k::attn_res_mix_f32(x_next, fwd.state->res_bank, x,
                                      (const float*)w.output_res_score.data, H,
                                      fwd.state->n_ckpt, cfg.rms_eps, stream);
                std::swap(x, x_next);
            }
        }
    }
    if (ok && !w.output_norm.ok()) ok = false;
    if (ok) {
        k3k::rms_norm_f32(x_next, x, (const float*)w.output_norm.data, H, cfg.rms_eps,
                          stream);
        std::swap(x, x_next);
        if (fwd.debug) fwd.debug("result_norm", -1, x, H);
    }

    float* logits_dev = nullptr;
    if (ok) {
        if (cudaMalloc(&logits_dev, (size_t)cfg.vocab * sizeof(float)) != cudaSuccess)
            ok = false;
    }
    if (ok && !w.output.ok()) ok = false;
    if (ok) {
        ok = qact_output()
            ? k3k::k3_proj_ggml_f32(logits_dev, x, w.output.data, w.output.type,
                                    cfg.vocab, H, fwd.s->proj_q8, stream)
            : k3k::k3_proj_f32(logits_dev, x, w.output.data, w.output.type,
                               cfg.vocab, H, stream);
    }
    // The head is outside the layer loop, so the per-phase guard does not cover it.
    // Checked BEFORE the sync so a failed lm_head launch is reported as a launch
    // failure rather than surfacing later as an unrelated error at some other call.
    if (ok) ok = k3_check_launch(cfg.n_layers, K3LayerPhase::All);
    if (ok) {
        cudaStreamSynchronize(stream);
        cudaMemcpy(out_logits, logits_dev, (size_t)cfg.vocab * sizeof(float),
                  cudaMemcpyDeviceToHost);
        ++fwd.state->position;
    }

    if (logits_dev) cudaFree(logits_dev);
    cudaFree(x);
    cudaFree(x_next);
    return ok;
}

// ---------------------------------------------------------------------------
// Layer-split pipeline
// ---------------------------------------------------------------------------

namespace {

// Per-layer device-byte cost, from the GGUF index (no data read). Used to balance
// stages: the leading dense layer is ~1.2 GiB against ~6 GiB for a MoE layer, so
// splitting by layer COUNT would leave stage 0 carrying a fraction of its share.
std::vector<long> layer_bytes(const GGUF& g, const KimiK3Config& cfg) {
    std::vector<long> out(cfg.n_layers, 0);
    for (const auto& name : g.tensor_names()) {
        if (name.rfind("blk.", 0) != 0) continue;
        const size_t dot = name.find('.', 4);
        if (dot == std::string::npos) continue;
        const int idx = std::atoi(name.substr(4, dot - 4).c_str());
        if (idx < 0 || idx >= cfg.n_layers) continue;
        const GGUFTensor* t = g.tensor(name);
        if (t) out[idx] += t->n_bytes;
    }
    return out;
}

}  // namespace

bool kimi_k3_pipeline_init(const GGUF& g, const KimiK3Config& cfg,
                          const K3PlanOptions& opt,
                          const std::vector<int>& devices, int max_ctx,
                          KimiK3Pipeline& out) {
    if (devices.empty()) return false;
    out.cfg = cfg;
    out.opt = opt;
    out.stages.clear();
    out.stages.resize(devices.size());
    out.host_hidden.assign(cfg.hidden, 0.0f);

    // Balance by cumulative byte cost rather than layer count.
    const std::vector<long> lb = layer_bytes(g, cfg);
    long total = 0;
    for (long v : lb) total += v;
    const long per_stage = total / (long)devices.size();

    int layer = 0;
    for (size_t si = 0; si < devices.size(); ++si) {
        KimiK3PipelineStage& st = out.stages[si];
        st.device = devices[si];
        st.first_layer = layer;
        long acc = 0;
        // Last stage takes whatever remains, so no layer is ever dropped by rounding.
        while (layer < cfg.n_layers &&
               (si + 1 == devices.size() || acc < per_stage || layer == st.first_layer)) {
            acc += lb[layer];
            ++layer;
            if (si + 1 < devices.size() && acc >= per_stage) break;
        }
        st.last_layer = layer - 1;
    }
    if (layer != cfg.n_layers) {
        std::fprintf(stderr, "[k3] pipeline split covered %d of %d layers\n",
                     layer, cfg.n_layers);
        return false;
    }

    for (size_t si = 0; si < out.stages.size(); ++si) {
        KimiK3PipelineStage& st = out.stages[si];
        if (cudaSetDevice(st.device) != cudaSuccess) return false;
        const bool first = (si == 0), last = (si + 1 == out.stages.size());
        if (!kimi_k3_load_weights_scoped(g, cfg, opt, st.weights,
                                        st.first_layer, st.last_layer,
                                        /*load_embed=*/first, /*load_head=*/last))
            return false;
        if (!kimi_k3_alloc_state(cfg, max_ctx, st.state, st.first_layer, st.last_layer))
            return false;
        st.fwd.cfg = &out.cfg;
        st.fwd.w = &st.weights;
        st.fwd.state = &st.state;
        st.fwd.opt = out.opt;
        st.fwd.stream = nullptr;
        if (!kimi_k3_forward_alloc_scratch(cfg, st.fwd)) return false;
        if (cudaMalloc(&st.hidden, (size_t)cfg.hidden * sizeof(float)) != cudaSuccess)
            return false;
        if (last && cudaMalloc(&st.logits, (size_t)cfg.vocab * sizeof(float)) != cudaSuccess)
            return false;
        std::fprintf(stderr, "[k3] stage %zu: device %d, layers %d-%d (%.2f GiB)\n",
                     si, st.device, st.first_layer, st.last_layer,
                     [&]{ long a = 0; for (int i = st.first_layer; i <= st.last_layer; ++i) a += lb[i];
                          return a / 1073741824.0; }());
    }
    out.host_bank.assign((size_t)cfg.hidden * out.stages[0].state.max_ckpt, 0.0f);
    return true;
}

bool kimi_k3_pipeline_forward_token(KimiK3Pipeline& p, int token_id, float* out_logits) {
    if (p.stages.empty()) return false;
    const KimiK3Config& cfg = p.cfg;
    const int H = cfg.hidden;

    // The residual bank is per-TOKEN and spans every layer, so it starts empty and
    // then follows the hidden state across every stage boundary. See the header.
    for (auto& st : p.stages) st.state.n_ckpt = 0;

    // ---- stage 0: embed ----
    KimiK3PipelineStage& s0 = p.stages[0];
    if (cudaSetDevice(s0.device) != cudaSuccess) return false;
    if (!s0.weights.token_embd.ok()) return false;
    {
        long row_bytes = 0;
        const int ty = s0.weights.token_embd.type;
        if (ty == 0) row_bytes = (long)H * sizeof(float);
        else if (ty == 8) row_bytes = (long)(H / 32) * 34;
        else return false;
        const char* base = (const char*)s0.weights.token_embd.data +
                           (size_t)token_id * row_bytes;
        if (!k3k::dequant_f32_by_type(s0.hidden, base, H, ty, nullptr)) return false;
    }

    int n_ckpt_carry = 0;
    for (size_t si = 0; si < p.stages.size(); ++si) {
        KimiK3PipelineStage& st = p.stages[si];
        if (cudaSetDevice(st.device) != cudaSuccess) return false;

        if (si > 0) {
            // HANDOFF: hidden state AND the residual bank. Transferring only the
            // hidden state would leave this stage's res_mix scoring against an
            // empty bank — fluent, wrong, and invisible without a reference.
            if (cudaMemcpy(st.hidden, p.host_hidden.data(), (size_t)H * sizeof(float),
                          cudaMemcpyHostToDevice) != cudaSuccess) return false;
            st.state.n_ckpt = n_ckpt_carry;
            if (n_ckpt_carry > 0 &&
                cudaMemcpy(st.state.res_bank, p.host_bank.data(),
                          (size_t)n_ckpt_carry * H * sizeof(float),
                          cudaMemcpyHostToDevice) != cudaSuccess) return false;
        }

        // Debug localization: env-gated per-op sub-tap dump for one target layer.
        st.fwd.debug = nullptr;
        if (const char* sl = std::getenv("K3_SUBTAP_LAYER")) {
            const int target = std::atoi(sl);
            if (const char* dd = std::getenv("K3_DUMP_DIR")) {
                st.fwd.debug = [target, dd](const char* tag, int layer, const float* dev, int64_t n) {
                    if (layer != target) return;
                    std::vector<float> hb((size_t)n);
                    if (cudaMemcpy(hb.data(), dev, (size_t)n * sizeof(float),
                                   cudaMemcpyDeviceToHost) == cudaSuccess) {
                        char p[512];
                        std::snprintf(p, sizeof(p), "%s/sub_%s_%d.bin", dd, tag, layer);
                        if (FILE* f = std::fopen(p, "wb")) {
                            std::fwrite(hb.data(), sizeof(float), (size_t)n, f); std::fclose(f);
                        }
                    }
                };
            }
        }
        for (int L = st.first_layer; L <= st.last_layer; ++L) {
            if (!kimi_k3_forward_layer(st.fwd, L, st.hidden, st.hidden)) {
                std::fprintf(stderr, "[k3] pipeline: layer %d failed on stage %zu\n", L, si);
                return false;
            }
            // Debug localization: env-gated per-layer residual-stream dump. Writes
            // st.hidden (== llama.cpp's l_out-<L>) so the two can be compared op-for-op.
            if (const char* dd = std::getenv("K3_DUMP_DIR")) {
                cudaStreamSynchronize(st.fwd.stream);
                std::vector<float> hb((size_t)H);
                if (cudaMemcpy(hb.data(), st.hidden, (size_t)H * sizeof(float),
                               cudaMemcpyDeviceToHost) == cudaSuccess) {
                    char pth[512];
                    std::snprintf(pth, sizeof(pth), "%s/our_l_out-%d.bin", dd, L);
                    if (FILE* f = std::fopen(pth, "wb")) {
                        std::fwrite(hb.data(), sizeof(float), (size_t)H, f);
                        std::fclose(f);
                    }
                }
            }
        }
        if (cudaDeviceSynchronize() != cudaSuccess) return false;

        if (si + 1 < p.stages.size()) {
            if (cudaMemcpy(p.host_hidden.data(), st.hidden, (size_t)H * sizeof(float),
                          cudaMemcpyDeviceToHost) != cudaSuccess) return false;
            n_ckpt_carry = st.state.n_ckpt;
            if (n_ckpt_carry > 0 &&
                cudaMemcpy(p.host_bank.data(), st.state.res_bank,
                          (size_t)n_ckpt_carry * H * sizeof(float),
                          cudaMemcpyDeviceToHost) != cudaSuccess) return false;
        }
    }

    // ---- last stage: final mix, norm, lm_head ----
    KimiK3PipelineStage& sl = p.stages.back();
    if (cudaSetDevice(sl.device) != cudaSuccess) return false;
    float* x = sl.hidden;
    float* tmp = sl.fwd.s->mixed;   // reuse an H-wide scratch slot
    if (cfg.attn_res_block_size > 0) {
        if (!sl.weights.has_output_res_score || !sl.weights.output_res_score.ok())
            return false;
        k3k::attn_res_mix_f32(tmp, sl.state.res_bank, x,
                              (const float*)sl.weights.output_res_score.data, H,
                              sl.state.n_ckpt, cfg.rms_eps, nullptr);
        std::swap(x, tmp);
    }
    if (!sl.weights.output_norm.ok()) return false;
    k3k::rms_norm_f32(tmp, x, (const float*)sl.weights.output_norm.data, H,
                      cfg.rms_eps, nullptr);
    if (!sl.weights.output.ok()) return false;
    const bool head_ok = qact_output()
        ? k3k::k3_proj_ggml_f32(sl.logits, tmp, sl.weights.output.data,
                                sl.weights.output.type, cfg.vocab, H,
                                sl.fwd.s->proj_q8, nullptr)
        : k3k::k3_proj_f32(sl.logits, tmp, sl.weights.output.data,
                           sl.weights.output.type, cfg.vocab, H, nullptr);
    if (!head_ok)
        return false;
    if (cudaDeviceSynchronize() != cudaSuccess) return false;
    if (cudaMemcpy(out_logits, sl.logits, (size_t)cfg.vocab * sizeof(float),
                  cudaMemcpyDeviceToHost) != cudaSuccess) return false;

    for (auto& st : p.stages) ++st.state.position;
    return true;
}

void kimi_k3_pipeline_free(KimiK3Pipeline& p) {
    for (auto& st : p.stages) {
        cudaSetDevice(st.device);
        kimi_k3_forward_free_scratch(st.fwd);
        kimi_k3_free_state(st.state);
        kimi_k3_free_weights(st.weights);
        if (st.hidden) cudaFree(st.hidden);
        if (st.logits) cudaFree(st.logits);
        st.hidden = nullptr; st.logits = nullptr;
    }
    p.stages.clear();
}

}  // namespace sparkinfer
