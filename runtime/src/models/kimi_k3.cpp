#include "sparkinfer/models/kimi_k3.h"
#include "sparkinfer/models/kimi_k3_prefill.h"

#include "sparkinfer/kernels/kimi_k3.h"
#include "sparkinfer/kernels/kimi_k3_fast.h"
#include "sparkinfer/models/kimi_k3_gguf_manifest.h"
#include "sparkinfer/tp/weight_residency.h"   // plan_tensor_residency for the sharded loader

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <set>
#include <mutex>
#include <utility>
#include <string>
#include <unordered_map>
#include <utility>

namespace sparkinfer {

namespace k3k = sparkinfer::kernels::k3;

namespace {

// Rejoins any DAG lane still outstanding when its scope exits, on every path
// including the error returns inside the layer phases. Joining is idempotent at
// the graph level -- a lane that did no work contributes a no-op edge -- so
// over-joining costs nothing, while under-joining fails the whole capture.
template <class F>
struct K3LaneScope {
    unsigned* open;
    F join;
    K3LaneScope(const K3LaneScope&) = delete;
    K3LaneScope& operator=(const K3LaneScope&) = delete;
    ~K3LaneScope() {
        while (*open) {
            const unsigned bit = *open & (~*open + 1u);   // lowest set lane
            int i = 0;
            while (!((bit >> i) & 1u)) ++i;
            *open &= ~bit;                                // clear FIRST, so this
            join(i);                                      // always terminates
        }
    }
};
template <class F> K3LaneScope(unsigned*, F) -> K3LaneScope<F>;


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

// F16 latent KV cache, ON by default for the same reason qact_proj is: the
// reference this model is scored against (llama.cpp, default type_k = F16)
// already stores its cache in half precision, so the f32 cache was paying twice
// the reference's KV bytes to hold MORE precision than the scoring bar assumes.
// SPARKINFER_K3_KV_F16=0 restores the f32 layout — the same-binary control.
bool kv_f16_on()   { return !env_zero("SPARKINFER_K3_KV_F16"); }

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
        // Load-time only (never inside a capture): build the SoA view of Q8_0
        // weights. Declines everything it does not apply to.
        k3k::k3_q8soa_register(out.data, out.type, out.rank_ne);
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
            k3k::k3_q8soa_register(out.data, out.type, out.rank_ne);
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
        //
        // A HOST-SIDE GATHER WAS TRIED HERE AND IS SLOWER. 2-D MoE's down slice is the
        // narrowest case in the model (1.6M rows of ~150 B), so staging it contiguously
        // before one flat H2D looked like the obvious win. Measured on 8x H200 it took
        // model load from 30.9 s to 48.3 s: the gather is 1.6M small memcpys per rank,
        // single-threaded, first-touching mmap'd page-cache pages, plus a second full
        // pass over the bytes — against which the driver's own strided walk is simply
        // better. Decode was identical, so it was correct and just slower. Left as this
        // note rather than deleted, so the next person sizing up this line does not
        // spend the same afternoon on it.
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
    out.kv_f16 = kv_f16_on();
    out.max_ckpt = cfg.attn_res_block_size > 0
        ? (cfg.n_layers + cfg.attn_res_block_size - 1) / cfg.attn_res_block_size
        : 0;

    auto alloc = [&](size_t n_floats) -> float* {
        void* p = nullptr;
        if (cudaMalloc(&p, n_floats * sizeof(float)) != cudaSuccess) return nullptr;
        out.owned.push_back(p);
        return (float*)p;
    };

    // The device's position. Four bytes, allocated here so it lives exactly as long as
    // the KV cache it indexes — a d_pos that outlived or predeceased the cache would be
    // a use-after-free that only manifests under capture.
    {
        void* p = nullptr;
        if (cudaMalloc(&p, sizeof(int)) != cudaSuccess) return false;
        out.owned.push_back(p);
        out.d_pos = (int*)p;
        // The expert-weight threshold's depth gate reads this same mirror; bind it
        // here, with the rank's device current and no capture in flight.
        k3k::k3_weps_bind_pos(out.d_pos);
    }

    // Pre-warm the MLA split scratch while allocation is still legal. Inside a stream
    // capture this same call would fail the CAPTURE rather than the allocation, from a
    // call site nowhere near the graph code.
    // Warmed at the model-wide MAXIMUM head count on purpose: the scratch is sized
    // n_head * max_splits * kv_lora, so warming large means every in-capture call hits
    // `need <= cap` and returns without allocating. Warming at the exact per-layer width
    // would leave a wider layer to allocate mid-capture.
    if (n_mla > 0) k3k::k3_mla_prewarm_split_scratch(cfg.n_q_heads, cfg.kv_lora_rank);

    // The IQ lattice tables, for the same reason and with worse symptoms: they upload
    // lazily with a synchronous cudaMemcpyToSymbol on first use, and under IQ1_S that
    // first use is the first MoE layer — so a capture sails through the leading dense
    // layer and dies at layer 1, blaming the MoE dispatch. Unconditional: the cost is one
    // small upload per device at init, and gating it on the quant type would just mean
    // rediscovering this the next time a weight type changed.
    k3k::k3_prewarm_quant_tables();

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
            // f16 rows are half the bytes; alloc() counts floats, so round the
            // half-element count up. key_length is even at every real config,
            // making this exact, but the +1 keeps an odd config safe.
            const size_t elems = (size_t)cfg.key_length * max_ctx;
            out.mla_kv_cache[k] = alloc(out.kv_f16 ? (elems + 1) / 2 : elems);
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

// THE ONLY SANCTIONED WAY TO MOVE THE POSITION.
//
// position exists twice — a host mirror that picks the launch plan and a device copy the
// kernels index with — and any caller that writes one without the other desyncs them.
// kimi_k3_tp_bench's --seek did exactly that: it set the host to ~131040 while d_pos stayed
// 0, so the host planned splits=32 for a full context while the kernels attended over ONE
// position and wrote KV row 0. Both arms then measured a token that was barely doing any
// attention, and the resulting "+19%" was measuring almost nothing.
//
// Making the two-write an API rather than a convention is the fix; leaving it to callers is
// what produced a fast, wrong benchmark that looked like a result.
bool kimi_k3_set_position(KimiK3RuntimeState& s, int pos) {
    if (pos < 0) return false;
    s.position = pos;
    if (s.d_pos && cudaMemcpy(s.d_pos, &pos, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
        return false;
    return true;
}

void kimi_k3_reset_state(KimiK3RuntimeState& s) {
    s.position = 0;
    // The device mirror resets with the host one. A reset that moved only the host would
    // leave the kernels attending over a stale length while the plan said otherwise.
    if (s.d_pos) cudaMemset(s.d_pos, 0, sizeof(int));
    s.n_ckpt = 0;
    auto z = [](float* p, size_t n_elems) {
        if (p && n_elems > 0) cudaMemset(p, 0, n_elems * sizeof(float));
    };
    for (float* p : s.conv_state_q) z(p, (size_t)s.conv_state_elems);
    for (float* p : s.conv_state_k) z(p, (size_t)s.conv_state_elems);
    for (float* p : s.conv_state_v) z(p, (size_t)s.conv_state_elems);
    for (float* p : s.delta_state)  z(p, (size_t)s.delta_state_elems);
    // kv_cache_elems counts CACHE ELEMENTS, which are 2 bytes under kv_f16 —
    // zeroed by bytes here so the f16 arm doesn't memset past its allocation.
    // (All-zero bits is 0.0 in both f32 and f16, so the cleared state means the
    // same thing on either arm.)
    const size_t kv_bytes = (size_t)s.kv_cache_elems * (s.kv_f16 ? 2 : 4);
    for (float* p : s.mla_kv_cache)
        if (p && kv_bytes > 0) cudaMemset(p, 0, kv_bytes);
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
    // routed_norm's out-of-place target, [expert_latent]. The norm used to run
    // in place on moe_out, and in-place is the one shape rms_norm_f32 cannot
    // grid-spread (a CTA that finished its reduction would overwrite x while
    // another CTA is still reading it). 14 KB buys the spread on 92 launches.
    float* moe_normed = nullptr;

    // KDA. The four marked [n_tok, qkv] are the ones the chunk crosses (see the
    // allocation); the rest are transient within ONE iteration of the scan loop and
    // stay a single row even when the projections take the whole chunk.
    float* qkv_q = nullptr, *qkv_k = nullptr, *qkv_v = nullptr;   // [n_tok, qkv]
    float* conv_q = nullptr, *conv_k = nullptr, *conv_v = nullptr; // [qkv]
    float* f_a_out = nullptr;      // [head_dim]
    float* g_raw = nullptr;        // [qkv]
    float* decay_g = nullptr;      // [qkv]
    float* beta_out = nullptr;     // [n_head]   <- 96, NOT qkv. No row axis.
    float* delta_out = nullptr;    // [qkv]
    float* g_proj_out = nullptr;   // [n_tok, qkv]
    float* gate_out = nullptr;     // [n_tok, qkv]

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
    // Softmax scratch for attn_res_mix_f32, [max_ckpt + 1] PER TOKEN ROW. Persistent
    // because the mix runs twice per layer: allocating it per call cost a
    // cudaMallocAsync / cudaFreeAsync pair ~185 times per token.
    //
    // FFN-SIDE (sized by ffn_cap), because the batched mix writes row r's scores at
    // (res_scores + r*(n_ckpt+1)). It is scratch WITHIN one mix, not across layers —
    // but it is live across the two launches of that mix, and the grid .y makes the
    // rows concurrent. See the allocation for what a single shared row does.
    float* res_scores = nullptr;       // [ceil(n_layers / attn_res_block_size) + 1, cap]
    void* proj_q8 = nullptr;           // optional llama CPU-compat block_q8_0 scratch
    // The HOISTED activation, quantised once and consumed by several projections. A
    // SEPARATE buffer from proj_q8 on purpose: every un-hoisted k3_proj_ggml_f32 call
    // quantises into proj_q8, so sharing it would let any projection landing between a
    // hoist and its consumers silently overwrite the bytes. That aliasing is the same
    // failure the cached quantise-once attempt shipped as top1 0.0.
    void* act_q8 = nullptr;
    void* moe_q8 = nullptr;            // optional llama CPU-compat block_q8_K scratch
    // Counting-sort workspace for the token-batched expert dispatch. Integer only
    // (counts, run offsets, and the (token, slot) index list) — the activations are
    // never copied. Sized for the FULL chunk cap, so a call with fewer tokens is
    // always within it; nullptr or a short size just declines the batched path.
    void* moe_batch_ws = nullptr;
    size_t moe_batch_ws_bytes = 0;

    // ---- DAG LANES -------------------------------------------------------
    //
    // The capture is one linear chain per rank, but a layer is not a chain. On an
    // MLA layer the kv_a branch and the attention-gate projection depend only on
    // `normed`, and neither is read until the decode kernel and the gate fold
    // respectively -- so main's stream order makes ~7 launches wait on work no
    // consumer of theirs is waiting for. Same shape on KDA (ssm_f_a/f_b/decay and
    // ssm_beta hang off `normed`) and on every MoE layer (the shared expert hangs
    // off `normed2` and is not read until the collective).
    //
    // Two side streams are enough for all three sites: the widest fork here is
    // main + 2. They are created ONCE per rank, on the rank's device, because a
    // stream created inside a capture is not capturable and one created per token
    // would be a per-token cudaStreamCreate on the critical path.
    static constexpr int kLanes = 2;
    cudaStream_t lane[kLanes] = {nullptr, nullptr};
    cudaEvent_t  ev_fork = nullptr;              // main -> lanes
    cudaEvent_t  ev_lane[kLanes] = {nullptr, nullptr};   // lane -> main
    // PER-LANE Q8 SCRATCH, AND WHY IT IS NOT AN OPTIMISATION.
    //
    // Every un-hoisted k3_proj_ggml_f32 / k3_proj_q8_multirow_1bar quantises its
    // activation into a scratch buffer and then reads it back. On one stream that
    // is a straight-line write-then-read; the moment two projections run
    // CONCURRENTLY they interleave a write with the other's read and both get a
    // torn activation. It is silent -- fluent output, wrong logits -- and it is
    // exactly the aliasing failure recorded above for act_q8. A lane therefore
    // never shares proj_q8 with the main stream or with the other lane.
    void* proj_q8_lane[kLanes] = {nullptr, nullptr};
    bool  lanes_ok = false;            // every lane object above was created

    // ---- CHUNKED PROMPT INGESTION --------------------------------------
    //
    // A chunked walk runs layer L for B tokens before layer L+1, so any buffer
    // written under one phase and read under a LATER one has to hold B values at
    // once rather than one. That is a much shorter list than it looks, and getting
    // it wrong is silent, so it is enumerated rather than guessed:
    //
    //   attn_out    written by Attn, read by FfnPartial's residual combine, and
    //               reduced in between.                                     -> xB
    //   moe_fused   written by FfnPartial (both views), read by FfnFinish, and
    //               reduced in between.                                     -> xB
    //   ffn_out     the LEADING DENSE layer's partial, same shape of use.   -> xB
    //   res_bank    per-token across ALL layers, with its own live count.   -> xB
    //
    // Everything else here (mixed, normed, normed2, the KDA and MLA scratch, the
    // router, moe_scratch) is written and read inside ONE phase, so the per-token
    // loop reuses it exactly as the token loop already does. s.mixed in particular
    // looks like a counterexample and is not: Attn writes it and Attn's own
    // rms_norm consumes it, so it never crosses a phase boundary.
    //
    // Slot selection is POINTER ARITHMETIC, not a copy: kimi_k3_forward_select_slot
    // repoints the four fields above into slot t and the entire existing layer body
    // then runs unmodified. That is the whole reason this is a small change — the
    // alternative, threading a token index through every kernel call in
    // forward_layer_phase, touches ~200 call sites and cannot be checked by
    // inspection.
    bool   batched = false;            // kimi_k3_forward_alloc_batch succeeded
    int    b_cap = 1;                  // slots allocated
    int    b_slot = 0;                 // slot the pointers above currently address
    float* attn_out_b = nullptr;       // [b_cap][hidden]
    float* ffn_out_b = nullptr;        // [b_cap][hidden]
    float* moe_fused_b = nullptr;      // [b_cap][expert_latent + hidden]
    float* res_bank_b = nullptr;       // [b_cap][max_ckpt][hidden]
    std::vector<int> b_ckpt;           // [b_cap] live checkpoint count per slot
    // What state->res_bank pointed at before the first select_slot, so the decode
    // path gets its own bank back when the chunk ends.
    float* res_bank_orig = nullptr;

    // How many TOKEN-MAJOR rows the batched buffers above were allocated for. That set
    // is now the FFN-side ones PLUS the MLA-side ones (mixed, normed, q_lora_out,
    // q_proj_out, kv_a_out, kv_cmpr_normed, absorbed_q, mla_attn_out, gate_proj_out),
    // because the Attn phase batches an MLA layer.
    //
    // The KDA-side buffers (qkv_*, conv_*, f_a_out, g_raw, decay_g, beta_out,
    // delta_out, gate_out) stay at ONE row, and that is the allocation enforcing the
    // refusal rather than merely agreeing with it: a KDA layer is a recurrence in the
    // conv state and the delta state, so a batched launch has nowhere correct to put
    // its rows and the phase refuses n_tok > 1 for it.
    //
    // s.attn_out and s.moe_fused ARE sized for the cap even though the chunk driver
    // repoints both through kimi_k3_swap_partial_buffer before it ever batches. They
    // are READ (attn_out) and WRITTEN (moe_fused) by the batched FFN phases at
    // per-token strides, so sizing them for one token would make "the caller forgot
    // to swap" an out-of-bounds write instead of a wrong-but-contained result. 1 MB
    // per rank to make a whole class of caller mistake impossible.
    int ffn_cap = 1;

    std::vector<void*> owned;
};

// How many token rows the FFN-side scratch is sized for.
//
// Read from the SAME env the chunk driver takes its chunk size from, so the two
// cannot be configured apart: a scratch sized 16 with a driver asking for 64 would
// be an overflow, and the only reason it is not is that both read this. The phase
// still refuses n_tok > ffn_cap, and kimi_k3_tp_prefill_chunk still clamps to it —
// belt and braces, because the failure mode is a silent stomp on the next buffer.
static int k3_ffn_batch_cap_env() {
    static const int cap = [] {
        const char* e = std::getenv("SPARKINFER_K3_PREFILL_CHUNK");
        // 64 IS MEASURED, NOT PICKED. Swept at the scored 32k shape on 8xH200, with every
        // arm byte-identical to every other so the sweep is pure scheduling. The curve
        // turns over here: the expert regroup's win scales as B/56 while the scratch grows
        // linearly, so past this point the larger allocation stops paying for itself.
        int v = e ? std::atoi(e) : 64;
        // 0 means "no chunked prefill" at the driver, which needs no batch capacity;
        // it must still be at least 1 so every buffer keeps its single-token size.
        if (v < 1) v = 1;
        // RAISED 256 -> 512 FOR THE EXPERT-MAJOR REGROUP, and 512 is where the
        // arithmetic stops paying, not where the machine stops coping.
        //
        // The regroup's win is proportional to rows-per-expert = B*top_k/n_experts =
        // B/56, so B = 256 gives 4.6 and B = 512 gives 9.1 — the last doubling that
        // still nearly doubles the amortisation. Past that the curve flattens
        // (published M-scaling is ~4.75 us/token by M = 128) while the scratch keeps
        // growing linearly.
        //
        // WHAT THE CEILING COSTS, since every FFN-side buffer is sized by it: ~700 KiB
        // per token row per rank (dense_gate/up/situ at dense_ffn are 396 KiB of that,
        // moe_scratch 48 KiB at the 2-D band, moe_fused 42 KiB, the four hidden-width
        // buffers 112 KiB, the two q8 stages 70 KiB). 512 is therefore ~350 MiB per
        // rank against ~175 MiB at 256. Nothing on a 141 GB H200 holding a ~69 GB
        // shard, and NOT paid by anyone who leaves the default of 16 alone — this
        // bound only limits what an operator may ask for.
        //
        // gridDim.y remains the hard structural limit at 65535 and is nowhere near.
        if (v > 512) v = 512;
        return v;
    }();
    return cap;
}

// This rank's per-expert FFN width: the model's under whole-expert sharding, its
// band under 2-D MoE.
//
// Read from the WEIGHTS' own ShardDims, never re-derived from cfg and tp_size. The
// weights were cut with this struct, the kernels are launched from it, and the one
// failure mode that does not announce itself is the two disagreeing — a scratch
// sized 3072 with weights packed 768 wide runs at full speed and reads three
// experts' worth of neighbouring memory as if it were this expert's.
int k3_moe_ffn_local(const KimiK3Forward& fwd, const KimiK3Config& cfg) {
    if (fwd.w && fwd.w->shard.moe_2d && fwd.w->shard.moe_ffn > 0)
        return fwd.w->shard.moe_ffn;
    return cfg.moe_ffn;
}

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

    // THE FFN-SIDE ROW AXIS. Every buffer multiplied by `cap` below is one the
    // batched FfnPartial/FfnFinish addresses as [n_tok, <one token's width>]; every
    // buffer left at one token is either attention-side (and the Attn phase refuses
    // n_tok > 1) or is transient WITHIN one token's pass through the batched loops
    // that stayed per-token (s.moe_scratch's q8k companion s.moe_q8).
    //
    // The multiply is applied at the ALLOCATION only. The per-token stride the phase
    // uses is the buffer's natural one-token width, never `cap` — so growing the cap
    // cannot move a row, and a cap of 1 is byte-for-byte the allocation this function
    // made before the axis existed.
    const int cap = k3_ffn_batch_cap_env();
    s.ffn_cap = cap;

    bool ok = true;
    // mixed / normed ARE ROW-AXIS BUFFERS NOW, because the Attn phase batches them on an
    // MLA layer (attn_res_mix and attn_norm both carry n_rows). A KDA layer still runs
    // one token at a time and simply uses row 0, so this is capacity the KDA path never
    // reads rather than a change to it.
    ok &= alloc_f(s.mixed, (size_t)H * cap);
    ok &= alloc_f(s.mixed2, (size_t)H * cap);
    ok &= alloc_f(s.normed, (size_t)H * cap);
    ok &= alloc_f(s.normed2, (size_t)H * cap);
    // attn_out is attention-side by name and FFN-side by use: the batched FfnPartial
    // READS it at (b * H) for the residual combine. See Scratch::ffn_cap.
    ok &= alloc_f(s.attn_out, (size_t)H * cap);
    ok &= alloc_f(s.ffn_out, (size_t)H * cap);
    ok &= alloc_f(s.moe_normed, (size_t)cfg.expert_latent * cap);

    // ---- KDA-SIDE ROW AXIS: THE FIVE BUFFERS THE CHUNK CROSSES, AND NO OTHERS ----
    //
    // Under SPARKINFER_K3_KDA_QKVG_BATCH the KDA branch computes q/k/v/g for the WHOLE
    // chunk before the scan walks it, and collects the scan's gate output for the whole
    // chunk before attn_output projects it. Those five buffers are therefore
    // [n_tok, qkv] and every one of them is addressed at (b * qkv).
    //
    // EVERY OTHER KDA BUFFER STAYS ONE ROW, DELIBERATELY. conv_*, f_a_out, g_raw,
    // decay_g, beta_out and delta_out are transient WITHIN one iteration of the scan
    // loop: token b writes them and the same iteration's decode step consumes them,
    // on one stream, before token b+1 overwrites them. Widening them would be dead
    // memory and would invite a stride where there is no row.
    //
    // beta_out IS THE ONE THAT WOULD HAVE BITTEN. It is n_q_heads (96) wide, not qkv
    // (1536), so a single shared row stride across "the KDA buffers" is silently wrong
    // from token 1 onward. It is not in this set at all, so there is no stride to get
    // wrong; the loop rewrites its one row per token.
    //
    // The multiply is at the ALLOCATION only — the phase strides by the buffer's own
    // one-token width — so a cap of 1 is byte-for-byte the allocation this made before.
    ok &= alloc_f(s.qkv_q, (size_t)qkv * cap);
    ok &= alloc_f(s.qkv_k, (size_t)qkv * cap);
    ok &= alloc_f(s.qkv_v, (size_t)qkv * cap);
    ok &= alloc_f(s.conv_q, qkv);
    ok &= alloc_f(s.conv_k, qkv);
    ok &= alloc_f(s.conv_v, qkv);
    // ---- KDA PRE-SCAN ROW AXIS ------------------------------------------------
    // These three are the KDA chain that touches NEITHER recurrence. f_a_out comes from
    // `normed` alone, g_raw from f_a_out, decay_g from g_raw — none of them reads the
    // conv state or the delta state, so token b's value does not depend on token b-1 and
    // the whole chain can be computed for a chunk in one pass. Giving them a row axis is
    // what lets the pre-scan batch below exist; conv_*, beta_out and delta_out stay at one
    // row because they ARE in the recurrence (or, for beta, are shard-offset and 96 wide).
    //
    // The multiply is at the ALLOCATION only. At cap == 1 this is byte-for-byte the
    // allocation it made before, and the per-token path strides by the buffer's own
    // one-token width exactly as it always did.
    ok &= alloc_f(s.f_a_out, (size_t)cfg.kda_head_dim * cap);
    ok &= alloc_f(s.g_raw, (size_t)qkv * cap);
    ok &= alloc_f(s.decay_g, (size_t)qkv * cap);
    ok &= alloc_f(s.beta_out, cfg.n_q_heads);
    ok &= alloc_f(s.delta_out, qkv);
    ok &= alloc_f(s.g_proj_out, (size_t)qkv * cap);
    ok &= alloc_f(s.gate_out, (size_t)qkv * cap);

    // ---- MLA-SIDE ROW AXIS ----------------------------------------------------
    //
    // Every buffer the batched Attn phase addresses as [n_tok, <one token's width>] is
    // multiplied by `cap` here. `qh` is the MODEL's head count and the phase strides by
    // THIS RANK's (fwd.w->mla.n_heads, which is qh/8 under head sharding), so the row
    // pitch is never wider than the pitch this allocation assumed — the batched rows sit
    // inside the allocation rather than at the edge of it.
    //
    // q_nope / q_pe are deliberately NOT in the set. They are written only when
    // k3_mla_absorb_q_strided DECLINES, and that fallback stays a per-token loop of the
    // two cudaMemcpy2DAsync de-interleaves plus mla_absorb_q_f32 — one token live at a
    // time, straight-line on one stream, so one row is all it can ever need.
    ok &= alloc_f(s.q_lora_out, (size_t)cfg.q_lora_rank * cap);
    ok &= alloc_f(s.q_proj_out, (size_t)qh * cfg.key_length_mla * cap);
    ok &= alloc_f(s.q_nope, (size_t)qh * qk_nope);
    ok &= alloc_f(s.q_pe, (size_t)qh * cfg.rope_dim);
    ok &= alloc_f(s.kv_a_out, (size_t)cfg.key_length * cap);
    ok &= alloc_f(s.kv_cmpr_normed, (size_t)cfg.kv_lora_rank * cap);
    ok &= alloc_f(s.absorbed_q, (size_t)qh * cfg.key_length * cap);
    ok &= alloc_f(s.mla_attn_out, (size_t)qh * cfg.value_length_mla * cap);
    ok &= alloc_f(s.gate_proj_out, (size_t)qh * cfg.value_length_mla * cap);

    // ROUTER, BATCHED. Both router kernels already index blockIdx.x as the token and
    // stride logits by n_expert and out_w/out_ids by top_k, so these three only had to
    // BE that wide — no kernel change, and the launch at n_tokens == 1 is unchanged.
    ok &= alloc_f(s.router_logits, (size_t)cfg.n_experts * cap);
    ok &= alloc_f(s.router_w, (size_t)cfg.top_k * cap);
    ok &= alloc_i(s.router_ids, (size_t)cfg.top_k * cap);
    ok &= alloc_f(s.routed_down_out, (size_t)cfg.expert_latent * cap);
    // THE SCRATCH IS THE RANK'S FFN WIDTH, NOT THE MODEL'S. Under 2-D MoE this rank
    // evaluates only its band of each expert's intermediate, so top_k * band is all
    // that is ever written — and sizing it from cfg would leave 3/4 of the buffer
    // untouched while the gate/up memset below still paid to clear it.
    const int moe_ffn_local = k3_moe_ffn_local(fwd, cfg);
    // ONE SLAB PER TOKEN ROW. The expert dispatch itself stays per-token (router_ids
    // differ per token, so batching it without an expert-major regroup buys launches
    // and no bandwidth), and a single slab would in fact be safe today because that
    // loop is straight-line on one stream. It is per-row anyway because the "foreign
    // expert slots read as zero" invariant below is a statement about A SLAB, and the
    // expert-major regroup that is the next win writes several tokens' slabs at once.
    // Making the invariant true per row now costs 3 MB and removes the trap later.
    ok &= alloc_f(s.moe_scratch, (size_t)cfg.top_k * moe_ffn_local * cap);
    // Establish the "foreign expert slots read as zero" invariant ONCE here, instead of
    // re-establishing it with a cudaMemsetAsync on every MoE layer. A rank's expert band
    // is fixed for the run and moe_gate_up writes only owned slots, so a foreign slot is
    // written by nobody and stays zero from this memset onward.
    //
    // THE LENGTH MUST COVER EVERY SLAB. A memset of one slab with cap slabs allocated
    // leaves rows 1..cap-1 holding whatever cudaMalloc returned, and the dispatch reads
    // a foreign slot expecting zero — fluent output, wrong logits, only on the prefill
    // path, only for tokens after the first of a chunk.
    if (ok && s.moe_scratch)
        cudaMemset(s.moe_scratch, 0,
                   (size_t)cfg.top_k * moe_ffn_local * cap * sizeof(float));
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
    //
    // BATCHED, and the per-token stride is (expert_latent + hidden) — the FUSED width,
    // not either view's own. That is the layout kimi_k3_tp_prefill_chunk allocates and
    // the layout one collective over nb*(latent+hidden) reduces, so the phase must
    // stride BOTH views by the fused width. moe_out row b is (moe_out + b*fused) and
    // shexp_out row b is (shexp_out + b*fused) — NOT (shexp_out + b*hidden).
    const size_t moe_fused_row = (size_t)cfg.expert_latent + (size_t)H;
    ok &= alloc_f(s.moe_fused, moe_fused_row * cap);
    if (ok) {
        s.moe_out   = s.moe_fused;
        s.shexp_out = s.moe_fused + cfg.expert_latent;
    }
    ok &= alloc_f(s.dense_gate, (size_t)cfg.dense_ffn * cap);
    ok &= alloc_f(s.dense_up, (size_t)cfg.dense_ffn * cap);
    ok &= alloc_f(s.dense_situ, (size_t)cfg.dense_ffn * cap);
    // Same bound kimi_k3_alloc_state uses for max_ckpt, +1 for the current stream.
    // Sized from cfg rather than from the state so the scratch stays allocatable
    // without one; a mix that somehow saw more checkpoints than the bank can hold
    // would already have failed the `n_ckpt >= max_ckpt` guard in the forward.
    //
    // TIMES `cap`, BECAUSE THE MIX IS BATCHED AND EACH ROW OWNS ITS OWN SOFTMAX.
    // attn_res_mix_f32's score kernel writes row r at (scores + r*(n_ckpt+1)) and its
    // apply kernel reads the same slots back. One row shared across the grid .y would
    // blend EVERY token of a chunk with the LAST token's softmax — no crash, no NaN,
    // and no top-1 move, because the weights are a softmax over 2-9 near-equal scores
    // and stay near-uniform. At short context n_ckpt is 0 or 1, so the parity probe
    // cannot see it either; it only shows up at depth. Same `cap` as every other
    // FFN-side row axis above, so the two cannot be configured apart. At cap 1 this is
    // byte-for-byte the allocation this function made before the axis existed.
    ok &= alloc_f(s.res_scores,
                  (cfg.attn_res_block_size > 0
                       ? (size_t)((cfg.n_layers + cfg.attn_res_block_size - 1) /
                                  cfg.attn_res_block_size) + 1
                       : 1) * (size_t)cap);
    // BOTH HOLD n_tok ROWS OF Q8_0 BLOCKS NOW. k3_proj_q8act_tok_f32 reads a
    // token-major [n_tok, K/32] block array with a leading dimension it is told, so
    // the staging has to be that wide; dense_ffn is still the widest K any projection
    // contracts over, so one row size covers every activation either buffer stages.
    // At cap 1 this is byte-for-byte the allocation above.
    ok &= alloc_bytes(s.proj_q8, k3k::k3_q8_0_bytes(cfg.dense_ffn) * (size_t)cap);
    // Sized like proj_q8: dense_ffn is the widest K any projection contracts over, so
    // one allocation covers every activation this hoists.
    ok &= alloc_bytes(s.act_q8, k3k::k3_q8_0_bytes(cfg.dense_ffn) * (size_t)cap);
    ok &= alloc_bytes(s.moe_q8,
                      k3k::k3_moe_q8_k_bytes(cfg.expert_latent, moe_ffn_local, cfg.top_k));

    // THE EXPERT-MAJOR REGROUP'S SORT SPACE. Sized from the SAME `cap` every other
    // batched buffer uses and from the rank's OWN expert count — the band under
    // tp > 1, the model's under tp = 1 — because that is the number of buckets the
    // histogram and the run-offset array are indexed by, and a bucket array sized
    // from cfg on a banded rank would have the scan writing past its end.
    //
    // Failing to allocate is NOT fatal: the dispatch checks the pointer, declines,
    // and the per-token loop runs. At cap 1 this is 3.6 KB that is never touched.
    {
        const int n_slots = fwd.w && fwd.w->shard.tp_size > 1
                                ? fwd.w->shard.expert_band.extent
                                : cfg.n_experts;
        const size_t nb = k3k::k3_moe_batch_ws_bytes(cap, cfg.top_k, n_slots);
        if (nb > 0 && alloc_bytes(s.moe_batch_ws, nb)) s.moe_batch_ws_bytes = nb;
    }

    if (!ok) { kimi_k3_forward_free_scratch(fwd); return false; }

    // ---- DAG lanes ----
    //
    // NON-BLOCKING on purpose. A stream from cudaStreamCreate carries an implicit
    // dependency on the legacy default stream; under SPARKINFER_K3_GRAPH=0 -- the
    // A/B control and the only mode the profiler runs in -- that would serialise
    // the lanes against anything the process does on the default stream and make
    // the fork measure as a loss for a reason that has nothing to do with the fork.
    //
    // Failing to create a lane is NOT fatal: lanes_ok stays false, the forward
    // issues main's linear order, and the run is a correct slow run rather than a
    // dead one. Same policy as every other decline in this file.
    bool lok = true;
    for (int i = 0; i < KimiK3Forward::Scratch::kLanes; ++i) {
        lok &= cudaStreamCreateWithFlags(&s.lane[i], cudaStreamNonBlocking) == cudaSuccess;
        lok &= cudaEventCreateWithFlags(&s.ev_lane[i], cudaEventDisableTiming) == cudaSuccess;
        // Sized like s.proj_q8 -- dense_ffn is the widest K any projection contracts
        // over, so one allocation per lane covers every activation a lane quantises.
        lok &= alloc_bytes(s.proj_q8_lane[i], k3k::k3_q8_0_bytes(cfg.dense_ffn));
    }
    lok &= cudaEventCreateWithFlags(&s.ev_fork, cudaEventDisableTiming) == cudaSuccess;
    s.lanes_ok = lok;
    return true;
}

int kimi_k3_ffn_batch_cap(const KimiK3Forward& fwd) {
    return fwd.s ? fwd.s->ffn_cap : 1;
}

// SPARKINFER_K3_ATTN_BATCH — the one gate that turns the batched Attn phase off.
//
// It must be read in EXACTLY ONE place, because the driver and the phase have to agree:
// the driver picks between one batched call and a per-token loop, and the phase refuses
// what it will not do. Two independent getenv lambdas would let a typo make the driver
// issue a batched call the phase then refuses, which is a failed prefill rather than a
// slow one. Default ON — the harness scores a default build — and =0 restores the
// per-token attention on the SAME binary, which is what makes the parity A/B exact.
static bool k3_attn_batch_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("SPARKINFER_K3_ATTN_BATCH");
        return !(e && e[0] == '0');
    }();
    return on;
}

// SPARKINFER_K3_KDA_QKVG_BATCH — DEFAULT OFF. Lets the Attn phase take a whole chunk
// on a KDA layer, for the PROJECTIONS only.
//
// WHAT THIS DOES AND DOES NOT CLAIM. The refusal above this was written about the
// SCAN, and that part of it is still true: k3_kda_decode_step_ip carries the delta
// state from token b-1 into token b and kda_conv_step_f32 shifts a 4-deep window per
// token, so neither has a token axis and neither gets one here. The scan stays a
// per-token loop in index order on one stream.
//
// But q/k/v/g are ELEMENTWISE in the token index — token b's projections are a function
// of token b's own hidden state and nothing else — so they can be computed for the whole
// chunk before the scan walks it. That is the entire change: WHEN the projections are
// issued, not what they compute. Same weights, same activation bytes, same accumulation
// order per output element (see the note at the batched call site).
//
// Read in exactly one place for the same reason SPARKINFER_K3_ATTN_BATCH is: the driver
// picks between one batched call and a per-token loop by asking the same predicate the
// phase enforces, so the two cannot disagree. Default OFF until measured — the scored
// build must not move under an unmeasured scheduling change.
// MEASURED 2026-08-06 ON 8xH200 AT THE SCORED SHAPE, which is why this now defaults ON.
//
// It was gated off because it had never been run at 32k, and the rule this file keeps is
// that the shipped build does not move under an unmeasured scheduling change. It has now
// been measured, on one binary against the same binary with the gate off, in one session:
//
// on one binary against the same binary with the gate off, in one session, and the logits
// are BYTE-IDENTICAL between the two — so this buys throughput without touching the
// arithmetic. That is the whole reason it can be defaulted on rather than left as a knob.
//
// SPARKINFER_K3_KDA_QKVG_BATCH=0 restores the per-token group on the same binary, so the
// A/B that produced those numbers stays reproducible.
static bool k3_kda_qkvg_batch_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("SPARKINFER_K3_KDA_QKVG_BATCH");
        // OPT-IN, as the block comment below and kimi_k3_attn_batch_ok() both state.
        //
        // This read used to be `!(e && e[0] == '0')`, which is the OPT-OUT form: unset
        // meant ON, so all 69 KDA layers took the batched projection path by default
        // while two comments said they did not. It faults --
        //   [k3] LAUNCH FAILED at layer 0, phase Attn: invalid argument
        // -- for every chunk >= 2 tokens at every prompt length, so with it on by
        // default the chunk driver could not ingest anything at all: the 4-token parity
        // probe died here, and so did a 128-token prompt at chunk widths 2, 4, 8 and 64.
        // The only reason a 32k measurement ever completed is that --checkpoints routes
        // the graded probes through the per-token loop instead.
        //
        // Turning it off restores the driver AND its speedup, because the projections it
        // batches are not where the win comes from. Left opt-in until the fault in the
        // batched KDA projection path is understood; the gate is the safe half of that
        // question, not the answer to it.
        return e && e[0] == '1';
    }();
    return on;
}

// THE KDA PRE-SCAN BATCH: the f_a -> f_b -> decay chain, computed for a whole chunk
// before the per-token scan loop instead of once per token inside it.
//
// WHY THERE IS ANYTHING LEFT TO BATCH ON A RECURRENT LAYER. A KDA layer has two
// recurrences — the conv state and the delta state — and both are genuinely sequential
// in the token index. But the decay chain is neither: f_a_out reads `normed`, g_raw reads
// f_a_out, decay_g reads g_raw, and not one of them touches either state. Token b's decay
// gate does not depend on token b-1 in any way, so the only reason it was computed inside
// the loop is that it sits between two things that are.
//
// WHAT IT IS WORTH, AND IT IS OCCUPANCY RATHER THAN LAUNCH COUNT. kda_decay_gate launches
// at dim3(n_head, n_rows) and n_head is 12 — twelve blocks on a 132-SM part, ~9% of the
// machine, once per token on each of the 69 KDA layers. Batching a chunk of 64 makes that
// dim3(12, 64) = 768 blocks and fills it. The three launches per token becoming three per
// chunk is the smaller half of the win.
//
// BIT-IDENTICAL BY CONSTRUCTION, and the reason is the same one rms_norm_wide leans on:
// every kernel involved already carries an (n_rows, row_stride) row axis whose row r
// reduces over row r's own elements with the same partition and the same tree, so nothing
// about a row's arithmetic depends on how many other rows are in flight. proj_hb is the
// token-batched projection the FFN side already uses and falls back to a per-token loop
// whenever its batched kernel declines. At n_rows == 1 every one of these is the call the
// per-token path makes today.
//
// Default OFF. The scored build must not move under an unmeasured scheduling change —
// this is measured on one binary against the same binary with the gate off.
static bool k3_kda_pre_batch_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("SPARKINFER_K3_KDA_PRE_BATCH");
        return e && e[0] == '1';
    }();
    return on;
}

bool kimi_k3_attn_batch_ok(const KimiK3Forward& fwd, int layer, int n_tok) {
    if (n_tok <= 1) return false;
    if (!fwd.s || !fwd.cfg || !fwd.w) return false;
    if (n_tok > fwd.s->ffn_cap) return false;
    if (!k3_attn_batch_enabled()) return false;
    if (layer < 0 || layer >= fwd.cfg->n_layers) return false;
    // KDA'S SCAN IS A RECURRENCE AND STAYS PER-TOKEN, WHICH IS THE HONEST HALF.
    //
    // k3_kda_decode_step_ip carries the delta state from token b-1 into token b and
    // kda_conv_step_f32 shifts a 4-deep conv window per token; neither has a token axis
    // and neither can have one without becoming a different (chunked-scan) algorithm
    // with a different summation order. The KDA branch of the phase therefore keeps a
    // per-token loop for everything from the convs to the gate.
    //
    // What it no longer refuses outright is the PROJECTION half. q/k/v/g and attn_output
    // are elementwise in the token index, so under SPARKINFER_K3_KDA_QKVG_BATCH they run
    // once for the chunk and the loop consumes their rows in order. Default OFF, so this
    // still returns false for all 69 KDA layers unless the gate is set.
    if (fwd.cfg->is_kda_layer(layer) && !k3_kda_qkvg_batch_enabled()) return false;
    // The debug taps read a scratch buffer straight after the launch that wrote it and
    // report ONE vector. On a batched phase they would report row 0 and silently say
    // nothing about rows 1..n-1, which is the opposite of what the localisation mode is
    // for. The validation path runs single-token anyway; this makes that explicit.
    if (fwd.debug) return false;
    return true;
}

void kimi_k3_forward_free_scratch(KimiK3Forward& fwd) {
    if (!fwd.s) return;
    for (int i = 0; i < KimiK3Forward::Scratch::kLanes; ++i) {
        if (fwd.s->lane[i])    cudaStreamDestroy(fwd.s->lane[i]);
        if (fwd.s->ev_lane[i]) cudaEventDestroy(fwd.s->ev_lane[i]);
    }
    if (fwd.s->ev_fork) cudaEventDestroy(fwd.s->ev_fork);
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

// ---------------------------------------------------------------------------
// Chunked prompt ingestion: allocate B slots, then address them one at a time.
// ---------------------------------------------------------------------------

bool kimi_k3_forward_alloc_batch(const KimiK3Config& cfg, KimiK3Forward& fwd,
                                 int batch) {
    if (!fwd.s || batch < 1) return false;
    auto& s = *fwd.s;
    if (s.batched && s.b_cap >= batch) return true;   // already big enough
    if (s.batched) return false;   // growing in place would strand the old owned ptrs

    // B == 1 IS NOT A SPECIAL CASE, AND THAT IS DELIBERATE. The chunked walk at B=1
    // must be the token loop bit-for-bit, so it takes the same code path with a
    // one-slot allocation rather than a bypass — otherwise "B=1 matches main" proves
    // nothing about B=8, because B=1 would not have run the batched code at all.
    const int H = cfg.hidden;
    const int Lat = cfg.expert_latent;
    const int max_ckpt = fwd.state ? fwd.state->max_ckpt : 0;

    auto alloc_f = [&](float*& ptr, size_t n) {
        void* p = nullptr;
        if (n == 0) { ptr = nullptr; return true; }
        if (cudaMalloc(&p, n * sizeof(float)) != cudaSuccess) return false;
        s.owned.push_back(p);
        ptr = (float*)p;
        return true;
    };

    bool ok = true;
    ok &= alloc_f(s.attn_out_b,  (size_t)batch * H);
    ok &= alloc_f(s.ffn_out_b,   (size_t)batch * H);
    ok &= alloc_f(s.moe_fused_b, (size_t)batch * (Lat + H));
    ok &= alloc_f(s.res_bank_b,  (size_t)batch * max_ckpt * H);
    if (!ok) return false;

    // The foreign-slot-reads-as-zero invariant the single-token allocator establishes
    // for moe_scratch has a twin here: a chunk shorter than b_cap leaves the tail
    // slots unwritten, and the collective reduces the WHOLE array. Stale bytes in an
    // unused slot would be summed across ranks into a value nothing later reads —
    // harmless today, but it makes every debug comparison of the reduced payload
    // depend on allocation history. Zero once and the tail is defined forever.
    if (s.moe_fused_b)
        cudaMemset(s.moe_fused_b, 0, (size_t)batch * (Lat + H) * sizeof(float));
    if (s.attn_out_b)
        cudaMemset(s.attn_out_b, 0, (size_t)batch * H * sizeof(float));

    s.b_ckpt.assign((size_t)batch, 0);
    s.b_cap = batch;
    s.b_slot = -1;              // no slot addressed yet; the first select must apply
    s.batched = true;
    return true;
}

int kimi_k3_forward_batch_capacity(const KimiK3Forward& fwd) {
    return (fwd.s && fwd.s->batched) ? fwd.s->b_cap : 0;
}

void kimi_k3_forward_select_slot(KimiK3Forward& fwd, int slot) {
    if (!fwd.s || !fwd.s->batched) return;
    auto& s = *fwd.s;
    if (slot < 0 || slot >= s.b_cap) return;
    const KimiK3Config& cfg = *fwd.cfg;
    const int H = cfg.hidden;
    const int Lat = cfg.expert_latent;

    // Park the OUTGOING slot's live checkpoint count before moving. n_ckpt is
    // per-token state that the layer body increments in place, so a slot switch that
    // did not save it would carry token t's count into token t+1 and mix the wrong
    // prefix into the residual — fluent output, wrong logits, and invisible to any
    // shape or launch check.
    if (fwd.state && s.b_slot >= 0 && s.b_slot < (int)s.b_ckpt.size())
        s.b_ckpt[(size_t)s.b_slot] = fwd.state->n_ckpt;

    s.b_slot = slot;
    s.attn_out  = s.attn_out_b  + (size_t)slot * H;
    s.ffn_out   = s.ffn_out_b   + (size_t)slot * H;
    s.moe_fused = s.moe_fused_b + (size_t)slot * (Lat + H);
    s.moe_out   = s.moe_fused;
    s.shexp_out = s.moe_fused + Lat;

    if (fwd.state && fwd.state->max_ckpt > 0 && s.res_bank_b) {
        if (!s.res_bank_orig) s.res_bank_orig = fwd.state->res_bank;
        fwd.state->res_bank =
            s.res_bank_b + (size_t)slot * fwd.state->max_ckpt * H;
        fwd.state->n_ckpt = s.b_ckpt[(size_t)slot];
    }
}

void kimi_k3_forward_batch_begin(KimiK3Forward& fwd) {
    if (!fwd.s || !fwd.s->batched) return;
    std::fill(fwd.s->b_ckpt.begin(), fwd.s->b_ckpt.end(), 0);
    fwd.s->b_slot = -1;
}

void kimi_k3_forward_batch_end(KimiK3Forward& fwd) {
    if (!fwd.s || !fwd.s->batched) return;
    auto& s = *fwd.s;
    // Hand the decode path its own bank back. Leaving state->res_bank aimed inside
    // the chunk arena would make the next forward_token push checkpoints into slot
    // b_slot's rows — which is exactly the kind of cross-path aliasing that only
    // shows up as a wrong answer several thousand tokens later.
    if (s.res_bank_orig && fwd.state) {
        fwd.state->res_bank = s.res_bank_orig;
        fwd.state->n_ckpt = 0;
    }
    s.b_slot = -1;
}

// The BASE of a phase's partial across all `n_slots` slots, and the element count
// the collective must reduce. Separate from kimi_k3_partial_buffer rather than a
// flag on it: the single-token entry point is on the decode path and must keep
// returning exactly what it always did.
float* kimi_k3_batch_partial_buffer(KimiK3Forward& fwd, int layer,
                                    K3LayerPhase phase, int n_slots, int* count) {
    if (!fwd.s || !fwd.s->batched || n_slots < 1 || n_slots > fwd.s->b_cap) {
        if (count) *count = 0;
        return nullptr;
    }
    const KimiK3Config& cfg = *fwd.cfg;
    auto& s = *fwd.s;
    if (phase == K3LayerPhase::Attn) {
        if (count) *count = n_slots * cfg.hidden;
        return s.attn_out_b;
    }
    if (phase == K3LayerPhase::FfnPartial) {
        if (layer < cfg.leading_dense) {
            if (count) *count = n_slots * cfg.hidden;
            return s.ffn_out_b;
        }
        if (count) *count = n_slots * (cfg.expert_latent + cfg.hidden);
        return s.moe_fused_b;
    }
    if (count) *count = 0;
    return nullptr;
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
    const char* pn = phase == K3LayerPhase::All          ? "All"
                   : phase == K3LayerPhase::Attn         ? "Attn"
                   : phase == K3LayerPhase::FfnPrepare   ? "FfnPrepare"
                   : phase == K3LayerPhase::FfnPartial   ? "FfnPartial" : "FfnFinish";
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
                                const float* hidden_in, float* hidden_out, int n_tok) {
    const KimiK3Config& cfg = *fwd.cfg;
    const KimiK3LayerWeights& L = fwd.w->layers[layer];
    KimiK3RuntimeState& st = *fwd.state;
    auto& s = *fwd.s;
    cudaStream_t stream = fwd.stream;
    const float eps = cfg.rms_eps;

    const bool do_attn   = (phase == K3LayerPhase::All || phase == K3LayerPhase::Attn);
    const bool do_ffn_prepare = (phase == K3LayerPhase::All ||
                                 phase == K3LayerPhase::FfnPrepare ||
                                 phase == K3LayerPhase::FfnPartial);
    const bool do_ffn_p  = (phase == K3LayerPhase::All || phase == K3LayerPhase::FfnPartial);
    const bool do_ffn_f  = (phase == K3LayerPhase::All || phase == K3LayerPhase::FfnFinish);

    // ---- THE TOKEN AXIS: what it costs to get wrong, and the three refusals ----
    //
    // 1. THE ATTN PHASE IS REFUSED UNLESS kimi_k3_attn_batch_ok() SAYS OTHERWISE, and
    //    the one thing it never says yes to is a KDA layer. The KDA delta rule carries
    //    state from token b-1 to token b and the short conv shifts a window per token;
    //    there is no launch that does either for a chunk without changing the
    //    algorithm. The MLA half IS batched (see the phase below and the contract in
    //    kimi_k3.h): its cross-token dependence is a LENGTH, and every attention kernel
    //    already lengths itself from its own *d_pos.
    //
    //    What made this refusable in the first place has not gone away. Running B
    //    tokens through attention with ONE d_pos makes every one of them write the same
    //    KV row and attend over the same prefix — silently wrong AND faster, i.e. it
    //    survives a timing run, which is the exact failure this project already paid a
    //    round for. The batched phase therefore indexes st.d_pos[b] and lengths token b
    //    at st.position + 1 + b, and the driver's job is to make that pointer the chunk
    //    position vector. `All` carries the Attn phase, so it is refused with it.
    // 2. n_tok must fit the scratch. A row past the cap is a stomp on whatever cudaMalloc
    //    handed out next, so it is checked rather than assumed from the driver's env.
    // 3. n_tok < 1 is nonsense, not "do nothing".
    if (n_tok < 1) return false;
    const bool attn_batched =
        (n_tok > 1) && do_attn && kimi_k3_attn_batch_ok(fwd, layer, n_tok);
    if (n_tok != 1 && do_attn && !attn_batched) return false;
    if (n_tok > s.ffn_cap) return false;

    const int H = cfg.hidden;
    // ---- PER-TOKEN STRIDES, in elements. Every one is the buffer's own single-token
    // width, so at n_tok == 1 none of them is ever added to a pointer. ----
    //
    // moe_row is the FUSED width, not expert_latent and not hidden: moe_out and
    // shexp_out are two views of ONE payload (see kimi_k3_forward_alloc_scratch), so a
    // token's shexp partial sits at (b*moe_row + expert_latent), and striding shexp_out
    // by H instead would walk into the NEXT token's expert accumulator.
    const int64_t moe_row  = (int64_t)cfg.expert_latent + H;
    // Token b's residual bank. Cross-layer per-token state, so the batched phases index
    // it explicitly; the caller binds TOKEN 0's bank and this walks from there.
    const int64_t bank_row = (int64_t)st.res_bank_row_elems * st.max_ckpt;
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

    // ---- MLA-SIDE PER-TOKEN STRIDES, in elements ----------------------------
    //
    // Each is the buffer's own single-token width, so at n_tok == 1 every one of them
    // is multiplied by b == 0 and no pointer moves. THEY ARE NOT ALL THE SAME NUMBER,
    // and that is the point of writing them out: kv_a_out is key_length (576) wide
    // because the KV store reads its rope tail past kv_lora, while kv_cmpr_normed is
    // kv_lora_rank (512). Handing both one shared stride is silently wrong from token 1
    // onward — the same shape of bug as striding s.shexp_out by hidden instead of by
    // the fused MoE width.
    //
    // The head-count is THIS RANK's (md.n_heads), so under head sharding these pitches
    // are an eighth of the allocation's per-row width. Narrower than the allocation
    // assumed is safe; wider would not be.
    const int64_t qlora_row = cfg.q_lora_rank;
    const int64_t qproj_row = (int64_t)qh * cfg.key_length_mla;
    const int64_t kva_row   = cfg.key_length;
    const int64_t kvcmpr_row = cfg.kv_lora_rank;
    const int64_t absq_row  = (int64_t)qh * cfg.key_length;
    const int64_t mlaout_row = (int64_t)qh * cfg.value_length_mla;
    // How many rows the Attn phase runs. ONE unless this layer took the batched path,
    // which keeps every expression below in the form the single-token path already had.
    const int a_tok = attn_batched ? n_tok : 1;

    // ---- DAG lanes: which stream (and which q8 scratch) a call is issued on ----
    //
    // A LANE IS A STREAM PLUS ITS OWN QUANTISATION SCRATCH, never one without the
    // other. Threading the stream alone through the projections would leave both
    // lanes writing s.proj_q8 and is the torn-activation race described on
    // Scratch::proj_q8_lane.
    struct Lane { cudaStream_t st; void* q8; };
    const Lane main_lane{stream, s.proj_q8};
    // Every site is gated twice: SPARKINFER_K3_DAG turns the whole thing off in one
    // move, and a per-site switch isolates one fork on ONE binary. A bundle whose
    // members cannot be measured separately hides a regression inside a win --
    // which is how the -2.6% bundle shipped.
    static const bool dag_all = [] {
        const char* e = std::getenv("SPARKINFER_K3_DAG");
        return !(e && e[0] == '0');
    }();
    auto site_on = [](const char* name) {
        const char* e = std::getenv(name);
        return !(e && e[0] == '0');
    };
    static const bool dag_mla = site_on("SPARKINFER_K3_DAG_MLA");
    static const bool dag_kda = site_on("SPARKINFER_K3_DAG_KDA");
    static const bool dag_moe = site_on("SPARKINFER_K3_DAG_MOE");
    // THE DEBUG TAPS ARE NOT DAG-SAFE, AND THAT IS THE RIGHT TRADE.
    //
    // fwd.debug reads a scratch buffer straight after the launch that wrote it, on
    // the MAIN stream. A buffer written on a lane is not ordered against that read
    // until the join, so under a fork the taps would compare a value that has not
    // been computed yet -- reporting a mismatch in the one mode whose whole job is
    // to localise mismatches. The per-layer validation path runs main's linear
    // order instead; the scored path never sets fwd.debug.
    //
    // AND THE LANES ARE OFF WHEN THE TOKEN AXIS IS ON. A lane is a stream PLUS its own
    // quantisation scratch, and the lane scratches are sized for ONE token — a batched
    // projection issued on a lane would stage n_tok rows of Q8_0 into a one-row buffer.
    // Sizing them for the cap would fix the overflow and buy nothing: the fork exists
    // to hide the latency of small dependent launches, which is exactly what batching
    // removes. So the batched FFN is a linear stream, and at n_tok == 1 this term is
    // `true` and the expression is unchanged.
    const bool dag = dag_all && s.lanes_ok && !fwd.debug && n_tok == 1;

    // Fork n lanes off the main stream HERE. One event, n waiters: the lanes all
    // branch from the same point, so recording once is the same graph as recording
    // n times and one node smaller.
    // OPEN-LANE BOOKKEEPING. Which lanes are outstanding is tracked in a bitmask
    // rather than assumed from control flow, because the phases below have error
    // returns between a fork and its join -- a refused projection, a weight that
    // fails .ok(). Returning with work still queued on a side stream makes the
    // enclosing cudaStreamEndCapture fail far from the cause. A mask, not a count:
    // joins do not always run in index order, so a count cannot say WHICH lane is
    // still open. These paths cannot fire on a well-formed model, which is exactly
    // why they must not be left to reasoning.
    unsigned lanes_open = 0;
    auto dag_fork = [&](int n) {
        if (!dag) return;
        cudaEventRecord(s.ev_fork, stream);
        for (int i = 0; i < n; ++i) cudaStreamWaitEvent(s.lane[i], s.ev_fork, 0);
        lanes_open |= (1u << n) - 1u;
    };
    // Join lane i back. MUST run for every lane that was forked, before the phase
    // returns: cudaStreamEndCapture fails outright on a capture with unjoined work,
    // so a missed join is a loud failure at capture rather than a quiet race.
    auto dag_join = [&](int i) {
        if (!dag) return;
        cudaEventRecord(s.ev_lane[i], s.lane[i]);
        cudaStreamWaitEvent(stream, s.ev_lane[i], 0);
        lanes_open &= ~(1u << i);
    };
    // Closes whatever is still open when this function returns, however it returns.
    // On the normal path every lane is already joined and this is a no-op.
    const K3LaneScope lane_scope{&lanes_open, dag_join};
    auto lane_of = [&](int i) {
        return dag ? Lane{s.lane[i], s.proj_q8_lane[i]} : main_lane;
    };

    // MEASUREMENT ONLY, and it never ships enabled. SPARKINFER_K3_PROJ_REPEAT=N runs every
    // Q8_0 projection N times instead of once.
    //
    // WHY REPEAT RATHER THAN SKIP. The obvious instrument is to ablate the projections and
    // take the delta, and it lies: the projections feed the router, so skipping them hands
    // the MoE dispatch garbage expert ids, which changes how many experts are LOCAL and
    // therefore how much work the dispatch does. The measurement would move two things and
    // attribute both to one. Repeating instead recomputes the SAME values into the SAME
    // buffer -- idempotent, so nothing downstream can tell, routing is untouched, accuracy
    // is untouched, and the graph still captures because only the launch count changes.
    //
    // (N-1) x the delta is the marginal cost of one projection pass, which is exactly the
    // quantity a batched prefill path would be amortising. It is the number that decides
    // whether the driver is worth building.
    static const bool k3_proj_dump = [] {
        const char* e = std::getenv("SPARKINFER_K3_PROJ_DUMP");
        return e && e[0] == '1';
    }();
    static const int k3_proj_repeat = [] {
        const char* e = std::getenv("SPARKINFER_K3_PROJ_REPEAT");
        const int v = e ? std::atoi(e) : 1;
        return (v >= 1 && v <= 8) ? v : 1;
    }();

    auto proj_once = [&](Lane ln, float* y, const float* x, const KimiK3Tensor& W,
                         int N, int K) {
        if (ggml_qact_proj) {
            // SoA-FIRST, restored deliberately: the SoA kernel now carries the
            // one-barrier stash/fold verbatim (P2), so the epilogue confound that
            // poisoned the first A/B is gone — enabling Q8SOA now changes ONLY the
            // load width (2x LDG.128 per block vs 17 narrow loads). Gate off, and
            // this line never fires; gate on, the A/B is clean.
            if (k3k::k3_proj_q8soa_f32(y, x, W.data, N, K, ln.q8, ln.st))
                return true;
            // The one-barrier multi-row kernel is bit-identical and declines every
            // shape it does not improve, so this is a launch-geometry choice and not
            // a numerical one.
            if (k3k::k3_proj_q8_multirow_1bar(y, x, W.data, W.type, N, K,
                                              ln.q8, ln.st))
                return true;
            return k3k::k3_proj_ggml_f32(y, x, W.data, W.type, N, K,
                                         ln.q8, ln.st);
        }
        return k3k::k3_proj_f32(y, x, W.data, W.type, N, K, ln.st);
    };
    auto proj_on = [&](Lane ln, float* y, const float* x, const KimiK3Tensor& W,
                       int N, int K) {
        if (!W.ok()) return false;
        // MEASUREMENT ONLY. Dump each distinct (N,K) once so the projection microbenchmark
        // can be run at the shapes the ENGINE actually issues per rank, not at the full
        // tensor. Getting this wrong once already produced a flattering number.
        if (k3_proj_dump) {
            static std::set<std::pair<int,int>> seen;
            static std::mutex m;
            std::lock_guard<std::mutex> g(m);
            if (seen.insert({N, K}).second)
                std::fprintf(stderr, "[projshape] N=%d K=%d type=%d\n", N, K, W.type);
        }
        bool ok = false;
        for (int rep = 0; rep < k3_proj_repeat; ++rep)
            ok = proj_once(ln, y, x, W, N, K);
        return ok;
    };
    auto proj = [&](float* y, const float* x, const KimiK3Tensor& W, int N, int K) {
        return proj_on(main_lane, y, x, W, N, K);
    };

    // ---- quantise an activation ONCE, then project from it repeatedly ----
    //
    // `normed` feeds three consumers on a KDA layer and three on an MLA one; `normed2`
    // feeds four on every MoE layer. Each of those calls re-quantised the identical
    // vector: 61,848 quantize_q8_0 launches per profile, 7.9% of GPU kernel time, of
    // which 462 per token per rank are recomputing bytes already in the scratch.
    //
    // hoist_act() writes s.act_q8 and proj_h() reads it. The window between them is
    // straight-line code on one stream, so there is nothing to promise and nothing to
    // invalidate -- which is the difference between this and the cached version that
    // shipped top1 0.0. proj_h falls back to a full k3_proj_ggml_f32 whenever the hoist
    // did not apply (f32 weights, or qact off), so a missed hoist is slow, never wrong.
    //
    // SPARKINFER_K3_QACT_HOIST=0 restores per-call quantisation on one binary. Default
    // ON: the harness scores a default build.
    static const bool want_hoist = [] {
        const char* e = std::getenv("SPARKINFER_K3_QACT_HOIST");
        return !(e && e[0] == '0');
    }();
    const float* hoisted_src = nullptr;      // which activation s.act_q8 currently holds
    auto hoist_act = [&](const float* x, int K) {
        hoisted_src = nullptr;
        if (!want_hoist || !ggml_qact_proj || !s.act_q8) return;
        if (n_tok > 1) {
            // The two activations this ever stages -- s.normed in the Attn phase and
            // s.normed2 in FfnPartial -- are both allocated CONTIGUOUS at exactly K
            // (== H) per token, which is the one precondition the flat row quantise
            // cannot check for itself. That is now load-bearing for BOTH: since the
            // batched MLA Attn phase reaches this too, s.normed had to grow a row axis
            // at exactly H (see kimi_k3_forward_alloc_scratch), not at some padded
            // leading dimension. A padded s.normed would quantise the padding as data.
            if (k3k::k3_quantize_act_rows_f32(s.act_q8, x, K, n_tok, stream))
                hoisted_src = x;
            return;
        }
        if (k3k::k3_quantize_act_f32(s.act_q8, x, K, stream)) hoisted_src = x;
    };
    // The pointer check is not a staleness guard -- it only asks "is this the activation
    // I just hoisted", and both are set within a few lines of each other. Anything else
    // takes the ordinary path.
    auto proj_h_on = [&](Lane ln, float* y, const float* x, const KimiK3Tensor& W,
                         int N, int K) {
        if (!W.ok()) return false;
        if (hoisted_src == x && W.type == 8) {
            // THE HOIST WAS SHORT-CIRCUITING THE ONE-BARRIER EPILOGUE.
            //
            // #90 added k3_proj_q8_multirow_1bar to replace the per-row block_sum --
            // two __syncthreads each -- with a single fold, and its own comment names
            // ffn_routed_down (N 3584, K 7168, ROWS 8, 2 main-loop iterations) as a
            // target shape. But this lambda returned k3_proj_q8act_f32 before the 1bar
            // path could ever be reached, so routed_down kept paying 16 serialised
            // block-wide barriers on all 92 MoE layers -- at 448 CTAs over 132 SMs
            // there is nothing resident to hide them behind. attn_q_a and attn_gate
            // (N 1536, ROWS 4, 8 barriers) lost it on all 24 MLA layers for the same
            // reason. The two optimisations were mutually exclusive and the hoist won
            // silently.
            //
            // x_pre_q8 is what reconciles them, and it already existed with no caller:
            // k3_quantize_act_f32 is a thin wrapper over the SAME k3_quantize_q8_0 the
            // 1bar path would call itself, so s.act_q8 already holds byte-identical
            // Q8_0 and the kernel simply skips re-quantising. BIT-IDENTICAL, and it
            // does not write the scratch, so later proj_h calls on the same hoisted
            // activation are unaffected. k3_proj_q8_fused4_1bar at the KDA group is
            // the precedent -- it has passed qkvg_pre this way since #94.
            // s.act_q8 is READ-ONLY here (x_pre_q8 skips the re-quantise), so every
            // lane can share it. That is the whole reason the hoist and proj_q8 are
            // separate buffers: a read-only activation is safe to fan out, a
            // scratch that each call rewrites is not.
            if (k3k::k3_proj_q8_multirow_1bar(y, x, W.data, W.type, N, K,
                                              s.act_q8, ln.st, /*x_pre_q8=*/true))
                return true;
            return k3k::k3_proj_q8act_f32(y, s.act_q8, W.data, W.type, N, K, ln.st);
        }
        return proj_on(ln, y, x, W, N, K);
    };
    auto proj_h = [&](float* y, const float* x, const KimiK3Tensor& W, int N, int K) {
        return proj_h_on(main_lane, y, x, W, N, K);
    };

    // ======================================================================
    // TOKEN-BATCHED PROJECTION. Everything below is dead code at n_tok == 1.
    // ======================================================================
    //
    // Ingestion's problem is the mirror image of decode's. Decode has one activation
    // and re-reads it from every output row, so the decode kernels amortise the
    // ACTIVATION. Ingestion re-streams every WEIGHT once per prompt token -- 15.2 GB
    // per rank per token against a 7.6 KB activation -- so a 32k prompt reads the model
    // 32,768 times. k3_proj_q8act_tok_f32 reads each weight tile ONCE for n_tok rows
    // and is documented bit-identical per output element to n_tok separate
    // k3_proj_q8act_f32 calls (same k-order, same int32 accumulation, same block
    // reduction; only the CUDA-block grouping moves).
    //
    // EVERY ONE OF THESE RETURNS false MEANING "USE THE SLOW PATH", NEVER "ERROR".
    // The fallback is the per-token loop, which at n_tok == 1 is the single call this
    // function has always made.

    // Stage n_tok token-major rows of `x` into `q8buf` as a CONTIGUOUS [n_tok, K/32]
    // Q8_0 array -- the layout k3_proj_q8act_tok_f32 reads with ld_act_blocks = K/32.
    //
    // TWO SHAPES, because the source is not always contiguous. When x's rows are
    // exactly K apart the whole thing is one flat run of blocks and quantises in ONE
    // launch (k3_quantize_act_rows_f32; the kernel is flat over 32-element blocks and
    // has no row index at all, which is what makes that free). When x sits in a buffer
    // with a PADDED leading dimension -- s.dense_situ is allocated at dense_ffn but the
    // shared expert only fills shexp_band of it -- there is no stride to hand a kernel
    // that has no row concept, so each row is quantised by its own call INTO ITS OWN
    // CONTIGUOUS SLOT. That is n_tok launches instead of 1, but it is still n_tok
    // launches instead of n_tok*(quantise + projection), and it is the same kernel over
    // the same 32 values per block either way: bit-identical, not merely equivalent.
    auto quant_rows = [&](void* q8buf, const float* x, int K, int64_t x_stride) {
        if (!q8buf || !x || K <= 0 || K % 32 != 0) return false;
        if (x_stride == (int64_t)K)
            return k3k::k3_quantize_act_rows_f32(q8buf, x, K, n_tok, stream);
        const size_t row_bytes = k3k::k3_q8_0_bytes(K);
        if (row_bytes == 0) return false;
        for (int b = 0; b < n_tok; ++b)
            if (!k3k::k3_quantize_act_f32((char*)q8buf + (size_t)b * row_bytes,
                                          x + (int64_t)b * x_stride, K, stream))
                return false;
        return true;
    };

    // One batched projection: [n_tok, K] activation -> [n_tok, N] output, y's rows ldy
    // apart and x's rows x_stride apart. `pre_q8` says q8buf ALREADY holds the n_tok
    // staged rows (the hoist wrote them), exactly like k3_proj_q8_multirow_1bar's
    // x_pre_q8 -- it is a statement about the two lines around the call, never a cache
    // lookup. The cached form of that promise is what shipped top1 0.0 once already.
    // ldy and x_stride are int64_t here because s.shexp_out's row stride is the FUSED
    // moe width and every other stride in this file is already an int64_t element
    // count; k3_proj_q8act_tok_f32 takes an int ldy, so the one narrowing happens here,
    // guarded, rather than silently at each of the six call sites.
    auto proj_tok = [&](float* y, int64_t ldy, const float* x, int64_t x_stride,
                        const KimiK3Tensor& W, int N, int K, void* q8buf, bool pre_q8) {
        if (n_tok <= 1) return false;          // nothing to amortise over
        if (!ggml_qact_proj) return false;     // f32 activations have no staged form
        if (!W.ok() || W.type != 8) return false;
        if (!q8buf || K <= 0 || K % 32 != 0) return false;
        if (ldy < (int64_t)N || ldy > 2147483647LL) return false;
        if (!pre_q8 && !quant_rows(q8buf, x, K, x_stride)) return false;
        return k3k::k3_proj_q8act_tok_f32(y, (int)ldy, q8buf, K / 32, W.data, W.type,
                                          N, K, n_tok, stream);
    };

    // The two call-site forms. Both fall back to a per-token loop of the EXISTING call,
    // so at n_tok == 1 they are `proj(...)` and `proj_h(...)` with the loop run once and
    // b == 0 adding nothing to any pointer.
    auto proj_b = [&](float* y, int64_t ldy, const float* x, int64_t x_stride,
                      const KimiK3Tensor& W, int N, int K) {
        if (proj_tok(y, ldy, x, x_stride, W, N, K, s.proj_q8, /*pre_q8=*/false))
            return true;
        for (int b = 0; b < n_tok; ++b)
            if (!proj(y + (int64_t)b * ldy, x + (int64_t)b * x_stride, W, N, K))
                return false;
        return true;
    };
    // Hoisted variant. When `x` is the activation hoist_act staged, the staged rows are
    // already in s.act_q8 and the quantise is skipped entirely; when it is not, the rows
    // are staged into s.proj_q8 here. s.act_q8 is READ-ONLY on the batched path for the
    // same reason it is on the decode one -- that is why the hoist and proj_q8 are
    // separate buffers, and why a projection landing between a hoist and its consumers
    // cannot corrupt it.
    auto proj_hb = [&](float* y, int64_t ldy, const float* x, int64_t x_stride,
                       const KimiK3Tensor& W, int N, int K) {
        const bool pre = (hoisted_src == x);
        if (proj_tok(y, ldy, x, x_stride, W, N, K, pre ? s.act_q8 : s.proj_q8, pre))
            return true;
        for (int b = 0; b < n_tok; ++b)
            if (!proj_h(y + (int64_t)b * ldy, x + (int64_t)b * x_stride, W, N, K))
                return false;
        return true;
    };
    // LANE-AWARE FORMS, for the two MLA sites that issue on a side stream.
    //
    // The batched kernel always goes to `stream`, and that is correct rather than a
    // shortcut: `dag` carries `n_tok == 1` (a lane's q8 scratch holds ONE row), so at
    // every n_tok where proj_tok can fire, dag is false and ln.st IS stream. At
    // n_tok == 1 proj_tok declines on its first line and the loop below is the exact
    // single call these sites have always made, on their own lane.
    auto proj_hb_on = [&](Lane ln, float* y, int64_t ldy, const float* x,
                          int64_t x_stride, const KimiK3Tensor& W, int N, int K) {
        const bool pre = (hoisted_src == x);
        if (proj_tok(y, ldy, x, x_stride, W, N, K, pre ? s.act_q8 : s.proj_q8, pre))
            return true;
        for (int b = 0; b < n_tok; ++b)
            if (!proj_h_on(ln, y + (int64_t)b * ldy, x + (int64_t)b * x_stride, W, N, K))
                return false;
        return true;
    };

    // ---- phase 1: attention. Ends holding a FULL-WIDTH PARTIAL SUM,
    // because attn_output is ColShard. The driver reduces it before phase 2. ----
    // ---- PREFILL TILE REUSE: take the fill's work instead of redoing it ----
    //
    // WHY THIS EXISTS, MEASURED. The tile fill already computes this token's mix, norm
    // and quantised activation, and the consumer used to recompute all three and then
    // COPY the four projections in — so a tile hit cost more launches than a tile miss.
    // Counted per token per KDA layer: a miss is mix, push, norm, hoist, fused4 = 5
    // nodes; the copying hit was 10 + 5/T. That is why the batched fill lost to its own
    // control at every tile width (49.86 vs 51.00 at T=4, 42.21 vs 45.24 at T=16) while
    // doing strictly less arithmetic.
    //
    // It matters because prefill under capture is bound by the SIZE of the recorded
    // graph, not by arithmetic: capture is worth 1.93x on the per-token path's 3308-node
    // graph and only 1.56x on a T=16 tile's 61258-node one. Nodes are the currency.
    //
    // Reusing instead of recomputing makes a hit 3 + 5/T — cheaper than a miss.
    //
    // WHAT IS NOT REUSED, AND MUST NOT BE. The bank PUSH stays here. The fill reads each
    // token's checkpoint bank as it stands on ENTRY to the layer and never writes it, so
    // the push is the consumer's job alone; skipping it would leave the bank one
    // checkpoint short for every later layer, and skipping the n_ckpt increment would
    // mix the wrong prefix. Both are fluent-wrong failures, not crashes.
    //
    // The pointers are ALIASED, not copied, and restored by the guard below on every
    // exit path — the same discipline res_bank_orig and kimi_k3_swap_partial_buffer
    // already use. Aliasing is safe because a tile row belongs to exactly one token and
    // is dead once that token's layer is done: the fill overwrites it at the next layer.
    // SPARKINFER_K3_PREFILL_REUSE=0 makes the consumer ignore the tile ENTIRELY — it
    // recomputes the mix, the norm and the quantise and runs its own fused4, which is
    // exactly what a tile miss does. So =0 is the fill-OFF control's consumer with the
    // fill still running, which isolates the fill's cost from the reuse's saving, and =1
    // is the change. One binary, both arms.
    static const bool want_tile_reuse = [] {
        const char* e = std::getenv("SPARKINFER_K3_PREFILL_REUSE");
        return !(e && e[0] == '0');
    }();
    const K3PrefillTile* pt_hit = nullptr;
    // `a_tok == 1` MAKES TILE REUSE AND THE TOKEN AXIS PROVABLY EXCLUSIVE, in one place
    // rather than at each of the five sites below that branch on pt_hit. A tile row is a
    // SINGLE token's q/k/v/g/normed, selected by fwd.prefill_tok; there is no such thing
    // as a tile hit for a chunk of a_tok rows, so reusing one would silently hand tokens
    // 1..a_tok-1 token 0's activations. Today the two cannot co-occur anyway — only
    // kimi_k3_tp_prefill sets fwd.prefill_tile and it drives one token at a time — so
    // this costs nothing and turns "cannot happen" into "cannot compile a path where it
    // happens".
    if (fwd.prefill_tile && want_tile_reuse && !fwd.debug && a_tok == 1) {
        // The debug taps read these buffers straight after the launch that wrote them,
        // and on a reuse there IS no such launch — the value was produced by the fill,
        // one call earlier. Localising a mismatch is the taps' whole job, so they get
        // the recomputing path.
        const K3PrefillTile* pt = (const K3PrefillTile*)fwd.prefill_tile;
        if (pt->layer == layer && pt->hidden == H &&
            fwd.prefill_tok >= 0 && fwd.prefill_tok < pt->n_live)
            pt_hit = pt;
    }
    // Holds the SLOTS rather than the scratch, so it never has to name
    // KimiK3Forward::Scratch (a nested type this free function cannot spell).
    struct K3TileAlias {
        float** normed_p = nullptr; float* normed0 = nullptr;
        void**  act_p    = nullptr; void*  act0    = nullptr;
        float **qp = nullptr, **kp = nullptr, **vp = nullptr, **gp = nullptr;
        float **gop = nullptr;
        float  *q0 = nullptr,  *k0 = nullptr,  *v0 = nullptr,  *g0 = nullptr;
        float  *go0 = nullptr;
        ~K3TileAlias() {
            if (!normed_p) return;
            *normed_p = normed0; *act_p = act0;
            *qp = q0; *kp = k0; *vp = v0; *gp = g0;
            if (gop) *gop = go0;
        }
    } tile_alias;
    if (pt_hit) {
        tile_alias.normed_p = &s.normed; tile_alias.normed0 = s.normed;
        tile_alias.act_p    = &s.act_q8; tile_alias.act0    = s.act_q8;
        tile_alias.qp = &s.qkv_q;      tile_alias.q0 = s.qkv_q;
        tile_alias.kp = &s.qkv_k;      tile_alias.k0 = s.qkv_k;
        tile_alias.vp = &s.qkv_v;      tile_alias.v0 = s.qkv_v;
        tile_alias.gp = &s.g_proj_out; tile_alias.g0 = s.g_proj_out;
        if (pt_hit->kda_out_layer == layer && pt_hit->kda_gate_out) {
            tile_alias.gop = &s.gate_out; tile_alias.go0 = s.gate_out;
            s.gate_out = pt_hit->kda_gate_out + (size_t)fwd.prefill_tok * pt_hit->qkv;
        }
    }

    if (do_attn) {
        // --- pre-attention mix, then bank push (raw pre-mix value) ---
        //
        // BATCHED. `out` and `cur` are token-major activations and move by H; `ckpts`
        // is the CROSS-LAYER residual bank and moves by bank_row. Two different pitches,
        // which is why the kernel takes two -- and the ORDER is (n_rows, act_row_stride,
        // bank_row_stride), not (n_rows, bank_row, H).
        //
        // ALL THREE MIX CALL SITES HAD THOSE TWO SWAPPED, and it hid because they are
        // EQUAL until the model is deep enough to bank twice: bank_row is
        // res_bank_row_elems * max_ckpt, max_ckpt is ceil(n_layers / 12), so at <= 12
        // layers bank_row == H and each wrong argument lands on the right value. At 13+
        // layers max_ckpt is 2, bank_row is 2H, and both strides are wrong -- the
        // activations get read 2H apart and the banks H apart. n_rows == 1 never touches
        // either stride, so decode, the token loop and a 1-token chunk were all exact.
        // Bisected on the 4-token probe: bit-identical at 12 layers, KLD 1.64 at 14.
        //
        // n_ckpt STAYS A SCALAR, and that is a statement about the model rather than a
        // simplification: banking is a function of the LAYER (res_bs > 0 && layer %
        // res_bs == 0), so every token of a chunk enters this layer having banked the
        // same count. The driver restores it before the call and reads it back after.
        if (res_bs > 0) {
            if (!L.attn_res_score.ok()) return false;
            // The mix is skipped on a hit, the PUSH below is not — see the note above.
            // The token axis rides through unchanged: a tile hit only ever happens on
            // the tile driver, where a_tok is 1, so the two gates never interact.
            if (!pt_hit) {
                k3k::attn_res_mix_f32(s.mixed, st.res_bank, hidden_in,
                                      (const float*)L.attn_res_score.data, H, st.n_ckpt,
                                      eps, stream, s.res_scores, /*cur_b=*/nullptr,
                                      /*sum_out=*/nullptr, a_tok,
                                      /*act_row_stride=*/H,
                                      /*bank_row_stride=*/bank_row);
            }
            if (banked) {
                if (st.n_ckpt >= st.max_ckpt) return false;
                // ONE PUSH PER TOKEN, INTO THAT TOKEN'S OWN BANK, AT THE SAME SLOT.
                // The slot is n_ckpt for every row (see above); the BANK is what moves,
                // by bank_row. Writing all a_tok rows at (res_bank + n_ckpt*H) would put
                // every token's checkpoint in token 0's bank — invisible at short
                // context, where n_ckpt is 0 or 1, and wrong at depth. n_ckpt advances
                // ONCE for the chunk, not once per row, for the same reason.
                for (int b = 0; b < a_tok; ++b)
                    cudaMemcpyAsync(st.res_bank + (int64_t)b * bank_row +
                                        (size_t)st.n_ckpt * H,
                                    hidden_in + (int64_t)b * H,
                                    (size_t)H * sizeof(float), cudaMemcpyDeviceToDevice,
                                    stream);
                ++st.n_ckpt;
            }
        } else if (!pt_hit) {
            // Both sides are contiguous at H, so the whole chunk is ONE copy.
            cudaMemcpyAsync(s.mixed, hidden_in, (size_t)H * a_tok * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        }
        if (fwd.debug) fwd.debug("attn_res_mix", layer, s.mixed, H);

        if (!L.attn_norm.ok()) return false;
        if (pt_hit) {
            // The fill produced BOTH of these for this row, from the same inputs with the
            // same kernels — bit-identical by construction, which the sweep's 0.0 KLD at
            // every tile width is the end-to-end check on.
            s.normed = pt_hit->normed + (size_t)fwd.prefill_tok * (size_t)H;
            s.act_q8 = (char*)pt_hit->q8 +
                       (size_t)fwd.prefill_tok * k3k::k3_q8_0_bytes(H);
            // Tell proj_h_on that act_q8 already holds THIS activation, so ssm_f_a and
            // ssm_beta skip their re-quantise too. Without this they would quantise
            // s.normed again and the saving would stop at the qkvg group.
            hoisted_src = s.normed;
        } else {
            k3k::rms_norm_f32(s.normed, s.mixed, (const float*)L.attn_norm.data, H, eps,
                              stream, a_tok, H, H);
            if (fwd.debug) fwd.debug("attn_norm", layer, s.normed, H);
            // normed feeds ssm_f_a and ssm_beta on a KDA layer, and q_a/q_dense, kv_a and
            // attn_gate on an MLA one. The fused qkvg group keeps its own quantisation --
            // it is one launch that already amortises it across four tensors.
            hoist_act(s.normed, H);
        }

        k3_profiler().start(L.is_kda ? "attn_kda" : "attn_mla", stream);
        if (L.is_kda) {
            // BELT AND BRACES ON THE ONE REFUSAL THAT MATTERS. A chunk reaches this
            // block only when SPARKINFER_K3_KDA_QKVG_BATCH is set; without it
            // kimi_k3_attn_batch_ok() still refuses every KDA layer and a_tok is 1.
            // The gate is re-read HERE rather than inferred from a_tok because the
            // single-row scratch below (conv_*, decay_g, beta_out, delta_out) would
            // otherwise take eight tokens into one token's buffers and produce a fluent
            // wrong answer rather than a fault. `L.is_kda` and `cfg.is_kda_layer(layer)`
            // are two derivations of the same fact; this is where they are made to agree.
            if (a_tok != 1 && !k3_kda_qkvg_batch_enabled()) return false;
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

            // --- FORK: the decay chain and beta ------------------------------
            //
            // ssm_f_a -> ssm_f_b -> decay_gate is three dependent launches that
            // read `normed` and are read only by the decode step; ssm_beta is one
            // more of the same. Main's order runs them AFTER the convs and the two
            // L2 norms, so the decode step waits for the sum of both chains when it
            // only ever needed the longer one.
            //
            // Every tensor either chain touches is checked here, before the fork,
            // so the failure paths below stay on a linear stream.
            //
            // ONE FORK, ONE JOIN, AND THE JOIN NOW SITS INSIDE A LOOP. `dag` already
            // carries n_tok == 1 (a lane owns ONE row of q8 scratch), so a chunk can
            // never reach the fork — but the join below is now per-token, and a fork
            // taken once against a_tok joins would be an unbalanced capture rather than
            // a wrong number. Said explicitly here instead of relied upon from `dag`.
            const bool kda_fork = a_tok == 1 && dag && dag_kda &&
                L.ssm_f_a.ok() && L.ssm_f_b.ok() && L.ssm_beta.ok() &&
                L.ssm_dt_bias.ok() && L.ssm_a.ok() &&
                // The convs are on the MAIN branch, but their check sits between
                // the fork and the join. Folding it in here keeps a missing conv a
                // plain early return on a linear stream instead of one that leaves
                // the capture unjoined.
                L.ssm_conv1d_q.ok() && L.ssm_conv1d_k.ok() && L.ssm_conv1d_v.ok();
            if (kda_fork) dag_fork(2);
            const Lane l_dec  = kda_fork ? lane_of(0) : main_lane;   // f_a -> f_b -> decay
            const Lane l_beta = kda_fork ? lane_of(1) : main_lane;   // beta

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
            // The shape check is shared; only the activation format differs. Under
            // qact this used to be `!ggml_qact_proj && ...`, which meant enabling
            // quantised activations silently DISABLED the fusion on all 69 KDA
            // layers — four launches instead of one, and four quantisations of the
            // identical s.normed, because k3_proj_ggml_f32 quantises per call.
            // That was 8.7% of GPU time in 59,696 quantize launches.
            // SPARKINFER_K3_FUSE_QKVG=0 forces the four separate projections, so
            // the fusion can be A/B'd on ONE binary. Every reliable measurement on
            // this branch has come from a same-binary control; the ones that were
            // not (rebuild vs rebuild) are the ones that had to be retracted.
            static const bool want_fuse = [] {
                const char* e = std::getenv("SPARKINFER_K3_FUSE_QKVG");
                return !(e && e[0] == '0');
            }();
            const bool fusable = want_fuse &&
                L.attn_q.ok() && L.attn_k.ok() && L.attn_v.ok() && L.ssm_g.ok() &&
                L.attn_q.type == 8 && L.attn_k.type == 8 &&
                L.attn_v.type == 8 && L.ssm_g.type == 8;
            // The hoist above already quantised `normed` into act_q8; hand the fused
            // group that buffer instead of letting it quantise the same bytes again.
            // Without this the hoist was a net LOSS on all 69 KDA layers -- it added a
            // launch that nothing consumed, because ssm_f_a and ssm_beta are f32-weighted
            // and never quantised in the first place. Measured -233 launches/token/rank
            // against the -462 the arithmetic predicted; this is the missing 69.
            const bool qkvg_pre = (hoisted_src == s.normed);
            if (k3_proj_dump) {
                static bool once = false;
                if (!once) { once = true;
                    std::fprintf(stderr, "[projshape] FUSED4 qkvg N=%d K=%d\n", qkv, H); }
            }
            // The KDA q/k/v/g group is the largest projection in the model and does NOT go
            // through proj_on, so the repeat instrument has to reach it here too or it
            // would price everything except the biggest thing. Same idempotence argument:
            // the extra passes recompute the same four outputs from the same activation.
            for (int rep = 1; rep < k3_proj_repeat && fusable && ggml_qact_proj; ++rep)
                k3k::k3_proj_q8_fused4_1bar(s.qkv_q, s.qkv_k, s.qkv_v, s.g_proj_out,
                                            s.normed, L.attn_q.data, L.attn_k.data,
                                            L.attn_v.data, L.ssm_g.data, L.attn_q.type,
                                            qkv, H, qkvg_pre ? s.act_q8 : s.proj_q8,
                                            stream, qkvg_pre);
            // PREFILL TILE HIT. The batched pass already projected this token's q/k/v/g
            // for this layer, so take the row instead of streaming ~11.7 MB of Q8_0 weights
            // to recompute it. This is the whole point of the batched path: the weight was
            // read once for the tile rather than once per token.
            //
            // The guards are not paranoia. `pt->layer == layer` catches a tile filled for a
            // different layer and `prefill_tok < pt->n_live` catches a ragged final tile
            // read as though it were full; either would produce FLUENT WRONG OUTPUT, which
            // is the failure this file refuses to ship everywhere else. `pt->qkv == qkv`
            // catches a shard-width mismatch, which would corrupt silently by copying the
            // wrong number of floats.
            // ALIASED, NOT COPIED. Four D2D memcpys per token per KDA layer is ~280 graph
            // nodes per token — on a path whose cost IS its node count, that was most of
            // the reason a tile hit lost to a tile miss. Pointing the scratch at the row
            // costs nothing and the downstream attention reads the identical floats.
            //
            // Safe because a tile row belongs to exactly one token and dies with that
            // token's layer: nothing re-reads row t after token t's attention, so even an
            // in-place consumer could not corrupt a value anyone else will see. The
            // restore is the guard declared at the top of this function, which runs on
            // every exit path including the error returns below.
            bool qkvg_from_tile = false;
            if (pt_hit && pt_hit->qkv == qkv) {
                // The qkv guard is the one check that could not be hoisted to pt_hit: it
                // catches a shard-width mismatch, and a mismatch would hand attention the
                // wrong number of floats rather than fail.
                const size_t off = (size_t)fwd.prefill_tok * (size_t)qkv;
                s.qkv_q      = pt_hit->q + off;
                s.qkv_k      = pt_hit->k + off;
                s.qkv_v      = pt_hit->v + off;
                s.g_proj_out = pt_hit->g + off;
                qkvg_from_tile = true;
            }
            // THE TILE ARM ABOVE AND THE CHUNK ARM BELOW CANNOT BOTH FIRE. `pt_hit` is
            // only ever set under `a_tok == 1` (see the gate at the top of this function),
            // so `qkvg_from_tile` implies a single token and `a_tok > 1` implies no tile.
            // That exclusivity is what lets the two writers of s.qkv_* coexist in one
            // function: the tile arm ALIASES those pointers at a foreign buffer and the
            // chunk arm WRITES through them, and if both could fire the chunk projections
            // would scribble into the tile driver's rows.
            //
            // ---- THE CHUNK ARM: FOUR TOKEN-BATCHED PROJECTIONS, NOT FOUR FUSED ONES ----
            //
            // WHAT IT REPLACES. At a_tok == 1 the group is ONE launch of
            // proj_q8_fused4_1bar_kernel that streams 4 x 1536 x 7168 of Q8_0 weight —
            // 46.8 MB per layer per rank — and spends it on ONE token. Over 69 KDA layers
            // that is 3.23 GB per token per rank, the single largest weight stream in the
            // model, and prompt ingestion re-reads all of it for every one of 32,768
            // tokens. proj_q8_0_q8_0_tok_kernel reads the same tile once and spends it on
            // a_tok tokens, so the term divides by a_tok; the activation it re-reads
            // instead is 7.6 KB against 11.7 MB.
            //
            // WHY FOUR LAUNCHES IS STILL FEWER. The fused kernel's win was amortising ONE
            // activation over four tensors, which was worth having when the activation was
            // re-quantised per call. On the token axis the weight is what repeats, so
            // four batched launches per layer per CHUNK replace a_tok fused launches per
            // layer; at a chunk of 16 that is 4 instead of 16, with 1/16th the weight
            // traffic. Fusing the token axis and the tensor axis in one kernel is the
            // strictly better shape and is deliberately NOT done here — it needs a new
            // kernel, and this arm needs none.
            //
            // BIT-IDENTICAL, PER OUTPUT ELEMENT, AND HERE IS THE WHOLE ARGUMENT.
            // Compare proj_q8_fused4_1bar_kernel (what runs today) with
            // proj_q8_0_q8_0_tok_kernel (what runs here) for a fixed output y[t][n]:
            //   * b-striding.  Both walk `for (b = tx; b < blocks_per_row; b += BLOCK)`
            //     with tx == threadIdx.x, so thread i owns exactly the same set of k
            //     blocks in the same ascending order. BLOCK comes from the same rule
            //     (block_for / proj_block_for, both `nb<=32?32:nb<=64?64:128`) applied to
            //     the same nb == K/32, so it is the same number: 128 at K == 7168.
            //   * inner product.  Both do `int sumi = 0; for i in 0..7:
            //     sumi = __dp4a(<weight>, <activation>, sumi)` over get_int_b2 in
            //     ascending i — int32, exact, same operand order (weight first).
            //   * float promotion.  Both do `acc += (float)sumi * (dw * dx)`, the same
            //     expression in the same associativity, accumulated in ascending b.
            //   * reduction TREE, which is the one that would have broken it. 1bar does a
            //     shfl_down butterfly off 16,8,4,2,1; lane 0 stashes; one __syncthreads();
            //     one thread folds shm[w] over ASCENDING w. block_sum<BLOCK> (what the tok
            //     kernel calls) does the identical butterfly over the identical lanes,
            //     stashes per warp, and folds w = 0..NWARP-1 ascending. Same partials,
            //     same order, same roundings — 1bar's own header makes this claim against
            //     block_sum and it is the same claim being leaned on here.
            // ROWS and TOKS change only which output elements share a CUDA block; neither
            // appears in any accumulation order. The activation bytes are the same bytes:
            // k3_quantize_act_rows_f32 is k3_quantize_q8_0 over a longer flat run, and
            // s.normed is contiguous at exactly H per token (see its allocation), which is
            // the precondition that flatness cannot check for itself.
            //
            // proj_hb falls back to a per-token loop of proj_h whenever the batched kernel
            // declines (f32 weights, n_tok == 1, a shape it was not written for), so a
            // decline here is the slow path and never a wrong one.
            if (a_tok > 1) {
                if (!proj_hb(s.qkv_q, qkv, s.normed, H, L.attn_q, qkv, H)) return false;
                if (!proj_hb(s.qkv_k, qkv, s.normed, H, L.attn_k, qkv, H)) return false;
                if (!proj_hb(s.qkv_v, qkv, s.normed, H, L.attn_v, qkv, H)) return false;
                if (!proj_hb(s.g_proj_out, qkv, s.normed, H, L.ssm_g, qkv, H)) return false;
            }
            // `fused_qkvg` means "g_proj_out is already written", which the chunk arm
            // above also satisfies — it projects ssm_g alongside q/k/v for the same
            // reason the fused kernel does, so the late g projection below stays skipped.
            //
            // THE SHORT CIRCUIT IS LOAD-BEARING, NOT STYLE. The right operand of this ||
            // LAUNCHES KERNELS. At a_tok > 1 it must not be evaluated at all, or the
            // fused single-token group would run over token 0 and overwrite the chunk's
            // row 0 with a duplicate of itself. `||` guarantees that; an eager form
            // (computing both and or-ing) would be the wrong-and-faster bug again.
            // The same argument covers `qkvg_from_tile`: on a tile hit the right-hand
            // side must not be evaluated either, or the work the tile exists to remove
            // would still be issued.
            const bool fused_qkvg = qkvg_from_tile || a_tok > 1 || (fusable &&
                (ggml_qact_proj
                     ? (k3k::k3_proj_q8_fused4_1bar(s.qkv_q, s.qkv_k, s.qkv_v,
                                                    s.g_proj_out, s.normed,
                                                    L.attn_q.data, L.attn_k.data,
                                                    L.attn_v.data, L.ssm_g.data,
                                                    L.attn_q.type, qkv, H,
                                                    qkvg_pre ? s.act_q8 : s.proj_q8,
                                                    stream, qkvg_pre) ||
                        k3k::k3_proj_ggml_f32_x4(s.qkv_q, s.qkv_k, s.qkv_v, s.g_proj_out,
                                                s.normed, L.attn_q.data, L.attn_k.data,
                                                L.attn_v.data, L.ssm_g.data,
                                                L.attn_q.type, qkv, H,
                                                qkvg_pre ? s.act_q8 : s.proj_q8,
                                                stream, qkvg_pre))
                     : k3k::k3_proj_f32_x4(s.qkv_q, s.qkv_k, s.qkv_v, s.g_proj_out,
                                           s.normed, L.attn_q.data, L.attn_k.data,
                                           L.attn_v.data, L.ssm_g.data,
                                           L.attn_q.type, qkv, H, stream)));
            if (!fused_qkvg) {
                if (!proj_h(s.qkv_q, s.normed, L.attn_q, qkv, H)) return false;
                if (!proj_h(s.qkv_k, s.normed, L.attn_k, qkv, H)) return false;
                if (!proj_h(s.qkv_v, s.normed, L.attn_v, qkv, H)) return false;
            }

            if (!L.ssm_conv1d_q.ok() || !L.ssm_conv1d_k.ok() || !L.ssm_conv1d_v.ok())
                return false;
            // q gets the extra 1/sqrt(head_dim) scale the reference applies before the
            // scan; k does not (see kda_decode_step_f32's contract on pre-scaled q).
            const float q_l2_scale = 1.0f / std::sqrt((float)head_dim);

            // ================= THE SCAN, STILL ONE TOKEN AT A TIME =================
            //
            // Everything from here to the gate is the RECURRENCE and keeps the token
            // loop, in index order, on ONE stream. st.conv_state_* shifts a 4-deep
            // window per token and st.delta_state carries the delta rule from token
            // b-1 into token b; both are updated IN PLACE by the calls below, so the
            // stream order IS the recurrence order and there is nothing else holding
            // it. At a_tok == 1 this loop runs exactly once with b == 0, every offset
            // below is + 0, and the launch sequence is byte-for-byte the one this
            // block has always issued.
            //
            // WHAT IT READS PER TOKEN. qkv_q/k/v and g_proj_out are the chunk-wide
            // buffers the projections above filled, indexed at b * qkv — token b can
            // only reach its own row, and the rows for tokens after b are already
            // written but are never addressed here. normed is indexed at b * H. The
            // scratch it WRITES (conv_*, f_a_out, g_raw, decay_g, beta_out, delta_out)
            // is one row that this iteration produces and consumes before the next
            // iteration overwrites it.
            // ---- PRE-SCAN BATCH: the decay chain for the WHOLE chunk, before the loop.
            //
            // f_a_out <- normed, g_raw <- f_a_out, decay_g <- g_raw. None of the three
            // reads the conv state or the delta state (see k3_kda_pre_batch_enabled), so
            // computing them for all a_tok tokens here is the same arithmetic the loop
            // would do one row at a time — it just does it at 12*a_tok blocks instead of
            // 12, which is the point.
            //
            // The loop below skips this chain when it fired, and reads decay_g at its own
            // row. Everything else in the loop is untouched.
            bool kda_pre = false;
            if (k3_kda_pre_batch_enabled() && a_tok > 1 && !fwd.debug &&
                L.ssm_f_a.ok() && L.ssm_f_b.ok() && L.ssm_dt_bias.ok() && L.ssm_a.ok()) {
                // proj_hb takes (y, ldy, x, x_stride, W, N, K) and falls back to a
                // per-token loop of the single-row projection when its batched kernel
                // declines — so a decline here is slower, never different.
                if (!proj_hb(s.f_a_out, head_dim, s.normed, H, L.ssm_f_a, head_dim, H))
                    return false;
                if (!proj_hb(s.g_raw, qkv, s.f_a_out, head_dim, L.ssm_f_b, qkv, head_dim))
                    return false;
                // Same fused-then-fallback pair the per-token path uses, with the row axis
                // supplied. dt_bias and A are SHARED across rows and are never strided —
                // the kernels take them as the learned parameters they are.
                if (!k3k::k3_kda_decay_gate_dt(s.decay_g, s.g_raw, w_dt, w_a, head_dim,
                                               n_head, cfg.kda_gate_lower_bound, stream,
                                               a_tok, qkv)) {
                    // THE FALLBACK LOOPS, and that is not laziness — k3_add_f32's row axis
                    // strides ALL THREE of its pointers (out, a AND b), which is right for
                    // two tensors of the same shape and wrong here: w_dt is a SHARED bias
                    // of one qkv-wide row, so a strided call would read w_dt + b*qkv, off
                    // the end of the tensor, for every token after the first. The decay
                    // gate's own row axis is safe (it leaves dt_bias and A unstrided), but
                    // there is no point being half-batched across a pair that must agree.
                    for (int pb = 0; pb < a_tok; ++pb) {
                        float* const gr = s.g_raw   + (int64_t)pb * qkv;
                        float* const dg = s.decay_g + (int64_t)pb * qkv;
                        k3k::k3_add_f32(gr, gr, w_dt, qkv, stream);
                        k3k::kda_decay_gate_f32(dg, gr, w_a, head_dim, n_head,
                                                cfg.kda_gate_lower_bound, stream);
                    }
                }
                kda_pre = true;
            }
            for (int b = 0; b < a_tok; ++b) {
            const float* const nrm     = s.normed     + (int64_t)b * H;
            // decay_g carries a row axis only when the pre-scan batch filled it; without
            // it the buffer is one row that this iteration produces and consumes.
            float* const       decay_b = s.decay_g + (kda_pre ? (int64_t)b * qkv : 0);
            float* const       qkv_q_b = s.qkv_q      + (int64_t)b * qkv;
            float* const       qkv_k_b = s.qkv_k      + (int64_t)b * qkv;
            float* const       qkv_v_b = s.qkv_v      + (int64_t)b * qkv;
            float* const       gproj_b = s.g_proj_out + (int64_t)b * qkv;
            float* const       gate_b  = s.gate_out   + (int64_t)b * qkv;

            // FIVE DEPENDENT LAUNCHES FOR ~330 KB, ON ALL 69 KDA LAYERS.
            //
            // The three convs are mutually independent, and the q/k norms reduce over
            // exactly the channels their own conv just wrote — so the group is one
            // kernel over grid (n_head, 3). The two norms are the reason this is worth
            // doing: they ran at n_head = 12 blocks on a 132-SM part.
            //
            // The debug taps below still see the same values; they just observe the
            // fused kernel's output. The fused path declines (and we fall through to
            // the five launches) for any head_dim wider than the block.
            const bool kda_fused = !fwd.debug &&
                k3k::k3_kda_conv_l2_fused(
                    s.conv_q, s.conv_k, s.conv_v,
                    st.conv_state_q[kda_ord], st.conv_state_k[kda_ord],
                    st.conv_state_v[kda_ord],
                    qkv_q_b, qkv_k_b, qkv_v_b, w_cq, w_ck, w_cv,
                    cfg.kda_conv_kernel, head_dim, n_head, q_l2_scale, eps, stream);

            if (!kda_fused) {
                k3k::kda_conv_step_f32(s.conv_q, st.conv_state_q[kda_ord], qkv_q_b,
                                       w_cq, cfg.kda_conv_kernel,
                                       qkv, stream);
                k3k::kda_conv_step_f32(s.conv_k, st.conv_state_k[kda_ord], qkv_k_b,
                                       w_ck, cfg.kda_conv_kernel,
                                       qkv, stream);
                k3k::kda_conv_step_f32(s.conv_v, st.conv_state_v[kda_ord], qkv_v_b,
                                       w_cv, cfg.kda_conv_kernel,
                                       qkv, stream);
                if (fwd.debug) fwd.debug("dbg_conv_q", layer, s.conv_q, qkv);
                if (fwd.debug) fwd.debug("dbg_conv_v", layer, s.conv_v, qkv);
                k3k::l2_norm_heads_f32(s.conv_q, s.conv_q, head_dim, n_head,
                                       q_l2_scale, eps, stream);
                k3k::l2_norm_heads_f32(s.conv_k, s.conv_k, head_dim, n_head, 1.0f, eps, stream);
            }
            if (fwd.debug) fwd.debug("dbg_l2_q", layer, s.conv_q, qkv);

            // `nrm` rather than s.normed, and that is load-bearing rather than cosmetic:
            // proj_h_on takes the hoisted Q8_0 rows only when its `x` IS the pointer
            // hoist_act staged, and s.act_q8 holds TOKEN 0's row at offset 0. Passing
            // s.normed here would hand every token of the chunk token 0's activation —
            // the same shape of bug as one shared d_pos, and just as fast. At b == 0
            // nrm == s.normed and the hoist is taken exactly as before; at b > 0 the
            // pointers differ, the hoist is declined, and the ordinary path quantises
            // this token's own row. Bit-identical either way (same quantiser, same
            // bytes), only a launch cheaper at b == 0.
            if (!L.ssm_dt_bias.ok()) return false;
            if (!L.ssm_a.ok()) return false;
            if (!kda_pre) {
            if (!proj_h_on(l_dec, s.f_a_out, nrm, L.ssm_f_a, head_dim, H)) return false;
            if (!proj_on(l_dec, s.g_raw, s.f_a_out, L.ssm_f_b, qkv, head_dim)) return false;
            // The gate reads g_raw anyway, so it adds the bias on the way in — one
            // launch instead of two, bit-identical. Declining runs both as main does.
            if (!k3k::k3_kda_decay_gate_dt(s.decay_g, s.g_raw, w_dt, w_a, head_dim,
                                           n_head, cfg.kda_gate_lower_bound, l_dec.st)) {
                k3k::k3_add_f32(s.g_raw, s.g_raw, w_dt, qkv, l_dec.st);
                k3k::kda_decay_gate_f32(s.decay_g, s.g_raw, w_a,
                                        head_dim, n_head, cfg.kda_gate_lower_bound,
                                        l_dec.st);
            }
            }   // ---- end of the !kda_pre decay chain ----

            // ssm_beta is [n_heads, hidden] and stays REPLICATED: its suffix is shared
            // with Qwen's GDN, so one weight_plan entry governs both models. Project
            // all cfg.n_q_heads rows and read this rank's band -- 96 rows of a
            // 7168-wide Q8_0 matrix is 0.7 MB against 468 MB of projections, so the
            // duplicated work is not worth a rule that would change another model.
            // ssm_beta IS PROJECTED AT FULL WIDTH ON EVERY RANK AND READ AT A BAND.
            //
            // The tensor is [n_q_heads, hidden] and stays Replicate in weight_plan for
            // a good reason — its suffix is shared with Qwen's GDN, so a rule here
            // would silently change another model — but the justification recorded
            // beside that rule is about VRAM residency ("0.7 MB against 468 MB of
            // projections"), and residency is not what this costs. The GEMV runs all
            // 96 rows on all 8 ranks and the decode step then reads 12 of them, so
            // 87.5% of a 96 x 7168 f32 read is discarded: 2.297 MiB per layer, 158 MiB
            // per token per rank, in a projection nobody re-checked after #63 banded
            // the heads.
            //
            // The fix is a pointer, not a rule. Rows are contiguous, so this rank's
            // band starts at row hd_off and the launch narrows to n_head rows written
            // at s.beta_out + hd_off — the exact slice the decode step already reads.
            // Bit-identical: row hd_off+i computed off a shifted base is the same dot
            // product over the same K in the same order. The weight stays Replicate,
            // so no other model's plan moves.
            //
            // SPARKINFER_K3_BETA_BAND=0 restores the full-width projection.
            static const bool beta_band = [] {
                const char* e = std::getenv("SPARKINFER_K3_BETA_BAND");
                return !(e && e[0] == '0');
            }();
            const int beta_rows = (beta_band && hd_off + n_head <= cfg.n_q_heads)
                                ? n_head : cfg.n_q_heads;
            const int beta_off  = (beta_rows == cfg.n_q_heads) ? 0 : hd_off;
            {
                if (!L.ssm_beta.ok()) return false;
                // Row stride in BYTES, because a banded base has to skip whole rows of
                // whatever the weight is stored as. Only the two types this path can
                // actually meet are handled; anything else keeps the full projection
                // rather than guessing a stride.
                long row_bytes = 0;
                if (L.ssm_beta.type == 0)      row_bytes = (long)H * (long)sizeof(float);
                else if (L.ssm_beta.type == 8) row_bytes = (long)(H / 32) * 34;
                const int  n_rows = row_bytes > 0 ? beta_rows : cfg.n_q_heads;
                const int  r_off  = row_bytes > 0 ? beta_off  : 0;
                const void* wbase = (const char*)L.ssm_beta.data + (size_t)r_off * row_bytes;
                // Mirrors proj_h: take #94's hoisted activation when it holds THIS x
                // and the weight is Q8_0, else the ordinary path. ssm_beta is f32 today
                // so this resolves to k3_proj_f32, but writing it this way means the
                // band does not silently opt out of the hoist if the weight ever moves.
                //
                // `nrm`, NOT s.normed, for the same reason as ssm_f_a above: s.act_q8
                // holds token 0's row, so the hoist may only be taken when this token IS
                // token 0. beta is also the buffer the parent trap is about — it is
                // n_q_heads (96) wide, not qkv (1536), so it never gets a row stride at
                // all; the loop rewrites its single row per token and the decode step
                // below consumes it in the same iteration.
                const bool okb =
                    (hoisted_src == nrm && L.ssm_beta.type == 8)
                        ? k3k::k3_proj_q8act_f32(s.beta_out + r_off, s.act_q8, wbase,
                                                 L.ssm_beta.type, n_rows, H, l_beta.st)
                        : (ggml_qact_proj
                               ? k3k::k3_proj_ggml_f32(s.beta_out + r_off, nrm,
                                                       wbase, L.ssm_beta.type, n_rows, H,
                                                       l_beta.q8, l_beta.st)
                               : k3k::k3_proj_f32(s.beta_out + r_off, nrm, wbase,
                                                  L.ssm_beta.type, n_rows, H, l_beta.st));
                if (!okb) return false;
            }
            // beta is the delta-rule update rate: llama.cpp's build_kda_layer applies
            // a sigmoid to the projection before the scan consumes it. Omitting this
            // left beta unbounded (rms ~1.7 vs the correct ~0.75) and scaled the whole
            // KDA output — the layer-0 divergence vs llama that this restores.
            // The decode step below applies the sigmoid itself when the fused path
            // takes the work, so this launch exists only for the fallback.
            const bool beta_sig_fused = k3k::k3_kda_fuse_enabled();
            if (!beta_sig_fused)
                // Only the rows the projection above actually wrote — the rest of the
                // scratch is stale and never read.
                k3k::sigmoid_inplace_f32(s.beta_out + beta_off, beta_rows, l_beta.st);

            if (fwd.debug) fwd.debug("dbg_decay_g", layer, decay_b, qkv);
            if (fwd.debug) fwd.debug("dbg_beta", layer, s.beta_out, cfg.n_q_heads);
            // JOIN both lanes. The decode step below reads decay_g (lane 0) and
            // beta_out (lane 1) alongside the convs it read from the main stream,
            // so this is the point where the three chains have to meet.
            if (kda_fork) { dag_join(0); dag_join(1); }
            // The warp-per-column step declines at any shape or alignment it is not
            // written for, and the tiled kernel below is then exactly what main runs.
            // Both read the same f32 state, so a decline is a slower path and never a
            // different answer.
            if (!k3k::k3_kda_decode_step_ip(s.delta_out, st.delta_state[kda_ord],
                                            s.conv_q, s.conv_k, s.conv_v, decay_b,
                                            s.beta_out + hd_off,
                                            head_dim, n_head, beta_sig_fused, stream)) {
                // The fused step declined, so beta is still raw if the fold was on.
                if (beta_sig_fused)
                    k3k::sigmoid_inplace_f32(s.beta_out + beta_off, beta_rows, stream);
                k3k::kda_decode_step_f32(s.delta_out, st.delta_state[kda_ord],
                                         s.conv_q, s.conv_k, s.conv_v, decay_b,
                                         s.beta_out + hd_off,
                                         head_dim, n_head, stream);
            }
            if (fwd.debug) fwd.debug("dbg_delta_out", layer, s.delta_out, qkv);

            // Already computed above when the q/k/v/g fusion took the fast path, and
            // by the chunk arm whenever a_tok > 1 — so this is a single-token fallback
            // and `gproj_b` is s.g_proj_out with b == 0.
            if (!fused_qkvg && !proj_h(gproj_b, nrm, L.ssm_g, qkv, H)) return false;
            if (!L.ssm_norm.ok()) return false;
            k3k::kda_gate_out_f32(gate_b, s.delta_out, (const float*)L.ssm_norm.data,
                                  gproj_b, head_dim, n_head, eps, stream);
            if (fwd.debug) fwd.debug("dbg_gate_out", layer, gate_b, qkv);
            }   // ---- end of the per-token scan loop ----

            // ---- attn_output, ONCE FOR THE CHUNK ----
            //
            // The scan is what forced the token loop, and it has finished: gate_out now
            // holds all a_tok rows and attn_output is elementwise in the token index
            // again. So the last projection of the layer — N 7168, K 1536, another
            // 11.7 MB of Q8_0 per layer per rank — is read once for the chunk instead of
            // once per token. proj_b is the same helper the MLA branch uses for its own
            // attn_output and falls back to a per-token loop of `proj` when the batched
            // kernel declines; at a_tok == 1 it declines on its first line and this IS
            // the call this site has always made.
            //
            // s.attn_out's per-token pitch is H, which is what the driver's ONE
            // swap_partial_buffer to the chunk-wide base assumes and what the nb*hidden
            // attention reduce then covers — the same layout the MLA branch writes.
            //
            // The tile driver projects all its rows together straight into the
            // collective slot, so a marker hit means s.attn_out is already written for
            // this token and re-projecting would overwrite it from an aliased gate_out.
            // A marker miss retains the exact path above.
            if (!(pt_hit && pt_hit->kda_out_layer == layer) &&
                !proj_b(s.attn_out, H, s.gate_out, qkv, L.attn_output, H, qkv))
                return false;
            if (fwd.debug) fwd.debug("kda_out", layer, s.attn_out, H);
        } else {
            const int mla_ord = kimi_k3_mla_ordinal(cfg, layer);
            if (mla_ord < 0) return false;

            // --- FORK: the latent-KV branch and the attention gate ------------
            //
            // Both read `normed` and nothing else this layer produces, and neither
            // is consumed until much later: the cache row not until the decode
            // kernel, the gate not until the fold after it. On one stream they sit
            // between the query projections and their own consumers for no reason
            // other than that they were written there.
            //
            // The weights are checked BEFORE the fork. An early return with a lane
            // still outstanding would leave the capture unjoined, so every path
            // that can fail a lane is resolved while the stream is still linear.
            const bool mla_fork = dag && dag_mla;
            const int  mla_lanes = mla_fork ? (L.has_attn_gate ? 2 : 1) : 0;
            if (mla_fork) {
                if (!L.attn_kv_a_mqa.ok() || !L.attn_kv_a_norm.ok()) return false;
                if (L.has_attn_gate && !L.attn_gate.ok()) return false;
                dag_fork(mla_lanes);
            }
            const Lane l_kv   = mla_fork ? lane_of(0) : main_lane;
            const Lane l_gate = mla_fork ? lane_of(1) : main_lane;

            // EVERY PROJECTION AND NORM BELOW TAKES THE WHOLE CHUNK. At a_tok == 1 each
            // of these is the single call it has always been: proj_tok declines at
            // n_tok <= 1 and the fallback loop runs once with b == 0, and every
            // rms_norm's n_rows == 1 grid is dim3(g, 1) == dim3(g).
            if (L.has_q_lora) {
                if (!proj_hb(s.q_lora_out, qlora_row, s.normed, H, L.attn_q_a,
                             cfg.q_lora_rank, H))
                    return false;
                if (!L.attn_q_a_norm.ok()) return false;
                // In place, so both strides are the same buffer's own row pitch.
                k3k::rms_norm_f32(s.q_lora_out, s.q_lora_out,
                                  (const float*)L.attn_q_a_norm.data, cfg.q_lora_rank, eps,
                                  stream, a_tok, qlora_row, qlora_row);
                if (!proj_b(s.q_proj_out, qproj_row, s.q_lora_out, qlora_row, L.attn_q_b,
                            qh * cfg.key_length_mla, cfg.q_lora_rank))
                    return false;
            } else {
                if (!proj_hb(s.q_proj_out, qproj_row, s.normed, H, L.attn_q_dense,
                             qh * cfg.key_length_mla, H))
                    return false;
            }

            // The de-interleave into q_nope / q_pe is done by INDEXING inside the
            // absorb kernel when it takes the work. It only happens here — as the two
            // strided D2D copies main issues — when that kernel declines, which is
            // decided below and recorded so the call site stays a single decision.
            bool absorb_strided = false;

            if (!proj_hb_on(l_kv, s.kv_a_out, kva_row, s.normed, H, L.attn_kv_a_mqa,
                            cfg.key_length, H))
                return false;
            if (!L.attn_kv_a_norm.ok()) return false;
            // TWO DIFFERENT STRIDES, and this is the site the shared-stride bug would
            // have hit first: the norm READS kv_a_out, which is key_length (576) wide,
            // and WRITES kv_cmpr_normed, which is kv_lora_rank (512) wide. rms_norm_f32
            // takes them separately for exactly this reason.
            k3k::rms_norm_f32(s.kv_cmpr_normed, s.kv_a_out,
                              (const float*)L.attn_kv_a_norm.data, cfg.kv_lora_rank, eps,
                              l_kv.st, a_tok, kvcmpr_row, kva_row);
            if (fwd.debug) fwd.debug("dbg_kvcmpr", layer, s.kv_cmpr_normed, cfg.kv_lora_rank);

            // K-cache rows for this chunk's positions: concat(normed kv_cmpr, RAW k_pe).
            //
            // The row ADDRESS used to be computed here, on the host, from st.position —
            // which a captured graph freezes. One kernel that derives the row from d_pos
            // replaces the two memcpys; it moves the same bytes in the same order, so it
            // is bit-identical, and it is the reason replay 2 writes a different row from
            // replay 1 instead of overwriting replay 1's. The f16 twin narrows on the way
            // in and is otherwise the same kernel, same d_pos, same row layout.
            //
            // BATCHED, AND THE POSITION IS NOW INDEXED RATHER THAN DEREFERENCED: row b
            // stores at st.d_pos[b]. That is the one place in this phase that depends on
            // st.d_pos being the chunk's CONTIGUOUS position vector bound at token 0
            // (see the preconditions on kimi_k3_forward_layer_phase). Handed a single
            // shared position, all a_tok rows would write the same cache row — the bug
            // that is both wrong and faster.
            //
            // Storing all a_tok rows BEFORE any token attends is not a reordering of the
            // arithmetic: every attention kernel below walks *its own* d_pos + 1 rows, so
            // token b reads [0, pos_b] and cannot see the rows written for the tokens
            // after it. Causality here is the LENGTH, not the store order.
            if (st.kv_f16)
                k3k::k3_mla_kv_store_f16(st.mla_kv_cache[mla_ord], s.kv_cmpr_normed,
                                         s.kv_a_out, st.d_pos, cfg.kv_lora_rank,
                                         cfg.rope_dim, cfg.key_length, l_kv.st,
                                         a_tok, kvcmpr_row, kva_row);
            else
                k3k::k3_mla_kv_store_f32(st.mla_kv_cache[mla_ord], s.kv_cmpr_normed,
                                         s.kv_a_out, st.d_pos, cfg.kv_lora_rank,
                                         cfg.rope_dim, cfg.key_length, l_kv.st,
                                         a_tok, kvcmpr_row, kva_row);

            if (!L.attn_k_b.ok() || !L.attn_v_b.ok()) return false;
            absorb_strided = k3k::k3_mla_absorb_q_strided(
                s.absorbed_q, s.q_proj_out, (const float*)L.attn_k_b.data, qk_nope,
                cfg.kv_lora_rank, cfg.rope_dim, cfg.key_length_mla, qh, stream,
                a_tok, absq_row, qproj_row);
            if (!absorb_strided) {
                // THE DECLINE PATH STAYS PER-TOKEN, and s.q_nope / s.q_pe therefore stay
                // one row wide. One token is live at a time and this is straight-line on
                // one stream, so the two de-interleaves and the absorb that reads them
                // cannot interleave across b. Buying a row axis for the two scratch
                // buffers here would be VRAM spent on the path that already declined.
                for (int b = 0; b < a_tok; ++b) {
                    const float* qp_b = s.q_proj_out + (int64_t)b * qproj_row;
                    cudaMemcpy2DAsync(s.q_nope, (size_t)qk_nope * sizeof(float),
                                      qp_b,
                                      (size_t)cfg.key_length_mla * sizeof(float),
                                      (size_t)qk_nope * sizeof(float), qh,
                                      cudaMemcpyDeviceToDevice, stream);
                    cudaMemcpy2DAsync(s.q_pe, (size_t)cfg.rope_dim * sizeof(float),
                                      (const char*)qp_b +
                                          (size_t)qk_nope * sizeof(float),
                                      (size_t)cfg.key_length_mla * sizeof(float),
                                      (size_t)cfg.rope_dim * sizeof(float), qh,
                                      cudaMemcpyDeviceToDevice, stream);
                    k3k::mla_absorb_q_f32(s.absorbed_q + (int64_t)b * absq_row,
                                          s.q_nope, s.q_pe,
                                          (const float*)L.attn_k_b.data, qk_nope,
                                          cfg.kv_lora_rank, cfg.rope_dim, qh, stream);
                }
            }
            if (fwd.debug) fwd.debug("mla_absorb_q", layer, s.absorbed_q, qh * cfg.key_length);

            // JOIN the KV lane. The decode kernel reads the rows this store just wrote,
            // so this is the edge that makes the fork legal at all.
            if (mla_fork) dag_join(0);

            const float mla_scale = 1.0f / std::sqrt((float)cfg.key_length_mla);
            // ---- THE ONE PER-TOKEN LAUNCH LEFT IN A BATCHED MLA ATTN PHASE ----
            //
            // THE ATTENTION KERNEL ITSELF IS NOT BATCHED, AND THAT IS A BIT-IDENTITY
            // DECISION, NOT AN EFFORT ONE. mla_decode_attn_launch derives `splits` from
            // n_ctx, and the refinement is continuous in it (`by_len = n_ctx / min_slice`
            // plus the reach-fill relaxation). One grid for the chunk means ONE split
            // count for tokens sitting at a_tok different depths, and the split count is
            // the partition of an online-softmax reduction — changing it re-associates
            // the running max/exp-sum and moves the last bits. There is a pin
            // (k3_mla_set_split_pin) that would make one count legal for the chunk, but
            // pinning is exactly the arithmetic change this phase is not allowed to make.
            //
            // It is also the launch with the least to gain. Everything above streams a
            // WEIGHT per token and amortises; this one streams the KV CACHE, whose bytes
            // are proportional to each token's own prefix and do not amortise across
            // tokens at all.
            //
            //   * host length picks the launch plan only, and token b sits b positions
            //     deeper than token 0: st.position is TOKEN 0's position.
            //   * device length is what the kernel attends over, and is st.d_pos[b].
            // At a_tok == 1 this loop runs once with b == 0 and is the call this site
            // has always made.
            for (int b = 0; b < a_tok; ++b) {
                float* out_b       = s.mla_attn_out + (int64_t)b * mlaout_row;
                const float* q_b   = s.absorbed_q   + (int64_t)b * absq_row;
                const int* pos_b   = st.d_pos + b;
                const int  n_ctx_b = st.position + 1 + b;
                if (st.kv_f16)
                    k3k::mla_decode_attn_kvf16(out_b, q_b,
                                               st.mla_kv_cache[mla_ord],
                                               (const float*)L.attn_v_b.data, cfg.key_length,
                                               cfg.kv_lora_rank, cfg.value_length_mla, qh,
                                               n_ctx_b, pos_b, mla_scale, stream);
                else
                    k3k::mla_decode_attn_f32(out_b, q_b,
                                             st.mla_kv_cache[mla_ord],
                                             (const float*)L.attn_v_b.data, cfg.key_length,
                                             cfg.kv_lora_rank, cfg.value_length_mla, qh,
                                             n_ctx_b, pos_b, mla_scale, stream);
            }
            if (fwd.debug) fwd.debug("dbg_preattn", layer, s.mla_attn_out, qh * cfg.value_length_mla);

            if (L.has_attn_gate) {
                // Issued on its own lane so it runs UNDER the attention rather than
                // after it. Host issue order is irrelevant here -- every launch is
                // async and the lane's only dependency is the fork event, so the
                // gate is free to start the moment the fork retires. (At a_tok > 1
                // `dag` is off and this lane IS the main stream.)
                if (!proj_hb_on(l_gate, s.gate_proj_out, mlaout_row, s.normed, H,
                                L.attn_gate, qh * cfg.value_length_mla, H))
                    return false;
                if (fwd.debug) fwd.debug("dbg_gateproj", layer, s.gate_proj_out, qh * cfg.value_length_mla);
                if (mla_fork) dag_join(1);
                k3k::mla_gate_out_f32(s.mla_attn_out, s.mla_attn_out, s.gate_proj_out,
                                      (int64_t)qh * cfg.value_length_mla, stream,
                                      a_tok, mlaout_row);
                if (fwd.debug) fwd.debug("dbg_postgate", layer, s.mla_attn_out, qh * cfg.value_length_mla);
            }

            // The partial the driver reduces. s.attn_out is chunk-wide (allocated for
            // the cap, and repointed at the collective's buffer by the driver), and its
            // per-token pitch is H — the same layout the batched FfnPartial then READS
            // it at, and the same one nb*hidden reduces.
            if (!proj_b(s.attn_out, H, s.mla_attn_out, mlaout_row, L.attn_output, H,
                        qh * cfg.value_length_mla))
                return false;
            if (fwd.debug) fwd.debug("mla_out", layer, s.attn_out, H);
        }

        k3_profiler().stop(L.is_kda ? "attn_kda" : "attn_mla", stream);
    }

    // ---- phase 2a: attention residual and FFN norm. -------------------------
    // A prefill tile may run this for all tokens before phase 2b, making the common
    // normalized rows available to one batched routed-down projection.
    const K3PrefillTile* ffn_pt = (const K3PrefillTile*)fwd.prefill_tile;
    const bool ffn_tile_row = ffn_pt && fwd.prefill_tok >= 0 &&
                              fwd.prefill_tok < ffn_pt->n_live;
    const bool ffn_norm_prepared = ffn_tile_row &&
                                   ffn_pt->ffn_norm_layer == layer;
    const bool ffn_prepared = ffn_tile_row && ffn_pt->ffn_layer == layer;
    const bool experts_prepared = ffn_tile_row && ffn_pt->expert_layer == layer;
    const float* attn_src = (ffn_tile_row && ffn_pt->attn_layer == layer)
        ? ffn_pt->attn_out + (size_t)fwd.prefill_tok * H : s.attn_out;
    float* normed2_dst = (phase == K3LayerPhase::FfnPrepare && ffn_tile_row)
        ? ffn_pt->normed2 + (size_t)fwd.prefill_tok * H : s.normed2;
    const float* normed2_src = ffn_norm_prepared
        ? ffn_pt->normed2 + (size_t)fwd.prefill_tok * H : normed2_dst;
    const float* router_logits_src = ffn_prepared
        ? ffn_pt->router_logits + (size_t)fwd.prefill_tok * cfg.n_experts
        : s.router_logits;
    const float* router_w_src = ffn_prepared
        ? ffn_pt->router_w + (size_t)fwd.prefill_tok * cfg.top_k : s.router_w;
    const int* router_ids_src = ffn_prepared
        ? ffn_pt->router_ids + (size_t)fwd.prefill_tok * cfg.top_k : s.router_ids;

    if (do_ffn_prepare && !ffn_norm_prepared) {
        // --- combine: replace on a checkpoint layer, add otherwise. Uses hidden_in
        // (the RAW pre-mix value), not s.mixed — the reference's residual add is
        // against the unmixed prefix_sum, only the norm/attention input was mixed. ---
        // The residual combine and the pre-FFN mix used to be separate launches: an
        // add (or, banked, a D2D copy) materialised hidden_out, then the mix re-read
        // it. The mix kernels now take the unsummed pair and write hidden_out
        // themselves — same adds, same rounding, one launch fewer on all 93 layers.
        // SPARKINFER_K3_RES_FUSE=0 restores the split form on one binary.
        static const bool res_fuse = [] {
            const char* e = std::getenv("SPARKINFER_K3_RES_FUSE");
            return !(e && e[0] == '0');
        }();
        // BOTH RESIDUAL MIXES ARE BATCHED. This was ~186 nodes per token, the second
        // biggest launch pool after the projections.
        //
        // attn_res_mix_f32 now carries the token axis on BOTH kinds of buffer it reads:
        // out/cur/cur_b/sum_out are token-major activations at H, and the cross-layer
        // residual BANK -- per-token state, [n_embd, max_ckpt] for THIS token -- strides
        // by bank_row, the same pitch the per-token loop was adding by hand. s.res_scores
        // is sized for the cap and the kernels give row r its own (n_ckpt+1) slots; see
        // the allocation note, because sharing one row is the silent-at-short-context
        // failure this change had to avoid.
        //
        // st.n_ckpt is one value for the whole chunk (it is a function of the layer, and
        // the chunk driver restores it before each token), so it stays a scalar argument.
        //
        // At n_tok == 1 the launch is dim3(g, 1) == dim3(g) and every stride is
        // multiplied by row 0: the decode path is byte-for-byte what it always was.
        if (res_fuse && res_bs > 0 && st.n_ckpt > 0) {
            if (!L.ffn_res_score.ok()) return false;
            k3k::attn_res_mix_f32(s.mixed2, st.res_bank,
                                  banked ? attn_src : hidden_in,
                                  (const float*)L.ffn_res_score.data, H, st.n_ckpt, eps,
                                  stream, s.res_scores,
                                  banked ? nullptr : attn_src,
                                  hidden_out,
                                  n_tok, /*act_row_stride=*/H,
                                  /*bank_row_stride=*/bank_row);
        } else {
            if (banked) {
                // ONE COPY FOR THE WHOLE CHUNK. hidden_out and s.attn_out are both
                // token-major at exactly H, so nb rows are one contiguous range; a copy
                // has no arithmetic to preserve.
                cudaMemcpyAsync(hidden_out, attn_src,
                                (size_t)H * n_tok * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream);
            } else {
                // All three operands are token-major activations at H. This is the
                // residual combine, which is what k3_add_f32's row axis is FOR -- not a
                // broadcast bias add, which would read past a shared vector.
                k3k::k3_add_f32(hidden_out, hidden_in, attn_src, H, stream, n_tok, H);
            }

            // --- pre-FFN mix, no bank push. Batched on the same axis as the fused form
            // above: s.mixed2 and hidden_out are token-major at H, the bank strides by
            // bank_row, and s.res_scores gives each row its own softmax slots. ---
            if (res_bs > 0) {
                if (!L.ffn_res_score.ok()) return false;
                k3k::attn_res_mix_f32(s.mixed2, st.res_bank, hidden_out,
                                      (const float*)L.ffn_res_score.data, H, st.n_ckpt,
                                      eps, stream, s.res_scores,
                                      nullptr, nullptr, n_tok,
                                      /*act_row_stride=*/H,
                                      /*bank_row_stride=*/bank_row);
            } else {
                cudaMemcpyAsync(s.mixed2, hidden_out, (size_t)H * n_tok * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream);
            }
        }
        if (!L.ffn_norm.ok()) return false;
        // BATCHED. Both buffers are contiguous at H, so both strides are H -- which is
        // what the launcher substitutes for a 0 stride anyway, making the n_tok == 1
        // launch identical. The row is a new grid .y stacked on the existing span
        // spread: span_units and grid .x still come from the PER-ROW unit count, so
        // every row sees the same slicing and the same frozen-128 sum of squares.
        k3k::rms_norm_f32(normed2_dst, s.mixed2, (const float*)L.ffn_norm.data, H, eps,
                          stream, n_tok, H, H);
        if (fwd.debug) fwd.debug("ffn_norm", layer, normed2_dst, H);
    }

    // ---- phase 2b: the FFN / MoE itself. ------------------------------------
    //
    // SEPARATE FROM 2a ON PURPOSE. 2a is guarded by `do_ffn_prepare &&
    // !ffn_norm_prepared` because a prefill tile computes the residual and the norm for
    // every token in ONE FfnPrepare pass and must not redo them per token. 2b is guarded
    // by `do_ffn_p` because the FFN itself runs in FfnPartial and must run EXACTLY once
    // per token either way.
    //
    // Fusing the two guards is silent and total: FfnPrepare then runs the whole FFN on a
    // tile-aliased normed2, and the FfnPartial that follows finds ffn_norm_prepared true
    // and skips the FFN altogether. Nothing refuses; the layer simply does not happen.
    if (do_ffn_p) {
        // normed2 has up to four consumers below -- the router, routed_down, and both
        // shared-expert projections -- and every one of them used to re-quantise it.
        // At n_tok > 1 this stages all n_tok rows in one launch.
        //
        // When the tile already produced the logits there is nothing on this activation
        // left to hoist, and leaving `hoisted_src` set would send the projections below
        // down the pre-quantised path against a staging buffer nobody filled.
        if (!ffn_prepared) {
            hoist_act(normed2_src, H);
        } else {
            hoisted_src = nullptr;
        }

        k3_profiler().start(layer < cfg.leading_dense ? "ffn_dense" : "ffn_moe", stream);
        if (layer < cfg.leading_dense) {
            // s.dense_gate/up/situ are token-major at cfg.dense_ffn and s.ffn_out at H,
            // so every ldy/stride here is the buffer's own single-token width.
            if (!proj_hb(s.dense_gate, cfg.dense_ffn, normed2_src, H, L.ffn_gate,
                         cfg.dense_ffn, H)) return false;
            if (!proj_hb(s.dense_up, cfg.dense_ffn, normed2_src, H, L.ffn_up,
                         cfg.dense_ffn, H)) return false;
            if (fwd.debug) fwd.debug("dbg_dense_gate", layer, s.dense_gate, cfg.dense_ffn);
            if (fwd.debug) fwd.debug("dbg_dense_up", layer, s.dense_up, cfg.dense_ffn);
            // Elementwise, so the row is purely a grid .y and there is no reduction to
            // re-partition.
            k3k::situ_f32(s.dense_situ, s.dense_gate, s.dense_up, cfg.dense_ffn,
                         cfg.situ_beta, cfg.situ_linear_beta, stream,
                         n_tok, cfg.dense_ffn);
            if (fwd.debug) fwd.debug("dbg_dense_situ", layer, s.dense_situ, cfg.dense_ffn);
            // s.ffn_out is read back in FfnFinish, one phase and (in the chunk driver)
            // one collective later, so it MUST be per-token. It was not, and the leading
            // dense layer's FFN output was therefore the LAST token of the chunk's for
            // every token of it -- see the note on this in the FfnFinish residual add.
            if (!proj_b(s.ffn_out, H, s.dense_situ, cfg.dense_ffn, L.ffn_down, H,
                        cfg.dense_ffn)) return false;
        } else {
            // --- FORK: routed_down and the whole shared expert ----------------
            //
            // Both read `normed2`. routed_down does NOT depend on the router --
            // it is the dispatch's activation, not its selection -- so main's
            // order makes a [3584 x 7168] projection wait behind a [896 x 7168]
            // one and a top-k it has no relationship with. The shared expert is
            // four more launches that nothing reads until the collective.
            //
            // moe_out and shexp_out are DISJOINT VIEWS of one allocation (see
            // kimi_k3_forward_alloc_scratch): the dispatch writes [0, expert_latent)
            // from the main stream while the lane writes [expert_latent, +H). One
            // buffer, two ranges, no overlap -- which is what makes it safe to have
            // two streams writing what one collective then reduces.
            // Every tensor touched between the fork and the joins is checked HERE, for
            // the same reason as the KDA site: an early return with a lane outstanding
            // leaves the capture unjoined, and "EndCapture failed" is a much worse
            // report than "this weight is missing". Declining to fork is always safe —
            // the checks below still run and still return false, just on a linear stream.
            const bool moe_fork = dag && dag_moe &&
                L.ffn_routed_down.ok() && L.ffn_gate_inp.ok() && L.exp_probs_b.ok() &&
                L.ffn_gate_exps.ok() && L.ffn_up_exps.ok() && L.ffn_down_exps.ok() &&
                (!L.has_shared_experts ||
                 (L.ffn_gate_shexp.ok() && L.ffn_up_shexp.ok() && L.ffn_down_shexp.ok()));
            if (moe_fork) dag_fork(2);
            const Lane l_rd  = moe_fork ? lane_of(0) : main_lane;   // routed_down
            const Lane l_shx = moe_fork ? lane_of(1) : main_lane;   // shared expert

            // BATCHED, and the router is the site where losing that is silent. The
            // k3_moe_router_fast call below already takes n_tokens = n_tok and strides
            // s.router_logits by cfg.n_experts, so a single-row fill here leaves rows
            // 1..n_tok-1 holding whatever the previous layer left and the selection
            // picks experts for them out of stale memory -- wrong output at a FASTER
            // time, because 91 of 92 rows of this projection never ran.
            // At n_tok == 1 proj_hb's loop runs once and this is the single call the
            // tile driver has always made; `ffn_prepared` still short-circuits it when
            // the tile already computed the logits.
            if (!ffn_prepared &&
                !proj_hb(s.router_logits, cfg.n_experts, normed2_src, H, L.ffn_gate_inp,
                         cfg.n_experts, H))
                return false;
            if (fwd.debug) fwd.debug("dbg_router_logits", layer, router_logits_src,
                                     cfg.n_experts);
            if (!L.exp_probs_b.ok()) return false;
            // Shared-memory selection first; it declines to the original below on any
            // shape it does not handle, and is bit-identical where it does.
            //
            // THE TOKEN COUNT WAS ALREADY A PARAMETER. Both kernels take `tok =
            // blockIdx.x` and stride logits by n_expert and out_w/out_ids by top_k, so
            // n_tokens = n_tok is the whole change and the three buffers only had to be
            // that wide. NOTE what it costs: n_tokens != 1 makes k3_moe_router_fast
            // DECLINE its register fast path (k3_moe_router_fast.cu:451 gates on
            // n_tokens == 1, which is what justifies its __launch_bounds__(,1)) and take
            // moe_router_shared_kernel instead. That kernel is the one the register path
            // was written to replace and is documented bit-identical to it -- same
            // sigmoid, same biased selection, same ascending normalisation sum -- so the
            // selection and the weights are unchanged; only the launch shape moves. At
            // n_tok == 1 the literal 1 is restored and the register path is taken exactly
            // as it is today.
            //
            // SKIPPED WHOLESALE when the tile already selected: the selection reads
            // s.router_logits, which `ffn_prepared` means was never written on this
            // stream, and it WRITES s.router_w/s.router_ids, which would then overwrite
            // the tile's selection with one derived from stale logits.
            if (!ffn_prepared) {
                if (!k3k::k3_moe_router_fast(s.router_w, s.router_ids, s.router_logits,
                                             (const float*)L.exp_probs_b.data, cfg.n_experts,
                                             cfg.top_k, /*n_tokens=*/n_tok, /*norm_w=*/true,
                                             /*w_scale=*/1.0f, stream))
                    k3k::moe_router_noaux_tc_f32(s.router_w, s.router_ids, s.router_logits,
                                                 (const float*)L.exp_probs_b.data,
                                                 cfg.n_experts, cfg.top_k,
                                                 /*n_tokens=*/n_tok, /*norm_w=*/true,
                                                 /*w_scale=*/1.0f, stream);
            }
            if (fwd.debug) fwd.debug("dbg_router_w", layer, router_w_src, cfg.top_k);

            // routed_down feeds the dispatch UNNORMALISED — routed_norm (if present)
            // normalises the dispatch's OUTPUT, not this. See build_latent_moe in the
            // reference: build_moe_ffn runs first, then "if (layer.ffn_routed_norm)
            // moe_out = build_norm(moe_out, ...)", and only after that does
            // ffn_routed_up run. Getting this backwards (norm between routed_down and
            // the dispatch) was an earlier version's bug, same class as the ssm_norm fix.
            //
            // BATCHED. At n_tok == 1 proj_tok declines immediately and the fallback loop
            // runs proj_h_on(l_rd, ...) once -- the lane is preserved, because `l_rd` is
            // main_lane whenever the fork did not happen and the fork is what n_tok > 1
            // turns off.
            const float* routed_down_src = s.routed_down_out;
            if (ffn_prepared) {
                routed_down_src = ffn_pt->routed_down +
                    (size_t)fwd.prefill_tok * cfg.expert_latent;
            } else if (n_tok > 1) {
                if (!proj_hb(s.routed_down_out, cfg.expert_latent, normed2_src, H,
                             L.ffn_routed_down, cfg.expert_latent, H))
                    return false;
            } else if (!proj_h_on(l_rd, s.routed_down_out, normed2_src, L.ffn_routed_down,
                                  cfg.expert_latent, H)) {
                return false;
            }

            if (!L.ffn_gate_exps.ok() || !L.ffn_up_exps.ok() || !L.ffn_down_exps.ok())
                return false;
            // JOIN routed_down: the dispatch reads it as its activation.
            if (moe_fork) dag_join(0);
            // THE EXPERT BAND. The router above ran on the replicated ffn_gate_inp,
            // so every rank picked the SAME top_k global ids; this rank evaluates only
            // the ones whose weights it holds, leaving a partial sum in s.moe_out that
            // the driver reduces at expert_latent before phase 3. At tp_size 1 the
            // band covers every expert and the call is unchanged.
            const int expert_begin  = fwd.w->shard.expert_band.offset;
            const int n_local_exp   = fwd.w->shard.tp_size > 1
                                        ? fwd.w->shard.expert_band.extent : 0;
            // UNDER 2-D MoE THE BAND ABOVE IS WIDER AND THIS WIDTH IS NARROWER, and
            // the kernels need no other change: they already index the packed weights
            // as (e * ffn + j) and (e * latent + o) with `ffn` as the stride, so
            // handing them the rank's 768-row band addresses the rank's own buffer
            // exactly as 3072 addressed the whole one. The expert-band test is
            // likewise unchanged — the group is still a CONTIGUOUS id range, just a
            // longer one. What changes is only which bytes the loader packed.
            const int moe_ffn_rank = k3_moe_ffn_local(fwd, cfg);
            // THE EXPERT DISPATCH, BATCHED BY AN EXPERT-MAJOR REGROUP.
            //
            // ~18% of GPU time, and the last phase of this FFN that was still per
            // token. It could not take a token axis the way the norms and the six
            // projections did, because `router_ids` DIFFERS PER TOKEN: the kernel is
            // "for each of my top_k selections, walk that expert's weights", so B
            // tokens with B different selections read B different sets of expert
            // rows and a token axis buys the LAUNCH COUNT and not one byte of weight
            // traffic. What pays is the inversion -- sort the B*top_k (token, slot)
            // selections by expert id and run each expert ONCE over its run of rows
            // -- and that is a different kernel, which is why it is a different front
            // door rather than an argument. See "4b. THE TOKEN-BATCHED DISPATCH" in
            // k3_kernels.cu for the shape and the bit-identity argument.
            //
            // THE ARITHMETIC THAT SIZES THE PRIZE: an expert sees B*top_k/n_experts =
            // B/56 rows. At the DEFAULT chunk of 16 that is 0.29 and the regroup is
            // worth only its launch count; it is the 256-512 range where each expert
            // sees 5-9 rows and the gate/up weight traffic drops ~9x. Raising
            // SPARKINFER_K3_PREFILL_CHUNK is what turns this on properly.
            //
            // DECLINING IS NOT AN ERROR. The batched door returns false for the q8k
            // reference path, for a quant type it has no kernel for, for a chunk under
            // the floor, and for SPARKINFER_K3_MOE_BATCH=0 -- and every one of those
            // means "run the loop below", which is the shipped path.
            //
            // n_tok == 1 NEVER REACHES IT: the guard is n_tok > 1, so decode issues
            // exactly the call it issued before, with the same arguments.
            //
            // `experts_prepared` means a prefill tile ALREADY ran this layer's dispatch
            // for this token into s.moe_out. Both forms below are then skipped -- redoing
            // it is what #136 added the flag to avoid, and dropping the flag turns the
            // tile's reuse back into a recompute silently, at no correctness cost and
            // full expert cost.
            bool moe_done = experts_prepared;
            if (!moe_done && n_tok > 1 && !ggml_qact_moe) {
                moe_done = k3k::moe_expert_ffn_batch_f32_by_type(
                    s.moe_out, moe_row,
                    s.moe_scratch, (int64_t)cfg.top_k * moe_ffn_rank,
                    routed_down_src, cfg.expert_latent,
                    router_ids_src, router_w_src,
                    L.ffn_gate_exps.data, L.ffn_up_exps.data, L.ffn_down_exps.data,
                    cfg.expert_latent, moe_ffn_rank, cfg.top_k, n_tok, cfg.situ_beta,
                    cfg.situ_linear_beta, L.ffn_gate_exps.type, stream,
                    expert_begin, n_local_exp, cfg.n_experts,
                    s.moe_batch_ws, s.moe_batch_ws_bytes);
            }
            // s.moe_q8 (the q8k staging) and the slab-zero invariant both hold per row:
            // the loop is straight-line on ONE stream, so token b's dispatch completes
            // before token b+1's begins, and each token owns its own moe_scratch slab.
            for (int b = 0; !moe_done && b < n_tok; ++b) {
                const bool moe_ok = k3k::moe_expert_ffn_f32_by_type(
                    s.moe_out + (int64_t)b * moe_row,
                    s.moe_scratch + (int64_t)b * cfg.top_k * moe_ffn_rank,
                    routed_down_src + (int64_t)b * cfg.expert_latent,
                    router_ids_src + (int64_t)b * cfg.top_k,
                    router_w_src + (int64_t)b * cfg.top_k,
                    L.ffn_gate_exps.data, L.ffn_up_exps.data, L.ffn_down_exps.data,
                    cfg.expert_latent, moe_ffn_rank, cfg.top_k, cfg.situ_beta,
                    cfg.situ_linear_beta, L.ffn_gate_exps.type, stream,
                    expert_begin, n_local_exp, ggml_qact_moe ? s.moe_q8 : nullptr);
                if (!moe_ok) return false;
            }
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
                // THE SHARED EXPERT'S BUFFERS ARE PADDED, AND THAT IS THE INTERESTING
                // PART OF BATCHING IT.
                //
                //   * s.dense_gate/up/situ are allocated at cfg.dense_ffn per token but
                //     the shared expert only fills shexp_band of each row. So the ldy
                //     and the activation stride are dense_ffn while N and K are
                //     shexp_band -- which is exactly what k3_proj_q8act_tok_f32's
                //     separate ldy/ld_act_blocks are for -- and the staging of
                //     dense_situ falls to quant_rows' per-row path, because a flat
                //     quantise has no row index to apply a padded stride to.
                //   * s.shexp_out's rows are moe_row apart, NOT H apart: it is the
                //     second view of the fused partial. Passing H as its ldy would
                //     write token b's shared-expert output over token b's own expert
                //     accumulator.
                //
                // The whole group is skipped when a tile already ran it for this token:
                // it WRITES s.shexp_out, which the driver reduces the moment this phase
                // returns, so recomputing it here is not merely wasted work -- it would
                // rewrite a partial the tile already placed.
                if (!ffn_prepared) {
                    if (n_tok > 1) {
                        if (!proj_hb(s.dense_gate, cfg.dense_ffn, normed2_src, H,
                                     L.ffn_gate_shexp, shexp_band, H)) return false;
                        if (!proj_hb(s.dense_up, cfg.dense_ffn, normed2_src, H,
                                     L.ffn_up_shexp, shexp_band, H)) return false;
                    } else {
                        if (!proj_h_on(l_shx, s.dense_gate, normed2_src, L.ffn_gate_shexp,
                                       shexp_band, H))
                            return false;
                        if (!proj_h_on(l_shx, s.dense_up, normed2_src, L.ffn_up_shexp,
                                       shexp_band, H))
                            return false;
                    }
                    k3k::situ_f32(s.dense_situ, s.dense_gate, s.dense_up, shexp_band,
                                 cfg.situ_beta, cfg.situ_linear_beta, l_shx.st,
                                 n_tok, cfg.dense_ffn);
                    if (n_tok > 1) {
                        if (!proj_b(s.shexp_out, moe_row, s.dense_situ, cfg.dense_ffn,
                                    L.ffn_down_shexp, H, shexp_band)) return false;
                    } else if (!proj_on(l_shx, s.shexp_out, s.dense_situ,
                                        L.ffn_down_shexp, H, shexp_band)) {
                        return false;
                    }
                }
                if (fwd.debug) fwd.debug("dbg_shexp_partial", layer, s.shexp_out, H);
            } else {
                // The fused buffer is reduced whole, so a layer without a shared
                // expert must still present a well-defined summand there.
                //
                // PER-TOKEN because the rows are moe_row apart while the range zeroed is
                // H: this is a strided clear, not a contiguous one, and a single memset
                // of n_tok*H would wipe the expert accumulators sitting between the
                // rows. One memset node per token on a layer that has no shared expert
                // at all -- K3 ships n_shared = 2, so this branch is not on the scored
                // path.
                if (!ffn_prepared)
                    for (int b = 0; b < n_tok; ++b)
                        cudaMemsetAsync(s.shexp_out + (int64_t)b * moe_row, 0,
                                        (size_t)H * sizeof(float), l_shx.st);
            }
            // JOIN the shared expert. Nothing on the main stream reads shexp_out,
            // but the DRIVER reduces it the moment this phase returns, and the
            // collective is enqueued on the main stream — so the join has to happen
            // here, inside the phase, or the reduce races the lane that fills it.
            // It is also what keeps the capture joined: an outstanding lane at
            // cudaStreamEndCapture fails the capture outright.
            if (moe_fork) dag_join(1);
        }
    }

    // ---- phase 3: everything downstream of the FFN collective. Every weight
    // here is Replicate and reads the ALREADY-REDUCED value — routed_norm is an
    // rms_norm, so the sum cannot be deferred past it. ----
    if (do_ffn_f) {
        // Set when the shared expert is present, i.e. when the layer's tail is the
        // adjacent PAIR of adds that k3_add3_f32 fuses.
        bool fused_tail = false;
        if (layer >= cfg.leading_dense) {
            // Post-collective: every rank must now hold the SAME complete expert sum,
            // equal to what tp=1 computed. If this tag differs, the dispatch or the
            // reduce is wrong; if it matches and ffn_out still differs, the fault is
            // downstream in routed_norm / routed_up / shexp.
            if (fwd.debug) fwd.debug("dbg_moe_reduced", layer, s.moe_out, cfg.expert_latent);
            // Out of place into moe_normed: same kernel, same reduction, same
            // values — but out != x is what lets rms_norm_f32 grid-spread its
            // output sweep (RMSG), which the in-place form must decline.
            const float* moe_src = s.moe_out;
            // THE TWO STRIDES ARE DIFFERENT HERE, which is the reason rms_norm_f32 takes
            // two. The source is the latent prefix of the FUSED partial, so its rows are
            // moe_row apart; the destination is moe_normed's own buffer, whose rows are
            // expert_latent apart. One shared stride would read every row but the first
            // from the wrong place. At n_tok == 1 the launcher substitutes n for a 0
            // stride and both of these ARE n, so the launch is unchanged.
            int64_t moe_src_row = moe_row;
            if (L.has_routed_norm) {
                if (!L.ffn_routed_norm.ok()) return false;
                k3k::rms_norm_f32(s.moe_normed, s.moe_out,
                                  (const float*)L.ffn_routed_norm.data, cfg.expert_latent,
                                  eps, stream, n_tok, cfg.expert_latent, moe_row);
                moe_src = s.moe_normed;
                moe_src_row = cfg.expert_latent;
            }

            if (fwd.debug) fwd.debug("dbg_moe_normed", layer, moe_src, cfg.expert_latent);
            if (!proj_b(s.ffn_out, H, moe_src, moe_src_row, L.ffn_routed_up, H,
                        cfg.expert_latent))
                return false;
            if (fwd.debug) fwd.debug("dbg_routed_up", layer, s.ffn_out, H);

            // The shared expert was computed in phase FfnPartial and its partial has
            // been reduced along with the expert accumulator, so what is left here is
            // the add. Both contributions are at hidden width by this point:
            // ffn_routed_up lifted the latent, shexp_out was already hidden-wide.
            if (L.has_shared_experts) {
                if (fwd.debug) fwd.debug("dbg_shexp", layer, s.shexp_out, H);
                // The shared-expert fold is issued BELOW, fused with the residual add
                // that consumes it — see k3_add3.cu. `fused_tail` records that decision
                // so the fallback path stays a single straight-line pair.
                fused_tail = true;
            }
        }
        k3_profiler().stop(layer < cfg.leading_dense ? "ffn_dense" : "ffn_moe", stream);

        // FFN residual is ALWAYS an add, never a replace. When the shared expert is
        // present its fold and this add are adjacent and share an operand, so they go
        // out as one launch; k3_add3_f32 declines to the original pair.
        //
        // THE FUSED TAIL CANNOT TAKE THE ROW AXIS, and the reason is a stride, not a
        // dependency. k3_add3_f32 has ONE row_stride for all five of its pointers, which
        // is right for five same-shaped activations. Here four of them (hidden_out,
        // s.ffn_out twice, hidden_out) stride by H and the fifth, s.shexp_out, strides
        // by moe_row -- it is the second view of the fused partial. So the fold is a
        // per-token loop, which costs exactly what it costs today (one node per token)
        // and loses only the amortisation. Giving k3_add3_f32 a separate stride for `b`
        // is a small additive kernel change and the obvious follow-up; it is not made
        // here because it cannot be compiled or measured in this change.
        //
        // The UNFUSED tail's second add IS uniform (hidden_out and s.ffn_out both at H)
        // and does take the axis.
        //
        // k3_add3_f32 declines as a WHOLE (SPARKINFER_K3_ADD3=0 / a null operand), never
        // for one token and not another, so `fused` below is a single decision for the
        // chunk and the split pair stays a straight-line fallback exactly as it is at
        // n_tok == 1.
        bool fused = false;
        if (fused_tail) {
            fused = true;
            for (int b = 0; b < n_tok && fused; ++b)
                fused = k3k::k3_add3_f32(hidden_out + (int64_t)b * H,
                                         s.ffn_out + (int64_t)b * H,
                                         s.ffn_out + (int64_t)b * H,
                                         s.shexp_out + (int64_t)b * moe_row,
                                         hidden_out + (int64_t)b * H, H, stream);
        }
        if (fused) {
            if (fwd.debug) fwd.debug("ffn_out", layer, s.ffn_out, H);
        } else {
            if (fused_tail)
                for (int b = 0; b < n_tok; ++b)
                    k3k::k3_add_f32(s.ffn_out + (int64_t)b * H,
                                    s.ffn_out + (int64_t)b * H,
                                    s.shexp_out + (int64_t)b * moe_row, H, stream);
            if (fwd.debug) fwd.debug("ffn_out", layer, s.ffn_out, H);
            // BATCHED: hidden_out and s.ffn_out are both token-major at H.
            //
            // s.ffn_out is per-token now. IT WAS NOT, and on the leading dense layer
            // that was a live wrong answer in the chunk driver: FfnPartial wrote one
            // shared s.ffn_out for every token of the chunk and FfnFinish then added the
            // LAST token's dense FFN output into every token's residual. Silent, and
            // only on the ingestion path.
            k3k::k3_add_f32(hidden_out, hidden_out, s.ffn_out, H, stream, n_tok, H);
        }
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
        // Device mirror advances with the host one on every path, not just the TP one.
        // A single-GPU run that advanced only the host would desync the moment anything
        // read the position from device memory.
        if (fwd.state->d_pos) k3k::k3_bump_pos(fwd.state->d_pos, stream);
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

    for (auto& st : p.stages) {
        ++st.state.position;
        // Each stage owns a different device, so the bump has to be issued with that
        // device current — d_pos lives in the stage's memory, not the caller's.
        if (st.state.d_pos) {
            if (cudaSetDevice(st.device) != cudaSuccess) return false;
            k3k::k3_bump_pos(st.state.d_pos, st.fwd.stream);
        }
    }
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
