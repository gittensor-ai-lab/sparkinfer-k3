#include "sparkinfer/models/kimi_k3.h"

#include "sparkinfer/kernels/kimi_k3.h"
#include "sparkinfer/kernels/kimi_k3_fast.h"
#include "sparkinfer/models/kimi_k3_gguf_manifest.h"
#include "sparkinfer/tp/weight_residency.h"   // plan_tensor_residency for the sharded loader

#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <mutex>
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
    // Rank threads issue concurrently; slots[key] inserts and may rehash. Without a
    // lock, a concurrent insert can invalidate Slot& held by another rank and hand
    // back a destroyed cudaEvent ("invalid resource handle") mid-forward.
    std::mutex mu;
    bool on = false;
    K3Profiler() { const char* e = std::getenv("SPARKINFER_K3_PROFILE"); on = e && e[0] == '1'; }

    static std::string key_for(const std::string& tag) {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) dev = -1;
        return tag + "@" + std::to_string(dev);
    }
    void start(const std::string& tag, cudaStream_t st) {
        if (!on) return;
        std::lock_guard<std::mutex> lock(mu);
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
        std::lock_guard<std::mutex> lock(mu);
        auto it = slots.find(key_for(tag));
        if (it != slots.end() && it->second.b) cudaEventRecord(it->second.b, st);
    }
    void report() {
        if (!on) return;
        std::lock_guard<std::mutex> lock(mu);
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
    // The HOISTED activation, quantised once and consumed by several projections. A
    // SEPARATE buffer from proj_q8 on purpose: every un-hoisted k3_proj_ggml_f32 call
    // quantises into proj_q8, so sharing it would let any projection landing between a
    // hoist and its consumers silently overwrite the bytes. That aliasing is the same
    // failure the cached quantise-once attempt shipped as top1 0.0.
    void* act_q8 = nullptr;
    void* moe_q8 = nullptr;            // optional llama CPU-compat block_q8_K scratch

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

    std::vector<void*> owned;
};

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
    // THE SCRATCH IS THE RANK'S FFN WIDTH, NOT THE MODEL'S. Under 2-D MoE this rank
    // evaluates only its band of each expert's intermediate, so top_k * band is all
    // that is ever written — and sizing it from cfg would leave 3/4 of the buffer
    // untouched while the gate/up memset below still paid to clear it.
    const int moe_ffn_local = k3_moe_ffn_local(fwd, cfg);
    ok &= alloc_f(s.moe_scratch, (size_t)cfg.top_k * moe_ffn_local);
    // Establish the "foreign expert slots read as zero" invariant ONCE here, instead of
    // re-establishing it with a cudaMemsetAsync on every MoE layer. A rank's expert band
    // is fixed for the run and moe_gate_up writes only owned slots, so a foreign slot is
    // written by nobody and stays zero from this memset onward.
    if (ok && s.moe_scratch)
        cudaMemset(s.moe_scratch, 0,
                   (size_t)cfg.top_k * moe_ffn_local * sizeof(float));
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
    // Sized like proj_q8: dense_ffn is the widest K any projection contracts over, so
    // one allocation covers every activation this hoists.
    ok &= alloc_bytes(s.act_q8, k3k::k3_q8_0_bytes(cfg.dense_ffn));
    ok &= alloc_bytes(s.moe_q8,
                      k3k::k3_moe_q8_k_bytes(cfg.expert_latent, moe_ffn_local, cfg.top_k));

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
    const bool dag = dag_all && s.lanes_ok && !fwd.debug;

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

    auto proj_on = [&](Lane ln, float* y, const float* x, const KimiK3Tensor& W,
                       int N, int K) {
        if (!W.ok()) return false;
        if (ggml_qact_proj) {
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
        // normed feeds ssm_f_a and ssm_beta on a KDA layer, and q_a/q_dense, kv_a and
        // attn_gate on an MLA one. The fused qkvg group keeps its own quantisation --
        // it is one launch that already amortises it across four tensors.
        hoist_act(s.normed, H);

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
            const bool kda_fork = dag && dag_kda &&
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
            const bool fused_qkvg = fusable &&
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
                                           L.attn_q.type, qkv, H, stream));
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
                    s.qkv_q, s.qkv_k, s.qkv_v, w_cq, w_ck, w_cv,
                    cfg.kda_conv_kernel, head_dim, n_head, q_l2_scale, eps, stream);

            if (!kda_fused) {
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
                k3k::l2_norm_heads_f32(s.conv_q, s.conv_q, head_dim, n_head,
                                       q_l2_scale, eps, stream);
                k3k::l2_norm_heads_f32(s.conv_k, s.conv_k, head_dim, n_head, 1.0f, eps, stream);
            }
            if (fwd.debug) fwd.debug("dbg_l2_q", layer, s.conv_q, qkv);

            if (!proj_h_on(l_dec, s.f_a_out, s.normed, L.ssm_f_a, head_dim, H)) return false;
            if (!proj_on(l_dec, s.g_raw, s.f_a_out, L.ssm_f_b, qkv, head_dim)) return false;
            if (!L.ssm_dt_bias.ok()) return false;
            if (!L.ssm_a.ok()) return false;
            // The gate reads g_raw anyway, so it adds the bias on the way in — one
            // launch instead of two, bit-identical. Declining runs both as main does.
            if (!k3k::k3_kda_decay_gate_dt(s.decay_g, s.g_raw, w_dt, w_a, head_dim,
                                           n_head, cfg.kda_gate_lower_bound, l_dec.st)) {
                k3k::k3_add_f32(s.g_raw, s.g_raw, w_dt, qkv, l_dec.st);
                k3k::kda_decay_gate_f32(s.decay_g, s.g_raw, w_a,
                                        head_dim, n_head, cfg.kda_gate_lower_bound,
                                        l_dec.st);
            }

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
                const bool okb =
                    (hoisted_src == s.normed && L.ssm_beta.type == 8)
                        ? k3k::k3_proj_q8act_f32(s.beta_out + r_off, s.act_q8, wbase,
                                                 L.ssm_beta.type, n_rows, H, l_beta.st)
                        : (ggml_qact_proj
                               ? k3k::k3_proj_ggml_f32(s.beta_out + r_off, s.normed,
                                                       wbase, L.ssm_beta.type, n_rows, H,
                                                       l_beta.q8, l_beta.st)
                               : k3k::k3_proj_f32(s.beta_out + r_off, s.normed, wbase,
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

            if (fwd.debug) fwd.debug("dbg_decay_g", layer, s.decay_g, qkv);
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
                                            s.conv_q, s.conv_k, s.conv_v, s.decay_g,
                                            s.beta_out + hd_off,
                                            head_dim, n_head, beta_sig_fused, stream)) {
                // The fused step declined, so beta is still raw if the fold was on.
                if (beta_sig_fused)
                    k3k::sigmoid_inplace_f32(s.beta_out + beta_off, beta_rows, stream);
                k3k::kda_decode_step_f32(s.delta_out, st.delta_state[kda_ord],
                                         s.conv_q, s.conv_k, s.conv_v, s.decay_g,
                                         s.beta_out + hd_off,
                                         head_dim, n_head, stream);
            }
            if (fwd.debug) fwd.debug("dbg_delta_out", layer, s.delta_out, qkv);

            // Already computed above when the q/k/v/g fusion took the fast path.
            if (!fused_qkvg && !proj_h(s.g_proj_out, s.normed, L.ssm_g, qkv, H)) return false;
            if (!L.ssm_norm.ok()) return false;
            k3k::kda_gate_out_f32(s.gate_out, s.delta_out, (const float*)L.ssm_norm.data,
                                  s.g_proj_out, head_dim, n_head, eps, stream);
            if (fwd.debug) fwd.debug("dbg_gate_out", layer, s.gate_out, qkv);

            if (!proj(s.attn_out, s.gate_out, L.attn_output, H, qkv)) return false;
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

            if (L.has_q_lora) {
                if (!proj_h(s.q_lora_out, s.normed, L.attn_q_a, cfg.q_lora_rank, H)) return false;
                if (!L.attn_q_a_norm.ok()) return false;
                k3k::rms_norm_f32(s.q_lora_out, s.q_lora_out,
                                  (const float*)L.attn_q_a_norm.data, cfg.q_lora_rank, eps,
                                  stream);
                if (!proj(s.q_proj_out, s.q_lora_out, L.attn_q_b, qh * cfg.key_length_mla,
                         cfg.q_lora_rank))
                    return false;
            } else {
                if (!proj_h(s.q_proj_out, s.normed, L.attn_q_dense, qh * cfg.key_length_mla, H))
                    return false;
            }

            // The de-interleave into q_nope / q_pe is done by INDEXING inside the
            // absorb kernel when it takes the work. It only happens here — as the two
            // strided D2D copies main issues — when that kernel declines, which is
            // decided below and recorded so the call site stays a single decision.
            bool absorb_strided = false;

            if (!proj_h_on(l_kv, s.kv_a_out, s.normed, L.attn_kv_a_mqa, cfg.key_length, H))
                return false;
            if (!L.attn_kv_a_norm.ok()) return false;
            k3k::rms_norm_f32(s.kv_cmpr_normed, s.kv_a_out,
                              (const float*)L.attn_kv_a_norm.data, cfg.kv_lora_rank, eps,
                              l_kv.st);
            if (fwd.debug) fwd.debug("dbg_kvcmpr", layer, s.kv_cmpr_normed, cfg.kv_lora_rank);

            // K-cache row for this position: concat(normed kv_cmpr, RAW k_pe).
            //
            // The row ADDRESS used to be computed here, on the host, from st.position —
            // which a captured graph freezes. One kernel that derives the row from d_pos
            // replaces the two memcpys; it moves the same bytes in the same order, so it
            // is bit-identical, and it is the reason replay 2 writes a different row from
            // replay 1 instead of overwriting replay 1's. The f16 twin narrows on the way
            // in and is otherwise the same kernel, same d_pos, same row layout.
            if (st.kv_f16)
                k3k::k3_mla_kv_store_f16(st.mla_kv_cache[mla_ord], s.kv_cmpr_normed,
                                         s.kv_a_out, st.d_pos, cfg.kv_lora_rank,
                                         cfg.rope_dim, cfg.key_length, l_kv.st);
            else
                k3k::k3_mla_kv_store_f32(st.mla_kv_cache[mla_ord], s.kv_cmpr_normed,
                                         s.kv_a_out, st.d_pos, cfg.kv_lora_rank,
                                         cfg.rope_dim, cfg.key_length, l_kv.st);

            if (!L.attn_k_b.ok() || !L.attn_v_b.ok()) return false;
            absorb_strided = k3k::k3_mla_absorb_q_strided(
                s.absorbed_q, s.q_proj_out, (const float*)L.attn_k_b.data, qk_nope,
                cfg.kv_lora_rank, cfg.rope_dim, cfg.key_length_mla, qh, stream);
            if (!absorb_strided) {
                cudaMemcpy2DAsync(s.q_nope, (size_t)qk_nope * sizeof(float),
                                  s.q_proj_out,
                                  (size_t)cfg.key_length_mla * sizeof(float),
                                  (size_t)qk_nope * sizeof(float), qh,
                                  cudaMemcpyDeviceToDevice, stream);
                cudaMemcpy2DAsync(s.q_pe, (size_t)cfg.rope_dim * sizeof(float),
                                  (const char*)s.q_proj_out +
                                      (size_t)qk_nope * sizeof(float),
                                  (size_t)cfg.key_length_mla * sizeof(float),
                                  (size_t)cfg.rope_dim * sizeof(float), qh,
                                  cudaMemcpyDeviceToDevice, stream);
                k3k::mla_absorb_q_f32(s.absorbed_q, s.q_nope, s.q_pe,
                                      (const float*)L.attn_k_b.data, qk_nope,
                                      cfg.kv_lora_rank, cfg.rope_dim, qh, stream);
            }
            if (fwd.debug) fwd.debug("mla_absorb_q", layer, s.absorbed_q, qh * cfg.key_length);

            // JOIN the KV lane. The decode kernel reads the row this token's store
            // just wrote, so this is the edge that makes the fork legal at all.
            if (mla_fork) dag_join(0);

            const float mla_scale = 1.0f / std::sqrt((float)cfg.key_length_mla);
            // host length: picks the launch plan only.
            // device length: what the kernel attends over.
            if (st.kv_f16)
                k3k::mla_decode_attn_kvf16(s.mla_attn_out, s.absorbed_q,
                                           st.mla_kv_cache[mla_ord],
                                           (const float*)L.attn_v_b.data, cfg.key_length,
                                           cfg.kv_lora_rank, cfg.value_length_mla, qh,
                                           st.position + 1, st.d_pos, mla_scale, stream);
            else
                k3k::mla_decode_attn_f32(s.mla_attn_out, s.absorbed_q,
                                         st.mla_kv_cache[mla_ord],
                                         (const float*)L.attn_v_b.data, cfg.key_length,
                                         cfg.kv_lora_rank, cfg.value_length_mla, qh,
                                         st.position + 1, st.d_pos, mla_scale, stream);
            if (fwd.debug) fwd.debug("dbg_preattn", layer, s.mla_attn_out, qh * cfg.value_length_mla);

            if (L.has_attn_gate) {
                // Issued on its own lane so it runs UNDER the attention rather than
                // after it. Host issue order is irrelevant here -- every launch is
                // async and the lane's only dependency is the fork event, so the
                // gate is free to start the moment the fork retires.
                if (!proj_h_on(l_gate, s.gate_proj_out, s.normed, L.attn_gate,
                               qh * cfg.value_length_mla, H))
                    return false;
                if (fwd.debug) fwd.debug("dbg_gateproj", layer, s.gate_proj_out, qh * cfg.value_length_mla);
                if (mla_fork) dag_join(1);
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
        // normed2 has up to four consumers below -- the router, routed_down, and both
        // shared-expert projections -- and every one of them used to re-quantise it.
        hoist_act(s.normed2, H);

        k3_profiler().start(layer < cfg.leading_dense ? "ffn_dense" : "ffn_moe", stream);
        if (layer < cfg.leading_dense) {
            if (!proj_h(s.dense_gate, s.normed2, L.ffn_gate, cfg.dense_ffn, H)) return false;
            if (!proj_h(s.dense_up, s.normed2, L.ffn_up, cfg.dense_ffn, H)) return false;
            if (fwd.debug) fwd.debug("dbg_dense_gate", layer, s.dense_gate, cfg.dense_ffn);
            if (fwd.debug) fwd.debug("dbg_dense_up", layer, s.dense_up, cfg.dense_ffn);
            k3k::situ_f32(s.dense_situ, s.dense_gate, s.dense_up, cfg.dense_ffn,
                         cfg.situ_beta, cfg.situ_linear_beta, stream);
            if (fwd.debug) fwd.debug("dbg_dense_situ", layer, s.dense_situ, cfg.dense_ffn);
            if (!proj(s.ffn_out, s.dense_situ, L.ffn_down, H, cfg.dense_ffn)) return false;
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

            if (!proj_h(s.router_logits, s.normed2, L.ffn_gate_inp, cfg.n_experts, H))
                return false;
            if (fwd.debug) fwd.debug("dbg_router_logits", layer, s.router_logits, cfg.n_experts);
            if (!L.exp_probs_b.ok()) return false;
            // Shared-memory selection first; it declines to the original below on any
            // shape it does not handle, and is bit-identical where it does.
            if (!k3k::k3_moe_router_fast(s.router_w, s.router_ids, s.router_logits,
                                         (const float*)L.exp_probs_b.data, cfg.n_experts,
                                         cfg.top_k, /*n_tokens=*/1, /*norm_w=*/true,
                                         /*w_scale=*/1.0f, stream))
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
            if (!proj_h_on(l_rd, s.routed_down_out, s.normed2, L.ffn_routed_down,
                           cfg.expert_latent, H))
                return false;

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
            const bool moe_ok = k3k::moe_expert_ffn_f32_by_type(
                s.moe_out, s.moe_scratch, s.routed_down_out, s.router_ids, s.router_w,
                L.ffn_gate_exps.data, L.ffn_up_exps.data, L.ffn_down_exps.data,
                cfg.expert_latent, moe_ffn_rank, cfg.top_k, cfg.situ_beta,
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
                if (!proj_h_on(l_shx, s.dense_gate, s.normed2, L.ffn_gate_shexp,
                               shexp_band, H))
                    return false;
                if (!proj_h_on(l_shx, s.dense_up, s.normed2, L.ffn_up_shexp,
                               shexp_band, H))
                    return false;
                k3k::situ_f32(s.dense_situ, s.dense_gate, s.dense_up, shexp_band,
                             cfg.situ_beta, cfg.situ_linear_beta, l_shx.st);
                if (!proj_on(l_shx, s.shexp_out, s.dense_situ, L.ffn_down_shexp, H,
                             shexp_band))
                    return false;
                if (fwd.debug) fwd.debug("dbg_shexp_partial", layer, s.shexp_out, H);
            } else {
                // The fused buffer is reduced whole, so a layer without a shared
                // expert must still present a well-defined summand there.
                cudaMemsetAsync(s.shexp_out, 0, (size_t)H * sizeof(float), l_shx.st);
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
        if (fused_tail &&
            k3k::k3_add3_f32(hidden_out, s.ffn_out, s.ffn_out, s.shexp_out,
                             hidden_out, H, stream)) {
            if (fwd.debug) fwd.debug("ffn_out", layer, s.ffn_out, H);
        } else {
            if (fused_tail)
                k3k::k3_add_f32(s.ffn_out, s.ffn_out, s.shexp_out, H, stream);
            if (fwd.debug) fwd.debug("ffn_out", layer, s.ffn_out, H);
            k3k::k3_add_f32(hidden_out, hidden_out, s.ffn_out, H, stream);
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
