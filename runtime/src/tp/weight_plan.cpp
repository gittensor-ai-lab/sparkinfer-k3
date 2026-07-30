// Tensor -> shard rule. No CUDA — see the header for why this is the second place
// TP bugs live, and why the resolver refuses unknown names instead of guessing.

#include "sparkinfer/tp/weight_plan.h"

#include <sstream>
#include <unordered_map>

namespace sparkinfer {
namespace tp {

const char* rule_name(Rule r) {
    switch (r) {
        case Rule::Replicate:   return "replicate";
        case Rule::RowShard:    return "row";
        case Rule::ColShard:    return "col";
        case Rule::ExpertShard: return "expert";
        case Rule::Unknown:     return "unknown";
    }
    return "?";
}

bool rule_needs_reduce(Rule r) {
    // Only a column shard produces a partial sum over the full output width.
    // A row shard's boundary IS the activation boundary, so it needs nothing.
    return r == Rule::ColShard;
}

namespace {

// The one table. Keyed by the tensor's suffix after the "blk.N." prefix, or by the
// whole name for the model-level tensors. Names taken from the loader in
// runtime/src/models/qwen35.cpp, so this covers what sparkinfer actually reads.
const std::unordered_map<std::string, Rule>& table() {
    static const std::unordered_map<std::string, Rule> t = {
        // --- norms: elementwise over hidden, and hidden is never sharded ---
        {"attn_norm.weight",        Rule::Replicate},
        {"attn_q_norm.weight",      Rule::Replicate},
        {"attn_k_norm.weight",      Rule::Replicate},
        {"attn_post_norm.weight",   Rule::Replicate},
        {"ffn_norm.weight",         Rule::Replicate},
        {"ssm_norm.weight",         Rule::Replicate},
        {"output_norm.weight",      Rule::Replicate},

        // --- attention: q/k/v/qkv produce sharded activations (row) ---
        {"attn_q.weight",           Rule::RowShard},
        {"attn_k.weight",           Rule::RowShard},
        {"attn_v.weight",           Rule::RowShard},
        {"attn_qkv.weight",         Rule::RowShard},
        {"attn_gate.weight",        Rule::RowShard},
        // Wo consumes the sharded attention output and produces a full-width
        // partial sum. One of the two tensors in the model that force a collective.
        {"attn_output.weight",      Rule::ColShard},

        // --- dense FFN ---
        {"ffn_gate.weight",         Rule::RowShard},
        {"ffn_up.weight",           Rule::RowShard},
        {"ffn_down.weight",         Rule::ColShard},   // the other collective

        // --- routed experts: split the expert axis, not a matrix dim ---
        {"ffn_gate_exps.weight",    Rule::ExpertShard},
        {"ffn_up_exps.weight",      Rule::ExpertShard},
        {"ffn_down_exps.weight",    Rule::ExpertShard},

        // --- shared expert: REPLICATED for now. This is a context-budget call,
        // not a correctness one, and it is revisited at 512k.
        //
        // MEASURED on the real UD-Q2_K_XL: the three shexp tensors are 12.0 GiB
        // replicated = 10.5 GiB per rank of the 32.9 GiB total duplication, putting
        // a rank at 129.0 GiB. Per-rank budget on a 140 GiB H200:
        //     ctx     KV      total     verdict
        //     128k    3.4     136.8     fits            <- current target
        //     256k    6.8     140.2     over by 0.2
        //     512k   13.5     146.9     over by 6.9
        // So 128k fits replicated; 512k does not, and needs weights <= 122.1 GiB.
        //
        // SHARDING THEM IS FREE WHEN IT IS NEEDED. The earlier reasoning here —
        // "sharding adds a third all-reduce per layer" — was wrong. The reference
        // computes moe_out = moe_out + shexp, so adding the shared-expert PARTIAL
        // sum before the existing after-FFN all-reduce lets one reduce cover both.
        // gate/up become RowShard (split the ffn width), down becomes ColShard,
        // and a rank drops to 118.5 GiB, which clears 512k.
        //
        // Not doing it yet, because it imposes an ordering constraint (the shexp
        // add MUST precede the reduce) on a forward that does not exist to verify
        // it. Replicated is provably correct regardless of ordering; switch when
        // the forward exists and 512k is the target.
        {"ffn_gate_shexp.weight",   Rule::Replicate},
        {"ffn_up_shexp.weight",     Rule::Replicate},
        {"ffn_down_shexp.weight",   Rule::Replicate},

        // --- router: must see the whole hidden vector to score experts, and its
        // output is a per-expert scalar set rather than an activation.
        {"ffn_gate_inp.weight",     Rule::Replicate},
        {"ffn_gate_inp_shexp.weight", Rule::Replicate},

        // --- Gated DeltaNet / SSM (Qwen3.5/3.6 linear-attention layers) ---
        // Conservatively replicated: the recurrent state is per-head and the conv
        // is over a small kernel, so sharding these needs the recurrent state
        // partitioned too. Correct-but-unsharded until that is designed; flagged in
        // docs/tensor-parallel.md as outstanding rather than silently row-sharded.
        {"ssm_conv1d.weight",       Rule::Replicate},
        {"ssm_beta.weight",         Rule::Replicate},
        {"ssm_alpha.weight",        Rule::Replicate},
        {"ssm_out.weight",          Rule::Replicate},

        // --- Kimi K3 additions -------------------------------------------------
        // Enumerated from the REAL UD-Q2_K_XL tensor table (2573 tensors), not from
        // the reference source, so nothing here is a guess about what ships.

        // MLA (24 layers). q_a/q_b is the LoRA-factored query: q_a is a down-proj
        // to q_lora_rank 1536 and must stay whole (its output is not a head axis),
        // while q_b produces n_head*key_length_mla and row-shards by head.
        {"attn_q_a.weight",         Rule::Replicate},
        {"attn_q_a_norm.weight",    Rule::Replicate},
        {"attn_q_b.weight",         Rule::RowShard},
        // kv_a_mqa produces the SHARED compressed kv (kv_lora + rope) that every
        // head reads — it is the MQA latent, so sharding it would split the one
        // thing all heads need. Replicated, like the kv side generally.
        {"attn_kv_a_mqa.weight",    Rule::Replicate},
        {"attn_kv_a_norm.weight",   Rule::Replicate},
        // k_b/v_b decompress the latent per head: [.., .., n_head] with the head
        // axis LAST, so they shard on the head axis, not on a matrix dim.
        {"attn_k_b.weight",         Rule::ExpertShard},   // head-axis band
        {"attn_v_b.weight",         Rule::ExpertShard},   // head-axis band
        // K3's sigmoid output gate: same shape as the attn output, so same rule.
        {"attn_gate.weight",        Rule::RowShard},

        // Cross-layer residual scoring vectors: [hidden], elementwise against the
        // residual stream, and every rank scores the same stream. Replicated.
        {"attn_res_score.weight",   Rule::Replicate},
        {"ffn_res_score.weight",    Rule::Replicate},
        {"output_res_score.weight", Rule::Replicate},

        // Latent MoE. routed_down/up bracket the expert block (7168<->3584). The
        // router scores the FULL-WIDTH input while the experts consume the latent
        // one, so routed_down consumes hidden (col-shard would need a reduce we do
        // not have there) and routed_up produces hidden from the latent.
        {"ffn_routed_down.weight",  Rule::Replicate},
        {"ffn_routed_norm.weight",  Rule::Replicate},
        {"ffn_routed_up.weight",    Rule::Replicate},
        // noaux_tc router bias, one scalar per expert.
        {"exp_probs_b.bias",        Rule::Replicate},

        // KDA (69 layers). q/k/v projections produce head-major activations and
        // row-shard; the recurrent-state machinery is per-head and small, and the
        // conv/state partitioning is not designed yet, so it replicates.
        {"ssm_a",                   Rule::Replicate},
        {"ssm_dt.bias",             Rule::Replicate},
        {"ssm_beta.weight",         Rule::Replicate},
        {"ssm_f_a.weight",          Rule::Replicate},
        {"ssm_f_b.weight",          Rule::Replicate},
        {"ssm_g.weight",            Rule::RowShard},
        {"ssm_conv1d_q.weight",     Rule::Replicate},
        {"ssm_conv1d_k.weight",     Rule::Replicate},
        {"ssm_conv1d_v.weight",     Rule::Replicate},

        // --- model level ---
        // token_embd is a gather, not a matmul: every rank needs any row.
        {"token_embd.weight",       Rule::Replicate},
        // lm_head: row shard over vocab. Each rank produces logits for its own
        // vocab band; a greedy argmax then exchanges only (max, index) per rank.
        {"output.weight",           Rule::RowShard},
    };
    return t;
}

// Strip a leading "blk.<N>." if present. Everything else is a model-level name.
std::string suffix_of(const std::string& name) {
    if (name.rfind("blk.", 0) != 0) return name;
    const std::size_t dot = name.find('.', 4);
    if (dot == std::string::npos) return name;
    return name.substr(dot + 1);
}

}  // namespace

Rule rule_for(const std::string& tensor_name) {
    const auto& t = table();
    auto it = t.find(suffix_of(tensor_name));
    return (it == t.end()) ? Rule::Unknown : it->second;
}

TensorPlan plan_for(const std::string& tensor_name, int rows, int cols,
                    const ShardDims& d) {
    TensorPlan p;
    p.rule = rule_for(tensor_name);
    p.rows = rows;
    p.cols = cols;

    if (p.rule == Rule::Unknown) {
        std::ostringstream os;
        os << "no shard rule for '" << tensor_name << "' — refusing to guess";
        p.note = os.str();
        return p;
    }

    // TP=1: every rule degenerates to the whole tensor. Keeps the loader on one
    // code path so the single-GPU build cannot diverge.
    if (d.tp_size <= 1) {
        p.rule = Rule::Replicate;
        p.band = Band{0, rows};
        p.experts = Band{0, d.n_experts_total};
        return p;
    }

    const std::string sfx = suffix_of(tensor_name);

    // KV replication. attn_k/attn_v are row-shard in the table, but a kv group
    // cannot be split across ranks — see ShardDims::kv_replicated. Downgrade and
    // SAY SO, so a load log shows duplicated weight rather than it looking like a
    // leak. attn_qkv is fused: when kv is replicated the fused tensor cannot be a
    // clean row shard either, so it is reported for the loader to split by hand.
    if (d.kv_replicated && (sfx == "attn_k.weight" || sfx == "attn_v.weight")) {
        p.rule = Rule::Replicate;
        p.replicated_fallback = true;
        p.band = Band{0, rows};
        std::ostringstream os;
        os << "kv replicated: " << d.n_kv_heads_total << " kv head(s) < tp_size "
           << d.tp_size << ", a kv group cannot be split";
        p.note = os.str();
        return p;
    }
    if (d.kv_replicated && sfx == "attn_qkv.weight") {
        p.rule = Rule::Unknown;
        std::ostringstream os;
        os << "fused attn_qkv with replicated kv (" << d.n_kv_heads_total
           << " kv head(s) < tp_size " << d.tp_size << "): the q part row-shards but "
           << "the kv part must replicate, so the loader must split the fused tensor "
           << "before sharding. Not representable as one rule.";
        p.note = os.str();
        return p;
    }

    switch (p.rule) {
        case Rule::Replicate:
            p.band = Band{0, rows};
            break;

        case Rule::RowShard: {
            if (!divisible(rows, d.tp_size)) {
                p.rule = Rule::Unknown;
                std::ostringstream os;
                os << "row shard of '" << tensor_name << "': rows=" << rows
                   << " not divisible by tp_size=" << d.tp_size;
                p.note = os.str();
                break;
            }
            p.band = even_band(rows, d.tp_size, d.rank);
            p.rows = p.band.extent;
            break;
        }

        case Rule::ColShard: {
            if (!divisible(cols, d.tp_size)) {
                p.rule = Rule::Unknown;
                std::ostringstream os;
                os << "col shard of '" << tensor_name << "': cols=" << cols
                   << " not divisible by tp_size=" << d.tp_size;
                p.note = os.str();
                break;
            }
            p.band = even_band(cols, d.tp_size, d.rank);
            p.cols = p.band.extent;
            break;
        }

        case Rule::ExpertShard: {
            if (!d.experts_sharded) {
                // shard_dims() fell back to splitting each expert's ffn width, so
                // every rank holds every expert but a narrower slice of each.
                p.rule = Rule::ColShard;
                p.experts = Band{0, d.n_experts_total};
                if (!divisible(cols, d.tp_size)) {
                    p.rule = Rule::Unknown;
                    p.note = "expert ffn-width fallback but cols not divisible";
                    break;
                }
                p.band = even_band(cols, d.tp_size, d.rank);
                p.cols = p.band.extent;
                p.note = "experts not divisible by tp_size — ffn-width shard instead";
                break;
            }
            p.experts = d.expert_band;
            p.band = d.expert_band;
            break;
        }

        case Rule::Unknown:
            break;
    }
    return p;
}

std::vector<std::string> known_tensor_suffixes() {
    std::vector<std::string> out;
    out.reserve(table().size());
    for (const auto& kv : table()) out.push_back(kv.first);
    return out;
}

}  // namespace tp
}  // namespace sparkinfer
