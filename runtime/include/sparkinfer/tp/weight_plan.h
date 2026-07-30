#pragma once
// Which shard rule applies to which GGUF tensor. DELIBERATELY CUDA-FREE so the
// whole mapping is unit-testable against real tensor names (see
// runtime/tests/tp_weight_plan_cpu_test.cpp).
//
// This is the second place TP bugs live, after the shard arithmetic itself, and it
// is worse than the first: applying the WRONG RULE to a tensor is not a crash and
// not a shape mismatch. Row-shard something that should be column-sharded and every
// rank computes a well-formed activation over the wrong slice of the input; the
// all-reduce sums eight wrong partials into a plausible wrong answer, the model
// stays fluent, and nothing in a smoke test notices.
//
// So the rule for every tensor is declared in one table, resolved by name, and the
// resolver REFUSES unknown names rather than guessing. A new tensor in a new model
// is then a loud "no rule for blk.0.foo.weight" at load time, not a silent
// mis-shard discovered as a quality regression three weeks later.
//
// THE INVARIANT that makes the rules assignable at all: for every matmul, exactly
// one of the operand dims is the collective's width (hidden). A weight that
// produces a sharded activation is row-sharded; a weight that consumes one is
// column-sharded; a weight on neither side of a shard boundary is replicated.
//
//   Replicate   norms, router (gate_inp), token_embd
//               Small, or needed whole on every rank. Norms are elementwise over
//               hidden and hidden is never sharded. The router must see the whole
//               hidden vector to score experts, and its output is a per-expert
//               scalar set, not an activation.
//   RowShard    attn_q/k/v/qkv, ffn_gate, ffn_up, output (lm_head)
//               Split the OUTPUT dim; produces this rank's slice only.
//   ColShard    attn_output (Wo), ffn_down
//               Split the INPUT dim; consumes this rank's slice, produces a
//               full-width PARTIAL SUM -> all-reduce. These two are the only
//               reason the collective exists.
//   ExpertShard ffn_*_exps
//               Split along the expert axis, not a matrix dim. Each rank owns
//               whole experts.
//   ReplicateKV attn_k/v when n_kv_heads < tp_size
//               Resolved dynamically: see ShardDims::kv_replicated.

#include <string>
#include <vector>

#include "sparkinfer/tp/shard.h"

namespace sparkinfer {
namespace tp {

enum class Rule {
    Replicate,     // every rank holds the whole tensor
    RowShard,      // split output rows; no collective
    ColShard,      // split input cols; REQUIRES an all-reduce after
    ExpertShard,   // split the expert axis
    Unknown,       // no rule — a hard load error, never a guess
};

const char* rule_name(Rule r);

// True when a tensor's rule means the layer must all-reduce after using it.
bool rule_needs_reduce(Rule r);

// One tensor's placement decision for one rank.
struct TensorPlan {
    Rule rule = Rule::Unknown;
    // For RowShard/ColShard: the band of the split dim this rank owns.
    Band band;
    // Sharded shape. For Replicate these equal the full shape.
    int rows = 0;
    int cols = 0;
    // ExpertShard: which experts this rank owns (band over the expert axis).
    Band experts;
    // Set when the rule was downgraded to Replicate because the axis could not be
    // split — the KV-head case. Recorded rather than hidden so a load log shows
    // the duplicated weight instead of it appearing as a leak.
    bool replicated_fallback = false;
    std::string note;   // why, when it is not obvious
};

// Resolve the rule for a GGUF tensor name. Accepts the fully-qualified name
// ("blk.7.attn_q.weight", "output.weight") and ignores the layer index.
//
// Returns Rule::Unknown for anything not in the table. The caller MUST treat that
// as fatal: continuing would mean either replicating a tensor that should shard
// (wasting 8x the memory, and for the 802 GiB target that does not fit) or
// sharding one that should not (silently wrong).
Rule rule_for(const std::string& tensor_name);

// Full placement for one tensor on one rank, given the model's per-rank dims.
// `rows`/`cols` are the FULL tensor shape as stored in the GGUF.
//
// Handles the KV-replication case: attn_k / attn_v carry RowShard in the table,
// but when d.kv_replicated is set they come back as Replicate with
// replicated_fallback = true, because a kv group cannot be split across ranks.
TensorPlan plan_for(const std::string& tensor_name, int rows, int cols,
                    const ShardDims& d);

// Every tensor name the table knows, for tests and for a load-time audit that
// reports coverage rather than discovering a gap per-tensor.
std::vector<std::string> known_tensor_suffixes();

}  // namespace tp
}  // namespace sparkinfer
