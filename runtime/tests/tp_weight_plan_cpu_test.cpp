// CPU-only tests for the tensor -> shard-rule mapping (no GPU needed).
//
// This is the worst-consequence layer in the TP stack. A wrong rule is not a crash
// and not a shape mismatch: row-shard something that should be column-sharded and
// every rank computes a well-formed activation over the wrong slice of the input,
// the all-reduce sums eight wrong partials into a plausible wrong answer, and the
// model stays fluent. Nothing in a smoke test notices. So every rule is asserted
// against the real GGUF tensor names sparkinfer's loader reads.
//
// Build/run:
//   g++ -std=c++17 -I runtime/include runtime/tests/tp_weight_plan_cpu_test.cpp
//       runtime/src/tp/weight_plan.cpp runtime/src/tp/shard.cpp -o /tmp/t && /tmp/t

#include "sparkinfer/tp/weight_plan.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace sparkinfer::tp;

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, ...)                                                        \
    do {                                                                        \
        ++g_checks;                                                             \
        if (!(cond)) {                                                          \
            std::printf("  FAIL %s:%d: %s\n        ", __FILE__, __LINE__, #cond); \
            std::printf(__VA_ARGS__);                                           \
            std::printf("\n");                                                  \
            ++g_failures;                                                       \
        }                                                                       \
    } while (0)

#define CHECK_RULE(name, want)                                                   \
    do {                                                                         \
        Rule got = rule_for(name);                                               \
        CHECK(got == (want), "%s: got %s want %s", name, rule_name(got),          \
              rule_name(want));                                                   \
    } while (0)

static ModelShape kimi_k3() {
    ModelShape s;
    s.hidden = 7168; s.n_q_heads = 96; s.n_kv_heads = 1; s.head_dim = 128;
    s.vocab = 163840; s.n_experts = 896; s.moe_ffn = 3072; s.dense_ffn = 0;
    return s;
}

static ModelShape qwen36() {
    ModelShape s;
    s.hidden = 2048; s.n_q_heads = 16; s.n_kv_heads = 2; s.head_dim = 256;
    s.vocab = 248320; s.n_experts = 256; s.moe_ffn = 512; s.dense_ffn = 0;
    return s;
}

// A shape with enough kv heads to actually shard them, so the non-replicated
// branch is covered too.
static ModelShape many_kv() {
    ModelShape s = qwen36();
    s.n_kv_heads = 8;
    return s;
}

static Config cfg_of(int tp) {
    Config c;
    c.tp_size = tp;
    for (int i = 0; i < tp; ++i) c.devices.push_back(i);
    return c;
}

static ShardDims dims_of(const ModelShape& s, int tp, int rank) {
    ShardDims d;
    ShardError e = shard_dims(s, cfg_of(tp), rank, &d);
    if (!e.ok()) std::printf("  (shard_dims error: %s)\n", e.message.c_str());
    return d;
}

// ---------------------------------------------------------------------------

static void test_the_two_collective_tensors() {
    std::printf("the_two_collective_tensors\n");
    // Wo and ffn_down are the ONLY tensors that force an all-reduce. If either
    // ever stops being a column shard, TP silently stops being correct.
    CHECK_RULE("blk.0.attn_output.weight", Rule::ColShard);
    CHECK_RULE("blk.42.ffn_down.weight", Rule::ColShard);
    CHECK(rule_needs_reduce(Rule::ColShard), "col shard must require a reduce");
    CHECK(!rule_needs_reduce(Rule::RowShard), "row shard must NOT require a reduce");
    CHECK(!rule_needs_reduce(Rule::Replicate), "replicate must NOT require a reduce");
    CHECK(!rule_needs_reduce(Rule::ExpertShard),
          "expert shard's combine folds into the ffn reduce, not its own");
}

static void test_attention_rules() {
    std::printf("attention_rules\n");
    CHECK_RULE("blk.0.attn_q.weight", Rule::RowShard);
    CHECK_RULE("blk.0.attn_k.weight", Rule::RowShard);
    CHECK_RULE("blk.0.attn_v.weight", Rule::RowShard);
    CHECK_RULE("blk.0.attn_qkv.weight", Rule::RowShard);
    CHECK_RULE("blk.0.attn_gate.weight", Rule::RowShard);
    // Norms are elementwise over hidden, and hidden is never sharded.
    CHECK_RULE("blk.0.attn_norm.weight", Rule::Replicate);
    CHECK_RULE("blk.0.attn_q_norm.weight", Rule::Replicate);
    CHECK_RULE("blk.0.attn_k_norm.weight", Rule::Replicate);
    CHECK_RULE("blk.0.attn_post_norm.weight", Rule::Replicate);
}

static void test_moe_rules() {
    std::printf("moe_rules\n");
    CHECK_RULE("blk.0.ffn_gate_exps.weight", Rule::ExpertShard);
    CHECK_RULE("blk.0.ffn_up_exps.weight", Rule::ExpertShard);
    CHECK_RULE("blk.0.ffn_down_exps.weight", Rule::ExpertShard);
    // The router must see the whole hidden vector to score experts, and its output
    // is a per-expert scalar set rather than an activation.
    CHECK_RULE("blk.0.ffn_gate_inp.weight", Rule::Replicate);
    // Shared expert runs for every token on every rank; replicating avoids adding
    // a third all-reduce per layer for a tensor that is small next to the routed set.
    CHECK_RULE("blk.0.ffn_gate_shexp.weight", Rule::Replicate);
    CHECK_RULE("blk.0.ffn_up_shexp.weight", Rule::Replicate);
    CHECK_RULE("blk.0.ffn_down_shexp.weight", Rule::Replicate);
}

static void test_model_level_rules() {
    std::printf("model_level_rules\n");
    // token_embd is a gather, not a matmul — every rank must be able to fetch any row.
    CHECK_RULE("token_embd.weight", Rule::Replicate);
    CHECK_RULE("output_norm.weight", Rule::Replicate);
    // lm_head row-shards over vocab: each rank produces its own logit band, and a
    // greedy argmax then exchanges only (max, index) per rank.
    CHECK_RULE("output.weight", Rule::RowShard);
}

static void test_unknown_names_are_refused_not_guessed() {
    std::printf("unknown_names_are_refused_not_guessed\n");
    // The whole point: a new tensor in a new model must be a loud load error, not
    // a silent mis-shard found later as a quality regression.
    CHECK_RULE("blk.0.something_new.weight", Rule::Unknown);
    CHECK_RULE("totally_made_up", Rule::Unknown);
    CHECK_RULE("", Rule::Unknown);
    // K3 tensors sparkinfer has no loader for yet must also refuse rather than
    // fall into a plausible-looking default.
    CHECK_RULE("blk.0.attn_kv_b.weight", Rule::Unknown);
    CHECK_RULE("blk.0.attn_res_proj.weight", Rule::Unknown);

    ShardDims d = dims_of(kimi_k3(), 8, 0);
    TensorPlan p = plan_for("blk.0.mystery.weight", 1024, 7168, d);
    CHECK(p.rule == Rule::Unknown, "unknown must stay unknown through plan_for");
    CHECK(!p.note.empty(), "must explain itself");
    CHECK(p.note.find("refusing to guess") != std::string::npos,
          "note should say it refuses to guess: %s", p.note.c_str());
}

static void test_layer_index_is_ignored() {
    std::printf("layer_index_is_ignored\n");
    // Same rule at layer 0 and layer 92 — the index carries no shard meaning.
    for (const char* n : {"blk.0.attn_q.weight", "blk.7.attn_q.weight",
                          "blk.92.attn_q.weight", "blk.999.attn_q.weight"}) {
        CHECK_RULE(n, Rule::RowShard);
    }
    CHECK_RULE("blk.0.attn_output.weight", Rule::ColShard);
    CHECK_RULE("blk.92.attn_output.weight", Rule::ColShard);
}

static void test_tp1_collapses_everything_to_replicate() {
    std::printf("tp1_collapses_everything_to_replicate\n");
    // Single-GPU keeps one loader code path, so it cannot diverge from the sharded one.
    ShardDims d = dims_of(kimi_k3(), 1, 0);
    for (const char* n : {"blk.0.attn_q.weight", "blk.0.attn_output.weight",
                          "blk.0.ffn_down_exps.weight", "output.weight"}) {
        TensorPlan p = plan_for(n, 12288, 7168, d);
        CHECK(p.rule == Rule::Replicate, "%s at tp=1 should be replicate, got %s",
              n, rule_name(p.rule));
        CHECK(p.rows == 12288 && p.cols == 7168, "%s shape must be untouched", n);
    }
}

static void test_row_shard_plan_bands_tile() {
    std::printf("row_shard_plan_bands_tile\n");
    // lm_head over K3's vocab: 163840 rows / 8 ranks.
    const int rows = 163840, cols = 7168, tp = 8;
    int prev_end = 0;
    for (int r = 0; r < tp; ++r) {
        ShardDims d = dims_of(kimi_k3(), tp, r);
        TensorPlan p = plan_for("output.weight", rows, cols, d);
        CHECK(p.rule == Rule::RowShard, "rank %d rule", r);
        CHECK(p.band.offset == prev_end, "rank %d gap: %d vs %d", r, p.band.offset, prev_end);
        CHECK(p.rows == rows / tp, "rank %d rows: %d", r, p.rows);
        CHECK(p.cols == cols, "row shard must not touch cols");
        prev_end = p.band.end();
    }
    CHECK(prev_end == rows, "bands must cover the whole vocab: %d", prev_end);
}

static void test_col_shard_plan_bands_tile() {
    std::printf("col_shard_plan_bands_tile\n");
    // Wo: [hidden, n_q_heads*head_dim] column-sharded on the input dim.
    const int rows = 7168, cols = 96 * 128, tp = 8;
    int prev_end = 0;
    for (int r = 0; r < tp; ++r) {
        ShardDims d = dims_of(kimi_k3(), tp, r);
        TensorPlan p = plan_for("blk.0.attn_output.weight", rows, cols, d);
        CHECK(p.rule == Rule::ColShard, "rank %d rule", r);
        CHECK(p.band.offset == prev_end, "rank %d gap", r);
        CHECK(p.cols == cols / tp, "rank %d cols: %d", r, p.cols);
        CHECK(p.rows == rows, "col shard must not touch rows");
        prev_end = p.band.end();
    }
    CHECK(prev_end == cols, "bands must cover the whole input dim: %d", prev_end);
}

static void test_kv_replication_is_applied_and_explained() {
    std::printf("kv_replication_is_applied_and_explained\n");
    // K3: 1 kv head at tp=8. attn_k/v are RowShard in the table but MUST come back
    // replicated — a kv group cannot be split across ranks. Row-sharding a single
    // kv head across 8 ranks is the bug this closes.
    ShardDims d = dims_of(kimi_k3(), 8, 3);
    CHECK(d.kv_replicated, "K3 at tp=8 must replicate kv");
    for (const char* n : {"blk.0.attn_k.weight", "blk.0.attn_v.weight"}) {
        TensorPlan p = plan_for(n, 128, 7168, d);
        CHECK(p.rule == Rule::Replicate, "%s must be replicated, got %s", n,
              rule_name(p.rule));
        CHECK(p.replicated_fallback, "%s must flag the fallback so the load log shows it", n);
        CHECK(p.note.find("kv group cannot be split") != std::string::npos,
              "%s note should explain: %s", n, p.note.c_str());
        CHECK(p.rows == 128, "replicated tensor keeps its full shape");
    }
    // q still shards — only the kv side replicates.
    TensorPlan q = plan_for("blk.0.attn_q.weight", 96 * 128, 7168, d);
    CHECK(q.rule == Rule::RowShard, "q must still row-shard");
    CHECK(!q.replicated_fallback, "q is not a fallback");

    // Qwen3.6: 2 kv heads at tp=8 — same replication path.
    ShardDims d36 = dims_of(qwen36(), 8, 0);
    CHECK(d36.kv_replicated, "Qwen3.6 at tp=8 must replicate kv");
    CHECK(plan_for("blk.0.attn_k.weight", 512, 2048, d36).rule == Rule::Replicate,
          "qwen3.6 kv must replicate at tp=8");
}

static void test_kv_shards_when_there_are_enough_heads() {
    std::printf("kv_shards_when_there_are_enough_heads\n");
    // 8 kv heads at tp=8: one each, so the row shard stands and nothing replicates.
    ShardDims d = dims_of(many_kv(), 8, 5);
    CHECK(!d.kv_replicated, "8 kv heads at tp=8 should shard, not replicate");
    TensorPlan p = plan_for("blk.0.attn_k.weight", 8 * 256, 2048, d);
    CHECK(p.rule == Rule::RowShard, "should row-shard, got %s", rule_name(p.rule));
    CHECK(!p.replicated_fallback, "no fallback expected");
    CHECK(p.rows == 256, "one kv head per rank: %d", p.rows);
}

static void test_fused_qkv_with_replicated_kv_is_refused() {
    std::printf("fused_qkv_with_replicated_kv_is_refused\n");
    // A fused attn_qkv cannot be expressed as ONE rule when q shards and kv
    // replicates. Refusing forces the loader to split the tensor first, rather
    // than row-sharding the whole thing and slicing a kv head in eight.
    ShardDims d = dims_of(kimi_k3(), 8, 0);
    TensorPlan p = plan_for("blk.0.attn_qkv.weight", 96 * 128 + 2 * 128, 7168, d);
    CHECK(p.rule == Rule::Unknown, "fused qkv + replicated kv must refuse, got %s",
          rule_name(p.rule));
    CHECK(p.note.find("split the fused tensor") != std::string::npos,
          "note should tell the loader what to do: %s", p.note.c_str());

    // With enough kv heads the fused tensor is a clean row shard again.
    ShardDims ok = dims_of(many_kv(), 8, 0);
    CHECK(plan_for("blk.0.attn_qkv.weight", 16 * 256 + 8 * 256, 2048, ok).rule
              == Rule::RowShard,
          "fused qkv should row-shard when kv is not replicated");
}

static void test_expert_shard_plan() {
    std::printf("expert_shard_plan\n");
    // 896 experts / 8 = 112 per rank, bands tiling the expert axis.
    int prev_end = 0;
    for (int r = 0; r < 8; ++r) {
        ShardDims d = dims_of(kimi_k3(), 8, r);
        TensorPlan p = plan_for("blk.0.ffn_down_exps.weight", 896, 3072, d);
        CHECK(p.rule == Rule::ExpertShard, "rank %d rule: %s", r, rule_name(p.rule));
        CHECK(p.experts.offset == prev_end, "rank %d expert gap", r);
        CHECK(p.experts.extent == 112, "rank %d owns %d experts", r, p.experts.extent);
        prev_end = p.experts.end();
    }
    CHECK(prev_end == 896, "expert bands must cover all 896: %d", prev_end);
}

static void test_expert_ffn_width_fallback() {
    std::printf("expert_ffn_width_fallback\n");
    // Experts don't divide but the width does: shard_dims() falls back, and the
    // plan must follow it to a column shard rather than staying ExpertShard with
    // a nonsense band.
    ModelShape s = kimi_k3();
    s.n_experts = 100;              // 100 / 8 does not divide
    s.moe_ffn = 3072;               // 3072 / 8 does
    ShardDims d = dims_of(s, 8, 2);
    CHECK(!d.experts_sharded, "should have fallen back");
    TensorPlan p = plan_for("blk.0.ffn_up_exps.weight", 100, 3072, d);
    CHECK(p.rule == Rule::ColShard, "fallback should become a col shard, got %s",
          rule_name(p.rule));
    CHECK(p.cols == 384, "3072/8: %d", p.cols);
    CHECK(p.experts.extent == 100, "all experts on every rank: %d", p.experts.extent);
    CHECK(p.note.find("ffn-width") != std::string::npos, "note: %s", p.note.c_str());
}

static void test_indivisible_shapes_are_refused_at_plan_time() {
    std::printf("indivisible_shapes_are_refused_at_plan_time\n");
    // shard_dims() checks the MODEL axes; plan_for() checks the actual TENSOR
    // shape, which can still fail to divide (e.g. an odd lm_head).
    ShardDims d = dims_of(kimi_k3(), 8, 0);
    TensorPlan row = plan_for("output.weight", 163841, 7168, d);   // odd rows
    CHECK(row.rule == Rule::Unknown, "odd row count must be refused");
    CHECK(row.note.find("not divisible") != std::string::npos, "note: %s", row.note.c_str());

    TensorPlan col = plan_for("blk.0.attn_output.weight", 7168, 12289, d);  // odd cols
    CHECK(col.rule == Rule::Unknown, "odd col count must be refused");
    CHECK(col.note.find("not divisible") != std::string::npos, "note: %s", col.note.c_str());
}

static void test_every_known_suffix_has_a_real_rule() {
    std::printf("every_known_suffix_has_a_real_rule\n");
    // No entry in the table may itself resolve to Unknown — that would be a typo
    // in the table silently behaving like a missing tensor.
    std::vector<std::string> sfx = known_tensor_suffixes();
    CHECK(sfx.size() >= 25, "table looks too small: %zu entries", sfx.size());
    for (const std::string& s : sfx) {
        Rule r = rule_for(s);
        CHECK(r != Rule::Unknown, "table entry '%s' resolves to Unknown", s.c_str());
        // And it must resolve identically with a blk prefix.
        Rule pref = rule_for("blk.5." + s);
        if (s.rfind("output", 0) != 0 && s.rfind("token_embd", 0) != 0) {
            CHECK(pref == r, "'%s' differs with a blk prefix: %s vs %s", s.c_str(),
                  rule_name(pref), rule_name(r));
        }
    }
}

static void test_rule_names_are_stable() {
    std::printf("rule_names_are_stable\n");
    // These strings appear in load logs that get pasted into PRs.
    CHECK(std::string(rule_name(Rule::Replicate)) == "replicate", "replicate");
    CHECK(std::string(rule_name(Rule::RowShard)) == "row", "row");
    CHECK(std::string(rule_name(Rule::ColShard)) == "col", "col");
    CHECK(std::string(rule_name(Rule::ExpertShard)) == "expert", "expert");
    CHECK(std::string(rule_name(Rule::Unknown)) == "unknown", "unknown");
}

// ---------------------------------------------------------------------------
// Kimi K3 KDA — the head-parallel plan, cross-checked against tp::kda_shard_dims
// ---------------------------------------------------------------------------

static KdaShape k3_kda_shape() {
    KdaShape sh;
    sh.hidden = 7168; sh.n_heads = 96; sh.head_dim = 128; sh.conv_kernel = 4;
    return sh;
}

static void test_kda_every_loader_tensor_has_a_rule() {
    std::printf("kda_every_loader_tensor_has_a_rule\n");
    // Exactly the names runtime/src/models/kimi_k3.cpp uploads for a KDA layer.
    // Rule::Unknown here is a hard load failure, so a missing entry must be caught
    // in CI and not on a rented node three hours into a run.
    const char* kda_tensors[] = {
        "blk.0.attn_norm.weight",     "blk.0.attn_res_score.weight",
        "blk.0.attn_q.weight",        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",        "blk.0.ssm_g.weight",
        "blk.0.ssm_conv1d_q.weight",  "blk.0.ssm_conv1d_k.weight",
        "blk.0.ssm_conv1d_v.weight",  "blk.0.ssm_f_a.weight",
        "blk.0.ssm_f_b.weight",       "blk.0.ssm_beta.weight",
        "blk.0.ssm_dt.bias",          "blk.0.ssm_a",
        "blk.0.ssm_norm.weight",      "blk.0.attn_output.weight",
    };
    for (const char* t : kda_tensors)
        CHECK(rule_for(t) != Rule::Unknown, "%s has no rule", t);
}

static void test_kda_plan_is_internally_consistent() {
    std::printf("kda_plan_is_internally_consistent\n");
    // The four projections that consume the full hidden state and produce the
    // qkv-wide activation must all take the SAME rule. One of them replicated
    // while the others shard gives that rank a full-width tensor indexed at
    // rank-local offsets — fluent output over the wrong slice.
    CHECK_RULE("blk.3.attn_q.weight", Rule::RowShard);
    CHECK_RULE("blk.3.attn_k.weight", Rule::RowShard);
    CHECK_RULE("blk.3.attn_v.weight", Rule::RowShard);
    CHECK_RULE("blk.3.ssm_g.weight",  Rule::RowShard);
    // ssm_f_b also lands on the qkv axis (its output is qkv wide), so it shards
    // with them. This is the row the plan was missing.
    CHECK_RULE("blk.3.ssm_f_b.weight", Rule::RowShard);
    // The one collective point in a KDA layer.
    CHECK_RULE("blk.3.attn_output.weight", Rule::ColShard);
    CHECK(rule_needs_reduce(Rule::ColShard), "attn_output must force a reduce");
    CHECK(!rule_needs_reduce(Rule::RowShard), "a row shard needs no collective");

    // MUST NOT be promoted: ssm_f_a's OUTPUT is the shared head_dim-wide vector
    // every rank's ssm_f_b shard consumes whole.
    CHECK_RULE("blk.3.ssm_f_a.weight", Rule::Replicate);
    // 1-D over head_dim, not over qkv. head_dim never shards.
    CHECK_RULE("blk.3.ssm_norm.weight", Rule::Replicate);
    // Replicate-and-offset: small, and indexed at the rank's band by the forward.
    CHECK_RULE("blk.3.ssm_dt.bias", Rule::Replicate);
    CHECK_RULE("blk.3.ssm_a", Rule::Replicate);
    CHECK_RULE("blk.3.ssm_conv1d_q.weight", Rule::Replicate);
    CHECK_RULE("blk.3.ssm_beta.weight", Rule::Replicate);
}

static void test_kda_plan_extents_match_kda_shard_dims() {
    std::printf("kda_plan_extents_match_kda_shard_dims\n");
    // THE CROSS-CHECK. weight_plan slices the weights; kda_shard_dims sizes the
    // activations and the recurrent state. They are computed independently, and a
    // disagreement between them is invisible at runtime — the shapes are each
    // self-consistent, the model loads, and every rank contracts over a slice that
    // does not match its own state.
    const int tp = 8;
    for (int r = 0; r < tp; ++r) {
        ShardDims d;
        CHECK(shard_dims(kimi_k3(), cfg_of(tp), r, &d).ok(), "shard_dims rank %d", r);
        KdaShardDims k;
        CHECK(kda_shard_dims(k3_kda_shape(), cfg_of(tp), r, &k).ok(), "kda rank %d", r);

        // GGUF stores [in, out]; plan_for takes (rows=out, cols=in).
        TensorPlan q = plan_for("blk.0.attn_q.weight", 12288, 7168, d);
        CHECK(q.rule == Rule::RowShard, "attn_q rule");
        CHECK(q.rows == k.qkv, "attn_q rows %d vs kda qkv %d", q.rows, k.qkv);
        CHECK(q.band.offset == k.qkv_band.offset,
              "attn_q band %d vs kda qkv_band %d", q.band.offset, k.qkv_band.offset);

        TensorPlan fb = plan_for("blk.0.ssm_f_b.weight", 12288, 128, d);
        CHECK(fb.rows == k.qkv, "ssm_f_b rows");
        CHECK(fb.band.offset == k.qkv_band.offset, "ssm_f_b band");

        // attn_output consumes the SAME band it produced: the col shard of Wo has
        // to line up with the row shard of q/k/v/g or the contraction is over a
        // different slice than the one this rank computed.
        TensorPlan o = plan_for("blk.0.attn_output.weight", 7168, 12288, d);
        CHECK(o.rule == Rule::ColShard, "attn_output rule");
        CHECK(o.cols == k.qkv, "attn_output cols %d vs kda qkv %d", o.cols, k.qkv);
        CHECK(o.band.offset == k.qkv_band.offset,
              "THE SEAM: attn_output col band %d must equal attn_q row band %d",
              o.band.offset, k.qkv_band.offset);

        // ssm_f_a stays whole on every rank, at every rank.
        TensorPlan fa = plan_for("blk.0.ssm_f_a.weight", 128, 7168, d);
        CHECK(fa.rule == Rule::Replicate, "ssm_f_a must stay whole");
        CHECK(fa.rows == 128, "ssm_f_a rows must be full head_dim, got %d", fa.rows);
    }
}

static void test_kda_attn_k_v_are_not_a_kv_group() {
    std::printf("kda_attn_k_v_are_not_a_kv_group\n");
    // THE TRAP. attn_k.weight / attn_v.weight are written by BOTH branches: on an
    // MLA layer they are absent (MLA uses attn_k_b/attn_v_b), on a KDA layer they
    // are per-head input projections at [hidden, qkv] with no kv group involved.
    //
    // weight_plan downgrades them to Replicate whenever kv_replicated is set, and
    // shard_dims() DOES set it for K3 (MLA stored as MQA, 1 kv head < 8 ranks). If
    // the KDA loader ever inherits that flag, every rank holds a full-width attn_k
    // beside a banded attn_q and the layer runs at full speed over mismatched
    // slices. The loader states the exemption explicitly; this pins both halves.
    ShardDims d;
    CHECK(shard_dims(kimi_k3(), cfg_of(8), 0, &d).ok(), "shard");
    CHECK(d.kv_replicated, "K3 at tp=8 must report kv_replicated — 1 kv head < 8 ranks");

    TensorPlan mla = plan_for("blk.3.attn_k.weight", 12288, 7168, d);
    CHECK(mla.rule == Rule::Replicate, "with kv_replicated set, the table downgrades");
    CHECK(mla.replicated_fallback, "and says so");

    // The exemption the KDA loader applies: same tensor, kv_replicated cleared.
    ShardDims kda = d;
    kda.kv_replicated = false;
    for (const char* t : {"blk.3.attn_k.weight", "blk.3.attn_v.weight",
                          "blk.3.attn_q.weight"}) {
        TensorPlan p = plan_for(t, 12288, 7168, kda);
        CHECK(p.rule == Rule::RowShard, "%s must row-shard on a KDA layer", t);
        CHECK(p.rows == 1536, "%s rows %d, want 1536", t, p.rows);
        CHECK(!p.replicated_fallback, "%s must not report a fallback", t);
    }
    // And all three must land on the SAME band, or q/k/v cover different heads.
    CHECK(plan_for("blk.3.attn_q.weight", 12288, 7168, kda).band.offset ==
          plan_for("blk.3.attn_k.weight", 12288, 7168, kda).band.offset,
          "attn_q and attn_k must band identically");
}

static void test_kda_table_has_no_shadowed_duplicates() {
    std::printf("kda_table_has_no_shadowed_duplicates\n");
    // The table is one std::unordered_map built from an initializer list, so a
    // repeated key is NOT an error: the first insertion wins and the second is
    // silently dropped. An author editing the second copy then sees no effect.
    // Both known collisions (attn_gate.weight, ssm_beta.weight) have been folded
    // into a single entry each; this pins the key count so a new one shows up as
    // a mismatch here rather than as an edit that quietly does nothing.
    std::vector<std::string> sfx = known_tensor_suffixes();
    CHECK(sfx.size() == 52, "table has %zu suffixes, expected 52 — if you added or "
          "removed a rule, update this count; if it did NOT change when you added "
          "one, you have shadowed an existing key", sfx.size());
    // And no key appears twice in what the table hands back.
    for (std::size_t i = 0; i < sfx.size(); ++i)
        for (std::size_t j = i + 1; j < sfx.size(); ++j)
            CHECK(sfx[i] != sfx[j], "duplicate suffix %s", sfx[i].c_str());
}

int main() {
    test_the_two_collective_tensors();
    test_attention_rules();
    test_moe_rules();
    test_model_level_rules();
    test_unknown_names_are_refused_not_guessed();
    test_layer_index_is_ignored();
    test_tp1_collapses_everything_to_replicate();
    test_row_shard_plan_bands_tile();
    test_col_shard_plan_bands_tile();
    test_kv_replication_is_applied_and_explained();
    test_kv_shards_when_there_are_enough_heads();
    test_fused_qkv_with_replicated_kv_is_refused();
    test_expert_shard_plan();
    test_expert_ffn_width_fallback();
    test_indivisible_shapes_are_refused_at_plan_time();
    test_every_known_suffix_has_a_real_rule();
    test_rule_names_are_stable();

    test_kda_every_loader_tensor_has_a_rule();
    test_kda_plan_is_internally_consistent();
    test_kda_plan_extents_match_kda_shard_dims();
    test_kda_attn_k_v_are_not_a_kv_group();
    test_kda_table_has_no_shadowed_duplicates();

    std::printf("\n%d checks, %d failure(s)\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
