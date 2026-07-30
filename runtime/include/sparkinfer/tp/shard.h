#pragma once
// Tensor-parallel shard math. DELIBERATELY CUDA-FREE so it is unit-testable on a
// machine with no GPU (see runtime/tests/tp_shard_cpu_test.cpp).
//
// This is where TP bugs actually live. The collectives are a solved problem —
// NCCL is correct — but an off-by-one in a row/column split, or splitting a GQA
// key-value group across two ranks, produces a model that loads cleanly, runs at
// full speed, and emits subtly wrong tokens. That class of bug is nearly
// invisible on a rented 8-GPU box and trivially catchable here, so every shard
// decision goes through this header and nothing computes an offset inline.
//
// Conventions, fixed once so the forward never has to re-derive them:
//
//   ROW shard  (split the OUTPUT dim)  Wq, Wk, Wv, gate, up, lm_head
//     Each rank owns a contiguous band of output rows and produces a partial
//     activation covering only its own outputs. No collective needed here —
//     the shard boundary IS the activation boundary.
//
//   COL shard  (split the INPUT dim)   Wo, down
//     Each rank owns a contiguous band of input columns, consumes the matching
//     band of the previous activation, and produces a FULL-width partial sum.
//     Requires an all-reduce to become the true output. This is why TP costs
//     exactly two collectives per layer: one after o_proj, one after down_proj.
//
// A row-shard followed by a col-shard is what makes that work: Wq/Wk/Wv split by
// output, Wo split by input, so the sharded QKV output feeds the sharded Wo input
// with no redistribution in between.

#include <cstddef>
#include <string>
#include <vector>

namespace sparkinfer {
namespace tp {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct Config {
    int tp_size = 1;              // number of ranks (== number of GPUs)
    std::vector<int> devices;     // CUDA device ordinals, devices[rank]

    // tp_size == 1 must be indistinguishable from the pre-TP single-GPU path:
    // no collectives, no sharding, no extra buffers. That is what makes TP
    // safe to land before it is validated — the default build cannot regress.
    bool enabled() const { return tp_size > 1; }
};

// A contiguous half-open band [offset, offset + extent) of some dimension.
struct Band {
    int offset = 0;
    int extent = 0;

    int end() const { return offset + extent; }
    bool empty() const { return extent <= 0; }
    bool operator==(const Band& o) const { return offset == o.offset && extent == o.extent; }
};

// Split `total` into `tp_size` bands and return rank `rank`'s.
//
// Requires exact divisibility. A ragged split is representable but poisonous for
// TP: every rank would need its own kernel launch geometry, every collective
// would need a non-uniform layout, and CUDA-graph capture would differ per rank.
// Callers check divisible() up front and fail loudly rather than limp.
Band even_band(int total, int tp_size, int rank);

bool divisible(int total, int tp_size);

// ---------------------------------------------------------------------------
// Derived per-rank dimensions
// ---------------------------------------------------------------------------

// Everything a rank needs to size its own buffers. Produced once at load time
// from the model config; the forward reads these and never divides again.
struct ShardDims {
    int tp_size = 1;
    int rank = 0;

    // Attention. head_dim is NEVER sharded — a head is the atomic unit, because
    // softmax is over a whole head.
    int n_q_heads_total = 0;
    int n_q_heads = 0;            // this rank's share
    int head_dim = 0;
    int q_rows = 0;               // n_q_heads * head_dim

    // KV heads. See kv_replicated below — this is the subtle one.
    int n_kv_heads_total = 0;
    int n_kv_heads = 0;
    int kv_rows = 0;
    bool kv_replicated = false;

    // MoE / FFN
    int n_experts_total = 0;
    int n_experts = 0;            // this rank's share of routed experts
    Band expert_band;             // which expert indices this rank owns
    int moe_ffn_total = 0;
    int moe_ffn = 0;              // per-expert intermediate width, if ffn-sharded
    bool experts_sharded = false; // false => experts replicated, ffn sharded instead

    // Dense FFN (Qwythos / dense_ffn configs): always ffn-width sharded.
    int dense_ffn_total = 0;
    int dense_ffn = 0;

    // lm_head / vocab
    int vocab_total = 0;
    int vocab = 0;
    Band vocab_band;

    int hidden = 0;               // never sharded: it is the collective's width
};

// Human-readable reason a config cannot be sharded. Empty string == fine.
// Returned rather than thrown so the loader can print every problem at once
// instead of dying on the first.
struct ShardError {
    std::string message;
    bool ok() const { return message.empty(); }
};

// Inputs the shard math needs, decoupled from Qwen35Config so K3 (or anything
// else) can feed the same layer without dragging a Qwen header in.
struct ModelShape {
    int hidden = 0;
    int n_q_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int vocab = 0;
    int n_experts = 0;            // 0 for a dense model
    int moe_ffn = 0;
    int dense_ffn = 0;            // 0 unless dense_ffn
};

// Derive rank `rank`'s dimensions, or explain why the shape cannot be split.
//
// KV-HEAD REPLICATION — the trap this function exists to close. Grouped-query
// attention often has FEWER kv heads than ranks: Qwen3.6 has 2 kv heads, and a
// Kimi K3 GGUF stores MLA as MQA with exactly 1. At tp_size 8 there is no way to
// give each rank a whole kv head, and splitting one across ranks is incorrect —
// the group's keys and values must be visible to every query head that attends
// through them. So when n_kv_heads < tp_size we REPLICATE the kv projections on
// every rank (kv_replicated = true) and shard only the query side. That costs a
// little duplicated weight and KV cache, and it is the only correct option;
// silently computing n_kv_heads / tp_size == 0 would produce a model with no
// keys at all on most ranks.
ShardError shard_dims(const ModelShape& shape, const Config& cfg, int rank,
                      ShardDims* out);

// ---------------------------------------------------------------------------
// Weight slicing
// ---------------------------------------------------------------------------

// A weight matrix is stored [rows, cols] row-major. These describe which
// elements rank `rank` must copy out of the full tensor, as a element-count
// offset plus a count, so the loader can memcpy without knowing the shard rule.
struct Slice {
    std::size_t src_offset = 0;   // in ELEMENTS, not bytes
    std::size_t count = 0;
    int rows = 0;                 // shape of the resulting shard
    int cols = 0;
};

// Row shard of a [rows, cols] row-major matrix: rank owns a contiguous block of
// rows, so the slice is one contiguous run. Used for the OUTPUT-split weights.
Slice row_shard(int rows, int cols, int tp_size, int rank);

// Column shard of a [rows, cols] row-major matrix: rank owns a band of columns,
// which is NOT contiguous — every row contributes `cols/tp_size` elements. The
// returned Slice describes the bounding region; use col_shard_row_offset() per
// row to copy. Kept explicit rather than hidden behind a strided memcpy so the
// discontiguity is impossible to forget.
Slice col_shard(int rows, int cols, int tp_size, int rank);

// Element offset, within the FULL matrix, of rank `rank`'s band on row `row`.
std::size_t col_shard_row_offset(int row, int cols, int tp_size, int rank);

// Per-row element count of a column shard.
int col_shard_row_extent(int cols, int tp_size);

// ---------------------------------------------------------------------------
// Collective points
// ---------------------------------------------------------------------------

// Where a TP forward must all-reduce. Enumerated so the forward's collective
// calls can be matched against the shard plan in a test rather than by reading
// 2400 lines of decode path.
enum class ReducePoint {
    AfterAttnOut,   // Wo is col-sharded -> partial sums, width = hidden
    AfterFfnDown,   // down is col-sharded -> partial sums, width = hidden
};

// Payload of one all-reduce, in elements. bf16 => 2 bytes each.
// At hidden 7168 that is 14 KiB per collective, 2 per layer, 93 layers => 186
// collectives per decoded token. Latency, not bandwidth, is the cost.
std::size_t reduce_elems(ReducePoint point, const ShardDims& d, int n_tokens);

// Total collectives per forward pass over `n_layers` layers.
int reduce_count_per_token(int n_layers);

}  // namespace tp
}  // namespace sparkinfer
