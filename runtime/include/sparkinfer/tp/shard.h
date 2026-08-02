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

// ---------------------------------------------------------------------------
// Kimi K3's KDA (gated delta net) attention — head-parallel shard math
// ---------------------------------------------------------------------------
//
// WHY THIS IS A SEPARATE PLAN AND NOT ShardDims.
//
// K3 is a HYBRID: 69 of its 93 layers are KDA (recurrent, context-independent)
// and 24 are MLA (full attention). ShardDims models one attention shape per
// model, which is exactly right for Qwen-style stacks and cannot express "shard
// the KDA layers, replicate the MLA ones".
//
// And that split is deliberate, not an implementation shortcut:
//
//   KDA  every head owns a private recurrent state (conv window + delta matrix)
//        and never reads another head's, so both the weights AND the state
//        divide by tp_size. One all-reduce at hidden after attn_output.
//
//   MLA  is MQA: all n_q_heads attend over ONE shared latent KV cache. Sharding
//        query heads divides the FLOPs but NOT the cache read, because every
//        rank still walks the whole cache. It buys a fraction of the cost and
//        pays a full collective for it. LEAVE MLA REPLICATED.
//
// Under ShardPolicy::ExpertsOnly the KDA projections are replicated, so at
// tp_size 8 all eight ranks read the same ~468 MB of Q8_0 attention weights per
// KDA layer every token. That is the redundancy this plan removes.

struct KdaShape {
    int hidden = 0;         // KimiK3Config::hidden
    int n_heads = 0;        // KimiK3Config::n_q_heads
    int head_dim = 0;       // KimiK3Config::kda_head_dim
    int conv_kernel = 0;    // KimiK3Config::kda_conv_kernel
};

// One rank's slice of a KDA layer. Every extent here is PER RANK; the _total
// fields keep the full dimension so a caller can assert it did not lose any.
struct KdaShardDims {
    int tp_size = 1;
    int rank = 0;

    int hidden = 0;         // NEVER sharded — it is the collective's width
    int head_dim = 0;       // NEVER sharded — a head is the atomic unit

    int n_heads_total = 0;
    int n_heads = 0;
    Band head_band;         // this rank's heads, in head units

    int qkv_total = 0;      // n_heads_total * head_dim
    int qkv = 0;            // n_heads * head_dim
    Band qkv_band;          // head_band scaled by head_dim — the INVARIANT below

    int conv_kernel = 0;
    // Per-rank recurrent state, in floats. These are the sizes kimi_k3_alloc_state
    // must use once the layer is sharded; allocating the full-width version and
    // indexing it with a rank-local head id reads another rank's state and returns
    // fluent garbage, which is why they are derived here and not inline.
    long conv_state_elems = 0;    // (conv_kernel - 1) * qkv, per conv stream (q/k/v)
    long delta_state_elems = 0;   // head_dim * head_dim * n_heads
};

// THE INVARIANT: qkv_band == head_band scaled by head_dim.
//
// The KDA branch addresses the same rank slice two ways — by head (l2_norm_heads,
// kda_decay_gate, kda_decode_step, kda_gate_out) and by qkv row (the four
// projections, the conv window, ssm_dt_bias). If those two disagree by even one
// head the layer still runs at full speed and emits subtly wrong tokens, because
// nothing bounds-checks a head id against a qkv offset. Checked in
// kda_shard_dims and asserted directly in the CPU test.
ShardError kda_shard_dims(const KdaShape& shape, const Config& cfg, int rank,
                          KdaShardDims* out);

// WHICH TENSOR TAKES WHICH RULE IS **NOT** DECLARED HERE.
//
// That belongs to weight_plan.h — "the rule for every tensor is declared in one
// table", and a second table in this header would be free to disagree with it,
// which is the precise failure both files exist to prevent. The KDA rows of that
// table are the plan; this header only supplies the per-rank extents and offsets
// they imply.
//
// REPLICATE-AND-OFFSET, for the small per-channel vectors.
//
// Four KDA tensors stay REPLICATED at full width even under a head-parallel
// shard, and the forward indexes them at this rank's band instead:
//
//   ssm_dt.bias      f32 [qkv]                  + qkv_band.offset
//   ssm_a            f32 [n_heads]              + head_band.offset
//   ssm_conv1d_{q,k,v}  f32 [conv_kernel,1,qkv] + qkv_band.offset * conv_kernel
//   ssm_f_a.weight   [head_dim, hidden]         no offset — needed WHOLE
//
// The first three are channel-major (kda_conv_step_kernel reads
// `w + c * d_conv`), so a channel band is contiguous and a pointer offset is the
// whole of the change. Together they are ~250 KB per layer against the ~468 MB of
// projections, so sharding them would buy nothing and cost a rule that does not
// fit any existing axis. ssm_f_a is different in kind: its OUTPUT is the shared
// head_dim-wide vector every rank's ssm_f_b shard consumes, so it must stay whole.
//
// What makes this safe rather than a landmine is that the offsets come from the
// bands above, which are the SAME bands the projections were sharded with. A
// forward that offsets by anything else reads another rank's heads and the model
// stays fluent — so the offsets are pinned in the CPU test, not left to the
// call site.

// ---------------------------------------------------------------------------
// MLA head-parallel shard — the 24 MLA layers
// ---------------------------------------------------------------------------
//
// WHY MLA AND NOT KDA, when KDA is 69 of the 93 layers. Measured on the shipped
// build (nsys, 8x H200, ctx 131072), as a share of all GPU kernel time:
//
//     mla_decode_attn_hbatch   36.9%   1344 inst   1.43 ms each   24 layers
//     kda_decode_step_smem      6.6%   3864 inst     89 us each   69 layers
//
// MLA costs 16x more per layer per rank. Both bands cost one all-reduce per
// layer, so the two decompose completely differently once the collective is
// priced in (NCCL is 8.1% of GPU time for 92 collectives, ~0.088%/collective):
//
//              compute saved      collectives added        net
//     KDA          +5.8%          -6.1%  (69 more)      -0.3%   LOSES MONEY
//     MLA         +32.3%          -2.1%  (24 more)     +30.2%
//
// Sharding the KDA two-thirds costs more in collectives than it saves in
// compute. That is why this exists alongside KdaShardDims rather than replacing
// it, and why the MLA band is the one wired on by default.
//
// AND WHY THE 8x IS REAL, which I got wrong once. MLA is MQA: 96 query heads
// share ONE latent KV cache, so it is tempting to say head-sharding divides the
// FLOPs but not the memory traffic, since every rank still walks the whole
// cache. That is wrong for THIS kernel. mla_decode_attn_hbatch stages 12 heads
// per block, so 96 heads means EIGHT passes over the 302 MB latent cache per
// layer. One band of 12 heads is ONE pass. The traffic divides by 8 because the
// unsharded kernel was already reading the cache eight times.
struct MlaShape {
    int hidden = 0;
    int n_q_heads = 0;          // 96
    int key_length_mla = 0;     // qk_nope + rope_dim, the q_b output stride (192)
    int value_length_mla = 0;   // 128
    int qk_nope = 0;            // 128
    int kv_lora_rank = 0;       // 512
};

// Bands for every MLA tensor a head shard touches. All four derived bands are
// the head band scaled by that tensor's per-head stride, so they cannot drift
// apart — the same invariant KdaShardDims enforces between head and qkv.
struct MlaShardDims {
    int tp_size = 1;
    int rank = 0;
    int hidden = 0;
    int n_heads_total = 0;
    int n_heads = 0;            // this rank's share
    Band head_band;

    // attn_q_b / attn_q_dense OUTPUT rows: [n_q_heads * key_length_mla, ...]
    Band q_rows_band;
    // attn_gate OUTPUT rows, and attn_output INPUT columns:
    // [n_q_heads * value_length_mla, ...]
    Band v_rows_band;
    // attn_k_b: [qk_nope, kv_lora, n_head], head SLOWEST -> contiguous per head.
    Band wk_b_band;
    // attn_v_b: [kv_lora, v_dim, n_head], head SLOWEST -> contiguous per head.
    Band wv_b_band;
};

// REPLICATED under an MLA head shard, and this is the whole point of MQA:
//
//   attn_kv_a_mqa / attn_kv_a_norm   the latent KV projection — shared by every
//                                    head, so every rank needs it whole
//   the MLA KV cache itself          likewise; each rank keeps the full cache
//   attn_q_a / attn_q_a_norm         the q_lora down-projection feeds ALL heads
//
// Only the per-head tensors band: q_b (up), k_b, v_b, gate, and the output
// projection's input columns.
ShardError mla_shard_dims(const MlaShape& shape, const Config& cfg, int rank,
                          MlaShardDims* out);

// Collectives per token under K3's hybrid plan.
//
// ExpertsOnly today: one per MoE layer (the expert partial at expert_latent).
// With KDA sharded: that, plus one per KDA layer (the attn_output partial at
// hidden). With MLA sharded: plus one per MLA layer, same partial, same width.
int k3_reduce_count_per_token(int n_kda_layers, int n_moe_layers, bool kda_sharded);
int k3_reduce_count_per_token(int n_kda_layers, int n_mla_layers, int n_moe_layers,
                              bool kda_sharded, bool mla_sharded);

}  // namespace tp
}  // namespace sparkinfer
