// Tensor-parallel shard math. No CUDA — see the header for why.

#include "sparkinfer/tp/shard.h"

#include <sstream>

namespace sparkinfer {
namespace tp {

bool divisible(int total, int tp_size) {
    return tp_size > 0 && total > 0 && (total % tp_size) == 0;
}

Band even_band(int total, int tp_size, int rank) {
    if (tp_size <= 0 || rank < 0 || rank >= tp_size) return {};
    const int per = total / tp_size;          // callers check divisible() first
    return Band{per * rank, per};
}

namespace {

void add_err(std::ostringstream& os, bool& any, const std::string& msg) {
    if (any) os << "; ";
    os << msg;
    any = true;
}

}  // namespace

ShardError shard_dims(const ModelShape& shape, const Config& cfg, int rank,
                      ShardDims* out) {
    std::ostringstream os;
    bool any = false;

    if (!out) return ShardError{"shard_dims: null out"};
    if (cfg.tp_size <= 0) return ShardError{"tp_size must be >= 1"};
    if (rank < 0 || rank >= cfg.tp_size) {
        os << "rank " << rank << " outside [0," << cfg.tp_size << ")";
        return ShardError{os.str()};
    }
    if (!cfg.devices.empty() && static_cast<int>(cfg.devices.size()) != cfg.tp_size) {
        os << "devices.size()=" << cfg.devices.size() << " != tp_size=" << cfg.tp_size;
        return ShardError{os.str()};
    }

    ShardDims d;
    d.tp_size = cfg.tp_size;
    d.rank = rank;
    d.hidden = shape.hidden;
    d.head_dim = shape.head_dim;

    // --- TP=1: the identity shard. Every extent is the full dimension, nothing
    // is replicated, no collective is needed. Keeping this in the same code path
    // (rather than branching around the whole TP layer) is what guarantees the
    // default single-GPU build cannot diverge from the sharded one.
    if (cfg.tp_size == 1) {
        d.n_q_heads_total = d.n_q_heads = shape.n_q_heads;
        d.q_rows = shape.n_q_heads * shape.head_dim;
        d.n_kv_heads_total = d.n_kv_heads = shape.n_kv_heads;
        d.kv_rows = shape.n_kv_heads * shape.head_dim;
        d.kv_replicated = false;
        d.n_experts_total = d.n_experts = shape.n_experts;
        d.expert_band = Band{0, shape.n_experts};
        d.experts_sharded = shape.n_experts > 0;
        d.moe_ffn_total = d.moe_ffn = shape.moe_ffn;
        d.dense_ffn_total = d.dense_ffn = shape.dense_ffn;
        d.vocab_total = d.vocab = shape.vocab;
        d.vocab_band = Band{0, shape.vocab};
        *out = d;
        return {};
    }

    // --- Query heads. A head is atomic: softmax runs over a whole head, so
    // head_dim never splits. Requiring n_q_heads % tp_size == 0 is the real
    // constraint on how wide TP can go.
    d.n_q_heads_total = shape.n_q_heads;
    if (!divisible(shape.n_q_heads, cfg.tp_size)) {
        std::ostringstream m;
        m << "n_q_heads=" << shape.n_q_heads << " not divisible by tp_size=" << cfg.tp_size
          << " (a head cannot be split across ranks)";
        add_err(os, any, m.str());
    } else {
        d.n_q_heads = shape.n_q_heads / cfg.tp_size;
        d.q_rows = d.n_q_heads * shape.head_dim;
    }

    // --- KV heads. See the header comment: when there are fewer kv heads than
    // ranks we replicate rather than split, because a kv group must be entirely
    // visible to every query head attending through it. This is the path Qwen3.6
    // (2 kv heads) and a Kimi K3 GGUF (MLA stored as MQA, 1 kv head) both take at
    // tp_size 8.
    d.n_kv_heads_total = shape.n_kv_heads;
    if (shape.n_kv_heads >= cfg.tp_size) {
        if (!divisible(shape.n_kv_heads, cfg.tp_size)) {
            std::ostringstream m;
            m << "n_kv_heads=" << shape.n_kv_heads << " >= tp_size=" << cfg.tp_size
              << " but not divisible (would split a kv group)";
            add_err(os, any, m.str());
        } else {
            d.n_kv_heads = shape.n_kv_heads / cfg.tp_size;
            d.kv_replicated = false;
        }
    } else {
        d.n_kv_heads = shape.n_kv_heads;   // full set on every rank
        d.kv_replicated = true;
    }
    d.kv_rows = d.n_kv_heads * shape.head_dim;

    // --- MoE. Prefer sharding EXPERTS over sharding each expert's ffn width:
    // an expert is already a natural unit, each rank runs whole experts for the
    // tokens routed to them, and the combine folds into the existing after-FFN
    // all-reduce. Sharding ffn width instead would make every rank touch every
    // expert, multiplying weight traffic by tp_size — the opposite of the point.
    d.n_experts_total = shape.n_experts;
    d.moe_ffn_total = shape.moe_ffn;
    if (shape.n_experts > 0) {
        if (divisible(shape.n_experts, cfg.tp_size)) {
            d.experts_sharded = true;
            d.expert_band = even_band(shape.n_experts, cfg.tp_size, rank);
            d.n_experts = d.expert_band.extent;
            d.moe_ffn = shape.moe_ffn;             // full width, fewer experts
        } else if (divisible(shape.moe_ffn, cfg.tp_size)) {
            // Fallback: experts do not divide, so replicate them and split each
            // one's intermediate width instead. Correct, just more weight traffic.
            d.experts_sharded = false;
            d.expert_band = Band{0, shape.n_experts};
            d.n_experts = shape.n_experts;
            d.moe_ffn = shape.moe_ffn / cfg.tp_size;
        } else {
            std::ostringstream m;
            m << "neither n_experts=" << shape.n_experts << " nor moe_ffn=" << shape.moe_ffn
              << " divisible by tp_size=" << cfg.tp_size;
            add_err(os, any, m.str());
        }
    }

    // --- Dense FFN: only the width is available to split.
    d.dense_ffn_total = shape.dense_ffn;
    if (shape.dense_ffn > 0) {
        if (!divisible(shape.dense_ffn, cfg.tp_size)) {
            std::ostringstream m;
            m << "dense_ffn=" << shape.dense_ffn << " not divisible by tp_size=" << cfg.tp_size;
            add_err(os, any, m.str());
        } else {
            d.dense_ffn = shape.dense_ffn / cfg.tp_size;
        }
    }

    // --- Vocab / lm_head: row shard. Each rank produces logits for its own
    // vocab band; a greedy argmax then needs only each rank's local (max, index)
    // pair, not the full logit vector — so the final collective is tiny.
    d.vocab_total = shape.vocab;
    if (shape.vocab > 0) {
        if (!divisible(shape.vocab, cfg.tp_size)) {
            std::ostringstream m;
            m << "vocab=" << shape.vocab << " not divisible by tp_size=" << cfg.tp_size;
            add_err(os, any, m.str());
        } else {
            d.vocab_band = even_band(shape.vocab, cfg.tp_size, rank);
            d.vocab = d.vocab_band.extent;
        }
    }

    if (any) return ShardError{os.str()};
    *out = d;
    return {};
}

// ---------------------------------------------------------------------------

Slice row_shard(int rows, int cols, int tp_size, int rank) {
    Slice s;
    if (!divisible(rows, tp_size) || rank < 0 || rank >= tp_size) return s;
    const int per = rows / tp_size;
    s.rows = per;
    s.cols = cols;
    s.src_offset = static_cast<std::size_t>(per) * rank * cols;
    s.count = static_cast<std::size_t>(per) * cols;
    return s;
}

int col_shard_row_extent(int cols, int tp_size) {
    if (!divisible(cols, tp_size)) return 0;
    return cols / tp_size;
}

std::size_t col_shard_row_offset(int row, int cols, int tp_size, int rank) {
    const int per = col_shard_row_extent(cols, tp_size);
    if (per == 0) return 0;
    return static_cast<std::size_t>(row) * cols + static_cast<std::size_t>(per) * rank;
}

Slice col_shard(int rows, int cols, int tp_size, int rank) {
    Slice s;
    const int per = col_shard_row_extent(cols, tp_size);
    if (per == 0 || rank < 0 || rank >= tp_size) return s;
    s.rows = rows;
    s.cols = per;
    // Bounding region: first element of rank's band on row 0, through the last
    // element of its band on the final row. NOT contiguous — the caller must
    // walk rows via col_shard_row_offset(). count is the SHARD's element count,
    // deliberately not the bounding span, so a caller that treats this as a
    // single memcpy under-copies loudly instead of reading out of bounds.
    s.src_offset = static_cast<std::size_t>(per) * rank;
    s.count = static_cast<std::size_t>(rows) * per;
    return s;
}

// ---------------------------------------------------------------------------

std::size_t reduce_elems(ReducePoint point, const ShardDims& d, int n_tokens) {
    if (n_tokens <= 0) return 0;
    switch (point) {
        // Both collectives carry full-width hidden states: a column-sharded
        // weight yields a partial sum over the FULL output, which is exactly
        // what makes it an all-reduce rather than an all-gather.
        case ReducePoint::AfterAttnOut:
        case ReducePoint::AfterFfnDown:
            return static_cast<std::size_t>(n_tokens) * d.hidden;
    }
    return 0;
}

int reduce_count_per_token(int n_layers) {
    return n_layers > 0 ? n_layers * 2 : 0;   // one after attn out, one after ffn down
}

}  // namespace tp
}  // namespace sparkinfer
