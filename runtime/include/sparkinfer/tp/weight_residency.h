#pragma once
// Which BYTES of a GGUF tensor land on which rank. The layer between weight_plan.h
// (which rule applies) and the upload (cudaMemcpy). DELIBERATELY CUDA-FREE so the
// byte arithmetic is unit-testable without a device — see
// runtime/tests/tp_weight_residency_cpu_test.cpp.
//
// weight_plan.h answers "row, column, expert, or replicate". That is not enough to
// copy anything, because of one fact about GGUF that quietly breaks the obvious
// implementation:
//
//   QUANTISATION BLOCKS RUN ALONG ne0, AND ne0 IS THE CONTRACTED (INPUT) DIM.
//
// GGUF stores dims in ggml ne order with ne0 fastest. For a matmul weight that is
// [in, out] — ne0 = input width, ne1 = output width — so memory is ne1 rows of ne0
// values each, and each row is tiled into blocks of `block_elems` values packed
// into `block_bytes` bytes. Therefore:
//
//   RowShard    splits the OUTPUT dim = ne1 = whole memory rows.
//               One contiguous byte range. No alignment constraint.
//   ExpertShard splits ne2 = whole (ne0 x ne1) matrices.
//               One contiguous byte range. No alignment constraint.
//   ColShard    splits the INPUT dim = ne0 = a sub-range WITHIN EVERY ROW.
//               ne1 separate byte ranges (a strided copy), and the cut must land
//               on a block boundary or it slices a quantised block in half.
//
// A block cut in half is not recoverable. A block is a scale plus packed
// codepoints; half a block has no scale and no aligned codepoint stream, so
// dequantising it yields noise rather than an error. Nothing downstream can detect
// it: the shape is right, the byte count is right, and the model stays fluent.
//
// So this planner NEVER cuts elementwise on a quantised axis. It bands over BLOCKS
// and converts back to elements, which makes every cut legal by construction. The
// price is that bands can differ by one block between ranks when the block count
// does not divide evenly — recorded in `block_uneven`, not hidden, because it means
// per-rank shapes are not identical and any code that assumes uniformity is wrong.
//
// This is not hypothetical for K3. The leading dense layer's ffn_down has
// ne0 = 33792 = 132 blocks of 256. At tp=8 the elementwise split is 4224 values =
// 16.5 blocks. Half a block, on every rank, on a real tensor in the target model.

#include <cstddef>
#include <string>
#include <vector>

#include "sparkinfer/tp/shard.h"
#include "sparkinfer/tp/weight_plan.h"

namespace sparkinfer {
namespace tp {

// One strided copy: `n_rows` runs of `row_bytes`, advancing by src/dst stride.
// Covers all four rules — the contiguous cases are simply n_rows == 1.
struct StridedCopy {
    std::size_t src_offset = 0;   // byte offset into the full tensor's data
    std::size_t dst_offset = 0;   // byte offset into this rank's buffer
    std::size_t row_bytes  = 0;   // bytes copied per row
    std::size_t src_stride = 0;   // full-tensor row pitch
    std::size_t dst_stride = 0;   // rank-buffer row pitch (== row_bytes: packed)
    long        n_rows     = 0;

    std::size_t total_bytes() const {
        return row_bytes * static_cast<std::size_t>(n_rows > 0 ? n_rows : 0);
    }
    // True when the whole thing is one memcpy — worth knowing, because the strided
    // case for ColShard on a 3-D expert tensor is millions of tiny copies and wants
    // cudaMemcpy2D (or a repack) rather than a loop.
    bool contiguous() const {
        return n_rows <= 1 || (src_stride == row_bytes && dst_stride == row_bytes);
    }
};

enum class ResidencyError {
    Ok = 0,
    UnknownRule,       // weight_plan has no rule for this name — never guess
    UnknownType,       // ggml type absent from the block table — cannot size
    NotBlockAligned,   // a cut landed mid-block (a planner bug if it ever fires)
    BadShape,          // non-positive dim, or ne0 not a whole number of blocks
    EmptyShard,        // this rank's band is empty (more ranks than units)
};

const char* residency_error_name(ResidencyError e);

// Which logical axis got split. Named in GGUF ne terms, not "row"/"col", because
// the whole class of bug here comes from conflating the two vocabularies.
enum class SplitAxis { None = -1, Ne0_Input = 0, Ne1_Output = 1, Ne2_Expert = 2 };

const char* split_axis_name(SplitAxis a);

struct TensorResidency {
    Rule           rule  = Rule::Unknown;
    ResidencyError error = ResidencyError::Ok;

    StridedCopy copy;

    long full_ne[4] = {1, 1, 1, 1};   // shape as stored in the GGUF
    long rank_ne[4] = {1, 1, 1, 1};   // this rank's logical shape

    int  ggml_type   = 0;
    long block_elems = 0;
    long block_bytes = 0;

    SplitAxis split_axis = SplitAxis::None;
    Band      band;                // band over the split axis, in ELEMENTS
    Band      block_band;          // same band in BLOCKS (only when axis == ne0)
    bool      block_uneven = false;   // bands differ across ranks by one block
    bool      replicated_fallback = false;  // rule downgraded (KV-head case)

    std::size_t full_bytes = 0;   // whole tensor
    std::size_t rank_bytes = 0;   // what this rank stores

    std::string note;

    bool ok() const { return error == ResidencyError::Ok; }
    // True when every rank holds the whole tensor: rank_bytes counts once per rank
    // but only once toward the model. The distinction is the entire reason TP costs
    // more total memory than layer-splitting.
    bool is_replicated() const { return rule == Rule::Replicate; }
};

// Band over an axis measured in BLOCKS, returned in ELEMENTS. Every rank's band is
// a whole number of blocks; a remainder is handed to the low ranks, so extents
// differ by at most one block. `block_elems == 1` (F32/F16/BF16) degenerates to an
// ordinary even element split.
//
// This is the function that makes ColShard legal on quantised weights. Prefer it
// over even_band() for ANY axis that carries blocks.
Band block_aligned_band(long n_elems, long block_elems, int tp_size, int rank,
                        bool* uneven = nullptr);

// Plan one tensor's residency on one rank.
//
// `ne` / `n_dims` are the shape exactly as GGUF reports it (ne0 fastest).
// Refuses rather than guesses on an unknown name or an unknown quant type.
TensorResidency plan_tensor_residency(const std::string& name,
                                      const long ne[4], int n_dims, int ggml_type,
                                      const ShardDims& d);

// ---------------------------------------------------------------------------

// A tensor to plan, decoupled from GGUF so tests can build shapes by hand.
struct TensorDesc {
    std::string name;
    long ne[4] = {1, 1, 1, 1};
    int  n_dims = 2;
    int  ggml_type = 0;
};

struct ResidencyReport {
    long tensors = 0;
    long replicated = 0;
    long sharded = 0;
    long failed = 0;
    long uneven = 0;

    std::size_t bytes_model = 0;        // sum of full tensor bytes
    std::size_t bytes_rank = 0;         // what one rank holds
    std::size_t bytes_replicated = 0;   // the duplicated portion of bytes_rank
    std::size_t bytes_all_ranks = 0;    // bytes_rank * tp_size — the real cost

    std::vector<std::string> errors;    // capped, for a readable load log
};

// Plan a whole model for one rank. Any tensor that fails is counted and reported;
// the caller must treat failed > 0 as fatal. A partial upload is a silently wrong
// model, which is worse than not starting.
ResidencyReport plan_model_residency(const std::vector<TensorDesc>& tensors,
                                     const ShardDims& d,
                                     std::vector<TensorResidency>* out = nullptr);

}  // namespace tp
}  // namespace sparkinfer
