#include "sparkinfer/tp/weight_residency.h"

#include "sparkinfer/gguf.h"   // gguf_block_info / gguf_type_known

#include <cstdio>

namespace sparkinfer {
namespace tp {

const char* residency_error_name(ResidencyError e) {
    switch (e) {
        case ResidencyError::Ok:              return "ok";
        case ResidencyError::UnknownRule:     return "unknown-rule";
        case ResidencyError::UnknownType:     return "unknown-ggml-type";
        case ResidencyError::NotBlockAligned: return "not-block-aligned";
        case ResidencyError::BadShape:        return "bad-shape";
        case ResidencyError::EmptyShard:      return "empty-shard";
    }
    return "?";
}

const char* split_axis_name(SplitAxis a) {
    switch (a) {
        case SplitAxis::None:       return "none";
        case SplitAxis::Ne0_Input:  return "ne0(input)";
        case SplitAxis::Ne1_Output: return "ne1(output)";
        case SplitAxis::Ne2_Expert: return "ne2(expert)";
    }
    return "?";
}

Band block_aligned_band(long n_elems, long block_elems, int tp_size, int rank,
                        bool* uneven) {
    if (uneven) *uneven = false;
    Band b;
    if (n_elems <= 0 || block_elems <= 0 || tp_size <= 0 || rank < 0 || rank >= tp_size)
        return b;

    // A tensor whose axis is not a whole number of blocks cannot exist in a valid
    // GGUF; refuse to invent a partial trailing block.
    if (n_elems % block_elems != 0) return b;

    const long nblk = n_elems / block_elems;
    const long base = nblk / tp_size;
    const long rem  = nblk % tp_size;
    if (uneven) *uneven = (rem != 0);

    // Remainder to the low ranks: rank r gets base + (r < rem ? 1 : 0).
    const long take   = base + (rank < rem ? 1 : 0);
    const long before = base * rank + (rank < rem ? rank : rem);

    b.offset = static_cast<int>(before * block_elems);
    b.extent = static_cast<int>(take * block_elems);
    return b;
}

namespace {

// Bytes occupied by `n_elems` values of a blocked type. Caller has already checked
// the type is known and the count is block-aligned.
inline std::size_t blocked_bytes(long n_elems, long block_elems, long block_bytes) {
    return static_cast<std::size_t>(n_elems / block_elems) *
           static_cast<std::size_t>(block_bytes);
}

// Which rule actually applies, reconciling this layer with weight_plan's.
//
// weight_plan owns WHICH RULE a tensor gets and the deliberate downgrades. This
// layer owns WHETHER A SPLIT IS LEGAL, because only it knows block sizes. The two
// disagree in one direction and it matters:
//
//   plan_for() vetoes a ColShard when cols % tp_size != 0. That check is
//   ELEMENTWISE, and elements are the wrong unit — the real constraint is blocks.
//   Banding over blocks is strictly more permissive (an axis of 132 blocks splits
//   fine across 8 ranks, 33792 % 8 notwithstanding) and strictly safer (it can
//   never cut a block). So an elementwise veto is NOT adopted here.
//
// Its other outcomes are adopted, because they encode facts this layer cannot see:
// the kv-group fallback, the experts-not-divisible fallback, and the fused-qkv
// case that genuinely has no single rule.
Rule resolve_rule(const std::string& name, const TensorPlan& p, Rule table_rule,
                  const ShardDims& d, bool* fatal_unknown, std::string* note) {
    *fatal_unknown = false;

    if (p.replicated_fallback) {           // kv group cannot be split
        if (note) *note = p.note;
        return Rule::Replicate;
    }
    // A fused qkv with replicated kv is the one case with no representable rule:
    // the q half row-shards and the kv half replicates. Must stay fatal — the
    // loader has to split the tensor before sharding it.
    if (p.rule == Rule::Unknown && table_rule == Rule::RowShard && d.kv_replicated) {
        *fatal_unknown = true;
        if (note) *note = p.note;
        return Rule::Unknown;
    }
    // shard_dims() decided the expert axis will not divide, so each rank holds every
    // expert but a slice of its ffn width — which is a ColShard on ne0.
    if (table_rule == Rule::ExpertShard && !d.experts_sharded) {
        if (note) *note = "experts not divisible by tp_size — ffn-width shard instead";
        return Rule::ColShard;
    }
    // 2-D MoE: the three routed-expert matrices get a SECOND band on top of the
    // expert one. Checked after the not-divisible fallback above, so a shape that
    // could not do 1-D expert sharding never silently arrives here.
    if (d.moe_2d && table_rule == Rule::ExpertShard) {
        const Rule two_d = routed_expert_2d_rule(name);
        if (two_d != Rule::Unknown) return two_d;
    }
    return table_rule;
}

}  // namespace

TensorResidency plan_tensor_residency(const std::string& name,
                                      const long ne[4], int n_dims, int ggml_type,
                                      const ShardDims& d) {
    TensorResidency r;
    r.ggml_type = ggml_type;
    for (int i = 0; i < 4; ++i) {
        r.full_ne[i] = (i < n_dims && ne[i] > 0) ? ne[i] : 1;
        r.rank_ne[i] = r.full_ne[i];
    }

    r.rule = rule_for(name);
    if (r.rule == Rule::Unknown) {
        r.error = ResidencyError::UnknownRule;
        r.note  = "no shard rule for '" + name + "' — refusing to guess";
        return r;
    }

    if (!gguf_type_known(ggml_type)) {
        r.error = ResidencyError::UnknownType;
        char buf[96];
        std::snprintf(buf, sizeof(buf), "ggml type %d absent from the block table",
                      ggml_type);
        r.note = buf;
        return r;
    }
    gguf_block_info(ggml_type, r.block_bytes, r.block_elems);

    const long ne0 = r.full_ne[0], ne1 = r.full_ne[1], ne2 = r.full_ne[2];
    const long ne3 = r.full_ne[3];
    if (ne0 <= 0 || ne1 <= 0 || ne2 <= 0 || ne3 <= 0) {
        r.error = ResidencyError::BadShape;
        r.note  = "non-positive dim";
        return r;
    }
    // Blocks tile ne0 only. A GGUF tensor whose ne0 is not a whole number of blocks
    // is malformed; sizing it would silently truncate.
    if (ne0 % r.block_elems != 0) {
        r.error = ResidencyError::BadShape;
        char buf[128];
        std::snprintf(buf, sizeof(buf),
                      "ne0=%ld is not a multiple of block_elems=%ld", ne0, r.block_elems);
        r.note = buf;
        return r;
    }

    const std::size_t row_bytes_full =
        blocked_bytes(ne0, r.block_elems, r.block_bytes);
    const long        rows_full      = ne1 * ne2 * ne3;
    r.full_bytes = row_bytes_full * static_cast<std::size_t>(rows_full);

    // TP=1: one code path with the multi-GPU build, so the single-GPU case cannot
    // diverge. Everything is "replicated" because there is one rank.
    if (d.tp_size <= 1) {
        r.rule = Rule::Replicate;
    } else {
        // plan_for() takes (rows=output, cols=input) in MATMUL vocabulary, which is
        // (ne1, ne0) in GGUF vocabulary. Getting this transposition wrong is the exact
        // confusion this file exists to contain, so it is spelled out at the call.
        const TensorPlan tp_plan = plan_for(name, /*rows=*/static_cast<int>(ne1),
                                            /*cols=*/static_cast<int>(ne0), d);
        r.replicated_fallback = tp_plan.replicated_fallback;
        bool fatal = false;
        std::string note;
        r.rule = resolve_rule(name, tp_plan, r.rule, d, &fatal, &note);
        if (!note.empty()) r.note = note;
        if (fatal || r.rule == Rule::Unknown) {
            r.error = ResidencyError::UnknownRule;
            if (r.note.empty()) r.note = "weight_plan could not resolve a single rule";
            return r;
        }
    }

    switch (r.rule) {
        case Rule::Replicate: {
            r.split_axis = SplitAxis::None;
            r.copy.src_offset = 0;
            r.copy.dst_offset = 0;
            r.copy.row_bytes  = r.full_bytes;
            r.copy.src_stride = r.full_bytes;
            r.copy.dst_stride = r.full_bytes;
            r.copy.n_rows     = 1;
            break;
        }

        case Rule::RowShard: {
            // Split ne1 = output width = whole memory rows. Contiguous, unaligned-safe.
            r.split_axis = SplitAxis::Ne1_Output;
            // block_elems = 1: ne1 counts whole memory rows, so there is no
            // alignment constraint — but the remainder must still be DISTRIBUTED
            // rather than dropped. even_band() truncates (per = total / tp), which
            // silently loses up to tp_size-1 output rows: a real weight slice that
            // no rank holds and nothing reports.
            r.band = block_aligned_band(ne1, 1, d.tp_size, d.rank, &r.block_uneven);
            if (r.band.empty()) {
                r.error = ResidencyError::EmptyShard;
                char buf[128];
                std::snprintf(buf, sizeof(buf),
                              "ne1=%ld cannot be split across %d ranks", ne1, d.tp_size);
                r.note = buf;
                return r;
            }
            r.rank_ne[1] = r.band.extent;
            // ne2/ne3 repeat the whole ne1 axis, so a banded ne1 is only contiguous
            // when there is exactly one such matrix. Treat higher dims as rows.
            const long mats = ne2 * ne3;
            r.copy.row_bytes  = row_bytes_full * static_cast<std::size_t>(r.band.extent);
            r.copy.src_offset = row_bytes_full * static_cast<std::size_t>(r.band.offset);
            r.copy.dst_offset = 0;
            r.copy.src_stride = row_bytes_full * static_cast<std::size_t>(ne1);
            r.copy.dst_stride = r.copy.row_bytes;
            r.copy.n_rows     = mats;
            break;
        }

        case Rule::ColShard: {
            // Split ne0 = input width = a window inside every row. THE aligned case.
            r.split_axis = SplitAxis::Ne0_Input;
            bool uneven = false;
            r.band = block_aligned_band(ne0, r.block_elems, d.tp_size, d.rank, &uneven);
            r.block_uneven = uneven;
            if (r.band.empty()) {
                r.error = ResidencyError::EmptyShard;
                char buf[160];
                std::snprintf(buf, sizeof(buf),
                              "ne0=%ld is only %ld block(s) of %ld — cannot split across "
                              "%d ranks; replicate this tensor or reduce tp",
                              ne0, ne0 / r.block_elems, r.block_elems, d.tp_size);
                r.note = buf;
                return r;
            }
            if (r.band.offset % r.block_elems != 0 || r.band.extent % r.block_elems != 0) {
                r.error = ResidencyError::NotBlockAligned;   // planner bug if hit
                r.note  = "block_aligned_band returned an unaligned band";
                return r;
            }
            r.block_band.offset = r.band.offset / static_cast<int>(r.block_elems);
            r.block_band.extent = r.band.extent / static_cast<int>(r.block_elems);
            r.rank_ne[0] = r.band.extent;

            r.copy.row_bytes  = blocked_bytes(r.band.extent, r.block_elems, r.block_bytes);
            r.copy.src_offset = blocked_bytes(r.band.offset, r.block_elems, r.block_bytes);
            r.copy.dst_offset = 0;
            r.copy.src_stride = row_bytes_full;
            r.copy.dst_stride = r.copy.row_bytes;
            r.copy.n_rows     = rows_full;
            if (uneven) {
                char buf[160];
                std::snprintf(buf, sizeof(buf),
                              "block-aligned uneven split: %ld blocks over %d ranks, "
                              "this rank holds %d",
                              ne0 / r.block_elems, d.tp_size, r.block_band.extent);
                r.note = buf;
            }
            break;
        }

        case Rule::ExpertShard: {
            // Split ne2 = whole per-expert matrices. Contiguous, unaligned-safe.
            //
            // For K3's attn_k_b / attn_v_b the split axis is also ne2 (head-major),
            // which is why weight_plan gives them ExpertShard: same mechanics, the
            // banded axis just counts heads rather than experts.
            r.split_axis = SplitAxis::Ne2_Expert;
            r.band = block_aligned_band(ne2, 1, d.tp_size, d.rank, &r.block_uneven);
            if (r.band.empty()) {
                r.error = ResidencyError::EmptyShard;
                char buf[128];
                std::snprintf(buf, sizeof(buf),
                              "ne2=%ld cannot be split across %d ranks", ne2, d.tp_size);
                r.note = buf;
                return r;
            }
            r.rank_ne[2] = r.band.extent;
            const std::size_t mat_bytes = row_bytes_full * static_cast<std::size_t>(ne1);
            r.copy.row_bytes  = mat_bytes * static_cast<std::size_t>(r.band.extent);
            r.copy.src_offset = mat_bytes * static_cast<std::size_t>(r.band.offset);
            r.copy.dst_offset = 0;
            r.copy.src_stride = r.copy.row_bytes;
            r.copy.dst_stride = r.copy.row_bytes;
            r.copy.n_rows     = ne3;
            break;
        }

        case Rule::ExpertRow2D:
        case Rule::ExpertCol2D: {
            // 2-D MoE. Both bands come from ShardDims rather than being re-derived
            // here, for the reason the KDA loader states: the kernel reads its
            // geometry from the same struct, and a residency that re-divided could
            // disagree with it. A disagreement here does not fault — it silently
            // feeds the GEMV another expert's weights.
            const Band eb = d.expert_band;      // over ne2
            const Band fb = d.moe_ffn_band;     // over ne1 (row) or ne0 (col)
            if (ne3 != 1) {
                r.error = ResidencyError::BadShape;
                r.note  = "2-D MoE expects ne3=1 on a routed-expert tensor";
                return r;
            }
            if (eb.extent <= 0 || fb.extent <= 0 || eb.end() > ne2) {
                r.error = ResidencyError::EmptyShard;
                r.note  = "2-D MoE band empty or past the expert axis";
                return r;
            }
            // One expert's matrix, in bytes. Both cases stride by this to walk the
            // expert axis; it is also the ne1 row-run for the col case.
            const std::size_t mat_bytes = row_bytes_full * static_cast<std::size_t>(ne1);
            r.rank_ne[2] = eb.extent;

            if (r.rule == Rule::ExpertRow2D) {
                // ne1 = the FFN output rows. Whole memory rows, so no alignment
                // constraint — the band is a contiguous run inside each expert.
                r.split_axis = SplitAxis::Ne1_Output;
                if (fb.end() > ne1) {
                    r.error = ResidencyError::BadShape;
                    r.note  = "2-D MoE ffn band past ne1 on a gate/up tensor";
                    return r;
                }
                r.rank_ne[1] = fb.extent;
                r.copy.row_bytes  = row_bytes_full * static_cast<std::size_t>(fb.extent);
                r.copy.src_offset = mat_bytes * static_cast<std::size_t>(eb.offset) +
                                    row_bytes_full * static_cast<std::size_t>(fb.offset);
                r.copy.src_stride = mat_bytes;                 // next expert
                r.copy.n_rows     = eb.extent;
            } else {
                // ne0 = the FFN width this tensor CONTRACTS over, which is the axis
                // quant blocks tile. The cut must be a whole number of blocks; the
                // group count is validated in moe_2d_dims, and this is the assertion
                // that it stayed true for THIS tensor's block size.
                r.split_axis = SplitAxis::Ne0_Input;
                if (fb.end() > ne0) {
                    r.error = ResidencyError::BadShape;
                    r.note  = "2-D MoE ffn band past ne0 on a down tensor";
                    return r;
                }
                if (fb.offset % r.block_elems != 0 || fb.extent % r.block_elems != 0) {
                    r.error = ResidencyError::NotBlockAligned;
                    char buf[160];
                    std::snprintf(buf, sizeof(buf),
                                  "2-D MoE ffn band [%d,%d) is not block-aligned to "
                                  "%ld elements", fb.offset, fb.end(), r.block_elems);
                    r.note = buf;
                    return r;
                }
                r.rank_ne[0] = fb.extent;
                r.copy.row_bytes  = blocked_bytes(fb.extent, r.block_elems, r.block_bytes);
                r.copy.src_offset = mat_bytes * static_cast<std::size_t>(eb.offset) +
                                    static_cast<std::size_t>(fb.offset / r.block_elems) *
                                        static_cast<std::size_t>(r.block_bytes);
                r.copy.src_stride = row_bytes_full;            // next output row
                // The expert band is contiguous, so its ne1 rows and the experts
                // themselves form ONE uniform run of rows — 1.6M of them at eg=2,
                // which is a single cudaMemcpy2D rather than the "millions of tiny
                // copies" the StridedCopy comment warns about.
                r.copy.n_rows     = static_cast<long>(ne1) * eb.extent;
            }
            r.copy.dst_offset = 0;
            r.copy.dst_stride = r.copy.row_bytes;              // packed
            break;
        }

        case Rule::Unknown:
            r.error = ResidencyError::UnknownRule;
            return r;
    }

    r.rank_bytes = r.copy.total_bytes();

    // Every byte of the tensor must be accounted for exactly once across the ranks
    // (replication aside). This catches an off-by-one band or a stride/pitch mixup
    // at plan time rather than as a quality regression.
    if (r.rule == Rule::Replicate && r.rank_bytes != r.full_bytes) {
        r.error = ResidencyError::BadShape;
        r.note  = "replicated tensor did not size to the full tensor";
    }
    if (r.rank_bytes == 0) {
        r.error = ResidencyError::EmptyShard;
        if (r.note.empty()) r.note = "zero bytes for this rank";
    }
    return r;
}

ResidencyReport plan_model_residency(const std::vector<TensorDesc>& tensors,
                                     const ShardDims& d,
                                     std::vector<TensorResidency>* out) {
    ResidencyReport rep;
    if (out) out->clear();

    for (const auto& t : tensors) {
        const TensorResidency r =
            plan_tensor_residency(t.name, t.ne, t.n_dims, t.ggml_type, d);
        if (out) out->push_back(r);

        ++rep.tensors;
        if (!r.ok()) {
            ++rep.failed;
            if (rep.errors.size() < 32) {
                rep.errors.push_back(t.name + ": " +
                                     residency_error_name(r.error) +
                                     (r.note.empty() ? "" : " — " + r.note));
            }
            continue;
        }
        if (r.block_uneven) ++rep.uneven;
        rep.bytes_model += r.full_bytes;
        rep.bytes_rank  += r.rank_bytes;
        if (r.is_replicated()) {
            ++rep.replicated;
            rep.bytes_replicated += r.rank_bytes;
        } else {
            ++rep.sharded;
        }
    }
    rep.bytes_all_ranks =
        rep.bytes_rank * static_cast<std::size_t>(d.tp_size > 0 ? d.tp_size : 1);
    return rep;
}

}  // namespace tp
}  // namespace sparkinfer
