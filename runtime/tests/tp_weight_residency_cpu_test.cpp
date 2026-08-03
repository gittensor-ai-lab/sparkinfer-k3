// Which BYTES of a GGUF tensor land on which rank. CPU-only: pure byte arithmetic.
//
// The property that matters most here is CONSERVATION. Summed over all ranks, a
// sharded tensor must be covered exactly once — no gap, no overlap. A gap is a
// slice of weight that no rank holds; an overlap is a slice two ranks both apply.
// Neither changes any shape, neither trips an assert at runtime, and both leave the
// model fluent. So every rule is checked by walking all 8 ranks and reassembling.
//
// The second property is BLOCK ALIGNMENT on the contracted axis, and it is not
// hypothetical — see the dense_ffn_down case below, which is a real tensor in the
// real target model whose naive elementwise split lands on half a block.

#include "sparkinfer/tp/weight_residency.h"
#include "sparkinfer/gguf.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace sparkinfer;
using namespace sparkinfer::tp;

static int g_fail = 0, g_checks = 0;

static void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) { ++g_fail; std::printf("  FAIL  %s\n", what.c_str()); }
}
static void check_eq(long got, long want, const std::string& what) {
    ++g_checks;
    if (got != want) {
        ++g_fail;
        std::printf("  FAIL  %s: got %ld want %ld\n", what.c_str(), got, want);
    }
}

// ggml type ids used below, named so the test reads as intent rather than magic.
enum : int { TY_F32 = 0, TY_Q8_0 = 8, TY_IQ2_XS = 17, TY_UNKNOWN = 200 };

static ShardDims dims8(int rank, int n_experts_total = 896, bool experts_sharded = true,
                       bool kv_replicated = false) {
    ShardDims d;
    d.tp_size = 8;
    d.rank = rank;
    d.hidden = 7168;
    d.n_experts_total = n_experts_total;
    d.experts_sharded = experts_sharded;
    d.n_experts = experts_sharded ? n_experts_total / 8 : n_experts_total;
    d.expert_band = experts_sharded ? even_band(n_experts_total, 8, rank)
                                    : Band{0, n_experts_total};
    d.kv_replicated = kv_replicated;
    d.n_kv_heads_total = kv_replicated ? 1 : 8;
    return d;
}

// Reassemble a sharded axis across all ranks: assert full coverage, no overlap.
static void check_conservation(const std::string& name, const long ne[4], int n_dims,
                              int ggml_type, int tp_size, SplitAxis want_axis,
                              long want_axis_total) {
    std::vector<char> covered(want_axis_total, 0);
    std::size_t sum_bytes = 0, full_bytes = 0;
    int next_expected_offset = 0;
    bool all_ok = true;

    for (int rank = 0; rank < tp_size; ++rank) {
        ShardDims d = dims8(rank);
        d.tp_size = tp_size;
        const TensorResidency r =
            plan_tensor_residency(name, ne, n_dims, ggml_type, d);
        if (!r.ok()) {
            std::printf("  FAIL  %s rank %d: %s — %s\n", name.c_str(), rank,
                        residency_error_name(r.error), r.note.c_str());
            ++g_fail; ++g_checks;
            all_ok = false;
            continue;
        }
        check(r.split_axis == want_axis,
              name + " rank " + std::to_string(rank) + " split axis is " +
                  split_axis_name(r.split_axis) + ", expected " +
                  split_axis_name(want_axis));

        // Bands must be contiguous and ascending across ranks — the loader relies on
        // rank r owning [offset, offset+extent) with no reordering.
        check_eq(r.band.offset, next_expected_offset,
                 name + " rank " + std::to_string(rank) + " band offset");
        next_expected_offset = r.band.end();

        for (int i = r.band.offset; i < r.band.end(); ++i) {
            if (i < 0 || i >= want_axis_total) { all_ok = false; continue; }
            if (covered[i]) { all_ok = false; }   // overlap
            covered[i] = 1;
        }
        sum_bytes += r.rank_bytes;
        full_bytes = r.full_bytes;
    }

    check_eq(next_expected_offset, want_axis_total, name + " bands must cover the axis");
    long uncovered = 0;
    for (char c : covered) if (!c) ++uncovered;
    check_eq(uncovered, 0, name + " uncovered elements on the split axis");
    check(all_ok, name + " no overlap / in-range bands");
    check_eq((long)sum_bytes, (long)full_bytes,
             name + " sum of per-rank bytes == full tensor bytes");
}

// 2-D MoE conservation, checked at the BYTE level rather than on a band index.
//
// The 1-D helper above can reason about a single split axis as an interval. A 2-D
// shard has no such interval: rank r owns a rectangle (expert band x ffn band), and
// the thing that can go wrong is precisely the arithmetic that turns that rectangle
// into a StridedCopy — a stride that should have been the expert pitch and was the
// row pitch instead still produces the right BYTE COUNT while reading the wrong
// bytes. Counting bytes therefore proves nothing; only walking the actual source
// ranges the loader will hand cudaMemcpy2D does.
//
// So this replays every rank's copy descriptor over a byte map and demands each
// byte be touched exactly once. A gap is weight no rank holds; an overlap is weight
// two ranks both apply and the all-reduce then double-counts.
static void check_conservation_2d(const std::string& name, const long ne[4],
                                  int n_dims, int ggml_type, int tp_size,
                                  int expert_groups, int n_experts, int moe_ffn,
                                  SplitAxis want_axis) {
    std::size_t full_bytes = 0, sum_bytes = 0;
    std::vector<unsigned char> hits;
    bool planned = true;

    for (int rank = 0; rank < tp_size; ++rank) {
        ShardDims d = dims8(rank, n_experts);
        d.tp_size = tp_size;
        const ShardError se = moe_2d_dims(n_experts, moe_ffn, tp_size, rank,
                                          expert_groups, /*block_elems=*/256, &d);
        if (!se.ok()) {
            std::printf("  FAIL  %s rank %d: moe_2d_dims: %s\n", name.c_str(), rank,
                        se.message.c_str());
            ++g_fail; ++g_checks; planned = false; break;
        }
        const TensorResidency r =
            plan_tensor_residency(name, ne, n_dims, ggml_type, d);
        if (!r.ok()) {
            std::printf("  FAIL  %s rank %d: %s — %s\n", name.c_str(), rank,
                        residency_error_name(r.error), r.note.c_str());
            ++g_fail; ++g_checks; planned = false; break;
        }
        check(r.split_axis == want_axis,
              name + " rank " + std::to_string(rank) + " split axis is " +
                  split_axis_name(r.split_axis));
        if (hits.empty()) {
            full_bytes = r.full_bytes;
            hits.assign(full_bytes, 0);
        }
        // The descriptor must be self-consistent before it is replayed: rank_bytes
        // is what the loader cudaMallocs, and a copy that writes more than that is
        // a heap overflow rather than a wrong answer.
        check_eq((long)r.copy.total_bytes(), (long)r.rank_bytes,
                 name + " rank " + std::to_string(rank) + " copy fills rank_bytes");
        for (long row = 0; row < r.copy.n_rows; ++row) {
            const std::size_t base = r.copy.src_offset +
                                     static_cast<std::size_t>(row) * r.copy.src_stride;
            if (base + r.copy.row_bytes > full_bytes) {
                check(false, name + " rank " + std::to_string(rank) +
                                 " copy runs past the end of the tensor");
                planned = false;
                break;
            }
            for (std::size_t b = 0; b < r.copy.row_bytes; ++b) ++hits[base + b];
        }
        sum_bytes += r.rank_bytes;
        if (!planned) break;
    }
    if (!planned) return;

    check_eq((long)sum_bytes, (long)full_bytes,
             name + " sum of per-rank bytes == full tensor bytes");
    std::size_t gaps = 0, overlaps = 0;
    for (unsigned char h : hits) {
        if (h == 0) ++gaps;
        else if (h > 1) ++overlaps;
    }
    check_eq((long)gaps, 0, name + " bytes covered by no rank");
    check_eq((long)overlaps, 0, name + " bytes covered by more than one rank");
}

int main() {
    std::printf("=== tp weight residency (GGUF bytes -> per-rank buffers) ===\n\n");

    // ---------------------------------------------------------------- band math
    std::printf("[block_aligned_band]\n");
    {
        // 132 blocks over 8 ranks: 16 each, remainder 4 to the low ranks.
        long total = 0;
        bool uneven = false;
        for (int rank = 0; rank < 8; ++rank) {
            const Band b = block_aligned_band(33792, 256, 8, rank, &uneven);
            check_eq(b.extent % 256, 0, "band extent is a whole number of blocks");
            check_eq(b.offset % 256, 0, "band offset is a whole number of blocks");
            check_eq(b.extent / 256, rank < 4 ? 17 : 16,
                     "rank " + std::to_string(rank) + " block count");
            total += b.extent;
        }
        check_eq(total, 33792, "block bands cover ne0 exactly");
        check(uneven, "132 blocks over 8 ranks is flagged uneven");

        // Even case must not be flagged.
        bool even_flag = true;
        block_aligned_band(12288, 256, 8, 0, &even_flag);
        check(!even_flag, "48 blocks over 8 ranks is not flagged uneven");

        // block_elems == 1 degenerates to a covering element split.
        long t2 = 0;
        for (int rank = 0; rank < 8; ++rank) t2 += block_aligned_band(93, 1, 8, rank).extent;
        check_eq(t2, 93, "block_elems=1 covers an indivisible axis (93 over 8)");

        // Refuse a partial trailing block rather than inventing one.
        check(block_aligned_band(300, 256, 8, 0).empty(),
              "ne0 not a whole number of blocks -> empty band");
        // Fewer blocks than ranks: some ranks get nothing, and that must be visible.
        check(block_aligned_band(256, 256, 8, 7).empty(),
              "1 block over 8 ranks leaves high ranks empty");
    }

    // -------------------------------------------------- THE case that corrupts
    std::printf("\n[K3 leading-dense ffn_down: the mid-block cut]\n");
    {
        // Real tensor: dense_ffn = 33792 is the contracted dim, hidden 7168 the output.
        // ffn_down is ColShard, so the split lands on ne0 — inside every row.
        const long ne[4] = {33792, 7168, 1, 1};

        // First, prove the naive split is illegal. This is what an elementwise
        // even_band gives, and what weight_plan's divisibility check accepts:
        const Band naive = even_band(33792, 8, 0);
        check_eq(naive.extent, 4224, "elementwise split extent");
        check(naive.extent % 256 != 0,
              "elementwise split of 33792 over 8 CUTS A BLOCK (4224 = 16.5 blocks) "
              "— this is why banding must be over blocks");

        // WHETHER THE NAIVE SPLIT CORRUPTS DEPENDS ON THE QUANT TYPE, not the shape
        // alone — and an unsloth dynamic quant picks the type per tensor, so the same
        // ffn_down can be either. Both are asserted, because a guard that only holds
        // for one of them is not a guard.
        //   256-value blocks (K-quant / IQ):  132 blocks / 8 = 16.5  -> would cut
        //    32-value blocks (Q8_0):         1056 blocks / 8 = 132   -> already legal
        check_eq(naive.extent % 32, 0, "Q8_0's 32-value blocks happen to divide evenly");

        // The 256-block case: planner must produce a legal, uneven, covering split.
        const TensorResidency r256 = plan_tensor_residency("blk.0.ffn_down.weight", ne, 2,
                                                           TY_IQ2_XS, dims8(0));
        check(r256.ok(), std::string("dense ffn_down (256-block) plans ok: ") + r256.note);
        check(r256.block_uneven,
              "256-value blocks: 132 over 8 ranks is uneven, and that is REPORTED "
              "(per-rank shapes differ, so uniform-shape assumptions are wrong)");
        check_eq(r256.band.extent % 256, 0, "256-block band stays aligned");
        check_eq(r256.band.extent / 256, 17, "rank 0 takes the extra block");
        check_conservation("blk.0.ffn_down.weight", ne, 2, TY_IQ2_XS, 8,
                           SplitAxis::Ne0_Input, 33792);

        // The 32-block case: same tensor, same tp, evenly splittable.
        const TensorResidency r0 = plan_tensor_residency("blk.0.ffn_down.weight", ne, 2,
                                                         TY_Q8_0, dims8(0));
        check(r0.ok(), std::string("dense ffn_down plans ok: ") + r0.note);
        check(r0.rule == Rule::ColShard, "dense ffn_down is ColShard");
        check(r0.split_axis == SplitAxis::Ne0_Input, "dense ffn_down splits ne0");
        check(!r0.block_uneven, "Q8_0: 1056 blocks over 8 ranks is even");
        check_eq(r0.band.extent, 4224, "Q8_0 band matches the elementwise split");
        check_eq(r0.band.extent % 32, 0, "Q8_0: band aligned to 32-value blocks");

        // A ColShard is inherently STRIDED: one range per output row, not one memcpy.
        // Getting this wrong copies a contiguous prefix of the tensor instead of a
        // column window — same byte count, completely different weights.
        check(!r0.copy.contiguous(), "ColShard copy is strided, not one memcpy");
        check_eq(r0.copy.n_rows, 7168, "ColShard has one range per output row");
        check(r0.copy.src_stride > r0.copy.row_bytes,
              "ColShard src stride is the FULL row pitch");
        check_eq((long)r0.copy.dst_stride, (long)r0.copy.row_bytes,
                 "destination is packed");

        check_conservation("blk.0.ffn_down.weight", ne, 2, TY_Q8_0, 8,
                           SplitAxis::Ne0_Input, 33792);
    }

    // ------------------------------------------------- real K3 expert tensors
    std::printf("\n[K3 routed experts: 3584 x 3072 x 896 IQ2_XS]\n");
    {
        // Verified against the real model (blk.1) earlier in this work:
        //   ffn_gate_exps / ffn_up_exps : 3584 x 3072 x 896
        //   ffn_down_exps               : 3072 x 3584 x 896
        const long gate_ne[4] = {3584, 3072, 896, 1};
        const long down_ne[4] = {3072, 3584, 896, 1};

        const TensorResidency g = plan_tensor_residency("blk.1.ffn_gate_exps.weight",
                                                        gate_ne, 3, TY_IQ2_XS, dims8(0));
        check(g.ok(), std::string("gate_exps plans ok: ") + g.note);
        check(g.rule == Rule::ExpertShard, "gate_exps is ExpertShard");
        check(g.split_axis == SplitAxis::Ne2_Expert, "gate_exps splits ne2");
        check_eq(g.rank_ne[2], 112, "896 experts / 8 ranks = 112 per rank");
        check(g.copy.contiguous(), "ExpertShard is one contiguous memcpy");
        check(!g.block_uneven, "896 over 8 is even");
        check_eq(g.block_bytes, 74, "IQ2_XS block is 74 bytes");
        check_eq(g.block_elems, 256, "IQ2_XS block is 256 values");

        // 3584/256 = 14 blocks/row * 74 B = 1036 B/row, 3072 rows/expert.
        check_eq((long)g.full_bytes, 1036L * 3072L * 896L, "gate_exps full bytes");
        check_eq((long)g.rank_bytes, 1036L * 3072L * 112L, "gate_exps per-rank bytes");

        check_conservation("blk.1.ffn_gate_exps.weight", gate_ne, 3, TY_IQ2_XS, 8,
                           SplitAxis::Ne2_Expert, 896);
        check_conservation("blk.1.ffn_down_exps.weight", down_ne, 3, TY_IQ2_XS, 8,
                           SplitAxis::Ne2_Expert, 896);

        // Sanity on scale: the three expert tensors dominate the 802 GiB model.
        const TensorResidency u = plan_tensor_residency("blk.1.ffn_up_exps.weight",
                                                        gate_ne, 3, TY_IQ2_XS, dims8(0));
        const TensorResidency dn = plan_tensor_residency("blk.1.ffn_down_exps.weight",
                                                         down_ne, 3, TY_IQ2_XS, dims8(0));
        const double per_layer_gib =
            (double)(g.full_bytes + u.full_bytes + dn.full_bytes) / (1024.0*1024*1024);
        std::printf("  note: one MoE layer's experts = %.2f GiB full, %.2f GiB per rank\n",
                    per_layer_gib, per_layer_gib / 8.0);
        check(per_layer_gib > 7.0 && per_layer_gib < 10.0,
              "one MoE layer's experts land in the expected GiB range");
    }

    // ------------------------------------------------------------ row shard
    std::printf("\n[RowShard: whole memory rows, no alignment constraint]\n");
    {
        // attn_q: [hidden, n_q_heads*head_dim] = [7168, 12288]
        const long ne[4] = {7168, 12288, 1, 1};
        const TensorResidency r = plan_tensor_residency("blk.5.attn_q.weight", ne, 2,
                                                        TY_Q8_0, dims8(3));
        check(r.ok(), std::string("attn_q plans ok: ") + r.note);
        check(r.rule == Rule::RowShard, "attn_q is RowShard");
        check(r.split_axis == SplitAxis::Ne1_Output, "attn_q splits ne1");
        check_eq(r.rank_ne[1], 1536, "12288 / 8 = 1536 output rows");
        check_eq(r.rank_ne[0], 7168, "RowShard leaves ne0 whole");
        check(r.copy.contiguous(), "RowShard of a 2-D tensor is one memcpy");
        // Row 3's slice starts a quarter of the way in, in bytes.
        const std::size_t row_b = (7168 / 32) * 34;   // Q8_0
        check_eq((long)r.copy.src_offset, (long)(row_b * 1536 * 3), "rank 3 byte offset");
        check_conservation("blk.5.attn_q.weight", ne, 2, TY_Q8_0, 8,
                           SplitAxis::Ne1_Output, 12288);

        // An output dim that does not divide must still be covered, not truncated.
        const long odd_ne[4] = {7168, 12289, 1, 1};
        long got = 0;
        for (int rank = 0; rank < 8; ++rank) {
            const TensorResidency rr = plan_tensor_residency("blk.5.attn_q.weight",
                                                             odd_ne, 2, TY_Q8_0, dims8(rank));
            check(rr.ok(), "indivisible RowShard still plans");
            got += rr.rank_ne[1];
        }
        check_eq(got, 12289, "indivisible output dim is covered, not truncated");
    }

    // ------------------------------------------------------------- replicate
    std::printf("\n[Replicate and TP=1]\n");
    {
        const long ne[4] = {7168, 163840, 1, 1};
        const TensorResidency r = plan_tensor_residency("token_embd.weight", ne, 2,
                                                        TY_Q8_0, dims8(2));
        check(r.ok(), "token_embd plans ok");
        check(r.rule == Rule::Replicate, "token_embd is Replicate");
        check(r.is_replicated(), "is_replicated()");
        check_eq((long)r.rank_bytes, (long)r.full_bytes, "replicated rank holds all bytes");
        check_eq((long)r.copy.src_offset, 0, "replicated copy starts at 0");
        check(r.split_axis == SplitAxis::None, "replicated tensor has no split axis");

        // TP=1 must take the same path and come out whole.
        ShardDims one; one.tp_size = 1; one.rank = 0;
        const TensorResidency r1 = plan_tensor_residency("blk.0.ffn_down.weight",
                                                         ne, 2, TY_Q8_0, one);
        check(r1.ok(), "tp=1 ffn_down plans ok");
        check(r1.rule == Rule::Replicate, "tp=1 degenerates every rule to Replicate");
        check_eq((long)r1.rank_bytes, (long)r1.full_bytes, "tp=1 holds the whole tensor");
    }

    // -------------------------------------------------------------- refusals
    std::printf("\n[refusals — these must NOT silently succeed]\n");
    {
        const long ne[4] = {7168, 7168, 1, 1};
        const TensorResidency unk = plan_tensor_residency("blk.0.brand_new_thing.weight",
                                                          ne, 2, TY_Q8_0, dims8(0));
        check(!unk.ok() && unk.error == ResidencyError::UnknownRule,
              "an unknown tensor name is refused, not guessed");

        const TensorResidency badty = plan_tensor_residency("blk.0.ffn_down.weight",
                                                             ne, 2, TY_UNKNOWN, dims8(0));
        check(!badty.ok() && badty.error == ResidencyError::UnknownType,
              "an unknown ggml type is refused (this is the IQ2_XS 745 GiB bug)");

        // ne0 not a whole number of blocks: malformed, must not size to a truncation.
        const long ragged[4] = {300, 128, 1, 1};
        const TensorResidency rag = plan_tensor_residency("blk.0.ffn_down.weight",
                                                           ragged, 2, TY_IQ2_XS, dims8(0));
        check(!rag.ok() && rag.error == ResidencyError::BadShape,
              "ne0 not block-aligned is refused");

        // A contracted axis with fewer blocks than ranks cannot ColShard. Must report
        // EmptyShard with a usable message rather than a zero-byte copy.
        const long tiny[4] = {256, 128, 1, 1};
        const TensorResidency ts = plan_tensor_residency("blk.0.ffn_down.weight",
                                                          tiny, 2, TY_IQ2_XS, dims8(7));
        check(!ts.ok() && ts.error == ResidencyError::EmptyShard,
              "1 block over 8 ranks is refused for the empty ranks");
        check(!ts.note.empty(), "EmptyShard carries an actionable note");
    }

    // ------------------------------------------------------------- 2-D MoE
    std::printf("\n[2-D MoE: expert groups x ffn band]\n");
    {
        // Shapes scaled down from K3's real ones but structurally identical: the
        // down tensor contracts over the FFN width, so ITS ne0 is the axis that must
        // stay block-aligned when fs cuts it. ffn=1024 is 4 blocks of 256, so fs=4
        // gives each rank exactly one whole block — the tightest legal case, which
        // is the one worth testing.
        const int n_experts = 16, moe_ffn = 1024, latent = 256;
        const long gate_ne[4] = {latent, moe_ffn, n_experts, 1};
        const long down_ne[4] = {moe_ffn, latent, n_experts, 1};

        check_conservation_2d("blk.1.ffn_gate_exps.weight", gate_ne, 3, TY_IQ2_XS,
                              8, /*eg=*/2, n_experts, moe_ffn, SplitAxis::Ne1_Output);
        check_conservation_2d("blk.1.ffn_up_exps.weight", gate_ne, 3, TY_IQ2_XS,
                              8, /*eg=*/2, n_experts, moe_ffn, SplitAxis::Ne1_Output);
        check_conservation_2d("blk.1.ffn_down_exps.weight", down_ne, 3, TY_IQ2_XS,
                              8, /*eg=*/2, n_experts, moe_ffn, SplitAxis::Ne0_Input);

        // eg=4 (4 groups x 2 ffn shards) must tile just as exactly — the point of
        // the descriptor is that the group count is a tuning knob, not a special case.
        check_conservation_2d("blk.1.ffn_down_exps.weight", down_ne, 3, TY_IQ2_XS,
                              8, /*eg=*/4, n_experts, moe_ffn, SplitAxis::Ne0_Input);

        // The bands themselves, at K3's real numbers.
        ShardDims d;
        d.tp_size = 8; d.rank = 5;
        const ShardError se = moe_2d_dims(896, 3072, 8, 5, 2, 256, &d);
        check(se.ok(), "K3 shape 896x3072 splits 2x4 at tp=8");
        check(d.moe_2d, "moe_2d is set");
        check_eq(d.moe_expert_groups, 2, "expert groups");
        check_eq(d.moe_ffn_shards, 4, "ffn shards");
        // rank 5 = group 1, shard 1 -> experts [448,896), ffn [768,1536)
        check_eq(d.expert_band.offset, 448, "rank 5 expert band offset");
        check_eq(d.expert_band.extent, 448, "rank 5 owns half the experts");
        check_eq(d.moe_ffn_band.offset, 768, "rank 5 ffn band offset");
        check_eq(d.moe_ffn_band.extent, 768, "rank 5 ffn band extent");

        // The DEFAULT policy. This is a behavioural default, not a flag, so it is
        // pinned by a test: an accidental change here silently re-shards every
        // deployment's experts.
        check_eq(k3_default_moe_expert_groups(8), 2, "tp=8 defaults to 2 expert groups");
        check_eq(k3_default_moe_expert_groups(4), 2, "tp=4 defaults to 2 expert groups");
        check_eq(k3_default_moe_expert_groups(2), 1, "tp=2 stays whole-expert");
        check_eq(k3_default_moe_expert_groups(1), 1, "tp=1 stays whole-expert");

        // A shape that cannot take the default must leave the ShardDims ALONE, because
        // the loader's fallback is simply to keep using the whole-expert band it wrote
        // earlier. If moe_2d_dims mutated before validating, that fallback would run on
        // a half-written descriptor.
        ShardDims keep;
        keep.tp_size = 8; keep.rank = 2;
        keep.expert_band = even_band(896, 8, 2);
        keep.n_experts = 112;
        // 900 over 8 groups leaves 4 over. (900 over *2* would be fine — 450 each —
        // which is why the indivisible case has to be chosen against the group count
        // actually being asked for, not against the tensor looking odd.)
        const ShardError bad_shape = moe_2d_dims(900, 3072, 8, 2, 8, 256, &keep);
        check(!bad_shape.ok(), "indivisible expert count is refused");
        check(!keep.moe_2d, "refusal leaves moe_2d false");
        check_eq(keep.expert_band.offset, 224, "refusal leaves the 1-D band untouched");
        check_eq(keep.n_experts, 112, "refusal leaves n_experts untouched");

        // eg=1 is the identity: it must not set moe_2d or disturb the 1-D band.
        ShardDims id;
        id.tp_size = 8; id.rank = 3;
        id.expert_band = even_band(896, 8, 3);
        const ShardError se1 = moe_2d_dims(896, 3072, 8, 3, 1, 256, &id);
        check(se1.ok() && !id.moe_2d, "expert_groups=1 leaves whole-expert sharding");
        check_eq(id.expert_band.offset, 336, "eg=1 does not touch the 1-D band");

        // Refusals, each naming the axis that failed rather than clamping.
        ShardDims bad;
        bad.tp_size = 8; bad.rank = 0;
        check(!moe_2d_dims(896, 3072, 8, 0, 3, 256, &bad).ok(),
              "eg must divide tp_size");
        check(!moe_2d_dims(900, 3072, 8, 0, 8, 256, &bad).ok(),
              "eg must divide n_experts");
        // fs=4 over an ffn of 3080 leaves 770, not a whole 256-block.
        check(!moe_2d_dims(896, 3080, 8, 0, 2, 256, &bad).ok(),
              "ffn shard must be a whole number of quant blocks");
    }

    // ------------------------------------------------- whole-model aggregation
    std::printf("\n[model-level residency: replication is the TP memory cost]\n");
    {
        std::vector<TensorDesc> ts;
        auto add = [&](const char* name, long a, long b, long c, int nd, int ty) {
            TensorDesc t; t.name = name;
            t.ne[0]=a; t.ne[1]=b; t.ne[2]=c; t.ne[3]=1; t.n_dims=nd; t.ggml_type=ty;
            ts.push_back(t);
        };
        // A representative MoE layer, with the verified expert shapes.
        add("blk.1.ffn_gate_exps.weight", 3584, 3072, 896, 3, TY_IQ2_XS);
        add("blk.1.ffn_up_exps.weight",   3584, 3072, 896, 3, TY_IQ2_XS);
        add("blk.1.ffn_down_exps.weight", 3072, 3584, 896, 3, TY_IQ2_XS);
        add("blk.1.attn_norm.weight",     7168,    1,   1, 1, TY_F32);
        add("blk.1.ffn_norm.weight",      7168,    1,   1, 1, TY_F32);
        add("blk.1.ffn_gate_inp.weight",  7168,  896,   1, 2, TY_F32);

        std::vector<TensorResidency> out;
        const ResidencyReport rep = plan_model_residency(ts, dims8(0), &out);
        check_eq(rep.failed, 0, "no tensor fails to plan");
        check_eq(rep.tensors, 6, "all tensors planned");
        check_eq(rep.sharded, 3, "the three expert tensors shard");
        check_eq(rep.replicated, 3, "norms + router replicate");
        check(rep.bytes_rank < rep.bytes_model, "a rank holds less than the model");
        check(rep.bytes_all_ranks > rep.bytes_model,
              "TP costs MORE total memory than the model — replication is why");
        std::printf("  model %.3f GiB | rank %.3f GiB | replicated %.6f GiB | all ranks %.3f GiB\n",
                    rep.bytes_model/1073741824.0, rep.bytes_rank/1073741824.0,
                    rep.bytes_replicated/1073741824.0, rep.bytes_all_ranks/1073741824.0);

        // A failure must be counted AND named — a partial upload is a silently wrong
        // model, so the loader has to be able to abort with a reason.
        add("blk.1.mystery.weight", 7168, 7168, 1, 2, TY_Q8_0);
        const ResidencyReport bad = plan_model_residency(ts, dims8(0));
        check_eq(bad.failed, 1, "the unknown tensor is counted as failed");
        check(!bad.errors.empty() &&
                  bad.errors[0].find("mystery") != std::string::npos,
              "the failing tensor is named in the report");
    }

    std::printf("\n%s: %d checks, %d failures\n", g_fail ? "FAIL" : "PASS",
                g_checks, g_fail);
    return g_fail ? 1 : 0;
}
