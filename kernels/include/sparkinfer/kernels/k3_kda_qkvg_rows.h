// Rows per CTA for the FUSED FOUR-TENSOR group projection.
//
// THE ONE SHAPE #107's WARP BUDGET LEFT ALONE. k3_proj_q8_fused4_1bar hardcodes
// ROWS = 4, and at K3's KDA group (N 1536, K 7168 -> BLOCK 128) that is
// 384 CTAs = 1536 warps, just under the 1792-warp budget the rest of the
// projection family was re-tiered onto. It was excluded deliberately and never
// measured, on the argument that lowering ROWS doubles the activation re-read to
// chase 256 more warps. This file exists to settle that by measurement instead.
//
// It is worth settling because the kernel is BIG: 24.3 us a call, 69 calls per
// token per rank, 6.5% of the whole token at the scored context — the third
// largest kernel in the profile after MLA attention and the collective.
//
// THE TRAFFIC IT COSTS, exactly. Four Q8_0 weights at 1536 x 7168 are
// 4 * 11.7 = 46.8 MB, and every CTA re-reads the whole quantised activation:
// 384 CTAs * 7168 * 4 B = 11 MB at ROWS 4, 22 MB at ROWS 2. So halving ROWS moves
// total traffic 57.8 -> 68.8 MB, +19%, to double the resident warps 1536 -> 3072.
// The activation is 11 MB against a 60 MB L2, so the re-read is L2-resident and
// the +19% is an upper bound on what it actually costs in HBM terms.
//
// That is a genuine trade rather than a free widening, which is why it is its own
// factor with its own toggle rather than folded into k3_proj_rowbudget.h: the
// budget file only ever LOWERS ROWS on shapes where nothing is traded, and this
// one trades. If it loses, it loses alone and the rest of the tier is unaffected.
//
// BIT-IDENTICAL either way. ROWS decides only which output rows share a CTA; BLOCK
// comes from K and is untouched, and each row's accumulator strides b = threadIdx.x,
// += BLOCK over the whole of K before the same one-barrier fold.
#pragma once

namespace sparkinfer::kernels::k3 {

// Rows per CTA for the fused-4 group projection of N rows at `block_threads`.
// Returns `legacy_rows` unchanged when SPARKINFER_K3_KDA_QKVG_ROWS=0, so the
// shipped ROWS=4 geometry stays reachable on the same binary.
int k3_kda_qkvg_rows_for_budget(int N, int block_threads, int legacy_rows);

}  // namespace sparkinfer::kernels::k3
