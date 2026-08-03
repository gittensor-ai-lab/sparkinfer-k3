// How many output rows one CTA of the Q8_0 multi-row projection should carry.
//
// THE DEFECT. The shipped tier picks ROWS from N alone, against a floor written as
// a CTA count: 16 rows above N=4096, 8 above 2048, 4 above 1024. But the block
// WIDTH is chosen separately, from K — block_for(K/32) returns 32, 64 or 128
// threads — so the same "448 CTAs" is 448 warps on a narrow-K shape and 1792 on a
// wide-K one. The floor is therefore not a floor on anything the hardware cares
// about, and four of K3's projections land far under it:
//
//   ffn_down_shexp  7168 x  768   BLOCK 32   ROWS 16 ->  448 warps  = 3.4 /SM
//   attn_output     7168 x 1536   BLOCK 64   ROWS 16 ->  896 warps  = 6.8 /SM
//   attn_q_b        2304 x 1536   BLOCK 64   ROWS  8 ->  576 warps  = 4.4 /SM
//   ssm_f_b         1536 x  128   BLOCK 32   ROWS  4 ->  384 warps  = 2.9 /SM
//
// against ffn_routed_down (3584 x 7168, BLOCK 128, ROWS 8) and ffn_routed_up
// (7168 x 3584, BLOCK 128, ROWS 16), which both land on 1792 warps. Those two are
// the SAME kernel over the same 34-byte block stride, and on an 8x H200 at ctx
// 131072 they move their weights at 1.8 TB/s while ffn_down_shexp manages 0.42 and
// attn_output 0.91. The access pattern is identical; the only variable that differs
// is how many warps are resident to keep loads in flight. The kernel's own header
// already diagnoses this — "a block issues one or two rounds of loads, then stops
// to reduce; there is nothing left in flight to cover the next miss" — and then
// tiers on the axis that does not fix it.
//
// THE FIX is to tier on the warp count the tier was always trying to express.
// Lowering ROWS RAISES the grid (grid = ceil(N/ROWS)), so this only ever widens:
// no launch is merged, no block is narrowed, nothing is traded. The cost is that
// the activation is re-read by more CTAs — 0.16 to 1.4 MB per call against 5.8 to
// 27 MB of weights, and it is L2-resident either way.
//
// BIT-IDENTICAL. A row's accumulator strides b = threadIdx.x, +BLOCK over the
// whole of K and then folds across warps in increasing order; ROWS only decides
// which rows share a CTA, and BLOCK — the one thing an accumulation order does
// depend on — is picked from K and is untouched. This is the property the kernel
// states about itself, applied to a different tier.
#pragma once

namespace sparkinfer::kernels::k3 {

// Rows per CTA for a projection of N rows at `block_threads` threads per CTA.
// Returns `legacy_rows` unchanged when SPARKINFER_K3_PROJ_ROWBUDGET=0, so the
// shipped tier stays reachable on the same binary.
//
// Deliberately NOT applied to the fused four-tensor group projection, which
// hardcodes ROWS=4 at 1536 warps: re-tiering it to 2 would double the activation
// re-read on the second-largest weight read in the token to chase 256 warps, and
// it has never been measured that way. One kernel, one tier, one factor.
int k3_proj_rows_for_budget(int N, int block_threads, int legacy_rows);

}  // namespace sparkinfer::kernels::k3
