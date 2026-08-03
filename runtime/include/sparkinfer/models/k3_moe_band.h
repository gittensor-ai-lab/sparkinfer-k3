// Front door for the banded MoE dense projections.
//
// THE COST BEING REMOVED. Three tensors on every one of K3's 92 MoE layers are
// REPLICATED — every rank holds the whole thing and every rank reads the whole
// thing, producing eight identical copies of the same answer:
//
//   ffn_gate_inp     896 x 7168  F32    25.7 MB   the router
//   ffn_routed_down 3584 x 7168  Q8_0   27.3 MB   hidden -> expert latent
//   ffn_routed_up   7168 x 3584  Q8_0   27.3 MB   expert latent -> hidden
//
// 80.3 MB per layer, 7.4 GB per token per rank, against a decode budget of about
// 22 GB. All three sit on strictly serial, fully exposed stretches of the layer —
// the router and routed_down between ffn_norm and the expert dispatch, routed_up
// between the MoE all-reduce and the residual add — with nothing beside them to
// hide behind. That is the shape the banded LM head (k3_head_band.h) transferred
// end-to-end at 1:1, and it is the same fix: give each rank an eighth of the rows
// and exchange the pieces.
//
// Every rank already HOLDS the weight, so this is a pointer offset and a smaller
// N with NO loader change. weight_plan.cpp declares all three Replicate, and it
// would not matter if it did not: upload_sliced's short-circuit only consults the
// rule table for the expert, KDA and MLA stacks, so everything else lands as a
// full replica on all eight devices regardless.
//
// WHAT IT COSTS. A band leaves each rank holding rows it did not compute, and the
// consumers need the whole vector — the router's top-k reads all 896 logits, an
// expert reads all 3584 latents, ffn_norm reads all 7168 hidden. So the pieces
// have to meet, and that is two new rendezvous per MoE layer:
//
//   ffn_norm -> router band + routed_down band -> [EXCHANGE] -> top-k, experts
//   MoE all-reduce -> routed_norm -> routed_up band -> [EXCHANGE] -> residual add
//
// The router and routed_down share ONE exchange because routed_down does not read
// the router's output — it reads ffn_norm's, exactly as the router does — so the
// two bands can be computed back to back and met once. That is what makes the
// router band free: it rides a rendezvous that routed_down was paying for anyway.
//
// THE EXCHANGE IS AN ALL-REDUCE, NOT A GATHER, and deliberately. Each rank zeroes
// the payload and writes only its own band, so the sum over eight ranks IS the
// concatenation: seven exact zeros and one value per element, and x + 0.0 is x
// for every finite float. That reuses the collective the driver already owns —
// same kernel, same slots, same rotation — instead of adding a second cross-rank
// primitive whose failure mode is a hang. The cost is one memset of the payload
// per exchange (4480 and 7168 floats), which is a launch, not bandwidth.
//
// BIT-IDENTICAL, and that is checkable rather than argued. A row band changes which
// rank computes a row, never how. The thread count per row is block_for(K/32) — it
// comes from K, which a ROW band does not touch — and each row's accumulator strides
// b = threadIdx.x, += BLOCK over the whole of K before the same fold. So every output
// element is the same sum of the same products in the same order, and mean_kld must
// match main to the last digit.
//
// WHICH KERNEL RUNS CHANGES, AND IN THE RIGHT DIRECTION. Both Q8_0 bands fall under
// k3_proj_q8_multirow_1bar's N >= 1024 floor, so they land on the single-row
// proj_q8_0_q8_0 kernel, whose grid is N rather than N/ROWS. That is a wash for
// routed_down and a WIDENING for routed_up, at a eighth of the bytes either way:
//
//   ffn_routed_down  3584 rows, ROWS 8  -> 448 CTAs   |  448 rows, 1 row/CTA -> 448
//   ffn_routed_up    7168 rows, ROWS 16 -> 448 CTAs   |  896 rows, 1 row/CTA -> 896
//
// at 128 threads throughout, i.e. 1792 warps -> 1792 and 1792 -> 3584. No launch is
// merged and no block is narrowed, so this passes "never trade concurrency" by
// construction — the same property the warp-budget tier was built on. The router
// keeps <<<N, 128>>> and simply loses seven eighths of its grid (896 CTAs -> 112);
// it also loses seven eighths of its bytes, which is why it still pays, but it is
// the one piece here that narrows and it has its own toggle for that reason.
//
// COLLECTIVE COUNT. Both exchanges together take K3 from 185 collectives per token
// to 369, and the single-barrier rotation in k3_coll_1bar.h needs (C-1) % slots
// >= 1 to stay sound. 368 % 3 = 2 holds, as 184 % 3 = 1 does with the factor off.
// Adding only ONE of the two exchanges would give 277 and 276 % 3 = 0, which is
// why they are one factor and one toggle rather than two: the driver would decline
// the rotation and quietly hand back what #107 measured at +0.91%.
#pragma once

#include <cstddef>

namespace sparkinfer {

// Rows [offset, offset + rows) of a replicated weight, and the byte offset of the
// first of them. `row_bytes` is derived from the tensor's own n_bytes rather than
// from its quant type, so the same call is correct for the F32 router and the two
// Q8_0 projections without this file knowing either encoding.
struct K3MoeBand {
    int    offset   = 0;
    int    rows     = 0;
    size_t byte_off = 0;
};

// SPARKINFER_K3_MOE_BAND=0 declines on the same binary: no phase split, no extra
// collectives, every projection full-width. Default ON — the harness scores a
// default build.
bool k3_moe_band_enabled();

// The router band alone, so its contribution is separable from routed_down's on one
// binary. SPARKINFER_K3_MOE_BAND_ROUTER=0 leaves ffn_gate_inp full-width while the
// two Q8_0 bands stay on; it does NOT change the collective count, because the
// exchange it rides exists for routed_down either way.
bool k3_moe_band_router_enabled();

// Is the band live for this geometry? ONE definition, called by both sides, because
// the forward and the driver have to agree exactly: the forward splits two phases on
// this predicate and the driver decides how many collectives a token issues on it,
// and a driver that expected an exchange the forward did not produce reduces a stale
// buffer while a forward that produced one the driver did not consume hangs waiting.
// All three widths must divide — routed_up is banded over hidden, the router over
// n_experts, routed_down over expert_latent — so a geometry that can only split some
// of them declines as a whole.
bool k3_moe_band_active(int n_experts, int expert_latent, int hidden, int tp_size);

// Fills `out` and returns true when the band is usable. Declines — leaving the
// caller on the full-width projection — unless the toggle is on, tp_size > 1
// divides `rows_total` exactly, the rank is in range, and `weight_bytes` divides
// evenly into rows.
//
// THE DIVISIBILITY IS REFUSED, NOT ROUNDED. An uneven split would still reduce to a
// complete vector as long as every rank agreed on the offsets, but a row stride
// built from a rounded n_bytes / rows reads the wrong weights at full speed and
// produces fluent, wrong logits. K3's three shapes are 896, 3584 and 7168 rows and
// all divide by 8; anything else takes the unbanded path.
bool k3_moe_band(int rows_total, size_t weight_bytes, int tp_size, int rank,
                 K3MoeBand* out);

}  // namespace sparkinfer
