// Factor — size the one-shot all-reduce's CTA to the payload, not to a constant.
//
// ===========================================================================
// THE SHAPE
// ===========================================================================
// peer_oneshot's f32 launch is `block = 512; grid = ceil(n_vec / block)`, capped at
// kMaxBlocks. That is vLLM's shape, and vLLM's payloads are a batch of hidden states.
// K3 at decode reduces ONE token: 7168 floats is 1792 uint4 vectors, so the grid comes
// out at FOUR CTAs on a 132-SM GPU, and the two narrower reduces in a banded MoE layer
// come out at three and six. Measured at the scored context, the f32 all-reduce is
// 11.5% of the trimmed per-kernel budget and 369 calls per token per rank, every one of
// them on at most six SMs.
//
// The work is FIXED at n_vec vectors — one uint4 per thread, eight peer loads each — so
// a bigger grid cannot be had by asking for more total threads. It has to come from a
// SMALLER CTA. 1792 vectors is 4 CTAs of 512, or 14 of 128, or 28 of 64: the same 1792
// threads, spread over 28 SMs instead of 4. Each SM has its own miss-handling budget,
// so what widens is the number of peer loads that can be in flight at once, which is
// the only thing a one-shot reduce is ever short of.
//
// ===========================================================================
// WHAT IT CANNOT FIX, AND WHY THE FACTOR IS SIZED SMALL
// ===========================================================================
// Most of a decode-sized reduce is NOT bandwidth. Measured on this box at 128k, the
// route exchange (4480 floats, 143 KB of peer reads) takes 4.51 us and the hidden
// reduce (7168 floats, 229 KB) takes 4.83 us — 1.6x the bytes for 1.07x the time. The
// marginal rate across that pair is ~268 GB/s, so the fixed term is ~4 us and the
// transfer term is well under 1 us. That fixed term is one NVLink round trip through
// barrier_at_start and no CTA geometry touches it.
//
// So this can only ever recover the transfer term — a few tenths of a microsecond per
// call against a ~4 us floor. It is worth doing because it is free and because 369
// calls multiply anything, and it is worth SAYING that the ceiling is small, because
// the same table also proves that widening the grid is not where a large collective
// win lives. It is not a substitute for issuing fewer rendezvous.
//
// ===========================================================================
// THE BOUND THAT IS NOT NEGOTIABLE
// ===========================================================================
// Signal carries start[kMaxBlocks][8] / end[kMaxBlocks][8], so a grid above kMaxBlocks
// writes past the flag arrays and corrupts a peer's barrier state — a hang or a silent
// wrong sum, not a launch failure. grid_for already clamps; the CTA chosen here only
// ever makes the clamp MORE likely to bind, and the clamp is what keeps the grid-stride
// loop covering the tail. The ladder below therefore REFUSES any width whose grid would
// be clamped, rather than picking it and letting the tail re-serialise.
//
// That refusal is why kMaxBlocks moves from 36 to 64 in the same change: at 36 the two
// widest of K3's four reduces (1792 and 2688 vectors) could not reach one warp per CTA
// without being clamped, so the ladder fell back to a wider CTA and fewer SMs. Measured
// on one binary: cap 36 gives 44.77 tok/s, cap 64 gives 44.90. The cap is a bound on
// the flag arrays, not a tuning knob — see the note beside it in tp_allreduce.cuh for
// why vLLM's contention argument does not apply to a single decode token.
#pragma once

namespace sparkinfer::tp {

// Threads per CTA for a one-shot reduce of `n_vec` 128-bit vectors on a device with
// `max_blocks` flag slots. Returns 512 — the shipped constant — when
// SPARKINFER_TP_COLL_CTA=0, so the previous geometry stays reachable on the same
// binary. An explicit 512/256/128/64/32 pins that width for an ablation.
int k3_coll_block_for(int n_vec, int max_blocks);

}  // namespace sparkinfer::tp
