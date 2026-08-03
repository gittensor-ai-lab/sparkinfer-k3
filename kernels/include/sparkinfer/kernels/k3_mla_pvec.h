// Factor — MLA latent phase: read the token's probabilities as one broadcast, not HPB.
//
// ===========================================================================
// THE SHAPE
// ===========================================================================
// mla_decode_attn_hbatch_kernel is the largest kernel in K3's decode token at the
// scored context — 181 us a call, 24 calls per token per rank, 22.4% of the trimmed
// per-kernel budget measured with the OFFICIAL --seek invocation. It runs 264 blocks
// of 256 threads at exactly 128 registers, which is two blocks per SM by one register,
// so nothing about occupancy is available (see the HPB/RSLOTS ladder beside the
// kernel: 129 registers costs 10%, and -maxrregcount=85 buys the third block at the
// price of 436 bytes of spill).
//
// What IS available is issue slots. The latent-accumulation phase runs
//
//     for t in tile:
//         p[hh] = s_p[hh][t]                       HPB shared loads
//         for u in RSLOTS: kv = k_cache[t][r]       RSLOTS global loads
//                          acc[u][hh] += p[hh]*kv   RSLOTS*HPB FMAs
//
// and at K3's HPB 12 / RSLOTS 2 that is 12 shared loads and 2 global loads against 24
// FMAs: 38 instructions of which only 63% do arithmetic. The kernel measures 28% of
// the card's FP32 issue rate and 17% of its HBM rate, so it is bound by neither
// resource — it is bound by instruction issue, and a third of this loop's issue slots
// are spent fetching twelve floats that every lane in the block wants identically.
//
// ===========================================================================
// THE CHANGE
// ===========================================================================
// s_p is stored t-major — s_p[t * HPB + hh] rather than s_p[hh * kMlaCtxTile + t] —
// so one token's HPB probabilities are ADJACENT. Every lane reads the same address,
// which is a shared-memory broadcast, and adjacency turns HPB scalar broadcasts into
// HPB/4 broadcast LDS.128. At HPB 12 the loop goes 12 loads -> 3, and 38 instructions
// -> 29, an 8.4% smaller inner loop on 22.4% of the token.
//
// The allocation is unchanged: HPB * kMlaCtxTile floats either way. hshm, the shared
// budget check, the slice-length floor, the register count and the two-blocks-per-SM
// fact are all untouched — this moves indices, not bytes.
//
// ALIGNMENT IS A PRECONDITION, NOT A HOPE. s_p begins at s_q + HPB*key_length floats
// and the float4 read is at s_p + t*HPB. HPB is 12, 8 or 4 at every instantiation this
// kernel offers, so t*HPB*4 bytes is always a multiple of 16; key_length is 576, so
// the base offset 12*576*4 = 27,648 bytes is too. Both hold for the shipped shape and
// the static_assert in the launcher pins the one that is actually instantiated.
//
// ===========================================================================
// WHAT IT COSTS, AND WHY THAT SIDE IS THE SMALL ONE
// ===========================================================================
// The online-softmax phase walks one head along the tile, so t-major turns its unit
// stride into a stride of HPB. At HPB 12 that is gcd(12,32) = 4 banks' worth of
// conflict. It is paid knowingly: phase 2 touches each probability three times, with
// 12 heads spread over NWARP warps, while phase 3 touches every one of them once per
// thread per RSLOTS slot — 256 threads against 32 lanes. The traffic on the two sides
// of the trade differs by more than an order of magnitude, which is why the layout is
// chosen for the phase that reads it most.
//
// ===========================================================================
// WHAT IT IS NOT
// ===========================================================================
// NOT an occupancy change — same registers, same shared bytes, same 264 blocks.
// NOT a change to the split count, the fill target or the slice floor (all swept
// in-tree already; 264 blocks wins and 330/396 lose).
// NOT the head-batching trade: HPB stays 12, so MLA's MQA cache read stays one pass.
// NOT numerically approximate. Same values in the same order into the same
// accumulators; the accuracy gate is expected to reproduce main to the last digit.
#pragma once

namespace sparkinfer::kernels::k3 {

// True unless SPARKINFER_K3_MLA_PVEC=0. Latched once; the launcher reads it to pick
// between two instantiations of the same kernel, so =0 restores main's layout on the
// SAME binary and a leave-one-out ablation needs no second build.
bool k3_mla_pvec_on();

}  // namespace sparkinfer::kernels::k3
