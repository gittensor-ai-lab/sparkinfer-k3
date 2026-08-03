// Front door for the single-barrier all-reduce.
//
// THE COST BEING REMOVED. The peer one-shot reduce is two cross-GPU rendezvous
// with a memory access between them: barrier_at_start (wait until every rank's
// input is written), read all 8 peers and sum, barrier_at_end. At K3's payload —
// 28 KB at the attention reduce, 42 KB at the MoE one — the reduce itself moves
// about 3% of NVLink in the time the kernel takes, so what the kernel costs is
// almost entirely the two rendezvous, and there are 185 of them per token.
//
// The exit barrier does not protect the RESULT. tp_allreduce.cuh states its only
// job: with final_sync=true every rank already holds the full sum locally, and
// the barrier exists purely to stop a fast rank overwriting the shared INPUT
// while a slow peer is still reading it. That hazard exists only because all 185
// collectives share one peer-visible input buffer per rank.
//
// Give each rank kSlotCount input buffers and rotate them, and the hazard is
// gone by construction: the write that precedes collective k+P is separated from
// the reads of collective k by the ENTRY barrier of every collective in between,
// each of which proves that all eight peers reached it — hence finished the
// kernel before it — before this rank went on. One rendezvous per collective
// instead of two, and nothing else changes: same reduce, same order, same bits.
//
// WHY THE HOST HAS TO CHECK THE ROTATION. The proof needs a minimum reuse
// distance of 2 between two writes to the same slot, and a captured CUDA graph
// bakes its pointers, so the rotation restarts at slot 0 on every token rather
// than running free. With C collectives per token that makes the wrap the
// binding case: slot (C-1) % P is written last in one token and again at index
// (C-1) % P of the next, a distance of ((C-1) % P) + 1. So the factor is only
// sound while (C-1) % P >= 1. It holds at K3's 185 with P=3 and fails at P=2 —
// but C follows the shard policy, so this is asserted against the collective
// count the driver actually issued rather than assumed from today's geometry.
#pragma once

namespace sparkinfer::tp {

// SPARKINFER_K3_COLL_1BAR=0 declines on the same binary: the kernel keeps its
// exit barrier, one slot is used, and every pointer is what main computes.
bool k3_coll_1bar_enabled();

// True when `collectives_per_token` admits the rotation at `slots`. The driver
// calls this once, before capture, with the count it is about to issue.
bool k3_coll_1bar_ok(int collectives_per_token, int slots);

}  // namespace sparkinfer::tp
