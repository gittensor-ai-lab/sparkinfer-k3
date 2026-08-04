#pragma once
// The TP collective: one all-reduce, three possible backends, one interface.
//
// Kimi K3 at tp_size 8 issues 186 all-reduces per decoded token (93 layers x 2),
// each carrying 14 KiB of bf16 hidden state. That is a LATENCY problem, not a
// bandwidth one — 14 KiB is nothing for NVLink, but 186 round-trips per token
// is the whole decode budget. Which backend wins is therefore decided by
// per-call overhead, and must be measured, not assumed.
//
// BACKEND ORDER IS DELIBERATE. NCCL is the default because it is the only one
// that is known-correct on real hardware. The faster candidates in
// /home/speedy/sn14/cacheon-sglang-miner both carry "UNTESTED on hardware as
// committed" in their own headers, and are additionally derived from vLLM
// (Apache-2.0) while this repo is MIT — so they are declared here as backends
// this interface can grow into, and deliberately NOT vendored yet:
//
//   Nccl      ncclAllReduce. Correct, portable, ~5-10 us of launch overhead.
//   PeerOneShot   every rank reads all peers and sums; in-kernel flag barrier,
//                 one kernel per rank, no host events. Wins at 14 KiB.
//   Multimem      NVSwitch multimem.ld_reduce: the switch does the reduction, so
//                 each rank issues ONE reduced load instead of N peer loads.
//                 Needs sm_90+ and NVLS — i.e. exactly an HGX H200/B200 node.
//
// Adding a backend must not touch the forward: it selects by enum, falls back
// down the list when a backend reports unavailable, and every backend produces
// bit-comparable results for the same input (checked by tp_allreduce_check).

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

// Backend, Capabilities, select_backend() and backend_from_string() live in a
// CUDA-free header so the fallback chain can be unit-tested without a GPU.
#include "sparkinfer/tp/backend_select.h"

namespace sparkinfer {
namespace tp {

// ---------------------------------------------------------------------------

// One all-reduce implementation over a fixed device set.
//
// TWO CALLING MODES, because the fast backends cannot use the obvious one.
//
//   Mode A — in-place, per-rank (NCCL, and the TP=1 no-op)
//     allreduce_bf16(buf, count, rank, stream) reduces the caller's own buffer.
//     Enqueue-and-return; the caller synchronizes.
//
//   Mode B — owned buffers, group launch (peer-oneshot, multimem)
//     owns_buffers() is true. Only multicast-bound (multimem) or peer-registered
//     (one-shot) allocations can back those backends' loads, so the reduction
//     buffer HAS to belong to the collective. The caller writes its activation
//     into reduce_in(rank), calls allreduce_group() once with every stream, and
//     reads the result from reduce_out(rank).
//
// Mode B is not an API wart, it is the hardware constraint made visible. Hiding
// it behind a copy into/out of a caller buffer would add two 14 KiB device-to-
// device copies per collective — 372 extra copies per K3 token — and erase the
// entire reason for using these backends. Input and output are separate regions
// for the same reason: in a one-shot, peers read every rank's input while each
// rank writes its own output, so a shared region would be a read-after-write race.
//
// A forward path should ask owns_buffers() once at setup and branch there, not
// per collective.
class Collective {
public:
    virtual ~Collective() = default;

    virtual Backend backend() const = 0;
    virtual int size() const = 0;

    // Mode A, GROUP form — the primary entry point, and the only correct one for a
    // single host thread driving every rank (which is sparkinfer's model).
    //
    // MEASURED: calling the per-rank allreduce_bf16() below in a loop from one
    // thread DEADLOCKS with NCCL. ncclAllReduce on rank 0's communicator blocks
    // waiting for the other ranks to join a collective that the same thread has not
    // enqueued yet. NCCL's fix is ncclGroupStart/ncclGroupEnd around the whole set,
    // which defers launch until every rank's call is registered. This hung on an
    // 8x H200 node at the first collective and produced no output at all, which is
    // exactly why the group form is now the default rather than a variant.
    //
    // `bufs[r]` is rank r's in-place buffer on devices[r]; ignored (may be empty)
    // when owns_buffers() is true, because those backends reduce their own memory.
    // `streams` size must equal size().
    virtual bool allreduce_bf16_group(const std::vector<void*>& bufs,
                                      std::size_t count,
                                      const std::vector<cudaStream_t>& streams) = 0;

    // Mode A, PER-RANK form. Safe ONLY when each rank is driven by its own host
    // thread, so the calls genuinely overlap. From a single thread iterating ranks
    // this deadlocks — use allreduce_bf16_group() instead.
    //
    // In-place sum-reduce of `count` bf16 elements in `buf` (device memory on rank
    // `rank`'s device). Returns false on failure, and a false must be treated as
    // fatal: continuing with a partial reduction presents as a quality regression
    // rather than a bug. Returns false unconditionally when owns_buffers() is true.
    virtual bool allreduce_bf16(void* buf, std::size_t count, int rank,
                                cudaStream_t stream) = 0;

    // ---- f32 ---------------------------------------------------------------
    //
    // WHY A SECOND DTYPE RATHER THAN CASTING. Kimi K3's executor runs its hidden
    // state in f32 end to end, and that is a decision with a stated reason
    // (kernels/include/sparkinfer/kernels/kimi_k3.h): every K3 kernel is
    // transcribed from a float64 reference and validated to near machine epsilon,
    // and bf16 would truncate to ~8 mantissa bits. TP adds 2 collectives per layer
    // x 93 layers, so routing K3's activations through allreduce_bf16 would
    // reintroduce that truncation 186 TIMES PER TOKEN — at the layer boundary,
    // which is precisely where the residual stream accumulates. The f32 path exists
    // so TP does not silently undo the executor's numerics.
    //
    // Qwen keeps using the bf16 entry points: its residual stream IS bf16, so a
    // bf16 collective is lossless there and moves half the bytes.
    //
    // Same group-call contract as allreduce_bf16_group (see the deadlock note above):
    // one host thread, every rank enqueued inside one NCCL group.
    virtual bool allreduce_f32_group(const std::vector<void*>& bufs,
                                     std::size_t count,
                                     const std::vector<cudaStream_t>& streams) {
        (void)bufs; (void)count; (void)streams;
        return false;
    }

    // False when this backend has no f32 reduce. Checked at SETUP, not per call:
    // a forward that needs f32 must refuse a backend that cannot do it rather than
    // discover it 186 collectives into a token. peer-oneshot and multimem both
    // report true — each has an f32 kernel that mirrors its bf16 one at 4 elements
    // per 128-bit transaction.
    virtual bool supports_f32() const { return false; }

    // Mode B. False for NCCL and the no-op; true for the peer/multimem backends.
    virtual bool owns_buffers() const { return false; }

    // Rank `rank`'s input / output region, on devices[rank]. nullptr unless
    // owns_buffers(). Sized for the `max_count` passed at construction.
    virtual void* reduce_in(int rank) const { (void)rank; return nullptr; }
    virtual void* reduce_out(int rank) const { (void)rank; return nullptr; }

    // Mode B launch: ONE kernel per rank across all streams, with the cross-rank
    // barrier inside the kernel — no host events, no cross-stream graph edges.
    // That barrier mechanism, not the reduce arithmetic, is what these backends
    // exist to change. `streams` size must equal size(). False if unsupported.
    virtual bool allreduce_group(std::size_t count,
                                 const std::vector<cudaStream_t>& streams) {
        (void)count; (void)streams; return false;
    }

    // Mode B, f32: allreduce_group's f32 twin. The caller has already written
    // rank r's f32 partial into reduce_in(r) and reads the sum from
    // reduce_out(r) — no staging copies. This, not the staged
    // allreduce_f32_group above, is the path the "372 extra copies" comment at
    // the top of this file is about. False if unsupported.
    virtual bool allreduce_f32_owned(std::size_t count,
                                     const std::vector<cudaStream_t>& streams) {
        (void)count; (void)streams; return false;
    }

    // ---- rotating input slots ------------------------------------------------
    // How many independently-writable input regions this backend offers. ONE by
    // default, which is the shape every backend has always had: a single peer-
    // visible input per rank whose reuse across the token's collectives is made
    // safe by the reduce kernel's own exit barrier.
    //
    // A backend that returns more is offering to drop that exit barrier in
    // exchange for the caller rotating the slot, so the write that precedes
    // collective k+slots() is separated from collective k's reads by every entry
    // barrier in between. The caller opts in by using the two calls below; a
    // caller that ignores them gets exactly the old behaviour, which is why these
    // are defaulted rather than pure.
    virtual int reduce_slots() const { return 1; }
    virtual void* reduce_in_slot(int rank, int slot) const {
        (void)slot; return reduce_in(rank);
    }
    virtual bool allreduce_f32_owned_slot(std::size_t count,
                                          const std::vector<cudaStream_t>& streams,
                                          int slot) {
        (void)slot; return allreduce_f32_owned(count, streams);
    }

    // Largest `count` the owned buffers can hold. 0 when !owns_buffers().
    virtual std::size_t max_count() const { return 0; }

    // Barrier across all ranks' streams. Only needed at graph boundaries.
    virtual bool barrier() = 0;
};

// tp_size == 1. Every call is a no-op that returns true, so the forward can be
// written once with unconditional collective calls and still be bit-identical
// to the pre-TP path at TP=1. That equivalence is what makes it safe to land
// this before any of it runs on 8 GPUs.
std::unique_ptr<Collective> make_single_device_collective();

// NCCL-backed. Returns nullptr when SPARKINFER_TP_NCCL was not compiled in, or
// when communicator init fails; `error` explains. Single process, one
// communicator per device (ncclCommInitAll), which matches sparkinfer's
// single-process multi-GPU runtime.
std::unique_ptr<Collective> make_nccl_collective(const std::vector<int>& devices,
                                                 std::string* error);

// Mode B backends. `max_count` sizes the owned buffers — pass the largest
// all-reduce the forward will issue (hidden_size for decode; hidden * ubatch if
// the same collective serves prefill). Both return nullptr with `error` set when
// the hardware cannot support them, so make_collective() can fall back to NCCL.
std::unique_ptr<Collective> make_peer_oneshot_collective(const std::vector<int>& devices,
                                                        std::size_t max_count,
                                                        std::string* error);

std::unique_ptr<Collective> make_multimem_collective(const std::vector<int>& devices,
                                                     std::size_t max_count,
                                                     std::string* error);

// Probe the machine. Safe to call with no GPU present: returns device_count 0.
Capabilities probe_capabilities();

// Build the right Collective for `devices`, honouring `requested` and falling
// back per select_backend(). Never returns nullptr: on total failure it returns
// the single-device no-op and sets `error`, so a caller that ignores the error
// gets a correct TP=1 run rather than a crash.
//
// `need_f32` (Kimi K3) narrows the choice to backends that implement the f32
// reduce. A fast backend that cannot is skipped with a logged reason rather than
// selected and failed later — the alternative is a run that loads 553 GiB and then
// dies at the first collective.
std::unique_ptr<Collective> make_collective(const std::vector<int>& devices,
                                            Backend requested,
                                            std::string* error,
                                            std::size_t max_count = 0,
                                            bool need_f32 = false);

}  // namespace tp
}  // namespace sparkinfer
