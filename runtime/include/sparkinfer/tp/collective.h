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

    // Mode A. In-place sum-reduce of `count` bf16 elements in `buf` (device
    // memory on rank `rank`'s device), across all ranks. Returns false on
    // failure — and a false must be treated as fatal, because continuing with a
    // partial reduction presents as a quality regression rather than a bug.
    // Returns false unconditionally when owns_buffers() is true.
    virtual bool allreduce_bf16(void* buf, std::size_t count, int rank,
                                cudaStream_t stream) = 0;

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
std::unique_ptr<Collective> make_collective(const std::vector<int>& devices,
                                            Backend requested,
                                            std::string* error,
                                            std::size_t max_count = 0);

}  // namespace tp
}  // namespace sparkinfer
