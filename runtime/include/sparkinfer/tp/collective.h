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
// Threading/stream contract: allreduce_bf16() is called once per rank per
// collective, with that rank's device current and that rank's stream. It is
// asynchronous — it enqueues and returns. The caller synchronizes.
class Collective {
public:
    virtual ~Collective() = default;

    virtual Backend backend() const = 0;
    virtual int size() const = 0;

    // In-place sum-reduce of `count` bf16 elements in `buf` (device memory on
    // rank `rank`'s device), across all ranks. Returns false on failure; the
    // caller must treat a false as fatal rather than continuing with a partial
    // reduction, which would look like a quality regression rather than a bug.
    virtual bool allreduce_bf16(void* buf, std::size_t count, int rank,
                                cudaStream_t stream) = 0;

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

// Probe the machine. Safe to call with no GPU present: returns device_count 0.
Capabilities probe_capabilities();

// Build the right Collective for `devices`, honouring `requested` and falling
// back per select_backend(). Never returns nullptr: on total failure it returns
// the single-device no-op and sets `error`, so a caller that ignores the error
// gets a correct TP=1 run rather than a crash.
std::unique_ptr<Collective> make_collective(const std::vector<int>& devices,
                                            Backend requested,
                                            std::string* error);

}  // namespace tp
}  // namespace sparkinfer
