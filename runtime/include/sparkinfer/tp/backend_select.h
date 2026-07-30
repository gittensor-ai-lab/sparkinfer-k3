#pragma once
// Which all-reduce backend to use, and why. DELIBERATELY CUDA-FREE, split out of
// collective.h so the fallback chain is unit-testable with no GPU present (see
// runtime/tests/tp_shard_cpu_test.cpp).
//
// Selecting a collective is not a detail. If a run silently falls back to a
// different backend than the one an operator asked for, then a benchmark
// attributes its number to the wrong mechanism — the comm "optimization" gets
// credited for a change it never made. So selection is pure, explicit, and it
// always reports the reason for a downgrade.

#include <string>

namespace sparkinfer {
namespace tp {

enum class Backend {
    None,          // tp_size == 1: no collective is issued at all
    Nccl,          // default whenever tp_size > 1
    PeerOneShot,   // peer-read one-shot with an in-kernel barrier (not vendored)
    Multimem,      // NVSwitch multimem.ld_reduce (not vendored)
};

const char* backend_name(Backend b);

// What the box can actually do. Probed on a real node; hand-built in tests.
struct Capabilities {
    int device_count = 0;
    bool nccl_available = false;
    bool peer_access_all_pairs = false;   // every ordered pair can peer-map
    bool multicast_supported = false;     // CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED
    int min_compute_capability = 0;       // 90 = H200, 100 = B200, 103 = B300
};

// Pure. Returns the backend that will actually be used; writes an explanation
// into `reason` whenever the answer differs from `requested`.
//
// Rules, in order:
//   tp_size <= 1                -> None   (no collective at all)
//   fewer devices than tp_size  -> None   (cannot form the group; hard downgrade)
//   Multimem without multicast
//     or below sm_90            -> Nccl
//   PeerOneShot without
//     all-pairs peer access     -> Nccl
//   Multimem/PeerOneShot at all -> Nccl   until they are vendored (see below)
//   Nccl without NCCL compiled  -> None   (and `reason` says so loudly)
//
// The "until vendored" rule is the important one: the two fast backends are
// declared so the interface and the benchmark tool can name them, but they are
// derived from vLLM (Apache-2.0) while this repo is MIT, and their own headers
// say UNTESTED on hardware. Returning Nccl for them keeps the build honest —
// there is no path by which an unvendored backend appears to have run.
Backend select_backend(Backend requested, int tp_size, const Capabilities& caps,
                       std::string* reason);

// True once a backend's implementation is actually present in this tree.
// Flipping one of these to true is the single edit that enables a fast path.
bool backend_vendored(Backend b);

// Parse SPARKINFER_TP_BACKEND ("nccl" | "peer" | "multimem" | "none" | "").
// Unknown strings return Nccl and explain, rather than failing a run over a typo.
Backend backend_from_string(const std::string& s, std::string* reason);

}  // namespace tp
}  // namespace sparkinfer
