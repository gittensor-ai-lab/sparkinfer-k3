// One-shot NVSwitch multimem all-reduce for the TP=8 decode collective (#74 T3).
//
// The current decode all-reduce (peer_sum_residual_rms_norm_bf16_kernel_n) has
// every rank read all N peers' buffers over NVLink and sum them — N× redundant
// reads, latency-bound for the 16 KiB (hidden×2) decode payload, ×160/token.
//
// multimem.ld_reduce (SM90 + NVSwitch multicast) reduces across all peers in a
// single instruction in the switch, so each rank issues one reduced load
// instead of N strided peer loads. This is the lever to make TP=8 decode scale
// (see docs/TP8_DECODE_OPTIMIZATION.md, logs/h200-8-june1-tp8-decode/).
//
// UNTESTED on hardware as committed — validate with `multimem_allreduce_bench`
// before integrating into the decode path (cuda/src/qwen2_forward_tp.cu). The
// implementation uses the CUDA driver multicast VMM API (cuMulticast*) +
// multimem.* PTX; alignment/granularity/PTX-variant details must be confirmed
// on the target 8× H200 box. Gated + fallback-safe: callers must check
// multimem_allreduce_supported() and fall back to the peer-N all-reduce.

#pragma once

#include <cstddef>
#include <vector>

namespace sparkinfer::tp {

// True iff every device supports CUDA multicast (CU_DEVICE_ATTRIBUTE_
// MULTICAST_SUPPORTED) and a multicast object spanning them can be created.
// Cheap; call once at construction to decide peer-N vs multimem.
bool multimem_allreduce_supported(const std::vector<int>& devices) noexcept;

// Owns a multicast object bound across `devices`, plus per-device unicast +
// multicast mappings of a single hidden-sized BF16 scratch. Reused across all
// decode all-reduces (allocate once).
//
// The class owns the all-reduce buffers (one physical multicast-bound BF16
// scratch per device, holding a distinct input and output region) and exposes
// each rank's via rank_buffer(rank)/rank_result(rank). The caller writes rank
// r's input into rank_buffer(r) on device devices[r], calls allreduce_bf16(),
// then reads the reduced sum from rank_result(r). Owning the buffers is what
// makes this copy-free: only multicast-bound physical allocations can back
// multimem.* loads, so an integration writes the AR input straight into
// rank_buffer() rather than copying an external buffer in/out.
//
// Input and output are SEPARATE regions on purpose: a one-shot multimem reduce
// has each rank load-reduce across every rank's input while writing its own
// result. Writing the result into the input region would race other ranks still
// reading it; the distinct output region removes the read-after-write hazard, so
// a single barrier (inputs-ready) suffices — matching the peer-N path's one
// event barrier.
//
// Assumes peer access is enabled across all pairs (the forward path does this
// at startup) and that callers serialize concurrent all-reduces on the buffers.
class MultimemAllreduce {
public:
    // Unusable instance if unsupported; check ok() before use. `count` = max
    // elements (e.g. hidden_size); buffers are sized for this.
    MultimemAllreduce(const std::vector<int>& devices, std::size_t count);
    ~MultimemAllreduce() noexcept;

    MultimemAllreduce(const MultimemAllreduce&) = delete;
    MultimemAllreduce& operator=(const MultimemAllreduce&) = delete;

    bool ok() const noexcept { return ok_; }

    // Rank r's input device pointer (on devices[r]): write the rank's input
    // here before allreduce_bf16(). nullptr if !ok().
    void* rank_buffer(int rank) const noexcept;

    // Rank r's output device pointer (on devices[r]): holds the reduced sum
    // after allreduce_bf16(). Distinct region from rank_buffer(r). nullptr if
    // !ok().
    void* rank_result(int rank) const noexcept;

    // All-reduce (sum) of `count` BF16 elements across all ranks' rank_buffer()
    // inputs; result lands in each rank_result(). ONE kernel per rank: an
    // in-kernel device-flag barrier (tp_allreduce.cuh) ensures all inputs are
    // written before any rank issues its multimem.ld_reduce — no host events,
    // no cross-stream graph edges. streams size must equal the device count.
    void allreduce_bf16(std::size_t count, const std::vector<void*>& streams);

    // Opaque PIMPL, defined in the .cu. Declared public so the file-local
    // setup/launch helpers can name the type; impl_ itself stays private.
    struct Impl;

private:
    Impl* impl_ = nullptr;
    bool ok_ = false;
};

}  // namespace sparkinfer::tp
