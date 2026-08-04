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
// f32 path + rotating slots (kInputSlots=3) land for Kimi K3's residual stream.
// Auto prefers this backend when multimem_allreduce_supported() passes (real
// multicast bind probe, not the device attribute alone). Construction failure
// falls back to peer-oneshot, then NCCL. Validate with `tp_allreduce_check`.

#pragma once

#include <cstddef>
#include <vector>

namespace sparkinfer::tp {

// True iff multimem can ACTUALLY run here: every device reports
// CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED *and* a minimal multicast object
// spanning them can be created, have every device added, and be BOUND to physical
// memory. The bind is the part that matters.
//
// MEASURED: on an 8x H200 container without Fabric Manager access, all eight
// devices report MULTICAST_SUPPORTED = 1 and cuMulticastBindMem then fails with
// CUDA error 401 (CUDA_ERROR_ILLEGAL_STATE). NCCL hits the same wall. The
// attribute describes the silicon, not this process's permissions, so an
// attribute-only check reports multimem as available on nodes where it cannot run.
//
// Costs one granularity-sized allocation, once. Call at construction to decide
// peer-N vs multimem.
bool multimem_allreduce_supported(const std::vector<int>& devices) noexcept;

// Owns a multicast object bound across `devices`, plus per-device unicast +
// multicast mappings of a single hidden-sized scratch sized for f32 (the wider
// of the two dtypes this class reduces; bf16 uses the front half). Reused across
// all decode all-reduces (allocate once).
//
// The class owns the all-reduce buffers (one physical multicast-bound scratch
// per device, holding a distinct input and output region) and exposes each
// rank's via rank_buffer(rank)/rank_result(rank). The caller writes rank r's
// input into rank_buffer(r) on device devices[r], calls allreduce_bf16() or
// allreduce_f32(), then reads the reduced sum from rank_result(r). Owning the
// buffers is what makes this copy-free: only multicast-bound physical
// allocations can back multimem.* loads, so an integration writes the AR input
// straight into rank_buffer() rather than copying an external buffer in/out.
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
    // Consulted by the collective adapter at compile time. True once the f32
    // multimem.ld_reduce path exists — Kimi K3 keeps its residual stream f32 by
    // design, so a bf16-only surface forced every K3 multimem request down to
    // NCCL before construction.
    static constexpr bool kSupportsF32 = true;
    // Rotating input slots for the single-barrier f32 path — same contract as
    // PeerOneShotAllreduce. THREE, and the count is load-bearing; see
    // sparkinfer/tp/k3_coll_1bar.h. static_assert'd against tp_allreduce.cuh's
    // kSlotCount in the .cu.
    static constexpr int kInputSlots = 3;

    // Unusable instance if unsupported; check ok() before use. `count` = max
    // elements (e.g. hidden_size); buffers are sized for this.
    MultimemAllreduce(const std::vector<int>& devices, std::size_t count);
    ~MultimemAllreduce() noexcept;

    MultimemAllreduce(const MultimemAllreduce&) = delete;
    MultimemAllreduce& operator=(const MultimemAllreduce&) = delete;

    bool ok() const noexcept { return ok_; }

    // Rank r's input device pointer (on devices[r]): write the rank's input
    // here before allreduce_bf16()/allreduce_f32(). nullptr if !ok().
    void* rank_buffer(int rank) const noexcept;
    // Slot-addressed input, for the single-barrier f32 path. Slot 0 is the
    // buffer rank_buffer(rank) returns, so a caller that never rotates sees
    // main exactly. See sparkinfer/tp/k3_coll_1bar.h.
    void* rank_buffer(int rank, int slot) const noexcept;
    static int slots() noexcept;

    // Rank r's output device pointer (on devices[r]): holds the reduced sum
    // after allreduce_bf16()/allreduce_f32(). Distinct region from
    // rank_buffer(r). nullptr if !ok().
    void* rank_result(int rank) const noexcept;

    // All-reduce (sum) of `count` BF16 elements across all ranks' rank_buffer()
    // inputs; result lands in each rank_result(). ONE kernel per rank: an
    // in-kernel device-flag barrier (tp_allreduce.cuh) ensures all inputs are
    // written before any rank issues its multimem.ld_reduce — no host events,
    // no cross-stream graph edges. streams size must equal the device count.
    void allreduce_bf16(std::size_t count, const std::vector<void*>& streams);

    // f32 mirror of allreduce_bf16 — same one-shot kernel shape, 128-bit packs of
    // 4 floats via multimem.ld_reduce.global.add.v4.f32. `count` must be a
    // multiple of 4 (the ctor's multiple-of-8 gate already guarantees it). Exists
    // because Kimi K3 keeps its residual stream f32 by design.
    void allreduce_f32(std::size_t count, const std::vector<void*>& streams);
    // slot >= 0 reduces out of that input slot with ONE rendezvous instead of
    // two; slot < 0 is main's two-barrier kernel over slot 0. The caller owns
    // the rotation and must have written its partial into rank_buffer(rank, slot).
    void allreduce_f32(std::size_t count, const std::vector<void*>& streams, int slot);

    // Opaque PIMPL, defined in the .cu. Declared public so the file-local
    // setup/launch helpers can name the type; impl_ itself stays private.
    struct Impl;

private:
    Impl* impl_ = nullptr;
    bool ok_ = false;
};

}  // namespace sparkinfer::tp
