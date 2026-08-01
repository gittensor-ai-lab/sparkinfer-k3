// Single-kernel one-shot all-reduce with an in-kernel device-flag barrier —
// vLLM comm trick 2/6 (#74). This is the direct, same-algorithm replacement for
// sparkinfer's host-event peer-N decode all-reduce: every rank reads all N peers'
// inputs and sums in FP32 (128-bit vectorized), exactly like vLLM's
// cross_device_reduce_1stage and sparkinfer's peer_sum_*, but the cross-rank
// barrier is done INSIDE the kernel (see tp_allreduce.cuh) so the whole
// all-reduce is ONE kernel per rank — no cudaEventRecord/cudaStreamWaitEvent,
// no cross-stream graph edges.
//
// Self-contained and isolated (own peer-access + buffer setup) so it can be
// validated by tp_allreduce_bench before being wired into qwen2_forward_tp.cu.
// Single-process multi-GPU only (matches sparkinfer's TP runtime). UNTESTED on the
// 8x H200 target as committed.

#pragma once

#include <cstddef>
#include <vector>

namespace sparkinfer::tp {

// One-shot all-reduce over a fixed device set, owning peer-accessible input
// buffers + sync Signals. Caller writes rank r's input into rank_buffer(r),
// calls allreduce_bf16(), reads the reduced sum from rank_result(r). Input and
// output are separate regions: in the one-shot, peers read every rank's input
// while each rank writes its own output, so a separate output avoids a
// read-after-write race (matches vLLM's out-of-place custom_all_reduce).
class PeerOneShotAllreduce {
public:
    // Consulted by the collective adapter at compile time; multimem stays false
    // until its own f32 path exists, so only this backend accepts K3's stream.
    static constexpr bool kSupportsF32 = true;
    // `count` = max elements (hidden_size); must be a multiple of 8 (128-bit
    // packed). Enables peer access across all device pairs. Check ok().
    PeerOneShotAllreduce(const std::vector<int>& devices, std::size_t count);
    ~PeerOneShotAllreduce() noexcept;

    PeerOneShotAllreduce(const PeerOneShotAllreduce&) = delete;
    PeerOneShotAllreduce& operator=(const PeerOneShotAllreduce&) = delete;

    bool ok() const noexcept { return ok_; }

    // Rank r's input pointer (on devices[r]) — write input here.
    void* rank_buffer(int rank) const noexcept;
    // Rank r's output pointer (on devices[r]) — reduced sum lands here.
    void* rank_result(int rank) const noexcept;

    // All-reduce (sum) of `count` bf16 elements across all ranks. ONE kernel per
    // rank; the in-kernel flag barrier replaces any host-side barrier. streams
    // size must equal the device count. NO barrier events — that's the point.
    void allreduce_bf16(std::size_t count, const std::vector<void*>& streams);

    // f32 mirror of allreduce_bf16 — same one-shot kernel shape, 128-bit packs of
    // 4 floats instead of 8 bf16. `count` must be a multiple of 4 (the ctor's
    // multiple-of-8 gate already guarantees it). Exists because Kimi K3 keeps its
    // residual stream f32 by design, so the bf16-only surface made every K3
    // collective fall back to NCCL.
    void allreduce_f32(std::size_t count, const std::vector<void*>& streams);

    // A/B baseline: the SAME packed reduce but synchronized with a host-event
    // 8-way cross-stream barrier (cudaEventRecord + cudaStreamWaitEvent), i.e.
    // sparkinfer's current mechanism. Exists only so tp_allreduce_bench can measure
    // the barrier-mechanism delta with the reduce held constant. barrier_events
    // size must equal the device count.
    void allreduce_bf16_host_barrier(std::size_t count,
                                     const std::vector<void*>& streams,
                                     const std::vector<void*>& barrier_events);

    // Two-shot all-reduce (vLLM cross_device_reduce_2stage): reduce-scatter to a
    // per-rank tmp + all-gather. Each rank moves ~2x the payload over NVLink
    // instead of one-shot's Nx, so it wins for LARGE messages — i.e. the prefill
    // all-reduce (vLLM uses it above 256 KB at TP=8), NOT the 16 KB decode AR.
    // Same in-kernel barrier; lowest-priority of the set for the decode score.
    void allreduce_bf16_twoshot(std::size_t count,
                                const std::vector<void*>& streams);

    // Opaque PIMPL, defined in the .cu. Declared public so the file-local
    // launch helpers can name the type; impl_ itself stays private (they
    // receive it from the member methods).
    struct Impl;

private:
    Impl* impl_ = nullptr;
    bool ok_ = false;
};

}  // namespace sparkinfer::tp
