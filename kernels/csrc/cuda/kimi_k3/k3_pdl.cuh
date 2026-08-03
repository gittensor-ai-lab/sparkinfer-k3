#pragma once
// Programmatic Dependent Launch for the K3 decode chain.
//
// ---------------------------------------------------------------------------
// WHY THIS IS THE RIGHT TOOL FOR THIS DECODE
// ---------------------------------------------------------------------------
// A K3 token issues roughly 4,000 kernels per rank, all dependent, all on one stream,
// in 47 ms. That is ~12 us per kernel for work whose bytes justify one or two. The gap
// between two dependent kernels is not free: the successor's grid cannot begin spinning
// up — scheduling blocks onto SMs, allocating registers, faulting in its first
// instructions — until the predecessor's last block has retired.
//
// PDL removes exactly that serialisation. A kernel launched with
// cudaLaunchAttributeProgrammaticStreamSerialization may begin spinning up while its
// predecessor is still draining; cudaGridDependencySynchronize() inside it is where it
// waits for the predecessor's writes to be visible. Everything before that call
// overlaps the predecessor's tail.
//
// THIS IS NOT CUDA-GRAPH CAPTURE and does not overlap with it. Graphs remove the HOST
// cost of submitting a launch. PDL removes the DEVICE-side gap between two kernels that
// are already submitted. They attack different halves of the same 12 us.
//
// THE MACHINERY IS ALREADY IN THIS TREE AND K3 NEVER REACHED IT.
// kernels/csrc/cuda/moe/expert_ffn_q4k.cu has carried si_pdl_lc/si_pdl_sync and a
// cudaLaunchKernelEx wrapper since the Qwen work, with the comment "overlap a kernel's
// grid spin-up with its predecessor's tail to hide bs=1 decode launch latency (the
// ncu-confirmed bottleneck)". K3's decode is bs=1 by definition. This is the same
// shape of gap #74 found in the collective default: the fast path was written, and
// nothing routed to it.
//
// ---------------------------------------------------------------------------
// THE CONTRACT, AND WHERE IT CAN GO WRONG
// ---------------------------------------------------------------------------
// k3_pdl_sync() MUST be executed by every thread before ANY read of memory the
// predecessor wrote. Put it at the top of the kernel unless you have specifically
// arranged for the reads before it to be independent. Placing it too late is a data
// race that produces plausible wrong numbers, not a crash — so every kernel here calls
// it as its first statement, and the ordering claim is a one-line property a reviewer
// can check rather than an argument about which loads alias what.
//
// It is a NO-OP unless the kernel was launched programmatically, so a kernel carrying
// the call is byte-identical when launched the ordinary way. That is what makes
// SPARKINFER_K3_PDL=0 an exact A/B on one binary rather than a different program.

#include <cuda_runtime.h>

#include <cstdlib>

namespace sparkinfer {
namespace kernels {
namespace k3 {

// Wait until the predecessor kernel's writes are visible. No-op when not launched
// programmatically, and on anything below sm_90.
__device__ __forceinline__ void k3_pdl_sync() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif
}

// Signal that this kernel's dependent writes are complete, before it actually retires.
// Only sound where the caller knows the writes a successor needs are already done.
__device__ __forceinline__ void k3_pdl_launch_complete() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

inline bool k3_pdl_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("SPARKINFER_K3_PDL");
        return !(e && e[0] == '0');
    }();
    return on;
}

// Launch programmatically when PDL is on, and exactly as <<<>>> would when it is not.
template <typename Kernel, typename... Args>
inline void k3_pdl_launch(dim3 grid, dim3 block, size_t smem, cudaStream_t stream,
                          Kernel kernel, Args... args) {
    if (!k3_pdl_enabled()) {
        kernel<<<grid, block, smem, stream>>>(args...);
        return;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kernel, args...);
}

}  // namespace k3
}  // namespace kernels
}  // namespace sparkinfer
