#pragma once
// Programmatic Dependent Launch for the tensor-parallel collective.
//
// ---------------------------------------------------------------------------
// WHY THE COLLECTIVE IS THE ONE KERNEL LEFT OUT
// ---------------------------------------------------------------------------
// Every kernel K3's decode launches now goes out programmatically: all 72 launch
// sites in kernels/csrc/cuda/kimi_k3/ route through k3_pdl_launch, and 40 kernel
// bodies open with k3_pdl_sync(). The all-reduce does not. It is launched with a
// plain <<<>>> from this directory, and it is not a small exception — it runs 185
// times per token per rank, more often than anything except the elementwise tails.
//
// So the decode chain is programmatic everywhere except at the one point where eight
// GPUs have to agree. That is the same shape as the gap #74 found in the collective
// default and #90 found in PDL itself: the fast path was written, and nothing routed
// to it.
//
// ---------------------------------------------------------------------------
// WHAT THIS DOES: the SUCCESSOR half, and only that
// ---------------------------------------------------------------------------
// tp_pdl_launch sets cudaLaunchAttributeProgrammaticStreamSerialization, so the reduce
// kernel's grid may begin spinning up — scheduling blocks onto SMs, allocating
// registers, faulting in its first instructions — while the kernel that fills this
// rank's input buffer is still draining. tp_pdl_sync() is where it stops and waits.
//
// Placing that call as the FIRST statement of the reduce kernel is the whole safety
// argument, and it is a one-line property a reviewer can check rather than an argument
// about which loads alias what: the very next thing the kernel does is publish a flag
// telling eight peers "my input is written". If the sync were below that store, a grid
// that started early would announce an input its predecessor had not finished writing,
// and eight ranks would sum a partial buffer.
//
// ---------------------------------------------------------------------------
// WHAT THIS DELIBERATELY DOES NOT DO: the predecessor half
// ---------------------------------------------------------------------------
// cudaTriggerProgrammaticLaunchCompletion() is defined in kernels/csrc/cuda/kimi_k3/
// k3_pdl.cuh and called from nowhere in K3, so today every programmatic port in the
// decode activates on the only other condition the driver accepts — all blocks having
// TERMINATED. The one-shot reduce looks like the ideal place to change that, because
// its result is final one instruction after the reduce loop and everything following is
// an eight-way exit barrier that tp_allreduce.cuh itself describes as protecting only
// the shared INPUT.
//
// It is not safe here, and peer_oneshot_allreduce.cu carries the full argument at the
// point where the call would go. In short: under the zero-copy path the next phase
// writes its partial straight into this rank's in_buf, which is the buffer the peers
// are reading, and collective.cpp's stated reason that this is safe is precisely the
// stream ordering an early trigger removes. Programmatic edges are pairwise, so
// releasing one successor releases the whole chain behind it.
//
// Recovering the exit barrier is a real opportunity and it needs a second input buffer
// to ping-pong between collectives — a different change, with a different proof, and
// not this one.
//
// ---------------------------------------------------------------------------
// IT IS NOT CUDA-GRAPH CAPTURE
// ---------------------------------------------------------------------------
// The decode step is already captured per rank (#89). That is a different mechanism
// solving a different half: capture removes the HOST cost of submitting a launch, PDL
// removes the DEVICE-side gap between two launches already submitted. This repository
// has measured them composing — 22.12 -> 26.08 with capture, 26.08 -> 29.97 with PDL on
// top of it. Under capture the programmatic attribute becomes a programmatic EDGE
// between the two graph nodes (cudaGraphKernelNodePortProgrammatic), so it is the same
// mechanism either way.

#include <cuda_runtime.h>

#include <cstdlib>

namespace sparkinfer::tp::tpar {

// SPARKINFER_K3_TP_PDL=0 turns this factor off and restores the exact <<<>>> launch.
// Its own switch, not shared with SPARKINFER_K3_PDL, so the collective and the kernels
// can be A/B'd independently inside ONE binary and one model load.
//
// Read once into a function-local static: a getenv per launch would run 185 times a
// token, and a value that could change mid-run would be worse than either arm.
inline bool tp_pdl_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("SPARKINFER_K3_TP_PDL");
        return !(e && e[0] == '0');
    }();
    return on;
}

// Wait for the predecessor's writes. A no-op unless this kernel was launched
// programmatically, and on anything below sm_90 — so a kernel carrying the call is
// byte-identical when launched the ordinary way, which is what makes the toggle an A/B
// of one binary rather than of two programs.
__device__ __forceinline__ void tp_pdl_sync() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif
}

// Launch programmatically when the factor is on, and exactly as <<<>>> would when it is
// off. Mirrors k3_pdl_launch so the two halves of the decode chain are launched by the
// same idiom rather than by two that have to be kept in step by hand.
template <typename Kernel, typename... Args>
inline void tp_pdl_launch(dim3 grid, dim3 block, size_t smem, cudaStream_t stream,
                          Kernel kernel, Args... args) {
    if (!tp_pdl_enabled()) {
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

}  // namespace sparkinfer::tp::tpar
