// In-kernel device-flag barrier + vectorized FP32 reduce for TP all-reduce.
//
// This is the core of the vLLM communication tricks ported into sparkinfer (#74).
// The decode all-reduce in qwen2_forward_tp.cu currently barriers across the 8
// ranks with HOST events (cudaEventRecord + cudaStreamWaitEvent): per all-reduce
// that's ~8 event-records + up to 56 cross-stream waits + 8 kernels, which the
// joint decode CUDA graph captures as ~64 nodes + cross-stream edges that
// serialize all 8 streams — ×160 all-reduces/token. vLLM v0.22.0 instead does
// the entire all-reduce as ONE kernel per rank, with the cross-rank barrier done
// *inside* the kernel via device-memory flags (st.release.sys / ld.acquire.sys
// on peer Signal buffers). No host orchestration, no cross-stream graph edges.
// That barrier difference — not the reduce math, not multimem — is the leading
// suspect for sparkinfer's 38.5 TPS vs vLLM's 92 at TP=8 (both use the same
// peer-read-and-sum one-shot for the 16 KiB decode payload; see
// docs/T3_MULTIMEM_ALLREDUCE.md and the run_summary).
//
// Adapted from vLLM `csrc/custom_all_reduce.cuh` (Apache-2.0, Copyright the vLLM
// project): the Signal layout, the dual-counter barrier, the acquire/release
// flag ops, and the packed FP32-accumulate reduce. Kept faithful so it inherits
// vLLM's correctness; only namespacing/types are cacheon-ized.
//
// UNTESTED on the 8x H200 target as committed (node off). Validate via
// tp_allreduce_bench before wiring into the decode path.

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>

namespace sparkinfer::tp::tpar {

// Counter may overflow, but unsigned overflow is well-defined and the barrier
// only ever compares for equality with the freshly-incremented value.
using FlagType = uint32_t;

// vLLM uses 36 blocks for the all-reduce: "too many SMs cause contention on the
// NVLink bus." We cap Signal storage to that and clamp the launch likewise.
constexpr int kMaxBlocks = 36;

// Per-rank synchronization buffer, peer-accessible from every rank. Two counter
// sets (start/end) are needed because a peer block can reach the second sync
// before this block has passed the first; alternating arrays avoid the race.
struct Signal {
    alignas(128) FlagType start[kMaxBlocks][8];
    alignas(128) FlagType end[kMaxBlocks][8];
    alignas(128) FlagType _flag[kMaxBlocks];  // per-rank incremental flag
};

template <int N>
struct RankSignals {
    Signal* sg[N];
};

// ---- flag load/store with the right memory ordering --------------------------
__device__ __forceinline__ void st_flag_release(FlagType* addr, FlagType flag) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(flag), "l"(addr));
#else
    asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" ::"r"(flag),
                 "l"(addr));
#endif
}
__device__ __forceinline__ FlagType ld_flag_acquire(FlagType* addr) {
    FlagType flag;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(addr));
#else
    asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;"
                 : "=r"(flag)
                 : "l"(addr));
#endif
    return flag;
}
__device__ __forceinline__ void st_flag_volatile(FlagType* addr, FlagType flag) {
    asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(addr));
}
__device__ __forceinline__ FlagType ld_flag_volatile(FlagType* addr) {
    FlagType flag;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(flag) : "l"(addr));
    return flag;
}

// First barrier: no visibility guarantees needed for prior accesses, so volatile
// flags suffice. Each of the first N threads signals its peer and spins on its
// own counter. After this returns, every rank's input is guaranteed written.
template <int N>
__device__ __forceinline__ void barrier_at_start(const RankSignals<N>& sg,
                                                  Signal* self_sg, int rank) {
    uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
    if (threadIdx.x < N) {
        FlagType* peer = &sg.sg[threadIdx.x]->start[blockIdx.x][rank];
        FlagType* self = &self_sg->start[blockIdx.x][threadIdx.x];
        st_flag_volatile(peer, flag);
        while (ld_flag_volatile(self) != flag);
    }
    __syncthreads();
    if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

// Second/final barrier. When final_sync=false it publishes prior writes with
// release/acquire so peers can read this rank's results; when true (one-shot,
// each rank already holds the full result locally) plain volatile is enough —
// its only job is to keep a fast rank from clobbering the shared input before a
// slow rank finishes reading it.
template <int N, bool final_sync>
__device__ __forceinline__ void barrier_at_end(const RankSignals<N>& sg,
                                               Signal* self_sg, int rank) {
    __syncthreads();
    uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
    if (threadIdx.x < N) {
        FlagType* peer = &sg.sg[threadIdx.x]->end[blockIdx.x][rank];
        FlagType* self = &self_sg->end[blockIdx.x][threadIdx.x];
        if constexpr (!final_sync) {
            st_flag_release(peer, flag);
            while (ld_flag_acquire(self) != flag);
        } else {
            st_flag_volatile(peer, flag);
            while (ld_flag_volatile(self) != flag);
        }
    }
    if constexpr (!final_sync) __syncthreads();
    if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

// ---- packed 128-bit vectorized reduce (ld.128/st.128, FP32 accumulate) -------
template <typename T, int sz>
struct __align__(sizeof(T) * sz) array_t {
    T data[sz];
    static constexpr int size = sz;
    using type = T;
};

// 16-byte packed load/store type (8 bf16 → one 128-bit transaction) and FP32
// accumulator. This is vLLM's "goal: generate ld.128 and st.128 instructions";
// sparkinfer's current peer_sum kernel loads one bf16 at a time instead.
template <typename T>
struct packed_t {
    using P = array_t<T, 16 / sizeof(T)>;
    using A = array_t<float, 16 / sizeof(T)>;
};

template <int N>
__device__ __forceinline__ array_t<float, N> upcast_bf16(array_t<__nv_bfloat16, N> v) {
    array_t<float, N> o;
#pragma unroll
    for (int i = 0; i < N; i++) o.data[i] = __bfloat162float(v.data[i]);
    return o;
}
template <int N>
__device__ __forceinline__ array_t<__nv_bfloat16, N> downcast_bf16(array_t<float, N> v) {
    array_t<__nv_bfloat16, N> o;
#pragma unroll
    for (int i = 0; i < N; i++) o.data[i] = __float2bfloat16(v.data[i]);
    return o;
}

// Sum the same packed index across all N ranks in FP32. Note: peers are read in
// a fixed order (NOT rank-rotated) so every rank accumulates in the same order
// and produces bitwise-identical results — matching vLLM's one-shot.
template <int N>
__device__ __forceinline__ array_t<__nv_bfloat16, 8>
packed_reduce_bf16(const array_t<__nv_bfloat16, 8>* const ptrs[N], int idx) {
    array_t<float, 8> acc = upcast_bf16<8>(ptrs[0][idx]);
#pragma unroll
    for (int r = 1; r < N; r++) {
        array_t<float, 8> v = upcast_bf16<8>(ptrs[r][idx]);
#pragma unroll
        for (int i = 0; i < 8; i++) acc.data[i] += v.data[i];
    }
    return downcast_bf16<8>(acc);
}

}  // namespace sparkinfer::tp::tpar
