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
//
// STORAGE ceiling only. vLLM's 36 was a budget in THREADS -- 36 blocks of 512 is
// 18432 -- and this kernel now launches 64-thread blocks, so 36 CTAs spends an
// eighth of the contention that number was chosen to bound. Signal is the reason
// the ceiling has to be a compile-time constant (it sizes four peer-visible flag
// arrays, ~6.4 KB at 64); the ceiling that is actually applied is grid_for's
// runtime cap, which still defaults to 36.
constexpr int kMaxBlocks = 64;

// Per-rank synchronization buffer, peer-accessible from every rank. Two counter
// sets (start/end) are needed because a peer block can reach the second sync
// before this block has passed the first; alternating arrays avoid the race.
//
// A THIRD ARRAY, for the single-barrier f32 kernel. The two-barrier kernels get
// their alternation for free: start and end are used strictly in turn, so a rank
// racing ahead always writes the array its peers are not spinning on. Delete the
// exit barrier and that property has to be supplied deliberately, because the
// spin compares for EQUALITY — a rank that stores flag v+1 to a peer still
// waiting for v does not merely reorder the two, it hangs the peer forever.
// Rotating the array by the same index that rotates the input buffer restores
// the guarantee: consecutive barriers never share an array, and the rotation
// period is 3, which is also what the buffer needs (see kSlotCount).
struct Signal {
    alignas(128) FlagType start[kMaxBlocks][8];
    alignas(128) FlagType end[kMaxBlocks][8];
    alignas(128) FlagType mid[kMaxBlocks][8];
    alignas(128) FlagType _flag[kMaxBlocks];  // per-rank incremental flag
};

// How many input slots the ping-pong rotates through, and therefore how many
// counter arrays. THREE, NOT TWO, and the reason is the token boundary.
//
// A captured CUDA graph bakes the pointers it saw, so the slot sequence RESETS
// every token instead of running free. K3 issues an ODD number of collectives
// per token (93 attention + 92 MoE = 185 under the default policy), so with two
// slots the last collective of token N and the first of token N+1 land on the
// same slot with NOTHING between them — no barrier, no reduce, and (see
// kimi_k3_tp.cpp) no cross-rank sync at the token boundary either, because only
// rank 0's stream is joined. A fast rank's layer-0 attention output would then
// overwrite a buffer a slow peer is still summing. Note which way round this is:
// the EAGER path is safe (the counter runs free and parity keeps alternating);
// it is graph capture that creates the collision.
//
// The requirement is a minimum reuse distance of 2 across the wrap as well as
// within the token: with C collectives and P slots that is (C-1) % P >= 1.
// 184 % 2 == 0 fails; 184 % 3 == 1 holds. The host asserts it rather than
// trusting this arithmetic to survive a policy change — see k3_coll_1bar.h.
constexpr int kSlotCount = 3;

// Which counter array a given rotation index uses.
__device__ __forceinline__ FlagType* flag_row(Signal* s, int arr, int blk) {
    return arr == 0 ? s->start[blk] : (arr == 1 ? s->end[blk] : s->mid[blk]);
}

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
// `arr` selects the counter array (0 start, 1 end, 2 mid). It defaults to 0, so
// every existing two-barrier caller keeps exactly the arrays and the ordering it
// had; only the single-barrier f32 kernel passes a rotating index.
template <int N>
__device__ __forceinline__ void barrier_at_start(const RankSignals<N>& sg,
                                                  Signal* self_sg, int rank,
                                                  int arr = 0) {
    uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
    if (threadIdx.x < N) {
        FlagType* peer = &flag_row(sg.sg[threadIdx.x], arr, blockIdx.x)[rank];
        FlagType* self = &flag_row(self_sg, arr, blockIdx.x)[threadIdx.x];
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
// f32 mirror of packed_reduce_bf16 below: a 128-bit pack is 4 floats and there is
// no up/downcast. The accumulation ORDER is identical — rank 0's pack first, then
// ranks 1..N-1 in index order — so the two dtypes differ in precision, never in the
// reduction tree. (Deterministic across calls; NOT the same rounding as NCCL's ring,
// which is why a backend switch moves logits in the last bits.)
template <int N>
__device__ __forceinline__ array_t<float, 4>
packed_reduce_f32(const array_t<float, 4>* const ptrs[N], int idx) {
    array_t<float, 4> acc = ptrs[0][idx];
#pragma unroll
    for (int r = 1; r < N; r++) {
        const array_t<float, 4> v = ptrs[r][idx];
#pragma unroll
        for (int i = 0; i < 4; i++) acc.data[i] += v.data[i];
    }
    return acc;
}

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
