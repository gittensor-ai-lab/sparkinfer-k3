// Single-kernel one-shot all-reduce with in-kernel flag barrier — impl.
// vLLM comm trick 2/6 (#74). See header + tp_allreduce.cuh.

#include "sparkinfer/tp/peer_oneshot_allreduce.h"
#include "tp_allreduce.cuh"
#include "tp_error.hpp"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>

namespace sparkinfer::tp {

using tpar::Signal;
using tpar::RankSignals;
using tpar::kMaxBlocks;
using tpar::array_t;
using tpar::packed_reduce_bf16;
using tpar::barrier_at_start;
using tpar::barrier_at_end;

// __global__ kernels live in a named namespace (matching the codebase's other
// kernel TUs), not an anonymous one.
namespace detail {

template <int N>
struct PeerInputs {
    const __nv_bfloat16* p[N];
};

// THE trick: barrier_at_start (waits until every rank's input is written) →
// read all N peers + sum FP32 (128-bit packed) → barrier_at_end. One kernel per
// rank; cross-rank sync is entirely in-kernel, no host events.
template <int N>
__global__ void __launch_bounds__(512, 1)
    peer_oneshot_allreduce_kernel(PeerInputs<N> peers, RankSignals<N> sg,
                                  Signal* self_sg, int rank,
                                  __nv_bfloat16* __restrict__ out, int n_vec) {
    using P = array_t<__nv_bfloat16, 8>;
    const P* ptrs[N];
#pragma unroll
    for (int r = 0; r < N; r++) ptrs[r] = reinterpret_cast<const P*>(peers.p[r]);

    barrier_at_start<N>(sg, self_sg, rank);
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n_vec;
         idx += gridDim.x * blockDim.x) {
        reinterpret_cast<P*>(out)[idx] = packed_reduce_bf16<N>(ptrs, idx);
    }
    barrier_at_end<N, true>(sg, self_sg, rank);
}

// Same reduce, NO in-kernel barrier — used only by the host-event baseline path
// in the bench so the A/B isolates the barrier mechanism.
template <int N>
__global__ void __launch_bounds__(512, 1)
    peer_oneshot_reduce_only_kernel(PeerInputs<N> peers,
                                    __nv_bfloat16* __restrict__ out, int n_vec) {
    using P = array_t<__nv_bfloat16, 8>;
    const P* ptrs[N];
#pragma unroll
    for (int r = 0; r < N; r++) ptrs[r] = reinterpret_cast<const P*>(peers.p[r]);
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n_vec;
         idx += gridDim.x * blockDim.x) {
        reinterpret_cast<P*>(out)[idx] = packed_reduce_bf16<N>(ptrs, idx);
    }
}

template <int N>
struct PeerTmps {
    __nv_bfloat16* p[N];
};

// Two-shot (vLLM cross_device_reduce_2stage): reduce-scatter into per-rank tmp,
// then all-gather. ~2x payload/rank vs one-shot's Nx — for large (prefill)
// messages. Pointers are rank-rotated for load balance (each partition is
// reduced by exactly one rank, so the rotation doesn't break determinism).
template <int N>
__global__ void __launch_bounds__(512, 1)
    peer_twoshot_allreduce_kernel(PeerInputs<N> ins, PeerTmps<N> tmps_all,
                                  RankSignals<N> sg, Signal* self_sg, int rank,
                                  __nv_bfloat16* __restrict__ out, int n_vec) {
    using P = array_t<__nv_bfloat16, 8>;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int part = n_vec / N;
    const int start = rank * part;
    const int end = (rank == N - 1) ? n_vec : start + part;
    const int largest = part + n_vec % N;
    const P* ptrs[N];
    P* tmps[N];
#pragma unroll
    for (int i = 0; i < N; i++) {
        int t = (rank + i) % N;
        ptrs[i] = reinterpret_cast<const P*>(ins.p[t]);
        tmps[i] = reinterpret_cast<P*>(tmps_all.p[t]);
    }
    P* tmp_out = tmps[0];  // this rank's own tmp

    barrier_at_start<N>(sg, self_sg, rank);
    // stage 1: reduce-scatter — reduce my partition reading all peers' inputs.
    for (int idx = start + tid; idx < end; idx += stride)
        tmp_out[idx - start] = packed_reduce_bf16<N>(ptrs, idx);
    barrier_at_end<N, false>(sg, self_sg, rank);
    // stage 2: all-gather — same tid gathers the reduced partitions from peers.
    for (int idx = tid; idx < largest; idx += stride) {
#pragma unroll
        for (int i = 0; i < N; i++) {
            int g = (rank + i) % N;
            if (g == N - 1 || idx < part)
                reinterpret_cast<P*>(out)[g * part + idx] = tmps[i][idx];
        }
    }
}

int grid_for(int n_vec, int block) {
    int g = (n_vec + block - 1) / block;
    return g < kMaxBlocks ? g : kMaxBlocks;  // vLLM block cap (NVLink contention)
}

}  // namespace detail

struct PeerOneShotAllreduce::Impl {
    std::vector<int> devices;
    std::size_t count = 0;
    std::vector<__nv_bfloat16*> in_buf;   // peer-accessible inputs (per rank)
    std::vector<__nv_bfloat16*> out_buf;  // per-rank outputs
    std::vector<__nv_bfloat16*> tmp_buf;  // peer-accessible two-shot scratch
    std::vector<Signal*> sig;             // peer-accessible sync (per rank)
};

namespace detail {
// Build the launch args and fire one kernel per rank. Templated on N so the
// barrier/reduce are the compile-time-unrolled vLLM forms. (Re-opens detail;
// these host launchers reference the kernels above.)
template <int N>
void launch_in_kernel(PeerOneShotAllreduce::Impl* s, int n_vec,
                      const std::vector<void*>& streams) {
    PeerInputs<N> peers{};
    RankSignals<N> sg{};
    for (int r = 0; r < N; r++) { peers.p[r] = s->in_buf[r]; sg.sg[r] = s->sig[r]; }
    const int block = 512;
    const int grid = grid_for(n_vec, block);
    for (int r = 0; r < N; r++) {
        SPARKINFER_TP_CUDA_CHECK(cudaSetDevice(s->devices[r]));
        peer_oneshot_allreduce_kernel<N><<<grid, block, 0,
            reinterpret_cast<cudaStream_t>(streams[r])>>>(
            peers, sg, s->sig[r], r, s->out_buf[r], n_vec);
        SPARKINFER_TP_CUDA_CHECK(cudaGetLastError());
    }
}

// A/B baseline: host-event 8-way cross-stream barrier + reduce-only kernel.
template <int N>
void launch_host_barrier(PeerOneShotAllreduce::Impl* s, int n_vec,
                         const std::vector<void*>& streams,
                         const std::vector<void*>& events) {
    PeerInputs<N> peers{};
    for (int r = 0; r < N; r++) peers.p[r] = s->in_buf[r];
    for (int r = 0; r < N; r++) {
        SPARKINFER_TP_CUDA_CHECK(cudaSetDevice(s->devices[r]));
        SPARKINFER_TP_CUDA_CHECK(cudaEventRecord(
            reinterpret_cast<cudaEvent_t>(events[r]),
            reinterpret_cast<cudaStream_t>(streams[r])));
    }
    for (int r = 0; r < N; r++) {
        SPARKINFER_TP_CUDA_CHECK(cudaSetDevice(s->devices[r]));
        for (int q = 0; q < N; q++) {
            if (q == r) continue;
            SPARKINFER_TP_CUDA_CHECK(cudaStreamWaitEvent(
                reinterpret_cast<cudaStream_t>(streams[r]),
                reinterpret_cast<cudaEvent_t>(events[q]), 0));
        }
    }
    const int block = 512;
    const int grid = grid_for(n_vec, block);
    for (int r = 0; r < N; r++) {
        SPARKINFER_TP_CUDA_CHECK(cudaSetDevice(s->devices[r]));
        peer_oneshot_reduce_only_kernel<N><<<grid, block, 0,
            reinterpret_cast<cudaStream_t>(streams[r])>>>(
            peers, s->out_buf[r], n_vec);
        SPARKINFER_TP_CUDA_CHECK(cudaGetLastError());
    }
}

template <int N>
void launch_twoshot(PeerOneShotAllreduce::Impl* s, int n_vec,
                    const std::vector<void*>& streams) {
    PeerInputs<N> ins{};
    PeerTmps<N> tmps{};
    RankSignals<N> sg{};
    for (int r = 0; r < N; r++) {
        ins.p[r] = s->in_buf[r];
        tmps.p[r] = s->tmp_buf[r];
        sg.sg[r] = s->sig[r];
    }
    const int block = 512;
    const int grid = grid_for(n_vec, block);
    for (int r = 0; r < N; r++) {
        SPARKINFER_TP_CUDA_CHECK(cudaSetDevice(s->devices[r]));
        peer_twoshot_allreduce_kernel<N><<<grid, block, 0,
            reinterpret_cast<cudaStream_t>(streams[r])>>>(
            ins, tmps, sg, s->sig[r], r, s->out_buf[r], n_vec);
        SPARKINFER_TP_CUDA_CHECK(cudaGetLastError());
    }
}
}  // namespace detail

PeerOneShotAllreduce::PeerOneShotAllreduce(const std::vector<int>& devices,
                                           std::size_t count) {
    impl_ = new Impl();
    impl_->devices = devices;
    impl_->count = count;
    const int N = static_cast<int>(devices.size());
    if (N != 2 && N != 4 && N != 8) {
        std::fprintf(stderr, "[peer_oneshot] N=%d unsupported (2/4/8)\n", N);
        return;
    }
    if (count % 8 != 0) {
        std::fprintf(stderr, "[peer_oneshot] count %zu not multiple of 8\n", count);
        return;
    }
    impl_->in_buf.assign(N, nullptr);
    impl_->out_buf.assign(N, nullptr);
    impl_->tmp_buf.assign(N, nullptr);
    impl_->sig.assign(N, nullptr);
    const std::size_t bytes = count * sizeof(__nv_bfloat16);
    for (int r = 0; r < N; r++) {
        if (cudaSetDevice(devices[r]) != cudaSuccess) return;
        // Enable peer access from devices[r] to every other rank.
        for (int q = 0; q < N; q++) {
            if (q == r) continue;
            cudaError_t e = cudaDeviceEnablePeerAccess(devices[q], 0);
            if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
                std::fprintf(stderr, "[peer_oneshot] no peer access %d->%d\n",
                             devices[r], devices[q]);
                return;
            }
            (void)cudaGetLastError();  // clear sticky AlreadyEnabled
        }
        if (cudaMalloc(&impl_->in_buf[r], bytes) != cudaSuccess) return;
        if (cudaMalloc(&impl_->out_buf[r], bytes) != cudaSuccess) return;
        if (cudaMalloc(&impl_->tmp_buf[r], bytes) != cudaSuccess) return;
        if (cudaMalloc(&impl_->sig[r], sizeof(Signal)) != cudaSuccess) return;
        if (cudaMemset(impl_->sig[r], 0, sizeof(Signal)) != cudaSuccess) return;
    }
    for (int r = 0; r < N; r++) {
        if (cudaSetDevice(devices[r]) != cudaSuccess) return;
        if (cudaDeviceSynchronize() != cudaSuccess) return;
    }
    ok_ = true;
}

PeerOneShotAllreduce::~PeerOneShotAllreduce() noexcept {
    if (impl_) {
        const int N = static_cast<int>(impl_->devices.size());
        for (int r = 0; r < N; r++) {
            if (cudaSetDevice(impl_->devices[r]) != cudaSuccess) continue;
            if (r < (int)impl_->in_buf.size() && impl_->in_buf[r]) cudaFree(impl_->in_buf[r]);
            if (r < (int)impl_->out_buf.size() && impl_->out_buf[r]) cudaFree(impl_->out_buf[r]);
            if (r < (int)impl_->tmp_buf.size() && impl_->tmp_buf[r]) cudaFree(impl_->tmp_buf[r]);
            if (r < (int)impl_->sig.size() && impl_->sig[r]) cudaFree(impl_->sig[r]);
        }
    }
    delete impl_;
    impl_ = nullptr;
}

void* PeerOneShotAllreduce::rank_buffer(int rank) const noexcept {
    if (!ok_ || rank < 0 || rank >= (int)impl_->in_buf.size()) return nullptr;
    return impl_->in_buf[rank];
}
void* PeerOneShotAllreduce::rank_result(int rank) const noexcept {
    if (!ok_ || rank < 0 || rank >= (int)impl_->out_buf.size()) return nullptr;
    return impl_->out_buf[rank];
}

void PeerOneShotAllreduce::allreduce_bf16(std::size_t count,
                                          const std::vector<void*>& streams) {
    if (!ok_) return;
    const int N = static_cast<int>(impl_->devices.size());
    const int n_vec = static_cast<int>(count / 8);
    switch (N) {
        case 2: detail::launch_in_kernel<2>(impl_, n_vec, streams); break;
        case 4: detail::launch_in_kernel<4>(impl_, n_vec, streams); break;
        case 8: detail::launch_in_kernel<8>(impl_, n_vec, streams); break;
        default: break;
    }
}

void PeerOneShotAllreduce::allreduce_bf16_host_barrier(
    std::size_t count, const std::vector<void*>& streams,
    const std::vector<void*>& barrier_events) {
    if (!ok_) return;
    const int N = static_cast<int>(impl_->devices.size());
    const int n_vec = static_cast<int>(count / 8);
    switch (N) {
        case 2: detail::launch_host_barrier<2>(impl_, n_vec, streams, barrier_events); break;
        case 4: detail::launch_host_barrier<4>(impl_, n_vec, streams, barrier_events); break;
        case 8: detail::launch_host_barrier<8>(impl_, n_vec, streams, barrier_events); break;
        default: break;
    }
}

void PeerOneShotAllreduce::allreduce_bf16_twoshot(
    std::size_t count, const std::vector<void*>& streams) {
    if (!ok_) return;
    const int N = static_cast<int>(impl_->devices.size());
    const int n_vec = static_cast<int>(count / 8);
    switch (N) {
        case 2: detail::launch_twoshot<2>(impl_, n_vec, streams); break;
        case 4: detail::launch_twoshot<4>(impl_, n_vec, streams); break;
        case 8: detail::launch_twoshot<8>(impl_, n_vec, streams); break;
        default: break;
    }
}

}  // namespace sparkinfer::tp
