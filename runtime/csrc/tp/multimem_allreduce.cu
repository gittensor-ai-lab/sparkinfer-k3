// One-shot NVSwitch multimem all-reduce — implementation. See the header and
// docs/TP8_DECODE_OPTIMIZATION.md (#74 T3) for the why.
//
// UNTESTED on hardware as committed. The CUDA multicast VMM sequence and the
// multimem.* PTX have several details that MUST be confirmed on the 8× H200
// target; each is marked `VERIFY:` below. Validate end-to-end with
// `multimem_allreduce_bench` (checks the reduced sum against an FP32 reference
// and times it vs the payload) BEFORE wiring this into qwen2_forward_tp.cu.
//
// Design (mirrors the peer-N all-reduce structure so it drops into the same
// host-orchestrated barrier the decode path already uses):
//   setup once: create a multicast object spanning all N devices; per device
//     create a PINNED physical allocation, bind it to the multicast, and map
//     two VAs — a unicast VA (this device's own bytes) and a multicast VA
//     (the switch-reduced view across all devices).
//   per all-reduce: caller writes rank r's input into the unicast input region;
//     a host event barrier ensures every rank's input is visible; each rank then
//     issues ONE multimem.ld_reduce per 8 elements over the multicast VA (the
//     NVSwitch sums across all ranks in hardware) and stores to its unicast
//     output region. Input/output are separate regions so the reduce never
//     races a peer still reading the inputs.

#include "sparkinfer/tp/multimem_allreduce.h"
#include "tp_allreduce.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <cstdio>

// Multicast VMM needs the CUDA driver API and CUDA >= 12.1.
#include <cuda.h>
#ifndef CUDA_VERSION
#define CUDA_VERSION 0
#endif
#define SPARKINFER_TP_MULTIMEM_DRIVER_OK (CUDA_VERSION >= 12010)

namespace sparkinfer::tp {

using tpar::Signal;
using tpar::RankSignals;
using tpar::barrier_at_start;
using tpar::barrier_at_end;

namespace {

constexpr int kVecBf16 = 8;  // bf16 per uint4 (one multimem.ld_reduce.v4.bf16x2)
constexpr int kVecF32  = 4;  // f32  per uint4 (one multimem.ld_reduce.v4.f32)

#if SPARKINFER_TP_MULTIMEM_DRIVER_OK
// Log + return false on driver-API error (setup is allowed to fail → fallback).
bool cu_ok(CUresult r, const char* what) {
    if (r == CUDA_SUCCESS) return true;
    const char* msg = nullptr;
    cuGetErrorString(r, &msg);
    std::fprintf(stderr, "[multimem] %s failed: %s\n", what, msg ? msg : "?");
    return false;
}
#endif

}  // namespace (anonymous: host-side helpers/constants)

// __global__ kernel in a named namespace (matches the codebase's kernel TUs).
namespace detail {

// multimem all-reduce kernel: in-kernel flag barrier (inputs ready) → each
// thread reduces 8 bf16 (one uint4) across ALL ranks in the NVSwitch via one
// multimem.ld_reduce, writes the result to the unicast output → barrier_at_end
// (don't let a fast rank clobber the multicast input before slow ranks read).
// One kernel per rank — same single-kernel, no-host-event structure as the peer
// one-shot (trick 1/2); multimem just replaces the N peer loads with one
// switch-reduced load.
//
// VERIFY (hardware): the PTX. We use the f32-accumulate form
//   multimem.ld_reduce.global.add.acc::f32.v4.bf16x2
// to match the FP32 accumulation of the peer-N path. If the assembler on the
// target rejects `.acc::f32` for this driver/arch, drop it (the bf16-accumulate
// form `multimem.ld_reduce.global.add.v4.bf16x2`) and RE-RUN Pass-2
// teacher-forcing — bf16 accumulation of an 8-way sum may move min_lp.
template <int N>
__global__ void multimem_allreduce_kernel(const uint4* __restrict__ mc_in,
                                          uint4* __restrict__ uc_out,
                                          RankSignals<N> sg, Signal* self_sg,
                                          int rank, std::size_t n_vec) {
    barrier_at_start<N>(sg, self_sg, rank);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_vec; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const uint4* addr = mc_in + i;
        uint4 res;
        asm volatile(
            "multimem.ld_reduce.global.add.acc::f32.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
            : "=r"(res.x), "=r"(res.y), "=r"(res.z), "=r"(res.w)
            : "l"(addr)
            : "memory");
        uc_out[i] = res;
    }
#else
    // multimem requires SM90+; host gates on multimem_allreduce_supported().
    (void)mc_in; (void)uc_out;
#endif
    barrier_at_end<N, true>(sg, self_sg, rank);
}

// f32 one-shot: same barrier shape as the bf16 kernel, but each thread reduces
// 4 floats (one uint4) via multimem.ld_reduce.global.add.v4.f32. Exists because
// Kimi K3 keeps its residual stream f32 by design — without this path every
// SPARKINFER_TP_BACKEND=multimem request fell back to NCCL at construction.
template <int N>
__global__ void multimem_allreduce_f32_kernel(const float* __restrict__ mc_in,
                                              float* __restrict__ uc_out,
                                              RankSignals<N> sg, Signal* self_sg,
                                              int rank, std::size_t n_vec) {
    barrier_at_start<N>(sg, self_sg, rank);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_vec; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const float* addr = mc_in + i * 4;
        float v0, v1, v2, v3;
        asm volatile(
            "multimem.ld_reduce.global.add.v4.f32 {%0,%1,%2,%3}, [%4];"
            : "=f"(v0), "=f"(v1), "=f"(v2), "=f"(v3)
            : "l"(addr)
            : "memory");
        float* out = uc_out + i * 4;
        out[0] = v0; out[1] = v1; out[2] = v2; out[3] = v3;
    }
#else
    (void)mc_in; (void)uc_out;
#endif
    barrier_at_end<N, true>(sg, self_sg, rank);
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Support probe
// ---------------------------------------------------------------------------
bool multimem_allreduce_supported(const std::vector<int>& devices) noexcept {
#if !SPARKINFER_TP_MULTIMEM_DRIVER_OK
    (void)devices;
    return false;
#else
    if (devices.size() < 2) return false;
    if (cuInit(0) != CUDA_SUCCESS) return false;

    // Step 1: the device attribute. NECESSARY BUT NOT SUFFICIENT — it reports what
    // the silicon can do, not what this process is permitted to do.
    for (int dev : devices) {
        int v = 0;
        if (cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev)
                != CUDA_SUCCESS || v == 0) {
            return false;
        }
    }

    // Step 2: actually bind a minimal multicast allocation, and tear it down.
    //
    // WHY THIS IS HERE. On an 8x H200 container without Fabric Manager access,
    // every device reports MULTICAST_SUPPORTED = 1 and then cuMulticastBindMem
    // fails with CUDA error 401 (CUDA_ERROR_ILLEGAL_STATE, "the operation cannot be
    // performed in the present state"). NCCL hits the same wall and says so:
    // "Failed to bind NVLink SHARP (NVLS) Multicast memory ... usually caused by a
    // system or configuration error in the Fabric Manager or NVSwitches."
    //
    // The attribute-only check therefore reported multimem as available on a node
    // where it cannot run, which made select_backend() choose it and left the real
    // failure to surface at construction time. Honest capability reporting matters
    // more than usual here: a benchmark that names its backend has to be right.
    //
    // Cost is one granularity-sized allocation, once, at startup.
    const int lead = devices[0];

    CUmulticastObjectProp mc_prop = {};
    mc_prop.numDevices = static_cast<unsigned>(devices.size());
    mc_prop.size = 0;
    mc_prop.handleTypes = CU_MEM_HANDLE_TYPE_NONE;

    std::size_t mc_gran = 0;
    if (cuMulticastGetGranularity(&mc_gran, &mc_prop,
                                  CU_MULTICAST_GRANULARITY_RECOMMENDED) != CUDA_SUCCESS
        || mc_gran == 0) {
        return false;
    }

    CUmemAllocationProp ap = {};
    ap.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    ap.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    ap.location.id = lead;

    std::size_t mem_gran = 0;
    if (cuMemGetAllocationGranularity(&mem_gran, &ap,
                                      CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) != CUDA_SUCCESS) {
        return false;
    }

    std::size_t bytes = mc_gran > mem_gran ? mc_gran : mem_gran;
    // Round up to a multiple of both.
    if (bytes % mc_gran) bytes += mc_gran - (bytes % mc_gran);

    mc_prop.size = bytes;
    CUmemGenericAllocationHandle mc = 0;
    if (cuMulticastCreate(&mc, &mc_prop) != CUDA_SUCCESS) return false;

    bool ok = true;
    for (int dev : devices) {
        if (cuMulticastAddDevice(mc, dev) != CUDA_SUCCESS) { ok = false; break; }
    }

    CUmemGenericAllocationHandle phys = 0;
    if (ok && cuMemCreate(&phys, bytes, &ap, 0) != CUDA_SUCCESS) {
        ok = false;
        phys = 0;
    }

    // THE step that fails with 401 when Fabric Manager is not reachable.
    bool bound = false;
    if (ok && cuMulticastBindMem(mc, /*mcOffset=*/0, phys, /*memOffset=*/0, bytes, 0)
                  == CUDA_SUCCESS) {
        bound = true;
    } else if (ok) {
        ok = false;
    }

    if (bound) cuMulticastUnbind(mc, lead, 0, bytes);
    if (phys) cuMemRelease(phys);
    cuMemRelease(mc);
    return ok;
#endif
}

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------
struct MultimemAllreduce::Impl {
    std::vector<int> devices;
    std::size_t count = 0;       // elements per region (hidden_size)
    std::size_t region_bytes = 0;  // bytes per input/output region
#if SPARKINFER_TP_MULTIMEM_DRIVER_OK
    CUmemGenericAllocationHandle mc_handle = 0;
    std::size_t alloc_bytes = 0;  // granularity-rounded total per device
    std::vector<CUmemGenericAllocationHandle> phys;  // per device
    std::vector<CUdeviceptr> uc_va;  // per device unicast VA (input @0, output @region_bytes)
    std::vector<CUdeviceptr> mc_va;  // per device multicast VA
    std::vector<Signal*> sig;        // per device in-kernel-barrier sync buffer
#endif
};

#if SPARKINFER_TP_MULTIMEM_DRIVER_OK
// Build the multicast object + per-device physical/maps. Returns false (and
// leaves ok_=false) on any failure so the caller falls back to peer-N.
static bool setup_multicast(MultimemAllreduce::Impl* s) {
    const int N = static_cast<int>(s->devices.size());

    // Region: sized for f32, the wider of the two element types this class now
    // reduces; bf16 uses the front half. One allocation serves both — same
    // discipline as PeerOneShotAllreduce.
    s->region_bytes = s->count * sizeof(float);
    const std::size_t want_bytes = s->region_bytes * 2;

    // Per-device PINNED allocation prop, reused for cuMemCreate below.
    CUmemAllocationProp ap = {};
    ap.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    ap.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    ap.location.id = s->devices[0];  // granularity is identical across like GPUs

    // 1) Multicast object spanning N devices. VERIFY: handleTypes = 0 (no IPC
    // export) — sparkinfer TP is single-process multi-GPU, so no fd sharing. If the
    // driver requires an explicit handle type here, set
    // .handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR.
    CUmulticastObjectProp mc_prop = {};
    mc_prop.numDevices = N;
    mc_prop.handleTypes = 0;
    mc_prop.flags = 0;

    // The allocation must be a multiple of BOTH the memory-allocation
    // granularity (for cuMemCreate) and the multicast granularity (for the
    // binding) — round up to the larger of the two.
    std::size_t mem_gran = 0, mc_gran = 0;
    if (!cu_ok(cuMemGetAllocationGranularity(&mem_gran, &ap,
                                             CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
               "cuMemGetAllocationGranularity"))
        return false;
    if (!cu_ok(cuMulticastGetGranularity(&mc_gran, &mc_prop,
                                         CU_MULTICAST_GRANULARITY_RECOMMENDED),
               "cuMulticastGetGranularity"))
        return false;
    const std::size_t gran = mem_gran > mc_gran ? mem_gran : mc_gran;
    s->alloc_bytes = ((want_bytes + gran - 1) / gran) * gran;
    mc_prop.size = s->alloc_bytes;

    if (!cu_ok(cuMulticastCreate(&s->mc_handle, &mc_prop), "cuMulticastCreate"))
        return false;
    for (int dev : s->devices) {
        if (!cu_ok(cuMulticastAddDevice(s->mc_handle, dev), "cuMulticastAddDevice"))
            return false;
    }

    s->phys.assign(N, 0);
    s->uc_va.assign(N, 0);
    s->mc_va.assign(N, 0);

    const std::size_t va_align = gran;

    for (int r = 0; r < N; ++r) {
        const int dev = s->devices[r];
        if (cudaSetDevice(dev) != cudaSuccess) return false;

        // 2) Physical allocation on device dev.
        ap.location.id = dev;
        if (!cu_ok(cuMemCreate(&s->phys[r], s->alloc_bytes, &ap, 0), "cuMemCreate"))
            return false;

        // 3) Bind this device's physical memory to the multicast at offset 0.
        // VERIFY: device must already be added to the multicast (done above).
        if (!cu_ok(cuMulticastBindMem(s->mc_handle, /*mcOffset=*/0, s->phys[r],
                                      /*memOffset=*/0, s->alloc_bytes, /*flags=*/0),
                   "cuMulticastBindMem"))
            return false;

        // 4) Unicast VA: this device's own bytes (read/write its input+output).
        if (!cu_ok(cuMemAddressReserve(&s->uc_va[r], s->alloc_bytes, va_align, 0, 0),
                   "cuMemAddressReserve(uc)"))
            return false;
        if (!cu_ok(cuMemMap(s->uc_va[r], s->alloc_bytes, 0, s->phys[r], 0),
                   "cuMemMap(uc)"))
            return false;

        // 5) Multicast VA: the switch-reduced view (multimem.* targets this).
        if (!cu_ok(cuMemAddressReserve(&s->mc_va[r], s->alloc_bytes, va_align, 0, 0),
                   "cuMemAddressReserve(mc)"))
            return false;
        if (!cu_ok(cuMemMap(s->mc_va[r], s->alloc_bytes, 0, s->mc_handle, 0),
                   "cuMemMap(mc)"))
            return false;

        // 6) Grant device dev read/write on both VAs.
        CUmemAccessDesc acc = {};
        acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        acc.location.id = dev;
        acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        if (!cu_ok(cuMemSetAccess(s->uc_va[r], s->alloc_bytes, &acc, 1),
                   "cuMemSetAccess(uc)"))
            return false;
        if (!cu_ok(cuMemSetAccess(s->mc_va[r], s->alloc_bytes, &acc, 1),
                   "cuMemSetAccess(mc)"))
            return false;
    }

    // In-kernel flag barrier (trick 1) needs peer-accessible Signal buffers.
    s->sig.assign(N, nullptr);
    for (int r = 0; r < N; ++r) {
        if (cudaSetDevice(s->devices[r]) != cudaSuccess) return false;
        for (int q = 0; q < N; ++q) {
            if (q == r) continue;
            cudaError_t e = cudaDeviceEnablePeerAccess(s->devices[q], 0);
            if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled)
                return false;
            (void)cudaGetLastError();  // clear sticky AlreadyEnabled
        }
        if (cudaMalloc(&s->sig[r], sizeof(Signal)) != cudaSuccess) return false;
        if (cudaMemset(s->sig[r], 0, sizeof(Signal)) != cudaSuccess) return false;
    }
    return true;
}

static void teardown_multicast(MultimemAllreduce::Impl* s) noexcept {
    if (!s) return;
    const int N = static_cast<int>(s->devices.size());
    for (int r = 0; r < N && r < static_cast<int>(s->uc_va.size()); ++r) {
        if (s->uc_va[r]) { cuMemUnmap(s->uc_va[r], s->alloc_bytes);
                           cuMemAddressFree(s->uc_va[r], s->alloc_bytes); }
        if (s->mc_va[r]) { cuMemUnmap(s->mc_va[r], s->alloc_bytes);
                           cuMemAddressFree(s->mc_va[r], s->alloc_bytes); }
        if (r < static_cast<int>(s->phys.size()) && s->phys[r]) {
            cuMulticastUnbind(s->mc_handle, s->devices[r], 0, s->alloc_bytes);
            cuMemRelease(s->phys[r]);
        }
        if (r < static_cast<int>(s->sig.size()) && s->sig[r]) {
            cudaSetDevice(s->devices[r]);
            cudaFree(s->sig[r]);
        }
    }
    if (s->mc_handle) cuMemRelease(s->mc_handle);
}
#endif  // SPARKINFER_TP_MULTIMEM_DRIVER_OK

MultimemAllreduce::MultimemAllreduce(const std::vector<int>& devices,
                                     std::size_t count) {
    impl_ = new Impl();
    impl_->devices = devices;
    impl_->count = count;
#if SPARKINFER_TP_MULTIMEM_DRIVER_OK
    if (count % kVecBf16 != 0) {
        std::fprintf(stderr,
                     "[multimem] count %zu not a multiple of %d; disabled\n",
                     count, kVecBf16);
        return;
    }
    if (!multimem_allreduce_supported(devices)) return;
    if (cuInit(0) != CUDA_SUCCESS) return;
    ok_ = setup_multicast(impl_);
    if (!ok_) teardown_multicast(impl_);  // partial cleanup, run peer-N instead
#endif
}

MultimemAllreduce::~MultimemAllreduce() noexcept {
#if SPARKINFER_TP_MULTIMEM_DRIVER_OK
    if (impl_ && ok_) teardown_multicast(impl_);
#endif
    delete impl_;
    impl_ = nullptr;
}

void* MultimemAllreduce::rank_buffer(int rank) const noexcept {
#if SPARKINFER_TP_MULTIMEM_DRIVER_OK
    if (!ok_ || rank < 0 || rank >= static_cast<int>(impl_->uc_va.size()))
        return nullptr;
    return reinterpret_cast<void*>(impl_->uc_va[rank]);  // input @ offset 0
#else
    (void)rank; return nullptr;
#endif
}

void* MultimemAllreduce::rank_result(int rank) const noexcept {
#if SPARKINFER_TP_MULTIMEM_DRIVER_OK
    if (!ok_ || rank < 0 || rank >= static_cast<int>(impl_->uc_va.size()))
        return nullptr;
    return reinterpret_cast<void*>(impl_->uc_va[rank] + impl_->region_bytes);
#else
    (void)rank; return nullptr;
#endif
}

#if SPARKINFER_TP_MULTIMEM_DRIVER_OK
// One multimem all-reduce kernel per rank (in-kernel barrier; no host events).
template <int N>
static void launch_mm(MultimemAllreduce::Impl* s, std::size_t n_vec, int grid,
                      int block, const std::vector<void*>& streams) {
    RankSignals<N> sg{};
    for (int r = 0; r < N; ++r) sg.sg[r] = s->sig[r];
    for (int r = 0; r < N; ++r) {
        cudaSetDevice(s->devices[r]);
        const auto* mc_in = reinterpret_cast<const uint4*>(s->mc_va[r]);
        auto* uc_out = reinterpret_cast<uint4*>(s->uc_va[r] + s->region_bytes);
        detail::multimem_allreduce_kernel<N><<<grid, block, 0,
            reinterpret_cast<cudaStream_t>(streams[r])>>>(
            mc_in, uc_out, sg, s->sig[r], r, n_vec);
    }
}
#endif

void MultimemAllreduce::allreduce_bf16(std::size_t count,
                                       const std::vector<void*>& streams) {
#if SPARKINFER_TP_MULTIMEM_DRIVER_OK
    if (!ok_) return;
    const int N = static_cast<int>(impl_->devices.size());
    const std::size_t n_vec = count / kVecBf16;
    const int block = 512;
    int grid = static_cast<int>((n_vec + block - 1) / block);
    if (grid > tpar::kMaxBlocks) grid = tpar::kMaxBlocks;  // NVLink contention cap

    // ONE kernel per rank: in-kernel flag barrier (inputs ready) → multimem
    // reduce over the multicast input → unicast output. No host events.
    switch (N) {
        case 2: launch_mm<2>(impl_, n_vec, grid, block, streams); break;
        case 4: launch_mm<4>(impl_, n_vec, grid, block, streams); break;
        case 8: launch_mm<8>(impl_, n_vec, grid, block, streams); break;
        default: break;
    }
#else
    (void)count; (void)streams;
#endif
}

#if SPARKINFER_TP_MULTIMEM_DRIVER_OK
template <int N>
static void launch_mm_f32(MultimemAllreduce::Impl* s, std::size_t n_vec, int grid,
                          int block, const std::vector<void*>& streams) {
    RankSignals<N> sg{};
    for (int r = 0; r < N; ++r) sg.sg[r] = s->sig[r];
    for (int r = 0; r < N; ++r) {
        cudaSetDevice(s->devices[r]);
        const auto* mc_in = reinterpret_cast<const float*>(s->mc_va[r]);
        auto* uc_out = reinterpret_cast<float*>(s->uc_va[r] + s->region_bytes);
        detail::multimem_allreduce_f32_kernel<N><<<grid, block, 0,
            reinterpret_cast<cudaStream_t>(streams[r])>>>(
            mc_in, uc_out, sg, s->sig[r], r, n_vec);
    }
}
#endif

void MultimemAllreduce::allreduce_f32(std::size_t count,
                                      const std::vector<void*>& streams) {
#if SPARKINFER_TP_MULTIMEM_DRIVER_OK
    if (!ok_) return;
    const int N = static_cast<int>(impl_->devices.size());
    const std::size_t n_vec = count / kVecF32;
    const int block = 512;
    int grid = static_cast<int>((n_vec + block - 1) / block);
    if (grid > tpar::kMaxBlocks) grid = tpar::kMaxBlocks;

    switch (N) {
        case 2: launch_mm_f32<2>(impl_, n_vec, grid, block, streams); break;
        case 4: launch_mm_f32<4>(impl_, n_vec, grid, block, streams); break;
        case 8: launch_mm_f32<8>(impl_, n_vec, grid, block, streams); break;
        default: break;
    }
#else
    (void)count; (void)streams;
#endif
}

}  // namespace sparkinfer::tp
