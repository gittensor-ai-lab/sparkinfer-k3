// TP all-reduce backends.
//
// NOT COMPILED IN THIS REPO'S CI without -DSPARKINFER_TP=ON, and not verified on
// hardware at the time of writing — there was no GPU on the authoring machine.
// The shard math and backend selection ARE verified (runtime/tests/tp_*_cpu_test);
// this file is the CUDA/NCCL plumbing between them and a device. Validate with
// `tp_allreduce_check` on a real node before any forward path trusts it.

#include "sparkinfer/tp/collective.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <utility>

#if defined(SPARKINFER_TP_NCCL)
#include <nccl.h>
#endif

#if defined(SPARKINFER_TP_CUDA_DRIVER)
#include <cuda.h>
#endif

#if defined(SPARKINFER_TP_FAST_COLLECTIVES)
#include "sparkinfer/tp/multimem_allreduce.h"
#include "sparkinfer/tp/peer_oneshot_allreduce.h"
#endif

namespace sparkinfer {
namespace tp {

namespace {

bool cuda_ok(cudaError_t e, const char* what, std::string* err) {
    if (e == cudaSuccess) return true;
    if (err) {
        std::ostringstream os;
        os << what << ": " << cudaGetErrorString(e);
        *err = os.str();
    }
    return false;
}

// ---------------------------------------------------------------------------
// tp_size == 1
// ---------------------------------------------------------------------------

// Every operation is a no-op that succeeds. This is what lets the forward be
// written with unconditional allreduce calls and still be bit-identical to the
// pre-TP single-GPU path — there is no "if (tp)" branch to get wrong, and the
// default build exercises exactly the same code as before.
class SingleDeviceCollective final : public Collective {
public:
    Backend backend() const override { return Backend::None; }
    int size() const override { return 1; }
    bool allreduce_bf16(void*, std::size_t, int, cudaStream_t) override { return true; }
    bool barrier() override { return true; }
};

// ---------------------------------------------------------------------------
// NCCL
// ---------------------------------------------------------------------------

#if defined(SPARKINFER_TP_NCCL)

class NcclCollective final : public Collective {
public:
    static std::unique_ptr<NcclCollective> create(const std::vector<int>& devices,
                                                  std::string* error) {
        auto self = std::unique_ptr<NcclCollective>(new NcclCollective());
        self->devices_ = devices;
        const int n = static_cast<int>(devices.size());
        self->comms_.assign(static_cast<std::size_t>(n), nullptr);

        // One communicator per device in a single process. ncclCommInitAll is the
        // right call here (rather than per-rank InitRank + a bootstrap) precisely
        // because sparkinfer's runtime is single-process multi-GPU.
        ncclResult_t r = ncclCommInitAll(self->comms_.data(), n, devices.data());
        if (r != ncclSuccess) {
            if (error) {
                std::ostringstream os;
                os << "ncclCommInitAll(n=" << n << "): " << ncclGetErrorString(r);
                *error = os.str();
            }
            self->comms_.clear();
            return nullptr;
        }
        return self;
    }

    ~NcclCollective() override {
        for (ncclComm_t c : comms_) {
            if (c) ncclCommDestroy(c);
        }
    }

    Backend backend() const override { return Backend::Nccl; }
    int size() const override { return static_cast<int>(devices_.size()); }

    bool allreduce_bf16(void* buf, std::size_t count, int rank,
                        cudaStream_t stream) override {
        if (rank < 0 || rank >= size() || !buf) return false;
        if (count == 0) return true;
        // In-place: NCCL permits sendbuff == recvbuff for AllReduce.
        ncclResult_t r = ncclAllReduce(buf, buf, count, ncclBfloat16, ncclSum,
                                       comms_[static_cast<std::size_t>(rank)], stream);
        if (r != ncclSuccess) {
            std::fprintf(stderr, "[tp] ncclAllReduce rank %d: %s\n", rank,
                         ncclGetErrorString(r));
            return false;
        }
        return true;
    }

    // Stream-level: each rank's stream is synchronized, which is enough for the
    // graph-boundary use. Deliberately NOT called per-collective — the point of
    // NCCL here is that the reduction is stream-ordered and needs no host barrier.
    bool barrier() override {
        int prev = -1;
        cudaGetDevice(&prev);
        bool ok = true;
        for (std::size_t i = 0; i < devices_.size(); ++i) {
            if (cudaSetDevice(devices_[i]) != cudaSuccess) { ok = false; continue; }
            if (cudaDeviceSynchronize() != cudaSuccess) ok = false;
        }
        if (prev >= 0) cudaSetDevice(prev);
        return ok;
    }

private:
    NcclCollective() = default;
    std::vector<int> devices_;
    std::vector<ncclComm_t> comms_;
};

#endif  // SPARKINFER_TP_NCCL

// ---------------------------------------------------------------------------
// Mode B: owned-buffer, group-launched backends
// ---------------------------------------------------------------------------

#if defined(SPARKINFER_TP_FAST_COLLECTIVES)

// Both wrappers are thin on purpose. The vendored classes already own the right
// memory (multicast-bound / peer-registered) and already do the group launch with
// an in-kernel barrier; adapting them means exposing that shape through
// Collective, not reshaping them into an in-place per-rank call they cannot
// support without a copy that would erase the win.
template <class Impl>
class OwnedBufferCollective final : public Collective {
public:
    OwnedBufferCollective(std::unique_ptr<Impl> impl, Backend b, int n, std::size_t max_count)
        : impl_(std::move(impl)), backend_(b), n_(n), max_count_(max_count) {}

    Backend backend() const override { return backend_; }
    int size() const override { return n_; }
    bool owns_buffers() const override { return true; }
    std::size_t max_count() const override { return max_count_; }

    void* reduce_in(int rank) const override {
        return (rank >= 0 && rank < n_) ? impl_->rank_buffer(rank) : nullptr;
    }
    void* reduce_out(int rank) const override {
        return (rank >= 0 && rank < n_) ? impl_->rank_result(rank) : nullptr;
    }

    // Mode A is not available: reducing a caller-owned pointer would require
    // staging it through the owned buffer, i.e. two extra 14 KiB copies per
    // collective. Returning false makes a mis-wired forward fail immediately
    // instead of quietly paying that on every one of 186 collectives per token.
    bool allreduce_bf16(void*, std::size_t, int, cudaStream_t) override { return false; }

    bool allreduce_group(std::size_t count,
                         const std::vector<cudaStream_t>& streams) override {
        if (static_cast<int>(streams.size()) != n_) return false;
        if (count == 0) return true;
        if (count > max_count_) {
            std::fprintf(stderr, "[tp] allreduce_group count %zu exceeds max_count %zu\n",
                         count, max_count_);
            return false;
        }
        // The vendored API takes void* streams.
        std::vector<void*> raw(streams.size());
        for (std::size_t i = 0; i < streams.size(); ++i) raw[i] = (void*)streams[i];
        impl_->allreduce_bf16(count, raw);
        return true;
    }

    bool barrier() override {
        int prev = -1;
        cudaGetDevice(&prev);
        bool ok = true;
        for (int r = 0; r < n_; ++r) {
            if (cudaSetDevice(r) != cudaSuccess) { ok = false; continue; }
            if (cudaDeviceSynchronize() != cudaSuccess) ok = false;
        }
        if (prev >= 0) cudaSetDevice(prev);
        return ok;
    }

private:
    std::unique_ptr<Impl> impl_;
    Backend backend_;
    int n_;
    std::size_t max_count_;
};

#endif  // SPARKINFER_TP_FAST_COLLECTIVES

}  // namespace

// ---------------------------------------------------------------------------

std::unique_ptr<Collective> make_single_device_collective() {
    return std::unique_ptr<Collective>(new SingleDeviceCollective());
}

std::unique_ptr<Collective> make_nccl_collective(const std::vector<int>& devices,
                                                 std::string* error) {
#if defined(SPARKINFER_TP_NCCL)
    if (devices.size() < 2) {
        if (error) *error = "make_nccl_collective needs >= 2 devices";
        return nullptr;
    }
    auto c = NcclCollective::create(devices, error);
    if (!c) return nullptr;
    return std::unique_ptr<Collective>(c.release());
#else
    (void)devices;
    if (error) {
        *error = "NCCL support not compiled in — configure with -DSPARKINFER_TP=ON";
    }
    return nullptr;
#endif
}

std::unique_ptr<Collective> make_peer_oneshot_collective(const std::vector<int>& devices,
                                                        std::size_t max_count,
                                                        std::string* error) {
#if defined(SPARKINFER_TP_FAST_COLLECTIVES)
    if (devices.size() < 2) { if (error) *error = "peer-oneshot needs >= 2 devices"; return nullptr; }
    // The packed reduce is 128-bit vectorized: 8 bf16 per uint4.
    if (max_count == 0 || (max_count % 8) != 0) {
        if (error) *error = "peer-oneshot max_count must be a non-zero multiple of 8";
        return nullptr;
    }
    auto impl = std::make_unique<PeerOneShotAllreduce>(devices, max_count);
    if (!impl->ok()) {
        if (error) *error = "PeerOneShotAllreduce setup failed (peer access or allocation)";
        return nullptr;
    }
    return std::unique_ptr<Collective>(new OwnedBufferCollective<PeerOneShotAllreduce>(
        std::move(impl), Backend::PeerOneShot, static_cast<int>(devices.size()), max_count));
#else
    (void)devices; (void)max_count;
    if (error) *error = "fast collectives not compiled in — configure with -DSPARKINFER_TP=ON";
    return nullptr;
#endif
}

std::unique_ptr<Collective> make_multimem_collective(const std::vector<int>& devices,
                                                     std::size_t max_count,
                                                     std::string* error) {
#if defined(SPARKINFER_TP_FAST_COLLECTIVES)
    if (devices.size() < 2) { if (error) *error = "multimem needs >= 2 devices"; return nullptr; }
    if (max_count == 0 || (max_count % 8) != 0) {
        if (error) *error = "multimem max_count must be a non-zero multiple of 8";
        return nullptr;
    }
    if (!multimem_allreduce_supported(devices)) {
        if (error) *error = "CUDA multicast (NVLS) unsupported on these devices";
        return nullptr;
    }
    auto impl = std::make_unique<MultimemAllreduce>(devices, max_count);
    if (!impl->ok()) {
        if (error) *error = "MultimemAllreduce setup failed (multicast bind or mapping)";
        return nullptr;
    }
    return std::unique_ptr<Collective>(new OwnedBufferCollective<MultimemAllreduce>(
        std::move(impl), Backend::Multimem, static_cast<int>(devices.size()), max_count));
#else
    (void)devices; (void)max_count;
    if (error) *error = "fast collectives not compiled in — configure with -DSPARKINFER_TP=ON";
    return nullptr;
#endif
}

Capabilities probe_capabilities() {
    Capabilities caps;

    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) {
        // No GPU (or no driver). Returning zeros rather than failing lets the
        // whole selection path be exercised on a CPU-only box.
        return caps;
    }
    caps.device_count = n;

#if defined(SPARKINFER_TP_NCCL)
    caps.nccl_available = true;
#endif

    int min_cc = 1 << 30;
    for (int d = 0; d < n; ++d) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, d) != cudaSuccess) continue;
        const int cc = p.major * 10 + p.minor;
        if (cc < min_cc) min_cc = cc;
    }
    caps.min_compute_capability = (min_cc == (1 << 30)) ? 0 : min_cc;

    // All-pairs peer access. Checked for every ORDERED pair: cudaDeviceCanAccessPeer
    // is not symmetric in general, and a one-way link would break a peer-read
    // collective in only one direction — the worst kind of intermittent wrongness.
    caps.peer_access_all_pairs = (n > 1);
    for (int a = 0; a < n && caps.peer_access_all_pairs; ++a) {
        for (int b = 0; b < n; ++b) {
            if (a == b) continue;
            int can = 0;
            if (cudaDeviceCanAccessPeer(&can, a, b) != cudaSuccess || !can) {
                caps.peer_access_all_pairs = false;
                break;
            }
        }
    }

#if defined(SPARKINFER_TP_FAST_COLLECTIVES)
    // Ask the multimem backend itself, which now attempts a real minimal bind
    // rather than reading CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED.
    //
    // MEASURED ON HARDWARE: on an 8x H200 container without Fabric Manager access,
    // all eight devices report MULTICAST_SUPPORTED = 1 and cuMulticastBindMem then
    // fails with CUDA error 401. The attribute describes the silicon; it does not
    // describe what this process may do. Trusting it made this probe report
    // multicast as available on a node where multimem cannot run, so
    // select_backend() would choose multimem and the failure would only surface at
    // construction. A capability report that a benchmark relies on has to be right.
    {
        std::vector<int> devs(static_cast<std::size_t>(n));
        for (int d = 0; d < n; ++d) devs[static_cast<std::size_t>(d)] = d;
        caps.multicast_supported = (n > 1) && multimem_allreduce_supported(devs);
    }
#elif defined(SPARKINFER_TP_CUDA_DRIVER)
    // No multimem implementation compiled in, so the attribute is all we can read.
    // Report it, but it is the weaker claim — see the comment above.
    caps.multicast_supported = (n > 1);
    for (int d = 0; d < n && caps.multicast_supported; ++d) {
        int v = 0;
        if (cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, d) != CUDA_SUCCESS
            || v == 0) {
            caps.multicast_supported = false;
        }
    }
#endif

    return caps;
}

std::unique_ptr<Collective> make_collective(const std::vector<int>& devices,
                                            Backend requested,
                                            std::string* error,
                                            std::size_t max_count) {
    const int tp_size = static_cast<int>(devices.size());
    Capabilities caps = probe_capabilities();

    std::string reason;
    const Backend chosen = select_backend(requested, tp_size, caps, &reason);
    if (!reason.empty()) {
        std::fprintf(stderr, "[tp] %s\n", reason.c_str());
        if (error) *error = reason;
    }

    // NCCL, used both as the requested backend and as the fallback for a fast
    // backend whose runtime setup failed. Correctness path, always available
    // when TP is compiled in.
    auto try_nccl = [&](const char* why) -> std::unique_ptr<Collective> {
        std::string nccl_err;
        auto c = make_nccl_collective(devices, &nccl_err);
        if (c) {
            if (why) std::fprintf(stderr, "[tp] %s — using nccl\n", why);
            std::fprintf(stderr, "[tp] all-reduce backend: nccl, %d rank(s)\n", tp_size);
            return c;
        }
        // A no-op collective at tp>1 leaks partial sums as if they were results,
        // so keep the error set: a caller that checks it must refuse to run.
        if (error) *error = nccl_err;
        std::fprintf(stderr, "[tp] FATAL for sharded execution: %s\n", nccl_err.c_str());
        return make_single_device_collective();
    };

    switch (chosen) {
        case Backend::Nccl:
            return try_nccl(nullptr);

        case Backend::PeerOneShot: {
            std::string e;
            auto c = make_peer_oneshot_collective(devices, max_count, &e);
            if (c) {
                std::fprintf(stderr, "[tp] all-reduce backend: peer-oneshot, %d rank(s), "
                                     "max_count %zu (owned buffers)\n", tp_size, max_count);
                return c;
            }
            return try_nccl(e.c_str());
        }

        case Backend::Multimem: {
            std::string e;
            auto c = make_multimem_collective(devices, max_count, &e);
            if (c) {
                std::fprintf(stderr, "[tp] all-reduce backend: multimem, %d rank(s), "
                                     "max_count %zu (owned buffers, NVSwitch reduce)\n",
                             tp_size, max_count);
                return c;
            }
            return try_nccl(e.c_str());
        }

        case Backend::None:
        default:
            return make_single_device_collective();
    }
}

}  // namespace tp
}  // namespace sparkinfer
