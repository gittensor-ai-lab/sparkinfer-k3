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
#include <sstream>

#if defined(SPARKINFER_TP_NCCL)
#include <nccl.h>
#endif

#if defined(SPARKINFER_TP_CUDA_DRIVER)
#include <cuda.h>
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

#if defined(SPARKINFER_TP_CUDA_DRIVER)
    // Multicast (NVLS) is a driver-API attribute; needed only by the multimem
    // backend, which is not vendored here. Probed anyway so the capability
    // report is honest about what the node could support.
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
                                            std::string* error) {
    const int tp_size = static_cast<int>(devices.size());
    Capabilities caps = probe_capabilities();

    std::string reason;
    const Backend chosen = select_backend(requested, tp_size, caps, &reason);
    if (!reason.empty()) {
        std::fprintf(stderr, "[tp] %s\n", reason.c_str());
        if (error) *error = reason;
    }

    switch (chosen) {
        case Backend::Nccl: {
            std::string nccl_err;
            auto c = make_nccl_collective(devices, &nccl_err);
            if (c) {
                std::fprintf(stderr, "[tp] all-reduce backend: nccl, %d rank(s)\n", tp_size);
                return c;
            }
            // Selection said NCCL was available but init failed. Fall back to the
            // no-op so the process does not crash, but keep the error set: a
            // caller that checks it must refuse to run a sharded model, because a
            // no-op collective at tp>1 leaks partial sums as if they were results.
            if (error) *error = nccl_err;
            std::fprintf(stderr, "[tp] FATAL for sharded execution: %s\n", nccl_err.c_str());
            return make_single_device_collective();
        }
        case Backend::PeerOneShot:
        case Backend::Multimem:
            // Unreachable: select_backend() downgrades unvendored backends to
            // Nccl. Handled explicitly so adding a backend cannot silently fall
            // through to the no-op.
            if (error) *error = std::string(backend_name(chosen)) + " has no implementation";
            return make_single_device_collective();
        case Backend::None:
        default:
            return make_single_device_collective();
    }
}

}  // namespace tp
}  // namespace sparkinfer
