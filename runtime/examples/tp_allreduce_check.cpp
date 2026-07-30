// Validate the TP all-reduce on real hardware, before any forward path trusts it.
//
//   tp_allreduce_check [tp_size] [iters]
//
// Prints the probed capabilities, the selected backend, a CORRECTNESS verdict,
// and per-call latency at the payload sizes that matter. Exit 0 only if the
// reduction is exactly right on every rank.
//
// WHY THIS EXISTS. Kimi K3 at tp_size 8 issues 186 all-reduces per decoded token
// (93 layers x 2), each 14 KiB of bf16. That is latency-bound, so this reports
// per-call microseconds rather than GB/s — a bandwidth number would look healthy
// while decode was starved by launch overhead. And it checks correctness FIRST,
// because a fast wrong reduction is the failure mode that survives review: the
// model still emits fluent text, just slightly wrong text.
//
// Every input is chosen so a wrong answer cannot look plausible. Rank r writes
// (r + 1) into every slot, so the expected sum is tp_size * (tp_size + 1) / 2 —
// a value no partial reduction can produce by accident. A missing rank, a
// double-counted rank, or a rank reducing over the wrong buffer all yield a
// different total.

#include "sparkinfer/tp/backend_select.h"
#include "sparkinfer/tp/collective.h"
#include "sparkinfer/tp/shard.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace sparkinfer;

namespace {

// bf16 helpers kept local: the values used here (small integers) are exactly
// representable, so a round-trip must be exact and any drift is a real bug.
uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    // round-to-nearest-even
    const uint32_t rounded = bits + 0x7FFF + ((bits >> 16) & 1);
    return static_cast<uint16_t>(rounded >> 16);
}

float bf16_to_f32(uint16_t h) {
    const uint32_t bits = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

bool check(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "!! %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    const int want_tp = (argc > 1) ? std::atoi(argv[1]) : 0;   // 0 = all visible
    const int iters = (argc > 2) ? std::atoi(argv[2]) : 200;

    tp::Capabilities caps = tp::probe_capabilities();
    std::printf("=== probed capabilities ===\n");
    std::printf("  devices              : %d\n", caps.device_count);
    std::printf("  min compute capability: sm_%d\n", caps.min_compute_capability);
    std::printf("  nccl available       : %s\n", caps.nccl_available ? "yes" : "no");
    std::printf("  peer access all pairs: %s\n", caps.peer_access_all_pairs ? "yes" : "no");
    std::printf("  multicast (NVLS)     : %s\n", caps.multicast_supported ? "yes" : "no");

    if (caps.device_count == 0) {
        std::printf("\nno CUDA device — nothing to validate (this is not a failure)\n");
        return 0;
    }

    const int tp_size = (want_tp > 0) ? want_tp : caps.device_count;
    std::vector<int> devices;
    for (int i = 0; i < tp_size; ++i) devices.push_back(i);

    std::string requested_env;
    if (const char* e = std::getenv("SPARKINFER_TP_BACKEND")) requested_env = e;
    std::string why;
    const tp::Backend requested = tp::backend_from_string(requested_env, &why);
    if (!why.empty()) std::printf("\n[env] %s\n", why.c_str());

    std::printf("\n=== backend selection (tp_size=%d) ===\n", tp_size);
    std::string sel_reason;
    const tp::Backend chosen = tp::select_backend(requested, tp_size, caps, &sel_reason);
    std::printf("  requested: %s\n", tp::backend_name(requested));
    std::printf("  selected : %s\n", tp::backend_name(chosen));
    if (!sel_reason.empty()) std::printf("  reason   : %s\n", sel_reason.c_str());

    if (tp_size > 1 && chosen == tp::Backend::None) {
        std::printf("\nFAIL: tp_size %d selected no collective — a sharded run here would "
                    "emit partial sums as results. Refusing.\n", tp_size);
        return 1;
    }

    std::string err;
    auto coll = tp::make_collective(devices, requested, &err);
    if (!coll) {
        std::printf("\nFAIL: could not build a collective: %s\n", err.c_str());
        return 1;
    }
    if (tp_size > 1 && coll->backend() == tp::Backend::None) {
        std::printf("\nFAIL: fell back to the no-op collective at tp_size %d: %s\n",
                    tp_size, err.c_str());
        return 1;
    }

    // Per-rank streams + buffers, sized for the largest payload we test.
    // K3 decode is 7168 elems (14 KiB); prefill at ubatch 512 is 512x that.
    const std::vector<std::size_t> payloads = {7168, 7168 * 8, 7168 * 64, 7168 * 512};
    const std::size_t max_count = payloads.back();

    std::vector<cudaStream_t> streams(tp_size);
    std::vector<void*> bufs(tp_size, nullptr);
    for (int r = 0; r < tp_size; ++r) {
        if (!check(cudaSetDevice(devices[r]), "cudaSetDevice")) return 1;
        if (!check(cudaStreamCreate(&streams[r]), "cudaStreamCreate")) return 1;
        if (!check(cudaMalloc(&bufs[r], max_count * sizeof(uint16_t)), "cudaMalloc")) return 1;
    }

    // ---- correctness, before any timing ----
    // Rank r fills with (r+1); the only correct sum is tp*(tp+1)/2. No partial
    // or double-counted reduction can land on that value by accident.
    const float expect = static_cast<float>(tp_size) * (tp_size + 1) / 2.0f;
    std::printf("\n=== correctness (rank r writes r+1; expected sum %.1f) ===\n", expect);
    bool all_correct = true;

    for (std::size_t count : payloads) {
        std::vector<uint16_t> host(count);
        for (int r = 0; r < tp_size; ++r) {
            const uint16_t v = f32_to_bf16(static_cast<float>(r + 1));
            for (std::size_t i = 0; i < count; ++i) host[i] = v;
            if (!check(cudaSetDevice(devices[r]), "set")) return 1;
            if (!check(cudaMemcpyAsync(bufs[r], host.data(), count * sizeof(uint16_t),
                                       cudaMemcpyHostToDevice, streams[r]), "H2D")) return 1;
        }
        for (int r = 0; r < tp_size; ++r) {
            if (!check(cudaSetDevice(devices[r]), "set")) return 1;
            if (!coll->allreduce_bf16(bufs[r], count, r, streams[r])) {
                std::printf("  FAIL: allreduce returned false (rank %d, count %zu)\n", r, count);
                return 1;
            }
        }
        for (int r = 0; r < tp_size; ++r) {
            if (!check(cudaSetDevice(devices[r]), "set")) return 1;
            if (!check(cudaStreamSynchronize(streams[r]), "sync")) return 1;
        }

        // EVERY rank is checked, not just rank 0. A collective that reduces
        // correctly on one rank and not another is a real and common bug, and
        // checking only rank 0 is exactly how it ships.
        bool ok = true;
        std::size_t bad_rank = 0, bad_idx = 0;
        float bad_val = 0;
        for (int r = 0; r < tp_size && ok; ++r) {
            if (!check(cudaSetDevice(devices[r]), "set")) return 1;
            if (!check(cudaMemcpy(host.data(), bufs[r], count * sizeof(uint16_t),
                                  cudaMemcpyDeviceToHost), "D2H")) return 1;
            for (std::size_t i = 0; i < count; ++i) {
                const float got = bf16_to_f32(host[i]);
                if (std::fabs(got - expect) > 1e-3f) {
                    ok = false; bad_rank = r; bad_idx = i; bad_val = got;
                    break;
                }
            }
        }
        std::printf("  count %8zu (%7.1f KiB)  %s", count,
                    count * 2.0 / 1024.0, ok ? "OK\n" : "");
        if (!ok) {
            std::printf("FAIL — rank %zu elem %zu = %.4f, want %.1f\n",
                        bad_rank, bad_idx, bad_val, expect);
            all_correct = false;
        }
    }

    if (!all_correct) {
        std::printf("\nFAIL: the collective does not reduce correctly. Do NOT wire this "
                    "into a forward path.\n");
        return 1;
    }

    // ---- latency ----
    // Per-CALL microseconds, because 186 collectives/token means launch overhead
    // dominates and a GB/s figure would hide it.
    std::printf("\n=== latency (%d iters, median-free mean of the steady state) ===\n", iters);
    std::printf("  %10s  %10s  %12s  %14s\n", "elems", "KiB", "us/call", "us/token(186)");
    for (std::size_t count : payloads) {
        for (int warm = 0; warm < 20; ++warm) {
            for (int r = 0; r < tp_size; ++r) {
                cudaSetDevice(devices[r]);
                coll->allreduce_bf16(bufs[r], count, r, streams[r]);
            }
        }
        for (int r = 0; r < tp_size; ++r) {
            cudaSetDevice(devices[r]);
            cudaStreamSynchronize(streams[r]);
        }

        const auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; ++it) {
            for (int r = 0; r < tp_size; ++r) {
                cudaSetDevice(devices[r]);
                coll->allreduce_bf16(bufs[r], count, r, streams[r]);
            }
        }
        for (int r = 0; r < tp_size; ++r) {
            cudaSetDevice(devices[r]);
            cudaStreamSynchronize(streams[r]);
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
        std::printf("  %10zu  %10.1f  %12.2f  %14.1f\n", count, count * 2.0 / 1024.0, us,
                    us * tp::reduce_count_per_token(93));
    }

    for (int r = 0; r < tp_size; ++r) {
        cudaSetDevice(devices[r]);
        cudaFree(bufs[r]);
        cudaStreamDestroy(streams[r]);
    }

    std::printf("\nPASS: %s reduces correctly across %d rank(s).\n",
                tp::backend_name(coll->backend()), tp_size);
    std::printf("The us/token column is the floor TP adds to K3 decode before any "
                "compute — compare it against the llama.cpp baseline's total.\n");
    return 0;
}
