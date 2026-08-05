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

    // Owned-buffer backends size their buffers at construction, so the largest
    // payload we intend to test has to be declared up front.
    const std::vector<std::size_t> payloads = {7168, 7168 * 8, 7168 * 64, 7168 * 512};
    const std::size_t max_count = payloads.back();

    std::string err;
    auto coll = tp::make_collective(devices, requested, &err, max_count);
    if (!coll) {
        std::printf("\nFAIL: could not build a collective: %s\n", err.c_str());
        return 1;
    }
    if (tp_size > 1 && coll->backend() == tp::Backend::None) {
        std::printf("\nFAIL: fell back to the no-op collective at tp_size %d: %s\n",
                    tp_size, err.c_str());
        return 1;
    }

    const bool owned = coll->owns_buffers();
    std::printf("  buffers  : %s\n", owned ? "collective-owned (group launch)"
                                            : "caller-owned (in-place, per rank)");

    std::vector<cudaStream_t> streams(tp_size);
    std::vector<void*> bufs(tp_size, nullptr);
    for (int r = 0; r < tp_size; ++r) {
        if (!check(cudaSetDevice(devices[r]), "cudaSetDevice")) return 1;
        if (!check(cudaStreamCreate(&streams[r]), "cudaStreamCreate")) return 1;
        // Sized for the WIDER dtype so the same buffers serve both passes: f32 is
        // 4 B/elem against bf16's 2. Allocating for bf16 and then running the f32
        // pass would overrun by exactly 2x — silently, since cudaMemcpy past a
        // suballocation usually lands in the same page.
        if (!check(cudaMalloc(&bufs[r], max_count * sizeof(float)), "cudaMalloc")) return 1;
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
            void* dst = owned ? coll->reduce_in(r) : bufs[r];
            if (!dst) { std::printf("  FAIL: no input buffer for rank %d\n", r); return 1; }
            if (!check(cudaMemcpyAsync(dst, host.data(), count * sizeof(uint16_t),
                                       cudaMemcpyHostToDevice, streams[r]), "H2D")) return 1;
        }
        // ONE group launch for every backend. For NCCL this is ncclGroupStart/End
        // (a single thread looping ncclAllReduce per rank deadlocks); for the fast
        // backends it is one kernel per rank with the barrier inside the kernel.
        if (!coll->allreduce_bf16_group(bufs, count, streams)) {
            std::printf("  FAIL: allreduce_bf16_group returned false (count %zu)\n", count);
            return 1;
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
            const void* src = owned ? coll->reduce_out(r) : bufs[r];
            if (!src) { std::printf("  FAIL: no output buffer for rank %d\n", r); return 1; }
            if (!check(cudaMemcpy(host.data(), src, count * sizeof(uint16_t),
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
        auto issue = [&]() { coll->allreduce_bf16_group(bufs, count, streams); };
        for (int warm = 0; warm < 20; ++warm) issue();
        for (int r = 0; r < tp_size; ++r) {
            cudaSetDevice(devices[r]);
            cudaStreamSynchronize(streams[r]);
        }

        const auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; ++it) issue();
        for (int r = 0; r < tp_size; ++r) {
            cudaSetDevice(devices[r]);
            cudaStreamSynchronize(streams[r]);
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
        std::printf("  %10zu  %10.1f  %12.2f  %14.1f\n", count, count * 2.0 / 1024.0, us,
                    us * tp::reduce_count_per_token(93));
    }

    // ---- f32: Kimi K3's dtype ----
    //
    // K3 runs an f32 residual stream by design, so its 186 collectives/token go
    // through allreduce_f32_group, NOT the bf16 path validated above. A backend that
    // reduces bf16 correctly tells you nothing about its f32 path — different NCCL
    // datatype, different element size, and for the fast backends a different kernel
    // entirely. So it is validated separately, with the same
    // correctness-before-timing discipline.
    //
    // The sum here is EXACT, not approximate: rank r writes the integer (r+1), and
    // integers through 36 are exactly representable in f32, so the comparison is ==
    // rather than a tolerance. Any drift at all is a real bug, not rounding.
    std::printf("\n=== f32 (Kimi K3's residual dtype) ===\n");
    if (!coll->supports_f32()) {
        std::printf("  SKIPPED: %s has no f32 reduce (bf16-only kernels).\n",
                    tp::backend_name(coll->backend()));
        std::printf("  K3 TP must select a backend that does — make_collective(need_f32=true)\n"
                    "  downgrades to nccl for exactly this reason.\n");
    } else {
        bool f32_ok = true;
        for (std::size_t count : payloads) {
            std::vector<float> host(count);
            for (int r = 0; r < tp_size; ++r) {
                for (std::size_t i = 0; i < count; ++i) host[i] = static_cast<float>(r + 1);
                if (!check(cudaSetDevice(devices[r]), "set")) return 1;
                void* dst = owned ? coll->reduce_in(r) : bufs[r];
                if (!dst) { std::printf("  FAIL: no input buffer for rank %d\n", r); return 1; }
                if (!check(cudaMemcpyAsync(dst, host.data(), count * sizeof(float),
                                           cudaMemcpyHostToDevice, streams[r]), "H2D")) return 1;
            }
            const bool launched = owned
                ? coll->allreduce_f32_owned(count, streams)
                : coll->allreduce_f32_group(bufs, count, streams);
            if (!launched) {
                std::printf("  FAIL: f32 all-reduce returned false (count %zu)\n", count);
                return 1;
            }
            for (int r = 0; r < tp_size; ++r) {
                if (!check(cudaSetDevice(devices[r]), "set")) return 1;
                if (!check(cudaStreamSynchronize(streams[r]), "sync")) return 1;
            }
            bool ok = true;
            int bad_rank = 0; std::size_t bad_idx = 0; float bad_val = 0;
            for (int r = 0; r < tp_size && ok; ++r) {
                if (!check(cudaSetDevice(devices[r]), "set")) return 1;
                const void* src = owned ? coll->reduce_out(r) : bufs[r];
                if (!check(cudaMemcpy(host.data(), src, count * sizeof(float),
                                      cudaMemcpyDeviceToHost), "D2H")) return 1;
                for (std::size_t i = 0; i < count; ++i) {
                    if (host[i] != expect) {
                        ok = false; bad_rank = r; bad_idx = i; bad_val = host[i];
                        break;
                    }
                }
            }
            std::printf("  count %8zu (%7.1f KiB)  %s", count, count * 4.0 / 1024.0,
                        ok ? "OK (exact)\n" : "");
            if (!ok) {
                std::printf("FAIL — rank %d elem %zu = %.6f, want %.1f\n",
                            bad_rank, bad_idx, bad_val, expect);
                f32_ok = false;
            }
        }
        if (!f32_ok) {
            std::printf("\nFAIL: the f32 reduce is wrong. K3 TP must not run on this.\n");
            return 1;
        }

        // K3's two real payloads, named: the attention collective is at hidden width
        // and the MoE one at expert_latent — they are NOT the same size, which is why
        // both are timed rather than assuming one number covers the token.
        std::printf("\n=== f32 latency (%d iters) — K3's actual per-layer payloads ===\n", iters);
        std::printf("  %10s  %10s  %12s  %s\n", "elems", "KiB", "us/call", "what");
        struct { std::size_t n; const char* what; } k3[] = {
            {3584, "MoE reduce  @ expert_latent (1/layer, 92 layers)"},
            {7168, "attn reduce @ hidden        (1/layer, 93 layers)"},
        };
        double us_attn = 0, us_moe = 0;
        for (const auto& p : k3) {
            auto issue = [&]() {
                if (owned) coll->allreduce_f32_owned(p.n, streams);
                else coll->allreduce_f32_group(bufs, p.n, streams);
            };
            for (int warm = 0; warm < 20; ++warm) issue();
            for (int r = 0; r < tp_size; ++r) { cudaSetDevice(devices[r]); cudaStreamSynchronize(streams[r]); }
            const auto t0 = std::chrono::steady_clock::now();
            for (int it = 0; it < iters; ++it) issue();
            for (int r = 0; r < tp_size; ++r) { cudaSetDevice(devices[r]); cudaStreamSynchronize(streams[r]); }
            const auto t1 = std::chrono::steady_clock::now();
            const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
            std::printf("  %10zu  %10.1f  %12.2f  %s\n", p.n, p.n * 4.0 / 1024.0, us, p.what);
            if (p.n == 7168) us_attn = us; else us_moe = us;
        }
        // 93 attention reduces + 92 MoE reduces (layer 0 is the leading dense layer,
        // whose FFN reduce is at hidden width — counted with the attention ones).
        const double per_token = us_attn * 93 + us_moe * 92 + us_attn * 1;
        std::printf("  => %.0f us/token of pure collective at tp=%d, before any compute\n",
                    per_token, tp_size);
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
