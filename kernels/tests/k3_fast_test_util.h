#pragma once
// Shared scaffolding for the per-factor GPU checks in kernels/tests/k3_*_gpu_test.cu.
//
// One test binary per optimisation, mirroring one source file per optimisation: a
// factor that has to be dropped after a re-measurement takes its own test with it and
// leaves the others building. This header is what stops that costing four copies of
// the same twenty lines.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace k3test {

inline int g_fail = 0;
inline int g_case = 0;

#define K3T_CU(expr)                                                              \
    do {                                                                          \
        cudaError_t e_ = (expr);                                                  \
        if (e_ != cudaSuccess) {                                                  \
            std::printf("  CUDA FAIL %s:%d %s: %s\n", __FILE__, __LINE__, #expr,  \
                        cudaGetErrorString(e_));                                  \
            std::exit(2);                                                         \
        }                                                                         \
    } while (0)

// Relative-L2 against a float64 reference. f32 kernels using __expf/rsqrtf will not
// match to 1e-7; 2e-5 is tight enough that any structural error (wrong index, wrong
// stride, a missing term) fails by orders of magnitude, and loose enough that fast
// intrinsics do not.
inline bool check(const char* what, const std::vector<float>& got,
                  const std::vector<double>& want, double tol = 2e-5) {
    ++g_case;
    double num = 0.0, den = 0.0, worst = 0.0;
    size_t wi = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = (double)got[i] - want[i];
        num += d * d;
        den += want[i] * want[i];
        const double rel = std::fabs(d) / (std::fabs(want[i]) + 1e-12);
        if (rel > worst) { worst = rel; wi = i; }
    }
    const double rl2 = std::sqrt(num / (den + 1e-30));
    const bool ok = rl2 <= tol;
    std::printf("  %-46s relL2=%.3e worst=%.3e @%zu %s\n", what, rl2, worst, wi,
                ok ? "OK" : "FAIL");
    if (!ok) ++g_fail;
    return ok;
}

inline void note(const char* what, bool ok, const char* detail = "") {
    ++g_case;
    std::printf("  %-46s %s%s%s\n", what, ok ? "OK" : "FAIL",
                detail[0] ? " — " : "", detail);
    if (!ok) ++g_fail;
}

inline double rel_l2f(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double)a[i] - (double)b[i];
        num += d * d; den += (double)b[i] * (double)b[i];
    }
    return std::sqrt(num / (den + 1e-30));
}

inline std::vector<float> rnd(size_t n, std::mt19937& rng, float lo = -1.0f,
                              float hi = 1.0f) {
    std::uniform_real_distribution<float> u(lo, hi);
    std::vector<float> v(n);
    for (auto& e : v) e = u(rng);
    return v;
}

inline float* to_dev(const std::vector<float>& h) {
    float* d = nullptr;
    K3T_CU(cudaMalloc(&d, h.size() * sizeof(float)));
    K3T_CU(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

inline std::vector<float> from_dev(const float* d, size_t n) {
    std::vector<float> h(n);
    K3T_CU(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

// Every one of these binaries is registered in ctest, and ctest runs on machines with
// no GPU (the CPU-only CI job builds the whole tree). Exiting 0 there is correct: the
// schedules are covered by the k3_*_cpu_test binaries, which need no device.
inline bool have_device() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) {
        std::printf("no CUDA device — skipping\n");
        return false;
    }
    return true;
}

inline int report() {
    std::printf("%d cases, %d failures\n", g_case, g_fail);
    return g_fail == 0 ? 0 : 1;
}

}  // namespace k3test
