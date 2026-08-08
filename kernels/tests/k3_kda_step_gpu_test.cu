// Factor A at K3's real shape: the KDA decode step with a warp per state column.
//
// kimi_k3_numeric_test covers the same operation, but head_dim 128 with 12 heads is
// the tp=8 band and the shape the fast path is gated on, and it checks the STATE as
// well as the output — the rank-1 update is elementwise in i, so any drift there is a
// split bug rather than a reassociation.
#include "sparkinfer/kernels/kimi_k3.h"
#include "sparkinfer/kernels/kimi_k3_fast.h"
#include "k3_fast_test_util.h"


using namespace sparkinfer::kernels::k3;
using namespace k3test;
#define CU K3T_CU

// ---------------------------------------------------------------------------
static void test_kda_step_ip() {
    std::printf("KDA decode step, warp per column (D=128, H=12)\n");
    const int D = 128, H = 12;
    std::mt19937 rng(4242);

    // A random NON-ZERO state. Production starts from zeros, which is precisely what
    // makes a layout error invisible end to end — the numeric test's own note.
    auto S0   = rnd((size_t)D * D * H, rng, -0.5f, 0.5f);
    auto q    = rnd((size_t)D * H, rng);
    auto k    = rnd((size_t)D * H, rng);
    auto v    = rnd((size_t)D * H, rng);
    auto g    = rnd((size_t)D * H, rng, -1.5f, -0.05f);   // decay: exp(g) < 1
    auto beta = rnd((size_t)H, rng, 0.1f, 0.9f);

    std::vector<double> ref_out((size_t)D * H, 0.0), ref_state(S0.size());
    for (size_t i = 0; i < S0.size(); ++i) ref_state[i] = (double)S0[i];
    for (int h = 0; h < H; ++h) {
        std::vector<double> ge(D);
        for (int i = 0; i < D; ++i) ge[i] = std::exp((double)g[(size_t)h * D + i]);
        for (int j = 0; j < D; ++j) {
            double* S = &ref_state[(size_t)h * D * D + (size_t)j * D];
            double sk = 0.0;
            for (int i = 0; i < D; ++i) sk += S[i] * ge[i] * (double)k[(size_t)h * D + i];
            const double dj = (double)beta[h] * ((double)v[(size_t)h * D + j] - sk);
            double o = 0.0;
            for (int i = 0; i < D; ++i) {
                const double s2 = S[i] * ge[i] + (double)k[(size_t)h * D + i] * dj;
                S[i] = s2;
                o += s2 * (double)q[(size_t)h * D + i];
            }
            ref_out[(size_t)h * D + j] = o;
        }
    }

    float *dS = to_dev(S0), *dq = to_dev(q), *dk = to_dev(k), *dv = to_dev(v),
          *dg = to_dev(g), *db = to_dev(beta), *dout = nullptr;
    CU(cudaMalloc(&dout, (size_t)D * H * sizeof(float)));

    const bool took = k3_kda_decode_step_ip(dout, dS, dq, dk, dv, dg, db, D, H,
                                            /*beta_sigmoid=*/false, 0);
    ++g_case;
    std::printf("  %-46s %s\n", "fast path engaged at the real shape",
                took ? "OK" : "FAIL (declined)");
    if (!took) ++g_fail;
    CU(cudaDeviceSynchronize());
    CU(cudaGetLastError());

    check("out vs float64", from_dev(dout, (size_t)D * H), ref_out, 2e-5);
    check("state vs float64", from_dev(dS, S0.size()), ref_state, 2e-5);

    // And against the kernel it replaces, on the same inputs: the difference must be
    // reassociation-sized, not structural.
    float* dS2 = to_dev(S0);
    float* dout2 = nullptr;
    CU(cudaMalloc(&dout2, (size_t)D * H * sizeof(float)));
    kda_decode_step_f32(dout2, dS2, dq, dk, dv, dg, db, D, H, 0);
    CU(cudaDeviceSynchronize());
    const auto a = from_dev(dout, (size_t)D * H);
    const auto b = from_dev(dout2, (size_t)D * H);
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double)a[i] - (double)b[i];
        num += d * d; den += (double)b[i] * (double)b[i];
    }
    const double rel = std::sqrt(num / (den + 1e-30));
    ++g_case;
    const bool ok = rel < 1e-4;
    std::printf("  %-46s relL2=%.3e %s\n", "vs the tiled kernel it replaces", rel,
                ok ? "OK" : "FAIL");
    if (!ok) ++g_fail;

    // Factor G: with beta_sigmoid the kernel must produce exactly what a standalone
    // sigmoid_inplace_f32 followed by the unfused step produces — same expression,
    // so bit-identical, and checked as bytes rather than as a tolerance.
    {
        // The reference must be the DEVICE sigmoid, not a host one: both kernels use
        // the __expf intrinsic and std::exp does not agree with it to the last bit, so
        // a host-computed reference tests the intrinsic rather than the fold. (It
        // reported 366/1536 differing before this was fixed — the fold was right and
        // the test was wrong.)
        float* dbs = to_dev(beta);
        sigmoid_inplace_f32(dbs, H, 0);
        CU(cudaDeviceSynchronize());
        float* dS3 = to_dev(S0); float* dS4 = to_dev(S0);
        float *o3 = nullptr, *o4 = nullptr;
        CU(cudaMalloc(&o3, (size_t)D * H * sizeof(float)));
        CU(cudaMalloc(&o4, (size_t)D * H * sizeof(float)));
        // fused: raw beta in, sigmoid applied inside
        k3_kda_decode_step_ip(o3, dS3, dq, dk, dv, dg, db, D, H, true, 0);
        // unfused: pre-sigmoided beta in
        k3_kda_decode_step_ip(o4, dS4, dq, dk, dv, dg, dbs, D, H, false, 0);
        CU(cudaDeviceSynchronize()); CU(cudaGetLastError());
        const auto p3 = from_dev(o3, (size_t)D * H), p4 = from_dev(o4, (size_t)D * H);
        size_t nd = 0;
        for (size_t i = 0; i < p3.size(); ++i) if (p3[i] != p4[i]) ++nd;
        ++g_case;
        std::printf("  %-46s %s (%zu/%zu differ)\n", "beta sigmoid fold is bit-identical",
                    nd == 0 ? "OK" : "FAIL", nd, p3.size());
        if (nd) ++g_fail;
        CU(cudaFree(dbs)); CU(cudaFree(dS3)); CU(cudaFree(dS4));
        CU(cudaFree(o3)); CU(cudaFree(o4));
    }

    CU(cudaFree(dS)); CU(cudaFree(dS2)); CU(cudaFree(dq)); CU(cudaFree(dk));
    CU(cudaFree(dv)); CU(cudaFree(dg)); CU(cudaFree(db));
    CU(cudaFree(dout)); CU(cudaFree(dout2));
}

int main() {
    if (!have_device()) return 0;
    std::printf("K3 KDA decode step, real dims\n\n");
    test_kda_step_ip();
    return report();
}
