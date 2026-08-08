// Factor C at K3's real shape: MLA query absorption, one warp per (head, row),
// reading q_proj_out in place instead of through two strided de-interleave copies.
#include "sparkinfer/kernels/kimi_k3.h"
#include "sparkinfer/kernels/kimi_k3_fast.h"
#include "k3_fast_test_util.h"


using namespace sparkinfer::kernels::k3;
using namespace k3test;
#define CU K3T_CU

// ---------------------------------------------------------------------------
// Graded against mla_absorb_q_f32 fed by an EXPLICIT de-interleave, which is what the
// forward does when this declines. So the case checks both halves at once: the
// reduction tree, and that reading q_proj in place picks the same 128 + 64 values the
// two cudaMemcpy2DAsync copies would have produced.
static void test_mla_absorb_strided() {
    const int qk_nope = 128, kv_lora = 512, rope = 64, n_head = 12;
    const int stride = qk_nope + rope;          // key_length_mla = 192
    std::printf("MLA absorb q: warp per (head, row), q read in place\n");

    std::mt19937 rng(31337);
    auto q_proj = rnd((size_t)n_head * stride, rng);
    auto wk_b   = rnd((size_t)qk_nope * kv_lora * n_head, rng, -0.3f, 0.3f);

    // float64 reference straight from the layout contract: wk_b is
    // [qk_nope, kv_lora, n_head] with qk_nope fastest, so column r of head h starts at
    // (h * kv_lora + r) * qk_nope.
    std::vector<double> ref((size_t)n_head * (kv_lora + rope), 0.0);
    for (int h = 0; h < n_head; ++h) {
        for (int r = 0; r < kv_lora; ++r) {
            double a = 0.0;
            for (int d = 0; d < qk_nope; ++d)
                a += (double)wk_b[((size_t)h * kv_lora + r) * qk_nope + d] *
                     (double)q_proj[(size_t)h * stride + d];
            ref[(size_t)h * (kv_lora + rope) + r] = a;
        }
        for (int d = 0; d < rope; ++d)
            ref[(size_t)h * (kv_lora + rope) + kv_lora + d] =
                (double)q_proj[(size_t)h * stride + qk_nope + d];
    }

    float* dq = to_dev(q_proj);
    float* dw = to_dev(wk_b);
    float* dout = nullptr;
    CU(cudaMalloc(&dout, (size_t)n_head * (kv_lora + rope) * sizeof(float)));
    CU(cudaMemset(dout, 0, (size_t)n_head * (kv_lora + rope) * sizeof(float)));

    const bool took = k3_mla_absorb_q_strided(dout, dq, dw, qk_nope, kv_lora, rope,
                                              stride, n_head, 0);
    ++g_case;
    std::printf("  %-46s %s\n", "fast path engaged at the real shape",
                took ? "OK" : "FAIL (declined)");
    if (!took) ++g_fail;
    CU(cudaDeviceSynchronize());
    CU(cudaGetLastError());
    check("absorbed q vs float64", from_dev(dout, (size_t)n_head * (kv_lora + rope)),
          ref, 2e-5);

    // Against the kernel it replaces, fed by the de-interleave the forward would do.
    std::vector<float> q_nope((size_t)n_head * qk_nope), q_pe((size_t)n_head * rope);
    for (int h = 0; h < n_head; ++h) {
        for (int d = 0; d < qk_nope; ++d)
            q_nope[(size_t)h * qk_nope + d] = q_proj[(size_t)h * stride + d];
        for (int d = 0; d < rope; ++d)
            q_pe[(size_t)h * rope + d] = q_proj[(size_t)h * stride + qk_nope + d];
    }
    float* dqn = to_dev(q_nope);
    float* dqp = to_dev(q_pe);
    float* dout2 = nullptr;
    CU(cudaMalloc(&dout2, (size_t)n_head * (kv_lora + rope) * sizeof(float)));
    mla_absorb_q_f32(dout2, dqn, dqp, dw, qk_nope, kv_lora, rope, n_head, 0);
    CU(cudaDeviceSynchronize());
    const auto a = from_dev(dout, (size_t)n_head * (kv_lora + rope));
    const auto b = from_dev(dout2, (size_t)n_head * (kv_lora + rope));
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double)a[i] - (double)b[i];
        num += d * d; den += (double)b[i] * (double)b[i];
    }
    const double rel = std::sqrt(num / (den + 1e-30));
    ++g_case;
    const bool ok = rel < 1e-5;
    std::printf("  %-46s relL2=%.3e %s\n", "vs de-interleave + mla_absorb_q_f32", rel,
                ok ? "OK" : "FAIL");
    if (!ok) ++g_fail;

    CU(cudaFree(dq)); CU(cudaFree(dw)); CU(cudaFree(dout));
    CU(cudaFree(dqn)); CU(cudaFree(dqp)); CU(cudaFree(dout2));
}

int main() {
    if (!have_device()) return 0;
    std::printf("K3 MLA absorb q, real dims\n\n");
    test_mla_absorb_strided();
    return report();
}
