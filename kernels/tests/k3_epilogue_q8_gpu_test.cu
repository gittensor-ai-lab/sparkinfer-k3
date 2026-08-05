// Factor — producer + Q8_0 in one launch, at K3's real dims.
// Compares the fused Q8 bytes AND the optional float mirror against the
// standalone producer followed by k3_quantize_q8_0 — bitwise, not by tolerance.
#include "sparkinfer/kernels/kimi_k3.h"
#include "sparkinfer/kernels/kimi_k3_fast.h"
#include "k3_fast_test_util.h"

#include <cstring>

using namespace sparkinfer::kernels::k3;
using namespace k3test;
#define CU K3T_CU

static bool bytes_eq(const void* a, const void* b, size_t n) {
    return std::memcmp(a, b, n) == 0;
}

static void test_situ_q8() {
    const int n = 768;   // 2-D MoE shexp band at tp=8
    std::printf("situ + Q8_0 (shexp-band width)\n");
    std::mt19937 rng(7);
    auto gate = rnd(n, rng);
    auto up   = rnd(n, rng, -2.f, 2.f);
    const float beta = 4.0f, lb = 25.0f;

    float* dg = to_dev(gate);
    float* du = to_dev(up);
    float* dsitu = nullptr;
    float* dsitu2 = nullptr;
    void* q8_fused = nullptr;
    void* q8_ref = nullptr;
    const size_t qb = k3_q8_0_bytes(n);
    CU(cudaMalloc(&dsitu, (size_t)n * 4));
    CU(cudaMalloc(&dsitu2, (size_t)n * 4));
    CU(cudaMalloc(&q8_fused, qb));
    CU(cudaMalloc(&q8_ref, qb));

    const bool took = k3_situ_q8(q8_fused, dsitu, dg, du, n, beta, lb, 0);
    ++g_case;
    std::printf("  %-46s %s\n", "fast path engaged", took ? "OK" : "FAIL (declined)");
    if (!took) ++g_fail;

    situ_f32(dsitu2, dg, du, n, beta, lb, 0);
    k3_quantize_q8_0(q8_ref, dsitu2, n / 32, 0);
    CU(cudaDeviceSynchronize());
    CU(cudaGetLastError());

    auto f_fused = from_dev(dsitu, n);
    auto f_ref   = from_dev(dsitu2, n);
    ++g_case;
    const bool f_ok = bytes_eq(f_fused.data(), f_ref.data(), (size_t)n * 4);
    std::printf("  %-46s %s\n", "float situ mirror bit-identical", f_ok ? "OK" : "FAIL");
    if (!f_ok) ++g_fail;

    std::vector<char> hq(qb), href(qb);
    CU(cudaMemcpy(hq.data(), q8_fused, qb, cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(href.data(), q8_ref, qb, cudaMemcpyDeviceToHost));
    ++g_case;
    const bool q_ok = bytes_eq(hq.data(), href.data(), qb);
    std::printf("  %-46s %s\n", "Q8_0 bytes bit-identical", q_ok ? "OK" : "FAIL");
    if (!q_ok) ++g_fail;

    CU(cudaFree(dg)); CU(cudaFree(du));
    CU(cudaFree(dsitu)); CU(cudaFree(dsitu2));
    CU(cudaFree(q8_fused)); CU(cudaFree(q8_ref));
}

static void test_mla_gate_q8() {
    const int n = 1536;   // 12 heads * 128 at tp=8
    std::printf("mla_gate + Q8_0\n");
    std::mt19937 rng(11);
    auto attn = rnd(n, rng);
    auto gate = rnd(n, rng);

    float* da = to_dev(attn);
    float* dg = to_dev(gate);
    float* dout = nullptr;
    float* dout2 = nullptr;
    void* q8_fused = nullptr;
    void* q8_ref = nullptr;
    const size_t qb = k3_q8_0_bytes(n);
    CU(cudaMalloc(&dout, (size_t)n * 4));
    CU(cudaMalloc(&dout2, (size_t)n * 4));
    CU(cudaMalloc(&q8_fused, qb));
    CU(cudaMalloc(&q8_ref, qb));
    CU(cudaMemcpy(dout2, da, (size_t)n * 4, cudaMemcpyDeviceToDevice));

    const bool took = k3_mla_gate_q8(q8_fused, dout, da, dg, n, 0);
    ++g_case;
    std::printf("  %-46s %s\n", "fast path engaged", took ? "OK" : "FAIL (declined)");
    if (!took) ++g_fail;

    mla_gate_out_f32(dout2, dout2, dg, n, 0);
    k3_quantize_q8_0(q8_ref, dout2, n / 32, 0);
    CU(cudaDeviceSynchronize());

    auto f_fused = from_dev(dout, n);
    auto f_ref   = from_dev(dout2, n);
    ++g_case;
    const bool f_ok = bytes_eq(f_fused.data(), f_ref.data(), (size_t)n * 4);
    std::printf("  %-46s %s\n", "float gate mirror bit-identical", f_ok ? "OK" : "FAIL");
    if (!f_ok) ++g_fail;

    std::vector<char> hq(qb), href(qb);
    CU(cudaMemcpy(hq.data(), q8_fused, qb, cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(href.data(), q8_ref, qb, cudaMemcpyDeviceToHost));
    ++g_case;
    const bool q_ok = bytes_eq(hq.data(), href.data(), qb);
    std::printf("  %-46s %s\n", "Q8_0 bytes bit-identical", q_ok ? "OK" : "FAIL");
    if (!q_ok) ++g_fail;

    CU(cudaFree(da)); CU(cudaFree(dg));
    CU(cudaFree(dout)); CU(cudaFree(dout2));
    CU(cudaFree(q8_fused)); CU(cudaFree(q8_ref));
}

static void test_kda_gate_q8() {
    const int head_dim = 128, n_head = 12;
    const int n = head_dim * n_head;
    std::printf("kda_gate + Q8_0 (tp=8 head band)\n");
    std::mt19937 rng(13);
    auto o  = rnd(n, rng);
    auto g2 = rnd(n, rng);
    auto w  = rnd(head_dim, rng, 0.5f, 1.5f);
    const float eps = 1e-5f;

    float* do_ = to_dev(o);
    float* dg = to_dev(g2);
    float* dw = to_dev(w);
    float* dout = nullptr;
    float* dout2 = nullptr;
    void* q8_fused = nullptr;
    void* q8_ref = nullptr;
    const size_t qb = k3_q8_0_bytes(n);
    CU(cudaMalloc(&dout, (size_t)n * 4));
    CU(cudaMalloc(&dout2, (size_t)n * 4));
    CU(cudaMalloc(&q8_fused, qb));
    CU(cudaMalloc(&q8_ref, qb));

    const bool took = k3_kda_gate_q8(q8_fused, dout, do_, dw, dg, head_dim, n_head,
                                     eps, 0);
    ++g_case;
    std::printf("  %-46s %s\n", "fast path engaged", took ? "OK" : "FAIL (declined)");
    if (!took) ++g_fail;

    kda_gate_out_f32(dout2, do_, dw, dg, head_dim, n_head, eps, 0);
    k3_quantize_q8_0(q8_ref, dout2, n / 32, 0);
    CU(cudaDeviceSynchronize());

    auto f_fused = from_dev(dout, n);
    auto f_ref   = from_dev(dout2, n);
    ++g_case;
    const bool f_ok = bytes_eq(f_fused.data(), f_ref.data(), (size_t)n * 4);
    std::printf("  %-46s %s\n", "float gate_out bit-identical", f_ok ? "OK" : "FAIL");
    if (!f_ok) ++g_fail;

    std::vector<char> hq(qb), href(qb);
    CU(cudaMemcpy(hq.data(), q8_fused, qb, cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(href.data(), q8_ref, qb, cudaMemcpyDeviceToHost));
    ++g_case;
    const bool q_ok = bytes_eq(hq.data(), href.data(), qb);
    std::printf("  %-46s %s\n", "Q8_0 bytes bit-identical", q_ok ? "OK" : "FAIL");
    if (!q_ok) ++g_fail;

    // Decline on a non-K3 head width rather than mishandle it.
    const bool declined = !k3_kda_gate_q8(q8_fused, dout, do_, dw, dg, 64, n_head,
                                          eps, 0);
    ++g_case;
    std::printf("  %-46s %s\n", "head_dim=64 declines", declined ? "OK" : "FAIL");
    if (!declined) ++g_fail;

    CU(cudaFree(do_)); CU(cudaFree(dg)); CU(cudaFree(dw));
    CU(cudaFree(dout)); CU(cudaFree(dout2));
    CU(cudaFree(q8_fused)); CU(cudaFree(q8_ref));
}

int main() {
    if (!have_device()) return 0;
    std::printf("K3 epilogue→Q8_0 folds, real dims\n\n");
    test_situ_q8();
    std::printf("\n");
    test_mla_gate_q8();
    std::printf("\n");
    test_kda_gate_q8();
    return report();
}
