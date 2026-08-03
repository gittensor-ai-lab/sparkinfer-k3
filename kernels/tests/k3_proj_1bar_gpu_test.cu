// Factor D: the one-barrier multi-row Q8_0 GEMV, checked as BYTES.
//
// The claim is bit-identical, so this is memcmp against k3_proj_ggml_f32 rather than a
// tolerance — a tolerance would pass for a kernel that reassociated the warp fold,
// which is the one thing the epilogue must not do.
#include "sparkinfer/kernels/kimi_k3.h"
#include "sparkinfer/kernels/kimi_k3_fast.h"
#include "k3_fast_test_util.h"

#include <cstring>
#include <cuda_fp16.h>

using namespace sparkinfer::kernels::k3;
using namespace k3test;
#define CU K3T_CU

// ---------------------------------------------------------------------------
// The claim is BIT-IDENTICAL, so the check is memcmp against k3_proj_ggml_f32, not a
// tolerance. A tolerance here would pass for a kernel that reassociated the warp fold,
// which is exactly the thing the epilogue must not do.
static void test_proj_one_barrier() {
    std::printf("Q8_0 multi-row GEMV: one barrier, bit-identical\n");
    struct Shape { int N, K; const char* what; };
    const Shape shapes[] = {
        {7168, 3584, "ffn_routed_up   N7168 K3584 nb112 ROWS16"},
        {3584, 7168, "ffn_routed_down N3584 K7168 nb224 ROWS8"},
        {7168, 1536, "attn_output     N7168 K1536 nb48  ROWS16"},
        {7168,  768, "ffn_down_shexp  N7168 K768  nb24  ROWS16"},
        {2048, 7168, "a ROWS8 shape   N2048 K7168"},
        {1024, 3584, "a ROWS4 shape   N1024 K3584"},
    };

    std::mt19937 rng(20260803);
    for (const Shape& sh : shapes) {
        const int nb = sh.K / 32;
        auto x = rnd((size_t)sh.K, rng);
        // Q8_0 weights as raw bytes: 34 per block, and the scale is an fp16 bit
        // pattern, so this is generated as bytes rather than as floats.
        std::vector<unsigned char> wbytes((size_t)sh.N * nb * 34);
        {
            std::uniform_int_distribution<int> ub(0, 255);
            for (auto& b : wbytes) b = (unsigned char)ub(rng);
            // Keep the fp16 scales finite and modest: exponent field small.
            for (size_t blk = 0; blk < (size_t)sh.N * nb; ++blk) {
                wbytes[blk * 34 + 1] = (unsigned char)(0x30 | (wbytes[blk * 34 + 1] & 0x03));
            }
        }

        float* dx = to_dev(x);
        void* dW = nullptr;
        CU(cudaMalloc(&dW, wbytes.size()));
        CU(cudaMemcpy(dW, wbytes.data(), wbytes.size(), cudaMemcpyHostToDevice));
        void* dq8 = nullptr;
        CU(cudaMalloc(&dq8, (size_t)nb * 34));
        float *dy1 = nullptr, *dy2 = nullptr;
        CU(cudaMalloc(&dy1, (size_t)sh.N * sizeof(float)));
        CU(cudaMalloc(&dy2, (size_t)sh.N * sizeof(float)));

        const bool took = k3_proj_q8_multirow_1bar(dy1, dx, dW, 8, sh.N, sh.K, dq8, 0);
        CU(cudaDeviceSynchronize());
        CU(cudaGetLastError());
        const bool ref_ok = k3_proj_ggml_f32(dy2, dx, dW, 8, sh.N, sh.K, dq8, 0);
        CU(cudaDeviceSynchronize());
        CU(cudaGetLastError());

        ++g_case;
        if (!took || !ref_ok) {
            std::printf("  %-46s %s\n", sh.what,
                        took ? "FAIL (reference declined)" : "FAIL (declined)");
            ++g_fail;
        } else {
            const auto a = from_dev(dy1, (size_t)sh.N);
            const auto b = from_dev(dy2, (size_t)sh.N);
            const bool same = std::memcmp(a.data(), b.data(),
                                          a.size() * sizeof(float)) == 0;
            size_t ndiff = 0;
            for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) ++ndiff;
            std::printf("  %-46s %s (%zu/%zu differ)\n", sh.what,
                        same ? "OK bit-identical" : "FAIL", ndiff, a.size());
            if (!same) ++g_fail;
        }
        CU(cudaFree(dx)); CU(cudaFree(dW)); CU(cudaFree(dq8));
        CU(cudaFree(dy1)); CU(cudaFree(dy2));
    }
}

// The fused four-tensor counterpart, same bit-identity claim, same memcmp.
static void test_fused4_one_barrier() {
    std::printf("Q8_0 fused-4 GEMV (KDA q/k/v/g): one barrier, bit-identical\n");
    const int N = 1536, K = 7168;           // this rank's qkv band at tp=8
    const int nb = K / 32;
    std::mt19937 rng(4);
    auto x = rnd((size_t)K, rng);
    std::vector<unsigned char> wb((size_t)N * nb * 34);
    std::uniform_int_distribution<int> ub(0, 255);
    float* dW[4]; float* dY1[4]; float* dY2[4];
    for (int t = 0; t < 4; ++t) {
        for (auto& b : wb) b = (unsigned char)ub(rng);
        for (size_t blk = 0; blk < (size_t)N * nb; ++blk)
            wb[blk * 34 + 1] = (unsigned char)(0x30 | (wb[blk * 34 + 1] & 0x03));
        CU(cudaMalloc(&dW[t], wb.size()));
        CU(cudaMemcpy(dW[t], wb.data(), wb.size(), cudaMemcpyHostToDevice));
        CU(cudaMalloc(&dY1[t], (size_t)N * sizeof(float)));
        CU(cudaMalloc(&dY2[t], (size_t)N * sizeof(float)));
    }
    float* dx = to_dev(x);
    void* dq8 = nullptr; CU(cudaMalloc(&dq8, (size_t)nb * 34));

    const bool a = k3_proj_q8_fused4_1bar(dY1[0], dY1[1], dY1[2], dY1[3], dx,
                                          dW[0], dW[1], dW[2], dW[3], 8, N, K, dq8, 0);
    CU(cudaDeviceSynchronize()); CU(cudaGetLastError());
    const bool b = k3_proj_ggml_f32_x4(dY2[0], dY2[1], dY2[2], dY2[3], dx,
                                       dW[0], dW[1], dW[2], dW[3], 8, N, K, dq8, 0);
    CU(cudaDeviceSynchronize()); CU(cudaGetLastError());
    note("both paths engaged", a && b);
    for (int t = 0; t < 4; ++t) {
        const auto p = from_dev(dY1[t], (size_t)N), q = from_dev(dY2[t], (size_t)N);
        size_t nd = 0;
        for (size_t i = 0; i < p.size(); ++i) if (p[i] != q[i]) ++nd;
        ++g_case;
        char lbl[64]; std::snprintf(lbl, sizeof(lbl), "tensor %d of 4 (N1536 K7168 ROWS4)", t);
        std::printf("  %-46s %s (%zu/%zu differ)\n", lbl,
                    nd == 0 ? "OK bit-identical" : "FAIL", nd, p.size());
        if (nd) ++g_fail;
        CU(cudaFree(dW[t])); CU(cudaFree(dY1[t])); CU(cudaFree(dY2[t]));
    }
    CU(cudaFree(dx)); CU(cudaFree(dq8));
}

// The warp-per-block quantiser must emit the SAME 34 bytes per block as the reference.
//
// The reference is computed on the HOST, not by calling back into the library: every
// library entry point now routes through the warp path, so a device-side "reference"
// would compare the kernel to itself. Checked as BYTES — the whole claim is that a max
// over magnitudes is order-independent, and a tolerance would not test that.
static void test_quantize_warp() {
    std::printf("Q8_0 activation quantise: one warp per block vs a host reference\n");
    std::mt19937 rng(90210);
    for (int K : {7168, 3584, 1536, 768, 128}) {
        const int nb = K / 32;
        auto x = rnd((size_t)K, rng, -8.f, 8.f);
        // A block of exact zeros exercises the amax==0 branch (id must be 0, not inf).
        if (K >= 64) for (int j = 32; j < 64; ++j) x[j] = 0.0f;
        float* dx = to_dev(x);
        void* dq = nullptr; CU(cudaMalloc(&dq, (size_t)nb * 34));
        CU(cudaMemset(dq, 0xCD, (size_t)nb * 34));
        k3_quantize_q8_0(dq, dx, nb, 0);
        CU(cudaDeviceSynchronize()); CU(cudaGetLastError());
        std::vector<unsigned char> got((size_t)nb * 34);
        CU(cudaMemcpy(got.data(), dq, got.size(), cudaMemcpyDeviceToHost));

        // Host reference: the reference kernel's arithmetic, line for line.
        // __float2int_rn is round-to-nearest-EVEN, which is std::nearbyint under the
        // default rounding mode.
        size_t nd = 0;
        for (int b = 0; b < nb; ++b) {
            float amax = 0.0f;
            for (int j = 0; j < 32; ++j) amax = std::fmax(amax, std::fabs(x[b * 32 + j]));
            const float d  = amax / 127.0f;
            const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
            // __float2half_rn / __half_as_ushort are device-only; __float2half is
            // host+device and is round-to-nearest-even, so take its bits directly.
            const __half dhalf = __float2half(d);
            unsigned short dh; std::memcpy(&dh, &dhalf, sizeof(dh));
            if (std::memcmp(&got[(size_t)b * 34], &dh, 2) != 0) ++nd;
            for (int j = 0; j < 32; ++j) {
                const signed char q = (signed char)(int)std::nearbyint(x[b * 32 + j] * id);
                if ((signed char)got[(size_t)b * 34 + 2 + j] != q) ++nd;
            }
        }
        ++g_case;
        char lbl[64]; std::snprintf(lbl, sizeof(lbl), "K=%d (nb=%d, incl. an all-zero block)", K, nb);
        std::printf("  %-46s %s (%zu mismatches)\n", lbl,
                    nd == 0 ? "OK bit-identical" : "FAIL", nd);
        if (nd) ++g_fail;
        CU(cudaFree(dx)); CU(cudaFree(dq));
    }
}

int main() {
    if (!have_device()) return 0;
    std::printf("K3 Q8_0 multi-row GEMV, one barrier\n\n");
    test_proj_one_barrier();   std::printf("\n");
    test_fused4_one_barrier(); std::printf("\n");
    test_quantize_warp();
    return report();
}
