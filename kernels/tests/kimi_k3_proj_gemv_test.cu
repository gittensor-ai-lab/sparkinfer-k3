// k3_proj_f32 (generic f32-activation GEMV) vs a float64 reference built from the SAME
// dequant math ggml uses for Q8_0 (y = qs*d), plus the trivial F32 dense case.
//
// This is the projection every non-expert K3 weight goes through (attn_q/k/v/output,
// ffn_gate/up/down for the leading dense layer, ffn_routed_down/up, ssm_f_a/f_b/beta,
// ssm_g, token_embd, output.weight — everything the real file showed as Q8_0 or F32).
// It runs in f32 throughout rather than reusing the existing bf16 GEMV path, because
// truncating K3's already float64-validated kernels to bf16 at every projection would
// reintroduce the precision loss this whole project's kernel tests exist to eliminate.

#include "sparkinfer/kernels/kimi_k3.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace sparkinfer::kernels::k3;

#define CU(e) do{ cudaError_t e_=(e); if(e_!=cudaSuccess){ \
  std::printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); return 1;} }while(0)

struct BlockQ8_0 { uint16_t d; int8_t qs[32]; };
static_assert(sizeof(BlockQ8_0) == 34, "Q8_0 block must be 34 bytes");

static float h2f(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const uint32_t exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) { bits = sign; }
    else if (exp == 31) bits = sign | 0x7f800000u | (man << 13);
    else bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    float f; std::memcpy(&f, &bits, 4); return f;
}

int main() {
    int fail = 0;
    std::mt19937 rng(20260730);
    std::uniform_real_distribution<float> U(-2.f, 2.f);

    // ---- Q8_0 case ----
    {
        const int N = 300, K = 512;   // K/32 = 16 blocks/row, N not a multiple of BLOCK
        std::vector<BlockQ8_0> W((size_t)N * (K / 32));
        for (auto& b : W) {
            const uint16_t exp = (uint16_t)(9 + (rng() % 8));
            b.d = (uint16_t)(((rng() & 1) << 15) | (exp << 10) | (rng() % 1024));
            for (auto& q : b.qs) q = (int8_t)((int)(rng() % 256) - 128);
        }
        std::vector<float> x(K); for (auto& v : x) v = U(rng);

        std::vector<double> ref(N, 0.0);
        for (int n = 0; n < N; ++n) {
            double acc = 0;
            for (int b = 0; b < K / 32; ++b) {
                const auto& blk = W[(size_t)n * (K / 32) + b];
                const double d = h2f(blk.d);
                for (int j = 0; j < 32; ++j) acc += d * (double)blk.qs[j] * (double)x[b * 32 + j];
            }
            ref[n] = acc;
        }

        void *dW, *dx, *dy;
        CU(cudaMalloc(&dW, W.size() * sizeof(BlockQ8_0)));
        CU(cudaMalloc(&dx, K * sizeof(float)));
        CU(cudaMalloc(&dy, N * sizeof(float)));
        CU(cudaMemcpy(dW, W.data(), W.size() * sizeof(BlockQ8_0), cudaMemcpyHostToDevice));
        CU(cudaMemcpy(dx, x.data(), K * sizeof(float), cudaMemcpyHostToDevice));
        const bool ok = k3_proj_f32((float*)dy, (const float*)dx, dW, /*wtype=*/8, N, K, 0);
        CU(cudaDeviceSynchronize());
        std::vector<float> got(N);
        CU(cudaMemcpy(got.data(), dy, N * sizeof(float), cudaMemcpyDeviceToHost));

        double num = 0, den = 0;
        for (int n = 0; n < N; ++n) { double d = got[n] - ref[n]; num += d*d; den += ref[n]*ref[n]; }
        const double rl2 = std::sqrt(num / (den + 1e-30));
        std::printf("[Q8_0] N=%d K=%d accepted=%s relL2=%.3e\n", N, K, ok ? "yes" : "NO", rl2);
        if (!ok || rl2 > 1e-5) ++fail;

        // llama.cpp CPU compatibility path: it first quantizes x to block_q8_0,
        // including the fp16 activation scale, then performs an integer dot.
        std::vector<BlockQ8_0> Xq(K / 32);
        for (int b = 0; b < K / 32; ++b) {
            float amax = 0.0f;
            for (int j = 0; j < 32; ++j) amax = std::fmax(amax, std::fabs(x[32*b+j]));
            const float d = amax / 127.0f;
            Xq[b].d = __half_as_ushort(__float2half_rn(d));
            const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
            for (int j = 0; j < 32; ++j)
                Xq[b].qs[j] = (int8_t)std::nearbyint(x[32*b+j] * id);
        }
        std::vector<double> qref(N, 0.0);
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int b = 0; b < K / 32; ++b) {
                int sumi = 0;
                const auto& wb = W[(size_t)n * (K / 32) + b];
                for (int j = 0; j < 32; ++j) sumi += (int)wb.qs[j] * (int)Xq[b].qs[j];
                sum += (float)sumi * (h2f(wb.d) * h2f(Xq[b].d));
            }
            qref[n] = sum;
        }
        void* dq = nullptr;
        CU(cudaMalloc(&dq, k3_q8_0_bytes(K)));
        const bool qok = k3_proj_ggml_f32((float*)dy, (const float*)dx, dW, 8, N, K, dq, 0);
        CU(cudaDeviceSynchronize());
        CU(cudaMemcpy(got.data(), dy, N * sizeof(float), cudaMemcpyDeviceToHost));
        double qnum = 0, qden = 0;
        for (int n = 0; n < N; ++n) {
            const double d = got[n] - qref[n];
            qnum += d*d; qden += qref[n]*qref[n];
        }
        const double qrl2 = std::sqrt(qnum / (qden + 1e-30));
        std::printf("[Q8_0 x Q8_0] accepted=%s relL2=%.3e\n",
                    qok ? "yes" : "NO", qrl2);
        if (!qok || qrl2 > 1e-6) ++fail;
        cudaFree(dq);
        cudaFree(dW); cudaFree(dx); cudaFree(dy);
    }

    // ---- Q8_0 through the MULTIROW path ----
    //
    // k3_proj_f32 switches to proj_q8_0_multirow_kernel at N >= MULTIROW_MIN_N (1024).
    // Every case above is N = 300 or 200, so the multirow kernel — which is the one that
    // actually runs for K3's real projections (attn_q/k/v/output are N = 12288) — had no
    // coverage here at all; its correctness rested on the node run alone.
    //
    // N = 1026 is deliberately not a multiple of ROWS (4). The kernel walks ROWS rows per
    // block and guards each with `if (n0 + r >= n_rows) continue;`, so the last block runs
    // partly out of range. An off-by-one there reads a row that does not exist, which is
    // in-bounds of the allocation for a wrong row and therefore produces a plausible
    // number rather than a fault — exactly the failure mode a shape test is for.
    {
        const int N = 1026, K = 2048;
        const int bpr = K / 32;
        std::vector<BlockQ8_0> W((size_t)N * bpr);
        for (auto& b : W) {
            const uint16_t exp = (uint16_t)(9 + (rng() % 8));
            b.d = (uint16_t)(((rng() & 1) << 15) | (exp << 10) | (rng() % 1024));
            for (auto& q : b.qs) q = (int8_t)((int)(rng() % 256) - 128);
        }
        std::vector<float> x(K); for (auto& v : x) v = U(rng);

        std::vector<double> ref(N, 0.0);
        for (int n = 0; n < N; ++n) {
            double acc = 0;
            for (int b = 0; b < bpr; ++b) {
                const auto& blk = W[(size_t)n * bpr + b];
                const double d = h2f(blk.d);
                for (int j = 0; j < 32; ++j) acc += d * (double)blk.qs[j] * (double)x[b * 32 + j];
            }
            ref[n] = acc;
        }

        void *dW, *dx, *dy;
        CU(cudaMalloc(&dW, W.size() * sizeof(BlockQ8_0)));
        CU(cudaMalloc(&dx, K * sizeof(float)));
        CU(cudaMalloc(&dy, N * sizeof(float)));
        CU(cudaMemcpy(dW, W.data(), W.size() * sizeof(BlockQ8_0), cudaMemcpyHostToDevice));
        CU(cudaMemcpy(dx, x.data(), K * sizeof(float), cudaMemcpyHostToDevice));
        const bool ok = k3_proj_f32((float*)dy, (const float*)dx, dW, /*wtype=*/8, N, K, 0);
        CU(cudaDeviceSynchronize());
        std::vector<float> got(N);
        CU(cudaMemcpy(got.data(), dy, N * sizeof(float), cudaMemcpyDeviceToHost));

        double num = 0, den = 0; int worst_n = -1; double worst = 0;
        for (int n = 0; n < N; ++n) {
            const double d = got[n] - ref[n];
            num += d*d; den += ref[n]*ref[n];
            const double rel = std::fabs(d) / (std::fabs(ref[n]) + 1e-12);
            if (rel > worst) { worst = rel; worst_n = n; }
        }
        const double rl2 = std::sqrt(num / (den + 1e-30));
        std::printf("[Q8_0] N=%d K=%d multirow (N%%ROWS=%d) accepted=%s relL2=%.3e worst_row=%d\n",
                    N, K, N % 4, ok ? "yes" : "NO", rl2, worst_n);
        if (!ok || rl2 > 1e-5 || !(den > 0)) ++fail;
        cudaFree(dW); cudaFree(dx); cudaFree(dy);
    }

    // ---- F32 dense case ----
    {
        const int N = 200, K = 384;
        std::vector<float> W((size_t)N * K), x(K);
        for (auto& v : W) v = U(rng);
        for (auto& v : x) v = U(rng);
        std::vector<double> ref(N, 0.0);
        for (int n = 0; n < N; ++n) {
            double acc = 0;
            for (int k = 0; k < K; ++k) acc += (double)W[(size_t)n*K+k] * (double)x[k];
            ref[n] = acc;
        }
        void *dW, *dx, *dy;
        CU(cudaMalloc(&dW, W.size()*4)); CU(cudaMalloc(&dx, K*4)); CU(cudaMalloc(&dy, N*4));
        CU(cudaMemcpy(dW, W.data(), W.size()*4, cudaMemcpyHostToDevice));
        CU(cudaMemcpy(dx, x.data(), K*4, cudaMemcpyHostToDevice));
        const bool ok = k3_proj_f32((float*)dy, (const float*)dx, dW, /*wtype=*/0, N, K, 0);
        CU(cudaDeviceSynchronize());
        std::vector<float> got(N);
        CU(cudaMemcpy(got.data(), dy, N*4, cudaMemcpyDeviceToHost));
        double num=0, den=0;
        for (int n = 0; n < N; ++n) { double d = got[n]-ref[n]; num += d*d; den += ref[n]*ref[n]; }
        const double rl2 = std::sqrt(num/(den+1e-30));
        std::printf("[F32]  N=%d K=%d accepted=%s relL2=%.3e\n", N, K, ok ? "yes" : "NO", rl2);
        if (!ok || rl2 > 1e-6) ++fail;
        cudaFree(dW); cudaFree(dx); cudaFree(dy);
    }

    // ---- unsupported type must be refused ----
    {
        float *dy=nullptr, *dx=nullptr; void* dW=nullptr;
        cudaMalloc(&dx, 32*4); cudaMalloc(&dy, 4); cudaMalloc(&dW, 74);
        const bool ok = k3_proj_f32(dy, dx, dW, /*wtype=*/17 /*IQ2_XS*/, 1, 32, 0);
        std::printf("[refuse] IQ2_XS (no plain-projection decoder) accepted=%s\n", ok?"YES -- BUG":"no (correct)");
        if (ok) ++fail;
        cudaFree(dx); cudaFree(dy); cudaFree(dW);
    }

    std::printf("\n%s\n", fail ? "FAIL" : "PASS: k3_proj_f32 matches float64 reference");
    return fail ? 1 : 0;
}
