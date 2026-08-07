#include "sparkinfer/kernels/k3_mla_prefill_attention.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace k3 = sparkinfer::kernels::k3;

static void check(cudaError_t e) { assert(e == cudaSuccess); }

template <typename T>
static T* upload(const std::vector<T>& src) {
    T* p = nullptr;
    check(cudaMalloc(&p, src.size() * sizeof(T)));
    check(cudaMemcpy(p, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

static std::vector<float> reference(
    const std::vector<float>& q, const std::vector<float>& cache,
    const std::vector<float>& w, const std::vector<float>* gate, int start,
    int tokens, int context, int heads, int key, int latent, int value,
    double scale) {
    std::vector<float> out((size_t)tokens * heads * value);
    for (int t = 0; t < tokens; ++t) for (int h = 0; h < heads; ++h) {
        const int end = std::min(context, start + t + 1);
        std::vector<double> score(end);
        double m = -INFINITY, z = 0.0;
        for (int p = 0; p < end; ++p) {
            double s = 0.0;
            for (int d = 0; d < key; ++d)
                s += (double)q[((size_t)t * heads + h) * key + d] *
                     cache[(size_t)p * key + d];
            score[p] = s * scale;
            m = std::max(m, score[p]);
        }
        for (double s : score) z += std::exp(s - m);
        for (int v = 0; v < value; ++v) {
            double y = 0.0;
            for (int p = 0; p < end; ++p) {
                double latent_value = 0.0;
                for (int d = 0; d < latent; ++d)
                    latent_value += cache[(size_t)p * key + d] *
                                    w[((size_t)h * value + v) * latent + d];
                y += std::exp(score[p] - m) * latent_value;
            }
            const size_t i = ((size_t)t * heads + h) * value + v;
            y /= z;
            if (gate) y /= 1.0 + std::exp(-(double)(*gate)[i]);
            out[i] = (float)y;
        }
    }
    return out;
}

static void compare(const std::vector<float>& got, const std::vector<float>& ref,
                    float atol) {
    assert(got.size() == ref.size());
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::fabs(got[i] - ref[i]) > atol) {
            std::fprintf(stderr, "mismatch[%zu]: got %.9g ref %.9g\n", i, got[i], ref[i]);
            assert(false);
        }
    }
}

static void run_case(int tokens, int context, int start, int heads, int key,
                     int latent, int value, int splits, bool with_gate,
                     int scale_width = 0) {
    std::mt19937 rng(11 + key + heads);
    std::uniform_real_distribution<float> dist(-0.08f, 0.08f);
    std::vector<float> q((size_t)tokens * heads * key);
    std::vector<float> cache((size_t)context * key);
    std::vector<float> w((size_t)heads * value * latent);
    std::vector<float> gate((size_t)tokens * heads * value);
    for (auto* a : {&q, &cache, &w, &gate}) for (float& x : *a) x = dist(rng);

    float* dq = upload(q); float* dc = upload(cache); float* dw = upload(w);
    float* dg = with_gate ? upload(gate) : nullptr;
    float *dy = nullptr, *dy_split = nullptr, *dy_half = nullptr;
    check(cudaMalloc(&dy, gate.size() * sizeof(float)));
    check(cudaMalloc(&dy_split, gate.size() * sizeof(float)));
    check(cudaMalloc(&dy_half, gate.size() * sizeof(float)));
    const size_t ws_bytes = k3::k3_mla_prefill_attention_workspace_bytes(
        tokens, heads, latent, splits);
    void* ws = nullptr; check(cudaMalloc(&ws, ws_bytes));
    const float scale = 1.0f / std::sqrt((float)(scale_width ? scale_width : key));

    assert(k3::k3_mla_prefill_attention_f32(
        dy, dq, dc, dw, dg, start, tokens, context, heads, key, latent, value, scale, 0));
    assert(k3::k3_mla_prefill_attention_split_f32(
        dy_split, ws, ws_bytes, dq, dc, dw, dg, start, tokens, context, heads,
        key, latent, value, splits, scale, 0));
    // Workspace size is a contract, not a suggestion: declining is safer than writing
    // the next tile buffer. This returns before launching anything.
    assert(!k3::k3_mla_prefill_attention_split_f32(
        dy_split, ws, ws_bytes - sizeof(float), dq, dc, dw, dg, start, tokens,
        context, heads, key, latent, value, splits, scale, 0));

    std::vector<__half> cache_h(cache.size());
    std::vector<float> cache_h_f32(cache.size());
    for (size_t i = 0; i < cache.size(); ++i) {
        cache_h[i] = __float2half(cache[i]);
        cache_h_f32[i] = __half2float(cache_h[i]);
    }
    __half* dc_h = upload(cache_h);
    assert(k3::k3_mla_prefill_attention_split_kvf16(
        dy_half, ws, ws_bytes, dq, dc_h, dw, dg, start, tokens, context, heads,
        key, latent, value, splits, scale, 0));
    check(cudaDeviceSynchronize());

    std::vector<float> y(gate.size()), y_split(gate.size()), y_half(gate.size());
    check(cudaMemcpy(y.data(), dy, y.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(y_split.data(), dy_split, y_split.size() * sizeof(float),
                     cudaMemcpyDeviceToHost));
    check(cudaMemcpy(y_half.data(), dy_half, y_half.size() * sizeof(float),
                     cudaMemcpyDeviceToHost));
    const std::vector<float>* gp = with_gate ? &gate : nullptr;
    compare(y, reference(q, cache, w, gp, start, tokens, context, heads, key,
                         latent, value, scale), 3e-5f);
    compare(y_split, y, 3e-5f);
    compare(y_half, reference(q, cache_h_f32, w, gp, start, tokens, context, heads,
                              key, latent, value, scale), 4e-5f);

    cudaFree(dq); cudaFree(dc); cudaFree(dc_h); cudaFree(dw); cudaFree(dg);
    cudaFree(dy); cudaFree(dy_split); cudaFree(dy_half); cudaFree(ws);
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::puts("k3_mla_prefill_attention_gpu_test: SKIP (no CUDA device)");
        return 0;
    }
    assert(k3::k3_mla_prefill_attention_supported(64, 32768, 12, 576, 512, 128));
    assert(!k3::k3_mla_prefill_attention_supported(1, 1, 1, 577, 512, 128));
    assert(k3::k3_mla_prefill_attention_workspace_bytes(1, 1, 512, 33) == 0);

    // Full band + ragged tail, empty early splits, gate and no-gate contracts.
    run_case(5, 12, 7, 5, 11, 7, 5, 4, true);
    run_case(2, 2, 0, 5, 11, 7, 5, 4, false);
    // Real per-rank K3 geometry. A short context keeps CI work bounded while exercising
    // the production shared-memory layout and 512-wide latent reconstruction.
    run_case(2, 4, 2, 12, 576, 512, 128, 4, true, 192);
    std::puts("k3_mla_prefill_attention_gpu_test: ok");
}
