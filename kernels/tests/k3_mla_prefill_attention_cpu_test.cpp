#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static void attention(float* out, const float* q, const float* cache, const float* w,
                      const float* gate, int start, int tokens, int context, int heads,
                      int key, int latent, int value, double scale) {
    for (int t = 0; t < tokens; ++t) for (int h = 0; h < heads; ++h) {
        std::vector<double> acc(latent, 0.0);
        double m = -INFINITY, l = 0.0;
        const float* qt = q + ((size_t)t * heads + h) * key;
        for (int p = 0; p <= start + t && p < context; ++p) {
            const float* row = cache + (size_t)p * key;
            double s = 0.0;
            for (int d = 0; d < key; ++d) s += (double)qt[d] * row[d];
            s *= scale;
            const double nm = std::max(m, s);
            const double a = std::isfinite(m) ? std::exp(m - nm) : 0.0;
            const double b = std::exp(s - nm);
            l = l * a + b;
            for (int d = 0; d < latent; ++d) acc[d] = acc[d] * a + row[d] * b;
            m = nm;
        }
        for (int v = 0; v < value; ++v) {
            double y = 0.0;
            const float* wh = w + ((size_t)h * value + v) * latent;
            for (int d = 0; d < latent; ++d) y += acc[d] * wh[d];
            y /= l;
            const size_t i = ((size_t)t * heads + h) * value + v;
            if (gate) y *= 1.0 / (1.0 + std::exp(-(double)gate[i]));
            out[i] = (float)y;
        }
    }
}

static void attention_split(float* out, const float* q, const float* cache,
                            const float* w, const float* gate, int start, int tokens,
                            int context, int heads, int key, int latent, int value,
                            int splits, double scale) {
    for (int t = 0; t < tokens; ++t) for (int h = 0; h < heads; ++h) {
        std::vector<double> pm(splits, -INFINITY), pl(splits, 0.0);
        std::vector<double> pa((size_t)splits * latent, 0.0);
        const int end = std::min(context, start + t + 1);
        const float* qt = q + ((size_t)t * heads + h) * key;
        for (int s = 0; s < splits; ++s) {
            const int p0 = end * s / splits, p1 = end * (s + 1) / splits;
            for (int p = p0; p < p1; ++p) {
                const float* row = cache + (size_t)p * key;
                double score = 0.0;
                for (int d = 0; d < key; ++d) score += (double)qt[d] * row[d];
                score *= scale;
                const double nm = std::max(pm[s], score);
                const double a = std::isfinite(pm[s]) ? std::exp(pm[s] - nm) : 0.0;
                const double b = std::exp(score - nm);
                pl[s] = pl[s] * a + b;
                for (int d = 0; d < latent; ++d)
                    pa[(size_t)s * latent + d] =
                        pa[(size_t)s * latent + d] * a + row[d] * b;
                pm[s] = nm;
            }
        }
        const double m = *std::max_element(pm.begin(), pm.end());
        double l = 0.0;
        std::vector<double> acc(latent, 0.0);
        for (int s = 0; s < splits; ++s) if (std::isfinite(pm[s])) {
            const double a = std::exp(pm[s] - m);
            l += pl[s] * a;
            for (int d = 0; d < latent; ++d)
                acc[d] += pa[(size_t)s * latent + d] * a;
        }
        for (int v = 0; v < value; ++v) {
            const float* wh = w + ((size_t)h * value + v) * latent;
            double y = 0.0;
            for (int d = 0; d < latent; ++d) y += acc[d] * wh[d];
            const size_t i = ((size_t)t * heads + h) * value + v;
            y /= l;
            if (gate) y /= 1.0 + std::exp(-(double)gate[i]);
            out[i] = (float)y;
        }
    }
}

int main() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
    constexpr int H = 3, K = 11, L = 7, V = 5;
    for (int start : {0, 1, 17, 63}) for (int T : {1, 2, 3, 8, 16, 31, 32, 33, 64}) {
        const int context = start + T;
        std::vector<float> q((size_t)T*H*K), cache((size_t)context*K), w((size_t)H*V*L);
        std::vector<float> gate((size_t)T*H*V), tiled(gate.size()), repeated(gate.size());
        std::vector<float> split(gate.size());
        for (auto* a : {&q, &cache, &w, &gate}) for (float& x : *a) x = dist(rng);
        attention(tiled.data(), q.data(), cache.data(), w.data(), gate.data(), start, T,
                  context, H, K, L, V, 1.0/std::sqrt((double)K));
        attention_split(split.data(), q.data(), cache.data(), w.data(), gate.data(), start,
                        T, context, H, K, L, V, 7, 1.0/std::sqrt((double)K));
        for (int t = 0; t < T; ++t)
            attention(repeated.data() + (size_t)t*H*V, q.data() + (size_t)t*H*K,
                      cache.data(), w.data(), gate.data() + (size_t)t*H*V,
                      start+t, 1, context, H, K, L, V, 1.0/std::sqrt((double)K));
        for (size_t i = 0; i < tiled.size(); ++i) {
            assert(tiled[i] == repeated[i]);
            assert(std::fabs(tiled[i] - split[i]) < 2e-7f);
        }
    }
    std::puts("k3_mla_prefill_attention_cpu_test: ok");
}
