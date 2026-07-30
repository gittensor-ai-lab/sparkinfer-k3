// MoE FFN-branch validation, inside the executor, against an independent float64
// reference. This closes the one major executor path with no end-to-end numerical
// check: kimi_k3_layer0_ref_check covered the DENSE FFN (leading layer), and
// kimi_k3_mla_ref_check stopped at the attention output. Neither touched the ROUTED
// MoE branch as the executor actually wires it — router -> routed_down -> expert
// dispatch -> routed_norm -> routed_up -> shared-experts-add.
//
// The individual kernels here are already validated standalone (the router in
// kimi_k3_numeric_test, the expert dispatch bit-close against float64 on real
// weights in kimi_k3_moe_real_test). What is NOT validated is their COMPOSITION in
// forward_layer: right buffer to the right kernel, routed_norm on the dispatch
// OUTPUT not its input, the shared-expert width and add, the latent hop on both
// sides. Those are exactly the wiring bugs this project keeps surfacing.
//
// ISOLATION, same method as the MLA check: the FFN branch is fed the executor's OWN
// normed2 (captured via the "ffn_norm" debug tag), so this tests the FFN branch
// given a shared input rather than re-deriving the attention branch. The comparison
// is against the executor's "ffn_out" debug tag (the FFN output before the residual
// add), on a real MoE layer (layer 1: KDA attention + MoE FFN, complete in shard 1),
// with the REAL IQ1_S expert weights dequantized through the already-bit-exact GPU
// dequant.

#include "sparkinfer/models/kimi_k3.h"
#include "sparkinfer/models/kimi_k3_gguf_manifest.h"
#include "sparkinfer/kernels/kimi_k3.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

using namespace sparkinfer;
namespace k3k = sparkinfer::kernels::k3;

struct BlockQ8_0 { uint16_t d; int8_t qs[32]; };
static double h2f(uint16_t h) {
    const uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t b;
    if (e == 0) {
        if (m == 0) b = s;
        else { int ee = -1; uint32_t mm = m; do { mm <<= 1; ++ee; } while (!(mm & 0x400));
               b = s | ((uint32_t)(127 - 15 - ee) << 23) | ((mm & 0x3ff) << 13); }  // subnormal
    } else if (e == 31) b = s | 0x7f800000u | (m << 13);
    else b = s | ((e - 15 + 127) << 23) | (m << 13);
    float f; std::memcpy(&f, &b, 4); return (double)f;
}
static std::vector<double> read_f32(const GGUFTensor* t, long n) {
    std::vector<double> o(n);
    const float* p = (const float*)t->data;
    for (long i = 0; i < n; ++i) o[i] = (double)p[i];
    return o;
}
static std::vector<double> read_matrix(const GGUFTensor* t, int N, int K) {
    if (t->ggml_type == 0) return read_f32(t, (long)N * K);
    if (t->ggml_type != 8) { std::fprintf(stderr, "type %d unhandled\n", t->ggml_type); std::exit(1); }
    std::vector<double> o((size_t)N * K);
    const BlockQ8_0* b = (const BlockQ8_0*)t->data;
    const int bpr = K / 32;
    for (int n = 0; n < N; ++n)
        for (int q = 0; q < bpr; ++q) {
            const double d = h2f(b[(size_t)n * bpr + q].d);
            for (int j = 0; j < 32; ++j) o[(size_t)n * K + q * 32 + j] = d * (double)b[(size_t)n * bpr + q].qs[j];
        }
    return o;
}
static std::vector<double> matvec(const std::vector<double>& W, const std::vector<double>& x,
                                  int N, int K) {
    std::vector<double> y(N, 0.0);
    for (int n = 0; n < N; ++n) {
        double a = 0; const double* w = &W[(size_t)n * K];
        for (int k = 0; k < K; ++k) a += w[k] * x[k];
        y[n] = a;
    }
    return y;
}

// Dequantize ONE expert's [rows x cols] matrix from the GGUF via the GPU dequant
// (bit-exact vs ggml), returned row-major as double. `expert_bytes` is the byte
// stride between experts; `blk_bytes`/`blk_elems` describe the quant type.
static std::vector<double> dequant_expert(const GGUFTensor* t, int e, int rows, int cols,
                                          long blk_bytes, long blk_elems) {
    const long blocks_per_row = cols / blk_elems;
    const long bytes_per_row = blocks_per_row * blk_bytes;
    const long bytes_per_expert = (long)rows * bytes_per_row;
    const long n_values = (long)rows * cols;

    const uint8_t* src = (const uint8_t*)t->data + (size_t)e * bytes_per_expert;
    void* d_src; float* d_out;
    cudaMalloc(&d_src, bytes_per_expert);
    cudaMalloc(&d_out, n_values * sizeof(float));
    cudaMemcpy(d_src, src, bytes_per_expert, cudaMemcpyHostToDevice);
    const bool ok = k3k::dequant_f32_by_type(d_out, d_src, n_values, t->ggml_type, 0);
    cudaDeviceSynchronize();
    if (!ok) { std::fprintf(stderr, "expert dequant failed (type %d)\n", t->ggml_type); std::exit(1); }
    std::vector<float> f(n_values);
    cudaMemcpy(f.data(), d_out, n_values * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_src); cudaFree(d_out);
    std::vector<double> o(n_values);
    for (long i = 0; i < n_values; ++i) o[i] = (double)f[i];
    return o;
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1]
        : "/workspace/models_k3/UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf";
    const int LAYER = argc > 2 ? std::atoi(argv[2]) : 1;   // layer 1: KDA + MoE, complete

    GGUF g; KimiK3Config cfg; KimiK3LayerCoverage cov;
    if (!kimi_k3_load_partial(path, cfg, g, &cov) || !cov.layer_complete[LAYER] ||
        LAYER < cfg.leading_dense) {
        std::printf("layer %d not a complete MoE layer\n", LAYER); return 1;
    }
    const int H = cfg.hidden, LAT = cfg.expert_latent, FF = cfg.moe_ffn;
    const int NE = cfg.n_experts, TK = cfg.top_k;
    const double eps = cfg.rms_eps;
    const double beta = cfg.situ_beta, lb = cfg.situ_linear_beta;

    char pfx[32]; std::snprintf(pfx, sizeof(pfx), "blk.%d.", LAYER);
    auto T = [&](const char* s) { return g.tensor((std::string(pfx) + s).c_str()); };

    K3PlanOptions opt;
    opt.has_q_lora        = T("attn_q_a.weight") != nullptr;
    opt.has_attn_gate     = T("attn_gate.weight") != nullptr;
    opt.has_fused_kv_b    = false;
    opt.has_routed_norm   = T("ffn_routed_norm.weight") != nullptr;
    opt.has_shared_experts= T("ffn_gate_shexp.weight") != nullptr;
    std::printf("layer %d: routed_norm=%d shared_experts=%d  (H=%d latent=%d moe_ffn=%d "
                "experts=%d top_k=%d)\n", LAYER, opt.has_routed_norm, opt.has_shared_experts,
                H, LAT, FF, NE, TK);

    // ---- executor: run the layer, capture normed2 (ffn_norm) and ffn_out ----
    KimiK3Weights w;
    if (!kimi_k3_load_weights(g, cfg, opt, w, LAYER, LAYER)) { std::printf("load failed\n"); return 1; }
    KimiK3RuntimeState st;
    if (!kimi_k3_alloc_state(cfg, 16, st)) { std::printf("state alloc failed\n"); return 1; }
    KimiK3Forward fwd;
    fwd.cfg = &cfg; fwd.w = &w; fwd.state = &st; fwd.opt = opt; fwd.stream = 0;
    if (!kimi_k3_forward_alloc_scratch(cfg, fwd)) { std::printf("scratch failed\n"); return 1; }

    std::vector<double> normed2(H);
    std::vector<float> exec_ffn_out(H);
    bool got_norm = false, got_out = false;
    fwd.debug = [&](const char* tag, int, const float* p, int64_t n) {
        if (std::string(tag) == "ffn_norm" && (int)n == H) {
            std::vector<float> tmp(H);
            cudaMemcpy(tmp.data(), p, H * sizeof(float), cudaMemcpyDeviceToHost);
            for (int i = 0; i < H; ++i) normed2[i] = (double)tmp[i];
            got_norm = true;
        } else if (std::string(tag) == "ffn_out" && (int)n == H) {
            cudaMemcpy(exec_ffn_out.data(), p, H * sizeof(float), cudaMemcpyDeviceToHost);
            got_out = true;
        }
    };

    std::vector<float> hx(H);
    for (int i = 0; i < H; ++i) hx[i] = (float)(0.5 * std::sin(0.3 * i) - 0.2 * std::cos(0.11 * i));
    float *d_in, *d_out;
    cudaMalloc(&d_in, H * sizeof(float)); cudaMalloc(&d_out, H * sizeof(float));
    cudaMemcpy(d_in, hx.data(), H * sizeof(float), cudaMemcpyHostToDevice);
    if (!kimi_k3_forward_layer(fwd, LAYER, d_in, d_out)) { std::printf("forward_layer failed\n"); return 1; }
    cudaDeviceSynchronize();
    if (!got_norm || !got_out) { std::printf("debug capture failed\n"); return 1; }

    // ---- independent float64 FFN branch, given the executor's normed2 ----
    auto rms_norm = [&](const std::vector<double>& v, const std::vector<double>& wv) {
        double ss = 0; for (double e : v) ss += e * e;
        const double inv = 1.0 / std::sqrt(ss / v.size() + eps);
        std::vector<double> o(v.size());
        for (size_t i = 0; i < v.size(); ++i) o[i] = v[i] * inv * wv[i];
        return o;
    };
    auto situ = [&](double gv, double uv) {
        const double a = beta * std::tanh(gv / beta) * (1.0 / (1.0 + std::exp(-gv)));
        const double ub = lb > 0 ? (lb * std::tanh(uv / lb)) : uv;
        return a * ub;
    };

    // router: sigmoid(logits) + bias for SELECTION, unbiased probs for weights, renorm.
    auto Wrouter = read_matrix(T("ffn_gate_inp.weight"), NE, H);
    auto bias    = read_f32(T("exp_probs_b.bias"), NE);
    auto logits  = matvec(Wrouter, normed2, NE, H);
    std::vector<double> probs(NE);
    for (int e = 0; e < NE; ++e) probs[e] = 1.0 / (1.0 + std::exp(-logits[e]));
    std::vector<int> order(NE);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        const double sa = probs[a] + bias[a], sb = probs[b] + bias[b];
        return sa != sb ? sa > sb : a < b;   // ties -> lower index, matching the kernel
    });
    std::vector<int> ids(order.begin(), order.begin() + TK);
    std::vector<double> weights(TK);
    double wsum = 0;
    for (int k = 0; k < TK; ++k) { weights[k] = probs[ids[k]]; wsum += weights[k]; }
    const double denom = std::max(wsum, 6.103515625e-5);
    for (double& x : weights) x /= denom;

    // routed_down: H -> latent
    auto Wrd = read_matrix(T("ffn_routed_down.weight"), LAT, H);
    auto routed_in = matvec(Wrd, normed2, LAT, H);

    // expert dispatch: for each selected expert, gate/up @ routed_in -> situ -> down.
    long ggb, gge; gguf_block_info(T("ffn_gate_exps.weight")->ggml_type, ggb, gge);
    const GGUFTensor* gt = T("ffn_gate_exps.weight");
    const GGUFTensor* ut = T("ffn_up_exps.weight");
    const GGUFTensor* dt = T("ffn_down_exps.weight");
    std::vector<double> moe_out(LAT, 0.0);
    for (int k = 0; k < TK; ++k) {
        const int e = ids[k];
        auto ge = dequant_expert(gt, e, FF, LAT, ggb, gge);   // [moe_ffn x latent]
        auto ue = dequant_expert(ut, e, FF, LAT, ggb, gge);
        auto de = dequant_expert(dt, e, LAT, FF, ggb, gge);   // [latent x moe_ffn]
        std::vector<double> act(FF);
        for (int j = 0; j < FF; ++j) {
            double gv = 0, uv = 0;
            const double* gr = &ge[(size_t)j * LAT];
            const double* ur = &ue[(size_t)j * LAT];
            for (int i = 0; i < LAT; ++i) { gv += gr[i] * routed_in[i]; uv += ur[i] * routed_in[i]; }
            act[j] = situ(gv, uv);
        }
        for (int o = 0; o < LAT; ++o) {
            double s = 0; const double* dr = &de[(size_t)o * FF];
            for (int j = 0; j < FF; ++j) s += dr[j] * act[j];
            moe_out[o] += weights[k] * s;
        }
    }

    // routed_norm on the DISPATCH OUTPUT, then routed_up: latent -> H.
    if (opt.has_routed_norm) {
        auto rn = read_f32(T("ffn_routed_norm.weight"), LAT);
        moe_out = rms_norm(moe_out, rn);
    }
    auto Wru = read_matrix(T("ffn_routed_up.weight"), H, LAT);
    auto ffn = matvec(Wru, moe_out, H, LAT);

    // shared experts at HIDDEN width, added.
    if (opt.has_shared_experts) {
        const int nsh = cfg.moe_ffn * cfg.n_shared;
        auto Wsg = read_matrix(T("ffn_gate_shexp.weight"), nsh, H);
        auto Wsu = read_matrix(T("ffn_up_shexp.weight"), nsh, H);
        auto Wsd = read_matrix(T("ffn_down_shexp.weight"), H, nsh);
        auto sg = matvec(Wsg, normed2, nsh, H);
        auto su = matvec(Wsu, normed2, nsh, H);
        std::vector<double> sa(nsh);
        for (int i = 0; i < nsh; ++i) sa[i] = situ(sg[i], su[i]);
        auto sh = matvec(Wsd, sa, H, nsh);
        for (int i = 0; i < H; ++i) ffn[i] += sh[i];
    }

    // ---- compare ----
    std::printf("router top-3 ids: %d %d %d (weights %.4f %.4f %.4f)\n",
                ids[0], ids[1], ids[2], weights[0], weights[1], weights[2]);
    double num = 0, den = 0, worst = 0; int wi = -1;
    for (int i = 0; i < H; ++i) {
        const double d = (double)exec_ffn_out[i] - ffn[i];
        num += d * d; den += ffn[i] * ffn[i];
        const double rel = std::fabs(d) / (std::fabs(ffn[i]) + 1e-6);
        if (rel > worst) { worst = rel; wi = i; }
    }
    const double rl2 = std::sqrt(num / (den + 1e-30));
    std::printf("\nMoE FFN branch: executor vs independent float64 (real IQ1_S experts)\n");
    std::printf("  relL2 = %.3e   worst_rel = %.3e @ %d (ref=%.6g exec=%.6g)\n",
                rl2, worst, wi, ffn[wi], (double)exec_ffn_out[wi]);
    std::printf("  ref RMS = %.6g\n", std::sqrt(den / H));
    // IQ1_S dispatch through top-16 experts + two latent hops + shexp is the longest
    // fp32 chain in the model; 5e-3 is the honest bar (same reasoning as the MLA
    // check, more so). A structural bug shows as relL2 ~ O(1), not near this.
    const double thresh = 5e-3;
    std::printf("\n%s\n", rl2 < thresh
        ? "PASS: MoE FFN branch wiring matches an independent reference"
        : "FAIL");
    return rl2 < thresh ? 0 : 1;
}
