// Full-layer numerical validation: layer 0 (KDA, leading-dense FFN) computed TWICE
// on the SAME real weights and the SAME input vector — once as the actual GPU
// executor (kimi_k3_forward_layer), once as an independent float64 CPU reference
// that dequantizes the same GGUF bytes by hand and transcribes the layer's math
// directly. Every kernel involved is already individually proven against float64
// (kernels/tests/kimi_k3_numeric_test.cu); what this adds is proof that the
// EXECUTOR wires them together correctly — right buffer to the right kernel, in
// the right order, with the real weight layout — which no per-kernel test can
// see. Layer 0 was chosen because it needs no MoE router/dispatch (leading_dense
// covers it) and no MLA absorbed-attention plumbing, so this is the least-moving-
// parts version of "does the whole layer match" before extending to harder layers.
//
// Building this reference caught THREE real bugs in the reference itself before it
// ever agreed with the executor — worth knowing before trusting the pattern for a
// harder layer:
//   - conv weight layout: kda_conv_step_kernel indexes `w[c*d_conv + t]` (channel
//     outer, time inner). An early draft had it backwards.
//   - silu vs sigmoid: the conv output is `acc * sigmoid(acc)`, not `sigmoid(acc)`.
//   - l2_norm_heads_kernel is a TRUE L2 norm (rsqrt(sum+eps)), not RMS
//     (rsqrt(mean+eps)) — distinct from kda_gate_out's own internal norm, which IS
//     RMS. Conflating the two is an easy, wrong-but-plausible mistake.
// And one real bug in an ASSUMPTION, not a formula:
//   - ssm_beta.weight is F32 in the real file, not Q8_0 like every other 2-D weight
//     checked here. Reinterpreting F32 bytes as fake Q8_0 blocks doesn't crash or
//     NaN — it produces plausible-looking garbage (a ~1700x-wrong RMS). Fixed by
//     reading the tensor's REAL ggml_type at runtime instead of assuming one per
//     name, mirroring what dequant_f32_by_type/k3_proj_f32 already do in production.
// And one real bug in the TEST's understanding of the architecture:
//   - the cross-layer residual bank mutates MID-LAYER: res_push (banking this
//     layer's raw input) runs between the attn-side and ffn-side res_mix calls, so
//     even on layer 0 (nothing banked at the START of the layer) the ffn-side mix
//     sees n_ckpt=1, not 0. An early draft treated both mixes as identity for layer
//     0, which is wrong for the second one. The executor gets this right by
//     construction (sequential state mutation); the reference had to catch up.
//
// All four are documented in place below, at the line each one was fixed.

#include "sparkinfer/models/kimi_k3.h"
#include "sparkinfer/models/kimi_k3_gguf_manifest.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

using namespace sparkinfer;

// ---- CPU-side Q8_0 dequant, bit-identical logic to k3_kernels.cu's proj_q8_0_kernel
struct BlockQ8_0 { uint16_t d; int8_t qs[32]; };
static double h2f(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const uint32_t exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) bits = sign;
    else if (exp == 31) bits = sign | 0x7f800000u | (man << 13);
    else bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    float f; std::memcpy(&f, &bits, 4); return (double)f;
}
// Dequantize a [N,K] Q8_0 weight (ne0=K fastest) to a flat double[N*K] row-major array.
static std::vector<double> dequant_q8_0_matrix(const GGUFTensor* t, int N, int K) {
    std::vector<double> out((size_t)N * K);
    const BlockQ8_0* blocks = (const BlockQ8_0*)t->data;
    const int bpr = K / 32;
    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < bpr; ++b) {
            const BlockQ8_0& blk = blocks[(size_t)n * bpr + b];
            const double d = h2f(blk.d);
            for (int j = 0; j < 32; ++j) out[(size_t)n * K + b * 32 + j] = d * (double)blk.qs[j];
        }
    }
    return out;
}
static std::vector<double> read_f32_vec(const GGUFTensor* t, int n) {
    std::vector<double> out(n);
    const float* p = (const float*)t->data;
    for (int i = 0; i < n; ++i) out[i] = (double)p[i];
    return out;
}
// y[N] = W[N,K] @ x[K]  (double precision, W row-major as produced above)
static std::vector<double> matvec(const std::vector<double>& W, const std::vector<double>& x,
                                  int N, int K) {
    std::vector<double> y(N, 0.0);
    for (int n = 0; n < N; ++n) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k) acc += W[(size_t)n * K + k] * x[k];
        y[n] = acc;
    }
    return y;
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1]
        : "/workspace/models_k3/UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf";

    GGUF g;
    KimiK3Config cfg;
    KimiK3LayerCoverage cov;
    if (!kimi_k3_load_partial(path, cfg, g, &cov) || !cov.layer_complete[0]) {
        std::printf("layer 0 not available\n");
        return 1;
    }
    const int H = cfg.hidden;
    const int qkv = cfg.n_q_heads * cfg.kda_head_dim;
    const int head_dim = cfg.kda_head_dim, n_head = cfg.n_q_heads;
    const double eps = cfg.rms_eps;
    std::printf("H=%d qkv=%d head_dim=%d n_head=%d dense_ffn=%d leading_dense=%d\n",
                H, qkv, head_dim, n_head, cfg.dense_ffn, cfg.leading_dense);

    auto T = [&](const char* suf) { return g.tensor((std::string("blk.0.") + suf).c_str()); };

    // Type-GENERIC matrix reader: dispatch on the tensor's REAL ggml_type rather than
    // assuming one per name. An earlier version of this test hardcoded Q8_0 for
    // ssm_beta.weight, which is actually F32 in the real file (unsloth's dynamic quant
    // does not uniformly quantize every 2-D weight — this one is small enough, and
    // apparently precision-sensitive enough, to keep at F32). Reinterpreting F32 bytes
    // as fake Q8_0 blocks doesn't crash or produce NaN — it produces plausible-looking
    // garbage, which is exactly the trap dequant_f32_by_type/k3_proj_f32 exist to
    // refuse in production code. This test now mirrors that: read the real type,
    // dispatch, never assume.
    auto read_matrix = [&](const GGUFTensor* t, int N, int K) -> std::vector<double> {
        if (t->ggml_type == 0) {
            std::vector<double> out((size_t)N * K);
            const float* p = (const float*)t->data;
            for (size_t i = 0; i < out.size(); ++i) out[i] = (double)p[i];
            return out;
        }
        if (t->ggml_type == 8) return dequant_q8_0_matrix(t, N, K);
        std::fprintf(stderr, "unhandled ggml type %d\n", t->ggml_type);
        std::exit(1);
    };

    // ---- read every layer-0 tensor the CPU reference needs, by its REAL type ----
    auto attn_norm_w = read_f32_vec(T("attn_norm.weight"), H);
    auto ffn_norm_w  = read_f32_vec(T("ffn_norm.weight"), H);
    auto Wq = read_matrix(T("attn_q.weight"), qkv, H);
    auto Wk = read_matrix(T("attn_k.weight"), qkv, H);
    auto Wv = read_matrix(T("attn_v.weight"), qkv, H);
    auto conv_q_w = read_f32_vec(T("ssm_conv1d_q.weight"), cfg.kda_conv_kernel * qkv);
    auto conv_k_w = read_f32_vec(T("ssm_conv1d_k.weight"), cfg.kda_conv_kernel * qkv);
    auto conv_v_w = read_f32_vec(T("ssm_conv1d_v.weight"), cfg.kda_conv_kernel * qkv);
    auto Wfa = read_matrix(T("ssm_f_a.weight"), head_dim, H);
    auto Wfb = read_matrix(T("ssm_f_b.weight"), qkv, head_dim);
    auto dt_bias = read_f32_vec(T("ssm_dt.bias"), qkv);
    auto ssm_a = read_f32_vec(T("ssm_a"), n_head);
    auto Wbeta = read_matrix(T("ssm_beta.weight"), n_head, H);
    auto Wg = read_matrix(T("ssm_g.weight"), qkv, H);
    auto ssm_norm_w = read_f32_vec(T("ssm_norm.weight"), head_dim);
    auto Wo = read_matrix(T("attn_output.weight"), H, qkv);
    auto Wgate = read_matrix(T("ffn_gate.weight"), cfg.dense_ffn, H);
    auto Wup = read_matrix(T("ffn_up.weight"), cfg.dense_ffn, H);
    auto Wdown = read_matrix(T("ffn_down.weight"), H, cfg.dense_ffn);

    // ---- random input, shared between CPU ref and GPU run ----
    std::mt19937 rng(20260730);
    std::normal_distribution<double> Nrm(0.0, 1.0);
    std::vector<double> x(H);
    for (auto& v : x) v = Nrm(rng);

    // ================= CPU float64 reference =================
    // Layer 0 is a checkpoint layer (0 % block_size == 0): attn_res_mix is identity
    // (n_ckpt==0), and the attention-side combine REPLACES rather than adds.
    auto rms_norm = [&](const std::vector<double>& v, const std::vector<double>& w) {
        double ss = 0; for (double e : v) ss += e * e;
        const double inv = 1.0 / std::sqrt(ss / v.size() + eps);
        std::vector<double> out(v.size());
        for (size_t i = 0; i < v.size(); ++i) out[i] = v[i] * inv * w[i];
        return out;
    };

    auto normed = rms_norm(x, attn_norm_w);
    auto q = matvec(Wq, normed, qkv, H);
    auto k = matvec(Wk, normed, qkv, H);
    auto v = matvec(Wv, normed, qkv, H);

    // causal conv, state=0 at position 0: window is (d_conv-1 zeros, x) with x at the
    // END. Weight layout verified against kda_conv_step_kernel directly: `wc = w +
    // c*d_conv` — channel is the OUTER/slower index (stride d_conv), time is INNER
    // (stride 1) for a fixed channel. State is all zero at position 0, so only the
    // wc[d_conv-1] tap (multiplying the current token) contributes.
    auto conv_step = [&](const std::vector<double>& xin, const std::vector<double>& w) {
        std::vector<double> out(qkv);
        for (int c = 0; c < qkv; ++c) {
            const double acc = xin[c] * w[c * cfg.kda_conv_kernel + (cfg.kda_conv_kernel - 1)];
            // silu(x) = x * sigmoid(x), AFTER the conv — not sigmoid(x) alone.
            out[c] = acc * (1.0 / (1.0 + std::exp(-acc)));
        }
        return out;
    };
    auto conv_q = conv_step(q, conv_q_w);
    auto conv_k = conv_step(k, conv_k_w);
    auto conv_v_ = conv_step(v, conv_v_w);   // v is NOT L2-normalised

    // TRUE L2 norm (sum, not mean) — verified against l2_norm_heads_kernel directly:
    // `inv = scale * rsqrtf(ss + eps)`, no /head_dim. Distinct from kda_gate_out's own
    // internal norm below, which IS an RMS norm (divides by head_dim) — two different
    // kernels, two different normalizations, and conflating them would be a real bug.
    auto l2_norm_heads = [&](std::vector<double> t, double scale) {
        for (int h = 0; h < n_head; ++h) {
            double ss = 0;
            for (int d = 0; d < head_dim; ++d) ss += t[h*head_dim+d]*t[h*head_dim+d];
            const double inv = scale / std::sqrt(ss + eps);
            for (int d = 0; d < head_dim; ++d) t[h*head_dim+d] *= inv;
        }
        return t;
    };
    auto l2_q = l2_norm_heads(conv_q, 1.0 / std::sqrt((double)head_dim));
    auto l2_k = l2_norm_heads(conv_k, 1.0);

    auto f_a = matvec(Wfa, normed, head_dim, H);
    auto g_raw = matvec(Wfb, f_a, qkv, head_dim);
    for (int i = 0; i < qkv; ++i) g_raw[i] += dt_bias[i];

    // decay_gate: lb * sigmoid(-(A*g_raw)), A per-head broadcast across head_dim
    std::vector<double> decay_g(qkv);
    for (int h = 0; h < n_head; ++h)
        for (int d = 0; d < head_dim; ++d) {
            const double v_ = ssm_a[h] * g_raw[h*head_dim+d];
            decay_g[h*head_dim+d] = cfg.kda_gate_lower_bound / (1.0 + std::exp(v_));
        }

    auto beta = matvec(Wbeta, normed, n_head, H);

    // gated delta rule, state=0 at position 0
    std::vector<double> delta_out(qkv);
    for (int h = 0; h < n_head; ++h) {
        std::vector<double> S(head_dim * head_dim, 0.0);   // S[i][j], i fastest
        const double* qh = &l2_q[h*head_dim]; const double* kh = &l2_k[h*head_dim];
        const double* vh = &conv_v_[h*head_dim]; const double* gh = &decay_g[h*head_dim];
        const double b = beta[h];
        std::vector<double> sk(head_dim, 0.0);
        for (int j = 0; j < head_dim; ++j) {
            const double decay = std::exp(gh[j]);
            double s = 0;
            for (int i = 0; i < head_dim; ++i) {
                S[j*head_dim+i] *= decay;   // per column j (state stored [j][i] here, i fastest within column)
                s += S[j*head_dim+i] * kh[i];
            }
            sk[j] = s;
        }
        std::vector<double> d(head_dim);
        for (int j = 0; j < head_dim; ++j) d[j] = b * (vh[j] - sk[j]);
        std::vector<double> o(head_dim, 0.0);
        for (int j = 0; j < head_dim; ++j) {
            for (int i = 0; i < head_dim; ++i) {
                S[j*head_dim+i] += kh[i] * d[j];
                o[j] += S[j*head_dim+i] * qh[i];
            }
        }
        // S[j*head_dim+i] represents S[i][j] in the kernel doc's notation — verified
        // directly against kda_decode_step_kernel's own memory layout (thread j reads
        ///writes S[j*head_dim+i]), not just the abstract formula, so this is a
        // line-for-line transcription rather than an independently-derived rearrangement.
        for (int j = 0; j < head_dim; ++j) delta_out[h*head_dim+j] = o[j];
    }

    auto g_proj = matvec(Wg, normed, qkv, H);
    std::vector<double> gate_out(qkv);
    for (int h = 0; h < n_head; ++h) {
        double ss = 0;
        for (int d = 0; d < head_dim; ++d) ss += delta_out[h*head_dim+d]*delta_out[h*head_dim+d];
        const double inv = 1.0 / std::sqrt(ss / head_dim + eps);
        for (int d = 0; d < head_dim; ++d) {
            const double normed_v = delta_out[h*head_dim+d] * inv * ssm_norm_w[d];
            const double gate = 1.0 / (1.0 + std::exp(-g_proj[h*head_dim+d]));
            gate_out[h*head_dim+d] = normed_v * gate;
        }
    }

    auto attn_out = matvec(Wo, gate_out, H, qkv);
    // combine: layer 0 is banked (checkpoint layer) -> REPLACE
    std::vector<double> prefix_sum = attn_out;

    auto rms = [](const std::vector<double>& v) {
        double s = 0; for (double e : v) s += e * e; return std::sqrt(s / v.size());
    };
    std::printf("[cpu ref] normed rms=%.6g  q[0..3]=%.6g,%.6g,%.6g,%.6g\n",
                rms(normed), q[0], q[1], q[2], q[3]);
    std::printf("[cpu ref] conv_q(pre-l2) rms=%.6g  l2_q rms=%.6g  conv_v(unnorm) rms=%.6g\n",
                rms(conv_q), rms(l2_q), rms(conv_v_));
    std::printf("[cpu ref] decay_g rms=%.6g  beta rms=%.6g\n", rms(decay_g), rms(beta));
    std::printf("[cpu ref] delta_out rms=%.6g  gate_out rms=%.6g\n", rms(delta_out), rms(gate_out));
    std::printf("[cpu ref] attn_out (pre-combine) rms=%.6g  first4=%.6g,%.6g,%.6g,%.6g\n",
                rms(attn_out), attn_out[0], attn_out[1], attn_out[2], attn_out[3]);

    // Layer 0's OWN raw input got banked BETWEEN the two res_mix calls (res_push
    // runs right after the attn-side mix, before build_norm/attention — see the
    // reference's per-layer loop). So the FFN-side mix is NOT identity even on
    // layer 0: it sees n_ckpt=1, the just-banked checkpoint (= x, this layer's raw
    // input). An earlier version of this test assumed n_ckpt stayed 0 for the whole
    // of layer 0 — wrong, the bank mutates mid-layer. The executor already gets this
    // right by construction (it processes state sequentially: mix, then push, then
    // attention, then the second mix); this is the CPU reference catching up.
    auto ffn_res_score_w = read_f32_vec(T("ffn_res_score.weight"), H);
    std::vector<double> ffn_mix_in;
    {
        auto rms_of = [&](const std::vector<double>& v) {
            double s = 0; for (double e : v) s += e * e;
            return 1.0 / std::sqrt(s / v.size() + eps);
        };
        double score_bank = 0, score_cur = 0;
        const double inv_bank = rms_of(x), inv_cur = rms_of(prefix_sum);
        for (int d = 0; d < H; ++d) {
            score_bank += (x[d] * inv_bank) * ffn_res_score_w[d];
            score_cur  += (prefix_sum[d] * inv_cur) * ffn_res_score_w[d];
        }
        const double m = std::max(score_bank, score_cur);
        const double e0 = std::exp(score_bank - m), e1 = std::exp(score_cur - m);
        const double p0 = e0 / (e0 + e1), p1 = e1 / (e0 + e1);
        ffn_mix_in.resize(H);
        for (int d = 0; d < H; ++d) ffn_mix_in[d] = p0 * x[d] + p1 * prefix_sum[d];
        std::printf("[cpu ref] ffn-side mix: p_bank=%.6g p_cur=%.6g\n", p0, p1);
    }

    auto normed2 = rms_norm(ffn_mix_in, ffn_norm_w);
    std::printf("[cpu ref] normed2 rms=%.6g\n", rms(normed2));
    auto gate_d = matvec(Wgate, normed2, cfg.dense_ffn, H);
    auto up_d = matvec(Wup, normed2, cfg.dense_ffn, H);
    {
        double gs=0,us=0; for(double e:gate_d) gs+=e*e; for(double e:up_d) us+=e*e;
        std::printf("[cpu ref] gate_d rms=%.6g  up_d rms=%.6g\n",
                    std::sqrt(gs/cfg.dense_ffn), std::sqrt(us/cfg.dense_ffn));
    }
    std::vector<double> situ_out(cfg.dense_ffn);
    const double beta_s = cfg.situ_beta, lb_s = cfg.situ_linear_beta;
    for (int i = 0; i < cfg.dense_ffn; ++i) {
        const double gg = gate_d[i], uu = up_d[i];
        const double a = beta_s * std::tanh(gg / beta_s) * (1.0 / (1.0 + std::exp(-gg)));
        const double ub = lb_s > 0 ? (lb_s * std::tanh(uu / lb_s)) : uu;
        situ_out[i] = a * ub;
    }
    {
        double ss=0; for(double e:situ_out) ss+=e*e;
        std::printf("[cpu ref] situ_out rms=%.6g\n", std::sqrt(ss/cfg.dense_ffn));
    }
    auto ffn_out = matvec(Wdown, situ_out, H, cfg.dense_ffn);
    std::printf("[cpu ref] ffn_out rms=%.6g\n", rms(ffn_out));
    std::vector<double> ref_out(H);
    for (int i = 0; i < H; ++i) ref_out[i] = prefix_sum[i] + ffn_out[i];   // FFN side always adds

    // ================= GPU executor =================
    K3PlanOptions opt;   // layer 0 is KDA; MLA-only opts are irrelevant here
    KimiK3Weights w;
    if (!kimi_k3_load_weights(g, cfg, opt, w, 0, 0)) { std::printf("load failed\n"); return 1; }
    KimiK3RuntimeState state;
    if (!kimi_k3_alloc_state(cfg, 64, state)) { std::printf("state alloc failed\n"); return 1; }
    KimiK3Forward fwd;
    fwd.cfg = &cfg; fwd.w = &w; fwd.state = &state; fwd.opt = opt; fwd.stream = 0;
    if (!kimi_k3_forward_alloc_scratch(cfg, fwd)) { std::printf("scratch alloc failed\n"); return 1; }
    fwd.debug = [](const char* tag, int layer, const float* dev_ptr, int64_t n) {
        std::vector<float> h(n < 8 ? n : 8);
        cudaMemcpy(h.data(), dev_ptr, h.size() * sizeof(float), cudaMemcpyDeviceToHost);
        double ss = 0; std::vector<float> full(n);
        cudaMemcpy(full.data(), dev_ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
        for (float v : full) ss += (double)v * v;
        std::printf("[gpu %s L%d] rms=%.6g  first4=%.6g,%.6g,%.6g,%.6g\n",
                    tag, layer, std::sqrt(ss / n), (double)h[0], (double)h[1], (double)h[2], (double)h[3]);
    };

    std::vector<float> hx(H); for (int i = 0; i < H; ++i) hx[i] = (float)x[i];
    float *d_in, *d_out;
    cudaMalloc(&d_in, H * sizeof(float));
    cudaMalloc(&d_out, H * sizeof(float));
    cudaMemcpy(d_in, hx.data(), H * sizeof(float), cudaMemcpyHostToDevice);

    if (!kimi_k3_forward_layer(fwd, 0, d_in, d_out)) { std::printf("forward_layer failed\n"); return 1; }
    cudaDeviceSynchronize();
    std::vector<float> gpu_out(H);
    cudaMemcpy(gpu_out.data(), d_out, H * sizeof(float), cudaMemcpyDeviceToHost);

    // ================= compare =================
    double num = 0, den = 0, worst = 0; int worst_i = -1;
    for (int i = 0; i < H; ++i) {
        const double d = (double)gpu_out[i] - ref_out[i];
        num += d * d; den += ref_out[i] * ref_out[i];
        const double rel = std::fabs(d) / (std::fabs(ref_out[i]) + 1e-8);
        if (rel > worst) { worst = rel; worst_i = i; }
    }
    const double rl2 = std::sqrt(num / (den + 1e-30));
    std::printf("\nlayer 0 FULL LAYER: GPU executor vs independent float64 CPU reference\n");
    std::printf("  relL2 = %.3e   worst_rel = %.3e @ %d (ref=%.6g gpu=%.6g)\n",
                rl2, worst, worst_i, ref_out[worst_i], (double)gpu_out[worst_i]);
    std::printf("  ref RMS = %.6g   gpu RMS = %.6g\n",
                std::sqrt(den / H),
                std::sqrt(std::inner_product(gpu_out.begin(), gpu_out.end(), gpu_out.begin(), 0.0) / H));
    const bool ok = rl2 < 1e-3;   // f32 GPU vs f64 CPU through ~10 sequential matmuls
    std::printf("\n%s\n", ok ? "PASS: executor matches an independent reference" : "FAIL");
    return ok ? 0 : 1;
}
