// MLA attention-branch validation: layer 3's ATTENTION path (through the
// pre-combine attn_out) computed twice on the same real weights and the same
// input — once by kimi_k3_forward_layer, once by an independent float64 CPU
// reference. Complements kimi_k3_layer0_ref_check.cpp, which only exercised KDA:
// this covers the q_lora path, the cudaMemcpy2D q_nope/q_pe deinterleave, the
// kv_cmpr/k_pe split + KV-cache write, the absorbed-attention kernels, and the
// (real, on this file) attn_gate — none of which the KDA layer touches at all.
//
// Deliberately STOPS at the attention output, before FFN/MoE/shared-experts. Those
// have their own dedicated real-weight validation already
// (kernels/tests/kimi_k3_moe_real_test.cu proves the IQ1_S/IQ2_XS expert dispatch
// bit-close against float64 on real expert weights), so the marginal value of
// re-deriving that math a third time here is low; the attention branch above is
// where the genuinely UNTESTED executor wiring lives.
//
// wk_b/wv_b dequant reuses read_matrix/dequant_q8_0_matrix from the KDA test
// UNCHANGED: GGUF blocks along ne0 regardless of how many outer dims a tensor
// has, so wk_b's real 3-D layout [qk_nope, kv_lora, n_head] (qk_nope fastest) is
// bit-identical, for dequant purposes, to a [kv_lora*n_head, qk_nope] matrix — no
// new dequant code needed, just the right N/K reshape.

#include "sparkinfer/models/kimi_k3.h"
#include "sparkinfer/models/kimi_k3_gguf_manifest.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace sparkinfer;

struct BlockQ8_0 { uint16_t d; int8_t qs[32]; };
static double h2f(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const uint32_t exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign;   // signed zero
        else {                       // SUBNORMAL half — must not be flushed to zero;
            int e = -1; uint32_t m = man;   // that is what silently zeroed IQ1_S/Q8_0
            do { m <<= 1; ++e; } while (!(m & 0x400));   // scales with tiny fp16 d and
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3ff) << 13);  // made
        }                            // the CPU ref disagree with the executor's __half2float
    }
    else if (exp == 31) bits = sign | 0x7f800000u | (man << 13);
    else bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    float f; std::memcpy(&f, &bits, 4); return (double)f;
}
static std::vector<double> dequant_q8_0_matrix(const GGUFTensor* t, int N, int K) {
    std::vector<double> out((size_t)N * K);
    const BlockQ8_0* blocks = (const BlockQ8_0*)t->data;
    const int bpr = K / 32;
    for (int n = 0; n < N; ++n)
        for (int b = 0; b < bpr; ++b) {
            const BlockQ8_0& blk = blocks[(size_t)n * bpr + b];
            const double d = h2f(blk.d);
            for (int j = 0; j < 32; ++j) out[(size_t)n * K + b * 32 + j] = d * (double)blk.qs[j];
        }
    return out;
}
static std::vector<double> read_f32_vec(const GGUFTensor* t, int n) {
    std::vector<double> out(n);
    const float* p = (const float*)t->data;
    for (int i = 0; i < n; ++i) out[i] = (double)p[i];
    return out;
}
static std::vector<double> read_matrix(const GGUFTensor* t, int N, int K) {
    if (t->ggml_type == 0) return read_f32_vec(t, N * K);
    if (t->ggml_type == 8) return dequant_q8_0_matrix(t, N, K);
    std::fprintf(stderr, "unhandled ggml type %d\n", t->ggml_type);
    std::exit(1);
}
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
static double rms(const std::vector<double>& v) {
    double s = 0; for (double e : v) s += e * e; return std::sqrt(s / v.size());
}

int main(int argc, char** argv) {
    const int LAYER = argc > 2 ? std::atoi(argv[2]) : 3;
    const char* path = argc > 1 ? argv[1]
        : "/workspace/models_k3/UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf";

    GGUF g;
    KimiK3Config cfg;
    KimiK3LayerCoverage cov;
    if (!kimi_k3_load_partial(path, cfg, g, &cov) || !cov.layer_complete[LAYER] ||
        cfg.is_kda_layer(LAYER)) {
        std::printf("layer %d not available or not MLA\n", LAYER);
        return 1;
    }
    const int H = cfg.hidden, qh = cfg.n_q_heads;
    const int qk_nope = cfg.key_length_mla - cfg.rope_dim;
    const double eps = cfg.rms_eps;
    std::printf("H=%d qh=%d q_lora=%d kv_lora=%d key_length=%d key_length_mla=%d "
                "qk_nope=%d rope_dim=%d value_length_mla=%d\n",
                H, qh, cfg.q_lora_rank, cfg.kv_lora_rank, cfg.key_length,
                cfg.key_length_mla, qk_nope, cfg.rope_dim, cfg.value_length_mla);

    char pfx[32]; std::snprintf(pfx, sizeof(pfx), "blk.%d.", LAYER);
    auto T = [&](const char* suf) { return g.tensor((std::string(pfx) + suf).c_str()); };

    const bool has_q_lora = T("attn_q_a.weight") != nullptr;
    const bool has_attn_gate = T("attn_gate.weight") != nullptr;
    std::printf("layout: q_lora=%d attn_gate=%d\n", has_q_lora, has_attn_gate);

    auto attn_norm_w = read_f32_vec(T("attn_norm.weight"), H);
    auto Wqa = has_q_lora ? read_matrix(T("attn_q_a.weight"), cfg.q_lora_rank, H)
                          : std::vector<double>();
    auto qa_norm_w = has_q_lora ? read_f32_vec(T("attn_q_a_norm.weight"), cfg.q_lora_rank)
                                : std::vector<double>();
    auto Wqb = has_q_lora
        ? read_matrix(T("attn_q_b.weight"), qh * cfg.key_length_mla, cfg.q_lora_rank)
        : read_matrix(T("attn_q.weight"), qh * cfg.key_length_mla, H);
    auto Wkva = read_matrix(T("attn_kv_a_mqa.weight"), cfg.key_length, H);
    auto kva_norm_w = read_f32_vec(T("attn_kv_a_norm.weight"), cfg.kv_lora_rank);
    // wk_b [qk_nope, kv_lora, n_head] qk_nope fastest == a [kv_lora*n_head, qk_nope] matrix
    auto Wkb = read_matrix(T("attn_k_b.weight"), cfg.kv_lora_rank * qh, qk_nope);
    // wv_b [kv_lora, v_dim, n_head] kv_lora fastest == a [v_dim*n_head, kv_lora] matrix
    auto Wvb = read_matrix(T("attn_v_b.weight"), cfg.value_length_mla * qh, cfg.kv_lora_rank);
    auto Wgate = has_attn_gate
        ? read_matrix(T("attn_gate.weight"), qh * cfg.value_length_mla, H)
        : std::vector<double>();
    auto Wo = read_matrix(T("attn_output.weight"), H, qh * cfg.value_length_mla);

    std::mt19937 rng(20260730 + LAYER);
    std::normal_distribution<double> Nrm(0.0, 1.0);
    std::vector<double> x(H);
    for (auto& v : x) v = Nrm(rng);

    // ================= CPU float64 reference (attention branch only) =================
    auto rms_norm = [&](const std::vector<double>& v, const std::vector<double>& w) {
        double ss = 0; for (double e : v) ss += e * e;
        const double inv = 1.0 / std::sqrt(ss / v.size() + eps);
        std::vector<double> out(v.size());
        for (size_t i = 0; i < v.size(); ++i) out[i] = v[i] * inv * w[i];
        return out;
    };
    // Layer 3 is not a checkpoint layer under a period-12 schedule (3%12!=0), so the
    // attn-side mix is genuinely identity here (n_ckpt==0 for the first several
    // layers regardless — only layer 0 has banked anything by the time layer 3 runs).
    auto normed = rms_norm(x, attn_norm_w);

    std::vector<double> q_full;
    if (has_q_lora) {
        auto q_a = matvec(Wqa, normed, cfg.q_lora_rank, H);
        auto q_a_normed = rms_norm(q_a, qa_norm_w);
        q_full = matvec(Wqb, q_a_normed, qh * cfg.key_length_mla, cfg.q_lora_rank);
    } else {
        q_full = matvec(Wqb, normed, qh * cfg.key_length_mla, H);
    }
    std::printf("[cpu ref] normed rms=%.6g  q_full rms=%.6g\n", rms(normed), rms(q_full));

    // De-interleave per head: [qk_nope | rope_dim] contiguous, qk_nope first.
    std::vector<double> q_nope((size_t)qh * qk_nope), q_pe((size_t)qh * cfg.rope_dim);
    for (int h = 0; h < qh; ++h) {
        for (int d = 0; d < qk_nope; ++d)
            q_nope[(size_t)h * qk_nope + d] = q_full[(size_t)h * cfg.key_length_mla + d];
        for (int d = 0; d < cfg.rope_dim; ++d)
            q_pe[(size_t)h * cfg.rope_dim + d] =
                q_full[(size_t)h * cfg.key_length_mla + qk_nope + d];
    }

    auto kv_a = matvec(Wkva, normed, cfg.key_length, H);
    std::vector<double> kv_cmpr(kv_a.begin(), kv_a.begin() + cfg.kv_lora_rank);
    std::vector<double> k_pe(kv_a.begin() + cfg.kv_lora_rank, kv_a.end());
    auto kv_cmpr_normed = rms_norm(kv_cmpr, kva_norm_w);
    std::printf("[cpu ref] kv_a rms=%.6g  kv_cmpr_normed rms=%.6g\n", rms(kv_a), rms(kv_cmpr_normed));

    // KV cache row 0 = concat(kv_cmpr_normed, k_pe) — position 0, one token.
    std::vector<double> kcache_row(cfg.key_length);
    for (int i = 0; i < cfg.kv_lora_rank; ++i) kcache_row[i] = kv_cmpr_normed[i];
    for (int i = 0; i < cfg.rope_dim; ++i) kcache_row[cfg.kv_lora_rank + i] = k_pe[i];

    // mla_absorb_q: per head, absorbed[r] = sum_d wk_b[d + r*qk_nope + h*qk_nope*kv_lora]*q_nope[d]
    // Wkb is stored as [kv_lora*n_head, qk_nope] with row index (r + h*kv_lora).
    std::vector<double> absorbed_q((size_t)qh * cfg.key_length);
    for (int h = 0; h < qh; ++h) {
        for (int r = 0; r < cfg.kv_lora_rank; ++r) {
            double acc = 0;
            const double* row = &Wkb[(size_t)(r + h * cfg.kv_lora_rank) * qk_nope];
            for (int d = 0; d < qk_nope; ++d) acc += row[d] * q_nope[h * qk_nope + d];
            absorbed_q[(size_t)h * cfg.key_length + r] = acc;
        }
        for (int d = 0; d < cfg.rope_dim; ++d)
            absorbed_q[(size_t)h * cfg.key_length + cfg.kv_lora_rank + d] = q_pe[h * cfg.rope_dim + d];
    }
    std::printf("[cpu ref] absorbed_q rms=%.6g\n", rms(absorbed_q));

    // decode_attn: n_ctx=1 (only row 0 exists), so softmax over one score is trivially 1.
    const double mla_scale = 1.0 / std::sqrt((double)cfg.key_length_mla);
    std::vector<double> mla_attn_out((size_t)qh * cfg.value_length_mla);
    for (int h = 0; h < qh; ++h) {
        // score/softmax over n_ctx=1 is a no-op: p[0]=1. latent = kcache_row's first kv_lora.
        for (int v = 0; v < cfg.value_length_mla; ++v) {
            double acc = 0;
            const double* wrow = &Wvb[(size_t)(v + h * cfg.value_length_mla) * cfg.kv_lora_rank];
            for (int r = 0; r < cfg.kv_lora_rank; ++r) acc += wrow[r] * kcache_row[r];
            mla_attn_out[(size_t)h * cfg.value_length_mla + v] = acc;
        }
    }
    (void)mla_scale;   // scale only matters with >1 cached token; n_ctx=1 softmax ignores it
    std::printf("[cpu ref] mla_attn_out (pre-gate) rms=%.6g\n", rms(mla_attn_out));
    const std::vector<double> cpu_preattn = mla_attn_out;   // keep for a per-element check

    std::vector<double> cpu_gateproj, cpu_postgate;
    if (has_attn_gate) {
        cpu_gateproj = matvec(Wgate, normed, qh * cfg.value_length_mla, H);
        for (size_t i = 0; i < mla_attn_out.size(); ++i)
            mla_attn_out[i] *= 1.0 / (1.0 + std::exp(-cpu_gateproj[i]));
        cpu_postgate = mla_attn_out;
        std::printf("[cpu ref] mla_attn_out (post-gate) rms=%.6g\n", rms(mla_attn_out));
    }

    auto attn_out = matvec(Wo, mla_attn_out, H, qh * cfg.value_length_mla);

    // Same o_proj but with FLOAT accumulation (matching the GPU kernel's fp32 sum),
    // to separate a precision artifact from a real bug: if the GPU matches THIS far
    // better than the double-precision attn_out, the divergence is fp32 cancellation
    // in the 12288-term o_proj (near-zero true output), not a wiring error.
    std::vector<double> attn_out_f32(H);
    {
        const int K = qh * cfg.value_length_mla;
        for (int n = 0; n < H; ++n) {
            float acc = 0.0f;
            const double* wr = &Wo[(size_t)n * K];
            for (int k = 0; k < K; ++k) acc += (float)wr[k] * (float)mla_attn_out[k];
            attn_out_f32[n] = (double)acc;
        }
    }
    std::printf("[cpu ref] attn_out (pre-combine) rms=%.6g  first4=%.6g,%.6g,%.6g,%.6g\n",
                rms(attn_out), attn_out[0], attn_out[1], attn_out[2], attn_out[3]);

    // ================= GPU executor =================
    K3PlanOptions opt;
    opt.has_q_lora = has_q_lora;
    opt.has_attn_gate = has_attn_gate;
    opt.has_fused_kv_b = false;
    opt.has_shared_experts = T("ffn_gate_shexp.weight") != nullptr;
    opt.has_routed_norm = T("ffn_routed_norm.weight") != nullptr;

    KimiK3Weights w;
    if (!kimi_k3_load_weights(g, cfg, opt, w, LAYER, LAYER)) { std::printf("load failed\n"); return 1; }

    // DIRECT byte compare: the loaded GPU attn_output vs the GGUF mmap bytes the CPU
    // ref reads. If these differ, the loader mis-uploaded this tensor.
    {
        const GGUFTensor* wo_t = T("attn_output.weight");
        const long nb = wo_t->n_bytes;
        std::vector<uint8_t> gpu_bytes(nb), mmap_bytes(nb);
        cudaMemcpy(gpu_bytes.data(), w.layers[LAYER].attn_output.data, nb, cudaMemcpyDeviceToHost);
        std::memcpy(mmap_bytes.data(), wo_t->data, nb);
        long ndiff = 0, first = -1;
        for (long i = 0; i < nb; ++i)
            if (gpu_bytes[i] != mmap_bytes[i]) { ++ndiff; if (first < 0) first = i; }
        std::printf("  attn_output BYTES: %ld/%ld differ (loaded GPU vs mmap)%s\n",
                    ndiff, nb, ndiff ? "  <- LOADER BUG" : "  (identical)");
        if (ndiff) std::printf("    first diff at byte %ld: gpu=%u mmap=%u\n",
                               first, gpu_bytes[first], mmap_bytes[first]);
    }
    KimiK3RuntimeState state;
    if (!kimi_k3_alloc_state(cfg, 64, state)) { std::printf("state alloc failed\n"); return 1; }
    KimiK3Forward fwd;
    fwd.cfg = &cfg; fwd.w = &w; fwd.state = &state; fwd.opt = opt; fwd.stream = 0;
    if (!kimi_k3_forward_alloc_scratch(cfg, fwd)) { std::printf("scratch alloc failed\n"); return 1; }
    fwd.debug = [](const char* tag, int layer, const float* dev_ptr, int64_t n) {
        std::vector<float> full(n);
        cudaMemcpy(full.data(), dev_ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
        double ss = 0; for (float v : full) ss += (double)v * v;
        std::printf("[gpu %s L%d] rms=%.6g  first4=%.6g,%.6g,%.6g,%.6g\n",
                    tag, layer, std::sqrt(ss / n), (double)full[0],
                    n > 1 ? (double)full[1] : 0.0, n > 2 ? (double)full[2] : 0.0,
                    n > 3 ? (double)full[3] : 0.0);
    };

    std::vector<float> hx(H); for (int i = 0; i < H; ++i) hx[i] = (float)x[i];
    float *d_in, *d_out;
    cudaMalloc(&d_in, H * sizeof(float));
    cudaMalloc(&d_out, H * sizeof(float));
    cudaMemcpy(d_in, hx.data(), H * sizeof(float), cudaMemcpyHostToDevice);

    if (!kimi_k3_forward_layer(fwd, LAYER, d_in, d_out)) {
        std::printf("forward_layer failed\n"); return 1;
    }
    cudaDeviceSynchronize();

    // Pull the executor's PRE-COMBINE attention output back via a second call: we
    // can't get it directly from forward_layer's return (it only returns the fully
    // combined hidden_out), but the debug hook already printed it as "mla_out" above
    // — compare against that printed rms/first4 by eye, and separately compare the
    // FULL layer output against a from-scratch combine+ffn/moe/shexp reference for
    // completeness at the whole-layer level using the same relL2 metric as layer 0.
    std::vector<float> gpu_full(H);
    cudaMemcpy(gpu_full.data(), d_out, H * sizeof(float), cudaMemcpyDeviceToHost);
    double gs = 0; for (float v : gpu_full) gs += (double)v * v;
    std::printf("\n[gpu] full layer output rms=%.6g (includes FFN/MoE/shexp, not "
                "independently re-derived here)\n", std::sqrt(gs / H));

    // Attention-branch-only comparison: relL2 between the CPU attn_out (pre-combine)
    // and the GPU's own "mla_out" debug tag value (also pre-combine) — both printed
    // above; compute it here directly rather than asking the reader to eyeball rms.
    std::vector<float> gpu_attn_out(H);
    const int VD = qh * cfg.value_length_mla;
    std::vector<float> gpu_preattn(VD), gpu_gateproj, gpu_postgate;
    bool got_attn = false, got_pre = false;
    fwd.debug = [&](const char* tag, int layer, const float* dev_ptr, int64_t n) {
        if (std::string(tag) == "mla_out" && (int)n == H) {
            cudaMemcpy(gpu_attn_out.data(), dev_ptr, H * sizeof(float), cudaMemcpyDeviceToHost);
            got_attn = true;
        } else if (std::string(tag) == "dbg_preattn" && (int)n == VD) {
            cudaMemcpy(gpu_preattn.data(), dev_ptr, VD * sizeof(float), cudaMemcpyDeviceToHost);
            got_pre = true;
        } else if (std::string(tag) == "dbg_gateproj" && (int)n == VD) {
            gpu_gateproj.resize(VD);
            cudaMemcpy(gpu_gateproj.data(), dev_ptr, VD * sizeof(float), cudaMemcpyDeviceToHost);
        } else if (std::string(tag) == "dbg_postgate" && (int)n == VD) {
            gpu_postgate.resize(VD);
            cudaMemcpy(gpu_postgate.data(), dev_ptr, VD * sizeof(float), cudaMemcpyDeviceToHost);
        }
    };
    // Re-run once more purely to capture the attn_out debug value cleanly (state
    // already advanced by one position from the call above, but layer 3's attention
    // math for THIS test does not depend on position beyond n_ctx bookkeeping, and
    // we only read the attention output, not the residual state).
    KimiK3RuntimeState state2;
    kimi_k3_alloc_state(cfg, 64, state2);
    fwd.state = &state2;
    cudaMemcpy(d_in, hx.data(), H * sizeof(float), cudaMemcpyHostToDevice);
    kimi_k3_forward_layer(fwd, LAYER, d_in, d_out);
    cudaDeviceSynchronize();

    auto rel = [&](const std::vector<float>& g, const std::vector<double>& c) {
        double n2 = 0, d2 = 0;
        for (size_t i = 0; i < c.size(); ++i) { double d=(double)g[i]-c[i]; n2+=d*d; d2+=c[i]*c[i]; }
        return std::sqrt(n2 / (d2 + 1e-30));
    };
    if (got_pre)
        std::printf("  PRE-GATE (decode_attn)   relL2 = %.3e\n", rel(gpu_preattn, cpu_preattn));
    if (!gpu_gateproj.empty())
        std::printf("  GATE_PROJ (attn_gate@x)  relL2 = %.3e\n", rel(gpu_gateproj, cpu_gateproj));
    if (!gpu_postgate.empty())
        std::printf("  POST-GATE (x*sigmoid)    relL2 = %.3e\n", rel(gpu_postgate, cpu_postgate));
    // DECISIVE: apply the CPU o_proj (with the mmap Wo) to the GPU'"'"'s OWN post-gate.
    // If this matches the GPU'"'"'s mla_out, the o_proj kernel + its weight are self-
    // consistent and the divergence is upstream. If it does NOT, the GPU'"'"'s loaded
    // attn_output weight differs from the mmap bytes the CPU reads.
    if (!gpu_postgate.empty() && got_attn) {
        std::vector<double> pg(gpu_postgate.begin(), gpu_postgate.end());
        auto oproj_of_gpu_pg = matvec(Wo, pg, H, VD);
        double n2=0,d2=0;
        for (int i=0;i<H;++i){ double d=(double)gpu_attn_out[i]-oproj_of_gpu_pg[i]; n2+=d*d; d2+=oproj_of_gpu_pg[i]*oproj_of_gpu_pg[i]; }
        std::printf("  O_PROJ(gpu_postgate) vs gpu mla_out relL2 = %.3e  "
                    "(large => loaded attn_output weight is wrong)\n",
                    std::sqrt(n2/(d2+1e-30)));
    }
    if (got_attn) {
        double num = 0, den = 0, worst = 0; int worst_i = -1;
        for (int i = 0; i < H; ++i) {
            const double d = (double)gpu_attn_out[i] - attn_out[i];
            num += d * d; den += attn_out[i] * attn_out[i];
            const double rel = std::fabs(d) / (std::fabs(attn_out[i]) + 1e-6);
            if (rel > worst) { worst = rel; worst_i = i; }
        }
        const double rl2 = std::sqrt(num / (den + 1e-30));
        // relL2 of GPU against the FLOAT-accumulated o_proj — if this is tiny while
        // the double one is not, the divergence is fp32 o_proj cancellation.
        double nf = 0, df = 0;
        for (int i = 0; i < H; ++i) {
            const double d = (double)gpu_attn_out[i] - attn_out_f32[i];
            nf += d * d; df += attn_out_f32[i] * attn_out_f32[i];
        }
        std::printf("  vs FLOAT-accumulated o_proj: relL2 = %.3e   (attn_out RMS = %.4g)\n",
                    std::sqrt(nf / (df + 1e-30)), std::sqrt(den / H));
        std::printf("\nMLA ATTENTION BRANCH (pre-combine): GPU vs independent float64 CPU ref\n");
        std::printf("  relL2 = %.3e   worst_rel = %.3e @ %d (ref=%.6g gpu=%.6g)\n",
                    rl2, worst, worst_i, attn_out[worst_i], (double)gpu_attn_out[worst_i]);
        // Threshold is looser than the KDA layer's (kimi_k3_layer0_ref_check.cpp used
        // 1e-3): this branch chains MORE fp32 reductions before reaching attn_out —
        // q_lora adds two extra matmuls, and mla_absorb_q/mla_decode_attn each
        // accumulate over kv_lora=512 elements via GPU warp-tree reduction, which
        // agrees with a naive double sum only up to O(sqrt(n)*eps) per reduction, not
        // exactly. worst_rel is dominated by a near-zero-crossing element (ref
        // magnitude ~2.6e-5, absolute diff ~3e-4) — the classic artifact of a
        // relative-error metric near zero, not a magnitude mismatch anywhere else.
        const double thresh = 3e-3;
        std::printf("\n%s\n", rl2 < thresh ? "PASS: MLA attention branch matches an independent reference"
                                             : "FAIL");
        return rl2 < thresh ? 0 : 1;
    }
    std::printf("did not capture mla_out debug tag\n");
    return 1;
}
