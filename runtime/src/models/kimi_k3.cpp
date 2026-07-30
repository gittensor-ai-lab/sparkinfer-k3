#include "sparkinfer/models/kimi_k3.h"

#include "sparkinfer/kernels/kimi_k3.h"
#include "sparkinfer/models/kimi_k3_gguf_manifest.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <utility>

namespace sparkinfer {

namespace k3k = sparkinfer::kernels::k3;

namespace {

std::string blk(int i, const char* suffix) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "blk.%d.%s", i, suffix);
    return std::string(buf);
}

// Upload one GGUF tensor's raw bytes to the device, native quant format preserved.
// This is the WHOLE loader primitive — every K3 weight is read natively at forward
// time (k3_proj_f32 / the MoE dispatch kernels decode Q8_0 / IQ1_S / IQ2_XS / F32
// directly), so there is no equivalent of a load-time dequantize-to-bf16 pass.
bool upload_raw(const GGUF& g, const std::string& name, KimiK3Weights& w,
                KimiK3Tensor& out, bool required) {
    const GGUFTensor* t = g.tensor(name);
    if (!t) {
        if (required)
            std::fprintf(stderr, "[k3] missing required tensor: %s\n", name.c_str());
        return !required;   // absent-and-optional is not an error
    }
    void* d = nullptr;
    if (cudaMalloc(&d, (size_t)t->n_bytes) != cudaSuccess) {
        std::fprintf(stderr, "[k3] cudaMalloc failed for %s (%ld bytes)\n",
                     name.c_str(), t->n_bytes);
        return false;
    }
    if (cudaMemcpy(d, t->data, (size_t)t->n_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "[k3] cudaMemcpy failed for %s\n", name.c_str());
        cudaFree(d);
        return false;
    }
    w.owned.push_back(d);
    out.data = d;
    out.type = t->ggml_type;
    out.n_bytes = t->n_bytes;
    return true;
}

// wk_b / wv_b feed mla_absorb_q_f32 / mla_decode_attn_f32, which take them as plain
// `const float*` — those two kernels have no quantized-input variant, unlike every
// other weight in the model, because the per-head contraction they do is not an
// ordinary GEMV. So if the file stores them quantized (expected: Q8_0, same tier as
// the rest of MLA's attention weights), this uploads native bytes AND THEN
// dequantises once into a fresh F32 buffer — a one-time cost at load, not per token
// — rather than writing a second quantized-aware kernel variant for a tensor this
// small (about 25 MB each at F32, per K3's dims).
bool upload_force_f32(const GGUF& g, const std::string& name, KimiK3Weights& w,
                     KimiK3Tensor& out, bool required, long n_values) {
    const GGUFTensor* t = g.tensor(name);
    if (!t) {
        if (required)
            std::fprintf(stderr, "[k3] missing required tensor: %s\n", name.c_str());
        return !required;
    }
    if (t->ggml_type == 0) {
        return upload_raw(g, name, w, out, required);
    }
    void* raw = nullptr;
    if (cudaMalloc(&raw, (size_t)t->n_bytes) != cudaSuccess) return false;
    if (cudaMemcpy(raw, t->data, (size_t)t->n_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(raw);
        return false;
    }
    void* f32 = nullptr;
    if (cudaMalloc(&f32, (size_t)n_values * sizeof(float)) != cudaSuccess) {
        cudaFree(raw);
        return false;
    }
    const bool ok = k3k::dequant_f32_by_type((float*)f32, raw, n_values, t->ggml_type, 0);
    cudaDeviceSynchronize();
    cudaFree(raw);
    if (!ok) {
        std::fprintf(stderr, "[k3] %s: ggml type %d has no dequant path (needed as f32 "
                             "for the absorbed-MLA kernels)\n", name.c_str(), t->ggml_type);
        cudaFree(f32);
        return false;
    }
    w.owned.push_back(f32);
    out.data = f32;
    out.type = 0;   // now F32, regardless of the file's original type
    out.n_bytes = n_values * (long)sizeof(float);
    return true;
}


// Opt-in phase timing, SPARKINFER_K3_PROFILE=1. Records cudaEvent pairs around the
// attention and FFN branches and accumulates GPU time per tag, reading the elapsed
// time back only when the tag is next reused (by which point the events are long
// since complete), so the hot loop takes no extra sync. Off by default and behind a
// single env check, so the production path is untouched — this exists to find the
// bottleneck for the H200 optimization pass, not to run in it.
struct K3Profiler {
    struct Slot { cudaEvent_t a = nullptr, b = nullptr; bool pending = false; double ms = 0; long n = 0; };
    std::unordered_map<std::string, Slot> slots;
    bool on = false;
    K3Profiler() { const char* e = std::getenv("SPARKINFER_K3_PROFILE"); on = e && e[0] == '1'; }
    void start(const std::string& tag, cudaStream_t st) {
        if (!on) return;
        auto& sl = slots[tag];
        if (!sl.a) { cudaEventCreate(&sl.a); cudaEventCreate(&sl.b); }
        if (sl.pending) {   // drain the previous pair for this tag first
            cudaEventSynchronize(sl.b);
            float ms = 0; cudaEventElapsedTime(&ms, sl.a, sl.b); sl.ms += ms; ++sl.n;
        }
        cudaEventRecord(sl.a, st);
        sl.pending = true;
    }
    void stop(const std::string& tag, cudaStream_t st) {
        if (!on) return;
        cudaEventRecord(slots[tag].b, st);
    }
    void report() {
        if (!on) return;
        double total = 0;
        std::vector<std::pair<std::string, double>> rows;
        for (auto& kv : slots) {
            auto& sl = kv.second;
            if (sl.pending) { cudaEventSynchronize(sl.b); float ms = 0;
                cudaEventElapsedTime(&ms, sl.a, sl.b); sl.ms += ms; ++sl.n; sl.pending = false; }
            rows.emplace_back(kv.first, sl.ms);
            total += sl.ms;
        }
        std::sort(rows.begin(), rows.end(), [](auto& a, auto& b){ return a.second > b.second; });
        std::fprintf(stderr, "\n[k3 profile] %.1f ms total across timed phases:\n", total);
        for (auto& r : rows)
            std::fprintf(stderr, "  %-18s %8.2f ms  (%5.1f%%)\n",
                         r.first.c_str(), r.second, total > 0 ? 100.0 * r.second / total : 0.0);
    }
};
K3Profiler& k3_profiler() { static K3Profiler p; return p; }

}  // namespace

void kimi_k3_profile_report() { k3_profiler().report(); }

bool kimi_k3_load_weights(const GGUF& g, const KimiK3Config& cfg,
                         const K3PlanOptions& opt, KimiK3Weights& out,
                         int first_layer, int last_layer) {
    return kimi_k3_load_weights_scoped(g, cfg, opt, out, first_layer, last_layer,
                                      /*load_embed=*/true, /*load_head=*/true);
}

bool kimi_k3_load_weights_scoped(const GGUF& g, const KimiK3Config& cfg,
                                const K3PlanOptions& opt, KimiK3Weights& out,
                                int first_layer, int last_layer,
                                bool load_embed, bool load_head) {
    out.layers.assign(cfg.n_layers, KimiK3LayerWeights{});

    bool ok = true;
    if (load_embed)
        ok &= upload_raw(g, "token_embd.weight", out, out.token_embd, true);
    if (load_head) {
        ok &= upload_raw(g, "output_norm.weight", out, out.output_norm, true);
        ok &= upload_raw(g, "output.weight", out, out.output, true);
        if (cfg.attn_res_block_size > 0) {
            out.has_output_res_score = true;
            ok &= upload_raw(g, "output_res_score.weight", out, out.output_res_score, true);
        }
    }
    if (!ok) return false;

    const int qh = cfg.n_q_heads;
    const int qk_nope = cfg.key_length_mla - cfg.rope_dim;

    for (int i = first_layer; i <= last_layer && i < cfg.n_layers; ++i) {
        if (i < 0) continue;
        KimiK3LayerWeights& L = out.layers[i];
        L.is_kda = cfg.is_kda_layer(i);

        ok &= upload_raw(g, blk(i, "attn_norm.weight"), out, L.attn_norm, true);
        ok &= upload_raw(g, blk(i, "ffn_norm.weight"), out, L.ffn_norm, true);
        if (cfg.attn_res_block_size > 0) {
            ok &= upload_raw(g, blk(i, "attn_res_score.weight"), out, L.attn_res_score, true);
            ok &= upload_raw(g, blk(i, "ffn_res_score.weight"), out, L.ffn_res_score, true);
        }

        if (L.is_kda) {
            ok &= upload_raw(g, blk(i, "attn_q.weight"), out, L.attn_q, true);
            ok &= upload_raw(g, blk(i, "attn_k.weight"), out, L.attn_k, true);
            ok &= upload_raw(g, blk(i, "attn_v.weight"), out, L.attn_v, true);
            ok &= upload_raw(g, blk(i, "ssm_conv1d_q.weight"), out, L.ssm_conv1d_q, true);
            ok &= upload_raw(g, blk(i, "ssm_conv1d_k.weight"), out, L.ssm_conv1d_k, true);
            ok &= upload_raw(g, blk(i, "ssm_conv1d_v.weight"), out, L.ssm_conv1d_v, true);
            if (L.ssm_conv1d_q.ok() && L.ssm_conv1d_q.type != 0) {
                std::fprintf(stderr, "[k3] blk.%d.ssm_conv1d_q.weight is ggml type %d, "
                                     "not F32 — kda_conv_step_f32 has no quantized path\n",
                             i, L.ssm_conv1d_q.type);
                ok = false;
            }
            ok &= upload_raw(g, blk(i, "ssm_f_a.weight"), out, L.ssm_f_a, true);
            ok &= upload_raw(g, blk(i, "ssm_f_b.weight"), out, L.ssm_f_b, true);
            ok &= upload_raw(g, blk(i, "ssm_beta.weight"), out, L.ssm_beta, true);
            ok &= upload_raw(g, blk(i, "ssm_dt.bias"), out, L.ssm_dt_bias, true);
            ok &= upload_raw(g, blk(i, "ssm_a"), out, L.ssm_a, true);
            ok &= upload_raw(g, blk(i, "ssm_g.weight"), out, L.ssm_g, true);
            ok &= upload_raw(g, blk(i, "ssm_norm.weight"), out, L.ssm_norm, true);
            ok &= upload_raw(g, blk(i, "attn_output.weight"), out, L.attn_output, true);
        } else {
            L.has_q_lora = opt.has_q_lora;
            if (L.has_q_lora) {
                ok &= upload_raw(g, blk(i, "attn_q_a.weight"), out, L.attn_q_a, true);
                ok &= upload_raw(g, blk(i, "attn_q_a_norm.weight"), out, L.attn_q_a_norm, true);
                ok &= upload_raw(g, blk(i, "attn_q_b.weight"), out, L.attn_q_b, true);
            } else {
                ok &= upload_raw(g, blk(i, "attn_q.weight"), out, L.attn_q_dense, true);
            }
            ok &= upload_raw(g, blk(i, "attn_kv_a_mqa.weight"), out, L.attn_kv_a_mqa, true);
            ok &= upload_raw(g, blk(i, "attn_kv_a_norm.weight"), out, L.attn_kv_a_norm, true);

            L.has_fused_kv_b = opt.has_fused_kv_b;
            if (L.has_fused_kv_b) {
                std::fprintf(stderr, "[k3] blk.%d: fused attn_kv_b is not numerically "
                                     "wired (the split this needs is not derived) — "
                                     "refusing rather than running untested code\n", i);
                ok = false;
            } else {
                const long kb_n = (long)qk_nope * cfg.kv_lora_rank * qh;
                const long vb_n = (long)cfg.kv_lora_rank * cfg.value_length_mla * qh;
                ok &= upload_force_f32(g, blk(i, "attn_k_b.weight"), out, L.attn_k_b,
                                       true, kb_n);
                ok &= upload_force_f32(g, blk(i, "attn_v_b.weight"), out, L.attn_v_b,
                                       true, vb_n);
            }
            L.has_attn_gate = opt.has_attn_gate;
            if (L.has_attn_gate)
                ok &= upload_raw(g, blk(i, "attn_gate.weight"), out, L.attn_gate, true);
            ok &= upload_raw(g, blk(i, "attn_output.weight"), out, L.attn_output, true);
        }

        if (i < cfg.leading_dense) {
            ok &= upload_raw(g, blk(i, "ffn_gate.weight"), out, L.ffn_gate, true);
            ok &= upload_raw(g, blk(i, "ffn_up.weight"), out, L.ffn_up, true);
            ok &= upload_raw(g, blk(i, "ffn_down.weight"), out, L.ffn_down, true);
        } else {
            ok &= upload_raw(g, blk(i, "ffn_gate_inp.weight"), out, L.ffn_gate_inp, true);
            ok &= upload_raw(g, blk(i, "exp_probs_b.bias"), out, L.exp_probs_b, true);
            ok &= upload_raw(g, blk(i, "ffn_gate_exps.weight"), out, L.ffn_gate_exps, true);
            ok &= upload_raw(g, blk(i, "ffn_up_exps.weight"), out, L.ffn_up_exps, true);
            ok &= upload_raw(g, blk(i, "ffn_down_exps.weight"), out, L.ffn_down_exps, true);
            ok &= upload_raw(g, blk(i, "ffn_routed_down.weight"), out, L.ffn_routed_down, true);
            ok &= upload_raw(g, blk(i, "ffn_routed_up.weight"), out, L.ffn_routed_up, true);
            L.has_routed_norm = opt.has_routed_norm;
            if (L.has_routed_norm)
                ok &= upload_raw(g, blk(i, "ffn_routed_norm.weight"), out, L.ffn_routed_norm, true);
            L.has_shared_experts = opt.has_shared_experts;
            if (L.has_shared_experts) {
                ok &= upload_raw(g, blk(i, "ffn_gate_shexp.weight"), out, L.ffn_gate_shexp, true);
                ok &= upload_raw(g, blk(i, "ffn_up_shexp.weight"), out, L.ffn_up_shexp, true);
                ok &= upload_raw(g, blk(i, "ffn_down_shexp.weight"), out, L.ffn_down_shexp, true);
            }
        }
    }
    return ok;
}

void kimi_k3_free_weights(KimiK3Weights& w) {
    for (void* p : w.owned) cudaFree(p);
    w.owned.clear();
    w.layers.clear();
}

int kimi_k3_kda_ordinal(const KimiK3Config& cfg, int layer) {
    if (layer < 0 || layer >= cfg.n_layers || !cfg.is_kda_layer(layer)) return -1;
    int ord = 0;
    for (int i = 0; i < layer; ++i) if (cfg.is_kda_layer(i)) ++ord;
    return ord;
}

int kimi_k3_mla_ordinal(const KimiK3Config& cfg, int layer) {
    if (layer < 0 || layer >= cfg.n_layers || cfg.is_kda_layer(layer)) return -1;
    int ord = 0;
    for (int i = 0; i < layer; ++i) if (!cfg.is_kda_layer(i)) ++ord;
    return ord;
}

bool kimi_k3_alloc_state(const KimiK3Config& cfg, int max_ctx, KimiK3RuntimeState& out) {
    const int qkv = cfg.n_q_heads * cfg.kda_head_dim;
    const int n_kda = cfg.n_kda_layers();
    const int n_mla = cfg.n_mla_layers();
    out.max_ctx = max_ctx;
    out.max_ckpt = cfg.attn_res_block_size > 0
        ? (cfg.n_layers + cfg.attn_res_block_size - 1) / cfg.attn_res_block_size
        : 0;

    auto alloc = [&](size_t n_floats) -> float* {
        void* p = nullptr;
        if (cudaMalloc(&p, n_floats * sizeof(float)) != cudaSuccess) return nullptr;
        out.owned.push_back(p);
        return (float*)p;
    };

    out.conv_state_q.resize(n_kda);
    out.conv_state_k.resize(n_kda);
    out.conv_state_v.resize(n_kda);
    out.delta_state.resize(n_kda);
    for (int k = 0; k < n_kda; ++k) {
        out.conv_state_q[k] = alloc((size_t)(cfg.kda_conv_kernel - 1) * qkv);
        out.conv_state_k[k] = alloc((size_t)(cfg.kda_conv_kernel - 1) * qkv);
        out.conv_state_v[k] = alloc((size_t)(cfg.kda_conv_kernel - 1) * qkv);
        out.delta_state[k]  = alloc((size_t)cfg.kda_head_dim * cfg.kda_head_dim * cfg.n_q_heads);
        if (!out.conv_state_q[k] || !out.conv_state_k[k] || !out.conv_state_v[k] ||
            !out.delta_state[k])
            return false;
    }

    out.mla_kv_cache.resize(n_mla);
    for (int k = 0; k < n_mla; ++k) {
        out.mla_kv_cache[k] = alloc((size_t)cfg.key_length * max_ctx);
        if (!out.mla_kv_cache[k]) return false;
    }

    if (out.max_ckpt > 0) {
        out.res_bank = alloc((size_t)cfg.hidden * out.max_ckpt);
        if (!out.res_bank) return false;
    }

    out.conv_state_elems = (cfg.kda_conv_kernel - 1) * qkv;
    out.delta_state_elems = cfg.kda_head_dim * cfg.kda_head_dim * cfg.n_q_heads;
    out.kv_cache_elems = cfg.key_length * max_ctx;
    out.res_bank_row_elems = cfg.hidden;

    kimi_k3_reset_state(out);
    return true;
}

void kimi_k3_reset_state(KimiK3RuntimeState& s) {
    s.position = 0;
    s.n_ckpt = 0;
    auto z = [](float* p, size_t n_elems) {
        if (p && n_elems > 0) cudaMemset(p, 0, n_elems * sizeof(float));
    };
    for (float* p : s.conv_state_q) z(p, (size_t)s.conv_state_elems);
    for (float* p : s.conv_state_k) z(p, (size_t)s.conv_state_elems);
    for (float* p : s.conv_state_v) z(p, (size_t)s.conv_state_elems);
    for (float* p : s.delta_state)  z(p, (size_t)s.delta_state_elems);
    for (float* p : s.mla_kv_cache) z(p, (size_t)s.kv_cache_elems);
    // res_bank is not zeroed — n_ckpt=0 means no row is read until pushed, and a
    // push always writes the full row before n_ckpt is incremented, so stale bytes
    // in unused rows are never observed.
}

void kimi_k3_free_state(KimiK3RuntimeState& s) {
    for (void* p : s.owned) cudaFree(p);
    s.owned.clear();
    s.conv_state_q.clear(); s.conv_state_k.clear(); s.conv_state_v.clear();
    s.delta_state.clear(); s.mla_kv_cache.clear();
    s.res_bank = nullptr;
}

// ---------------------------------------------------------------------------
// Scratch
// ---------------------------------------------------------------------------

struct KimiK3Forward::Scratch {
    float* mixed = nullptr;        // [H]
    float* mixed2 = nullptr;       // [H]
    float* normed = nullptr;       // [H]
    float* normed2 = nullptr;      // [H]
    float* attn_out = nullptr;     // [H]
    float* ffn_out = nullptr;      // [H]

    // KDA
    float* qkv_q = nullptr, *qkv_k = nullptr, *qkv_v = nullptr;   // [qkv]
    float* conv_q = nullptr, *conv_k = nullptr, *conv_v = nullptr; // [qkv]
    float* f_a_out = nullptr;      // [head_dim]
    float* g_raw = nullptr;        // [qkv]
    float* decay_g = nullptr;      // [qkv]
    float* beta_out = nullptr;     // [n_head]
    float* delta_out = nullptr;    // [qkv]
    float* g_proj_out = nullptr;   // [qkv]
    float* gate_out = nullptr;     // [qkv]

    // MLA
    float* q_lora_out = nullptr;   // [q_lora_rank]
    float* q_proj_out = nullptr;   // [qh*key_length_mla]
    float* q_nope = nullptr;       // [qh*qk_nope]
    float* q_pe = nullptr;         // [qh*rope_dim]
    float* kv_a_out = nullptr;     // [key_length]
    float* kv_cmpr_normed = nullptr;  // [kv_lora_rank]
    float* absorbed_q = nullptr;   // [qh*key_length]
    float* mla_attn_out = nullptr; // [qh*value_length_mla]
    float* gate_proj_out = nullptr;   // [qh*value_length_mla]

    // FFN / MoE
    float* router_logits = nullptr;   // [n_experts]
    float* router_w = nullptr;        // [top_k]
    int*   router_ids = nullptr;      // [top_k]
    float* routed_down_out = nullptr; // [expert_latent]
    float* moe_scratch = nullptr;     // [top_k*moe_ffn]
    float* moe_out = nullptr;         // [expert_latent]
    float* dense_gate = nullptr, *dense_up = nullptr, *dense_situ = nullptr; // [dense_ffn]
    float* shexp_out = nullptr;        // [H] — shared-expert output, H-wide

    std::vector<void*> owned;
};

bool kimi_k3_forward_alloc_scratch(const KimiK3Config& cfg, KimiK3Forward& fwd) {
    fwd.s = new KimiK3Forward::Scratch();
    auto& s = *fwd.s;
    const int H = cfg.hidden;
    const int qkv = cfg.n_q_heads * cfg.kda_head_dim;
    const int qh = cfg.n_q_heads;
    const int qk_nope = cfg.key_length_mla - cfg.rope_dim;

    auto alloc_f = [&](float*& ptr, size_t n) {
        void* p = nullptr;
        if (cudaMalloc(&p, n * sizeof(float)) != cudaSuccess) return false;
        s.owned.push_back(p);
        ptr = (float*)p;
        return true;
    };
    auto alloc_i = [&](int*& ptr, size_t n) {
        void* p = nullptr;
        if (cudaMalloc(&p, n * sizeof(int)) != cudaSuccess) return false;
        s.owned.push_back(p);
        ptr = (int*)p;
        return true;
    };

    bool ok = true;
    ok &= alloc_f(s.mixed, H);
    ok &= alloc_f(s.mixed2, H);
    ok &= alloc_f(s.normed, H);
    ok &= alloc_f(s.normed2, H);
    ok &= alloc_f(s.attn_out, H);
    ok &= alloc_f(s.ffn_out, H);

    ok &= alloc_f(s.qkv_q, qkv);
    ok &= alloc_f(s.qkv_k, qkv);
    ok &= alloc_f(s.qkv_v, qkv);
    ok &= alloc_f(s.conv_q, qkv);
    ok &= alloc_f(s.conv_k, qkv);
    ok &= alloc_f(s.conv_v, qkv);
    ok &= alloc_f(s.f_a_out, cfg.kda_head_dim);
    ok &= alloc_f(s.g_raw, qkv);
    ok &= alloc_f(s.decay_g, qkv);
    ok &= alloc_f(s.beta_out, cfg.n_q_heads);
    ok &= alloc_f(s.delta_out, qkv);
    ok &= alloc_f(s.g_proj_out, qkv);
    ok &= alloc_f(s.gate_out, qkv);

    ok &= alloc_f(s.q_lora_out, cfg.q_lora_rank);
    ok &= alloc_f(s.q_proj_out, (size_t)qh * cfg.key_length_mla);
    ok &= alloc_f(s.q_nope, (size_t)qh * qk_nope);
    ok &= alloc_f(s.q_pe, (size_t)qh * cfg.rope_dim);
    ok &= alloc_f(s.kv_a_out, cfg.key_length);
    ok &= alloc_f(s.kv_cmpr_normed, cfg.kv_lora_rank);
    ok &= alloc_f(s.absorbed_q, (size_t)qh * cfg.key_length);
    ok &= alloc_f(s.mla_attn_out, (size_t)qh * cfg.value_length_mla);
    ok &= alloc_f(s.gate_proj_out, (size_t)qh * cfg.value_length_mla);

    ok &= alloc_f(s.router_logits, cfg.n_experts);
    ok &= alloc_f(s.router_w, cfg.top_k);
    ok &= alloc_i(s.router_ids, cfg.top_k);
    ok &= alloc_f(s.routed_down_out, cfg.expert_latent);
    ok &= alloc_f(s.moe_scratch, (size_t)cfg.top_k * cfg.moe_ffn);
    ok &= alloc_f(s.moe_out, cfg.expert_latent);
    ok &= alloc_f(s.dense_gate, cfg.dense_ffn);
    ok &= alloc_f(s.dense_up, cfg.dense_ffn);
    ok &= alloc_f(s.dense_situ, cfg.dense_ffn);
    ok &= alloc_f(s.shexp_out, H);

    if (!ok) { kimi_k3_forward_free_scratch(fwd); return false; }
    return true;
}

void kimi_k3_forward_free_scratch(KimiK3Forward& fwd) {
    if (!fwd.s) return;
    for (void* p : fwd.s->owned) cudaFree(p);
    delete fwd.s;
    fwd.s = nullptr;
}

// ---------------------------------------------------------------------------
// Forward, one layer
// ---------------------------------------------------------------------------

bool kimi_k3_forward_layer(KimiK3Forward& fwd, int layer, const float* hidden_in,
                          float* hidden_out) {
    const KimiK3Config& cfg = *fwd.cfg;
    const KimiK3LayerWeights& L = fwd.w->layers[layer];
    KimiK3RuntimeState& st = *fwd.state;
    auto& s = *fwd.s;
    cudaStream_t stream = fwd.stream;
    const float eps = cfg.rms_eps;

    const int H = cfg.hidden;
    const int qkv = cfg.n_q_heads * cfg.kda_head_dim;
    const int qh = cfg.n_q_heads;
    const int qk_nope = cfg.key_length_mla - cfg.rope_dim;
    const int res_bs = cfg.attn_res_block_size;
    const bool banked = res_bs > 0 && (layer % res_bs == 0);

    auto proj = [&](float* y, const float* x, const KimiK3Tensor& W, int N, int K) {
        if (!W.ok()) return false;
        return k3k::k3_proj_f32(y, x, W.data, W.type, N, K, stream);
    };

    // --- pre-attention mix, then bank push (raw pre-mix value) ---
    if (res_bs > 0) {
        if (!L.attn_res_score.ok()) return false;
        k3k::attn_res_mix_f32(s.mixed, st.res_bank, hidden_in,
                              (const float*)L.attn_res_score.data, H, st.n_ckpt,
                              eps, stream);
        if (banked) {
            if (st.n_ckpt >= st.max_ckpt) return false;
            cudaMemcpyAsync(st.res_bank + (size_t)st.n_ckpt * H, hidden_in,
                            (size_t)H * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            ++st.n_ckpt;
        }
    } else {
        cudaMemcpyAsync(s.mixed, hidden_in, (size_t)H * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }
    if (fwd.debug) fwd.debug("attn_res_mix", layer, s.mixed, H);

    if (!L.attn_norm.ok()) return false;
    k3k::rms_norm_f32(s.normed, s.mixed, (const float*)L.attn_norm.data, H, eps, stream);
    if (fwd.debug) fwd.debug("attn_norm", layer, s.normed, H);

    k3_profiler().start(L.is_kda ? "attn_kda" : "attn_mla", stream);
    if (L.is_kda) {
        const int kda_ord = kimi_k3_kda_ordinal(cfg, layer);
        if (kda_ord < 0) return false;
        const int head_dim = cfg.kda_head_dim, n_head = cfg.n_q_heads;

        if (!proj(s.qkv_q, s.normed, L.attn_q, qkv, H)) return false;
        if (!proj(s.qkv_k, s.normed, L.attn_k, qkv, H)) return false;
        if (!proj(s.qkv_v, s.normed, L.attn_v, qkv, H)) return false;

        if (!L.ssm_conv1d_q.ok() || !L.ssm_conv1d_k.ok() || !L.ssm_conv1d_v.ok())
            return false;
        k3k::kda_conv_step_f32(s.conv_q, st.conv_state_q[kda_ord], s.qkv_q,
                               (const float*)L.ssm_conv1d_q.data, cfg.kda_conv_kernel,
                               qkv, stream);
        k3k::kda_conv_step_f32(s.conv_k, st.conv_state_k[kda_ord], s.qkv_k,
                               (const float*)L.ssm_conv1d_k.data, cfg.kda_conv_kernel,
                               qkv, stream);
        k3k::kda_conv_step_f32(s.conv_v, st.conv_state_v[kda_ord], s.qkv_v,
                               (const float*)L.ssm_conv1d_v.data, cfg.kda_conv_kernel,
                               qkv, stream);
        if (fwd.debug) fwd.debug("dbg_conv_q", layer, s.conv_q, qkv);
        if (fwd.debug) fwd.debug("dbg_conv_v", layer, s.conv_v, qkv);

        // q gets the extra 1/sqrt(head_dim) scale the reference applies before the
        // scan; k does not (see kda_decode_step_f32's contract on pre-scaled q).
        k3k::l2_norm_heads_f32(s.conv_q, s.conv_q, head_dim, n_head,
                               1.0f / std::sqrt((float)head_dim), eps, stream);
        k3k::l2_norm_heads_f32(s.conv_k, s.conv_k, head_dim, n_head, 1.0f, eps, stream);
        if (fwd.debug) fwd.debug("dbg_l2_q", layer, s.conv_q, qkv);

        if (!proj(s.f_a_out, s.normed, L.ssm_f_a, head_dim, H)) return false;
        if (!proj(s.g_raw, s.f_a_out, L.ssm_f_b, qkv, head_dim)) return false;
        if (!L.ssm_dt_bias.ok()) return false;
        k3k::k3_add_f32(s.g_raw, s.g_raw, (const float*)L.ssm_dt_bias.data, qkv, stream);

        if (!L.ssm_a.ok()) return false;
        k3k::kda_decay_gate_f32(s.decay_g, s.g_raw, (const float*)L.ssm_a.data,
                                head_dim, n_head, cfg.kda_gate_lower_bound, stream);

        if (!proj(s.beta_out, s.normed, L.ssm_beta, n_head, H)) return false;

        if (fwd.debug) fwd.debug("dbg_decay_g", layer, s.decay_g, qkv);
        if (fwd.debug) fwd.debug("dbg_beta", layer, s.beta_out, n_head);
        k3k::kda_decode_step_f32(s.delta_out, st.delta_state[kda_ord],
                                 s.conv_q, s.conv_k, s.conv_v, s.decay_g, s.beta_out,
                                 head_dim, n_head, stream);
        if (fwd.debug) fwd.debug("dbg_delta_out", layer, s.delta_out, qkv);

        if (!proj(s.g_proj_out, s.normed, L.ssm_g, qkv, H)) return false;
        if (!L.ssm_norm.ok()) return false;
        k3k::kda_gate_out_f32(s.gate_out, s.delta_out, (const float*)L.ssm_norm.data,
                              s.g_proj_out, head_dim, n_head, eps, stream);
        if (fwd.debug) fwd.debug("dbg_gate_out", layer, s.gate_out, qkv);

        if (!proj(s.attn_out, s.gate_out, L.attn_output, H, qkv)) return false;
        if (fwd.debug) fwd.debug("kda_out", layer, s.attn_out, H);
    } else {
        const int mla_ord = kimi_k3_mla_ordinal(cfg, layer);
        if (mla_ord < 0) return false;

        if (L.has_q_lora) {
            if (!proj(s.q_lora_out, s.normed, L.attn_q_a, cfg.q_lora_rank, H)) return false;
            if (!L.attn_q_a_norm.ok()) return false;
            k3k::rms_norm_f32(s.q_lora_out, s.q_lora_out,
                              (const float*)L.attn_q_a_norm.data, cfg.q_lora_rank, eps,
                              stream);
            if (!proj(s.q_proj_out, s.q_lora_out, L.attn_q_b, qh * cfg.key_length_mla,
                     cfg.q_lora_rank))
                return false;
        } else {
            if (!proj(s.q_proj_out, s.normed, L.attn_q_dense, qh * cfg.key_length_mla, H))
                return false;
        }

        // De-interleave [qh, key_length_mla] into q_nope [qh, qk_nope] and
        // q_pe [qh, rope_dim] — each head's 192 values are qk_nope(128) then
        // rope_dim(64) concatenated, so this is a strided 2-D copy, not a plain split.
        cudaMemcpy2DAsync(s.q_nope, (size_t)qk_nope * sizeof(float),
                          s.q_proj_out, (size_t)cfg.key_length_mla * sizeof(float),
                          (size_t)qk_nope * sizeof(float), qh,
                          cudaMemcpyDeviceToDevice, stream);
        cudaMemcpy2DAsync(s.q_pe, (size_t)cfg.rope_dim * sizeof(float),
                          (const char*)s.q_proj_out + (size_t)qk_nope * sizeof(float),
                          (size_t)cfg.key_length_mla * sizeof(float),
                          (size_t)cfg.rope_dim * sizeof(float), qh,
                          cudaMemcpyDeviceToDevice, stream);

        if (!proj(s.kv_a_out, s.normed, L.attn_kv_a_mqa, cfg.key_length, H)) return false;
        if (!L.attn_kv_a_norm.ok()) return false;
        k3k::rms_norm_f32(s.kv_cmpr_normed, s.kv_a_out,
                          (const float*)L.attn_kv_a_norm.data, cfg.kv_lora_rank, eps,
                          stream);

        // K-cache row for this position: concat(normed kv_cmpr, RAW k_pe).
        float* row = st.mla_kv_cache[mla_ord] + (size_t)st.position * cfg.key_length;
        cudaMemcpyAsync(row, s.kv_cmpr_normed, (size_t)cfg.kv_lora_rank * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(row + cfg.kv_lora_rank, s.kv_a_out + cfg.kv_lora_rank,
                        (size_t)cfg.rope_dim * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);

        if (!L.attn_k_b.ok() || !L.attn_v_b.ok()) return false;
        k3k::mla_absorb_q_f32(s.absorbed_q, s.q_nope, s.q_pe,
                              (const float*)L.attn_k_b.data, qk_nope, cfg.kv_lora_rank,
                              cfg.rope_dim, qh, stream);
        if (fwd.debug) fwd.debug("mla_absorb_q", layer, s.absorbed_q, qh * cfg.key_length);

        const float mla_scale = 1.0f / std::sqrt((float)cfg.key_length_mla);
        k3k::mla_decode_attn_f32(s.mla_attn_out, s.absorbed_q, st.mla_kv_cache[mla_ord],
                                 (const float*)L.attn_v_b.data, cfg.key_length,
                                 cfg.kv_lora_rank, cfg.value_length_mla, qh,
                                 st.position + 1, mla_scale, stream);

        if (L.has_attn_gate) {
            if (!proj(s.gate_proj_out, s.normed, L.attn_gate, qh * cfg.value_length_mla, H))
                return false;
            k3k::mla_gate_out_f32(s.mla_attn_out, s.mla_attn_out, s.gate_proj_out,
                                  (int64_t)qh * cfg.value_length_mla, stream);
        }

        if (!proj(s.attn_out, s.mla_attn_out, L.attn_output, H, qh * cfg.value_length_mla))
            return false;
        if (fwd.debug) fwd.debug("mla_out", layer, s.attn_out, H);
    }

    k3_profiler().stop(L.is_kda ? "attn_kda" : "attn_mla", stream);

    // --- combine: replace on a checkpoint layer, add otherwise. Uses hidden_in
    // (the RAW pre-mix value), not s.mixed — the reference's residual add is
    // against the unmixed prefix_sum, only the norm/attention input was mixed. ---
    if (banked) {
        cudaMemcpyAsync(hidden_out, s.attn_out, (size_t)H * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    } else {
        k3k::k3_add_f32(hidden_out, hidden_in, s.attn_out, H, stream);
    }

    // --- pre-FFN mix, no bank push ---
    if (res_bs > 0) {
        if (!L.ffn_res_score.ok()) return false;
        k3k::attn_res_mix_f32(s.mixed2, st.res_bank, hidden_out,
                              (const float*)L.ffn_res_score.data, H, st.n_ckpt, eps,
                              stream);
    } else {
        cudaMemcpyAsync(s.mixed2, hidden_out, (size_t)H * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }
    if (!L.ffn_norm.ok()) return false;
    k3k::rms_norm_f32(s.normed2, s.mixed2, (const float*)L.ffn_norm.data, H, eps, stream);
    if (fwd.debug) fwd.debug("ffn_norm", layer, s.normed2, H);

    k3_profiler().start(layer < cfg.leading_dense ? "ffn_dense" : "ffn_moe", stream);
    if (layer < cfg.leading_dense) {
        if (!proj(s.dense_gate, s.normed2, L.ffn_gate, cfg.dense_ffn, H)) return false;
        if (!proj(s.dense_up, s.normed2, L.ffn_up, cfg.dense_ffn, H)) return false;
        if (fwd.debug) fwd.debug("dbg_dense_gate", layer, s.dense_gate, cfg.dense_ffn);
        if (fwd.debug) fwd.debug("dbg_dense_up", layer, s.dense_up, cfg.dense_ffn);
        k3k::situ_f32(s.dense_situ, s.dense_gate, s.dense_up, cfg.dense_ffn,
                     cfg.situ_beta, cfg.situ_linear_beta, stream);
        if (fwd.debug) fwd.debug("dbg_dense_situ", layer, s.dense_situ, cfg.dense_ffn);
        if (!proj(s.ffn_out, s.dense_situ, L.ffn_down, H, cfg.dense_ffn)) return false;
    } else {
        if (!proj(s.router_logits, s.normed2, L.ffn_gate_inp, cfg.n_experts, H))
            return false;
        if (!L.exp_probs_b.ok()) return false;
        k3k::moe_router_noaux_tc_f32(s.router_w, s.router_ids, s.router_logits,
                                     (const float*)L.exp_probs_b.data, cfg.n_experts,
                                     cfg.top_k, /*n_tokens=*/1, /*norm_w=*/true,
                                     /*w_scale=*/1.0f, stream);

        // routed_down feeds the dispatch UNNORMALISED — routed_norm (if present)
        // normalises the dispatch's OUTPUT, not this. See build_latent_moe in the
        // reference: build_moe_ffn runs first, then "if (layer.ffn_routed_norm)
        // moe_out = build_norm(moe_out, ...)", and only after that does
        // ffn_routed_up run. Getting this backwards (norm between routed_down and
        // the dispatch) was an earlier version's bug, same class as the ssm_norm fix.
        if (!proj(s.routed_down_out, s.normed2, L.ffn_routed_down, cfg.expert_latent, H))
            return false;

        if (!L.ffn_gate_exps.ok() || !L.ffn_up_exps.ok() || !L.ffn_down_exps.ok())
            return false;
        const bool moe_ok = k3k::moe_expert_ffn_f32_by_type(
            s.moe_out, s.moe_scratch, s.routed_down_out, s.router_ids, s.router_w,
            L.ffn_gate_exps.data, L.ffn_up_exps.data, L.ffn_down_exps.data,
            cfg.expert_latent, cfg.moe_ffn, cfg.top_k, cfg.situ_beta,
            cfg.situ_linear_beta, L.ffn_gate_exps.type, stream);
        if (!moe_ok) return false;

        if (L.has_routed_norm) {
            if (!L.ffn_routed_norm.ok()) return false;
            k3k::rms_norm_f32(s.moe_out, s.moe_out,
                              (const float*)L.ffn_routed_norm.data, cfg.expert_latent,
                              eps, stream);
        }

        if (!proj(s.ffn_out, s.moe_out, L.ffn_routed_up, H, cfg.expert_latent))
            return false;

        // Shared experts run on `normed2` directly (hidden width throughout, no
        // latent detour) and add into the routed output AFTER ffn_routed_up — both
        // contributions are at hidden width when combined. Reuses the dense-FFN
        // scratch buffers: a layer is either the leading-dense branch or this MoE
        // branch, never both, and dense_ffn (33792) comfortably covers n_ff_shexp
        // (6144), so a dedicated buffer would be pure duplication.
        if (L.has_shared_experts) {
            if (!L.ffn_gate_shexp.ok() || !L.ffn_up_shexp.ok() || !L.ffn_down_shexp.ok())
                return false;
            const int n_ff_shexp = cfg.moe_ffn * cfg.n_shared;
            if (!proj(s.dense_gate, s.normed2, L.ffn_gate_shexp, n_ff_shexp, H))
                return false;
            if (!proj(s.dense_up, s.normed2, L.ffn_up_shexp, n_ff_shexp, H))
                return false;
            k3k::situ_f32(s.dense_situ, s.dense_gate, s.dense_up, n_ff_shexp,
                         cfg.situ_beta, cfg.situ_linear_beta, stream);
            if (!proj(s.shexp_out, s.dense_situ, L.ffn_down_shexp, H, n_ff_shexp))
                return false;
            k3k::k3_add_f32(s.ffn_out, s.ffn_out, s.shexp_out, H, stream);
        }
    }
    k3_profiler().stop(layer < cfg.leading_dense ? "ffn_dense" : "ffn_moe", stream);
    if (fwd.debug) fwd.debug("ffn_out", layer, s.ffn_out, H);

    // FFN residual is ALWAYS an add, never a replace.
    k3k::k3_add_f32(hidden_out, hidden_out, s.ffn_out, H, stream);
    if (fwd.debug) fwd.debug("l_out", layer, hidden_out, H);
    return true;
}

// ---------------------------------------------------------------------------
// Forward, one token
// ---------------------------------------------------------------------------

bool kimi_k3_forward_token(KimiK3Forward& fwd, int token_id, float* out_logits) {
    const KimiK3Config& cfg = *fwd.cfg;
    const KimiK3Weights& w = *fwd.w;
    cudaStream_t stream = fwd.stream;
    const int H = cfg.hidden;

    // THE CROSS-LAYER RESIDUAL BANK IS PER-TOKEN, NOT PERSISTENT. In the reference
    // it is `ckpts`, a member of the per-forward-pass `graph` object, so it is
    // constructed empty on every forward pass and discarded at the end of it — the
    // checkpoints a token banks are visible only to LATER LAYERS OF THAT SAME TOKEN,
    // never to the next token.
    //
    // This is the opposite lifetime from the KDA recurrent state and the MLA KV
    // cache in the same struct, which DO persist across tokens (that is the entire
    // point of them). Three kinds of state, two different lifetimes, one struct —
    // so resetting it is easy to forget, and forgetting it is not subtle: with
    // max_ckpt = ceil(93/12) = 8 exactly filled by one token's eight checkpoint
    // layers, token 2's first push hits the `n_ckpt >= max_ckpt` guard in
    // forward_layer and the whole call fails. Caught by reasoning through a
    // multi-token decode before ever running one.
    //
    // Deliberately NOT reset inside forward_layer: that entry point is also used
    // standalone for per-layer validation, where the caller owns bank lifetime.
    fwd.state->n_ckpt = 0;

    float* x = nullptr;
    float* x_next = nullptr;
    if (cudaMalloc(&x, (size_t)H * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&x_next, (size_t)H * sizeof(float)) != cudaSuccess) {
        cudaFree(x);
        return false;
    }

    bool ok = true;
    if (!w.token_embd.ok()) ok = false;
    if (ok) {
        // Row gather: token_id's row is hidden contiguous values starting at
        // token_id * row_bytes(hidden, wtype).
        long row_bytes = 0;
        if (w.token_embd.type == 0) row_bytes = (long)H * sizeof(float);
        else if (w.token_embd.type == 8) row_bytes = (long)(H / 32) * 34;
        else ok = false;
        if (ok) {
            const char* base = (const char*)w.token_embd.data + (size_t)token_id * row_bytes;
            ok = k3k::dequant_f32_by_type(x, base, H, w.token_embd.type, stream);
        }
    }
    if (ok && fwd.debug) fwd.debug("inp_embd", -1, x, H);

    for (int layer = 0; ok && layer < cfg.n_layers; ++layer) {
        ok = kimi_k3_forward_layer(fwd, layer, x, x_next);
        std::swap(x, x_next);
    }

    if (ok) {
        if (cfg.attn_res_block_size > 0) {
            if (!w.has_output_res_score || !w.output_res_score.ok()) {
                ok = false;
            } else {
                k3k::attn_res_mix_f32(x_next, fwd.state->res_bank, x,
                                      (const float*)w.output_res_score.data, H,
                                      fwd.state->n_ckpt, cfg.rms_eps, stream);
                std::swap(x, x_next);
            }
        }
    }
    if (ok && !w.output_norm.ok()) ok = false;
    if (ok) {
        k3k::rms_norm_f32(x_next, x, (const float*)w.output_norm.data, H, cfg.rms_eps,
                          stream);
        std::swap(x, x_next);
        if (fwd.debug) fwd.debug("result_norm", -1, x, H);
    }

    float* logits_dev = nullptr;
    if (ok) {
        if (cudaMalloc(&logits_dev, (size_t)cfg.vocab * sizeof(float)) != cudaSuccess)
            ok = false;
    }
    if (ok && !w.output.ok()) ok = false;
    if (ok) {
        ok = k3k::k3_proj_f32(logits_dev, x, w.output.data, w.output.type, cfg.vocab, H,
                              stream);
    }
    if (ok) {
        cudaStreamSynchronize(stream);
        cudaMemcpy(out_logits, logits_dev, (size_t)cfg.vocab * sizeof(float),
                  cudaMemcpyDeviceToHost);
        ++fwd.state->position;
    }

    if (logits_dev) cudaFree(logits_dev);
    cudaFree(x);
    cudaFree(x_next);
    return ok;
}

// ---------------------------------------------------------------------------
// Layer-split pipeline
// ---------------------------------------------------------------------------

namespace {

// Per-layer device-byte cost, from the GGUF index (no data read). Used to balance
// stages: the leading dense layer is ~1.2 GiB against ~6 GiB for a MoE layer, so
// splitting by layer COUNT would leave stage 0 carrying a fraction of its share.
std::vector<long> layer_bytes(const GGUF& g, const KimiK3Config& cfg) {
    std::vector<long> out(cfg.n_layers, 0);
    for (const auto& name : g.tensor_names()) {
        if (name.rfind("blk.", 0) != 0) continue;
        const size_t dot = name.find('.', 4);
        if (dot == std::string::npos) continue;
        const int idx = std::atoi(name.substr(4, dot - 4).c_str());
        if (idx < 0 || idx >= cfg.n_layers) continue;
        const GGUFTensor* t = g.tensor(name);
        if (t) out[idx] += t->n_bytes;
    }
    return out;
}

}  // namespace

bool kimi_k3_pipeline_init(const GGUF& g, const KimiK3Config& cfg,
                          const K3PlanOptions& opt,
                          const std::vector<int>& devices, int max_ctx,
                          KimiK3Pipeline& out) {
    if (devices.empty()) return false;
    out.cfg = cfg;
    out.opt = opt;
    out.stages.clear();
    out.stages.resize(devices.size());
    out.host_hidden.assign(cfg.hidden, 0.0f);

    // Balance by cumulative byte cost rather than layer count.
    const std::vector<long> lb = layer_bytes(g, cfg);
    long total = 0;
    for (long v : lb) total += v;
    const long per_stage = total / (long)devices.size();

    int layer = 0;
    for (size_t si = 0; si < devices.size(); ++si) {
        KimiK3PipelineStage& st = out.stages[si];
        st.device = devices[si];
        st.first_layer = layer;
        long acc = 0;
        // Last stage takes whatever remains, so no layer is ever dropped by rounding.
        while (layer < cfg.n_layers &&
               (si + 1 == devices.size() || acc < per_stage || layer == st.first_layer)) {
            acc += lb[layer];
            ++layer;
            if (si + 1 < devices.size() && acc >= per_stage) break;
        }
        st.last_layer = layer - 1;
    }
    if (layer != cfg.n_layers) {
        std::fprintf(stderr, "[k3] pipeline split covered %d of %d layers\n",
                     layer, cfg.n_layers);
        return false;
    }

    for (size_t si = 0; si < out.stages.size(); ++si) {
        KimiK3PipelineStage& st = out.stages[si];
        if (cudaSetDevice(st.device) != cudaSuccess) return false;
        const bool first = (si == 0), last = (si + 1 == out.stages.size());
        if (!kimi_k3_load_weights_scoped(g, cfg, opt, st.weights,
                                        st.first_layer, st.last_layer,
                                        /*load_embed=*/first, /*load_head=*/last))
            return false;
        if (!kimi_k3_alloc_state(cfg, max_ctx, st.state)) return false;
        st.fwd.cfg = &out.cfg;
        st.fwd.w = &st.weights;
        st.fwd.state = &st.state;
        st.fwd.opt = out.opt;
        st.fwd.stream = nullptr;
        if (!kimi_k3_forward_alloc_scratch(cfg, st.fwd)) return false;
        if (cudaMalloc(&st.hidden, (size_t)cfg.hidden * sizeof(float)) != cudaSuccess)
            return false;
        if (last && cudaMalloc(&st.logits, (size_t)cfg.vocab * sizeof(float)) != cudaSuccess)
            return false;
        std::fprintf(stderr, "[k3] stage %zu: device %d, layers %d-%d (%.2f GiB)\n",
                     si, st.device, st.first_layer, st.last_layer,
                     [&]{ long a = 0; for (int i = st.first_layer; i <= st.last_layer; ++i) a += lb[i];
                          return a / 1073741824.0; }());
    }
    out.host_bank.assign((size_t)cfg.hidden * out.stages[0].state.max_ckpt, 0.0f);
    return true;
}

bool kimi_k3_pipeline_forward_token(KimiK3Pipeline& p, int token_id, float* out_logits) {
    if (p.stages.empty()) return false;
    const KimiK3Config& cfg = p.cfg;
    const int H = cfg.hidden;

    // The residual bank is per-TOKEN and spans every layer, so it starts empty and
    // then follows the hidden state across every stage boundary. See the header.
    for (auto& st : p.stages) st.state.n_ckpt = 0;

    // ---- stage 0: embed ----
    KimiK3PipelineStage& s0 = p.stages[0];
    if (cudaSetDevice(s0.device) != cudaSuccess) return false;
    if (!s0.weights.token_embd.ok()) return false;
    {
        long row_bytes = 0;
        const int ty = s0.weights.token_embd.type;
        if (ty == 0) row_bytes = (long)H * sizeof(float);
        else if (ty == 8) row_bytes = (long)(H / 32) * 34;
        else return false;
        const char* base = (const char*)s0.weights.token_embd.data +
                           (size_t)token_id * row_bytes;
        if (!k3k::dequant_f32_by_type(s0.hidden, base, H, ty, nullptr)) return false;
    }

    int n_ckpt_carry = 0;
    for (size_t si = 0; si < p.stages.size(); ++si) {
        KimiK3PipelineStage& st = p.stages[si];
        if (cudaSetDevice(st.device) != cudaSuccess) return false;

        if (si > 0) {
            // HANDOFF: hidden state AND the residual bank. Transferring only the
            // hidden state would leave this stage's res_mix scoring against an
            // empty bank — fluent, wrong, and invisible without a reference.
            if (cudaMemcpy(st.hidden, p.host_hidden.data(), (size_t)H * sizeof(float),
                          cudaMemcpyHostToDevice) != cudaSuccess) return false;
            st.state.n_ckpt = n_ckpt_carry;
            if (n_ckpt_carry > 0 &&
                cudaMemcpy(st.state.res_bank, p.host_bank.data(),
                          (size_t)n_ckpt_carry * H * sizeof(float),
                          cudaMemcpyHostToDevice) != cudaSuccess) return false;
        }

        for (int L = st.first_layer; L <= st.last_layer; ++L) {
            if (!kimi_k3_forward_layer(st.fwd, L, st.hidden, st.hidden)) {
                std::fprintf(stderr, "[k3] pipeline: layer %d failed on stage %zu\n", L, si);
                return false;
            }
        }
        if (cudaDeviceSynchronize() != cudaSuccess) return false;

        if (si + 1 < p.stages.size()) {
            if (cudaMemcpy(p.host_hidden.data(), st.hidden, (size_t)H * sizeof(float),
                          cudaMemcpyDeviceToHost) != cudaSuccess) return false;
            n_ckpt_carry = st.state.n_ckpt;
            if (n_ckpt_carry > 0 &&
                cudaMemcpy(p.host_bank.data(), st.state.res_bank,
                          (size_t)n_ckpt_carry * H * sizeof(float),
                          cudaMemcpyDeviceToHost) != cudaSuccess) return false;
        }
    }

    // ---- last stage: final mix, norm, lm_head ----
    KimiK3PipelineStage& sl = p.stages.back();
    if (cudaSetDevice(sl.device) != cudaSuccess) return false;
    float* x = sl.hidden;
    float* tmp = sl.fwd.s->mixed;   // reuse an H-wide scratch slot
    if (cfg.attn_res_block_size > 0) {
        if (!sl.weights.has_output_res_score || !sl.weights.output_res_score.ok())
            return false;
        k3k::attn_res_mix_f32(tmp, sl.state.res_bank, x,
                              (const float*)sl.weights.output_res_score.data, H,
                              sl.state.n_ckpt, cfg.rms_eps, nullptr);
        std::swap(x, tmp);
    }
    if (!sl.weights.output_norm.ok()) return false;
    k3k::rms_norm_f32(tmp, x, (const float*)sl.weights.output_norm.data, H,
                      cfg.rms_eps, nullptr);
    if (!sl.weights.output.ok()) return false;
    if (!k3k::k3_proj_f32(sl.logits, tmp, sl.weights.output.data,
                          sl.weights.output.type, cfg.vocab, H, nullptr))
        return false;
    if (cudaDeviceSynchronize() != cudaSuccess) return false;
    if (cudaMemcpy(out_logits, sl.logits, (size_t)cfg.vocab * sizeof(float),
                  cudaMemcpyDeviceToHost) != cudaSuccess) return false;

    for (auto& st : p.stages) ++st.state.position;
    return true;
}

void kimi_k3_pipeline_free(KimiK3Pipeline& p) {
    for (auto& st : p.stages) {
        cudaSetDevice(st.device);
        kimi_k3_forward_free_scratch(st.fwd);
        kimi_k3_free_state(st.state);
        kimi_k3_free_weights(st.weights);
        if (st.hidden) cudaFree(st.hidden);
        if (st.logits) cudaFree(st.logits);
        st.hidden = nullptr; st.logits = nullptr;
    }
    p.stages.clear();
}

}  // namespace sparkinfer
