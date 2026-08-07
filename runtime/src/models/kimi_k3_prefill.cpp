// Batched prompt prefill — tile scratch and the attention-side projection fill.
// See sparkinfer/models/kimi_k3_prefill.h for why this exists and what it is worth.

#include "sparkinfer/models/kimi_k3_prefill.h"
#include "sparkinfer/kernels/kimi_k3.h"
#include "sparkinfer/kernels/kimi_k3_fast.h"
#include "sparkinfer/kernels/kimi_k3_prefill.h"

#include <cstdio>
#include <cstddef>
#include <cstdlib>

namespace sparkinfer {

namespace k3k = ::sparkinfer::kernels::k3;

namespace {

bool alloc_f(float*& p, size_t n) {
    void* q = nullptr;
    if (n == 0 || cudaMalloc(&q, n * sizeof(float)) != cudaSuccess) return false;
    p = (float*)q;
    return true;
}

bool alloc_i(int*& p, size_t n) {
    void* q = nullptr;
    if (n == 0 || cudaMalloc(&q, n * sizeof(int)) != cudaSuccess) return false;
    p = (int*)q;
    return true;
}

bool alloc_bytes(void*& p, size_t n) {
    if (n == 0) return false;
    return cudaMalloc(&p, n) == cudaSuccess;
}

}  // namespace

bool k3_prefill_tile_alloc(const KimiK3Config& cfg, int tile, int qkv, int max_ckpt,
                           K3PrefillTile& t) {
    if (tile <= 0 || qkv <= 0 || cfg.hidden <= 0) return false;
    // The activation quantiser works in 32-value blocks, and every projection here reads
    // `normed` at full hidden width. A hidden that is not a multiple of 32 would silently
    // drop the tail block, so refuse rather than round.
    if (cfg.hidden % 32 != 0) {
        std::fprintf(stderr, "[k3-prefill] hidden %d is not a multiple of 32\n", cfg.hidden);
        return false;
    }

    t.tile = tile;
    t.hidden = cfg.hidden;
    t.qkv = qkv;
    t.n_live = 0;
    t.layer = -1;
    t.max_ckpt = max_ckpt;

    const size_t H = (size_t)cfg.hidden;
    const size_t Q = (size_t)qkv;
    bool ok = true;
    ok = ok && alloc_f(t.x, (size_t)tile * H);
    ok = ok && alloc_f(t.x_next, (size_t)tile * H);
    ok = ok && alloc_f(t.mixed, (size_t)tile * H);
    ok = ok && alloc_f(t.normed, (size_t)tile * H);
    ok = ok && alloc_f(t.res_scores, (size_t)tile *
                       (size_t)(max_ckpt > 0 ? max_ckpt + 1 : 1));
    ok = ok && alloc_f(t.q, (size_t)tile * Q);
    ok = ok && alloc_f(t.k, (size_t)tile * Q);
    ok = ok && alloc_f(t.v, (size_t)tile * Q);
    ok = ok && alloc_f(t.g, (size_t)tile * Q);
    ok = ok && alloc_f(t.kda_gate_out, (size_t)tile * Q);
    ok = ok && alloc_f(t.normed2, (size_t)tile * H);
    ok = ok && alloc_f(t.routed_down, (size_t)tile * (size_t)cfg.expert_latent);
    ok = ok && alloc_f(t.router_logits, (size_t)tile * (size_t)cfg.n_experts);
    ok = ok && alloc_f(t.router_w, (size_t)tile * (size_t)cfg.top_k);
    ok = ok && alloc_i(t.router_ids, (size_t)tile * (size_t)cfg.top_k);
    const size_t shared_cap = (size_t)cfg.moe_ffn * (size_t)cfg.n_shared;
    ok = ok && alloc_f(t.shexp_gate, (size_t)tile * shared_cap);
    ok = ok && alloc_f(t.shexp_up, (size_t)tile * shared_cap);
    ok = ok && alloc_f(t.shexp_situ, (size_t)tile * shared_cap);
    ok = ok && alloc_f(t.moe_normed, (size_t)tile * (size_t)cfg.expert_latent);
    ok = ok && alloc_f(t.routed_up, (size_t)tile * H);
    ok = ok && alloc_f(t.attn_out, (size_t)tile * H);
    ok = ok && alloc_f(t.moe_fused, (size_t)tile *
                       ((size_t)cfg.expert_latent + H));
    const size_t pairs = (size_t)tile * (size_t)cfg.top_k;
    const size_t latent_blocks = (size_t)cfg.expert_latent / 32;
    const size_t expert_ffn_cap = (size_t)cfg.moe_ffn;
    ok = ok && alloc_bytes((void*&)t.expert_xq, (size_t)tile * cfg.expert_latent);
    ok = ok && alloc_f(t.expert_xscale, (size_t)tile * latent_blocks);
    ok = ok && alloc_i(t.expert_rows, pairs);
    ok = ok && alloc_i(t.expert_slots, pairs);
    ok = ok && alloc_i(t.expert_inverse, pairs);
    ok = ok && alloc_i(t.expert_bounds, (size_t)cfg.n_experts + 1);
    ok = ok && alloc_f(t.expert_gate, pairs * expert_ffn_cap);
    ok = ok && alloc_f(t.expert_up, pairs * expert_ffn_cap);
    ok = ok && alloc_f(t.expert_situ, pairs * expert_ffn_cap);
    ok = ok && alloc_bytes((void*&)t.expert_hq, pairs * expert_ffn_cap);
    ok = ok && alloc_f(t.expert_hscale, pairs * ((expert_ffn_cap + 31) / 32));
    ok = ok && alloc_f(t.expert_down, pairs * (size_t)cfg.expert_latent);
    ok = ok && alloc_f(t.expert_out, (size_t)tile * (size_t)cfg.expert_latent);
    if (ok) {
        const size_t qb = k3k::k3_prefill_act_q8_bytes(cfg.hidden, tile);
        ok = qb != 0 && cudaMalloc(&t.q8, qb) == cudaSuccess;
    }
    if (!ok) {
        k3_prefill_tile_free(t);
        std::fprintf(stderr, "[k3-prefill] tile alloc failed (tile=%d H=%d qkv=%d)\n",
                     tile, cfg.hidden, qkv);
        return false;
    }
    return true;
}

void k3_prefill_tile_free(K3PrefillTile& t) {
    auto f = [](void*& p) { if (p) { cudaFree(p); p = nullptr; } };
    f((void*&)t.x); f((void*&)t.x_next); f((void*&)t.mixed); f((void*&)t.normed);
    f((void*&)t.q); f((void*&)t.k); f((void*&)t.v); f((void*&)t.g);
    f((void*&)t.kda_gate_out);
    f((void*&)t.normed2); f((void*&)t.routed_down);
    f((void*&)t.router_logits); f((void*&)t.router_w); f((void*&)t.router_ids);
    f((void*&)t.shexp_gate); f((void*&)t.shexp_up); f((void*&)t.shexp_situ);
    f((void*&)t.moe_normed); f((void*&)t.routed_up);
    f((void*&)t.attn_out); f((void*&)t.moe_fused);
    f((void*&)t.expert_xq); f((void*&)t.expert_xscale);
    f((void*&)t.expert_rows); f((void*&)t.expert_slots);
    f((void*&)t.expert_inverse); f((void*&)t.expert_bounds);
    f((void*&)t.expert_gate); f((void*&)t.expert_up); f((void*&)t.expert_situ);
    f((void*&)t.expert_hq); f((void*&)t.expert_hscale);
    f((void*&)t.expert_down); f((void*&)t.expert_out);
    f((void*&)t.res_scores); f(t.q8);
    t.tile = t.hidden = t.qkv = t.max_ckpt = 0;
    t.n_live = 0;
    t.layer = -1;
    t.ffn_norm_layer = -1;
    t.ffn_layer = -1;
    t.expert_layer = -1;
    t.attn_layer = -1;
    t.kda_out_layer = -1;
}

bool k3_prefill_fill_qkvg(K3PrefillTile& t, const KimiK3LayerWeights& L,
                          const KimiK3Config& cfg, int layer, int n_tok,
                          float* const* banks, const int* n_ckpt,
                          cudaStream_t stream) {
    if (n_tok <= 0 || n_tok > t.tile) return false;
    if (!t.x || !t.mixed || !t.normed || !t.q8 || !t.q || !t.k || !t.v || !t.g)
        return false;
    if (!L.attn_norm.ok() || !L.attn_q.ok() || !L.attn_k.ok() ||
        !L.attn_v.ok() || !L.ssm_g.ok()) return false;
    // Q8_0 only. The four weights must agree, because they share one quantised activation
    // and one launch geometry; a mixed group would need four different readers and this
    // path was not written for that.
    if (L.attn_q.type != 8 || L.attn_k.type != 8 ||
        L.attn_v.type != 8 || L.ssm_g.type != 8) return false;

    const int H = t.hidden, Q = t.qkv;
    const int res_bs = cfg.attn_res_block_size;
    if (res_bs > 0) {
        if (!L.attn_res_score.ok() || !banks || !n_ckpt || !t.res_scores) return false;
    }

    // THE PROJECTIONS' INPUT, DERIVED THE SAME WAY THE ATTENTION PHASE DERIVES IT.
    //
    // kimi_k3.cpp does res_mix THEN attn_norm, and the norm reads the MIX, not the raw
    // residual stream. Normalising t.x directly was this module's first shipped defect:
    // it left the argmax correct and the final logit 4.77 low over a 1024-token prompt,
    // which no timing benchmark and no "the output looks fine" reading would have caught.
    //
    // The BANK PUSH is deliberately absent. In the forward the push follows the mix and
    // stores the RAW pre-mix value, and it is the consumer's attention phase that will do
    // it — for this same token, from the same hidden_in, at the same point in the layer.
    // Doing it here as well would double-push, so this call reads the bank and never
    // writes it.
    //
    // Per token, not batched, and deliberately: both kernels are bit-identical this way by
    // construction, and they are small against the projections' 25%. A batched norm would
    // mean copying rms_norm_block_for and block_sum into this file, and a duplicated tier
    // is what produced the block_for defect the prefill CPU test pins.
    for (int i = 0; i < n_tok; ++i) {
        const float* xi = t.x + (size_t)i * H;
        float* mi = t.mixed + (size_t)i * H;
        if (res_bs > 0)
            k3k::attn_res_mix_f32(mi, banks[i], xi,
                                  (const float*)L.attn_res_score.data, H, n_ckpt[i],
                                  cfg.rms_eps, stream, t.res_scores);
        else
            cudaMemcpyAsync(mi, xi, (size_t)H * sizeof(float), cudaMemcpyDeviceToDevice,
                            stream);
    }
    // ONE norm for the whole tile instead of n_tok of them. `mixed` and `normed` are both
    // [tile][H], so the rows are H apart and the batched launch is the same kernel with a
    // token axis — bit-identical to the loop it replaces, which is why this is a launcher
    // change and not a numerics change.
    //
    // It declines rather than guessing for any geometry outside the wide kernel, and the
    // loop below is the fallback. That path does not fire at K3's H (7168 is a multiple of
    // 4 and the tile buffers are 16-byte aligned, so vec4 holds), but a shape that reached
    // it would otherwise get a different reduction tree and silently different logits.
    if (!k3k::rms_norm_f32_rows(t.normed, t.mixed, (const float*)L.attn_norm.data, H,
                                cfg.rms_eps, n_tok, (long long)H, stream)) {
        for (int i = 0; i < n_tok; ++i)
            k3k::rms_norm_f32(t.normed + (size_t)i * H, t.mixed + (size_t)i * H,
                              (const float*)L.attn_norm.data, H, cfg.rms_eps, stream);
    }

    // One quantise for the whole tile. Per row this is byte-for-byte what the decode path's
    // k3_quantize_act_f32 writes for that row alone.
    if (!k3k::k3_prefill_quantize_act(t.q8, t.normed, H, n_tok, stream)) return false;

    // THE POINT OF THE MODULE. Each weight block is read once per (block, output row) and
    // multiplied into a tile of tokens, instead of being re-streamed for every token.
    float* out[4] = { t.q, t.k, t.v, t.g };
    const void* W[4] = { L.attn_q.data, L.attn_k.data, L.attn_v.data, L.ssm_g.data };
    for (int j = 0; j < 4; ++j)
        if (!k3k::k3_prefill_proj_q8act(out[j], t.q8, W[j], 8, Q, H, n_tok, stream))
            return false;

    t.n_live = n_tok;
    t.layer = layer;
    t.kda_out_layer = (L.attn_output.ok() && L.attn_output.type == 8 &&
                       t.kda_gate_out) ? layer : -1;
    return true;
}

bool k3_prefill_finish_kda_output(K3PrefillTile& t, const KimiK3LayerWeights& L,
                                  const KimiK3Config& cfg, int layer, int n_tok,
                                  float* dst, int64_t dst_row_stride,
                                  cudaStream_t stream) {
    if (t.kda_out_layer != layer || n_tok <= 0 || n_tok > t.n_live || !dst ||
        !t.kda_gate_out || !t.q8 || !L.attn_output.ok() ||
        L.attn_output.type != 8 || t.qkv % 32 != 0) return false;
    if (!k3k::k3_prefill_quantize_act(t.q8, t.kda_gate_out, t.qkv, n_tok, stream))
        return false;
    return k3k::k3_prefill_proj_q8act(dst, t.q8, L.attn_output.data,
                                     L.attn_output.type, cfg.hidden, t.qkv,
                                     n_tok, stream, dst_row_stride);
}

bool k3_prefill_fill_routed_down(K3PrefillTile& t, const KimiK3LayerWeights& L,
                                 const KimiK3Config& cfg, int layer, int n_tok,
                                 cudaStream_t stream) {
    static const bool want = [] {
        const char* e = std::getenv("SPARKINFER_K3_PREFILL_ROUTED_DOWN");
        return !(e && e[0] == '0');
    }();
    t.ffn_layer = -1;
    if (!want || layer < cfg.leading_dense || n_tok <= 0 || n_tok > t.tile) return false;
    if (!t.normed2 || !t.routed_down || !t.router_logits || !t.router_w ||
        !t.router_ids || !t.q8 || !L.ffn_routed_down.ok() ||
        !L.ffn_gate_inp.ok() || !L.exp_probs_b.ok()) return false;
    if (L.ffn_routed_down.type != 8 || L.ffn_gate_inp.type != 8 ||
        cfg.hidden % 32 != 0) return false;
    if (!k3k::k3_prefill_quantize_act(t.q8, t.normed2, cfg.hidden, n_tok, stream))
        return false;
    if (!k3k::k3_prefill_proj_q8act(t.routed_down, t.q8, L.ffn_routed_down.data,
                                    L.ffn_routed_down.type, cfg.expert_latent,
                                    cfg.hidden, n_tok, stream))
        return false;
    if (!k3k::k3_prefill_proj_q8act(t.router_logits, t.q8, L.ffn_gate_inp.data,
                                    L.ffn_gate_inp.type, cfg.n_experts,
                                    cfg.hidden, n_tok, stream))
        return false;
    if (!k3k::k3_moe_router_fast(t.router_w, t.router_ids, t.router_logits,
                                 (const float*)L.exp_probs_b.data, cfg.n_experts,
                                 cfg.top_k, n_tok, /*norm_w=*/true,
                                 /*w_scale=*/1.0f, stream))
        k3k::moe_router_noaux_tc_f32(t.router_w, t.router_ids, t.router_logits,
                                     (const float*)L.exp_probs_b.data, cfg.n_experts,
                                     cfg.top_k, n_tok, /*norm_w=*/true,
                                     /*w_scale=*/1.0f, stream);

    const int moe_width = cfg.expert_latent + cfg.hidden;
    float* const shexp_dst = t.moe_fused + cfg.expert_latent;
    if (L.has_shared_experts) {
        if (!t.shexp_gate || !t.shexp_up || !t.shexp_situ ||
            !L.ffn_gate_shexp.ok() || !L.ffn_up_shexp.ok() ||
            !L.ffn_down_shexp.ok()) return false;
        const int band = (int)L.ffn_gate_shexp.rank_ne[1];
        const int cap = cfg.moe_ffn * cfg.n_shared;
        if (band <= 0 || band > cap || (int)L.ffn_down_shexp.rank_ne[0] != band ||
            L.ffn_gate_shexp.type != 8 || L.ffn_up_shexp.type != 8 ||
            L.ffn_down_shexp.type != 8) return false;
        if (!k3k::k3_prefill_proj_q8act(t.shexp_gate, t.q8,
                                        L.ffn_gate_shexp.data, 8,
                                        band, cfg.hidden, n_tok, stream) ||
            !k3k::k3_prefill_proj_q8act(t.shexp_up, t.q8,
                                        L.ffn_up_shexp.data, 8,
                                        band, cfg.hidden, n_tok, stream))
            return false;
        k3k::situ_f32(t.shexp_situ, t.shexp_gate, t.shexp_up,
                      (int64_t)n_tok * band, cfg.situ_beta,
                      cfg.situ_linear_beta, stream);
        // q8 is no longer needed for normed2 after router/routed/shared gate-up have
        // consumed it, so reuse the allocation for the narrower SiTU rows.
        if (!k3k::k3_prefill_quantize_act(t.q8, t.shexp_situ, band, n_tok, stream) ||
            !k3k::k3_prefill_proj_q8act(
                shexp_dst, t.q8, L.ffn_down_shexp.data, 8, cfg.hidden, band,
                n_tok, stream, moe_width))
            return false;
    } else {
        if (cudaMemset2DAsync(shexp_dst, (size_t)moe_width * sizeof(float), 0,
                              (size_t)cfg.hidden * sizeof(float), n_tok, stream)
            != cudaSuccess) return false;
    }
    t.ffn_layer = layer;
    return true;
}

bool k3_prefill_dispatch_experts(K3PrefillTile& t, const KimiK3LayerWeights& L,
                                 const KimiK3Config& cfg, int layer, int n_tok,
                                 int moe_ffn_rank, int expert_begin,
                                 int n_local_experts, cudaStream_t stream) {
    t.expert_layer = -1;
    if (layer < cfg.leading_dense || t.ffn_layer != layer || n_tok <= 0 ||
        n_tok > t.tile || moe_ffn_rank <= 0 || moe_ffn_rank > cfg.moe_ffn ||
        n_local_experts <= 0 || L.ffn_gate_exps.type != 19 ||
        L.ffn_up_exps.type != 19 || L.ffn_down_exps.type != 19)
        return false;
    if ((cfg.expert_latent % 256) || (moe_ffn_rank % 256) ||
        !t.expert_xq || !t.expert_xscale || !t.expert_rows ||
        !t.expert_slots || !t.expert_inverse || !t.expert_bounds ||
        !t.expert_gate || !t.expert_up || !t.expert_situ || !t.expert_hq ||
        !t.expert_hscale || !t.expert_down || !t.expert_out)
        return false;

    // Default: CSR + grouped MMA. One launch per local expert with BM from
    // max_rows (= n_tok). SPARKINFER_K3_PREFILL_MOE_PAIRS=1 restores the M=1
    // pairs path #136 first wired.
    static const bool want_pairs = [] {
        const char* e = std::getenv("SPARKINFER_K3_PREFILL_MOE_PAIRS");
        return e && e[0] == '1';
    }();

    const int P = n_tok * cfg.top_k;
    if (!k3k::k3_moe_iq1s_mma_quantize_rows(
            t.expert_xq, t.expert_xscale, t.routed_down,
            n_tok, cfg.expert_latent, stream))
        return false;

    if (!want_pairs) {
        if (!k3k::k3_moe_build_expert_csr(
                t.expert_rows, t.expert_slots, t.expert_inverse, t.expert_bounds,
                t.router_ids, n_tok, cfg.top_k, expert_begin, n_local_experts,
                stream) ||
            !k3k::k3_moe_iq1s_mma_grouped(
                t.expert_gate, t.expert_xq, t.expert_xscale, L.ffn_gate_exps.data,
                t.expert_rows, t.expert_bounds, n_local_experts, n_tok,
                moe_ffn_rank, cfg.expert_latent, stream) ||
            !k3k::k3_moe_iq1s_mma_grouped(
                t.expert_up, t.expert_xq, t.expert_xscale, L.ffn_up_exps.data,
                t.expert_rows, t.expert_bounds, n_local_experts, n_tok,
                moe_ffn_rank, cfg.expert_latent, stream) ||
            !k3k::k3_moe_situ_compact(
                t.expert_situ, t.expert_gate, t.expert_up, t.expert_inverse,
                n_tok, cfg.top_k, moe_ffn_rank, cfg.situ_beta,
                cfg.situ_linear_beta, stream) ||
            !k3k::k3_moe_iq1s_mma_quantize_rows(
                t.expert_hq, t.expert_hscale, t.expert_situ,
                P, moe_ffn_rank, stream) ||
            !k3k::k3_moe_iq1s_mma_grouped(
                t.expert_down, t.expert_hq, t.expert_hscale, L.ffn_down_exps.data,
                /*rows=*/nullptr, t.expert_bounds, n_local_experts, n_tok,
                cfg.expert_latent, moe_ffn_rank, stream) ||
            !k3k::k3_moe_combine_compact(
                t.expert_out, t.expert_down, t.expert_inverse, t.router_w,
                n_tok, cfg.expert_latent, cfg.top_k, stream))
            return false;
    } else if (!k3k::k3_moe_iq1s_mma_pairs(
                   t.expert_gate, t.expert_xq, t.expert_xscale, L.ffn_gate_exps.data,
                   t.router_ids, P, cfg.top_k, expert_begin, n_local_experts,
                   moe_ffn_rank, cfg.expert_latent, stream) ||
               !k3k::k3_moe_iq1s_mma_pairs(
                   t.expert_up, t.expert_xq, t.expert_xscale, L.ffn_up_exps.data,
                   t.router_ids, P, cfg.top_k, expert_begin, n_local_experts,
                   moe_ffn_rank, cfg.expert_latent, stream) ||
               !k3k::k3_moe_situ_pairs(
                   t.expert_situ, t.expert_gate, t.expert_up, t.router_ids,
                   P, moe_ffn_rank, expert_begin, n_local_experts, cfg.situ_beta,
                   cfg.situ_linear_beta, stream) ||
               !k3k::k3_moe_iq1s_mma_quantize_rows(
                   t.expert_hq, t.expert_hscale, t.expert_situ,
                   P, moe_ffn_rank, stream) ||
               !k3k::k3_moe_iq1s_mma_pairs(
                   t.expert_down, t.expert_hq, t.expert_hscale, L.ffn_down_exps.data,
                   t.router_ids, P, 1, expert_begin, n_local_experts,
                   cfg.expert_latent, moe_ffn_rank, stream) ||
               !k3k::k3_moe_combine_pairs(
                   t.expert_out, t.expert_down, t.router_ids, t.router_w,
                   n_tok, cfg.expert_latent, cfg.top_k, expert_begin,
                   n_local_experts, stream)) {
        return false;
    }
    t.expert_layer = layer;
    return true;
}

bool k3_prefill_prepare_ffn(K3PrefillTile& t, const KimiK3LayerWeights& L,
                            const KimiK3Config& cfg, int layer, int n_tok,
                            const float* banks, int64_t bank_stride, int n_ckpt,
                            cudaStream_t stream) {
    t.ffn_norm_layer = -1;
    if (n_tok <= 0 || n_tok > t.tile || !t.x || !t.x_next || !t.attn_out ||
        !t.mixed || !t.normed2 || !L.ffn_norm.ok()) return false;
    const int H = cfg.hidden;
    const int res_bs = cfg.attn_res_block_size;
    const bool banked = res_bs > 0 && (layer % res_bs == 0);
    if (res_bs > 0 && (!banks || bank_stride <= 0 || !L.ffn_res_score.ok() ||
                       !t.res_scores || n_ckpt < 0)) return false;

    static const bool res_fuse = [] {
        const char* e = std::getenv("SPARKINFER_K3_RES_FUSE");
        return !(e && e[0] == '0');
    }();
    const int64_t elems = (int64_t)n_tok * H;
    if (res_fuse && res_bs > 0 && n_ckpt > 0) {
        k3k::attn_res_mix_f32(
            t.mixed, banks, banked ? t.attn_out : t.x,
            (const float*)L.ffn_res_score.data, H, n_ckpt, cfg.rms_eps, stream,
            t.res_scores, banked ? nullptr : t.attn_out, t.x_next,
            n_tok, H, bank_stride, t.max_ckpt + 1);
    } else {
        if (banked)
            cudaMemcpyAsync(t.x_next, t.attn_out, (size_t)elems * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        else
            k3k::k3_add_f32(t.x_next, t.x, t.attn_out, elems, stream);

        if (res_bs > 0)
            k3k::attn_res_mix_f32(
                t.mixed, banks, t.x_next, (const float*)L.ffn_res_score.data,
                H, n_ckpt, cfg.rms_eps, stream, t.res_scores, nullptr, nullptr,
                n_tok, H, bank_stride, t.max_ckpt + 1);
        else
            cudaMemcpyAsync(t.mixed, t.x_next, (size_t)elems * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
    }
    if (!k3k::rms_norm_f32_rows(t.normed2, t.mixed,
                                (const float*)L.ffn_norm.data, H, cfg.rms_eps,
                                n_tok, H, stream)) {
        for (int i = 0; i < n_tok; ++i)
            k3k::rms_norm_f32(t.normed2 + (size_t)i * H,
                              t.mixed + (size_t)i * H,
                              (const float*)L.ffn_norm.data, H,
                              cfg.rms_eps, stream);
    }
    // This stage is also the producer of the tile-wide attention rows on MLA layers,
    // where qkvg fill did not run and therefore could not establish n_live itself.
    t.n_live = n_tok;
    t.attn_layer = layer;
    t.ffn_norm_layer = layer;
    return true;
}

bool k3_prefill_finish_ffn(K3PrefillTile& t, const KimiK3LayerWeights& L,
                           const KimiK3Config& cfg, int n_tok,
                           cudaStream_t stream) {
    if (n_tok <= 0 || n_tok > t.tile || !t.moe_fused || !t.moe_normed ||
        !t.routed_up || !t.x_next || !t.q8 || !L.ffn_routed_up.ok() ||
        L.ffn_routed_up.type != 8) return false;
    const int latent = cfg.expert_latent;
    const int H = cfg.hidden;
    const int moe_width = latent + H;
    if (L.has_routed_norm) {
        if (!L.ffn_routed_norm.ok()) return false;
        // Input and output have different row strides. PR #144's row launcher has
        // one shared stride, so loop rather than silently indexing compact output at
        // the fused-input pitch.
        for (int i = 0; i < n_tok; ++i)
            k3k::rms_norm_f32(t.moe_normed + (size_t)i * latent,
                              t.moe_fused + (size_t)i * moe_width,
                              (const float*)L.ffn_routed_norm.data, latent,
                              cfg.rms_eps, stream);
    } else {
        if (cudaMemcpy2DAsync(t.moe_normed, (size_t)latent * sizeof(float),
                              t.moe_fused, (size_t)moe_width * sizeof(float),
                              (size_t)latent * sizeof(float), n_tok,
                              cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
            return false;
    }
    if (!k3k::k3_prefill_quantize_act(t.q8, t.moe_normed, latent, n_tok, stream) ||
        !k3k::k3_prefill_proj_q8act(t.routed_up, t.q8, L.ffn_routed_up.data, 8,
                                    H, latent, n_tok, stream))
        return false;

    const int64_t elems = (int64_t)n_tok * H;
    if (L.has_shared_experts) {
        if (!k3k::k3_add3_rows_f32(
                t.x_next, t.routed_up, t.routed_up,
                t.moe_fused + latent, moe_width, t.x_next, n_tok, H, stream))
            return false;
    } else {
        k3k::k3_add_f32(t.x_next, t.x_next, t.routed_up, elems, stream);
    }
    return true;
}

}  // namespace sparkinfer
