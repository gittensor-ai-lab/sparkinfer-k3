// Batched prompt prefill — tile scratch and the attention-side projection fill.
// See sparkinfer/models/kimi_k3_prefill.h for why this exists and what it is worth.

#include "sparkinfer/models/kimi_k3_prefill.h"
#include "sparkinfer/kernels/kimi_k3.h"
#include "sparkinfer/kernels/kimi_k3_prefill.h"

#include <cstdio>
#include <cstddef>

namespace sparkinfer {

namespace k3k = ::sparkinfer::kernels::k3;

namespace {

bool alloc_f(float*& p, size_t n) {
    void* q = nullptr;
    if (n == 0 || cudaMalloc(&q, n * sizeof(float)) != cudaSuccess) return false;
    p = (float*)q;
    return true;
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
    ok = ok && alloc_f(t.res_scores, (size_t)(max_ckpt > 0 ? max_ckpt + 1 : 1));
    ok = ok && alloc_f(t.q, (size_t)tile * Q);
    ok = ok && alloc_f(t.k, (size_t)tile * Q);
    ok = ok && alloc_f(t.v, (size_t)tile * Q);
    ok = ok && alloc_f(t.g, (size_t)tile * Q);
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
    f((void*&)t.res_scores); f(t.q8);
    t.tile = t.hidden = t.qkv = t.max_ckpt = 0;
    t.n_live = 0;
    t.layer = -1;
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
        k3k::rms_norm_f32(t.normed + (size_t)i * H, mi,
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
    return true;
}

}  // namespace sparkinfer
