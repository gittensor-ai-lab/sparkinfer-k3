#pragma once
// Batched prompt prefill for Kimi K3 — tile scratch and the stages that consume it.
//
// WHY A SEPARATE MODULE, AND WHY IT ALLOCATES ITS OWN BUFFERS
// -----------------------------------------------------------
// Decode's scratch is ONE TOKEN WIDE by construction — `alloc_f(s.normed, H)`, not
// `T * H` — so there is no point in the existing runtime where two tokens' activations
// coexist at the same layer. That is the whole reason prefill runs at decode speed: the
// loop is `for token: for layer: read every weight`, and every weight in the model is
// re-streamed for every token.
//
// This module holds the other loop order. It is ADDITIVE: it allocates its own tile-wide
// buffers and never touches a decode buffer, which is what structurally protects the 128k
// decode guard — a code path decode never enters cannot regress it. That matters because
// the eval refuses to score the whole round if decode slips 1% (#132).
//
// WHAT THE TILE BUYS, MEASURED
// ----------------------------
// The Q8_0 projections are 25.3% of a prefill token (4.406 ms of 17.441 ms, measured with
// SPARKINFER_K3_PROJ_REPEAT — an idempotent instrument, so routing and accuracy are
// untouched and the delta is a true marginal cost). Batching them measures 2.48x-6.18x at
// the shapes the engine actually issues per rank:
//
//     FUSED4 qkvg  N=1536 K=7168   2.48x vs k3_proj_q8_fused4_1bar, the real baseline
//     attn_output  N=7168 K=1536   3.60x
//     routed_up    N=7168 K=3584   2.69x
//     shexp down   N=7168 K=768    6.18x
//
// Narrower N helps the batched path rather than hurting it: at N=1536 the per-token kernel
// underfills the device, so there is more for the token axis to recover.
//
// NORMS ARE DELIBERATELY NOT BATCHED. Looping rms_norm_f32 per token is bit-identical by
// construction and norms are not where the time is. Batching them would mean duplicating
// rms_norm_block_for and block_sum into this file, and a duplicated tier is exactly what
// produced the block_for defect the prefill CPU test now pins. Not worth it for ~6%.

#include "sparkinfer/models/kimi_k3.h"

#include <cuda_runtime.h>

namespace sparkinfer {

// Tile-wide activation scratch. Sized once for the largest tile the driver will use;
// a shorter final tile just leaves the tail rows unread.
struct K3PrefillTile {
    int tile = 0;              // tokens this scratch can hold
    int hidden = 0;            // H
    int qkv = 0;               // this rank's KDA q/k/v/g width

    float* x      = nullptr;   // [tile][H]   layer input, one row per token
    float* x_next = nullptr;   // [tile][H]   layer output; swapped with x each layer
    // THE PROJECTIONS DO NOT READ `x`. They read rms_norm(attn_res_mix(x, bank)).
    //
    // Getting this wrong is not a tolerance question, it is a different tensor: the
    // cross-layer residual mix blends the current stream with a softmax over the
    // token's banked checkpoints, and normalising `x` directly skips it entirely.
    // Measured, that produced the CORRECT argmax with the final logit 4.77 off
    // (20.43 vs 25.20) — fluent and wrong, which is the exact failure mode this
    // module's header claims to protect against. Hence a real `mixed` row here
    // rather than normalising straight out of `x`.
    float* mixed  = nullptr;   // [tile][H]   attn_res_mix output, the norm's input
    float* normed = nullptr;   // [tile][H]   attn_norm output
    void*  q8     = nullptr;   // [tile][H/32] BlockQ8_0, `normed` quantised
    float* q      = nullptr;   // [tile][qkv] KDA attn_q
    float* k      = nullptr;   // [tile][qkv] KDA attn_k
    float* v      = nullptr;   // [tile][qkv] KDA attn_v
    float* g      = nullptr;   // [tile][qkv] KDA ssm_g
    float* kda_gate_out = nullptr; // [tile][qkv] deferred KDA output-proj input
    float* normed2 = nullptr;  // [tile][H] FFN-normalized activation
    float* routed_down = nullptr; // [tile][expert_latent]
    float* router_logits = nullptr; // [tile][n_experts]
    float* router_w = nullptr;      // [tile][top_k]
    int*   router_ids = nullptr;    // [tile][top_k]
    float* shexp_gate = nullptr;    // [tile][moe_ffn*n_shared] capacity
    float* shexp_up = nullptr;      // [tile][moe_ffn*n_shared] capacity
    float* shexp_situ = nullptr;    // [tile][moe_ffn*n_shared] capacity
    float* moe_normed = nullptr;    // [tile][expert_latent]
    float* routed_up = nullptr;     // [tile][H]
    float* attn_out = nullptr; // [tile][H] full attention result after TP reduction
    // Per-token TP partials kept adjacent so one collective can reduce the whole
    // MoE tile. Each row fuses routed latent and shared-expert hidden exactly like
    // the decode scratch exposed by kimi_k3_partial_buffer(FfnPartial).
    float* moe_fused = nullptr; // [tile][expert_latent + H]
    // Device-only grouped expert dispatch scratch. Capacity is tile*top_k pairs;
    // expert_bounds makes the fixed worst-case launch graph-safe without a host sync.
    signed char* expert_xq = nullptr;
    float* expert_xscale = nullptr;
    int* expert_rows = nullptr;
    int* expert_slots = nullptr;
    int* expert_inverse = nullptr;
    int* expert_bounds = nullptr;
    float* expert_gate = nullptr;
    float* expert_up = nullptr;
    float* expert_situ = nullptr;
    signed char* expert_hq = nullptr;
    float* expert_hscale = nullptr;
    float* expert_down = nullptr;
    float* expert_out = nullptr;
    // Softmax scratch for attn_res_mix, [max_ckpt + 1]. Owned so the mix never falls
    // back to the cudaMallocAsync/cudaFreeAsync pair it uses when handed nullptr —
    // which is not merely slow here but uncapturable, and a tile is the unit a graph
    // gets recorded over.
    float* res_scores = nullptr;
    int    max_ckpt = 0;

    // How many rows of the above are live. Set by the fill; read by the consumer so a
    // ragged final tile cannot be mistaken for a full one.
    int n_live = 0;
    // The layer the live rows belong to, or -1. Guards against a consumer reading a tile
    // that was filled for a different layer — the failure mode would be fluent, wrong
    // output rather than a crash, which is the one this project keeps refusing to ship.
    int layer = -1;
    int ffn_norm_layer = -1;
    int ffn_layer = -1;
    int expert_layer = -1;
    int attn_layer = -1;
    int kda_out_layer = -1;
};

bool k3_prefill_tile_alloc(const KimiK3Config& cfg, int tile, int qkv, int max_ckpt,
                           K3PrefillTile& t);

// Swap the tile's residual buffers between layers.
//
// The per-token driver swaps R.x / R.x_next once per layer and has to reset the pair every
// token, because 93 swaps is an ODD number: without the reset the assignment alternates
// between tokens and a captured graph — which bakes the addresses it recorded — would
// replay against the wrong pair. A tile driver inherits exactly that hazard, one level up:
// it must reset per TILE, and the tile is the unit a graph would be captured over.
inline void k3_prefill_tile_swap(K3PrefillTile& t) {
    float* tmp = t.x; t.x = t.x_next; t.x_next = tmp;
}
void k3_prefill_tile_free(K3PrefillTile& t);

// Fill q/k/v/g for `n_tok` tokens of one KDA layer from rows already staged in `t.x`.
//
// Reproduces the attention phase's ENTIRE pre-projection chain, per token and in its
// order: attn_res_mix over that token's checkpoint bank, then attn_norm over the mix.
// `banks[i]` is token i's res_bank and `n_ckpt[i]` its live checkpoint count, both as
// they stand on ENTRY to this layer — which is exactly what the consumer will see,
// because a token's bank push happens inside its own attention phase, after its mix.
// Both may be null when cfg.attn_res_block_size is 0, in which case the mix degenerates
// to a copy exactly as it does in the forward.
//
// The mix is deliberately recomputed here rather than shared with the consumer. It is
// one small kernel over H floats against the ~11.7 MB of Q8_0 weights this call exists
// to stop re-streaming, and having each side compute its own from the same inputs with
// the same kernel is what makes them equal by construction instead of by plumbing.
//
// Then: one tile-wide quantise, and four batched projections. Bit-identical to what the
// per-token path computes for each token — the batched projection preserves the
// thread-to-block striding, the dp4a operand order, the (float)sumi * (dw * dx)
// contraction, the shuffle butterfly and the increasing-warp fold. Verified on device
// over 630,784 cells (k3_prefill_proj_gpu_test) and in host float
// (k3_prefill_order_cpu_test).
//
// Returns false — never an error — for anything outside its contract, so a caller can fall
// back to the per-token path rather than silently getting a different number.
bool k3_prefill_fill_qkvg(K3PrefillTile& t, const KimiK3LayerWeights& L,
                          const KimiK3Config& cfg, int layer, int n_tok,
                          float* const* banks, const int* n_ckpt,
                          cudaStream_t stream);

// Project all deferred KDA gate-output rows directly into a strided collective input.
bool k3_prefill_finish_kda_output(K3PrefillTile& t, const KimiK3LayerWeights& L,
                                  const KimiK3Config& cfg, int layer, int n_tok,
                                  float* dst, int64_t dst_row_stride,
                                  cudaStream_t stream);

// Batch the replicated router and routed-down projections after every token in the tile
// has completed attention and FFN preparation, then select top-k for every row. The
// consumer aliases the three rows directly.
bool k3_prefill_fill_routed_down(K3PrefillTile& t, const KimiK3LayerWeights& L,
                                 const KimiK3Config& cfg, int layer, int n_tok,
                                 cudaStream_t stream);

bool k3_prefill_dispatch_experts(K3PrefillTile& t, const KimiK3LayerWeights& L,
                                 const KimiK3Config& cfg, int layer, int n_tok,
                                 int moe_ffn_rank, int expert_begin,
                                 int n_local_experts, cudaStream_t stream);

// Batch phase 2a (attention residual combine, residual mix, FFN norm) over the tile.
// `banks` is the base of [tile][bank_stride] checkpoint storage and every token must
// enter the layer with the same n_ckpt, as the model's layer-indexed bank schedule does.
bool k3_prefill_prepare_ffn(K3PrefillTile& t, const KimiK3LayerWeights& L,
                            const KimiK3Config& cfg, int layer, int n_tok,
                            const float* banks, int64_t bank_stride, int n_ckpt,
                            cudaStream_t stream);

// Batch phase 3 after the tile-wide MoE collective: routed norm/up and the final
// shared-expert plus residual fold. Writes x_next in place and leaves routed_up as the
// same debug-visible intermediate the scalar phase calls ffn_out.
bool k3_prefill_finish_ffn(K3PrefillTile& t, const KimiK3LayerWeights& L,
                           const KimiK3Config& cfg, int n_tok,
                           cudaStream_t stream);

}  // namespace sparkinfer
