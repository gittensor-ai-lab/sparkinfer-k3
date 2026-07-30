#pragma once
// Kimi K3 forward pass: weight loading + per-token decode, single GPU (tp_size=1).
//
// Everything upstream of this file — the kernels (kernels/include/sparkinfer/kernels/
// kimi_k3.h), the GGUF reader with partial-shard support (sparkinfer/gguf.h), the
// decode SCHEDULE (kimi_k3_decode_plan.h), and the tensor manifest — was built and
// validated independently, several of them against real weights or real ggml. This
// file is the executor: it walks the schedule that plan describes and issues the
// kernel calls, on real device buffers, against a real (possibly partial) GGUF.
//
// SCOPE, DELIBERATELY. This is the correctness-first pass, not the throughput pass:
//
//   - fp32 hidden state end to end, matching the already-validated K3 kernels
//     (which are all f32 in/out — see kimi_k3.h's own header comment on why this is
//     a real decision, not a shortcut: bf16 would reintroduce the precision loss
//     those kernels' float64 validation exists to eliminate).
//   - One CUDA stream, sequential kernel launches, no CUDA graph capture. The
//     existing Qwen3.5 path (runtime/src/models/qwen35.cpp) graph-captures and
//     multi-streams for throughput; K3 does not need that to be CORRECT, and
//     correctness against llama.cpp is the open question right now, not tok/s.
//   - tp_size=1 only. The decode plan already carries shard rules and reduce
//     points (kimi_k3_decode_plan.h) for when this executor grows a sharded path;
//     at tp_size=1 every rule degenerates to Replicate and there is nothing to
//     shard or reduce, so this is not a stub — it is the same code TP will run,
//     minus the collective calls this file does not yet issue.
//
// WEIGHT LOADING IS SIMPLER THAN QWEN3.5'S. Every K3 weight is read NATIVE at
// forward time — k3_proj_f32 reads Q8_0/F32 bytes directly, the MoE dispatch
// kernels read IQ1_S/IQ2_XS bytes directly — so there is no equivalent of
// Qwen3.5's load-time dequantize-to-bf16 pass. Loading is: for every tensor the
// manifest requires, cudaMalloc + cudaMemcpy the raw GGUF bytes, record the
// pointer and ggml type. That's the whole loader.

#include "sparkinfer/gguf.h"
#include "sparkinfer/models/kimi_k3_config.h"
#include "sparkinfer/models/kimi_k3_decode_plan.h"
#include "sparkinfer/tp/shard.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace sparkinfer {

// One native GGUF tensor, uploaded as-is. `type` is the ggml type id; callers that
// need to read it (k3_proj_f32, the MoE dispatch kernels, or a plain memcpy for a
// pre-dequantized F32 vector) dispatch on `type` themselves — this struct does not
// interpret the bytes.
struct KimiK3Tensor {
    const void* data = nullptr;   // device pointer, owned by KimiK3Weights
    int         type = -1;       // ggml type id; -1 = tensor absent
    long        n_bytes = 0;

    bool ok() const { return data != nullptr; }
};

struct KimiK3LayerWeights {
    bool is_kda = false;

    KimiK3Tensor attn_norm, ffn_norm;
    KimiK3Tensor attn_res_score, ffn_res_score;   // 1-D scorers, see kimi_k3_decode_plan.h

    // KDA
    KimiK3Tensor attn_q, attn_k, attn_v;
    KimiK3Tensor ssm_conv1d_q, ssm_conv1d_k, ssm_conv1d_v;   // f32, [d_conv,1,qkv]
    KimiK3Tensor ssm_f_a, ssm_f_b;
    KimiK3Tensor ssm_beta;
    KimiK3Tensor ssm_dt_bias;    // f32, 1-D [qkv]
    KimiK3Tensor ssm_a;          // f32, 1-D [n_head]
    KimiK3Tensor ssm_g;
    KimiK3Tensor ssm_norm;       // f32, 1-D [head_dim]
    KimiK3Tensor attn_output;    // KDA's wo; MLA's wo is the same field, see below

    // MLA
    bool has_q_lora = false;
    KimiK3Tensor attn_q_a, attn_q_a_norm, attn_q_b;   // q_lora path
    KimiK3Tensor attn_q_dense;    // dense-q fallback when !has_q_lora (MLA's attn_q.weight;
                                 // a distinct field from KDA's attn_q above, since a KDA
                                 // layer and an MLA layer never coexist in the same
                                 // KimiK3LayerWeights instance but the struct is shared)
    KimiK3Tensor attn_kv_a_mqa, attn_kv_a_norm;
    bool has_fused_kv_b = false;
    KimiK3Tensor attn_kv_b;                 // fused fallback (NOT numerically wired yet —
                                            // see kimi_k3_decode_plan.h's note on why the
                                            // split it needs is not derived)
    KimiK3Tensor attn_k_b, attn_v_b;         // split path (expected for the real file)
    bool has_attn_gate = false;
    KimiK3Tensor attn_gate;
    // attn_output (wo) reuses the KDA field above — both branches write the same
    // GGUF tensor name and the same [*, hidden] shape contract.

    // FFN — leading dense
    KimiK3Tensor ffn_gate, ffn_up, ffn_down;

    // FFN — routed MoE
    KimiK3Tensor ffn_gate_inp;
    KimiK3Tensor exp_probs_b;                // 1-D [n_expert] bias
    KimiK3Tensor ffn_gate_exps, ffn_up_exps, ffn_down_exps;   // native quant, per-expert
    KimiK3Tensor ffn_routed_down, ffn_routed_up;
    bool has_routed_norm = false;
    KimiK3Tensor ffn_routed_norm;
    bool has_shared_experts = false;
    KimiK3Tensor ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp;
};

struct KimiK3Weights {
    KimiK3Tensor token_embd;
    KimiK3Tensor output_norm;
    KimiK3Tensor output;             // lm_head
    bool has_output_res_score = false;
    KimiK3Tensor output_res_score;
    std::vector<KimiK3LayerWeights> layers;

    // Every device buffer this struct owns, for bulk free in the destructor.
    std::vector<void*> owned;
};

// Load every tensor a layer RANGE needs from the GGUF, uploading native bytes.
// `first_layer`/`last_layer` (inclusive) let a caller load only what a partial
// shard set actually has — see kimi_k3_layer_coverage() in
// kimi_k3_gguf_manifest.h to find out which layers that is. Layers outside the
// range are left as default-constructed (all KimiK3Tensor::ok() == false).
//
// Returns false on a real error (cudaMalloc/cudaMemcpy failure, or a required
// tensor within the requested range that turned out absent) — never on a tensor
// outside the range, since that is expected under partial loading.
bool kimi_k3_load_weights(const GGUF& g, const KimiK3Config& cfg,
                         const K3PlanOptions& opt, KimiK3Weights& out,
                         int first_layer, int last_layer);

void kimi_k3_free_weights(KimiK3Weights& w);

// ---------------------------------------------------------------------------
// Runtime state: everything that persists ACROSS decode steps and must reset at
// position 0. Three kinds, one per K3 mechanism that isn't stateless matmuls:
//
//   KDA recurrent state    conv window (x3: Q/K/V) + delta-rule state matrix,
//                           one slot per KDA layer
//   MLA KV cache            a flat [key_length, max_ctx] buffer per MLA layer —
//                           NOT a paged pool; this executor targets correctness
//                           validation against short prompts, not production
//                           context lengths. max_ctx bounds it.
//   Cross-layer residual   a growing bank of raw hidden-width checkpoints,
//   ring buffer             at most ceil(n_layers / block_size) entries for the
//                           whole model (shared across ALL layers, not per-layer)
struct KimiK3RuntimeState {
    int max_ctx = 0;
    int position = 0;   // next token's position; incremented by forward_token

    // Sizes needed to zero each buffer correctly in kimi_k3_reset_state() — stored
    // here rather than re-threading a KimiK3Config through that call. Populated by
    // kimi_k3_alloc_state().
    int conv_state_elems = 0;    // (kda_conv_kernel-1) * qkv, per KDA layer
    int delta_state_elems = 0;   // kda_head_dim^2 * n_q_heads, per KDA layer
    int kv_cache_elems = 0;      // key_length * max_ctx, per MLA layer
    int res_bank_row_elems = 0;  // hidden, per banked checkpoint

    // Indexed by KDA-layer ordinal (0..n_kda_layers-1), NOT the global layer index —
    // see kda_layer_ordinal() below.
    std::vector<float*> conv_state_q, conv_state_k, conv_state_v;   // [d_conv-1, qkv]
    std::vector<float*> delta_state;                               // [head_dim, head_dim, n_head]

    // Indexed by MLA-layer ordinal (0..n_mla_layers-1).
    std::vector<float*> mla_kv_cache;   // [key_length, max_ctx]

    // Shared across the whole model — one bank, not one per layer.
    float* res_bank = nullptr;   // [hidden, max_ckpt]
    int    n_ckpt = 0;           // live count
    int    max_ckpt = 0;

    std::vector<void*> owned;
};

bool kimi_k3_alloc_state(const KimiK3Config& cfg, int max_ctx, KimiK3RuntimeState& out);
void kimi_k3_reset_state(KimiK3RuntimeState& s);   // zero everything, position=0, n_ckpt=0
void kimi_k3_free_state(KimiK3RuntimeState& s);

// Map a global layer index to its ordinal within its own type (KDA or MLA), so
// per-type state vectors can be indexed contiguously rather than one slot per
// layer with most of them unused.
int kimi_k3_kda_ordinal(const KimiK3Config& cfg, int layer);   // -1 if layer is MLA
int kimi_k3_mla_ordinal(const KimiK3Config& cfg, int layer);   // -1 if layer is KDA

// ---------------------------------------------------------------------------
// The executor.
//
// Optional debug callback, invoked at named checkpoints with a DEVICE pointer and
// element count — mirrors the reference's `cb(tensor, "tag", il)` calls closely
// enough that a tag/layer pair here should name-match what a debug build of
// llama.cpp would report for the same graph node, which is what makes per-layer
// comparison possible before the whole model is validated end to end.
//
// Two tiers of tags. The coarse ones ("attn_res_mix", "attn_norm", "kda_out"/
// "mla_out", "ffn_norm", "ffn_out", "l_out") are the ones a real cross-check
// against llama.cpp's own graph would target. The "dbg_*" tags inside the KDA
// branch (dbg_conv_q, dbg_conv_v, dbg_l2_q, dbg_decay_g, dbg_beta, dbg_delta_out,
// dbg_gate_out) and the dense-FFN branch (dbg_dense_gate/up/situ) exist because
// bisecting kimi_k3_layer0_ref_check.cpp's first real divergence against an
// independent float64 reference needed them — the coarse tags alone narrowed the
// bug to "somewhere in the KDA branch", not to the specific step. Kept rather than
// stripped back out: the next layer this executor gets extended to validate (an
// MLA layer, or a MoE one) will need the same kind of bisection, and these cost
// nothing when fwd.debug is unset.
using KimiK3DebugFn = std::function<void(const char* tag, int layer, const float* dev_ptr, int64_t n)>;

struct KimiK3Forward {
    const KimiK3Config* cfg = nullptr;
    const KimiK3Weights* w = nullptr;
    KimiK3RuntimeState* state = nullptr;
    K3PlanOptions opt;
    cudaStream_t stream = nullptr;   // null = the default stream
    KimiK3DebugFn debug;             // may be empty

    // Scratch buffers, sized once and reused every call. Allocated by
    // kimi_k3_forward_alloc_scratch(); forward_token() assumes they exist.
    struct Scratch;
    Scratch* s = nullptr;
};

bool kimi_k3_forward_alloc_scratch(const KimiK3Config& cfg, KimiK3Forward& fwd);
void kimi_k3_forward_free_scratch(KimiK3Forward& fwd);

// Run one decode step. `out_logits` is host-visible (a cudaMemcpy is done into it)
// and must have room for cfg.vocab floats. Advances state->position. Every layer
// referenced must have its weights loaded (KimiK3Tensor::ok()) or this returns
// false rather than reading a null pointer.
bool kimi_k3_forward_token(KimiK3Forward& fwd, int token_id, float* out_logits);

// Run a single layer only, for per-layer validation against llama.cpp before the
// whole model's weights are available. `hidden_in`/`hidden_out` are device
// pointers, [hidden] each; hidden_in is NOT modified. Consumes/produces the same
// state slot forward_token would for this layer, so a sequence of single-layer
// calls at increasing positions is equivalent to forward_token calling this layer
// as part of the full loop — the two are meant to be cross-checked against each
// other, not just against llama.cpp.
bool kimi_k3_forward_layer(KimiK3Forward& fwd, int layer, const float* hidden_in,
                          float* hidden_out);

}  // namespace sparkinfer
