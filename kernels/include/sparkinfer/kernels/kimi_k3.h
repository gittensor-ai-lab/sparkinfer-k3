#pragma once
// Kimi K3 specific kernels.
//
// K3 is not "another MoE transformer with different numbers". The ops below have
// no equivalent in the Qwen stack sparkinfer was built for (or a subtly different
// form of one), and each is implemented with the reference semantics stated
// exactly, because every one of them is a silent-wrong-output risk rather than a
// crash risk:
//
//   1. situ activation           replaces SwiGLU everywhere
//   2. KDA decode step           gated delta rule, PER-CHANNEL decay
//   3. KDA output gating         rms_norm then a full-rank sigmoid gate
//   4. cross-layer attn residual softmax over banked checkpoints
//   5. MLA output gate           sigmoid before o_proj
//   6. KDA causal short conv     depthwise, silu AFTER, current token at END
//   7. KDA decay gate            lower_bound*sigmoid form (NOT softplus)
//   8. L2-norm heads             per-head; optional 1/sqrt scale for Q
//   9. MLA absorb Q              wk_b @ q_nope, concat q_pe → key_length
//  10. MLA NoPE decode attn      MQA over K-cache + wv_b decompress
//
// SEMANTICS ARE TRANSCRIBED, NOT INFERRED. Each function below cites the reference
// implementation it must match (unslothai/llama.cpp @ kimi-k3-fullsize-vision,
// commit efc8bc38). A kernel that is merely plausible here produces fluent wrong
// text, so kernels/tests/kimi_k3_numeric_test.cu checks every one of them against an
// independent float64 CPU implementation of the same formula.

#include <cstdint>
#include <cuda_runtime.h>

namespace sparkinfer {
namespace kernels {
namespace k3 {

// ---------------------------------------------------------------------------
// 1. situ activation
// ---------------------------------------------------------------------------
// Reference: src/models/kimi-k3.cpp kimi_k3_situ()
//
//   situ(gate, up) = beta*tanh(gate/beta) * sigmoid(gate) * lb*tanh(up/lb)
//
// linear_beta (lb) <= 0 disables the transform on the up branch, in which case the
// up value is used raw. K3 ships beta = 4.0, linear_beta = 25.0.
//
// NOTE the asymmetry: the gate branch gets BOTH a scaled-tanh and a sigmoid; the up
// branch gets only the scaled-tanh. Treating this as a symmetric "tanh both sides"
// or as SwiGLU-with-tanh is the obvious wrong guess and it stays numerically close
// enough to look like it works.
void situ_f32(float* out, const float* gate, const float* up, int64_t n,
              float beta, float linear_beta, cudaStream_t stream);

// ---------------------------------------------------------------------------
// 2. KDA decode step (gated delta rule, one token)
// ---------------------------------------------------------------------------
// Reference: src/models/delta-net-base.cpp build_delta_net_autoregressive()
//
// Per head, with state S indexed [i][j] (i = ne0 = fastest), all dims = head_dim:
//
//   S[i][j] *= exp(g[j])              per-CHANNEL decay. GDA decays by a scalar;
//                                     KDA has a full head_dim-wide g. Collapsing it
//                                     to a scalar is a wrong model that still runs.
//   sk[j]    = sum_i S[i][j]*k[i]
//   d[m]     = beta * (v[m] - sk[m])
//   S[i][j] += k[i]*d[j]
//   o[j]     = sum_i S[i][j]*q[i]
//
// q must ALREADY be scaled by 1/sqrt(head_dim) and q,k must ALREADY be L2-normalised
// by the caller — kimi-k3.cpp does both before calling the scan, so doing them again
// here would double-apply. state is updated in place.
//
// Layout: q,k,v,g are [head_dim, n_head]; beta is [n_head]; state is
// [head_dim, head_dim, n_head] with i fastest; out is [head_dim, n_head].
void kda_decode_step_f32(float* out, float* state,
                         const float* q, const float* k, const float* v,
                         const float* g, const float* beta,
                         int head_dim, int n_head, cudaStream_t stream);

// ---------------------------------------------------------------------------
// 3. KDA output gating
// ---------------------------------------------------------------------------
// Reference: src/models/kimi-k3.cpp, the tail of the KDA branch:
//     normed = rms_norm(o, ssm_o_norm);  gated = normed * sigmoid(g2)
//
// The RMS norm is per (head, token) over head_dim, with a learned weight, and the
// gate is FULL-RANK in K3 (a single ssm_g projection) where kimi-linear factors it
// as g_b(g_a(x)). g2 is the raw projection output; the sigmoid is applied here.
void kda_gate_out_f32(float* out, const float* o, const float* norm_w,
                      const float* g2, int head_dim, int n_head,
                      float eps, cudaStream_t stream);

// ---------------------------------------------------------------------------
// 4. cross-layer attention residual mix
// ---------------------------------------------------------------------------
// Reference: src/models/kimi-k3.cpp llama_model_kimi_k3::graph::res_mix()
//
// Every attn_res.block_size (12) layers, the residual stream is mixed with a softmax
// over previously banked checkpoints:
//
//   score_i = sum_d rms_norm(ckpt_i)[d] * score_w[d]     for each banked ckpt
//   score_c = sum_d rms_norm(cur)[d]    * score_w[d]
//   p       = softmax([score_0..score_{n-1}, score_c])
//   out     = sum_i p_i * ckpt_i_RAW + p_c * cur_RAW
//
// THE TRAP: the scores are computed from the NORMALISED values but the weighted sum
// uses the RAW ones. Using normalised values in the sum (the natural thing to write)
// changes the residual stream's scale on every mix layer and compounds over 93
// layers. The reference comments this explicitly; it is transcribed, not chosen.
//
// ckpts is [n_embd, n_ckpt] with n_embd fastest; cur and out are [n_embd].
void attn_res_mix_f32(float* out, const float* ckpts, const float* cur,
                      const float* score_w, int n_embd, int n_ckpt,
                      float eps, cudaStream_t stream);

// ---------------------------------------------------------------------------
// 6. KDA causal short conv, decode step
// ---------------------------------------------------------------------------
// Reference: src/models/kimi-k3.cpp kimi_k3_conv1d() + ggml_ssm_conv.
//
// K3 runs a depthwise causal conv (short_conv_kernel_size = 4) over EACH of Q, K and
// V before the delta-net scan. For one decoded token, per channel c:
//
//   window[t] = state[t][c]  for t in [0, d_conv-1),  window[d_conv-1] = x[c]
//   out[c]    = silu( sum_t window[t] * w[t][c] )
//   state[t][c] <- window[t+1]        for t in [0, d_conv-1)     (shift left by one)
//
// Two things to get right:
//   - The new sample is appended at the END of the window (ggml_concat puts the
//     transposed x after the state), so w[d_conv-1] multiplies the CURRENT token.
//     Reversing the window silently gives a different, still-plausible filter.
//   - The conv weight is PER CHANNEL: [d_conv, d_inner], not shared across channels.
//   - silu is applied AFTER the convolution, not before.
//
// Q, K and V each own a distinct third of the recurrent conv state
// (n_embd_r = 3 * (d_conv-1) * d_inner), so this is called three times per layer with
// different state/weight pointers. state is updated in place.
//
// x, out are [d_inner]; state is [d_conv-1, d_inner] with the time index fastest;
// w is [d_conv, d_inner] with the time index fastest.
void kda_conv_step_f32(float* out, float* state, const float* x, const float* w,
                       int d_conv, int d_inner, cudaStream_t stream);

// ---------------------------------------------------------------------------
// 11. IQ2_XS dequantisation
// ---------------------------------------------------------------------------
// Reference: ggml dequantize_row_iq2_xs() + block_iq2_xs in ggml-common.h.
//
// THIS IS THE ONE THAT UNBLOCKS EVERYTHING. Kimi K3's UD-Q2_K_XL keeps all 896
// routed experts in IQ2_XS (ggml type 17) — 744.5 GiB of the model's 802.1 GiB.
// The rest of sparkinfer's quant kernels cover Q4_K/Q5_K/Q6_K/Q8_0/Q8_1 only, so
// until this exists the runtime can read the attention and norms and none of the
// weights that do the work.
//
// Block layout (74 bytes / 256 values, 2.3125 bpw):
//     ggml_half d;                 // fp16 super-scale
//     uint16_t  qs[32];            // per 8 values: low 9 bits = grid index,
//                                  //               high 7 bits = sign index
//     uint8_t   scales[8];         // two 4-bit sub-scales per 32-value group
//
// Per 32-value group ib32, the two sub-scales are
//     db[0] = d * (0.5 + (scales[ib32] & 0xf)) * 0.25
//     db[1] = d * (0.5 + (scales[ib32] >> 4)) * 0.25
// and l = 0,1 use db[0] while l = 2,3 use db[1]  (db[l/2] — NOT db[l&1]).
//
// Each qs entry expands to 8 values: an 8-byte lattice point from iq2xs_grid,
// scaled, with per-element sign flips from ksigns_iq2xs/kmask_iq2xs.
//
// The lattice and sign tables are EXTRACTED, not retyped — see iq2xs_tables.h.
//
// n must be a multiple of 256. `src` is the packed block stream, `out` the
// dequantised f32 values.
void dequant_iq2_xs_f32(float* out, const void* src, int64_t n, cudaStream_t stream);

// ---------------------------------------------------------------------------
// 12. noaux_tc MoE router (sigmoid + expert bias + top-k + renormalise)
// ---------------------------------------------------------------------------
// Reference: llm_graph_context::build_moe_ffn() in src/llama-graph.cpp, with
// K3's settings: gating_op = SIGMOID, exp_probs_b present, norm_w = true,
// w_scale = routed_scaling_factor = 1.0, n_expert_groups = 1 (so the DeepSeek
// group-masking path is skipped entirely).
//
//   probs           = sigmoid(logits)                 [n_expert]
//   selection_probs = probs + exp_probs_b             bias, for SELECTION ONLY
//   selected        = top_k(selection_probs)          [top_k] expert ids
//   weights         = probs[selected]                 <- UNBIASED probs
//   weights        /= clamp(sum(weights), 6.103515625e-5, INF)
//   weights        *= w_scale   (skipped when 1.0)
//
// THE TRAP, and the reference calls it out in a comment: the bias steers WHICH
// experts are chosen but must NOT appear in the weights they are combined with.
// Using selection_probs for both is the obvious implementation and it is wrong —
// every expert's contribution is then scaled by its routing bias, which shifts the
// output smoothly rather than breaking it. Fluent, plausible, wrong.
//
// The clamp constant is the smallest normal F16 (6.103515625e-5), not an epsilon
// someone picked; it is there so a token that routes to 16 near-zero experts does
// not divide by ~0.
//
// Ties break toward the lower expert index, matching a stable descending sort.
//
// logits/probs are [n_expert] per token; out_w and out_ids are [top_k] per token.
// bias may be null. n_tokens tokens are processed independently.
void moe_router_noaux_tc_f32(float* out_w, int* out_ids, const float* logits,
                             const float* bias, int n_expert, int top_k,
                             int n_tokens, bool norm_w, float w_scale,
                             cudaStream_t stream);

// ---------------------------------------------------------------------------
// 5. MLA output gate
// ---------------------------------------------------------------------------
// Reference: src/models/kimi-k3.cpp — "K3: sigmoid output gate applied to the
// attention output before o_proj" (layer.wqkv_gate / LLM_TENSOR_ATTN_GATE).
//
// out = attn_out * sigmoid(gate_proj). Elementwise; kept separate from the KDA
// gating above because it has no rms_norm and applies to the MLA branch.
void mla_gate_out_f32(float* out, const float* attn_out, const float* gate_proj,
                      int64_t n, cudaStream_t stream);

// ---------------------------------------------------------------------------
// 7. KDA decay gate (K3 lower_bound form)
// ---------------------------------------------------------------------------
// Reference: src/models/kimi-k3.cpp build_kda_layer(), gated by
// linear_attn_config.gate_lower_bound. K3 ships lower_bound = -5.0, which
// SWAPS the activation entirely vs kimi-linear:
//
//   kimi-linear (unset):  g = A * softplus(g_raw)          A = -exp(A_log)
//   K3 (lower_bound set): g = lb * sigmoid(-(A * g_raw))
//                       = lb * sigmoid( exp(A_log) * g_raw )
//
// g_raw is already f_b(f_a(x)) + dt_bias — the GEMMs stay outside this kernel.
// A is per-head (shape [n_head]); it broadcasts across head_dim. ssm_a in the
// GGUF already holds -exp(A_log) (folded at conversion).
//
// THE TRAP: using softplus here is the kimi-linear path. It still produces a
// plausible negative decay and the model stays fluent.
//
// g_raw, out: [head_dim, n_head]; A: [n_head].
void kda_decay_gate_f32(float* out, const float* g_raw, const float* A,
                        int head_dim, int n_head, float lower_bound,
                        cudaStream_t stream);

// ---------------------------------------------------------------------------
// 8. L2-norm over heads (+ optional scale)
// ---------------------------------------------------------------------------
// Reference: ggml_l2_norm on Q/K before build_delta_net, then
// build_delta_net_autoregressive scales q by 1/sqrt(head_dim). Combined here so
// the KDA micro-path is one launch: pass scale=1/sqrt(D) for Q, scale=1 for K.
//
//   out[h,d] = x[h,d] * scale / sqrt(sum_d x[h,d]^2 + eps)
//
// In-place is fine (out may alias x). Layout [head_dim, n_head].
void l2_norm_heads_f32(float* out, const float* x, int head_dim, int n_head,
                       float scale, float eps, cudaStream_t stream);

// ---------------------------------------------------------------------------
// 9. MLA absorb Q (wk_b @ q_nope, concat q_pe)
// ---------------------------------------------------------------------------
// Reference: src/models/kimi-k3.cpp build_mla_layer() absorbed path
// (layer.wk_b && layer.wv_b):
//
//   q_nope_absorbed[h] = wk_b[h] @ q_nope[h]     // [kv_lora] <- [qk_nope]
//   Q[h]               = concat(q_nope_absorbed[h], q_pe[h])  // length key_length
//
// GGUF layout of wk_b is [qk_nope, kv_lora, n_head] with qk_nope fastest —
// per head: absorbed[r] = sum_d wk_b[d + r*qk_nope + h*qk_nope*kv_lora] * q_nope[d].
//
// q_nope: [qk_nope, n_head]; q_pe: [rope_dim, n_head];
// wk_b:   [qk_nope, kv_lora, n_head];
// out:    [key_length, n_head] where key_length = kv_lora + rope_dim (= 576).
void mla_absorb_q_f32(float* out, const float* q_nope, const float* q_pe,
                      const float* wk_b, int qk_nope, int kv_lora, int rope_dim,
                      int n_head, cudaStream_t stream);

// ---------------------------------------------------------------------------
// 10. MLA NoPE decode attention (MQA over K-cache + wv_b decompress)
// ---------------------------------------------------------------------------
// Reference: build_mla_layer absorbed path + build_attn(inp_attn_k, ..., wv_b).
// K3 is NoPE-only; the K cache stores concat(kv_cmpr, k_pe) of length
// key_length = kv_lora + rope_dim, V is the leading kv_lora of that same cache,
// and wv_b decompresses the attended latent back to v_dim per head.
//
// Per head h:
//   score[t]  = scale * sum_d q[h,d] * k_cache[t,d]     d in [0, key_length)
//   p         = softmax(score[0..n_ctx))
//   latent[r] = sum_t p[t] * k_cache[t,r]               r in [0, kv_lora)
//   out[h,v]  = sum_r wv_b[r,v,h] * latent[r]           v in [0, v_dim)
//
// THE TRAP: kq_scale is 1/sqrt(n_embd_head_k_mla) = 1/sqrt(qk_nope+rope) =
// 1/sqrt(192), NOT 1/sqrt(key_length=576). Scaling by the cached key width is
// the natural wrong guess and shifts every attention distribution.
//
// q:       [key_length, n_head]     (output of mla_absorb_q_f32)
// k_cache: [key_length, n_ctx]      token-major rows; MQA (one K shared by heads)
// wv_b:    [kv_lora, v_dim, n_head]  kv_lora fastest
// out:     [v_dim, n_head]
void mla_decode_attn_f32(float* out, const float* q, const float* k_cache,
                         const float* wv_b, int key_length, int kv_lora,
                         int v_dim, int n_head, int n_ctx, float scale,
                         cudaStream_t stream);

}  // namespace k3
}  // namespace kernels
}  // namespace sparkinfer
