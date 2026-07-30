#pragma once
// Kimi K3 specific kernels.
//
// K3 is not "another MoE transformer with different numbers". Five things in it have
// no equivalent in the Qwen stack sparkinfer was built for, and each is implemented
// here with the reference semantics stated exactly, because every one of them is a
// silent-wrong-output risk rather than a crash risk:
//
//   1. situ activation           replaces SwiGLU everywhere
//   2. KDA decode step           gated delta rule, PER-CHANNEL decay
//   3. KDA output gating         rms_norm then a full-rank sigmoid gate
//   4. cross-layer attn residual softmax over banked checkpoints
//   5. MLA output gate           sigmoid before o_proj
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
// 5. MLA output gate
// ---------------------------------------------------------------------------
// Reference: src/models/kimi-k3.cpp — "K3: sigmoid output gate applied to the
// attention output before o_proj" (layer.wqkv_gate / LLM_TENSOR_ATTN_GATE).
//
// out = attn_out * sigmoid(gate_proj). Elementwise; kept separate from the KDA
// gating above because it has no rms_norm and applies to the MLA branch.
void mla_gate_out_f32(float* out, const float* attn_out, const float* gate_proj,
                      int64_t n, cudaStream_t stream);

}  // namespace k3
}  // namespace kernels
}  // namespace sparkinfer
