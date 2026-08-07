#pragma once
// Kimi K3 decode fast paths, one translation unit per factor.
//
// WHY THESE ARE NOT IN k3_kernels.cu. Each entry point below is an alternative
// implementation of something k3_kernels.cu already does correctly, kept in its own
// .cu file with its own environment toggle. Three reasons, in order of how much they
// cost when ignored:
//
//   1. THE FALLBACK MUST STAY BIT-FOR-BIT REACHABLE. Every speedup this repo has
//      merged was measured as an A/B on ONE binary, and the ones that had to be
//      retracted were the ones measured across two builds. A front-door function that
//      declines and calls the original leaves main's kernel byte-identical on the
//      other arm — there is no edit to it to get wrong.
//   2. REBASE. main merges several PRs a day, and k3_kernels.cu is where all of them
//      land. A factor that lives in its own file conflicts with none of them.
//   3. Each factor can be dropped on its own if a re-measurement on a moved main
//      turns it negative, without unpicking the others. #81 measured +4.8%, then #74
//      landed underneath it and the same change measured -31.5%.
//
// Every entry point returns bool: false means "declined, use the original", never
// "failed". A declined shape is a slower path, never a wrong one.

#include <cuda_runtime.h>

#include <cstddef>

namespace sparkinfer {
namespace kernels {
namespace k3 {

// ---------------------------------------------------------------------------
// Factor A — KDA decode step, one warp per state column
// ---------------------------------------------------------------------------
// k3_kernels.cu launches this as (n_head, head_dim/BV) blocks of BV=32 threads: at
// tp=8 that is 48 one-warp blocks on a 132-SM part, and its own header says why it
// stopped there ("splitting the i-reduction across R threads would parallelise
// further but turns a sequential sum into a tree"). This is that split.
//
// Declines (returns false) unless head_dim is a multiple of 128 and every pointer is
// 16-byte aligned, because the i-axis is read as float4.
//
// `beta_sigmoid` folds sigmoid_inplace_f32 into this kernel — pass what
// k3_kda_fuse_enabled() returns, and skip the standalone launch only when BOTH this
// returns true and that is set.
bool k3_kda_decode_step_ip(float* out, float* state,
                           const float* q, const float* k, const float* v,
                           const float* g, const float* beta,
                           int head_dim, int n_head, bool beta_sigmoid,
                           cudaStream_t stream);

// ---------------------------------------------------------------------------
// Factor G — the KDA layer's elementwise tails, folded into their consumers
// ---------------------------------------------------------------------------
// 207 launches per token per rank (the dt_bias add, the decay gate, the beta sigmoid)
// whose whole content is one operation per element, on a path where a launch costs
// ~1.5-1.9 us however small it is. Both folds are bit-identical.
//
// TOKEN AXIS. n_rows > 1 processes n_rows token-major rows of [head_dim * n_head] in one
// launch, row r at (base + r * row_stride); row_stride == 0 means contiguous, i.e.
// head_dim * n_head. `out` and `g_raw` are per-token and are strided; `dt_bias` and `A`
// are learned parameters shared by every row and are read at the same address for all of
// them. Bit-identical: the row is a new grid .y, the block, the thread-to-element map and
// the arithmetic per element are untouched. Declines above the 65535 gridDim.y cap.
bool k3_kda_fuse_enabled();
bool k3_kda_decay_gate_dt(float* out, const float* g_raw, const float* dt_bias,
                          const float* A, int head_dim, int n_head, float lower_bound,
                          cudaStream_t stream, int n_rows = 1, int64_t row_stride = 0);

// ---------------------------------------------------------------------------
// Factor C — MLA query absorption, one warp per (head, latent row)
// ---------------------------------------------------------------------------
// Replaces mla_absorb_q_f32 AND the two cudaMemcpy2DAsync de-interleaves that feed
// it: `q_proj` is q_proj_out itself, laid out [n_head, q_stride] with each head's
// qk_nope values followed by its rope_dim ones, so the split is an index rather than
// a copy. Declines unless qk_nope is 128 and both bases are 16-byte aligned.
//
// TOKEN AXIS. n_tok > 1 absorbs n_tok token-major rows in one launch: token t reads
// (q_proj + t * q_tok_stride) and writes (out + t * out_tok_stride); 0 on either means
// the natural single-token width for that buffer (n_head * q_stride for the projection,
// n_head * (kv_lora + rope_dim) for the output). `wk_b` is a WEIGHT and is never
// strided. The token is a new grid .z, so warp (h, r) of token t does exactly what
// warp (h, r) did when it was launched alone on token t's pointers — same lane map,
// same float4 loads, same shuffle tree — which is what makes it bit-identical to n_tok
// separate calls. Declines above the 65535 gridDim.z cap.
bool k3_mla_absorb_q_strided(float* out, const float* q_proj, const float* wk_b,
                             int qk_nope, int kv_lora, int rope_dim, int q_stride,
                             int n_head, cudaStream_t stream, int n_tok = 1,
                             int64_t out_tok_stride = 0, int64_t q_tok_stride = 0);

// ---------------------------------------------------------------------------
// Factor D — Q8_0 x Q8_0 multi-row GEMV with one barrier, not two per row
// ---------------------------------------------------------------------------
// Same signature and same contract as k3_proj_ggml_f32, and BIT-IDENTICAL to it:
// only the number of __syncthreads() in the epilogue changes (2 * ROWS -> 1). Handles
// the multi-row shapes only; declines everything else so the single-row kernel the
// numeric test pins keeps running unchanged.
// Q8_0 activation quantisation, one WARP per 32-value block instead of one thread.
// 859 launches per token per rank at 1-2 CUDA blocks each; this is 32x the threads and
// bit-identical (the scan is a max over MAGNITUDES, so it is order-independent).
// SPARKINFER_K3_QUANT_WARP=0 restores the reference shape.
//
// NO ROW AXIS IS NEEDED HERE, AND ADDING ONE WOULD BE DEAD WEIGHT. Both shapes are FLAT
// over 32-element blocks with no row concept: the warp kernel derives b from a global
// thread id and reads x[b*32 + lane] into out[b]; the reference kernel derives b from a
// global thread id and reads x[b*32 + j] into out[b]. Nothing indexes a row, a K, or a
// leading dimension. So a TOKEN-MAJOR [n_tok, K] f32 activation with K % 32 == 0 is
// already a flat run of n_tok * (K/32) blocks whose block t*(K/32)+c is token t's c-th
// block, and quantising all of it is one call with n_blocks = n_tok * (K/32), producing
// a token-major [n_tok, K/32] Q8_0 array. Bit-identical per block by construction — the
// same kernel over a longer range, with the same per-block magnitude max and the same
// per-element rounding. k3_quantize_act_rows_f32 (kimi_k3.h) is exactly that call.
void k3_quantize_q8_0(void* out, const float* x, int n_blocks, cudaStream_t stream);

// x_pre_q8 mirrors k3_proj_ggml_f32_x4's: q8_scratch ALREADY holds this activation, so
// skip the quantise. #94 hoists the KDA q/k/v/g quantise out of the projection, and a
// path that re-quantised would put back exactly the launch that change removes.
bool k3_proj_q8_multirow_1bar(float* y, const float* x, const void* W, int wtype,
                              int N, int K, void* q8_scratch, cudaStream_t stream,
                              bool x_pre_q8 = false);

// The four-tensor fused counterpart (the KDA q/k/v/g group). Same contract as
// k3_proj_ggml_f32_x4 and bit-identical to it; it calls block_sum 4 * ROWS times, so at
// K3's ROWS 4 it pays 32 barriers on every one of the 69 KDA layers.
bool k3_proj_q8_fused4_1bar(float* y0, float* y1, float* y2, float* y3, const float* x,
                            const void* W0, const void* W1, const void* W2, const void* W3,
                            int wtype, int N, int K, void* q8_scratch,
                            cudaStream_t stream, bool x_pre_q8 = false);

// ---------------------------------------------------------------------------
// Factor — MoE router, selection kept in shared memory
// ---------------------------------------------------------------------------
// moe_router_noaux_tc_kernel publishes each of its 16 winners straight to the caller's
// GLOBAL buffers and then re-reads all 16 back from a single thread to normalise them —
// sixteen dependent global loads at the end of a kernel that runs 92 times per token per
// rank and measures 17.8 us. This keeps the winners in shared and writes once, and folds
// the per-warp candidates with a warp reduce instead of a serial walk by thread 0.
//
// Bit-identical, including the tie-break: the comparison is a total order on
// (value desc, index asc), so the fold order cannot change the winner.
//
// Declines — returns false, caller runs the original — on SPARKINFER_K3_ROUTER_FAST=0,
// a top_k above 32, or a shared request past the 48 KiB budget.
// Q8_0 structure-of-arrays weights (k3_q8soa.cu). Register at LOAD time;
// the proj entries decline (return false) for any unregistered tensor.
bool k3_q8soa_register(const void* wdata, int wtype, const long ne[4]);
bool k3_proj_q8soa(float* y, const void* q8_act, const void* wdata,
                   int N, int K, cudaStream_t stream);
bool k3_proj_q8soa_f32(float* y, const float* x, const void* wdata,
                       int N, int K, void* q8_scratch, cudaStream_t stream);

bool k3_moe_router_fast(float* out_w, int* out_ids, const float* logits,
                        const float* bias, int n_expert, int top_k, int n_tokens,
                        bool norm_w, float w_scale, cudaStream_t stream);


// ---------------------------------------------------------------------------
// Factor — the FFN tail's two residual adds, in one launch
// ---------------------------------------------------------------------------
// Every MoE layer ends with `ffn_out += shexp_out` immediately followed by
// `hidden_out += ffn_out`, with no kernel between them: 184 launches per token per rank
// for two 28 KiB elementwise adds, each of which costs 2.95 us regardless of content.
// This does both in one launch and is bit-identical — same two operations, same order,
// same f32 intermediate. out_ab is still written because the debug hook between the two
// original calls reads it.
//
// Declines on SPARKINFER_K3_ADD3=0.
//
// TOKEN AXIS. n_rows > 1 does n_rows token-major rows of n elements in one launch, row r
// at (base + r * row_stride) for ALL FIVE pointers; row_stride == 0 means contiguous
// (== n). Every operand here is an activation — there is no shared weight to hold still.
// Bit-identical: the row is a new grid .y and nothing inside a row moves. Declines above
// the 65535 gridDim.y cap.
bool k3_add3_f32(float* out, float* out_ab, const float* a, const float* b,
                 const float* c, int64_t n, cudaStream_t stream,
                 int n_rows = 1, int64_t row_stride = 0);
// #136's separate entry point, kept alongside the row axis above rather than folded into
// it: this one strides `b` ALONE and takes a (rows, cols) shape, a different contract.
bool k3_add3_rows_f32(float* out, float* out_ab, const float* a,
                      const float* b, int64_t b_row_stride, const float* c,
                      int rows, int cols, cudaStream_t stream);

}  // namespace k3
}  // namespace kernels
}  // namespace sparkinfer
