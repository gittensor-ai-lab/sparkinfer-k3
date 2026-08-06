#pragma once
// Batched prompt-prefill primitives for Kimi K3.
//
// WHY THIS FILE EXISTS
// --------------------
// K3 has no prefill path. Every prompt token goes through the single-token decode step,
// so ingesting a prompt costs exactly what generating it costs: 32,768 tokens measured
// 812.2 s on main, and the eval now scores prefill at 32k against llama.cpp's 143.88
// tok/s (#131). Our 40.35 is 3.57x behind.
//
// The reason is not the arithmetic, it is the LOOP ORDER. Decode runs
//
//     for token: for layer: for weight tile: read W, do a GEMV
//
// so every weight in the model is re-streamed from HBM for every token. At ~22 GB of
// active weights per token per rank that is the whole cost, and it is why prefill
// tok/s and decode tok/s are the same number. Prefill inverts it:
//
//     for layer: for weight tile: read W ONCE, do a GEMM against T tokens
//
// and the weight stream is divided by T. Nothing about the math changes — only how many
// activations are in flight when each weight block is resident.
//
// BIT-IDENTICAL, BY CONSTRUCTION AND NOT BY TOLERANCE
// ---------------------------------------------------
// Every kernel here computes, for each (token, output row), the SAME float program as
// its single-token counterpart: the same thread-to-block striding, the same dp4a operand
// order, the same `(float)sumi * (dw * dx)` contraction, the same shuffle butterfly, and
// the same increasing-warp fold. A prefilled token and a decoded token therefore produce
// identical logits, which is what lets the existing parity references validate a path
// they were never captured against.
//
// That property is load-bearing rather than decorative: prefill and decode SHARE the KV
// cache. If they disagreed by even one ULP, a prompt ingested by prefill would leave a
// cache that the decode step then continues from, and the divergence would compound over
// the generation rather than showing up as a clean parity failure.

#include <cstddef>
#include <cuda_runtime.h>

namespace sparkinfer {
namespace kernels {
namespace k3 {

// Bytes for T tokens of Q8_0-quantised activation at width K. K must be a multiple of 32.
size_t k3_prefill_act_q8_bytes(int K, int n_tok);

// Quantise T activation rows to Q8_0. `x` is [n_tok][K] f32, row-major; `q8_out` receives
// [n_tok][K/32] BlockQ8_0. Per row this is byte-for-byte what k3_quantize_act_f32 writes
// for that row alone -- amax/127, __float2int_rn, no sign rule -- so the two are
// interchangeable and a mixed prefill/decode sequence stays exact.
bool k3_prefill_quantize_act(void* q8_out, const float* x, int K, int n_tok,
                             cudaStream_t stream);

// Y[n_tok][N] = X_q8[n_tok][K] . W[N][K], Q8_0 weights against Q8_0 activations.
//
// The weight block is loaded ONCE per (block, output row) and multiplied into TOKS
// resident activations, so weight traffic falls by the token-tile width rather than by
// the batch size -- see k3_prefill_proj_token_tile(). `wtype` must be 8 (Q8_0); anything
// else returns false so the caller falls back rather than reading a format this was not
// written for.
//
// Returns false (never an error) for shapes outside its contract: the caller should treat
// that as "use the per-token path".
bool k3_prefill_proj_q8act(float* y, const void* q8_acts, const void* W, int wtype,
                           int N, int K, int n_tok, cudaStream_t stream);

// The token-tile width the projection will actually use for this shape. Exposed so a
// caller can reason about the amortisation it is buying (weight traffic is divided by
// this, not by n_tok) and so a bench can sweep it. 0 means the shape is declined.
int k3_prefill_proj_token_tile(int N, int K, int n_tok);

}  // namespace k3
}  // namespace kernels
}  // namespace sparkinfer
