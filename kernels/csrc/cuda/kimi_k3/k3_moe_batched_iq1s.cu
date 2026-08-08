// ===========================================================================
// BATCHED IQ1_S MoE EXPERT FFN — the consumer that gives the tensor-core GEMM an M.
// ===========================================================================
// k3_moe_iq1s_mma.cu establishes that IQ1_S is exactly an int8 format and that its
// 32-value sub-block is exactly mma.m16n8k32.s8's k-extent. That kernel is a bare
// GEMM: it needs a caller that hands it M rows of the SAME expert at once. This file
// is that caller.
//
// The shipped decode op (moe_expert_ffn_iq1s_f32) is one token against top_k experts,
// so every expert sees M=1 and every GEMM is a GEMV. Batching inverts the loop: T
// tokens are bucketed BY EXPERT, and each expert runs once over all the rows routed to
// it. The expert's weights — 6.45 MB at K3's dims — are then read once for M rows
// instead of once per row, which is the entire prize. Measured on the bare GEMM, the
// per-token cost of that read falls 254 -> 4.7 us as M goes 1 -> 128.
//
// ---------------------------------------------------------------------------
// WHY THE BUCKETING IS A COUNTING SORT AND NOT A MASK
// ---------------------------------------------------------------------------
// The obvious batched form keeps the [T, top_k] shape and masks: run every expert over
// every token and throw away the rows it did not win. At 896 experts and top_k 16 that
// computes 56x the work it keeps, which is worse than the GEMV it replaced. The rows
// have to be physically gathered, so the GEMM sees a dense M.
//
// The gather is a two-pass counting sort over the (token, slot) pairs — count into
// per-expert bins, exclusive-scan to offsets, then place. It produces a CSR whose
// row-index array is exactly the `rows` gather list k3_moe_iq1s_mma_gemm already takes,
// so no extra copy of the activations is made: the GEMM reads scattered rows of the
// quantized activation directly out of shared-memory staging.
//
// ---------------------------------------------------------------------------
// THIS IS NOT BIT-IDENTICAL TO THE SCALAR PATH, AND CANNOT BE
// ---------------------------------------------------------------------------
// Two reasons, both structural rather than sloppy:
//
//   1. The activation is quantized to int8 with a per-32 scale. The scalar path uses
//      block_q8_K — int8 with a per-256 scale — so both quantize, but not identically.
//      (llama.cpp's own vec_dot contract for IQ1_S also quantizes the activation, so
//      this is a different int8, not a new approximation. Per-32 is FINER than the
//      per-256 it is graded against, which is why this term shrank when the scale
//      granularity moved off per-row.)
//   2. The top_k combine accumulates with atomicAdd, because experts are processed in
//      bucket order and a token's top_k contributions land from different launches.
//      The scalar path sums k ascending. Same terms, different order.
//
// The weights themselves are NOT re-rounded — that is the whole point of the IQ1_S
// identity — so what moves is the activation quantization and the summation order,
// which is what the parity gate exists to adjudicate.
//
// ---------------------------------------------------------------------------
// WHAT THIS STILL DOES NOT DO
// ---------------------------------------------------------------------------
// Nothing calls this yet either. It is the MoE half of a batched prefill; the other
// halves — a chunked MLA attention over T causal queries, the KDA recurrent scan in
// chunk-parallel form at K3's dims, and TP collectives at [T, hidden] — do not exist,
// and without them there is no producer of a [T, latent] activation block to feed it.
// It is landed at this size because it is independently testable against the shipped
// scalar op (kernels/tests/k3_moe_batched_iq1s_gpu_test.cu runs both over the same
// tokens and the same routing), which the rest of the prefill is not until it is whole.

#include "sparkinfer/kernels/kimi_k3.h"
#include "k3_pdl.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <vector>

namespace sparkinfer {
namespace kernels {
namespace k3 {
namespace {

constexpr int kQK = 256;

// One warp owns one local expert. It scans pair indices in increasing order, so the
// compact rows within an expert are deterministic without atomics. Counting IDs below
// the owned expert gives its exclusive CSR offset directly, following llama.cpp's
// device MMID strategy; repeating this small router scan is cheaper than introducing
// a host-visible count or a separate device scan into every captured MoE layer.
__global__ void moe_build_csr_kernel(int* __restrict__ rows,
                                     int* __restrict__ slots,
                                     int* __restrict__ inverse,
                                     int* __restrict__ bounds,
                                     const int* __restrict__ ids,
                                     int P, int top_k, int expert_begin,
                                     int n_local) {
    k3_pdl_sync();
    const int e = (int)blockIdx.x;
    const int lane = threadIdx.x;
    const int global_e = expert_begin + e;
    int own = 0, lower = 0;
    for (int p = lane; p < P; p += 32) {
        const int id = ids[p] - expert_begin;
        own   += (id == e);
        lower += (id >= 0 && id < e);
    }
#pragma unroll
    for (int d = 16; d; d >>= 1) {
        own   += __shfl_down_sync(0xffffffffu, own, d);
        lower += __shfl_down_sync(0xffffffffu, lower, d);
    }
    const int base = __shfl_sync(0xffffffffu, lower, 0);
    const int count = __shfl_sync(0xffffffffu, own, 0);
    if (lane == 0) {
        bounds[e] = base;
        if (e + 1 == n_local) bounds[n_local] = base + count;
    }

    int cursor = 0;
    for (int p0 = 0; p0 < P; p0 += 32) {
        const int p = p0 + lane;
        const bool hit = p < P && ids[p] == global_e;
        const unsigned mask = __ballot_sync(0xffffffffu, hit);
        if (hit) {
            const int at = base + cursor + __popc(mask & ((1u << lane) - 1u));
            rows[at] = p / top_k;
            slots[at] = p % top_k;
            inverse[p] = at;
        }
        cursor += __popc(mask);
    }
}

// Bin the (token, slot) pairs this rank owns. `ids` holds GLOBAL expert indices — every
// rank's router sees the same replicated ffn_gate_inp — and a rank stores only
// [expert_begin, expert_begin + n_local). Out-of-band selections belong to another rank
// and contribute nothing here; the all-reduce that follows sums the bands back.
__global__ void moe_count_kernel(int* __restrict__ counts,
                                 const int* __restrict__ ids,
                                 int T, int top_k, int expert_begin, int n_local) {
    k3_pdl_sync();
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= T * top_k) return;
    const int e = ids[p] - expert_begin;
    if (e < 0 || e >= n_local) return;
    atomicAdd(&counts[e], 1);
}

// Place each pair at its expert's cursor. The resulting `tok` array IS the gather list
// k3_moe_iq1s_mma_gemm takes as `rows`; `slot` remembers which of the token's top_k
// this was, so the combine can find the routing weight.
__global__ void moe_place_kernel(int* __restrict__ tok, int* __restrict__ slot,
                                 int* __restrict__ cursor,
                                 const int* __restrict__ ids,
                                 int T, int top_k, int expert_begin, int n_local) {
    k3_pdl_sync();
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= T * top_k) return;
    const int e = ids[p] - expert_begin;
    if (e < 0 || e >= n_local) return;
    const int at = atomicAdd(&cursor[e], 1);
    tok[at]  = p / top_k;
    slot[at] = p % top_k;
}

// situ, applied over a whole [M, ffn] tile at once.
//
// Transcribed from moe_gate_up_situ's tail rather than reused from situ_f32, because
// this needs the SAME expression on a 2-D tile and the shipped helper takes a flat
// length — the formula is the asymmetric one (gate gets scaled-tanh AND sigmoid, up
// gets only scaled-tanh), and getting it symmetric is the classic silent-wrong here.
__global__ void moe_situ_tile_kernel(float* __restrict__ h,
                                     const float* __restrict__ gate,
                                     const float* __restrict__ up,
                                     int64_t n, float beta, float lb) {
    k3_pdl_sync();
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i], u = up[i];
    const float a  = beta * tanhf(g / beta) * (1.0f / (1.0f + __expf(-g)));
    const float ub = (lb > 0.0f) ? (lb * tanhf(u / lb)) : u;
    h[i] = a * ub;
}

// Scatter this expert's [M, latent] output into the token rows, weighted by the
// routing weight of the slot each row came from. atomicAdd because a token's top_k
// contributions arrive from different experts and therefore different launches — see
// the bit-identity note in the file header.
__global__ void moe_scatter_add_kernel(float* __restrict__ out,
                                       const float* __restrict__ down,
                                       const int* __restrict__ tok,
                                       const int* __restrict__ slot,
                                       const float* __restrict__ w,
                                       int M, int latent, int top_k, int base) {
    k3_pdl_sync();
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)M * latent) return;
    const int r = (int)(i / latent), c = (int)(i % latent);
    const int t = tok[base + r];
    const float wt = w[(size_t)t * top_k + slot[base + r]];
    atomicAdd(&out[(size_t)t * latent + c], wt * down[i]);
}

// Convert gate/up from compact CSR order to the compact hidden activation. Launching
// over the original pair index makes the inverse map the only dispatch metadata this
// stage needs and gives it a fixed graph shape.
__global__ void moe_situ_compact_kernel(float* __restrict__ h,
                                        const float* __restrict__ gate,
                                        const float* __restrict__ up,
                                        const int* __restrict__ inverse,
                                        int P, int ffn, float beta, float lb) {
    k3_pdl_sync();
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)P * ffn) return;
    const int p = (int)(i / ffn), j = (int)(i % ffn);
    const int at = inverse[p];
    if (at < 0) return;
    const float g = gate[(size_t)at * ffn + j];
    const float u = up[(size_t)at * ffn + j];
    const float a = beta * tanhf(g / beta) * (1.0f / (1.0f + __expf(-g)));
    const float ub = lb > 0.0f ? lb * tanhf(u / lb) : u;
    h[(size_t)at * ffn + j] = a * ub;
}

// One owner per output cell, with slots accumulated in their original top-k order.
// This avoids the nondeterministic atomic combine used by the eager batching prototype.
__global__ void moe_combine_compact_kernel(float* __restrict__ out,
                                           const float* __restrict__ down,
                                           const int* __restrict__ inverse,
                                           const float* __restrict__ weights,
                                           int T, int latent, int top_k) {
    k3_pdl_sync();
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)T * latent) return;
    const int t = (int)(i / latent), c = (int)(i % latent);
    float sum = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        const int p = t * top_k + k;
        const int at = inverse[p];
        if (at >= 0) sum += weights[p] * down[(size_t)at * latent + c];
    }
    out[i] = sum;
}

__global__ void moe_situ_pairs_kernel(float* __restrict__ h,
                                      const float* __restrict__ gate,
                                      const float* __restrict__ up,
                                      const int* __restrict__ ids,
                                      int P, int ffn, int expert_begin,
                                      int n_local, float beta, float lb) {
    k3_pdl_sync();
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)P * ffn) return;
    const int p = (int)(i / ffn), j = (int)(i % ffn);
    const int e = ids[p] - expert_begin;
    if (e < 0 || e >= n_local) return;
    const float g = gate[(size_t)p * ffn + j];
    const float u = up[(size_t)p * ffn + j];
    const float a = beta * tanhf(g / beta) * (1.0f / (1.0f + __expf(-g)));
    const float ub = lb > 0.0f ? lb * tanhf(u / lb) : u;
    h[(size_t)p * ffn + j] = a * ub;
}

__global__ void moe_combine_pairs_kernel(float* __restrict__ out,
                                         const float* __restrict__ down,
                                         const int* __restrict__ ids,
                                         const float* __restrict__ weights,
                                         int T, int latent, int top_k,
                                         int expert_begin, int n_local) {
    k3_pdl_sync();
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)T * latent) return;
    const int t = (int)(i / latent), c = (int)(i % latent);
    float sum = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        const int p = t * top_k + k;
        const int e = ids[p] - expert_begin;
        if (e >= 0 && e < n_local)
            sum += weights[p] * down[(size_t)p * latent + c];
    }
    out[i] = sum;
}

struct Scratch {
    void* p = nullptr;
    size_t n = 0;
    bool ok = true;
    cudaStream_t s = nullptr;
    template <class T> T* take(size_t count) {
        if (!ok || count == 0) return nullptr;
        void* q = nullptr;
        if (cudaMallocAsync(&q, count * sizeof(T), s) != cudaSuccess) { ok = false; return nullptr; }
        bufs.push_back(q);
        return static_cast<T*>(q);
    }
    std::vector<void*> bufs;
    void free_all() { for (void* b : bufs) cudaFreeAsync(b, s); bufs.clear(); }
};

}  // namespace

bool k3_moe_build_expert_csr(int* rows, int* slots, int* inverse,
                             int* expert_bounds, const int* ids,
                             int T, int top_k, int expert_begin,
                             int n_local_experts, cudaStream_t stream) {
    if (!rows || !slots || !inverse || !expert_bounds || !ids || T <= 0 ||
        top_k <= 0 || n_local_experts <= 0) return false;
    const size_t P = (size_t)T * top_k;
    if (P > (size_t)INT32_MAX) return false;
    if (cudaMemsetAsync(inverse, 0xff, P * sizeof(int), stream) != cudaSuccess)
        return false;
    k3_pdl_launch((unsigned)n_local_experts, 32, 0, stream, moe_build_csr_kernel,
                  rows, slots, inverse, expert_bounds, ids, (int)P, top_k,
                  expert_begin, n_local_experts);
    return true;
}

bool k3_moe_situ_compact(float* h, const float* gate, const float* up,
                         const int* inverse, int T, int top_k, int ffn,
                         float situ_beta, float situ_linear_beta,
                         cudaStream_t stream) {
    if (!h || !gate || !up || !inverse || T <= 0 || top_k <= 0 || ffn <= 0)
        return false;
    const int P = T * top_k;
    if (cudaMemsetAsync(h, 0, (size_t)P * ffn * sizeof(float), stream) != cudaSuccess)
        return false;
    const int64_t n = (int64_t)P * ffn;
    k3_pdl_launch((unsigned)((n + 255) / 256), 256, 0, stream,
                  moe_situ_compact_kernel, h, gate, up, inverse, P, ffn,
                  situ_beta, situ_linear_beta);
    return true;
}

bool k3_moe_combine_compact(float* out, const float* down,
                            const int* inverse, const float* weights,
                            int T, int latent, int top_k, cudaStream_t stream) {
    if (!out || !down || !inverse || !weights || T <= 0 || latent <= 0 || top_k <= 0)
        return false;
    const int64_t n = (int64_t)T * latent;
    k3_pdl_launch((unsigned)((n + 255) / 256), 256, 0, stream,
                  moe_combine_compact_kernel, out, down, inverse, weights,
                  T, latent, top_k);
    return true;
}

bool k3_moe_situ_pairs(float* h, const float* gate, const float* up,
                       const int* ids, int pairs, int ffn,
                       int expert_begin, int n_local_experts,
                       float situ_beta, float situ_linear_beta,
                       cudaStream_t stream) {
    if (!h || !gate || !up || !ids || pairs <= 0 || ffn <= 0 ||
        n_local_experts <= 0) return false;
    if (cudaMemsetAsync(h, 0, (size_t)pairs * ffn * sizeof(float), stream) != cudaSuccess)
        return false;
    const int64_t n = (int64_t)pairs * ffn;
    k3_pdl_launch((unsigned)((n + 255) / 256), 256, 0, stream,
                  moe_situ_pairs_kernel, h, gate, up, ids, pairs, ffn,
                  expert_begin, n_local_experts, situ_beta, situ_linear_beta);
    return true;
}

bool k3_moe_combine_pairs(float* out, const float* down, const int* ids,
                          const float* weights, int T, int latent, int top_k,
                          int expert_begin, int n_local_experts,
                          cudaStream_t stream) {
    if (!out || !down || !ids || !weights || T <= 0 || latent <= 0 ||
        top_k <= 0 || n_local_experts <= 0) return false;
    const int64_t n = (int64_t)T * latent;
    k3_pdl_launch((unsigned)((n + 255) / 256), 256, 0, stream,
                  moe_combine_pairs_kernel, out, down, ids, weights,
                  T, latent, top_k, expert_begin, n_local_experts);
    return true;
}

bool k3_moe_expert_ffn_batched_iq1s(float* out, const float* x,
                                    const int* ids, const float* w,
                                    const void* gate_exps, const void* up_exps,
                                    const void* down_exps,
                                    int T, int latent, int ffn, int top_k,
                                    float situ_beta, float situ_linear_beta,
                                    int expert_begin, int n_local_experts,
                                    cudaStream_t stream) {
    if (T <= 0 || latent <= 0 || ffn <= 0 || top_k <= 0) return false;
    if ((latent % kQK) || (ffn % kQK)) return false;      // the GEMM has no partial-block form
    if (n_local_experts <= 0) return false;               // caller must state the band

    Scratch sc; sc.s = stream;

    // The output is WRITTEN, not accumulated into: a token whose every selection fell
    // outside this rank's band must read as exactly zero, not as whatever the buffer
    // held. The scalar path gets this from a memset for the same reason.
    if (cudaMemsetAsync(out, 0, (size_t)T * latent * sizeof(float), stream) != cudaSuccess)
        return false;

    // 1. Quantize the activation once for the whole chunk. Per-ROW scales, which is
    //    what lets the GEMM hoist the scale out of its k loop entirely.
    // sx is [T][latent/32] now, not [T]: the quantizer carries a scale per 32 values so
    // the GEMM can apply it at the same drain boundary it already applies the weight
    // scale at. See the granularity note in k3_moe_iq1s_mma.cu.
    signed char* xq = sc.take<signed char>((size_t)T * latent);
    float*       sx = sc.take<float>((size_t)T * (latent / 32));
    if (!sc.ok) { sc.free_all(); return false; }
    if (!k3_moe_iq1s_mma_quantize_rows(xq, sx, x, T, latent, stream)) { sc.free_all(); return false; }

    // 2. Bucket the (token, slot) pairs by expert.
    const int P = T * top_k;
    int* counts = sc.take<int>((size_t)n_local_experts);
    int* cursor = sc.take<int>((size_t)n_local_experts);
    int* tok    = sc.take<int>((size_t)P);
    int* slot   = sc.take<int>((size_t)P);
    if (!sc.ok) { sc.free_all(); return false; }
    cudaMemsetAsync(counts, 0, (size_t)n_local_experts * sizeof(int), stream);

    const int TPB = 256, blocks = (P + TPB - 1) / TPB;
    k3_pdl_launch(blocks, TPB, 0, stream, moe_count_kernel, counts, ids, T, top_k,
                  expert_begin, n_local_experts);

    // Offsets on the host: the per-expert launch geometry depends on the counts, so one
    // sync per call is structural rather than lazy. It is a chunk-level cost (once per
    // MoE layer per chunk), not a per-token one, which is the whole reason batching
    // makes it affordable.
    std::vector<int> h_counts((size_t)n_local_experts, 0);
    if (cudaMemcpyAsync(h_counts.data(), counts, (size_t)n_local_experts * sizeof(int),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) { sc.free_all(); return false; }

    std::vector<int> off((size_t)n_local_experts + 1, 0);
    int maxM = 0;
    for (int e = 0; e < n_local_experts; ++e) {
        off[e + 1] = off[e] + h_counts[e];
        if (h_counts[e] > maxM) maxM = h_counts[e];
    }
    if (off[n_local_experts] == 0) { sc.free_all(); return true; }   // nothing local: zeros stand

    if (cudaMemcpyAsync(cursor, off.data(), (size_t)n_local_experts * sizeof(int),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) { sc.free_all(); return false; }
    k3_pdl_launch(blocks, TPB, 0, stream, moe_place_kernel, tok, slot, cursor, ids, T,
                  top_k, expert_begin, n_local_experts);

    // 3. Per-expert tiles, sized to the BUSIEST expert so one allocation serves all.
    //    Routing is imbalanced, so maxM is meaningfully above the T*top_k/n_expert mean
    //    and sizing to the mean would fault on the tail.
    float*       gate = sc.take<float>((size_t)maxM * ffn);
    float*       up   = sc.take<float>((size_t)maxM * ffn);
    float*       h    = sc.take<float>((size_t)maxM * ffn);
    signed char* hq   = sc.take<signed char>((size_t)maxM * ffn);
    float*       sh   = sc.take<float>((size_t)maxM * (ffn / 32));
    float*       down = sc.take<float>((size_t)maxM * latent);
    if (!sc.ok) { sc.free_all(); return false; }

    const size_t gbpr = (size_t)latent / kQK;      // blocks per gate/up row
    const size_t dbpr = (size_t)ffn / kQK;         // blocks per down row
    const uint8_t* gbase = static_cast<const uint8_t*>(gate_exps);
    const uint8_t* ubase = static_cast<const uint8_t*>(up_exps);
    const uint8_t* dbase = static_cast<const uint8_t*>(down_exps);
    constexpr size_t kBlkBytes = 50;               // sizeof(block_iq1_s)

    for (int e = 0; e < n_local_experts; ++e) {
        const int M = h_counts[e];
        if (M == 0) continue;
        const int base = off[e];
        const int* rows = tok + base;

        // Expert e's gate/up row j lives at block (e*ffn + j)*gbpr — so the matrix is
        // [ffn, latent], already the [N, K] the GEMM wants. down is [latent, ffn].
        const void* gW = gbase + (size_t)e * ffn    * gbpr * kBlkBytes;
        const void* uW = ubase + (size_t)e * ffn    * gbpr * kBlkBytes;
        const void* dW = dbase + (size_t)e * latent * dbpr * kBlkBytes;

        if (!k3_moe_iq1s_mma_gemm(gate, xq, sx, gW, rows, M, ffn, latent, stream) ||
            !k3_moe_iq1s_mma_gemm(up,   xq, sx, uW, rows, M, ffn, latent, stream)) {
            sc.free_all(); return false;
        }

        const int64_t n = (int64_t)M * ffn;
        k3_pdl_launch((unsigned)((n + 255) / 256), 256, 0, stream, moe_situ_tile_kernel,
                      h, gate, up, n, situ_beta, situ_linear_beta);

        // The down GEMM reads h's rows compactly (0..M-1), so no gather list — and its
        // scale array is h's own per-row scale, not the activation's.
        if (!k3_moe_iq1s_mma_quantize_rows(hq, sh, h, M, ffn, stream) ||
            !k3_moe_iq1s_mma_gemm(down, hq, sh, dW, nullptr, M, latent, ffn, stream)) {
            sc.free_all(); return false;
        }

        const int64_t ns = (int64_t)M * latent;
        k3_pdl_launch((unsigned)((ns + 255) / 256), 256, 0, stream, moe_scatter_add_kernel,
                      out, down, tok, slot, w, M, latent, top_k, base);
    }

    sc.free_all();
    return true;
}

}  // namespace k3
}  // namespace kernels
}  // namespace sparkinfer
