// MLA query absorption: one WARP per (head, latent row), reading the projection in place.
//
// ---------------------------------------------------------------------------
// TWO THINGS ARE WRONG WITH THE SEQUENCE THIS REPLACES
// ---------------------------------------------------------------------------
// 1. THE LAUNCH SHAPE. mla_absorb_q_kernel runs grid (kv_lora, n_head) = (512, 12) =
//    6,144 BLOCKS of 128 threads per layer, and each thread contributes exactly ONE
//    product before the block pays a full block_sum — warp shuffles, a shared round
//    trip and two __syncthreads — to reduce 128 values. Across 24 MLA layers that is
//    147,456 blocks and 294,912 barriers per token per rank, to do 12 * 512 * 128 =
//    786K multiply-adds per layer. The arithmetic is trivial; the kernel spends its
//    time in reduction machinery and block scheduling.
//
//    qk_nope is 128 and a warp is 32 lanes, so one warp with a float4 per lane covers
//    a whole row: 6,144 warps instead of 6,144 blocks, 768 blocks of 256 threads, one
//    shuffle reduction, and no barrier anywhere. wk_b is [qk_nope, kv_lora, n_head]
//    with qk_nope FASTEST, so the 128 weights of a (head, row) pair are contiguous and
//    the lane reads are one aligned 512-byte line.
//
// 2. THE TWO STRIDED COPIES THAT FEED IT. The forward de-interleaves q_proj_out —
//    [n_head, key_length_mla] with each head's 192 values laid out as qk_nope(128)
//    then rope_dim(64) — into two scratch buffers with a pair of cudaMemcpy2DAsync,
//    purely so this kernel can take two flat pointers. That is 48 two-dimensional D2D
//    copies per token per rank on a path #63 measured as submission-dominated, and the
//    driver implements each as its own strided-copy kernel.
//
//    They are unnecessary: a stride is an index. This kernel takes q_proj_out and
//    key_length_mla and reads q_nope at [h * stride + d] and q_pe at
//    [h * stride + qk_nope + d]. Both offsets stay 16-byte aligned at K3's dims
//    (192 * 4 = 768 and 768 + 512), so the float4 reads are legal on both halves, and
//    s.q_nope / s.q_pe stop being written at all.
//
// ---------------------------------------------------------------------------
// WHAT IT COSTS
// ---------------------------------------------------------------------------
// The dot product is reassociated: lane l sums four CONSECUTIVE elements and the warp
// then folds 32 partials, where the original gave one element to each of 128 threads
// and folded through block_sum. Same terms, different tree, ~1 ulp — and unlike the
// context-split kernels this one IS reachable by the accuracy gate, which runs at
// n_ctx=4. It therefore gets its own toggle rather than riding another factor's.
//
// SPARKINFER_K3_MLA_ABSORB=0 declines, and the forward then does the two
// cudaMemcpy2DAsync de-interleaves and calls mla_absorb_q_f32 exactly as main does.

#include "sparkinfer/kernels/kimi_k3_fast.h"
#include "k3_pdl.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>

namespace sparkinfer {
namespace kernels {
namespace k3 {

namespace {

// WPB = warps per block. One warp owns one (head, kv_lora row).
// TOKEN AXIS (blockIdx.z). Token t's projection starts at (q_proj + t * q_tok_stride)
// and its output at (out + t * out_tok_stride); `wk_b` is a weight and is shared by
// every token. Nothing about a token's own launch changes — same warp-to-(head, row)
// map, same float4 lane reads, same 5-step shuffle fold — so warp (h, r) of token t
// computes exactly what it computed when token t was launched alone. Only which grid a
// warp belongs to moved, which is why this is bit-identical rather than merely close.
template <int WPB>
__global__ void mla_absorb_warp_kernel(float* __restrict__ out,
                                       const float* __restrict__ q_proj,
                                       const float* __restrict__ wk_b,
                                       int qk_nope, int kv_lora, int rope_dim,
                                       int q_stride, int n_head,
                                       int64_t out_tok_stride, int64_t q_tok_stride) {
    k3_pdl_sync();   // no-op unless launched programmatically
    const int lane = threadIdx.x & 31;
    const int wid  = blockIdx.x * WPB + (threadIdx.x >> 5);
    const int h    = wid / kv_lora;
    const int r    = wid - h * kv_lora;
    if (h >= n_head) return;

    // At n_tok == 1 blockIdx.z is 0 and neither term is ever added to a pointer.
    float* __restrict__ out_t = out + (int64_t)blockIdx.z * out_tok_stride;
    const float* __restrict__ q_t = q_proj + (int64_t)blockIdx.z * q_tok_stride;

    // wk_b is [qk_nope, kv_lora, n_head] with qk_nope fastest, so this pair's 128
    // weights are contiguous: one aligned float4 per lane covers the row.
    const float4* wr = (const float4*)(wk_b + ((size_t)h * kv_lora + r) * qk_nope);
    const float4* qh = (const float4*)(q_t + (size_t)h * q_stride);

    const float4 w4 = wr[lane];
    const float4 q4 = qh[lane];
    float acc = w4.x * q4.x + w4.y * q4.y + w4.z * q4.z + w4.w * q4.w;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);

    float* oh = out_t + (size_t)h * (kv_lora + rope_dim);
    if (lane == 0) oh[r] = acc;

    // The rope tail is a copy, not a product, and one warp per head owns it. It reads
    // the SAME projection row past qk_nope, which is what removes the second
    // de-interleave along with the first.
    if (r == 0) {
        const float* pe = q_t + (size_t)h * q_stride + qk_nope;
        for (int d = lane; d < rope_dim; d += 32) oh[kv_lora + d] = pe[d];
    }
}

inline bool aligned16(const void* p) { return ((uintptr_t)p & 15u) == 0; }

}  // namespace

bool k3_mla_absorb_q_strided(float* out, const float* q_proj, const float* wk_b,
                             int qk_nope, int kv_lora, int rope_dim, int q_stride,
                             int n_head, cudaStream_t stream, int n_tok,
                             int64_t out_tok_stride, int64_t q_tok_stride) {
    static const bool want = [] {
        const char* e = std::getenv("SPARKINFER_K3_MLA_ABSORB");
        return !(e && e[0] == '0');
    }();
    if (!want) return false;
    if (n_head <= 0 || kv_lora <= 0 || qk_nope <= 0 || rope_dim < 0) return false;
    // gridDim.z, not .y: .x already carries the (head, row) warps. The .z cap is 65535
    // like .y, and a chunk is orders of magnitude below it — checked rather than
    // assumed, because exceeding it is a launch failure and the forward's launch guard
    // is polled only once per phase.
    if (n_tok <= 0 || n_tok > 65535) return false;
    // The natural single-token widths, so the n_tok == 1 launch is byte-for-byte the
    // one this function has always made (blockIdx.z is 0 and neither stride is used).
    if (out_tok_stride == 0) out_tok_stride = (int64_t)n_head * (kv_lora + rope_dim);
    if (q_tok_stride == 0)   q_tok_stride   = (int64_t)n_head * q_stride;

    // One float4 per lane covers exactly one row. K3 runs qk_nope 128; anything else
    // declines rather than growing a loop this kernel is not shaped for.
    if (qk_nope != 128) return false;

    // Alignment is CHECKED, not assumed. q_proj is an offset into a scratch arena and
    // wk_b is a pointer into mmap'd GGUF tensor data, so 16-byte alignment is a
    // property of where they landed. A misaligned float4 load is a fault, not a
    // slowdown. The per-head strides must also be multiples of four floats, or head h
    // would start mid-vector.
    if ((q_stride & 3) != 0) return false;
    if (!aligned16(q_proj) || !aligned16(wk_b)) return false;
    // Token t reads at (q_proj + t * q_tok_stride), so the TOKEN stride has to preserve
    // the 16-byte alignment the head stride already preserves. A token stride that is
    // not a multiple of four floats would fault on token 1 and never on token 0, which
    // is the worst shape of bug this path can have.
    if ((q_tok_stride & 3) != 0) return false;

    // Warps per block, MEASURED on the node rather than picked: 8x H200, real shape,
    // 30 reps — WPB 2/4/8/16 gave 8.22 / 7.23 / 7.10 / 6.88 us. Monotone, so 16.
    static const int wpb = [] {
        const char* e = std::getenv("SPARKINFER_K3_MLA_ABSORB_WPB");
        const int v = e ? std::atoi(e) : 0;
        return (v == 2 || v == 4 || v == 8 || v == 16) ? v : 16;
    }();
    const long long warps = (long long)n_head * kv_lora;
    // dim3(g, 1, 1) IS dim3(g), so the n_tok == 1 grid is the one this always launched.
#define K3_ABSORB_LAUNCH(W)                                                          \
    k3_pdl_launch(dim3((unsigned)((warps + (W) - 1) / (W)), 1u, (unsigned)n_tok),     \
                  dim3((W) * 32), 0,                                                  \
                  stream, mla_absorb_warp_kernel<(W)>,                                \
                  out, q_proj, wk_b, qk_nope, kv_lora, rope_dim, q_stride, n_head,    \
                  out_tok_stride, q_tok_stride)
    switch (wpb) {
        case 2:  K3_ABSORB_LAUNCH(2);  break;
        case 4:  K3_ABSORB_LAUNCH(4);  break;
        case 8:  K3_ABSORB_LAUNCH(8);  break;
        default: K3_ABSORB_LAUNCH(16); break;
    }
#undef K3_ABSORB_LAUNCH
    return true;
}

}  // namespace k3
}  // namespace kernels
}  // namespace sparkinfer
