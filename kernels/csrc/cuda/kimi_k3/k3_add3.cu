// Factor — the FFN tail's two residual adds, in one launch.
//
// ===========================================================================
// THE SHAPE
// ===========================================================================
// Every MoE layer ends with two elementwise adds over the hidden width, back to back
// with no kernel between them:
//
//     k3_add_f32(s.ffn_out,  s.ffn_out,  s.shexp_out, H)   // fold in the shared expert
//     k3_add_f32(hidden_out, hidden_out, s.ffn_out,   H)   // FFN residual
//
// The second reads exactly what the first wrote. At 92 MoE layers that is 184 launches
// per token per rank for 2 x 28 KiB of arithmetic, and an elementwise add of 7168 floats
// costs 2.95 us on this part whatever it does — measured back-to-back, with the ~4 us of
// event overhead a single timed launch carries already removed. The work is nothing; the
// launch is everything.
//
// This does both in one kernel: 92 launches per token per rank removed.
//
// ===========================================================================
// WHY IT IS BIT-IDENTICAL
// ===========================================================================
// The original rounds ffn + shexp to f32, stores it, then adds hidden to that value.
// This computes the same sum into a register — a float, so the same rounding — writes it
// to the same place, and adds hidden to the same value. Same two operations, same order,
// same intermediate precision. Not "close": identical.
//
// s.ffn_out IS STILL WRITTEN, and that is deliberate rather than a leftover. The debug
// hook between the two adds reads it (`fwd.debug("ffn_out", ...)`), and a fused kernel
// that only produced hidden_out would silently change what a bisect sees at exactly the
// point someone would be bisecting. It costs one store of a buffer that is already in L2.
//
// ===========================================================================
// WHAT IT IS NOT
// ===========================================================================
// NOT a change to the residual structure, the order of accumulation, or which buffer
// anything lives in — the same three inputs produce the same two outputs.
// NOT the KDA elementwise fusion (#90's k3_kda_fuse.cu, merged) and not the KDA conv /
// l2-norm batching that #55 proposes: this is the MoE layer's tail, which neither
// touches, and it fuses a PAIR OF CALL SITES rather than batching independent tensors
// into one grid.
// NOT applicable to the attention residual add: that one has no adjacent partner.

#include "sparkinfer/kernels/kimi_k3_fast.h"
#include "k3_pdl.cuh"

#include <cuda_runtime.h>

#include <cstdlib>

namespace sparkinfer {
namespace kernels {
namespace k3 {
namespace {

// out_ab = a + b            (the shared-expert fold, kept for the debug hook)
// out    = c + (a + b)      (the FFN residual)
//
// DELIBERATELY NOT __restrict__ ON THE ALIASING PAIRS. Both original calls are in place:
// out_ab IS a (s.ffn_out) and out IS c (hidden_out). Promising the compiler they do not
// alias would be a lie it is entitled to act on. Only `b` is genuinely distinct, and it
// is the only one marked. Nothing is lost — every address here is read once and written
// once, so there is no reordering for __restrict__ to unlock.
//
// ROW AXIS (blockIdx.y). All five operands are row-major activations of the same width,
// so all five are offset by the same row_stride; there is no shared weight here. Block
// (bx, r) performs exactly what block bx performed, on row r — same block size, same
// thread-to-element map, same two f32 operations in the same order. Elementwise, so
// there is no reduction that could be re-partitioned.
__global__ void add3_f32_kernel(float* out, float* out_ab,
                                const float* a, const float* __restrict__ b,
                                const float* c, int64_t n, int64_t row_stride) {
    k3_pdl_sync();
    const int64_t roff = (int64_t)blockIdx.y * row_stride;
    float* o_r = out + roff;
    float* oab_r = out_ab + roff;
    const float* a_r = a + roff;
    const float* __restrict__ b_r = b + roff;
    const float* c_r = c + roff;
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float ab = a_r[i] + b_r[i];
    oab_r[i] = ab;
    o_r[i] = c_r[i] + ab;
}

__global__ void add3_rows_f32_kernel(float* out, float* out_ab,
                                     const float* a, const float* b,
                                     int64_t b_row_stride, const float* c,
                                     int rows, int cols) {
    k3_pdl_sync();
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= (int64_t)rows * cols) return;
    const int row = (int)(i / cols), col = (int)(i % cols);
    const float ab = a[i] + b[(int64_t)row * b_row_stride + col];
    out_ab[i] = ab;
    out[i] = c[i] + ab;
}

}  // namespace

bool k3_add3_f32(float* out, float* out_ab, const float* a, const float* b,
                 const float* c, int64_t n, cudaStream_t stream,
                 int n_rows, int64_t row_stride) {
    static const bool want = [] {
        const char* e = std::getenv("SPARKINFER_K3_ADD3");
        return !(e && e[0] == '0');
    }();
    if (!want) return false;
    if (!out || !out_ab || !a || !b || !c || n <= 0 || n_rows <= 0) return false;
    // gridDim.y is capped at 65535; above it there is no equivalent launch, so decline
    // rather than silently drop rows.
    if (n_rows > 65535) return false;
    const int T = 256;
    const int64_t blocks = (n + T - 1) / T;
    if (blocks > 0x7fffffffLL) return false;
    if (row_stride == 0) row_stride = n;
    k3_pdl_launch(dim3((unsigned)blocks, (unsigned)n_rows), T, 0, stream, add3_f32_kernel,
                  out, out_ab, a, b, c, n, row_stride);
    return true;
}

bool k3_add3_rows_f32(float* out, float* out_ab, const float* a,
                      const float* b, int64_t b_row_stride, const float* c,
                      int rows, int cols, cudaStream_t stream) {
    if (!out || !out_ab || !a || !b || !c || rows <= 0 || cols <= 0 ||
        b_row_stride < cols) return false;
    const int64_t n = (int64_t)rows * cols;
    const int64_t blocks = (n + 255) / 256;
    if (blocks > 0x7fffffffLL) return false;
    k3_pdl_launch((unsigned)blocks, 256, 0, stream, add3_rows_f32_kernel,
                  out, out_ab, a, b, b_row_stride, c, rows, cols);
    return true;
}

}  // namespace k3
}  // namespace kernels
}  // namespace sparkinfer
