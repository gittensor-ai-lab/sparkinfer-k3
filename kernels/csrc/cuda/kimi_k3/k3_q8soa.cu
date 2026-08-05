// ===========================================================================
// Q8_0 STRUCTURE-OF-ARRAYS WEIGHTS — kill the 34-byte stride at LOAD time.
// ===========================================================================
// A BlockQ8_0 is 34 bytes (2 B scale + 32 B quants) with alignment 2, so a
// row's quants can never be read wider than 16 bits (see get_int_b2's note in
// k3_kernels.cu). Per block that costs 17 memory instructions, and a warp's 32
// consecutive blocks span ~9 cache lines per instruction — the projections are
// ISSUE-bound on exactly this. Splitting each tensor ONCE at load into
//   d[]  — one half per block, contiguous
//   qs[] — 32 bytes per block, contiguous and 32-byte aligned
// lets the dot kernel read a block's quants as two 128-bit loads: 3 memory
// instructions per block instead of 17, with the SAME bytes feeding the SAME
// dp4a chain in the SAME order — bit-identical by construction.
//
// The repack runs at WEIGHT-LOAD time on the default stream (never inside a
// graph capture) and registers the SoA pair against the original device
// pointer, so call sites need no plumbing: they ask "is there an SoA view of
// this pointer" and fall through to the shipped path when there is not.
// SPARKINFER_K3_Q8SOA=0 restores the AoS path; default ON (bit-identical).

#include "sparkinfer/kernels/kimi_k3.h"
#include "sparkinfer/kernels/kimi_k3_fast.h"
#include "sparkinfer/kernels/k3_proj_rowbudget.h"
#include "k3_pdl.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace sparkinfer {
namespace kernels {
namespace k3 {
namespace {

constexpr int kQ8BlockValues = 32;
constexpr int kQ8BlockBytes  = 34;   // 2 (fp16 d) + 32 (qs), the GGUF layout

// Local view of the packed activation block (the canonical struct is private to
// k3_kernels.cu). Layout is fixed by GGUF: 2 B fp16 scale then 32 int8 quants.
#pragma pack(push, 2)
struct Q8Blk { uint16_t d; int8_t qs[kQ8BlockValues]; };
#pragma pack(pop)
static_assert(sizeof(Q8Blk) == kQ8BlockBytes, "bad q8_0 layout");

// EXACTLY the reduction tree the shipped single-row kernel uses (warp
// shfl_down, per-warp partials to shared, thread 0 folds warps ascending).
// Same tree = same summation order = bit-identical result.
template <int BLOCK>
__device__ __forceinline__ float soa_block_sum(float v, float* shm) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) shm[warp] = v;
    __syncthreads();
    constexpr int NWARP = BLOCK / 32;
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int w = 0; w < NWARP; ++w) s += shm[w];
        shm[NWARP] = s;
    }
    __syncthreads();
    return shm[NWARP];
}

struct SoaView { const __half* d; const int8_t* qs; long nblocks_per_row; };

std::mutex g_soa_mu;
std::unordered_map<const void*, SoaView> g_soa;

bool q8soa_enabled() {
    static const bool on = [] {
        // Default ON: load-time SoA repack cuts Q8_0 proj issue stalls (17→3 mem
        // ops/block) with the same dp4a order. =0 restores AoS on one binary.
        const char* e = std::getenv("SPARKINFER_K3_Q8SOA");
        return !(e && e[0] == '0');
    }();
    return on;
}

// One thread per block: copy the scale and the 32 quants out of the packed
// layout. Byte moves only — no arithmetic, nothing to get wrong numerically.
__global__ void q8soa_repack_kernel(__half* __restrict__ d_out,
                                    int8_t* __restrict__ qs_out,
                                    const uint8_t* __restrict__ src,
                                    long n_blocks) {
    const long b = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const uint8_t* blk = src + b * kQ8BlockBytes;
    // The scale is 2-byte aligned by the block layout.
    d_out[b] = __ushort_as_half(*(const uint16_t*)blk);
    int8_t* q = qs_out + b * kQ8BlockValues;
    const uint8_t* s = blk + 2;
#pragma unroll
    for (int j = 0; j < kQ8BlockValues; ++j) q[j] = (int8_t)s[j];
}

// The dot kernel. ROWS output rows per CTA; within a row, thread t owns blocks
// t, t+BLOCK, ... — the SAME thread-to-block map and the SAME block_sum
// reduction as the shipped single-row kernel, so each row's summation order is
// identical to the fallback's and the result is bit-for-bit the fallback's.
// The activation stays in its packed BlockQ8_0 form: it is a few KB, re-read by
// every CTA, and therefore L1-resident — its 17-instruction cost is paid from
// cache, not DRAM, and repacking it per token would cost a launch.
template <int BLOCK, int ROWS>
__global__ void proj_q8soa_kernel(float* __restrict__ y,
                                  const Q8Blk* __restrict__ x,
                                  const __half* __restrict__ wd,
                                  const int8_t* __restrict__ wqs,
                                  long nbpr, int n_rows) {
    k3_pdl_sync();
    constexpr int NWARP = BLOCK / 32;
    const int r0   = blockIdx.x * ROWS;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    // P2: ONE barrier, not 2*ROWS. The old epilogue called the two-sync
    // block_sum per row — 32 barriers at ROWS=16 — which is exactly the defect
    // the one-barrier kernels exist to fix, and it confounded the first SoA
    // measurement. This stash + single sync + ascending-w fold mirrors the
    // shipped one-barrier kernel exactly, so each row's sum is the same float
    // it produces: same butterfly over the same lanes, same fold order.
    __shared__ float shm[NWARP * ROWS];

    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;

    for (long b = threadIdx.x; b < nbpr; b += BLOCK) {
        int xa[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint16_t* x16 = (const uint16_t*)x[b].qs;
            xa[i] = (int)x16[2 * i] | ((int)x16[2 * i + 1] << 16);
        }
        const float dx = __half2float(__ushort_as_half(x[b].d));
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const long row = (long)(r0 + r);
            if (r0 + r >= n_rows) break;
            const int8_t* q = wqs + (row * nbpr + b) * kQ8BlockValues;
            // TWO 128-bit loads: the whole point (32-byte aligned rows).
            const int4 w0 = ((const int4*)q)[0];
            const int4 w1 = ((const int4*)q)[1];
            const int wv[8] = { w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w };
            int sumi = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) sumi = __dp4a(wv[i], xa[i], sumi);
            const float dw = __half2float(wd[row * nbpr + b]);
            acc[r] += (float)sumi * (dw * dx);
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc[r] += __shfl_down_sync(0xffffffff, acc[r], off);
        if (lane == 0) shm[warp * ROWS + r] = acc[r];
    }

    __syncthreads();   // the only one

    if (threadIdx.x < ROWS) {
        const int r = threadIdx.x;
        if (r0 + r < n_rows) {
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < NWARP; ++w) s += shm[w * ROWS + r];
            y[r0 + r] = s;
        }
    }
}

}  // namespace

// Register (and build) the SoA view of a Q8_0 device tensor. Load-time only:
// allocates and launches on the default stream, then synchronises. Safe to call
// for any tensor — declines everything that is not a 2-D Q8_0 weight.
bool k3_q8soa_register(const void* wdata, int wtype, const long ne[4]) {
    if (!q8soa_enabled() || !wdata || wtype != 8) return false;
    if (ne[0] <= 0 || ne[1] <= 0 || (ne[2] > 1) || (ne[3] > 1)) return false;
    if (ne[0] % kQ8BlockValues) return false;
    const long nbpr = ne[0] / kQ8BlockValues, nb = nbpr * ne[1];
    {
        std::lock_guard<std::mutex> lk(g_soa_mu);
        if (g_soa.count(wdata)) return true;
    }
    __half* d = nullptr; int8_t* qs = nullptr;
    if (cudaMalloc(&d, nb * sizeof(__half)) != cudaSuccess) return false;
    if (cudaMalloc(&qs, nb * (size_t)kQ8BlockValues) != cudaSuccess) {
        cudaFree(d); return false;
    }
    const int T = 256;
    q8soa_repack_kernel<<<(unsigned)((nb + T - 1) / T), T>>>(
        d, qs, (const uint8_t*)wdata, nb);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        cudaFree(d); cudaFree(qs); return false;
    }
    std::lock_guard<std::mutex> lk(g_soa_mu);
    g_soa[wdata] = SoaView{d, qs, nbpr};
    return true;
}

// The projection entry: y[N] = W(NxK) . x_q8. Returns false when this tensor
// has no SoA view (gate off, never registered, or wrong type) — the caller then
// runs the shipped path unchanged.
bool k3_proj_q8soa(float* y, const void* q8_act, const void* wdata,
                   int N, int K, cudaStream_t stream) {
    if (!q8soa_enabled() || !wdata || N <= 0 || K <= 0) return false;
    SoaView v{};
    {
        std::lock_guard<std::mutex> lk(g_soa_mu);
        auto it = g_soa.find(wdata);
        if (it == g_soa.end()) return false;
        v = it->second;
    }
    const long nbpr = (long)K / kQ8BlockValues;
    if (v.nblocks_per_row != nbpr) return false;
    // THE BLOCK SIZE IS PART OF THE NUMERICS. Thread t owns blocks t, t+BLOCK,
    // ...; change BLOCK and you change which partials each thread accumulates,
    // i.e. the summation order, i.e. the bits. The shipped dispatch picks 32/64/
    // 128 from blocks_per_row (proj_block_for), so this launcher mirrors that
    // rule EXACTLY. BLOCK=256 here is what T1 caught as *** DIFFERS *** on
    // 2026-08-04 — a $0.30 catch of a $35/hr mistake.
    const int TB = nbpr <= 32 ? 32 : (nbpr <= 64 ? 64 : 128);
    const int rows = k3_proj_rows_for_budget(N, TB, 16);
    const unsigned grid = (unsigned)((N + rows - 1) / rows);
#define K3_SOA_LAUNCH(B, R)                                                        \
    k3_pdl_launch(grid, (B), 0, stream, proj_q8soa_kernel<(B), (R)>,               \
                  y, (const Q8Blk*)q8_act, v.d, v.qs, nbpr, N)
    switch (TB * 100 + rows) {
        case 3216: K3_SOA_LAUNCH(32, 16);  break;
        case 3208: K3_SOA_LAUNCH(32, 8);   break;
        case 3204: K3_SOA_LAUNCH(32, 4);   break;
        case 3202: K3_SOA_LAUNCH(32, 2);   break;
        case 3201: K3_SOA_LAUNCH(32, 1);   break;
        case 6416: K3_SOA_LAUNCH(64, 16);  break;
        case 6408: K3_SOA_LAUNCH(64, 8);   break;
        case 6404: K3_SOA_LAUNCH(64, 4);   break;
        case 6402: K3_SOA_LAUNCH(64, 2);   break;
        case 6401: K3_SOA_LAUNCH(64, 1);   break;
        case 12816: K3_SOA_LAUNCH(128, 16); break;
        case 12808: K3_SOA_LAUNCH(128, 8);  break;
        case 12804: K3_SOA_LAUNCH(128, 4);  break;
        case 12802: K3_SOA_LAUNCH(128, 2);  break;
        case 12801: K3_SOA_LAUNCH(128, 1);  break;
        default: return false;
    }
#undef K3_SOA_LAUNCH
    return true;
}

// Float-activation entry for the big projections: quantise x with the shipped
// public helper (identical serial-max semantics), then run the SoA dot.
// q8_scratch must hold (K/32)*34 bytes — the same contract as every q8act caller.
bool k3_proj_q8soa_f32(float* y, const float* x, const void* wdata,
                       int N, int K, void* q8_scratch, cudaStream_t stream) {
    if (!q8soa_enabled() || !wdata || !q8_scratch) return false;
    {
        std::lock_guard<std::mutex> lk(g_soa_mu);
        if (!g_soa.count(wdata)) return false;
    }
    if (!k3_quantize_act_f32(q8_scratch, x, K, stream)) return false;
    return k3_proj_q8soa(y, q8_scratch, wdata, N, K, stream);
}

}  // namespace k3
}  // namespace kernels
}  // namespace sparkinfer
