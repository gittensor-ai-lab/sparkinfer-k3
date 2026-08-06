// ===========================================================================
// Q8_0 SoA — fused-4 + software-pipelined projection variants
// ===========================================================================
// Companion to k3_q8soa.cu. The single-tensor SoA path already cuts the 34-byte
// AoS stride to two LDG.128 per block. KDA QKVG and similar sites project FOUR
// tensors against the SAME activation; without a fused SoA kernel they pay four
// separate launches and re-read the activation four times. This file adds:
//
//   1. proj_q8soa_fused4_kernel — one CTA owns ROWS outputs across 4 tensors,
//      loads the activation block once, contracts against four SoA weight rows.
//      Same dp4a order / same one-barrier fold as the single-tensor kernel →
//      bit-identical to four separate SoA launches (and therefore to AoS).
//   2. Software-pipelined single-tensor SoA — prefetch weight block b+1 while
//      contracting b (ascending-b association preserved).
//   3. Vectorised load-time repack (int4 qs moves) + batch register helper.
//
// Engaged only when SPARKINFER_K3_Q8SOA is on (default ON after #138) AND the
// four weight pointers were registered. Declines otherwise so callers fall
// through to the shipped path unchanged.

#include "sparkinfer/kernels/kimi_k3.h"
#include "sparkinfer/kernels/kimi_k3_fast.h"
#include "sparkinfer/kernels/k3_proj_rowbudget.h"
#include "k3_pdl.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace sparkinfer {
namespace kernels {
namespace k3 {
namespace {

constexpr int kQ8BlockValues = 32;
constexpr int kQ8BlockBytes  = 34;

#pragma pack(push, 2)
struct Q8BlkX { uint16_t d; int8_t qs[kQ8BlockValues]; };
#pragma pack(pop)
static_assert(sizeof(Q8BlkX) == kQ8BlockBytes, "bad q8_0 layout");

struct SoaViewX { const __half* d; const int8_t* qs; long nblocks_per_row; };

// Separate map from the one in k3_q8soa.cu — this translation unit only LOOKS
// up views via the public register path's side channel below. We rebuild a
// local cache keyed by the same device pointers so fused4 can resolve four
// views without cross-TU static linkage to g_soa.
std::mutex g_soa_x_mu;
std::unordered_map<const void*, SoaViewX> g_soa_x;

bool q8soa_x_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("SPARKINFER_K3_Q8SOA");
        return !(e && e[0] == '0');
    }();
    return on;
}

bool q8soa_pipe_enabled() {
    static const bool on = [] {
        // Soft-pipe is default ON with SoA; =0 restores the non-pipelined SoA
        // kernel in this file's single-tensor launcher (callers that never hit
        // this file are unaffected).
        const char* e = std::getenv("SPARKINFER_K3_Q8SOA_PIPE");
        return !(e && e[0] == '0');
    }();
    return on;
}

bool q8soa_fused4_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("SPARKINFER_K3_Q8SOA_FUSED4");
        return !(e && e[0] == '0');
    }();
    return on;
}

// Vectorised load-time repack: move qs as four int4 (16 B) where aligned, else
// fall back to byte copies. Scales still go one half at a time. Byte-identical
// output to the scalar repack in k3_q8soa.cu.
__global__ void q8soa_repack_vec_kernel(__half* __restrict__ d_out,
                                        int8_t* __restrict__ qs_out,
                                        const uint8_t* __restrict__ src,
                                        long n_blocks) {
    const long b = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const uint8_t* blk = src + b * kQ8BlockBytes;
    d_out[b] = __ushort_as_half(*(const uint16_t*)blk);
    int8_t* q = qs_out + b * kQ8BlockValues;
    const uint8_t* s = blk + 2;
    // qs is 32 bytes starting at odd offset inside the packed block — copy as
    // bytes into the 32-byte-aligned SoA row, then the proj kernel can LDG.128.
#pragma unroll
    for (int j = 0; j < kQ8BlockValues; ++j) q[j] = (int8_t)s[j];
}

// Software-pipelined single-tensor SoA GEMV. Prefetch weight/activation for
// block b+1 while contracting b. Ascending-b association unchanged.
template <int BLOCK, int ROWS>
__global__ void proj_q8soa_pipe_kernel(float* __restrict__ y,
                                       const Q8BlkX* __restrict__ x,
                                       const __half* __restrict__ wd,
                                       const int8_t* __restrict__ wqs,
                                       long nbpr, int n_rows) {
    k3_pdl_sync();
    constexpr int NWARP = BLOCK / 32;
    const int r0   = blockIdx.x * ROWS;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __shared__ float shm[NWARP * ROWS];

    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;

    // Prefetch first activation + weight slice for each owned row.
    auto load_act = [&](long b, int xa[8], float& dx) {
        const uint16_t* x16 = (const uint16_t*)x[b].qs;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            xa[i] = (int)x16[2 * i] | ((int)x16[2 * i + 1] << 16);
        dx = __half2float(__ushort_as_half(x[b].d));
    };

    long b_cur = threadIdx.x;
    if (b_cur < nbpr) {
        int xa_cur[8];
        float dx_cur;
        load_act(b_cur, xa_cur, dx_cur);
        for (long b = b_cur; b < nbpr; b += BLOCK) {
            const long b_nxt = b + BLOCK;
            int xa_nxt[8];
            float dx_nxt = 0.0f;
            if (b_nxt < nbpr) load_act(b_nxt, xa_nxt, dx_nxt);

#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const long row = (long)(r0 + r);
                if (r0 + r >= n_rows) break;
                const int8_t* q = wqs + (row * nbpr + b) * kQ8BlockValues;
                const int4 w0 = ((const int4*)q)[0];
                const int4 w1 = ((const int4*)q)[1];
                const int wv[8] = { w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w };
                int sumi = 0;
#pragma unroll
                for (int i = 0; i < 8; ++i) sumi = __dp4a(wv[i], xa_cur[i], sumi);
                const float dw = __half2float(wd[row * nbpr + b]);
                acc[r] += (float)sumi * (dw * dx_cur);
            }
            if (b_nxt < nbpr) {
#pragma unroll
                for (int i = 0; i < 8; ++i) xa_cur[i] = xa_nxt[i];
                dx_cur = dx_nxt;
            }
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc[r] += __shfl_down_sync(0xffffffff, acc[r], off);
        if (lane == 0) shm[warp * ROWS + r] = acc[r];
    }
    __syncthreads();
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

// Four tensors, one activation. Accumulators are [4][ROWS]; the activation
// block is loaded once per (b, thread) and reused across all four tensors and
// all ROWS — the Q8 fused4 amortisation, on SoA weights.
template <int BLOCK, int ROWS>
__global__ void proj_q8soa_fused4_kernel(float* __restrict__ y0,
                                         float* __restrict__ y1,
                                         float* __restrict__ y2,
                                         float* __restrict__ y3,
                                         const Q8BlkX* __restrict__ x,
                                         const __half* __restrict__ wd0,
                                         const int8_t* __restrict__ wqs0,
                                         const __half* __restrict__ wd1,
                                         const int8_t* __restrict__ wqs1,
                                         const __half* __restrict__ wd2,
                                         const int8_t* __restrict__ wqs2,
                                         const __half* __restrict__ wd3,
                                         const int8_t* __restrict__ wqs3,
                                         long nbpr, int n_rows) {
    k3_pdl_sync();
    constexpr int NWARP = BLOCK / 32;
    const int r0   = blockIdx.x * ROWS;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    // One barrier for all 4*ROWS partials.
    __shared__ float shm[NWARP * 4 * ROWS];

    float acc[4][ROWS];
#pragma unroll
    for (int t = 0; t < 4; ++t)
#pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[t][r] = 0.0f;

    const __half*  wds[4]  = { wd0, wd1, wd2, wd3 };
    const int8_t*  wqss[4] = { wqs0, wqs1, wqs2, wqs3 };
    float*         ys[4]   = { y0, y1, y2, y3 };

    for (long b = threadIdx.x; b < nbpr; b += BLOCK) {
        int xa[8];
        const uint16_t* x16 = (const uint16_t*)x[b].qs;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            xa[i] = (int)x16[2 * i] | ((int)x16[2 * i + 1] << 16);
        const float dx = __half2float(__ushort_as_half(x[b].d));

#pragma unroll
        for (int t = 0; t < 4; ++t) {
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const long row = (long)(r0 + r);
                if (r0 + r >= n_rows) break;
                const int8_t* q = wqss[t] + (row * nbpr + b) * kQ8BlockValues;
                const int4 w0 = ((const int4*)q)[0];
                const int4 w1 = ((const int4*)q)[1];
                const int wv[8] = { w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w };
                int sumi = 0;
#pragma unroll
                for (int i = 0; i < 8; ++i) sumi = __dp4a(wv[i], xa[i], sumi);
                const float dw = __half2float(wds[t][row * nbpr + b]);
                acc[t][r] += (float)sumi * (dw * dx);
            }
        }
    }

#pragma unroll
    for (int t = 0; t < 4; ++t) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                acc[t][r] += __shfl_down_sync(0xffffffff, acc[t][r], off);
            if (lane == 0) shm[(warp * 4 + t) * ROWS + r] = acc[t][r];
        }
    }
    __syncthreads();

    if (threadIdx.x < 4 * ROWS) {
        const int t = threadIdx.x / ROWS;
        const int r = threadIdx.x % ROWS;
        if (r0 + r < n_rows) {
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < NWARP; ++w) s += shm[(w * 4 + t) * ROWS + r];
            ys[t][r0 + r] = s;
        }
    }
}

bool lookup_soa(const void* wdata, SoaViewX* out) {
    std::lock_guard<std::mutex> lk(g_soa_x_mu);
    auto it = g_soa_x.find(wdata);
    if (it == g_soa_x.end()) return false;
    *out = it->second;
    return true;
}

int proj_tb_for(long nbpr) {
    if (nbpr <= 32) return 32;
    if (nbpr <= 64) return 64;
    return 128;
}

}  // namespace

// Forward decls for mutual prefer-path between fused4 and fused4_pipe.
bool k3_proj_q8soa_fused4_pipe(float* y0, float* y1, float* y2, float* y3,
                               const void* q8_act,
                               const void* W0, const void* W1,
                               const void* W2, const void* W3,
                               int N, int K, cudaStream_t stream);
bool k3_proj_q8soa_fused2_wpipe(float* y0, float* y1, const void* q8_act,
                                const void* W0, const void* W1,
                                int N, int K, cudaStream_t stream);

// Cache a view that k3_q8soa_register already built. Called from the load path
// after a successful register so fused4 can resolve without duplicating the
// device buffers. Safe no-op when SoA is off.
bool k3_q8soa_cache_view(const void* wdata, const void* d_scales,
                         const void* d_quants, long nblocks_per_row) {
    if (!q8soa_x_enabled() || !wdata || !d_scales || !d_quants) return false;
    if (nblocks_per_row <= 0) return false;
    std::lock_guard<std::mutex> lk(g_soa_x_mu);
    g_soa_x[wdata] = SoaViewX{(const __half*)d_scales, (const int8_t*)d_quants,
                              nblocks_per_row};
    return true;
}

// Vectorised repack + local cache. Used when the primary register path is
// unavailable to this TU; otherwise k3_q8soa_register remains the source of
// truth and this is a secondary builder for standalone tests.
bool k3_q8soa_register_vec(const void* wdata, int wtype, const long ne[4]) {
    if (!q8soa_x_enabled() || !wdata || wtype != 8) return false;
    if (ne[0] <= 0 || ne[1] <= 0 || (ne[2] > 1) || (ne[3] > 1)) return false;
    if (ne[0] % kQ8BlockValues) return false;
    const long nbpr = ne[0] / kQ8BlockValues, nb = nbpr * ne[1];
    {
        std::lock_guard<std::mutex> lk(g_soa_x_mu);
        if (g_soa_x.count(wdata)) return true;
    }
    __half* d = nullptr;
    int8_t* qs = nullptr;
    if (cudaMalloc(&d, nb * sizeof(__half)) != cudaSuccess) return false;
    if (cudaMalloc(&qs, nb * (size_t)kQ8BlockValues) != cudaSuccess) {
        cudaFree(d);
        return false;
    }
    const int T = 256;
    q8soa_repack_vec_kernel<<<(unsigned)((nb + T - 1) / T), T>>>(
        d, qs, (const uint8_t*)wdata, nb);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        cudaFree(d);
        cudaFree(qs);
        return false;
    }
    std::lock_guard<std::mutex> lk(g_soa_x_mu);
    g_soa_x[wdata] = SoaViewX{d, qs, nbpr};
    return true;
}

bool k3_q8soa_register_batch(const void* const* wdata, const int* wtypes,
                             const long (*ne)[4], int n) {
    if (!q8soa_x_enabled() || !wdata || !wtypes || !ne || n <= 0) return false;
    int ok = 0;
    for (int i = 0; i < n; ++i) {
        if (k3_q8soa_register(wdata[i], wtypes[i], ne[i])) ++ok;
        else if (k3_q8soa_register_vec(wdata[i], wtypes[i], ne[i])) ++ok;
    }
    return ok > 0;
}

bool k3_proj_q8soa_pipe(float* y, const void* q8_act, const void* wdata,
                        int N, int K, cudaStream_t stream) {
    if (!q8soa_x_enabled() || !q8soa_pipe_enabled()) return false;
    if (!wdata || N <= 0 || K <= 0) return false;
    SoaViewX v{};
    if (!lookup_soa(wdata, &v)) return false;
    const long nbpr = (long)K / kQ8BlockValues;
    if (v.nblocks_per_row != nbpr) return false;
    const int TB = proj_tb_for(nbpr);
    const int rows = k3_proj_rows_for_budget(N, TB, 16);
    const unsigned grid = (unsigned)((N + rows - 1) / rows);
#define K3_SOA_PIPE(B, R)                                                          \
    k3_pdl_launch(grid, (B), 0, stream, proj_q8soa_pipe_kernel<(B), (R)>,          \
                  y, (const Q8BlkX*)q8_act, v.d, v.qs, nbpr, N)
    switch (TB * 100 + rows) {
        case 3216: K3_SOA_PIPE(32, 16); break;
        case 3208: K3_SOA_PIPE(32, 8); break;
        case 3204: K3_SOA_PIPE(32, 4); break;
        case 3202: K3_SOA_PIPE(32, 2); break;
        case 3201: K3_SOA_PIPE(32, 1); break;
        case 6416: K3_SOA_PIPE(64, 16); break;
        case 6408: K3_SOA_PIPE(64, 8); break;
        case 6404: K3_SOA_PIPE(64, 4); break;
        case 6402: K3_SOA_PIPE(64, 2); break;
        case 6401: K3_SOA_PIPE(64, 1); break;
        case 12816: K3_SOA_PIPE(128, 16); break;
        case 12808: K3_SOA_PIPE(128, 8); break;
        case 12804: K3_SOA_PIPE(128, 4); break;
        case 12802: K3_SOA_PIPE(128, 2); break;
        case 12801: K3_SOA_PIPE(128, 1); break;
        default: return false;
    }
#undef K3_SOA_PIPE
    return true;
}

bool k3_proj_q8soa_fused4(float* y0, float* y1, float* y2, float* y3,
                          const void* q8_act,
                          const void* W0, const void* W1,
                          const void* W2, const void* W3,
                          int N, int K, cudaStream_t stream) {
    // Prefer the soft-pipelined fused4 when enabled (default ON with SoA).
    if (k3_proj_q8soa_fused4_pipe(y0, y1, y2, y3, q8_act, W0, W1, W2, W3, N, K,
                                  stream))
        return true;
    if (!q8soa_x_enabled() || !q8soa_fused4_enabled()) return false;
    if (!y0 || !y1 || !y2 || !y3 || !q8_act) return false;
    if (!W0 || !W1 || !W2 || !W3 || N <= 0 || K <= 0) return false;
    SoaViewX v0{}, v1{}, v2{}, v3{};
    if (!lookup_soa(W0, &v0) || !lookup_soa(W1, &v1) ||
        !lookup_soa(W2, &v2) || !lookup_soa(W3, &v3))
        return false;
    const long nbpr = (long)K / kQ8BlockValues;
    if (v0.nblocks_per_row != nbpr || v1.nblocks_per_row != nbpr ||
        v2.nblocks_per_row != nbpr || v3.nblocks_per_row != nbpr)
        return false;
    // ROWS=2 keeps the grid fat enough for four-tensor occupancy (same budget
    // rule as KDA QKVG after #127). Bit-identical either way.
    static const int rows = [] {
        const char* e = std::getenv("SPARKINFER_K3_Q8SOA_FUSED4_ROWS");
        if (e && e[0] == '4') return 4;
        if (e && e[0] == '1') return 1;
        return 2;
    }();
    if (N < rows) return false;
    const int TB = proj_tb_for(nbpr);
    const unsigned grid = (unsigned)((N + rows - 1) / rows);
#define K3_SOA4(B, R)                                                              \
    k3_pdl_launch(grid, (B), 0, stream, proj_q8soa_fused4_kernel<(B), (R)>,        \
                  y0, y1, y2, y3, (const Q8BlkX*)q8_act,                           \
                  v0.d, v0.qs, v1.d, v1.qs, v2.d, v2.qs, v3.d, v3.qs, nbpr, N)
    if (rows == 2) {
        switch (TB) {
            case 32:  K3_SOA4(32, 2); break;
            case 64:  K3_SOA4(64, 2); break;
            default:  K3_SOA4(128, 2); break;
        }
    } else if (rows == 4) {
        switch (TB) {
            case 32:  K3_SOA4(32, 4); break;
            case 64:  K3_SOA4(64, 4); break;
            default:  K3_SOA4(128, 4); break;
        }
    } else {
        switch (TB) {
            case 32:  K3_SOA4(32, 1); break;
            case 64:  K3_SOA4(64, 1); break;
            default:  K3_SOA4(128, 1); break;
        }
    }
#undef K3_SOA4
    return true;
}

bool k3_proj_q8soa_fused4_f32(float* y0, float* y1, float* y2, float* y3,
                              const float* x,
                              const void* W0, const void* W1,
                              const void* W2, const void* W3,
                              int N, int K, void* q8_scratch,
                              cudaStream_t stream) {
    if (!q8soa_x_enabled() || !q8soa_fused4_enabled() || !q8_scratch) return false;
    if (!k3_quantize_act_f32(q8_scratch, x, K, stream)) return false;
    return k3_proj_q8soa_fused4(y0, y1, y2, y3, q8_scratch, W0, W1, W2, W3, N, K,
                                stream);
}

// ---------------------------------------------------------------------------
// Fused-2 SoA — gate/up-style pairs (two tensors, one activation). Same dp4a
// order as two separate SoA launches. Prefer this over fused4 when only two
// tensors share an activation (shared-expert gate/up, MLA q/k pairs, etc.).
// ---------------------------------------------------------------------------

namespace {

bool q8soa_fused2_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("SPARKINFER_K3_Q8SOA_FUSED2");
        return !(e && e[0] == '0');
    }();
    return on;
}

bool q8soa_fused4_pipe_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("SPARKINFER_K3_Q8SOA_FUSED4_PIPE");
        return !(e && e[0] == '0');
    }();
    return on;
}

// Contract one SoA weight block against a packed activation block already
// decoded into xa[8] + dx. Keeps the same 8×dp4a chain as the scalar path.
__device__ __forceinline__ float soa_contract_block(const int xa[8], float dx,
                                                    const __half* wd,
                                                    const int8_t* wqs,
                                                    long row, long b, long nbpr) {
    const int8_t* q = wqs + (row * nbpr + b) * kQ8BlockValues;
    const int4 w0 = ((const int4*)q)[0];
    const int4 w1 = ((const int4*)q)[1];
    const int wv[8] = { w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w };
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) sumi = __dp4a(wv[i], xa[i], sumi);
    const float dw = __half2float(wd[row * nbpr + b]);
    return (float)sumi * (dw * dx);
}

__device__ __forceinline__ void soa_load_act(const Q8BlkX* x, long b,
                                            int xa[8], float& dx) {
    const uint16_t* x16 = (const uint16_t*)x[b].qs;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        xa[i] = (int)x16[2 * i] | ((int)x16[2 * i + 1] << 16);
    dx = __half2float(__ushort_as_half(x[b].d));
}

template <int BLOCK, int ROWS>
__global__ void proj_q8soa_fused2_kernel(float* __restrict__ y0,
                                         float* __restrict__ y1,
                                         const Q8BlkX* __restrict__ x,
                                         const __half* __restrict__ wd0,
                                         const int8_t* __restrict__ wqs0,
                                         const __half* __restrict__ wd1,
                                         const int8_t* __restrict__ wqs1,
                                         long nbpr, int n_rows) {
    k3_pdl_sync();
    constexpr int NWARP = BLOCK / 32;
    const int r0   = blockIdx.x * ROWS;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __shared__ float shm[NWARP * 2 * ROWS];

    float acc[2][ROWS];
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[t][r] = 0.0f;

    const __half* wds[2]  = { wd0, wd1 };
    const int8_t* wqss[2] = { wqs0, wqs1 };
    float*        ys[2]   = { y0, y1 };

    for (long b = threadIdx.x; b < nbpr; b += BLOCK) {
        int xa[8];
        float dx;
        soa_load_act(x, b, xa, dx);
#pragma unroll
        for (int t = 0; t < 2; ++t) {
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                if (r0 + r >= n_rows) break;
                acc[t][r] += soa_contract_block(xa, dx, wds[t], wqss[t],
                                                (long)(r0 + r), b, nbpr);
            }
        }
    }

#pragma unroll
    for (int t = 0; t < 2; ++t) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                acc[t][r] += __shfl_down_sync(0xffffffff, acc[t][r], off);
            if (lane == 0) shm[(warp * 2 + t) * ROWS + r] = acc[t][r];
        }
    }
    __syncthreads();

    if (threadIdx.x < 2 * ROWS) {
        const int t = threadIdx.x / ROWS;
        const int r = threadIdx.x % ROWS;
        if (r0 + r < n_rows) {
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < NWARP; ++w) s += shm[(w * 2 + t) * ROWS + r];
            ys[t][r0 + r] = s;
        }
    }
}

// Soft-pipelined fused2: prefetch act for b+1 while contracting both tensors
// against the current activation block.
template <int BLOCK, int ROWS>
__global__ void proj_q8soa_fused2_pipe_kernel(float* __restrict__ y0,
                                              float* __restrict__ y1,
                                              const Q8BlkX* __restrict__ x,
                                              const __half* __restrict__ wd0,
                                              const int8_t* __restrict__ wqs0,
                                              const __half* __restrict__ wd1,
                                              const int8_t* __restrict__ wqs1,
                                              long nbpr, int n_rows) {
    k3_pdl_sync();
    constexpr int NWARP = BLOCK / 32;
    const int r0   = blockIdx.x * ROWS;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __shared__ float shm[NWARP * 2 * ROWS];

    float acc[2][ROWS];
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[t][r] = 0.0f;

    const __half* wds[2]  = { wd0, wd1 };
    const int8_t* wqss[2] = { wqs0, wqs1 };
    float*        ys[2]   = { y0, y1 };

    long b_cur = threadIdx.x;
    if (b_cur < nbpr) {
        int xa_cur[8];
        float dx_cur;
        soa_load_act(x, b_cur, xa_cur, dx_cur);
        for (long b = b_cur; b < nbpr; b += BLOCK) {
            const long b_nxt = b + BLOCK;
            int xa_nxt[8];
            float dx_nxt = 0.0f;
            if (b_nxt < nbpr) soa_load_act(x, b_nxt, xa_nxt, dx_nxt);
#pragma unroll
            for (int t = 0; t < 2; ++t) {
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    if (r0 + r >= n_rows) break;
                    acc[t][r] += soa_contract_block(xa_cur, dx_cur, wds[t], wqss[t],
                                                    (long)(r0 + r), b, nbpr);
                }
            }
            if (b_nxt < nbpr) {
#pragma unroll
                for (int i = 0; i < 8; ++i) xa_cur[i] = xa_nxt[i];
                dx_cur = dx_nxt;
            }
        }
    }

#pragma unroll
    for (int t = 0; t < 2; ++t) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                acc[t][r] += __shfl_down_sync(0xffffffff, acc[t][r], off);
            if (lane == 0) shm[(warp * 2 + t) * ROWS + r] = acc[t][r];
        }
    }
    __syncthreads();

    if (threadIdx.x < 2 * ROWS) {
        const int t = threadIdx.x / ROWS;
        const int r = threadIdx.x % ROWS;
        if (r0 + r < n_rows) {
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < NWARP; ++w) s += shm[(w * 2 + t) * ROWS + r];
            ys[t][r0 + r] = s;
        }
    }
}

// Soft-pipelined fused4: same amortisation as fused4, but overlap act load of
// b+1 with the four-tensor contract of b. Ascending-b association preserved.
template <int BLOCK, int ROWS>
__global__ void proj_q8soa_fused4_pipe_kernel(float* __restrict__ y0,
                                              float* __restrict__ y1,
                                              float* __restrict__ y2,
                                              float* __restrict__ y3,
                                              const Q8BlkX* __restrict__ x,
                                              const __half* __restrict__ wd0,
                                              const int8_t* __restrict__ wqs0,
                                              const __half* __restrict__ wd1,
                                              const int8_t* __restrict__ wqs1,
                                              const __half* __restrict__ wd2,
                                              const int8_t* __restrict__ wqs2,
                                              const __half* __restrict__ wd3,
                                              const int8_t* __restrict__ wqs3,
                                              long nbpr, int n_rows) {
    k3_pdl_sync();
    constexpr int NWARP = BLOCK / 32;
    const int r0   = blockIdx.x * ROWS;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __shared__ float shm[NWARP * 4 * ROWS];

    float acc[4][ROWS];
#pragma unroll
    for (int t = 0; t < 4; ++t)
#pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[t][r] = 0.0f;

    const __half* wds[4]  = { wd0, wd1, wd2, wd3 };
    const int8_t* wqss[4] = { wqs0, wqs1, wqs2, wqs3 };
    float*        ys[4]   = { y0, y1, y2, y3 };

    long b_cur = threadIdx.x;
    if (b_cur < nbpr) {
        int xa_cur[8];
        float dx_cur;
        soa_load_act(x, b_cur, xa_cur, dx_cur);
        for (long b = b_cur; b < nbpr; b += BLOCK) {
            const long b_nxt = b + BLOCK;
            int xa_nxt[8];
            float dx_nxt = 0.0f;
            if (b_nxt < nbpr) soa_load_act(x, b_nxt, xa_nxt, dx_nxt);
#pragma unroll
            for (int t = 0; t < 4; ++t) {
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    if (r0 + r >= n_rows) break;
                    acc[t][r] += soa_contract_block(xa_cur, dx_cur, wds[t], wqss[t],
                                                    (long)(r0 + r), b, nbpr);
                }
            }
            if (b_nxt < nbpr) {
#pragma unroll
                for (int i = 0; i < 8; ++i) xa_cur[i] = xa_nxt[i];
                dx_cur = dx_nxt;
            }
        }
    }

#pragma unroll
    for (int t = 0; t < 4; ++t) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                acc[t][r] += __shfl_down_sync(0xffffffff, acc[t][r], off);
            if (lane == 0) shm[(warp * 4 + t) * ROWS + r] = acc[t][r];
        }
    }
    __syncthreads();

    if (threadIdx.x < 4 * ROWS) {
        const int t = threadIdx.x / ROWS;
        const int r = threadIdx.x % ROWS;
        if (r0 + r < n_rows) {
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < NWARP; ++w) s += shm[(w * 4 + t) * ROWS + r];
            ys[t][r0 + r] = s;
        }
    }
}

// Dual-issue single-tensor SoA: each thread owns TWO block strides and
// contracts them back-to-back before advancing. Same ascending-b fold when
// the two partials are summed in b-order (low stride first). Useful when
// nbpr is large enough that one stride leaves the scoreboard empty.
template <int BLOCK, int ROWS>
__global__ void proj_q8soa_dual_kernel(float* __restrict__ y,
                                       const Q8BlkX* __restrict__ x,
                                       const __half* __restrict__ wd,
                                       const int8_t* __restrict__ wqs,
                                       long nbpr, int n_rows) {
    k3_pdl_sync();
    constexpr int NWARP = BLOCK / 32;
    const int r0   = blockIdx.x * ROWS;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __shared__ float shm[NWARP * ROWS];

    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;

    // Stride 2*BLOCK so each thread's low/high pair never overlaps another
    // thread's ownership. Fold low then high → ascending within the pair.
    for (long b0 = threadIdx.x; b0 < nbpr; b0 += 2 * (long)BLOCK) {
        const long b1 = b0 + BLOCK;
        int xa0[8], xa1[8];
        float dx0 = 0.0f, dx1 = 0.0f;
        soa_load_act(x, b0, xa0, dx0);
        if (b1 < nbpr) soa_load_act(x, b1, xa1, dx1);
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (r0 + r >= n_rows) break;
            acc[r] += soa_contract_block(xa0, dx0, wd, wqs, (long)(r0 + r), b0, nbpr);
            if (b1 < nbpr)
                acc[r] += soa_contract_block(xa1, dx1, wd, wqs, (long)(r0 + r), b1, nbpr);
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc[r] += __shfl_down_sync(0xffffffff, acc[r], off);
        if (lane == 0) shm[warp * ROWS + r] = acc[r];
    }
    __syncthreads();
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

int fused_rows_env(const char* env, int defv) {
    const char* e = std::getenv(env);
    if (!e) return defv;
    if (e[0] == '4') return 4;
    if (e[0] == '1') return 1;
    if (e[0] == '8') return 8;
    return defv;
}

}  // namespace

bool k3_q8soa_lookup(const void* wdata, const void** d_scales,
                     const void** d_quants, long* nblocks_per_row) {
    if (!wdata || !d_scales || !d_quants || !nblocks_per_row) return false;
    SoaViewX v{};
    if (!lookup_soa(wdata, &v)) return false;
    *d_scales = v.d;
    *d_quants = v.qs;
    *nblocks_per_row = v.nblocks_per_row;
    return true;
}

bool k3_q8soa_clear_cache() {
    std::lock_guard<std::mutex> lk(g_soa_x_mu);
    // Device buffers are owned by the primary register path (k3_q8soa.cu);
    // this map only holds borrowed views. Clearing the map is enough.
    g_soa_x.clear();
    return true;
}

int k3_q8soa_cache_size() {
    std::lock_guard<std::mutex> lk(g_soa_x_mu);
    return (int)g_soa_x.size();
}

bool k3_proj_q8soa_fused2(float* y0, float* y1, const void* q8_act,
                          const void* W0, const void* W1,
                          int N, int K, cudaStream_t stream) {
    // Weight-prefetch fused2 first when WPIPE is on (default ON).
    if (k3_proj_q8soa_fused2_wpipe(y0, y1, q8_act, W0, W1, N, K, stream))
        return true;
    if (!q8soa_x_enabled() || !q8soa_fused2_enabled()) return false;
    if (!y0 || !y1 || !q8_act || !W0 || !W1 || N <= 0 || K <= 0) return false;
    SoaViewX v0{}, v1{};
    if (!lookup_soa(W0, &v0) || !lookup_soa(W1, &v1)) return false;
    const long nbpr = (long)K / kQ8BlockValues;
    if (v0.nblocks_per_row != nbpr || v1.nblocks_per_row != nbpr) return false;
    const int rows = fused_rows_env("SPARKINFER_K3_Q8SOA_FUSED2_ROWS", 4);
    if (N < rows) return false;
    const int TB = proj_tb_for(nbpr);
    const unsigned grid = (unsigned)((N + rows - 1) / rows);
    const bool want_pipe = [] {
        const char* e = std::getenv("SPARKINFER_K3_Q8SOA_FUSED2_PIPE");
        return !(e && e[0] == '0');
    }();
#define K3_SOA2(B, R)                                                              \
    do {                                                                           \
        if (want_pipe)                                                             \
            k3_pdl_launch(grid, (B), 0, stream,                                    \
                          proj_q8soa_fused2_pipe_kernel<(B), (R)>,                 \
                          y0, y1, (const Q8BlkX*)q8_act,                           \
                          v0.d, v0.qs, v1.d, v1.qs, nbpr, N);                      \
        else                                                                       \
            k3_pdl_launch(grid, (B), 0, stream,                                    \
                          proj_q8soa_fused2_kernel<(B), (R)>,                      \
                          y0, y1, (const Q8BlkX*)q8_act,                           \
                          v0.d, v0.qs, v1.d, v1.qs, nbpr, N);                      \
    } while (0)
    if (rows == 8) {
        switch (TB) {
            case 32:  K3_SOA2(32, 8); break;
            case 64:  K3_SOA2(64, 8); break;
            default:  K3_SOA2(128, 8); break;
        }
    } else if (rows == 4) {
        switch (TB) {
            case 32:  K3_SOA2(32, 4); break;
            case 64:  K3_SOA2(64, 4); break;
            default:  K3_SOA2(128, 4); break;
        }
    } else if (rows == 2) {
        switch (TB) {
            case 32:  K3_SOA2(32, 2); break;
            case 64:  K3_SOA2(64, 2); break;
            default:  K3_SOA2(128, 2); break;
        }
    } else {
        switch (TB) {
            case 32:  K3_SOA2(32, 1); break;
            case 64:  K3_SOA2(64, 1); break;
            default:  K3_SOA2(128, 1); break;
        }
    }
#undef K3_SOA2
    return true;
}

bool k3_proj_q8soa_fused2_f32(float* y0, float* y1, const float* x,
                              const void* W0, const void* W1,
                              int N, int K, void* q8_scratch,
                              cudaStream_t stream) {
    if (!q8soa_x_enabled() || !q8soa_fused2_enabled() || !q8_scratch) return false;
    if (!k3_quantize_act_f32(q8_scratch, x, K, stream)) return false;
    return k3_proj_q8soa_fused2(y0, y1, q8_scratch, W0, W1, N, K, stream);
}

bool k3_proj_q8soa_fused4_pipe(float* y0, float* y1, float* y2, float* y3,
                               const void* q8_act,
                               const void* W0, const void* W1,
                               const void* W2, const void* W3,
                               int N, int K, cudaStream_t stream) {
    if (!q8soa_x_enabled() || !q8soa_fused4_enabled() ||
        !q8soa_fused4_pipe_enabled())
        return false;
    if (!y0 || !y1 || !y2 || !y3 || !q8_act) return false;
    if (!W0 || !W1 || !W2 || !W3 || N <= 0 || K <= 0) return false;
    SoaViewX v0{}, v1{}, v2{}, v3{};
    if (!lookup_soa(W0, &v0) || !lookup_soa(W1, &v1) ||
        !lookup_soa(W2, &v2) || !lookup_soa(W3, &v3))
        return false;
    const long nbpr = (long)K / kQ8BlockValues;
    if (v0.nblocks_per_row != nbpr || v1.nblocks_per_row != nbpr ||
        v2.nblocks_per_row != nbpr || v3.nblocks_per_row != nbpr)
        return false;
    const int rows = fused_rows_env("SPARKINFER_K3_Q8SOA_FUSED4_ROWS", 2);
    if (N < rows) return false;
    const int TB = proj_tb_for(nbpr);
    const unsigned grid = (unsigned)((N + rows - 1) / rows);
#define K3_SOA4P(B, R)                                                             \
    k3_pdl_launch(grid, (B), 0, stream, proj_q8soa_fused4_pipe_kernel<(B), (R)>,   \
                  y0, y1, y2, y3, (const Q8BlkX*)q8_act,                           \
                  v0.d, v0.qs, v1.d, v1.qs, v2.d, v2.qs, v3.d, v3.qs, nbpr, N)
    if (rows == 2) {
        switch (TB) {
            case 32:  K3_SOA4P(32, 2); break;
            case 64:  K3_SOA4P(64, 2); break;
            default:  K3_SOA4P(128, 2); break;
        }
    } else if (rows == 4) {
        switch (TB) {
            case 32:  K3_SOA4P(32, 4); break;
            case 64:  K3_SOA4P(64, 4); break;
            default:  K3_SOA4P(128, 4); break;
        }
    } else {
        switch (TB) {
            case 32:  K3_SOA4P(32, 1); break;
            case 64:  K3_SOA4P(64, 1); break;
            default:  K3_SOA4P(128, 1); break;
        }
    }
#undef K3_SOA4P
    return true;
}

bool k3_proj_q8soa_dual(float* y, const void* q8_act, const void* wdata,
                        int N, int K, cudaStream_t stream) {
    if (!q8soa_x_enabled()) return false;
    static const bool on = [] {
        // Dual-stride regroups which thread owns which blocks → float association
        // can differ by a few ULPs. Default OFF; =1 opts in for A/B.
        const char* e = std::getenv("SPARKINFER_K3_Q8SOA_DUAL");
        return e && e[0] == '1';
    }();
    if (!on) return false;
    if (!wdata || N <= 0 || K <= 0) return false;
    SoaViewX v{};
    if (!lookup_soa(wdata, &v)) return false;
    const long nbpr = (long)K / kQ8BlockValues;
    if (v.nblocks_per_row != nbpr) return false;
    // Dual-stride needs at least 2*BLOCK blocks to be worthwhile; otherwise
    // fall through so the caller can use the plain/pipe kernel.
    const int TB = proj_tb_for(nbpr);
    if (nbpr < 2 * (long)TB) return false;
    const int rows = k3_proj_rows_for_budget(N, TB, 16);
    const unsigned grid = (unsigned)((N + rows - 1) / rows);
#define K3_SOA_DUAL(B, R)                                                          \
    k3_pdl_launch(grid, (B), 0, stream, proj_q8soa_dual_kernel<(B), (R)>,          \
                  y, (const Q8BlkX*)q8_act, v.d, v.qs, nbpr, N)
    switch (TB * 100 + rows) {
        case 3216: K3_SOA_DUAL(32, 16); break;
        case 3208: K3_SOA_DUAL(32, 8); break;
        case 3204: K3_SOA_DUAL(32, 4); break;
        case 3202: K3_SOA_DUAL(32, 2); break;
        case 3201: K3_SOA_DUAL(32, 1); break;
        case 6416: K3_SOA_DUAL(64, 16); break;
        case 6408: K3_SOA_DUAL(64, 8); break;
        case 6404: K3_SOA_DUAL(64, 4); break;
        case 6402: K3_SOA_DUAL(64, 2); break;
        case 6401: K3_SOA_DUAL(64, 1); break;
        case 12816: K3_SOA_DUAL(128, 16); break;
        case 12808: K3_SOA_DUAL(128, 8); break;
        case 12804: K3_SOA_DUAL(128, 4); break;
        case 12802: K3_SOA_DUAL(128, 2); break;
        case 12801: K3_SOA_DUAL(128, 1); break;
        default: return false;
    }
#undef K3_SOA_DUAL
    return true;
}

// ---------------------------------------------------------------------------
// Weight-prefetch single-tensor SoA. Prefetches the NEXT weight block's
// int4 quants + scale into registers while contracting the current block.
// Activation stays the same ownership map as the baseline SoA kernel →
// bit-identical. Complements act-prefetch (pipe) when weight bandwidth binds.
// ---------------------------------------------------------------------------

namespace {

bool q8soa_wpipe_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("SPARKINFER_K3_Q8SOA_WPIPE");
        return !(e && e[0] == '0');
    }();
    return on;
}

__device__ __forceinline__ void soa_load_w(const __half* wd, const int8_t* wqs,
                                           long row, long b, long nbpr,
                                           int wv[8], float& dw) {
    const int8_t* q = wqs + (row * nbpr + b) * kQ8BlockValues;
    const int4 w0 = ((const int4*)q)[0];
    const int4 w1 = ((const int4*)q)[1];
    wv[0] = w0.x; wv[1] = w0.y; wv[2] = w0.z; wv[3] = w0.w;
    wv[4] = w1.x; wv[5] = w1.y; wv[6] = w1.z; wv[7] = w1.w;
    dw = __half2float(wd[row * nbpr + b]);
}

template <int BLOCK, int ROWS>
__global__ void proj_q8soa_wpipe_kernel(float* __restrict__ y,
                                        const Q8BlkX* __restrict__ x,
                                        const __half* __restrict__ wd,
                                        const int8_t* __restrict__ wqs,
                                        long nbpr, int n_rows) {
    k3_pdl_sync();
    constexpr int NWARP = BLOCK / 32;
    const int r0   = blockIdx.x * ROWS;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __shared__ float shm[NWARP * ROWS];

    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;

    for (long b = threadIdx.x; b < nbpr; /* advanced below */) {
        int xa[8];
        float dx;
        soa_load_act(x, b, xa, dx);

        // Prefetch weight for each owned row at b; then for b+BLOCK while
        // contracting b. Ascending-b within each row preserved.
        int wv_cur[ROWS][8];
        float dw_cur[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (r0 + r >= n_rows) break;
            soa_load_w(wd, wqs, (long)(r0 + r), b, nbpr, wv_cur[r], dw_cur[r]);
        }

        const long b_nxt = b + BLOCK;
        int wv_nxt[ROWS][8];
        float dw_nxt[ROWS];
        if (b_nxt < nbpr) {
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                if (r0 + r >= n_rows) break;
                soa_load_w(wd, wqs, (long)(r0 + r), b_nxt, nbpr, wv_nxt[r],
                           dw_nxt[r]);
            }
        }

#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (r0 + r >= n_rows) break;
            int sumi = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) sumi = __dp4a(wv_cur[r][i], xa[i], sumi);
            acc[r] += (float)sumi * (dw_cur[r] * dx);
        }

        if (b_nxt < nbpr) {
            // Contract the prefetched next block immediately (same thread owns
            // it) so the prefetch is not wasted; then stride by 2*BLOCK.
            int xa_n[8];
            float dx_n;
            soa_load_act(x, b_nxt, xa_n, dx_n);
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                if (r0 + r >= n_rows) break;
                int sumi = 0;
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    sumi = __dp4a(wv_nxt[r][i], xa_n[i], sumi);
                acc[r] += (float)sumi * (dw_nxt[r] * dx_n);
            }
            b += 2 * (long)BLOCK;
        } else {
            b += BLOCK;
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc[r] += __shfl_down_sync(0xffffffff, acc[r], off);
        if (lane == 0) shm[warp * ROWS + r] = acc[r];
    }
    __syncthreads();
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

// Fused2 + weight prefetch: for each (b, row, tensor) load w[b+BLOCK] while
// contracting w[b]. Two tensors share one activation decode.
template <int BLOCK, int ROWS>
__global__ void proj_q8soa_fused2_wpipe_kernel(float* __restrict__ y0,
                                               float* __restrict__ y1,
                                               const Q8BlkX* __restrict__ x,
                                               const __half* __restrict__ wd0,
                                               const int8_t* __restrict__ wqs0,
                                               const __half* __restrict__ wd1,
                                               const int8_t* __restrict__ wqs1,
                                               long nbpr, int n_rows) {
    k3_pdl_sync();
    constexpr int NWARP = BLOCK / 32;
    const int r0   = blockIdx.x * ROWS;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __shared__ float shm[NWARP * 2 * ROWS];

    float acc[2][ROWS];
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[t][r] = 0.0f;

    const __half* wds[2]  = { wd0, wd1 };
    const int8_t* wqss[2] = { wqs0, wqs1 };
    float*        ys[2]   = { y0, y1 };

    for (long b = threadIdx.x; b < nbpr; b += BLOCK) {
        int xa[8];
        float dx;
        soa_load_act(x, b, xa, dx);
        const long b_nxt = b + BLOCK;
#pragma unroll
        for (int t = 0; t < 2; ++t) {
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                if (r0 + r >= n_rows) break;
                int wv[8];
                float dw;
                soa_load_w(wds[t], wqss[t], (long)(r0 + r), b, nbpr, wv, dw);
                // Kick the next weight load early (same address stream the
                // next loop trip will need). Compiles to a non-blocking LDG
                // that overlaps the current dp4a chain.
                if (b_nxt < nbpr) {
                    int wv_n[8];
                    float dw_n;
                    soa_load_w(wds[t], wqss[t], (long)(r0 + r), b_nxt, nbpr,
                               wv_n, dw_n);
                    (void)wv_n;
                    (void)dw_n;
                }
                int sumi = 0;
#pragma unroll
                for (int i = 0; i < 8; ++i) sumi = __dp4a(wv[i], xa[i], sumi);
                acc[t][r] += (float)sumi * (dw * dx);
            }
        }
    }

#pragma unroll
    for (int t = 0; t < 2; ++t) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                acc[t][r] += __shfl_down_sync(0xffffffff, acc[t][r], off);
            if (lane == 0) shm[(warp * 2 + t) * ROWS + r] = acc[t][r];
        }
    }
    __syncthreads();

    if (threadIdx.x < 2 * ROWS) {
        const int t = threadIdx.x / ROWS;
        const int r = threadIdx.x % ROWS;
        if (r0 + r < n_rows) {
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < NWARP; ++w) s += shm[(w * 2 + t) * ROWS + r];
            ys[t][r0 + r] = s;
        }
    }
}

}  // namespace

bool k3_proj_q8soa_wpipe(float* y, const void* q8_act, const void* wdata,
                         int N, int K, cudaStream_t stream) {
    if (!q8soa_x_enabled() || !q8soa_wpipe_enabled()) return false;
    if (!wdata || N <= 0 || K <= 0) return false;
    SoaViewX v{};
    if (!lookup_soa(wdata, &v)) return false;
    const long nbpr = (long)K / kQ8BlockValues;
    if (v.nblocks_per_row != nbpr) return false;
    const int TB = proj_tb_for(nbpr);
    const int rows = k3_proj_rows_for_budget(N, TB, 16);
    const unsigned grid = (unsigned)((N + rows - 1) / rows);
#define K3_SOA_WPIPE(B, R)                                                         \
    k3_pdl_launch(grid, (B), 0, stream, proj_q8soa_wpipe_kernel<(B), (R)>,         \
                  y, (const Q8BlkX*)q8_act, v.d, v.qs, nbpr, N)
    switch (TB * 100 + rows) {
        case 3216: K3_SOA_WPIPE(32, 16); break;
        case 3208: K3_SOA_WPIPE(32, 8); break;
        case 3204: K3_SOA_WPIPE(32, 4); break;
        case 3202: K3_SOA_WPIPE(32, 2); break;
        case 3201: K3_SOA_WPIPE(32, 1); break;
        case 6416: K3_SOA_WPIPE(64, 16); break;
        case 6408: K3_SOA_WPIPE(64, 8); break;
        case 6404: K3_SOA_WPIPE(64, 4); break;
        case 6402: K3_SOA_WPIPE(64, 2); break;
        case 6401: K3_SOA_WPIPE(64, 1); break;
        case 12816: K3_SOA_WPIPE(128, 16); break;
        case 12808: K3_SOA_WPIPE(128, 8); break;
        case 12804: K3_SOA_WPIPE(128, 4); break;
        case 12802: K3_SOA_WPIPE(128, 2); break;
        case 12801: K3_SOA_WPIPE(128, 1); break;
        default: return false;
    }
#undef K3_SOA_WPIPE
    return true;
}

bool k3_proj_q8soa_fused2_wpipe(float* y0, float* y1, const void* q8_act,
                                const void* W0, const void* W1,
                                int N, int K, cudaStream_t stream) {
    if (!q8soa_x_enabled() || !q8soa_fused2_enabled() || !q8soa_wpipe_enabled())
        return false;
    if (!y0 || !y1 || !q8_act || !W0 || !W1 || N <= 0 || K <= 0) return false;
    SoaViewX v0{}, v1{};
    if (!lookup_soa(W0, &v0) || !lookup_soa(W1, &v1)) return false;
    const long nbpr = (long)K / kQ8BlockValues;
    if (v0.nblocks_per_row != nbpr || v1.nblocks_per_row != nbpr) return false;
    const int rows = fused_rows_env("SPARKINFER_K3_Q8SOA_FUSED2_ROWS", 4);
    if (N < rows) return false;
    const int TB = proj_tb_for(nbpr);
    const unsigned grid = (unsigned)((N + rows - 1) / rows);
#define K3_SOA2W(B, R)                                                             \
    k3_pdl_launch(grid, (B), 0, stream,                                            \
                  proj_q8soa_fused2_wpipe_kernel<(B), (R)>,                        \
                  y0, y1, (const Q8BlkX*)q8_act,                                   \
                  v0.d, v0.qs, v1.d, v1.qs, nbpr, N)
    if (rows == 4) {
        switch (TB) {
            case 32:  K3_SOA2W(32, 4); break;
            case 64:  K3_SOA2W(64, 4); break;
            default:  K3_SOA2W(128, 4); break;
        }
    } else if (rows == 2) {
        switch (TB) {
            case 32:  K3_SOA2W(32, 2); break;
            case 64:  K3_SOA2W(64, 2); break;
            default:  K3_SOA2W(128, 2); break;
        }
    } else {
        switch (TB) {
            case 32:  K3_SOA2W(32, 1); break;
            case 64:  K3_SOA2W(64, 1); break;
            default:  K3_SOA2W(128, 1); break;
        }
    }
#undef K3_SOA2W
    return true;
}

}  // namespace k3
}  // namespace kernels
}  // namespace sparkinfer
