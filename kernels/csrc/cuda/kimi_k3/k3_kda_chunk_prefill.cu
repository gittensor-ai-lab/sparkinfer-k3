// ===========================================================================
// KDA CHUNK-PARALLEL PREFILL SCAN.
// ===========================================================================
// kda_decode_step_f32 (k3_kernels.cu) is ONE token: prompt ingestion currently calls
// it T times in a sequential loop, so a KDA layer's prefill cost is T dependent
// kernel launches whose own body is a single rank-1 state update. This file is the
// chunk-parallel (WY / UT-transform) reformulation of the SAME recurrence — the
// delta-rule family sparkinfer already carries for Qwen's GDN
// (kernels/csrc/cuda/fused/prefill_gdn_chunk.cu), specialised to K3's own gate
// (per-channel, A_log/lower_bound-sigmoid) and scalar-per-head beta.
//
// See kernels/include/sparkinfer/kernels/kimi_k3.h, op 6b, for the full derivation
// (re-derived from kda_decode_step_kernel's own pinned formula, cross-checked
// against Moonshot's own FlashKDA — github.com/MoonshotAI/FlashKDA, whose
// H=96/D=128 benchmark shape and lower_bound range are K3's own) and for exactly
// what is and is not the same contract as the per-token step. The algorithm was
// proven against the token-by-token recurrence, in float64, BEFORE this file was
// written: kernels/tests/k3_kda_chunk_prefill_cpu_test.cpp.
//
// TWO KERNELS, SPLIT ALONG THE ONE REAL DEPENDENCY.
//
//   K1 (prep)  grid = (n_chunks, n_head), FULLY PARALLEL. Every quantity through
//              INV = (I+L)^-1 is a pure function of this chunk's own q/k/v/g/beta —
//              gate activation, L2 norm, beta sigmoid, the decayed/inverse K and Q,
//              the two [16,16] matrices — and depends on the state entering the
//              chunk NOT AT ALL. Writes k_decayed, q_decayed, k_restored, g_total,
//              beta_act, Mqk, INV to a workspace.
//   K2 (scan)  grid = n_head, SEQUENTIAL OVER CHUNKS ONLY (not tokens): T/16
//              dependent steps instead of T, each a handful of small dense matmuls
//              against K1's precomputed workspace, not a fresh triangular solve.
//
// This is FlashKDA's own two-kernel split, kept for FlashKDA's own measured reason
// (their deep-dive: fusing the two lost >=15% because the token-parallel work was
// bottlenecked by the recurrence's low parallelism). K2 at grid = n_head is 12
// blocks per rank under the sharded policy — poor occupancy, and structurally so;
// the split exists precisely so that K1, which is (T/16) x n_head blocks, is where
// the bulk of the arithmetic lands.
//
// F32 THROUGHOUT, PLAIN CUDA, NO TENSOR CORES — matching this file family's stated
// convention (k3_kernels.cu: "correctness-first... fused variants come after the
// math is pinned, not before") and NOT FlashKDA's own implementation, which is
// CUTLASS/CUTE, sm_90a-specific (TMA, warpgroup MMA), bf16/fp16 throughout, and
// inverts (I+L) by a doubling Neumann series to stay in bf16 range at CHUNK=16.
// This inverts by plain forward substitution instead: at CHUNK=16 the triangular
// solve is O(16^3) against O(16*head_dim^2) for the two decayed projections, a
// rounding error either way once nothing is fighting for tensor-core throughput.
//
// CHUNK=16 IS LOAD-BEARING, NOT A TUNING KNOB. G_t is a cumsum of per-channel
// log-gates bounded below by lower_bound = -5, so |G| <= 16*5 = 80 within a chunk,
// and exp(+-G) spans [1.8e-35, 5.5e34] — inside f32's normal range (ln(f32 max) is
// ~88.7) with the margin FlashKDA's own chunk-size analysis picked it for. A
// CHUNK of 32 would put exp(-G) past f32 infinity on exactly the deep-decay
// channels that carry the recency structure.
//
// THE THREAD-TO-COLUMN MAPPING IN K2 IS THE EXISTING kda_decode_step_kernel's,
// not a new one. Thread j owns state row s[j][:] (S[i][j] in math notation, at
// s[j*head_dim+i] — op 6's own layout), reading and writing all head_dim elements
// of it, exactly the ownership kda_decode_step_kernel already uses for one token.
// Generalising an already-proven ownership to 16 tokens per chunk is lower risk
// than inventing a new one: an axis mistake here is the exact bug class op 6's own
// header documents shipping once, undetected by three separate coverage gaps,
// because a zero starting state cannot distinguish the two axes.
//
// NOT WIRED TO ANY CALLER. Nothing in runtime/ produces a [T, n_head, head_dim]
// block of raw q/k/v/g/beta_logit for this to consume — that is the batched-
// prefill driver, which does not exist yet. This lands as an independently
// testable piece, the same shape as k3_moe_iq1s_mma.cu before it: the GPU test
// (k3_kda_chunk_prefill_gpu_test.cu) grades the full folded contract against
// float64 at K3's real head_dim, and skips cleanly on a box with no device.

#include "sparkinfer/kernels/kimi_k3.h"
#include "k3_pdl.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace sparkinfer {
namespace kernels {
namespace k3 {
namespace {

constexpr int CHUNK = 16;

__device__ __forceinline__ float k3c_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// Per-(chunk,head) workspace element counts. Named constants rather than inlined
// arithmetic so the launcher's allocation size and the two kernels' offset
// arithmetic cannot independently drift. head_dim is fixed at 128 throughout this
// file — see the launcher's guard and the note on K1.
constexpr int kD = 128;
struct WsLayout {
    static constexpr size_t kKD   = (size_t)CHUNK * kD;    // k_decayed  [16,128]
    static constexpr size_t kQD   = (size_t)CHUNK * kD;    // q_decayed  [16,128]
    static constexpr size_t kKR   = (size_t)CHUNK * kD;    // k_restored [16,128]
    static constexpr size_t kGT   = kD;                    // g_total    [128]
    static constexpr size_t kBeta = CHUNK;                 // beta_act   [16]
    static constexpr size_t kMqk  = (size_t)CHUNK * CHUNK; // [16,16]
    static constexpr size_t kInv  = (size_t)CHUNK * CHUNK; // [16,16]
};

// ---------------------------------------------------------------------------
// Kernel 1: prep. grid = (n_chunks, n_head), block = 128 threads = head_dim.
// One block per (chunk, head), independent of every other block and of the state.
// ---------------------------------------------------------------------------
// head_dim is FIXED at 128 (K3's kda_head_dim), not a template/runtime parameter:
// the thread count, the (t,s)-pair stride and the shared layout are all sized
// against it directly. K3 has exactly one KDA head_dim; this is written against
// that fact rather than generalising for a case that does not exist in the model.
//
// BLOCK = 128 SO EVERY BARRIER IS UNIFORM. The phases below alternate between
// channel-parallel work (one thread per d) and token-parallel work (one thread
// per t), with __syncthreads() between them; at 128 threads every thread reaches
// every barrier, which a wider block with guarded phases would not guarantee.
__global__ __launch_bounds__(kD, 4)
void k3_kda_chunk_prep_kernel(float* __restrict__ ws_kd, float* __restrict__ ws_qd,
                              float* __restrict__ ws_kr, float* __restrict__ ws_gt,
                              float* __restrict__ ws_beta, float* __restrict__ ws_mqk,
                              float* __restrict__ ws_inv,
                              const float* __restrict__ q, const float* __restrict__ k,
                              const float* __restrict__ g_raw,
                              const float* __restrict__ beta_logit,
                              const float* __restrict__ A,
                              int T, int n_head, float lower_bound, float l2_eps) {
    k3_pdl_sync();
    const int chunk_idx = blockIdx.x;
    const int h = blockIdx.y;
    const int t0 = chunk_idx * CHUNK;
    const int actual_len = min(CHUNK, T - t0);
    const int tid = threadIdx.x;

    __shared__ float s_kd[CHUNK * kD];      // staging for raw k, then k_decayed
    __shared__ float s_qd[CHUNK * kD];      // staging for raw q, then q_decayed
    __shared__ float s_ki[CHUNK * kD];      // k_inv
    __shared__ float s_gt[kD];              // g_total
    __shared__ float s_qn[CHUNK];           // per-token q inv-norm (incl. 1/sqrt(D))
    __shared__ float s_kn[CHUNK];           // per-token k inv-norm
    __shared__ float s_beta[CHUNK];
    __shared__ float s_L[CHUNK * CHUNK];
    __shared__ float s_Mqk[CHUNK * CHUNK];
    __shared__ float s_INV[CHUNK * CHUNK];
    static_assert(sizeof(s_kd) + sizeof(s_qd) + sizeof(s_ki) + sizeof(s_gt) +
                  sizeof(s_qn) + sizeof(s_kn) + sizeof(s_beta) +
                  sizeof(s_L) + sizeof(s_Mqk) + sizeof(s_INV) <= 48 * 1024,
                  "K1 static shared exceeds the 48 KB default per-block limit");

    // ---- Phase 1a: channel-parallel. Thread d activates the gate for its channel
    // (op 7's exact formula: g = lb * sigmoid(-(A[h] * g_raw))) and stages RAW q/k.
    // Padding rows past actual_len get g = 0 (no decay contribution) and q = k = 0;
    // combined with beta = 0 below, a padding row's entire row of L is zero, so its
    // INV row is the identity row, so its U is exactly zero — it cannot reach the
    // state or any real row's output. Proven end-to-end by the CPU test's ragged
    // and T=1 cases.
    const int d = tid;
    const float Ah = A[h];
    float g_reg[CHUNK];
    for (int t = 0; t < CHUNK; ++t) {
        float gv = 0.0f, qv = 0.0f, kv = 0.0f;
        if (t < actual_len) {
            const size_t off = ((size_t)(t0 + t) * n_head + h) * kD + d;
            gv = lower_bound * k3c_sigmoid(-(Ah * g_raw[off]));
            qv = q[off];
            kv = k[off];
        }
        g_reg[t] = gv;
        s_qd[t * kD + d] = qv;
        s_kd[t * kD + d] = kv;
    }
    __syncthreads();

    // ---- Phase 1b: token-parallel. Thread t (t < 16) reduces its token's
    // sum-of-squares serially over the 128 staged channels and folds the norm into
    // one multiplier — op 8's contract, scale = 1/sqrt(head_dim) for Q, 1 for K.
    // An all-zero row (padding, or a genuinely zero activation) hits rsqrtf(eps),
    // large but finite, times zero — the same behaviour l2_norm_heads_kernel has.
    if (tid < CHUNK) {
        const int t = tid;
        float qs = 0.0f, ks = 0.0f;
        for (int i = 0; i < kD; ++i) {
            const float qv = s_qd[t * kD + i], kv = s_kd[t * kD + i];
            qs += qv * qv;
            ks += kv * kv;
        }
        s_qn[t] = rsqrtf(qs + l2_eps) * rsqrtf((float)kD);
        s_kn[t] = rsqrtf(ks + l2_eps);
        s_beta[t] = (t < actual_len)
            ? k3c_sigmoid(beta_logit[(size_t)(t0 + t) * n_head + h]) : 0.0f;
    }
    __syncthreads();

    // ---- Phase 1c: channel-parallel again. Thread d walks its channel's cumsum
    // and rewrites the staging in place — each element is read and written by the
    // same thread, so no barrier is needed inside the loop.
    float G = 0.0f;
    for (int t = 0; t < CHUNK; ++t) {
        G += g_reg[t];
        const float eg  = __expf(G);
        const float egn = __expf(-G);
        const float qn = s_qd[t * kD + d] * s_qn[t];
        const float kn = s_kd[t * kD + d] * s_kn[t];
        s_qd[t * kD + d] = qn * eg;
        s_kd[t * kD + d] = kn * eg;
        s_ki[t * kD + d] = kn * egn;
    }
    s_gt[d] = G;
    __syncthreads();

    // ---- Phase 2: one (t,s) pair per thread, 256 pairs strided over 128 threads.
    // Each is a serial 128-length dot from shared. L[t][s] = beta_t*(kd_t.ki_s)
    // strictly below the diagonal; Mqk[t][s] = qd_t.ki_s on and below it.
    for (int p = tid; p < CHUNK * CHUNK; p += kD) {
        const int t = p / CHUNK, s = p % CHUNK;
        float l = 0.0f, m = 0.0f;
        if (s <= t) {
            float dot_kk = 0.0f, dot_qk = 0.0f;
            for (int i = 0; i < kD; ++i) {
                dot_kk += s_kd[t * kD + i] * s_ki[s * kD + i];
                dot_qk += s_qd[t * kD + i] * s_ki[s * kD + i];
            }
            l = (s < t) ? s_beta[t] * dot_kk : 0.0f;
            m = dot_qk;
        }
        s_L[p] = l;
        s_Mqk[p] = m;
    }
    __syncthreads();

    // ---- Phase 3: forward substitution, (I+L) INV = I. Thread `col` owns column
    // col; INV[s][col] for s < t is this thread's own register history, so the
    // 16-step loop needs no synchronisation inside it.
    if (tid < CHUNK) {
        const int col = tid;
        float inv_col[CHUNK];
#pragma unroll
        for (int t = 0; t < CHUNK; ++t) {
            float acc = (t == col) ? 1.0f : 0.0f;
            for (int s = 0; s < t; ++s) acc -= s_L[t * CHUNK + s] * inv_col[s];
            inv_col[t] = acc;
            s_INV[t * CHUNK + col] = acc;
        }
    }
    __syncthreads();

    // ---- Phase 4: write the workspace. k_restored = k_inv * exp(g_total) is
    // formed here from what shared already holds rather than kept as a fourth
    // 8 KB staging array.
    const size_t tile = (size_t)h * gridDim.x + chunk_idx;   // gridDim.x == n_chunks
    float* out_kd   = ws_kd   + tile * WsLayout::kKD;
    float* out_qd   = ws_qd   + tile * WsLayout::kQD;
    float* out_kr   = ws_kr   + tile * WsLayout::kKR;
    float* out_gt   = ws_gt   + tile * WsLayout::kGT;
    float* out_beta = ws_beta + tile * WsLayout::kBeta;
    float* out_mqk  = ws_mqk  + tile * WsLayout::kMqk;
    float* out_inv  = ws_inv  + tile * WsLayout::kInv;

    for (int i = tid; i < CHUNK * kD; i += kD) {
        out_kd[i] = s_kd[i];
        out_qd[i] = s_qd[i];
        out_kr[i] = s_ki[i] * __expf(s_gt[i % kD]);
    }
    out_gt[tid] = s_gt[tid];
    if (tid < CHUNK) out_beta[tid] = s_beta[tid];
    for (int i = tid; i < CHUNK * CHUNK; i += kD) {
        out_mqk[i] = s_Mqk[i];
        out_inv[i] = s_INV[i];
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: scan. grid = n_head, block = 128 threads = head_dim. One block per
// head, sequential over its chunks — the one real dependency, T/16 steps.
// ---------------------------------------------------------------------------
// Thread j owns state row j (S[i][j] at state[j*head_dim+i]) for the whole
// kernel, reading and writing all head_dim elements of it straight from global —
// the access pattern kda_decode_step_kernel already uses, deliberately NOT staged
// through shared here. The staged variant (kda_decode_step_smem_kernel's IC=32
// column-chunk trick) is the natural next step once this is validated on
// hardware, not before: this file's job is the algorithm, correct first.
__global__ __launch_bounds__(kD, 4)
void k3_kda_chunk_scan_kernel(float* __restrict__ out, float* __restrict__ state,
                              const float* __restrict__ v,
                              const float* __restrict__ ws_kd, const float* __restrict__ ws_qd,
                              const float* __restrict__ ws_kr, const float* __restrict__ ws_gt,
                              const float* __restrict__ ws_beta, const float* __restrict__ ws_mqk,
                              const float* __restrict__ ws_inv,
                              int T, int n_head, int n_chunks) {
    k3_pdl_sync();
    const int h = blockIdx.x;
    const int j = threadIdx.x;

    __shared__ float s_kd[CHUNK * kD];
    __shared__ float s_qd[CHUNK * kD];
    __shared__ float s_kr[CHUNK * kD];
    __shared__ float s_gt[kD];
    __shared__ float s_beta[CHUNK];
    __shared__ float s_mqk[CHUNK * CHUNK];
    __shared__ float s_inv[CHUNK * CHUNK];
    static_assert(sizeof(s_kd) + sizeof(s_qd) + sizeof(s_kr) + sizeof(s_gt) +
                  sizeof(s_beta) + sizeof(s_mqk) + sizeof(s_inv) <= 48 * 1024,
                  "K2 static shared exceeds the 48 KB default per-block limit");

    float* S = state + (size_t)h * kD * kD;

    for (int c = 0; c < n_chunks; ++c) {
        const int t0 = c * CHUNK;
        const int actual_len = min(CHUNK, T - t0);
        const size_t tile = (size_t)h * n_chunks + c;

        // Cooperative load of this chunk's workspace tile — every thread needs all
        // of it (broadcast), so one coalesced stride-128 pass per array.
        const float* in_kd   = ws_kd   + tile * WsLayout::kKD;
        const float* in_qd   = ws_qd   + tile * WsLayout::kQD;
        const float* in_kr   = ws_kr   + tile * WsLayout::kKR;
        const float* in_gt   = ws_gt   + tile * WsLayout::kGT;
        const float* in_beta = ws_beta + tile * WsLayout::kBeta;
        const float* in_mqk  = ws_mqk  + tile * WsLayout::kMqk;
        const float* in_inv  = ws_inv  + tile * WsLayout::kInv;
        for (int i = j; i < CHUNK * kD; i += kD) {
            s_kd[i] = in_kd[i]; s_qd[i] = in_qd[i]; s_kr[i] = in_kr[i];
        }
        s_gt[j] = in_gt[j];
        if (j < CHUNK) s_beta[j] = in_beta[j];
        for (int i = j; i < CHUNK * CHUNK; i += kD) {
            s_mqk[i] = in_mqk[i]; s_inv[i] = in_inv[i];
        }
        __syncthreads();

        // RHS_t = beta_t * (v_t - k_decayed_t @ S_in). All 16 read the SAME S_in
        // (the state is only written at the end of the chunk), so they are computed
        // together. Both t-loops are unrolled so RHS/U index with compile-time
        // constants and stay in registers.
        float RHS[CHUNK];
#pragma unroll
        for (int t = 0; t < CHUNK; ++t) {
            float proj = 0.0f;
            for (int i = 0; i < kD; ++i) proj += s_kd[t * kD + i] * S[(size_t)j * kD + i];
            const float v_val = (t < actual_len)
                ? v[((size_t)(t0 + t) * n_head + h) * kD + j] : 0.0f;
            RHS[t] = s_beta[t] * (v_val - proj);
        }

        // U = INV @ RHS. INV is lower triangular by construction (K1's substitution
        // never writes above the diagonal), so s <= t covers every nonzero entry.
        float U[CHUNK];
#pragma unroll
        for (int t = 0; t < CHUNK; ++t) {
            float acc = 0.0f;
#pragma unroll
            for (int s = 0; s <= t; ++s) acc += s_inv[t * CHUNK + s] * RHS[s];
            U[t] = acc;
        }

        // O = Q_decayed @ S_in + Mqk @ U. The loop runs the full unrolled CHUNK with
        // the STORE guarded, rather than a runtime-bounded loop, so U/Mqk keep their
        // compile-time indexing; a store past actual_len would write past [T,...].
#pragma unroll
        for (int t = 0; t < CHUNK; ++t) {
            if (t < actual_len) {
                float proj = 0.0f;
                for (int i = 0; i < kD; ++i) proj += s_qd[t * kD + i] * S[(size_t)j * kD + i];
                float mu = 0.0f;
#pragma unroll
                for (int s = 0; s <= t; ++s) mu += s_mqk[t * CHUNK + s] * U[s];
                out[((size_t)(t0 + t) * n_head + h) * kD + j] = proj + mu;
            }
        }

        // S_out[i][j] = S_in[i][j]*exp(g_total[i]) + sum_t k_restored_t[i]*U[t].
        // Thread j owns the whole physical row j — the same exclusive ownership
        // kda_decode_step_kernel uses — so the read-modify-write needs no barrier
        // against other threads; nothing else ever touches this row.
        for (int i = 0; i < kD; ++i) {
            float acc = S[(size_t)j * kD + i] * __expf(s_gt[i]);
#pragma unroll
            for (int t = 0; t < CHUNK; ++t) acc += s_kr[t * kD + i] * U[t];
            S[(size_t)j * kD + i] = acc;
        }
        __syncthreads();   // the shared tile must be fully consumed before the next load
    }
}

}  // namespace

bool k3_kda_chunk_prefill(float* out, float* state,
                          const float* q, const float* k, const float* v,
                          const float* g_raw, const float* beta_logit,
                          const float* A,
                          int T, int head_dim, int n_head,
                          float lower_bound, float l2_eps, cudaStream_t stream) {
    if (T <= 0 || n_head <= 0) return false;
    if (head_dim != kD) return false;    // sized against K3's 128, not generic — see K1

    const int n_chunks = (T + CHUNK - 1) / CHUNK;
    const size_t n_tiles = (size_t)n_head * n_chunks;

    // Stream-ordered workspace: freed on the same stream after K2, so the frees
    // execute only once the scan has consumed it. ~412 KB per (chunk,head) pair
    // per 1e3 tiles — at T=2048, n_head=12 that is 1536 tiles = ~13 MB, transient.
    struct Arena {
        std::vector<void*> bufs;
        cudaStream_t s;
        bool ok = true;
        template <class E> E* take(size_t count) {
            if (!ok) return nullptr;
            void* p = nullptr;
            if (cudaMallocAsync(&p, count * sizeof(E), s) != cudaSuccess) { ok = false; return nullptr; }
            bufs.push_back(p);
            return static_cast<E*>(p);
        }
        void free_all() { for (void* b : bufs) cudaFreeAsync(b, s); bufs.clear(); }
    } arena{{}, stream};

    float* ws_kd   = arena.take<float>(n_tiles * WsLayout::kKD);
    float* ws_qd   = arena.take<float>(n_tiles * WsLayout::kQD);
    float* ws_kr   = arena.take<float>(n_tiles * WsLayout::kKR);
    float* ws_gt   = arena.take<float>(n_tiles * WsLayout::kGT);
    float* ws_beta = arena.take<float>(n_tiles * WsLayout::kBeta);
    float* ws_mqk  = arena.take<float>(n_tiles * WsLayout::kMqk);
    float* ws_inv  = arena.take<float>(n_tiles * WsLayout::kInv);
    if (!arena.ok) { arena.free_all(); return false; }

    dim3 g1((unsigned)n_chunks, (unsigned)n_head);
    k3_pdl_launch(g1, kD, 0, stream, k3_kda_chunk_prep_kernel,
                  ws_kd, ws_qd, ws_kr, ws_gt, ws_beta, ws_mqk, ws_inv,
                  q, k, g_raw, beta_logit, A, T, n_head, lower_bound, l2_eps);

    k3_pdl_launch((unsigned)n_head, kD, 0, stream, k3_kda_chunk_scan_kernel,
                  out, state, v, ws_kd, ws_qd, ws_kr, ws_gt, ws_beta, ws_mqk, ws_inv,
                  T, n_head, n_chunks);

    arena.free_all();
    return true;
}

}  // namespace k3
}  // namespace kernels
}  // namespace sparkinfer
