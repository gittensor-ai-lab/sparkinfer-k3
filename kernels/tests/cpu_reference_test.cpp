// CPU reference correctness tests for the sparkinfer kernel algorithms.
//
// These re-implement each CUDA kernel's exact numerical algorithm in plain C++
// and check it against an INDEPENDENT double-precision ground truth (different
// loop order / higher precision). A match is real evidence the algorithm is
// correct; the device-side sm_120 compile (see .cudaverify) separately proves
// the same code targets the RTX 5090. Together they cover "valid for the 5090"
// and "computes the right thing" — the two halves a GPU-less environment allows.
//
// Build: g++ -O2 -std=c++17 cpu_reference_test.cpp -o cpu_reference_test

#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <utility>

using std::vector;
static std::mt19937 rng(1234);
static float frand() { return std::uniform_real_distribution<float>(-1.f, 1.f)(rng); }

// Round a float to bf16 (round-to-nearest-even), returned as float. Models the
// __float2bfloat16 round-trip the kernels do when reading/writing bf16 rows.
static float to_bf16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return f;  // NaN: leave as-is
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    u &= 0xffff0000u;
    float r; std::memcpy(&r, &u, 4);
    return r;
}

static int g_fail = 0;
static void check(const char* name, double max_err, double tol) {
    bool ok = max_err <= tol;
    printf("  [%s] %-34s max_err=%.3e (tol=%.0e)\n", ok ? "PASS" : "FAIL", name, max_err, tol);
    if (!ok) g_fail++;
}

// The inverse assertion: a negative control has to STAY apart. A "wrong variant differs"
// case that quietly converges is worse than no case at all, because the suite still
// reports PASS while checking nothing.
static void check_min(const char* name, double diff, double floor_) {
    bool ok = diff >= floor_;
    printf("  [%s] %-34s diff=%.3e (min=%.0e)\n", ok ? "PASS" : "FAIL", name, diff, floor_);
    if (!ok) g_fail++;
}

static float silu(float x) { return x / (1.f + std::exp(-x)); }

// ---------------------------------------------------------------------------
// 1. Flash decode: online-softmax (kernel algorithm) vs naive full softmax.
// ---------------------------------------------------------------------------
static double test_attention(int HD, int kvlen) {
    vector<float> q(HD), K(kvlen * HD), V(kvlen * HD);
    for (auto& x : q) x = frand();
    for (auto& x : K) x = frand();
    for (auto& x : V) x = frand();
    const float scale = 1.f / std::sqrt((float)HD);

    // Ground truth (double precision, two-pass softmax).
    vector<double> scores(kvlen);
    double mx = -1e300;
    for (int t = 0; t < kvlen; t++) {
        double d = 0; for (int i = 0; i < HD; i++) d += (double)q[i] * K[t * HD + i];
        scores[t] = d * scale; mx = std::max(mx, scores[t]);
    }
    double denom = 0; for (int t = 0; t < kvlen; t++) denom += std::exp(scores[t] - mx);
    vector<double> ref(HD, 0);
    for (int t = 0; t < kvlen; t++) {
        double p = std::exp(scores[t] - mx) / denom;
        for (int i = 0; i < HD; i++) ref[i] += p * V[t * HD + i];
    }

    // Kernel algorithm: single-pass online softmax in float.
    float m = -1e30f, l = 0.f; vector<float> acc(HD, 0.f);
    for (int t = 0; t < kvlen; t++) {
        float d = 0; for (int i = 0; i < HD; i++) d += q[i] * K[t * HD + i];
        float score = d * scale;
        float m_new = std::max(m, score);
        float corr = std::exp(m - m_new), p = std::exp(score - m_new);
        l = l * corr + p;
        for (int i = 0; i < HD; i++) acc[i] = acc[i] * corr + p * V[t * HD + i];
        m = m_new;
    }
    double err = 0; for (int i = 0; i < HD; i++) err = std::max(err, std::abs(acc[i] / l - ref[i]));
    return err;
}

// ---------------------------------------------------------------------------
// 2. Router top-k: kernel mask-argmax algorithm vs sort-based reference.
// ---------------------------------------------------------------------------
static double test_router(int E, int K) {
    vector<float> logits(E); for (auto& x : logits) x = frand();

    // Reference: stable sort by (value desc, index asc), take K; softmax over them.
    vector<int> idx(E); for (int i = 0; i < E; i++) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
        return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); });
    vector<int> ref_id(idx.begin(), idx.begin() + K);
    double rmx = logits[ref_id[0]], rden = 0;
    for (int j = 0; j < K; j++) rden += std::exp((double)logits[ref_id[j]] - rmx);
    vector<double> ref_w(K);
    for (int j = 0; j < K; j++) ref_w[j] = std::exp((double)logits[ref_id[j]] - rmx) / rden;

    // Kernel algorithm: K passes of arg-max with masking, then softmax over picks.
    vector<float> s = logits; vector<int> sel(K); vector<float> sl(K);
    for (int j = 0; j < K; j++) {
        float best = -1e30f; int bi = -1;
        for (int e = 0; e < E; e++) if (s[e] > best || (s[e] == best && e < bi)) { best = s[e]; bi = e; }
        sel[j] = bi; sl[j] = best; s[bi] = -1e30f;
    }
    float kmx = sl[0]; for (int j = 1; j < K; j++) kmx = std::max(kmx, sl[j]);
    float kden = 0; for (int j = 0; j < K; j++) kden += std::exp(sl[j] - kmx);

    double err = 0;
    for (int j = 0; j < K; j++) {
        if (sel[j] != ref_id[j]) err = std::max(err, 1.0);
        err = std::max(err, std::abs(std::exp(sl[j] - kmx) / kden - ref_w[j]));
    }
    return err;
}

// ---------------------------------------------------------------------------
// 3. SwiGLU expert FFN: kernel math (float) vs double ground truth.
// ---------------------------------------------------------------------------
static double test_swiglu(int H, int F) {
    vector<float> X(H), gate(H * F), up(H * F), down(F * H);
    for (auto& x : X) x = frand();
    for (auto& x : gate) x = frand() * 0.1f;
    for (auto& x : up) x = frand() * 0.1f;
    for (auto& x : down) x = frand() * 0.1f;
    const float w = 0.37f;

    vector<double> hbuf_d(F), ref(H, 0);
    for (int f = 0; f < F; f++) {
        double g = 0, u = 0;
        for (int h = 0; h < H; h++) { g += (double)X[h] * gate[h * F + f]; u += (double)X[h] * up[h * F + f]; }
        hbuf_d[f] = (g / (1.0 + std::exp(-g))) * u;
    }
    for (int h = 0; h < H; h++) { double y = 0; for (int f = 0; f < F; f++) y += hbuf_d[f] * down[f * H + h]; ref[h] = w * y; }

    vector<float> hbuf(F), acc(H, 0.f);
    for (int f = 0; f < F; f++) {
        float g = 0, u = 0;
        for (int h = 0; h < H; h++) { g += X[h] * gate[h * F + f]; u += X[h] * up[h * F + f]; }
        hbuf[f] = silu(g) * u;
    }
    for (int h = 0; h < H; h++) { float y = 0; for (int f = 0; f < F; f++) y += hbuf[f] * down[f * H + h]; acc[h] = w * y; }

    double err = 0; for (int h = 0; h < H; h++) err = std::max(err, std::abs((double)acc[h] - ref[h]));
    return err;
}

// ---------------------------------------------------------------------------
// 4. GEMM: tiled accumulation order vs double triple-loop.
// ---------------------------------------------------------------------------
static double test_gemm(int M, int N, int Kd) {
    vector<float> A(M * Kd), B(Kd * N);
    for (auto& x : A) x = frand();
    for (auto& x : B) x = frand();
    vector<double> ref(M * N, 0);
    for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) { double s = 0; for (int k = 0; k < Kd; k++) s += (double)A[i*Kd+k]*B[k*N+j]; ref[i*N+j] = s; }

    const int TILE = 16; vector<float> C(M * N, 0.f);
    for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
        float acc = 0.f;
        for (int k0 = 0; k0 < Kd; k0 += TILE) { float t = 0.f; for (int k = k0; k < std::min(k0+TILE,Kd); k++) t += A[i*Kd+k]*B[k*N+j]; acc += t; }
        C[i*N+j] = acc;
    }
    double err = 0; for (int i = 0; i < M*N; i++) err = std::max(err, std::abs((double)C[i] - ref[i]));
    return err;
}

// ---------------------------------------------------------------------------
// 5. RMSNorm: kernel math vs double ground truth.
// ---------------------------------------------------------------------------
static double test_rmsnorm(int cols) {
    vector<float> x(cols), wt(cols); for (auto& v : x) v = frand(); for (auto& v : wt) v = frand();
    const float eps = 1e-6f;
    double ss = 0; for (int c = 0; c < cols; c++) ss += (double)x[c]*x[c];
    double inv = 1.0 / std::sqrt(ss / cols + eps);
    vector<double> ref(cols); for (int c = 0; c < cols; c++) ref[c] = x[c]*inv*wt[c];

    float fss = 0; for (int c = 0; c < cols; c++) fss += x[c]*x[c];
    float finv = 1.f/std::sqrt(fss/cols + eps);
    double err = 0; for (int c = 0; c < cols; c++) err = std::max(err, std::abs((double)(x[c]*finv*wt[c]) - ref[c]));
    return err;
}

// ---------------------------------------------------------------------------
// 5b. Vectorized RMSNorm (PR #44): the kernel reads the row in 8-wide (uint4 =
//     8 bf16) packs and accumulates the sum-of-squares per-pack with FMA. Only
//     the per-thread element grouping in the SS reduction changes (FP assoc.);
//     this checks the 8-wide grouped reduction still matches the fp64 ground
//     truth. cols here are multiples of 8, as on every real RMSNorm call site.
// ---------------------------------------------------------------------------
static double test_rmsnorm_vec8(int cols) {
    vector<float> x(cols), wt(cols);
    for (auto& v : x) v = frand();
    for (auto& v : wt) v = frand();
    const float eps = 1e-6f;

    double ss = 0; for (int c = 0; c < cols; c++) ss += (double)x[c]*x[c];
    double inv = 1.0 / std::sqrt(ss / cols + eps);
    vector<double> ref(cols); for (int c = 0; c < cols; c++) ref[c] = x[c]*inv*wt[c];

    // 8-wide packs, FMA accumulation (mirrors rn_unpack8 + __fmaf_rn).
    const int npack = cols >> 3;
    float fss = 0.f;
    for (int p = 0; p < npack; p++) {
        #pragma GCC unroll 8
        for (int j = 0; j < 8; j++) { float v = x[p*8+j]; fss = std::fma(v, v, fss); }
    }
    for (int c = (npack<<3); c < cols; c++) { float v = x[c]; fss = std::fma(v, v, fss); }
    float finv = 1.f/std::sqrt(fss/cols + eps);
    double err = 0; for (int c = 0; c < cols; c++) err = std::max(err, std::abs((double)(x[c]*finv*wt[c]) - ref[c]));
    return err;
}

// ---------------------------------------------------------------------------
// 5c. add_rmsnorm2 sequencing (PR #44): the fused residual+norm kernel must keep
//     its exact numeric sequencing under vectorization — SS accumulates on the
//     fp32 sum (x+residual), the sum is round-tripped through bf16 into out_sum,
//     and the norm pass re-reads that bf16 sum. This models that scalar vs 8-wide
//     grouped sequencing produce the same normalized output (bf16-exact), so the
//     vectorized rewrite is byte-faithful, not just close.
// ---------------------------------------------------------------------------
static double test_add_rmsnorm2_seq(int cols) {
    vector<float> x(cols), r(cols), wt(cols);
    for (auto& v : x)  v = frand();
    for (auto& v : r)  v = frand();
    for (auto& v : wt) v = frand();
    const float eps = 1e-6f;

    // Scalar reference sequencing (original kernel).
    vector<float> sum_bf(cols);
    float ss_s = 0.f;
    for (int c = 0; c < cols; c++) {
        float v = x[c] + r[c];          // fp32 sum
        sum_bf[c] = to_bf16(v);          // out_sum stored as bf16
        ss_s = std::fma(v, v, ss_s);     // SS on the fp32 sum
    }
    float inv_s = 1.f/std::sqrt(ss_s/cols + eps);
    vector<float> norm_s(cols);
    for (int c = 0; c < cols; c++) norm_s[c] = to_bf16(sum_bf[c] * inv_s * wt[c]);

    // 8-wide grouped sequencing (PR kernel): same per-element ops, packed.
    const int npack = cols >> 3;
    vector<float> sum_bf2(cols);
    float ss_v = 0.f;
    for (int p = 0; p < npack; p++)
        for (int j = 0; j < 8; j++) {
            float v = x[p*8+j] + r[p*8+j];
            sum_bf2[p*8+j] = to_bf16(v);
            ss_v = std::fma(v, v, ss_v);
        }
    float inv_v = 1.f/std::sqrt(ss_v/cols + eps);
    vector<float> norm_v(cols);
    for (int p = 0; p < npack; p++)
        for (int j = 0; j < 8; j++)
            norm_v[p*8+j] = to_bf16(sum_bf2[p*8+j] * inv_v * wt[p*8+j]);

    double err = 0;
    for (int c = 0; c < cols; c++) {
        err = std::max(err, std::abs((double)sum_bf2[c] - sum_bf[c]));
        err = std::max(err, std::abs((double)norm_v[c]  - norm_s[c]));
    }
    return err;  // expect 0: scalar and 8-wide grouping are bit-identical here
}

// ---------------------------------------------------------------------------
// argmax two-pass (decode): the multi-block scan + final reduce must return the
// SAME index as a serial argmax, including the smallest-index tie-break.
// ---------------------------------------------------------------------------
static double test_argmax_twopass(int vocab, int nblocks, bool ties) {
    vector<float> L(vocab);
    for (auto& v : L) v = frand();
    if (ties) { for (auto& v : L) v = 0.f; for (int i : {1000 % vocab, 5, 77777 % vocab, 250}) L[i] = 7.f; }

    auto merge = [](float& bv, int& bi, float ov, int oi) {
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    };
    // serial ground truth (smallest index on ties)
    float gv = -1e30f; int gi = 0;
    for (int v = 0; v < vocab; v++) merge(gv, gi, L[v], v);

    // pass 1: nblocks grid-stride partials, each block 256 threads
    const int BT = 256;
    vector<float> pv(nblocks); vector<int> pi(nblocks);
    for (int b = 0; b < nblocks; b++) {
        float bbv = -1e30f; int bbi = 0;
        vector<float> tv(BT, -1e30f); vector<int> ti(BT, 0);
        for (int t = 0; t < BT; t++)
            for (int v = b * BT + t; v < vocab; v += BT * nblocks) merge(tv[t], ti[t], L[v], v);
        for (int t = 0; t < BT; t++) merge(bbv, bbi, tv[t], ti[t]);
        pv[b] = bbv; pi[b] = bbi;
    }
    // pass 2: reduce partials
    float rv = -1e30f; int ri = 0;
    for (int b = 0; b < nblocks; b++) merge(rv, ri, pv[b], pi[b]);
    return (double)std::abs(ri - gi);   // expect 0: same index as serial argmax
}

// ---------------------------------------------------------------------------
// 9. Kimi K3 MLA NoPE decode attention: TILED ONLINE SOFTMAX vs a two-pass
//    double-precision reference.
// ---------------------------------------------------------------------------
// This models mla_decode_attn_kernel (kernels/csrc/cuda/kimi_k3/k3_kernels.cu)
// structurally, not just formulaically: BLOCK 256 / 8 warps / kMlaCtxTile 128,
// one warp per scored token with lanes striding over d, the __shfl_down_sync
// reduction tree, and the shared-memory block reductions in warp order. So the
// SUMMATION ORDER matches the device kernel, and the check is on the algorithm
// as scheduled rather than on the algebra alone.
//
// Why it earns a slot here. The GPU coherence test (kimi_k3_numeric_test.cu)
// runs n_ctx 48 — one tile — which is exactly the case where an online softmax
// degenerates to the two-pass one it replaced. The interesting states (a running
// max that a later tile raises, an accumulator rescaled by corr) only appear at
// n_ctx > 128, and they are reachable with no GPU at all.
static const int MLA_BLOCK = 256;
static const int MLA_NWARP = MLA_BLOCK / 32;
static const int MLA_TILE  = 128;

// __shfl_down_sync(0xffffffff, v, off) for off = 16..1: lane 0 ends with the sum.
static float mla_warp_sum(vector<float> v) {
    for (int off = 16; off > 0; off >>= 1)
        for (int i = 0; i < off; i++) v[i] += v[i + off];
    return v[0];
}
static float mla_warp_max(vector<float> v) {
    for (int off = 16; off > 0; off >>= 1)
        for (int i = 0; i < off; i++) v[i] = std::fmax(v[i], v[i + off]);
    return v[0];
}
// block_sum<BLOCK> / block_max<BLOCK>: warp reduce, then thread 0 folds the
// per-warp partials in warp order.
static float mla_block_sum(const vector<float>& per_thread) {
    float s = 0.f;
    for (int w = 0; w < MLA_NWARP; w++)
        s += mla_warp_sum(vector<float>(per_thread.begin() + w * 32,
                                        per_thread.begin() + w * 32 + 32));
    return s;
}
static float mla_block_max(const vector<float>& per_thread) {
    float s = -1e30f;
    for (int w = 0; w < MLA_NWARP; w++)
        s = std::fmax(s, mla_warp_max(vector<float>(per_thread.begin() + w * 32,
                                                    per_thread.begin() + w * 32 + 32)));
    return s;
}

static double test_mla_decode_attn(int key_length, int kv_lora, int v_dim,
                                   int n_ctx, int n_head) {
    const int rope = key_length - kv_lora;
    (void)rope;
    // Scale is the caller's business (1/sqrt(qk_nope+rope), NOT 1/sqrt(key_length)
    // — see the trap in kernels/include/sparkinfer/kernels/kimi_k3.h); the kernel
    // just multiplies by it, so any positive value exercises the same code.
    const float scale = 1.f / std::sqrt((float)key_length);

    vector<float> q((size_t)key_length * n_head), K((size_t)key_length * n_ctx),
                  wv((size_t)kv_lora * v_dim * n_head);
    for (auto& x : q) x = frand();
    for (auto& x : K) x = frand();
    for (auto& x : wv) x = 0.2f * frand();

    double worst = 0.0;
    for (int h = 0; h < n_head; h++) {
        // ---- ground truth: two-pass softmax in double ----
        vector<double> sc(n_ctx);
        double mx = -1e300;
        for (int t = 0; t < n_ctx; t++) {
            double d = 0;
            for (int i = 0; i < key_length; i++)
                d += (double)q[(size_t)h * key_length + i] * K[(size_t)t * key_length + i];
            sc[t] = d * scale;
            mx = std::max(mx, sc[t]);
        }
        double sum = 0;
        for (int t = 0; t < n_ctx; t++) { sc[t] = std::exp(sc[t] - mx); sum += sc[t]; }
        vector<double> latent(kv_lora, 0.0);
        for (int r = 0; r < kv_lora; r++) {
            double a = 0;
            for (int t = 0; t < n_ctx; t++) a += sc[t] * K[(size_t)t * key_length + r];
            latent[r] = a / sum;
        }
        vector<double> want(v_dim);
        for (int v = 0; v < v_dim; v++) {
            double a = 0;
            for (int r = 0; r < kv_lora; r++)
                a += (double)wv[((size_t)h * v_dim + v) * kv_lora + r] * latent[r];
            want[v] = a;
        }

        // ---- kernel algorithm ----
        vector<float> s_q(q.begin() + (size_t)h * key_length,
                          q.begin() + (size_t)h * key_length + key_length);
        vector<float> s_acc(kv_lora, 0.f), s_p(MLA_TILE, 0.f);
        float m = -1e30f, l = 0.f;

        for (int t0 = 0; t0 < n_ctx; t0 += MLA_TILE) {
            const int tn = std::min(MLA_TILE, n_ctx - t0);

            for (int t = 0; t < tn; t++) {          // one warp per token
                vector<float> part(32, 0.f);
                for (int lane = 0; lane < 32; lane++)
                    for (int d = lane; d < key_length; d += 32)
                        part[lane] += s_q[d] * K[(size_t)(t0 + t) * key_length + d];
                s_p[t] = mla_warp_sum(part) * scale;
            }

            vector<float> tv(MLA_BLOCK, -1e30f);
            for (int tid = 0; tid < MLA_BLOCK; tid++)
                for (int t = tid; t < tn; t += MLA_BLOCK) tv[tid] = std::fmax(tv[tid], s_p[t]);
            const float m_new = std::fmax(m, mla_block_max(tv));
            const float corr = std::exp(m - m_new);   // 0 on the first tile

            vector<float> lv(MLA_BLOCK, 0.f);
            for (int tid = 0; tid < MLA_BLOCK; tid++)
                for (int t = tid; t < tn; t += MLA_BLOCK) {
                    const float e = std::exp(s_p[t] - m_new);
                    s_p[t] = e;
                    lv[tid] += e;
                }
            l = l * corr + mla_block_sum(lv);
            m = m_new;

            for (int r = 0; r < kv_lora; r++) {
                float a = s_acc[r] * corr;
                for (int t = 0; t < tn; t++)
                    a += s_p[t] * K[(size_t)(t0 + t) * key_length + r];
                s_acc[r] = a;
            }
        }

        const float inv = l > 0.f ? 1.f / l : 0.f;
        for (int r = 0; r < kv_lora; r++) s_acc[r] *= inv;

        for (int v = 0; v < v_dim; v++) {           // one warp per output element
            vector<float> part(32, 0.f);
            for (int lane = 0; lane < 32; lane++)
                for (int r = lane; r < kv_lora; r += 32)
                    part[lane] += wv[((size_t)h * v_dim + v) * kv_lora + r] * s_acc[r];
            worst = std::max(worst, std::abs((double)mla_warp_sum(part) - want[v]));
        }
    }
    return worst;
}

// ---------------------------------------------------------------------------
// 9b. Kimi K3 MLA decode, HEAD-BATCHED AND CONTEXT-SPLIT: the schedule
//     mla_decode_attn_hbatch_kernel + mla_decode_combine_kernel actually run.
// ---------------------------------------------------------------------------
// Three things differ from 9 and each can be wrong on its own, so each is modelled
// rather than assumed:
//
//   - HPB heads share a block. A lane loads k_cache[t][d] once and multiplies it into
//     HPB accumulators. Getting the s_q stride wrong here reads another head's query
//     and still produces plausible numbers. The device kernel scores kMlaTokensPerWarp
//     tokens per pass over the staged queries; that is a scheduling choice and not a
//     summation-order one — a given (head, token) still accumulates d = lane, lane+32,
//     … in that order under the same shuffle tree — so one token at a time models it.
//   - The per-tile softmax reduces over ONE WARP (32 lanes, t += 32), not over BLOCK
//     threads (t += 256). Different tree, different rounding — modelled exactly so the
//     tolerance is measuring the algorithm and not this.
//   - The slice partials are UNNORMALISED and merged by a second kernel that rescales
//     each slice by exp(m_i - m) before dividing by the merged l. A slice boundary
//     landing mid-tile, or an m that a later slice raises, is only exercised here.
//
// n_ctx values are chosen to land on and off tile and slice boundaries. Modelled on
// the CPU because reaching this path on a device needs a context past
// kMlaSplitMinCtx, and the whole point of a CPU reference is that CI can run it.
static double test_mla_decode_hbatch(int key_length, int kv_lora, int v_dim,
                                     int n_ctx, int n_head, int hpb, int splits) {
    const float scale = 1.f / std::sqrt((float)key_length);

    vector<float> q((size_t)key_length * n_head), K((size_t)key_length * n_ctx),
                  wv((size_t)kv_lora * v_dim * n_head);
    for (auto& x : q) x = frand();
    for (auto& x : K) x = frand();
    for (auto& x : wv) x = 0.2f * frand();

    // --- the batched slice kernel: one entry per (head, slice) ---
    vector<float> part_acc((size_t)n_head * splits * kv_lora, 0.f);
    vector<float> part_m((size_t)n_head * splits, -1e30f);
    vector<float> part_l((size_t)n_head * splits, 0.f);

    const int chunk = (n_ctx + splits - 1) / splits;
    for (int hg = 0; hg * hpb < n_head; hg++) {
        for (int sp = 0; sp < splits; sp++) {
            const int t_beg = sp * chunk;
            const int t_end = std::min(n_ctx, t_beg + chunk);
            if (t_end <= t_beg) continue;                 // empty slice contributes nothing

            // s_q: hpb staged queries. acc: RSLOTS x hpb, one r per thread.
            vector<float> s_p((size_t)hpb * MLA_TILE, 0.f);
            vector<float> acc((size_t)hpb * kv_lora, 0.f);
            vector<float> m(hpb, -1e30f), l(hpb, 0.f), corr(hpb, 0.f);

            for (int t0 = t_beg; t0 < t_end; t0 += MLA_TILE) {
                const int tn = std::min(MLA_TILE, t_end - t0);

                // 1. score: one warp per token, lanes over d, hpb accumulators
                for (int t = 0; t < tn; t++) {
                    for (int hh = 0; hh < hpb; hh++) {
                        const int h = hg * hpb + hh;
                        vector<float> part(32, 0.f);
                        for (int lane = 0; lane < 32; lane++)
                            for (int d = lane; d < key_length; d += 32)
                                part[lane] += q[(size_t)h * key_length + d] *
                                              K[(size_t)(t0 + t) * key_length + d];
                        s_p[(size_t)hh * MLA_TILE + t] = mla_warp_sum(part) * scale;
                    }
                }

                // 2. online softmax, ONE WARP per head: t += 32, then the shuffle tree
                for (int hh = 0; hh < hpb; hh++) {
                    float* pr = &s_p[(size_t)hh * MLA_TILE];
                    vector<float> tv(32, -1e30f);
                    for (int lane = 0; lane < 32; lane++)
                        for (int t = lane; t < tn; t += 32) tv[lane] = std::fmax(tv[lane], pr[t]);
                    const float m_new = std::fmax(m[hh], mla_warp_max(tv));
                    corr[hh] = std::exp(m[hh] - m_new);

                    vector<float> lv(32, 0.f);
                    for (int lane = 0; lane < 32; lane++)
                        for (int t = lane; t < tn; t += 32) {
                            const float e = std::exp(pr[t] - m_new);
                            pr[t] = e;
                            lv[lane] += e;
                        }
                    l[hh] = l[hh] * corr[hh] + mla_warp_sum(lv);
                    m[hh] = m_new;
                }

                // 3. latent: one k_cache load per (t, r) feeding hpb accumulators
                for (int hh = 0; hh < hpb; hh++)
                    for (int r = 0; r < kv_lora; r++) acc[(size_t)hh * kv_lora + r] *= corr[hh];
                for (int t = 0; t < tn; t++)
                    for (int r = 0; r < kv_lora; r++) {
                        const float kv = K[(size_t)(t0 + t) * key_length + r];
                        for (int hh = 0; hh < hpb; hh++)
                            acc[(size_t)hh * kv_lora + r] += s_p[(size_t)hh * MLA_TILE + t] * kv;
                    }
            }

            for (int hh = 0; hh < hpb; hh++) {
                const int h = hg * hpb + hh;
                for (int r = 0; r < kv_lora; r++)
                    part_acc[((size_t)h * splits + sp) * kv_lora + r] = acc[(size_t)hh * kv_lora + r];
                part_m[(size_t)h * splits + sp] = m[hh];
                part_l[(size_t)h * splits + sp] = l[hh];
            }
        }
    }

    // --- the combine kernel, then compare against a two-pass float64 reference ---
    double worst = 0.0;
    for (int h = 0; h < n_head; h++) {
        vector<double> sc(n_ctx);
        double mx = -1e300;
        for (int t = 0; t < n_ctx; t++) {
            double d = 0;
            for (int i = 0; i < key_length; i++)
                d += (double)q[(size_t)h * key_length + i] * K[(size_t)t * key_length + i];
            sc[t] = d * scale;
            mx = std::max(mx, sc[t]);
        }
        double sum = 0;
        for (int t = 0; t < n_ctx; t++) { sc[t] = std::exp(sc[t] - mx); sum += sc[t]; }
        vector<double> want(v_dim);
        {
            vector<double> latent(kv_lora, 0.0);
            for (int r = 0; r < kv_lora; r++) {
                double a = 0;
                for (int t = 0; t < n_ctx; t++) a += sc[t] * K[(size_t)t * key_length + r];
                latent[r] = a / sum;
            }
            for (int v = 0; v < v_dim; v++) {
                double a = 0;
                for (int r = 0; r < kv_lora; r++)
                    a += (double)wv[((size_t)h * v_dim + v) * kv_lora + r] * latent[r];
                want[v] = a;
            }
        }

        float mm = -1e30f;
        for (int i = 0; i < splits; i++) mm = std::fmax(mm, part_m[(size_t)h * splits + i]);
        vector<float> w(splits);
        float ll = 0.f;
        for (int i = 0; i < splits; i++) {
            w[i] = std::exp(part_m[(size_t)h * splits + i] - mm);
            ll += part_l[(size_t)h * splits + i] * w[i];
        }
        const float inv = ll > 0.f ? 1.f / ll : 0.f;
        vector<float> lat(kv_lora, 0.f);
        for (int r = 0; r < kv_lora; r++) {
            float a = 0.f;
            for (int i = 0; i < splits; i++)
                a += part_acc[((size_t)h * splits + i) * kv_lora + r] * w[i];
            lat[r] = a * inv;
        }
        for (int v = 0; v < v_dim; v++) {           // one warp per output element
            vector<float> part(32, 0.f);
            for (int lane = 0; lane < 32; lane++)
                for (int r = lane; r < kv_lora; r += 32)
                    part[lane] += wv[((size_t)h * v_dim + v) * kv_lora + r] * lat[r];
            worst = std::max(worst, std::abs((double)mla_warp_sum(part) - want[v]));
        }
    }
    return worst;
}

// NEGATIVE CONTROL for the merge. Each slice returns an UNNORMALISED latent under its
// OWN running max, so the combine has to rescale slice i by exp(m_i - max_i m_i) before
// summing. Dropping that rescale — just adding the partials and dividing by the summed
// l — is the natural-looking bug, and on well-conditioned inputs it produces a finite,
// plausible vector rather than anything obviously broken.
//
// This runs the same two-slice split twice, once merged correctly and once without the
// rescale, and returns how far apart they are. It must be LARGE. If it ever collapses
// toward zero, the per-slice maxima have stopped differing and the tolerance checks
// above are no longer exercising the merge at all.
static double test_mla_split_merge_needs_rescale() {
    const int n = 8;
    // Two slices whose score ranges are far apart, which is the case the rescale is for.
    float m0 = 12.0f, m1 = -3.0f;
    vector<float> a0(n), a1(n);
    for (int r = 0; r < n; r++) { a0[r] = frand(); a1[r] = frand(); }
    const float l0 = 3.5f, l1 = 2.25f;

    const float m = std::fmax(m0, m1);
    const float w0 = std::exp(m0 - m), w1 = std::exp(m1 - m);
    const float lr = l0 * w0 + l1 * w1;

    double worst = 0.0;
    for (int r = 0; r < n; r++) {
        const double right = (a0[r] * w0 + a1[r] * w1) / lr;
        const double naive = (a0[r] + a1[r]) / (l0 + l1);      // the missing-rescale bug
        worst = std::max(worst, std::abs(right - naive));
    }
    return worst;
}

// ---------------------------------------------------------------------------
// 12. Tensor-parallel attention head banding.
//
// Two separate claims hold up kimi_k3_tp's banded attention, and neither needs a
// GPU to check.
//
// (a) THE LAYOUTS. Every per-head tensor in K3's attention carries the head as its
//     SLOWEST index, which is the only reason a rank can take its band with a flat
//     pointer offset of h0*per_head. Each case below writes the offset the forward
//     uses next to an indexer derived independently from the shape in
//     kernels/include/sparkinfer/kernels/kimi_k3.h, and fails if they disagree.
//     The conv state is the one that bites: [d_conv-1, d_inner] with the TIME index
//     fastest means channel-major, so a channel's history is contiguous. Read the
//     other way — time-major over all channels, which is the same shape written in
//     the other order — a flat per-head offset silently addresses another head's
//     history. test_band_conv_state_is_channel_major pins which reading is assumed.
//
// (b) THE PARTITION. attn_output is reduced over a column band per rank, so the
//     answer is a sum of tp_size partial dot products instead of one. That is a
//     REASSOCIATION, not a rewrite: the same products are summed in a different
//     order, so the result is within rounding of the full-width dot but NOT bitwise
//     equal to it. Both halves are asserted — that it agrees to f32 rounding, and
//     that it is genuinely a different summation order rather than an accident of
//     small test data.
// ---------------------------------------------------------------------------
static int g_ne_fail = 0;
static void check_differs(const char* name, bool differs) {
    printf("  [%s] %-34s differs=%s\n", differs ? "PASS" : "FAIL", name,
           differs ? "yes" : "no");
    if (!differs) g_ne_fail++;
}

// A miniature of the KDA head pipeline: the causal short conv (which carries state
// across tokens) followed by a per-head gate. Run once over all heads, then again
// one band at a time with the offsets the forward applies, over a FRESH copy of the
// same state. Returns the largest disagreement.
//
// `state_stride` and `w_stride` are the per-channel pitches the band offset is built
// from. Passing the right ones (d_conv-1 and d_conv) must reproduce the full run
// exactly — same values, same order, so the tolerance is 0. Passing wrong ones is
// what the companion negative case does.
static double test_band_kda_conv(int n_head, int tp_size, int head_dim,
                                 int state_stride, int w_stride) {
    const int d_conv = 4;
    const int d_inner = n_head * head_dim;
    const int hn = n_head / tp_size;

    vector<float> x(d_inner), w((size_t)d_inner * d_conv), st0((size_t)d_inner * (d_conv - 1));
    for (auto& v : x) v = frand();
    for (auto& v : w) v = frand();
    for (auto& v : st0) v = frand();

    // One channel's step, given base pointers into its own history and filter.
    auto step = [&](const float* hist, const float* filt, float xc, float* hist_out) {
        float acc = 0.f;
        for (int t = 0; t < d_conv - 1; ++t) acc += hist[t] * filt[t];
        acc += xc * filt[d_conv - 1];
        for (int t = 0; t < d_conv - 2; ++t) hist_out[t] = hist[t + 1];
        hist_out[d_conv - 2] = xc;
        return silu(acc);
    };

    // (a) all channels at once.
    vector<float> full(d_inner), st_full = st0;
    for (int c = 0; c < d_inner; ++c)
        full[c] = step(&st_full[(size_t)c * (d_conv - 1)], &w[(size_t)c * d_conv], x[c],
                       &st_full[(size_t)c * (d_conv - 1)]);

    // (b) band by band, from the same starting state, using the forward's offsets.
    vector<float> banded(d_inner, 0.f), st_band = st0;
    for (int r = 0; r < tp_size; ++r) {
        const int c0 = r * hn * head_dim, cn = hn * head_dim;
        float* st_base = &st_band[(size_t)c0 * state_stride];
        const float* w_base = &w[(size_t)c0 * w_stride];
        for (int c = 0; c < cn; ++c)
            banded[c0 + c] = step(st_base + (size_t)c * (d_conv - 1),
                                  w_base + (size_t)c * d_conv, x[c0 + c],
                                  st_base + (size_t)c * (d_conv - 1));
    }

    double worst = 0;
    for (int c = 0; c < d_inner; ++c)
        worst = std::max(worst, (double)std::abs(full[c] - banded[c]));
    for (size_t i = 0; i < st_full.size(); ++i)
        worst = std::max(worst, (double)std::abs(st_full[i] - st_band[i]));
    return worst;
}

// The MLA absorb, whose weight is [qk_nope, kv_lora, n_head] — head slowest, so a
// band is h0*qk_nope*kv_lora away. `w_head_stride` is that pitch, passed in so the
// negative case can supply the wrong one.
static double test_band_mla_absorb(int n_head, int tp_size, int qk_nope, int kv_lora,
                                   long w_head_stride) {
    const int hn = n_head / tp_size;
    vector<float> q((size_t)n_head * qk_nope), wk((size_t)n_head * qk_nope * kv_lora);
    for (auto& v : q) v = frand();
    for (auto& v : wk) v = frand();

    vector<float> full((size_t)n_head * kv_lora), banded((size_t)n_head * kv_lora, 0.f);
    for (int h = 0; h < n_head; ++h)
        for (int r = 0; r < kv_lora; ++r) {
            float a = 0.f;
            for (int d = 0; d < qk_nope; ++d)
                a += wk[(size_t)h * qk_nope * kv_lora + (size_t)r * qk_nope + d] *
                     q[(size_t)h * qk_nope + d];
            full[(size_t)h * kv_lora + r] = a;
        }

    for (int rk = 0; rk < tp_size; ++rk) {
        const int h0 = rk * hn;
        const float* wb = &wk[(size_t)h0 * w_head_stride];
        const float* qb = &q[(size_t)h0 * qk_nope];
        float* ob = &banded[(size_t)h0 * kv_lora];
        for (int h = 0; h < hn; ++h)
            for (int r = 0; r < kv_lora; ++r) {
                float a = 0.f;
                for (int d = 0; d < qk_nope; ++d)
                    a += wb[(size_t)h * qk_nope * kv_lora + (size_t)r * qk_nope + d] *
                         qb[(size_t)h * qk_nope + d];
                ob[(size_t)h * kv_lora + r] = a;
            }
    }

    double worst = 0;
    for (size_t i = 0; i < full.size(); ++i)
        worst = std::max(worst, (double)std::abs(full[i] - banded[i]));
    return worst;
}

// Q8_0 GEMV, one row, split into tp_size column bands. Models the kernel exactly:
// per 32-value block, an f32 scale times a sum of eight 4-wide products accumulated
// in the thread's own partial, then a block reduction. Returns the error of the
// summed bands against a float64 ground truth.
static double test_proj_cols_partition(int K, int tp_size, int rows, int* n_differ) {
    const int nb = K / 32;
    const int BLOCK = 128;
    double worst = 0;
    if (n_differ) *n_differ = 0;

    for (int row = 0; row < rows; ++row) {
        vector<float> x(K);
        vector<int8_t> q(K);
        vector<float> d(nb);
        for (auto& v : x) v = frand();
        for (auto& v : q) v = (int8_t)std::lround(frand() * 100.f);
        for (auto& v : d) v = frand() * 0.01f;

        double want = 0;
        for (int b = 0; b < nb; ++b)
            for (int j = 0; j < 32; ++j)
                want += (double)d[b] * (double)q[b * 32 + j] * (double)x[b * 32 + j];

        // One block of BLOCK threads striding over the blocks it was given, exactly
        // as proj_q8_0_kernel does. `first`/`count` are the band's blocks — which is
        // all the column slice changes: the row's OTHER blocks are still there, at
        // the full row pitch, and are simply not visited.
        auto dot = [&](int first, int count) {
            vector<float> acc(BLOCK, 0.f);
            for (int tid = 0; tid < BLOCK; ++tid)
                for (int b = tid; b < count; b += BLOCK) {
                    float s = 0.f;
                    for (int i = 0; i < 8; ++i)
                        for (int j = 0; j < 4; ++j)
                            s += (float)q[(first + b) * 32 + 4 * i + j] *
                                 x[(first + b) * 32 + 4 * i + j];
                    acc[tid] += d[first + b] * s;
                }
            // Tree reduction over the block, the shape block_sum produces.
            for (int stride = BLOCK / 2; stride > 0; stride >>= 1)
                for (int t = 0; t < stride; ++t) acc[t] += acc[t + stride];
            return acc[0];
        };

        const float full = dot(0, nb);
        float summed = 0.f;
        const int per = nb / tp_size;
        for (int r = 0; r < tp_size; ++r) summed += dot(r * per, per);
        if (n_differ && summed != full) ++*n_differ;
        worst = std::max(worst, std::abs((double)summed - want));
    }
    return worst;
}

int main() {
    printf("sparkinfer kernel algorithm correctness (CPU reference)\n");
    check("attention hd128 kv1",   test_attention(128, 1),    1e-4);
    check("attention hd128 kv333", test_attention(128, 333),  1e-4);
    check("attention hd256 kv1024",test_attention(256, 1024), 2e-4);
    check("attention hd512 kv777", test_attention(512, 777),  2e-4);
    check("router E256 k8",        test_router(256, 8),       1e-6);
    check("router E128 k8",        test_router(128, 8),       1e-6);
    check("swiglu H2048 F512",     test_swiglu(2048, 512),    1e-3);
    check("swiglu H512 F1536",     test_swiglu(512, 1536),    1e-3);
    check("gemm 64x96x128",        test_gemm(64, 96, 128),    1e-3);
    check("gemm 17x33x49",         test_gemm(17, 33, 49),     1e-3);
    check("rmsnorm cols2048",      test_rmsnorm(2048),        1e-4);
    check("rmsnorm vec8 cols2048", test_rmsnorm_vec8(2048),   1e-4);
    check("rmsnorm vec8 cols128",  test_rmsnorm_vec8(128),    1e-4);
    check("add_rmsnorm2 seq c2048",test_add_rmsnorm2_seq(2048),1e-9);
    check("add_rmsnorm2 seq c1536",test_add_rmsnorm2_seq(1536),1e-9);
    check("argmax 2pass qwen vocab",test_argmax_twopass(151936, 512, false), 0.0);
    check("argmax 2pass gemma vocab",test_argmax_twopass(262144, 512, false), 0.0);
    check("argmax 2pass tie-break",  test_argmax_twopass(151936, 512, true),  0.0);
    // Kimi K3 MLA decode. First case is one tile (the shape the GPU coherence
    // test covers); the rest cross tile boundaries, including a ragged tail and
    // a context that the pre-online-softmax kernel could not have launched at.
    check("mla decode ctx96 (1 tile)", test_mla_decode_attn(576, 512, 128, 96, 2),   2e-4);
    check("mla decode ctx128 (exact)", test_mla_decode_attn(576, 512, 128, 128, 2),  2e-4);
    check("mla decode ctx1000",        test_mla_decode_attn(576, 512, 128, 1000, 2), 2e-4);
    check("mla decode ctx16384",       test_mla_decode_attn(576, 512, 128, 16384, 1), 2e-4);
    // Head-batched + context-split, at K3's real MLA dims. 12 heads per block is the
    // shipped batch; the contexts straddle tile and slice boundaries, and 5000/7 gives
    // slices that are not a whole number of tiles. hpb 8 and 4 are the fallbacks
    // k3_mla_heads_per_block takes for head counts 12 does not divide.
    check("mla hbatch ctx8192 hpb12", test_mla_decode_hbatch(576, 512, 128, 8192, 12, 12, 8), 2e-4);
    check("mla hbatch ctx5000 sp7",   test_mla_decode_hbatch(576, 512, 128, 5000, 12, 12, 7), 2e-4);
    check("mla hbatch ctx1000 sp3",   test_mla_decode_hbatch(576, 512, 128, 1000, 12, 12, 3), 2e-4);
    check("mla hbatch ctx8192 hpb8",  test_mla_decode_hbatch(576, 512, 128, 8192, 8,  8,  8), 2e-4);
    check("mla hbatch ctx300 hpb4",   test_mla_decode_hbatch(576, 512, 128, 300,  4,  4,  2), 2e-4);
    check_min("mla split merge needs rescale", test_mla_split_merge_needs_rescale(), 1e-2);

    // Tensor-parallel attention head banding. The positives are exact — a band runs
    // the same arithmetic on the same values — so they carry a zero tolerance, and
    // each is paired with the wrong-pitch version that a plausible misreading of the
    // layout would produce.
    check("kda band conv 96h/tp8",   test_band_kda_conv(96, 8, 128, 3, 4),  0.0);
    check("kda band conv 96h/tp4",   test_band_kda_conv(96, 4, 128, 3, 4),  0.0);
    check("kda band conv 96h/tp1",   test_band_kda_conv(96, 1, 128, 3, 4),  0.0);
    check_differs("kda band conv state pitch 1",
                  test_band_kda_conv(96, 8, 128, 1, 4) != 0.0);
    check_differs("kda band conv weight pitch 3",
                  test_band_kda_conv(96, 8, 128, 3, 3) != 0.0);
    check("mla band absorb 16h/tp4", test_band_mla_absorb(16, 4, 16, 32, 16L * 32), 0.0);
    check("mla band absorb 16h/tp2", test_band_mla_absorb(16, 2, 16, 32, 16L * 32), 0.0);
    check_differs("mla band absorb pitch qk_nope",
                  test_band_mla_absorb(16, 4, 16, 32, 16L) != 0.0);

    // attn_output's column partition: within f32 rounding of the whole-row dot, and
    // genuinely a different summation order rather than an accident of the data.
    int n_differ = 0;
    check("proj cols K7168/tp8", test_proj_cols_partition(7168, 8, 16, &n_differ), 2e-3);
    check_differs("proj cols K7168/tp8 reassociates", n_differ > 0);
    check("proj cols K12288/tp8", test_proj_cols_partition(12288, 8, 16, &n_differ), 3e-3);
    check_differs("proj cols K12288/tp8 reassociates", n_differ > 0);

    g_fail += g_ne_fail;
    printf("%s (%d failures)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail);
    return g_fail ? 1 : 0;
}
