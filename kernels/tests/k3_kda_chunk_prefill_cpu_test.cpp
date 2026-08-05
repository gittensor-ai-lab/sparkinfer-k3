// Chunk-parallel KDA prefill — the algorithm, checked against the verified token-by-token
// recurrence before any CUDA is written.
//
// WHERE THE ALGORITHM COMES FROM. Moonshot's own FlashKDA (github.com/MoonshotAI/FlashKDA,
// "high-performance Kimi Delta Attention kernels") is the chunk-parallel WY/UT-transform form
// of this exact operator: `lower_bound` in its README is documented as "-5.0 to 0", matching
// K3's kda_gate_lower_bound = -5.0f, and its benchmark shape is literally H=96, D=128 — K3's
// own dims. This file does NOT vendor FlashKDA's kernel (CUTLASS/CUTE, sm_90a TMA + GMMA,
// bf16/fp16 tensor-core throughout — a different idiom than anything else in this tree).
// It re-derives the SAME algorithm from K3's own pinned single-step formula
// (kda_decode_step_kernel in k3_kernels.cu, the one k3_kda_step_cpu_test.cpp already proves
// against float64) and cross-checks the result against FlashKDA's published structure.
// Both derivations converge on the same matrices; see the file-header comment in
// k3_kda_chunk_prefill.cu for the full derivation. Two things differ from FlashKDA on
// purpose, not by omission:
//   - no `* scale` folded into q_decayed here. K3's own contract (kimi_k3.h, kda_decode_step_f32)
//     already requires the CALLER to pre-scale q by 1/sqrt(head_dim) before it reaches the
//     recurrence — this kernel is a drop-in replacement for N calls to that step, so it keeps
//     the same input contract rather than inventing a second one.
//   - the inverse is computed by forward substitution, not FlashKDA's doubling-Neumann-series
//     trick. Doubling exists to keep the series in fp16/bf16 range under tensor-core throughput
//     pressure; this file (and the kernel it verifies) is f32 throughout, matching every other
//     K3 kernel's stated convention, so there is no precision pressure motivating it. At
//     CHUNK=16 the triangular solve is O(256) FMA against O(2048) FMA for the two decayed
//     projections — a rounding error either way. sparkinfer's own sibling file for Qwen's GDN
//     (prefill_gdn_chunk.cu, same problem family, same repo) makes the identical choice.
//
// THE DERIVATION, IN BRIEF (full version in the .cu file). Per chunk of C=16 local tokens
// t=0..C-1, contraction index i (state row, what q/k contract over), output index j (state
// col, what v/out are indexed by), state entering the chunk S_in:
//
//   G_t[i]        = sum_{r=0}^{t} g_r[i]                        (inclusive cumsum, per channel)
//   k_decayed_t   = k_t * exp(G_t)          q_decayed_t = q_t * exp(G_t)   (already q-scaled)
//   k_inv_t       = k_t * exp(-G_t)         k_restored_t = k_inv_t * exp(G_total)
//   L[t][s]       = beta_t * (k_decayed_t . k_inv_s)   for s <  t, else 0   (strictly lower)
//   Mqk[t][s]     =          (q_decayed_t . k_inv_s)   for s <= t, else 0   (lower, incl. diag)
//   RHS_t         = beta_t * (v_t - k_decayed_t @ S_in)
//   U             = (I + L)^{-1} @ RHS                            [C, head_dim]
//   O             = Q_decayed @ S_in + Mqk @ U                    [C, head_dim]  <- chunk output
//   S_out[i][j]   = S_in[i][j] * exp(G_total[i]) + (K_restored^T @ U)[i][j]
//
// WHAT THIS FILE CHECKS, since a wrong-but-plausible chunk form produces a fluent, well-formed,
// wrong sequence exactly the way a wrong-but-plausible single step used to (see k3_kernels.cu's
// own history: the axis-of-decay bug survived three separate coverage gaps because token 0's
// zero state makes both axes agree):
//   1. the chunk form matches N sequential token-by-token steps, to float64 tolerance, over a
//      MULTI-chunk, NON-zero-starting-state sequence — a single chunk cannot exercise the
//      cross-chunk state carry, and a zero start state cannot exercise decay at all.
//   2. a RAGGED final chunk (fewer than 16 real tokens) matches too, using the same
//      zero-pad-the-tail scheme FlashKDA's own reference uses (q=k=v=g=beta=0 past the real
//      length) — proven self-consistent below: beta=0 on a padded row zeros its ENTIRE row of
//      L, so U is 0 there regardless of q/k content, so it contributes nothing to the state or
//      to any later row's output.
//   3. the axis-of-decay swap (S_in[j][i] role of i/j exchanged) — the exact bug class this
//      codebase has already shipped once — is measurably different, not just "some places
//      differ": if the negative control were also accidentally close, the tolerance above
//      would not be testing what this claims to test.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

int g_fail = 0;
int g_case = 0;

void check(const char* what, double rel, double tol) {
    ++g_case;
    const bool ok = rel <= tol;
    std::printf("  %-58s rel=%.3e tol=%.1e  %s\n", what, rel, tol, ok ? "OK" : "FAIL");
    if (!ok) ++g_fail;
}

void check_differs(const char* what, double rel, double floor_) {
    ++g_case;
    const bool ok = rel >= floor_;
    std::printf("  %-58s rel=%.3e floor=%.1e  %s\n", what, rel, floor_,
                ok ? "OK" : "FAIL (variant is indistinguishable)");
    if (!ok) ++g_fail;
}

double rel_l2(const std::vector<double>& a, const std::vector<double>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        num += d * d;
        den += b[i] * b[i];
    }
    return std::sqrt(num / (den + 1e-30));
}

constexpr int CHUNK = 16;

// ===========================================================================
// Ground truth: K3's own pinned single-step recurrence, token by token, float64.
// Transcribed VERBATIM from kda_reference() in k3_kda_step_cpu_test.cpp — same
// formula, extended to walk a whole sequence (that file only ever takes one step).
// D = head_dim, H = n_head. Layout: q/k/v/g are [T, H, D]; beta is [T, H]; state
// is [H, D(j), D(i)] i.e. state[h][j][i] = S[i][j], i fastest — the SAME physical
// layout kda_decode_step_kernel reads and writes (state[j*D+i]).
// ===========================================================================
struct SeqIO {
    std::vector<double> out;    // [T, H, D]
    std::vector<double> state;  // [H, D, D]
};

SeqIO kda_token_by_token(const std::vector<double>& S0, const std::vector<double>& q,
                         const std::vector<double>& k, const std::vector<double>& v,
                         const std::vector<double>& g, const std::vector<double>& beta,
                         int T, int D, int H) {
    SeqIO r;
    r.out.assign((size_t)T * H * D, 0.0);
    r.state = S0;
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < H; ++h) {
            const double* qh = &q[((size_t)t * H + h) * D];
            const double* kh = &k[((size_t)t * H + h) * D];
            const double* vh = &v[((size_t)t * H + h) * D];
            const double* gh = &g[((size_t)t * H + h) * D];
            const double  bh = beta[(size_t)t * H + h];
            std::vector<double> ge(D);
            for (int i = 0; i < D; ++i) ge[i] = std::exp(gh[i]);
            double* Sh = &r.state[(size_t)h * D * D];
            for (int j = 0; j < D; ++j) {
                double* S = &Sh[(size_t)j * D];
                double sk = 0.0;
                for (int i = 0; i < D; ++i) sk += S[i] * ge[i] * kh[i];
                const double d_j = bh * (vh[j] - sk);
                double o = 0.0;
                for (int i = 0; i < D; ++i) {
                    const double s2 = S[i] * ge[i] + kh[i] * d_j;
                    S[i] = s2;
                    o += s2 * qh[i];
                }
                r.out[((size_t)t * H + h) * D + j] = o;
            }
        }
    }
    return r;
}

// ===========================================================================
// Candidate: the chunk-parallel form, forward-substitution inverse, float64.
// AXIS_BUG: when true, swaps which index the decay is indexed by (the negative
// control for [3] above) — G is accumulated over j instead of i, which is the
// exact wrong-operator class k3_kernels.cu's own history warns about.
// ===========================================================================
template <bool AXIS_BUG = false>
SeqIO kda_chunk_parallel(const std::vector<double>& S0, const std::vector<double>& q,
                         const std::vector<double>& k, const std::vector<double>& v,
                         const std::vector<double>& g, const std::vector<double>& beta,
                         int T, int D, int H) {
    SeqIO r;
    r.out.assign((size_t)T * H * D, 0.0);
    r.state = S0;

    const int n_chunks = (T + CHUNK - 1) / CHUNK;

    for (int h = 0; h < H; ++h) {
        double* Sh = &r.state[(size_t)h * D * D];  // Sh[j*D+i] = S[i][j]

        for (int c = 0; c < n_chunks; ++c) {
            const int t0 = c * CHUNK;
            const int actual_len = std::min(CHUNK, T - t0);

            // Zero-pad the tail exactly as FlashKDA's own reference does: q/k/v/g/beta all
            // read as zero past actual_len. Proven self-consistent in the header comment.
            auto qt = [&](int t, int i) { return t < actual_len ? q[((size_t)(t0 + t) * H + h) * D + i] : 0.0; };
            auto kt = [&](int t, int i) { return t < actual_len ? k[((size_t)(t0 + t) * H + h) * D + i] : 0.0; };
            auto vt = [&](int t, int j) { return t < actual_len ? v[((size_t)(t0 + t) * H + h) * D + j] : 0.0; };
            auto gt = [&](int t, int i) { return t < actual_len ? g[((size_t)(t0 + t) * H + h) * D + i] : 0.0; };
            auto bt = [&](int t)        { return t < actual_len ? beta[(size_t)(t0 + t) * H + h]        : 0.0; };

            // Cumulative log-gate, inclusive, per channel. AXIS_BUG accumulates over the
            // OUTPUT index (as if reused across j) instead of the state-row/contraction
            // index i -- a well-formed matrix of the wrong shape's meaning.
            std::vector<std::vector<double>> G(CHUNK, std::vector<double>(D, 0.0));
            for (int dcol = 0; dcol < D; ++dcol) {
                double acc = 0.0;
                for (int t = 0; t < CHUNK; ++t) {
                    acc += AXIS_BUG ? gt(t, D - 1 - dcol) : gt(t, dcol);
                    G[t][dcol] = acc;
                }
            }

            std::vector<std::vector<double>> k_decayed(CHUNK, std::vector<double>(D)),
                q_decayed(CHUNK, std::vector<double>(D)), k_inv(CHUNK, std::vector<double>(D)),
                k_restored(CHUNK, std::vector<double>(D));
            std::vector<double> g_total(D);
            for (int dcol = 0; dcol < D; ++dcol) g_total[dcol] = G[CHUNK - 1][dcol];

            for (int t = 0; t < CHUNK; ++t) {
                for (int i = 0; i < D; ++i) {
                    const double eg  = std::exp(G[t][i]);
                    const double egn = std::exp(-G[t][i]);
                    k_decayed[t][i] = kt(t, i) * eg;
                    q_decayed[t][i] = qt(t, i) * eg;   // NO *scale -- caller already applied it
                    k_inv[t][i]     = kt(t, i) * egn;
                    k_restored[t][i] = k_inv[t][i] * std::exp(g_total[i]);
                }
            }

            // L[t][s] = beta_t * (k_decayed_t . k_inv_s), s < t.  Mqk[t][s] = q_decayed_t . k_inv_s, s <= t.
            std::vector<std::vector<double>> L(CHUNK, std::vector<double>(CHUNK, 0.0));
            std::vector<std::vector<double>> Mqk(CHUNK, std::vector<double>(CHUNK, 0.0));
            for (int t = 0; t < CHUNK; ++t) {
                for (int s = 0; s <= t; ++s) {
                    double dot_qk = 0.0, dot_kk = 0.0;
                    for (int i = 0; i < D; ++i) {
                        dot_kk += k_decayed[t][i] * k_inv[s][i];
                        dot_qk += q_decayed[t][i] * k_inv[s][i];
                    }
                    if (s < t) L[t][s] = bt(t) * dot_kk;
                    Mqk[t][s] = dot_qk;
                }
            }

            // PREP/SCAN SPLIT, MATCHING WHAT THE CUDA KERNEL ACTUALLY COMPUTES. Everything
            // above this line is state-independent (a pure function of this chunk's own
            // q/k/v/g/beta) and is where K1 (the parallel-over-all-chunks kernel) stops.
            // INV = (I+L)^{-1} is ALSO state-independent -- built here explicitly via
            // forward substitution against the identity, once per chunk, so K2 (the part
            // that is sequential over chunks because S_in depends on the previous chunk's
            // S_out) only ever does a plain [16,16]@[16,128] matmul, never another solve.
            // This is FlashKDA's K1/K2 split; only HOW the inverse is built differs
            // (forward-substitution here, doubling-Neumann-series there) -- see the file
            // header for why that substitution is the right one for an f32, non-tensor-core
            // kernel. Building INV this way and then multiplying is mathematically the same
            // as solving (I+L)U=RHS directly by forward substitution (both are the unique
            // solution of a nonsingular linear system); the point of proving THIS specific
            // sequence of operations is that it is the one k3_kda_chunk_prefill.cu performs,
            // so a transcription error in the split shows up here, not only in the algebra.
            std::vector<std::vector<double>> INV(CHUNK, std::vector<double>(CHUNK, 0.0));
            for (int col = 0; col < CHUNK; ++col) {
                for (int t = 0; t < CHUNK; ++t) {
                    double acc = (t == col) ? 1.0 : 0.0;   // RHS = identity column `col`
                    for (int s = 0; s < t; ++s) acc -= L[t][s] * INV[s][col];
                    INV[t][col] = acc;
                }
            }

            // RHS_t[j] = beta_t * (v_t[j] - (k_decayed_t @ S_in)[j])   -- STATE-DEPENDENT,
            // this is where K2 starts.
            std::vector<std::vector<double>> RHS(CHUNK, std::vector<double>(D));
            for (int t = 0; t < CHUNK; ++t) {
                for (int j = 0; j < D; ++j) {
                    double proj = 0.0;
                    for (int i = 0; i < D; ++i) proj += k_decayed[t][i] * Sh[(size_t)j * D + i];
                    RHS[t][j] = bt(t) * (vt(t, j) - proj);
                }
            }

            // U = INV @ RHS -- a plain matmul now, no triangular solve left in K2.
            std::vector<std::vector<double>> U(CHUNK, std::vector<double>(D, 0.0));
            for (int t = 0; t < CHUNK; ++t) {
                for (int j = 0; j < D; ++j) {
                    double acc = 0.0;
                    for (int s = 0; s <= t; ++s) acc += INV[t][s] * RHS[s][j];
                    U[t][j] = acc;
                }
            }

            // O = Q_decayed @ S_in + Mqk @ U
            for (int t = 0; t < actual_len; ++t) {
                for (int j = 0; j < D; ++j) {
                    double proj = 0.0;
                    for (int i = 0; i < D; ++i) proj += q_decayed[t][i] * Sh[(size_t)j * D + i];
                    double mu = 0.0;
                    for (int s = 0; s <= t; ++s) mu += Mqk[t][s] * U[s][j];
                    r.out[((size_t)(t0 + t) * H + h) * D + j] = proj + mu;
                }
            }

            // S_out[i][j] = S_in[i][j] * exp(g_total[i]) + sum_t k_restored_t[i] * U[t][j]
            std::vector<double> new_state((size_t)D * D);
            for (int j = 0; j < D; ++j) {
                for (int i = 0; i < D; ++i) {
                    double acc = Sh[(size_t)j * D + i] * std::exp(g_total[i]);
                    for (int t = 0; t < CHUNK; ++t) acc += k_restored[t][i] * U[t][j];
                    new_state[(size_t)j * D + i] = acc;
                }
            }
            std::memcpy(Sh, new_state.data(), sizeof(double) * D * D);
        }
    }
    return r;
}

std::vector<double> rnd(size_t n, std::mt19937& rng, double lo, double hi) {
    std::uniform_real_distribution<double> u(lo, hi);
    std::vector<double> x(n);
    for (auto& e : x) e = u(rng);
    return x;
}

// ===========================================================================
// The FOLDED contract: raw inputs in, activations applied, then the chunk form.
//
// Tests [1]-[4] grade the chunk ALGEBRA given already-activated q/k/g/beta. The
// kernel does not take those — k3_kda_chunk_prep_kernel folds three elementwise
// stages in (gate op 7, L2 norm op 8, beta sigmoid), because folding is what
// keeps prefill to one launch per chunk-batch instead of one per token for the
// elementwise work too. Those three stages are exactly where a scale or an axis
// can slip while every downstream matrix still looks well-formed, so they are
// graded here rather than left to inspection.
//
// Both sides apply the SAME activations from the SAME raw tensors; only what
// happens afterwards differs (chunk form vs token-by-token). So this isolates
// the folding + algebra together, against the pinned per-token recurrence.
// ===========================================================================
struct Activated {
    std::vector<double> q, k, g, beta;
};

// op 7:  g    = lower_bound * sigmoid(-(A[h] * g_raw))
// op 8:  q,k  = L2-normalised over head_dim per (token, head); q also * 1/sqrt(D)
//        beta = sigmoid(beta_logit)
Activated activate(const std::vector<double>& q_raw, const std::vector<double>& k_raw,
                   const std::vector<double>& g_raw, const std::vector<double>& beta_raw,
                   const std::vector<double>& A, int T, int D, int H,
                   double lower_bound, double l2_eps) {
    Activated a;
    a.q.assign(q_raw.size(), 0.0);
    a.k.assign(k_raw.size(), 0.0);
    a.g.assign(g_raw.size(), 0.0);
    a.beta.assign(beta_raw.size(), 0.0);
    const double q_scale = 1.0 / std::sqrt((double)D);
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < H; ++h) {
            const size_t base = ((size_t)t * H + h) * D;
            double qs = 0.0, ks = 0.0;
            for (int d = 0; d < D; ++d) {
                qs += q_raw[base + d] * q_raw[base + d];
                ks += k_raw[base + d] * k_raw[base + d];
            }
            const double qn = q_scale / std::sqrt(qs + l2_eps);
            const double kn = 1.0 / std::sqrt(ks + l2_eps);
            for (int d = 0; d < D; ++d) {
                a.q[base + d] = q_raw[base + d] * qn;
                a.k[base + d] = k_raw[base + d] * kn;
                a.g[base + d] = lower_bound / (1.0 + std::exp(A[h] * g_raw[base + d]));
            }
            const double bl = beta_raw[(size_t)t * H + h];
            a.beta[(size_t)t * H + h] = 1.0 / (1.0 + std::exp(-bl));
        }
    }
    return a;
}

void test_folded_contract() {
    std::printf("\n[5] FOLDED contract: raw in -> gate/L2norm/sigmoid -> chunk form\n");
    const int D = 128, H = 3, T = 2 * CHUNK + 7;   // multi-chunk AND ragged
    const double lower_bound = -5.0, l2_eps = 1e-6;
    std::mt19937 rng(0xFACADE);

    auto S0       = rnd((size_t)D * D * H, rng, -0.4, 0.4);
    auto q_raw    = rnd((size_t)T * H * D, rng, -2.0, 2.0);
    auto k_raw    = rnd((size_t)T * H * D, rng, -2.0, 2.0);
    auto v        = rnd((size_t)T * H * D, rng, -1.0, 1.0);
    auto g_raw    = rnd((size_t)T * H * D, rng, -3.0, 3.0);
    auto beta_raw = rnd((size_t)T * H, rng, -3.0, 3.0);
    // A is ssm_a from the GGUF = -exp(A_log), so it is NEGATIVE for every head.
    auto A        = rnd((size_t)H, rng, -4.0, -0.05);

    const Activated act = activate(q_raw, k_raw, g_raw, beta_raw, A, T, D, H,
                                   lower_bound, l2_eps);

    // Sanity: the gate must land inside (lower_bound, 0) — a positive g would be
    // GROWTH, not decay, and would blow up over a long prompt. If this ever fires,
    // the sign convention on A or on the sigmoid argument is wrong.
    double gmin = 1e300, gmax = -1e300;
    for (double x : act.g) { gmin = std::fmin(gmin, x); gmax = std::fmax(gmax, x); }
    std::printf("  gate range [%.4f, %.4f], must lie within (%.1f, 0)\n",
                gmin, gmax, lower_bound);
    check_differs("gate strictly negative (decay, not growth)", -gmax, 1e-12);
    check("gate above lower_bound", (gmin < lower_bound) ? 1.0 : 0.0, 0.5);

    const auto ref = kda_token_by_token(S0, act.q, act.k, v, act.g, act.beta, T, D, H);
    const auto got = kda_chunk_parallel<false>(S0, act.q, act.k, v, act.g, act.beta, T, D, H);

    check("folded output vs token-by-token", rel_l2(got.out, ref.out), 1e-9);
    check("folded final state vs token-by-token", rel_l2(got.state, ref.state), 1e-9);
}

void test_multi_chunk_exact() {
    std::printf("[1] chunk-parallel vs token-by-token, 3 exact chunks (T=48), non-zero state\n");
    const int D = 128, H = 2, T = 3 * CHUNK;
    std::mt19937 rng(20260805);

    auto S0    = rnd((size_t)D * D * H, rng, -0.4, 0.4);
    auto q     = rnd((size_t)T * H * D, rng, -1.0, 1.0);
    auto k     = rnd((size_t)T * H * D, rng, -1.0, 1.0);
    auto v     = rnd((size_t)T * H * D, rng, -1.0, 1.0);
    // lower_bound = -5, gate is lb*sigmoid(...) so g in (-5, 0) -- the real range.
    auto g     = rnd((size_t)T * H * D, rng, -4.5, -0.02);
    auto beta  = rnd((size_t)T * H, rng, 0.1, 0.9);

    const auto ref = kda_token_by_token(S0, q, k, v, g, beta, T, D, H);
    const auto got = kda_chunk_parallel<false>(S0, q, k, v, g, beta, T, D, H);

    check("output vs token-by-token", rel_l2(got.out, ref.out), 1e-9);
    check("final state vs token-by-token", rel_l2(got.state, ref.state), 1e-9);
}

void test_ragged_tail() {
    std::printf("\n[2] ragged final chunk (T=37 = 2 full + 5 of 16), zero-padded tail\n");
    const int D = 128, H = 2, T = 2 * CHUNK + 5;
    std::mt19937 rng(777);

    auto S0    = rnd((size_t)D * D * H, rng, -0.4, 0.4);
    auto q     = rnd((size_t)T * H * D, rng, -1.0, 1.0);
    auto k     = rnd((size_t)T * H * D, rng, -1.0, 1.0);
    auto v     = rnd((size_t)T * H * D, rng, -1.0, 1.0);
    auto g     = rnd((size_t)T * H * D, rng, -4.5, -0.02);
    auto beta  = rnd((size_t)T * H, rng, 0.1, 0.9);

    const auto ref = kda_token_by_token(S0, q, k, v, g, beta, T, D, H);
    const auto got = kda_chunk_parallel<false>(S0, q, k, v, g, beta, T, D, H);

    check("output vs token-by-token (incl. ragged chunk)", rel_l2(got.out, ref.out), 1e-9);
    check("final state vs token-by-token (ragged tail)", rel_l2(got.state, ref.state), 1e-9);
}

void test_single_token_chunk() {
    std::printf("\n[3] degenerate T=1 (one real token, 15 padding rows)\n");
    const int D = 128, H = 1, T = 1;
    std::mt19937 rng(42);

    auto S0    = rnd((size_t)D * D * H, rng, -0.4, 0.4);
    auto q     = rnd((size_t)T * H * D, rng, -1.0, 1.0);
    auto k     = rnd((size_t)T * H * D, rng, -1.0, 1.0);
    auto v     = rnd((size_t)T * H * D, rng, -1.0, 1.0);
    auto g     = rnd((size_t)T * H * D, rng, -4.5, -0.02);
    auto beta  = rnd((size_t)T * H, rng, 0.1, 0.9);

    const auto ref = kda_token_by_token(S0, q, k, v, g, beta, T, D, H);
    const auto got = kda_chunk_parallel<false>(S0, q, k, v, g, beta, T, D, H);

    check("output vs token-by-token (T=1)", rel_l2(got.out, ref.out), 1e-9);
    check("final state vs token-by-token (T=1)", rel_l2(got.state, ref.state), 1e-9);
}

void test_axis_bug_control() {
    std::printf("\n[4] control: decay indexed by the wrong axis must be measurably wrong\n");
    const int D = 128, H = 1, T = 2 * CHUNK;
    std::mt19937 rng(31337);

    auto S0    = rnd((size_t)D * D * H, rng, -0.4, 0.4);
    auto q     = rnd((size_t)T * H * D, rng, -1.0, 1.0);
    auto k     = rnd((size_t)T * H * D, rng, -1.0, 1.0);
    auto v     = rnd((size_t)T * H * D, rng, -1.0, 1.0);
    auto g     = rnd((size_t)T * H * D, rng, -4.5, -0.02);
    auto beta  = rnd((size_t)T * H, rng, 0.1, 0.9);

    const auto ref = kda_token_by_token(S0, q, k, v, g, beta, T, D, H);
    const auto bug = kda_chunk_parallel<true>(S0, q, k, v, g, beta, T, D, H);

    check_differs("axis-swapped decay vs token-by-token (must diverge)",
                  rel_l2(bug.out, ref.out), 1e-3);
}

}  // namespace

int main() {
    std::printf("Chunk-parallel KDA prefill: candidate vs the pinned token-by-token recurrence\n\n");
    test_multi_chunk_exact();
    test_ragged_tail();
    test_single_token_chunk();
    test_axis_bug_control();
    test_folded_contract();
    std::printf("\n%d cases, %d failures\n", g_case, g_fail);
    return g_fail == 0 ? 0 : 1;
}
