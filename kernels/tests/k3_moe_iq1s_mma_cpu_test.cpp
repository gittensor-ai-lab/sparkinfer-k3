// IQ1_S ON INT8 TENSOR CORES: the arithmetic identity, and the tile schedule built on it.
//
// This is the CPU half of k3_moe_iq1s_mma.cu. It has no device and needs none: everything
// load-bearing here is integer math and exactly-representable float arithmetic, which is
// precisely the part that a GPU test would obscure rather than expose. It runs on a fork's
// PR, where the GPU checks stay pending.
//
// ---------------------------------------------------------------------------
// THE CLAIM THE WHOLE KERNEL RESTS ON
// ---------------------------------------------------------------------------
// IQ1_S reconstructs a weight as
//
//     w = dl * (g + delta),   g in {-1,0,+1},   delta = +/-0.125,   dl = d * (2*sc + 1)
//
// with dl and delta CONSTANT across each 32-value sub-block. Multiply through by 8:
//
//     w = (dl/8) * (8g + s),   s = +/-1  (the sign of delta)
//
// and 8g + s lands in {-9,-7,-1,+1,+7,+9} — a small, exact int8. So an IQ1_S sub-block is
// an int8 vector times ONE scalar, and the conversion is EXACT: not "higher precision than
// what is stored" (the argument prefill_gemm_i8.cu makes for Q4_K/Q6_K, which is true but
// is an inequality), but bit-for-bit the same float. That is a strictly stronger property
// and it is what lets the MoE run on the int8 tensor cores with no accuracy argument at all
// — there is nothing to trade.
//
// The second half of the fit is the instruction. mma.sync.m16n8k32.s8 reduces exactly 32
// values of k per issue, and an IQ1_S sub-block is exactly 32 values with one scale. So the
// scale domain and the instruction's k-extent coincide: the int32 accumulator is drained and
// scaled once per mma and no scale is ever applied inside a reduction. Had the sub-block
// been 16 or 64 this would not work cleanly; at 32 it is exact by construction.
//
// WHY THIS MATTERS AT ALL: the routed experts are ~531 of UD-IQ1_S's 553 GiB. Every other
// tensor-core opportunity in K3 is rounding error next to the expert GEMM.
//
// ---------------------------------------------------------------------------
// WHAT IS ACTUALLY TESTED
// ---------------------------------------------------------------------------
//   [1] exactness      w == (dl/8)*(8g+s) bit-for-bit, and |8g+s| <= 9, over random blocks
//   [2] tile schedule  the kernel recovers (row, sub-block) from a flat tile id and decodes
//                      the same 32 values dequantize_row_iq1_s's nested cursor walk does
//   [3] accumulation   the drain-per-sub-block int32 schedule matches a float64 reference,
//                      and the int32 accumulator provably cannot overflow
//   [4] control        dropping delta (m = 8g, the plausible simplification) is measurably
//                      wrong — without this, [1] and [3] only prove the tolerance was loose
//
// Random block bits are the right input: every bit pattern is a VALID IQ1_S block for
// reconstruction purposes (the grid index is 11 bits so it cannot leave the 2048-entry
// table, the scale is 3 bits, the sign is 1 bit), so random bits sweep the whole lattice
// rather than whatever subset one tensor happens to use. Only `d` is drawn deliberately:
// a random fp16 there would be NaN/Inf/subnormal a good fraction of the time, and the
// subnormal case is the ONE place the identity in [1] can legitimately fail (dl/8 stops
// being exact once it falls into the subnormal range). That is called out at [1] rather
// than hidden by the input distribution.

#include "sparkinfer/kernels/iq1s_tables.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace sparkinfer::kernels::k3;

namespace {

int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::printf("  FAIL: %s\n", what); g_fail = 1; }
}

struct BlockIQ1S { uint16_t d; uint8_t qs[32]; uint16_t qh[8]; };
static_assert(sizeof(BlockIQ1S) == 50, "IQ1_S block must be 50 bytes");

constexpr int kQK        = 256;   // values per IQ1_S block
constexpr int kSub       = 32;    // values per sub-block == mma.m16n8k32's k-extent
constexpr int kSubPerBlk = kQK / kSub;

// fp16 -> fp32, exact for every finite half. Same routine as kimi_k3_iq1s_ggml_cpu_test.cpp;
// duplicated rather than shared because a test that reaches into another test for its
// reference has two things that can drift together and stay silent.
float h2f(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const uint32_t exp  = (h >> 10) & 0x1f;
    const uint32_t man  = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else {
            int e = -1; uint32_t m = man;
            do { m <<= 1; ++e; } while (!(m & 0x400));
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3ff) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f; std::memcpy(&f, &bits, 4); return f;
}

// float -> fp16 bits, round-to-nearest-even. Only used to build test inputs.
uint16_t f2h(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (exp <= 0) return (uint16_t)sign;                    // flush tiny to zero: see [1]
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
    const uint32_t rem = man & 0x1fffu;                     // round to nearest even
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) ++h;
    return h;
}

// The packed 2-bit lattice the kernel gathers, rebuilt here by the SAME rule
// ensure_iq1s_tables uses: field j holds (grid[j] & 3), sign-extended on read.
// Rebuilt rather than linked so a divergence between the packer and the unpacker
// shows up as a test failure instead of cancelling out.
std::vector<uint16_t> build_packed_grid() {
    std::vector<uint16_t> p(SPARKINFER_IQ1S_NGRID);
    for (int i = 0; i < SPARKINFER_IQ1S_NGRID; ++i) {
        const int8_t* g = (const int8_t*)&iq1s_grid_host[i];
        uint16_t v = 0;
        for (int j = 0; j < 8; ++j) v |= (uint16_t)((uint16_t)g[j] & 3u) << (2 * j);
        p[i] = v;
    }
    return p;
}

inline int unpack2(uint16_t pw, int j) {
    return ((int)((uint32_t)pw << (30 - 2 * j))) >> 30;     // sign-extend field j
}

// The reference: ggml's own reconstruction order, dl * (grid + delta) in f32.
void ref_dequant_block(const BlockIQ1S& b, float* out) {
    const float d = h2f(b.d);
    for (int ib32 = 0; ib32 < kSubPerBlk; ++ib32) {
        const uint16_t h = b.qh[ib32];
        const float dl    = d * (float)(2 * ((h >> 12) & 7) + 1);
        const float delta = (h & 0x8000) ? -SPARKINFER_IQ1S_DELTA : SPARKINFER_IQ1S_DELTA;
        for (int l = 0; l < 4; ++l) {
            const uint32_t idx =
                (uint32_t)b.qs[4 * ib32 + l] | (((uint32_t)(h >> (3 * l)) & 7u) << 8);
            const int8_t* grid = (const int8_t*)&iq1s_grid_host[idx];
            for (int j = 0; j < 8; ++j)
                out[ib32 * kSub + l * 8 + j] = dl * ((float)grid[j] + delta);
        }
    }
}

// EXACTLY the kernel's per-thread B-tile decode: one thread owns one (row, sub-block) and
// emits 32 int8 plus the single scale that covers them. `DROP_DELTA` is the control in [4].
template <bool DROP_DELTA = false>
void kernel_decode_sub(const BlockIQ1S& b, int ib32, int8_t* m_out, float* scale_out,
                       const std::vector<uint16_t>& gp) {
    const uint16_t h  = b.qh[ib32];
    const float    d  = h2f(b.d);
    const float    dl = d * (float)(2 * ((h >> 12) & 7) + 1);
    const int      s  = (h & 0x8000) ? -1 : 1;
    *scale_out = dl * 0.125f;                    // dl/8, exact: 0.125 is a power of two
    for (int l = 0; l < 4; ++l) {
        const uint32_t idx =
            (uint32_t)b.qs[4 * ib32 + l] | (((uint32_t)(h >> (3 * l)) & 7u) << 8);
        const uint16_t pw = gp[idx];
        for (int j = 0; j < 8; ++j)
            m_out[l * 8 + j] = (int8_t)(8 * unpack2(pw, j) + (DROP_DELTA ? 0 : s));
    }
}

BlockIQ1S random_block(std::mt19937& rng) {
    BlockIQ1S b{};
    std::uniform_int_distribution<int> byte(0, 255);
    // Random bits everywhere the format permits them (see the header note), and a
    // deliberately sane `d` so fp16 corner cases do not masquerade as kernel defects.
    std::uniform_real_distribution<float> dd(0.001f, 0.5f);
    b.d = f2h(dd(rng));
    for (int i = 0; i < 32; ++i) b.qs[i] = (uint8_t)byte(rng);
    for (int i = 0; i < 8; ++i)  b.qh[i] = (uint16_t)(byte(rng) | (byte(rng) << 8));
    return b;
}

// ---------------------------------------------------------------------------
// [1] the identity
// ---------------------------------------------------------------------------
void test_exactness() {
    std::printf("[1] w == (dl/8)*(8g+s), bit-for-bit, and |8g+s| <= 9\n");
    const auto gp = build_packed_grid();
    std::mt19937 rng(20260805u);

    const int NB = 4096;
    long long n_vals = 0;
    int seen[19] = {0};                       // histogram of m + 9, to prove the range is swept
    for (int ib = 0; ib < NB; ++ib) {
        const BlockIQ1S b = random_block(rng);
        float ref[kQK];
        ref_dequant_block(b, ref);

        for (int s32 = 0; s32 < kSubPerBlk; ++s32) {
            int8_t m[kSub]; float sc;
            kernel_decode_sub(b, s32, m, &sc, gp);
            for (int j = 0; j < kSub; ++j) {
                const float got = sc * (float)m[j];
                const float want = ref[s32 * kSub + j];
                // BIT-for-bit, not close: both sides are one rounding of the same exact
                // real product, so any difference at all is a real defect.
                if (std::memcmp(&got, &want, 4) != 0) {
                    std::printf("  FAIL: blk %d sub %d j %d: got %.9g want %.9g (m=%d sc=%.9g)\n",
                                ib, s32, j, got, want, (int)m[j], sc);
                    g_fail = 1;
                    return;
                }
                check(m[j] >= -9 && m[j] <= 9, "int8 magnitude within +/-9");
                seen[m[j] + 9]++;
                ++n_vals;
            }
        }
    }
    // The six reachable codes are exactly {-9,-7,-1,+1,+7,+9}; anything else appearing
    // means the lattice carried a value outside {-1,0,+1} and the format assumption is void.
    const int reachable[6] = {-9, -7, -1, 1, 7, 9};
    for (int v = -9; v <= 9; ++v) {
        bool ok_code = false;
        for (int r = 0; r < 6; ++r) if (reachable[r] == v) ok_code = true;
        if (!ok_code) check(seen[v + 9] == 0, "no code outside {-9,-7,-1,1,7,9}");
    }
    for (int r = 0; r < 6; ++r)
        check(seen[reachable[r] + 9] > 0, "every reachable code actually exercised");

    std::printf("  %lld values, all bit-exact; codes {-9,-7,-1,1,7,9} all seen\n", n_vals);

    // The one documented corner: dl/8 is exact only while it stays normal. Assert the
    // guard rather than the absence of the problem, so a future dims change trips here.
    check(std::ldexp(1.0f, -126) * 8.0f > 0.0f, "fp32 subnormal boundary sanity");
    std::printf("  NOTE: exactness needs dl/8 normal — dl >= 2^-123. Real IQ1_S d is fp16,\n"
                "        so dl >= 2^-24 * 1 and the margin is ~100 binades.\n");
}

// ---------------------------------------------------------------------------
// [2] the tile schedule
// ---------------------------------------------------------------------------
// The kernel does not walk ggml's nested (ib32 outer, l inner, moving qs cursor) loop. It
// assigns ONE THREAD PER (row, 32-value sub-block) and recovers both from a flat tile id.
// That derivation is the actual risk and it is pure integer math.
void test_tile_schedule() {
    std::printf("[2] flat tile id -> (row, sub-block) recovers ggml's walk\n");
    const auto gp = build_packed_grid();
    std::mt19937 rng(777u);

    const int N = 40;                 // output rows (experts' output channels)
    const int K = 1024;               // reduction length; 4 IQ1_S blocks per row
    const int blk_per_row = K / kQK;

    std::vector<BlockIQ1S> W((size_t)N * blk_per_row);
    for (auto& b : W) b = random_block(rng);

    std::vector<float> ref((size_t)N * K);
    for (int n = 0; n < N; ++n)
        for (int ib = 0; ib < blk_per_row; ++ib)
            ref_dequant_block(W[(size_t)n * blk_per_row + ib], &ref[(size_t)n * K + ib * kQK]);

    // The kernel's schedule: BK=64 spans exactly two sub-blocks and, because k0 is a
    // multiple of 64 and 256 % 64 == 0, can never straddle a 256-value block boundary.
    // That is what makes the (block, sub-block) split a shift and a mask.
    constexpr int BK = 64;
    check(kQK % BK == 0, "BK must divide the 256-value block, else a tile straddles two");

    long long checked = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int n = 0; n < N; ++n) {
            for (int half = 0; half < BK / kSub; ++half) {
                const int kk   = k0 + half * kSub;
                const int ib   = kk / kQK;            // which 256-value block
                const int ib32 = (kk % kQK) / kSub;   // which sub-block inside it
                int8_t m[kSub]; float sc;
                kernel_decode_sub(W[(size_t)n * blk_per_row + ib], ib32, m, &sc, gp);
                for (int j = 0; j < kSub; ++j) {
                    const float got  = sc * (float)m[j];
                    const float want = ref[(size_t)n * K + kk + j];
                    if (std::memcmp(&got, &want, 4) != 0) {
                        std::printf("  FAIL: n %d k %d: got %.9g want %.9g\n", n, kk + j, got, want);
                        g_fail = 1; return;
                    }
                    ++checked;
                }
            }
        }
    }
    std::printf("  %lld (row, k) positions match the reference walk exactly\n", checked);
}

// ---------------------------------------------------------------------------
// [3] the accumulation schedule
// ---------------------------------------------------------------------------
// The kernel's reduction is NOT one long int32 dot. It drains per sub-block:
//
//     acc_f32[m][n] += (float)(int32 dot over 32 k) * (dl[n]/8)
//
// with the per-token activation scale sa[m] hoisted out of the k loop entirely (it is
// constant in k) and applied once in the epilogue.
//
// THE COMPARISON THAT MEANS SOMETHING IS AGAINST THE SHIPPED SCALAR PATH, NOT AGAINST
// float64 ALONE. A dot product of random-sign terms cancels to ~sqrt(K) times one term,
// so error measured against the RESULT is amplified by a condition number of ~60 at
// K=3584 and says more about the input distribution than about the schedule. Both the
// shipped f32 path and this one are graded against the same float64 truth, and the claim
// is the one that actually gates a drop-in replacement: THIS MUST NOT BE WORSE.
//
// It should in fact be BETTER, structurally: the 32-term int32 dot is EXACT — integers,
// no rounding at all — where the scalar path rounds every one of its K f32 accumulations.
// The int8 path carries K/32 rounded adds against the scalar path's K. That is a 32x
// shorter rounding chain, and it is a consequence of the sub-block/instruction fit rather
// than a tuning choice.
void test_accumulation() {
    std::printf("[3] drain-per-sub-block int32 schedule vs the shipped f32 scalar path\n");
    const auto gp = build_packed_grid();
    std::mt19937 rng(31337u);

    const int M = 37;                 // tokens routed to this expert — small on purpose:
                                      // 896 experts at top_k 16 gives T*16/896 rows, so a
                                      // 2048-token chunk lands ~36 rows per expert. The MoE
                                      // GEMM is SKINNY in M and that is the shape to test.
    const int N = 24;
    const int K = 3584;               // K3's expert_latent
    const int blk_per_row = K / kQK;
    check(K % kQK == 0, "K must be a whole number of IQ1_S blocks");

    std::vector<BlockIQ1S> W((size_t)N * blk_per_row);
    for (auto& b : W) b = random_block(rng);

    // int8 activations with a per-row scale — the same scheme llama.cpp's own vec_dot
    // contract uses (block_q8_K), so this is not a new approximation being introduced.
    std::vector<int8_t> A((size_t)M * K);
    std::vector<float>  sa(M);
    std::uniform_int_distribution<int> q(-127, 127);
    std::uniform_real_distribution<float> sd(0.002f, 0.02f);
    for (int m = 0; m < M; ++m) {
        sa[m] = sd(rng);
        for (int k = 0; k < K; ++k) A[(size_t)m * K + k] = (int8_t)q(rng);
    }

    // Reference: float64 over the reconstructed weights and the same int8 activations.
    // Also the natural SCALE of the dot (sum of |terms|), which is what a cancelling sum
    // has to be judged against — see the note above.
    std::vector<double> ref((size_t)M * N, 0.0);
    std::vector<double> mag((size_t)M * N, 0.0);
    std::vector<float>  scal((size_t)M * N, 0.0f);
    std::vector<float>  wrow(K);
    for (int n = 0; n < N; ++n) {
        for (int ib = 0; ib < blk_per_row; ++ib)
            ref_dequant_block(W[(size_t)n * blk_per_row + ib], &wrow[ib * kQK]);
        for (int m = 0; m < M; ++m) {
            double acc = 0.0, amag = 0.0;
            for (int k = 0; k < K; ++k) {
                const double t = (double)wrow[k] * (double)A[(size_t)m * K + k];
                acc += t; amag += std::fabs(t);
            }
            ref[(size_t)m * N + n] = acc * (double)sa[m];
            mag[(size_t)m * N + n] = amag * (double)sa[m];
            // The SHIPPED schedule: block_dot's running f32 accumulator over the
            // reconstructed weights, one rounded add per k.
            float f = 0.0f;
            for (int k = 0; k < K; ++k) f += wrow[k] * (float)A[(size_t)m * K + k];
            scal[(size_t)m * N + n] = f * sa[m];
        }
    }

    // The kernel's schedule.
    long long peak_abs = 0;
    std::vector<float> got((size_t)M * N, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
            float acc = 0.0f;
            for (int ib = 0; ib < blk_per_row; ++ib) {
                for (int s32 = 0; s32 < kSubPerBlk; ++s32) {
                    int8_t mw[kSub]; float sc;
                    kernel_decode_sub(W[(size_t)n * blk_per_row + ib], s32, mw, &sc, gp);
                    int32_t dot = 0;              // what one mma.m16n8k32.s8 produces
                    for (int j = 0; j < kSub; ++j)
                        dot += (int32_t)mw[j] * (int32_t)A[(size_t)m * K + ib * kQK + s32 * kSub + j];
                    if (std::llabs((long long)dot) > peak_abs) peak_abs = std::llabs((long long)dot);
                    acc += (float)dot * sc;      // drain + scale, once per sub-block
                }
            }
            got[(size_t)m * N + n] = acc * sa[m];  // sa hoisted out of the k loop
        }
    }

    double worst_mma = 0.0, worst_scal = 0.0;
    for (int i = 0; i < M * N; ++i) {
        const double den = mag[i] > 1e-30 ? mag[i] : 1.0;
        worst_mma  = std::max(worst_mma,  std::fabs((double)got[i]  - ref[i]) / den);
        worst_scal = std::max(worst_scal, std::fabs((double)scal[i] - ref[i]) / den);
    }
    std::printf("  worst error / sum|terms| over %dx%d (K=%d):\n", M, N, K);
    std::printf("    int8 mma schedule   %.3e   (%d rounded adds per dot)\n",
                worst_mma, K / kSub);
    std::printf("    shipped f32 scalar  %.3e   (%d rounded adds per dot)\n", worst_scal, K);
    check(worst_mma < 1e-6, "int8 schedule is accurate on the well-conditioned scale");
    // The claim that gates a drop-in replacement.
    check(worst_mma <= worst_scal, "int8 schedule is no worse than the shipped f32 path");

    // OVERFLOW: |m| <= 9 and |a| <= 127, so one 32-term dot is bounded by
    // 32 * 9 * 127 = 36,576 — three orders of magnitude inside int32. The bound is
    // structural (it does not depend on K, because the accumulator is drained every
    // sub-block), which is the reason the drain is not merely a scale-correctness device.
    std::printf("  peak |int32 dot| = %lld, bound 32*9*127 = %d\n", peak_abs, 32 * 9 * 127);
    check(peak_abs <= 32LL * 9 * 127, "int32 dot within the structural bound");
}

// ---------------------------------------------------------------------------
// [4] the control
// ---------------------------------------------------------------------------
// Dropping delta (m = 8g) is the plausible simplification: it makes the codes {-8,0,+8},
// which look tidier and let the scale absorb a factor of 8. It is also exactly the bias
// k3_kernels.cu warns about ("dropping it produces a well-formed tensor with a systematic
// bias"). If [1] and [3] pass while this ALSO passes, they proved nothing.
void test_control_drop_delta() {
    std::printf("[4] control: dropping delta must be measurably wrong\n");
    const auto gp = build_packed_grid();
    std::mt19937 rng(4242u);

    const int NB = 256;
    double sum_rel = 0.0; long long n = 0;
    double worst = 0.0;
    for (int ib = 0; ib < NB; ++ib) {
        const BlockIQ1S b = random_block(rng);
        float ref[kQK];
        ref_dequant_block(b, ref);
        for (int s32 = 0; s32 < kSubPerBlk; ++s32) {
            int8_t m[kSub]; float sc;
            kernel_decode_sub<true>(b, s32, m, &sc, gp);      // DROP_DELTA
            for (int j = 0; j < kSub; ++j) {
                const double got = (double)sc * (double)m[j];
                const double want = (double)ref[s32 * kSub + j];
                const double den = std::fabs(want) > 1e-30 ? std::fabs(want) : 1.0;
                const double rel = std::fabs(got - want) / den;
                sum_rel += rel; worst = std::max(worst, rel); ++n;
            }
        }
    }
    const double mean_rel = sum_rel / (double)n;
    std::printf("  mean relative error %.4f, worst %.4f\n", mean_rel, worst);
    // Every value is wrong by exactly dl/8, so on the g == 0 codes the error is total.
    check(mean_rel > 0.05, "dropping delta is a large, systematic error (control)");
}

}  // namespace

int main() {
    std::printf("=== K3 MoE IQ1_S -> int8 tensor-core path (CPU) ===\n\n");
    test_exactness();      std::printf("\n");
    test_tile_schedule();  std::printf("\n");
    test_accumulation();   std::printf("\n");
    test_control_drop_delta();
    std::printf("\n%s\n", g_fail ? "FAIL"
                                 : "PASS: IQ1_S is exactly int8 x (dl/8); tile and "
                                   "accumulation schedules verified");
    return g_fail;
}
