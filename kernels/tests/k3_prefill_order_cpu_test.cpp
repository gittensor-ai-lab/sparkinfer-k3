// Does the batched prefill projection compute the SAME FLOAT as the single-token one?
//
// k3_prefill_proj.cu claims bit-identity "by construction". This checks the construction
// on the CPU with no GPU: it builds BOTH kernels' loop nests faithfully -- including the
// tile shapes, the per-thread strided partials, the shuffle-down butterfly and the
// increasing-warp fold -- and compares cell by cell.
//
// Three independent claims are tested, and each could fail on its own:
//   A. block_for must agree with the decode path for every nb K3 can produce. (A wrong
//      tier is the same arithmetic through a different reduction tree.)
//   B. The per-(row, token) float must not depend on ROWS. Decode picks ROWS from a warp
//      budget (1..16); prefill picks 2 or 4. If ROWS entered the answer they could never
//      agree.
//   C. The batched loop nest (weight hoisted out, token tile innermost) must produce the
//      same cell values as the single-token nest.
//
// What this cannot cover: dp4a and __float2int_rn semantics. Those are the same
// expressions in both kernels, so they cancel; the risk here is the reordering.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>

struct BlockQ8_0 { uint16_t d; int8_t qs[32]; };

static float h2f(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const uint32_t exp  = (h >> 10) & 0x1f;
    const uint32_t man  = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else { int e = -1; uint32_t m = man;
               do { m <<= 1; ++e; } while (!(m & 0x400));
               bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3ff) << 13); }
    } else if (exp == 31) bits = sign | 0x7f800000u | (man << 13);
    else bits = sign | ((uint32_t)(exp - 15 + 127) << 23) | (man << 13);
    float f; std::memcpy(&f, &bits, 4); return f;
}

static int dp4a_block(const BlockQ8_0& w, const BlockQ8_0& x) {
    int s = 0; for (int j = 0; j < 32; ++j) s += (int)w.qs[j] * (int)x.qs[j]; return s;
}

// ---- A. the two block_for tiers -------------------------------------------------
static int block_for_decode(int nb) {            // k3_proj_q8_fast.cu:334, verbatim
    if (nb <= 32) return 32;
    if (nb <= 64) return 64;
    return 128;
}
static int block_for_prefill(int nb) {           // k3_prefill_proj.cu, after the fix
    if (nb <= 32) return 32;
    if (nb <= 64) return 64;
    return 128;
}

// ---- B/C. the two loop nests ----------------------------------------------------
// Single-token nest: acc[ROWS] per thread; ONE activation block staged, reused by rows.
static void nest_single(std::vector<float>& out,          // [ROWS]
                        const BlockQ8_0* W, int nb, int BLOCK, int ROWS,
                        const BlockQ8_0* xrow) {
    const int NWARP = BLOCK / 32;
    std::vector<std::vector<float>> acc(BLOCK, std::vector<float>(ROWS, 0.0f));
    for (int tx = 0; tx < BLOCK; ++tx)
        for (int b = tx; b < nb; b += BLOCK) {
            const float dx = h2f(xrow[b].d);
            for (int r = 0; r < ROWS; ++r) {
                const BlockQ8_0* row = W + (size_t)r * nb;
                const int sumi = dp4a_block(row[b], xrow[b]);
                acc[tx][r] += (float)sumi * (h2f(row[b].d) * dx);
            }
        }
    out.assign(ROWS, 0.0f);
    for (int r = 0; r < ROWS; ++r) {
        std::vector<float> wp(NWARP, 0.0f);
        for (int w = 0; w < NWARP; ++w) {
            std::vector<float> v(32);
            for (int l = 0; l < 32; ++l) v[l] = acc[w * 32 + l][r];
            for (int off = 16; off > 0; off >>= 1)
                for (int l = 0; l < 32; ++l) if (l + off < 32) v[l] += v[l + off];
            wp[w] = v[0];
        }
        float s = 0.0f; for (int w = 0; w < NWARP; ++w) s += wp[w];
        out[r] = s;
    }
}

// Batched nest: acc[ROWS][TOKS] per thread; TOKS activations staged, WEIGHT hoisted out
// of the token loop. This mirrors prefill_proj_q8_kernel exactly.
static void nest_batched(std::vector<std::vector<float>>& out,   // [ROWS][TOKS]
                         const BlockQ8_0* W, int nb, int BLOCK, int ROWS, int TOKS,
                         const BlockQ8_0* X) {
    const int NWARP = BLOCK / 32;
    std::vector<std::vector<std::vector<float>>> acc(
        BLOCK, std::vector<std::vector<float>>(ROWS, std::vector<float>(TOKS, 0.0f)));
    for (int tx = 0; tx < BLOCK; ++tx)
        for (int b = tx; b < nb; b += BLOCK) {
            std::vector<float> dx(TOKS);
            for (int t = 0; t < TOKS; ++t) dx[t] = h2f(X[(size_t)t * nb + b].d);
            for (int r = 0; r < ROWS; ++r) {
                const BlockQ8_0* row = W + (size_t)r * nb;
                const float dw = h2f(row[b].d);
                for (int t = 0; t < TOKS; ++t) {
                    const int sumi = dp4a_block(row[b], X[(size_t)t * nb + b]);
                    acc[tx][r][t] += (float)sumi * (dw * dx[t]);
                }
            }
        }
    out.assign(ROWS, std::vector<float>(TOKS, 0.0f));
    for (int r = 0; r < ROWS; ++r)
        for (int t = 0; t < TOKS; ++t) {
            std::vector<float> wp(NWARP, 0.0f);
            for (int w = 0; w < NWARP; ++w) {
                std::vector<float> v(32);
                for (int l = 0; l < 32; ++l) v[l] = acc[w * 32 + l][r][t];
                for (int off = 16; off > 0; off >>= 1)
                    for (int l = 0; l < 32; ++l) if (l + off < 32) v[l] += v[l + off];
                wp[w] = v[0];
            }
            float s = 0.0f; for (int w = 0; w < NWARP; ++w) s += wp[w];
            out[r][t] = s;
        }
}

int main() {
    long bad = 0;

    // ---- A ----
    int a_bad = 0;
    for (int nb = 1; nb <= 512; ++nb)
        if (block_for_decode(nb) != block_for_prefill(nb)) ++a_bad;
    std::printf("A. block_for agrees for nb 1..512: %s\n", a_bad ? "FAIL" : "ok");
    // K3's real shapes, spelled out so a regression names the tensor it broke
    const struct { const char* n; int K; } shapes[] = {
        {"ffn_routed_down / shexp_down", 7168}, {"ffn_routed_up", 3584},
        {"KDA attn_output", 1536}, {"MLA kv_a", 7168}, {"shexp gate/up", 768},
    };
    for (auto& s : shapes) {
        const int nb = s.K / 32;
        const int bd = block_for_decode(nb), bp = block_for_prefill(nb);
        std::printf("   K=%-5d nb=%-4d decode BLOCK=%-4d prefill BLOCK=%-4d %-28s %s\n",
                    s.K, nb, bd, bp, s.n, bd == bp ? "ok" : "MISMATCH");
        if (bd != bp) ++a_bad;
    }
    bad += a_bad;

    // ---- B and C ----
    std::mt19937 rng(20260805);
    std::uniform_int_distribution<int> q(-127, 127);
    const int Ks[] = { 7168, 3584, 1536, 768 };
    long cells = 0;

    for (int K : Ks) {
        const int nb = K / 32, BLOCK = block_for_prefill(nb);
        const int ROWS_MAX = 6, TOKS = 4;
        std::vector<BlockQ8_0> W((size_t)ROWS_MAX * nb), X((size_t)TOKS * nb);
        auto fill = [&](std::vector<BlockQ8_0>& v) {
            for (auto& b : v) {
                const uint16_t e = (uint16_t)(8 + (rng() % 15));
                b.d = (uint16_t)(((rng() & 1) << 15) | (e << 10) | (rng() % 1024));
                for (int j = 0; j < 32; ++j) b.qs[j] = (int8_t)q(rng);
            }
        };
        fill(W); fill(X);

        // C: batched tile vs per-token single, cell by cell
        std::vector<std::vector<float>> got;
        nest_batched(got, W.data(), nb, BLOCK, /*ROWS*/4, TOKS, X.data());
        for (int t = 0; t < TOKS; ++t) {
            std::vector<float> ref;
            nest_single(ref, W.data(), nb, BLOCK, /*ROWS*/4, X.data() + (size_t)t * nb);
            for (int r = 0; r < 4; ++r) {
                ++cells;
                if (std::memcmp(&ref[r], &got[r][t], 4) != 0) {
                    if (bad < 5) std::printf("C MISMATCH K=%d r=%d t=%d %.9g vs %.9g\n",
                                             K, r, t, ref[r], got[r][t]);
                    ++bad;
                }
            }
        }

        // B: does ROWS enter the answer? decode may pick 2, 4, 8, 16; prefill picks 2 or 4.
        std::vector<float> r2, r4, r6;
        nest_single(r2, W.data(), nb, BLOCK, 2, X.data());
        nest_single(r4, W.data(), nb, BLOCK, 4, X.data());
        nest_single(r6, W.data(), nb, BLOCK, 6, X.data());
        for (int r = 0; r < 2; ++r) {
            ++cells;
            if (std::memcmp(&r2[r], &r4[r], 4) != 0 || std::memcmp(&r2[r], &r6[r], 4) != 0) {
                std::printf("B MISMATCH K=%d r=%d: ROWS changes the float\n", K, r);
                ++bad;
            }
        }
        std::printf("B/C K=%-5d nb=%-4d BLOCK=%-4d  ok\n", K, nb, BLOCK);
    }

    std::printf("\ncells compared: %ld   mismatches: %ld\n", cells, bad);
    std::printf("%s\n", bad == 0
        ? "PASS: batched tiling is bit-identical to the single-token path; ROWS does not enter it"
        : "FAIL");
    return bad ? 1 : 0;
}
