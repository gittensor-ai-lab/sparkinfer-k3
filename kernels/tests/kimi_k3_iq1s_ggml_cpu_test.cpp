// IQ1_S dequant: my kernel's arithmetic vs REAL ggml, on the CPU.
//
// The lattice table is already proven identical to ggml's by mechanical extraction.
// What that does NOT prove is the INDEX ARITHMETIC. ggml walks nested loops
// (ib32 = 0..7 outer, l = 0..3 inner, with `qs += 4` advancing a cursor); the kernel
// assigns ONE THREAD PER 8-VALUE GROUP and must recover (ib32, l) from a flat global
// id. That derivation — and the `qs[4*ib32 + l]` indexing that replaces ggml's moving
// cursor — is the actual risk, and it is pure integer math that needs no GPU to check.
//
// Random block bits are the right input: every bit pattern is a VALID IQ1_S block for
// dequant purposes (the grid index is 11 bits so it can never exceed the 2048-entry
// table, the scale is 3 bits, the sign is 1 bit), so random bits sweep the whole
// lattice instead of whatever subset a real tensor happens to use.
//
// Requires bit-exactness, not closeness: both sides evaluate dl * (grid + delta) in
// the same order on the same floats, so any difference is a real defect.

#include "sparkinfer/kernels/iq1s_tables.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace sparkinfer::kernels::k3;

struct BlockIQ1S { uint16_t d; uint8_t qs[32]; uint16_t qh[8]; };
static_assert(sizeof(BlockIQ1S) == 50, "IQ1_S block must be 50 bytes");

extern "C" void dequantize_row_iq1_s(const void* x, float* y, int64_t k);

// fp16 -> fp32 is exact for every finite half, so any correct implementation agrees
// with ggml's bit for bit. Avoid NaN/Inf inputs and there is no ambiguity.
static float h2f(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const uint32_t exp  = (h >> 10) & 0x1f;
    const uint32_t man  = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else {                                   // subnormal half -> normal float
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

// EXACTLY the kernel's per-thread body: recover (ib32, l) from a flat group id.
static void kernel_port(float* out, const BlockIQ1S* blocks, int64_t n_groups) {
    for (int64_t g = 0; g < n_groups; ++g) {
        const int64_t ib   = g >> 5;
        const int     l32  = (int)(g & 31);
        const int     ib32 = l32 >> 2;
        const int     l    = l32 & 3;

        const BlockIQ1S& b = blocks[ib];
        const float    d   = h2f(b.d);
        const uint16_t h   = b.qh[ib32];

        const float dl    = d * (float)(2 * ((h >> 12) & 7) + 1);
        const float delta = (h & 0x8000) ? -SPARKINFER_IQ1S_DELTA : SPARKINFER_IQ1S_DELTA;
        const uint32_t idx =
            (uint32_t)b.qs[4 * ib32 + l] | (((uint32_t)(h >> (3 * l)) & 7u) << 8);
        const int8_t* grid = (const int8_t*)&iq1s_grid_host[idx];

        float* dst = out + g * 8;
        for (int j = 0; j < 8; ++j) dst[j] = dl * ((float)grid[j] + delta);
    }
}

int main() {
    const int64_t NB = 8192;                 // blocks
    const int64_t N  = NB * 256;             // values
    std::vector<BlockIQ1S> blocks(NB);
    std::mt19937 rng(20260730);
    std::uniform_int_distribution<int> byte_d(0, 255);

    for (auto& b : blocks) {
        // d: a finite, non-tiny half. Exponent in [8, 22] keeps it well-scaled.
        const uint16_t exp = (uint16_t)(8 + (rng() % 15));
        b.d = (uint16_t)((rng() & 1) << 15 | (exp << 10) | (rng() % 1024));
        for (int i = 0; i < 32; ++i) b.qs[i] = (uint8_t)byte_d(rng);
        for (int i = 0; i < 8; ++i)  b.qh[i] = (uint16_t)(rng() & 0xffff);
    }

    std::vector<float> ref(N), got(N);
    dequantize_row_iq1_s(blocks.data(), ref.data(), N);
    kernel_port(got.data(), blocks.data(), N / 8);

    int64_t mismatch = 0; double worst = 0; int64_t first = -1;
    for (int64_t i = 0; i < N; ++i) {
        if (memcmp(&ref[i], &got[i], 4) != 0) {
            ++mismatch;
            if (first < 0) first = i;
            const double dd = std::abs((double)ref[i] - (double)got[i]);
            if (dd > worst) worst = dd;
        }
    }
    // Coverage: how much of the 2048-entry lattice did these blocks actually touch?
    std::vector<char> touched(SPARKINFER_IQ1S_NGRID, 0);
    for (const auto& b : blocks)
        for (int ib32 = 0; ib32 < 8; ++ib32)
            for (int l = 0; l < 4; ++l)
                touched[(uint32_t)b.qs[4*ib32+l] | (((uint32_t)(b.qh[ib32] >> (3*l)) & 7u) << 8)] = 1;
    int cov = 0; for (char c : touched) cov += c;

    std::printf("IQ1_S dequant vs real ggml dequantize_row_iq1_s\n");
    std::printf("  %lld values (%lld blocks), lattice coverage %d/%d entries\n",
                (long long)N, (long long)NB, cov, SPARKINFER_IQ1S_NGRID);
    std::printf("  bit mismatches: %lld", (long long)mismatch);
    if (mismatch) std::printf("  (first at %lld, worst abs diff %.6g)", (long long)first, worst);
    std::printf("\n\n%s\n", mismatch == 0
        ? "PASS: kernel index arithmetic is bit-identical to ggml"
        : "FAIL: kernel diverges from ggml");
    return mismatch == 0 ? 0 : 1;
}
