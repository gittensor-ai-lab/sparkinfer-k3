// IQ1_S on the device: the dequant kernel, and the MoE dispatch built on it.
//
// Two things this adds over the CPU crosscheck (kimi_k3_iq1s_ggml_cpu_test.cpp, which
// already proves the index arithmetic is bit-identical to ggml):
//
//   1. THE CONSTANT-MEMORY PATH. The 2048-entry lattice lives in __constant__ and is
//      uploaded once by ensure_iq1s_tables(). A failed or partial upload gives a table
//      of zeros, which dequantises every weight to dl*delta — a small, uniform,
//      entirely plausible-looking tensor. Comparing the kernel against the
//      ggml-verified host arithmetic catches that; a self-consistency check would not.
//
//   2. THE MoE DISPATCH AT IQ1_S. The dispatch is templated on the block type, so
//      instantiating it for BlockIQ1S is a different code path from the validated
//      IQ2_XS one: different block size (50 vs 74 bytes), different decode, different
//      stride arithmetic in the row pointers.
//
// SYNTHETIC BLOCKS ARE THE RIGHT INPUT, not a compromise. Every bit pattern is a valid
// IQ1_S block for dequant purposes (the grid index is 11 bits, so it cannot exceed the
// table), so random blocks sweep the whole lattice — more coverage than any single real
// tensor, which only touches the codepoints its own weights happen to use.

#include "sparkinfer/kernels/kimi_k3.h"
#include "sparkinfer/kernels/iq1s_tables.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace sparkinfer::kernels::k3;

static int g_fail = 0;
#define CU(e) do{ cudaError_t e_=(e); if(e_!=cudaSuccess){ \
  std::printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); return 1;} }while(0)

struct BlockIQ1S { uint16_t d; uint8_t qs[32]; uint16_t qh[8]; };
static_assert(sizeof(BlockIQ1S) == 50, "IQ1_S block must be 50 bytes");

// fp16 -> fp32, exact for every finite half.
static float h2f(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const uint32_t exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else { int e=-1; uint32_t m=man; do{ m<<=1; ++e; } while(!(m & 0x400));
               bits = sign | ((uint32_t)(127-15-e) << 23) | ((m & 0x3ff) << 13); }
    } else if (exp == 31) bits = sign | 0x7f800000u | (man << 13);
    else bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    float f; std::memcpy(&f, &bits, 4); return f;
}

// The ggml-verified host arithmetic (see the CPU crosscheck).
static void host_dequant(float* out, const BlockIQ1S* blocks, int64_t n_groups) {
    for (int64_t g = 0; g < n_groups; ++g) {
        const int64_t ib = g >> 5;
        const int l32 = (int)(g & 31), ib32 = l32 >> 2, l = l32 & 3;
        const BlockIQ1S& b = blocks[ib];
        const float d = h2f(b.d);
        const uint16_t h = b.qh[ib32];
        const float dl = d * (float)(2 * ((h >> 12) & 7) + 1);
        const float delta = (h & 0x8000) ? -SPARKINFER_IQ1S_DELTA : SPARKINFER_IQ1S_DELTA;
        const uint32_t idx = (uint32_t)b.qs[4*ib32+l] | (((uint32_t)(h >> (3*l)) & 7u) << 8);
        const int8_t* grid = (const int8_t*)&iq1s_grid_host[idx];
        for (int j = 0; j < 8; ++j) out[g*8+j] = dl * ((float)grid[j] + delta);
    }
}

static void fill_blocks(std::vector<BlockIQ1S>& v, std::mt19937& rng) {
    for (auto& b : v) {
        const uint16_t exp = (uint16_t)(9 + (rng() % 12));   // well-scaled, finite
        b.d = (uint16_t)(((rng() & 1) << 15) | (exp << 10) | (rng() % 1024));
        for (int i = 0; i < 32; ++i) b.qs[i] = (uint8_t)(rng() & 0xff);
        for (int i = 0; i < 8; ++i)  b.qh[i] = (uint16_t)(rng() & 0xffff);
    }
}

static std::vector<float> q8k_roundtrip(const std::vector<float>& x) {
    std::vector<float> out(x.size());
    for (size_t b = 0; b < x.size() / 256; ++b) {
        const float* xb = x.data() + b*256;
        float maxv=0, amax=0;
        for (int j=0; j<256; ++j) {
            const float ax=std::fabs(xb[j]);
            if (ax > amax) { amax=ax; maxv=xb[j]; }
        }
        if (!amax) {
            for (int j=0; j<256; ++j) out[b*256+j]=0;
            continue;
        }
        const float iscale=-127.0f/maxv, d=1.0f/iscale;
        for (int j=0; j<256; ++j) {
            int q=(int)std::nearbyint(iscale*xb[j]);
            q=std::min(127, q);
            out[b*256+j]=d*(float)q;
        }
    }
    return out;
}

int main() {
    std::printf("=== Kimi K3 IQ1_S on device ===\n\n");
    std::mt19937 rng(20260730);

    // ---------------------------------------------------------------- 1. dequant
    {
        const int64_t NB = 4096, N = NB * 256;
        std::vector<BlockIQ1S> blocks(NB);
        fill_blocks(blocks, rng);

        void* dsrc; float* dout;
        CU(cudaMalloc(&dsrc, blocks.size()*sizeof(BlockIQ1S)));
        CU(cudaMalloc(&dout, N*sizeof(float)));
        CU(cudaMemcpy(dsrc, blocks.data(), blocks.size()*sizeof(BlockIQ1S),
                      cudaMemcpyHostToDevice));
        dequant_iq1_s_f32(dout, dsrc, N, 0);
        CU(cudaDeviceSynchronize());
        std::vector<float> got(N), ref(N);
        CU(cudaMemcpy(got.data(), dout, N*sizeof(float), cudaMemcpyDeviceToHost));
        host_dequant(ref.data(), blocks.data(), N/8);

        int64_t mism = 0; double worst = 0;
        for (int64_t i = 0; i < N; ++i)
            if (std::memcmp(&ref[i], &got[i], 4) != 0) {
                ++mism;
                worst = std::fmax(worst, std::fabs((double)ref[i]-(double)got[i]));
            }
        std::vector<char> touched(SPARKINFER_IQ1S_NGRID, 0);
        for (const auto& b : blocks)
            for (int i = 0; i < 8; ++i) for (int l = 0; l < 4; ++l)
                touched[(uint32_t)b.qs[4*i+l] | (((uint32_t)(b.qh[i] >> (3*l)) & 7u) << 8)] = 1;
        int cov = 0; for (char c : touched) cov += c;

        std::printf("[1] dequant_iq1_s_f32 vs ggml-verified host arithmetic\n");
        std::printf("    %lld values, lattice coverage %d/%d\n", (long long)N, cov,
                    SPARKINFER_IQ1S_NGRID);
        std::printf("    bit mismatches: %lld%s\n", (long long)mism,
                    mism ? "" : "  <- device lookup tables verified");
        if (mism) { std::printf("    worst abs diff %.6g\n", worst); ++g_fail; }
        // A zeroed lattice would still produce finite plausible numbers, so assert the
        // output actually varies — the specific failure this guards.
        double mn = got[0], mx = got[0];
        for (float v : got) { mn = std::fmin(mn, v); mx = std::fmax(mx, v); }
        if (!(mx > mn)) { std::printf("    FAIL: output is constant (zeroed table?)\n"); ++g_fail; }
        cudaFree(dsrc); cudaFree(dout);
    }

    // ------------------------------------------------------------ 2. MoE dispatch
    {
        const int LAT = 512, FFN = 512, TOPK = 2;
        const int bpr_gu = LAT/256, bpr_d = FFN/256;
        const float beta = 4.0f, lb = 25.0f;

        std::vector<BlockIQ1S> hg((size_t)TOPK*FFN*bpr_gu), hu(hg.size()),
                               hd((size_t)TOPK*LAT*bpr_d);
        fill_blocks(hg, rng); fill_blocks(hu, rng); fill_blocks(hd, rng);

        std::vector<float> x(LAT);
        std::uniform_real_distribution<float> U(-1.f, 1.f);
        for (auto& v : x) v = U(rng);
        std::vector<int> ids{0, 1};
        std::vector<float> w{0.6f, 0.4f};

        // Reference weights via the verified host dequant, then float64 throughout.
        std::vector<float> G(hg.size()*256), Uw(hu.size()*256), Dw(hd.size()*256);
        host_dequant(G.data(),  hg.data(), (int64_t)hg.size()*32);
        host_dequant(Uw.data(), hu.data(), (int64_t)hu.size()*32);
        host_dequant(Dw.data(), hd.data(), (int64_t)hd.size()*32);

        std::vector<double> ref(LAT, 0.0);
        for (int k = 0; k < TOPK; ++k) {
            std::vector<double> act(FFN);
            for (int j = 0; j < FFN; ++j) {
                double gv = 0, uv = 0;
                for (int i = 0; i < LAT; ++i) {
                    gv += (double)G [((size_t)k*FFN + j)*LAT + i] * x[i];
                    uv += (double)Uw[((size_t)k*FFN + j)*LAT + i] * x[i];
                }
                const double a = beta * std::tanh(gv/beta) * (1.0/(1.0+std::exp(-gv)));
                act[j] = a * (lb * std::tanh(uv/lb));
            }
            for (int o = 0; o < LAT; ++o) {
                double s = 0;
                for (int j = 0; j < FFN; ++j) s += (double)Dw[((size_t)k*LAT + o)*FFN + j] * act[j];
                ref[o] += (double)w[k] * s;
            }
        }

        void *dg, *du, *dd; float *dx, *dw, *dout, *dscr; int* did;
        CU(cudaMalloc(&dg, hg.size()*50)); CU(cudaMalloc(&du, hu.size()*50));
        CU(cudaMalloc(&dd, hd.size()*50));
        CU(cudaMemcpy(dg, hg.data(), hg.size()*50, cudaMemcpyHostToDevice));
        CU(cudaMemcpy(du, hu.data(), hu.size()*50, cudaMemcpyHostToDevice));
        CU(cudaMemcpy(dd, hd.data(), hd.size()*50, cudaMemcpyHostToDevice));
        CU(cudaMalloc(&dx, LAT*4)); CU(cudaMemcpy(dx, x.data(), LAT*4, cudaMemcpyHostToDevice));
        CU(cudaMalloc(&dw, TOPK*4)); CU(cudaMemcpy(dw, w.data(), TOPK*4, cudaMemcpyHostToDevice));
        CU(cudaMalloc(&did, TOPK*4)); CU(cudaMemcpy(did, ids.data(), TOPK*4, cudaMemcpyHostToDevice));
        CU(cudaMalloc(&dout, LAT*4)); CU(cudaMalloc(&dscr, (size_t)TOPK*FFN*4));

        // Go through the RUNTIME-TYPE front door, so the dispatch table is tested too.
        const bool ok = moe_expert_ffn_f32_by_type(dout, dscr, dx, did, dw, dg, du, dd,
                                                   LAT, FFN, TOPK, beta, lb,
                                                   /*ggml_type=*/19, 0);
        CU(cudaDeviceSynchronize());
        std::vector<float> got(LAT);
        CU(cudaMemcpy(got.data(), dout, LAT*4, cudaMemcpyDeviceToHost));

        double num=0, den=0, worst=0;
        for (int i = 0; i < LAT; ++i) {
            const double d = got[i]-ref[i];
            num += d*d; den += ref[i]*ref[i];
            worst = std::fmax(worst, std::fabs(d)/(std::fabs(ref[i])+1e-12));
        }
        const double rl2 = std::sqrt(num/(den+1e-30));
        std::printf("\n[2] moe_expert_ffn (IQ1_S) vs float64, latent=%d ffn=%d top_k=%d\n",
                    LAT, FFN, TOPK);
        std::printf("    dispatch_by_type(19) accepted : %s\n", ok ? "yes" : "NO");
        std::printf("    relL2 = %.3e   worst_rel = %.3e   ref RMS = %.6g\n",
                    rl2, worst, std::sqrt(den/LAT));
        if (!ok || !(rl2 < 2e-5) || !(den > 0)) ++g_fail;

        // Reference-compatible activation path: Q8_K before gate/up and again
        // after situ, exactly as llama.cpp's IQ1_S vec_dot contract requires.
        const std::vector<float> xq = q8k_roundtrip(x);
        std::vector<double> qref(LAT, 0.0);
        for (int k = 0; k < TOPK; ++k) {
            std::vector<float> act(FFN);
            for (int j = 0; j < FFN; ++j) {
                double gv=0, uv=0;
                for (int i=0; i<LAT; ++i) {
                    gv += (double)G [((size_t)k*FFN+j)*LAT+i]*xq[i];
                    uv += (double)Uw[((size_t)k*FFN+j)*LAT+i]*xq[i];
                }
                const double a=beta*std::tanh(gv/beta)/(1.0+std::exp(-gv));
                act[j]=(float)(a*(lb*std::tanh(uv/lb)));
            }
            const std::vector<float> aq=q8k_roundtrip(act);
            for (int o=0; o<LAT; ++o) {
                double v=0;
                for (int j=0; j<FFN; ++j)
                    v += (double)Dw[((size_t)k*LAT+o)*FFN+j]*aq[j];
                qref[o] += (double)w[k]*v;
            }
        }
        void* dq8=nullptr;
        CU(cudaMalloc(&dq8, k3_moe_q8_k_bytes(LAT, FFN, TOPK)));
        const bool qok = moe_expert_ffn_f32_by_type(
            dout, dscr, dx, did, dw, dg, du, dd, LAT, FFN, TOPK, beta, lb,
            /*ggml_type=*/19, 0, 0, 0, dq8);
        CU(cudaDeviceSynchronize());
        CU(cudaMemcpy(got.data(), dout, LAT*4, cudaMemcpyDeviceToHost));
        num=0; den=0;
        for (int i=0; i<LAT; ++i) {
            const double d=got[i]-qref[i];
            num+=d*d; den+=qref[i]*qref[i];
        }
        const double qrl2=std::sqrt(num/(den+1e-30));
        std::printf("    Q8_K activation relL2     : %.3e (%s)\n",
                    qrl2, qok ? "accepted" : "REFUSED");
        if (!qok || !(qrl2 < 5e-5)) ++g_fail;
        cudaFree(dq8);

        // An unknown type must be REFUSED, not silently decoded with the wrong reader.
        const bool bad = moe_expert_ffn_f32_by_type(dout, dscr, dx, did, dw, dg, du, dd,
                                                    LAT, FFN, TOPK, beta, lb, 12, 0);
        std::printf("    unknown type 12 refused      : %s\n", bad ? "NO -- BUG" : "yes");
        if (bad) ++g_fail;

        cudaFree(dg); cudaFree(du); cudaFree(dd); cudaFree(dx);
        cudaFree(dw); cudaFree(did); cudaFree(dout); cudaFree(dscr);
    }

    std::printf("\n%s\n", g_fail ? "FAIL" : "PASS: IQ1_S device path verified");
    return g_fail ? 1 : 0;
}
