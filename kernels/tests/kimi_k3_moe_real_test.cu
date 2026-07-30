#include <cstring>
// Latent MoE expert dispatch, checked against a float64 reference built from
// REAL Kimi K3 expert weights.
//
// The reference dequantises the same IQ2_XS blocks with dequant_iq2_xs_f32 —
// which is itself bit-exact against ggml — then does gate/up/situ/down/combine in
// double. So this isolates the DISPATCH (indexing, shapes, situ placement,
// weighted combine) from the dequant, which is already independently proven.
//
// Shapes are cut down from K3's real 3584/3072/896 so the double-precision
// reference is tractable, but the weights and their layout are the real thing.

#include "sparkinfer/kernels/kimi_k3.h"
#include "sparkinfer/gguf.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace sparkinfer;
using namespace sparkinfer::kernels::k3;

#define CU(e) do{ cudaError_t e_=(e); if(e_!=cudaSuccess){ \
  std::printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); return 1;} }while(0)

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1]
        : "/workspace/models_k3/UD-Q2_K_XL/Kimi-K3-UD-Q2_K_XL-00001-of-00019.gguf";
    GGUF g;
    if (!g.open(path)) { std::printf("open failed\n"); return 1; }

    const GGUFTensor* gt = g.tensor("blk.1.ffn_gate_exps.weight");
    const GGUFTensor* ut = g.tensor("blk.1.ffn_up_exps.weight");
    const GGUFTensor* dt = g.tensor("blk.1.ffn_down_exps.weight");
    if (!gt || !ut || !dt) { std::printf("expert tensors missing\n"); return 1; }
    std::printf("gate %ldx%ldx%ld  up %ldx%ldx%ld  down %ldx%ldx%ld  (type %d)\n",
                gt->dims[0], gt->dims[1], gt->dims[2],
                ut->dims[0], ut->dims[1], ut->dims[2],
                dt->dims[0], dt->dims[1], dt->dims[2], gt->ggml_type);

    // Cut-down but REAL: first LAT columns / FFN rows of the first experts.
    const int LAT = 512, FFN = 512, TOPK = 2;
    const int bpr_gu = LAT / 256, bpr_d = FFN / 256;
    const size_t BB = 74;
    const int real_latent = (int)gt->dims[0];
    const int real_ffn    = (int)gt->dims[1];
    if (real_latent < LAT || real_ffn < FFN) { std::printf("tensor too small\n"); return 1; }

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<float> x(LAT); for (auto& v : x) v = U(rng);
    std::vector<int> ids{0, 1};
    std::vector<float> w{0.6f, 0.4f};
    const float beta = 4.0f, lb = 25.0f;

    // Repack the sub-blocks we need into contiguous [expert][row][block] order,
    // matching what the kernel indexes: (e*ffn + j) * blocks_per_row.
    auto pack_gu = [&](const GGUFTensor* t) {
        std::vector<uint8_t> out((size_t)TOPK * FFN * bpr_gu * BB);
        const uint8_t* src = (const uint8_t*)t->data;
        for (int e = 0; e < TOPK; ++e)
          for (int j = 0; j < FFN; ++j)
            memcpy(&out[((size_t)e*FFN + j)*bpr_gu*BB],
                   src + ((size_t)e*real_ffn + j) * (real_latent/256) * BB, bpr_gu*BB);
        return out;
    };
    auto pack_d = [&](const GGUFTensor* t) {
        const int rl = (int)t->dims[0], ro = (int)t->dims[1];
        std::vector<uint8_t> out((size_t)TOPK * LAT * bpr_d * BB);
        const uint8_t* src = (const uint8_t*)t->data;
        for (int e = 0; e < TOPK; ++e)
          for (int o = 0; o < LAT; ++o)
            memcpy(&out[((size_t)e*LAT + o)*bpr_d*BB],
                   src + ((size_t)e*ro + o) * (rl/256) * BB, bpr_d*BB);
        return out;
    };
    auto hg = pack_gu(gt), hu = pack_gu(ut), hd = pack_d(dt);

    // --- float64 reference, using the BIT-EXACT dequant for the weights ---
    auto deq = [&](const std::vector<uint8_t>& packed, size_t nblocks) {
        void* dsrc; float* dout;
        cudaMalloc(&dsrc, packed.size()); cudaMalloc(&dout, nblocks*256*sizeof(float));
        cudaMemcpy(dsrc, packed.data(), packed.size(), cudaMemcpyHostToDevice);
        dequant_iq2_xs_f32(dout, dsrc, (int64_t)nblocks*256, 0);
        cudaDeviceSynchronize();
        std::vector<float> h(nblocks*256);
        cudaMemcpy(h.data(), dout, h.size()*sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(dsrc); cudaFree(dout);
        return h;
    };
    auto G = deq(hg, (size_t)TOPK*FFN*bpr_gu);
    auto Uw = deq(hu, (size_t)TOPK*FFN*bpr_gu);
    auto D = deq(hd, (size_t)TOPK*LAT*bpr_d);

    std::vector<double> ref(LAT, 0.0);
    for (int k = 0; k < TOPK; ++k) {
        std::vector<double> act(FFN);
        for (int j = 0; j < FFN; ++j) {
            double gv = 0, uv = 0;
            for (int i = 0; i < LAT; ++i) {
                gv += (double)G[((size_t)k*FFN + j)*LAT + i] * x[i];
                uv += (double)Uw[((size_t)k*FFN + j)*LAT + i] * x[i];
            }
            const double a = beta * std::tanh(gv/beta) * (1.0/(1.0+std::exp(-gv)));
            act[j] = a * (lb * std::tanh(uv/lb));
        }
        for (int o = 0; o < LAT; ++o) {
            double s = 0;
            for (int j = 0; j < FFN; ++j) s += (double)D[((size_t)k*LAT + o)*FFN + j] * act[j];
            ref[o] += (double)w[k] * s;
        }
    }

    // --- kernel ---
    void *dg, *du, *dd; float *dx, *dw, *dout, *dscr; int* did;
    cudaMalloc(&dg, hg.size()); cudaMalloc(&du, hu.size()); cudaMalloc(&dd, hd.size());
    cudaMemcpy(dg, hg.data(), hg.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(du, hu.data(), hu.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dd, hd.data(), hd.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&dx, LAT*sizeof(float)); cudaMemcpy(dx, x.data(), LAT*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&dw, TOPK*sizeof(float)); cudaMemcpy(dw, w.data(), TOPK*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&did, TOPK*sizeof(int)); cudaMemcpy(did, ids.data(), TOPK*sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&dout, LAT*sizeof(float)); cudaMalloc(&dscr, (size_t)TOPK*FFN*sizeof(float));

    moe_expert_ffn_iq2xs_f32(dout, dscr, dx, did, dw, dg, du, dd, LAT, FFN, TOPK, beta, lb, 0);
    cudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::printf("CUDA: %s\n", cudaGetErrorString(e)); return 1; }

    std::vector<float> got(LAT);
    cudaMemcpy(got.data(), dout, LAT*sizeof(float), cudaMemcpyDeviceToHost);

    double num=0, den=0, worst=0;
    for (int i = 0; i < LAT; ++i) {
        double d = got[i]-ref[i]; num += d*d; den += ref[i]*ref[i];
        worst = std::fmax(worst, std::fabs(d)/(std::fabs(ref[i])+1e-12));
    }
    const double rl2 = std::sqrt(num/(den+1e-30));
    std::printf("\nlatent=%d ffn=%d top_k=%d (REAL blk.1 expert weights)\n", LAT, FFN, TOPK);
    std::printf("moe dispatch vs float64 ref : relL2=%.3e worst_rel=%.3e\n", rl2, worst);
    std::printf("reference RMS               : %.6g\n", std::sqrt(den/LAT));
    const bool ok = rl2 < 2e-5 && den > 0;
    std::printf("\n%s\n", ok ? "PASS: latent MoE dispatch matches on real K3 weights" : "FAIL");
    return ok ? 0 : 1;
}
