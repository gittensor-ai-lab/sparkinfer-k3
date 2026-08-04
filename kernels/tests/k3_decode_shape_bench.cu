// PER-KERNEL TIME AT THE SHAPES THE DECODE ACTUALLY LAUNCHES.
//
// Why this exists: nsys cannot run in this container (it dies on
// /sys/devices/virtual/nvidia-pci-gpu), and the in-tree K3Profiler is phase-level and
// incompatible with graph capture. That left only proxies -- node COUNT, then idle
// SM-SLOTS -- and both have now misled a decision here:
//
//   * node count said the AttnRes 3-way fusion was worth 370 launches; it measured -6.0%
//     because it collapsed 28 CTAs into 1.
//   * idle SM-slots said the activation quantiser was 25.7% of the problem; the launch
//     -shape census then showed 370 of its 557 launches are on K <= 1536, i.e. a few KB
//     of work behind ~1.5 us of launch latency, where widening the grid recovers nothing.
//
// A proxy is not the claim. This measures the claim: microseconds per launch, at the
// exact (n, head_dim, n_head, ...) the census read off the captured graph, multiplied by
// the census launch count to give microseconds PER TOKEN PER RANK. That product is the
// only number that says whether a kernel is worth touching.
//
// WHAT THIS DOES NOT MODEL, and it matters when reading the totals:
//   * No contention. Each kernel runs alone on an idle GPU, so it gets the whole memory
//     system. In the real graph it shares with whatever overlaps it.
//   * No overlap. The per-token column assumes serial execution; the captured graph
//     overlaps independent branches (that is what #114 was), so the column SUMS TO MORE
//     than the token. Read it as a ranking and a budget ceiling, not as a decomposition.
// Both distortions push the same way for every row, which is what keeps the RANKING
// sound even though the absolute numbers are optimistic.
#include "sparkinfer/kernels/kimi_k3.h"
#include "sparkinfer/kernels/kimi_k3_fast.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <string>
#include <vector>

using namespace sparkinfer::kernels::k3;

#define CU(e) do{ cudaError_t e_=(e); if(e_!=cudaSuccess){ \
    std::printf("CUDA %s at line %d\n", cudaGetErrorString(e_), __LINE__); \
    std::exit(1);} }while(0)

namespace {

float* dev_rand(size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<float> h(n);
    for (auto& v : h) v = U(rng);
    float* d = nullptr;
    CU(cudaMalloc(&d, n * sizeof(float)));
    CU(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

struct Result { std::string name; int count; double us; };
std::vector<Result> results;

// Time one launcher. WARMUP MATTERS MORE THAN ITERATIONS HERE: these kernels are
// 1-3 us, so the first launch's module load and the first touch of every buffer would
// otherwise dominate a short loop.
template <typename F>
void bench(const char* name, int census_count, F&& launch) {
    cudaStream_t st;
    CU(cudaStreamCreate(&st));
    for (int i = 0; i < 200; ++i) launch(st);
    CU(cudaStreamSynchronize(st));

    cudaEvent_t a, b;
    CU(cudaEventCreate(&a));
    CU(cudaEventCreate(&b));
    const int iters = 2000;
    CU(cudaEventRecord(a, st));
    for (int i = 0; i < iters; ++i) launch(st);
    CU(cudaEventRecord(b, st));
    CU(cudaEventSynchronize(b));
    float ms = 0;
    CU(cudaEventElapsedTime(&ms, a, b));
    const double us = (double)ms * 1000.0 / iters;

    results.push_back({name, census_count, us});
    std::printf("  %-42s %7.2f us x %4d = %8.1f us/token\n",
                name, us, census_count, us * census_count);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaStreamDestroy(st);
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("no CUDA device — skipping\n");
        return 0;
    }
    cudaDeviceProp prop{};
    CU(cudaGetDeviceProperties(&prop, 0));

    // K3 at the scored shape, PER RANK (tp=8). hidden is replicated; the KDA attention
    // and the MoE ffn are sharded, which is why n_head is 12 and the ffn band is 768.
    const int H         = 7168;    // hidden
    const int LAT       = 3584;    // expert_latent
    const int QKV       = 1536;    // this rank's KDA qkv width = n_head * head_dim
    const int BAND      = 768;     // this rank's MoE ffn band
    const int HEAD_DIM  = 128;
    const int N_HEAD    = 12;
    const float eps     = 1e-5f;

    std::printf("k3 decode kernels at census shapes — %s, %d SMs\n",
                prop.name, prop.multiProcessorCount);
    std::printf("counts are launches per rank per token, read off the captured graph\n\n");

    float* x_h   = dev_rand(H, 1);
    float* y_h   = dev_rand(H, 2);
    float* w_h   = dev_rand(H, 3);
    float* x_lat = dev_rand(LAT, 4);
    float* x_qkv = dev_rand(QKV, 5);
    float* x_bnd = dev_rand(BAND, 6);
    float* a_h   = dev_rand(H, 7);
    float* b_h   = dev_rand(H, 8);
    float* c_h   = dev_rand(H, 9);

    void* q8 = nullptr;
    CU(cudaMalloc(&q8, k3_q8_0_bytes(H) + 64));

    std::printf("[activation quantiser] 557 launches, 25.7%% of idle SM-slots\n");
    bench("k3_quantize_act_f32 K=7168", 186, [&](cudaStream_t s){ k3_quantize_act_f32(q8, x_h, H, s); });
    bench("k3_quantize_act_f32 K=3584", 92,  [&](cudaStream_t s){ k3_quantize_act_f32(q8, x_lat, LAT, s); });
    bench("k3_quantize_act_f32 K=1536", 117, [&](cudaStream_t s){ k3_quantize_act_f32(q8, x_qkv, QKV, s); });
    bench("k3_quantize_act_f32 K=768",  92,  [&](cudaStream_t s){ k3_quantize_act_f32(q8, x_bnd, BAND, s); });
    bench("k3_quantize_act_f32 K=256",  69,  [&](cudaStream_t s){ k3_quantize_act_f32(q8, x_bnd, 256, s); });

    std::printf("\n[rms norm] 327 launches — #115 widened these, #127's RMSG spreads them\n");
    bench("rms_norm_f32 n=7168", 187, [&](cudaStream_t s){ rms_norm_f32(y_h, x_h, w_h, H, eps, s); });
    bench("rms_norm_f32 n=1536", 92,  [&](cudaStream_t s){ rms_norm_f32(y_h, x_qkv, w_h, QKV, eps, s); });

    std::printf("\n[attention residual] score+apply, 372 launches, 16.7%% of idle\n");
    float* ckpts  = dev_rand((size_t)H * 8, 10);
    float* scores = dev_rand(16, 11);
    bench("attn_res_mix_f32 n_ckpt=8", 186,
          [&](cudaStream_t s){ attn_res_mix_f32(y_h, ckpts, x_h, w_h, H, 8, eps, s, scores); });

    std::printf("\n[KDA state chain] 69 layers x 4 kernels, all 12-36 blocks of 128 threads\n");
    float* g_raw = dev_rand(QKV, 12);
    float* A_h   = dev_rand(N_HEAD, 13);
    float* o_h   = dev_rand(QKV, 14);
    float* g2_h  = dev_rand(QKV, 15);
    bench("kda_gate_out_f32", 69,
          [&](cudaStream_t s){ kda_gate_out_f32(y_h, o_h, w_h, g2_h, HEAD_DIM, N_HEAD, eps, s); });
    bench("kda_decay_gate_f32", 69,
          [&](cudaStream_t s){ kda_decay_gate_f32(y_h, g_raw, A_h, HEAD_DIM, N_HEAD, -8.f, s); });

    const int D_CONV = 4;
    float* stq = dev_rand((size_t)QKV * D_CONV, 16);
    float* stk = dev_rand((size_t)QKV * D_CONV, 17);
    float* stv = dev_rand((size_t)QKV * D_CONV, 18);
    float* wq  = dev_rand((size_t)QKV * D_CONV, 19);
    float* wk  = dev_rand((size_t)QKV * D_CONV, 20);
    float* wv  = dev_rand((size_t)QKV * D_CONV, 21);
    float* oq  = dev_rand(QKV, 22);
    float* ok_ = dev_rand(QKV, 23);
    float* ov  = dev_rand(QKV, 24);
    bench("k3_kda_conv_l2_fused", 69, [&](cudaStream_t s){
        k3_kda_conv_l2_fused(oq, ok_, ov, stq, stk, stv, x_qkv, x_qkv, x_qkv,
                             wq, wk, wv, D_CONV, HEAD_DIM, N_HEAD, 1.0f, eps, s);
    });

    std::printf("\n[elementwise] 178 launches at 28 blocks — 7.2%% of idle\n");
    bench("k3_add3_f32 n=7168", 92,
          [&](cudaStream_t s){ k3_add3_f32(y_h, x_h, a_h, b_h, c_h, H, s); });
    bench("situ_f32 n=768", 92,
          [&](cudaStream_t s){ situ_f32(y_h, a_h, b_h, BAND, 1.0f, 1.0f, s); });

    // THE DECISIVE PAIR for the fused norm->Q8_0. Unfused, standalone, the two kernels
    // cost 3.15 + 2.98 = 6.13 us. If the fused form measures ~3.2 us it saves ~2.9 us at
    // each of 186 sites = 0.54 ms = 2.5% -- and the decode measuring -0.70% would then
    // mean those sites are OFF THE CRITICAL PATH, i.e. the norm and the quantise were
    // already overlapping something. If instead it measures ~6 us, the kernel itself is
    // at fault: its quantise runs on the norm's single CTA (32 warps over 224 q8 blocks)
    // rather than the quantiser's 28 CTAs. Two diagnoses, opposite fixes, one measurement
    // — and neither is guessable from the decode number alone.
#ifdef K3_HAVE_FUSED_RMS_Q8
    std::printf("\n[fused norm->Q8_0] is the kernel slow, or is the site off-path?\n");
    bench("rms_norm + quantise (unfused pair)", 186, [&](cudaStream_t s){
        rms_norm_f32(y_h, x_h, w_h, H, eps, s);
        k3_quantize_act_f32(q8, y_h, H, s);
    });
    bench("k3_rms_norm_q8_f32 (fused)", 186, [&](cudaStream_t s){
        k3_rms_norm_q8_f32(y_h, q8, x_h, w_h, H, eps, s);
    });
#endif

    // THE MoE EXPERTS — 52.9% of GPU time by the phase profiler, and the only region
    // where a 2% token win is reachable (4% of this phase clears the gate).
    //
    // The weights are RANDOM BYTES, which is legitimate for timing and only for timing:
    // IQ1_S decodes qs (8 bits) | qh (3 bits) into an 11-bit index, so every possible
    // byte pattern indexes inside the 2048-entry lattice. There is no out-of-range case
    // to hit and no data-dependent branch, so the instruction stream is the real one.
    // The gather ADDRESSES are random, which is what the real distribution looks like
    // anyway. Do not read any accuracy claim out of this: the outputs are garbage.
    {
        const int LATENT = 3584, FFN = 768, TOPK = 16, NLOCAL = 448;
        const int gu_blocks = LATENT / 256;          // 14
        const int dn_blocks = FFN / 256;             // 3
        const size_t IQ1S_BLK = 50;                  // 2 (d) + 32 (qs) + 16 (qh)
        const size_t gu_bytes = (size_t)NLOCAL * FFN * gu_blocks * IQ1S_BLK;
        const size_t dn_bytes = (size_t)NLOCAL * LATENT * dn_blocks * IQ1S_BLK;

        void *gate_e = nullptr, *up_e = nullptr, *down_e = nullptr;
        if (cudaMalloc(&gate_e, gu_bytes) != cudaSuccess ||
            cudaMalloc(&up_e,   gu_bytes) != cudaSuccess ||
            cudaMalloc(&down_e, dn_bytes) != cudaSuccess) {
            std::printf("\n[MoE experts] SKIPPED — could not allocate %.1f GiB of expert "
                        "weights\n", (double)(2 * gu_bytes + dn_bytes) / (1 << 30));
        } else {
            // Fill with a cheap pseudo-random pattern so the lattice gather is scattered
            // the way it is in the real model; an all-zero table would make every lane
            // hit codepoint 0 and turn a divergent gather into a broadcast — which would
            // flatter this kernel by exactly the effect being measured.
            {
                std::vector<unsigned char> h(1u << 20);
                std::mt19937 rng(12345);
                for (auto& v : h) v = (unsigned char)(rng() & 0xff);
                for (size_t off = 0; off < gu_bytes; off += h.size()) {
                    const size_t n = std::min(h.size(), gu_bytes - off);
                    CU(cudaMemcpy((char*)gate_e + off, h.data(), n, cudaMemcpyHostToDevice));
                    CU(cudaMemcpy((char*)up_e   + off, h.data(), n, cudaMemcpyHostToDevice));
                }
                for (size_t off = 0; off < dn_bytes; off += h.size()) {
                    const size_t n = std::min(h.size(), dn_bytes - off);
                    CU(cudaMemcpy((char*)down_e + off, h.data(), n, cudaMemcpyHostToDevice));
                }
            }
            float* x_lat2   = dev_rand(LATENT, 30);
            float* out_lat  = dev_rand(LATENT, 31);
            float* scratch  = dev_rand((size_t)TOPK * FFN, 32);
            float* wsel     = dev_rand(TOPK, 33);
            // HALF THE SELECTIONS ARE FOREIGN, which is the real tp=8 case: the router
            // picks 16 of 896 global experts and this rank holds 448 of them. Making all
            // 16 local would double the work and overstate the kernel by 2x.
            int* ids = nullptr;
            {
                std::vector<int> h(TOPK);
                for (int i = 0; i < TOPK; ++i) h[i] = (i % 2) ? (448 + i * 13) : (i * 27);
                CU(cudaMalloc(&ids, TOPK * sizeof(int)));
                CU(cudaMemcpy(ids, h.data(), TOPK * sizeof(int), cudaMemcpyHostToDevice));
            }
            // BIT-IDENTITY AT THE REAL SHAPE, dumped for cross-run comparison.
            //
            // kimi_k3_numeric_test and kimi_k3_iq1s_gpu_test both pass, and neither one
            // proves anything about SPARKINFER_K3_MOE_BPR: they run shrunk dims, and the
            // specialisation only engages at latent == 3584 && ffn == 768. A test that
            // cannot reach the code path under test is not coverage.
            //
            // The gate is read once into a function-local static, so one process cannot
            // run both arms. Instead dump the output as raw bytes; the caller runs the
            // binary twice, with the gate on and off, and diffs. Same input, same seeds,
            // so any difference at all is the specialisation changing arithmetic.
            if (std::getenv("K3_BENCH_DUMP_MOE")) {
                moe_expert_ffn_f32_by_type(out_lat, scratch, x_lat2, ids, wsel,
                                           gate_e, up_e, down_e, LATENT, FFN, TOPK,
                                           1.0f, 1.0f, 19, 0, 0, NLOCAL);
                CU(cudaDeviceSynchronize());
                std::vector<float> h(LATENT);
                CU(cudaMemcpy(h.data(), out_lat, LATENT * sizeof(float),
                              cudaMemcpyDeviceToHost));
                std::FILE* f = std::fopen(std::getenv("K3_BENCH_DUMP_MOE"), "wb");
                if (f) { std::fwrite(h.data(), sizeof(float), LATENT, f); std::fclose(f); }
                std::printf("[MoE experts] dumped %d floats for bit-comparison\n", LATENT);
            }
            std::printf("\n[MoE experts] 52.9%% of GPU time — a 4%% win here is 2%% of the token\n");
            bench("moe_expert_ffn IQ1_S (gate/up + down)", 92, [&](cudaStream_t s){
                moe_expert_ffn_f32_by_type(out_lat, scratch, x_lat2, ids, wsel,
                                           gate_e, up_e, down_e, LATENT, FFN, TOPK,
                                           1.0f, 1.0f, /*ggml_type=*/19, s,
                                           /*expert_begin=*/0, /*n_local_experts=*/NLOCAL);
            });
            std::printf("      (blocks_per_row: gate/up %d, down %d — both compile-time "
                        "constants passed as runtime args)\n", gu_blocks, dn_blocks);

            // THE OTHER EXPERT PATH. Passing q8k_scratch selects block_dot_q8k, which
            // quantises the activation to Q8_K and keeps the dot in INTEGERS until the
            // whole 256-value block is reduced, instead of decoding each lattice entry to
            // float inside the inner loop. It is the llama.cpp-contract path and the
            // decode never takes it. Whether that is leaving time on the table has, as far
            // as this tree records, never been measured — so measure it.
            //
            // NOT bit-identical to the f32 path and not proposed as a drop-in: quantising
            // the activation is a real numerical change that would need a KLD argument.
            // This is a scouting number to decide whether that argument is worth making.
            void* q8k = nullptr;
            if (cudaMalloc(&q8k, 1u << 20) == cudaSuccess) {
                bench("moe_expert_ffn IQ1_S via Q8_K (integer dot)", 92, [&](cudaStream_t s){
                    moe_expert_ffn_f32_by_type(out_lat, scratch, x_lat2, ids, wsel,
                                               gate_e, up_e, down_e, LATENT, FFN, TOPK,
                                               1.0f, 1.0f, 19, s, 0, NLOCAL, q8k);
                });
            }
        }
    }

    double total = 0;
    for (const auto& r : results) total += r.us * r.count;
    std::printf("\n  %-42s %22.1f us/token/rank\n", "SUM OF THE ABOVE", total);
    std::printf("  %-42s %22.1f us/token/rank\n", "measured token (main @ de47bfe)", 21500.0);
    std::printf("\nThe sum exceeds no bound and proves nothing on its own — these ran alone,\n"
                "the real graph overlaps them. Use the RANKING; confirm any candidate with a\n"
                "paired A/B on the full decode before believing a number.\n");
    return 0;
}
