#include "sparkinfer/kernels/kimi_k3.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
using namespace sparkinfer::kernels::k3;
#define CU(e) do{ cudaError_t e_=(e); if(e_!=cudaSuccess){ std::printf("CUDA %s\n",cudaGetErrorString(e_)); return false;} }while(0)

// The launcher picks a block size from the width and takes a float4 path only when the
// width is a multiple of 4 and all three pointers are 16B-aligned. Both halves of that
// decision are load-bearing and neither is visible from the outside: a wrong block size
// still produces a plausible-looking vector, and a float4 load on a 4B-aligned pointer
// is an unspecified-address fault rather than a wrong number. So the sweep covers every
// width K3 actually norms, at every alignment combination that can reach the launcher.
//
// Tolerance is against a float64 reference, so it bounds the f32 reduction's own error
// rather than anything about this change; 2e-6 is loose enough for a 7168-wide f32 sum
// reduced across 32 warps and tight enough that a mis-sized block or a mis-taken path
// (errors of order 1e-1, or a fault) cannot hide under it.
static bool run_case(int n, int off_out, int off_x, int off_w) {
  const float eps = 1e-5f;
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> U(-3, 3);
  std::vector<float> x(n), w(n);
  for (auto& v : x) v = U(rng);
  for (auto& v : w) v = U(rng);

  double ss = 0;
  for (int i = 0; i < n; ++i) ss += (double)x[i] * x[i];
  const double inv = 1.0 / std::sqrt(ss / n + eps);
  std::vector<double> ref(n);
  for (int i = 0; i < n; ++i) ref[i] = x[i] * inv * w[i];

  // cudaMalloc returns 256B-aligned, so a +1 float offset is 4B-aligned and not 16B --
  // exactly the shape of a scratch or GGUF tensor that did not land on a float4 boundary.
  float *bout, *bx, *bw;
  CU(cudaMalloc(&bout, (size_t)(n + 4) * 4));
  CU(cudaMalloc(&bx,   (size_t)(n + 4) * 4));
  CU(cudaMalloc(&bw,   (size_t)(n + 4) * 4));
  float* dout = bout + off_out;
  float* dx   = bx   + off_x;
  float* dw   = bw   + off_w;
  CU(cudaMemcpy(dx, x.data(), (size_t)n * 4, cudaMemcpyHostToDevice));
  CU(cudaMemcpy(dw, w.data(), (size_t)n * 4, cudaMemcpyHostToDevice));

  rms_norm_f32(dout, dx, dw, n, eps, 0);
  CU(cudaDeviceSynchronize());
  CU(cudaGetLastError());

  std::vector<float> got(n);
  CU(cudaMemcpy(got.data(), dout, (size_t)n * 4, cudaMemcpyDeviceToHost));
  cudaFree(bout); cudaFree(bx); cudaFree(bw);

  double num = 0, den = 0;
  for (int i = 0; i < n; ++i) { double d = got[i] - ref[i]; num += d * d; den += ref[i] * ref[i]; }
  const double rl2 = std::sqrt(num / (den + 1e-30));
  const bool ok = rl2 < 2e-6;
  std::printf("  n=%-5d off(out,x,w)=(%d,%d,%d) %-6s relL2=%.3e %s\n",
              n, off_out, off_x, off_w,
              (n % 4 == 0 && !off_out && !off_x && !off_w) ? "vec4" : "scalar",
              rl2, ok ? "PASS" : "FAIL");
  return ok;
}

int main() {
  // The widths K3 actually norms: hidden 7168 (attn_norm + ffn_norm, 93 layers each),
  // latent 3584 (routed_norm), and the MLA q_lora 1536 / kv_lora 512 pair. 4096 is the
  // width this test has always used; 4095 and 4097 are the not-divisible-by-4 widths
  // that must fall to the scalar kernel however well-aligned the pointers are.
  const int widths[] = {7168, 3584, 1536, 512, 4096, 4095, 4097, 3, 1};
  bool ok = true;

  std::printf("aligned (float4 path where the width allows):\n");
  for (int n : widths) ok &= run_case(n, 0, 0, 0);

  // One misaligned pointer is enough to disqualify the vector path, so each is tried on
  // its own -- a guard that only checked x would pass a test that moved all three.
  std::printf("misaligned, one pointer at a time (must fall back, must not fault):\n");
  for (int n : {7168, 3584, 1536, 512}) {
    ok &= run_case(n, 1, 0, 0);
    ok &= run_case(n, 0, 1, 0);
    ok &= run_case(n, 0, 0, 1);
  }

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
