#include "sparkinfer/kernels/kimi_k3.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
using namespace sparkinfer::kernels::k3;
#define CU(e) do{ cudaError_t e_=(e); if(e_!=cudaSuccess){ std::printf("CUDA %s\n",cudaGetErrorString(e_)); return false;} }while(0)

// The wide launcher freezes the sum of squares on the original 128-thread scalar
// partition and only widens the elementwise apply. That is load-bearing for the
// accuracy ratchet: changing the sum association moved KL vs main 4.63x on #115.
// So besides a float64 tolerance check, every case below is also compared against a
// second launch of the same entry point on a fresh buffer — if the path were
// nondeterministic across launches the memcmp would fail; the real bit-identical
// claim vs main is what the node eval re-measures.
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

  float *bout, *bx, *bw, *bout2;
  CU(cudaMalloc(&bout, (size_t)(n + 4) * 4));
  CU(cudaMalloc(&bout2, (size_t)(n + 4) * 4));
  CU(cudaMalloc(&bx,   (size_t)(n + 4) * 4));
  CU(cudaMalloc(&bw,   (size_t)(n + 4) * 4));
  float* dout = bout + off_out;
  float* dout2 = bout2 + off_out;
  float* dx   = bx   + off_x;
  float* dw   = bw   + off_w;
  CU(cudaMemcpy(dx, x.data(), (size_t)n * 4, cudaMemcpyHostToDevice));
  CU(cudaMemcpy(dw, w.data(), (size_t)n * 4, cudaMemcpyHostToDevice));

  rms_norm_f32(dout, dx, dw, n, eps, 0);
  rms_norm_f32(dout2, dx, dw, n, eps, 0);
  CU(cudaDeviceSynchronize());
  CU(cudaGetLastError());

  std::vector<float> got(n), got2(n);
  CU(cudaMemcpy(got.data(), dout, (size_t)n * 4, cudaMemcpyDeviceToHost));
  CU(cudaMemcpy(got2.data(), dout2, (size_t)n * 4, cudaMemcpyDeviceToHost));
  cudaFree(bout); cudaFree(bout2); cudaFree(bx); cudaFree(bw);

  const bool stable = std::memcmp(got.data(), got2.data(), (size_t)n * 4) == 0;
  double num = 0, den = 0;
  for (int i = 0; i < n; ++i) { double d = got[i] - ref[i]; num += d * d; den += ref[i] * ref[i]; }
  const double rl2 = std::sqrt(num / (den + 1e-30));
  const bool ok = rl2 < 2e-6 && stable;
  std::printf("  n=%-5d off(out,x,w)=(%d,%d,%d) %-6s relL2=%.3e stable=%d %s\n",
              n, off_out, off_x, off_w,
              (n % 4 == 0 && !off_out && !off_x && !off_w) ? "vec4" : "scalar",
              rl2, (int)stable, ok ? "PASS" : "FAIL");
  return ok;
}

int main() {
  const int widths[] = {7168, 3584, 1536, 512, 4096, 4095, 4097, 3, 1};
  bool ok = true;

  std::printf("aligned (float4 apply where the width allows):\n");
  for (int n : widths) ok &= run_case(n, 0, 0, 0);

  std::printf("misaligned, one pointer at a time (must fall back, must not fault):\n");
  for (int n : {7168, 3584, 1536, 512}) {
    ok &= run_case(n, 1, 0, 0);
    ok &= run_case(n, 0, 1, 0);
    ok &= run_case(n, 0, 0, 1);
  }

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
