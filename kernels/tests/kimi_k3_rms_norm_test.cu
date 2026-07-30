#include "sparkinfer/kernels/kimi_k3.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
using namespace sparkinfer::kernels::k3;
#define CU(e) do{ cudaError_t e_=(e); if(e_!=cudaSuccess){ std::printf("CUDA %s\n",cudaGetErrorString(e_)); return 1;} }while(0)
int main(){
  const int n=4096; const float eps=1e-5f;
  std::mt19937 rng(3); std::uniform_real_distribution<float> U(-3,3);
  std::vector<float> x(n), w(n); for(auto&v:x)v=U(rng); for(auto&v:w)v=U(rng);
  double ss=0; for(int i=0;i<n;++i) ss+=(double)x[i]*x[i];
  double inv=1.0/std::sqrt(ss/n+eps);
  std::vector<double> ref(n); for(int i=0;i<n;++i) ref[i]=x[i]*inv*w[i];
  float *dx,*dw,*dout; CU(cudaMalloc(&dx,n*4)); CU(cudaMalloc(&dw,n*4)); CU(cudaMalloc(&dout,n*4));
  CU(cudaMemcpy(dx,x.data(),n*4,cudaMemcpyHostToDevice)); CU(cudaMemcpy(dw,w.data(),n*4,cudaMemcpyHostToDevice));
  rms_norm_f32(dout,dx,dw,n,eps,0); CU(cudaDeviceSynchronize());
  std::vector<float> got(n); CU(cudaMemcpy(got.data(),dout,n*4,cudaMemcpyDeviceToHost));
  double num=0,den=0; for(int i=0;i<n;++i){double d=got[i]-ref[i]; num+=d*d; den+=ref[i]*ref[i];}
  double rl2=std::sqrt(num/(den+1e-30));
  std::printf("relL2=%.3e\n%s\n", rl2, rl2<1e-6?"PASS":"FAIL");
  return rl2<1e-6?0:1;
}
