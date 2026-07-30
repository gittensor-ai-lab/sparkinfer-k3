#include "sparkinfer/kernels/kimi_k3.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
using namespace sparkinfer::kernels::k3;
#define CU(e) do{ cudaError_t e_=(e); if(e_!=cudaSuccess){ std::printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); return 1;} }while(0)
struct BlockQ8_0 { uint16_t d; int8_t qs[32]; };
static float h2f(uint16_t h){ uint32_t s=(uint32_t)(h&0x8000)<<16, e=(h>>10)&0x1f, m=h&0x3ff, b;
  if(e==0) b=s; else if(e==31) b=s|0x7f800000u|(m<<13); else b=s|((e-15+127)<<23)|(m<<13);
  float f; std::memcpy(&f,&b,4); return f; }
int main(){
  int fail=0; std::mt19937 rng(1);
  // Q8_0 row dequant
  { const int K=512;
    std::vector<BlockQ8_0> row(K/32);
    for(auto&b:row){ uint16_t e=(uint16_t)(9+(rng()%8)); b.d=(uint16_t)(((rng()&1)<<15)|(e<<10)|(rng()%1024));
      for(auto&q:b.qs) q=(int8_t)((int)(rng()%256)-128); }
    std::vector<double> ref(K); for(int b=0;b<K/32;++b) for(int j=0;j<32;++j) ref[b*32+j]=h2f(row[b].d)*(double)row[b].qs[j];
    void* d; CU(cudaMalloc(&d,row.size()*sizeof(BlockQ8_0))); CU(cudaMemcpy(d,row.data(),row.size()*sizeof(BlockQ8_0),cudaMemcpyHostToDevice));
    float* dout; CU(cudaMalloc(&dout,K*4));
    bool ok=dequant_f32_by_type(dout,d,K,8,0); CU(cudaDeviceSynchronize());
    std::vector<float> got(K); CU(cudaMemcpy(got.data(),dout,K*4,cudaMemcpyDeviceToHost));
    double num=0,den=0; for(int i=0;i<K;++i){double dd=got[i]-ref[i]; num+=dd*dd; den+=ref[i]*ref[i];}
    double rl2=std::sqrt(num/(den+1e-30));
    std::printf("[Q8_0 row] accepted=%s relL2=%.3e\n", ok?"yes":"NO", rl2);
    if(!ok||rl2>1e-6) ++fail;
  }
  // F32 passthrough
  { const int K=300; std::vector<float> row(K); for(auto&v:row) v=(float)(rng()%1000)/17.0f - 30.f;
    void* d; CU(cudaMalloc(&d,K*4)); CU(cudaMemcpy(d,row.data(),K*4,cudaMemcpyHostToDevice));
    float* dout; CU(cudaMalloc(&dout,K*4));
    bool ok=dequant_f32_by_type(dout,d,K,0,0); CU(cudaDeviceSynchronize());
    std::vector<float> got(K); CU(cudaMemcpy(got.data(),dout,K*4,cudaMemcpyDeviceToHost));
    bool same=true; for(int i=0;i<K;++i) if(got[i]!=row[i]) same=false;
    std::printf("[F32 row] accepted=%s bit-identical=%s\n", ok?"yes":"NO", same?"yes":"NO");
    if(!ok||!same) ++fail;
  }
  std::printf("\n%s\n", fail?"FAIL":"PASS");
  return fail?1:0;
}
