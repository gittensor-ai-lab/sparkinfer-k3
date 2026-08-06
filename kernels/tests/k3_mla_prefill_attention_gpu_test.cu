#include "sparkinfer/kernels/k3_mla_prefill_attention.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using sparkinfer::kernels::k3::k3_mla_prefill_attention_f32;

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::puts("k3_mla_prefill_attention_gpu_test: SKIP (no CUDA device)"); return 0;
    }
    constexpr int T=5, C=12, S=7, H=3, K=11, L=7, V=5;
    std::mt19937 rng(11); std::uniform_real_distribution<float> d(-0.2f,0.2f);
    std::vector<float> q((size_t)T*H*K), c((size_t)C*K), w((size_t)H*V*L), g((size_t)T*H*V);
    for (auto* a : {&q,&c,&w,&g}) for (float& x : *a) x=d(rng);
    float *dq,*dc,*dw,*dg,*dy;
    cudaMalloc(&dq,q.size()*4); cudaMalloc(&dc,c.size()*4); cudaMalloc(&dw,w.size()*4);
    cudaMalloc(&dg,g.size()*4); cudaMalloc(&dy,g.size()*4);
    cudaMemcpy(dq,q.data(),q.size()*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dc,c.data(),c.size()*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dw,w.data(),w.size()*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dg,g.data(),g.size()*4,cudaMemcpyHostToDevice);
    assert(k3_mla_prefill_attention_f32(dy,dq,dc,dw,dg,S,T,C,H,K,L,V,
                                        1.0f/std::sqrt((float)K),0));
    assert(cudaDeviceSynchronize()==cudaSuccess);
    std::vector<float> y(g.size()); cudaMemcpy(y.data(),dy,y.size()*4,cudaMemcpyDeviceToHost);
    // Independent direct-softmax reference.
    for(int t=0;t<T;++t) for(int h=0;h<H;++h) {
        std::vector<double> score(S+t+1); double m=-INFINITY,z=0;
        for(int p=0;p<=S+t;++p){ double s=0; for(int k=0;k<K;++k)s+=(double)q[((size_t)t*H+h)*K+k]*c[(size_t)p*K+k]; score[p]=s/std::sqrt((double)K); m=std::max(m,score[p]); }
        for(double s:score) z+=std::exp(s-m);
        for(int v=0;v<V;++v){ double ref=0; for(int p=0;p<=S+t;++p){ double latent=0; for(int l=0;l<L;++l) latent+=(double)c[(size_t)p*K+l]*w[((size_t)h*V+v)*L+l]; ref+=std::exp(score[p]-m)*latent; } ref/=z; size_t i=((size_t)t*H+h)*V+v; ref/=1.0+std::exp(-(double)g[i]); assert(std::fabs(y[i]-ref)<2e-5); }
    }
    cudaFree(dq); cudaFree(dc); cudaFree(dw); cudaFree(dg); cudaFree(dy);
    std::puts("k3_mla_prefill_attention_gpu_test: ok");
}
