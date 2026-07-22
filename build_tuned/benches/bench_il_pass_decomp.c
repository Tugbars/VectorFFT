#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include "prime_dispatch.h"
#include "exhaustive_plan.h"
#include "il_execute.h"
#include "generator/generated/registry.h"
#include <x86intrin.h>
#ifdef USE512
#define ISA avx512
#else
#define ISA avx2
#endif
#define CAT2(a,b) a##b
#define CAT(a,b) CAT2(a,b)
#define STR_F CAT(radix64_n1_fwd_,CAT(ISA,_strided))
#define STR_F_ILO CAT(radix64_n1_fwd_,CAT(ISA,_strided_il_out))
void STR_F(double*,double*,const double*,const double*,size_t,size_t);
void STR_F_ILO(const double*,const double*,double*,const double*,const double*,size_t,size_t);
#define TIME(dst,stmt) { double _b=1e18; for(int _t=0;_t<7;_t++){ double _0=(double)__rdtsc(); for(int _r=0;_r<8;_r++){ stmt; } double _v=((double)__rdtsc()-_0)/8; if(_v<_b)_b=_v;} dst=_b; }
int main(void){
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    stride_set_num_threads(1);
    size_t K=4096,n=64*K;
    stride_plan_t *P0=vfft_proto_exhaustive_plan(64,K,&reg,0);
    stride_plan_t *P1=vfft_proto_exhaustive_plan(64,64,&reg,0);
    double *re=aligned_alloc(64,n*8),*im=aligned_alloc(64,n*8);
    double *cr=aligned_alloc(64,n*8),*ci=aligned_alloc(64,n*8);
    double *z=aligned_alloc(64,n*16),*zo=aligned_alloc(64,n*16);
    srand(4); for(size_t i=0;i<n;i++){re[i]=rand()*1e-9;im[i]=rand()*1e-9;z[2*i]=re[i];z[2*i+1]=im[i];}
    double a0e,a0i,a1,rwe,rwi;
    TIME(a0e, vfft_proto_execute_fwd(P0,re,im,K));
    TIME(a0i, vfft_proto_execute_fwd_ilin(P0,z,cr,ci,K));
    TIME(a1,  for(size_t s=0;s<64;s++) vfft_proto_execute_fwd(P1,re+s*K,im+s*K,64));
    TIME(rwe, for(size_t s=0;s<64;s++) STR_F(re+s*K,im+s*K,NULL,NULL,64,64));
    TIME(rwi, for(size_t s=0;s<64;s++) STR_F_ILO(re+s*K,im+s*K,zo+2*s*K,NULL,NULL,64,64));
    printf("axis0  engine %9.0f | ilin   %9.0f  (+%.0f, %.3fx)\n",a0e,a0i,a0i-a0e,a0i/a0e);
    printf("axis1  shared %9.0f\n",a1);
    printf("rows   split  %9.0f | il_out %9.0f  (+%.0f, %.3fx)\n",rwe,rwi,rwi-rwe,rwi/rwe);
    printf("sum floor %9.0f | sum IL %9.0f | ratio %.3fx\n", a0e+a1+rwe, a0i+a1+rwi, (a0i+a1+rwi)/(a0e+a1+rwe));
    return 0;
}
