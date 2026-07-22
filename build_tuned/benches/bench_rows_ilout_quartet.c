#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>
#ifdef USE512
#define SPLIT radix64_n1_fwd_avx512_strided
#define DRV   radix64_n1_fwd_avx512_strided_il_out_DRV
#define EMIT  radix64_n1_fwd_avx512_strided_il_out
#define NT    radix64_n1_fwd_avx512_strided_il_out_nt
#else
#define SPLIT radix64_n1_fwd_avx2_strided
#define DRV   radix64_n1_fwd_avx2_strided_il_out_DRV
#define EMIT  radix64_n1_fwd_avx2_strided_il_out
#define NT    radix64_n1_fwd_avx2_strided_il_out_nt
#endif
void SPLIT(double*,double*,const double*,const double*,size_t,size_t);
void DRV(const double*,const double*,double*,const double*,const double*,size_t,size_t);
void EMIT(const double*,const double*,double*,const double*,const double*,size_t,size_t);
void NT(const double*,const double*,double*,const double*,const double*,size_t,size_t);
#define TIME(dst,stmt) { double _b=1e18; for(int _t=0;_t<7;_t++){ double _0=(double)__rdtsc(); for(int _r=0;_r<10;_r++){ stmt; } double _v=((double)__rdtsc()-_0)/10; if(_v<_b)_b=_v;} dst=_b; }
int main(void){
    size_t K=4096,n=64*K;
    double *re=aligned_alloc(64,n*8),*im=aligned_alloc(64,n*8),*z=aligned_alloc(64,n*16);
    srand(4); for(size_t i=0;i<n;i++){re[i]=rand()*1e-9;im[i]=rand()*1e-9;}
    double ts,td,te,tn;
    TIME(ts, for(size_t s=0;s<64;s++) SPLIT(re+s*K,im+s*K,NULL,NULL,64,64));
    TIME(td, for(size_t s=0;s<64;s++) DRV(re+s*K,im+s*K,z+2*s*K,NULL,NULL,64,64));
    TIME(te, for(size_t s=0;s<64;s++) EMIT(re+s*K,im+s*K,z+2*s*K,NULL,NULL,64,64));
    TIME(tn, for(size_t s=0;s<64;s++) NT(re+s*K,im+s*K,z+2*s*K,NULL,NULL,64,64));
    printf("split native (in-place) : %9.0f  1.000x\n",ts);
    printf("il_out derived          : %9.0f  %.3fx (+%.0f)\n",td,td/ts,td-ts);
    printf("il_out emitted          : %9.0f  %.3fx (+%.0f)\n",te,te/ts,te-ts);
    printf("il_out emitted NT       : %9.0f  %.3fx (+%.0f)\n",tn,tn/ts,tn-ts);
    return 0;
}
