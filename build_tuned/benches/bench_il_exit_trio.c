/* bwd exit trio at the plan's own stage-0 geometry: engine inplace split /
 * old per-line il_out / fused-pair il_out. Same binary, gated. */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>
#include "prime_dispatch.h"
#include "exhaustive_plan.h"
#include "generator/generated/registry.h"
#ifdef USE512
#define ISA avx512
#else
#define ISA avx2
#endif
#define CAT2(a,b) a##b
#define CAT(a,b) CAT2(a,b)
typedef void(*ip_fn)(double*,double*,const double*,const double*,size_t,size_t);
typedef void(*il_fn)(const double*,const double*,double*,const double*,const double*,size_t,size_t);
void CAT(radix4_n1_bwd_,ISA)(double*,double*,const double*,const double*,size_t,size_t);
void CAT(radix16_n1_bwd_,ISA)(double*,double*,const double*,const double*,size_t,size_t);
void CAT(radix4_n1_bwd_,CAT(ISA,_il_out))(const double*,const double*,double*,const double*,const double*,size_t,size_t);
void CAT(radix16_n1_bwd_,CAT(ISA,_il_out))(const double*,const double*,double*,const double*,const double*,size_t,size_t);
void CAT(radix4_n1_bwd_,CAT(ISA,_il_out_OLD))(const double*,const double*,double*,const double*,const double*,size_t,size_t);
void CAT(radix16_n1_bwd_,CAT(ISA,_il_out_OLD))(const double*,const double*,double*,const double*,const double*,size_t,size_t);
#ifndef USE512
void radix4_n1_bwd_avx2_il_out_NT(const double*,const double*,double*,const double*,const double*,size_t,size_t);
#endif
#define TIME(dst,stmt) { double _b=1e18; for(int _t=0;_t<7;_t++){ double _0=(double)__rdtsc(); for(int _r=0;_r<10;_r++){ stmt; } double _v=((double)__rdtsc()-_0)/10; if(_v<_b)_b=_v;} dst=_b; }
int main(void){
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    stride_set_num_threads(1);
    stride_plan_t *P0=vfft_proto_exhaustive_plan(64,4096,&reg,0);
    const stride_stage_t *st=&P0->stages[0];
    int R=P0->factors[0], G=st->num_groups; size_t S=st->stride, ME=4096;
    printf("st0 geometry: r=%d groups=%d stride=%zu\n",R,G,S);
    ip_fn ip = R==4?CAT(radix4_n1_bwd_,ISA):CAT(radix16_n1_bwd_,ISA);
    il_fn nw = R==4?CAT(radix4_n1_bwd_,CAT(ISA,_il_out)):CAT(radix16_n1_bwd_,CAT(ISA,_il_out));
    il_fn od = R==4?CAT(radix4_n1_bwd_,CAT(ISA,_il_out_OLD)):CAT(radix16_n1_bwd_,CAT(ISA,_il_out_OLD));
    size_t n=262144;
    double *re=aligned_alloc(64,n*8),*im=aligned_alloc(64,n*8);
    double *z1=aligned_alloc(64,n*16),*z2=aligned_alloc(64,n*16);
    srand(5); for(size_t i=0;i<n;i++){re[i]=rand()*1e-9;im[i]=rand()*1e-9;}
    memset(z1,0xEE,n*16); memset(z2,0xEE,n*16);
    for(int g=0;g<G;g++){ size_t b=st->group_base[g];
        od(re+b,im+b,z1+2*b,NULL,NULL,S,ME); nw(re+b,im+b,z2+2*b,NULL,NULL,S,ME); }
    printf("fused vs old (full z)      : %s\n", memcmp(z1,z2,n*16)?"**FAIL**":"BIT-EXACT");
    double t0,t1,t2;
    TIME(t0, for(int g=0;g<G;g++){size_t b=st->group_base[g]; ip(re+b,im+b,NULL,NULL,S,ME);} );
    TIME(t1, for(int g=0;g<G;g++){size_t b=st->group_base[g]; od(re+b,im+b,z1+2*b,NULL,NULL,S,ME);} );
    TIME(t2, for(int g=0;g<G;g++){size_t b=st->group_base[g]; nw(re+b,im+b,z1+2*b,NULL,NULL,S,ME);} );
    printf("engine inplace split : %9.0f  1.000x\n",t0);
    printf("il_out per-line OLD  : %9.0f  %.3fx (+%.0f)\n",t1,t1/t0,t1-t0);
    printf("il_out FUSED-PAIR    : %9.0f  %.3fx (+%.0f)\n",t2,t2/t0,t2-t0);
#ifndef USE512
    if(R==4){ double t3; memset(z2,0xEE,n*16);
        for(int g=0;g<G;g++){size_t b=st->group_base[g];
            radix4_n1_bwd_avx2_il_out_NT(re+b,im+b,z2+2*b,NULL,NULL,S,ME);}
        printf("NT vs fused (full z)       : %s\n", memcmp(z1,z2,n*16)?"**FAIL**":"BIT-EXACT");
        TIME(t3, for(int g=0;g<G;g++){size_t b=st->group_base[g];
            radix4_n1_bwd_avx2_il_out_NT(re+b,im+b,z1+2*b,NULL,NULL,S,ME);} );
        printf("il_out FUSED-NT      : %9.0f  %.3fx (+%.0f)\n",t3,t3/t0,t3-t0); }
#endif
    return 0;
}
