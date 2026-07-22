/* v6: same-binary fwd comparison — floor / P1a / IL-derived / IL-emitted / IL-NT.
 * All arms share one container window; ratios within a run are comparable. */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>
#include "prime_dispatch.h"
#include "exhaustive_plan.h"
#include "il_layout.h"
#include "il_execute.h"
#include "generator/generated/registry.h"
#ifdef USE512
#define ISA avx512
#else
#define ISA avx2
#endif
#define CAT2(a,b) a##b
#define CAT(a,b) CAT2(a,b)
#define STR_F  CAT(radix64_n1_fwd_,CAT(ISA,_strided))
#define ILO_D  CAT(radix64_n1_fwd_,CAT(ISA,_strided_il_out_DRV))
#define ILO_E  CAT(radix64_n1_fwd_,CAT(ISA,_strided_il_out))
#define ILO_N  CAT(radix64_n1_fwd_,CAT(ISA,_strided_il_out_nt))
typedef void (*ilo_fn)(const double*,const double*,double*,const double*,const double*,size_t,size_t);
void STR_F(double*,double*,const double*,const double*,size_t,size_t);
void ILO_D(const double*,const double*,double*,const double*,const double*,size_t,size_t);
void ILO_E(const double*,const double*,double*,const double*,const double*,size_t,size_t);
void ILO_N(const double*,const double*,double*,const double*,const double*,size_t,size_t);
static stride_plan_t *P0,*P1;
static const size_t K2=4096, NTOT=262144;
static void fwd_il(ilo_fn f, const double *z, double *zo, double *cr, double *ci){
    vfft_proto_execute_fwd_ilin(P0,z,cr,ci,K2);
    for(size_t s=0;s<64;s++) vfft_proto_execute_fwd(P1,cr+s*K2,ci+s*K2,64);
    for(size_t s=0;s<64;s++) f(cr+s*K2,ci+s*K2,zo+2*s*K2,NULL,NULL,64,64);
}
static void fwd_floor(double *re,double *im){
    vfft_proto_execute_fwd(P0,re,im,K2);
    for(size_t s=0;s<64;s++) vfft_proto_execute_fwd(P1,re+s*K2,im+s*K2,64);
    for(size_t s=0;s<64;s++) STR_F(re+s*K2,im+s*K2,NULL,NULL,64,64);
}
int main(void){
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    stride_set_num_threads(1);
    P0=vfft_proto_exhaustive_plan(64,K2,&reg,0);
    P1=vfft_proto_exhaustive_plan(64,64,&reg,0);
    size_t n=NTOT;
    double *z=aligned_alloc(64,n*16),*z1=aligned_alloc(64,n*16),*z2=aligned_alloc(64,n*16);
    double *cr=aligned_alloc(64,n*8),*ci=aligned_alloc(64,n*8);
    double *re=aligned_alloc(64,n*8),*im=aligned_alloc(64,n*8);
    srand(3); for(size_t i=0;i<2*n;i++) z[i]=2.0*rand()/RAND_MAX-1;
    fwd_il(ILO_D,z,z1,cr,ci); fwd_il(ILO_E,z,z2,cr,ci);
    printf("emitted vs derived (whole) : %s\n", memcmp(z1,z2,n*16)?"**FAIL**":"BIT-EXACT");
    fwd_il(ILO_N,z,z2,cr,ci);
    printf("NT vs derived (whole)      : %s\n", memcmp(z1,z2,n*16)?"**FAIL**":"BIT-EXACT");
    int reps=6; double t_fl=1e18,t_p1a=1e18,t_d=1e18,t_e=1e18,t_n=1e18;
    for(int t=0;t<7;t++){
        double t0,v;
        t0=(double)__rdtsc(); for(int r=0;r<reps;r++) fwd_floor(re,im);
        v=((double)__rdtsc()-t0)/reps; if(v<t_fl)t_fl=v;
        t0=(double)__rdtsc(); for(int r=0;r<reps;r++){ vfft_il2sp(z,re,im,n); fwd_floor(re,im); vfft_sp2il(re,im,z1,n); }
        v=((double)__rdtsc()-t0)/reps; if(v<t_p1a)t_p1a=v;
        t0=(double)__rdtsc(); for(int r=0;r<reps;r++) fwd_il(ILO_D,z,z1,cr,ci);
        v=((double)__rdtsc()-t0)/reps; if(v<t_d)t_d=v;
        t0=(double)__rdtsc(); for(int r=0;r<reps;r++) fwd_il(ILO_E,z,z1,cr,ci);
        v=((double)__rdtsc()-t0)/reps; if(v<t_e)t_e=v;
        t0=(double)__rdtsc(); for(int r=0;r<reps;r++) fwd_il(ILO_N,z,z1,cr,ci);
        v=((double)__rdtsc()-t0)/reps; if(v<t_n)t_n=v;
    }
    printf("floor %9.0f | P1a %.3fx | IL-drv %.3fx | IL-emit %.3fx | IL-NT %.3fx\n",
           t_fl, t_p1a/t_fl, t_d/t_fl, t_e/t_fl, t_n/t_fl);
    return 0;
}
