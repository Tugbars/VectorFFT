/* v8: production-candidate stack. fwd: ilin -> fused(axis1+ILO_emitted).
 * bwd: fused(ILI_{drv|emitted}+axis1) -> ilout. Same-binary arms. */
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
#define STR_B  CAT(radix64_n1_bwd_,CAT(ISA,_strided))
#define ILO_E  CAT(radix64_n1_fwd_,CAT(ISA,_strided_il_out))
#define ILI_E  CAT(radix64_n1_bwd_,CAT(ISA,_strided_il_in))
#define ILI_D  CAT(radix64_n1_bwd_,CAT(ISA,_strided_il_in_DRV))
typedef void (*ili_fn)(const double*,double*,double*,const double*,const double*,size_t,size_t);
void STR_F(double*,double*,const double*,const double*,size_t,size_t);
void STR_B(double*,double*,const double*,const double*,size_t,size_t);
void ILO_E(const double*,const double*,double*,const double*,const double*,size_t,size_t);
void ILI_E(const double*,double*,double*,const double*,const double*,size_t,size_t);
void ILI_D(const double*,double*,double*,const double*,const double*,size_t,size_t);
static stride_plan_t *P0,*P1;
static const size_t K2=4096, NTOT=262144;
static void fwd_prod(const double *z, double *zo, double *cr, double *ci){
    vfft_proto_execute_fwd_ilin(P0,z,cr,ci,K2);
    for(size_t s=0;s<64;s++){
        vfft_proto_execute_fwd(P1,cr+s*K2,ci+s*K2,64);
        ILO_E(cr+s*K2,ci+s*K2,zo+2*s*K2,NULL,NULL,64,64);
    }
}
static void bwd_prod(ili_fn f, const double *z, double *zo, double *cr, double *ci){
    for(size_t s=0;s<64;s++){
        f(z+2*s*K2,cr+s*K2,ci+s*K2,NULL,NULL,64,64);
        vfft_proto_execute_bwd(P1,cr+s*K2,ci+s*K2,64);
    }
    vfft_proto_execute_bwd_ilout(P0,cr,ci,zo,K2);
}
static void fwd_floor(double *re,double *im){
    vfft_proto_execute_fwd(P0,re,im,K2);
    for(size_t s=0;s<64;s++) vfft_proto_execute_fwd(P1,re+s*K2,im+s*K2,64);
    for(size_t s=0;s<64;s++) STR_F(re+s*K2,im+s*K2,NULL,NULL,64,64);
}
static void bwd_floor(double *re,double *im){
    for(size_t s=0;s<64;s++) STR_B(re+s*K2,im+s*K2,NULL,NULL,64,64);
    for(size_t s=0;s<64;s++) vfft_proto_execute_bwd(P1,re+s*K2,im+s*K2,64);
    vfft_proto_execute_bwd(P0,re,im,K2);
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
    fwd_prod(z,z1,cr,ci);
    bwd_prod(ILI_D,z1,z2,cr,ci); { double *zb=aligned_alloc(64,n*16);
    bwd_prod(ILI_E,z1,zb,cr,ci);
    printf("bwd ILI emitted vs derived : %s\n", memcmp(z2,zb,n*16)?"**FAIL**":"BIT-EXACT"); free(zb); }
    double mx=0; srand(3);
    for(size_t i=0;i<2*n;i++){ double x0=2.0*rand()/RAND_MAX-1; double d=fabs(z2[i]/(double)NTOT-x0); if(d>mx)mx=d; }
    printf("roundtrip max err (scaled) : %.2e %s\n", mx, mx<1e-12?"OK":"**FAIL**");
    int reps=6; double t_fl=1e18,t_fp=1e18,t_bfl=1e18,t_bd=1e18,t_be=1e18;
    for(int t=0;t<7;t++){
        double t0,v;
        t0=(double)__rdtsc(); for(int r=0;r<reps;r++) fwd_floor(re,im);
        v=((double)__rdtsc()-t0)/reps; if(v<t_fl)t_fl=v;
        t0=(double)__rdtsc(); for(int r=0;r<reps;r++) fwd_prod(z,z1,cr,ci);
        v=((double)__rdtsc()-t0)/reps; if(v<t_fp)t_fp=v;
        t0=(double)__rdtsc(); for(int r=0;r<reps;r++) bwd_floor(re,im);
        v=((double)__rdtsc()-t0)/reps; if(v<t_bfl)t_bfl=v;
        t0=(double)__rdtsc(); for(int r=0;r<reps;r++) bwd_prod(ILI_D,z,z1,cr,ci);
        v=((double)__rdtsc()-t0)/reps; if(v<t_bd)t_bd=v;
        t0=(double)__rdtsc(); for(int r=0;r<reps;r++) bwd_prod(ILI_E,z,z1,cr,ci);
        v=((double)__rdtsc()-t0)/reps; if(v<t_be)t_be=v;
    }
    printf("FWD floor %9.0f | PROD %.3fx\n", t_fl, t_fp/t_fl);
    printf("BWD floor %9.0f | fused+drvILI %.3fx | fused+emitILI %.3fx\n", t_bfl, t_bd/t_bfl, t_be/t_bfl);
    return 0;
}
