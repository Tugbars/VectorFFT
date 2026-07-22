/* Zero-sweep 64^3 IL transform: derived codelets at both boundaries.
 * fwd: n1_oop_il_in (z->cube axis0) + engine axis1 + strided_il_out (rows->z)
 * bwd: strided_il_in (z->cube rows) + engine axis1 bwd + n1_oop_il_out_sw (cube->z, swapped ins)
 * Gates: bit vs (il2sp + split 3-pass + sp2il).  Bench vs split floor + P1a. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "prime_dispatch.h"
#include "il_layout.h"
#include "generator/generated/registry.h"
#include <x86intrin.h>
#ifdef USE512
#define ISA avx512
#else
#define ISA avx2
#endif
#define CAT2(a,b) a##b
#define CAT(a,b) CAT2(a,b)
#define OOP_ORIG CAT(radix64_n1_oop_fwd_,CAT(ISA,_UG_UG))
#define OOP_ILIN CAT(radix64_n1_oop_fwd_,CAT(ISA,_UG_UG_il_in))
#define OOP_ILOSW CAT(radix64_n1_oop_fwd_,CAT(ISA,_UG_UG_il_out_sw))
#define STR_F CAT(radix64_n1_fwd_,CAT(ISA,_strided))
#define STR_B CAT(radix64_n1_bwd_,CAT(ISA,_strided))
#define STR_F_ILO CAT(radix64_n1_fwd_,CAT(ISA,_strided_il_out))
#define STR_B_ILI CAT(radix64_n1_bwd_,CAT(ISA,_strided_il_in))
void OOP_ORIG(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void OOP_ILIN(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void OOP_ILOSW(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void STR_F(double*,double*,const double*,const double*,size_t,size_t);
void STR_B(double*,double*,const double*,const double*,size_t,size_t);
void STR_F_ILO(const double*,const double*,double*,const double*,const double*,size_t,size_t);
void STR_B_ILI(const double*,double*,double*,const double*,const double*,size_t,size_t);

static stride_plan_t *P1;   /* axis-1 plan (64,64) */
static const size_t N=64, K2=4096, NTOT=262144;

static void fwd_il(double *z, double *cr, double *ci){
    OOP_ILIN(z,NULL,cr,ci,NULL,NULL,K2,1,K2,1,K2);
    for(size_t s=0;s<N;s++) vfft_proto_execute_fwd(P1,cr+s*K2,ci+s*K2,64);
    for(size_t s=0;s<N;s++) STR_F_ILO(cr+s*K2,ci+s*K2,z+2*s*K2,NULL,NULL,64,64);
}
static void bwd_il(double *z, double *cr, double *ci){
    for(size_t s=0;s<N;s++) STR_B_ILI(z+2*s*K2,cr+s*K2,ci+s*K2,NULL,NULL,64,64);
    for(size_t s=0;s<N;s++) vfft_proto_execute_bwd(P1,cr+s*K2,ci+s*K2,64);
    OOP_ILOSW(ci,cr,z,NULL,NULL,NULL,K2,1,K2,1,K2);
}
static void fwd_split(double *re,double *im,double *cr,double *ci){
    OOP_ORIG(re,im,cr,ci,NULL,NULL,K2,1,K2,1,K2);
    for(size_t s=0;s<N;s++) vfft_proto_execute_fwd(P1,cr+s*K2,ci+s*K2,64);
    for(size_t s=0;s<N;s++) STR_F(cr+s*K2,ci+s*K2,NULL,NULL,64,64);
}
int main(void){
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    stride_set_num_threads(1);
    P1 = vfft_proto_auto_plan_dispatch(64,64,&reg,NULL);
    size_t n=NTOT;
    double *z=aligned_alloc(64,n*16),*z0=aligned_alloc(64,n*16),*zr=aligned_alloc(64,n*16);
    double *cr=aligned_alloc(64,n*8),*ci=aligned_alloc(64,n*8);
    double *re=aligned_alloc(64,n*8),*im=aligned_alloc(64,n*8);
    srand(3); for(size_t i=0;i<n;i++){z[2*i]=2.0*rand()/RAND_MAX-1;z[2*i+1]=2.0*rand()/RAND_MAX-1;}
    memcpy(z0,z,n*16);
    /* ---- gate fwd: IL path vs reference ---- */
    fwd_il(z,cr,ci);
    vfft_il2sp(z0,re,im,n);
    fwd_split(re,im,cr,ci);      /* cube holds split spectrum */
    vfft_sp2il(cr,ci,zr,n);
    int okf = !memcmp(z,zr,n*16);
    printf("fwd zero-sweep vs reference : %s\n", okf?"BIT-EXACT":"**FAIL**");
    /* ---- gate roundtrip: bwd(fwd(z0)) == z0 * NTOT ---- */
    bwd_il(z,cr,ci);
    double mx=0; for(size_t i=0;i<2*n;i++){double d=fabs(z[i]/ (double)NTOT - z0[i]); if(d>mx)mx=d;}
    printf("roundtrip max err (scaled)  : %.2e %s\n", mx, mx<1e-12?"OK":"**FAIL**");
    if(!okf||mx>=1e-12) return 1;
    /* ---- bench ---- */
    int reps=6; double t_split=1e18,t_p1a=1e18,t_il=1e18;
    for(int t=0;t<7;t++){
        double t0=(double)__rdtsc();
        for(int r=0;r<reps;r++) fwd_split(re,im,cr,ci);
        double v=((double)__rdtsc()-t0)/reps; if(v<t_split)t_split=v;
        t0=(double)__rdtsc();
        for(int r=0;r<reps;r++){ vfft_il2sp(z0,re,im,n); fwd_split(re,im,cr,ci); vfft_sp2il(cr,ci,zr,n); }
        v=((double)__rdtsc()-t0)/reps; if(v<t_p1a)t_p1a=v;
        t0=(double)__rdtsc();
        for(int r=0;r<reps;r++) fwd_il(z,cr,ci);
        v=((double)__rdtsc()-t0)/reps; if(v<t_il)t_il=v;
    }
    printf("split 3-pass floor : %10.0f cyc   1.000x\n", t_split);
    printf("P1a sweeps         : %10.0f cyc   %.3fx\n", t_p1a, t_p1a/t_split);
    printf("ZERO-SWEEP (P2)    : %10.0f cyc   %.3fx\n", t_il,  t_il/t_split);
    return 0;
}
