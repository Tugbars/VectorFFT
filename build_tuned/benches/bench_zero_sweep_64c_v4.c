/* Zero-sweep 64^3 v4 — adds the z-in-place variant (output pairs reuse the
 * input z buffer, legal because axis-0 fully consumes z before rows writes).
 * Footprints: floor 4MB | IL zo-separate 12MB | IL z-inplace 8MB. */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "prime_dispatch.h"
#include "exhaustive_plan.h"
#include "il_layout.h"
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
#define STR_B CAT(radix64_n1_bwd_,CAT(ISA,_strided))
#define STR_F_ILO CAT(radix64_n1_fwd_,CAT(ISA,_strided_il_out))
#define STR_B_ILI CAT(radix64_n1_bwd_,CAT(ISA,_strided_il_in))
void STR_F(double*,double*,const double*,const double*,size_t,size_t);
void STR_B(double*,double*,const double*,const double*,size_t,size_t);
void STR_F_ILO(const double*,const double*,double*,const double*,const double*,size_t,size_t);
void STR_B_ILI(const double*,double*,double*,const double*,const double*,size_t,size_t);
static stride_plan_t *P0,*P1;
static const size_t K2=4096, NTOT=262144;

static void fwd_il(const double *z, double *zo, double *cr, double *ci){
    vfft_proto_execute_fwd_ilin(P0,z,cr,ci,K2);
    for(size_t s=0;s<64;s++) vfft_proto_execute_fwd(P1,cr+s*K2,ci+s*K2,64);
    for(size_t s=0;s<64;s++) STR_F_ILO(cr+s*K2,ci+s*K2,zo+2*s*K2,NULL,NULL,64,64);
}
static void bwd_il(const double *z, double *zo, double *cr, double *ci){
    for(size_t s=0;s<64;s++) STR_B_ILI(z+2*s*K2,cr+s*K2,ci+s*K2,NULL,NULL,64,64);
    for(size_t s=0;s<64;s++) vfft_proto_execute_bwd(P1,cr+s*K2,ci+s*K2,64);
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
    double *z=aligned_alloc(64,n*16),*zo=aligned_alloc(64,n*16),*zi=aligned_alloc(64,n*16);
    double *cr=aligned_alloc(64,n*8),*ci=aligned_alloc(64,n*8);
    double *re=aligned_alloc(64,n*8),*im=aligned_alloc(64,n*8);
    srand(3); for(size_t i=0;i<2*n;i++) z[i]=2.0*rand()/RAND_MAX-1;
    /* gates: aliased output bit-identical to separate output */
    fwd_il(z,zo,cr,ci);
    memcpy(zi,z,n*16); fwd_il(zi,zi,cr,ci);
    printf("fwd z-inplace vs zo-separate : %s\n", memcmp(zi,zo,n*16)?"**FAIL**":"BIT-EXACT");
    bwd_il(zo,z,cr,ci);            /* z now holds bwd(spectrum), scaled */
    memcpy(zi,zo,n*16); bwd_il(zi,zi,cr,ci);
    printf("bwd z-inplace vs zo-separate : %s\n", memcmp(zi,z,n*16)?"**FAIL**":"BIT-EXACT");
    double mx=0; srand(3);
    for(size_t i=0;i<2*n;i++){ double x0=2.0*rand()/RAND_MAX-1; double d=fabs(z[i]/(double)NTOT-x0); if(d>mx)mx=d; }
    printf("roundtrip max err (scaled)   : %.2e %s\n", mx, mx<1e-12?"OK":"**FAIL**");
    /* bench */
    int reps=6; double t_fl=1e18,t_p1a=1e18,t_oop=1e18,t_ipz=1e18,t_bfl=1e18,t_boop=1e18,t_bipz=1e18;
    srand(3); for(size_t i=0;i<2*n;i++) z[i]=2.0*rand()/RAND_MAX-1;
    for(int t=0;t<7;t++){
        double t0=(double)__rdtsc();
        for(int r=0;r<reps;r++) fwd_floor(re,im);
        double v=((double)__rdtsc()-t0)/reps; if(v<t_fl)t_fl=v;
        t0=(double)__rdtsc();
        for(int r=0;r<reps;r++){ vfft_il2sp(z,re,im,n); fwd_floor(re,im); vfft_sp2il(re,im,zo,n); }
        v=((double)__rdtsc()-t0)/reps; if(v<t_p1a)t_p1a=v;
        t0=(double)__rdtsc();
        for(int r=0;r<reps;r++) fwd_il(z,zo,cr,ci);
        v=((double)__rdtsc()-t0)/reps; if(v<t_oop)t_oop=v;
        t0=(double)__rdtsc();
        for(int r=0;r<reps;r++) fwd_il(zi,zi,cr,ci);   /* compounds, same as floor */
        v=((double)__rdtsc()-t0)/reps; if(v<t_ipz)t_ipz=v;
        t0=(double)__rdtsc();
        for(int r=0;r<reps;r++) bwd_floor(re,im);
        v=((double)__rdtsc()-t0)/reps; if(v<t_bfl)t_bfl=v;
        t0=(double)__rdtsc();
        for(int r=0;r<reps;r++) bwd_il(z,zo,cr,ci);
        v=((double)__rdtsc()-t0)/reps; if(v<t_boop)t_boop=v;
        t0=(double)__rdtsc();
        for(int r=0;r<reps;r++) bwd_il(zi,zi,cr,ci);
        v=((double)__rdtsc()-t0)/reps; if(v<t_bipz)t_bipz=v;
    }
    printf("FWD floor %9.0f | P1a %.3fx | IL-oop %.3fx | IL-ipz %.3fx\n",
           t_fl, t_p1a/t_fl, t_oop/t_fl, t_ipz/t_fl);
    printf("BWD floor %9.0f |            IL-oop %.3fx | IL-ipz %.3fx\n",
           t_bfl, t_boop/t_bfl, t_bipz/t_bfl);
    return 0;
}
