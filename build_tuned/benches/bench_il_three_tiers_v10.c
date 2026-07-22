/* v10: three executor tiers on the 64^3 chain plans, AVX2 (JIT tier is avx2).
 * generic (NULL) / baked (wisdom static) / JIT (forced compile_load). */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <x86intrin.h>
#include "prime_dispatch.h"
#include "exhaustive_plan.h"
#include "il_layout.h"
#include "il_execute.h"
#include "jit_runtime.h"
#include "generator/generated/registry.h"
static stride_plan_t *P0,*P1;
static const size_t K2=4096, NTOT=262144;
#define TIME(dst,stmt) { double _b=1e18; for(int _t=0;_t<7;_t++){ double _0=(double)__rdtsc(); for(int _r=0;_r<6;_r++){ stmt; } double _v=((double)__rdtsc()-_0)/6; if(_v<_b)_b=_v;} dst=_b; }
static double maxrel(const double*a,const double*b,size_t n){
    double mx=0,sc=0;
    for(size_t i=0;i<n;i++){double d=fabs(a[i]-b[i]);if(d>mx)mx=d;double v=fabs(a[i]);if(v>sc)sc=v;}
    return sc>0?mx/sc:mx;
}
int main(void){
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    stride_set_num_threads(1);
    P0=vfft_proto_exhaustive_plan(64,K2,&reg,0);
    P1=vfft_proto_exhaustive_plan(64,64,&reg,0);
    size_t n=NTOT;
    double *z=aligned_alloc(64,n*16),*z0=aligned_alloc(64,n*16),*z1=aligned_alloc(64,n*16);
    double *r0=aligned_alloc(64,n*8),*i0=aligned_alloc(64,n*8);
    double *r1=aligned_alloc(64,n*8),*i1=aligned_alloc(64,n*8);
    srand(3); for(size_t i=0;i<2*n;i++) z[i]=2.0*rand()/RAND_MAX-1;
    vfft_proto_exec_fn bkF=_vfft_proto_lookup_fwd(P0), bkB=_vfft_proto_lookup_bwd(P0);
    char kf[256],kb[256];
    vfft_proto_jit_key(P0,"avx2","fwd",kf,sizeof kf);
    vfft_proto_jit_key(P0,"avx2","bwd",kb,sizeof kb);
    vfft_proto_exec_fn jF=vfft_proto_jit_compile_load(P0,"avx2","fwd",kf);
    vfft_proto_exec_fn jB=vfft_proto_jit_compile_load(P0,"avx2","bwd",kb);
    printf("tiers P0: baked %s/%s | jit %s/%s\n",
        bkF?"ok":"NULL",bkB?"ok":"NULL",jF?"ok":"NULL",jB?"ok":"NULL");
    if(!jF||!jB){ printf("JIT unavailable (rsp/build pending?)\n"); return 2; }
    /* gates: jit vs baked through the adapters */
    vfft_proto_execute_fwd_ilin_jit(P0,z,r0,i0,K2,bkF);
    vfft_proto_execute_fwd_ilin_jit(P0,z,r1,i1,K2,jF);
    double g1a=maxrel(r0,r1,n), g1b=maxrel(i0,i1,n);
    printf("fwd resume jit vs baked   : maxrel %.2e %s\n",
        g1a>g1b?g1a:g1b, (g1a<1e-12&&g1b<1e-12)?"OK":"**FAIL**");
    memcpy(r0,r1,n*8); memcpy(i0,i1,n*8);
    vfft_proto_execute_bwd_ilout_jit(P0,r0,i0,z0,K2,bkB);
    vfft_proto_execute_bwd_ilout_jit(P0,r1,i1,z1,K2,jB);
    double g2=maxrel(z0,z1,2*n);
    printf("bwd resume jit vs baked   : maxrel %.2e %s\n", g2, g2<1e-12?"OK":"**FAIL**");
    /* axis-0 resume timings, three tiers each direction */
    double fg,fb,fj,bg,bb,bj;
    TIME(fg, vfft_proto_execute_fwd_ilin_jit(P0,z,r0,i0,K2,NULL));
    TIME(fb, vfft_proto_execute_fwd_ilin_jit(P0,z,r0,i0,K2,bkF));
    TIME(fj, vfft_proto_execute_fwd_ilin_jit(P0,z,r0,i0,K2,jF));
    TIME(bg, vfft_proto_execute_bwd_ilout_jit(P0,r1,i1,z1,K2,NULL));
    TIME(bb, vfft_proto_execute_bwd_ilout_jit(P0,r1,i1,z1,K2,bkB));
    TIME(bj, vfft_proto_execute_bwd_ilout_jit(P0,r1,i1,z1,K2,jB));
    printf("AXIS0 fwd  gen %9.0f | baked %.3fx | jit %.3fx\n",fg,fb/fg,fj/fg);
    printf("AXIS0 bwd  gen %9.0f | baked %.3fx | jit %.3fx\n",bg,bb/bg,bj/bg);
    /* whole-plan P1 (64,64): the general claim, one-shot start_stage=0 */
    vfft_proto_exec_fn p1b=_vfft_proto_lookup_fwd(P1);
    char k1[256]; vfft_proto_jit_key(P1,"avx2","fwd",k1,sizeof k1);
    vfft_proto_exec_fn p1j=vfft_proto_jit_compile_load(P1,"avx2","fwd",k1);
    if(p1b&&p1j){
        double s1g,s1b,s1j;
        TIME(s1g, for(size_t s=0;s<64;s++) vfft_proto_execute_fwd_generic(P1,r0+s*4096,i0+s*4096,64));
        TIME(s1b, for(size_t s=0;s<64;s++) p1b(P1,r0+s*4096,i0+s*4096,64,64,0));
        TIME(s1j, for(size_t s=0;s<64;s++) p1j(P1,r0+s*4096,i0+s*4096,64,64,0));
        printf("P1 x64     gen %9.0f | baked %.3fx | jit %.3fx\n",s1g,s1b/s1g,s1j/s1g);
    }
    return 0;
}
