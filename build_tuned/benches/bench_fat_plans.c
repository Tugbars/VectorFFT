/* Fat-vs-thin plan race at (256,256): DP pick vs forced [16,16]{T1S,LOG3}
 * vs [64,4], each under BOTH tiers (specialized = handle exec fn, true
 * generic), MKL-ctg anchor. Decides the sweep-through-generic rank-inversion
 * question (il_architecture.md section 6a10). Run on stable hardware. */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>
#include "prime_dispatch.h"
#include "plan_orchestrator.h"
#include "generator/generated/registry.h"
#include "mkl_dfti.h"
#define TIME(dst,stmt) { double _b=1e18; for(int _t=0;_t<9;_t++){ \
    double _0=(double)__rdtsc(); for(int _r=0;_r<REPS;_r++){ stmt; } \
    double _v=((double)__rdtsc()-_0)/REPS; if(_v<_b)_b=_v;} dst=_b; }
static const int N=256; static const size_t K=256; static int REPS=60;
static int plan_forced(vfft_proto_handle_t *h, const char *wline,
                       const vfft_proto_registry_t *reg){
    char p[]="/tmp/wforceXXXXXX"; int fd=mkstemp(p);
    dprintf(fd,"@version 6\n%s\n",wline); close(fd);
    static vfft_proto_wisdom_t w; memset(&w,0,sizeof w);
    if(vfft_proto_wisdom_load(&w,p)!=0){ fprintf(stderr,"wload fail\n"); return -1; }
    return vfft_proto_plan(h,N,K,VFFT_PROTO_WISDOM_ONLY,reg,&w,NULL);
}
static void race(const char *nm, vfft_proto_handle_t *h,
                 double *re, double *im, double base_g, double base_s){
    if(!h->plan){ printf("%-14s plan-fail\n",nm); return; }
    double tg,ts=0;
    TIME(tg, { vfft_proto_execute_fwd_generic(h->plan,re,im,K);
               vfft_proto_execute_bwd_generic(h->plan,re,im,K); });
    if(h->exec_fwd && h->exec_bwd){
        TIME(ts, { h->exec_fwd(h->plan,re,im,K,h->plan->K,0);
                   h->exec_bwd(h->plan,re,im,K,h->plan->K,0); });
    }
    printf("%-14s fac=[",nm);
    for(int s=0;s<h->plan->num_stages;s++) printf("%d%s",h->plan->factors[s],
        s+1<h->plan->num_stages?",":"");
    printf("]  generic %9.0f (%.2fx)  |  spec %9.0f (%.2fx)  [tier gain %.2fx]\n",
        tg, base_g>0?base_g/tg:1.0, ts, (ts>0&&base_s>0)?base_s/ts:1.0,
        ts>0?tg/ts:0.0);
}
int main(void){
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    stride_set_num_threads(1);
    size_t n=(size_t)N*K;
    double *re=aligned_alloc(64,n*8),*im=aligned_alloc(64,n*8),*z=aligned_alloc(64,n*16);
    srand(9); for(size_t i=0;i<n;i++){re[i]=rand()*1e-9;im[i]=rand()*1e-9;}
    vfft_proto_handle_t hdp,h16t,h16l,h64;
    static vfft_proto_wisdom_t wdp; memset(&wdp,0,sizeof wdp);
    vfft_proto_plan(&hdp,N,K,VFFT_PROTO_MEASURE,&reg,&wdp,NULL);
    plan_forced(&h16t,"256 256 2 16 16 0.0 0 0 0 0 0 2 0",&reg);
    plan_forced(&h16l,"256 256 2 16 16 0.0 0 0 0 0 0 1 0",&reg);
    plan_forced(&h64 ,"256 256 2 64 4 0.0 0 0 0 0 0 2 0",&reg);
    DFTI_DESCRIPTOR_HANDLE dc; MKL_LONG st[2]={0,1};
    DftiCreateDescriptor(&dc,DFTI_DOUBLE,DFTI_COMPLEX,1,(MKL_LONG)N);
    DftiSetValue(dc,DFTI_NUMBER_OF_TRANSFORMS,(MKL_LONG)K);
    DftiSetValue(dc,DFTI_INPUT_STRIDES,st); DftiSetValue(dc,DFTI_OUTPUT_STRIDES,st);
    DftiSetValue(dc,DFTI_INPUT_DISTANCE,(MKL_LONG)N);
    DftiSetValue(dc,DFTI_OUTPUT_DISTANCE,(MKL_LONG)N);
    DftiCommitDescriptor(dc);
    double tm; TIME(tm,{DftiComputeForward(dc,z);DftiComputeBackward(dc,z);});
    double bg,bs;
    TIME(bg,{vfft_proto_execute_fwd_generic(hdp.plan,re,im,K);
             vfft_proto_execute_bwd_generic(hdp.plan,re,im,K);});
    TIME(bs,{hdp.exec_fwd(hdp.plan,re,im,K,hdp.plan->K,0);
             hdp.exec_bwd(hdp.plan,re,im,K,hdp.plan->K,0);});
    printf("MKL-ctg rt: %.0f\n", tm);
    race("DP-pick",&hdp,re,im,bg,bs);
    race("[16,16] T1S",&h16t,re,im,bg,bs);
    race("[16,16] LOG3",&h16l,re,im,bg,bs);
    race("[64,4] T1S",&h64,re,im,bg,bs);
    return 0;
}
