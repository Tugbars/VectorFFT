/* bench_r2c_modelb_ab.c — the doc-58 "needs metal" go/no-go.
 *
 * Stride r2c forward, N=256, inner=(16,8) (last radix 8 = the model-b r8 codelet,
 * and a guard-whitelisted shape). A/B:
 *   DEFAULT : d->ls_fwd = NULL  -> the streamlined postprocess path.
 *   FUSED   : d->ls_fwd = radix256_r2c_term_ls_r8_fwd_avx2 (BLOCKED two-pass)
 *             -> the fused last stage, deletes the scratch round-trip.
 * Correctness: max|fused - default| (expect ~few-ulp, the fusion reschedules).
 * Speed: default vs fused vs MKL DFTI_REAL, single thread (MKL pinned), min-of-N.
 *
 * The container could not see this (L3≈DRAM, no prefetcher). This box can.
 * Usage: bench_r2c_modelb_ab [K=64]
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mkl_dfti.h>
#include <mkl_service.h>

#include "executor.h"
#include "planner.h"
#include "dp_planner.h"   /* vfft_proto_now_ns */
#include "proto_stride_compat.h"  /* threads pool + STRIDE_ALIGNED_ALLOC, before r2c.h */
#include "r2c.h"          /* stride_r2c_plan, stride_execute_r2c, stride_r2c_data_t */
#include "generator/generated/registry.h"

/* the blocked (doc-58 two-pass) fused last-stage codelet (r=8, m=16) */
extern void radix256_r2c_term_ls_r8_fwd_avx2(
    const double*, const double*, const double*, const double*,
    double*, double*, double*, double*,
    const double*, const double*, ptrdiff_t, ptrdiff_t, ptrdiff_t, size_t);

static double *alloc_d(size_t n){ double*p=NULL;
    if (vfft_proto_posix_memalign((void**)&p,64,n*sizeof(double))!=0){fprintf(stderr,"alloc\n");exit(1);} return p; }

static int reps_for(size_t total){ int r=(int)(4e6/(total+1)); if(r<20)r=20; if(r>200000)r=200000; return r; }

/* min-of-trials ns for the stride r2c forward (ls_fwd already set or not). */
static double bench_r2c(const stride_plan_t*p,const double*in,double*re,double*im,size_t total){
    for(int w=0;w<10;w++) stride_execute_r2c(p,in,re,im);
    int reps=reps_for(total); double best=1e18;
    for(int t=0;t<7;t++){ double t0=vfft_proto_now_ns();
        for(int i=0;i<reps;i++) stride_execute_r2c(p,in,re,im);
        double ns=(vfft_proto_now_ns()-t0)/reps; if(ns<best)best=ns; }
    return best;
}

int main(int argc,char**argv){
    mkl_set_num_threads(1);
    size_t K = (argc>1)?(size_t)atoi(argv[1]):64;
    int N=256, halfN=128;
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);

    /* inner c2c (16,8): last radix 8 (matches r8 codelet), guard-whitelisted. */
    int f[2]={16,8}, v[2]={2,2};
    stride_plan_t *inner = vfft_proto_plan_create_ex(halfN,K,f,v,2,/*use_dif=*/0,&reg);
    if(!inner){printf("inner plan NULL\n");return 1;}
    stride_plan_t *p = stride_r2c_plan(N,K,K,inner);
    if(!p){printf("stride_r2c_plan NULL (guard?)\n");return 1;}
    stride_r2c_data_t *d=(stride_r2c_data_t*)p->override_data;

    size_t total=(size_t)N*K;
    double *in=alloc_d(total), *re_def=alloc_d(total), *im_def=alloc_d(total),
           *re_fus=alloc_d(total), *im_fus=alloc_d(total);
    srand(7); for(size_t i=0;i<total;i++) in[i]=(double)rand()/RAND_MAX*2-1;

    /* DEFAULT (ls_fwd NULL by calloc) */
    d->ls_fwd=NULL;
    memset(re_def,0,total*8); memset(im_def,0,total*8);
    stride_execute_r2c(p,in,re_def,im_def);
    double def_ns=bench_r2c(p,in,re_def,im_def,total);

    /* FUSED (model-b: blocked two-pass r8 codelet AS the last stage) */
    d->ls_fwd=radix256_r2c_term_ls_r8_fwd_avx2; d->term_r=8; d->term_m=16;
    memset(re_fus,0,total*8); memset(im_fus,0,total*8);
    stride_execute_r2c(p,in,re_fus,im_fus);
    double fus_ns=bench_r2c(p,in,re_fus,im_fus,total);

    /* correctness: fused vs default over the halfN+1 spectrum */
    double maxd=0;
    for(size_t t=0;t<K;t++) for(int kf=0;kf<=halfN;kf++){
        size_t o=(size_t)kf*K+t;
        double dr=fabs(re_fus[o]-re_def[o]), di=fabs(im_fus[o]-im_def[o]);
        if(dr>maxd)maxd=dr; if(di>maxd)maxd=di;
    }

    /* MKL r2c forward (DFTI_REAL, transform-major) */
    DFTI_DESCRIPTOR_HANDLE h=0; double mkl_ns=0;
    DftiCreateDescriptor(&h,DFTI_DOUBLE,DFTI_REAL,1,(MKL_LONG)N);
    DftiSetValue(h,DFTI_NUMBER_OF_TRANSFORMS,(MKL_LONG)K);
    DftiSetValue(h,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
    DftiSetValue(h,DFTI_CONJUGATE_EVEN_STORAGE,DFTI_COMPLEX_COMPLEX);
    DftiSetValue(h,DFTI_INPUT_DISTANCE,(MKL_LONG)N);
    DftiSetValue(h,DFTI_OUTPUT_DISTANCE,(MKL_LONG)(halfN+1));
    if(DftiCommitDescriptor(h)==DFTI_NO_ERROR){
        double *xin=alloc_d(total), *cce=alloc_d((size_t)(halfN+1)*K*2);
        for(size_t t=0;t<K;t++) for(int n=0;n<N;n++) xin[t*N+n]=in[(size_t)n*K+t];
        for(int w=0;w<10;w++) DftiComputeForward(h,xin,cce);
        int reps=reps_for(total); double best=1e18;
        for(int t=0;t<7;t++){ double t0=vfft_proto_now_ns();
            for(int i=0;i<reps;i++) DftiComputeForward(h,xin,cce);
            double ns=(vfft_proto_now_ns()-t0)/reps; if(ns<best)best=ns; }
        mkl_ns=best;
    }
    DftiFreeDescriptor(&h);

    printf("=== model-b A/B (stride r2c, N=256 inner=(16,8), K=%zu, ST) ===\n",K);
    printf("  default : %10.1f ns   (mkl/def = %.3f)\n", def_ns, mkl_ns/def_ns);
    printf("  fused   : %10.1f ns   (mkl/fus = %.3f)\n", fus_ns, mkl_ns/fus_ns);
    printf("  mkl     : %10.1f ns\n", mkl_ns);
    printf("  FUSED vs DEFAULT speedup: %.3fx   (>1 => fusion faster on metal)\n", def_ns/fus_ns);
    printf("  fused-vs-default max abs diff: %.2e  (%s)\n", maxd, maxd<1e-9?"CORRECT":"WRONG");
    return 0;
}
