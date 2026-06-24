/* bench_tailtax.c — how much does the tail cost vs the nearest full-vector K?
 * ONE process (same thermal window): per-transform ns (= ns_per_call / K) for
 * each K, plus the "tail tax" = pt(K) / pt(pow2 anchor), for BOTH vfft and MKL.
 * The question: does our tail inflate per-transform cost more or less than MKL's
 * remainder handling, relative to the clean full-vector K? Plan N=1024 [4,4,4,4,4]
 * T1S, rem-aware HYBRID tail (scalar@rem1, masked@rem>=2).
 *
 * Per engine we sweep the whole K list adjacently (best-of-5 min, cachebust
 * between) so anchor-vs-member ratios are thermal-fair within the engine.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "executor.h"
#include "planner.h"
#include "dp_planner.h"
#include <mkl_dfti.h>
#include <mkl_service.h>

static double *ad(size_t n){ double *p=NULL; if(vfft_proto_posix_memalign((void**)&p,64,n*sizeof(double))!=0){fprintf(stderr,"alloc\n");exit(1);} return p; }
static void afree(double *p){ vfft_proto_aligned_free(p); }
static void cachebust(void){ size_t s=32*1024*1024/sizeof(double); double *j=ad(s); for(size_t i=0;i<s;i++)j[i]=(double)i; volatile double a=0; for(size_t i=0;i<s;i++)a+=j[i]; (void)a; afree(j); }

static double best_proto(stride_plan_t *p,double *re,double *im,size_t K,int reps){
    for(int w=0;w<10;w++) vfft_proto_execute_fwd(p,re,im,K);
    double b=1e18; for(int t=0;t<5;t++){ double t0=vfft_proto_now_ns(); for(int i=0;i<reps;i++) vfft_proto_execute_fwd(p,re,im,K); double ns=(vfft_proto_now_ns()-t0)/reps; if(ns<b)b=ns; } return b;
}
static double best_mkl(DFTI_DESCRIPTOR_HANDLE d,double *re,double *im,int reps){
    for(int w=0;w<10;w++) DftiComputeForward(d,re,im);
    double b=1e18; for(int t=0;t<5;t++){ double t0=vfft_proto_now_ns(); for(int i=0;i<reps;i++) DftiComputeForward(d,re,im); double ns=(vfft_proto_now_ns()-t0)/reps; if(ns<b)b=ns; } return b;
}
static DFTI_DESCRIPTOR_HANDLE mkl_make(int N,size_t K){
    DFTI_DESCRIPTOR_HANDLE d=NULL; MKL_LONG str[2]={0,(MKL_LONG)K};
    DftiCreateDescriptor(&d,DFTI_DOUBLE,DFTI_COMPLEX,1,(MKL_LONG)N);
    DftiSetValue(d,DFTI_COMPLEX_STORAGE,DFTI_REAL_REAL); DftiSetValue(d,DFTI_PLACEMENT,DFTI_INPLACE);
    DftiSetValue(d,DFTI_NUMBER_OF_TRANSFORMS,(MKL_LONG)K);
    DftiSetValue(d,DFTI_INPUT_DISTANCE,1); DftiSetValue(d,DFTI_OUTPUT_DISTANCE,1);
    DftiSetValue(d,DFTI_INPUT_STRIDES,str); DftiSetValue(d,DFTI_OUTPUT_STRIDES,str);
    DftiCommitDescriptor(d); return d;
}

int main(void){
    setvbuf(stdout,NULL,_IONBF,0);
    mkl_set_num_threads(1);
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    int N=1024, nf=5, factors[5]={4,4,4,4,4}, variants[5]={0,2,2,2,2};
    /* families: anchor (pow2) then members. tax = pt(K)/pt(anchor). */
    size_t Ks[]   = { 8, 7, 9, 16,15,17,23, 32,31,33, 64,63,65 };
    size_t anch[] = { 8, 8, 8, 16,16,16,16, 32,32,32, 64,64,64 };
    int n = sizeof(Ks)/sizeof(Ks[0]);
    double vpt[16], mpt[16];

    for(int i=0;i<n;i++){                         /* vfft sweep */
        size_t K=Ks[i], total=(size_t)N*K; int reps=(int)(60000000ull/total); if(reps<300)reps=300;
        stride_plan_t *p=vfft_proto_plan_create(N,K,factors,variants,nf,&reg);
        double *re=ad(total),*im=ad(total);
        for(size_t j=0;j<total;j++){ re[j]=(double)rand()/RAND_MAX-0.5; im[j]=(double)rand()/RAND_MAX-0.5; }
        cachebust(); vpt[i]=best_proto(p,re,im,K,reps)/(double)K; afree(re);afree(im);
    }
    for(int i=0;i<n;i++){                          /* mkl sweep */
        size_t K=Ks[i], total=(size_t)N*K; int reps=(int)(60000000ull/total); if(reps<300)reps=300;
        DFTI_DESCRIPTOR_HANDLE d=mkl_make(N,K);
        double *re=ad(total),*im=ad(total);
        for(size_t j=0;j<total;j++){ re[j]=(double)rand()/RAND_MAX-0.5; im[j]=(double)rand()/RAND_MAX-0.5; }
        cachebust(); mpt[i]=best_mkl(d,re,im,reps)/(double)K; DftiFreeDescriptor(&d); afree(re);afree(im);
    }

    printf("=== N=1024 per-TRANSFORM ns + tail-tax vs nearest pow2 (hybrid tail, 1 thr) ===\n");
    printf("%-5s %-4s %10s %10s %8s %10s %10s\n","K","rem","vfft_pt","mkl_pt","margin","vfft_tax","mkl_tax");
    for(int i=0;i<n;i++){
        double va=0,ma=0; for(int j=0;j<n;j++) if(Ks[j]==anch[i]){ va=vpt[j]; ma=mpt[j]; }
        printf("%-5zu %-4zu %10.2f %10.2f %7.2fx %9.3fx %9.3fx%s\n",
               Ks[i], Ks[i]%4, vpt[i], mpt[i], mpt[i]/vpt[i], vpt[i]/va, mpt[i]/ma,
               (Ks[i]==anch[i])?"  <- anchor":"");
    }
    return 0;
}
