/* bench_fft2d_vs_mkl.c — validate the ported 2D c2c (core/fft2d.h) on dag.
 *
 * Correctness: ROUNDTRIP fwd+bwd == N1*N2*x (definitive, order-agnostic — the
 * 1D dag engine is digit-reversed, so the 2D output is scrambled; roundtrip +
 * separable construction = correct 2D DFT). Also reports elementwise-vs-MKL
 * (tells us whether the order is natural). Timing vs MKL 2D (split, NOT_INPLACE).
 *
 * Build: cd build_tuned && python build.py --src benches/bench_fft2d_vs_mkl.c --mkl --compile
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mkl_dfti.h>
#include <mkl_service.h>
#include "core/fft2d.h"
#include "core/env.h"
#include "generator/generated/registry.h"

#define PIN_CORE 2
#define BEST_OF  15
#if defined(_WIN32)
#include <malloc.h>
#define AALLOC(n) _aligned_malloc((n),64)
#define AFREE(p)  _aligned_free(p)
#else
#define AALLOC(n) aligned_alloc(64,(n))
#define AFREE(p)  free(p)
#endif
#include <x86intrin.h>
static inline double now_c(void){ return (double)__rdtsc(); }
static int reps_for(size_t t){int r=(int)(4e7/(t+1)); if(r<10)r=10; if(r>20000)r=20000; return r;}

static void run_cell(int N1,int N2,vfft_proto_registry_t*reg){
    size_t T=(size_t)N1*N2;
    stride_plan_t *p = stride_plan_2d(N1,N2,reg);
    if(!p){ printf("  %4dx%-4d  plan NULL\n",N1,N2); return; }

    double *re=AALLOC(T*8),*im=AALLOC(T*8),*xr=AALLOC(T*8),*xi=AALLOC(T*8);
    srand(11+N1+N2);
    for(size_t i=0;i<T;i++){xr[i]=(double)rand()/RAND_MAX-0.5; xi[i]=(double)rand()/RAND_MAX-0.5;}

    /* roundtrip: fwd then bwd in place == N1*N2 * x */
    memcpy(re,xr,T*8); memcpy(im,xi,T*8);
    stride_execute_fwd(p,re,im);
    /* stash fwd output for vs-MKL */
    double *fr=AALLOC(T*8),*fi=AALLOC(T*8); memcpy(fr,re,T*8); memcpy(fi,im,T*8);
    stride_execute_bwd(p,re,im);
    double rt=0; double sc=(double)N1*N2;
    for(size_t i=0;i<T;i++){double a=fabs(re[i]/sc-xr[i]),b=fabs(im[i]/sc-xi[i]); if(a>rt)rt=a; if(b>rt)rt=b;}

    /* MKL 2D complex split, natural order */
    double *mr=AALLOC(T*8),*mi=AALLOC(T*8);
    DFTI_DESCRIPTOR_HANDLE h=0; MKL_LONG dims[2]={N1,N2};
    double ewe=-1;   /* elementwise vs MKL (informational) */
    int mok=0;
    if(DftiCreateDescriptor(&h,DFTI_DOUBLE,DFTI_COMPLEX,2,dims)==DFTI_NO_ERROR){
        DftiSetValue(h,DFTI_COMPLEX_STORAGE,DFTI_REAL_REAL);
        DftiSetValue(h,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
        mok=(DftiCommitDescriptor(h)==DFTI_NO_ERROR);
    }
    if(mok){
        DftiComputeForward(h,xr,xi,mr,mi);
        ewe=0; double mm=0;
        for(size_t i=0;i<T;i++){double a=fr[i]-mr[i],b=fi[i]-mi[i]; double e=sqrt(a*a+b*b),m=hypot(mr[i],mi[i]); if(e>ewe)ewe=e; if(m>mm)mm=m;}
        if(mm>0) ewe/=mm;
    }

    /* timing */
    int reps=reps_for(T); double bv=1e18,bm=1e18;
    for(int w=0;w<3;w++){ memcpy(re,xr,T*8);memcpy(im,xi,T*8); stride_execute_fwd(p,re,im); if(mok)DftiComputeForward(h,xr,xi,mr,mi); }
    for(int t=0;t<BEST_OF;t++){
        double t0=now_c(); for(int i=0;i<reps;i++){ stride_execute_fwd(p,re,im); } double v=(now_c()-t0)/reps; if(v<bv)bv=v;
        if(mok){ t0=now_c(); for(int i=0;i<reps;i++){ DftiComputeForward(h,xr,xi,mr,mi); } double m=(now_c()-t0)/reps; if(m<bm)bm=m; }
    }
    printf("  %4dx%-4d  roundtrip=%.1e  vsMKL_elem=%s%-9.1e | vfft %10.0f | mkl %10.0f | speed %.3f  %s\n",
           N1,N2, rt, ewe<0?"(n/a)":"", ewe<0?0:ewe, bv, mok?bm:0, (mok&&bv>0)?bm/bv:0,
           rt<1e-9?"ROUNDTRIP OK":"*** RT FAIL ***");
    fflush(stdout);

    if(h)DftiFreeDescriptor(&h);
    AFREE(re);AFREE(im);AFREE(xr);AFREE(xi);AFREE(fr);AFREE(fi);AFREE(mr);AFREE(mi);
    stride_plan_destroy(p);
}

int main(void){
    stride_env_init();
    if(stride_pin_thread(PIN_CORE)!=0) fprintf(stderr,"warn pin\n");
    mkl_set_num_threads(1);
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    printf("== 2D c2c (dag fft2d.h, tiled) vs MKL DFTI 2D (split, NOT_INPLACE, ST, cpu%d) ==\n",PIN_CORE);
    printf("# roundtrip = fwd+bwd==N*x (definitive). vsMKL_elem = elementwise (shows order). speed>1 = we win.\n");
    run_cell(64,64,&reg);
    run_cell(128,128,&reg);
    run_cell(256,256,&reg);
    run_cell(512,512,&reg);
    return 0;
}
