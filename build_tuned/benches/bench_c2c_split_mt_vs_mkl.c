/* bench_c2c_split_mt_vs_mkl.c — LAYOUT-CONTROLLED c2c MT: dag vs MKL, BOTH in
 * split-complex (separate re/im) lane-batched layout. Removes layout as a variable.
 *
 * MKL DFTI supports split storage for COMPLEX transforms via
 *   DFTI_COMPLEX_STORAGE = DFTI_REAL_REAL  (separate re/im arrays).
 * We give MKL the SAME data[n*K+lane] lane-batched split buffers dag uses:
 *   strides = {0, K}, number_of_transforms = K, distance = 1.
 * Both run in-place, both at 1 and 8 threads (dag: pool K-split over
 * vfft_proto_execute_fwd; MKL: mkl_set_num_threads). Caller pinned core 0.
 *
 * Correctness: roundtrip fwd+bwd == N*x per engine (dag is zero-perm DIT/DIF, so
 * fwd+bwd is natural even though fwd output is digit-reversed); plus dag MT==T1.
 * We do NOT compare dag-vs-MKL output element-wise (dag fwd is scrambled order).
 *
 * Build: cd build_tuned && python build.py --src benches/bench_c2c_split_mt_vs_mkl.c --mkl --compile
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mkl_dfti.h>
#include <mkl_service.h>

#include "core/executor.h"     /* vfft_proto_execute_fwd/bwd (in-place, per-slice) */
#include "core/planner.h"
#include "core/threads.h"
#include "core/env.h"
#include "generator/generated/registry.h"

#define BEST_OF 13
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
static int reps_for(size_t t){int r=(int)(2e8/(t+1)); if(r<3)r=3; if(r>2000)r=2000; return r;}
static double maxdiff(const double*a,const double*b,size_t n){double e=0;for(size_t i=0;i<n;i++){double d=fabs(a[i]-b[i]);if(d>e)e=d;}return e;}

/* ---- dag in-place c2c, pool K-split (same mechanism as bench_c2c_mt.c) ---- */
typedef struct { const stride_plan_t*p; double*re,*im; size_t k0,S; int dir; } ip_arg_t;
static void ip_tramp(void*a){ip_arg_t*x=(ip_arg_t*)a;
    if(x->dir) vfft_proto_execute_fwd(x->p,x->re+x->k0,x->im+x->k0,x->S);
    else       vfft_proto_execute_bwd(x->p,x->re+x->k0,x->im+x->k0,x->S);}
static void dag_mt(const stride_plan_t*p,double*re,double*im,int dir){
    size_t K=p->K; int T=stride_get_num_threads();
    if(T>_stride_pool_size+1)T=_stride_pool_size+1;
    if(T<=1||K<8){ if(dir)vfft_proto_execute_fwd(p,re,im,K); else vfft_proto_execute_bwd(p,re,im,K); return; }
    size_t S=((K/(size_t)T)+7)&~(size_t)7; ip_arg_t a[64]; int nd=0;
    for(int t=1;t<T&&t<=_stride_pool_size;t++){size_t k0=(size_t)t*S;if(k0>=K)break;size_t ke=k0+S;if(ke>K)ke=K;
        a[nd]=(ip_arg_t){p,re,im,k0,ke-k0,dir};_stride_pool_dispatch(&_stride_workers[nd],ip_tramp,&a[nd]);nd++;}
    size_t s0=S<K?S:K; if(dir)vfft_proto_execute_fwd(p,re,im,s0); else vfft_proto_execute_bwd(p,re,im,s0);
    if(nd)_stride_pool_wait_all();
}

/* ---- MKL split (REAL_REAL) batched c2c, in-place ---- */
static DFTI_DESCRIPTOR_HANDLE mkl_split_c2c(int N,size_t K){
    DFTI_DESCRIPTOR_HANDLE h=0;
    if(DftiCreateDescriptor(&h,DFTI_DOUBLE,DFTI_COMPLEX,1,(MKL_LONG)N)!=DFTI_NO_ERROR) return 0;
    MKL_LONG str[2]={0,(MKL_LONG)K};
    DftiSetValue(h,DFTI_COMPLEX_STORAGE,DFTI_REAL_REAL);          /* split re/im */
    DftiSetValue(h,DFTI_NUMBER_OF_TRANSFORMS,(MKL_LONG)K);
    DftiSetValue(h,DFTI_INPUT_DISTANCE,1);
    DftiSetValue(h,DFTI_OUTPUT_DISTANCE,1);
    DftiSetValue(h,DFTI_INPUT_STRIDES,str);
    DftiSetValue(h,DFTI_OUTPUT_STRIDES,str);
    DftiSetValue(h,DFTI_PLACEMENT,DFTI_INPLACE);
    if(DftiCommitDescriptor(h)!=DFTI_NO_ERROR){DftiFreeDescriptor(&h);return 0;}
    return h;
}

static void run(int N,size_t K,vfft_proto_registry_t*reg){
    size_t NK=(size_t)N*K;
    stride_plan_t*p=vfft_proto_auto_plan(N,K,reg,NULL);
    DFTI_DESCRIPTOR_HANDLE h=mkl_split_c2c(N,K);
    if(!p||!h){printf("N=%-5d K=%-5zu  plan/mkl NULL (p=%p h=%p)\n",N,K,(void*)p,(void*)h);return;}
    double ws=(double)NK*16.0/(1024*1024);

    double*re=AALLOC(NK*8),*im=AALLOC(NK*8),*x=AALLOC(NK*8),*xi=AALLOC(NK*8),*ref=AALLOC(NK*8),*refi=AALLOC(NK*8);
    double*mre=AALLOC(NK*8),*mim=AALLOC(NK*8);
    srand(7+N);for(size_t i=0;i<NK;i++){x[i]=(double)rand()/RAND_MAX-0.5;xi[i]=(double)rand()/RAND_MAX-0.5;}

    /* dag roundtrip correctness (T1): fwd+bwd == N*x */
    memcpy(re,x,NK*8);memcpy(im,xi,NK*8);
    stride_set_num_threads(1); dag_mt(p,re,im,1); dag_mt(p,re,im,0);
    double drt=0; for(size_t i=0;i<NK;i++){double a=re[i]/N-x[i];if(fabs(a)>drt)drt=fabs(a);double b=im[i]/N-xi[i];if(fabs(b)>drt)drt=fabs(b);}
    /* dag MT==T1 (forward only) */
    memcpy(re,x,NK*8);memcpy(im,xi,NK*8);stride_set_num_threads(1);dag_mt(p,re,im,1);
    memcpy(ref,re,NK*8);memcpy(refi,im,NK*8);
    memcpy(re,x,NK*8);memcpy(im,xi,NK*8);stride_set_num_threads(8);dag_mt(p,re,im,1);
    double dmt=maxdiff(re,ref,NK); if(maxdiff(im,refi,NK)>dmt)dmt=maxdiff(im,refi,NK);
    /* MKL roundtrip correctness */
    memcpy(mre,x,NK*8);memcpy(mim,xi,NK*8);
    mkl_set_num_threads(1); DftiComputeForward(h,mre,mim); DftiComputeBackward(h,mre,mim);
    double mrt=0; for(size_t i=0;i<NK;i++){double a=mre[i]/N-x[i];if(fabs(a)>mrt)mrt=fabs(a);double b=mim[i]/N-xi[i];if(fabs(b)>mrt)mrt=fabs(b);}

    int Ts[]={1,8}; double dt[2],mt[2];
    for(int ti=0;ti<2;ti++){
        stride_set_num_threads(Ts[ti]);
        double bi=1e18; for(int w=0;w<2;w++){memcpy(re,x,NK*8);memcpy(im,xi,NK*8);dag_mt(p,re,im,1);}
        for(int r=0;r<BEST_OF;r++){memcpy(re,x,NK*8);memcpy(im,xi,NK*8);double t0=now_c();dag_mt(p,re,im,1);double v=now_c()-t0;if(v<bi)bi=v;}
        dt[ti]=bi;
        mkl_set_num_threads(Ts[ti]);
        double bm=1e18; for(int w=0;w<2;w++){memcpy(mre,x,NK*8);memcpy(mim,xi,NK*8);DftiComputeForward(h,mre,mim);}
        for(int r=0;r<BEST_OF;r++){memcpy(mre,x,NK*8);memcpy(mim,xi,NK*8);double t0=now_c();DftiComputeForward(h,mre,mim);double v=now_c()-t0;if(v<bm)bm=v;}
        mt[ti]=bm;
    }
    stride_set_num_threads(1);
    printf("N=%-5d K=%-5zu %7.1fMB  dag[T1 %9.0f T8 %9.0f %4.2fx]  MKL[T1 %9.0f T8 %9.0f %4.2fx]  T1 m/d %5.2fx  T8 m/d %5.2fx  (drt %.0e mrt %.0e dMT %.0e)\n",
           N,K,ws, dt[0],dt[1],dt[0]/dt[1], mt[0],mt[1],mt[0]/mt[1],
           mt[0]/dt[0], mt[1]/dt[1], drt,mrt,dmt);
    fflush(stdout);
    AFREE(re);AFREE(im);AFREE(x);AFREE(xi);AFREE(ref);AFREE(refi);AFREE(mre);AFREE(mim);
    vfft_proto_plan_destroy(p); DftiFreeDescriptor(&h);
}

int main(void){
    stride_env_init();
    if(stride_pin_thread(0)!=0) fprintf(stderr,"warn pin\n");
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    printf("== c2c SPLIT-vs-SPLIT MT: dag vs MKL (DFTI_REAL_REAL), in-place, caller core0 ==\n");
    printf("# layout identical (separate re/im, lane-batched). T1 m/d>1 = dag faster ST; T8 m/d>1 = dag faster @8.\n");
    printf("# cyc = __rdtsc. drt/mrt = roundtrip err; dMT = dag T8==T1 err.\n");
    run(256, 1024,&reg);   /* 4 MB  */
    run(512, 1024,&reg);   /* 8 MB  */
    run(1024,1024,&reg);   /* 16 MB */
    run(1024,2048,&reg);   /* 32 MB */
    run(4096, 512,&reg);   /* 32 MB, larger N */
    return 0;
}
