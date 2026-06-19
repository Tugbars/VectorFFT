/* bench_r2c_mt_vs_mkl.c — MULTITHREADED r2c: dag (8 P-cores, caller-pinned core 0)
 * vs MKL (mkl_set_num_threads). Both compute K real->complex transforms of len N.
 *
 * dag: split + lane-batched (data[n*K+lane]); MT = block-parallel over K. The plan
 *      is built with block_K = K/8 so there are 8 blocks (one per worker) — the
 *      production dispatcher uses block_K=K (1 block = serial), so we build the
 *      inner r2c plan directly here. Plan scratch is snapshotted at CREATE, so we
 *      build at T=8 then vary T at execute.
 * MKL: batched DFTI REAL (CCE complex-complex), contiguous per transform.
 *
 * Caller pinned to core 0 (pool pins workers to cores 1..T-1; all P-cores 0..7).
 *
 * Build: cd build_tuned && python build.py --src benches/bench_r2c_mt_vs_mkl.c --mkl --compile
 * Run  : PATH += MKL bin + C:\mingw152\mingw64\bin.
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mkl_dfti.h>
#include <mkl_service.h>

#include "threads.h"       /* pool: set/get num threads, pin (before r2c.h) */
#include "planner.h"       /* vfft_proto_auto_plan (before compat) */
#include "proto_stride_compat.h"  /* STRIDE_ALIGNED_ALLOC + slice helpers (before r2c.h) */
#include "r2c.h"            /* stride_r2c_plan + stride_execute_r2c (MT inside) */
#include "env.h"
#include "dp_planner.h"    /* vfft_proto_now_ns (wall clock) */
#include "generator/generated/registry.h"

#define BEST_OF 15
static double *alloc_d(size_t n){double*p=NULL;
    if(vfft_proto_posix_memalign((void**)&p,64,n*sizeof(double))!=0){exit(1);} return p;}
static int reps_for(size_t t){int r=(int)(2e7/(t+1)); if(r<5)r=5; if(r>5000)r=5000; return r;}

/* reference DFT for lane 0 (correctness gate) */
static double check(const double*o_re,const double*o_im,const double*x,int N,int halfN,size_t K){
    double me=0;
    for(int k=0;k<=halfN;k++){double rr=0,ri=0;
        for(int n=0;n<N;n++){double xn=x[(size_t)n*K];double a=-2.0*M_PI*k*n/(double)N;rr+=xn*cos(a);ri+=xn*sin(a);}
        double er=fabs(o_re[(size_t)k*K]-rr),ei=fabs(o_im[(size_t)k*K]-ri);
        if(er>me)me=er; if(ei>me)me=ei;}
    return me;
}
static double bench_dag(stride_plan_t*p,const double*x,double*orr,double*oii,size_t total,int T){
    stride_set_num_threads(T);
    for(int w=0;w<5;w++) stride_execute_r2c(p,x,orr,oii);
    int reps=reps_for(total); double best=1e18;
    for(int b=0;b<BEST_OF;b++){double t0=vfft_proto_now_ns();
        for(int i=0;i<reps;i++) stride_execute_r2c(p,x,orr,oii);
        double ns=(vfft_proto_now_ns()-t0)/reps; if(ns<best)best=ns;}
    return best;
}
static double bench_mkl(DFTI_DESCRIPTOR_HANDLE h,const double*xin,double*cce,size_t total){
    for(int w=0;w<5;w++) DftiComputeForward(h,(void*)xin,cce);
    int reps=reps_for(total); double best=1e18;
    for(int b=0;b<BEST_OF;b++){double t0=vfft_proto_now_ns();
        for(int i=0;i<reps;i++) DftiComputeForward(h,(void*)xin,cce);
        double ns=(vfft_proto_now_ns()-t0)/reps; if(ns<best)best=ns;}
    return best;
}
/* commit a batched real DFTI descriptor at the current mkl thread count */
static DFTI_DESCRIPTOR_HANDLE mkl_make(int N,int halfN,size_t K){
    DFTI_DESCRIPTOR_HANDLE h=0;
    DftiCreateDescriptor(&h,DFTI_DOUBLE,DFTI_REAL,1,(MKL_LONG)N);
    DftiSetValue(h,DFTI_NUMBER_OF_TRANSFORMS,(MKL_LONG)K);
    DftiSetValue(h,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
    DftiSetValue(h,DFTI_CONJUGATE_EVEN_STORAGE,DFTI_COMPLEX_COMPLEX);
    DftiSetValue(h,DFTI_INPUT_DISTANCE,(MKL_LONG)N);
    DftiSetValue(h,DFTI_OUTPUT_DISTANCE,(MKL_LONG)(halfN+1));
    if(DftiCommitDescriptor(h)!=DFTI_NO_ERROR){DftiFreeDescriptor(&h);return 0;}
    return h;
}

int main(void){
    stride_env_init();
    if(stride_pin_thread(0)!=0) fprintf(stderr,"warn pin\n");   /* caller = core 0 */
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);

    const int N=256, halfN=N/2;
    const size_t Ks[]={256,512,1024,2048,4096,8192}; const int nK=6;

    printf("=== r2c MT: dag (8 P-cores, core0-pinned) vs MKL (8 threads), N=%d, SPLIT ===\n",N);
    printf("# working set ~= N*K*8 bytes (real in). dag MT = block-parallel over K (block_K=K/8).\n");
    printf("%-6s %9s %10s %10s   %10s %10s   %8s %8s   %9s %9s\n",
           "K","WS(MB)","dagT1_ns","dagT8_ns","mklT1_ns","mklT8_ns",
           "dagScal","mklScal","T1 mkl/dag","T8 mkl/dag");
    printf("-------+---------+----------+----------+----------+----------+--------+--------+---------+---------\n");

    for(int ki=0;ki<nK;ki++){
        size_t K=Ks[ki]; size_t total=(size_t)N*K;
        size_t BK=K/8; if(BK<8)BK=8;            /* 8 blocks => one per P-core */
        double ws=(double)total*8.0/(1024*1024);

        /* dag plan (built at T=8 so scratch has 8 slots) */
        stride_set_num_threads(8);
        stride_plan_t*inner=vfft_proto_auto_plan(N/2,BK,&reg,NULL);
        stride_plan_t*p=inner?stride_r2c_plan(N,K,BK,inner):NULL;
        if(!p){printf("%-6zu dag plan NULL\n",K);continue;}

        double*x=alloc_d(total); srand(7+(int)K);
        for(size_t i=0;i<total;i++)x[i]=(double)rand()/RAND_MAX*2-1;
        double*orr=alloc_d((size_t)(halfN+1)*K),*oii=alloc_d((size_t)(halfN+1)*K);

        /* correctness (lane 0) */
        memset(orr,0,(size_t)(halfN+1)*K*8);memset(oii,0,(size_t)(halfN+1)*K*8);
        stride_set_num_threads(8); stride_execute_r2c(p,x,orr,oii);
        double derr=check(orr,oii,x,N,halfN,K);
        if(derr>1e-9) printf("%-6zu dag CORRECTNESS %.2e\n",K,derr);

        /* MKL input: contiguous per transform (xin[t*N+n]) */
        double*xin=alloc_d(total),*cce=alloc_d((size_t)(halfN+1)*K*2);
        for(size_t t=0;t<K;t++)for(int n=0;n<N;n++)xin[t*N+n]=x[(size_t)n*K+t];

        /* dag timings */
        double d1=bench_dag(p,x,orr,oii,total,1);
        double d8=bench_dag(p,x,orr,oii,total,8);

        /* MKL timings (re-commit per thread count) */
        mkl_set_num_threads(1); DFTI_DESCRIPTOR_HANDLE h1=mkl_make(N,halfN,K);
        double m1=h1?bench_mkl(h1,xin,cce,total):0; if(h1)DftiFreeDescriptor(&h1);
        mkl_set_num_threads(8); DFTI_DESCRIPTOR_HANDLE h8=mkl_make(N,halfN,K);
        double m8=h8?bench_mkl(h8,xin,cce,total):0; if(h8)DftiFreeDescriptor(&h8);

        printf("%-6zu %9.1f %10.0f %10.0f   %10.0f %10.0f   %7.2fx %7.2fx   %8.3fx %8.3fx\n",
               K, ws, d1, d8, m1, m8,
               d1/d8, (m8>0)?m1/m8:0, (d1>0)?m1/d1:0, (d8>0&&m8>0)?m8/d8:0);
        fflush(stdout);

        vfft_proto_aligned_free(x);vfft_proto_aligned_free(orr);vfft_proto_aligned_free(oii);
        vfft_proto_aligned_free(xin);vfft_proto_aligned_free(cce);
        stride_plan_destroy(p);
    }
    stride_set_num_threads(1);
    printf("\n# dagScal/mklScal = T1/T8 self-speedup. T8 mkl/dag = MKL8_ns/dag8_ns (>1 = dag faster at T8).\n");
    return 0;
}
