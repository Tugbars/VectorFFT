/* bench_r2c_dispatch_mt_vs_mkl.c — validate r2c MT through the PUBLIC dispatcher.
 *
 * Unlike bench_r2c_mt_vs_mkl.c (which hand-builds the stride plan with block_K=K/8),
 * this uses vfft_r2c_plan_create / vfft_r2c_execute_fwd — the real entry point. The
 * dispatcher now picks block_K<K when stride_set_num_threads()>1 at plan-create
 * (r2c_dispatch.h _vfft_r2c_block_k), so MT engages automatically. We verify:
 *   - correctness (lane 0 vs reference DFT),
 *   - MT==T1 output (err),
 *   - scaling T1->T8, and dag T8 vs MKL T8.
 *
 * Build: cd build_tuned && python build.py --src benches/bench_r2c_dispatch_mt_vs_mkl.c --mkl --compile
 */
#define _GNU_SOURCE 1
#define VFFT_RFFT_MAX_RADIX 32
#define VFFT_RFFT_RANGED 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mkl_dfti.h>
#include <mkl_service.h>

#include "rfft_registry_avx2.h"
#include "core/r2c_dispatch.h"
#include "core/env.h"
#include "core/dp_planner.h"          /* vfft_proto_now_ns */
#include "generator/generated/registry.h"

#define BEST_OF 15
static double *alloc_d(size_t n){double*p=NULL;
    if(vfft_proto_posix_memalign((void**)&p,64,n*sizeof(double))!=0){exit(1);} return p;}
static int reps_for(size_t t){int r=(int)(2e7/(t+1)); if(r<5)r=5; if(r>5000)r=5000; return r;}

static double check(const double*orr,const double*oii,const double*x,int N,int halfN,size_t K){
    double me=0;
    for(int k=0;k<=halfN;k++){double rr=0,ri=0;
        for(int n=0;n<N;n++){double xn=x[(size_t)n*K];double a=-2.0*M_PI*k*n/(double)N;rr+=xn*cos(a);ri+=xn*sin(a);}
        double er=fabs(orr[(size_t)k*K]-rr),ei=fabs(oii[(size_t)k*K]-ri);
        if(er>me)me=er; if(ei>me)me=ei;}
    return me;
}
static double bench_dag(const vfft_r2c_plan_t*p,const double*x,double*orr,double*oii,size_t total,int T){
    stride_set_num_threads(T);
    for(int w=0;w<5;w++) vfft_r2c_execute_fwd(p,x,orr,oii);
    int reps=reps_for(total); double best=1e18;
    for(int b=0;b<BEST_OF;b++){double t0=vfft_proto_now_ns();
        for(int i=0;i<reps;i++) vfft_r2c_execute_fwd(p,x,orr,oii);
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
    if(stride_pin_thread(0)!=0) fprintf(stderr,"warn pin\n");
    rfft_codelets_t rreg; memset(&rreg,0,sizeof rreg); rfft_register_all_avx2(&rreg);
    vfft_proto_registry_t creg; vfft_proto_registry_init(&creg);
    vfft_proto_wisdom_t wis; int hw=
        (vfft_proto_wisdom_load(&wis,"../src/dag-fft-compiler/generator/generated/spike_wisdom.txt")==0);
    if(!hw) hw=(vfft_proto_wisdom_load(&wis,"../../src/dag-fft-compiler/generator/generated/spike_wisdom.txt")==0);
    if(hw) vfft_r2c_dispatch_set_c2c_wisdom(&wis);

    const int N=256, halfN=N/2;
    const size_t Ks[]={256,512,1024,2048,4096,8192}; const int nK=6;

    printf("=== r2c via PUBLIC DISPATCHER: dag MT (8 P-cores) vs MKL (8 thr), N=%d, SPLIT ===\n",N);
    printf("# c2c inner wisdom: %s. dispatcher picks block_K<K when threads>1 at plan-create.\n",hw?"OK":"MISSING");
    printf("%-6s %8s %6s %10s %10s   %10s %10s   %8s   %9s   %9s\n",
           "K","path","mtErr","dagT1_ns","dagT8_ns","mklT1_ns","mklT8_ns","dagScal","T8mkl/dag","correct");
    printf("-------+--------+------+----------+----------+----------+----------+--------+----------+--------\n");

    for(int ki=0;ki<nK;ki++){
        size_t K=Ks[ki]; size_t total=(size_t)N*K;
        /* build at T=8 so the dispatcher selects a sub-K block + 8 scratch slots */
        stride_set_num_threads(8);
        vfft_r2c_plan_t*p=vfft_r2c_plan_create(N,K,VFFT_R2C_SPLIT,&rreg,NULL,&creg);
        if(!p){printf("%-6zu plan NULL\n",K);continue;}
        const char*path=(p->path==VFFT_R2C_PATH_STRIDE)?"stride":"rfft";

        double*x=alloc_d(total); srand(7+(int)K);
        for(size_t i=0;i<total;i++)x[i]=(double)rand()/RAND_MAX*2-1;
        double*orr=alloc_d((size_t)(halfN+1)*K),*oii=alloc_d((size_t)(halfN+1)*K),*ref=alloc_d((size_t)(halfN+1)*K);

        /* correctness + MT==T1 */
        stride_set_num_threads(1); vfft_r2c_execute_fwd(p,x,orr,oii);
        double cerr=check(orr,oii,x,N,halfN,K); memcpy(ref,orr,(size_t)(halfN+1)*K*8);
        stride_set_num_threads(8); vfft_r2c_execute_fwd(p,x,orr,oii);
        double mterr=0; for(size_t i=0;i<(size_t)(halfN+1)*K;i++){double a=orr[i]-ref[i];if(a<0)a=-a;if(a>mterr)mterr=a;}

        double*xin=alloc_d(total),*cce=alloc_d((size_t)(halfN+1)*K*2);
        for(size_t t=0;t<K;t++)for(int n=0;n<N;n++)xin[t*N+n]=x[(size_t)n*K+t];

        double d1=bench_dag(p,x,orr,oii,total,1);
        double d8=bench_dag(p,x,orr,oii,total,8);
        mkl_set_num_threads(1); DFTI_DESCRIPTOR_HANDLE h1=mkl_make(N,halfN,K);
        double m1=h1?bench_mkl(h1,xin,cce,total):0; if(h1)DftiFreeDescriptor(&h1);
        mkl_set_num_threads(8); DFTI_DESCRIPTOR_HANDLE h8=mkl_make(N,halfN,K);
        double m8=h8?bench_mkl(h8,xin,cce,total):0; if(h8)DftiFreeDescriptor(&h8);

        printf("%-6zu %8s %6.0e %10.0f %10.0f   %10.0f %10.0f   %7.2fx   %8.3fx   %s\n",
               K, path, mterr, d1, d8, m1, m8, d1/d8,
               (d8>0&&m8>0)?m8/d8:0, (cerr<1e-9)?"ok":"BAD");
        fflush(stdout);

        vfft_proto_aligned_free(x);vfft_proto_aligned_free(orr);vfft_proto_aligned_free(oii);
        vfft_proto_aligned_free(ref);vfft_proto_aligned_free(xin);vfft_proto_aligned_free(cce);
        vfft_r2c_plan_destroy(p);
    }
    if(hw) vfft_proto_wisdom_free(&wis);
    stride_set_num_threads(1);
    printf("\n# T8mkl/dag>1 = dag faster at 8 threads. mtErr = |T8-T1| (0 => MT correct). path=stride means MT-capable.\n");
    return 0;
}
