/* bench_r2c_highN_vs_mkl.c — TEST the "compute dominates at high N" theory.
 *
 * Hypothesis (Tugbars): N=256 is the regime LEAST favorable to the decoupled
 * path — the inner FFT is tiny so the O(N) pack tax is a big fraction. As N
 * grows, FFT compute O(N log N) outpaces pack/recombine O(N), and our split
 * layout's ~2x SIMD edge on the FFT (the C2C 238/238 winner) should dominate
 * the pack tax → decoupled-stride should beat MKL by MORE at high N. Counter:
 * high N*K can go memory/DTLB-bound, erasing the compute edge. Measure it.
 *
 * Sweeps N (pow2) at fixed K, SPLIT layout, via the production dispatcher:
 *   - DECOUPLED stride: decouple_min_k=0 (force).   - RFFT: decouple_min_k=MAX.
 * vs single-thread MKL r2c. Spot-checks 8 bins vs reference DFT per cell.
 *
 * Build: cd build_tuned && python build.py --src benches/bench_r2c_highN_vs_mkl.c --mkl --compile
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
#include "r2c_dispatch.h"
#include "env.h"
#include "dp_planner.h"
#include "generator/generated/registry.h"

#define PIN_CORE 2
#define BEST_OF  15
static double *alloc_d(size_t n){double*p=NULL;
    if(vfft_proto_posix_memalign((void**)&p,64,n*sizeof(double))!=0){exit(1);} return p;}
static int reps_for(size_t t){int r=(int)(8e6/(t+1)); if(r<20)r=20; if(r>100000)r=100000; return r;}

/* spot-check: 8 evenly spaced bins, lane 0, vs direct DFT */
static double spot(const double *o_re,const double *o_im,const double *x,int N,size_t K){
    double me=0;
    for(int q=0;q<8;q++){
        int k=(int)((long)q*(N/2)/7); if(k>N/2)k=N/2;
        double rr=0,ri=0;
        for(int n=0;n<N;n++){double xn=x[(size_t)n*K];double a=-2.0*M_PI*k*n/(double)N;rr+=xn*cos(a);ri+=xn*sin(a);}
        double er=fabs(o_re[(size_t)k*K]-rr),ei=fabs(o_im[(size_t)k*K]-ri);
        if(er>me)me=er; if(ei>me)me=ei;
    }
    return me;
}
static double bench_plan(const vfft_r2c_plan_t*p,const double*x,double*o_re,double*o_im,size_t total){
    for(int w=0;w<8;w++) vfft_r2c_execute_fwd(p,x,o_re,o_im);
    int reps=reps_for(total); double best=1e18;
    for(int t=0;t<BEST_OF;t++){double t0=vfft_proto_now_ns();
        for(int i=0;i<reps;i++) vfft_r2c_execute_fwd(p,x,o_re,o_im);
        double ns=(vfft_proto_now_ns()-t0)/reps; if(ns<best)best=ns;}
    return best;
}
static double bench_mkl(DFTI_DESCRIPTOR_HANDLE h,const double*xin,double*cce,size_t total){
    for(int w=0;w<8;w++) DftiComputeForward(h,(void*)xin,cce);
    int reps=reps_for(total); double best=1e18;
    for(int t=0;t<BEST_OF;t++){double t0=vfft_proto_now_ns();
        for(int i=0;i<reps;i++) DftiComputeForward(h,(void*)xin,cce);
        double ns=(vfft_proto_now_ns()-t0)/reps; if(ns<best)best=ns;}
    return best;
}

int main(int argc,char**argv){
    stride_env_init();
    if(stride_pin_thread(PIN_CORE)!=0) fprintf(stderr,"warn pin\n");
    mkl_set_num_threads(1);
    size_t K = argc>1?(size_t)atoi(argv[1]):64;
    const int Ns[]={256,512,1024,2048,4096,8192}; const int nN=6;

    rfft_codelets_t rreg; memset(&rreg,0,sizeof rreg); rfft_register_all_avx2(&rreg);
    vfft_proto_registry_t creg; vfft_proto_registry_init(&creg);
    vfft_proto_wisdom_t wis; int hw=
        (vfft_proto_wisdom_load(&wis,"../src/dag-fft-compiler/generator/generated/spike_wisdom.txt")==0);
    if(!hw) hw=(vfft_proto_wisdom_load(&wis,"../../src/dag-fft-compiler/generator/generated/spike_wisdom.txt")==0);
    if(hw) vfft_r2c_dispatch_set_c2c_wisdom(&wis);

    printf("=== r2c high-N theory test (SPLIT, ST, cpu%d, K=%zu) — does decoupled/MKL improve with N? ===\n",PIN_CORE,K);
    printf("# c2c inner wisdom: %s\n", hw?"OK":"FAILED");
    printf("%-6s %12s %12s %12s %10s %10s %9s\n","N","rfft_ns","stride_ns","mkl_ns","mkl/rfft","mkl/strd","strd/rfft");
    printf("------+------------+------------+------------+----------+----------+---------\n");

    for(int ni=0;ni<nN;ni++){
        int N=Ns[ni]; int halfN=N/2; size_t total=(size_t)N*K;
        double *x=alloc_d(total); srand(7+N);
        for(size_t i=0;i<total;i++) x[i]=(double)rand()/RAND_MAX*2-1;
        double *oor=alloc_d((size_t)(halfN+1)*K),*ooi=alloc_d((size_t)(halfN+1)*K);

        vfft_r2c_dispatch_set_decouple_min_k((size_t)-1);
        vfft_r2c_plan_t*pr=vfft_r2c_plan_create(N,K,VFFT_R2C_SPLIT,&rreg,NULL,&creg);
        vfft_r2c_dispatch_set_decouple_min_k(0);
        vfft_r2c_plan_t*ps=vfft_r2c_plan_create(N,K,VFFT_R2C_SPLIT,&rreg,NULL,&creg);

        double er=9,es=9;
        if(pr){memset(oor,0,(size_t)(halfN+1)*K*8);memset(ooi,0,(size_t)(halfN+1)*K*8);
            vfft_r2c_execute_fwd(pr,x,oor,ooi); er=spot(oor,ooi,x,N,K);}
        if(ps){memset(oor,0,(size_t)(halfN+1)*K*8);memset(ooi,0,(size_t)(halfN+1)*K*8);
            vfft_r2c_execute_fwd(ps,x,oor,ooi); es=spot(oor,ooi,x,N,K);}
        if((pr&&er>=1e-8)||(ps&&es>=1e-8)) printf("# N=%d CORRECTNESS rfft=%.2e strd=%.2e\n",N,er,es);

        DFTI_DESCRIPTOR_HANDLE h=0;
        double *xin=alloc_d(total),*cce=alloc_d((size_t)(halfN+1)*K*2);
        for(size_t t=0;t<K;t++)for(int n=0;n<N;n++)xin[t*N+n]=x[(size_t)n*K+t];
        DftiCreateDescriptor(&h,DFTI_DOUBLE,DFTI_REAL,1,(MKL_LONG)N);
        DftiSetValue(h,DFTI_NUMBER_OF_TRANSFORMS,(MKL_LONG)K);
        DftiSetValue(h,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
        DftiSetValue(h,DFTI_CONJUGATE_EVEN_STORAGE,DFTI_COMPLEX_COMPLEX);
        DftiSetValue(h,DFTI_INPUT_DISTANCE,(MKL_LONG)N);
        DftiSetValue(h,DFTI_OUTPUT_DISTANCE,(MKL_LONG)(halfN+1));
        int mok=(DftiCommitDescriptor(h)==DFTI_NO_ERROR);

        double r_ns=pr?bench_plan(pr,x,oor,ooi,total):0;
        double s_ns=ps?bench_plan(ps,x,oor,ooi,total):0;
        double m_ns=mok?bench_mkl(h,xin,cce,total):0;
        printf("%-6d %12.1f %12.1f %12.1f %9.3fx %9.3fx %8.3fx\n",
               N,r_ns,s_ns,m_ns,(m_ns>0&&r_ns>0)?m_ns/r_ns:0,(m_ns>0&&s_ns>0)?m_ns/s_ns:0,(s_ns>0&&r_ns>0)?r_ns/s_ns:0);

        if(h)DftiFreeDescriptor(&h);
        vfft_r2c_plan_destroy(pr);vfft_r2c_plan_destroy(ps);
        vfft_proto_aligned_free(x);vfft_proto_aligned_free(oor);vfft_proto_aligned_free(ooi);
        vfft_proto_aligned_free(xin);vfft_proto_aligned_free(cce);
    }
    if(hw) vfft_proto_wisdom_free(&wis);
    printf("\n# THEORY: if mkl/strd RISES with N, compute-dominance + split 2x SIMD beats the pack tax at scale.\n");
    return 0;
}
