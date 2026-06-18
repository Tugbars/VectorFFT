/* bench_r2c_dispatch_vs_mkl.c — PRODUCTION dispatch validation + crossover.
 *
 * Uses the REAL public dispatcher (vfft_r2c_plan_create / vfft_r2c_execute_fwd,
 * SPLIT layout) to time both routings vs MKL across K, N=256:
 *   - RFFT path   : threshold disabled (default) -> rfft natural-split.
 *   - STRIDE path : decouple_min_k=0 -> decoupled (pack-fused + general recombine,
 *                   wisdom-best inner via set_c2c_wisdom).
 * Correctness-gated (both paths) vs reference DFT. The crossover sets the
 * hybrid threshold (_vfft_r2c_decouple_min_k) in r2c_dispatch.h.
 *
 * Build: cd build_tuned && python build.py --src benches/bench_r2c_dispatch_vs_mkl.c --mkl --compile
 * Run  : PATH += MKL bin + C:\mingw152\mingw64\bin.
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

#include "rfft_registry_avx2.h"          /* rfft_codelets_t + rfft_register_all_avx2 */
#include "core/r2c_dispatch.h"           /* the production dispatcher */
#include "core/env.h"
#include "core/dp_planner.h"             /* vfft_proto_now_ns */
#include "generator/generated/registry.h"

#define PIN_CORE 2
#define BEST_OF  21

static double *alloc_d(size_t n){double*p=NULL;
    if(vfft_proto_posix_memalign((void**)&p,64,n*sizeof(double))!=0){exit(1);} return p;}
static int reps_for(size_t t){int r=(int)(4e6/(t+1)); if(r<30)r=30; if(r>200000)r=200000; return r;}

static double check(const double *o_re,const double *o_im,const double *x,
                    int N,int halfN,size_t K,size_t lane){
    double me=0;
    for(int k=0;k<=halfN;k++){double rr=0,ri=0;
        for(int n=0;n<N;n++){double xn=x[(size_t)n*K+lane];double a=-2.0*M_PI*k*n/(double)N;rr+=xn*cos(a);ri+=xn*sin(a);}
        double er=fabs(o_re[(size_t)k*K+lane]-rr),ei=fabs(o_im[(size_t)k*K+lane]-ri);
        if(er>me)me=er; if(ei>me)me=ei;}
    return me;
}
static double bench_plan(const vfft_r2c_plan_t *p,const double *x,double *o_re,double *o_im,size_t total){
    for(int w=0;w<10;w++) vfft_r2c_execute_fwd(p,x,o_re,o_im);
    int reps=reps_for(total); double best=1e18;
    for(int t=0;t<BEST_OF;t++){double t0=vfft_proto_now_ns();
        for(int i=0;i<reps;i++) vfft_r2c_execute_fwd(p,x,o_re,o_im);
        double ns=(vfft_proto_now_ns()-t0)/reps; if(ns<best)best=ns;}
    return best;
}
static double bench_mkl(DFTI_DESCRIPTOR_HANDLE h,const double *xin,double *cce,size_t total){
    for(int w=0;w<10;w++) DftiComputeForward(h,(void*)xin,cce);
    int reps=reps_for(total); double best=1e18;
    for(int t=0;t<BEST_OF;t++){double t0=vfft_proto_now_ns();
        for(int i=0;i<reps;i++) DftiComputeForward(h,(void*)xin,cce);
        double ns=(vfft_proto_now_ns()-t0)/reps; if(ns<best)best=ns;}
    return best;
}

int main(void){
    stride_env_init();
    if(stride_pin_thread(PIN_CORE)!=0) fprintf(stderr,"warn pin\n");
    mkl_set_num_threads(1);
    const int N=256, halfN=N/2;
    const size_t Ks[]={8,16,32,64,128,256}; const int nK=6;

    rfft_codelets_t rreg; memset(&rreg,0,sizeof rreg); rfft_register_all_avx2(&rreg);
    vfft_proto_registry_t creg; vfft_proto_registry_init(&creg);
    vfft_proto_wisdom_t wis; int hw=
        (vfft_proto_wisdom_load(&wis,"../src/dag-fft-compiler/generator/generated/spike_wisdom.txt")==0);
    if(!hw) hw=(vfft_proto_wisdom_load(&wis,"../../src/dag-fft-compiler/generator/generated/spike_wisdom.txt")==0);
    if(hw) vfft_r2c_dispatch_set_c2c_wisdom(&wis);
    printf("# c2c inner wisdom: %s\n", hw?"OK":"FAILED (degenerate inner)");

    printf("=== production r2c dispatch: RFFT vs DECOUPLED-STRIDE vs MKL (N=256, SPLIT, ST, cpu%d) ===\n",PIN_CORE);
    printf("%-5s %12s %12s %12s %10s %10s   %s\n",
           "K","rfft_ns","stride_ns","mkl_ns","mkl/rfft","mkl/strd","winner(for SPLIT)");
    printf("------+------------+------------+------------+----------+----------+-----------\n");

    for(int ki=0;ki<nK;ki++){
        size_t K=Ks[ki]; size_t total=(size_t)N*K;
        double *x=alloc_d(total); srand(7+(int)K);
        for(size_t i=0;i<total;i++) x[i]=(double)rand()/RAND_MAX*2-1;
        double *oor=alloc_d((size_t)(halfN+1)*K),*ooi=alloc_d((size_t)(halfN+1)*K);

        /* RFFT path: threshold disabled */
        vfft_r2c_dispatch_set_decouple_min_k((size_t)-1);
        vfft_r2c_plan_t *pr=vfft_r2c_plan_create(N,K,VFFT_R2C_SPLIT,&rreg,NULL,&creg);
        /* STRIDE path: force decoupled */
        vfft_r2c_dispatch_set_decouple_min_k(0);
        vfft_r2c_plan_t *ps=vfft_r2c_plan_create(N,K,VFFT_R2C_SPLIT,&rreg,NULL,&creg);

        if(!pr||pr->path!=VFFT_R2C_PATH_RFFT){printf("%-5zu rfft plan/path bad\n",K);}
        if(!ps||ps->path!=VFFT_R2C_PATH_STRIDE){printf("%-5zu stride plan/path bad\n",K);}

        /* correctness */
        double er=9,es=9;
        if(pr){memset(oor,0,(size_t)(halfN+1)*K*8);memset(ooi,0,(size_t)(halfN+1)*K*8);
            vfft_r2c_execute_fwd(pr,x,oor,ooi); er=check(oor,ooi,x,N,halfN,K,0);}
        if(ps){memset(oor,0,(size_t)(halfN+1)*K*8);memset(ooi,0,(size_t)(halfN+1)*K*8);
            vfft_r2c_execute_fwd(ps,x,oor,ooi); es=check(oor,ooi,x,N,halfN,K,0);}
        if(er>=1e-9||es>=1e-9){printf("%-5zu CORRECTNESS rfft=%.2e stride=%.2e\n",K,er,es);}

        /* MKL */
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
        const char *win = (s_ns>0&&r_ns>0)?(s_ns<r_ns?"STRIDE":"rfft"):"?";
        printf("%-5zu %12.1f %12.1f %12.1f %9.3fx %9.3fx   %s\n",
               K,r_ns,s_ns,m_ns, (m_ns>0&&r_ns>0)?m_ns/r_ns:0, (m_ns>0&&s_ns>0)?m_ns/s_ns:0, win);

        if(h)DftiFreeDescriptor(&h);
        vfft_r2c_plan_destroy(pr); vfft_r2c_plan_destroy(ps);
        vfft_proto_aligned_free(x);vfft_proto_aligned_free(oor);vfft_proto_aligned_free(ooi);
        vfft_proto_aligned_free(xin);vfft_proto_aligned_free(cce);
    }
    if(hw) vfft_proto_wisdom_free(&wis);
    printf("\n# mkl/rfft, mkl/strd = MKL_ns/ours (>1 = we beat MKL). winner = faster of the two for SPLIT.\n");
    return 0;
}
