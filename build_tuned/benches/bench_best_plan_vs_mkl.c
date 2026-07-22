/* Best IL plan at (1024,4) [64,16] (DP pick, 5/5) vs MKL native interleaved.
 * JIT executors; IL roundtrip fully folded (fwd_ilin + bwd_ilout).
 * MKL threading via env only (mkl_set_num_threads segfaults in pip build). */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <x86intrin.h>
#include "prime_dispatch.h"
#include "plan_orchestrator.h"
#include "il_layout.h"
#include "il_execute.h"
#include "generator/generated/registry.h"
#include "mkl_dfti.h"
static const int N=1024; static size_t K=4;
static double maxrel(const double*a,const double*b,size_t n,double s){
    double mx=0,sc=0; for(size_t i=0;i<n;i++){double d=fabs(a[i]/s-b[i]);
        if(d>mx)mx=d; double v=fabs(b[i]); if(v>sc)sc=v;} return sc>0?mx/sc:mx; }
int main(int argc, char**argv){
    int warm = argc>1 && !strcmp(argv[1],"warm");
    if(argc>2) K=(size_t)atoi(argv[2]);
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    stride_set_num_threads(1);
    size_t n=(size_t)N*K;
    /* forced [64,16] */
    char p[]="/tmp/wfXXXXXX"; int fd=mkstemp(p);
    const char *pf = getenv("VFFT_PLAN");
    if(pf && !strcmp(pf,"8168"))
        dprintf(fd,"@version 6\n1024 %zu 3 8 16 8 0.0 0 0 0 0 0 2 2 0\n",K);
    else
        dprintf(fd,"@version 6\n1024 %zu 2 64 16 0.0 0 0 0 0 0 2 0\n",K); close(fd);
    vfft_proto_wisdom_t *w=calloc(1,sizeof *w);
    if(vfft_proto_wisdom_load(w,p)!=0){puts("wload fail");return 1;}
    unlink(p);
    vfft_proto_handle_t h;
    if(vfft_proto_plan(&h,N,K,VFFT_PROTO_WISDOM_ONLY,&reg,w,NULL)!=0){puts("plan fail");return 1;}
    double *z=aligned_alloc(64,n*16),*z0=aligned_alloc(64,n*16),*zc=aligned_alloc(64,n*16);
    double *cr=aligned_alloc(64,n*8),*ci=aligned_alloc(64,n*8);
    srand(123); for(size_t i=0;i<2*n;i++) z0[i]=2.0*rand()/RAND_MAX-1;
    memcpy(z,z0,n*16); memcpy(zc,z0,n*16);
    int fi = vfft_proto_execute_fwd_ilin_jit(h.plan,z,cr,ci,K,h.exec_fwd)>=0;
    int fo = vfft_proto_execute_bwd_ilout_jit(h.plan,cr,ci,z,K,h.exec_bwd)>=0;
    double ev=maxrel(z,z0,2*n,(double)N);
    if(warm){ fprintf(stderr,"warmed fold=%d/%d\n",fi,fo); return 0; }
    DFTI_DESCRIPTOR_HANDLE dl,dc; MKL_LONG sl[2]={0,(MKL_LONG)K}, sc2[2]={0,1};
    DftiCreateDescriptor(&dl,DFTI_DOUBLE,DFTI_COMPLEX,1,(MKL_LONG)N);
    DftiSetValue(dl,DFTI_NUMBER_OF_TRANSFORMS,(MKL_LONG)K);
    DftiSetValue(dl,DFTI_INPUT_STRIDES,sl); DftiSetValue(dl,DFTI_OUTPUT_STRIDES,sl);
    DftiSetValue(dl,DFTI_INPUT_DISTANCE,1); DftiSetValue(dl,DFTI_OUTPUT_DISTANCE,1);
    DftiCommitDescriptor(dl);
    DftiCreateDescriptor(&dc,DFTI_DOUBLE,DFTI_COMPLEX,1,(MKL_LONG)N);
    DftiSetValue(dc,DFTI_NUMBER_OF_TRANSFORMS,(MKL_LONG)K);
    DftiSetValue(dc,DFTI_INPUT_STRIDES,sc2); DftiSetValue(dc,DFTI_OUTPUT_STRIDES,sc2);
    DftiSetValue(dc,DFTI_INPUT_DISTANCE,(MKL_LONG)N); DftiSetValue(dc,DFTI_OUTPUT_DISTANCE,(MKL_LONG)N);
    DftiCommitDescriptor(dc);
    memcpy(z,z0,n*16); DftiComputeForward(dl,z); DftiComputeBackward(dl,z);
    double em=maxrel(z,z0,2*n,(double)N);
    int reps=(int)(4e6/(double)n); if(reps<2)reps=2; if(reps>400)reps=400;
    double til=1e18,tsp=1e18,tl=1e18,tc=1e18,tvf=1e18,tlf=1e18;
    for(int t=0;t<9;t++){
        double t0=(double)__rdtsc();
        for(int r=0;r<reps;r++){
            vfft_proto_execute_fwd_ilin_jit(h.plan,z,cr,ci,K,h.exec_fwd);
            vfft_proto_execute_bwd_ilout_jit(h.plan,cr,ci,z,K,h.exec_bwd); }
        double v=((double)__rdtsc()-t0)/reps; if(v<til)til=v;
        t0=(double)__rdtsc();
        for(int r=0;r<reps;r++){ vfft_proto_plan_execute_fwd(&h,cr,ci);
                                 vfft_proto_plan_execute_bwd(&h,cr,ci); }
        v=((double)__rdtsc()-t0)/reps; if(v<tsp)tsp=v;
        t0=(double)__rdtsc();
        for(int r=0;r<reps;r++){ DftiComputeForward(dl,z); DftiComputeBackward(dl,z); }
        v=((double)__rdtsc()-t0)/reps; if(v<tl)tl=v;
        t0=(double)__rdtsc();
        for(int r=0;r<reps;r++){ DftiComputeForward(dc,zc); DftiComputeBackward(dc,zc); }
        v=((double)__rdtsc()-t0)/reps; if(v<tc)tc=v;
        t0=(double)__rdtsc();
        for(int r=0;r<reps;r++){
            vfft_proto_execute_fwd_ilin_jit(h.plan,z,cr,ci,K,h.exec_fwd);
            vfft_sp2il(cr,ci,z,n); }
        v=((double)__rdtsc()-t0)/reps; if(v<tvf)tvf=v;
        t0=(double)__rdtsc();
        for(int r=0;r<reps;r++) DftiComputeForward(dl,z);
        v=((double)__rdtsc()-t0)/reps; if(v<tlf)tlf=v;
    }
    printf("(1024,%zu) %s jit fold=%s/%s  [err v %.1e m %.1e]\n", K, (getenv("VFFT_PLAN")&&!strcmp(getenv("VFFT_PLAN"),"8168"))?"[8,16,8]":"[64,16]",
        fi?"in":"IN!", fo?"out":"OUT!", ev, em);
    printf("VFFT IL rt   %8.0f\nVFFT split   %8.0f\nMKL lane rt  %8.0f  -> %.2fx\nMKL ctg  rt  %8.0f  -> %.2fx\nfwd-only: VFFT(ilin+sp2il) %8.0f  MKL lane %8.0f  -> %.2fx\n",
        til, tsp, tl, tl/til, tc, tc/til, tvf, tlf, tlf/tvf);
    return 0;
}
