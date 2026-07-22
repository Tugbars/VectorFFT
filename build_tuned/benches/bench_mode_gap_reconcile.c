/* Reconciliation: ours-split (jit) vs MKL-SPLIT-mode (the v1.0 comparator,
 * DFTI_REAL_REAL, identical lane-major layout) vs MKL native interleaved.
 * The MKLsplit/MKLctg ratio is the "mode gap" that maps v1.0 margins onto
 * native-interleaved baselines. */
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
static void cell(int N, size_t K, const char *wline, const vfft_proto_registry_t *reg, int warm){
    size_t n=(size_t)N*K;
    char p[]="/tmp/wrXXXXXX"; int fd=mkstemp(p);
    dprintf(fd,"@version 6\n%s\n",wline); close(fd);
    vfft_proto_wisdom_t *w=calloc(1,sizeof *w);
    vfft_proto_wisdom_load(w,p); unlink(p);
    vfft_proto_handle_t h;
    if(vfft_proto_plan(&h,N,K,VFFT_PROTO_WISDOM_ONLY,reg,w,NULL)!=0){printf("planfail\n");return;}
    double *cr=aligned_alloc(64,n*8),*ci=aligned_alloc(64,n*8);
    double *sr=aligned_alloc(64,n*8),*si=aligned_alloc(64,n*8);
    double *z=aligned_alloc(64,n*16),*zc=aligned_alloc(64,n*16);
    srand(9); for(size_t i=0;i<n;i++){cr[i]=2.0*rand()/RAND_MAX-1;ci[i]=2.0*rand()/RAND_MAX-1;
        sr[i]=cr[i];si[i]=ci[i];}
    for(size_t i=0;i<n;i++){z[2*i]=cr[i];z[2*i+1]=ci[i];zc[2*i]=cr[i];zc[2*i+1]=ci[i];}
    if(warm){ vfft_proto_plan_execute_fwd(&h,cr,ci); vfft_proto_plan_execute_bwd(&h,cr,ci);
        fprintf(stderr,"warm %d %zu ok\n",N,K); return; }
    /* MKL split-mode (v1.0 comparator): REAL_REAL, lane-major {0,K}, dist 1 */
    DFTI_DESCRIPTOR_HANDLE ds,dl,dc; MKL_LONG sl[2]={0,(MKL_LONG)K}, sc[2]={0,1};
    DftiCreateDescriptor(&ds,DFTI_DOUBLE,DFTI_COMPLEX,1,(MKL_LONG)N);
    DftiSetValue(ds,DFTI_COMPLEX_STORAGE,DFTI_REAL_REAL);
    DftiSetValue(ds,DFTI_NUMBER_OF_TRANSFORMS,(MKL_LONG)K);
    DftiSetValue(ds,DFTI_INPUT_STRIDES,sl); DftiSetValue(ds,DFTI_OUTPUT_STRIDES,sl);
    DftiSetValue(ds,DFTI_INPUT_DISTANCE,1); DftiSetValue(ds,DFTI_OUTPUT_DISTANCE,1);
    DftiCommitDescriptor(ds);
    DftiCreateDescriptor(&dl,DFTI_DOUBLE,DFTI_COMPLEX,1,(MKL_LONG)N);
    DftiSetValue(dl,DFTI_NUMBER_OF_TRANSFORMS,(MKL_LONG)K);
    DftiSetValue(dl,DFTI_INPUT_STRIDES,sl); DftiSetValue(dl,DFTI_OUTPUT_STRIDES,sl);
    DftiSetValue(dl,DFTI_INPUT_DISTANCE,1); DftiSetValue(dl,DFTI_OUTPUT_DISTANCE,1);
    DftiCommitDescriptor(dl);
    DftiCreateDescriptor(&dc,DFTI_DOUBLE,DFTI_COMPLEX,1,(MKL_LONG)N);
    DftiSetValue(dc,DFTI_NUMBER_OF_TRANSFORMS,(MKL_LONG)K);
    DftiSetValue(dc,DFTI_INPUT_STRIDES,sc); DftiSetValue(dc,DFTI_OUTPUT_STRIDES,sc);
    DftiSetValue(dc,DFTI_INPUT_DISTANCE,(MKL_LONG)N); DftiSetValue(dc,DFTI_OUTPUT_DISTANCE,(MKL_LONG)N);
    DftiCommitDescriptor(dc);
    int reps=(int)(4e6/(double)n); if(reps<2)reps=2; if(reps>400)reps=400;
    double tv=1e18,ts=1e18,tl=1e18,tc=1e18;
    for(int t=0;t<9;t++){
        double t0=(double)__rdtsc();
        for(int r=0;r<reps;r++){ vfft_proto_plan_execute_fwd(&h,cr,ci);
                                 vfft_proto_plan_execute_bwd(&h,cr,ci); }
        double v=((double)__rdtsc()-t0)/reps; if(v<tv)tv=v;
        t0=(double)__rdtsc();
        for(int r=0;r<reps;r++){ DftiComputeForward(ds,sr,si); DftiComputeBackward(ds,sr,si); }
        v=((double)__rdtsc()-t0)/reps; if(v<ts)ts=v;
        t0=(double)__rdtsc();
        for(int r=0;r<reps;r++){ DftiComputeForward(dl,z); DftiComputeBackward(dl,z); }
        v=((double)__rdtsc()-t0)/reps; if(v<tl)tl=v;
        t0=(double)__rdtsc();
        for(int r=0;r<reps;r++){ DftiComputeForward(dc,zc); DftiComputeBackward(dc,zc); }
        v=((double)__rdtsc()-t0)/reps; if(v<tc)tc=v;
    }
    printf("(%d,%zu) ours-split %8.0f | MKLsplit %8.0f (we %.2fx) | MKLil-lane %8.0f | MKLil-ctg %8.0f | mode gap split/ctg %.2fx\n",
        N,K,tv,ts,ts/tv,tl,tc,ts/tc);
    DftiFreeDescriptor(&ds); DftiFreeDescriptor(&dl); DftiFreeDescriptor(&dc);
    free(cr);free(ci);free(sr);free(si);free(z);free(zc);
}
int main(int argc,char**argv){
    int warm = argc>1 && !strcmp(argv[1],"warm");
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    stride_set_num_threads(1);
    cell(1024,4,"1024 4 2 64 16 0.0 0 0 0 0 0 2 0",&reg,warm);
    cell(64,256,"64 256 3 4 4 4 0.0 0 0 0 0 0 2 2 0",&reg,warm);
    return 0;
}
