/* bench_il_1d_vs_mkl — 1D c2c on MKL's home format (interleaved complex).
 * Mirror of the v1.0 protocol (which forced MKL into split storage): here BOTH
 * sides run lane-major interleaved, plus an MKL-preferred contiguous arm.
 * Headline = fwd+bwd ROUNDTRIP z->z: order-neutral (in-place output is
 * digit-scrambled per the convolution contract) and fully folded on our side.
 * Reference columns: split-orch (handle exec on split, no IL) and split-exh
 * (patient-planner reference) isolate IL cost from engine cost.
 * Plans: DP MEASURE via the orchestrator; executors baked-or-JIT.
 * MKL threading: env only (MKL_THREADING_LAYER=SEQUENTIAL MKL_NUM_THREADS=1);
 * mkl_set_num_threads() segfaults in the pip mkl_rt build.
 * Usage: ./bench [first_cell last_cell]   (MEASURE sweeps are per-invocation) */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <x86intrin.h>
#include "prime_dispatch.h"
#include "exhaustive_plan.h"
#include "plan_orchestrator.h"
#include "il_layout.h"
#include "il_execute.h"
#include "generator/generated/registry.h"
#include "mkl_dfti.h"
/* Win32: MSVCRT has no C11 aligned_alloc; the repo convention (cf. zsplit.h)
 * is _aligned_malloc with the arguments swapped. The matching free MUST be
 * _aligned_free — passing _aligned_malloc memory to plain free() is UB on
 * Windows and crashes at the end of the first cell. Hence AFREE. */
#ifdef _WIN32
#include <malloc.h>
#define aligned_alloc(a, sz) _aligned_malloc((sz), (a))
#define AFREE(p) _aligned_free(p)
#else
#define AFREE(p) free(p)
#endif
static double now_min(double *best, double t0, int reps){
    double v=((double)__rdtsc()-t0)/reps; if(v<*best)*best=v; return v; }
static double maxrel(const double*a,const double*b,size_t n,double s){
    double mx=0,sc=0; for(size_t i=0;i<n;i++){double d=fabs(a[i]/s-b[i]);
        if(d>mx)mx=d; double v=fabs(b[i]); if(v>sc)sc=v;} return sc>0?mx/sc:mx; }
int main(int argc, char**argv){
    int c0 = argc>1?atoi(argv[1]):0, c1 = argc>2?atoi(argv[2]):99;
    static const int CN[] = {64, 256, 1024, 4096, 320, 1000};
    static const int CK[] = {4, 4, 4, 4, 4, 4};
    int NC = 6;
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    stride_set_num_threads(1);
    static vfft_proto_wisdom_t wis; memset(&wis,0,sizeof wis);
    printf("%6s %4s %6s | %10s %10s %7s | %10s %7s | %7s\n",
        "N","K","folded","VFFT rt","MKLlane rt","speedx","MKLctg rt","speedx","fwd x");
    for(int c=c0;c<NC && c<=c1;c++){
        int N=CN[c]; size_t K=(size_t)CK[c], n=(size_t)N*K;
        vfft_proto_handle_t h;
        if(vfft_proto_plan(&h,N,K,VFFT_PROTO_MEASURE,&reg,&wis,NULL)!=0){
            printf("%6d %4zu  plan-fail\n",N,K); continue; }
        fprintf(stderr,"plan N=%d K=%zu: nst=%d dif=%d fac=[",N,K,
            h.plan->num_stages,h.plan->use_dif_forward);
        for(int s=0;s<h.plan->num_stages;s++)
            fprintf(stderr,"%d%s",h.plan->factors[s],s+1<h.plan->num_stages?",":"");
        fprintf(stderr,"] exec=%s/%s\n",h.exec_fwd?"jit":"gen",h.exec_bwd?"jit":"gen");
        double *z=aligned_alloc(64,n*16),*z0=aligned_alloc(64,n*16),*zc=aligned_alloc(64,n*16);
        double *cr=aligned_alloc(64,n*8),*ci=aligned_alloc(64,n*8);
        srand(42+c); for(size_t i=0;i<2*n;i++) z0[i]=2.0*rand()/RAND_MAX-1;
        memcpy(zc,z0,n*16);
        int fold_in=1, fold_out=1;
        memcpy(z,z0,n*16);
        if(vfft_proto_execute_fwd_ilin_jit(h.plan,z,cr,ci,K,h.exec_fwd)<0){
            fold_in=0; vfft_il2sp(z,cr,ci,n); vfft_proto_plan_execute_fwd(&h,cr,ci); }
        if(vfft_proto_execute_bwd_ilout_jit(h.plan,cr,ci,z,K,h.exec_bwd)<0){
            fold_out=0; vfft_proto_plan_execute_bwd(&h,cr,ci); vfft_sp2il(cr,ci,z,n); }
        double e=maxrel(z,z0,2*n,(double)N);
        DFTI_DESCRIPTOR_HANDLE dl,dc; MKL_LONG st_l[2]={0,(MKL_LONG)K}, st_c[2]={0,1};
        DftiCreateDescriptor(&dl,DFTI_DOUBLE,DFTI_COMPLEX,1,(MKL_LONG)N);
        DftiSetValue(dl,DFTI_NUMBER_OF_TRANSFORMS,(MKL_LONG)K);
        DftiSetValue(dl,DFTI_INPUT_STRIDES,st_l); DftiSetValue(dl,DFTI_OUTPUT_STRIDES,st_l);
        DftiSetValue(dl,DFTI_INPUT_DISTANCE,(MKL_LONG)1); DftiSetValue(dl,DFTI_OUTPUT_DISTANCE,(MKL_LONG)1);
        DftiCommitDescriptor(dl);
        DftiCreateDescriptor(&dc,DFTI_DOUBLE,DFTI_COMPLEX,1,(MKL_LONG)N);
        DftiSetValue(dc,DFTI_NUMBER_OF_TRANSFORMS,(MKL_LONG)K);
        DftiSetValue(dc,DFTI_INPUT_STRIDES,st_c); DftiSetValue(dc,DFTI_OUTPUT_STRIDES,st_c);
        DftiSetValue(dc,DFTI_INPUT_DISTANCE,(MKL_LONG)N); DftiSetValue(dc,DFTI_OUTPUT_DISTANCE,(MKL_LONG)N);
        DftiCommitDescriptor(dc);
        memcpy(z,z0,n*16); DftiComputeForward(dl,z); DftiComputeBackward(dl,z);
        double em=maxrel(z,z0,2*n,(double)N);
        int reps=(int)(4e6/(double)n); if(reps<2)reps=2; if(reps>400)reps=400;
        stride_plan_t *pe = vfft_proto_exhaustive_plan(N,K,&reg,0);
        double tv=1e18,tl=1e18,tc=1e18,tvf=1e18,tlf=1e18,tsp=1e18,tse=1e18;
        for(int t=0;t<5;t++){
            double t0;
            t0=(double)__rdtsc();
            for(int r=0;r<reps;r++){
                if(fold_in) vfft_proto_execute_fwd_ilin_jit(h.plan,z,cr,ci,K,h.exec_fwd);
                else { vfft_il2sp(z,cr,ci,n); vfft_proto_plan_execute_fwd(&h,cr,ci); }
                if(fold_out) vfft_proto_execute_bwd_ilout_jit(h.plan,cr,ci,z,K,h.exec_bwd);
                else { vfft_proto_plan_execute_bwd(&h,cr,ci); vfft_sp2il(cr,ci,z,n); }
            }
            now_min(&tv,t0,reps);
            t0=(double)__rdtsc();
            for(int r=0;r<reps;r++){ DftiComputeForward(dl,z); DftiComputeBackward(dl,z); }
            now_min(&tl,t0,reps);
            t0=(double)__rdtsc();
            for(int r=0;r<reps;r++){ vfft_proto_plan_execute_fwd(&h,cr,ci); vfft_proto_plan_execute_bwd(&h,cr,ci); }
            now_min(&tsp,t0,reps);
            t0=(double)__rdtsc();
            for(int r=0;r<reps;r++){ vfft_proto_execute_fwd(pe,cr,ci,K); vfft_proto_execute_bwd(pe,cr,ci,K); }
            now_min(&tse,t0,reps);
            t0=(double)__rdtsc();
            for(int r=0;r<reps;r++){ DftiComputeForward(dc,zc); DftiComputeBackward(dc,zc); }
            now_min(&tc,t0,reps);
            t0=(double)__rdtsc();
            for(int r=0;r<reps;r++){
                if(fold_in) vfft_proto_execute_fwd_ilin_jit(h.plan,z,cr,ci,K,h.exec_fwd);
                else { vfft_il2sp(z,cr,ci,n); vfft_proto_plan_execute_fwd(&h,cr,ci); }
                vfft_sp2il(cr,ci,z,n);
            }
            now_min(&tvf,t0,reps);
            t0=(double)__rdtsc();
            for(int r=0;r<reps;r++) DftiComputeForward(dl,z);
            now_min(&tlf,t0,reps);
        }
        printf("%6d %4zu %3s/%-3s | %10.0f %10.0f %6.2fx | %10.0f %6.2fx | %6.2fx | split-orch %.0f split-exh %.0f [err v %.1e m %.1e]\n",
            N,K, fold_in?"in":"IN!", fold_out?"out":"OUT!",
            tv, tl, tl/tv, tc, tc/tv, tlf/tvf, tsp, tse, e, em);
        DftiFreeDescriptor(&dl); DftiFreeDescriptor(&dc);
        AFREE(z);AFREE(z0);AFREE(zc);AFREE(cr);AFREE(ci);
        vfft_proto_handle_destroy(&h);
    }
    return 0;
}
