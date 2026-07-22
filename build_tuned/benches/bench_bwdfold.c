/* bwd/fwd fold bench: sweep vs core-il2il vs jit-il2il vs MKL (IL lane). */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "prime_dispatch.h"
#include "plan_orchestrator.h"
#include "il_execute.h"
#include "jit_runtime.h"
#include "generator/generated/registry.h"
#include <mkl_dfti.h>

static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e6 + t.tv_nsec*1e-3; }
static int cmpd(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b;
    return (x>y)-(x<y); }
static void fillr(double*p,size_t n,int s){srand(s);for(size_t i=0;i<n;i++)p[i]=2.0*rand()/RAND_MAX-1;}

#define TRIALS 11
static double med_of(double*v,int n){ qsort(v,n,sizeof(double),cmpd); return v[n/2]; }

int main(void){
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    vfft_proto_wisdom_t *w=calloc(1,sizeof *w);
    if(vfft_proto_wisdom_load(w,"/mnt/user-data/uploads/spike_wisdom.txt")){puts("wload");return 1;}
    struct { int N; size_t K; int L; } cs[]={{1024,4,400},{100,4,4000},{1000,4,300}};
    printf("%-10s %-14s %9s %9s\n","cell","arm","us/iter","note");
    for(int c=0;c<3;c++){
        int N=cs[c].N; size_t K=cs[c].K, NK=(size_t)N*K; int L=cs[c].L;
        vfft_proto_handle_t h;
        if(vfft_proto_plan(&h,N,K,VFFT_PROTO_WISDOM_ONLY,&reg,w,NULL)){printf("plan fail %d\n",N);continue;}
        int dif=h.plan->use_dif_forward;
        vfft_proto_exec_range_fn rff=vfft_proto_plan_jit_fwd_range(h.plan);
        vfft_proto_exec_range_fn rfb=vfft_proto_plan_jit_bwd_range(h.plan);
        printf("[%d,%zu] stages=%d dif=%d jit(fwd=%s,bwd=%s)\n",N,K,h.plan->num_stages,dif,
               rff?"ok":"NIL",rfb?"ok":"NIL");
        double *zi=aligned_alloc(64,2*NK*8),*zo=aligned_alloc(64,2*NK*8);
        double *r=aligned_alloc(64,NK*8),*im=aligned_alloc(64,NK*8);
        fillr(zi,2*NK,42+c);
        MKL_Complex16 *mz=aligned_alloc(64,NK*sizeof(MKL_Complex16));
        memcpy(mz,zi,2*NK*8);
        DFTI_DESCRIPTOR_HANDLE mh; 
        DftiCreateDescriptor(&mh,DFTI_DOUBLE,DFTI_COMPLEX,1,(MKL_LONG)N);
        DftiSetValue(mh,DFTI_NUMBER_OF_TRANSFORMS,(MKL_LONG)K);
        DftiSetValue(mh,DFTI_INPUT_DISTANCE,(MKL_LONG)N);
        DftiSetValue(mh,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
        DftiSetValue(mh,DFTI_OUTPUT_DISTANCE,(MKL_LONG)N);
        DftiCommitDescriptor(mh);
        MKL_Complex16 *mo=aligned_alloc(64,NK*sizeof(MKL_Complex16));
        double tv[TRIALS];
        #define MEASURE(label,STMT) do{ \
            for(int wu=0;wu<3;wu++){STMT;} \
            for(int t=0;t<TRIALS;t++){ double t0=now_us(); \
                for(int it=0;it<L;it++){STMT;} \
                tv[t]=(now_us()-t0)/L; } \
            printf("  %-12s %9.3f\n",label,med_of(tv,TRIALS)); }while(0)
        /* ---- BWD ---- */
        MEASURE("b.sweep", {
            for(size_t i=0;i<NK;i++){r[i]=zi[2*i];im[i]=zi[2*i+1];}
            if(dif) vfft_proto_execute_bwd_generic_dif(h.plan,r,im,K);
            else    vfft_proto_execute_bwd_generic(h.plan,r,im,K);
            for(size_t i=0;i<NK;i++){zo[2*i]=r[i];zo[2*i+1]=im[i];} });
        MEASURE("b.core", { vfft_proto_execute_bwd_il2il_core(h.plan,zi,r,im,zo,K); });
        if(rfb) MEASURE("b.jit", { vfft_proto_execute_bwd_il2il_jit(h.plan,zi,r,im,zo,K,rfb); });
        MEASURE("b.mkl", { DftiComputeBackward(mh,mz,mo); });
        /* ---- FWD ---- */
        MEASURE("f.core", { vfft_proto_execute_fwd_il2il_core(h.plan,zi,r,im,zo,K); });
        if(rff) MEASURE("f.jit", { vfft_proto_execute_fwd_il2il_jit(h.plan,zi,r,im,zo,K,rff); });
        MEASURE("f.mkl", { DftiComputeForward(mh,mz,mo); });
        DftiFreeDescriptor(&mh);
        free(zi);free(zo);free(r);free(im);free(mz);free(mo);
    }
    return 0;
}
