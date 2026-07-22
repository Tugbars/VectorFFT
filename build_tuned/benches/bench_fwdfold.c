/* Fwd z->z fold gain: il2il vs (ilin+sp2il) vs MKL-IL lane. Matched-ISA. */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "prime_dispatch.h"
#include "plan_orchestrator.h"
#include "il_layout.h"
#include "il_execute.h"
#include "generator/generated/registry.h"
#include "mkl_dfti.h"

static double now_ns(void){ return vfft_proto_now_ns(); }
static int cmpd(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return x<y?-1:x>y;}
static double med5(double*v){qsort(v,5,8,cmpd);return v[2];}

int main(int argc,char**argv){
    int N=argc>1?atoi(argv[1]):1024; size_t K=argc>2?(size_t)atoi(argv[2]):4;
    size_t NK=(size_t)N*K;
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    vfft_proto_wisdom_t *w=calloc(1,sizeof *w);
    if(vfft_proto_wisdom_load(w,"/mnt/user-data/uploads/spike_wisdom.txt")){puts("wload");return 1;}
    vfft_proto_handle_t h;
    if(vfft_proto_plan(&h,N,K,VFFT_PROTO_WISDOM_ONLY,&reg,w,NULL)){puts("plan");return 1;}
    printf("(%d,%zu) dif=%d stages=%d\n",N,K,h.plan->use_dif_forward,h.plan->num_stages);
    double *z0=aligned_alloc(64,2*NK*8);
    double *zA=aligned_alloc(64,2*NK*8),*zB=aligned_alloc(64,2*NK*8),*zC=aligned_alloc(64,2*NK*8);
    double *wr=aligned_alloc(64,NK*8),*wi=aligned_alloc(64,NK*8);
    srand(7); for(size_t i=0;i<2*NK;i++) z0[i]=2.0*rand()/RAND_MAX-1;
    /* MKL lane-major descriptor */
    DFTI_DESCRIPTOR_HANDLE d=NULL;
    DftiCreateDescriptor(&d,DFTI_DOUBLE,DFTI_COMPLEX,1,(MKL_LONG)N);
    DftiSetValue(d,DFTI_PLACEMENT,DFTI_INPLACE);
    DftiSetValue(d,DFTI_NUMBER_OF_TRANSFORMS,(MKL_LONG)K);
    { MKL_LONG str[2]={0,(MKL_LONG)K};
      DftiSetValue(d,DFTI_INPUT_DISTANCE,1); DftiSetValue(d,DFTI_OUTPUT_DISTANCE,1);
      DftiSetValue(d,DFTI_INPUT_STRIDES,str); DftiSetValue(d,DFTI_OUTPUT_STRIDES,str); }
    if(DftiCommitDescriptor(d)!=DFTI_NO_ERROR){puts("mkl commit");return 1;}
    /* verify */
    memcpy(zA,z0,2*NK*8);
    if(vfft_proto_execute_fwd_il2il_core(h.plan,zA,wr,wi,zA,K)){puts("il2il REJECT");return 1;}
    memcpy(zC,z0,2*NK*8); DftiComputeForward(d,zC);
    double mx=0,sc=0; for(size_t i=0;i<2*NK;i++){double e=fabs(zA[i]-zC[i]);if(e>mx)mx=e;
        double v=fabs(zC[i]); if(v>sc)sc=v;}
    printf("verify il2il vs MKL maxrel=%.2e\n",mx/sc);
    /* old path check */
    memcpy(zB,z0,2*NK*8);
    if(vfft_proto_execute_fwd_ilin_core(h.plan,zB,wr,wi,K,h.exec_fwd)){puts("ilin REJECT");return 1;}
    vfft_sp2il(wr,wi,zB,NK);
    mx=0; for(size_t i=0;i<2*NK;i++){double e=fabs(zB[i]-zA[i]);if(e>mx)mx=e;}
    printf("verify old vs il2il (same order) maxrel=%.2e %s\n",mx/sc,mx/sc<1e-12?"OK":"**FAIL**");
    int reps = NK<=8192 ? 2000 : 400;
    double tA[5],tB[5],tC[5];
    for(int s=0;s<5;s++){
        /* A: il2il */
        memcpy(zA,z0,2*NK*8);
        for(int i=0;i<reps/10;i++) vfft_proto_execute_fwd_il2il_core(h.plan,zA,wr,wi,zA,K);
        double t0=now_ns();
        for(int i=0;i<reps;i++) vfft_proto_execute_fwd_il2il_core(h.plan,zA,wr,wi,zA,K);
        tA[s]=(now_ns()-t0)/reps;
        /* B: ilin + sp2il */
        memcpy(zB,z0,2*NK*8);
        for(int i=0;i<reps/10;i++){vfft_proto_execute_fwd_ilin_core(h.plan,zB,wr,wi,K,h.exec_fwd);vfft_sp2il(wr,wi,zB,NK);}
        t0=now_ns();
        for(int i=0;i<reps;i++){vfft_proto_execute_fwd_ilin_core(h.plan,zB,wr,wi,K,h.exec_fwd);vfft_sp2il(wr,wi,zB,NK);}
        tB[s]=(now_ns()-t0)/reps;
        /* C: MKL */
        memcpy(zC,z0,2*NK*8);
        for(int i=0;i<reps/10;i++) DftiComputeForward(d,zC);
        t0=now_ns();
        for(int i=0;i<reps;i++) DftiComputeForward(d,zC);
        tC[s]=(now_ns()-t0)/reps;
    }
    double a=med5(tA),b=med5(tB),c=med5(tC);
    printf("il2il    %8.0f ns\nold-path %8.0f ns\nMKL-lane %8.0f ns\n",a,b,c);
    printf("fold gain (old/new)   %.3fx\n",b/a);
    printf("vs MKL-IL-lane (new)  %.3fx   (old was %.3fx)\n",c/a,c/b);
    return 0;
}
