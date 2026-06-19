/* bench_oop_dp.c — validate the DP-planner-backed OOP c2c path (oop_dp.h).
 *
 * For each cell: build the OOP plan via vfft_oop_plan_create_dp (LEAF/BAILEY2
 * rule, else DP-measured MODEB), roundtrip-validate (fwd+bwd == N*x, order-
 * agnostic so it works for scrambled MODEB), and time vs MKL OOP split.
 * For the N=1024 K=256 loss cell, ALSO force DP-MODEB (skip BAILEY2) to see if
 * the DP general-N plan beats the aliasing-masked BAILEY2 (0.74x earlier).
 *
 * Build: cd build_tuned && python build.py --src benches/bench_oop_dp.c --mkl --compile
 * Run  : PATH += MKL bin + C:\mingw152\mingw64\bin.
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mkl_dfti.h>
#include <mkl_service.h>
#include "core/executor.h"
#include "core/env.h"
#include "core/oop_dp.h"

#define PIN_CORE 2
#define BEST_OF  15
#if defined(_WIN32)
#include <malloc.h>
#define AALLOC(n) _aligned_malloc((n),64)
#define AFREE(p)  _aligned_free(p)
#else
#define AALLOC(n) aligned_alloc(64,(n))
#define AFREE(p)  free(p)
#endif

static int reps_for(size_t t){int r=(int)(8e6/(t+1)); if(r<20)r=20; if(r>50000)r=50000; return r;}
static const char *kindname(const vfft_oop_plan_t*p){
    return p->kind==VFFT_OOP_KIND_LEAF?"LEAF":p->kind==VFFT_OOP_KIND_BAILEY2?"BAILEY2":"MODEB";
}
static void facstr(const vfft_oop_plan_t*p,char*buf,size_t n){
    if(p->kind==VFFT_OOP_KIND_BAILEY2){snprintf(buf,n,"%dx%d",p->R1,p->R2);return;}
    if(p->kind==VFFT_OOP_KIND_MODEB&&p->mb){size_t o=0;
        for(int s=0;s<p->mb->num_stages;s++) o+=(size_t)snprintf(buf+o,n-o,"%s%d",s?",":"",p->mb->factors[s]);
        return;}
    buf[0]='\0';
}
/* Correctness per kind:
 *   LEAF/BAILEY2 (natural order): roundtrip fwd+bwd == N*x.
 *   MODEB (scrambled order): bit-exact vs the in-place stride dataflow on the
 *     same plan (the established MODEB validation — the swap-trip can't recover
 *     scrambled output). */
static double correctness(vfft_oop_plan_t*p,int N,size_t K){
    size_t T=(size_t)N*K;
    double *sr=AALLOC(T*8),*si=AALLOC(T*8),*dr=AALLOC(T*8),*di=AALLOC(T*8);
    srand(91+N);
    for(size_t i=0;i<T;i++){sr[i]=(double)rand()/RAND_MAX-0.5; si[i]=(double)rand()/RAND_MAX-0.5;}
    vfft_oop_execute_fwd(p,sr,si,dr,di);
    double me=0;
    if(p->kind==VFFT_OOP_KIND_MODEB && p->mb){
        double *tr=AALLOC(T*8),*ti=AALLOC(T*8);
        memcpy(tr,sr,T*8); memcpy(ti,si,T*8);
        vfft_proto_execute_fwd(p->mb,tr,ti,K);   /* in-place reference, same scramble */
        for(size_t i=0;i<T;i++){double a=fabs(dr[i]-tr[i]),b=fabs(di[i]-ti[i]); if(a>me)me=a; if(b>me)me=b;}
        AFREE(tr);AFREE(ti);
    } else {
        double *er=AALLOC(T*8),*ei=AALLOC(T*8);
        vfft_oop_execute_bwd(p,dr,di,er,ei);
        for(size_t i=0;i<T;i++){double a=fabs(er[i]/(double)N-sr[i]),b=fabs(ei[i]/(double)N-si[i]); if(a>me)me=a; if(b>me)me=b;}
        AFREE(er);AFREE(ei);
    }
    AFREE(sr);AFREE(si);AFREE(dr);AFREE(di);
    return me;
}
static double time_oop(vfft_oop_plan_t*p,const double*sr,const double*si,double*dr,double*di,size_t T){
    for(int w=0;w<5;w++) vfft_oop_execute_fwd(p,sr,si,dr,di);
    int reps=reps_for(T); double best=1e18;
    for(int t=0;t<BEST_OF;t++){double t0=vfft_proto_now_ns();
        for(int i=0;i<reps;i++) vfft_oop_execute_fwd(p,sr,si,dr,di);
        double ns=(vfft_proto_now_ns()-t0)/reps; if(ns<best)best=ns;}
    return best;
}
static double time_mkl(DFTI_DESCRIPTOR_HANDLE d,const double*sr,const double*si,double*mr,double*mi,size_t T){
    for(int w=0;w<5;w++) DftiComputeForward(d,(void*)sr,(void*)si,mr,mi);
    int reps=reps_for(T); double best=1e18;
    for(int t=0;t<BEST_OF;t++){double t0=vfft_proto_now_ns();
        for(int i=0;i<reps;i++) DftiComputeForward(d,(void*)sr,(void*)si,mr,mi);
        double ns=(vfft_proto_now_ns()-t0)/reps; if(ns<best)best=ns;}
    return best;
}

static DFTI_DESCRIPTOR_HANDLE mkl_make(int N,size_t K){
    DFTI_DESCRIPTOR_HANDLE d=0; MKL_LONG str[2]={0,(MKL_LONG)K};
    if(DftiCreateDescriptor(&d,DFTI_DOUBLE,DFTI_COMPLEX,1,(MKL_LONG)N)!=DFTI_NO_ERROR) return 0;
    DftiSetValue(d,DFTI_COMPLEX_STORAGE,DFTI_REAL_REAL);
    DftiSetValue(d,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
    DftiSetValue(d,DFTI_NUMBER_OF_TRANSFORMS,(MKL_LONG)K);
    DftiSetValue(d,DFTI_INPUT_DISTANCE,1); DftiSetValue(d,DFTI_OUTPUT_DISTANCE,1);
    DftiSetValue(d,DFTI_INPUT_STRIDES,str); DftiSetValue(d,DFTI_OUTPUT_STRIDES,str);
    if(DftiCommitDescriptor(d)!=DFTI_NO_ERROR){DftiFreeDescriptor(&d);return 0;}
    return d;
}

static void run_cell(int N,size_t K,vfft_proto_registry_t*reg,int force_modeb_too){
    size_t T=(size_t)N*K;
    vfft_proto_dp_context_t ctx; vfft_proto_dp_init(&ctx,K,N);   /* MEASURE mode default */

    vfft_oop_plan_t *p = vfft_oop_plan_create_dp(N,K,&ctx,reg);
    if(!p){ printf("  N=%-5d K=%-4zu  dp plan NULL\n",N,K); vfft_proto_dp_destroy(&ctx); return; }

    double rt = correctness(p,N,K);
    char fs[96]; facstr(p,fs,sizeof fs);

    double *sr=AALLOC(T*8),*si=AALLOC(T*8),*dr=AALLOC(T*8),*di=AALLOC(T*8),*mr=AALLOC(T*8),*mi=AALLOC(T*8);
    srand(53+N); for(size_t i=0;i<T;i++){sr[i]=(double)rand()/RAND_MAX-0.5; si[i]=(double)rand()/RAND_MAX-0.5;}
    DFTI_DESCRIPTOR_HANDLE d=mkl_make(N,K);
    double v=time_oop(p,sr,si,dr,di,T);
    double m=d?time_mkl(d,sr,si,mr,mi,T):0;
    printf("  N=%-5d K=%-4zu  dp->%-7s %-14s rt=%.1e | vfft %9.1f | mkl %9.1f | vs MKL %.3f\n",
           N,K,kindname(p),fs,rt,v,m,m>0?m/v:0);

    if(force_modeb_too){
        vfft_oop_plan_t *pm = vfft_oop_plan_create_dp_modeb(N,K,&ctx,reg);
        if(pm){
            double rtm=correctness(pm,N,K); char fm[96]; facstr(pm,fm,sizeof fm);
            double vm=time_oop(pm,sr,si,dr,di,T);
            printf("        forced DP-MODEB (%s) rt=%.1e | vfft %9.1f | vs MKL %.3f  (vs the rule's %s above)\n",
                   fm,rtm,vm,m>0?m/vm:0,kindname(p));
            vfft_oop_plan_destroy(pm);
        }
    }
    if(d)DftiFreeDescriptor(&d);
    AFREE(sr);AFREE(si);AFREE(dr);AFREE(di);AFREE(mr);AFREE(mi);
    vfft_oop_plan_destroy(p);
    vfft_proto_dp_destroy(&ctx);
}

int main(void){
    stride_env_init();
    if(stride_pin_thread(PIN_CORE)!=0) fprintf(stderr,"warn pin\n");
    mkl_set_num_threads(1);
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    printf("== DP-backed OOP c2c vs MKL (NOT_INPLACE split, ST, cpu%d, best-of-%d) ==\n",PIN_CORE,BEST_OF);
    printf("# rt = roundtrip err (fwd+bwd == N*x). dp-> = kind the DP path chose.\n");
    run_cell(1024, 256, &reg, 1);   /* the BAILEY2 loss cell — also force DP-MODEB */
    run_cell(4096, 256, &reg, 0);   /* general-N MODEB (wisdom gave 1.66x) */
    run_cell(2310, 32,  &reg, 0);   /* prime-factored MODEB (wisdom gave 2.09x) */
    run_cell(1024, 120, &reg, 1);   /* BAILEY2 32x32 winner at this K — DP-MODEB compare */
    printf("# (DP planning MEASURES at plan time; one-time cost amortized by the ctx cache.)\n");
    return 0;
}
