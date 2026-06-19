/* bench_c2c_mt.c — multithreaded c2c: in-place (built-in K-split) + OOP
 * (K-split across the same pool). Scaling T=1,2,4,8; MT output verified
 * bit-identical to T=1.
 *
 *   in-place: stride_execute_fwd (stride_executor.h) — K-split when K/T>=256,
 *             else group-parallel. The production-ported MT path.
 *   OOP     : oop_mt_fwd (below) — split the K batch across the pool; each
 *             thread runs the kind-appropriate slice (LEAF / BAILEY2 / MODEB).
 *             The OOP transform is embarrassingly K-parallel (lanes independent).
 *
 * Build: cd build_tuned && python build.py --src benches/bench_c2c_mt.c --compile
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "core/stride_executor.h"   /* MT in-place stride_execute_fwd + pool slice */
#include "core/planner.h"
#include "core/threads.h"
#include "core/oop_auto.h"          /* OOP plan + leaf/t1p; pulls oop_execute (MODEB) */
#include "core/env.h"
#include "generator/generated/registry.h"

#define PIN_CORE 2
#define BEST_OF  11
#if defined(_WIN32)
#include <malloc.h>
#define AALLOC(n) _aligned_malloc((n),64)
#define AFREE(p)  _aligned_free(p)
#else
#define AALLOC(n) aligned_alloc(64,(n))
#define AFREE(p)  free(p)
#endif
#include <x86intrin.h>
static inline double now_c(void){ return (double)__rdtsc(); }
static int reps_for(size_t t){int r=(int)(2e8/(t+1)); if(r<3)r=3; if(r>2000)r=2000; return r;}

/* ── OOP K-split: run one lane-slice [k0,k0+S) of the OOP transform ── */
static void oop_slice_run(const vfft_oop_plan_t*p,const double*sr,const double*si,
                          double*dr,double*di,size_t k0,size_t S){
    size_t K=p->K;
    if(p->kind==VFFT_OOP_KIND_LEAF){
        p->leaf(sr+k0,si+k0,dr+k0,di+k0,0,0,K,1,K,1,S);
    } else if(p->kind==VFFT_OOP_KIND_BAILEY2){
        int R1=p->R1,R2=p->R2;
        for(int n1=0;n1<R1;n1++)
            p->leaf(sr+(size_t)n1*K+k0, si+(size_t)n1*K+k0,
                    dr+(size_t)n1*R2*K+k0, di+(size_t)n1*R2*K+k0,
                    0,0,(size_t)R1*K,1,K,1,S);
        p->t1p(dr+k0,di+k0,dr+k0,di+k0,p->Qr,p->Qi,
               (size_t)R2*K,1,(size_t)R2*K,1,S);   /* twiddle table is lane-replicated */
    } else { /* MODEB */
        vfft_proto_execute_fwd_oop(p->mb, sr+k0, si+k0, dr+k0, di+k0, S);
    }
}
typedef struct { const vfft_oop_plan_t*p; const double*sr,*si; double*dr,*di; size_t k0,S; } oop_arg_t;
static void oop_tramp(void*a){ oop_arg_t*x=(oop_arg_t*)a; oop_slice_run(x->p,x->sr,x->si,x->dr,x->di,x->k0,x->S); }
static void oop_mt_fwd(const vfft_oop_plan_t*p,const double*sr,const double*si,double*dr,double*di){
    size_t K=p->K; int T=stride_get_num_threads();
    if(T>_stride_pool_size+1) T=_stride_pool_size+1;
    if(T<=1 || K<8){ oop_slice_run(p,sr,si,dr,di,0,K); return; }
    size_t S=((K/(size_t)T)+7)&~(size_t)7;
    oop_arg_t args[64]; int nd=0;
    for(int t=1;t<T && t<=_stride_pool_size;t++){
        size_t k0=(size_t)t*S; if(k0>=K)break; size_t ke=k0+S; if(ke>K)ke=K;
        args[nd]=(oop_arg_t){p,sr,si,dr,di,k0,ke-k0};
        _stride_pool_dispatch(&_stride_workers[nd],oop_tramp,&args[nd]); nd++;
    }
    size_t s0=S<K?S:K; oop_slice_run(p,sr,si,dr,di,0,s0);
    if(nd) _stride_pool_wait_all();
}

static void run(int N,size_t K,vfft_proto_registry_t*reg){
    size_t NK=(size_t)N*K;
    stride_plan_t*ip=vfft_proto_auto_plan(N,K,reg,NULL);
    vfft_oop_plan_t*op=vfft_oop_plan_create(N,K,NULL,0,reg);
    if(!ip||!op){printf("N=%-5d K=%-5zu  plan NULL (ip=%p op=%p)\n",N,K,(void*)ip,(void*)op);return;}
    const char*okind=op->kind==VFFT_OOP_KIND_LEAF?"LEAF":op->kind==VFFT_OOP_KIND_BAILEY2?"BAILEY2":"MODEB";

    double *re=AALLOC(NK*8),*im=AALLOC(NK*8),*x0=AALLOC(NK*8),*xi0=AALLOC(NK*8);
    double *sr=AALLOC(NK*8),*si=AALLOC(NK*8),*dr=AALLOC(NK*8),*di=AALLOC(NK*8),*ref=AALLOC(NK*8);
    srand(7+N); for(size_t i=0;i<NK;i++){x0[i]=(double)rand()/RAND_MAX-0.5; xi0[i]=(double)rand()/RAND_MAX-0.5;}
    for(size_t i=0;i<NK;i++){sr[i]=x0[i]; si[i]=xi0[i];}

    int Ts[]={1,2,4,8};
    printf("N=%-5d K=%-5zu  (in-place vs OOP[%s])  T:   ",N,K,okind);
    /* serial reference (in-place, T=1) for correctness */
    double ipt[4]={0},opt[4]={0}; double ip_ref_err=0, op_ref_err=0;
    double *ipref=AALLOC(NK*8);
    for(int ti=0;ti<4;ti++){
        int T=Ts[ti]; stride_set_num_threads(T);
        /* in-place */
        int reps=reps_for(NK); double bi=1e18;
        for(int w=0;w<2;w++){memcpy(re,x0,NK*8);memcpy(im,xi0,NK*8); stride_execute_fwd(ip,re,im);}
        for(int r=0;r<BEST_OF;r++){memcpy(re,x0,NK*8);memcpy(im,xi0,NK*8);
            double t0=now_c(); stride_execute_fwd(ip,re,im); double v=now_c()-t0; if(v<bi)bi=v;}
        ipt[ti]=bi;
        if(ti==0){memcpy(ipref,re,NK*8);} else {double e=0;for(size_t i=0;i<NK;i++){double a=fabs(re[i]-ipref[i]);if(a>e)e=a;} if(e>ip_ref_err)ip_ref_err=e;}
        /* OOP */
        double bo=1e18;
        for(int w=0;w<2;w++) oop_mt_fwd(op,sr,si,dr,di);
        for(int r=0;r<BEST_OF;r++){double t0=now_c(); oop_mt_fwd(op,sr,si,dr,di); double v=now_c()-t0; if(v<bo)bo=v;}
        opt[ti]=bo;
        if(ti==0){memcpy(ref,dr,NK*8);} else {double e=0;for(size_t i=0;i<NK;i++){double a=fabs(dr[i]-ref[i]);if(a>e)e=a;} if(e>op_ref_err)op_ref_err=e;}
    }
    stride_set_num_threads(1);
    printf("\n  in-place cyc:");for(int ti=0;ti<4;ti++)printf(" T%d=%.2fx",Ts[ti],ipt[0]/ipt[ti]);
    printf("   (MT==T1 err %.0e)\n",ip_ref_err);
    printf("  OOP      cyc:");for(int ti=0;ti<4;ti++)printf(" T%d=%.2fx",Ts[ti],opt[0]/opt[ti]);
    printf("   (MT==T1 err %.0e)\n",op_ref_err);
    fflush(stdout);

    AFREE(re);AFREE(im);AFREE(x0);AFREE(xi0);AFREE(sr);AFREE(si);AFREE(dr);AFREE(di);AFREE(ref);AFREE(ipref);
    stride_plan_destroy(ip); vfft_oop_plan_destroy(op);
}

int main(void){
    stride_env_init();
    if(stride_pin_thread(PIN_CORE)!=0) fprintf(stderr,"warn pin\n");
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    printf("== c2c multithreading: in-place (K-split) + OOP (pool K-split), scaling vs T=1 ==\n");
    run(256,4096,&reg);
    run(1024,4096,&reg);
    return 0;
}
