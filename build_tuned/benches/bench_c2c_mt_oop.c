/* bench_c2c_mt_oop.c — OOP c2c multithreading via pool K-split.
 * The OOP transform is embarrassingly K-parallel (lanes independent): split the
 * K batch across the thread pool, each thread runs the kind-appropriate slice
 * (LEAF / BAILEY2 / MODEB). Scaling T=1,2,4,8; MT output verified == T=1.
 *
 * Build: cd build_tuned && python build.py --src benches/bench_c2c_mt_oop.c --compile
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "core/oop_auto.h"     /* OOP plan + leaf/t1p + vfft_proto_execute_fwd_oop */
#include "core/threads.h"      /* pool: dispatch/wait, set/get num threads */
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
               (size_t)R2*K,1,(size_t)R2*K,1,S);
    } else {
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
    /* K=4096 masks every BAILEY2 pair (alias period), so give MODEB radix-4
     * factors. MODEB (stride-core OOP adapter) is the K-splittable MT path. */
    int factors[16], nf=0, n=N;
    while(n%4==0){factors[nf++]=4; n/=4;}
    if(n>1) factors[nf++]=n;
    vfft_oop_plan_t*op=vfft_oop_plan_create(N,K,factors,nf,reg);
    if(!op){printf("N=%-5d K=%-5zu OOP plan NULL\n",N,K);return;}
    const char*kn=op->kind==VFFT_OOP_KIND_LEAF?"LEAF":op->kind==VFFT_OOP_KIND_BAILEY2?"BAILEY2":"MODEB";
    char fac[16]=""; if(op->kind==VFFT_OOP_KIND_BAILEY2)snprintf(fac,sizeof fac,"%dx%d",op->R1,op->R2);
    double *sr=AALLOC(NK*8),*si=AALLOC(NK*8),*dr=AALLOC(NK*8),*di=AALLOC(NK*8),*ref=AALLOC(NK*8);
    srand(7+N); for(size_t i=0;i<NK;i++){sr[i]=(double)rand()/RAND_MAX-0.5; si[i]=(double)rand()/RAND_MAX-0.5;}
    int Ts[]={1,2,4,8}; double t[4]; double mt_err=0;
    for(int ti=0;ti<4;ti++){
        stride_set_num_threads(Ts[ti]);
        for(int w=0;w<2;w++) oop_mt_fwd(op,sr,si,dr,di);
        int reps=reps_for(NK); double b=1e18;
        for(int r=0;r<BEST_OF;r++){double t0=now_c(); for(int i=0;i<reps;i++) oop_mt_fwd(op,sr,si,dr,di); double v=(now_c()-t0)/reps; if(v<b)b=v;}
        t[ti]=b;
        if(ti==0)memcpy(ref,dr,NK*8); else {double e=0;for(size_t i=0;i<NK;i++){double a=dr[i]-ref[i];if(a<0)a=-a;if(a>e)e=a;} if(e>mt_err)mt_err=e;}
    }
    stride_set_num_threads(1);
    printf("N=%-5d K=%-5zu OOP[%s%s]  T1=%.0f  scale:",N,K,kn,fac,t[0]);
    for(int ti=0;ti<4;ti++)printf(" T%d=%.2fx",Ts[ti],t[0]/t[ti]);
    printf("  (MT==T1 err %.0e)\n",mt_err); fflush(stdout);
    AFREE(sr);AFREE(si);AFREE(dr);AFREE(di);AFREE(ref); vfft_oop_plan_destroy(op);
}

int main(void){
    stride_env_init();
    if(stride_pin_thread(PIN_CORE)!=0) fprintf(stderr,"warn pin\n");
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    printf("== OOP c2c MT (pool K-split), scaling vs T=1 ==\n");
    run(256,4096,&reg);
    run(512,4096,&reg);
    run(1024,4096,&reg);
    return 0;
}
