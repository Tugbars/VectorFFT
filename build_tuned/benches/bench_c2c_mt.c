/* bench_c2c_mt.c — c2c multithreading, in-place + OOP, via pool K-split.
 * Both transforms are embarrassingly K-parallel: split the K batch across the
 * thread pool, each thread runs a lane-slice. (Same mechanism as production's
 * in-place K-split; done here over vfft_proto_execute_fwd / the OOP kinds to
 * avoid the stride_executor.h-vs-executor.h header clash.)
 *
 * Tests K=4096 (= 32KB L1-alias period, pathological) vs K=4104 (non-aliasing)
 * to separate true MT scaling from the lane-batched-layout aliasing catastrophe.
 *
 * Build: cd build_tuned && python build.py --src benches/bench_c2c_mt.c --compile
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "executor.h"     /* vfft_proto_execute_fwd (in-place, per-slice) */
#include "planner.h"
#include "threads.h"      /* pool */
#include "oop_auto.h"     /* OOP plan + leaf/t1p + vfft_proto_execute_fwd_oop */
#include "env.h"
#include "generator/generated/registry.h"

#define PIN_CORE 0   /* pool pins workers to cores 1..T-1; caller must be core 0 */
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

/* ---- in-place K-split ---- */
typedef struct { const stride_plan_t*p; double*re,*im; size_t k0,S; } ip_arg_t;
static void ip_tramp(void*a){ip_arg_t*x=(ip_arg_t*)a; vfft_proto_execute_fwd(x->p,x->re+x->k0,x->im+x->k0,x->S);}
static void inplace_mt(const stride_plan_t*p,double*re,double*im){
    size_t K=p->K; int T=stride_get_num_threads();
    if(T>_stride_pool_size+1)T=_stride_pool_size+1;
    if(T<=1||K<8){vfft_proto_execute_fwd(p,re,im,K);return;}
    size_t S=((K/(size_t)T)+7)&~(size_t)7; ip_arg_t a[64]; int nd=0;
    for(int t=1;t<T&&t<=_stride_pool_size;t++){size_t k0=(size_t)t*S;if(k0>=K)break;size_t ke=k0+S;if(ke>K)ke=K;
        a[nd]=(ip_arg_t){p,re,im,k0,ke-k0};_stride_pool_dispatch(&_stride_workers[nd],ip_tramp,&a[nd]);nd++;}
    size_t s0=S<K?S:K; vfft_proto_execute_fwd(p,re,im,s0); if(nd)_stride_pool_wait_all();
}
/* ---- OOP K-split ---- */
static void oop_slice(const vfft_oop_plan_t*p,const double*sr,const double*si,double*dr,double*di,size_t k0,size_t S){
    size_t K=p->K;
    if(p->kind==VFFT_OOP_KIND_LEAF) p->leaf(sr+k0,si+k0,dr+k0,di+k0,0,0,K,1,K,1,S);
    else if(p->kind==VFFT_OOP_KIND_BAILEY2){int R1=p->R1,R2=p->R2;
        for(int n1=0;n1<R1;n1++)p->leaf(sr+(size_t)n1*K+k0,si+(size_t)n1*K+k0,dr+(size_t)n1*R2*K+k0,di+(size_t)n1*R2*K+k0,0,0,(size_t)R1*K,1,K,1,S);
        p->t1p(dr+k0,di+k0,dr+k0,di+k0,p->Qr,p->Qi,(size_t)R2*K,1,(size_t)R2*K,1,S);}
    else vfft_proto_execute_fwd_oop(p->mb,sr+k0,si+k0,dr+k0,di+k0,S);
}
typedef struct { const vfft_oop_plan_t*p; const double*sr,*si; double*dr,*di; size_t k0,S; } oop_arg_t;
static void oop_tramp(void*a){oop_arg_t*x=(oop_arg_t*)a; oop_slice(x->p,x->sr,x->si,x->dr,x->di,x->k0,x->S);}
static void oop_mt(const vfft_oop_plan_t*p,const double*sr,const double*si,double*dr,double*di){
    size_t K=p->K; int T=stride_get_num_threads();
    if(T>_stride_pool_size+1)T=_stride_pool_size+1;
    if(T<=1||K<8){oop_slice(p,sr,si,dr,di,0,K);return;}
    size_t S=((K/(size_t)T)+7)&~(size_t)7; oop_arg_t a[64]; int nd=0;
    for(int t=1;t<T&&t<=_stride_pool_size;t++){size_t k0=(size_t)t*S;if(k0>=K)break;size_t ke=k0+S;if(ke>K)ke=K;
        a[nd]=(oop_arg_t){p,sr,si,dr,di,k0,ke-k0};_stride_pool_dispatch(&_stride_workers[nd],oop_tramp,&a[nd]);nd++;}
    size_t s0=S<K?S:K; oop_slice(p,sr,si,dr,di,0,s0); if(nd)_stride_pool_wait_all();
}

static void run(int N,size_t K,vfft_proto_registry_t*reg){
    size_t NK=(size_t)N*K;
    stride_plan_t*ip=vfft_proto_auto_plan(N,K,reg,NULL);
    int factors[16],nf=0,n=N; while(n%4==0){factors[nf++]=4;n/=4;} if(n>1)factors[nf++]=n;
    vfft_oop_plan_t*op=vfft_oop_plan_create(N,K,factors,nf,reg);
    if(!ip||!op){printf("N=%-5d K=%-5zu plan NULL\n",N,K);return;}
    double *re=AALLOC(NK*8),*im=AALLOC(NK*8),*x0=AALLOC(NK*8),*xi0=AALLOC(NK*8);
    double *sr=AALLOC(NK*8),*si=AALLOC(NK*8),*dr=AALLOC(NK*8),*di=AALLOC(NK*8),*r1=AALLOC(NK*8),*r2=AALLOC(NK*8);
    srand(7+N);for(size_t i=0;i<NK;i++){x0[i]=(double)rand()/RAND_MAX-0.5;xi0[i]=(double)rand()/RAND_MAX-0.5;sr[i]=x0[i];si[i]=xi0[i];}
    int Ts[]={1,2,4,8}; double ipt[4],opt[4],iperr=0,operr=0;
    for(int ti=0;ti<4;ti++){
        stride_set_num_threads(Ts[ti]); int reps=reps_for(NK);
        double bi=1e18; for(int w=0;w<2;w++){memcpy(re,x0,NK*8);memcpy(im,xi0,NK*8);inplace_mt(ip,re,im);}
        for(int r=0;r<BEST_OF;r++){memcpy(re,x0,NK*8);memcpy(im,xi0,NK*8);double t0=now_c();inplace_mt(ip,re,im);double v=now_c()-t0;if(v<bi)bi=v;}
        ipt[ti]=bi; if(ti==0)memcpy(r1,re,NK*8);else{double e=0;for(size_t i=0;i<NK;i++){double a=re[i]-r1[i];if(a<0)a=-a;if(a>e)e=a;}if(e>iperr)iperr=e;}
        double bo=1e18; for(int w=0;w<2;w++)oop_mt(op,sr,si,dr,di);
        for(int r=0;r<BEST_OF;r++){double t0=now_c();oop_mt(op,sr,si,dr,di);double v=now_c()-t0;if(v<bo)bo=v;}
        opt[ti]=bo; if(ti==0)memcpy(r2,dr,NK*8);else{double e=0;for(size_t i=0;i<NK;i++){double a=dr[i]-r2[i];if(a<0)a=-a;if(a>e)e=a;}if(e>operr)operr=e;}
    }
    stride_set_num_threads(1);
    printf("N=%-5d K=%-5zu%s\n",N,K,(K&511)?"  (non-aliasing)":"  (32KB-alias)");
    printf("  in-place:");for(int ti=0;ti<4;ti++)printf(" T%d=%.2fx",Ts[ti],ipt[0]/ipt[ti]); printf("  (err %.0e)\n",iperr);
    printf("  OOP     :");for(int ti=0;ti<4;ti++)printf(" T%d=%.2fx",Ts[ti],opt[0]/opt[ti]); printf("  (err %.0e)\n",operr);
    fflush(stdout);
    AFREE(re);AFREE(im);AFREE(x0);AFREE(xi0);AFREE(sr);AFREE(si);AFREE(dr);AFREE(di);AFREE(r1);AFREE(r2);
    vfft_proto_plan_destroy(ip);vfft_oop_plan_destroy(op);
}

int main(void){
    stride_env_init();
    if(stride_pin_thread(PIN_CORE)!=0) fprintf(stderr,"warn pin\n");
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    printf("== c2c MT (pool K-split), in-place + OOP, scaling vs T=1 ==\n");
    printf("# working set = N*K*16 bytes. <2MB = L2-resident (compute-bound); >>2MB = DRAM-bound.\n");
    run(128,512,&reg);   /* 0.5MB */
    run(256,512,&reg);   /* 2MB   */
    run(256,1024,&reg);  /* 4MB   */
    run(512,1024,&reg);  /* 8MB   */
    run(1024,4096,&reg); /* 64MB DRAM-bound */
    return 0;
}
