/* bench_trig_dag.c — validate the ported trig transforms on the dag tree:
 * DCT-II, DCT-IV, DST-II, DHT (core/{dct,dct4,dst,dht}.h).
 *
 * Correctness: (1) direct reference formula at small N (definitive, lane 0),
 * (2) roundtrip fwd+inv == scale*x at the doc cells. Plus single-thread timing.
 *   DCT-II : inv=dct3, scale=2N.   DST-II: inv=dst3, scale=2N.
 *   DCT-IV : self-inverse, 2N.     DHT  : self-inverse, N.
 *
 * Inner-plan chain: DCT-II/DHT need an r2c plan of N (stride_r2c_plan over an
 * auto_plan(N/2) inner); DCT-IV needs a c2c plan of N/2; DST-II wraps a DCT-II.
 *
 * Build: cd build_tuned && python build.py --src benches/bench_trig_dag.c --compile
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "core/executor.h"
#include "core/planner.h"
#include "core/threads.h"
#include "core/proto_stride_compat.h"
#include "core/r2c.h"
#include "core/dct.h"
#include "core/dct4.h"
#include "core/dst.h"
#include "core/dht.h"
#include "core/env.h"
#include "generator/generated/registry.h"

#define PIN_CORE 2
#define BEST_OF  9
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
static int reps_for(size_t t){int r=(int)(2e7/(t+1)); if(r<5)r=5; if(r>5000)r=5000; return r;}

/* ---- direct references (lane 0), FFTW unnormalized conventions ---- */
static double ref_dct2(const double*x,int N,size_t K,int k){double s=0;for(int n=0;n<N;n++)s+=x[(size_t)n*K]*cos(M_PI*(2*n+1)*k/(2.0*N));return 2*s;}
static double ref_dct4(const double*x,int N,size_t K,int k){double s=0;for(int n=0;n<N;n++)s+=x[(size_t)n*K]*cos(M_PI*(2*k+1)*(2*n+1)/(4.0*N));return 2*s;}
static double ref_dst2(const double*x,int N,size_t K,int k){double s=0;for(int n=0;n<N;n++)s+=x[(size_t)n*K]*sin(M_PI*(2*n+1)*(k+1)/(2.0*N));return 2*s;}
static double ref_dht (const double*x,int N,size_t K,int k){double s=0;for(int n=0;n<N;n++){double a=2.0*M_PI*k*n/N; s+=x[(size_t)n*K]*(cos(a)+sin(a));}return s;}

typedef enum { T_DCT2, T_DCT4, T_DST2, T_DHT } trig_t;
static const char* tname(trig_t t){return t==T_DCT2?"DCT-II":t==T_DCT4?"DCT-IV":t==T_DST2?"DST-II":"DHT";}

/* build the transform plan (owns its inner plans via override_destroy). */
static stride_plan_t* build(trig_t t,int N,size_t K,vfft_proto_registry_t*reg){
    if(t==T_DCT4){
        stride_plan_t* c2c=vfft_proto_auto_plan(N/2,K,reg,NULL);
        return c2c?stride_dct4_plan(N,K,c2c):NULL;
    }
    /* DCT-II / DST-II / DHT all start from an r2c plan of N */
    stride_plan_t* inner=vfft_proto_auto_plan(N/2,K,reg,NULL);
    stride_plan_t* r2c=inner?stride_r2c_plan(N,K,K,inner):NULL;
    if(!r2c) return NULL;
    if(t==T_DHT)  return stride_dht_plan(N,K,r2c);
    stride_plan_t* dct2=stride_dct2_plan(N,K,r2c);
    if(t==T_DCT2) return dct2;
    return dct2?stride_dst2_plan(N,K,dct2):NULL;   /* DST-II wraps DCT-II */
}
static void fwd(trig_t t,const stride_plan_t*p,const double*in,double*out){
    if(t==T_DCT2)stride_execute_dct2(p,in,out); else if(t==T_DCT4)stride_execute_dct4(p,in,out);
    else if(t==T_DST2)stride_execute_dst2(p,in,out); else stride_execute_dht(p,in,out);
}
static void inv(trig_t t,const stride_plan_t*p,const double*in,double*out){
    if(t==T_DCT2)stride_execute_dct3(p,in,out); else if(t==T_DCT4)stride_execute_dct4(p,in,out);
    else if(t==T_DST2)stride_execute_dst3(p,in,out); else stride_execute_dht(p,in,out);
}
static double refval(trig_t t,const double*x,int N,size_t K,int k){
    return t==T_DCT2?ref_dct2(x,N,K,k):t==T_DCT4?ref_dct4(x,N,K,k):t==T_DST2?ref_dst2(x,N,K,k):ref_dht(x,N,K,k);
}

int main(void){
    stride_env_init();
    if(stride_pin_thread(PIN_CORE)!=0) fprintf(stderr,"warn pin\n");
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    trig_t all[]={T_DCT2,T_DCT4,T_DST2,T_DHT};

    printf("=== trig transforms on dag (correctness + ST timing, cpu%d) ===\n",PIN_CORE);
    printf("# ref = max rel err vs direct formula @N=16 K=8. rt = roundtrip fwd+inv==scale*x.\n\n");

    /* (1) reference correctness at small N */
    printf("%-8s %-12s %-12s\n","trig","ref_err@16","rt_err@16");
    for(int ti=0;ti<4;ti++){
        trig_t t=all[ti]; int N=16; size_t K=8, NK=(size_t)N*K;
        stride_plan_t*p=build(t,N,K,&reg);
        if(!p){printf("%-8s build NULL\n",tname(t));continue;}
        double*x=AALLOC(NK*8),*y=AALLOC(NK*8),*z=AALLOC(NK*8);
        srand(7+ti); for(size_t i=0;i<NK;i++)x[i]=(double)rand()/RAND_MAX-0.5;
        fwd(t,p,x,y);
        double re=0,nrm=0; for(int k=0;k<N;k++){double r=refval(t,x,N,K,k),g=y[(size_t)k*K]; double e=fabs(g-r); if(e>re)re=e; if(fabs(r)>nrm)nrm=fabs(r);}
        double ref_err = nrm>0?re/nrm:re;
        inv(t,p,y,z);
        double scale = (t==T_DHT)?(double)N:2.0*N;
        double rt=0; for(size_t i=0;i<NK;i++){double a=fabs(z[i]/scale-x[i]); if(a>rt)rt=a;}
        printf("%-8s %-12.2e %-12.2e %s\n",tname(t),ref_err,rt,(ref_err<1e-12&&rt<1e-12)?"OK":"*** FAIL ***");
        AFREE(x);AFREE(y);AFREE(z); stride_plan_destroy(p);
    }

    /* (2) roundtrip + timing at the doc cells */
    struct { int N; size_t K; } cells[]={{256,1024},{1024,1024},{4096,1024}};
    printf("\n%-8s %-12s %-10s %-12s\n","trig","cell","rt_err","fwd_ns");
    for(int ci=0;ci<3;ci++){
        int N=cells[ci].N; size_t K=cells[ci].K, NK=(size_t)N*K;
        for(int ti=0;ti<4;ti++){
            trig_t t=all[ti];
            stride_plan_t*p=build(t,N,K,&reg);
            if(!p){printf("%-8s N=%-4d K=%-4zu build NULL\n",tname(t),N,K);continue;}
            double*x=AALLOC(NK*8),*y=AALLOC(NK*8),*z=AALLOC(NK*8);
            srand(7+ti+N); for(size_t i=0;i<NK;i++)x[i]=(double)rand()/RAND_MAX-0.5;
            fwd(t,p,x,y); inv(t,p,y,z);
            double scale=(t==T_DHT)?(double)N:2.0*N, rt=0;
            for(size_t i=0;i<NK;i++){double a=fabs(z[i]/scale-x[i]); if(a>rt)rt=a;}
            int reps=reps_for(NK); double bv=1e18;
            for(int w=0;w<2;w++) fwd(t,p,x,y);
            for(int r=0;r<BEST_OF;r++){double t0=now_c(); for(int i=0;i<reps;i++) fwd(t,p,x,y); double v=(now_c()-t0)/reps; if(v<bv)bv=v;}
            printf("%-8s N=%-4d K=%-4zu  %-10.1e %.0f %s\n",tname(t),N,K,rt,bv,rt<1e-9?"":" *** RT FAIL ***");
            fflush(stdout);
            AFREE(x);AFREE(y);AFREE(z); stride_plan_destroy(p);
        }
    }
    return 0;
}
