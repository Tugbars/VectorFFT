/* bench_trig_dct1.c — validate DCT-I (REDFT00) + DST-I (RODFT00) on dag
 * (core/dct1.h). Real->real, self-inverse. DCT-I uses an r2c of M=2(N-1);
 * DST-I uses M=2(N+1). Validate at N whose N-1 / N+1 factor cleanly.
 *
 * Correctness: direct reference formula (definitive, lane 0) + self-inverse
 * roundtrip fwd(fwd(x)) == scale*x  (DCT-I: 2(N-1); DST-I: 2(N+1)).
 *
 * Build: cd build_tuned && python build.py --src benches/bench_trig_dct1.c --compile
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "executor.h"
#include "planner.h"
#include "threads.h"
#include "proto_stride_compat.h"
#include "r2c.h"
#include "dct1.h"
#include "env.h"
#include "generator/generated/registry.h"

#define PIN_CORE 2
#if defined(_WIN32)
#include <malloc.h>
#define AALLOC(n) _aligned_malloc((n),64)
#define AFREE(p)  _aligned_free(p)
#else
#define AALLOC(n) aligned_alloc(64,(n))
#define AFREE(p)  free(p)
#endif

/* REDFT00 (DCT-I): Y[k] = x0 + (-1)^k x[N-1] + 2 sum_{n=1..N-2} x[n] cos(pi n k/(N-1)) */
static double ref_dct1(const double*x,int N,size_t K,int k){
    double s=x[0]+((k&1)?-1.0:1.0)*x[(size_t)(N-1)*K];
    for(int n=1;n<=N-2;n++) s+=2.0*x[(size_t)n*K]*cos(M_PI*(double)n*k/(double)(N-1));
    return s;
}
/* RODFT00 (DST-I): Y[k] = 2 sum_{n=0..N-1} x[n] sin(pi (n+1)(k+1)/(N+1)) */
static double ref_dst1(const double*x,int N,size_t K,int k){
    double s=0; for(int n=0;n<N;n++) s+=x[(size_t)n*K]*sin(M_PI*(double)(n+1)*(k+1)/(double)(N+1));
    return 2.0*s;
}

static void run(int is_dst,int N,size_t K,vfft_proto_registry_t*reg){
    const char*nm=is_dst?"DST-I":"DCT-I";
    int M = is_dst ? 2*(N+1) : 2*(N-1);
    int innerN = M/2;                       /* N+1 (DST-I) or N-1 (DCT-I) */
    stride_plan_t* inner=vfft_proto_auto_plan(innerN,K,reg,NULL);
    stride_plan_t* r2c = inner?stride_r2c_plan(M,K,K,inner):NULL;
    if(!r2c){ printf("  %-6s N=%-4d  r2c(M=%d,inner=%d) NULL\n",nm,N,M,innerN); return; }
    stride_plan_t* p = is_dst?stride_dst1_plan(N,K,r2c):stride_dct1_plan(N,K,r2c);
    if(!p){ printf("  %-6s N=%-4d  plan NULL\n",nm,N); return; }

    size_t NK=(size_t)N*K;
    double*x=AALLOC(NK*8),*y=AALLOC(NK*8),*z=AALLOC(NK*8);
    srand(13+N+is_dst); for(size_t i=0;i<NK;i++)x[i]=(double)rand()/RAND_MAX-0.5;
    if(is_dst) stride_execute_dst1(p,x,y); else stride_execute_dct1(p,x,y);
    double re=0,nrm=0;
    for(int k=0;k<N;k++){double r=is_dst?ref_dst1(x,N,K,k):ref_dct1(x,N,K,k),g=y[(size_t)k*K];
        double e=fabs(g-r); if(e>re)re=e; if(fabs(r)>nrm)nrm=fabs(r);}
    double ref_err=nrm>0?re/nrm:re;
    /* self-inverse: apply forward twice -> scale*x */
    if(is_dst) stride_execute_dst1(p,y,z); else stride_execute_dct1(p,y,z);
    double scale = is_dst ? 2.0*(N+1) : 2.0*(N-1);
    double rt=0; for(size_t i=0;i<NK;i++){double a=fabs(z[i]/scale-x[i]); if(a>rt)rt=a;}
    printf("  %-6s N=%-4d (M=%-4d inner=%-3d)  ref_err=%-10.2e rt_err=%-10.2e %s\n",
           nm,N,M,innerN,ref_err,rt,(ref_err<1e-11&&rt<1e-11)?"OK":"*** FAIL ***");
    AFREE(x);AFREE(y);AFREE(z); stride_plan_destroy(p);
}

int main(void){
    stride_env_init();
    if(stride_pin_thread(PIN_CORE)!=0) fprintf(stderr,"warn pin\n");
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    printf("=== DCT-I / DST-I on dag (ref + self-inverse roundtrip, K=8) ===\n");
    int Ns[]={8,9,16,17,33,65};   /* N-1 / N+1 factor over available radixes */
    for(int i=0;i<(int)(sizeof Ns/sizeof Ns[0]);i++){ run(0,Ns[i],8,&reg); run(1,Ns[i],8,&reg); }
    return 0;
}
