/* bench_fft2d_mt_vs_mkl.c — MULTITHREADED 2D c2c: dag (8 P-cores, caller-pinned
 * core 0) vs MKL DFTI 2D (mkl_set_num_threads). Same N1xN2 complex plane, split
 * layout, out-of-place MKL. Mirrors bench_r2c_mt_vs_mkl.c's fair pattern: build
 * the dag plan at T=8 (scratch sized for 8 tile-workers), then time each side at
 * its own thread count. Pool workers park when idle, so they don't fight MKL.
 *
 * dag 2D MT = tile-parallel row pass + K-split column pass (both barrier-free).
 *
 * Build: cd build_tuned && python build.py --src benches/bench_fft2d_mt_vs_mkl.c --mkl --compile
 * Run with MKL bin + C:\mingw152\mingw64\bin on PATH.
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mkl_dfti.h>
#include <mkl_service.h>
#include "threads.h"       /* pool: set/get num threads, pin (before fft2d.h) */
#include "fft2d.h"
#include "env.h"
#include "generator/generated/registry.h"

#define BEST_OF 15
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
static int reps_for(size_t t){int r=(int)(4e7/(t+1)); if(r<10)r=10; if(r>20000)r=20000; return r;}

/* time dag 2D fwd at thread count T (in-place; repeated fwd = same compute cost). */
static double bench_dag(stride_plan_t*p,double*re,double*im,size_t total,int T){
    stride_set_num_threads(T);
    for(int w=0;w<3;w++) stride_execute_fwd(p,re,im);
    int reps=reps_for(total); double best=1e18;
    for(int b=0;b<BEST_OF;b++){double t0=now_c();
        for(int i=0;i<reps;i++) stride_execute_fwd(p,re,im);
        double v=(now_c()-t0)/reps; if(v<best)best=v;}
    return best;
}
static double bench_mkl(DFTI_DESCRIPTOR_HANDLE h,const double*xr,const double*xi,
                        double*mr,double*mi,size_t total){
    for(int w=0;w<3;w++) DftiComputeForward(h,(void*)xr,(void*)xi,mr,mi);
    int reps=reps_for(total); double best=1e18;
    for(int b=0;b<BEST_OF;b++){double t0=now_c();
        for(int i=0;i<reps;i++) DftiComputeForward(h,(void*)xr,(void*)xi,mr,mi);
        double v=(now_c()-t0)/reps; if(v<best)best=v;}
    return best;
}
/* commit a 2D complex split, NOT_INPLACE DFTI descriptor at current mkl threads. */
static DFTI_DESCRIPTOR_HANDLE mkl_make2d(int N1,int N2){
    DFTI_DESCRIPTOR_HANDLE h=0; MKL_LONG dims[2]={N1,N2};
    if(DftiCreateDescriptor(&h,DFTI_DOUBLE,DFTI_COMPLEX,2,dims)!=DFTI_NO_ERROR) return 0;
    DftiSetValue(h,DFTI_COMPLEX_STORAGE,DFTI_REAL_REAL);
    DftiSetValue(h,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
    if(DftiCommitDescriptor(h)!=DFTI_NO_ERROR){DftiFreeDescriptor(&h);return 0;}
    return h;
}

int main(void){
    stride_env_init();
    if(stride_pin_thread(0)!=0) fprintf(stderr,"warn pin\n");   /* caller = core 0 */
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);

    const int Ns[]={64,128,256,512,1024}; const int nN=5;

    printf("=== 2D c2c MT: dag (8 P-cores, core0-pinned) vs MKL (8 threads), SPLIT, NOT_INPLACE ===\n");
    printf("# dag MT = tile-parallel rows + K-split cols (barrier-free). plan built at T=8.\n");
    printf("%-9s %9s %10s %10s   %10s %10s   %8s %8s   %9s %9s\n",
           "size","WS(MB)","dagT1_ns","dagT8_ns","mklT1_ns","mklT8_ns",
           "dagScal","mklScal","T1 mkl/dag","T8 mkl/dag");
    printf("----------+---------+----------+----------+----------+----------+--------+--------+---------+---------\n");

    for(int ni=0;ni<nN;ni++){
        int N=Ns[ni]; size_t total=(size_t)N*N; double ws=(double)total*16.0/(1024*1024);

        stride_set_num_threads(8);                       /* scratch sized for 8 tile-workers */
        stride_plan_t*p=stride_plan_2d(N,N,&reg);
        if(!p){printf("%dx%-4d  dag plan NULL\n",N,N);continue;}

        double*re=AALLOC(total*8),*im=AALLOC(total*8),*xr=AALLOC(total*8),*xi=AALLOC(total*8);
        double*mr=AALLOC(total*8),*mi=AALLOC(total*8);
        srand(11+N);
        for(size_t i=0;i<total;i++){xr[i]=(double)rand()/RAND_MAX-0.5; xi[i]=(double)rand()/RAND_MAX-0.5;}

        /* correctness: roundtrip fwd+bwd == N*N*x at T8 */
        memcpy(re,xr,total*8); memcpy(im,xi,total*8);
        stride_set_num_threads(8);
        stride_execute_fwd(p,re,im); stride_execute_bwd(p,re,im);
        double rt=0,sc=(double)N*N;
        for(size_t i=0;i<total;i++){double a=fabs(re[i]/sc-xr[i]),b=fabs(im[i]/sc-xi[i]); if(a>rt)rt=a; if(b>rt)rt=b;}
        if(rt>1e-9) printf("%dx%-4d  ROUNDTRIP FAIL %.2e\n",N,N,rt);

        /* dag timings (in-place; reset data first so values stay bounded) */
        memcpy(re,xr,total*8); memcpy(im,xi,total*8);
        double d1=bench_dag(p,re,im,total,1);
        double d8=bench_dag(p,re,im,total,8);

        /* MKL timings (re-commit per thread count) */
        mkl_set_num_threads(1); DFTI_DESCRIPTOR_HANDLE h1=mkl_make2d(N,N);
        double m1=h1?bench_mkl(h1,xr,xi,mr,mi,total):0; if(h1)DftiFreeDescriptor(&h1);
        mkl_set_num_threads(8); DFTI_DESCRIPTOR_HANDLE h8=mkl_make2d(N,N);
        double m8=h8?bench_mkl(h8,xr,xi,mr,mi,total):0; if(h8)DftiFreeDescriptor(&h8);

        char sz[16]; snprintf(sz,sizeof sz,"%dx%d",N,N);
        printf("%-9s %9.1f %10.0f %10.0f   %10.0f %10.0f   %7.2fx %7.2fx   %8.3fx %8.3fx\n",
               sz, ws, d1, d8, m1, m8,
               (d8>0)?d1/d8:0, (m8>0)?m1/m8:0, (d1>0)?m1/d1:0, (d8>0&&m8>0)?m8/d8:0);
        fflush(stdout);

        AFREE(re);AFREE(im);AFREE(xr);AFREE(xi);AFREE(mr);AFREE(mi);
        stride_plan_destroy(p);
    }
    stride_set_num_threads(1);
    printf("\n# dagScal/mklScal = T1/T8 self-speedup. T8 mkl/dag = MKL8_ns/dag8_ns (>1 = dag faster at T8).\n");
    return 0;
}
