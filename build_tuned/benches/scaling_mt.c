/* scaling_mt.c — MT scaling of in-place c2c through the PUBLIC vfft API (FFT + reorder),
 * separating SCRAMBLED (order=DEFAULT, FFT only) from NATURAL (FFT + reorder) to see whether
 * the reorder pass is the scaling bottleneck. Trusted methodology (bench_1d_vs_mkl): caller
 * pinned to P-core 0, HIGH_PRIORITY, 10 warmup, best-of-5 with 32MB cachebust between trials.
 * Creates a FRESH plan per thread-count (create cost not timed). Sweeps T=1,2,4,8.
 * Usage: scaling_mt <wisdom_dir> <N> <K> [order:def|nat]   (default: both)
 * Build: python build.py --src benches/scaling_mt.c --vfft --jit
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "vfft.h"

static double now_ms(void){ LARGE_INTEGER f,c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return 1000.0*c.QuadPart/f.QuadPart; }
static void cachebust(void){ size_t s=32*1024*1024/8; double*j=malloc(s*8); volatile double a=0;
    for(size_t i=0;i<s;i++) j[i]=(double)i*0.5; for(size_t i=0;i<s;i++) a+=j[i]; (void)a; free(j); }

/* best-of-5 ns/exec for a fresh plan at nthreads=T. Returns -1 on create failure. */
static double measure(int N, size_t K, vfft_order_t order, int T) {
    vfft_config_t c; memset(&c,0,sizeof c);
    c.transform=VFFT_C2C; c.placement=VFFT_INPLACE; c.rigor=VFFT_MEASURE;
    c.dims=1; c.n[0]=N; c.howmany=K; c.nthreads=T; c.order=order;
    vfft_plan h=vfft_create(&c); if(!h) return -1;
    size_t tot=(size_t)N*K; double *re=malloc(tot*8),*im=malloc(tot*8);
    srand(3); for(size_t i=0;i<tot;i++){ re[i]=(double)rand()/RAND_MAX-0.5; im[i]=(double)rand()/RAND_MAX-0.5; }
    for(int w=0;w<10;w++) vfft_execute(h,VFFT_FORWARD,re,im,re,im);
    int reps=(int)(4e6/(tot+1)); if(reps<20) reps=20; if(reps>200000) reps=200000;
    double best=1e18;
    for(int t=0;t<5;t++){ if(t){ cachebust(); }
        double t0=now_ms();
        for(int i=0;i<reps;i++) vfft_execute(h,VFFT_FORWARD,re,im,re,im);
        double ns=(now_ms()-t0)*1e6/reps; if(ns<best) best=ns; }
    free(re);free(im); vfft_destroy(h);
    return best;
}

static void sweep(int N, size_t K, vfft_order_t order, const char *tag) {
    printf("  %-4s N=%-5d K=%-4zu : ", tag, N, K);
    double t1=-1; int Ts[4]={1,2,4,8};
    for(int i=0;i<4;i++){ double ns=measure(N,K,order,Ts[i]); if(i==0)t1=ns;
        if(ns<0){ printf(" T%d=NULL", Ts[i]); continue; }
        printf(" T%d=%.3gus(%.2fx)", Ts[i], ns/1000.0, (t1>0&&ns>0)?t1/ns:0.0); }
    printf("\n");
}

int main(int argc, char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    SetThreadAffinityMask(GetCurrentThread(),(DWORD_PTR)1);   /* caller on P-core 0 */
    SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
    const char *wis = argc>1?argv[1]:"natmt_wis";
    char env[512]; snprintf(env,sizeof env,"VFFT_WISDOM_DIR=%s",wis); putenv(env);
    int N = argc>2?atoi(argv[2]):1024;
    size_t K = argc>3?(size_t)atoi(argv[3]):256;
    const char *ord = argc>4?argv[4]:"both";
    printf("# MT scaling (public API, best-of-5, cachebust). PIN_STRIDE=%s\n", getenv("VFFT_PIN_STRIDE")?getenv("VFFT_PIN_STRIDE"):"2(default)");
    if(strcmp(ord,"nat")!=0) sweep(N,K,VFFT_ORDER_DEFAULT,"scr");
    if(strcmp(ord,"def")!=0) sweep(N,K,VFFT_ORDER_NATURAL,"nat");
    return 0;
}
