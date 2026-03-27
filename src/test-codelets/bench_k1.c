/**
 * bench_k1.c -- K=1 N1 codelet benchmark: scalar vs AVX2 vs FFTW
 *
 * K=1 codelets are sub-nanosecond, so we batch a large inner loop
 * and report the amortized per-call time.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>
#include "bench_compat.h"
#include "fft_n1_k1.h"
#include "fft_n1_k1_simd.h"

typedef void (*k1_fn)(const double*,const double*,double*,double*);

static double bench_k1(k1_fn fn, const double *ir, const double *ii,
    double *or_, double *oi, int reps) {
    /* warm up */
    for(int i=0;i<500;i++) fn(ir,ii,or_,oi);
    double best=1e18;
    for(int t=0;t<15;t++){
        double t0=now_ns();
        for(int i=0;i<reps;i++) fn(ir,ii,or_,oi);
        double ns=(now_ns()-t0)/reps;
        if(ns<best) best=ns;
    }
    return best;
}

static double bench_fftw(int R, double *ri, double *ii, double *ro, double *io, int reps) {
    fftw_iodim dim={.n=R,.is=1,.os=1};
    fftw_plan p=fftw_plan_guru_split_dft(1,&dim,0,NULL,ri,ii,ro,io,FFTW_MEASURE);
    if(!p) return -1;
    for(int i=0;i<500;i++) fftw_execute(p);
    double best=1e18;
    for(int t=0;t<15;t++){
        double t0=now_ns();
        for(int i=0;i<reps;i++) fftw_execute_split_dft(p,ri,ii,ro,io);
        double ns=(now_ns()-t0)/reps;
        if(ns<best) best=ns;
    }
    fftw_destroy_plan(p);
    return best;
}

int main(void) {
    srand(42);
    /* 20M reps to get above timer noise for sub-ns calls */
    int reps = 20000000;
    double __attribute__((aligned(64))) ir[32],ii[32],or_[32],oi[32];
    for(int i=0;i<32;i++){ir[i]=(double)rand()/RAND_MAX-0.5;ii[i]=(double)rand()/RAND_MAX-0.5;}

    /* FFTW needs its own buffers */
    double *fri=fftw_malloc(32*8),*fii=fftw_malloc(32*8),*fro=fftw_malloc(32*8),*fio=fftw_malloc(32*8);
    for(int i=0;i<32;i++){fri[i]=ir[i];fii[i]=ii[i];}

    printf("==================================================================\n");
    printf("  K=1 N1 codelet benchmark: scalar vs AVX2 vs FFTW_MEASURE\n");
    printf("  %d reps per trial, 15 trials, best-of\n", reps);
    printf("==================================================================\n\n");

    printf("%-8s %10s %10s %10s %10s %10s\n",
           "Radix", "scalar", "SIMD", "FFTW", "vs_scalar", "vs_FFTW");

    /* R=4 (scalar only -- no SIMD K=1 codelet for R=4) */
    {
        double ns_s = bench_k1((k1_fn)dft4_k1_fwd, ir,ii,or_,oi, reps);
        double ns_f = bench_fftw(4, fri,fii,fro,fio, reps);
        printf("R=4      %7.2f ns %10s %7.2f ns %10s %8.2fx\n",
               ns_s, "--", ns_f, "--", ns_f/ns_s);
    }

    /* R=8 (scalar only -- no SIMD K=1 codelet for R=8) */
    {
        double ns_s = bench_k1((k1_fn)dft8_k1_fwd, ir,ii,or_,oi, reps);
        double ns_f = bench_fftw(8, fri,fii,fro,fio, reps);
        printf("R=8      %7.2f ns %10s %7.2f ns %10s %8.2fx\n",
               ns_s, "--", ns_f, "--", ns_f/ns_s);
    }

    /* R=16 (scalar + AVX2) */
    {
        double ns_s = bench_k1((k1_fn)dft16_k1_fwd, ir,ii,or_,oi, reps);
        double ns_v = bench_k1((k1_fn)dft16_k1_fwd_avx2, ir,ii,or_,oi, reps);
        double ns_f = bench_fftw(16, fri,fii,fro,fio, reps);
        printf("R=16     %7.2f ns %7.2f ns %7.2f ns %8.2fx %8.2fx\n",
               ns_s, ns_v, ns_f, ns_s/ns_v, ns_f/ns_v);
    }

    /* R=32 (scalar only -- AVX2 K=1 codelet not available, AVX-512 only) */
    {
        double ns_s = bench_k1((k1_fn)dft32_k1_fwd, ir,ii,or_,oi, reps);
        double ns_f = bench_fftw(32, fri,fii,fro,fio, reps);
        printf("R=32     %7.2f ns %10s %7.2f ns %10s %8.2fx\n",
               ns_s, "--", ns_f, "--", ns_f/ns_s);
    }

    printf("\nvs_scalar = scalar/SIMD speedup\n");
    printf("vs_FFTW = FFTW/our_best\n");

    fftw_free(fri);fftw_free(fii);fftw_free(fro);fftw_free(fio);
    return 0;
}
