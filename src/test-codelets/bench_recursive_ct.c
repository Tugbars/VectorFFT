/**
 * bench_recursive_ct.c -- Permutation-free CT executor
 *
 * Uses n1 (separate is/os) + t1 (in-place twiddle) codelets.
 * No permutation, no transpose, no gather/scatter.
 *
 * 1-level CT for N = R*M:
 *   n1(in, out, is=R, os=M, vl=M)  -- reads in[n*R+k], writes out[n*M+k]
 *   t1(out, W, ios=M, me=M)        -- in-place twiddle + butterfly
 *
 * 2-level CT for N = R1*(R0*M0):
 *   outer n1 decimates input into R1 contiguous blocks of M1=R0*M0
 *   inner 1-level CT on each block
 *   outer t1 combines
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>
#include "bench_compat.h"

#include "fft_radix4_avx2.h"
#include "fft_radix8_avx2.h"
#include "fft_radix16_avx2_notw.h"
#include "fft_radix16_avx2_ct_n1.h"
#include "fft_radix16_avx2_ct_t1_dit.h"

typedef void (*n1_fn)(const double*, const double*, double*, double*,
                      size_t, size_t, size_t);
typedef void (*t1_fn)(double*, double*, const double*, const double*,
                      size_t, size_t);
typedef void (*exec_fn)(const double*, const double*, double*, double*);

static void init_t1_tw(double *W_re, double *W_im, size_t R, size_t me) {
    size_t N = R * me;
    for (size_t n = 1; n < R; n++)
        for (size_t m = 0; m < me; m++) {
            double a = -2.0 * M_PI * (double)(n * m) / (double)N;
            W_re[(n-1)*me + m] = cos(a);
            W_im[(n-1)*me + m] = sin(a);
        }
}

/* ================================================================
 * 1-level CT: n1 + t1
 * ================================================================ */

static void ct_1level(
    n1_fn n1, t1_fn t1,
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *W_re, const double *W_im,
    size_t R, size_t M)
{
    n1(in_re, in_im, out_re, out_im, R, M, M);
    t1(out_re, out_im, W_re, W_im, M, M);
}

/* ================================================================
 * 2-level CT: outer n1 + R1 x inner 1-level + outer t1
 * ================================================================ */

static void ct_2level(
    n1_fn n1_1, t1_fn t1_1, size_t R1,
    n1_fn n1_0, t1_fn t1_0, size_t R0, size_t M0,
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    double *tmp_re, double *tmp_im,
    const double *W0_re, const double *W0_im,
    const double *W1_re, const double *W1_im)
{
    size_t M1 = R0 * M0;
    n1_1(in_re, in_im, tmp_re, tmp_im, R1, M1, M1);
    for (size_t r = 0; r < R1; r++)
        ct_1level(n1_0, t1_0,
                  tmp_re + r*M1, tmp_im + r*M1,
                  out_re + r*M1, out_im + r*M1,
                  W0_re, W0_im, R0, M0);
    t1_1(out_re, out_im, W1_re, W1_im, M1, M1);
}

/* ================================================================
 * FFTW reference
 * ================================================================ */

static double bench_fftw(size_t N, int reps) {
    double *ri=fftw_malloc(N*8),*ii=fftw_malloc(N*8);
    double *ro=fftw_malloc(N*8),*io=fftw_malloc(N*8);
    for(size_t i=0;i<N;i++){ri[i]=(double)rand()/RAND_MAX;ii[i]=(double)rand()/RAND_MAX;}
    fftw_iodim d={.n=(int)N,.is=1,.os=1};
    fftw_iodim h={.n=1,.is=(int)N,.os=(int)N};
    fftw_plan p=fftw_plan_guru_split_dft(1,&d,1,&h,ri,ii,ro,io,FFTW_MEASURE);
    if(!p){fftw_free(ri);fftw_free(ii);fftw_free(ro);fftw_free(io);return -1;}
    for(int i=0;i<20;i++)fftw_execute(p);
    double best=1e18;
    for(int t=0;t<7;t++){double t0=now_ns();
        for(int i=0;i<reps;i++)fftw_execute_split_dft(p,ri,ii,ro,io);
        double ns=(now_ns()-t0)/reps;if(ns<best)best=ns;}
    fftw_destroy_plan(p);fftw_free(ri);fftw_free(ii);fftw_free(ro);fftw_free(io);
    return best;
}

/* ================================================================
 * Test harness
 * ================================================================ */

static void test_exec(const char *label, size_t N, exec_fn fn, int reps) {
    double *in_re=(double*)aligned_alloc(32,N*8);
    double *in_im=(double*)aligned_alloc(32,N*8);
    double *out_re=(double*)aligned_alloc(32,N*8);
    double *out_im=(double*)aligned_alloc(32,N*8);
    srand(42);
    for(size_t i=0;i<N;i++){in_re[i]=(double)rand()/RAND_MAX-.5;in_im[i]=(double)rand()/RAND_MAX-.5;}

    double *fre=fftw_malloc(N*8),*fim=fftw_malloc(N*8);
    double *fro=fftw_malloc(N*8),*fio=fftw_malloc(N*8);
    memcpy(fre,in_re,N*8);memcpy(fim,in_im,N*8);
    fftw_iodim d={.n=(int)N,.is=1,.os=1};
    fftw_iodim h={.n=1,.is=(int)N,.os=(int)N};
    fftw_plan fp=fftw_plan_guru_split_dft(1,&d,1,&h,fre,fim,fro,fio,FFTW_ESTIMATE);
    fftw_execute(fp);

    fn(in_re,in_im,out_re,out_im);
    double max_err=0;
    for(size_t i=0;i<N;i++){double e=fabs(out_re[i]-fro[i])+fabs(out_im[i]-fio[i]);if(e>max_err)max_err=e;}
    printf("  %-28s N=%-6zu err=%.2e %s",label,N,max_err,max_err<1e-10?"OK":"FAIL");

    if(max_err>1e-10){printf("\n");goto cleanup;}

    for(int i=0;i<20;i++)fn(in_re,in_im,out_re,out_im);
    double best=1e18;
    for(int t=0;t<7;t++){double t0=now_ns();
        for(int i=0;i<reps;i++)fn(in_re,in_im,out_re,out_im);
        double ns=(now_ns()-t0)/reps;if(ns<best)best=ns;}
    double fftw_ns=bench_fftw(N,reps);
    printf("  %8.0f ns  (%.2fx vs FFTW %.0f ns)\n",best,fftw_ns/best,fftw_ns);

cleanup:
    fftw_destroy_plan(fp);fftw_free(fre);fftw_free(fim);fftw_free(fro);fftw_free(fio);
    aligned_free(in_re);aligned_free(in_im);aligned_free(out_re);aligned_free(out_im);
}

/* ================================================================
 * Test executors with captured state
 * ================================================================ */

/* N=64 = 8x8 */
static double W64_re[7*8], W64_im[7*8];
static void exec_64(const double*ir,const double*ii,double*or_,double*oi){
    ct_1level((n1_fn)radix8_n1_fwd_avx2,(t1_fn)radix8_t1_dit_fwd_avx2,
              ir,ii,or_,oi,W64_re,W64_im,8,8);
}

/* N=128 = 16x8 */
static double W128_re[15*8], W128_im[15*8];
static void exec_128(const double*ir,const double*ii,double*or_,double*oi){
    ct_1level((n1_fn)radix16_n1_fwd_avx2,(t1_fn)radix16_t1_dit_fwd_avx2,
              ir,ii,or_,oi,W128_re,W128_im,16,8);
}

/* N=256 = 16x16 */
static double W256_re[15*16], W256_im[15*16];
static void exec_256(const double*ir,const double*ii,double*or_,double*oi){
    ct_1level((n1_fn)radix16_n1_fwd_avx2,(t1_fn)radix16_t1_dit_fwd_avx2,
              ir,ii,or_,oi,W256_re,W256_im,16,16);
}

/* N=2048 = 16x(8x16) */
static double W2048_0_re[7*16], W2048_0_im[7*16];
static double W2048_1_re[15*128], W2048_1_im[15*128];
static double tmp2048_re[2048], tmp2048_im[2048];
static void exec_2048(const double*ir,const double*ii,double*or_,double*oi){
    ct_2level((n1_fn)radix16_n1_fwd_avx2,(t1_fn)radix16_t1_dit_fwd_avx2,16,
              (n1_fn)radix8_n1_fwd_avx2,(t1_fn)radix8_t1_dit_fwd_avx2,8,16,
              ir,ii,or_,oi,tmp2048_re,tmp2048_im,
              W2048_0_re,W2048_0_im,W2048_1_re,W2048_1_im);
}

/* N=4096 = 16x(16x16) */
static double W4096_0_re[15*16], W4096_0_im[15*16];
static double W4096_1_re[15*256], W4096_1_im[15*256];
static double tmp4096_re[4096], tmp4096_im[4096];
static void exec_4096(const double*ir,const double*ii,double*or_,double*oi){
    ct_2level((n1_fn)radix16_n1_fwd_avx2,(t1_fn)radix16_t1_dit_fwd_avx2,16,
              (n1_fn)radix16_n1_fwd_avx2,(t1_fn)radix16_t1_dit_fwd_avx2,16,16,
              ir,ii,or_,oi,tmp4096_re,tmp4096_im,
              W4096_0_re,W4096_0_im,W4096_1_re,W4096_1_im);
}

/* ================================================================ */

int main(void) {
    printf("================================================================\n");
    printf("  Permutation-free CT executor (n1 + t1)\n");
    printf("  No permutation, no transpose, no gather/scatter\n");
    printf("================================================================\n\n");
    fflush(stdout);

    init_t1_tw(W64_re,W64_im,8,8);
    init_t1_tw(W128_re,W128_im,16,8);
    init_t1_tw(W256_re,W256_im,16,16);
    init_t1_tw(W2048_0_re,W2048_0_im,8,16);
    init_t1_tw(W2048_1_re,W2048_1_im,16,128);
    init_t1_tw(W4096_0_re,W4096_0_im,16,16);
    init_t1_tw(W4096_1_re,W4096_1_im,16,256);

    test_exec("8x8 (1-level)",64,exec_64,200000);
    test_exec("16x8 (1-level)",128,exec_128,100000);
    test_exec("16x16 (1-level)",256,exec_256,50000);
    test_exec("16x(8x16) (2-level)",2048,exec_2048,5000);
    test_exec("16x(16x16) (2-level)",4096,exec_4096,2000);

    printf("\n");
    return 0;
}
