/**
 * bench_ct_factor.c -- Find best factorization for permutation-free CT
 *
 * Tests all valid 1-level and 2-level factorizations for each N.
 * Reports correctness + timing for each, highlights the winner.
 * Compares best against FFTW_MEASURE.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>
#include "bench_compat.h"

#include "fft_radix4_avx2.h"
#include "fft_radix8_avx2.h"
#include "fft_radix16_avx2_ct_n1.h"
#include "fft_radix16_avx2_ct_t1_dit.h"
#include "fft_radix32_avx2_notw.h"
#include "fft_radix32_avx2_dit_tw.h"
#include "r64_unified_avx2.h"

/* Also need R=32/R=64 CT headers for n1_ovs and t1_dit */
/* R=32 ct_n1 has radix32_n1_ovs_fwd_avx2 */
/* R=64 ct_n1 has radix64_n1_ovs_fwd_avx2 -- in r64_unified_avx2.h via --variant all */

typedef void (*n1_ovs_fn)(const double*, const double*, double*, double*,
                          size_t, size_t, size_t, size_t);
typedef void (*t1_fn)(double*, double*, const double*, const double*,
                      size_t, size_t);

/* ================================================================ */

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
 * Codelet registry
 * ================================================================ */

typedef struct {
    size_t R;
    n1_ovs_fn n1_ovs;
    t1_fn t1_dit;
} codelet_entry;

static codelet_entry codelets[] = {
    { 4,  (n1_ovs_fn)radix4_n1_ovs_fwd_avx2,  (t1_fn)radix4_t1_dit_fwd_avx2 },
    { 8,  (n1_ovs_fn)radix8_n1_ovs_fwd_avx2,   (t1_fn)radix8_t1_dit_fwd_avx2 },
    { 16, (n1_ovs_fn)radix16_n1_ovs_fwd_avx2,  (t1_fn)radix16_t1_dit_fwd_avx2 },
    /* R=32 and R=64 n1_ovs are in their ct_n1 headers */
    /* { 32, (n1_ovs_fn)radix32_n1_ovs_fwd_avx2, (t1_fn)radix32_t1_dit_fwd_avx2 }, */
    /* { 64, (n1_ovs_fn)radix64_n1_ovs_fwd_avx2, (t1_fn)radix64_t1_dit_fwd_avx2 }, */
};
#define N_CODELETS (sizeof(codelets)/sizeof(codelets[0]))

static codelet_entry *find_codelet(size_t R) {
    for (size_t i = 0; i < N_CODELETS; i++)
        if (codelets[i].R == R) return &codelets[i];
    return NULL;
}

/* ================================================================
 * 1-level CT executor
 * ================================================================ */

static void ct_1level(n1_ovs_fn n1, t1_fn t1,
    const double *ir, const double *ii,
    double *or_, double *oi,
    const double *W_re, const double *W_im,
    size_t R, size_t M)
{
    n1(ir, ii, or_, oi, R, 1, R, M);
    t1(or_, oi, W_re, W_im, R, M);
}

/* ================================================================
 * 2-level CT executor
 * ================================================================ */

static void ct_2level(
    n1_ovs_fn n1_outer, t1_fn t1_outer, size_t R1,
    n1_ovs_fn n1_inner, t1_fn t1_inner, size_t R0, size_t M0,
    const double *ir, const double *ii,
    double *or_, double *oi,
    double *tmp_re, double *tmp_im,
    const double *W0_re, const double *W0_im,
    const double *W1_re, const double *W1_im)
{
    size_t M1 = R0 * M0;
    /* Outer n1_ovs: decimates input into R1 contiguous blocks of M1 */
    n1_outer(ir, ii, tmp_re, tmp_im, R1, 1, R1, M1);
    /* Inner 1-level CT on each block */
    for (size_t r = 0; r < R1; r++)
        ct_1level(n1_inner, t1_inner,
                  tmp_re + r*M1, tmp_im + r*M1,
                  or_ + r*M1, oi + r*M1,
                  W0_re, W0_im, R0, M0);
    /* Outer t1 combines */
    t1_outer(or_, oi, W1_re, W1_im, M1, M1);
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
 * Correctness check
 * ================================================================ */

static double check_vs_fftw(size_t N, const double *in_re, const double *in_im,
                             const double *out_re, const double *out_im) {
    double *fre=fftw_malloc(N*8),*fim=fftw_malloc(N*8);
    double *fro=fftw_malloc(N*8),*fio=fftw_malloc(N*8);
    memcpy(fre,in_re,N*8); memcpy(fim,in_im,N*8);
    fftw_iodim d={.n=(int)N,.is=1,.os=1};
    fftw_iodim h={.n=1,.is=(int)N,.os=(int)N};
    fftw_plan fp=fftw_plan_guru_split_dft(1,&d,1,&h,fre,fim,fro,fio,FFTW_ESTIMATE);
    fftw_execute(fp);
    double max_err=0;
    for(size_t i=0;i<N;i++){
        double e=fabs(out_re[i]-fro[i])+fabs(out_im[i]-fio[i]);
        if(e>max_err)max_err=e;
    }
    fftw_destroy_plan(fp);fftw_free(fre);fftw_free(fim);fftw_free(fro);fftw_free(fio);
    return max_err;
}

/* ================================================================
 * Benchmark one configuration
 * ================================================================ */

static double bench_once(size_t N, const double *in_re, const double *in_im,
                         double *out_re, double *out_im,
                         void (*fn)(const double*,const double*,double*,double*),
                         int reps) {
    for(int i=0;i<20;i++) fn(in_re,in_im,out_re,out_im);
    double best=1e18;
    for(int t=0;t<7;t++){
        double t0=now_ns();
        for(int i=0;i<reps;i++) fn(in_re,in_im,out_re,out_im);
        double ns=(now_ns()-t0)/reps;
        if(ns<best)best=ns;
    }
    return best;
}

/* ================================================================
 * Test all 1-level factorizations for given N
 * ================================================================ */

/* Captured state for executor closures */
static n1_ovs_fn g_n1; static t1_fn g_t1;
static double *g_W_re, *g_W_im;
static size_t g_R, g_M;

static void exec_1level(const double *ir, const double *ii, double *or_, double *oi) {
    ct_1level(g_n1, g_t1, ir, ii, or_, oi, g_W_re, g_W_im, g_R, g_M);
}

static n1_ovs_fn g2_n1_outer, g2_n1_inner;
static t1_fn g2_t1_outer, g2_t1_inner;
static double *g2_W0_re, *g2_W0_im, *g2_W1_re, *g2_W1_im;
static double *g2_tmp_re, *g2_tmp_im;
static size_t g2_R1, g2_R0, g2_M0;

static void exec_2level(const double *ir, const double *ii, double *or_, double *oi) {
    ct_2level(g2_n1_outer, g2_t1_outer, g2_R1,
              g2_n1_inner, g2_t1_inner, g2_R0, g2_M0,
              ir, ii, or_, oi, g2_tmp_re, g2_tmp_im,
              g2_W0_re, g2_W0_im, g2_W1_re, g2_W1_im);
}

static void test_N(size_t N) {
    int reps = (int)(2e6 / (N+1));
    if (reps < 200) reps = 200;
    if (reps > 2000000) reps = 2000000;

    double *in_re = (double*)aligned_alloc(32, N*8);
    double *in_im = (double*)aligned_alloc(32, N*8);
    double *out_re = (double*)aligned_alloc(32, N*8);
    double *out_im = (double*)aligned_alloc(32, N*8);

    srand(42);
    for (size_t i = 0; i < N; i++) {
        in_re[i] = (double)rand()/RAND_MAX - 0.5;
        in_im[i] = (double)rand()/RAND_MAX - 0.5;
    }

    double fftw_ns = bench_fftw(N, reps);

    printf("\n  N=%zu  FFTW=%.0f ns\n", N, fftw_ns);
    printf("  %-20s %8s %10s %8s\n", "Factorization", "ns", "vs FFTW", "err");
    fflush(stdout);

    double best_ns = 1e18;
    char best_label[64] = "";

    /* 1-level: N = R * M, need codelet for both R (t1) and M (n1 child DFT) */
    size_t radixes[] = {4, 8, 16};
    int n_radixes = 3;

    for (int ri = 0; ri < n_radixes; ri++) {
        size_t R = radixes[ri];
        if (N % R != 0) continue;
        size_t M = N / R;
        if (M < 4) continue; /* need vl >= VL=4 */

        /* Need n1_ovs for DFT-M (child) and t1 for radix-R (butterfly) */
        codelet_entry *child = find_codelet(M);  /* n1_ovs for M-point DFT */
        codelet_entry *outer = find_codelet(R);   /* t1 for R-point butterfly */
        if (!child || !outer) continue;
        printf("    Trying %zux%zu ... ", R, M); fflush(stdout);

        /* Setup */
        g_n1 = child->n1_ovs;
        g_t1 = outer->t1_dit;
        g_R = R; g_M = M;
        g_W_re = (double*)aligned_alloc(32, (R-1)*M*8);
        g_W_im = (double*)aligned_alloc(32, (R-1)*M*8);
        init_t1_tw(g_W_re, g_W_im, R, M);

        /* Correctness */
        exec_1level(in_re, in_im, out_re, out_im);
        double err = check_vs_fftw(N, in_re, in_im, out_re, out_im);

        char label[64];
        snprintf(label, sizeof(label), "%zux%zu (1L)", R, M);

        if (err < 1e-10) {
            double ns = bench_once(N, in_re, in_im, out_re, out_im, exec_1level, reps);
            printf("  %-20s %8.0f %9.2fx  %.1e %s\n", label, ns, fftw_ns/ns, err,
                   ns < best_ns ? " <-- best" : "");
            if (ns < best_ns) { best_ns = ns; snprintf(best_label, 64, "%s", label); }
        } else {
            printf("  %-20s %8s %10s  %.1e FAIL\n", label, "--", "--", err);
        }

        aligned_free(g_W_re); aligned_free(g_W_im);
    }

    /* 2-level: N = R1 * R0 * M0 */
    for (int r1i = 0; r1i < n_radixes; r1i++) {
        size_t R1 = radixes[r1i];
        if (N % R1 != 0) continue;
        size_t M1 = N / R1;

        for (int r0i = 0; r0i < n_radixes; r0i++) {
            size_t R0 = radixes[r0i];
            if (M1 % R0 != 0) continue;
            size_t M0 = M1 / R0;
            if (M0 < 4) continue;

            codelet_entry *c_inner = find_codelet(M0);  /* n1_ovs for innermost */
            codelet_entry *t_inner = find_codelet(R0);   /* t1 for inner butterfly */
            codelet_entry *c_outer = find_codelet(R1);   /* n1_ovs for outer decimation (radix R1) */
            codelet_entry *t_outer = find_codelet(R1);   /* t1 for outer butterfly */
            if (!c_inner || !t_inner || !c_outer || !t_outer) continue;

            /* Setup */
            g2_n1_outer = c_outer->n1_ovs;
            g2_t1_outer = t_outer->t1_dit;
            g2_R1 = R1;
            g2_n1_inner = c_inner->n1_ovs;
            g2_t1_inner = t_inner->t1_dit;
            g2_R0 = R0; g2_M0 = M0;

            g2_W0_re = (double*)aligned_alloc(32, (R0-1)*M0*8);
            g2_W0_im = (double*)aligned_alloc(32, (R0-1)*M0*8);
            g2_W1_re = (double*)aligned_alloc(32, (R1-1)*M1*8);
            g2_W1_im = (double*)aligned_alloc(32, (R1-1)*M1*8);
            g2_tmp_re = (double*)aligned_alloc(32, N*8);
            g2_tmp_im = (double*)aligned_alloc(32, N*8);
            init_t1_tw(g2_W0_re, g2_W0_im, R0, M0);
            init_t1_tw(g2_W1_re, g2_W1_im, R1, M1);

            exec_2level(in_re, in_im, out_re, out_im);
            double err = check_vs_fftw(N, in_re, in_im, out_re, out_im);

            char label[64];
            snprintf(label, sizeof(label), "%zux(%zux%zu) (2L)", R1, R0, M0);

            if (err < 1e-10) {
                double ns = bench_once(N, in_re, in_im, out_re, out_im, exec_2level, reps);
                printf("  %-20s %8.0f %9.2fx  %.1e %s\n", label, ns, fftw_ns/ns, err,
                       ns < best_ns ? " <-- best" : "");
                if (ns < best_ns) { best_ns = ns; snprintf(best_label, 64, "%s", label); }
            } else {
                printf("  %-20s %8s %10s  %.1e FAIL\n", label, "--", "--", err);
            }

            aligned_free(g2_W0_re); aligned_free(g2_W0_im);
            aligned_free(g2_W1_re); aligned_free(g2_W1_im);
            aligned_free(g2_tmp_re); aligned_free(g2_tmp_im);
        }
    }

    if (best_ns < 1e18)
        printf("  >> BEST: %-20s %.0f ns (%.2fx vs FFTW)\n", best_label, best_ns, fftw_ns/best_ns);
    fflush(stdout);

    aligned_free(in_re); aligned_free(in_im);
    aligned_free(out_re); aligned_free(out_im);
}

/* ================================================================ */

int main(void) {
    printf("================================================================\n");
    printf("  CT Factorization Search\n");
    printf("  Permutation-free: n1_ovs + t1, all valid factorizations\n");
    printf("  Radixes available: 4, 8, 16\n");
    printf("================================================================\n");
    fflush(stdout);

    size_t sizes[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    int n_sizes = sizeof(sizes)/sizeof(sizes[0]);

    for (int i = 0; i < n_sizes; i++)
        test_N(sizes[i]);

    printf("\n");
    return 0;
}
