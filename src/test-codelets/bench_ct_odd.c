/**
 * bench_ct_odd.c -- CT executor for odd radixes (R=3,5,7,25) mixed with pow2
 *
 * Tests 1-level and 2-level factorizations for composite N values
 * containing odd prime factors, using pow2 n1_ovs (child) + odd t1 (parent).
 * Compares against FFTW_MEASURE for correctness + performance.
 *
 * Strategy: odd radix always goes in the t1 (parent) position.
 * Power-of-2 radix goes in the n1_ovs (child) position.
 * This avoids the odd-R n1_ovs bin-scatter alignment issues.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>
#include "bench_compat.h"

/* Power-of-2 codelets: n1_ovs (child DFT) */
#include "fft_radix4_avx2.h"
#include "fft_radix8_avx2.h"
#include "fft_radix16_avx2_ct_n1.h"
#include "fft_radix16_avx2_ct_t1_dit.h"
#include "fft_radix32_avx2_ct_n1.h"
#include "fft_radix32_avx2_ct_t1_dit.h"

/* Odd radix codelets: t1_dit (parent twiddle+butterfly) */
#include "fft_radix3_avx2_ct_t1_dit.h"
#include "fft_radix5_avx2_ct_t1_dit.h"
#include "fft_radix7_avx2_ct_t1_dit.h"
#include "fft_radix25_avx2_ct_t1_dit.h"

/* Also need pow2 t1 for 2-level factorizations */
/* (already included via radix4/8/16/32 headers above) */

/* Also need odd n1_ovs for when odd radix is the child */
#include "fft_radix3_avx2_ct_n1.h"
#include "fft_radix5_avx2_ct_n1.h"
#include "fft_radix7_avx2_ct_n1.h"
#include "fft_radix25_avx2_ct_n1.h"

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

/* Like init_t1_tw but uses explicit N for the phase angle computation.
 * Needed when me != N/R (e.g., M² buffer with odd child radix). */
static void init_t1_tw_N(double *W_re, double *W_im, size_t R, size_t me, size_t N) {
    for (size_t n = 1; n < R; n++)
        for (size_t m = 0; m < me; m++) {
            double a = -2.0 * M_PI * (double)(n * m) / (double)N;
            W_re[(n-1)*me + m] = cos(a);
            W_im[(n-1)*me + m] = sin(a);
        }
}

/* ================================================================
 * Codelet registry: n1_ovs and t1 per radix
 * ================================================================ */

typedef struct {
    size_t R;
    n1_ovs_fn n1_ovs;
    t1_fn     t1_dit;
} codelet_entry;

static codelet_entry codelets[] = {
    /* Power-of-2: have both n1_ovs and t1_dit */
    {  4, (n1_ovs_fn)radix4_n1_ovs_fwd_avx2,  (t1_fn)radix4_t1_dit_fwd_avx2  },
    {  8, (n1_ovs_fn)radix8_n1_ovs_fwd_avx2,  (t1_fn)radix8_t1_dit_fwd_avx2  },
    { 16, (n1_ovs_fn)radix16_n1_ovs_fwd_avx2, (t1_fn)radix16_t1_dit_fwd_avx2 },
    { 32, (n1_ovs_fn)radix32_n1_ovs_fwd_avx2, (t1_fn)radix32_t1_dit_fwd_avx2 },

    /* Odd primes: t1_dit for parent role. n1_ovs available but requires M >= R. */
    {  3, (n1_ovs_fn)radix3_n1_ovs_fwd_avx2,  (t1_fn)radix3_t1_dit_fwd_avx2  },
    {  5, (n1_ovs_fn)radix5_n1_ovs_fwd_avx2,  (t1_fn)radix5_t1_dit_fwd_avx2  },
    {  7, (n1_ovs_fn)radix7_n1_ovs_fwd_avx2,  (t1_fn)radix7_t1_dit_fwd_avx2  },
    { 25, (n1_ovs_fn)radix25_n1_ovs_fwd_avx2, (t1_fn)radix25_t1_dit_fwd_avx2 },

    {  0, NULL, NULL }
};

static codelet_entry *find_codelet(size_t R) {
    for (codelet_entry *c = codelets; c->R; c++)
        if (c->R == R) return c;
    return NULL;
}

/* ================================================================
 * 1-level CT executor: n1_ovs (child) + t1 (parent)
 * ================================================================ */

static void ct_1level(n1_ovs_fn n1, t1_fn t1,
    const double *ir, const double *ii,
    double *or_, double *oi,
    double *tmp_re, double *tmp_im,
    const double *W_re, const double *W_im,
    size_t R, size_t M)
{
    /* n1_ovs: is=R (stride between bins), os=1, vl=M (columns per bin), ovs=M
     *
     * The n1_ovs output layout uses M*M positions (M columns × M stride).
     * When R < M (odd child with pow2 parent), positions beyond R*M are
     * written by the SIMD scatter but contain valid data from the DFT.
     * The t1 then operates on the M*M buffer.
     * After t1, the valid DFT output is in the first R*M = N elements.
     *
     * tmp buffer must be M*M elements (not just N = R*M).
     */
    size_t N = R * M;
    memset(tmp_re, 0, M * M * sizeof(double));
    memset(tmp_im, 0, M * M * sizeof(double));
    /* is=M: bins are columns of R×M matrix, stride M between columns.
     * os=1, vl=M (process M elements per bin), ovs=M. */
    n1(ir, ii, tmp_re, tmp_im, M, 1, M, M);
    t1(tmp_re, tmp_im, W_re, W_im, M, M);
    memcpy(or_, tmp_re, N * sizeof(double));
    memcpy(oi,  tmp_im, N * sizeof(double));
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
    double *inner_tmp_re, double *inner_tmp_im,
    const double *W0_re, const double *W0_im,
    const double *W1_re, const double *W1_im)
{
    size_t M1 = R0 * M0;
    /* Outer n1_ovs */
    n1_outer(ir, ii, tmp_re, tmp_im, R1, 1, M1, M1);
    /* Inner 1-level CT on each block */
    for (size_t r = 0; r < R1; r++)
        ct_1level(n1_inner, t1_inner,
                  tmp_re + r*M1, tmp_im + r*M1,
                  or_ + r*M1, oi + r*M1,
                  inner_tmp_re, inner_tmp_im,
                  W0_re, W0_im, R0, M0);
    /* Outer t1 */
    t1_outer(or_, oi, W1_re, W1_im, M1, M1);
}

/* ================================================================ */

static double bench_fftw(size_t N, int reps) {
    double *ri=fftw_malloc(N*8),*ii=fftw_malloc(N*8);
    double *ro=fftw_malloc(N*8),*io=fftw_malloc(N*8);
    for(size_t i=0;i<N;i++){ri[i]=(double)rand()/RAND_MAX-.5;ii[i]=(double)rand()/RAND_MAX-.5;}
    fftw_iodim d={.n=(int)N,.is=1,.os=1};
    fftw_iodim h={.n=1,.is=1,.os=1};
    fftw_plan p=fftw_plan_guru_split_dft(1,&d,1,&h,ri,ii,ro,io,FFTW_MEASURE);
    if(!p){fftw_free(ri);fftw_free(ii);fftw_free(ro);fftw_free(io);return -1;}
    for(size_t i=0;i<N;i++){ri[i]=(double)rand()/RAND_MAX-.5;ii[i]=(double)rand()/RAND_MAX-.5;}
    for(int i=0;i<20;i++)fftw_execute(p);
    double best=1e18;
    for(int t=0;t<7;t++){double t0=now_ns();
        for(int i=0;i<reps;i++)fftw_execute_split_dft(p,ri,ii,ro,io);
        double ns=(now_ns()-t0)/reps;if(ns<best)best=ns;}
    fftw_destroy_plan(p);fftw_free(ri);fftw_free(ii);fftw_free(ro);fftw_free(io);
    return best;
}

static double check_vs_fftw(size_t N, const double *in_re, const double *in_im,
                             const double *out_re, const double *out_im) {
    double *fre=fftw_malloc(N*8),*fim=fftw_malloc(N*8);
    double *fro=fftw_malloc(N*8),*fio=fftw_malloc(N*8);
    memcpy(fre,in_re,N*8); memcpy(fim,in_im,N*8);
    fftw_iodim d={.n=(int)N,.is=1,.os=1};
    fftw_iodim h={.n=1,.is=1,.os=1};
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
 * Globals for benchmark closures
 * ================================================================ */

static n1_ovs_fn g_n1; static t1_fn g_t1;
static double *g_W_re, *g_W_im;
static double *g_tmp_re, *g_tmp_im;
static size_t g_R, g_M;

__attribute__((noinline))
static void exec_1level(const double *ir, const double *ii, double *or_, double *oi) {
    ct_1level(g_n1, g_t1, ir, ii, or_, oi, g_tmp_re, g_tmp_im, g_W_re, g_W_im, g_R, g_M);
}

static n1_ovs_fn g2_n1_outer, g2_n1_inner;
static t1_fn g2_t1_outer, g2_t1_inner;
static double *g2_W0_re, *g2_W0_im, *g2_W1_re, *g2_W1_im;
static double *g2_tmp_re, *g2_tmp_im;
static double *g2_inner_tmp_re, *g2_inner_tmp_im;
static size_t g2_R1, g2_R0, g2_M0;

__attribute__((noinline))
static void exec_2level(const double *ir, const double *ii, double *or_, double *oi) {
    ct_2level(g2_n1_outer, g2_t1_outer, g2_R1,
              g2_n1_inner, g2_t1_inner, g2_R0, g2_M0,
              ir, ii, or_, oi, g2_tmp_re, g2_tmp_im,
              g2_inner_tmp_re, g2_inner_tmp_im,
              g2_W0_re, g2_W0_im, g2_W1_re, g2_W1_im);
}

static double bench_exec(size_t N, const double *in_re, const double *in_im,
                         void (*fn)(const double*,const double*,double*,double*),
                         int reps) {
    double *or_ = (double*)aligned_alloc(32, (N+64)*8);
    double *oi_ = (double*)aligned_alloc(32, (N+64)*8);
    for(int i=0;i<20;i++) fn(in_re,in_im,or_,oi_);
    double best=1e18;
    for(int t=0;t<7;t++){
        double t0=now_ns();
        for(int i=0;i<reps;i++) fn(in_re,in_im,or_,oi_);
        double ns=(now_ns()-t0)/reps;
        if(ns<best)best=ns;
    }
    aligned_free(or_); aligned_free(oi_);
    return best;
}

/* ================================================================
 * Test driver
 * ================================================================ */

static int is_pow2(size_t n) {
    return n > 0 && (n & (n-1)) == 0;
}

static void test_N(size_t N) {
    int reps = (int)(2e6 / (N+1));
    if (reps < 100) reps = 100;
    if (reps > 2000000) reps = 2000000;

    size_t alloc_sz = (N + 256) * 8;
    double *in_re = (double*)aligned_alloc(32, alloc_sz);
    double *in_im = (double*)aligned_alloc(32, alloc_sz);
    double *out_re = (double*)aligned_alloc(32, alloc_sz);
    double *out_im = (double*)aligned_alloc(32, alloc_sz);
    memset(in_re, 0, alloc_sz); memset(in_im, 0, alloc_sz);

    srand(42);
    for (size_t i = 0; i < N; i++) {
        in_re[i] = (double)rand()/RAND_MAX - 0.5;
        in_im[i] = (double)rand()/RAND_MAX - 0.5;
    }

    double fftw_ns = bench_fftw(N, reps);
    printf("\n  N=%-6zu FFTW=%.0f ns\n", N, fftw_ns);
    printf("  %-30s %8s %10s %8s\n", "Factorization", "ns", "vs FFTW", "err");

    double best_ns = 1e18;
    char best_label[64] = "";

    /* Try all 1-level factorizations: N = R_n1 * R_t1
     *
     * n1_ovs of radix R_n1 (child DFT): processes M = R_t1 columns per bin
     *   Call: n1(is=R_n1, os=1, vl=M, ovs=M)  where M = N/R_n1
     *   Constraint: M must be >= 4 (SIMD) and M >= R_n1 (bins fit in output row)
     *
     * t1 of radix R_t1 (parent twiddle+butterfly): operates on M = R_n1 columns
     *   Call: t1(ios=M, me=M)  where M = N/R_t1
     *   Constraint: M must be multiple of 4 (SIMD m-loop)
     *
     * For mixed odd×pow2: odd as n1_ovs, pow2 as t1.
     * The pow2 t1 ensures me is a multiple of 4.
     * The odd n1_ovs with vl=M works because M = pow2 >= 4.
     */
    size_t all_radixes[] = {3, 4, 5, 7, 8, 16, 25, 32};
    int n_all = (int)(sizeof(all_radixes)/sizeof(all_radixes[0]));

    for (int ni = 0; ni < n_all; ni++) {
        size_t R_n1 = all_radixes[ni];      /* n1_ovs radix (child) */
        if (N % R_n1 != 0) continue;
        size_t R_t1 = N / R_n1;             /* t1 radix (parent) */

        /* Skip pure pow2 factorizations — only test mixed odd×pow2 */
        if (is_pow2(R_n1) && is_pow2(R_t1)) continue;

        codelet_entry *ce_n1 = find_codelet(R_n1);
        codelet_entry *ce_t1 = find_codelet(R_t1);
        if (!ce_n1 || !ce_n1->n1_ovs || !ce_t1 || !ce_t1->t1_dit) continue;

        size_t M = R_t1;  /* columns processed by n1_ovs, also me for t1 */

        /* t1 constraint: me must be multiple of VL=4 */
        if (M % 4 != 0) continue;
        if (M < 4) continue;

        /* n1_ovs constraint: M >= R_n1 (bins fit in output row of width M) */
        if (M < R_n1) continue;

        /* Setup twiddles for the t1 (parent).
         * me = M = R_t1 (columns to process in the M² buffer),
         * but phase angle uses actual N = R_n1 * R_t1. */
        double *tw_re = (double*)aligned_alloc(32, (R_t1-1)*M*8 + 64);
        double *tw_im = (double*)aligned_alloc(32, (R_t1-1)*M*8 + 64);
        init_t1_tw_N(tw_re, tw_im, R_t1, M, N);

        /* M*M tmp buffer for n1_ovs when R < M */
        double *tmp1_re = (double*)aligned_alloc(32, M*M*8 + 64);
        double *tmp1_im = (double*)aligned_alloc(32, M*M*8 + 64);

        g_n1 = ce_n1->n1_ovs;
        g_t1 = ce_t1->t1_dit;
        g_R = R_n1; g_M = M;
        g_W_re = tw_re; g_W_im = tw_im;
        g_tmp_re = tmp1_re; g_tmp_im = tmp1_im;

        memset(out_re, 0, N*8); memset(out_im, 0, N*8);
        exec_1level(in_re, in_im, out_re, out_im);
        double err = check_vs_fftw(N, in_re, in_im, out_re, out_im);

        char label[64];
        snprintf(label, sizeof(label), "n1_R%zu + t1_R%zu (1L)", R_n1, R_t1);

        if (err < 1e-9) {
            double ns = bench_exec(N, in_re, in_im, exec_1level, reps);
            printf("  %-30s %8.0f %9.2fx  %.1e%s\n", label, ns, fftw_ns/ns, err,
                   ns < best_ns ? "  <-- best" : "");
            if (ns < best_ns) { best_ns = ns; snprintf(best_label, 64, "%s", label); }
        } else {
            printf("  %-30s %8s %10s  %.1e  FAIL\n", label, "--", "--", err);
        }

        aligned_free(tw_re); aligned_free(tw_im);
        aligned_free(tmp1_re); aligned_free(tmp1_im);
    }

    /* Try 2-level: N = R1_n1 * R1_t1 * M0  (outer n1 + [inner n1 + inner t1] + outer t1)
     * Actually the 2-level structure is:
     *   N = R_outer * M_outer, where M_outer = R_inner * M_inner
     *   Outer: n1_ovs(R_outer) decimates, then inner ct_1level on each block, then t1(R_outer)
     *
     * Constraints:
     *   Outer n1_ovs: M_outer >= R_outer, M_outer % 4 == 0
     *   Outer t1: M_outer % 4 == 0 (its me = M_outer)
     *   Inner n1_ovs: M_inner >= R_inner, M_inner % 4 == 0
     *   Inner t1: M_inner % 4 == 0 (its me = M_inner) */
    for (int outer_i = 0; outer_i < n_all; outer_i++) {
        size_t R_outer = all_radixes[outer_i];
        if (N % R_outer != 0) continue;
        size_t M_outer = N / R_outer;
        if (M_outer < R_outer) continue;
        if (M_outer % 4 != 0) continue;

        codelet_entry *ce_n1_out = find_codelet(R_outer);
        codelet_entry *ce_t1_out = find_codelet(R_outer);
        if (!ce_n1_out || !ce_n1_out->n1_ovs) continue;
        if (!ce_t1_out || !ce_t1_out->t1_dit) continue;

        for (int inner_i = 0; inner_i < n_all; inner_i++) {
            size_t R_inner = all_radixes[inner_i];
            if (M_outer % R_inner != 0) continue;
            size_t M_inner = M_outer / R_inner;
            if (M_inner < 4) continue;
            if (M_inner % 4 != 0) continue;
            if (M_inner < R_inner) continue;

            /* Skip pure pow2 factorizations */
            if (is_pow2(R_outer) && is_pow2(R_inner)) continue;

            codelet_entry *ce_n1_in = find_codelet(R_inner);
            codelet_entry *ce_t1_in = find_codelet(R_inner);
            if (!ce_n1_in || !ce_n1_in->n1_ovs) continue;
            if (!ce_t1_in || !ce_t1_in->t1_dit) continue;

            /* Skip pure identity (R_outer=1 etc.) */
            if (R_outer < 2 || R_inner < 2) continue;

            g2_n1_outer = ce_n1_out->n1_ovs;
            g2_t1_outer = ce_t1_out->t1_dit;
            g2_R1 = R_outer;
            g2_n1_inner = ce_n1_in->n1_ovs;
            g2_t1_inner = ce_t1_in->t1_dit;
            g2_R0 = R_inner; g2_M0 = M_inner;

            g2_W0_re = (double*)aligned_alloc(32, (R_inner-1)*M_inner*8 + 64);
            g2_W0_im = (double*)aligned_alloc(32, (R_inner-1)*M_inner*8 + 64);
            g2_W1_re = (double*)aligned_alloc(32, (R_outer-1)*M_outer*8 + 64);
            g2_W1_im = (double*)aligned_alloc(32, (R_outer-1)*M_outer*8 + 64);
            /* Outer tmp: M_outer*M_outer for odd-R n1_ovs layout */
            g2_tmp_re = (double*)aligned_alloc(32, M_outer*M_outer*8 + 64);
            g2_tmp_im = (double*)aligned_alloc(32, M_outer*M_outer*8 + 64);
            /* Inner tmp: M_inner*M_inner for inner ct_1level */
            g2_inner_tmp_re = (double*)aligned_alloc(32, M_inner*M_inner*8 + 64);
            g2_inner_tmp_im = (double*)aligned_alloc(32, M_inner*M_inner*8 + 64);
            /* Inner twiddle: N_inner = R_inner * M_inner = M_outer */
            init_t1_tw_N(g2_W0_re, g2_W0_im, R_inner, M_inner, R_inner * M_inner);
            /* Outer twiddle: N_outer = R_outer * M_outer = N */
            init_t1_tw_N(g2_W1_re, g2_W1_im, R_outer, M_outer, N);

            memset(out_re, 0, N*8); memset(out_im, 0, N*8);
            exec_2level(in_re, in_im, out_re, out_im);
            double err = check_vs_fftw(N, in_re, in_im, out_re, out_im);

            char label[64];
            snprintf(label, sizeof(label), "%zux(%zux%zu) (2L)", R_outer, R_inner, M_inner);

            if (err < 1e-9) {
                double ns = bench_exec(N, in_re, in_im, exec_2level, reps);
                printf("  %-30s %8.0f %9.2fx  %.1e%s\n", label, ns, fftw_ns/ns, err,
                       ns < best_ns ? "  <-- best" : "");
                if (ns < best_ns) { best_ns = ns; snprintf(best_label, 64, "%s", label); }
            } else {
                printf("  %-30s %8s %10s  %.1e  FAIL\n", label, "--", "--", err);
            }

            aligned_free(g2_W0_re); aligned_free(g2_W0_im);
            aligned_free(g2_W1_re); aligned_free(g2_W1_im);
            aligned_free(g2_tmp_re); aligned_free(g2_tmp_im);
            aligned_free(g2_inner_tmp_re); aligned_free(g2_inner_tmp_im);
        }
    }

    if (best_ns < 1e18)
        printf("  >>> Best: %-30s  %.0f ns  (%.2fx FFTW)\n", best_label, best_ns, fftw_ns/best_ns);

    aligned_free(in_re); aligned_free(in_im);
    aligned_free(out_re); aligned_free(out_im);
}

/* ================================================================ */

int main(void) {
    srand(42);
    printf("VectorFFT Odd-Radix CT Executor Benchmark\n");
    printf("Strategy: pow2 n1_ovs (child) + odd t1_dit (parent)\n\n");

    /* Composite sizes with odd factors × power-of-2 */
    size_t test_sizes[] = {
        /* R=3 × pow2 */
        12, 24, 48, 96, 192, 384, 768,
        /* R=5 × pow2 */
        20, 40, 80, 160, 320, 640,
        /* R=7 × pow2 */
        28, 56, 112, 224, 448,
        /* R=25 × pow2 */
        100, 200, 400, 800,
        /* Multi-odd: 3×5=15, 5×7=35, etc. × pow2 */
        60, 120, 140, 280, 300, 600,
        /* Larger smooth numbers with odd factors */
        1200, 1500, 2400, 3000,
    };
    int n_tests = (int)(sizeof(test_sizes)/sizeof(test_sizes[0]));

    for (int i = 0; i < n_tests; i++)
        test_N(test_sizes[i]);

    printf("\nDone.\n");
    return 0;
}
