/**
 * bench_recursive_ct.c -- Permutation-free CT: systematic factorization search
 *
 * Tests all valid 1-level R*M factorizations for each N using
 * n1_ovs (SIMD transpose stores) + t1_dit (in-place twiddle).
 * Each factorization verified against FFTW, best highlighted.
 *
 * Available radixes: 4, 8, 16, 32, 64
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>
#include "bench_compat.h"

/* All CT codelet headers */
#include "fft_radix4_avx2.h"
#include "fft_radix8_avx2.h"
#include "fft_radix16_avx2_ct_n1.h"
#include "fft_radix16_avx2_ct_t1_dit.h"
#include "fft_radix32_avx2_ct_n1.h"
#include "fft_radix32_avx2_ct_t1_dit.h"
#include "r64_unified_avx2.h"

typedef void (*n1_ovs_fn)(const double*, const double*, double*, double*,
                          size_t, size_t, size_t, size_t);
typedef void (*t1_fn)(double*, double*, const double*, const double*,
                      size_t, size_t);

/* ================================================================ */

static void init_tw(double *W_re, double *W_im, size_t R, size_t me) {
    size_t N = R * me;
    for (size_t n = 1; n < R; n++)
        for (size_t m = 0; m < me; m++) {
            double a = -2.0 * M_PI * (double)(n * m) / (double)N;
            W_re[(n-1)*me + m] = cos(a);
            W_im[(n-1)*me + m] = sin(a);
        }
}

/* ================================================================
 * Codelet table
 * ================================================================ */

typedef struct { size_t R; n1_ovs_fn n1; t1_fn t1; } codelet_t;

static const codelet_t CODELETS[] = {
    {  4, (n1_ovs_fn)radix4_n1_ovs_fwd_avx2,  (t1_fn)radix4_t1_dit_fwd_avx2  },
    {  8, (n1_ovs_fn)radix8_n1_ovs_fwd_avx2,  (t1_fn)radix8_t1_dit_fwd_avx2  },
    { 16, (n1_ovs_fn)radix16_n1_ovs_fwd_avx2, (t1_fn)radix16_t1_dit_fwd_avx2 },
    { 32, (n1_ovs_fn)radix32_n1_ovs_fwd_avx2, (t1_fn)radix32_t1_dit_fwd_avx2 },
    { 64, (n1_ovs_fn)radix64_n1_ovs_fwd_avx2, (t1_fn)radix64_t1_dit_fwd_avx2 },
    {  0, NULL, NULL }
};

static const codelet_t *find(size_t R) {
    for (const codelet_t *c = CODELETS; c->R; c++)
        if (c->R == R) return c;
    return NULL;
}

/* ================================================================
 * 1-level CT: n1_ovs + t1
 * ================================================================ */

static void ct_1level(n1_ovs_fn n1, t1_fn t1,
    const double *ir, const double *ii, double *or_, double *oi,
    const double *W_re, const double *W_im, size_t R, size_t M)
{
    n1(ir, ii, or_, oi, R, 1, R, M);
    t1(or_, oi, W_re, W_im, M, M);
}

/* ================================================================
 * Fused 1-level CT: direct calls (compiler can inline both)
 * Eliminates function-pointer indirection so ICX can inline
 * the static-inline n1_ovs and t1 bodies into one function.
 * ================================================================ */

typedef void (*fused_fn)(const double*, const double*,
                         double*, double*,
                         const double*, const double*);

#define FUSED_1LEVEL(name, n1_func, t1_func, R_val, M_val)       \
static void name(                                                  \
    const double * __restrict__ ir, const double * __restrict__ ii,\
    double * __restrict__ or_, double * __restrict__ oi,           \
    const double * __restrict__ W_re, const double * __restrict__ W_im) \
{                                                                  \
    n1_func(ir, ii, or_, oi, R_val, 1, R_val, M_val);             \
    t1_func(or_, oi, W_re, W_im, M_val, M_val);                   \
}

/* N=16 */
FUSED_1LEVEL(fused_4x4,   radix4_n1_ovs_fwd_avx2,  radix4_t1_dit_fwd_avx2,   4,  4)
/* N=32 */
FUSED_1LEVEL(fused_4x8,   radix8_n1_ovs_fwd_avx2,  radix4_t1_dit_fwd_avx2,   4,  8)
/* N=64 */
FUSED_1LEVEL(fused_4x16,  radix16_n1_ovs_fwd_avx2, radix4_t1_dit_fwd_avx2,   4, 16)
FUSED_1LEVEL(fused_8x8,   radix8_n1_ovs_fwd_avx2,  radix8_t1_dit_fwd_avx2,   8,  8)
/* N=128 */
FUSED_1LEVEL(fused_4x32,  radix32_n1_ovs_fwd_avx2, radix4_t1_dit_fwd_avx2,   4, 32)
FUSED_1LEVEL(fused_8x16,  radix16_n1_ovs_fwd_avx2, radix8_t1_dit_fwd_avx2,   8, 16)
/* N=256 */
FUSED_1LEVEL(fused_4x64,  radix64_n1_ovs_fwd_avx2, radix4_t1_dit_fwd_avx2,   4, 64)
FUSED_1LEVEL(fused_8x32,  radix32_n1_ovs_fwd_avx2, radix8_t1_dit_fwd_avx2,   8, 32)
FUSED_1LEVEL(fused_16x16, radix16_n1_ovs_fwd_avx2, radix16_t1_dit_fwd_avx2, 16, 16)
/* N=512 */
FUSED_1LEVEL(fused_8x64,  radix64_n1_ovs_fwd_avx2, radix8_t1_dit_fwd_avx2,   8, 64)
FUSED_1LEVEL(fused_16x32, radix32_n1_ovs_fwd_avx2, radix16_t1_dit_fwd_avx2, 16, 32)
/* N=1024 */
FUSED_1LEVEL(fused_16x64, radix64_n1_ovs_fwd_avx2, radix16_t1_dit_fwd_avx2, 16, 64)
FUSED_1LEVEL(fused_32x32, radix32_n1_ovs_fwd_avx2, radix32_t1_dit_fwd_avx2, 32, 32)
/* N=2048 */
FUSED_1LEVEL(fused_32x64, radix64_n1_ovs_fwd_avx2, radix32_t1_dit_fwd_avx2, 32, 64)
/* N=4096 */
FUSED_1LEVEL(fused_64x64, radix64_n1_ovs_fwd_avx2, radix64_t1_dit_fwd_avx2, 64, 64)

typedef struct { size_t R; size_t M; fused_fn fn; } fused_entry;

static const fused_entry FUSED[] = {
    {  4,   4, fused_4x4   },
    {  4,   8, fused_4x8   },
    {  4,  16, fused_4x16  },
    {  8,   8, fused_8x8   },
    {  4,  32, fused_4x32  },
    {  8,  16, fused_8x16  },
    {  4,  64, fused_4x64  },
    {  8,  32, fused_8x32  },
    { 16,  16, fused_16x16 },
    {  8,  64, fused_8x64  },
    { 16,  32, fused_16x32 },
    { 16,  64, fused_16x64 },
    { 32,  32, fused_32x32 },
    { 32,  64, fused_32x64 },
    { 64,  64, fused_64x64 },
    {  0,   0, NULL }
};

/* ================================================================
 * Noinline wrappers: direct calls that ICX CANNOT inline.
 * Tests whether inlining helps or hurts for each codelet size.
 * ================================================================ */

#define NOINLINE_N1(name, func) \
__attribute__((noinline)) static void name( \
    const double *ir, const double *ii, double *or_, double *oi, \
    size_t is, size_t os, size_t vl, size_t ovs) { \
    func(ir, ii, or_, oi, is, os, vl, ovs); }

#define NOINLINE_T1(name, func) \
__attribute__((noinline)) static void name( \
    double *rio_re, double *rio_im, \
    const double *W_re, const double *W_im, \
    size_t ios, size_t me) { \
    func(rio_re, rio_im, W_re, W_im, ios, me); }

NOINLINE_N1(r4_n1_ni,  radix4_n1_ovs_fwd_avx2)
NOINLINE_T1(r4_t1_ni,  radix4_t1_dit_fwd_avx2)
NOINLINE_N1(r8_n1_ni,  radix8_n1_ovs_fwd_avx2)
NOINLINE_T1(r8_t1_ni,  radix8_t1_dit_fwd_avx2)
NOINLINE_N1(r16_n1_ni, radix16_n1_ovs_fwd_avx2)
NOINLINE_T1(r16_t1_ni, radix16_t1_dit_fwd_avx2)
NOINLINE_N1(r32_n1_ni, radix32_n1_ovs_fwd_avx2)
NOINLINE_T1(r32_t1_ni, radix32_t1_dit_fwd_avx2)
NOINLINE_N1(r64_n1_ni, radix64_n1_ovs_fwd_avx2)
NOINLINE_T1(r64_t1_ni, radix64_t1_dit_fwd_avx2)

/* Fused noinline: direct calls but bodies stay separate */
/* N=16 */
FUSED_1LEVEL(fni_4x4,   r4_n1_ni,  r4_t1_ni,   4,  4)
/* N=32 */
FUSED_1LEVEL(fni_4x8,   r8_n1_ni,  r4_t1_ni,   4,  8)
/* N=64 */
FUSED_1LEVEL(fni_4x16,  r16_n1_ni, r4_t1_ni,   4, 16)
FUSED_1LEVEL(fni_8x8,   r8_n1_ni,  r8_t1_ni,   8,  8)
/* N=128 */
FUSED_1LEVEL(fni_4x32,  r32_n1_ni, r4_t1_ni,   4, 32)
FUSED_1LEVEL(fni_8x16,  r16_n1_ni, r8_t1_ni,   8, 16)
/* N=256 */
FUSED_1LEVEL(fni_4x64,  r64_n1_ni, r4_t1_ni,   4, 64)
FUSED_1LEVEL(fni_8x32,  r32_n1_ni, r8_t1_ni,   8, 32)
FUSED_1LEVEL(fni_16x16, r16_n1_ni, r16_t1_ni, 16, 16)
/* N=512 */
FUSED_1LEVEL(fni_8x64,  r64_n1_ni, r8_t1_ni,   8, 64)
FUSED_1LEVEL(fni_16x32, r32_n1_ni, r16_t1_ni, 16, 32)
/* N=1024 */
FUSED_1LEVEL(fni_16x64, r64_n1_ni, r16_t1_ni, 16, 64)
FUSED_1LEVEL(fni_32x32, r32_n1_ni, r32_t1_ni, 32, 32)
/* N=2048 */
FUSED_1LEVEL(fni_32x64, r64_n1_ni, r32_t1_ni, 32, 64)
/* N=4096 */
FUSED_1LEVEL(fni_64x64, r64_n1_ni, r64_t1_ni, 64, 64)

static const fused_entry FUSED_NOINL[] = {
    {  4,   4, fni_4x4   },
    {  4,   8, fni_4x8   },
    {  4,  16, fni_4x16  },
    {  8,   8, fni_8x8   },
    {  4,  32, fni_4x32  },
    {  8,  16, fni_8x16  },
    {  4,  64, fni_4x64  },
    {  8,  32, fni_8x32  },
    { 16,  16, fni_16x16 },
    {  8,  64, fni_8x64  },
    { 16,  32, fni_16x32 },
    { 16,  64, fni_16x64 },
    { 32,  32, fni_32x32 },
    { 32,  64, fni_32x64 },
    { 64,  64, fni_64x64 },
    {  0,   0, NULL }
};

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
 * 2-level CT: outer n1_ovs + R1 x inner ct_1level + outer t1
 * ================================================================ */

static void ct_2level(
    t1_fn t1_outer, size_t R1,
    n1_ovs_fn n1_inner, t1_fn t1_inner, size_t R0, size_t M0,
    const double *ir, const double *ii,
    double *or_, double *oi,
    double *tmp_re, double *tmp_im,
    const double *W0_re, const double *W0_im,
    const double *W1_re, const double *W1_im)
{
    size_t M1 = R0 * M0;
    /* Step 1: Gather — rearrange input into R1 contiguous M1-blocks.
     * Sub-sequence r reads elements r, r+R1, r+2*R1, ..., r+(M1-1)*R1 */
    for (size_t r = 0; r < R1; r++)
        for (size_t m = 0; m < M1; m++) {
            tmp_re[r * M1 + m] = ir[r + m * R1];
            tmp_im[r * M1 + m] = ii[r + m * R1];
        }
    /* Step 2: Inner 1-level CT on each M1-block */
    for (size_t r = 0; r < R1; r++)
        ct_1level(n1_inner, t1_inner,
                  tmp_re + r*M1, tmp_im + r*M1,
                  or_ + r*M1, oi + r*M1,
                  W0_re, W0_im, R0, M0);
    /* Step 3: Outer t1 combines R1 blocks */
    t1_outer(or_, oi, W1_re, W1_im, M1, M1);
}

/* ================================================================
 * Test one 1-level factorization: correctness + bench
 * Returns ns, or -1 on failure
 * ================================================================ */

static double test_one(size_t N, size_t R, size_t M,
                       n1_ovs_fn n1, t1_fn t1,
                       const double *in_re, const double *in_im,
                       const double *fftw_re, const double *fftw_im,
                       int reps)
{
    double *W_re = (double*)aligned_alloc(32, (R-1)*M*8);
    double *W_im = (double*)aligned_alloc(32, (R-1)*M*8);
    double *out_re = (double*)aligned_alloc(32, N*8);
    double *out_im = (double*)aligned_alloc(32, N*8);
    init_tw(W_re, W_im, R, M);

    /* Correctness */
    ct_1level(n1, t1, in_re, in_im, out_re, out_im, W_re, W_im, R, M);
    double max_err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fabs(out_re[i] - fftw_re[i]) + fabs(out_im[i] - fftw_im[i]);
        if (e > max_err) max_err = e;
    }

    double ns = -1;
    if (max_err < 1e-10) {
        /* Bench */
        for (int i = 0; i < 20; i++)
            ct_1level(n1, t1, in_re, in_im, out_re, out_im, W_re, W_im, R, M);
        double best = 1e18;
        for (int t = 0; t < 7; t++) {
            double t0 = now_ns();
            for (int i = 0; i < reps; i++)
                ct_1level(n1, t1, in_re, in_im, out_re, out_im, W_re, W_im, R, M);
            double elapsed = (now_ns() - t0) / reps;
            if (elapsed < best) best = elapsed;
        }
        ns = best;
    }

    aligned_free(W_re); aligned_free(W_im);
    aligned_free(out_re); aligned_free(out_im);

    if (max_err < 1e-10)
        printf("    %2zux%-3zu  %8.0f ns  %.2e", R, M, ns, max_err);
    else
        printf("    %2zux%-3zu  %8s     %.2e FAIL", R, M, "--", max_err);

    return ns;
}

/* ================================================================
 * Test one 2-level factorization: correctness + bench
 * ================================================================ */

static double test_two(size_t N, size_t R1, size_t R0, size_t M0,
                       t1_fn t1_outer,
                       n1_ovs_fn n1_inner, t1_fn t1_inner,
                       const double *in_re, const double *in_im,
                       const double *fftw_re, const double *fftw_im,
                       int reps)
{
    size_t M1 = R0 * M0;
    double *W0_re = (double*)aligned_alloc(32, (R0-1)*M0*8);
    double *W0_im = (double*)aligned_alloc(32, (R0-1)*M0*8);
    double *W1_re = (double*)aligned_alloc(32, (R1-1)*M1*8);
    double *W1_im = (double*)aligned_alloc(32, (R1-1)*M1*8);
    double *tmp_re = (double*)aligned_alloc(32, N*8);
    double *tmp_im = (double*)aligned_alloc(32, N*8);
    double *out_re = (double*)aligned_alloc(32, N*8);
    double *out_im = (double*)aligned_alloc(32, N*8);
    init_tw(W0_re, W0_im, R0, M0);
    init_tw(W1_re, W1_im, R1, M1);

    /* Correctness */
    ct_2level(t1_outer, R1,
              n1_inner, t1_inner, R0, M0,
              in_re, in_im, out_re, out_im, tmp_re, tmp_im,
              W0_re, W0_im, W1_re, W1_im);
    double max_err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fabs(out_re[i] - fftw_re[i]) + fabs(out_im[i] - fftw_im[i]);
        if (e > max_err) max_err = e;
    }

    double ns = -1;
    if (max_err < 1e-10) {
        for (int i = 0; i < 20; i++)
            ct_2level(t1_outer, R1,
                      n1_inner, t1_inner, R0, M0,
                      in_re, in_im, out_re, out_im, tmp_re, tmp_im,
                      W0_re, W0_im, W1_re, W1_im);
        double best = 1e18;
        for (int t = 0; t < 7; t++) {
            double t0 = now_ns();
            for (int i = 0; i < reps; i++)
                ct_2level(t1_outer, R1,
                          n1_inner, t1_inner, R0, M0,
                          in_re, in_im, out_re, out_im, tmp_re, tmp_im,
                          W0_re, W0_im, W1_re, W1_im);
            double elapsed = (now_ns() - t0) / reps;
            if (elapsed < best) best = elapsed;
        }
        ns = best;
    }

    aligned_free(W0_re); aligned_free(W0_im);
    aligned_free(W1_re); aligned_free(W1_im);
    aligned_free(tmp_re); aligned_free(tmp_im);
    aligned_free(out_re); aligned_free(out_im);

    if (max_err < 1e-10)
        printf("    %2zux(%zux%-2zu) %6.0f ns  %.2e", R1, R0, M0, ns, max_err);
    else
        printf("    %2zux(%zux%-2zu) %6s     %.2e FAIL", R1, R0, M0, "--", max_err);

    return ns;
}

/* ================================================================
 * Test one fused 1-level: direct calls, no function pointers
 * ================================================================ */

static double test_fused(size_t N, size_t R, size_t M,
                         fused_fn fn, const char *label,
                         const double *in_re, const double *in_im,
                         const double *fftw_re, const double *fftw_im,
                         int reps)
{
    double *W_re = (double*)aligned_alloc(32, (R-1)*M*8);
    double *W_im = (double*)aligned_alloc(32, (R-1)*M*8);
    double *out_re = (double*)aligned_alloc(32, N*8);
    double *out_im = (double*)aligned_alloc(32, N*8);
    init_tw(W_re, W_im, R, M);

    /* Correctness */
    fn(in_re, in_im, out_re, out_im, W_re, W_im);
    double max_err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fabs(out_re[i] - fftw_re[i]) + fabs(out_im[i] - fftw_im[i]);
        if (e > max_err) max_err = e;
    }

    double ns = -1;
    if (max_err < 1e-10) {
        for (int i = 0; i < 20; i++)
            fn(in_re, in_im, out_re, out_im, W_re, W_im);
        double best = 1e18;
        for (int t = 0; t < 7; t++) {
            double t0 = now_ns();
            for (int i = 0; i < reps; i++)
                fn(in_re, in_im, out_re, out_im, W_re, W_im);
            double elapsed = (now_ns() - t0) / reps;
            if (elapsed < best) best = elapsed;
        }
        ns = best;
    }

    aligned_free(W_re); aligned_free(W_im);
    aligned_free(out_re); aligned_free(out_im);

    if (max_err < 1e-10)
        printf("    %2zux%-3zu %s %5.0f ns  %.2e", R, M, label, ns, max_err);
    else
        printf("    %2zux%-3zu %s %5s     %.2e FAIL", R, M, label, "--", max_err);

    return ns;
}

/* ================================================================
 * Test all 1-level and 2-level factorizations for one N
 * ================================================================ */

static void test_N(size_t N) {
    int reps = (int)(2e6 / (N+1));
    if (reps < 200) reps = 200;
    if (reps > 200000) reps = 200000;

    double *in_re = (double*)aligned_alloc(32, N*8);
    double *in_im = (double*)aligned_alloc(32, N*8);
    srand(42);
    for (size_t i = 0; i < N; i++) {
        in_re[i] = (double)rand()/RAND_MAX - 0.5;
        in_im[i] = (double)rand()/RAND_MAX - 0.5;
    }

    /* FFTW reference */
    double *fre = fftw_malloc(N*8), *fim = fftw_malloc(N*8);
    double *fro = fftw_malloc(N*8), *fio = fftw_malloc(N*8);
    memcpy(fre, in_re, N*8); memcpy(fim, in_im, N*8);
    fftw_iodim d = {.n=(int)N, .is=1, .os=1};
    fftw_iodim h = {.n=1, .is=(int)N, .os=(int)N};
    fftw_plan fp = fftw_plan_guru_split_dft(1, &d, 1, &h, fre, fim, fro, fio, FFTW_ESTIMATE);
    fftw_execute(fp);

    double fftw_ns = bench_fftw(N, reps);

    printf("\n  N=%-6zu  FFTW=%.0f ns  (reps=%d)\n", N, fftw_ns, reps);
    printf("    %-7s %8s     %s\n", "RxM", "ns", "err");

    double best_ns = 1e18;
    size_t best_R = 0, best_M = 0;

    /* Try all R*M = N where both R and M have codelets, R <= M */
    for (const codelet_t *cr = CODELETS; cr->R; cr++) {
        size_t R = cr->R;
        if (N % R != 0) continue;
        size_t M = N / R;
        if (M < 4) continue;  /* SIMD needs vl >= 4 */
        if (R > M) continue;  /* skip R>M for now */

        const codelet_t *cm = find(M);  /* child n1_ovs */
        if (!cm) continue;

        double ns = test_one(N, R, M, cm->n1, cr->t1,
                             in_re, in_im, fro, fio, reps);
        if (ns > 0 && ns < best_ns) {
            printf("  <-- best");
            best_ns = ns; best_R = R; best_M = M;
        }
        printf("\n");
    }

    /* 2-level: N = R1 * R0 * M0, all three must have codelets */
    for (const codelet_t *c1 = CODELETS; c1->R; c1++) {
        size_t R1 = c1->R;
        if (N % R1 != 0) continue;
        size_t M1 = N / R1;

        for (const codelet_t *c0 = CODELETS; c0->R; c0++) {
            size_t R0 = c0->R;
            if (M1 % R0 != 0) continue;
            size_t M0 = M1 / R0;
            if (M0 < 4) continue;       /* SIMD minimum */
            if (R0 > M0) continue;       /* skip R0>M0 */

            const codelet_t *cm = find(M0);  /* inner child n1_ovs */
            if (!cm) continue;

            /* outer t1 = R1's t1 */
            /* inner n1_ovs = M0's n1, inner t1 = R0's t1 */
            double ns = test_two(N, R1, R0, M0,
                                 c1->t1,
                                 cm->n1, c0->t1,
                                 in_re, in_im, fro, fio, reps);
            if (ns > 0 && ns < best_ns) {
                printf("  <-- best");
                best_ns = ns; best_R = R1; best_M = M1;
            }
            printf("\n");
        }
    }

    /* Fused 1-level: direct calls (compiler inlines n1_ovs + t1) */
    for (const fused_entry *f = FUSED; f->R; f++) {
        if (f->R * f->M != N) continue;
        double ns = test_fused(N, f->R, f->M, f->fn, " F",
                               in_re, in_im, fro, fio, reps);
        if (ns > 0 && ns < best_ns) {
            printf("  <-- best");
            best_ns = ns;
        }
        printf("\n");
    }

    /* Fused noinline: direct calls, compiler CANNOT inline bodies */
    for (const fused_entry *f = FUSED_NOINL; f->R; f++) {
        if (f->R * f->M != N) continue;
        double ns = test_fused(N, f->R, f->M, f->fn, "NI",
                               in_re, in_im, fro, fio, reps);
        if (ns > 0 && ns < best_ns) {
            printf("  <-- best");
            best_ns = ns;
        }
        printf("\n");
    }

    if (best_ns < 1e18)
        printf("    >> BEST: %.0f ns  (%.2fx vs FFTW)\n", best_ns, fftw_ns / best_ns);

    fftw_destroy_plan(fp);
    fftw_free(fre); fftw_free(fim); fftw_free(fro); fftw_free(fio);
    aligned_free(in_re); aligned_free(in_im);
}

/* ================================================================ */

int main(void) {
    printf("================================================================\n");
    printf("  Permutation-free CT: factorization search\n");
    printf("  Indirect vs Fused-inline (F) vs Fused-noinline (NI)\n");
    printf("  Radixes: 4, 8, 16, 32, 64\n");
    printf("================================================================\n");
    fflush(stdout);

    size_t sizes[] = {
        16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
        8192, 16384, 32768, 65536,
        0
    };

    for (size_t *p = sizes; *p; p++) {
        test_N(*p);
        fflush(stdout);
    }

    printf("\n");
    return 0;
}
