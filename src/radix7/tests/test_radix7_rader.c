/**
 * @file  test_radix7_rader.c
 * @brief Correctness tests for Rader radix-7 butterfly (scalar + AVX2)
 *
 * Tests:
 *   1. Single-butterfly (K=1) N1 vs naive DFT7  — forward + backward
 *   2. Multi-K N1 (K=1,4,7,16,31,64,100) — forward roundtrip
 *   3. Twiddled butterfly (K=4,8,16,64) — forward + backward roundtrip
 *   4. AVX2 vs scalar cross-check at all K values
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "fft_r7_platform.h"
#include "scalar/fft_radix7_scalar.h"
#include "avx2/fft_radix7_avx2.h"

#define ALIGN 64
#define MAX_K 256

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, ...)         \
    do                           \
    {                            \
        if (cond)                \
        {                        \
            g_pass++;            \
        }                        \
        else                     \
        {                        \
            g_fail++;            \
            printf("FAIL: ");    \
            printf(__VA_ARGS__); \
            printf("\n");        \
        }                        \
    } while (0)

/* ---- Aligned allocation ---- */
static void *aligned_alloc_safe(size_t align, size_t size)
{
    void *p = r7_aligned_alloc(align, size);
    if (p)
        memset(p, 0, size);
    return p;
}

/* ---- Naive DFT7 ---- */
static void naive_dft7(const double *xr, const double *xi,
                       double *Yr, double *Yi, int sign)
{
    for (int m = 0; m < 7; m++)
    {
        double sr = 0, si = 0;
        for (int j = 0; j < 7; j++)
        {
            double angle = sign * 2.0 * M_PI * j * m / 7.0;
            double wr = cos(angle), wi = sin(angle);
            sr += xr[j] * wr - xi[j] * wi;
            si += xr[j] * wi + xi[j] * wr;
        }
        Yr[m] = sr;
        Yi[m] = si;
    }
}

/* ---- Max error between two arrays ---- */
static double max_err(const double *a, const double *b, int n)
{
    double e = 0;
    for (int i = 0; i < n; i++)
    {
        double d = fabs(a[i] - b[i]);
        if (d > e)
            e = d;
    }
    return e;
}

/* ---- Generate BLOCKED3 twiddles for K elements ---- */
static void gen_twiddles(int K, double *w1r, double *w1i,
                         double *w2r, double *w2i,
                         double *w3r, double *w3i)
{
    for (int k = 0; k < K; k++)
    {
        double base = -2.0 * M_PI * k / (7.0 * K);
        w1r[k] = cos(1.0 * base);
        w1i[k] = sin(1.0 * base);
        w2r[k] = cos(2.0 * base);
        w2i[k] = sin(2.0 * base);
        w3r[k] = cos(3.0 * base);
        w3i[k] = sin(3.0 * base);
    }
}

/* ================================================================== */
/*  Test 1: N1 single butterfly vs naive DFT7                         */
/* ================================================================== */

static void test_n1_vs_naive(void)
{
    printf("--- Test 1: N1 single butterfly vs naive DFT7 ---\n");

    double xr[7], xi[7];
    srand(12345);
    for (int i = 0; i < 7; i++)
    {
        xr[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        xi[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    }

    /* Naive forward */
    double Yr_ref[7], Yi_ref[7];
    naive_dft7(xr, xi, Yr_ref, Yi_ref, -1);

    /* Scalar N1 forward */
    double yr[7], yi[7];
    radix7_rader_fwd_scalar_N1(
        &xr[0], &xi[0], &xr[1], &xi[1], &xr[2], &xi[2],
        &xr[3], &xi[3], &xr[4], &xi[4], &xr[5], &xi[5],
        &xr[6], &xi[6],
        &yr[0], &yi[0], &yr[1], &yi[1], &yr[2], &yi[2],
        &yr[3], &yi[3], &yr[4], &yi[4], &yr[5], &yi[5],
        &yr[6], &yi[6], 1);

    double err_fwd = max_err(yr, Yr_ref, 7);
    double err_fwd_im = max_err(yi, Yi_ref, 7);
    double e_fwd = fmax(err_fwd, err_fwd_im);
    printf("  Scalar N1 fwd: max_err = %.2e\n", e_fwd);
    CHECK(e_fwd < 1e-13, "Scalar N1 fwd error %.2e", e_fwd);

    /* Naive backward */
    double Yr_bwd[7], Yi_bwd[7];
    naive_dft7(xr, xi, Yr_bwd, Yi_bwd, +1);

    /* Scalar N1 backward */
    double yr_b[7], yi_b[7];
    radix7_rader_bwd_scalar_N1(
        &xr[0], &xi[0], &xr[1], &xi[1], &xr[2], &xi[2],
        &xr[3], &xi[3], &xr[4], &xi[4], &xr[5], &xi[5],
        &xr[6], &xi[6],
        &yr_b[0], &yi_b[0], &yr_b[1], &yi_b[1], &yr_b[2], &yi_b[2],
        &yr_b[3], &yi_b[3], &yr_b[4], &yi_b[4], &yr_b[5], &yi_b[5],
        &yr_b[6], &yi_b[6], 1);

    double e_bwd = fmax(max_err(yr_b, Yr_bwd, 7), max_err(yi_b, Yi_bwd, 7));
    printf("  Scalar N1 bwd: max_err = %.2e\n", e_bwd);
    CHECK(e_bwd < 1e-13, "Scalar N1 bwd error %.2e", e_bwd);
}

/* ================================================================== */
/*  Test 2: N1 roundtrip at multiple K values                          */
/* ================================================================== */

static void test_n1_roundtrip(void)
{
    printf("\n--- Test 2: N1 forward→backward roundtrip ---\n");

    int K_vals[] = {1, 4, 7, 16, 31, 64, 100};
    int n_K = sizeof(K_vals) / sizeof(K_vals[0]);

    for (int ki = 0; ki < n_K; ki++)
    {
        int K = K_vals[ki];
        size_t sz = K * sizeof(double);

        double *in_re = aligned_alloc_safe(ALIGN, 7 * sz);
        double *in_im = aligned_alloc_safe(ALIGN, 7 * sz);
        double *mid_re = aligned_alloc_safe(ALIGN, 7 * sz);
        double *mid_im = aligned_alloc_safe(ALIGN, 7 * sz);
        double *out_re = aligned_alloc_safe(ALIGN, 7 * sz);
        double *out_im = aligned_alloc_safe(ALIGN, 7 * sz);

        srand(42 + K);
        for (int i = 0; i < 7 * K; i++)
        {
            in_re[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
            in_im[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        }

        /* Forward N1 */
        radix7_rader_fwd_scalar_N1(
            in_re + 0 * K, in_im + 0 * K, in_re + 1 * K, in_im + 1 * K,
            in_re + 2 * K, in_im + 2 * K, in_re + 3 * K, in_im + 3 * K,
            in_re + 4 * K, in_im + 4 * K, in_re + 5 * K, in_im + 5 * K,
            in_re + 6 * K, in_im + 6 * K,
            mid_re + 0 * K, mid_im + 0 * K, mid_re + 1 * K, mid_im + 1 * K,
            mid_re + 2 * K, mid_im + 2 * K, mid_re + 3 * K, mid_im + 3 * K,
            mid_re + 4 * K, mid_im + 4 * K, mid_re + 5 * K, mid_im + 5 * K,
            mid_re + 6 * K, mid_im + 6 * K, K);

        /* Backward N1 */
        radix7_rader_bwd_scalar_N1(
            mid_re + 0 * K, mid_im + 0 * K, mid_re + 1 * K, mid_im + 1 * K,
            mid_re + 2 * K, mid_im + 2 * K, mid_re + 3 * K, mid_im + 3 * K,
            mid_re + 4 * K, mid_im + 4 * K, mid_re + 5 * K, mid_im + 5 * K,
            mid_re + 6 * K, mid_im + 6 * K,
            out_re + 0 * K, out_im + 0 * K, out_re + 1 * K, out_im + 1 * K,
            out_re + 2 * K, out_im + 2 * K, out_re + 3 * K, out_im + 3 * K,
            out_re + 4 * K, out_im + 4 * K, out_re + 5 * K, out_im + 5 * K,
            out_re + 6 * K, out_im + 6 * K, K);

        /* Scale by 1/7 */
        double e = 0;
        for (int i = 0; i < 7 * K; i++)
        {
            double dr = fabs(out_re[i] / 7.0 - in_re[i]);
            double di = fabs(out_im[i] / 7.0 - in_im[i]);
            if (dr > e)
                e = dr;
            if (di > e)
                e = di;
        }

        printf("  K=%3d: roundtrip err = %.2e\n", K, e);
        CHECK(e < 1e-12, "N1 roundtrip K=%d err=%.2e", K, e);

        r7_aligned_free(in_re);
        r7_aligned_free(in_im);
        r7_aligned_free(mid_re);
        r7_aligned_free(mid_im);
        r7_aligned_free(out_re);
        r7_aligned_free(out_im);
    }
}

/* ================================================================== */
/*  Test 3: Twiddled butterfly roundtrip                               */
/* ================================================================== */

static void test_twiddled_roundtrip(void)
{
    printf("\n--- Test 3: Twiddled butterfly roundtrip ---\n");

    int K_vals[] = {4, 8, 16, 32, 64};
    int n_K = sizeof(K_vals) / sizeof(K_vals[0]);

    for (int ki = 0; ki < n_K; ki++)
    {
        int K = K_vals[ki];
        size_t sz = K * sizeof(double);

        double *in_re = aligned_alloc_safe(ALIGN, 7 * sz);
        double *in_im = aligned_alloc_safe(ALIGN, 7 * sz);
        double *mid_re = aligned_alloc_safe(ALIGN, 7 * sz);
        double *mid_im = aligned_alloc_safe(ALIGN, 7 * sz);
        double *out_re = aligned_alloc_safe(ALIGN, 7 * sz);
        double *out_im = aligned_alloc_safe(ALIGN, 7 * sz);
        double *tw1_re = aligned_alloc_safe(ALIGN, sz);
        double *tw1_im = aligned_alloc_safe(ALIGN, sz);
        double *tw2_re = aligned_alloc_safe(ALIGN, sz);
        double *tw2_im = aligned_alloc_safe(ALIGN, sz);
        double *tw3_re = aligned_alloc_safe(ALIGN, sz);
        double *tw3_im = aligned_alloc_safe(ALIGN, sz);

        gen_twiddles(K, tw1_re, tw1_im, tw2_re, tw2_im, tw3_re, tw3_im);

        srand(777 + K);
        for (int i = 0; i < 7 * K; i++)
        {
            in_re[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
            in_im[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        }

        /* Forward twiddled (scalar) */
        radix7_rader_fwd_scalar_1(
            in_re + 0 * K, in_im + 0 * K, in_re + 1 * K, in_im + 1 * K,
            in_re + 2 * K, in_im + 2 * K, in_re + 3 * K, in_im + 3 * K,
            in_re + 4 * K, in_im + 4 * K, in_re + 5 * K, in_im + 5 * K,
            in_re + 6 * K, in_im + 6 * K,
            mid_re + 0 * K, mid_im + 0 * K, mid_re + 1 * K, mid_im + 1 * K,
            mid_re + 2 * K, mid_im + 2 * K, mid_re + 3 * K, mid_im + 3 * K,
            mid_re + 4 * K, mid_im + 4 * K, mid_re + 5 * K, mid_im + 5 * K,
            mid_re + 6 * K, mid_im + 6 * K,
            tw1_re, tw1_im, tw2_re, tw2_im, tw3_re, tw3_im, K);

        /* Backward twiddled (scalar) */
        radix7_rader_bwd_scalar_1(
            mid_re + 0 * K, mid_im + 0 * K, mid_re + 1 * K, mid_im + 1 * K,
            mid_re + 2 * K, mid_im + 2 * K, mid_re + 3 * K, mid_im + 3 * K,
            mid_re + 4 * K, mid_im + 4 * K, mid_re + 5 * K, mid_im + 5 * K,
            mid_re + 6 * K, mid_im + 6 * K,
            out_re + 0 * K, out_im + 0 * K, out_re + 1 * K, out_im + 1 * K,
            out_re + 2 * K, out_im + 2 * K, out_re + 3 * K, out_im + 3 * K,
            out_re + 4 * K, out_im + 4 * K, out_re + 5 * K, out_im + 5 * K,
            out_re + 6 * K, out_im + 6 * K,
            tw1_re, tw1_im, tw2_re, tw2_im, tw3_re, tw3_im, K);

        double e = 0;
        for (int i = 0; i < 7 * K; i++)
        {
            double dr = fabs(out_re[i] / 7.0 - in_re[i]);
            double di = fabs(out_im[i] / 7.0 - in_im[i]);
            if (dr > e)
                e = dr;
            if (di > e)
                e = di;
        }

        printf("  K=%3d: twiddled roundtrip err = %.2e\n", K, e);
        CHECK(e < 1e-12, "Twiddled roundtrip K=%d err=%.2e", K, e);

        r7_aligned_free(in_re);
        r7_aligned_free(in_im);
        r7_aligned_free(mid_re);
        r7_aligned_free(mid_im);
        r7_aligned_free(out_re);
        r7_aligned_free(out_im);
        r7_aligned_free(tw1_re);
        r7_aligned_free(tw1_im);
        r7_aligned_free(tw2_re);
        r7_aligned_free(tw2_im);
        r7_aligned_free(tw3_re);
        r7_aligned_free(tw3_im);
    }
}

/* ================================================================== */
/*  Test 4: AVX2 vs scalar cross-check                                 */
/* ================================================================== */

static void test_avx2_vs_scalar(void)
{
    printf("\n--- Test 4: AVX2 vs scalar cross-check ---\n");

    int K_vals[] = {4, 7, 8, 15, 16, 31, 32, 64, 100};
    int n_K = sizeof(K_vals) / sizeof(K_vals[0]);

    for (int ki = 0; ki < n_K; ki++)
    {
        int K = K_vals[ki];
        size_t sz = K * sizeof(double);

        double *in_re = aligned_alloc_safe(ALIGN, 7 * sz);
        double *in_im = aligned_alloc_safe(ALIGN, 7 * sz);
        double *s_re = aligned_alloc_safe(ALIGN, 7 * sz);
        double *s_im = aligned_alloc_safe(ALIGN, 7 * sz);
        double *v_re = aligned_alloc_safe(ALIGN, 7 * sz);
        double *v_im = aligned_alloc_safe(ALIGN, 7 * sz);
        double *tw1_re = aligned_alloc_safe(ALIGN, sz);
        double *tw1_im = aligned_alloc_safe(ALIGN, sz);
        double *tw2_re = aligned_alloc_safe(ALIGN, sz);
        double *tw2_im = aligned_alloc_safe(ALIGN, sz);
        double *tw3_re = aligned_alloc_safe(ALIGN, sz);
        double *tw3_im = aligned_alloc_safe(ALIGN, sz);

        gen_twiddles(K, tw1_re, tw1_im, tw2_re, tw2_im, tw3_re, tw3_im);

        srand(999 + K);
        for (int i = 0; i < 7 * K; i++)
        {
            in_re[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
            in_im[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        }

        /* --- Forward: N1 --- */
        radix7_rader_fwd_scalar_N1(
            in_re + 0 * K, in_im + 0 * K, in_re + 1 * K, in_im + 1 * K,
            in_re + 2 * K, in_im + 2 * K, in_re + 3 * K, in_im + 3 * K,
            in_re + 4 * K, in_im + 4 * K, in_re + 5 * K, in_im + 5 * K,
            in_re + 6 * K, in_im + 6 * K,
            s_re + 0 * K, s_im + 0 * K, s_re + 1 * K, s_im + 1 * K,
            s_re + 2 * K, s_im + 2 * K, s_re + 3 * K, s_im + 3 * K,
            s_re + 4 * K, s_im + 4 * K, s_re + 5 * K, s_im + 5 * K,
            s_re + 6 * K, s_im + 6 * K, K);

        radix7_rader_fwd_avx2_N1(
            in_re + 0 * K, in_im + 0 * K, in_re + 1 * K, in_im + 1 * K,
            in_re + 2 * K, in_im + 2 * K, in_re + 3 * K, in_im + 3 * K,
            in_re + 4 * K, in_im + 4 * K, in_re + 5 * K, in_im + 5 * K,
            in_re + 6 * K, in_im + 6 * K,
            v_re + 0 * K, v_im + 0 * K, v_re + 1 * K, v_im + 1 * K,
            v_re + 2 * K, v_im + 2 * K, v_re + 3 * K, v_im + 3 * K,
            v_re + 4 * K, v_im + 4 * K, v_re + 5 * K, v_im + 5 * K,
            v_re + 6 * K, v_im + 6 * K, K);

        double e_n1 = fmax(max_err(s_re, v_re, 7 * K),
                           max_err(s_im, v_im, 7 * K));
        printf("  K=%3d N1 fwd: avx2 vs scalar err = %.2e\n", K, e_n1);
        CHECK(e_n1 < 1e-14, "AVX2 N1 fwd K=%d err=%.2e", K, e_n1);

        /* --- Forward: twiddled --- */
        radix7_rader_fwd_scalar_1(
            in_re + 0 * K, in_im + 0 * K, in_re + 1 * K, in_im + 1 * K,
            in_re + 2 * K, in_im + 2 * K, in_re + 3 * K, in_im + 3 * K,
            in_re + 4 * K, in_im + 4 * K, in_re + 5 * K, in_im + 5 * K,
            in_re + 6 * K, in_im + 6 * K,
            s_re + 0 * K, s_im + 0 * K, s_re + 1 * K, s_im + 1 * K,
            s_re + 2 * K, s_im + 2 * K, s_re + 3 * K, s_im + 3 * K,
            s_re + 4 * K, s_im + 4 * K, s_re + 5 * K, s_im + 5 * K,
            s_re + 6 * K, s_im + 6 * K,
            tw1_re, tw1_im, tw2_re, tw2_im, tw3_re, tw3_im, K);

        radix7_rader_fwd_avx2(
            in_re + 0 * K, in_im + 0 * K, in_re + 1 * K, in_im + 1 * K,
            in_re + 2 * K, in_im + 2 * K, in_re + 3 * K, in_im + 3 * K,
            in_re + 4 * K, in_im + 4 * K, in_re + 5 * K, in_im + 5 * K,
            in_re + 6 * K, in_im + 6 * K,
            v_re + 0 * K, v_im + 0 * K, v_re + 1 * K, v_im + 1 * K,
            v_re + 2 * K, v_im + 2 * K, v_re + 3 * K, v_im + 3 * K,
            v_re + 4 * K, v_im + 4 * K, v_re + 5 * K, v_im + 5 * K,
            v_re + 6 * K, v_im + 6 * K,
            tw1_re, tw1_im, tw2_re, tw2_im, tw3_re, tw3_im, K);

        double e_tw = fmax(max_err(s_re, v_re, 7 * K),
                           max_err(s_im, v_im, 7 * K));
        printf("  K=%3d tw fwd:  avx2 vs scalar err = %.2e\n", K, e_tw);
        CHECK(e_tw < 1e-14, "AVX2 tw fwd K=%d err=%.2e", K, e_tw);

        /* --- Backward: twiddled --- */
        radix7_rader_bwd_scalar_1(
            in_re + 0 * K, in_im + 0 * K, in_re + 1 * K, in_im + 1 * K,
            in_re + 2 * K, in_im + 2 * K, in_re + 3 * K, in_im + 3 * K,
            in_re + 4 * K, in_im + 4 * K, in_re + 5 * K, in_im + 5 * K,
            in_re + 6 * K, in_im + 6 * K,
            s_re + 0 * K, s_im + 0 * K, s_re + 1 * K, s_im + 1 * K,
            s_re + 2 * K, s_im + 2 * K, s_re + 3 * K, s_im + 3 * K,
            s_re + 4 * K, s_im + 4 * K, s_re + 5 * K, s_im + 5 * K,
            s_re + 6 * K, s_im + 6 * K,
            tw1_re, tw1_im, tw2_re, tw2_im, tw3_re, tw3_im, K);

        radix7_rader_bwd_avx2(
            in_re + 0 * K, in_im + 0 * K, in_re + 1 * K, in_im + 1 * K,
            in_re + 2 * K, in_im + 2 * K, in_re + 3 * K, in_im + 3 * K,
            in_re + 4 * K, in_im + 4 * K, in_re + 5 * K, in_im + 5 * K,
            in_re + 6 * K, in_im + 6 * K,
            v_re + 0 * K, v_im + 0 * K, v_re + 1 * K, v_im + 1 * K,
            v_re + 2 * K, v_im + 2 * K, v_re + 3 * K, v_im + 3 * K,
            v_re + 4 * K, v_im + 4 * K, v_re + 5 * K, v_im + 5 * K,
            v_re + 6 * K, v_im + 6 * K,
            tw1_re, tw1_im, tw2_re, tw2_im, tw3_re, tw3_im, K);

        double e_bw = fmax(max_err(s_re, v_re, 7 * K),
                           max_err(s_im, v_im, 7 * K));
        printf("  K=%3d tw bwd:  avx2 vs scalar err = %.2e\n", K, e_bw);
        CHECK(e_bw < 1e-14, "AVX2 tw bwd K=%d err=%.2e", K, e_bw);

        /* --- AVX2 roundtrip --- */
        radix7_rader_fwd_avx2(
            in_re + 0 * K, in_im + 0 * K, in_re + 1 * K, in_im + 1 * K,
            in_re + 2 * K, in_im + 2 * K, in_re + 3 * K, in_im + 3 * K,
            in_re + 4 * K, in_im + 4 * K, in_re + 5 * K, in_im + 5 * K,
            in_re + 6 * K, in_im + 6 * K,
            s_re + 0 * K, s_im + 0 * K, s_re + 1 * K, s_im + 1 * K,
            s_re + 2 * K, s_im + 2 * K, s_re + 3 * K, s_im + 3 * K,
            s_re + 4 * K, s_im + 4 * K, s_re + 5 * K, s_im + 5 * K,
            s_re + 6 * K, s_im + 6 * K,
            tw1_re, tw1_im, tw2_re, tw2_im, tw3_re, tw3_im, K);

        radix7_rader_bwd_avx2(
            s_re + 0 * K, s_im + 0 * K, s_re + 1 * K, s_im + 1 * K,
            s_re + 2 * K, s_im + 2 * K, s_re + 3 * K, s_im + 3 * K,
            s_re + 4 * K, s_im + 4 * K, s_re + 5 * K, s_im + 5 * K,
            s_re + 6 * K, s_im + 6 * K,
            v_re + 0 * K, v_im + 0 * K, v_re + 1 * K, v_im + 1 * K,
            v_re + 2 * K, v_im + 2 * K, v_re + 3 * K, v_im + 3 * K,
            v_re + 4 * K, v_im + 4 * K, v_re + 5 * K, v_im + 5 * K,
            v_re + 6 * K, v_im + 6 * K,
            tw1_re, tw1_im, tw2_re, tw2_im, tw3_re, tw3_im, K);

        double e_rt = 0;
        for (int i = 0; i < 7 * K; i++)
        {
            double dr = fabs(v_re[i] / 7.0 - in_re[i]);
            double di = fabs(v_im[i] / 7.0 - in_im[i]);
            if (dr > e_rt)
                e_rt = dr;
            if (di > e_rt)
                e_rt = di;
        }
        printf("  K=%3d AVX2 roundtrip err = %.2e\n", K, e_rt);
        CHECK(e_rt < 1e-12, "AVX2 roundtrip K=%d err=%.2e", K, e_rt);

        r7_aligned_free(in_re);
        r7_aligned_free(in_im);
        r7_aligned_free(s_re);
        r7_aligned_free(s_im);
        r7_aligned_free(v_re);
        r7_aligned_free(v_im);
        r7_aligned_free(tw1_re);
        r7_aligned_free(tw1_im);
        r7_aligned_free(tw2_re);
        r7_aligned_free(tw2_im);
        r7_aligned_free(tw3_re);
        r7_aligned_free(tw3_im);
    }
}

/* ================================================================== */
/*  Main                                                               */
/* ================================================================== */

int main(void)
{
    printf("=== Rader Radix-7 Correctness Tests ===\n\n");

    test_n1_vs_naive();
    test_n1_roundtrip();
    test_twiddled_roundtrip();
    test_avx2_vs_scalar();

    printf("\n=== Results: %d PASS, %d FAIL ===\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}