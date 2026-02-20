/**
 * @file test_radix4_fv.c
 * @brief Test harness for forward radix-4 FFT
 *
 * Tests both n1 (twiddle-less) and twiddle paths against scalar reference.
 */

#include "vfft_compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#include "fft_radix4.h"

/* Declared in fft_radix4_fv.c */
extern void fft_radix4_fv(
    double *restrict out_re, double *restrict out_im,
    const double *restrict in_re, const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw, int K);

extern void fft_radix4_fv_n1(
    double *restrict out_re, double *restrict out_im,
    const double *restrict in_re, const double *restrict in_im,
    int K);

/*==========================================================================*/
/* Scalar reference implementations                                         */
/*==========================================================================*/

static void ref_radix4_fv_n1(int K,
    const double *in_re, const double *in_im,
    double *out_re, double *out_im)
{
    const double *a_re = in_re,       *a_im = in_im;
    const double *b_re = in_re + K,   *b_im = in_im + K;
    const double *c_re = in_re + 2*K, *c_im = in_im + 2*K;
    const double *d_re = in_re + 3*K, *d_im = in_im + 3*K;

    double *y0_re = out_re,       *y0_im = out_im;
    double *y1_re = out_re + K,   *y1_im = out_im + K;
    double *y2_re = out_re + 2*K, *y2_im = out_im + 2*K;
    double *y3_re = out_re + 3*K, *y3_im = out_im + 3*K;

    for (int k = 0; k < K; k++)
    {
        double ar = a_re[k], ai = a_im[k];
        double br = b_re[k], bi = b_im[k];
        double cr = c_re[k], ci = c_im[k];
        double dr = d_re[k], di = d_im[k];

        double sAr = ar + cr, sAi = ai + ci;
        double dAr = ar - cr, dAi = ai - ci;
        double sBr = br + dr, sBi = bi + di;
        double dBr = br - dr, dBi = bi - di;

        /* Forward: rot = (+i)*difBD = (-dBi, +dBr) */
        double rot_r = -dBi, rot_i = dBr;

        y0_re[k] = sAr + sBr; y0_im[k] = sAi + sBi;
        y1_re[k] = dAr - rot_r; y1_im[k] = dAi - rot_i;
        y2_re[k] = sAr - sBr; y2_im[k] = sAi - sBi;
        y3_re[k] = dAr + rot_r; y3_im[k] = dAi + rot_i;
    }
}

static void ref_radix4_fv_tw(int K,
    const double *in_re, const double *in_im,
    const double *tw_re, const double *tw_im,
    double *out_re, double *out_im)
{
    const double *a_re = in_re,       *a_im = in_im;
    const double *b_re = in_re + K,   *b_im = in_im + K;
    const double *c_re = in_re + 2*K, *c_im = in_im + 2*K;
    const double *d_re = in_re + 3*K, *d_im = in_im + 3*K;

    const double *w1r = tw_re,       *w1i = tw_im;
    const double *w2r = tw_re + K,   *w2i = tw_im + K;
    const double *w3r = tw_re + 2*K, *w3i = tw_im + 2*K;

    double *y0_re = out_re,       *y0_im = out_im;
    double *y1_re = out_re + K,   *y1_im = out_im + K;
    double *y2_re = out_re + 2*K, *y2_im = out_im + 2*K;
    double *y3_re = out_re + 3*K, *y3_im = out_im + 3*K;

    for (int k = 0; k < K; k++)
    {
        double ar = a_re[k], ai = a_im[k];
        double br = b_re[k], bi = b_im[k];
        double cr = c_re[k], ci = c_im[k];
        double dr = d_re[k], di = d_im[k];

        /* Twiddle multiply */
        double tBr = br*w1r[k] - bi*w1i[k], tBi = br*w1i[k] + bi*w1r[k];
        double tCr = cr*w2r[k] - ci*w2i[k], tCi = cr*w2i[k] + ci*w2r[k];
        double tDr = dr*w3r[k] - di*w3i[k], tDi = dr*w3i[k] + di*w3r[k];

        double sAr = ar + tCr, sAi = ai + tCi;
        double dAr = ar - tCr, dAi = ai - tCi;
        double sBr = tBr + tDr, sBi = tBi + tDi;
        double dBr = tBr - tDr, dBi = tBi - tDi;

        /* Forward: rot = (+i)*difBD = (-dBi, +dBr) */
        double rot_r = -dBi, rot_i = dBr;

        y0_re[k] = sAr + sBr; y0_im[k] = sAi + sBi;
        y1_re[k] = dAr - rot_r; y1_im[k] = dAi - rot_i;
        y2_re[k] = sAr - sBr; y2_im[k] = sAi - sBi;
        y3_re[k] = dAr + rot_r; y3_im[k] = dAi + rot_i;
    }
}

/*==========================================================================*/
/* Aligned allocation helper                                                */
/*==========================================================================*/

static double *alloc_aligned(size_t n)
{
    double *p = (double *)vfft_aligned_alloc(64, n * sizeof(double));
    if (!p)
    {
        fprintf(stderr, "alloc failed\n");
        exit(1);
    }
    memset(p, 0, n * sizeof(double));
    return p;
}

/*==========================================================================*/
/* Test runner                                                               */
/*==========================================================================*/

static int test_n1(int K)
{
    int N = 4 * K;
    double *in_re  = alloc_aligned((size_t)N);
    double *in_im  = alloc_aligned((size_t)N);
    double *ref_re = alloc_aligned((size_t)N);
    double *ref_im = alloc_aligned((size_t)N);
    double *out_re = alloc_aligned((size_t)N);
    double *out_im = alloc_aligned((size_t)N);

    /* Fill with deterministic pseudo-random */
    for (int i = 0; i < N; i++)
    {
        in_re[i] = sin(0.7 * i + 0.1);
        in_im[i] = cos(0.3 * i + 0.5);
    }

    ref_radix4_fv_n1(K, in_re, in_im, ref_re, ref_im);
    fft_radix4_fv_n1(out_re, out_im, in_re, in_im, K);

    double max_err_re = 0, max_err_im = 0;
    for (int i = 0; i < N; i++)
    {
        double er = fabs(out_re[i] - ref_re[i]);
        double ei = fabs(out_im[i] - ref_im[i]);
        if (er > max_err_re) max_err_re = er;
        if (ei > max_err_im) max_err_im = ei;
    }

    const double tol = 1e-10;
    bool pass_re = max_err_re < tol;
    bool pass_im = max_err_im < tol;

    printf("  n1       K=%4d N=%5d  max_err=%.2e  %s\n",
           K, N, max_err_re, pass_re ? "PASS" : "FAIL");
    printf("  n1_im    K=%4d N=%5d  max_err=%.2e  %s\n",
           K, N, max_err_im, pass_im ? "PASS" : "FAIL");

    vfft_aligned_free(in_re);  vfft_aligned_free(in_im);
    vfft_aligned_free(ref_re); vfft_aligned_free(ref_im);
    vfft_aligned_free(out_re); vfft_aligned_free(out_im);

    return (pass_re && pass_im) ? 0 : 1;
}

static int test_tw(int K)
{
    int N = 4 * K;
    double *in_re  = alloc_aligned((size_t)N);
    double *in_im  = alloc_aligned((size_t)N);
    double *ref_re = alloc_aligned((size_t)N);
    double *ref_im = alloc_aligned((size_t)N);
    double *out_re = alloc_aligned((size_t)N);
    double *out_im = alloc_aligned((size_t)N);
    /* Twiddles: 3*K for W1,W2,W3 in blocked SoA */
    double *tw_re = alloc_aligned((size_t)(3 * K));
    double *tw_im = alloc_aligned((size_t)(3 * K));

    /* Fill data */
    for (int i = 0; i < N; i++)
    {
        in_re[i] = sin(0.7 * i + 0.1);
        in_im[i] = cos(0.3 * i + 0.5);
    }

    /* Generate twiddle factors: W_k^n = exp(-2*pi*i*k/N) for DIF forward */
    for (int k = 0; k < K; k++)
    {
        double angle1 = -2.0 * M_PI * k / N;
        double angle2 = -2.0 * M_PI * 2 * k / N;
        double angle3 = -2.0 * M_PI * 3 * k / N;
        tw_re[k]       = cos(angle1); tw_im[k]       = sin(angle1);  /* W1 */
        tw_re[K + k]   = cos(angle2); tw_im[K + k]   = sin(angle2);  /* W2 */
        tw_re[2*K + k] = cos(angle3); tw_im[2*K + k] = sin(angle3);  /* W3 */
    }

    fft_twiddles_soa tw = { .re = tw_re, .im = tw_im };

    ref_radix4_fv_tw(K, in_re, in_im, tw_re, tw_im, ref_re, ref_im);
    fft_radix4_fv(out_re, out_im, in_re, in_im, &tw, K);

    double max_err_re = 0, max_err_im = 0;
    for (int i = 0; i < N; i++)
    {
        double er = fabs(out_re[i] - ref_re[i]);
        double ei = fabs(out_im[i] - ref_im[i]);
        if (er > max_err_re) max_err_re = er;
        if (ei > max_err_im) max_err_im = ei;
    }

    const double tol = 1e-10;
    bool pass_re = max_err_re < tol;
    bool pass_im = max_err_im < tol;

    printf("  tw       K=%4d N=%5d  max_err=%.2e  %s\n",
           K, N, max_err_re, pass_re ? "PASS" : "FAIL");
    printf("  tw_im    K=%4d N=%5d  max_err=%.2e  %s\n",
           K, N, max_err_im, pass_im ? "PASS" : "FAIL");

    vfft_aligned_free(in_re);  vfft_aligned_free(in_im);
    vfft_aligned_free(ref_re); vfft_aligned_free(ref_im);
    vfft_aligned_free(out_re); vfft_aligned_free(out_im);
    vfft_aligned_free(tw_re);  vfft_aligned_free(tw_im);

    return (pass_re && pass_im) ? 0 : 1;
}

int main(void)
{
    int K_values[] = {1, 2, 3, 4, 5, 7, 8, 12, 16, 17, 32, 64, 128, 256, 1024, 4096};
    int n_tests = sizeof(K_values) / sizeof(K_values[0]);
    int failures = 0;

    printf("=== Radix-4 Forward FFT Tests ===\n\n");

    printf("--- N1 (twiddle-less) ---\n");
    for (int i = 0; i < n_tests; i++)
        failures += test_n1(K_values[i]);

    printf("\n--- Twiddle ---\n");
    for (int i = 0; i < n_tests; i++)
        failures += test_tw(K_values[i]);

    printf("\n%s\n", failures == 0 ? "All tests PASSED" : "SOME TESTS FAILED");
    return failures;
}