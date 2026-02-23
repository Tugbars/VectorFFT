/**
 * @file test_radix8_n1.c
 * @brief Tests for twiddle-less (N1) radix-8 AVX2 stage drivers
 *
 * Test strategy:
 * 1. N1 forward vs scalar reference (all-unity twiddles) → correctness
 * 2. N1 backward vs scalar reference (all-unity twiddles) → correctness
 * 3. N1 forward vs BLOCKED4 forward with unity twiddles → cross-validation
 * 4. N1 backward vs BLOCKED4 backward with unity twiddles → cross-validation
 * 5. K sweep from 8..2048 → robustness across sizes
 *
 * Compile: gcc -O2 -mavx2 -mfma -o test_n1 test_radix8_n1.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "fft_radix8_avx2_n1.h"

/*============================================================================
 * Utilities (same as twiddled test)
 *============================================================================*/

/*============================================================================
 * PORTABLE ALIGNED ALLOCATION
 *============================================================================*/
#ifdef _MSC_VER
#include <malloc.h>
static double *alloc_aligned(size_t n)
{
    double *p = (double *)_aligned_malloc(n * sizeof(double), 32);
    if (!p) { fprintf(stderr, "alloc_aligned failed for %zu doubles\n", n); exit(1); }
    memset(p, 0, n * sizeof(double));
    return p;
}
#define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
static double *alloc_aligned(size_t n)
{
    double *p = NULL;
    if (posix_memalign((void **)&p, 32, n * sizeof(double)) != 0) {
        fprintf(stderr, "alloc_aligned failed for %zu doubles\n", n);
        exit(1);
    }
    memset(p, 0, n * sizeof(double));
    return p;
}
#define ALIGNED_FREE(ptr) free(ptr)
#endif

static void fill_random(double *buf, size_t n, unsigned seed)
{
    /* Simple xorshift32 for reproducibility */
    unsigned s = seed ? seed : 1;
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        buf[i] = ((double)(int)s) / 2147483648.0;
    }
}

static double max_abs_error(const double *a, const double *b, size_t n)
{
    double mx = 0.0;
    for (size_t i = 0; i < n; i++) {
        double e = fabs(a[i] - b[i]);
        if (e > mx) mx = e;
    }
    return mx;
}

/*============================================================================
 * Scalar reference: radix-8 stage with unity twiddles (N1)
 *
 * out[m*K + k] = sum_{j=0}^{7} in[j*K + k] * W_8^{j*m}
 *
 * No stage twiddles: W_N^(j*k) = 1 for all j,k (first/last stage).
 *============================================================================*/

static void ref_radix8_n1_stage(
    double *out_re, double *out_im,
    const double *in_re, const double *in_im,
    int K, int direction)
{
    const double sign = (double)direction; /* -1 forward, +1 backward */
    const int N = 8 * K;

    for (int k = 0; k < K; k++) {
        /* Gather 8 inputs at stride K */
        double xr[8], xi[8];
        for (int j = 0; j < 8; j++) {
            xr[j] = in_re[j * K + k];
            xi[j] = in_im[j * K + k];
        }

        /* 8-point DFT: out[m] = sum_j x[j] * W_8^(j*m) */
        for (int m = 0; m < 8; m++) {
            double sumr = 0.0, sumi = 0.0;
            for (int j = 0; j < 8; j++) {
                double angle = sign * 2.0 * M_PI * (double)(j * m) / 8.0;
                double wr = cos(angle), wi = sin(angle);
                sumr += xr[j] * wr - xi[j] * wi;
                sumi += xr[j] * wi + xi[j] * wr;
            }
            out_re[m * K + k] = sumr;
            out_im[m * K + k] = sumi;
        }
    }
    (void)N;
}

/*============================================================================
 * TEST: N1 vs scalar reference
 *============================================================================*/

static int test_n1_vs_ref(int K, int direction)
{
    const int N = 8 * K;
    double *in_re  = alloc_aligned(N);
    double *in_im  = alloc_aligned(N);
    double *n1_re  = alloc_aligned(N);
    double *n1_im  = alloc_aligned(N);
    double *ref_re = alloc_aligned(N);
    double *ref_im = alloc_aligned(N);

    fill_random(in_re, N, 123 + K + direction);
    fill_random(in_im, N, 456 + K + direction);

    /* N1 AVX2 */
    if (direction == -1)
        radix8_stage_n1_forward_avx2((size_t)K, in_re, in_im, n1_re, n1_im);
    else
        radix8_stage_n1_backward_avx2((size_t)K, in_re, in_im, n1_re, n1_im);

    /* Scalar reference */
    ref_radix8_n1_stage(ref_re, ref_im, in_re, in_im, K, direction);

    double err = fmax(max_abs_error(n1_re, ref_re, N),
                      max_abs_error(n1_im, ref_im, N));
    double tol = 1e-10 * (1 + log2((double)N));
    int pass = (err < tol);

    printf("  N1 %-3s  K=%-5d N=%-6d max_err=%.2e tol=%.2e %s\n",
           direction == -1 ? "fwd" : "bwd", K, N, err, tol,
           pass ? "PASS" : "*** FAIL ***");

    if (!pass) {
        /* Print first mismatches */
        int shown = 0;
        for (int i = 0; i < N && shown < 4; i++) {
            double er = fabs(n1_re[i] - ref_re[i]);
            double ei = fabs(n1_im[i] - ref_im[i]);
            double e = fmax(er, ei);
            if (e > tol) {
                int m = i / K, kk = i % K;
                printf("    [m=%d,k=%d] n1=(%.8f,%.8f) ref=(%.8f,%.8f) err=%.2e\n",
                       m, kk, n1_re[i], n1_im[i], ref_re[i], ref_im[i], e);
                shown++;
            }
        }
    }

    ALIGNED_FREE(in_re); ALIGNED_FREE(in_im); ALIGNED_FREE(n1_re); ALIGNED_FREE(n1_im);
    ALIGNED_FREE(ref_re); ALIGNED_FREE(ref_im);
    return pass;
}

/*============================================================================
 * TEST: N1 vs BLOCKED4 with unity twiddles (cross-validation)
 *
 * If we fill the BLOCKED4 twiddle table with all 1+0i, the twiddled
 * stage should produce the same result as N1.
 *============================================================================*/

static int test_n1_vs_blocked4_unity(int K, int direction)
{
    const int N = 8 * K;
    double *in_re  = alloc_aligned(N);
    double *in_im  = alloc_aligned(N);
    double *n1_re  = alloc_aligned(N);
    double *n1_im  = alloc_aligned(N);
    double *b4_re  = alloc_aligned(N);
    double *b4_im  = alloc_aligned(N);
    double *tw_re  = alloc_aligned(4 * K);
    double *tw_im  = alloc_aligned(4 * K);

    fill_random(in_re, N, 789 + K + direction);
    fill_random(in_im, N, 321 + K + direction);

    /* Unity twiddles: W_N^(j*k) = 1+0i for all j,k */
    for (int i = 0; i < 4 * K; i++) {
        tw_re[i] = 1.0;
        tw_im[i] = 0.0;
    }

    radix8_stage_twiddles_blocked4_t tw = { .re = tw_re, .im = tw_im };

    /* N1 */
    if (direction == -1)
        radix8_stage_n1_forward_avx2((size_t)K, in_re, in_im, n1_re, n1_im);
    else
        radix8_stage_n1_backward_avx2((size_t)K, in_re, in_im, n1_re, n1_im);

    /* BLOCKED4 with unity twiddles */
    if (direction == -1)
        radix8_stage_blocked4_forward_avx2((size_t)K, in_re, in_im, b4_re, b4_im, &tw);
    else
        radix8_stage_blocked4_backward_avx2((size_t)K, in_re, in_im, b4_re, b4_im, &tw);

    double err = fmax(max_abs_error(n1_re, b4_re, N),
                      max_abs_error(n1_im, b4_im, N));
    /* Expect bit-exact or near (only difference is eliminated twiddle muls by 1+0i) */
    double tol = 1e-14;
    int pass = (err < tol);

    printf("  N1vsB4 %-3s K=%-5d N=%-6d max_err=%.2e tol=%.2e %s\n",
           direction == -1 ? "fwd" : "bwd", K, N, err, tol,
           pass ? "PASS" : "*** FAIL ***");

    if (!pass) {
        int shown = 0;
        for (int i = 0; i < N && shown < 4; i++) {
            double er = fabs(n1_re[i] - b4_re[i]);
            double ei = fabs(n1_im[i] - b4_im[i]);
            double e = fmax(er, ei);
            if (e > tol) {
                int m = i / K, kk = i % K;
                printf("    [m=%d,k=%d] n1=(%.14f,%.14f) b4=(%.14f,%.14f) err=%.2e\n",
                       m, kk, n1_re[i], n1_im[i], b4_re[i], b4_im[i], e);
                shown++;
            }
        }
    }

    ALIGNED_FREE(in_re); ALIGNED_FREE(in_im); ALIGNED_FREE(n1_re); ALIGNED_FREE(n1_im);
    ALIGNED_FREE(b4_re); ALIGNED_FREE(b4_im); ALIGNED_FREE(tw_re); ALIGNED_FREE(tw_im);
    return pass;
}

/*============================================================================
 * MAIN
 *============================================================================*/

int main(void)
{
    printf("============================================\n");
    printf("  Radix-8 AVX2 N1 (Twiddle-less) Tests\n");
    printf("============================================\n");

    int passed = 0, total = 0;

    /* N1 Forward vs Reference */
    printf("\n--- N1 Forward vs Scalar Reference ---\n");
    int Ks[] = {8, 12, 16, 32, 64, 128, 256, 512, 1024};
    int nK = (int)(sizeof(Ks) / sizeof(Ks[0]));
    for (int i = 0; i < nK; i++) {
        total++;
        passed += test_n1_vs_ref(Ks[i], -1);
    }

    /* N1 Backward vs Reference */
    printf("\n--- N1 Backward vs Scalar Reference ---\n");
    for (int i = 0; i < nK; i++) {
        total++;
        passed += test_n1_vs_ref(Ks[i], +1);
    }

    /* N1 vs BLOCKED4 unity forward */
    printf("\n--- N1 Forward vs BLOCKED4(unity twiddles) ---\n");
    int Ks_b4[] = {8, 16, 32, 64, 128, 256};
    int nK_b4 = (int)(sizeof(Ks_b4) / sizeof(Ks_b4[0]));
    for (int i = 0; i < nK_b4; i++) {
        total++;
        passed += test_n1_vs_blocked4_unity(Ks_b4[i], -1);
    }

    /* N1 vs BLOCKED4 unity backward */
    printf("\n--- N1 Backward vs BLOCKED4(unity twiddles) ---\n");
    for (int i = 0; i < nK_b4; i++) {
        total++;
        passed += test_n1_vs_blocked4_unity(Ks_b4[i], +1);
    }

    printf("\n============================================\n");
    if (passed == total)
        printf("  Results: %d/%d passed  \xe2\x9c\x93 ALL PASS\n", passed, total);
    else
        printf("  Results: %d/%d passed  \xe2\x9c\x97 %d FAILED\n", passed, total, total - passed);
    printf("============================================\n");

    return (passed == total) ? 0 : 1;
}
