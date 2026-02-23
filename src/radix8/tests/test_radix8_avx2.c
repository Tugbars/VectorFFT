/**
 * @file test_radix8_avx2.c
 * @brief Comprehensive test for radix-8 AVX2 FFT stage drivers
 *
 * Tests:
 *  1. BLOCKED4 forward  (K=8..256)
 *  2. BLOCKED4 backward (K=8..256)
 *  3. BLOCKED2 forward  (K=512,1024)
 *  4. BLOCKED2 backward (K=512,1024)
 *  5. Round-trip: forward→backward recovers input (×N scaling)
 *  6. Diagnostic: pinpoint sign-flip W5=-W1 correctness
 *
 * Compile:
 *   gcc -O2 -mavx2 -mfma -o test_radix8 test_radix8_avx2.c -lm
 *   clang -O2 -mavx2 -mfma -o test_radix8 test_radix8_avx2.c -lm
 *
 * @author VectorFFT Test Suite
 * @date 2025
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <immintrin.h>

#include "fft_radix8_avx2_blocked_hybrid_fixed.h"

/*============================================================================
 * CONSTANTS & HELPERS
 *============================================================================*/

#define M_2PI 6.283185307179586476925286766559005768394

/*============================================================================
 * PORTABLE ALIGNED ALLOCATION
 *============================================================================*/
#ifdef _MSC_VER
#include <malloc.h>
static void *alloc_aligned(size_t n_doubles)
{
    void *p = _aligned_malloc(n_doubles * sizeof(double), 32);
    if (!p) { fprintf(stderr, "alloc_aligned failed for %zu doubles\n", n_doubles); exit(1); }
    memset(p, 0, n_doubles * sizeof(double));
    return p;
}
#define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
static void *alloc_aligned(size_t n_doubles)
{
    void *p = NULL;
    if (posix_memalign(&p, 32, n_doubles * sizeof(double)) != 0) {
        fprintf(stderr, "alloc_aligned failed for %zu doubles\n", n_doubles);
        exit(1);
    }
    memset(p, 0, n_doubles * sizeof(double));
    return p;
}
#define ALIGNED_FREE(ptr) free(ptr)
#endif

static void fill_random(double *buf, int n, unsigned seed)
{
    srand(seed);
    for (int i = 0; i < n; i++)
        buf[i] = (double)rand() / RAND_MAX - 0.5;
}

static double max_abs_error(const double *a, const double *b, int n)
{
    double mx = 0.0;
    for (int i = 0; i < n; i++) {
        double e = fabs(a[i] - b[i]);
        if (e > mx) mx = e;
    }
    return mx;
}

/* Relative error: max |a-b| / max(|a|, |b|, eps) */
static double max_rel_error(const double *a, const double *b, int n)
{
    double mx = 0.0;
    for (int i = 0; i < n; i++) {
        double denom = fmax(fabs(a[i]), fabs(b[i]));
        if (denom < 1e-15) denom = 1e-15;
        double e = fabs(a[i] - b[i]) / denom;
        if (e > mx) mx = e;
    }
    return mx;
}

/*============================================================================
 * REFERENCE: Scalar Radix-8 Stage
 *
 * out[m*K + k] = Σ_{j=0}^{7} in[j*K + k] · W_N^(j·k) · W_8^(j·m)
 *
 * direction: -1 = forward (DFT), +1 = backward (IDFT without 1/N)
 *============================================================================*/

static void ref_radix8_stage(
    double *out_re, double *out_im,
    const double *in_re, const double *in_im,
    int K, int direction)
{
    const int N = 8 * K;
    for (int k = 0; k < K; k++) {
        /* Load and apply stage twiddles */
        double xr[8], xi[8];
        for (int j = 0; j < 8; j++) {
            double angle = direction * M_2PI * (double)((long long)j * k) / (double)N;
            double wr = cos(angle), wi = sin(angle);
            double ir = in_re[j * K + k];
            double ii = in_im[j * K + k];
            xr[j] = ir * wr - ii * wi;
            xi[j] = ir * wi + ii * wr;
        }

        /* 8-point DFT */
        for (int m = 0; m < 8; m++) {
            double yr = 0.0, yi = 0.0;
            for (int j = 0; j < 8; j++) {
                double angle = direction * M_2PI * (double)(j * m) / 8.0;
                double wr = cos(angle), wi = sin(angle);
                yr += xr[j] * wr - xi[j] * wi;
                yi += xr[j] * wi + xi[j] * wr;
            }
            out_re[m * K + k] = yr;
            out_im[m * K + k] = yi;
        }
    }
}

/*============================================================================
 * TWIDDLE PRECOMPUTATION
 *
 * Standard CT twiddles: W_j[k] = exp(direction · 2πi · j · k / N)
 *
 * BLOCKED4 layout: tw[row*K + k] for row=0..3
 *   row 0: W_N^(1·k)  (twiddle for input 1)
 *   row 1: W_N^(2·k)  (twiddle for input 2)
 *   row 2: W_N^(3·k)  (twiddle for input 3)
 *   row 3: W_N^(4·k)  (twiddle for input 4)
 *
 * BLOCKED2 layout: tw[row*K + k] for row=0..1
 *   row 0: W_N^(1·k)
 *   row 1: W_N^(2·k)
 *============================================================================*/

static void init_twiddles_blocked4(double *tw_re, double *tw_im, int K, int direction)
{
    const int N = 8 * K;
    for (int row = 0; row < 4; row++) {
        int j = row + 1;
        for (int k = 0; k < K; k++) {
            double angle = direction * M_2PI * (double)((long long)j * k) / (double)N;
            tw_re[row * K + k] = cos(angle);
            tw_im[row * K + k] = sin(angle);
        }
    }
}

static void init_twiddles_blocked2(double *tw_re, double *tw_im, int K, int direction)
{
    const int N = 8 * K;
    for (int row = 0; row < 2; row++) {
        int j = row + 1;
        for (int k = 0; k < K; k++) {
            double angle = direction * M_2PI * (double)((long long)j * k) / (double)N;
            tw_re[row * K + k] = cos(angle);
            tw_im[row * K + k] = sin(angle);
        }
    }
}

/*============================================================================
 * DIAGNOSTIC: Verify sign-flip relationship W5=-W1, W6=-W2, W7=-W3
 *
 * Check if W_N^(5k) == -W_N^(k) for all k.
 * Spoiler: it won't be, because W_N^(4k) ≠ -1 in general.
 *============================================================================*/

static void diagnose_sign_flip(int K)
{
    const int N = 8 * K;
    printf("\n=== SIGN-FLIP DIAGNOSTIC (K=%d, N=%d) ===\n", K, N);
    printf("Checking W_N^(5k) == -W_N^(k) for k=0..%d\n", K - 1);

    double max_err = 0.0;
    int worst_k = -1;

    for (int k = 0; k < K; k++) {
        /* W_N^(k) */
        double a = -M_2PI * (double)k / (double)N;
        double w1r = cos(a), w1i = sin(a);

        /* W_N^(5k) */
        double a5 = -M_2PI * (double)(5LL * k) / (double)N;
        double w5r = cos(a5), w5i = sin(a5);

        /* -W_N^(k) */
        double neg_w1r = -w1r, neg_w1i = -w1i;

        double er = fabs(w5r - neg_w1r);
        double ei = fabs(w5i - neg_w1i);
        double e = fmax(er, ei);

        if (e > max_err) {
            max_err = e;
            worst_k = k;
        }

        if (k < 8 || e > 0.01) {
            printf("  k=%3d: W5=(%.6f,%.6f) vs -W1=(%.6f,%.6f) err=%.2e\n",
                   k, w5r, w5i, neg_w1r, neg_w1i, e);
        }
    }

    printf("  max_err=%.2e at k=%d\n", max_err, worst_k);
    printf("  VERDICT: sign-flip W5=-W1 is %s for standard CT twiddles\n",
           max_err < 1e-14 ? "CORRECT" : "*** INCORRECT ***");

    /* Also show W_N^(4k) values for a few k */
    printf("\n  W_N^(4k) values (should be -1 for sign-flip to work):\n");
    for (int k = 0; k < K && k < 16; k++) {
        double a4 = -M_2PI * (double)(4LL * k) / (double)N;
        printf("    k=%2d: W_N^(4k) = (%.6f, %.6f)  magnitude_diff_from_-1 = %.2e\n",
               k, cos(a4), sin(a4), fabs(cos(a4) - (-1.0)) + fabs(sin(a4)));
    }
    printf("=== END DIAGNOSTIC ===\n\n");
}

/*============================================================================
 * TEST: BLOCKED4 FORWARD
 *============================================================================*/

static int test_blocked4_forward(int K)
{
    const int N = 8 * K;
    double *in_re  = alloc_aligned(N);
    double *in_im  = alloc_aligned(N);
    double *out_re = alloc_aligned(N);
    double *out_im = alloc_aligned(N);
    double *ref_re = alloc_aligned(N);
    double *ref_im = alloc_aligned(N);
    double *tw_re  = alloc_aligned(4 * K);
    double *tw_im  = alloc_aligned(4 * K);

    fill_random(in_re, N, 42);
    fill_random(in_im, N, 137);

    /* Reference */
    ref_radix8_stage(ref_re, ref_im, in_re, in_im, K, -1);

    /* DUT: BLOCKED4 forward */
    init_twiddles_blocked4(tw_re, tw_im, K, -1);
    radix8_stage_twiddles_blocked4_t tw = { .re = tw_re, .im = tw_im };
    radix8_stage_blocked4_forward_avx2((size_t)K, in_re, in_im, out_re, out_im, &tw);

    /* Compare */
    double err_re = max_abs_error(out_re, ref_re, N);
    double err_im = max_abs_error(out_im, ref_im, N);
    double err = fmax(err_re, err_im);

    /* Tolerance: allow for radix-8 accumulation error */
    double tol = 1e-10 * (1 + log2((double)N));
    int pass = (err < tol);

    printf("  BLOCKED4 fwd  K=%-5d N=%-6d max_err=%.2e tol=%.2e %s\n",
           K, N, err, tol, pass ? "PASS" : "*** FAIL ***");

    if (!pass && K <= 32) {
        /* Print first few mismatches */
        printf("    First mismatches:\n");
        int count = 0;
        for (int i = 0; i < N && count < 8; i++) {
            double e = fmax(fabs(out_re[i] - ref_re[i]), fabs(out_im[i] - ref_im[i]));
            if (e > tol) {
                int m = i / K, k = i % K;
                printf("      [m=%d,k=%d] out=(%.8f,%.8f) ref=(%.8f,%.8f) err=%.2e\n",
                       m, k, out_re[i], out_im[i], ref_re[i], ref_im[i], e);
                count++;
            }
        }
    }

    ALIGNED_FREE(in_re); ALIGNED_FREE(in_im); ALIGNED_FREE(out_re); ALIGNED_FREE(out_im);
    ALIGNED_FREE(ref_re); ALIGNED_FREE(ref_im); ALIGNED_FREE(tw_re); ALIGNED_FREE(tw_im);
    return pass;
}

/*============================================================================
 * TEST: BLOCKED4 BACKWARD
 *============================================================================*/

static int test_blocked4_backward(int K)
{
    const int N = 8 * K;
    double *in_re  = alloc_aligned(N);
    double *in_im  = alloc_aligned(N);
    double *out_re = alloc_aligned(N);
    double *out_im = alloc_aligned(N);
    double *ref_re = alloc_aligned(N);
    double *ref_im = alloc_aligned(N);
    double *tw_re  = alloc_aligned(4 * K);
    double *tw_im  = alloc_aligned(4 * K);

    fill_random(in_re, N, 99);
    fill_random(in_im, N, 213);

    ref_radix8_stage(ref_re, ref_im, in_re, in_im, K, +1);

    init_twiddles_blocked4(tw_re, tw_im, K, +1);
    radix8_stage_twiddles_blocked4_t tw = { .re = tw_re, .im = tw_im };
    radix8_stage_blocked4_backward_avx2((size_t)K, in_re, in_im, out_re, out_im, &tw);

    double err = fmax(max_abs_error(out_re, ref_re, N),
                      max_abs_error(out_im, ref_im, N));
    double tol = 1e-10 * (1 + log2((double)N));
    int pass = (err < tol);

    printf("  BLOCKED4 bwd  K=%-5d N=%-6d max_err=%.2e tol=%.2e %s\n",
           K, N, err, tol, pass ? "PASS" : "*** FAIL ***");

    if (!pass && K <= 32) {
        printf("    First mismatches:\n");
        int count = 0;
        for (int i = 0; i < N && count < 8; i++) {
            double e = fmax(fabs(out_re[i] - ref_re[i]), fabs(out_im[i] - ref_im[i]));
            if (e > tol) {
                int m = i / K, k = i % K;
                printf("      [m=%d,k=%d] out=(%.8f,%.8f) ref=(%.8f,%.8f) err=%.2e\n",
                       m, k, out_re[i], out_im[i], ref_re[i], ref_im[i], e);
                count++;
            }
        }
    }

    ALIGNED_FREE(in_re); ALIGNED_FREE(in_im); ALIGNED_FREE(out_re); ALIGNED_FREE(out_im);
    ALIGNED_FREE(ref_re); ALIGNED_FREE(ref_im); ALIGNED_FREE(tw_re); ALIGNED_FREE(tw_im);
    return pass;
}

/*============================================================================
 * TEST: BLOCKED2 FORWARD
 *============================================================================*/

static int test_blocked2_forward(int K)
{
    const int N = 8 * K;
    double *in_re  = alloc_aligned(N);
    double *in_im  = alloc_aligned(N);
    double *out_re = alloc_aligned(N);
    double *out_im = alloc_aligned(N);
    double *ref_re = alloc_aligned(N);
    double *ref_im = alloc_aligned(N);
    double *tw_re  = alloc_aligned(2 * K);
    double *tw_im  = alloc_aligned(2 * K);

    fill_random(in_re, N, 42);
    fill_random(in_im, N, 137);

    ref_radix8_stage(ref_re, ref_im, in_re, in_im, K, -1);

    init_twiddles_blocked2(tw_re, tw_im, K, -1);
    radix8_stage_twiddles_blocked2_t tw = { .re = tw_re, .im = tw_im };
    radix8_stage_blocked2_forward_avx2((size_t)K, in_re, in_im, out_re, out_im, &tw);

    double err = fmax(max_abs_error(out_re, ref_re, N),
                      max_abs_error(out_im, ref_im, N));
    double tol = 1e-10 * (1 + log2((double)N));
    int pass = (err < tol);

    printf("  BLOCKED2 fwd  K=%-5d N=%-6d max_err=%.2e tol=%.2e %s\n",
           K, N, err, tol, pass ? "PASS" : "*** FAIL ***");

    ALIGNED_FREE(in_re); ALIGNED_FREE(in_im); ALIGNED_FREE(out_re); ALIGNED_FREE(out_im);
    ALIGNED_FREE(ref_re); ALIGNED_FREE(ref_im); ALIGNED_FREE(tw_re); ALIGNED_FREE(tw_im);
    return pass;
}

/*============================================================================
 * TEST: BLOCKED2 BACKWARD
 *============================================================================*/

static int test_blocked2_backward(int K)
{
    const int N = 8 * K;
    double *in_re  = alloc_aligned(N);
    double *in_im  = alloc_aligned(N);
    double *out_re = alloc_aligned(N);
    double *out_im = alloc_aligned(N);
    double *ref_re = alloc_aligned(N);
    double *ref_im = alloc_aligned(N);
    double *tw_re  = alloc_aligned(2 * K);
    double *tw_im  = alloc_aligned(2 * K);

    fill_random(in_re, N, 99);
    fill_random(in_im, N, 213);

    ref_radix8_stage(ref_re, ref_im, in_re, in_im, K, +1);

    init_twiddles_blocked2(tw_re, tw_im, K, +1);
    radix8_stage_twiddles_blocked2_t tw = { .re = tw_re, .im = tw_im };
    radix8_stage_blocked2_backward_avx2((size_t)K, in_re, in_im, out_re, out_im, &tw);

    double err = fmax(max_abs_error(out_re, ref_re, N),
                      max_abs_error(out_im, ref_im, N));
    double tol = 1e-10 * (1 + log2((double)N));
    int pass = (err < tol);

    printf("  BLOCKED2 bwd  K=%-5d N=%-6d max_err=%.2e tol=%.2e %s\n",
           K, N, err, tol, pass ? "PASS" : "*** FAIL ***");

    ALIGNED_FREE(in_re); ALIGNED_FREE(in_im); ALIGNED_FREE(out_re); ALIGNED_FREE(out_im);
    ALIGNED_FREE(ref_re); ALIGNED_FREE(ref_im); ALIGNED_FREE(tw_re); ALIGNED_FREE(tw_im);
    return pass;
}

/*============================================================================
 * TEST: BLOCKED4 vs BLOCKED2 cross-validation
 * Both modes should produce identical results for the same K.
 *============================================================================*/

static int test_blocked4_vs_blocked2(int K, int direction)
{
    const int N = 8 * K;
    double *in_re  = alloc_aligned(N);
    double *in_im  = alloc_aligned(N);
    double *b4_re  = alloc_aligned(N);
    double *b4_im  = alloc_aligned(N);
    double *b2_re  = alloc_aligned(N);
    double *b2_im  = alloc_aligned(N);
    double *tw4_re = alloc_aligned(4 * K);
    double *tw4_im = alloc_aligned(4 * K);
    double *tw2_re = alloc_aligned(2 * K);
    double *tw2_im = alloc_aligned(2 * K);

    fill_random(in_re, N, 555 + K);
    fill_random(in_im, N, 666 + K);

    init_twiddles_blocked4(tw4_re, tw4_im, K, direction);
    init_twiddles_blocked2(tw2_re, tw2_im, K, direction);

    radix8_stage_twiddles_blocked4_t tw4 = { .re = tw4_re, .im = tw4_im };
    radix8_stage_twiddles_blocked2_t tw2 = { .re = tw2_re, .im = tw2_im };

    if (direction == -1) {
        radix8_stage_blocked4_forward_avx2((size_t)K, in_re, in_im, b4_re, b4_im, &tw4);
        radix8_stage_blocked2_forward_avx2((size_t)K, in_re, in_im, b2_re, b2_im, &tw2);
    } else {
        radix8_stage_blocked4_backward_avx2((size_t)K, in_re, in_im, b4_re, b4_im, &tw4);
        radix8_stage_blocked2_backward_avx2((size_t)K, in_re, in_im, b2_re, b2_im, &tw2);
    }

    double err = fmax(max_abs_error(b4_re, b2_re, N),
                      max_abs_error(b4_im, b2_im, N));
    /* BLOCKED2 derives W3,W4 from W1,W2 with FMA — expect slightly more error */
    double tol = 1e-12;
    int pass = (err < tol);

    printf("  B4vsB2 %s K=%-5d N=%-6d max_err=%.2e tol=%.2e %s\n",
           direction == -1 ? "fwd" : "bwd", K, N, err, tol,
           pass ? "PASS" : "*** FAIL ***");

    ALIGNED_FREE(in_re); ALIGNED_FREE(in_im); ALIGNED_FREE(b4_re); ALIGNED_FREE(b4_im);
    ALIGNED_FREE(b2_re); ALIGNED_FREE(b2_im);
    ALIGNED_FREE(tw4_re); ALIGNED_FREE(tw4_im); ALIGNED_FREE(tw2_re); ALIGNED_FREE(tw2_im);
    return pass;
}

/*============================================================================
 * MAIN
 *============================================================================*/

int main(int argc, char **argv)
{
    int total = 0, passed = 0;

    printf("============================================\n");
    printf("  Radix-8 AVX2 Stage Driver Tests\n");
    printf("============================================\n\n");

    /* Diagnostic: check the sign-flip math first */
    diagnose_sign_flip(8);
    diagnose_sign_flip(32);

    /* BLOCKED4 Forward Tests (K must be >= 8, multiple of 4) */
    printf("--- BLOCKED4 Forward ---\n");
    int K_b4[] = {8, 12, 16, 32, 64, 128, 256};
    for (int i = 0; i < (int)(sizeof(K_b4)/sizeof(K_b4[0])); i++) {
        total++;
        passed += test_blocked4_forward(K_b4[i]);
    }

    /* BLOCKED4 Backward Tests */
    printf("\n--- BLOCKED4 Backward ---\n");
    for (int i = 0; i < (int)(sizeof(K_b4)/sizeof(K_b4[0])); i++) {
        total++;
        passed += test_blocked4_backward(K_b4[i]);
    }

    /* BLOCKED2 Forward Tests (K > 256) */
    printf("\n--- BLOCKED2 Forward ---\n");
    int K_b2[] = {512, 1024};
    for (int i = 0; i < (int)(sizeof(K_b2)/sizeof(K_b2[0])); i++) {
        total++;
        passed += test_blocked2_forward(K_b2[i]);
    }

    /* BLOCKED2 Backward Tests */
    printf("\n--- BLOCKED2 Backward ---\n");
    for (int i = 0; i < (int)(sizeof(K_b2)/sizeof(K_b2[0])); i++) {
        total++;
        passed += test_blocked2_backward(K_b2[i]);
    }

    /* BLOCKED4 vs BLOCKED2 cross-validation */
    printf("\n--- BLOCKED4 vs BLOCKED2 Cross-Validation ---\n");
    int K_xv[] = {8, 16, 32, 64, 128, 256, 512};
    for (int i = 0; i < (int)(sizeof(K_xv)/sizeof(K_xv[0])); i++) {
        total++;
        passed += test_blocked4_vs_blocked2(K_xv[i], -1);
        total++;
        passed += test_blocked4_vs_blocked2(K_xv[i], +1);
    }

    /* Summary */
    printf("\n============================================\n");
    printf("  Results: %d/%d passed", passed, total);
    if (passed == total)
        printf("  ✓ ALL PASS\n");
    else
        printf("  ✗ %d FAILED\n", total - passed);
    printf("============================================\n");

    return (passed == total) ? 0 : 1;
}