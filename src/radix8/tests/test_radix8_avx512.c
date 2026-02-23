/**
 * @file test_radix8_avx512.c
 * @brief Tests for AVX-512 radix-8 stage drivers (BLOCKED4/BLOCKED2/N1)
 * Compile: gcc -O2 -mavx512f -mavx512dq -mfma -o test_512 test_radix8_avx512.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "fft_radix8_avx512_blocked_hybrid_fixed.h"

/*============================================================================*/
static double *alloc64(size_t n)
{
    double *p = NULL;
    if (posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) {
        fprintf(stderr, "alloc failed\n"); exit(1);
    }
    memset(p, 0, n * sizeof(double));
    return p;
}

static void fill_random(double *buf, size_t n, unsigned seed)
{
    unsigned s = seed ? seed : 1;
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        buf[i] = ((double)(int)s) / 2147483648.0;
    }
}

static double max_err(const double *a, const double *b, size_t n)
{
    double mx = 0.0;
    for (size_t i = 0; i < n; i++) {
        double e = fabs(a[i] - b[i]);
        if (e > mx) mx = e;
    }
    return mx;
}

/*============================================================================
 * Scalar reference radix-8 stage
 *============================================================================*/
static void ref_radix8_stage(
    double *out_re, double *out_im,
    const double *in_re, const double *in_im,
    int K, int direction, int use_twiddles)
{
    double sign = (double)direction;
    int N = 8 * K;
    for (int k = 0; k < K; k++) {
        double xr[8], xi[8];
        for (int j = 0; j < 8; j++) {
            xr[j] = in_re[j * K + k];
            xi[j] = in_im[j * K + k];
        }
        /* Apply stage twiddles if requested */
        if (use_twiddles) {
            for (int j = 1; j < 8; j++) {
                double angle = sign * 2.0 * M_PI * (double)(j * k) / (double)N;
                double wr = cos(angle), wi = sin(angle);
                double tr = xr[j]*wr - xi[j]*wi;
                double ti = xr[j]*wi + xi[j]*wr;
                xr[j] = tr; xi[j] = ti;
            }
        }
        /* 8-point DFT */
        for (int m = 0; m < 8; m++) {
            double sumr = 0.0, sumi = 0.0;
            for (int j = 0; j < 8; j++) {
                double angle = sign * 2.0 * M_PI * (double)(j * m) / 8.0;
                double wr = cos(angle), wi = sin(angle);
                sumr += xr[j]*wr - xi[j]*wi;
                sumi += xr[j]*wi + xi[j]*wr;
            }
            out_re[m * K + k] = sumr;
            out_im[m * K + k] = sumi;
        }
    }
}

/*============================================================================
 * Twiddle init (same as AVX2 test)
 *============================================================================*/
static void init_twiddles_blocked4_512(double *tw_re, double *tw_im, int K, int dir)
{
    int N = 8 * K;
    for (int j = 1; j <= 4; j++) {
        for (int k = 0; k < K; k++) {
            double angle = (double)dir * 2.0 * M_PI * (double)(j * k) / (double)N;
            tw_re[(j-1)*K + k] = cos(angle);
            tw_im[(j-1)*K + k] = sin(angle);
        }
    }
}

static void init_twiddles_blocked2_512(double *tw_re, double *tw_im, int K, int dir)
{
    int N = 8 * K;
    for (int j = 1; j <= 2; j++) {
        for (int k = 0; k < K; k++) {
            double angle = (double)dir * 2.0 * M_PI * (double)(j * k) / (double)N;
            tw_re[(j-1)*K + k] = cos(angle);
            tw_im[(j-1)*K + k] = sin(angle);
        }
    }
}

/*============================================================================
 * BLOCKED4 tests
 *============================================================================*/
static int test_blocked4_512(int K, int direction)
{
    int N = 8 * K;
    double *in_re=alloc64(N), *in_im=alloc64(N);
    double *avx_re=alloc64(N), *avx_im=alloc64(N);
    double *ref_re=alloc64(N), *ref_im=alloc64(N);
    double *tw_re=alloc64(4*K), *tw_im=alloc64(4*K);

    fill_random(in_re, N, 100+K+direction);
    fill_random(in_im, N, 200+K+direction);
    init_twiddles_blocked4_512(tw_re, tw_im, K, direction);

    radix8_stage_twiddles_blocked4_512_t tw = { .re = tw_re, .im = tw_im };
    if (direction == -1)
        radix8_stage_blocked4_forward_avx512((size_t)K, in_re, in_im, avx_re, avx_im, &tw);
    else
        radix8_stage_blocked4_backward_avx512((size_t)K, in_re, in_im, avx_re, avx_im, &tw);

    ref_radix8_stage(ref_re, ref_im, in_re, in_im, K, direction, 1);

    double e = fmax(max_err(avx_re, ref_re, N), max_err(avx_im, ref_im, N));
    double tol = 1e-10 * (1 + log2((double)N));
    int pass = (e < tol);
    printf("  B4-512  %-3s K=%-5d N=%-6d err=%.2e tol=%.2e %s\n",
           direction==-1?"fwd":"bwd", K, N, e, tol, pass?"PASS":"*** FAIL ***");

    free(in_re); free(in_im); free(avx_re); free(avx_im);
    free(ref_re); free(ref_im); free(tw_re); free(tw_im);
    return pass;
}

/*============================================================================
 * BLOCKED2 tests
 *============================================================================*/
static int test_blocked2_512(int K, int direction)
{
    int N = 8 * K;
    double *in_re=alloc64(N), *in_im=alloc64(N);
    double *avx_re=alloc64(N), *avx_im=alloc64(N);
    double *ref_re=alloc64(N), *ref_im=alloc64(N);
    double *tw_re=alloc64(2*K), *tw_im=alloc64(2*K);

    fill_random(in_re, N, 300+K+direction);
    fill_random(in_im, N, 400+K+direction);
    init_twiddles_blocked2_512(tw_re, tw_im, K, direction);

    radix8_stage_twiddles_blocked2_512_t tw = { .re = tw_re, .im = tw_im };
    if (direction == -1)
        radix8_stage_blocked2_forward_avx512((size_t)K, in_re, in_im, avx_re, avx_im, &tw);
    else
        radix8_stage_blocked2_backward_avx512((size_t)K, in_re, in_im, avx_re, avx_im, &tw);

    ref_radix8_stage(ref_re, ref_im, in_re, in_im, K, direction, 1);

    double e = fmax(max_err(avx_re, ref_re, N), max_err(avx_im, ref_im, N));
    double tol = 1e-10 * (1 + log2((double)N));
    int pass = (e < tol);
    printf("  B2-512  %-3s K=%-5d N=%-6d err=%.2e tol=%.2e %s\n",
           direction==-1?"fwd":"bwd", K, N, e, tol, pass?"PASS":"*** FAIL ***");

    free(in_re); free(in_im); free(avx_re); free(avx_im);
    free(ref_re); free(ref_im); free(tw_re); free(tw_im);
    return pass;
}

/*============================================================================
 * N1 tests
 *============================================================================*/
static int test_n1_512(int K, int direction)
{
    int N = 8 * K;
    double *in_re=alloc64(N), *in_im=alloc64(N);
    double *avx_re=alloc64(N), *avx_im=alloc64(N);
    double *ref_re=alloc64(N), *ref_im=alloc64(N);

    fill_random(in_re, N, 500+K+direction);
    fill_random(in_im, N, 600+K+direction);

    if (direction == -1)
        radix8_stage_n1_forward_avx512((size_t)K, in_re, in_im, avx_re, avx_im);
    else
        radix8_stage_n1_backward_avx512((size_t)K, in_re, in_im, avx_re, avx_im);

    ref_radix8_stage(ref_re, ref_im, in_re, in_im, K, direction, 0);

    double e = fmax(max_err(avx_re, ref_re, N), max_err(avx_im, ref_im, N));
    double tol = 1e-10 * (1 + log2((double)N));
    int pass = (e < tol);
    printf("  N1-512  %-3s K=%-5d N=%-6d err=%.2e tol=%.2e %s\n",
           direction==-1?"fwd":"bwd", K, N, e, tol, pass?"PASS":"*** FAIL ***");

    free(in_re); free(in_im); free(avx_re); free(avx_im);
    free(ref_re); free(ref_im);
    return pass;
}

/*============================================================================
 * B4 vs B2 cross-validation
 *============================================================================*/
static int test_b4_vs_b2_512(int K, int direction)
{
    int N = 8 * K;
    double *in_re=alloc64(N), *in_im=alloc64(N);
    double *b4_re=alloc64(N), *b4_im=alloc64(N);
    double *b2_re=alloc64(N), *b2_im=alloc64(N);
    double *tw4_re=alloc64(4*K), *tw4_im=alloc64(4*K);
    double *tw2_re=alloc64(2*K), *tw2_im=alloc64(2*K);

    fill_random(in_re, N, 700+K+direction);
    fill_random(in_im, N, 800+K+direction);
    init_twiddles_blocked4_512(tw4_re, tw4_im, K, direction);
    init_twiddles_blocked2_512(tw2_re, tw2_im, K, direction);

    radix8_stage_twiddles_blocked4_512_t tw4 = { .re = tw4_re, .im = tw4_im };
    radix8_stage_twiddles_blocked2_512_t tw2 = { .re = tw2_re, .im = tw2_im };

    if (direction == -1) {
        radix8_stage_blocked4_forward_avx512((size_t)K, in_re, in_im, b4_re, b4_im, &tw4);
        radix8_stage_blocked2_forward_avx512((size_t)K, in_re, in_im, b2_re, b2_im, &tw2);
    } else {
        radix8_stage_blocked4_backward_avx512((size_t)K, in_re, in_im, b4_re, b4_im, &tw4);
        radix8_stage_blocked2_backward_avx512((size_t)K, in_re, in_im, b2_re, b2_im, &tw2);
    }

    double e = fmax(max_err(b4_re, b2_re, N), max_err(b4_im, b2_im, N));
    double tol = 1e-12;
    int pass = (e < tol);
    printf("  B4vsB2  %-3s K=%-5d N=%-6d err=%.2e tol=%.2e %s\n",
           direction==-1?"fwd":"bwd", K, N, e, tol, pass?"PASS":"*** FAIL ***");

    free(in_re); free(in_im); free(b4_re); free(b4_im);
    free(b2_re); free(b2_im);
    free(tw4_re); free(tw4_im); free(tw2_re); free(tw2_im);
    return pass;
}

/*============================================================================
 * N1 vs B4(unity) cross-validation
 *============================================================================*/
static int test_n1_vs_b4_unity_512(int K, int direction)
{
    int N = 8 * K;
    double *in_re=alloc64(N), *in_im=alloc64(N);
    double *n1_re=alloc64(N), *n1_im=alloc64(N);
    double *b4_re=alloc64(N), *b4_im=alloc64(N);
    double *tw_re=alloc64(4*K), *tw_im=alloc64(4*K);

    fill_random(in_re, N, 900+K+direction);
    fill_random(in_im, N, 1000+K+direction);

    for (int i = 0; i < 4*K; i++) { tw_re[i] = 1.0; tw_im[i] = 0.0; }

    radix8_stage_twiddles_blocked4_512_t tw = { .re = tw_re, .im = tw_im };

    if (direction == -1) {
        radix8_stage_n1_forward_avx512((size_t)K, in_re, in_im, n1_re, n1_im);
        radix8_stage_blocked4_forward_avx512((size_t)K, in_re, in_im, b4_re, b4_im, &tw);
    } else {
        radix8_stage_n1_backward_avx512((size_t)K, in_re, in_im, n1_re, n1_im);
        radix8_stage_blocked4_backward_avx512((size_t)K, in_re, in_im, b4_re, b4_im, &tw);
    }

    double e = fmax(max_err(n1_re, b4_re, N), max_err(n1_im, b4_im, N));
    double tol = 1e-14;
    int pass = (e < tol);
    printf("  N1vsB4  %-3s K=%-5d N=%-6d err=%.2e tol=%.2e %s\n",
           direction==-1?"fwd":"bwd", K, N, e, tol, pass?"PASS":"*** FAIL ***");

    free(in_re); free(in_im); free(n1_re); free(n1_im);
    free(b4_re); free(b4_im); free(tw_re); free(tw_im);
    return pass;
}

/*============================================================================*/
int main(void)
{
    printf("============================================\n");
    printf("  Radix-8 AVX-512 Stage Driver Tests\n");
    printf("============================================\n");

    int passed = 0, total = 0;

    /* K values: must be multiple of 8, min 16 */
    int K_b4[] = {16, 32, 64, 128, 256};
    int nB4 = sizeof(K_b4)/sizeof(K_b4[0]);
    int K_b2[] = {512, 1024};
    int nB2 = sizeof(K_b2)/sizeof(K_b2[0]);
    int K_n1[] = {16, 32, 64, 128, 256, 512, 1024};
    int nN1 = sizeof(K_n1)/sizeof(K_n1[0]);

    printf("\n--- BLOCKED4 Forward ---\n");
    for (int i = 0; i < nB4; i++) { total++; passed += test_blocked4_512(K_b4[i], -1); }

    printf("\n--- BLOCKED4 Backward ---\n");
    for (int i = 0; i < nB4; i++) { total++; passed += test_blocked4_512(K_b4[i], +1); }

    printf("\n--- BLOCKED2 Forward ---\n");
    for (int i = 0; i < nB2; i++) { total++; passed += test_blocked2_512(K_b2[i], -1); }

    printf("\n--- BLOCKED2 Backward ---\n");
    for (int i = 0; i < nB2; i++) { total++; passed += test_blocked2_512(K_b2[i], +1); }

    printf("\n--- N1 Forward ---\n");
    for (int i = 0; i < nN1; i++) { total++; passed += test_n1_512(K_n1[i], -1); }

    printf("\n--- N1 Backward ---\n");
    for (int i = 0; i < nN1; i++) { total++; passed += test_n1_512(K_n1[i], +1); }

    printf("\n--- B4 vs B2 Cross-Validation ---\n");
    int K_xv[] = {16, 32, 64, 128, 256, 512};
    int nXV = sizeof(K_xv)/sizeof(K_xv[0]);
    for (int i = 0; i < nXV; i++) { total++; passed += test_b4_vs_b2_512(K_xv[i], -1); }
    for (int i = 0; i < nXV; i++) { total++; passed += test_b4_vs_b2_512(K_xv[i], +1); }

    printf("\n--- N1 vs B4(unity) Cross-Validation ---\n");
    for (int i = 0; i < nB4; i++) { total++; passed += test_n1_vs_b4_unity_512(K_b4[i], -1); }
    for (int i = 0; i < nB4; i++) { total++; passed += test_n1_vs_b4_unity_512(K_b4[i], +1); }

    printf("\n============================================\n");
    if (passed == total)
        printf("  Results: %d/%d passed  \xe2\x9c\x93 ALL PASS\n", passed, total);
    else
        printf("  Results: %d/%d passed  \xe2\x9c\x97 %d FAILED\n", passed, total, total-passed);
    printf("============================================\n");

    return (passed == total) ? 0 : 1;
}
