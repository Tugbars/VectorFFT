/**
 * @file test_radix8_drivers.c
 * @brief Integration test for fft_radix8.h public API
 *
 * Tests fft_radix8_fv, fft_radix8_bv, fft_radix8_fv_n1, fft_radix8_bv_n1
 * through the driver dispatch layer.
 *
 * Compile (AVX-512): gcc -O2 -mavx512f -mavx512dq -mfma test_radix8_drivers.c fft_radix8_fv.c fft_radix8_bv.c -lm
 * Compile (AVX2):    gcc -O2 -mavx2 -mfma test_radix8_drivers.c fft_radix8_fv.c fft_radix8_bv.c -lm
 * Compile (scalar):  gcc -O2 -mfma test_radix8_drivers.c fft_radix8_fv.c fft_radix8_bv.c -lm
 * Compile (MSVC):    cl /O2 /arch:AVX2 test_radix8_drivers.c fft_radix8_fv.c fft_radix8_bv.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "fft_radix8.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*============================================================================
 * PORTABLE ALIGNED ALLOCATION
 *============================================================================*/
#ifdef _MSC_VER
#include <malloc.h>
static double *alloc64(size_t n)
{
    double *p = (double *)_aligned_malloc(n * sizeof(double), 64);
    if (!p) { fprintf(stderr, "alloc failed\n"); exit(1); }
    memset(p, 0, n * sizeof(double));
    return p;
}
#define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
static double *alloc64(size_t n)
{
    double *p = NULL;
    if (posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) {
        fprintf(stderr, "alloc failed\n"); exit(1);
    }
    memset(p, 0, n * sizeof(double));
    return p;
}
#define ALIGNED_FREE(ptr) free(ptr)
#endif

/*============================================================================*/
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
 * Scalar reference
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
        if (use_twiddles) {
            for (int j = 1; j < 8; j++) {
                double angle = sign * 2.0 * M_PI * (double)(j * k) / (double)N;
                double wr = cos(angle), wi = sin(angle);
                double tr = xr[j]*wr - xi[j]*wi;
                double ti = xr[j]*wi + xi[j]*wr;
                xr[j] = tr; xi[j] = ti;
            }
        }
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
 * Twiddle init
 *============================================================================*/
static void init_tw4(double *tw_re, double *tw_im, int K, int dir)
{
    int N = 8 * K;
    for (int j = 1; j <= 4; j++)
        for (int k = 0; k < K; k++) {
            double ang = (double)dir * 2.0 * M_PI * (double)(j * k) / (double)N;
            tw_re[(j-1)*K + k] = cos(ang);
            tw_im[(j-1)*K + k] = sin(ang);
        }
}

static void init_tw2(double *tw_re, double *tw_im, int K, int dir)
{
    int N = 8 * K;
    for (int j = 1; j <= 2; j++)
        for (int k = 0; k < K; k++) {
            double ang = (double)dir * 2.0 * M_PI * (double)(j * k) / (double)N;
            tw_re[(j-1)*K + k] = cos(ang);
            tw_im[(j-1)*K + k] = sin(ang);
        }
}

/*============================================================================
 * Test: twiddled forward/backward through public API
 *============================================================================*/
static int test_twiddled(int K, int direction)
{
    int N = 8 * K;
    double *in_re=alloc64(N), *in_im=alloc64(N);
    double *drv_re=alloc64(N), *drv_im=alloc64(N);
    double *ref_re=alloc64(N), *ref_im=alloc64(N);
    double *tw4_re=alloc64(4*K), *tw4_im=alloc64(4*K);
    double *tw2_re=alloc64(2*K), *tw2_im=alloc64(2*K);

    fill_random(in_re, N, 42 + K + direction);
    fill_random(in_im, N, 99 + K + direction);
    init_tw4(tw4_re, tw4_im, K, direction);
    init_tw2(tw2_re, tw2_im, K, direction);

    radix8_stage_twiddles_blocked4_t tw4 = { .re = tw4_re, .im = tw4_im };
    radix8_stage_twiddles_blocked2_t tw2 = { .re = tw2_re, .im = tw2_im };

    if (direction == -1)
        fft_radix8_fv(drv_re, drv_im, in_re, in_im, &tw4, &tw2, K);
    else
        fft_radix8_bv(drv_re, drv_im, in_re, in_im, &tw4, &tw2, K);

    ref_radix8_stage(ref_re, ref_im, in_re, in_im, K, direction, 1);

    double e = fmax(max_err(drv_re, ref_re, N), max_err(drv_im, ref_im, N));
    double tol = 1e-11;
    const char *mode = (K <= RADIX8_BLOCKED4_THRESHOLD) ? "B4" : "B2";
    int pass = (e < tol);
    printf("  %-3s %s %-3s K=%-5d N=%-6d err=%.2e %s\n",
           direction==-1?"fv":"bv", mode,
           direction==-1?"fwd":"bwd", K, N, e, pass?"PASS":"*** FAIL ***");

    ALIGNED_FREE(in_re); ALIGNED_FREE(in_im); ALIGNED_FREE(drv_re); ALIGNED_FREE(drv_im);
    ALIGNED_FREE(ref_re); ALIGNED_FREE(ref_im);
    ALIGNED_FREE(tw4_re); ALIGNED_FREE(tw4_im); ALIGNED_FREE(tw2_re); ALIGNED_FREE(tw2_im);
    return pass;
}

/*============================================================================
 * Test: N1 forward/backward through public API
 *============================================================================*/
static int test_n1(int K, int direction)
{
    int N = 8 * K;
    double *in_re=alloc64(N), *in_im=alloc64(N);
    double *drv_re=alloc64(N), *drv_im=alloc64(N);
    double *ref_re=alloc64(N), *ref_im=alloc64(N);

    fill_random(in_re, N, 500 + K + direction);
    fill_random(in_im, N, 600 + K + direction);

    if (direction == -1)
        fft_radix8_fv_n1(drv_re, drv_im, in_re, in_im, K);
    else
        fft_radix8_bv_n1(drv_re, drv_im, in_re, in_im, K);

    ref_radix8_stage(ref_re, ref_im, in_re, in_im, K, direction, 0);

    double e = fmax(max_err(drv_re, ref_re, N), max_err(drv_im, ref_im, N));
    double tol = 1e-11;
    int pass = (e < tol);
    printf("  %-3s N1 %-3s K=%-5d N=%-6d err=%.2e %s\n",
           direction==-1?"fv":"bv",
           direction==-1?"fwd":"bwd", K, N, e, pass?"PASS":"*** FAIL ***");

    ALIGNED_FREE(in_re); ALIGNED_FREE(in_im); ALIGNED_FREE(drv_re); ALIGNED_FREE(drv_im);
    ALIGNED_FREE(ref_re); ALIGNED_FREE(ref_im);
    return pass;
}

/*============================================================================
 * Test: forward then backward roundtrip (N1)
 *============================================================================*/
static int test_roundtrip_n1(int K)
{
    int N = 8 * K;
    double *orig_re=alloc64(N), *orig_im=alloc64(N);
    double *fwd_re=alloc64(N),  *fwd_im=alloc64(N);
    double *bwd_re=alloc64(N),  *bwd_im=alloc64(N);

    fill_random(orig_re, N, 1234 + K);
    fill_random(orig_im, N, 5678 + K);

    fft_radix8_fv_n1(fwd_re, fwd_im, orig_re, orig_im, K);
    fft_radix8_bv_n1(bwd_re, bwd_im, fwd_re, fwd_im, K);

    /* Scale by 1/8 (radix-8 DFT normalization) */
    for (int i = 0; i < N; i++) { bwd_re[i] /= 8.0; bwd_im[i] /= 8.0; }

    double e = fmax(max_err(orig_re, bwd_re, N), max_err(orig_im, bwd_im, N));
    double tol = 1e-12;
    int pass = (e < tol);
    printf("  RT  N1     K=%-5d N=%-6d err=%.2e %s\n",
           K, N, e, pass?"PASS":"*** FAIL ***");

    ALIGNED_FREE(orig_re); ALIGNED_FREE(orig_im);
    ALIGNED_FREE(fwd_re);  ALIGNED_FREE(fwd_im);
    ALIGNED_FREE(bwd_re);  ALIGNED_FREE(bwd_im);
    return pass;
}

/*============================================================================
 * Test: forward then backward roundtrip (twiddled)
 *============================================================================*/
static int test_roundtrip_twiddled(int K)
{
    int N = 8 * K;
    double *orig_re=alloc64(N), *orig_im=alloc64(N);
    double *fwd_re=alloc64(N),  *fwd_im=alloc64(N);
    double *bwd_re=alloc64(N),  *bwd_im=alloc64(N);
    double *tw4f_re=alloc64(4*K), *tw4f_im=alloc64(4*K);
    double *tw2f_re=alloc64(2*K), *tw2f_im=alloc64(2*K);
    double *tw4b_re=alloc64(4*K), *tw4b_im=alloc64(4*K);
    double *tw2b_re=alloc64(2*K), *tw2b_im=alloc64(2*K);

    fill_random(orig_re, N, 9999 + K);
    fill_random(orig_im, N, 8888 + K);

    init_tw4(tw4f_re, tw4f_im, K, -1);
    init_tw2(tw2f_re, tw2f_im, K, -1);
    init_tw4(tw4b_re, tw4b_im, K, +1);
    init_tw2(tw2b_re, tw2b_im, K, +1);

    radix8_stage_twiddles_blocked4_t tw4f = { .re=tw4f_re, .im=tw4f_im };
    radix8_stage_twiddles_blocked2_t tw2f = { .re=tw2f_re, .im=tw2f_im };
    radix8_stage_twiddles_blocked4_t tw4b = { .re=tw4b_re, .im=tw4b_im };
    radix8_stage_twiddles_blocked2_t tw2b = { .re=tw2b_re, .im=tw2b_im };

    fft_radix8_fv(fwd_re, fwd_im, orig_re, orig_im, &tw4f, &tw2f, K);
    fft_radix8_bv(bwd_re, bwd_im, fwd_re, fwd_im, &tw4b, &tw2b, K);

    /* Scale by 1/8 */
    for (int i = 0; i < N; i++) { bwd_re[i] /= 8.0; bwd_im[i] /= 8.0; }

    double e = fmax(max_err(orig_re, bwd_re, N), max_err(orig_im, bwd_im, N));
    double tol = 1e-11;
    const char *mode = (K <= RADIX8_BLOCKED4_THRESHOLD) ? "B4" : "B2";
    int pass = (e < tol);
    printf("  RT  %s     K=%-5d N=%-6d err=%.2e %s\n",
           mode, K, N, e, pass?"PASS":"*** FAIL ***");

    ALIGNED_FREE(orig_re); ALIGNED_FREE(orig_im);
    ALIGNED_FREE(fwd_re);  ALIGNED_FREE(fwd_im);
    ALIGNED_FREE(bwd_re);  ALIGNED_FREE(bwd_im);
    ALIGNED_FREE(tw4f_re); ALIGNED_FREE(tw4f_im); ALIGNED_FREE(tw2f_re); ALIGNED_FREE(tw2f_im);
    ALIGNED_FREE(tw4b_re); ALIGNED_FREE(tw4b_im); ALIGNED_FREE(tw2b_re); ALIGNED_FREE(tw2b_im);
    return pass;
}

/*============================================================================*/
int main(void)
{
    printf("============================================\n");
    printf("  Radix-8 Public API Integration Tests\n");
#if defined(__AVX512F__)
    printf("  ISA: AVX-512\n");
#elif defined(__AVX2__)
    printf("  ISA: AVX2\n");
#else
    printf("  ISA: Scalar (FMA)\n");
#endif
    printf("============================================\n");

    int passed = 0, total = 0;

    /* ---- Forward vs reference ---- */
    int Ks_b4[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    int n_b4 = sizeof(Ks_b4)/sizeof(Ks_b4[0]);
    int Ks_b2[] = {512, 1024};
    int n_b2 = sizeof(Ks_b2)/sizeof(Ks_b2[0]);

    printf("\n--- Forward (fv) vs reference ---\n");
    for (int i = 0; i < n_b4; i++) { total++; passed += test_twiddled(Ks_b4[i], -1); }
    for (int i = 0; i < n_b2; i++) { total++; passed += test_twiddled(Ks_b2[i], -1); }

    printf("\n--- Backward (bv) vs reference ---\n");
    for (int i = 0; i < n_b4; i++) { total++; passed += test_twiddled(Ks_b4[i], +1); }
    for (int i = 0; i < n_b2; i++) { total++; passed += test_twiddled(Ks_b2[i], +1); }

    /* ---- N1 forward/backward ---- */
    int Ks_n1[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    int n_n1 = sizeof(Ks_n1)/sizeof(Ks_n1[0]);

    printf("\n--- Forward N1 (fv_n1) vs reference ---\n");
    for (int i = 0; i < n_n1; i++) { total++; passed += test_n1(Ks_n1[i], -1); }

    printf("\n--- Backward N1 (bv_n1) vs reference ---\n");
    for (int i = 0; i < n_n1; i++) { total++; passed += test_n1(Ks_n1[i], +1); }

    /* ---- Roundtrips ---- */
    printf("\n--- Roundtrip fv_n1 → bv_n1 → /8 ---\n");
    for (int i = 0; i < n_n1; i++) { total++; passed += test_roundtrip_n1(Ks_n1[i]); }

    printf("\n--- Roundtrip fv → bv → /8 ---\n");
    for (int i = 0; i < n_b4; i++) { total++; passed += test_roundtrip_twiddled(Ks_b4[i]); }
    for (int i = 0; i < n_b2; i++) { total++; passed += test_roundtrip_twiddled(Ks_b2[i]); }

    printf("\n============================================\n");
    if (passed == total)
        printf("  Results: %d/%d passed  \xe2\x9c\x93 ALL PASS\n", passed, total);
    else
        printf("  Results: %d/%d passed  \xe2\x9c\x97 %d FAILED\n", passed, total, total-passed);
    printf("============================================\n");

    return (passed == total) ? 0 : 1;
}
