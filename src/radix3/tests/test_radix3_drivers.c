/**
 * @file test_radix3_drivers.c
 * @brief Integration test for radix-3 public API (fft_radix3_fv/bv/fv_n1/bv_n1)
 *
 * Tests all K values including non-power-of-2 and non-multiples of 4/8
 * to exercise ISA dispatch and scalar tail paths.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "fft_radix3.h"
#include "vfft_compat.h"   

/*============================================================================
 * PORTABLE ALIGNED ALLOCATION
 *============================================================================*/
#ifdef _MSC_VER
#include <malloc.h>
static double *alloc64(size_t n) {
    double *p = (double *)_aligned_malloc(n * sizeof(double), 64);
    if (!p) { fprintf(stderr, "alloc failed\n"); exit(1); }
    memset(p, 0, n * sizeof(double));
    return p;
}
#define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
static double *alloc64(size_t n) {
    double *p = NULL;
    if (posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) {
        fprintf(stderr, "alloc failed\n"); exit(1);
    }
    memset(p, 0, n * sizeof(double));
    return p;
}
#define ALIGNED_FREE(ptr) free(ptr)
#endif

/*============================================================================
 * TEST INFRASTRUCTURE
 *============================================================================*/
static int g_pass = 0, g_fail = 0;

static double max_abs_error(const double *a, const double *b, size_t n) {
    double mx = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

#define CHECK(cond, fmt, ...) do { \
    if (cond) { printf("  PASS: " fmt "\n", ##__VA_ARGS__); g_pass++; } \
    else      { printf("  FAIL: " fmt "\n", ##__VA_ARGS__); g_fail++; } \
} while(0)

#define TOL 1e-12

/*============================================================================
 * SCALAR REFERENCE (naive DFT-3)
 *============================================================================*/

static void ref_dft3_forward(
    size_t K,
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im)
{
    const double W3_re[3] = {1.0, -0.5, -0.5};
    const double W3_im[3] = {0.0, -0.86602540378443864676, 0.86602540378443864676};

    for (size_t k = 0; k < K; k++) {
        double x_re[3], x_im[3];
        x_re[0] = in_re[k];     x_im[0] = in_im[k];
        double br = in_re[K+k], bi = in_im[K+k];
        x_re[1] = br*tw_re[k]     - bi*tw_im[k];
        x_im[1] = br*tw_im[k]     + bi*tw_re[k];
        double cr = in_re[2*K+k], ci = in_im[2*K+k];
        x_re[2] = cr*tw_re[K+k]   - ci*tw_im[K+k];
        x_im[2] = cr*tw_im[K+k]   + ci*tw_re[K+k];

        for (int j = 0; j < 3; j++) {
            double yr = 0.0, yi = 0.0;
            for (int m = 0; m < 3; m++) {
                int idx = (j*m) % 3;
                yr += x_re[m]*W3_re[idx] - x_im[m]*W3_im[idx];
                yi += x_re[m]*W3_im[idx] + x_im[m]*W3_re[idx];
            }
            out_re[j*K+k] = yr;
            out_im[j*K+k] = yi;
        }
    }
}

static void ref_dft3_backward(
    size_t K,
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im)
{
    const double W3_re[3] = {1.0, -0.5, -0.5};
    const double W3_im[3] = {0.0, 0.86602540378443864676, -0.86602540378443864676};

    for (size_t k = 0; k < K; k++) {
        double x_re[3], x_im[3];
        x_re[0] = in_re[k];     x_im[0] = in_im[k];
        double br = in_re[K+k], bi = in_im[K+k];
        x_re[1] = br*tw_re[k]     - bi*tw_im[k];
        x_im[1] = br*tw_im[k]     + bi*tw_re[k];
        double cr = in_re[2*K+k], ci = in_im[2*K+k];
        x_re[2] = cr*tw_re[K+k]   - ci*tw_im[K+k];
        x_im[2] = cr*tw_im[K+k]   + ci*tw_re[K+k];

        for (int j = 0; j < 3; j++) {
            double yr = 0.0, yi = 0.0;
            for (int m = 0; m < 3; m++) {
                int idx = (j*m) % 3;
                yr += x_re[m]*W3_re[idx] - x_im[m]*W3_im[idx];
                yi += x_re[m]*W3_im[idx] + x_im[m]*W3_re[idx];
            }
            out_re[j*K+k] = yr;
            out_im[j*K+k] = yi;
        }
    }
}

/*============================================================================
 * HELPERS
 *============================================================================*/

static void gen_twiddles(size_t K, double *tw_re, double *tw_im) {
    const double TWO_PI = 6.283185307179586476925286766559;
    for (size_t k = 0; k < K; k++) {
        double a1 = -TWO_PI * (double)k / (3.0 * (double)K);
        tw_re[k]   = cos(a1);  tw_im[k]   = sin(a1);
        double a2 = -TWO_PI * 2.0 * (double)k / (3.0 * (double)K);
        tw_re[K+k] = cos(a2);  tw_im[K+k] = sin(a2);
    }
}

static void fill_data(double *arr, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; i++) {
        seed = seed * 1103515245u + 12345u;
        arr[i] = ((double)(seed >> 16) / 32768.0) - 1.0;
    }
}

/*============================================================================
 * TESTS
 *============================================================================*/

static void test_fv_bv_vs_ref(void) {
    printf("\n=== fft_radix3_fv / fft_radix3_bv vs reference ===\n");
    size_t Ks[] = {1,2,3,4,5,7,8,9,16,27,32,64,81,128,243,256,512,1024};
    int nK = sizeof(Ks)/sizeof(Ks[0]);

    for (int i = 0; i < nK; i++) {
        size_t K = Ks[i];
        size_t N = 3*K;
        double *in_re = alloc64(N), *in_im = alloc64(N);
        double *out_re = alloc64(N), *out_im = alloc64(N);
        double *ref_re = alloc64(N), *ref_im = alloc64(N);
        double *tw_re = alloc64(2*K), *tw_im = alloc64(2*K);

        fill_data(in_re, N, 100+(unsigned)K);
        fill_data(in_im, N, 200+(unsigned)K);
        gen_twiddles(K, tw_re, tw_im);
        radix3_stage_twiddles_t tw = {tw_re, tw_im};

        /* Forward */
        ref_dft3_forward(K, in_re, in_im, ref_re, ref_im, tw_re, tw_im);
        fft_radix3_fv(K, in_re, in_im, out_re, out_im, &tw);
        double err = fmax(max_abs_error(out_re, ref_re, N),
                          max_abs_error(out_im, ref_im, N));
        CHECK(err < TOL, "fv  K=%4zu  err=%.2e", K, err);

        /* Backward */
        ref_dft3_backward(K, in_re, in_im, ref_re, ref_im, tw_re, tw_im);
        fft_radix3_bv(K, in_re, in_im, out_re, out_im, &tw);
        err = fmax(max_abs_error(out_re, ref_re, N),
                   max_abs_error(out_im, ref_im, N));
        CHECK(err < TOL, "bv  K=%4zu  err=%.2e", K, err);

        ALIGNED_FREE(in_re); ALIGNED_FREE(in_im);
        ALIGNED_FREE(out_re); ALIGNED_FREE(out_im);
        ALIGNED_FREE(ref_re); ALIGNED_FREE(ref_im);
        ALIGNED_FREE(tw_re); ALIGNED_FREE(tw_im);
    }
}

static void test_n1_vs_ref(void) {
    printf("\n=== fft_radix3_fv_n1 / fft_radix3_bv_n1 vs reference ===\n");
    size_t Ks[] = {1,2,3,4,5,7,8,9,16,27,32,64,81,128,243,256,512,1024};
    int nK = sizeof(Ks)/sizeof(Ks[0]);

    for (int i = 0; i < nK; i++) {
        size_t K = Ks[i];
        size_t N = 3*K;
        double *in_re = alloc64(N), *in_im = alloc64(N);
        double *out_re = alloc64(N), *out_im = alloc64(N);
        double *ref_re = alloc64(N), *ref_im = alloc64(N);
        /* Unit twiddles */
        double *tw_re = alloc64(2*K), *tw_im = alloc64(2*K);
        for (size_t k = 0; k < 2*K; k++) { tw_re[k] = 1.0; tw_im[k] = 0.0; }

        fill_data(in_re, N, 300+(unsigned)K);
        fill_data(in_im, N, 400+(unsigned)K);

        /* Forward N1 */
        ref_dft3_forward(K, in_re, in_im, ref_re, ref_im, tw_re, tw_im);
        fft_radix3_fv_n1(K, in_re, in_im, out_re, out_im);
        double err = fmax(max_abs_error(out_re, ref_re, N),
                          max_abs_error(out_im, ref_im, N));
        CHECK(err < TOL, "fv_n1  K=%4zu  err=%.2e", K, err);

        /* Backward N1 */
        ref_dft3_backward(K, in_re, in_im, ref_re, ref_im, tw_re, tw_im);
        fft_radix3_bv_n1(K, in_re, in_im, out_re, out_im);
        err = fmax(max_abs_error(out_re, ref_re, N),
                   max_abs_error(out_im, ref_im, N));
        CHECK(err < TOL, "bv_n1  K=%4zu  err=%.2e", K, err);

        ALIGNED_FREE(in_re); ALIGNED_FREE(in_im);
        ALIGNED_FREE(out_re); ALIGNED_FREE(out_im);
        ALIGNED_FREE(ref_re); ALIGNED_FREE(ref_im);
        ALIGNED_FREE(tw_re); ALIGNED_FREE(tw_im);
    }
}

static void test_n1_roundtrip(void) {
    printf("\n=== N1 roundtrip: fv_n1 → bv_n1 → /3 = identity ===\n");
    size_t Ks[] = {1,2,3,4,5,7,8,9,16,27,32,64,81,128,243,256,512,1024};
    int nK = sizeof(Ks)/sizeof(Ks[0]);

    for (int i = 0; i < nK; i++) {
        size_t K = Ks[i];
        size_t N = 3*K;
        double *in_re = alloc64(N), *in_im = alloc64(N);
        double *mid_re = alloc64(N), *mid_im = alloc64(N);
        double *out_re = alloc64(N), *out_im = alloc64(N);

        fill_data(in_re, N, 500+(unsigned)K);
        fill_data(in_im, N, 600+(unsigned)K);

        fft_radix3_fv_n1(K, in_re, in_im, mid_re, mid_im);
        fft_radix3_bv_n1(K, mid_re, mid_im, out_re, out_im);

        for (size_t j = 0; j < N; j++) {
            out_re[j] /= 3.0;
            out_im[j] /= 3.0;
        }

        double err = fmax(max_abs_error(out_re, in_re, N),
                          max_abs_error(out_im, in_im, N));
        CHECK(err < 1e-14, "roundtrip K=%4zu  err=%.2e", K, err);

        ALIGNED_FREE(in_re); ALIGNED_FREE(in_im);
        ALIGNED_FREE(mid_re); ALIGNED_FREE(mid_im);
        ALIGNED_FREE(out_re); ALIGNED_FREE(out_im);
    }
}

/*============================================================================
 * MAIN
 *============================================================================*/

int main(void) {
    printf("Radix-3 Driver Integration Tests\n");
    printf("=================================\n");
    printf("ISA dispatch: scalar");
#if defined(__AVX2__) && defined(__FMA__)
    printf(" + AVX2");
#endif
#if defined(__AVX512F__)
    printf(" + AVX-512");
#endif
    printf("\n");

    test_fv_bv_vs_ref();
    test_n1_vs_ref();
    test_n1_roundtrip();

    printf("\n=================================\n");
    printf("TOTAL: %d passed, %d failed out of %d\n",
           g_pass, g_fail, g_pass + g_fail);
    return g_fail > 0 ? 1 : 0;
}
