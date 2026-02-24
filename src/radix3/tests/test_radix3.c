/**
 * @file test_radix3.c
 * @brief Comprehensive radix-3 stage tests
 *
 * Tests scalar, AVX2, AVX-512 forward/backward × twiddled/N1.
 * Reference: naive O(N) DFT-3 butterfly.
 *
 * Build (Linux, AVX-512):
 *   gcc -O2 -mavx512f -mavx512dq -mfma -o test_radix3 test_radix3.c -lm
 *
 * Build (Linux, AVX2 only):
 *   gcc -O2 -mavx2 -mfma -o test_radix3 test_radix3.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Include all ISA headers */
#include "scalar/fft_radix3_scalar.h"
#include "scalar/fft_radix3_scalar_n1.h"

#if defined(__AVX2__) && defined(__FMA__)
#include "avx2/fft_radix3_avx2.h"
#include "avx2/fft_radix3_avx2_n1.h"
#define HAVE_AVX2 1
#else
#define HAVE_AVX2 0
#endif

#if defined(__AVX512F__)
#include "avx512/fft_radix3_avx512.h"
#include "avx512/fft_radix3_avx512_n1.h"
#define HAVE_AVX512 1
#else
#define HAVE_AVX512 0
#endif

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
 * NAIVE DFT-3 REFERENCE (scalar, no optimisation)
 *============================================================================*/

/**
 * Naive radix-3 forward butterfly with twiddles.
 * y[j] = sum_{m=0}^{2} x[m*K+k] · W_stage[m][k] · ω3^{j·m}
 * where ω3 = e^{-2πi/3} (forward).
 */
static void naive_dft3_forward(
    size_t K,
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im)  /* tw_re/im: [2*K] */
{
    const double W3_re[3] = {1.0, -0.5, -0.5};
    const double W3_im[3] = {0.0, -0.86602540378443864676, 0.86602540378443864676};

    for (size_t k = 0; k < K; k++) {
        /* Apply twiddles: row 0 has W^0=1, row 1 has W1, row 2 has W2 */
        double x_re[3], x_im[3];
        x_re[0] = in_re[k];       x_im[0] = in_im[k];
        /* x[1] = in[1] * W1 */
        double b_r = in_re[K+k], b_i = in_im[K+k];
        x_re[1] = b_r * tw_re[k]     - b_i * tw_im[k];
        x_im[1] = b_r * tw_im[k]     + b_i * tw_re[k];
        /* x[2] = in[2] * W2 */
        double c_r = in_re[2*K+k], c_i = in_im[2*K+k];
        x_re[2] = c_r * tw_re[K+k]   - c_i * tw_im[K+k];
        x_im[2] = c_r * tw_im[K+k]   + c_i * tw_re[K+k];

        for (int j = 0; j < 3; j++) {
            double yr = 0.0, yi = 0.0;
            for (int m = 0; m < 3; m++) {
                int idx = (j * m) % 3;
                yr += x_re[m] * W3_re[idx] - x_im[m] * W3_im[idx];
                yi += x_re[m] * W3_im[idx] + x_im[m] * W3_re[idx];
            }
            out_re[j*K + k] = yr;
            out_im[j*K + k] = yi;
        }
    }
}

static void naive_dft3_backward(
    size_t K,
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im)
{
    /* ω3_inv = e^{+2πi/3}: conjugate of forward */
    const double W3_re[3] = {1.0, -0.5, -0.5};
    const double W3_im[3] = {0.0, 0.86602540378443864676, -0.86602540378443864676};

    for (size_t k = 0; k < K; k++) {
        double x_re[3], x_im[3];
        x_re[0] = in_re[k]; x_im[0] = in_im[k];
        double b_r = in_re[K+k], b_i = in_im[K+k];
        x_re[1] = b_r * tw_re[k]   - b_i * tw_im[k];
        x_im[1] = b_r * tw_im[k]   + b_i * tw_re[k];
        double c_r = in_re[2*K+k], c_i = in_im[2*K+k];
        x_re[2] = c_r * tw_re[K+k] - c_i * tw_im[K+k];
        x_im[2] = c_r * tw_im[K+k] + c_i * tw_re[K+k];

        for (int j = 0; j < 3; j++) {
            double yr = 0.0, yi = 0.0;
            for (int m = 0; m < 3; m++) {
                int idx = (j * m) % 3;
                yr += x_re[m] * W3_re[idx] - x_im[m] * W3_im[idx];
                yi += x_re[m] * W3_im[idx] + x_im[m] * W3_re[idx];
            }
            out_re[j*K + k] = yr;
            out_im[j*K + k] = yi;
        }
    }
}

/*============================================================================
 * TWIDDLE GENERATION (SoA contiguous)
 *============================================================================*/

/**
 * Generate stage twiddles: W^{mk} for m=1,2 and k=0..K-1
 * where W = e^{-2πi / (3K)}  (forward convention, matches our headers)
 *
 * Layout:
 *   tw_re[0..K-1]   = W1_re = cos(-2π·k / (3K))
 *   tw_re[K..2K-1]  = W2_re = cos(-2π·2k / (3K))
 *   tw_im same
 */
static void gen_twiddles(size_t K, double *tw_re, double *tw_im) {
    const double TWO_PI = 6.283185307179586476925286766559;
    for (size_t k = 0; k < K; k++) {
        double angle1 = -TWO_PI * (double)k / (3.0 * (double)K);
        tw_re[k]   = cos(angle1);
        tw_im[k]   = sin(angle1);
        double angle2 = -TWO_PI * 2.0 * (double)k / (3.0 * (double)K);
        tw_re[K+k] = cos(angle2);
        tw_im[K+k] = sin(angle2);
    }
}

/**
 * Fill array with deterministic pseudo-random data.
 */
static void fill_data(double *arr, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; i++) {
        seed = seed * 1103515245u + 12345u;
        arr[i] = ((double)(seed >> 16) / 32768.0) - 1.0;
    }
}

/*============================================================================
 * TEST GROUPS
 *============================================================================*/

static void test_scalar_twiddled(void) {
    printf("\n=== SCALAR TWIDDLED ===\n");
    size_t Ks[] = {1, 2, 3, 4, 7, 8, 9, 16, 27, 32, 64, 81, 128, 243, 256};
    int nK = sizeof(Ks)/sizeof(Ks[0]);

    for (int i = 0; i < nK; i++) {
        size_t K = Ks[i];
        size_t N = 3 * K;

        double *in_re  = alloc64(N), *in_im  = alloc64(N);
        double *out_re = alloc64(N), *out_im = alloc64(N);
        double *ref_re = alloc64(N), *ref_im = alloc64(N);
        double *tw_re  = alloc64(2*K), *tw_im = alloc64(2*K);

        fill_data(in_re, N, 1000 + (unsigned)K);
        fill_data(in_im, N, 2000 + (unsigned)K);
        gen_twiddles(K, tw_re, tw_im);

        radix3_stage_twiddles_t tw = {tw_re, tw_im};

        /* Forward */
        naive_dft3_forward(K, in_re, in_im, ref_re, ref_im, tw_re, tw_im);
        radix3_stage_forward_scalar(K, in_re, in_im, out_re, out_im, &tw);
        double err_re = max_abs_error(out_re, ref_re, N);
        double err_im = max_abs_error(out_im, ref_im, N);
        CHECK(err_re < TOL && err_im < TOL,
              "fwd K=%3zu  err=%.2e", K, fmax(err_re, err_im));

        /* Backward */
        naive_dft3_backward(K, in_re, in_im, ref_re, ref_im, tw_re, tw_im);
        radix3_stage_backward_scalar(K, in_re, in_im, out_re, out_im, &tw);
        err_re = max_abs_error(out_re, ref_re, N);
        err_im = max_abs_error(out_im, ref_im, N);
        CHECK(err_re < TOL && err_im < TOL,
              "bwd K=%3zu  err=%.2e", K, fmax(err_re, err_im));

        ALIGNED_FREE(in_re); ALIGNED_FREE(in_im);
        ALIGNED_FREE(out_re); ALIGNED_FREE(out_im);
        ALIGNED_FREE(ref_re); ALIGNED_FREE(ref_im);
        ALIGNED_FREE(tw_re); ALIGNED_FREE(tw_im);
    }
}

static void test_scalar_n1(void) {
    printf("\n=== SCALAR N1 (twiddle-less) ===\n");
    size_t Ks[] = {1, 2, 3, 4, 7, 8, 16, 32, 64, 128, 256};
    int nK = sizeof(Ks)/sizeof(Ks[0]);

    for (int i = 0; i < nK; i++) {
        size_t K = Ks[i];
        size_t N = 3 * K;

        double *in_re  = alloc64(N), *in_im  = alloc64(N);
        double *out_re = alloc64(N), *out_im = alloc64(N);
        double *ref_re = alloc64(N), *ref_im = alloc64(N);
        /* Unit twiddles */
        double *tw_re  = alloc64(2*K), *tw_im = alloc64(2*K);
        for (size_t k = 0; k < 2*K; k++) { tw_re[k] = 1.0; tw_im[k] = 0.0; }

        fill_data(in_re, N, 3000 + (unsigned)K);
        fill_data(in_im, N, 4000 + (unsigned)K);

        /* Forward N1 vs naive with unit twiddles */
        naive_dft3_forward(K, in_re, in_im, ref_re, ref_im, tw_re, tw_im);
        radix3_stage_n1_forward_scalar(K, in_re, in_im, out_re, out_im);
        double err = fmax(max_abs_error(out_re, ref_re, N),
                          max_abs_error(out_im, ref_im, N));
        CHECK(err < TOL, "fwd_n1 K=%3zu  err=%.2e", K, err);

        /* Backward N1 */
        naive_dft3_backward(K, in_re, in_im, ref_re, ref_im, tw_re, tw_im);
        radix3_stage_n1_backward_scalar(K, in_re, in_im, out_re, out_im);
        err = fmax(max_abs_error(out_re, ref_re, N),
                   max_abs_error(out_im, ref_im, N));
        CHECK(err < TOL, "bwd_n1 K=%3zu  err=%.2e", K, err);

        ALIGNED_FREE(in_re); ALIGNED_FREE(in_im);
        ALIGNED_FREE(out_re); ALIGNED_FREE(out_im);
        ALIGNED_FREE(ref_re); ALIGNED_FREE(ref_im);
        ALIGNED_FREE(tw_re); ALIGNED_FREE(tw_im);
    }
}

static void test_n1_roundtrip(void) {
    printf("\n=== N1 ROUNDTRIP (fwd→bwd→/3 = identity) ===\n");
    size_t Ks[] = {1, 2, 3, 4, 8, 16, 32, 64, 128, 256};
    int nK = sizeof(Ks)/sizeof(Ks[0]);

    for (int i = 0; i < nK; i++) {
        size_t K = Ks[i];
        size_t N = 3 * K;

        double *in_re  = alloc64(N), *in_im  = alloc64(N);
        double *mid_re = alloc64(N), *mid_im = alloc64(N);
        double *out_re = alloc64(N), *out_im = alloc64(N);

        fill_data(in_re, N, 5000 + (unsigned)K);
        fill_data(in_im, N, 6000 + (unsigned)K);

        /* Use highest available ISA for roundtrip */
#if HAVE_AVX512
        radix3_stage_n1_forward_avx512(K, in_re, in_im, mid_re, mid_im);
        radix3_stage_n1_backward_avx512(K, mid_re, mid_im, out_re, out_im);
        const char *isa = "avx512";
#elif HAVE_AVX2
        radix3_stage_n1_forward_avx2(K, in_re, in_im, mid_re, mid_im);
        radix3_stage_n1_backward_avx2(K, mid_re, mid_im, out_re, out_im);
        const char *isa = "avx2";
#else
        radix3_stage_n1_forward_scalar(K, in_re, in_im, mid_re, mid_im);
        radix3_stage_n1_backward_scalar(K, mid_re, mid_im, out_re, out_im);
        const char *isa = "scalar";
#endif
        /* Divide by 3 */
        for (size_t j = 0; j < N; j++) {
            out_re[j] /= 3.0;
            out_im[j] /= 3.0;
        }

        double err = fmax(max_abs_error(out_re, in_re, N),
                          max_abs_error(out_im, in_im, N));
        CHECK(err < 1e-14, "roundtrip [%s] K=%3zu  err=%.2e", isa, K, err);

        ALIGNED_FREE(in_re); ALIGNED_FREE(in_im);
        ALIGNED_FREE(mid_re); ALIGNED_FREE(mid_im);
        ALIGNED_FREE(out_re); ALIGNED_FREE(out_im);
    }
}

#if HAVE_AVX2
static void test_avx2_vs_scalar(void) {
    printf("\n=== AVX2 vs SCALAR ===\n");
    /* Include K values not multiple of 4 to test scalar tail */
    size_t Ks[] = {1, 2, 3, 4, 5, 7, 8, 9, 16, 27, 32, 64, 81, 128, 243, 256};
    int nK = sizeof(Ks)/sizeof(Ks[0]);

    for (int i = 0; i < nK; i++) {
        size_t K = Ks[i];
        size_t N = 3 * K;

        double *in_re  = alloc64(N),  *in_im  = alloc64(N);
        double *s_re   = alloc64(N),  *s_im   = alloc64(N);
        double *v_re   = alloc64(N),  *v_im   = alloc64(N);
        double *tw_re  = alloc64(2*K), *tw_im = alloc64(2*K);

        fill_data(in_re, N, 7000 + (unsigned)K);
        fill_data(in_im, N, 8000 + (unsigned)K);
        gen_twiddles(K, tw_re, tw_im);
        radix3_stage_twiddles_t tw = {tw_re, tw_im};

        /* Forward */
        radix3_stage_forward_scalar(K, in_re, in_im, s_re, s_im, &tw);
        radix3_stage_forward_avx2(K, in_re, in_im, v_re, v_im, &tw);
        double err = fmax(max_abs_error(v_re, s_re, N),
                          max_abs_error(v_im, s_im, N));
        CHECK(err < 1e-14, "fwd K=%3zu  err=%.2e", K, err);

        /* Backward */
        radix3_stage_backward_scalar(K, in_re, in_im, s_re, s_im, &tw);
        radix3_stage_backward_avx2(K, in_re, in_im, v_re, v_im, &tw);
        err = fmax(max_abs_error(v_re, s_re, N),
                   max_abs_error(v_im, s_im, N));
        CHECK(err < 1e-14, "bwd K=%3zu  err=%.2e", K, err);

        /* N1 forward */
        radix3_stage_n1_forward_scalar(K, in_re, in_im, s_re, s_im);
        radix3_stage_n1_forward_avx2(K, in_re, in_im, v_re, v_im);
        err = fmax(max_abs_error(v_re, s_re, N),
                   max_abs_error(v_im, s_im, N));
        CHECK(err < 1e-14, "fwd_n1 K=%3zu  err=%.2e", K, err);

        /* N1 backward */
        radix3_stage_n1_backward_scalar(K, in_re, in_im, s_re, s_im);
        radix3_stage_n1_backward_avx2(K, in_re, in_im, v_re, v_im);
        err = fmax(max_abs_error(v_re, s_re, N),
                   max_abs_error(v_im, s_im, N));
        CHECK(err < 1e-14, "bwd_n1 K=%3zu  err=%.2e", K, err);

        ALIGNED_FREE(in_re); ALIGNED_FREE(in_im);
        ALIGNED_FREE(s_re); ALIGNED_FREE(s_im);
        ALIGNED_FREE(v_re); ALIGNED_FREE(v_im);
        ALIGNED_FREE(tw_re); ALIGNED_FREE(tw_im);
    }
}
#endif

#if HAVE_AVX512
static void test_avx512_vs_scalar(void) {
    printf("\n=== AVX-512 vs SCALAR ===\n");
    size_t Ks[] = {1, 2, 3, 4, 5, 7, 8, 9, 16, 27, 32, 64, 81, 128, 243, 256};
    int nK = sizeof(Ks)/sizeof(Ks[0]);

    for (int i = 0; i < nK; i++) {
        size_t K = Ks[i];
        size_t N = 3 * K;

        double *in_re  = alloc64(N),  *in_im  = alloc64(N);
        double *s_re   = alloc64(N),  *s_im   = alloc64(N);
        double *v_re   = alloc64(N),  *v_im   = alloc64(N);
        double *tw_re  = alloc64(2*K), *tw_im = alloc64(2*K);

        fill_data(in_re, N, 9000 + (unsigned)K);
        fill_data(in_im, N, 10000 + (unsigned)K);
        gen_twiddles(K, tw_re, tw_im);
        radix3_stage_twiddles_t tw = {tw_re, tw_im};

        /* Forward */
        radix3_stage_forward_scalar(K, in_re, in_im, s_re, s_im, &tw);
        radix3_stage_forward_avx512(K, in_re, in_im, v_re, v_im, &tw);
        double err = fmax(max_abs_error(v_re, s_re, N),
                          max_abs_error(v_im, s_im, N));
        CHECK(err < 1e-14, "fwd K=%3zu  err=%.2e", K, err);

        /* Backward */
        radix3_stage_backward_scalar(K, in_re, in_im, s_re, s_im, &tw);
        radix3_stage_backward_avx512(K, in_re, in_im, v_re, v_im, &tw);
        err = fmax(max_abs_error(v_re, s_re, N),
                   max_abs_error(v_im, s_im, N));
        CHECK(err < 1e-14, "bwd K=%3zu  err=%.2e", K, err);

        /* N1 forward */
        radix3_stage_n1_forward_scalar(K, in_re, in_im, s_re, s_im);
        radix3_stage_n1_forward_avx512(K, in_re, in_im, v_re, v_im);
        err = fmax(max_abs_error(v_re, s_re, N),
                   max_abs_error(v_im, s_im, N));
        CHECK(err < 1e-14, "fwd_n1 K=%3zu  err=%.2e", K, err);

        /* N1 backward */
        radix3_stage_n1_backward_scalar(K, in_re, in_im, s_re, s_im);
        radix3_stage_n1_backward_avx512(K, in_re, in_im, v_re, v_im);
        err = fmax(max_abs_error(v_re, s_re, N),
                   max_abs_error(v_im, s_im, N));
        CHECK(err < 1e-14, "bwd_n1 K=%3zu  err=%.2e", K, err);

        ALIGNED_FREE(in_re); ALIGNED_FREE(in_im);
        ALIGNED_FREE(s_re); ALIGNED_FREE(s_im);
        ALIGNED_FREE(v_re); ALIGNED_FREE(v_im);
        ALIGNED_FREE(tw_re); ALIGNED_FREE(tw_im);
    }
}
#endif

/*============================================================================
 * MAIN
 *============================================================================*/

int main(void) {
    printf("Radix-3 Stage Tests\n");
    printf("====================\n");
    printf("ISA: scalar");
#if HAVE_AVX2
    printf(" + AVX2");
#endif
#if HAVE_AVX512
    printf(" + AVX-512");
#endif
    printf("\n");

    test_scalar_twiddled();
    test_scalar_n1();
    test_n1_roundtrip();

#if HAVE_AVX2
    test_avx2_vs_scalar();
#endif

#if HAVE_AVX512
    test_avx512_vs_scalar();
#endif

    printf("\n====================\n");
    printf("TOTAL: %d passed, %d failed out of %d\n",
           g_pass, g_fail, g_pass + g_fail);

    return g_fail > 0 ? 1 : 0;
}