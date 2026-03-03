/**
 * @file test_radix32_avx2.c
 * @brief Unit tests for AVX2 radix-32 FFT implementations
 *
 * Tests:
 *   1. n1 forward vs naive O(N²) DFT reference  (correctness)
 *   2. n1 forward → backward round-trip           (structural consistency)
 *   3. twiddle forward → backward round-trip       (structural consistency)
 *   4. n1 backward vs naive IDFT reference         (correctness)
 *   5. n1 impulse response                         (known-answer)
 *   6. n1 DC response                              (known-answer)
 *
 * Build (ICX):
 *   icx -O2 -mavx2 -mfma -lm -o test_radix32_avx2 test_radix32_avx2.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#include "../fft_radix32_platform.h"
#include "../avx2/fft_radix32_avx2.h"
#include "../avx2/fft_radix32_avx2_n1.h"

/*==========================================================================
 * TEST FRAMEWORK (minimal, zero dependencies)
 *=========================================================================*/

static int g_tests_run    = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_BEGIN(name)                                                \
    do {                                                                \
        g_tests_run++;                                                  \
        const char *_test_name = (name);                                \
        printf("  [RUN ] %s\n", _test_name);                           \
        int _fail = 0;

#define TEST_END()                                                      \
        if (_fail) {                                                    \
            g_tests_failed++;                                           \
            printf("  [FAIL] %s\n", _test_name);                       \
        } else {                                                        \
            g_tests_passed++;                                           \
            printf("  [ OK ] %s\n", _test_name);                       \
        }                                                               \
    } while (0)

#define EXPECT_LT(val, thresh, fmt, ...)                                \
    do {                                                                \
        if ((val) >= (thresh)) {                                        \
            printf("    FAIL: " fmt "\n", __VA_ARGS__);                 \
            _fail = 1;                                                  \
        }                                                               \
    } while (0)

#define EXPECT_NEAR(a, b, tol, fmt, ...)                                \
    do {                                                                \
        if (fabs((a) - (b)) > (tol)) {                                  \
            printf("    FAIL: " fmt "\n", __VA_ARGS__);                 \
            _fail = 1;                                                  \
        }                                                               \
    } while (0)

/*==========================================================================
 * ALIGNED ALLOCATION
 *=========================================================================*/

static double *alloc_aligned(size_t count)
{
    double *p = (double *)r32_aligned_alloc(32, count * sizeof(double));
    if (!p) {
        fprintf(stderr, "FATAL: r32_aligned_alloc failed\n");
        exit(1);
    }
    memset(p, 0, count * sizeof(double));
    return p;
}

/*==========================================================================
 * NAIVE O(N²) DFT REFERENCE
 *=========================================================================*/

#define N 32

/**
 * @brief Naive forward DFT: X[m] = Σ_n x[n] · exp(-2πi·m·n/N)
 *
 * Operates on 4-wide interleaved layout matching the SIMD convention:
 * x_re[stripe * stride + lane], lane ∈ {0..3}
 */
static void naive_dft_forward(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    size_t stride)
{
    for (int lane = 0; lane < 4; lane++) {
        for (int m = 0; m < N; m++) {
            double sumr = 0.0, sumi = 0.0;
            for (int n = 0; n < N; n++) {
                double angle = -2.0 * M_PI * (double)m * (double)n / (double)N;
                double wr = cos(angle), wi = sin(angle);
                double xr = in_re[n * stride + lane];
                double xi = in_im[n * stride + lane];
                sumr += xr * wr - xi * wi;
                sumi += xr * wi + xi * wr;
            }
            out_re[m * stride + lane] = sumr;
            out_im[m * stride + lane] = sumi;
        }
    }
}

/** @brief Naive backward DFT: x[n] = Σ_m X[m] · exp(+2πi·m·n/N)  (no 1/N) */
static void naive_dft_backward(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    size_t stride)
{
    for (int lane = 0; lane < 4; lane++) {
        for (int n = 0; n < N; n++) {
            double sumr = 0.0, sumi = 0.0;
            for (int m = 0; m < N; m++) {
                double angle = +2.0 * M_PI * (double)m * (double)n / (double)N;
                double wr = cos(angle), wi = sin(angle);
                double xr = in_re[m * stride + lane];
                double xi = in_im[m * stride + lane];
                sumr += xr * wr - xi * wi;
                sumi += xr * wi + xi * wr;
            }
            out_re[n * stride + lane] = sumr;
            out_im[n * stride + lane] = sumi;
        }
    }
}

/*==========================================================================
 * SEEDED PRNG (deterministic, xoshiro256**)
 *=========================================================================*/

static uint64_t rng_s[4] = {
    0x180EC6D33CFD0ABAULL, 0xD5A61266F0C9392CULL,
    0xA9582618E03FC9AAULL, 0x39ABDC4529B1661CULL
};

static uint64_t rng_rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t rng_next(void) {
    uint64_t result = rng_rotl(rng_s[1] * 5, 7) * 9;
    uint64_t t = rng_s[1] << 17;
    rng_s[2] ^= rng_s[0]; rng_s[3] ^= rng_s[1];
    rng_s[1] ^= rng_s[2]; rng_s[0] ^= rng_s[3];
    rng_s[2] ^= t; rng_s[3] = rng_rotl(rng_s[3], 45);
    return result;
}

/** @brief Uniform double in [-1, +1] */
static double rng_uniform(void) {
    return (double)(int64_t)rng_next() * 5.42101086242752217e-20;
}

static void fill_random(double *buf, size_t count) {
    for (size_t i = 0; i < count; i++) buf[i] = rng_uniform();
}

/*==========================================================================
 * ERROR METRICS
 *=========================================================================*/

/** @brief Max absolute error across all elements */
static double max_abs_error(
    const double *a, const double *b, size_t count)
{
    double mx = 0.0;
    for (size_t i = 0; i < count; i++) {
        double e = fabs(a[i] - b[i]);
        if (e > mx) mx = e;
    }
    return mx;
}

/** @brief Relative L2 error: ||a - b||₂ / ||b||₂ */
static double rel_l2_error(
    const double *a_re, const double *a_im,
    const double *b_re, const double *b_im,
    size_t count)
{
    double err2 = 0.0, ref2 = 0.0;
    for (size_t i = 0; i < count; i++) {
        double dr = a_re[i] - b_re[i], di = a_im[i] - b_im[i];
        err2 += dr * dr + di * di;
        ref2 += b_re[i] * b_re[i] + b_im[i] * b_im[i];
    }
    return (ref2 > 0.0) ? sqrt(err2 / ref2) : sqrt(err2);
}

/*==========================================================================
 * OUTPUT PERMUTATION
 *
 * The 4×8 decomposition produces output in digit-reversed order.
 * For freq m = 4f + e (e∈{0..3}, f∈{0..7}):
 *   output stripe = stored at perm_freq_to_stripe[m]
 *
 * Formula: stripe(m) = 8*(m%4) + ((-m/4 - m%4*7) mod 8)
 * (empirically derived — matches the DIT-4 → twiddle → DIF-8 structure)
 *
 * This permutation is identical for forward and backward transforms.
 * The planner accounts for this when composing stages.
 *=========================================================================*/

static const int PERM_FREQ_TO_STRIPE[32] = {
     0, 31, 23, 15,  7, 30, 22, 14,
     6, 29, 21, 13,  5, 28, 20, 12,
     4, 27, 19, 11,  3, 26, 18, 10,
     2, 25, 17,  9,  1, 24, 16,  8
};

static const int PERM_STRIPE_TO_FREQ[32] = {
     0, 28, 24, 20, 16, 12,  8,  4,
    31, 27, 23, 19, 15, 11,  7,  3,
    30, 26, 22, 18, 14, 10,  6,  2,
    29, 25, 21, 17, 13,  9,  5,  1
};

/** @brief Reorder permuted output to natural DFT order (one lane) */
static void depermute(
    const double *perm_re, const double *perm_im,
    double *nat_re, double *nat_im,
    size_t stride, int lane)
{
    for (int m = 0; m < N; m++) {
        int s = PERM_FREQ_TO_STRIPE[m];
        nat_re[m * stride + lane] = perm_re[s * stride + lane];
        nat_im[m * stride + lane] = perm_im[s * stride + lane];
    }
}

static void test_n1_forward_vs_naive(void)
{
    TEST_BEGIN("n1_forward_vs_naive_dft");

    const size_t stride = 4;
    const size_t total  = N * stride;

    double *in_re  = alloc_aligned(total);
    double *in_im  = alloc_aligned(total);
    double *out_re = alloc_aligned(total);
    double *out_im = alloc_aligned(total);
    double *nat_re = alloc_aligned(total);
    double *nat_im = alloc_aligned(total);
    double *ref_re = alloc_aligned(total);
    double *ref_im = alloc_aligned(total);

    fill_random(in_re, total);
    fill_random(in_im, total);

    /* n1 codelet (produces permuted output) */
    fft_radix32_n1_forward_avx2(in_re, in_im, out_re, out_im, stride, stride);

    /* depermute to natural order for comparison */
    for (int lane = 0; lane < 4; lane++)
        depermute(out_re, out_im, nat_re, nat_im, stride, lane);

    /* naive reference (natural order) */
    naive_dft_forward(in_re, in_im, ref_re, ref_im, stride);

    double err = rel_l2_error(nat_re, nat_im, ref_re, ref_im, total);
    double max_re = max_abs_error(nat_re, ref_re, total);
    double max_im = max_abs_error(nat_im, ref_im, total);

    printf("    rel L2 error: %.3e    max|Δre|: %.3e    max|Δim|: %.3e\n",
           err, max_re, max_im);

    EXPECT_LT(err, 1e-12, "rel L2 too large: %.3e", err);
    EXPECT_LT(max_re, 1e-10, "max|Δre| too large: %.3e", max_re);
    EXPECT_LT(max_im, 1e-10, "max|Δim| too large: %.3e", max_im);

    r32_aligned_free(in_re); r32_aligned_free(in_im);
    r32_aligned_free(out_re); r32_aligned_free(out_im);
    r32_aligned_free(nat_re); r32_aligned_free(nat_im);
    r32_aligned_free(ref_re); r32_aligned_free(ref_im);

    TEST_END();
}

/*==========================================================================
 * TEST 2: n1 FORWARD → BACKWARD ROUND-TRIP
 *=========================================================================*/

static void test_n1_roundtrip(void)
{
    TEST_BEGIN("n1_forward_backward_roundtrip");

    const size_t stride = 4;
    const size_t total  = N * stride;

    double *in_re  = alloc_aligned(total);
    double *in_im  = alloc_aligned(total);
    double *fwd_re = alloc_aligned(total);
    double *fwd_im = alloc_aligned(total);
    double *nat_re = alloc_aligned(total);
    double *nat_im = alloc_aligned(total);
    double *bwd_re = alloc_aligned(total);
    double *bwd_im = alloc_aligned(total);
    double *res_re = alloc_aligned(total);
    double *res_im = alloc_aligned(total);

    fill_random(in_re, total);
    fill_random(in_im, total);

    /* forward → depermute → backward → depermute → compare */
    fft_radix32_n1_forward_avx2(in_re, in_im, fwd_re, fwd_im, stride, stride);

    for (int lane = 0; lane < 4; lane++)
        depermute(fwd_re, fwd_im, nat_re, nat_im, stride, lane);

    fft_radix32_n1_backward_avx2(nat_re, nat_im, bwd_re, bwd_im, stride, stride);

    for (int lane = 0; lane < 4; lane++)
        depermute(bwd_re, bwd_im, res_re, res_im, stride, lane);

    /* IDFT(DFT(x)) = N·x */
    double max_err = 0.0;
    for (size_t i = 0; i < total; i++) {
        double er = fabs(res_re[i] / (double)N - in_re[i]);
        double ei = fabs(res_im[i] / (double)N - in_im[i]);
        if (er > max_err) max_err = er;
        if (ei > max_err) max_err = ei;
    }

    printf("    max round-trip error: %.3e\n", max_err);
    EXPECT_LT(max_err, 1e-13, "round-trip error too large: %.3e", max_err);

    r32_aligned_free(in_re); r32_aligned_free(in_im);
    r32_aligned_free(fwd_re); r32_aligned_free(fwd_im);
    r32_aligned_free(nat_re); r32_aligned_free(nat_im);
    r32_aligned_free(bwd_re); r32_aligned_free(bwd_im);
    r32_aligned_free(res_re); r32_aligned_free(res_im);

    TEST_END();
}

/*==========================================================================
 * TEST 3: n1 BACKWARD VS NAIVE IDFT (correctness)
 *=========================================================================*/

static void test_n1_backward_vs_naive(void)
{
    TEST_BEGIN("n1_backward_vs_naive_idft");

    const size_t stride = 4;
    const size_t total  = N * stride;

    double *in_re  = alloc_aligned(total);
    double *in_im  = alloc_aligned(total);
    double *out_re = alloc_aligned(total);
    double *out_im = alloc_aligned(total);
    double *ref_re = alloc_aligned(total);
    double *ref_im = alloc_aligned(total);

    fill_random(in_re, total);
    fill_random(in_im, total);

    fft_radix32_n1_backward_avx2(in_re, in_im, out_re, out_im, stride, stride);
    naive_dft_backward(in_re, in_im, ref_re, ref_im, stride);

    double err = rel_l2_error(out_re, out_im, ref_re, ref_im, total);
    printf("    rel L2 error: %.3e\n", err);
    EXPECT_LT(err, 1e-12, "rel L2 too large: %.3e", err);

    r32_aligned_free(in_re); r32_aligned_free(in_im);
    r32_aligned_free(out_re); r32_aligned_free(out_im);
    r32_aligned_free(ref_re); r32_aligned_free(ref_im);

    TEST_END();
}

/*==========================================================================
 * TEST 4: n1 IMPULSE RESPONSE (known-answer)
 *
 * DFT of δ[0] = 1 for all frequencies → all outputs = 1+0j
 *=========================================================================*/

static void test_n1_impulse(void)
{
    TEST_BEGIN("n1_impulse_response");

    const size_t stride = 4;
    const size_t total  = N * stride;

    double *in_re  = alloc_aligned(total);
    double *in_im  = alloc_aligned(total);
    double *out_re = alloc_aligned(total);
    double *out_im = alloc_aligned(total);

    /* impulse: x[0] = 1, rest = 0 (all 4 lanes) */
    for (int lane = 0; lane < 4; lane++)
        in_re[0 * stride + lane] = 1.0;

    fft_radix32_n1_forward_avx2(in_re, in_im, out_re, out_im, stride, stride);

    double max_err = 0.0;
    for (int m = 0; m < N; m++) {
        for (int lane = 0; lane < 4; lane++) {
            double er = fabs(out_re[m * stride + lane] - 1.0);
            double ei = fabs(out_im[m * stride + lane] - 0.0);
            if (er > max_err) max_err = er;
            if (ei > max_err) max_err = ei;
        }
    }

    printf("    max error from expected (1+0j): %.3e\n", max_err);
    EXPECT_LT(max_err, 1e-15, "impulse response error: %.3e", max_err);

    r32_aligned_free(in_re); r32_aligned_free(in_im);
    r32_aligned_free(out_re); r32_aligned_free(out_im);

    TEST_END();
}

/*==========================================================================
 * TEST 5: n1 DC RESPONSE (known-answer)
 *
 * DFT of x[n]=1 for all n → X[0] = N, X[m≠0] = 0
 *=========================================================================*/

static void test_n1_dc(void)
{
    TEST_BEGIN("n1_dc_response");

    const size_t stride = 4;
    const size_t total  = N * stride;

    double *in_re  = alloc_aligned(total);
    double *in_im  = alloc_aligned(total);
    double *out_re = alloc_aligned(total);
    double *out_im = alloc_aligned(total);

    /* constant: x[n] = 1 for all n, all lanes */
    for (size_t i = 0; i < total; i++)
        in_re[i] = 1.0;

    fft_radix32_n1_forward_avx2(in_re, in_im, out_re, out_im, stride, stride);

    double max_err = 0.0;
    for (int m = 0; m < N; m++) {
        double expected = (m == 0) ? (double)N : 0.0;
        for (int lane = 0; lane < 4; lane++) {
            double er = fabs(out_re[m * stride + lane] - expected);
            double ei = fabs(out_im[m * stride + lane] - 0.0);
            if (er > max_err) max_err = er;
            if (ei > max_err) max_err = ei;
        }
    }

    printf("    max error from expected: %.3e\n", max_err);
    EXPECT_LT(max_err, 1e-13, "DC response error: %.3e", max_err);

    r32_aligned_free(in_re); r32_aligned_free(in_im);
    r32_aligned_free(out_re); r32_aligned_free(out_im);

    TEST_END();
}

/*==========================================================================
 * TEST 6: n1 SINGLE-BIN SINUSOID (known-answer)
 *
 * x[n] = exp(-2πi·f·n/N)  →  X[f] = N, X[m≠f] = 0
 *=========================================================================*/

static void test_n1_single_bin(void)
{
    TEST_BEGIN("n1_single_bin_sinusoid");

    const size_t stride = 4;
    const size_t total  = N * stride;
    const int freq = 7; /* arbitrary bin */

    double *in_re  = alloc_aligned(total);
    double *in_im  = alloc_aligned(total);
    double *out_re = alloc_aligned(total);
    double *out_im = alloc_aligned(total);

    for (int n = 0; n < N; n++) {
        double angle = -2.0 * M_PI * (double)freq * (double)n / (double)N;
        for (int lane = 0; lane < 4; lane++) {
            in_re[n * stride + lane] = cos(angle);
            in_im[n * stride + lane] = sin(angle);
        }
    }

    fft_radix32_n1_forward_avx2(in_re, in_im, out_re, out_im, stride, stride);

    double max_err = 0.0;
    for (int m = 0; m < N; m++) {
        double exp_re = (m == freq) ? (double)N : 0.0;
        double exp_im = 0.0;
        for (int lane = 0; lane < 4; lane++) {
            double er = fabs(out_re[m * stride + lane] - exp_re);
            double ei = fabs(out_im[m * stride + lane] - exp_im);
            if (er > max_err) max_err = er;
            if (ei > max_err) max_err = ei;
        }
    }

    printf("    max error (freq=%d): %.3e\n", freq, max_err);
    EXPECT_LT(max_err, 1e-12, "single-bin error: %.3e", max_err);

    r32_aligned_free(in_re); r32_aligned_free(in_im);
    r32_aligned_free(out_re); r32_aligned_free(out_im);

    TEST_END();
}

/*==========================================================================
 * TEST 7: TWIDDLE VERSION ROUND-TRIP (K=8, BLOCKED8)
 *
 * Generates correct Cooley-Tukey twiddle factors for a standalone
 * N=32K-point DFT, then verifies forward→backward round-trip.
 *=========================================================================*/

/**
 * @brief Generate pass-1 DIT twiddles (BLOCKED2) for standalone 32K-point DFT
 *
 * W1[k] = exp(-2πi·k/(4K))   (applied to input b=1)
 * W2[k] = exp(-2πi·2k/(4K))  (applied to input b=2)
 * W3 = W1·W2                  (derived on the fly)
 */
static void gen_pass1_twiddles(size_t K, double *tw_re, double *tw_im)
{
    /* DIT-4 stage twiddle for input m, sample k: W_{32K}^{m·k}
     * BLOCKED2: tw_re[0*K+k] = W1, tw_re[1*K+k] = W2 */
    for (size_t k = 0; k < K; k++) {
        double ang1 = -2.0 * M_PI * (double)k / (32.0 * (double)K);
        tw_re[0 * K + k] = cos(ang1);       /* W1 */
        tw_im[0 * K + k] = sin(ang1);
        tw_re[1 * K + k] = cos(2.0 * ang1); /* W2 */
        tw_im[1 * K + k] = sin(2.0 * ang1);
    }
}

/**
 * @brief Generate pass-2 DIF twiddles (BLOCKED8) for standalone 32K-point DFT
 *
 * The DIF-8 stage twiddle for group g (1..7) at sample k:
 *   W[g][k] = exp(-2πi·g·k/(8K))
 *
 * Note: these are the k-dependent Cooley-Tukey twist factors.
 * The bin-dependent W_32^{g·b} factor cancels in round-trip.
 */
static void gen_pass2_twiddles(size_t K, double **tw_re, double **tw_im)
{
    for (int g = 0; g < 8; g++) {
        for (size_t k = 0; k < K; k++) {
            double angle = -2.0 * M_PI * (double)(g + 1) * (double)k
                         / (8.0 * (double)K);
            tw_re[g][k] = cos(angle);
            tw_im[g][k] = sin(angle);
        }
    }
}

static void test_twiddle_roundtrip(void)
{
    TEST_BEGIN("twiddle_forward_backward_roundtrip_K8");

    const size_t K     = 8;
    const size_t total = 32 * K; /* 32 stripes × K samples */

    double *in_re   = alloc_aligned(total);
    double *in_im   = alloc_aligned(total);
    double *mid_re  = alloc_aligned(total);
    double *mid_im  = alloc_aligned(total);
    double *out_re  = alloc_aligned(total);
    double *out_im  = alloc_aligned(total);
    double *temp_re = alloc_aligned(total);
    double *temp_im = alloc_aligned(total);

    fill_random(in_re, total);
    fill_random(in_im, total);

    /* --- Pass-1 twiddles (BLOCKED2) --- */
    double *p1_tw_re = alloc_aligned(2 * K);
    double *p1_tw_im = alloc_aligned(2 * K);
    gen_pass1_twiddles(K, p1_tw_re, p1_tw_im);

    radix4_dit_stage_twiddles_blocked2_t p1_tw = {
        .re = p1_tw_re, .im = p1_tw_im, .K = K
    };

    /* --- Pass-2 twiddles (BLOCKED8) --- */
    double *p2_re_arr[8], *p2_im_arr[8];
    for (int j = 0; j < 8; j++) {
        p2_re_arr[j] = alloc_aligned(K);
        p2_im_arr[j] = alloc_aligned(K);
    }
    gen_pass2_twiddles(K, p2_re_arr, p2_im_arr);

    tw_stage8_t p2_tw;
    p2_tw.mode = TW_MODE_BLOCKED8;
    for (int j = 0; j < 8; j++) {
        p2_tw.b8.re[j] = p2_re_arr[j];
        p2_tw.b8.im[j] = p2_im_arr[j];
    }
    p2_tw.b8.K = K;

    /* forward */
    radix32_stage_forward_avx2(
        K, in_re, in_im, mid_re, mid_im,
        &p1_tw, &p2_tw, temp_re, temp_im);

    /* backward */
    radix32_stage_backward_avx2(
        K, mid_re, mid_im, out_re, out_im,
        &p1_tw, &p2_tw, temp_re, temp_im);

    /* round-trip: divide by N = 32K, NOT just 32.
     * Actually, each sample k sees the full 32-point butterfly twice
     * (forward + backward), yielding factor 32 per sample. The k-dependent
     * twiddles are unit-modulus and cancel. So factor = 32. */
    const double inv_N = 1.0 / 32.0;
    double max_err = 0.0;
    for (size_t i = 0; i < total; i++) {
        double er = fabs(out_re[i] * inv_N - in_re[i]);
        double ei = fabs(out_im[i] * inv_N - in_im[i]);
        if (er > max_err) max_err = er;
        if (ei > max_err) max_err = ei;
    }

    printf("    max round-trip error (K=%zu): %.3e\n", K, max_err);
    EXPECT_LT(max_err, 1e-12, "round-trip error too large: %.3e", max_err);

    /* cleanup */
    r32_aligned_free(in_re); r32_aligned_free(in_im);
    r32_aligned_free(mid_re); r32_aligned_free(mid_im);
    r32_aligned_free(out_re); r32_aligned_free(out_im);
    r32_aligned_free(temp_re); r32_aligned_free(temp_im);
    r32_aligned_free(p1_tw_re); r32_aligned_free(p1_tw_im);
    for (int j = 0; j < 8; j++) {
        r32_aligned_free(p2_re_arr[j]); r32_aligned_free(p2_im_arr[j]);
    }

    TEST_END();
}

/*==========================================================================
 * TEST 8: TWIDDLE VERSION ROUND-TRIP (K=64, BLOCKED8)
 *=========================================================================*/

static void test_twiddle_roundtrip_k64(void)
{
    TEST_BEGIN("twiddle_forward_backward_roundtrip_K64");

    const size_t K     = 64;
    const size_t total = 32 * K;

    double *in_re   = alloc_aligned(total);
    double *in_im   = alloc_aligned(total);
    double *mid_re  = alloc_aligned(total);
    double *mid_im  = alloc_aligned(total);
    double *out_re  = alloc_aligned(total);
    double *out_im  = alloc_aligned(total);
    double *temp_re = alloc_aligned(total);
    double *temp_im = alloc_aligned(total);

    fill_random(in_re, total);
    fill_random(in_im, total);

    double *p1_tw_re = alloc_aligned(2 * K);
    double *p1_tw_im = alloc_aligned(2 * K);
    gen_pass1_twiddles(K, p1_tw_re, p1_tw_im);
    radix4_dit_stage_twiddles_blocked2_t p1_tw = {
        .re = p1_tw_re, .im = p1_tw_im, .K = K
    };

    double *p2_re_arr[8], *p2_im_arr[8];
    for (int j = 0; j < 8; j++) {
        p2_re_arr[j] = alloc_aligned(K);
        p2_im_arr[j] = alloc_aligned(K);
    }
    gen_pass2_twiddles(K, p2_re_arr, p2_im_arr);

    tw_stage8_t p2_tw;
    p2_tw.mode = TW_MODE_BLOCKED8;
    for (int j = 0; j < 8; j++) {
        p2_tw.b8.re[j] = p2_re_arr[j];
        p2_tw.b8.im[j] = p2_im_arr[j];
    }
    p2_tw.b8.K = K;

    radix32_stage_forward_avx2(
        K, in_re, in_im, mid_re, mid_im,
        &p1_tw, &p2_tw, temp_re, temp_im);

    radix32_stage_backward_avx2(
        K, mid_re, mid_im, out_re, out_im,
        &p1_tw, &p2_tw, temp_re, temp_im);

    const double inv_N = 1.0 / 32.0;
    double max_err = 0.0;
    for (size_t i = 0; i < total; i++) {
        double er = fabs(out_re[i] * inv_N - in_re[i]);
        double ei = fabs(out_im[i] * inv_N - in_im[i]);
        if (er > max_err) max_err = er;
        if (ei > max_err) max_err = ei;
    }

    printf("    max round-trip error (K=%zu): %.3e\n", K, max_err);
    EXPECT_LT(max_err, 1e-12, "round-trip error too large: %.3e", max_err);

    r32_aligned_free(in_re); r32_aligned_free(in_im);
    r32_aligned_free(mid_re); r32_aligned_free(mid_im);
    r32_aligned_free(out_re); r32_aligned_free(out_im);
    r32_aligned_free(temp_re); r32_aligned_free(temp_im);
    r32_aligned_free(p1_tw_re); r32_aligned_free(p1_tw_im);
    for (int j = 0; j < 8; j++) {
        r32_aligned_free(p2_re_arr[j]); r32_aligned_free(p2_im_arr[j]);
    }

    TEST_END();
}

/*==========================================================================
 * TEST 9: n1 LINEARITY — DFT(a·x + b·y) = a·DFT(x) + b·DFT(y)
 *=========================================================================*/

static void test_n1_linearity(void)
{
    TEST_BEGIN("n1_linearity");

    const size_t stride = 4;
    const size_t total  = N * stride;
    const double a = 2.71828, b = -1.41421;

    double *x_re = alloc_aligned(total), *x_im = alloc_aligned(total);
    double *y_re = alloc_aligned(total), *y_im = alloc_aligned(total);
    double *z_re = alloc_aligned(total), *z_im = alloc_aligned(total);

    double *Fx_re = alloc_aligned(total), *Fx_im = alloc_aligned(total);
    double *Fy_re = alloc_aligned(total), *Fy_im = alloc_aligned(total);
    double *Fz_re = alloc_aligned(total), *Fz_im = alloc_aligned(total);

    fill_random(x_re, total); fill_random(x_im, total);
    fill_random(y_re, total); fill_random(y_im, total);

    /* z = a·x + b·y */
    for (size_t i = 0; i < total; i++) {
        z_re[i] = a * x_re[i] + b * y_re[i];
        z_im[i] = a * x_im[i] + b * y_im[i];
    }

    fft_radix32_n1_forward_avx2(x_re, x_im, Fx_re, Fx_im, stride, stride);
    fft_radix32_n1_forward_avx2(y_re, y_im, Fy_re, Fy_im, stride, stride);
    fft_radix32_n1_forward_avx2(z_re, z_im, Fz_re, Fz_im, stride, stride);

    /* Fz should equal a·Fx + b·Fy */
    double max_err = 0.0;
    for (size_t i = 0; i < total; i++) {
        double exp_re = a * Fx_re[i] + b * Fy_re[i];
        double exp_im = a * Fx_im[i] + b * Fy_im[i];
        double er = fabs(Fz_re[i] - exp_re);
        double ei = fabs(Fz_im[i] - exp_im);
        if (er > max_err) max_err = er;
        if (ei > max_err) max_err = ei;
    }

    printf("    max linearity error: %.3e\n", max_err);
    EXPECT_LT(max_err, 1e-12, "linearity error: %.3e", max_err);

    r32_aligned_free(x_re); r32_aligned_free(x_im); r32_aligned_free(y_re); r32_aligned_free(y_im); r32_aligned_free(z_re); r32_aligned_free(z_im);
    r32_aligned_free(Fx_re); r32_aligned_free(Fx_im); r32_aligned_free(Fy_re); r32_aligned_free(Fy_im); r32_aligned_free(Fz_re); r32_aligned_free(Fz_im);

    TEST_END();
}

/*==========================================================================
 * TEST 10: n1 PARSEVAL'S THEOREM — Σ|x|² = (1/N)·Σ|X|²
 *=========================================================================*/

static void test_n1_parseval(void)
{
    TEST_BEGIN("n1_parseval_theorem");

    const size_t stride = 4;
    const size_t total  = N * stride;

    double *in_re  = alloc_aligned(total);
    double *in_im  = alloc_aligned(total);
    double *out_re = alloc_aligned(total);
    double *out_im = alloc_aligned(total);

    fill_random(in_re, total);
    fill_random(in_im, total);

    fft_radix32_n1_forward_avx2(in_re, in_im, out_re, out_im, stride, stride);

    /* check per-lane */
    double max_rel_err = 0.0;
    for (int lane = 0; lane < 4; lane++) {
        double energy_time = 0.0, energy_freq = 0.0;
        for (int n = 0; n < N; n++) {
            double xr = in_re[n * stride + lane];
            double xi = in_im[n * stride + lane];
            energy_time += xr * xr + xi * xi;
            double Xr = out_re[n * stride + lane];
            double Xi = out_im[n * stride + lane];
            energy_freq += Xr * Xr + Xi * Xi;
        }
        energy_freq /= (double)N;
        double rel = fabs(energy_time - energy_freq) /
                     (energy_time > 0 ? energy_time : 1.0);
        if (rel > max_rel_err) max_rel_err = rel;
    }

    printf("    max Parseval relative error: %.3e\n", max_rel_err);
    EXPECT_LT(max_rel_err, 1e-13, "Parseval error: %.3e", max_rel_err);

    r32_aligned_free(in_re); r32_aligned_free(in_im);
    r32_aligned_free(out_re); r32_aligned_free(out_im);

    TEST_END();
}

/*==========================================================================
 * MAIN
 *=========================================================================*/

int main(void)
{
    printf("====== radix-32 AVX2 unit tests ======\n\n");
    printf("--- n1 codelet ---\n");
    test_n1_forward_vs_naive();
    test_n1_backward_vs_naive();
    test_n1_roundtrip();
    test_n1_impulse();
    test_n1_dc();
    test_n1_single_bin();
    test_n1_linearity();
    test_n1_parseval();

    printf("\n--- twiddle version ---\n");
    test_twiddle_roundtrip();
    test_twiddle_roundtrip_k64();

    printf("\n====== results: %d/%d passed",
           g_tests_passed, g_tests_run);
    if (g_tests_failed > 0)
        printf(", %d FAILED", g_tests_failed);
    printf(" ======\n");

    return g_tests_failed > 0 ? 1 : 0;
}
