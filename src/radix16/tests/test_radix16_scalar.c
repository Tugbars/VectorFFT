/**
 * @file test_radix16_scalar.c
 * @brief Comprehensive tests for radix-16 scalar twiddle-less butterfly
 *
 * Tests:
 *   1. Forward butterfly vs naive DFT-16 reference
 *   2. Backward butterfly vs naive IDFT-16 reference
 *   3. Forward → Backward roundtrip (identity within 1/N scaling)
 *   4. Known input patterns (impulse, DC, alternating, single-frequency)
 *   5. Edge cases (K=1, K=2 tail, large K)
 *   6. In-place operation (src == dst)
 *   7. W₄ twiddle unit tests (axis-aligned root verification)
 *
 * Build:
 *   gcc -O2 -std=c11 -lm test_radix16_scalar.c -o test_radix16_scalar
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

#include "fft_radix16_scalar_butterfly.h"

#define R16  16
#define PI   3.14159265358979323846

/* ========================================================================= */
/* TEST INFRASTRUCTURE                                                        */
/* ========================================================================= */

static int g_tests_run    = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_BEGIN(name) \
    do { \
        g_tests_run++; \
        const char *_test_name = (name); \
        int _test_ok = 1; \
        (void)_test_ok;

#define TEST_END() \
        if (_test_ok) { \
            g_tests_passed++; \
            printf("  [PASS] %s\n", _test_name); \
        } else { \
            g_tests_failed++; \
            printf("  [FAIL] %s\n", _test_name); \
        } \
    } while (0)

#define CHECK_TOL(val, ref, tol, msg) \
    do { \
        double _err = fabs((val) - (ref)); \
        if (_err > (tol)) { \
            printf("    MISMATCH %s: got %.17e, expected %.17e, err=%.3e (tol=%.3e)\n", \
                   (msg), (val), (ref), _err, (double)(tol)); \
            _test_ok = 0; \
        } \
    } while (0)

/* ========================================================================= */
/* NAIVE DFT-16 REFERENCE                                                     */
/* ========================================================================= */

/**
 * Naive DFT-16 for one column k:
 *   Y[m] = sum_{n=0}^{15} X[n] * W_{16}^{mn}
 *   Forward: W = exp(-2πi/16), Backward: W = exp(+2πi/16)
 */
static void naive_dft16(
    const double *in_re, const double *in_im, size_t K, size_t k,
    double *out_re, double *out_im, int forward)
{
    double sign = forward ? -1.0 : 1.0;

    for (int m = 0; m < R16; m++)
    {
        double sum_re = 0.0, sum_im = 0.0;
        for (int n = 0; n < R16; n++)
        {
            double angle = sign * 2.0 * PI * (double)(m * n) / (double)R16;
            double w_re = cos(angle);
            double w_im = sin(angle);

            double x_re = in_re[n * K + k];
            double x_im = in_im[n * K + k];

            sum_re += x_re * w_re - x_im * w_im;
            sum_im += x_re * w_im + x_im * w_re;
        }
        out_re[m] = sum_re;
        out_im[m] = sum_im;
    }
}

/* ========================================================================= */
/* HELPER: Fill with random data in [-1, 1]                                   */
/* ========================================================================= */

static void fill_random(double *buf, size_t n, unsigned int *seed)
{
    for (size_t i = 0; i < n; i++)
    {
        *seed = *seed * 1103515245u + 12345u;
        buf[i] = ((double)(*seed & 0x7fffffffu) / (double)0x7fffffffu) * 2.0 - 1.0;
    }
}

/* ========================================================================= */
/* HELPER: Max absolute error across all elements                             */
/* ========================================================================= */

static double max_abs_error(const double *a, const double *b, size_t n)
{
    double mx = 0.0;
    for (size_t i = 0; i < n; i++)
    {
        double e = fabs(a[i] - b[i]);
        if (e > mx) mx = e;
    }
    return mx;
}

/* ========================================================================= */
/* TEST 1: Forward butterfly vs naive DFT-16                                  */
/* ========================================================================= */

static void test_forward_vs_naive(size_t K)
{
    char name[128];
    snprintf(name, sizeof(name), "Forward vs naive DFT-16 (K=%zu)", K);

    TEST_BEGIN(name);

    double *in_re  = calloc(R16 * K, sizeof(double));
    double *in_im  = calloc(R16 * K, sizeof(double));
    double *out_re = calloc(R16 * K, sizeof(double));
    double *out_im = calloc(R16 * K, sizeof(double));

    unsigned int seed = 42u + (unsigned int)K;
    fill_random(in_re, R16 * K, &seed);
    fill_random(in_im, R16 * K, &seed);

    radix16_butterfly_forward_scalar(K, in_re, in_im, out_re, out_im);

    double ref_re[R16], ref_im[R16];
    double max_err = 0.0;

    for (size_t k = 0; k < K; k++)
    {
        naive_dft16(in_re, in_im, K, k, ref_re, ref_im, 1);

        for (int m = 0; m < R16; m++)
        {
            double err_re = fabs(out_re[m * K + k] - ref_re[m]);
            double err_im = fabs(out_im[m * K + k] - ref_im[m]);
            if (err_re > max_err) max_err = err_re;
            if (err_im > max_err) max_err = err_im;
        }
    }

    double tol = 5.0e-14;
    if (max_err > tol)
    {
        printf("    max error = %.3e (tol = %.3e)\n", max_err, tol);
        _test_ok = 0;
    }

    free(in_re); free(in_im); free(out_re); free(out_im);

    TEST_END();
}

/* ========================================================================= */
/* TEST 2: Backward butterfly vs naive IDFT-16                                */
/* ========================================================================= */

static void test_backward_vs_naive(size_t K)
{
    char name[128];
    snprintf(name, sizeof(name), "Backward vs naive IDFT-16 (K=%zu)", K);

    TEST_BEGIN(name);

    double *in_re  = calloc(R16 * K, sizeof(double));
    double *in_im  = calloc(R16 * K, sizeof(double));
    double *out_re = calloc(R16 * K, sizeof(double));
    double *out_im = calloc(R16 * K, sizeof(double));

    unsigned int seed = 137u + (unsigned int)K;
    fill_random(in_re, R16 * K, &seed);
    fill_random(in_im, R16 * K, &seed);

    radix16_butterfly_backward_scalar(K, in_re, in_im, out_re, out_im);

    double ref_re[R16], ref_im[R16];
    double max_err = 0.0;

    for (size_t k = 0; k < K; k++)
    {
        naive_dft16(in_re, in_im, K, k, ref_re, ref_im, 0);

        for (int m = 0; m < R16; m++)
        {
            double err_re = fabs(out_re[m * K + k] - ref_re[m]);
            double err_im = fabs(out_im[m * K + k] - ref_im[m]);
            if (err_re > max_err) max_err = err_re;
            if (err_im > max_err) max_err = err_im;
        }
    }

    double tol = 5.0e-14;
    if (max_err > tol)
    {
        printf("    max error = %.3e (tol = %.3e)\n", max_err, tol);
        _test_ok = 0;
    }

    free(in_re); free(in_im); free(out_re); free(out_im);

    TEST_END();
}

/* ========================================================================= */
/* TEST 3: Forward → Backward roundtrip                                       */
/* ========================================================================= */

static void test_roundtrip(size_t K)
{
    char name[128];
    snprintf(name, sizeof(name), "Roundtrip fwd→bwd (K=%zu)", K);

    TEST_BEGIN(name);

    double *in_re   = calloc(R16 * K, sizeof(double));
    double *in_im   = calloc(R16 * K, sizeof(double));
    double *mid_re  = calloc(R16 * K, sizeof(double));
    double *mid_im  = calloc(R16 * K, sizeof(double));
    double *out_re  = calloc(R16 * K, sizeof(double));
    double *out_im  = calloc(R16 * K, sizeof(double));

    unsigned int seed = 271828u + (unsigned int)K;
    fill_random(in_re, R16 * K, &seed);
    fill_random(in_im, R16 * K, &seed);

    radix16_butterfly_forward_scalar(K, in_re, in_im, mid_re, mid_im);
    radix16_butterfly_backward_scalar(K, mid_re, mid_im, out_re, out_im);

    /* Should get back 16 * input (DFT followed by IDFT = N * identity) */
    double max_err = 0.0;
    for (size_t i = 0; i < R16 * K; i++)
    {
        double err_re = fabs(out_re[i] - 16.0 * in_re[i]);
        double err_im = fabs(out_im[i] - 16.0 * in_im[i]);
        if (err_re > max_err) max_err = err_re;
        if (err_im > max_err) max_err = err_im;
    }

    double tol = 5.0e-14;
    if (max_err > tol)
    {
        printf("    max roundtrip error = %.3e (tol = %.3e)\n", max_err, tol);
        _test_ok = 0;
    }

    free(in_re); free(in_im); free(mid_re); free(mid_im);
    free(out_re); free(out_im);

    TEST_END();
}

/* ========================================================================= */
/* TEST 4: Known input patterns                                               */
/* ========================================================================= */

static void test_impulse(void)
{
    TEST_BEGIN("Impulse response (delta at n=0)");

    size_t K = 1;
    double in_re[R16] = {0}, in_im[R16] = {0};
    double out_re[R16], out_im[R16];

    in_re[0] = 1.0; /* delta at row 0 */

    radix16_butterfly_forward_scalar(K, in_re, in_im, out_re, out_im);

    /* DFT of impulse = all ones */
    for (int m = 0; m < R16; m++)
    {
        CHECK_TOL(out_re[m], 1.0, 1e-15, "re");
        CHECK_TOL(out_im[m], 0.0, 1e-15, "im");
    }

    TEST_END();
}

static void test_dc(void)
{
    TEST_BEGIN("DC input (all ones)");

    size_t K = 1;
    double in_re[R16], in_im[R16] = {0};
    double out_re[R16], out_im[R16];

    for (int i = 0; i < R16; i++) in_re[i] = 1.0;

    radix16_butterfly_forward_scalar(K, in_re, in_im, out_re, out_im);

    /* DFT of all-ones: Y[0]=16, Y[m]=0 for m>0 */
    CHECK_TOL(out_re[0], 16.0, 1e-14, "Y[0] re");
    CHECK_TOL(out_im[0], 0.0,  1e-14, "Y[0] im");
    for (int m = 1; m < R16; m++)
    {
        CHECK_TOL(out_re[m], 0.0, 1e-14, "Y[m] re");
        CHECK_TOL(out_im[m], 0.0, 1e-14, "Y[m] im");
    }

    TEST_END();
}

static void test_single_frequency(void)
{
    TEST_BEGIN("Single frequency (bin 3)");

    size_t K = 1;
    double in_re[R16], in_im[R16];
    double out_re[R16], out_im[R16];

    /* Input = exp(+2πi·3·n/16) for n=0..15 → forward DFT peak at bin 3 */
    for (int n = 0; n < R16; n++)
    {
        double angle = 2.0 * PI * 3.0 * n / 16.0;
        in_re[n] = cos(angle);
        in_im[n] = sin(angle);
    }

    radix16_butterfly_forward_scalar(K, in_re, in_im, out_re, out_im);

    /* Expect Y[3] = 16, all others = 0 */
    for (int m = 0; m < R16; m++)
    {
        double exp_re = (m == 3) ? 16.0 : 0.0;
        double exp_im = 0.0;
        CHECK_TOL(out_re[m], exp_re, 2e-13, "re");
        CHECK_TOL(out_im[m], exp_im, 2e-13, "im");
    }

    TEST_END();
}

static void test_alternating(void)
{
    TEST_BEGIN("Alternating (+1,-1) → bin 8");

    size_t K = 1;
    double in_re[R16], in_im[R16] = {0};
    double out_re[R16], out_im[R16];

    for (int n = 0; n < R16; n++)
        in_re[n] = (n % 2 == 0) ? 1.0 : -1.0;

    radix16_butterfly_forward_scalar(K, in_re, in_im, out_re, out_im);

    /* Alternating = exp(-πi·n) = exp(-2πi·8·n/16) → DFT peak at bin 8 */
    for (int m = 0; m < R16; m++)
    {
        double exp_re = (m == 8) ? 16.0 : 0.0;
        CHECK_TOL(out_re[m], exp_re, 1e-14, "re");
        CHECK_TOL(out_im[m], 0.0,    1e-14, "im");
    }

    TEST_END();
}

/* ========================================================================= */
/* TEST 5: W₁₆ twiddle unit tests                                            */
/* ========================================================================= */

static void test_w16_twiddle_fwd_bwd_roundtrip(void)
{
    TEST_BEGIN("W16 twiddle fwd→bwd roundtrip = identity");

    double T_re[4][4], T_im[4][4];
    double S_re[4][4], S_im[4][4];

    unsigned int seed = 12345u;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            fill_random(&T_re[i][j], 1, &seed);
            fill_random(&T_im[i][j], 1, &seed);
            S_re[i][j] = T_re[i][j];
            S_im[i][j] = T_im[i][j];
        }

    r16s_twiddle_fwd(T_re, T_im);
    r16s_twiddle_bwd(T_re, T_im);

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            char msg[32];
            snprintf(msg, sizeof(msg), "T[%d][%d] re", i, j);
            CHECK_TOL(T_re[i][j], S_re[i][j], 1e-15, msg);
            snprintf(msg, sizeof(msg), "T[%d][%d] im", i, j);
            CHECK_TOL(T_im[i][j], S_im[i][j], 1e-15, msg);
        }

    TEST_END();
}

static void test_w16_twiddle_trivial_entries(void)
{
    TEST_BEGIN("W16 twiddle: trivial entries preserved");

    /* Set up T with known values, check that n2=0 row and k1=0 col are unchanged */
    double T_re[4][4], T_im[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            T_re[i][j] = (double)(i * 4 + j + 1);
            T_im[i][j] = (double)(i * 4 + j + 17);
        }

    double S_re[4][4], S_im[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            S_re[i][j] = T_re[i][j];
            S_im[i][j] = T_im[i][j];
        }

    r16s_twiddle_fwd(T_re, T_im);

    /* n2=0 row: all identity */
    for (int j = 0; j < 4; j++) {
        CHECK_TOL(T_re[0][j], S_re[0][j], 0.0, "n2=0 re");
        CHECK_TOL(T_im[0][j], S_im[0][j], 0.0, "n2=0 im");
    }
    /* k1=0 column: all identity */
    for (int i = 0; i < 4; i++) {
        CHECK_TOL(T_re[i][0], S_re[i][0], 0.0, "k1=0 re");
        CHECK_TOL(T_im[i][0], S_im[i][0], 0.0, "k1=0 im");
    }

    /* T[2][2] *= W^4 = -j: (re,im) → (im, -re) */
    CHECK_TOL(T_re[2][2], S_im[2][2], 1e-15, "W4 re");
    CHECK_TOL(T_im[2][2], -S_re[2][2], 1e-15, "W4 im");

    TEST_END();
}

static void test_w16_vs_naive(void)
{
    TEST_BEGIN("W16 twiddle fwd vs naive complex multiply");

    double T_re[4][4], T_im[4][4];
    unsigned int seed = 9999u;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            fill_random(&T_re[i][j], 1, &seed);
            fill_random(&T_im[i][j], 1, &seed);
        }

    /* Compute expected via naive: T[n2][k1] *= exp(-2πi·n2·k1/16) */
    double E_re[4][4], E_im[4][4];
    for (int n2 = 0; n2 < 4; n2++)
        for (int k1 = 0; k1 < 4; k1++) {
            int exp_val = (n2 * k1) % 16;
            double angle = -2.0 * PI * exp_val / 16.0;
            double wr = cos(angle), wi = sin(angle);
            double a = T_re[n2][k1], b = T_im[n2][k1];
            E_re[n2][k1] = a * wr - b * wi;
            E_im[n2][k1] = a * wi + b * wr;
        }

    r16s_twiddle_fwd(T_re, T_im);

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            char msg[32];
            snprintf(msg, sizeof(msg), "T[%d][%d] re", i, j);
            CHECK_TOL(T_re[i][j], E_re[i][j], 2e-15, msg);
            snprintf(msg, sizeof(msg), "T[%d][%d] im", i, j);
            CHECK_TOL(T_im[i][j], E_im[i][j], 2e-15, msg);
        }

    TEST_END();
}

/* ========================================================================= */
/* TEST 6: Edge cases                                                         */
/* ========================================================================= */

/* K=1 is covered by test_various_K */

static void test_inplace(void)
{
    TEST_BEGIN("In-place operation (src == dst)");

    size_t K = 4;
    double *buf_re = calloc(R16 * K, sizeof(double));
    double *buf_im = calloc(R16 * K, sizeof(double));
    double *ref_re = calloc(R16 * K, sizeof(double));
    double *ref_im = calloc(R16 * K, sizeof(double));

    unsigned int seed = 31415u;
    fill_random(buf_re, R16 * K, &seed);
    fill_random(buf_im, R16 * K, &seed);
    memcpy(ref_re, buf_re, R16 * K * sizeof(double));
    memcpy(ref_im, buf_im, R16 * K * sizeof(double));

    /* Out-of-place reference */
    double *oop_re = calloc(R16 * K, sizeof(double));
    double *oop_im = calloc(R16 * K, sizeof(double));
    radix16_butterfly_forward_scalar(K, ref_re, ref_im, oop_re, oop_im);

    /* In-place */
    radix16_butterfly_forward_scalar(K, buf_re, buf_im, buf_re, buf_im);

    double err_re = max_abs_error(buf_re, oop_re, R16 * K);
    double err_im = max_abs_error(buf_im, oop_im, R16 * K);
    double max_err = (err_re > err_im) ? err_re : err_im;

    if (max_err > 1e-15)
    {
        printf("    in-place vs out-of-place error = %.3e\n", max_err);
        _test_ok = 0;
    }

    free(buf_re); free(buf_im); free(ref_re); free(ref_im);
    free(oop_re); free(oop_im);

    TEST_END();
}

static void test_pure_real(void)
{
    TEST_BEGIN("Pure real input");

    size_t K = 3;
    double *in_re  = calloc(R16 * K, sizeof(double));
    double *in_im  = calloc(R16 * K, sizeof(double)); /* zeros */
    double *out_re = calloc(R16 * K, sizeof(double));
    double *out_im = calloc(R16 * K, sizeof(double));

    unsigned int seed = 99u;
    fill_random(in_re, R16 * K, &seed);

    radix16_butterfly_forward_scalar(K, in_re, in_im, out_re, out_im);

    /* Verify conjugate symmetry: Y[m] = conj(Y[16-m]) for pure real input */
    double max_err = 0.0;
    for (size_t k = 0; k < K; k++)
    {
        for (int m = 1; m < 8; m++)
        {
            int m_conj = 16 - m;
            double err_re = fabs(out_re[m * K + k] - out_re[m_conj * K + k]);
            double err_im = fabs(out_im[m * K + k] + out_im[m_conj * K + k]);
            if (err_re > max_err) max_err = err_re;
            if (err_im > max_err) max_err = err_im;
        }
        /* Y[0] and Y[8] should be purely real */
        double err0 = fabs(out_im[0 * K + k]);
        double err8 = fabs(out_im[8 * K + k]);
        if (err0 > max_err) max_err = err0;
        if (err8 > max_err) max_err = err8;
    }

    if (max_err > 2e-14)
    {
        printf("    conjugate symmetry error = %.3e\n", max_err);
        _test_ok = 0;
    }

    free(in_re); free(in_im); free(out_re); free(out_im);

    TEST_END();
}

static void test_pure_imaginary(void)
{
    TEST_BEGIN("Pure imaginary input");

    size_t K = 2;
    double *in_re  = calloc(R16 * K, sizeof(double)); /* zeros */
    double *in_im  = calloc(R16 * K, sizeof(double));
    double *out_re = calloc(R16 * K, sizeof(double));
    double *out_im = calloc(R16 * K, sizeof(double));

    unsigned int seed = 777u;
    fill_random(in_im, R16 * K, &seed);

    radix16_butterfly_forward_scalar(K, in_re, in_im, out_re, out_im);

    /* Pure imaginary: Y[m]_re = -Y[16-m]_re, Y[m]_im = Y[16-m]_im */
    double max_err = 0.0;
    for (size_t k = 0; k < K; k++)
    {
        for (int m = 1; m < 8; m++)
        {
            int m_conj = 16 - m;
            double err_re = fabs(out_re[m * K + k] + out_re[m_conj * K + k]);
            double err_im = fabs(out_im[m * K + k] - out_im[m_conj * K + k]);
            if (err_re > max_err) max_err = err_re;
            if (err_im > max_err) max_err = err_im;
        }
        /* Y[0] and Y[8] should be purely imaginary */
        double err0 = fabs(out_re[0 * K + k]);
        double err8 = fabs(out_re[8 * K + k]);
        if (err0 > max_err) max_err = err0;
        if (err8 > max_err) max_err = err8;
    }

    if (max_err > 2e-14)
    {
        printf("    anti-conjugate symmetry error = %.3e\n", max_err);
        _test_ok = 0;
    }

    free(in_re); free(in_im); free(out_re); free(out_im);

    TEST_END();
}

/* ========================================================================= */
/* TEST 7: Parseval's theorem (energy conservation)                           */
/* ========================================================================= */

static void test_parseval(void)
{
    TEST_BEGIN("Parseval's theorem (energy conservation)");

    size_t K = 5;
    double *in_re  = calloc(R16 * K, sizeof(double));
    double *in_im  = calloc(R16 * K, sizeof(double));
    double *out_re = calloc(R16 * K, sizeof(double));
    double *out_im = calloc(R16 * K, sizeof(double));

    unsigned int seed = 54321u;
    fill_random(in_re, R16 * K, &seed);
    fill_random(in_im, R16 * K, &seed);

    radix16_butterfly_forward_scalar(K, in_re, in_im, out_re, out_im);

    /* For each column: sum|X[n]|² * N = sum|Y[m]|² */
    for (size_t k = 0; k < K; k++)
    {
        double energy_in = 0.0, energy_out = 0.0;
        for (int m = 0; m < R16; m++)
        {
            double xr = in_re[m * K + k],  xi = in_im[m * K + k];
            double yr = out_re[m * K + k], yi = out_im[m * K + k];
            energy_in  += xr * xr + xi * xi;
            energy_out += yr * yr + yi * yi;
        }
        energy_in *= 16.0; /* N * sum|x|² = sum|Y|² */

        double rel_err = fabs(energy_in - energy_out) / energy_in;
        if (rel_err > 1e-13)
        {
            printf("    k=%zu: energy_in=%.6e, energy_out=%.6e, rel_err=%.3e\n",
                   k, energy_in, energy_out, rel_err);
            _test_ok = 0;
        }
    }

    free(in_re); free(in_im); free(out_re); free(out_im);

    TEST_END();
}

/* ========================================================================= */
/* TEST 8: Multiple K values (exercises U=2 main loop + U=1 tail)             */
/* ========================================================================= */

static void test_various_K(void)
{
    /* K=1: tail only, K=2: one U=2 iteration, K=3: one U=2 + tail,
       K=7: three U=2 + tail, K=16: eight U=2, K=64: larger stress */
    size_t Ks[] = {1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 64, 100};
    size_t nK = sizeof(Ks) / sizeof(Ks[0]);

    for (size_t i = 0; i < nK; i++)
    {
        test_forward_vs_naive(Ks[i]);
        test_backward_vs_naive(Ks[i]);
        test_roundtrip(Ks[i]);
    }
}

/* ========================================================================= */
/* TEST 9: Linearity                                                          */
/* ========================================================================= */

static void test_linearity(void)
{
    TEST_BEGIN("Linearity: DFT(a*X + b*Y) = a*DFT(X) + b*DFT(Y)");

    size_t K = 4;
    double a = 2.5, b = -1.3;

    double *x_re = calloc(R16 * K, sizeof(double));
    double *x_im = calloc(R16 * K, sizeof(double));
    double *y_re = calloc(R16 * K, sizeof(double));
    double *y_im = calloc(R16 * K, sizeof(double));
    double *z_re = calloc(R16 * K, sizeof(double)); /* a*X + b*Y */
    double *z_im = calloc(R16 * K, sizeof(double));

    double *fx_re = calloc(R16 * K, sizeof(double));
    double *fx_im = calloc(R16 * K, sizeof(double));
    double *fy_re = calloc(R16 * K, sizeof(double));
    double *fy_im = calloc(R16 * K, sizeof(double));
    double *fz_re = calloc(R16 * K, sizeof(double));
    double *fz_im = calloc(R16 * K, sizeof(double));

    unsigned int seed = 11111u;
    fill_random(x_re, R16 * K, &seed);
    fill_random(x_im, R16 * K, &seed);
    fill_random(y_re, R16 * K, &seed);
    fill_random(y_im, R16 * K, &seed);

    for (size_t i = 0; i < R16 * K; i++)
    {
        z_re[i] = a * x_re[i] + b * y_re[i];
        z_im[i] = a * x_im[i] + b * y_im[i];
    }

    radix16_butterfly_forward_scalar(K, x_re, x_im, fx_re, fx_im);
    radix16_butterfly_forward_scalar(K, y_re, y_im, fy_re, fy_im);
    radix16_butterfly_forward_scalar(K, z_re, z_im, fz_re, fz_im);

    /* Check: fz == a*fx + b*fy */
    double max_err = 0.0;
    for (size_t i = 0; i < R16 * K; i++)
    {
        double exp_re = a * fx_re[i] + b * fy_re[i];
        double exp_im = a * fx_im[i] + b * fy_im[i];
        double err_re = fabs(fz_re[i] - exp_re);
        double err_im = fabs(fz_im[i] - exp_im);
        if (err_re > max_err) max_err = err_re;
        if (err_im > max_err) max_err = err_im;
    }

    if (max_err > 5e-14)
    {
        printf("    linearity error = %.3e\n", max_err);
        _test_ok = 0;
    }

    free(x_re); free(x_im); free(y_re); free(y_im);
    free(z_re); free(z_im);
    free(fx_re); free(fx_im); free(fy_re); free(fy_im);
    free(fz_re); free(fz_im);

    TEST_END();
}

/* ========================================================================= */
/* MAIN                                                                       */
/* ========================================================================= */

int main(void)
{
    printf("=== Radix-16 Scalar Butterfly Test Suite ===\n\n");

    printf("--- Known patterns ---\n");
    test_impulse();
    test_dc();
    test_single_frequency();
    test_alternating();

    printf("\n--- W16 twiddle unit tests ---\n");
    test_w16_twiddle_fwd_bwd_roundtrip();
    test_w16_twiddle_trivial_entries();
    test_w16_vs_naive();

    printf("\n--- Forward/backward vs naive DFT (various K) ---\n");
    test_various_K();

    printf("\n--- Structural properties ---\n");
    test_linearity();
    test_parseval();
    test_pure_real();
    test_pure_imaginary();
    test_inplace();

    printf("\n=== Results: %d/%d passed", g_tests_passed, g_tests_run);
    if (g_tests_failed > 0)
        printf(", %d FAILED", g_tests_failed);
    printf(" ===\n");

    return g_tests_failed > 0 ? 1 : 0;
}
