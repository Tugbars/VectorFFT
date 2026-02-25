/**
 * @file test_radix16_avx512.c
 * @brief Comprehensive test suite for AVX-512 radix-16 butterfly
 *
 * Tests: known patterns, forward/backward vs naive DFT, structural
 * properties, in-place operation, cross-validation against scalar.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "fft_radix16_avx512_butterfly.h"
#include "fft_radix16_scalar_butterfly.h"   /* cross-validation */

/* ============================================================================
 * UTILITIES
 * ========================================================================= */

static int g_pass = 0, g_fail = 0;

static void check(const char *name, int cond)
{
    if (cond) { printf("  [PASS] %s\n", name); g_pass++; }
    else      { printf("  [FAIL] %s\n", name); g_fail++; }
}

static double max_error(const double *a, const double *b, size_t n)
{
    double mx = 0.0;
    for (size_t i = 0; i < n; i++)
    {
        double e = fabs(a[i] - b[i]);
        if (e > mx) mx = e;
    }
    return mx;
}

/* Naive DFT-16 for reference (O(N²), exact formula) */
static void naive_dft16(size_t K,
    const double *in_re, const double *in_im,
    double *out_re, double *out_im, int forward)
{
    const double sign = forward ? -1.0 : 1.0;
    for (size_t k = 0; k < K; k++)
    {
        for (int m = 0; m < 16; m++)
        {
            double sum_re = 0.0, sum_im = 0.0;
            for (int n = 0; n < 16; n++)
            {
                double angle = sign * 2.0 * M_PI * (double)m * (double)n / 16.0;
                double wr, wi;
                sincos(angle, &wi, &wr);
                double xr = in_re[n * K + k];
                double xi = in_im[n * K + k];
                sum_re += xr * wr - xi * wi;
                sum_im += xr * wi + xi * wr;
            }
            out_re[m * K + k] = sum_re;
            out_im[m * K + k] = sum_im;
        }
    }
}

/* Allocate 64-byte aligned memory */
static double *alloc64(size_t n)
{
    double *p = NULL;
    if (posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) abort();
    memset(p, 0, n * sizeof(double));
    return p;
}

/* Fill buffer with deterministic pseudo-random values */
static void fill_random(double *buf, size_t n, unsigned seed)
{
    unsigned s = seed;
    for (size_t i = 0; i < n; i++)
    {
        s = s * 1103515245u + 12345u;
        buf[i] = ((double)(s >> 16) / 32768.0) - 1.0;
    }
}

/* ============================================================================
 * TEST: KNOWN PATTERNS (K=1)
 * ========================================================================= */

static void test_impulse(void)
{
    const size_t N = 16, K = 1;
    double *xr = alloc64(N), *xi = alloc64(N);
    double *yr = alloc64(N), *yi = alloc64(N);

    /* Delta at n=0 → all outputs = 1+0i */
    xr[0] = 1.0;
    radix16_butterfly_forward_avx512(K, xr, xi, yr, yi);

    double err = 0.0;
    for (int m = 0; m < 16; m++)
    {
        err = fmax(err, fabs(yr[m] - 1.0));
        err = fmax(err, fabs(yi[m]));
    }
    check("Impulse response (delta at n=0)", err < 1e-15);
    if (err >= 1e-15)
    {
        for (int m = 0; m < 16; m++)
            printf("    Y[%2d] = %.15e + %.15ei\n", m, yr[m], yi[m]);
    }

    free(xr); free(xi); free(yr); free(yi);
}

static void test_dc(void)
{
    const size_t N = 16, K = 1;
    double *xr = alloc64(N), *xi = alloc64(N);
    double *yr = alloc64(N), *yi = alloc64(N);

    /* All ones → Y[0]=16, Y[m]=0 for m>0 */
    for (int n = 0; n < 16; n++) xr[n] = 1.0;
    radix16_butterfly_forward_avx512(K, xr, xi, yr, yi);

    double err = fabs(yr[0] - 16.0) + fabs(yi[0]);
    for (int m = 1; m < 16; m++)
    {
        err = fmax(err, fabs(yr[m]));
        err = fmax(err, fabs(yi[m]));
    }
    check("DC input (all ones)", err < 2e-13);
    if (err >= 2e-13)
    {
        for (int m = 0; m < 16; m++)
            printf("    Y[%2d] = %.15e + %.15ei\n", m, yr[m], yi[m]);
    }

    free(xr); free(xi); free(yr); free(yi);
}

static void test_single_frequency(void)
{
    const size_t N = 16, K = 1;
    double *xr = alloc64(N), *xi = alloc64(N);
    double *yr = alloc64(N), *yi = alloc64(N);

    /* x[n] = exp(+2πi·3·n/16) → Y[3] = 16, rest = 0 (forward DFT convention) */
    for (int n = 0; n < 16; n++)
    {
        double angle = 2.0 * M_PI * 3.0 * (double)n / 16.0;
        xr[n] = cos(angle);
        xi[n] = sin(angle);
    }
    radix16_butterfly_forward_avx512(K, xr, xi, yr, yi);

    double err = fabs(yr[3] - 16.0) + fabs(yi[3]);
    for (int m = 0; m < 16; m++)
    {
        if (m == 3) continue;
        err = fmax(err, fabs(yr[m]));
        err = fmax(err, fabs(yi[m]));
    }
    check("Single frequency (bin 3)", err < 2e-13);
    if (err >= 2e-13)
    {
        for (int m = 0; m < 16; m++)
            printf("    Y[%2d] = %.15e + %.15ei\n", m, yr[m], yi[m]);
    }

    free(xr); free(xi); free(yr); free(yi);
}

static void test_alternating(void)
{
    const size_t N = 16, K = 1;
    double *xr = alloc64(N), *xi = alloc64(N);
    double *yr = alloc64(N), *yi = alloc64(N);

    /* x[n] = (-1)^n → Y[8] = 16, rest = 0 */
    for (int n = 0; n < 16; n++) xr[n] = (n % 2 == 0) ? 1.0 : -1.0;
    radix16_butterfly_forward_avx512(K, xr, xi, yr, yi);

    double err = fabs(yr[8] - 16.0) + fabs(yi[8]);
    for (int m = 0; m < 16; m++)
    {
        if (m == 8) continue;
        err = fmax(err, fabs(yr[m]));
        err = fmax(err, fabs(yi[m]));
    }
    check("Alternating (+1,-1) → bin 8", err < 1e-14);

    free(xr); free(xi); free(yr); free(yi);
}

/* ============================================================================
 * TEST: FORWARD/BACKWARD VS NAIVE DFT (various K)
 * ========================================================================= */

static void test_vs_naive(size_t K)
{
    const size_t N = 16;
    const size_t sz = N * K;
    double *xr = alloc64(sz), *xi = alloc64(sz);
    double *yr = alloc64(sz), *yi = alloc64(sz);
    double *nr = alloc64(sz), *ni = alloc64(sz);
    double *rr = alloc64(sz), *ri = alloc64(sz);
    char buf[128];

    fill_random(xr, sz, (unsigned)(K * 31 + 7));
    fill_random(xi, sz, (unsigned)(K * 37 + 13));

    /* Forward: AVX-512 vs naive */
    radix16_butterfly_forward_avx512(K, xr, xi, yr, yi);
    naive_dft16(K, xr, xi, nr, ni, 1);

    double err_fwd_re = max_error(yr, nr, sz);
    double err_fwd_im = max_error(yi, ni, sz);
    double err_fwd = fmax(err_fwd_re, err_fwd_im);
    double tol = 6.0e-14;

    snprintf(buf, sizeof(buf), "Forward vs naive DFT-16 (K=%zu)", K);
    if (err_fwd >= tol) printf("    max error = %.3e (tol = %.3e)\n", err_fwd, tol);
    check(buf, err_fwd < tol);

    /* Backward: AVX-512 vs naive */
    radix16_butterfly_backward_avx512(K, xr, xi, yr, yi);
    naive_dft16(K, xr, xi, nr, ni, 0);

    double err_bwd = fmax(max_error(yr, nr, sz), max_error(yi, ni, sz));
    snprintf(buf, sizeof(buf), "Backward vs naive IDFT-16 (K=%zu)", K);
    if (err_bwd >= tol) printf("    max error = %.3e (tol = %.3e)\n", err_bwd, tol);
    check(buf, err_bwd < tol);

    /* Roundtrip: fwd → bwd should give 16·x */
    radix16_butterfly_forward_avx512(K, xr, xi, yr, yi);
    radix16_butterfly_backward_avx512(K, yr, yi, rr, ri);

    double err_rt = 0.0;
    for (size_t i = 0; i < sz; i++)
    {
        err_rt = fmax(err_rt, fabs(rr[i] - 16.0 * xr[i]));
        err_rt = fmax(err_rt, fabs(ri[i] - 16.0 * xi[i]));
    }
    double rt_tol = 6.0e-14;
    snprintf(buf, sizeof(buf), "Roundtrip fwd→bwd (K=%zu)", K);
    if (err_rt >= rt_tol) printf("    max roundtrip error = %.3e (tol = %.3e)\n", err_rt, rt_tol);
    check(buf, err_rt < rt_tol);

    free(xr); free(xi); free(yr); free(yi);
    free(nr); free(ni); free(rr); free(ri);
}

/* ============================================================================
 * TEST: AVX-512 VS SCALAR CROSS-VALIDATION
 * ========================================================================= */

static void test_vs_scalar(size_t K)
{
    const size_t N = 16;
    const size_t sz = N * K;
    double *xr = alloc64(sz), *xi = alloc64(sz);
    double *yr_avx = alloc64(sz), *yi_avx = alloc64(sz);
    double *yr_scl = alloc64(sz), *yi_scl = alloc64(sz);
    char buf[128];

    fill_random(xr, sz, (unsigned)(K * 41 + 3));
    fill_random(xi, sz, (unsigned)(K * 43 + 5));

    /* Forward */
    radix16_butterfly_forward_avx512(K, xr, xi, yr_avx, yi_avx);
    radix16_butterfly_forward_scalar(K, xr, xi, yr_scl, yi_scl);

    double err_fwd = fmax(max_error(yr_avx, yr_scl, sz),
                          max_error(yi_avx, yi_scl, sz));
    double tol = 4e-15;  /* few ULP difference between SIMD and scalar FP paths */
    snprintf(buf, sizeof(buf), "AVX512 vs scalar forward (K=%zu)", K);
    if (err_fwd >= tol) printf("    max error = %.3e (tol = %.3e)\n", err_fwd, tol);
    check(buf, err_fwd < tol);

    /* Backward */
    radix16_butterfly_backward_avx512(K, xr, xi, yr_avx, yi_avx);
    radix16_butterfly_backward_scalar(K, xr, xi, yr_scl, yi_scl);

    double err_bwd = fmax(max_error(yr_avx, yr_scl, sz),
                          max_error(yi_avx, yi_scl, sz));
    snprintf(buf, sizeof(buf), "AVX512 vs scalar backward (K=%zu)", K);
    if (err_bwd >= tol) printf("    max error = %.3e (tol = %.3e)\n", err_bwd, tol);
    check(buf, err_bwd < tol);

    free(xr); free(xi);
    free(yr_avx); free(yi_avx);
    free(yr_scl); free(yi_scl);
}

/* ============================================================================
 * TEST: STRUCTURAL PROPERTIES
 * ========================================================================= */

static void test_linearity(void)
{
    const size_t K = 10;
    const size_t sz = 16 * K;
    double *xr = alloc64(sz), *xi = alloc64(sz);
    double *yr_x = alloc64(sz), *yi_x = alloc64(sz);
    double *zr = alloc64(sz), *zi = alloc64(sz);
    double *yr_z = alloc64(sz), *yi_z = alloc64(sz);
    double *yy_r = alloc64(sz), *yy_i = alloc64(sz);

    fill_random(xr, sz, 100);
    fill_random(xi, sz, 101);
    fill_random(yr_x, sz, 200);  /* reuse as Y input */
    fill_random(yi_x, sz, 201);

    /* Save Y input before overwrite */
    double *yr_in = alloc64(sz), *yi_in = alloc64(sz);
    memcpy(yr_in, yr_x, sz * sizeof(double));
    memcpy(yi_in, yi_x, sz * sizeof(double));

    double a = 2.7, b = -1.3;

    /* Z = a*X + b*Y */
    for (size_t i = 0; i < sz; i++)
    {
        zr[i] = a * xr[i] + b * yr_in[i];
        zi[i] = a * xi[i] + b * yi_in[i];
    }

    /* DFT(X), DFT(Y), DFT(Z) */
    radix16_butterfly_forward_avx512(K, xr, xi, yr_x, yi_x);
    radix16_butterfly_forward_avx512(K, yr_in, yi_in, yy_r, yy_i);
    radix16_butterfly_forward_avx512(K, zr, zi, yr_z, yi_z);

    /* Check: DFT(Z) = a·DFT(X) + b·DFT(Y) */
    double err = 0.0;
    for (size_t i = 0; i < sz; i++)
    {
        double expected_re = a * yr_x[i] + b * yy_r[i];
        double expected_im = a * yi_x[i] + b * yy_i[i];
        err = fmax(err, fabs(yr_z[i] - expected_re));
        err = fmax(err, fabs(yi_z[i] - expected_im));
    }
    check("Linearity: DFT(a*X + b*Y) = a*DFT(X) + b*DFT(Y)", err < 1e-12);

    free(xr); free(xi); free(yr_x); free(yi_x); free(yr_in); free(yi_in);
    free(zr); free(zi); free(yr_z); free(yi_z); free(yy_r); free(yy_i);
}

static void test_parseval(void)
{
    const size_t K = 8;
    const size_t sz = 16 * K;
    double *xr = alloc64(sz), *xi = alloc64(sz);
    double *yr = alloc64(sz), *yi = alloc64(sz);

    fill_random(xr, sz, 300);
    fill_random(xi, sz, 301);

    radix16_butterfly_forward_avx512(K, xr, xi, yr, yi);

    /* Parseval: sum|X|² = (1/N) sum|Y|² */
    double energy_x = 0.0, energy_y = 0.0;
    for (size_t i = 0; i < sz; i++)
    {
        energy_x += xr[i] * xr[i] + xi[i] * xi[i];
        energy_y += yr[i] * yr[i] + yi[i] * yi[i];
    }
    energy_y /= 16.0;
    double err = fabs(energy_x - energy_y) / fmax(energy_x, 1e-30);
    check("Parseval's theorem (energy conservation)", err < 1e-13);

    free(xr); free(xi); free(yr); free(yi);
}

static void test_real_input(void)
{
    const size_t K = 12;
    const size_t sz = 16 * K;
    double *xr = alloc64(sz), *xi = alloc64(sz);
    double *yr = alloc64(sz), *yi = alloc64(sz);

    fill_random(xr, sz, 400);
    /* xi = 0 (already zeroed) */

    radix16_butterfly_forward_avx512(K, xr, xi, yr, yi);

    /* Conjugate symmetry: Y[m] = conj(Y[16-m]) for real input */
    double err = 0.0;
    for (size_t kk = 0; kk < K; kk++)
    {
        for (int m = 1; m < 8; m++)
        {
            int mc = 16 - m;
            err = fmax(err, fabs(yr[m * K + kk] - yr[mc * K + kk]));
            err = fmax(err, fabs(yi[m * K + kk] + yi[mc * K + kk]));
        }
    }
    if (err >= 1e-13) printf("    conjugate symmetry error = %.3e\n", err);
    check("Pure real input", err < 1e-13);

    free(xr); free(xi); free(yr); free(yi);
}

static void test_imag_input(void)
{
    const size_t K = 12;
    const size_t sz = 16 * K;
    double *xr = alloc64(sz), *xi = alloc64(sz);
    double *yr = alloc64(sz), *yi = alloc64(sz);

    fill_random(xi, sz, 500);
    /* xr = 0 (already zeroed) */

    radix16_butterfly_forward_avx512(K, xr, xi, yr, yi);

    /* Anti-conjugate symmetry: Y[m] = -conj(Y[16-m]) for pure imaginary */
    double err = 0.0;
    for (size_t kk = 0; kk < K; kk++)
    {
        for (int m = 1; m < 8; m++)
        {
            int mc = 16 - m;
            err = fmax(err, fabs(yr[m * K + kk] + yr[mc * K + kk]));
            err = fmax(err, fabs(yi[m * K + kk] - yi[mc * K + kk]));
        }
    }
    if (err >= 1e-13) printf("    anti-conjugate symmetry error = %.3e\n", err);
    check("Pure imaginary input", err < 1e-13);

    free(xr); free(xi); free(yr); free(yi);
}

static void test_inplace(void)
{
    const size_t K = 15;
    const size_t sz = 16 * K;
    double *xr = alloc64(sz), *xi = alloc64(sz);
    double *yr = alloc64(sz), *yi = alloc64(sz);
    double *br = alloc64(sz), *bi = alloc64(sz);

    fill_random(xr, sz, 600);
    fill_random(xi, sz, 601);
    memcpy(br, xr, sz * sizeof(double));
    memcpy(bi, xi, sz * sizeof(double));

    /* Out-of-place */
    radix16_butterfly_forward_avx512(K, xr, xi, yr, yi);

    /* In-place (using br,bi as both src and dst) */
    radix16_butterfly_forward_avx512(K, br, bi, br, bi);

    double err = fmax(max_error(yr, br, sz), max_error(yi, bi, sz));
    if (err >= 1e-15) printf("    in-place vs out-of-place error = %.3e\n", err);
    check("In-place operation (src == dst)", err < 1e-15);

    free(xr); free(xi); free(yr); free(yi); free(br); free(bi);
}

/* ============================================================================
 * TEST: TAIL HANDLING (various K values to exercise mask paths)
 * ========================================================================= */

static void test_tail_k_values(void)
{
    /* K values that exercise:
     * K=1:   pure mask (1 lane)
     * K=3:   mask (3 lanes)
     * K=7:   mask (7 lanes)
     * K=8:   exact 1 batch
     * K=9:   1 batch + 1 mask
     * K=15:  1 batch + 7 mask
     * K=16:  2 batches exact
     * K=23:  2 batches + 7 mask
     * K=24:  3 batches exact (U=3 main loop)
     * K=25:  U=3 + 1 mask
     * K=47:  U=3 + 2 batches + 7 mask
     * K=48:  2× U=3 exact
     */
    const size_t test_K[] = {1, 3, 7, 8, 9, 15, 16, 23, 24, 25, 47, 48};
    const int nK = sizeof(test_K) / sizeof(test_K[0]);
    char buf[128];

    for (int i = 0; i < nK; i++)
    {
        const size_t K = test_K[i];
        const size_t sz = 16 * K;
        double *xr = alloc64(sz), *xi = alloc64(sz);
        double *yr = alloc64(sz), *yi = alloc64(sz);
        double *nr = alloc64(sz), *ni = alloc64(sz);

        fill_random(xr, sz, (unsigned)(K * 71 + 11));
        fill_random(xi, sz, (unsigned)(K * 73 + 17));

        radix16_butterfly_forward_avx512(K, xr, xi, yr, yi);
        naive_dft16(K, xr, xi, nr, ni, 1);

        double err = fmax(max_error(yr, nr, sz), max_error(yi, ni, sz));
        double tol = 6.0e-14;
        snprintf(buf, sizeof(buf), "Tail handling K=%zu (rem=%zu)", K, K % 8);
        if (err >= tol) printf("    max error = %.3e (tol = %.3e)\n", err, tol);
        check(buf, err < tol);

        free(xr); free(xi); free(yr); free(yi); free(nr); free(ni);
    }
}

/* ============================================================================
 * MAIN
 * ========================================================================= */

int main(void)
{
    printf("=== Radix-16 AVX-512 Butterfly Test Suite ===\n\n");

    printf("--- Known patterns ---\n");
    test_impulse();
    test_dc();
    test_single_frequency();
    test_alternating();

    printf("\n--- Forward/backward vs naive DFT (various K) ---\n");
    static const size_t Ks[] = {1, 2, 3, 4, 5, 7, 8, 15, 16, 24, 31, 32, 48, 64, 100};
    for (int i = 0; i < (int)(sizeof(Ks)/sizeof(Ks[0])); i++)
        test_vs_naive(Ks[i]);

    printf("\n--- AVX-512 vs scalar cross-validation ---\n");
    static const size_t Ks2[] = {1, 3, 7, 8, 9, 15, 16, 23, 24, 25, 48, 64, 100};
    for (int i = 0; i < (int)(sizeof(Ks2)/sizeof(Ks2[0])); i++)
        test_vs_scalar(Ks2[i]);

    printf("\n--- Tail handling (mask paths) ---\n");
    test_tail_k_values();

    printf("\n--- Structural properties ---\n");
    test_linearity();
    test_parseval();
    test_real_input();
    test_imag_input();
    test_inplace();

    printf("\n=== Results: %d/%d passed", g_pass, g_pass + g_fail);
    if (g_fail > 0) printf(", %d FAILED", g_fail);
    printf(" ===\n");

    return g_fail > 0 ? 1 : 0;
}
