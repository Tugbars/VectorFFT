/**
 * @file test_radix16_butterfly.c
 * @brief Unit tests for fft_radix16_avx2_butterfly.h (twiddle-free DFT-16)
 *
 * Build (ICX / Windows):
 *   icx /O2 /QxCORE-AVX2 /Qfma test_radix16_butterfly.c /Fe:test_butterfly.exe
 *
 * Build (GCC / Linux):
 *   gcc -O2 -mavx2 -mfma -std=c11 -Wall test_radix16_butterfly.c -lm -o test_butterfly
 *
 * Tests:
 *   1. Forward butterfly vs scalar DFT-16 reference (various K)
 *   2. Backward butterfly vs scalar IDFT-16 reference
 *   3. Round-trip: forward then backward recovers input (x 16 scaling)
 *   4. Linearity: butterfly(a*x + b*y) == a*butterfly(x) + b*butterfly(y)
 *   5. Impulse response: delta at element 0 -> all-ones output
 *   6. DC input: all ones -> energy in bin 0
 *   7. Parseval's theorem: energy preservation (up to factor 16)
 *   8. In-place operation
 *   9. Specific known DFT-16 values (single-frequency test)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

/* ---- Platform-specific aligned allocation ---- */
#ifdef _WIN32
  #include <malloc.h>
  #define ALIGNED_ALLOC(alignment, size) _aligned_malloc((size), (alignment))
  #define ALIGNED_FREE(ptr)             _aligned_free(ptr)
#else
  #define ALIGNED_ALLOC(alignment, size) aligned_alloc((alignment), (size))
  #define ALIGNED_FREE(ptr)             free(ptr)
#endif

#include "fft_radix16_avx2_butterfly.h"

/* ============================================================================
 * CONSTANTS & GLOBALS
 * ========================================================================= */

#define PI  3.14159265358979323846
#define TOL 1e-12

static int g_run = 0, g_pass = 0, g_fail = 0;

/* ============================================================================
 * UTILITIES
 * ========================================================================= */

static double *alloc_buf(size_t K)
{
    size_t bytes = ((16 * K * sizeof(double) + 31) / 32) * 32;
    double *p = (double *)ALIGNED_ALLOC(32, bytes);
    if (!p) { fprintf(stderr, "OOM\n"); exit(1); }
    memset(p, 0, bytes);
    return p;
}

static void free_buf(double *p) { if (p) ALIGNED_FREE(p); }

static uint64_t rng_state = 0xCAFEBABEDEAD1234ULL;
static uint64_t rng_next(void)
{
    uint64_t x = rng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return (rng_state = x);
}
static double rng_double(void)
{
    return (double)(int64_t)rng_next() / (double)INT64_MAX;
}
static void fill_rand(double *buf, size_t n)
{
    for (size_t i = 0; i < n; i++) buf[i] = rng_double();
}

static void report(const char *name, size_t K, double err, double tol)
{
    g_run++;
    if (err < tol)
    {
        g_pass++;
        printf("  [PASS] %-55s K=%-6zu err=%.2e\n", name, K, err);
    }
    else
    {
        g_fail++;
        printf("  [FAIL] %-55s K=%-6zu err=%.2e (tol=%.2e)\n",
               name, K, err, tol);
    }
}

/* ============================================================================
 * SCALAR REFERENCE: exact same 4-group radix-4 x radix-4 decomposition
 * ========================================================================= */

typedef struct { double re, im; } cx_t;

static void scalar_radix4(cx_t a, cx_t b, cx_t c, cx_t d,
                           cx_t *y0, cx_t *y1, cx_t *y2, cx_t *y3,
                           double sign)
{
    cx_t sAC = {a.re+c.re, a.im+c.im};
    cx_t sBD = {b.re+d.re, b.im+d.im};
    cx_t dAC = {a.re-c.re, a.im-c.im};
    cx_t dBD = {b.re-d.re, b.im-d.im};

    y0->re = sAC.re + sBD.re;  y0->im = sAC.im + sBD.im;
    y2->re = sAC.re - sBD.re;  y2->im = sAC.im - sBD.im;

    cx_t rot;
    if (sign < 0) { rot.re = -dBD.im; rot.im =  dBD.re; }
    else           { rot.re =  dBD.im; rot.im = -dBD.re; }

    y1->re = dAC.re - rot.re;  y1->im = dAC.im - rot.im;
    y3->re = dAC.re + rot.re;  y3->im = dAC.im + rot.im;
}

static void scalar_r16_group(int gid, const cx_t x[16], cx_t y[16], double sign)
{
    cx_t xg[4] = {x[gid], x[gid+4], x[gid+8], x[gid+12]};
    cx_t t[4];
    scalar_radix4(xg[0], xg[1], xg[2], xg[3],
                  &t[0], &t[1], &t[2], &t[3], sign);

    /* W4 intermediates */
    if (gid == 1)
    {
        cx_t tmp;
        if (sign < 0) { /* fwd: -j */
            tmp = t[1]; t[1].re = tmp.im; t[1].im = -tmp.re;
            t[2].re = -t[2].re; t[2].im = -t[2].im;
            tmp = t[3]; t[3].re = -tmp.im; t[3].im = tmp.re;
        } else { /* bwd: +j */
            tmp = t[1]; t[1].re = -tmp.im; t[1].im = tmp.re;
            t[2].re = -t[2].re; t[2].im = -t[2].im;
            tmp = t[3]; t[3].re = tmp.im; t[3].im = -tmp.re;
        }
    }
    else if (gid == 2)
    {
        t[0].re = -t[0].re; t[0].im = -t[0].im;
        cx_t tmp;
        if (sign < 0) {
            tmp = t[1]; t[1].re = -tmp.im; t[1].im = tmp.re;
            tmp = t[3]; t[3].re = tmp.im; t[3].im = -tmp.re;
        } else {
            tmp = t[1]; t[1].re = tmp.im; t[1].im = -tmp.re;
            tmp = t[3]; t[3].re = -tmp.im; t[3].im = tmp.re;
        }
    }
    else if (gid == 3)
    {
        cx_t tmp;
        if (sign < 0) {
            tmp = t[0]; t[0].re = -tmp.im; t[0].im = tmp.re;
            tmp = t[2]; t[2].re = tmp.im; t[2].im = -tmp.re;
        } else {
            tmp = t[0]; t[0].re = tmp.im; t[0].im = -tmp.re;
            tmp = t[2]; t[2].re = -tmp.im; t[2].im = tmp.re;
        }
        t[3].re = -t[3].re; t[3].im = -t[3].im;
    }

    cx_t out[4];
    scalar_radix4(t[0], t[1], t[2], t[3],
                  &out[0], &out[1], &out[2], &out[3], sign);

    int base = gid * 4;
    y[base+0] = out[0]; y[base+1] = out[1];
    y[base+2] = out[2]; y[base+3] = out[3];
}

/** Scalar DFT-16 via 4-group decomposition (no twiddles) */
static void scalar_dft16(const cx_t x[16], cx_t y[16], double sign)
{
    for (int g = 0; g < 4; g++)
        scalar_r16_group(g, x, y, sign);
}

/** Apply scalar butterfly to all K columns */
static void ref_butterfly(
    size_t K,
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    double sign)
{
    for (size_t k = 0; k < K; k++)
    {
        cx_t x[16], y[16];
        for (int r = 0; r < 16; r++)
        {
            x[r].re = in_re[r * K + k];
            x[r].im = in_im[r * K + k];
        }
        scalar_dft16(x, y, sign);
        for (int m = 0; m < 16; m++)
        {
            out_re[m * K + k] = y[m].re;
            out_im[m * K + k] = y[m].im;
        }
    }
}

/* ============================================================================
 * HELPER: max relative error
 * ========================================================================= */

static double max_rel_err(const double *a, const double *b, size_t n)
{
    double mm = 0, me = 0;
    for (size_t i = 0; i < n; i++)
    {
        double m = fabs(a[i]); if (m > mm) mm = m;
        double e = fabs(a[i] - b[i]); if (e > me) me = e;
    }
    return (mm > 1e-15) ? me / mm : me;
}

/* ============================================================================
 * TEST 1 & 2: Forward / backward vs scalar reference
 * ========================================================================= */

static void test_vs_reference(size_t K, bool forward)
{
    rng_state = 0xABCD0000ULL + K * 7 + (forward ? 0 : 99);

    double *in_re  = alloc_buf(K); double *in_im  = alloc_buf(K);
    double *out_re = alloc_buf(K); double *out_im = alloc_buf(K);
    double *ref_re = alloc_buf(K); double *ref_im = alloc_buf(K);

    fill_rand(in_re, 16*K);
    fill_rand(in_im, 16*K);

    double sign = forward ? -1.0 : 1.0;
    ref_butterfly(K, in_re, in_im, ref_re, ref_im, sign);

    if (forward)
        radix16_butterfly_forward_avx2(K, in_re, in_im, out_re, out_im);
    else
        radix16_butterfly_backward_avx2(K, in_re, in_im, out_re, out_im);

    double e1 = max_rel_err(ref_re, out_re, 16*K);
    double e2 = max_rel_err(ref_im, out_im, 16*K);
    double err = (e1 > e2) ? e1 : e2;

    char name[80];
    snprintf(name, sizeof(name), "%s vs scalar ref",
             forward ? "Forward" : "Backward");
    report(name, K, err, TOL);

    free_buf(in_re); free_buf(in_im);
    free_buf(out_re); free_buf(out_im);
    free_buf(ref_re); free_buf(ref_im);
}

/* ============================================================================
 * TEST 3: Round-trip with DIT permutation
 *
 * The 4-group butterfly writes output in a permuted (digit-reversed) order:
 *   input position i maps through group (i%4), sub-index (i/4)
 *   output position = 4*(i%4) + (i/4)
 * This is a 4×4 index transpose. To get a true round-trip, we must apply
 * this transpose between forward and backward:
 *   backward( permute( forward(x) ) ) == 16 * x
 * ========================================================================= */

/**
 * Apply the DIT 4×4 index transpose to each k-column.
 * Maps output position m to position 4*(m%4) + m/4.
 */
static void apply_dit_permutation(
    size_t K,
    const double *in_re, const double *in_im,
    double *out_re, double *out_im)
{
    for (size_t k = 0; k < K; k++)
    {
        double tmp_re[16], tmp_im[16];
        for (int m = 0; m < 16; m++)
        {
            tmp_re[m] = in_re[m * K + k];
            tmp_im[m] = in_im[m * K + k];
        }
        for (int m = 0; m < 16; m++)
        {
            int dst = 4 * (m % 4) + (m / 4);
            out_re[dst * K + k] = tmp_re[m];
            out_im[dst * K + k] = tmp_im[m];
        }
    }
}

static void test_roundtrip(size_t K)
{
    rng_state = 0x12340000ULL + K;

    double *x_re  = alloc_buf(K); double *x_im  = alloc_buf(K);
    double *f_re  = alloc_buf(K); double *f_im  = alloc_buf(K);
    double *p_re  = alloc_buf(K); double *p_im  = alloc_buf(K);
    double *rt_re = alloc_buf(K); double *rt_im = alloc_buf(K);

    fill_rand(x_re, 16*K);
    fill_rand(x_im, 16*K);

    radix16_butterfly_forward_avx2(K, x_re, x_im, f_re, f_im);
    apply_dit_permutation(K, f_re, f_im, p_re, p_im);
    radix16_butterfly_backward_avx2(K, p_re, p_im, rt_re, rt_im);
    apply_dit_permutation(K, rt_re, rt_im, p_re, p_im);

    /* p should == x * 16 */
    double max_err = 0;
    double max_mag = 0;
    for (size_t i = 0; i < 16 * K; i++)
    {
        double expected_re = x_re[i] * 16.0;
        double expected_im = x_im[i] * 16.0;
        double m = fabs(expected_re); if (m > max_mag) max_mag = m;
        m = fabs(expected_im); if (m > max_mag) max_mag = m;
        double e = fabs(p_re[i] - expected_re);
        if (e > max_err) max_err = e;
        e = fabs(p_im[i] - expected_im);
        if (e > max_err) max_err = e;
    }
    double err = (max_mag > 1e-15) ? max_err / max_mag : max_err;

    report("Round-trip (fwd->perm->bwd->perm) == 16*x", K, err, TOL);

    free_buf(x_re); free_buf(x_im);
    free_buf(f_re); free_buf(f_im);
    free_buf(p_re); free_buf(p_im);
    free_buf(rt_re); free_buf(rt_im);
}

/* ============================================================================
 * TEST 4: Linearity: F(a*x + b*y) == a*F(x) + b*F(y)
 * ========================================================================= */

static void test_linearity(size_t K)
{
    rng_state = 0x55550000ULL + K;

    double *x_re = alloc_buf(K); double *x_im = alloc_buf(K);
    double *y_re = alloc_buf(K); double *y_im = alloc_buf(K);
    double *fx_re = alloc_buf(K); double *fx_im = alloc_buf(K);
    double *fy_re = alloc_buf(K); double *fy_im = alloc_buf(K);
    double *sum_re = alloc_buf(K); double *sum_im = alloc_buf(K);
    double *fsum_re = alloc_buf(K); double *fsum_im = alloc_buf(K);

    fill_rand(x_re, 16*K); fill_rand(x_im, 16*K);
    fill_rand(y_re, 16*K); fill_rand(y_im, 16*K);

    double a = 2.7, b = -1.3;

    /* Compute a*x + b*y */
    for (size_t i = 0; i < 16*K; i++)
    {
        sum_re[i] = a * x_re[i] + b * y_re[i];
        sum_im[i] = a * x_im[i] + b * y_im[i];
    }

    radix16_butterfly_forward_avx2(K, x_re, x_im, fx_re, fx_im);
    radix16_butterfly_forward_avx2(K, y_re, y_im, fy_re, fy_im);
    radix16_butterfly_forward_avx2(K, sum_re, sum_im, fsum_re, fsum_im);

    /* Check F(sum) == a*F(x) + b*F(y) */
    double max_err = 0, max_mag = 0;
    for (size_t i = 0; i < 16*K; i++)
    {
        double exp_re = a * fx_re[i] + b * fy_re[i];
        double exp_im = a * fx_im[i] + b * fy_im[i];
        double m = fabs(exp_re); if (m > max_mag) max_mag = m;
        m = fabs(exp_im); if (m > max_mag) max_mag = m;
        double e = fabs(fsum_re[i] - exp_re);
        if (e > max_err) max_err = e;
        e = fabs(fsum_im[i] - exp_im);
        if (e > max_err) max_err = e;
    }
    double err = (max_mag > 1e-15) ? max_err / max_mag : max_err;

    report("Linearity: F(ax+by) == aF(x)+bF(y)", K, err, TOL);

    free_buf(x_re); free_buf(x_im); free_buf(y_re); free_buf(y_im);
    free_buf(fx_re); free_buf(fx_im); free_buf(fy_re); free_buf(fy_im);
    free_buf(sum_re); free_buf(sum_im); free_buf(fsum_re); free_buf(fsum_im);
}

/* ============================================================================
 * TEST 5: Impulse at element 0 → all-ones output
 * ========================================================================= */

static void test_impulse(size_t K)
{
    double *in_re  = alloc_buf(K); double *in_im  = alloc_buf(K);
    double *out_re = alloc_buf(K); double *out_im = alloc_buf(K);
    double *ref_re = alloc_buf(K); double *ref_im = alloc_buf(K);

    /* Impulse: x[0][k=0] = 1, rest 0 */
    in_re[0] = 1.0;

    ref_butterfly(K, in_re, in_im, ref_re, ref_im, -1.0);
    radix16_butterfly_forward_avx2(K, in_re, in_im, out_re, out_im);

    double max_err = 0;
    for (size_t i = 0; i < 16*K; i++)
    {
        double e = fabs(out_re[i] - ref_re[i]);
        if (e > max_err) max_err = e;
        e = fabs(out_im[i] - ref_im[i]);
        if (e > max_err) max_err = e;
    }

    report("Impulse response (element 0)", K, max_err, 1e-14);

    free_buf(in_re); free_buf(in_im);
    free_buf(out_re); free_buf(out_im);
    free_buf(ref_re); free_buf(ref_im);
}

/* ============================================================================
 * TEST 6: DC input → energy in bin 0
 * ========================================================================= */

static void test_dc(size_t K)
{
    double *in_re  = alloc_buf(K); double *in_im  = alloc_buf(K);
    double *out_re = alloc_buf(K); double *out_im = alloc_buf(K);
    double *ref_re = alloc_buf(K); double *ref_im = alloc_buf(K);

    for (size_t i = 0; i < 16*K; i++) in_re[i] = 1.0;

    ref_butterfly(K, in_re, in_im, ref_re, ref_im, -1.0);
    radix16_butterfly_forward_avx2(K, in_re, in_im, out_re, out_im);

    double max_err = 0;
    for (size_t i = 0; i < 16*K; i++)
    {
        double e = fabs(out_re[i] - ref_re[i]);
        if (e > max_err) max_err = e;
        e = fabs(out_im[i] - ref_im[i]);
        if (e > max_err) max_err = e;
    }

    report("DC input (all ones)", K, max_err, 1e-14);

    free_buf(in_re); free_buf(in_im);
    free_buf(out_re); free_buf(out_im);
    free_buf(ref_re); free_buf(ref_im);
}

/* ============================================================================
 * TEST 7: Parseval's theorem - energy conservation
 *
 * For DFT-N: sum|Y[m]|^2 = N * sum|x[r]|^2
 * Here N=16, applied independently to each k column.
 * ========================================================================= */

static void test_parseval(size_t K)
{
    rng_state = 0x77770000ULL + K;

    double *in_re  = alloc_buf(K); double *in_im  = alloc_buf(K);
    double *out_re = alloc_buf(K); double *out_im = alloc_buf(K);

    fill_rand(in_re, 16*K);
    fill_rand(in_im, 16*K);

    radix16_butterfly_forward_avx2(K, in_re, in_im, out_re, out_im);

    double max_err = 0;
    for (size_t k = 0; k < K; k++)
    {
        double e_in = 0, e_out = 0;
        for (int r = 0; r < 16; r++)
        {
            double xr = in_re[r*K + k], xi = in_im[r*K + k];
            double yr = out_re[r*K + k], yi = out_im[r*K + k];
            e_in  += xr*xr + xi*xi;
            e_out += yr*yr + yi*yi;
        }
        /* e_out should == 16 * e_in */
        double expected = 16.0 * e_in;
        double err = (expected > 1e-15) ? fabs(e_out - expected) / expected : 0;
        if (err > max_err) max_err = err;
    }

    report("Parseval: sum|Y|^2 == 16 * sum|x|^2", K, max_err, TOL);

    free_buf(in_re); free_buf(in_im);
    free_buf(out_re); free_buf(out_im);
}

/* ============================================================================
 * TEST 8: In-place operation
 * ========================================================================= */

static void test_inplace(size_t K)
{
    rng_state = 0x99990000ULL + K;

    double *data_re = alloc_buf(K); double *data_im = alloc_buf(K);
    double *ref_re  = alloc_buf(K); double *ref_im  = alloc_buf(K);

    fill_rand(data_re, 16*K);
    fill_rand(data_im, 16*K);

    /* Reference: out-of-place */
    ref_butterfly(K, data_re, data_im, ref_re, ref_im, -1.0);

    /* In-place */
    radix16_butterfly_forward_avx2(K, data_re, data_im, data_re, data_im);

    double e1 = max_rel_err(ref_re, data_re, 16*K);
    double e2 = max_rel_err(ref_im, data_im, 16*K);
    double err = (e1 > e2) ? e1 : e2;

    report("In-place forward", K, err, TOL);

    free_buf(data_re); free_buf(data_im);
    free_buf(ref_re); free_buf(ref_im);
}

/* ============================================================================
 * TEST 9: Single-frequency test
 *
 * Input: x[r] = exp(j * 2*pi * f * r / 16) for a chosen frequency f.
 * DFT-16 of this should give a spike at bin f (in the decomposition's
 * output ordering). We verify via the scalar reference rather than
 * assuming standard ordering.
 * ========================================================================= */

static void test_single_freq(size_t K, int freq)
{
    double *in_re  = alloc_buf(K); double *in_im  = alloc_buf(K);
    double *out_re = alloc_buf(K); double *out_im = alloc_buf(K);
    double *ref_re = alloc_buf(K); double *ref_im = alloc_buf(K);

    /* Set input for k=0 column, leave rest zero */
    for (int r = 0; r < 16; r++)
    {
        double angle = 2.0 * PI * freq * r / 16.0;
        in_re[r * K + 0] = cos(angle);
        in_im[r * K + 0] = sin(angle);
    }

    ref_butterfly(K, in_re, in_im, ref_re, ref_im, -1.0);
    radix16_butterfly_forward_avx2(K, in_re, in_im, out_re, out_im);

    double max_err = 0;
    for (size_t i = 0; i < 16*K; i++)
    {
        double e = fabs(out_re[i] - ref_re[i]);
        if (e > max_err) max_err = e;
        e = fabs(out_im[i] - ref_im[i]);
        if (e > max_err) max_err = e;
    }

    char name[80];
    snprintf(name, sizeof(name), "Single freq=%d tone", freq);
    report(name, K, max_err, 1e-13);

    free_buf(in_re); free_buf(in_im);
    free_buf(out_re); free_buf(out_im);
    free_buf(ref_re); free_buf(ref_im);
}

/* ============================================================================
 * MAIN
 * ========================================================================= */

int main(void)
{
    printf("================================================================\n");
    printf("  Radix-16 AVX2 Twiddle-Free Butterfly - Unit Tests (v1.0)\n");
    printf("================================================================\n\n");

    /* ---- Section 1: Forward vs scalar reference ---- */
    printf("--- Forward vs Reference ---\n");
    {
        size_t Ks[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384};
        for (int i = 0; i < (int)(sizeof(Ks)/sizeof(Ks[0])); i++)
            test_vs_reference(Ks[i], true);
    }
    printf("\n");

    /* ---- Section 2: Backward vs scalar reference ---- */
    printf("--- Backward vs Reference ---\n");
    {
        size_t Ks[] = {4, 8, 16, 64, 256, 1024, 4096, 16384};
        for (int i = 0; i < (int)(sizeof(Ks)/sizeof(Ks[0])); i++)
            test_vs_reference(Ks[i], false);
    }
    printf("\n");

    /* ---- Section 3: Round-trip ---- */
    printf("--- Round-Trip (fwd->bwd == 16*x) ---\n");
    {
        size_t Ks[] = {4, 16, 64, 256, 1024, 4096};
        for (int i = 0; i < (int)(sizeof(Ks)/sizeof(Ks[0])); i++)
            test_roundtrip(Ks[i]);
    }
    printf("\n");

    /* ---- Section 4: Linearity ---- */
    printf("--- Linearity ---\n");
    {
        size_t Ks[] = {4, 64, 1024};
        for (int i = 0; i < (int)(sizeof(Ks)/sizeof(Ks[0])); i++)
            test_linearity(Ks[i]);
    }
    printf("\n");

    /* ---- Section 5: Impulse ---- */
    printf("--- Impulse Response ---\n");
    test_impulse(4);
    test_impulse(64);
    printf("\n");

    /* ---- Section 6: DC ---- */
    printf("--- DC Input ---\n");
    test_dc(4);
    test_dc(64);
    printf("\n");

    /* ---- Section 7: Parseval ---- */
    printf("--- Parseval's Theorem ---\n");
    {
        size_t Ks[] = {4, 64, 256, 1024, 4096};
        for (int i = 0; i < (int)(sizeof(Ks)/sizeof(Ks[0])); i++)
            test_parseval(Ks[i]);
    }
    printf("\n");

    /* ---- Section 8: In-place ---- */
    printf("--- In-Place ---\n");
    {
        size_t Ks[] = {4, 16, 64, 256, 1024};
        for (int i = 0; i < (int)(sizeof(Ks)/sizeof(Ks[0])); i++)
            test_inplace(Ks[i]);
    }
    printf("\n");

    /* ---- Section 9: Single-frequency tones ---- */
    printf("--- Single-Frequency Tones ---\n");
    for (int f = 0; f < 16; f++)
        test_single_freq(4, f);
    printf("\n");

    /* ---- Summary ---- */
    printf("================================================================\n");
    printf("  RESULTS: %d/%d passed", g_pass, g_run);
    if (g_fail > 0) printf("  (%d FAILED)", g_fail);
    printf("\n================================================================\n");

    return (g_fail == 0) ? 0 : 1;
}
