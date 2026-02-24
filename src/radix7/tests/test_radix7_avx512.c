/**
 * @file  test_radix7_avx512.c
 * @brief Test harness for AVX-512 Rader radix-7 butterflies
 *
 * Tests: N1 fwd/bwd roundtrip, twiddled roundtrip, AVX-512 vs scalar
 *        cross-check for U=2 body, U=1 remainder, and scalar tail.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "fft_r7_platform.h"
#include "scalar/fft_radix7_scalar.h"
#include "avx512/fft_radix7_avx512.h"

#define ALIGN64 __attribute__((aligned(64)))

static int g_pass = 0, g_fail = 0;

static double maxerr_arr(const double *a, const double *b, int n)
{
    double mx = 0.0;
    for (int i = 0; i < n; i++)
    {
        double e = fabs(a[i] - b[i]);
        if (e > mx)
            mx = e;
    }
    return mx;
}

static void check(const char *name, double err, double tol)
{
    if (err <= tol)
    {
        g_pass++;
        printf("  PASS %-40s err=%.2e\n", name, err);
    }
    else
    {
        g_fail++;
        printf("  FAIL %-40s err=%.2e > %.2e\n", name, err, tol);
    }
}

/* Fill aligned buffer with deterministic pseudo-random data */
static void fill(double *buf, int n, unsigned seed)
{
    for (int i = 0; i < n; i++)
    {
        seed = seed * 1103515245u + 12345u;
        buf[i] = ((double)(seed >> 16) / 32768.0) - 1.0;
    }
}

/* ------------------------------------------------------------------ */
/*  Twiddled roundtrip: fwd→bwd should recover original (÷7)          */
/* ------------------------------------------------------------------ */

static void test_twiddled_roundtrip(int K)
{
    /* Allocate 64-byte aligned arrays */
    double *a_re = (double *)r7_aligned_alloc(64, K * sizeof(double));
    double *a_im = (double *)r7_aligned_alloc(64, K * sizeof(double));
    double *b_re = (double *)r7_aligned_alloc(64, K * sizeof(double));
    double *b_im = (double *)r7_aligned_alloc(64, K * sizeof(double));
    double *c_re = (double *)r7_aligned_alloc(64, K * sizeof(double));
    double *c_im = (double *)r7_aligned_alloc(64, K * sizeof(double));
    double *d_re = (double *)r7_aligned_alloc(64, K * sizeof(double));
    double *d_im = (double *)r7_aligned_alloc(64, K * sizeof(double));
    double *e_re = (double *)r7_aligned_alloc(64, K * sizeof(double));
    double *e_im = (double *)r7_aligned_alloc(64, K * sizeof(double));
    double *f_re = (double *)r7_aligned_alloc(64, K * sizeof(double));
    double *f_im = (double *)r7_aligned_alloc(64, K * sizeof(double));
    double *g_re = (double *)r7_aligned_alloc(64, K * sizeof(double));
    double *g_im = (double *)r7_aligned_alloc(64, K * sizeof(double));

    double *y0r = (double *)r7_aligned_alloc(64, K * 8), *y0i = (double *)r7_aligned_alloc(64, K * 8);
    double *y1r = (double *)r7_aligned_alloc(64, K * 8), *y1i = (double *)r7_aligned_alloc(64, K * 8);
    double *y2r = (double *)r7_aligned_alloc(64, K * 8), *y2i = (double *)r7_aligned_alloc(64, K * 8);
    double *y3r = (double *)r7_aligned_alloc(64, K * 8), *y3i = (double *)r7_aligned_alloc(64, K * 8);
    double *y4r = (double *)r7_aligned_alloc(64, K * 8), *y4i = (double *)r7_aligned_alloc(64, K * 8);
    double *y5r = (double *)r7_aligned_alloc(64, K * 8), *y5i = (double *)r7_aligned_alloc(64, K * 8);
    double *y6r = (double *)r7_aligned_alloc(64, K * 8), *y6i = (double *)r7_aligned_alloc(64, K * 8);

    double *z0r = (double *)r7_aligned_alloc(64, K * 8), *z0i = (double *)r7_aligned_alloc(64, K * 8);
    double *z1r = (double *)r7_aligned_alloc(64, K * 8), *z1i = (double *)r7_aligned_alloc(64, K * 8);
    double *z2r = (double *)r7_aligned_alloc(64, K * 8), *z2i = (double *)r7_aligned_alloc(64, K * 8);
    double *z3r = (double *)r7_aligned_alloc(64, K * 8), *z3i = (double *)r7_aligned_alloc(64, K * 8);
    double *z4r = (double *)r7_aligned_alloc(64, K * 8), *z4i = (double *)r7_aligned_alloc(64, K * 8);
    double *z5r = (double *)r7_aligned_alloc(64, K * 8), *z5i = (double *)r7_aligned_alloc(64, K * 8);
    double *z6r = (double *)r7_aligned_alloc(64, K * 8), *z6i = (double *)r7_aligned_alloc(64, K * 8);

    double *tw1r = (double *)r7_aligned_alloc(64, K * 8), *tw1i = (double *)r7_aligned_alloc(64, K * 8);
    double *tw2r = (double *)r7_aligned_alloc(64, K * 8), *tw2i = (double *)r7_aligned_alloc(64, K * 8);
    double *tw3r = (double *)r7_aligned_alloc(64, K * 8), *tw3i = (double *)r7_aligned_alloc(64, K * 8);

    fill(a_re, K, 100 + K);
    fill(a_im, K, 200 + K);
    fill(b_re, K, 300 + K);
    fill(b_im, K, 400 + K);
    fill(c_re, K, 500 + K);
    fill(c_im, K, 600 + K);
    fill(d_re, K, 700 + K);
    fill(d_im, K, 800 + K);
    fill(e_re, K, 900 + K);
    fill(e_im, K, 1000 + K);
    fill(f_re, K, 1100 + K);
    fill(f_im, K, 1200 + K);
    fill(g_re, K, 1300 + K);
    fill(g_im, K, 1400 + K);

    /* Generate unit-modulus twiddles */
    for (int k = 0; k < K; k++)
    {
        double ang1 = 2.0 * M_PI * k / (7.0 * K);
        tw1r[k] = cos(ang1);
        tw1i[k] = sin(ang1);
        double ang2 = 2.0 * ang1;
        tw2r[k] = cos(ang2);
        tw2i[k] = sin(ang2);
        double ang3 = 3.0 * ang1;
        tw3r[k] = cos(ang3);
        tw3i[k] = sin(ang3);
    }

    /* Forward */
    radix7_rader_fwd_avx512(a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                            e_re, e_im, f_re, f_im, g_re, g_im,
                            y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i,
                            y4r, y4i, y5r, y5i, y6r, y6i,
                            tw1r, tw1i, tw2r, tw2i, tw3r, tw3i, K);

    /* Backward */
    radix7_rader_bwd_avx512(y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i,
                            y4r, y4i, y5r, y5i, y6r, y6i,
                            z0r, z0i, z1r, z1i, z2r, z2i, z3r, z3i,
                            z4r, z4i, z5r, z5i, z6r, z6i,
                            tw1r, tw1i, tw2r, tw2i, tw3r, tw3i, K);

    /* Check z[j]/7 == original */
    double mx = 0.0;
    for (int k = 0; k < K; k++)
    {
        const double *orig[7] = {a_re, b_re, c_re, d_re, e_re, f_re, g_re};
        const double *oimi[7] = {a_im, b_im, c_im, d_im, e_im, f_im, g_im};
        double *zr[7] = {z0r, z1r, z2r, z3r, z4r, z5r, z6r};
        double *zi[7] = {z0i, z1i, z2i, z3i, z4i, z5i, z6i};
        for (int j = 0; j < 7; j++)
        {
            double er = fabs(zr[j][k] / 7.0 - orig[j][k]);
            double ei = fabs(zi[j][k] / 7.0 - oimi[j][k]);
            if (er > mx)
                mx = er;
            if (ei > mx)
                mx = ei;
        }
    }

    char label[80];
    snprintf(label, sizeof(label), "twiddled roundtrip K=%d", K);
    check(label, mx, 3.0e-15);

    r7_aligned_free(a_re);
    r7_aligned_free(a_im);
    r7_aligned_free(b_re);
    r7_aligned_free(b_im);
    r7_aligned_free(c_re);
    r7_aligned_free(c_im);
    r7_aligned_free(d_re);
    r7_aligned_free(d_im);
    r7_aligned_free(e_re);
    r7_aligned_free(e_im);
    r7_aligned_free(f_re);
    r7_aligned_free(f_im);
    r7_aligned_free(g_re);
    r7_aligned_free(g_im);
    r7_aligned_free(y0r);
    r7_aligned_free(y0i);
    r7_aligned_free(y1r);
    r7_aligned_free(y1i);
    r7_aligned_free(y2r);
    r7_aligned_free(y2i);
    r7_aligned_free(y3r);
    r7_aligned_free(y3i);
    r7_aligned_free(y4r);
    r7_aligned_free(y4i);
    r7_aligned_free(y5r);
    r7_aligned_free(y5i);
    r7_aligned_free(y6r);
    r7_aligned_free(y6i);
    r7_aligned_free(z0r);
    r7_aligned_free(z0i);
    r7_aligned_free(z1r);
    r7_aligned_free(z1i);
    r7_aligned_free(z2r);
    r7_aligned_free(z2i);
    r7_aligned_free(z3r);
    r7_aligned_free(z3i);
    r7_aligned_free(z4r);
    r7_aligned_free(z4i);
    r7_aligned_free(z5r);
    r7_aligned_free(z5i);
    r7_aligned_free(z6r);
    r7_aligned_free(z6i);
    r7_aligned_free(tw1r);
    r7_aligned_free(tw1i);
    r7_aligned_free(tw2r);
    r7_aligned_free(tw2i);
    r7_aligned_free(tw3r);
    r7_aligned_free(tw3i);
}

/* ------------------------------------------------------------------ */
/*  AVX-512 vs scalar cross-check                                      */
/* ------------------------------------------------------------------ */

static void test_cross_check(int K)
{
    double *a_re = (double *)r7_aligned_alloc(64, K * 8), *a_im = (double *)r7_aligned_alloc(64, K * 8);
    double *b_re = (double *)r7_aligned_alloc(64, K * 8), *b_im = (double *)r7_aligned_alloc(64, K * 8);
    double *c_re = (double *)r7_aligned_alloc(64, K * 8), *c_im = (double *)r7_aligned_alloc(64, K * 8);
    double *d_re = (double *)r7_aligned_alloc(64, K * 8), *d_im = (double *)r7_aligned_alloc(64, K * 8);
    double *e_re = (double *)r7_aligned_alloc(64, K * 8), *e_im = (double *)r7_aligned_alloc(64, K * 8);
    double *f_re = (double *)r7_aligned_alloc(64, K * 8), *f_im = (double *)r7_aligned_alloc(64, K * 8);
    double *g_re = (double *)r7_aligned_alloc(64, K * 8), *g_im = (double *)r7_aligned_alloc(64, K * 8);

    double *sy0r = (double *)r7_aligned_alloc(64, K * 8), *sy0i = (double *)r7_aligned_alloc(64, K * 8);
    double *sy1r = (double *)r7_aligned_alloc(64, K * 8), *sy1i = (double *)r7_aligned_alloc(64, K * 8);
    double *sy2r = (double *)r7_aligned_alloc(64, K * 8), *sy2i = (double *)r7_aligned_alloc(64, K * 8);
    double *sy3r = (double *)r7_aligned_alloc(64, K * 8), *sy3i = (double *)r7_aligned_alloc(64, K * 8);
    double *sy4r = (double *)r7_aligned_alloc(64, K * 8), *sy4i = (double *)r7_aligned_alloc(64, K * 8);
    double *sy5r = (double *)r7_aligned_alloc(64, K * 8), *sy5i = (double *)r7_aligned_alloc(64, K * 8);
    double *sy6r = (double *)r7_aligned_alloc(64, K * 8), *sy6i = (double *)r7_aligned_alloc(64, K * 8);

    double *vy0r = (double *)r7_aligned_alloc(64, K * 8), *vy0i = (double *)r7_aligned_alloc(64, K * 8);
    double *vy1r = (double *)r7_aligned_alloc(64, K * 8), *vy1i = (double *)r7_aligned_alloc(64, K * 8);
    double *vy2r = (double *)r7_aligned_alloc(64, K * 8), *vy2i = (double *)r7_aligned_alloc(64, K * 8);
    double *vy3r = (double *)r7_aligned_alloc(64, K * 8), *vy3i = (double *)r7_aligned_alloc(64, K * 8);
    double *vy4r = (double *)r7_aligned_alloc(64, K * 8), *vy4i = (double *)r7_aligned_alloc(64, K * 8);
    double *vy5r = (double *)r7_aligned_alloc(64, K * 8), *vy5i = (double *)r7_aligned_alloc(64, K * 8);
    double *vy6r = (double *)r7_aligned_alloc(64, K * 8), *vy6i = (double *)r7_aligned_alloc(64, K * 8);

    double *tw1r = (double *)r7_aligned_alloc(64, K * 8), *tw1i = (double *)r7_aligned_alloc(64, K * 8);
    double *tw2r = (double *)r7_aligned_alloc(64, K * 8), *tw2i = (double *)r7_aligned_alloc(64, K * 8);
    double *tw3r = (double *)r7_aligned_alloc(64, K * 8), *tw3i = (double *)r7_aligned_alloc(64, K * 8);

    fill(a_re, K, 2000 + K);
    fill(a_im, K, 2100 + K);
    fill(b_re, K, 2200 + K);
    fill(b_im, K, 2300 + K);
    fill(c_re, K, 2400 + K);
    fill(c_im, K, 2500 + K);
    fill(d_re, K, 2600 + K);
    fill(d_im, K, 2700 + K);
    fill(e_re, K, 2800 + K);
    fill(e_im, K, 2900 + K);
    fill(f_re, K, 3000 + K);
    fill(f_im, K, 3100 + K);
    fill(g_re, K, 3200 + K);
    fill(g_im, K, 3300 + K);

    for (int k = 0; k < K; k++)
    {
        double a1 = 2.0 * M_PI * k / (7.0 * K);
        tw1r[k] = cos(a1);
        tw1i[k] = sin(a1);
        tw2r[k] = cos(2 * a1);
        tw2i[k] = sin(2 * a1);
        tw3r[k] = cos(3 * a1);
        tw3i[k] = sin(3 * a1);
    }

    /* Scalar reference */
    radix7_rader_fwd_scalar_1(a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                              e_re, e_im, f_re, f_im, g_re, g_im,
                              sy0r, sy0i, sy1r, sy1i, sy2r, sy2i, sy3r, sy3i,
                              sy4r, sy4i, sy5r, sy5i, sy6r, sy6i,
                              tw1r, tw1i, tw2r, tw2i, tw3r, tw3i, K);

    /* AVX-512 */
    radix7_rader_fwd_avx512(a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                            e_re, e_im, f_re, f_im, g_re, g_im,
                            vy0r, vy0i, vy1r, vy1i, vy2r, vy2i, vy3r, vy3i,
                            vy4r, vy4i, vy5r, vy5i, vy6r, vy6i,
                            tw1r, tw1i, tw2r, tw2i, tw3r, tw3i, K);

    double mx = 0.0;
    double *sr[7] = {sy0r, sy1r, sy2r, sy3r, sy4r, sy5r, sy6r};
    double *si[7] = {sy0i, sy1i, sy2i, sy3i, sy4i, sy5i, sy6i};
    double *vr[7] = {vy0r, vy1r, vy2r, vy3r, vy4r, vy5r, vy6r};
    double *vi[7] = {vy0i, vy1i, vy2i, vy3i, vy4i, vy5i, vy6i};
    for (int j = 0; j < 7; j++)
    {
        double er = maxerr_arr(sr[j], vr[j], K);
        double ei = maxerr_arr(si[j], vi[j], K);
        if (er > mx)
            mx = er;
        if (ei > mx)
            mx = ei;
    }

    char label[80];
    snprintf(label, sizeof(label), "fwd cross K=%d (512 vs scalar)", K);
    check(label, mx, 2.0e-15);

    /* Backward cross-check */
    radix7_rader_bwd_scalar_1(a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                              e_re, e_im, f_re, f_im, g_re, g_im,
                              sy0r, sy0i, sy1r, sy1i, sy2r, sy2i, sy3r, sy3i,
                              sy4r, sy4i, sy5r, sy5i, sy6r, sy6i,
                              tw1r, tw1i, tw2r, tw2i, tw3r, tw3i, K);

    radix7_rader_bwd_avx512(a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                            e_re, e_im, f_re, f_im, g_re, g_im,
                            vy0r, vy0i, vy1r, vy1i, vy2r, vy2i, vy3r, vy3i,
                            vy4r, vy4i, vy5r, vy5i, vy6r, vy6i,
                            tw1r, tw1i, tw2r, tw2i, tw3r, tw3i, K);

    mx = 0.0;
    for (int j = 0; j < 7; j++)
    {
        double er = maxerr_arr(sr[j], vr[j], K);
        double ei = maxerr_arr(si[j], vi[j], K);
        if (er > mx)
            mx = er;
        if (ei > mx)
            mx = ei;
    }
    snprintf(label, sizeof(label), "bwd cross K=%d (512 vs scalar)", K);
    check(label, mx, 2.0e-15);

    r7_aligned_free(a_re);
    r7_aligned_free(a_im);
    r7_aligned_free(b_re);
    r7_aligned_free(b_im);
    r7_aligned_free(c_re);
    r7_aligned_free(c_im);
    r7_aligned_free(d_re);
    r7_aligned_free(d_im);
    r7_aligned_free(e_re);
    r7_aligned_free(e_im);
    r7_aligned_free(f_re);
    r7_aligned_free(f_im);
    r7_aligned_free(g_re);
    r7_aligned_free(g_im);
    r7_aligned_free(sy0r);
    r7_aligned_free(sy0i);
    r7_aligned_free(sy1r);
    r7_aligned_free(sy1i);
    r7_aligned_free(sy2r);
    r7_aligned_free(sy2i);
    r7_aligned_free(sy3r);
    r7_aligned_free(sy3i);
    r7_aligned_free(sy4r);
    r7_aligned_free(sy4i);
    r7_aligned_free(sy5r);
    r7_aligned_free(sy5i);
    r7_aligned_free(sy6r);
    r7_aligned_free(sy6i);
    r7_aligned_free(vy0r);
    r7_aligned_free(vy0i);
    r7_aligned_free(vy1r);
    r7_aligned_free(vy1i);
    r7_aligned_free(vy2r);
    r7_aligned_free(vy2i);
    r7_aligned_free(vy3r);
    r7_aligned_free(vy3i);
    r7_aligned_free(vy4r);
    r7_aligned_free(vy4i);
    r7_aligned_free(vy5r);
    r7_aligned_free(vy5i);
    r7_aligned_free(vy6r);
    r7_aligned_free(vy6i);
    r7_aligned_free(tw1r);
    r7_aligned_free(tw1i);
    r7_aligned_free(tw2r);
    r7_aligned_free(tw2i);
    r7_aligned_free(tw3r);
    r7_aligned_free(tw3i);
}

int main(void)
{
    printf("=== Rader Radix-7 AVX-512 Tests ===\n\n");

    printf("--- Twiddled fwd→bwd roundtrip ---\n");
    /* K values chosen to exercise: U2 body only, U2+U1, U2+U1+scalar, scalar only */
    int rt_Ks[] = {1, 3, 7, 8, 15, 16, 17, 24, 31, 32, 48, 64, 100, 128, 256};
    for (int i = 0; i < (int)(sizeof(rt_Ks) / sizeof(rt_Ks[0])); i++)
        test_twiddled_roundtrip(rt_Ks[i]);

    printf("\n--- AVX-512 vs scalar cross-check ---\n");
    /* Same K values */
    for (int i = 0; i < (int)(sizeof(rt_Ks) / sizeof(rt_Ks[0])); i++)
        test_cross_check(rt_Ks[i]);

    printf("\n--- N1 fwd→bwd roundtrip ---\n");
    for (int i = 0; i < (int)(sizeof(rt_Ks) / sizeof(rt_Ks[0])); i++)
    {
        int K = rt_Ks[i];
        double *ar = (double *)r7_aligned_alloc(64, K * 8), *ai = (double *)r7_aligned_alloc(64, K * 8);
        double *br = (double *)r7_aligned_alloc(64, K * 8), *bi = (double *)r7_aligned_alloc(64, K * 8);
        double *cr = (double *)r7_aligned_alloc(64, K * 8), *ci = (double *)r7_aligned_alloc(64, K * 8);
        double *dr = (double *)r7_aligned_alloc(64, K * 8), *di = (double *)r7_aligned_alloc(64, K * 8);
        double *er = (double *)r7_aligned_alloc(64, K * 8), *ei = (double *)r7_aligned_alloc(64, K * 8);
        double *fr = (double *)r7_aligned_alloc(64, K * 8), *fi = (double *)r7_aligned_alloc(64, K * 8);
        double *gr = (double *)r7_aligned_alloc(64, K * 8), *gi = (double *)r7_aligned_alloc(64, K * 8);
        double *y0r = (double *)r7_aligned_alloc(64, K * 8), *y0i = (double *)r7_aligned_alloc(64, K * 8);
        double *y1r = (double *)r7_aligned_alloc(64, K * 8), *y1i = (double *)r7_aligned_alloc(64, K * 8);
        double *y2r = (double *)r7_aligned_alloc(64, K * 8), *y2i = (double *)r7_aligned_alloc(64, K * 8);
        double *y3r = (double *)r7_aligned_alloc(64, K * 8), *y3i = (double *)r7_aligned_alloc(64, K * 8);
        double *y4r = (double *)r7_aligned_alloc(64, K * 8), *y4i = (double *)r7_aligned_alloc(64, K * 8);
        double *y5r = (double *)r7_aligned_alloc(64, K * 8), *y5i = (double *)r7_aligned_alloc(64, K * 8);
        double *y6r = (double *)r7_aligned_alloc(64, K * 8), *y6i = (double *)r7_aligned_alloc(64, K * 8);
        double *z0r = (double *)r7_aligned_alloc(64, K * 8), *z0i = (double *)r7_aligned_alloc(64, K * 8);
        double *z1r = (double *)r7_aligned_alloc(64, K * 8), *z1i = (double *)r7_aligned_alloc(64, K * 8);
        double *z2r = (double *)r7_aligned_alloc(64, K * 8), *z2i = (double *)r7_aligned_alloc(64, K * 8);
        double *z3r = (double *)r7_aligned_alloc(64, K * 8), *z3i = (double *)r7_aligned_alloc(64, K * 8);
        double *z4r = (double *)r7_aligned_alloc(64, K * 8), *z4i = (double *)r7_aligned_alloc(64, K * 8);
        double *z5r = (double *)r7_aligned_alloc(64, K * 8), *z5i = (double *)r7_aligned_alloc(64, K * 8);
        double *z6r = (double *)r7_aligned_alloc(64, K * 8), *z6i = (double *)r7_aligned_alloc(64, K * 8);
        fill(ar, K, 5000 + K);
        fill(ai, K, 5100 + K);
        fill(br, K, 5200 + K);
        fill(bi, K, 5300 + K);
        fill(cr, K, 5400 + K);
        fill(ci, K, 5500 + K);
        fill(dr, K, 5600 + K);
        fill(di, K, 5700 + K);
        fill(er, K, 5800 + K);
        fill(ei, K, 5900 + K);
        fill(fr, K, 6000 + K);
        fill(fi, K, 6100 + K);
        fill(gr, K, 6200 + K);
        fill(gi, K, 6300 + K);
        radix7_rader_fwd_avx512_N1(ar, ai, br, bi, cr, ci, dr, di, er, ei, fr, fi, gr, gi,
                                   y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, y4r, y4i, y5r, y5i, y6r, y6i, K);
        radix7_rader_bwd_avx512_N1(y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i, y4r, y4i, y5r, y5i, y6r, y6i,
                                   z0r, z0i, z1r, z1i, z2r, z2i, z3r, z3i, z4r, z4i, z5r, z5i, z6r, z6i, K);
        double mx = 0.0;
        double *origs_r[7] = {ar, br, cr, dr, er, fr, gr};
        double *origs_i[7] = {ai, bi, ci, di, ei, fi, gi};
        double *zrs[7] = {z0r, z1r, z2r, z3r, z4r, z5r, z6r};
        double *zis[7] = {z0i, z1i, z2i, z3i, z4i, z5i, z6i};
        for (int k2 = 0; k2 < K; k2++)
            for (int j = 0; j < 7; j++)
            {
                double e1 = fabs(zrs[j][k2] / 7.0 - origs_r[j][k2]);
                double e2 = fabs(zis[j][k2] / 7.0 - origs_i[j][k2]);
                if (e1 > mx)
                    mx = e1;
                if (e2 > mx)
                    mx = e2;
            }
        char label[80];
        snprintf(label, 80, "N1 roundtrip K=%d", K);
        check(label, mx, 3.0e-15);
        r7_aligned_free(ar);
        r7_aligned_free(ai);
        r7_aligned_free(br);
        r7_aligned_free(bi);
        r7_aligned_free(cr);
        r7_aligned_free(ci);
        r7_aligned_free(dr);
        r7_aligned_free(di);
        r7_aligned_free(er);
        r7_aligned_free(ei);
        r7_aligned_free(fr);
        r7_aligned_free(fi);
        r7_aligned_free(gr);
        r7_aligned_free(gi);
        r7_aligned_free(y0r);
        r7_aligned_free(y0i);
        r7_aligned_free(y1r);
        r7_aligned_free(y1i);
        r7_aligned_free(y2r);
        r7_aligned_free(y2i);
        r7_aligned_free(y3r);
        r7_aligned_free(y3i);
        r7_aligned_free(y4r);
        r7_aligned_free(y4i);
        r7_aligned_free(y5r);
        r7_aligned_free(y5i);
        r7_aligned_free(y6r);
        r7_aligned_free(y6i);
        r7_aligned_free(z0r);
        r7_aligned_free(z0i);
        r7_aligned_free(z1r);
        r7_aligned_free(z1i);
        r7_aligned_free(z2r);
        r7_aligned_free(z2i);
        r7_aligned_free(z3r);
        r7_aligned_free(z3i);
        r7_aligned_free(z4r);
        r7_aligned_free(z4i);
        r7_aligned_free(z5r);
        r7_aligned_free(z5i);
        r7_aligned_free(z6r);
        r7_aligned_free(z6i);
    }

    printf("\n=== Results: %d PASS, %d FAIL ===\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}