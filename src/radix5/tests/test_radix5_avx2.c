/**
 * @file  test_radix5_avx2.c
 * @brief Tests for radix-5 WFTA AVX2 butterfly kernels
 *
 * Tests:
 *   1. N1 forward vs naive DFT-5  (correctness)
 *   2. N1 forward → backward roundtrip  (fwd then bwd / 5 == identity)
 *   3. Twiddled forward vs naive DFT-5  (correctness)
 *   4. Twiddled forward → backward roundtrip
 *   5. Scalar tail (K not multiple of 4)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "fft_radix5_avx2.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define TOL 1e-12

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, fmt, ...) do {                                      \
    if (!(cond)) {                                                      \
        printf("  FAIL: " fmt "\n", ##__VA_ARGS__);                     \
        g_fail++;                                                       \
    } else { g_pass++; }                                                \
} while (0)

/* ── Naive DFT-5 (forward, no twiddles) ── */
static void naive_dft5_fwd(const double *xr, const double *xi,
                            double *yr, double *yi)
{
    for (int k = 0; k < 5; k++) {
        yr[k] = 0.0;  yi[k] = 0.0;
        for (int n = 0; n < 5; n++) {
            double angle = -2.0 * M_PI * k * n / 5.0;
            yr[k] += xr[n] * cos(angle) - xi[n] * sin(angle);
            yi[k] += xr[n] * sin(angle) + xi[n] * cos(angle);
        }
    }
}

/* ── Naive DFT-5 (backward, no twiddles) ── */
static void naive_dft5_bwd(const double *xr, const double *xi,
                            double *yr, double *yi)
{
    for (int k = 0; k < 5; k++) {
        yr[k] = 0.0;  yi[k] = 0.0;
        for (int n = 0; n < 5; n++) {
            double angle = +2.0 * M_PI * k * n / 5.0;
            yr[k] += xr[n] * cos(angle) - xi[n] * sin(angle);
            yi[k] += xr[n] * sin(angle) + xi[n] * cos(angle);
        }
    }
}

/* ── Test 1: N1 forward correctness ── */
static void test_n1_fwd_correctness(void)
{
    printf("Test 1: N1 forward vs naive DFT-5\n");
    const int K = 1;

    /* SoA with stride K=1: each leg is a single complex number */
    double a_re[] = {1.0}, a_im[] = {0.5};
    double b_re[] = {2.0}, b_im[] = {-1.0};
    double c_re[] = {-0.5}, c_im[] = {3.0};
    double d_re[] = {0.7}, d_im[] = {-2.2};
    double e_re[] = {1.3}, e_im[] = {0.8};

    double y0r[1], y0i[1], y1r[1], y1i[1], y2r[1], y2i[1];
    double y3r[1], y3i[1], y4r[1], y4i[1];

    radix5_wfta_fwd_avx2_N1(a_re, a_im, b_re, b_im, c_re, c_im,
                             d_re, d_im, e_re, e_im,
                             y0r, y0i, y1r, y1i, y2r, y2i,
                             y3r, y3i, y4r, y4i, K);

    /* Naive reference */
    double xr[5] = {a_re[0], b_re[0], c_re[0], d_re[0], e_re[0]};
    double xi[5] = {a_im[0], b_im[0], c_im[0], d_im[0], e_im[0]};
    double ref_r[5], ref_i[5];
    naive_dft5_fwd(xr, xi, ref_r, ref_i);

    double *yr[] = {y0r, y1r, y2r, y3r, y4r};
    double *yi[] = {y0i, y1i, y2i, y3i, y4i};
    for (int j = 0; j < 5; j++) {
        double er = fabs(yr[j][0] - ref_r[j]);
        double ei = fabs(yi[j][0] - ref_i[j]);
        CHECK(er < TOL && ei < TOL,
              "y[%d]: got (%.15f, %.15f) expected (%.15f, %.15f) err=(%e,%e)",
              j, yr[j][0], yi[j][0], ref_r[j], ref_i[j], er, ei);
    }
}

/* ── Test 2: N1 roundtrip (fwd then bwd / 5 == identity) ── */
static void test_n1_roundtrip(void)
{
    printf("Test 2: N1 fwd -> bwd roundtrip\n");
    const int K = 8;  /* Tests AVX2 path + no tail */

    double a_re[8], a_im[8], b_re[8], b_im[8], c_re[8], c_im[8];
    double d_re[8], d_im[8], e_re[8], e_im[8];

    srand(42);
    for (int i = 0; i < K; i++) {
        a_re[i] = (double)rand()/RAND_MAX - 0.5;
        a_im[i] = (double)rand()/RAND_MAX - 0.5;
        b_re[i] = (double)rand()/RAND_MAX - 0.5;
        b_im[i] = (double)rand()/RAND_MAX - 0.5;
        c_re[i] = (double)rand()/RAND_MAX - 0.5;
        c_im[i] = (double)rand()/RAND_MAX - 0.5;
        d_re[i] = (double)rand()/RAND_MAX - 0.5;
        d_im[i] = (double)rand()/RAND_MAX - 0.5;
        e_re[i] = (double)rand()/RAND_MAX - 0.5;
        e_im[i] = (double)rand()/RAND_MAX - 0.5;
    }

    double y0r[8], y0i[8], y1r[8], y1i[8], y2r[8], y2i[8];
    double y3r[8], y3i[8], y4r[8], y4i[8];

    /* Forward */
    radix5_wfta_fwd_avx2_N1(a_re, a_im, b_re, b_im, c_re, c_im,
                             d_re, d_im, e_re, e_im,
                             y0r, y0i, y1r, y1i, y2r, y2i,
                             y3r, y3i, y4r, y4i, K);

    double z0r[8], z0i[8], z1r[8], z1i[8], z2r[8], z2i[8];
    double z3r[8], z3i[8], z4r[8], z4i[8];

    /* Backward */
    radix5_wfta_bwd_avx2_N1(y0r, y0i, y1r, y1i, y2r, y2i,
                             y3r, y3i, y4r, y4i,
                             z0r, z0i, z1r, z1i, z2r, z2i,
                             z3r, z3i, z4r, z4i, K);

    /* Check: z[j]/5 == original */
    double *orig_re[] = {a_re, b_re, c_re, d_re, e_re};
    double *orig_im[] = {a_im, b_im, c_im, d_im, e_im};
    double *zr[] = {z0r, z1r, z2r, z3r, z4r};
    double *zi[] = {z0i, z1i, z2i, z3i, z4i};

    for (int j = 0; j < 5; j++) {
        for (int i = 0; i < K; i++) {
            double er = fabs(zr[j][i] / 5.0 - orig_re[j][i]);
            double ei = fabs(zi[j][i] / 5.0 - orig_im[j][i]);
            CHECK(er < TOL && ei < TOL,
                  "leg %d, k=%d: roundtrip err=(%e,%e)", j, i, er, ei);
        }
    }
}

/* ── Test 3: Twiddled forward correctness ── */
static void test_twiddled_fwd_correctness(void)
{
    printf("Test 3: Twiddled forward vs naive (K=1, stride=5)\n");

    /* Simulate a single 5-point DFT at position k in a larger FFT.
     * Twiddle W^j for leg j, stride K: W^j = exp(-2πi·j·k/(5·K))
     * For K=1, W^j = exp(-2πi·j/5) — but that's just the DFT matrix!
     * Use trivial twiddles (all 1) to compare with naive DFT-5. */
    const int K = 1;
    double a_re[] = {1.0}, a_im[] = {2.0};
    double b_re[] = {-1.0}, b_im[] = {0.5};
    double c_re[] = {0.3}, c_im[] = {-0.7};
    double d_re[] = {2.1}, d_im[] = {1.4};
    double e_re[] = {-0.8}, e_im[] = {0.9};

    /* Trivial twiddles: W1 = W2 = 1+0i */
    double tw1r[] = {1.0}, tw1i[] = {0.0};
    double tw2r[] = {1.0}, tw2i[] = {0.0};

    double y0r[1], y0i[1], y1r[1], y1i[1], y2r[1], y2i[1];
    double y3r[1], y3i[1], y4r[1], y4i[1];

    radix5_wfta_fwd_avx2(a_re, a_im, b_re, b_im, c_re, c_im,
                          d_re, d_im, e_re, e_im,
                          y0r, y0i, y1r, y1i, y2r, y2i,
                          y3r, y3i, y4r, y4i,
                          tw1r, tw1i, tw2r, tw2i, K);

    /* With trivial twiddles, should match naive DFT-5 */
    double xr[5] = {a_re[0], b_re[0], c_re[0], d_re[0], e_re[0]};
    double xi[5] = {a_im[0], b_im[0], c_im[0], d_im[0], e_im[0]};
    double ref_r[5], ref_i[5];
    naive_dft5_fwd(xr, xi, ref_r, ref_i);

    double *yr[] = {y0r, y1r, y2r, y3r, y4r};
    double *yi[] = {y0i, y1i, y2i, y3i, y4i};
    for (int j = 0; j < 5; j++) {
        double er = fabs(yr[j][0] - ref_r[j]);
        double ei = fabs(yi[j][0] - ref_i[j]);
        CHECK(er < TOL && ei < TOL,
              "y[%d]: got (%.15f, %.15f) expected (%.15f, %.15f) err=(%e,%e)",
              j, yr[j][0], yi[j][0], ref_r[j], ref_i[j], er, ei);
    }
}

/* ── Test 4: Twiddled roundtrip ── */
static void test_twiddled_roundtrip(void)
{
    printf("Test 4: Twiddled fwd -> bwd roundtrip (K=8)\n");
    const int K = 8;

    double a_re[8], a_im[8], b_re[8], b_im[8], c_re[8], c_im[8];
    double d_re[8], d_im[8], e_re[8], e_im[8];
    double tw1r[8], tw1i[8], tw2r[8], tw2i[8];

    srand(123);
    for (int i = 0; i < K; i++) {
        a_re[i] = (double)rand()/RAND_MAX - 0.5;
        a_im[i] = (double)rand()/RAND_MAX - 0.5;
        b_re[i] = (double)rand()/RAND_MAX - 0.5;
        b_im[i] = (double)rand()/RAND_MAX - 0.5;
        c_re[i] = (double)rand()/RAND_MAX - 0.5;
        c_im[i] = (double)rand()/RAND_MAX - 0.5;
        d_re[i] = (double)rand()/RAND_MAX - 0.5;
        d_im[i] = (double)rand()/RAND_MAX - 0.5;
        e_re[i] = (double)rand()/RAND_MAX - 0.5;
        e_im[i] = (double)rand()/RAND_MAX - 0.5;

        /* Generate proper twiddle factors: W^k = exp(-2πi·k / (5·K)) */
        double ang1 = -2.0 * M_PI * (double)i / (5.0 * K);
        double ang2 = -2.0 * M_PI * 2.0 * (double)i / (5.0 * K);
        tw1r[i] = cos(ang1);  tw1i[i] = sin(ang1);
        tw2r[i] = cos(ang2);  tw2i[i] = sin(ang2);
    }

    double y0r[8], y0i[8], y1r[8], y1i[8], y2r[8], y2i[8];
    double y3r[8], y3i[8], y4r[8], y4i[8];

    radix5_wfta_fwd_avx2(a_re, a_im, b_re, b_im, c_re, c_im,
                          d_re, d_im, e_re, e_im,
                          y0r, y0i, y1r, y1i, y2r, y2i,
                          y3r, y3i, y4r, y4i,
                          tw1r, tw1i, tw2r, tw2i, K);

    double z0r[8], z0i[8], z1r[8], z1i[8], z2r[8], z2i[8];
    double z3r[8], z3i[8], z4r[8], z4i[8];

    radix5_wfta_bwd_avx2(y0r, y0i, y1r, y1i, y2r, y2i,
                          y3r, y3i, y4r, y4i,
                          z0r, z0i, z1r, z1i, z2r, z2i,
                          z3r, z3i, z4r, z4i,
                          tw1r, tw1i, tw2r, tw2i, K);

    double *orig_re[] = {a_re, b_re, c_re, d_re, e_re};
    double *orig_im[] = {a_im, b_im, c_im, d_im, e_im};
    double *zr[] = {z0r, z1r, z2r, z3r, z4r};
    double *zi[] = {z0i, z1i, z2i, z3i, z4i};

    for (int j = 0; j < 5; j++) {
        for (int i = 0; i < K; i++) {
            double er = fabs(zr[j][i] / 5.0 - orig_re[j][i]);
            double ei = fabs(zi[j][i] / 5.0 - orig_im[j][i]);
            CHECK(er < TOL && ei < TOL,
                  "leg %d, k=%d: roundtrip err=(%e,%e)", j, i, er, ei);
        }
    }
}

/* ── Test 5: Scalar tail (K=7, not multiple of 4) ── */
static void test_scalar_tail(void)
{
    printf("Test 5: Scalar tail K=7 (4+3)\n");
    const int K = 7;

    double a_re[7], a_im[7], b_re[7], b_im[7], c_re[7], c_im[7];
    double d_re[7], d_im[7], e_re[7], e_im[7];

    srand(999);
    for (int i = 0; i < K; i++) {
        a_re[i] = (double)rand()/RAND_MAX - 0.5;
        a_im[i] = (double)rand()/RAND_MAX - 0.5;
        b_re[i] = (double)rand()/RAND_MAX - 0.5;
        b_im[i] = (double)rand()/RAND_MAX - 0.5;
        c_re[i] = (double)rand()/RAND_MAX - 0.5;
        c_im[i] = (double)rand()/RAND_MAX - 0.5;
        d_re[i] = (double)rand()/RAND_MAX - 0.5;
        d_im[i] = (double)rand()/RAND_MAX - 0.5;
        e_re[i] = (double)rand()/RAND_MAX - 0.5;
        e_im[i] = (double)rand()/RAND_MAX - 0.5;
    }

    double y0r[7], y0i[7], y1r[7], y1i[7], y2r[7], y2i[7];
    double y3r[7], y3i[7], y4r[7], y4i[7];

    radix5_wfta_fwd_avx2_N1(a_re, a_im, b_re, b_im, c_re, c_im,
                             d_re, d_im, e_re, e_im,
                             y0r, y0i, y1r, y1i, y2r, y2i,
                             y3r, y3i, y4r, y4i, K);

    /* Verify each of the 7 independent DFT-5s */
    for (int i = 0; i < K; i++) {
        double xr[5] = {a_re[i], b_re[i], c_re[i], d_re[i], e_re[i]};
        double xi[5] = {a_im[i], b_im[i], c_im[i], d_im[i], e_im[i]};
        double ref_r[5], ref_i[5];
        naive_dft5_fwd(xr, xi, ref_r, ref_i);

        double *yr_ptrs[] = {y0r, y1r, y2r, y3r, y4r};
        double *yi_ptrs[] = {y0i, y1i, y2i, y3i, y4i};

        for (int j = 0; j < 5; j++) {
            double er = fabs(yr_ptrs[j][i] - ref_r[j]);
            double ei = fabs(yi_ptrs[j][i] - ref_i[j]);
            CHECK(er < TOL && ei < TOL,
                  "k=%d y[%d]: err=(%e,%e)", i, j, er, ei);
        }
    }
}

/* ── Test 6: Twiddled with non-trivial twiddles, verify against manual DFT ── */
static void test_twiddled_vs_manual(void)
{
    printf("Test 6: Twiddled fwd vs manual twiddled DFT-5 (K=4)\n");
    const int K = 4;
    const int N = 5 * K;  /* = 20 */

    double a_re[4], a_im[4], b_re[4], b_im[4], c_re[4], c_im[4];
    double d_re[4], d_im[4], e_re[4], e_im[4];
    double tw1r[4], tw1i[4], tw2r[4], tw2i[4];

    srand(777);
    for (int i = 0; i < K; i++) {
        a_re[i] = (double)rand()/RAND_MAX - 0.5;
        a_im[i] = (double)rand()/RAND_MAX - 0.5;
        b_re[i] = (double)rand()/RAND_MAX - 0.5;
        b_im[i] = (double)rand()/RAND_MAX - 0.5;
        c_re[i] = (double)rand()/RAND_MAX - 0.5;
        c_im[i] = (double)rand()/RAND_MAX - 0.5;
        d_re[i] = (double)rand()/RAND_MAX - 0.5;
        d_im[i] = (double)rand()/RAND_MAX - 0.5;
        e_re[i] = (double)rand()/RAND_MAX - 0.5;
        e_im[i] = (double)rand()/RAND_MAX - 0.5;

        double ang1 = -2.0 * M_PI * (double)i / (double)N;
        double ang2 = -2.0 * M_PI * 2.0 * (double)i / (double)N;
        tw1r[i] = cos(ang1);  tw1i[i] = sin(ang1);
        tw2r[i] = cos(ang2);  tw2i[i] = sin(ang2);
    }

    double y0r[4], y0i[4], y1r[4], y1i[4], y2r[4], y2i[4];
    double y3r[4], y3i[4], y4r[4], y4i[4];

    radix5_wfta_fwd_avx2(a_re, a_im, b_re, b_im, c_re, c_im,
                          d_re, d_im, e_re, e_im,
                          y0r, y0i, y1r, y1i, y2r, y2i,
                          y3r, y3i, y4r, y4i,
                          tw1r, tw1i, tw2r, tw2i, K);

    /* Manual: for each k, apply twiddles then DFT-5 */
    for (int i = 0; i < K; i++) {
        /* Derive all 4 twiddles from W1, W2 */
        double w1r = tw1r[i], w1i = tw1i[i];
        double w2r = tw2r[i], w2i = tw2i[i];
        double w3r = w1r*w2r - w1i*w2i, w3i = w1r*w2i + w1i*w2r;
        double w4r = w2r*w2r - w2i*w2i, w4i = 2.0*w2r*w2i;

        /* Apply twiddles */
        double tb_r = b_re[i]*w1r - b_im[i]*w1i;
        double tb_i = b_re[i]*w1i + b_im[i]*w1r;
        double tc_r = c_re[i]*w2r - c_im[i]*w2i;
        double tc_i = c_re[i]*w2i + c_im[i]*w2r;
        double td_r = d_re[i]*w3r - d_im[i]*w3i;
        double td_i = d_re[i]*w3i + d_im[i]*w3r;
        double te_r = e_re[i]*w4r - e_im[i]*w4i;
        double te_i = e_re[i]*w4i + e_im[i]*w4r;

        /* Naive DFT-5 on twiddled input */
        double xr[5] = {a_re[i], tb_r, tc_r, td_r, te_r};
        double xi[5] = {a_im[i], tb_i, tc_i, td_i, te_i};
        double ref_r[5], ref_i[5];
        naive_dft5_fwd(xr, xi, ref_r, ref_i);

        double *yr[] = {y0r, y1r, y2r, y3r, y4r};
        double *yi[] = {y0i, y1i, y2i, y3i, y4i};
        for (int j = 0; j < 5; j++) {
            double er = fabs(yr[j][i] - ref_r[j]);
            double ei = fabs(yi[j][i] - ref_i[j]);
            CHECK(er < TOL && ei < TOL,
                  "k=%d y[%d]: err=(%e,%e)", i, j, er, ei);
        }
    }
}

/* forward declarations for U=2 stress tests appended at end of file */
static int test7_u2_stress_k16(void);
static int test8_u2_twiddled_k32(void);
static int test9_u2_tail_k9(void);

int main(void)
{
    printf("=== Radix-5 WFTA AVX2 U=2 Tests ===\n\n");

    test_n1_fwd_correctness();
    test_n1_roundtrip();
    test_twiddled_fwd_correctness();
    test_twiddled_roundtrip();
    test_scalar_tail();
    test_twiddled_vs_manual();

    int f7 = test7_u2_stress_k16();
    int f8 = test8_u2_twiddled_k32();
    int f9 = test9_u2_tail_k9();
    int u2_checks = (16+32+9)*5*2;
    int u2_fail = f7+f8+f9;
    g_pass += (u2_checks - u2_fail);
    g_fail += u2_fail;

    printf("\n--- Results: %d passed, %d failed ---\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}

/* ── Test 7: U=2 path stress — K=16 (2 full U=2 iterations) ── */
static int test7_u2_stress_k16(void) {
    printf("Test 7: U=2 stress K=16 fwd->bwd roundtrip\n");
    int K = 16, fails = 0;
    double a_re[16],a_im[16],b_re[16],b_im[16],c_re[16],c_im[16];
    double d_re[16],d_im[16],e_re[16],e_im[16];
    double y0r[16],y0i[16],y1r[16],y1i[16],y2r[16],y2i[16];
    double y3r[16],y3i[16],y4r[16],y4i[16];
    double z0r[16],z0i[16],z1r[16],z1i[16],z2r[16],z2i[16];
    double z3r[16],z3i[16],z4r[16],z4i[16];

    srand(12345);
    for (int i = 0; i < K; i++) {
        a_re[i] = (rand()/(double)RAND_MAX - 0.5)*10;
        a_im[i] = (rand()/(double)RAND_MAX - 0.5)*10;
        b_re[i] = (rand()/(double)RAND_MAX - 0.5)*10;
        b_im[i] = (rand()/(double)RAND_MAX - 0.5)*10;
        c_re[i] = (rand()/(double)RAND_MAX - 0.5)*10;
        c_im[i] = (rand()/(double)RAND_MAX - 0.5)*10;
        d_re[i] = (rand()/(double)RAND_MAX - 0.5)*10;
        d_im[i] = (rand()/(double)RAND_MAX - 0.5)*10;
        e_re[i] = (rand()/(double)RAND_MAX - 0.5)*10;
        e_im[i] = (rand()/(double)RAND_MAX - 0.5)*10;
    }
    radix5_wfta_fwd_avx2_N1(a_re,a_im,b_re,b_im,c_re,c_im,d_re,d_im,e_re,e_im,
                            y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,K);
    radix5_wfta_bwd_avx2_N1(y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,
                            z0r,z0i,z1r,z1i,z2r,z2i,z3r,z3i,z4r,z4i,K);
    const double *orig[5][2] = {{a_re,a_im},{b_re,b_im},{c_re,c_im},{d_re,d_im},{e_re,e_im}};
    double *out[5][2] = {{z0r,z0i},{z1r,z1i},{z2r,z2i},{z3r,z3i},{z4r,z4i}};
    for (int leg = 0; leg < 5; leg++)
        for (int i = 0; i < K; i++) {
            if (fabs(out[leg][0][i]/5.0 - orig[leg][0][i]) > 1e-10) fails++;
            if (fabs(out[leg][1][i]/5.0 - orig[leg][1][i]) > 1e-10) fails++;
        }
    return fails;
}

/* ── Test 8: U=2 twiddled K=32 (4 full U=2 iters) ── */
static int test8_u2_twiddled_k32(void) {
    printf("Test 8: U=2 twiddled K=32 fwd->bwd roundtrip\n");
    int K = 32, fails = 0;
    double a_re[32],a_im[32],b_re[32],b_im[32],c_re[32],c_im[32];
    double d_re[32],d_im[32],e_re[32],e_im[32];
    double tw1r[32],tw1i[32],tw2r[32],tw2i[32];
    double y0r[32],y0i[32],y1r[32],y1i[32],y2r[32],y2i[32];
    double y3r[32],y3i[32],y4r[32],y4i[32];
    double z0r[32],z0i[32],z1r[32],z1i[32],z2r[32],z2i[32];
    double z3r[32],z3i[32],z4r[32],z4i[32];

    srand(54321);
    int N = 5 * K;
    for (int i = 0; i < K; i++) {
        a_re[i] = (rand()/(double)RAND_MAX - 0.5)*10;
        a_im[i] = (rand()/(double)RAND_MAX - 0.5)*10;
        b_re[i] = (rand()/(double)RAND_MAX - 0.5)*10;
        b_im[i] = (rand()/(double)RAND_MAX - 0.5)*10;
        c_re[i] = (rand()/(double)RAND_MAX - 0.5)*10;
        c_im[i] = (rand()/(double)RAND_MAX - 0.5)*10;
        d_re[i] = (rand()/(double)RAND_MAX - 0.5)*10;
        d_im[i] = (rand()/(double)RAND_MAX - 0.5)*10;
        e_re[i] = (rand()/(double)RAND_MAX - 0.5)*10;
        e_im[i] = (rand()/(double)RAND_MAX - 0.5)*10;
        double angle = -2.0 * M_PI * i / N;
        tw1r[i] = cos(angle);   tw1i[i] = sin(angle);
        tw2r[i] = cos(2*angle); tw2i[i] = sin(2*angle);
    }
    radix5_wfta_fwd_avx2(a_re,a_im,b_re,b_im,c_re,c_im,d_re,d_im,e_re,e_im,
                         y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,
                         tw1r,tw1i,tw2r,tw2i,K);
    radix5_wfta_bwd_avx2(y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,
                         z0r,z0i,z1r,z1i,z2r,z2i,z3r,z3i,z4r,z4i,
                         tw1r,tw1i,tw2r,tw2i,K);
    const double *orig[5][2] = {{a_re,a_im},{b_re,b_im},{c_re,c_im},{d_re,d_im},{e_re,e_im}};
    double *out[5][2] = {{z0r,z0i},{z1r,z1i},{z2r,z2i},{z3r,z3i},{z4r,z4i}};
    for (int leg = 0; leg < 5; leg++)
        for (int i = 0; i < K; i++) {
            if (fabs(out[leg][0][i]/5.0 - orig[leg][0][i]) > 1e-10) fails++;
            if (fabs(out[leg][1][i]/5.0 - orig[leg][1][i]) > 1e-10) fails++;
        }
    return fails;
}

/* ── Test 9: K=9 (1 U=2 iter + 1 scalar tail) mixed alignment ── */
static int test9_u2_tail_k9(void) {
    printf("Test 9: K=9 (U=2 + scalar tail) fwd->bwd roundtrip\n");
    int K = 9, fails = 0;
    double a_re[9],a_im[9],b_re[9],b_im[9],c_re[9],c_im[9];
    double d_re[9],d_im[9],e_re[9],e_im[9];
    double y0r[9],y0i[9],y1r[9],y1i[9],y2r[9],y2i[9];
    double y3r[9],y3i[9],y4r[9],y4i[9];
    double z0r[9],z0i[9],z1r[9],z1i[9],z2r[9],z2i[9];
    double z3r[9],z3i[9],z4r[9],z4i[9];

    srand(99999);
    for (int i = 0; i < K; i++) {
        a_re[i] = (rand()/(double)RAND_MAX - 0.5)*10;
        a_im[i] = (rand()/(double)RAND_MAX - 0.5)*10;
        b_re[i] = (rand()/(double)RAND_MAX - 0.5)*10;
        b_im[i] = (rand()/(double)RAND_MAX - 0.5)*10;
        c_re[i] = (rand()/(double)RAND_MAX - 0.5)*10;
        c_im[i] = (rand()/(double)RAND_MAX - 0.5)*10;
        d_re[i] = (rand()/(double)RAND_MAX - 0.5)*10;
        d_im[i] = (rand()/(double)RAND_MAX - 0.5)*10;
        e_re[i] = (rand()/(double)RAND_MAX - 0.5)*10;
        e_im[i] = (rand()/(double)RAND_MAX - 0.5)*10;
    }
    radix5_wfta_fwd_avx2_N1(a_re,a_im,b_re,b_im,c_re,c_im,d_re,d_im,e_re,e_im,
                            y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,K);
    radix5_wfta_bwd_avx2_N1(y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,
                            z0r,z0i,z1r,z1i,z2r,z2i,z3r,z3i,z4r,z4i,K);
    const double *orig[5][2] = {{a_re,a_im},{b_re,b_im},{c_re,c_im},{d_re,d_im},{e_re,e_im}};
    double *out[5][2] = {{z0r,z0i},{z1r,z1i},{z2r,z2i},{z3r,z3i},{z4r,z4i}};
    for (int leg = 0; leg < 5; leg++)
        for (int i = 0; i < K; i++) {
            if (fabs(out[leg][0][i]/5.0 - orig[leg][0][i]) > 1e-10) fails++;
            if (fabs(out[leg][1][i]/5.0 - orig[leg][1][i]) > 1e-10) fails++;
        }
    return fails;
}
