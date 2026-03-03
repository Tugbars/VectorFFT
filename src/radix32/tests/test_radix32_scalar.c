/**
 * @file test_radix32_scalar.c
 * @brief Comprehensive test suite for scalar fused single-pass 4×8 radix-32
 *
 * Tests:
 *   1. Scalar ≡ AVX2 (bit-exact match, forward + backward)
 *   2. Trivial-twiddle roundtrip (butterfly correctness)
 *   3. Linearity: F(a·x + b·y) = a·F(x) + b·F(y)
 *   4. Parseval energy: Σ|X|² = 32·Σ|x|²
 *   5. All K values × both BLOCKED8/BLOCKED4 modes
 */
#include "fft_radix32_avx2.h"
#include "fft_radix32_scalar.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static double *aa(size_t n) {
    void *p = NULL; posix_memalign(&p, 64, n * sizeof(double));
    memset(p, 0, n * sizeof(double)); return (double *)p;
}

static void fill_rand(double *buf, size_t n, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < n; i++)
        buf[i] = 2.0 * (double)rand() / RAND_MAX - 1.0;
}

/* ========================================================================
 * TWIDDLE GENERATION
 * ======================================================================== */

static void gen_p1(size_t K, double *re, double *im) {
    for (size_t k = 0; k < K; k++) {
        double a = -2.0 * M_PI * k / (32.0 * K);
        re[0*K+k] = cos(a);   im[0*K+k] = sin(a);
        re[1*K+k] = cos(2*a); im[1*K+k] = sin(2*a);
    }
}

static void gen_p2(size_t K, double **re, double **im) {
    for (int j = 0; j < 8; j++) {
        double base = -2.0 * M_PI * (j + 1) / (8.0 * K);
        for (size_t k = 0; k < K; k++) {
            double a = base * k;
            re[j][k] = cos(a); im[j][k] = sin(a);
        }
    }
}

typedef struct {
    double *p1r, *p1i, *p2r[8], *p2i[8];
    radix4_dit_stage_twiddles_blocked2_t p1;
    tw_stage8_t p2;
} tw_ctx_t;

static void tw_init(size_t K, tw_mode_t mode, tw_ctx_t *t) {
    t->p1r = aa(2*K); t->p1i = aa(2*K);
    gen_p1(K, t->p1r, t->p1i);
    t->p1 = (radix4_dit_stage_twiddles_blocked2_t){.re=t->p1r, .im=t->p1i, .K=K};
    for (int j = 0; j < 8; j++) { t->p2r[j] = aa(K); t->p2i[j] = aa(K); }
    gen_p2(K, t->p2r, t->p2i);
    if (mode == TW_MODE_BLOCKED8) {
        t->p2.mode = TW_MODE_BLOCKED8;
        for (int j = 0; j < 8; j++) {
            t->p2.b8.re[j] = t->p2r[j]; t->p2.b8.im[j] = t->p2i[j];
        }
        t->p2.b8.K = K;
    } else {
        t->p2.mode = TW_MODE_BLOCKED4;
        for (int j = 0; j < 4; j++) {
            t->p2.b4.re[j] = t->p2r[j]; t->p2.b4.im[j] = t->p2i[j];
        }
        t->p2.b4.K = K;
    }
}

static void tw_init_trivial(size_t K, tw_ctx_t *t) {
    t->p1r = aa(2*K); t->p1i = aa(2*K);
    for (size_t i = 0; i < 2*K; i++) { t->p1r[i] = 1.0; t->p1i[i] = 0.0; }
    t->p1 = (radix4_dit_stage_twiddles_blocked2_t){.re=t->p1r, .im=t->p1i, .K=K};
    for (int j = 0; j < 8; j++) {
        t->p2r[j] = aa(K); t->p2i[j] = aa(K);
        for (size_t k = 0; k < K; k++) { t->p2r[j][k] = 1.0; t->p2i[j][k] = 0.0; }
    }
    t->p2.mode = TW_MODE_BLOCKED8;
    for (int j = 0; j < 8; j++) {
        t->p2.b8.re[j] = t->p2r[j]; t->p2.b8.im[j] = t->p2i[j];
    }
    t->p2.b8.K = K;
}

static void tw_free(tw_ctx_t *t) {
    free(t->p1r); free(t->p1i);
    for (int j = 0; j < 8; j++) { free(t->p2r[j]); free(t->p2i[j]); }
}

/* ========================================================================
 * TEST 1: Scalar ≡ AVX2 (bit-exact match)
 * ======================================================================== */

static int test_vs_avx2(size_t K, tw_mode_t mode) {
    size_t n = 32 * K;
    double *in_re = aa(n), *in_im = aa(n);
    double *sr = aa(n), *si = aa(n), *ar = aa(n), *ai = aa(n);
    double *sbr = aa(n), *sbi = aa(n), *abr = aa(n), *abi = aa(n);
    double *tr = aa(n), *ti = aa(n);
    tw_ctx_t tw;

    fill_rand(in_re, n, 100 + (unsigned)K + (unsigned)mode);
    fill_rand(in_im, n, 200 + (unsigned)K + (unsigned)mode);
    tw_init(K, mode, &tw);

    radix32_stage_forward_scalar(K, in_re, in_im, sr, si, &tw.p1, &tw.p2);
    radix32_stage_forward_avx2(K, in_re, in_im, ar, ai, &tw.p1, &tw.p2, tr, ti);

    double fwd_diff = 0;
    for (size_t i = 0; i < n; i++) {
        double e = fabs(sr[i] - ar[i]); if (e > fwd_diff) fwd_diff = e;
        e = fabs(si[i] - ai[i]); if (e > fwd_diff) fwd_diff = e;
    }

    radix32_stage_backward_scalar(K, sr, si, sbr, sbi, &tw.p1, &tw.p2);
    radix32_stage_backward_avx2(K, ar, ai, abr, abi, &tw.p1, &tw.p2, tr, ti);

    double bwd_diff = 0;
    for (size_t i = 0; i < n; i++) {
        double e = fabs(sbr[i] - abr[i]); if (e > bwd_diff) bwd_diff = e;
        e = fabs(sbi[i] - abi[i]); if (e > bwd_diff) bwd_diff = e;
    }

    const char *nm = (mode == TW_MODE_BLOCKED8) ? "B8" : "B4";
    int pass = (fwd_diff == 0.0) && (bwd_diff == 0.0);
    printf("  [%s] vs_avx2 K=%-5zu  fwd=%.1e  bwd=%.1e  %s\n",
           nm, K, fwd_diff, bwd_diff, pass ? "PASS" : "FAIL");

    free(in_re); free(in_im); free(sr); free(si); free(ar); free(ai);
    free(sbr); free(sbi); free(abr); free(abi); free(tr); free(ti);
    tw_free(&tw);
    return pass;
}

/* ========================================================================
 * TEST 2: Trivial-twiddle roundtrip
 * ======================================================================== */

static int test_roundtrip(size_t K) {
    size_t n = 32 * K;
    double *in_re = aa(n), *in_im = aa(n);
    double *fwd_re = aa(n), *fwd_im = aa(n);
    double *bwd_re = aa(n), *bwd_im = aa(n);
    tw_ctx_t tw;

    fill_rand(in_re, n, 300 + (unsigned)K);
    fill_rand(in_im, n, 400 + (unsigned)K);
    tw_init_trivial(K, &tw);

    radix32_stage_forward_scalar(K, in_re, in_im, fwd_re, fwd_im, &tw.p1, &tw.p2);
    radix32_stage_backward_scalar(K, fwd_re, fwd_im, bwd_re, bwd_im, &tw.p1, &tw.p2);

    double max_err = 0;
    for (size_t i = 0; i < n; i++) {
        double e = fabs(bwd_re[i] / 32.0 - in_re[i]); if (e > max_err) max_err = e;
        e = fabs(bwd_im[i] / 32.0 - in_im[i]); if (e > max_err) max_err = e;
    }

    int pass = max_err < 1e-12;
    printf("  roundtrip K=%-5zu  err=%.2e  %s\n", K, max_err, pass ? "PASS" : "FAIL");

    free(in_re); free(in_im); free(fwd_re); free(fwd_im);
    free(bwd_re); free(bwd_im); tw_free(&tw);
    return pass;
}

/* ========================================================================
 * TEST 3: Linearity — F(a·x + b·y) = a·F(x) + b·F(y)
 * ======================================================================== */

static int test_linearity(size_t K, tw_mode_t mode) {
    size_t n = 32 * K;
    double *xr = aa(n), *xi = aa(n), *yr = aa(n), *yi = aa(n);
    double *zr = aa(n), *zi = aa(n);
    double *Fxr = aa(n), *Fxi = aa(n), *Fyr = aa(n), *Fyi = aa(n);
    double *Fzr = aa(n), *Fzi = aa(n);
    tw_ctx_t tw;

    fill_rand(xr, n, 500 + (unsigned)K); fill_rand(xi, n, 501 + (unsigned)K);
    fill_rand(yr, n, 502 + (unsigned)K); fill_rand(yi, n, 503 + (unsigned)K);
    tw_init(K, mode, &tw);

    double a = 1.7, b = -0.3;
    for (size_t i = 0; i < n; i++) {
        zr[i] = a * xr[i] + b * yr[i];
        zi[i] = a * xi[i] + b * yi[i];
    }

    radix32_stage_forward_scalar(K, xr, xi, Fxr, Fxi, &tw.p1, &tw.p2);
    radix32_stage_forward_scalar(K, yr, yi, Fyr, Fyi, &tw.p1, &tw.p2);
    radix32_stage_forward_scalar(K, zr, zi, Fzr, Fzi, &tw.p1, &tw.p2);

    double max_err = 0;
    for (size_t i = 0; i < n; i++) {
        double expect_r = a * Fxr[i] + b * Fyr[i];
        double expect_i = a * Fxi[i] + b * Fyi[i];
        double e = fabs(Fzr[i] - expect_r); if (e > max_err) max_err = e;
        e = fabs(Fzi[i] - expect_i); if (e > max_err) max_err = e;
    }

    const char *nm = (mode == TW_MODE_BLOCKED8) ? "B8" : "B4";
    int pass = max_err < 1e-10;
    printf("  [%s] linearity K=%-5zu  err=%.2e  %s\n",
           nm, K, max_err, pass ? "PASS" : "FAIL");

    free(xr); free(xi); free(yr); free(yi); free(zr); free(zi);
    free(Fxr); free(Fxi); free(Fyr); free(Fyi); free(Fzr); free(Fzi);
    tw_free(&tw);
    return pass;
}

/* ========================================================================
 * TEST 4: Parseval energy — Σ|X|² = 32·Σ|x|²
 * ======================================================================== */

static int test_parseval(size_t K, tw_mode_t mode) {
    size_t n = 32 * K;
    double *xr = aa(n), *xi = aa(n);
    double *Xr = aa(n), *Xi = aa(n);
    tw_ctx_t tw;

    fill_rand(xr, n, 600 + (unsigned)K); fill_rand(xi, n, 601 + (unsigned)K);
    tw_init_trivial(K, &tw);

    radix32_stage_forward_scalar(K, xr, xi, Xr, Xi, &tw.p1, &tw.p2);

    /* Per-k energy check */
    double max_rel = 0;
    for (size_t k = 0; k < K; k++) {
        double ein = 0, eout = 0;
        for (int s = 0; s < 32; s++) {
            ein  += xr[s*K+k]*xr[s*K+k] + xi[s*K+k]*xi[s*K+k];
            eout += Xr[s*K+k]*Xr[s*K+k] + Xi[s*K+k]*Xi[s*K+k];
        }
        double rel = fabs(eout - 32.0 * ein) / (32.0 * ein + 1e-300);
        if (rel > max_rel) max_rel = rel;
    }

    int pass = max_rel < 1e-12;
    printf("  parseval K=%-5zu  rel_err=%.2e  %s\n", K, max_rel, pass ? "PASS" : "FAIL");

    free(xr); free(xi); free(Xr); free(Xi); tw_free(&tw);
    return pass;
}

/* ========================================================================
 * MAIN
 * ======================================================================== */

int main(void) {
    int ok = 1, total = 0, passed = 0;

    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  Scalar Fused Single-Pass 4×8 Radix-32 — Test Suite    ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    /* Test 1: Scalar ≡ AVX2 */
    printf("── Test 1: Scalar ≡ AVX2 (bit-exact) ──\n");
    size_t b8_Ks[] = {8, 16, 32, 64, 128, 256};
    size_t b4_Ks[] = {8, 64, 256, 512, 1024};
    for (int i = 0; i < 6; i++) { int p = test_vs_avx2(b8_Ks[i], TW_MODE_BLOCKED8); ok &= p; total++; passed += p; }
    for (int i = 0; i < 5; i++) { int p = test_vs_avx2(b4_Ks[i], TW_MODE_BLOCKED4); ok &= p; total++; passed += p; }

    /* Test 2: Roundtrip */
    printf("\n── Test 2: Trivial-twiddle roundtrip ──\n");
    size_t rt_Ks[] = {4, 8, 16, 32, 64, 128, 256, 512};
    for (int i = 0; i < 8; i++) { int p = test_roundtrip(rt_Ks[i]); ok &= p; total++; passed += p; }

    /* Test 3: Linearity */
    printf("\n── Test 3: Linearity ──\n");
    for (int i = 0; i < 4; i++) { int p = test_linearity(b8_Ks[i], TW_MODE_BLOCKED8); ok &= p; total++; passed += p; }
    for (int i = 0; i < 3; i++) { int p = test_linearity(b4_Ks[i], TW_MODE_BLOCKED4); ok &= p; total++; passed += p; }

    /* Test 4: Parseval */
    printf("\n── Test 4: Parseval energy ──\n");
    for (int i = 0; i < 6; i++) { int p = test_parseval(b8_Ks[i], TW_MODE_BLOCKED8); ok &= p; total++; passed += p; }

    printf("\n══════════════════════════════════════════\n");
    printf("  %d/%d passed  %s\n", passed, total, ok ? "✓ ALL PASSED" : "✗ FAILURES");
    printf("══════════════════════════════════════════\n");
    return ok ? 0 : 1;
}
