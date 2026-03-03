/**
 * @file test_radix32_avx512_driver.c
 * @brief Verify AVX-512 radix-32 U=1 driver against scalar reference
 *
 * Tests:
 *   1. AVX-512 ≡ scalar (forward, multiple K)
 *   2. AVX-512 ≡ scalar (backward, multiple K)
 *   3. Forward→backward roundtrip: IFFT(FFT(x)) = 32·x
 *   4. Parseval: Σ|X|² = 32·Σ|x|²
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../fft_radix32_platform.h"

/* Scalar reference */
#include "fft_radix32_scalar.h"

/* AVX-512 under test */
#include "fft_radix32_avx512.h"

/*==========================================================================
 * Helpers
 *=========================================================================*/

static double *aa(size_t n)
{
    double *p = (double *)r32_aligned_alloc(64, n * sizeof(double));
    memset(p, 0, n * sizeof(double));
    return p;
}

static void fill_rand(double *buf, size_t n, unsigned seed)
{
    for (size_t i = 0; i < n; i++) {
        seed = seed * 1103515245 + 12345;
        buf[i] = ((double)(seed >> 16) / 32768.0) - 1.0;
    }
}

static double max_abs_diff(const double *a, const double *b, size_t n)
{
    double mx = 0;
    for (size_t i = 0; i < n; i++) {
        double e = fabs(a[i] - b[i]);
        if (e > mx) mx = e;
    }
    return mx;
}

/*==========================================================================
 * Twiddle generation (same formulas as scalar test)
 *=========================================================================*/

static void gen_p1(size_t K, double *re, double *im)
{
    for (size_t k = 0; k < K; k++) {
        double a = -2.0 * M_PI * (double)k / (32.0 * (double)K);
        re[0 * K + k] = cos(a);   im[0 * K + k] = sin(a);
        re[1 * K + k] = cos(2*a); im[1 * K + k] = sin(2*a);
    }
}

static void gen_p2(size_t K, double *re[8], double *im[8])
{
    for (int j = 0; j < 8; j++) {
        double base = -2.0 * M_PI * (double)(j + 1) / (8.0 * (double)K);
        for (size_t k = 0; k < K; k++) {
            double a = base * (double)k;
            re[j][k] = cos(a);
            im[j][k] = sin(a);
        }
    }
}

typedef struct {
    double *p1r, *p1i;
    double *p2r[8], *p2i[8];
    radix4_dit_stage_twiddles_blocked2_t p1;
    tw_stage8_t p2_stage; /* for scalar */
    tw_blocked8_t p2_b8;  /* for AVX-512 */
} tw_ctx_t;

static void tw_init(size_t K, tw_ctx_t *t)
{
    t->p1r = aa(2 * K);
    t->p1i = aa(2 * K);
    gen_p1(K, t->p1r, t->p1i);
    t->p1 = (radix4_dit_stage_twiddles_blocked2_t){
        .re = t->p1r, .im = t->p1i, .K = K};

    for (int j = 0; j < 8; j++) {
        t->p2r[j] = aa(K);
        t->p2i[j] = aa(K);
    }
    gen_p2(K, t->p2r, t->p2i);

    /* BLOCKED8 for scalar (via tw_stage8_t) */
    t->p2_stage.mode = TW_MODE_BLOCKED8;
    for (int j = 0; j < 8; j++) {
        t->p2_stage.b8.re[j] = t->p2r[j];
        t->p2_stage.b8.im[j] = t->p2i[j];
    }
    t->p2_stage.b8.K = K;

    /* BLOCKED8 for AVX-512 (direct) */
    for (int j = 0; j < 8; j++) {
        t->p2_b8.re[j] = t->p2r[j];
        t->p2_b8.im[j] = t->p2i[j];
    }
    t->p2_b8.K = K;
}

static void tw_free(tw_ctx_t *t)
{
    r32_aligned_free(t->p1r);
    r32_aligned_free(t->p1i);
    for (int j = 0; j < 8; j++) {
        r32_aligned_free(t->p2r[j]);
        r32_aligned_free(t->p2i[j]);
    }
}

/*==========================================================================
 * TEST 1: AVX-512 forward ≡ scalar forward
 *=========================================================================*/

TARGET_AVX512
static int test_vs_scalar_fwd(size_t K)
{
    const size_t n = 32 * K;
    double *in_re = aa(n), *in_im = aa(n);
    double *sc_re = aa(n), *sc_im = aa(n);
    double *av_re = aa(n), *av_im = aa(n);
    double *tmp_re = aa(n), *tmp_im = aa(n);
    tw_ctx_t tw;

    fill_rand(in_re, n, 100 + (unsigned)K);
    fill_rand(in_im, n, 200 + (unsigned)K);
    tw_init(K, &tw);

    /* Scalar reference */
    radix32_stage_forward_scalar(K, in_re, in_im, sc_re, sc_im,
                                  &tw.p1, &tw.p2_stage, NULL);

    /* AVX-512 */
    radix32_stage_forward_avx512(K, in_re, in_im, av_re, av_im,
                                  &tw.p1, &tw.p2_b8, tmp_re, tmp_im);

    double err_r = max_abs_diff(sc_re, av_re, n);
    double err_i = max_abs_diff(sc_im, av_im, n);
    double err = fmax(err_r, err_i);

    int pass = err < 1e-12;
    printf("  fwd K=%-5zu err=%.2e  %s\n", K, err, pass ? "PASS" : "FAIL");

    if (!pass) {
        /* Diagnostic: find first divergent element */
        for (size_t i = 0; i < n; i++) {
            double e = fmax(fabs(sc_re[i] - av_re[i]),
                           fabs(sc_im[i] - av_im[i]));
            if (e > 1e-12) {
                printf("    first diff at i=%zu: scalar=(%.15e, %.15e) "
                       "avx512=(%.15e, %.15e)\n",
                       i, sc_re[i], sc_im[i], av_re[i], av_im[i]);
                break;
            }
        }
    }

    tw_free(&tw);
    r32_aligned_free(in_re); r32_aligned_free(in_im);
    r32_aligned_free(sc_re); r32_aligned_free(sc_im);
    r32_aligned_free(av_re); r32_aligned_free(av_im);
    r32_aligned_free(tmp_re); r32_aligned_free(tmp_im);
    return pass;
}

/*==========================================================================
 * TEST 2: AVX-512 backward ≡ scalar backward
 *=========================================================================*/

TARGET_AVX512
static int test_vs_scalar_bwd(size_t K)
{
    const size_t n = 32 * K;
    double *in_re = aa(n), *in_im = aa(n);
    double *sc_re = aa(n), *sc_im = aa(n);
    double *av_re = aa(n), *av_im = aa(n);
    double *tmp_re = aa(n), *tmp_im = aa(n);
    tw_ctx_t tw;

    fill_rand(in_re, n, 300 + (unsigned)K);
    fill_rand(in_im, n, 400 + (unsigned)K);
    tw_init(K, &tw);

    /* Scalar reference */
    radix32_stage_backward_scalar(K, in_re, in_im, sc_re, sc_im,
                                   &tw.p1, &tw.p2_stage, NULL);

    /* AVX-512 */
    radix32_stage_backward_avx512(K, in_re, in_im, av_re, av_im,
                                   &tw.p1, &tw.p2_b8, tmp_re, tmp_im);

    double err_r = max_abs_diff(sc_re, av_re, n);
    double err_i = max_abs_diff(sc_im, av_im, n);
    double err = fmax(err_r, err_i);

    int pass = err < 1e-12;
    printf("  bwd K=%-5zu err=%.2e  %s\n", K, err, pass ? "PASS" : "FAIL");

    tw_free(&tw);
    r32_aligned_free(in_re); r32_aligned_free(in_im);
    r32_aligned_free(sc_re); r32_aligned_free(sc_im);
    r32_aligned_free(av_re); r32_aligned_free(av_im);
    r32_aligned_free(tmp_re); r32_aligned_free(tmp_im);
    return pass;
}

/*==========================================================================
 * TEST 3: Trivial-twiddle roundtrip: IFFT(FFT(x)) = 32·x
 *
 * With all twiddles = 1+0i, the radix-32 stage reduces to pure butterflies.
 * Forward→backward with identity twiddles IS a proper inverse pair.
 * This tests that the DIT-4 and DIF-8 butterflies are correct inverses.
 *=========================================================================*/

TARGET_AVX512
static int test_roundtrip(size_t K)
{
    const size_t n = 32 * K;
    double *in_re = aa(n), *in_im = aa(n);
    double *fwd_re = aa(n), *fwd_im = aa(n);
    double *bwd_re = aa(n), *bwd_im = aa(n);
    double *tmp_re = aa(n), *tmp_im = aa(n);

    fill_rand(in_re, n, 500 + (unsigned)K);
    fill_rand(in_im, n, 600 + (unsigned)K);

    /* Trivial twiddles: all 1+0i */
    double *p1r = aa(2 * K), *p1i = aa(2 * K);
    for (size_t i = 0; i < 2 * K; i++) p1r[i] = 1.0;
    radix4_dit_stage_twiddles_blocked2_t p1 = {.re = p1r, .im = p1i, .K = K};

    double *p2r[8], *p2i[8];
    for (int j = 0; j < 8; j++) {
        p2r[j] = aa(K); p2i[j] = aa(K);
        for (size_t k = 0; k < K; k++) p2r[j][k] = 1.0;
    }
    tw_blocked8_t b8;
    b8.K = K;
    for (int j = 0; j < 8; j++) { b8.re[j] = p2r[j]; b8.im[j] = p2i[j]; }

    /* Forward then backward */
    radix32_stage_forward_avx512(K, in_re, in_im, fwd_re, fwd_im,
                                  &p1, &b8, tmp_re, tmp_im);
    radix32_stage_backward_avx512(K, fwd_re, fwd_im, bwd_re, bwd_im,
                                   &p1, &b8, tmp_re, tmp_im);

    /* Compare: bwd = 32·in */
    double err = 0;
    for (size_t i = 0; i < n; i++) {
        double e = fmax(fabs(bwd_re[i] / 32.0 - in_re[i]),
                       fabs(bwd_im[i] / 32.0 - in_im[i]));
        if (e > err) err = e;
    }

    int pass = err < 1e-12;
    printf("  roundtrip K=%-5zu err=%.2e  %s\n", K, err, pass ? "PASS" : "FAIL");

    r32_aligned_free(p1r); r32_aligned_free(p1i);
    for (int j = 0; j < 8; j++) { r32_aligned_free(p2r[j]); r32_aligned_free(p2i[j]); }
    r32_aligned_free(in_re); r32_aligned_free(in_im);
    r32_aligned_free(fwd_re); r32_aligned_free(fwd_im);
    r32_aligned_free(bwd_re); r32_aligned_free(bwd_im);
    r32_aligned_free(tmp_re); r32_aligned_free(tmp_im);
    return pass;
}

/*==========================================================================
 * TEST 4: Parseval energy: Σ|X|² = 32·Σ|x|²
 *=========================================================================*/

TARGET_AVX512
static int test_parseval(size_t K)
{
    const size_t n = 32 * K;
    double *in_re = aa(n), *in_im = aa(n);
    double *out_re = aa(n), *out_im = aa(n);
    double *tmp_re = aa(n), *tmp_im = aa(n);
    tw_ctx_t tw;

    fill_rand(in_re, n, 700 + (unsigned)K);
    fill_rand(in_im, n, 800 + (unsigned)K);
    tw_init(K, &tw);

    radix32_stage_forward_avx512(K, in_re, in_im, out_re, out_im,
                                  &tw.p1, &tw.p2_b8, tmp_re, tmp_im);

    double energy_in = 0, energy_out = 0;
    for (size_t i = 0; i < n; i++) {
        energy_in  += in_re[i]  * in_re[i]  + in_im[i]  * in_im[i];
        energy_out += out_re[i] * out_re[i] + out_im[i] * out_im[i];
    }

    double ratio = energy_out / (32.0 * energy_in);
    double err = fabs(ratio - 1.0);

    int pass = err < 1e-10;
    printf("  parseval K=%-5zu ratio=%.14f  err=%.2e  %s\n",
           K, ratio, err, pass ? "PASS" : "FAIL");

    tw_free(&tw);
    r32_aligned_free(in_re); r32_aligned_free(in_im);
    r32_aligned_free(out_re); r32_aligned_free(out_im);
    r32_aligned_free(tmp_re); r32_aligned_free(tmp_im);
    return pass;
}

/*==========================================================================
 * MAIN
 *=========================================================================*/

int main(void)
{
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  AVX-512 Radix-32 U=1 Driver — Verification Suite     ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    int total = 0, passed = 0;

    /* K values: must be multiple of 8 for AVX-512 */
    const size_t Ks[] = {8, 16, 32, 64, 128, 256, 512, 1024};
    const int nK = (int)(sizeof(Ks) / sizeof(Ks[0]));

    printf("── Forward: AVX-512 vs scalar ──\n");
    for (int i = 0; i < nK; i++) {
        int p = test_vs_scalar_fwd(Ks[i]);
        total++; passed += p;
    }

    printf("\n── Backward: AVX-512 vs scalar ──\n");
    for (int i = 0; i < nK; i++) {
        int p = test_vs_scalar_bwd(Ks[i]);
        total++; passed += p;
    }

    printf("\n── Roundtrip: IFFT(FFT(x)) = 32·x ──\n");
    for (int i = 0; i < nK; i++) {
        int p = test_roundtrip(Ks[i]);
        total++; passed += p;
    }

    printf("\n── Parseval: energy conservation ──\n");
    for (int i = 0; i < nK; i++) {
        int p = test_parseval(Ks[i]);
        total++; passed += p;
    }

    printf("\n══════════════════════════════════════════\n");
    printf("  %d/%d passed  %s\n", passed, total,
           (passed == total) ? "✓ ALL PASSED" : "✗ FAILURES");
    printf("══════════════════════════════════════════\n");

    return (passed == total) ? 0 : 1;
}
