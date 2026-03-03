/**
 * @file test_radix32_n1.c
 * @brief Verify twiddle-less (N=1) radix-32 codelets
 *
 * Tests:
 *   1. Scalar N=1 forward  ≡ scalar twiddled with identity twiddles
 *   2. Scalar N=1 backward ≡ scalar twiddled with identity twiddles
 *   3. AVX-512 N=1 forward  ≡ AVX-512 twiddled with identity twiddles
 *   4. AVX-512 N=1 backward ≡ AVX-512 twiddled with identity twiddles
 *   5. Scalar N=1 ≡ AVX-512 N=1 (cross-ISA)
 *   6. N=1 roundtrip: backward(forward(x)) = 32·x
 *   7. N=1 Parseval energy conservation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../fft_radix32_platform.h"
#include "fft_radix32_scalar_n1.h"
#include "fft_radix32_avx512_n1.h"
#include "fft_radix32_avx512.h"  /* twiddled AVX-512 driver for comparison */

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

static double max_err(const double *a, const double *b, size_t n)
{
    double mx = 0;
    for (size_t i = 0; i < n; i++) {
        double e = fabs(a[i] - b[i]);
        if (e > mx) mx = e;
    }
    return mx;
}

/*==========================================================================
 * Identity twiddle generation: all W = 1+0i
 *=========================================================================*/

typedef struct {
    double *p1r, *p1i, *p2r[8], *p2i[8];
    radix4_dit_stage_twiddles_blocked2_t p1;
    tw_stage8_t p2;
    tw_blocked8_t b8;
} identity_tw_t;

static void identity_tw_init(size_t K, identity_tw_t *t)
{
    t->p1r = aa(2 * K);
    t->p1i = aa(2 * K);
    for (size_t k = 0; k < 2 * K; k++) {
        t->p1r[k] = 1.0;
        t->p1i[k] = 0.0;
    }
    t->p1 = (radix4_dit_stage_twiddles_blocked2_t){
        .re = t->p1r, .im = t->p1i, .K = K};

    for (int j = 0; j < 8; j++) {
        t->p2r[j] = aa(K);
        t->p2i[j] = aa(K);
        for (size_t k = 0; k < K; k++) {
            t->p2r[j][k] = 1.0;
            t->p2i[j][k] = 0.0;
        }
    }

    t->p2.mode = TW_MODE_BLOCKED8;
    for (int j = 0; j < 8; j++) {
        t->p2.b8.re[j] = t->p2r[j];
        t->p2.b8.im[j] = t->p2i[j];
    }
    t->p2.b8.K = K;

    /* Direct tw_blocked8_t for AVX-512 */
    t->b8.K = K;
    for (int j = 0; j < 8; j++) {
        t->b8.re[j] = t->p2r[j];
        t->b8.im[j] = t->p2i[j];
    }
}

static void identity_tw_free(identity_tw_t *t)
{
    r32_aligned_free(t->p1r); r32_aligned_free(t->p1i);
    for (int j = 0; j < 8; j++) { r32_aligned_free(t->p2r[j]); r32_aligned_free(t->p2i[j]); }
}

/*==========================================================================
 * TEST 1: Scalar N=1 forward ≡ scalar twiddled (identity twiddles)
 *=========================================================================*/

TARGET_AVX512
static int test_scalar_n1_vs_twiddled_fwd(size_t K)
{
    const size_t n = 32 * K;
    double *ir = aa(n), *ii = aa(n);
    double *n1r = aa(n), *n1i = aa(n);
    double *twr = aa(n), *twi = aa(n);
    identity_tw_t tw;

    fill_rand(ir, n, 100 + (unsigned)K);
    fill_rand(ii, n, 200 + (unsigned)K);
    identity_tw_init(K, &tw);

    radix32_n1_forward_scalar(K, ir, ii, n1r, n1i);
    radix32_stage_forward_scalar(K, ir, ii, twr, twi, &tw.p1, &tw.p2, NULL);

    double err = fmax(max_err(n1r, twr, n), max_err(n1i, twi, n));
    int pass = (err == 0.0); /* must be bit-exact: same ops */
    printf("  scalar fwd K=%-5zu  n1≡tw=%.1e  %s\n", K, err, pass ? "PASS" : "FAIL");

    identity_tw_free(&tw);
    r32_aligned_free(ir); r32_aligned_free(ii); r32_aligned_free(n1r); r32_aligned_free(n1i); r32_aligned_free(twr); r32_aligned_free(twi);
    return pass;
}

/*==========================================================================
 * TEST 2: Scalar N=1 backward ≡ scalar twiddled (identity twiddles)
 *=========================================================================*/

TARGET_AVX512
static int test_scalar_n1_vs_twiddled_bwd(size_t K)
{
    const size_t n = 32 * K;
    double *ir = aa(n), *ii = aa(n);
    double *n1r = aa(n), *n1i = aa(n);
    double *twr = aa(n), *twi = aa(n);
    identity_tw_t tw;

    fill_rand(ir, n, 300 + (unsigned)K);
    fill_rand(ii, n, 400 + (unsigned)K);
    identity_tw_init(K, &tw);

    radix32_n1_backward_scalar(K, ir, ii, n1r, n1i);
    radix32_stage_backward_scalar(K, ir, ii, twr, twi, &tw.p1, &tw.p2, NULL);

    double err = fmax(max_err(n1r, twr, n), max_err(n1i, twi, n));
    int pass = (err == 0.0);
    printf("  scalar bwd K=%-5zu  n1≡tw=%.1e  %s\n", K, err, pass ? "PASS" : "FAIL");

    identity_tw_free(&tw);
    r32_aligned_free(ir); r32_aligned_free(ii); r32_aligned_free(n1r); r32_aligned_free(n1i); r32_aligned_free(twr); r32_aligned_free(twi);
    return pass;
}

/*==========================================================================
 * TEST 3: AVX-512 N=1 forward ≡ AVX-512 twiddled (identity twiddles)
 *=========================================================================*/

TARGET_AVX512
static int test_avx512_n1_vs_twiddled_fwd(size_t K)
{
    const size_t n = 32 * K;
    double *ir = aa(n), *ii = aa(n);
    double *n1r = aa(n), *n1i = aa(n);
    double *twr = aa(n), *twi = aa(n);
    double *tmp1r = aa(n), *tmp1i = aa(n);
    double *tmp2r = aa(n), *tmp2i = aa(n);
    identity_tw_t tw;

    fill_rand(ir, n, 500 + (unsigned)K);
    fill_rand(ii, n, 600 + (unsigned)K);
    identity_tw_init(K, &tw);

    radix32_n1_forward_avx512(K, ir, ii, n1r, n1i, tmp1r, tmp1i);
    radix32_stage_forward_avx512(K, ir, ii, twr, twi, &tw.p1, &tw.b8, tmp2r, tmp2i);

    double err = fmax(max_err(n1r, twr, n), max_err(n1i, twi, n));
    int pass = (err < 2.3e-16); /* may differ by FMA rounding from identity cmul */
    printf("  avx512 fwd K=%-5zu  n1≡tw=%.1e  %s\n", K, err, pass ? "PASS" : "FAIL");

    identity_tw_free(&tw);
    r32_aligned_free(ir); r32_aligned_free(ii); r32_aligned_free(n1r); r32_aligned_free(n1i); r32_aligned_free(twr); r32_aligned_free(twi);
    r32_aligned_free(tmp1r); r32_aligned_free(tmp1i); r32_aligned_free(tmp2r); r32_aligned_free(tmp2i);
    return pass;
}

/*==========================================================================
 * TEST 4: AVX-512 N=1 backward ≡ AVX-512 twiddled (identity twiddles)
 *=========================================================================*/

TARGET_AVX512
static int test_avx512_n1_vs_twiddled_bwd(size_t K)
{
    const size_t n = 32 * K;
    double *ir = aa(n), *ii = aa(n);
    double *n1r = aa(n), *n1i = aa(n);
    double *twr = aa(n), *twi = aa(n);
    double *tmp1r = aa(n), *tmp1i = aa(n);
    double *tmp2r = aa(n), *tmp2i = aa(n);
    identity_tw_t tw;

    fill_rand(ir, n, 700 + (unsigned)K);
    fill_rand(ii, n, 800 + (unsigned)K);
    identity_tw_init(K, &tw);

    radix32_n1_backward_avx512(K, ir, ii, n1r, n1i, tmp1r, tmp1i);
    radix32_stage_backward_avx512(K, ir, ii, twr, twi, &tw.p1, &tw.b8, tmp2r, tmp2i);

    double err = fmax(max_err(n1r, twr, n), max_err(n1i, twi, n));
    int pass = (err < 2.3e-16);
    printf("  avx512 bwd K=%-5zu  n1≡tw=%.1e  %s\n", K, err, pass ? "PASS" : "FAIL");

    identity_tw_free(&tw);
    r32_aligned_free(ir); r32_aligned_free(ii); r32_aligned_free(n1r); r32_aligned_free(n1i); r32_aligned_free(twr); r32_aligned_free(twi);
    r32_aligned_free(tmp1r); r32_aligned_free(tmp1i); r32_aligned_free(tmp2r); r32_aligned_free(tmp2i);
    return pass;
}

/*==========================================================================
 * TEST 5: Scalar N=1 ≡ AVX-512 N=1 (cross-ISA)
 *=========================================================================*/

TARGET_AVX512
static int test_cross_isa_n1(size_t K, int direction)
{
    const size_t n = 32 * K;
    double *ir = aa(n), *ii = aa(n);
    double *sr = aa(n), *si = aa(n);
    double *ar = aa(n), *ai = aa(n);
    double *tr = aa(n), *ti = aa(n);

    fill_rand(ir, n, 900 + (unsigned)K + direction * 50);
    fill_rand(ii, n, 950 + (unsigned)K + direction * 50);

    if (direction == 0) {
        radix32_n1_forward_scalar(K, ir, ii, sr, si);
        radix32_n1_forward_avx512(K, ir, ii, ar, ai, tr, ti);
    } else {
        radix32_n1_backward_scalar(K, ir, ii, sr, si);
        radix32_n1_backward_avx512(K, ir, ii, ar, ai, tr, ti);
    }

    double err = fmax(max_err(sr, ar, n), max_err(si, ai, n));
    int pass = (err < 2e-15); /* 1-2 ULP from SIMD vs scalar FMA rounding */
    printf("  %s K=%-5zu  scalar≡avx512=%.1e  %s\n",
           direction == 0 ? "fwd" : "bwd", K, err, pass ? "PASS" : "FAIL");

    r32_aligned_free(ir); r32_aligned_free(ii); r32_aligned_free(sr); r32_aligned_free(si); r32_aligned_free(ar); r32_aligned_free(ai);
    r32_aligned_free(tr); r32_aligned_free(ti);
    return pass;
}

/*==========================================================================
 * TEST 6: N=1 roundtrip: backward(forward(x)) = 32·x
 *=========================================================================*/

TARGET_AVX512
static int test_n1_roundtrip(size_t K, int use_avx512)
{
    const size_t n = 32 * K;
    double *ir = aa(n), *ii = aa(n);
    double *fr = aa(n), *fi = aa(n);
    double *br = aa(n), *bi = aa(n);
    double *t1r = aa(n), *t1i = aa(n);
    double *t2r = aa(n), *t2i = aa(n);

    fill_rand(ir, n, 1100 + (unsigned)K + use_avx512 * 50);
    fill_rand(ii, n, 1200 + (unsigned)K + use_avx512 * 50);

    if (use_avx512) {
        radix32_n1_forward_avx512(K, ir, ii, fr, fi, t1r, t1i);
        radix32_n1_backward_avx512(K, fr, fi, br, bi, t2r, t2i);
    } else {
        radix32_n1_forward_scalar(K, ir, ii, fr, fi);
        radix32_n1_backward_scalar(K, fr, fi, br, bi);
    }

    /* backward(forward(x)) should = 32·x */
    double err = 0;
    for (size_t i = 0; i < n; i++) {
        double e = fmax(fabs(br[i] - 32.0 * ir[i]),
                        fabs(bi[i] - 32.0 * ii[i]));
        if (e > err) err = e;
    }
    /* Normalize by magnitude */
    double mx = 0;
    for (size_t i = 0; i < n; i++) {
        double m = fmax(fabs(ir[i]), fabs(ii[i]));
        if (m > mx) mx = m;
    }
    double rel = (mx > 0) ? err / (32.0 * mx) : err;

    int pass = (rel < 1e-14);
    printf("  roundtrip %s K=%-5zu  rel_err=%.2e  %s\n",
           use_avx512 ? "avx512" : "scalar", K, rel, pass ? "PASS" : "FAIL");

    r32_aligned_free(ir); r32_aligned_free(ii); r32_aligned_free(fr); r32_aligned_free(fi); r32_aligned_free(br); r32_aligned_free(bi);
    r32_aligned_free(t1r); r32_aligned_free(t1i); r32_aligned_free(t2r); r32_aligned_free(t2i);
    return pass;
}

/*==========================================================================
 * TEST 7: Parseval energy conservation
 *=========================================================================*/

TARGET_AVX512
static int test_n1_parseval(size_t K, int use_avx512)
{
    const size_t n = 32 * K;
    double *ir = aa(n), *ii = aa(n);
    double *fr = aa(n), *fi = aa(n);
    double *tr = aa(n), *ti = aa(n);

    fill_rand(ir, n, 1300 + (unsigned)K + use_avx512 * 50);
    fill_rand(ii, n, 1400 + (unsigned)K + use_avx512 * 50);

    if (use_avx512)
        radix32_n1_forward_avx512(K, ir, ii, fr, fi, tr, ti);
    else
        radix32_n1_forward_scalar(K, ir, ii, fr, fi);

    double e_in = 0, e_out = 0;
    for (size_t i = 0; i < n; i++) {
        e_in  += ir[i] * ir[i] + ii[i] * ii[i];
        e_out += fr[i] * fr[i] + fi[i] * fi[i];
    }
    /* Parseval: sum|X|² = 32 · sum|x|² for unnormalized DFT-32 */
    double ratio = e_out / (32.0 * e_in);
    double err = fabs(ratio - 1.0);

    int pass = (err < 1e-13);
    printf("  parseval %s K=%-5zu  ratio=%.14f  err=%.2e  %s\n",
           use_avx512 ? "avx512" : "scalar", K, ratio, err, pass ? "PASS" : "FAIL");

    r32_aligned_free(ir); r32_aligned_free(ii); r32_aligned_free(fr); r32_aligned_free(fi); r32_aligned_free(tr); r32_aligned_free(ti);
    return pass;
}

/*==========================================================================
 * MAIN
 *=========================================================================*/

int main(void)
{
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  Radix-32 N=1 (Twiddle-less) — Verification Suite     ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    int total = 0, passed = 0;

    /* Scalar twiddled comparison needs K >= 4 */
    const size_t sc_tw_Ks[] = {4, 7, 8, 16, 32, 64};
    const int nsc_tw = 6;

    /* Scalar N=1 standalone tests: any K >= 1 */
    const size_t sc_Ks[] = {1, 2, 3, 4, 7, 8, 16, 32, 64};
    const int nsc = 9;

    /* AVX-512 tests: K must be multiple of 8, >=16 for twiddled comparison */
    const size_t av_Ks[] = {8, 16, 32, 64, 128, 256};
    const int nav = 6;

    printf("── Scalar N=1 fwd ≡ twiddled(identity) ──\n");
    for (int i = 0; i < nsc_tw; i++) { total++; passed += test_scalar_n1_vs_twiddled_fwd(sc_tw_Ks[i]); }

    printf("\n── Scalar N=1 bwd ≡ twiddled(identity) ──\n");
    for (int i = 0; i < nsc_tw; i++) { total++; passed += test_scalar_n1_vs_twiddled_bwd(sc_tw_Ks[i]); }

    printf("\n── AVX-512 N=1 fwd ≡ twiddled(identity) ──\n");
    for (int i = 0; i < nav; i++) { total++; passed += test_avx512_n1_vs_twiddled_fwd(av_Ks[i]); }

    printf("\n── AVX-512 N=1 bwd ≡ twiddled(identity) ──\n");
    for (int i = 0; i < nav; i++) { total++; passed += test_avx512_n1_vs_twiddled_bwd(av_Ks[i]); }

    printf("\n── Cross-ISA: scalar N=1 ≡ AVX-512 N=1 ──\n");
    for (int i = 0; i < nav; i++) {
        total++; passed += test_cross_isa_n1(av_Ks[i], 0);
        total++; passed += test_cross_isa_n1(av_Ks[i], 1);
    }

    printf("\n── Roundtrip: backward(forward(x)) = 32·x ──\n");
    for (int i = 0; i < nsc; i++) { total++; passed += test_n1_roundtrip(sc_Ks[i], 0); }
    for (int i = 0; i < nav; i++) { total++; passed += test_n1_roundtrip(av_Ks[i], 1); }

    printf("\n── Parseval energy conservation ──\n");
    { size_t pk[] = {4, 8, 16, 64};
      for (int i = 0; i < 4; i++) { total++; passed += test_n1_parseval(pk[i], 0); } }
    { size_t pk[] = {32, 64, 128, 256};
      for (int i = 0; i < 4; i++) { total++; passed += test_n1_parseval(pk[i], 1); } }

    printf("\n══════════════════════════════════════════\n");
    printf("  %d/%d passed  %s\n", passed, total,
           (passed == total) ? "✓ ALL PASSED" : "✗ FAILURES");
    printf("══════════════════════════════════════════\n");

    return (passed == total) ? 0 : 1;
}
