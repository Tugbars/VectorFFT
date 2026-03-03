/**
 * @file test_radix32_dispatch.c
 * @brief Verify radix-32 dispatch layer (split fv.c / bv.c)
 *
 * Tests:
 *   1. Three-way forward:  AVX-512 ≡ AVX2 ≡ scalar
 *   2. Three-way backward: AVX-512 ≡ AVX2 ≡ scalar
 *   3. Auto-dispatch ISA selection
 *   4. Auto ≡ forced (bit-exact)
 *   5. BLOCKED4 downgrade
 *
 * Links against: fft_radix32_fv.o + fft_radix32_bv.o
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../fft_radix32_platform.h"

/* Uniform header for types, ISA detection, effective_isa */
#include "fft_radix32_uniform.h"

/*==========================================================================
 * Extern declarations for symbols from fv.o / bv.o
 *=========================================================================*/

extern radix32_isa_level_t radix32_forward(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT pass1_tw,
    const tw_stage8_t *RESTRICT pass2_tw,
    const tw_recurrence_scalar_t *RESTRICT rec_tw,
    double *RESTRICT temp_re, double *RESTRICT temp_im);

extern radix32_isa_level_t radix32_backward(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT pass1_tw,
    const tw_stage8_t *RESTRICT pass2_tw,
    const tw_recurrence_scalar_t *RESTRICT rec_tw,
    double *RESTRICT temp_re, double *RESTRICT temp_im);

extern void radix32_forward_force_avx512(
    size_t, const double *, const double *, double *, double *,
    const radix4_dit_stage_twiddles_blocked2_t *,
    const tw_stage8_t *, double *, double *);
extern void radix32_forward_force_avx2(
    size_t, const double *, const double *, double *, double *,
    const radix4_dit_stage_twiddles_blocked2_t *,
    const tw_stage8_t *, double *, double *);
extern void radix32_forward_force_scalar(
    size_t, const double *, const double *, double *, double *,
    const radix4_dit_stage_twiddles_blocked2_t *,
    const tw_stage8_t *, const tw_recurrence_scalar_t *);

extern void radix32_backward_force_avx512(
    size_t, const double *, const double *, double *, double *,
    const radix4_dit_stage_twiddles_blocked2_t *,
    const tw_stage8_t *, double *, double *);
extern void radix32_backward_force_avx2(
    size_t, const double *, const double *, double *, double *,
    const radix4_dit_stage_twiddles_blocked2_t *,
    const tw_stage8_t *, double *, double *);
extern void radix32_backward_force_scalar(
    size_t, const double *, const double *, double *, double *,
    const radix4_dit_stage_twiddles_blocked2_t *,
    const tw_stage8_t *, const tw_recurrence_scalar_t *);

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
    for (size_t i = 0; i < n; i++)
    {
        seed = seed * 1103515245 + 12345;
        buf[i] = ((double)(seed >> 16) / 32768.0) - 1.0;
    }
}

static double max_err(const double *a, const double *b, size_t n)
{
    double mx = 0;
    for (size_t i = 0; i < n; i++)
    {
        double e = fabs(a[i] - b[i]);
        if (e > mx)
            mx = e;
    }
    return mx;
}

/*==========================================================================
 * Twiddle generation
 *=========================================================================*/

typedef struct
{
    double *p1r, *p1i, *p2r[8], *p2i[8];
    radix4_dit_stage_twiddles_blocked2_t p1;
    tw_stage8_t p2;
} tw_ctx_t;

static void tw_init(size_t K, tw_mode_t mode, tw_ctx_t *t)
{
    t->p1r = aa(2 * K);
    t->p1i = aa(2 * K);
    for (size_t k = 0; k < K; k++)
    {
        double a = -2.0 * M_PI * (double)k / (32.0 * (double)K);
        t->p1r[k] = cos(a);
        t->p1i[k] = sin(a);
        t->p1r[K + k] = cos(2 * a);
        t->p1i[K + k] = sin(2 * a);
    }
    t->p1 = (radix4_dit_stage_twiddles_blocked2_t){
        .re = t->p1r, .im = t->p1i, .K = K};

    for (int j = 0; j < 8; j++)
    {
        t->p2r[j] = aa(K);
        t->p2i[j] = aa(K);
        double base = -2.0 * M_PI * (double)(j + 1) / (8.0 * (double)K);
        for (size_t k = 0; k < K; k++)
        {
            double a = base * (double)k;
            t->p2r[j][k] = cos(a);
            t->p2i[j][k] = sin(a);
        }
    }

    if (mode == TW_MODE_BLOCKED8)
    {
        t->p2.mode = TW_MODE_BLOCKED8;
        for (int j = 0; j < 8; j++)
        {
            t->p2.b8.re[j] = t->p2r[j];
            t->p2.b8.im[j] = t->p2i[j];
        }
        t->p2.b8.K = K;
    }
    else
    {
        t->p2.mode = TW_MODE_BLOCKED4;
        for (int j = 0; j < 4; j++)
        {
            t->p2.b4.re[j] = t->p2r[j];
            t->p2.b4.im[j] = t->p2i[j];
        }
        t->p2.b4.K = K;
    }
}

static void tw_free(tw_ctx_t *t)
{
    r32_aligned_free(t->p1r);
    r32_aligned_free(t->p1i);
    for (int j = 0; j < 8; j++)
    {
        r32_aligned_free(t->p2r[j]);
        r32_aligned_free(t->p2i[j]);
    }
}

/*==========================================================================
 * TEST 1: Three-way forward
 *=========================================================================*/

static int test_3way_fwd(size_t K)
{
    const size_t n = 32 * K;
    double *ir = aa(n), *ii = aa(n);
    double *sr = aa(n), *si = aa(n);
    double *a2r = aa(n), *a2i = aa(n);
    double *a5r = aa(n), *a5i = aa(n);
    double *tr = aa(n), *ti = aa(n);
    tw_ctx_t tw;

    fill_rand(ir, n, 1000 + (unsigned)K);
    fill_rand(ii, n, 2000 + (unsigned)K);
    tw_init(K, TW_MODE_BLOCKED8, &tw);

    radix32_forward_force_scalar(K, ir, ii, sr, si, &tw.p1, &tw.p2, NULL);
    radix32_forward_force_avx2(K, ir, ii, a2r, a2i, &tw.p1, &tw.p2, tr, ti);
    radix32_forward_force_avx512(K, ir, ii, a5r, a5i, &tw.p1, &tw.p2, tr, ti);

    double e_s2 = fmax(max_err(sr, a2r, n), max_err(si, a2i, n));
    double e_s5 = fmax(max_err(sr, a5r, n), max_err(si, a5i, n));
    double e_25 = fmax(max_err(a2r, a5r, n), max_err(a2i, a5i, n));

    int pass = e_s2 < 1e-12 && e_s5 < 1e-12 && e_25 < 1e-13;
    printf("  fwd K=%-5zu  sc≡a2=%.1e  sc≡a5=%.1e  a2≡a5=%.1e  %s\n",
           K, e_s2, e_s5, e_25, pass ? "PASS" : "FAIL");

    tw_free(&tw);
    r32_aligned_free(ir);
    r32_aligned_free(ii);
    r32_aligned_free(sr);
    r32_aligned_free(si);
    r32_aligned_free(a2r);
    r32_aligned_free(a2i);
    r32_aligned_free(a5r);
    r32_aligned_free(a5i);
    r32_aligned_free(tr);
    r32_aligned_free(ti);
    return pass;
}

/*==========================================================================
 * TEST 2: Three-way backward
 *=========================================================================*/

static int test_3way_bwd(size_t K)
{
    const size_t n = 32 * K;
    double *ir = aa(n), *ii = aa(n);
    double *sr = aa(n), *si = aa(n);
    double *a2r = aa(n), *a2i = aa(n);
    double *a5r = aa(n), *a5i = aa(n);
    double *tr = aa(n), *ti = aa(n);
    tw_ctx_t tw;

    fill_rand(ir, n, 3000 + (unsigned)K);
    fill_rand(ii, n, 4000 + (unsigned)K);
    tw_init(K, TW_MODE_BLOCKED8, &tw);

    radix32_backward_force_scalar(K, ir, ii, sr, si, &tw.p1, &tw.p2, NULL);
    radix32_backward_force_avx2(K, ir, ii, a2r, a2i, &tw.p1, &tw.p2, tr, ti);
    radix32_backward_force_avx512(K, ir, ii, a5r, a5i, &tw.p1, &tw.p2, tr, ti);

    double e_s2 = fmax(max_err(sr, a2r, n), max_err(si, a2i, n));
    double e_s5 = fmax(max_err(sr, a5r, n), max_err(si, a5i, n));
    double e_25 = fmax(max_err(a2r, a5r, n), max_err(a2i, a5i, n));

    int pass = e_s2 < 1e-12 && e_s5 < 1e-12 && e_25 < 1e-13;
    printf("  bwd K=%-5zu  sc≡a2=%.1e  sc≡a5=%.1e  a2≡a5=%.1e  %s\n",
           K, e_s2, e_s5, e_25, pass ? "PASS" : "FAIL");

    tw_free(&tw);
    r32_aligned_free(ir);
    r32_aligned_free(ii);
    r32_aligned_free(sr);
    r32_aligned_free(si);
    r32_aligned_free(a2r);
    r32_aligned_free(a2i);
    r32_aligned_free(a5r);
    r32_aligned_free(a5i);
    r32_aligned_free(tr);
    r32_aligned_free(ti);
    return pass;
}

/*==========================================================================
 * TEST 3: Auto-dispatch ISA selection
 *=========================================================================*/

static int test_dispatch_isa(size_t K, tw_mode_t mode,
                             radix32_isa_level_t expected)
{
    const size_t n = 32 * K;
    double *ir = aa(n), *ii = aa(n);
    double *or_ = aa(n), *oi = aa(n);
    double *tr = aa(n), *ti = aa(n);
    tw_ctx_t tw;

    fill_rand(ir, n, 5000 + (unsigned)K);
    fill_rand(ii, n, 6000 + (unsigned)K);
    tw_init(K, mode, &tw);

    radix32_isa_level_t got = radix32_forward(
        K, ir, ii, or_, oi, &tw.p1, &tw.p2, NULL, tr, ti);

    int pass = (got == expected);
    printf("  K=%-5zu mode=%-9s  expected=%-8s got=%-8s  %s\n",
           K,
           mode == TW_MODE_BLOCKED8 ? "BLOCKED8" : "BLOCKED4",
           radix32_isa_name(expected), radix32_isa_name(got),
           pass ? "PASS" : "FAIL");

    tw_free(&tw);
    r32_aligned_free(ir);
    r32_aligned_free(ii);
    r32_aligned_free(or_);
    r32_aligned_free(oi);
    r32_aligned_free(tr);
    r32_aligned_free(ti);
    return pass;
}

/*==========================================================================
 * TEST 4: Auto ≡ forced (bit-exact)
 *=========================================================================*/

static int test_auto_vs_forced(size_t K, int direction)
{
    const size_t n = 32 * K;
    double *ir = aa(n), *ii = aa(n);
    double *ar = aa(n), *ai = aa(n);
    double *fr = aa(n), *fi = aa(n);
    double *t1r = aa(n), *t1i = aa(n);
    double *t2r = aa(n), *t2i = aa(n);
    tw_ctx_t tw;

    fill_rand(ir, n, 7000 + (unsigned)K + direction * 100);
    fill_rand(ii, n, 8000 + (unsigned)K + direction * 100);
    tw_init(K, TW_MODE_BLOCKED8, &tw);

    radix32_isa_level_t isa;
    if (direction == 0)
    {
        isa = radix32_forward(K, ir, ii, ar, ai, &tw.p1, &tw.p2, NULL, t1r, t1i);
        switch (isa)
        {
        case ISA_AVX512:
            radix32_forward_force_avx512(K, ir, ii, fr, fi, &tw.p1, &tw.p2, t2r, t2i);
            break;
        case ISA_AVX2:
            radix32_forward_force_avx2(K, ir, ii, fr, fi, &tw.p1, &tw.p2, t2r, t2i);
            break;
        default:
            radix32_forward_force_scalar(K, ir, ii, fr, fi, &tw.p1, &tw.p2, NULL);
            break;
        }
    }
    else
    {
        isa = radix32_backward(K, ir, ii, ar, ai, &tw.p1, &tw.p2, NULL, t1r, t1i);
        switch (isa)
        {
        case ISA_AVX512:
            radix32_backward_force_avx512(K, ir, ii, fr, fi, &tw.p1, &tw.p2, t2r, t2i);
            break;
        case ISA_AVX2:
            radix32_backward_force_avx2(K, ir, ii, fr, fi, &tw.p1, &tw.p2, t2r, t2i);
            break;
        default:
            radix32_backward_force_scalar(K, ir, ii, fr, fi, &tw.p1, &tw.p2, NULL);
            break;
        }
    }

    double err = fmax(max_err(ar, fr, n), max_err(ai, fi, n));
    int pass = (err == 0.0);
    printf("  %s K=%-5zu isa=%-8s  auto≡forced=%.1e  %s\n",
           direction == 0 ? "fwd" : "bwd", K, radix32_isa_name(isa),
           err, pass ? "PASS" : "FAIL");

    tw_free(&tw);
    r32_aligned_free(ir);
    r32_aligned_free(ii);
    r32_aligned_free(ar);
    r32_aligned_free(ai);
    r32_aligned_free(fr);
    r32_aligned_free(fi);
    r32_aligned_free(t1r);
    r32_aligned_free(t1i);
    r32_aligned_free(t2r);
    r32_aligned_free(t2i);
    return pass;
}

/*==========================================================================
 * TEST 5: BLOCKED4 via AVX-512 — correctness vs scalar
 *=========================================================================*/

static int test_blocked4(size_t K)
{
    const size_t n = 32 * K;
    double *ir = aa(n), *ii = aa(n);
    double *ar = aa(n), *ai = aa(n);
    double *sr = aa(n), *si = aa(n);
    double *tr = aa(n), *ti = aa(n);
    tw_ctx_t tw;

    fill_rand(ir, n, 9000 + (unsigned)K);
    fill_rand(ii, n, 9500 + (unsigned)K);
    tw_init(K, TW_MODE_BLOCKED4, &tw);

    radix32_isa_level_t isa = radix32_forward(
        K, ir, ii, ar, ai, &tw.p1, &tw.p2, NULL, tr, ti);
    radix32_forward_force_scalar(K, ir, ii, sr, si, &tw.p1, &tw.p2, NULL);

    double err = fmax(max_err(sr, ar, n), max_err(si, ai, n));
    int val_ok = (err < 1e-12);

    /* On AVX-512 hw with K%8==0 && K>=16, BLOCKED4 now hits AVX-512 */
    int isa_ok;
    const char *expect_str;
    if ((K & 7) == 0 && K >= 16 && radix32_get_isa_level() >= ISA_AVX512)
    {
        isa_ok = (isa == ISA_AVX512);
        expect_str = "AVX-512";
    }
    else if ((K & 3) == 0 && K >= 8)
    {
        isa_ok = (isa == ISA_AVX2);
        expect_str = "AVX2";
    }
    else
    {
        isa_ok = (isa == ISA_SCALAR);
        expect_str = "scalar";
    }
    int pass = isa_ok && val_ok;

    printf("  BLOCKED4 K=%-5zu isa=%-8s (exp=%s:%s) err=%.1e  %s\n",
           K, radix32_isa_name(isa), expect_str, isa_ok ? "ok" : "WRONG",
           err, pass ? "PASS" : "FAIL");

    tw_free(&tw);
    r32_aligned_free(ir);
    r32_aligned_free(ii);
    r32_aligned_free(ar);
    r32_aligned_free(ai);
    r32_aligned_free(sr);
    r32_aligned_free(si);
    r32_aligned_free(tr);
    r32_aligned_free(ti);
    return pass;
}

/*==========================================================================
 * MAIN
 *=========================================================================*/

int main(void)
{
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  Radix-32 Dispatch (fv.c + bv.c) — Verification Suite  ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    radix32_isa_level_t hw = radix32_get_isa_level();
    printf("Hardware ISA: %s\n\n", radix32_isa_name(hw));

    int total = 0, passed = 0;
    const size_t Ks[] = {16, 32, 64, 128, 256};
    const int nK = 5;

    printf("── Three-way forward: AVX-512 ≡ AVX2 ≡ scalar ──\n");
    for (int i = 0; i < nK; i++)
    {
        total++;
        passed += test_3way_fwd(Ks[i]);
    }

    printf("\n── Three-way backward: AVX-512 ≡ AVX2 ≡ scalar ──\n");
    for (int i = 0; i < nK; i++)
    {
        total++;
        passed += test_3way_bwd(Ks[i]);
    }

    printf("\n── Auto-dispatch ISA selection ──\n");
    if (hw >= ISA_AVX512)
    {
        total++;
        passed += test_dispatch_isa(64, TW_MODE_BLOCKED8, ISA_AVX512);
        total++;
        passed += test_dispatch_isa(256, TW_MODE_BLOCKED8, ISA_AVX512);
        total++;
        passed += test_dispatch_isa(64, TW_MODE_BLOCKED4, ISA_AVX512);
        total++;
        passed += test_dispatch_isa(512, TW_MODE_BLOCKED4, ISA_AVX512);
    }
    if (hw >= ISA_AVX2)
    {
        total++;
        passed += test_dispatch_isa(8, TW_MODE_BLOCKED8, ISA_AVX2);
    }

    printf("\n── Auto ≡ forced (bit-exact) ──\n");
    for (int i = 0; i < nK; i++)
    {
        total++;
        passed += test_auto_vs_forced(Ks[i], 0);
        total++;
        passed += test_auto_vs_forced(Ks[i], 1);
    }

    printf("\n── BLOCKED4 via AVX-512 ──\n");
    {
        size_t b4K[] = {16, 64, 256, 512, 1024, 4096};
        for (int i = 0; i < 6; i++)
        {
            total++;
            passed += test_blocked4(b4K[i]);
        }
    }

    printf("\n══════════════════════════════════════════\n");
    printf("  %d/%d passed  %s\n", passed, total,
           (passed == total) ? "✓ ALL PASSED" : "✗ FAILURES");
    printf("══════════════════════════════════════════\n");

    return (passed == total) ? 0 : 1;
}