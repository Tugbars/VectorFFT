/**
 * test_dispatch.c — Comprehensive test for fft_radix32_dispatch.h
 *
 * Tests:
 *   1. notw forward/backward across K=1..1024 (hits scalar, AVX2, AVX-512)
 *   2. tw forward/backward across K=1..1024
 *   3. Verifies ISA selection per K
 *   4. Round-trip: forward then backward should recover input (up to scale)
 *
 * Build:
 *   gcc -O3 -march=native -mavx512f -mavx512dq -mavx2 -mfma \
 *       -o test_dispatch test_dispatch.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <malloc.h>
#define _USE_MATH_DEFINES
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "fft_radix32_dispatch.h"

/* ── Aligned allocation ── */
static double *alloc64(size_t n) {
    double *p = NULL;
#ifdef _WIN32
    p = (double *)_aligned_malloc(n * sizeof(double), 64);
    if (!p) { fprintf(stderr, "alloc failed\n"); exit(1); }
#else
    if (posix_memalign((void**)&p, 64, n * sizeof(double)) != 0) {
        fprintf(stderr, "alloc failed\n"); exit(1);
    }
#endif
    memset(p, 0, n * sizeof(double));
    return p;
}

static void free64(double *p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

/* ── Scalar reference: pure DFT-32, no twiddles ── */
static void ref_notw_dft32(const double *ir, const double *ii,
                           double *or_, double *oi, size_t K, int fwd) {
    const double sign = fwd ? -1.0 : 1.0;
    for (size_t k = 0; k < K; k++) {
        for (int m = 0; m < 32; m++) {
            double sr = 0, si = 0;
            for (int n = 0; n < 32; n++) {
                double a = sign * 2.0 * M_PI * (double)(m * n) / 32.0;
                double wr = cos(a), wi = sin(a);
                sr += ir[n*K+k]*wr - ii[n*K+k]*wi;
                si += ir[n*K+k]*wi + ii[n*K+k]*wr;
            }
            or_[m*K+k] = sr;
            oi[m*K+k] = si;
        }
    }
}

/* ── Scalar reference: twiddled DFT-32 ── */
static void ref_tw_dft32(const double *ir, const double *ii,
                         double *or_, double *oi, size_t K, int fwd) {
    const size_t NN = 32 * K;
    const double sign = fwd ? -1.0 : 1.0;
    for (size_t k = 0; k < K; k++) {
        double xr[32], xi[32];
        for (int n = 0; n < 32; n++) {
            double dr = ir[n*K+k], di = ii[n*K+k];
            if (n > 0) {
                double a = sign * 2.0 * M_PI * (double)(n * k) / (double)NN;
                double wr = cos(a), wi = sin(a);
                double tr = dr*wr - di*wi;
                di = dr*wi + di*wr;
                dr = tr;
            }
            xr[n] = dr; xi[n] = di;
        }
        for (int m = 0; m < 32; m++) {
            double sr = 0, si = 0;
            for (int n = 0; n < 32; n++) {
                double a = sign * 2.0 * M_PI * (double)(m * n) / 32.0;
                double wr = cos(a), wi = sin(a);
                sr += xr[n]*wr - xi[n]*wi;
                si += xr[n]*wi + xi[n]*wr;
            }
            or_[m*K+k] = sr;
            oi[m*K+k] = si;
        }
    }
}

/* ── Twiddle generation ── */
static void gen_flat_tw(double *re, double *im, size_t K) {
    const size_t NN = 32 * K;
    for (int n = 1; n < 32; n++)
        for (size_t k = 0; k < K; k++) {
            double a = -2.0 * M_PI * (double)(n * k) / (double)NN;
            re[(n-1)*K+k] = cos(a);
            im[(n-1)*K+k] = sin(a);
        }
}

static void gen_ladder_tw(double *re, double *im, size_t K) {
    const size_t NN = 32 * K;
    const int pows[5] = {1, 2, 4, 8, 16};
    for (int i = 0; i < 5; i++)
        for (size_t k = 0; k < K; k++) {
            double a = -2.0 * M_PI * (double)(pows[i] * k) / (double)NN;
            re[i*K+k] = cos(a);
            im[i*K+k] = sin(a);
        }
}

static double maxerr(const double *ar, const double *ai,
                     const double *br, const double *bi, size_t n) {
    double mx = 0;
    for (size_t i = 0; i < n; i++) {
        double dr = fabs(ar[i] - br[i]), di = fabs(ai[i] - bi[i]);
        if (dr > mx) mx = dr;
        if (di > mx) mx = di;
    }
    return mx;
}

/* ── Test runner ── */

static int test_notw(size_t K) {
    const size_t NN = 32 * K;
    double *ir = alloc64(NN), *ii = alloc64(NN);
    double *or_ = alloc64(NN), *oi = alloc64(NN);
    double *rr = alloc64(NN), *ri = alloc64(NN);
    int pass = 1;

    srand(42 + (unsigned)K);
    for (size_t i = 0; i < NN; i++) {
        ir[i] = (double)rand()/RAND_MAX - 0.5;
        ii[i] = (double)rand()/RAND_MAX - 0.5;
    }

    /* Forward */
    ref_notw_dft32(ir, ii, rr, ri, K, 1);
    radix32_notw_forward(K, ir, ii, or_, oi);
    double err_fwd = maxerr(rr, ri, or_, oi, NN);
    if (err_fwd > 1e-11) { pass = 0; }

    /* Backward */
    ref_notw_dft32(ir, ii, rr, ri, K, 0);
    radix32_notw_backward(K, ir, ii, or_, oi);
    double err_bwd = maxerr(rr, ri, or_, oi, NN);
    if (err_bwd > 1e-11) { pass = 0; }

    vfft_isa_level_t isa = radix32_effective_isa(K);
    printf("  notw K=%-5zu %-7s  fwd=%.2e bwd=%.2e  %s\n",
           K, radix32_isa_name(isa), err_fwd, err_bwd,
           pass ? "PASS" : "FAIL");

    free64(ir); free64(ii); free64(or_); free64(oi); free64(rr); free64(ri);
    return pass;
}

static int test_tw(size_t K) {
    const size_t NN = 32 * K;
    double *ir = alloc64(NN), *ii = alloc64(NN);
    double *or_ = alloc64(NN), *oi = alloc64(NN);
    double *rr = alloc64(NN), *ri = alloc64(NN);
    int pass = 1;

    srand(123 + (unsigned)K);
    for (size_t i = 0; i < NN; i++) {
        ir[i] = (double)rand()/RAND_MAX - 0.5;
        ii[i] = (double)rand()/RAND_MAX - 0.5;
    }

    /* Twiddle tables */
    double *ftw_re = alloc64(radix32_flat_tw_size(K));
    double *ftw_im = alloc64(radix32_flat_tw_size(K));
    gen_flat_tw(ftw_re, ftw_im, K);

    double *btw_re = NULL, *btw_im = NULL;
    if (radix32_needs_ladder(K)) {
        btw_re = alloc64(radix32_ladder_tw_size(K));
        btw_im = alloc64(radix32_ladder_tw_size(K));
        gen_ladder_tw(btw_re, btw_im, K);
    }

    /* Forward */
    ref_tw_dft32(ir, ii, rr, ri, K, 1);
    radix32_tw_forward(K, ir, ii, or_, oi,
                       ftw_re, ftw_im, btw_re, btw_im);
    double err_fwd = maxerr(rr, ri, or_, oi, NN);
    if (err_fwd > 1e-11) { pass = 0; }

    /* Backward — same twiddle tables! The _bwd kernel applies conjugate
     * multiply internally (re*wr + im*wi, im*wr - re*wi), so the table
     * should contain the same forward-direction twiddles. */
    ref_tw_dft32(ir, ii, rr, ri, K, 0);
    radix32_tw_backward(K, ir, ii, or_, oi,
                        ftw_re, ftw_im, btw_re, btw_im);
    double err_bwd = maxerr(rr, ri, or_, oi, NN);
    if (err_bwd > 1e-11) { pass = 0; }

    const char *variant = "flat";
    if (radix32_needs_ladder(K)) variant = "ladder";
    vfft_isa_level_t isa = radix32_effective_isa(K);
    printf("  tw   K=%-5zu %-7s %-6s fwd=%.2e bwd=%.2e  %s\n",
           K, radix32_isa_name(isa), variant, err_fwd, err_bwd,
           pass ? "PASS" : "FAIL");

    free64(ir); free64(ii); free64(or_); free64(oi); free64(rr); free64(ri);
    free64(ftw_re); free64(ftw_im); free64(btw_re); free64(btw_im);
    return pass;
}

static int test_roundtrip(size_t K) {
    const size_t NN = 32 * K;
    double *ir = alloc64(NN), *ii = alloc64(NN);
    double *mid_re = alloc64(NN), *mid_im = alloc64(NN);
    double *out_re = alloc64(NN), *out_im = alloc64(NN);

    srand(777 + (unsigned)K);
    for (size_t i = 0; i < NN; i++) {
        ir[i] = (double)rand()/RAND_MAX - 0.5;
        ii[i] = (double)rand()/RAND_MAX - 0.5;
    }

    /* Forward then backward: should get back 32 * input */
    radix32_notw_forward(K, ir, ii, mid_re, mid_im);
    radix32_notw_backward(K, mid_re, mid_im, out_re, out_im);

    /* Scale by 1/32 */
    double mx = 0;
    for (size_t i = 0; i < NN; i++) {
        double er = fabs(out_re[i] / 32.0 - ir[i]);
        double ei = fabs(out_im[i] / 32.0 - ii[i]);
        if (er > mx) mx = er;
        if (ei > mx) mx = ei;
    }
    int pass = mx < 1e-12;
    printf("  roundtrip K=%-5zu %-7s  err=%.2e  %s\n",
           K, radix32_isa_name(radix32_effective_isa(K)), mx,
           pass ? "PASS" : "FAIL");

    free64(ir); free64(ii); free64(mid_re); free64(mid_im);
    free64(out_re); free64(out_im);
    return pass;
}

int main(void) {
    int total = 0, passed = 0;

    printf("═══ ISA Detection ═══\n");
    printf("  Hardware ISA: %s\n\n", radix32_isa_name(vfft_detect_isa()));

    /* K values that exercise every dispatch path:
     *   1,2,3  → scalar (not aligned to 4)
     *   4,12   → AVX2 (aligned to 4 but not 8)
     *   8,16   → AVX-512 (aligned to 8)
     *   32,64  → AVX-512 flat
     *   128    → AVX-512 ladder U1
     *   256+   → AVX-512 ladder U2
     */
    size_t test_Ks[] = {1, 2, 3, 4, 8, 12, 16, 32, 64, 128, 256, 512, 1024};
    int nK = sizeof(test_Ks) / sizeof(test_Ks[0]);

    printf("═══ NOTW (twiddle-less DFT-32) ═══\n");
    for (int i = 0; i < nK; i++) {
        total++; passed += test_notw(test_Ks[i]);
    }

    printf("\n═══ TW (twiddled DFT-32) ═══\n");
    for (int i = 0; i < nK; i++) {
        total++; passed += test_tw(test_Ks[i]);
    }

    printf("\n═══ ROUNDTRIP (fwd→bwd, should recover input) ═══\n");
    for (int i = 0; i < nK; i++) {
        total++; passed += test_roundtrip(test_Ks[i]);
    }

    printf("\n═══ SUMMARY ═══\n");
    printf("  %d / %d tests passed\n", passed, total);
    printf("  needs_flat(64)=%d  needs_flat(128)=%d\n",
           radix32_needs_flat(64), radix32_needs_flat(128));
    printf("  needs_ladder(64)=%d  needs_ladder(128)=%d\n",
           radix32_needs_ladder(64), radix32_needs_ladder(128));

    return (passed == total) ? 0 : 1;
}