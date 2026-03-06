/**
 * test_radix7_dispatch.c — Correctness test for fft_radix7_dispatch.h
 *
 * Tests all dispatch paths: notw + tw × fwd + bwd × scalar + AVX2 + AVX-512
 * Plus roundtrip: fwd → bwd should recover input (up to scale 7).
 *
 * Build:
 *   gcc -O3 -march=native -mavx512f -mavx512dq -mavx2 -mfma \
 *       -o test_radix7_dispatch test_radix7_dispatch.c -lm
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

#include "fft_radix7_dispatch.h"

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

/* ── Scalar reference DFT-7 ── */
static void ref_notw_dft7(const double *ir, const double *ii,
                          double *or_, double *oi, size_t K, int fwd) {
    const double sign = fwd ? -1.0 : 1.0;
    for (size_t k = 0; k < K; k++) {
        for (int m = 0; m < 7; m++) {
            double sr = 0, si = 0;
            for (int n = 0; n < 7; n++) {
                double a = sign * 2.0 * M_PI * (double)(m * n) / 7.0;
                double wr = cos(a), wi = sin(a);
                sr += ir[n*K+k]*wr - ii[n*K+k]*wi;
                si += ir[n*K+k]*wi + ii[n*K+k]*wr;
            }
            or_[m*K+k] = sr;
            oi[m*K+k] = si;
        }
    }
}

static void ref_tw_dft7(const double *ir, const double *ii,
                        double *or_, double *oi, size_t K, int fwd) {
    const size_t NN = 7 * K;
    const double sign = fwd ? -1.0 : 1.0;
    for (size_t k = 0; k < K; k++) {
        double xr[7], xi[7];
        for (int n = 0; n < 7; n++) {
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
        for (int m = 0; m < 7; m++) {
            double sr = 0, si = 0;
            for (int n = 0; n < 7; n++) {
                double a = sign * 2.0 * M_PI * (double)(m * n) / 7.0;
                double wr = cos(a), wi = sin(a);
                sr += xr[n]*wr - xi[n]*wi;
                si += xr[n]*wi + xi[n]*wr;
            }
            or_[m*K+k] = sr;
            oi[m*K+k] = si;
        }
    }
}

static void gen_flat_tw(double *re, double *im, size_t K) {
    const size_t NN = 7 * K;
    for (int n = 1; n < 7; n++)
        for (size_t k = 0; k < K; k++) {
            double a = -2.0 * M_PI * (double)(n * k) / (double)NN;
            re[(n-1)*K+k] = cos(a);
            im[(n-1)*K+k] = sin(a);
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

/* ── Test runners ── */

static int test_notw(size_t K) {
    const size_t NN = 7 * K;
    double *ir = alloc64(NN), *ii = alloc64(NN);
    double *or_ = alloc64(NN), *oi = alloc64(NN);
    double *rr = alloc64(NN), *ri = alloc64(NN);
    int pass = 1;

    srand(42 + (unsigned)K);
    for (size_t i = 0; i < NN; i++) {
        ir[i] = (double)rand()/RAND_MAX - 0.5;
        ii[i] = (double)rand()/RAND_MAX - 0.5;
    }

    ref_notw_dft7(ir, ii, rr, ri, K, 1);
    radix7_notw_forward(K, ir, ii, or_, oi);
    double err_fwd = maxerr(rr, ri, or_, oi, NN);
    if (err_fwd > 1e-11) pass = 0;

    ref_notw_dft7(ir, ii, rr, ri, K, 0);
    radix7_notw_backward(K, ir, ii, or_, oi);
    double err_bwd = maxerr(rr, ri, or_, oi, NN);
    if (err_bwd > 1e-11) pass = 0;

    vfft_isa_level_t isa = radix7_effective_isa(K);
    printf("  notw K=%-5zu %-7s  fwd=%.2e bwd=%.2e  %s\n",
           K, radix7_isa_name(isa), err_fwd, err_bwd,
           pass ? "PASS" : "FAIL");

    free64(ir); free64(ii); free64(or_); free64(oi); free64(rr); free64(ri);
    return pass;
}

static int test_tw(size_t K) {
    const size_t NN = 7 * K;
    double *ir = alloc64(NN), *ii = alloc64(NN);
    double *or_ = alloc64(NN), *oi = alloc64(NN);
    double *rr = alloc64(NN), *ri = alloc64(NN);
    int pass = 1;

    srand(123 + (unsigned)K);
    for (size_t i = 0; i < NN; i++) {
        ir[i] = (double)rand()/RAND_MAX - 0.5;
        ii[i] = (double)rand()/RAND_MAX - 0.5;
    }

    double *twr = alloc64(radix7_flat_tw_size(K));
    double *twi = alloc64(radix7_flat_tw_size(K));
    gen_flat_tw(twr, twi, K);

    ref_tw_dft7(ir, ii, rr, ri, K, 1);
    radix7_tw_forward(K, ir, ii, or_, oi, twr, twi);
    double err_fwd = maxerr(rr, ri, or_, oi, NN);
    if (err_fwd > 1e-11) pass = 0;

    ref_tw_dft7(ir, ii, rr, ri, K, 0);
    radix7_tw_backward(K, ir, ii, or_, oi, twr, twi);
    double err_bwd = maxerr(rr, ri, or_, oi, NN);
    if (err_bwd > 1e-11) pass = 0;

    vfft_isa_level_t isa = radix7_effective_isa(K);
    printf("  tw   K=%-5zu %-7s  fwd=%.2e bwd=%.2e  %s\n",
           K, radix7_isa_name(isa), err_fwd, err_bwd,
           pass ? "PASS" : "FAIL");

    free64(ir); free64(ii); free64(or_); free64(oi); free64(rr); free64(ri);
    free64(twr); free64(twi);
    return pass;
}

static int test_roundtrip(size_t K) {
    const size_t NN = 7 * K;
    double *ir = alloc64(NN), *ii = alloc64(NN);
    double *mid_re = alloc64(NN), *mid_im = alloc64(NN);
    double *out_re = alloc64(NN), *out_im = alloc64(NN);

    srand(777 + (unsigned)K);
    for (size_t i = 0; i < NN; i++) {
        ir[i] = (double)rand()/RAND_MAX - 0.5;
        ii[i] = (double)rand()/RAND_MAX - 0.5;
    }

    radix7_notw_forward(K, ir, ii, mid_re, mid_im);
    radix7_notw_backward(K, mid_re, mid_im, out_re, out_im);

    double mx = 0;
    for (size_t i = 0; i < NN; i++) {
        double er = fabs(out_re[i] / 7.0 - ir[i]);
        double ei = fabs(out_im[i] / 7.0 - ii[i]);
        if (er > mx) mx = er;
        if (ei > mx) mx = ei;
    }
    int pass = mx < 1e-12;
    printf("  roundtrip K=%-5zu %-7s  err=%.2e  %s\n",
           K, radix7_isa_name(radix7_effective_isa(K)), mx,
           pass ? "PASS" : "FAIL");

    free64(ir); free64(ii); free64(mid_re); free64(mid_im);
    free64(out_re); free64(out_im);
    return pass;
}

int main(void) {
    int total = 0, passed = 0;

    printf("═══ ISA Detection ═══\n");
    printf("  Hardware ISA: %s\n\n", radix7_isa_name(vfft_detect_isa()));

    size_t test_Ks[] = {1, 2, 3, 4, 7, 8, 12, 16, 32, 64, 128, 256, 512, 1024};
    int nK = sizeof(test_Ks) / sizeof(test_Ks[0]);

    printf("═══ NOTW (twiddle-less DFT-7) ═══\n");
    for (int i = 0; i < nK; i++) { total++; passed += test_notw(test_Ks[i]); }

    printf("\n═══ TW (twiddled DFT-7) ═══\n");
    for (int i = 0; i < nK; i++) { total++; passed += test_tw(test_Ks[i]); }

    printf("\n═══ ROUNDTRIP (fwd→bwd, should recover input) ═══\n");
    for (int i = 0; i < nK; i++) { total++; passed += test_roundtrip(test_Ks[i]); }

    printf("\n═══ SUMMARY ═══\n");
    printf("  %d / %d tests passed\n", passed, total);

    return (passed == total) ? 0 : 1;
}
