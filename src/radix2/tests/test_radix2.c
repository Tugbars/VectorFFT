/**
 * test_radix2.c — Radix-2 standalone codelet tests
 * Portable: Windows + Linux, AVX-512 / AVX2 / scalar
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Portable aligned allocation */
#ifdef _MSC_VER
#include <malloc.h>
static double *alloc64(size_t n) {
    double *p = (double *)_aligned_malloc(n * sizeof(double), 64);
    if (p) memset(p, 0, n * sizeof(double));
    return p;
}
static void free64(void *p) { _aligned_free(p); }
#else
static double *alloc64(size_t n) {
    double *p = NULL;
    posix_memalign((void**)&p, 64, n * sizeof(double));
    if (p) memset(p, 0, n * sizeof(double));
    return p;
}
static void free64(void *p) { free(p); }
#endif

#include "fft_radix2_dispatch.h"

/* Reference DFT-2 notw */
static void ref_notw_dft2(const double *ir, const double *ii,
                          double *or_, double *oi, size_t K, int fwd) {
    (void)fwd; /* DFT-2 notw is direction-independent */
    for (size_t k = 0; k < K; k++) {
        or_[0*K+k] = ir[0*K+k] + ir[1*K+k];
        oi[0*K+k] = ii[0*K+k] + ii[1*K+k];
        or_[1*K+k] = ir[0*K+k] - ir[1*K+k];
        oi[1*K+k] = ii[0*K+k] - ii[1*K+k];
    }
}

/* Reference DFT-2 twiddled */
static void ref_tw_dft2(const double *ir, const double *ii,
                        double *or_, double *oi, size_t K, int fwd) {
    const size_t NN = 2 * K;
    const double sign = fwd ? -1.0 : 1.0;
    for (size_t k = 0; k < K; k++) {
        double x0r = ir[k], x0i = ii[k];
        double r1r = ir[K+k], r1i = ii[K+k];
        double a = sign * 2.0 * M_PI * (double)k / (double)NN;
        double wr = cos(a), wi = sin(a);
        double x1r = r1r*wr - r1i*wi;
        double x1i = r1r*wi + r1i*wr;
        or_[0*K+k] = x0r + x1r;
        oi[0*K+k] = x0i + x1i;
        or_[1*K+k] = x0r - x1r;
        oi[1*K+k] = x0i - x1i;
    }
}

static void gen_flat_tw(double *re, double *im, size_t K) {
    const size_t NN = 2 * K;
    for (size_t k = 0; k < K; k++) {
        double a = -2.0 * M_PI * (double)k / (double)NN;
        re[k] = cos(a);
        im[k] = sin(a);
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

int main(void) {
    int total = 0, passed = 0;
    size_t Ks[] = {1, 2, 3, 4, 7, 8, 12, 16, 32, 64, 128, 256};
    const int nK = 12;

    printf("=== Radix-2 Codelet Correctness Tests ===\n\n");

    for (int ki = 0; ki < nK; ki++) {
        size_t K = Ks[ki], NN = 2 * K;
        double *ir    = alloc64(NN);
        double *ii    = alloc64(NN);
        double *ref_r = alloc64(NN);
        double *ref_i = alloc64(NN);
        double *out_r = alloc64(NN);
        double *out_i = alloc64(NN);
        double *twr   = alloc64(K);
        double *twi   = alloc64(K);

        srand(42 + (unsigned)K);
        for (size_t i = 0; i < NN; i++) {
            ir[i] = (double)rand() / RAND_MAX - 0.5;
            ii[i] = (double)rand() / RAND_MAX - 0.5;
        }
        gen_flat_tw(twr, twi, K);

        for (int fwd = 1; fwd >= 0; fwd--) {
            const char *dir = fwd ? "fwd" : "bwd";

            /* notw */
            ref_notw_dft2(ir, ii, ref_r, ref_i, K, fwd);
            if (fwd) radix2_notw_forward(K, ir, ii, out_r, out_i);
            else     radix2_notw_backward(K, ir, ii, out_r, out_i);
            {
                double e = maxerr(ref_r, ref_i, out_r, out_i, NN);
                int ok = e < 1e-14;
                total++; passed += ok;
                printf("  notw %s K=%-4zu dispatch err=%.1e %s\n", dir, K, e, ok ? "PASS" : "FAIL");
            }

            /* tw */
            ref_tw_dft2(ir, ii, ref_r, ref_i, K, fwd);
            if (fwd) radix2_tw_forward(K, ir, ii, out_r, out_i, twr, twi);
            else     radix2_tw_backward(K, ir, ii, out_r, out_i, twr, twi);
            {
                double e = maxerr(ref_r, ref_i, out_r, out_i, NN);
                int ok = e < 1e-14;
                total++; passed += ok;
                printf("  tw   %s K=%-4zu dispatch err=%.1e %s\n", dir, K, e, ok ? "PASS" : "FAIL");
            }
        }

        free64(ir); free64(ii); free64(ref_r); free64(ref_i);
        free64(out_r); free64(out_i); free64(twr); free64(twi);
    }

    printf("\n=== %d / %d passed ===\n", passed, total);
    return (passed == total) ? 0 : 1;
}
