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

#if defined(__AVX512F__) || defined(__AVX512F)
#include <immintrin.h>
#define R8_512_LD(p) _mm512_loadu_pd(p)
#define R8_512_ST(p,v) _mm512_storeu_pd((p),(v))
#endif
#ifdef __AVX2__
#include <immintrin.h>
#define R8_256_LD(p) _mm256_loadu_pd(p)
#define R8_256_ST(p,v) _mm256_storeu_pd((p),(v))
#endif

#include "fft_radix8_dispatch.h"

/* ═══════════════════════════════════════════════════════════════
 * REFERENCE DFT-8
 * ═══════════════════════════════════════════════════════════════ */

static void ref_notw_dft8(const double *ir, const double *ii,
                          double *or_, double *oi, size_t K, int fwd) {
    const double sign = fwd ? -1.0 : 1.0;
    for (size_t k = 0; k < K; k++)
        for (int m = 0; m < 8; m++) {
            double sr = 0, si = 0;
            for (int n = 0; n < 8; n++) {
                double a = sign * 2.0 * M_PI * (double)(m * n) / 8.0;
                sr += ir[n*K+k]*cos(a) - ii[n*K+k]*sin(a);
                si += ir[n*K+k]*sin(a) + ii[n*K+k]*cos(a);
            }
            or_[m*K+k] = sr;
            oi[m*K+k] = si;
        }
}

static void ref_tw_dft8(const double *ir, const double *ii,
                        double *or_, double *oi, size_t K, int fwd) {
    const size_t NN = 8 * K;
    const double sign = fwd ? -1.0 : 1.0;
    for (size_t k = 0; k < K; k++) {
        double xr[8], xi[8];
        for (int n = 0; n < 8; n++) {
            double dr = ir[n*K+k], di = ii[n*K+k];
            if (n > 0) {
                double a = sign * 2.0 * M_PI * (double)(n * k) / (double)NN;
                double wr = cos(a), wi = sin(a);
                double tr = dr*wr - di*wi;
                di = dr*wi + di*wr;
                dr = tr;
            }
            xr[n] = dr;
            xi[n] = di;
        }
        for (int m = 0; m < 8; m++) {
            double sr = 0, si = 0;
            for (int n = 0; n < 8; n++) {
                double a = sign * 2.0 * M_PI * (double)(m * n) / 8.0;
                sr += xr[n]*cos(a) - xi[n]*sin(a);
                si += xr[n]*sin(a) + xi[n]*cos(a);
            }
            or_[m*K+k] = sr;
            oi[m*K+k] = si;
        }
    }
}

static void gen_flat_tw(double *re, double *im, size_t K) {
    const size_t NN = 8 * K;
    for (int n = 1; n < 8; n++)
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

int main(void) {
    int total = 0, passed = 0;
    size_t Ks[] = {1, 2, 3, 4, 7, 8, 12, 16, 32, 64, 128, 256};
    const int nK = sizeof(Ks) / sizeof(Ks[0]);

    printf("=== Radix-8 Codelet Correctness Tests ===\n\n");

    for (int ki = 0; ki < nK; ki++) {
        size_t K = Ks[ki], NN = 8 * K;
        double *ir    = alloc64(NN);
        double *ii    = alloc64(NN);
        double *ref_r = alloc64(NN);
        double *ref_i = alloc64(NN);
        double *out_r = alloc64(NN);
        double *out_i = alloc64(NN);
        double *twr   = alloc64(7 * K);
        double *twi   = alloc64(7 * K);

        srand(42 + (unsigned)K);
        for (size_t i = 0; i < NN; i++) {
            ir[i] = (double)rand() / RAND_MAX - 0.5;
            ii[i] = (double)rand() / RAND_MAX - 0.5;
        }
        gen_flat_tw(twr, twi, K);

        for (int fwd = 1; fwd >= 0; fwd--) {
            const char *dir = fwd ? "fwd" : "bwd";

            /* ── NOTW ── */
            ref_notw_dft8(ir, ii, ref_r, ref_i, K, fwd);

            if (fwd) radix8_notw_forward(K, ir, ii, out_r, out_i);
            else     radix8_notw_backward(K, ir, ii, out_r, out_i);
            {
                double e = maxerr(ref_r, ref_i, out_r, out_i, NN);
                int ok = e < 1e-11;
                total++; passed += ok;
                printf("  notw %s K=%-4zu dispatch err=%.1e %s\n", dir, K, e, ok ? "PASS" : "FAIL");
            }

            /* ── TW ── */
            ref_tw_dft8(ir, ii, ref_r, ref_i, K, fwd);

            if (fwd) radix8_tw_forward(K, ir, ii, out_r, out_i, twr, twi);
            else     radix8_tw_backward(K, ir, ii, out_r, out_i, twr, twi);
            {
                double e = maxerr(ref_r, ref_i, out_r, out_i, NN);
                int ok = e < 1e-11;
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