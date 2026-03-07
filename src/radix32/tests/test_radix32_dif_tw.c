/**
 * test_radix32_dif_tw.c — Verify DIF twiddled DFT-32 codelets
 *
 * DIF tw codelet computes: DFT-32(input) then multiply output by twiddle
 * DIT tw codelet computes: multiply input by twiddle then DFT-32(input)
 *
 * Test: compare DIF tw codelet against naive reference:
 *   1. Compute notw DFT-32 (naive)
 *   2. Multiply outputs 1..31 by twiddle table
 *   3. Compare against DIF tw codelet output
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Include the DIF tw headers */
#include "fft_radix32_scalar_dif_tw.h"

#ifdef __AVX2__
#include "fft_radix32_avx2_dif_tw.h"
#endif

#ifdef __AVX512F__
#include "fft_radix32_avx512_dif_tw.h"
#endif

/* Also include notw for reference */
#include "fft_radix32_scalar_notw.h"

static double *aa64(size_t n) {
    void *p = NULL;
    posix_memalign(&p, 64, n * sizeof(double));
    memset(p, 0, n * sizeof(double));
    return (double *)p;
}

/**
 * Build flat twiddle table for DFT of size R*K.
 * tw_re[(n-1)*K + k] = cos(2π·n·k / (R*K))
 * tw_im[(n-1)*K + k] = dir·sin(2π·n·k / (R*K))
 * dir = -1 for forward, +1 for backward
 */
static void build_twiddles(size_t R, size_t K, int dir,
                           double *tw_re, double *tw_im)
{
    const size_t N = R * K;
    for (size_t n = 1; n < R; n++)
        for (size_t k = 0; k < K; k++) {
            double a = 2.0 * M_PI * (double)n * (double)k / (double)N;
            tw_re[(n-1)*K + k] = cos(a);
            tw_im[(n-1)*K + k] = dir * sin(a);
        }
}

/**
 * Reference: naive DFT-32 at stride K, for a single k
 */
static void naive_dft32(const double *in_re, const double *in_im,
                        double *out_re, double *out_im,
                        size_t K, size_t k)
{
    for (size_t m = 0; m < 32; m++) {
        double sr = 0, si = 0;
        for (size_t n = 0; n < 32; n++) {
            double a = -2.0 * M_PI * (double)m * (double)n / 32.0;
            sr += in_re[n*K + k] * cos(a) - in_im[n*K + k] * sin(a);
            si += in_re[n*K + k] * sin(a) + in_im[n*K + k] * cos(a);
        }
        out_re[m] = sr;
        out_im[m] = si;
    }
}

static void naive_idft32(const double *in_re, const double *in_im,
                         double *out_re, double *out_im,
                         size_t K, size_t k)
{
    for (size_t m = 0; m < 32; m++) {
        double sr = 0, si = 0;
        for (size_t n = 0; n < 32; n++) {
            double a = 2.0 * M_PI * (double)m * (double)n / 32.0;
            sr += in_re[n*K + k] * cos(a) - in_im[n*K + k] * sin(a);
            si += in_re[n*K + k] * sin(a) + in_im[n*K + k] * cos(a);
        }
        out_re[m] = sr;
        out_im[m] = si;
    }
}

/**
 * Reference DIF operation:
 *   1. DFT-32 on stride-K input
 *   2. Multiply outputs 1..31 by twiddle
 */
static void ref_dif_fwd(const double *in_re, const double *in_im,
                        double *out_re, double *out_im,
                        const double *tw_re, const double *tw_im,
                        size_t K)
{
    double dft_re[32], dft_im[32];
    for (size_t k = 0; k < K; k++) {
        naive_dft32(in_re, in_im, dft_re, dft_im, K, k);
        /* Apply external twiddle to outputs 1..31 */
        out_re[0*K + k] = dft_re[0];
        out_im[0*K + k] = dft_im[0];
        for (size_t m = 1; m < 32; m++) {
            double wr = tw_re[(m-1)*K + k];
            double wi = tw_im[(m-1)*K + k];
            out_re[m*K + k] = dft_re[m]*wr - dft_im[m]*wi;
            out_im[m*K + k] = dft_re[m]*wi + dft_im[m]*wr;
        }
    }
}

static void ref_dif_bwd(const double *in_re, const double *in_im,
                        double *out_re, double *out_im,
                        const double *tw_re, const double *tw_im,
                        size_t K)
{
    double dft_re[32], dft_im[32];
    for (size_t k = 0; k < K; k++) {
        naive_idft32(in_re, in_im, dft_re, dft_im, K, k);
        /* Apply conjugated twiddle to outputs 1..31 */
        out_re[0*K + k] = dft_re[0];
        out_im[0*K + k] = dft_im[0];
        for (size_t m = 1; m < 32; m++) {
            double wr = tw_re[(m-1)*K + k];
            double wi = tw_im[(m-1)*K + k];
            /* Conjugate: wr + j*wi → wr - j*wi */
            out_re[m*K + k] = dft_re[m]*wr + dft_im[m]*wi;
            out_im[m*K + k] = dft_im[m]*wr - dft_re[m]*wi;
        }
    }
}

static int test_dif_tw(size_t K, const char *isa_name,
    void (*dif_fwd)(const double *, const double *, double *, double *,
                    const double *, const double *, size_t),
    void (*dif_bwd)(const double *, const double *, double *, double *,
                    const double *, const double *, size_t))
{
    const size_t N = 32 * K;
    double *in_re = aa64(N), *in_im = aa64(N);
    double *ref_re = aa64(N), *ref_im = aa64(N);
    double *got_re = aa64(N), *got_im = aa64(N);
    double *tw_re = aa64(31 * K), *tw_im = aa64(31 * K);

    srand(42 + (unsigned)K);
    for (size_t i = 0; i < N; i++) {
        in_re[i] = (double)rand()/RAND_MAX * 2.0 - 1.0;
        in_im[i] = (double)rand()/RAND_MAX * 2.0 - 1.0;
    }

    /* Forward */
    build_twiddles(32, K, -1, tw_re, tw_im);
    ref_dif_fwd(in_re, in_im, ref_re, ref_im, tw_re, tw_im, K);
    dif_fwd(in_re, in_im, got_re, got_im, tw_re, tw_im, K);

    double fwd_err = 0, mag = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(ref_re[i]-got_re[i]), fabs(ref_im[i]-got_im[i]));
        double m = fmax(fabs(ref_re[i]), fabs(ref_im[i]));
        if (e > fwd_err) fwd_err = e;
        if (m > mag) mag = m;
    }
    double fwd_rel = mag > 0 ? fwd_err / mag : fwd_err;

    /* Backward */
    build_twiddles(32, K, -1, tw_re, tw_im);  /* same table, bwd codelet conjugates */
    ref_dif_bwd(in_re, in_im, ref_re, ref_im, tw_re, tw_im, K);
    dif_bwd(in_re, in_im, got_re, got_im, tw_re, tw_im, K);

    double bwd_err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(ref_re[i]-got_re[i]), fabs(ref_im[i]-got_im[i]));
        if (e > bwd_err) bwd_err = e;
    }
    double bwd_rel = mag > 0 ? bwd_err / mag : bwd_err;

    double tol = 1e-13;
    int pass = (fwd_rel < tol) && (bwd_rel < tol);

    printf("  K=%-4zu  %-8s  fwd=%.1e  bwd=%.1e  %s\n",
           K, isa_name, fwd_rel, bwd_rel, pass ? "PASS" : "FAIL");

    free(in_re); free(in_im); free(ref_re); free(ref_im);
    free(got_re); free(got_im); free(tw_re); free(tw_im);
    return pass;
}

int main(void) {
    printf("=== Radix-32 DIF Twiddled Codelet Test ===\n\n");

    int p = 0, t = 0;
    size_t Ks[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};

    printf("── Scalar DIF tw ──\n");
    for (size_t i = 0; i < sizeof(Ks)/sizeof(Ks[0]); i++) {
        t++; p += test_dif_tw(Ks[i], "scalar",
            radix32_tw_flat_dif_kernel_fwd_scalar,
            radix32_tw_flat_dif_kernel_bwd_scalar);
    }

#ifdef __AVX2__
    printf("\n── AVX2 DIF tw ──\n");
    size_t avx2_Ks[] = {4, 8, 16, 32, 64, 128, 256};
    for (size_t i = 0; i < sizeof(avx2_Ks)/sizeof(avx2_Ks[0]); i++) {
        t++; p += test_dif_tw(avx2_Ks[i], "avx2",
            radix32_tw_flat_dif_kernel_fwd_avx2,
            radix32_tw_flat_dif_kernel_bwd_avx2);
    }
#endif

#ifdef __AVX512F__
    printf("\n── AVX-512 DIF tw ──\n");
    size_t avx512_Ks[] = {8, 16, 32, 64, 128, 256};
    for (size_t i = 0; i < sizeof(avx512_Ks)/sizeof(avx512_Ks[0]); i++) {
        t++; p += test_dif_tw(avx512_Ks[i], "avx512",
            radix32_tw_flat_dif_kernel_fwd_avx512,
            radix32_tw_flat_dif_kernel_bwd_avx512);
    }
#endif

    printf("\n=== %d/%d %s ===\n", p, t, p == t ? "ALL PASSED" : "FAILURES");
    return p != t;
}
