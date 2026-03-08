#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* DIT tw */
#include "fft_radix25_scalar_tw.h"
#ifdef __AVX2__
#include "fft_radix25_avx2_tw.h"
#endif
#ifdef __AVX512F__
#include "fft_radix25_avx512_tw.h"
#endif

/* DIF tw (separate files now) */
#include "fft_radix25_scalar_dif_tw.h"
#ifdef __AVX2__
#include "fft_radix25_avx2_dif_tw.h"
#endif
#ifdef __AVX512F__
#include "fft_radix25_avx512_dif_tw.h"
#endif

static double *aa64(size_t n)
{
#ifdef _WIN32
    double *p = (double *)_aligned_malloc(n * sizeof(double), 64);
#else
    double *p;
    posix_memalign((void **)&p, 64, n * sizeof(double));
#endif
    memset(p, 0, n * sizeof(double));
    return p;
}
static void aa64_free(void *p)
{
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

static void build_tw(size_t K, double *twr, double *twi)
{
    const size_t NN = 25 * K;
    for (size_t n = 1; n < 25; n++)
        for (size_t k = 0; k < K; k++)
        {
            double a = 2.0 * M_PI * (double)n * (double)k / (double)NN;
            twr[(n - 1) * K + k] = cos(a);
            twi[(n - 1) * K + k] = -sin(a);
        }
}

static void naive_dft25(const double *ir, const double *ii, double *dr, double *di,
                        size_t K, size_t k, int dir)
{
    double s = (dir < 0) ? -1.0 : 1.0;
    for (size_t m = 0; m < 25; m++)
    {
        double sr = 0, si = 0;
        for (size_t n = 0; n < 25; n++)
        {
            double a = s * 2.0 * M_PI * (double)m * (double)n / 25.0;
            sr += ir[n * K + k] * cos(a) - ii[n * K + k] * sin(a);
            si += ir[n * K + k] * sin(a) + ii[n * K + k] * cos(a);
        }
        dr[m] = sr;
        di[m] = si;
    }
}

typedef void (*tw_fn)(const double *, const double *, double *, double *,
                      const double *, const double *, size_t);

static int test_dit(size_t K, const char *isa, tw_fn fwd, tw_fn bwd)
{
    size_t N = 25 * K;
    double *ir = aa64(N), *ii = aa64(N), *ref_r = aa64(N), *ref_i = aa64(N);
    double *got_r = aa64(N), *got_i = aa64(N);
    double *twr = aa64(24 * K), *twi = aa64(24 * K);
    double *tir = aa64(N), *tii = aa64(N);
    srand(42 + (unsigned)K);
    for (size_t i = 0; i < N; i++)
    {
        ir[i] = (double)rand() / RAND_MAX * 2 - 1;
        ii[i] = (double)rand() / RAND_MAX * 2 - 1;
    }
    build_tw(K, twr, twi);

    /* Fwd ref: twiddle inputs then naive DFT */
    memcpy(tir, ir, N * sizeof(double));
    memcpy(tii, ii, N * sizeof(double));
    for (size_t k = 0; k < K; k++)
        for (size_t n = 1; n < 25; n++)
        {
            double wr = twr[(n - 1) * K + k], wi = twi[(n - 1) * K + k], a = tir[n * K + k], b = tii[n * K + k];
            tir[n * K + k] = a * wr - b * wi;
            tii[n * K + k] = a * wi + b * wr;
        }
    double dr[25], di[25];
    for (size_t k = 0; k < K; k++)
    {
        naive_dft25(tir, tii, dr, di, K, k, -1);
        for (size_t m = 0; m < 25; m++)
        {
            ref_r[m * K + k] = dr[m];
            ref_i[m * K + k] = di[m];
        }
    }
    fwd(ir, ii, got_r, got_i, twr, twi, K);
    double err = 0, mag = 0;
    for (size_t i = 0; i < N; i++)
    {
        double e = fmax(fabs(ref_r[i] - got_r[i]), fabs(ref_i[i] - got_i[i]));
        double m = fmax(fabs(ref_r[i]), fabs(ref_i[i]));
        if (e > err)
            err = e;
        if (m > mag)
            mag = m;
    }
    double frel = mag > 0 ? err / mag : err;

    /* Bwd ref: conj twiddle inputs then naive IDFT */
    memcpy(tir, ir, N * sizeof(double));
    memcpy(tii, ii, N * sizeof(double));
    for (size_t k = 0; k < K; k++)
        for (size_t n = 1; n < 25; n++)
        {
            double wr = twr[(n - 1) * K + k], wi = twi[(n - 1) * K + k], a = tir[n * K + k], b = tii[n * K + k];
            tir[n * K + k] = a * wr + b * wi;
            tii[n * K + k] = b * wr - a * wi;
        }
    for (size_t k = 0; k < K; k++)
    {
        naive_dft25(tir, tii, dr, di, K, k, +1);
        for (size_t m = 0; m < 25; m++)
        {
            ref_r[m * K + k] = dr[m];
            ref_i[m * K + k] = di[m];
        }
    }
    bwd(ir, ii, got_r, got_i, twr, twi, K);
    err = 0;
    mag = 0;
    for (size_t i = 0; i < N; i++)
    {
        double e = fmax(fabs(ref_r[i] - got_r[i]), fabs(ref_i[i] - got_i[i]));
        double m = fmax(fabs(ref_r[i]), fabs(ref_i[i]));
        if (e > err)
            err = e;
        if (m > mag)
            mag = m;
    }
    double brel = mag > 0 ? err / mag : err;

    int pass = frel < 1e-12 && brel < 1e-12;
    printf("  K=%-4zu %-8s DIT fwd=%.1e bwd=%.1e %s\n", K, isa, frel, brel, pass ? "PASS" : "FAIL");
    aa64_free(ir);
    aa64_free(ii);
    aa64_free(ref_r);
    aa64_free(ref_i);
    aa64_free(got_r);
    aa64_free(got_i);
    aa64_free(twr);
    aa64_free(twi);
    aa64_free(tir);
    aa64_free(tii);
    return pass;
}

static int test_roundtrip(size_t K, const char *isa, tw_fn dit_fwd, tw_fn dif_bwd)
{
    size_t N = 25 * K;
    double *ir = aa64(N), *ii = aa64(N), *fr = aa64(N), *fi = aa64(N), *rr = aa64(N), *ri = aa64(N);
    double *twr = aa64(24 * K), *twi = aa64(24 * K);
    srand(99 + (unsigned)K);
    for (size_t i = 0; i < N; i++)
    {
        ir[i] = (double)rand() / RAND_MAX * 2 - 1;
        ii[i] = (double)rand() / RAND_MAX * 2 - 1;
    }
    build_tw(K, twr, twi);
    dit_fwd(ir, ii, fr, fi, twr, twi, K);
    dif_bwd(fr, fi, rr, ri, twr, twi, K);
    double err = 0, mag = 0;
    for (size_t i = 0; i < N; i++)
    {
        rr[i] /= 25.0;
        ri[i] /= 25.0;
        double e = fmax(fabs(ir[i] - rr[i]), fabs(ii[i] - ri[i]));
        double m = fmax(fabs(ir[i]), fabs(ii[i]));
        if (e > err)
            err = e;
        if (m > mag)
            mag = m;
    }
    double rel = mag > 0 ? err / mag : err;
    int pass = rel < 1e-12;
    printf("  K=%-4zu %-8s roundtrip=%.1e %s\n", K, isa, rel, pass ? "PASS" : "FAIL");
    aa64_free(ir);
    aa64_free(ii);
    aa64_free(fr);
    aa64_free(fi);
    aa64_free(rr);
    aa64_free(ri);
    aa64_free(twr);
    aa64_free(twi);
    return pass;
}

int main(void)
{
    printf("=== Radix-25 (5x5 CT) Log3 Twiddled Test ===\n\n");
    int p = 0, t = 0;
    size_t Ks[] = {1, 2, 4, 8, 16, 32, 64};

    printf("-- Scalar DIT (fwd+bwd) --\n");
    for (size_t i = 0; i < 7; i++)
    {
        t++;
        p += test_dit(Ks[i], "scalar",
                      radix25_tw_flat_dit_kernel_fwd_scalar, radix25_tw_flat_dit_kernel_bwd_scalar);
    }

    printf("\n-- Scalar roundtrip (DIT fwd -> DIF bwd) --\n");
    for (size_t i = 0; i < 7; i++)
    {
        t++;
        p += test_roundtrip(Ks[i], "scalar",
                            radix25_tw_flat_dit_kernel_fwd_scalar, radix25_tw_flat_dif_kernel_bwd_scalar);
    }

#ifdef __AVX2__
    printf("\n-- AVX2 DIT (fwd+bwd) --\n");
    size_t aKs[] = {4, 8, 16, 32, 64};
    for (size_t i = 0; i < 5; i++)
    {
        t++;
        p += test_dit(aKs[i], "avx2",
                      radix25_tw_flat_dit_kernel_fwd_avx2, radix25_tw_flat_dit_kernel_bwd_avx2);
    }

    printf("\n-- AVX2 roundtrip (DIT fwd -> DIF bwd) --\n");
    for (size_t i = 0; i < 5; i++)
    {
        t++;
        p += test_roundtrip(aKs[i], "avx2",
                            radix25_tw_flat_dit_kernel_fwd_avx2, radix25_tw_flat_dif_kernel_bwd_avx2);
    }
#endif

#ifdef __AVX512F__
    printf("\n-- AVX-512 DIT (fwd+bwd) --\n");
    size_t sKs[] = {8, 16, 32, 64};
    for (size_t i = 0; i < 4; i++)
    {
        t++;
        p += test_dit(sKs[i], "avx512",
                      radix25_tw_flat_dit_kernel_fwd_avx512, radix25_tw_flat_dit_kernel_bwd_avx512);
    }

    printf("\n-- AVX-512 roundtrip (DIT fwd -> DIF bwd) --\n");
    for (size_t i = 0; i < 4; i++)
    {
        t++;
        p += test_roundtrip(sKs[i], "avx512",
                            radix25_tw_flat_dit_kernel_fwd_avx512, radix25_tw_flat_dif_kernel_bwd_avx512);
    }
#endif

    printf("\n=== %d/%d %s ===\n", p, t, p == t ? "ALL PASSED" : "FAILURES");
    return p != t;
}