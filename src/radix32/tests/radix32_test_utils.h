/**
 * @file radix32_test_utils.h
 * @brief Shared utilities for radix-32 correctness tests and benchmarks
 *
 * Provides: aligned allocation, scalar reference DFT-32 (twiddled + notw),
 * twiddle table generation (flat + ladder), and max-error comparison.
 */

#ifndef RADIX32_TEST_UTILS_H
#define RADIX32_TEST_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════════════════════════════════════════════
 * Aligned allocation
 * ═══════════════════════════════════════════════════════════════ */

static inline double *r32t_alloc(size_t n)
{
    double *p = NULL;
#ifdef _WIN32
    p = (double *)_aligned_malloc(n * sizeof(double), 64);
    if (!p) { fprintf(stderr, "alloc failed\n"); exit(1); }
#else
    if (posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) {
        fprintf(stderr, "alloc failed\n"); exit(1);
    }
#endif
    memset(p, 0, n * sizeof(double));
    return p;
}

static inline void r32t_free(double *p)
{
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

/* ═══════════════════════════════════════════════════════════════
 * Scalar reference: twiddle-less DFT-32
 *
 * K pure DFT-32 butterflies, stride-K layout.
 * sign = -1.0 for forward, +1.0 for backward.
 * ═══════════════════════════════════════════════════════════════ */

static void r32t_ref_notw(const double *in_re, const double *in_im,
                          double *out_re, double *out_im,
                          size_t K, int fwd)
{
    const double sign = fwd ? -1.0 : 1.0;
    for (size_t k = 0; k < K; k++) {
        for (int m = 0; m < 32; m++) {
            double sr = 0, si = 0;
            for (int n = 0; n < 32; n++) {
                double a = sign * 2.0 * M_PI * (double)(m * n) / 32.0;
                double wr = cos(a), wi = sin(a);
                sr += in_re[n*K+k] * wr - in_im[n*K+k] * wi;
                si += in_re[n*K+k] * wi + in_im[n*K+k] * wr;
            }
            out_re[m*K+k] = sr;
            out_im[m*K+k] = si;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Scalar reference: twiddled DFT-32
 *
 * Apply W_{32K}^{n·k} twiddles to inputs, then DFT-32.
 * ═══════════════════════════════════════════════════════════════ */

static void r32t_ref_tw(const double *in_re, const double *in_im,
                        double *out_re, double *out_im,
                        size_t K, int fwd)
{
    const size_t NN = 32 * K;
    const double sign = fwd ? -1.0 : 1.0;
    for (size_t k = 0; k < K; k++) {
        double xr[32], xi[32];
        for (int n = 0; n < 32; n++) {
            double dr = in_re[n*K+k], di = in_im[n*K+k];
            if (n > 0) {
                double a = sign * 2.0 * M_PI * (double)(n * k) / (double)NN;
                double wr = cos(a), wi = sin(a);
                double tr = dr * wr - di * wi;
                di = dr * wi + di * wr;
                dr = tr;
            }
            xr[n] = dr; xi[n] = di;
        }
        for (int m = 0; m < 32; m++) {
            double sr = 0, si = 0;
            for (int n = 0; n < 32; n++) {
                double a = sign * 2.0 * M_PI * (double)(m * n) / 32.0;
                double wr = cos(a), wi = sin(a);
                sr += xr[n] * wr - xi[n] * wi;
                si += xr[n] * wi + xi[n] * wr;
            }
            out_re[m*K+k] = sr;
            out_im[m*K+k] = si;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Twiddle table generation
 * ═══════════════════════════════════════════════════════════════ */

/** Flat twiddle table: tw_re[(n-1)*K + k] = cos(-2π·n·k / (32·K)) */
static void r32t_gen_flat_tw(double *tw_re, double *tw_im, size_t K)
{
    const size_t NN = 32 * K;
    for (int n = 1; n < 32; n++)
        for (size_t k = 0; k < K; k++) {
            double a = -2.0 * M_PI * (double)(n * k) / (double)NN;
            tw_re[(n-1)*K + k] = cos(a);
            tw_im[(n-1)*K + k] = sin(a);
        }
}

/** Ladder base twiddles: base[i*K+k] = W^{(2^i)·k}, i=0..4 → powers 1,2,4,8,16 */
static void r32t_gen_ladder_tw(double *base_re, double *base_im, size_t K)
{
    const size_t NN = 32 * K;
    const int pows[5] = {1, 2, 4, 8, 16};
    for (int i = 0; i < 5; i++)
        for (size_t k = 0; k < K; k++) {
            double a = -2.0 * M_PI * (double)(pows[i] * k) / (double)NN;
            base_re[i*K + k] = cos(a);
            base_im[i*K + k] = sin(a);
        }
}

/* ═══════════════════════════════════════════════════════════════
 * Error measurement
 * ═══════════════════════════════════════════════════════════════ */

static double r32t_maxerr(const double *a_re, const double *a_im,
                          const double *b_re, const double *b_im, size_t n)
{
    double mx = 0;
    for (size_t i = 0; i < n; i++) {
        double dr = fabs(a_re[i] - b_re[i]);
        double di = fabs(a_im[i] - b_im[i]);
        if (dr > mx) mx = dr;
        if (di > mx) mx = di;
    }
    return mx;
}

/* ═══════════════════════════════════════════════════════════════
 * Random input generation
 * ═══════════════════════════════════════════════════════════════ */

static void r32t_rand_fill(double *re, double *im, size_t n, unsigned seed)
{
    srand(seed);
    for (size_t i = 0; i < n; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Timing (ns resolution)
 * ═══════════════════════════════════════════════════════════════ */

#ifdef _WIN32
#include <windows.h>
static inline double r32t_now_ns(void)
{
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart * 1e9;
}
#else
#include <time.h>
static inline double r32t_now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
#endif

/** 5·N·log2(N) real flops per complex DFT-N */
static inline double r32t_dft32_flops(void) { return 5.0 * 32 * 5; }

#endif /* RADIX32_TEST_UTILS_H */
