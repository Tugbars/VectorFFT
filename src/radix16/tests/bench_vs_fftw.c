/**
 * @file bench_vs_fftw.c
 * @brief FFTW vs VectorFFT: Single radix-16 DFT butterfly, no external twiddles
 *
 * Compares K independent DFT-16 transforms:
 *   - FFTW: fftw_plan_guru_split_dft (SoA layout, FFTW_MEASURE)
 *   - VectorFFT scalar: radix16_butterfly_forward_scalar
 *   - VectorFFT AVX-512: radix16_butterfly_forward_avx512
 *
 * FFTW gets FFTW_MEASURE planning (tries multiple algorithms).
 * All use SoA split-complex layout for fairness.
 * Single-threaded (no FFTW threads).
 *
 * Compile:
 *   gcc -O2 -mavx512f -mavx512dq -o bench_vs_fftw bench_vs_fftw.c -lfftw3 -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <fftw3.h>

#include "fft_radix16_scalar_butterfly.h"
#include "fft_radix16_avx512_butterfly.h"

/* ============================================================================
 * TIMING
 * ========================================================================= */

static inline uint64_t rdtsc(void)
{
    unsigned lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

static double wall_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ============================================================================
 * ALLOCATION
 * ========================================================================= */

static double *alloc64(size_t n)
{
    /* Use fftw_malloc for SIMD alignment */
    double *p = (double *)fftw_malloc(n * sizeof(double));
    if (!p) abort();
    memset(p, 0, n * sizeof(double));
    return p;
}

static void free64(double *p) { fftw_free(p); }

static void fill_random(double *buf, size_t n, unsigned seed)
{
    unsigned s = seed;
    for (size_t i = 0; i < n; i++)
    {
        s = s * 1103515245u + 12345u;
        buf[i] = ((double)(s >> 16) / 32768.0) - 1.0;
    }
}

/* ============================================================================
 * FFTW SETUP: K independent DFT-16 using guru split interface
 *
 * SoA layout: re[r * K + k], im[r * K + k]  for r=0..15, k=0..K-1
 *
 * This means for each column k, the 16 elements are at stride K.
 * guru_split_dft:
 *   rank=1, n=16, stride=K (between consecutive DFT elements)
 *   howmany_rank=1, howmany_n=K, stride=1 (between consecutive DFTs)
 * ========================================================================= */

static fftw_plan create_fftw_plan_split(size_t K,
    double *in_re, double *in_im,
    double *out_re, double *out_im,
    int sign, unsigned flags)
{
    fftw_iodim dims[1];
    dims[0].n  = 16;
    dims[0].is = (int)K;   /* input stride: elements are K apart */
    dims[0].os = (int)K;   /* output stride: same layout */

    fftw_iodim howmany[1];
    howmany[0].n  = (int)K;
    howmany[0].is = 1;     /* consecutive columns are stride-1 */
    howmany[0].os = 1;

    return fftw_plan_guru_split_dft(
        1, dims, 1, howmany,
        in_re, in_im, out_re, out_im, flags);
}

/* ============================================================================
 * CORRECTNESS CHECK: FFTW vs VectorFFT
 * ========================================================================= */

static void verify_correctness(size_t K)
{
    const size_t sz = 16 * K;
    double *xr = alloc64(sz), *xi = alloc64(sz);
    double *yr_fftw = alloc64(sz), *yi_fftw = alloc64(sz);
    double *yr_vfft = alloc64(sz), *yi_vfft = alloc64(sz);

    fill_random(xr, sz, (unsigned)(K * 17 + 3));
    fill_random(xi, sz, (unsigned)(K * 19 + 7));

    /* FFTW forward (sign = FFTW_FORWARD = -1) */
    fftw_plan plan = create_fftw_plan_split(K, xr, xi, yr_fftw, yi_fftw,
                                            FFTW_FORWARD, FFTW_ESTIMATE);
    if (!plan) { printf("  [SKIP] FFTW plan failed for K=%zu\n", K); goto cleanup; }
    fftw_execute_split_dft(plan, xr, xi, yr_fftw, yi_fftw);
    fftw_destroy_plan(plan);

    /* VectorFFT AVX-512 forward */
    radix16_butterfly_forward_avx512(K, xr, xi, yr_vfft, yi_vfft);

    /* Compare */
    double max_err = 0.0;
    for (size_t i = 0; i < sz; i++)
    {
        max_err = fmax(max_err, fabs(yr_fftw[i] - yr_vfft[i]));
        max_err = fmax(max_err, fabs(yi_fftw[i] - yi_vfft[i]));
    }

    int pass = max_err < 1e-13;
    printf("  [%s] K=%-6zu  FFTW vs AVX512  max_err = %.3e\n",
           pass ? "PASS" : "FAIL", K, max_err);

cleanup:
    free64(xr); free64(xi);
    free64(yr_fftw); free64(yi_fftw);
    free64(yr_vfft); free64(yi_vfft);
}

/* ============================================================================
 * BENCHMARK CORE
 * ========================================================================= */

typedef struct {
    double best_cycles;
    double wall_ns_per_col;
    double ns_per_call;
} bench_result_t;

static bench_result_t bench_fftw(size_t K, int bench_iters)
{
    const size_t sz = 16 * K;
    double *xr = alloc64(sz), *xi = alloc64(sz);
    double *yr = alloc64(sz), *yi = alloc64(sz);
    fill_random(xr, sz, 42);
    fill_random(xi, sz, 43);

    /* FFTW_MEASURE: let FFTW try multiple algorithms */
    fftw_plan plan = create_fftw_plan_split(K, xr, xi, yr, yi,
                                            FFTW_FORWARD, FFTW_MEASURE);
    if (!plan)
    {
        printf("    FFTW plan failed for K=%zu\n", K);
        free64(xr); free64(xi); free64(yr); free64(yi);
        return (bench_result_t){0, 0, 0};
    }

    /* Warmup */
    for (int i = 0; i < 100; i++)
        fftw_execute_split_dft(plan, xr, xi, yr, yi);

    /* Cycle measurement: best of 5 */
    uint64_t best = UINT64_MAX;
    for (int rep = 0; rep < 5; rep++)
    {
        uint64_t t0 = rdtsc();
        for (int i = 0; i < bench_iters; i++)
            fftw_execute_split_dft(plan, xr, xi, yr, yi);
        uint64_t t1 = rdtsc();
        if (t1 - t0 < best) best = t1 - t0;
    }

    /* Wall clock */
    int wall_iters = bench_iters * 5;
    double t0 = wall_sec();
    for (int i = 0; i < wall_iters; i++)
        fftw_execute_split_dft(plan, xr, xi, yr, yi);
    double t1 = wall_sec();

    fftw_destroy_plan(plan);

    bench_result_t r;
    r.best_cycles = (double)best / (double)bench_iters;
    r.wall_ns_per_col = (t1 - t0) * 1e9 / ((double)wall_iters * (double)K);
    r.ns_per_call = (t1 - t0) * 1e9 / (double)wall_iters;
    free64(xr); free64(xi); free64(yr); free64(yi);
    return r;
}

typedef void (*butterfly_fn)(size_t, const double *, const double *,
                             double *, double *);

static bench_result_t bench_vfft(size_t K, butterfly_fn fn, int bench_iters)
{
    const size_t sz = 16 * K;
    double *xr = alloc64(sz), *xi = alloc64(sz);
    double *yr = alloc64(sz), *yi = alloc64(sz);
    fill_random(xr, sz, 42);
    fill_random(xi, sz, 43);

    for (int i = 0; i < 100; i++)
        fn(K, xr, xi, yr, yi);

    uint64_t best = UINT64_MAX;
    for (int rep = 0; rep < 5; rep++)
    {
        uint64_t t0 = rdtsc();
        for (int i = 0; i < bench_iters; i++)
            fn(K, xr, xi, yr, yi);
        uint64_t t1 = rdtsc();
        if (t1 - t0 < best) best = t1 - t0;
    }

    int wall_iters = bench_iters * 5;
    double t0 = wall_sec();
    for (int i = 0; i < wall_iters; i++)
        fn(K, xr, xi, yr, yi);
    double t1 = wall_sec();

    bench_result_t r;
    r.best_cycles = (double)best / (double)bench_iters;
    r.wall_ns_per_col = (t1 - t0) * 1e9 / ((double)wall_iters * (double)K);
    r.ns_per_call = (t1 - t0) * 1e9 / (double)wall_iters;
    free64(xr); free64(xi); free64(yr); free64(yi);
    return r;
}

/* ============================================================================
 * MAIN
 * ========================================================================= */

int main(void)
{
    printf("=== VectorFFT vs FFTW 3.3 — Single Radix-16 DFT (twiddle-less) ===\n");
    printf("=== Single core, SoA split-complex, FFTW_MEASURE planning      ===\n");
    printf("=== NOTE: container environment, no CPU pinning/freq locking    ===\n\n");

    /* ---- Correctness ---- */
    printf("--- Correctness: FFTW vs VectorFFT AVX-512 ---\n");
    static const size_t verify_K[] = {1, 3, 7, 8, 16, 24, 32, 64, 100, 256};
    for (int i = 0; i < (int)(sizeof(verify_K)/sizeof(verify_K[0])); i++)
        verify_correctness(verify_K[i]);

    /* ---- Performance ---- */
    printf("\n--- Performance: K independent DFT-16 transforms ---\n");
    printf("  %-6s  %10s  %10s  %10s  |  %10s  %10s  | %s\n",
           "K", "FFTW cyc", "scalar cyc", "AVX512 cyc",
           "FFTW ns/col", "Z512 ns/col", "speedup");
    printf("  %-6s  %10s  %10s  %10s  |  %10s  %10s  | %s\n",
           "------", "----------", "----------", "----------",
           "-----------", "-----------", "-------");

    static const size_t perf_K[] = {1, 2, 4, 8, 16, 24, 32, 48, 64, 96,
                                    128, 256, 512, 1024, 2048, 4096};
    const int n_perf = (int)(sizeof(perf_K) / sizeof(perf_K[0]));

    for (int i = 0; i < n_perf; i++)
    {
        size_t K = perf_K[i];
        int iters;
        if      (K <= 8)    iters = 500000;
        else if (K <= 64)   iters = 200000;
        else if (K <= 256)  iters = 50000;
        else if (K <= 1024) iters = 10000;
        else                iters = 2000;

        bench_result_t r_fftw   = bench_fftw(K, iters);
        bench_result_t r_scalar = bench_vfft(K, radix16_butterfly_forward_scalar, iters);
        bench_result_t r_avx512 = bench_vfft(K, radix16_butterfly_forward_avx512, iters);

        double speedup = (r_fftw.wall_ns_per_col > 0 && r_avx512.wall_ns_per_col > 0)
                         ? r_fftw.wall_ns_per_col / r_avx512.wall_ns_per_col : 0.0;

        printf("  K=%-4zu  %10.1f  %10.1f  %10.1f  |  %8.1f     %8.1f     | ",
               K, r_fftw.best_cycles, r_scalar.best_cycles, r_avx512.best_cycles,
               r_fftw.wall_ns_per_col, r_avx512.wall_ns_per_col);

        if (speedup >= 1.0)
            printf("%.2fx faster\n", speedup);
        else if (speedup > 0)
            printf("%.2fx slower\n", 1.0 / speedup);
        else
            printf("N/A\n");
    }

    /* ---- Summary at sweet spot ---- */
    printf("\n--- Amortized cost per DFT-16 at large K ---\n");
    for (int i = 0; i < n_perf; i++)
    {
        size_t K = perf_K[i];
        if (K < 64) continue;

        int iters = (K <= 256) ? 50000 : (K <= 1024) ? 10000 : 2000;
        bench_result_t r_fftw   = bench_fftw(K, iters);
        bench_result_t r_avx512 = bench_vfft(K, radix16_butterfly_forward_avx512, iters);

        double cyc_fftw = r_fftw.best_cycles / (double)K;
        double cyc_vfft = r_avx512.best_cycles / (double)K;

        printf("  K=%-5zu  FFTW: %5.1f cyc/DFT16  AVX512: %5.1f cyc/DFT16  ratio: %.2fx\n",
               K, cyc_fftw, cyc_vfft, cyc_fftw / cyc_vfft);
    }

    printf("\n=== Done ===\n");
    return 0;
}
