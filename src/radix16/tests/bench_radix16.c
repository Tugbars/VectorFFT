/**
 * @file bench_radix16.c
 * @brief Benchmark & roundtrip test for VectorFFT radix-16 components
 *
 * Measures:
 *   1. External twiddle latency: scalar vs AVX2 vs AVX-512
 *   2. Butterfly latency: scalar vs AVX-512
 *   3. Full stage (butterfly + twiddle): scalar vs AVX2+scalar vs AVX-512
 *   4. Roundtrip correctness: fwd_bfly -> twiddle -> conj_twiddle -> bwd_bfly = 16*x
 *
 * Timing: rdtsc for cycle-accurate measurement, wall clock for throughput.
 * Each test runs many iterations with cache-warm data.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#include "fft_radix16_scalar_butterfly.h"
#include "fft_radix16_avx512_butterfly.h"
#include "fft_twiddle_scalar.h"
#include "fft_twiddle_avx2.h"
#include "fft_twiddle_avx512.h"

/* ============================================================================
 * TIMING
 * ========================================================================= */

static inline uint64_t rdtsc(void)
{
#if defined(__x86_64__) || defined(_M_X64)
    unsigned lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
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
    double *p = NULL;
    size_t bytes = (n > 0 ? n : 1) * sizeof(double);
    if (posix_memalign((void **)&p, 64, bytes) != 0) abort();
    memset(p, 0, bytes);
    return p;
}

static void fill_random(double *buf, size_t n, unsigned seed)
{
    unsigned s = seed;
    for (size_t i = 0; i < n; i++)
    {
        s = s * 1103515245u + 12345u;
        buf[i] = ((double)(s >> 16) / 32768.0) - 1.0;
    }
}

/* Build DFT twiddle table W_N^{r*k} for radix R, N = R*K */
static void build_twiddle_table(size_t R, size_t K,
    double *tw_re, double *tw_im, int forward)
{
    const size_t N = R * K;
    const double sign = forward ? -1.0 : 1.0;
    for (size_t r = 0; r < R; r++)
    {
        for (size_t k = 0; k < K; k++)
        {
            double angle = sign * 2.0 * M_PI * (double)(r * k) / (double)N;
            tw_re[r * K + k] = cos(angle);
            tw_im[r * K + k] = sin(angle);
        }
    }
}

/* ============================================================================
 * BENCHMARK: EXTERNAL TWIDDLE LATENCY
 * ========================================================================= */

typedef void (*twiddle_fn)(size_t, double *, double *, const double *, const double *);

static void bench_twiddle_latency(const char *label, twiddle_fn fn,
    size_t count, int warmup_iters, int bench_iters)
{
    double *dr = alloc64(count), *di = alloc64(count);
    double *wr = alloc64(count), *wi = alloc64(count);
    double *orig_r = alloc64(count), *orig_i = alloc64(count);

    fill_random(orig_r, count, 42);
    fill_random(orig_i, count, 43);
    fill_random(wr, count, 44);
    fill_random(wi, count, 45);

    /* Warmup */
    for (int i = 0; i < warmup_iters; i++)
    {
        memcpy(dr, orig_r, count * sizeof(double));
        memcpy(di, orig_i, count * sizeof(double));
        fn(count, dr, di, wr, wi);
    }

    /* Benchmark: best of 5 runs */
    uint64_t best_cycles = UINT64_MAX;
    for (int rep = 0; rep < 5; rep++)
    {
        memcpy(dr, orig_r, count * sizeof(double));
        memcpy(di, orig_i, count * sizeof(double));

        uint64_t t0 = rdtsc();
        for (int i = 0; i < bench_iters; i++)
            fn(count, dr, di, wr, wi);
        uint64_t t1 = rdtsc();

        uint64_t total = t1 - t0;
        if (total < best_cycles) best_cycles = total;
    }

    /* Wall clock for throughput */
    int wall_iters = bench_iters * 5;
    double t_start = wall_sec();
    for (int i = 0; i < wall_iters; i++)
    {
        memcpy(dr, orig_r, count * sizeof(double));
        memcpy(di, orig_i, count * sizeof(double));
        fn(count, dr, di, wr, wi);
    }
    double t_end = wall_sec();

    double cycles_per_elem = (double)best_cycles / ((double)bench_iters * (double)count);
    double ns_per_elem = (t_end - t_start) * 1e9 / ((double)wall_iters * (double)count);
    double gelem_per_s = ((double)wall_iters * (double)count) / (t_end - t_start) / 1e9;

    printf("  %-14s  count=%-6zu  %6.2f cyc/elem  %6.2f ns/elem  %5.2f Gelem/s\n",
           label, count, cycles_per_elem, ns_per_elem, gelem_per_s);

    free(dr); free(di); free(wr); free(wi); free(orig_r); free(orig_i);
}

/* ============================================================================
 * BENCHMARK: BUTTERFLY LATENCY
 * ========================================================================= */

typedef void (*butterfly_fn)(size_t, const double *, const double *,
                             double *, double *);

static void bench_butterfly_latency(const char *label, butterfly_fn fn,
    size_t K, int warmup_iters, int bench_iters)
{
    const size_t R = 16;
    const size_t sz = R * K;
    double *xr = alloc64(sz), *xi = alloc64(sz);
    double *yr = alloc64(sz), *yi = alloc64(sz);

    fill_random(xr, sz, 100);
    fill_random(xi, sz, 101);

    for (int i = 0; i < warmup_iters; i++)
        fn(K, xr, xi, yr, yi);

    uint64_t best_cycles = UINT64_MAX;
    for (int rep = 0; rep < 5; rep++)
    {
        uint64_t t0 = rdtsc();
        for (int i = 0; i < bench_iters; i++)
            fn(K, xr, xi, yr, yi);
        uint64_t t1 = rdtsc();
        if (t1 - t0 < best_cycles) best_cycles = t1 - t0;
    }

    int wall_iters = bench_iters * 5;
    double t_start = wall_sec();
    for (int i = 0; i < wall_iters; i++)
        fn(K, xr, xi, yr, yi);
    double t_end = wall_sec();

    double cycles_per_col = (double)best_cycles / ((double)bench_iters * (double)K);
    double ns_per_col = (t_end - t_start) * 1e9 / ((double)wall_iters * (double)K);

    printf("  %-14s  K=%-6zu  %6.1f cyc/DFT16  %5.1f ns/DFT16\n",
           label, K, cycles_per_col, ns_per_col);

    free(xr); free(xi); free(yr); free(yi);
}

/* ============================================================================
 * BENCHMARK: FULL STAGE (butterfly + twiddle)
 * ========================================================================= */

static void bench_full_stage(const char *label,
    butterfly_fn bfly, twiddle_fn twiddle,
    size_t K, int warmup_iters, int bench_iters)
{
    const size_t R = 16;
    const size_t sz = R * K;
    double *xr = alloc64(sz), *xi = alloc64(sz);
    double *yr = alloc64(sz), *yi = alloc64(sz);
    double *wr = alloc64(sz), *wi = alloc64(sz);

    fill_random(xr, sz, 200);
    fill_random(xi, sz, 201);
    build_twiddle_table(R, K, wr, wi, 1);

    for (int i = 0; i < warmup_iters; i++)
    {
        bfly(K, xr, xi, yr, yi);
        twiddle((R - 1) * K, yr + K, yi + K, wr + K, wi + K);
    }

    uint64_t best_cycles = UINT64_MAX;
    for (int rep = 0; rep < 5; rep++)
    {
        uint64_t t0 = rdtsc();
        for (int i = 0; i < bench_iters; i++)
        {
            bfly(K, xr, xi, yr, yi);
            twiddle((R - 1) * K, yr + K, yi + K, wr + K, wi + K);
        }
        uint64_t t1 = rdtsc();
        if (t1 - t0 < best_cycles) best_cycles = t1 - t0;
    }

    int wall_iters = bench_iters * 5;
    double t_start = wall_sec();
    for (int i = 0; i < wall_iters; i++)
    {
        bfly(K, xr, xi, yr, yi);
        twiddle((R - 1) * K, yr + K, yi + K, wr + K, wi + K);
    }
    double t_end = wall_sec();

    double cycles_per_col = (double)best_cycles / ((double)bench_iters * (double)K);
    double ns_per_col = (t_end - t_start) * 1e9 / ((double)wall_iters * (double)K);

    printf("  %-14s  K=%-6zu  %6.1f cyc/DFT16  %5.1f ns/DFT16\n",
           label, K, cycles_per_col, ns_per_col);

    free(xr); free(xi); free(yr); free(yi); free(wr); free(wi);
}

/* ============================================================================
 * ROUNDTRIP: fwd_bfly -> twiddle -> conj_twiddle -> bwd_bfly = 16*x
 * ========================================================================= */

static int roundtrip_test(const char *label,
    butterfly_fn bfly_fwd, butterfly_fn bfly_bwd,
    twiddle_fn twiddle_apply, size_t K)
{
    const size_t R = 16;
    const size_t sz = R * K;
    double *xr = alloc64(sz), *xi = alloc64(sz);
    double *yr = alloc64(sz), *yi = alloc64(sz);
    double *zr = alloc64(sz), *zi = alloc64(sz);
    double *tw_fwd_re = alloc64(sz), *tw_fwd_im = alloc64(sz);
    double *tw_bwd_re = alloc64(sz), *tw_bwd_im = alloc64(sz);

    fill_random(xr, sz, (unsigned)(K * 311 + 7));
    fill_random(xi, sz, (unsigned)(K * 313 + 11));

    build_twiddle_table(R, K, tw_fwd_re, tw_fwd_im, 1);
    build_twiddle_table(R, K, tw_bwd_re, tw_bwd_im, 0);

    /* Forward: butterfly -> twiddle */
    bfly_fwd(K, xr, xi, yr, yi);
    twiddle_apply((R - 1) * K, yr + K, yi + K, tw_fwd_re + K, tw_fwd_im + K);

    /* Backward: conj_twiddle -> butterfly */
    twiddle_apply((R - 1) * K, yr + K, yi + K, tw_bwd_re + K, tw_bwd_im + K);
    bfly_bwd(K, yr, yi, zr, zi);

    /* Check: z = 16*x */
    double max_err = 0.0;
    for (size_t i = 0; i < sz; i++)
    {
        max_err = fmax(max_err, fabs(zr[i] - 16.0 * xr[i]));
        max_err = fmax(max_err, fabs(zi[i] - 16.0 * xi[i]));
    }

    int pass = max_err < 1e-12;
    printf("  [%s] %-20s K=%-6zu  max_err = %.3e\n",
           pass ? "PASS" : "FAIL", label, K, max_err);

    free(xr); free(xi); free(yr); free(yi); free(zr); free(zi);
    free(tw_fwd_re); free(tw_fwd_im); free(tw_bwd_re); free(tw_bwd_im);
    return pass;
}

/* ============================================================================
 * MAIN
 * ========================================================================= */

int main(void)
{
    int rt_pass = 0, rt_total = 0;

    printf("=== VectorFFT Radix-16 Benchmark ===\n");
    printf("=== NOTE: cycle counts are approximate (container, no pinning) ===\n\n");

    /* ---- Twiddle latency ---- */
    printf("--- External twiddle latency ---\n");
    static const size_t tw_sizes[] = {16, 64, 256, 1024, 4096, 16384};
    const int n_tw = (int)(sizeof(tw_sizes) / sizeof(tw_sizes[0]));

    for (int i = 0; i < n_tw; i++)
    {
        size_t count = tw_sizes[i];
        int iters = (count <= 256) ? 100000 : (count <= 4096) ? 20000 : 5000;
        bench_twiddle_latency("scalar", fft_twiddle_apply_scalar, count, 100, iters);
        bench_twiddle_latency("AVX2", fft_twiddle_apply_avx2, count, 100, iters);
        bench_twiddle_latency("AVX-512", fft_twiddle_apply_avx512, count, 100, iters);
        printf("\n");
    }

    /* ---- Butterfly latency ---- */
    printf("--- Butterfly latency (forward) ---\n");
    static const size_t bfly_K[] = {1, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    const int n_bfly = (int)(sizeof(bfly_K) / sizeof(bfly_K[0]));

    for (int i = 0; i < n_bfly; i++)
    {
        size_t K = bfly_K[i];
        int iters = (K <= 32) ? 200000 : (K <= 256) ? 50000 : 10000;
        bench_butterfly_latency("scalar", radix16_butterfly_forward_scalar, K, 100, iters);
        bench_butterfly_latency("AVX-512", radix16_butterfly_forward_avx512, K, 100, iters);
        printf("\n");
    }

    /* ---- Full stage ---- */
    printf("--- Full stage: butterfly + twiddle (forward) ---\n");
    for (int i = 0; i < n_bfly; i++)
    {
        size_t K = bfly_K[i];
        int iters = (K <= 32) ? 200000 : (K <= 256) ? 50000 : 10000;
        bench_full_stage("scalar",
            radix16_butterfly_forward_scalar, fft_twiddle_apply_scalar,
            K, 100, iters);
        bench_full_stage("AVX2+scalar",
            radix16_butterfly_forward_scalar, fft_twiddle_apply_avx2,
            K, 100, iters);
        bench_full_stage("AVX-512",
            radix16_butterfly_forward_avx512, fft_twiddle_apply_avx512,
            K, 100, iters);
        printf("\n");
    }

    /* ---- Roundtrip correctness ---- */
    printf("--- Roundtrip: fwd_bfly -> twiddle -> conj_twiddle -> bwd_bfly = 16*x ---\n\n");

    static const size_t rt_K[] = {1, 3, 7, 8, 9, 15, 16, 23, 24, 25, 32, 48, 64, 100, 128, 256};
    const int n_rt = (int)(sizeof(rt_K) / sizeof(rt_K[0]));

    printf("  Scalar pipeline:\n");
    for (int i = 0; i < n_rt; i++)
    {
        rt_total++;
        rt_pass += roundtrip_test("scalar",
            radix16_butterfly_forward_scalar, radix16_butterfly_backward_scalar,
            fft_twiddle_apply_scalar, rt_K[i]);
    }

    printf("\n  AVX-512 pipeline:\n");
    for (int i = 0; i < n_rt; i++)
    {
        rt_total++;
        rt_pass += roundtrip_test("AVX-512",
            radix16_butterfly_forward_avx512, radix16_butterfly_backward_avx512,
            fft_twiddle_apply_avx512, rt_K[i]);
    }

    printf("\n  Cross-ISA (AVX-512 bfly + AVX2 twiddle):\n");
    for (int i = 0; i < n_rt; i++)
    {
        rt_total++;
        rt_pass += roundtrip_test("AVX512+AVX2",
            radix16_butterfly_forward_avx512, radix16_butterfly_backward_avx512,
            fft_twiddle_apply_avx2, rt_K[i]);
    }

    printf("\n=== Roundtrip: %d/%d passed ===\n", rt_pass, rt_total);
    printf("=== Benchmark complete ===\n");
    return rt_pass < rt_total ? 1 : 0;
}
