/**
 * @file bench_radix4.c
 * @brief Latency benchmark for radix-4 FFT stages
 *
 * Measures per-call latency (ns) and throughput (cycles/element) for:
 *   - Forward twiddle (fv)
 *   - Forward n1 (fv_n1)
 *   - Backward twiddle (bv)
 *   - Backward n1 (bv_n1)
 *   - Scalar twiddle reference (bv scalar)
 *
 * Usage: ./bench_radix4 [min_K] [max_K]
 *   defaults: min_K=4, max_K=65536
 *
 * Build:
 *   gcc -mavx2 -mfma -O2 -o bench_radix4 bench_radix4.c fft_radix4_fv.c fft_radix4_bv.c -lm
 *   gcc -mavx512f -mavx2 -mfma -O2 -o bench_radix4 bench_radix4.c fft_radix4_fv.c fft_radix4_bv.c -lm
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "vfft_compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "fft_radix4.h"
#include "scalar/fft_radix4_scalar.h"

/* ── Declarations ── */

extern void fft_radix4_fv(
    double *restrict out_re, double *restrict out_im,
    const double *restrict in_re, const double *restrict in_im,
    const fft_twiddles_soa *restrict tw, int K);

extern void fft_radix4_fv_n1(
    double *restrict out_re, double *restrict out_im,
    const double *restrict in_re, const double *restrict in_im, int K);

extern void fft_radix4_bv(
    double *restrict out_re, double *restrict out_im,
    const double *restrict in_re, const double *restrict in_im,
    const fft_twiddles_soa *restrict tw, int K);

extern void fft_radix4_bv_n1(
    double *restrict out_re, double *restrict out_im,
    const double *restrict in_re, const double *restrict in_im, int K);

/* ── rdtsc ── */

static inline uint64_t rdtsc(void)
{
#if defined(_MSC_VER) || defined(__INTEL_LLVM_COMPILER)
    return __rdtsc();
#elif defined(__x86_64__) || defined(_M_X64)
    unsigned lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
#elif defined(__aarch64__)
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
#else
    return (uint64_t)vfft_now_ns();
#endif
}

/* ── Helpers ── */

static double *alloc_a(size_t n)
{
    double *p = (double *)vfft_aligned_alloc(64, n * sizeof(double));
    if (!p) {
        fprintf(stderr, "alloc failed\n");
        exit(1);
    }
    return p;
}

static void fill_random(double *p, size_t n, unsigned seed)
{
    /* Simple LCG — just need non-zero, non-degenerate data */
    uint64_t s = seed;
    for (size_t i = 0; i < n; i++) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        p[i] = (double)(int64_t)(s >> 33) * 1e-9;
    }
}

static int cmp_u64(const void *a, const void *b)
{
    uint64_t va = *(const uint64_t *)a, vb = *(const uint64_t *)b;
    return (va > vb) - (va < vb);
}

/* ── Estimate TSC frequency ── */

static double estimate_tsc_ghz(void)
{
    double t0 = vfft_now_ns();
    uint64_t c0 = rdtsc();

    /* Spin ~50ms */
    volatile double dummy = 0;
    for (int i = 0; i < 5000000; i++)
        dummy += 1.0;
    (void)dummy;

    uint64_t c1 = rdtsc();
    double t1 = vfft_now_ns();

    double dt_ns = t1 - t0;
    double dc = (double)(c1 - c0);
    return dc / dt_ns; /* cycles per ns = GHz */
}

/* ── Benchmark kernel ── */

typedef void (*bench_fn_tw)(double *, double *, const double *, const double *,
                            const fft_twiddles_soa *, int);
typedef void (*bench_fn_n1)(double *, double *, const double *, const double *, int);

typedef struct {
    double median_ns;
    double median_cycles;
    double min_ns;
    uint64_t min_ticks;
} bench_result;

#define WARMUP_ITERS 8
#define MAX_SAMPLES  64

static bench_result bench_tw(bench_fn_tw fn, int K,
                             double *in_re, double *in_im,
                             double *out_re, double *out_im,
                             fft_twiddles_soa *tw,
                             double tsc_ghz)
{
    int N = 4 * K;
    /* Target ~2ms per measurement, min 16 iters */
    int inner = (int)(2e6 / (10.0 * N + 100));
    if (inner < 16) inner = 16;
    if (inner > 100000) inner = 100000;

    uint64_t samples[MAX_SAMPLES];

    /* Warmup */
    for (int w = 0; w < WARMUP_ITERS; w++)
        fn(out_re, out_im, in_re, in_im, tw, K);

    for (int s = 0; s < MAX_SAMPLES; s++) {
        uint64_t t0 = rdtsc();
        for (int i = 0; i < inner; i++)
            fn(out_re, out_im, in_re, in_im, tw, K);
        uint64_t t1 = rdtsc();
        samples[s] = (t1 - t0) / (unsigned)inner;
    }

    qsort(samples, MAX_SAMPLES, sizeof(uint64_t), cmp_u64);

    bench_result r;
    r.min_ticks = samples[0];
    r.median_cycles = (double)samples[MAX_SAMPLES / 2];
    r.median_ns = r.median_cycles / tsc_ghz;
    r.min_ns = (double)r.min_ticks / tsc_ghz;
    return r;
}

static bench_result bench_n1(bench_fn_n1 fn, int K,
                             double *in_re, double *in_im,
                             double *out_re, double *out_im,
                             double tsc_ghz)
{
    int N = 4 * K;
    int inner = (int)(2e6 / (10.0 * N + 100));
    if (inner < 16) inner = 16;
    if (inner > 100000) inner = 100000;

    uint64_t samples[MAX_SAMPLES];

    for (int w = 0; w < WARMUP_ITERS; w++)
        fn(out_re, out_im, in_re, in_im, K);

    for (int s = 0; s < MAX_SAMPLES; s++) {
        uint64_t t0 = rdtsc();
        for (int i = 0; i < inner; i++)
            fn(out_re, out_im, in_re, in_im, K);
        uint64_t t1 = rdtsc();
        samples[s] = (t1 - t0) / (unsigned)inner;
    }

    qsort(samples, MAX_SAMPLES, sizeof(uint64_t), cmp_u64);

    bench_result r;
    r.min_ticks = samples[0];
    r.median_cycles = (double)samples[MAX_SAMPLES / 2];
    r.median_ns = r.median_cycles / tsc_ghz;
    r.min_ns = (double)r.min_ticks / tsc_ghz;
    return r;
}

/* Scalar backward twiddle for comparison */
static void scalar_bv_wrapper(double *out_re, double *out_im,
                               const double *in_re, const double *in_im,
                               const fft_twiddles_soa *tw, int K)
{
    radix4_stage_baseptr_bv_scalar((size_t)(4 * K), (size_t)K,
                                   in_re, in_im, out_re, out_im, tw);
}

/* ── Main ── */

int main(int argc, char **argv)
{
    int min_K = 4, max_K = 65536;
    if (argc > 1) min_K = atoi(argv[1]);
    if (argc > 2) max_K = atoi(argv[2]);

    printf("Estimating TSC frequency...\n");
    double tsc_ghz = estimate_tsc_ghz();
    printf("TSC: %.3f GHz\n\n", tsc_ghz);

#ifdef __AVX512F__
    printf("ISA: AVX-512F + AVX2 + FMA\n");
#elif defined(__AVX2__)
    printf("ISA: AVX2 + FMA\n");
#else
    printf("ISA: Scalar\n");
#endif

    printf("\n");
    printf("%-6s %-8s │ %-12s %-12s %-12s %-12s │ %-12s │ %-10s\n",
           "K", "N",
           "fv_tw(ns)", "fv_n1(ns)", "bv_tw(ns)", "bv_n1(ns)",
           "scalar(ns)", "n1 speedup");
    printf("──────────────────────────────────────────────────────────────────────────────────────────\n");

    for (int K = min_K; K <= max_K; K *= 4)
    {
        int N = 4 * K;
        double *in_re  = alloc_a((size_t)N);
        double *in_im  = alloc_a((size_t)N);
        double *out_re = alloc_a((size_t)N);
        double *out_im = alloc_a((size_t)N);
        double *tw_re  = alloc_a((size_t)(3 * K));
        double *tw_im  = alloc_a((size_t)(3 * K));

        fill_random(in_re, (size_t)N, 42);
        fill_random(in_im, (size_t)N, 137);

        /* Generate twiddles */
        for (int k = 0; k < K; k++) {
            double a1 = -2.0 * M_PI * k / N;
            double a2 = 2.0 * a1, a3 = 3.0 * a1;
            tw_re[k]       = cos(a1); tw_im[k]       = sin(a1);
            tw_re[K + k]   = cos(a2); tw_im[K + k]   = sin(a2);
            tw_re[2*K + k] = cos(a3); tw_im[2*K + k] = sin(a3);
        }

        fft_twiddles_soa tw = { .re = tw_re, .im = tw_im };

        bench_result r_fv_tw = bench_tw(fft_radix4_fv, K, in_re, in_im, out_re, out_im, &tw, tsc_ghz);
        bench_result r_fv_n1 = bench_n1(fft_radix4_fv_n1, K, in_re, in_im, out_re, out_im, tsc_ghz);
        bench_result r_bv_tw = bench_tw(fft_radix4_bv, K, in_re, in_im, out_re, out_im, &tw, tsc_ghz);
        bench_result r_bv_n1 = bench_n1(fft_radix4_bv_n1, K, in_re, in_im, out_re, out_im, tsc_ghz);
        bench_result r_sc_bv = bench_tw(scalar_bv_wrapper, K, in_re, in_im, out_re, out_im, &tw, tsc_ghz);

        double n1_speedup = r_bv_tw.median_ns / r_bv_n1.median_ns;

        printf("%-6d %-8d │ %10.1f   %10.1f   %10.1f   %10.1f   │ %10.1f   │ %8.2fx\n",
               K, N,
               r_fv_tw.median_ns, r_fv_n1.median_ns,
               r_bv_tw.median_ns, r_bv_n1.median_ns,
               r_sc_bv.median_ns,
               n1_speedup);

        vfft_aligned_free(in_re);  vfft_aligned_free(in_im);
        vfft_aligned_free(out_re); vfft_aligned_free(out_im);
        vfft_aligned_free(tw_re);  vfft_aligned_free(tw_im);
    }

    printf("\n");
    printf("Legend:\n");
    printf("  fv_tw  = forward twiddle    fv_n1  = forward no-twiddle\n");
    printf("  bv_tw  = backward twiddle   bv_n1  = backward no-twiddle\n");
    printf("  scalar = scalar backward twiddle (baseline)\n");
    printf("  n1 speedup = bv_tw / bv_n1 (expected 1.4-2.5x)\n");
    printf("\nAll times are median of %d samples (ns per call).\n", MAX_SAMPLES);

    /* ── Detailed per-element view for select sizes ── */
    printf("\n\n=== Cycles/element (median) ===\n\n");
    printf("%-6s %-8s │ %-14s %-14s %-14s %-14s │ %-14s\n",
           "K", "N", "fv_tw(c/el)", "fv_n1(c/el)", "bv_tw(c/el)", "bv_n1(c/el)", "scalar(c/el)");
    printf("──────────────────────────────────────────────────────────────────────────────────────────\n");

    for (int K = min_K; K <= max_K; K *= 4)
    {
        int N = 4 * K;
        double *in_re  = alloc_a((size_t)N);
        double *in_im  = alloc_a((size_t)N);
        double *out_re = alloc_a((size_t)N);
        double *out_im = alloc_a((size_t)N);
        double *tw_re  = alloc_a((size_t)(3 * K));
        double *tw_im  = alloc_a((size_t)(3 * K));

        fill_random(in_re, (size_t)N, 42);
        fill_random(in_im, (size_t)N, 137);

        for (int k = 0; k < K; k++) {
            double a1 = -2.0 * M_PI * k / N;
            tw_re[k]       = cos(a1);     tw_im[k]       = sin(a1);
            tw_re[K + k]   = cos(2*a1);   tw_im[K + k]   = sin(2*a1);
            tw_re[2*K + k] = cos(3*a1);   tw_im[2*K + k] = sin(3*a1);
        }

        fft_twiddles_soa tw = { .re = tw_re, .im = tw_im };

        bench_result r_fv_tw = bench_tw(fft_radix4_fv, K, in_re, in_im, out_re, out_im, &tw, tsc_ghz);
        bench_result r_fv_n1 = bench_n1(fft_radix4_fv_n1, K, in_re, in_im, out_re, out_im, tsc_ghz);
        bench_result r_bv_tw = bench_tw(fft_radix4_bv, K, in_re, in_im, out_re, out_im, &tw, tsc_ghz);
        bench_result r_bv_n1 = bench_n1(fft_radix4_bv_n1, K, in_re, in_im, out_re, out_im, tsc_ghz);
        bench_result r_sc_bv = bench_tw(scalar_bv_wrapper, K, in_re, in_im, out_re, out_im, &tw, tsc_ghz);

        double elems = (double)N; /* N complex elements = 4K */
        printf("%-6d %-8d │ %12.2f   %12.2f   %12.2f   %12.2f   │ %12.2f\n",
               K, N,
               r_fv_tw.median_cycles / elems, r_fv_n1.median_cycles / elems,
               r_bv_tw.median_cycles / elems, r_bv_n1.median_cycles / elems,
               r_sc_bv.median_cycles / elems);

        vfft_aligned_free(in_re);  vfft_aligned_free(in_im);
        vfft_aligned_free(out_re); vfft_aligned_free(out_im);
        vfft_aligned_free(tw_re);  vfft_aligned_free(tw_im);
    }

    return 0;
}