/**
 * @file bench_radix32_avx2.c
 * @brief Latency benchmark for AVX2 radix-32 FFT implementations
 *
 * Measures:
 *   - n1 codelet forward/backward latency (cycles + ns)
 *   - twiddle version forward/backward latency for K=8,64,256
 *   - serialised latency via output→input chain (not throughput)
 *
 * Methodology:
 *   - rdtsc with lfence serialisation
 *   - warmup iterations discarded
 *   - reports median of N trials (robust to outliers)
 *
 * Build (ICX):
 *   icx -O2 -mavx2 -mfma -lm -o bench_radix32_avx2 bench_radix32_avx2.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#define R32_NEED_TIMER
#include "../fft_radix32_platform.h"

#include "../avx2/fft_radix32_avx2.h"
#include "../avx2/fft_radix32_avx2_n1.h"

/*==========================================================================
 * TIMING
 *=========================================================================*/

static inline uint64_t rdtsc_fenced(void)
{
#if R32_MSVC
    _mm_lfence();
    return __rdtsc();
#else
    unsigned int lo, hi;
    __asm__ __volatile__(
        "lfence\n\t"
        "rdtsc"
        : "=a"(lo), "=d"(hi)
        :
        : "memory");
    return ((uint64_t)hi << 32) | lo;
#endif
}

static inline double now_ns(void)
{
    return r32_timer_sec() * 1e9;
}

/*==========================================================================
 * STATISTICS
 *=========================================================================*/

static int cmp_u64(const void *a, const void *b)
{
    uint64_t va = *(const uint64_t *)a;
    uint64_t vb = *(const uint64_t *)b;
    return (va > vb) - (va < vb);
}

static uint64_t median_u64(uint64_t *arr, size_t n)
{
    qsort(arr, n, sizeof(uint64_t), cmp_u64);
    return arr[n / 2];
}

static int cmp_dbl(const void *a, const void *b)
{
    double va = *(const double *)a;
    double vb = *(const double *)b;
    return (va > vb) - (va < vb);
}

static double median_dbl(double *arr, size_t n)
{
    qsort(arr, n, sizeof(double), cmp_dbl);
    return arr[n / 2];
}

/*==========================================================================
 * ALIGNED ALLOCATION
 *=========================================================================*/

static double *alloc_aligned(size_t count)
{
    double *p = (double *)r32_aligned_alloc(64, count * sizeof(double));
    if (!p) {
        fprintf(stderr, "FATAL: r32_aligned_alloc failed\n");
        exit(1);
    }
    memset(p, 0, count * sizeof(double));
    return p;
}

/*==========================================================================
 * SEEDED PRNG (deterministic)
 *=========================================================================*/

static uint64_t rng_s[4] = {
    0x180EC6D33CFD0ABAULL, 0xD5A61266F0C9392CULL,
    0xA9582618E03FC9AAULL, 0x39ABDC4529B1661CULL
};

static uint64_t rng_next(void) {
    uint64_t result = ((rng_s[1] * 5) << 7 | (rng_s[1] * 5) >> 57) * 9;
    uint64_t t = rng_s[1] << 17;
    rng_s[2] ^= rng_s[0]; rng_s[3] ^= rng_s[1];
    rng_s[1] ^= rng_s[2]; rng_s[0] ^= rng_s[3];
    rng_s[2] ^= t; rng_s[3] = (rng_s[3] << 45) | (rng_s[3] >> 19);
    return result;
}

static double rng_uniform(void) {
    return (double)(int64_t)rng_next() * 5.42101086242752217e-20;
}

static void fill_random(double *buf, size_t count) {
    for (size_t i = 0; i < count; i++) buf[i] = rng_uniform();
}

/*==========================================================================
 * TWIDDLE GENERATION (for bench only — copy from test file)
 *=========================================================================*/

static void gen_pass1_twiddles(size_t K, double *tw_re, double *tw_im)
{
    for (size_t k = 0; k < K; k++) {
        double ang = -2.0 * M_PI * (double)k / (32.0 * (double)K);
        tw_re[0 * K + k] = cos(ang);
        tw_im[0 * K + k] = sin(ang);
        tw_re[1 * K + k] = cos(2.0 * ang);
        tw_im[1 * K + k] = sin(2.0 * ang);
    }
}

static void gen_pass2_twiddles(size_t K, double **tw_re, double **tw_im)
{
    for (int g = 0; g < 8; g++) {
        for (size_t k = 0; k < K; k++) {
            double angle = -2.0 * M_PI * (double)(g + 1) * (double)k
                         / (8.0 * (double)K);
            tw_re[g][k] = cos(angle);
            tw_im[g][k] = sin(angle);
        }
    }
}

/*==========================================================================
 * N1 CODELET BENCH
 *=========================================================================*/

#define WARMUP   200
#define TRIALS  1000

static void bench_n1(void)
{
    const size_t stride = 4;
    const size_t total  = 32 * stride;

    double *in_re  = alloc_aligned(total);
    double *in_im  = alloc_aligned(total);
    double *out_re = alloc_aligned(total);
    double *out_im = alloc_aligned(total);

    fill_random(in_re, total);
    fill_random(in_im, total);

    uint64_t cyc_fwd[TRIALS], cyc_bwd[TRIALS];
    double   ns_fwd[TRIALS],  ns_bwd[TRIALS];

    /* --- forward --- */
    for (int i = 0; i < WARMUP; i++)
        fft_radix32_n1_forward_avx2(in_re, in_im, out_re, out_im, stride, stride);

    for (int i = 0; i < TRIALS; i++) {
        double t0 = now_ns();
        uint64_t c0 = rdtsc_fenced();
        fft_radix32_n1_forward_avx2(in_re, in_im, out_re, out_im, stride, stride);
        uint64_t c1 = rdtsc_fenced();
        double t1 = now_ns();
        cyc_fwd[i] = c1 - c0;
        ns_fwd[i]  = t1 - t0;
        /* chain: use output as next input to serialise */
        double *tmp;
        tmp = in_re; in_re = out_re; out_re = tmp;
        tmp = in_im; in_im = out_im; out_im = tmp;
    }

    /* --- backward --- */
    for (int i = 0; i < WARMUP; i++)
        fft_radix32_n1_backward_avx2(in_re, in_im, out_re, out_im, stride, stride);

    for (int i = 0; i < TRIALS; i++) {
        double t0 = now_ns();
        uint64_t c0 = rdtsc_fenced();
        fft_radix32_n1_backward_avx2(in_re, in_im, out_re, out_im, stride, stride);
        uint64_t c1 = rdtsc_fenced();
        double t1 = now_ns();
        cyc_bwd[i] = c1 - c0;
        ns_bwd[i]  = t1 - t0;
        double *tmp;
        tmp = in_re; in_re = out_re; out_re = tmp;
        tmp = in_im; in_im = out_im; out_im = tmp;
    }

    printf("  n1 forward:  median %5lu cyc  %6.1f ns\n",
           (unsigned long)median_u64(cyc_fwd, TRIALS), median_dbl(ns_fwd, TRIALS));
    printf("  n1 backward: median %5lu cyc  %6.1f ns\n",
           (unsigned long)median_u64(cyc_bwd, TRIALS), median_dbl(ns_bwd, TRIALS));

    /* free whichever pointers we ended up with */
    r32_aligned_free(in_re); r32_aligned_free(in_im);
    r32_aligned_free(out_re); r32_aligned_free(out_im);
}

/*==========================================================================
 * TWIDDLE VERSION BENCH
 *=========================================================================*/

static void bench_twiddle(size_t K)
{
    const size_t total = 32 * K;

    double *in_re   = alloc_aligned(total);
    double *in_im   = alloc_aligned(total);
    double *out_re  = alloc_aligned(total);
    double *out_im  = alloc_aligned(total);
    double *temp_re = alloc_aligned(total);
    double *temp_im = alloc_aligned(total);

    fill_random(in_re, total);
    fill_random(in_im, total);

    /* pass-1 twiddles */
    double *p1_re = alloc_aligned(2 * K);
    double *p1_im = alloc_aligned(2 * K);
    gen_pass1_twiddles(K, p1_re, p1_im);
    radix4_dit_stage_twiddles_blocked2_t p1_tw = {
        .re = p1_re, .im = p1_im, .K = K
    };

    /* pass-2 twiddles (BLOCKED8) */
    double *p2_re[8], *p2_im[8];
    for (int j = 0; j < 8; j++) {
        p2_re[j] = alloc_aligned(K);
        p2_im[j] = alloc_aligned(K);
    }
    gen_pass2_twiddles(K, p2_re, p2_im);

    tw_stage8_t p2_tw;
    p2_tw.mode = TW_MODE_BLOCKED8;
    for (int j = 0; j < 8; j++) {
        p2_tw.b8.re[j] = p2_re[j];
        p2_tw.b8.im[j] = p2_im[j];
    }
    p2_tw.b8.K = K;

    uint64_t cyc_fwd[TRIALS], cyc_bwd[TRIALS];
    double   ns_fwd[TRIALS],  ns_bwd[TRIALS];

    /* --- forward --- */
    for (int i = 0; i < WARMUP; i++)
        radix32_stage_forward_avx2(
            K, in_re, in_im, out_re, out_im,
            &p1_tw, &p2_tw, temp_re, temp_im);

    for (int i = 0; i < TRIALS; i++) {
        double t0 = now_ns();
        uint64_t c0 = rdtsc_fenced();
        radix32_stage_forward_avx2(
            K, in_re, in_im, out_re, out_im,
            &p1_tw, &p2_tw, temp_re, temp_im);
        uint64_t c1 = rdtsc_fenced();
        double t1 = now_ns();
        cyc_fwd[i] = c1 - c0;
        ns_fwd[i]  = t1 - t0;
        /* chain */
        double *tmp;
        tmp = in_re; in_re = out_re; out_re = tmp;
        tmp = in_im; in_im = out_im; out_im = tmp;
    }

    /* --- backward --- */
    for (int i = 0; i < WARMUP; i++)
        radix32_stage_backward_avx2(
            K, in_re, in_im, out_re, out_im,
            &p1_tw, &p2_tw, temp_re, temp_im);

    for (int i = 0; i < TRIALS; i++) {
        double t0 = now_ns();
        uint64_t c0 = rdtsc_fenced();
        radix32_stage_backward_avx2(
            K, in_re, in_im, out_re, out_im,
            &p1_tw, &p2_tw, temp_re, temp_im);
        uint64_t c1 = rdtsc_fenced();
        double t1 = now_ns();
        cyc_bwd[i] = c1 - c0;
        ns_bwd[i]  = t1 - t0;
        double *tmp;
        tmp = in_re; in_re = out_re; out_re = tmp;
        tmp = in_im; in_im = out_im; out_im = tmp;
    }

    printf("  twiddle K=%-4zu forward:  median %7lu cyc  %8.1f ns  (%.1f cyc/point)\n",
           K, (unsigned long)median_u64(cyc_fwd, TRIALS), median_dbl(ns_fwd, TRIALS),
           (double)median_u64(cyc_fwd, TRIALS) / (32.0 * (double)K));
    printf("  twiddle K=%-4zu backward: median %7lu cyc  %8.1f ns  (%.1f cyc/point)\n",
           K, (unsigned long)median_u64(cyc_bwd, TRIALS), median_dbl(ns_bwd, TRIALS),
           (double)median_u64(cyc_bwd, TRIALS) / (32.0 * (double)K));

    r32_aligned_free(in_re); r32_aligned_free(in_im);
    r32_aligned_free(out_re); r32_aligned_free(out_im);
    r32_aligned_free(temp_re); r32_aligned_free(temp_im);
    r32_aligned_free(p1_re); r32_aligned_free(p1_im);
    for (int j = 0; j < 8; j++) {
        r32_aligned_free(p2_re[j]); r32_aligned_free(p2_im[j]);
    }
}

/*==========================================================================
 * MAIN
 *=========================================================================*/

int main(void)
{
    printf("====== radix-32 AVX2 latency bench ======\n");
    printf("  %d warmup, %d trials, median reported\n\n", WARMUP, TRIALS);

    bench_n1();
    printf("\n");
    bench_twiddle(8);
    bench_twiddle(64);
    bench_twiddle(256);

    printf("\n====== done ======\n");
    return 0;
}
