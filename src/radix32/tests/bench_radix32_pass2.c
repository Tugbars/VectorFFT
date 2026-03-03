/**
 * @file bench_radix32_pass2.c
 * @brief Performance benchmark for radix-32 AVX2 DIF-8 stage (Pass 2)
 *
 * Measures:
 *   1) Pass 2 (DIF-8) in isolation — directly shows fused-butterfly gains
 *   2) Full radix-32 stage (Pass 1 + Pass 2) — end-to-end impact
 *
 * Covers all twiddle modes:
 *   - BLOCKED8  (K ≤ 256)
 *   - BLOCKED4  (256 < K ≤ 4096)
 *   - RECURRENCE (K > 4096)
 *
 * Reports: median cycles, ns, cycles/point, GFlop/s estimate
 *
 * Methodology:
 *   - rdtsc with lfence serialisation (latency)
 *   - output→input chain to prevent OoO from overlapping iterations
 *   - warmup phase, 1000 trials, median (robust to outliers)
 *   - deterministic PRNG for reproducibility
 *
 * Build:
 *   gcc -O2 -mavx2 -mfma -march=native -lm -o bench_pass2 bench_radix32_pass2.c
 *   # or via cmake: cmake --build . --target bench_radix32_pass2
 */

/* CPU affinity (Linux only — optional for pinning) */
#if defined(__linux__)
#  ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#  endif
#  include <sched.h>
#  define R32_HAS_SCHED_AFFINITY 1
#else
#  define R32_HAS_SCHED_AFFINITY 0
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#define R32_NEED_TIMER
#include "../fft_radix32_platform.h"

#include "fft_radix32_avx2.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*==========================================================================
 * CONFIGURATION
 *=========================================================================*/

#define WARMUP    500
#define TRIALS   2000

/* Radix-8 DIF flops per point:
 * 7 twiddle cmuls (6 FMA = 24 flops, +1 free -j) + 8-pt butterfly (34 add/sub)
 * ≈ 58 flops/point is a reasonable estimate for throughput calculation */
#define DIF8_FLOPS_PER_POINT  58.0
#define R32_FLOPS_PER_POINT  120.0  /* both passes combined */

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

static int cmp_dbl(const void *a, const void *b)
{
    double va = *(const double *)a;
    double vb = *(const double *)b;
    return (va > vb) - (va < vb);
}

typedef struct {
    uint64_t median_cyc;
    uint64_t p10_cyc;
    uint64_t p90_cyc;
    double   median_ns;
} bench_result_t;

static bench_result_t compute_stats(uint64_t *cyc, double *ns, size_t n)
{
    qsort(cyc, n, sizeof(uint64_t), cmp_u64);
    qsort(ns,  n, sizeof(double),   cmp_dbl);

    bench_result_t r;
    r.median_cyc = cyc[n / 2];
    r.p10_cyc    = cyc[n / 10];
    r.p90_cyc    = cyc[n * 9 / 10];
    r.median_ns  = ns[n / 2];
    return r;
}

/*==========================================================================
 * ALIGNED ALLOCATION
 *=========================================================================*/

static double *alloc_aligned(size_t count)
{
    double *p = (double *)r32_aligned_alloc(64, count * sizeof(double));
    if (!p) {
        fprintf(stderr, "FATAL: r32_aligned_alloc(%zu) failed\n", count);
        exit(1);
    }
    memset(p, 0, count * sizeof(double));
    return p;
}

/*==========================================================================
 * SEEDED PRNG (xoshiro256**)
 *=========================================================================*/

static uint64_t rng_s[4] = {
    0x180EC6D33CFD0ABAULL, 0xD5A61266F0C9392CULL,
    0xA9582618E03FC9AAULL, 0x39ABDC4529B1661CULL
};

static uint64_t rng_next(void)
{
    uint64_t result = ((rng_s[1] * 5) << 7 | (rng_s[1] * 5) >> 57) * 9;
    uint64_t t = rng_s[1] << 17;
    rng_s[2] ^= rng_s[0]; rng_s[3] ^= rng_s[1];
    rng_s[1] ^= rng_s[2]; rng_s[0] ^= rng_s[3];
    rng_s[2] ^= t; rng_s[3] = (rng_s[3] << 45) | (rng_s[3] >> 19);
    return result;
}

static double rng_uniform(void)
{
    return (double)(int64_t)rng_next() * 5.42101086242752217e-20;
}

static void fill_random(double *buf, size_t count)
{
    for (size_t i = 0; i < count; i++)
        buf[i] = rng_uniform();
}

/*==========================================================================
 * TWIDDLE GENERATION
 *=========================================================================*/

/** Pass-1 DIT-4 twiddles: W_m(k) = exp(-2πi·m·k / (32K)) for m=1,2 */
static void gen_pass1_twiddles(size_t K, double *tw_re, double *tw_im)
{
    const double base = -2.0 * M_PI / (32.0 * (double)K);
    for (size_t k = 0; k < K; k++) {
        double ang = base * (double)k;
        tw_re[0 * K + k] = cos(ang);
        tw_im[0 * K + k] = sin(ang);
        tw_re[1 * K + k] = cos(2.0 * ang);
        tw_im[1 * K + k] = sin(2.0 * ang);
    }
}

/**
 * Pass-2 DIF-8 twiddles: Wj(k) = exp(-2πi·(j+1)·k / (8K))
 * for j=0..7, k=0..K-1
 *
 * Populates all 8 arrays (always needed for BLOCKED8/seed generation).
 */
static void gen_pass2_twiddles_raw(size_t K, double **tw_re, double **tw_im)
{
    for (int j = 0; j < 8; j++) {
        const double base = -2.0 * M_PI * (double)(j + 1) / (8.0 * (double)K);
        for (size_t k = 0; k < K; k++) {
            double ang = base * (double)k;
            tw_re[j][k] = cos(ang);
            tw_im[j][k] = sin(ang);
        }
    }
}

/*==========================================================================
 * TWIDDLE TABLE SETUP PER MODE
 *=========================================================================*/

typedef struct {
    tw_stage8_t stage;

    /* Backing storage (freed in teardown) */
    double *p2_re[8];
    double *p2_im[8];
    double *seed_re;    /* RECURRENCE only */
    double *seed_im;
    double *delta_re;
    double *delta_im;
} pass2_tw_t;

static void setup_blocked8(pass2_tw_t *tw, size_t K)
{
    memset(tw, 0, sizeof(*tw));
    tw->stage.mode = TW_MODE_BLOCKED8;
    for (int j = 0; j < 8; j++) {
        tw->p2_re[j] = alloc_aligned(K);
        tw->p2_im[j] = alloc_aligned(K);
    }
    gen_pass2_twiddles_raw(K, tw->p2_re, tw->p2_im);
    for (int j = 0; j < 8; j++) {
        tw->stage.b8.re[j] = tw->p2_re[j];
        tw->stage.b8.im[j] = tw->p2_im[j];
    }
    tw->stage.b8.K = K;
}

static void setup_blocked4(pass2_tw_t *tw, size_t K)
{
    memset(tw, 0, sizeof(*tw));
    tw->stage.mode = TW_MODE_BLOCKED4;

    /* Generate all 8, but only store first 4 in the BLOCKED4 struct */
    for (int j = 0; j < 8; j++) {
        tw->p2_re[j] = alloc_aligned(K);
        tw->p2_im[j] = alloc_aligned(K);
    }
    gen_pass2_twiddles_raw(K, tw->p2_re, tw->p2_im);

    for (int j = 0; j < 4; j++) {
        tw->stage.b4.re[j] = tw->p2_re[j];
        tw->stage.b4.im[j] = tw->p2_im[j];
    }
    tw->stage.b4.K = K;
}

static void setup_recurrence(pass2_tw_t *tw, size_t K)
{
    memset(tw, 0, sizeof(*tw));
    tw->stage.mode = TW_MODE_RECURRENCE;

    /* tile_len: 64 steps of 4 elements = 256 elements per tile */
    const int tile_len = 256;

    /* Allocate backing storage for gen (all 8 full arrays) */
    for (int j = 0; j < 8; j++) {
        tw->p2_re[j] = alloc_aligned(K);
        tw->p2_im[j] = alloc_aligned(K);
    }
    gen_pass2_twiddles_raw(K, tw->p2_re, tw->p2_im);

    /* Seeds: [8][K] — exact twiddle values at every k position.
     * rec8_tile_init loads seeds[j*K + k] at tile boundaries,
     * then steps with delta between boundaries. For a correct bench,
     * we store the exact values everywhere and let the stepping
     * introduce its natural (tiny) drift. */
    tw->seed_re = alloc_aligned(8 * K);
    tw->seed_im = alloc_aligned(8 * K);

    for (int j = 0; j < 8; j++) {
        for (size_t k = 0; k < K; k++) {
            tw->seed_re[j * K + k] = tw->p2_re[j][k];
            tw->seed_im[j * K + k] = tw->p2_im[j][k];
        }
    }

    /* Deltas: δj⁴ = exp(-2πi·(j+1)·4 / (8K)) */
    tw->delta_re = alloc_aligned(8);
    tw->delta_im = alloc_aligned(8);
    for (int j = 0; j < 8; j++) {
        double ang = -2.0 * M_PI * (double)(j + 1) * 4.0 / (8.0 * (double)K);
        tw->delta_re[j] = cos(ang);
        tw->delta_im[j] = sin(ang);
    }

    tw->stage.rec.tile_len = tile_len;
    tw->stage.rec.seed_re  = tw->seed_re;
    tw->stage.rec.seed_im  = tw->seed_im;
    tw->stage.rec.delta_re = tw->delta_re;
    tw->stage.rec.delta_im = tw->delta_im;
    tw->stage.rec.K        = K;
}

static void teardown_tw(pass2_tw_t *tw)
{
    for (int j = 0; j < 8; j++) {
        r32_aligned_free(tw->p2_re[j]);
        r32_aligned_free(tw->p2_im[j]);
    }
    r32_aligned_free(tw->seed_re);
    r32_aligned_free(tw->seed_im);
    r32_aligned_free(tw->delta_re);
    r32_aligned_free(tw->delta_im);
}

/*==========================================================================
 * AUTO MODE SETUP
 *=========================================================================*/

static void setup_pass2_auto(pass2_tw_t *tw, size_t K)
{
    tw_mode_t mode = pick_tw_mode(K);
    switch (mode) {
    case TW_MODE_BLOCKED8:   setup_blocked8(tw, K);    break;
    case TW_MODE_BLOCKED4:   setup_blocked4(tw, K);    break;
    case TW_MODE_RECURRENCE: setup_recurrence(tw, K);  break;
    }
}

/*==========================================================================
 * BENCH: PASS 2 ONLY (DIF-8 stage in isolation)
 *=========================================================================*/

static void bench_pass2(size_t K)
{
    const size_t n_points = 8 * K;   /* 8 stripes × K */
    const size_t total = 8 * K;

    double *in_re  = alloc_aligned(total);
    double *in_im  = alloc_aligned(total);
    double *out_re = alloc_aligned(total);
    double *out_im = alloc_aligned(total);

    fill_random(in_re, total);
    fill_random(in_im, total);

    pass2_tw_t tw;
    setup_pass2_auto(&tw, K);

    const char *mode_name = "???";
    switch (tw.stage.mode) {
    case TW_MODE_BLOCKED8:   mode_name = "BLOCKED8";   break;
    case TW_MODE_BLOCKED4:   mode_name = "BLOCKED4";   break;
    case TW_MODE_RECURRENCE: mode_name = "RECURRENCE";  break;
    }

    uint64_t cyc_fwd[TRIALS], cyc_bwd[TRIALS];
    double   ns_fwd[TRIALS],  ns_bwd[TRIALS];

    /* --- warmup --- */
    for (int i = 0; i < WARMUP; i++) {
        radix8_dif_stage_multimode_avx2(K, in_re, in_im, out_re, out_im,
                                         &tw.stage, 0);
    }

    /* --- forward --- */
    for (int i = 0; i < TRIALS; i++) {
        _mm256_zeroupper();
        double t0 = now_ns();
        uint64_t c0 = rdtsc_fenced();
        radix8_dif_stage_multimode_avx2(K, in_re, in_im, out_re, out_im,
                                         &tw.stage, 0);
        uint64_t c1 = rdtsc_fenced();
        double t1 = now_ns();
        cyc_fwd[i] = c1 - c0;
        ns_fwd[i]  = t1 - t0;
        /* chain: output → input */
        double *tmp;
        tmp = in_re; in_re = out_re; out_re = tmp;
        tmp = in_im; in_im = out_im; out_im = tmp;
    }

    bench_result_t fwd = compute_stats(cyc_fwd, ns_fwd, TRIALS);

    /* --- warmup bwd --- */
    for (int i = 0; i < WARMUP; i++) {
        radix8_dif_stage_multimode_avx2(K, in_re, in_im, out_re, out_im,
                                         &tw.stage, 1);
    }

    /* --- backward --- */
    for (int i = 0; i < TRIALS; i++) {
        _mm256_zeroupper();
        double t0 = now_ns();
        uint64_t c0 = rdtsc_fenced();
        radix8_dif_stage_multimode_avx2(K, in_re, in_im, out_re, out_im,
                                         &tw.stage, 1);
        uint64_t c1 = rdtsc_fenced();
        double t1 = now_ns();
        cyc_bwd[i] = c1 - c0;
        ns_bwd[i]  = t1 - t0;
        double *tmp;
        tmp = in_re; in_re = out_re; out_re = tmp;
        tmp = in_im; in_im = out_im; out_im = tmp;
    }

    bench_result_t bwd = compute_stats(cyc_bwd, ns_bwd, TRIALS);

    double fwd_cpp = (double)fwd.median_cyc / (double)n_points;
    double bwd_cpp = (double)bwd.median_cyc / (double)n_points;
    double data_kb = (double)(n_points * 2 * sizeof(double)) / 1024.0;

    printf("  %-10s K=%-5zu | %5.0f KB | fwd %7lu cyc (%4.1f c/pt) %7.0f ns"
           " [p10=%lu p90=%lu]\n",
           mode_name, K, data_kb,
           (unsigned long)fwd.median_cyc, fwd_cpp, fwd.median_ns,
           (unsigned long)fwd.p10_cyc, (unsigned long)fwd.p90_cyc);
    printf("  %10s %7s |          | bwd %7lu cyc (%4.1f c/pt) %7.0f ns"
           " [p10=%lu p90=%lu]\n",
           "", "",
           (unsigned long)bwd.median_cyc, bwd_cpp, bwd.median_ns,
           (unsigned long)bwd.p10_cyc, (unsigned long)bwd.p90_cyc);

    r32_aligned_free(in_re); r32_aligned_free(in_im);
    r32_aligned_free(out_re); r32_aligned_free(out_im);
    teardown_tw(&tw);
}

/*==========================================================================
 * BENCH: FULL RADIX-32 STAGE (Pass 1 + Pass 2)
 *=========================================================================*/

static void bench_full_stage(size_t K)
{
    const size_t total   = 32 * K;
    const size_t n_points = total;

    double *in_re   = alloc_aligned(total);
    double *in_im   = alloc_aligned(total);
    double *out_re  = alloc_aligned(total);
    double *out_im  = alloc_aligned(total);
    double *temp_re = alloc_aligned(total);
    double *temp_im = alloc_aligned(total);

    fill_random(in_re, total);
    fill_random(in_im, total);

    /* Pass-1 twiddles */
    double *p1_re = alloc_aligned(2 * K);
    double *p1_im = alloc_aligned(2 * K);
    gen_pass1_twiddles(K, p1_re, p1_im);
    radix4_dit_stage_twiddles_blocked2_t p1_tw = {
        .re = p1_re, .im = p1_im, .K = K
    };

    /* Pass-2 twiddles (auto-selected mode) */
    pass2_tw_t p2;
    setup_pass2_auto(&p2, K);

    const char *mode_name = "???";
    switch (p2.stage.mode) {
    case TW_MODE_BLOCKED8:   mode_name = "BLOCKED8";   break;
    case TW_MODE_BLOCKED4:   mode_name = "BLOCKED4";   break;
    case TW_MODE_RECURRENCE: mode_name = "RECURRENCE";  break;
    }

    uint64_t cyc_fwd[TRIALS], cyc_bwd[TRIALS];
    double   ns_fwd[TRIALS],  ns_bwd[TRIALS];

    /* --- warmup fwd --- */
    for (int i = 0; i < WARMUP; i++) {
        radix32_stage_forward_avx2(K, in_re, in_im, out_re, out_im,
                                    &p1_tw, &p2.stage, temp_re, temp_im);
    }

    /* --- forward --- */
    for (int i = 0; i < TRIALS; i++) {
        _mm256_zeroupper();
        double t0 = now_ns();
        uint64_t c0 = rdtsc_fenced();
        radix32_stage_forward_avx2(K, in_re, in_im, out_re, out_im,
                                    &p1_tw, &p2.stage, temp_re, temp_im);
        uint64_t c1 = rdtsc_fenced();
        double t1 = now_ns();
        cyc_fwd[i] = c1 - c0;
        ns_fwd[i]  = t1 - t0;
        double *tmp;
        tmp = in_re; in_re = out_re; out_re = tmp;
        tmp = in_im; in_im = out_im; out_im = tmp;
    }

    bench_result_t fwd = compute_stats(cyc_fwd, ns_fwd, TRIALS);

    /* --- warmup bwd --- */
    for (int i = 0; i < WARMUP; i++) {
        radix32_stage_backward_avx2(K, in_re, in_im, out_re, out_im,
                                     &p1_tw, &p2.stage, temp_re, temp_im);
    }

    /* --- backward --- */
    for (int i = 0; i < TRIALS; i++) {
        _mm256_zeroupper();
        double t0 = now_ns();
        uint64_t c0 = rdtsc_fenced();
        radix32_stage_backward_avx2(K, in_re, in_im, out_re, out_im,
                                     &p1_tw, &p2.stage, temp_re, temp_im);
        uint64_t c1 = rdtsc_fenced();
        double t1 = now_ns();
        cyc_bwd[i] = c1 - c0;
        ns_bwd[i]  = t1 - t0;
        double *tmp;
        tmp = in_re; in_re = out_re; out_re = tmp;
        tmp = in_im; in_im = out_im; out_im = tmp;
    }

    bench_result_t bwd = compute_stats(cyc_bwd, ns_bwd, TRIALS);

    double fwd_cpp = (double)fwd.median_cyc / (double)n_points;
    double bwd_cpp = (double)bwd.median_cyc / (double)n_points;
    double data_kb = (double)(n_points * 2 * sizeof(double)) / 1024.0;

    printf("  %-10s K=%-5zu | %6.0f KB | fwd %8lu cyc (%4.1f c/pt) %8.0f ns"
           " [p10=%lu p90=%lu]\n",
           mode_name, K, data_kb,
           (unsigned long)fwd.median_cyc, fwd_cpp, fwd.median_ns,
           (unsigned long)fwd.p10_cyc, (unsigned long)fwd.p90_cyc);
    printf("  %10s %7s |          | bwd %8lu cyc (%4.1f c/pt) %8.0f ns"
           " [p10=%lu p90=%lu]\n",
           "", "",
           (unsigned long)bwd.median_cyc, bwd_cpp, bwd.median_ns,
           (unsigned long)bwd.p10_cyc, (unsigned long)bwd.p90_cyc);

    r32_aligned_free(in_re);  r32_aligned_free(in_im);
    r32_aligned_free(out_re); r32_aligned_free(out_im);
    r32_aligned_free(temp_re); r32_aligned_free(temp_im);
    r32_aligned_free(p1_re);  r32_aligned_free(p1_im);
    teardown_tw(&p2);
}

/*==========================================================================
 * FREQUENCY ESTIMATION (for reporting)
 *=========================================================================*/

static double estimate_freq_ghz(void)
{
    /* Quick and dirty: measure rdtsc rate over 50ms of busy spinning */
    volatile double sink = 0.0;
    double t0 = now_ns();
    uint64_t c0 = rdtsc_fenced();
    while (now_ns() - t0 < 50e6) {
        sink += 1.0;
    }
    uint64_t c1 = rdtsc_fenced();
    double t1 = now_ns();
    (void)sink;
    return (double)(c1 - c0) / (t1 - t0);
}

/*==========================================================================
 * MAIN
 *=========================================================================*/

int main(int argc, char **argv)
{
    /* Try to pin to core 0 for stability (non-fatal if fails) */
#if R32_HAS_SCHED_AFFINITY
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
#endif

    double freq_ghz = estimate_freq_ghz();

    printf("================================================================\n");
    printf(" radix-32 AVX2 performance bench\n");
    printf("================================================================\n");
    printf("  Estimated TSC freq: %.2f GHz\n", freq_ghz);
    printf("  Warmup: %d  Trials: %d  Metric: median (serialised latency)\n", WARMUP, TRIALS);
    printf("================================================================\n\n");

    /* ---- Pass 2 isolation ---- */
    printf("--- Pass 2: DIF-8 Stage Only (8 stripes × K) ---\n");
    printf("  [This directly measures the fused-butterfly + spill reduction]\n\n");

    /* BLOCKED8: K ≤ 256 */
    bench_pass2(4);
    bench_pass2(8);
    bench_pass2(16);
    bench_pass2(32);
    bench_pass2(64);
    bench_pass2(128);
    bench_pass2(256);
    printf("\n");

    /* BLOCKED4: 256 < K ≤ 4096 */
    bench_pass2(512);
    bench_pass2(1024);
    bench_pass2(2048);
    bench_pass2(4096);
    printf("\n");

    /* RECURRENCE: K > 4096 */
    bench_pass2(8192);
    bench_pass2(16384);
    printf("\n");

    /* ---- Full radix-32 stage ---- */
    printf("--- Full Radix-32 Stage (Pass 1 DIT-4 + Pass 2 DIF-8, 32×K points) ---\n\n");

    /* BLOCKED8 range */
    bench_full_stage(4);
    bench_full_stage(8);
    bench_full_stage(32);
    bench_full_stage(64);
    bench_full_stage(256);
    printf("\n");

    /* BLOCKED4 range */
    bench_full_stage(512);
    bench_full_stage(1024);
    bench_full_stage(4096);
    printf("\n");

    /* RECURRENCE range */
    bench_full_stage(8192);
    bench_full_stage(16384);
    printf("\n");

    printf("================================================================\n");
    printf(" Notes:\n");
    printf("  - c/pt = cycles per complex point (lower is better)\n");
    printf("  - p10/p90 = 10th/90th percentile cycles (variability)\n");
    printf("  - Pass 2 operates on 8×K points; full stage on 32×K points\n");
    printf("  - Serialised latency (output→input chained, not throughput)\n");
    printf("  - For throughput, multiply by OoO overlap factor (~1.5-2×)\n");
    printf("================================================================\n");

    return 0;
}
