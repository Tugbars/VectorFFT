/**
 * @file bench_radix32_vs_fftw.c
 * @brief Radix-32 FFT stage vs FFTW performance comparison
 *
 * Compares our radix-32 implementation against FFTW across multiple scenarios:
 *
 *   SCENARIO A — N=1 codelet (K batch of 32-point DFTs, no twiddles)
 *     A1: Radix-32 N=1 AVX2 codelet
 *     A2: Radix-32 N=1 AVX-512 codelet
 *     A3: FFTW batch 32-pt (strided, our layout)
 *     A4: FFTW batch 32-pt (contiguous, best-case for FFTW)
 *
 *   SCENARIO B — Twiddle stage (radix-32 stage within larger FFT)
 *     B1: Radix-32 auto-dispatch (AVX-512 → AVX2 → scalar)
 *     B2: Radix-32 forced AVX-512
 *     B3: Radix-32 forced AVX2
 *     B4: FFTW full 32·K-point FFT (split complex, FFTW_MEASURE)
 *     B5: FFTW full 32·K-point FFT (split complex, FFTW_PATIENT)
 *
 * Data layout: [32 stripes][K samples], split real/imag (SoA).
 * FFTW uses guru split-complex interface for zero-copy comparison.
 *
 * Methodology:
 *   - rdtsc with lfence (serialised latency, not throughput)
 *   - FFTW plans created with FFTW_MEASURE (and FFTW_PATIENT for reference)
 *   - Warmup discarded, median of N trials (robust to outliers)
 *   - Output→input chain prevents OoO overlap
 *   - Deterministic PRNG for reproducibility
 *
 * Build:
 *   cmake --build . --target bench_radix32_vs_fftw
 *   # or manually:
 *   gcc -O2 -mavx2 -mfma -mavx512f -mavx512dq -mavx512vl \
 *       -o bench_vs_fftw bench_radix32_vs_fftw.c -lfftw3 -lm
 *
 * @author Tugbars
 * @date 2025
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#define R32_NEED_TIMER
#include "../fft_radix32_platform.h"

/* Our radix-32 headers */
#include "fft_radix32_uniform.h"    /* ISA detection, types, dispatch externs */

/* FFTW */
#include <fftw3.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*==========================================================================
 * Extern declarations: dispatch drivers (from fft_radix32_fv.c / bv.c)
 *=========================================================================*/

extern radix32_isa_level_t radix32_forward(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT pass1_tw,
    const tw_stage8_t *RESTRICT pass2_tw,
    const tw_recurrence_scalar_t *RESTRICT rec_tw,
    double *RESTRICT temp_re, double *RESTRICT temp_im);

extern void radix32_forward_force_avx2(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT pass1_tw,
    const tw_stage8_t *RESTRICT pass2_tw,
    const tw_recurrence_scalar_t *RESTRICT rec_tw);

extern void radix32_forward_force_scalar(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT pass1_tw,
    const tw_stage8_t *RESTRICT pass2_tw,
    const tw_recurrence_scalar_t *RESTRICT rec_tw);

#ifdef __AVX512F__
extern void radix32_forward_force_avx512(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    const radix4_dit_stage_twiddles_blocked2_t *RESTRICT pass1_tw,
    const tw_stage8_t *RESTRICT pass2_tw,
    const tw_recurrence_scalar_t *RESTRICT rec_tw,
    double *RESTRICT temp_re, double *RESTRICT temp_im);
#endif

/* N=1 codelet headers (all-inline, static functions) */
#include "fft_radix32_avx2_n1.h"
#include "fft_radix32_avx512_n1.h"
#include "fft_radix32_scalar_n1.h"

/*==========================================================================
 * CONFIGURATION
 *=========================================================================*/

#define WARMUP    500
#define TRIALS   3000

/* Approx flops per 32-point: ~8.9 (N=1) or ~24 (twiddled) */
#define N1_FLOPS_PER_POINT    8.9
#define TW_FLOPS_PER_POINT   24.0

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
    double   p10_ns;
    double   p90_ns;
} bench_stats_t;

static bench_stats_t compute_stats(uint64_t *cyc, double *ns, size_t n)
{
    qsort(cyc, n, sizeof(uint64_t), cmp_u64);
    qsort(ns,  n, sizeof(double),   cmp_dbl);

    bench_stats_t r;
    r.median_cyc = cyc[n / 2];
    r.p10_cyc    = cyc[n / 10];
    r.p90_cyc    = cyc[n * 9 / 10];
    r.median_ns  = ns[n / 2];
    r.p10_ns     = ns[n / 10];
    r.p90_ns     = ns[n * 9 / 10];
    return r;
}

/*==========================================================================
 * ALIGNED ALLOCATION
 *=========================================================================*/

static double *aa(size_t count)
{
    double *p = (double *)r32_aligned_alloc(64, count * sizeof(double));
    if (!p) {
        fprintf(stderr, "FATAL: r32_aligned_alloc(%zu) failed\n", count);
        exit(1);
    }
    memset(p, 0, count * sizeof(double));
    return p;
}

static void fill_rand(double *buf, size_t n, unsigned seed)
{
    for (size_t i = 0; i < n; i++) {
        seed = seed * 1103515245 + 12345;
        buf[i] = ((double)(seed >> 16) / 32768.0) - 1.0;
    }
}

/*==========================================================================
 * TWIDDLE SETUP (reused from dispatch test pattern)
 *=========================================================================*/

typedef struct {
    double *p1r, *p1i;
    double *p2r[8], *p2i[8];
    radix4_dit_stage_twiddles_blocked2_t p1;
    tw_stage8_t p2;
} tw_ctx_t;

static void tw_init(size_t K, tw_ctx_t *t)
{
    tw_mode_t mode = pick_tw_mode(K);

    t->p1r = aa(2 * K);
    t->p1i = aa(2 * K);
    for (size_t k = 0; k < K; k++) {
        double a = -2.0 * M_PI * (double)k / (32.0 * (double)K);
        t->p1r[k]     = cos(a);      t->p1i[k]     = sin(a);
        t->p1r[K + k] = cos(2 * a);  t->p1i[K + k] = sin(2 * a);
    }
    t->p1 = (radix4_dit_stage_twiddles_blocked2_t){
        .re = t->p1r, .im = t->p1i, .K = K
    };

    for (int j = 0; j < 8; j++) {
        t->p2r[j] = aa(K);
        t->p2i[j] = aa(K);
        double base = -2.0 * M_PI * (double)(j + 1) / (8.0 * (double)K);
        for (size_t k = 0; k < K; k++) {
            double a = base * (double)k;
            t->p2r[j][k] = cos(a);
            t->p2i[j][k] = sin(a);
        }
    }

    if (mode == TW_MODE_BLOCKED8) {
        t->p2.mode = TW_MODE_BLOCKED8;
        for (int j = 0; j < 8; j++) {
            t->p2.b8.re[j] = t->p2r[j];
            t->p2.b8.im[j] = t->p2i[j];
        }
        t->p2.b8.K = K;
    } else {
        /* BLOCKED4 (and RECURRENCE not yet wired) */
        t->p2.mode = TW_MODE_BLOCKED4;
        for (int j = 0; j < 4; j++) {
            t->p2.b4.re[j] = t->p2r[j];
            t->p2.b4.im[j] = t->p2i[j];
        }
        t->p2.b4.K = K;
    }
}

static void tw_free(tw_ctx_t *t)
{
    r32_aligned_free(t->p1r);
    r32_aligned_free(t->p1i);
    for (int j = 0; j < 8; j++) {
        r32_aligned_free(t->p2r[j]);
        r32_aligned_free(t->p2i[j]);
    }
}

/*==========================================================================
 * RESULT REPORTING
 *=========================================================================*/

static void print_header(const char *title, size_t K, size_t total_points)
{
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  %-59s  ║\n", title);
    printf("║  K = %-6zu   total = %-8zu doubles (%.1f KB)              ║\n",
           K, total_points, (double)(total_points * 2 * sizeof(double)) / 1024.0);
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║  %-14s %9s %9s %9s %8s %7s ║\n",
           "implementation", "med_cyc", "med_ns", "p90_ns", "cyc/pt", "speedup");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
}

static void print_row(const char *name, bench_stats_t s,
                       size_t total_points, double ref_ns)
{
    double cyc_per_pt = (double)s.median_cyc / (double)total_points;
    double speedup = (ref_ns > 0) ? ref_ns / s.median_ns : 0.0;
    printf("║  %-14s %9lu %9.1f %9.1f %8.2f %6.2fx ║\n",
           name,
           (unsigned long)s.median_cyc,
           s.median_ns,
           s.p90_ns,
           cyc_per_pt,
           speedup);
}

static void print_footer(void)
{
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
}

/*==========================================================================
 * SCENARIO A: N=1 codelet vs FFTW batch of 32-point DFTs
 *
 * Apples-to-apples: both compute K independent 32-pt forward DFTs.
 * Data layout: [32 stripes][K] split re/im.
 *=========================================================================*/

static void bench_n1_vs_fftw(size_t K)
{
    const size_t N = 32;
    const size_t total = N * K;
    char title[80];
    snprintf(title, sizeof(title),
             "SCENARIO A: N=1 codelet vs FFTW batch (%zu x 32-pt)", K);
    print_header(title, K, total);

    /* Allocate split-complex buffers */
    double *in_re  = aa(total);
    double *in_im  = aa(total);
    double *out_re = aa(total);
    double *out_im = aa(total);

    /* FFTW contiguous buffers (K DFTs of length 32, packed) */
    double *fw_ir = aa(total);
    double *fw_ii = aa(total);
    double *fw_or = aa(total);
    double *fw_oi = aa(total);

    fill_rand(in_re, total, 42);
    fill_rand(in_im, total, 137);

    uint64_t *cyc_buf = (uint64_t *)malloc(TRIALS * sizeof(uint64_t));
    double   *ns_buf  = (double *)  malloc(TRIALS * sizeof(double));

    double fftw_strided_ns = 0.0;
    double fftw_contig_ns  = 0.0;

    /* ── FFTW batch, strided (our [32][K] layout) ────────────────── */
    {
        fftw_iodim dims       = { .n = (int)N, .is = (int)K, .os = (int)K };
        fftw_iodim howmany    = { .n = (int)K, .is = 1,      .os = 1 };

        fftw_plan plan = fftw_plan_guru_split_dft(
            1, &dims, 1, &howmany,
            in_re, in_im, out_re, out_im,
            FFTW_MEASURE);

        if (!plan) {
            printf("║  %-14s  FFTW plan failed!                            ║\n",
                   "fftw_strided");
        } else {
            /* warmup */
            for (int i = 0; i < WARMUP; i++)
                fftw_execute(plan);

            for (int i = 0; i < TRIALS; i++) {
                uint64_t c0 = rdtsc_fenced();
                double   t0 = now_ns();
                fftw_execute(plan);
                uint64_t c1 = rdtsc_fenced();
                double   t1 = now_ns();
                cyc_buf[i] = c1 - c0;
                ns_buf[i]  = t1 - t0;
                /* serialise: copy one output back to input */
                in_re[0] = out_re[0];
            }
            bench_stats_t s = compute_stats(cyc_buf, ns_buf, TRIALS);
            fftw_strided_ns = s.median_ns;
            print_row("fftw_strided", s, total, fftw_strided_ns);
            fftw_destroy_plan(plan);
        }
    }

    /* ── FFTW batch, contiguous (best-case FFTW layout) ──────────── */
    {
        /* Copy data to contiguous layout: [K][32] */
        for (size_t k = 0; k < K; k++)
            for (size_t s = 0; s < N; s++) {
                fw_ir[k * N + s] = in_re[s * K + k];
                fw_ii[k * N + s] = in_im[s * K + k];
            }

        fftw_iodim dims    = { .n = (int)N, .is = 1, .os = 1 };
        fftw_iodim howmany = { .n = (int)K, .is = (int)N, .os = (int)N };

        fftw_plan plan = fftw_plan_guru_split_dft(
            1, &dims, 1, &howmany,
            fw_ir, fw_ii, fw_or, fw_oi,
            FFTW_MEASURE);

        if (!plan) {
            printf("║  %-14s  FFTW plan failed!                            ║\n",
                   "fftw_contig");
        } else {
            for (int i = 0; i < WARMUP; i++)
                fftw_execute(plan);

            for (int i = 0; i < TRIALS; i++) {
                uint64_t c0 = rdtsc_fenced();
                double   t0 = now_ns();
                fftw_execute(plan);
                uint64_t c1 = rdtsc_fenced();
                double   t1 = now_ns();
                cyc_buf[i] = c1 - c0;
                ns_buf[i]  = t1 - t0;
                fw_ir[0] = fw_or[0];
            }
            bench_stats_t s = compute_stats(cyc_buf, ns_buf, TRIALS);
            fftw_contig_ns = s.median_ns;
            print_row("fftw_contig", s, total, fftw_strided_ns);
            fftw_destroy_plan(plan);
        }
    }

    /* Reference: use best FFTW time for speedup column */
    double fftw_best_ns = (fftw_contig_ns > 0 && fftw_contig_ns < fftw_strided_ns)
                          ? fftw_contig_ns : fftw_strided_ns;

    /* ── R32 N=1 AVX2 (processes 4 DFTs per call, loop K/4 times) ── */
    {
        const size_t VW = 4;  /* AVX2 double lanes */
        assert(K >= VW && (K % VW) == 0);

        /* Warmup */
        for (int i = 0; i < WARMUP; i++)
            for (size_t k = 0; k < K; k += VW)
                fft_radix32_n1_forward_avx2(
                    in_re + k, in_im + k, out_re + k, out_im + k, K, K);

        for (int i = 0; i < TRIALS; i++) {
            uint64_t c0 = rdtsc_fenced();
            double   t0 = now_ns();
            for (size_t k = 0; k < K; k += VW)
                fft_radix32_n1_forward_avx2(
                    in_re + k, in_im + k, out_re + k, out_im + k, K, K);
            uint64_t c1 = rdtsc_fenced();
            double   t1 = now_ns();
            cyc_buf[i] = c1 - c0;
            ns_buf[i]  = t1 - t0;
            in_re[0] = out_re[0];
        }
        bench_stats_t s = compute_stats(cyc_buf, ns_buf, TRIALS);
        print_row("r32_n1_avx2", s, total, fftw_best_ns);
    }

    /* ── R32 N=1 AVX-512 (processes all K DFTs in one call) ─────── */
    if (K >= 8 && (K % 8) == 0 && K <= 128) {
        /* K > 128 can exceed internal temp sizing in the AVX-512 N=1 codelet.
         * This is a codelet-level limitation, not a dispatch limitation. */
        double *t_re = aa(total);
        double *t_im = aa(total);

        for (int i = 0; i < WARMUP; i++)
            radix32_n1_forward_avx512(K, in_re, in_im, out_re, out_im,
                                       t_re, t_im);

        for (int i = 0; i < TRIALS; i++) {
            uint64_t c0 = rdtsc_fenced();
            double   t0 = now_ns();
            radix32_n1_forward_avx512(K, in_re, in_im, out_re, out_im,
                                       t_re, t_im);
            uint64_t c1 = rdtsc_fenced();
            double   t1 = now_ns();
            cyc_buf[i] = c1 - c0;
            ns_buf[i]  = t1 - t0;
            in_re[0] = out_re[0];
        }
        bench_stats_t s = compute_stats(cyc_buf, ns_buf, TRIALS);
        print_row("r32_n1_avx512", s, total, fftw_best_ns);

        r32_aligned_free(t_re);
        r32_aligned_free(t_im);
    }

    print_footer();

    r32_aligned_free(in_re);  r32_aligned_free(in_im);
    r32_aligned_free(out_re); r32_aligned_free(out_im);
    r32_aligned_free(fw_ir);  r32_aligned_free(fw_ii);
    r32_aligned_free(fw_or);  r32_aligned_free(fw_oi);
    free(cyc_buf); free(ns_buf);
}

/*==========================================================================
 * SCENARIO B: Twiddle stage vs FFTW full N=32·K-point FFT
 *
 * Our twiddle stage is ONE stage of a larger Cooley-Tukey FFT, so this
 * comparison is NOT apples-to-apples. But it answers the practical
 * question: "How fast is our radix-32 stage relative to a complete
 * FFTW FFT of the same total size?"
 *
 * If our single stage is faster than FFTW's complete FFT, that's a
 * strong result — the remaining stages just need to keep up.
 *=========================================================================*/

static void bench_twiddle_vs_fftw(size_t K)
{
    const size_t N = 32;
    const size_t total = N * K;
    tw_mode_t mode = pick_tw_mode(K);
    const char *mode_str = (mode == TW_MODE_BLOCKED8)  ? "BLK8" :
                           (mode == TW_MODE_BLOCKED4)  ? "BLK4" :
                                                          "REC";
    char title[80];
    snprintf(title, sizeof(title),
             "SCENARIO B: twiddle(%s) vs FFTW full %zuK-pt",
             mode_str, total / 1024 > 0 ? total / 1024 : total);
    if (total < 1024)
        snprintf(title, sizeof(title),
                 "SCENARIO B: twiddle(%s) vs FFTW full %zu-pt",
                 mode_str, total);
    print_header(title, K, total);

    /* Allocate */
    double *in_re   = aa(total);
    double *in_im   = aa(total);
    double *out_re  = aa(total);
    double *out_im  = aa(total);
    double *temp_re = aa(total);
    double *temp_im = aa(total);

    fill_rand(in_re, total, 42);
    fill_rand(in_im, total, 137);

    uint64_t *cyc_buf = (uint64_t *)malloc(TRIALS * sizeof(uint64_t));
    double   *ns_buf  = (double *)  malloc(TRIALS * sizeof(double));

    /* Twiddle tables */
    tw_ctx_t tw;
    tw_init(K, &tw);

    double fftw_measure_ns = 0.0;

    /* ── FFTW full 32·K point FFT (MEASURE) ──────────────────────── */
    {
        /* FFTW uses our strided layout: transform length=total, stride=1 */
        fftw_iodim dims = { .n = (int)total, .is = 1, .os = 1 };

        fftw_plan plan = fftw_plan_guru_split_dft(
            1, &dims, 0, NULL,
            in_re, in_im, out_re, out_im,
            FFTW_MEASURE);

        if (!plan) {
            printf("║  %-14s  FFTW plan failed!                            ║\n",
                   "fftw_measure");
        } else {
            for (int i = 0; i < WARMUP; i++)
                fftw_execute(plan);

            for (int i = 0; i < TRIALS; i++) {
                uint64_t c0 = rdtsc_fenced();
                double   t0 = now_ns();
                fftw_execute(plan);
                uint64_t c1 = rdtsc_fenced();
                double   t1 = now_ns();
                cyc_buf[i] = c1 - c0;
                ns_buf[i]  = t1 - t0;
                in_re[0] = out_re[0];
            }
            bench_stats_t s = compute_stats(cyc_buf, ns_buf, TRIALS);
            fftw_measure_ns = s.median_ns;
            print_row("fftw_measure", s, total, fftw_measure_ns);
            fftw_destroy_plan(plan);
        }
    }

    /* ── FFTW full 32·K point FFT (PATIENT) ──────────────────────── */
    /* Skip PATIENT for large sizes (planning can take minutes) */
    if (total <= 16384) {
        fftw_iodim dims = { .n = (int)total, .is = 1, .os = 1 };

        fftw_plan plan = fftw_plan_guru_split_dft(
            1, &dims, 0, NULL,
            in_re, in_im, out_re, out_im,
            FFTW_PATIENT);

        if (!plan) {
            printf("║  %-14s  FFTW plan failed!                            ║\n",
                   "fftw_patient");
        } else {
            for (int i = 0; i < WARMUP; i++)
                fftw_execute(plan);

            for (int i = 0; i < TRIALS; i++) {
                uint64_t c0 = rdtsc_fenced();
                double   t0 = now_ns();
                fftw_execute(plan);
                uint64_t c1 = rdtsc_fenced();
                double   t1 = now_ns();
                cyc_buf[i] = c1 - c0;
                ns_buf[i]  = t1 - t0;
                in_re[0] = out_re[0];
            }
            bench_stats_t s = compute_stats(cyc_buf, ns_buf, TRIALS);
            print_row("fftw_patient", s, total, fftw_measure_ns);
            fftw_destroy_plan(plan);
        }

        /* Re-fill input (FFTW_PATIENT may have clobbered it) */
        fill_rand(in_re, total, 42);
        fill_rand(in_im, total, 137);
    }

    /* ── R32 auto-dispatch ───────────────────────────────────────── */
    {
        for (int i = 0; i < WARMUP; i++)
            radix32_forward(K, in_re, in_im, out_re, out_im,
                            &tw.p1, &tw.p2, NULL, temp_re, temp_im);

        for (int i = 0; i < TRIALS; i++) {
            uint64_t c0 = rdtsc_fenced();
            double   t0 = now_ns();
            radix32_forward(K, in_re, in_im, out_re, out_im,
                            &tw.p1, &tw.p2, NULL, temp_re, temp_im);
            uint64_t c1 = rdtsc_fenced();
            double   t1 = now_ns();
            cyc_buf[i] = c1 - c0;
            ns_buf[i]  = t1 - t0;
            in_re[0] = out_re[0];
        }
        bench_stats_t s = compute_stats(cyc_buf, ns_buf, TRIALS);
        print_row("r32_auto", s, total, fftw_measure_ns);
    }

    /* ── R32 forced AVX2 ─────────────────────────────────────────── */
    if (mode == TW_MODE_BLOCKED8 || mode == TW_MODE_BLOCKED4) {
        for (int i = 0; i < WARMUP; i++)
            radix32_forward_force_avx2(K, in_re, in_im, out_re, out_im,
                                        &tw.p1, &tw.p2, NULL);

        for (int i = 0; i < TRIALS; i++) {
            uint64_t c0 = rdtsc_fenced();
            double   t0 = now_ns();
            radix32_forward_force_avx2(K, in_re, in_im, out_re, out_im,
                                        &tw.p1, &tw.p2, NULL);
            uint64_t c1 = rdtsc_fenced();
            double   t1 = now_ns();
            cyc_buf[i] = c1 - c0;
            ns_buf[i]  = t1 - t0;
            in_re[0] = out_re[0];
        }
        bench_stats_t s = compute_stats(cyc_buf, ns_buf, TRIALS);
        print_row("r32_avx2", s, total, fftw_measure_ns);
    }

    /* ── R32 forced AVX-512 (BLOCKED8 only) ──────────────────────── */
#ifdef __AVX512F__
    if (mode == TW_MODE_BLOCKED8) {
        for (int i = 0; i < WARMUP; i++)
            radix32_forward_force_avx512(K, in_re, in_im, out_re, out_im,
                                          &tw.p1, &tw.p2, NULL,
                                          temp_re, temp_im);

        for (int i = 0; i < TRIALS; i++) {
            uint64_t c0 = rdtsc_fenced();
            double   t0 = now_ns();
            radix32_forward_force_avx512(K, in_re, in_im, out_re, out_im,
                                          &tw.p1, &tw.p2, NULL,
                                          temp_re, temp_im);
            uint64_t c1 = rdtsc_fenced();
            double   t1 = now_ns();
            cyc_buf[i] = c1 - c0;
            ns_buf[i]  = t1 - t0;
            in_re[0] = out_re[0];
        }
        bench_stats_t s = compute_stats(cyc_buf, ns_buf, TRIALS);
        print_row("r32_avx512", s, total, fftw_measure_ns);
    }
#endif

    /* ── R32 forced scalar (reference) ───────────────────────────── */
    {
        for (int i = 0; i < WARMUP; i++)
            radix32_forward_force_scalar(K, in_re, in_im, out_re, out_im,
                                          &tw.p1, &tw.p2, NULL);

        for (int i = 0; i < TRIALS; i++) {
            uint64_t c0 = rdtsc_fenced();
            double   t0 = now_ns();
            radix32_forward_force_scalar(K, in_re, in_im, out_re, out_im,
                                          &tw.p1, &tw.p2, NULL);
            uint64_t c1 = rdtsc_fenced();
            double   t1 = now_ns();
            cyc_buf[i] = c1 - c0;
            ns_buf[i]  = t1 - t0;
            in_re[0] = out_re[0];
        }
        bench_stats_t s = compute_stats(cyc_buf, ns_buf, TRIALS);
        print_row("r32_scalar", s, total, fftw_measure_ns);
    }

    print_footer();

    tw_free(&tw);
    r32_aligned_free(in_re);  r32_aligned_free(in_im);
    r32_aligned_free(out_re); r32_aligned_free(out_im);
    r32_aligned_free(temp_re); r32_aligned_free(temp_im);
    free(cyc_buf); free(ns_buf);
}

/*==========================================================================
 * SCENARIO C: Scaling sweep — how does our stage scale with K?
 *
 * Compact table: one row per K, columns = r32_auto vs fftw_measure
 *=========================================================================*/

static void bench_scaling_sweep(void)
{
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  SCENARIO C: Scaling sweep (radix-32 auto-dispatch vs FFTW MEASURE)    ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════════╣\n");
    printf("║  %6s %6s %5s │ %10s %8s │ %10s %8s │ %7s ║\n",
           "K", "N·K", "mode", "r32_ns", "r32_c/pt",
           "fftw_ns", "fftw_c/pt", "speedup");
    printf("╠══════════════════════════════════════════════════════════════════════════╣\n");

    static const size_t K_values[] = {
        8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
    };
    const size_t nK = sizeof(K_values) / sizeof(K_values[0]);

    uint64_t *cyc_buf = (uint64_t *)malloc(TRIALS * sizeof(uint64_t));
    double   *ns_buf  = (double *)  malloc(TRIALS * sizeof(double));

    for (size_t ki = 0; ki < nK; ki++) {
        const size_t K = K_values[ki];
        const size_t total = 32 * K;
        tw_mode_t mode = pick_tw_mode(K);
        const char *mode_str = (mode == TW_MODE_BLOCKED8) ? "BLK8" :
                               (mode == TW_MODE_BLOCKED4) ? "BLK4" :
                                                             "REC";

        double *in_re   = aa(total);
        double *in_im   = aa(total);
        double *out_re  = aa(total);
        double *out_im  = aa(total);
        double *temp_re = aa(total);
        double *temp_im = aa(total);

        fill_rand(in_re, total, 42);
        fill_rand(in_im, total, 137);

        tw_ctx_t tw;
        tw_init(K, &tw);

        /* Bench r32 auto-dispatch */
        for (int i = 0; i < WARMUP; i++)
            radix32_forward(K, in_re, in_im, out_re, out_im,
                            &tw.p1, &tw.p2, NULL, temp_re, temp_im);

        for (int i = 0; i < TRIALS; i++) {
            uint64_t c0 = rdtsc_fenced();
            double   t0 = now_ns();
            radix32_forward(K, in_re, in_im, out_re, out_im,
                            &tw.p1, &tw.p2, NULL, temp_re, temp_im);
            uint64_t c1 = rdtsc_fenced();
            double   t1 = now_ns();
            cyc_buf[i] = c1 - c0;
            ns_buf[i]  = t1 - t0;
            in_re[0] = out_re[0];
        }
        bench_stats_t r32_s = compute_stats(cyc_buf, ns_buf, TRIALS);

        /* Re-fill (dispatch may have clobbered) */
        fill_rand(in_re, total, 42);
        fill_rand(in_im, total, 137);

        /* Bench FFTW (batch of K 32-pt DFTs, strided) — same work as our stage */
        fftw_iodim dims    = { .n = 32,     .is = (int)K, .os = (int)K };
        fftw_iodim howmany = { .n = (int)K, .is = 1,      .os = 1 };

        fftw_plan plan = fftw_plan_guru_split_dft(
            1, &dims, 1, &howmany,
            in_re, in_im, out_re, out_im,
            FFTW_MEASURE);

        bench_stats_t fftw_s = {0};
        if (plan) {
            for (int i = 0; i < WARMUP; i++)
                fftw_execute(plan);

            for (int i = 0; i < TRIALS; i++) {
                uint64_t c0 = rdtsc_fenced();
                double   t0 = now_ns();
                fftw_execute(plan);
                uint64_t c1 = rdtsc_fenced();
                double   t1 = now_ns();
                cyc_buf[i] = c1 - c0;
                ns_buf[i]  = t1 - t0;
                in_re[0] = out_re[0];
            }
            fftw_s = compute_stats(cyc_buf, ns_buf, TRIALS);
            fftw_destroy_plan(plan);
        }

        double r32_cpt  = (double)r32_s.median_cyc / (double)total;
        double fftw_cpt = (fftw_s.median_ns > 0)
                          ? (double)fftw_s.median_cyc / (double)total : 0;
        double speedup  = (fftw_s.median_ns > 0)
                          ? fftw_s.median_ns / r32_s.median_ns : 0;

        printf("║  %6zu %6zu %5s │ %10.1f %8.2f │ %10.1f %8.2f │ %6.2fx ║\n",
               K, total, mode_str,
               r32_s.median_ns, r32_cpt,
               fftw_s.median_ns, fftw_cpt,
               speedup);

        tw_free(&tw);
        r32_aligned_free(in_re);  r32_aligned_free(in_im);
        r32_aligned_free(out_re); r32_aligned_free(out_im);
        r32_aligned_free(temp_re); r32_aligned_free(temp_im);
    }

    printf("╚══════════════════════════════════════════════════════════════════════════╝\n");
    printf("  speedup = fftw_ns / r32_ns  (>1.0 = r32 faster)\n");
    printf("  Note: FFTW does batch 32-pt DFTs (same work as r32 stage).\n");
    printf("        Our twiddle stage also applies stage twiddles (extra work vs FFTW).\n");

    free(cyc_buf); free(ns_buf);
}

/*==========================================================================
 * MAIN
 *=========================================================================*/

int main(void)
{
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  radix-32 FFT vs FFTW %s performance comparison\n",
           fftw_version);
    printf("  ISA: %s\n",
           radix32_get_isa_level() == ISA_AVX512 ? "AVX-512" :
           radix32_get_isa_level() == ISA_AVX2   ? "AVX2"    :
                                                    "scalar");
    printf("  Trials: %d (warmup: %d)   Metric: median latency\n",
           TRIALS, WARMUP);
    printf("═══════════════════════════════════════════════════════════════\n");

    /* Scenario A: N=1 codelet vs FFTW batch */
    bench_n1_vs_fftw(8);     /*  256 points */
    bench_n1_vs_fftw(64);    /* 2048 points */
    bench_n1_vs_fftw(256);   /* 8192 points */
    bench_n1_vs_fftw(1024);  /* 32K points  */

    /* Scenario B: Twiddle stage vs FFTW full FFT */
    bench_twiddle_vs_fftw(8);      /* BLOCKED8,  256 pt */
    bench_twiddle_vs_fftw(64);     /* BLOCKED8, 2048 pt */
    bench_twiddle_vs_fftw(256);    /* BLOCKED8, 8192 pt */
    bench_twiddle_vs_fftw(512);    /* BLOCKED4, 16K pt  */

    /* Scenario C: Full scaling sweep */
    bench_scaling_sweep();

    fftw_cleanup();

    printf("\nDone.\n");
    return 0;
}
