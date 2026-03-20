/**
 * bench_walk.c — Find optimal walk crossover K for each radix
 *
 * For each walk-capable radix R, sweeps K values and times:
 *   1. STRIDED: tw_fwd(in, out, tw_re, tw_im, K)  — direct strided access
 *   2. WALK:    vfft_block_walk_tw(tw_fwd, ..., walk_state, block_bufs)
 *
 * Output: per-radix crossover K where walk starts winning.
 * Can be used to set per-radix walk thresholds in the planner.
 *
 * Build: same includes as bench_full_fft.c
 *   cl /O2 /arch:AVX2 bench_walk.c -I... -lfftw3
 *   or add to CMakeLists.txt alongside bench_factorize
 */

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════════════════════════════════════════════
 * ISA preamble (same as bench_full_fft.c)
 * ═══════════════════════════════════════════════════════════════ */

#ifndef VFFT_ISA_LEVEL_DEFINED
#define VFFT_ISA_LEVEL_DEFINED
typedef enum {
    VFFT_ISA_SCALAR = 0,
    VFFT_ISA_AVX2   = 1,
    VFFT_ISA_AVX512 = 2
} vfft_isa_level_t;
#endif

static inline vfft_isa_level_t vfft_detect_isa(void)
{
#if defined(__AVX512F__)
    return VFFT_ISA_AVX512;
#elif defined(__AVX2__)
    return VFFT_ISA_AVX2;
#else
    return VFFT_ISA_SCALAR;
#endif
}

/* ── All dispatch headers ── */
#include "fft_radix2_dispatch.h"
#include "fft_radix3_dispatch.h"
#include "fft_radix4_dispatch.h"
#include "fft_radix5_dispatch.h"

#define vfft_detect_isa _bench_walk_detect_isa_r7
#include "fft_radix7_dispatch.h"
#undef vfft_detect_isa

#include "fft_radix8_dispatch.h"
#include "fft_radix16_dispatch.h"

#define vfft_detect_isa _bench_walk_detect_isa_r32
#include "fft_radix32_dispatch.h"
#undef vfft_detect_isa

#include "fft_radix10_dispatch.h"
#include "fft_radix25_dispatch.h"

/* DIF */
#include "fft_radix2_dif_dispatch.h"
#include "fft_radix3_dif_dispatch.h"
#include "fft_radix4_dif_dispatch.h"
#include "fft_radix5_dif_dispatch.h"
#include "fft_radix7_dif_dispatch.h"
#include "fft_radix8_dif_dispatch.h"
#include "fft_radix16_dif_dispatch.h"
#include "fft_radix32_dif_dispatch.h"
#include "fft_radix10_dif_dispatch.h"
#include "fft_radix25_dif_dispatch.h"

/* Genfft primes */
#include "fft_radix11_genfft.h"
#include "fft_radix13_genfft.h"
#include "fft_radix17_genfft.h"
#include "fft_radix19_genfft.h"
#include "fft_radix23_genfft.h"

/* N1 */
#include "fft_radix64_n1.h"
#include "fft_radix128_n1.h"

/* Planner (for walk state, block walk, aligned alloc) */
#include "vfft_planner.h"
#include "vfft_calibration.h"

#define vfft_detect_isa _bench_walk_detect_isa_reg
#include "vfft_register_codelets.h"
#undef vfft_detect_isa

/* ═══════════════════════════════════════════════════════════════
 * Timing
 * ═══════════════════════════════════════════════════════════════ */

#ifdef _WIN32
#include <windows.h>
static double get_ns(void)
{
    static LARGE_INTEGER freq = {0};
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart * 1e9;
}
#else
#include <time.h>
static double get_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
#endif

/* ═══════════════════════════════════════════════════════════════
 * Walk Benchmark for One (R, K) Configuration
 *
 * Returns: strided_ns, walk_ns
 * ═══════════════════════════════════════════════════════════════ */

static void bench_one_rk(
    size_t R, size_t K,
    vfft_tw_codelet_fn tw_fwd,
    double *strided_ns_out, double *walk_ns_out)
{
    size_t N = R * K;

    /* Allocate data */
    double *in_re  = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    double *in_im  = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    double *out_re = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    double *out_im = (double *)vfft_aligned_alloc(64, N * sizeof(double));

    /* Build twiddle table */
    size_t tw_size = (R - 1) * K;
    double *tw_re = (double *)vfft_aligned_alloc(64, tw_size * sizeof(double));
    double *tw_im = (double *)vfft_aligned_alloc(64, tw_size * sizeof(double));
    for (size_t n = 1; n < R; n++)
        for (size_t k = 0; k < K; k++) {
            double a = -2.0 * M_PI * (double)(n * k) / (double)N;
            tw_re[(n-1)*K + k] = cos(a);
            tw_im[(n-1)*K + k] = sin(a);
        }

    /* Init input */
    srand(42 + (unsigned)(R * 1000 + K));
    for (size_t i = 0; i < N; i++) {
        in_re[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        in_im[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    }

    int reps;
    if (N <= 512) reps = 10000;
    else if (N <= 4096) reps = 5000;
    else if (N <= 32768) reps = 1000;
    else reps = 200;

    double t0, t1, best;

    /* ── STRIDED benchmark ── */
    /* Warm up */
    for (int r = 0; r < 5; r++)
        tw_fwd(in_re, in_im, out_re, out_im, tw_re, tw_im, K);

    best = 1e18;
    for (int trial = 0; trial < 5; trial++) {
        t0 = get_ns();
        for (int r = 0; r < reps; r++)
            tw_fwd(in_re, in_im, out_re, out_im, tw_re, tw_im, K);
        t1 = get_ns();
        double ns = (t1 - t0) / reps;
        if (ns < best) best = ns;
    }
    *strided_ns_out = best;

    /* ── WALK benchmark ── */
    size_t T = vfft_detect_T(K);
    if (T == 0 || K < T || (K % T) != 0 || R > VFFT_MAX_WALK_ARMS + 1) {
        /* Walk not applicable */
        *walk_ns_out = -1.0;
    } else {
        vfft_walk_state *ws = vfft_walk_create(R, K, T);
        size_t block_sz = R * T;
        double *blk_re     = (double *)vfft_aligned_alloc(64, block_sz * sizeof(double));
        double *blk_im     = (double *)vfft_aligned_alloc(64, block_sz * sizeof(double));
        double *blk_out_re = (double *)vfft_aligned_alloc(64, block_sz * sizeof(double));
        double *blk_out_im = (double *)vfft_aligned_alloc(64, block_sz * sizeof(double));

        /* Warm up */
        for (int r = 0; r < 5; r++)
            vfft_block_walk_tw(tw_fwd, in_re, in_im, out_re, out_im,
                               ws, blk_re, blk_im, blk_out_re, blk_out_im);

        best = 1e18;
        for (int trial = 0; trial < 5; trial++) {
            t0 = get_ns();
            for (int r = 0; r < reps; r++)
                vfft_block_walk_tw(tw_fwd, in_re, in_im, out_re, out_im,
                                   ws, blk_re, blk_im, blk_out_re, blk_out_im);
            t1 = get_ns();
            double ns = (t1 - t0) / reps;
            if (ns < best) best = ns;
        }
        *walk_ns_out = best;

        vfft_aligned_free(blk_re);
        vfft_aligned_free(blk_im);
        vfft_aligned_free(blk_out_re);
        vfft_aligned_free(blk_out_im);
        vfft_walk_destroy(ws);
    }

    vfft_aligned_free(in_re);  vfft_aligned_free(in_im);
    vfft_aligned_free(out_re); vfft_aligned_free(out_im);
    vfft_aligned_free(tw_re);  vfft_aligned_free(tw_im);
}

/* ═══════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    size_t R;
    const char *name;
    vfft_tw_codelet_fn tw_fwd;
} walk_radix_t;

int main(void)
{
    printf("══════════════════════════════════════════════════════════════\n");
    printf("  VectorFFT Walk Threshold Benchmark\n");
    printf("  L1d: %zu KB\n", vfft_l1d_bytes() / 1024);
    printf("══════════════════════════════════════════════════════════════\n\n");

    vfft_codelet_registry reg;
    vfft_register_all(&reg);

    /* Radixes with walk-capable tw codelets.
     * We need the raw tw_fwd function pointer for direct calls. */
    walk_radix_t radixes[] = {
        { 8,  "R=8",  reg.tw_fwd[8]  },
        { 10, "R=10", reg.tw_fwd[10] },
        { 11, "R=11", reg.tw_fwd[11] },
        { 13, "R=13", reg.tw_fwd[13] },
        { 16, "R=16", reg.tw_fwd[16] },
        { 25, "R=25", reg.tw_fwd[25] },
        { 32, "R=32", reg.tw_fwd[32] },
        { 64, "R=64", reg.tw_fwd[64] },
        { 0, NULL, NULL }
    };

    /* K values to sweep — must be SIMD-aligned (multiple of 4 for AVX2) */
    static const size_t K_VALUES[] = {
        4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 0
    };

    printf("%-6s %8s %10s %10s %8s  %s\n",
           "Radix", "K", "strided", "walk", "ratio", "winner");
    printf("%-6s %8s %10s %10s %8s  %s\n",
           "------", "--------", "----------", "----------", "--------", "------");

    /* Store results for summary */
    size_t crossovers[VFFT_MAX_RADIX];
    memset(crossovers, 0, sizeof(crossovers));

    for (walk_radix_t *wr = radixes; wr->R; wr++) {
        if (!wr->tw_fwd) {
            printf("%-6s  (no tw codelet)\n", wr->name);
            continue;
        }

        int walk_ever_wins = 0;

        for (const size_t *kp = K_VALUES; *kp; kp++) {
            size_t K = *kp;

            /* Skip if K too small for the radix (need at least T elements) */
            size_t T = vfft_detect_T(K);
            if (T == 0) continue;

            double strided_ns, walk_ns;
            bench_one_rk(wr->R, K, wr->tw_fwd, &strided_ns, &walk_ns);

            if (walk_ns < 0) {
                printf("%-6s %8zu %10.0f %10s %8s  strided (walk N/A)\n",
                       wr->name, K, strided_ns, "-", "-");
            } else {
                double ratio = strided_ns / walk_ns;
                int walk_wins = (walk_ns < strided_ns * 0.95);  /* 5% margin */
                if (walk_wins && !walk_ever_wins) {
                    crossovers[wr->R] = K;
                    walk_ever_wins = 1;
                }

                const char *winner = walk_wins ? "\033[92mWALK\033[0m" :
                                     (ratio > 1.05 ? "\033[93mwalk?\033[0m" : "strided");

                printf("%-6s %8zu %10.0f %10.0f %8.2f  %s\n",
                       wr->name, K, strided_ns, walk_ns, ratio, winner);
            }
        }

        if (walk_ever_wins) {
            printf("%-6s  ──→ crossover at K=%zu (tw_table = %zuKB)\n\n",
                   wr->name, crossovers[wr->R],
                   (wr->R - 1) * crossovers[wr->R] * 16 / 1024);
        } else {
            printf("%-6s  ──→ walk NEVER wins in tested range\n\n", wr->name);
        }
    }

    /* ── Save calibration ── */
    printf("\n══════════════════════════════════════════════════════════════\n");
    printf("  WALK THRESHOLD SUMMARY\n");
    printf("  L1d = %zu KB\n", vfft_l1d_bytes() / 1024);
    printf("══════════════════════════════════════════════════════════════\n\n");

    /* Load existing calibration (preserves IL data from bench_il) */
    vfft_calibration cal;
    vfft_calibration_init(&cal);
    vfft_calibration_load(&cal, "vfft_calibration.txt");  /* OK if missing */

    /* Merge walk results */
    for (walk_radix_t *wr = radixes; wr->R; wr++) {
        if (!wr->tw_fwd) continue;
        cal.walk_threshold[wr->R] = crossovers[wr->R];

        if (crossovers[wr->R] > 0)
            printf("  R=%-3zu  walk at K >= %zu  (tw=%zuKB)\n",
                   wr->R, crossovers[wr->R],
                   (wr->R - 1) * crossovers[wr->R] * 16 / 1024);
        else
            printf("  R=%-3zu  walk never wins\n", wr->R);
    }

    const char *isa_name =
#if defined(__AVX512F__)
        "AVX-512";
#elif defined(__AVX2__)
        "AVX2";
#else
        "scalar";
#endif

    if (vfft_calibration_save(&cal, "vfft_calibration.txt",
                               vfft_l1d_bytes(), isa_name) == 0) {
        printf("\n  Calibration written to: vfft_calibration.txt\n");
    } else {
        printf("\n  ERROR: failed to write vfft_calibration.txt\n");
    }

    return 0;
}
