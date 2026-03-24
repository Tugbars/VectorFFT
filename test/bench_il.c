/**
 * bench_il.c — Find optimal IL (interleaved) crossover K for each radix
 *
 * For each IL-capable radix R, sweeps K values and times:
 *   1. SPLIT:     tw_fwd(in_re, in_im, out_re, out_im, tw_re, tw_im, K)
 *   2. HYBRID IL: tw_fwd_il(in_il, out_il, tw_re, tw_im, K)
 *   3. NATIVE IL: tw_fwd_il_native(in_il, out_il, tw_il, K)
 *
 * No layout conversion in the timing loop — we pre-arrange data in the
 * correct format. In the real executor, conversion happens once at the
 * split↔IL transition, not per-stage.
 *
 * Native IL codelets take pre-interleaved tw_il instead of split tw_re/tw_im.
 * The planner allocates tw_il at plan time via vfft_repack_tw_to_il().
 * Native IL eliminates permutex2var overhead from hybrid IL codelets.
 *
 * Output: per-radix crossover K where IL (best of hybrid/native) starts winning.
 * Results replace the hardcoded 256/512 values in vfft_register_codelets.h.
 *
 * Build: same as bench_walk.c / bench_full_fft.c
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
 * ISA preamble
 * ═══════════════════════════════════════════════════════════════ */

#ifndef VFFT_ISA_LEVEL_DEFINED
#define VFFT_ISA_LEVEL_DEFINED
typedef enum
{
    VFFT_ISA_SCALAR = 0,
    VFFT_ISA_AVX2 = 1,
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

#define vfft_detect_isa _bench_il_detect_isa_r7
#include "fft_radix7_dispatch.h"
#undef vfft_detect_isa

#include "fft_radix8_dispatch.h"
#include "fft_radix16_dispatch.h"

#define vfft_detect_isa _bench_il_detect_isa_r32
#include "fft_radix32_dispatch.h"
#undef vfft_detect_isa

#include "fft_radix10_dispatch.h"
#include "fft_radix25_dispatch.h"

/* R=20 dispatch (if available) */
#ifdef FFT_RADIX20_DISPATCH_H_AVAILABLE
#include "fft_radix20_dispatch.h"
#endif

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
/* R=128 not needed — N1 only, no IL capability */

/* Planner (for aligned alloc, types) */
#include "vfft_planner.h"
#include "vfft_calibration.h"

#define vfft_detect_isa _bench_il_detect_isa_reg
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
    if (!freq.QuadPart)
        QueryPerformanceFrequency(&freq);
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
 * IL Benchmark for One (R, K) Configuration
 *
 * Compares split tw codelet vs hybrid IL vs native IL, raw timing.
 * Data pre-arranged in correct layout — no conversion overhead.
 *
 * native_ns_out set to -1.0 if no native IL codelet available.
 * ═══════════════════════════════════════════════════════════════ */

static void bench_one_il(
    size_t R, size_t K,
    vfft_tw_codelet_fn tw_fwd_split,
    vfft_tw_il_codelet_fn tw_fwd_il,
    vfft_tw_il_native_codelet_fn tw_fwd_il_native,
    double *split_ns_out, double *il_ns_out, double *native_ns_out)
{
    size_t N = R * K;

    /* ── Split data ── */
    double *s_in_re = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    double *s_in_im = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    double *s_out_re = (double *)vfft_aligned_alloc(64, N * sizeof(double));
    double *s_out_im = (double *)vfft_aligned_alloc(64, N * sizeof(double));

    /* ── IL data (interleaved: re[0],im[0],re[1],im[1],...) ── */
    double *il_in = (double *)vfft_aligned_alloc(64, 2 * N * sizeof(double));
    double *il_out = (double *)vfft_aligned_alloc(64, 2 * N * sizeof(double));

    /* ── Twiddle tables ── */
    size_t tw_entries = (R - 1) * K;
    double *tw_re = (double *)vfft_aligned_alloc(64, tw_entries * sizeof(double));
    double *tw_im = (double *)vfft_aligned_alloc(64, tw_entries * sizeof(double));

    /* Pre-interleaved twiddle table for native IL */
    double *tw_il = NULL;
    if (tw_fwd_il_native)
    {
        tw_il = (double *)vfft_aligned_alloc(64, tw_entries * 2 * sizeof(double));
    }

    for (size_t n = 1; n < R; n++)
        for (size_t k = 0; k < K; k++)
        {
            double a = -2.0 * M_PI * (double)(n * k) / (double)N;
            double wr = cos(a), wi = sin(a);
            tw_re[(n - 1) * K + k] = wr;
            tw_im[(n - 1) * K + k] = wi;
            if (tw_il)
            {
                tw_il[(n - 1) * K * 2 + k * 2 + 0] = wr;
                tw_il[(n - 1) * K * 2 + k * 2 + 1] = wi;
            }
        }

    /* ── Fill input ── */
    srand(42 + (unsigned)(R * 1000 + K));
    for (size_t i = 0; i < N; i++)
    {
        double re = (double)rand() / RAND_MAX * 2.0 - 1.0;
        double im = (double)rand() / RAND_MAX * 2.0 - 1.0;
        s_in_re[i] = re;
        s_in_im[i] = im;
        il_in[2 * i] = re;
        il_in[2 * i + 1] = im;
    }

    int reps;
    if (N <= 512)
        reps = 10000;
    else if (N <= 4096)
        reps = 5000;
    else if (N <= 32768)
        reps = 1000;
    else
        reps = 200;

    double t0, t1, best;

    /* ── SPLIT benchmark ── */
    if (tw_fwd_split)
    {
        for (int r = 0; r < 5; r++)
            tw_fwd_split(s_in_re, s_in_im, s_out_re, s_out_im, tw_re, tw_im, K);

        best = 1e18;
        for (int trial = 0; trial < 5; trial++)
        {
            t0 = get_ns();
            for (int r = 0; r < reps; r++)
                tw_fwd_split(s_in_re, s_in_im, s_out_re, s_out_im, tw_re, tw_im, K);
            t1 = get_ns();
            double ns = (t1 - t0) / reps;
            if (ns < best)
                best = ns;
        }
        *split_ns_out = best;
    }
    else
    {
        *split_ns_out = -1.0;
    }

    /* ── HYBRID IL benchmark ── */
    if (tw_fwd_il)
    {
        for (int r = 0; r < 5; r++)
            tw_fwd_il(il_in, il_out, tw_re, tw_im, K);

        best = 1e18;
        for (int trial = 0; trial < 5; trial++)
        {
            t0 = get_ns();
            for (int r = 0; r < reps; r++)
                tw_fwd_il(il_in, il_out, tw_re, tw_im, K);
            t1 = get_ns();
            double ns = (t1 - t0) / reps;
            if (ns < best)
                best = ns;
        }
        *il_ns_out = best;
    }
    else
    {
        *il_ns_out = -1.0;
    }

    /* ── NATIVE IL benchmark ── */
    if (tw_fwd_il_native && tw_il)
    {
        for (int r = 0; r < 5; r++)
            tw_fwd_il_native(il_in, il_out, tw_il, K);

        best = 1e18;
        for (int trial = 0; trial < 5; trial++)
        {
            t0 = get_ns();
            for (int r = 0; r < reps; r++)
                tw_fwd_il_native(il_in, il_out, tw_il, K);
            t1 = get_ns();
            double ns = (t1 - t0) / reps;
            if (ns < best)
                best = ns;
        }
        *native_ns_out = best;
    }
    else
    {
        *native_ns_out = -1.0;
    }

    vfft_aligned_free(s_in_re);
    vfft_aligned_free(s_in_im);
    vfft_aligned_free(s_out_re);
    vfft_aligned_free(s_out_im);
    vfft_aligned_free(il_in);
    vfft_aligned_free(il_out);
    vfft_aligned_free(tw_re);
    vfft_aligned_free(tw_im);
    vfft_aligned_free(tw_il);
}

/* ═══════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════ */

typedef struct
{
    size_t R;
    const char *name;
    vfft_tw_codelet_fn tw_fwd_split;
    vfft_tw_il_codelet_fn tw_fwd_il;               /* hybrid: takes tw_re/tw_im */
    vfft_tw_il_native_codelet_fn tw_fwd_il_native; /* native: takes tw_il (NULL if unavailable) */
} il_radix_t;

int main(void)
{
    printf("══════════════════════════════════════════════════════════════\n");
    printf("  VectorFFT IL Crossover Benchmark\n");
    printf("  L1d: %zu KB\n", vfft_l1d_bytes() / 1024);
    printf("══════════════════════════════════════════════════════════════\n\n");

    vfft_codelet_registry reg;
    vfft_register_all(&reg);

    /* IL-capable radixes with split, hybrid IL, and/or native IL codelets.
     * A radix can have hybrid IL only, native IL only, or both.
     * The bench tests all available paths and picks the best IL. */
    il_radix_t radixes[] = {
        {5, "R=5", reg.tw_fwd[5], reg.tw_fwd_il[5], reg.tw_fwd_il_native[5]},
        {7, "R=7", reg.tw_fwd[7], reg.tw_fwd_il[7], reg.tw_fwd_il_native[7]},
        {8, "R=8", reg.tw_fwd[8], reg.tw_fwd_il[8], reg.tw_fwd_il_native[8]},
        {10, "R=10", reg.tw_fwd[10], reg.tw_fwd_il[10], reg.tw_fwd_il_native[10]},
        {16, "R=16", reg.tw_fwd[16], reg.tw_fwd_il[16], reg.tw_fwd_il_native[16]},
        {20, "R=20", reg.tw_fwd[20], reg.tw_fwd_il[20], reg.tw_fwd_il_native[20]},
        {25, "R=25", reg.tw_fwd[25], reg.tw_fwd_il[25], reg.tw_fwd_il_native[25]},
        {32, "R=32", reg.tw_fwd[32], reg.tw_fwd_il[32], reg.tw_fwd_il_native[32]},
        {64, "R=64", reg.tw_fwd[64], reg.tw_fwd_il[64], reg.tw_fwd_il_native[64]},
        {0, NULL, NULL, NULL, NULL}};

    /* K values: must be SIMD-aligned (multiple of 4 for AVX2, 8 for AVX-512) */
    static const size_t K_VALUES[] = {
        4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 0};

    /* Store crossover results */
    size_t crossovers[VFFT_MAX_RADIX];
    memset(crossovers, 0, sizeof(crossovers));

    printf("%-6s %8s %10s %10s %10s %8s  %s\n",
           "Radix", "K", "split_ns", "hybIL_ns", "natIL_ns", "ratio", "winner");
    printf("%-6s %8s %10s %10s %10s %8s  %s\n",
           "------", "--------", "----------", "----------", "----------", "--------", "------");

    for (il_radix_t *ir = radixes; ir->R; ir++)
    {
        int have_split = (ir->tw_fwd_split != NULL);
        int have_hybrid = (ir->tw_fwd_il != NULL);
        int have_native = (ir->tw_fwd_il_native != NULL);

        if (!have_split && !have_hybrid && !have_native)
        {
            printf("%-6s  (no codelets registered)\n\n", ir->name);
            continue;
        }
        if (!have_split)
        {
            printf("%-6s  (no split TW codelet — skip crossover)\n\n", ir->name);
            continue;
        }
        if (!have_hybrid && !have_native)
        {
            printf("%-6s  (no IL codelet — split only)\n\n", ir->name);
            continue;
        }

        int il_ever_wins = 0;

        for (const size_t *kp = K_VALUES; *kp; kp++)
        {
            size_t K = *kp;

            /* SIMD alignment check */
            if ((K & 3) != 0)
                continue;

            /* Skip unreasonably large working sets */
            if (ir->R * K > 200000)
                continue;

            double split_ns, hybrid_ns, native_ns;
            bench_one_il(ir->R, K,
                         ir->tw_fwd_split,
                         ir->tw_fwd_il,
                         ir->tw_fwd_il_native,
                         &split_ns, &hybrid_ns, &native_ns);

            /* Best IL time: pick the faster of hybrid and native (if available) */
            double best_il_ns = 1e18;
            const char *il_type = "---";

            if (have_hybrid && hybrid_ns > 0)
            {
                best_il_ns = hybrid_ns;
                il_type = "hyb";
            }
            if (have_native && native_ns > 0 && native_ns < best_il_ns)
            {
                best_il_ns = native_ns;
                il_type = "nat";
            }

            double ratio = (best_il_ns < 1e17) ? split_ns / best_il_ns : 0.0;
            int il_wins = (best_il_ns < split_ns * 0.95); /* 5% margin */

            if (il_wins && !il_ever_wins)
            {
                crossovers[ir->R] = K;
                il_ever_wins = 1;
            }

            /* Winner string */
            const char *winner;
            if (il_wins)
                winner = (native_ns > 0 && native_ns <= hybrid_ns) ? "\033[92mIL(nat)\033[0m" : "\033[92mIL(hyb)\033[0m";
            else if (ratio > 1.05)
                winner = (native_ns > 0 && native_ns <= hybrid_ns) ? "\033[93mil(nat)?\033[0m" : "\033[93mil(hyb)?\033[0m";
            else if (ratio < 0.90)
                winner = "SPLIT";
            else
                winner = "~tie";

            /* If only native IL (no hybrid), adjust label */
            if (!have_hybrid && have_native && il_wins)
                winner = "\033[92mIL(nat)\033[0m";

            /* Working set info */
            size_t ws_split = ir->R * K * 32; /* 4 arrays × 8 bytes */
            size_t ws_il = ir->R * K * 16;    /* 2 arrays × 8 bytes */

            /* Format ns columns */
            char hyb_str[16], nat_str[16];
            if (have_hybrid && hybrid_ns > 0)
                snprintf(hyb_str, sizeof(hyb_str), "%10.0f", hybrid_ns);
            else
                snprintf(hyb_str, sizeof(hyb_str), "%10s", "---");

            if (have_native && native_ns > 0)
                snprintf(nat_str, sizeof(nat_str), "%10.0f", native_ns);
            else
                snprintf(nat_str, sizeof(nat_str), "%10s", "---");

            printf("%-6s %8zu %10.0f %s %s %8.2f  %-16s  split=%3zuKB  il=%3zuKB\n",
                   ir->name, K, split_ns, hyb_str, nat_str,
                   ratio, winner, ws_split / 1024, ws_il / 1024);
        }

        if (il_ever_wins)
        {
            printf("%-6s  ──→ IL crossover at K=%zu\n\n",
                   ir->name, crossovers[ir->R]);
        }
        else
        {
            printf("%-6s  ──→ IL NEVER wins in tested range\n\n", ir->name);
        }
    }

    /* ── Save calibration ── */
    printf("\n══════════════════════════════════════════════════════════════\n");
    printf("  IL CROSSOVER SUMMARY\n");
    printf("  L1d = %zu KB\n", vfft_l1d_bytes() / 1024);
    printf("══════════════════════════════════════════════════════════════\n\n");

    /* Load existing calibration (preserves walk data from bench_walk) */
    vfft_calibration cal;
    vfft_calibration_init(&cal);
    vfft_calibration_load(&cal, "vfft_calibration.txt"); /* OK if missing */

    /* Merge IL results */
    for (il_radix_t *ir = radixes; ir->R; ir++)
    {
        int have_split = (ir->tw_fwd_split != NULL);
        int have_il = (ir->tw_fwd_il != NULL) || (ir->tw_fwd_il_native != NULL);
        if (!have_split || !have_il)
            continue;

        cal.il_crossover[ir->R] = crossovers[ir->R];

        if (crossovers[ir->R] > 0)
            printf("  R=%-3zu  IL at K >= %zu\n", ir->R, crossovers[ir->R]);
        else
            printf("  R=%-3zu  IL never wins\n", ir->R);
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
                              vfft_l1d_bytes(), isa_name) == 0)
    {
        printf("\n  Calibration written to: vfft_calibration.txt\n");
    }
    else
    {
        printf("\n  ERROR: failed to write vfft_calibration.txt\n");
    }

    return 0;
}