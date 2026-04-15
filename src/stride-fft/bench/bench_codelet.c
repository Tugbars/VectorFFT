/**
 * bench_codelet.c — Unified codelet microbenchmark for VTune / perf stat
 *
 * Runs isolated codelet variants for R=2, 4, 8, 16, 32 in tight loops.
 * Measures ns/call, CPE, GFLOP/s.
 *
 * Usage:
 *   vfft_bench_codelet [K] [duration_ms] [radix] [codelet_filter]
 *
 * Examples:
 *   vfft_bench_codelet 256                       # all radixes, all codelets
 *   vfft_bench_codelet 256 3000 4                # R=4 only
 *   vfft_bench_codelet 256 5000 8 t1_dit_fwd     # single codelet for VTune
 *
 * VTune:
 *   vtune -collect uarch-exploration -- vfft_bench_codelet 256 5000 8 t1_dit_fwd
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../core/compat.h"
#include "../core/env.h"
#include "../core/executor.h"  /* STRIDE_ALIGNED_ALLOC/FREE */
#include "../core/prefetch.h"

/* Include all pow2 codelet headers */
#include "../codelets/avx2/fft_radix2_avx2.h"
#include "../codelets/avx2/fft_radix4_avx2.h"
#include "../codelets/avx2/fft_radix8_avx2.h"
#include "../codelets/avx2/fft_radix16_avx2_ct_n1.h"
#include "../codelets/avx2/fft_radix16_avx2_ct_t1_dit.h"
#include "../codelets/avx2/fft_radix16_avx2_ct_t1_dif.h"
#include "../codelets/avx2/fft_radix10_avx2_ct_n1.h"
#include "../codelets/avx2/fft_radix10_avx2_ct_t1_dit.h"
#include "../codelets/avx2/fft_radix11_avx2_ct_n1.h"
#include "../codelets/avx2/fft_radix11_avx2_ct_t1_dit.h"
#include "../codelets/avx2/fft_radix12_avx2_ct_n1.h"
#include "../codelets/avx2/fft_radix12_avx2_ct_t1_dit.h"
#include "../codelets/avx2/fft_radix13_avx2_ct_n1.h"
#include "../codelets/avx2/fft_radix13_avx2_ct_t1_dit.h"
#include "../codelets/avx2/fft_radix20_avx2_ct_n1.h"
#include "../codelets/avx2/fft_radix20_avx2_ct_t1_dit.h"
#include "../codelets/avx2/fft_radix25_avx2_ct_n1.h"
#include "../codelets/avx2/fft_radix25_avx2_ct_t1_dit.h"
#include "../codelets/avx2/fft_radix32_avx2_ct_n1.h"
#include "../codelets/avx2/fft_radix32_avx2_ct_t1_dit.h"
#include "../codelets/avx2/fft_radix32_avx2_ct_t1_dif.h"
#include "../codelets/avx2/fft_radix64_avx2_ct_n1.h"
#include "../codelets/avx2/fft_radix64_avx2_ct_t1_dit.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── Frequency estimation ── */

static double estimate_ghz(void) {
    volatile double x = 1.0;
    double t0 = now_ns();
    for (int i = 0; i < 100000000; i++)
        x = x * 1.0000001;
    double ns = now_ns() - t0;
    return 100000000.0 * 4.0 / ns;
}
/* NOTE: est_freq from this function is unreliable on Raptor Lake — reads ~2.4 GHz
 * because the CPU doesn't sustain turbo through this short calibration loop.
 * Actual codelet runs are at ~5.7 GHz turbo (VTune confirmed). The CPE column
 * is approximate; ns/call is the source of truth. */

/* ── Globals for zero-overhead codelet calls ── */

static double *g_re, *g_im;
static double *g_tw_re, *g_tw_im;
static size_t g_stride, g_K;

/* ── R=2 wrappers ── */

static void r2_n1_fwd(void) {
    radix2_n1_fwd_avx2(g_re, g_im, g_re, g_im, g_stride, g_stride, g_K);
}
static void r2_n1_bwd(void) {
    radix2_n1_bwd_avx2(g_re, g_im, g_re, g_im, g_stride, g_stride, g_K);
}
static void r2_t1_dit_fwd(void) {
    radix2_t1_dit_fwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}
static void r2_t1_dit_bwd(void) {
    radix2_t1_dit_bwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}
static void r2_t1_dif_fwd(void) {
    radix2_t1_dif_fwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}
static void r2_t1_dif_bwd(void) {
    radix2_t1_dif_bwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}

/* ── R=4 wrappers ── */

static void r4_n1_fwd(void) {
    radix4_n1_fwd_avx2(g_re, g_im, g_re, g_im, g_stride, g_stride, g_K);
}
static void r4_n1_bwd(void) {
    radix4_n1_bwd_avx2(g_re, g_im, g_re, g_im, g_stride, g_stride, g_K);
}
static void r4_t1_dit_fwd(void) {
    radix4_t1_dit_fwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}
static void r4_t1_dit_bwd(void) {
    radix4_t1_dit_bwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}
static void r4_t1_dif_fwd(void) {
    radix4_t1_dif_fwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}
static void r4_t1_dif_bwd(void) {
    radix4_t1_dif_bwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}

/* ── R=8 wrappers ── */

static void r8_n1_fwd(void) {
    radix8_n1_fwd_avx2(g_re, g_im, g_re, g_im, g_stride, g_stride, g_K);
}
static void r8_n1_bwd(void) {
    radix8_n1_bwd_avx2(g_re, g_im, g_re, g_im, g_stride, g_stride, g_K);
}
static void r8_t1_dit_fwd(void) {
    radix8_t1_dit_fwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}
static void r8_t1_dit_bwd(void) {
    radix8_t1_dit_bwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}
static void r8_t1_dif_fwd(void) {
    radix8_t1_dif_fwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}
static void r8_t1_dif_bwd(void) {
    radix8_t1_dif_bwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}

/* ── R=16 wrappers ── */

static void r16_n1_fwd(void) {
    radix16_n1_fwd_avx2(g_re, g_im, g_re, g_im, g_stride, g_stride, g_K);
}
static void r16_n1_bwd(void) {
    radix16_n1_bwd_avx2(g_re, g_im, g_re, g_im, g_stride, g_stride, g_K);
}
static void r16_t1_dit_fwd(void) {
    radix16_t1_dit_fwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}
static void r16_t1_dit_bwd(void) {
    radix16_t1_dit_bwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}
static void r16_t1_dif_fwd(void) {
    radix16_t1_dif_fwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}
static void r16_t1_dif_bwd(void) {
    radix16_t1_dif_bwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}

/* ── R=10 wrappers ── */

static void r10_n1_fwd(void) {
    radix10_n1_fwd_avx2(g_re, g_im, g_re, g_im, g_stride, g_stride, g_K);
}
static void r10_n1_bwd(void) {
    radix10_n1_bwd_avx2(g_re, g_im, g_re, g_im, g_stride, g_stride, g_K);
}
static void r10_t1_dit_fwd(void) {
    radix10_t1_dit_fwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}
static void r10_t1_dit_bwd(void) {
    radix10_t1_dit_bwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}

/* ── R=11 wrappers (genfft prime) ── */

static void r11_n1_fwd(void) {
    radix11_n1_fwd_avx2(g_re, g_im, g_re, g_im, g_stride, g_stride, g_K);
}
static void r11_n1_bwd(void) {
    radix11_n1_bwd_avx2(g_re, g_im, g_re, g_im, g_stride, g_stride, g_K);
}
static void r11_t1_dit_fwd(void) {
    radix11_t1_dit_fwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}
static void r11_t1_dit_bwd(void) {
    radix11_t1_dit_bwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}

/* ── R=12 wrappers ── */

static void r12_n1_fwd(void) {
    radix12_n1_fwd_avx2(g_re, g_im, g_re, g_im, g_stride, g_stride, g_K);
}
static void r12_n1_bwd(void) {
    radix12_n1_bwd_avx2(g_re, g_im, g_re, g_im, g_stride, g_stride, g_K);
}
static void r12_t1_dit_fwd(void) {
    radix12_t1_dit_fwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}
static void r12_t1_dit_bwd(void) {
    radix12_t1_dit_bwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}

/* ── R=13 wrappers (genfft prime) ── */

static void r13_n1_fwd(void) {
    radix13_n1_fwd_avx2(g_re, g_im, g_re, g_im, g_stride, g_stride, g_K);
}
static void r13_n1_bwd(void) {
    radix13_n1_bwd_avx2(g_re, g_im, g_re, g_im, g_stride, g_stride, g_K);
}
static void r13_t1_dit_fwd(void) {
    radix13_t1_dit_fwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}
static void r13_t1_dit_bwd(void) {
    radix13_t1_dit_bwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}

/* ── R=20 wrappers ── */

static void r20_n1_fwd(void) {
    radix20_n1_fwd_avx2(g_re, g_im, g_re, g_im, g_stride, g_stride, g_K);
}
static void r20_n1_bwd(void) {
    radix20_n1_bwd_avx2(g_re, g_im, g_re, g_im, g_stride, g_stride, g_K);
}
static void r20_t1_dit_fwd(void) {
    radix20_t1_dit_fwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}
static void r20_t1_dit_bwd(void) {
    radix20_t1_dit_bwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}

/* ── R=25 wrappers ── */

static void r25_n1_fwd(void) {
    radix25_n1_fwd_avx2(g_re, g_im, g_re, g_im, g_stride, g_stride, g_K);
}
static void r25_n1_bwd(void) {
    radix25_n1_bwd_avx2(g_re, g_im, g_re, g_im, g_stride, g_stride, g_K);
}
static void r25_t1_dit_fwd(void) {
    radix25_t1_dit_fwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}
static void r25_t1_dit_bwd(void) {
    radix25_t1_dit_bwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}

/* ── R=32 wrappers ── */

static void r32_n1_fwd(void) {
    radix32_n1_fwd_avx2(g_re, g_im, g_re, g_im, g_stride, g_stride, g_K);
}
static void r32_n1_bwd(void) {
    radix32_n1_bwd_avx2(g_re, g_im, g_re, g_im, g_stride, g_stride, g_K);
}
static void r32_t1_dit_fwd(void) {
    radix32_t1_dit_fwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}
static void r32_t1_dit_bwd(void) {
    radix32_t1_dit_bwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}
static void r32_t1_dif_fwd(void) {
    radix32_t1_dif_fwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}
static void r32_t1_dif_bwd(void) {
    radix32_t1_dif_bwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}

/* ── R=64 wrappers ── */

static void r64_n1_fwd(void) {
    radix64_n1_fwd_avx2(g_re, g_im, g_re, g_im, g_stride, g_stride, g_K);
}
static void r64_n1_bwd(void) {
    radix64_n1_bwd_avx2(g_re, g_im, g_re, g_im, g_stride, g_stride, g_K);
}
static void r64_t1_dit_fwd(void) {
    radix64_t1_dit_fwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}
static void r64_t1_dit_bwd(void) {
    radix64_t1_dit_bwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}

/* ── Benchmark harness ── */

static double bench_one(void (*body)(void), int reps) {
    double best = 1e18;
    for (int trial = 0; trial < 5; trial++) {
        double t0 = now_ns();
        for (int r = 0; r < reps; r++)
            body();
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    return best;
}

/* ── Codelet descriptor ── */

typedef struct {
    const char *name;
    int radix;
    void (*fn)(void);
    int flops_per_K;  /* total FLOPs = flops_per_K * K  (FMA counted as 2) */
} codelet_entry_t;

/*
 * FLOPs per K (FMA=2), from header op-count tables:
 *   R=2: notw  4, t1  10
 *   R=4: notw 16, t1  34
 *   R=8: notw 52, t1  87
 * R=16/32: estimated from 5*R*log2(R) scaling
 *   R=16: notw ~136, t1 ~210
 *   R=32: notw ~340, t1 ~520
 */

static const codelet_entry_t g_tests[] = {
    /* R=2 */
    {"R2  n1_fwd (notw)",  2,  r2_n1_fwd,      4},
    {"R2  n1_bwd (notw)",  2,  r2_n1_bwd,      4},
    {"R2  t1_dit_fwd",     2,  r2_t1_dit_fwd, 10},
    {"R2  t1_dit_bwd",     2,  r2_t1_dit_bwd, 10},
    {"R2  t1_dif_fwd",     2,  r2_t1_dif_fwd, 10},
    {"R2  t1_dif_bwd",     2,  r2_t1_dif_bwd, 10},

    /* R=4 */
    {"R4  n1_fwd (notw)",  4,  r4_n1_fwd,     16},
    {"R4  n1_bwd (notw)",  4,  r4_n1_bwd,     16},
    {"R4  t1_dit_fwd",     4,  r4_t1_dit_fwd, 34},
    {"R4  t1_dit_bwd",     4,  r4_t1_dit_bwd, 34},
    {"R4  t1_dif_fwd",     4,  r4_t1_dif_fwd, 34},
    {"R4  t1_dif_bwd",     4,  r4_t1_dif_bwd, 34},

    /* R=8 */
    {"R8  n1_fwd (notw)",  8,  r8_n1_fwd,     52},
    {"R8  n1_bwd (notw)",  8,  r8_n1_bwd,     52},
    {"R8  t1_dit_fwd",     8,  r8_t1_dit_fwd, 87},
    {"R8  t1_dit_bwd",     8,  r8_t1_dit_bwd, 87},
    {"R8  t1_dif_fwd",     8,  r8_t1_dif_fwd, 87},
    {"R8  t1_dif_bwd",     8,  r8_t1_dif_bwd, 87},

    /* R=16 (no DIF variant) */
    {"R16 n1_fwd (notw)", 16, r16_n1_fwd,    136},
    {"R16 n1_bwd (notw)", 16, r16_n1_bwd,    136},
    {"R16 t1_dit_fwd",    16, r16_t1_dit_fwd,210},
    {"R16 t1_dit_bwd",    16, r16_t1_dit_bwd,210},
    {"R16 t1_dif_fwd",    16, r16_t1_dif_fwd,210},
    {"R16 t1_dif_bwd",    16, r16_t1_dif_bwd,210},

    /* R=10 (composite 2x5, no DIF variant) */
    {"R10 n1_fwd (notw)", 10, r10_n1_fwd,      80},
    {"R10 n1_bwd (notw)", 10, r10_n1_bwd,      80},
    {"R10 t1_dit_fwd",    10, r10_t1_dit_fwd, 130},
    {"R10 t1_dit_bwd",    10, r10_t1_dit_bwd, 130},

    /* R=11 (genfft prime, Winograd-style, no DIF variant) */
    {"R11 n1_fwd (notw)", 11, r11_n1_fwd,      88},
    {"R11 n1_bwd (notw)", 11, r11_n1_bwd,      88},
    {"R11 t1_dit_fwd",    11, r11_t1_dit_fwd, 140},
    {"R11 t1_dit_bwd",    11, r11_t1_dit_bwd, 140},

    /* R=12 (composite 4x3, no DIF variant) */
    {"R12 n1_fwd (notw)", 12, r12_n1_fwd,      96},
    {"R12 n1_bwd (notw)", 12, r12_n1_bwd,      96},
    {"R12 t1_dit_fwd",    12, r12_t1_dit_fwd, 160},
    {"R12 t1_dit_bwd",    12, r12_t1_dit_bwd, 160},

    /* R=13 (genfft prime, Winograd-style, no DIF variant) */
    {"R13 n1_fwd (notw)", 13, r13_n1_fwd,     110},
    {"R13 n1_bwd (notw)", 13, r13_n1_bwd,     110},
    {"R13 t1_dit_fwd",    13, r13_t1_dit_fwd, 175},
    {"R13 t1_dit_bwd",    13, r13_t1_dit_bwd, 175},

    /* R=20 (composite 4x5, no DIF variant) */
    {"R20 n1_fwd (notw)", 20, r20_n1_fwd,     180},
    {"R20 n1_bwd (notw)", 20, r20_n1_bwd,     180},
    {"R20 t1_dit_fwd",    20, r20_t1_dit_fwd, 280},
    {"R20 t1_dit_bwd",    20, r20_t1_dit_bwd, 280},

    /* R=25 (composite 5x5, no DIF variant) */
    {"R25 n1_fwd (notw)", 25, r25_n1_fwd,     240},
    {"R25 n1_bwd (notw)", 25, r25_n1_bwd,     240},
    {"R25 t1_dit_fwd",    25, r25_t1_dit_fwd, 380},
    {"R25 t1_dit_bwd",    25, r25_t1_dit_bwd, 380},

    /* R=32 (no DIF variant) */
    {"R32 n1_fwd (notw)", 32, r32_n1_fwd,    340},
    {"R32 n1_bwd (notw)", 32, r32_n1_bwd,    340},
    {"R32 t1_dit_fwd",    32, r32_t1_dit_fwd,520},
    {"R32 t1_dit_bwd",    32, r32_t1_dit_bwd,520},
    {"R32 t1_dif_fwd",    32, r32_t1_dif_fwd,520},
    {"R32 t1_dif_bwd",    32, r32_t1_dif_bwd,520},

    /* R=64 (no DIF variant) */
    {"R64 n1_fwd (notw)", 64, r64_n1_fwd,     816},
    {"R64 n1_bwd (notw)", 64, r64_n1_bwd,     816},
    {"R64 t1_dit_fwd",    64, r64_t1_dit_fwd, 1260},
    {"R64 t1_dit_bwd",    64, r64_t1_dit_bwd, 1260},
};

#define N_TESTS (int)(sizeof(g_tests) / sizeof(g_tests[0]))

int main(int argc, char **argv) {
    stride_env_init();
    stride_pin_thread(0);

    size_t K = 256;
    int duration_ms = 3000;
    int radix_filter = 0;           /* 0 = all */
    const char *codelet_filter = NULL;

    if (argc > 1) K = (size_t)atoi(argv[1]);
    if (argc > 2) duration_ms = atoi(argv[2]);
    if (argc > 3) radix_filter = atoi(argv[3]);
    if (argc > 4) codelet_filter = argv[4];

    double ghz = estimate_ghz();

    printf("=== Codelet Microbenchmark (R=2,4,8,16,32) ===\n");
    printf("K=%zu  duration=%dms/codelet  est_freq=%.2f GHz\n",
           K, duration_ms, ghz);
    if (radix_filter) printf("radix_filter=R%d\n", radix_filter);
    if (codelet_filter) printf("codelet_filter=%s\n", codelet_filter);
    printf("\n");

    /* Allocate for max R=64 */
    size_t max_R = 64;
    size_t max_elem = max_R * K;
    size_t max_tw   = (max_R - 1) * K;

    g_K = K;
    g_stride = K;
    g_re    = (double *)STRIDE_ALIGNED_ALLOC(64, max_elem * sizeof(double));
    g_im    = (double *)STRIDE_ALIGNED_ALLOC(64, max_elem * sizeof(double));
    g_tw_re = (double *)STRIDE_ALIGNED_ALLOC(64, max_tw * sizeof(double));
    g_tw_im = (double *)STRIDE_ALIGNED_ALLOC(64, max_tw * sizeof(double));

    /* Fill data */
    srand(42);
    for (size_t i = 0; i < max_elem; i++) {
        g_re[i] = (double)rand() / RAND_MAX - 0.5;
        g_im[i] = (double)rand() / RAND_MAX - 0.5;
    }
    for (size_t i = 0; i < max_tw; i++) {
        double angle = -2.0 * M_PI * (double)(i % K) / (double)(max_R * K);
        g_tw_re[i] = cos(angle);
        g_tw_im[i] = sin(angle);
    }

    printf("%-22s %4s %10s %8s %10s %10s\n",
           "Codelet", "R", "ns/call", "CPE", "GFLOP/s", "reps");
    printf("────────────────────── ──── ────────── ──────── ────────── ──────────\n");

    int prev_radix = 0;

    for (int ti = 0; ti < N_TESTS; ti++) {
        int R = g_tests[ti].radix;

        /* Radix filter */
        if (radix_filter && R != radix_filter)
            continue;

        /* Codelet name filter */
        if (codelet_filter && !strstr(g_tests[ti].name, codelet_filter))
            continue;

        /* Section separator */
        if (R != prev_radix) {
            if (prev_radix) printf("\n");
            prev_radix = R;
        }

        size_t total_elem = (size_t)R * K;

        /* Re-randomize data region for this radix */
        for (size_t i = 0; i < total_elem; i++) {
            g_re[i] = (double)rand() / RAND_MAX - 0.5;
            g_im[i] = (double)rand() / RAND_MAX - 0.5;
        }

        /* Regenerate twiddles for this radix */
        size_t n_tw = (size_t)(R - 1) * K;
        for (size_t i = 0; i < n_tw; i++) {
            double angle = -2.0 * M_PI * (double)(i % K) / (double)((size_t)R * K);
            g_tw_re[i] = cos(angle);
            g_tw_im[i] = sin(angle);
        }

        /* Warmup */
        for (int w = 0; w < 5000; w++)
            g_tests[ti].fn();

        /* Calibrate reps */
        int reps = 10000;
        double t0 = now_ns();
        for (int r = 0; r < reps; r++)
            g_tests[ti].fn();
        double calib_ns = (now_ns() - t0) / reps;
        reps = (int)((double)duration_ms * 1e6 / calib_ns);
        if (reps < 10000) reps = 10000;

        /* Timed run */
        double best_ns = bench_one(g_tests[ti].fn, reps);

        double cpe = best_ns * ghz / (double)total_elem;
        int total_flops = g_tests[ti].flops_per_K * (int)K;
        double gf = (double)total_flops / best_ns;

        printf("%-22s %4d %9.1f %7.3f %9.2f %10d\n",
               g_tests[ti].name, R, best_ns, cpe, gf, reps);
    }

    printf("\n");
    printf("CPE = cycles / (R * K) = cycles per element\n");
    printf("FMA counted as 2 FLOPs\n");
    printf("\nUsage: vfft_bench_codelet [K] [duration_ms] [radix] [codelet_filter]\n");
    printf("  K=256              L2-resident\n");
    printf("  K=32               L1-resident\n");
    printf("  radix: 2,4,8,16,32 or 0 for all\n");
    printf("  codelet_filter: substring match (n1, t1_dit_fwd, etc.)\n");
    printf("\nVTune:\n");
    printf("  vtune -collect uarch-exploration -- vfft_bench_codelet 256 5000 8 t1_dit_fwd\n");

    STRIDE_ALIGNED_FREE(g_re);
    STRIDE_ALIGNED_FREE(g_im);
    STRIDE_ALIGNED_FREE(g_tw_re);
    STRIDE_ALIGNED_FREE(g_tw_im);

    return 0;
}
