/**
 * bench_r4_codelet.c — Isolated R=4 codelet microbenchmark
 *
 * Runs each R=4 codelet variant in a tight loop to measure:
 *   - Cycles per element (CPE)
 *   - GFLOP/s
 *   - ns per call
 *
 * Use with VTune or perf stat for microarchitectural analysis:
 *   vtune -collect uarch-exploration -- vfft_bench_r4 [K] [duration_ms]
 *
 * Each codelet runs >99% of execution time — counter attribution is clean.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../core/compat.h"
#include "../core/env.h"
#include "../core/executor.h"  /* for STRIDE_ALIGNED_ALLOC/FREE */

/* Include the R=4 codelets directly */
#include "../codelets/avx2/fft_radix4_avx2.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double estimate_ghz(void) {
    volatile double x = 1.0;
    double t0 = now_ns();
    for (int i = 0; i < 100000000; i++)
        x = x * 1.0000001;
    double ns = now_ns() - t0;
    return 100000000.0 * 4.0 / ns;
}

/* ── Benchmark one codelet variant ── */

typedef struct {
    const char *name;
    int flops_per_elem;  /* FLOPs per R*K elements (FMA=2) */
} variant_t;

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

/* ── Globals for codelet calls (avoid function pointer overhead) ── */
static double *g_re, *g_im;
static double *g_tw_re, *g_tw_im;
static size_t g_stride, g_K;

static void call_n1_fwd(void) {
    radix4_n1_fwd_avx2(g_re, g_im, g_re, g_im, g_stride, g_stride, g_K);
}
static void call_n1_bwd(void) {
    radix4_n1_bwd_avx2(g_re, g_im, g_re, g_im, g_stride, g_stride, g_K);
}
static void call_t1_dit_fwd(void) {
    radix4_t1_dit_fwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}
static void call_t1_dit_bwd(void) {
    radix4_t1_dit_bwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}
static void call_t1_dif_fwd(void) {
    radix4_t1_dif_fwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}
static void call_t1_dif_bwd(void) {
    radix4_t1_dif_bwd_avx2(g_re, g_im, g_tw_re, g_tw_im, g_stride, g_K);
}

int main(int argc, char **argv) {
    stride_env_init();
    stride_pin_thread(0);

    size_t K = 256;
    int duration_ms = 3000;
    const char *only_codelet = NULL;  /* NULL = run all, else filter by name */

    if (argc > 1) K = (size_t)atoi(argv[1]);
    if (argc > 2) duration_ms = atoi(argv[2]);
    if (argc > 3) only_codelet = argv[3];  /* e.g. "t1_dit_fwd" */

    double ghz = estimate_ghz();
    size_t R = 4;
    size_t total_elem = R * K;

    printf("=== R=4 Codelet Microbenchmark ===\n");
    printf("K=%zu  R=%zu  total=%zu elements  duration=%dms/codelet  est_freq=%.2f GHz\n\n",
           K, R, total_elem, duration_ms, ghz);

    /* Allocate */
    g_stride = K;
    g_K = K;
    g_re    = (double *)STRIDE_ALIGNED_ALLOC(64, total_elem * sizeof(double));
    g_im    = (double *)STRIDE_ALIGNED_ALLOC(64, total_elem * sizeof(double));
    g_tw_re = (double *)STRIDE_ALIGNED_ALLOC(64, (R - 1) * K * sizeof(double));
    g_tw_im = (double *)STRIDE_ALIGNED_ALLOC(64, (R - 1) * K * sizeof(double));

    srand(42);
    for (size_t i = 0; i < total_elem; i++) {
        g_re[i] = (double)rand() / RAND_MAX - 0.5;
        g_im[i] = (double)rand() / RAND_MAX - 0.5;
    }
    for (size_t i = 0; i < (R - 1) * K; i++) {
        double angle = -2.0 * M_PI * (double)(i % K) / (double)(R * K);
        g_tw_re[i] = cos(angle);
        g_tw_im[i] = sin(angle);
    }

    /* Codelet table */
    struct {
        const char *name;
        void (*fn)(void);
        int flops_per_call;  /* FMA counted as 2 FLOPs */
    } tests[] = {
        {"n1_fwd (notw)",  call_n1_fwd,     16 * (int)K},
        {"n1_bwd (notw)",  call_n1_bwd,     16 * (int)K},
        {"t1_dit_fwd",     call_t1_dit_fwd, 34 * (int)K},
        {"t1_dit_bwd",     call_t1_dit_bwd, 34 * (int)K},
        {"t1_dif_fwd",     call_t1_dif_fwd, 34 * (int)K},
        {"t1_dif_bwd",     call_t1_dif_bwd, 34 * (int)K},
    };
    int ntests = sizeof(tests) / sizeof(tests[0]);

    printf("%-20s %10s %8s %10s %10s %10s\n",
           "Codelet", "ns/call", "CPE", "GFLOP/s", "FLOPs/call", "reps");
    printf("──────────────────── ────────── ──────── ────────── ────────── ──────────\n");

    for (int ti = 0; ti < ntests; ti++) {
        /* Filter: if --codelet specified, skip non-matching */
        if (only_codelet && !strstr(tests[ti].name, only_codelet))
            continue;

        /* Re-randomize */
        for (size_t i = 0; i < total_elem; i++) {
            g_re[i] = (double)rand() / RAND_MAX - 0.5;
            g_im[i] = (double)rand() / RAND_MAX - 0.5;
        }

        /* Warmup */
        for (int w = 0; w < 5000; w++)
            tests[ti].fn();

        /* Calibrate reps */
        int reps = 10000;
        double t0 = now_ns();
        for (int r = 0; r < reps; r++)
            tests[ti].fn();
        double calib_ns = (now_ns() - t0) / reps;
        reps = (int)((double)duration_ms * 1e6 / calib_ns);
        if (reps < 10000) reps = 10000;

        /* Timed run */
        double best_ns = bench_one(tests[ti].fn, reps);

        double cpe = best_ns * ghz / (double)total_elem;
        double gf = (double)tests[ti].flops_per_call / best_ns;

        printf("%-20s %9.1f %7.3f %9.2f %10d %10d\n",
               tests[ti].name, best_ns, cpe, gf, tests[ti].flops_per_call, reps);
    }

    printf("\n");
    printf("CPE = cycles / (R * K) = cycles per element\n");
    printf("FMA counted as 2 FLOPs\n");
    printf("\nUsage: vfft_bench_r4 [K] [duration_ms] [codelet_filter]\n");
    printf("\nSweep K to see cache effects:\n");
    printf("  vfft_bench_r4 32             # L1-resident, all codelets\n");
    printf("  vfft_bench_r4 256            # L2-resident, all codelets\n");
    printf("  vfft_bench_r4 256 3000 t1_dit_fwd   # single codelet for VTune\n");
    printf("\nVTune:\n");
    printf("  vtune -collect uarch-exploration -- vfft_bench_r4 256 5000 t1_dit_fwd\n");

    STRIDE_ALIGNED_FREE(g_re);
    STRIDE_ALIGNED_FREE(g_im);
    STRIDE_ALIGNED_FREE(g_tw_re);
    STRIDE_ALIGNED_FREE(g_tw_im);

    return 0;
}
