/**
 * bench_r16_dif_pipeline.c — Pipeline-level A/B test for R=16 DIF vs DIT.
 *
 * For each case with R=16 in its factorization, measures:
 *   - All-DIT baseline (unmodified registry)
 *   - All-DIF (registry patched to swap R=16 t1 entries at runtime)
 *   - Roundtrip accuracy under both
 *
 * Reports ns/call for fwd and bwd, plus roundtrip error vs MKL reference.
 *
 * Skips planner re-exploration on purpose: we want to see the DIRECT swap
 * impact at the same factorization, so any speedup is attributable to the
 * codelet-level change alone.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../core/compat.h"
#include "../core/env.h"
#include "../core/planner.h"
#include "../codelets/avx2/fft_radix16_avx2_ct_t1_dif.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef VFFT_BENCH_DIR
  #define WISDOM_PATH VFFT_BENCH_DIR "/vfft_wisdom.txt"
#else
  #define WISDOM_PATH "vfft_wisdom.txt"
#endif

static double now_seconds(void) { return now_ns() / 1e9; }

/* Bench a plan forward-then-backward roundtrip over `reps` iterations.
 * Returns ns per (fwd+bwd) call. */
static double bench_roundtrip(stride_plan_t *plan, int N, size_t K, int reps,
                              double *out_maxerr)
{
    size_t total = (size_t)N * K;
    double *re = STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im = STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *re0 = STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im0 = STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));

    srand(42);
    for (size_t i = 0; i < total; i++) {
        re0[i] = (double)rand() / RAND_MAX - 0.5;
        im0[i] = (double)rand() / RAND_MAX - 0.5;
    }

    /* Warmup + correctness check */
    memcpy(re, re0, total * sizeof(double));
    memcpy(im, im0, total * sizeof(double));
    for (int w = 0; w < 10; w++) {
        stride_execute_fwd(plan, re, im);
        stride_execute_bwd(plan, re, im);
    }
    double maxerr = 0.0;
    for (size_t i = 0; i < total; i++) {
        double scale = (double)N * 10.0;  /* 10 roundtrips, each scales by N */
        double er = fabs(re[i] / scale - re0[i]);
        double ei = fabs(im[i] / scale - im0[i]);
        if (er > maxerr) maxerr = er;
        if (ei > maxerr) maxerr = ei;
    }
    *out_maxerr = maxerr;

    /* Timed loop */
    double best = 1e18;
    for (int trial = 0; trial < 7; trial++) {
        memcpy(re, re0, total * sizeof(double));
        memcpy(im, im0, total * sizeof(double));
        double t0 = now_seconds();
        for (int r = 0; r < reps; r++) {
            stride_execute_fwd(plan, re, im);
            stride_execute_bwd(plan, re, im);
        }
        double dt = now_seconds() - t0;
        double ns = dt * 1e9 / reps;
        if (ns < best) best = ns;
    }

    STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);
    STRIDE_ALIGNED_FREE(re0); STRIDE_ALIGNED_FREE(im0);
    return best;
}

static void patch_r16_to_dif(stride_registry_t *reg) {
    reg->t1_fwd[16] = (stride_t1_fn)radix16_t1_dif_fwd_avx2;
    reg->t1_bwd[16] = (stride_t1_fn)radix16_t1_dif_bwd_avx2;
}

static void run_case(int N, size_t K, const char *label) {
    printf("\n=== %s  N=%d K=%zu ===\n", label, N, K);

    /* Calibrate (or load) to get a consistent factorization across both runs */
    stride_wisdom_t wis;
    stride_wisdom_init(&wis);
    stride_wisdom_load(&wis, WISDOM_PATH);

    const stride_wisdom_entry_t *e = stride_wisdom_lookup(&wis, N, K);
    if (!e) {
        printf("  SKIP — not in wisdom\n");
        return;
    }
    int has_r16 = 0;
    printf("  factors: ");
    for (int s = 0; s < e->nfactors; s++) {
        printf("%s%d", s ? "x" : "", e->factors[s]);
        if (e->factors[s] == 16) has_r16 = 1;
    }
    printf("  %s\n", has_r16 ? "(has R=16)" : "(NO R=16 — DIF won't affect)");
    if (!has_r16) return;

    size_t total = (size_t)N * K;
    int reps = (int)(1e9 / ((double)total * 20.0));
    if (reps < 10) reps = 10;
    if (reps > 5000) reps = 5000;

    /* === Run 1: DIT baseline === */
    stride_registry_t reg_dit;
    stride_registry_init(&reg_dit);
    stride_plan_t *plan_dit = stride_wise_plan(N, K, &reg_dit, &wis);
    double err_dit = 0.0;
    double ns_dit = bench_roundtrip(plan_dit, N, K, reps, &err_dit);

    /* === Run 2: DIF swap === */
    stride_registry_t reg_dif;
    stride_registry_init(&reg_dif);
    patch_r16_to_dif(&reg_dif);
    stride_plan_t *plan_dif = stride_wise_plan(N, K, &reg_dif, &wis);
    double err_dif = 0.0;
    double ns_dif = bench_roundtrip(plan_dif, N, K, reps, &err_dif);

    printf("  DIT roundtrip: %10.0f ns/call   err=%.2e\n", ns_dit, err_dit);
    printf("  DIF roundtrip: %10.0f ns/call   err=%.2e   Δ=%+.2f%%\n",
           ns_dif, err_dif, 100.0 * (ns_dif - ns_dit) / ns_dit);
}

int main(void) {
    stride_env_init();
    stride_pin_thread(0);
    stride_set_num_threads(1);

    struct { int N; size_t K; } cases[] = {
        {4096,   4},
        {8192,   4},
        {16384,  4},
        {32768,  4},
        {4096,   32},
        {8192,   32},
        {16384,  32},
        {32768,  256},
        {4096,   256},
        {16384,  256},
    };
    int n = sizeof(cases) / sizeof(cases[0]);

    printf("Pipeline A/B: R=16 DIT vs DIF (forward+backward roundtrip)\n");
    printf("Same wisdom factorization used in both runs.\n");

    for (int i = 0; i < n; i++) {
        run_case(cases[i].N, cases[i].K, "case");
    }
    return 0;
}
