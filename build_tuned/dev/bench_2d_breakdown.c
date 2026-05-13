/* bench_2d_breakdown.c — attribute 2D FFT time to phases.
 *
 * Goal: confirm (or refute) the 33-day-old claim in 2d_fft_strategy.md
 * that gather/scatter is ~10% of total 2D time. If yes, strided-batch
 * codelets (which eliminate gather/scatter) have ~10% headroom to
 * recover; if no, the picture has changed and the strided-batch project
 * needs requalification.
 *
 * For each (N1, N2):
 *   total       = stride_execute_fwd(plan2d)                       — full 2D
 *   col_only    = stride_execute_fwd(plan_col)                      — phase 1
 *   row_total   = total - col_only                                  — phase 2 sum
 *   tile_gs     = stride_transpose_pair × (N1/B) × 2                — pure gather+scatter
 *   tile_fft    = stride_execute_fwd(plan_row, scratch) × (N1/B)    — pure inner FFT
 *
 * tile_gs + tile_fft should ≈ row_total. The fraction tile_gs/total is
 * what strided-batch can theoretically claw back.
 *
 * Also reports MKL 2D forward as a sanity check that we're still in
 * the same ballpark as before.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

#include "planner.h"
#include "fft2d.h"
#include "executor.h"
#include "transpose.h"
#include "env.h"

#ifdef VFFT_HAS_MKL
#include "mkl_dfti.h"
#endif

static double bk_now_ns(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
}

/* Run `fn(arg)` `reps` times across `trials` trials, return best ns. */
static double bench_best(void (*fn)(void *), void *arg, int reps, int trials) {
    /* Warmup */
    for (int i = 0; i < 50; i++) fn(arg);
    double best = 1e18;
    for (int t = 0; t < trials; t++) {
        double t0 = bk_now_ns();
        for (int r = 0; r < reps; r++) fn(arg);
        double dt = (bk_now_ns() - t0) / (double)reps;
        if (dt < best) best = dt;
    }
    return best;
}

/* Adapter: 2D fwd via opaque plan. */
typedef struct { stride_plan_t *plan; double *re, *im; } fn_2d_t;
static void run_2d(void *p) {
    fn_2d_t *a = (fn_2d_t *)p;
    stride_execute_fwd(a->plan, a->re, a->im);
}

/* Adapter: column-only (phase 1 alone). */
typedef struct { stride_plan_t *plan_col; double *re, *im; } fn_col_t;
static void run_col(void *p) {
    fn_col_t *a = (fn_col_t *)p;
    stride_execute_fwd(a->plan_col, a->re, a->im);
}

/* Adapter: gather + scatter for ALL tiles (no inner FFT). */
typedef struct {
    int N1, N2;
    size_t B;
    double *re, *im;
    double *sr, *si;
} fn_gs_t;
static void run_gather_scatter(void *p) {
    fn_gs_t *a = (fn_gs_t *)p;
    int N1 = a->N1, N2 = a->N2;
    size_t B = a->B;
    for (size_t i = 0; i < (size_t)N1; i += B) {
        size_t this_B = B;
        if (i + B > (size_t)N1) this_B = (size_t)N1 - i;
        stride_transpose_pair(
            a->re + i * N2, a->im + i * N2, a->sr, a->si,
            (size_t)N2, B, this_B, (size_t)N2);
        stride_transpose_pair(
            a->sr, a->si, a->re + i * N2, a->im + i * N2,
            B, (size_t)N2, (size_t)N2, this_B);
    }
}

/* Adapter: inner row FFT for ALL tiles (no gather/scatter — scratch
 * already populated, run plan_row on it for each tile). */
typedef struct {
    int N1, N2;
    size_t B;
    stride_plan_t *plan_row;
    double *sr, *si;
} fn_innerfft_t;
static void run_inner_fft(void *p) {
    fn_innerfft_t *a = (fn_innerfft_t *)p;
    int N1 = a->N1;
    size_t B = a->B;
    for (size_t i = 0; i < (size_t)N1; i += B) {
        size_t this_B = B;
        if (i + B > (size_t)N1) this_B = (size_t)N1 - i;
        _stride_execute_fwd_slice(a->plan_row, a->sr, a->si, this_B, B);
    }
}

#ifdef VFFT_HAS_MKL
typedef struct { DFTI_DESCRIPTOR_HANDLE h; double *re, *im; } fn_mkl_t;
static void run_mkl(void *p) {
    fn_mkl_t *a = (fn_mkl_t *)p;
    DftiComputeForward(a->h, a->re, a->im);
}
#endif

static void run_one(int N1, int N2, stride_registry_t *reg, stride_wisdom_t *wis) {
    size_t NK = (size_t)N1 * N2;
    double *re = (double *)_aligned_malloc(NK * sizeof(double), 64);
    double *im = (double *)_aligned_malloc(NK * sizeof(double), 64);
    /* Per-thread B-tile scratch (single-threaded bench: one tile) */
    size_t B = FFT2D_DEFAULT_TILE;
    double *sr = (double *)_aligned_malloc(B * (size_t)N2 * sizeof(double), 64);
    double *si = (double *)_aligned_malloc(B * (size_t)N2 * sizeof(double), 64);

    srand(13 + N1 * 100 + N2);
    for (size_t i = 0; i < NK; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }
    memset(sr, 0, B * (size_t)N2 * sizeof(double));
    memset(si, 0, B * (size_t)N2 * sizeof(double));

    stride_set_num_threads(1);  /* single-threaded attribution */
    stride_plan_t *plan2d = stride_plan_2d_wise(N1, N2, reg, wis);
    if (!plan2d) {
        printf("  N1=%-5d N2=%-5d  plan2d failed\n", N1, N2);
        goto done;
    }

    /* Get the inner column + row plans out of the 2D wrapper. plan2d's
     * override_data field points to stride_fft2d_data_t (2D goes
     * through the override dispatch, not the standard staged loop). */
    stride_fft2d_data_t *d = (stride_fft2d_data_t *)plan2d->override_data;
    if (d->use_bailey) {
        printf("  N1=%-5d N2=%-5d  bailey path (no tile breakdown)\n", N1, N2);
        goto cleanup;
    }
    int reps   = NK <= 64*1024 ? 200 : (NK <= 256*1024 ? 50 : 20);
    int trials = 5;

    fn_2d_t       a_2d  = { plan2d, re, im };
    fn_col_t      a_col = { d->plan_col, re, im };
    fn_gs_t       a_gs  = { N1, N2, d->B, re, im, sr, si };
    fn_innerfft_t a_in  = { N1, N2, d->B, d->plan_row, sr, si };

    double t_total = bench_best(run_2d,  &a_2d,  reps, trials);
    double t_col   = bench_best(run_col, &a_col, reps, trials);
    double t_gs    = bench_best(run_gather_scatter, &a_gs, reps, trials);
    double t_in    = bench_best(run_inner_fft,     &a_in, reps, trials);
    double t_row   = t_total - t_col;
    double t_mkl = 0.0;

#ifdef VFFT_HAS_MKL
    DFTI_DESCRIPTOR_HANDLE h = NULL;
    MKL_LONG dims[2] = { N1, N2 };
    DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 2, dims);
    DftiSetValue(h, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
    DftiSetValue(h, DFTI_PLACEMENT, DFTI_INPLACE);
    DftiCommitDescriptor(h);
    fn_mkl_t a_mkl = { h, re, im };
    t_mkl = bench_best(run_mkl, &a_mkl, reps, trials);
    DftiFreeDescriptor(&h);
#endif

    double gs_pct  = 100.0 * t_gs / t_total;
    double in_pct  = 100.0 * t_in / t_total;
    double col_pct = 100.0 * t_col / t_total;
    double row_pct = 100.0 * t_row / t_total;
    double ratio_mkl = t_mkl > 0 ? t_total / t_mkl : 0.0;

    printf("N1=%-5d N2=%-5d  B=%zu  total=%8.1f µs  col=%5.1f%%  row=%5.1f%%  "
           "[gather/scatter=%5.1f%%  innerFFT=%5.1f%%]",
           N1, N2, d->B, t_total/1000.0, col_pct, row_pct, gs_pct, in_pct);
    if (t_mkl > 0) printf("  vs MKL=%6.1f µs  ratio=%5.2fx",
                          t_mkl/1000.0, ratio_mkl);
    printf("\n");

cleanup:
    stride_plan_destroy(plan2d);
done:
    _aligned_free(re); _aligned_free(im);
    _aligned_free(sr); _aligned_free(si);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("================================================================\n");
    printf("  2D FFT breakdown — single-threaded\n");
    printf("  Goal: confirm gather/scatter is still ~10%% of total time\n");
    printf("  (per 2d_fft_strategy.md, 2026-04-09)\n");
    printf("================================================================\n");
    printf("  col   = phase 1 column FFTs alone\n");
    printf("  row   = total - col (everything in phase 2)\n");
    printf("  gs    = gather+scatter only (no inner FFT)\n");
    printf("  in    = inner row FFT only (no gather+scatter)\n");
    printf("  gs+in ≈ row (modulo loop overhead)\n");
    printf("================================================================\n");

    stride_env_init();
    stride_registry_t reg;
    stride_registry_init(&reg);
    stride_wisdom_t   wis;
    stride_wisdom_init(&wis);
    stride_wisdom_load(&wis, "vfft_wisdom_tuned.txt");

    int sizes[][2] = {
        {32, 32}, {64, 64}, {128, 128}, {256, 256}, {512, 512}, {1024, 1024},
        {64, 128}, {128, 256}, {256, 512},
        /* Cells that exercise Design C strided codelet (N2 ∈ {16, 32, 64}): */
        {16, 16}, {32, 16}, {64, 16}, {128, 16}, {256, 16},
        {64, 32}, {128, 32}, {256, 32}, {512, 32},
        {128, 64}, {256, 64}, {512, 64}, {1024, 64},
    };
    for (size_t i = 0; i < sizeof(sizes)/sizeof(sizes[0]); i++) {
        run_one(sizes[i][0], sizes[i][1], &reg, &wis);
    }
    return 0;
}
