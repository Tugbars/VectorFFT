/**
 * bench_2d_csv.c — Comprehensive 2D FFT benchmark with CSV output
 *
 * Phase 1: Load wisdom, calibrate missing sub-plan entries, save back.
 * Phase 2: Benchmark VectorFFT 2D vs MKL, GFLOP/s + roundtrip error.
 *
 * Outputs:
 *   vfft_perf_2d.csv  — performance + accuracy
 *   vfft_wisdom.txt   — updated wisdom (shared)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../core/compat.h"
#include "../core/env.h"
#include "../core/planner.h"

#ifdef VFFT_HAS_MKL
#include <mkl_dfti.h>
#include <mkl_service.h>
#endif

#ifdef VFFT_BENCH_DIR
  #define WISDOM_PATH VFFT_BENCH_DIR "/vfft_wisdom.txt"
#else
  #define WISDOM_PATH "vfft_wisdom.txt"
#endif

static double gflops_2d(int N1, int N2, double ns) {
    if (ns <= 0) return 0;
    double N = (double)N1 * N2;
    return 5.0 * N * (log2((double)N1) + log2((double)N2)) / ns;
}

static double bench_vfft_2d(stride_plan_t *plan, size_t total) {
    double *re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    for (size_t i = 0; i < total; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }
    for (int w = 0; w < 20; w++)
        stride_execute_fwd(plan, re, im);
    int reps = (int)(2e6 / (total + 1));
    if (reps < 20) reps = 20;
    if (reps > 50000) reps = 50000;
    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++)
            stride_execute_fwd(plan, re, im);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    STRIDE_ALIGNED_FREE(re);
    STRIDE_ALIGNED_FREE(im);
    return best;
}

#ifdef VFFT_HAS_MKL
static double bench_mkl_2d(int N1, int N2) {
    size_t total = (size_t)N1 * N2;
    double *re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    for (size_t i = 0; i < total; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }
    DFTI_DESCRIPTOR_HANDLE h = NULL;
    MKL_LONG dims[2] = {N1, N2};
    MKL_LONG strides[3] = {0, N2, 1};
    DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 2, dims);
    DftiSetValue(h, DFTI_PLACEMENT, DFTI_INPLACE);
    DftiSetValue(h, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
    DftiSetValue(h, DFTI_INPUT_STRIDES, strides);
    DftiSetValue(h, DFTI_OUTPUT_STRIDES, strides);
    if (DftiCommitDescriptor(h) != DFTI_NO_ERROR) {
        DftiFreeDescriptor(&h); STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);
        return 1e18;
    }
    for (int w = 0; w < 20; w++)
        DftiComputeForward(h, re, im);
    int reps = (int)(2e6 / (total + 1));
    if (reps < 20) reps = 20;
    if (reps > 50000) reps = 50000;
    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++)
            DftiComputeForward(h, re, im);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    DftiFreeDescriptor(&h); STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);
    return best;
}
#endif

static double roundtrip_err_2d(stride_plan_t *plan, size_t total) {
    double *re  = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im  = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *ref = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *rfi = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    for (size_t i = 0; i < total; i++) {
        ref[i] = re[i] = (double)rand() / RAND_MAX - 0.5;
        rfi[i] = im[i] = (double)rand() / RAND_MAX - 0.5;
    }
    stride_execute_fwd(plan, re, im);
    stride_execute_bwd(plan, re, im);
    double N = (double)plan->N, mx = 0;
    for (size_t i = 0; i < total; i++) {
        double d = fabs(re[i] / N - ref[i]);
        if (d > mx) mx = d;
        d = fabs(im[i] / N - rfi[i]);
        if (d > mx) mx = d;
    }
    STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);
    STRIDE_ALIGNED_FREE(ref); STRIDE_ALIGNED_FREE(rfi);
    return mx;
}

typedef struct { int N1, N2; const char *category; } size2d_t;

static const size2d_t sizes_2d[] = {
    /* Square power-of-2 */
    {32,   32,   "sq_pow2"},
    {64,   64,   "sq_pow2"},
    {128,  128,  "sq_pow2"},
    {256,  256,  "sq_pow2"},
    {512,  512,  "sq_pow2"},
    {1024, 1024, "sq_pow2"},
    {2048, 2048, "sq_pow2"},

    /* Rectangular power-of-2 */
    {64,   128,  "rect_pow2"},
    {128,  256,  "rect_pow2"},
    {256,  512,  "rect_pow2"},
    {512,  1024, "rect_pow2"},

    /* HD video resolutions */
    {480,  720,  "hd"},
    {720,  1280, "hd"},
    {1080, 1920, "hd"},

    /* Non-power-of-2 square */
    {100,  100,  "sq_comp"},
    {200,  200,  "sq_comp"},
    {300,  300,  "sq_comp"},
    {500,  500,  "sq_comp"},

    /* Non-power-of-2 rectangular */
    {100,  200,  "rect_comp"},
    {200,  500,  "rect_comp"},
    {300,  500,  "rect_comp"},
};
static const int n_sizes_2d = sizeof(sizes_2d) / sizeof(sizes_2d[0]);

int main(void) {
    stride_env_init();
    stride_pin_thread(0);
    stride_set_num_threads(1);
#ifdef VFFT_HAS_MKL
    mkl_set_num_threads(1);
#endif

    stride_registry_t reg;
    stride_registry_init(&reg);

    /* ═══════════════════════════════════════════════════════
     * Phase 1: Load wisdom, calibrate missing sub-plan entries
     *
     * 2D plans use two 1D sub-plans: (N1, K=N2) and (N2, K=B).
     * Calibrate both if missing from wisdom.
     * ═══════════════════════════════════════════════════════ */

    stride_wisdom_t wis;
    stride_wisdom_init(&wis);
    stride_wisdom_load(&wis, WISDOM_PATH);

    printf("=== Phase 1: Wisdom calibration for 2D sub-plans ===\n");
    printf("Loaded %d existing entries from %s\n", wis.count, WISDOM_PATH);

    int calibrated = 0;
    for (int si = 0; si < n_sizes_2d; si++) {
        int N1 = sizes_2d[si].N1, N2 = sizes_2d[si].N2;
        size_t B = FFT2D_DEFAULT_TILE;
        if (B > (size_t)N1) B = (size_t)N1;

        /* Column sub-plan: N1-point, K=N2 */
        if (!stride_wisdom_lookup(&wis, N1, (size_t)N2)) {
            printf("  Calibrating col N=%d K=%d ... ", N1, N2);
            fflush(stdout);
            double t0 = now_ns();
            stride_wisdom_calibrate(&wis, N1, (size_t)N2, &reg);
            printf("%.1fs\n", (now_ns() - t0) / 1e9);
            calibrated++;
        }

        /* Row sub-plan: N2-point, K=B */
        if (!stride_wisdom_lookup(&wis, N2, B)) {
            printf("  Calibrating row N=%d K=%zu ... ", N2, B);
            fflush(stdout);
            double t0 = now_ns();
            stride_wisdom_calibrate(&wis, N2, B, &reg);
            printf("%.1fs\n", (now_ns() - t0) / 1e9);
            calibrated++;
        }
    }

    if (calibrated > 0) {
        stride_wisdom_save(&wis, WISDOM_PATH);
        printf("Calibrated %d new entries, saved to %s (%d total)\n\n",
               calibrated, WISDOM_PATH, wis.count);
    } else {
        printf("All entries present, no calibration needed.\n\n");
    }

    /* ═══════════════════════════════════════════════════════
     * Phase 2: Benchmark → CSV
     * ═══════════════════════════════════════════════════════ */

    const char *csv_path = "vfft_perf_2d.csv";
    FILE *fp = fopen(csv_path, "w");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", csv_path); return 1; }
    fprintf(fp, "N1,N2,total,category,vfft_ns,mkl_ns,vfft_gflops,mkl_gflops,ratio_vs_mkl,roundtrip_err\n");

    printf("=== Phase 2: 2D Benchmark → %s ===\n\n", csv_path);
    printf("%-12s %-10s %-12s %10s %10s %8s %8s %7s %10s\n",
           "Size", "total", "category", "vfft_ns", "mkl_ns",
           "vfft_GF", "mkl_GF", "ratio", "err");
    printf("------------+----------+------------+----------+"
           "----------+--------+--------+-------+----------\n");

    for (int si = 0; si < n_sizes_2d; si++) {
        int N1 = sizes_2d[si].N1, N2 = sizes_2d[si].N2;
        const char *cat = sizes_2d[si].category;
        size_t total = (size_t)N1 * N2;

        stride_plan_t *plan = stride_plan_2d_wise(N1, N2, &reg, &wis);
        if (!plan) {
            printf("%4dx%-7d PLAN FAILED\n", N1, N2);
            continue;
        }

        double err = roundtrip_err_2d(plan, total);
        double vns = bench_vfft_2d(plan, total);
        double vgf = gflops_2d(N1, N2, vns);

        double mns = 0, mgf = 0;
#ifdef VFFT_HAS_MKL
        mns = bench_mkl_2d(N1, N2);
        mgf = gflops_2d(N1, N2, mns);
#endif
        double ratio = (mns > 0) ? mns / vns : 0;

        char label[32];
        snprintf(label, sizeof(label), "%dx%d", N1, N2);
        printf("%-12s %-10zu %-12s %9.0f %9.0f %7.2f %7.2f %6.2fx %9.1e\n",
               label, total, cat, vns, mns, vgf, mgf, ratio, err);
        fprintf(fp, "%d,%d,%zu,%s,%.0f,%.0f,%.3f,%.3f,%.3f,%.3e\n",
                N1, N2, total, cat, vns, mns, vgf, mgf, ratio, err);

        stride_plan_destroy(plan);
    }

    fclose(fp);
    printf("\nResults → %s\n", csv_path);
    printf("Done.\n");
    return 0;
}
