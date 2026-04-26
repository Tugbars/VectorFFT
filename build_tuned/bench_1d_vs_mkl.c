/**
 * bench_1d_vs_mkl.c — VectorFFT (new core, MEASURE wisdom) vs MKL.
 *
 * Companion to calibrate_tuned: the calibrator produces
 *   build_tuned/vfft_wisdom_tuned.txt  (v5, with explicit variant codes)
 * and this binary loads that wisdom, builds plans through the new core's
 * stride_wise_plan (which dispatches v5 entries via
 * _stride_build_plan_explicit), and benches each plan against MKL on
 * the same (N, K) grid bench_1d_csv.c uses.
 *
 * Phase 1 (calibration) is intentionally absent — that's calibrate.py's
 * job. This binary refuses to bench cells with no wisdom entry; run
 * calibrate.py first to populate the grid you want to measure.
 *
 * Usage:
 *   bench_1d_vs_mkl                    # default paths
 *   bench_1d_vs_mkl <wisdom_path> <perf_csv> <acc_csv>
 *
 * Outputs:
 *   vfft_perf_tuned_1d.csv  — performance (vfft_ns, mkl_ns, ratio)
 *   vfft_acc_tuned_1d.csv   — roundtrip error
 *
 * Build:
 *   The new-core build.py harness needs MKL include + link flags added.
 *   See note at end of file for the minimal extension. Without MKL,
 *   the binary still runs and reports VFFT-only numbers (mkl columns
 *   = 0 in the CSV).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "planner.h"          /* src/core/planner.h via -I src/core */
#include "compat.h"
#include "env.h"

#ifdef VFFT_HAS_MKL
#include <mkl_dfti.h>
#include <mkl_service.h>
#endif

#ifndef DEFAULT_WISDOM_PATH
#define DEFAULT_WISDOM_PATH  "vfft_wisdom_tuned.txt"
#endif
#ifndef DEFAULT_PERF_PATH
#define DEFAULT_PERF_PATH    "vfft_perf_tuned_1d.csv"
#endif
#ifndef DEFAULT_ACC_PATH
#define DEFAULT_ACC_PATH     "vfft_acc_tuned_1d.csv"
#endif

/* ─────────────────────────────────────────────────────────────────
 * Helpers
 * ───────────────────────────────────────────────────────────────── */

static double gflops(int N, size_t K, double ns) {
    if (ns <= 0) return 0;
    return 5.0 * N * log2((double)N) * K / ns;
}

static double bench_vfft(stride_plan_t *plan, int N, size_t K) {
    size_t total = (size_t)N * K;
    double *re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    if (!re || !im) {
        if (re) STRIDE_ALIGNED_FREE(re);
        if (im) STRIDE_ALIGNED_FREE(im);
        return 1e18;
    }
    for (size_t i = 0; i < total; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }
    /* Warmup */
    for (int w = 0; w < 10; w++)
        stride_execute_fwd(plan, re, im);

    int reps = (int)(2e6 / (total + 1));
    if (reps < 20) reps = 20;
    if (reps > 100000) reps = 100000;

    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++)
            stride_execute_fwd(plan, re, im);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);
    return best;
}

#ifdef VFFT_HAS_MKL
static double bench_mkl(int N, size_t K) {
    size_t total = (size_t)N * K;
    double *re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    if (!re || !im) {
        if (re) STRIDE_ALIGNED_FREE(re);
        if (im) STRIDE_ALIGNED_FREE(im);
        return 1e18;
    }
    for (size_t i = 0; i < total; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }
    DFTI_DESCRIPTOR_HANDLE desc = NULL;
    MKL_LONG strides[2] = {0, (MKL_LONG)K};
    DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
    DftiSetValue(desc, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
    DftiSetValue(desc, DFTI_PLACEMENT, DFTI_INPLACE);
    DftiSetValue(desc, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    DftiSetValue(desc, DFTI_INPUT_DISTANCE, 1);
    DftiSetValue(desc, DFTI_OUTPUT_DISTANCE, 1);
    DftiSetValue(desc, DFTI_INPUT_STRIDES, strides);
    DftiSetValue(desc, DFTI_OUTPUT_STRIDES, strides);
    if (DftiCommitDescriptor(desc) != DFTI_NO_ERROR) {
        DftiFreeDescriptor(&desc);
        STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);
        return 1e18;
    }
    /* Warmup */
    for (int w = 0; w < 10; w++)
        DftiComputeForward(desc, re, im);

    int reps = (int)(2e6 / (total + 1));
    if (reps < 20) reps = 20;
    if (reps > 100000) reps = 100000;

    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++)
            DftiComputeForward(desc, re, im);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    DftiFreeDescriptor(&desc);
    STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);
    return best;
}
#endif

static double roundtrip_err(stride_plan_t *plan, int N, size_t K) {
    size_t total = (size_t)N * K;
    double *re  = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im  = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *ref = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *rfi = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    if (!re || !im || !ref || !rfi) {
        if (re) STRIDE_ALIGNED_FREE(re);  if (im) STRIDE_ALIGNED_FREE(im);
        if (ref) STRIDE_ALIGNED_FREE(ref); if (rfi) STRIDE_ALIGNED_FREE(rfi);
        return 1e30;
    }
    srand(42 + N + (int)K);
    for (size_t i = 0; i < total; i++) {
        ref[i] = re[i] = (double)rand() / RAND_MAX - 0.5;
        rfi[i] = im[i] = (double)rand() / RAND_MAX - 0.5;
    }
    stride_execute_fwd(plan, re, im);
    stride_execute_bwd(plan, re, im);
    double Nd = (double)N, mx = 0;
    for (size_t i = 0; i < total; i++) {
        double d = fabs(re[i] / Nd - ref[i]);
        if (d > mx) mx = d;
        d = fabs(im[i] / Nd - rfi[i]);
        if (d > mx) mx = d;
    }
    STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);
    STRIDE_ALIGNED_FREE(ref); STRIDE_ALIGNED_FREE(rfi);
    return mx;
}

static void format_plan(char *buf, size_t buflen, const stride_plan_t *plan) {
    buf[0] = 0;
    if (plan->num_stages == 0) {
        snprintf(buf, buflen, "[override]");   /* Rader/Bluestein */
        return;
    }
    size_t pos = 0;
    for (int s = 0; s < plan->num_stages && pos < buflen - 16; s++) {
        int n = snprintf(buf + pos, buflen - pos, "%s%d",
                         s ? "x" : "", plan->factors[s]);
        if (n < 0) break;
        pos += (size_t)n;
    }
}

/* ─────────────────────────────────────────────────────────────────
 * Size grid (mirrors bench_1d_csv.c)
 * ───────────────────────────────────────────────────────────────── */

typedef struct { int N; const char *category; } size_entry_t;

static const size_entry_t all_sizes[] = {
    {8, "small"}, {16, "small"}, {32, "small"}, {64, "small"}, {128, "small"},
    {256, "pow2"}, {512, "pow2"}, {1024, "pow2"}, {2048, "pow2"},
    {4096, "pow2"}, {8192, "pow2"}, {16384, "pow2"}, {32768, "pow2"},
    {65536, "pow2"}, {131072, "pow2"},

    {60, "composite"}, {100, "composite"}, {200, "composite"},
    {500, "composite"}, {1000, "composite"}, {2000, "composite"},
    {5000, "composite"}, {10000, "composite"}, {20000, "composite"},
    {50000, "composite"}, {100000, "composite"},

    {243, "prime_pow"},   /* 3^5 */
    {625, "prime_pow"},   /* 5^4 */
    {2401, "prime_pow"},  /* 7^4 */
    {3125, "prime_pow"},  /* 5^5 */
    {15625, "prime_pow"}, /* 5^6 */
    {16807, "prime_pow"}, /* 7^5 */
    {78125, "prime_pow"}, /* 5^7 */
    {117649, "prime_pow"},/* 7^6 */
    {390625, "prime_pow"},/* 5^8 */
    {823543, "prime_pow"},/* 7^7 */

    {1331, "genfft"},     /* 11^3 */
    {14641, "genfft"},    /* 11^4 */
    {161051, "genfft"},   /* 11^5 */
    {2197, "genfft"},     /* 13^3 */
    {28561, "genfft"},    /* 13^4 */

    {127, "rader"}, {251, "rader"}, {257, "rader"}, {401, "rader"},
    {641, "rader"}, {1009, "rader"}, {2801, "rader"}, {4001, "rader"},

    {175, "odd_comp"}, {525, "odd_comp"}, {1225, "odd_comp"},
    {2205, "odd_comp"}, {6615, "odd_comp"}, {11025, "odd_comp"},

    {2310, "mixed_deep"}, {6930, "mixed_deep"}, {30030, "mixed_deep"},
    {60060, "mixed_deep"}, {4620, "mixed_deep"}, {13860, "mixed_deep"},
};
static const int n_sizes = (int)(sizeof(all_sizes) / sizeof(all_sizes[0]));

static const size_t Ks[] = {4, 32, 256};
static const int n_Ks = (int)(sizeof(Ks) / sizeof(Ks[0]));

/* ─────────────────────────────────────────────────────────────────
 * Main
 * ───────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    const char *wisdom_path = DEFAULT_WISDOM_PATH;
    const char *perf_path   = DEFAULT_PERF_PATH;
    const char *acc_path    = DEFAULT_ACC_PATH;
    if (argc >= 2) wisdom_path = argv[1];
    if (argc >= 3) perf_path   = argv[2];
    if (argc >= 4) acc_path    = argv[3];

    stride_env_init();
    stride_pin_thread(0);
    stride_set_num_threads(1);
#ifdef VFFT_HAS_MKL
    mkl_set_num_threads(1);
#endif

    printf("=== bench_1d_vs_mkl: new core (MEASURE wisdom) vs MKL ===\n");
    printf("wisdom : %s\n", wisdom_path);
    printf("perf   : %s\n", perf_path);
    printf("acc    : %s\n", acc_path);
#ifdef VFFT_HAS_MKL
    printf("MKL    : enabled\n");
#else
    printf("MKL    : DISABLED (rebuild with -DVFFT_HAS_MKL to enable)\n");
#endif

    stride_registry_t reg;
    stride_registry_init(&reg);

    stride_wisdom_t wis;
    stride_wisdom_init(&wis);

    int loaded = stride_wisdom_load(&wis, wisdom_path);
    if (loaded < 0 || wis.count == 0) {
        fprintf(stderr,
                "fatal: no wisdom entries loaded from %s "
                "(run calibrate.py --mode=measure first)\n",
                wisdom_path);
        return 2;
    }
    printf("loaded : %d wisdom entries\n\n", wis.count);

    /* ───────────────────────────────────────────────
     * Phase 2: Performance benchmark → CSV
     * ─────────────────────────────────────────────── */

    FILE *fp = fopen(perf_path, "w");
    if (!fp) { fprintf(stderr, "cannot open %s\n", perf_path); return 1; }
    fprintf(fp, "N,K,category,factors,vfft_ns,mkl_ns,"
                "vfft_gflops,mkl_gflops,ratio_vs_mkl\n");

    printf("=== Phase 2: Performance ===\n\n");
    printf("%-8s %-5s %-12s %-20s %10s %10s %8s %8s %7s\n",
           "N", "K", "category", "factors", "vfft_ns", "mkl_ns",
           "vfft_GF", "mkl_GF", "ratio");
    printf("--------+-----+------------+--------------------+----------+"
           "----------+--------+--------+-------\n");

    int n_benched = 0, n_skipped = 0;
    for (int si = 0; si < n_sizes; si++) {
        int N = all_sizes[si].N;
        const char *cat = all_sizes[si].category;
        for (int ki = 0; ki < n_Ks; ki++) {
            size_t K = Ks[ki];

            /* Skip cells with no wisdom entry — we explicitly do not
             * fall back to estimate-mode plans, since the whole point
             * of this bench is to measure the MEASURE-tuned plan. */
            const stride_wisdom_entry_t *e =
                stride_wisdom_lookup(&wis, N, K);
            if (!e) { n_skipped++; continue; }

            stride_plan_t *plan = stride_wise_plan(N, K, &reg, &wis);
            if (!plan) {
                fprintf(stderr,
                        "  N=%d K=%zu: wisdom present but stride_wise_plan "
                        "failed; check codelet registry\n", N, K);
                n_skipped++;
                continue;
            }

            char fstr[64];
            format_plan(fstr, sizeof(fstr), plan);

            double vns = bench_vfft(plan, N, K);
            double vgf = gflops(N, K, vns);

            double mns = 0, mgf = 0;
#ifdef VFFT_HAS_MKL
            mns = bench_mkl(N, K);
            mgf = gflops(N, K, mns);
#endif
            double ratio = (mns > 0) ? mns / vns : 0;

            printf("%-8d %-5zu %-12s %-20s %9.0f %9.0f %7.2f %7.2f %6.2fx\n",
                   N, K, cat, fstr, vns, mns, vgf, mgf, ratio);
            fprintf(fp, "%d,%zu,%s,%s,%.0f,%.0f,%.3f,%.3f,%.3f\n",
                    N, K, cat, fstr, vns, mns, vgf, mgf, ratio);
            fflush(fp);

            stride_plan_destroy(plan);
            n_benched++;
        }
    }
    fclose(fp);
    printf("\nBenched %d cells, skipped %d (no wisdom entry).\n",
           n_benched, n_skipped);
    printf("Performance → %s\n\n", perf_path);

    /* ───────────────────────────────────────────────
     * Phase 3: Accuracy → CSV
     * ─────────────────────────────────────────────── */

    FILE *fa = fopen(acc_path, "w");
    if (!fa) { fprintf(stderr, "cannot open %s\n", acc_path); return 1; }
    fprintf(fa, "N,category,roundtrip_err\n");

    printf("=== Phase 3: Accuracy ===\n\n");
    printf("%-8s %-12s %12s\n", "N", "category", "roundtrip_err");
    printf("--------+------------+------------\n");

    for (int si = 0; si < n_sizes; si++) {
        int N = all_sizes[si].N;
        const char *cat = all_sizes[si].category;
        size_t K = 4;

        if (!stride_wisdom_lookup(&wis, N, K)) continue;

        stride_plan_t *plan = stride_wise_plan(N, K, &reg, &wis);
        if (!plan) continue;

        double err = roundtrip_err(plan, N, K);
        printf("%-8d %-12s %11.2e\n", N, cat, err);
        fprintf(fa, "%d,%s,%.3e\n", N, cat, err);
        stride_plan_destroy(plan);
    }
    fclose(fa);
    printf("\nAccuracy → %s\n", acc_path);
    printf("\nDone.\n");
    return 0;
}

/* ─────────────────────────────────────────────────────────────────
 * Build notes — MKL extension for build.py
 *
 * Production CMake uses ILP64 sequential. To match, build.py needs a
 * `--mkl` flag that appends (when MKLROOT is set in the environment):
 *
 *   includes:  -I"$MKLROOT/include"
 *   flags:     -DVFFT_HAS_MKL -DMKL_ILP64
 *   link:      -L"$MKLROOT/lib/intel64"
 *              mkl_intel_ilp64.lib  mkl_sequential.lib  mkl_core.lib
 *              (Linux: -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core)
 *
 * Discovery probe (mirrors src/stride-fft/CMakeLists.txt:139-177):
 *   - exists $MKLROOT/include/mkl_dfti.h  → header found
 *   - exists $MKLROOT/lib/intel64/mkl_intel_ilp64.lib  → libs present
 *
 * Without --mkl the binary still compiles (VFFT-only path), and the
 * mkl_ns column is 0 in the CSV. Useful as a sanity gate.
 * ───────────────────────────────────────────────────────────────── */
