/**
 * bench_1d_vs_fftw.c — VectorFFT (new core, MEASURE wisdom) vs FFTW3.
 *
 * Companion to bench_1d_vs_mkl.c. Same 207-cell grid, same VFFT side
 * (stride_wise_plan with calibrated wisdom), but the reference is
 * FFTW3 instead of MKL. Uses FFTW's split-complex API
 * (fftw_plan_guru_split_dft) so the layout matches VectorFFT exactly
 * — no interleave/deinterleave overhead on the FFTW side.
 *
 * Phase 1 (calibration) is calibrate.py's job. This binary refuses to
 * bench cells with no wisdom entry; run calibrate.py first.
 *
 * Usage:
 *   bench_1d_vs_fftw                     # default paths
 *   bench_1d_vs_fftw <wisdom_path> <perf_csv> <acc_csv>
 *   bench_1d_vs_fftw --cells "1024:256,4096:32"
 *   bench_1d_vs_fftw --phase perf|acc|both
 *
 * Outputs:
 *   vfft_perf_tuned_1d_fftw.csv  — performance (vfft_ns, fftw_ns, ratio)
 *   vfft_acc_tuned_1d_fftw.csv   — roundtrip error
 *
 * Build:
 *   python build.py --vfft --src build_tuned/bench_1d_vs_fftw.c --fftw
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "planner.h"
#include "compat.h"
#include "env.h"

#include "fftw3.h"

#ifndef DEFAULT_WISDOM_PATH
#define DEFAULT_WISDOM_PATH  "vfft_wisdom_tuned.txt"
#endif
#ifndef DEFAULT_PERF_PATH
#define DEFAULT_PERF_PATH    "vfft_perf_tuned_1d_fftw.csv"
#endif
#ifndef DEFAULT_ACC_PATH
#define DEFAULT_ACC_PATH     "vfft_acc_tuned_1d_fftw.csv"
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

/* FFTW split-complex C2C forward, batched K, stride=K, dist=1 — matches
 * VectorFFT's row-major split-complex layout exactly. */
static double bench_fftw(int N, size_t K) {
    size_t total = (size_t)N * K;
    /* fftw_malloc for SIMD alignment. */
    double *re = (double *)fftw_malloc(total * sizeof(double));
    double *im = (double *)fftw_malloc(total * sizeof(double));
    if (!re || !im) {
        if (re) fftw_free(re);
        if (im) fftw_free(im);
        return 1e18;
    }
    for (size_t i = 0; i < total; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    fftw_iodim dims[1];
    dims[0].n  = N;
    dims[0].is = (int)K;     /* stride within transform: contiguous on K dim */
    dims[0].os = (int)K;
    fftw_iodim howmany_dims[1];
    howmany_dims[0].n  = (int)K;
    howmany_dims[0].is = 1;  /* stride between transforms */
    howmany_dims[0].os = 1;

    /* In-place: ri==ro, ii==io. FFTW_MEASURE for fair comparison. */
    fftw_plan p = fftw_plan_guru_split_dft(
        1, dims, 1, howmany_dims,
        re, im, re, im,
        FFTW_MEASURE);
    if (!p) {
        fftw_free(re); fftw_free(im);
        return 1e18;
    }

    /* FFTW_MEASURE may have trashed the buffers — refill. */
    for (size_t i = 0; i < total; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    /* Warmup */
    for (int w = 0; w < 10; w++)
        fftw_execute_split_dft(p, re, im, re, im);

    int reps = (int)(2e6 / (total + 1));
    if (reps < 20) reps = 20;
    if (reps > 100000) reps = 100000;

    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++)
            fftw_execute_split_dft(p, re, im, re, im);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    fftw_destroy_plan(p);
    fftw_free(re); fftw_free(im);
    return best;
}

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
 * Size grid (mirrors bench_1d_vs_mkl.c exactly — 69 sizes × 3 K = 207)
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

    {47, "bluestein"},   /* p-1 = 2*23  */
    {59, "bluestein"},   /* p-1 = 2*29  */
    {83, "bluestein"},   /* p-1 = 2*41  */
    {107, "bluestein"},  /* p-1 = 2*53  */
    {167, "bluestein"},  /* p-1 = 2*83  */
    {179, "bluestein"},  /* p-1 = 2*89  */
    {263, "bluestein"},  /* p-1 = 2*131 */
    {311, "bluestein"},  /* p-1 = 2*5*31 (31 > 19) */

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

static int _cell_in_filter(int N, size_t K, const char *filter) {
    if (!filter || !*filter) return 1;
    const char *p = filter;
    while (*p) {
        int fN = 0, fK = 0;
        while (*p == ' ' || *p == ',') p++;
        while (*p >= '0' && *p <= '9') { fN = fN*10 + (*p++ - '0'); }
        if (*p == ':') {
            p++;
            while (*p >= '0' && *p <= '9') { fK = fK*10 + (*p++ - '0'); }
            if (fN == N && (size_t)fK == K) return 1;
        }
        while (*p && *p != ',') p++;
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *wisdom_path = DEFAULT_WISDOM_PATH;
    const char *perf_path   = DEFAULT_PERF_PATH;
    const char *acc_path    = DEFAULT_ACC_PATH;
    const char *cells       = NULL;
    int run_perf = 1, run_acc = 1;

    int posn = 1;
    int positional_idx = 0;
    while (posn < argc) {
        if (strcmp(argv[posn], "--cells") == 0 && posn + 1 < argc) {
            cells = argv[posn + 1];
            posn += 2;
            continue;
        }
        if (strcmp(argv[posn], "--phase") == 0 && posn + 1 < argc) {
            const char *p = argv[posn + 1];
            if      (strcmp(p, "perf") == 0) { run_perf = 1; run_acc = 0; }
            else if (strcmp(p, "acc")  == 0) { run_perf = 0; run_acc = 1; }
            else if (strcmp(p, "both") == 0) { run_perf = 1; run_acc = 1; }
            else {
                fprintf(stderr, "--phase must be one of: perf, acc, both\n");
                return 2;
            }
            posn += 2;
            continue;
        }
        switch (positional_idx) {
            case 0: wisdom_path = argv[posn]; break;
            case 1: perf_path   = argv[posn]; break;
            case 2: acc_path    = argv[posn]; break;
            default: break;
        }
        positional_idx++;
        posn++;
    }

    stride_env_init();
    stride_pin_thread(0);
    stride_set_num_threads(1);

    printf("=== bench_1d_vs_fftw: VFFT (MEASURE wisdom) vs FFTW3 ===\n");
    printf("wisdom : %s\n", wisdom_path);
    printf("perf   : %s\n", perf_path);
    printf("acc    : %s\n", acc_path);
    printf("FFTW3  : single-threaded, FFTW_MEASURE planning\n");

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

  if (run_perf) {
    FILE *fp = fopen(perf_path, "w");
    if (!fp) { fprintf(stderr, "cannot open %s\n", perf_path); return 1; }
    fprintf(fp, "N,K,category,factors,vfft_ns,fftw_ns,"
                "vfft_gflops,fftw_gflops,ratio_vs_fftw\n");

    printf("=== Phase 2: Performance ===\n\n");
    printf("%-8s %-5s %-12s %-20s %10s %10s %8s %8s %7s\n",
           "N", "K", "category", "factors", "vfft_ns", "fftw_ns",
           "vfft_GF", "fftw_GF", "ratio");
    printf("--------+-----+------------+--------------------+----------+"
           "----------+--------+--------+-------\n");

    if (cells) printf("cells filter active: %s\n", cells);

    int n_benched = 0, n_skipped = 0, n_wins = 0;
    double sum_ratio = 0, min_ratio = 1e30, max_ratio = 0;

    for (int si = 0; si < n_sizes; si++) {
        int N = all_sizes[si].N;
        const char *cat = all_sizes[si].category;
        for (int ki = 0; ki < n_Ks; ki++) {
            size_t K = Ks[ki];

            if (!_cell_in_filter(N, K, cells)) { n_skipped++; continue; }

            stride_plan_t *plan = stride_wise_plan(N, K, &reg, &wis);
            if (!plan) {
                fprintf(stderr,
                        "  N=%d K=%zu: stride_wise_plan failed\n", N, K);
                n_skipped++;
                continue;
            }

            char fstr[64];
            format_plan(fstr, sizeof(fstr), plan);

            double vns = bench_vfft(plan, N, K);
            double vgf = gflops(N, K, vns);

            double fns = bench_fftw(N, K);
            double fgf = gflops(N, K, fns);

            double ratio = (fns > 0 && fns < 1e17) ? fns / vns : 0;

            printf("%-8d %-5zu %-12s %-20s %9.0f %9.0f %7.2f %7.2f %6.2fx\n",
                   N, K, cat, fstr, vns, fns, vgf, fgf, ratio);
            fprintf(fp, "%d,%zu,%s,%s,%.0f,%.0f,%.3f,%.3f,%.3f\n",
                    N, K, cat, fstr, vns, fns, vgf, fgf, ratio);
            fflush(fp);

            stride_plan_destroy(plan);
            n_benched++;
            if (ratio > 1.0) n_wins++;
            if (ratio > 0) {
                sum_ratio += ratio;
                if (ratio < min_ratio) min_ratio = ratio;
                if (ratio > max_ratio) max_ratio = ratio;
            }
        }
    }
    fclose(fp);

    printf("\n=== Summary ===\n");
    printf("Benched %d cells, skipped %d.\n", n_benched, n_skipped);
    if (n_benched > 0) {
        printf("Wins vs FFTW3: %d/%d (%.0f%%)\n",
               n_wins, n_benched, 100.0 * n_wins / n_benched);
        printf("Ratio: min=%.2fx  mean=%.2fx  max=%.2fx\n",
               min_ratio, sum_ratio / n_benched, max_ratio);
    }
    printf("Performance → %s\n\n", perf_path);
  } else {
    printf("=== Phase 2: SKIPPED (--phase acc) ===\n\n");
  }

    /* ───────────────────────────────────────────────
     * Phase 3: Accuracy → CSV  (VFFT-only roundtrip)
     * ─────────────────────────────────────────────── */

  if (run_acc) {
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

        if (!_cell_in_filter(N, K, cells)) continue;
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
  } else {
    printf("=== Phase 3: SKIPPED (--phase perf) ===\n");
  }

    fftw_cleanup();
    printf("\nDone.\n");
    return 0;
}
