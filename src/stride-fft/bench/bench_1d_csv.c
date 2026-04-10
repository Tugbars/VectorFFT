/**
 * bench_1d_csv.c — Comprehensive 1D FFT benchmark with CSV output
 *
 * Phase 1: Load wisdom, calibrate missing (N,K) entries, save back.
 * Phase 2: Benchmark wisdom-plan vs MKL, compute GFLOP/s.
 * Phase 3: Accuracy (roundtrip error per N).
 *
 * Outputs:
 *   vfft_perf_1d.csv  — performance
 *   vfft_acc_1d.csv   — accuracy
 *   vfft_wisdom.txt   — updated wisdom (shared with bench_planner)
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

/* ── Helpers ── */

static double gflops(int N, size_t K, double ns) {
    if (ns <= 0) return 0;
    return 5.0 * N * log2((double)N) * K / ns;
}

static double bench_vfft(stride_plan_t *plan, int N, size_t K) {
    size_t total = (size_t)N * K;
    double *re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    for (size_t i = 0; i < total; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }
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
    STRIDE_ALIGNED_FREE(re);
    STRIDE_ALIGNED_FREE(im);
    return best;
}

#ifdef VFFT_HAS_MKL
static double bench_mkl(int N, size_t K) {
    size_t total = (size_t)N * K;
    double *re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
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
        DftiFreeDescriptor(&desc); STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);
        return 1e18;
    }
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
    DftiFreeDescriptor(&desc); STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);
    return best;
}
#endif

static double roundtrip_err(stride_plan_t *plan, int N, size_t K) {
    size_t total = (size_t)N * K;
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

static void format_plan(char *buf, const stride_plan_t *plan) {
    buf[0] = 0;
    for (int s = 0; s < plan->num_stages; s++) {
        char tmp[16];
        sprintf(tmp, "%s%d", s ? "x" : "", plan->factors[s]);
        strcat(buf, tmp);
    }
}

/* ── Size tables ── */

typedef struct { int N; const char *category; } size_entry_t;

static const size_entry_t all_sizes[] = {
    /* Small */
    {8,      "small"},
    {16,     "small"},
    {32,     "small"},
    {64,     "small"},
    {128,    "small"},

    /* Power-of-2 */
    {256,    "pow2"},
    {512,    "pow2"},
    {1024,   "pow2"},
    {2048,   "pow2"},
    {4096,   "pow2"},
    {8192,   "pow2"},
    {16384,  "pow2"},
    {32768,  "pow2"},
    {65536,  "pow2"},
    {131072, "pow2"},

    /* Composite (mixed radix) */
    {60,     "composite"},
    {100,    "composite"},
    {200,    "composite"},
    {500,    "composite"},
    {1000,   "composite"},
    {2000,   "composite"},
    {5000,   "composite"},
    {10000,  "composite"},
    {20000,  "composite"},
    {50000,  "composite"},
    {100000, "composite"},

    /* Pure prime powers — small */
    {243,    "prime_pow"},   /* 3^5 */
    {625,    "prime_pow"},   /* 5^4 */
    {2401,   "prime_pow"},   /* 7^4 */
    {3125,   "prime_pow"},   /* 5^5 */
    {15625,  "prime_pow"},   /* 5^6 */
    {16807,  "prime_pow"},   /* 7^5 */
    {78125,  "prime_pow"},   /* 5^7 */

    /* Pure prime powers — large */
    {59049,  "prime_pow"},   /* 3^10 */
    {117649, "prime_pow"},   /* 7^6 */
    {390625, "prime_pow"},   /* 5^8 */
    {531441, "prime_pow"},   /* 3^12 */
    {823543, "prime_pow"},   /* 7^7 */

    /* Genfft prime powers (R=11, R=13) */
    {1331,   "genfft"},      /* 11^3 */
    {14641,  "genfft"},      /* 11^4 */
    {161051, "genfft"},      /* 11^5 */
    {2197,   "genfft"},      /* 13^3 */
    {28561,  "genfft"},      /* 13^4 */

    /* Rader primes (smooth p-1) */
    {127,    "rader"},       /* p-1 = 2*3^2*7 */
    {251,    "rader"},       /* p-1 = 2*5^3 */
    {257,    "rader"},       /* p-1 = 2^8 */
    {401,    "rader"},       /* p-1 = 2^4*5^2 */
    {641,    "rader"},       /* p-1 = 2^7*5 */
    {1009,   "rader"},       /* p-1 = 2^4*3^2*7 */
    {2801,   "rader"},       /* p-1 = 2^4*5^2*7 */
    {4001,   "rader"},       /* p-1 = 2^5*5^3 */

    /* Odd composites */
    {175,    "odd_comp"},    /* 5^2 * 7 */
    {525,    "odd_comp"},    /* 3 * 5^2 * 7 */
    {1225,   "odd_comp"},    /* 5^2 * 7^2 */
    {2205,   "odd_comp"},    /* 3^2 * 5 * 7^2 */
    {6615,   "odd_comp"},    /* 3^3 * 5 * 7^2 */
    {11025,  "odd_comp"},    /* 3^2 * 5^2 * 7^2 */

    /* Mixed high-radix composites — deep factorizations */
    {2310,   "mixed_deep"},  /* 2 * 3 * 5 * 7 * 11 */
    {6930,   "mixed_deep"},  /* 2 * 3^2 * 5 * 7 * 11 */
    {30030,  "mixed_deep"},  /* 2 * 3 * 5 * 7 * 11 * 13 */
    {60060,  "mixed_deep"},  /* 2^2 * 3 * 5 * 7 * 11 * 13 */
    {4620,   "mixed_deep"},  /* 2^2 * 3 * 5 * 7 * 11 */
    {13860,  "mixed_deep"},  /* 2^2 * 3^2 * 5 * 7 * 11 */
};
static const int n_sizes = sizeof(all_sizes) / sizeof(all_sizes[0]);

/* K=4 minimum (K=1 crashes on AVX2 codelets). K=4 serves as single-transform proxy. */
static const size_t Ks[] = {4, 32, 256};
static const int n_Ks = sizeof(Ks) / sizeof(Ks[0]);

int main(int argc, char **argv) {
    stride_env_init();
    stride_pin_thread(0);
    stride_set_num_threads(1);
#ifdef VFFT_HAS_MKL
    mkl_set_num_threads(1);
#endif

    /* --recalibrate: wipe wisdom and re-run all entries from scratch.
     * Use after codelet changes to ensure optimal factorizations. */
    int recalibrate = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--recalibrate") == 0) recalibrate = 1;
    }

    stride_registry_t reg;
    stride_registry_init(&reg);

    /* ═══════════════════════════════════════════════════════
     * Phase 1: Load wisdom, calibrate missing entries, save
     * ═══════════════════════════════════════════════════════ */

    stride_wisdom_t wis;
    stride_wisdom_init(&wis);

    if (recalibrate) {
        printf("=== Phase 1: RECALIBRATE (all entries from scratch) ===\n");
    } else {
        stride_wisdom_load(&wis, WISDOM_PATH);
        printf("=== Phase 1: Wisdom calibration ===\n");
        printf("Loaded %d existing entries from %s\n", wis.count, WISDOM_PATH);
    }

    /* Shared DP contexts per K — reuses sub-problem memoization across sizes */
    int max_N = 0;
    for (int si = 0; si < n_sizes; si++)
        if (all_sizes[si].N > max_N) max_N = all_sizes[si].N;

    stride_dp_context_t dp_4, dp_32, dp_256;
    stride_dp_init(&dp_4,   4,   max_N);
    stride_dp_init(&dp_32,  32,  max_N);
    stride_dp_init(&dp_256, 256, max_N);

    #define BENCH_EXHAUSTIVE_THRESH 2048

    int calibrated = 0;
    for (int si = 0; si < n_sizes; si++) {
        int N = all_sizes[si].N;
        for (int ki = 0; ki < n_Ks; ki++) {
            size_t K = Ks[ki];

            /* Check if already in wisdom before calling */
            int had_entry = (stride_wisdom_lookup(&wis, N, K) != NULL);

            stride_dp_context_t *dp = (K == 4) ? &dp_4 :
                                      (K == 32) ? &dp_32 : &dp_256;

            const char *method = (N <= BENCH_EXHAUSTIVE_THRESH) ? "exhaustive" : "DP";
            printf("  N=%-6d K=%-4zu ", N, K);
            fflush(stdout);

            double t0 = now_ns();
            double ns = stride_wisdom_calibrate_full(
                &wis, N, K, &reg, dp,
                recalibrate,              /* force */
                1,                        /* verbose */
                BENCH_EXHAUSTIVE_THRESH,
                WISDOM_PATH);             /* save to disk after each entry */
            double elapsed = (now_ns() - t0) / 1e9;

            if (!had_entry || recalibrate) {
                if (ns < 1e17) {
                    printf("    → saved [%s] (%.1fs)\n", method, elapsed);
                    calibrated++;
                } else {
                    printf("    → FAILED [%s] (%.1fs)\n", method, elapsed);
                }
            } else {
                /* Was cached — calibrate_full returned early */
                const stride_wisdom_entry_t *e = stride_wisdom_lookup(&wis, N, K);
                printf("[cached] ");
                for (int f = 0; f < e->nfactors; f++)
                    printf("%s%d", f ? "x" : "", e->factors[f]);
                printf("\n");
            }
        }
    }

    stride_dp_destroy(&dp_4);
    stride_dp_destroy(&dp_32);
    stride_dp_destroy(&dp_256);

    if (calibrated > 0) {
        printf("\nCalibrated %d new entries, wisdom has %d total in %s\n\n",
               calibrated, wis.count, WISDOM_PATH);
    } else {
        printf("\nAll entries present, no calibration needed.\n\n");
    }

    /* ═══════════════════════════════════════════════════════
     * Phase 2: Performance benchmark → CSV
     * ═══════════════════════════════════════════════════════ */

    const char *perf_path = "vfft_perf_1d.csv";
    FILE *fp = fopen(perf_path, "w");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", perf_path); return 1; }
    fprintf(fp, "N,K,category,factors,vfft_ns,mkl_ns,vfft_gflops,mkl_gflops,ratio_vs_mkl\n");

    printf("=== Phase 2: Performance → %s ===\n\n", perf_path);

    /* Pre-build all plans, verify wisdom → plan factorization match */
    printf("Verifying wisdom → plan factorizations:\n");
    for (int si = 0; si < n_sizes; si++) {
        int N = all_sizes[si].N;
        for (int ki = 0; ki < n_Ks; ki++) {
            size_t K = Ks[ki];
            const stride_wisdom_entry_t *e = stride_wisdom_lookup(&wis, N, K);
            stride_plan_t *p = stride_wise_plan(N, K, &reg, &wis);
            if (!e || !p) continue;

            /* Wisdom factorization */
            printf("  N=%-6d K=%-4zu  wis=", N, K);
            for (int f = 0; f < e->nfactors; f++)
                printf("%s%d", f ? "x" : "", e->factors[f]);

            /* Plan factorization (should match for staged plans) */
            if (p->num_stages > 0) {
                printf("  plan=");
                for (int s = 0; s < p->num_stages; s++)
                    printf("%s%d", s ? "x" : "", p->factors[s]);

                /* Check match */
                int match = (p->num_stages == e->nfactors);
                if (match)
                    for (int s = 0; s < p->num_stages; s++)
                        if (p->factors[s] != e->factors[s]) { match = 0; break; }
                printf("  %s", match ? "OK" : "MISMATCH!");
            } else {
                printf("  plan=[override]");  /* Rader/Bluestein */
            }
            printf("\n");
            stride_plan_destroy(p);
        }
    }
    printf("\nBenchmarking...\n\n");

    printf("%-8s %-5s %-12s %-16s %10s %10s %8s %8s %7s\n",
           "N", "K", "category", "factors", "vfft_ns", "mkl_ns",
           "vfft_GF", "mkl_GF", "ratio");
    printf("--------+-----+------------+----------------+----------+"
           "----------+--------+--------+-------\n");

    for (int si = 0; si < n_sizes; si++) {
        int N = all_sizes[si].N;
        const char *cat = all_sizes[si].category;

        for (int ki = 0; ki < n_Ks; ki++) {
            size_t K = Ks[ki];

            stride_plan_t *plan = stride_wise_plan(N, K, &reg, &wis);
            if (!plan) continue;

            char fstr[64];
            format_plan(fstr, plan);

            double vns = bench_vfft(plan, N, K);
            double vgf = gflops(N, K, vns);

            double mns = 0, mgf = 0;
#ifdef VFFT_HAS_MKL
            mns = bench_mkl(N, K);
            mgf = gflops(N, K, mns);
#endif
            double ratio = (mns > 0) ? mns / vns : 0;

            printf("%-8d %-5zu %-12s %-16s %9.0f %9.0f %7.2f %7.2f %6.2fx\n",
                   N, K, cat, fstr, vns, mns, vgf, mgf, ratio);
            fprintf(fp, "%d,%zu,%s,%s,%.0f,%.0f,%.3f,%.3f,%.3f\n",
                    N, K, cat, fstr, vns, mns, vgf, mgf, ratio);

            stride_plan_destroy(plan);
        }
    }
    fclose(fp);
    printf("\nPerformance → %s\n", perf_path);

    /* ═══════════════════════════════════════════════════════
     * Phase 3: Accuracy → CSV
     * ═══════════════════════════════════════════════════════ */

    const char *acc_path = "vfft_acc_1d.csv";
    FILE *fa = fopen(acc_path, "w");
    if (!fa) { fprintf(stderr, "Cannot open %s\n", acc_path); return 1; }
    fprintf(fa, "N,category,roundtrip_err\n");

    printf("\n=== Phase 3: Accuracy → %s ===\n\n", acc_path);
    printf("%-8s %-12s %12s\n", "N", "category", "roundtrip_err");
    printf("--------+------------+------------\n");

    for (int si = 0; si < n_sizes; si++) {
        int N = all_sizes[si].N;
        const char *cat = all_sizes[si].category;
        size_t K = 4;

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
