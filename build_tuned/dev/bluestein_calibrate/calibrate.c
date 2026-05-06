/* calibrate.c -- Bluestein/Rader wisdom production calibrator.
 *
 * Sweeps the prime cells in the production bench grid and writes
 * vfft_wisdom_tuned_bluestein.txt with per-(N, K) optimal (M, B).
 *
 * For each prime N:
 *   - If N-1 factors smoothly into the radix set: Rader path. M is fixed
 *     at N-1; only B is tuned. Sweep B in {4, 16, 32, 64, 128, 256} (capped
 *     by K, must divide K).
 *   - Else: Bluestein path. Both M and B are tuned. M ranges over factorable
 *     values in [2N-1, 4N]; B same as Rader.
 *
 * Each (M, B) pair is built and timed for ~0.3s, then the best per-cell
 * (M, B) is recorded.
 *
 * Usage:
 *   calibrate.exe                                 -- bench grid, default output
 *   calibrate.exe --output PATH                   -- override output path
 *   calibrate.exe --primes 47,59,83 --Ks 4,256    -- subset for quick tests
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

/* MSVC accepts __restrict (single underscore) but not __restrict__ (GCC).
 * Production builds use ICX which understands the GCC form natively. */
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER) && !defined(__INTEL_LLVM_COMPILER)
  #define __restrict__ __restrict
#endif

#include "compat.h"
#include "planner.h"
#include "env.h"
#include "bluestein.h"
#include "bluestein_wisdom.h"
#include "rader.h"

/* ── radix set (matches the production planner) ──────────────── */
static int is_radix_smooth(int n) {
    static const int primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 0};
    for (const int *p = primes; *p; p++)
        while (n % *p == 0) n /= *p;
    return n == 1;
}
static int is_prime(int n) {
    if (n < 2) return 0;
    if (n == 2) return 1;
    if ((n & 1) == 0) return 0;
    for (int p = 3; (long long)p * p <= n; p += 2)
        if (n % p == 0) return 0;
    return 1;
}

/* ── M-candidate enumeration (Bluestein only) ────────────────── */
static int enumerate_m_candidates(int N, int *out, int max_out) {
    int m_min = 2 * N - 1;
    int m_max = 4 * N;
    int n = 0;
    for (int m = m_min; m <= m_max && n < max_out; m++) {
        if (is_radix_smooth(m)) out[n++] = m;
    }
    return n;
}

/* ── factorization string ────────────────────────────────────── */
static void factorization_str(int m, char *buf, size_t buflen) {
    static const int radixes[] = {64, 32, 25, 20, 16, 12, 10, 8, 7, 6, 5, 4, 3, 2,
                                   19, 17, 13, 11, 0};
    size_t pos = 0;
    int first = 1;
    int n = m;
    for (const int *r = radixes; *r && n > 1; r++) {
        while (n % *r == 0) {
            int w = snprintf(buf + pos, buflen - pos, "%s%d",
                             first ? "" : "x", *r);
            if (w < 0 || (size_t)w >= buflen - pos) return;
            pos += (size_t)w;
            first = 0;
            n /= *r;
        }
    }
}

/* ── single-(M, B) Bluestein measurement ─────────────────────── *
 * Multi-trial-min: do n_trials independent timing runs (one warmup,
 * many trials per build) and return the minimum. Mean / median bias
 * upward when occasional samples hit cache pollution / scheduler
 * preemption / thermal blips; min reflects the true peak-of-CPU and
 * eliminates outliers without false-flooring. */
static double bench_bluestein(int N, size_t K, int M, size_t B,
                              stride_registry_t *reg, stride_wisdom_t *wis,
                              double *re, double *im,
                              double per_trial_budget, int n_trials)
{
    stride_plan_t *inner = stride_wise_plan(M, B, reg, wis);
    if (!inner) return -1.0;
    stride_plan_t *plan = stride_bluestein_plan(N, K, B, inner, M);
    if (!plan) { stride_plan_destroy(inner); return -1.0; }

    /* Warm up once before all trials -- caches, branch predictors. */
    for (int w = 0; w < 5; w++) stride_execute_fwd(plan, re, im);

    double best_ns = 1e30;
    for (int trial = 0; trial < n_trials; trial++) {
        double t0 = now_ns();
        stride_execute_fwd(plan, re, im);
        double sample = now_ns() - t0;
        int reps = (sample > 0) ? (int)(per_trial_budget * 1e9 / sample) : 1000;
        if (reps < 20)        reps = 20;
        if (reps > 200000)    reps = 200000;

        double ts = now_ns();
        for (int r = 0; r < reps; r++) stride_execute_fwd(plan, re, im);
        double te = now_ns();

        double trial_ns = (te - ts) / reps;
        if (trial_ns < best_ns) best_ns = trial_ns;
    }

    stride_plan_destroy(plan);
    return best_ns;
}

/* ── single-B Rader measurement (M is fixed at N-1) ──────────── */
static double bench_rader(int N, size_t K, size_t B,
                          stride_registry_t *reg, stride_wisdom_t *wis,
                          double *re, double *im,
                          double per_trial_budget, int n_trials)
{
    int nm1 = N - 1;
    stride_plan_t *inner = stride_wise_plan(nm1, B, reg, wis);
    if (!inner) return -1.0;
    stride_plan_t *plan = stride_rader_plan(N, K, B, inner);
    if (!plan) { stride_plan_destroy(inner); return -1.0; }

    for (int w = 0; w < 5; w++) stride_execute_fwd(plan, re, im);

    double best_ns = 1e30;
    for (int trial = 0; trial < n_trials; trial++) {
        double t0 = now_ns();
        stride_execute_fwd(plan, re, im);
        double sample = now_ns() - t0;
        int reps = (sample > 0) ? (int)(per_trial_budget * 1e9 / sample) : 1000;
        if (reps < 20)        reps = 20;
        if (reps > 200000)    reps = 200000;

        double ts = now_ns();
        for (int r = 0; r < reps; r++) stride_execute_fwd(plan, re, im);
        double te = now_ns();

        double trial_ns = (te - ts) / reps;
        if (trial_ns < best_ns) best_ns = trial_ns;
    }

    stride_plan_destroy(plan);
    return best_ns;
}

/* ── per-cell calibration ────────────────────────────────────── */
typedef struct {
    int N; size_t K;
    int is_rader;     /* 1 = Rader path (M fixed at N-1), 0 = Bluestein */
    int M_best;
    size_t B_best;
    double ns_best;
    int n_candidates_tried;
    char fact[64];
} cell_result_t;

/* B candidates in priority order; calibrator filters to those that divide K. */
static const size_t B_CANDIDATES[] = {16, 32, 64, 128, 256, 4, 8};
static const int N_B_CANDIDATES = sizeof(B_CANDIDATES) / sizeof(B_CANDIDATES[0]);

static void calibrate_cell(int N, size_t K,
                           stride_registry_t *reg, stride_wisdom_t *wis,
                           double *re, double *im,
                           double per_trial_budget, int n_trials,
                           cell_result_t *out)
{
    out->N = N; out->K = K;
    out->ns_best = 1e30;
    out->M_best = 0; out->B_best = 0;
    out->n_candidates_tried = 0;

    int is_rader = is_prime(N) && is_radix_smooth(N - 1);
    out->is_rader = is_rader;

    if (is_rader) {
        /* Rader: M = N-1 is fixed. Sweep B only. */
        int M = N - 1;
        out->M_best = M;
        for (int bi = 0; bi < N_B_CANDIDATES; bi++) {
            size_t B = B_CANDIDATES[bi];
            if (B > K) continue;
            if (K % B != 0) continue;
            double ns = bench_rader(N, K, B, reg, wis, re, im,
                                    per_trial_budget, n_trials);
            if (ns > 0) {
                out->n_candidates_tried++;
                if (ns < out->ns_best) {
                    out->ns_best = ns;
                    out->B_best = B;
                }
            }
        }
        factorization_str(M, out->fact, sizeof(out->fact));
    } else {
        /* Bluestein: enumerate M candidates in [2N-1, 4N]. */
        int M_set[1024];
        int n_M = enumerate_m_candidates(N, M_set, 1024);
        for (int mi = 0; mi < n_M; mi++) {
            int M = M_set[mi];
            for (int bi = 0; bi < N_B_CANDIDATES; bi++) {
                size_t B = B_CANDIDATES[bi];
                if (B > K) continue;
                if (K % B != 0) continue;
                double ns = bench_bluestein(N, K, M, B, reg, wis, re, im,
                                            per_trial_budget, n_trials);
                if (ns > 0) {
                    out->n_candidates_tried++;
                    if (ns < out->ns_best) {
                        out->ns_best = ns;
                        out->M_best = M;
                        out->B_best = B;
                    }
                }
            }
        }
        factorization_str(out->M_best, out->fact, sizeof(out->fact));
    }
}

/* ── prime grid (matches production bench's [override] cells) ── */
static const int DEFAULT_PRIMES[] = {
    47, 59, 83, 107, 127, 167, 179, 251, 257, 263, 311,
    401, 641, 1009, 2801, 4001
};
static const int N_DEFAULT_PRIMES =
    sizeof(DEFAULT_PRIMES) / sizeof(DEFAULT_PRIMES[0]);
static const size_t DEFAULT_KS[] = {4, 32, 256};
static const int N_DEFAULT_KS =
    sizeof(DEFAULT_KS) / sizeof(DEFAULT_KS[0]);

/* ── main ────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    const char *output_path = "vfft_wisdom_tuned_bluestein.txt";
    const int *primes = DEFAULT_PRIMES;
    int n_primes = N_DEFAULT_PRIMES;
    const size_t *Ks = DEFAULT_KS;
    int n_Ks = N_DEFAULT_KS;
    /* Default measurement: 3 trials × 0.15s/trial, take min.
     * Min-of-N filters out cache pollution and scheduler-preemption
     * outliers without false-flooring. */
    double per_trial_budget = 0.15;
    int n_trials = 3;
    int cooldown_ms = 3000;   /* between cells, lets thermal recover */

    /* Custom buffers for --primes / --Ks parsing */
    static int  user_primes[64];
    static size_t user_Ks[16];

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--output") && i + 1 < argc) {
            output_path = argv[++i];
        } else if (!strcmp(argv[i], "--primes") && i + 1 < argc) {
            char *s = argv[++i];
            int n = 0;
            char *tok = strtok(s, ",");
            while (tok && n < 64) { user_primes[n++] = atoi(tok); tok = strtok(NULL, ","); }
            primes = user_primes; n_primes = n;
        } else if (!strcmp(argv[i], "--Ks") && i + 1 < argc) {
            char *s = argv[++i];
            int n = 0;
            char *tok = strtok(s, ",");
            while (tok && n < 16) { user_Ks[n++] = (size_t)atoll(tok); tok = strtok(NULL, ","); }
            Ks = user_Ks; n_Ks = n;
        } else if (!strcmp(argv[i], "--budget") && i + 1 < argc) {
            per_trial_budget = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--trials") && i + 1 < argc) {
            n_trials = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--cooldown-ms") && i + 1 < argc) {
            cooldown_ms = atoi(argv[++i]);
        }
    }

    fprintf(stderr, "[calibrate] grid: %d primes x %d Ks = %d cells\n",
            n_primes, n_Ks, n_primes * n_Ks);
    fprintf(stderr, "[calibrate] per-trial budget: %.2fs, n_trials=%d (min-of-%d)\n",
            per_trial_budget, n_trials, n_trials);
    fprintf(stderr, "[calibrate] cooldown between cells: %d ms\n", cooldown_ms);
    fprintf(stderr, "[calibrate] output: %s\n", output_path);

    stride_set_num_threads(1);

    stride_registry_t reg;
    stride_registry_init(&reg);
    stride_wisdom_t wis;
    stride_wisdom_init(&wis);
    int wrc = stride_wisdom_load(&wis,
        "C:/Users/Tugbars/Desktop/highSpeedFFT/build_tuned/vfft_wisdom_tuned.txt");
    fprintf(stderr, "[calibrate] stride wisdom load rc=%d, %d entries\n",
            wrc, wis.count);

    /* Allocate buffer for the largest (N, K) pair. */
    size_t max_NK = 0;
    for (int p = 0; p < n_primes; p++)
        for (int k = 0; k < n_Ks; k++) {
            size_t nk = (size_t)primes[p] * Ks[k];
            if (nk > max_NK) max_NK = nk;
        }
    double *re = (double *)STRIDE_ALIGNED_ALLOC(64, max_NK * sizeof(double));
    double *im = (double *)STRIDE_ALIGNED_ALLOC(64, max_NK * sizeof(double));
    if (!re || !im) { fprintf(stderr, "alloc failed\n"); return 1; }
    srand(42);
    for (size_t i = 0; i < max_NK; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    /* Bluestein wisdom we'll fill in. */
    bluestein_wisdom_t bw;
    bluestein_wisdom_init(&bw);

    int total_cells = n_primes * n_Ks;
    int cell_idx = 0;
    double t_grid_start = now_ns();

    printf("\n%-5s %-6s %-7s %-5s %-5s %-19s %12s %s\n",
           "N", "K", "algo", "M", "B", "factorization", "ns", "tried");
    printf("---------------------------------------------------------------------------------\n");

    for (int p = 0; p < n_primes; p++) {
        int N = primes[p];
        if (!is_prime(N)) {
            fprintf(stderr, "warn: N=%d is not prime, skipping\n", N);
            continue;
        }
        for (int k = 0; k < n_Ks; k++) {
            size_t K = Ks[k];
            cell_idx++;

            /* Cooldown before each cell (skip before the first one).
             * Lets thermal headroom recover and L3 / heap state quiesce
             * after the prior cell's plan-churn. */
            if (cell_idx > 1 && cooldown_ms > 0) {
                Sleep((DWORD)cooldown_ms);
            }

            double t_start = now_ns();
            cell_result_t r;
            calibrate_cell(N, K, &reg, &wis, re, im,
                           per_trial_budget, n_trials, &r);
            double t_end = now_ns();

            if (r.M_best > 0 && r.ns_best < 1e29) {
                bluestein_wisdom_add(&bw, N, K, r.M_best, r.B_best, r.ns_best);
                printf("%-5d %-6zu %-7s %-5d %-5zu %-19s %12.0f %d (%.1fs)\n",
                       N, K, r.is_rader ? "RADER" : "BLUE",
                       r.M_best, r.B_best, r.fact, r.ns_best,
                       r.n_candidates_tried, (t_end - t_start) / 1e9);
            } else {
                printf("%-5d %-6zu %-7s SKIP -- no valid (M,B) found\n",
                       N, K, r.is_rader ? "RADER" : "BLUE");
            }
            fflush(stdout);

            /* Periodic save: persist progress every 5 cells in case the run is
             * interrupted -- losing 2 hours of calibration to a crash is bad. */
            if (cell_idx % 5 == 0 || cell_idx == total_cells) {
                bluestein_wisdom_save(&bw, output_path);
            }
        }
    }

    double t_grid_end = now_ns();
    fprintf(stderr,
            "\n[calibrate] DONE in %.1fs total. %d entries written to %s\n",
            (t_grid_end - t_grid_start) / 1e9, bw.count, output_path);

    return 0;
}
