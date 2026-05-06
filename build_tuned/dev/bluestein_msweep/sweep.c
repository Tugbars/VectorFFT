/* sweep.c — Bluestein (M, B) parameter sweep harness
 *
 * For a given (N, K), enumerates every M in [2N-1, 4N] that factors into
 * the available radix set, sweeps a small set of B values per M, builds a
 * Bluestein plan with each (M, B), times forward execution, prints a
 * sorted table.
 *
 * Goal: empirically compare M-selection strategies for the v1.0 Bluestein
 * wisdom track:
 *   - Strategy A (current heuristic): _bluestein_choose_m
 *   - Strategy B (pow2 + smallest smooth composite): 2 candidates
 *   - Strategy C (full enumeration in [2N-1, 4N]): all factorable
 *
 * Reaches into stride_bluestein_plan directly (the planner's internal
 * entry point that takes M as a parameter), bypassing _bluestein_choose_m.
 *
 * Usage:
 *   sweep.exe N K              — single cell
 *   sweep.exe N1 K1 N2 K2 ...  — multiple cells
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

#include "compat.h"       /* now_ns + STRIDE_ALIGNED_ALLOC */
#include "planner.h"      /* pulls in registry, executor, bluestein, wisdom */
#include "env.h"
#include "bluestein.h"    /* explicit for stride_bluestein_plan + helpers */

/* ── factorability + stage count (matches bluestein.h) ─────────── */
static int sweep_is_factorable(int m) {
    static const int primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 0};
    for (const int *p = primes; *p; p++)
        while (m % *p == 0) m /= *p;
    return m == 1;
}
static int sweep_count_stages(int m) {
    static const int radixes[] = {64, 32, 25, 20, 16, 12, 10, 8, 7, 6, 5, 4, 3, 2,
                                   19, 17, 13, 11, 0};
    int stages = 0;
    for (const int *r = radixes; *r && m > 1; r++)
        while (m % *r == 0) { m /= *r; stages++; }
    return (m == 1) ? stages : 999;
}
static void factorization_str(int m, char *buf, size_t buflen) {
    static const int radixes[] = {64, 32, 25, 20, 16, 12, 10, 8, 7, 6, 5, 4, 3, 2,
                                   19, 17, 13, 11, 0};
    size_t pos = 0;
    int first = 1;
    for (const int *r = radixes; *r && m > 1; r++) {
        while (m % *r == 0) {
            int n = snprintf(buf + pos, buflen - pos, "%s%d", first ? "" : "x", *r);
            if (n < 0 || (size_t)n >= buflen - pos) return;
            pos += (size_t)n;
            first = 0;
            m /= *r;
        }
    }
}

/* ── single (M, B) measurement ─────────────────────────────────── */
typedef struct {
    int M, stages;
    size_t B;
    double ns;
    char fact[64];
} result_t;

static double bench_one(int N, size_t K, int M, size_t B,
                        stride_registry_t *reg, stride_wisdom_t *wis,
                        double *re, double *im)
{
    /* Inner M-point FFT plan at K=B uses existing stride wisdom for that
     * (M, B) — falls back to auto if missing. */
    stride_plan_t *inner = stride_wise_plan(M, B, reg, wis);
    if (!inner) return -1.0;

    stride_plan_t *plan = stride_bluestein_plan(N, K, B, inner, M);
    if (!plan) { stride_plan_destroy(inner); return -2.0; }

    /* Warmup */
    for (int w = 0; w < 5; w++) stride_execute_fwd(plan, re, im);

    /* Auto-rep to ~0.5 sec/cell */
    double t0 = now_ns();
    stride_execute_fwd(plan, re, im);
    double sample_ns = now_ns() - t0;
    int reps = (sample_ns > 0) ? (int)(0.5 * 1e9 / sample_ns) : 1000;
    if (reps < 30)        reps = 30;
    if (reps > 200000)    reps = 200000;

    double t_start = now_ns();
    for (int r = 0; r < reps; r++) stride_execute_fwd(plan, re, im);
    double t_end = now_ns();

    stride_plan_destroy(plan);
    return (t_end - t_start) / reps;
}

static int cmp_ns(const void *a, const void *b) {
    double na = ((const result_t *)a)->ns;
    double nb = ((const result_t *)b)->ns;
    return (na < nb) ? -1 : (na > nb) ? 1 : 0;
}

static void sweep_cell(int N, size_t K,
                       stride_registry_t *reg, stride_wisdom_t *wis,
                       double *re, double *im)
{
    int m_min = 2 * N - 1;
    int m_max = 4 * N;
    int M_pow2 = 1; while (M_pow2 < m_min) M_pow2 *= 2;
    int M_smooth = 0;
    for (int m = m_min; m <= m_max; m++) {
        if (sweep_is_factorable(m)) { M_smooth = m; break; }
    }
    int M_heuristic = _bluestein_choose_m(N);

    size_t B_candidates[] = {16, 32, 64, 128, 256};
    int n_B = sizeof(B_candidates) / sizeof(B_candidates[0]);

    result_t *results = (result_t *)calloc(2048, sizeof(result_t));
    int n_results = 0;

    for (int m = m_min; m <= m_max; m++) {
        if (!sweep_is_factorable(m)) continue;
        for (int bi = 0; bi < n_B; bi++) {
            size_t B = B_candidates[bi];
            if (B > K) continue;
            if (K % B != 0) continue;

            double ns = bench_one(N, K, m, B, reg, wis, re, im);
            if (ns < 0) continue;

            results[n_results].M = m;
            results[n_results].B = B;
            results[n_results].stages = sweep_count_stages(m);
            results[n_results].ns = ns;
            factorization_str(m, results[n_results].fact, sizeof(results[n_results].fact));
            n_results++;
        }
    }

    qsort(results, n_results, sizeof(result_t), cmp_ns);

    printf("\n══════════════════════════════════════════════════════════════════════\n");
    printf("  Sweep: N=%d  K=%zu     candidates: %d (M, B) pairs\n", N, K, n_results);
    printf("══════════════════════════════════════════════════════════════════════\n");
    printf("  Reference Ms:\n");
    printf("    M_heuristic (current code) : %d\n", M_heuristic);
    printf("    M_pow2 (smallest >= 2N-1)  : %d\n", M_pow2);
    printf("    M_smooth (smallest valid)  : %d\n", M_smooth);
    printf("\n");
    printf("  Rank  M     factorization        stages  B     ns         vs_best\n");
    printf("  ----  ----  -------------------  ------  ----  ---------  -------\n");

    double best_ns = (n_results > 0) ? results[0].ns : 1.0;
    for (int i = 0; i < n_results && i < 30; i++) {
        const char *tag = "";
        if (results[i].M == M_heuristic) tag = "  ← heuristic";
        else if (results[i].M == M_pow2 && results[i].M != M_smooth) tag = "  ← pow2";
        else if (results[i].M == M_smooth && results[i].M != M_pow2) tag = "  ← smallest_smooth";

        printf("  %-4d  %-4d  %-19s  %-6d  %-4zu  %9.0f  %5.2fx%s\n",
               i + 1, results[i].M, results[i].fact, results[i].stages,
               results[i].B, results[i].ns, results[i].ns / best_ns, tag);
    }

    /* Per-strategy summary */
    printf("\n  Strategy comparison (best (M,B) per strategy):\n");
    double strat_a_best = 1e30; int strat_a_M = 0; size_t strat_a_B = 0;
    double strat_b_best = 1e30; int strat_b_M = 0; size_t strat_b_B = 0;
    double strat_c_best = 1e30; int strat_c_M = 0; size_t strat_c_B = 0;
    for (int i = 0; i < n_results; i++) {
        if (results[i].M == M_heuristic && results[i].ns < strat_a_best) {
            strat_a_best = results[i].ns; strat_a_M = results[i].M; strat_a_B = results[i].B;
        }
        if ((results[i].M == M_pow2 || results[i].M == M_smooth) && results[i].ns < strat_b_best) {
            strat_b_best = results[i].ns; strat_b_M = results[i].M; strat_b_B = results[i].B;
        }
        if (results[i].ns < strat_c_best) {
            strat_c_best = results[i].ns; strat_c_M = results[i].M; strat_c_B = results[i].B;
        }
    }
    if (strat_a_best < 1e29)
        printf("    A (current heuristic)  : M=%-4d B=%-3zu  %9.0f ns\n",
               strat_a_M, strat_a_B, strat_a_best);
    if (strat_b_best < 1e29)
        printf("    B (pow2 + smooth)      : M=%-4d B=%-3zu  %9.0f ns  %5.2fx vs A\n",
               strat_b_M, strat_b_B, strat_b_best, strat_a_best / strat_b_best);
    if (strat_c_best < 1e29)
        printf("    C (full enumeration)   : M=%-4d B=%-3zu  %9.0f ns  %5.2fx vs A\n",
               strat_c_M, strat_c_B, strat_c_best, strat_a_best / strat_c_best);
    fflush(stdout);

    free(results);
}

int main(int argc, char **argv) {
    if (argc < 3 || argc % 2 == 0) {
        fprintf(stderr, "usage: sweep N K [N K ...]\n");
        return 1;
    }

    /* Single-thread for stable measurements. */
    stride_set_num_threads(1);

    stride_registry_t reg;
    stride_registry_init(&reg);
    stride_wisdom_t wis;
    stride_wisdom_init(&wis);
    int wrc = stride_wisdom_load(&wis,
        "C:/Users/Tugbars/Desktop/highSpeedFFT/build_tuned/vfft_wisdom_tuned.txt");
    fprintf(stderr, "[stride wisdom] load rc=%d, %d entries\n", wrc, wis.count);

    /* Allocate buffer big enough for the largest cell */
    size_t max_NK = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        int N = atoi(argv[i]);
        size_t K = (size_t)atoll(argv[i + 1]);
        if ((size_t)N * K > max_NK) max_NK = (size_t)N * K;
    }
    double *re = (double *)STRIDE_ALIGNED_ALLOC(64, max_NK * sizeof(double));
    double *im = (double *)STRIDE_ALIGNED_ALLOC(64, max_NK * sizeof(double));
    if (!re || !im) { fprintf(stderr, "alloc failed\n"); return 1; }
    srand(42);
    for (size_t i = 0; i < max_NK; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    for (int i = 1; i + 1 < argc; i += 2) {
        int N = atoi(argv[i]);
        size_t K = (size_t)atoll(argv[i + 1]);
        sweep_cell(N, K, &reg, &wis, re, im);
    }

    return 0;
}
