/**
 * stride_exhaustive.h — Exhaustive factorization search for stride executor
 *
 * For a given N and K, enumerates ALL valid factorizations into available
 * radixes, tries ALL orderings of each, benchmarks each combination,
 * and returns the fastest.
 *
 * This is the ground truth for evaluating heuristic factorizers.
 * Run once per (N, K) pair, cache result in wisdom file.
 *
 * Usage:
 *   stride_registry_t reg;
 *   stride_registry_init(&reg);
 *   stride_factorization_t best;
 *   double best_ns = stride_exhaustive_search(N, K, &reg, &best, re, im);
 */
#ifndef STRIDE_EXHAUSTIVE_H
#define STRIDE_EXHAUSTIVE_H

#include "stride_registry.h"
#include "stride_factorizer.h"  /* for stride_factorization_t, stride_cpu_info_t */
#include "../bench_compat.h"    /* for now_ns() */

#include <string.h>
#include <stdio.h>
#include <fftw3.h>

/* ═══════════════════════════════════════════════════════════════
 * ENUMERATE ALL VALID FACTORIZATIONS
 *
 * Recursive: at each level, try every available radix that divides
 * remaining, recurse on remaining/R. Collect all complete factorizations.
 * ═══════════════════════════════════════════════════════════════ */

#define EXHAUST_MAX_RESULTS 512

typedef struct {
    stride_factorization_t results[EXHAUST_MAX_RESULTS];
    int count;
} factorization_list_t;

static void _enumerate_factorizations(int remaining, const stride_registry_t *reg,
                                      int *current, int depth,
                                      factorization_list_t *list) {
    if (remaining == 1) {
        if (depth > 0 && list->count < EXHAUST_MAX_RESULTS) {
            stride_factorization_t *f = &list->results[list->count];
            f->nfactors = depth;
            memcpy(f->factors, current, depth * sizeof(int));
            list->count++;
        }
        return;
    }
    if (depth >= FACT_MAX_STAGES) return;

    for (const int *rp = STRIDE_AVAILABLE_RADIXES; *rp; rp++) {
        int R = *rp;
        if (remaining % R != 0) continue;
        if (!stride_registry_has(reg, R)) continue;

        /* Avoid duplicate decompositions: only allow non-increasing order
         * in the base decomposition. Permutations are handled separately. */
        if (depth > 0 && R > current[depth - 1]) continue;

        current[depth] = R;
        _enumerate_factorizations(remaining / R, reg, current, depth + 1, list);
    }
}

static void stride_enumerate_factorizations(int N, const stride_registry_t *reg,
                                            factorization_list_t *list) {
    list->count = 0;
    int current[FACT_MAX_STAGES];
    _enumerate_factorizations(N, reg, current, 0, list);
}

/* ═══════════════════════════════════════════════════════════════
 * GENERATE ALL PERMUTATIONS OF A FACTORIZATION
 * ═══════════════════════════════════════════════════════════════ */

#define EXHAUST_MAX_PERMS 720  /* 6! = 720, max for nf=6 */

typedef struct {
    int perms[EXHAUST_MAX_PERMS][FACT_MAX_STAGES];
    int count;
    int nf;
} permutation_list_t;

/* Sort ascending (for next-permutation algorithm) */
static void _sort_int(int *arr, int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i]; int j = i - 1;
        while (j >= 0 && arr[j] > key) { arr[j+1] = arr[j]; j--; }
        arr[j+1] = key;
    }
}

/* Reverse arr[l..r] in-place */
static void _reverse_int(int *arr, int l, int r) {
    while (l < r) { int t = arr[l]; arr[l] = arr[r]; arr[r] = t; l++; r--; }
}

/* Standard next-permutation (handles duplicates correctly) */
static int _next_perm(int *arr, int n) {
    int i = n - 2;
    while (i >= 0 && arr[i] >= arr[i+1]) i--;
    if (i < 0) return 0; /* last permutation */
    int j = n - 1;
    while (arr[j] <= arr[i]) j--;
    int t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    _reverse_int(arr, i+1, n-1);
    return 1;
}

static void stride_gen_permutations(const int *factors, int nf, permutation_list_t *list) {
    list->count = 0;
    list->nf = nf;
    int work[FACT_MAX_STAGES];
    memcpy(work, factors, nf * sizeof(int));
    _sort_int(work, nf); /* start from sorted (smallest permutation) */
    do {
        if (list->count >= EXHAUST_MAX_PERMS) break;
        memcpy(list->perms[list->count], work, nf * sizeof(int));
        list->count++;
    } while (_next_perm(work, nf));
}

/* ═══════════════════════════════════════════════════════════════
 * BENCHMARK A SINGLE FACTORIZATION
 *
 * Creates a plan, warms up, times multiple runs, returns best ns.
 * Uses caller-provided aligned buffers (avoids alloc in hot loop).
 * ═══════════════════════════════════════════════════════════════ */

/* Bench with explicit t1 arrays (allows caller to mix flat/log3) */
static double stride_bench_one_ex(int N, size_t K, const int *factors, int nf,
                                  stride_n1_fn *n1f, stride_n1_fn *n1b,
                                  stride_t1_fn *t1f, stride_t1_fn *t1b,
                                  double *re, double *im, double *orig_re, double *orig_im) {
    size_t total = (size_t)N * K;

    stride_plan_t *plan = stride_plan_create(N, K, factors, nf, n1f, n1b, t1f, t1b);
    if (!plan) return 1e18;

    /* Warm up */
    int reps = (int)(1e5 / (total + 1));
    if (reps < 10) reps = 10;
    if (reps > 50000) reps = 50000;

    for (int i = 0; i < 3; i++) {
        memcpy(re, orig_re, total * sizeof(double));
        memcpy(im, orig_im, total * sizeof(double));
        stride_execute_fwd(plan, re, im);
    }

    /* Benchmark: best of 3 trials */
    double best = 1e18;
    for (int t = 0; t < 3; t++) {
        memcpy(re, orig_re, total * sizeof(double));
        memcpy(im, orig_im, total * sizeof(double));
        double t0 = now_ns();
        for (int i = 0; i < reps; i++)
            stride_execute_fwd(plan, re, im);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }

    stride_plan_destroy(plan);
    return best;
}

/* Convenience wrapper: builds codelet arrays from registry using heuristic log3 selection */
static double stride_bench_one(int N, size_t K, const int *factors, int nf,
                               const stride_registry_t *reg,
                               double *re, double *im, double *orig_re, double *orig_im) {
    stride_n1_fn n1f[FACT_MAX_STAGES], n1b[FACT_MAX_STAGES];
    stride_t1_fn t1f[FACT_MAX_STAGES], t1b[FACT_MAX_STAGES];
    for (int s = 0; s < nf; s++) {
        int R = factors[s];
        if (!reg->n1_fwd[R] || !reg->n1_bwd[R]) return 1e18;
        n1f[s] = reg->n1_fwd[R];
        n1b[s] = reg->n1_bwd[R];
        t1f[s] = stride_select_t1_fwd(R, K, reg);
        t1b[s] = stride_select_t1_bwd(R, K, reg);
    }
    return stride_bench_one_ex(N, K, factors, nf, n1f, n1b, t1f, t1b,
                               re, im, orig_re, orig_im);
}

/* ═══════════════════════════════════════════════════════════════
 * EXHAUSTIVE SEARCH
 *
 * For a given (N, K):
 *   1. Enumerate all valid factorizations of N
 *   2. For each, generate all unique permutations
 *   3. For each permutation, try all flat/log3 combinations per stage
 *   4. Benchmark each candidate, return the best
 *
 * Log3 combinations: for nf stages with s twiddled stages,
 * try 2^s variants (flat/log3 per twiddled stage).
 * Stage 0 is never twiddled. Max s = nf-1.
 * For nf=3: 4 log3 combos. For nf=4: 8. Manageable.
 *
 * verbose: 0=silent, 1=summary
 * ═══════════════════════════════════════════════════════════════ */

static double stride_exhaustive_search(int N, size_t K,
                                       const stride_registry_t *reg,
                                       stride_factorization_t *best_fact,
                                       int verbose) {
    size_t total = (size_t)N * K;

    double *re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *orig_re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *orig_im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    for (size_t i = 0; i < total; i++) {
        orig_re[i] = (double)rand() / RAND_MAX - 0.5;
        orig_im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    factorization_list_t *flist = (factorization_list_t *)malloc(sizeof(*flist));
    stride_enumerate_factorizations(N, reg, flist);

    if (verbose >= 1)
        printf("  N=%d K=%zu: %d unique decompositions\n", N, K, flist->count);

    double global_best_ns = 1e18;
    int total_candidates = 0;
    int best_log3_mask = 0;

    for (int fi = 0; fi < flist->count; fi++) {
        const stride_factorization_t *f = &flist->results[fi];

        permutation_list_t *plist = (permutation_list_t *)malloc(sizeof(*plist));
        stride_gen_permutations(f->factors, f->nfactors, plist);

        for (int pi = 0; pi < plist->count; pi++) {
            /* Verify product */
            { int prod = 1; for (int s = 0; s < f->nfactors; s++) prod *= plist->perms[pi][s];
              if (prod != N) continue; }

            const int *factors_p = plist->perms[pi];
            int nf = f->nfactors;

            /* Count how many twiddled stages have log3 available */
            int n_log3_stages = 0;
            int log3_stage_idx[FACT_MAX_STAGES];
            for (int s = 1; s < nf; s++) {
                if (reg->t1_fwd_log3[factors_p[s]])
                    log3_stage_idx[n_log3_stages++] = s;
            }

            /* Try all 2^n_log3_stages combinations */
            int n_combos = 1 << n_log3_stages;
            for (int combo = 0; combo < n_combos; combo++) {
                stride_n1_fn n1f[FACT_MAX_STAGES], n1b[FACT_MAX_STAGES];
                stride_t1_fn t1f[FACT_MAX_STAGES], t1b[FACT_MAX_STAGES];

                for (int s = 0; s < nf; s++) {
                    int R = factors_p[s];
                    if (!reg->n1_fwd[R]) goto skip_combo;
                    n1f[s] = reg->n1_fwd[R];
                    n1b[s] = reg->n1_bwd[R];
                    t1f[s] = reg->t1_fwd[R];  /* default: flat */
                    t1b[s] = reg->t1_bwd[R];
                }

                /* Apply log3 where this combo says to */
                for (int li = 0; li < n_log3_stages; li++) {
                    if (combo & (1 << li)) {
                        int s = log3_stage_idx[li];
                        int R = factors_p[s];
                        t1f[s] = reg->t1_fwd_log3[R];
                        t1b[s] = reg->t1_bwd_log3[R];
                    }
                }

                {
                    double ns = stride_bench_one_ex(N, K, factors_p, nf,
                                                    n1f, n1b, t1f, t1b,
                                                    re, im, orig_re, orig_im);
                    total_candidates++;

                    if (ns < global_best_ns) {
                        global_best_ns = ns;
                        best_fact->nfactors = nf;
                        memcpy(best_fact->factors, factors_p, nf * sizeof(int));
                        best_log3_mask = combo;
                    }
                }
                continue;
                skip_combo:;
            }
        }
        free(plist);
    }

    if (verbose >= 1) {
        printf("  Best: ");
        for (int s = 0; s < best_fact->nfactors; s++)
            printf("%s%d", s ? "x" : "", best_fact->factors[s]);
        if (best_log3_mask)
            printf(" (log3 mask=%d)", best_log3_mask);
        printf(" = %.1f ns (%d total candidates)\n", global_best_ns, total_candidates);
    }

    free(flist);
    STRIDE_ALIGNED_FREE(re);
    STRIDE_ALIGNED_FREE(im);
    STRIDE_ALIGNED_FREE(orig_re);
    STRIDE_ALIGNED_FREE(orig_im);

    return global_best_ns;
}

/* ═══════════════════════════════════════════════════════════════
 * COMPARE: HEURISTIC vs EXHAUSTIVE
 *
 * Runs both, reports the heuristic's factorization, the exhaustive
 * best, and the ratio (how close the heuristic gets to optimal).
 * ═══════════════════════════════════════════════════════════════ */

static void stride_compare_strategies(int N, size_t K,
                                      const stride_registry_t *reg) {
    printf("\n== N=%d  K=%zu ==\n", N, K);

    /* Heuristic */
    stride_factorization_t heur_fact;
    stride_factorize(N, K, reg, &heur_fact);

    printf("  Heuristic: ");
    for (int s = 0; s < heur_fact.nfactors; s++)
        printf("%s%d", s ? "x" : "", heur_fact.factors[s]);

    /* Score */
    stride_cpu_info_t cpu = stride_detect_cpu();
    double heur_score = stride_score_factorization(heur_fact.factors, heur_fact.nfactors, K, N, &cpu);
    printf(" (score=%.0f)\n", heur_score);

    /* Bench heuristic */
    size_t total = (size_t)N * K;
    double *re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *orig_re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *orig_im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    for (size_t i = 0; i < total; i++) {
        orig_re[i] = (double)rand() / RAND_MAX - 0.5;
        orig_im[i] = (double)rand() / RAND_MAX - 0.5;
    }
    double heur_ns = stride_bench_one(N, K, heur_fact.factors, heur_fact.nfactors,
                                      reg, re, im, orig_re, orig_im);
    STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);
    STRIDE_ALIGNED_FREE(orig_re); STRIDE_ALIGNED_FREE(orig_im);

    printf("  Heuristic time: %.1f ns\n", heur_ns);

    /* Exhaustive */
    stride_factorization_t exh_fact;
    double exh_ns = stride_exhaustive_search(N, K, reg, &exh_fact, 1);

    /* FFTW reference */
    fflush(stdout);
    double *fr = (double *)fftw_malloc(total * sizeof(double));
    double *fi = (double *)fftw_malloc(total * sizeof(double));
    double *fo = (double *)fftw_malloc(total * sizeof(double));
    double *fo2 = (double *)fftw_malloc(total * sizeof(double));
    for (size_t i = 0; i < total; i++) { fr[i] = (double)rand()/RAND_MAX - 0.5; fi[i] = (double)rand()/RAND_MAX - 0.5; }
    fftw_iodim dim = {.n = N, .is = (int)K, .os = (int)K};
    fftw_iodim howm = {.n = (int)K, .is = 1, .os = 1};
    fftw_plan fp = fftw_plan_guru_split_dft(1, &dim, 1, &howm, fr, fi, fo, fo2, FFTW_MEASURE);
    for (size_t i = 0; i < total; i++) { fr[i] = (double)rand()/RAND_MAX - 0.5; fi[i] = (double)rand()/RAND_MAX - 0.5; }
    int reps = (int)(2e5 / (total + 1)); if (reps < 20) reps = 20; if (reps > 100000) reps = 100000;
    for (int i = 0; i < 10; i++) fftw_execute(fp);
    double fftw_best = 1e18;
    for (int t = 0; t < 5; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++) fftw_execute_split_dft(fp, fr, fi, fo, fo2);
        double ns = (now_ns() - t0) / reps;
        if (ns < fftw_best) fftw_best = ns;
    }
    fftw_destroy_plan(fp); fftw_free(fr); fftw_free(fi); fftw_free(fo); fftw_free(fo2);

    /* Compare */
    double ratio = heur_ns / exh_ns;
    double vs_fftw = fftw_best / exh_ns;
    printf("  FFTW_MEASURE: %.1f ns\n", fftw_best);
    printf("  Heuristic/Exhaustive: %.2fx (%s)\n",
           ratio, ratio < 1.05 ? "OPTIMAL" : ratio < 1.20 ? "GOOD" : "NEEDS TUNING");
    printf("  Best vs FFTW: %.2fx\n", vs_fftw);
}

#endif /* STRIDE_EXHAUSTIVE_H */
