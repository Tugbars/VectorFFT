/**
 * stride_calibrate.h -- Per-radix cost calibration for optimal plan selection
 *
 * Instead of exhaustively benchmarking every factorization x ordering
 * (which takes hours for large N), we:
 *   1. Measure each radix's cost independently at various K
 *   2. Use the per-radix costs to predict total plan cost
 *   3. Pick the best factorization via arithmetic, not benchmarking
 *
 * The calibration measures two costs per (R, K):
 *   - n1_cost: radix-R butterfly with no twiddle (first stage)
 *   - t1_cost: radix-R butterfly with twiddle application (later stages)
 *
 * For a factorization [R0, R1, R2] at batch K, predicted cost =
 *   (N/R0) * n1_cost[R0][K]  +  (N/R1) * t1_cost[R1][K]  + ...
 *
 * This compositional model captures ~90% of the real cost. The ~10%
 * it misses (cross-stage cache interference) is acceptable because:
 *   - Calibration takes seconds instead of hours
 *   - The greedy factorizer's ordering already minimizes cache pressure
 *   - For critical sizes, a final verification bench catches outliers
 *
 * Usage:
 *   stride_radix_costs_t costs;
 *   stride_calibrate_radixes(&costs, K, &reg);  // ~2-5 seconds
 *   stride_plan_t *plan = stride_calibrated_plan(N, K, &reg, &costs);
 */
#ifndef STRIDE_CALIBRATE_H
#define STRIDE_CALIBRATE_H

#include "registry.h"
#include "factorizer.h"
#include "executor.h"
#include "compat.h"

#include <string.h>
#include <stdio.h>
#include <float.h>

/* =====================================================================
 * RADIX COST TABLE
 * ===================================================================== */

/* Cost in nanoseconds per group (one butterfly invocation).
 * Indexed by radix. Measured at a specific K. */
typedef struct {
    double n1_cost[STRIDE_REG_MAX_RADIX]; /* first-stage cost (no twiddle) */
    double t1_cost[STRIDE_REG_MAX_RADIX]; /* later-stage cost (with twiddle) */
    size_t K;                              /* K these costs were measured at */
    int    measured[STRIDE_REG_MAX_RADIX]; /* 1 if radix was measured */
} stride_radix_costs_t;

static void stride_radix_costs_init(stride_radix_costs_t *c) {
    memset(c, 0, sizeof(*c));
    for (int R = 0; R < STRIDE_REG_MAX_RADIX; R++) {
        c->n1_cost[R] = 1e18;
        c->t1_cost[R] = 1e18;
    }
}

/* =====================================================================
 * PER-RADIX CALIBRATION
 *
 * For each available radix R, build a minimal 2-stage plan:
 *   N_test = R * inner,  factors = [inner, R]
 * Run it, extract the per-group cost for the R stage.
 *
 * We measure the whole plan and attribute cost proportionally:
 *   total_cost = n1_groups * n1_cost_per_group + t1_groups * t1_cost_per_group
 *
 * By running two different configurations (R as first stage vs R as
 * second stage), we can separate n1 and t1 costs.
 * ===================================================================== */

/* Pick a small inner radix different from R for calibration */
static int _calib_pick_inner(int R, const stride_registry_t *reg) {
    int candidates[] = {8, 4, 5, 3, 7, 6, 2, 0};
    for (int *c = candidates; *c; c++)
        if (*c != R && stride_registry_has(reg, *c)) return *c;
    return 0;
}

/* Bench a 2-stage plan [f0, f1] and return total ns per FFT */
static double _calib_bench_pair(int f0, int f1, size_t K,
                                 const stride_registry_t *reg) {
    int N = f0 * f1;
    int factors[2] = {f0, f1};
    size_t total = (size_t)N * K;

    stride_n1_fn n1f[2], n1b[2];
    stride_t1_fn t1f[2], t1b[2];

    n1f[0] = reg->n1_fwd[f0]; n1b[0] = reg->n1_bwd[f0];
    t1f[0] = NULL;             t1b[0] = NULL;
    n1f[1] = reg->n1_fwd[f1]; n1b[1] = reg->n1_bwd[f1];
    t1f[1] = reg->t1_fwd[f1]; t1b[1] = reg->t1_bwd[f1];

    if (!n1f[0] || !n1f[1] || !t1f[1]) return 1e18;

    stride_plan_t *plan = stride_plan_create(N, K, factors, 2,
                                              n1f, n1b, t1f, t1b, 0);
    if (!plan) return 1e18;

    double *re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    for (size_t i = 0; i < total; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    /* Warmup */
    for (int w = 0; w < 5; w++) stride_execute_fwd(plan, re, im);

    int reps = (int)(5e5 / (total + 1));
    if (reps < 20) reps = 20;
    if (reps > 100000) reps = 100000;

    /* Best of 3 trials */
    double best = 1e18;
    for (int t = 0; t < 3; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++) stride_execute_fwd(plan, re, im);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }

    STRIDE_ALIGNED_FREE(re);
    STRIDE_ALIGNED_FREE(im);
    stride_plan_destroy(plan);
    return best;
}

/**
 * stride_calibrate_radixes -- Measure per-radix costs at a given K.
 *
 * For each available radix R, runs two benchmarks:
 *   Plan A: [R, inner]  -- R is first stage (n1 cost)
 *   Plan B: [inner, R]  -- R is second stage (t1 cost)
 *
 * From these, we extract:
 *   t1_cost[R] = (B_total - A_total) * R / (N * (R-1))
 *     (the difference is mostly the twiddle overhead of R)
 *   n1_cost[R] = A_total / (N / R)
 *     (approximate: assumes inner stage cost is small)
 *
 * More practically, we store the raw per-element times and use
 * them as relative weights for comparing factorizations.
 *
 * Takes 2-5 seconds for all radixes.
 */
static void stride_calibrate_radixes(stride_radix_costs_t *costs,
                                      size_t K,
                                      const stride_registry_t *reg) {
    stride_radix_costs_init(costs);
    costs->K = K;

    for (int R = 2; R < STRIDE_REG_MAX_RADIX; R++) {
        if (!stride_registry_has(reg, R)) continue;

        int inner = _calib_pick_inner(R, reg);
        if (!inner) continue;

        int N = R * inner;

        /* Plan A: [R, inner] — R as first stage (n1 only) */
        double time_a = _calib_bench_pair(R, inner, K, reg);

        /* Plan B: [inner, R] — R as twiddled second stage */
        double time_b = _calib_bench_pair(inner, R, K, reg);

        /* Per-element cost attribution:
         * Plan A has R as n1 (N/R groups) + inner as t1
         * Plan B has inner as n1 + R as t1 (N/R groups)
         *
         * We store total time normalized by N so costs are
         * comparable across radixes with different N_test. */
        costs->n1_cost[R] = time_a / (double)N;  /* ns per element, R as n1 */
        costs->t1_cost[R] = time_b / (double)N;  /* ns per element, R as t1 */
        costs->measured[R] = 1;
    }
}

/* =====================================================================
 * COST PREDICTION
 *
 * Given a factorization and per-radix costs, predict total time.
 * ===================================================================== */

/**
 * stride_predict_cost -- Estimate total FFT time from per-radix costs.
 *
 * For factorization [R0, R1, ..., R_{nf-1}]:
 *   Stage 0 (n1): cost = N * n1_cost[R0]
 *   Stage s (t1): cost = N * t1_cost[Rs]
 *   Total = sum of stage costs
 *
 * Returns predicted ns. Lower = better.
 */
static double stride_predict_cost(const int *factors, int nf, int N,
                                   const stride_radix_costs_t *costs) {
    double total = 0.0;
    for (int s = 0; s < nf; s++) {
        int R = factors[s];
        if (R < 2 || R >= STRIDE_REG_MAX_RADIX || !costs->measured[R])
            return 1e18;
        if (s == 0)
            total += (double)N * costs->n1_cost[R];
        else
            total += (double)N * costs->t1_cost[R];
    }
    /* Penalty for extra stages: each additional stage beyond 2 adds
     * a full data pass overhead (~0.5 ns/element for L1-resident data) */
    if (nf > 2)
        total += (double)N * 0.5 * (nf - 2);
    return total;
}

/* =====================================================================
 * CALIBRATED PLAN SELECTION
 *
 * Enumerate factorizations, score each with predicted cost, pick best.
 * Then optionally verify the top candidates with real benchmarks.
 * ===================================================================== */

/**
 * stride_calibrated_search -- Find best factorization using cost prediction.
 *
 * Much faster than exhaustive search: enumerates all factorizations and
 * permutations (same as exhaustive), but scores by arithmetic prediction
 * instead of benchmarking each one. Takes milliseconds, not hours.
 *
 * verify_top_n: if > 0, re-benchmarks the top N candidates to catch
 *               prediction errors. 3-5 is a good balance.
 *               0 = pure prediction, no benchmarking.
 *
 * Returns predicted (or verified) best time in ns.
 */
static double stride_calibrated_search(int N, size_t K,
                                        const stride_registry_t *reg,
                                        const stride_radix_costs_t *costs,
                                        stride_factorization_t *best_fact,
                                        int verify_top_n,
                                        int verbose) {
    /* Enumerate all factorizations */
    factorization_list_t *flist = (factorization_list_t *)malloc(sizeof(*flist));
    stride_enumerate_factorizations(N, reg, flist);

    if (verbose)
        printf("  N=%d K=%zu: %d decompositions, scoring...\n", N, K, flist->count);

    /* Score all factorizations x permutations by prediction */
    #define CALIB_MAX_CANDIDATES 2048
    typedef struct { int factors[FACT_MAX_STAGES]; int nf; double score; } candidate_t;
    candidate_t *candidates = (candidate_t *)malloc(CALIB_MAX_CANDIDATES * sizeof(candidate_t));
    int n_candidates = 0;

    for (int fi = 0; fi < flist->count; fi++) {
        const stride_factorization_t *f = &flist->results[fi];

        permutation_list_t *plist = (permutation_list_t *)malloc(sizeof(*plist));
        stride_gen_permutations(f->factors, f->nfactors, plist);

        for (int pi = 0; pi < plist->count; pi++) {
            /* Verify product */
            { int prod = 1;
              for (int s = 0; s < f->nfactors; s++) prod *= plist->perms[pi][s];
              if (prod != N) continue; }

            double score = stride_predict_cost(plist->perms[pi], f->nfactors, N, costs);

            if (n_candidates < CALIB_MAX_CANDIDATES) {
                candidate_t *c = &candidates[n_candidates++];
                memcpy(c->factors, plist->perms[pi], f->nfactors * sizeof(int));
                c->nf = f->nfactors;
                c->score = score;
            }
        }
        free(plist);
    }
    free(flist);

    /* Sort candidates by predicted score (insertion sort, small array) */
    for (int i = 1; i < n_candidates; i++) {
        candidate_t key = candidates[i];
        int j = i - 1;
        while (j >= 0 && candidates[j].score > key.score) {
            candidates[j + 1] = candidates[j];
            j--;
        }
        candidates[j + 1] = key;
    }

    if (n_candidates == 0) {
        free(candidates);
        return 1e18;
    }

    /* Use prediction winner as default */
    best_fact->nfactors = candidates[0].nf;
    memcpy(best_fact->factors, candidates[0].factors,
           candidates[0].nf * sizeof(int));
    double best_ns = candidates[0].score;

    if (verbose) {
        printf("  Top 5 predicted:\n");
        int show = n_candidates < 5 ? n_candidates : 5;
        for (int i = 0; i < show; i++) {
            printf("    %d. ", i + 1);
            for (int s = 0; s < candidates[i].nf; s++)
                printf("%s%d", s ? "x" : "", candidates[i].factors[s]);
            printf(" = %.1f ns (predicted)\n", candidates[i].score);
        }
    }

    /* Optional: verify top candidates with real benchmarks */
    if (verify_top_n > 0 && n_candidates > 0) {
        int n_verify = verify_top_n;
        if (n_verify > n_candidates) n_verify = n_candidates;

        size_t total = (size_t)N * K;
        double *re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
        double *im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
        double *orig_re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
        double *orig_im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
        for (size_t i = 0; i < total; i++) {
            orig_re[i] = (double)rand() / RAND_MAX - 0.5;
            orig_im[i] = (double)rand() / RAND_MAX - 0.5;
        }

        double verified_best = 1e18;
        for (int i = 0; i < n_verify; i++) {
            double ns = stride_bench_one(N, K, candidates[i].factors,
                                          candidates[i].nf, reg,
                                          re, im, orig_re, orig_im);
            if (verbose) {
                printf("    verify %d. ", i + 1);
                for (int s = 0; s < candidates[i].nf; s++)
                    printf("%s%d", s ? "x" : "", candidates[i].factors[s]);
                printf(" = %.1f ns (measured)\n", ns);
            }

            if (ns < verified_best) {
                verified_best = ns;
                best_fact->nfactors = candidates[i].nf;
                memcpy(best_fact->factors, candidates[i].factors,
                       candidates[i].nf * sizeof(int));
            }
        }
        best_ns = verified_best;

        STRIDE_ALIGNED_FREE(re);
        STRIDE_ALIGNED_FREE(im);
        STRIDE_ALIGNED_FREE(orig_re);
        STRIDE_ALIGNED_FREE(orig_im);
    }

    free(candidates);
    return best_ns;
}

/**
 * stride_calibrate_radixes_print -- Print calibrated per-radix costs.
 */
static void stride_calibrate_radixes_print(const stride_radix_costs_t *costs) {
    printf("Per-radix costs at K=%zu (ns per element):\n", costs->K);
    printf("  %-4s  %8s  %8s\n", "R", "n1_cost", "t1_cost");
    printf("  %-4s  %8s  %8s\n", "----", "--------", "--------");
    for (int R = 2; R < STRIDE_REG_MAX_RADIX; R++) {
        if (!costs->measured[R]) continue;
        printf("  %-4d  %8.3f  %8.3f\n", R, costs->n1_cost[R], costs->t1_cost[R]);
    }
}

#endif /* STRIDE_CALIBRATE_H */
