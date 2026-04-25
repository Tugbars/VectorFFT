/**
 * stride_dp_planner.h -- Recursive dynamic programming planner
 *
 * FFTW-style recursive decomposition with memoization.
 * Instead of trying all factorizations x orderings (exponential),
 * we decompose recursively and cache sub-problem solutions.
 *
 * Algorithm:
 *   1. To plan N at batch K, try each valid radix R as first stage
 *   2. Recursively get the best plan for N/R (from cache if available)
 *   3. Build candidate [R, sub_plan...], benchmark the FULL plan
 *   4. Cache the winner (factorization SET) for N
 *   5. Try all orderings of the winning set, benchmark each
 *   6. Store the best ordering as the final plan
 *
 * Complexity: O(S * R) sub-problem benchmarks + O(S!) ordering benchmarks
 *   where S = number of unique sub-sizes (~log N), R = number of radixes (~16)
 *   Total: ~150 benchmarks for N=100000, vs ~61000 for exhaustive.
 *
 * Usage:
 *   stride_dp_context_t ctx;
 *   stride_dp_init(&ctx, K);
 *   stride_factorization_t best;
 *   double ns = stride_dp_plan(&ctx, N, &reg, &best);
 *   stride_dp_destroy(&ctx);
 */
#ifndef STRIDE_DP_PLANNER_H
#define STRIDE_DP_PLANNER_H

#include "registry.h"
#include "factorizer.h"
#include "executor.h"
#include "compat.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* =====================================================================
 * DP CACHE
 * ===================================================================== */

#define DP_CACHE_MAX 512

typedef struct {
    int N;
    int factors[FACT_MAX_STAGES];
    int nfactors;
    double cost_ns;
} dp_entry_t;

typedef struct {
    dp_entry_t entries[DP_CACHE_MAX];
    int count;
    size_t K;

    /* Shared benchmark buffers (allocated for max N*K) */
    double *re, *im, *orig_re, *orig_im;
    size_t buf_total;  /* current buffer size in elements */
    int max_N;         /* largest N we'll plan for */

    /* Statistics */
    int n_benchmarks;
    int n_cache_hits;
} stride_dp_context_t;

static void stride_dp_init(stride_dp_context_t *ctx, size_t K, int max_N) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->K = K;
    ctx->max_N = max_N;
    ctx->buf_total = (size_t)max_N * K;

    ctx->re      = (double *)STRIDE_ALIGNED_ALLOC(64, ctx->buf_total * sizeof(double));
    ctx->im      = (double *)STRIDE_ALIGNED_ALLOC(64, ctx->buf_total * sizeof(double));
    ctx->orig_re = (double *)STRIDE_ALIGNED_ALLOC(64, ctx->buf_total * sizeof(double));
    ctx->orig_im = (double *)STRIDE_ALIGNED_ALLOC(64, ctx->buf_total * sizeof(double));

    srand(42);
    for (size_t i = 0; i < ctx->buf_total; i++) {
        ctx->orig_re[i] = (double)rand() / RAND_MAX - 0.5;
        ctx->orig_im[i] = (double)rand() / RAND_MAX - 0.5;
    }
}

static void stride_dp_destroy(stride_dp_context_t *ctx) {
    STRIDE_ALIGNED_FREE(ctx->re);
    STRIDE_ALIGNED_FREE(ctx->im);
    STRIDE_ALIGNED_FREE(ctx->orig_re);
    STRIDE_ALIGNED_FREE(ctx->orig_im);
    memset(ctx, 0, sizeof(*ctx));
}

/* Cache lookup */
static dp_entry_t *_dp_lookup(stride_dp_context_t *ctx, int N) {
    for (int i = 0; i < ctx->count; i++)
        if (ctx->entries[i].N == N) return &ctx->entries[i];
    return NULL;
}

/* Cache insert */
static dp_entry_t *_dp_insert(stride_dp_context_t *ctx, int N) {
    if (ctx->count >= DP_CACHE_MAX) return NULL;
    dp_entry_t *e = &ctx->entries[ctx->count++];
    memset(e, 0, sizeof(*e));
    e->N = N;
    e->cost_ns = 1e18;
    return e;
}

/* =====================================================================
 * BENCHMARK HELPER
 *
 * Benchmarks a full plan using the context's shared buffers.
 * Lighter than stride_bench_one: fewer warmup/trials for DP
 * (speed matters here since we run ~100+ benchmarks).
 * ===================================================================== */

static double _dp_bench(stride_dp_context_t *ctx, int N,
                         const int *factors, int nf,
                         const stride_registry_t *reg) {
    size_t K = ctx->K;
    size_t total = (size_t)N * K;

    /* Sanity check: all requested radixes must have n1 codelets registered. */
    for (int s = 0; s < nf; s++) {
        if (!reg->n1_fwd[factors[s]]) return 1e18;
    }

    /* Phase 2: route through _stride_build_plan so that codelet-side
     * plan_wisdom drives protocol selection (flat / t1s / DIT-log3) at
     * plan time. This keeps the DP's benchmarks honest: the plan being
     * measured here is the exact same plan shape that stride_auto_plan
     * and stride_wise_plan will produce at deploy time. Without this,
     * DP would compare flat-only plans and might pick factorization A
     * over B even though B wins decisively when its stages can use log3. */
    stride_plan_t *plan = _stride_build_plan(N, K, factors, nf, reg);
    if (!plan) return 1e18;

    /* Warmup */
    memcpy(ctx->re, ctx->orig_re, total * sizeof(double));
    memcpy(ctx->im, ctx->orig_im, total * sizeof(double));
    stride_execute_fwd(plan, ctx->re, ctx->im);

    int reps = (int)(1e5 / (total + 1));
    if (reps < 5) reps = 5;
    if (reps > 20000) reps = 20000;

    /* Best of 2 trials (fast, DP will verify winner later) */
    double best = 1e18;
    for (int t = 0; t < 2; t++) {
        memcpy(ctx->re, ctx->orig_re, total * sizeof(double));
        memcpy(ctx->im, ctx->orig_im, total * sizeof(double));
        double t0 = now_ns();
        for (int i = 0; i < reps; i++)
            stride_execute_fwd(plan, ctx->re, ctx->im);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }

    stride_plan_destroy(plan);
    ctx->n_benchmarks++;
    return best;
}

/* =====================================================================
 * RECURSIVE DP PLANNER
 *
 * For a given N, tries each valid radix R as the first stage.
 * Recursively plans N/R, builds [R, sub_plan...], benchmarks full plan.
 * Caches the best factorization for N.
 *
 * The recursion naturally produces plans with R first, but the final
 * step (stride_dp_plan) tries all orderings of the winning set.
 * ===================================================================== */

/* Available radixes for DP decomposition, largest first */
static const int DP_RADIXES[] = {
    64, 32, 25, 20, 16, 12, 10, 8, 7, 6, 5, 4, 3, 2,
    19, 17, 13, 11, 0
};

static double _dp_solve(stride_dp_context_t *ctx, int N,
                         const stride_registry_t *reg,
                         int *out_factors, int *out_nf) {
    /* Check cache */
    dp_entry_t *cached = _dp_lookup(ctx, N);
    if (cached) {
        ctx->n_cache_hits++;
        memcpy(out_factors, cached->factors, cached->nfactors * sizeof(int));
        *out_nf = cached->nfactors;
        return cached->cost_ns;
    }

    /* Base case: N is itself a registered radix — single stage */
    if (stride_registry_has(reg, N) && N <= 64) {
        int factors[1] = {N};
        double ns = _dp_bench(ctx, N, factors, 1, reg);

        dp_entry_t *e = _dp_insert(ctx, N);
        if (e) {
            e->factors[0] = N;
            e->nfactors = 1;
            e->cost_ns = ns;
        }
        out_factors[0] = N;
        *out_nf = 1;
        return ns;
    }

    /* Recursive case: try each valid radix as first stage */
    double best_ns = 1e18;
    int best_factors[FACT_MAX_STAGES];
    int best_nf = 0;

    for (const int *rp = DP_RADIXES; *rp; rp++) {
        int R = *rp;
        if (N % R != 0) continue;
        if (!stride_registry_has(reg, R)) continue;

        int M = N / R;
        if (M < 1) continue;

        /* Base case for M=1: single stage R */
        if (M == 1) {
            int factors[1] = {R};
            double ns = _dp_bench(ctx, N, factors, 1, reg);
            if (ns < best_ns) {
                best_ns = ns;
                best_factors[0] = R;
                best_nf = 1;
            }
            continue;
        }

        /* Recursively solve M */
        int sub_factors[FACT_MAX_STAGES];
        int sub_nf = 0;
        double sub_cost = _dp_solve(ctx, M, reg, sub_factors, &sub_nf);
        if (sub_cost >= 1e17) continue;
        if (sub_nf + 1 > FACT_MAX_STAGES) continue;

        /* Build candidate: [R, sub_factors...] */
        int candidate[FACT_MAX_STAGES];
        candidate[0] = R;
        memcpy(candidate + 1, sub_factors, sub_nf * sizeof(int));
        int nf = sub_nf + 1;

        /* Benchmark the FULL plan (captures real cache behavior) */
        double ns = _dp_bench(ctx, N, candidate, nf, reg);

        if (ns < best_ns) {
            best_ns = ns;
            memcpy(best_factors, candidate, nf * sizeof(int));
            best_nf = nf;
        }
    }

    /* Cache result */
    dp_entry_t *e = _dp_insert(ctx, N);
    if (e && best_nf > 0) {
        memcpy(e->factors, best_factors, best_nf * sizeof(int));
        e->nfactors = best_nf;
        e->cost_ns = best_ns;
    }

    memcpy(out_factors, best_factors, best_nf * sizeof(int));
    *out_nf = best_nf;
    return best_ns;
}

/* =====================================================================
 * TOP-LEVEL DP PLANNER
 *
 * 1. Run recursive DP to find best factorization SET
 * 2. Try all orderings of the winning set (captures ordering effects
 *    that the fixed-first-radix recursion misses)
 * 3. Return the best ordering
 * ===================================================================== */

static double stride_dp_plan(stride_dp_context_t *ctx, int N,
                              const stride_registry_t *reg,
                              stride_factorization_t *best_fact,
                              int verbose) {
    /* Phase 1: recursive DP to find best radix set */
    int dp_factors[FACT_MAX_STAGES];
    int dp_nf = 0;
    double dp_ns = _dp_solve(ctx, N, reg, dp_factors, &dp_nf);

    if (dp_nf == 0 || dp_ns >= 1e17) {
        if (verbose) printf("  N=%d: DP failed to find a plan\n", N);
        return 1e18;
    }

    if (verbose) {
        printf("  N=%d K=%zu: DP found ", N, ctx->K);
        for (int s = 0; s < dp_nf; s++)
            printf("%s%d", s ? "x" : "", dp_factors[s]);
        printf(" = %.1f ns (%d benchmarks, %d cache hits)\n",
               dp_ns, ctx->n_benchmarks, ctx->n_cache_hits);
    }

    /* Phase 2: try all orderings of the winning set.
     * The DP always puts the chosen radix first, which may not be optimal
     * (e.g., large radix last for sequential stride access). */
    permutation_list_t *plist = (permutation_list_t *)malloc(sizeof(*plist));
    stride_gen_permutations(dp_factors, dp_nf, plist);

    double global_best = dp_ns;
    best_fact->nfactors = dp_nf;
    memcpy(best_fact->factors, dp_factors, dp_nf * sizeof(int));

    for (int pi = 0; pi < plist->count; pi++) {
        /* Skip the ordering we already benchmarked in DP */
        int same = 1;
        for (int s = 0; s < dp_nf; s++)
            if (plist->perms[pi][s] != dp_factors[s]) { same = 0; break; }
        if (same) continue;

        double ns = _dp_bench(ctx, N, plist->perms[pi], dp_nf, reg);
        if (ns < global_best) {
            global_best = ns;
            memcpy(best_fact->factors, plist->perms[pi], dp_nf * sizeof(int));
        }
    }

    if (verbose && global_best < dp_ns) {
        printf("  N=%d: reordering improved to ", N);
        for (int s = 0; s < best_fact->nfactors; s++)
            printf("%s%d", s ? "x" : "", best_fact->factors[s]);
        printf(" = %.1f ns\n", global_best);
    }

    free(plist);
    ctx->n_benchmarks += plist ? 0 : 0; /* already counted in _dp_bench */
    return global_best;
}

#endif /* STRIDE_DP_PLANNER_H */
