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
 *
 * Upgrade A (2026-04-26): cache key is now (N, K_eff), not just N.
 *   K_eff is the *effective* batch size at this call site:
 *     K_eff = K_outer * product(prefix radixes consumed before reaching N).
 *   Two calls for the same N at different K_eff produce different cache
 *   slots, so a sub-plan winner found in one composition context cannot
 *   pollute lookups from a different context. This is the principled fix
 *   for the v1.1 "lock-in" failure mode where M's cache returned a
 *   factorization that was best as-isolated but suboptimal as-substage.
 *
 * Upgrade C: believe_subplan_cost toggle (FFTW BELIEVE_PCOST analog).
 *   When 1 (default, MEASURE-style): cache hit returns cached cost,
 *   no re-measurement.
 *   When 0 (PATIENT-style): cache hit returns the cached factorization
 *   but the cost is re-measured fresh. Structural memoization is kept
 *   so the radix-search isn't repeated, but variance is re-absorbed
 *   on every encounter.
 * ===================================================================== */

#define DP_CACHE_MAX 512

typedef struct {
    int N;
    size_t K_eff;                          /* effective batch at lookup time */
    int factors[FACT_MAX_STAGES];
    int nfactors;
    double cost_ns;
} dp_entry_t;

typedef struct {
    dp_entry_t entries[DP_CACHE_MAX];
    int count;
    size_t K;                              /* top-level K_outer */

    /* Shared benchmark buffers (allocated for max N*K) */
    double *re, *im, *orig_re, *orig_im;
    size_t buf_total;  /* current buffer size in elements */
    int max_N;         /* largest N we'll plan for */

    /* Compositional-trust toggle (set by user before stride_dp_plan).
     * 1 = trust cached pcost (default, MEASURE).
     * 0 = re-measure on cache hit (PATIENT). */
    int believe_subplan_cost;

    /* Statistics */
    int n_benchmarks;
    int n_cache_hits;
} stride_dp_context_t;

static void stride_dp_init(stride_dp_context_t *ctx, size_t K, int max_N) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->K = K;
    ctx->max_N = max_N;
    ctx->buf_total = (size_t)max_N * K;
    ctx->believe_subplan_cost = 1;          /* default: MEASURE semantics */

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

/* Cache lookup — keyed by (N, K_eff). */
static dp_entry_t *_dp_lookup(stride_dp_context_t *ctx, int N, size_t K_eff) {
    for (int i = 0; i < ctx->count; i++)
        if (ctx->entries[i].N == N && ctx->entries[i].K_eff == K_eff)
            return &ctx->entries[i];
    return NULL;
}

/* Cache insert — keyed by (N, K_eff). */
static dp_entry_t *_dp_insert(stride_dp_context_t *ctx, int N, size_t K_eff) {
    if (ctx->count >= DP_CACHE_MAX) return NULL;
    dp_entry_t *e = &ctx->entries[ctx->count++];
    memset(e, 0, sizeof(*e));
    e->N = N;
    e->K_eff = K_eff;
    e->cost_ns = 1e18;
    return e;
}

/* =====================================================================
 * BENCHMARK HELPER
 *
 * Benchmarks a full plan at (N, K_eff) using the context's shared buffers.
 *
 * Upgrade B (2026-04-26): timing harness now mirrors FFTW's
 * measure_execution_time (kernel/timer.c). Adaptive iteration count:
 * doubles `reps` until tmin*reps >= TIME_MIN, then takes best-of-N
 * across DP_TIME_REPEAT trials. Hard wall-clock cap per call.
 * Per-trial buffer reset (zero-init via copy from orig) keeps the
 * data path consistent across repeats and absorbs denormals.
 *
 * Buffer-size invariant: total = N * K_eff, and N*K_eff is conserved
 * across the recursion (sub_N * sub_K_eff = N_outer * K_outer), so
 * the once-allocated ctx buffers always fit.
 * ===================================================================== */

#ifndef DP_TIME_REPEAT
#define DP_TIME_REPEAT  6        /* number of best-of trials */
#endif
#ifndef DP_TIME_MIN_NS
#define DP_TIME_MIN_NS  2.0e6    /* min wall-clock per trial (2 ms) */
#endif
#ifndef DP_TIME_LIMIT_NS
#define DP_TIME_LIMIT_NS  5.0e8  /* per-bench cap (~0.5 s) */
#endif

/* Intra-cell thermal pacing.
 *
 * At K > MEASURE_PACE_K_THRESHOLD, sustained 100% core load during the
 * search heats the core enough that bench numbers drift up over the
 * minutes-long search at a single cell. The package thermal envelope is
 * shared, so even core-pinned runs are affected at large K.
 *
 * Sleep MEASURE_PACE_MS ms every MEASURE_PACE_EVERY benches to let the
 * core temp recover. ~5% wall overhead at K=256, much smoother numbers. */
#ifndef MEASURE_PACE_K_THRESHOLD
#define MEASURE_PACE_K_THRESHOLD 64
#endif
#ifndef MEASURE_PACE_EVERY
#define MEASURE_PACE_EVERY 25
#endif
#ifndef MEASURE_PACE_MS
#define MEASURE_PACE_MS 200
#endif

#ifdef _WIN32
  /* Forward-declare Sleep without pulling all of windows.h into every
   * TU that includes this header. _WINDOWS_ is windows.h's own guard;
   * if it's defined, Sleep is already declared. */
  #ifndef _WINDOWS_
  extern __declspec(dllimport) void __stdcall Sleep(unsigned long ms);
  #endif
  static void _dp_pace_sleep(int ms) { if (ms > 0) Sleep((unsigned long)ms); }
#else
  #include <unistd.h>
  static void _dp_pace_sleep(int ms) {
    if (ms > 0) usleep((useconds_t)ms * 1000);
  }
#endif

static void _dp_maybe_pace(stride_dp_context_t *ctx) {
    if (ctx->K <= (size_t)MEASURE_PACE_K_THRESHOLD) return;
    if ((ctx->n_benchmarks % MEASURE_PACE_EVERY) != 0) return;
    _dp_pace_sleep(MEASURE_PACE_MS);
}

static double _dp_bench(stride_dp_context_t *ctx, int N,
                         const int *factors, int nf, size_t K_eff,
                         const stride_registry_t *reg) {
    size_t total = (size_t)N * K_eff;

    /* Sanity check: all requested radixes must have n1 codelets registered. */
    for (int s = 0; s < nf; s++) {
        if (!reg->n1_fwd[factors[s]]) return 1e18;
    }

    /* Route through _stride_build_plan so that codelet-side plan_wisdom
     * drives protocol selection (flat / t1s / DIT-log3) at plan time.
     * The plan shape we measure here matches what stride_auto_plan and
     * stride_wise_plan will produce at deploy time. */
    stride_plan_t *plan = _stride_build_plan(N, K_eff, factors, nf, reg);
    if (!plan) return 1e18;

    /* Warmup (also serves as a single-iter calibration baseline). */
    memcpy(ctx->re, ctx->orig_re, total * sizeof(double));
    memcpy(ctx->im, ctx->orig_im, total * sizeof(double));
    stride_execute_fwd(plan, ctx->re, ctx->im);

    /* Adaptive iter: double `reps` until one trial >= DP_TIME_MIN_NS,
     * then collect DP_TIME_REPEAT best-of trials at that rep count.
     * Mirrors FFTW kernel/timer.c::measure_execution_time. */
    double best = 1e30;
    double total_elapsed = 0.0;
    int reps = 1;
    int calibrated = 0;

    for (int outer = 0; outer < 32 && total_elapsed < DP_TIME_LIMIT_NS; outer++) {
        double tmin_trial = 1e30;

        for (int t = 0; t < DP_TIME_REPEAT; t++) {
            memcpy(ctx->re, ctx->orig_re, total * sizeof(double));
            memcpy(ctx->im, ctx->orig_im, total * sizeof(double));
            double t0 = now_ns();
            for (int i = 0; i < reps; i++)
                stride_execute_fwd(plan, ctx->re, ctx->im);
            double trial_ns = now_ns() - t0;
            if (trial_ns < tmin_trial) tmin_trial = trial_ns;
            total_elapsed += trial_ns;
            if (total_elapsed >= DP_TIME_LIMIT_NS) break;
        }

        if (!calibrated) {
            /* If trial duration is too short, double reps and retry. */
            if (tmin_trial < DP_TIME_MIN_NS) {
                reps *= 2;
                if (reps > (1 << 24)) { calibrated = 1; }   /* sanity cap */
                continue;
            }
            calibrated = 1;
        }

        double per_iter = tmin_trial / (double)reps;
        if (per_iter < best) best = per_iter;

        /* Calibrated and DP_TIME_REPEAT collected — done. */
        break;
    }

    stride_plan_destroy(plan);
    ctx->n_benchmarks++;
    _dp_maybe_pace(ctx);
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

static double _dp_solve(stride_dp_context_t *ctx, int N, size_t K_eff,
                         const stride_registry_t *reg,
                         int *out_factors, int *out_nf) {
    /* Check cache (keyed by N + K_eff). */
    dp_entry_t *cached = _dp_lookup(ctx, N, K_eff);
    if (cached) {
        ctx->n_cache_hits++;
        memcpy(out_factors, cached->factors, cached->nfactors * sizeof(int));
        *out_nf = cached->nfactors;
        if (ctx->believe_subplan_cost) {
            return cached->cost_ns;
        }
        /* PATIENT: re-measure with the cached factorization. */
        double fresh = _dp_bench(ctx, N, cached->factors, cached->nfactors,
                                 K_eff, reg);
        cached->cost_ns = fresh;        /* refresh the stored cost */
        return fresh;
    }

    /* Base case: N is itself a registered radix — single stage */
    if (stride_registry_has(reg, N) && N <= 64) {
        int factors[1] = {N};
        double ns = _dp_bench(ctx, N, factors, 1, K_eff, reg);

        dp_entry_t *e = _dp_insert(ctx, N, K_eff);
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
            double ns = _dp_bench(ctx, N, factors, 1, K_eff, reg);
            if (ns < best_ns) {
                best_ns = ns;
                best_factors[0] = R;
                best_nf = 1;
            }
            continue;
        }

        /* Recursively solve M.
         * K_eff for the sub-problem grows by R: when R is consumed as the
         * first stage of N, M's stages execute at batch K_eff * R. */
        size_t K_eff_sub = K_eff * (size_t)R;
        int sub_factors[FACT_MAX_STAGES];
        int sub_nf = 0;
        double sub_cost = _dp_solve(ctx, M, K_eff_sub, reg, sub_factors, &sub_nf);
        if (sub_cost >= 1e17) continue;
        if (sub_nf + 1 > FACT_MAX_STAGES) continue;

        /* Build candidate: [R, sub_factors...] */
        int candidate[FACT_MAX_STAGES];
        candidate[0] = R;
        memcpy(candidate + 1, sub_factors, sub_nf * sizeof(int));
        int nf = sub_nf + 1;

        /* Benchmark the FULL plan at this K_eff (captures composition). */
        double ns = _dp_bench(ctx, N, candidate, nf, K_eff, reg);

        if (ns < best_ns) {
            best_ns = ns;
            memcpy(best_factors, candidate, nf * sizeof(int));
            best_nf = nf;
        }
    }

    /* Cache result */
    dp_entry_t *e = _dp_insert(ctx, N, K_eff);
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
    /* Phase 1: recursive DP to find best radix set.
     * Top-level call uses K_eff = ctx->K (i.e., the user-requested batch). */
    int dp_factors[FACT_MAX_STAGES];
    int dp_nf = 0;
    double dp_ns = _dp_solve(ctx, N, ctx->K, reg, dp_factors, &dp_nf);

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

        double ns = _dp_bench(ctx, N, plist->perms[pi], dp_nf, ctx->K, reg);
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

/* =====================================================================
 * VFFT_MEASURE — TOP-K + VARIANT-AWARE PLANNER (FFTW PATIENT-style)
 *
 * Two-pass design:
 *
 *   1. Coarse pass: enumerate every (factorization, permutation) pair,
 *      bench with wisdom-default variants. Cheap because wisdom-default
 *      is one bench per pair, and the FFTW-style adaptive timer keeps
 *      each bench bounded.
 *
 *   2. Refine pass: take the K_top lowest-cost (factorization, permutation)
 *      pairs from the coarse pass; for each, run the full variant
 *      cartesian × {DIT, DIF}. Pick global best.
 *
 * This addresses the v1.2 lesson that variant choice can flip the
 * factorization ranking (the K=4 case where joint picked 8x32x16 over
 * legacy 4x4x16x16): if the right-factorization survives the coarse
 * pass at top-K, the refine pass discovers its variant-optimal cost.
 *
 * Cost vs. EXTREME (full joint cartesian, calibrate_cell_joint):
 *   coarse:  C × bench_default       (~ N=4096 K=256: ~500 × 100ms = 50s)
 *   refine:  K_top × V × 2 × bench   (~ 3 × 256 × 2 × 100ms = 150s)
 *   Total:   ~200s/cell at N=4096 K=256, vs EXTREME ~3000s = ~7%.
 *
 * K_top is the search-budget knob:
 *   K_top=1  ≈ legacy DP + variant cartesian (the previous MEASURE)
 *   K_top=3  default — captures most variant-axis-flips-fact cases
 *   K_top=∞  ≈ EXTREME (every coarse pair gets full refine)
 * ===================================================================== */

#ifndef MEASURE_TOPK_DEFAULT
#define MEASURE_TOPK_DEFAULT 5
#endif
#ifndef MEASURE_MAX_CANDIDATES
#define MEASURE_MAX_CANDIDATES 1024
#endif
#ifndef MEASURE_COARSE_RUNS
#define MEASURE_COARSE_RUNS 2     /* coarse-pass sweeps; per-cand min */
#endif

typedef struct {
    stride_factorization_t fact;
    vfft_variant_t         variants[FACT_MAX_STAGES];
    int                    use_dif_forward;
    double                 cost_ns;
} stride_plan_decision_t;

/* Single bench at fully-explicit (factors, variants, orientation, K_eff).
 * Builds via _stride_build_plan_explicit, runs the FFTW-style adaptive
 * timer (same harness as _dp_bench), returns ns/iter. 1e18 on failure. */
static double _dp_bench_explicit_one(stride_dp_context_t *ctx, int N,
                                     const int *factors, int nf,
                                     const vfft_variant_t *variants,
                                     int use_dif_forward, size_t K_eff,
                                     const stride_registry_t *reg) {
    size_t total = (size_t)N * K_eff;

    for (int s = 0; s < nf; s++) {
        if (!reg->n1_fwd[factors[s]]) return 1e18;
        if (!vfft_variant_available(reg, factors[s], use_dif_forward,
                                    variants[s])) {
            /* Stage 0 in DIT and stage nf-1 in DIF have no twiddle
             * codelet; the iterator still emits a variant code (FLAT)
             * but the registry slot is unused. Don't fail on those. */
            int is_no_tw = use_dif_forward ? (s == nf - 1) : (s == 0);
            if (!is_no_tw) return 1e18;
        }
    }

    stride_plan_t *plan = _stride_build_plan_explicit(
            N, K_eff, factors, nf, variants, use_dif_forward, reg);
    if (!plan) return 1e18;

    memcpy(ctx->re, ctx->orig_re, total * sizeof(double));
    memcpy(ctx->im, ctx->orig_im, total * sizeof(double));
    stride_execute_fwd(plan, ctx->re, ctx->im);

    double best = 1e30;
    double total_elapsed = 0.0;
    int reps = 1;
    int calibrated = 0;

    for (int outer = 0; outer < 32 && total_elapsed < DP_TIME_LIMIT_NS; outer++) {
        double tmin_trial = 1e30;

        for (int t = 0; t < DP_TIME_REPEAT; t++) {
            memcpy(ctx->re, ctx->orig_re, total * sizeof(double));
            memcpy(ctx->im, ctx->orig_im, total * sizeof(double));
            double t0 = now_ns();
            for (int i = 0; i < reps; i++)
                stride_execute_fwd(plan, ctx->re, ctx->im);
            double trial_ns = now_ns() - t0;
            if (trial_ns < tmin_trial) tmin_trial = trial_ns;
            total_elapsed += trial_ns;
            if (total_elapsed >= DP_TIME_LIMIT_NS) break;
        }

        if (!calibrated) {
            if (tmin_trial < DP_TIME_MIN_NS) {
                reps *= 2;
                if (reps > (1 << 24)) calibrated = 1;
                continue;
            }
            calibrated = 1;
        }

        double per_iter = tmin_trial / (double)reps;
        if (per_iter < best) best = per_iter;
        break;
    }

    stride_plan_destroy(plan);
    ctx->n_benchmarks++;
    _dp_maybe_pace(ctx);
    return best;
}

/* Variant cartesian search at one (factors, orientation, K_eff). Walks
 * vfft_variant_iter_*, benches each, returns the best (cost, variants).
 * Returns 1e18 if no valid assignment exists in this orientation. */
static double _dp_variant_search(stride_dp_context_t *ctx, int N,
                                 const int *factors, int nf,
                                 int use_dif_forward, size_t K_eff,
                                 const stride_registry_t *reg,
                                 vfft_variant_t *out_best,
                                 long *out_n_assignments,
                                 int verbose) {
    vfft_variant_iter_t it;
    if (!vfft_variant_iter_init(&it, factors, nf, use_dif_forward, reg)) {
        if (out_n_assignments) *out_n_assignments = 0;
        return 1e18;
    }

    double best_ns = 1e18;
    long count = 0;
    do {
        vfft_variant_t cur[STRIDE_MAX_STAGES];
        vfft_variant_iter_get(&it, cur);
        double ns = _dp_bench_explicit_one(ctx, N, factors, nf, cur,
                                           use_dif_forward, K_eff, reg);
        count++;
        if (verbose) {
            printf("    [%s] ", use_dif_forward ? "DIF" : "DIT");
            for (int s = 0; s < nf; s++)
                printf("%s%s", s ? "/" : "", vfft_variant_name(cur[s]));
            printf(" = %.1f ns\n", ns);
        }
        if (ns < best_ns) {
            best_ns = ns;
            if (out_best) memcpy(out_best, cur, nf * sizeof(*out_best));
        }
    } while (vfft_variant_iter_next(&it));

    if (out_n_assignments) *out_n_assignments = count;
    return best_ns;
}

/* Coarse-pass candidate, sorted by default-variant cost. */
typedef struct {
    int    factors[FACT_MAX_STAGES];
    int    nf;
    double cost_ns;
} _measure_candidate_t;

static int _measure_cmp(const void *a, const void *b) {
    double ca = ((const _measure_candidate_t *)a)->cost_ns;
    double cb = ((const _measure_candidate_t *)b)->cost_ns;
    if (ca < cb) return -1;
    if (ca > cb) return  1;
    return 0;
}

/* Top-level VFFT_MEASURE entry point (top-K + variant cartesian).
 *
 * Coarse pass: bench every (factorization, permutation) with default
 * variants. Sort. Take top-K.
 * Refine pass: variant cartesian × {DIT, DIF} on each top-K, pick global
 * best.
 *
 * decision->cost_ns is the variant-best ns/iter at the winning orientation
 * (measured by the FFTW-style adaptive timer; the calibrator re-benches
 * with deploy-quality bench_plan_min before writing wisdom). */
static double stride_dp_plan_measure(stride_dp_context_t *ctx, int N,
                                     const stride_registry_t *reg,
                                     stride_plan_decision_t *decision,
                                     int verbose) {
    /* COARSE PASS: enumerate factorizations × permutations × default
     * variants. _dp_bench routes through _stride_build_plan so the
     * coarse cost reflects what the wisdom path would deploy. */
    factorization_list_t flist;
    stride_enumerate_factorizations(N, reg, &flist);
    if (flist.count == 0) {
        if (verbose) printf("  N=%d MEASURE: no factorizations enumerated\n", N);
        return 1e18;
    }

    static _measure_candidate_t cands[MEASURE_MAX_CANDIDATES];
    int n_cands = 0;

    for (int fi = 0; fi < flist.count && n_cands < MEASURE_MAX_CANDIDATES; fi++) {
        const int  nf = flist.results[fi].nfactors;
        const int *base_factors = flist.results[fi].factors;

        permutation_list_t plist;
        stride_gen_permutations(base_factors, nf, &plist);

        for (int pi = 0; pi < plist.count && n_cands < MEASURE_MAX_CANDIDATES; pi++) {
            const int *perm = plist.perms[pi];

            /* Validate: every radix needs an n1 codelet. */
            int can_build = 1;
            for (int s = 0; s < nf; s++) {
                int R = perm[s];
                if (R <= 0 || R >= STRIDE_REG_MAX_RADIX || !reg->n1_fwd[R]) {
                    can_build = 0;
                    break;
                }
            }
            if (!can_build) continue;

            double ns = _dp_bench(ctx, N, perm, nf, ctx->K, reg);
            if (ns >= 1e17) continue;

            cands[n_cands].nf = nf;
            for (int s = 0; s < nf; s++) cands[n_cands].factors[s] = perm[s];
            cands[n_cands].cost_ns = ns;
            n_cands++;
        }
    }

    if (n_cands == 0) {
        if (verbose) printf("  N=%d MEASURE: no working coarse plans\n", N);
        return 1e18;
    }

    /* Best-of-runs: extra coarse sweeps over the same candidate set.
     * Each (factorization, permutation) gets its min cost across runs.
     * Reduces variance-driven ranking flips that knocked the actual
     * best multiset out of top-K in earlier pilots. */
    for (int run = 1; run < MEASURE_COARSE_RUNS; run++) {
        for (int i = 0; i < n_cands; i++) {
            double ns = _dp_bench(ctx, N, cands[i].factors, cands[i].nf,
                                  ctx->K, reg);
            if (ns < cands[i].cost_ns) cands[i].cost_ns = ns;
        }
    }

    /* Sort, take top-K. */
    qsort(cands, n_cands, sizeof(*cands), _measure_cmp);
    int K_top = MEASURE_TOPK_DEFAULT;
    int n_topk = (n_cands < K_top) ? n_cands : K_top;

    /* REFINE: variant cartesian × {DIT, DIF} on each top-K candidate. */
    double best_ns = 1e18;
    int    best_factors[FACT_MAX_STAGES];
    int    best_nf = 0;
    vfft_variant_t best_variants[FACT_MAX_STAGES];
    int    best_use_dif = 0;
    long   total_refine = 0;

    for (int k = 0; k < n_topk; k++) {
        const _measure_candidate_t *c = &cands[k];

        for (int orient = 0; orient < 2; orient++) {
            vfft_variant_t cur_best[FACT_MAX_STAGES];
            long n_this = 0;
            double ns = _dp_variant_search(ctx, N, c->factors, c->nf,
                                           orient, ctx->K, reg,
                                           cur_best, &n_this, 0);
            total_refine += n_this;
            if (ns < best_ns) {
                best_ns = ns;
                best_nf = c->nf;
                memcpy(best_factors, c->factors, c->nf * sizeof(int));
                memcpy(best_variants, cur_best,
                       c->nf * sizeof(vfft_variant_t));
                best_use_dif = orient;
            }
        }
    }

    if (best_ns >= 1e17) {
        if (verbose) printf("  N=%d MEASURE: no valid refine\n", N);
        return 1e18;
    }

    decision->fact.nfactors = best_nf;
    memcpy(decision->fact.factors, best_factors, best_nf * sizeof(int));
    memcpy(decision->variants, best_variants,
           best_nf * sizeof(vfft_variant_t));
    decision->use_dif_forward = best_use_dif;
    decision->cost_ns = best_ns;

    if (verbose) {
        printf("  N=%d K=%zu MEASURE-topk(%d): coarse=%d top=%d refine=%ld -> ",
               N, ctx->K, K_top, n_cands, n_topk, total_refine);
        for (int s = 0; s < best_nf; s++)
            printf("%s%d", s ? "x" : "", best_factors[s]);
        printf(" %s ", best_use_dif ? "DIF" : "DIT");
        for (int s = 0; s < best_nf; s++)
            printf("%s%s", s ? "/" : "", vfft_variant_name(best_variants[s]));
        printf(" = %.1f ns\n", best_ns);
    }

    return best_ns;
}

#endif /* STRIDE_DP_PLANNER_H */
