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

/* Top-K-at-every-level (Upgrade D, 2026-04-27).
 *
 * Each cache row stores up to DP_TOPK_MAX best plans for (N, K_eff).
 * The recursion exposes runners-up to outer levels so that a
 * factorization that lost the top-1 race in isolation can still be
 * composed under a different outer radix and win there.
 *
 * Concretely fixes the N=32768 K=4 regression where outer R=4 got
 * sub-DP(8192, K_eff=16)'s top-1 plan but missed [4,32,64] (a runner-up
 * that, wrapped under R=4, would have produced [4,4,32,64] beating the
 * eventual [64,8,64] winner).
 *
 * Cost: |R| * DP_TOPK_MAX benches per cache-miss call (vs |R|), ~3x DP
 * overhead at K=3. Cache memory: DP_CACHE_MAX * DP_TOPK_MAX * sizeof(plan)
 * = ~120KB at K=3, bounded. */
#ifndef DP_TOPK_MAX
#define DP_TOPK_MAX 3
#endif

typedef struct
{
    int factors[FACT_MAX_STAGES];
    int nfactors;
    double cost_ns;
} dp_subplan_t;

typedef struct
{
    int N;
    size_t K_eff;                    /* effective batch at lookup time */
    dp_subplan_t plans[DP_TOPK_MAX]; /* sorted by cost, ascending */
    int n_plans;                     /* 0 .. DP_TOPK_MAX */
} dp_entry_t;

typedef struct
{
    dp_entry_t entries[DP_CACHE_MAX];
    int count;
    size_t K; /* top-level K_outer */

    /* Shared benchmark buffers (allocated for max N*K) */
    double *re, *im, *orig_re, *orig_im;
    size_t buf_total; /* current buffer size in elements */
    int max_N;        /* largest N we'll plan for */

    /* Compositional-trust toggle (set by user before stride_dp_plan).
     * 1 = trust cached pcost (default, MEASURE).
     * 0 = re-measure on cache hit (PATIENT). */
    int believe_subplan_cost;

    /* Statistics */
    int n_benchmarks;
    int n_cache_hits;
} stride_dp_context_t;

static void stride_dp_init(stride_dp_context_t *ctx, size_t K, int max_N)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->K = K;
    ctx->max_N = max_N;
    ctx->buf_total = (size_t)max_N * K;
    ctx->believe_subplan_cost = 1; /* default: MEASURE semantics */

    ctx->re = (double *)STRIDE_ALIGNED_ALLOC(64, ctx->buf_total * sizeof(double));
    ctx->im = (double *)STRIDE_ALIGNED_ALLOC(64, ctx->buf_total * sizeof(double));
    ctx->orig_re = (double *)STRIDE_ALIGNED_ALLOC(64, ctx->buf_total * sizeof(double));
    ctx->orig_im = (double *)STRIDE_ALIGNED_ALLOC(64, ctx->buf_total * sizeof(double));

    srand(42);
    for (size_t i = 0; i < ctx->buf_total; i++)
    {
        ctx->orig_re[i] = (double)rand() / RAND_MAX - 0.5;
        ctx->orig_im[i] = (double)rand() / RAND_MAX - 0.5;
    }
}

static void stride_dp_destroy(stride_dp_context_t *ctx)
{
    STRIDE_ALIGNED_FREE(ctx->re);
    STRIDE_ALIGNED_FREE(ctx->im);
    STRIDE_ALIGNED_FREE(ctx->orig_re);
    STRIDE_ALIGNED_FREE(ctx->orig_im);
    memset(ctx, 0, sizeof(*ctx));
}

/* Cache lookup — keyed by (N, K_eff). */
static dp_entry_t *_dp_lookup(stride_dp_context_t *ctx, int N, size_t K_eff)
{
    for (int i = 0; i < ctx->count; i++)
        if (ctx->entries[i].N == N && ctx->entries[i].K_eff == K_eff)
            return &ctx->entries[i];
    return NULL;
}

/* Cache insert — keyed by (N, K_eff). Plans array starts empty. */
static dp_entry_t *_dp_insert(stride_dp_context_t *ctx, int N, size_t K_eff)
{
    if (ctx->count >= DP_CACHE_MAX)
        return NULL;
    dp_entry_t *e = &ctx->entries[ctx->count++];
    memset(e, 0, sizeof(*e));
    e->N = N;
    e->K_eff = K_eff;
    e->n_plans = 0;
    return e;
}

/* Sort comparator for dp_subplan_t (ascending by cost_ns). */
static int _dp_subplan_cmp(const void *a, const void *b)
{
    double ca = ((const dp_subplan_t *)a)->cost_ns;
    double cb = ((const dp_subplan_t *)b)->cost_ns;
    if (ca < cb)
        return -1;
    if (ca > cb)
        return 1;
    return 0;
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
#define DP_TIME_REPEAT 6 /* number of best-of trials */
#endif
#ifndef DP_TIME_MIN_NS
#define DP_TIME_MIN_NS 2.0e6 /* min wall-clock per trial (2 ms) */
#endif
#ifndef DP_TIME_LIMIT_NS
#define DP_TIME_LIMIT_NS 5.0e8 /* per-bench cap (~0.5 s) */
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
static void _dp_pace_sleep(int ms)
{
    if (ms > 0)
        Sleep((unsigned long)ms);
}
#else
#include <unistd.h>
static void _dp_pace_sleep(int ms)
{
    if (ms > 0)
        usleep((useconds_t)ms * 1000);
}
#endif

static void _dp_maybe_pace(stride_dp_context_t *ctx)
{
    if (ctx->K <= (size_t)MEASURE_PACE_K_THRESHOLD)
        return;
    if ((ctx->n_benchmarks % MEASURE_PACE_EVERY) != 0)
        return;
    _dp_pace_sleep(MEASURE_PACE_MS);
}

static double _dp_bench(stride_dp_context_t *ctx, int N,
                        const int *factors, int nf, size_t K_eff,
                        const stride_registry_t *reg)
{
    size_t total = (size_t)N * K_eff;

    /* Sanity check: all requested radixes must have n1 codelets registered. */
    for (int s = 0; s < nf; s++)
    {
        if (!reg->n1_fwd[factors[s]])
            return 1e18;
    }

    /* Route through _stride_build_plan so that codelet-side plan_wisdom
     * drives protocol selection (flat / t1s / DIT-log3) at plan time.
     * The plan shape we measure here matches what stride_auto_plan and
     * stride_wise_plan will produce at deploy time. */
    stride_plan_t *plan = _stride_build_plan(N, K_eff, factors, nf, reg);
    if (!plan)
        return 1e18;

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

    for (int outer = 0; outer < 32 && total_elapsed < DP_TIME_LIMIT_NS; outer++)
    {
        double tmin_trial = 1e30;

        for (int t = 0; t < DP_TIME_REPEAT; t++)
        {
            memcpy(ctx->re, ctx->orig_re, total * sizeof(double));
            memcpy(ctx->im, ctx->orig_im, total * sizeof(double));
            double t0 = now_ns();
            for (int i = 0; i < reps; i++)
                stride_execute_fwd(plan, ctx->re, ctx->im);
            double trial_ns = now_ns() - t0;
            if (trial_ns < tmin_trial)
                tmin_trial = trial_ns;
            total_elapsed += trial_ns;
            if (total_elapsed >= DP_TIME_LIMIT_NS)
                break;
        }

        if (!calibrated)
        {
            /* If trial duration is too short, double reps and retry. */
            if (tmin_trial < DP_TIME_MIN_NS)
            {
                reps *= 2;
                if (reps > (1 << 24))
                {
                    calibrated = 1;
                } /* sanity cap */
                continue;
            }
            calibrated = 1;
        }

        double per_iter = tmin_trial / (double)reps;
        if (per_iter < best)
            best = per_iter;

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
    19, 17, 13, 11, 0};

/* Top-K recursive DP solver. Returns up to max_out best plans for
 * (N, K_eff), sorted by cost ascending. Cached on first call. The
 * cache row stores up to DP_TOPK_MAX plans; subsequent calls return
 * min(cached_count, max_out) of them.
 *
 * BELIEVE_PCOST behavior: when 0, the top-ranked cached plan's cost is
 * re-measured fresh and the cache row is updated; runners-up are kept
 * with their original (stale) costs. This matches the production
 * intent — the BELIEVE flag affects winner selection variance, not the
 * shape of the runner-up list. */
static int _dp_solve_topk(stride_dp_context_t *ctx, int N, size_t K_eff,
                          const stride_registry_t *reg,
                          dp_subplan_t *out, int max_out)
{
    if (max_out <= 0)
        return 0;

    /* Cache check (keyed by N + K_eff). */
    dp_entry_t *cached = _dp_lookup(ctx, N, K_eff);
    if (cached)
    {
        ctx->n_cache_hits++;
        int n = cached->n_plans < max_out ? cached->n_plans : max_out;
        for (int i = 0; i < n; i++)
            out[i] = cached->plans[i];
        if (!ctx->believe_subplan_cost && n > 0)
        {
            /* PATIENT: re-bench top-1 with the cached factorization. */
            double fresh = _dp_bench(ctx, N, cached->plans[0].factors,
                                     cached->plans[0].nfactors, K_eff, reg);
            cached->plans[0].cost_ns = fresh;
            out[0].cost_ns = fresh;
        }
        return n;
    }

    /* Accumulator: collect every viable candidate produced by this call.
     * Sized for |DP_RADIXES| * DP_TOPK_MAX + slack. */
    enum
    {
        _DP_ACCUM_MAX = 64
    };
    dp_subplan_t accum[_DP_ACCUM_MAX];
    int n_accum = 0;

    /* Base case: N is itself a registered radix — emits a single
     * one-stage plan. The recursive case below also tries N=R candidates
     * via the M=1 branch, but that requires N to be a registered radix
     * AND a divisor of itself; emitting here covers the 2 ≤ N ≤ 64 base. */
    if (stride_registry_has(reg, N) && N <= 64)
    {
        int factors[1] = {N};
        double ns = _dp_bench(ctx, N, factors, 1, K_eff, reg);
        if (ns < 1e17 && n_accum < _DP_ACCUM_MAX)
        {
            accum[n_accum].factors[0] = N;
            accum[n_accum].nfactors = 1;
            accum[n_accum].cost_ns = ns;
            n_accum++;
        }
    }

    /* Recursive case: try each radix R as first stage. For each R,
     * recurse on M = N/R requesting top-K_sub plans, then assemble
     * [R, sub_plan_i] for each returned sub-plan and bench. */
    for (const int *rp = DP_RADIXES; *rp && n_accum < _DP_ACCUM_MAX; rp++)
    {
        int R = *rp;
        if (N % R != 0)
            continue;
        if (!stride_registry_has(reg, R))
            continue;
        if (R == N)
            continue; /* base case already emitted above */

        int M = N / R;
        if (M < 1)
            continue;

        if (M == 1)
        {
            /* Single-stage [R] (when R itself == N is a registered radix). */
            int factors[1] = {R};
            double ns = _dp_bench(ctx, N, factors, 1, K_eff, reg);
            if (ns < 1e17 && n_accum < _DP_ACCUM_MAX)
            {
                accum[n_accum].factors[0] = R;
                accum[n_accum].nfactors = 1;
                accum[n_accum].cost_ns = ns;
                n_accum++;
            }
            continue;
        }

        /* Recurse on M with top-K. K_eff for the sub-problem grows by R:
         * when R is consumed as the first stage of N, M's stages execute
         * at batch K_eff * R. */
        size_t K_eff_sub = K_eff * (size_t)R;
        dp_subplan_t sub[DP_TOPK_MAX];
        int n_sub = _dp_solve_topk(ctx, M, K_eff_sub, reg, sub, DP_TOPK_MAX);

        for (int s = 0; s < n_sub && n_accum < _DP_ACCUM_MAX; s++)
        {
            if (sub[s].cost_ns >= 1e17)
                continue;
            if (sub[s].nfactors + 1 > FACT_MAX_STAGES)
                continue;

            int candidate[FACT_MAX_STAGES];
            candidate[0] = R;
            memcpy(candidate + 1, sub[s].factors,
                   sub[s].nfactors * sizeof(int));
            int nf = sub[s].nfactors + 1;

            double ns = _dp_bench(ctx, N, candidate, nf, K_eff, reg);
            if (ns >= 1e17)
                continue;

            memcpy(accum[n_accum].factors, candidate, nf * sizeof(int));
            accum[n_accum].nfactors = nf;
            accum[n_accum].cost_ns = ns;
            n_accum++;
        }
    }

    /* Sort by cost ascending. */
    if (n_accum > 1)
        qsort(accum, n_accum, sizeof(dp_subplan_t), _dp_subplan_cmp);

    /* Cache top DP_TOPK_MAX. */
    int n_keep = n_accum < DP_TOPK_MAX ? n_accum : DP_TOPK_MAX;
    dp_entry_t *e = _dp_insert(ctx, N, K_eff);
    if (e)
    {
        for (int i = 0; i < n_keep; i++)
            e->plans[i] = accum[i];
        e->n_plans = n_keep;
    }

    /* Output up to max_out plans. */
    int n_out = n_keep < max_out ? n_keep : max_out;
    for (int i = 0; i < n_out; i++)
        out[i] = accum[i];
    return n_out;
}

/* Backward-compat wrapper: returns top-1 in the legacy (factors, nf, cost)
 * shape. All existing callers (stride_dp_plan, stride_dp_plan_joint_blocked,
 * etc.) keep working unchanged. */
static double _dp_solve(stride_dp_context_t *ctx, int N, size_t K_eff,
                        const stride_registry_t *reg,
                        int *out_factors, int *out_nf)
{
    dp_subplan_t top1;
    int n = _dp_solve_topk(ctx, N, K_eff, reg, &top1, 1);
    if (n == 0)
    {
        *out_nf = 0;
        return 1e18;
    }
    memcpy(out_factors, top1.factors, top1.nfactors * sizeof(int));
    *out_nf = top1.nfactors;
    return top1.cost_ns;
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
                             int verbose)
{
    /* Phase 1: recursive DP to find best radix set.
     * Top-level call uses K_eff = ctx->K (i.e., the user-requested batch). */
    int dp_factors[FACT_MAX_STAGES];
    int dp_nf = 0;
    double dp_ns = _dp_solve(ctx, N, ctx->K, reg, dp_factors, &dp_nf);

    if (dp_nf == 0 || dp_ns >= 1e17)
    {
        if (verbose)
            printf("  N=%d: DP failed to find a plan\n", N);
        return 1e18;
    }

    if (verbose)
    {
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

    for (int pi = 0; pi < plist->count; pi++)
    {
        /* Skip the ordering we already benchmarked in DP */
        int same = 1;
        for (int s = 0; s < dp_nf; s++)
            if (plist->perms[pi][s] != dp_factors[s])
            {
                same = 0;
                break;
            }
        if (same)
            continue;

        double ns = _dp_bench(ctx, N, plist->perms[pi], dp_nf, ctx->K, reg);
        if (ns < global_best)
        {
            global_best = ns;
            memcpy(best_fact->factors, plist->perms[pi], dp_nf * sizeof(int));
        }
    }

    if (verbose && global_best < dp_ns)
    {
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
#define MEASURE_COARSE_RUNS 2 /* coarse-pass sweeps; per-cand min */
#endif
#ifndef MEASURE_REFINE_RUNS
#define MEASURE_REFINE_RUNS 4 /* refine variant cartesian sweeps; per-variant min.   \
                               * Smooths noise at the variant axis — important for \
                               * low-K prime-power cells where the search collapses  \
                               * onto variants alone and per-config measurements     \
                               * sit close to the Windows ~5-10% noise floor. */
#endif
#ifndef MEASURE_DEPLOY_THRESHOLD_PCT
#define MEASURE_DEPLOY_THRESHOLD_PCT 10 /* candidates within X% of refine-best go to \
                                         * deploy rebench. Matches Windows noise     \
                                         * floor (~12%); tighter would miss real     \
                                         * tied-winners, looser would waste deploy   \
                                         * benches on genuinely-slower plans. */
#endif
#ifndef MEASURE_DEPLOY_TOPK_MAX
#define MEASURE_DEPLOY_TOPK_MAX 5 /* hard cap on candidates to deploy-bench  \
                                   * post-refine; prevents pathological      \
                                   * blow-up on cells with near-flat variant \
                                   * cartesian. */
#endif

/* Above this N, MEASURE switches from exhaustive (factorization, permutation)
 * enumeration to DP-driven candidate collection. Below: exhaustive is cheap.
 * Above: DP recursion (now keyed on (N, K_eff), see Upgrade A) finds
 * good multisets without walking the full Cartesian. Mirrors the
 * production split point used by bench_1d_csv. */
#ifndef MEASURE_EXH_THRESHOLD
#define MEASURE_EXH_THRESHOLD 2048
#endif
/* Number of top-ranked multisets the DP collector keeps at the OUTERMOST
 * recursion frame. Sub-problems still use top-1 (regular _dp_solve);
 * only the outer level varies, which is enough to recover the v1.2
 * variant-axis-flips-multiset cases at large N (e.g., 8x32x16 vs 4x4x16x16). */
#ifndef MEASURE_DP_TOPK_MULTISETS
#define MEASURE_DP_TOPK_MULTISETS 3
#endif

typedef struct
{
    stride_factorization_t fact;
    vfft_variant_t variants[FACT_MAX_STAGES];
    int use_dif_forward;
    double cost_ns;
} stride_plan_decision_t;

/* Single bench at fully-explicit (factors, variants, orientation, K_eff).
 * Builds via _stride_build_plan_explicit, runs the FFTW-style adaptive
 * timer (same harness as _dp_bench), returns ns/iter. 1e18 on failure. */
static double _dp_bench_explicit_one(stride_dp_context_t *ctx, int N,
                                     const int *factors, int nf,
                                     const vfft_variant_t *variants,
                                     int use_dif_forward, size_t K_eff,
                                     const stride_registry_t *reg)
{
    size_t total = (size_t)N * K_eff;

    for (int s = 0; s < nf; s++)
    {
        if (!reg->n1_fwd[factors[s]])
            return 1e18;
        if (!vfft_variant_available(reg, factors[s], use_dif_forward,
                                    variants[s]))
        {
            /* Stage 0 in DIT and stage nf-1 in DIF have no twiddle
             * codelet; the iterator still emits a variant code (FLAT)
             * but the registry slot is unused. Don't fail on those. */
            int is_no_tw = use_dif_forward ? (s == nf - 1) : (s == 0);
            if (!is_no_tw)
                return 1e18;
        }
    }

    stride_plan_t *plan = _stride_build_plan_explicit(
        N, K_eff, factors, nf, variants, use_dif_forward, reg);
    if (!plan)
        return 1e18;

    memcpy(ctx->re, ctx->orig_re, total * sizeof(double));
    memcpy(ctx->im, ctx->orig_im, total * sizeof(double));
    stride_execute_fwd(plan, ctx->re, ctx->im);

    double best = 1e30;
    double total_elapsed = 0.0;
    int reps = 1;
    int calibrated = 0;

    for (int outer = 0; outer < 32 && total_elapsed < DP_TIME_LIMIT_NS; outer++)
    {
        double tmin_trial = 1e30;

        for (int t = 0; t < DP_TIME_REPEAT; t++)
        {
            memcpy(ctx->re, ctx->orig_re, total * sizeof(double));
            memcpy(ctx->im, ctx->orig_im, total * sizeof(double));
            double t0 = now_ns();
            for (int i = 0; i < reps; i++)
                stride_execute_fwd(plan, ctx->re, ctx->im);
            double trial_ns = now_ns() - t0;
            if (trial_ns < tmin_trial)
                tmin_trial = trial_ns;
            total_elapsed += trial_ns;
            if (total_elapsed >= DP_TIME_LIMIT_NS)
                break;
        }

        if (!calibrated)
        {
            if (tmin_trial < DP_TIME_MIN_NS)
            {
                reps *= 2;
                if (reps > (1 << 24))
                    calibrated = 1;
                continue;
            }
            calibrated = 1;
        }

        double per_iter = tmin_trial / (double)reps;
        if (per_iter < best)
            best = per_iter;
        break;
    }

    stride_plan_destroy(plan);
    ctx->n_benchmarks++;
    _dp_maybe_pace(ctx);
    return best;
}

/* Per-call refine top-K candidate (variants only; multiset/orient supplied
 * by caller). Used by stride_dp_plan_measure to feed deploy rebench. */
typedef struct
{
    vfft_variant_t variants[STRIDE_MAX_STAGES];
    double cost_ns;
} _refine_top_t;

/* Variant cartesian search at one (factors, orientation, K_eff). Walks
 * vfft_variant_iter_*, benches each, returns the best (cost, variants).
 * Returns 1e18 if no valid assignment exists in this orientation.
 *
 * If top_out and max_top > 0, also populates a sorted-ascending top-K list
 * of seen (variants, cost). Caller passes a buffer of size max_top.
 * *top_count is set to the number of valid entries (≤ max_top). */
static double _dp_variant_search(stride_dp_context_t *ctx, int N,
                                 const int *factors, int nf,
                                 int use_dif_forward, size_t K_eff,
                                 const stride_registry_t *reg,
                                 vfft_variant_t *out_best,
                                 long *out_n_assignments,
                                 _refine_top_t *top_out,
                                 int *top_count,
                                 int max_top,
                                 int verbose)
{
    vfft_variant_iter_t it;
    if (!vfft_variant_iter_init(&it, factors, nf, use_dif_forward, reg))
    {
        if (out_n_assignments)
            *out_n_assignments = 0;
        if (top_count)
            *top_count = 0;
        return 1e18;
    }

    double best_ns = 1e18;
    long count = 0;
    int n_top = 0;
    do
    {
        vfft_variant_t cur[STRIDE_MAX_STAGES];
        vfft_variant_iter_get(&it, cur);

        /* Per-variant best-of-runs (Upgrade G, 2026-04-29): each variant
         * config is benched MEASURE_REFINE_RUNS times and we keep the min.
         * Each call to _dp_bench_explicit_one already does best-of-6 trials
         * internally, but adjacent trials within one call have correlated
         * cache state. Multiple calls add fresh memcpy warmups in between,
         * decorrelating the noise. Critical for low-K prime-power cells
         * where variant signal is close to the noise floor — without this
         * the calibrator's pick is noise-driven on those cells. */
        double ns = 1e18;
        for (int run = 0; run < MEASURE_REFINE_RUNS; run++)
        {
            double r = _dp_bench_explicit_one(ctx, N, factors, nf, cur,
                                              use_dif_forward, K_eff, reg);
            if (r < ns)
                ns = r;
        }
        count += MEASURE_REFINE_RUNS;
        if (verbose)
        {
            printf("    [%s] ", use_dif_forward ? "DIF" : "DIT");
            for (int s = 0; s < nf; s++)
                printf("%s%s", s ? "/" : "", vfft_variant_name(cur[s]));
            printf(" = %.1f ns\n", ns);
        }
        if (ns < best_ns)
        {
            best_ns = ns;
            if (out_best)
                memcpy(out_best, cur, nf * sizeof(*out_best));
        }

        /* Maintain sorted top-K (Upgrade H, 2026-04-29): insertion sort into
         * fixed-size array. Cost: O(K) per insert, K = MEASURE_DEPLOY_TOPK_MAX,
         * trivial vs the bench cost. Caller filters by threshold% later. */
        if (top_out && max_top > 0)
        {
            if (n_top < max_top)
            {
                int pos = n_top;
                while (pos > 0 && top_out[pos - 1].cost_ns > ns)
                    pos--;
                for (int i = n_top; i > pos; i--)
                    top_out[i] = top_out[i - 1];
                memcpy(top_out[pos].variants, cur, nf * sizeof(*cur));
                top_out[pos].cost_ns = ns;
                n_top++;
            }
            else if (ns < top_out[max_top - 1].cost_ns)
            {
                int pos = max_top - 1;
                while (pos > 0 && top_out[pos - 1].cost_ns > ns)
                    pos--;
                for (int i = max_top - 1; i > pos; i--)
                    top_out[i] = top_out[i - 1];
                memcpy(top_out[pos].variants, cur, nf * sizeof(*cur));
                top_out[pos].cost_ns = ns;
            }
        }
    } while (vfft_variant_iter_next(&it));

    if (out_n_assignments)
        *out_n_assignments = count;
    if (top_count)
        *top_count = n_top;
    return best_ns;
}

/* Coarse-pass candidate, sorted by default-variant cost. */
typedef struct
{
    int factors[FACT_MAX_STAGES];
    int nf;
    double cost_ns;
} _measure_candidate_t;

static int _measure_cmp(const void *a, const void *b)
{
    double ca = ((const _measure_candidate_t *)a)->cost_ns;
    double cb = ((const _measure_candidate_t *)b)->cost_ns;
    if (ca < cb)
        return -1;
    if (ca > cb)
        return 1;
    return 0;
}

/* Comparator for stride_plan_decision_t by cost_ns ascending. Used for
 * sorting the global pool of top-K refine candidates before threshold
 * filtering (Upgrade H). */
static int _decision_cost_cmp(const void *a, const void *b)
{
    double ca = ((const stride_plan_decision_t *)a)->cost_ns;
    double cb = ((const stride_plan_decision_t *)b)->cost_ns;
    if (ca < cb)
        return -1;
    if (ca > cb)
        return 1;
    return 0;
}

/* DP-driven candidate collection. Calls the top-K-at-every-level
 * recursive solver (_dp_solve_topk) to get the K best multisets for
 * (N, ctx->K), then expands each multiset into all permutations and
 * coarse-benches them. Populates _measure_candidate_t entries for the
 * shared refine pass.
 *
 * Used by stride_dp_plan_measure when N > MEASURE_EXH_THRESHOLD. The
 * top-K-at-every-level recursion (Upgrade D) means runners-up at sub
 * problems are exposed up the stack, so a multiset that would have been
 * pruned by top-1 sub-DP can still surface here when wrapped under a
 * different outer radix. */
static int _measure_collect_via_dp(stride_dp_context_t *ctx, int N,
                                   const stride_registry_t *reg,
                                   int K_top_multisets,
                                   _measure_candidate_t *cands_out,
                                   int max_cands)
{
    if (K_top_multisets > DP_TOPK_MAX)
        K_top_multisets = DP_TOPK_MAX;

    /* Step 1: get top-K multisets via recursive top-K DP. */
    dp_subplan_t plans[DP_TOPK_MAX];
    int n_plans = _dp_solve_topk(ctx, N, ctx->K, reg, plans, K_top_multisets);
    if (n_plans == 0)
        return 0;

    /* Step 2: expand each multiset into all permutations and coarse-bench. */
    int n_cands = 0;
    for (int p = 0; p < n_plans && n_cands < max_cands; p++)
    {
        const int nf = plans[p].nfactors;
        const int *base_factors = plans[p].factors;

        permutation_list_t plist;
        stride_gen_permutations(base_factors, nf, &plist);

        for (int pi = 0; pi < plist.count && n_cands < max_cands; pi++)
        {
            const int *perm = plist.perms[pi];

            int can_build = 1;
            for (int s = 0; s < nf; s++)
            {
                int R = perm[s];
                if (R <= 0 || R >= STRIDE_REG_MAX_RADIX || !reg->n1_fwd[R])
                {
                    can_build = 0;
                    break;
                }
            }
            if (!can_build)
                continue;

            double ns = _dp_bench(ctx, N, perm, nf, ctx->K, reg);
            if (ns >= 1e17)
                continue;

            cands_out[n_cands].nf = nf;
            for (int s = 0; s < nf; s++)
                cands_out[n_cands].factors[s] = perm[s];
            cands_out[n_cands].cost_ns = ns;
            n_cands++;
        }
    }

    return n_cands;
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
 * with deploy-quality bench_plan_min before writing wisdom).
 *
 * Optional top-K outputs (Upgrade H, 2026-04-29): if top_k_out and top_k_count
 * are non-NULL, populates an array of up to MEASURE_DEPLOY_TOPK_MAX candidates
 * within MEASURE_DEPLOY_THRESHOLD_PCT of the global best refine cost. Caller
 * (the calibrator) deploy-rebenches each one with bench_plan_min and picks
 * the actual fastest, resolving variant-axis ties that refine's noisy
 * best-of-N can't disambiguate. top_k_out[0] always equals the same plan
 * as `decision`. */
static double stride_dp_plan_measure(stride_dp_context_t *ctx, int N,
                                     const stride_registry_t *reg,
                                     stride_plan_decision_t *decision,
                                     stride_plan_decision_t *top_k_out,
                                     int *top_k_count,
                                     int verbose)
{
    static _measure_candidate_t cands[MEASURE_MAX_CANDIDATES];
    int n_cands = 0;
    const char *coarse_path = NULL;

    /* Hybrid (Upgrade E, 2026-04-27): pow2 N >threshold uses DP top-K to
     * stay tractable (factorization space explodes); non-pow2 N stays on
     * exhaustive regardless of size because its multiset space is small
     * (constrained by prime factorization), and DP top-K's noise-driven
     * mis-picks at non-pow2 cells were producing systematic regressions
     * (see follow-up notes for N=100000 K=32 etc.). */
    int n_is_pow2 = (N > 0) && ((N & (N - 1)) == 0);
    int use_exhaustive = (N <= MEASURE_EXH_THRESHOLD) || !n_is_pow2;

    if (!use_exhaustive)
    {
        /* Large pow2: DP-driven candidate collection. Outer-only top-K
         * multisets, inner sub-plans solved by top-1 _dp_solve recursion
         * (memoized on (N, K_eff) per Upgrade A). Each top-K multiset is
         * expanded into all permutations; each (multiset, permutation)
         * becomes a _measure_candidate_t for the shared refine pass. */
        coarse_path = "DP";
        n_cands = _measure_collect_via_dp(ctx, N, reg,
                                          MEASURE_DP_TOPK_MULTISETS,
                                          cands, MEASURE_MAX_CANDIDATES);
        if (n_cands == 0)
        {
            if (verbose)
                printf("  N=%d MEASURE-DP: no DP candidates\n", N);
            return 1e18;
        }
    }
    else
    {
        /* Exhaustive enumeration: small pow2 (multiset space small) OR
         * any non-pow2 N (multiset space constrained by prime factors). */
        coarse_path = "exh";
        factorization_list_t flist;
        stride_enumerate_factorizations(N, reg, &flist);
        if (flist.count == 0)
        {
            if (verbose)
                printf("  N=%d MEASURE: no factorizations enumerated\n", N);
            return 1e18;
        }

        for (int fi = 0; fi < flist.count && n_cands < MEASURE_MAX_CANDIDATES; fi++)
        {
            const int nf = flist.results[fi].nfactors;
            const int *base_factors = flist.results[fi].factors;

            permutation_list_t plist;
            stride_gen_permutations(base_factors, nf, &plist);

            for (int pi = 0; pi < plist.count && n_cands < MEASURE_MAX_CANDIDATES; pi++)
            {
                const int *perm = plist.perms[pi];

                int can_build = 1;
                for (int s = 0; s < nf; s++)
                {
                    int R = perm[s];
                    if (R <= 0 || R >= STRIDE_REG_MAX_RADIX || !reg->n1_fwd[R])
                    {
                        can_build = 0;
                        break;
                    }
                }
                if (!can_build)
                    continue;

                double ns = _dp_bench(ctx, N, perm, nf, ctx->K, reg);
                if (ns >= 1e17)
                    continue;

                /* LOG3-aware coarse probe (Upgrade F, 2026-04-29): also bench
                 * with LOG3 forced on every stage where it's registered (DIT
                 * orientation; LOG3 has limited DIF coverage). LOG3 is a
                 * high-leverage variant — for prime radixes (5, 7, 11, 13)
                 * the Winograd codelet can be ±30% vs T1S, with no analytical
                 * predictor. Without this, the coarse pass benches every
                 * multiset with default variants only and a LOG3-friendly
                 * multiset can be eliminated before refine has a chance to
                 * try LOG3 on it. Doubles coarse cost when any stage has
                 * LOG3 available; refine still picks the actual best variants. */
                {
                    vfft_variant_t log3_variants[FACT_MAX_STAGES];
                    int has_log3_eligible = 0;
                    for (int s = 0; s < nf; s++)
                    {
                        if (s == 0)
                        {
                            /* Stage 0 in DIT has no twiddle codelet. */
                            log3_variants[s] = VFFT_VAR_FLAT;
                            continue;
                        }
                        if (vfft_variant_available(reg, perm[s],
                                                   /*use_dif=*/0,
                                                   VFFT_VAR_LOG3))
                        {
                            log3_variants[s] = VFFT_VAR_LOG3;
                            has_log3_eligible = 1;
                        }
                        else if (vfft_variant_available(reg, perm[s], 0,
                                                        VFFT_VAR_T1S))
                        {
                            log3_variants[s] = VFFT_VAR_T1S;
                        }
                        else
                        {
                            log3_variants[s] = VFFT_VAR_FLAT;
                        }
                    }
                    if (has_log3_eligible)
                    {
                        double ns_log3 = _dp_bench_explicit_one(
                            ctx, N, perm, nf, log3_variants,
                            /*use_dif_forward=*/0, ctx->K, reg);
                        if (ns_log3 < ns)
                            ns = ns_log3;
                    }
                }

                cands[n_cands].nf = nf;
                for (int s = 0; s < nf; s++)
                    cands[n_cands].factors[s] = perm[s];
                cands[n_cands].cost_ns = ns;
                n_cands++;
            }
        }

        if (n_cands == 0)
        {
            if (verbose)
                printf("  N=%d MEASURE-exh: no working coarse plans\n", N);
            return 1e18;
        }
    }

    /* Best-of-runs: extra coarse sweeps over the same candidate set.
     * Each (factorization, permutation) gets its min cost across runs.
     * Reduces variance-driven ranking flips that knocked the actual
     * best multiset out of top-K in earlier pilots. */
    for (int run = 1; run < MEASURE_COARSE_RUNS; run++)
    {
        for (int i = 0; i < n_cands; i++)
        {
            double ns = _dp_bench(ctx, N, cands[i].factors, cands[i].nf,
                                  ctx->K, reg);
            if (ns < cands[i].cost_ns)
                cands[i].cost_ns = ns;
        }
    }

    /* Sort, take top-K. */
    qsort(cands, n_cands, sizeof(*cands), _measure_cmp);
    int K_top = MEASURE_TOPK_DEFAULT;
    int n_topk = (n_cands < K_top) ? n_cands : K_top;

    /* REFINE: variant cartesian × {DIT, DIF} on each top-K candidate. */
    double best_ns = 1e18;
    int best_factors[FACT_MAX_STAGES];
    int best_nf = 0;
    vfft_variant_t best_variants[FACT_MAX_STAGES];
    int best_use_dif = 0;
    long total_refine = 0;

    /* Global top-K pool (Upgrade H, 2026-04-29). Each (multiset × orient)
     * call to _dp_variant_search returns its own top-K-by-cost variants;
     * we pool them all here, then filter at the end to within
     * MEASURE_DEPLOY_THRESHOLD_PCT of the global best, capped at
     * MEASURE_DEPLOY_TOPK_MAX. Caller deploy-rebenches the survivors. */
    static stride_plan_decision_t pool[MEASURE_TOPK_DEFAULT * 2 * MEASURE_DEPLOY_TOPK_MAX];
    int n_pool = 0;
    const int pool_max = (int)(sizeof(pool) / sizeof(pool[0]));

    for (int k = 0; k < n_topk; k++)
    {
        const _measure_candidate_t *c = &cands[k];

        for (int orient = 0; orient < 2; orient++)
        {
            vfft_variant_t cur_best[FACT_MAX_STAGES];
            _refine_top_t per_call_top[MEASURE_DEPLOY_TOPK_MAX];
            int per_call_count = 0;
            long n_this = 0;
            double ns = _dp_variant_search(
                ctx, N, c->factors, c->nf,
                orient, ctx->K, reg,
                cur_best, &n_this,
                (top_k_out ? per_call_top : NULL),
                (top_k_out ? &per_call_count : NULL),
                (top_k_out ? MEASURE_DEPLOY_TOPK_MAX : 0),
                0);
            total_refine += n_this;

            /* Aggregate this call's top-K into global pool. */
            if (top_k_out)
            {
                for (int i = 0; i < per_call_count && n_pool < pool_max; i++)
                {
                    stride_plan_decision_t *p = &pool[n_pool++];
                    p->fact.nfactors = c->nf;
                    memcpy(p->fact.factors, c->factors,
                           c->nf * sizeof(int));
                    memcpy(p->variants, per_call_top[i].variants,
                           c->nf * sizeof(vfft_variant_t));
                    p->use_dif_forward = orient;
                    p->cost_ns = per_call_top[i].cost_ns;
                }
            }

            if (ns < best_ns)
            {
                best_ns = ns;
                best_nf = c->nf;
                memcpy(best_factors, c->factors, c->nf * sizeof(int));
                memcpy(best_variants, cur_best,
                       c->nf * sizeof(vfft_variant_t));
                best_use_dif = orient;
            }
        }
    }

    /* Threshold-filter the pool and emit to top_k_out (Upgrade H). */
    if (top_k_out && top_k_count)
    {
        if (n_pool > 0)
        {
            qsort(pool, n_pool, sizeof(*pool), _decision_cost_cmp);
            double pool_best = pool[0].cost_ns;
            double thresh = pool_best *
                            (1.0 + (double)MEASURE_DEPLOY_THRESHOLD_PCT / 100.0);
            int n_emit = 0;
            for (int i = 0; i < n_pool && n_emit < MEASURE_DEPLOY_TOPK_MAX; i++)
            {
                if (pool[i].cost_ns > thresh)
                    break;
                top_k_out[n_emit++] = pool[i];
            }
            *top_k_count = n_emit;
        }
        else
        {
            *top_k_count = 0;
        }
    }

    if (best_ns >= 1e17)
    {
        if (verbose)
            printf("  N=%d MEASURE: no valid refine\n", N);
        return 1e18;
    }

    decision->fact.nfactors = best_nf;
    memcpy(decision->fact.factors, best_factors, best_nf * sizeof(int));
    memcpy(decision->variants, best_variants,
           best_nf * sizeof(vfft_variant_t));
    decision->use_dif_forward = best_use_dif;
    decision->cost_ns = best_ns;

    if (verbose)
    {
        printf("  N=%d K=%zu MEASURE-topk(%d) [%s]: coarse=%d top=%d refine=%ld -> ",
               N, ctx->K, K_top,
               coarse_path ? coarse_path : "?",
               n_cands, n_topk, total_refine);
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
