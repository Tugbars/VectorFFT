/* exhaustive_recursive.h — FFTW-style recursive-solver-tree exhaustive.
 *
 * Why this exists
 * ────────────────
 * Flat exhaustive (see exhaustive_plan.h) enumerates every (multiset ×
 * permutation) of N and benches each as an independent, fully-built
 * plan. At N=8192 K=4 that's 500-1500 candidates; at N=16384 K=256 it
 * runs into hours of bench time.
 *
 * FFTW's exhaustive avoids the blowup by walking a RECURSIVE SOLVER
 * TREE keyed on the problem SIGNATURE (N_sub, K_eff_sub). Many
 * decomposition paths converge to the same sub-problem signature
 * (e.g., from (8192,4): the R=4 child lands at (2048,16); the
 * R=2 → R=2 grandchild also lands at (2048,16)). Memoization at
 * each signature eliminates re-exploration.
 *
 * Estimated bench count for pow2 N=8192 K=4:
 *   ~13 unique signatures × ~6 valid R per signature ≈ 78 benches
 *   (vs ~500-1500 for flat exhaustive)
 *
 * Algorithm (memoized recursion):
 *   plan(N, K_eff):
 *     if cached(N, K_eff): return cached         # signature hit
 *     best = ∅
 *     # Leaf: N itself is a registered radix
 *     if reg.has_codelet(N) && N <= LEAF_MAX:
 *       best = bench_and_score([N], N, K_eff)
 *     # CT-DIT: each applicable radix R as first stage
 *     for R in radixes where R divides N and R != N:
 *       sub = plan(N/R, K_eff * R)                # recurse, cached
 *       cand = [R, *sub.factors]
 *       cost = bench(cand, N, K_eff)
 *       if cost < best.cost: best = (cand, cost)
 *     cache(N, K_eff, best)
 *     return best
 *
 * Differences from src/core/exhaustive.h (flat):
 *   - Flat enumerates multisets then permutations; recursive grows the
 *     plan tree top-down, naturally covering all orderings as a side
 *     effect of trying every R at every level.
 *   - Flat re-builds each candidate from scratch; recursive composes
 *     each parent candidate from already-cached sub-winners.
 *   - Flat is BELIEVE_PCOST==0 (every measurement fresh); recursive is
 *     BELIEVE_PCOST==1 (sub-tree's measured winner is trusted as the
 *     sub-plan for parent composition). Parent ALWAYS benches its own
 *     full composite — so the only "trust" is in WHICH sub-plan to
 *     compose with, not in the composite's cost.
 *
 * Scope:
 *   - DIT only (matches prototype-core).
 *   - T1S variant defaults (matches exhaustive_plan.h).
 *   - Forward direction only.
 *   - Pow2 + supported non-pow2 multisets — same constraint as flat.
 */
#ifndef VFFT_PROTO_CORE_EXHAUSTIVE_RECURSIVE_H
#define VFFT_PROTO_CORE_EXHAUSTIVE_RECURSIVE_H

#include "plan.h"
#include "executor.h"
#include "planner.h"
#include "dp_planner.h"          /* re-use now_ns, factorization_t, pacing */
#include "exhaustive_plan.h"     /* re-use VFFT_PROTO_AVAILABLE_RADIXES_EXH + bench_one */
#include "../prototype/generated/registry.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Leaf cap: only treat N itself as a leaf codelet when N ≤ 64. Above
 * that the codelets are deferred / stubbed / regress vs decomposition.
 * Matches the convention dp_planner.h uses for its base case. */
#ifndef VFFT_PROTO_REX_LEAF_MAX
#define VFFT_PROTO_REX_LEAF_MAX 64
#endif

#ifndef VFFT_PROTO_REX_CACHE_MAX
#define VFFT_PROTO_REX_CACHE_MAX 1024
#endif

/* TOP-K at each signature: number of sub-plan candidates retained.
 *
 * Why >1: a sub-plan benched in isolation at (N_sub, K_eff_sub) measures
 * with cold cache and no upstream state. When that sub-plan runs as the
 * TAIL of a parent composite, its in-context cost can differ enough to
 * change the global winner. Keeping top-K alternatives at each sig lets
 * the parent try each and pick the in-context best.
 *
 * Trade-off: bench count scales ~linearly with K. At K=1 (BELIEVE_PCOST
 * extreme), recursive on N=8192 K=4 ran 63 benches in 0.11s but picked
 * a 28%-worse plan than flat. At K=3, expect ~3× more benches but
 * accuracy approaches flat. Production's dp_planner.h uses K=3 (same
 * default).
 *
 * Higher K = more accurate, lower K = faster. K=1 is FFTW BELIEVE_PCOST.
 * K=∞ effectively reduces to flat exhaustive without the multiset/perm
 * machinery. */
#ifndef VFFT_PROTO_REX_TOPK
#define VFFT_PROTO_REX_TOPK 3
#endif

/* ─────────────────────────────────────────────────────────────────
 * SUB-PLAN CACHE — keyed by (N, K_eff).
 *
 * Why (N, K_eff) is the right signature for our scope:
 *   - Direction is fwd-only (no axis needed).
 *   - In-place (no axis).
 *   - All buffers are 64-aligned (no axis).
 *   - All work is on stride-K layouts; the per-stage stride is
 *     determined by (N_sub, K_eff_sub) which captures group stride
 *     and group count.
 *
 * Production / FFTW would key on more axes (precision, strides,
 * alignment, in/out-of-place, direction). Adding those is mechanical
 * but unnecessary until we have multiple variants of those axes.
 * ───────────────────────────────────────────────────────────────── */

typedef struct {
    int    factors[STRIDE_MAX_STAGES];
    int    nfactors;
    double cost_ns;
    int    valid;   /* 0 if (N, K_eff) was searched and yielded no plan */
} vfft_proto_rex_subplan_t;

typedef struct {
    int    N;
    size_t K_eff;
    vfft_proto_rex_subplan_t plans[VFFT_PROTO_REX_TOPK];
    int    n_plans;
    int    searched;  /* set once the sub-tree at (N, K_eff) has been fully explored */
} vfft_proto_rex_entry_t;

typedef struct {
    vfft_proto_rex_entry_t entries[VFFT_PROTO_REX_CACHE_MAX];
    int count;

    /* Shared bench buffers (re-used across every level). Sized for the
     * top-level (N * K). Sub-levels need (N_sub * K_eff_sub) which is
     * always == N * K (since N_sub * K_eff_sub == N * K is an invariant
     * of the recursion). */
    double *re, *im, *orig_re, *orig_im;
    size_t  buf_total;

    /* Stats. */
    int n_benchmarks;
    int n_cache_hits;
    int n_signatures_planned;

    int verbose;
} vfft_proto_rex_context_t;

static inline void vfft_proto_rex_init(
    vfft_proto_rex_context_t *ctx, int N, size_t K)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->buf_total = (size_t)N * K;

    vfft_proto_posix_memalign((void**)&ctx->re,      64, ctx->buf_total * sizeof(double));
    vfft_proto_posix_memalign((void**)&ctx->im,      64, ctx->buf_total * sizeof(double));
    vfft_proto_posix_memalign((void**)&ctx->orig_re, 64, ctx->buf_total * sizeof(double));
    vfft_proto_posix_memalign((void**)&ctx->orig_im, 64, ctx->buf_total * sizeof(double));

    srand(42);
    for (size_t i = 0; i < ctx->buf_total; i++) {
        ctx->orig_re[i] = (double)rand() / RAND_MAX - 0.5;
        ctx->orig_im[i] = (double)rand() / RAND_MAX - 0.5;
    }
}

static inline void vfft_proto_rex_destroy(vfft_proto_rex_context_t *ctx) {
    vfft_proto_aligned_free(ctx->re);
    vfft_proto_aligned_free(ctx->im);
    vfft_proto_aligned_free(ctx->orig_re);
    vfft_proto_aligned_free(ctx->orig_im);
    memset(ctx, 0, sizeof(*ctx));
}

static inline vfft_proto_rex_entry_t *
_vfft_proto_rex_lookup(vfft_proto_rex_context_t *ctx, int N, size_t K_eff) {
    /* Linear scan — cache typically holds 10-30 entries for pow2 N up
     * to 131072; not worth a hash table at this scale. */
    for (int i = 0; i < ctx->count; i++) {
        if (ctx->entries[i].N == N && ctx->entries[i].K_eff == K_eff)
            return &ctx->entries[i];
    }
    return NULL;
}

static inline vfft_proto_rex_entry_t *
_vfft_proto_rex_insert(vfft_proto_rex_context_t *ctx, int N, size_t K_eff) {
    if (ctx->count >= VFFT_PROTO_REX_CACHE_MAX) return NULL;
    vfft_proto_rex_entry_t *e = &ctx->entries[ctx->count++];
    memset(e, 0, sizeof(*e));
    e->N = N;
    e->K_eff = K_eff;
    return e;
}

/* ─────────────────────────────────────────────────────────────────
 * BENCH — uses exhaustive_plan.h's vfft_proto_bench_one directly
 * (same harness as flat). Shared buffers come from the rex context.
 * ───────────────────────────────────────────────────────────────── */

static inline double _vfft_proto_rex_bench(
    vfft_proto_rex_context_t *ctx, int N, size_t K_eff,
    const int *factors, int nf,
    const vfft_proto_registry_t *reg)
{
    /* Note: bench buffers always sized for (top_N × top_K). At any
     * sub-level (N_sub, K_eff_sub) we still have N_sub * K_eff_sub ≤
     * top_N × top_K (in fact, equal at every level — see header
     * comment for why). */
    size_t total = (size_t)N * K_eff;
    if (total > ctx->buf_total) return 1e18;

    double ns = vfft_proto_bench_one(N, K_eff, factors, nf, reg,
                                      ctx->re, ctx->im,
                                      ctx->orig_re, ctx->orig_im);
    ctx->n_benchmarks++;

    /* Pace exactly like dp_planner.h does — keeps the thermal envelope
     * stable across a long recursive search. Set VFFT_PROTO_DP_PACE_MS=0
     * to disable for fast runs. */
    if (VFFT_PROTO_DP_PACE_MS > 0 &&
        (ctx->n_benchmarks % VFFT_PROTO_DP_PACE_EVERY) == 0) {
        _vfft_proto_dp_sleep_ms(VFFT_PROTO_DP_PACE_MS);
    }
    return ns;
}

/* ─────────────────────────────────────────────────────────────────
 * RECURSIVE PLANNER — the heart of the algorithm.
 *
 * Plans the sub-tree at (N, K_eff) and fills the cache entry with up
 * to TOPK candidates sorted ascending by cost. Returns the number of
 * valid plans found (0 if unplannable).
 *
 * Caller (the parent) reads from ctx's cache directly to get all TOPK
 * sub-plans for composition.
 * ───────────────────────────────────────────────────────────────── */

static int _vfft_proto_rex_subplan_cmp(const void *a, const void *b) {
    double ca = ((const vfft_proto_rex_subplan_t *)a)->cost_ns;
    double cb = ((const vfft_proto_rex_subplan_t *)b)->cost_ns;
    return (ca < cb) ? -1 : (ca > cb) ? 1 : 0;
}

static int _vfft_proto_rex_plan(
    vfft_proto_rex_context_t *ctx, int N, size_t K_eff,
    const vfft_proto_registry_t *reg,
    vfft_proto_rex_entry_t **out_entry)
{
    /* Cache check. */
    vfft_proto_rex_entry_t *cached = _vfft_proto_rex_lookup(ctx, N, K_eff);
    if (cached && cached->searched) {
        ctx->n_cache_hits++;
        if (out_entry) *out_entry = cached;
        return cached->n_plans;
    }

    /* Local accumulator — gather all (leaf, R × cached_sub_TOPK) candidates,
     * sort at the end, retain top-K + base-case-preservation. */
    enum { _ACCUM_MAX = 128 };
    vfft_proto_rex_subplan_t accum[_ACCUM_MAX];
    int n_accum = 0;

    /* ── Solver 1: leaf codelet (if applicable) ───────────────── */
    if (N > 0 && N < VFFT_PROTO_REG_MAX_RADIX && N <= VFFT_PROTO_REX_LEAF_MAX
        && reg->n1_fwd[N] && reg->t1s_dit_fwd[N])
    {
        int leaf[1] = { N };
        double ns = _vfft_proto_rex_bench(ctx, N, K_eff, leaf, 1, reg);
        if (ns < 1e17 && n_accum < _ACCUM_MAX) {
            accum[n_accum].factors[0] = N;
            accum[n_accum].nfactors = 1;
            accum[n_accum].cost_ns = ns;
            accum[n_accum].valid = 1;
            n_accum++;
        }
    }

    /* ── Solver 2: CT-DIT — for each (R, sub-plan in cached TOPK) ── */
    for (const int *rp = VFFT_PROTO_AVAILABLE_RADIXES_EXH;
         *rp && n_accum < _ACCUM_MAX; rp++)
    {
        int R = *rp;
        if (N % R != 0) continue;
        if (R >= N) continue;                          /* leaf handled above */
        if (R >= VFFT_PROTO_REG_MAX_RADIX) continue;
        if (!reg->n1_fwd[R] || !reg->t1s_dit_fwd[R]) continue;

        int M = N / R;
        size_t K_eff_sub = K_eff * (size_t)R;

        /* RECURSIVE: plan (M, K_eff_sub), filling its TOPK cache. */
        vfft_proto_rex_entry_t *sub_entry = NULL;
        int n_sub = _vfft_proto_rex_plan(ctx, M, K_eff_sub, reg, &sub_entry);
        if (!n_sub || !sub_entry) continue;

        /* For EACH cached sub-plan in (M, K_eff_sub)'s TOPK, compose
         * a parent candidate [R, *sub.factors] and bench at (N, K_eff).
         *
         * This is the key difference from BELIEVE_PCOST=K=1: a sub-plan
         * benched in isolation at (M, K_eff_sub) reflects cold-cache
         * isolated behavior. The IN-CONTEXT winner (composed under
         * different parents) may not be the isolated winner. Letting
         * the parent try all top-K sub-candidates restores that signal. */
        for (int si = 0; si < n_sub && n_accum < _ACCUM_MAX; si++) {
            const vfft_proto_rex_subplan_t *sub = &sub_entry->plans[si];
            if (!sub->valid) continue;
            if (sub->nfactors + 1 > STRIDE_MAX_STAGES) continue;

            int cand[STRIDE_MAX_STAGES];
            cand[0] = R;
            memcpy(cand + 1, sub->factors, sub->nfactors * sizeof(int));
            int nf = sub->nfactors + 1;

            double ns = _vfft_proto_rex_bench(ctx, N, K_eff, cand, nf, reg);
            if (ns >= 1e17) continue;

            memcpy(accum[n_accum].factors, cand, nf * sizeof(int));
            accum[n_accum].nfactors = nf;
            accum[n_accum].cost_ns = ns;
            accum[n_accum].valid = 1;
            n_accum++;
        }
    }

    /* Sort candidates ascending by cost, keep top-K. */
    if (n_accum > 1)
        qsort(accum, n_accum, sizeof(vfft_proto_rex_subplan_t),
              _vfft_proto_rex_subplan_cmp);

    int n_keep = (n_accum < VFFT_PROTO_REX_TOPK) ? n_accum : VFFT_PROTO_REX_TOPK;

    /* Base-case preservation: same reasoning as dp_planner.h's TOP-K
     * fix. If N itself is a registered radix ≤ LEAF_MAX, ensure the
     * single-stage [N] plan survives top-K pruning even if it ranks
     * poorly here. [N] benched in isolation at (N, K_eff) measures
     * with cold cache; as the TAIL of a parent composite at a smaller
     * K it may be the right pick. */
    if (N > 0 && N < VFFT_PROTO_REG_MAX_RADIX && N <= VFFT_PROTO_REX_LEAF_MAX
        && reg->n1_fwd[N] && reg->t1s_dit_fwd[N] && n_accum > 0)
    {
        int base_in = 0;
        for (int i = 0; i < n_keep; i++) {
            if (accum[i].nfactors == 1 && accum[i].factors[0] == N) {
                base_in = 1; break;
            }
        }
        if (!base_in) {
            for (int i = n_keep; i < n_accum; i++) {
                if (accum[i].nfactors == 1 && accum[i].factors[0] == N) {
                    if (n_keep > 0) {
                        accum[n_keep - 1] = accum[i];  /* displace worst kept */
                    } else {
                        accum[0] = accum[i]; n_keep = 1;
                    }
                    break;
                }
            }
        }
    }

    /* Cache the result. */
    vfft_proto_rex_entry_t *e = cached ? cached : _vfft_proto_rex_insert(ctx, N, K_eff);
    if (e) {
        for (int i = 0; i < n_keep; i++) e->plans[i] = accum[i];
        e->n_plans = n_keep;
        e->searched = 1;
    }
    ctx->n_signatures_planned++;

    if (ctx->verbose >= 2 && n_keep > 0) {
        printf("    [rex] cached (N=%d, K_eff=%zu) topK:\n", N, K_eff);
        for (int i = 0; i < n_keep; i++) {
            printf("      ");
            for (int s = 0; s < accum[i].nfactors; s++)
                printf("%s%d", s ? "x" : "", accum[i].factors[s]);
            printf(" = %.1f ns\n", accum[i].cost_ns);
        }
    }

    if (out_entry) *out_entry = e;
    return n_keep;
}

/* ─────────────────────────────────────────────────────────────────
 * TOP-LEVEL ENTRIES
 * ───────────────────────────────────────────────────────────────── */

static inline double vfft_proto_recursive_exhaustive_search(
    int N, size_t K,
    const vfft_proto_registry_t *reg,
    vfft_proto_factorization_t *best_fact,
    int verbose)
{
    vfft_proto_rex_context_t ctx;
    vfft_proto_rex_init(&ctx, N, K);
    ctx.verbose = verbose;

    if (verbose >= 1)
        printf("  [recursive-exhaustive] N=%d K=%zu — descending...\n",
               N, (size_t)K);

    vfft_proto_rex_entry_t *top = NULL;
    int n_top = _vfft_proto_rex_plan(&ctx, N, K, reg, &top);

    double result_ns = 1e18;
    best_fact->nfactors = 0;
    if (n_top > 0 && top && top->plans[0].valid) {
        result_ns = top->plans[0].cost_ns;
        best_fact->nfactors = top->plans[0].nfactors;
        memcpy(best_fact->factors, top->plans[0].factors,
               (size_t)top->plans[0].nfactors * sizeof(int));
    }

    if (verbose >= 1) {
        printf("  [recursive-exhaustive] Best: ");
        for (int s = 0; s < best_fact->nfactors; s++)
            printf("%s%d", s ? "x" : "", best_fact->factors[s]);
        printf(" = %.1f ns\n", result_ns);
        printf("  [recursive-exhaustive] stats: %d signatures planned, "
               "%d benches, %d cache hits (TOP-K=%d)\n",
               ctx.n_signatures_planned, ctx.n_benchmarks, ctx.n_cache_hits,
               VFFT_PROTO_REX_TOPK);
    }

    vfft_proto_rex_destroy(&ctx);
    return result_ns;
}

/* Top-level convenience: returns a plan built from the recursive
 * search's pick. Matches vfft_proto_exhaustive_plan in exhaustive_plan.h. */
static inline stride_plan_t *vfft_proto_recursive_exhaustive_plan(
    int N, size_t K, const vfft_proto_registry_t *reg, int verbose)
{
    vfft_proto_factorization_t best;
    double ns = vfft_proto_recursive_exhaustive_search(N, K, reg, &best, verbose);
    (void)ns;
    if (best.nfactors == 0) return NULL;
    return vfft_proto_plan_create(N, K, best.factors, /*variants=*/NULL,
                                  best.nfactors, reg);
}

/* Verbose variant: reports picked factorization + measured ns + walltime. */
static inline stride_plan_t *vfft_proto_recursive_exhaustive_plan_verbose(
    int N, size_t K, const vfft_proto_registry_t *reg,
    int *out_factors, int *out_nf, double *out_ns,
    int *out_n_benches, int *out_n_sigs, int *out_n_hits,
    int verbose)
{
    vfft_proto_rex_context_t ctx;
    vfft_proto_rex_init(&ctx, N, K);
    ctx.verbose = verbose;

    if (verbose >= 1)
        printf("  [recursive-exhaustive] N=%d K=%zu — descending...\n",
               N, (size_t)K);

    vfft_proto_rex_entry_t *top = NULL;
    int n_top = _vfft_proto_rex_plan(&ctx, N, K, reg, &top);

    stride_plan_t *plan = NULL;
    if (n_top > 0 && top && top->plans[0].valid) {
        const vfft_proto_rex_subplan_t *best = &top->plans[0];
        if (out_factors)
            memcpy(out_factors, best->factors,
                   (size_t)best->nfactors * sizeof(int));
        if (out_nf) *out_nf = best->nfactors;
        if (out_ns) *out_ns = best->cost_ns;
        plan = vfft_proto_plan_create(N, K, best->factors, /*variants=*/NULL,
                                      best->nfactors, reg);
    } else {
        if (out_nf) *out_nf = 0;
        if (out_ns) *out_ns = 1e18;
    }

    if (out_n_benches) *out_n_benches = ctx.n_benchmarks;
    if (out_n_sigs)    *out_n_sigs    = ctx.n_signatures_planned;
    if (out_n_hits)    *out_n_hits    = ctx.n_cache_hits;

    vfft_proto_rex_destroy(&ctx);
    return plan;
}

#endif /* VFFT_PROTO_CORE_EXHAUSTIVE_RECURSIVE_H */
