/* dp_planner.h — prototype-core port of src/core/dp_planner.h
 *
 * Recursive DP planner with memoization. Same algorithm as production:
 *
 *   1. For N at batch K_eff, try each valid radix R as first stage.
 *   2. Recursively get top-K best plans for N/R at K_eff * R.
 *   3. For each sub-plan, build [R, sub_factors...] and bench the full plan.
 *   4. Cache top-K by (N, K_eff). Subsequent same-key lookups return
 *      cached plans (BELIEVE_PCOST semantics — equivalent to FFTW MEASURE).
 *   5. After DP yields the best multiset, enumerate its permutations and
 *      pick the fastest ordering.
 *
 * Scope vs production's dp_planner.h (1283 lines):
 *   IN:   core DP + top-K + permutation phase 2 (the algorithm)
 *   OUT:  VFFT_MEASURE variant cartesian (stride_dp_plan_measure et al.) —
 *         needs a variant iterator we haven't ported; separate workstream.
 *
 * Differences from production:
 *   - Calls vfft_proto_plan_create with variants=NULL (T1S defaults). The
 *     DP measures pure-T1S plans; a follow-up variant pass would refine.
 *   - Self-contained now_ns + permutation_list_t (no compat.h / exhaustive.h
 *     dependency).
 *
 * Usage:
 *   vfft_proto_dp_context_t ctx;
 *   vfft_proto_dp_init(&ctx, K, max_N);
 *   vfft_proto_factorization_t best;
 *   double ns = vfft_proto_dp_plan(&ctx, N, &reg, &best, verbose);
 *   vfft_proto_dp_destroy(&ctx);
 */
#ifndef VFFT_PROTO_DP_PLANNER_H
#define VFFT_PROTO_DP_PLANNER_H

#include "plan.h"
#include "planner.h"           /* vfft_proto_plan_create, factorize helpers */
#include "executor.h"          /* vfft_proto_execute_fwd */
#include "../prototype/generated/registry.h"  /* registry types */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
static inline double vfft_proto_now_ns(void) {
    static LARGE_INTEGER freq = {0};
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart * 1e9;
}
#else
#  include <time.h>
static inline double vfft_proto_now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
#endif

/* ─────────────────────────────────────────────────────────────────
 * Factorization + permutation types — minimal port from exhaustive.h
 * ───────────────────────────────────────────────────────────────── */

#define VFFT_PROTO_DP_MAX_STAGES STRIDE_MAX_STAGES
#define VFFT_PROTO_DP_MAX_PERMS  720  /* 6! = max useful */

typedef struct {
    int factors[VFFT_PROTO_DP_MAX_STAGES];
    int nfactors;
} vfft_proto_factorization_t;

typedef struct {
    int perms[VFFT_PROTO_DP_MAX_PERMS][VFFT_PROTO_DP_MAX_STAGES];
    int count;
    int nf;
} vfft_proto_perm_list_t;

static inline void _vfft_proto_sort_asc(int *arr, int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) { arr[j+1] = arr[j]; j--; }
        arr[j+1] = key;
    }
}

static inline void _vfft_proto_reverse(int *arr, int l, int r) {
    while (l < r) { int t = arr[l]; arr[l] = arr[r]; arr[r] = t; l++; r--; }
}

static inline int _vfft_proto_next_perm(int *arr, int n) {
    int i = n - 2;
    while (i >= 0 && arr[i] >= arr[i+1]) i--;
    if (i < 0) return 0;
    int j = n - 1;
    while (arr[j] <= arr[i]) j--;
    int t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    _vfft_proto_reverse(arr, i + 1, n - 1);
    return 1;
}

static inline void vfft_proto_gen_permutations(
    const int *factors, int nf, vfft_proto_perm_list_t *list)
{
    list->count = 0;
    list->nf = nf;
    int work[VFFT_PROTO_DP_MAX_STAGES];
    memcpy(work, factors, nf * sizeof(int));
    _vfft_proto_sort_asc(work, nf);
    do {
        if (list->count >= VFFT_PROTO_DP_MAX_PERMS) break;
        memcpy(list->perms[list->count], work, nf * sizeof(int));
        list->count++;
    } while (_vfft_proto_next_perm(work, nf));
}

/* ─────────────────────────────────────────────────────────────────
 * DP CACHE (top-K-at-every-level, keyed by (N, K_eff))
 * ───────────────────────────────────────────────────────────────── */

#define VFFT_PROTO_DP_CACHE_MAX 512
#ifndef VFFT_PROTO_DP_TOPK_MAX
#define VFFT_PROTO_DP_TOPK_MAX 3
#endif

typedef struct {
    int    factors[VFFT_PROTO_DP_MAX_STAGES];
    int    nfactors;
    double cost_ns;
} vfft_proto_dp_subplan_t;

typedef struct {
    int    N;
    size_t K_eff;
    vfft_proto_dp_subplan_t plans[VFFT_PROTO_DP_TOPK_MAX];
    int    n_plans;
} vfft_proto_dp_entry_t;

typedef struct {
    vfft_proto_dp_entry_t entries[VFFT_PROTO_DP_CACHE_MAX];
    int    count;
    size_t K;
    /* Shared bench buffers, sized for max_N * K. */
    double *re, *im, *orig_re, *orig_im;
    size_t  buf_total;
    int     max_N;
    /* MEASURE (default) trusts cached cost on hit; PATIENT (=0) re-benches. */
    int     believe_subplan_cost;
    int     n_benchmarks;
    int     n_cache_hits;
} vfft_proto_dp_context_t;

static inline void vfft_proto_dp_init(vfft_proto_dp_context_t *ctx,
                                       size_t K, int max_N)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->K = K;
    ctx->max_N = max_N;
    ctx->buf_total = (size_t)max_N * K;
    ctx->believe_subplan_cost = 1;

    vfft_proto_posix_memalign((void **)&ctx->re,      64, ctx->buf_total * sizeof(double));
    vfft_proto_posix_memalign((void **)&ctx->im,      64, ctx->buf_total * sizeof(double));
    vfft_proto_posix_memalign((void **)&ctx->orig_re, 64, ctx->buf_total * sizeof(double));
    vfft_proto_posix_memalign((void **)&ctx->orig_im, 64, ctx->buf_total * sizeof(double));

    srand(42);
    for (size_t i = 0; i < ctx->buf_total; i++) {
        ctx->orig_re[i] = (double)rand() / RAND_MAX - 0.5;
        ctx->orig_im[i] = (double)rand() / RAND_MAX - 0.5;
    }
}

static inline void vfft_proto_dp_destroy(vfft_proto_dp_context_t *ctx) {
    vfft_proto_aligned_free(ctx->re);
    vfft_proto_aligned_free(ctx->im);
    vfft_proto_aligned_free(ctx->orig_re);
    vfft_proto_aligned_free(ctx->orig_im);
    memset(ctx, 0, sizeof(*ctx));
}

static inline vfft_proto_dp_entry_t *
_vfft_proto_dp_lookup(vfft_proto_dp_context_t *ctx, int N, size_t K_eff) {
    for (int i = 0; i < ctx->count; i++)
        if (ctx->entries[i].N == N && ctx->entries[i].K_eff == K_eff)
            return &ctx->entries[i];
    return NULL;
}

static inline vfft_proto_dp_entry_t *
_vfft_proto_dp_insert(vfft_proto_dp_context_t *ctx, int N, size_t K_eff) {
    if (ctx->count >= VFFT_PROTO_DP_CACHE_MAX) return NULL;
    vfft_proto_dp_entry_t *e = &ctx->entries[ctx->count++];
    memset(e, 0, sizeof(*e));
    e->N = N;
    e->K_eff = K_eff;
    return e;
}

static inline int _vfft_proto_dp_subplan_cmp(const void *a, const void *b) {
    double ca = ((const vfft_proto_dp_subplan_t *)a)->cost_ns;
    double cb = ((const vfft_proto_dp_subplan_t *)b)->cost_ns;
    return (ca < cb) ? -1 : (ca > cb) ? 1 : 0;
}

/* ─────────────────────────────────────────────────────────────────
 * BENCH HELPER — FFTW-style adaptive timer (mirrors production's
 * src/core/dp_planner.h:_dp_bench). Buf-reset per trial keeps the
 * data path deterministic across repeats.
 * ───────────────────────────────────────────────────────────────── */

#ifndef VFFT_PROTO_DP_TIME_REPEAT
#define VFFT_PROTO_DP_TIME_REPEAT 6
#endif
#ifndef VFFT_PROTO_DP_TIME_MIN_NS
#define VFFT_PROTO_DP_TIME_MIN_NS 2.0e6
#endif
#ifndef VFFT_PROTO_DP_TIME_LIMIT_NS
#define VFFT_PROTO_DP_TIME_LIMIT_NS 5.0e8
#endif

/* Inter-bench pacing — sleep MS between benches to keep package thermal
 * envelope from drifting up over a minutes-long DP search. Production's
 * version only paces at K > 64; we always pace by default (configurable).
 *
 * Pacing sleeps a fixed amount per bench (set VFFT_PROTO_DP_PACE_MS=0 to
 * disable for short searches where heat isn't an issue). 1000ms matches
 * the user's "1 second delay" request to also let HW prefetchers /
 * cache state quiet down between benches. */
#ifndef VFFT_PROTO_DP_PACE_MS
#define VFFT_PROTO_DP_PACE_MS 1000
#endif
#ifndef VFFT_PROTO_DP_PACE_EVERY
#define VFFT_PROTO_DP_PACE_EVERY 1     /* sleep after every bench */
#endif

#ifdef _WIN32
#ifndef _WINDOWS_
extern __declspec(dllimport) void __stdcall Sleep(unsigned long ms);
#endif
static inline void _vfft_proto_dp_sleep_ms(int ms) {
    if (ms > 0) Sleep((unsigned long)ms);
}
#else
#include <unistd.h>
static inline void _vfft_proto_dp_sleep_ms(int ms) {
    if (ms > 0) usleep((useconds_t)ms * 1000);
}
#endif

static inline double _vfft_proto_dp_bench(
    vfft_proto_dp_context_t *ctx, int N,
    const int *factors, int nf, size_t K_eff,
    const vfft_proto_registry_t *reg)
{
    size_t total = (size_t)N * K_eff;

    /* Bench buffers must be large enough. */
    if (total > ctx->buf_total) return 1e18;

    /* All radixes must have n1 + t1s available (current path: T1S only). */
    for (int s = 0; s < nf; s++) {
        int R = factors[s];
        if (!reg->n1_fwd[R] || !reg->t1s_dit_fwd[R]) return 1e18;
    }

    /* Build a plan with T1S defaults (variants=NULL) — same shape as
     * vfft_proto_auto_plan in estimate mode. */
    stride_plan_t *plan = vfft_proto_plan_create(
        N, K_eff, factors, /*variants=*/NULL, nf, reg);
    if (!plan) return 1e18;

    /* Warmup. */
    memcpy(ctx->re, ctx->orig_re, total * sizeof(double));
    memcpy(ctx->im, ctx->orig_im, total * sizeof(double));
    vfft_proto_execute_fwd(plan, ctx->re, ctx->im, K_eff);

    /* Adaptive iter count: double `reps` until trial >= DP_TIME_MIN_NS,
     * then take best-of DP_TIME_REPEAT trials. */
    double best = 1e30;
    double total_elapsed = 0.0;
    int reps = 1;
    int calibrated = 0;

    for (int outer = 0; outer < 32 && total_elapsed < VFFT_PROTO_DP_TIME_LIMIT_NS; outer++) {
        double tmin_trial = 1e30;
        for (int t = 0; t < VFFT_PROTO_DP_TIME_REPEAT; t++) {
            memcpy(ctx->re, ctx->orig_re, total * sizeof(double));
            memcpy(ctx->im, ctx->orig_im, total * sizeof(double));
            double t0 = vfft_proto_now_ns();
            for (int i = 0; i < reps; i++)
                vfft_proto_execute_fwd(plan, ctx->re, ctx->im, K_eff);
            double trial_ns = vfft_proto_now_ns() - t0;
            if (trial_ns < tmin_trial) tmin_trial = trial_ns;
            total_elapsed += trial_ns;
            if (total_elapsed >= VFFT_PROTO_DP_TIME_LIMIT_NS) break;
        }
        if (!calibrated) {
            if (tmin_trial < VFFT_PROTO_DP_TIME_MIN_NS) {
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

    vfft_proto_plan_destroy(plan);
    ctx->n_benchmarks++;
    /* Thermal + cache pacing — let the package cool and HW prefetchers
     * settle before the next bench. Critical for stable measurements at
     * K > 64 where sustained 100% core load over a long DP search drifts
     * numbers up minute by minute. */
    if (VFFT_PROTO_DP_PACE_MS > 0 &&
        (ctx->n_benchmarks % VFFT_PROTO_DP_PACE_EVERY) == 0) {
        _vfft_proto_dp_sleep_ms(VFFT_PROTO_DP_PACE_MS);
    }
    return best;
}

/* ─────────────────────────────────────────────────────────────────
 * RECURSIVE TOP-K DP SOLVER
 * ───────────────────────────────────────────────────────────────── */

/* Available radixes for DP decomposition, largest first. Mirrors
 * production's DP_RADIXES (src/core/dp_planner.h:351). */
static const int VFFT_PROTO_DP_RADIXES[] = {
    64, 32, 25, 20, 16, 12, 10, 8, 7, 6, 5, 4, 3, 2,
    19, 17, 13, 11, 0
};

static int _vfft_proto_dp_solve_topk(
    vfft_proto_dp_context_t *ctx, int N, size_t K_eff,
    const vfft_proto_registry_t *reg,
    vfft_proto_dp_subplan_t *out, int max_out)
{
    if (max_out <= 0) return 0;

    /* Cache check. */
    vfft_proto_dp_entry_t *cached = _vfft_proto_dp_lookup(ctx, N, K_eff);
    if (cached) {
        ctx->n_cache_hits++;
        int n = (cached->n_plans < max_out) ? cached->n_plans : max_out;
        for (int i = 0; i < n; i++) out[i] = cached->plans[i];
        if (!ctx->believe_subplan_cost && n > 0) {
            double fresh = _vfft_proto_dp_bench(
                ctx, N, cached->plans[0].factors,
                cached->plans[0].nfactors, K_eff, reg);
            cached->plans[0].cost_ns = fresh;
            out[0].cost_ns = fresh;
        }
        return n;
    }

    enum { _ACCUM_MAX = 64 };
    vfft_proto_dp_subplan_t accum[_ACCUM_MAX];
    int n_accum = 0;

    /* Base case: N itself is a registered radix ≤ 64 — one-stage plan. */
    if (N > 0 && N < VFFT_PROTO_REG_MAX_RADIX && reg->n1_fwd[N] && N <= 64) {
        int factors[1] = {N};
        double ns = _vfft_proto_dp_bench(ctx, N, factors, 1, K_eff, reg);
        if (ns < 1e17 && n_accum < _ACCUM_MAX) {
            accum[n_accum].factors[0] = N;
            accum[n_accum].nfactors = 1;
            accum[n_accum].cost_ns = ns;
            n_accum++;
        }
    }

    /* Recursive case: try each R as first stage, recurse on N/R. */
    for (const int *rp = VFFT_PROTO_DP_RADIXES; *rp && n_accum < _ACCUM_MAX; rp++) {
        int R = *rp;
        if (N % R != 0) continue;
        if (R >= VFFT_PROTO_REG_MAX_RADIX || !reg->n1_fwd[R]) continue;
        if (R == N) continue; /* base case above */

        int M = N / R;
        if (M < 1) continue;

        if (M == 1) {
            int factors[1] = {R};
            double ns = _vfft_proto_dp_bench(ctx, N, factors, 1, K_eff, reg);
            if (ns < 1e17 && n_accum < _ACCUM_MAX) {
                accum[n_accum].factors[0] = R;
                accum[n_accum].nfactors = 1;
                accum[n_accum].cost_ns = ns;
                n_accum++;
            }
            continue;
        }

        /* Sub-problem's K_eff scales by R. */
        size_t K_eff_sub = K_eff * (size_t)R;
        vfft_proto_dp_subplan_t sub[VFFT_PROTO_DP_TOPK_MAX];
        int n_sub = _vfft_proto_dp_solve_topk(ctx, M, K_eff_sub, reg, sub,
                                               VFFT_PROTO_DP_TOPK_MAX);

        for (int s = 0; s < n_sub && n_accum < _ACCUM_MAX; s++) {
            if (sub[s].cost_ns >= 1e17) continue;
            if (sub[s].nfactors + 1 > VFFT_PROTO_DP_MAX_STAGES) continue;

            int cand[VFFT_PROTO_DP_MAX_STAGES];
            cand[0] = R;
            memcpy(cand + 1, sub[s].factors, sub[s].nfactors * sizeof(int));
            int nf = sub[s].nfactors + 1;

            double ns = _vfft_proto_dp_bench(ctx, N, cand, nf, K_eff, reg);
            if (ns >= 1e17) continue;

            memcpy(accum[n_accum].factors, cand, nf * sizeof(int));
            accum[n_accum].nfactors = nf;
            accum[n_accum].cost_ns = ns;
            n_accum++;
        }
    }

    if (n_accum > 1)
        qsort(accum, n_accum, sizeof(vfft_proto_dp_subplan_t),
              _vfft_proto_dp_subplan_cmp);

    int n_keep = (n_accum < VFFT_PROTO_DP_TOPK_MAX) ? n_accum : VFFT_PROTO_DP_TOPK_MAX;
    vfft_proto_dp_entry_t *e = _vfft_proto_dp_insert(ctx, N, K_eff);
    if (e) {
        for (int i = 0; i < n_keep; i++) e->plans[i] = accum[i];
        e->n_plans = n_keep;
    }

    int n_out = (n_keep < max_out) ? n_keep : max_out;
    for (int i = 0; i < n_out; i++) out[i] = accum[i];
    return n_out;
}

/* Backward-compat: returns top-1 in (factors, nf, cost) shape. */
static inline double _vfft_proto_dp_solve(
    vfft_proto_dp_context_t *ctx, int N, size_t K_eff,
    const vfft_proto_registry_t *reg, int *out_factors, int *out_nf)
{
    vfft_proto_dp_subplan_t top1;
    int n = _vfft_proto_dp_solve_topk(ctx, N, K_eff, reg, &top1, 1);
    if (n == 0) { *out_nf = 0; return 1e18; }
    memcpy(out_factors, top1.factors, top1.nfactors * sizeof(int));
    *out_nf = top1.nfactors;
    return top1.cost_ns;
}

/* ─────────────────────────────────────────────────────────────────
 * TOP-LEVEL DP PLANNER — DP + permutation phase 2
 * ───────────────────────────────────────────────────────────────── */

static inline double vfft_proto_dp_plan(
    vfft_proto_dp_context_t *ctx, int N,
    const vfft_proto_registry_t *reg,
    vfft_proto_factorization_t *best_fact,
    int verbose)
{
    /* Phase 1: recursive DP yields the best multiset (factor set) for N. */
    int dp_factors[VFFT_PROTO_DP_MAX_STAGES];
    int dp_nf = 0;
    double dp_ns = _vfft_proto_dp_solve(ctx, N, ctx->K, reg, dp_factors, &dp_nf);

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

    /* Phase 2: enumerate all permutations of the winning multiset, bench
     * each, return the fastest ordering. The DP recursion always places
     * R first, which may not be optimal (e.g., large radix last for
     * better stride access). */
    vfft_proto_perm_list_t plist;
    vfft_proto_gen_permutations(dp_factors, dp_nf, &plist);

    double global_best = dp_ns;
    best_fact->nfactors = dp_nf;
    memcpy(best_fact->factors, dp_factors, dp_nf * sizeof(int));

    for (int pi = 0; pi < plist.count; pi++) {
        /* Skip the ordering we already benched. */
        int same = 1;
        for (int s = 0; s < dp_nf; s++)
            if (plist.perms[pi][s] != dp_factors[s]) { same = 0; break; }
        if (same) continue;

        double ns = _vfft_proto_dp_bench(ctx, N, plist.perms[pi], dp_nf, ctx->K, reg);
        if (ns < global_best) {
            global_best = ns;
            memcpy(best_fact->factors, plist.perms[pi], dp_nf * sizeof(int));
        }
    }

    if (verbose && global_best < dp_ns) {
        printf("  N=%d: reordering improved to ", N);
        for (int s = 0; s < best_fact->nfactors; s++)
            printf("%s%d", s ? "x" : "", best_fact->factors[s]);
        printf(" = %.1f ns\n", global_best);
    }

    return global_best;
}

#endif /* VFFT_PROTO_DP_PLANNER_H */
