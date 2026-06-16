/**
 * vfft_proto_dp_planner.h -- Recursive dynamic programming planner
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
 *   vfft_proto_dp_context_t ctx;
 *   vfft_proto_dp_init(&ctx, K);
 *   vfft_proto_factorization_t best;
 *   double ns = vfft_proto_dp_plan(&ctx, N, &reg, &best);
 *   vfft_proto_dp_destroy(&ctx);
 */
#ifndef VFFT_PROTO_DP_PLANNER_H
#define VFFT_PROTO_DP_PLANNER_H

/* Wholesale port of src/core/dp_planner.h (lines 1-598; MEASURE wrapper
 * skipped — separate variant-cartesian workstream). Mechanical
 * stride_* → vfft_proto_* renames. Dependency wiring to prototype-core:
 *
 *   _stride_build_plan(N, K, factors, nf, reg)  →
 *     vfft_proto_plan_create(N, K, factors, NULL, nf, reg)
 *   stride_execute_fwd(plan, re, im)             →
 *     vfft_proto_execute_fwd(plan, re, im, K_eff)
 *   vfft_proto_plan_destroy(plan)                    →
 *     vfft_proto_plan_destroy(plan)
 *   (R > 0 && R < VFFT_PROTO_REG_MAX_RADIX && reg->n1_fwd[R])                  →
 *     (R > 0 && R < VFFT_PROTO_REG_MAX_RADIX && reg->n1_fwd[R] != NULL)
 *   STRIDE_ALIGNED_ALLOC(64, sz)                 →
 *     _vfft_proto_dp_aligned_alloc(64, sz)  (vfft_proto_posix_memalign wrapper)
 *   STRIDE_ALIGNED_FREE                          →
 *     vfft_proto_aligned_free
 *   FACT_MAX_STAGES                              →  STRIDE_MAX_STAGES
 *   permutation_list_t                           →  vfft_proto_perm_list_t
 *   stride_gen_permutations                      →  vfft_proto_gen_permutations
 *   now_ns                                       →  vfft_proto_now_ns
 */

#include "plan.h"          /* stride_plan_t, vfft_proto_posix_memalign */
#include "planner.h"       /* vfft_proto_plan_create, vfft_proto_plan_destroy */
#include "executor.h"      /* vfft_proto_execute_fwd */
#include "../generator/generated/registry.h"  /* vfft_proto_registry_t, VFFT_PROTO_REG_MAX_RADIX */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* now_ns + aligned-alloc helpers + permutation_list_t types — pulled
 * in from prototype-core's bits. */

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
static inline double vfft_proto_now_ns(void) {
    static LARGE_INTEGER freq = {0};
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER t; QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart * 1e9;
}
#else
#  include <time.h>
static inline double vfft_proto_now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
#endif

/* Aligned-alloc wrapper. Production uses STRIDE_ALIGNED_ALLOC (returns
 * pointer); prototype-core's vfft_proto_posix_memalign uses POSIX shape
 * (returns int, writes pointer via out-arg). Adapt with a thin wrapper. */
static inline void *_vfft_proto_dp_aligned_alloc(size_t align, size_t size) {
    void *p = NULL;
    if (vfft_proto_posix_memalign(&p, align, size) != 0) return NULL;
    return p;
}

/* Factorization + permutation types — ported from exhaustive_plan.h via
 * forward references. The same types are used by exhaustive_plan.h. */
#define VFFT_PROTO_DP_MAX_STAGES STRIDE_MAX_STAGES
#define VFFT_PROTO_DP_MAX_PERMS  720  /* 6! = max useful */

typedef struct {
    int factors[VFFT_PROTO_DP_MAX_STAGES];
    int nfactors;
    /* Winning per-stage codelet variant from the joint search (0=FLAT 1=LOG3
     * 2=T1S). All-T1S until a variant-aware search fills it. Stage 0 is the
     * no-twiddle stage so its entry is moot. */
    int variants[VFFT_PROTO_DP_MAX_STAGES];
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

#define VFFT_PROTO_DP_CACHE_MAX 512

/* Top-K-at-every-level (Upgrade D, 2026-04-27).
 *
 * Each cache row stores up to VFFT_PROTO_DP_TOPK_MAX best plans for (N, K_eff).
 * The recursion exposes runners-up to outer levels so that a
 * factorization that lost the top-1 race in isolation can still be
 * composed under a different outer radix and win there.
 *
 * Concretely fixes the N=32768 K=4 regression where outer R=4 got
 * sub-DP(8192, K_eff=16)'s top-1 plan but missed [4,32,64] (a runner-up
 * that, wrapped under R=4, would have produced [4,4,32,64] beating the
 * eventual [64,8,64] winner).
 *
 * Cost: |R| * VFFT_PROTO_DP_TOPK_MAX benches per cache-miss call (vs |R|), ~3x DP
 * overhead at K=3. Cache memory: VFFT_PROTO_DP_CACHE_MAX * VFFT_PROTO_DP_TOPK_MAX * sizeof(plan)
 * = ~120KB at K=3, bounded. */
#ifndef VFFT_PROTO_DP_TOPK_MAX
#define VFFT_PROTO_DP_TOPK_MAX 3
#endif

typedef struct
{
    int factors[STRIDE_MAX_STAGES];
    int nfactors;
    double cost_ns;
} vfft_proto_dp_subplan_t;

typedef struct
{
    int N;
    size_t K_eff;                    /* effective batch at lookup time */
    vfft_proto_dp_subplan_t plans[VFFT_PROTO_DP_TOPK_MAX]; /* sorted by cost, ascending */
    int n_plans;                     /* 0 .. VFFT_PROTO_DP_TOPK_MAX */
} vfft_proto_dp_entry_t;

typedef struct
{
    vfft_proto_dp_entry_t entries[VFFT_PROTO_DP_CACHE_MAX];
    int count;
    size_t K; /* top-level K_outer */

    /* Shared benchmark buffers (allocated for max N*K) */
    double *re, *im, *orig_re, *orig_im;
    size_t buf_total; /* current buffer size in elements */
    int max_N;        /* largest N we'll plan for */

    /* Compositional-trust toggle (set by user before vfft_proto_dp_plan).
     * 1 = trust cached pcost (default, MEASURE).
     * 0 = re-measure on cache hit (PATIENT). */
    int believe_subplan_cost;

    /* Statistics */
    int n_benchmarks;
    int n_cache_hits;
} vfft_proto_dp_context_t;

static void vfft_proto_dp_init(vfft_proto_dp_context_t *ctx, size_t K, int max_N)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->K = K;
    ctx->max_N = max_N;
    ctx->buf_total = (size_t)max_N * K;
    ctx->believe_subplan_cost = 1; /* default: MEASURE semantics */

    ctx->re = (double *)_vfft_proto_dp_aligned_alloc(64, ctx->buf_total * sizeof(double));
    ctx->im = (double *)_vfft_proto_dp_aligned_alloc(64, ctx->buf_total * sizeof(double));
    ctx->orig_re = (double *)_vfft_proto_dp_aligned_alloc(64, ctx->buf_total * sizeof(double));
    ctx->orig_im = (double *)_vfft_proto_dp_aligned_alloc(64, ctx->buf_total * sizeof(double));

    srand(42);
    for (size_t i = 0; i < ctx->buf_total; i++)
    {
        ctx->orig_re[i] = (double)rand() / RAND_MAX - 0.5;
        ctx->orig_im[i] = (double)rand() / RAND_MAX - 0.5;
    }
}

static void vfft_proto_dp_destroy(vfft_proto_dp_context_t *ctx)
{
    vfft_proto_aligned_free(ctx->re);
    vfft_proto_aligned_free(ctx->im);
    vfft_proto_aligned_free(ctx->orig_re);
    vfft_proto_aligned_free(ctx->orig_im);
    memset(ctx, 0, sizeof(*ctx));
}

/* Cache lookup — keyed by (N, K_eff). */
static vfft_proto_dp_entry_t *_vfft_proto_dp_lookup(vfft_proto_dp_context_t *ctx, int N, size_t K_eff)
{
    for (int i = 0; i < ctx->count; i++)
        if (ctx->entries[i].N == N && ctx->entries[i].K_eff == K_eff)
            return &ctx->entries[i];
    return NULL;
}

/* Cache insert — keyed by (N, K_eff). Plans array starts empty. */
static vfft_proto_dp_entry_t *_vfft_proto_dp_insert(vfft_proto_dp_context_t *ctx, int N, size_t K_eff)
{
    if (ctx->count >= VFFT_PROTO_DP_CACHE_MAX)
        return NULL;
    vfft_proto_dp_entry_t *e = &ctx->entries[ctx->count++];
    memset(e, 0, sizeof(*e));
    e->N = N;
    e->K_eff = K_eff;
    e->n_plans = 0;
    return e;
}

/* Sort comparator for vfft_proto_dp_subplan_t (ascending by cost_ns). */
static int _vfft_proto_dp_subplan_cmp(const void *a, const void *b)
{
    double ca = ((const vfft_proto_dp_subplan_t *)a)->cost_ns;
    double cb = ((const vfft_proto_dp_subplan_t *)b)->cost_ns;
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
 * across VFFT_PROTO_DP_TIME_REPEAT trials. Hard wall-clock cap per call.
 * Per-trial buffer reset (zero-init via copy from orig) keeps the
 * data path consistent across repeats and absorbs denormals.
 *
 * Buffer-size invariant: total = N * K_eff, and N*K_eff is conserved
 * across the recursion (sub_N * sub_K_eff = N_outer * K_outer), so
 * the once-allocated ctx buffers always fit.
 * ===================================================================== */

#ifndef VFFT_PROTO_DP_TIME_REPEAT
#define VFFT_PROTO_DP_TIME_REPEAT 6 /* number of best-of trials */
#endif
#ifndef VFFT_PROTO_DP_TIME_MIN_NS
#define VFFT_PROTO_DP_TIME_MIN_NS 2.0e6 /* min wall-clock per trial (2 ms) */
#endif
#ifndef VFFT_PROTO_DP_TIME_LIMIT_NS
#define VFFT_PROTO_DP_TIME_LIMIT_NS 5.0e8 /* per-bench cap (~0.5 s) */
#endif

/* Intra-cell thermal pacing.
 *
 * Sustained 100% core load during a single cell's search heats the core enough
 * that bench numbers drift up over the run; the package thermal envelope is
 * shared, so even core-pinned runs are affected. Either trigger arms pacing:
 *   - K   >  VFFT_PROTO_DP_PACE_K_THRESHOLD     : deep batches (the original
 *     K=256 minutes-long-search case).
 *   - N*K >= VFFT_PROTO_DP_PACE_TOTAL_THRESHOLD : large per-bench working set, so
 *     LOW-K big-N cells pace too (K=4, N>=8192). A ~700-candidate 16384 K=4
 *     coarse sweep heat-soaks the core enough to drift the coarse ranking;
 *     before this gate K=4 never paced (verified 2026-06-16).
 *
 * Sleep VFFT_PROTO_DP_PACE_MS ms every VFFT_PROTO_DP_PACE_EVERY benches to let the
 * core recover (PACE_EVERY auto-skips searches with < PACE_EVERY benches). ~5%
 * wall overhead at K=256; ~33% at K=4 (faster benches) — the heavier duty cycle
 * is the point: it holds the K=4 coarse pass at a steady clock. */
#ifndef VFFT_PROTO_DP_PACE_K_THRESHOLD
#define VFFT_PROTO_DP_PACE_K_THRESHOLD 64
#endif
#ifndef VFFT_PROTO_DP_PACE_TOTAL_THRESHOLD
#define VFFT_PROTO_DP_PACE_TOTAL_THRESHOLD 32768 /* N*K; arms pacing for big-N low-K (K=4 N>=8192) */
#endif
#ifndef VFFT_PROTO_DP_PACE_EVERY
#define VFFT_PROTO_DP_PACE_EVERY 25
#endif
#ifndef VFFT_PROTO_DP_PACE_MS
#define VFFT_PROTO_DP_PACE_MS 200
#endif

#ifdef _WIN32
/* Forward-declare Sleep without pulling all of windows.h into every
 * TU that includes this header. _WINDOWS_ is windows.h's own guard;
 * if it's defined, Sleep is already declared. */
#ifndef _WINDOWS_
extern __declspec(dllimport) void __stdcall Sleep(unsigned long ms);
#endif
static void _vfft_proto_dp_sleep_ms(int ms)
{
    if (ms > 0)
        Sleep((unsigned long)ms);
}
#else
#include <unistd.h>
static void _vfft_proto_dp_sleep_ms(int ms)
{
    if (ms > 0)
        usleep((useconds_t)ms * 1000);
}
#endif

static void _vfft_proto_dp_maybe_pace(vfft_proto_dp_context_t *ctx, size_t total)
{
    /* Arm if EITHER trigger fires: deep batch (K) or large per-bench work (N*K). */
    if (ctx->K <= (size_t)VFFT_PROTO_DP_PACE_K_THRESHOLD
        && total < (size_t)VFFT_PROTO_DP_PACE_TOTAL_THRESHOLD)
        return;
    if ((ctx->n_benchmarks % VFFT_PROTO_DP_PACE_EVERY) != 0)
        return;
    _vfft_proto_dp_sleep_ms(VFFT_PROTO_DP_PACE_MS);
}

static double _vfft_proto_dp_bench(vfft_proto_dp_context_t *ctx, int N,
                        const int *factors, int nf, size_t K_eff,
                        const vfft_proto_registry_t *reg)
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
    stride_plan_t *plan = vfft_proto_plan_create(N, K_eff, factors, /*variants=*/NULL, nf, reg);
    if (!plan)
        return 1e18;

    /* Warmup (also serves as a single-iter calibration baseline). */
    memcpy(ctx->re, ctx->orig_re, total * sizeof(double));
    memcpy(ctx->im, ctx->orig_im, total * sizeof(double));
    vfft_proto_execute_fwd(plan, ctx->re, ctx->im, K_eff);

    /* Adaptive iter: double `reps` until one trial >= VFFT_PROTO_DP_TIME_MIN_NS,
     * then collect VFFT_PROTO_DP_TIME_REPEAT best-of trials at that rep count.
     * Mirrors FFTW kernel/timer.c::measure_execution_time. */
    double best = 1e30;
    double total_elapsed = 0.0;
    int reps = 1;
    int calibrated = 0;

    for (int outer = 0; outer < 32 && total_elapsed < VFFT_PROTO_DP_TIME_LIMIT_NS; outer++)
    {
        double tmin_trial = 1e30;

        for (int t = 0; t < VFFT_PROTO_DP_TIME_REPEAT; t++)
        {
            memcpy(ctx->re, ctx->orig_re, total * sizeof(double));
            memcpy(ctx->im, ctx->orig_im, total * sizeof(double));
            double t0 = vfft_proto_now_ns();
            for (int i = 0; i < reps; i++)
                vfft_proto_execute_fwd(plan, ctx->re, ctx->im, K_eff);
            double trial_ns = vfft_proto_now_ns() - t0;
            if (trial_ns < tmin_trial)
                tmin_trial = trial_ns;
            total_elapsed += trial_ns;
            if (total_elapsed >= VFFT_PROTO_DP_TIME_LIMIT_NS)
                break;
        }

        if (!calibrated)
        {
            /* If trial duration is too short, double reps and retry. */
            if (tmin_trial < VFFT_PROTO_DP_TIME_MIN_NS)
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

        /* Calibrated and VFFT_PROTO_DP_TIME_REPEAT collected — done. */
        break;
    }

    vfft_proto_plan_destroy(plan);
    ctx->n_benchmarks++;
    _vfft_proto_dp_maybe_pace(ctx, total);
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
 * step (vfft_proto_dp_plan) tries all orderings of the winning set.
 * ===================================================================== */

/* Available radixes for DP decomposition, largest first */
static const int VFFT_PROTO_DP_RADIXES[] = {
    64, 32, 25, 20, 16, 12, 10, 8, 7, 6, 5, 4, 3, 2,
    19, 17, 13, 11, 0};

/* Top-K recursive DP solver. Returns up to max_out best plans for
 * (N, K_eff), sorted by cost ascending. Cached on first call. The
 * cache row stores up to VFFT_PROTO_DP_TOPK_MAX plans; subsequent calls return
 * min(cached_count, max_out) of them.
 *
 * BELIEVE_PCOST behavior: when 0, the top-ranked cached plan's cost is
 * re-measured fresh and the cache row is updated; runners-up are kept
 * with their original (stale) costs. This matches the production
 * intent — the BELIEVE flag affects winner selection variance, not the
 * shape of the runner-up list. */
static int _vfft_proto_dp_solve_topk(vfft_proto_dp_context_t *ctx, int N, size_t K_eff,
                          const vfft_proto_registry_t *reg,
                          vfft_proto_dp_subplan_t *out, int max_out)
{
    if (max_out <= 0)
        return 0;

    /* Cache check (keyed by N + K_eff). */
    vfft_proto_dp_entry_t *cached = _vfft_proto_dp_lookup(ctx, N, K_eff);
    if (cached)
    {
        ctx->n_cache_hits++;
        int n = cached->n_plans < max_out ? cached->n_plans : max_out;
        for (int i = 0; i < n; i++)
            out[i] = cached->plans[i];
        if (!ctx->believe_subplan_cost && n > 0)
        {
            /* PATIENT: re-bench top-1 with the cached factorization. */
            double fresh = _vfft_proto_dp_bench(ctx, N, cached->plans[0].factors,
                                     cached->plans[0].nfactors, K_eff, reg);
            cached->plans[0].cost_ns = fresh;
            out[0].cost_ns = fresh;
        }
        return n;
    }

    /* Accumulator: collect every viable candidate produced by this call.
     * Sized for |VFFT_PROTO_DP_RADIXES| * VFFT_PROTO_DP_TOPK_MAX + slack. */
    enum
    {
        _DP_ACCUM_MAX = 64
    };
    vfft_proto_dp_subplan_t accum[_DP_ACCUM_MAX];
    int n_accum = 0;

    /* Base case: N is itself a registered radix — emits a single
     * one-stage plan. The recursive case below also tries N=R candidates
     * via the M=1 branch, but that requires N to be a registered radix
     * AND a divisor of itself; emitting here covers the 2 ≤ N ≤ 64 base. */
    if ((N > 0 && N < VFFT_PROTO_REG_MAX_RADIX && reg->n1_fwd[N]) && N <= 64)
    {
        int factors[1] = {N};
        double ns = _vfft_proto_dp_bench(ctx, N, factors, 1, K_eff, reg);
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
    for (const int *rp = VFFT_PROTO_DP_RADIXES; *rp && n_accum < _DP_ACCUM_MAX; rp++)
    {
        int R = *rp;
        if (N % R != 0)
            continue;
        if (!(R > 0 && R < VFFT_PROTO_REG_MAX_RADIX && reg->n1_fwd[R]))
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
            double ns = _vfft_proto_dp_bench(ctx, N, factors, 1, K_eff, reg);
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
        vfft_proto_dp_subplan_t sub[VFFT_PROTO_DP_TOPK_MAX];
        int n_sub = _vfft_proto_dp_solve_topk(ctx, M, K_eff_sub, reg, sub, VFFT_PROTO_DP_TOPK_MAX);

        for (int s = 0; s < n_sub && n_accum < _DP_ACCUM_MAX; s++)
        {
            if (sub[s].cost_ns >= 1e17)
                continue;
            if (sub[s].nfactors + 1 > STRIDE_MAX_STAGES)
                continue;

            int candidate[STRIDE_MAX_STAGES];
            candidate[0] = R;
            memcpy(candidate + 1, sub[s].factors,
                   sub[s].nfactors * sizeof(int));
            int nf = sub[s].nfactors + 1;

            double ns = _vfft_proto_dp_bench(ctx, N, candidate, nf, K_eff, reg);
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
        qsort(accum, n_accum, sizeof(vfft_proto_dp_subplan_t), _vfft_proto_dp_subplan_cmp);

    /* Cache top VFFT_PROTO_DP_TOPK_MAX. */
    int n_keep = n_accum < VFFT_PROTO_DP_TOPK_MAX ? n_accum : VFFT_PROTO_DP_TOPK_MAX;
    vfft_proto_dp_entry_t *e = _vfft_proto_dp_insert(ctx, N, K_eff);
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
 * shape. All existing callers (vfft_proto_dp_plan, stride_dp_plan_joint_blocked,
 * etc.) keep working unchanged. */
static double _vfft_proto_dp_solve(vfft_proto_dp_context_t *ctx, int N, size_t K_eff,
                        const vfft_proto_registry_t *reg,
                        int *out_factors, int *out_nf)
{
    vfft_proto_dp_subplan_t top1;
    int n = _vfft_proto_dp_solve_topk(ctx, N, K_eff, reg, &top1, 1);
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

static double vfft_proto_dp_plan(vfft_proto_dp_context_t *ctx, int N,
                             const vfft_proto_registry_t *reg,
                             vfft_proto_factorization_t *best_fact,
                             int verbose)
{
    /* Phase 1: recursive DP to find best radix set.
     * Top-level call uses K_eff = ctx->K (i.e., the user-requested batch). */
    int dp_factors[STRIDE_MAX_STAGES];
    int dp_nf = 0;
    double dp_ns = _vfft_proto_dp_solve(ctx, N, ctx->K, reg, dp_factors, &dp_nf);

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
    vfft_proto_perm_list_t *plist = (vfft_proto_perm_list_t *)malloc(sizeof(*plist));
    vfft_proto_gen_permutations(dp_factors, dp_nf, plist);

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

        double ns = _vfft_proto_dp_bench(ctx, N, plist->perms[pi], dp_nf, ctx->K, reg);
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
    ctx->n_benchmarks += plist ? 0 : 0; /* already counted in _vfft_proto_dp_bench */
    return global_best;
}

#endif /* STRIDE_DP_PLANNER_H */
