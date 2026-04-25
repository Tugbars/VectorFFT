/**
 * stride_planner.h -- Top-level planner for stride-based FFT executor
 *
 * Ties together: registry + factorizer + plan creation into a single API.
 *
 * Usage:
 *   stride_registry_t reg;
 *   stride_registry_init(&reg);
 *
 *   // Heuristic (fast, within 1.0-1.3x of optimal for most cases):
 *   stride_plan_t *plan = stride_auto_plan(N, K, &reg);
 *
 *   // Exhaustive (slow, tries all factorizations x orderings):
 *   stride_plan_t *plan = stride_exhaustive_plan(N, K, &reg);
 *
 *   // Execute:
 *   stride_execute_fwd(plan, re, im);
 *   stride_execute_bwd(plan, re, im);
 *   stride_plan_destroy(plan);
 *
 * Wisdom (optional):
 *   stride_wisdom_t wis;
 *   stride_wisdom_init(&wis);
 *   stride_wisdom_load(&wis, "stride_wisdom.txt");
 *
 *   // Uses wisdom if available, falls back to heuristic:
 *   stride_plan_t *plan = stride_wise_plan(N, K, &reg, &wis);
 *
 *   // Populate wisdom via exhaustive search:
 *   stride_wisdom_calibrate(&wis, N, K, &reg);
 *   stride_wisdom_save(&wis, "stride_wisdom.txt");
 */
#ifndef STRIDE_PLANNER_H
#define STRIDE_PLANNER_H

#include "registry.h"
#include "factorizer.h"
#include "exhaustive.h"
#include "dp_planner.h"
#include "executor_blocked.h"
#include "rader.h"      /* includes bluestein.h (shared SIMD helpers) */
#include "r2c.h"       /* real-to-complex / complex-to-real */

/* Blocked executor heuristic threshold: K <= this triggers blocking check */
#ifndef STRIDE_BLOCKED_K_THRESHOLD
#define STRIDE_BLOCKED_K_THRESHOLD 8
#endif

/* =====================================================================
 * BLOCKED DISPATCH — wraps stride_execute_fwd/bwd with blocked path
 *
 * When plan->use_blocked is set (by wisdom), dispatches to the blocked
 * executor. Otherwise falls through to the standard executor.
 * These are defined here (after both executor.h and executor_blocked.h)
 * to avoid circular includes.
 * ===================================================================== */

static inline void stride_execute_fwd_auto(const stride_plan_t *plan,
                                            double *re, double *im) {
    if (plan->use_blocked) {
        _stride_execute_fwd_blocked(plan, re, im,
                                     plan->split_stage, plan->block_groups);
        return;
    }
    stride_execute_fwd(plan, re, im);
}

static inline void stride_execute_bwd_auto(const stride_plan_t *plan,
                                            double *re, double *im) {
    if (plan->use_blocked) {
        _stride_execute_bwd_blocked(plan, re, im,
                                     plan->split_stage, plan->block_groups);
        return;
    }
    stride_execute_bwd(plan, re, im);
}

/* =====================================================================
 * WISDOM -- cached exhaustive search results
 * ===================================================================== */

#define WISDOM_MAX_ENTRIES 256

typedef struct {
    int N;
    size_t K;
    int factors[FACT_MAX_STAGES];
    int nfactors;
    double best_ns;   /* best time found */

    /* Blocked executor selection (version 3+).
     * Determined by joint calibration over (factorization × executor × split).
     * use_blocked=0 means standard sweep executor (default, backward compat). */
    int use_blocked;     /* 0 = standard, 1 = blocked */
    int split_stage;     /* first blocked stage */
    int block_groups;    /* groups per block at split stage */
} stride_wisdom_entry_t;

typedef struct {
    stride_wisdom_entry_t entries[WISDOM_MAX_ENTRIES];
    int count;
} stride_wisdom_t;

static void stride_wisdom_init(stride_wisdom_t *wis) {
    wis->count = 0;
}

/* Find wisdom entry for (N, K). Returns NULL if not found. */
static const stride_wisdom_entry_t *stride_wisdom_lookup(
        const stride_wisdom_t *wis, int N, size_t K) {
    for (int i = 0; i < wis->count; i++) {
        if (wis->entries[i].N == N && wis->entries[i].K == K)
            return &wis->entries[i];
    }
    return NULL;
}

/* Add or update wisdom entry (full version with blocked fields) */
static void stride_wisdom_add_full(stride_wisdom_t *wis, int N, size_t K,
                                    const int *factors, int nf, double best_ns,
                                    int use_blocked, int split_stage,
                                    int block_groups) {
    /* Update existing */
    for (int i = 0; i < wis->count; i++) {
        if (wis->entries[i].N == N && wis->entries[i].K == K) {
            if (best_ns < wis->entries[i].best_ns) {
                memcpy(wis->entries[i].factors, factors, nf * sizeof(int));
                wis->entries[i].nfactors = nf;
                wis->entries[i].best_ns = best_ns;
                wis->entries[i].use_blocked = use_blocked;
                wis->entries[i].split_stage = split_stage;
                wis->entries[i].block_groups = block_groups;
            }
            return;
        }
    }
    /* Insert new */
    if (wis->count < WISDOM_MAX_ENTRIES) {
        stride_wisdom_entry_t *e = &wis->entries[wis->count++];
        e->N = N;
        e->K = K;
        memcpy(e->factors, factors, nf * sizeof(int));
        e->nfactors = nf;
        e->best_ns = best_ns;
        e->use_blocked = use_blocked;
        e->split_stage = split_stage;
        e->block_groups = block_groups;
    }
}

/* Legacy wrapper (standard executor, no blocking) */
static void stride_wisdom_add(stride_wisdom_t *wis, int N, size_t K,
                               const int *factors, int nf, double best_ns) {
    stride_wisdom_add_full(wis, N, K, factors, nf, best_ns, 0, 0, 0);
}

/* -- Wisdom file I/O --
 * Format:
 *   Line 1: @version N  (format version, reject if mismatch)
 *   Remaining: N K nf f0 f1 ... best_ns
 *   Lines starting with # are comments.
 *
 * Example:
 *   @version 2
 *   1000 256 4 8 5 5 5 14.20
 *
 * Bump WISDOM_VERSION when the format changes — old files will be
 * silently rejected and re-calibrated on next run. */
#define WISDOM_VERSION 3

static int stride_wisdom_save(const stride_wisdom_t *wis, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "@version %d\n", WISDOM_VERSION);
    fprintf(f, "# VectorFFT stride wisdom — %d entries\n", wis->count);
    fprintf(f, "# N K nf factors... best_ns use_blocked split_stage block_groups\n");
    for (int i = 0; i < wis->count; i++) {
        const stride_wisdom_entry_t *e = &wis->entries[i];
        fprintf(f, "%d %zu %d", e->N, e->K, e->nfactors);
        for (int j = 0; j < e->nfactors; j++)
            fprintf(f, " %d", e->factors[j]);
        fprintf(f, " %.2f %d %d %d\n", e->best_ns,
                e->use_blocked, e->split_stage, e->block_groups);
    }
    fclose(f);
    return 0;
}

static int stride_wisdom_load(stride_wisdom_t *wis, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[256];
    int version_ok = 0;

    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;

        /* Version check — must be first non-comment line */
        if (line[0] == '@') {
            int ver = 0;
            if (sscanf(line, "@version %d", &ver) == 1 && ver == WISDOM_VERSION)
                version_ok = 1;
            continue;
        }

        /* Reject entries if version doesn't match (or missing) */
        if (!version_ok) {
            fclose(f);
            return -1;  /* stale wisdom — caller will re-calibrate */
        }

        stride_wisdom_entry_t e;
        memset(&e, 0, sizeof(e));
        int pos = 0, n;
        if (sscanf(line, "%d %zu %d%n", &e.N, &e.K, &e.nfactors, &n) < 3)
            continue;
        pos = n;
        if (e.nfactors < 1 || e.nfactors > FACT_MAX_STAGES) continue;
        int ok = 1;
        for (int j = 0; j < e.nfactors; j++) {
            if (sscanf(line + pos, "%d%n", &e.factors[j], &n) < 1) { ok = 0; break; }
            pos += n;
        }
        if (!ok) continue;
        if (sscanf(line + pos, "%lf%n", &e.best_ns, &n) < 1)
            continue;
        pos += n;
        /* Version 3: blocked executor fields (optional, default 0) */
        sscanf(line + pos, "%d %d %d",
               &e.use_blocked, &e.split_stage, &e.block_groups);
        stride_wisdom_add_full(wis, e.N, e.K, e.factors, e.nfactors, e.best_ns,
                               e.use_blocked, e.split_stage, e.block_groups);
    }
    fclose(f);
    return version_ok ? 0 : -1;
}

/* =====================================================================
 * PLAN CONSTRUCTION
 * =====================================================================
 *
 * Phase 2 codelet-wisdom integration (Apr 2026)
 * ---------------------------------------------
 * At plan time, consult the codelet-side plan_wisdom headers to decide
 * whether each stage should use the DIT-log3 codelet or the flat
 * (DIT) codelet. Wisdom is indexed by (me, ios):
 *   me  = per-thread slice size (K if single-threaded or group-parallel;
 *         K / num_threads under K-split multithreading)
 *   ios = stage stride = K * prod(factors[s+1..nf-1])
 *
 * When wisdom says DIT-log3 wins:
 *   - Swap stage->t1_fwd to reg->t1_fwd_log3[R]
 *   - Set log3_mask bit s, so stride_plan_create sets stage->use_log3
 *   - The executor's log3 path applies cf to all R legs and calls
 *     t1_fwd with raw per-leg twiddles.
 *   - The executor's R≥64 n1_fallback override is bypassed for log3
 *     stages (see executor.h line 1040).
 *
 * Why DIT-log3 specifically (not DIF-log3)?
 *   The forward executor path is DIT-structured. Phase 2 only activates
 *   protocols the executor can run today. DIF-log3 wins additional
 *   cells (5/13 at R=64) but requires a Phase 3 executor refactor —
 *   DIT-log3 and DIF-log3 are NOT swappable codelets despite sharing
 *   the same call signature; bench-time cross-validation confirms they
 *   produce different output buffers.
 *
 * Missing wisdom (radix has no prefer_dit_log3, or it returns 0 at
 * this (me, ios)) falls back to the legacy flat + t1s behaviour —
 * same as pre-wisdom planner.
 */

#include "wisdom_bridge.h"  /* stride_prefer_dit_log3() — DIT-only safe query */
#include "threads.h"        /* stride_get_num_threads() */

/* Compute the expected per-thread slice size at plan time.
 *
 * Under K-split multithreading (the default path for K >= some threshold
 * — see executor's _stride_slice_trampoline), each of T threads processes
 * a slice of size K/T. Under group-parallel execution (small-K path), each
 * thread processes the full K.
 *
 * At plan time we don't know which path the executor will pick, but the
 * wisdom is robust to this — wisdom predicates are typically me-monotone
 * within a few me values, and the K/T estimate errs toward smaller me,
 * which is the regime with more variant-choice sensitivity. Single-
 * threaded runs collapse to K. */
static inline size_t _stride_me_plan(size_t K) {
    int T = stride_get_num_threads();
    if (T <= 1) return K;
    /* K-split slice, rounded up so the planner never under-estimates. */
    return (K + (size_t)T - 1) / (size_t)T;
}

/* Compute the ios (stride between butterfly legs) at stage s.
 *
 * Matches the executor's dim_stride[s] from plan_compute_groups:
 *   ios[s] = K * prod(factors[s+1 .. nf-1])
 */
static inline size_t _stride_ios_at_stage(size_t K, const int *factors,
                                           int nf, int s) {
    size_t ios = K;
    for (int d = s + 1; d < nf; d++)
        ios *= (size_t)factors[d];
    return ios;
}

/**
 * Build a plan from registry for a given factorization.
 *
 * Consults codelet-side plan_wisdom at each stage to choose between
 * flat (default), t1s (if present), and DIT-log3 (if registered and
 * wisdom-preferred for this me/ios).
 */
static stride_plan_t *_stride_build_plan(
        int N, size_t K,
        const int *factors, int nf,
        const stride_registry_t *reg) {
    stride_n1_fn n1f[FACT_MAX_STAGES], n1b[FACT_MAX_STAGES];
    stride_t1_fn t1f[FACT_MAX_STAGES], t1b[FACT_MAX_STAGES];

    /* Per-stage protocol decisions. stage_uses_log3 doubles as the
     * log3_mask bits passed to stride_plan_create. */
    int stage_uses_log3[FACT_MAX_STAGES] = {0};
    int stage_skip_t1s [FACT_MAX_STAGES] = {0};

    size_t me_plan = _stride_me_plan(K);

    for (int s = 0; s < nf; s++) {
        int R = factors[s];
        n1f[s] = reg->n1_fwd[R];
        n1b[s] = reg->n1_bwd[R];
        if (s == 0) {
            /* Stage 0 has no twiddles (the outermost butterfly). */
            t1f[s] = NULL;
            t1b[s] = NULL;
            continue;
        }

        /* Consult wisdom: does DIT-log3 specifically win at this stage?
         *
         * We use prefer_DIT_log3 (conservative) rather than the generic
         * prefer_log3 (union):
         *   1. The forward executor path is DIT-structured. It can run
         *      the DIT-log3 codelet directly (via reg->t1_fwd_log3[R]).
         *      It cannot run DIF-log3 — Phase 3 territory.
         *   2. At some (me, ios) cells at R=64, DIT-log3 is *slower*
         *      than flat even though DIF-log3 wins. Using the union
         *      query would activate log3 there and regress vs flat.
         *      The conservative query only returns 1 where DIT-log3
         *      was the specific bench winner — safe to activate.
         *
         * Plus: we have a log3 codelet registered for this R. */
        size_t ios_s = _stride_ios_at_stage(K, factors, nf, s);
        int want_log3 =
            stride_prefer_dit_log3(R, me_plan, ios_s) &&
            (R < STRIDE_REG_MAX_RADIX) &&
            (reg->t1_fwd_log3[R] != NULL);

        if (want_log3) {
            t1f[s] = reg->t1_fwd_log3[R];
            t1b[s] = reg->t1_bwd_log3[R];
            stage_uses_log3[s] = 1;
            /* Prevent the t1s post-pass from shadowing the log3 path.
             * (Executor's runtime branch order would prefer log3 anyway,
             * but clearing t1s_fwd keeps stage state consistent with
             * the plan-time decision.) */
            stage_skip_t1s[s] = 1;
        } else {
            t1f[s] = reg->t1_fwd[R];
            t1b[s] = reg->t1_bwd[R];
        }
    }

    /* Assemble log3_mask from per-stage flags. */
    int log3_mask = 0;
    for (int s = 0; s < nf; s++)
        if (stage_uses_log3[s]) log3_mask |= (1 << s);

    stride_plan_t *plan = stride_plan_create(N, K, factors, nf,
                                              n1f, n1b, t1f, t1b,
                                              log3_mask);
    if (!plan) return NULL;

    /* Attach scalar-broadcast twiddle codelets where available and
     * wisdom hasn't claimed the stage for log3. The executor prefers
     * t1s over t1 when t1s_fwd is set (see executor.h ~line 218). */
    for (int s = 1; s < nf; s++) {
        if (stage_skip_t1s[s]) continue;
        int R = factors[s];
        if (R < STRIDE_REG_MAX_RADIX && reg->t1s_fwd[R]) {
            plan->stages[s].t1s_fwd = reg->t1s_fwd[R];
            plan->stages[s].t1s_bwd = reg->t1s_bwd[R];
        }
    }

    /* Attach out-of-place twiddle codelets for strided first-stage use.
     * Used by R2C fused pack and 2D FFT strided executor. */
    for (int s = 0; s < nf; s++) {
        int R = factors[s];
        if (R < STRIDE_REG_MAX_RADIX && reg->t1_oop_fwd[R]) {
            plan->stages[s].t1_oop_fwd = reg->t1_oop_fwd[R];
            plan->stages[s].t1_oop_bwd = reg->t1_oop_bwd[R];
        }
    }

    /* Attach scaled n1 codelets for C2R fused unpack.
     * Used by backward R2C: last stage writes ×2 scaled output at stride 2K. */
    for (int s = 0; s < nf; s++) {
        int R = factors[s];
        if (R < STRIDE_REG_MAX_RADIX && reg->n1_scaled_fwd[R]) {
            plan->stages[s].n1_scaled_fwd = reg->n1_scaled_fwd[R];
            plan->stages[s].n1_scaled_bwd = reg->n1_scaled_bwd[R];
        }
    }

    return plan;
}

/* =====================================================================
 * PUBLIC API
 * ===================================================================== */

/* -- Helpers for prime-size dispatch -- */

static int _stride_is_prime(int n) {
    if (n < 2) return 0;
    if (n < 4) return 1;
    if (n % 2 == 0 || n % 3 == 0) return 0;
    for (int i = 5; (long long)i * i <= n; i += 6)
        if (n % i == 0 || n % (i + 2) == 0) return 0;
    return 1;
}

/* N-1 factors entirely into primes covered by our radix set */
static int _stride_is_rader_friendly(int n) {
    int m = n - 1;
    static const int primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 0};
    for (const int *p = primes; *p; p++)
        while (m % *p == 0) m /= *p;
    return m == 1;
}

/**
 * stride_auto_plan -- Heuristic planner (fast).
 *
 * 1. Direct factorization into available radixes     -> staged plan
 * 2. Prime N, non-smooth N-1 (p-1 has factors > 19) -> Bluestein
 * 3. Prime N, smooth N-1 (p-1 is 19-smooth)         -> Rader
 * 4. Composite with unfactorable prime factor        -> NULL (TODO)
 */
static stride_plan_t *stride_auto_plan(int N, size_t K,
                                        const stride_registry_t *reg) {
    /* 1. Direct factorization */
    stride_factorization_t fact;
    if (stride_factorize(N, K, reg, &fact) == 0)
        return _stride_build_plan(N, K, fact.factors, fact.nfactors, reg);

    /* 2. Prime with non-smooth N-1: Bluestein */
    if (_stride_is_prime(N) && !_stride_is_rader_friendly(N)) {
        int M = _bluestein_choose_m(N);
        size_t B = _bluestein_block_size(M, K);
        stride_plan_t *inner = stride_auto_plan(M, B, reg);
        if (inner) return stride_bluestein_plan(N, K, B, inner, M);
    }

    /* 3. Prime with smooth N-1: Rader (convolution of size N-1) */
    if (_stride_is_prime(N) && _stride_is_rader_friendly(N)) {
        int nm1 = N - 1;
        size_t B = _bluestein_block_size(nm1, K);
        stride_plan_t *inner = stride_auto_plan(nm1, B, reg);
        if (inner) return stride_rader_plan(N, K, B, inner);
    }

    /* 4. Composite with unfactorable prime factor: reserved (TODO) */
    return NULL;
}

/**
 * stride_exhaustive_plan -- Exhaustive search planner (slow).
 *
 * Tries all factorizations x orderings, benchmarks each,
 * returns a plan built from the best one found.
 */
static stride_plan_t *stride_exhaustive_plan(int N, size_t K,
                                              const stride_registry_t *reg) {
    stride_factorization_t best_fact;
    double best_ns = stride_exhaustive_search(N, K, reg, &best_fact, 0);
    if (best_ns >= 1e17)
        return NULL;
    return _stride_build_plan(N, K, best_fact.factors, best_fact.nfactors, reg);
}

/**
 * stride_wise_plan -- Wisdom-aware planner.
 *
 * Checks wisdom for a cached result. If found, builds plan from it.
 * Otherwise falls back to heuristic (stride_auto_plan).
 */
static stride_plan_t *stride_wise_plan(int N, size_t K,
                                        const stride_registry_t *reg,
                                        const stride_wisdom_t *wis) {
    const stride_wisdom_entry_t *e = stride_wisdom_lookup(wis, N, K);
    if (e) {
        stride_plan_t *plan = _stride_build_plan(N, K, e->factors, e->nfactors, reg);
        if (plan) {
            plan->use_blocked = e->use_blocked;
            plan->split_stage = e->split_stage;
            plan->block_groups = e->block_groups;
        }
        return plan;
    }
    return stride_auto_plan(N, K, reg);
}

/**
 * stride_r2c_auto_plan -- Create a real-to-complex FFT plan.
 *
 * N must be even. Returns a plan where:
 *   stride_execute_fwd(plan, re, im):  re has N*K reals in, (N/2+1)*K re out, im has (N/2+1)*K im out
 *   stride_execute_bwd(plan, re, im):  reverse (C2R), unnormalized (divide output by N)
 *
 * Or use the explicit API:
 *   stride_execute_r2c(plan, real_in, out_re, out_im)
 *   stride_execute_c2r(plan, in_re, in_im, real_out)
 */
static stride_plan_t *stride_r2c_auto_plan(int N, size_t K,
                                            const stride_registry_t *reg) {
    if (N < 2 || (N & 1)) return NULL;
    int halfN = N / 2;
    size_t B = _bluestein_block_size(halfN, K);
    stride_plan_t *inner = stride_auto_plan(halfN, B, reg);
    if (!inner) return NULL;
    return stride_r2c_plan(N, K, B, inner);
}

/**
 * _stride_refine_bench -- Re-bench a factorization with high accuracy.
 *
 * The exhaustive search uses reduced reps for speed, producing noisy timings.
 * This function re-benchmarks the winner with proper warmup, reps, and trials
 * to get an accurate timing for the wisdom file.
 */
static double _stride_refine_bench(int N, size_t K,
                                    const int *factors, int nf,
                                    const stride_registry_t *reg) {
    stride_plan_t *plan = _stride_build_plan(N, K, factors, nf, reg);
    if (!plan) return 1e18;

    size_t total = (size_t)N * K;
    double *re = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    for (size_t i = 0; i < total; i++) {
        re[i] = (double)rand()/RAND_MAX - 0.5;
        im[i] = (double)rand()/RAND_MAX - 0.5;
    }

    /* Warmup */
    for (int i = 0; i < 10; i++)
        stride_execute_fwd(plan, re, im);

    int reps = (int)(1e6 / (total + 1));
    if (reps < 20) reps = 20;
    if (reps > 100000) reps = 100000;

    /* Best of 5 trials */
    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++)
            stride_execute_fwd(plan, re, im);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }

    STRIDE_ALIGNED_FREE(re);
    STRIDE_ALIGNED_FREE(im);
    stride_plan_destroy(plan);
    return best;
}

/**
 * stride_wisdom_calibrate -- Find best factorization and store in wisdom.
 *
 * Two strategies based on exhaustive_threshold:
 *   N <= threshold: exhaustive search (all factorizations x orderings)
 *   N >  threshold: recursive DP planner (FFTW-style with memoization)
 *
 * Parameters:
 *   wis:         wisdom database to update
 *   N, K:        transform size and batch count
 *   reg:         codelet registry
 *   dp_ctx:      optional shared DP context (NULL = create internally)
 *   force:       0 = skip if (N,K) already in wisdom, 1 = always recalibrate
 *   verbose:     0 = quiet, 1 = print decompositions + winner
 *   exhaustive_threshold: N <= this uses exhaustive, N > uses DP
 *   save_path:   if non-NULL, save wisdom to this file after each new entry
 *                (crash protection for long calibration runs)
 *
 * Returns: best time in ns, or 1e18 on failure.
 */
static double stride_wisdom_calibrate_full(
        stride_wisdom_t *wis, int N, size_t K,
        const stride_registry_t *reg,
        stride_dp_context_t *dp_ctx,
        int force, int verbose, int exhaustive_threshold,
        const char *save_path)
{
    /* Skip if already calibrated (unless force) */
    if (!force) {
        const stride_wisdom_entry_t *e = stride_wisdom_lookup(wis, N, K);
        if (e) return e->best_ns;
    }

    stride_factorization_t best_fact;
    double best_ns;

    if (N <= exhaustive_threshold) {
        best_ns = stride_exhaustive_search(N, K, reg, &best_fact, verbose);
    } else {
        stride_dp_context_t local_ctx;
        int own_ctx = 0;
        if (!dp_ctx) {
            stride_dp_init(&local_ctx, K, N);
            dp_ctx = &local_ctx;
            own_ctx = 1;
        }
        best_ns = stride_dp_plan(dp_ctx, N, reg, &best_fact, verbose);
        if (own_ctx) stride_dp_destroy(&local_ctx);
    }

    if (best_ns >= 1e17) return best_ns;

    /* Re-bench the standard winner with full accuracy */
    double refined_ns = _stride_refine_bench(N, K, best_fact.factors,
                                              best_fact.nfactors, reg);

    int win_blocked = 0, win_split = 0, win_bg = 0;

    /* ── Joint blocked search for small K where DTLB matters ──
     * At K<=8 and N>512, try all factorizations with the blocked executor
     * at each valid split point. The standard search above found the best
     * factorization for the standard executor; the blocked search may find
     * a different factorization + split that's faster overall.
     *
     * We re-enumerate factorizations and test each with both executors.
     * Cost: ~2x the standard exhaustive search (acceptable for small K). */
    if (K <= STRIDE_BLOCKED_K_THRESHOLD && N > 512 && N <= exhaustive_threshold) {
        if (verbose)
            printf("  Joint blocked search (K<=%d, N>512)...\n",
                   STRIDE_BLOCKED_K_THRESHOLD);

        factorization_list_t *flist = (factorization_list_t *)malloc(sizeof(*flist));
        stride_enumerate_factorizations(N, reg, flist);

        size_t total = (size_t)N * K;
        double *jre      = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
        double *jim      = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
        double *jorig_re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
        double *jorig_im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
        for (size_t i = 0; i < total; i++) {
            jorig_re[i] = (double)rand() / RAND_MAX - 0.5;
            jorig_im[i] = (double)rand() / RAND_MAX - 0.5;
        }

        int jreps = (int)(5e5 / (total + 1));
        if (jreps < 10) jreps = 10;
        if (jreps > 50000) jreps = 50000;

        for (int fi = 0; fi < flist->count; fi++) {
            permutation_list_t *plist = (permutation_list_t *)malloc(sizeof(*plist));
            stride_gen_permutations(flist->results[fi].factors,
                                    flist->results[fi].nfactors, plist);

            for (int pi = 0; pi < plist->count; pi++) {
                const int *fac = plist->perms[pi];
                int nf = flist->results[fi].nfactors;
                int prod = 1;
                for (int s = 0; s < nf; s++) prod *= fac[s];
                if (prod != N) continue;

                stride_plan_t *jp = _stride_build_plan(N, K, fac, nf, reg);
                if (!jp) continue;

                /* Try blocked at each valid split point */
                for (int sp = 0; sp < jp->num_stages; sp++) {
                    size_t ws = (size_t)jp->stages[sp].radix *
                                jp->stages[sp].stride * K * 2 * sizeof(double);
                    if (ws > STRIDE_BLOCKED_L1_BYTES) continue;

                    int bg = _stride_compute_block_groups(jp, sp);

                    /* Quick bench */
                    for (int w = 0; w < 3; w++)
                        _stride_execute_fwd_blocked(jp, jorig_re, jorig_im, sp, bg);

                    double jbest = 1e18;
                    for (int t = 0; t < 3; t++) {
                        memcpy(jre, jorig_re, total * sizeof(double));
                        memcpy(jim, jorig_im, total * sizeof(double));
                        double t0 = now_ns();
                        for (int r = 0; r < jreps; r++)
                            _stride_execute_fwd_blocked(jp, jre, jim, sp, bg);
                        double ns = (now_ns() - t0) / jreps;
                        if (ns < jbest) jbest = ns;
                    }

                    if (jbest < refined_ns) {
                        refined_ns = jbest;
                        best_fact.nfactors = nf;
                        memcpy(best_fact.factors, fac, nf * sizeof(int));
                        win_blocked = 1;
                        win_split = sp;
                        win_bg = bg;
                    }
                }
                stride_plan_destroy(jp);
            }
            free(plist);
        }

        STRIDE_ALIGNED_FREE(jre); STRIDE_ALIGNED_FREE(jim);
        STRIDE_ALIGNED_FREE(jorig_re); STRIDE_ALIGNED_FREE(jorig_im);
        free(flist);

        if (verbose && win_blocked) {
            printf("  Blocked winner: ");
            for (int s = 0; s < best_fact.nfactors; s++)
                printf("%s%d", s ? "x" : "", best_fact.factors[s]);
            printf(" blocked split@%d bg=%d = %.0f ns\n",
                   win_split, win_bg, refined_ns);
        }
    }

    stride_wisdom_add_full(wis, N, K, best_fact.factors, best_fact.nfactors,
                           refined_ns, win_blocked, win_split, win_bg);

    /* Save to disk immediately if path provided */
    if (save_path)
        stride_wisdom_save(wis, save_path);

    return refined_ns;
}

/**
 * stride_dp_plan_joint_blocked -- DP-style joint factorization + blocked search.
 *
 * Designed for large N (where exhaustive enumeration of all factorizations is
 * too slow). Strategy:
 *
 *   1. Run standard DP planner to find the best factorization for the standard
 *      executor (memoized across sub-problems, fast).
 *   2. Take the DP-winning factorization. Try all permutations of its factor
 *      set with both the standard executor AND the blocked executor at each
 *      valid split point.
 *   3. Return the global winner (factorization + use_blocked + split + bg).
 *
 * Tradeoff vs exhaustive joint search: enumerates permutations of ONE
 * factorization (the DP winner), not all factorizations. At large N (mostly
 * pow2), the radix choices are constrained, so the DP-winner is typically the
 * right base. Misses cases where a different factorization is uniquely
 * blocked-friendly — acceptable for the close-call regime.
 */
static double stride_dp_plan_joint_blocked(
        stride_dp_context_t *ctx, int N, size_t K,
        const stride_registry_t *reg,
        stride_factorization_t *best_fact,
        int *out_use_blocked, int *out_split, int *out_bg,
        int verbose)
{
    *out_use_blocked = 0;
    *out_split = 0;
    *out_bg = 0;

    /* Phase 1: standard DP factorization (memoized) */
    stride_factorization_t dp_fact;
    double dp_ns = stride_dp_plan(ctx, N, reg, &dp_fact, verbose);
    if (dp_ns >= 1e17) return dp_ns;

    /* Baseline: standard winner */
    double best_ns = dp_ns;
    *best_fact = dp_fact;

    /* Phase 2: enumerate permutations × split points × executor variants */
    permutation_list_t *plist = (permutation_list_t *)malloc(sizeof(*plist));
    stride_gen_permutations(dp_fact.factors, dp_fact.nfactors, plist);

    size_t total = (size_t)N * K;
    double *jre      = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *jim      = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *jorig_re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *jorig_im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    for (size_t i = 0; i < total; i++) {
        jorig_re[i] = (double)rand() / RAND_MAX - 0.5;
        jorig_im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    int jreps = (int)(5e5 / (total + 1));
    if (jreps < 5) jreps = 5;
    if (jreps > 50000) jreps = 50000;

    int variants_tried = 0;
    int blocked_winners = 0;

    for (int pi = 0; pi < plist->count; pi++) {
        const int *fac = plist->perms[pi];
        int nf = dp_fact.nfactors;

        stride_plan_t *jp = _stride_build_plan(N, K, fac, nf, reg);
        if (!jp) continue;

        /* Bench standard executor for this permutation */
        for (int w = 0; w < 3; w++)
            _stride_execute_fwd_slice(jp, jorig_re, jorig_im, K, K);

        double std_best = 1e18;
        for (int t = 0; t < 3; t++) {
            memcpy(jre, jorig_re, total * sizeof(double));
            memcpy(jim, jorig_im, total * sizeof(double));
            double t0 = now_ns();
            for (int r = 0; r < jreps; r++)
                _stride_execute_fwd_slice(jp, jre, jim, K, K);
            double ns = (now_ns() - t0) / jreps;
            if (ns < std_best) std_best = ns;
        }
        variants_tried++;

        if (std_best < best_ns) {
            best_ns = std_best;
            best_fact->nfactors = nf;
            memcpy(best_fact->factors, fac, nf * sizeof(int));
            *out_use_blocked = 0;
            *out_split = 0;
            *out_bg = 0;
        }

        /* Bench blocked executor at each valid split point */
        for (int sp = 0; sp < jp->num_stages; sp++) {
            size_t ws = (size_t)jp->stages[sp].radix *
                        jp->stages[sp].stride * K * 2 * sizeof(double);
            if (ws > STRIDE_BLOCKED_L1_BYTES) continue;

            int bg = _stride_compute_block_groups(jp, sp);

            for (int w = 0; w < 3; w++)
                _stride_execute_fwd_blocked(jp, jorig_re, jorig_im, sp, bg);

            double jbest = 1e18;
            for (int t = 0; t < 3; t++) {
                memcpy(jre, jorig_re, total * sizeof(double));
                memcpy(jim, jorig_im, total * sizeof(double));
                double t0 = now_ns();
                for (int r = 0; r < jreps; r++)
                    _stride_execute_fwd_blocked(jp, jre, jim, sp, bg);
                double ns = (now_ns() - t0) / jreps;
                if (ns < jbest) jbest = ns;
            }
            variants_tried++;

            if (jbest < best_ns) {
                best_ns = jbest;
                best_fact->nfactors = nf;
                memcpy(best_fact->factors, fac, nf * sizeof(int));
                *out_use_blocked = 1;
                *out_split = sp;
                *out_bg = bg;
                blocked_winners++;
            }
        }

        stride_plan_destroy(jp);
    }

    STRIDE_ALIGNED_FREE(jre);
    STRIDE_ALIGNED_FREE(jim);
    STRIDE_ALIGNED_FREE(jorig_re);
    STRIDE_ALIGNED_FREE(jorig_im);
    free(plist);

    if (verbose) {
        printf("  Joint DP-blocked search: %d variants tried, blocked %s\n",
               variants_tried,
               *out_use_blocked ? "WON" : "lost");
        if (*out_use_blocked) {
            printf("    Winner: ");
            for (int s = 0; s < best_fact->nfactors; s++)
                printf("%s%d", s ? "x" : "", best_fact->factors[s]);
            printf(" blocked split@%d bg=%d = %.0f ns (vs DP std %.0f ns)\n",
                   *out_split, *out_bg, best_ns, dp_ns);
        }
    }

    return best_ns;
}

/**
 * stride_wisdom_recalibrate_with_blocked -- Opt-in joint blocked recalibration.
 *
 * Forces a joint (factorization × blocked executor) search for a single
 * (N, K) entry, regardless of the gates in stride_wisdom_calibrate_full.
 * Updates the wisdom entry IN-MEMORY ONLY (no file I/O).
 *
 * Strategy:
 *   - Small N (<= STRIDE_EXHAUSTIVE_THRESHOLD): existing exhaustive joint
 *     search via stride_wisdom_calibrate_full (its gate fires naturally).
 *   - Large N: stride_dp_plan_joint_blocked (DP factorization + permutation
 *     × split-point joint blocked search).
 *
 * Use this for the close-call cases where the standard calibration's blocked
 * gate (N <= 2048) excludes the case from joint blocked consideration.
 *
 * Notes:
 *   - The wisdom entry is added via stride_wisdom_add_full. If an entry for
 *     (N, K) already exists with a better best_ns, it will NOT be overwritten.
 *     For experiments, start with a fresh in-memory wisdom.
 *   - No save_path — caller is responsible for any persistence decisions.
 */
static double stride_wisdom_recalibrate_with_blocked(
        stride_wisdom_t *wis, int N, size_t K,
        const stride_registry_t *reg,
        stride_dp_context_t *dp_ctx,
        int verbose)
{
    /* Outside blocked-relevant range — fall back to standard calibration */
    if (K > STRIDE_BLOCKED_K_THRESHOLD || N <= 512) {
        return stride_wisdom_calibrate_full(wis, N, K, reg, dp_ctx,
                                             1, verbose, 1024, NULL);
    }

    /* Small N: existing exhaustive joint search applies (its gate fires) */
    if (N <= 1024) {
        return stride_wisdom_calibrate_full(wis, N, K, reg, dp_ctx,
                                             1, verbose, 1024, NULL);
    }

    /* Large N: DP-style joint blocked search */
    stride_factorization_t best_fact;
    int use_blocked, split, bg;
    double best_ns = stride_dp_plan_joint_blocked(
        dp_ctx, N, K, reg, &best_fact,
        &use_blocked, &split, &bg, verbose);

    if (best_ns >= 1e17) return best_ns;

    stride_wisdom_add_full(wis, N, K, best_fact.factors, best_fact.nfactors,
                           best_ns, use_blocked, split, bg);
    return best_ns;
}

#define STRIDE_EXHAUSTIVE_THRESHOLD 1024

/* Legacy wrappers for backward compatibility */
static void stride_wisdom_calibrate_ex(stride_wisdom_t *wis, int N, size_t K,
                                        const stride_registry_t *reg,
                                        stride_dp_context_t *dp_ctx) {
    stride_wisdom_calibrate_full(wis, N, K, reg, dp_ctx,
                                  1, 0, STRIDE_EXHAUSTIVE_THRESHOLD, NULL);
}

static void stride_wisdom_calibrate(stride_wisdom_t *wis, int N, size_t K,
                                     const stride_registry_t *reg) {
    stride_wisdom_calibrate_full(wis, N, K, reg, NULL,
                                  1, 0, STRIDE_EXHAUSTIVE_THRESHOLD, NULL);
}

/* 2D FFT — must be after stride_auto_plan is defined */
#include "fft2d.h"

#endif /* STRIDE_PLANNER_H */
