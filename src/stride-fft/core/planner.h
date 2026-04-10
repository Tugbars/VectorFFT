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
#include "rader.h"      /* includes bluestein.h (shared SIMD helpers) */
#include "r2c.h"       /* real-to-complex / complex-to-real */

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

/* Add or update wisdom entry */
static void stride_wisdom_add(stride_wisdom_t *wis, int N, size_t K,
                               const int *factors, int nf, double best_ns) {
    /* Update existing */
    for (int i = 0; i < wis->count; i++) {
        if (wis->entries[i].N == N && wis->entries[i].K == K) {
            if (best_ns < wis->entries[i].best_ns) {
                memcpy(wis->entries[i].factors, factors, nf * sizeof(int));
                wis->entries[i].nfactors = nf;
                wis->entries[i].best_ns = best_ns;
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
    }
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
#define WISDOM_VERSION 2

static int stride_wisdom_save(const stride_wisdom_t *wis, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "@version %d\n", WISDOM_VERSION);
    fprintf(f, "# VectorFFT stride wisdom — %d entries\n", wis->count);
    for (int i = 0; i < wis->count; i++) {
        const stride_wisdom_entry_t *e = &wis->entries[i];
        fprintf(f, "%d %zu %d", e->N, e->K, e->nfactors);
        for (int j = 0; j < e->nfactors; j++)
            fprintf(f, " %d", e->factors[j]);
        fprintf(f, " %.2f\n", e->best_ns);
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
        if (sscanf(line + pos, "%lf", &e.best_ns) < 1)
            continue;
        stride_wisdom_add(wis, e.N, e.K, e.factors, e.nfactors, e.best_ns);
    }
    fclose(f);
    return version_ok ? 0 : -1;
}

/* =====================================================================
 * PLAN CONSTRUCTION
 * ===================================================================== */

/**
 * Build a plan from registry for a given factorization.
 * Always uses flat (non-log3) twiddle codelets.
 */
static stride_plan_t *_stride_build_plan(
        int N, size_t K,
        const int *factors, int nf,
        const stride_registry_t *reg) {
    stride_n1_fn n1f[FACT_MAX_STAGES], n1b[FACT_MAX_STAGES];
    stride_t1_fn t1f[FACT_MAX_STAGES], t1b[FACT_MAX_STAGES];

    for (int s = 0; s < nf; s++) {
        int R = factors[s];
        n1f[s] = reg->n1_fwd[R];
        n1b[s] = reg->n1_bwd[R];
        if (s == 0) {
            t1f[s] = NULL;
            t1b[s] = NULL;
        } else {
            t1f[s] = reg->t1_fwd[R];
            t1b[s] = reg->t1_bwd[R];
        }
    }

    stride_plan_t *plan = stride_plan_create(N, K, factors, nf, n1f, n1b, t1f, t1b, 0);
    if (!plan) return NULL;

    /* Attach scalar-broadcast twiddle codelets where available.
     * These use _mm256_broadcast_sd internally — zero twiddle cache pressure.
     * The executor prefers t1s over t1 when both are present. */
    for (int s = 1; s < nf; s++) {
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
    if (e)
        return _stride_build_plan(N, K, e->factors, e->nfactors, reg);
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

    /* Re-bench the winner with full accuracy */
    double refined_ns = _stride_refine_bench(N, K, best_fact.factors,
                                              best_fact.nfactors, reg);

    stride_wisdom_add(wis, N, K, best_fact.factors, best_fact.nfactors,
                      refined_ns);

    /* Save to disk immediately if path provided */
    if (save_path)
        stride_wisdom_save(wis, save_path);

    return refined_ns;
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
