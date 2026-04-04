/**
 * stride_planner.h — Top-level planner for stride-based FFT executor
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
 *   // Exhaustive (slow, tries all factorizations × orderings × log3 combos):
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
#include "rader.h"      /* includes bluestein.h (shared SIMD helpers) */

/* ═══════════════════════════════════════════════════════════════
 * WISDOM — cached exhaustive search results
 * ═══════════════════════════════════════════════════════════════ */

#define WISDOM_MAX_ENTRIES 256

typedef struct {
    int N;
    size_t K;
    int factors[FACT_MAX_STAGES];
    int nfactors;
    int log3_mask;    /* bitmask: bit s = use log3 for stage s */
    double best_ns;   /* best time found */
} stride_wisdom_entry_t;

typedef struct {
    stride_wisdom_entry_t entries[WISDOM_MAX_ENTRIES];
    int count;
    stride_log3_thresholds_t log3;  /* calibrated per-radix log3 thresholds */
} stride_wisdom_t;

static void stride_wisdom_init(stride_wisdom_t *wis) {
    wis->count = 0;
    stride_log3_thresholds_init(&wis->log3);
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
                               const int *factors, int nf, int log3_mask,
                               double best_ns) {
    /* Update existing */
    for (int i = 0; i < wis->count; i++) {
        if (wis->entries[i].N == N && wis->entries[i].K == K) {
            if (best_ns < wis->entries[i].best_ns) {
                memcpy(wis->entries[i].factors, factors, nf * sizeof(int));
                wis->entries[i].nfactors = nf;
                wis->entries[i].log3_mask = log3_mask;
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
        e->log3_mask = log3_mask;
        e->best_ns = best_ns;
    }
}

/* ── Wisdom file I/O ──
 * Format: one entry per line
 *   N K nf f0 f1 ... log3_mask best_ns
 * Example:
 *   1000 256 4 8 5 5 5 7 14.20
 */
/* ── Wisdom file I/O ──
 * Format:
 *   @log3 R threshold_K         — per-radix log3 threshold (calibrated)
 *   N K nf f0 f1 ... log3_mask best_ns  — per-(N,K) plan
 * Lines starting with # are comments.
 *
 * Example:
 *   @log3 3 512
 *   @log3 7 128
 *   @log3 16 64
 *   1000 256 4 8 5 5 5 6 14.20
 */
static int stride_wisdom_save(const stride_wisdom_t *wis, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "# VectorFFT stride wisdom\n");

    /* Log3 thresholds */
    int has_log3 = 0;
    for (int R = 2; R < STRIDE_REG_MAX_RADIX; R++) {
        if (wis->log3.calibrated[R]) {
            if (!has_log3) { fprintf(f, "# log3 thresholds: @log3 R threshold_K\n"); has_log3 = 1; }
            fprintf(f, "@log3 %d %zu\n", R, wis->log3.threshold_K[R]);
        }
    }

    /* Per-(N,K) plans */
    if (wis->count > 0)
        fprintf(f, "# plans: N K nf factors... log3_mask best_ns\n");
    for (int i = 0; i < wis->count; i++) {
        const stride_wisdom_entry_t *e = &wis->entries[i];
        fprintf(f, "%d %zu %d", e->N, e->K, e->nfactors);
        for (int j = 0; j < e->nfactors; j++)
            fprintf(f, " %d", e->factors[j]);
        fprintf(f, " %d %.2f\n", e->log3_mask, e->best_ns);
    }
    fclose(f);
    return 0;
}

static int stride_wisdom_load(stride_wisdom_t *wis, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;

        /* @log3 R threshold_K */
        if (line[0] == '@') {
            int R; size_t tK;
            if (sscanf(line, "@log3 %d %zu", &R, &tK) == 2) {
                if (R >= 2 && R < STRIDE_REG_MAX_RADIX) {
                    wis->log3.threshold_K[R] = tK;
                    wis->log3.calibrated[R] = 1;
                }
            }
            continue;
        }

        /* N K nf factors... log3_mask best_ns */
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
        if (sscanf(line + pos, "%d %lf", &e.log3_mask, &e.best_ns) < 2)
            continue;
        stride_wisdom_add(wis, e.N, e.K, e.factors, e.nfactors,
                          e.log3_mask, e.best_ns);
    }
    fclose(f);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * LOG3 THRESHOLD CALIBRATION
 *
 * For each radix R with log3 support, sweep K values and find
 * the crossover point where log3 beats flat. Uses a simple
 * inner × R plan (2 stages, stage 1 is twiddled R).
 * ═══════════════════════════════════════════════════════════════ */

/* Pick an inner radix different from R for calibration */
static int _log3_calib_inner(int R, const stride_registry_t *reg) {
    int candidates[] = {8, 16, 5, 3, 7, 4, 6, 2, 0};
    for (int *c = candidates; *c; c++)
        if (*c != R && stride_registry_has(reg, *c)) return *c;
    return 2;
}

static double _log3_calib_bench(int N, size_t K, const int *factors, int nf,
                                 int log3_mask, const stride_registry_t *reg) {
    stride_n1_fn n1f[FACT_MAX_STAGES], n1b[FACT_MAX_STAGES];
    stride_t1_fn t1f[FACT_MAX_STAGES], t1b[FACT_MAX_STAGES];
    for (int s = 0; s < nf; s++) {
        int R = factors[s];
        n1f[s] = reg->n1_fwd[R]; n1b[s] = reg->n1_bwd[R];
        if (s > 0 && ((log3_mask >> s) & 1) && reg->t1_fwd_log3[R]) {
            t1f[s] = reg->t1_fwd_log3[R]; t1b[s] = reg->t1_bwd_log3[R];
        } else {
            t1f[s] = reg->t1_fwd[R]; t1b[s] = reg->t1_bwd[R];
        }
    }
    stride_plan_t *plan = stride_plan_create(N, K, factors, nf,
                                              n1f, n1b, t1f, t1b, log3_mask);
    if (!plan) return 1e18;

    size_t total = (size_t)N * K;
    double *re = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im = (double*)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    for (size_t i = 0; i < total; i++) {
        re[i] = (double)rand()/RAND_MAX - 0.5;
        im[i] = (double)rand()/RAND_MAX - 0.5;
    }

    for (int i = 0; i < 5; i++) stride_execute_fwd(plan, re, im);

    int reps = (int)(5e5 / (total + 1));
    if (reps < 10) reps = 10;
    if (reps > 50000) reps = 50000;

    double best = 1e18;
    for (int t = 0; t < 3; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++) stride_execute_fwd(plan, re, im);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }

    STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);
    stride_plan_destroy(plan);
    return best;
}

/**
 * stride_calibrate_log3 -- Measure per-radix log3 thresholds.
 *
 * For each radix R with log3 support, benchmarks flat vs log3
 * across K = 4, 8, 16, ..., 1024. The threshold is the smallest K
 * where log3 wins. If log3 never wins, threshold is set to SIZE_MAX
 * (effectively disabling it).
 *
 * Results are stored in wis->log3 and will be saved to the wisdom file.
 */
static void stride_calibrate_log3(stride_wisdom_t *wis,
                                   const stride_registry_t *reg) {
    static const size_t sweep_Ks[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 0};

    for (int R = 2; R < STRIDE_REG_MAX_RADIX; R++) {
        if (!reg->t1_fwd_log3[R]) continue;

        int inner = _log3_calib_inner(R, reg);
        int N = inner * R;
        int factors[2] = {inner, R};
        int nf = 2;

        size_t threshold = (size_t)-1;  /* default: log3 never wins */

        for (const size_t *kp = sweep_Ks; *kp; kp++) {
            size_t K = *kp;

            double flat_ns = _log3_calib_bench(N, K, factors, nf, 0, reg);
            double log3_ns = _log3_calib_bench(N, K, factors, nf, (1 << 1), reg);

            /* Log3 must win by >2% to filter timing noise */
            if (log3_ns < flat_ns * 0.98) {
                threshold = K;
                break;  /* first K where log3 reliably wins */
            }
        }

        wis->log3.threshold_K[R] = threshold;
        wis->log3.calibrated[R] = 1;
    }
}

/* ═══════════════════════════════════════════════════════════════
 * PLAN CONSTRUCTION HELPERS
 * ═══════════════════════════════════════════════════════════════ */

/**
 * Build codelet arrays from registry for a given factorization.
 * log3_mask: bitmask where bit s means use log3 for stage s.
 * If log3_mask == -1, use heuristic (or calibrated thresholds if provided).
 * thresholds: optional calibrated log3 thresholds (NULL = use heuristic).
 */
static stride_plan_t *_stride_build_plan_ex(
        int N, size_t K,
        const int *factors, int nf, int log3_mask,
        const stride_registry_t *reg,
        const stride_log3_thresholds_t *thresholds) {
    stride_n1_fn n1f[FACT_MAX_STAGES], n1b[FACT_MAX_STAGES];
    stride_t1_fn t1f[FACT_MAX_STAGES], t1b[FACT_MAX_STAGES];

    for (int s = 0; s < nf; s++) {
        int R = factors[s];
        n1f[s] = reg->n1_fwd[R];
        n1b[s] = reg->n1_bwd[R];

        if (s == 0) {
            t1f[s] = NULL;
            t1b[s] = NULL;
        } else if (log3_mask == -1) {
            /* Heuristic or calibrated selection */
            t1f[s] = stride_select_t1_fwd_calibrated(R, K, reg, thresholds);
            t1b[s] = stride_select_t1_bwd_calibrated(R, K, reg, thresholds);
        } else {
            /* Explicit log3 mask */
            if ((log3_mask >> s) & 1) {
                t1f[s] = reg->t1_fwd_log3[R] ? reg->t1_fwd_log3[R] : reg->t1_fwd[R];
                t1b[s] = reg->t1_bwd_log3[R] ? reg->t1_bwd_log3[R] : reg->t1_bwd[R];
            } else {
                t1f[s] = reg->t1_fwd[R];
                t1b[s] = reg->t1_bwd[R];
            }
        }
    }

    /* Resolve log3_mask: if heuristic (-1), compute actual mask using
     * calibrated thresholds (if available) or heuristic fallback */
    int resolved_mask = 0;
    if (log3_mask == -1) {
        for (int s = 1; s < nf; s++) {
            if (stride_should_use_log3_calibrated(factors[s], K, reg, thresholds))
                resolved_mask |= (1 << s);
        }
    } else {
        resolved_mask = log3_mask;
    }

    return stride_plan_create(N, K, factors, nf, n1f, n1b, t1f, t1b, resolved_mask);
}

/* Convenience wrapper: no calibrated thresholds (uses heuristic fallback) */
static stride_plan_t *_stride_build_plan(
        int N, size_t K,
        const int *factors, int nf, int log3_mask,
        const stride_registry_t *reg) {
    return _stride_build_plan_ex(N, K, factors, nf, log3_mask, reg, NULL);
}

/* ═══════════════════════════════════════════════════════════════
 * PUBLIC API
 * ═══════════════════════════════════════════════════════════════ */

/* ── Helpers for prime-size dispatch ── */

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
 * stride_auto_plan — Heuristic planner (fast).
 *
 * 1. Direct factorization into available radixes     → staged plan
 * 2. Prime N, non-smooth N-1 (p-1 has factors > 19) �� Bluestein
 * 3. Prime N, smooth N-1 (p-1 is 19-smooth)         → Rader
 * 4. Composite with unfactorable prime factor        → NULL (TODO)
 */
static stride_plan_t *stride_auto_plan(int N, size_t K,
                                        const stride_registry_t *reg) {
    /* 1. Direct factorization */
    stride_factorization_t fact;
    if (stride_factorize(N, K, reg, &fact) == 0)
        return _stride_build_plan(N, K, fact.factors, fact.nfactors, -1, reg);

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
 * stride_exhaustive_plan — Exhaustive search planner (slow).
 *
 * Tries all factorizations × orderings × log3 combos, benchmarks each,
 * returns a plan built from the best one found.
 *
 * Returns NULL if N cannot be factored.
 */
static stride_plan_t *stride_exhaustive_plan(int N, size_t K,
                                              const stride_registry_t *reg) {
    stride_factorization_t best_fact;
    int log3_mask = 0;
    double best_ns = stride_exhaustive_search(N, K, reg, &best_fact, &log3_mask, NULL, 0);
    if (best_ns >= 1e17)
        return NULL;

    /* Rebuild with the exact log3_mask the exhaustive search found best */
    return _stride_build_plan(N, K, best_fact.factors, best_fact.nfactors,
                              log3_mask, reg);
}

/**
 * stride_wise_plan — Wisdom-aware planner.
 *
 * Checks wisdom for a cached result. If found, builds plan from it.
 * Otherwise falls back to heuristic (stride_auto_plan).
 *
 * Returns NULL if N cannot be factored.
 */
static stride_plan_t *stride_wise_plan(int N, size_t K,
                                        const stride_registry_t *reg,
                                        const stride_wisdom_t *wis) {
    const stride_wisdom_entry_t *e = stride_wisdom_lookup(wis, N, K);
    if (e) {
        /* Explicit log3_mask from wisdom — use as-is */
        return _stride_build_plan(N, K, e->factors, e->nfactors,
                                  e->log3_mask, reg);
    }
    /* No wisdom for this (N,K): use heuristic with calibrated thresholds */
    stride_factorization_t fact;
    if (stride_factorize(N, K, reg, &fact) == 0)
        return _stride_build_plan_ex(N, K, fact.factors, fact.nfactors,
                                     -1, reg, &wis->log3);
    return stride_auto_plan(N, K, reg);
}

/**
 * _stride_refine_bench — Re-bench a factorization with high accuracy.
 *
 * The exhaustive search uses reduced reps for speed, producing noisy timings.
 * This function re-benchmarks the winner with proper warmup, reps, and trials
 * to get an accurate timing for the wisdom file.
 */
static double _stride_refine_bench(int N, size_t K,
                                    const int *factors, int nf, int log3_mask,
                                    const stride_registry_t *reg) {
    stride_plan_t *plan = _stride_build_plan(N, K, factors, nf, log3_mask, reg);
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
 * stride_wisdom_calibrate — Run exhaustive search and store result in wisdom.
 *
 * After exhaustive search finds the best factorization, re-benchmarks it
 * with full accuracy before storing. This avoids the noisy timings from
 * the reduced-rep exhaustive search.
 */
/**
 * stride_wisdom_ensure_log3 -- Calibrate log3 thresholds if not done yet.
 *
 * Called automatically before the first exhaustive search.
 * Runs stride_calibrate_log3() once, stores results in wis->log3.
 * Subsequent calls are no-ops (thresholds already calibrated).
 */
static void stride_wisdom_ensure_log3(stride_wisdom_t *wis,
                                       const stride_registry_t *reg) {
    /* Check if any radix is already calibrated */
    for (int R = 2; R < STRIDE_REG_MAX_RADIX; R++)
        if (wis->log3.calibrated[R]) return;  /* already done */
    stride_calibrate_log3(wis, reg);
}

static void stride_wisdom_calibrate(stride_wisdom_t *wis, int N, size_t K,
                                     const stride_registry_t *reg) {
    /* Ensure log3 thresholds are calibrated before exhaustive search */
    stride_wisdom_ensure_log3(wis, reg);

    stride_factorization_t best_fact;
    int log3_mask = 0;
    double search_ns = stride_exhaustive_search(N, K, reg, &best_fact, &log3_mask, &wis->log3, 0);
    if (search_ns >= 1e17) return;

    /* Re-bench the winner with full accuracy using its exact log3_mask */
    double refined_ns = _stride_refine_bench(N, K, best_fact.factors,
                                              best_fact.nfactors, log3_mask, reg);

    stride_wisdom_add(wis, N, K, best_fact.factors, best_fact.nfactors,
                      log3_mask, refined_ns);
}

#endif /* STRIDE_PLANNER_H */
