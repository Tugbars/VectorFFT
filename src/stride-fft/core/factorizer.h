/**
 * stride_factorizer.h — CPU-aware factorizer for stride-based FFT executor
 *
 * Given N, decomposes into available radixes with cache-aware ordering.
 *
 * Design:
 *   1. Find all possible radix decompositions of N
 *   2. For each decomposition, score based on cache behavior
 *   3. Pick the best, or try all orderings (auto-tune mode)
 *
 * Cache model:
 *   Stage s processes groups of R_s butterflies at stride S_s.
 *   Each butterfly touches R_s elements separated by S_s doubles.
 *   Working set per group = R_s * S_s * 16 bytes (split complex).
 *   Twiddle table per stage = (R_s - 1) * K * 16 bytes.
 *
 *   Optimal: working set fits L1 for all stages.
 *   Reality: outer stages exceed L1 → minimize damage.
 */
#ifndef STRIDE_FACTORIZER_H
#define STRIDE_FACTORIZER_H

#include "registry.h"
#include <string.h>

#define FACT_MAX_STAGES 8
#define FACT_MAX_DECOMPOSITIONS 64

/* ═══════════════════════════════════════════════════════════════
 * CPU CACHE DETECTION
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    size_t l1d_bytes;    /* L1 data cache size */
    size_t l2_bytes;     /* L2 cache size */
    size_t cache_line;   /* cache line size (bytes) */
} stride_cpu_info_t;

#ifdef _WIN32
#include <intrin.h>
#endif

static stride_cpu_info_t stride_detect_cpu(void) {
    stride_cpu_info_t info = {48 * 1024, 2 * 1024 * 1024, 64}; /* defaults */

#if defined(_WIN32) && (defined(_MSC_VER) || defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER))
    /* CPUID leaf 4: deterministic cache parameters */
    for (int sub = 0; sub < 8; sub++) {
        int cpuinfo[4] = {0};
        __cpuidex(cpuinfo, 4, sub);
        int type = cpuinfo[0] & 0x1F;
        if (type == 0) break; /* no more caches */
        int level = (cpuinfo[0] >> 5) & 0x7;
        size_t ways  = (size_t)(((unsigned)cpuinfo[1] >> 22) & 0x3FF) + 1;
        size_t parts = (size_t)(((unsigned)cpuinfo[1] >> 12) & 0x3FF) + 1;
        size_t line  = (size_t)((unsigned)cpuinfo[1] & 0xFFF) + 1;
        size_t sets  = (size_t)(unsigned)cpuinfo[2] + 1;
        size_t sz = ways * parts * line * sets;
        if (level == 1 && (type == 1 || type == 3)) info.l1d_bytes = sz; /* data or unified */
        if (level == 2) info.l2_bytes = sz;
        info.cache_line = line;
    }
#elif defined(__linux__)
    {
        long l1 = sysconf(_SC_LEVEL1_DCACHE_SIZE);
        long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
        if (l1 > 0) info.l1d_bytes = (size_t)l1;
        if (l2 > 0) info.l2_bytes = (size_t)l2;
    }
#endif
    return info;
}

/* ═══════════════════════════════════════════════════════════════
 * FACTORIZATION RESULT
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    int factors[FACT_MAX_STAGES];
    int nfactors;
} stride_factorization_t;

/* ═══════════════════════════════════════════════════════════════
 * GREEDY FACTORIZER — composite-preferring
 *
 * Step 1: Decompose N into available radixes, preferring composites
 *         (R=25,20,12,10) over equivalent small-factor decompositions.
 *         This minimizes stage count.
 *
 * Step 2: Order factors based on K.
 *         Small K (<=16): descending (largest first) — everything fits L1.
 *         Large K (>16):  ascending (smallest first) — reduce stride pressure
 *                         on outer stages, put big radix last (stride=K, sequential).
 *
 * Step 3: Push R=64 away from first stage (its 2225-op codelet has too many
 *         compiler spills to be efficient as the first pass).
 * ═══════════════════════════════════════════════════════════════ */

/* Radixes sorted by preference for decomposition: composites first to absorb factors */
static const int FACTORIZE_RADIXES[] = {
    25, 20, 12, 10, 32, 16, 8, 7, 6, 5, 4, 3, 2,
    64, /* R=64 last in decomposition preference — only if needed */
    19, 17, 13, 11, /* odd primes last */
    0
};

static void _sort_factors_ascending(int *f, int n) {
    for (int i = 1; i < n; i++) {
        int key = f[i]; int j = i - 1;
        while (j >= 0 && f[j] > key) { f[j+1] = f[j]; j--; }
        f[j+1] = key;
    }
}

static void _sort_factors_descending(int *f, int n) {
    for (int i = 1; i < n; i++) {
        int key = f[i]; int j = i - 1;
        while (j >= 0 && f[j] < key) { f[j+1] = f[j]; j--; }
        f[j+1] = key;
    }
}

/* Check if a composite radix should be decomposed at this K.
 * Returns 1 if the composite's twiddle table overflows the L1 twiddle budget,
 * meaning we should use smaller sub-radixes instead.
 *
 * Budget = L1 / 3 (twiddle shares L1 with data and codelet spills).
 * Overflow when: (R-1) * K * 16 > budget */
static int _should_decompose(int R, size_t K, size_t l1_bytes) {
    size_t tw_budget = l1_bytes / 3;
    size_t tw_bytes = (size_t)(R - 1) * K * 16;
    return tw_bytes > tw_budget;
}

static int stride_factorize_greedy(int N, size_t K,
                                   const stride_registry_t *reg,
                                   const stride_cpu_info_t *cpu,
                                   stride_factorization_t *fact) {
    memset(fact, 0, sizeof(*fact));
    if (N <= 1) return 0;

    const size_t l1 = cpu->l1d_bytes;
    int remaining = N;
    int nf = 0;

    /* Decompose N into available radixes.
     *
     * At small K: prefer composites (fewer stages, less overhead).
     * At large K: skip composites whose twiddle tables overflow L1,
     * let them decompose into smaller radixes that fit.
     *
     * The K threshold is per-radix: (R-1)*K*16 > L1/3. */
    while (remaining > 1 && nf < FACT_MAX_STAGES) {
        int best_R = 0;
        for (const int *rp = FACTORIZE_RADIXES; *rp; rp++) {
            int R = *rp;
            if (remaining % R != 0) continue;
            if (!stride_registry_has(reg, R)) continue;

            /* For twiddled stages (nf > 0): check if this radix's twiddle
             * table would overflow L1. If so, skip to smaller radixes. */
            if (nf > 0 && _should_decompose(R, K, l1)) continue;

            best_R = R;
            break;
        }

        /* Fallback: if nothing fits L1, pick smallest available */
        if (!best_R) {
            for (const int *rp = FACTORIZE_RADIXES; *rp; rp++) {
                int R = *rp;
                if (remaining % R != 0) continue;
                if (!stride_registry_has(reg, R)) continue;
                if (!best_R || R < best_R) best_R = R;
            }
        }

        if (!best_R) return -1;
        fact->factors[nf++] = best_R;
        remaining /= best_R;
    }
    if (remaining != 1) return -1;
    fact->nfactors = nf;

    /* Order based on K:
     * Small K (<=16): descending — large radixes first, everything fits L1.
     * Large K (>16): ascending — small radixes first (small stride on outer stages),
     *                large radixes last (stride=K, sequential access). */
    if (K <= 16) {
        _sort_factors_descending(fact->factors, nf);
    } else {
        _sort_factors_ascending(fact->factors, nf);
    }

    /* Push R=64 away from first stage: its 2225-op codelet has too many
     * compiler spills. Better as innermost (stride=K, sequential). */
    if (nf > 1 && fact->factors[0] >= 64) {
        int tmp = fact->factors[0];
        for (int i = 0; i < nf - 1; i++) fact->factors[i] = fact->factors[i+1];
        fact->factors[nf-1] = tmp;
    }

    /* ── Blocked decomposition for R=32 and R=64 ──
     *
     * Bench data:
     *   R=32 → [4,8]: wins at K=16..256 (~1.05-1.8x), loses at K<=4, K>=512
     *   R=64 → [8,8]: wins at K=4..256  (~1.2-2.0x),  loses at K>=512
     *
     * Split monolithic R=32/64 into sub-radixes when K is in the sweet spot.
     * The twiddle tables of the sub-radixes fit L1 where the monolithic
     * tables overflow, eliminating the cache cliff.
     */
    if (K >= 8 && K <= 256) {
        int new_factors[FACT_MAX_STAGES];
        int new_nf = 0;
        for (int i = 0; i < nf && new_nf < FACT_MAX_STAGES - 1; i++) {
            if (fact->factors[i] == 64) {
                new_factors[new_nf++] = 8;
                new_factors[new_nf++] = 8;
            } else if (fact->factors[i] == 32) {
                new_factors[new_nf++] = 4;
                new_factors[new_nf++] = 8;
            } else {
                new_factors[new_nf++] = fact->factors[i];
            }
        }
        memcpy(fact->factors, new_factors, new_nf * sizeof(int));
        fact->nfactors = new_nf;
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * SCORE A FACTORIZATION ORDER
 *
 * Lower score = better. Estimates total cache misses.
 *
 * For each stage s (processed in order 0, 1, ..., nf-1):
 *   stride_s = K * product of factors[s+1 .. nf-1]
 *   working_set_s = R_s * stride_s * 16 bytes
 *   groups_s = N / R_s
 *
 * If working_set fits L1: cost = groups * R * K (pure arithmetic)
 * If working_set fits L2: cost = groups * R * K * 2 (L2 latency penalty)
 * If exceeds L2:          cost = groups * R * K * 8 (L3/DRAM penalty)
 *
 * Twiddle overhead:
 *   For stage 0: no twiddle
 *   For stages 1+: tw_size = (R-1) * K_accumulated * 16
 *   If tw_size > L1: add penalty (twiddle misses)
 * ═══════════════════════════════════════════════════════════════ */

static double stride_score_factorization(const int *factors, int nf, size_t K,
                                         int N, const stride_cpu_info_t *cpu) {
    const size_t l1 = cpu->l1d_bytes;
    const size_t l2 = cpu->l2_bytes;
    double score = 0.0;

    size_t accumulated_K = K;

    for (int s = 0; s < nf; s++) {
        int R = factors[s];
        int groups = N / R;

        /* Stride for this stage */
        size_t stride = K;
        for (int d = s + 1; d < nf; d++) stride *= factors[d];

        /* Working set per group */
        size_t ws = (size_t)R * stride * 16;

        /* Data access cost */
        double data_cost;
        if (ws <= l1)
            data_cost = (double)groups * R * K;
        else if (ws <= l2)
            data_cost = (double)groups * R * K * 3.0;
        else
            data_cost = (double)groups * R * K * 10.0;

        /* Twiddle cost (stages 1+) */
        double tw_cost = 0.0;
        if (s > 0) {
            size_t tw_bytes = (size_t)(R - 1) * accumulated_K * 16;
            if (tw_bytes > l1)
                tw_cost = (double)(R - 1) * accumulated_K * 4.0;
            else
                tw_cost = (double)(R - 1) * accumulated_K;
        }

        score += data_cost + tw_cost;
        accumulated_K *= R;
    }

    return score;
}

/* ═══════════════════════════════════════════════════════════════
 * PERMUTATION SEARCH
 *
 * Given a set of radixes (from greedy factorizer), try all orderings
 * and pick the one with best score. For nf <= 4 (24 permutations)
 * this is instant. For nf=5 (120) still fast.
 *
 * For nf > 5, use greedy order only (720+ permutations too slow
 * for heuristic scoring; use auto-tune with real benchmarks instead).
 * ═══════════════════════════════════════════════════════════════ */

static void _permute_and_score(int *arr, int n, int depth,
                               size_t K, int N, const stride_cpu_info_t *cpu,
                               int *best_order, double *best_score) {
    if (depth == n) {
        double sc = stride_score_factorization(arr, n, K, N, cpu);
        if (sc < *best_score) {
            *best_score = sc;
            memcpy(best_order, arr, n * sizeof(int));
        }
        return;
    }
    for (int i = depth; i < n; i++) {
        /* swap arr[depth] and arr[i] */
        int tmp = arr[depth]; arr[depth] = arr[i]; arr[i] = tmp;
        _permute_and_score(arr, n, depth + 1, K, N, cpu, best_order, best_score);
        arr[depth] = arr[i]; arr[i] = tmp; /* swap back */
    }
}

static void stride_optimize_order(stride_factorization_t *fact, size_t K,
                                  int N, const stride_cpu_info_t *cpu) {
    if (fact->nfactors <= 1) return;

    /* For large nf, skip permutation search */
    if (fact->nfactors > 5) return;

    int work[FACT_MAX_STAGES];
    int best[FACT_MAX_STAGES];
    double best_score = 1e30;

    memcpy(work, fact->factors, fact->nfactors * sizeof(int));
    memcpy(best, fact->factors, fact->nfactors * sizeof(int));

    _permute_and_score(work, fact->nfactors, 0, K, N, cpu, best, &best_score);

    memcpy(fact->factors, best, fact->nfactors * sizeof(int));
}

/* ═══════════════════════════════════════════════════════════════
 * TOP-LEVEL FACTORIZER
 *
 * 1. Greedy decomposition (finds the radix SET)
 * 2. Permutation search (finds the best ORDER)
 * ═══════════════════════════════════════════════════════════════ */

static int stride_factorize(int N, size_t K,
                            const stride_registry_t *reg,
                            stride_factorization_t *fact) {
    stride_cpu_info_t cpu = stride_detect_cpu();

    /* Greedy decomposition with K-aware ordering (v2).
     * The greedy factorizer already sorts based on K:
     *   small K → descending, large K → ascending.
     * No additional permutation search — the scoring model is unreliable
     * and the swap-based permuter has corruption bugs.
     * Use exhaustive search (wisdom) for truly optimal ordering. */
    return stride_factorize_greedy(N, K, reg, &cpu, fact);
}

/* ═══════════════════════════════════════════════════════════════
 * LOG3 TWIDDLE SELECTION
 *
 * For each radix R with a log3 codelet, there's a threshold K
 * above which log3 beats flat. This threshold is either:
 *   1. Calibrated: measured by sweeping K, stored in wisdom file
 *   2. Estimated: physics-based model using codelet profiles
 *
 * The calibrated thresholds live in stride_log3_thresholds_t,
 * which is populated from the wisdom file's @log3 lines.
 * When not calibrated, the estimate model is used as fallback.
 *
 * Physics model:
 *   Log3 trades memory loads for compute (cmul derivations).
 *   Two regimes govern the crossover:
 *
 *   1. Register-pressure regime (spill_bytes >= 768):
 *      The flat codelet already spills twiddles to stack.
 *      Log3 reduces live registers → eliminates spills.
 *      Threshold is LOW (16-128) — log3 wins at small K.
 *      K_est = L1 / (deriv_cost_ratio * row_bytes * pressure_factor)
 *
 *   2. L1-capacity regime (spill_bytes < 768):
 *      The flat codelet fits in registers at small K.
 *      Log3 only wins when twiddle TABLE exceeds L1.
 *      K_est = L1 / (2 * row_bytes) — classic cache model.
 *
 *   3. Genfft primes (R=11,13,17,19):
 *      DAG-optimized with Sethi-Ullman scheduling.
 *      Analytical model unreliable → use per-radix heuristic.
 * ═══════════════════════════════════════════════════════════════ */

/* ── Codelet profiles for log3 estimation ──
 *
 * These describe the computational structure of each radix's
 * t1 DIT codelet, used to predict log3 crossover without benching.
 *
 * n_bases:    twiddle rows loaded from memory in log3 mode
 * n_derived:  twiddle rows computed via cmul chains
 * spill_bytes: approximate stack spill for flat codelet's twiddle
 *             handling (0 = all in registers, >0 = spills to stack).
 *             Computed as: simultaneous_twiddle_rows × 64 bytes (AVX2).
 * bf_flops:   FMA-equivalent ops in the butterfly (excl. twiddles)
 * is_genfft:  1 if DAG-optimized prime (Sethi-Ullman), 0 otherwise */

typedef struct {
    int     n_bases;
    int     n_derived;
    int     spill_bytes;
    int     bf_flops;
    int     is_genfft;
} stride_codelet_profile_t;

/* Profile table — indexed by radix.
 * Entries for radixes without log3 support are zero-filled (unused). */
static const stride_codelet_profile_t STRIDE_LOG3_PROFILES[] = {
    /*  R=0  */ { 0,  0,    0,    0, 0 },
    /*  R=1  */ { 0,  0,    0,    0, 0 },
    /*  R=2  */ { 0,  0,    0,    0, 0 },
    /*  R=3  */ { 1,  1,    0,   20, 0 },
    /*  R=4  */ { 1,  2,    0,   24, 0 },
    /*  R=5  */ { 1,  3,    0,   48, 0 },
    /*  R=6  */ { 1,  4,  320,   56, 0 },
    /*  R=7  */ { 1,  5,  384,   82, 0 },
    /*  R=8  */ { 0,  0,    0,    0, 0 },
    /*  R=9  */ { 0,  0,    0,    0, 0 },
    /* R=10  */ { 1,  8,  576,  120, 0 },
    /* R=11  */ { 1,  9,    0,   60, 1 },
    /* R=12  */ { 1, 10,  704,  140, 0 },
    /* R=13  */ { 1, 11,    0,   78, 1 },
    /* R=14  */ { 0,  0,    0,    0, 0 },
    /* R=15  */ { 0,  0,    0,    0, 0 },
    /* R=16  */ { 4, 10, 1024,  212, 0 },
    /* R=17  */ { 1, 15,    0,  136, 1 },
    /* R=18  */ { 0,  0,    0,    0, 0 },
    /* R=19  */ { 1, 17,    0,  190, 1 },
    /* R=20  */ { 2, 17, 1280,  280, 0 },
    /* R=21  */ { 0,  0,    0,    0, 0 },
    /* R=22  */ { 0,  0,    0,    0, 0 },
    /* R=23  */ { 0,  0,    0,    0, 0 },
    /* R=24  */ { 0,  0,    0,    0, 0 },
    /* R=25  */ { 2, 22, 1600,  580, 0 },
    /* R=26  */ { 0,  0,    0,    0, 0 },
    /* R=27  */ { 0,  0,    0,    0, 0 },
    /* R=28  */ { 0,  0,    0,    0, 0 },
    /* R=29  */ { 0,  0,    0,    0, 0 },
    /* R=30  */ { 0,  0,    0,    0, 0 },
    /* R=31  */ { 0,  0,    0,    0, 0 },
    /* R=32  */ { 5, 26, 2048,  876, 0 },
};
#define STRIDE_LOG3_PROFILE_COUNT (sizeof(STRIDE_LOG3_PROFILES)/sizeof(STRIDE_LOG3_PROFILES[0]))

/* Spill threshold: above this, codelet is in register-pressure regime */
#define STRIDE_SPILL_REGIME_BYTES 768

/* Estimate the log3 threshold K for a given radix using physics model.
 *
 * Returns estimated K threshold, or (size_t)-1 for NEVER.
 * The estimate is conservative — when in doubt, returns a higher K
 * (meaning log3 activates later, falling back to the safe flat path). */
static size_t stride_estimate_log3_threshold(int R, size_t l1_bytes) {
    if (R < 2 || (size_t)R >= STRIDE_LOG3_PROFILE_COUNT) return (size_t)-1;

    const stride_codelet_profile_t *p = &STRIDE_LOG3_PROFILES[R];
    if (p->bf_flops == 0) return (size_t)-1; /* no profile → no log3 */

    size_t row_bytes = (size_t)(R - 1) * 16; /* bytes per twiddle row per K */

    /* ── Genfft primes: per-radix heuristic ──
     * DAG-optimized codelets resist analytical modeling.
     * R=13,17: derivation hides behind long butterfly → wins early (K=32)
     * R=11: shorter butterfly, moderate derivation → wins at K=256
     * R=19: derivation cost exceeds butterfly capacity → NEVER */
    if (p->is_genfft) {
        if (R == 19) return (size_t)-1;  /* never wins */
        if (R == 13 || R == 17) return 32;
        if (R == 11) return 256;
        return 128; /* safe default for unknown genfft primes */
    }

    /* ── Register-pressure regime ──
     * Flat codelet spills twiddles to stack (spill >= 1024B).
     * Log3 reduces live twiddles from (R-1) to n_bases, eliminating
     * most spills. Benefit is immediate even at small K.
     *
     * K_est = L1 / (row_bytes * net_benefit)
     * net_benefit = spill_factor * hide_factor * load_reduction
     *
     * i-cache penalty: codelets >= 800 SIMD ops generate huge code
     * that doesn't fit in the uop cache, adding overhead that offsets
     * spill savings at small K. Multiply threshold by 8 for R >= 32. */
    if (p->spill_bytes >= 1024) {
        double spill_factor = (double)p->spill_bytes / 512.0;
        double hide_factor = (double)p->bf_flops / ((double)p->n_derived * 8.0);
        double load_red = (double)(R - 1 - p->n_bases) / (double)(R - 1);

        double benefit = spill_factor * hide_factor * load_red;
        if (benefit < 0.5) benefit = 0.5;
        if (benefit > 32.0) benefit = 32.0;

        size_t est = (size_t)((double)l1_bytes / ((double)row_bytes * benefit));

        /* i-cache penalty for very large codelets.
         * The log3 variant adds derivation code on top of an already huge
         * butterfly. With many bases, the codelet also still loads multiple
         * twiddle rows, reducing the load-elimination benefit.
         * Scale penalty by n_bases: more bases = less benefit = higher K. */
        if (p->bf_flops >= 800) est *= (unsigned)p->n_bases * 4;

        /* Round down to power of 2 */
        size_t k = 1;
        while (k * 2 <= est) k *= 2;
        if (k < 16) k = 16;
        if (k > 1024) k = 1024;
        return k;
    }

    /* ── Derivation-cost regime ──
     * For codelets with moderate or no spill, the crossover is governed
     * by how many twiddles must be derived (cmul chains) vs how much
     * butterfly work can hide the derivation latency.
     *
     * n_derived is the strongest predictor:
     *   - n_derived >= 8 (R=10,12): derivation is expensive, butterfly
     *     can't hide it. Log3 only wins at very large K (1024+) where
     *     the twiddle table completely overflows L1.
     *   - n_derived 3-7 (R=5,6,7): moderate cost. Log3 wins when
     *     table reaches ~L1 size. K_est = L1 / row_bytes.
     *   - n_derived < 3 (R=3,4): cheap derivation. Log3 wins even
     *     before full L1 overflow. K_est = L1 / (3 * row_bytes). */
    if (p->n_derived >= 8) {
        /* Heavy derivation — need massive L1 pressure to justify.
         * The twiddle table must substantially exceed L1 before the
         * cache-miss penalty outweighs the derivation overhead of
         * 8+ cmul chains. Use 2× L1 as the crossover point. */
        size_t k = (2 * l1_bytes) / row_bytes;
        /* Round up to power of 2 for safety margin */
        size_t p2 = 1;
        while (p2 < k) p2 *= 2;
        if (p2 > 2048) p2 = 2048;
        return p2;
    }

    if (p->n_derived >= 3) {
        /* Moderate derivation — table needs to approach L1 size. */
        size_t est = l1_bytes / row_bytes;
        size_t k = 1;
        while (k * 2 <= est) k *= 2;
        if (k < 128) k = 128;
        if (k > 1024) k = 1024;
        return k;
    }

    /* Light derivation (1-2 derived) — low overhead, wins early. */
    {
        size_t est = l1_bytes / (3 * row_bytes);
        size_t k = 1;
        while (k * 2 <= est) k *= 2;
        if (k < 64) k = 64;
        if (k > 1024) k = 1024;
        return k;
    }
}

typedef struct {
    size_t threshold_K[STRIDE_REG_MAX_RADIX];  /* 0 = not calibrated, use estimate */
    int calibrated[STRIDE_REG_MAX_RADIX];      /* 1 = threshold was measured */
} stride_log3_thresholds_t;

static void stride_log3_thresholds_init(stride_log3_thresholds_t *t) {
    memset(t, 0, sizeof(*t));
}

static inline int stride_should_use_log3(int R, size_t K,
                                         const stride_registry_t *reg) {
    if (!reg->t1_fwd_log3[R]) return 0;

    /* Physics-based estimate using codelet profiles + detected L1 size */
    stride_cpu_info_t cpu = stride_detect_cpu();
    size_t est_K = stride_estimate_log3_threshold(R, cpu.l1d_bytes);
    if (est_K == (size_t)-1) return 0; /* NEVER */
    return K >= est_K;
}

static inline int stride_should_use_log3_calibrated(
        int R, size_t K,
        const stride_registry_t *reg,
        const stride_log3_thresholds_t *thresholds) {
    if (!reg->t1_fwd_log3[R]) return 0;
    if (thresholds && thresholds->calibrated[R])
        return K >= thresholds->threshold_K[R];
    /* Fallback to heuristic if not calibrated */
    return stride_should_use_log3(R, K, reg);
}

/* Select t1 codelet (flat or log3) for a given radix and K */
static inline stride_t1_fn stride_select_t1_fwd(int R, size_t K,
                                                  const stride_registry_t *reg) {
    if (stride_should_use_log3(R, K, reg))
        return reg->t1_fwd_log3[R];
    return reg->t1_fwd[R];
}

static inline stride_t1_fn stride_select_t1_bwd(int R, size_t K,
                                                  const stride_registry_t *reg) {
    if (stride_should_use_log3(R, K, reg))
        return reg->t1_bwd_log3[R];
    return reg->t1_bwd[R];
}

static inline stride_t1_fn stride_select_t1_fwd_calibrated(
        int R, size_t K,
        const stride_registry_t *reg,
        const stride_log3_thresholds_t *thresholds) {
    if (stride_should_use_log3_calibrated(R, K, reg, thresholds))
        return reg->t1_fwd_log3[R];
    return reg->t1_fwd[R];
}

static inline stride_t1_fn stride_select_t1_bwd_calibrated(
        int R, size_t K,
        const stride_registry_t *reg,
        const stride_log3_thresholds_t *thresholds) {
    if (stride_should_use_log3_calibrated(R, K, reg, thresholds))
        return reg->t1_bwd_log3[R];
    return reg->t1_bwd[R];
}


#endif /* STRIDE_FACTORIZER_H */
