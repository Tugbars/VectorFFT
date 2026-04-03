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
 * For each twiddled stage, decide flat vs log3 t1 codelet.
 *
 * Heuristic: use log3 when the flat twiddle table overflows L1/2:
 *   (R-1) * K * 16 > 24576 bytes (24KB)
 *
 * Exceptions: R=5 and R=10 always flat — their log3 derivation
 * chain overhead exceeds the cache savings at all K values.
 *
 * R>=64: always n1_fallback (handled by executor, not here).
 * ═══════════════════════════════════════════════════════════════ */

#define STRIDE_LOG3_THRESHOLD 24576  /* bytes: L1/2 on 48KB L1 */

static inline int stride_should_use_log3(int R, size_t K,
                                         const stride_registry_t *reg) {
    /* No log3 codelet available */
    if (!reg->t1_fwd_log3[R]) return 0;

    /* R=5, R=10: flat always wins (bench verified) */
    if (R == 5 || R == 10) return 0;

    /* Use log3 when flat twiddle table overflows L1/2 */
    size_t tw_bytes = (size_t)(R - 1) * K * 16;
    return tw_bytes > STRIDE_LOG3_THRESHOLD;
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


#endif /* STRIDE_FACTORIZER_H */
