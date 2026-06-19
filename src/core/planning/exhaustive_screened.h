/* exhaustive_screened.h — V4-cost-model-screened exhaustive search.
 *
 * Why this exists
 * ───────────────
 * Flat exhaustive (exhaustive_plan.h) benches every (multiset ×
 * permutation) of N at the parent (N, K) context. The bench at parent
 * context is what makes it accurate — it captures the real cache/TLB
 * interactions of the FULL plan. But ~500-1500 benches per cell is
 * expensive, and most of those candidates are obviously bad and never
 * threaten the winner.
 *
 * Recursive memoized exhaustive (FFTW-style) was tried and hits the
 * same wall the DP planner hits — caching sub-plans by ISOLATED
 * measurement at (N_sub, K_eff_sub) loses the in-context signal that
 * makes tail-heavy patterns ([..., 32, 16]) win in composition. TOP-K
 * alone can't recover it. See memory note v4_estimate_and_exhaustive_port.
 *
 * V4-SCREENED: same enumeration as flat, but rank candidates by the
 * V4 cost model BEFORE benching. V4 scores the FULL plan at parent
 * (N, K) — same context the final bench will be at — so its ranking
 * preserves tail-heavy patterns by design (the wide-radix-outer
 * penalty + per-stage buffer-pass terms encode "outer-small +
 * inner-large = good"). Bench only the top-M by V4 rank.
 *
 * The cost-model journey memory ([cost_model_recalibration]) and
 * [v4_estimate_and_exhaustive_port] establish that V4 lands rank-1 on
 * the canonical cells. M=10 catches small ranking errors; M=20-25 is
 * conservative for cells where V4's wide-R-outer penalty pushes the
 * true winner to rank ~20. The trade-off is concrete:
 *
 *   Flat:             ~580 benches, picks true winner
 *   V4-Screened M=10:  ~10 benches, picks within 0-30% of true winner
 *   V4-Screened M=25:  ~25 benches, picks within 0-3% of true winner
 *
 * Algorithm:
 *   1. Enumerate every (multiset × permutation) using the same code
 *      paths as flat (vfft_proto_enumerate_factorizations + perm gen).
 *   2. Score each candidate with V4.
 *   3. Keep top-M lowest V4-score.
 *   4. Bench each top-M at parent (N, K) using the exhaustive_plan.h
 *      bench harness.
 *   5. Return the MEASUREMENT winner.
 *
 * Notes:
 *   - "Slightly more efficient" exhaustive — preserves coverage, cuts
 *     the bench count by ~50×.
 *   - V4-rank is a STRUCTURAL prior, not a measurement. Avoids the
 *     isolated-bench trap that DP and FFTW-recursive fall into.
 *   - When V4 has rank-2 or rank-3 error (rare), M needs to be large
 *     enough to catch it. M=10 is robust on tested cells; raise if
 *     you find a cell where V4 underranks the true winner by >10.
 */
#ifndef VFFT_PROTO_CORE_EXHAUSTIVE_SCREENED_H
#define VFFT_PROTO_CORE_EXHAUSTIVE_SCREENED_H

#include "plan.h"
#include "executor.h"
#include "planner.h"
#include "exhaustive_plan.h"     /* enumeration, perm gen, bench_one */
#include "estimate_plan.h"       /* _vfft_proto_v4_score, stride_detect_cpu */
#include "dp_planner.h"          /* now_ns, pacing */
#include "registry.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef VFFT_PROTO_SCREEN_TOPM_DEFAULT
#define VFFT_PROTO_SCREEN_TOPM_DEFAULT 10
#endif

/* A scored candidate — one (multiset × permutation) plus its V4 cost. */
typedef struct {
    int    factors[STRIDE_MAX_STAGES];
    int    nfactors;
    double v4_score;
} vfft_proto_screen_cand_t;

static int _vfft_proto_screen_cand_cmp(const void *a, const void *b) {
    double sa = ((const vfft_proto_screen_cand_t *)a)->v4_score;
    double sb = ((const vfft_proto_screen_cand_t *)b)->v4_score;
    return (sa < sb) ? -1 : (sa > sb) ? 1 : 0;
}

/* ─────────────────────────────────────────────────────────────────
 * V4-SCREENED EXHAUSTIVE
 *
 * Returns measured ns of the winner. Fills best_fact with its
 * factorization. top_m is the number of benched candidates; default
 * via VFFT_PROTO_SCREEN_TOPM_DEFAULT.
 * ───────────────────────────────────────────────────────────────── */

static inline double vfft_proto_screened_exhaustive_search(
    int N, size_t K,
    const vfft_proto_registry_t *reg,
    int top_m,
    vfft_proto_factorization_t *best_fact,
    int verbose)
{
    if (top_m <= 0) top_m = VFFT_PROTO_SCREEN_TOPM_DEFAULT;

    /* Step 1: enumerate every (multiset × permutation). Re-uses
     * exhaustive_plan.h's enumerator + permutation generator. */
    vfft_proto_factorization_list_t *flist =
        (vfft_proto_factorization_list_t *)malloc(sizeof(*flist));
    vfft_proto_enumerate_factorizations(N, reg, flist);

    /* Score buffer — sized to hold all (mset × perm) candidates. With
     * STRIDE_MAX_STAGES=16 and max perms = 720 per multiset, and ~30
     * multisets for pow2 N, we expect <= ~5000 candidates. Allocate
     * conservatively. */
    int max_cands = flist->count * VFFT_PROTO_DP_MAX_PERMS;
    if (max_cands < 1024) max_cands = 1024;
    vfft_proto_screen_cand_t *cands =
        (vfft_proto_screen_cand_t *)malloc(max_cands * sizeof(*cands));
    int n_cands = 0;

    /* Step 2: score each candidate with V4. */
    stride_cpu_info_t cpu = stride_detect_cpu();

    for (int fi = 0; fi < flist->count; fi++) {
        const vfft_proto_factorization_t *f = &flist->results[fi];

        vfft_proto_perm_list_t *plist =
            (vfft_proto_perm_list_t *)malloc(sizeof(*plist));
        vfft_proto_gen_permutations(f->factors, f->nfactors, plist);

        for (int pi = 0; pi < plist->count; pi++) {
            int prod = 1;
            for (int s = 0; s < f->nfactors; s++) prod *= plist->perms[pi][s];
            if (prod != N) continue;
            if (n_cands >= max_cands) break;

            const int *factors_p = plist->perms[pi];
            int nf = f->nfactors;

            /* Sanity: every factor must have an n1+t1s codelet. */
            int ok = 1;
            for (int s = 0; s < nf; s++) {
                int R = factors_p[s];
                if (R <= 0 || R >= VFFT_PROTO_REG_MAX_RADIX
                    || !reg->n1_fwd[R] || !reg->t1s_dit_fwd[R]) {
                    ok = 0; break;
                }
            }
            if (!ok) continue;

            double sc = _vfft_proto_v4_score(factors_p, nf, K, N, &cpu);

            memcpy(cands[n_cands].factors, factors_p, nf * sizeof(int));
            cands[n_cands].nfactors = nf;
            cands[n_cands].v4_score = sc;
            n_cands++;
        }
        free(plist);
    }
    free(flist);

    if (verbose >= 1)
        printf("  [v4-screened] N=%d K=%zu: %d candidates enumerated, "
               "ranking by V5 cost model (V4 minus wide_penalty)...\n",
               N, (size_t)K, n_cands);

    if (n_cands == 0) {
        free(cands);
        best_fact->nfactors = 0;
        return 1e18;
    }

    /* Step 3: sort by V4 ascending, keep top-M. */
    qsort(cands, n_cands, sizeof(*cands), _vfft_proto_screen_cand_cmp);
    int n_bench = (n_cands < top_m) ? n_cands : top_m;

    if (verbose >= 1) {
        printf("  [v4-screened] V4 top-%d:\n", n_bench);
        for (int i = 0; i < n_bench; i++) {
            printf("    rank %2d: ", i + 1);
            for (int s = 0; s < cands[i].nfactors; s++)
                printf("%s%d", s ? "x" : "", cands[i].factors[s]);
            printf("  v4=%.3e\n", cands[i].v4_score);
        }
    }

    /* Step 4: bench each top-M at parent (N, K). */
    size_t total = (size_t)N * K;
    double *re = NULL, *im = NULL, *orig_re = NULL, *orig_im = NULL;
    vfft_proto_posix_memalign((void**)&re,      64, total * sizeof(double));
    vfft_proto_posix_memalign((void**)&im,      64, total * sizeof(double));
    vfft_proto_posix_memalign((void**)&orig_re, 64, total * sizeof(double));
    vfft_proto_posix_memalign((void**)&orig_im, 64, total * sizeof(double));
    srand(42);
    for (size_t i = 0; i < total; i++) {
        orig_re[i] = (double)rand() / RAND_MAX - 0.5;
        orig_im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    double global_best_ns = 1e18;
    int    global_best_i  = -1;

    for (int i = 0; i < n_bench; i++) {
        double ns = vfft_proto_bench_one(
            N, K, cands[i].factors, cands[i].nfactors, reg,
            re, im, orig_re, orig_im);

        if (verbose >= 1) {
            printf("    bench rank %2d: ", i + 1);
            for (int s = 0; s < cands[i].nfactors; s++)
                printf("%s%d", s ? "x" : "", cands[i].factors[s]);
            printf("  = %.1f ns%s\n", ns,
                   (ns < global_best_ns) ? "  ← new best" : "");
        }

        if (ns < global_best_ns) {
            global_best_ns = ns;
            global_best_i = i;
        }

        if (VFFT_PROTO_DP_PACE_MS > 0 &&
            ((i + 1) % VFFT_PROTO_DP_PACE_EVERY) == 0) {
            _vfft_proto_dp_sleep_ms(VFFT_PROTO_DP_PACE_MS);
        }
    }

    /* Step 5: return the measured winner. */
    best_fact->nfactors = 0;
    if (global_best_i >= 0) {
        best_fact->nfactors = cands[global_best_i].nfactors;
        memcpy(best_fact->factors, cands[global_best_i].factors,
               (size_t)cands[global_best_i].nfactors * sizeof(int));
    }

    if (verbose >= 1 && global_best_i >= 0) {
        printf("  [v4-screened] Best (V4-rank %d / %d enumerated): ",
               global_best_i + 1, n_cands);
        for (int s = 0; s < best_fact->nfactors; s++)
            printf("%s%d", s ? "x" : "", best_fact->factors[s]);
        printf(" = %.1f ns\n", global_best_ns);
    }

    free(cands);
    vfft_proto_aligned_free(re);
    vfft_proto_aligned_free(im);
    vfft_proto_aligned_free(orig_re);
    vfft_proto_aligned_free(orig_im);
    return global_best_ns;
}

/* Top-level convenience: returns a plan built from the screened
 * search's pick. */
static inline stride_plan_t *vfft_proto_screened_exhaustive_plan(
    int N, size_t K, const vfft_proto_registry_t *reg,
    int top_m, int verbose)
{
    vfft_proto_factorization_t best;
    double ns = vfft_proto_screened_exhaustive_search(
        N, K, reg, top_m, &best, verbose);
    (void)ns;
    if (best.nfactors == 0) return NULL;
    return vfft_proto_plan_create(N, K, best.factors, /*variants=*/NULL,
                                  best.nfactors, reg);
}

/* Verbose variant: reports picked factorization + measured ns. */
static inline stride_plan_t *vfft_proto_screened_exhaustive_plan_verbose(
    int N, size_t K, const vfft_proto_registry_t *reg,
    int top_m,
    int *out_factors, int *out_nf, double *out_ns,
    int verbose)
{
    vfft_proto_factorization_t best;
    double ns = vfft_proto_screened_exhaustive_search(
        N, K, reg, top_m, &best, verbose);
    if (out_ns) *out_ns = ns;
    if (best.nfactors == 0) {
        if (out_nf) *out_nf = 0;
        return NULL;
    }
    if (out_factors) memcpy(out_factors, best.factors,
                            (size_t)best.nfactors * sizeof(int));
    if (out_nf)      *out_nf = best.nfactors;
    return vfft_proto_plan_create(N, K, best.factors, /*variants=*/NULL,
                                  best.nfactors, reg);
}

#endif /* VFFT_PROTO_CORE_EXHAUSTIVE_SCREENED_H */
