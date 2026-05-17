/* exhaustive_patient.h — high-fidelity exhaustive (no pre-screen,
 *                        deeper bench, inter-candidate pacing).
 *
 * Why this exists
 * ───────────────
 * The "flat" exhaustive in exhaustive_plan.h is a wholesale port of
 * production's `stride_exhaustive_search`. Its bench harness is the
 * same one DP-MEASURE uses: 3 warmups, best-of-3 trials, 1.5× quick
 * pre-screen. That's fast enough to clear a few hundred candidates
 * per cell but it's NOT a true exhaustive — bench noise + thermal
 * drift over a long run can shuffle results by 5-10%, and quick
 * pre-screen prunes candidates that might have measured below the
 * current best on a less-noisy trial.
 *
 * PATIENT mode does what its name suggests:
 *   - 5 warmups (instead of 3) to fully prime branch predictors + caches
 *   - 7 trials best-of (instead of 3) to drown bench jitter
 *   - NO pre-screen — every candidate gets a full bench
 *   - Configurable inter-candidate sleep (default 200 ms) to keep the
 *     package thermal envelope stable across a long run
 *   - Optional second-pass re-bench of top-N candidates at the end
 *     (default 5) for the final winner determination
 *
 * Wall-time cost: 3-5× slower than flat on the same enumeration count.
 * At N=16384 K=4 (873 candidates), expect ~25-35 seconds vs flat's ~7.
 * At N=131072 K=4 (887 candidates), expect ~8 min vs flat's 2.25 min.
 *
 * This is the right tool when you want to *establish* the true winner
 * for a cell, not for production planning. For production, the
 * V4-screened exhaustive in exhaustive_screened.h is the workhorse.
 */
#ifndef VFFT_PROTO_CORE_EXHAUSTIVE_PATIENT_H
#define VFFT_PROTO_CORE_EXHAUSTIVE_PATIENT_H

#include "plan.h"
#include "executor.h"
#include "planner.h"
#include "exhaustive_plan.h"     /* enumeration, perm gen, bench harness re-use */
#include "dp_planner.h"          /* now_ns + _sleep_ms */
#include "../prototype/generated/registry.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef VFFT_PROTO_PATIENT_WARMUPS
#define VFFT_PROTO_PATIENT_WARMUPS 5
#endif

#ifndef VFFT_PROTO_PATIENT_TRIALS
#define VFFT_PROTO_PATIENT_TRIALS 7
#endif

#ifndef VFFT_PROTO_PATIENT_REPS_MIN
#define VFFT_PROTO_PATIENT_REPS_MIN 20
#endif

#ifndef VFFT_PROTO_PATIENT_REPS_MAX
#define VFFT_PROTO_PATIENT_REPS_MAX 50000
#endif

/* Sleep between candidates (ms). Set 0 to disable. Production wisdom
 * generation uses ~1000 ms; for a one-off true-winner search, 200 ms
 * is a reasonable middle ground (bench harness already does intra-
 * candidate warmups, so we mostly need to dissipate package heat). */
#ifndef VFFT_PROTO_PATIENT_PACE_MS
#define VFFT_PROTO_PATIENT_PACE_MS 200
#endif

/* Number of top candidates to re-bench at the end as a tie-breaker.
 * The first-pass winner can drift across runs by 1-2%; a second pass
 * on the leading group with the same patient bench produces a more
 * stable verdict. Set 0 to disable. */
#ifndef VFFT_PROTO_PATIENT_REBENCH_TOPN
#define VFFT_PROTO_PATIENT_REBENCH_TOPN 5
#endif

/* ─────────────────────────────────────────────────────────────────
 * DEEPER BENCH — same shape as vfft_proto_bench_one but with bigger
 * warmup/trial counts and configurable rep floor.
 * ───────────────────────────────────────────────────────────────── */

static inline double vfft_proto_bench_patient(
    int N, size_t K, const int *factors, int nf,
    const vfft_proto_registry_t *reg,
    double *re, double *im, double *orig_re, double *orig_im)
{
    size_t total = (size_t)N * K;

    for (int s = 0; s < nf; s++) {
        int R = factors[s];
        if (R <= 0 || R >= VFFT_PROTO_REG_MAX_RADIX || !reg->n1_fwd[R])
            return 1e18;
    }

    stride_plan_t *plan = vfft_proto_plan_create(
        N, K, factors, /*variants=*/NULL, nf, reg);
    if (!plan) return 1e18;

    /* Deeper warmup. */
    for (int w = 0; w < VFFT_PROTO_PATIENT_WARMUPS; w++) {
        memcpy(re, orig_re, total * sizeof(double));
        memcpy(im, orig_im, total * sizeof(double));
        vfft_proto_execute_fwd(plan, re, im, K);
    }

    int reps = (int)(2e5 / (total + 1));
    if (reps < VFFT_PROTO_PATIENT_REPS_MIN) reps = VFFT_PROTO_PATIENT_REPS_MIN;
    if (reps > VFFT_PROTO_PATIENT_REPS_MAX) reps = VFFT_PROTO_PATIENT_REPS_MAX;

    /* Best of N trials. */
    double best = 1e18;
    for (int t = 0; t < VFFT_PROTO_PATIENT_TRIALS; t++) {
        memcpy(re, orig_re, total * sizeof(double));
        memcpy(im, orig_im, total * sizeof(double));
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++)
            vfft_proto_execute_fwd(plan, re, im, K);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }

    vfft_proto_plan_destroy(plan);
    return best;
}

/* ─────────────────────────────────────────────────────────────────
 * PATIENT EXHAUSTIVE — every candidate benched at full fidelity.
 *
 * Two passes:
 *   1. Patient-bench every candidate, track top-N by cost.
 *   2. Re-bench top-N at end with the same patient-bench, pick winner.
 *
 * Returns measured ns of the winner.
 * ───────────────────────────────────────────────────────────────── */

typedef struct {
    int    factors[STRIDE_MAX_STAGES];
    int    nfactors;
    double cost_ns;
} vfft_proto_patient_top_t;

static int _vfft_proto_patient_top_cmp(const void *a, const void *b) {
    double ca = ((const vfft_proto_patient_top_t *)a)->cost_ns;
    double cb = ((const vfft_proto_patient_top_t *)b)->cost_ns;
    return (ca < cb) ? -1 : (ca > cb) ? 1 : 0;
}

static inline double vfft_proto_patient_exhaustive_search(
    int N, size_t K,
    const vfft_proto_registry_t *reg,
    vfft_proto_factorization_t *best_fact,
    int verbose)
{
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

    vfft_proto_factorization_list_t *flist =
        (vfft_proto_factorization_list_t *)malloc(sizeof(*flist));
    vfft_proto_enumerate_factorizations(N, reg, flist);

    if (verbose >= 1) {
        printf("  [patient] N=%d K=%zu: %d unique decompositions\n",
               N, (size_t)K, flist->count);
        printf("  [patient] config: %d warmups, %d trials, "
               "pace=%dms, top-%d re-bench\n",
               VFFT_PROTO_PATIENT_WARMUPS, VFFT_PROTO_PATIENT_TRIALS,
               VFFT_PROTO_PATIENT_PACE_MS, VFFT_PROTO_PATIENT_REBENCH_TOPN);
    }

    /* Top-N tracker (replacement-sort). */
    enum { _TOPN_MAX = 32 };
    int topn_cap = VFFT_PROTO_PATIENT_REBENCH_TOPN;
    if (topn_cap < 1) topn_cap = 1;
    if (topn_cap > _TOPN_MAX) topn_cap = _TOPN_MAX;
    vfft_proto_patient_top_t topn[_TOPN_MAX];
    int n_top = 0;

    int total_candidates = 0;
    double global_best_ns = 1e18;

    for (int fi = 0; fi < flist->count; fi++) {
        const vfft_proto_factorization_t *f = &flist->results[fi];

        vfft_proto_perm_list_t *plist =
            (vfft_proto_perm_list_t *)malloc(sizeof(*plist));
        vfft_proto_gen_permutations(f->factors, f->nfactors, plist);

        for (int pi = 0; pi < plist->count; pi++) {
            int prod = 1;
            for (int s = 0; s < f->nfactors; s++) prod *= plist->perms[pi][s];
            if (prod != N) continue;

            const int *factors_p = plist->perms[pi];
            int nf = f->nfactors;

            double ns = vfft_proto_bench_patient(
                N, K, factors_p, nf, reg, re, im, orig_re, orig_im);
            total_candidates++;

            if (ns < global_best_ns) {
                global_best_ns = ns;
                if (verbose >= 2) {
                    printf("    [patient] new best (#%d) ", total_candidates);
                    for (int s = 0; s < nf; s++)
                        printf("%s%d", s ? "x" : "", factors_p[s]);
                    printf(" = %.1f ns\n", ns);
                }
            }

            /* Top-N insertion sort (small N, simple). */
            if (n_top < topn_cap) {
                memcpy(topn[n_top].factors, factors_p, nf * sizeof(int));
                topn[n_top].nfactors = nf;
                topn[n_top].cost_ns = ns;
                n_top++;
                if (n_top == topn_cap)
                    qsort(topn, n_top, sizeof(*topn),
                          _vfft_proto_patient_top_cmp);
            } else if (ns < topn[topn_cap - 1].cost_ns) {
                memcpy(topn[topn_cap - 1].factors, factors_p, nf * sizeof(int));
                topn[topn_cap - 1].nfactors = nf;
                topn[topn_cap - 1].cost_ns = ns;
                qsort(topn, topn_cap, sizeof(*topn),
                      _vfft_proto_patient_top_cmp);
            }

            if (VFFT_PROTO_PATIENT_PACE_MS > 0)
                _vfft_proto_dp_sleep_ms(VFFT_PROTO_PATIENT_PACE_MS);
        }
        free(plist);
    }
    free(flist);

    if (verbose >= 1) {
        printf("  [patient] first pass complete: %d candidates benched\n",
               total_candidates);
        printf("  [patient] top-%d (first pass):\n", n_top);
        for (int i = 0; i < n_top; i++) {
            printf("    rank %2d: ", i + 1);
            for (int s = 0; s < topn[i].nfactors; s++)
                printf("%s%d", s ? "x" : "", topn[i].factors[s]);
            printf(" = %.1f ns\n", topn[i].cost_ns);
        }
    }

    /* Second pass: re-bench top-N for stable winner. */
    if (n_top > 0 && VFFT_PROTO_PATIENT_REBENCH_TOPN > 0) {
        if (verbose >= 1)
            printf("  [patient] second pass: re-benching top-%d...\n", n_top);

        for (int i = 0; i < n_top; i++) {
            double ns = vfft_proto_bench_patient(
                N, K, topn[i].factors, topn[i].nfactors, reg,
                re, im, orig_re, orig_im);
            if (verbose >= 1) {
                printf("    rebench rank %2d: ", i + 1);
                for (int s = 0; s < topn[i].nfactors; s++)
                    printf("%s%d", s ? "x" : "", topn[i].factors[s]);
                printf(" first=%.1f  rebench=%.1f  delta=%.1f%%\n",
                       topn[i].cost_ns, ns,
                       (ns - topn[i].cost_ns) / topn[i].cost_ns * 100.0);
            }
            topn[i].cost_ns = ns;
            if (VFFT_PROTO_PATIENT_PACE_MS > 0)
                _vfft_proto_dp_sleep_ms(VFFT_PROTO_PATIENT_PACE_MS);
        }
        qsort(topn, n_top, sizeof(*topn), _vfft_proto_patient_top_cmp);
        global_best_ns = topn[0].cost_ns;
    }

    best_fact->nfactors = 0;
    if (n_top > 0) {
        best_fact->nfactors = topn[0].nfactors;
        memcpy(best_fact->factors, topn[0].factors,
               (size_t)topn[0].nfactors * sizeof(int));
    }

    if (verbose >= 1) {
        printf("  [patient] Winner: ");
        for (int s = 0; s < best_fact->nfactors; s++)
            printf("%s%d", s ? "x" : "", best_fact->factors[s]);
        printf(" = %.1f ns (%d total candidates)\n",
               global_best_ns, total_candidates);
    }

    vfft_proto_aligned_free(re);
    vfft_proto_aligned_free(im);
    vfft_proto_aligned_free(orig_re);
    vfft_proto_aligned_free(orig_im);
    return global_best_ns;
}

static inline stride_plan_t *vfft_proto_patient_exhaustive_plan_verbose(
    int N, size_t K, const vfft_proto_registry_t *reg,
    int *out_factors, int *out_nf, double *out_ns,
    int verbose)
{
    vfft_proto_factorization_t best;
    double ns = vfft_proto_patient_exhaustive_search(N, K, reg, &best, verbose);
    if (out_ns) *out_ns = ns;
    if (best.nfactors == 0) {
        if (out_nf) *out_nf = 0;
        return NULL;
    }
    if (out_factors) memcpy(out_factors, best.factors,
                            (size_t)best.nfactors * sizeof(int));
    if (out_nf) *out_nf = best.nfactors;
    return vfft_proto_plan_create(N, K, best.factors, /*variants=*/NULL,
                                  best.nfactors, reg);
}

#endif /* VFFT_PROTO_CORE_EXHAUSTIVE_PATIENT_H */
