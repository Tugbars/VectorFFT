/* exhaustive_plan.h — exhaustive measurement-based planner for prototype-core.
 *
 * Ported wholesale from src/core/exhaustive.h (production), with
 * mechanical renames + adaptation to prototype-core's API surface:
 *
 *   stride_registry_t        →   vfft_proto_registry_t
 *   stride_factorization_t   →   vfft_proto_factorization_t
 *   STRIDE_AVAILABLE_RADIXES →   VFFT_PROTO_DP_RADIXES (from dp_planner.h)
 *   stride_registry_has      →   reg->n1_fwd[R] != NULL check
 *   stride_enumerate_factor. →   vfft_proto_enumerate_factorizations
 *   stride_gen_permutations  →   vfft_proto_gen_permutations (in dp_planner.h)
 *   _stride_build_plan       →   vfft_proto_plan_create(...,NULL,...)  (variants=T1S defaults)
 *   stride_execute_fwd       →   vfft_proto_execute_fwd(plan, re, im, K)  (extra K arg)
 *   stride_plan_destroy      →   vfft_proto_plan_destroy
 *   STRIDE_ALIGNED_ALLOC/FREE→   vfft_proto_posix_memalign / vfft_proto_aligned_free
 *
 * Stripped:
 *   - Bluestein/Rader fallback (override paths). Prototype-core has no
 *     override hooks; non-factorizable N just returns NULL.
 *   - The "compare strategies" reporting wrapper. Not core to search.
 *
 * The bench harness (warmup → adaptive reps → best-of-3 trials with
 * fresh memcpy each trial) is copied verbatim from production. It's the
 * production-validated methodology — we'd been re-inventing this and
 * losing details. Now it's the same code path.
 *
 * VARIANT AXIS: this exhaustive uses variants=NULL → T1S defaults per
 * stage. Production's exhaustive ALSO defaults variants via wisdom_bridge
 * predicates at plan-build time. Variant-cartesian search is layered on
 * top in production's VFFT_MEASURE wrapper; we don't have that yet.
 *
 * DIF axis: production runs DIT pass + DIF pass separately, takes min.
 * Prototype-core is DIT-only — we cover half the production search space.
 */
#ifndef VFFT_PROTO_CORE_EXHAUSTIVE_PLAN_H
#define VFFT_PROTO_CORE_EXHAUSTIVE_PLAN_H

#include "plan.h"
#include "executor.h"
#include "planner.h"
#include "dp_planner.h"     /* re-use vfft_proto_now_ns + perm gen + factorization_t */
#include "../prototype/generated/registry.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─────────────────────────────────────────────────────────────────
 * AVAILABLE RADIXES for enumeration. Mirrors STRIDE_AVAILABLE_RADIXES
 * in production. Largest-first so quick-pre-screen (below) lands on
 * a tight current_best fast.
 *
 * Cap at R=512 — R=1024 codelets are stubbed (no-op) in prototype's
 * build (see scripts/build_demo_*.sh). Including R=1024 in the search
 * would pick the stub at "5 ns" and win meaninglessly. R≤512 covers
 * every real codelet we ship.
 * ───────────────────────────────────────────────────────────────── */
static const int VFFT_PROTO_AVAILABLE_RADIXES_EXH[] = {
    512, 256, 128,
    64, 32, 25, 20, 16, 12, 10, 8, 7, 6, 5, 4, 3, 2,
    19, 17, 13, 11, 0
};

/* ─────────────────────────────────────────────────────────────────
 * ENUMERATE ALL VALID FACTORIZATIONS (multisets)
 *
 * Recursive: at each level, try every available radix that divides
 * remaining, recurse on remaining/R. Collect all complete multisets
 * (sorted non-increasing — duplicates avoided here, permutations are
 * generated separately).
 * ───────────────────────────────────────────────────────────────── */

#define VFFT_PROTO_EXH_MAX_RESULTS 512

typedef struct {
    vfft_proto_factorization_t results[VFFT_PROTO_EXH_MAX_RESULTS];
    int count;
} vfft_proto_factorization_list_t;

/* Two-bucket depth caps mirror production. Pow2 N has shallow optimal
 * decompositions (most pow2 wisdom is 4-5 stages), so 5 keeps search
 * tractable. Non-pow2 N with many small primes (e.g., 6615=3^3*5*7^2)
 * needs deeper search; 9 matches STRIDE_MAX_STAGES bound. */
#define VFFT_PROTO_EXH_MAX_DEPTH_POW2    5
#define VFFT_PROTO_EXH_MAX_DEPTH_NONPOW2 9

static inline void _vfft_proto_enumerate_factorizations(
    int remaining, const vfft_proto_registry_t *reg,
    int *current, int depth, int max_depth,
    vfft_proto_factorization_list_t *list)
{
    if (remaining == 1) {
        if (depth > 0 && list->count < VFFT_PROTO_EXH_MAX_RESULTS) {
            vfft_proto_factorization_t *f = &list->results[list->count];
            f->nfactors = depth;
            memcpy(f->factors, current, depth * sizeof(int));
            list->count++;
        }
        return;
    }
    if (depth >= max_depth) return;

    for (const int *rp = VFFT_PROTO_AVAILABLE_RADIXES_EXH; *rp; rp++) {
        int R = *rp;
        if (remaining % R != 0) continue;
        if (R >= VFFT_PROTO_REG_MAX_RADIX || !reg->n1_fwd[R]) continue;

        /* Avoid duplicate decompositions: only allow non-increasing order
         * in the base decomposition. Permutations are handled separately. */
        if (depth > 0 && R > current[depth - 1]) continue;

        current[depth] = R;
        _vfft_proto_enumerate_factorizations(
            remaining / R, reg, current, depth + 1, max_depth, list);
    }
}

static inline void vfft_proto_enumerate_factorizations(
    int N, const vfft_proto_registry_t *reg,
    vfft_proto_factorization_list_t *list)
{
    list->count = 0;
    int current[STRIDE_MAX_STAGES];
    int n_is_pow2 = (N > 0) && ((N & (N - 1)) == 0);
    int max_depth = n_is_pow2 ? VFFT_PROTO_EXH_MAX_DEPTH_POW2
                              : VFFT_PROTO_EXH_MAX_DEPTH_NONPOW2;
    _vfft_proto_enumerate_factorizations(N, reg, current, 0, max_depth, list);
}

/* Permutation generation lives in dp_planner.h as vfft_proto_gen_permutations.
 * Already a wholesale match of production's stride_gen_permutations. */

/* ─────────────────────────────────────────────────────────────────
 * BENCHMARK A SINGLE FACTORIZATION
 *
 * Verbatim port of production's stride_bench_one. Same warmup count
 * (3), same adaptive reps (cap 10..50000), same best-of-3 trials with
 * memcpy refresh each trial.
 *
 * Caller provides aligned buffers (re/im are working buffers, orig_re/im
 * are source for memcpy refresh).
 * ───────────────────────────────────────────────────────────────── */

static inline double vfft_proto_bench_one(
    int N, size_t K, const int *factors, int nf,
    const vfft_proto_registry_t *reg,
    double *re, double *im, double *orig_re, double *orig_im)
{
    size_t total = (size_t)N * K;

    /* Sanity check: every radix must have n1 codelet. */
    for (int s = 0; s < nf; s++) {
        int R = factors[s];
        if (R <= 0 || R >= VFFT_PROTO_REG_MAX_RADIX || !reg->n1_fwd[R])
            return 1e18;
    }

    /* Build plan with T1S variant defaults (variants=NULL).
     * Production routes through _stride_build_plan which consults
     * wisdom_bridge for variant selection — prototype-core doesn't have
     * wisdom_bridge so we default to T1S. Variant cartesian is a separate
     * search layer (not in scope here). */
    stride_plan_t *plan = vfft_proto_plan_create(
        N, K, factors, /*variants=*/NULL, nf, reg);
    if (!plan) return 1e18;

    /* Warm up — enough to stabilize branch predictors and fill caches. */
    for (int w = 0; w < 3; w++) {
        memcpy(re, orig_re, total * sizeof(double));
        memcpy(im, orig_im, total * sizeof(double));
        vfft_proto_execute_fwd(plan, re, im, K);
    }

    int reps = (int)(2e5 / (total + 1));
    if (reps < 10)    reps = 10;
    if (reps > 50000) reps = 50000;

    /* Benchmark: best of 3 trials. */
    double best = 1e18;
    for (int t = 0; t < 3; t++) {
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
 * EXHAUSTIVE SEARCH
 *
 * Verbatim port of production's stride_exhaustive_search.
 *   1. Enumerate all valid multisets of N
 *   2. For each, generate all unique permutations
 *   3. Quick pre-screen, then full bench, track global best
 *
 * verbose: 0=silent, 1=summary
 * ───────────────────────────────────────────────────────────────── */

static inline double vfft_proto_exhaustive_search(
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
    for (size_t i = 0; i < total; i++) {
        orig_re[i] = (double)rand() / RAND_MAX - 0.5;
        orig_im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    vfft_proto_factorization_list_t *flist =
        (vfft_proto_factorization_list_t *)malloc(sizeof(*flist));
    vfft_proto_enumerate_factorizations(N, reg, flist);

    if (verbose >= 1)
        printf("  N=%d K=%zu: %d unique decompositions\n", N, (size_t)K, flist->count);

    double global_best_ns = 1e18;
    int total_candidates = 0;
    best_fact->nfactors = 0;

    for (int fi = 0; fi < flist->count; fi++) {
        const vfft_proto_factorization_t *f = &flist->results[fi];

        vfft_proto_perm_list_t *plist =
            (vfft_proto_perm_list_t *)malloc(sizeof(*plist));
        vfft_proto_gen_permutations(f->factors, f->nfactors, plist);

        for (int pi = 0; pi < plist->count; pi++) {
            /* Verify product. */
            int prod = 1;
            for (int s = 0; s < f->nfactors; s++)
                prod *= plist->perms[pi][s];
            if (prod != N) continue;

            const int *factors_p = plist->perms[pi];
            int nf = f->nfactors;

            /* Quick single-trial pre-screen: skip if > 1.5× current best. */
            double quick_ns = vfft_proto_bench_one(
                N, K, factors_p, nf, reg, re, im, orig_re, orig_im);
            total_candidates++;

            if (quick_ns > global_best_ns * 1.5 && global_best_ns < 1e17)
                continue;

            /* Full bench (re-run with fresh data). */
            double ns = vfft_proto_bench_one(
                N, K, factors_p, nf, reg, re, im, orig_re, orig_im);

            if (ns < global_best_ns) {
                global_best_ns = ns;
                best_fact->nfactors = nf;
                memcpy(best_fact->factors, factors_p, nf * sizeof(int));
                if (verbose >= 1) {
                    printf("    new best ");
                    for (int s = 0; s < nf; s++)
                        printf("%s%d", s ? "x" : "", factors_p[s]);
                    printf(" = %.1f ns\n", ns);
                }
            }
        }
        free(plist);
    }

    if (verbose >= 1) {
        printf("  Best: ");
        for (int s = 0; s < best_fact->nfactors; s++)
            printf("%s%d", s ? "x" : "", best_fact->factors[s]);
        printf(" = %.1f ns (%d total candidates)\n",
               global_best_ns, total_candidates);
    }

    free(flist);
    vfft_proto_aligned_free(re);
    vfft_proto_aligned_free(im);
    vfft_proto_aligned_free(orig_re);
    vfft_proto_aligned_free(orig_im);

    return global_best_ns;
}

/* Top-level convenience: returns a plan built from exhaustive's pick. */
static inline stride_plan_t *vfft_proto_exhaustive_plan(
    int N, size_t K, const vfft_proto_registry_t *reg, int verbose)
{
    vfft_proto_factorization_t best;
    double ns = vfft_proto_exhaustive_search(N, K, reg, &best, verbose);
    (void)ns;
    if (best.nfactors == 0) return NULL;
    return vfft_proto_plan_create(N, K, best.factors, /*variants=*/NULL,
                                  best.nfactors, reg);
}

/* Verbose variant that also reports picked factorization + measured ns. */
static inline stride_plan_t *vfft_proto_exhaustive_plan_verbose(
    int N, size_t K, const vfft_proto_registry_t *reg,
    int *out_factors, int *out_nf, double *out_ns,
    int verbose)
{
    vfft_proto_factorization_t best;
    double ns = vfft_proto_exhaustive_search(N, K, reg, &best, verbose);
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

#endif /* VFFT_PROTO_CORE_EXHAUSTIVE_PLAN_H */
