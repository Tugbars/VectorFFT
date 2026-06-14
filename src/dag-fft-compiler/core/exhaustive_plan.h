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
    64, 32, 25, 20, 19, 17, 16, 13, 12, 11, 10, 8, 7, 6, 5, 4, 3, 2, 0
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
    /* Research override: VFFT_PROTO_EXH_MAX_DEPTH lifts the stage cap (set =16
     * for absolutely-exhaustive depth on small N). Default unchanged
     * (5 pow2 / 9 non-pow2). Clamped to STRIDE_MAX_STAGES. */
    { const char *e = getenv("VFFT_PROTO_EXH_MAX_DEPTH");
      if (e) { int d = atoi(e); if (d > 0) max_depth = d; } }
    if (max_depth > STRIDE_MAX_STAGES) max_depth = STRIDE_MAX_STAGES;
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

/* ── Variant axis (joint search) ──────────────────────────────────────
 * The per-stage codelet variant is the THIRD search axis (after multiset and
 * ordering). Inner stages (1..nf-1) each choose FLAT / LOG3 / T1S; stage 0 is
 * the no-twiddle stage so its variant is moot. BUF is omitted (it falls back
 * to T1S in plan_create for DIT). The assignment space is 3^(nf-1). */
static const int VFFT_PROTO_VARIANT_CHOICES[3] = {
    VFFT_PROTO_VARIANT_FLAT, VFFT_PROTO_VARIANT_LOG3, VFFT_PROTO_VARIANT_T1S
};

static inline int vfft_proto_variant_count(int nf) {
    int c = 1;
    for (int s = 1; s < nf; s++) c *= 3;   /* 3^(nf-1) */
    return c;
}

/* Decode index in [0, 3^(nf-1)) into a per-stage variant array (stage 0 = T1S,
 * moot; inner stages mixed-radix base 3 over {FLAT,LOG3,T1S}). */
static inline void vfft_proto_variant_decode(int idx, int nf, int *v) {
    v[0] = VFFT_PROTO_VARIANT_T1S;
    for (int s = 1; s < nf; s++) {
        v[s] = VFFT_PROTO_VARIANT_CHOICES[idx % 3];
        idx /= 3;
    }
}

/* Variant-aware single-candidate bench. variants==NULL → plan_create's T1S
 * default (so vfft_proto_bench_one below is just this with NULL). */
static inline double vfft_proto_bench_one_v(
    int N, size_t K, const int *factors, const int *variants, int nf,
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

    stride_plan_t *plan = vfft_proto_plan_create(
        N, K, factors, variants, nf, reg);
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

/* Default-variant (T1S) bench — unchanged interface for callers that don't
 * search the variant axis (e.g. exhaustive_screened, quick pre-screens). */
static inline double vfft_proto_bench_one(
    int N, size_t K, const int *factors, int nf,
    const vfft_proto_registry_t *reg,
    double *re, double *im, double *orig_re, double *orig_im)
{
    return vfft_proto_bench_one_v(N, K, factors, /*variants=*/NULL, nf, reg,
                                  re, im, orig_re, orig_im);
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

    /* Variant pre-screen factor: skip a factorization's variant cartesian when
     * its default-variant bench exceeds prune_factor× the global best. Research
     * override VFFT_PROTO_EXH_PRUNE (default 2.0; set huge to disable). */
    double prune_factor = 2.0;
    { const char *e = getenv("VFFT_PROTO_EXH_PRUNE");
      if (e) { double p = atof(e); if (p > 0) prune_factor = p; } }

    /* Research: best ns seen per stage-count, to answer "does deeper help?". */
    double best_by_nf[STRIDE_MAX_STAGES];
    for (int i = 0; i < STRIDE_MAX_STAGES; i++) best_by_nf[i] = 1e18;

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

            /* Factorization pre-screen with the default (T1S) variant: if even
             * that is > 2× the current global best, the variant cartesian for
             * this ordering is very unlikely to win, so skip it. (2× not 1.5×
             * since a different variant assignment could still help somewhat.) */
            double screen_ns = vfft_proto_bench_one_v(
                N, K, factors_p, /*variants=*/NULL, nf, reg,
                re, im, orig_re, orig_im);
            total_candidates++;
            if (nf < STRIDE_MAX_STAGES && screen_ns < best_by_nf[nf])
                best_by_nf[nf] = screen_ns;
            if (screen_ns > global_best_ns * prune_factor && global_best_ns < 1e17)
                continue;

            /* Variant cartesian: every per-stage codelet assignment
             * (3^(nf-1)) on this factorization+ordering, full-benched. */
            int vcount = vfft_proto_variant_count(nf);
            for (int vi = 0; vi < vcount; vi++) {
                int v[STRIDE_MAX_STAGES];
                vfft_proto_variant_decode(vi, nf, v);

                double ns = vfft_proto_bench_one_v(
                    N, K, factors_p, v, nf, reg, re, im, orig_re, orig_im);
                total_candidates++;
                if (nf < STRIDE_MAX_STAGES && ns < best_by_nf[nf])
                    best_by_nf[nf] = ns;

                if (ns < global_best_ns) {
                    global_best_ns = ns;
                    best_fact->nfactors = nf;
                    memcpy(best_fact->factors,  factors_p, nf * sizeof(int));
                    memcpy(best_fact->variants, v,         nf * sizeof(int));
                    if (verbose >= 1) {
                        printf("    new best ");
                        for (int s = 0; s < nf; s++)
                            printf("%s%d", s ? "x" : "", factors_p[s]);
                        /* stage 0 is the no-twiddle (n1) stage; its variant
                         * code is moot, so show it as n1 not a variant. */
                        printf(" v=[n1");
                        for (int s = 1; s < nf; s++)
                            printf(",%d", v[s]);
                        printf("] = %.1f ns\n", ns);
                    }
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
        printf("  best by stage-count:");
        for (int k = 1; k < STRIDE_MAX_STAGES; k++)
            if (best_by_nf[k] < 1e17)
                printf("  %dst=%.0f", k, best_by_nf[k]);
        printf("\n");
    }

    free(flist);
    vfft_proto_aligned_free(re);
    vfft_proto_aligned_free(im);
    vfft_proto_aligned_free(orig_re);
    vfft_proto_aligned_free(orig_im);

    return global_best_ns;
}

/* Top-level convenience: returns a plan built from exhaustive's pick
 * (factorization + joint-searched per-stage variants). */
static inline stride_plan_t *vfft_proto_exhaustive_plan(
    int N, size_t K, const vfft_proto_registry_t *reg, int verbose)
{
    vfft_proto_factorization_t best;
    double ns = vfft_proto_exhaustive_search(N, K, reg, &best, verbose);
    (void)ns;
    if (best.nfactors == 0) return NULL;
    return vfft_proto_plan_create(N, K, best.factors, best.variants,
                                  best.nfactors, reg);
}

/* Verbose variant that also reports picked factorization + measured ns. The
 * returned plan uses the joint-searched per-stage variants. */
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
    return vfft_proto_plan_create(N, K, best.factors, best.variants,
                                  best.nfactors, reg);
}

/* Joint accessor: like _verbose but ALSO outputs the per-stage variant codes
 * (0=FLAT 1=LOG3 2=T1S) so a calibrator can write them into the wisdom file
 * (vfft_proto_wisdom_entry_t.variants → vfft_proto_wisdom_save). Any out_*
 * pointer may be NULL. */
static inline stride_plan_t *vfft_proto_exhaustive_plan_joint(
    int N, size_t K, const vfft_proto_registry_t *reg,
    int *out_factors, int *out_variants, int *out_nf, double *out_ns,
    int verbose)
{
    vfft_proto_factorization_t best;
    double ns = vfft_proto_exhaustive_search(N, K, reg, &best, verbose);
    if (out_ns) *out_ns = ns;
    if (best.nfactors == 0) { if (out_nf) *out_nf = 0; return NULL; }
    if (out_factors)  memcpy(out_factors,  best.factors,
                             (size_t)best.nfactors * sizeof(int));
    if (out_variants) memcpy(out_variants, best.variants,
                             (size_t)best.nfactors * sizeof(int));
    if (out_nf)       *out_nf = best.nfactors;
    return vfft_proto_plan_create(N, K, best.factors, best.variants,
                                  best.nfactors, reg);
}

#endif /* VFFT_PROTO_CORE_EXHAUSTIVE_PLAN_H */
