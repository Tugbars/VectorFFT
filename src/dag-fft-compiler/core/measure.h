#ifndef VFFT_PROTO_MEASURE_H
#define VFFT_PROTO_MEASURE_H
/* measure.h — VFFT_MEASURE: top-K + variant-aware (DIT/DIF) planner.
 *
 * TRANSFERRED from production src/core/dp_planner.h (`stride_dp_plan_measure`
 * + helpers — the "MEASURE wrapper" that dag's dp_planner.h port at lines
 * 1-598 explicitly SKIPPED: "MEASURE wrapper skipped — separate variant-
 * cartesian workstream"). That gap is why dag's old patient/exhaustive
 * calibrator searched only the FACTORIZATION axis (T1S/DIT), producing
 * all-T1S/DIT wisdom that lost to production's variant+DIF-tuned plans even
 * with identical codelets on a healthy box.
 *
 * Two-pass method:
 *   COARSE : enumerate (factorization x permutation); bench with default
 *            (T1S) variants at DIT (+ a LOG3-forced probe so LOG3-friendly
 *            multisets are not eliminated before refine); sort; take top-K.
 *   REFINE : on each top-K (factors, ordering), run the variant cartesian
 *            (FLAT/LOG3/T1S, 3^(nf-1)) x {DIT, DIF}; track the global best
 *            and a top-K-within-threshold pool for the caller to deploy-bench.
 *
 * PORT DECISIONS (with Tugbars, 2026-06-16):
 *   A. variant enumeration — production's registry-driven vfft_variant_iter_*
 *      is replaced by dag's existing variant_count/variant_decode (3^(nf-1))
 *      plus a small vfft_proto_variant_available() that reads dag's GENERATED
 *      registry slots (the OCaml pipeline emits registry_{avx2,avx512}.h; this
 *      header only READS those slots, never edits them). Equivalent in result.
 *   B. measurement — the per-bench FFTW-style adaptive timer is ported
 *      verbatim (it lives in _vfft_proto_dp_bench already); since dag runs on
 *      an UNCAPPED-TURBO box (not clock-locked), a ONE-TIME sustained warmup
 *      is added at the top of dp_plan_measure to reach the steady clock before
 *      any timing (a cold cell catches burst turbo, a hot one throttles).
 *   C. LOG3-aware coarse probe (production "Upgrade F") — PORTED; high-leverage
 *      for prime radixes (5/7/11/13 Winograd +/-30%, no analytic predictor).
 *   D. deploy split — core proposes top-K-within-threshold; the calibrator
 *      deploy-rebenches with the clean protocol and picks the fastest.
 *   - BUF variant: OBSOLETE in dag (no codelet generated) — search is
 *     FLAT/LOG3/T1S only (dag's VARIANT_CHOICES already excludes BUF).
 *
 * Placement: production kept MEASURE in dp_planner.h, but dag's include graph
 * is exhaustive_plan.h -> dp_planner.h (one-way), so MEASURE (needing both)
 * lives here, one level up, including exhaustive_plan.h.
 */
#include "exhaustive_plan.h"   /* enumerate_factorizations, variant_count/decode,
                                * factorization_list_t, + dp_planner.h transitively
                                * (vfft_proto_dp_context_t, _vfft_proto_dp_bench,
                                * _dp_solve_topk, gen_permutations, now_ns,
                                * VFFT_PROTO_DP_TIME_* , maybe_pace) */
#include <math.h>

/* ─────────────────────────────────────────────────────────────────
 * Step 1: tunables (production MEASURE_* values, VFFT_PROTO_ prefixed)
 * ───────────────────────────────────────────────────────────────── */
#ifndef VFFT_PROTO_MEASURE_TOPK
#define VFFT_PROTO_MEASURE_TOPK 5        /* top-K coarse candidates to refine */
#endif
#ifndef VFFT_PROTO_MEASURE_MAX_CANDIDATES
#define VFFT_PROTO_MEASURE_MAX_CANDIDATES 1024
#endif
#ifndef VFFT_PROTO_MEASURE_COARSE_RUNS
#define VFFT_PROTO_MEASURE_COARSE_RUNS 2 /* coarse sweeps; per-candidate min */
#endif
#ifndef VFFT_PROTO_MEASURE_REFINE_RUNS
#define VFFT_PROTO_MEASURE_REFINE_RUNS 4 /* refine sweeps; per-variant min — decorrelates
                                          * noise for low-K prime-power cells near the floor */
#endif
#ifndef VFFT_PROTO_MEASURE_DEPLOY_PCT
#define VFFT_PROTO_MEASURE_DEPLOY_PCT 10.0 /* candidates within X% of refine-best -> deploy pool */
#endif
#ifndef VFFT_PROTO_MEASURE_DEPLOY_MAX
#define VFFT_PROTO_MEASURE_DEPLOY_MAX 5    /* cap on deploy pool size */
#endif
#ifndef VFFT_PROTO_MEASURE_EXH_THRESHOLD
/* Above this pow2 N: DP top-K coarse (top-DP_TOPK_MAX multisets only); at/below:
 * EXHAUSTIVE coarse (full multiset coverage). Raised from production's 2048 to
 * 16384 (2026-06-16, Tugbars): parity reached, now widen coverage — the DP path
 * missed [4,4,8,8,8] for 8192 (faster than the [4,4,4,4,32] it+production pick).
 * MEASURE-only knob: does NOT affect the runtime DP planner's cost. */
#define VFFT_PROTO_MEASURE_EXH_THRESHOLD 16384
#endif
#ifndef VFFT_PROTO_MEASURE_DP_TOPK_MULTISETS
#define VFFT_PROTO_MEASURE_DP_TOPK_MULTISETS 3 /* outer-frame top-K multisets for DP coarse */
#endif

/* ─────────────────────────────────────────────────────────────────
 * Step 2: structs + comparators
 * ───────────────────────────────────────────────────────────────── */
typedef struct {
    int    factors[STRIDE_MAX_STAGES];
    int    variants[STRIDE_MAX_STAGES];
    int    nf;
    int    use_dif_forward;
    double cost_ns;
} vfft_proto_plan_decision_t;

/* Coarse-pass candidate, sorted by default-variant cost. */
typedef struct {
    int    factors[STRIDE_MAX_STAGES];
    int    nf;
    double cost_ns;
} _vfft_proto_measure_cand_t;

static int _vfft_proto_measure_cmp(const void *a, const void *b) {
    double ca = ((const _vfft_proto_measure_cand_t *)a)->cost_ns;
    double cb = ((const _vfft_proto_measure_cand_t *)b)->cost_ns;
    return (ca < cb) ? -1 : (ca > cb);
}
static int _vfft_proto_decision_cmp(const void *a, const void *b) {
    double ca = ((const vfft_proto_plan_decision_t *)a)->cost_ns;
    double cb = ((const vfft_proto_plan_decision_t *)b)->cost_ns;
    return (ca < cb) ? -1 : (ca > cb);
}

/* ─────────────────────────────────────────────────────────────────
 * Step 3: bench one fully-explicit (factors, variants, DIT/DIF) plan.
 *
 * Ported from production _dp_bench_explicit_one: builds via plan_create_ex
 * (= production's _stride_build_plan_explicit) and runs the same FFTW-style
 * adaptive timer as _vfft_proto_dp_bench. Returns ns/iter, 1e18 on failure.
 * Per-stage variant availability is enforced implicitly — plan_create_ex
 * returns NULL when a requested (variant, orientation) codelet is absent.
 * ───────────────────────────────────────────────────────────────── */
static double _vfft_proto_dp_bench_explicit(vfft_proto_dp_context_t *ctx, int N,
        const int *factors, int nf, const int *variants, int use_dif_forward,
        size_t K_eff, const vfft_proto_registry_t *reg)
{
    size_t total = (size_t)N * K_eff;
    for (int s = 0; s < nf; s++)
        if (!reg->n1_fwd[factors[s]]) return 1e18;

    stride_plan_t *plan = vfft_proto_plan_create_ex(
        N, K_eff, factors, variants, nf, use_dif_forward, reg);
    if (!plan) return 1e18;

    /* Warmup / single-iter calibration baseline. */
    memcpy(ctx->re, ctx->orig_re, total * sizeof(double));
    memcpy(ctx->im, ctx->orig_im, total * sizeof(double));
    vfft_proto_execute_fwd(plan, ctx->re, ctx->im, K_eff);

    double best = 1e30, total_elapsed = 0.0;
    int reps = 1, calibrated = 0;
    for (int outer = 0; outer < 32 && total_elapsed < VFFT_PROTO_DP_TIME_LIMIT_NS; outer++) {
        double tmin = 1e30;
        for (int t = 0; t < VFFT_PROTO_DP_TIME_REPEAT; t++) {
            memcpy(ctx->re, ctx->orig_re, total * sizeof(double));
            memcpy(ctx->im, ctx->orig_im, total * sizeof(double));
            double t0 = vfft_proto_now_ns();
            for (int i = 0; i < reps; i++)
                vfft_proto_execute_fwd(plan, ctx->re, ctx->im, K_eff);
            double tn = vfft_proto_now_ns() - t0;
            if (tn < tmin) tmin = tn;
            total_elapsed += tn;
            if (total_elapsed >= VFFT_PROTO_DP_TIME_LIMIT_NS) break;
        }
        if (!calibrated) {
            if (tmin < VFFT_PROTO_DP_TIME_MIN_NS) {
                reps *= 2;
                if (reps > (1 << 24)) calibrated = 1;
                continue;
            }
            calibrated = 1;
        }
        double per = tmin / (double)reps;
        if (per < best) best = per;
        break;
    }

    vfft_proto_plan_destroy(plan);
    ctx->n_benchmarks++;
    _vfft_proto_dp_maybe_pace(ctx, total);
    return best;
}

/* ─────────────────────────────────────────────────────────────────
 * Step 4: variant layer — availability, cartesian iterator, refine search.
 *
 * Ported from production registry.h (vfft_stage_variants, vfft_variant_iter_*)
 * + dp_planner.h (_dp_variant_search). Decision A: registry-driven, reading
 * dag's GENERATED slots (never editing them). The slot selection MIRRORS dag's
 * wire_stage_codelets exactly, so availability agrees with what plan_create_ex
 * actually builds. DIF lists only {FLAT, LOG3} (T1S aliases FLAT in DIF → the
 * de-dup); BUF is obsolete (no dag codelet). Variant codes reused from planner.h.
 * ───────────────────────────────────────────────────────────────── */
#ifndef VFFT_PROTO_VARIANT_COUNT
#define VFFT_PROTO_VARIANT_COUNT 4   /* code space FLAT/LOG3/T1S/BUF; BUF unused in dag */
#endif

/* Variants registered for radix R in this orientation, into out[]; count.
 * (== production vfft_stage_variants, dag slot names.) */
static inline int vfft_proto_stage_variants(const vfft_proto_registry_t *reg,
        int R, int use_dif_forward, int *out) {
    int n = 0;
    if (R <= 0 || R >= VFFT_PROTO_REG_MAX_RADIX) return 0;
    if (use_dif_forward) {
        if (reg->t1_dif_fwd[R])      out[n++] = VFFT_PROTO_VARIANT_FLAT;
        if (reg->t1_dif_log3_fwd[R]) out[n++] = VFFT_PROTO_VARIANT_LOG3;
    } else {
        if (reg->t1_dit_fwd[R])      out[n++] = VFFT_PROTO_VARIANT_FLAT;
        if (reg->t1_dit_log3_fwd[R]) out[n++] = VFFT_PROTO_VARIANT_LOG3;
        if (reg->t1s_dit_fwd[R])     out[n++] = VFFT_PROTO_VARIANT_T1S;
    }
    return n;
}

/* Is variant v buildable on radix R in this orientation? Mirrors
 * wire_stage_codelets' return condition (incl. DIT T1S→FLAT fallback).
 * Used by the LOG3-aware coarse probe (step 6, decision C). */
static inline int vfft_proto_variant_available(const vfft_proto_registry_t *reg,
        int R, int use_dif_forward, int v) {
    if (R <= 0 || R >= VFFT_PROTO_REG_MAX_RADIX) return 0;
    if (use_dif_forward) {
        if (v == VFFT_PROTO_VARIANT_LOG3) return reg->t1_dif_log3_fwd[R] != NULL;
        return reg->t1_dif_fwd[R] != NULL;            /* FLAT (T1S aliases FLAT) */
    }
    switch (v) {
        case VFFT_PROTO_VARIANT_LOG3: return reg->t1_dit_log3_fwd[R] != NULL;
        case VFFT_PROTO_VARIANT_FLAT: return reg->t1_dit_fwd[R] != NULL;
        default: return reg->t1s_dit_fwd[R] != NULL   /* T1S, with FLAT fallback */
                     || reg->t1_dit_fwd[R] != NULL;
    }
}

/* Cartesian iterator over per-stage variant choices (ported verbatim from
 * production). Stage 0 fixed to FLAT (no-twiddle); others enumerate
 * stage_variants. Lex order, last stage fastest. */
typedef struct {
    int nf;
    int counts [STRIDE_MAX_STAGES];
    int options[STRIDE_MAX_STAGES][VFFT_PROTO_VARIANT_COUNT];
    int counter[STRIDE_MAX_STAGES];
    int done;
} vfft_proto_variant_iter_t;

static inline int vfft_proto_variant_iter_init(vfft_proto_variant_iter_t *it,
        const int *factors, int nf, int use_dif_forward,
        const vfft_proto_registry_t *reg) {
    it->nf = nf; it->done = 0;
    /* DAG ADAPTATION (not verbatim): production pins stage 0 for both
     * orientations, but dag's no-twiddle stage is stage 0 for DIT and stage
     * nf-1 for DIF (per wire_stage_codelets). Pin the orientation-correct
     * no-twiddle stage to FLAT (moot there) and search the twiddled stages. */
    int notw = use_dif_forward ? (nf - 1) : 0;
    for (int s = 0; s < nf; s++) {
        it->counter[s] = 0;
        if (s == notw) { it->counts[s] = 1; it->options[s][0] = VFFT_PROTO_VARIANT_FLAT; continue; }
        int n = vfft_proto_stage_variants(reg, factors[s], use_dif_forward, it->options[s]);
        if (n == 0) { it->done = 1; return 0; }   /* stage has no variants this orient */
        it->counts[s] = n;
    }
    return 1;
}
static inline void vfft_proto_variant_iter_get(const vfft_proto_variant_iter_t *it, int *out) {
    for (int s = 0; s < it->nf; s++) out[s] = it->options[s][it->counter[s]];
}
static inline int vfft_proto_variant_iter_next(vfft_proto_variant_iter_t *it) {
    if (it->done) return 0;
    for (int s = it->nf - 1; s >= 0; s--) {
        if (++it->counter[s] < it->counts[s]) return 1;
        it->counter[s] = 0;
    }
    it->done = 1; return 0;
}

/* Per-call refine top-K candidate (variants only; factors/orient by caller). */
typedef struct { int variants[STRIDE_MAX_STAGES]; double cost_ns; } _vfft_proto_refine_top_t;

/* Variant cartesian search at one (factors, orientation, K_eff). Benches each
 * assignment best-of-REFINE_RUNS (fresh memcpy warmups decorrelate noise — the
 * production rationale for the low-K-near-floor cells). Returns best ns +
 * out_best variants; optionally fills a sorted-ascending top-K pool for the
 * caller's deploy rebench (decision D). 1e18 if no assignment is valid. */
static double _vfft_proto_dp_variant_search(vfft_proto_dp_context_t *ctx, int N,
        const int *factors, int nf, int use_dif_forward, size_t K_eff,
        const vfft_proto_registry_t *reg, int *out_best,
        _vfft_proto_refine_top_t *top_out, int *top_count, int max_top, int verbose) {
    vfft_proto_variant_iter_t it;
    if (!vfft_proto_variant_iter_init(&it, factors, nf, use_dif_forward, reg)) {
        if (top_count) *top_count = 0;
        return 1e18;
    }
    double best_ns = 1e18; int n_top = 0;
    do {
        int cur[STRIDE_MAX_STAGES];
        vfft_proto_variant_iter_get(&it, cur);
        double ns = 1e18;
        for (int run = 0; run < VFFT_PROTO_MEASURE_REFINE_RUNS; run++) {
            double r = _vfft_proto_dp_bench_explicit(ctx, N, factors, nf, cur,
                                                     use_dif_forward, K_eff, reg);
            if (r < ns) ns = r;
        }
        if (verbose) {
            printf("    [%s] ", use_dif_forward ? "DIF" : "DIT");
            for (int s = 0; s < nf; s++) printf("%s%d", s ? "/" : "", cur[s]);
            printf(" = %.1f ns\n", ns);
        }
        if (ns < best_ns) { best_ns = ns; if (out_best) memcpy(out_best, cur, nf * sizeof(int)); }
        if (top_out && max_top > 0) {
            if (n_top < max_top) {
                int pos = n_top;
                while (pos > 0 && top_out[pos-1].cost_ns > ns) pos--;
                for (int i = n_top; i > pos; i--) top_out[i] = top_out[i-1];
                memcpy(top_out[pos].variants, cur, nf * sizeof(int));
                top_out[pos].cost_ns = ns; n_top++;
            } else if (ns < top_out[max_top-1].cost_ns) {
                int pos = max_top-1;
                while (pos > 0 && top_out[pos-1].cost_ns > ns) pos--;
                for (int i = max_top-1; i > pos; i--) top_out[i] = top_out[i-1];
                memcpy(top_out[pos].variants, cur, nf * sizeof(int));
                top_out[pos].cost_ns = ns;
            }
        }
    } while (vfft_proto_variant_iter_next(&it));
    if (top_count) *top_count = n_top;
    return best_ns;
}

/* ─────────────────────────────────────────────────────────────────
 * Step 5: DP-driven coarse collection (large pow2 path).
 *
 * Ported from production _measure_collect_via_dp: top-K multisets via dag's
 * _vfft_proto_dp_solve_topk (already present), expand each to permutations,
 * coarse-bench (default T1S/DIT) into the candidate array.
 * ───────────────────────────────────────────────────────────────── */
static int _vfft_proto_measure_collect_via_dp(vfft_proto_dp_context_t *ctx, int N,
        const vfft_proto_registry_t *reg, int K_top_multisets,
        _vfft_proto_measure_cand_t *cands_out, int max_cands) {
    if (K_top_multisets > VFFT_PROTO_DP_TOPK_MAX) K_top_multisets = VFFT_PROTO_DP_TOPK_MAX;
    vfft_proto_dp_subplan_t plans[VFFT_PROTO_DP_TOPK_MAX];
    int n_plans = _vfft_proto_dp_solve_topk(ctx, N, ctx->K, reg, plans, K_top_multisets);
    if (n_plans == 0) return 0;

    int n_cands = 0;
    for (int p = 0; p < n_plans && n_cands < max_cands; p++) {
        const int nf = plans[p].nfactors;
        vfft_proto_perm_list_t *plist = (vfft_proto_perm_list_t *)malloc(sizeof(*plist));
        vfft_proto_gen_permutations(plans[p].factors, nf, plist);
        for (int pi = 0; pi < plist->count && n_cands < max_cands; pi++) {
            const int *perm = plist->perms[pi];
            int can_build = 1;
            for (int s = 0; s < nf; s++) {
                int R = perm[s];
                if (R <= 0 || R >= VFFT_PROTO_REG_MAX_RADIX || !reg->n1_fwd[R]) { can_build = 0; break; }
            }
            if (!can_build) continue;
            double ns = _vfft_proto_dp_bench(ctx, N, perm, nf, ctx->K, reg);
            if (ns >= 1e17) continue;
            cands_out[n_cands].nf = nf;
            for (int s = 0; s < nf; s++) cands_out[n_cands].factors[s] = perm[s];
            cands_out[n_cands].cost_ns = ns;
            n_cands++;
        }
        free(plist);
    }
    return n_cands;
}

/* readable variant label for verbose output */
static inline const char *vfft_proto_variant_name(int v) {
    switch (v) {
        case VFFT_PROTO_VARIANT_FLAT: return "flat";
        case VFFT_PROTO_VARIANT_LOG3: return "log3";
        case VFFT_PROTO_VARIANT_T1S:  return "t1s";
        default: return "?";
    }
}

#ifndef VFFT_PROTO_MEASURE_WARMUP_NS
#define VFFT_PROTO_MEASURE_WARMUP_NS 1.0e9  /* decision B: one-time sustained warmup (unlocked-turbo box) */
#endif

/* ─────────────────────────────────────────────────────────────────
 * Step 6: top-level VFFT_MEASURE entry — the 2-pass orchestrator.
 *
 * Ported from production stride_dp_plan_measure. `decision` receives the
 * global-best plan. If top_k_out/top_k_count are non-NULL, also emits the
 * pool of candidates within DEPLOY_PCT of the refine-best (capped DEPLOY_MAX)
 * for the calibrator to deploy-rebench (decision D). Returns best ns/iter.
 * ───────────────────────────────────────────────────────────────── */
static double vfft_proto_dp_plan_measure(vfft_proto_dp_context_t *ctx, int N,
        const vfft_proto_registry_t *reg, vfft_proto_plan_decision_t *decision,
        vfft_proto_plan_decision_t *top_k_out, int *top_k_count, int verbose) {
    static _vfft_proto_measure_cand_t cands[VFFT_PROTO_MEASURE_MAX_CANDIDATES];
    int n_cands = 0;
    const char *coarse_path = NULL;

    /* Decision B: ONE-TIME sustained warmup to reach steady turbo before any
     * timing (dag runs unlocked; a cold cell would catch burst turbo). */
    {
        vfft_proto_factorization_list_t *wl = (vfft_proto_factorization_list_t *)malloc(sizeof(*wl));
        vfft_proto_enumerate_factorizations(N, reg, wl);
        stride_plan_t *wp = NULL;
        for (int fi = 0; fi < wl->count && !wp; fi++)
            wp = vfft_proto_plan_create_ex(N, ctx->K, wl->results[fi].factors, NULL,
                                           wl->results[fi].nfactors, 0, reg);
        free(wl);
        if (wp) {
            size_t total = (size_t)N * ctx->K;
            double t0 = vfft_proto_now_ns();
            while (vfft_proto_now_ns() - t0 < VFFT_PROTO_MEASURE_WARMUP_NS) {
                memcpy(ctx->re, ctx->orig_re, total * sizeof(double));
                memcpy(ctx->im, ctx->orig_im, total * sizeof(double));
                vfft_proto_execute_fwd(wp, ctx->re, ctx->im, ctx->K);
            }
            vfft_proto_plan_destroy(wp);
        }
    }

    /* ---- COARSE ---- */
    int n_is_pow2 = (N > 0) && ((N & (N - 1)) == 0);
    int use_exhaustive = (N <= VFFT_PROTO_MEASURE_EXH_THRESHOLD) || !n_is_pow2;

    if (!use_exhaustive) {
        coarse_path = "DP";
        /* PATIENT (believe=0) pulls the WIDENED beam's worth of distinct multisets
         * into the coarse pool; MEASURE keeps the narrow default. */
        int dp_multisets = ctx->believe_subplan_cost ? VFFT_PROTO_MEASURE_DP_TOPK_MULTISETS : ctx->beam;
        n_cands = _vfft_proto_measure_collect_via_dp(ctx, N, reg,
                    dp_multisets, cands, VFFT_PROTO_MEASURE_MAX_CANDIDATES);
        if (n_cands == 0) { if (verbose) printf("  N=%d MEASURE-DP: no candidates\n", N); return 1e18; }
    } else {
        coarse_path = "exh";
        vfft_proto_factorization_list_t *flist = (vfft_proto_factorization_list_t *)malloc(sizeof(*flist));
        vfft_proto_enumerate_factorizations(N, reg, flist);
        if (flist->count == 0) { free(flist); if (verbose) printf("  N=%d MEASURE: no factorizations\n", N); return 1e18; }

        for (int fi = 0; fi < flist->count && n_cands < VFFT_PROTO_MEASURE_MAX_CANDIDATES; fi++) {
            const int nf = flist->results[fi].nfactors;
            vfft_proto_perm_list_t *plist = (vfft_proto_perm_list_t *)malloc(sizeof(*plist));
            vfft_proto_gen_permutations(flist->results[fi].factors, nf, plist);
            for (int pi = 0; pi < plist->count && n_cands < VFFT_PROTO_MEASURE_MAX_CANDIDATES; pi++) {
                const int *perm = plist->perms[pi];
                int can_build = 1;
                for (int s = 0; s < nf; s++) {
                    int R = perm[s];
                    if (R <= 0 || R >= VFFT_PROTO_REG_MAX_RADIX || !reg->n1_fwd[R]) { can_build = 0; break; }
                }
                if (!can_build) continue;

                double ns = _vfft_proto_dp_bench(ctx, N, perm, nf, ctx->K, reg);
                if (ns >= 1e17) continue;

                /* Decision C: LOG3-forced coarse probe (DIT). Keeps LOG3-friendly
                 * multisets (prime radixes 5/7/11/13) alive into the refine pass. */
                {
                    int log3v[STRIDE_MAX_STAGES]; int has_log3 = 0;
                    for (int s = 0; s < nf; s++) {
                        if (s == 0) { log3v[s] = VFFT_PROTO_VARIANT_FLAT; continue; } /* DIT no-tw */
                        if (vfft_proto_variant_available(reg, perm[s], 0, VFFT_PROTO_VARIANT_LOG3)) {
                            log3v[s] = VFFT_PROTO_VARIANT_LOG3; has_log3 = 1;
                        } else if (vfft_proto_variant_available(reg, perm[s], 0, VFFT_PROTO_VARIANT_T1S)) {
                            log3v[s] = VFFT_PROTO_VARIANT_T1S;
                        } else {
                            log3v[s] = VFFT_PROTO_VARIANT_FLAT;
                        }
                    }
                    if (has_log3) {
                        double ns_l3 = _vfft_proto_dp_bench_explicit(ctx, N, perm, nf, log3v, 0, ctx->K, reg);
                        if (ns_l3 < ns) ns = ns_l3;
                    }
                }

                cands[n_cands].nf = nf;
                for (int s = 0; s < nf; s++) cands[n_cands].factors[s] = perm[s];
                cands[n_cands].cost_ns = ns;
                n_cands++;
            }
            free(plist);
        }
        free(flist);
        if (n_cands == 0) { if (verbose) printf("  N=%d MEASURE-exh: no plans\n", N); return 1e18; }
    }

    /* best-of-COARSE_RUNS extra sweeps (per-candidate min). */
    for (int run = 1; run < VFFT_PROTO_MEASURE_COARSE_RUNS; run++)
        for (int i = 0; i < n_cands; i++) {
            double ns = _vfft_proto_dp_bench(ctx, N, cands[i].factors, cands[i].nf, ctx->K, reg);
            if (ns < cands[i].cost_ns) cands[i].cost_ns = ns;
        }

    qsort(cands, n_cands, sizeof(*cands), _vfft_proto_measure_cmp);
    int K_top = VFFT_PROTO_MEASURE_TOPK;
    int n_topk = (n_cands < K_top) ? n_cands : K_top;

    /* ---- REFINE: variant cartesian x {DIT, DIF} on each top-K ---- */
    double best_ns = 1e18; int best_nf = 0, best_use_dif = 0;
    int best_factors[STRIDE_MAX_STAGES], best_variants[STRIDE_MAX_STAGES];
    long total_refine = 0;
    static vfft_proto_plan_decision_t pool[VFFT_PROTO_MEASURE_TOPK * 2 * VFFT_PROTO_MEASURE_DEPLOY_MAX];
    int n_pool = 0; const int pool_max = (int)(sizeof(pool) / sizeof(pool[0]));

    for (int k = 0; k < n_topk; k++) {
        const _vfft_proto_measure_cand_t *c = &cands[k];
        for (int orient = 0; orient < 2; orient++) {
            int cur_best[STRIDE_MAX_STAGES];
            _vfft_proto_refine_top_t per_call[VFFT_PROTO_MEASURE_DEPLOY_MAX];
            int per_call_n = 0; (void)per_call_n;
            double ns = _vfft_proto_dp_variant_search(ctx, N, c->factors, c->nf, orient, ctx->K, reg,
                            cur_best,
                            (top_k_out ? per_call : NULL),
                            (top_k_out ? &per_call_n : NULL),
                            (top_k_out ? VFFT_PROTO_MEASURE_DEPLOY_MAX : 0), 0);
            total_refine++;
            if (top_k_out) {
                for (int i = 0; i < per_call_n && n_pool < pool_max; i++) {
                    vfft_proto_plan_decision_t *p = &pool[n_pool++];
                    p->nf = c->nf;
                    memcpy(p->factors, c->factors, c->nf * sizeof(int));
                    memcpy(p->variants, per_call[i].variants, c->nf * sizeof(int));
                    p->use_dif_forward = orient;
                    p->cost_ns = per_call[i].cost_ns;
                }
            }
            if (ns < best_ns) {
                best_ns = ns; best_nf = c->nf; best_use_dif = orient;
                memcpy(best_factors, c->factors, c->nf * sizeof(int));
                memcpy(best_variants, cur_best, c->nf * sizeof(int));
            }
        }
    }

    /* threshold-filter pool -> top_k_out (decision D) */
    if (top_k_out && top_k_count) {
        if (n_pool > 0) {
            qsort(pool, n_pool, sizeof(*pool), _vfft_proto_decision_cmp);
            double thresh = pool[0].cost_ns * (1.0 + VFFT_PROTO_MEASURE_DEPLOY_PCT / 100.0);
            int n_emit = 0;
            for (int i = 0; i < n_pool && n_emit < VFFT_PROTO_MEASURE_DEPLOY_MAX; i++) {
                if (pool[i].cost_ns > thresh) break;
                top_k_out[n_emit++] = pool[i];
            }
            *top_k_count = n_emit;
        } else *top_k_count = 0;
    }

    if (best_ns >= 1e17) { if (verbose) printf("  N=%d MEASURE: no valid refine\n", N); return 1e18; }

    decision->nf = best_nf; decision->use_dif_forward = best_use_dif; decision->cost_ns = best_ns;
    memcpy(decision->factors, best_factors, best_nf * sizeof(int));
    memcpy(decision->variants, best_variants, best_nf * sizeof(int));

    if (verbose) {
        printf("  N=%d K=%zu MEASURE-topk(%d) [%s]: coarse=%d top=%d refine=%ld -> ",
               N, ctx->K, K_top, coarse_path ? coarse_path : "?", n_cands, n_topk, total_refine);
        for (int s = 0; s < best_nf; s++) printf("%s%d", s ? "x" : "", best_factors[s]);
        printf(" %s ", best_use_dif ? "DIF" : "DIT");
        for (int s = 0; s < best_nf; s++) printf("%s%s", s ? "/" : "", vfft_proto_variant_name(best_variants[s]));
        printf(" = %.1f ns\n", best_ns);
    }
    return best_ns;
}

#endif /* VFFT_PROTO_MEASURE_H */
