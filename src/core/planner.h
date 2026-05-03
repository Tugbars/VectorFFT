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

/* Forward declarations. _stride_build_plan / _stride_build_plan_explicit
 * are defined later in this file but called from exhaustive.h and
 * dp_planner.h (included below) — those files were updated to route
 * through these builders so wisdom-driven log3/buf activation and
 * caller-driven explicit variant assignment propagate into the
 * exhaustive/DP search paths. registry.h above provides stride_registry_t
 * and stride_plan_t (via executor.h) so we can reference them directly. */
static stride_plan_t *_stride_build_plan(
    int N, size_t K,
    const int *factors, int nf,
    const stride_registry_t *reg);

static stride_plan_t *_stride_build_plan_explicit(
    int N, size_t K,
    const int *factors, int nf,
    const vfft_variant_t *variants,
    int use_dif_forward,
    const stride_registry_t *reg);

#include "factorizer.h"
#include "exhaustive.h"
#include "dp_planner.h"
#include "executor_blocked.h"
#include "rader.h" /* includes bluestein.h (shared SIMD helpers) */
#include "r2c.h"   /* real-to-complex / complex-to-real */
#include "dct.h"   /* DCT-II (built atop R2C) */
#include "dht.h"   /* DHT (built atop R2C, self-inverse) */
#include "dst.h"   /* DST-II / DST-III (built atop DCT-II/III) */
#include "dct4.h"  /* DCT-IV (built atop DCT-III + DST-III) */
#include "fft2d_r2c.h" /* 2D R2C / C2R (built atop 1D R2C + 1D C2C col) */

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
                                           double *re, double *im)
{
    if (plan->use_blocked)
    {
        _stride_execute_fwd_blocked(plan, re, im,
                                    plan->split_stage, plan->block_groups);
        return;
    }
    stride_execute_fwd(plan, re, im);
}

static inline void stride_execute_bwd_auto(const stride_plan_t *plan,
                                           double *re, double *im)
{
    if (plan->use_blocked)
    {
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

typedef struct
{
    int N;
    size_t K;
    int factors[FACT_MAX_STAGES];
    int nfactors;
    double best_ns; /* best time found */

    /* Blocked executor selection (version 3+).
     * Determined by joint calibration over (factorization × executor × split).
     * use_blocked=0 means standard sweep executor (default, backward compat). */
    int use_blocked;  /* 0 = standard, 1 = blocked */
    int split_stage;  /* first blocked stage */
    int block_groups; /* groups per block at split stage */

    /* DIT/DIF orientation selection (version 4+).
     * 0 = DIT-forward + DIF-style backward (current default)
     * 1 = DIF-forward + DIT-style backward
     * Determined by per-cell DIT-vs-DIF bench in the calibrator. Mutually
     * exclusive with use_blocked=1 in v1.1 — the DIF executor doesn't have
     * a blocked variant yet. */
    int use_dif_forward;

    /* Per-stage variant codes (version 5+). Each entry is a vfft_variant_t
     * value (FLAT/LOG3/T1S/BUF), one per stage 0..nfactors-1. Stage 0 is
     * conventionally FLAT (no twiddle codelet anyway). For other stages,
     * the calibrator's plan-level variant search picks the value that
     * wins as a whole plan in the chosen orientation.
     *
     * `has_variant_codes` is set when the wisdom entry carries explicit
     * codes (loaded from a v5 file or produced by the v1.2 calibrator);
     * it's 0 for legacy v3/v4 entries where the deploy-side build must
     * fall back to wisdom_bridge predicate consultation. */
    int has_variant_codes;
    int variant_codes[STRIDE_MAX_STAGES];
} stride_wisdom_entry_t;

typedef struct
{
    stride_wisdom_entry_t entries[WISDOM_MAX_ENTRIES];
    int count;
} stride_wisdom_t;

static void stride_wisdom_init(stride_wisdom_t *wis)
{
    wis->count = 0;
}

/* Find wisdom entry for (N, K). Returns NULL if not found. */
static const stride_wisdom_entry_t *stride_wisdom_lookup(
    const stride_wisdom_t *wis, int N, size_t K)
{
    for (int i = 0; i < wis->count; i++)
    {
        if (wis->entries[i].N == N && wis->entries[i].K == K)
            return &wis->entries[i];
    }
    return NULL;
}

/* Add or update wisdom entry — full version with blocked + DIF + per-stage
 * variant codes. The plan-level calibrator (v1.2+) calls this with
 * has_variant_codes=1 and explicit per-stage choices; legacy callers go
 * through the v4 wrapper below with has_variant_codes=0. */
static void stride_wisdom_add_v5(stride_wisdom_t *wis, int N, size_t K,
                                 const int *factors, int nf, double best_ns,
                                 int use_blocked, int split_stage,
                                 int block_groups, int use_dif_forward,
                                 int has_variant_codes,
                                 const int *variant_codes)
{
    /* Update existing */
    for (int i = 0; i < wis->count; i++)
    {
        if (wis->entries[i].N == N && wis->entries[i].K == K)
        {
            if (best_ns < wis->entries[i].best_ns)
            {
                memcpy(wis->entries[i].factors, factors, nf * sizeof(int));
                wis->entries[i].nfactors = nf;
                wis->entries[i].best_ns = best_ns;
                wis->entries[i].use_blocked = use_blocked;
                wis->entries[i].split_stage = split_stage;
                wis->entries[i].block_groups = block_groups;
                wis->entries[i].use_dif_forward = use_dif_forward;
                wis->entries[i].has_variant_codes = has_variant_codes;
                if (has_variant_codes && variant_codes)
                {
                    for (int s = 0; s < nf; s++)
                        wis->entries[i].variant_codes[s] = variant_codes[s];
                }
                else
                {
                    for (int s = 0; s < nf; s++)
                        wis->entries[i].variant_codes[s] = 0;
                }
            }
            return;
        }
    }
    /* Insert new */
    if (wis->count < WISDOM_MAX_ENTRIES)
    {
        stride_wisdom_entry_t *e = &wis->entries[wis->count++];
        e->N = N;
        e->K = K;
        memcpy(e->factors, factors, nf * sizeof(int));
        e->nfactors = nf;
        e->best_ns = best_ns;
        e->use_blocked = use_blocked;
        e->split_stage = split_stage;
        e->block_groups = block_groups;
        e->use_dif_forward = use_dif_forward;
        e->has_variant_codes = has_variant_codes;
        if (has_variant_codes && variant_codes)
        {
            for (int s = 0; s < nf; s++)
                e->variant_codes[s] = variant_codes[s];
        }
        else
        {
            for (int s = 0; s < nf; s++)
                e->variant_codes[s] = 0;
        }
    }
}

/* Legacy v4 wrapper — kept for the in-process calibrator phases that
 * still write entries before the variant search runs. Sets
 * has_variant_codes=0 so the deploy-side build falls back to wisdom-
 * predicate consultation rather than mistaking unset codes for FLAT. */
static void stride_wisdom_add_v4(stride_wisdom_t *wis, int N, size_t K,
                                 const int *factors, int nf, double best_ns,
                                 int use_blocked, int split_stage,
                                 int block_groups, int use_dif_forward)
{
    stride_wisdom_add_v5(wis, N, K, factors, nf, best_ns,
                         use_blocked, split_stage, block_groups,
                         use_dif_forward,
                         /*has_variant_codes=*/0, /*variant_codes=*/NULL);
}

/* Legacy v3 wrapper. */
static void stride_wisdom_add_full(stride_wisdom_t *wis, int N, size_t K,
                                   const int *factors, int nf, double best_ns,
                                   int use_blocked, int split_stage,
                                   int block_groups)
{
    stride_wisdom_add_v4(wis, N, K, factors, nf, best_ns,
                         use_blocked, split_stage, block_groups,
                         /*use_dif_forward=*/0);
}

/* Legacy wrapper (standard executor, no blocking, no DIF, no variants) */
static void stride_wisdom_add(stride_wisdom_t *wis, int N, size_t K,
                              const int *factors, int nf, double best_ns)
{
    stride_wisdom_add_v5(wis, N, K, factors, nf, best_ns, 0, 0, 0, 0,
                         /*has_variant_codes=*/0, /*variant_codes=*/NULL);
}

/* -- Wisdom file I/O --
 * Format v5:
 *   Line 1: @version 5
 *   Remaining: N K nf f0..f{nf-1} best_ns use_blocked split_stage block_groups
 *              use_dif_forward v0..v{nf-1}
 *
 * Example v5 (3-stage plan):
 *   @version 5
 *   256 4 3 4 4 16 879.10 0 0 0 0 0 0 1
 *   (256, K=4, factors=4x4x16, best=879ns, no blocked, DIT, variants=FLAT/FLAT/LOG3)
 *
 * Bump WISDOM_VERSION when the format changes — old files are silently
 * rejected on load and re-calibrated on next run. */
#define WISDOM_VERSION 5

static int stride_wisdom_save(const stride_wisdom_t *wis, const char *path)
{
    FILE *f = fopen(path, "w");
    if (!f)
        return -1;
    fprintf(f, "@version %d\n", WISDOM_VERSION);
    fprintf(f, "# VectorFFT stride wisdom — %d entries\n", wis->count);
    fprintf(f, "# N K nf factors... best_ns use_blocked split_stage block_groups "
               "use_dif_forward variant_codes... (v=0:FLAT 1:LOG3 2:T1S 3:BUF)\n");
    for (int i = 0; i < wis->count; i++)
    {
        const stride_wisdom_entry_t *e = &wis->entries[i];
        fprintf(f, "%d %zu %d", e->N, e->K, e->nfactors);
        for (int j = 0; j < e->nfactors; j++)
            fprintf(f, " %d", e->factors[j]);
        fprintf(f, " %.2f %d %d %d %d", e->best_ns,
                e->use_blocked, e->split_stage, e->block_groups,
                e->use_dif_forward);
        /* Per-stage variant codes. If has_variant_codes=0 (legacy entry
         * not produced by the v1.2+ search), emit -1 placeholders so the
         * loader can distinguish "no codes" from "all-FLAT codes". */
        for (int j = 0; j < e->nfactors; j++)
        {
            if (e->has_variant_codes)
                fprintf(f, " %d", e->variant_codes[j]);
            else
                fprintf(f, " -1");
        }
        fprintf(f, "\n");
    }
    fclose(f);
    return 0;
}

static int stride_wisdom_load(stride_wisdom_t *wis, const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f)
        return -1;
    char line[512];
    int version_ok = 0;

    while (fgets(line, sizeof(line), f))
    {
        if (line[0] == '#' || line[0] == '\n')
            continue;

        if (line[0] == '@')
        {
            int ver = 0;
            if (sscanf(line, "@version %d", &ver) == 1 && ver == WISDOM_VERSION)
                version_ok = 1;
            continue;
        }

        if (!version_ok)
        {
            fclose(f);
            return -1;
        }

        stride_wisdom_entry_t e;
        memset(&e, 0, sizeof(e));
        int pos = 0, n;
        if (sscanf(line, "%d %zu %d%n", &e.N, &e.K, &e.nfactors, &n) < 3)
            continue;
        pos = n;
        if (e.nfactors < 1 || e.nfactors > FACT_MAX_STAGES)
            continue;
        int ok = 1;
        for (int j = 0; j < e.nfactors; j++)
        {
            if (sscanf(line + pos, "%d%n", &e.factors[j], &n) < 1)
            {
                ok = 0;
                break;
            }
            pos += n;
        }
        if (!ok)
            continue;
        if (sscanf(line + pos, "%lf%n", &e.best_ns, &n) < 1)
            continue;
        pos += n;
        /* v5 trailing fields: use_blocked split_stage block_groups
         * use_dif_forward [variant_codes...]. */
        if (sscanf(line + pos, "%d %d %d %d%n",
                   &e.use_blocked, &e.split_stage, &e.block_groups,
                   &e.use_dif_forward, &n) >= 4)
        {
            pos += n;
            /* Variant codes: -1 sentinel means "no explicit codes for this
             * stage" — entry is legacy-style, has_variant_codes stays 0. */
            int any_real_code = 0;
            int parsed_codes[STRIDE_MAX_STAGES] = {0};
            int parsed_ok = 1;
            for (int j = 0; j < e.nfactors; j++)
            {
                int v = -1;
                if (sscanf(line + pos, "%d%n", &v, &n) < 1)
                {
                    parsed_ok = 0;
                    break;
                }
                pos += n;
                parsed_codes[j] = v;
                if (v >= 0)
                    any_real_code = 1;
            }
            if (parsed_ok && any_real_code)
            {
                e.has_variant_codes = 1;
                for (int j = 0; j < e.nfactors; j++)
                {
                    /* -1 placeholders within an otherwise-explicit entry
                     * (shouldn't normally happen) collapse to FLAT. */
                    e.variant_codes[j] = parsed_codes[j] >= 0 ? parsed_codes[j] : 0;
                }
            }
        }
        stride_wisdom_add_v5(wis, e.N, e.K, e.factors, e.nfactors, e.best_ns,
                             e.use_blocked, e.split_stage, e.block_groups,
                             e.use_dif_forward,
                             e.has_variant_codes,
                             e.has_variant_codes ? e.variant_codes : NULL);
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

#include "wisdom_bridge.h" /* stride_prefer_dit_log3() — DIT-only safe query */
#include "threads.h"       /* stride_get_num_threads() */

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
static inline size_t _stride_me_plan(size_t K)
{
    int T = stride_get_num_threads();
    if (T <= 1)
        return K;
    /* K-split slice, rounded up so the planner never under-estimates. */
    return (K + (size_t)T - 1) / (size_t)T;
}

/* Compute the ios (stride between butterfly legs) at stage s.
 *
 * Matches the executor's dim_stride[s] from plan_compute_groups:
 *   ios[s] = K * prod(factors[s+1 .. nf-1])
 */
static inline size_t _stride_ios_at_stage(size_t K, const int *factors,
                                          int nf, int s)
{
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
    const stride_registry_t *reg)
{
    stride_n1_fn n1f[FACT_MAX_STAGES], n1b[FACT_MAX_STAGES];
    stride_t1_fn t1f[FACT_MAX_STAGES], t1b[FACT_MAX_STAGES];

    /* Per-stage protocol decisions. stage_uses_log3 doubles as the
     * log3_mask bits passed to stride_plan_create. */
    int stage_uses_log3[FACT_MAX_STAGES] = {0};
    int stage_skip_t1s[FACT_MAX_STAGES] = {0};

    size_t me_plan = _stride_me_plan(K);

    for (int s = 0; s < nf; s++)
    {
        int R = factors[s];
        n1f[s] = reg->n1_fwd[R];
        n1b[s] = reg->n1_bwd[R];
        if (s == 0)
        {
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

        /* Buf consultation: at cells where flat won cross-protocol AND
         * buf won within flat (t1_buf_dit dispatcher beats t1_dit), use
         * the buf dispatcher in the t1_fwd slot. Same flat protocol —
         * twiddle layout is unchanged — so no flag bit is set; the
         * planner just picks a different codelet pointer.
         *
         * Buf is checked AFTER log3: if log3 won cross-protocol, log3
         * is faster than any flat variant including buf. The wisdom is
         * exclusive (prefer_buf returns 0 wherever flat lost cross-
         * protocol), but the explicit ordering here makes the precedence
         * structurally obvious.
         */
        int want_buf =
            !want_log3 &&
            stride_prefer_buf(R, me_plan, ios_s) &&
            (R < STRIDE_REG_MAX_RADIX) &&
            (reg->t1_buf_fwd[R] != NULL);

        if (want_log3)
        {
            t1f[s] = reg->t1_fwd_log3[R];
            t1b[s] = reg->t1_bwd_log3[R];
            stage_uses_log3[s] = 1;
            /* Prevent the t1s post-pass from shadowing the log3 path.
             * (Executor's runtime branch order would prefer log3 anyway,
             * but clearing t1s_fwd keeps stage state consistent with
             * the plan-time decision.) */
            stage_skip_t1s[s] = 1;
        }
        else if (want_buf)
        {
            t1f[s] = reg->t1_buf_fwd[R];
            t1b[s] = reg->t1_buf_bwd[R];
            /* Buf wins flat — t1s would shadow it via the executor's
             * runtime preference. Skip t1s for this stage. */
            stage_skip_t1s[s] = 1;
        }
        else
        {
            t1f[s] = reg->t1_fwd[R];
            t1b[s] = reg->t1_bwd[R];
        }
    }

    /* Assemble log3_mask from per-stage flags. */
    int log3_mask = 0;
    for (int s = 0; s < nf; s++)
        if (stage_uses_log3[s])
            log3_mask |= (1 << s);

    stride_plan_t *plan = stride_plan_create(N, K, factors, nf,
                                             n1f, n1b, t1f, t1b,
                                             log3_mask);
    if (!plan)
        return NULL;

    /* Attach scalar-broadcast twiddle codelets where wisdom prefers t1s.
     *
     * Gating rule: attach t1s_fwd ONLY when:
     *   1. The stage hasn't been claimed by log3 or buf (stage_skip_t1s)
     *   2. A t1s codelet is registered for this radix
     *   3. The wisdom predicate prefer_t1s fires at this (me, ios)
     *
     * Without rule 3, the executor's runtime preference (which always
     * picks t1s when t1s_fwd is set) would activate t1s at cells where
     * flat or log3 actually wins. The bench data shows ~31% of R=16
     * cells have flat winning over t1s, for example. Gating here lets
     * the wisdom decide per-stage.
     *
     * Backward compatibility: if the radix has no plan_wisdom (e.g.
     * R=2), stride_prefer_t1s returns 0 from its default switch case,
     * and t1s is NOT attached. That's stricter than the pre-Phase-2.1
     * behaviour which attached t1s unconditionally — but R=2 has no
     * t1s codelet anyway, so this is a no-op in practice. Other
     * radixes without wisdom would lose t1s; if that becomes an
     * operational issue, a heuristic fallback (e.g. attach t1s when
     * me <= 256) can be added here. */
    for (int s = 1; s < nf; s++)
    {
        if (stage_skip_t1s[s])
            continue;
        int R = factors[s];
        if (R >= STRIDE_REG_MAX_RADIX || !reg->t1s_fwd[R])
            continue;
        size_t ios_s = _stride_ios_at_stage(K, factors, nf, s);
        if (!stride_prefer_t1s(R, me_plan, ios_s))
            continue;
        plan->stages[s].t1s_fwd = reg->t1s_fwd[R];
        plan->stages[s].t1s_bwd = reg->t1s_bwd[R];
    }

    /* Attach out-of-place twiddle codelets for strided first-stage use.
     * Used by R2C fused pack and 2D FFT strided executor. */
    for (int s = 0; s < nf; s++)
    {
        int R = factors[s];
        if (R < STRIDE_REG_MAX_RADIX && reg->t1_oop_fwd[R])
        {
            plan->stages[s].t1_oop_fwd = reg->t1_oop_fwd[R];
            plan->stages[s].t1_oop_bwd = reg->t1_oop_bwd[R];
        }
    }

    /* Attach scaled n1 codelets for C2R fused unpack.
     * Used by backward R2C: last stage writes ×2 scaled output at stride 2K. */
    for (int s = 0; s < nf; s++)
    {
        int R = factors[s];
        if (R < STRIDE_REG_MAX_RADIX && reg->n1_scaled_fwd[R])
        {
            plan->stages[s].n1_scaled_fwd = reg->n1_scaled_fwd[R];
            plan->stages[s].n1_scaled_bwd = reg->n1_scaled_bwd[R];
        }
    }

    return plan;
}

/* DIF-orientation plan builder.
 *
 * Returns NULL if any stage's radix lacks a DIF codelet — the calibrator
 * uses this NULL return to skip the DIF orientation for plans whose
 * factorization includes non-pow2 factors (R=3, 5, 7, etc., which have
 * no DIF codelets emitted in v1.1).
 *
 * Simpler than _stride_build_plan: no log3/buf/t1s wisdom paths for v1.1
 * (the DIF executor only handles flat). All stages get reg->t1_dif_*
 * pointers; stride_plan_create_ex computes DIF twiddle layout via
 * plan_compute_twiddles_dif_c. */
static stride_plan_t *_stride_build_plan_dif(
    int N, size_t K,
    const int *factors, int nf,
    const stride_registry_t *reg)
{
    stride_n1_fn n1f[FACT_MAX_STAGES], n1b[FACT_MAX_STAGES];
    stride_t1_fn t1f[FACT_MAX_STAGES], t1b[FACT_MAX_STAGES];

    for (int s = 0; s < nf; s++)
    {
        int R = factors[s];
        if (R >= STRIDE_REG_MAX_RADIX || !reg->n1_fwd[R] ||
            !reg->t1_dif_fwd[R] || !reg->t1_dif_bwd[R])
        {
            /* DIF not available for this radix (non-pow2 or untuned). */
            return NULL;
        }
        n1f[s] = reg->n1_fwd[R];
        n1b[s] = reg->n1_bwd[R];
        t1f[s] = reg->t1_dif_fwd[R];
        t1b[s] = reg->t1_dif_bwd[R];
    }

    stride_plan_t *plan = stride_plan_create_ex(
        N, K, factors, nf, n1f, n1b, t1f, t1b,
        /*log3_mask=*/0, /*use_dif_forward=*/1);
    return plan;
}

/* Explicit-variant plan builder. Used by the plan-level calibrator's
 * search loop and by stride_wise_plan when loading wisdom v5+ entries.
 *
 * Caller dictates per-stage variant choice via `variants[]` (length nf)
 * and the orientation via `use_dif_forward`. No wisdom predicates are
 * consulted — this function is the answer to "the codelet-isolation
 * bench was wrong; the plan calibrator's bench is the verdict".
 *
 * `variants[0]` is ignored (stage 0 has no twiddle codelet). For other
 * stages, the variant code maps to registry slots per the table in
 * registry.h's PER-STAGE VARIANT ENUMERATION block.
 *
 * Returns NULL if any (R, use_dif_forward, variant) tuple has no
 * registered codelet — the caller's search loop must already have
 * filtered out unavailable assignments via vfft_stage_variants(). A
 * NULL return here indicates a bug or registry inconsistency, not a
 * routine search-pruning case.
 *
 * Auxiliary slots (t1_oop, n1_scaled) are attached unconditionally
 * where registered — same behavior as _stride_build_plan. */
static stride_plan_t *_stride_build_plan_explicit(
    int N, size_t K,
    const int *factors, int nf,
    const vfft_variant_t *variants,
    int use_dif_forward,
    const stride_registry_t *reg)
{
    stride_n1_fn n1f[FACT_MAX_STAGES], n1b[FACT_MAX_STAGES];
    stride_t1_fn t1f[FACT_MAX_STAGES], t1b[FACT_MAX_STAGES];
    int log3_mask = 0;
    int t1s_mask = 0; /* stages where variant=T1S; t1s_fwd attached after create */

    for (int s = 0; s < nf; s++)
    {
        int R = factors[s];
        if (R <= 0 || R >= STRIDE_REG_MAX_RADIX || !reg->n1_fwd[R])
            return NULL;
        n1f[s] = reg->n1_fwd[R];
        n1b[s] = reg->n1_bwd[R];

        /* Which stage has no twiddle codelet depends on orientation:
         *   DIT: stage 0 (input-edge twiddle to stages 1..nf-1)
         *   DIF: stage nf-1 (output-edge twiddle on stages 0..nf-2)
         * The executor's needs_tw[g] is set to 0 for that stage by
         * plan_compute_twiddles_{c,dif_c}, so leaving t1_fwd NULL is
         * safe — the executor falls through to n1_fwd. */
        int is_no_tw_stage =
            use_dif_forward ? (s == nf - 1) : (s == 0);
        if (is_no_tw_stage)
        {
            t1f[s] = NULL;
            t1b[s] = NULL;
            continue;
        }

        vfft_variant_t v = variants[s];
        if (use_dif_forward)
        {
            switch (v)
            {
            case VFFT_VAR_FLAT:
                if (!reg->t1_dif_fwd[R])
                    return NULL;
                t1f[s] = reg->t1_dif_fwd[R];
                t1b[s] = reg->t1_dif_bwd[R];
                break;
            case VFFT_VAR_LOG3:
                if (!reg->t1_dif_log3_fwd[R])
                    return NULL;
                t1f[s] = reg->t1_dif_log3_fwd[R];
                t1b[s] = reg->t1_dif_log3_bwd[R];
                log3_mask |= (1 << s);
                break;
            default:
                /* T1S, BUF: no DIF analog. */
                return NULL;
            }
        }
        else
        {
            switch (v)
            {
            case VFFT_VAR_FLAT:
                if (!reg->t1_fwd[R])
                    return NULL;
                t1f[s] = reg->t1_fwd[R];
                t1b[s] = reg->t1_bwd[R];
                break;
            case VFFT_VAR_LOG3:
                if (!reg->t1_fwd_log3[R])
                    return NULL;
                t1f[s] = reg->t1_fwd_log3[R];
                t1b[s] = reg->t1_bwd_log3[R];
                log3_mask |= (1 << s);
                break;
            case VFFT_VAR_T1S:
                /* T1S sits on top of flat: codelet uses t1_fwd path
                 * for grp_tw, plus t1s_fwd is attached and the
                 * executor's runtime preference dispatches to it. */
                if (!reg->t1_fwd[R] || !reg->t1s_fwd[R])
                    return NULL;
                t1f[s] = reg->t1_fwd[R];
                t1b[s] = reg->t1_bwd[R];
                t1s_mask |= (1 << s);
                break;
            case VFFT_VAR_BUF:
                if (!reg->t1_buf_fwd[R])
                    return NULL;
                t1f[s] = reg->t1_buf_fwd[R];
                t1b[s] = reg->t1_buf_bwd[R];
                break;
            default:
                return NULL;
            }
        }
    }

    stride_plan_t *plan = stride_plan_create_ex(
        N, K, factors, nf, n1f, n1b, t1f, t1b,
        log3_mask, use_dif_forward);
    if (!plan)
        return NULL;

    /* Attach t1s_fwd to stages whose variant chose T1S. The executor's
     * runtime dispatch checks t1s_fwd != NULL and prefers it on flat
     * stages — so T1S stages get t1s, flat stages stay flat. */
    for (int s = 0; s < nf; s++)
    {
        if (!(t1s_mask & (1 << s)))
            continue;
        int R = factors[s];
        plan->stages[s].t1s_fwd = reg->t1s_fwd[R];
        plan->stages[s].t1s_bwd = reg->t1s_bwd[R];
    }

    /* Attach auxiliary codelets (t1_oop for R2C/2D, n1_scaled for C2R)
     * unconditionally — same as _stride_build_plan. These don't depend
     * on variant choice. */
    for (int s = 0; s < nf; s++)
    {
        int R = factors[s];
        if (R < STRIDE_REG_MAX_RADIX && reg->t1_oop_fwd[R])
        {
            plan->stages[s].t1_oop_fwd = reg->t1_oop_fwd[R];
            plan->stages[s].t1_oop_bwd = reg->t1_oop_bwd[R];
        }
        if (R < STRIDE_REG_MAX_RADIX && reg->n1_scaled_fwd[R])
        {
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

static int _stride_is_prime(int n)
{
    if (n < 2)
        return 0;
    if (n < 4)
        return 1;
    if (n % 2 == 0 || n % 3 == 0)
        return 0;
    for (int i = 5; (long long)i * i <= n; i += 6)
        if (n % i == 0 || n % (i + 2) == 0)
            return 0;
    return 1;
}

/* N-1 factors entirely into primes covered by our radix set */
static int _stride_is_rader_friendly(int n)
{
    int m = n - 1;
    static const int primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 0};
    for (const int *p = primes; *p; p++)
        while (m % *p == 0)
            m /= *p;
    return m == 1;
}

/**
 * stride_auto_plan_wis -- Wisdom-aware heuristic planner.
 *
 *   1. If wisdom has a (N, K) entry, build from it (uses variant codes).
 *   2. Else direct factorization into available radixes -> staged plan.
 *   3. Prime N, non-smooth N-1 -> Bluestein (recurse with wisdom).
 *   4. Prime N, smooth N-1     -> Rader     (recurse with wisdom).
 *   5. Composite with unfactorable prime factor -> NULL.
 *
 * Wisdom plumbing matters for Bluestein/Rader: their inner FFT runs at
 * size M (or N-1) with block-size K, sizes that aren't standard bench
 * cells. Calibrating those (M, B) cells and passing wisdom through the
 * recursion lets the inner plan pick variant codes (LOG3/T1S/BUF) the
 * heuristic would never select. Pass wis=NULL to opt out (legacy path).
 */
static stride_plan_t *stride_auto_plan_wis(int N, size_t K,
                                           const stride_registry_t *reg,
                                           const stride_wisdom_t *wis)
{
    /* 1. Wisdom hit: build with explicit variant codes (v5+) or
     * legacy factor list (v3/v4). Mirrors stride_wise_plan. */
    if (wis)
    {
        const stride_wisdom_entry_t *e = stride_wisdom_lookup(wis, N, K);
        if (e)
        {
            stride_plan_t *plan = NULL;
            if (e->has_variant_codes)
            {
                vfft_variant_t variants[STRIDE_MAX_STAGES];
                for (int s = 0; s < e->nfactors; s++)
                    variants[s] = (vfft_variant_t)e->variant_codes[s];
                plan = _stride_build_plan_explicit(
                    N, K, e->factors, e->nfactors,
                    variants, e->use_dif_forward, reg);
            }
            if (!plan && e->use_dif_forward)
            {
                plan = _stride_build_plan_dif(N, K, e->factors, e->nfactors, reg);
            }
            if (!plan)
            {
                plan = _stride_build_plan(N, K, e->factors, e->nfactors, reg);
            }
            if (plan)
            {
                plan->use_blocked = e->use_blocked;
                plan->split_stage = e->split_stage;
                plan->block_groups = e->block_groups;
                return plan;
            }
            /* Build failed (registry shape mismatch): fall through. */
        }
    }

    /* 2. Direct factorization */
    stride_factorization_t fact;
    if (stride_factorize(N, K, reg, &fact) == 0)
        return _stride_build_plan(N, K, fact.factors, fact.nfactors, reg);

    /* 3. Prime with non-smooth N-1: Bluestein (recurse with wisdom).
     * Block size is T-aware: more blocks at higher T for outer-loop MT. */
    if (_stride_is_prime(N) && !_stride_is_rader_friendly(N))
    {
        int M = _bluestein_choose_m(N);
        size_t B = _bluestein_block_size_T(M, K, stride_get_num_threads());
        stride_plan_t *inner = stride_auto_plan_wis(M, B, reg, wis);
        if (inner)
            return stride_bluestein_plan(N, K, B, inner, M);
    }

    /* 4. Prime with smooth N-1: Rader (recurse with wisdom). */
    if (_stride_is_prime(N) && _stride_is_rader_friendly(N))
    {
        int nm1 = N - 1;
        size_t B = _bluestein_block_size_T(nm1, K, stride_get_num_threads());
        stride_plan_t *inner = stride_auto_plan_wis(nm1, B, reg, wis);
        if (inner)
            return stride_rader_plan(N, K, B, inner);
    }

    /* 5. Composite with unfactorable prime factor: reserved (TODO) */
    return NULL;
}

/* Legacy no-wisdom wrapper (back-compat for callers that don't have
 * wisdom available, e.g., test code or pre-wisdom code paths). */
static stride_plan_t *stride_auto_plan(int N, size_t K,
                                       const stride_registry_t *reg)
{
    return stride_auto_plan_wis(N, K, reg, /*wis=*/NULL);
}

/**
 * stride_exhaustive_plan -- Exhaustive search planner (slow).
 *
 * Tries all factorizations x orderings, benchmarks each,
 * returns a plan built from the best one found.
 */
static stride_plan_t *stride_exhaustive_plan(int N, size_t K,
                                             const stride_registry_t *reg)
{
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
 * Otherwise falls back to wisdom-aware heuristic (stride_auto_plan_wis),
 * which carries wisdom through Bluestein/Rader inner-plan recursion so
 * those inner FFTs can pick up variant-tuned wisdom too.
 */
static stride_plan_t *stride_wise_plan(int N, size_t K,
                                       const stride_registry_t *reg,
                                       const stride_wisdom_t *wis)
{
    const stride_wisdom_entry_t *e = stride_wisdom_lookup(wis, N, K);
    if (!e)
        return stride_auto_plan_wis(N, K, reg, wis);

    /* Preferred path (v5+): explicit per-stage variant codes. The plan
     * calibrator picked these by benching the full plan, so we trust
     * them over any wisdom-bridge predicate consultation. */
    if (e->has_variant_codes)
    {
        vfft_variant_t variants[STRIDE_MAX_STAGES];
        for (int s = 0; s < e->nfactors; s++)
            variants[s] = (vfft_variant_t)e->variant_codes[s];
        stride_plan_t *plan = _stride_build_plan_explicit(
            N, K, e->factors, e->nfactors,
            variants, e->use_dif_forward, reg);
        if (plan)
        {
            plan->use_blocked = e->use_blocked;
            plan->split_stage = e->split_stage;
            plan->block_groups = e->block_groups;
            return plan;
        }
        /* If explicit build fails (wisdom loaded from a host with
         * different registry shape), fall through to legacy build. */
    }

    /* Legacy path (v3/v4 entries, or v5 fallback). Wisdom-bridge
     * predicates pick variants per stage at plan-build time. */
    stride_plan_t *plan = NULL;
    if (e->use_dif_forward)
    {
        plan = _stride_build_plan_dif(N, K, e->factors, e->nfactors, reg);
    }
    if (!plan)
    {
        plan = _stride_build_plan(N, K, e->factors, e->nfactors, reg);
        if (plan)
        {
            plan->use_blocked = e->use_blocked;
            plan->split_stage = e->split_stage;
            plan->block_groups = e->block_groups;
        }
    }
    return plan;
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
/* R2C requires DIT inner plans: the fused first/last-stage paths assume
 * stage 0 is twiddle-free and is the FIRST executed stage, which is only
 * true under DIT semantics. The perm/iperm tables also assume DIT digit-
 * reversal output ordering. A DIF inner plan would silently produce
 * 1-2 unit roundtrip errors with no warning.
 *
 * If a wisdom-tuned inner is DIF, drop it and fall back to a
 * factorization-driven DIT plan (slightly less variant-tuned but correct).
 * Variant codes (LOG3/T1S/BUF) on individual stages are still picked up
 * from wisdom_bridge predicates inside _stride_build_plan. */
static inline stride_plan_t *_r2c_force_dit_inner(
    int halfN, size_t B, const stride_registry_t *reg,
    const stride_wisdom_t *wis)
{
    stride_plan_t *inner = stride_wise_plan(halfN, B, reg, wis);
    if (inner && inner->use_dif_forward)
    {
        stride_plan_destroy(inner);
        inner = NULL;
    }
    if (!inner)
        inner = stride_auto_plan_wis(halfN, B, reg, /*wis=*/NULL);
    return inner;
}

/* Wisdom-aware R2C planner. Inner halfN-point FFT picks variant-tuned codelets
 * via wisdom, but DIF orientation is rejected (incompatible with R2C's fused
 * first/last-stage paths).
 *
 * v1.0 constraint: K must be >= 2. K=1 hits a SIMD edge case where the
 * inner FFT codelets receive vl=1 (block size collapses to 1) and overrun
 * scratch on aligned loads. Returning NULL is safer than corrupting memory.
 * Caller can pad to K>=2 (zero-fill the second batch) as a workaround.
 * Proper fix: v1.1 (B-padding inside the planner). */
static stride_plan_t *stride_r2c_auto_plan_wis(int N, size_t K,
                                               const stride_registry_t *reg,
                                               const stride_wisdom_t *wis)
{
    if (N < 2 || (N & 1))
        return NULL;
    if (K < 2)
        return NULL;  /* v1.0: K=1 corrupts via inner-codelet SIMD overrun */
    int halfN = N / 2;
    size_t B = _bluestein_block_size_T(halfN, K, stride_get_num_threads());
    stride_plan_t *inner = _r2c_force_dit_inner(halfN, B, reg, wis);
    if (!inner)
        return NULL;
    return stride_r2c_plan(N, K, B, inner);
}

/* Legacy no-wisdom wrapper. */
static stride_plan_t *stride_r2c_auto_plan(int N, size_t K,
                                           const stride_registry_t *reg)
{
    return stride_r2c_auto_plan_wis(N, K, reg, /*wis=*/NULL);
}

/* Wisdom-aware R2C plan, mirror of stride_wise_plan for R2C. Same DIT
 * constraint as stride_r2c_auto_plan_wis, and same K>=2 v1.0 constraint. */
static stride_plan_t *stride_r2c_wise_plan(int N, size_t K,
                                           const stride_registry_t *reg,
                                           const stride_wisdom_t *wis)
{
    if (N < 2 || (N & 1))
        return NULL;
    if (K < 2)
        return NULL;  /* v1.0: K=1 corrupts via inner-codelet SIMD overrun */
    int halfN = N / 2;
    size_t B = _bluestein_block_size_T(halfN, K, stride_get_num_threads());
    stride_plan_t *inner = _r2c_force_dit_inner(halfN, B, reg, wis);
    if (!inner)
        return NULL;
    return stride_r2c_plan(N, K, B, inner);
}

/**
 * stride_dct2_auto_plan_wis -- DCT-II (FFTW REDFT10 convention).
 *
 * Y[k] = 2 * sum_{n=0..N-1} x[n] * cos(π k (2n+1) / (2N))   for k=0..N-1
 *
 * Implementation: Makhoul's algorithm — built atop an **N-point R2C**
 * (NOT 2N-point) plus a clever pre-permutation. ~2× faster than the
 * textbook 2N-R2C approach. Matches FFTW's reodft010e-r2hc.c.
 *
 * Constraint: N must be even (our R2C requires even input size).
 *
 *   stride_execute_dct2(plan, in, out)         -- 1D DCT-II, batched K
 */
static stride_plan_t *stride_dct2_auto_plan_wis(int N, size_t K,
                                                const stride_registry_t *reg,
                                                const stride_wisdom_t *wis)
{
    if (N < 2 || (N & 1))
        return NULL; /* Makhoul needs even N */
    stride_plan_t *r2c = stride_r2c_auto_plan_wis(N, K, reg, wis);
    if (!r2c)
        return NULL;
    return stride_dct2_plan(N, K, r2c);
}

/* No-wisdom convenience wrapper. */
static stride_plan_t *stride_dct2_auto_plan(int N, size_t K,
                                            const stride_registry_t *reg)
{
    return stride_dct2_auto_plan_wis(N, K, reg, /*wis=*/NULL);
}

/* Wisdom-aware DCT-II — uses wisdom for the inner N-point R2C plan
 * (subject to R2C's DIT-only v1.0 constraint). */
static stride_plan_t *stride_dct2_wise_plan(int N, size_t K,
                                            const stride_registry_t *reg,
                                            const stride_wisdom_t *wis)
{
    if (N < 2 || (N & 1))
        return NULL;
    stride_plan_t *r2c = stride_r2c_wise_plan(N, K, reg, wis);
    if (!r2c)
        return NULL;
    return stride_dct2_plan(N, K, r2c);
}

/**
 * stride_dht_auto_plan_wis -- Discrete Hartley Transform.
 *
 * H[k] = sum_{n=0..N-1} x[n] * (cos(2*pi*k*n/N) + sin(2*pi*k*n/N))
 *
 * Self-inverse up to 1/N: DHT(DHT(x)) = N*x. Caller divides by N to
 * recover the original. Convention matches FFTW's FFTW_DHT.
 *
 * Built atop N-point R2C plus an O(N*K) butterfly. ~2x faster than
 * a full N-point complex FFT.
 *
 * Constraint: N must be even (our R2C requires even input).
 *
 *   stride_execute_dht(plan, in, out)   -- 1D DHT, batched K
 */
static stride_plan_t *stride_dht_auto_plan_wis(int N, size_t K,
                                               const stride_registry_t *reg,
                                               const stride_wisdom_t *wis)
{
    if (N < 2 || (N & 1))
        return NULL;
    stride_plan_t *r2c = stride_r2c_auto_plan_wis(N, K, reg, wis);
    if (!r2c)
        return NULL;
    return stride_dht_plan(N, K, r2c);
}

/* No-wisdom convenience wrapper. */
static stride_plan_t *stride_dht_auto_plan(int N, size_t K,
                                           const stride_registry_t *reg)
{
    return stride_dht_auto_plan_wis(N, K, reg, /*wis=*/NULL);
}

/* Wisdom-aware DHT — uses wisdom for the inner N-point R2C plan
 * (subject to R2C's DIT-only v1.0 constraint). */
static stride_plan_t *stride_dht_wise_plan(int N, size_t K,
                                           const stride_registry_t *reg,
                                           const stride_wisdom_t *wis)
{
    if (N < 2 || (N & 1))
        return NULL;
    stride_plan_t *r2c = stride_r2c_wise_plan(N, K, reg, wis);
    if (!r2c)
        return NULL;
    return stride_dht_plan(N, K, r2c);
}

/**
 * stride_dst2_auto_plan_wis -- DST-II (FFTW RODFT10) and DST-III (RODFT01).
 *
 *   Y_DST2[k] = 2 * sum_{n=0..N-1} x[n] * sin(pi*(k+1)*(2n+1)/(2N))
 *   Y_DST3[k] = (-1)^k * X[N-1] + 2 * sum_{n=0..N-2} X[n] * sin(pi*(n+1)*(2k+1)/(2N))
 *
 * One plan handles both directions:
 *   stride_execute_dst2(plan, in, out)   -- forward (RODFT10)
 *   stride_execute_dst3(plan, in, out)   -- backward (RODFT01); inverse up to 2N
 *
 * Built atop DCT-II/III with sign-flip + reversal wrapper.
 *
 * Constraint: N must be even (inherits R2C's even-N constraint).
 */
static stride_plan_t *stride_dst2_auto_plan_wis(int N, size_t K,
                                                const stride_registry_t *reg,
                                                const stride_wisdom_t *wis)
{
    if (N < 2 || (N & 1))
        return NULL;
    stride_plan_t *dct = stride_dct2_auto_plan_wis(N, K, reg, wis);
    if (!dct)
        return NULL;
    return stride_dst2_plan(N, K, dct);
}

/* No-wisdom convenience wrapper. */
static stride_plan_t *stride_dst2_auto_plan(int N, size_t K,
                                            const stride_registry_t *reg)
{
    return stride_dst2_auto_plan_wis(N, K, reg, /*wis=*/NULL);
}

/* Wisdom-aware DST-II/III — uses wisdom for the inner DCT-II/III's R2C plan
 * (subject to R2C's DIT-only v1.0 constraint). */
static stride_plan_t *stride_dst2_wise_plan(int N, size_t K,
                                            const stride_registry_t *reg,
                                            const stride_wisdom_t *wis)
{
    if (N < 2 || (N & 1))
        return NULL;
    stride_plan_t *dct = stride_dct2_wise_plan(N, K, reg, wis);
    if (!dct)
        return NULL;
    return stride_dst2_plan(N, K, dct);
}

/**
 * stride_dct4_auto_plan_wis -- DCT-IV (FFTW REDFT11).
 *
 *   Y[k] = 2 * sum_{n=0..N-1} x[n] * cos(pi*(2k+1)*(2n+1)/(4N))
 *
 * Involutory up to scale 2N: DCT-IV(DCT-IV(x))/(2N) = x.
 *
 * Algorithm: Lee 1984 -- single N/2-point complex FFT plus pre/post twiddles.
 * Reuses our existing C2C plan; no new codelets needed.
 *
 * Constraint: N must be even.
 *
 *   stride_execute_dct4(plan, in, out)   -- DCT-IV (involutory)
 */
static stride_plan_t *stride_dct4_auto_plan_wis(int N, size_t K,
                                                const stride_registry_t *reg,
                                                const stride_wisdom_t *wis)
{
    if (N < 2 || (N & 1))
        return NULL;
    if (K < 2)
        return NULL;  /* inherits R2C K>=2 v1.0 constraint */
    int halfN = N / 2;
    stride_plan_t *fft = stride_auto_plan_wis(halfN, K, reg, wis);
    if (!fft)
        return NULL;
    return stride_dct4_plan(N, K, fft);
}

/* No-wisdom convenience wrapper. */
static stride_plan_t *stride_dct4_auto_plan(int N, size_t K,
                                            const stride_registry_t *reg)
{
    return stride_dct4_auto_plan_wis(N, K, reg, /*wis=*/NULL);
}

/* Wisdom-aware DCT-IV. */
static stride_plan_t *stride_dct4_wise_plan(int N, size_t K,
                                            const stride_registry_t *reg,
                                            const stride_wisdom_t *wis)
{
    if (N < 2 || (N & 1))
        return NULL;
    if (K < 2)
        return NULL;
    int halfN = N / 2;
    stride_plan_t *fft = stride_wise_plan(halfN, K, reg, wis);
    if (!fft)
        return NULL;
    return stride_dct4_plan(N, K, fft);
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
                                   const stride_registry_t *reg)
{
    stride_plan_t *plan = _stride_build_plan(N, K, factors, nf, reg);
    if (!plan)
        return 1e18;

    size_t total = (size_t)N * K;
    double *re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    for (size_t i = 0; i < total; i++)
    {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    /* Warmup */
    for (int i = 0; i < 10; i++)
        stride_execute_fwd(plan, re, im);

    int reps = (int)(1e6 / (total + 1));
    if (reps < 20)
        reps = 20;
    if (reps > 100000)
        reps = 100000;

    /* Best of 5 trials */
    double best = 1e18;
    for (int t = 0; t < 5; t++)
    {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++)
            stride_execute_fwd(plan, re, im);
        double ns = (now_ns() - t0) / reps;
        if (ns < best)
            best = ns;
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
    if (!force)
    {
        const stride_wisdom_entry_t *e = stride_wisdom_lookup(wis, N, K);
        if (e)
            return e->best_ns;
    }

    stride_factorization_t best_fact;
    double best_ns;

    if (N <= exhaustive_threshold)
    {
        best_ns = stride_exhaustive_search(N, K, reg, &best_fact, verbose);
    }
    else
    {
        stride_dp_context_t local_ctx;
        int own_ctx = 0;
        if (!dp_ctx)
        {
            stride_dp_init(&local_ctx, K, N);
            dp_ctx = &local_ctx;
            own_ctx = 1;
        }
        best_ns = stride_dp_plan(dp_ctx, N, reg, &best_fact, verbose);
        if (own_ctx)
            stride_dp_destroy(&local_ctx);
    }

    if (best_ns >= 1e17)
        return best_ns;

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
    if (K <= STRIDE_BLOCKED_K_THRESHOLD && N > 512 && N <= exhaustive_threshold)
    {
        if (verbose)
            printf("  Joint blocked search (K<=%d, N>512)...\n",
                   STRIDE_BLOCKED_K_THRESHOLD);

        factorization_list_t *flist = (factorization_list_t *)malloc(sizeof(*flist));
        stride_enumerate_factorizations(N, reg, flist);

        size_t total = (size_t)N * K;
        double *jre = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
        double *jim = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
        double *jorig_re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
        double *jorig_im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
        for (size_t i = 0; i < total; i++)
        {
            jorig_re[i] = (double)rand() / RAND_MAX - 0.5;
            jorig_im[i] = (double)rand() / RAND_MAX - 0.5;
        }

        int jreps = (int)(5e5 / (total + 1));
        if (jreps < 10)
            jreps = 10;
        if (jreps > 50000)
            jreps = 50000;

        for (int fi = 0; fi < flist->count; fi++)
        {
            permutation_list_t *plist = (permutation_list_t *)malloc(sizeof(*plist));
            stride_gen_permutations(flist->results[fi].factors,
                                    flist->results[fi].nfactors, plist);

            for (int pi = 0; pi < plist->count; pi++)
            {
                const int *fac = plist->perms[pi];
                int nf = flist->results[fi].nfactors;
                int prod = 1;
                for (int s = 0; s < nf; s++)
                    prod *= fac[s];
                if (prod != N)
                    continue;

                stride_plan_t *jp = _stride_build_plan(N, K, fac, nf, reg);
                if (!jp)
                    continue;

                /* Try blocked at each valid split point */
                for (int sp = 0; sp < jp->num_stages; sp++)
                {
                    size_t ws = (size_t)jp->stages[sp].radix *
                                jp->stages[sp].stride * K * 2 * sizeof(double);
                    if (ws > STRIDE_BLOCKED_L1_BYTES)
                        continue;

                    int bg = _stride_compute_block_groups(jp, sp);

                    /* Quick bench */
                    for (int w = 0; w < 3; w++)
                        _stride_execute_fwd_blocked(jp, jorig_re, jorig_im, sp, bg);

                    double jbest = 1e18;
                    for (int t = 0; t < 3; t++)
                    {
                        memcpy(jre, jorig_re, total * sizeof(double));
                        memcpy(jim, jorig_im, total * sizeof(double));
                        double t0 = now_ns();
                        for (int r = 0; r < jreps; r++)
                            _stride_execute_fwd_blocked(jp, jre, jim, sp, bg);
                        double ns = (now_ns() - t0) / jreps;
                        if (ns < jbest)
                            jbest = ns;
                    }

                    if (jbest < refined_ns)
                    {
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

        STRIDE_ALIGNED_FREE(jre);
        STRIDE_ALIGNED_FREE(jim);
        STRIDE_ALIGNED_FREE(jorig_re);
        STRIDE_ALIGNED_FREE(jorig_im);
        free(flist);

        if (verbose && win_blocked)
        {
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
    if (dp_ns >= 1e17)
        return dp_ns;

    /* Baseline: standard winner */
    double best_ns = dp_ns;
    *best_fact = dp_fact;

    /* Phase 2: enumerate permutations × split points × executor variants */
    permutation_list_t *plist = (permutation_list_t *)malloc(sizeof(*plist));
    stride_gen_permutations(dp_fact.factors, dp_fact.nfactors, plist);

    size_t total = (size_t)N * K;
    double *jre = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *jim = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *jorig_re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *jorig_im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    for (size_t i = 0; i < total; i++)
    {
        jorig_re[i] = (double)rand() / RAND_MAX - 0.5;
        jorig_im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    int jreps = (int)(5e5 / (total + 1));
    if (jreps < 5)
        jreps = 5;
    if (jreps > 50000)
        jreps = 50000;

    int variants_tried = 0;
    int blocked_winners = 0;

    for (int pi = 0; pi < plist->count; pi++)
    {
        const int *fac = plist->perms[pi];
        int nf = dp_fact.nfactors;

        stride_plan_t *jp = _stride_build_plan(N, K, fac, nf, reg);
        if (!jp)
            continue;

        /* Bench standard executor for this permutation */
        for (int w = 0; w < 3; w++)
            _stride_execute_fwd_slice(jp, jorig_re, jorig_im, K, K);

        double std_best = 1e18;
        for (int t = 0; t < 3; t++)
        {
            memcpy(jre, jorig_re, total * sizeof(double));
            memcpy(jim, jorig_im, total * sizeof(double));
            double t0 = now_ns();
            for (int r = 0; r < jreps; r++)
                _stride_execute_fwd_slice(jp, jre, jim, K, K);
            double ns = (now_ns() - t0) / jreps;
            if (ns < std_best)
                std_best = ns;
        }
        variants_tried++;

        if (std_best < best_ns)
        {
            best_ns = std_best;
            best_fact->nfactors = nf;
            memcpy(best_fact->factors, fac, nf * sizeof(int));
            *out_use_blocked = 0;
            *out_split = 0;
            *out_bg = 0;
        }

        /* Bench blocked executor at each valid split point */
        for (int sp = 0; sp < jp->num_stages; sp++)
        {
            size_t ws = (size_t)jp->stages[sp].radix *
                        jp->stages[sp].stride * K * 2 * sizeof(double);
            if (ws > STRIDE_BLOCKED_L1_BYTES)
                continue;

            int bg = _stride_compute_block_groups(jp, sp);

            for (int w = 0; w < 3; w++)
                _stride_execute_fwd_blocked(jp, jorig_re, jorig_im, sp, bg);

            double jbest = 1e18;
            for (int t = 0; t < 3; t++)
            {
                memcpy(jre, jorig_re, total * sizeof(double));
                memcpy(jim, jorig_im, total * sizeof(double));
                double t0 = now_ns();
                for (int r = 0; r < jreps; r++)
                    _stride_execute_fwd_blocked(jp, jre, jim, sp, bg);
                double ns = (now_ns() - t0) / jreps;
                if (ns < jbest)
                    jbest = ns;
            }
            variants_tried++;

            if (jbest < best_ns)
            {
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

    if (verbose)
    {
        printf("  Joint DP-blocked search: %d variants tried, blocked %s\n",
               variants_tried,
               *out_use_blocked ? "WON" : "lost");
        if (*out_use_blocked)
        {
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
    if (K > STRIDE_BLOCKED_K_THRESHOLD || N <= 512)
    {
        return stride_wisdom_calibrate_full(wis, N, K, reg, dp_ctx,
                                            1, verbose, 1024, NULL);
    }

    /* Small N: existing exhaustive joint search applies (its gate fires) */
    if (N <= 1024)
    {
        return stride_wisdom_calibrate_full(wis, N, K, reg, dp_ctx,
                                            1, verbose, 1024, NULL);
    }

    /* Large N: DP-style joint blocked search */
    stride_factorization_t best_fact;
    int use_blocked, split, bg;
    double best_ns = stride_dp_plan_joint_blocked(
        dp_ctx, N, K, reg, &best_fact,
        &use_blocked, &split, &bg, verbose);

    if (best_ns >= 1e17)
        return best_ns;

    stride_wisdom_add_full(wis, N, K, best_fact.factors, best_fact.nfactors,
                           best_ns, use_blocked, split, bg);
    return best_ns;
}

#define STRIDE_EXHAUSTIVE_THRESHOLD 1024

/* Legacy wrappers for backward compatibility */
static void stride_wisdom_calibrate_ex(stride_wisdom_t *wis, int N, size_t K,
                                       const stride_registry_t *reg,
                                       stride_dp_context_t *dp_ctx)
{
    stride_wisdom_calibrate_full(wis, N, K, reg, dp_ctx,
                                 1, 0, STRIDE_EXHAUSTIVE_THRESHOLD, NULL);
}

static void stride_wisdom_calibrate(stride_wisdom_t *wis, int N, size_t K,
                                    const stride_registry_t *reg)
{
    stride_wisdom_calibrate_full(wis, N, K, reg, NULL,
                                 1, 0, STRIDE_EXHAUSTIVE_THRESHOLD, NULL);
}

/* 2D FFT — must be after stride_auto_plan is defined */
#include "fft2d.h"


/* ═══════════════════════════════════════════════════════════════
 * 2D R2C planner — must be after stride_r2c_auto_plan and stride_auto_plan.
 *
 * Forward: N1*N2 reals -> N1*(N2/2+1) complex (reduces along inner axis).
 * Backward: reverse, scaled bwd(fwd(x)) = (N1*N2) * x.
 *
 * Constraint: N2 must be even.
 * ═══════════════════════════════════════════════════════════════ */

/* Tile size selector: B must equal the inner R2C plan's K (they index the
 * same scratch). We pick B = min(DEFAULT_TILE, N1), clamped to >=2 (R2C K>=2
 * v1.0 constraint). For N1=1 we'd need a different approach (just call 1D R2C);
 * v1.0 rejects N1<2. */
static inline size_t _fft2d_r2c_choose_tile(int N1) {
    size_t B = FFT2D_R2C_DEFAULT_TILE;
    if (B > (size_t)N1) B = (size_t)N1;
    return B;
}

/** 2D R2C plan (no wisdom). Uses 1D R2C inner (K=B) and 1D C2C col (K=K_pad).
 *
 * K_pad = ceil((N2/2+1)/4)*4 — col FFT batch is padded to multiple of 4
 * because our codelets' SIMD loops have no scalar tail at vl<4. The
 * pad columns are zero-filled internally; convenience wrappers pack/unpack
 * to the user-facing N1*(N2/2+1) layout. */
static stride_plan_t *stride_plan_2d_r2c(int N1, int N2,
                                          const stride_registry_t *reg)
{
    if (N1 < 2 || N2 < 2 || (N2 & 1)) return NULL;  /* R2C needs K>=2 */
    size_t B = _fft2d_r2c_choose_tile(N1);
    size_t hp1 = (size_t)(N2 / 2 + 1);
    size_t K_pad = (hp1 + 3) & ~(size_t)3;  /* round up to multiple of 4 */

    stride_plan_t *plan_r2c = stride_r2c_auto_plan(N2, B, reg);
    if (!plan_r2c) return NULL;
    stride_plan_t *plan_col = stride_auto_plan(N1, K_pad, reg);
    if (!plan_col) { stride_plan_destroy(plan_r2c); return NULL; }

    return stride_plan_2d_r2c_from(N1, N2, B, K_pad, plan_r2c, plan_col);
}

/** Wisdom-aware 2D R2C plan.
 *
 * v1.0 SAFETY: matches the 2D C2C wisdom-disable for the column FFT
 * (see stride_plan_2d_wise note in fft2d.h about K-split + variant-coded
 * plan corruption at large K). Both inner plans here use NON-wisdom paths;
 * row R2C also bypasses wisdom for the same paranoia.
 *
 * Cost: ~3-5% per-stage tuning loss vs full wisdom. v1.1 will re-enable
 * once the K-split + variant-coded corruption is root-caused. */
static stride_plan_t *stride_plan_2d_r2c_wise(int N1, int N2,
                                               const stride_registry_t *reg,
                                               const stride_wisdom_t *wis)
{
    (void)wis;  /* unused in v1.0 — see safety note above */
    return stride_plan_2d_r2c(N1, N2, reg);
}


#endif /* STRIDE_PLANNER_H */
