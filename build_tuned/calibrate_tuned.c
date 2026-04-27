/**
 * calibrate_tuned.c — calibrate the new tuned core for a fixed cell grid.
 *
 * For each (N, K) cell:
 *   1. Find the best factorization on this host:
 *        - exhaustive search for N <= 2048
 *        - DP search          for N >= 4096
 *      Both route through _stride_build_plan, so the chosen plan already
 *      reflects per-stage wisdom (log3 / buf / t1s / flat).
 *   2. Re-bench the chosen plan with the deploy-quality protocol used by
 *      bench_planner.c: 10-rep warmup, 5 trials of N>0 reps, take min.
 *   3. Verify roundtrip error < 1e-12 (correctness gate).
 *   4. Add to wisdom; print one row per cell: factors, codelets, ns.
 *
 * Output: vfft_wisdom_tuned.txt next to this binary, in the same v3
 * format produced by stride_wisdom_save (so it is a drop-in replacement
 * for production's vfft_wisdom.txt for any tooling that loads either).
 *
 * Power-state requirement: this binary expects the active Windows power
 * plan to be High Performance (matches orchestrator's calibration
 * conditions). The Python launcher (calibrate.py) sets and restores it.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "planner.h"
#include "dp_planner.h"
#include "exhaustive.h"
#include "env.h"
#include "compat.h"        /* now_ns(), STRIDE_ALIGNED_ALLOC/FREE */
#include "wisdom_bridge.h"

/* ===========================================================================
 * CALIBRATION MODE
 *
 * CALIB_PILOT_JOINT=1 (set below):
 *   - Small (N, K) grid for fast turnaround
 *   - JOINT search: exhaustive over factorizations × permutations × per-stage
 *     variants × orientations. No wisdom-predicate shortcut. The plan-level
 *     bench is the verdict for both factorization AND variant choice.
 *   - DP planner skipped entirely.
 *
 * CALIB_PILOT_JOINT=0:
 *   - Full pow2 grid (40 cells)
 *   - SEQUENTIAL search: Phase A picks factorization via wisdom-driven
 *     stride_wisdom_calibrate_full (DP for large N, exh for small N), then
 *     Phase C cartesian-searches variants on that one winning factorization.
 *
 * The pilot/joint mode runs much fewer cells but each cell is more
 * expensive per cell (~10-100x more bench plans). For total wall time the
 * pilot is faster because it's 4 cells instead of 40.
 * ========================================================================= */

/* Default 0 (full grid mirroring bench_1d_csv) post-v1.2-MEASURE-topk
 * landing. Pilot mode (CALIB_PILOT_JOINT=1) provides a 2-cell fast
 * iteration grid for development; enable via -DCALIB_PILOT_JOINT=1. */
#ifndef CALIB_PILOT_JOINT
  #define CALIB_PILOT_JOINT 0
#endif

#if CALIB_PILOT_JOINT
  /* Pilot: original v1.2 validation cells. */
  static const int   GRID_N[] = { 4096 };
  static const size_t GRID_K[] = { 4, 256 };
  #define EXHAUSTIVE_MAX_N 65536
#else
  /* Full grid: same N values as bench_1d_csv.c::all_sizes[] so every
   * cell the MKL bench knows about gets a wisdom entry (modulo prime N
   * which calibrate_cell_measure skips — those fall back to
   * stride_auto_plan at bench time). K values match bench_1d_csv's. */
  static const int   GRID_N[] = {
      /* small */
      8, 16, 32, 64, 128,
      /* pow2 */
      256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
      /* composite */
      60, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000,
      /* prime_pow */
      243, 625, 2401, 3125, 15625, 16807, 78125, 117649, 390625, 823543,
      /* genfft */
      1331, 14641, 161051, 2197, 28561,
      /* rader (primes — calibrate_cell_measure skips these) */
      127, 251, 257, 401, 641, 1009, 2801, 4001,
      /* odd_comp */
      175, 525, 1225, 2205, 6615, 11025,
      /* mixed_deep */
      2310, 6930, 30030, 60060, 4620, 13860,
  };
  static const size_t GRID_K[] = { 4, 32, 256 };
  #define EXHAUSTIVE_MAX_N 2048
#endif

/* Inter-cell pacing (ms). Lets thermal/cache state settle between cells so
 * one cell's residual heat or cache contents doesn't bias the next cell's
 * search measurements. Override at runtime via argv[3]. */
#define DEFAULT_PACE_MS 1000

#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
  static void pace_ms(int ms) { if (ms > 0) Sleep((DWORD)ms); }
#else
  #include <unistd.h>
  static void pace_ms(int ms) { if (ms > 0) usleep((useconds_t)ms * 1000); }
#endif

/* ===========================================================================
 * Per-stage codelet classification (mirrors test_tuned_core.c).
 * ========================================================================= */

typedef enum { CL_N1=0, CL_LOG3, CL_BUF, CL_T1S, CL_FLAT } codelet_kind_t;

static const char *codelet_short(codelet_kind_t k) {
    switch (k) {
        case CL_N1:   return "n1";
        case CL_LOG3: return "log3";
        case CL_BUF:  return "buf";
        case CL_T1S:  return "t1s";
        case CL_FLAT: return "flat";
    }
    return "?";
}

static codelet_kind_t classify_stage(const stride_plan_t *plan, size_t K, int s) {
    if (s == 0) return CL_N1;
    int R = plan->factors[s];
    size_t ios = K;
    for (int d = s + 1; d < plan->num_stages; d++) ios *= (size_t)plan->factors[d];
    size_t me = K;
    if (stride_prefer_dit_log3(R, me, ios)) return CL_LOG3;
    if (stride_prefer_buf(R, me, ios))      return CL_BUF;
    if (stride_prefer_t1s(R, me, ios) && plan->stages[s].t1s_fwd) return CL_T1S;
    return CL_FLAT;
}

/* ===========================================================================
 * Bench protocol — matches bench_planner.c exactly so the numbers we
 * write into vfft_wisdom_tuned.txt are directly comparable to what
 * bench_planner.c writes into vfft_wisdom.txt.
 * ========================================================================= */

static double bench_plan_min(const stride_plan_t *plan, int N, size_t K) {
    size_t total = (size_t)N * K;
    double *re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    if (!re || !im) {
        if (re) STRIDE_ALIGNED_FREE(re);
        if (im) STRIDE_ALIGNED_FREE(im);
        return 1e18;
    }
    for (size_t i = 0; i < total; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    /* Warmup */
    for (int i = 0; i < 10; i++) stride_execute_fwd_auto(plan, re, im);

    int reps = (int)(1e6 / (total + 1));
    if (reps < 20) reps = 20;
    if (reps > 100000) reps = 100000;

    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++) stride_execute_fwd_auto(plan, re, im);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }

    STRIDE_ALIGNED_FREE(re);
    STRIDE_ALIGNED_FREE(im);
    return best;
}

/* Roundtrip correctness check — same protocol as test_tuned_core.c. */
static double roundtrip_err(const stride_plan_t *plan, int N, size_t K) {
    size_t total = (size_t)N * K;
    double *re  = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im  = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *re0 = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im0 = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    if (!re || !im || !re0 || !im0) {
        if (re)  STRIDE_ALIGNED_FREE(re);  if (im)  STRIDE_ALIGNED_FREE(im);
        if (re0) STRIDE_ALIGNED_FREE(re0); if (im0) STRIDE_ALIGNED_FREE(im0);
        return 1e30;
    }
    srand(42 + N + (int)K);
    for (size_t i = 0; i < total; i++) {
        re0[i] = (double)rand() / RAND_MAX - 0.5;
        im0[i] = (double)rand() / RAND_MAX - 0.5;
    }
    memcpy(re, re0, total * sizeof(double));
    memcpy(im, im0, total * sizeof(double));
    stride_execute_fwd_auto(plan, re, im);
    stride_execute_bwd_auto(plan, re, im);
    double max_err = 0.0;
    double scale = 1.0 / (double)N;
    for (size_t i = 0; i < total; i++) {
        double er = fabs(re[i] * scale - re0[i]);
        double ei = fabs(im[i] * scale - im0[i]);
        if (er > max_err) max_err = er;
        if (ei > max_err) max_err = ei;
    }
    STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);
    STRIDE_ALIGNED_FREE(re0); STRIDE_ALIGNED_FREE(im0);
    return max_err;
}

/* ===========================================================================
 * Per-cell calibration
 * ========================================================================= */

typedef struct {
    int                 nfactors;
    int                 factors[FACT_MAX_STAGES];
    codelet_kind_t      codelets[FACT_MAX_STAGES];
    double              best_ns;
    double              dit_ns;          /* baseline DIT (wisdom predicate pick) */
    double              dif_ns;          /* DIF bench (best across DIF variants), or -1 if N/A */
    double              err;
    const char         *method;
    int                 use_blocked;
    int                 split_stage;
    int                 block_groups;
    int                 use_dif_forward;
    int                 variant_codes[FACT_MAX_STAGES];
    long                variants_searched;  /* how many cartesian assignments benched */
} cell_result_t;

/* Calibrate one cell. Routes through stride_wisdom_calibrate_full so we
 * inherit production's exact protocol (search → refine_bench → joint
 * blocked search for K<=8 N>512 N<=threshold). For K<=8 cells with
 * N > threshold we additionally call stride_dp_plan_joint_blocked
 * explicitly — that's how production gets blocked wisdom for N=2048,
 * 4096+ at K=4 (via bench_blocked_vs_mkl.c). */
static int calibrate_cell(int N, size_t K,
                          const stride_registry_t *reg,
                          stride_dp_context_t *dp_ctx,
                          stride_wisdom_t *wis,
                          cell_result_t *out) {
    out->method = (N <= EXHAUSTIVE_MAX_N) ? "exh" : "dp";

    /* ── Phase A: standard search + small-N joint blocked ───────────── */
    double std_ns = stride_wisdom_calibrate_full(
            wis, N, K, reg, dp_ctx,
            /*force=*/1, /*verbose=*/0,
            EXHAUSTIVE_MAX_N, /*save_path=*/NULL);
    if (std_ns >= 1e17) {
        fprintf(stderr, "  N=%d K=%zu: standard search returned 1e18\n", N, K);
        return 1;
    }

    /* ── Phase B: large-N joint blocked for small K ─────────────────── *
     * stride_wisdom_calibrate_full's joint blocked path only fires for
     * N <= EXHAUSTIVE_MAX_N (the K<=8 N>512 N<=threshold gate). For
     * larger N at K<=8 we run stride_dp_plan_joint_blocked manually so
     * we don't miss blocked wins like N=4096 K=4. */
    if (K <= STRIDE_BLOCKED_K_THRESHOLD && N > EXHAUSTIVE_MAX_N) {
        stride_factorization_t jb_fact;
        int jb_use_blocked = 0, jb_split = 0, jb_bg = 0;
        double joint_ns = stride_dp_plan_joint_blocked(
                dp_ctx, N, K, reg, &jb_fact,
                &jb_use_blocked, &jb_split, &jb_bg, /*verbose=*/0);
        if (joint_ns < 1e17) {
            /* Refine-bench the joint winner with the same protocol used
             * for the standard winner so the comparison is fair. */
            stride_plan_t *jp = _stride_build_plan(
                    N, K, jb_fact.factors, jb_fact.nfactors, reg);
            if (jp) {
                if (jb_use_blocked) {
                    jp->use_blocked  = 1;
                    jp->split_stage  = jb_split;
                    jp->block_groups = jb_bg;
                }
                double refined = bench_plan_min(jp, N, K);
                stride_plan_destroy(jp);
                if (refined < std_ns) {
                    /* stride_wisdom_add_full updates only if best_ns
                     * decreases, so this is idempotent. */
                    stride_wisdom_add_full(wis, N, K,
                                           jb_fact.factors, jb_fact.nfactors,
                                           refined,
                                           jb_use_blocked, jb_split, jb_bg);
                    out->method = (N <= EXHAUSTIVE_MAX_N) ? "exh+jb" : "dp+jb";
                }
            }
        }
    }

    /* ── Phase C: read DIT timing baseline ──────────────────────────── */
    const stride_wisdom_entry_t *e = stride_wisdom_lookup(wis, N, K);
    if (!e) {
        fprintf(stderr, "  N=%d K=%zu: wisdom lookup failed\n", N, K);
        return 1;
    }
    out->dit_ns = e->best_ns;
    out->dif_ns = -1.0;

    /* Capture the factorization before we start updating wisdom; the
     * pointer `e` may become stale across stride_wisdom_add_v5 calls. */
    int factors[STRIDE_MAX_STAGES];
    int nf = e->nfactors;
    for (int s = 0; s < nf; s++) factors[s] = e->factors[s];
    int blocked_won  = e->use_blocked;
    int split_stage  = e->split_stage;
    int block_groups = e->block_groups;
    double baseline_ns = e->best_ns;

    /* ── Phase D: variant cartesian search ──────────────────────────── *
     * Walks (orientation × per-stage variant assignment) on the winning
     * factorization. Skipped when the blocked executor won — variants
     * don't compose with blocking in v1.1, so the blocked plan stays
     * the verdict. The user can still load it; the v5 entry just has
     * has_variant_codes=0 and stride_wise_plan falls back to the
     * legacy build path.
     *
     * For each variant assignment that beats the prior best, we update
     * wisdom incrementally — stride_wisdom_add_v5 only commits when
     * best_ns decreases, so this is idempotent.
     */
    double best_ns = baseline_ns;
    int best_use_dif = 0;
    vfft_variant_t best_variants[STRIDE_MAX_STAGES] = {0};
    int has_winner = 0;
    long total_searched = 0;

    if (!blocked_won) {
        for (int orient = 0; orient < 2; orient++) {
            vfft_variant_iter_t it;
            if (!vfft_variant_iter_init(&it, factors, nf, orient, reg))
                continue;
            do {
                vfft_variant_t variants[STRIDE_MAX_STAGES];
                vfft_variant_iter_get(&it, variants);

                stride_plan_t *plan = _stride_build_plan_explicit(
                        N, K, factors, nf, variants, orient, reg);
                if (!plan) continue;

                double ns = bench_plan_min(plan, N, K);
                stride_plan_destroy(plan);
                total_searched++;

                if (ns < best_ns) {
                    best_ns = ns;
                    best_use_dif = orient;
                    for (int s = 0; s < nf; s++) best_variants[s] = variants[s];
                    has_winner = 1;
                }
                /* Also track best DIF result (any variant) for diagnostic. */
                if (orient == 1 && (out->dif_ns < 0 || ns < out->dif_ns))
                    out->dif_ns = ns;
            } while (vfft_variant_iter_next(&it));
        }

        if (has_winner) {
            int code_ints[STRIDE_MAX_STAGES];
            for (int s = 0; s < nf; s++) code_ints[s] = (int)best_variants[s];
            stride_wisdom_add_v5(wis, N, K, factors, nf,
                                  best_ns,
                                  /*use_blocked=*/0, /*split=*/0, /*bg=*/0,
                                  best_use_dif,
                                  /*has_variant_codes=*/1, code_ints);
            const char *base = (out->method && strstr(out->method, "+jb"))
                              ? ((N <= EXHAUSTIVE_MAX_N) ? "exh+jb" : "dp+jb")
                              : ((N <= EXHAUSTIVE_MAX_N) ? "exh"    : "dp");
            out->method = best_use_dif ? (strstr(base, "+jb") ?
                                          (N <= EXHAUSTIVE_MAX_N ? "exh+jb/dif" : "dp+jb/dif")
                                        : (N <= EXHAUSTIVE_MAX_N ? "exh/dif"    : "dp/dif"))
                                       : (strstr(base, "+jb") ?
                                          (N <= EXHAUSTIVE_MAX_N ? "exh+jb/var" : "dp+jb/var")
                                        : (N <= EXHAUSTIVE_MAX_N ? "exh/var"    : "dp/var"));
        }
    }
    out->variants_searched = total_searched;

    /* ── Phase E: copy final result, build plan, verify roundtrip ──── */
    e = stride_wisdom_lookup(wis, N, K);  /* re-fetch after any updates */
    out->nfactors        = e->nfactors;
    for (int s = 0; s < e->nfactors; s++) out->factors[s] = e->factors[s];
    out->best_ns         = e->best_ns;
    out->use_blocked     = e->use_blocked;
    out->split_stage     = e->split_stage;
    out->block_groups    = e->block_groups;
    out->use_dif_forward = e->use_dif_forward;
    for (int s = 0; s < e->nfactors; s++)
        out->variant_codes[s] = e->has_variant_codes ? e->variant_codes[s] : -1;

    stride_plan_t *plan;
    if (e->has_variant_codes) {
        vfft_variant_t variants[STRIDE_MAX_STAGES];
        for (int s = 0; s < e->nfactors; s++)
            variants[s] = (vfft_variant_t)e->variant_codes[s];
        plan = _stride_build_plan_explicit(
                N, K, e->factors, e->nfactors, variants,
                e->use_dif_forward, reg);
    } else if (e->use_dif_forward) {
        plan = _stride_build_plan_dif(N, K, e->factors, e->nfactors, reg);
    } else {
        plan = _stride_build_plan(N, K, e->factors, e->nfactors, reg);
    }
    if (!plan) {
        fprintf(stderr, "  N=%d K=%zu: build plan from wisdom failed\n", N, K);
        return 1;
    }
    if (e->use_blocked) {
        plan->use_blocked  = e->use_blocked;
        plan->split_stage  = e->split_stage;
        plan->block_groups = e->block_groups;
    }

    /* Codelet display kind. v5 entries: derive directly from variant codes.
     * Legacy entries (blocked won, or DIT/DIF without explicit search):
     * fall back to wisdom-bridge classify_stage. */
    if (e->has_variant_codes) {
        out->codelets[0] = CL_N1;
        for (int s = 1; s < plan->num_stages; s++) {
            switch ((vfft_variant_t)e->variant_codes[s]) {
                case VFFT_VAR_FLAT: out->codelets[s] = CL_FLAT; break;
                case VFFT_VAR_LOG3: out->codelets[s] = CL_LOG3; break;
                case VFFT_VAR_T1S:  out->codelets[s] = CL_T1S;  break;
                case VFFT_VAR_BUF:  out->codelets[s] = CL_BUF;  break;
                default:            out->codelets[s] = CL_FLAT; break;
            }
        }
    } else if (e->use_dif_forward) {
        out->codelets[0] = CL_N1;
        for (int s = 1; s < plan->num_stages; s++) out->codelets[s] = CL_FLAT;
    } else {
        for (int s = 0; s < plan->num_stages; s++)
            out->codelets[s] = classify_stage(plan, K, s);
    }
    out->err = roundtrip_err(plan, N, K);
    stride_plan_destroy(plan);
    return 0;
}

/* ===========================================================================
 * JOINT SEARCH — exhaustive over factorizations × permutations × per-stage
 * variants × orientations. The plan-level bench is the verdict for every
 * decision dimension (no wisdom-predicate shortcut anywhere in the search).
 *
 * This is what the user describes as "for each (stage, radix) combination,
 * find the best codelet". Used in pilot mode (small grid) where the cost
 * of joint search is manageable. For the full grid, sequential search
 * (calibrate_cell above) is the affordable approximation.
 * ========================================================================= */

static int calibrate_cell_joint(int N, size_t K,
                                 const stride_registry_t *reg,
                                 stride_wisdom_t *wis,
                                 cell_result_t *out) {
    factorization_list_t flist;
    stride_enumerate_factorizations(N, reg, &flist);
    if (flist.count == 0) {
        fprintf(stderr, "  N=%d K=%zu: no factorizations enumerated\n", N, K);
        return 1;
    }

    double best_ns = 1e18;
    int    best_nf = 0;
    int    best_factors[STRIDE_MAX_STAGES] = {0};
    int    best_use_dif = 0;
    vfft_variant_t best_variants[STRIDE_MAX_STAGES] = {0};
    long   total_searched = 0;

    for (int fi = 0; fi < flist.count; fi++) {
        const int nf = flist.results[fi].nfactors;
        const int *base_factors = flist.results[fi].factors;

        permutation_list_t plist;
        stride_gen_permutations(base_factors, nf, &plist);

        for (int pi = 0; pi < plist.count; pi++) {
            const int *perm = plist.perms[pi];

            /* Validate: all stage radixes must have n1 codelet registered.
             * (Some non-pow2 factorizations may include a radix without
             * SIMD codelets; skip those silently.) */
            int can_build = 1;
            for (int s = 0; s < nf; s++) {
                int R = perm[s];
                if (R <= 0 || R >= STRIDE_REG_MAX_RADIX || !reg->n1_fwd[R]) {
                    can_build = 0;
                    break;
                }
            }
            if (!can_build) continue;

            for (int orient = 0; orient < 2; orient++) {
                vfft_variant_iter_t iter;
                if (!vfft_variant_iter_init(&iter, perm, nf, orient, reg))
                    continue;
                do {
                    vfft_variant_t variants[STRIDE_MAX_STAGES];
                    vfft_variant_iter_get(&iter, variants);

                    stride_plan_t *plan = _stride_build_plan_explicit(
                            N, K, perm, nf, variants, orient, reg);
                    if (!plan) continue;

                    double ns = bench_plan_min(plan, N, K);
                    stride_plan_destroy(plan);
                    total_searched++;

                    if (ns < best_ns) {
                        best_ns = ns;
                        best_nf = nf;
                        for (int s = 0; s < nf; s++) best_factors[s] = perm[s];
                        best_use_dif = orient;
                        for (int s = 0; s < nf; s++) best_variants[s] = variants[s];
                    }
                } while (vfft_variant_iter_next(&iter));
            }
        }
    }

    if (best_ns >= 1e17) {
        fprintf(stderr, "  N=%d K=%zu: joint search found no working plan\n", N, K);
        return 1;
    }

    /* Commit to wisdom (v5 with explicit codes). */
    int code_ints[STRIDE_MAX_STAGES];
    for (int s = 0; s < best_nf; s++) code_ints[s] = (int)best_variants[s];
    stride_wisdom_add_v5(wis, N, K, best_factors, best_nf, best_ns,
                          /*use_blocked=*/0, /*split=*/0, /*bg=*/0,
                          best_use_dif,
                          /*has_variant_codes=*/1, code_ints);

    /* Fill cell_result_t for the harness's display + CSV. */
    out->method        = "exh+joint";
    out->variants_searched = total_searched;
    out->dit_ns        = -1;
    out->dif_ns        = -1;
    out->nfactors      = best_nf;
    for (int s = 0; s < best_nf; s++) out->factors[s] = best_factors[s];
    out->best_ns       = best_ns;
    out->use_blocked   = 0;
    out->split_stage   = 0;
    out->block_groups  = 0;
    out->use_dif_forward = best_use_dif;
    for (int s = 0; s < best_nf; s++) out->variant_codes[s] = code_ints[s];

    /* Codelet display from variant codes directly. */
    out->codelets[0] = CL_N1;
    for (int s = 1; s < best_nf; s++) {
        switch (best_variants[s]) {
            case VFFT_VAR_FLAT: out->codelets[s] = CL_FLAT; break;
            case VFFT_VAR_LOG3: out->codelets[s] = CL_LOG3; break;
            case VFFT_VAR_T1S:  out->codelets[s] = CL_T1S;  break;
            case VFFT_VAR_BUF:  out->codelets[s] = CL_BUF;  break;
            default:            out->codelets[s] = CL_FLAT; break;
        }
    }

    /* Roundtrip verification on the winner. */
    stride_plan_t *plan = _stride_build_plan_explicit(
            N, K, best_factors, best_nf, best_variants, best_use_dif, reg);
    if (!plan) {
        fprintf(stderr, "  N=%d K=%zu: winner build failed\n", N, K);
        return 1;
    }
    out->err = roundtrip_err(plan, N, K);
    stride_plan_destroy(plan);
    return 0;
}

/* ===========================================================================
 * Blocked-executor refine for K <= STRIDE_BLOCKED_K_THRESHOLD medium N
 *
 * MEASURE picks (factorization, permutation, variants, orientation) using
 * the standard executor. For small batch (K<=8) at medium N, the blocked
 * executor can beat the standard one by easing DTLB pressure. This helper
 * takes MEASURE's variant-tuned plan and tries the blocked executor at
 * each valid split point. Variants flow through naturally because the
 * blocked executor calls the same plan->stages[s].t1_fwd codelet pointers
 * that _stride_build_plan_explicit set up.
 *
 * Returns 1 if blocked beats MEASURE's deploy bench, 0 otherwise.
 * Mirrors the L1 working-set guard + block_groups computation from
 * stride_dp_plan_joint_blocked. */
static int try_blocked_refine(
        int N, size_t K,
        const stride_plan_decision_t *dec,
        const stride_registry_t *reg,
        double measure_deploy_ns,
        int *out_split, int *out_bg, double *out_blocked_ns)
{
    *out_split = 0;
    *out_bg = 0;
    *out_blocked_ns = 1e18;

    if (K > STRIDE_BLOCKED_K_THRESHOLD) return 0;
    if (N <= 512) return 0;

    stride_plan_t *plan = _stride_build_plan_explicit(
            N, K, dec->fact.factors, dec->fact.nfactors,
            dec->variants, dec->use_dif_forward, reg);
    if (!plan) return 0;

    size_t total = (size_t)N * K;
    double *re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    double *im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    if (!re || !im) {
        if (re) STRIDE_ALIGNED_FREE(re);
        if (im) STRIDE_ALIGNED_FREE(im);
        stride_plan_destroy(plan);
        return 0;
    }

    int reps = (int)(2e6 / (total + 1));
    if (reps < 20) reps = 20;
    if (reps > 100000) reps = 100000;

    double best_blocked = 1e18;
    int best_sp = 0, best_bg = 0;

    for (int sp = 0; sp < plan->num_stages; sp++) {
        size_t ws = (size_t)plan->stages[sp].radix *
                    plan->stages[sp].stride * K * 2 * sizeof(double);
        if (ws > STRIDE_BLOCKED_L1_BYTES) continue;

        int bg = _stride_compute_block_groups(plan, sp);

        srand(42 + N + (int)K + sp);
        for (size_t i = 0; i < total; i++) {
            re[i] = (double)rand() / RAND_MAX - 0.5;
            im[i] = (double)rand() / RAND_MAX - 0.5;
        }
        for (int w = 0; w < 10; w++)
            _stride_execute_fwd_blocked(plan, re, im, sp, bg);

        double trial_best = 1e18;
        for (int t = 0; t < 5; t++) {
            srand(42 + N + (int)K + sp);
            for (size_t i = 0; i < total; i++) {
                re[i] = (double)rand() / RAND_MAX - 0.5;
                im[i] = (double)rand() / RAND_MAX - 0.5;
            }
            double t0 = now_ns();
            for (int r = 0; r < reps; r++)
                _stride_execute_fwd_blocked(plan, re, im, sp, bg);
            double ns = (now_ns() - t0) / reps;
            if (ns < trial_best) trial_best = ns;
        }

        if (trial_best < best_blocked) {
            best_blocked = trial_best;
            best_sp = sp;
            best_bg = bg;
        }
    }

    STRIDE_ALIGNED_FREE(re);
    STRIDE_ALIGNED_FREE(im);
    stride_plan_destroy(plan);

    if (best_blocked < measure_deploy_ns) {
        *out_split = best_sp;
        *out_bg = best_bg;
        *out_blocked_ns = best_blocked;
        return 1;
    }
    return 0;
}

/* Force-replace any existing entry for (N, K) before adding a new one.
 * stride_wisdom_add_v5 is "update if better" — for selective re-calibration
 * we want the new bench to win regardless of cost (the calibrator's logic
 * may have changed). */
static void calib_remove_entry(stride_wisdom_t *wis, int N, size_t K) {
    for (int i = 0; i < wis->count; i++) {
        if (wis->entries[i].N == N && wis->entries[i].K == K) {
            for (int j = i; j < wis->count - 1; j++)
                wis->entries[j] = wis->entries[j + 1];
            wis->count--;
            return;
        }
    }
}

/* ===========================================================================
 * VFFT_MEASURE per-cell calibration
 *
 * Uses stride_dp_plan_measure to pick (factorization × variants × orient)
 * via DP-recursion-then-variant-cartesian. Cheaper than calibrate_cell_joint
 * by ~100-1000x; misses cases where variant choice would flip the
 * factorization ranking (the v1.2 K=4 finding pattern). Re-benches the
 * winner with deploy-quality bench_plan_min so the wisdom cost numbers
 * are directly comparable to the joint-mode entries.
 *
 * Post-MEASURE: tries the blocked executor at each valid split point on
 * the same variant-tuned plan. If blocked beats MEASURE's bench, the
 * wisdom entry is stamped with use_blocked=1 + split + bg.
 * ========================================================================= */

static int calibrate_cell_measure(int N, size_t K,
                                   const stride_registry_t *reg,
                                   stride_dp_context_t *dp_ctx,
                                   stride_wisdom_t *wis,
                                   cell_result_t *out) {
    stride_plan_decision_t dec;
    memset(&dec, 0, sizeof(dec));

    long pre_benches = dp_ctx->n_benchmarks;
    double cost = stride_dp_plan_measure(dp_ctx, N, reg, &dec, /*verbose=*/0);
    long benches_used = dp_ctx->n_benchmarks - pre_benches;
    (void)cost;  /* deploy-bench below produces the wisdom-quality number */

    if (dec.fact.nfactors <= 0) {
        fprintf(stderr, "  N=%d K=%zu: MEASURE failed\n", N, K);
        return 1;
    }

    /* Build the chosen plan and re-bench with deploy-quality protocol so
     * the cost we record in wisdom v5 matches what calibrate_cell_joint
     * would have written for the same plan. */
    stride_plan_t *plan = _stride_build_plan_explicit(
            N, K, dec.fact.factors, dec.fact.nfactors,
            dec.variants, dec.use_dif_forward, reg);
    if (!plan) {
        fprintf(stderr, "  N=%d K=%zu: MEASURE winner build failed\n", N, K);
        return 1;
    }

    double deploy_ns = bench_plan_min(plan, N, K);
    double err = roundtrip_err(plan, N, K);
    stride_plan_destroy(plan);

    /* Post-MEASURE blocked refine for K <= STRIDE_BLOCKED_K_THRESHOLD
     * medium N. If blocked wins, deploy_ns is overridden and the entry
     * gets stamped with use_blocked=1. */
    int blocked_split = 0, blocked_bg = 0;
    double blocked_ns = 1e18;
    int use_blocked = try_blocked_refine(N, K, &dec, reg, deploy_ns,
                                          &blocked_split, &blocked_bg,
                                          &blocked_ns);
    if (use_blocked) {
        deploy_ns = blocked_ns;
    }

    /* Commit to wisdom v5 (explicit variant codes). Force-replace any
     * existing entry first so re-calibration runs always overwrite. */
    int code_ints[STRIDE_MAX_STAGES];
    for (int s = 0; s < dec.fact.nfactors; s++)
        code_ints[s] = (int)dec.variants[s];
    calib_remove_entry(wis, N, K);
    stride_wisdom_add_v5(wis, N, K, dec.fact.factors, dec.fact.nfactors,
                          deploy_ns,
                          use_blocked, blocked_split, blocked_bg,
                          dec.use_dif_forward,
                          /*has_variant_codes=*/1, code_ints);

    /* Fill cell_result_t for the harness's display + CSV. */
    out->method            = use_blocked ? "dp+measure+blk" : "dp+measure";
    out->variants_searched = benches_used;
    out->dit_ns            = -1;
    out->dif_ns            = -1;
    out->nfactors          = dec.fact.nfactors;
    for (int s = 0; s < dec.fact.nfactors; s++)
        out->factors[s] = dec.fact.factors[s];
    out->best_ns           = deploy_ns;
    out->use_blocked       = use_blocked;
    out->split_stage       = blocked_split;
    out->block_groups      = blocked_bg;
    out->use_dif_forward   = dec.use_dif_forward;
    for (int s = 0; s < dec.fact.nfactors; s++)
        out->variant_codes[s] = code_ints[s];

    /* Codelet display from variant codes directly. */
    out->codelets[0] = CL_N1;
    for (int s = 1; s < dec.fact.nfactors; s++) {
        switch (dec.variants[s]) {
            case VFFT_VAR_FLAT: out->codelets[s] = CL_FLAT; break;
            case VFFT_VAR_LOG3: out->codelets[s] = CL_LOG3; break;
            case VFFT_VAR_T1S:  out->codelets[s] = CL_T1S;  break;
            case VFFT_VAR_BUF:  out->codelets[s] = CL_BUF;  break;
            default:            out->codelets[s] = CL_FLAT; break;
        }
    }

    out->err = err;
    return 0;
}

/* ===========================================================================
 * MAIN
 * ========================================================================= */

typedef enum {
    CALIB_MODE_MEASURE = 0,    /* default: DP + variant cartesian */
    CALIB_MODE_EXTREME = 1,    /* full joint cartesian (PATIENT-style) */
} calib_mode_t;

typedef struct { int N; size_t K; } cell_t;

#define MAX_SELECTED_CELLS 256

/* Parse "N1:K1,N2:K2,..." into out[]. Returns count, or -1 on malformed. */
static int parse_cells_arg(const char *s, cell_t *out, int max_out) {
    int count = 0;
    while (*s && count < max_out) {
        while (*s == ' ' || *s == ',') s++;
        if (!*s) break;
        int N = 0;
        size_t K = 0;
        while (*s >= '0' && *s <= '9') N = N * 10 + (*s++ - '0');
        if (*s != ':') return -1;
        s++;
        while (*s >= '0' && *s <= '9') K = K * 10 + (*s++ - '0');
        if (N == 0 || K == 0) return -1;
        out[count].N = N;
        out[count].K = K;
        count++;
    }
    return count;
}

static int cell_cmp_by_K(const void *a, const void *b) {
    const cell_t *ca = (const cell_t *)a;
    const cell_t *cb = (const cell_t *)b;
    if (ca->K < cb->K) return -1;
    if (ca->K > cb->K) return  1;
    if (ca->N < cb->N) return -1;
    if (ca->N > cb->N) return  1;
    return 0;
}

int main(int argc, char **argv) {
    const char *out_path = "vfft_wisdom_tuned.txt";
    const char *info_csv = "vfft_wisdom_tuned_codelets.csv";
    int pace_ms_arg = DEFAULT_PACE_MS;
    calib_mode_t mode = CALIB_MODE_MEASURE;
    const char *cells_arg = NULL;

    /* Positional: out_path info_csv pace_ms mode cells
     * `mode`  = "measure" (default) | "extreme"
     * `cells` = optional "N1:K1,N2:K2,..."; if given, only those cells
     *           run, and existing wisdom for other (N, K) is preserved. */
    if (argc >= 2) out_path  = argv[1];
    if (argc >= 3) info_csv  = argv[2];
    if (argc >= 4) pace_ms_arg = atoi(argv[3]);
    if (argc >= 5) {
        if (strcmp(argv[4], "extreme") == 0)      mode = CALIB_MODE_EXTREME;
        else if (strcmp(argv[4], "measure") == 0) mode = CALIB_MODE_MEASURE;
        else {
            fprintf(stderr, "fatal: unknown mode '%s' (expected measure|extreme)\n",
                    argv[4]);
            return 2;
        }
    }
    if (argc >= 6) cells_arg = argv[5];

    cell_t selected[MAX_SELECTED_CELLS];
    int n_selected = 0;
    if (cells_arg) {
        n_selected = parse_cells_arg(cells_arg, selected, MAX_SELECTED_CELLS);
        if (n_selected <= 0) {
            fprintf(stderr, "fatal: --cells arg malformed: '%s'\n", cells_arg);
            return 2;
        }
        qsort(selected, n_selected, sizeof(cell_t), cell_cmp_by_K);
    }

    printf("=== calibrate_tuned: new-core wisdom generator ===\n");
    printf("output : %s\n", out_path);
    printf("info   : %s\n", info_csv);
    printf("pacing : %d ms between cells\n", pace_ms_arg);
    printf("mode   : %s\n",
           mode == CALIB_MODE_EXTREME ? "extreme (full joint cartesian)"
                                       : "measure (DP + variant cartesian)");
    if (n_selected > 0) {
        printf("cells  : %d selected (full grid bypassed)\n", n_selected);
    }

    stride_registry_t reg;
    stride_registry_init(&reg);

    stride_wisdom_t wis;
    stride_wisdom_init(&wis);

    /* Always load existing wisdom at startup so a mid-run interruption
     * doesn't overwrite the file with a partial snapshot. Full-grid mode
     * still re-calibrates every cell (calib_remove_entry forces overwrite
     * before each wisdom_add); selective mode preserves un-touched cells. */
    if (stride_wisdom_load(&wis, out_path) >= 0 && wis.count > 0) {
        printf("loaded : %d existing entries from %s\n", wis.count, out_path);
    }

    FILE *info = fopen(info_csv, "w");
    if (!info) {
        fprintf(stderr, "fatal: cannot open %s for writing\n", info_csv);
        return 2;
    }
    fprintf(info, "N,K,method,nf,factors,codelets,variant_codes,best_ns,"
                  "use_blocked,split_stage,block_groups,use_dif_forward,"
                  "dit_ns,dif_ns,variants_searched,roundtrip_err\n");

    int max_N = 0;
    for (size_t i = 0; i < sizeof(GRID_N)/sizeof(GRID_N[0]); i++)
        if (GRID_N[i] > max_N) max_N = GRID_N[i];

    /* Unified cell list: either selected via --cells, or full GRID_N x GRID_K.
     * Sorted by K so we can keep one DP context per K group (memoization
     * reuse across same-K cells). */
    cell_t all_cells[MAX_SELECTED_CELLS + 256];   /* full grid worst case */
    int n_all = 0;
    if (n_selected > 0) {
        for (int i = 0; i < n_selected; i++) all_cells[n_all++] = selected[i];
    } else {
        for (size_t ki = 0; ki < sizeof(GRID_K)/sizeof(GRID_K[0]); ki++) {
            for (size_t ni = 0; ni < sizeof(GRID_N)/sizeof(GRID_N[0]); ni++) {
                if (n_all >= (int)(sizeof(all_cells)/sizeof(all_cells[0]))) break;
                all_cells[n_all].N = GRID_N[ni];
                all_cells[n_all].K = GRID_K[ki];
                n_all++;
            }
        }
        /* Already K-grouped by construction, but explicit qsort is cheap and
         * keeps the invariant local. */
        qsort(all_cells, n_all, sizeof(cell_t), cell_cmp_by_K);
    }

    int done = 0;
    int failures = 0;
    int n_cells = n_all;

    /* DP context: one per K group, reused across same-K cells. */
    size_t prev_K = (size_t)-1;
    stride_dp_context_t dp_ctx;
    int ctx_active = 0;

    for (int i = 0; i < n_all; i++) {
        int    N = all_cells[i].N;
        size_t K = all_cells[i].K;

        if (K != prev_K) {
            if (ctx_active) stride_dp_destroy(&dp_ctx);
            stride_dp_init(&dp_ctx, K, max_N);
            ctx_active = 1;
            prev_K = K;
        }

        /* Pace before each cell (except the very first) so thermal
         * and cache state from the prior cell don't bias this one's
         * search measurements. */
        if (done > 0) pace_ms(pace_ms_arg);

        cell_result_t r;
        memset(&r, 0, sizeof(r));
        int rc;
        if (mode == CALIB_MODE_EXTREME) {
            rc = calibrate_cell_joint(N, K, &reg, &wis, &r);
        } else {
            rc = calibrate_cell_measure(N, K, &reg, &dp_ctx, &wis, &r);
        }
        done++;
        if (rc != 0) {
            failures++;
            continue;
        }

        /* Print human-readable progress */
        printf("[%2d/%2d] N=%-5d K=%-3zu method=%-14s factors=",
               done, n_cells, N, K, r.method);
        for (int s = 0; s < r.nfactors; s++)
            printf("%s%d", s ? "x" : "", r.factors[s]);
        printf("  codelets=");
        for (int s = 0; s < r.nfactors; s++)
            printf("%s%s", s ? "/" : "", codelet_short(r.codelets[s]));
        if (r.use_blocked)
            printf("  BLOCKED@split=%d,bg=%d", r.split_stage, r.block_groups);
        if (r.use_dif_forward) printf("  DIF");
        printf("  best=%.1f ns", r.best_ns);
        if (r.variants_searched > 0) {
            if (r.dit_ns > 0) {
                printf("  (DIT_baseline=%.1f, %ld variants searched, best speedup=%.3f)",
                       r.dit_ns, r.variants_searched,
                       r.best_ns > 0 ? r.dit_ns / r.best_ns : 1.0);
            } else {
                printf("  (%ld variants searched)", r.variants_searched);
            }
        }
        printf("  err=%.2e %s\n",
               r.err, r.err < 1e-12 ? "" : "[PRECISION FAIL]");
        fflush(stdout);

        if (r.err >= 1e-12) failures++;

        /* Sidecar CSV. */
        fprintf(info, "%d,%zu,%s,%d,", N, K, r.method, r.nfactors);
        for (int s = 0; s < r.nfactors; s++)
            fprintf(info, "%s%d", s ? "x" : "", r.factors[s]);
        fprintf(info, ",");
        for (int s = 0; s < r.nfactors; s++)
            fprintf(info, "%s%s", s ? "/" : "", codelet_short(r.codelets[s]));
        fprintf(info, ",");
        for (int s = 0; s < r.nfactors; s++)
            fprintf(info, "%s%d", s ? "/" : "", r.variant_codes[s]);
        fprintf(info, ",%.2f,%d,%d,%d,%d,%.2f,%.2f,%ld,%.2e\n",
                r.best_ns,
                r.use_blocked, r.split_stage, r.block_groups,
                r.use_dif_forward,
                r.dit_ns,
                r.dif_ns >= 0 ? r.dif_ns : 0.0,
                r.variants_searched,
                r.err);
        fflush(info);

        /* Incremental save — crash-resilient. */
        int srv = stride_wisdom_save(&wis, out_path);
        if (srv != 0) {
            fprintf(stderr,
                    "warn: incremental stride_wisdom_save(%s) failed at "
                    "N=%d K=%zu (continuing)\n", out_path, N, K);
        }
    }

    if (ctx_active) stride_dp_destroy(&dp_ctx);

    fclose(info);

    /* Final save — redundant after incremental, but kept as safety net. */
    int srv = stride_wisdom_save(&wis, out_path);
    if (srv != 0) {
        fprintf(stderr, "fatal: final stride_wisdom_save(%s) failed\n", out_path);
        return 3;
    }

    printf("===\n");
    printf("wrote %d entries to %s\n", wis.count, out_path);
    printf("wrote per-cell codelet info to %s\n", info_csv);
    if (failures) printf("WARNING: %d cell(s) had failures\n", failures);
    printf("done.\n");
    return failures ? 1 : 0;
}
