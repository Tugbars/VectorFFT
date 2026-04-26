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

#ifndef CALIB_PILOT_JOINT
  #define CALIB_PILOT_JOINT 1
#endif

#if CALIB_PILOT_JOINT
  /* Pilot cells. Selected to overlap with production wisdom cells in
   * src/stride-fft/bench/vfft_wisdom.txt so ab_compare can show diffs.
   * N=256 and N=1024 keep the joint search bounded — N=2048 with K=256
   * triples the bench count (longer plans → more permutations × variant
   * cartesian). Add larger N to GRID_N for more coverage once the pilot
   * shape is validated. */
  static const int   GRID_N[] = { 256, 1024 };
  static const size_t GRID_K[] = { 4, 256 };
  /* No DP path used in joint mode; threshold is ignored. Set high to be safe. */
  #define EXHAUSTIVE_MAX_N 65536
#else
  static const int   GRID_N[] = { 64, 128, 256, 512, 1024, 2048,
                                  4096, 8192, 16384, 32768 };
  static const size_t GRID_K[] = { 4, 32, 128, 256 };
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
 * MAIN
 * ========================================================================= */

int main(int argc, char **argv) {
    const char *out_path = "vfft_wisdom_tuned.txt";
    const char *info_csv = "vfft_wisdom_tuned_codelets.csv";
    int pace_ms_arg = DEFAULT_PACE_MS;
    if (argc >= 2) out_path = argv[1];
    if (argc >= 3) info_csv = argv[2];
    if (argc >= 4) pace_ms_arg = atoi(argv[3]);

    printf("=== calibrate_tuned: new-core wisdom generator ===\n");
    printf("output : %s\n", out_path);
    printf("info   : %s\n", info_csv);
    printf("pacing : %d ms between cells\n", pace_ms_arg);

    stride_registry_t reg;
    stride_registry_init(&reg);

    stride_wisdom_t wis;
    stride_wisdom_init(&wis);

    FILE *info = fopen(info_csv, "w");
    if (!info) {
        fprintf(stderr, "fatal: cannot open %s for writing\n", info_csv);
        return 2;
    }
    fprintf(info, "N,K,method,nf,factors,codelets,variant_codes,best_ns,"
                  "use_blocked,split_stage,block_groups,use_dif_forward,"
                  "dit_ns,dif_ns,variants_searched,roundtrip_err\n");

    int n_cells = (int)(sizeof(GRID_N)/sizeof(GRID_N[0]) *
                        sizeof(GRID_K)/sizeof(GRID_K[0]));
    int done = 0;
    int failures = 0;

    /* DP context per K — max_N = max value in GRID_N to size the buffers
     * once. Re-using the context across cells with same K lets DP cache
     * sub-problem solutions, which is the entire point of DP. */
    int max_N = 0;
    for (size_t i = 0; i < sizeof(GRID_N)/sizeof(GRID_N[0]); i++)
        if (GRID_N[i] > max_N) max_N = GRID_N[i];

    for (size_t ki = 0; ki < sizeof(GRID_K)/sizeof(GRID_K[0]); ki++) {
        size_t K = GRID_K[ki];

        stride_dp_context_t dp_ctx;
        stride_dp_init(&dp_ctx, K, max_N);

        for (size_t ni = 0; ni < sizeof(GRID_N)/sizeof(GRID_N[0]); ni++) {
            int N = GRID_N[ni];

            /* Pace before each cell (except the very first) so thermal
             * and cache state from the prior cell don't bias this one's
             * search measurements. */
            if (done > 0) pace_ms(pace_ms_arg);

            cell_result_t r;
            memset(&r, 0, sizeof(r));
#if CALIB_PILOT_JOINT
            int rc = calibrate_cell_joint(N, K, &reg, &wis, &r);
            (void)dp_ctx;  /* unused in joint mode */
#else
            int rc = calibrate_cell(N, K, &reg, &dp_ctx, &wis, &r);
#endif
            done++;
            if (rc != 0) {
                failures++;
                continue;
            }

            /* Print human-readable progress */
            printf("[%2d/%2d] N=%-5d K=%-3zu method=%-10s factors=",
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
            if (r.variants_searched > 0)
                printf("  (DIT_baseline=%.1f, %ld variants searched, best speedup=%.3f)",
                       r.dit_ns, r.variants_searched,
                       r.best_ns > 0 ? r.dit_ns / r.best_ns : 1.0);
            printf("  err=%.2e %s\n",
                   r.err, r.err < 1e-12 ? "" : "[PRECISION FAIL]");
            fflush(stdout);

            if (r.err >= 1e-12) failures++;

            /* Sidecar CSV: per-stage codelets, variant codes, blocked, DIF,
             * search budget so the merge script has full visibility. */
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
        }

        stride_dp_destroy(&dp_ctx);
    }

    fclose(info);

    int srv = stride_wisdom_save(&wis, out_path);
    if (srv != 0) {
        fprintf(stderr, "fatal: stride_wisdom_save(%s) failed\n", out_path);
        return 3;
    }

    printf("===\n");
    printf("wrote %d entries to %s\n", wis.count, out_path);
    printf("wrote per-cell codelet info to %s\n", info_csv);
    if (failures) printf("WARNING: %d cell(s) had failures\n", failures);
    printf("done.\n");
    return failures ? 1 : 0;
}
