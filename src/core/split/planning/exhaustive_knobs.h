/* exhaustive_knobs.h — the split EXHAUSTIVE planner's depth / prune knobs.
 *
 * Carved verbatim out of support/env.h (its part 2) in layout separation phase 5:
 * env.h part 1 (FTZ/DAZ, allocation, pinning, version) is neutral and lives in
 * common/support/; these knobs are read by split/planning/exhaustive_plan.h only. */
#ifndef VFFT_EXHAUSTIVE_KNOBS_H
#define VFFT_EXHAUSTIVE_KNOBS_H

#include <stdlib.h>

/* ===========================================================================
 * PART 2 — EXHAUSTIVE SEARCH TUNING KNOBS
 *
 *  VALIDATION SCOPE: the defaults were tuned on N=1024 K=4 only (i9-14900KF).
 *  There an unpruned run (262,428 candidates: all 35 decompositions x all
 *  orderings x all variants) found latency monotonically worse with stage
 *  count (2-stage 64x16 = 3.2us; 10-stage = 13.6us), and the depth-5 cap +
 *  2x pre-screen reached the same winner with 32x fewer candidates.
 *  Unvalidated elsewhere: larger pow2 may want 4-5 stages (depth-5 could
 *  clip), and many-small-prime N may find the 2x prune too aggressive.
 *  To re-check a size class, run the unpruned sweep and compare the
 *  best-per-stage-count curve (env overrides, no recompile):
 *      VFFT_PROTO_EXH_MAX_DEPTH=16  VFFT_PROTO_EXH_PRUNE=1e9
 * ===========================================================================
 */

/* Stage-depth cap (exhaustive enumeration). Pow2 optima are shallow: each
 * extra stage is one more full memory pass. Non-pow2 needs deeper plans.
 * Override: VFFT_PROTO_EXH_MAX_DEPTH. */
#ifndef VFFT_PROTO_EXH_MAX_DEPTH_POW2
#define VFFT_PROTO_EXH_MAX_DEPTH_POW2    5
#endif
#ifndef VFFT_PROTO_EXH_MAX_DEPTH_NONPOW2
#define VFFT_PROTO_EXH_MAX_DEPTH_NONPOW2 9
#endif

/* Variant pre-screen factor: skip a factorization's variant cartesian when its
 * default-variant bench > FACTOR x the running global best.
 * Override: VFFT_PROTO_EXH_PRUNE (set huge, e.g. 1e9, to disable). */
#ifndef VFFT_PROTO_EXH_PRUNE_FACTOR
#define VFFT_PROTO_EXH_PRUNE_FACTOR 2.0
#endif

/* A third cap, the per-decomposition permutation limit VFFT_PROTO_DP_MAX_PERMS
 * (720), lives in dp_planner.h; it can clip orderings for non-pow2 N with many
 * distinct small primes. */

/* Accessors: default unless the matching env var overrides. hard_cap clamps to
 * the stage-array bound (pass STRIDE_MAX_STAGES); 0 = no clamp. */
static inline int vfft_proto_env_max_depth(int n_is_pow2, int hard_cap) {
    int d = n_is_pow2 ? VFFT_PROTO_EXH_MAX_DEPTH_POW2
                      : VFFT_PROTO_EXH_MAX_DEPTH_NONPOW2;
    const char *e = getenv("VFFT_PROTO_EXH_MAX_DEPTH");
    if (e) { int v = atoi(e); if (v > 0) d = v; }
    if (hard_cap > 0 && d > hard_cap) d = hard_cap;
    return d;
}

static inline double vfft_proto_env_prune_factor(void) {
    double p = VFFT_PROTO_EXH_PRUNE_FACTOR;
    const char *e = getenv("VFFT_PROTO_EXH_PRUNE");
    if (e) { double v = atof(e); if (v > 0) p = v; }
    return p;
}

/* Wisdom-write overwrite flag, shared by the calibrator (patient/measure) and
 * the planner write paths. 0 = preserve already-calibrated cells (skip them;
 * only fill in missing ones) — the safe incremental default. 1 = re-calibrate
 * and overwrite the cell with the new winner (vfft_proto_wisdom_add collapses to
 * one entry). Override: VFFT_PROTO_WISDOM_OVERWRITE=1. */
#ifndef VFFT_PROTO_WISDOM_OVERWRITE
#define VFFT_PROTO_WISDOM_OVERWRITE 0
#endif

static inline int vfft_proto_env_wisdom_overwrite(void) {
    int v = VFFT_PROTO_WISDOM_OVERWRITE;
    const char *e = getenv("VFFT_PROTO_WISDOM_OVERWRITE");
    if (e && *e) v = atoi(e);
    return v;
}

#endif /* VFFT_EXHAUSTIVE_KNOBS_H */
