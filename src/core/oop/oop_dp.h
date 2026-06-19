/* oop_dp.h — DP-planner-backed OOP c2c plan creation.
 *
 * Wires the recursive DP planner (dp_planner.h — FFTW-PATIENT-style measured
 * factorization search with sub-problem memoization) into the OOP c2c path as
 * the MODEB factorization source, replacing the wisdom-file lookup with an
 * on-the-fly measured plan.
 *
 * Rule order mirrors vfft_oop_plan_create: LEAF -> rule-spine BAILEY2 -> MODEB,
 * but MODEB's factorization now comes from vfft_proto_dp_plan rather than a
 * wisdom entry. So a host with no wisdom file still gets a measured-optimal
 * general-N MODEB plan.
 *
 * Ownership / contract:
 *   - The DP context is CALLER-OWNED and amortized across calls (it caches
 *     sub-problem solutions). It MUST be init'd with the SAME K as the plan
 *     and a max_N >= N: vfft_proto_dp_init(&ctx, K, maxN). The planner measures
 *     at ctx->K, so a K mismatch is rejected (returns NULL via the rule path).
 *   - This header pulls in dp_planner.h (heavy: executor + planner). Include it
 *     only in consumers that want the DP path; default OOP consumers use
 *     oop_auto.h and pay no DP dependency.
 *
 * The DP planner builds DIT plans (vfft_proto_plan_create, use_dif_forward=0),
 * which is exactly MODEB's requirement.
 */
#ifndef VFFT_OOP_DP_H
#define VFFT_OOP_DP_H

#include "oop_auto.h"
#include "dp_planner.h"

/* Build an OOP plan, using the DP planner for the MODEB (general-N) fallback.
 * `dp` must be init'd with the same K (vfft_proto_dp_init(dp, K, >=N)). Returns
 * NULL if nothing covers (N, K). */
static inline vfft_oop_plan_t *vfft_oop_plan_create_dp(
    int N, size_t K, vfft_proto_dp_context_t *dp,
    const vfft_proto_registry_t *reg)
{
    if (K == 0 || (K % 8u) != 0)
        return NULL;

    /* Native OOP fast paths first (LEAF at N<=128, then rule-spine BAILEY2). */
    vfft_oop_plan_t *p = vfft_oop_plan_create(N, K, NULL, 0, reg);
    if (p)
        return p;

    /* MODEB via DP-measured factorization (no wisdom file needed). */
    if (dp && reg && dp->K == K)
    {
        vfft_proto_factorization_t best;
        double ns = vfft_proto_dp_plan(dp, N, reg, &best, 0);
        if (ns < 1e17 && best.nfactors > 0)
            return _vfft_oop_make_modeb(N, K, best.factors, best.variants,
                                        best.nfactors, reg);
    }
    return NULL;
}

/* Force the DP-MODEB path (skip LEAF/BAILEY2) — for A/B against the native OOP
 * kernels, e.g. to check whether DP-MODEB beats a poor aliasing-masked BAILEY2
 * pair on a given cell. Returns NULL if DP can't plan (N, K). */
static inline vfft_oop_plan_t *vfft_oop_plan_create_dp_modeb(
    int N, size_t K, vfft_proto_dp_context_t *dp,
    const vfft_proto_registry_t *reg)
{
    if (K == 0 || (K % 8u) != 0 || !dp || !reg || dp->K != K)
        return NULL;
    vfft_proto_factorization_t best;
    double ns = vfft_proto_dp_plan(dp, N, reg, &best, 0);
    if (ns >= 1e17 || best.nfactors <= 0)
        return NULL;
    return _vfft_oop_make_modeb(N, K, best.factors, best.variants,
                                best.nfactors, reg);
}

/* The full 2-axis joint chooser (CALIBRATION-TIME):
 *   Axis 2 (factorization within a kind):
 *     - native champion = the TUNER's best of {LEAF, all unmasked BAILEY2 pairs}
 *       (vfft_oop_tune_pairs measures them same-binary).
 *     - MODEB champion  = the DP planner's best multi-factor decomposition.
 *   Axis 1 (kind): measure the two champions round-robin, return the faster.
 * LEAF short-circuits (direct single codelet — always best at its N).
 *
 * This resolves the K-dependent kind choice by measurement (N=1024: BAILEY2
 * 32x32 wins at K=120; MODEB 4^5 wins at K=256, where every unmasked BAILEY2
 * pair aliases). Cache its verdict — (N,K) -> {kind, factorization} — in OOP
 * wisdom so the runtime path is a pure lookup with no measurement. */
static inline vfft_oop_plan_t *vfft_oop_plan_create_dp_best(
    int N, size_t K, vfft_proto_dp_context_t *dp,
    const vfft_proto_registry_t *reg)
{
    if (K == 0 || (K % 8u) != 0)
        return NULL;

    /* Axis 2 within the native kinds: tuner picks LEAF or the best BAILEY2 pair. */
    int r1 = 0, r2 = 0;
    int nc = vfft_oop_tune_pairs(N, K, &r1, &r2, 0);
    vfft_oop_plan_t *native = NULL;
    if (nc > 0) {
        if (r1 == 0)                              /* LEAF won the tuner */
            native = vfft_oop_plan_create(N, K, NULL, 0, reg);
        else                                      /* best BAILEY2 pair */
            native = vfft_oop_plan_create_pair(N, K, r1, r2);
    }
    if (native && native->kind == VFFT_OOP_KIND_LEAF)
        return native;                            /* LEAF: no contest */

    /* Axis 2 within MODEB: DP's best multi-factor decomposition. */
    vfft_oop_plan_t *modeb = vfft_oop_plan_create_dp_modeb(N, K, dp, reg);
    if (!modeb) return native;                    /* only the native plan exists */
    if (!native) return modeb;                    /* only MODEB exists */

    /* Axis 1: time the two champions, keep the faster. */
    vfft_oop_plan_t *rule = native;               /* (alias kept for the loop below) */
    size_t T = (size_t)N * K;
    double *sr = (double *)VFFT_OOP_AALLOC(T * 8), *si = (double *)VFFT_OOP_AALLOC(T * 8);
    double *dr = (double *)VFFT_OOP_AALLOC(T * 8), *di = (double *)VFFT_OOP_AALLOC(T * 8);
    if (!sr || !si || !dr || !di) {
        VFFT_OOP_AFREE(sr); VFFT_OOP_AFREE(si); VFFT_OOP_AFREE(dr); VFFT_OOP_AFREE(di);
        vfft_oop_plan_destroy(modeb); return rule;    /* OOM: fall back to the rule */
    }
    for (size_t i = 0; i < T; i++) {
        sr[i] = (double)(i % 251) * 0.013 - 1.6;
        si[i] = (double)(i % 257) * 0.011 - 1.4;
    }
    vfft_oop_execute_fwd(rule,  sr, si, dr, di);       /* warm both */
    vfft_oop_execute_fwd(modeb, sr, si, dr, di);
    /* __rdtsc cycles, min-of-9 — same timer the pair tuner uses (axis-2 native),
     * so both halves of the joint chooser measure on one clock. */
    unsigned long long br = ~0ULL, bm = ~0ULL;
    for (int r = 0; r < 9; r++) {
        unsigned long long t0 = __rdtsc();
        vfft_oop_execute_fwd(rule, sr, si, dr, di);
        unsigned long long a = __rdtsc() - t0; if (a < br) br = a;
        t0 = __rdtsc();
        vfft_oop_execute_fwd(modeb, sr, si, dr, di);
        unsigned long long b = __rdtsc() - t0; if (b < bm) bm = b;
    }
    VFFT_OOP_AFREE(sr); VFFT_OOP_AFREE(si); VFFT_OOP_AFREE(dr); VFFT_OOP_AFREE(di);
    if (br <= bm) { vfft_oop_plan_destroy(modeb); return rule; }
    vfft_oop_plan_destroy(rule); return modeb;
}

#endif /* VFFT_OOP_DP_H */
