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
        {
            p = (vfft_oop_plan_t *)calloc(1, sizeof(*p));
            if (!p)
                return NULL;
            p->N = N;
            p->K = K;
            p->mb = vfft_proto_plan_create(N, K, best.factors, best.variants,
                                           best.nfactors,
                                           (vfft_proto_registry_t *)reg);
            if (p->mb && !p->mb->use_dif_forward)
            {
                p->kind = VFFT_OOP_KIND_MODEB;
                return p;
            }
            free(p);
        }
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
    vfft_oop_plan_t *p = (vfft_oop_plan_t *)calloc(1, sizeof(*p));
    if (!p)
        return NULL;
    p->N = N;
    p->K = K;
    p->mb = vfft_proto_plan_create(N, K, best.factors, best.variants,
                                   best.nfactors, (vfft_proto_registry_t *)reg);
    if (p->mb && !p->mb->use_dif_forward)
    {
        p->kind = VFFT_OOP_KIND_MODEB;
        return p;
    }
    free(p->mb);
    free(p);
    return NULL;
}

/* Measure-and-keep-faster: build the rule plan (LEAF/BAILEY2) AND the DP-MODEB
 * plan, time both same-binary round-robin (min-of-rounds), return the faster and
 * destroy the loser. LEAF short-circuits (direct single codelet — effectively
 * always best at N<=128, no point measuring a multi-stage MODEB against it).
 *
 * This is the CALIBRATION-TIME chooser: it resolves the K-dependent kind choice
 * by measurement (e.g. N=1024 — BAILEY2 32x32 wins at K=120, but MODEB 4^5 wins
 * at K=256, where the only unmasked BAILEY2 pair aliases). Cache its verdict in
 * OOP wisdom so the runtime path is a pure lookup with no measurement. */
static inline vfft_oop_plan_t *vfft_oop_plan_create_dp_best(
    int N, size_t K, vfft_proto_dp_context_t *dp,
    const vfft_proto_registry_t *reg)
{
    if (K == 0 || (K % 8u) != 0)
        return NULL;
    vfft_oop_plan_t *rule = vfft_oop_plan_create(N, K, NULL, 0, reg);
    if (rule && rule->kind == VFFT_OOP_KIND_LEAF)
        return rule;                                  /* LEAF: no contest */
    vfft_oop_plan_t *modeb = vfft_oop_plan_create_dp_modeb(N, K, dp, reg);
    if (!modeb) return rule;                          /* only the rule plan exists */
    if (!rule)  return modeb;                         /* only MODEB exists */

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
    double br = 1e18, bm = 1e18;
    for (int r = 0; r < 9; r++) {
        double t0 = vfft_proto_now_ns();
        vfft_oop_execute_fwd(rule, sr, si, dr, di);
        double a = vfft_proto_now_ns() - t0; if (a < br) br = a;
        t0 = vfft_proto_now_ns();
        vfft_oop_execute_fwd(modeb, sr, si, dr, di);
        double b = vfft_proto_now_ns() - t0; if (b < bm) bm = b;
    }
    VFFT_OOP_AFREE(sr); VFFT_OOP_AFREE(si); VFFT_OOP_AFREE(dr); VFFT_OOP_AFREE(di);
    if (br <= bm) { vfft_oop_plan_destroy(modeb); return rule; }
    vfft_oop_plan_destroy(rule); return modeb;
}

#endif /* VFFT_OOP_DP_H */
