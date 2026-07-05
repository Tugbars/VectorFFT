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
    /* Odd / non-multiple-of-8 K is served by MODEB only: MODEB rides the
     * in-place codelets (rem-aware tail, docs/performance/arbitrary_k_tail_handling.md),
     * so it handles any K. The native LEAF/BAILEY2 kernels (codelet_oop family)
     * stay K%8-only — their own creators return NULL below for odd K, so the
     * native path falls through to MODEB automatically. */
    if (K == 0)
        return NULL;

    /* Native OOP fast paths first (LEAF at N<=128, then rule-spine BAILEY2).
     * Returns NULL for K%8!=0 — those kernels are vector-lane-only. */
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
    /* MODEB handles any K (rides the tailed in-place codelets). */
    if (K == 0 || !dp || !reg || dp->K != K)
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
/* Build BOTH OOP champions for (N,K) and time each (rdtsc min-of-9) — the raw material for the
 * order axis AND the DEFAULT joint pick. *out_nat = the native champion (LEAF or best BAILEY2 pair =
 * NATURAL order); *out_mb = the DP-MODEB champion (SCRAMBLED order). Either may be NULL (e.g. no
 * native candidate at odd K). *out_*_ns = measured cycles (comparable on one clock; 1e30 if a
 * champion is absent or timing OOMs). The caller persists each present champion as its own
 * (N,K,kind-class) wisdom cell, so every config.order is served from wisdom without re-tuning.
 * NOTE: unlike the old create_dp_best, no LEAF short-circuit — LEAF is MEASURED against MODEB so the
 * create-time pick (min-ns) matches the wisdom-lookup pick (lookup_ord min-ns). LEAF still wins its
 * small-N cells on time; the extra MODEB build there is cheap and one-off. */
static inline void vfft_oop_plan_create_champions(
    int N, size_t K, vfft_proto_dp_context_t *dp, const vfft_proto_registry_t *reg,
    vfft_oop_plan_t **out_nat, double *out_nat_ns,
    vfft_oop_plan_t **out_mb, double *out_mb_ns)
{
    *out_nat = NULL; *out_mb = NULL; *out_nat_ns = 1e30; *out_mb_ns = 1e30;
    if (K == 0) return;
    /* native champion: tuner picks LEAF or the best BAILEY2 pair + t1p variant (nc==0 for odd K). */
    int r1 = 0, r2 = 0, t1p = 1;
    int nc = vfft_oop_tune_pairs_v(N, K, &r1, &r2, &t1p, 0);
    vfft_oop_plan_t *nat = NULL;
    if (nc > 0)
        nat = (r1 == 0) ? vfft_oop_plan_create(N, K, NULL, 0, reg)
                        : vfft_oop_plan_create_pair_v(N, K, r1, r2, t1p);
    /* MODEB champion: DP's best multi-factor decomposition. */
    vfft_oop_plan_t *mb = vfft_oop_plan_create_dp_modeb(N, K, dp, reg);
    *out_nat = nat; *out_mb = mb;
    if (!nat && !mb) return;
    size_t T = (size_t)N * K;
    double *sr = (double *)VFFT_OOP_AALLOC(T * 8), *si = (double *)VFFT_OOP_AALLOC(T * 8);
    double *dr = (double *)VFFT_OOP_AALLOC(T * 8), *di = (double *)VFFT_OOP_AALLOC(T * 8);
    if (!sr || !si || !dr || !di) {
        VFFT_OOP_AFREE(sr); VFFT_OOP_AFREE(si); VFFT_OOP_AFREE(dr); VFFT_OOP_AFREE(di);
        return;                                   /* OOM: leave ns at 1e30 — caller still persists */
    }
    for (size_t i = 0; i < T; i++) {
        sr[i] = (double)(i % 251) * 0.013 - 1.6;
        si[i] = (double)(i % 257) * 0.011 - 1.4;
    }
    if (nat) {
        vfft_oop_execute_fwd(nat, sr, si, dr, di);            /* warm */
        unsigned long long bn = ~0ULL;
        for (int r = 0; r < 9; r++) {
            unsigned long long t0 = __rdtsc();
            vfft_oop_execute_fwd(nat, sr, si, dr, di);
            unsigned long long a = __rdtsc() - t0; if (a < bn) bn = a;
        }
        *out_nat_ns = (double)bn;
    }
    if (mb) {
        vfft_oop_execute_fwd(mb, sr, si, dr, di);             /* warm */
        unsigned long long bm = ~0ULL;
        for (int r = 0; r < 9; r++) {
            unsigned long long t0 = __rdtsc();
            vfft_oop_execute_fwd(mb, sr, si, dr, di);
            unsigned long long b = __rdtsc() - t0; if (b < bm) bm = b;
        }
        *out_mb_ns = (double)bm;
    }
    VFFT_OOP_AFREE(sr); VFFT_OOP_AFREE(si); VFFT_OOP_AFREE(dr); VFFT_OOP_AFREE(di);
}

/* DEFAULT (order-agnostic) joint chooser: both champions, keep the faster. Thin over champions(). */
static inline vfft_oop_plan_t *vfft_oop_plan_create_dp_best(
    int N, size_t K, vfft_proto_dp_context_t *dp,
    const vfft_proto_registry_t *reg)
{
    vfft_oop_plan_t *nat = NULL, *mb = NULL; double nns = 1e30, mns = 1e30;
    vfft_oop_plan_create_champions(N, K, dp, reg, &nat, &nns, &mb, &mns);
    if (nat && mb) {
        if (nns <= mns) { vfft_oop_plan_destroy(mb); return nat; }
        vfft_oop_plan_destroy(nat); return mb;
    }
    return nat ? nat : mb;
}

#endif /* VFFT_OOP_DP_H */
