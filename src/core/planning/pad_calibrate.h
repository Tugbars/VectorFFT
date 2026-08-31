/* pad_calibrate.h - the pad-vs-tail calibrator.
 *
 * Extracted from vfft.c as migration step 13; see
 * docs/design/refactor_migration_plan.md.
 *
 * THE DECISION
 * ------------
 * A batch of K transforms is processed VW lanes at a time. When K is not a
 * multiple of VW there is a ragged tail, and there are two ways to pay for it:
 *
 *   TIGHT   run the exact (N, K) and handle the leftover 1..VW-1 lanes with a
 *           narrow path - one scalar lane at rem==1, one masked vector pass at
 *           rem>=2.
 *   PADDED  round K up to Kp = roundup(K, VW) and compute the wasted lanes, so
 *           every iteration stays full width.
 *
 * PADDED does arithmetic it will throw away; TIGHT pays for a narrower loop.
 * Which wins depends on how large the tail is relative to the body, so it is
 * raced rather than ruled, and the verdict banks as pad_me=.
 *
 * THE 3% LEAN TOWARD TIGHT IS DELIBERATE
 * --------------------------------------
 * Hysteresis points at the TIGHT arm, which is the always-safe one: it computes
 * exactly the lanes the caller asked for. On a near-tie a measurement wobble
 * should not push the plan into computing lanes it will discard, so the
 * challenger has to win by a margin rather than by noise.
 *
 * WHY THIS CALIBRATOR NEEDED A PRECONDITION BEFORE IT COULD BE MOVED
 * ------------------------------------------------------------------
 * _calibrate_pad contains NO clock call of its own - it times through the
 * _pad_burst helper. A race census that enumerates clock calls therefore misses
 * it entirely, and a 244-line racer would have moved unaudited. The migration
 * plan makes its presence in the census a hard precondition for this step for
 * exactly that reason; the census enumerator resolves helper callers as a
 * second pass so that it appears.
 *
 * _VFFT_PADVW LIVES HERE, AND ONE CALLER OUTSIDE STILL USES IT
 * -----------------------------------------------------------
 * The lane-width macro moved with the group that defines the padding rule.
 * _pad_stride_c2c, which stays in vfft.c because it dereferences vfft_plan_s
 * and touches wisdom, still reads it - which is fine, since this header is
 * included far above that call site.
 *
 * FLOOR-LEGAL BY CONSTRUCTION
 * ---------------------------
 * No vfft_plan_s, no wisdom - the caller banks. It increments the shared
 * create-race counter, declared extern below and defined once in vfft.c (a
 * tentative definition with external linkage, deliberately not a static: a
 * static in a header is one copy per includer, and the accessor would then read
 * a different object than the increment writes).
 */
#ifndef VFFT_PLANNING_PAD_CALIBRATE_H
#define VFFT_PLANNING_PAD_CALIBRATE_H

#include <stdlib.h>
#include <string.h>

#include "planner.h"                /* the proto plan + its create/destroy */
#include "executor.h"               /* vfft_proto_execute_fwd */
#include "support/race_timing.h"    /* the shared clock */

/* Defined in vfft.c (tentative definition, external linkage). */
extern long _vfft_create_race_count;

/* TIGHT-vs-PADDED A/B for misaligned K: race te at stride K (me=K) against ae at stride Kp
 * (me=Kp), alternating order + median; returns Kp/K (3% toward K), or 0 if the winner fails roundtrip. */
#define _VFFT_PADVW 4
static int _pad_dcmp(const void *a, const void *b)
{
    double d = *(const double *)a - *(const double *)b;
    return d < 0 ? -1 : d > 0 ? 1
                              : 0;
}
static double _pad_med(double *v, int n)
{
    qsort(v, n, sizeof(double), _pad_dcmp);
    return n & 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
static void _pad_fill(double *re, double *im, int N, size_t K, size_t Kp)
{
    srand(7 + N + (int)K);
    for (size_t e = 0; e < (size_t)N; e++)
        for (size_t b = 0; b < Kp; b++)
        {
            re[e * Kp + b] = (b < K) ? (double)rand() / RAND_MAX - 0.5 : 0.0;
            im[e * Kp + b] = (b < K) ? (double)rand() / RAND_MAX - 0.5 : 0.0;
        }
}
static double _pad_burst(stride_plan_t *p, vfft_proto_exec_fn jf, double *re, double *im, size_t me, int reps)
{
    double t0 = vfft_proto_now_ns();
    if (jf)
        for (int i = 0; i < reps; i++)
            jf(p, re, im, me, p->K, 0);
    else
        for (int i = 0; i < reps; i++)
            vfft_proto_execute_fwd(p, re, im, me);
    return vfft_proto_now_ns() - t0;
}
static int _calibrate_pad(int N, size_t K, vfft_rigor_t rigor, const vfft_proto_registry_t *reg,
                          const vfft_proto_wisdom_entry_t *te, const vfft_proto_wisdom_entry_t *ae)
{
    _vfft_create_race_count++;   /* HARNESS: this racer is about to time */
    if (!te || te->nf <= 0 || !ae || ae->nf <= 0)
        return 0;
    size_t Kp = (K + (size_t)(_VFFT_PADVW - 1)) & ~(size_t)(_VFFT_PADVW - 1);
    if (Kp == K)
        return (int)K; /* already aligned: nothing to decide */

    /* TIGHT arm at stride K (te = the (N,K) factorization), PADDED arm at
     * stride Kp (ae = the aligned (N,Kp) factorization). Different strides —
     * that is the whole point of the comparison. */
    stride_plan_t *pT = vfft_proto_plan_create_ex(N, K, te->factors, te->variants, te->nf, te->use_dif_forward, reg);
    stride_plan_t *pP = vfft_proto_plan_create_ex(N, Kp, ae->factors, ae->variants, ae->nf, ae->use_dif_forward, reg);
    if (!pT || !pP)
    {
        if (pT)
            vfft_proto_plan_destroy(pT);
        if (pP)
            vfft_proto_plan_destroy(pP);
        return 0;
    }
    /* Each arm at its best: give BOTH a baked/JIT executor where one exists
     * (giving it to only one arm silently favours that arm). */
    vfft_proto_exec_fn jfT = NULL, jfP = NULL;
#ifdef VFFT_USE_JIT
    if (pT->num_stages > 0)
        jfT = vfft_proto_plan_jit_fwd(pT);
    if (pP->num_stages > 0)
        jfP = vfft_proto_plan_jit_fwd(pP);
#endif
    /* ONE arena, each arm's region at its TRUE size (so the tight arm's smaller
     * cache/TLB footprint is authentic), 64B skew between regions so no two
     * regions are mutually page-aligned (4KB aliasing -> bimodal timings). */
    const size_t SKEW = 8;                                      /* doubles == 64 B */
    const size_t szT = (size_t)N * K;                           /* tight plane  */
    const size_t szP = (size_t)N * Kp;                          /* padded plane */
    const size_t need = 4 * SKEW + 2 * szT + 2 * szP + 2 * szT; /* + reference planes */
    double *arena = NULL;
    if (vfft_proto_posix_memalign((void **)&arena, 64, need * sizeof(double)))
    {
        vfft_proto_plan_destroy(pT);
        vfft_proto_plan_destroy(pP);
        return 0;
    }
    double *rT = arena;             /* tight  re, N*K  */
    double *iT = rT + szT + SKEW;   /* tight  im, N*K  */
    double *rP = iT + szT + SKEW;   /* padded re, N*Kp */
    double *iP = rP + szP + SKEW;   /* padded im, N*Kp */
    double *refR = iP + szP + SKEW; /* roundtrip reference, N*K */
    double *refI = refR + szT;

    /* Identical data in the K live lanes (same seed); the padded arm zero-fills
     * its waste lanes, the tight arm has none. */
    _pad_fill(rT, iT, N, K, K);
    _pad_fill(rP, iP, N, K, Kp);
    int reps = (int)(8000000ull / szP);
    if (reps < 40)
        reps = 40;
    for (int w = 0; w < 5; w++)
    {
        _pad_burst(pT, jfT, rT, iT, K, reps);
        _pad_burst(pP, jfP, rP, iP, Kp, reps);
    }
    int RR = (rigor == VFFT_MEASURE) ? 31 : 81;
    double rt[128], rp[128];
    if (RR > 128)
        RR = 128;
    for (int r = 0; r < RR; r++)
    {
        double t, p;
        if (r & 1)
        {
            t = _pad_burst(pT, jfT, rT, iT, K, reps);
            p = _pad_burst(pP, jfP, rP, iP, Kp, reps);
        }
        else
        {
            p = _pad_burst(pP, jfP, rP, iP, Kp, reps);
            t = _pad_burst(pT, jfT, rT, iT, K, reps);
        }
        rt[r] = t / reps;
        rp[r] = p / reps;
    }
    double tight_ns = _pad_med(rt, RR), pad_ns = _pad_med(rp, RR);
    int pad_wins = (pad_ns < tight_ns * 0.97); /* 3% hysteresis toward TIGHT */
    int exec_me = pad_wins ? (int)Kp : (int)K;

    /* Roundtrip-gate the winner AT ITS OWN STRIDE (recover N*x on the K live
     * lanes). The winner owns its buffer, so refill it and keep an independent
     * reference of the K live lanes. */
    {
        stride_plan_t *wp = pad_wins ? pP : pT;
        double *wr = pad_wins ? rP : rT;
        double *wi = pad_wins ? iP : iT;
        const size_t st = pad_wins ? Kp : K; /* the winner's stride */
        _pad_fill(wr, wi, N, K, st);
        for (size_t e = 0; e < (size_t)N; e++)
            for (size_t l = 0; l < K; l++)
            {
                refR[e * K + l] = wr[e * st + l];
                refI[e * K + l] = wi[e * st + l];
            }
        vfft_proto_execute_fwd(wp, wr, wi, (size_t)exec_me);
        vfft_proto_execute_bwd(wp, wr, wi, (size_t)exec_me);
        double rtg = 0, inv = 1.0 / (double)N;
        for (size_t e = 0; e < (size_t)N; e++)
            for (size_t l = 0; l < K; l++)
            {
                double dr = fabs(wr[e * st + l] * inv - refR[e * K + l]);
                double di = fabs(wi[e * st + l] * inv - refI[e * K + l]);
                if (dr > rtg)
                    rtg = dr;
                if (di > rtg)
                    rtg = di;
            }
        if (rtg > 1e-7)
            exec_me = 0; /* winner failed the roundtrip -> caller falls back to tight */
    }

    vfft_proto_aligned_free(arena);
    vfft_proto_plan_destroy(pT);
    vfft_proto_plan_destroy(pP);
    return exec_me;
}

#endif /* VFFT_PLANNING_PAD_CALIBRATE_H */
