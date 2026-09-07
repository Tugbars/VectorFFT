/* cascade_calibrate.h - the cascade terminator (t2q) calibrators.
 *
 * Extracted from vfft.c as migration step 12; see
 * docs/design/refactor_migration_plan.md.
 *
 * WHY A BIT-IDENTICAL PAIR NEEDS A RACE AT ALL
 * -------------------------------------------
 * sterm vs sterm2, and stf vs stf2, are DIFFERENT CODE with IDENTICAL OUTPUT -
 * not renamed copies. The "2" forms are 2-quad unroll-and-jam: one loop
 * iteration processes two quads instead of one. That is a real difference in
 * instruction schedule, register pressure and code size (radix8 sterm is 211
 * lines, sterm2 is 571).
 *
 * What is identical is the RESULT. Unroll-and-jam interleaves two INDEPENDENT
 * iterations, so no floating-point operation is reordered within a lane and the
 * output matches bit for bit - which is memcmp-gated here before either arm is
 * timed. So the choice cannot be made on numerics: there is no more-accurate
 * arm, and no "better algorithm" to reason about.
 *
 * What is left is roughly 5%, and it is code-placement luck - which form wins
 * depends on how this binary happened to lay out, not on the construction. That
 * is precisely why it is measured on THIS binary at first create and banked,
 * rather than picked once by a human and frozen into a constant.
 *
 * PROTOCOL
 * --------
 * The _il_ab_race shape: alternating arm order per round, median of rounds, and
 * hysteresis toward the compiled default so a tie does not thrash the banked
 * verdict. Roughly a 10 ms budget. Returns the winner's median; on OOM or a
 * sanity failure it returns 0.0 and the plan's own t2q field still holds a
 * usable verdict.
 *
 * `aliased` SELECTS THE CALL FORM, AND THAT MATTERS
 * -------------------------------------------------
 * aliased=1 times the IN-PLACE call form (dst == src). An in-place caller
 * builds its own plans and its memory-access structure differs from the
 * out-of-place one, so a verdict measured through the wrong door is a verdict
 * for a different question. The bit-identity sanity check stays OOP-buffered
 * regardless, because it needs the input preserved to compare against.
 *
 * THE LEGACY ARM IS NOT DEAD CODE
 * -------------------------------
 * Since the 2026-07-27 ZTURN-only cutover the zsplit calibrator runs only under
 * the VFFT_NO_ZTURN kill switch, VFFT_FORCE_ZROUTE=legacy, or as the degrade
 * when the zturn create fails for a given N. Reachable-under-kill-switch legacy
 * paths stay: deleting one removes the fallback AND the control arm that makes
 * the zturn verdict falsifiable.
 *
 * FLOOR-LEGAL BY CONSTRUCTION
 * ---------------------------
 * Takes the cascade plans by pointer, never a vfft_plan_s, and touches no
 * wisdom - the caller banks. It does increment the shared create-race counter,
 * which is why that counter is a tentative definition with external linkage in
 * vfft.c rather than a static: a static in a header is one copy per includer,
 * and the accessor would then read a different object than the increment
 * writes. Declared extern below; defined once, in vfft.c.
 */
#ifndef VFFT_PLANNING_CASCADE_CALIBRATE_H
#define VFFT_PLANNING_CASCADE_CALIBRATE_H

#include <stdlib.h>
#include <string.h>

#include "zsplit.h"                 /* vfft_zsplit_plan_t + its execute */
#include "zturn.h"                  /* vfft_zturn2_plan_t + its execute */
#include "support/race.h"           /* the shared race body */

/* Defined in vfft.c (tentative definition, external linkage). See the note
 * above on why this is not a static. */
extern long _vfft_create_race_count;

/* ════════════════════════════════════════════════════════════════════════
 * ZSPLIT TERMINATOR PICK (K=1 SCRAMBLED cascade, z_cascade_plan §4.9993) —
 * sterm vs sterm2 are BIT-IDENTICAL schedules whose delta (±5%) is the same
 * order as code-placement luck, so the pick is measured on THIS binary at
 * first create and banked as a kind-4 oop_wisdom line. ~10 ms budget in the
 * _il_ab_race shape: alternating arm order per round, median-of-rounds, 3%
 * hysteresis toward the compiled default. Returns the winner's median ns
 * (0.0 on OOM/sanity failure; zs->t2q holds the verdict either way).
 * REACHABILITY since the 2026-07-27 ZTURN-only cutover: this legacy race is
 * NOT dead code — it runs only under the VFFT_NO_ZTURN kill switch /
 * VFFT_FORCE_ZROUTE=legacy, or as the degrade when the zturn create/race
 * fails for this N (fallback intact; hygiene rule: reachable-under-kill-
 * switch legacy paths stay). */
/* aliased=1 times the IN-PLACE call form (dst==src; alias-safety is the
 * P0a memcmp-proven contract, data saturating to inf is the house-accepted
 * in-place timing mode) — the in-place caller's own memory-access
 * structure, not the OOP one (owner, 2026-08-25: in-place creates its own
 * plans; verdicts can differ by placement). The bit-identity sanity check
 * stays OOP-buffered (it needs the preserved input). */
/* the arms of the t2q races: one plan, the terminator pick toggled */
typedef struct { vfft_zsplit_plan_t *p; double *zi, *zd; } _zs_t2q_arm_t;
static void _zs_t2q_arm0(void *v)
{
    _zs_t2q_arm_t *c = (_zs_t2q_arm_t *)v;
    c->p->t2q = 0;
    vfft_zsplit_execute_fwd(c->p, c->zi, c->zd);
}
static void _zs_t2q_arm1(void *v)
{
    _zs_t2q_arm_t *c = (_zs_t2q_arm_t *)v;
    c->p->t2q = 1;
    vfft_zsplit_execute_fwd(c->p, c->zi, c->zd);
}
typedef struct { vfft_zturn2_plan_t *p; double *zi, *zd; } _zt_t2q_arm_t;
static void _zt_t2q_arm0(void *v)
{
    _zt_t2q_arm_t *c = (_zt_t2q_arm_t *)v;
    c->p->t2q = 0;
    vfft_zturn2_execute_fwd(c->p, c->zi, c->zd);
}
static void _zt_t2q_arm1(void *v)
{
    _zt_t2q_arm_t *c = (_zt_t2q_arm_t *)v;
    c->p->t2q = 1;
    vfft_zturn2_execute_fwd(c->p, c->zi, c->zd);
}
static double _calibrate_zsplit_t2q(vfft_zsplit_plan_t *zs,
                                    vfft_rigor_t rigor, int aliased)
{
    _vfft_create_race_count++;   /* HARNESS: this racer is about to time */
    const int N = zs->N;
    const size_t sz = (size_t)2 * (size_t)N * sizeof(double);
    const int inc = zs->t2q; /* compiled default = incumbent */
    double *zi = NULL, *zo = NULL, *zo2 = NULL;
    if (vfft_proto_posix_memalign((void **)&zi, 64, sz) ||
        vfft_proto_posix_memalign((void **)&zo, 64, sz) ||
        vfft_proto_posix_memalign((void **)&zo2, 64, sz))
    {
        vfft_proto_aligned_free(zi);
        vfft_proto_aligned_free(zo);
        vfft_proto_aligned_free(zo2);
        return 0.0;
    }
    srand(11 + N);
    for (int i = 0; i < 2 * N; i++)
        zi[i] = (double)rand() / RAND_MAX - 0.5;

    /* sanity: the pair is bit-identical by contract; if a build ever breaks
     * that, keep the incumbent and don't bank. */
    zs->t2q = 0;
    vfft_zsplit_execute_fwd(zs, zi, zo);
    zs->t2q = 1;
    vfft_zsplit_execute_fwd(zs, zi, zo2);
    if (memcmp(zo, zo2, sz) != 0)
    {
        zs->t2q = inc;
        vfft_proto_aligned_free(zi);
        vfft_proto_aligned_free(zo);
        vfft_proto_aligned_free(zo2);
        return 0.0;
    }

    /* size bursts to ~0.3 ms from one estimated exec */
    double *zd = aliased ? zi : zo; /* the timed call form's destination */
    double t0 = vfft_proto_now_ns();
    vfft_zsplit_execute_fwd(zs, zi, zd);
    double est = vfft_proto_now_ns() - t0;
    if (est < 1.0)
        est = 1.0;
    int reps = (int)(300000.0 / est);
    if (reps < 2)
        reps = 2;
    if (reps > 64)
        reps = 64;

    int RR = (rigor == VFFT_MEASURE) ? 9 : 21;
    double n0, n1;
    {
        _zs_t2q_arm_t c = { zs, zi, zd };
        const vfft_race_arm_t arms[2] = { { "t2q0", _zs_t2q_arm0, &c },
                                          { "t2q1", _zs_t2q_arm1, &c } };
        /* RR rounds alternated, median; the incumbent takes 3% hysteresis below */
        const vfft_race_proto_t proto = { RR, reps, VFFT_RACE_MEDIAN, 1, 0, NULL, NULL };
        double ns[2];
        vfft_race_run(&proto, arms, 2, ns);
        n0 = ns[0];
        n1 = ns[1];
    }
    int win;
    if (inc == 0)
        win = (n1 < n0 * 0.97) ? 1 : 0; /* 3% hysteresis toward the default */
    else
        win = (n0 < n1 * 0.97) ? 0 : 1;
    zs->t2q = win;
    if (getenv("VFFT_ZRACE_VERBOSE"))
        fprintf(stderr, "[zroute] N=%d legacy-t2q race: reps=%d RR=%d "
                        "burst~300us hyst=3%% alt-order median | sterm=%.0f "
                        "sterm2=%.0f -> t2q=%d\n",
                N, reps, RR, n0, n1, win);
    vfft_proto_aligned_free(zi);
    vfft_proto_aligned_free(zo);
    vfft_proto_aligned_free(zo2);
    return win ? n1 : n0;
}

/* stf/stf2 twin of _calibrate_zsplit_t2q — same mechanics, fwd-only. This is the cascade's
 * create-time miss race; engine (zsplit vs zturn) and chain are searched offline, not here.
 * aliased: same contract as the zsplit twin (in-place call-form timing).
 * See docs/design/vfft_front_door.md. */
static double _calibrate_zturn_t2q(vfft_zturn2_plan_t *zt, vfft_rigor_t rigor,
                                   int aliased)
{
    _vfft_create_race_count++;   /* HARNESS: this racer is about to time */
    /* last==4 chains (radix-4 terminator) have NO stf2 twin — zturn.h's
     * create forces t2q=0 and the execute dispatch is structural about it —
     * so a "race" here would time one kernel against itself. Pin the only
     * legal pick and refuse loudly (0.0 = no verdict; the caller degrades
     * to the legacy race, exactly the create/sanity-failure path). Only
     * reachable if the default chain ever ends in 4 — today the defaults
     * (vfft_zsplit_default_chain) all end in 8; last==4 winners come from
     * the offline planner (dp_planner_il.h), which banks t2q=0. */
    if (zt->chain[zt->nf - 1] == 4 || zt->r0 == 8)
    {
        /* (r0 = 8 chains have no stf2 twin either: create pins t2q = 0)
         * one form, so no race — but the candidate must still carry a
         * time: 0.0 read as "no verdict" destroyed every last==4 seed
         * before it could race the cell (the sub-2048 natural seeds all
         * end in 4). One batch >= 1 ms, its median-of-one. */
        const int N4 = zt->N;
        const size_t sz4 = (size_t)2 * (size_t)N4 * sizeof(double);
        double *a = NULL, *b = NULL, est, t0;
        int reps, i;
        zt->t2q = 0;
        if (vfft_proto_posix_memalign((void **)&a, 64, sz4) ||
            vfft_proto_posix_memalign((void **)&b, 64, sz4))
        { vfft_proto_aligned_free(a); vfft_proto_aligned_free(b); return 1.0; }
        srand(11 + N4);
        for (i = 0; i < 2 * N4; i++) a[i] = (double)rand() / RAND_MAX - 0.5;
        vfft_zturn2_execute_fwd(zt, a, aliased ? a : b);
        t0 = vfft_proto_now_ns(); vfft_zturn2_execute_fwd(zt, a, aliased ? a : b); est = vfft_proto_now_ns() - t0;
        if (est < 1.0) est = 1.0;
        reps = (int)(1.0e6 / est); if (reps < 2) reps = 2; if (reps > (1 << 16)) reps = 1 << 16;
        t0 = vfft_proto_now_ns();
        for (i = 0; i < reps; i++) vfft_zturn2_execute_fwd(zt, a, aliased ? a : b);
        est = (vfft_proto_now_ns() - t0) / reps;
        vfft_proto_aligned_free(a); vfft_proto_aligned_free(b);
        return est > 0.0 ? est : 1.0;
    }
    const int N = zt->N;
    const size_t sz = (size_t)2 * (size_t)N * sizeof(double);
    const int inc = zt->t2q; /* compiled default (0 = stf) = incumbent */
    double *zi = NULL, *zo = NULL, *zo2 = NULL;
    if (vfft_proto_posix_memalign((void **)&zi, 64, sz) ||
        vfft_proto_posix_memalign((void **)&zo, 64, sz) ||
        vfft_proto_posix_memalign((void **)&zo2, 64, sz))
    {
        vfft_proto_aligned_free(zi);
        vfft_proto_aligned_free(zo);
        vfft_proto_aligned_free(zo2);
        return 0.0;
    }
    srand(11 + N);
    for (int i = 0; i < 2 * N; i++)
        zi[i] = (double)rand() / RAND_MAX - 0.5;

    /* sanity: stf/stf2 are bit-identical by contract (Phase-3 GATE0); if a
     * build ever breaks that, keep the incumbent and don't bank. */
    zt->t2q = 0;
    vfft_zturn2_execute_fwd(zt, zi, zo);
    zt->t2q = 1;
    vfft_zturn2_execute_fwd(zt, zi, zo2);
    if (memcmp(zo, zo2, sz) != 0)
    {
        zt->t2q = inc;
        vfft_proto_aligned_free(zi);
        vfft_proto_aligned_free(zo);
        vfft_proto_aligned_free(zo2);
        return 0.0;
    }

    double *zd = aliased ? zi : zo; /* the timed call form's destination */
    double t0 = vfft_proto_now_ns();
    vfft_zturn2_execute_fwd(zt, zi, zd);
    double est = vfft_proto_now_ns() - t0;
    if (est < 1.0)
        est = 1.0;
    int reps = (int)(300000.0 / est);
    if (reps < 2)
        reps = 2;
    if (reps > 64)
        reps = 64;

    int RR = (rigor == VFFT_MEASURE) ? 9 : 21;
    double n0, n1;
    {
        _zt_t2q_arm_t c = { zt, zi, zd };
        const vfft_race_arm_t arms[2] = { { "t2q0", _zt_t2q_arm0, &c },
                                          { "t2q1", _zt_t2q_arm1, &c } };
        /* RR rounds alternated, median; the incumbent takes 3% hysteresis below */
        const vfft_race_proto_t proto = { RR, reps, VFFT_RACE_MEDIAN, 1, 0, NULL, NULL };
        double ns[2];
        vfft_race_run(&proto, arms, 2, ns);
        n0 = ns[0];
        n1 = ns[1];
    }
    int win;
    if (inc == 0)
        win = (n1 < n0 * 0.97) ? 1 : 0;
    else
        win = (n0 < n1 * 0.97) ? 0 : 1;
    zt->t2q = win;
    if (getenv("VFFT_ZRACE_VERBOSE"))
        fprintf(stderr, "[zroute] N=%d zturn-t2q race: reps=%d RR=%d "
                        "burst~300us hyst=3%% alt-order median | stf=%.0f "
                        "stf2=%.0f -> t2q=%d\n",
                N, reps, RR, n0, n1, win);
    vfft_proto_aligned_free(zi);
    vfft_proto_aligned_free(zo);
    vfft_proto_aligned_free(zo2);
    return win ? n1 : n0;
}

/* ── the terminator FORM race (2026-09-07, the sub-2048 campaign) ───────────
 * Both order classes of ONE cascade recipe: the SCRAMBLED terminator
 * (tform: stf/stf2 by the t2q just picked, vs stfl) and the NATURAL one
 * (ntform: stfn vs stfnl), each a two-arm whole-forward race, median of RR
 * alternated rounds, batches >= 1 ms, 3% hysteresis toward form 0. The
 * loaded twin differs from the squaring tree at ROUNDING level (exact
 * cos/sin per power vs products), so the sanity check is a relative-error
 * bound, never memcmp; a form that fails it is pinned to 0 and not raced.
 * Runs after _calibrate_zturn_t2q on the same plan; leaves the plan at the
 * winning forms with the stream built (freed when both are 0). Never reads
 * the env: the pin (VFFT_ZT_TFORM) is applied by the commit, after banking. */
typedef struct { vfft_zturn2_plan_t *p; double *zi, *zd; int nat; int form; } _zt_tf_arm_t;
static void _zt_tf_arm(void *v)
{
    _zt_tf_arm_t *c = (_zt_tf_arm_t *)v;
    if (c->nat) c->p->ntform = c->form; else c->p->tform = c->form;
    vfft_zturn2_execute_fwd(c->p, c->zi, c->zd);
}
static double _zt_tf_relerr(const double *a, const double *b, long n2)
{
    double m = 0, e = 0;
    for (long i = 0; i < n2; i++) {
        const double d = a[i] - b[i] < 0 ? b[i] - a[i] : a[i] - b[i];
        const double v = b[i] < 0 ? -b[i] : b[i];
        if (v > m) m = v;
        if (d > e) e = d;
    }
    return m > 0 ? e / m : e;
}
/* one class: form 0 vs 1 on the plan as configured (natord set by the caller) */
static int _zt_tf_race_class(vfft_zturn2_plan_t *zt, int nat, double *zi, double *zd,
                             double *zo2, size_t sz, int RR, int *ns_out)
{
    int *slot = nat ? &zt->ntform : &zt->tform;
    double n0, n1, err;
    _zt_tf_arm_t c0 = { zt, zi, zd, nat, 0 }, c1 = { zt, zi, zd, nat, 1 };
    /* sanity at rounding level (the forms are not bitwise twins) */
    *slot = 0; vfft_zturn2_execute_fwd(zt, zi, zd);
    *slot = 1; vfft_zturn2_execute_fwd(zt, zi, zo2);
    err = _zt_tf_relerr(zo2, zd, (long)(sz / sizeof(double)));
    if (!(err < 1e-12)) { *slot = 0; (void)ns_out; return -1; }
    {
        double est, ns[2];
        int reps;
        *slot = 0;
        vfft_zturn2_execute_fwd(zt, zi, zd);
        est = vfft_proto_now_ns();
        vfft_zturn2_execute_fwd(zt, zi, zd);
        est = vfft_proto_now_ns() - est;
        if (est < 1.0) est = 1.0;
        reps = (int)(1.0e6 / est);           /* >= 1 ms per batch */
        if (reps < 2) reps = 2;
        if (reps > (1 << 16)) reps = 1 << 16;
        {
            const vfft_race_arm_t arms[2] = { { nat ? "stfn" : "stf", _zt_tf_arm, &c0 },
                                              { nat ? "stfnl" : "stfl", _zt_tf_arm, &c1 } };
            const vfft_race_proto_t proto = { RR, reps, VFFT_RACE_MEDIAN, 1, 1, NULL, NULL };
            vfft_race_run(&proto, arms, 2, ns);
            n0 = ns[0]; n1 = ns[1];
        }
        *slot = (n1 < n0 * 0.97) ? 1 : 0;    /* 3% hysteresis toward the squaring tree */
        if (getenv("VFFT_ZRACE_VERBOSE"))
            fprintf(stderr, "[zroute] N=%d zturn-tform race (%s): reps=%d RR=%d | form0=%.0f "
                            "form1=%.0f -> %s=%d\n", zt->N, nat ? "natural" : "scrambled",
                    reps, RR, n0, n1, nat ? "ntform" : "tform", *slot);
    }
    return *slot;
}
static void _calibrate_zturn_tform(vfft_zturn2_plan_t *zt, vfft_rigor_t rigor, int aliased)
{
    const int N = zt->N;
    const size_t sz = (size_t)2 * (size_t)N * sizeof(double);
    const int RR = (rigor == VFFT_MEASURE) ? 9 : 21;
    const int tfuse0 = zt->tfuse;
    double *zi = NULL, *zo = NULL, *zo2 = NULL, *zd;
    if (zt->lanes_u) return;                          /* no loaded twin of stfu */
    if (!vfft_zturn2_set_tforms(zt, 1, 1)) return;    /* the stream, both forms available */
    if (vfft_proto_posix_memalign((void **)&zi, 64, sz) ||
        vfft_proto_posix_memalign((void **)&zo, 64, sz) ||
        vfft_proto_posix_memalign((void **)&zo2, 64, sz))
    {
        vfft_proto_aligned_free(zi); vfft_proto_aligned_free(zo); vfft_proto_aligned_free(zo2);
        (void)vfft_zturn2_set_tforms(zt, 0, 0);
        return;
    }
    _vfft_create_race_count++;   /* HARNESS: this racer is about to time */
    srand(13 + N);
    for (int i = 0; i < 2 * N; i++) zi[i] = (double)rand() / RAND_MAX - 0.5;
    zd = aliased ? zi : zo;
    /* SCRAMBLED class (the plan as built: natord off) */
    if (aliased) memcpy(zo2, zi, sz);   /* zd == zi: reseed for the second arm's sanity run */
    (void)_zt_tf_race_class(zt, 0, zi, zd, zo2, sz, RR, NULL);
    /* NATURAL class: the natord twin of the same recipe, then back */
    if (aliased) { srand(13 + N); for (int i = 0; i < 2 * N; i++) zi[i] = (double)rand() / RAND_MAX - 0.5; }
    if (vfft_zturn2_set_natord(zt, 1))
    {
        (void)_zt_tf_race_class(zt, 1, zi, zd, zo2, sz, RR, NULL);
        (void)vfft_zturn2_set_natord(zt, 0);
    }
    else
        zt->ntform = 0;
    zt->tfuse = tfuse0;                     /* set_natord(1) clears it; restore the scrambled plan's */
    if (!zt->tform && !zt->ntform) (void)vfft_zturn2_set_tforms(zt, 0, 0);
    vfft_proto_aligned_free(zi); vfft_proto_aligned_free(zo); vfft_proto_aligned_free(zo2);
}

/* ═══════════ SUB-2048 CHAIN RACE (the natural tier, 2026-09-07) ═══════════
 * Below 2048 the cascade serves NATURAL cells only and the dp planner does
 * not enumerate there, so the CHAIN is raced at the create: every ordered
 * {4,8} chain with product N (nf 3..VFFT_ZSPLIT_MAX_NF), BOTH ingest radices
 * (r0 = 4: the 4-section geometry; r0 = 8: the two-quartet geometry, one
 * pass fewer), each the zturn create's to admit. Per chain the natural
 * twin's terminator form is raced (stfn vs stfnl, _zt_tf_race_class), then
 * every chain's whole natural forward is one arm of ONE race — same-run
 * arms, alternated rounds, batches >= 1 ms, median — and the fastest chain
 * survives with its forms calibrated (the scrambled class too, so the
 * banked zt_tf / zt_ntf pair is complete). Returns the plan with natord
 * OFF (the caller's natural site sets it), *ns_out = its natural forward
 * time; NULL when no chain is admitted (the caller's "no zturn arm").
 * The arm count is the chain count (<= 12 at 1024); the cost is one
 * create-time race per cell, banked as the comp row's cc_chain. */
typedef struct { vfft_zturn2_plan_t *p; const double *zi; double *zd; } _zt_chain_arm_t;
static void _zt_chain_arm(void *v)
{
    const _zt_chain_arm_t *c = (const _zt_chain_arm_t *)v;
    vfft_zturn2_execute_fwd(c->p, c->zi, c->zd);
}
/* in place: re-seed the aliased buffer before every timed sample (repeated
 * in-place forwards walk into inf; the race protocol's reset hook) */
typedef struct { double *zi; const double *seed; size_t sz; } _zt_chain_reset_t;
static void _zt_chain_reset(void *v)
{
    const _zt_chain_reset_t *r = (const _zt_chain_reset_t *)v;
    memcpy(r->zi, r->seed, r->sz);
}
#define _ZT_CHAIN_RACE_MAX 32
static vfft_zturn2_plan_t *_calibrate_zturn_chain_sub2048(int N, vfft_rigor_t rigor,
                                                          int aliased, double *ns_out)
{
    vfft_zturn2_plan_t *plans[_ZT_CHAIN_RACE_MAX];
    vfft_race_arm_t arms[_ZT_CHAIN_RACE_MAX];
    _zt_chain_arm_t ctx[_ZT_CHAIN_RACE_MAX];
    char names[_ZT_CHAIN_RACE_MAX][24];
    double ns[_ZT_CHAIN_RACE_MAX];
    int nplan = 0, dropped = 0, best = -1;
    const size_t sz = (size_t)2 * (size_t)N * sizeof(double);
    const int RR = (rigor == VFFT_MEASURE) ? 9 : 21;
    double *zi = NULL, *zo = NULL, *zo2 = NULL, *seed = NULL, *zd;
    if (ns_out) *ns_out = 0.0;
    if (vfft_proto_posix_memalign((void **)&zi, 64, sz) ||
        vfft_proto_posix_memalign((void **)&zo, 64, sz) ||
        vfft_proto_posix_memalign((void **)&zo2, 64, sz) ||
        vfft_proto_posix_memalign((void **)&seed, 64, sz))
    {
        vfft_proto_aligned_free(zi); vfft_proto_aligned_free(zo);
        vfft_proto_aligned_free(zo2); vfft_proto_aligned_free(seed);
        return NULL;
    }
    srand(29 + N);
    for (int i = 0; i < 2 * N; i++) zi[i] = (double)rand() / RAND_MAX - 0.5;
    memcpy(seed, zi, sz);
    zd = aliased ? zi : zo;
    /* enumerate: ordered {4,8} chains with product N — the same walk as the
     * dp planner's scrambled cell; legality is the create's (the law) */
    for (int nf = 3; nf <= VFFT_ZSPLIT_MAX_NF; nf++)
    {
        const long combos = 1L << nf;
        for (long mask = 0; mask < combos; mask++)
        {
            int chain[VFFT_ZSPLIT_MAX_NF];
            long prod = 1;
            for (int i = 0; i < nf; i++)
            {
                chain[i] = ((mask >> i) & 1) ? 8 : 4;
                prod *= chain[i];
            }
            if (prod != (long)N) continue;
            vfft_zturn2_plan_t *p = vfft_zturn2_create_chain(N, chain, nf);
            if (!p) continue;
            if (nplan >= _ZT_CHAIN_RACE_MAX) { dropped++; vfft_zturn2_destroy(p); continue; }
            /* the natural twin, its terminator form raced */
            if (!vfft_zturn2_set_natord(p, 1) || !vfft_zturn2_set_tforms(p, 1, 1))
            {
                vfft_zturn2_destroy(p);
                continue;
            }
            if (aliased) memcpy(zo2, zi, sz);
            (void)_zt_tf_race_class(p, 1, zi, zd, zo2, sz, RR, NULL);
            if (aliased) memcpy(zi, seed, sz);
            {
                int off = 0;
                for (int s = 0; s < nf && off < 20; s++)
                    off += snprintf(names[nplan] + off, sizeof names[nplan] - (size_t)off,
                                    "%s%d", s ? "." : "", chain[s]);
            }
            plans[nplan] = p;
            ctx[nplan].p = p; ctx[nplan].zi = zi; ctx[nplan].zd = zd;
            arms[nplan].name = names[nplan];
            arms[nplan].run = _zt_chain_arm;
            arms[nplan].ctx = &ctx[nplan];
            nplan++;
        }
    }
    if (dropped)
        fprintf(stderr, "[zroute] N=%d: %d cascade chains did not fit the race array "
                        "(_ZT_CHAIN_RACE_MAX=%d) — raise it\n", N, dropped, _ZT_CHAIN_RACE_MAX);
    if (nplan)
    {
        double est;
        int reps;
        _vfft_create_race_count++;   /* HARNESS: this racer is about to time */
        vfft_zturn2_execute_fwd(plans[0], zi, zd);
        est = vfft_proto_now_ns();
        vfft_zturn2_execute_fwd(plans[0], zi, zd);
        est = vfft_proto_now_ns() - est;
        if (est < 1.0) est = 1.0;
        reps = (int)(1.0e6 / est);           /* >= 1 ms per batch */
        if (reps < 2) reps = 2;
        if (reps > (1 << 16)) reps = 1 << 16;
        {
            _zt_chain_reset_t rs = { zi, seed, sz };
            const vfft_race_proto_t proto = { RR, reps, VFFT_RACE_MEDIAN, 1, 1,
                                              aliased ? _zt_chain_reset : NULL,
                                              aliased ? (void *)&rs : NULL };
            vfft_race_run(&proto, arms, nplan, ns);
        }
        for (int i = 0; i < nplan; i++)
            if (best < 0 || ns[i] < ns[best]) best = i;
        if (getenv("VFFT_ZRACE_VERBOSE"))
        {
            fprintf(stderr, "[zroute] N=%d sub-2048 chain race (natural, %s): reps=%d RR=%d |",
                    N, aliased ? "in place" : "oop", reps, RR);
            for (int i = 0; i < nplan; i++)
                fprintf(stderr, " %s%s=%.0f(ntf%d)", i == best ? "*" : "", names[i], ns[i],
                        plans[i]->ntform);
            fprintf(stderr, "\n");
        }
    }
    for (int i = 0; i < nplan; i++)
        if (i != best) vfft_zturn2_destroy(plans[i]);
    vfft_proto_aligned_free(zi); vfft_proto_aligned_free(zo);
    vfft_proto_aligned_free(zo2); vfft_proto_aligned_free(seed);
    if (best < 0) return NULL;
    {
        /* the winner: natord off for the caller; its scrambled form raced
         * too so the banked (zt_tf, zt_ntf) pair is complete — the natural
         * class is re-raced inside, on ONE plan (create-time only) */
        vfft_zturn2_plan_t *w = plans[best];
        (void)vfft_zturn2_set_natord(w, 0);
        _calibrate_zturn_tform(w, rigor, aliased);
        if (ns_out) *ns_out = ns[best];
        return w;
    }
}

#endif /* VFFT_PLANNING_CASCADE_CALIBRATE_H */
