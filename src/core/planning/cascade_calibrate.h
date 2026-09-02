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
    if (zt->chain[zt->nf - 1] == 4)
    {
        zt->t2q = 0;
        return 0.0;
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

#endif /* VFFT_PLANNING_CASCADE_CALIBRATE_H */
