/* il_flatdit_race.h — the flat DIT's create-time races on the shared race
 * body (2026-09-07). Two axes the planner banks on the kind-3 row:
 *
 *   FORMS  il_forms= : per stage s >= 1, the tail letters t | n | o and the
 *                      msz letter m, decided in pipeline order (stage s is
 *                      raced with every earlier stage already at its verdict)
 *   TILE   il_tw=    : the walk width, 0 = untiled, raced after the forms
 *
 * Both arms are the WHOLE FORWARD with one stage's form (or the width)
 * flipped, so the table footprint, the walk and the last stage's scatter
 * are inside every sample. The sample is a batch of reps executions sized
 * to VFFT_ILFD_RACE_SAMPLE_NS: the single-call race this replaces timed ONE
 * stage per sample against the 100 ns QPC tick, so at N <= 512 every
 * sample read 0 / 100 / 200 ns, arms tied, and the strict "<" kept the
 * default — it banked a 1.50x slower tail form at 512 64.8 (t = 701 ns over
 * o = 467) and a 1.35x slower one at 2048 64.8.4. The property this header
 * guarantees is the counter: vfft_ilfd_race_short_samples() reads 0 when
 * no arm's batch was under VFFT_ILFD_RACE_SHORT_NS (2000 ticks).
 *
 * Verdict rule: the bare "<" with the FIRST arm keeping ties — arm 0 is the
 * default form (t, non-msz, untiled), so a tie never promotes a
 * challenger. Constants: VFFT_ILFD_RACE_ROUNDS samples per arm, MIN,
 * alternating order, one untimed warm pass. il_flatdit.h stays engine-pure:
 * it owns the plan, the bind and the executor; this header owns the clock.
 * Positioned after support/race.h and before planning/dp_planner_il.h. */
#ifndef VFFT_IL_FLATDIT_RACE_H
#define VFFT_IL_FLATDIT_RACE_H

#include "il_flatdit.h"
#include "support/race.h"

#define VFFT_ILFD_RACE_SAMPLE_NS 1.0e6   /* one timed batch >= 1 ms */
#define VFFT_ILFD_RACE_ROUNDS    5
#define VFFT_ILFD_RACE_MAX_REPS  (1 << 20)

extern long _vfft_ilfd_short_count;      /* vfft.c; vfft_ilfd_race_short_samples() */

/* one arm: the plan at ONE stage's form (or one width), the whole forward.
 * The plan is shared by every arm, so the run switches the plan to its own
 * form on demand and rebinds only when the fields differ — the race body
 * runs an arm reps times back to back, so a switch happens once per round
 * per arm, never per execution. */
typedef struct {
    vfft_ilfd_plan_t *p;
    const double *zin;
    double *zout;
    int s;            /* the stage this arm sets; -1 = the tile arm */
    int gl, gord, msz;
    int tw;
    char name[24];
} _ilfd_race_ctx_t;

static void _ilfd_race_arm_run(void *ctx)
{
    _ilfd_race_ctx_t *c = (_ilfd_race_ctx_t *)ctx;
    vfft_ilfd_plan_t *p = c->p;
    if (c->s < 0) {
        if (p->tw != c->tw) vfft_ilfd_apply_tw(p, c->tw);   /* rebinds */
    } else if (p->gl[c->s] != c->gl || p->msz[c->s] != c->msz ||
               (c->s == p->K - 1 && p->gord != c->gord)) {
        p->gl[c->s] = c->gl;
        p->msz[c->s] = c->msz;
        if (c->s == p->K - 1) p->gord = c->gord;
        vfft_ilfd_bind(p);
    }
    vfft_ilfd_execute_fwd(p, c->zin, c->zout);
}

/* reps so one batch is >= VFFT_ILFD_RACE_SAMPLE_NS, piloted on the plan as
 * bound AT STEADY STATE: the planner paces with a 200 ms sleep, and a pilot
 * taken right after it sees a cold core (1.5-5x slow for the first
 * milliseconds), sizes reps short, and every warm arm then lands under the
 * target — the property counter caught exactly that. So: an untimed warm
 * of >= 2 ms first, then the per-execution cost as the MIN over three
 * batches of >= 200 us each, so the pilot never reads the tick either. */
static inline int _ilfd_race_reps(vfft_ilfd_plan_t *p, const double *zin, double *zout)
{
    int n = 1, k;
    double per = 1e300;
    {   /* warm: >= 2 ms of executions, untimed */
        const double t0 = _il_ab_now();
        do { vfft_ilfd_execute_fwd(p, zin, zout); } while (_il_ab_now() - t0 < 2.0e6);
    }
    for (;;) {   /* the batch size that spans >= 200 us */
        const double t0 = _il_ab_now();
        int i;
        for (i = 0; i < n; i++) vfft_ilfd_execute_fwd(p, zin, zout);
        if (_il_ab_now() - t0 >= 2.0e5 || n >= VFFT_ILFD_RACE_MAX_REPS) break;
        n *= 4;
    }
    for (k = 0; k < 3; k++) {
        const double t0 = _il_ab_now();
        double dt;
        int i;
        for (i = 0; i < n; i++) vfft_ilfd_execute_fwd(p, zin, zout);
        dt = (_il_ab_now() - t0) / (double)n;
        if (dt < per) per = dt;
    }
    if (per < 1.0) per = 1.0;
    n = (int)(VFFT_ILFD_RACE_SAMPLE_NS / per) + 1;
    if (n < 4) n = 4;
    if (n > VFFT_ILFD_RACE_MAX_REPS) n = VFFT_ILFD_RACE_MAX_REPS;
    return n;
}

static inline int _ilfd_race_log(void)
{
    return getenv("VFFT_NAT_LOG") != NULL || getenv("VFFT_IL2D_LOG") != NULL;
}

/* the property counter: an arm whose batch was under VFFT_ILFD_RACE_SHORT_NS
 * (200 us = 2000 ticks of the 100 ns clock: quantization 0.05%). Reps are
 * sized on the default forms at steady state, so an arm lands here only by
 * running 5x faster than that pilot — which is not a form, it is the pilot
 * having measured a disturbed core. Named in the log. */
#define VFFT_ILFD_RACE_SHORT_NS 2.0e5
static inline void _ilfd_race_audit(const vfft_race_arm_t *arms, const double *ns, int na,
                                    int reps, int N)
{
    int a;
    for (a = 0; a < na; a++)
        if (ns[a] * (double)reps < VFFT_ILFD_RACE_SHORT_NS) {
            _vfft_ilfd_short_count++;
            if (_ilfd_race_log())
                fprintf(stderr, "[k1fd-race] SHORT N=%d %s: %.0f ns x %d reps = %.0f us\n",
                        N, arms[a].name, ns[a], reps, ns[a] * reps / 1e3);
        }
}

/* Per-stage FORM race: on each tail-capable stage the letters t / n / o
 * (o = natural-base order, last stage of the natural class only), then on
 * each msz-eligible stage m against the stage's best non-msz form; pipeline
 * order. Leaves the plan bound at the verdict and zout transformed. */
static inline void vfft_ilfd_race_forms(vfft_ilfd_plan_t *p, const double *zin, double *zout)
{
    _ilfd_race_ctx_t cx[3];
    vfft_race_arm_t arms[3];
    double ns[3];
    const int log = _ilfd_race_log();
    int s, reps;
    vfft_ilfd_bind(p);
    reps = _ilfd_race_reps(p, zin, zout);
    if (log) fprintf(stderr, "[k1fd-race] N=%d K=%d forms: reps=%d (sample >= %.0f us)\n",
                     p->N, p->K, reps, VFFT_ILFD_RACE_SAMPLE_NS / 1e3);
    for (s = 1; s < p->K; s++) {
        const vfft_race_proto_t proto = { VFFT_ILFD_RACE_ROUNDS, reps, VFFT_RACE_MIN, 1, 1, NULL, NULL };
        int na, a, best;
#define ILFD_FARM(GL, GORD, MSZ, NAME) do { \
            cx[na].p = p; cx[na].zin = zin; cx[na].zout = zout; cx[na].s = s; \
            cx[na].gl = (GL); cx[na].gord = (GORD); cx[na].msz = (MSZ); cx[na].tw = 0; \
            snprintf(cx[na].name, sizeof cx[na].name, "s%d:%s", s, NAME); \
            arms[na].name = cx[na].name; arms[na].run = _ilfd_race_arm_run; arms[na].ctx = &cx[na]; na++; \
        } while (0)
        if (p->fgl[s]) {
            /* the tail forms, msz off meanwhile: t (per group), n (t2csgn in
             * block order), o (t2csgn in natural-base order, last stage only) */
            const int last = (s == p->K - 1);
            na = 0;
            ILFD_FARM(0, 0, 0, "t");
            ILFD_FARM(1, 0, 0, "n");
            if (last && !p->scr) ILFD_FARM(1, 1, 0, "o");
            best = vfft_race_run(&proto, arms, na, ns);
            if (best < 0) best = 0;
            _ilfd_race_audit(arms, ns, na, reps, p->N);
            if (log) {
                fprintf(stderr, "[k1fd-race]  N=%d", p->N);
                for (a = 0; a < na; a++) fprintf(stderr, " %s=%.1f", cx[a].name, ns[a]);
                fprintf(stderr, " -> %s\n", cx[best].name);
            }
            p->gl[s] = cx[best].gl;
            if (last) p->gord = cx[best].gord;
            p->msz[s] = 0;
            vfft_ilfd_bind(p);
        }
        if (p->tz[s]) {
            /* m against the stage's best non-msz form */
            const int gl = p->gl[s], gord = p->gord;
            na = 0;
            ILFD_FARM(gl, gord, 0, p->fgl[s] ? (gl ? (gord && s == p->K - 1 ? "o" : "n") : "t") : "t");
            ILFD_FARM(gl, gord, 1, "m");
            best = vfft_race_run(&proto, arms, na, ns);
            if (best < 0) best = 0;
            _ilfd_race_audit(arms, ns, na, reps, p->N);
            if (log) fprintf(stderr, "[k1fd-race]  N=%d %s=%.1f %s=%.1f -> %s\n", p->N,
                             cx[0].name, ns[0], cx[1].name, ns[1], cx[best].name);
            p->msz[s] = cx[best].msz;
            vfft_ilfd_bind(p);
        }
#undef ILFD_FARM
    }
    vfft_ilfd_bind(p);
}

/* the TILE race: every legal width whose tile fits cache_bytes (<= 0 = no
 * gate), on the whole forward, after the forms. Applies and returns the
 * winner (0 = untiled). Leaves zout transformed. */
static inline int vfft_ilfd_race_tw(vfft_ilfd_plan_t *p, const double *zin, double *zout,
                                    long cache_bytes)
{
    _ilfd_race_ctx_t cx[VFFT_ILFD_MAX_K + 1];
    vfft_race_arm_t arms[VFFT_ILFD_MAX_K + 1];
    double ns[VFFT_ILFD_MAX_K + 1];
    int cand[VFFT_ILFD_MAX_K + 1], n, i, best, reps;
    n = vfft_ilfd_tw_candidates(p, cache_bytes, cand, VFFT_ILFD_MAX_K + 1);
    if (n <= 1) { vfft_ilfd_apply_tw(p, 0); return 0; }
    vfft_ilfd_apply_tw(p, 0);
    reps = _ilfd_race_reps(p, zin, zout);
    for (i = 0; i < n; i++) {
        cx[i].p = p; cx[i].zin = zin; cx[i].zout = zout; cx[i].s = -1;
        cx[i].gl = cx[i].gord = cx[i].msz = 0; cx[i].tw = cand[i];
        snprintf(cx[i].name, sizeof cx[i].name, "tw%d", cand[i]);
        arms[i].name = cx[i].name; arms[i].run = _ilfd_race_arm_run; arms[i].ctx = &cx[i];
    }
    {
        const vfft_race_proto_t proto = { VFFT_ILFD_RACE_ROUNDS, reps, VFFT_RACE_MIN, 1, 1, NULL, NULL };
        best = vfft_race_run(&proto, arms, n, ns);
        if (best < 0) best = 0;
        _ilfd_race_audit(arms, ns, n, reps, p->N);
    }
    if (_ilfd_race_log()) {
        fprintf(stderr, "[k1fd-race] N=%d tile: reps=%d", p->N, reps);
        for (i = 0; i < n; i++) fprintf(stderr, " %s=%.1f", cx[i].name, ns[i]);
        fprintf(stderr, " -> %s\n", cx[best].name);
    }
    vfft_ilfd_apply_tw(p, cand[best]);
    return cand[best];
}

#endif /* VFFT_IL_FLATDIT_RACE_H */
