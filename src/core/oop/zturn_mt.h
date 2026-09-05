/* zturn_mt.h - the zturn cascade's multithreaded tile and phase kernels.
 *
 * Extracted from vfft.c as migration step 10; see
 * docs/design/refactor_migration_plan.md.
 *
 * WHY A CASCADE CAN BE THREADED AT ALL
 * ------------------------------------
 * The cascade already contains its own parallel axes - threading it is a matter
 * of restricting existing loops, not of inventing a decomposition. Each phase
 * has a natural range, and each range is a handful of pointer edits:
 *
 *   INGEST      a pure map over s0t-columns, linear in k with no tables, so a
 *               count range is TWO pointer edits.
 *   MIDS        one twiddle record per GROUP, the table walked linearly, so a
 *               group range is THREE pointer edits. Groups are the atom -
 *               powers derive in-register inside one.
 *   TERMINATOR  linear in k, and the tcut cut-form already ships the needed
 *               arithmetic; ranges are 8-column-aligned so both quad forms hold.
 *
 * Because every range is a RESTRICTION of the serving loop rather than a
 * different computation, MT output is bitwise identical to ST. That is the
 * property the gates check, and it is a consequence of this design choice.
 *
 * Stages stay ordered, with one join each - measured at roughly 100 ns.
 *
 * WHAT DECLINES TO THREAD, AND WHY
 * --------------------------------
 * Two configurations opt out rather than risk it: the tiled driver (env
 * experimental) and natord, whose rho-order table walks are not a simple range
 * restriction. Declining is the correct outcome, not a gap.
 *
 * WHAT STAYED IN vfft.c
 * ---------------------
 * The engagement counter and its accessor (mutable file-scope state - a static
 * in a header is one copy per includer), the dispatcher _zt_execute_mt (it
 * increments that counter and dereferences vfft_plan_s), and the create-time
 * racer (it belongs with the wisdom write path). This header is the kernels
 * only.
 *
 * FLOOR-LEGAL BY CONSTRUCTION
 * ---------------------------
 * Takes the zturn plan by pointer, never a vfft_plan_s. No mutable file-scope
 * state, no wisdom. Does NOT pull engine/stride_executor.h.
 */
#ifndef VFFT_OOP_ZTURN_MT_H
#define VFFT_OOP_ZTURN_MT_H

#include <stdlib.h>

#include "zturn.h"             /* vfft_zturn2_plan_t and the stage kernels */
#include "support/threads.h"   /* the pool: dispatch, wait_all */

/* ══ K=1 cascade MT (INC-Z — the IL 2D design ported to the 1D zturn
 * walk, docs/design/il2d_real_mt.md §methodology). The cascade already
 * contains the 2D axes, verified in the emitted bodies:
 *   INGEST (s0t / s0tb): a pure map over s0t-columns — everything is
 *     linear in k with no tables => a count range is TWO pointer edits.
 *   MIDS (msg): one twiddle record per GROUP, table walked linearly at
 *     (chain-1)*8 doubles/group, group g owning plane[2gD, 2(g+1)D) =>
 *     a group range is THREE pointer edits (the 2D digit split, 1D
 *     form). Groups are the atom (powers derive in-register inside).
 *   TERMINATOR (stf/stf2/stfb): linear in k (plane 4k, zout/tzq 2k) —
 *     the tcut cut-form (_vfft_zt_term_*) already ships this
 *     arithmetic; ranges are 8-column-aligned so both quad forms hold.
 * Stages stay ordered (one join each, ~100 ns per INC-2); every range
 * is a restriction of the serving loop => MT == ST bitwise. Declines:
 * tiled (env-experimental driver), natord (rho-order table walks). */

typedef struct
{
    const vfft_zturn2_plan_t *p;
    const double *zin;
    double *zout;
    int phase; /* 0 = ingest, 1 = mid stage s, 2 = terminator,
                * 3 = tile units (tiled suffix mids; the band analog),
                * 4 = FUSED tile units (suffix mids + the tile's
                *     terminator cut — t outer, q inner, term last on
                *     fwd / FIRST on bwd, the documented invariant) */
    int s, fwd;
    long lo, hi; /* s0t-column range (0/2), group range (1),
                  * unit range (3: (q,t) pairs; 4: t) */
} _zt_mt_arg;

/* the tiled suffix for one tile window (section q, tile t) */
static void _zt_mt_tile_mids(const vfft_zturn2_plan_t *p, long q, long t,
                             int fwd)
{
    const long SECD = (long)p->N / 2, w = p->tw;
    double *tile = p->plane + q * SECD + 2 * t * w;
    int s;
    if (fwd)
        for (s = p->tcut + 1; s <= p->nf - 2; s++)
        {
            const long span = p->D[s - 1];
            _vfft_zt_msg((vfft_zturn2_plan_t *)p, s, tile,
                         _vfft_zt_tw(p, s, q, t * w / span, 1),
                         w / span, 1);
        }
    else
        for (s = p->nf - 2; s >= p->tcut + 1; s--)
        {
            const long span = p->D[s - 1];
            _vfft_zt_msg((vfft_zturn2_plan_t *)p, s, tile,
                         _vfft_zt_tw(p, s, q, t * w / span, 0),
                         w / span, 0);
        }
}

static void _zt_mt_tramp(void *v)
{
    _zt_mt_arg *a = (_zt_mt_arg *)v;
    const vfft_zturn2_plan_t *p = a->p;
    const long k0 = a->lo, w = a->hi - a->lo;
    if (w <= 0)
        return;
    switch (a->phase)
    {
    case 0: /* ingest: fwd = s0t zin->plane, bwd = s0tb plane->zout */
        if (a->fwd)
            _vfft_zt_s0t_fwd_pick(p->lanes_u)(a->zin + 2 * k0, 0,
                                     p->plane + 2 * k0, 0, 0, 0,
                                     (size_t)p->N / 4, 0, 0, 0,
                                     (size_t)w);
        else
            radix4_z_s0t_r4_bwd_avx2(p->plane + 2 * k0, 0,
                                     a->zout + 2 * k0, 0, 0, 0,
                                     (size_t)p->N / 4, 0, 0, 0,
                                     (size_t)w);
        break;
    case 1: /* one mid stage, groups [lo,hi) — the digit split. The
             * emitted wrapper's per-group bumps (VERIFIED in the body,
             * radix8_z_msg_avx2.c:174): bp += 2*R*Ls (a group owns
             * R*D[s] complex — the R legs at stride D[s]), twg +=
             * (R-1)*8. So the range base is 2*g*R*D, NOT 2*g*D. */
    {
        const int s = a->s;
        const double *tbl = a->fwd ? p->twz[s] : p->twzb[s];
        _vfft_zt_msg((vfft_zturn2_plan_t *)p, s,
                     p->plane + 2 * k0 * p->chain[s] * p->D[s],
                     tbl + (size_t)k0 * (p->chain[s] - 1) * 8, w,
                     a->fwd);
        break;
    }
    case 3: /* tiled suffix, units = (q,t) pairs — self-contained */
    {
        const long NT = ((long)p->N / 4) / p->tw;
        long u;
        for (u = a->lo; u < a->hi; u++)
            _zt_mt_tile_mids(p, u / NT, u % NT, a->fwd);
        break;
    }
    case 4: /* FUSED tiles: unit = tile t across all 4 sections + its
             * terminator cut. fwd: mids then term (reads the plane);
             * bwd: term FIRST (it WRITES the tile window in all four
             * sections), then mids — hoisting q above t or reordering
             * term is the documented silently-wrong shape. */
    {
        long t, q;
        for (t = a->lo; t < a->hi; t++)
        {
            if (!a->fwd)
                _vfft_zt_term_bwd(p, a->zin, t, 0, p->tw);
            for (q = 0; q < 4; q++)
                _zt_mt_tile_mids(p, q, t, a->fwd);
            if (a->fwd)
                _vfft_zt_term_fwd((vfft_zturn2_plan_t *)p, a->zout, t,
                                  0, p->tw);
        }
        break;
    }
    default: /* terminator over s0t-columns [lo,hi) — the cut-form
              * arithmetic of _vfft_zt_term_fwd/bwd with a free base */
        if (p->chain[p->nf - 1] == 4)
        {
            if (a->fwd)
                _vfft_zt_stf4_fwd_pick(p->lanes_u)(p->plane + 2 * k0, 0,
                                         a->zout + 2 * k0, 0,
                                         p->tzq + 2 * k0, 0, 0, 0,
                                         (size_t)p->N / 4, 0,
                                         (size_t)w);
            else
                radix4_z_stf_r4_bwd_avx2(a->zin + 2 * k0, 0,
                                         p->plane + 2 * k0, 0,
                                         p->tzqb + 2 * k0, 0, 0, 0,
                                         (size_t)p->N / 4, 0,
                                         (size_t)w);
        }
        else if (a->fwd)
            _vfft_zt_stf8_fwd_pick(p->lanes_u, p->t2q)(
                p->plane + 2 * k0, 0, a->zout + k0, 0, p->tzq + k0, 0,
                0, 0, (size_t)p->N / 8, 0, (size_t)w / 2);
        else
            radix8_z_stf_r4_bwd_avx2(a->zin + k0, 0, p->plane + 2 * k0,
                                     0, p->tzqb + k0, 0, 0, 0,
                                     (size_t)p->N / 8, 0,
                                     (size_t)w / 2);
    }
}

/* one phase across T workers; ranges 8-aligned for the column phases */
static void _zt_mt_phase(const vfft_zturn2_plan_t *p, const double *zin,
                         double *zout, int phase, int s, int fwd,
                         long units, int align8, int T)
{
    _zt_mt_arg a[STRIDE_POOL_MAX_DISPATCH];
    int t;
    for (t = 0; t < T; t++)
    {
        a[t].p = p;
        a[t].zin = zin;
        a[t].zout = zout;
        a[t].phase = phase;
        a[t].s = s;
        a[t].fwd = fwd;
        a[t].lo = units * t / T;
        a[t].hi = units * (t + 1) / T;
        if (align8)
        {
            a[t].lo &= ~7L;
            a[t].hi = (t == T - 1) ? units : ((units * (t + 1) / T) & ~7L);
        }
    }
    /* the pool's fork-join: workers take a[1..T-1], the caller runs a[0] */
    stride_pool_run(T, _zt_mt_tramp, a, sizeof a[0]);
}

/* -- THE DISPATCHER AND THE RACER (migration step 20) --------------------
 * Step 10 moved the kernels and left these behind for two reasons, both now
 * resolved: _zt_execute_mt dereferences vfft_plan_s (step 15 lifted it) and
 * increments the engagement counter (step 21a gave it external linkage).
 *
 * THE RACE IS PLAN-LOCAL, WHICH IS A KNOWN GAP
 * --------------------------------------------
 * The verdict is NOT banked: the zturn chain's own wisdom rows predate
 * wisdom2, so cmt-style banking of this axis rides the wisdom2 1D wave. It
 * therefore re-races on every create, in every process. A cell that cannot
 * engage banks its "no" implicitly by leaving zt_mt at 0. Kill/force switch:
 * VFFT_ZT_NO_MT (0 forces on - the A/B hook).
 * ----------------------------------------------------------------------- */

/* DEFINED in vfft.c with external linkage. Not here: a static in a header is
 * one copy per includer, and vfft_zt_mt_passes() would then read a different
 * object than this increment writes - a confident zero while threading ran. */
extern long _vfft_zt_mt_count;

/* Returns 1 when it ran threaded, 0 = caller runs the serial walk. */
static int _zt_execute_mt(struct vfft_plan_s *h, vfft_dir_t dir,
                          const double *zin, double *zout, int T)
{
    const vfft_zturn2_plan_t *p = h->zturn;
    const long SEC = (long)p->N / 4;
    const int fwd = (dir == VFFT_FORWARD);
    const int smax = p->tiled ? p->tcut : p->nf - 2;
    int s;
    if (p->natord || p->tiled == 2)
        return 0; /* rho-order table walks; A1 = gate-only control arm */
    /* T arrives as the plan's snapshot (h->nthreads); the pool's one clamp
     * bounds it by the live pool and the arg-array size. */
    T = stride_pool_workers_for(T);
    if (T < 2 || SEC < 8 * T)
        return 0;
    if (fwd)
    {
        _zt_mt_phase(p, zin, zout, 0, 0, 1, SEC, 1, T);
        for (s = 1; s <= smax; s++)
        {
            const int Ts = p->G[s] < T ? (int)p->G[s] : T;
            _zt_mt_phase(p, zin, zout, 1, s, 1, p->G[s], 0, Ts);
        }
        if (!p->tiled)
            _zt_mt_phase(p, zin, zout, 2, 0, 1, SEC, 1, T);
        else
        {
            const long NT = SEC / p->tw;
            if (!p->tfuse)
            {
                const long u = 4 * NT;
                _zt_mt_phase(p, zin, zout, 3, 0, 1, u,
                             0, u < T ? (int)u : T);
                _zt_mt_phase(p, zin, zout, 2, 0, 1, SEC, 1, T);
            }
            else
                _zt_mt_phase(p, zin, zout, 4, 0, 1, NT, 0,
                             NT < T ? (int)NT : T);
        }
    }
    else
    {
        if (!p->tiled)
            _zt_mt_phase(p, zin, zout, 2, 0, 0, SEC, 1, T);
        else
        {
            const long NT = SEC / p->tw;
            if (!p->tfuse)
            {
                const long u = 4 * NT;
                _zt_mt_phase(p, zin, zout, 2, 0, 0, SEC, 1, T);
                _zt_mt_phase(p, zin, zout, 3, 0, 0, u,
                             0, u < T ? (int)u : T);
            }
            else
                _zt_mt_phase(p, zin, zout, 4, 0, 0, NT, 0,
                             NT < T ? (int)NT : T);
        }
        for (s = smax; s >= 1; s--)
        {
            const int Ts = p->G[s] < T ? (int)p->G[s] : T;
            _zt_mt_phase(p, zin, zout, 1, s, 0, p->G[s], 0, Ts);
        }
        _zt_mt_phase(p, zin, zout, 0, 0, 0, SEC, 1, T);
    }
    _vfft_zt_mt_count++; /* engagement, see vfft.h */
    return 1;
}

/* ── the INC-Z verdict race: serial vs threaded walk through the very
 * functions execute serves with, min-of-3 alternated on scratch, both
 * plan-local (the zturn chain's own wisdom rows are pre-wisdom2; the
 * cmt-style banking of this axis rides the wisdom2 1D wave). A cell
 * that cannot engage banks the "no" implicitly (zt_mt stays 0). Kill
 * switch VFFT_ZT_NO_MT (0 forces on — the A/B hook). */
static int _zt_execute_mt(struct vfft_plan_s *h, vfft_dir_t dir,
                          const double *zin, double *zout, int T);
/* the two arms of the zt-mt race: the same functions execute serves with */
typedef struct { struct vfft_plan_s *h; double *zi, *zo; } _zt_mt_arm_t;
static void _zt_mt_arm_st(void *v)
{
    _zt_mt_arm_t *c = (_zt_mt_arm_t *)v;
    vfft_zturn2_execute_fwd(c->h->zturn, c->zi, c->zo);
}
static void _zt_mt_arm_mt(void *v)
{
    _zt_mt_arm_t *c = (_zt_mt_arm_t *)v;
    _zt_execute_mt(c->h, VFFT_FORWARD, c->zi, c->zo, c->h->nthreads);
}
/* per-burst reseed for the ALIASED (in-place) arms: each pass transforms
 * the plane in place, so the input is restored from a pristine copy before
 * every timed burst — the arms then time exactly what the in-place execute
 * does (z -> z), never an out-of-place proxy. */
typedef struct { double *z, *z0; size_t nb; } _zt_mt_reseed_t;
static void _zt_mt_reseed(void *v)
{
    _zt_mt_reseed_t *r = (_zt_mt_reseed_t *)v;
    memcpy(r->z, r->z0, r->nb);
}

static void _zt_mt_race(struct vfft_plan_s *h)
{
    const int N = h->zturn->N;
    const int ip = (h->placement == VFFT_INPLACE);   /* placement-honest arms */
    const size_t nb = 2 * (size_t)N * sizeof(double);
    double *zi = (double *)malloc(nb);
    double *zo = ip ? zi : (double *)malloc(nb);     /* in-place: aliased z->z */
    double *z0 = ip ? (double *)malloc(nb) : NULL;   /* pristine copy to reseed */
    double st = 1e300, mt = 1e300;
    int p;
    size_t i;
    const char *ce = getenv("VFFT_ZT_NO_MT");
    if (ce)
    {
        h->zt_mt = (atoi(ce) == 0);
        free(zi); if (!ip) free(zo); free(z0);
        return;
    }
    if (!zi || !zo || (ip && !z0))
    {
        free(zi); if (!ip) free(zo); free(z0);
        return;
    }
    for (i = 0; i < 2 * (size_t)N; i++)
        zi[i] = 1.0 + 1e-6 * (double)(i & 511);
    if (ip) memcpy(z0, zi, nb);
    if (!_zt_execute_mt(h, VFFT_FORWARD, zi, zo, h->nthreads))
    {
        if (getenv("VFFT_ZT_LOG") || getenv("VFFT_IL2D_LOG"))
            fprintf(stderr, "[zt-mt] race N=%d T=%d %s: cannot engage -> "
                            "serial\n", N, h->nthreads, ip ? "ip" : "oop");
        free(zi); if (!ip) free(zo); free(z0);
        return; /* cannot engage: zt_mt stays 0 — the verdict */
    }
    if (ip) memcpy(zi, z0, nb);
    vfft_zturn2_execute_fwd(h->zturn, zi, zo); /* warm the serial arm too
                                                * — both arms hot before
                                                * the alternated timing */
    {
        _zt_mt_arm_t c = { h, zi, zo };
        _zt_mt_reseed_t rs = { zi, z0, nb };
        const vfft_race_arm_t arms[2] = { { "serial", _zt_mt_arm_st, &c },
                                          { "threaded", _zt_mt_arm_mt, &c } };
        const vfft_race_proto_t proto = { 3, 1, VFFT_RACE_MIN, 0, 0,
                                          ip ? _zt_mt_reseed : NULL,
                                          ip ? &rs : NULL }; /* min-of-3, A then B */
        double ns[2];
        (void)p;
        vfft_race_run(&proto, arms, 2, ns);
        st = ns[0];
        mt = ns[1];
    }
    h->zt_mt = (mt < st);
    if (getenv("VFFT_ZT_LOG") || getenv("VFFT_IL2D_LOG"))
        fprintf(stderr, "[zt-mt] race N=%d T=%d %s: st=%.0f mt=%.0f -> %s\n",
                N, h->nthreads, ip ? "ip" : "oop", st, mt,
                h->zt_mt ? "THREADED" : "serial");
    free(zi); if (!ip) free(zo); free(z0);
}

#endif /* VFFT_OOP_ZTURN_MT_H */
