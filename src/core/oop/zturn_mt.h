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
            radix4_z_s0t_r4_fwd_avx2(a->zin + 2 * k0, 0,
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
                radix4_z_stf_r4_fwd_avx2(p->plane + 2 * k0, 0,
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
            (p->t2q ? radix8_z_stf2_r4_fwd_avx2
                    : radix8_z_stf_r4_fwd_avx2)(
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
    _zt_mt_arg a[64];
    int t, nd = 0;
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
    for (t = 1; t < T; t++)
        _stride_pool_dispatch(&_stride_workers[nd++], _zt_mt_tramp,
                              &a[t]);
    _zt_mt_tramp(&a[0]);
    if (nd)
        _stride_pool_wait_all();
}

#endif /* VFFT_OOP_ZTURN_MT_H */
