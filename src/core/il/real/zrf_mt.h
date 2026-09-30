/* zrf_mt.h - the real flat DIT's THREADED forms: the c2c flat DIT's tiles arm
 * (il/rank1/il_flatdit_mt.h) on the real engine's levels, in two arms.
 *
 * Every cut is a loop restriction (same kernels, same tables, same
 * per-element order: the threaded output is bitwise the serial plan's,
 * gated in gauntlet/zrf_mt_check.c). The units of a level:
 *   the leaf      by COLUMN ranges of whole vectors. A range's complex
 *                 digits land in their own columns of blocks 1..h; its real
 *                 digit-0 run cannot take the same pointer shift (one double
 *                 per column against two), so it lands inside the unused half
 *                 of block 0 at twice the range's start and is compacted after
 *                 the join (N/R0 doubles, serial). c2r spreads the run the
 *                 same way before its leaf.
 *   a wide stage  by COLUMN ranges of every block at once (a stage's twiddles
 *                 are constant along a block's run).
 *   the tiles     by ranges of the WALK ORDER (zrf.h: ascending min(Q, P - Q));
 *                 a contiguous range of a level's walk owns a contiguous set
 *                 of that level's output residues, so the workers' sweeps
 *                 write disjoint lines.
 * The two arms, raced at the plan's T and banked on the threaded plan's row
 * (eng=zrf ... mt=1|2):
 *   mt=1 FIRST    the first level's units cut; the levels under it (the real
 *                 run of N/R0, 1/R0 of the work) run serial on the caller
 *                 inside the tiles dispatch, which takes that much less of
 *                 the tiles; each tile's sweep follows its stages from L1.
 *                 One fork-join for the leaf, one per wide stage, one for the
 *                 tiles.
 *   mt=2 LEVELS   every level with a tile per worker is cut: the leaves
 *                 level by level, every level's wide records of one index in
 *                 one dispatch, then ONE tiles dispatch over all those
 *                 levels' walks, the ranges balanced by tile work (the small
 *                 levels and the mono serial on slot 0, charged to it); a
 *                 worker sweeps its tiles in GROUPS in output order (zrf.h:
 *                 runs instead of a comb per tile). More fork-joins, no
 *                 serial run on the critical path.
 * Measured 2026-09-30 at N = 531441, T = 8: under FIRST the caller's slot
 * ran 181 us (the serial levels) where the others ran 122 for r2c, while
 * c2r's tile jobs balanced against it by themselves; LEVELS took r2c from
 * 354 to 309 us and c2r from 227 to 250. Hence two arms, the race decides.
 *
 * This header is the execution only. Included by vfft.c ONLY, after zrf.h
 * and the pool (the pool's state is per translation unit). Nesting is
 * forbidden by the pool's law: this runs on the caller thread of a plan
 * whose own nthreads > 1, never from a worker.
 */
#ifndef VFFT_ZRF_MT_H
#define VFFT_ZRF_MT_H

#include "zrf.h"

extern long _vfft_zrf_mt_count;   /* vfft.c: the engagement counter */

#if defined(__AVX2__)
typedef struct {
    const vfft_zrf_plan_t *p;
    const _zrf_level_t *lv;        /* kinds 0, 1: the level; kind 2 under FIRST: the first level */
    const double *in;              /* the leaf's input (r2c) or the half spectrum (c2r) */
    double *out;                   /* the half spectrum (r2c) or the leaf's output (c2r) */
    const vfft_ilfd_call_t *rec;   /* kind 1: the wide record */
    size_t lo, hi;                 /* kinds 0, 1: columns; kind 2 under FIRST: the walk range */
    const size_t *tlo, *thi;       /* kind 2 under LEVELS: this worker's walk range per threaded level */
    int kind, bwd, deep, arm;      /* 0 the leaf, 1 a wide stage, 2 tiles; deep = this slot runs the serial levels */
} _zrf_mt_arg;

/* the serial levels and the mono, forward from level j0 - 1's real run */
static inline void _zrf_serial_tail_fwd(const vfft_zrf_plan_t *p, int j0, double *out)
{
    const double *src = p->lv[j0 - 1].plane;
    int j;
    for (j = j0; j < p->J; j++)
    {
        const _zrf_level_t *lv = &p->lv[j];
        lv->lf(src, NULL, lv->plane, NULL, NULL, NULL, lv->D, 0, lv->D, 0, lv->D);
        _zrf_level_fwd(lv, out);
        src = lv->plane;
    }
    p->mf(src, NULL, out, NULL, NULL, NULL, 1, 0, p->MJ, 0, 1);
}
/* and backward, down to level j0 - 1's real run */
static inline void _zrf_serial_tail_bwd(const vfft_zrf_plan_t *p, int j0, const double *in)
{
    int j;
    p->mb(in, NULL, p->lv[p->J - 1].plane, NULL, NULL, NULL, p->MJ, 0, 1, 0, 1);
    for (j = p->J - 1; j >= j0; j--)
    {
        const _zrf_level_t *lv = &p->lv[j];
        _zrf_level_bwd(lv, in);
        lv->lb(lv->plane, NULL, p->lv[j - 1].plane, NULL, NULL, NULL, lv->D, 0, lv->D, 0, lv->D);
    }
}

/* the tiles at walk entries [k0, k1) of one level, this direction */
static inline void _zrf_mt_tiles_run(const _zrf_level_t *lv, const double *in, double *out, size_t k0, size_t k1,
                                     int bwd, int group)
{
    if (!bwd)
        _zrf_tiles_fwd(lv, out, k0, k1, group);
    else
    {
        const int ntl = lv->ns - lv->nwide;
        size_t k;
        int i;
        for (k = k0; k < k1; k++)
        {
            const size_t t = lv->tord[k];
            _zrf_gather(lv, in, (t - lv->t0) * lv->bpt, (t - lv->t0 + 1) * lv->bpt);
            for (i = 0; i < ntl; i++) _ilfd_call(lv->fd, &lv->ct[i], t, lv->plane, lv->plane);
        }
    }
}

static void _zrf_mt_tramp(void *v)
{
    const _zrf_mt_arg *a = (const _zrf_mt_arg *)v;
    const _zrf_level_t *lv = a->lv;
    if (a->kind == 0)
    {
        if (a->lo >= a->hi) return;
        if (!a->bwd) lv->lf(a->in + a->lo, NULL, lv->plane + 2 * a->lo, NULL, NULL, NULL, lv->D, 0, lv->D, 0, a->hi - a->lo);
        else lv->lb(lv->plane + 2 * a->lo, NULL, a->out + a->lo, NULL, NULL, NULL, lv->D, 0, lv->D, 0, a->hi - a->lo);
        return;
    }
    if (a->kind == 1)
    {
        vfft_ilfd_call_t c;
        if (a->lo >= a->hi) return;
        c = *a->rec;
        c.count = a->hi - a->lo;
        _ilfd_call(lv->fd, &c, 1, lv->plane + 2 * a->lo, lv->plane + 2 * a->lo);
        return;
    }
    {
        const vfft_zrf_plan_t *p = a->p;
        const int j0 = a->arm == 2 ? p->mt_lv : 1;   /* the first serial level */
        int j;
        if (a->deep && a->bwd) _zrf_serial_tail_bwd(p, j0, a->in);
        if (a->arm == 2)
            for (j = 0; j < p->mt_lv; j++)
                _zrf_mt_tiles_run(&p->lv[j], a->in, a->out, a->tlo[j], a->thi[j], a->bwd, 1);
        else
            _zrf_mt_tiles_run(lv, a->in, a->out, a->lo, a->hi, a->bwd, 0);
        if (a->deep && !a->bwd) _zrf_serial_tail_fwd(p, j0, a->out);
    }
}

/* a level can run threaded when it is tiled with at least one tile per worker */
static inline int _zrf_mt_level_ok(const _zrf_level_t *lv, int T)
{
    return lv->tw && lv->t1 - lv->t0 >= (size_t)T;
}

/* Bind an arm for the plan's T. FIRST needs the first level tiled with two
 * tiles; LEVELS the first level with a tile per worker, and cuts every
 * worker's walk range of each threaded level, balanced by tile work (a
 * tile's width times its stages plus its sweep) with the serial levels'
 * work charged to slot 0. 0 = declined. */
static inline int vfft_zrf_mt_bind(vfft_zrf_plan_t *p, int T, int arm)
{
    p->mt = 0; p->mt_t = 0; p->mt_lv = 0;
    if (T < 2 || T > THREAD_POOL_MAX_DISPATCH || arm < 1 || arm > 2 || !p->lv[0].tw) return 0;
    if (arm == 1)
    {
        if (p->lv[0].t1 - p->lv[0].t0 < 2) return 0;
        p->mt_lv = 1;
    }
    else
    {
        double wt[VFFT_ZRF_MAX_LV], W = 0, deep = 0, s0, rest;
        int j, w;
        if (!_zrf_mt_level_ok(&p->lv[0], T)) return 0;
        for (j = 0; j < p->J && _zrf_mt_level_ok(&p->lv[j], T); j++) p->mt_lv = j + 1;
        for (j = 0; j < p->mt_lv; j++)
        {
            const _zrf_level_t *lv = &p->lv[j];
            wt[j] = (double)lv->tw * (double)(lv->ns - lv->nwide + 1);   /* per tile: its stages and its sweep */
            W += wt[j] * (double)(lv->t1 - lv->t0);
        }
        for (j = p->mt_lv; j < p->J; j++)
            deep += (double)p->lv[j].Nj * (double)(p->lv[j].ns + 2) / 2.0;   /* a serial level: leaf, stages, sweep */
        deep += (double)p->NJ * 4.0;
        s0 = (W + deep) / (double)T - deep;
        if (s0 < 0.0) s0 = 0.0;
        rest = W - s0;
        /* the concatenated walks of the threaded levels, in level order, cut at every
         * worker's [lo, hi) of the work axis; then made exact and contiguous */
        for (w = 0; w < T; w++)
        {
            const double lo = w == 0 ? 0.0 : s0 + rest * (double)(w - 1) / (double)(T - 1);
            const double hi = w == 0 ? s0 : s0 + rest * (double)w / (double)(T - 1);
            double base = 0.0;
            for (j = 0; j < p->mt_lv; j++)
            {
                const _zrf_level_t *lv = &p->lv[j];
                const size_t nt = lv->t1 - lv->t0;
                const double top = base + wt[j] * (double)nt;
                const double a = lo > base ? lo : base, b = hi < top ? hi : top;
                size_t ka = 0, kb = 0;
                if (b > a)
                {
                    ka = (size_t)((a - base) / wt[j] + 0.5);
                    kb = (size_t)((b - base) / wt[j] + 0.5);
                    if (kb > nt) kb = nt;
                    if (ka > kb) ka = kb;
                }
                p->mt_lo[w][j] = ka; p->mt_hi[w][j] = kb;
                base = top;
            }
        }
        for (j = 0; j < p->mt_lv; j++)
        {
            size_t at = 0;
            for (w = 0; w < T; w++)
            {
                size_t kb = p->mt_hi[w][j];
                if (kb < at) kb = at;
                p->mt_lo[w][j] = at; p->mt_hi[w][j] = kb; at = kb;
            }
            p->mt_hi[T - 1][j] = p->lv[j].t1 - p->lv[j].t0;
        }
    }
    p->mt = arm;
    p->mt_t = T;
    return 1;
}

/* column ranges in whole vectors of four: worker w takes [4*(q*w/T), 4*(q*(w+1)/T)),
 * the last one the remainder */
static inline void _zrf_mt_cols(_zrf_mt_arg *a, int T, size_t D)
{
    const size_t q = D / 4;
    int w;
    for (w = 0; w < T; w++)
    {
        a[w].lo = 4 * (q * (size_t)w / (size_t)T);
        a[w].hi = (w == T - 1) ? D : 4 * (q * (size_t)(w + 1) / (size_t)T);
    }
}

/* the leaf of one level by column ranges, then the real run compacted (r2c) */
static inline void _zrf_mt_leaf_fwd(const _zrf_level_t *lv, const double *src, _zrf_mt_arg *a, int T)
{
    int w;
    for (w = 0; w < T; w++) { a[w].kind = 0; a[w].lv = lv; a[w].in = src; a[w].rec = NULL; }
    _zrf_mt_cols(a, T, lv->D);
    thread_pool_run(T, _zrf_mt_tramp, a, sizeof a[0]);
    for (w = 0; w < T; w++)   /* compact: ascending, a range never lands on a later one's source */
        if (a[w].lo && a[w].hi > a[w].lo)
            memmove(lv->plane + a[w].lo, lv->plane + 2 * a[w].lo, (a[w].hi - a[w].lo) * sizeof(double));
}
/* the real run spread, then the leaf of one level by column ranges into dst (c2r) */
static inline void _zrf_mt_leaf_bwd(const _zrf_level_t *lv, double *dst, _zrf_mt_arg *a, int T)
{
    int w;
    for (w = 0; w < T; w++) { a[w].kind = 0; a[w].lv = lv; a[w].out = dst; a[w].rec = NULL; }
    _zrf_mt_cols(a, T, lv->D);
    for (w = T - 1; w >= 1; w--)   /* spread: descending, the mirror of the compaction */
        if (a[w].lo && a[w].hi > a[w].lo)
            memmove(lv->plane + 2 * a[w].lo, lv->plane + a[w].lo, (a[w].hi - a[w].lo) * sizeof(double));
    thread_pool_run(T, _zrf_mt_tramp, a, sizeof a[0]);
}
/* one wide record of one level by column ranges */
static inline void _zrf_mt_wide(const _zrf_level_t *lv, const vfft_ilfd_call_t *rec, _zrf_mt_arg *a, int T)
{
    int w;
    for (w = 0; w < T; w++) { a[w].kind = 1; a[w].lv = lv; a[w].rec = rec; }
    _zrf_mt_cols(a, T, rec->count);
    thread_pool_run(T, _zrf_mt_tramp, a, sizeof a[0]);
}

/* Returns 1 when it ran threaded, 0 when the caller must run serial (no arm
 * bound, or a live pool clamped below the bound T: the verdict was raced at
 * that T). */
static inline int vfft_zrf_execute_mt(const vfft_zrf_plan_t *p, const double *in, double *out, int bwd)
{
    const int T = thread_pool_workers_for(p->mt_t);
    const _zrf_level_t *lv = &p->lv[0];
    _zrf_mt_arg a[THREAD_POOL_MAX_DISPATCH];
    const int nlv = p->mt == 2 ? p->mt_lv : 1;
    int w, j, i;
    if (p->mt <= 0 || T < 2 || T != p->mt_t || p->mt_lv < 1 || !lv->tw) return 0;
    for (w = 0; w < T; w++)
    {
        a[w].p = p; a[w].lv = lv; a[w].in = in; a[w].out = out; a[w].rec = NULL; a[w].bwd = bwd; a[w].deep = 0; a[w].arm = p->mt;
        a[w].tlo = p->mt_lo[w]; a[w].thi = p->mt_hi[w];
    }
    if (p->mt == 1)
    {   /* FIRST: slot 0 runs the serial levels (1/R0 of the work) and that much less of
         * the tiles; the others share the rest evenly */
        const size_t nt = lv->t1 - lv->t0;
        const double f = 1.0 / (double)lv->R, s0 = 1.0 / (double)T - f;
        const size_t n0 = s0 > 0.0 ? (size_t)((double)nt * s0 / (1.0 - f)) : 0, rest = nt - n0;
        for (w = 0; w < T; w++)
        {
            a[w].tlo = a[w].thi = NULL;
            if (w == 0) { a[w].lo = 0; a[w].hi = n0; }
            else { a[w].lo = n0 + rest * (size_t)(w - 1) / (size_t)(T - 1); a[w].hi = n0 + rest * (size_t)w / (size_t)(T - 1); }
        }
    }
    if (!bwd)
    {
        const double *src = in;
        _zrf_mt_arg b[THREAD_POOL_MAX_DISPATCH];
        memcpy(b, a, sizeof b);
        for (j = 0; j < nlv; j++) { _zrf_mt_leaf_fwd(&p->lv[j], src, b, T); src = p->lv[j].plane; }
        for (i = 0; ; i++)
        {
            int any = 0;
            for (j = 0; j < nlv; j++)
                if (i < p->lv[j].nwide) { _zrf_mt_wide(&p->lv[j], &p->lv[j].cf[i], b, T); any = 1; }
            if (!any) break;
        }
        for (w = 0; w < T; w++) { a[w].kind = 2; a[w].deep = (w == 0); }
        thread_pool_run(T, _zrf_mt_tramp, a, sizeof a[0]);
    }
    else
    {
        _zrf_mt_arg b[THREAD_POOL_MAX_DISPATCH];
        for (w = 0; w < T; w++) { a[w].kind = 2; a[w].deep = (w == 0); }
        thread_pool_run(T, _zrf_mt_tramp, a, sizeof a[0]);
        memcpy(b, a, sizeof b);
        for (w = 0; w < T; w++) b[w].deep = 0;
        for (i = 0; ; i++)
        {   /* the transposed wide records follow the tiled ones in ct: index ntl + i */
            int any = 0;
            for (j = 0; j < nlv; j++)
            {
                const _zrf_level_t *l2 = &p->lv[j];
                const int ntl = l2->ns - l2->nwide;
                if (ntl + i < l2->ns) { _zrf_mt_wide(l2, &l2->ct[ntl + i], b, T); any = 1; }
            }
            if (!any) break;
        }
        for (j = nlv - 1; j >= 0; j--)
            _zrf_mt_leaf_bwd(&p->lv[j], j ? p->lv[j - 1].plane : out, b, T);
    }
    _vfft_zrf_mt_count++;
    return 1;
}
#else
static inline int vfft_zrf_mt_bind(vfft_zrf_plan_t *p, int T, int arm) { (void)T; (void)arm; p->mt = 0; p->mt_t = 0; p->mt_lv = 0; return 0; }
static inline int vfft_zrf_execute_mt(const vfft_zrf_plan_t *p, const double *in, double *out, int bwd)
{ (void)p; (void)in; (void)out; (void)bwd; return 0; }
#endif

#endif /* VFFT_ZRF_MT_H */
