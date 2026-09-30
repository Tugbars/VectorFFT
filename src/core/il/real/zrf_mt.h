/* zrf_mt.h - the real flat DIT's THREADED form: the c2c flat DIT's tiles arm
 * (il/rank1/il_flatdit_mt.h) on the real engine's first level.
 *
 * The first level carries (R0-1)/R0 of the work; the levels under it are
 * the real run of N/R0 and run serial. Its pieces are loops over
 * independent units, each cut over the plan's threads as a loop restriction
 * (same kernels, same tables, same per-element order: the threaded output is
 * bitwise the serial plan's, gated in gauntlet/zrf_mt_check.c):
 *   the leaf      by COLUMN ranges (whole vectors of four). A range's
 *                 complex digits land in their own columns of blocks 1..h;
 *                 its real digit-0 run cannot take the same pointer shift
 *                 (one double per column against two), so it lands inside
 *                 the unused half of block 0 at twice the range's start and
 *                 is compacted after the join (N/R0 doubles, serial). c2r
 *                 spreads the run the same way before its leaf.
 *   a wide stage  by COLUMN ranges of every block at once (the stage's
 *                 twiddles are constant along a block's run).
 *   the tiles     by ranges of the WALK ORDER (zrf.h: ascending
 *                 min(Q, P - Q)): a worker walks its tiles depth-first,
 *                 each tile's stages and its order sweep (c2r: its gather
 *                 and its transposed stages). A contiguous range of the
 *                 walk owns a contiguous set of output residues, so the
 *                 workers' sweeps write disjoint lines. Slot 0 -- the
 *                 caller -- also runs every level under the first, and
 *                 takes that much less of the tiles.
 * One fork-join for the leaf, one per wide stage, one for the tiles. The
 * first level must be tiled with at least two tiles; the threaded verdict is
 * raced at the plan's T and banked on the threaded plan's row (eng=zrf ...
 * mt=1); this header is the execution only.
 *
 * Included by vfft.c ONLY, after zrf.h and the pool (the pool's state is per
 * translation unit). Nesting is forbidden by the pool's law: this runs on
 * the caller thread of a plan whose own nthreads > 1, never from a worker.
 */
#ifndef VFFT_ZRF_MT_H
#define VFFT_ZRF_MT_H

#include "zrf.h"

extern long _vfft_zrf_mt_count;   /* vfft.c: the engagement counter */

#if defined(__AVX2__)
typedef struct {
    const vfft_zrf_plan_t *p;
    const double *in;
    double *out;
    const vfft_ilfd_call_t *rec;   /* kind 1: the wide record */
    size_t lo, hi;                 /* columns (kinds 0, 1) or entries of the tiles' walk order (kind 2) */
    int kind, bwd, deep;           /* 0 the leaf, 1 a wide stage, 2 tiles; deep = this slot runs the levels under the first */
} _zrf_mt_arg;

static void _zrf_mt_tramp(void *v)
{
    const _zrf_mt_arg *a = (const _zrf_mt_arg *)v;
    const vfft_zrf_plan_t *p = a->p;
    const _zrf_level_t *lv = &p->lv[0];
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
        const int ntl = lv->ns - lv->nwide;
        size_t k;
        int i, j;
        if (a->deep && a->bwd)
        {   /* the levels under the first, backward: their last leaf writes the first level's real run */
            p->mb(a->in, NULL, p->lv[p->J - 1].plane, NULL, NULL, NULL, p->MJ, 0, 1, 0, 1);
            for (j = p->J - 1; j >= 1; j--)
            {
                const _zrf_level_t *l2 = &p->lv[j];
                _zrf_level_bwd(l2, a->in);
                l2->lb(l2->plane, NULL, p->lv[j - 1].plane, NULL, NULL, NULL, l2->D, 0, l2->D, 0, l2->D);
            }
        }
        for (k = a->lo; k < a->hi; k++)
        {
            const size_t t = lv->tord[k];
            if (!a->bwd)
            {
                for (i = lv->nwide; i < lv->ns; i++) _ilfd_call(lv->fd, &lv->cf[i], t, lv->plane, lv->plane);
                _zrf_sweep(lv, a->out, (t - lv->t0) * lv->bpt, (t - lv->t0 + 1) * lv->bpt);
            }
            else
            {
                _zrf_gather(lv, a->in, (t - lv->t0) * lv->bpt, (t - lv->t0 + 1) * lv->bpt);
                for (i = 0; i < ntl; i++) _ilfd_call(lv->fd, &lv->ct[i], t, lv->plane, lv->plane);
            }
        }
        if (a->deep && !a->bwd)
        {   /* the levels under the first, forward: from the first level's real run */
            const double *src = lv->plane;
            for (j = 1; j < p->J; j++)
            {
                const _zrf_level_t *l2 = &p->lv[j];
                l2->lf(src, NULL, l2->plane, NULL, NULL, NULL, l2->D, 0, l2->D, 0, l2->D);
                _zrf_level_fwd(l2, a->out);
                src = l2->plane;
            }
            p->mf(src, NULL, a->out, NULL, NULL, NULL, 1, 0, p->MJ, 0, 1);
        }
    }
}

/* bind the threaded form for the plan's T; 0 = declined (T < 2, or the first
 * level is untiled or has fewer than two tiles) */
static inline int vfft_zrf_mt_bind(vfft_zrf_plan_t *p, int T)
{
    p->mt = 0; p->mt_t = 0;
    if (T < 2 || T > THREAD_POOL_MAX_DISPATCH || !p->lv[0].tw || p->lv[0].t1 - p->lv[0].t0 < 2) return 0;
    p->mt = 1;
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

/* Returns 1 when it ran threaded, 0 when the caller must run serial (no form
 * bound, or a live pool clamped below the bound T: the verdict was raced at
 * that T). */
static inline int vfft_zrf_execute_mt(const vfft_zrf_plan_t *p, const double *in, double *out, int bwd)
{
    const _zrf_level_t *lv = &p->lv[0];
    const int T = thread_pool_workers_for(p->mt_t);
    _zrf_mt_arg a[THREAD_POOL_MAX_DISPATCH];
    size_t nt, n0, rest;
    int w, i;
    if (p->mt <= 0 || T < 2 || T != p->mt_t || !lv->tw) return 0;
    nt = lv->t1 - lv->t0;
    if (nt < 2) return 0;
    for (w = 0; w < T; w++)
    {
        a[w].p = p; a[w].in = in; a[w].out = out; a[w].rec = NULL; a[w].bwd = bwd; a[w].deep = 0;
    }
    /* the tiles: slot 0 runs the levels under the first (1/R0 of the work) and that
     * much less of the tiles; the others share the rest evenly */
    {
        const double f = 1.0 / (double)lv->R, s0 = 1.0 / (double)T - f;
        n0 = s0 > 0.0 ? (size_t)((double)nt * s0 / (1.0 - f)) : 0;
        rest = nt - n0;
    }
    if (!bwd)
    {
        for (w = 0; w < T; w++) a[w].kind = 0;
        _zrf_mt_cols(a, T, lv->D);
        thread_pool_run(T, _zrf_mt_tramp, a, sizeof a[0]);
        for (w = 0; w < T; w++)   /* compact the real run: ascending, a range never lands on a later one's source */
            if (a[w].lo && a[w].hi > a[w].lo)
                memmove(lv->plane + a[w].lo, lv->plane + 2 * a[w].lo, (a[w].hi - a[w].lo) * sizeof(double));
        for (i = 0; i < lv->nwide; i++)
        {
            for (w = 0; w < T; w++) { a[w].kind = 1; a[w].rec = &lv->cf[i]; }
            _zrf_mt_cols(a, T, lv->cf[i].count);
            thread_pool_run(T, _zrf_mt_tramp, a, sizeof a[0]);
        }
    }
    for (w = 0; w < T; w++)
    {
        a[w].kind = 2; a[w].rec = NULL; a[w].deep = (w == 0);
        if (w == 0) { a[w].lo = 0; a[w].hi = n0; }
        else
        {
            a[w].lo = n0 + rest * (size_t)(w - 1) / (size_t)(T - 1);
            a[w].hi = n0 + rest * (size_t)w / (size_t)(T - 1);
        }
    }
    thread_pool_run(T, _zrf_mt_tramp, a, sizeof a[0]);
    if (bwd)
    {
        const int ntl = lv->ns - lv->nwide;
        for (w = 0; w < T; w++) a[w].deep = 0;
        for (i = ntl; i < lv->ns; i++)
        {
            for (w = 0; w < T; w++) { a[w].kind = 1; a[w].rec = &lv->ct[i]; }
            _zrf_mt_cols(a, T, lv->ct[i].count);
            thread_pool_run(T, _zrf_mt_tramp, a, sizeof a[0]);
        }
        for (w = 0; w < T; w++) { a[w].kind = 0; a[w].rec = NULL; }
        _zrf_mt_cols(a, T, lv->D);
        for (w = T - 1; w >= 1; w--)   /* spread the real run: descending, the mirror of the compaction */
            if (a[w].lo && a[w].hi > a[w].lo)
                memmove(lv->plane + 2 * a[w].lo, lv->plane + a[w].lo, (a[w].hi - a[w].lo) * sizeof(double));
        thread_pool_run(T, _zrf_mt_tramp, a, sizeof a[0]);
    }
    _vfft_zrf_mt_count++;
    return 1;
}
#else
static inline int vfft_zrf_mt_bind(vfft_zrf_plan_t *p, int T) { (void)T; p->mt = 0; p->mt_t = 0; return 0; }
static inline int vfft_zrf_execute_mt(const vfft_zrf_plan_t *p, const double *in, double *out, int bwd)
{ (void)p; (void)in; (void)out; (void)bwd; return 0; }
#endif

#endif /* VFFT_ZRF_MT_H */
