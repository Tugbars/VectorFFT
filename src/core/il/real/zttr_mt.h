/* zttr_mt.h - ZTT-r's THREADED arms: the ZTT's threaded walk (il/rank1/
 * ztt_mt.h) with the Hermitian fold staying FUSED.
 *
 * ZTT-r is the ZTT at M = N/2 with the fold inside its last forward stage
 * (the terminator tlfhc) and its first backward stage (the ingest t0h). Every
 * stage between is the ZTT's own, and the ZTT's threaded walk already cuts
 * those by units -- columns (the ingest, the backward's last stage), groups
 * (a mid), tiles (the tiled prefix). The two fused kernels are loops over
 * the same columns, so they take a RANGE the same way:
 *   tlfhc  column quads k of [0, L/2). The run stays in the destination
 *          plane (no scratch), where the stage's input is split blocks and
 *          its output interleaved pairs: the block at a range boundary is
 *          read by both neighbours and overwritten by both, so each
 *          boundary block is SNAPSHOT before the workers start -- one
 *          range takes it as its carry, the other as its last partner
 *          block. The peeled centre column is computed before the workers
 *          start and stored after they join.
 *   t0h    column pairs k of [0, ncol/2): X is read only and every pair
 *          writes its own blocks; the centre column after the join.
 * Two arms, the ZTT's: mt=1 BLOCKS (every stage cut by units, one fork-join
 * a stage), mt=2 TILES (each tile's whole prefix on one thread). The arm is
 * a RACED plan input, banked per thread count (eng=zttr ... mt=); this
 * header is the execution only. No separate fold pass and no second
 * fork-join for it: zr2c's threaded form pays both.
 *
 * Included by vfft.c ONLY, after ztt_mt.h (the pool's state is per
 * translation unit). Nesting is forbidden by the pool's law: this runs on
 * the caller thread of a plan whose own nthreads > 1, never from a worker.
 */
#ifndef VFFT_ZTTR_MT_H
#define VFFT_ZTTR_MT_H

#include "zttr.h"
#include "il/rank1/ztt_mt.h"

extern long _vfft_zttr_mt_count;   /* vfft.c: the engagement counter */

#if defined(__AVX2__)
typedef struct { _zttr_job_fn fn; _zttr_job_t jb; const double *a; double *b; } _zttr_mt_arg;
static void _zttr_mt_tramp(void *v)
{
    _zttr_mt_arg *w = (_zttr_mt_arg *)v;
    if (w->jb.lo < w->jb.hi)
        _zttr_call_job(w->fn, &w->jb, w->a, w->b);
}

/* bind the arm for the plan's T; 0 = declined (T < 2, a terminator form
 * without ranges, or TILES without at least two tiles) */
static inline int vfft_zttr_mt_bind(vfft_zttr_plan_t *p, int T, int arm)
{
    const vfft_ztt_plan_t *zt = p->zt;
    p->mt = 0; p->mt_t = 0;
    if (T < 2 || arm < 1 || arm > 2 || p->blocked != 2) return 0;
    if (arm == 2 && (zt->tile == 0 || (size_t)p->M / zt->tile < 2)) return 0;
    p->mt = arm;
    p->mt_t = T;
    return 1;
}

/* Returns 1 when it ran threaded, 0 when the caller must run serial (no arm
 * bound, or a live pool clamped below the bound T: the verdict was raced at
 * that T). in -> out through the plane W, as _zttr_run_fwd / _zttr_run_bwd. */
static inline int vfft_zttr_execute_mt(const vfft_zttr_plan_t *p, const double *in, double *W, double *out, int bwd)
{
    const vfft_ztt_plan_t *zt = p->zt;
    const int T = thread_pool_workers_for(p->mt_t), nf = zt->nf;
    const size_t M = (size_t)p->M, tile = zt->tile;
    const double *tw = bwd ? zt->twb : zt->tw;
    const vfft_ztt_kfn *st = bwd ? zt->st_bwd : zt->st_fwd;
    _zttr_mt_arg a[THREAD_POOL_MAX_DISPATCH];
    double snapL[THREAD_POOL_MAX_DISPATCH][64], snapH[THREAD_POOL_MAX_DISPATCH][64];   /* R <= 8 legs x 8 doubles */
    int s, w;
    if (p->mt <= 0 || T < 2 || T != p->mt_t || p->blocked != 2) return 0;
    if (p->mt == 2 && (tile == 0 || M / tile < 2)) return 0;
    if (!bwd)
        _ztt_mt_columns(T, st[0], in, W, 0, 0, 0, 0, (size_t)zt->ncol, 0, (size_t)zt->ncol, zt->rb);
    else
    {   /* the fused ingest: ranges of column pairs, then the centre column */
        const _zttr_job_fn fn = zt->chain[0] == 4 ? _zttr_t0h4 : _zttr_t0h8;
        const long np = (long)(zt->ncol / 4);   /* k steps by 2 over [0, ncol/2) */
        for (w = 0; w < T; w++)
        {
            a[w].fn = fn; a[w].a = in; a[w].b = W;
            a[w].jb.p = p; a[w].jb.mode = 0; a[w].jb.cen = NULL; a[w].jb.carry = NULL; a[w].jb.edge = NULL;
            a[w].jb.lo = 2 * (np * w / T); a[w].jb.hi = 2 * (np * (w + 1) / T);
        }
        thread_pool_run(T, _zttr_mt_tramp, a, sizeof a[0]);
        {
            const _zttr_job_t jc = { p, 0, 0, 2, NULL, NULL, NULL };
            _zttr_call_job(fn, &jc, in, W);
        }
    }
    if (p->mt == 2)
        _ztt_mt_tiles(T, zt, 1, tw, W, 0, 0, bwd, M / tile);
    for (s = 1; s < nf - 1; s++)
    {
        const size_t RL = (size_t)zt->L[s] * (size_t)zt->chain[s];
        if (p->mt == 1 || tile % RL)
            _ztt_mt_groups(T, st[s], W, 2 * RL, tw + zt->twoff[s], (size_t)zt->L[s], (size_t)zt->Gs[s], (size_t)zt->L[s]);
    }
    if (!bwd)
    {   /* the fused terminator: the centre first, ranges of column quads, the centre's store last */
        const _zttr_job_fn fn = p->R == 4 ? _zttr_tlfhc4 : _zttr_tlfhc8;
        const long nq = (p->L / 2) / 4;
        double cen[16];
        {
            const _zttr_job_t jc = { p, 0, 0, 1, cen, NULL, NULL };
            _zttr_call_job(fn, &jc, W, out);
        }
        for (w = 0; w < T; w++)
        {
            a[w].fn = fn; a[w].a = W; a[w].b = out;
            const long lo = 4 * (nq * w / T), hi = 4 * (nq * (w + 1) / T);
            a[w].jb.p = p; a[w].jb.mode = 0; a[w].jb.cen = cen; a[w].jb.carry = NULL; a[w].jb.edge = NULL;
            a[w].jb.lo = lo; a[w].jb.hi = hi;
            if (lo < hi)
            {   /* the two boundary blocks, raw, before any worker stores */
                for (int r = 0; r < p->R; r++)
                {
                    if (lo > 0) memcpy(snapL[w] + 8 * r, W + 2 * ((size_t)r * (size_t)p->L + (size_t)(p->L - lo)), 8 * sizeof(double));
                    if (2 * hi != p->L) memcpy(snapH[w] + 8 * r, W + 2 * ((size_t)r * (size_t)p->L + (size_t)(p->L - hi)), 8 * sizeof(double));
                }
                if (lo > 0) a[w].jb.carry = snapL[w];
                if (2 * hi != p->L) a[w].jb.edge = snapH[w];
            }
        }
        thread_pool_run(T, _zttr_mt_tramp, a, sizeof a[0]);
        for (int r = 0; r < p->R; r++)
        {
            double *o = out + 2 * ((size_t)r * (size_t)p->L + (size_t)(p->L / 2));
            o[0] = cen[2 * r];
            o[1] = cen[2 * r + 1];
        }
    }
    else
    {   /* the backward's last stage: the ZTT's, cut by columns */
        const size_t L = (size_t)zt->L[nf - 1];
        const int Rl = zt->chain[nf - 1];
        const vfft_ztt_kfn last = (W != out && zt->tl_bwd_plane) ? zt->tl_bwd_plane : st[nf - 1];
        _ztt_mt_columns(T, last, W, out, 1, tw + zt->twoff[nf - 1], 0, (Rl - 1) * 8, L, L, L, 0);
    }
    _vfft_zttr_mt_count++;
    return 1;
}
#else
static inline int vfft_zttr_mt_bind(vfft_zttr_plan_t *p, int T, int arm) { (void)T; (void)arm; p->mt = 0; p->mt_t = 0; return 0; }
static inline int vfft_zttr_execute_mt(const vfft_zttr_plan_t *p, const double *in, double *W, double *out, int bwd)
{ (void)p; (void)in; (void)W; (void)out; (void)bwd; return 0; }
#endif

#endif /* VFFT_ZTTR_MT_H */
