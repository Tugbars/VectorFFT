/* zts_mt.h — zturn-ts's SECTIONED walk: the threaded form of ONE transform
 * (ZTURN-T's ztt_mt.h on the split layout, 2026-11-06).
 *
 * Every stage of the staged walk (zts.h _zts_staged_s / _zts_staged_v) is a
 * loop over units — columns (the ingest, the terminator), groups (a mid) or
 * tiles (the tiled prefix) — whose iterations touch disjoint spans and share
 * only read-only tables. A thread takes a RANGE of units: the same kernel
 * call with shifted pointers and a shorter count. Nothing is recomputed and
 * no order inside a unit changes, so the threaded output is BITWISE the
 * serial walk's. One fork-join per stage (thread_pool_run: the caller is
 * slot 0, the workers spin).
 *
 * Two arms, ZTURN-T's: mt = 1 BLOCKS — every stage cut by units, the tile
 * ignored (tiling changes group order only); mt = 2 TILES — the tiled prefix
 * cut by tiles, each tile's whole prefix on one thread, the sweeps cut by
 * units. The arm is a RACED plan parameter, banked per T on the cell's
 * threading row (eng=tcb tcmt=1|2), c2c_create_split.h _zts_sect_decide.
 *
 * At T > 1 a pow2 cell runs this walk, not its fused driver: the fusion is
 * worth 0-3 % above 2048, the threads are worth multiples. Both families:
 * S (K = 1; the contiguous batch of K < T transforms shares the dispatch,
 * every stage covering all K transforms with T / K ranges each), V (one lane
 * group, K = VW lanes: the lane-major batch too small for lane blocks).
 *
 * Column ranges are cut at multiples of VW (the S kinds run whole VW-column
 * iterations; every column count is a multiple of VW by the create's laws);
 * the V kinds run one column per iteration and cut anywhere. The plane is
 * one and shared: the threads write disjoint spans. Nesting is forbidden by
 * the pool's law: this runs on the caller thread of a plan whose own
 * nthreads > 1, never from a worker. Included by vfft.c only (the pool state
 * is per translation unit). */
#ifndef VFFT_ZTS_MT_H
#define VFFT_ZTS_MT_H

#include "split/rank1/zts.h"
#include "common/support/threads.h"

/* one thread's work item: a kernel call of one kind, or a tile range */
typedef struct
{
    int kind;                  /* 0 S ingest, 1 S mid, 2 S terminator, 3 S tile range,
                                * 4 V ingest, 5 V mid, 6 V terminator, 7 V tile range */
    const vfft_zts_plan_t *p;
    int bwd;
    const double *ire, *iim;   /* the caller's planes (shifted) */
    double *ore, *oim;
    double *W, *Wi;            /* the plane(s) (shifted for a range) */
    const double *tw;
    const size_t *rb;
    size_t Ls, Gs, OLs, count, K;
    size_t t_lo, t_hi;         /* tile ranges */
    int s;                     /* the stage (mids) */
} _zts_sect_arg;

/* the tiled prefix of ONE tile, as the staged executors run it */
static inline void _zts_sect_tile_s(const vfft_zts_plan_t *p, int bwd, const double *tw, double *W, size_t t)
{
    const size_t tile = p->tile;
    double *B = W + t * tile * 2;
    int s;
    for (s = 1; s < p->nf - 1; s++)
    {
        const size_t RL = (size_t)p->L[s] * (size_t)p->chain[s];
        if (tile % RL == 0)
            (bwd ? p->smb[s] : p->smf[s])(B, B, tw + p->twoff[s], NULL, (size_t)p->L[s], tile / RL, 0, (size_t)p->L[s]);
    }
}
static inline void _zts_sect_tile_v(const vfft_zts_plan_t *p, int bwd, const double *tw, double *W, double *Wi, size_t t)
{
    const size_t tile = p->tile, VW = (size_t)VFFT_ZTS_VW;
    double *B = W + t * tile * VW, *Bi = Wi + t * tile * VW;
    int s;
    for (s = 1; s < p->nf - 1; s++)
    {
        const size_t RL = (size_t)p->L[s] * (size_t)p->chain[s];
        if (tile % RL == 0)
            (bwd ? p->vmb[s] : p->vmf[s])(B, Bi, B, Bi, tw + p->twoff[s], NULL, (size_t)p->L[s], tile / RL, 0, (size_t)p->L[s]);
    }
}

static void _zts_sect_tramp(void *v)
{
    const _zts_sect_arg *a = (const _zts_sect_arg *)v;
    const vfft_zts_plan_t *p = a->p;
    size_t t;
    if (a->kind != 3 && a->kind != 7 && a->count == 0) return;
    switch (a->kind)
    {
    case 0: (a->bwd ? p->s0b : p->s0f)(a->ire, a->iim, a->W, NULL, a->rb, a->Ls, 0, 0, a->count); break;
    case 1: (a->bwd ? p->smb[a->s] : p->smf[a->s])(a->W, a->W, a->tw, NULL, a->Ls, a->Gs, 0, a->count); break;
    case 2: (a->bwd ? p->slb : p->slf)(a->W, a->ore, a->oim, a->tw, NULL, a->Ls, 1, a->OLs, a->count); break;
    case 3: for (t = a->t_lo; t < a->t_hi; t++) _zts_sect_tile_s(p, a->bwd, a->tw, a->W, t); break;
    case 4: (a->bwd ? p->v0b : p->v0f)(a->ire, a->iim, a->W, a->Wi, NULL, a->rb, a->Ls, 0, 0, a->count, a->K, 1, p->gp); break;
    case 5: (a->bwd ? p->vmb[a->s] : p->vmf[a->s])(a->W, a->Wi, a->W, a->Wi, a->tw, NULL, a->Ls, a->Gs, 0, a->count); break;
    case 6: (a->bwd ? p->vlb : p->vlf)(a->W, a->Wi, a->ore, a->oim, a->tw, NULL, a->Ls, 1, a->OLs, a->count, a->K, 1, p->gp); break;
    default: for (t = a->t_lo; t < a->t_hi; t++) _zts_sect_tile_v(p, a->bwd, a->tw, a->W, a->Wi, t); break;
    }
}

/* ── the S family: ONE transform's stages over T threads (the caller's
 * ranges are cut at multiples of VW). When the plan runs K < T contiguous
 * transforms, every stage covers all K at once: slot = transform * (T/K) +
 * its range, K * (T/K) slots. ── */
static inline void _zts_sect_s(const vfft_zts_plan_t *p, int T, int bwd, size_t K,
                               const double *ire, const double *iim, double *ore, double *oim, double *W)
{
    const double *tw = bwd ? p->twb : p->tw;
    const size_t N = (size_t)p->N, tile = p->tile, VW = (size_t)VFFT_ZTS_VW;
    const size_t ncol = (size_t)p->ncol, L = (size_t)p->L[p->nf - 1];
    const int per = T / (int)K;                   /* threads per transform */
    const int nslots = per * (int)K;
    _zts_sect_arg a[THREAD_POOL_MAX_DISPATCH];
    int s, w;
    size_t k;
    /* the plane per transform: the plan's plane holds ONE transform; K > 1
     * sectioned needs K planes — the slab planes (mtplane) serve, one per
     * transform beyond the first */
    for (k = 0; k < K; k++)
    {   /* the ingest: columns */
        double *Wk = k == 0 ? W : p->mtplane[k - 1];
        const size_t ni = ncol / VW;
        for (w = 0; w < per; w++)
        {
            const size_t lo = VW * (ni * (size_t)w / (size_t)per), hi = VW * (ni * (size_t)(w + 1) / (size_t)per);
            _zts_sect_arg *x = &a[k * per + w];
            memset(x, 0, sizeof *x);
            x->kind = 0; x->p = p; x->bwd = bwd;
            x->ire = ire + k * N + lo; x->iim = iim + k * N + lo; x->W = Wk;
            x->rb = p->rb + lo; x->Ls = ncol; x->count = hi - lo;
        }
    }
    thread_pool_run(nslots, _zts_sect_tramp, a, sizeof a[0]);
    if (p->mt == 2 && tile && N / tile >= 2)
    {   /* TILES: the tiled prefix per tile range */
        const size_t nt = N / tile;
        for (k = 0; k < K; k++)
            for (w = 0; w < per; w++)
            {
                _zts_sect_arg *x = &a[k * per + w];
                memset(x, 0, sizeof *x);
                x->kind = 3; x->p = p; x->bwd = bwd; x->tw = tw;
                x->W = k == 0 ? W : p->mtplane[k - 1];
                x->t_lo = nt * (size_t)w / (size_t)per; x->t_hi = nt * (size_t)(w + 1) / (size_t)per;
            }
        thread_pool_run(nslots, _zts_sect_tramp, a, sizeof a[0]);
    }
    for (s = 1; s < p->nf - 1; s++)
    {   /* the mids: groups (BLOCKS: every mid; TILES: the sweeps only) */
        const size_t RL = (size_t)p->L[s] * (size_t)p->chain[s];
        const size_t Gs = (size_t)p->Gs[s], pitch = 2 * RL;
        if (p->mt == 2 && tile && N / tile >= 2 && tile % RL == 0) continue;
        for (k = 0; k < K; k++)
            for (w = 0; w < per; w++)
            {
                const size_t lo = Gs * (size_t)w / (size_t)per, hi = Gs * (size_t)(w + 1) / (size_t)per;
                _zts_sect_arg *x = &a[k * per + w];
                memset(x, 0, sizeof *x);
                x->kind = 1; x->p = p; x->bwd = bwd; x->s = s;
                x->W = (k == 0 ? W : p->mtplane[k - 1]) + lo * pitch;
                x->tw = tw + p->twoff[s]; x->Ls = (size_t)p->L[s]; x->Gs = hi - lo; x->count = hi > lo ? (size_t)p->L[s] : 0;
            }
        thread_pool_run(nslots, _zts_sect_tramp, a, sizeof a[0]);
    }
    {   /* the terminator: columns of the last stage, the stream per column quad */
        const int Rl = p->chain[p->nf - 1];
        const size_t ni = L / VW;
        for (k = 0; k < K; k++)
            for (w = 0; w < per; w++)
            {
                const size_t lo = VW * (ni * (size_t)w / (size_t)per), hi = VW * (ni * (size_t)(w + 1) / (size_t)per);
                _zts_sect_arg *x = &a[k * per + w];
                memset(x, 0, sizeof *x);
                x->kind = 2; x->p = p; x->bwd = bwd;
                x->W = (k == 0 ? W : p->mtplane[k - 1]) + 2 * lo;
                x->ore = ore + k * N + lo; x->oim = oim + k * N + lo;
                x->tw = tw + p->twoff[p->nf - 1] + (lo / VW) * (size_t)(Rl - 1) * 8;
                x->Ls = L; x->OLs = L; x->count = hi - lo;
            }
        thread_pool_run(nslots, _zts_sect_tramp, a, sizeof a[0]);
    }
}

/* ── the V family: ONE lane group (K = VW lanes) over T threads ── */
static inline void _zts_sect_v(const vfft_zts_plan_t *p, int T, int bwd,
                               const double *ire, const double *iim, double *ore, double *oim, double *W, double *Wi)
{
    const double *tw = bwd ? p->twb : p->tw;
    const size_t N = (size_t)p->N, tile = p->tile, VW = (size_t)VFFT_ZTS_VW, K = (size_t)p->lanes;
    const size_t ncol = (size_t)p->ncol, L = (size_t)p->L[p->nf - 1];
    _zts_sect_arg a[THREAD_POOL_MAX_DISPATCH];
    int s, w;
    for (w = 0; w < T; w++)
    {
        const size_t lo = ncol * (size_t)w / (size_t)T, hi = ncol * (size_t)(w + 1) / (size_t)T;
        memset(&a[w], 0, sizeof a[w]);
        a[w].kind = 4; a[w].p = p; a[w].bwd = bwd;
        a[w].ire = ire + lo * K; a[w].iim = iim + lo * K; a[w].W = W; a[w].Wi = Wi;
        a[w].rb = p->rb + lo; a[w].Ls = ncol; a[w].count = hi - lo; a[w].K = K;
    }
    thread_pool_run(T, _zts_sect_tramp, a, sizeof a[0]);
    if (p->mt == 2 && tile && N / tile >= 2)
    {
        const size_t nt = N / tile;
        for (w = 0; w < T; w++)
        {
            memset(&a[w], 0, sizeof a[w]);
            a[w].kind = 7; a[w].p = p; a[w].bwd = bwd; a[w].tw = tw; a[w].W = W; a[w].Wi = Wi;
            a[w].t_lo = nt * (size_t)w / (size_t)T; a[w].t_hi = nt * (size_t)(w + 1) / (size_t)T;
        }
        thread_pool_run(T, _zts_sect_tramp, a, sizeof a[0]);
    }
    for (s = 1; s < p->nf - 1; s++)
    {
        const size_t RL = (size_t)p->L[s] * (size_t)p->chain[s];
        const size_t Gs = (size_t)p->Gs[s], pitch = VW * RL;
        if (p->mt == 2 && tile && N / tile >= 2 && tile % RL == 0) continue;
        for (w = 0; w < T; w++)
        {
            const size_t lo = Gs * (size_t)w / (size_t)T, hi = Gs * (size_t)(w + 1) / (size_t)T;
            memset(&a[w], 0, sizeof a[w]);
            a[w].kind = 5; a[w].p = p; a[w].bwd = bwd; a[w].s = s;
            a[w].W = W + lo * pitch; a[w].Wi = Wi + lo * pitch;
            a[w].tw = tw + p->twoff[s]; a[w].Ls = (size_t)p->L[s]; a[w].Gs = hi - lo; a[w].count = hi > lo ? (size_t)p->L[s] : 0;
        }
        thread_pool_run(T, _zts_sect_tramp, a, sizeof a[0]);
    }
    {
        const int Rl = p->chain[p->nf - 1];
        for (w = 0; w < T; w++)
        {
            const size_t lo = L * (size_t)w / (size_t)T, hi = L * (size_t)(w + 1) / (size_t)T;
            memset(&a[w], 0, sizeof a[w]);
            a[w].kind = 6; a[w].p = p; a[w].bwd = bwd;
            a[w].W = W + lo * VW; a[w].Wi = Wi + lo * VW;
            a[w].ore = ore + lo * K; a[w].oim = oim + lo * K;
            a[w].tw = tw + p->twoff[p->nf - 1] + lo * (size_t)(Rl - 1) * 2;   /* the pair records per column */
            a[w].Ls = L; a[w].OLs = L; a[w].count = hi - lo; a[w].K = K;
        }
        thread_pool_run(T, _zts_sect_tramp, a, sizeof a[0]);
    }
}

/* the entry: 1 = ran sectioned, 0 = the caller runs serial (no arm, no
 * stage table, a pool clamped below the bound T, or a shape the walk does
 * not cut) */
static inline int vfft_zts_execute_sect(const vfft_zts_plan_t *p, int T, int bwd, size_t K,
                                        const double *ire, const double *iim, double *ore, double *oim)
{
    if (p->mt < 1 || T < 2) return 0;
    if (p->lanes == 1)
    {
        double *W;
        if (!p->s0f || !p->slf || (bwd && (!p->s0b || !p->slb))) return 0;
        if (K < 1 || (size_t)T / K < 2 || (K > 1 && p->mt_n < (int)K - 1)) return 0;
        W = _zts_plane_for(p, ire, iim, ore, oim);
        _zts_sect_s(p, T, bwd, K, ire, iim, ore, oim, W);
        return 1;
    }
    else
    {
        double *Wr, *Wi;
        if (p->lanes != VFFT_ZTS_VW || !p->v0f || !p->vlf) return 0;
        Wr = _zts_vplane(p->plane_re, 1024u);
        Wi = _zts_vplane(p->plane_im, 3072u);
        _zts_sect_v(p, T, bwd, ire, iim, ore, oim, Wr, Wi);
        return 1;
    }
}

#endif /* VFFT_ZTS_MT_H */
