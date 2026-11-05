/* OUT OF THE BUILD (owner 2026-11-07): the split Bluestein is too weak to refine and release unless
 * customers ask for it. The file stays, compiled only under VFFT_SPLIT_BLUESTEIN, which no build defines;
 * the lengths it served (a prime among them) are refused at create on the split layout. */
/* zts_blu_mt.h — the THREADED FORMS of Bluestein over zturn-ts
 * (zts_bluestein.h), 2026-11-06. The owner's law for the convolutions
 * (2026-10-03): they take c2c's threading. Three forms, the zturn-ts forms
 * around the inner M cell:
 *
 *   SECTIONED   one transform (lanes = 1, K = 1), and the lane-major single
 *               group (lanes = VW): the three passes — the modulation with
 *               its zero pad (over M), the pointwise multiply (over M), the
 *               demodulation (over N) — cut into T ranges, one fork-join
 *               each, around the inner M cell's own sectioned walk
 *               (zts_mt.h: mt = 1 BLOCKS, mt = 2 TILES, the arm is the
 *               plan's). Six fork-joins plus the inner's stages.
 *   SLABS       the contiguous batch (lanes = 1, K > 1): the K transforms
 *               in slabs of ceil(K/T) over the pool's slots, slot 0 the
 *               caller on the plan's planes, slot t >= 1 worker t-1 on its
 *               own za planes and its own inner block plane. One dispatch.
 *   LANE BLOCKS the lane-major batch of two or more blocks: the lane groups
 *               in ranges of whole blocks over the slots, every slot running
 *               the whole convolution on its lane range (the za planes are
 *               shared, the lane columns disjoint; the inner's lane-major
 *               pass on [g0, g1) touches those lanes only), slot t >= 1 on
 *               its own inner plane set. One dispatch.
 *
 * Every pass is elementwise and the ranges only shift pointers and shorten
 * counts, so each element takes the serial run's arithmetic: the threaded
 * output is BITWISE the serial plan's (the inner's sectioned walk is bitwise
 * its fused driver's, zts_mt.h). The verdict (b->mt, raced at create on the
 * cell at the plan's thread count and banked eng=tcb tcmt= on its threading
 * row, c2c_create_split.h _zts_blu_tcmt_decide) names the arm; mt_sect says
 * the form. Nesting is forbidden by the pool's law: this runs on the caller
 * thread of a plan whose own nthreads > 1. Included by vfft.c only. */
#ifndef VFFT_ZTS_BLU_MT_H
#define VFFT_ZTS_BLU_MT_H

#include "split/rank1/zts_bluestein.h"
#include "split/rank1/zts_mt.h"
#include "common/support/threads.h"

/* one thread's work item */
typedef struct
{
    int kind;                    /* 0 S modulate, 1 S multiply, 2 S demodulate (an index range);
                                  * 3 V modulate, 4 V multiply, 5 V demodulate (an n range, all lanes);
                                  * 6 a SLAB of whole transforms; 7 a LANE BLOCK range (the whole convolution);
                                  * 8 the TURNED form's slab: blocks of lanes [g0, g1) on this slot's row scratch */
    const vfft_zts_blu_t *b;
    int bwd;
    const double *ire, *iim;
    double *ore, *oim;
    double *zr, *zi;             /* the za planes this slot uses */
    const double *plane;         /* the inner's plane allocation for this slot (slabs) */
    double *pre, *pim;           /* the inner's plane set for this slot (lane blocks) */
    size_t lo, hi;               /* the index / n range, or the transform range [lo, hi) */
    size_t g0, g1;               /* the lane range */
} _zts_blu_arg;

static void _zts_blu_tramp(void *v)
{
    const _zts_blu_arg *a = (const _zts_blu_arg *)v;
    const vfft_zts_blu_t *b = a->b;
    switch (a->kind)
    {
    case 0: _zts_blu_mod_s(b, a->bwd, a->ire, a->iim, a->zr, a->zi, a->lo, a->hi); break;
    case 1: _zts_blu_mul_s(b, a->bwd, a->zr, a->zi, a->lo, a->hi); break;
    case 2: _zts_blu_dem_s(b, a->bwd, a->zr, a->zi, a->ore, a->oim, a->lo, a->hi); break;
    case 3: _zts_blu_mod_v(b, a->bwd, a->ire, a->iim, a->zr, a->zi, a->lo, a->hi, 0, (size_t)b->lanes); break;
    case 4: _zts_blu_mul_v(b, a->bwd, a->zr, a->zi, a->lo, a->hi, 0, (size_t)b->lanes); break;
    case 5: _zts_blu_dem_v(b, a->bwd, a->zr, a->zi, a->ore, a->oim, a->lo, a->hi, 0, (size_t)b->lanes); break;
    case 6:
    {   /* a slab: whole transforms on this slot's za planes and inner plane */
        const size_t N = (size_t)b->N;
        size_t t;
        for (t = a->lo; t < a->hi; t++)
            _zts_blu_one_on(b, a->bwd, a->ire + t * N, a->iim + t * N, a->ore + t * N, a->oim + t * N,
                            a->zr, a->zi, a->plane);
        break;
    }
    case 8:
        _zts_blu_turned(b, a->bwd, a->ire, a->iim, a->ore, a->oim, a->g0, a->g1, a->zr, a->zi, a->plane);
        break;
    default:
    {   /* a lane block range: the whole convolution on the lanes [g0, g1) */
        const size_t M = (size_t)b->M, N = (size_t)b->N;
        _zts_blu_mod_v(b, a->bwd, a->ire, a->iim, a->zr, a->zi, 0, M, a->g0, a->g1);
        vfft_zts_run_lanes(b->inner, 0, a->zr, a->zi, a->zr, a->zi, a->g0, a->g1, a->pre, a->pim);
        _zts_blu_mul_v(b, a->bwd, a->zr, a->zi, 0, M, a->g0, a->g1);
        vfft_zts_run_lanes(b->inner, 1, a->zr, a->zi, a->zr, a->zi, a->g0, a->g1, a->pre, a->pim);
        _zts_blu_dem_v(b, a->bwd, a->zr, a->zi, a->ore, a->oim, 0, N, a->g0, a->g1);
        break;
    }
    }
}

/* one pass over [0, n) cut into T ranges */
static inline void _zts_blu_pass(int T, const _zts_blu_arg *proto, int kind, size_t n)
{
    _zts_blu_arg a[THREAD_POOL_MAX_DISPATCH];
    int w;
    for (w = 0; w < T; w++)
    {
        a[w] = *proto;
        a[w].kind = kind;
        a[w].lo = n * (size_t)w / (size_t)T;
        a[w].hi = n * (size_t)(w + 1) / (size_t)T;
    }
    thread_pool_run(T, _zts_blu_tramp, a, sizeof a[0]);
}

/* the inner M cell through its sectioned walk; the serial walk when the pool
 * is clamped (the same bits) */
static inline void _zts_blu_inner_sect(const vfft_zts_blu_t *b, int T, int bwd, double *zr, double *zi)
{
    if (!vfft_zts_execute_sect(b->inner, T, bwd, 1, zr, zi, zr, zi))
    {
        if (bwd) vfft_zts_execute_bwd(b->inner, zr, zi, zr, zi, 1);
        else     vfft_zts_execute_fwd(b->inner, zr, zi, zr, zi, 1);
    }
}

/* the entry: 1 = ran threaded, 0 = the caller runs serial (no arm, T < 2,
 * a pool clamped below the form's slot count, or a form the plan lacks).
 * K = the contiguous count (lanes = 1); in place when out == in. */
static inline int vfft_zts_blu_execute_mt(const vfft_zts_blu_t *b, int T, int bwd, size_t K,
                                          const double *ire, const double *iim, double *ore, double *oim)
{
    const size_t M = (size_t)b->M, N = (size_t)b->N;
    _zts_blu_arg a;
    if (b->mt < 1 || T < 2) return 0;
    memset(&a, 0, sizeof a);
    a.b = b; a.bwd = bwd; a.ire = ire; a.iim = iim; a.ore = ore; a.oim = oim;
    a.zr = b->za_re; a.zi = b->za_im;
    if (b->lanes == 1 && K == 1 && b->mt_sect)
    {   /* SECTIONED, one transform */
        if (!b->inner->s0f || !b->inner->slf) return 0;
        _zts_blu_pass(T, &a, 0, M);
        _zts_blu_inner_sect(b, T, 0, b->za_re, b->za_im);
        _zts_blu_pass(T, &a, 1, M);
        _zts_blu_inner_sect(b, T, 1, b->za_re, b->za_im);
        _zts_blu_pass(T, &a, 2, N);
        return 1;
    }
    if (b->lanes > 1 && b->turned)
    {   /* the TURNED form's SLABS: blocks of VFFT_ZTS_BLU_ROWS lanes over the slots, each on
         * its own row scratch and inner plane */
        const int Ts = thread_pool_workers_for(1 + b->mt_n);
        const size_t lanes = (size_t)b->lanes, RB = VFFT_ZTS_BLU_ROWS, nb = (lanes + RB - 1) / RB;
        _zts_blu_arg s[THREAD_POOL_MAX_DISPATCH];
        size_t S;
        int n = 0, t;
        if (b->mt_sect || Ts < 2 || b->mt_n < 1) return 0;
        S = ((nb + (size_t)Ts - 1) / (size_t)Ts) * RB;
        for (t = 0; t < Ts; t++)
        {
            const size_t g0 = (size_t)t * S;
            size_t g1 = g0 + S;
            if (g0 >= lanes) break;
            if (g1 > lanes) g1 = lanes;
            s[n] = a;
            s[n].kind = 8;
            s[n].g0 = g0; s[n].g1 = g1;
            s[n].zr = t == 0 ? b->za_re : b->mtza[t - 1];
            s[n].zi = t == 0 ? b->za_im : b->mtza[t - 1] + RB * M;
            s[n].plane = t == 0 ? b->inner->plane : b->inner->mtplane[t - 1];
            n++;
        }
        thread_pool_run(n, _zts_blu_tramp, s, sizeof s[0]);
        return 1;
    }
    if (b->lanes > 1 && b->mt_sect)
    {   /* SECTIONED, the lane-major batch (the inner block by block) */
        if (b->lanes % VFFT_ZTS_VW || !b->inner->v0f || !b->inner->vlf) return 0;
        _zts_blu_pass(T, &a, 3, M);
        _zts_blu_inner_sect(b, T, 0, b->za_re, b->za_im);
        _zts_blu_pass(T, &a, 4, M);
        _zts_blu_inner_sect(b, T, 1, b->za_re, b->za_im);
        _zts_blu_pass(T, &a, 5, N);
        return 1;
    }
    if (b->lanes == 1 && K > 1 && !b->mt_sect)
    {   /* SLABS of whole transforms */
        const int Ts = thread_pool_workers_for(1 + b->mt_n);
        _zts_blu_arg s[THREAD_POOL_MAX_DISPATCH];
        size_t S;
        int n = 0, t;
        if (Ts < 2 || b->mt_n < 1) return 0;
        S = (K + (size_t)Ts - 1) / (size_t)Ts;
        for (t = 0; t < Ts; t++)
        {
            const size_t t0 = (size_t)t * S;
            size_t te = t0 + S;
            if (t0 >= K) break;
            if (te > K) te = K;
            s[n] = a;
            s[n].kind = 6;
            s[n].lo = t0; s[n].hi = te;
            s[n].zr = t == 0 ? b->za_re : b->mtza[t - 1];
            s[n].zi = t == 0 ? b->za_im : b->mtza[t - 1] + M;
            s[n].plane = t == 0 ? b->inner->plane : b->inner->mtplane[t - 1];
            n++;
        }
        thread_pool_run(n, _zts_blu_tramp, s, sizeof s[0]);
        return 1;
    }
    if (b->lanes > 1 && !b->mt_sect)
    {   /* LANE BLOCKS: ranges of whole blocks of the inner's lane groups */
        const int Ts = thread_pool_workers_for(1 + b->mt_n);
        const size_t all = (size_t)b->lanes;
        const size_t lanes = all - all % (size_t)VFFT_ZTS_VW;   /* the full groups; the tail runs with the last range */
        const size_t unit = b->mt_unit ? b->mt_unit : (size_t)b->inner->blk * (size_t)VFFT_ZTS_VW; /* a block, or one cache line of lanes */
        const size_t nb = (lanes + unit - 1) / unit;
        _zts_blu_arg s[THREAD_POOL_MAX_DISPATCH];
        size_t S;
        int n = 0, t;
        if (Ts < 2 || b->mt_n < 1) return 0;
        S = ((nb + (size_t)Ts - 1) / (size_t)Ts) * unit;
        for (t = 0; t < Ts; t++)
        {
            const size_t g0 = (size_t)t * S;
            size_t g1 = g0 + S;
            if (g0 >= lanes) break;
            if (g1 >= lanes) g1 = all;
            s[n] = a;
            s[n].kind = 7;
            s[n].g0 = g0; s[n].g1 = g1;
            s[n].pre = t == 0 ? b->inner->plane_re : b->inner->mtplane[t - 1];
            s[n].pim = t == 0 ? b->inner->plane_im : NULL;
            n++;
        }
        thread_pool_run(n, _zts_blu_tramp, s, sizeof s[0]);
        return 1;
    }
    return 0;
}

#endif /* VFFT_ZTS_BLU_MT_H */
