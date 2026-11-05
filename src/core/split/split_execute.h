/* split_execute.h — the SPLIT execute: the split side of vfft_execute's one fork.
 *
 * One engine serves every split C2C plan: zturn-ts (split/rank1/zts.h), both
 * placements, natural order both directions, K = 1 / the contiguous batch
 * (serial, or in slabs over the pool) / the lane-major batch. No real plan
 * exists on this side: the split library is C2C only.
 *
 * Included by vfft_execute.h under VFFT_EXECUTE_IMPL.
 */
#ifndef VFFT_SPLIT_EXECUTE_H
#define VFFT_SPLIT_EXECUTE_H

/* one pool slot of zturn-ts's threaded contiguous form (zts.h
 * vfft_zts_run_slab): a slab of whole transforms on the slot's own plane */
typedef struct
{
    const vfft_zts_plan_t *p;
    int bwd;
    const double *ire, *iim;
    double *ore, *oim;
    size_t t0, cnt;
    const double *plane;
} _zts_mt_arg;
static void _zts_mt_tramp(void *v)
{
    const _zts_mt_arg *a = (const _zts_mt_arg *)v;
    vfft_zts_run_slab(a->p, a->bwd, a->ire, a->iim, a->ore, a->oim, a->t0, a->cnt, a->plane);
}
/* one pool slot of a 3D plan's middle axis, PLANES OVER SLOTS: the
 * lane-major pass (all lanes, serial) on every plane of [p0, p1), on the
 * slot's own plane set */
typedef struct
{
    const vfft_zts_plan_t *p;
    int bwd;
    double *re, *im;
    size_t plane, p0, p1;
    double *pre, *pim;
} _zts_pl_arg;
static void _zts_pl_tramp(void *v)
{
    const _zts_pl_arg *a = (const _zts_pl_arg *)v;
    size_t p;
    for (p = a->p0; p < a->p1; p++)
        vfft_zts_run_lanes(a->p, a->bwd, a->re + p * a->plane, a->im + p * a->plane,
                           a->re + p * a->plane, a->im + p * a->plane, 0, (size_t)a->p->lanes, a->pre, a->pim);
}
/* one pool slot of the threaded LANE-MAJOR form (zts.h vfft_zts_run_lanes):
 * the lane range [g0, g1) on the slot's own plane set */
typedef struct
{
    const vfft_zts_plan_t *p;
    int bwd;
    const double *ire, *iim;
    double *ore, *oim;
    size_t g0, g1;
    double *pre, *pim;
} _zts_lm_arg;
static void _zts_lm_tramp(void *v)
{
    const _zts_lm_arg *a = (const _zts_lm_arg *)v;
    vfft_zts_run_lanes(a->p, a->bwd, a->ire, a->iim, a->ore, a->oim, a->g0, a->g1, a->pre, a->pim);
}

static void _vfft_split_execute(vfft_plan h, vfft_dir_t dir,
                                double *sre, double *sim, double *dre, double *dim)
{
    if (h->zts_row && h->transform == VFFT_C2C)
    { /* rank 2 and 3 (split/rankn/c2c_nd_create_split.h): the axes in turn,
       * last to first — the rows as the contiguous batch in the plan's
       * placement (every later pass in place on where the rows landed), a 3D
       * plan's middle axis per plane, the first axis as the lane-major pass.
       * The passes commute, so the backward runs the same order with every
       * axis backward. */
        double *ore = h->placement == VFFT_INPLACE ? sre : dre;
        double *oim = h->placement == VFFT_INPLACE ? sim : dim;
        _vfft_split_execute(h->zts_row, dir, sre, sim, dre, dim);
        if (h->zts_mid)
        {
            const size_t plane = (size_t)h->N2 * (size_t)h->N3;
            size_t p;
            int done = 0;
            if (h->zts_pmt && h->zts_mid->zts && h->zts_mid->zts->mt_n > 0)
            { /* PLANES OVER SLOTS (the 3D plan's own verdict): one dispatch, slot t
               * running the middle axis serially on its range of planes, on its
               * own plane set — the per-plane pass's dispatch cost paid once */
                int T;
                _vfft_pool_arm(h->nthreads);
                T = thread_pool_workers_for(1 + h->zts_mid->zts->mt_n);
                if (T > 1)
                {
                    const size_t NP = (size_t)h->N, S = (NP + (size_t)T - 1) / (size_t)T;
                    _zts_pl_arg a[THREAD_POOL_MAX_DISPATCH];
                    int n = 0, t;
                    for (t = 0; t < T; t++)
                    {
                        const size_t p0 = (size_t)t * S;
                        size_t p1 = p0 + S;
                        if (p0 >= NP) break;
                        if (p1 > NP) p1 = NP;
                        a[n].p = h->zts_mid->zts;
                        a[n].bwd = dir != VFFT_FORWARD;
                        a[n].re = ore;
                        a[n].im = oim;
                        a[n].plane = plane;
                        a[n].p0 = p0;
                        a[n].p1 = p1;
                        a[n].pre = t == 0 ? h->zts_mid->zts->plane_re : h->zts_mid->zts->mtplane[t - 1];
                        a[n].pim = t == 0 ? h->zts_mid->zts->plane_im : NULL;
                        n++;
                    }
                    thread_pool_run(n, _zts_pl_tramp, a, sizeof a[0]);
                    _vfft_tc_mt_dispatch_count += n - 1;
                    done = 1;
                }
            }
            if (!done)
                for (p = 0; p < (size_t)h->N; p++)
                    _vfft_split_execute(h->zts_mid, dir, ore + p * plane, oim + p * plane, ore + p * plane, oim + p * plane);
        }
        _vfft_split_execute(h->zts_col, dir, ore, oim, ore, oim);
        return;
    }
    if (h->zts_blu && h->transform == VFFT_C2C)
    { /* Bluestein over zturn-ts (zts_bluestein.h): K = 1, the contiguous batch
       * or the lane-major batch; in place when the plan is. Its threaded forms
       * (zts_blu_mt.h) under the cell's verdict: the passes cut around the
       * inner's sectioned walk, slabs, lane blocks. */
        double *ore = h->placement == VFFT_INPLACE ? sre : dre;
        double *oim = h->placement == VFFT_INPLACE ? sim : dim;
        const vfft_zts_blu_t *b = h->zts_blu;
        const size_t K = b->lanes > 1 ? 1 : h->K;
        if (b->mt)
        {
            int T;
            _vfft_pool_arm(h->nthreads);
            T = thread_pool_workers_for(b->mt_sect ? h->nthreads : 1 + b->mt_n);
            if (T > 1 && vfft_zts_blu_execute_mt(b, T, dir != VFFT_FORWARD, K, sre, sim, ore, oim))
            {
                _vfft_tc_mt_dispatch_count += T - 1;
                return;
            }
        }
        vfft_zts_blu_execute(b, dir != VFFT_FORWARD, sre, sim, ore, oim, K);
        return;
    }
    if (!h->zts || h->transform != VFFT_C2C)
    {
        _vfft_warn("vfft_execute: a split plan without its zturn-ts engine — create/execute wiring bug");
        return;
    }
    { /* zturn-ts: the whole 1D c2c execute, natural order both directions.
       * In place: the caller's planes are both ends (dre/dim NULL or aliased
       * — the signature check admitted either); the plan's lanes decide the
       * batch form: lanes = 1 runs h->K contiguous transforms (K=1 is the
       * plain driver), lanes = K the one lane-major batch. Both forms thread
       * under their own verdict (eng=tcb tcmt= on the cell's row at the
       * plan's thread count): the contiguous batch in slabs of transforms,
       * the lane-major batch in ranges of lane blocks. */
        double *ore = h->placement == VFFT_INPLACE ? sre : dre;
        double *oim = h->placement == VFFT_INPLACE ? sim : dim;
        const size_t K = h->zts->lanes == 1 ? h->K : 1;
        const int bwd = dir != VFFT_FORWARD;
        if (h->zts->mt && h->zts->lanes == 1 && (K == 1 || (K > 1 && h->zts->mt_sect)))
        { /* THE SECTIONED WALK (zts_mt.h): one transform's stages over the
           * pool, or K < T transforms sharing every stage's dispatch; the
           * verdict (mt = 1 blocks, 2 tiles) raced on the cell at T */
            int T;
            _vfft_pool_arm(h->nthreads);
            T = thread_pool_workers_for(h->nthreads);
            if (T > 1 && vfft_zts_execute_sect(h->zts, T, bwd, K, sre, sim, ore, oim))
            {
                _vfft_tc_mt_dispatch_count += T - 1;
                return;
            }
        }
        if (h->zts->mt && h->zts->lanes > 1 && h->zts->mt_sect)
        { /* the lane-major batch, sectioned: block by block, every stage over the pool */
            int T;
            _vfft_pool_arm(h->nthreads);
            T = thread_pool_workers_for(h->nthreads);
            if (T > 1 && vfft_zts_execute_sect(h->zts, T, bwd, 1, sre, sim, ore, oim))
            {
                _vfft_tc_mt_dispatch_count += T - 1;
                return;
            }
        }
        if (K > 1 && h->zts->mt && h->zts->mt_n > 0 && !h->zts->mt_sect)
        { /* THE THREADED FORM (the verdict h->zts->mt, raced at create on
           * this cell at the plan's thread count, or replayed from its
           * eng=tcb row): the K transforms in slabs of ceil(K/T) over the
           * pool's slots, slot 0 the caller on the plan's plane, slot t >= 1
           * worker t-1 on its own plane. No tail by construction: the unit
           * of work is one whole transform. T = 1 + the worker planes built
           * at create, clamped once by the pool (the live pool, the arg
           * array). */
            int T;
            _vfft_pool_arm(h->nthreads);
            T = thread_pool_workers_for(1 + h->zts->mt_n);
            if (T > 1)
            {
                const size_t S = (K + (size_t)T - 1) / (size_t)T;
                _zts_mt_arg a[THREAD_POOL_MAX_DISPATCH];
                int n = 0, t;
                for (t = 0; t < T; t++)
                {
                    const size_t t0 = (size_t)t * S;
                    size_t te = t0 + S;
                    if (t0 >= K)
                        break;
                    if (te > K)
                        te = K;
                    a[n].p = h->zts;
                    a[n].bwd = bwd;
                    a[n].ire = sre;
                    a[n].iim = sim;
                    a[n].ore = ore;
                    a[n].oim = oim;
                    a[n].t0 = t0;
                    a[n].cnt = te - t0;
                    a[n].plane = t == 0 ? h->zts->plane : h->zts->mtplane[t - 1];
                    n++;
                }
                thread_pool_run(n, _zts_mt_tramp, a, sizeof a[0]);
                _vfft_tc_mt_dispatch_count += n - 1; /* one per worker dispatched (vfft_diagnostics.h) */
                return;
            }
        }
        if (h->zts->lanes > 1 && h->zts->mt && h->zts->mt_n > 0 && !h->zts->mt_sect)
        { /* THE THREADED LANE-MAJOR FORM (the same verdict, raced on the
           * lane-major cell at the plan's thread count): the lane groups in
           * ranges of whole blocks over the pool's slots, slot 0 the caller
           * on the plan's planes, slot t >= 1 worker t-1 on its own plane
           * set; every range a multiple of the block, so each slot's blocked
           * passes are the serial form's -- bitwise the serial result. */
            int T;
            _vfft_pool_arm(h->nthreads);
            T = thread_pool_workers_for(1 + h->zts->mt_n);
            if (T > 1)
            {
                const size_t all = (size_t)h->zts->lanes;
                const size_t lanes = all - all % (size_t)VFFT_ZTS_VW;       /* the full groups; the tail runs with the last range */
                const size_t unit = h->zts->mt_unit ? h->zts->mt_unit : (size_t)h->zts->blk * (size_t)VFFT_ZTS_VW; /* lanes per range unit: a block, or one cache line (the verdict's) */
                const size_t nb = (lanes + unit - 1) / unit;                  /* units */
                const size_t S = ((nb + (size_t)T - 1) / (size_t)T) * unit;   /* lanes per slot */
                _zts_lm_arg a[THREAD_POOL_MAX_DISPATCH];
                int n = 0, t;
                for (t = 0; t < T; t++)
                {
                    const size_t g0 = (size_t)t * S;
                    size_t g1 = g0 + S;
                    if (g0 >= lanes)
                        break;
                    if (g1 >= lanes)
                        g1 = all;   /* the last range reaches the end: vfft_zts_run_lanes runs the tail there */
                    a[n].p = h->zts;
                    a[n].bwd = bwd;
                    a[n].ire = sre;
                    a[n].iim = sim;
                    a[n].ore = ore;
                    a[n].oim = oim;
                    a[n].g0 = g0;
                    a[n].g1 = g1;
                    a[n].pre = t == 0 ? h->zts->plane_re : h->zts->mtplane[t - 1];
                    a[n].pim = t == 0 ? h->zts->plane_im : NULL;
                    n++;
                }
                thread_pool_run(n, _zts_lm_tramp, a, sizeof a[0]);
                _vfft_tc_mt_dispatch_count += n - 1;
                return;
            }
        }
        if (!bwd)
            vfft_zts_execute_fwd(h->zts, sre, sim, ore, oim, K);
        else
            vfft_zts_execute_bwd(h->zts, sre, sim, ore, oim, K);
    }
}

#endif /* VFFT_SPLIT_EXECUTE_H */
