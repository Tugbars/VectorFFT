/* split_execute.h — the SPLIT execute: the split side of vfft_execute's one fork.
 *
 * One engine serves every split C2C plan: zturn-ts (split/rank1/zts.h), both
 * placements, natural order both directions, K = 1 / the contiguous batch
 * (serial, or in slabs over the pool) / the lane-major batch. 1D real plans
 * run in bridge/real_bridge_exec.h until their interleaved engine exists.
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

static void _vfft_split_execute(vfft_plan h, vfft_dir_t dir,
                                double *sre, double *sim, double *dre, double *dim)
{
    if (!h->zts || h->transform != VFFT_C2C)
    {
        _vfft_warn("vfft_execute: a split plan without its zturn-ts engine — create/execute wiring bug");
        return;
    }
    { /* zturn-ts: the whole 1D c2c execute, natural order both directions.
       * In place: the caller's planes are both ends (dre/dim NULL or aliased
       * — the signature check admitted either); the plan's lanes decide the
       * batch form: lanes = 1 runs h->K contiguous transforms (K=1 is the
       * plain driver), lanes = K the one lane-major batch. */
        double *ore = h->placement == VFFT_INPLACE ? sre : dre;
        double *oim = h->placement == VFFT_INPLACE ? sim : dim;
        const size_t K = h->zts->lanes == 1 ? h->K : 1;
        const int bwd = dir != VFFT_FORWARD;
        if (K > 1 && h->zts->mt && h->zts->mt_n > 0)
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
        if (!bwd)
            vfft_zts_execute_fwd(h->zts, sre, sim, ore, oim, K);
        else
            vfft_zts_execute_bwd(h->zts, sre, sim, ore, oim, K);
    }
}

#endif /* VFFT_SPLIT_EXECUTE_H */
