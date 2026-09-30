/* real_bridge_exec.h — the 1D r2c / c2r EXECUTE: where the layouts still meet.
 *
 * The execute twin of bridge/real_bridge.h (owner decision D1: TEMPORARY until
 * the IL real engine lands). The odd-real bridge's c2c child (either layout),
 * the zr2c composite (IL), and the split real engines behind their
 * interleaved z-doors (vfft_r2c_execute_fwd_z / vfft_c2r_disp_execute_z) or
 * their split planes.
 *
 * Included by vfft_execute.h under VFFT_EXECUTE_IMPL.
 */
#ifndef VFFT_BRIDGE_REAL_BRIDGE_EXEC_H
#define VFFT_BRIDGE_REAL_BRIDGE_EXEC_H

static void _vfft_real_bridge_execute(vfft_plan h, vfft_dir_t dir,
                                      double *sre, double *sim, double *dre, double *dim)
{
    if (h->oddr_child)
    { /* the ODD-REAL BRIDGE (struct comment at oddr_child): 1D K==1
       * odd-N real transforms through the c2c child. Both layouts —
       * the split spellings pack/unpack around the same child. */
        const size_t n = (size_t)h->N, hp1 = n / 2 + 1;
        double *b1 = h->oddr_buf, *b2 = h->oddr_buf + 2 * n;
        size_t k;
        if (h->transform == VFFT_R2C)
        {
            _il2d_row_promote(sre, b1, n);
            vfft_execute((vfft_plan)h->oddr_child, VFFT_FORWARD, b1,
                         NULL, b2, NULL);
            if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
                memcpy(dre, b2, 2 * hp1 * sizeof(double));
            else
                for (k = 0; k < hp1; k++)
                {
                    dre[k] = b2[2 * k];
                    dim[k] = b2[2 * k + 1];
                }
        }
        else
        {
            if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
                _il2d_row_extend(sre, b1, n, hp1);
            else
            {
                for (k = 0; k < hp1; k++)
                {
                    b1[2 * k] = sre[k];
                    b1[2 * k + 1] = sim[k];
                }
                for (k = 1; k < hp1; k++)
                {
                    b1[2 * (n - k)] = sre[k];
                    b1[2 * (n - k) + 1] = -sim[k];
                }
            }
            vfft_execute((vfft_plan)h->oddr_child, VFFT_BACKWARD, b1,
                         NULL, b2, NULL);
            _il2d_row_re(b2, dre, n);
        }
        return;
    }
    if (h->transform == VFFT_R2C)
    {
        /* forward only: real in (sre); spectrum out per the committed layout
         * (SPLIT dre/dim planes, or INTERLEAVED packed CCE z in dre — §6a24).
         * MT internal.
         *
         * 🔴 THE ZR2C COMPOSITE IS POOL-FREE, AND MUST STAY THAT WAY.
         * The pool re-assert below is deliberately AFTER the zr2c branch, not
         * before it. _exec_zr2c is a pure fold plus vfft_execute on the child,
         * and the child was created with c2.nthreads = cfg->nthreads, so it
         * re-asserts the identical snapshot itself -- the outer call was pure
         * duplication. Removing it is what lets a zr2c plan serve as a
         * TRANSFORM-CONTIGUOUS worker clone (_tc_inner_mt_safe): a clone runs
         * on a POOL THREAD, and vfft_set_num_threads from a worker
         * creates/destroys the very pool it is running on. Same edit in the
         * C2R branch below; keep the two in step. */
        if (h->zrm)
        {
            _exec_zrm(h, sre, dre); /* the real mono: one kernel, pool-free */
            return;
        }
        if (h->zfsr)
        {
            _exec_zfsr(h, sre, dre); /* the real four-step: the c2c four-step + the fused order sweep */
            return;
        }
        if (h->zrf)
        {
            _exec_zrf(h, sre, dre); /* the real flat DIT: serial kernels on the plan's planes, pool-free */
            return;
        }
        if (h->zttr)
        {
            _exec_zttr(h, sre, dre); /* ZTT-r: the fold fused into the terminator, pool-free */
            return;
        }
        if (h->zrp)
        {
            _exec_zrp(h, sre, dre); /* the real pair: two kernels, pool-free */
            return;
        }
        if (h->zr2c_child)
        {
            _exec_zr2c(h, sre, dre); /* §D2 composite (incl. in place) */
            return;
        }
        _vfft_pool_arm(h->nthreads);
        if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
            vfft_r2c_execute_fwd_z(h->rplan, sre, dre); /* dre = packed CCE spectrum */
        else
            vfft_r2c_execute_fwd(h->rplan, sre, dre, dim); /* split out */
        return;
    }
    if (h->transform == VFFT_C2R)
    {
        /* the inverse: spectrum in per the committed layout (SPLIT sre/sim, or
         * INTERLEAVED packed CCE z in sre — §6a24) -> real out (dre). dir
         * ignored. NATURAL or STRIDE per the bakeoff/wisdom.
         *
         * 🔴 Pool-free zr2c: the mirror of the R2C branch above -- read
         * that comment before moving either call. */
        if (h->zrm)
        {
            _exec_zrm(h, sre, dre); /* the real mono's c2r: one kernel, pool-free */
            return;
        }
        if (h->zfsr)
        {
            _exec_zfsr(h, sre, dre); /* the real four-step's c2r: the fused sweep, then the four-step backward */
            return;
        }
        if (h->zrf)
        {
            _exec_zrf(h, sre, dre); /* the real flat DIT's c2r */
            return;
        }
        if (h->zttr)
        {
            _exec_zttr(h, sre, dre); /* ZTT-r's c2r: the fold fused into the ingest, pool-free */
            return;
        }
        if (h->zrp)
        {
            _exec_zrp(h, sre, dre); /* the real pair's c2r: the mirror, pool-free */
            return;
        }
        if (h->zr2c_child)
        {
            _exec_zr2c(h, sre, dre); /* §D2 composite (incl. in place) */
            return;
        }
        _vfft_pool_arm(h->nthreads);
        if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
            vfft_c2r_disp_execute_z(h->c2rdisp, sre, dre); /* sre = packed CCE spectrum in */
        else
            vfft_c2r_disp_execute(h->c2rdisp, sre, sim, dre);
        return;
    }
}

#endif /* VFFT_BRIDGE_REAL_BRIDGE_EXEC_H */
