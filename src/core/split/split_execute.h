/* split_execute.h — the SPLIT execute: the split side of vfft_execute's one fork.
 *
 * SPLIT handles run here after the front door's checks: the 2D / rank-N
 * stride plans (c2c, real, natural tapes), the 1D c2c engines in place and
 * out of place (the K=1 split engine, the classic OOP path), and trig. 1D
 * real runs in bridge/real_bridge_exec.h.
 *
 * Included by vfft_execute.h under VFFT_EXECUTE_IMPL.
 */
#ifndef VFFT_SPLIT_EXECUTE_H
#define VFFT_SPLIT_EXECUTE_H

/* K=1 engine, SPLIT-plane side (natural order both directions; split bwd =
 * the pointer-swap identity on the forward route). Extracted verbatim from
 * the dispatch so the OOP INTERLEAVED convert fallback can reuse it. */
static void _exec_k1_split(struct vfft_plan_s *h, int fwd,
                           double *sre, double *sim, double *dre, double *dim)
{
    const double *ar = fwd ? sre : sim, *ai = fwd ? sim : sre;
    double *br = fwd ? dre : dim, *bi = fwd ? dim : dre;
#ifdef VFFT_USE_JIT
    if (h->k1_jit)
    { /* stride-baked whole-route kernel; bwd rides the same
       * pointer-swap identity (natural order) */
        h->k1_jit(ar, ai, br, bi, h->k1sp->col_re, h->k1sp->col_im,
                  h->k1_jit_qr, h->k1_jit_qi);
        return;
    }
#endif
    switch (h->k1_sp_route)
    {
    case VFFT_K1_SP_MONO:
        h->k1_mono(ar, ai, br, bi, 0, 0, 0, 0, 0, 0, 0);
        return;
    case VFFT_K1_SP_2PA:
        vfft_oop_execute_fwd_2pa(h->k1sp, ar, ai, br, bi);
        return;
    case VFFT_K1_SP_2PB:
        vfft_oop_execute_fwd_2pb(h->k1sp, ar, ai, br, bi);
        return;
    case VFFT_K1_SP_TWL:
        vfft_oop_execute_fwd_2pa_twl(h->k1sp, ar, ai, br, bi);
        return;
    case VFFT_K1_SP_CCOL:
        vfft_oop_execute_fwd_ccol(h->k1sp, ar, ai, br, bi);
        return;
    default:
        vfft_oop_execute_fwd(h->k1sp, ar, ai, br, bi);
        return;
    }
}

static void _vfft_split_execute(vfft_plan h, vfft_dir_t dir,
                                double *sre, double *sim, double *dre, double *dim)
{
    if (h->N2 > 0)
    { /* ── 2D (dispatch before the same-named 1D transforms) ── */
        _vfft_pool_arm(h->nthreads);
        if (h->transform == VFFT_C2C)
        {
            /* tiled-row + native-col, in-place. OOP = copy src->dst then in-place. */
            size_t plane = (size_t)h->N * h->N2 * (h->N3 ? (size_t)h->N3 : 1) * (h->N4 ? (size_t)h->N4 : 1);
            if (!dre && !dim)
            { /* validated in-place convenience: result stays in sre/sim */
                dre = sre;
                dim = sim;
            }
            if (dre != sre)
                memcpy(dre, sre, plane * sizeof(double));
            if (dim != sim)
                memcpy(dim, sim, plane * sizeof(double));
            if (dir == VFFT_FORWARD)
            {
                stride_execute_fwd(h->tplan, dre, dim);
                if (h->nat2d)
                    _natorder_2d(h, dre, dim, 0); /* scrambled -> natural (per-axis) */
            }
            else
            {
                if (h->nat2d)
                    _natorder_2d(h, dre, dim, 1); /* natural -> scrambled before the inverse FFT */
                stride_execute_bwd(h->tplan, dre, dim);
            }
        }
        else if (h->transform == VFFT_R2C && h->N3 > 0)
        { /* §6a47/Q1: 3D real fwd — rows, axes, unpack (SPLIT only: an interleaved
           * rank-3/4 real request is refused at create). */
            stride_fftnd_r2c_data_t *d3 =
                (stride_fftnd_r2c_data_t *)h->tplan->override_data;
            _fndr_execute_fwd_oop(d3, sre, dre, dim); /* the module owns
                                                       * the walk (A2) */
        }
        else if (h->transform == VFFT_C2R && h->N3 > 0)
        {
            stride_fftnd_r2c_data_t *d3 =
                (stride_fftnd_r2c_data_t *)h->tplan->override_data;
            _fndr_execute_bwd_oop(d3, sre, sim, dre); /* the module owns
                                                       * the walk (A2) */
        }
        else if (h->transform == VFFT_R2C)
        {
            if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
                /* OWNER LAW (M3, 2026-08-26): the §6a30 z-veneer no
                 * longer serves IL callers — an IL real 2D plan is
                 * native (il2d_row) or was refused at create. */
                _vfft_warn("vfft_execute: IL 2D r2c plan without the "
                           "native tier — create/execute wiring bug");
            else
                stride_execute_2d_r2c(h->tplan, sre, dre, dim); /* real plane -> split spectrum */
        }
        else if (h->transform == VFFT_C2R)
        {
            if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
                _vfft_warn("vfft_execute: IL 2D c2r plan without the "
                           "native tier — create/execute wiring bug");
            else
                stride_execute_2d_c2r(h->tplan, sre, sim, dre); /* split spectrum -> real plane */
        }
        return;
    }
    if (h->transform == VFFT_C2C && h->placement == VFFT_INPLACE)
    {
        _exec_c2c_inplace(h, dir, sre, sim);
        return;
    }
    if (h->transform == VFFT_C2C && h->placement == VFFT_OUTOFPLACE)
    {
        if (h->k1_on)
        { /* K=1 engine, SPLIT planes: natural order; bwd = pointer-swap
           * identity on the forward route. */
            _exec_k1_split(h, dir == VFFT_FORWARD, sre, sim, dre, dim);
            return;
        }
        /* MT via the pool K-split (LEAF/MODEB lane-independent; BAILEY2 + small K run
         * whole-batch — see _oop_mt). vfft_oop_execute_fwd/bwd are kind-correct (natural-
         * order swap for LEAF/BAILEY2; in-place DIF-bwd-on-copy for MODEB) and are the
         * single-thread fallback inside _oop_mt. Caller pins core 0 (workers pin 1..T-1). */
        _vfft_pool_arm(h->nthreads);
        _oop_mt(h->oplan, sre, sim, dre, dim, dir == VFFT_FORWARD ? 1 : 0);
        return;
    }
    if (_VFFT_IS_TRIG(h->transform))
    {
        /* real in (sre) -> real out (dre). Involutory kinds (DCT-I/IV, DST-I, DHT)
         * ignore `dir`; for II<->III the forward enum picks the matching member and
         * BACKWARD runs its inverse (DCT-III for a DCT-II plan, etc.). */
        _vfft_pool_arm(h->nthreads);
        /* HARNESS: the trig family's only engagement signal. A DCT/DST/DHT plan
         * sets tplan and never touches tcb, so none of the four pre-existing
         * counters can move for it and an MT==ST bitwise pass here is vacuous -
         * it passes just as happily when no thread ever ran.
         *
         * HONEST LIMIT: this counts trig executes issued with a THREADED POOL,
         * not work proven dispatched. Dispatch happens inside the
         * stride_execute_dctN entry points, below this file. Closing that last
         * gap needs a counter inside the trig executor; until then a non-zero
         * value proves the pool was armed, which is strictly more than the
         * nothing that was observable before. */
        if (h->nthreads > 1)
            _vfft_trig_mt_count++;
        const stride_plan_t *p = h->tplan;
        int f = (dir == VFFT_FORWARD);
        switch (h->transform)
        {
        case VFFT_DCT1:
            stride_execute_dct1(p, sre, dre);
            break;
        case VFFT_DCT2:
            if (f)
                stride_execute_dct2(p, sre, dre);
            else
                stride_execute_dct3(p, sre, dre);
            break;
        case VFFT_DCT3:
            if (f)
                stride_execute_dct3(p, sre, dre);
            else
                stride_execute_dct2(p, sre, dre);
            break;
        case VFFT_DCT4:
            stride_execute_dct4(p, sre, dre);
            break;
        case VFFT_DST1:
            stride_execute_dst1(p, sre, dre);
            break;
        case VFFT_DST2:
            if (f)
                stride_execute_dst2(p, sre, dre);
            else
                stride_execute_dst3(p, sre, dre);
            break;
        case VFFT_DST3:
            if (f)
                stride_execute_dst3(p, sre, dre);
            else
                stride_execute_dst2(p, sre, dre);
            break;
        case VFFT_DHT:
            stride_execute_dht(p, sre, dre);
            break;
        default:
            break;
        }
        return;
    }
}

#endif /* VFFT_SPLIT_EXECUTE_H */
