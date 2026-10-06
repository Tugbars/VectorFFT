/* il_execute.h — the INTERLEAVED execute: the IL side of vfft_execute's one fork.
 *
 * vfft_execute validates the call (_vfft_sig_bad), tries the bound K=1 IL
 * dispatch (k1_exec), routes 1D real to bridge/real_bridge_exec.h (the D1
 * crossing), and then forks ONCE on the committed layout. INTERLEAVED handles
 * run here: the 2D plane queue, the transform-contiguous batch, the native
 * IL 2D / rank-N tiers, and the IL c2c engines in and out of place. Also the
 * bound K=1 fast path's arms (_k1x_*) and its binder (_vfft_k1_bind_exec,
 * called at the c2c create exits).
 *
 * Included by vfft_execute.h under VFFT_EXECUTE_IMPL.
 */
#ifndef VFFT_IL_EXECUTE_H
#define VFFT_IL_EXECUTE_H

/* TRANSFORM-CONTIGUOUS batch MT: worker t runs transforms [t0, t0+tc) of
 * the batch on its OWN clone handle (tcbw comment on the struct) — full
 * independence, no barriers, disjoint blocks. The clone's route is pool-free
 * by _tc_inner_mt_safe, so this re-entry from a pool thread can never touch
 * the pool. */
/* One transform of a transform-contiguous batch: the inner handle's bound
 * executor itself. The batch call paid the door's checks once; paying them
 * again per transform through the public execute cost 1 ns a transform at
 * N = 16 (10% of the transform) and with the bridge's engine walk 2-3 ns at
 * 32..1024 (measured 2026-10-01). An unbound or declining inner takes the
 * public execute. */
static inline void _tc_one(struct vfft_plan_s *in, vfft_dir_t dir, double *s, double *d)
{
    if (!in->k1_exec || in->k1_exec(in, dir, s, d) != 0)
        vfft_execute(in, dir, s, NULL, d, NULL);
}
typedef struct
{
    struct vfft_plan_s *p;
    vfft_dir_t dir;
    double *s, *d;
    size_t t0, tc, sn, dn; /* sn/dn: source and destination block strides --
                            * EQUAL for C2C, DIFFERENT for r2c/c2r (see
                            * h->tcb_sn/tcb_dn at create) */
} _tc_mt_arg;
static void _tc_mt_tramp(void *v)
{
    _tc_mt_arg *a = (_tc_mt_arg *)v;
    for (size_t t = 0; t < a->tc; t++)
        _tc_one(a->p, a->dir, a->s + (a->t0 + t) * a->sn, a->d + (a->t0 + t) * a->dn);
}

/* the flat DIT's serving: the threaded verdict at the plan's T when it
 * engages (il_flatdit_mt.h), the bound serial lists otherwise. Both
 * directions, both order classes, z -> z legal in both. */
static void _ilfd_serve(struct vfft_plan_s *h, vfft_dir_t dir,
                        const double *zin, double *zout)
{
    const vfft_ilfd_plan_t *p = h->k1ilfd;
    if (p->mt > 0 && h->nthreads > 1)
    {
        _vfft_pool_arm(h->nthreads); /* re-assert the snapshot pool */
        if (vfft_ilfd_execute_mt(p, zin, zout, dir == VFFT_BACKWARD))
            return;
    }
    if (dir == VFFT_FORWARD)
        vfft_ilfd_execute_fwd(p, zin, zout);
    else
        vfft_ilfd_execute_bwd(p, zin, zout);
}

/* ── THE BOUND K=1 IL DISPATCH (2026-09-07) ──────────────────────────────
 * One entry per engine, the exact call the general dispatch below makes
 * for that engine (both placements: zout = dre ? dre : sre, validated by
 * _vfft_sig_bad). Bound once at create by _vfft_k1_bind_exec; the general
 * dispatch stays as the path for everything else and for a trampoline that
 * returns nonzero. */
static int _k1x_mono(struct vfft_plan_s *h, vfft_dir_t dir, const double *zin, double *zout)
{   /* ONE LEG: Ls = OLs = 1, count = 1 (the solo kernels index zin[2*(r*Ls + k)]) */
    (dir == VFFT_FORWARD ? h->k1_mono_ilf : h->k1_mono_ilb)(zin, 0, zout, 0, 0, 0, 1, 0, 1, 0, 1);
    return 0;
}
static int _k1x_il2p(struct vfft_plan_s *h, vfft_dir_t dir, const double *zin, double *zout)
{
    if (dir == VFFT_FORWARD) { vfft_il2p_execute_fwd(h->k1il2p, zin, zout); return 0; }
    return vfft_il2p_execute_bwd(h->k1il2p, zin, zout);   /* nonzero: the general path decides */
}
static int _k1x_il3p(struct vfft_plan_s *h, vfft_dir_t dir, const double *zin, double *zout)
{
    if (dir == VFFT_FORWARD) vfft_il3p_execute_fwd(h->k1il3p, zin, zout);
    else vfft_il3p_execute_bwd(h->k1il3p, zin, zout);
    return 0;
}
static int _k1x_ilfd(struct vfft_plan_s *h, vfft_dir_t dir, const double *zin, double *zout)
{
    _ilfd_serve(h, dir, zin, zout);
    return 0;
}
/* ZTURN-T's serving: the threaded arm at the plan's T when it engages
 * (ztt_mt.h: the staged walk sectioned), else the serial form — the fused
 * codelet at a pow2 cell, the staged walk at an odd cell. Both directions,
 * both order classes, both placements. */
static void _ztt_serve(struct vfft_plan_s *h, vfft_dir_t dir, const double *zin, double *zout)
{
    const vfft_ztt_plan_t *p = h->k1ztt;
    if (p->mt > 0 && h->nthreads > 1)
    {
        _vfft_pool_arm(h->nthreads); /* re-assert the snapshot pool */
        if (vfft_ztt_execute_mt(p, zin, zout, dir == VFFT_BACKWARD))
            return;
    }
    if (dir == VFFT_FORWARD) vfft_ztt_execute_fwd(p, zin, zout);
    else vfft_ztt_execute_bwd(p, zin, zout);
}
static int _k1x_ztt(struct vfft_plan_s *h, vfft_dir_t dir, const double *zin, double *zout)
{
    _ztt_serve(h, dir, zin, zout);
    return 0;
}
static int _k1x_fs(struct vfft_plan_s *h, vfft_dir_t dir, const double *zin, double *zout)
{
    if (h->nthreads > 1)
        _vfft_pool_arm(h->nthreads); /* the 2D child's threaded verdict runs on the snapshot pool */
    vfft_k1fs_execute(h->k1fs, dir, zin, zout);
    return 0;
}
/* the prime cell: its threaded form when the race bound one at this T
 * (il_prime_mt.h: the inner's walk + the passes cut), else the serial run.
 * Both directions, both placements (the convolution is alias-safe). */
static void _ilpr_serve(struct vfft_plan_s *h, vfft_dir_t dir, const double *zin, double *zout)
{
    const vfft_ilprime_plan_t *p = h->k1ilpr;
    if (p->mt > 0 && h->nthreads > 1)
    {
        _vfft_pool_arm(h->nthreads); /* re-assert the snapshot pool */
        if (vfft_ilprime_execute_mt(p, zin, zout, dir == VFFT_BACKWARD))
            return;
    }
    if (dir == VFFT_FORWARD) vfft_ilprime_execute_fwd(p, zin, zout);
    else vfft_ilprime_execute_bwd(p, zin, zout);
}
static int _k1x_ilpr(struct vfft_plan_s *h, vfft_dir_t dir, const double *zin, double *zout)
{
    _ilpr_serve(h, dir, zin, zout);
    return 0;
}
/* Bind at the c2c create exits. The conditions are exactly those under
 * which the general dispatch reaches the K=1 IL engines: 1D, K == 1,
 * INTERLEAVED, no wrapper (tcb / plane queue / odd-real bridge / rank-N),
 * not the cascade (its dispatcher carries the MT arm). Out of place the
 * route is the committed k1_il_route (route truthfulness at create); in
 * place the engine pointer, in the dispatch's own order. NULL plans pass. */
/* THE BOUND 1D REAL DISPATCH (2026-09-30): a 1D, K == 1, INTERLEAVED real
 * plan without a wrapper (tcb / plane queue / rank-N) -- the odd-real bridge
 * included -- executes through the real bridge in one indirect call, the
 * same fast path the c2c plans take; the general signature walk it skips
 * cost 4-6 ns per call, half of a 3-point r2c (measured 2026-09-30). */
static void _vfft_real_bridge_execute(vfft_plan h, vfft_dir_t dir, double *sre, double *sim, double *dre, double *dim); /* bridge/real_bridge_exec.h, later in this TU */
static int _k1x_real(struct vfft_plan_s *h, vfft_dir_t dir, const double *zin, double *zout)
{
    _vfft_real_bridge_execute(h, dir, (double *)zin, NULL, zout, NULL);
    return 0;
}
/* the real mono (il/real/zrm.h): the kernel call itself, nothing between the
 * public execute and the transform */
static int _k1x_zrm(struct vfft_plan_s *h, vfft_dir_t dir, const double *zin, double *zout)
{
    (void)dir; /* the kernel is the plan's direction */
    vfft_zrm_execute(h->zrm, zin, zout);
    return 0;
}
/* the real flat DIT (il/real/zrf.h): the engine's execute, bound by the
 * plan's direction */
static int _k1x_zrf_fwd(struct vfft_plan_s *h, vfft_dir_t dir, const double *zin, double *zout)
{
    (void)dir;
    vfft_zrf_execute_fwd(h->zrf, zin, zout);
    return 0;
}
static int _k1x_zrf_bwd(struct vfft_plan_s *h, vfft_dir_t dir, const double *zin, double *zout)
{
    (void)dir;
    vfft_zrf_execute_bwd(h->zrf, zin, zout);
    return 0;
}
/* the real Bluestein (il/real/zrb.h): the engine's execute, bound by the
 * plan's direction */
static int _k1x_zrb_fwd(struct vfft_plan_s *h, vfft_dir_t dir, const double *zin, double *zout)
{
    (void)dir;
    vfft_zrb_execute_fwd(h->zrb, zin, zout);
    return 0;
}
static int _k1x_zrb_bwd(struct vfft_plan_s *h, vfft_dir_t dir, const double *zin, double *zout)
{
    (void)dir;
    vfft_zrb_execute_bwd(h->zrb, zin, zout);
    return 0;
}
/* the real four-step, ZTT-r, the real pair and the zr2c composite
 * (il/real/zrp_build.h, zr2c_build.h): the engine's execute itself, the call
 * the bridge's engine walk ends in. Each arms the pool itself when its
 * threaded form is bound and takes either placement. */
static int _k1x_zfsr(struct vfft_plan_s *h, vfft_dir_t dir, const double *zin, double *zout)
{
    (void)dir;
    _exec_zfsr(h, zin, zout);
    return 0;
}
static int _k1x_zttr(struct vfft_plan_s *h, vfft_dir_t dir, const double *zin, double *zout)
{
    (void)dir;
    _exec_zttr(h, zin, zout);
    return 0;
}
static int _k1x_zrp(struct vfft_plan_s *h, vfft_dir_t dir, const double *zin, double *zout)
{
    (void)dir;
    _exec_zrp(h, zin, zout);
    return 0;
}
static int _k1x_zr2c(struct vfft_plan_s *h, vfft_dir_t dir, const double *zin, double *zout)
{
    (void)dir;
    _exec_zr2c(h, zin, zout);
    return 0;
}
static vfft_plan _vfft_real_bind_exec(vfft_plan hp)
{
    struct vfft_plan_s *h = (struct vfft_plan_s *)hp;
    if (!h) return hp;
    h->k1_exec = NULL;
    if (h->transform != VFFT_R2C && h->transform != VFFT_C2R) return hp;
    if (h->layout != (int)VFFT_LAYOUT_INTERLEAVED) return hp;
    if (h->zrm)
    {
        h->k1_exec = _k1x_zrm;
        return hp;
    }
    if (h->zrb && h->K == 1 && !h->zrb->mt)
    {   /* one row, serial: the engine's execute itself (a lane-major batch, and a
         * threaded form, go through the bridge, which arms the pool) */
        h->k1_exec = h->transform == VFFT_R2C ? _k1x_zrb_fwd : _k1x_zrb_bwd;
        return hp;
    }
    if (h->zrf && !h->zrf->mt)
    {   /* serial: the engine's execute itself (a threaded form goes through the bridge, which arms the pool) */
        h->k1_exec = h->transform == VFFT_R2C ? _k1x_zrf_fwd : _k1x_zrf_bwd;
        return hp;
    }
    if (!h->pq_inner && !h->tcb && h->N2 == 0 && h->K == 1)
    {   /* the bridge's own order */
        if (h->zfsr) { h->k1_exec = _k1x_zfsr; return hp; }
        if (h->zttr) { h->k1_exec = _k1x_zttr; return hp; }
        if (h->zrp) { h->k1_exec = _k1x_zrp; return hp; }
        if (h->zr2c_kid) { h->k1_exec = _k1x_zr2c; return hp; }
    }
    if (!h->pq_inner && !h->tcb && h->N2 == 0)
        h->k1_exec = _k1x_real;
    return hp;
}

/* THE BOUND 2D REAL DISPATCH (2026-10-02): a serial interleaved 2D r2c / c2r
 * plan -- one plane, one thread, the column pass not threaded -- executes its
 * two passes straight from the door's fast path: no signature walk, no
 * transform fork, no pool re-assert (nothing under a serial plan dispatches).
 * The passes are the ones the general path runs, in its order. Fixed cost was
 * the whole story on tiny planes (2x4: 16 ns against the comparator's 9,
 * measured 2026-10-01). An aliased call (the plan is out of place) declines
 * to the general path, which says why. */
static int _k2x_il2d_r2c(struct vfft_plan_s *h, vfft_dir_t dir, const double *zin, double *zout)
{
    (void)dir; /* r2c is the forward math, as in 1D */
    if (zin == (const double *)zout)
        return 1;
    _il2d_real_rows_fwd(h, zin, zout);
    _il2d_real_cols(h, zout, zout, /*reverse=*/0);
    return 0;
}
/* TWO KERNELS: the rows kernel, then the column plan's leaf (a one-stage
 * chain's kernel, or the N1-point blocked column leaf) over the hp1 columns,
 * both stack-insensitive by their races (rxs = cxs = any). The calls are the
 * ones _il2d_rowx_body and _il2d_colx_body make for such a plan, argument for
 * argument, without the layers between. */
static int _k2x_il2d_r2c_2k(struct vfft_plan_s *h, vfft_dir_t dir, const double *zin, double *zout)
{
    const size_t rn2 = (size_t)h->N2, hp1 = rn2 / 2 + 1;
    (void)dir;
    if (zin == (const double *)zout)
        return 1;
    h->il2d_rx_lm(zin, NULL, zout, NULL, NULL, NULL, rn2, 0, hp1, 0, (size_t)h->N);
    h->il2d_cx_leaf(zout, NULL, zout, NULL, NULL, NULL, hp1, 0, hp1, 0, hp1);
    return 0;
}
static int _k2x_il2d_c2r(struct vfft_plan_s *h, vfft_dir_t dir, const double *zin, double *zout)
{
    (void)dir; /* c2r is the inverse math, unnormalized */
    if (zin == (const double *)zout)
        return 1;
    _il2d_real_cols(h, zin, h->il2d_rscr, /*reverse=*/1);
    _il2d_real_rows_bwd(h, h->il2d_rscr, zout);
    return 0;
}
/* the c2r twin of the two-kernel form: the column plan's backward leaf out of
 * place into the column-inverse plane, then the backward rows kernel; both
 * stack-insensitive by their races (rxs_c2r = cxs_c2r = any). The calls are
 * the ones _il2d_colx_body_bwd and _il2d_rowx_body_bwd make. */
static int _k2x_il2d_c2r_2k(struct vfft_plan_s *h, vfft_dir_t dir, const double *zin, double *zout)
{
    const size_t rn2 = (size_t)h->N2, hp1 = rn2 / 2 + 1;
    (void)dir;
    if (zin == (const double *)zout)
        return 1;
    h->il2d_cx_leaf(zin, NULL, h->il2d_rscr, NULL, NULL, NULL, hp1, 0, hp1, 0, hp1);
    h->il2d_rx_lm(h->il2d_rscr, NULL, zout, NULL, NULL, NULL, hp1, 0, rn2, 0, (size_t)h->N);
    return 0;
}
/* THE DESTROYING C2R (the request's destroy_input permission, where the
 * destroying column plan won its race): the reverse column pass as one
 * kernel IN PLACE on the caller's plane, the row pass from it; the
 * column-inverse plane is not touched. The caller's z holds the
 * column-inverse plane afterwards. */
static int _k2x_il2d_c2r_d(struct vfft_plan_s *h, vfft_dir_t dir, const double *zin, double *zout)
{
    (void)dir;
    if (zin == (const double *)zout)
        return 1;
    _il2d_cxd_cols(h, (double *)zin);
    _il2d_real_rows_bwd(h, zin, zout);
    return 0;
}
/* its two-kernel form: the in-place column kernel, then the backward rows
 * kernel, both stack-insensitive by their races */
static int _k2x_il2d_c2r_d2k(struct vfft_plan_s *h, vfft_dir_t dir, const double *zin, double *zout)
{
    const size_t rn2 = (size_t)h->N2, hp1 = rn2 / 2 + 1;
    (void)dir;
    if (zin == (const double *)zout)
        return 1;
    h->il2d_cxd_leaf(zin, NULL, (double *)zin, NULL, NULL, NULL, hp1, 0, hp1, 0, hp1);
    h->il2d_rx_lm(zin, NULL, zout, NULL, NULL, NULL, hp1, 0, rn2, 0, (size_t)h->N);
    return 0;
}
static vfft_plan _vfft_k1_bind_exec(vfft_plan hp)
{
    struct vfft_plan_s *h = (struct vfft_plan_s *)hp;
    if (!h) return hp;
    h->k1_exec = NULL;
    if ((h->transform == VFFT_R2C || h->transform == VFFT_C2R) &&
        h->layout == (int)VFFT_LAYOUT_INTERLEAVED && h->N2 > 0 && h->il2d_row &&
        !h->pq_inner && !h->tcb && !h->ilnd &&
        h->nthreads <= 1 && !h->il2d_col.colmt)
    {
        h->k1_exec = h->transform == VFFT_R2C ? _k2x_il2d_r2c : _k2x_il2d_c2r;
        if (h->transform == VFFT_R2C && h->il2d_rx_on && h->il2d_rx_lm && h->il2d_rx_stk < 0 &&
            h->il2d_cx_on && h->il2d_cx_leaf && !h->il2d_cx_perk && h->il2d_cx_stk < 0)
            h->k1_exec = _k2x_il2d_r2c_2k;
        if (h->transform == VFFT_C2R && h->il2d_rx_on && h->il2d_rx_lm && h->il2d_rx_stk < 0 &&
            h->il2d_cx_on && h->il2d_cx_leaf && !h->il2d_cx_perk && h->il2d_cx_stk < 0)
            h->k1_exec = _k2x_il2d_c2r_2k;
        if (h->transform == VFFT_C2R && h->il2d_cxd_on && h->il2d_cxd_leaf)
            h->k1_exec = (h->il2d_rx_on && h->il2d_rx_lm && h->il2d_rx_stk < 0 && h->il2d_cxd_stk < 0)
                             ? _k2x_il2d_c2r_d2k : _k2x_il2d_c2r_d;
        return hp;
    }
    if (h->transform != VFFT_C2C || h->layout != (int)VFFT_LAYOUT_INTERLEAVED) return hp;
    if (h->K != 1 || h->N2 > 0 || h->tcb || h->pq_inner || h->ilnd) return hp;
    if (h->placement == VFFT_OUTOFPLACE)
    {
        if (!h->k1_on) return hp;
        switch (h->k1_il_route)
        {
        case VFFT_K1_IL_MONO:    if (h->k1_mono_ilf && h->k1_mono_ilb) h->k1_exec = _k1x_mono; break;
        case VFFT_K1_IL_2P_PURE: if (h->k1il2p) h->k1_exec = _k1x_il2p; break;
        case VFFT_K1_IL_CHAIN3:  if (h->k1il3p) h->k1_exec = _k1x_il3p; break;
        case VFFT_K1_IL_FLAT:    if (h->k1ilfd) h->k1_exec = _k1x_ilfd; break;
        case VFFT_K1_IL_ZTT:     if (h->k1ztt)  h->k1_exec = _k1x_ztt;  break;
        case VFFT_K1_IL_FS:      if (h->k1fs)   h->k1_exec = _k1x_fs;   break;
        case VFFT_K1_IL_PRIME:   if (h->k1ilpr) h->k1_exec = _k1x_ilpr; break;
        default: break;
        }
        return hp;
    }
    if (h->placement != VFFT_INPLACE) return hp;
    if (h->k1_mono_ilf && h->k1_mono_ilb) h->k1_exec = _k1x_mono;
    else if (h->k1il2p) h->k1_exec = _k1x_il2p;
    else if (h->k1il3p) h->k1_exec = _k1x_il3p;
    else if (h->k1ilfd) h->k1_exec = _k1x_ilfd;
    else if (h->k1ztt) h->k1_exec = _k1x_ztt;
    else if (h->k1fs) h->k1_exec = _k1x_fs;
    else if (h->k1ilpr) h->k1_exec = _k1x_ilpr;
    return hp;
}

static void _vfft_il_execute(vfft_plan h, vfft_dir_t dir,
                             double *sre, double *sim, double *dre, double *dim)
{
    (void)sim;
    (void)dim;
    if (h->pq_inner)
    { /* 2D PLANE QUEUE (howmany > 1): loop or atomic-counter queue per
       * the raced verdict — see _pq_execute. */
        _pq_execute(h, dir, sre, dre);
        return;
    }
    if (h->tcb)
        { /* TRANSFORM-CONTIGUOUS batch: K independent K=1 transforms. Block strides are h->tcb_sn/tcb_dn
     * (equal for C2C, different for r2c/c2r); the inner handle carries route, placement and order.
     * See docs/design/vfft_front_door.md. */
        double *d = dre;
        const size_t sn = h->tcb_sn, dn = h->tcb_dn;
        int T = 1 + h->tcbw_n;
        /* The THREADING verdict (h->tc_mt): raced serial-vs-slabs at create
         * on this cell, or replayed from its eng=tcb row; T-free (one
         * transform per core). No verdict => the serial loop. */
        if (T > 1 && h->tc_mt)
        {
            _vfft_pool_arm(h->nthreads); /* re-assert snapshot pool */
            /* T = 1 + clones built at create (the plan's own snapshot); the
             * pool's one clamp also bounds it by the live pool and the
             * arg-array size, and never above the clone count. */
            T = thread_pool_workers_for(T);
        }
        else
            T = 1;
        if (T > 1)
        {
            /* 🔴 NO TAIL, BY CONSTRUCTION — and note the contrast with the
             * lane-major arm right below (_il_mt_arg), whose slab size is
             * `(ceil(K/T) + 7) & ~7`: there a slab is a set of SIMD LANES,
             * so it must stay a whole multiple of the vector width and the
             * leftover lanes need padded/SSE2 tail machinery. Here the unit
             * of work is ONE WHOLE K=1 TRANSFORM, so ceil(K/T) needs no
             * rounding at all: a ragged K just gives the last worker fewer
             * complete transforms, each running the identical kernel. This
             * is the "loop the K=1 solution for any K" contract — no `me`,
             * no partial-lane count, no padding, nothing to get wrong.
             * Gated at K=43 over 8 threads (slabs 6,6,6,6,6,6,6,1).
             *
             * Slot 0 is the caller on the PRIMARY plan h->tcb (on its serial
             * twin h->tcb0 when the primary runs a threaded form); slot t>=1
             * is worker t-1 on its own clone h->tcbw[t-1] (a clone per worker
             * is what makes the pool-free inner route safe to run
             * concurrently). The pool's fork-join dispatches exactly that. */
            const size_t S = (h->K + (size_t)T - 1) / (size_t)T;
            _tc_mt_arg a[THREAD_POOL_MAX_DISPATCH];
            int n = 0;
            for (int t = 0; t < T; t++)
            {
                size_t t0 = (size_t)t * S;
                if (t0 >= h->K)
                    break;
                size_t te = t0 + S;
                if (te > h->K)
                    te = h->K;
                a[n++] = (_tc_mt_arg){t == 0 ? (h->tcb0 ? h->tcb0 : h->tcb) : h->tcbw[t - 1], dir, sre, d,
                                      t0, te - t0, sn, dn};
            }
            thread_pool_run(n, _tc_mt_tramp, a, sizeof a[0]);
            _vfft_tc_mt_dispatch_count += n - 1; /* one per worker dispatched, see vfft.h */
            return;
        }
        for (size_t t = 0; t < h->K; t++)
            _tc_one(h->tcb, dir, sre + t * sn, d + t * dn);
        return;
    }
    if (h->N2 > 0)
    { /* ── 2D (dispatch before the same-named 1D transforms) ── */
        _vfft_pool_arm(h->nthreads);
        if (h->transform == VFFT_C2C)
        {
            /* tiled-row + native-col, in-place. OOP = copy src->dst then in-place. */
            if (h->ilnd)
            {   /* the rank-N INTERLEAVED c2c tier (fftnd_il.h): axis 0
                 * src -> dst, then the raced per-plane arm on dst */
                if (!dre)
                    dre = sre;
                vfft_ilnd_execute(h->ilnd, dir, sre, dre);
                return;
            }
            if (h->il2d_row)
            {
                /* ── native IL 2D tier (M1/M2): the column chain —
                 * t2c stages then the n1c leaf, block-looped, same
                 * slots (simulator-proven maps, il2d_proto.h) — then
                 * per-row K=1 IL children. The column and row passes
                 * COMMUTE (no inter-pass twiddle) so BWD runs the same
                 * stage order with the bwd pair + conjugated tables.
                 * OOP: stage 0 performs the src->dst move (the kinds
                 * are alias-tolerant both ways). nst == 1 leaves i in
                 * natural order; nst > 1 leaves i digit-reversed by
                 * the chain (the scrambled contract). */
                const int fwd = (dir == VFFT_FORWARD);
                size_t i, rn = (size_t)h->N2;
                const size_t wc = (h->il2d_col.wc > 0)
                                      ? (size_t)h->il2d_col.wc
                                      : rn;
                if (!dre)
                    dre = sre; /* in-place convenience */
                /* INC-C: the raced MT walk of the banked route -- the
                 * chain's bands/strips/block (self-contained units because
                 * rows commute: the same fact that legalizes tfuse), the
                 * turn's and the skewed pass's row slabs (2026-09-24).
                 * Declines back to the serial walk below when it cannot
                 * engage. */
                if (h->il2d_col.colmt && h->nthreads > 1 &&
                    _il2d_c2c_mt(h, sre, dre, dir, h->nthreads))
                    return;
                if (h->il2d_turn)
                {   /* the TURN route (2026-09-23): the whole plane through
                     * the 1D engine -- rows turned into the N2 x N1
                     * scratch, the columns as its rows, one back-turn */
                    _il2d_turn_exec(h, dir, sre, dre);
                    return;
                }
                if (h->il2d_csk)
                {   /* the SKEWED column pass (2026-09-23): the single column
                     * stage into the skewed scratch, the rows from it into
                     * the plane */
                    _il2d_csk_exec(h, dir, sre, dre);
                    return;
                }
                if (h->il2d_col.blu && h->il2d_col.tpc)
                {   /* PRIME N1: the TURNED pass (2026-09-24), then the rows */
                    _il2d_tpc_cols_range(&h->il2d_col, sre, dre, rn, 0, rn, !fwd);
                    _il2d_rows_exec(h, 0, dir, dre, rn, rn, 0, 1, (size_t)h->N);
                    return;
                }
                if (h->il2d_col.blu)
                { /* ODD/PRIME N1: the column-axis Bluestein — the
                   * shared pipeline (_il2d_blu_cols), then the rows
                   * (commute). n1 NATURAL on this route. */
                    _il2d_blu_cols(sre, dre, h->N, rn, h->il2d_col.blu,
                                   h->il2d_col.nst, h->il2d_col.R, h->il2d_col.L,
                                   h->il2d_col.f, h->il2d_col.b, h->il2d_col.tf,
                                   h->il2d_col.tb,
                                   fwd ? h->il2d_col.bluchf
                                       : h->il2d_col.bluchb,
                                   fwd ? h->il2d_col.blukf
                                       : h->il2d_col.blukb,
                                   h->il2d_col.bluscr);
                    _il2d_rows_exec(h, 0, dir, dre, rn, rn, 0, 1, (size_t)h->N);
                    return;
                }
                /* strip loop-interchange: all stages depth-first per
                 * column strip — the strip stays cache-resident
                 * across stages (one DRAM sweep, not nst). Legal
                 * because columns are independent within the column
                 * pass; Gs stays the FULL row pitch. wc = rn is the
                 * untiled M2 walk, path-identical. */
                if (h->il2d_col.nat && h->il2d_col.wl > 0)
                {
                    /* ── NATURAL BANDED walk (2026-09-05): the natural
                     * pass with the scrambled walk's cache banding.
                     * fwd = wide prefix stages 0..cut-1 sre -> natscr
                     * (stage 0 is the OOP move, the rest in place),
                     * then per band of wl rows: the suffix stages
                     * cut..nst-2 in place on the band, the band's leaf
                     * blocks SCATTERED to their natural rows of dre
                     * (perm is block-affine: block b's R rows land at
                     * perm[bR] + r*(N1/R)), and (tfuse) the row pass
                     * of exactly those rows while hot. Same kernel
                     * calls, same tables, same count as the unbanded
                     * natural pass — only the traversal order across
                     * independent blocks differs (bitwise-identical;
                     * the row pass sees the same leaf output either
                     * way, so the MT partitions stay bitwise too).
                     * bwd, staged (2026-10-06, _il2d_nat_bwd_fused): THE
                     * FORWARD'S WALK with the forward kernels and the
                     * backward row child fused after the leaf, the leaf
                     * blocks scattered to the rows (N1 - i) mod N1 (the
                     * inverse DFT along the columns is the forward DFT
                     * with its output index negated): the same memory
                     * pattern as the forward; the MT block and tile arms
                     * run the same shape, so bwd stays bitwise with them.
                     * bwd otherwise (the strided leaf, a strips verdict,
                     * the four-step child) = the mirrored walk: per band
                     * the leaf GATHERS its blocks from sre's natural rows
                     * into the scratch comb, the suffix runs REVERSED on
                     * the band (into dre when nothing is left wide), then
                     * the reversed prefix wide scratch -> dre, the rows
                     * LAST on dre (unfused, bitwise with the strips arm).
                     * The width verdict is a forward measurement anyway. */
                    const int cut = h->il2d_col.cut, nst = h->il2d_col.nst;
                    const int shape_fwd = fwd || _il2d_nat_bwd_fused(h);
                    const int Rl = h->il2d_col.R[nst - 1];
                    const size_t wl = (size_t)h->il2d_col.wl;
                    const size_t nstride = (size_t)h->N / (size_t)Rl;
                    const int *perm = h->il2d_col.natperm;
                    double *scr = h->il2d_col.natscr;
                    vfft_il2p_fn const *fns = shape_fwd ? h->il2d_col.f
                                                        : h->il2d_col.b;
                    double *const *tabs = shape_fwd ? h->il2d_col.tf
                                                    : h->il2d_col.tb;
                    size_t b0;
                    if (shape_fwd)
                    {
                        if (cut > 0)
                            _il2d_col_stages(sre, scr, h->N, rn, 0, cut,
                                             h->il2d_col.R, h->il2d_col.L, fns,
                                             tabs, 0);
                        for (b0 = 0; b0 < (size_t)h->N; b0 += wl)
                        {
                            const size_t blo = b0 / (size_t)Rl;
                            const size_t bhi = (b0 + wl) / (size_t)Rl;
                            const double *lf_from = (cut > 0) ? scr : sre;
                            size_t b;
                            if (cut < nst - 1)
                            {
                                _il2d_col_stages(lf_from + 2 * b0 * rn,
                                                 scr + 2 * b0 * rn,
                                                 (int)wl, rn, cut,
                                                 nst - 1, h->il2d_col.R,
                                                 h->il2d_col.L, fns, tabs, 0);
                                lf_from = scr;
                            }
                            (void)_il2d_nat_leaf_blocks(h, 0, dir, lf_from, dre, blo, bhi,
                                                        fwd ? h->il2d_col.tfuse : 1);
                        }
                        if (fwd && !h->il2d_col.tfuse)
                            _il2d_rows_exec(h, 0, dir, dre, rn, rn, 0, 1, (size_t)h->N);
                        return;
                    }
                    for (b0 = 0; b0 < (size_t)h->N; b0 += wl)
                    {
                        const size_t blo = b0 / (size_t)Rl;
                        const size_t bhi = (b0 + wl) / (size_t)Rl;
                        (void)_il2d_nat_leaf_blocks(h, 0, dir, sre, scr, blo, bhi, 0);
                        if (cut < nst - 1)
                            _il2d_col_stages(scr + 2 * b0 * rn,
                                             (cut > 0 ? scr : dre) + 2 * b0 * rn,
                                             (int)wl, rn, cut, nst - 1,
                                             h->il2d_col.R, h->il2d_col.L, fns,
                                             tabs, 1);
                    }
                    if (cut > 0)
                        _il2d_col_stages(scr, dre, h->N, rn, 0, cut,
                                         h->il2d_col.R, h->il2d_col.L, fns, tabs,
                                         1);
                    _il2d_rows_exec(h, 0, dir, dre, rn, rn, 0, 1, (size_t)h->N);
                    return;
                }
                if (h->il2d_col.wl > 0)
                {
                    /* ── BANDED walk (the cascade's tcut, 2D form):
                     * fwd = wide prefix stages 0..cut-1, then per
                     * band of wl rows the stage SUFFIX depth-first
                     * (+ tfuse: that band's row pass, while hot).
                     * bwd mirrors the Hermitian chain: per band
                     * rows-bwd then the REVERSED suffix, then the
                     * reversed wide prefix. Same kernel calls, same
                     * tables, same count — only loop order and base
                     * pointers differ (F0: memcmp-identical). */
                    /* Rows commute with every column stage (disjoint
                     * axes), so BOTH directions keep rows LAST in the
                     * band: the band's first op is the OOP-capable
                     * suffix kernel — no copy for OOP bwd. Execution:
                     * fwd = prefix wide, then per band [suffix fwd,
                     * rows]; bwd = per band [suffix REVERSED (the
                     * Hermitian chain), rows-bwd], then prefix
                     * reversed wide (in place on dre by then). */
                    const int cut = h->il2d_col.cut, nst = h->il2d_col.nst;
                    const size_t wl = (size_t)h->il2d_col.wl;
                    vfft_il2p_fn const *fns = fwd ? h->il2d_col.f
                                                  : h->il2d_col.b;
                    double *const *tabs = fwd ? h->il2d_col.tf
                                              : h->il2d_col.tb;
                    size_t b0;
                    /* the four-step's backward (il2d_fs_tw): its rows carry
                     * the conjugate inter-pass twiddle and must precede
                     * every column stage — unfused rows run here on dst
                     * first, fused rows run first in their band below */
                    const int hookb = (h->il2d_fs_tw != NULL) && !fwd;
                    if (hookb && !h->il2d_col.tfuse)
                    {
                        if (dre != sre)
                            memcpy(dre, sre, 2 * (size_t)h->N * rn * sizeof(double));
                        _il2d_rows_exec(h, 0, dir, dre, rn, rn, 0, 1, (size_t)h->N);
                        sre = dre;
                    }
                    if (fwd && cut > 0)
                        _il2d_col_stages(sre, dre, h->N, rn, 0, cut,
                                         h->il2d_col.R, h->il2d_col.L, fns,
                                         tabs, 0);
                    for (b0 = 0; b0 < (size_t)h->N; b0 += wl)
                    {
                        const double *bs =
                            (fwd && cut > 0) ? dre + 2 * b0 * rn
                                             : sre + 2 * b0 * rn;
                        double *bd = dre + 2 * b0 * rn;
                        if (h->il2d_col.staged)
                        {
                            /* §10b staged: band -> skewed scratch
                             * (kills the 4KB set-group aliasing,
                             * priced 2.4-3x on wide stages), suffix
                             * + rows there, copy back. count stays
                             * rn: identical arithmetic (F0). */
                            const size_t pit =
                                (size_t)h->il2d_col.pitch;
                            double *sc = h->il2d_col.bandscr;
                            for (i = 0; i < wl; i++)
                                memcpy(sc + 2 * i * pit,
                                       bs + 2 * i * rn,
                                       2 * rn * sizeof(double));
                            if (hookb && h->il2d_col.tfuse)
                                _il2d_rows_exec(h, 0, dir, sc, rn, pit, b0, 1, wl);
                            _il2d_col_stages2(sc, sc, (int)wl,
                                              pit, rn, cut, nst,
                                              h->il2d_col.R, h->il2d_col.L,
                                              fns, tabs, !fwd);
                            if (h->il2d_col.tfuse && !hookb)
                                _il2d_rows_exec(h, 0, dir, sc, rn, pit, b0, 1, wl);
                            for (i = 0; i < wl; i++)
                                memcpy(bd + 2 * i * rn,
                                       sc + 2 * i * pit,
                                       2 * rn * sizeof(double));
                            continue;
                        }
                        if (hookb && h->il2d_col.tfuse)
                        {
                            if (bd != bs)
                                memcpy(bd, bs, 2 * wl * rn * sizeof(double));
                            _il2d_rows_exec(h, 0, dir, bd, rn, rn, b0, 1, wl);
                            bs = bd;
                        }
                        _il2d_col_stages(bs, bd, (int)wl, rn, cut,
                                         nst, h->il2d_col.R, h->il2d_col.L,
                                         fns, tabs, !fwd);
                        if (h->il2d_col.tfuse && !hookb)
                            _il2d_rows_exec(h, 0, dir, bd, rn, rn, b0, 1, wl);
                    }
                    if (!fwd && cut > 0)
                        _il2d_col_stages(dre, dre, h->N, rn, 0, cut,
                                         h->il2d_col.R, h->il2d_col.L, fns,
                                         tabs, 1);
                    if (!h->il2d_col.tfuse && !hookb)
                        _il2d_rows_exec(h, 0, dir, dre, rn, rn, 0, 1, (size_t)h->N);
                    return;
                }
                if (h->il2d_fs_tw && !fwd)
                {   /* the four-step's backward: the twiddled rows first,
                     * then the column pass reversed in place on dst */
                    if (dre != sre)
                        memcpy(dre, sre, 2 * (size_t)h->N * rn * sizeof(double));
                    _il2d_rows_exec(h, 0, dir, dre, rn, rn, 0, 1, (size_t)h->N);
                    _il2d_col_pass(dre, dre, h->N, rn, wc, h->il2d_col.nst,
                                   h->il2d_col.R, h->il2d_col.L, h->il2d_col.b,
                                   h->il2d_col.tb, 1);
                    return;
                }
                if (h->il2d_col.nat && !fwd && _il2d_nat_bwd_fused(h))
                {   /* the unbanded natural backward through the forward's walk
                     * (2026-10-06): the mids with the forward kernels sre ->
                     * scratch (stage 0 the move), the forward leaf through the
                     * staging with the backward rows fused, scattered to the
                     * rows (N1 - i) mod N1 -- no row sweep */
                    const int nst = h->il2d_col.nst, Rl = h->il2d_col.R[nst - 1];
                    double *scr = h->il2d_col.natscr;
                    int s;
                    for (s = 0; s < nst - 1; s++)
                        _il2d_col_stages(s == 0 ? sre : scr, scr, h->N, rn, s, s + 1,
                                         h->il2d_col.R, h->il2d_col.L, h->il2d_col.f,
                                         h->il2d_col.tf, 0);
                    (void)_il2d_nat_leaf_blocks(h, 0, dir, scr, dre, 0, (size_t)h->N / (size_t)Rl, 1);
                    return;
                }
                if (h->il2d_col.nat)
                    _il2d_col_pass_nat(sre, dre, h->N, rn,
                                       h->il2d_col.nst, h->il2d_col.R,
                                       h->il2d_col.L,
                                       fwd ? h->il2d_col.f : h->il2d_col.b,
                                       fwd ? h->il2d_col.tf
                                           : h->il2d_col.tb,
                                       /*reverse=*/!fwd,
                                       h->il2d_col.natperm,
                                       h->il2d_col.natscr, _il2d_nat_stage_of(h, 0));
                else
                    _il2d_col_pass(sre, dre, h->N, rn, wc,
                                   h->il2d_col.nst, h->il2d_col.R,
                                   h->il2d_col.L,
                                   fwd ? h->il2d_col.f : h->il2d_col.b,
                                   fwd ? h->il2d_col.tf : h->il2d_col.tb,
                                   /*reverse=*/!fwd);
                _il2d_rows_exec(h, 0, dir, dre, rn, rn, 0, 1, (size_t)h->N);
                return;
            }
            /* OWNER LAW (2026-08-25): the convert wrapper is
             * GONE — an IL 2D c2c plan is native or was refused at
             * create; reaching here without il2d_row is a bug. */
            _vfft_warn("vfft_execute: IL 2D c2c plan without the "
                       "native tier — create/execute wiring bug");
            return;
        }
        else if (h->transform == VFFT_R2C && h->il2d_row)
        {
            /* ── native IL 2D REAL fwd (fft2d_real_il_design.md): the
             * batched TC K=N1 zr2c row door does the OOP move (real rows
             * at pitch N2 -> CCE half-spectrum plane at pitch hp1), then
             * the column chain runs IN PLACE over the hp1 columns.
             * Two-phase law (§2.5): the Hermitian fold is R-linear and
             * does not commute with the column stages — ALL rows fold
             * before column stage 0. nst>1 leaves the N1 axis
             * digit-reversed (the scrambled contract); rows (CCE bins)
             * stay natural. dir is ignored (r2c = forward math, the 1D
             * contract). */
            _il2d_real_rows_fwd(h, sre, dre);
            /* INC-3: the column pass as it serves -- the plan's form, threaded
             * under the colmt verdict (band or strip arm, both pure loop
             * restrictions => bitwise identical), serial when the threaded
             * walk cannot engage or the verdict is serial (the dispatcher). */
            _il2d_real_cols(h, dre, dre, /*reverse=*/0);
        }
        else if (h->transform == VFFT_C2R && h->il2d_row)
        {
            /* ── native IL 2D REAL bwd: the reversed column chain (the
             * Hermitian-transpose pair — conjugated tables pre-butterfly,
             * consuming the r2c pair's scrambled-N1 comb) moves the
             * caller's z into the il2d_rscr plane on its FIRST executed
             * stage (§2.6 input-preserving contract; c2r usually destroys its
             * input here — we don't), then the batched TC K=N1 c2r row
             * door folds rows scratch -> the caller's real plane. dir is
             * ignored (c2r = inverse math, unnormalized: caller divides
             * by N1*N2). */
            if (h->il2d_cxd_on && h->il2d_cxd_leaf && (const void *)sre != (const void *)dre)
            {   /* the destroying form (the request's permission; one thread): the
                 * column kernel in place on the caller's plane, the rows from it
                 * (never on an aliased call: the rows would read what they write) */
                _il2d_cxd_cols(h, (double *)sre);
                _il2d_real_rows_bwd(h, sre, dre);
                return;
            }
            _il2d_real_cols(h, sre, h->il2d_rscr, /*reverse=*/1);   /* threaded under the colmt verdict (the dispatcher) */
            _il2d_real_rows_bwd(h, h->il2d_rscr, dre);
        }
        return;
    }
    if (h->transform == VFFT_C2C && h->placement == VFFT_INPLACE)
    {
    /* interleaved z contract: the IL engines only (the convert executor
       * was deleted 2026-09-03). Padded plans can't get here:
       * batch+INTERLEAVED is rejected at create. */
        if (h->k1_mono_ilf)
        { /* MONO tier in place (2026-09-04): the alias-tolerant n1c solo,
           * one leg, z -> z legal by construction (no __restrict__). */
            double *zo = dre ? dre : (double *)sre;
            (dir == VFFT_FORWARD ? h->k1_mono_ilf : h->k1_mono_ilb)
                (sre, 0, zo, 0, 0, 0, 1, 0, 1, 0, 1);
            return;
        }
        if (h->k1il2p || h->k1il3p || h->k1ilpr || h->k1ilfd || h->k1ztt || h->k1fs)
        { /* Phase B (il_coverage_plan.md): sub-2048 native IL tier,
           * ALIASED — two-stage engines through internal scratch, zout
           * written only by the last stage (alias-gated, A3 record);
           * ilprime documents zin==zout safe in both methods.
           * Attach implies verdict (the ord=scr mode cell, mode=ILP);
           * all order spellings land here (identity under SCRAMBLED —
           * Phase A; primes/single-stage are natural = FREE). */
            double *zo = dre ? dre : (double *)sre;
            if (h->k1il2p)
            {
                if (dir == VFFT_FORWARD)
                    vfft_il2p_execute_fwd(h->k1il2p, sre, zo);
                else
                    (void)vfft_il2p_execute_bwd(h->k1il2p, sre, zo);
            }
            else if (h->k1il3p)
            {
                if (dir == VFFT_FORWARD)
                    vfft_il3p_execute_fwd(h->k1il3p, sre, zo);
                else
                    vfft_il3p_execute_bwd(h->k1il3p, sre, zo);
            }
            else if (h->k1ilfd)
            {   /* the flat DIT, z -> z legal (the leaf consumes zin first) */
                _ilfd_serve(h, dir, sre, zo);
            }
            else if (h->k1ztt)
            {   /* ZTURN-T, z -> z legal (the ingest consumes zin first) */
                _ztt_serve(h, dir, sre, zo);
            }
            else if (h->k1fs)
            {   /* the four-step in place: the 2D child in place, the natural
                 * class through its scratch plane */
                _k1x_fs(h, dir, sre, zo);
            }
            else
                _ilpr_serve(h, dir, sre, zo);
            return;
        }
        /* OWNER LAW (2026-09-03): no split-behind-convert engine for an
         * interleaved caller. An in-place interleaved handle with none of
         * the IL engines attached is a create bug, never a slow path. */
        _vfft_warn("vfft_execute: in-place IL c2c plan (N=%d) without an IL "
                   "engine — create/execute wiring bug; output NOT computed",
                   h->N);
        return;
    }
    if (h->transform == VFFT_C2C && h->placement == VFFT_OUTOFPLACE)
    {
    /* z -> z, by the committed axis (signature already validated). */
        if (h->k1_on)
        { /* K=1 engine (§13), IL routes; natural order both directions. */
            int fwd = (dir == VFFT_FORWARD);
            switch (h->k1_il_route)
            {
            case VFFT_K1_IL_MONO:
                /* ONE LEG: Ls = OLs = 1, count = 1 (the solo kernels
                 * index zin[2*(r*Ls + k)]); mono64 voids every size_t. */
                (fwd ? h->k1_mono_ilf : h->k1_mono_ilb)(sre, 0, dre, 0,
                                                        0, 0, 1, 0, 1, 0, 1);
                return;
            case VFFT_K1_IL_2P_PURE:
                /* Route truthfulness at create makes k1il2p non-NULL for this route; the guard
                 * is defensive — an unresolvable bwd arm breaks to convert, never to silence. */
                if (h->k1il2p)
                {
                    if (fwd)
                    {
                        vfft_il2p_execute_fwd(h->k1il2p, sre, dre);
                        return;
                    }
                    if (vfft_il2p_execute_bwd(h->k1il2p, sre, dre) == 0)
                        return;
                }
                break; /* -> convert fallback (NEVER a silent no-op) */
            case VFFT_K1_IL_CHAIN3:
                /* 3-STAGE PURE-IL CHAIN (odd·2^k N): both directions
                 * gated (fwd 12/12, bwd 13/13 — il_odd_chain.md). Route
                 * truthfulness guarantees k1il3p != NULL here; the guard
                 * is defensive, falling to convert, never to silence. */
                if (h->k1il3p)
                {
                    if (fwd)
                        vfft_il3p_execute_fwd(h->k1il3p, sre, dre);
                    else
                        vfft_il3p_execute_bwd(h->k1il3p, sre, dre);
                    return;
                }
                break; /* -> convert fallback (NEVER a silent no-op) */
            case VFFT_K1_IL_FLAT:
                /* the FLAT mixed-radix DIT (odd N): both directions,
                 * natural order; route truthfulness at create makes
                 * k1ilfd non-NULL here, the guard is defensive. */
                if (h->k1ilfd)
                {
                    _ilfd_serve(h, dir, sre, dre);
                    return;
                }
                break; /* -> convert fallback (NEVER a silent no-op) */
            case VFFT_K1_IL_ZTT:
                /* ZTURN-T: both directions, natural order; route
                 * truthfulness at create makes k1ztt non-NULL here. */
                if (h->k1ztt)
                {
                    _ztt_serve(h, dir, sre, dre);
                    return;
                }
                break; /* -> convert fallback (NEVER a silent no-op) */
            case VFFT_K1_IL_FS:
                /* the four-step: both directions, both classes */
                if (h->k1fs)
                {
                    _k1x_fs(h, dir, sre, dre);
                    return;
                }
                break;
            case VFFT_K1_IL_PRIME:
                /* PRIME N via Rader/Bluestein on IL inner plans
                 * (il_prime.h); both directions, natural order,
                 * unnormalized inverse like every IL bwd. */
                if (h->k1ilpr)
                {
                    _ilpr_serve(h, fwd ? VFFT_FORWARD : VFFT_BACKWARD, sre, dre);
                    return;
                }
                break; /* -> convert fallback (NEVER a silent no-op) */
            default:
                break; /* no IL route emitted for this N -> wiring bug below
                        * fallback below (NEVER a silent no-op) */
            }
        }
        /* No native z route on this cell (K>1, cascade-uncovered N, or
         * no K=1 IL route): convert around the split engines. */
        /* OWNER LAW (2026-09-03): the OOP create refuses an interleaved
         * request with no IL route; reaching here is a wiring bug. */
        _vfft_warn("vfft_execute: OOP IL c2c plan (N=%d K=%zu) without an IL "
                   "engine — create/execute wiring bug; output NOT computed",
                   h->N, h->K);
        return;
    }
}

#endif /* VFFT_IL_EXECUTE_H */
