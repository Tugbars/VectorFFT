/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Tuğbars Heptaşkın */
/* lm_batch3dr_il.h -- THE LANE-MAJOR 3D REAL BATCH, r2c and c2r (docs/roadmap/lane_major_batch_design.md,
 * step 6 one rank up, 2026-10-19): K real volumes of N1 x N2 x N3 side by side,
 * x[((i N2 + j) N3 + k) K + t]; the spectrum z[2 (((i N2 + j) H + f) K + t)], f < H = N3 / 2 + 1. The 3D
 * c2c batch's shape (lm_batch3d_il.h) with the axes in the real order (the 2D real batch's,
 * lm_batch2dr_il.h):
 *   r2c  the LAST axes first: N1 blocks of the lane-major 2D REAL batch at (N2, N3, K) -- block i is the
 *        contiguous N2 N3 K reals of plane i (in place: its N2 (N3 + 2) K-double padded block), the child
 *        in role on a private store (rp_*; its own 1D real child nests inside it as rp_1_rp_*) -- into
 *        the spectrum's plane blocks of 2 N2 H K doubles; then the FIRST axis: one complex column pass
 *        of N1 rows over N2 H K lanes, in place on the spectrum, natural (a real spectrum is natural);
 *   c2r  the mirror: the backward column pass over the spectrum, then the blocks through the 2D c2r
 *        batch. Out of place the caller's spectrum must survive, so the column pass moves it into a
 *        scratch of the spectrum's size (lm_W) and the blocks read from there; in place both passes run
 *        on the one padded buffer (N1 N2 rows of (N3 + 2) K doubles, the reals first in each row, the
 *        bins over them).
 * The executors are the 2D real batch's (_lm2dr_exec_r2c / _lm2dr_exec_c2r), rank-aware through their
 * block sizes (N3 > 0: the 2D child's planes), so the race arm, the lean binding and the destroy serve
 * unchanged. Threaded: as the 2D real batch -- the first axis through its raced forms, the blocks over
 * the workers on clones of the 2D child (serial inside), one threaded verdict (cmt=) raced with the
 * column pass's arms, the arms running the whole executor. The row: t=r2c n=N1xN2xN3 q=K on the lay=lm
 * axis, direction-shared as the 1D and 2D real rows are (the c2r twin replays the r2c's verdicts), and
 * IN PLACE ITS OWN ROW (place=ip, the single-volume rank-3 real tier's spelling, fftnd_real_il.h: the
 * in-place form runs one more pass over every plane in the 1D child) -- unlike the placement-blind 1D
 * and 2D real lane-major rows. Included after lm_batch2dr_il.h and lm_batch3d_il.h, before the door.
 *
 * RANK 4 TO 7 (2026-11-22): the same door one rank up again, out of place. K real arrays of N1 x .. x Nr side
 * by side, the element at row-major position e of transform t at x[e K + t]; r2c runs the N1 blocks behind the
 * first axis through the lane-major real batch one rank down at (N2, .., Nr, K) -- this door's own plan at rank
 * r - 1, in role under rp_ -- then the first axis over N2 .. N(r-1) H K lanes, H = Nr / 2 + 1; c2r the mirror.
 * The row: the real cell of that rank and shape at q = K on the lay=lm axis. */
#ifndef VFFT_IL_LM_BATCH3DR_IL_H
#define VFFT_IL_LM_BATCH3DR_IL_H

/* ── THE DOOR: the lane-major 3D real batch's create, r2c or c2r by the request ─────── */
static vfft_plan _vfft_create_lm3dr_il(const vfft_cfg_t *cfg, struct vfft_wisdom_s *W, size_t K)
{
    const int T = _vfft_plan_threads(cfg), rank = cfg->dims, N1 = cfg->n[0], N2 = cfg->n[1], N3 = cfg->n[2];
    const int c2r = cfg->transform == VFFT_C2R, ip = cfg->placement == VFFT_INPLACE;
    const size_t H = (size_t)cfg->n[rank >= 3 && rank <= 7 ? rank - 1 : 2] / 2 + 1;   /* the last axis's bins */
    size_t lanes = H * K;                                                              /* x the extents between */
    vw2_ilcol_key_t ck;
    vw2_key_t pk;
    vfft_ilcol_t col;
    vfft_cfg_t rc;
    char forms[64], shp[96];
    int bwl = -1, btf = -1, bro = -1, bcmt = -1, bcmtt = -1, bblu = -1, oom = 0, i;
    struct vfft_plan_s *h;
    {   /* the shape for the messages: N1xN2x..xNr */
        size_t off = 0;
        shp[0] = 0;
        for (i = 0; i < rank && i < 7 && off < sizeof shp; i++)
        {
            const int w = snprintf(shp + off, sizeof shp - off, "%s%d", i ? "x" : "", cfg->n[i]);
            if (w < 0)
                break;
            off += (size_t)w;
        }
    }
    if (rank < 3 || rank > 7 || !vfft_policy_batch_lane_major(cfg))
    {
        _vfft_warn("vfft_create: the lane-major interleaved %dD real batch serves R2C and C2R at howmany >= 2, every "
                   "extent >= 2 and an even last extent >= 4, above rank 3 out of place (got %s %s howmany=%zu) -- an "
                   "odd last extent has no real column engine; lay the %zu transforms out one after another "
                   "(VFFT_BATCH_DEFAULT or VFFT_BATCH_CONTIGUOUS) for it",
                   rank, _vfft_tname(cfg->transform), shp, K, K);
        return NULL;
    }
    for (i = 1; i < rank - 1; i++)
        lanes *= (size_t)cfg->n[i];
    memset(&ck, 0, sizeof ck);
    ck.rank = rank; ck.n0 = N1; ck.n1 = (size_t)N2; ck.n2 = (size_t)N3; ck.q = K; ck.ord = VW2_ORD_NAT; ck.real = 1; ck.nthreads = T; ck.lm = 1;
    for (i = 3; i < rank; i++)   /* rank 4 to 7: the extents after the third */
        ck.nx[i - 3] = cfg->n[i];
    ck.ip = ip;   /* the in-place cell's own row (place=ip), as the single-volume rank-3 real tier keys it */
    _il2d_blu_ctx.W = W;
    _il2d_blu_ctx.cfg = cfg;
    _il2d_blu_ctx.N2 = lanes;
    _il2d_blu_chain_hook = _il2d_blu_m_chain;
    memset(&col, 0, sizeof col);
    forms[0] = 0;
    /* the first axis: a complex column pass over the spectrum's width, natural (the leaf redirection) */
    if (!_il2d_col_build(W, cfg, &ck, N1, lanes, /*nat_req=*/1, &col, forms, sizeof forms,
                         &bwl, &btf, &bro, &bcmt, &bcmtt, &bblu, &oom))
    {
        if (!oom)   /* out of memory: the builder's reason stands */
            _vfft_warn("vfft_create: lane-major interleaved %dD %s batch %s x %zu: no column pass serves the first axis "
                       "(the column builder refused); unsupported", rank, _vfft_tname(cfg->transform), shp, K);
        return NULL;
    }
    h = (struct vfft_plan_s *)calloc(1, sizeof *h);
    if (!h)
    {
        _il2d_col_free(&col);
        return NULL;
    }
    h->transform = c2r ? VFFT_C2R : VFFT_R2C;
    h->placement = cfg->placement;
    h->layout = (int)VFI_LAYOUT_IL;
    h->N = N1;
    h->N2 = N2;
    h->N3 = N3;
    if (rank > 3)
    {   /* rank 4 to 7: the extents after the third */
        h->N4 = cfg->n[3];
        h->N5 = rank > 4 ? cfg->n[4] : 0;
        h->N6 = rank > 5 ? cfg->n[5] : 0;
        h->N7 = rank > 6 ? cfg->n[6] : 0;
    }
    h->K = K;
    h->nthreads = T;
    h->lm = 1;
    h->il2d_col = col;
    h->il2d_col.N = N1;
    h->il2d_col.rn = lanes;
    h->il2d_col.wl = 0;      /* the batch runs unbanded: the later axes are the child's, never fused here */
    h->il2d_col.cut = 0;
    h->il2d_col.tfuse = 0;
    h->il2d_col.wc = 0;
    /* THE CHILD IN ROLE (owner, 2026-10-02): the lane-major 2D REAL batch at (N2, N3, K), the request's
     * direction and placement, on its own store seeded from this cell's row (rp_*); it never reads or
     * writes a 2D row */
    vw2__ilcol_key(&ck, &pk);
    h->lm_rowS = vfft_child_store_for(&W->vw2, &pk, "rp_");
    memset(&rc, 0, sizeof rc);
    rc.transform = cfg->transform;
    rc.placement = cfg->placement;
    rc.rigor = cfg->rigor;
    rc.dims = rank - 1;   /* the lane-major real batch one rank down: the 2D one at rank 3, this door's above it */
    for (i = 1; i < rank; i++)
        rc.n[i - 1] = cfg->n[i];
    rc.howmany = K;
    rc.order = VFI_ORDER_DEFAULT;
    rc.layout = VFI_LAYOUT_IL;
    rc.batch_geom = VFFT_BATCH_LANE_MAJOR;
    rc.nthreads = 1;
    rc.wisdom = (vfft_wisdom *)h->lm_rowS;
    rc.recalibrate = cfg->recalibrate;
    h->lm_row = h->lm_rowS ? (struct vfft_plan_s *)_vfft_create_cfg(&rc) : NULL;
    if (!h->lm_row)
    {   /* the child's own reason first (an out of memory among them), then whose child it was */
        char why[VFFT_ERR_CAP];
        snprintf(why, sizeof why, "%s", h->lm_rowS ? _vfft_err_buf : "vfft_create: out of memory (the child's store)");
        _vfft_warn("%s [the child of the lane-major interleaved %dD %s batch %s x %zu]", why, rank,
                   _vfft_tname(cfg->transform), shp, K);
        vfft_child_store_free(h->lm_rowS);
        _il2d_col_free(&h->il2d_col);
        free(h);
        return NULL;
    }
    if (c2r && !ip)
    {   /* the scratch the backward column pass lands in: the spectrum's size */
        h->lm_W = (double *)vfft_aligned_alloc((2 * (size_t)N1 * lanes + 8) * sizeof(double));
        if (!h->lm_W)
        {   /* OUT OF MEMORY: the spectrum-sized scratch -- refused with that cause */
            _vfft_warn("vfft_create: lane-major interleaved %dD %s batch %s x %zu: out of memory (its backward "
                       "column scratch needs %.1f MB)", rank, _vfft_tname(cfg->transform), shp, K,
                       (2.0 * (double)N1 * (double)lanes + 8.0) * sizeof(double) / 1048576.0);
            vfft_destroy((vfft_plan)h->lm_row);
            vfft_child_store_free(h->lm_rowS);
            _il2d_col_free(&h->il2d_col);
            free(h);
            return NULL;
        }
    }
    if (T > 1)
    {   /* the clones before the race: its threaded arms run the blocks over them */
        rc.recalibrate = 0;   /* a clone replays what the primary banked */
        (void)_lm2d_clones_build(h, &rc, T);
    }
    if (T <= 1 && !h->il2d_col.blu && !h->il2d_col.tpc)
        (void)_lm_sscr_build(&h->il2d_col, N1, lanes, 1);   /* the first axis's serial dense tiles' scratch */
    if (_lm_race(h, W, cfg, &ck, bcmt))
    {   /* OUT OF MEMORY in the race (its reason recorded). Not retained yet (the front door retains the plan it
         * returns): its own release is skipped */
        h->nthreads = 1;
        vfft_destroy((vfft_plan)h);
        return NULL;
    }
    if (!h->il2d_col.colmt)
        _lm2d_clones_drop(h);   /* a serial verdict runs the blocks on the primary alone */
    if (W && !W->vw2_off_2d)
    {   /* the child's recipe onto the cell's row (and a turned prime pass's, as the 1D create banks it) */
        int changed = vfft_child_row_update(&W->vw2, &pk, "rp_", h->lm_rowS);
        if (h->il2d_col.tpcS)
            changed |= vfft_child_row_update(&W->vw2, &pk, "tpc_", h->il2d_col.tpcS);
        if (changed)
            _vw2_persist(W, cfg);
    }
    h->k1_exec = c2r ? _lm2dr_exec_c2r : _lm2dr_exec_r2c;   /* the lean door: the 2D real batch's executors, the blocks from N3 */
    h->k1_go[VFFT_FORWARD] = c2r ? NULL : _lm2dr_exec_r2c;
    h->k1_go[VFFT_BACKWARD] = c2r ? _lm2dr_exec_c2r : NULL;
    (void)bcmtt; (void)bwl; (void)btf; (void)bro; (void)bblu;
    return (vfft_plan)h;
}
#endif /* VFFT_IL_LM_BATCH3DR_IL_H */
