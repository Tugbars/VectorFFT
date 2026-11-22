/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Tuğbars Heptaşkın */
/* fftnd_realn_il.h -- R2C AND C2R OF RANK 4 TO 7 (2026-11-22): one real array of N1 x N2 x .. x Nr, row-major,
 * the last axis the real one; its spectrum N1 x .. x N(r-1) x H complex, H = Nr / 2 + 1, interleaved. Out of
 * place, natural order, one transform per plan.
 *
 * The lane-major real batch's shape (rank3/lm_batch3dr_il.h) at one transform, with the axes in the real order:
 *   r2c  the LATER axes first: N1 blocks of the real plan one rank down at (N2, .., Nr) -- block i is the
 *        contiguous N2 .. Nr reals behind index i of the first axis, its spectrum the N2 .. N(r-1) H complex
 *        behind the same index -- then the FIRST axis: one complex column pass of N1 rows over the spectrum
 *        block's width, in place on the spectrum, natural;
 *   c2r  the mirror: the backward column pass over the spectrum, then the blocks through the child's c2r. The
 *        caller's spectrum survives: the column pass moves it into a scratch of the spectrum's size (lm_W) and
 *        the blocks read from there.
 * The child is the rank-3 real tier's plan at rank 4 (rank3/fftnd_real_il.h) and this door's own plan above it,
 * in role on a private store (its recipe rides on this cell's row as rp_*), one thread, out of place. The first
 * axis is the lane-major doors' column pass: the 1D batch's descriptor (il2d_col at rn = the block's width),
 * walker, race and dense scratch, unchanged; the blocks run through the real batch's block walker
 * (_lm2dr_rows, one block a call through the child's door). Threaded: the column pass through its raced forms,
 * the blocks over the workers on clones of the child, one threaded verdict (cmt=) raced on the whole execute.
 * THE BLOCK PHASE AT T (the complex tier's block batch, fftnd_il.h _ilnd_build_batch): the N1 blocks are also a
 * contiguous batch of N1 real transforms one rank down -- block i's reals at x + i N2 .. Nr, its spectrum at
 * z + i 2 N2 .. N(r-1) H -- and where that batch's way puts the threads on the blocks in turn (few blocks, each
 * past the cache: a first extent of 2 to 4 over big blocks, where the clones would leave threads idle) its plan
 * runs the block phase (lm_rowq, in role on the child's store beside the child); where its way is whole blocks
 * per worker the clones serve.
 * The row: t=r2c n=N1x..xNr q=1 on the lay=lm axis (the batch's row at one transform: at K = 1 the two batch
 * geometries are the same addressing), direction-shared, the first axis's column tokens on it.
 *
 * Not served here: in place, scrambled order, and a batch -- a contiguous batch (howmany >= 2) is the queue over
 * this plan (rank2/fft2d_create_il.h _vfft_create_pq_il). Included after lm_batch3dr_il.h. */
#ifndef VFFT_IL_RANK3_FFTND_REALN_IL_H
#define VFFT_IL_RANK3_FFTND_REALN_IL_H

/* the blocks behind one index of the first axis, in doubles: the reals (N2 .. Nr) and the spectrum
 * (2 N2 .. N(r-1) H) */
static void _ilrn_blocks(const struct vfft_plan_s *h, size_t *rb, size_t *cb)
{
    const int e[6] = { h->N2, h->N3, h->N4, h->N5, h->N6, h->N7 };
    size_t lead = 1;
    int r = 3, i;   /* the extents after the first: three at rank 4 */
    while (r < 6 && e[r] > 0)
        r++;
    for (i = 0; i < r - 1; i++)
        lead *= (size_t)e[i];
    *rb = lead * (size_t)e[r - 1];
    *cb = 2 * lead * ((size_t)e[r - 1] / 2 + 1);
}
/* THE BLOCK PHASE AT T: the contiguous batch of the N1 blocks, the real transform one rank down at howmany = N1
 * on T threads -- the request rc of the block child with the batch's count, geometry and threads -- planned in
 * role on the child's store (the batch's row and its plan of one block at T beside the child's rows). Kept where
 * its way is not whole blocks per worker (the queue's verdict, raced or replayed: all threads on each block in
 * turn, or groups); a banked way of whole blocks per worker builds nothing: the clones serve. NULL when not
 * built or not kept. */
static struct vfft_plan_s *_ilrn_build_batch(struct vfft_plan_s *h, const vfft_cfg_t *cfg, const vfft_cfg_t *rc, int T)
{
    vfft_cfg_t bc;
    struct vfft_plan_s *q;
    if (T < 2 || h->N < 2 || !h->lm_rowS)
        return NULL;
    bc = *rc;
    bc.howmany = (size_t)h->N;
    bc.batch_geom = VFFT_BATCH_CONTIGUOUS;
    bc.nthreads = T;
    bc.wisdom_write = 0;
    bc.recalibrate = cfg->recalibrate;
    if (!cfg->recalibrate)
    {   /* the batch's row (plane_queue.h _pq_row_key): whole blocks per worker banked -> nothing to build */
        vw2_key_t k;
        int i;
        vw2__pq_key(&k, rc->transform == VFFT_C2R ? VW2_T_C2R : VW2_T_R2C, rc->dims, rc->n[0], rc->n[1], rc->n[2],
                    (size_t)h->N, vfft_policy_ord_rankn(&bc), VW2_PL_OOP, VW2_LAY_IL, T);
        for (i = 3; i < rc->dims; i++)
            k.n[i] = rc->n[i];
        if (vw2_pq_lookup(&h->lm_rowS->vw2, &k) == 1)
            return NULL;
    }
    q = (struct vfft_plan_s *)_vfft_create_cfg(&bc);
    if (q && (!q->pq_inner || q->pq_mt == 1))
    {   /* not the queue, or its way is whole blocks per worker */
        vfft_destroy((vfft_plan)q);
        q = NULL;
    }
    return q;
}
/* the executors (the door binds them: a direction each). The blocks: through the block phase's batch plan where
 * the plan carries one (its geometry is the blocks': one after another on both sides), else the block walker */
static int _ilrn_exec_r2c(struct vfft_plan_s *h, vfft_dir_t dir, const vfft_real *xin, vfft_real *zout)
{
    size_t rb, cb;
    (void)dir;
    if (h->lm_rowq)
        _vfft_exec6((vfft_plan)h->lm_rowq, VFFT_FORWARD, (vfft_real *)xin, NULL, zout, NULL);
    else
    {
        _ilrn_blocks(h, &rb, &cb);
        _lm2dr_rows(h, xin, zout, rb, cb, VFFT_FORWARD);
    }
    _lm_cols(h, zout, zout, 0);   /* the first axis over the spectrum, in place */
    return 0;
}
static int _ilrn_exec_c2r(struct vfft_plan_s *h, vfft_dir_t dir, const vfft_real *zin, vfft_real *xout)
{
    size_t rb, cb;
    (void)dir;
    _lm_cols(h, zin, h->lm_W, 1);   /* the caller's spectrum survives: the backward column pass moves it into the scratch */
    if (h->lm_rowq)
        _vfft_exec6((vfft_plan)h->lm_rowq, VFFT_BACKWARD, h->lm_W, NULL, xout, NULL);
    else
    {
        _ilrn_blocks(h, &rb, &cb);
        _lm2dr_rows(h, h->lm_W, xout, cb, rb, VFFT_BACKWARD);
    }
    return 0;
}

/* ── THE DOOR: R2C or C2R of rank 4 to 7, by the request ─────────────────── */
static vfft_plan _vfft_create_realn_il(const vfft_cfg_t *cfg, struct vfft_wisdom_s *W, size_t K)
{
    const int T = _vfft_plan_threads(cfg), rank = cfg->dims, N1 = cfg->n[0];
    const int c2r = cfg->transform == VFFT_C2R;
    size_t lanes = 1;   /* the spectrum block's width in complex: N2 .. N(r-1) H */
    vw2_ilcol_key_t ck;
    vw2_key_t pk;
    vfft_ilcol_t col;
    vfft_cfg_t rc;
    char forms[64], shp[96];
    int bwl = -1, btf = -1, bro = -1, bcmt = -1, bcmtt = -1, bblu = -1, oom = 0, i, small = 0;
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
            small |= cfg->n[i] < 2;
        }
    }
    if (rank < 4 || rank > 7 || (cfg->transform != VFFT_R2C && cfg->transform != VFFT_C2R) || K != 1 ||
        cfg->layout != VFI_LAYOUT_IL || cfg->placement == VFFT_INPLACE || cfg->order == VFI_ORDER_SCRAMBLED || small)
    {
        _vfft_warn("vfft_create: %dD INTERLEAVED %s serves one transform per plan, out of place, natural order, every "
                   "extent >= 2 (got %s, howmany=%zu, %s%s); a contiguous batch runs through the queue over this plan, "
                   "out of place; lane-major and the in-place form are served up to rank 3",
                   rank, _vfft_tname(cfg->transform), shp, K, cfg->placement == VFFT_INPLACE ? "in place" : "out of place",
                   cfg->order == VFI_ORDER_SCRAMBLED ? ", scrambled order" : "");
        return NULL;
    }
    for (i = 1; i < rank - 1; i++)
        lanes *= (size_t)cfg->n[i];
    lanes *= (size_t)cfg->n[rank - 1] / 2 + 1;
    memset(&ck, 0, sizeof ck);
    ck.rank = rank; ck.n0 = N1; ck.n1 = (size_t)cfg->n[1]; ck.n2 = (size_t)cfg->n[2]; ck.q = 1; ck.ord = VW2_ORD_NAT;
    ck.real = 1; ck.nthreads = T; ck.lm = 1;
    for (i = 3; i < rank; i++)   /* the extents after the third */
        ck.nx[i - 3] = cfg->n[i];
    _il2d_blu_ctx.W = W;
    _il2d_blu_ctx.cfg = cfg;
    _il2d_blu_ctx.N2 = lanes;
    _il2d_blu_chain_hook = _il2d_blu_m_chain;
    memset(&col, 0, sizeof col);
    forms[0] = 0;
    /* the first axis: a complex column pass over the spectrum block's width, natural (the leaf redirection) */
    if (!_il2d_col_build(W, cfg, &ck, N1, lanes, /*nat_req=*/1, &col, forms, sizeof forms,
                         &bwl, &btf, &bro, &bcmt, &bcmtt, &bblu, &oom))
    {
        if (!oom)   /* out of memory: the builder's reason stands */
            _vfft_warn("vfft_create: %dD INTERLEAVED %s %s: no column pass serves the first axis (the column builder "
                       "refused); unsupported", rank, _vfft_tname(cfg->transform), shp);
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
    h->N2 = cfg->n[1];
    h->N3 = cfg->n[2];
    h->N4 = cfg->n[3];
    h->N5 = rank > 4 ? cfg->n[4] : 0;
    h->N6 = rank > 5 ? cfg->n[5] : 0;
    h->N7 = rank > 6 ? cfg->n[6] : 0;
    h->K = 1;
    h->nthreads = T;
    h->lm = 1;               /* the lane-major doors' handle: the column pass, the block child, the scratch */
    h->il2d_col = col;
    h->il2d_col.N = N1;
    h->il2d_col.rn = lanes;
    h->il2d_col.wl = 0;      /* unbanded: the later axes are the child's, never fused here */
    h->il2d_col.cut = 0;
    h->il2d_col.tfuse = 0;
    h->il2d_col.wc = 0;
    /* THE CHILD IN ROLE (owner, 2026-10-02): the real plan one rank down at (N2, .., Nr), the request's
     * direction, out of place, on its own store seeded from this cell's row (rp_*); it never reads or writes
     * a row of its own rank */
    vw2__ilcol_key(&ck, &pk);
    h->lm_rowS = vfft_child_store_for(&W->vw2, &pk, "rp_");
    memset(&rc, 0, sizeof rc);
    rc.transform = cfg->transform;
    rc.placement = VFFT_OUTOFPLACE;
    rc.rigor = cfg->rigor;
    rc.dims = rank - 1;
    for (i = 1; i < rank; i++)
        rc.n[i - 1] = cfg->n[i];
    rc.howmany = 1;
    rc.order = VFI_ORDER_DEFAULT;
    rc.layout = VFI_LAYOUT_IL;
    rc.batch_geom = VFFT_BATCH_DEFAULT;
    rc.nthreads = 1;
    rc.wisdom = (vfft_wisdom *)h->lm_rowS;
    rc.recalibrate = cfg->recalibrate;
    h->lm_row = h->lm_rowS ? (struct vfft_plan_s *)_vfft_create_cfg(&rc) : NULL;
    if (!h->lm_row)
    {   /* the child's own reason first (an out of memory among them), then whose child it was */
        char why[VFFT_ERR_CAP];
        snprintf(why, sizeof why, "%s", h->lm_rowS ? _vfft_err_buf : "vfft_create: out of memory (the child's store)");
        _vfft_warn("%s [the child of the %dD INTERLEAVED %s %s]", why, rank, _vfft_tname(cfg->transform), shp);
        vfft_child_store_free(h->lm_rowS);
        _il2d_col_free(&h->il2d_col);
        free(h);
        return NULL;
    }
    if (c2r)
    {   /* the scratch the backward column pass lands in: the spectrum's size */
        h->lm_W = (double *)vfft_aligned_alloc((2 * (size_t)N1 * lanes + 8) * sizeof(double));
        if (!h->lm_W)
        {   /* OUT OF MEMORY: the spectrum-sized scratch -- refused with that cause */
            _vfft_warn("vfft_create: %dD INTERLEAVED %s %s: out of memory (its backward column scratch needs %.1f MB)",
                       rank, _vfft_tname(cfg->transform), shp,
                       (2.0 * (double)N1 * (double)lanes + 8.0) * sizeof(double) / 1048576.0);
            vfft_destroy((vfft_plan)h->lm_row);
            vfft_child_store_free(h->lm_rowS);
            _il2d_col_free(&h->il2d_col);
            free(h);
            return NULL;
        }
    }
    if (T > 1)
        h->lm_rowq = _ilrn_build_batch(h, cfg, &rc, T);   /* the block phase's batch plan, where it serves */
    if (T > 1 && !h->lm_rowq)
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
    {   /* the child's recipe onto the cell's row */
        int changed = vfft_child_row_update(&W->vw2, &pk, "rp_", h->lm_rowS);
        if (h->il2d_col.tpcS)
            changed |= vfft_child_row_update(&W->vw2, &pk, "tpc_", h->il2d_col.tpcS);
        if (changed)
            _vw2_persist(W, cfg);
    }
    h->k1_exec = c2r ? _ilrn_exec_c2r : _ilrn_exec_r2c;   /* the lean door: this door's executors */
    h->k1_go[VFFT_FORWARD] = c2r ? NULL : _ilrn_exec_r2c;
    h->k1_go[VFFT_BACKWARD] = c2r ? _ilrn_exec_c2r : NULL;
    (void)bcmtt; (void)bwl; (void)btf; (void)bro; (void)bblu;
    return (vfft_plan)h;
}
#endif /* VFFT_IL_RANK3_FFTND_REALN_IL_H */
