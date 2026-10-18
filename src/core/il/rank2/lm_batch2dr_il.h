/* lm_batch2dr_il.h -- THE LANE-MAJOR 2D REAL BATCH, r2c and c2r (docs/roadmap/lane_major_batch_design.md,
 * step 6, 2026-10-15): K real planes of N1 x N2 side by side, x[(r N2 + c) K + t]; the spectrum
 * z[2 ((r H + f) K + t)], f < H = N2 / 2 + 1. The 2D c2c batch's shape (lm_batch2d_il.h) with the
 * axes in the real order:
 *   r2c  the LAST axis first: N1 blocks of the 1D real lane-major batch at (N2, K) -- block r is
 *        the contiguous N2 K reals of row r (in place: its (N2 + 2) K-double plane), the child in
 *        role on a private store (rp_*), its fused strips -- into the spectrum's row-blocks; then
 *        the FIRST axis: one complex column pass of N1 rows over H K lanes, in place on the
 *        spectrum, natural (the cell's order; a real spectrum is natural);
 *   c2r  the mirror: the backward column pass over the spectrum, then the blocks through the 1D c2r
 *        batch. Out of place the caller's spectrum must survive, so the column pass moves it into
 *        a scratch of the spectrum's size (lm_W) and the blocks read from there; in place both
 *        passes run on the one padded plane (N1 blocks of (N2 + 2) K doubles).
 * Threaded: the column pass through its raced forms, the blocks over the workers on clones of the
 * row child, one threaded verdict (cmt=) raced with the column pass's arms (the arms run the whole
 * executor). The row: t=r2c n=N1xN2 q=K on the lay=lm axis, direction-shared as the 1D real row is,
 * the first axis's column tokens on it. The 3D real batch (rank3/lm_batch3dr_il.h) runs through the
 * same block walker and executors: its blocks are the planes of N2 (N3 + 2) K doubles, its child this
 * 2D real batch. Included after lm_batch2d_il.h and before the door. */
#ifndef VFFT_IL_LM_BATCH2DR_IL_H
#define VFFT_IL_LM_BATCH2DR_IL_H

/* the row blocks: worker t runs blocks [lo, hi) through its own row plan, block r of the input at
 * in + r sblk, of the output at out + r dblk (the same base and block in place) */
typedef struct { struct vfft_plan_s *h; const double *in; double *out; size_t sblk, dblk, lo, hi; vfft_dir_t dir; int tid; } _lm2dr_rows_arg_t;
static void _lm2dr_rows_tramp(void *v)
{
    _lm2dr_rows_arg_t *a = (_lm2dr_rows_arg_t *)v;
    struct vfft_plan_s *rp = a->tid > 0 ? a->h->lm_roww[a->tid - 1] : a->h->lm_row;
    size_t r;
    for (r = a->lo; r < a->hi; r++)
        _vfft_exec6((vfft_plan)rp, a->dir, (double *)a->in + r * a->sblk, NULL, a->out + r * a->dblk, NULL);
}
static void _lm2dr_rows(struct vfft_plan_s *h, const double *in, double *out, size_t sblk, size_t dblk, vfft_dir_t dir)
{
    _lm2dr_rows_arg_t a[THREAD_POOL_MAX_DISPATCH];
    const size_t N1 = (size_t)h->N;
    int T = (h->nthreads > 1 && h->il2d_col.colmt && h->lm_roww_n > 0) ? thread_pool_workers_for(h->nthreads) : 1;
    int t;
    if (T > h->lm_roww_n + 1)
        T = h->lm_roww_n + 1;
    if ((size_t)T > N1)
        T = (int)N1;
    for (t = 0; t < (T < 2 ? 1 : T); t++)
    {
        a[t].h = h; a[t].in = in; a[t].out = out; a[t].sblk = sblk; a[t].dblk = dblk; a[t].dir = dir; a[t].tid = t;
        a[t].lo = T < 2 ? 0 : N1 * (size_t)t / (size_t)T;
        a[t].hi = T < 2 ? N1 : N1 * (size_t)(t + 1) / (size_t)T;
    }
    if (T < 2)
        _lm2dr_rows_tramp(&a[0]);
    else
        thread_pool_run(T, _lm2dr_rows_tramp, a, sizeof a[0]);
}
/* the executors (bound at create: a direction each, the other refused by the door). The blocks are the
 * child's: at rank 2 the 1D real batch's rows (L = N2 reals a lane, M = 1), at rank 3 (the 3D real batch,
 * rank3/lm_batch3dr_il.h) the 2D real batch's planes (L = N3, M = N2 rows each) -- a real block M L K,
 * its spectrum block M (L + 2) K (2 (L / 2 + 1) = L + 2 at an even L), the padded block the same */
static int _lm2dr_exec_r2c(struct vfft_plan_s *h, vfft_dir_t dir, const double *xin, double *zout)
{
    const size_t K = h->K, L = (size_t)(h->N3 > 0 ? h->N3 : h->N2), M = (size_t)(h->N3 > 0 ? h->N2 : 1);
    const int ip = h->placement == VFFT_INPLACE;
    (void)dir;
    _lm2dr_rows(h, xin, zout, M * (ip ? L + 2 : L) * K, M * (L + 2) * K, VFFT_FORWARD);
    _lm_cols(h, zout, zout, 0);   /* the first axis over the spectrum, in place */
    return 0;
}
static int _lm2dr_exec_c2r(struct vfft_plan_s *h, vfft_dir_t dir, const double *zin, double *xout)
{
    const size_t K = h->K, L = (size_t)(h->N3 > 0 ? h->N3 : h->N2), M = (size_t)(h->N3 > 0 ? h->N2 : 1);
    const int ip = h->placement == VFFT_INPLACE;
    (void)dir;
    if (ip)
    {
        _lm_cols(h, zin, xout, 1);   /* zin == xout: in place on the padded plane */
        _lm2dr_rows(h, xout, xout, M * (L + 2) * K, M * (L + 2) * K, VFFT_BACKWARD);
    }
    else
    {   /* the caller's spectrum survives: the backward column pass moves it into the scratch */
        _lm_cols(h, zin, h->lm_W, 1);
        _lm2dr_rows(h, h->lm_W, xout, M * (L + 2) * K, M * L * K, VFFT_BACKWARD);
    }
    return 0;
}

/* ── THE DOOR: the lane-major 2D real batch's create, r2c or c2r by the request ─────── */
static vfft_plan _vfft_create_lm2dr_il(const vfft_cfg_t *cfg, struct vfft_wisdom_s *W, size_t K)
{
    const int T = _vfft_plan_threads(cfg), N1 = cfg->n[0], N2 = cfg->n[1], c2r = cfg->transform == VFFT_C2R;
    const size_t H = (size_t)N2 / 2 + 1, lanes = H * K;
    vw2_ilcol_key_t ck;
    vw2_key_t pk;
    vfft_ilcol_t col;
    vfft_cfg_t rc;
    char forms[64];
    int bwl = -1, btf = -1, bro = -1, bcmt = -1, bcmtt = -1, bblu = -1;
    struct vfft_plan_s *h;
    if (!vfft_policy_batch_lane_major(cfg))
    {
        _vfft_warn("vfft_create: the lane-major interleaved 2D real batch serves R2C and C2R at howmany >= 2, N1 >= 2 "
                   "and an even N2 >= 4 (got %s %dx%d howmany=%zu) -- an odd N2 has no real column engine",
                   _vfft_tname(cfg->transform), N1, N2, K);
        return NULL;
    }
    memset(&ck, 0, sizeof ck);
    ck.rank = 2; ck.n0 = N1; ck.n1 = N2; ck.n2 = (int)K; ck.ord = VW2_ORD_NAT; ck.real = 1; ck.nthreads = T; ck.lm = 1;
    _il2d_blu_ctx.W = W;
    _il2d_blu_ctx.cfg = cfg;
    _il2d_blu_ctx.N2 = (int)lanes;
    _il2d_blu_chain_hook = _il2d_blu_m_chain;
    memset(&col, 0, sizeof col);
    forms[0] = 0;
    /* the first axis: a complex column pass over the spectrum's width, natural (the leaf redirection) */
    if (!_il2d_col_build(W, cfg, &ck, N1, lanes, /*nat_req=*/1, &col, forms, sizeof forms,
                         &bwl, &btf, &bro, &bcmt, &bcmtt, &bblu))
    {
        _vfft_warn("vfft_create: lane-major interleaved 2D %s batch %dx%d x %zu: no column pass serves the first axis "
                   "(the column builder refused); unsupported", _vfft_tname(cfg->transform), N1, N2, K);
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
    h->K = K;
    h->nthreads = T;
    h->lm = 1;
    h->il2d_col = col;
    h->il2d_col.N = N1;
    h->il2d_col.rn = lanes;
    h->il2d_col.wl = 0;
    h->il2d_col.cut = 0;
    h->il2d_col.tfuse = 0;
    h->il2d_col.wc = 0;
    /* THE ROW CHILD IN ROLE: the 1D real lane-major batch at (N2, K), the request's direction and
     * placement, on its own store seeded from this cell's row (rp_*) */
    vw2__ilcol_key(&ck, &pk);
    h->lm_rowS = vfft_child_store_for(&W->vw2, &pk, "rp_");
    memset(&rc, 0, sizeof rc);
    rc.transform = cfg->transform;
    rc.placement = cfg->placement;
    rc.rigor = cfg->rigor;
    rc.dims = 1;
    rc.n[0] = N2;
    rc.howmany = K;
    rc.order = VFI_ORDER_DEFAULT;
    rc.layout = VFI_LAYOUT_IL;
    rc.batch_geom = VFFT_BATCH_LANE_MAJOR;
    rc.nthreads = 1;
    rc.wisdom = (vfft_wisdom *)h->lm_rowS;
    rc.recalibrate = cfg->recalibrate;
    h->lm_row = h->lm_rowS ? (struct vfft_plan_s *)_vfft_create_cfg(&rc) : NULL;
    if (!h->lm_row)
    {
        _vfft_warn("vfft_create: lane-major interleaved 2D %s batch %dx%d x %zu: the row child (the 1D real batch at "
                   "%d x %zu) did not build; unsupported", _vfft_tname(cfg->transform), N1, N2, K, N2, K);
        vfft_child_store_free(h->lm_rowS);
        _il2d_col_free(&h->il2d_col);
        free(h);
        return NULL;
    }
    if (c2r && cfg->placement != VFFT_INPLACE)
    {   /* the scratch the backward column pass lands in: the spectrum's size */
        h->lm_W = (double *)vfft_aligned_alloc((2 * (size_t)N1 * lanes + 8) * sizeof(double));
        if (!h->lm_W)
        {
            vfft_destroy((vfft_plan)h->lm_row);
            vfft_child_store_free(h->lm_rowS);
            _il2d_col_free(&h->il2d_col);
            free(h);
            return NULL;
        }
    }
    if (T > 1)
    {
        rc.recalibrate = 0;
        (void)_lm2d_clones_build(h, &rc, T);
    }
    if (T <= 1 && !h->il2d_col.blu && !h->il2d_col.tpc)
        (void)_lm_sscr_build(&h->il2d_col, N1, (int)lanes, 1);
    _lm_race(h, W, cfg, &ck, bcmt);
    if (!h->il2d_col.colmt)
        _lm2d_clones_drop(h);
    if (W && !W->vw2_off_2d)
    {
        int changed = vfft_child_row_update(&W->vw2, &pk, "rp_", h->lm_rowS);
        if (h->il2d_col.tpcS)
            changed |= vfft_child_row_update(&W->vw2, &pk, "tpc_", h->il2d_col.tpcS);
        if (changed)
            _vw2_persist(W, cfg);
    }
    h->k1_exec = c2r ? _lm2dr_exec_c2r : _lm2dr_exec_r2c;   /* the lean door: the direction's own executor */
    h->k1_go[VFFT_FORWARD] = c2r ? NULL : _lm2dr_exec_r2c;
    h->k1_go[VFFT_BACKWARD] = c2r ? _lm2dr_exec_c2r : NULL;
    (void)bcmtt; (void)bwl; (void)btf; (void)bro; (void)bblu;
    return (vfft_plan)h;
}
#endif /* VFFT_IL_LM_BATCH2DR_IL_H */
