/* lm_batch2d_il.h -- THE LANE-MAJOR 2D BATCH (docs/roadmap/lane_major_batch_design.md, step 6,
 * 2026-10-15): K >= 2 planes of N1 x N2 side by side, element (r, c) of transform t at
 * z[2 ((r N2 + c) K + t)]. Every axis is a column pass. The FIRST axis is one column pass of
 * N1 rows over N2 K lanes, the plane's full width: the 1D batch's descriptor (il2d_col at
 * rn = N2 K), walker, race and dense scratch, unchanged. The LAST axis is N1 blocks of the 1D
 * lane-major batch at (N2, K): block r is the contiguous N2 K complexes of row r of the first
 * axis, so the child is the 1D batch's own plan, in role on a private store (its recipe rides
 * on this cell's row as rp_*), run per block in place on the plane after the first axis. The
 * passes commute (no twiddle between them), so the backward runs the same order with the
 * backward kernels. Order: the first axis takes the cell's class (natural, or the scrambled
 * comb), the row child is natural, as the 2D tier's row child is. Threaded: the first axis
 * through its raced column forms, the blocks over the workers, each on its own clone of the
 * row child (a plan's scratch is not reentrant), the whole execute one threaded verdict
 * (cmt=) raced with the column pass's arms. Both placements: the column pass moves src -> dst
 * in its first stage, the rows run in place on dst. Included after lm_batch_il.h (the 1D
 * batch and the race) and before the door. */
#ifndef VFFT_IL_LM_BATCH2D_IL_H
#define VFFT_IL_LM_BATCH2D_IL_H

/* the row blocks: worker t runs blocks [lo, hi) of the plane through its own row plan (the
 * primary for the caller, clone t - 1 for a worker), each block in place */
typedef struct { struct vfft_plan_s *h; double *z; vfft_dir_t dir; size_t lo, hi; int tid; } _lm2d_rows_arg_t;
static void _lm2d_rows_tramp(void *v)
{
    _lm2d_rows_arg_t *a = (_lm2d_rows_arg_t *)v;
    struct vfft_plan_s *h = a->h;
    struct vfft_plan_s *rp = a->tid > 0 ? h->lm_roww[a->tid - 1] : h->lm_row;
    const size_t blk = 2 * (size_t)h->N2 * h->K;
    size_t r;
    for (r = a->lo; r < a->hi; r++)
        _vfft_exec6((vfft_plan)rp, a->dir, a->z + r * blk, NULL, a->z + r * blk, NULL);
}
static void _lm2d_rows(struct vfft_plan_s *h, double *z, vfft_dir_t dir)
{
    _lm2d_rows_arg_t a[THREAD_POOL_MAX_DISPATCH];
    const size_t N1 = (size_t)h->N;
    int T = (h->nthreads > 1 && h->il2d_col.colmt && h->lm_roww_n > 0) ? thread_pool_workers_for(h->nthreads) : 1;
    int t;
    if (T > h->lm_roww_n + 1)
        T = h->lm_roww_n + 1;
    if ((size_t)T > N1)
        T = (int)N1;
    if (T < 2)
    {
        a[0].h = h; a[0].z = z; a[0].dir = dir; a[0].lo = 0; a[0].hi = N1; a[0].tid = 0;
        _lm2d_rows_tramp(&a[0]);
        return;
    }
    for (t = 0; t < T; t++)
    {
        a[t].h = h; a[t].z = z; a[t].dir = dir; a[t].tid = t;
        a[t].lo = N1 * (size_t)t / (size_t)T;
        a[t].hi = N1 * (size_t)(t + 1) / (size_t)T;
    }
    thread_pool_run(T, _lm2d_rows_tramp, a, sizeof a[0]);
}
/* the executor (the lean door binds it: never declining): the first axis over the plane, the
 * row blocks in place on the result */
static int _lm2d_exec(struct vfft_plan_s *h, vfft_dir_t dir, const double *zin, double *zout)
{
    _lm_cols(h, zin, zout, dir == VFFT_BACKWARD);
    _lm2d_rows(h, zout, dir);
    return 0;
}

/* the row child's clones, T - 1 of them from its store (they replay what the primary banked),
 * each structurally equivalent to the primary (_tc_clone_equiv's lane-major branch); any failure
 * drops the set and the rows stay serial */
static void _lm2d_clones_drop(struct vfft_plan_s *h)
{
    int t;
    for (t = 0; t < h->lm_roww_n; t++)
        if (h->lm_roww[t])
            vfft_destroy((vfft_plan)h->lm_roww[t]);
    free(h->lm_roww);
    h->lm_roww = NULL;
    h->lm_roww_n = 0;
}
static int _lm2d_clones_build(struct vfft_plan_s *h, const vfft_cfg_t *rc, int T)
{
    const int n = (T > THREAD_POOL_MAX_DISPATCH ? THREAD_POOL_MAX_DISPATCH : T) - 1;
    int t;
    if (n <= 0 || h->lm_roww || !h->lm_row)
        return 0;
    h->lm_roww = (struct vfft_plan_s **)calloc((size_t)n, sizeof *h->lm_roww);
    if (!h->lm_roww)
        return 0;
    for (t = 0; t < n; t++)
    {
        struct vfft_plan_s *c = (struct vfft_plan_s *)_vfft_create_cfg(rc);
        h->lm_roww[t] = c;
        h->lm_roww_n = t + 1;
        if (!c || !_tc_clone_equiv(h->lm_row, c))
        {
            _vfft_warn("lane-major 2D batch %dx%d x %zu: row clone %d %s; the rows stay serial",
                       h->N, h->N2, h->K, t, c ? "differs from the primary" : "failed to create");
            _lm2d_clones_drop(h);
            return 0;
        }
    }
    return 1;
}

/* ── THE DOOR: the lane-major 2D c2c batch's create ──────────────────────── */
static vfft_plan _vfft_create_lm2d_il(const vfft_cfg_t *cfg, struct vfft_wisdom_s *W, size_t K)
{
    const int T = _vfft_plan_threads(cfg), N1 = cfg->n[0], N2 = cfg->n[1];
    const int ord = vfft_policy_ord_rankn(cfg);   /* the cell's order class: the first axis carries it */
    const size_t lanes = (size_t)N2 * K;
    vw2_ilcol_key_t ck;
    vw2_key_t pk;
    vfft_ilcol_t col;
    vfft_cfg_t rc;
    char forms[64];
    int bwl = -1, btf = -1, bro = -1, bcmt = -1, bcmtt = -1, bblu = -1;
    struct vfft_plan_s *h;
    if (!vfft_policy_batch_lane_major(cfg))
    {
        _vfft_warn("vfft_create: the lane-major interleaved 2D batch serves C2C at howmany >= 2 with both extents "
                   ">= 2 (got %s %dx%d howmany=%zu) -- name batch=VFFT_BATCH_CONTIGUOUS for it",
                   _vfft_tname(cfg->transform), N1, N2, K);
        return NULL;
    }
    memset(&ck, 0, sizeof ck);
    ck.rank = 2; ck.n0 = N1; ck.n1 = N2; ck.n2 = (int)K; ck.ord = ord; ck.nthreads = T; ck.lm = 1;
    /* the Bluestein inner's chain provider for this create, as the 2D create sets it: the lanes are the width */
    _il2d_blu_ctx.W = W;
    _il2d_blu_ctx.cfg = cfg;
    _il2d_blu_ctx.N2 = (int)lanes;
    _il2d_blu_chain_hook = _il2d_blu_m_chain;
    memset(&col, 0, sizeof col);
    forms[0] = 0;
    if (!_il2d_col_build(W, cfg, &ck, N1, lanes, vfft_policy_rankn_axis_nat(2, 0, ord), &col, forms, sizeof forms,
                         &bwl, &btf, &bro, &bcmt, &bcmtt, &bblu))
    {
        _vfft_warn("vfft_create: lane-major interleaved 2D batch %dx%d x %zu: no column pass serves the first axis "
                   "(the column builder refused); unsupported", N1, N2, K);
        return NULL;
    }
    h = (struct vfft_plan_s *)calloc(1, sizeof *h);
    if (!h)
    {
        _il2d_col_free(&col);
        return NULL;
    }
    h->transform = VFFT_C2C;
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
    h->il2d_col.wl = 0;      /* the batch runs unbanded: the row axis is the child's, never fused here */
    h->il2d_col.cut = 0;
    h->il2d_col.tfuse = 0;
    h->il2d_col.wc = 0;
    /* THE ROW CHILD IN ROLE (owner, 2026-10-02): the 1D lane-major batch at (N2, K) on its own
     * store, seeded from this cell's row (rp_*); it never reads or writes a 1D row */
    vw2__ilcol_key(&ck, &pk);
    h->lm_rowS = vfft_child_store_for(&W->vw2, &pk, "rp_");
    memset(&rc, 0, sizeof rc);
    rc.transform = VFFT_C2C;
    rc.placement = VFFT_INPLACE;
    rc.rigor = cfg->rigor;
    rc.dims = 1;
    rc.n[0] = N2;
    rc.howmany = K;
    rc.order = VFI_ORDER_NATURAL;
    rc.layout = VFI_LAYOUT_IL;
    rc.batch_geom = VFFT_BATCH_LANE_MAJOR;
    rc.nthreads = 1;
    rc.wisdom = (vfft_wisdom *)h->lm_rowS;
    rc.recalibrate = cfg->recalibrate;
    h->lm_row = h->lm_rowS ? (struct vfft_plan_s *)_vfft_create_cfg(&rc) : NULL;
    if (!h->lm_row)
    {
        _vfft_warn("vfft_create: lane-major interleaved 2D batch %dx%d x %zu: the row child (the 1D batch at %d x %zu) "
                   "did not build; unsupported", N1, N2, K, N2, K);
        vfft_child_store_free(h->lm_rowS);
        _il2d_col_free(&h->il2d_col);
        free(h);
        return NULL;
    }
    if (T > 1)
    {   /* the clones before the race: its threaded arms run the blocks over them */
        rc.recalibrate = 0;   /* a clone replays what the primary banked */
        (void)_lm2d_clones_build(h, &rc, T);
    }
    if (T <= 1 && !h->il2d_col.blu && !h->il2d_col.tpc)
        (void)_lm_sscr_build(&h->il2d_col, N1, (int)lanes, 1);   /* the first axis's serial dense tiles' scratch */
    _lm_race(h, W, cfg, &ck, bcmt);
    if (!h->il2d_col.colmt)
        _lm2d_clones_drop(h);   /* a serial verdict runs the blocks on the primary alone */
    if (W && !W->vw2_off_2d)
    {   /* the row child's recipe onto the cell's row (and a turned prime pass's, as the 1D create banks it) */
        int changed = vfft_child_row_update(&W->vw2, &pk, "rp_", h->lm_rowS);
        if (h->il2d_col.tpcS)
            changed |= vfft_child_row_update(&W->vw2, &pk, "tpc_", h->il2d_col.tpcS);
        if (changed)
            _vw2_persist(W, cfg);
    }
    (void)bcmtt; (void)bwl; (void)btf; (void)bro; (void)bblu;
    return (vfft_plan)h;   /* the door binds _lm2d_exec (il_execute.h) */
}
#endif /* VFFT_IL_LM_BATCH2D_IL_H */
