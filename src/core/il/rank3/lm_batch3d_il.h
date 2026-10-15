/* lm_batch3d_il.h -- THE LANE-MAJOR 3D BATCH (docs/roadmap/lane_major_batch_design.md, step 6,
 * 2026-10-15): K >= 2 volumes of N1 x N2 x N3 side by side, element (i, j, k) of transform t
 * at z[2 ((i N2 N3 + j N3 + k) K + t)]: the 2D batch's door one rank up. The FIRST axis is one
 * column pass of N1 rows over N2 N3 K lanes, the volume's full width: the 1D batch's
 * descriptor (il2d_col at rn = N2 N3 K), walker, race and dense scratch, unchanged. The rest
 * is N1 blocks of the lane-major 2D batch at (N2, N3, K): block i is the contiguous N2 N3 K
 * complexes of row i of the first axis, so the child is the 2D batch's own plan
 * (lm_batch2d_il.h: its own column pass over N3 K lanes and its own 1D row child), in role on
 * a private store (its recipe, its row child's inside it, rides on this cell's row as rp_*),
 * run per block in place after the first axis through the 2D batch's block walker and
 * executor (_lm2d_rows takes the block from N3, the door binds _lm2d_exec). The passes
 * commute, so the backward runs the same order with the backward kernels. Order: the first
 * axis takes the cell's class, the child is natural. Threaded: the first axis through its
 * raced column forms, the blocks over the workers, each on its own clone of the 2D child
 * (serial inside: its thread count is one), the whole execute one threaded verdict (cmt=)
 * raced with the column pass's arms. Both placements: the column pass moves src -> dst in
 * its first stage, the blocks run in place on dst. The row: the 3D cell at q = K on the
 * lay=lm axis. C2C only. Included after lm_batch2d_il.h and before the door. */
#ifndef VFFT_IL_LM_BATCH3D_IL_H
#define VFFT_IL_LM_BATCH3D_IL_H

/* ── THE DOOR: the lane-major 3D c2c batch's create ──────────────────────── */
static vfft_plan _vfft_create_lm3d_il(const vfft_cfg_t *cfg, struct vfft_wisdom_s *W, size_t K)
{
    const int T = _vfft_plan_threads(cfg), N1 = cfg->n[0], N2 = cfg->n[1], N3 = cfg->n[2];
    const int ord = vfft_policy_ord_rankn(cfg);   /* the cell's order class: the first axis carries it */
    const size_t lanes = (size_t)N2 * (size_t)N3 * K;
    vw2_ilcol_key_t ck;
    vw2_key_t pk;
    vfft_ilcol_t col;
    vfft_cfg_t rc;
    char forms[64];
    int bwl = -1, btf = -1, bro = -1, bcmt = -1, bcmtt = -1, bblu = -1;
    struct vfft_plan_s *h;
    if (!vfft_policy_batch_lane_major(cfg))
    {
        _vfft_warn("vfft_create: the lane-major interleaved 3D batch serves C2C at howmany >= 2 with every extent "
                   ">= 2 (got %s %dx%dx%d howmany=%zu); unsupported",
                   _vfft_tname(cfg->transform), N1, N2, N3, K);
        return NULL;
    }
    memset(&ck, 0, sizeof ck);
    ck.rank = 3; ck.n0 = N1; ck.n1 = N2; ck.n2 = N3; ck.q = (int)K; ck.ord = ord; ck.nthreads = T; ck.lm = 1;
    /* the Bluestein inner's chain provider for this create, as the 2D create sets it: the lanes are the width */
    _il2d_blu_ctx.W = W;
    _il2d_blu_ctx.cfg = cfg;
    _il2d_blu_ctx.N2 = (int)lanes;
    _il2d_blu_chain_hook = _il2d_blu_m_chain;
    memset(&col, 0, sizeof col);
    forms[0] = 0;
    if (!_il2d_col_build(W, cfg, &ck, N1, lanes, vfft_policy_lm_axis_nat(ord), &col, forms, sizeof forms,
                         &bwl, &btf, &bro, &bcmt, &bcmtt, &bblu))
    {
        _vfft_warn("vfft_create: lane-major interleaved 3D batch %dx%dx%d x %zu: no column pass serves the first axis "
                   "(the column builder refused); unsupported", N1, N2, N3, K);
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
    h->N3 = N3;
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
    /* THE CHILD IN ROLE (owner, 2026-10-02): the lane-major 2D batch at (N2, N3, K) on its own
     * store, seeded from this cell's row (rp_*); it never reads or writes a 2D row */
    vw2__ilcol_key(&ck, &pk);
    h->lm_rowS = vfft_child_store_for(&W->vw2, &pk, "rp_");
    memset(&rc, 0, sizeof rc);
    rc.transform = VFFT_C2C;
    rc.placement = VFFT_INPLACE;
    rc.rigor = cfg->rigor;
    rc.dims = 2;
    rc.n[0] = N2;
    rc.n[1] = N3;
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
        _vfft_warn("vfft_create: lane-major interleaved 3D batch %dx%dx%d x %zu: the child (the lane-major 2D batch at "
                   "%dx%d x %zu) did not build; unsupported", N1, N2, N3, K, N2, N3, K);
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
    {   /* the child's recipe onto the cell's row (and a turned prime pass's, as the 1D create banks it) */
        int changed = vfft_child_row_update(&W->vw2, &pk, "rp_", h->lm_rowS);
        if (h->il2d_col.tpcS)
            changed |= vfft_child_row_update(&W->vw2, &pk, "tpc_", h->il2d_col.tpcS);
        if (changed)
            _vw2_persist(W, cfg);
    }
    (void)bcmtt; (void)bwl; (void)btf; (void)bro; (void)bblu;
    return (vfft_plan)h;   /* the door binds _lm2d_exec (il_execute.h): the 2D batch's executor, the block N2 N3 K */
}
#endif /* VFFT_IL_LM_BATCH3D_IL_H */
