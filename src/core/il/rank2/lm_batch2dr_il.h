/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Tuğbars Heptaşkın */
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
 * The blocks go in GROUPS (_lm2dr_rows_grp): policy's count of blocks at a time through one dense scratch,
 * the group kernels around the child's chain, so a batch of few lanes still fills the registers and a short
 * row's setup is paid once a group; threaded, every worker groups its own blocks on its own scratch. Where
 * policy names no group (a long row of four lanes or more, a wide batch) the group is one block, through
 * the same walk: the child's door would cost the door, the child's executor and its strip logic once a
 * row. Only where the child runs a Rader, Bluestein or turned chain (the chains the group kernels do not
 * wrap) do the blocks go one at a time through the child's door.
 * Threaded: the column pass through its raced forms, the blocks over the workers on clones of the
 * row child, one threaded verdict (cmt=) raced with the column pass's arms (the arms run the whole
 * executor). The row: t=r2c n=N1xN2 q=K on the lay=lm axis, direction-shared as the 1D real row is,
 * the first axis's column tokens on it. The 3D real batch (rank3/lm_batch3dr_il.h) runs through the
 * same block walker and executors: its blocks are the planes of N2 (N3 + 2) K doubles, its child this
 * 2D real batch. Included after lm_batch2d_il.h and before the door. */
#ifndef VFFT_IL_LM_BATCH2DR_IL_H
#define VFFT_IL_LM_BATCH2DR_IL_H

/* THE ROW BLOCKS IN GROUPS (the 2D batch whose child is the 1D real batch): blocks [lo, hi), lm_G at a
 * time, through one dense scratch -- the group kernels (rank1/lm_batch_il.h) around the row child's own
 * chain, run once over the group's lanes. A group consumes its blocks whole before it writes one, so the
 * in-place plane (the bins over the reals, block by block) needs nothing more. The group is policy's
 * (vfft_policy_lm_row_group; 1 = one block); the chain, its permutation and its twiddles are the child's and are only
 * read, so every worker runs them on a scratch of its own. The bins are the block-at-a-time form's within
 * the last bit (bit for bit at an even lane count). The body is compiled twice: for a group of one, with
 * G the constant 1, the group kernels lose their loop over the blocks, which a group of one has no use
 * for (at two and four lanes a row step is one or two vector steps, and that loop cost a fifth of the row
 * pass, 2026-11-19); for a group of two or more, with G the plan's. */
static inline __attribute__((always_inline)) void _lm2dr_rows_grp_g(struct vfft_plan_s *h, const vfft_real *in, vfft_real *out,
                                                                    size_t sblk, size_t dblk, int c2r, size_t lo, size_t hi,
                                                                    vfft_real *scr, const size_t G)
{
    const struct vfft_plan_s *rp = h->lm_row;
    const vfft_ilcol_t *c = &rp->il2d_col;
    const size_t K = h->K, M = (size_t)rp->lm_M;
    size_t r;
    for (r = lo; r < hi; r += G)
    {
        const size_t gn = hi - r < G ? hi - r : G, w = gn * K;
        if (!c2r)
        {
            _lmr_pack_grp(in + r * sblk, sblk, gn, K, scr, M);
            _il2d_col_pass_range(scr, scr, (int)M, w, 0, w, c->nst, c->R, c->L, c->f, c->tf, 0);
            _lmr_fold_grp(scr, gn, K, out + r * dblk, dblk, M, rp->lm_perm, rp->lm_tw);
        }
        else
        {
            _lmc_unfold_grp(in + r * sblk, sblk, gn, K, scr, M, rp->lm_perm, rp->lm_tw);
            _il2d_col_pass_range(scr, scr, (int)M, w, 0, w, c->nst, c->R, c->L, c->b, c->tb, 1);
            _lmc_unpack_grp(scr, gn, K, out + r * dblk, dblk, M);
        }
    }
}
static void _lm2dr_rows_grp(struct vfft_plan_s *h, const vfft_real *in, vfft_real *out, size_t sblk, size_t dblk, int c2r,
                            size_t lo, size_t hi, vfft_real *scr)
{
    if (h->lm_G == 1)
        _lm2dr_rows_grp_g(h, in, out, sblk, dblk, c2r, lo, hi, scr, 1);
    else
        _lm2dr_rows_grp_g(h, in, out, sblk, dblk, c2r, lo, hi, scr, (size_t)h->lm_G);
}
/* the row blocks: worker t runs blocks [lo, hi) -- in groups on its own scratch where the plan has a group
 * (lm_G >= 1), else (a Rader, Bluestein or turned chain) through its own row plan -- block r of the input at
 * in + r sblk, of the output at out + r dblk (the same base and block in place) */
typedef struct { struct vfft_plan_s *h; const vfft_real *in; vfft_real *out; size_t sblk, dblk, lo, hi; vfft_dir_t dir; int tid; } _lm2dr_rows_arg_t;
static void _lm2dr_rows_tramp(void *v)
{
    _lm2dr_rows_arg_t *a = (_lm2dr_rows_arg_t *)v;
    struct vfft_plan_s *rp = a->tid > 0 ? a->h->lm_roww[a->tid - 1] : a->h->lm_row;
    size_t r;
    if (a->h->lm_G > 0)
    {
        _lm2dr_rows_grp(a->h, a->in, a->out, a->sblk, a->dblk, a->dir == VFFT_BACKWARD, a->lo, a->hi,
                        a->h->lm_gscr + (size_t)a->tid * a->h->lm_gscr_sz);
        return;
    }
    for (r = a->lo; r < a->hi; r++)
        _vfft_exec6((vfft_plan)rp, a->dir, (vfft_real *)a->in + r * a->sblk, NULL, a->out + r * a->dblk, NULL);
}
static void _lm2dr_rows(struct vfft_plan_s *h, const vfft_real *in, vfft_real *out, size_t sblk, size_t dblk, vfft_dir_t dir)
{
    _lm2dr_rows_arg_t a[THREAD_POOL_MAX_DISPATCH];
    const size_t N1 = (size_t)h->N;
    int T = (h->nthreads > 1 && h->il2d_col.colmt && h->lm_roww_n > 0) ? thread_pool_workers_for(h->nthreads) : 1;
    int t;
    if (T > h->lm_roww_n + 1)
        T = h->lm_roww_n + 1;
    if (h->lm_G > 0 && T > h->lm_gscr_n)
        T = h->lm_gscr_n;   /* a scratch a worker */
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
/* a block's last extent L and the rows of it a block holds, M: rank 2 (the 1D child) L = N2, M = 1; rank 3 L = N3,
 * M = N2; rank 4 to 7 L the last extent, M the extents between the first and the last */
static void _lm2dr_LM(const struct vfft_plan_s *h, size_t *L, size_t *M)
{
    const int e[6] = { h->N2, h->N3, h->N4, h->N5, h->N6, h->N7 };
    int r = 1, i;
    while (r < 6 && e[r] > 0)
        r++;
    *M = 1;
    for (i = 0; i < r - 1; i++)
        *M *= (size_t)e[i];
    *L = (size_t)e[r - 1];
}
/* the executors (bound at create: a direction each, the other refused by the door). The blocks are the
 * child's: at rank 2 the 1D real batch's rows (L = N2 reals a lane, M = 1), at rank 3 (the 3D real batch,
 * rank3/lm_batch3dr_il.h) the 2D real batch's planes (L = N3, M = N2 rows each) -- a real block M L K,
 * its spectrum block M (L + 2) K (2 (L / 2 + 1) = L + 2 at an even L), the padded block the same */
static int _lm2dr_exec_r2c(struct vfft_plan_s *h, vfft_dir_t dir, const vfft_real *xin, vfft_real *zout)
{
    const size_t K = h->K;
    size_t L, M;
    const int ip = h->placement == VFFT_INPLACE;
    (void)dir;
    _lm2dr_LM(h, &L, &M);
    _lm2dr_rows(h, xin, zout, M * (ip ? L + 2 : L) * K, M * (L + 2) * K, VFFT_FORWARD);
    _lm_cols(h, zout, zout, 0);   /* the first axis over the spectrum, in place */
    return 0;
}
static int _lm2dr_exec_c2r(struct vfft_plan_s *h, vfft_dir_t dir, const vfft_real *zin, vfft_real *xout)
{
    const size_t K = h->K;
    size_t L, M;
    const int ip = h->placement == VFFT_INPLACE;
    (void)dir;
    _lm2dr_LM(h, &L, &M);
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
    int bwl = -1, btf = -1, bro = -1, bcmt = -1, bcmtt = -1, bblu = -1, oom = 0;
    struct vfft_plan_s *h;
    if (!vfft_policy_batch_lane_major(cfg))
    {
        _vfft_warn("vfft_create: the lane-major interleaved 2D real batch serves R2C and C2R at howmany >= 2, N1 >= 2 "
                   "and an even N2 >= 4 (got %s %dx%d howmany=%zu) -- an odd N2 has no real column engine",
                   _vfft_tname(cfg->transform), N1, N2, K);
        return NULL;
    }
    memset(&ck, 0, sizeof ck);
    ck.rank = 2; ck.n0 = N1; ck.n1 = (size_t)N2; ck.n2 = K; ck.ord = VW2_ORD_NAT; ck.real = 1; ck.nthreads = T; ck.lm = 1;
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
    {   /* the child's own reason first (an out of memory among them), then whose child it was */
        char why[VFFT_ERR_CAP];
        snprintf(why, sizeof why, "%s", h->lm_rowS ? _vfft_err_buf : "vfft_create: out of memory (the row child's store)");
        _vfft_warn("%s [the row child of the lane-major interleaved 2D %s batch %dx%d x %zu]", why,
                   _vfft_tname(cfg->transform), N1, N2, K);
        vfft_child_store_free(h->lm_rowS);
        _il2d_col_free(&h->il2d_col);
        free(h);
        return NULL;
    }
    if (c2r && cfg->placement != VFFT_INPLACE)
    {   /* the scratch the backward column pass lands in: the spectrum's size */
        h->lm_W = (vfft_real *)vfft_aligned_alloc((2 * (size_t)N1 * lanes + 8) * sizeof(vfft_real));
        if (!h->lm_W)
        {   /* OUT OF MEMORY: the spectrum-sized scratch -- refused with that cause */
            _vfft_warn("vfft_create: lane-major interleaved 2D %s batch %dx%d x %zu: out of memory (its backward "
                       "column scratch needs %.1f MB)", _vfft_tname(cfg->transform), N1, N2, K,
                       (2.0 * (double)N1 * (double)lanes + 8.0) * sizeof(vfft_real) / 1048576.0);
            vfft_destroy((vfft_plan)h->lm_row);
            vfft_child_store_free(h->lm_rowS);
            _il2d_col_free(&h->il2d_col);
            free(h);
            return NULL;
        }
    }
    {   /* the row pass in groups wherever the row child runs the plain chain (the chain the group kernels
         * wrap: no Rader stage, no Bluestein or turned axis): policy's group, one block where policy names
         * none; built before the race, whose arms run this plan's executor */
        const vfft_ilcol_t *rcol = &h->lm_row->il2d_col;
        if (rcol->nst > 0 && !rcol->nat && !rcol->blu && !rcol->tpc && rcol->wl <= 0 &&
            (c2r ? rcol->b[0] : rcol->f[0]) != NULL)
        {   /* a scratch a worker, each a whole number of cache lines */
            const int G = vfft_policy_lm_row_group(N1, N2, K, vfft_cpu_l1d_bytes());   /* >= 1 */
            const size_t sz = ((size_t)N2 * (size_t)G * K + 8 + 7) & ~(size_t)7;
            const int ns = T > 1 ? T : 1;
            h->lm_gscr = (vfft_real *)vfft_aligned_alloc((size_t)ns * sz * sizeof(vfft_real));
            if (!h->lm_gscr)
            {   /* OUT OF MEMORY: the row pass's scratch -- refused with that cause (not retained yet: its own
                 * release is skipped, as after the race below) */
                _vfft_warn("vfft_create: lane-major interleaved 2D %s batch %dx%d x %zu: out of memory (its row "
                           "scratch needs %.1f MB)", _vfft_tname(cfg->transform), N1, N2, K,
                           (double)ns * (double)sz * sizeof(vfft_real) / 1048576.0);
                h->nthreads = 1;
                vfft_destroy((vfft_plan)h);
                return NULL;
            }
            h->lm_G = G;
            h->lm_gscr_sz = sz;
            h->lm_gscr_n = ns;
        }
    }
    if (T > 1)
    {
        rc.recalibrate = 0;
        (void)_lm2d_clones_build(h, &rc, T);
    }
    if (T <= 1 && !h->il2d_col.blu && !h->il2d_col.tpc)
        (void)_lm_sscr_build(&h->il2d_col, N1, lanes, 1);
    if (_lm_race(h, W, cfg, &ck, bcmt))
    {   /* OUT OF MEMORY in the race (its reason recorded). Not retained yet (the front door retains the plan it
         * returns): its own release is skipped */
        h->nthreads = 1;
        vfft_destroy((vfft_plan)h);
        return NULL;
    }
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
