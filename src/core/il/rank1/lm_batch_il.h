/* lm_batch_il.h — THE LANE-MAJOR 1D BATCH on the column pass
 * (docs/roadmap/lane_major_batch_design.md, 2026-10-14).
 *
 * K >= 2 transforms of length N in the lane-major geometry (config.batch_geom =
 * VFFT_BATCH_LANE_MAJOR; include/vfft.h): element e of transform t at
 * z[2 (e K + t)]. The plane is N rows of K complex at pitch K -- the 2D tier's
 * column plane with N1 = N, N2 = K and no row pass -- so the plan owns one
 * column-axis descriptor (il2d_col.h), built by the column builder at (N, K) in
 * THIS plan's role on THIS plan's own row (the 1D cell at q = K on the
 * lane-major layout axis, lay=lm: the transform-contiguous wrapper's q = K row
 * is another cell), and run by the column pass: the serial pass with its lane
 * tile (sw=), the threaded forms (strips over lanes, blocks over digits) under
 * the cell's raced verdict at the plan's T (cmt= mtarm= msw=). Everything the
 * 2D and 3D column axes have, the batch inherits: the chain race on the serving
 * walk, the Rader stage with its inner and width race, the forms race, the
 * natural leaf redirection, the column Bluestein and the turned prime pass at
 * a prime N. Both placements (the pass runs in place on the plane), both order
 * classes (NATURAL by the leaf redirection; SCRAMBLED = the chain's comb per
 * lane, the column class's contract). c2c; the real batches are the next
 * pieces of the design.
 *
 * The geometries never race each other here: a lane-major request is served
 * lane-major or refused with its reason (the owner's rule, 2026-10-14); the
 * comparison of layouts is the planning model's max-performance mode.
 *
 * Included after the 2D tier (il2d_tier.h: the column builder, the passes, the
 * threaded forms, the races' scaffolding) and before il_create.h (the door). */
#ifndef VFFT_IL_LM_BATCH_IL_H
#define VFFT_IL_LM_BATCH_IL_H

/* the plan's column pass, one direction: the threaded forms under the cell's
 * verdict when the pool is live, the serial pass otherwise (the natural leaf
 * redirection, the Bluestein and turned axes and the banded walk through the
 * column-only serial execute; the plain chain pass with its lane tile) */
static void _lm_cols(struct vfft_plan_s *h, const double *src, double *dst, int reverse)
{
    vfft_ilcol_t *c = &h->il2d_col;
    if (h->nthreads > 1 && c->colmt &&
        _il2d_cols_mt_desc(c, h->N, src, dst, reverse, h->nthreads, c->nat ? c->natstage : NULL, -1))
    {
        _vfft_il2d_col_mt_count++;   /* the engagement (vfft.h): the caller's duty, as the 2D tiers count theirs */
        return;
    }
    if (c->nat || c->blu || c->wl > 0)
    {
        _il2d_col_exec_st(c, src, dst, reverse, c->natstage);
        return;
    }
    _il2d_col_pass(src, dst, h->N, c->rn, c->wc > 0 ? (size_t)c->wc : 0, c->nst, c->R, c->L,
                   reverse ? c->b : c->f, reverse ? c->tb : c->tf, reverse);
}
/* the bound executor (the lean door: it never declines) */
static int _lm_exec(struct vfft_plan_s *h, vfft_dir_t dir, const double *zin, double *zout)
{
    _lm_cols(h, zin, zout, dir == VFFT_BACKWARD);
    return 0;
}

/* ── THE CELL'S OWN VERDICTS BEYOND THE CHAIN: the serial lane tile and the
 * threaded form, replayed from the row at this T or raced on a plane of the
 * plan's own placement and banked (sw= on the row; cmt= / cmtt= through the
 * chain bank, mtarm= / msw= beside them). Every arm runs the plan's own pass
 * (_lm_cols) with the arm's knobs on the descriptor; a threaded arm whose pass
 * does not engage on this cell is out (the engagement counter says). ── */
typedef struct { struct vfft_plan_s *h; double *z, *zo; int colmt, natarm, msw, wc, ok; char name[24]; } _lm_arm_t;
static void _lm_arm_run(void *v)
{
    _lm_arm_t *a = (_lm_arm_t *)v;
    vfft_ilcol_t *c = &a->h->il2d_col;
    c->colmt = a->colmt;
    c->natarm = a->natarm;
    c->msw = a->msw;
    c->wc = a->wc;
    _lm_cols(a->h, a->z, a->zo, 0);
}
static void _lm_race(struct vfft_plan_s *h, struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                     const vw2_ilcol_key_t *key, int bcmt)
{
    vfft_ilcol_t *c = &h->il2d_col;
    const int N = h->N, K = (int)h->K, T = h->nthreads, ip = h->placement == VFFT_INPLACE;
    const int workers = T > 1 ? thread_pool_workers_for(T) : 1;
    const int usable = W && !W->vw2_off_2d && !cfg->recalibrate;
    const int plain = !c->nat && !c->blu && c->wl <= 0;   /* the lane tile is the plain chain pass's */
    const int strips_ok = !c->blu && !c->tpc && (!c->nat || c->nst >= 2);
    const size_t PN = 2 * (size_t)N * (size_t)K;
    _lm_arm_t ac[14];
    vfft_race_arm_t arms[14];
    double ns[14], *z, *zo, *seed = NULL;
    _il2d_ip_reset_t rs;
    int lad[6], nl, na = 0, i, best = 0, bsw = -1, need_sw, need_mt;
    size_t q;
    c->wc = 0; c->colmt = 0; c->natarm = 0; c->msw = 0;
    bsw = (plain && usable) ? vw2_ilcol_tok_geti(&W->vw2, key, "sw", -1) : -1;
    need_sw = plain && bsw < 0;
    if (bsw >= 0)
        c->wc = bsw;
    need_mt = workers >= 2 && !(usable && bcmt >= 0);
    if (workers >= 2 && !need_mt)
    {   /* the threaded verdict from the row at this T */
        c->colmt = bcmt > 0;
        c->natarm = c->colmt ? vw2_ilcol_tok_geti(&W->vw2, key, "mtarm", 0) : 0;
        c->msw = c->colmt ? vw2_ilcol_tok_geti(&W->vw2, key, "msw", 0) : 0;
        if (c->colmt && c->natarm == 1 && c->nat)
            (void)_il2d_nat_sscr_build(c, N, K, workers);
    }
    if (!need_sw && !need_mt)
        return;
    z = (double *)vfft_aligned_alloc((PN + 8) * sizeof(double));
    zo = ip ? z : (double *)vfft_aligned_alloc((PN + 8) * sizeof(double));
    if (ip) seed = (double *)vfft_aligned_alloc((PN + 8) * sizeof(double));
    if (!z || !zo || (ip && !seed))
    {
        vfft_aligned_free(z); if (!ip) vfft_aligned_free(zo); vfft_aligned_free(seed);
        return;
    }
    for (q = 0; q < PN + 8; q++)
        z[q] = 1.0 + 1e-6 * (double)(q & 1023);
    if (ip) { memcpy(seed, z, (PN + 8) * sizeof(double)); rs.p = z; rs.seed = seed; rs.n = PN + 8; }
#define _LM_ARM(NAME_, MT_, SHAPE_, SW_, WC_) do { \
        ac[na].h = h; ac[na].z = z; ac[na].zo = zo; ac[na].colmt = (MT_); ac[na].natarm = (SHAPE_); \
        ac[na].msw = (SW_); ac[na].wc = (WC_); ac[na].ok = 1; \
        snprintf(ac[na].name, sizeof ac[na].name, "%s", NAME_); \
        arms[na].name = ac[na].name; arms[na].run = _lm_arm_run; arms[na].ctx = &ac[na]; na++; } while (0)
    _LM_ARM("serial", 0, 0, 0, bsw >= 0 ? bsw : 0);
    if (need_sw)
    {   /* the serial lane tiles: the ladder's widths that divide K and keep a tile in L2 */
        nl = _il2d_sw_ladder(N, K, 1, lad, 6);
        for (i = 0; i < nl && na < 14; i++)
        {
            char nm[24];
            snprintf(nm, sizeof nm, "sw%d", lad[i]);
            _LM_ARM(nm, 0, 0, 0, lad[i]);
        }
    }
    if (need_mt)
    {
        _LM_ARM("threaded", 1, 0, 0, 0);
        if (strips_ok)
        {
            if (c->nat)
                (void)_il2d_nat_sscr_build(c, N, K, workers);
            _LM_ARM("strips", 1, 1, 0, 0);
            nl = _il2d_sw_ladder(N, K, T, lad, 6);
            for (i = 0; i < nl && na < 14; i++)
            {
                char nm[24];
                snprintf(nm, sizeof nm, "strips/sw%d", lad[i]);
                _LM_ARM(nm, 1, 1, lad[i], 0);
            }
        }
        for (i = 1; i < na; i++)
            if (ac[i].colmt)
            {   /* a threaded arm that does not engage on this cell is out */
                const long e0 = _vfft_il2d_col_mt_count;
                if (ip) memcpy(z, seed, (PN + 8) * sizeof(double));
                _lm_arm_run(&ac[i]);
                ac[i].ok = _vfft_il2d_col_mt_count != e0;
            }
    }
#undef _LM_ARM
    {
        vfft_race_proto_t proto = need_mt
            ? (vfft_race_proto_t){ 3, 1, VFFT_RACE_MIN, 0, 2, NULL, NULL, 0 }   /* threaded arms: two untimed passes first, never paused */
            : (vfft_race_proto_t){ 3, 1, VFFT_RACE_MIN, 0, 0, NULL, NULL, 1 };  /* single-thread arms: paced */
        if (ip) { proto.reset = _il2d_ip_reset; proto.reset_ctx = &rs; }
        if (ip) memcpy(z, seed, (PN + 8) * sizeof(double));
        proto.reps = _il2d_race_reps(_lm_arm_run, &ac[0], ip ? (double)N : 0.0);
        _vfft_create_race_count++;
        vfft_race_run(&proto, arms, na, ns);
    }
    for (i = 1; i < na; i++)
        if (ac[i].ok && ns[i] < ns[best])
            best = i;
    c->wc = ac[best].colmt ? (bsw >= 0 ? bsw : 0) : ac[best].wc;
    if (ac[best].colmt && need_sw)
    {   /* the serial tile stays the best serial arm's (the threaded arm may fall back to it) */
        int bs = 0;
        for (i = 1; i < na; i++)
            if (!ac[i].colmt && ns[i] < ns[bs]) bs = i;
        c->wc = ac[bs].wc;
    }
    c->colmt = ac[best].colmt;
    c->natarm = ac[best].natarm;
    c->msw = ac[best].msw;
    if (c->natarm != 1 || !c->nat || !c->colmt)
        _il2d_nat_sscr_free(c);   /* the dense scratch serves the strips alone */
    if (getenv("VFFT_IL2D_LOG"))
    {
        fprintf(stderr, "[lm] %d x %d T=%d %s:", N, K, T, ip ? "ip" : "oop");
        for (i = 0; i < na; i++) fprintf(stderr, " %s=%.0f%s", ac[i].name, ns[i], ac[i].ok ? "" : "(x)");
        fprintf(stderr, " -> %s (sw=%d cmt=%d mtarm=%d msw=%d)\n", ac[best].name, c->wc, c->colmt, c->natarm, c->msw);
    }
    if (W && !W->vw2_off_2d)
    {   /* the verdicts on the row (the chain bank made it) */
        int changed = 0;
        if (need_sw) changed |= vw2_ilcol_tok_seti(&W->vw2, key, "sw", c->wc) == VW2_OK;
        if (need_mt)
        {
            changed |= vw2_ilcol_chain_bank(&W->vw2, key, c->R, c->nst, -1, -1, -1, c->colmt, T, c->blu, 0.0) >= 0;
            changed |= vw2_ilcol_tok_seti(&W->vw2, key, "mtarm", c->natarm) == VW2_OK;
            changed |= vw2_ilcol_tok_seti(&W->vw2, key, "msw", c->msw) == VW2_OK;
        }
        if (changed) _vw2_persist(W, cfg);
    }
    vfft_aligned_free(z); if (!ip) vfft_aligned_free(zo); vfft_aligned_free(seed);
}

/* ── THE DOOR: the lane-major 1D batch's create ─────────────────────────── */
static vfft_plan _vfft_create_lm_il(const vfft_config_t *cfg, struct vfft_wisdom_s *W, int N, size_t K)
{
    const int T = _vfft_plan_threads(cfg);
    const int ip = cfg->placement == VFFT_INPLACE;
    const int ord = vfft_policy_ord_k1(cfg, N, ip);
    const int nat = ord == VW2_ORD_NAT;
    vw2_ilcol_key_t ck;
    vfft_ilcol_t col;
    char forms[64];
    int bwl = -1, btf = -1, bro = -1, bcmt = -1, bcmtt = -1, bblu = -1;
    struct vfft_plan_s *h;
    if (!vfft_policy_batch_lane_major(cfg))
    {
        _vfft_warn("vfft_create: the lane-major interleaved batch serves 1D C2C at howmany >= 2 and N >= 2 "
                   "(got %s N=%d howmany=%zu); the real batches are not served lane-major yet -- name "
                   "batch_geom=VFFT_BATCH_TRANSFORM_CONTIGUOUS for them", _vfft_tname(cfg->transform), N, K);
        return NULL;
    }
    memset(&ck, 0, sizeof ck);
    ck.rank = 1; ck.n0 = N; ck.n1 = (int)K; ck.ord = ord; ck.nthreads = T; ck.lm = 1;
    /* the Bluestein inner's chain provider for this create, as the 2D create sets it */
    _il2d_blu_ctx.W = W;
    _il2d_blu_ctx.cfg = cfg;
    _il2d_blu_ctx.N2 = (int)K;
    _il2d_blu_chain_hook = _il2d_blu_m_chain;
    memset(&col, 0, sizeof col);
    forms[0] = 0;
    if (!_il2d_col_build(W, cfg, &ck, N, K, nat, &col, forms, sizeof forms,
                         &bwl, &btf, &bro, &bcmt, &bcmtt, &bblu))
    {
        _vfft_warn("vfft_create: lane-major interleaved C2C batch N=%d x K=%zu: no column pass serves N "
                   "(the column builder refused); unsupported", N, K);
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
    h->layout = (int)VFFT_LAYOUT_INTERLEAVED;
    h->N = N;
    h->N2 = 0;
    h->K = K;
    h->nthreads = T;
    h->lm = 1;
    h->il2d_col = col;
    h->il2d_col.N = N;
    h->il2d_col.rn = K;
    h->il2d_col.wl = 0;      /* the banded walk is the 2D plan's fused-row form; the batch runs unbanded */
    h->il2d_col.cut = 0;
    h->il2d_col.tfuse = 0;
    h->il2d_col.wc = 0;
    _lm_race(h, W, cfg, &ck, bcmt);
    if (W && !W->vw2_off_2d && h->il2d_col.tpcS)
    {   /* the turned prime pass's child recipe onto the cell's row (the 2D create's last act;
         * without it the replay re-races the 1D prime cell and lands elsewhere) */
        vw2_key_t pk;
        vw2__ilcol_key(&ck, &pk);
        if (vfft_child_row_update(&W->vw2, &pk, "tpc_", h->il2d_col.tpcS))
            _vw2_persist(W, cfg);
    }
    (void)bcmtt; (void)bwl; (void)btf; (void)bro; (void)bblu;
    return (vfft_plan)h;
}
#endif /* VFFT_IL_LM_BATCH_IL_H */
