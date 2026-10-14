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
    vfft_ilcol_t *c = &h->il2d_col;   /* c->N is the chain's length: N for the c2c batch, N / 2 for the real one */
    if (h->nthreads > 1 && c->colmt &&
        _il2d_cols_mt_desc(c, c->N, src, dst, reverse, h->nthreads, c->nat ? c->natstage : NULL, -1))
    {
        _vfft_il2d_col_mt_count++;   /* the engagement (vfft.h): the caller's duty, as the 2D tiers count theirs */
        return;
    }
    if (c->nat || c->blu || c->wl > 0)
    {
        _il2d_col_exec_st(c, src, dst, reverse, c->natstage);
        return;
    }
    _il2d_col_pass(src, dst, c->N, c->rn, c->wc > 0 ? (size_t)c->wc : 0, c->nst, c->R, c->L,
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
    const int N = c->N, K = (int)h->K, T = h->nthreads, ip = h->placement == VFFT_INPLACE && h->transform == VFFT_C2C;
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

/* ── THE LANE-MAJOR REAL BATCH, r2c at an even N (the design's step 2) ─────
 * x[e K + t] real (N rows of K) -> z[2 (f K + t)], f < N / 2 + 1, three passes, the
 * real-axis form's own (il2d_real_axis.h) with its rows removed:
 *   PACK    W[m][k] = x[2 m][k] + i x[2 m + 1][k], m < M = N / 2 (a copy: the two reals
 *           of a packed sample sit K apart in lane-major data);
 *   COLUMNS the c2c column chain at M down every lane of W, in place, the scrambled
 *           class (the chain the column builder serves at (M, K), raced in this plan's
 *           role on this plan's own row; a prime M takes the column Bluestein);
 *   FOLD    per lane, the column spectrum read through the chain's permutation:
 *           A[0] = Re Z[0] + Im Z[0], A[M] = Re Z[0] - Im Z[0], and for k = 1..M/2
 *           E = (Z[k] + conj Z[M-k]) / 2, O = -i (Z[k] - conj Z[M-k]) / 2,
 *           A[k] = E + w^k O, A[M-k] = conj(E - w^k O), w = e^{-2 pi i / N}:
 *           the M + 1 rows of K bins, natural, straight into the caller's spectrum.
 * The chain threads through the column MT forms under the cell's verdict; the pack and
 * the fold split by lane ranges across the workers whenever that form serves. Out of
 * place (the in-place plane is the design's step 5). The c2r twin is its own piece. */
static void _lmr_pack(const double *x, double *W, size_t M, size_t K, size_t k0, size_t k1)
{
    size_t m, n;
    for (m = 0; m < M; m++)
    {
        const double *a = x + 2 * m * K + k0, *b = a + K;
        double *w = W + 2 * (m * K + k0);
        const size_t w1 = k1 - k0;
        n = 0;
#if defined(__AVX2__)
        for (; n + 4 <= w1; n += 4)
        {
            const __m256d va = _mm256_loadu_pd(a + n), vb = _mm256_loadu_pd(b + n);
            const __m256d lo = _mm256_unpacklo_pd(va, vb), hi = _mm256_unpackhi_pd(va, vb);
            _mm256_storeu_pd(w + 2 * n, _mm256_permute2f128_pd(lo, hi, 0x20));
            _mm256_storeu_pd(w + 2 * n + 4, _mm256_permute2f128_pd(lo, hi, 0x31));
        }
#endif
        for (; n < w1; n++)
        {
            w[2 * n] = a[n];
            w[2 * n + 1] = b[n];
        }
    }
}
static void _lmr_fold(const double *W, double *A, size_t M, size_t K, const int *perm, const double *tw,
                      size_t k0, size_t k1)
{
    const double *Z0 = W + 2 * (perm ? (size_t)perm[0] : 0) * K;
    double *A0 = A, *AM = A + 2 * M * K;
    size_t k, n;
    for (n = k0; n < k1; n++)
    {
        const double r = Z0[2 * n], i = Z0[2 * n + 1];
        A0[2 * n] = r + i;
        A0[2 * n + 1] = 0.0;
        AM[2 * n] = r - i;
        AM[2 * n + 1] = 0.0;
    }
    for (k = 1; 2 * k <= M; k++)
    {
        const size_t rk = perm ? (size_t)perm[k] : k, rm = perm ? (size_t)perm[M - k] : M - k;
        const double *Pk = W + 2 * rk * K, *Pm = W + 2 * rm * K;
        double *Ak = A + 2 * k * K, *Am = A + 2 * (M - k) * K;
        const double wr = tw[2 * k], wi = tw[2 * k + 1];
        const int mirror = (2 * k < M);
        n = k0;
#if defined(__AVX2__)
        {
            const __m256d cj = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0), half = _mm256_set1_pd(0.5);
            const __m256d vwr = _mm256_set1_pd(wr), vwi = _mm256_setr_pd(-wi, wi, -wi, wi);
            for (; n + 2 <= k1; n += 2)
            {
                const __m256d P = _mm256_loadu_pd(Pk + 2 * n), Q = _mm256_xor_pd(_mm256_loadu_pd(Pm + 2 * n), cj);
                const __m256d E = _mm256_mul_pd(half, _mm256_add_pd(P, Q)), D = _mm256_mul_pd(half, _mm256_sub_pd(P, Q));
                const __m256d O = _mm256_xor_pd(_mm256_permute_pd(D, 0x5), cj);                          /* -i D */
                const __m256d T = _mm256_fmadd_pd(O, vwr, _mm256_mul_pd(_mm256_permute_pd(O, 0x5), vwi)); /* w^k O */
                _mm256_storeu_pd(Ak + 2 * n, _mm256_add_pd(E, T));
                if (mirror)
                    _mm256_storeu_pd(Am + 2 * n, _mm256_xor_pd(_mm256_sub_pd(E, T), cj));
            }
        }
#endif
        for (; n < k1; n++)
        {
            const double pr = Pk[2 * n], pi = Pk[2 * n + 1], qr = Pm[2 * n], qi = -Pm[2 * n + 1];
            const double er = 0.5 * (pr + qr), ei = 0.5 * (pi + qi), dr = 0.5 * (pr - qr), di = 0.5 * (pi - qi);
            const double orr = di, oi = -dr;   /* -i D */
            const double tr = orr * wr - oi * wi, ti = orr * wi + oi * wr;
            Ak[2 * n] = er + tr;
            Ak[2 * n + 1] = ei + ti;
            if (mirror)
            {
                Am[2 * n] = er - tr;
                Am[2 * n + 1] = -(ei - ti);
            }
        }
    }
}
/* the lane-range dispatch of the pack and the fold (phase 0 / 1) across the workers */
typedef struct { struct vfft_plan_s *h; const double *x; double *z; size_t k0, k1; int phase; } _lmr_arg_t;
static void _lmr_tramp(void *v)
{
    _lmr_arg_t *a = (_lmr_arg_t *)v;
    struct vfft_plan_s *h = a->h;
    if (a->phase == 0)
        _lmr_pack(a->x, h->lm_W, (size_t)h->lm_M, h->K, a->k0, a->k1);
    else
        _lmr_fold(h->lm_W, a->z, (size_t)h->lm_M, h->K, h->lm_perm, h->lm_tw, a->k0, a->k1);
}
static void _lmr_ranges(struct vfft_plan_s *h, const double *x, double *z, int phase)
{
    const int T = (h->nthreads > 1 && h->il2d_col.colmt) ? thread_pool_workers_for(h->nthreads) : 1;
    const size_t K = h->K;
    _lmr_arg_t a[THREAD_POOL_MAX_DISPATCH];
    int t, Ts = T;
    if (Ts > 1 && K < (size_t)(2 * Ts)) Ts = (int)(K / 2) > 1 ? (int)(K / 2) : 1;   /* at least two lanes a worker (a vector) */
    if (Ts <= 1)
    {
        a[0].h = h; a[0].x = x; a[0].z = z; a[0].k0 = 0; a[0].k1 = K; a[0].phase = phase;
        _lmr_tramp(&a[0]);
        return;
    }
    for (t = 0; t < Ts; t++)
    {
        a[t].h = h; a[t].x = x; a[t].z = z; a[t].phase = phase;
        a[t].k0 = (K * (size_t)t / (size_t)Ts) & ~(size_t)1;         /* even lane bounds: whole vectors */
        a[t].k1 = t + 1 == Ts ? K : (K * (size_t)(t + 1) / (size_t)Ts) & ~(size_t)1;
    }
    thread_pool_run(Ts, _lmr_tramp, a, sizeof a[0]);
}
static int _lmr_exec(struct vfft_plan_s *h, vfft_dir_t dir, const double *xin, double *zout)
{
    (void)dir;   /* an r2c plan is forward-only: the door refuses the other direction before this */
    _lmr_ranges(h, xin, zout, 0);
    _lm_cols(h, h->lm_W, h->lm_W, 0);
    _lmr_ranges(h, xin, zout, 1);
    return 0;
}
static vfft_plan _vfft_create_lmr_il(const vfft_config_t *cfg, struct vfft_wisdom_s *W, int N, size_t K)
{
    const int T = _vfft_plan_threads(cfg), M = N / 2;
    vw2_ilcol_key_t ck;
    vfft_ilcol_t col;
    char forms[64];
    int bwl = -1, btf = -1, bro = -1, bcmt = -1, bcmtt = -1, bblu = -1, k;
    struct vfft_plan_s *h;
    if (!vfft_policy_batch_lane_major(cfg))
    {
        _vfft_warn("vfft_create: the lane-major interleaved real batch serves R2C at an even N >= 4 and howmany >= 2, "
                   "out of place (got %s N=%d howmany=%zu %s); an odd N has no real column engine, and the C2R "
                   "twin is not served yet -- name batch_geom=VFFT_BATCH_TRANSFORM_CONTIGUOUS for those",
                   _vfft_tname(cfg->transform), N, K, cfg->placement == VFFT_INPLACE ? "in place" : "out of place");
        return NULL;
    }
    memset(&ck, 0, sizeof ck);
    ck.rank = 1; ck.n0 = N; ck.n1 = (int)K; ck.ord = VW2_ORD_NAT; ck.real = 1; ck.nthreads = T; ck.lm = 1;
    _il2d_blu_ctx.W = W;
    _il2d_blu_ctx.cfg = cfg;
    _il2d_blu_ctx.N2 = (int)K;
    _il2d_blu_chain_hook = _il2d_blu_m_chain;
    memset(&col, 0, sizeof col);
    forms[0] = 0;
    if (!_il2d_col_build(W, cfg, &ck, M, K, /*nat_req=*/0, &col, forms, sizeof forms,
                         &bwl, &btf, &bro, &bcmt, &bcmtt, &bblu))
    {
        _vfft_warn("vfft_create: lane-major interleaved R2C batch N=%d x K=%zu: no column pass serves the packed "
                   "length M=%d (the column builder refused); unsupported", N, K, M);
        return NULL;
    }
    h = (struct vfft_plan_s *)calloc(1, sizeof *h);
    if (!h)
    {
        _il2d_col_free(&col);
        return NULL;
    }
    h->transform = VFFT_R2C;
    h->placement = cfg->placement;
    h->layout = (int)VFFT_LAYOUT_INTERLEAVED;
    h->N = N;
    h->N2 = 0;
    h->K = K;
    h->nthreads = T;
    h->lm = 1;
    h->lm_M = M;
    h->il2d_col = col;
    h->il2d_col.N = M;
    h->il2d_col.rn = K;
    h->il2d_col.wl = 0;
    h->il2d_col.cut = 0;
    h->il2d_col.tfuse = 0;
    h->il2d_col.wc = 0;
    if (!h->il2d_col.blu && h->il2d_col.nst >= 2)
    {   /* the scrambled pass's row order: the fold reads row k at the pass row that holds it */
        int *np = _il2d_nat_perm(h->il2d_col.R, h->il2d_col.nst, M);
        if (!np || !(h->lm_perm = (int *)malloc((size_t)M * sizeof(int))))
        {
            free(np);
            _vfft_warn("vfft_create: lane-major interleaved R2C batch N=%d x K=%zu: the chain's permutation at M=%d "
                       "does not derive; unsupported", N, K, M);
            _il2d_col_free(&h->il2d_col);
            free(h);
            return NULL;
        }
        for (k = 0; k < M; k++)
            h->lm_perm[np[k]] = k;
        free(np);
    }
    h->lm_W = (double *)vfft_aligned_alloc((2 * (size_t)M * K + 8) * sizeof(double));
    h->lm_tw = (double *)vfft_aligned_alloc(2 * ((size_t)M + 1) * sizeof(double));
    if (!h->lm_W || !h->lm_tw)
    {
        vfft_aligned_free(h->lm_W); vfft_aligned_free(h->lm_tw); free(h->lm_perm);
        _il2d_col_free(&h->il2d_col);
        free(h);
        return NULL;
    }
    for (k = 0; k <= M; k++)
    {   /* w^k = e^{-2 pi i k / N}, once-rounded (common/math/tw_exact.h) */
        double c, s;
        vfft_cs2pi_exact((long long)k, (long long)N, &c, &s);
        h->lm_tw[2 * k] = c;
        h->lm_tw[2 * k + 1] = -s;
    }
    _lm_race(h, W, cfg, &ck, bcmt);
    h->k1_exec = _lmr_exec;
    h->k1_go[VFFT_FORWARD] = _lmr_exec;   /* the lean door: forward only (the door refuses a backward call) */
    h->k1_go[VFFT_BACKWARD] = NULL;
    (void)bcmtt; (void)bwl; (void)btf; (void)bro; (void)bblu;
    return (vfft_plan)h;
}
#endif /* VFFT_IL_LM_BATCH_IL_H */
