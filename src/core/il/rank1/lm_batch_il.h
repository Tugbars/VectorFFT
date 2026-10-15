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
/* the lane-major batches' dense strip scratch (the descriptor's natsscr fields, freed with it): T blocks
 * (one at T = 1) of M x swcap complexes, swcap the widest ladder width at this T or a worker's whole
 * range where that fits L2. The real batch's fused strips (serial tiles and threaded sub-strips) and the
 * c2c batch's scrambled blocks run in it; the natural chain keeps its own builder (its leaf). */
static int _lm_sscr_build(vfft_ilcol_t *c, int M, int K, int T)
{
    int lad[6], nl, k, swcap = 0, t;
    const int Tb = T > 1 ? T : 1;
    if (c->natsscr || c->blu || c->tpc)
        return c->natsscr != NULL;
    nl = _il2d_sw_ladder(M, K, T, lad, 6);
    for (k = 0; k < nl; k++)
        if (lad[k] > swcap) swcap = lad[k];
    if (T > 1 && K / T >= 8 && vfft_policy_fits_l2((long)M * (K / T + 8) * 16) && K / T + 8 > swcap)
        swcap = K / T + 8;   /* a worker's whole range, its bounds cut at 8 lanes */
    if (swcap < 2)
        return 0;
    c->natsscr = (double **)calloc((size_t)Tb, sizeof *c->natsscr);
    if (!c->natsscr)
        return 0;
    for (t = 0; t < Tb; t++)
    {
        c->natsscr[t] = (double *)vfft_aligned_alloc((2 * (size_t)M * (size_t)swcap + 8) * sizeof(double));
        if (!c->natsscr[t])
        {
            c->nnatsscr = t;
            _il2d_nat_sscr_free(c);
            return 0;
        }
    }
    c->nnatsscr = Tb;
    c->natswcap = swcap;
    c->natdense = 1;
    return 1;
}
/* the scratch stays when a verdict runs in it: a real batch's fused strips (any threaded verdict, or a
 * serial tile), a c2c batch's blocks; freed otherwise */
static void _lm_sscr_settle(struct vfft_plan_s *h)
{
    vfft_ilcol_t *c = &h->il2d_col;
    const int real = h->transform != VFFT_C2C;
    const int keep = c->colmt ? (real || c->natarm == 1) : (real && c->wc > 0);
    if (!keep)
        _il2d_nat_sscr_free(c);
}
static int _lmr_exec(struct vfft_plan_s *h, vfft_dir_t dir, const double *xin, double *zout);   /* below */
static int _lmc_exec(struct vfft_plan_s *h, vfft_dir_t dir, const double *zin, double *xout);
typedef struct { struct vfft_plan_s *h; double *z, *zo; int colmt, natarm, msw, wc, ok; char name[24]; } _lm_arm_t;
static void _lm_arm_run(void *v)
{   /* a real batch's arm runs its executor (the fused strips with the arm's knobs, or the three-phase
     * form): the pack and the fold are part of what the knob decides; a c2c batch's arm is the pass */
    _lm_arm_t *a = (_lm_arm_t *)v;
    vfft_ilcol_t *c = &a->h->il2d_col;
    c->colmt = a->colmt;
    c->natarm = a->natarm;
    c->msw = a->msw;
    c->wc = a->wc;
    if (a->h->transform == VFFT_R2C)
        _lmr_exec(a->h, VFFT_FORWARD, a->z, a->zo);
    else if (a->h->transform == VFFT_C2R)
        _lmc_exec(a->h, VFFT_BACKWARD, a->z, a->zo);
    else
        _lm_cols(a->h, a->z, a->zo, 0);
}
static void _lm_race(struct vfft_plan_s *h, struct vfft_wisdom_s *W, const vfft_cfg_t *cfg,
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
        if (c->colmt && c->natarm == 1)
            (void)(c->nat ? _il2d_nat_sscr_build(c, N, K, workers) : _lm_sscr_build(c, N, K, workers));
    }
    if (!need_sw && !need_mt)
    {
        _lm_sscr_settle(h);
        return;
    }
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
            (void)(c->nat ? _il2d_nat_sscr_build(c, N, K, workers) : _lm_sscr_build(c, N, K, workers));
            if (h->transform == VFFT_C2C)   /* a real batch's whole-range strips are the threaded arm */
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
    _lm_sscr_settle(h);   /* the dense scratch stays for the verdicts that run in it */
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
static vfft_plan _vfft_create_lm_il(const vfft_cfg_t *cfg, struct vfft_wisdom_s *W, int N, size_t K)
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
                   "(got %s N=%d howmany=%zu) -- name batch=VFFT_BATCH_CONTIGUOUS for it",
                   _vfft_tname(cfg->transform), N, K);
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
    h->layout = (int)VFI_LAYOUT_IL;
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
 * the fold split by lane ranges across the workers whenever that form serves. Both
 * placements: in place the buffer is (N + 2) K doubles, the N K reals first and the bins
 * over them, and nothing else changes, since the pack copies the whole input into W before
 * the fold writes a bin and the c2r unfold consumes every bin before the unpack writes a
 * real (each phase a whole dispatch). The c2r twin is its own piece. The three phases over the plane
 * are the form for a prime M (the Bluestein and turned axes) and for a width past the scratch; the
 * plain chain runs the FUSED STRIPS below (the dense strip arm with the pack and the fold around it). */
static void _lmr_pack(const double *x, size_t K, double *W, size_t Wp, size_t wk, size_t M, size_t k0, size_t k1)
{
    size_t m, n;
    for (m = 0; m < M; m++)
    {
        const double *a = x + 2 * m * K + k0, *b = a + K;
        double *w = W + 2 * (m * Wp + wk);
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
static void _lmr_fold(const double *W, size_t Wp, size_t wk, double *A, size_t M, size_t K, const int *perm,
                      const double *tw, size_t k0, size_t k1)
{
    const double *Z0 = W + 2 * ((perm ? (size_t)perm[0] : 0) * Wp + wk);
    double *A0 = A, *AM = A + 2 * M * K;
    size_t k, n;
    for (n = k0; n < k1; n++)
    {
        const double r = Z0[2 * (n - k0)], i = Z0[2 * (n - k0) + 1];
        A0[2 * n] = r + i;
        A0[2 * n + 1] = 0.0;
        AM[2 * n] = r - i;
        AM[2 * n + 1] = 0.0;
    }
    for (k = 1; 2 * k <= M; k++)
    {
        const size_t rk = perm ? (size_t)perm[k] : k, rm = perm ? (size_t)perm[M - k] : M - k;
        const double *Pk = W + 2 * (rk * Wp + wk), *Pm = W + 2 * (rm * Wp + wk);
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
                const __m256d P = _mm256_loadu_pd(Pk + 2 * (n - k0)), Q = _mm256_xor_pd(_mm256_loadu_pd(Pm + 2 * (n - k0)), cj);
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
            const double pr = Pk[2 * (n - k0)], pi = Pk[2 * (n - k0) + 1], qr = Pm[2 * (n - k0)], qi = -Pm[2 * (n - k0) + 1];
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
/* ── THE C2R TWIN (the design's step 3): z[2 (f K + t)] (the N / 2 + 1 bins) -> x[e K + t],
 * unnormalized (N x, as every backward in the tree), the fold and the pack inverted:
 *   UNFOLD  per lane, the halves of the fold dropped (that factor 2 and the unnormalized
 *           M-point backward make the N): Z[0] = (A[0] + A[M]) + i (A[0] - A[M]), and for
 *           k = 1..M/2  E = A[k] + conj A[M-k], O = conj(w^k) (A[k] - conj A[M-k]),
 *           Z[k] = E + i O, Z[M-k] = conj(E - i O), written at the pass row the backward
 *           chain reads row k from (the chain's comb);
 *   COLUMNS the backward column chain at M down every lane of W, in place;
 *   UNPACK  x[2 m][k] = Re W[m][k], x[2 m + 1][k] = Im W[m][k]. */
static void _lmc_unfold(const double *A, size_t K, double *W, size_t Wp, size_t wk, size_t M, const int *perm,
                        const double *tw, size_t k0, size_t k1)
{
    double *Z0 = W + 2 * ((perm ? (size_t)perm[0] : 0) * Wp + wk);
    const double *A0 = A, *AM = A + 2 * M * K;
    size_t k, n;
    for (n = k0; n < k1; n++)
    {
        const double a0 = A0[2 * n], am = AM[2 * n];
        Z0[2 * (n - k0)] = a0 + am;
        Z0[2 * (n - k0) + 1] = a0 - am;
    }
    for (k = 1; 2 * k <= M; k++)
    {
        const size_t rk = perm ? (size_t)perm[k] : k, rm = perm ? (size_t)perm[M - k] : M - k;
        const double *Ak = A + 2 * k * K, *Am = A + 2 * (M - k) * K;
        double *Zk = W + 2 * (rk * Wp + wk), *Zm = W + 2 * (rm * Wp + wk);
        const double wr = tw[2 * k], wi = -tw[2 * k + 1];   /* conj(w^k) */
        const int mirror = (2 * k < M);
        n = k0;
#if defined(__AVX2__)
        {
            const __m256d cj = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0);
            const __m256d vwr = _mm256_set1_pd(wr), vwi = _mm256_setr_pd(-wi, wi, -wi, wi);
            for (; n + 2 <= k1; n += 2)
            {
                const __m256d P = _mm256_loadu_pd(Ak + 2 * n), Q = _mm256_xor_pd(_mm256_loadu_pd(Am + 2 * n), cj);
                const __m256d E = _mm256_add_pd(P, Q), D = _mm256_sub_pd(P, Q);
                const __m256d O = _mm256_fmadd_pd(D, vwr, _mm256_mul_pd(_mm256_permute_pd(D, 0x5), vwi));   /* conj(w^k) D */
                const __m256d iO = _mm256_xor_pd(_mm256_permute_pd(O, 0x5), _mm256_setr_pd(-0.0, 0.0, -0.0, 0.0)); /* i O */
                _mm256_storeu_pd(Zk + 2 * (n - k0), _mm256_add_pd(E, iO));
                if (mirror)
                    _mm256_storeu_pd(Zm + 2 * (n - k0), _mm256_xor_pd(_mm256_sub_pd(E, iO), cj));
            }
        }
#endif
        for (; n < k1; n++)
        {
            const double pr = Ak[2 * n], pi = Ak[2 * n + 1], qr = Am[2 * n], qi = -Am[2 * n + 1];
            const double er = pr + qr, ei = pi + qi, dr = pr - qr, di = pi - qi;
            const double orr = dr * wr - di * wi, oi = dr * wi + di * wr;   /* conj(w^k) D */
            const double ior = -oi, ioi = orr;                               /* i O */
            Zk[2 * (n - k0)] = er + ior;
            Zk[2 * (n - k0) + 1] = ei + ioi;
            if (mirror)
            {
                Zm[2 * (n - k0)] = er - ior;
                Zm[2 * (n - k0) + 1] = -(ei - ioi);
            }
        }
    }
}
static void _lmc_unpack(const double *W, size_t Wp, size_t wk, double *x, size_t K, size_t M, size_t k0, size_t k1)
{
    size_t m, n;
    for (m = 0; m < M; m++)
    {
        const double *w = W + 2 * (m * Wp + wk);
        double *a = x + 2 * m * K + k0, *b = a + K;
        const size_t w1 = k1 - k0;
        n = 0;
#if defined(__AVX2__)
        for (; n + 4 <= w1; n += 4)
        {
            const __m256d w0 = _mm256_loadu_pd(w + 2 * n), w1v = _mm256_loadu_pd(w + 2 * n + 4);
            const __m256d lo = _mm256_permute2f128_pd(w0, w1v, 0x20), hi = _mm256_permute2f128_pd(w0, w1v, 0x31);
            _mm256_storeu_pd(a + n, _mm256_unpacklo_pd(lo, hi));
            _mm256_storeu_pd(b + n, _mm256_unpackhi_pd(lo, hi));
        }
#endif
        for (; n < w1; n++)
        {
            a[n] = w[2 * n];
            b[n] = w[2 * n + 1];
        }
    }
}
/* the lane-range dispatch of the elementwise passes across the workers: phase 0 pack, 1 fold
 * (r2c), 2 unfold, 3 unpack (c2r) */
typedef struct { struct vfft_plan_s *h; const double *x; double *z; size_t k0, k1; int phase; } _lmr_arg_t;
static void _lmr_tramp(void *v)
{
    _lmr_arg_t *a = (_lmr_arg_t *)v;
    struct vfft_plan_s *h = a->h;
    switch (a->phase)
    {
    case 0: _lmr_pack(a->x, h->K, h->lm_W, h->K, a->k0, (size_t)h->lm_M, a->k0, a->k1); break;
    case 1: _lmr_fold(h->lm_W, h->K, a->k0, a->z, (size_t)h->lm_M, h->K, h->lm_perm, h->lm_tw, a->k0, a->k1); break;
    case 2: _lmc_unfold(a->x, h->K, h->lm_W, h->K, a->k0, (size_t)h->lm_M, h->lm_perm, h->lm_tw, a->k0, a->k1); break;
    default: _lmc_unpack(h->lm_W, h->K, a->k0, a->z, h->K, (size_t)h->lm_M, a->k0, a->k1); break;
    }
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
/* ── THE FUSED STRIPS (the dense strip arm with the pack and the fold around it, 2026-10-15): a
 * strip of w lanes [k, k + w) goes through a dense scratch of M x w complexes (pitch w, in L2): the
 * pack from the caller's plane into the scratch, the chain in place in the scratch (the plain walker
 * at pitch w: the same kernels, tables and values, so the output is bitwise the plane form's), the
 * fold from the scratch into the caller's spectrum; c2r the mirror. The plane W is never walked at
 * its pitch (at a power-of-two K every row sits a power of two apart, so a sub-strip's rows share
 * one cache set group and every stage streams), the plane is read once and written once, and one
 * dispatch serves the execute. Threaded: each worker's lane range in sub-strips of msw (0 = the
 * whole range), the ranges cut at 8 lanes so no cache line is written by two workers; serial: tiles
 * of wc. The plain chain class with a scratch wide enough (natswcap); a Bluestein or turned axis at
 * a prime M works over the plane, and the three-phase form below serves those and any width past
 * the scratch. In place stays safe: a strip's lanes are packed before any of its bins are written,
 * and no other lane is touched. */
static void _lmr_strip(struct vfft_plan_s *h, const double *x, double *z, size_t k, size_t w, double *scr, int c2r)
{
    const vfft_ilcol_t *c = &h->il2d_col;
    const size_t M = (size_t)h->lm_M, K = h->K;
    if (!c2r)
    {
        _lmr_pack(x, K, scr, w, 0, M, k, k + w);
        _il2d_col_pass_range(scr, scr, (int)M, w, 0, w, c->nst, c->R, c->L, c->f, c->tf, 0);
        _lmr_fold(scr, w, 0, z, M, K, h->lm_perm, h->lm_tw, k, k + w);
    }
    else
    {
        _lmc_unfold(x, K, scr, w, 0, M, h->lm_perm, h->lm_tw, k, k + w);
        _il2d_col_pass_range(scr, scr, (int)M, w, 0, w, c->nst, c->R, c->L, c->b, c->tb, 1);
        _lmc_unpack(scr, w, 0, z, K, M, k, k + w);
    }
}
typedef struct { struct vfft_plan_s *h; const double *x; double *z; size_t lo, hi, sw; int tid, c2r; } _lmf_arg_t;
static void _lmf_tramp(void *v)
{
    _lmf_arg_t *a = (_lmf_arg_t *)v;
    double *scr = a->h->il2d_col.natsscr[a->tid];
    size_t k;
    for (k = a->lo; k < a->hi; k += a->sw)
        _lmr_strip(a->h, a->x, a->z, k, (k + a->sw < a->hi) ? a->sw : a->hi - k, scr, a->c2r);
}
/* 1 = the fused strips served (a threaded run counted as an engagement), 0 = the three-phase form's turn */
static int _lmr_fused(struct vfft_plan_s *h, const double *x, double *z, int c2r)
{
    vfft_ilcol_t *c = &h->il2d_col;
    const size_t K = h->K, cap = (size_t)c->natswcap;
    _lmf_arg_t a[THREAD_POOL_MAX_DISPATCH];
    int t, T;
    if (!c->natsscr || c->nnatsscr < 1 || c->nat || c->blu || c->tpc || c->wl > 0)
        return 0;
    if (h->nthreads > 1 && c->colmt)
    {
        const size_t msw = c->msw > 0 ? (size_t)c->msw : 0;
        int ok = msw <= cap;
        T = thread_pool_workers_for(h->nthreads);
        if (T > c->nnatsscr) T = c->nnatsscr;
        if (K / 8 < (size_t)T) T = (int)(K / 8);   /* eight lanes a worker at least: whole cache lines */
        if (T < 2)
            return 0;
        for (t = 0; t < T; t++)
        {
            a[t].h = h; a[t].x = x; a[t].z = z; a[t].tid = t; a[t].c2r = c2r;
            a[t].lo = (K * (size_t)t / (size_t)T) & ~(size_t)7;
            a[t].hi = t + 1 == T ? K : (K * (size_t)(t + 1) / (size_t)T) & ~(size_t)7;
            a[t].sw = msw ? msw : a[t].hi - a[t].lo;
            if (a[t].sw > cap || a[t].hi <= a[t].lo) ok = 0;
        }
        if (!ok)
            return 0;
        thread_pool_run(T, _lmf_tramp, a, sizeof a[0]);
        _vfft_il2d_col_mt_count++;   /* the engagement (vfft_diagnostics.h): the caller's duty */
        return 1;
    }
    if (c->wc <= 0 || (size_t)c->wc > cap)
        return 0;
    a[0].h = h; a[0].x = x; a[0].z = z; a[0].lo = 0; a[0].hi = K; a[0].sw = (size_t)c->wc; a[0].tid = 0; a[0].c2r = c2r;
    _lmf_tramp(&a[0]);
    return 1;
}
static int _lmr_exec(struct vfft_plan_s *h, vfft_dir_t dir, const double *xin, double *zout)
{
    (void)dir;   /* an r2c plan is forward-only: the door refuses the other direction before this */
    if (_lmr_fused(h, xin, zout, 0))
        return 0;
    _lmr_ranges(h, xin, zout, 0);
    _lm_cols(h, h->lm_W, h->lm_W, 0);
    _lmr_ranges(h, xin, zout, 1);
    return 0;
}
static int _lmc_exec(struct vfft_plan_s *h, vfft_dir_t dir, const double *zin, double *xout)
{
    (void)dir;   /* a c2r plan is backward-only */
    if (_lmr_fused(h, zin, xout, 1))
        return 0;
    _lmr_ranges(h, zin, xout, 2);
    _lm_cols(h, h->lm_W, h->lm_W, 1);
    _lmr_ranges(h, zin, xout, 3);
    return 0;
}
/* the real batch's create, r2c or c2r by the request (the same chain at M on the shared real row,
 * the direction's own executor) */
static vfft_plan _vfft_create_lmr_il(const vfft_cfg_t *cfg, struct vfft_wisdom_s *W, int N, size_t K)
{
    const int T = _vfft_plan_threads(cfg), M = N / 2, c2r = cfg->transform == VFFT_C2R;
    vw2_ilcol_key_t ck;
    vfft_ilcol_t col;
    char forms[64];
    int bwl = -1, btf = -1, bro = -1, bcmt = -1, bcmtt = -1, bblu = -1, k;
    struct vfft_plan_s *h;
    if (!vfft_policy_batch_lane_major(cfg))
    {
        _vfft_warn("vfft_create: the lane-major interleaved real batch serves R2C and C2R at an even N >= 4 and "
                   "howmany >= 2, in either placement (got %s N=%d howmany=%zu); an odd N has no real column "
                   "engine -- name batch=VFFT_BATCH_CONTIGUOUS for it",
                   _vfft_tname(cfg->transform), N, K);
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
    h->transform = c2r ? VFFT_C2R : VFFT_R2C;
    h->placement = cfg->placement;
    h->layout = (int)VFI_LAYOUT_IL;
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
    if (!h->il2d_col.blu && !h->il2d_col.tpc)
        (void)_lm_sscr_build(&h->il2d_col, M, (int)K, T);   /* the fused strips' scratch, raced below */
    _lm_race(h, W, cfg, &ck, bcmt);
    h->k1_exec = c2r ? _lmc_exec : _lmr_exec;   /* the lean door: the direction's own executor, the other refused by the door */
    h->k1_go[VFFT_FORWARD] = c2r ? NULL : _lmr_exec;
    h->k1_go[VFFT_BACKWARD] = c2r ? _lmc_exec : NULL;
    (void)bcmtt; (void)bwl; (void)btf; (void)bro; (void)bblu;
    return (vfft_plan)h;
}
#endif /* VFFT_IL_LM_BATCH_IL_H */
