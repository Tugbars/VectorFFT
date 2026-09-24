/* fft2d_create.h — the rank-2 CREATE tier (migration step 23).
 *
 * WHAT THIS IS
 * ------------
 * The dims==2 arm of _vfft_create_inner: the largest single tier in the
 * dispatcher, and the one that decides between the two rank-2 servings the
 * library actually has. It returns on every path, so it lifts out behind a
 * guard without disturbing the rank-1 tiers that follow it.
 *
 * n[0]=N1 (rows), n[1]=N2 (columns).
 *
 * THE TWO SERVINGS, and how the tier chooses
 * ------------------------------------------
 * INTERLEAVED (lay=il) is the native rank-2 route. Its passes, MT and racers
 * live in transforms/fft2d/il2d_tier.h; this tier is what BUILDS that plan --
 * it settles the row child, the chain, and the banked axes (wl, cut, tfuse,
 * rowoop) before handing off. The il2d_* locals declared at the top of the
 * block are exactly those decisions in flight.
 *
 * SPLIT is the other serving, and it is a different machine end to end --
 * different codelets, different executor, different planner. The two do not
 * share an interior; the choice is made here and never revisited.
 *
 * c2c is in-place (tiled-row + native-column). r2c/c2r are out-of-place: a
 * real plane against an N1 x (N2/2+1) spectrum, one plan serving both
 * directions.
 *
 * WHAT IS DECIDED HERE vs WHAT IS RACED
 * -------------------------------------
 * This tier does not invent a plan. Where a choice is open it calls a racer
 * (_il2d_real_rowrace and the rest, all in il2d_tier.h) and banks the verdict;
 * where wisdom already holds a verdict it replays it. A banked line reads back
 * as a verdict, never as a heuristic -- so nothing in this file may grow a
 * hand-written cutoff.
 *
 * POSITION IN vfft.c IS LOAD-BEARING
 * ----------------------------------
 * Not a standalone header. Like il2d_tier.h, k1_commit.h and zr2c_build.h it
 * calls file-scope statics that live in vfft.c (_vfft_plan_threads,
 * _vw2_lay_of, _vw2_persist, _build_2d), so it must be included after those
 * are defined and before _vfft_create_inner.
 *
 * The four parameters are the block's complete free-variable set, derived
 * rather than guessed: cfg, W, reg, K. N1/N2 are locals declared inside the
 * block; the enclosing N and ob are NOT used by it.
 */
#ifndef VFFT_TRANSFORMS_FFT2D_CREATE_H
#define VFFT_TRANSFORMS_FFT2D_CREATE_H

/* the two arms of the N1-arm race: the pow2 chain column pass vs the
 * Bluestein column pass, both in place on one scratch plane */
static vfft_plan _vfft_create_2d(const vfft_config_t *cfg,
                                 struct vfft_wisdom_s *W,
                                 const vfft_proto_registry_t *reg,
                                 size_t K)
{
    if (cfg->dims == 2)
    {
        /* the Bluestein inner chain provider for THIS create (2026-09-02):
         * the (M, N2) chain row serves the length-M column chain */
        _il2d_blu_ctx.W = W;
        _il2d_blu_ctx.cfg = cfg;
        _il2d_blu_ctx.N2 = cfg->n[1];
        _il2d_blu_chain_hook = _il2d_blu_m_chain;
        /* §6a50/Q4: the 2D executors are K-blind — howmany > 1 is served
         * by the PLANE QUEUE (2026-08-27, the designed sequential-plane
         * batching): a wrapper over one primary howmany=1 plan (loop
         * mode, keeps its intra-MT verdicts) + serial clones pulled by
         * an atomic plane counter (queue mode), loop-vs-queue RACED at
         * create. Contiguous planes only (the canonical dist for each
         * transform); layouts/transforms the tier cannot express keep
         * the loud refusal. */
        if (K != 1)
        {
            const int N1q = cfg->n[0], N2q = cfg->n[1];
            const size_t hp1q = (size_t)N2q / 2 + 1;
            vfft_config_t ic;
            struct vfft_plan_s *h;
            if (cfg->layout != VFFT_LAYOUT_INTERLEAVED ||
                (cfg->transform != VFFT_C2C &&
                 cfg->transform != VFFT_R2C &&
                 cfg->transform != VFFT_C2R))
            {
                _vfft_warn("vfft_create: dims=2 howmany=%zu is served by "
                           "the plane queue for INTERLEAVED C2C/R2C/C2R "
                           "only (got %s, layout=%d) — batch other 2D "
                           "plans sequentially",
                           K, _vfft_tname(cfg->transform),
                           (int)cfg->layout);
                return NULL;
            }
            ic = *cfg;
            ic.howmany = 1;
            h = (struct vfft_plan_s *)calloc(1, sizeof *h);
            if (!h)
                return NULL;
            h->pq_inner =
                (struct vfft_plan_s *)vfft_create(&ic); /* warns itself */
            if (!h->pq_inner)
            {
                free(h);
                return NULL;
            }
            h->transform = cfg->transform;
            h->placement = cfg->placement;
            h->layout = (int)cfg->layout;
            h->N = N1q;
            h->N2 = N2q;
            h->K = K;
            h->nthreads = _vfft_plan_threads(cfg);
            h->pq_n = K;
            if (cfg->transform == VFFT_C2C)
            {
                h->pq_sdist = 2 * (size_t)N1q * N2q;
                h->pq_ddist = h->pq_sdist;
            }
            else if (cfg->transform == VFFT_R2C)
            {
                h->pq_sdist = (size_t)N1q * N2q;
                h->pq_ddist = 2 * (size_t)N1q * hp1q;
            }
            else
            {
                h->pq_sdist = 2 * (size_t)N1q * hp1q;
                h->pq_ddist = (size_t)N1q * N2q;
            }
            /* queue clones: SERIAL instances (a queue worker must not
             * nest-dispatch), wisdom-served from the verdicts the
             * primary just banked, each BITWISE-verified on a probe
             * plane — any mismatch tears the set down and the loop
             * serves. */
            if (h->nthreads > 1 && K >= 2)
            {
                /* h->nthreads is already <= the live setting
                 * (_vfft_plan_threads), so the pool's one clamp = the
                 * plan snapshot bounded by the pool and the dispatch
                 * array; _pq_execute takes the same clamp on pq_wn. */
                int T = stride_pool_workers_for(h->nthreads);
                const vfft_dir_t pd = (cfg->transform == VFFT_C2R)
                                          ? VFFT_BACKWARD
                                          : VFFT_FORWARD;
                double *ps, *p0, *p1;
                int t, ok = 1;
                if ((size_t)T > K)
                    T = (int)K;
                ic.nthreads = 1;
                ic.wisdom_write = 0;
                ic.recalibrate = 0;   /* the PRIMARY above (:98) kept the caller's
                                       * flag and has already re-raced and banked;
                                       * a clone that also carries it re-races the
                                       * same cell T more times and contradicts the
                                       * law stated at :127-131 ("wisdom-served from
                                       * the verdicts the primary just banked") */
                ps = (double *)malloc(h->pq_sdist * sizeof(double));
                p0 = (double *)malloc(h->pq_ddist * sizeof(double));
                p1 = (double *)malloc(h->pq_ddist * sizeof(double));
                h->pq_w = (struct vfft_plan_s **)calloc(
                    (size_t)T, sizeof *h->pq_w);
                if (ps && p0 && p1 && h->pq_w && T >= 2)
                {
                    size_t i2;
                    for (i2 = 0; i2 < h->pq_sdist; i2++)
                        ps[i2] = 1.0 + 1e-6 * (double)(i2 & 511);
                    vfft_execute((vfft_plan)h->pq_inner, pd, ps, NULL,
                                 p0, NULL);
                    for (t = 0; t < T && ok; t++)
                    {
                        h->pq_w[t] =
                            (struct vfft_plan_s *)vfft_create(&ic);
                        if (!h->pq_w[t])
                        {
                            ok = 0;
                            break;
                        }
                        vfft_execute((vfft_plan)h->pq_w[t], pd, ps,
                                     NULL, p1, NULL);
                        if (memcmp(p0, p1,
                                   h->pq_ddist * sizeof(double)) != 0)
                            ok = 0;
                    }
                    if (ok)
                        h->pq_wn = T;
                    else
                    {
                        _vfft_warn("plane queue %dx%d: clone build/"
                                   "bitwise probe failed — queue "
                                   "declines, the serial loop serves",
                                   N1q, N2q);
                        for (t = 0; t < T; t++)
                            if (h->pq_w[t])
                                vfft_destroy(h->pq_w[t]);
                        free(h->pq_w);
                        h->pq_w = NULL;
                        h->pq_wn = 0;
                    }
                }
                free(ps);
                free(p0);
                free(p1);
                if (h->pq_wn > 0)
                    _pq_mt_replay_or_race(h, W, cfg); /* banked per (P,T) */
            }
            return h;
        }
        int N1 = cfg->n[0], N2 = cfg->n[1];
        /* ── native IL 2D c2c tier — THE serving for IL callers (OWNER
         * LAW 2026-08-25: no convert wrapper, split is not a fallback of
         * IL). Cold cells race the chain + axes and bank the lay=il
         * verdict; inexpressible cells (no chain; natural at multi-stage
         * until the rho tables; child failure) REFUSE loudly. The split
         * tplan below is built ONLY for split-layout callers. */
        struct vfft_plan_s *il2d_row = NULL;
        char il2d_fm[64] = "";   /* the raced per-stage forms (E1.11); re-banked once the chain row lands */
        int il2d_nst = 0;
        int il2d_wc = 0;
        int il2d_wl = 0, il2d_cut = 0, il2d_tfuse = 0;
        int il2d_bwl = -1, il2d_btf = -1, il2d_bro = -1; /* banked axes */
        int il2d_axmt = 0;  /* the T-AWARE axis verdict serves this create (axt= == the plan's T, 2026-09-24) */
        int il2d_staged = 0, il2d_pitch = 0;
        double *il2d_bandscr = NULL;
        double *il2d_rscr = NULL;
        struct vfft_plan_s *il2d_rows = NULL;
        int il2d_rw = 0;
        int il2d_brw = -1; /* banked row-route verdict; -1 = unraced */
        int il2d_oddn2 = 0;        /* odd-N2 real: c2c row child */
        double *il2d_orbuf = NULL; /* its 2 x 2*N2 row pair buffer  */
        int il2d_blu = 0;          /* odd/prime N1: column Bluestein M */
        int il2d_bblu = -1;        /* banked N1-arm verdict; -1 = unraced */
        int il2d_rowb = 0;         /* row route 2: the BATCHED rows (2026-09-23) */
        vfft_il2p_fn il2d_rowb_f = NULL, il2d_rowb_b = NULL; /* its n1ccs pair at N2, when the radix has one */
        int il2d_rowb2 = 0;        /* row route 3: the batched TWO-PASS rows (2026-09-23) */
        vfft_il2p_fn il2d_rowb2_leaf_f = NULL, il2d_rowb2_mid_f = NULL, il2d_rowb2_t2t_b = NULL, il2d_rowb2_n1_b = NULL;
        double *il2d_rowb2_scr = NULL;
        int il2d_rowb2_ch = 0;     /* its tile in rows, from the banked rbk= / the env pin */
        int il2d_turn = 0;         /* the TURN route (2026-09-23): the whole plane through the 1D engine */
        int il2d_csk = 0;          /* the SKEWED column pass (2026-09-23) */
        double *il2d_csk_scr = NULL;
        struct vfft_plan_s *il2d_csk_row = NULL;
        vfft_il2p_fn il2d_csk_f = NULL, il2d_csk_b = NULL;
        struct vfft_plan_s *il2d_turn_plan = NULL;
        double *il2d_turn_scr = NULL;
        int il2d_nat = 0;          /* NATURAL n1 via the leaf redirection */
        int *il2d_natperm = NULL;
        double *il2d_natscr = NULL;
        /* the wisdom ORDER axis of this cell (2026-09-04): natural cells
         * race their chain under the natural pass and bank on their own
         * ord=nat row — never sharing the scr row's chain. */
        const int il2d_ord = vfft_policy_ord_rankn(cfg);
        int il2d_tbl_done = 0;     /* N1 tables built early (the N1-arm race) */
        double *il2d_bluchf = NULL, *il2d_bluchb = NULL;
        double *il2d_blukf = NULL, *il2d_blukb = NULL;
        double *il2d_bluscr = NULL;
        int il2d_tpc = 0;                     /* the turned prime pass (2026-09-24) */
        struct vfft_plan_s *il2d_tpcplan = NULL;
        double *il2d_tpcscr = NULL;
        int il2d_bcmt = -1, il2d_bcmtt = -1; /* banked column-MT verdict
                                              * and the T it was raced at */
        double *il2d_lx = NULL, *il2d_lre = NULL, *il2d_lim = NULL;
        double *il2d_tre = NULL, *il2d_tim = NULL;
        int il2d_R[8] = { 0 }, il2d_L[8] = { 0 };
        vfft_il2p_fn il2d_f[8] = { 0 }, il2d_b[8] = { 0 };
        double *il2d_tf[8] = { 0 }, *il2d_tb[8] = { 0 };
        if (cfg->transform == VFFT_C2C &&
            cfg->layout == VFFT_LAYOUT_INTERLEAVED)
        {
            {   /* THE COLUMN-AXIS PASS BUILD (2026-09-06): chain (env > banked
                 * > raced > greedy), forms, Bluestein, the natural leaf
                 * redirection, the N1-arm race, tables — _il2d_col_build in
                 * il2d_tier.h, the same function a rank-N IL plan runs per
                 * column axis. The tier's locals below take its result; the
                 * plan commit further down copies them into h->il2d_col. */
                const vw2_ilcol_key_t ck = { 2, N1, N2, 0, il2d_ord, 0, 0 };
                vfft_ilcol_t col;
                memset(&col, 0, sizeof col);
                if (!_il2d_col_build(W, cfg, &ck, N1, (size_t)N2,
                                     vfft_policy_rankn_axis_nat(2, 0, il2d_ord), &col,
                                     il2d_fm, sizeof il2d_fm, &il2d_bwl, &il2d_btf,
                                     &il2d_bro, &il2d_bcmt, &il2d_bcmtt, &il2d_bblu))
                    return NULL;
                /* the T-AWARE axis verdict (2026-09-24): at T > 1 the axis race ran
                 * every arm threaded and banked beside the serial verdict as
                 * axt= rot= wlt= swt= rbkt= turnt= cskt= (the T raced at, like
                 * cmtt). Served at that T only; another T re-races. The serial
                 * tokens stay the one-thread verdict. */
                {
                    const int thr = _vfft_plan_threads(cfg) > 0 ? _vfft_plan_threads(cfg) : 1;   /* the plan's T (h is committed later) */
                    il2d_axmt = (thr > 1 && !cfg->recalibrate && W &&
                                 vw2_2d_il_tok_geti(&W->vw2, N1, N2, il2d_ord, "axt", 0) == thr);
                }
                if (il2d_axmt)
                {
                    il2d_bro = vw2_2d_il_tok_geti(&W->vw2, N1, N2, il2d_ord, "rot", -1);
                    il2d_bwl = vw2_2d_il_tok_geti(&W->vw2, N1, N2, il2d_ord, "wlt", -1);
                    il2d_btf = il2d_bwl > 0;
                }
                il2d_nst = col.nst;
                memcpy(il2d_R, col.R, sizeof il2d_R);
                memcpy(il2d_L, col.L, sizeof il2d_L);
                memcpy(il2d_f, col.f, sizeof il2d_f);
                memcpy(il2d_b, col.b, sizeof il2d_b);
                memcpy(il2d_tf, col.tf, sizeof il2d_tf);
                memcpy(il2d_tb, col.tb, sizeof il2d_tb);
                il2d_blu = col.blu;
                il2d_bluchf = col.bluchf; il2d_bluchb = col.bluchb;
                il2d_blukf = col.blukf; il2d_blukb = col.blukb;
                il2d_bluscr = col.bluscr;
                il2d_tpc = col.tpc;
                il2d_tpcplan = col.tpcplan;
                il2d_tpcscr = col.tpcscr;
                il2d_nat = col.nat;
                il2d_natperm = col.natperm;
                il2d_natscr = col.natscr;
                il2d_tbl_done = 1;         /* the builder built the tables */
            }
            {
                vfft_config_t rc;
                memset(&rc, 0, sizeof rc);
                rc.transform = VFFT_C2C;
                rc.placement = VFFT_INPLACE;
                rc.rigor = cfg->rigor;
                rc.dims = 1;
                rc.n[0] = N2;
                rc.howmany = 1;
                rc.order = VFFT_ORDER_NATURAL;
                rc.layout = VFFT_LAYOUT_INTERLEAVED;
                rc.nthreads = 1;
                rc.wisdom = cfg->wisdom;
                rc.wisdom_write = cfg->wisdom_write;
                il2d_row = (struct vfft_plan_s *)vfft_create(&rc);
                if (il2d_row && !il2d_blu && !il2d_tbl_done &&
                    _il2d_build_tables(N1, il2d_nst, il2d_R,
                                       il2d_L, il2d_tf, il2d_tb))
                {
                    vfft_destroy(il2d_row);
                    il2d_row = NULL;
                }
                if (!il2d_row)
                {
                    _vfft_warn("vfft_create: IL 2D c2c %dx%d — native "
                               "row child / stage tables failed; "
                               "unsupported (no wrapper by owner law)",
                               N1, N2);
                    return NULL;
                }
                /* the BATCHED row route (ro=2, 2026-09-23): the n1ccs pair
                 * at radix N2 -- one kernel call per run of rows, lane k =
                 * row k, two rows per vector, no per-row door. Bound
                 * whenever the radix has the pair; the axis race (or the
                 * banked verdict, or the env pin VFFT_IL2D_ROWOOP=2) decides
                 * whether it serves. A banked ro=2 this build has no kernel
                 * for is re-raced, never served by another route. */
                il2d_rowb_f = vfft_il_n1ccs_fn(N2, 0);
                il2d_rowb_b = vfft_il_n1ccs_fn(N2, 1);
                if (!il2d_rowb_f || !il2d_rowb_b)
                    il2d_rowb_f = il2d_rowb_b = NULL;
                if (il2d_row && il2d_bro == 2)
                {
                    if (il2d_rowb_f && !getenv("VFFT_IL2D_ROWOOP"))
                        il2d_rowb = 1;
                    else if (!il2d_rowb_f)
                        il2d_bro = -1;
                }
                if (il2d_row && il2d_rowb_f && getenv("VFFT_IL2D_ROWOOP") &&
                    atoi(getenv("VFFT_IL2D_ROWOOP")) == 2)
                    il2d_rowb = 1;
                /* the BATCHED TWO-PASS rows (ro=3, 2026-09-23): the row child's
                 * own two-pass factorization through the row-loop twins of its
                 * four stage kernels -- per chunk of rows one call per stage,
                 * staged through a per-worker contiguous scratch. Bound when the
                 * child is a two-pass plan whose kernels all have twins; the
                 * axis race (or the banked ro=3, or VFFT_IL2D_ROWOOP=3) decides. */
                if (il2d_row && il2d_row->k1il2p)
                {
                    const vfft_il2p_plan_t *pp = il2d_row->k1il2p;
                    il2d_rowb2_leaf_f = _il2d_rowloop_twin(pp->leaf_f);
                    il2d_rowb2_mid_f = _il2d_rowloop_twin(pp->mid_f);
                    il2d_rowb2_t2t_b = _il2d_rowloop_twin(pp->t2t_b);
                    il2d_rowb2_n1_b = _il2d_rowloop_twin(pp->n1_b_r2);
                    if (il2d_rowb2_leaf_f && il2d_rowb2_mid_f && il2d_rowb2_t2t_b && il2d_rowb2_n1_b)
                    {
                        const int Tn = _vfft_plan_threads(cfg) > 0 ? _vfft_plan_threads(cfg) : 1;
                        il2d_rowb2_scr = (double *)VFFT_ZS_ALLOC((size_t)Tn * 2 * (size_t)VFFT_IL2D_RB2_CHUNK
                                                                * (size_t)N2 * sizeof(double));
                    }
                    if (!il2d_rowb2_scr)
                        il2d_rowb2_leaf_f = il2d_rowb2_mid_f = il2d_rowb2_t2t_b = il2d_rowb2_n1_b = NULL;
                }
                if (il2d_row && il2d_bro == 3)
                {
                    if (il2d_rowb2_leaf_f && !getenv("VFFT_IL2D_ROWOOP"))
                        il2d_rowb2 = 1;
                    else if (!il2d_rowb2_leaf_f)
                        il2d_bro = -1;
                }
                if (il2d_row && il2d_rowb2_leaf_f && getenv("VFFT_IL2D_ROWOOP") &&
                    atoi(getenv("VFFT_IL2D_ROWOOP")) == 3)
                    il2d_rowb2 = 1;
                /* the TURN route (2026-09-23): the whole plane through the 1D
                 * engine -- the batched mono row kernel with TURNED stores into
                 * an N2 x N1 scratch, the N2 columns as its rows through the
                 * in-place K=1 natural plan at N1, one back-turn. NATURAL cells
                 * whose N2 has the n1ccs pair and whose N1 has an in-place K=1
                 * plan (the door creates that cell here: replayed or raced and
                 * banked like any 1D cell). The axis race decides (one arm);
                 * banked turn=1; VFFT_IL2D_ROWOOP=4 pins it. */
                if (il2d_row && !il2d_blu && il2d_rowb_f && vfft_policy_rankn_axis_nat(2, 0, il2d_ord))
                {
                    vfft_config_t tc;
                    memset(&tc, 0, sizeof tc);
                    tc.transform = VFFT_C2C;
                    tc.placement = VFFT_INPLACE;
                    tc.rigor = cfg->rigor;
                    tc.dims = 1;
                    tc.n[0] = N1;
                    tc.howmany = 1;
                    tc.order = VFFT_ORDER_NATURAL;
                    tc.layout = VFFT_LAYOUT_INTERLEAVED;
                    tc.nthreads = 1;
                    tc.wisdom = cfg->wisdom;
                    tc.wisdom_write = cfg->wisdom_write;
                    il2d_turn_plan = (struct vfft_plan_s *)vfft_create(&tc);
                    if (il2d_turn_plan)
                    {
                        il2d_turn_scr = (double *)VFFT_ZS_ALLOC(2 * VFFT_IL2D_TURN_PITCH(N1) * (size_t)N2 * sizeof(double));
                        if (!il2d_turn_scr)
                        {
                            vfft_destroy(il2d_turn_plan);
                            il2d_turn_plan = NULL;
                        }
                    }
                    if (il2d_turn_plan && !getenv("VFFT_IL2D_ROWOOP") &&
                        vw2_2d_il_tok_geti(&W->vw2, N1, N2, il2d_ord, il2d_axmt ? "turnt" : "turn", 0) == 1)
                        il2d_turn = 1;
                    if (il2d_turn_plan && getenv("VFFT_IL2D_ROWOOP") && atoi(getenv("VFFT_IL2D_ROWOOP")) == 4)
                        il2d_turn = 1;
                }
                /* the SKEWED column pass (csk, 2026-09-23): a single-stage column
                 * chain writes the scratch at pitch N2 + 8, the rows move it into
                 * the plane. The scratch and the out-of-place K=1 plan at N2 (the
                 * per-row route from the scratch; the batched routes need none)
                 * are built whenever the chain is single-stage; the axis race
                 * decides (crossed with the row routes); banked csk=1;
                 * VFFT_IL2D_CSK=1 pins it. */
                il2d_csk_f = vfft_k1_mono_ilc_fn(N1, 0);
                il2d_csk_b = vfft_k1_mono_ilc_fn(N1, 1);
                if (!il2d_csk_f || !il2d_csk_b)
                    il2d_csk_f = il2d_csk_b = NULL;
                if (il2d_row && !il2d_blu && il2d_csk_f)
                {   /* its own single column stage (the n1c pair at radix N1, natural
                     * by construction): the route stands beside the banked chain
                     * whatever the chain race picked */
                    il2d_csk_scr = (double *)VFFT_ZS_ALLOC(2 * (size_t)N1 * ((size_t)N2 + 8) * sizeof(double));
                    if (il2d_csk_scr)
                    {
                        vfft_config_t oc;
                        memset(&oc, 0, sizeof oc);
                        oc.transform = VFFT_C2C;
                        oc.placement = VFFT_OUTOFPLACE;
                        oc.rigor = cfg->rigor;
                        oc.dims = 1;
                        oc.n[0] = N2;
                        oc.howmany = 1;
                        oc.order = VFFT_ORDER_NATURAL;
                        oc.layout = VFFT_LAYOUT_INTERLEAVED;
                        oc.nthreads = 1;
                        oc.wisdom = cfg->wisdom;
                        oc.wisdom_write = cfg->wisdom_write;
                        il2d_csk_row = (struct vfft_plan_s *)vfft_create(&oc);
                    }
                    if (il2d_csk_scr && !getenv("VFFT_IL2D_ROWOOP") && !getenv("VFFT_IL2D_CSK") &&
                        vw2_2d_il_tok_geti(&W->vw2, N1, N2, il2d_ord, il2d_axmt ? "cskt" : "csk", 0) == 1)
                        il2d_csk = 1;
                    if (il2d_csk_scr && getenv("VFFT_IL2D_CSK") && atoi(getenv("VFFT_IL2D_CSK")) == 1)
                        il2d_csk = 1;
                }
                /* the tile: the banked rbk= (KB of chunk scratch; 8 where a row
                 * predates the token), VFFT_IL2D_RB2_KB pinning it for probes */
                if (il2d_rowb2_leaf_f)
                {
                    int kb = vw2_2d_il_tok_geti(&W->vw2, N1, N2, il2d_ord, il2d_axmt ? "rbkt" : "rbk", 8);
                    if (getenv("VFFT_IL2D_RB2_KB") && atoi(getenv("VFFT_IL2D_RB2_KB")) > 0)
                        kb = atoi(getenv("VFFT_IL2D_RB2_KB"));
                    il2d_rowb2_ch = (int)_il2d_rb2_rows(kb, (size_t)N2);
                }
                /* column-tile width: env override (raced axis; wisdom
                 * banking follows the falsifier run — tcut precedent:
                 * env BEATS wisdom). 0/absent/invalid = untiled. */
                {
                    const char *wce = getenv("VFFT_IL2D_WC");
                    il2d_wc = (wce && atoi(wce) > 0 && atoi(wce) < N2)
                                  ? atoi(wce)
                                  : 0;
                }
                /* the row routes (2026-09-23): 0 = the in-place child, 2 = the
                 * batched rows, 3 = the batched two-pass rows, raced below and
                 * banked as ro=; VFFT_IL2D_ROWOOP=2|3 pins one for a probe. The
                 * out-of-place child (ro=1, the copy-back) is DELETED entirely:
                 * the in-place K=1 tier serves every N2 (its last candidate is
                 * the prime engine); a row banked on it re-races. */
                if (il2d_row && il2d_bro == 1)
                    il2d_bro = -1;
                /* staged band route: VFFT_IL2D_STAGED=1 (needs a
                 * band; checked after the wl parse below). */
                /* banded walk: VFFT_IL2D_WL = band width in ROWS (the
                 * width is the INPUT, the cut is DERIVED — the tcut law).
                 * Legal iff wl | N1 and some suffix stage has L_s | wl;
                 * anything else warns and stays unbanded. VFFT_IL2D_TFUSE
                 * =0 opts out of the per-band row pass (default ON when
                 * banded — the fusion is the point). */
                if (il2d_row && !il2d_blu)
                {   /* natural cells too (2026-09-05): the natural banded
                     * walk in vfft_execute.h honours wl / cut / tfuse */
                    const char *we = getenv("VFFT_IL2D_WL");
                    const char *tfe = getenv("VFFT_IL2D_TFUSE");
                    int wl = we ? atoi(we) : (il2d_bwl > 0 ? il2d_bwl : 0);
                    il2d_wl = 0;
                    il2d_cut = 0;
                    il2d_tfuse = 0;
                    if (wl > 0)
                    {
                        int cut = -1, s2;
                        if (wl <= N1 && N1 % wl == 0)
                            for (s2 = 0; s2 < il2d_nst; s2++)
                                if (wl % il2d_L[s2] == 0)
                                {
                                    cut = s2;
                                    break;
                                }
                        if (cut < 0)
                            _vfft_warn("VFFT_IL2D_WL=%d illegal at %dx%d "
                                       "(needs wl | N1 and a stage with "
                                       "L_s | wl) — unbanded",
                                       wl, N1, N2);
                        else
                        {
                            il2d_wl = wl;
                            il2d_cut = cut;
                            il2d_tfuse = !(tfe && atoi(tfe) == 0);
                        }
                    }
                    /* the banked strip width of an unbanded serial verdict
                     * (il2d_large_plane_design.md, 2026-09-15); env beats it */
                    if (il2d_wl == 0 && il2d_bwl == 0 && !getenv("VFFT_IL2D_WC") && !getenv("VFFT_IL2D_WL") &&
                        !cfg->recalibrate && W)
                        il2d_wc = vw2_2d_il_tok_geti(&W->vw2, N1, N2, il2d_ord,
                                                     il2d_axmt ? "swt" : "sw", 0);
                    if (il2d_wl > 0 && !il2d_nat && getenv("VFFT_IL2D_STAGED") &&
                        atoi(getenv("VFFT_IL2D_STAGED")) == 1)
                    {
                        /* skew selection: smallest even pad where every
                         * suffix stage's leg stride 16*D*pitch AND the
                         * leaf stride 16*pitch are non-0 mod 4096. */
                        int sk;
                        for (sk = 2; sk <= 32; sk += 2)
                        {
                            const int pit = N2 + sk;
                            int s3, ok2 = ((16 * (size_t)pit) % 4096) != 0;
                            for (s3 = il2d_cut;
                                 ok2 && s3 < il2d_nst; s3++)
                            {
                                const int Dv =
                                    il2d_L[s3] / il2d_R[s3];
                                if (Dv > 1 &&
                                    ((16 * (size_t)Dv * pit) % 4096) == 0)
                                    ok2 = 0;
                            }
                            if (ok2)
                            {
                                il2d_pitch = pit;
                                break;
                            }
                        }
                        if (il2d_pitch > 0)
                        {
                            il2d_bandscr = (double *)malloc(
                                2 * (size_t)il2d_wl * il2d_pitch
                                * sizeof(double));
                            if (il2d_bandscr)
                                il2d_staged = 1;
                            else
                                il2d_pitch = 0;
                        }
                        else
                            _vfft_warn("VFFT_IL2D_STAGED: no skew <=32 "
                                       "de-aliases every stage at %dx%d "
                                       "— staying direct", N1, N2);
                    }
                }
            }
        }
        /* ── native IL 2D REAL tier (docs/roadmap/fft2d_real_il_design.md)
         * — M3: THE serving for IL real 2D callers (OWNER LAW: split is
         * not a fallback of IL — native or LOUD refusal; the env gate is
         * GONE, the c2c wrapper-deletion pattern). Pure IL end-to-end:
         * rows = the raced row route (per-row TC door or ROWSPLIT),
         * columns = the n1c/t2c chain over hp1 = N2/2+1 columns with the
         * raced banded walk. Two-phase law (§2.5): the Hermitian fold is
         * R-linear and does not commute with the column stages — fwd
         * rows complete before column stage 0, bwd rows follow the last
         * column stage; no tfuse, and the c2c cells' banked wl/tf
         * verdicts do not port. OOP only (2D real in-place is refused
         * above; the in-place door needs the padded-pitch caller
         * contract, §2.7). SPLIT-layout callers keep the split engine
         * untouched. Inexpressible cells (odd N2 — the zr2c row door is
         * even-only; NATURAL order — waits on the rho tapes; chain/row
         * failures) REFUSE loudly. */
        if ((cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R) &&
            cfg->layout == VFFT_LAYOUT_INTERLEAVED &&
            cfg->placement == VFFT_OUTOFPLACE)
        {
            int rok = 1;
            const int oddn2 = (N2 % 2) != 0;
            /* ODD N2 (2026-08-27, owner "we can support it and we
             * should"): the zr2c reinterpret needs even N2, so odd rows
             * ride a K=1 c2c child instead — promote real -> complex ->
             * keep hp1 bins fwd; Hermitian-extend -> inverse -> Re bwd.
             * Any odd N2 (the child covers odd/prime/awkward via the
             * pair/chain/prime engines). hp1 = N2/2+1 = (N2+1)/2 falls
             * out of the same integer division, so the column pass and
             * the rscr sizing below are the even path untouched. */
            /* order=NATURAL: single-stage chains are natural-native;
             * blu is natural by construction; multi-stage chains take
             * the M4-lite leaf redirection — resolved AFTER the chain
             * builds (below), never refused up front any more. */
            if (rok)
            {
                /* THE SHARED COLUMN BUILDER (R6, 2026-09-17,
                 * il2d_real_on_shared_builder_design.md). Until today this
                 * block was ~300 lines: the c2c builder's decision -- env pin
                 * > banked row > the chain race > the column-axis Bluestein,
                 * the forms, the natural leaf, the replay-rebuild, the N1-arm
                 * race -- copied inline with the real tier's own lookup and
                 * bank, so every divergence the builder fixed this week had
                 * to be found twice. The real tier's row IS the ilcol key with
                 * real = 1 ({t=r2c rank 2 N1xN2 ord lay=il}, exactly
                 * vw2_2d_rl_*'s key); chain= and blu= are direction-shared on
                 * it, so the builder's lookup and bank land where they always
                 * did. What is NOT shared -- rw= wl= cmt= cmtt=, spelled per
                 * direction by vw2__rl_tok -- the builder does not know: its
                 * plain-name out-params are ignored below and the four are
                 * re-read exactly as before. */
                const vw2_ilcol_key_t ck = { 2, N1, N2, 0, il2d_ord, 0, /*real=*/1 };
                vfft_ilcol_t col;
                int bwl_ = -1, btf_ = -1, bro_ = -1, bcmt_ = -1, bcmtt_ = -1;
                memset(&col, 0, sizeof col);
                rok = _il2d_col_build(W, cfg, &ck, N1, (size_t)N2 / 2 + 1,
                                      vfft_policy_rankn_axis_nat(2, 0, il2d_ord), &col,
                                      il2d_fm, sizeof il2d_fm, &bwl_, &btf_, &bro_,
                                      &bcmt_, &bcmtt_, &il2d_bblu);
                if (rok)
                {
                    /* the copy-out, verbatim from the c2c branch */
                    il2d_nst = col.nst;
                    memcpy(il2d_R, col.R, sizeof il2d_R);
                    memcpy(il2d_L, col.L, sizeof il2d_L);
                    memcpy(il2d_f, col.f, sizeof il2d_f);
                    memcpy(il2d_b, col.b, sizeof il2d_b);
                    memcpy(il2d_tf, col.tf, sizeof il2d_tf);
                    memcpy(il2d_tb, col.tb, sizeof il2d_tb);
                    il2d_blu = col.blu;
                    il2d_bluchf = col.bluchf; il2d_bluchb = col.bluchb;
                    il2d_blukf = col.blukf; il2d_blukb = col.blukb;
                    il2d_bluscr = col.bluscr;
                    il2d_nat = col.nat;
                    il2d_natperm = col.natperm;
                    il2d_natscr = col.natscr;
                    /* the per-direction verdicts (rw wl cmt cmtt), as before:
                     * from the row, this direction's tokens, never under
                     * recalibrate; -1 = unraced, and the row-route race below
                     * fills them in */
                    if (!cfg->recalibrate)
                    {
                        int tR[8], tn = 0, tblu = -1;
                        (void)vw2_2d_rl_lookup(&W->vw2, N1, N2,
                                               cfg->transform == VFFT_C2R, tR, &tn,
                                               &il2d_brw, &il2d_bwl, &il2d_bcmt,
                                               &il2d_bcmtt, &tblu, il2d_ord);
                    }
                }
            }
            if (rok && oddn2)
            {
                /* the odd row child: K=1 c2c at N2, NATURAL (the CCE
                 * bins must come out in order), OOP into the row pair
                 * buffer. Serial — the row loop is plain; threading the
                 * odd rows via clones is the noted follow-up. */
                vfft_config_t rc;
                memset(&rc, 0, sizeof rc);
                rc.transform = VFFT_C2C;
                rc.placement = VFFT_OUTOFPLACE;
                rc.rigor = cfg->rigor;
                rc.dims = 1;
                rc.n[0] = N2;
                rc.howmany = 1;
                rc.order = VFFT_ORDER_NATURAL;
                rc.layout = VFFT_LAYOUT_INTERLEAVED;
                rc.nthreads = 1;
                rc.wisdom = cfg->wisdom;
                rc.wisdom_write = cfg->wisdom_write;
                il2d_row = (struct vfft_plan_s *)vfft_create(&rc);
                if (il2d_row)
                {
                    il2d_orbuf = (double *)malloc(
                        4 * (size_t)N2 * sizeof(double));
                    if (!il2d_orbuf)
                    {
                        vfft_destroy(il2d_row);
                        il2d_row = NULL;
                    }
                }
                if (!il2d_row)
                {
                    _vfft_warn("vfft_create: IL 2D %s %dx%d — odd N2 "
                               "row child (c2c %d) failed; the cell "
                               "refuses (no split fallback by owner "
                               "law)",
                               _vfft_tname(cfg->transform), N1, N2, N2);
                    return NULL;
                }
                if (cfg->transform == VFFT_C2R)
                {
                    il2d_rscr = (double *)malloc(
                        (2 * (size_t)N1 * ((size_t)N2 / 2 + 1) + 8)
                        * sizeof(double));
                    if (!il2d_rscr)
                    {
                        vfft_destroy(il2d_row);
                        free(il2d_orbuf);
                        return NULL;
                    }
                }
                il2d_oddn2 = 1;
            }
            else if (rok)
            {
                vfft_config_t rc;
                memset(&rc, 0, sizeof rc);
                rc.transform = cfg->transform;
                rc.placement = VFFT_OUTOFPLACE;
                rc.rigor = cfg->rigor;
                rc.dims = 1;
                rc.n[0] = N2;
                rc.howmany = (size_t)N1;
                rc.batch_geom = VFFT_BATCH_TRANSFORM_CONTIGUOUS;
                rc.layout = VFFT_LAYOUT_INTERLEAVED;
                /* MT INC-1: the row pass IS a transform-contiguous batch of
                 * N1 whole rows — exactly the shape the TC clone MT already
                 * threads (clones gated by _tc_inner_mt_safe: the zr2c route
                 * is pool-free, and _tc_clone_equiv proves each clone
                 * bit-equivalent). Passing the caller's budget through is the
                 * whole change; the column pass stays serial until INC-3. */
                rc.nthreads = cfg->nthreads;
                rc.wisdom = cfg->wisdom;
                rc.wisdom_write = cfg->wisdom_write;
                il2d_row = (struct vfft_plan_s *)vfft_create(&rc);
                /* PURITY GATE: the TC inner must be the zr2c composite —
                 * the 1D OOP real create quietly falls through to the
                 * split-interior CCE path when the zr2c child fails, and
                 * serving that here would rebuild the veneer under a
                 * native flag (never_build_hybrid_il_split_codelets,
                 * route level). */
                if (il2d_row &&
                    !(il2d_row->tcb && il2d_row->tcb->zr2c_child))
                {
                    _vfft_warn("vfft_create: IL 2D real %dx%d — the row "
                               "door at N2=%d is not the zr2c route "
                               "(purity gate); the cell refuses",
                               N1, N2, N2);
                    vfft_destroy(il2d_row);
                    il2d_row = NULL;
                }
                if (il2d_row && cfg->transform == VFFT_C2R)
                {
                    /* §2.6 contract: input-preserving OOP c2r — the
                     * reversed column chain's first executed stage moves
                     * the caller's z into this plane; the rows read it
                     * and write the caller's real dst. */
                    /* +8 dbl pad: the fused c2r unzip reads full 4-wide
                     * e-blocks past the last row's tail (benign lanes). */
                    il2d_rscr = (double *)malloc(
                        (2 * (size_t)N1 * ((size_t)N2 / 2 + 1) + 8)
                        * sizeof(double));
                    if (!il2d_rscr)
                    {
                        vfft_destroy(il2d_row);
                        il2d_row = NULL;
                    }
                }
                /* ── the ROWSPLIT route (struct comment). Precedence:
                 * env VFFT_IL2D_ROWSPLIT (0 pins the per-row door,
                 * W>0 pins rowsplit) > the banked rw= verdict > the
                 * create-time race (after the commits below).
                 * Constraints: W%8 (the split engines' lane grain),
                 * W | N1, N2%4 (the 4x4 transpose grain). Any build
                 * failure keeps the per-row TC door — never a refusal. */
                if (il2d_row)
                {
                    const char *rse = getenv("VFFT_IL2D_ROWSPLIT");
                    const int Wb = rse ? atoi(rse)
                                       : (il2d_brw > 0 ? il2d_brw : 0);
                    if (Wb > 0)
                    {
                        if (Wb >= 8 && Wb % 8 == 0 && Wb <= N1 &&
                            N1 % Wb == 0 && (N2 % 4) == 0)
                        {
                            if (_il2d_rowsplit_build(cfg, Wb, N2,
                                                     &il2d_rows,
                                                     &il2d_lx, &il2d_lre,
                                                     &il2d_lim, &il2d_tre,
                                                     &il2d_tim))
                                il2d_rw = Wb;
                            else
                                _vfft_warn("il2d rowsplit W=%d: split "
                                           "row engine unavailable at "
                                           "%dx%d — per-row door serves",
                                           Wb, N1, N2);
                        }
                        else
                            _vfft_warn("il2d rowsplit W=%d illegal at "
                                       "%dx%d (needs W%%8==0, W|N1, "
                                       "N2%%4==0) — per-row door serves",
                                       Wb, N1, N2);
                    }
                }
                /* ── the banded column walk's width (env VFFT_IL2D_WL,
                 * shared name with c2c; 0 pins unbanded) > banked wl= >
                 * the create-time race. Legality: wl | N1 and a suffix
                 * stage with L_s | wl (cut derived); illegal warns and
                 * stays unbanded. Rows are OUTSIDE the walk (§2.5). */
                if (il2d_row)
                {
                    const char *we = getenv("VFFT_IL2D_WL");
                    const int wlv = we ? atoi(we)
                                       : (il2d_bwl > 0 ? il2d_bwl : 0);
                    il2d_wl = 0;
                    il2d_cut = 0;
                    if (wlv > 0)
                    {
                        int cut = -1, s2;
                        if (wlv <= N1 && N1 % wlv == 0)
                            for (s2 = 0; s2 < il2d_nst; s2++)
                                if (wlv % il2d_L[s2] == 0)
                                {
                                    cut = s2;
                                    break;
                                }
                        if (cut < 0)
                            _vfft_warn("il2d real wl=%d illegal at "
                                       "%dx%d (needs wl | N1 and a "
                                       "stage with L_s | wl) — unbanded",
                                       wlv, N1, N2);
                        else
                        {
                            il2d_wl = wlv;
                            il2d_cut = cut;
                        }
                    }
                }
                if (!il2d_row)
                    rok = 0;
            }
            if (!rok && il2d_nst)
            {
                /* tables built for a cell that then refused */
                int s2;
                for (s2 = 0; s2 < il2d_nst; s2++)
                {
                    free(il2d_tf[s2]);
                    free(il2d_tb[s2]);
                    il2d_tf[s2] = il2d_tb[s2] = NULL;
                }
                il2d_nst = 0;
            }
            if (!rok)
            {
                /* OWNER LAW: split is NOT a fallback of IL — no veneer.
                 * (row door / purity / chain / tables failed; the
                 * specific cause warned above.) */
                _vfft_warn("vfft_create: IL 2D %s %dx%d — native tier "
                           "construction failed; unsupported for now "
                           "(no split fallback by owner law)",
                           _vfft_tname(cfg->transform), N1, N2);
                return NULL;
            }
            if (il2d_row && getenv("VFFT_IL2D_LOG"))
                fprintf(stderr, "[il2d-real] native %s %dx%d nst=%d "
                                "engaged\n",
                        cfg->transform == VFFT_C2R ? "c2r" : "r2c",
                        N1, N2, il2d_nst);
        }
        stride_plan_t *tp = NULL;
        if (!il2d_row)
        {
            tp = _build_2d(cfg->transform, N1, N2, cfg->rigor, reg, W, cfg->recalibrate,
                           cfg->order, _vw2_lay_of(cfg));
            /* wave-4: the inner-cell spike save is GONE — _inner_c2c banks into
             * the wisdom2 store; the guarded _vw2_persist below covers disk. */
            if (!tp)
                return NULL;
            /* wave-3 flip: the legacy per-create unconditional rewrites of the
             * three fft2d files are GONE (they ran even when the create FAILED,
             * and clobber-rewrote on pure warm hits — those files are frozen
             * now). _build_2d banked into the wisdom2 store's memory; disk
             * persistence is the guarded save, and only after a SUCCESSFUL
             * create. (The native path banks nothing 2D — its row child
             * persisted its own 1D verdicts inside its create.) */
            _vw2_persist(W, cfg);
        }
        struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
        if (!h)
        {
            if (tp)
                stride_plan_destroy(tp);
            if (il2d_row)
                vfft_destroy(il2d_row);
            free(il2d_rscr);
            return NULL;
        }
        h->transform = cfg->transform;
        h->placement = cfg->placement;
        h->layout = (int)cfg->layout;
        h->N = N1;
        h->N2 = N2;
        h->K = K;
        h->nthreads = _vfft_plan_threads(cfg);
        h->tplan = tp; /* NULL when the native IL 2D tier engaged */
        h->il2d_row = il2d_row;
        h->il2d_col.N = N1;
        h->il2d_col.rn = (cfg->transform == VFFT_C2C) ? (size_t)N2 : (size_t)N2 / 2 + 1;
        h->il2d_col.nst = il2d_nst;
        h->il2d_col.wc = il2d_wc;
        h->il2d_col.wl = il2d_wl;
        h->il2d_col.cut = il2d_cut;
        h->il2d_col.tfuse = il2d_tfuse;
        h->il2d_rowb = il2d_rowb;
        h->il2d_rowb_f = il2d_rowb_f;
        h->il2d_rowb_b = il2d_rowb_b;
        h->il2d_rowb2 = il2d_rowb2;
        h->il2d_rowb2_leaf_f = il2d_rowb2_leaf_f;
        h->il2d_rowb2_mid_f = il2d_rowb2_mid_f;
        h->il2d_rowb2_t2t_b = il2d_rowb2_t2t_b;
        h->il2d_rowb2_n1_b = il2d_rowb2_n1_b;
        h->il2d_rowb2_scr = il2d_rowb2_scr;
        h->il2d_rowb2_ch = il2d_rowb2_ch;
        h->il2d_turn = il2d_turn;
        h->il2d_turn_plan = il2d_turn_plan;
        h->il2d_turn_scr = il2d_turn_scr;
        h->il2d_csk = il2d_csk;
        h->il2d_csk_scr = il2d_csk_scr;
        h->il2d_csk_row = il2d_csk_row;
        h->il2d_csk_f = il2d_csk_f;
        h->il2d_csk_b = il2d_csk_b;
        h->il2d_col.staged = il2d_staged;
        h->il2d_col.pitch = il2d_pitch;
        h->il2d_col.bandscr = il2d_bandscr;
        h->il2d_rscr = il2d_rscr;
        h->il2d_rows = il2d_rows;
        h->il2d_rw = il2d_rw;
        h->il2d_oddn2 = il2d_oddn2;
        h->il2d_orbuf = il2d_orbuf;
        h->il2d_col.nat = il2d_nat;
        h->il2d_col.natperm = il2d_natperm;
        h->il2d_col.natscr = il2d_natscr;
        h->il2d_col.natstage = NULL;
        h->il2d_col.natst = 1;
        if (il2d_nat && il2d_natscr && il2d_nst >= 1)
        {   /* the leaf's staging, one block per worker (il2d_natural_leaf_design.md) */
            const int Rl = il2d_R[il2d_nst - 1];
            const int T = h->nthreads > 0 ? h->nthreads : 1;
            h->il2d_col.natstage = (double *)VFFT_ZS_ALLOC((size_t)T * 2 * (size_t)Rl * h->il2d_col.rn * sizeof(double));
        }
        h->il2d_col.blu = il2d_blu;
        h->il2d_col.bluchf = il2d_bluchf;
        h->il2d_col.bluchb = il2d_bluchb;
        h->il2d_col.blukf = il2d_blukf;
        h->il2d_col.blukb = il2d_blukb;
        h->il2d_col.bluscr = il2d_bluscr;
        h->il2d_col.tpc = il2d_tpc;
        h->il2d_col.tpcplan = il2d_tpcplan;
        h->il2d_col.tpcscr = il2d_tpcscr;
        /* A/B race knob (struct comment): create-time env read only. */
        h->il2d_norowz = getenv("VFFT_IL2D_NO_ROWZ") != NULL;
        h->il2d_lx = il2d_lx;
        h->il2d_lre = il2d_lre;
        h->il2d_lim = il2d_lim;
        h->il2d_tre = il2d_tre;
        h->il2d_tim = il2d_tim;
        memcpy(h->il2d_col.R, il2d_R, sizeof il2d_R);
        memcpy(h->il2d_col.L, il2d_L, sizeof il2d_L);
        memcpy(h->il2d_col.f, il2d_f, sizeof il2d_f);
        memcpy(h->il2d_col.b, il2d_b, sizeof il2d_b);
        memcpy(h->il2d_col.tf, il2d_tf, sizeof il2d_tf);
        memcpy(h->il2d_col.tb, il2d_tb, sizeof il2d_tb);
        /* ── the AXIS RACE (§10a): wl and rowoop timed on the FULL
         * execute (they involve the rows), the winner set on the plan
         * and banked WITH the chain as one verdict. Runs only when the
         * axes are unknown: no env override and no banked verdict.
         * MUST sit AFTER the stage-array commits above — it executes h.
         * c2c ONLY: the real tier has no banded walk / row route to race
         * (§2.5 — banding+tfuse on a real plan is the illegal fusion). */
        if (h->transform == VFFT_C2C && h->il2d_row && !il2d_blu &&
            !getenv("VFFT_IL2D_WL") && !getenv("VFFT_IL2D_CSK") &&
            !getenv("VFFT_IL2D_ROWOOP") && !getenv("VFFT_IL2D_TFUSE") &&
            (h->nthreads > 1 ? !il2d_axmt : (il2d_bwl < 0 || il2d_bro < 0)))
        {   /* at T > 1 the T-aware race (2026-09-24): every route's clone set
             * first, every arm threaded, the unneeded sets dropped after */
            if (h->nthreads > 1)
            {
                _il2d_c2c_build_clone_sets_all(h, cfg, h->nthreads);
                _il2d_nat_sscr_build(&h->il2d_col, N1, N2, h->nthreads);   /* the strips' dense scratch before the race: its arms run the form that serves */
            }
            _il2d_axis_race(h, W, cfg, N1, N2);
            if (h->nthreads > 1)
            {
                _il2d_c2c_drop_unneeded_clones(h);
                /* the threading verdict was raced for the OLD route: a new
                 * route at this T re-races it (a stale strips verdict replayed
                 * onto the tiled rows made 128x256 slower at T=8 than at 1) */
                il2d_bcmt = il2d_bcmtt = -1;
            }
        }
        /* INC-C: c2c MT. Build the per-worker row clones (the serving
         * row path mutates shared plan state), then serve the banked
         * cmt verdict ONLY at the T it was raced at, else race and
         * bank. Runs AFTER the axis race — the row route (rowoop) the
         * clones must match is final only then. */
        if (h->transform == VFFT_C2C && h->il2d_row && h->nthreads > 1)
        {   /* (Bluestein cells race too since 2026-09-02: the window pipeline;
             * the turn and the skewed pass since 2026-09-24: their row-slab
             * walks, one threaded arm each) */
            const char *ce = getenv("VFFT_IL2D_NO_COLMT");
            _il2d_c2c_build_clones(h, cfg, h->nthreads);
            _il2d_nat_sscr_build(&h->il2d_col, N1, N2, h->nthreads);   /* the strips' dense scratch (2026-09-24) */
            if (ce)
                h->il2d_col.colmt = (atoi(ce) == 0);
            else if (il2d_bcmt >= 0 &&
                     vfft_policy_replays_at_T(il2d_bcmtt, h->nthreads))
            {   /* the verdict and its shape (mtarm/msw), at the T raced */
                const int ord = il2d_ord;   /* the cell's order class (policy.h, L4) */
                h->il2d_col.colmt = il2d_bcmt;
                h->il2d_col.natarm = il2d_bcmt ? vw2_2d_il_tok_geti(&W->vw2, N1, N2, ord, "mtarm", 0) : 0;
                h->il2d_col.msw = il2d_bcmt ? vw2_2d_il_tok_geti(&W->vw2, N1, N2, ord, "msw", 0) : 0;
                h->il2d_col.natst = il2d_bcmt ? vw2_2d_il_tok_geti(&W->vw2, N1, N2, ord, "nls", 1) : 1;
            }
            else
                _il2d_c2c_mt_race(h, W, cfg, N1, N2);
        }
        /* ── the REAL tier's row-route race (per-row door vs ROWSPLIT W
         * pool): runs only when env is FULLY silent (an env-pinned chain
         * skips the banked-row read AND must never bank — env beats
         * wisdom, never writes it: the tcut law) and the rl cell carries
         * no rw= verdict; banks chain+rw direction-shared. Same
         * after-the-commits law as the c2c axis race — it executes h. */
        if ((h->transform == VFFT_R2C || h->transform == VFFT_C2R) &&
            h->il2d_row && !il2d_oddn2 && !il2d_blu && !il2d_nat &&
            !getenv("VFFT_IL2D_ROWSPLIT") &&
            !getenv("VFFT_IL2D_CHAIN") && !getenv("VFFT_IL2D_WL") &&
            (il2d_brw < 0 || il2d_bwl < 0))
            _il2d_real_rowrace(h, W, cfg, N1, N2);
        /* the raced per-stage forms land on the real chain row HERE (2026-09-04):
         * on a cold real cell that row is first written by the rowrace's
         * rl bank above, after the forms step ran — without this re-bank
         * the next create (a column-MT clone, the replay) re-raced the forms
         * and could serve different kernels than this handle (MT != ST). */
        if ((h->transform == VFFT_R2C || h->transform == VFFT_C2R) &&
            il2d_fm[0] && W && !W->vw2_off_2d && !getenv("VFFT_IL2D_FORMS"))
        {
            int ok = vw2_2d_forms_bank(&W->vw2, 1, N1, N2, il2d_fm, il2d_ord);
            if (!ok)
            {   /* no real row yet: the rowrace did not run (an env pin on the
                 * row axis, e.g. a gate's VFFT_IL2D_ROWSPLIT) or refused. The
                 * forms verdict still has a home — the row with the served
                 * chain and NO row-axis tokens (rw/wl unraced, never erased:
                 * a later rowrace MERGES into this row). */
                vw2_2d_rl_bank(&W->vw2, N1, N2, h->transform == VFFT_C2R,
                               h->il2d_col.R, h->il2d_col.nst, -1, -1, -1, 0,
                               (N1 & (N1 - 1)) ? h->il2d_col.blu : -1, 0.0, il2d_ord);
                ok = vw2_2d_forms_bank(&W->vw2, 1, N1, N2, il2d_fm, il2d_ord);
            }
            if (ok)
                _vw2_persist(W, cfg);
            else if (getenv("VFFT_IL2D_LOG"))
                fprintf(stderr, "[il2d] forms %dx%d: %s could not be banked on the real row\n",
                        N1, N2, il2d_fm, il2d_ord);
        }
        /* INC-3: the column-MT verdict. Serve a banked one ONLY when it
         * was raced at THIS thread count; otherwise race and bank. A
         * single-threaded plan never threads columns and never races. */
        if ((h->transform == VFFT_R2C || h->transform == VFFT_C2R) &&
            h->il2d_row && h->nthreads > 1)
        {   /* (Bluestein cells race too since 2026-09-02) */
            const char *ce = getenv("VFFT_IL2D_NO_COLMT");
            if (ce)
                h->il2d_col.colmt = (atoi(ce) == 0);
            else if (il2d_bcmt >= 0 &&
                     vfft_policy_replays_at_T(il2d_bcmtt, h->nthreads))
                h->il2d_col.colmt = il2d_bcmt;
            else
                _il2d_real_colmt_race(h, W, cfg, N1, N2);
        }
        /* §6a31: rfft-engine row inner for the R2C 2D row pass — the rfft
         * path wins at the tile's low K (−27%/call measured). Force the rfft
         * dispatch; adopt only if it landed (RFFT path, split, plan bound).
         * tp guard: the native IL real tier leaves tp NULL — veneer only. */
        if (cfg->transform == VFFT_R2C && tp)
        {
            stride_fft2d_r2c_data_t *d2 = (stride_fft2d_r2c_data_t *)tp->override_data;
            size_t saved2 = vfft_r2c_dispatch_get_decouple_min_k();
            vfft_r2c_dispatch_set_decouple_min_k((size_t)-1);
            h->rfft_row = vfft_r2c_plan_create(N2, d2->B, VFFT_R2C_SPLIT,
                                               _rfft_registry(), NULL,
                                               (vfft_proto_registry_t *)reg);
            vfft_r2c_dispatch_set_decouple_min_k(saved2);
            if (h->rfft_row && h->rfft_row->path == VFFT_R2C_PATH_RFFT && h->rfft_row->layout == VFFT_R2C_SPLIT && h->rfft_row->rfft)
            {
                /* §6a31: MEASURED adoption — "rfft wins at low K" does not
                 * survive N-scaling ((512,8) regressed +66% before this
                 * gate). A/B both inners on tile scratch at create
                 * (same-process, 64 reps each, sub-ms) and keep the winner. */
                double *sr0 = _fft2d_r2c_scratch_re(d2, 0);
                double *si0 = _fft2d_r2c_scratch_im(d2, 0);
                size_t tsz = d2->tile_real_sz;
                double *bak2 = (double *)malloc(tsz * sizeof(double));
                for (size_t ii = 0; ii < tsz; ii++)
                    bak2[ii] = 1.0 + 1e-3 * (double)(ii & 63);
                rfft_plan_t *rp2 = h->rfft_row->rfft;
                struct timespec t0_, t1_;
                double t_str, t_rff;
                /* per-rep refill BOTH arms (unnormalized reps compound to
                 * inf otherwise; equal handicap keeps the ratio honest). */
                memcpy(sr0, bak2, tsz * sizeof(double));
                _fft2d_r2c_inner_fwd(d2->plan_r2c, sr0, si0, 0); /* warm */
                clock_gettime(CLOCK_MONOTONIC, &t0_);
                for (int rr2 = 0; rr2 < 64; rr2++)
                {
                    memcpy(sr0, bak2, tsz * sizeof(double));
                    _fft2d_r2c_inner_fwd(d2->plan_r2c, sr0, si0, 0);
                }
                clock_gettime(CLOCK_MONOTONIC, &t1_);
                t_str = (t1_.tv_sec - t0_.tv_sec) * 1e9 + (t1_.tv_nsec - t0_.tv_nsec);
                memcpy(sr0, bak2, tsz * sizeof(double));
                rfft_execute_fwd_natural(rp2, sr0, sr0, si0, NULL); /* warm */
                clock_gettime(CLOCK_MONOTONIC, &t0_);
                for (int rr2 = 0; rr2 < 64; rr2++)
                {
                    memcpy(sr0, bak2, tsz * sizeof(double));
                    rfft_execute_fwd_natural(rp2, sr0, sr0, si0, NULL);
                }
                clock_gettime(CLOCK_MONOTONIC, &t1_);
                t_rff = (t1_.tv_sec - t0_.tv_sec) * 1e9 + (t1_.tv_nsec - t0_.tv_nsec);
                free(bak2);
                /* §6a34: hysteresis — engine deltas measured <=3%, inside
                 * regime-to-regime noise; create-time gates flipped winners
                 * across weather regimes. The challenger must beat the
                 * stride incumbent by >5% or the incumbent stays. */
                if (t_rff * 20 < t_str * 19)
                    d2->rfft_row = rp2;
                else
                {
                    vfft_r2c_plan_destroy(h->rfft_row);
                    h->rfft_row = NULL;
                }
            }
            else if (h->rfft_row)
            {
                vfft_r2c_plan_destroy(h->rfft_row);
                h->rfft_row = NULL;
            }
        }
        /* §6a32: bwd twin — c2r natural-engine row inner for the C2R 2D
         * plan, measured-adopted exactly like the fwd gate. tp guard as
         * §6a31: the native IL real tier leaves tp NULL. */
        if (cfg->transform == VFFT_C2R && tp)
        {
            stride_fft2d_r2c_data_t *d2 = (stride_fft2d_r2c_data_t *)tp->override_data;
            h->c2r_row = vfft_c2r_disp_create(N2, d2->B, VFFT_C2R_NATURAL,
                                              _rfft_registry(),
                                              (vfft_proto_registry_t *)reg);
            if (h->c2r_row && h->c2r_row->packed && h->c2r_row->packed->nat_init)
            {
                double *sr0 = _fft2d_r2c_scratch_re(d2, 0);
                double *si0 = _fft2d_r2c_scratch_im(d2, 0);
                size_t tcz = d2->tile_complex_sz, trz = d2->tile_real_sz;
                double *bkr = (double *)malloc((tcz > trz ? tcz : trz) * sizeof(double));
                double *bki = (double *)malloc(tcz * sizeof(double));
                for (size_t ii = 0; ii < tcz; ii++)
                {
                    bkr[ii] = 1.0 + 1e-3 * (double)(ii & 63);
                    bki[ii] = 0.5 - 1e-3 * (double)(ii & 31);
                }
                c2r_plan_t *cp2 = h->c2r_row->packed;
                struct timespec t0_, t1_;
                double t_str, t_c2r;
                memcpy(sr0, bkr, tcz * sizeof(double));
                memcpy(si0, bki, tcz * sizeof(double));
                _fft2d_r2c_inner_bwd(d2->plan_r2c, sr0, si0, 0); /* warm */
                clock_gettime(CLOCK_MONOTONIC, &t0_);
                for (int rr2 = 0; rr2 < 64; rr2++)
                {
                    memcpy(sr0, bkr, tcz * sizeof(double));
                    memcpy(si0, bki, tcz * sizeof(double));
                    _fft2d_r2c_inner_bwd(d2->plan_r2c, sr0, si0, 0);
                }
                clock_gettime(CLOCK_MONOTONIC, &t1_);
                t_str = (t1_.tv_sec - t0_.tv_sec) * 1e9 + (t1_.tv_nsec - t0_.tv_nsec);
                memcpy(sr0, bkr, tcz * sizeof(double));
                memcpy(si0, bki, tcz * sizeof(double));
                c2r_execute_natural(cp2, sr0, si0, sr0, NULL); /* warm */
                clock_gettime(CLOCK_MONOTONIC, &t0_);
                for (int rr2 = 0; rr2 < 64; rr2++)
                {
                    memcpy(sr0, bkr, tcz * sizeof(double));
                    memcpy(si0, bki, tcz * sizeof(double));
                    c2r_execute_natural(cp2, sr0, si0, sr0, NULL);
                }
                clock_gettime(CLOCK_MONOTONIC, &t1_);
                t_c2r = (t1_.tv_sec - t0_.tv_sec) * 1e9 + (t1_.tv_nsec - t0_.tv_nsec);
                free(bkr);
                free(bki);
                if (t_c2r * 20 < t_str * 19) /* §6a34 hysteresis */
                    d2->c2r_row = cp2;
                else
                {
                    vfft_c2r_disp_destroy(h->c2r_row);
                    h->c2r_row = NULL;
                }
            }
            else if (h->c2r_row)
            {
                vfft_c2r_disp_destroy(h->c2r_row);
                h->c2r_row = NULL;
            }
        }
        /* ORDER_NATURAL (2D c2c): build the two per-axis digit-reversal reorder tapes from the inner
         * plans' chains. SCRAMBLED/DEFAULT leave nat2d==0 (byte-identical scrambled path). Refuse
         * (free + NULL) if orientation detect fails on either multi-stage axis — no silent wrong order. */
        if (cfg->transform == VFFT_C2C && cfg->order == VFFT_ORDER_NATURAL &&
            !h->il2d_row) /* native tier serves natural already; tp is NULL there */
        {
            stride_fft2d_data_t *d = (stride_fft2d_data_t *)tp->override_data;
            int col_is_pairs = 0; /* dim2 runs cycle_pass in fft2d.h scratch -> never a pair tape */
            /* dim1 (whole-row): try PSWAP (involution) — the free latency win when the calibrated column
             * chain is palindromic (forcing a palindromic chain is a wash — its FFT slowdown offsets the
             * reorder win, natural_order §). dim2 (within-row): cycle only (fft2d.h scratch pass). */
            if (!d || !d->plan_col || !d->plan_row ||
                !vfft_natorder_2d_build_axis(N1, d->plan_col, &h->nat2d_row_list, &h->nat2d_row_is_pairs, 1) ||
                !vfft_natorder_2d_build_axis(N2, d->plan_row, &h->nat2d_col_list, &col_is_pairs, 0))
            {
                _vfft_warn("vfft_create: 2D %dx%d order=NATURAL — axis reorder-tape build "
                           "failed for this chain (orientation detect); the cell is "
                           "unsupported in natural order, use DEFAULT/SCRAMBLED",
                           N1, N2);
                vfft_destroy(h);
                return NULL;
            }
            /* dim1 MT bookkeeping: unit count + cycle start-offsets (for the per-worker range split),
             * mirroring the 1D natorder setup. NULL row tape = dim1 FREE (no reorder, no MT). */
            if (h->nat2d_row_list)
            {
                if (h->nat2d_row_is_pairs)
                    h->nat2d_ncyc = vfft_natorder_pair_count(h->nat2d_row_list);
                else
                {
                    h->nat2d_cyc_off = vfft_natorder_cycle_offsets(h->nat2d_row_list, &h->nat2d_ncyc);
                    if (!h->nat2d_cyc_off)
                    {
                        vfft_destroy(h);
                        return NULL;
                    }
                }
            }
            /* h->nthreads slots of 2*N2 doubles: one dim1 cycle-scratch slot per worker (+ main).
             * Sized by the PLAN'S SNAPSHOT, not the live pool -- the pool is grow-only, so
             * the live count here can be smaller than the one _natorder_2d sees at execute;
             * that side clamps by the same h->nthreads (natorder_mt.h), so the slot count
             * and the slot index come from one number. natorder_scratch_gate asserts this. */
            h->nat2d_tmp = (double *)malloc((size_t)(h->nthreads < 1 ? 1 : h->nthreads) * 2 * N2 * sizeof(double));
            if (!h->nat2d_tmp)
            {
                vfft_destroy(h);
                return NULL;
            }
            /* dim2 (within-row) is applied in the row-FFT scratch (mechanism-2): borrow the col tape
             * into the fft2d data. h owns the malloc (freed in vfft_destroy); _fft2d_destroy must NOT
             * free it. dim1 stays a whole-row pass in _natorder_2d. */
            d->nat_col_list = h->nat2d_col_list;
            h->nat2d = 1;
        }
        return h;
    }
    return NULL; /* unreachable: the one call site guards on the same
                  * condition, and every path in the block above returns. */
}

#endif /* VFFT_TRANSFORMS_FFT2D_CREATE_H */
