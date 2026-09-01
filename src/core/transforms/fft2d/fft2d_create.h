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

static vfft_plan _vfft_create_2d(const vfft_config_t *cfg,
                                 struct vfft_wisdom_s *W,
                                 const vfft_proto_registry_t *reg,
                                 size_t K)
{
    if (cfg->dims == 2)
    {
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
                    _pq_mt_race(h);
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
        int il2d_nst = 0;
        int il2d_wc = 0;
        int il2d_wl = 0, il2d_cut = 0, il2d_tfuse = 0;
        int il2d_rowoop = 0;
        struct vfft_plan_s *il2d_rowo = NULL;
        double *il2d_rowscr = NULL;
        int il2d_bwl = -1, il2d_btf = -1, il2d_bro = -1; /* banked axes */
        int il2d_staged = 0, il2d_pitch = 0;
        double *il2d_bandscr = NULL;
        double *il2d_rscr = NULL;
        struct vfft_plan_s *il2d_rows = NULL;
        int il2d_rw = 0;
        int il2d_brw = -1; /* banked row-route verdict; -1 = unraced */
        int il2d_oddn2 = 0;        /* odd-N2 real: c2c row child */
        double *il2d_orbuf = NULL; /* its 2 x 2*N2 row pair buffer  */
        int il2d_blu = 0;          /* odd/prime N1: column Bluestein M */
        int il2d_rof = 0;          /* row route FORCED oop (odd N2 c2c) */
        int il2d_nat = 0;          /* NATURAL n1 via the leaf redirection */
        int *il2d_natperm = NULL;
        double *il2d_natscr = NULL;
        int il2d_tbl_done = 0;     /* N1 tables built early (the N1-arm race) */
        double *il2d_bluchf = NULL, *il2d_bluchb = NULL;
        double *il2d_blukf = NULL, *il2d_blukb = NULL;
        double *il2d_bluscr = NULL;
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
            int chain_ok = 0;
            {
                /* chain precedence: env > banked lay=il verdict > RACE
                 * the full composition pool (multi-stage cells only;
                 * component-pinned: the race times the column pass, the
                 * only thing the axis changes) > greedy. */
                if (getenv("VFFT_IL2D_CHAIN"))
                    chain_ok = _il2d_build_chain(N1, il2d_R, il2d_f,
                                                 il2d_b, &il2d_nst);
                else if (vw2_2d_il_chain_lookup(&W->vw2, N1, N2, il2d_R,
                                                &il2d_nst, &il2d_bwl,
                                                &il2d_btf, &il2d_bro,
                                                &il2d_bcmt,
                                                &il2d_bcmtt) &&
                         _il2d_chain_prod(il2d_R, il2d_nst) == N1 &&
                         _il2d_resolve(il2d_R, il2d_nst, il2d_f, il2d_b))
                    chain_ok = 1;
                else
                {
                    int cand[VFFT_IL2D_MAXCAND][8], lens[VFFT_IL2D_MAXCAND];
                    int cur[8], ncand = 0, dropped = 0;
                    _il2d_enum_rec(N1, 0, cur, cand, lens, &ncand,
                                   &dropped);
                    if (dropped)
                        _vfft_warn("il2d chain race: pool capped at %d "
                                   "(%d candidate(s) dropped) at %dx%d",
                                   VFFT_IL2D_MAXCAND, dropped, N1, N2);
                    if (ncand > 1)
                    {
                        double bns = 0;
                        int win = _il2d_race_chains(N1, N2, ncand, cand,
                                                    lens, &bns);
                        if (win >= 0 &&
                            _il2d_resolve(cand[win], lens[win], il2d_f,
                                          il2d_b))
                        {
                            memcpy(il2d_R, cand[win],
                                   sizeof cand[win]);
                            il2d_nst = lens[win];
                            chain_ok = 1;
                            vw2_2d_il_chain_bank(&W->vw2, N1, N2,
                                                 il2d_R, il2d_nst,
                                                 -1, -1, -1, -1, -1,
                                                 bns);
                            _vw2_persist(W, cfg);
                        }
                    }
                    if (!chain_ok)
                        chain_ok = _il2d_build_chain(N1, il2d_R, il2d_f,
                                                     il2d_b, &il2d_nst);
                }
            }
            if (!chain_ok)
            {
                /* ODD/PRIME N1: the COLUMN-AXIS BLUESTEIN (struct
                 * comment at il2d_blu; _il2d_blu_build). Reached only
                 * when no chain exists — with the odd t2c/n1c kinds
                 * emitted, that now means prime / unexpressible N1.
                 * n1 comes out NATURAL by construction, so ALL order
                 * spellings are served (M4-lite closed the old
                 * DEFAULT-only gate 2026-08-27). */
                il2d_blu = _il2d_blu_build(N1, (size_t)N2, il2d_R,
                                           il2d_L, il2d_f, il2d_b,
                                           il2d_tf, il2d_tb, &il2d_nst,
                                           &il2d_bluchf, &il2d_bluchb,
                                           &il2d_blukf, &il2d_blukb,
                                           &il2d_bluscr);
                if (il2d_blu)
                    chain_ok = 1;
            }
            else if (chain_ok && cfg->order != VFFT_ORDER_NATURAL &&
                     !getenv("VFFT_IL2D_CHAIN"))
            {
                /* THE RACED CHAIN ARM (owner directive): for a chain
                 * that carries an ODD radix (the newly emitted kinds),
                 * race it against the Bluestein column route — the two
                 * serve DIFFERENT n1 orders (chain = scrambled comb,
                 * blu = natural), and both are self-consistent, so the
                 * pick is pure speed. Env VFFT_IL2D_BLU=1 pins blu,
                 * =0 pins the chain (env never banks); unset = race
                 * min-of-3 alternated on scratch through the SERVING
                 * functions. pow2 chains never race (blu is pointless
                 * there). Verdict plan-local (the wisdom banking of a
                 * blu marker rides the layout-audit wave). */
                int hasodd = 0, s3;
                const char *be = getenv("VFFT_IL2D_BLU");
                for (s3 = 0; s3 < il2d_nst; s3++)
                    if (il2d_R[s3] & 1)
                        hasodd = 1;
                /* the chain arm times the SERVING column pass, so the
                 * N1 tables must exist BEFORE the race (they are
                 * otherwise built at the row-child block below — timing
                 * with empty tabs was a NULL-load crash, caught by the
                 * cell sweep 2026-08-27). il2d_tbl_done stops the later
                 * shared build from double-building the winner's. */
                if (hasodd && (!be || atoi(be) == 1) &&
                    !_il2d_build_tables(N1, il2d_nst, il2d_R, il2d_L,
                                        il2d_tf, il2d_tb))
                {
                    il2d_tbl_done = 1;
                    int bR[8], bL[8], bnst = 0, M2;
                    vfft_il2p_fn bf[8], bb[8];
                    double *btf[8], *btb[8];
                    double *bchf, *bchb, *bkf, *bkb, *bscr;
                    memset(btf, 0, sizeof btf);
                    memset(btb, 0, sizeof btb);
                    M2 = _il2d_blu_build(N1, (size_t)N2, bR, bL, bf, bb,
                                         btf, btb, &bnst, &bchf, &bchb,
                                         &bkf, &bkb, &bscr);
                    if (M2)
                    {
                        double *sc = (double *)malloc(
                            2 * (size_t)N1 * N2 * sizeof(double));
                        double tc = 1e300, tbu = 1e300;
                        int rr, use_blu = (be != NULL); /* env pin */
                        size_t i3;
                        if (sc && !use_blu)
                        {
                            for (i3 = 0; i3 < 2 * (size_t)N1 * N2; i3++)
                                sc[i3] = 1.0 + 1e-6 * (double)(i3 & 511);
                            for (rr = 0; rr < 3; rr++)
                            {
                                struct timespec t0, t1;
                                double d;
                                clock_gettime(CLOCK_MONOTONIC, &t0);
                                _il2d_col_pass(sc, sc, N1, (size_t)N2,
                                               (size_t)N2, il2d_nst,
                                               il2d_R, il2d_L, il2d_f,
                                               il2d_tf, 0);
                                clock_gettime(CLOCK_MONOTONIC, &t1);
                                d = (t1.tv_sec - t0.tv_sec) * 1e9
                                    + (t1.tv_nsec - t0.tv_nsec);
                                if (d < tc)
                                    tc = d;
                                clock_gettime(CLOCK_MONOTONIC, &t0);
                                _il2d_blu_cols(sc, sc, N1, (size_t)N2,
                                               M2, bnst, bR, bL, bf, bb,
                                               btf, btb, bchf, bkf,
                                               bscr);
                                clock_gettime(CLOCK_MONOTONIC, &t1);
                                d = (t1.tv_sec - t0.tv_sec) * 1e9
                                    + (t1.tv_nsec - t0.tv_nsec);
                                if (d < tbu)
                                    tbu = d;
                            }
                        }
                        free(sc);
                        if (!use_blu)
                            use_blu = (tbu < tc);
                        if (getenv("VFFT_IL2D_LOG"))
                            fprintf(stderr, "[il2d] N1-arm race %dx%d: "
                                            "chain=%.0f blu=%.0f -> %s\n",
                                    N1, N2, tc, tbu,
                                    use_blu ? "BLUESTEIN" : "chain");
                        if (use_blu)
                        {
                            for (s3 = 0; s3 < il2d_nst; s3++)
                            {
                                free(il2d_tf[s3]);
                                free(il2d_tb[s3]);
                            }
                            memcpy(il2d_R, bR, sizeof bR);
                            memcpy(il2d_L, bL, sizeof bL);
                            memcpy(il2d_f, bf, sizeof bf);
                            memcpy(il2d_b, bb, sizeof bb);
                            memcpy(il2d_tf, btf, sizeof btf);
                            memcpy(il2d_tb, btb, sizeof btb);
                            il2d_nst = bnst;
                            il2d_blu = M2;
                            il2d_bluchf = bchf;
                            il2d_bluchb = bchb;
                            il2d_blukf = bkf;
                            il2d_blukb = bkb;
                            il2d_bluscr = bscr;
                        }
                        else
                        {
                            for (s3 = 0; s3 < bnst; s3++)
                            {
                                free(btf[s3]);
                                free(btb[s3]);
                            }
                            free(bchf); free(bchb);
                            free(bkf); free(bkb); free(bscr);
                        }
                    }
                }
                else if (hasodd && be && atoi(be) == 0)
                    ; /* env pins the chain: nothing to do */
            }
            if (!chain_ok)
            {
                /* OWNER LAW: split is NOT a fallback of IL — no convert
                 * wrapper. An inexpressible N1 refuses loudly. */
                _vfft_warn("vfft_create: IL 2D c2c %dx%d — N1 has no "
                           "native column chain (radices 4..64, no "
                           "leftover factor)%s",
                           N1, N2,
                           cfg->order == VFFT_ORDER_NATURAL
                               ? " and the Bluestein column route "
                                 "serves DEFAULT order only"
                               : " and the Bluestein column route "
                                 "could not be built");
                return NULL;
            }
            if (cfg->order == VFFT_ORDER_NATURAL && il2d_nst > 1 &&
                !il2d_blu)
            {
                /* M4-lite (2026-08-27, struct comment at il2d_nat):
                 * natural n1 via the LEAF REDIRECTION — driver-only,
                 * any chain. The perm builder settles the digit
                 * convention empirically and refuses on any mismatch. */
                il2d_natperm = _il2d_nat_perm(il2d_R, il2d_nst, N1);
                if (il2d_natperm)
                    il2d_natscr = (double *)malloc(
                        2 * (size_t)N1 * N2 * sizeof(double));
                if (!il2d_natperm || !il2d_natscr)
                {
                    free(il2d_natperm);
                    il2d_natperm = NULL;
                    _vfft_warn("vfft_create: IL 2D c2c %dx%d "
                               "order=NATURAL — the natural leaf "
                               "permutation could not be built for "
                               "this chain; unsupported",
                               N1, N2);
                    return NULL;
                }
                il2d_nat = 1;
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
                if (!il2d_row)
                {
                    /* no IN-PLACE K=1 route at this N2 (odd/awkward N2
                     * — 129 = 3*43 serves OOP-only via the prime
                     * engine): fall back to the tier's OWN rowoop
                     * mechanism — the OOP child + row scratch + copy-
                     * back that _il2d_row_exec already serves. il2d_row
                     * aliases the OOP child as the dispatch sentinel
                     * (never executed directly when rowoop is set);
                     * destroy skips the alias. The row route is FORCED
                     * here, so the axis race must not flip it. */
                    rc.placement = VFFT_OUTOFPLACE;
                    il2d_rowo = (struct vfft_plan_s *)vfft_create(&rc);
                    if (il2d_rowo)
                    {
                        il2d_rowscr = (double *)malloc(
                            2 * (size_t)N2 * sizeof(double));
                        if (il2d_rowscr)
                        {
                            il2d_rowoop = 1;
                            il2d_rof = 1;
                            il2d_row = il2d_rowo;
                        }
                        else
                        {
                            vfft_destroy(il2d_rowo);
                            il2d_rowo = NULL;
                        }
                    }
                }
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
                /* column-tile width: env override (raced axis; wisdom
                 * banking follows the falsifier run — tcut precedent:
                 * env BEATS wisdom). 0/absent/invalid = untiled. */
                {
                    const char *wce = getenv("VFFT_IL2D_WC");
                    il2d_wc = (wce && atoi(wce) > 0 && atoi(wce) < N2)
                                  ? atoi(wce)
                                  : 0;
                }
                /* row route: VFFT_IL2D_ROWOOP=1 swaps the per-row
                 * child for an OOP NATURAL one + scratch (the mono
                 * route). Falls back to the in-place child if the OOP
                 * create or the scratch fails. */
                if (il2d_row && !getenv("VFFT_IL2D_ROWOOP") &&
                    il2d_bro == 1)
                {
                    /* banked row-route verdict (env silent): build the
                     * OOP child; on failure fall back to in-place. */
                    vfft_config_t ro;
                    memset(&ro, 0, sizeof ro);
                    ro.transform = VFFT_C2C;
                    ro.placement = VFFT_OUTOFPLACE;
                    ro.rigor = cfg->rigor;
                    ro.dims = 1;
                    ro.n[0] = N2;
                    ro.howmany = 1;
                    ro.order = VFFT_ORDER_NATURAL;
                    ro.layout = VFFT_LAYOUT_INTERLEAVED;
                    ro.nthreads = 1;
                    ro.wisdom = cfg->wisdom;
                    ro.wisdom_write = cfg->wisdom_write;
                    il2d_rowo = (struct vfft_plan_s *)vfft_create(&ro);
                    if (il2d_rowo)
                    {
                        il2d_rowscr = (double *)malloc(
                            2 * (size_t)N2 * sizeof(double));
                        if (il2d_rowscr)
                            il2d_rowoop = 1;
                        else
                        {
                            vfft_destroy(il2d_rowo);
                            il2d_rowo = NULL;
                        }
                    }
                }
                if (il2d_row && getenv("VFFT_IL2D_ROWOOP") &&
                    atoi(getenv("VFFT_IL2D_ROWOOP")) == 1)
                {
                    vfft_config_t ro;
                    memset(&ro, 0, sizeof ro);
                    ro.transform = VFFT_C2C;
                    ro.placement = VFFT_OUTOFPLACE;
                    ro.rigor = cfg->rigor;
                    ro.dims = 1;
                    ro.n[0] = N2;
                    ro.howmany = 1;
                    ro.order = VFFT_ORDER_NATURAL;
                    ro.layout = VFFT_LAYOUT_INTERLEAVED;
                    ro.nthreads = 1;
                    ro.wisdom = cfg->wisdom;
                    ro.wisdom_write = cfg->wisdom_write;
                    il2d_rowo = (struct vfft_plan_s *)vfft_create(&ro);
                    if (il2d_rowo)
                    {
                        il2d_rowscr = (double *)malloc(
                            2 * (size_t)N2 * sizeof(double));
                        if (il2d_rowscr)
                            il2d_rowoop = 1;
                        else
                        {
                            vfft_destroy(il2d_rowo);
                            il2d_rowo = NULL;
                        }
                    }
                }
                /* staged band route: VFFT_IL2D_STAGED=1 (needs a
                 * band; checked after the wl parse below). */
                /* banded walk: VFFT_IL2D_WL = band width in ROWS (the
                 * width is the INPUT, the cut is DERIVED — the tcut law).
                 * Legal iff wl | N1 and some suffix stage has L_s | wl;
                 * anything else warns and stays unbanded. VFFT_IL2D_TFUSE
                 * =0 opts out of the per-band row pass (default ON when
                 * banded — the fusion is the point). */
                if (il2d_row && !il2d_blu && !il2d_nat)
                {
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
                    if (il2d_wl > 0 && getenv("VFFT_IL2D_STAGED") &&
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
                /* chain precedence: env > the banked lay=il real cell
                 * (direction-shared, keyed t=r2c ord=scr — the pair law
                 * requires one chain for both directions) > greedy. The
                 * banked row also carries rw= (the row-route verdict). */
                if (getenv("VFFT_IL2D_CHAIN"))
                    rok = _il2d_build_chain(N1, il2d_R, il2d_f, il2d_b,
                                            &il2d_nst);
                else if (!cfg->recalibrate &&
                         vw2_2d_rl_lookup(&W->vw2, N1, N2, il2d_R,
                                          &il2d_nst, &il2d_brw,
                                          &il2d_bwl, &il2d_bcmt,
                                          &il2d_bcmtt) &&
                         _il2d_chain_prod(il2d_R, il2d_nst) == N1 &&
                         _il2d_resolve(il2d_R, il2d_nst, il2d_f,
                                       il2d_b))
                    rok = 1;
                else
                    rok = _il2d_build_chain(N1, il2d_R, il2d_f, il2d_b,
                                            &il2d_nst);
            }
            if (!rok)
            {
                /* PRIME/unexpressible N1 for the REAL tier: the same
                 * column-axis Bluestein, over the hp1-wide CCE plane
                 * (rn = hp1 — the pipeline is C-linear over any count).
                 * n1 comes out NATURAL on this route; wl/rw/colmt races
                 * are skipped (guards below). */
                il2d_blu = _il2d_blu_build(N1, (size_t)N2 / 2 + 1,
                                           il2d_R, il2d_L, il2d_f,
                                           il2d_b, il2d_tf, il2d_tb,
                                           &il2d_nst, &il2d_bluchf,
                                           &il2d_bluchb, &il2d_blukf,
                                           &il2d_blukb, &il2d_bluscr);
                if (il2d_blu)
                    rok = 1;
            }
            if (rok && !il2d_blu &&
                _il2d_build_tables(N1, il2d_nst, il2d_R, il2d_L,
                                   il2d_tf, il2d_tb))
                rok = 0;
            if (rok && !il2d_blu && il2d_nst > 1 &&
                cfg->order == VFFT_ORDER_NATURAL)
            {
                il2d_natperm = _il2d_nat_perm(il2d_R, il2d_nst, N1);
                if (il2d_natperm)
                    il2d_natscr = (double *)malloc(
                        2 * (size_t)N1 * ((size_t)N2 / 2 + 1)
                        * sizeof(double));
                if (!il2d_natperm || !il2d_natscr)
                {
                    free(il2d_natperm);
                    il2d_natperm = NULL;
                    _vfft_warn("vfft_create: IL 2D %s %dx%d "
                               "order=NATURAL — the natural leaf "
                               "permutation could not be built; "
                               "unsupported",
                               _vfft_tname(cfg->transform), N1, N2);
                    return NULL;
                }
                il2d_nat = 1;
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
        h->il2d_nst = il2d_nst;
        h->il2d_wc = il2d_wc;
        h->il2d_wl = il2d_wl;
        h->il2d_cut = il2d_cut;
        h->il2d_tfuse = il2d_tfuse;
        h->il2d_rowoop = il2d_rowoop;
        h->il2d_rowo = il2d_rowo;
        h->il2d_rowscr = il2d_rowscr;
        h->il2d_staged = il2d_staged;
        h->il2d_pitch = il2d_pitch;
        h->il2d_bandscr = il2d_bandscr;
        h->il2d_rscr = il2d_rscr;
        h->il2d_rows = il2d_rows;
        h->il2d_rw = il2d_rw;
        h->il2d_oddn2 = il2d_oddn2;
        h->il2d_orbuf = il2d_orbuf;
        h->il2d_nat = il2d_nat;
        h->il2d_natperm = il2d_natperm;
        h->il2d_natscr = il2d_natscr;
        h->il2d_blu = il2d_blu;
        h->il2d_bluchf = il2d_bluchf;
        h->il2d_bluchb = il2d_bluchb;
        h->il2d_blukf = il2d_blukf;
        h->il2d_blukb = il2d_blukb;
        h->il2d_bluscr = il2d_bluscr;
        /* A/B race knob (struct comment): create-time env read only. */
        h->il2d_norowz = getenv("VFFT_IL2D_NO_ROWZ") != NULL;
        h->il2d_lx = il2d_lx;
        h->il2d_lre = il2d_lre;
        h->il2d_lim = il2d_lim;
        h->il2d_tre = il2d_tre;
        h->il2d_tim = il2d_tim;
        memcpy(h->il2d_R, il2d_R, sizeof il2d_R);
        memcpy(h->il2d_L, il2d_L, sizeof il2d_L);
        memcpy(h->il2d_f, il2d_f, sizeof il2d_f);
        memcpy(h->il2d_b, il2d_b, sizeof il2d_b);
        memcpy(h->il2d_tf, il2d_tf, sizeof il2d_tf);
        memcpy(h->il2d_tb, il2d_tb, sizeof il2d_tb);
        /* ── the AXIS RACE (§10a): wl and rowoop timed on the FULL
         * execute (they involve the rows), the winner set on the plan
         * and banked WITH the chain as one verdict. Runs only when the
         * axes are unknown: no env override and no banked verdict.
         * MUST sit AFTER the stage-array commits above — it executes h.
         * c2c ONLY: the real tier has no banded walk / row route to race
         * (§2.5 — banding+tfuse on a real plan is the illegal fusion). */
        if (h->transform == VFFT_C2C && h->il2d_row && !il2d_blu &&
            !il2d_rof && !il2d_nat &&
            !getenv("VFFT_IL2D_WL") &&
            !getenv("VFFT_IL2D_ROWOOP") && !getenv("VFFT_IL2D_TFUSE") &&
            (il2d_bwl < 0 || il2d_bro < 0))
            _il2d_axis_race(h, W, cfg, N1, N2);
        /* INC-C: c2c MT. Build the per-worker row clones (the serving
         * row path mutates shared plan state), then serve the banked
         * cmt verdict ONLY at the T it was raced at, else race and
         * bank. Runs AFTER the axis race — the row route (rowoop) the
         * clones must match is final only then. */
        if (h->transform == VFFT_C2C && h->il2d_row && !il2d_blu &&
            !il2d_nat && h->nthreads > 1)
        {
            const char *ce = getenv("VFFT_IL2D_NO_COLMT");
            _il2d_c2c_build_clones(h, cfg, h->nthreads);
            if (ce)
                h->il2d_colmt = (atoi(ce) == 0);
            else if (il2d_bcmt >= 0 && il2d_bcmtt == h->nthreads)
                h->il2d_colmt = il2d_bcmt;
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
        /* INC-3: the column-MT verdict. Serve a banked one ONLY when it
         * was raced at THIS thread count; otherwise race and bank. A
         * single-threaded plan never threads columns and never races. */
        if ((h->transform == VFFT_R2C || h->transform == VFFT_C2R) &&
            h->il2d_row && !il2d_blu && !il2d_nat && h->nthreads > 1)
        {
            const char *ce = getenv("VFFT_IL2D_NO_COLMT");
            if (ce)
                h->il2d_colmt = (atoi(ce) == 0);
            else if (il2d_bcmt >= 0 && il2d_bcmtt == h->nthreads)
                h->il2d_colmt = il2d_bcmt;
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
