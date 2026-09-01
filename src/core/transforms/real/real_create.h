/* real_create.h — the r2c / c2r CREATE tier (migration step 26).
 *
 * WHAT THIS IS
 * ------------
 * The three real-transform arms of _vfft_create_inner, in their original
 * order. Each returns on every path, so the group lifts out behind one guard:
 *
 *   1. the ODD-N BRIDGE — odd N, K==1, out-of-place. Builds the transform on
 *      a c2c child (_oddr_build) rather than on a real codelet, and refuses
 *      LOUDLY when that child cannot be built. This arm is why odd and prime
 *      real sizes are served at all; before it they were a silent refusal.
 *      r2c takes the bridge only when N is NOT radix-smooth (a smooth odd N is
 *      better served by rfft), while c2r takes it unconditionally — the two
 *      directions do not have the same incumbent.
 *   2. r2c;
 *   3. c2r.
 *
 * BOTH REAL DIRECTIONS ARE A 2-AXIS CHOICE
 * ----------------------------------------
 * NATURAL (the fast packed cascade run on split input via the stage-0 natural
 * initiator — no repack, the low/mid-K winner) vs STRIDE (decoupled, the
 * high-K and threaded winner). Both consume split re/im, so the pick is
 * invisible to the caller.
 *
 * 🔴 THE CROSSOVER IS ON K, NOT N. natural's win is non-monotonic in K, so a
 * fixed threshold cannot capture it: at high rigor the tier MEASURES both arms
 * over the contested low/mid-K zone; otherwise it reads wisdom first and only
 * then falls back to a threshold. No forced path, no hardcode.
 *
 * WHAT THIS TIER DOES NOT DECIDE
 * ------------------------------
 * The route race itself lives in transforms/real/real_route_race.h — those are
 * racers, not deciders. This tier is what calls them and what banks the
 * verdict.
 *
 * POSITION IN vfft.c IS LOAD-BEARING
 * ----------------------------------
 * Not a standalone header. It calls file-scope statics that live in vfft.c
 * (_oddr_build among them), so it must be included after those are defined and
 * before _vfft_create_inner.
 *
 * The six parameters are the union of the three blocks' free variables,
 * derived rather than guessed: cfg, ob, W, reg, N, K.
 */
#ifndef VFFT_TRANSFORMS_REAL_CREATE_H
#define VFFT_TRANSFORMS_REAL_CREATE_H

/* the two arms of the smooth-odd bridge race: two finished handles */
typedef struct { struct vfft_plan_s *h; double *xr, *zr; } _oddr_arm_t;
static void _oddr_arm_exec(void *v)
{
    _oddr_arm_t *c = (_oddr_arm_t *)v;
    vfft_execute((vfft_plan)c->h, VFFT_FORWARD, c->xr, NULL, c->zr, NULL);
}
static vfft_plan _vfft_create_real(const vfft_config_t *cfg,
                                   vfft_batch ob,
                                   struct vfft_wisdom_s *W,
                                   const vfft_proto_registry_t *reg,
                                   int N,
                                   size_t K)
{
    if ((cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R) &&
        K == 1 && (N & 1) && N >= 3 &&
        cfg->placement == VFFT_OUTOFPLACE &&
        (cfg->transform == VFFT_C2R || !_vfft_is_radix_smooth(N) ||
         getenv("VFFT_ODDR_FORCE") != NULL))
    {
        struct vfft_plan_s *hh = _oddr_build(cfg, N);
        if (hh)
            return hh;
        _vfft_warn("vfft_create: %s odd N=%d - the c2c bridge child "
                   "could not be built; unsupported",
                   _vfft_tname(cfg->transform), N);
        return NULL;
    }
    if (cfg->transform == VFFT_R2C)
    {
        /* PADDED (opt-in): a Kp-wide handle -> build the plan at Kp (the ORDINARY aligned
         * (N,Kp) rfft cell — full-SIMD, no tail) so it strides the caller's Kp-wide buffers
         * exactly. r2c/c2r executors bake K with no runtime `me`, so a K-plan can't run the
         * tail on a Kp-strided buffer -> padded mode is pad-ONLY (the wisdom is unchanged; no
         * exec_me verdict). Payoff lives in the cascade regime (small Kp<32); a Kp that routes
         * to the K%8-gated stride path simply yields NULL (padding unsupported for that cell,
         * caller falls back to the tight tail). */
        size_t bK = K; /* build width: Kp when padded, else K */
        int padded = 0;
        if (ob)
        {
            vfft_batch b = ob;
            if (b->xform != (int)VFFT_R2C || b->N != N || b->K != K)
            { /* handle must match the descriptor exactly */
                _vfft_warn("vfft_create: config.batch does not match this R2C descriptor "
                           "(batch: %s N=%d K=%zu; config: R2C N=%d K=%zu) — allocate with "
                           "vfft_alloc_batch_for(THIS config)",
                           _vfft_tname(b->xform), b->N, b->K, N, K);
                return NULL;
            }
            bK = b->Kp;
            padded = 1;
        }
        /* §D2 zr2c route: even N, K==1, INTERLEAVED — reinterpret + child
         * c2c(N/2) + fold. Also the ONLY in-place real path (the in-place
         * refusal above admits exactly this combo). K>1 keeps the
         * split-interior CCE path below; the batched composite is the V9
         * workstream. This branch runs BEFORE the split-path calibrate-on-
         * miss blocks below on purpose: a zr2c-served cell must not pay for
         * (or bank) c2c(N/2, K)/rfft rows it never reads — the child rides
         * the K=1 engine tables through its own recursive create. Child-
         * create failure falls through to the split path, which then
         * calibrates exactly as before. */
        if (cfg->layout == VFFT_LAYOUT_INTERLEAVED && K == 1 && (N % 2) == 0 && !ob)
        {
            struct vfft_plan_s *hz = _zr2c_build(cfg, N, W);
            if (hz)
                return hz;
            /* 🔴 NO SILENT DEGRADE TO OUT-OF-PLACE. The in-place refusal
             * above ADMITTED this shape, so falling through would stamp
             * h->placement = INPLACE onto a handle whose executor is the OOP
             * CCE path -- engines that stream an N-double real plane into an
             * N+2-double CCE plane and were never gated for aliasing. The
             * caller then makes the documented (z,NULL,z,NULL) call and gets
             * an out-of-place executor whose source aliases its destination.
             * zr2c is the ONLY in-place real path, so if it could not be
             * built there is no in-place plan to give: refuse loudly.
             * Out-of-place callers keep the fall-through unchanged. */
            if (cfg->placement == VFFT_INPLACE)
            {
                _vfft_warn("vfft_create: in-place %s N=%d could not build the zr2c route "
                           "(the only in-place real path); no out-of-place fallback exists "
                           "for an in-place plan -- use VFFT_OUTOFPLACE",
                           _vfft_tname(cfg->transform), N);
                return NULL;
            }
        }
        /* The r2c dispatcher rides the c2c wisdom for its decoupled inner FFT and
         * the rfft wisdom for the rfft path; it auto-threads (sub-K block) when the
         * pool is sized >1 at create. Calibrate-on-miss for the inner cell ensures
         * `rigor` reaches the dominant work (the inner c2c). */
        {
            vfft_proto_wisdom_entry_t neb;
            int have = !cfg->recalibrate &&
                (W->vw2_off_stride
                     ? (vfft_proto_wisdom_lookup(&W->c2c, N / 2, bK) != NULL)
                     : vw2_stride_lookup(&W->vw2, 0, N / 2, bK, &neb));
            if (have && !W->vw2_off_stride)
                vfft_proto_wisdom_set(&W->c2c, &neb);
            if (!have && (N % 2) == 0 &&
                _calibrate_c2c(N / 2, bK, cfg->rigor, reg, &neb) == 0)
            {
                vfft_proto_wisdom_add(&W->c2c, &neb, 1);
                vw2_stride_bank_entry(&W->vw2, &neb, 0);
                _vw2_persist(W, cfg);
            }
        }
        /* rfft axis: the rfft PATH (low K, and odd/prime/fallback cells) picks a
         * factorization + per-stage variant. Calibrate-on-miss so `rigor` reaches the
         * rfft side too, not just the fewest-stage heuristic. Only worth it in the rfft
         * regime (K at/below the decouple crossover); the stride path owns high K and
         * ignores rfft wisdom. The rfft search space is small → the sweep is exhaustive
         * + fast at any rigor (it's the calibrate-at-all that closes the gap). */
        if (bK <= 64)
        {
            vfft_proto_wisdom_entry_t rfe;
            int have = !cfg->recalibrate &&
                (W->vw2_off_stride
                     ? (vfft_proto_wisdom_lookup(&W->rfft, N, bK) != NULL)
                     : vw2_stride_lookup(&W->vw2, /*is_rfft=*/1, N, bK, &rfe));
            if (have && !W->vw2_off_stride)
                vfft_proto_wisdom_set(&W->rfft, &rfe);
            if (!have && vfft_rfft_calibrate(N, bK, _rfft_registry(), &rfe) == 0)
            {
                vfft_proto_wisdom_add(&W->rfft, &rfe, 1);
                vw2_stride_bank_entry(&W->vw2, &rfe, /*is_rfft=*/1);
                _vw2_persist(W, cfg);
            }
        }
        vfft_r2c_dispatch_set_c2c_wisdom(&W->c2c);
        vfft_r2c_dispatch_set_wisdom(&W->rfft);
        /* Route axis (§W2). A BANKED verdict serves at every rigor tier; the
         * race that produces one is confined to the rfft-competitive zone
         * (K<=64, N even, not MEASURE), and MEASURE / high-K fall through to
         * the fixed-threshold dispatch exactly as before. */
        vfft_r2c_plan_t *rp =
            /* bK > 1: the route race is a LANE-BATCH question and the
             * split engine has no K=1 batch (owner law 2026-08-24: K counts
             * the FFTs running; split lanes hold independent FFTs). At K=1
             * the structural default serves — racing there would re-race on
             * every create with nowhere legal to bank. q=1 real cells
             * belong to the interleaved zr2c verdicts alone. */
            _r2c_route_decide(W, cfg, N, bK, reg,
                              cfg->rigor != VFFT_MEASURE && (N % 2) == 0 &&
                                  bK > 1 && bK <= 64);
        if (!rp)
            return NULL;
        struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
        if (!h)
        {
            vfft_r2c_plan_destroy(rp);
            return NULL;
        }
        h->transform = VFFT_R2C;
        h->placement = cfg->placement;
        h->layout = (int)cfg->layout; /* INTERLEAVED == the packed CCE spectrum contract */
        h->N = N;
        h->K = K;
        h->nthreads = _vfft_plan_threads(cfg);
        h->rplan = rp;
        h->padded = padded;
        h->exec_me = (int)bK; /* informational: the width the plan was built at */
        /* SMOOTH-ODD r2c: race this (rfft-served) handle against the
         * c2c bridge - both arms FINISHED handles (the strawman law),
         * min-of-3 alternated, loser destroyed. Winner flips per cell
         * (the pricing). K==1 OOP IL only; verdict plan-local. */
        if (K == 1 && (N & 1) && N >= 3 &&
            cfg->placement == VFFT_OUTOFPLACE &&
            cfg->layout == VFFT_LAYOUT_INTERLEAVED &&
            !getenv("VFFT_ODDR_NORACE"))
        {
            struct vfft_plan_s *hb = _oddr_build(cfg, N);
            if (hb)
            {
                const size_t hp1r = (size_t)N / 2 + 1;
                double *xr = (double *)malloc((size_t)N
                                              * sizeof(double));
                double *zr2 = (double *)calloc(2 * (hp1r + 8),
                                               sizeof(double));
                double ta = 1e300, tb2 = 1e300;
                if (xr && zr2)
                {
                    int r2, j2;
                    for (j2 = 0; j2 < N; j2++)
                        xr[j2] = 1.0 + 1e-6 * (double)(j2 & 511);
                    vfft_execute((vfft_plan)h, VFFT_FORWARD, xr, NULL,
                                 zr2, NULL);
                    vfft_execute((vfft_plan)hb, VFFT_FORWARD, xr, NULL,
                                 zr2, NULL);
                    {
                        _oddr_arm_t ca = { h, xr, zr2 }, cb = { hb, xr, zr2 };
                        const vfft_race_arm_t arms[2] = {
                            { "rfft", _oddr_arm_exec, &ca },
                            { "bridge", _oddr_arm_exec, &cb } };
                        const vfft_race_proto_t proto = { 3, 1, VFFT_RACE_MIN, 0, 0, NULL, NULL }; /* min-of-3, A then B */
                        double ns[2];
                        (void)r2;
                        vfft_race_run(&proto, arms, 2, ns);
                        ta = ns[0];
                        tb2 = ns[1];
                    }
                    if (getenv("VFFT_ODDR_LOG"))
                        fprintf(stderr, "[oddr] race N=%d: rfft=%.0f "
                                        "bridge=%.0f -> %s\n",
                                N, ta, tb2,
                                tb2 < ta ? "BRIDGE" : "rfft");
                }
                free(xr);
                free(zr2);
                if (tb2 < ta)
                {
                    vfft_destroy((vfft_plan)h);
                    return hb;
                }
                vfft_destroy((vfft_plan)hb);
            }
        }
        return h;
    }

    /* ── c2r (complex -> real; the r2c inverse), SPLIT input (sre/sim). 2-axis,
     * mirroring r2c: NATURAL (the fast packed cascade run on split input via the
     * stage-0 natural initiator — no repack, low/mid-K winner) vs STRIDE (decoupled,
     * high-K + threads). BOTH consume split re/im, so the pick is transparent to the
     * caller. High rigor MEASURES both at create over the contested low/mid-K zone
     * (natural's win is non-monotonic in K — a fixed threshold can't capture it);
     * else wisdom-first (c2r_path.txt) then threshold. No forced path / no hardcode. ── */
    if (cfg->transform == VFFT_C2R)
    {
        if ((N % 2) != 0)
        {
            _vfft_warn("vfft_create: C2R odd N=%d — served at K==1 "
                       "OUT-OF-PLACE (the c2c bridge); this shape "
                       "(K=%zu, placement=%d) is unsupported",
                       N, K, (int)cfg->placement);
            return NULL;
        }
        /* PADDED (opt-in): build at Kp (ordinary aligned (N,Kp) c2r cell) so the plan strides
         * the caller's Kp-wide split-input / real-output buffers exactly. Pad-only (see the r2c
         * branch: baked-K executors, no runtime `me`); wisdom unchanged; cascade regime. */
        size_t bK = K;
        int padded = 0;
        if (ob)
        {
            vfft_batch b = ob;
            if (b->xform != (int)VFFT_C2R || b->N != N || b->K != K)
            {
                _vfft_warn("vfft_create: config.batch does not match this C2R descriptor "
                           "(batch: %s N=%d K=%zu; config: C2R N=%d K=%zu) — allocate with "
                           "vfft_alloc_batch_for(THIS config)",
                           _vfft_tname(b->xform), b->N, b->K, N, K);
                return NULL;
            }
            bK = b->Kp;
            padded = 1;
        }
        /* §D2 zr2c route (mirror of the r2c branch): even N, K==1,
         * INTERLEAVED CCE input — fold + child c2c(N/2) backward. */
        if (cfg->layout == VFFT_LAYOUT_INTERLEAVED && K == 1 && (N % 2) == 0 && !ob)
        {
            struct vfft_plan_s *hz = _zr2c_build(cfg, N, W);
            if (hz)
                return hz;
            /* 🔴 NO SILENT DEGRADE TO OUT-OF-PLACE. The in-place refusal
             * above ADMITTED this shape, so falling through would stamp
             * h->placement = INPLACE onto a handle whose executor is the OOP
             * CCE path -- engines that stream an N-double real plane into an
             * N+2-double CCE plane and were never gated for aliasing. The
             * caller then makes the documented (z,NULL,z,NULL) call and gets
             * an out-of-place executor whose source aliases its destination.
             * zr2c is the ONLY in-place real path, so if it could not be
             * built there is no in-place plan to give: refuse loudly.
             * Out-of-place callers keep the fall-through unchanged. */
            if (cfg->placement == VFFT_INPLACE)
            {
                _vfft_warn("vfft_create: in-place %s N=%d could not build the zr2c route "
                           "(the only in-place real path); no out-of-place fallback exists "
                           "for an in-place plan -- use VFFT_OUTOFPLACE",
                           _vfft_tname(cfg->transform), N);
                return NULL;
            }
        }
        /* the STRIDE inner is a c2c(N/2): calibrate-on-miss so it rides c2c wisdom
         * (NATURAL uses the rfft/c2r codelets directly — no inner c2c). */
        {
            vfft_proto_wisdom_entry_t neb;
            int have = !cfg->recalibrate &&
                (W->vw2_off_stride
                     ? (vfft_proto_wisdom_lookup(&W->c2c, N / 2, bK) != NULL)
                     : vw2_stride_lookup(&W->vw2, 0, N / 2, bK, &neb));
            if (have && !W->vw2_off_stride)
                vfft_proto_wisdom_set(&W->c2c, &neb);
            if (!have && _calibrate_c2c(N / 2, bK, cfg->rigor, reg, &neb) == 0)
            {
                vfft_proto_wisdom_add(&W->c2c, &neb, 1);
                vw2_stride_bank_entry(&W->vw2, &neb, 0);
                _vw2_persist(W, cfg);
            }
        }
        vfft_r2c_dispatch_set_c2c_wisdom(&W->c2c);
        /* Route axis (§W2) — see the r2c site. A banked verdict serves at
         * every rigor tier; only the race is window-confined. */
        vfft_c2r_disp_t *cd =
            _c2r_route_decide(W, cfg, N, bK, reg,   /* bK > 1: same law
                               * as the r2c window above */
                              cfg->rigor != VFFT_MEASURE && bK > 1 &&
                                  bK <= 128);
        if (!cd)
            return NULL;
        struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
        if (!h)
        {
            vfft_c2r_disp_destroy(cd);
            return NULL;
        }
        h->transform = VFFT_C2R;
        h->placement = cfg->placement;
        h->layout = (int)cfg->layout; /* INTERLEAVED == CCE spectrum INPUT contract */
        h->N = N;
        h->K = K;
        h->nthreads = _vfft_plan_threads(cfg);
        h->c2rdisp = cd;
        h->padded = padded;
        h->exec_me = (int)bK;
        return h;
    }
    return NULL; /* unreachable: the one call site guards on the same
                  * condition, and every path in the block above returns. */
}

#endif /* VFFT_TRANSFORMS_REAL_CREATE_H */
