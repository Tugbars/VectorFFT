/* real_create_split.h — the r2c / c2r CREATE, split tier.
 *
 * The split real engines: r2c (NATURAL cascade vs STRIDE decoupled, see
 * bridge/real_bridge.h for the 2-axis choice) and c2r, each with the
 * owned-batch (padded) arm, calibrate-on-miss for the inner c2c(N/2) and rfft
 * cells, and the banked/raced route axis. Returns the handle bare; the front
 * dispatcher owns the finish, the odd-real bridge and the zr2c IL route.
 *
 * An interleaved K>1 real request still lands here (the CCE spectrum
 * contract on the split interior) — that is a bridge, per D1 of
 * docs/roadmap/layout_separation_plan.md; it moves in phase 7.
 *
 * POSITION IN vfft.c IS LOAD-BEARING: calls file-scope statics of vfft.c, so
 * it is included (via the dispatcher) after those are defined.
 */
#ifndef VFFT_SPLIT_REAL_CREATE_SPLIT_H
#define VFFT_SPLIT_REAL_CREATE_SPLIT_H

static struct vfft_plan_s *_vfft_create_real_split(const vfft_config_t *cfg,
                                                   vfft_batch ob,
                                                   struct vfft_wisdom_s *W,
                                                   const vfft_proto_registry_t *reg,
                                                   int N,
                                                   size_t K)
{
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
        /* Route axis. A BANKED verdict serves at every rigor tier; the race
         * that produces one is confined to the rfft-competitive zone
         * (K<=64, N even, not MEASURE), and MEASURE / high-K fall through to
         * the fixed-threshold dispatch. */
        vfft_r2c_plan_t *rp =
            /* bK > 1: the route race is a LANE-BATCH question and the
             * split engine has no K=1 batch (K counts the FFTs running;
             * split lanes hold independent FFTs). At K=1
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
        return h;
    }

    /* ── c2r (complex -> real; the r2c inverse), SPLIT input (sre/sim). 2-axis,
     * mirroring r2c: NATURAL (the fast packed cascade run on split input via the
     * stage-0 natural initiator — no repack, low/mid-K winner) vs STRIDE (decoupled,
     * high-K + threads). BOTH consume split re/im, so the pick is transparent to the
     * caller. High rigor MEASURES both at create over the contested low/mid-K zone
     * (natural's win is non-monotonic in K — a fixed threshold can't capture it);
     * else the banked route verdict first, then the threshold. No forced path / no
     * hardcode. ── */
    if (cfg->transform == VFFT_C2R)
    {
        if ((N % 2) != 0)
        {
            _vfft_warn("vfft_create: split C2R odd N=%d (K=%zu): no split real "
                       "engine serves an odd length backward; unsupported",
                       N, K);
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
        /* Route axis — see the r2c site. A banked verdict serves at every
         * rigor tier; only the race is window-confined. */
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
    return NULL; /* unreachable: the dispatcher guards on the same
                  * condition, and every path in the block above returns. */
}

#endif /* VFFT_SPLIT_REAL_CREATE_SPLIT_H */
