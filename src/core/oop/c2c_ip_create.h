/* c2c_ip_create.h — the c2c IN-PLACE create tier (migration step 24).
 *
 * WHAT THIS IS
 * ------------
 * Both in-place c2c arms of _vfft_create_inner, in their original order:
 *
 *   1. the CALLER-SUPPLIED BATCH arm (`... && ob`) — an owned-batch descriptor
 *      was handed in, so the plan serves that exact handle;
 *   2. the general in-place arm, which allocates its own buffers.
 *
 * Arm 1 falls through to arm 2 only by not matching; each returns on every
 * path, so the pair lifts out behind one guard.
 *
 * WHY THE HANDLE IS CHECKED EXACTLY
 * ---------------------------------
 * Arm 1 refuses unless xform, oop, K and N all match. The shapes are genuinely
 * incompatible rather than merely different: an r2c handle's re/im planes are
 * (N/2+1)*Kp, and an OOP handle is 4-plane. A mismatched handle is a caller
 * error and is refused LOUDLY, per the tree's diagnostic directive.
 *
 * PADDED vs UNPADDED IS A WISDOM QUESTION, NOT A BRANCH
 * ----------------------------------------------------
 * Kp is the batch's padded lane count. The padded verdict is not a separate
 * planning path: it is the (N,K) entry's exec_me, and the pad plan IS the
 * aligned (N,Kp) entry -- both ordinary c2c cells in the one unified store.
 * `misaligned = (Kp != K)` selects between them; nothing here invents a cutoff.
 *
 * 🔴 te/ae are re-looked-up after every set: `wisdom_set` may realloc, so a
 * pointer held across a set is a dangling read. The original code does this
 * deliberately and the move preserves it.
 *
 * POSITION IN vfft.c IS LOAD-BEARING
 * ----------------------------------
 * Not a standalone header. It calls file-scope statics that live in vfft.c, so
 * it must be included after those are defined and before _vfft_create_inner.
 *
 * The six parameters are the block's complete free-variable set, derived
 * rather than guessed: this is the first tier that genuinely needs `ob` (the
 * owned batch it serves) and `N` alongside cfg/W/reg/K.
 */
#ifndef VFFT_OOP_C2C_IP_CREATE_H
#define VFFT_OOP_C2C_IP_CREATE_H

/* ── the arms of the natural / scr-mode MEASURE races ────────────────
 * One context serves the five sites (four here, one in c2c_oop_create.h):
 * the incumbent is THIS handle's real execute (in-place door, or the OOP
 * door when oop=1); the challenger is whichever candidate the site built —
 * the natord cascade (zt/zs by zroute) or the IL engine (il2/il3/ilp). The
 * protocol constants stay at each site (support/race.h). */
typedef struct
{
    struct vfft_plan_s *h;
    int oop;                    /* 1: vfft_execute(h, FWD, r0 -> rz) */
    vfft_zturn2_plan_t *zt;     /* cascade challenger, route by zroute */
    vfft_zsplit_plan_t *zs;
    int zroute;
    vfft_il2p_plan_t *il2;      /* IL challenger: first non-NULL serves */
    vfft_il3p_plan_t *il3;
    vfft_ilprime_plan_t *ilp;
    double *rz, *r0;            /* the aliased race buffer and its seed */
    size_t nb;                  /* bytes to re-seed per burst */
} _c2c_race_ctx_t;
static void _c2c_race_inc(void *v)
{
    _c2c_race_ctx_t *c = (_c2c_race_ctx_t *)v;
    if (c->oop)
        vfft_execute(c->h, VFFT_FORWARD, c->r0, NULL, c->rz, NULL);
    else
        _exec_c2c_interleaved(c->h, VFFT_FORWARD, c->rz, c->rz);
}
static void _c2c_race_chal(void *v)
{
    _c2c_race_ctx_t *c = (_c2c_race_ctx_t *)v;
    const double *in = c->oop ? c->r0 : c->rz;
    if (c->zt || c->zs)
    {
        if (c->zroute)
            vfft_zturn2_execute_fwd(c->zt, in, c->rz);
        else
            vfft_zsplit_execute_fwd(c->zs, in, c->rz);
    }
    else if (c->il2)
        vfft_il2p_execute_fwd(c->il2, in, c->rz);
    else if (c->il3)
        vfft_il3p_execute_fwd(c->il3, in, c->rz);
    else
        vfft_ilprime_execute_fwd(c->ilp, in, c->rz);
}
static void _c2c_race_reseed(void *v)
{
    _c2c_race_ctx_t *c = (_c2c_race_ctx_t *)v;
    memcpy(c->rz, c->r0, c->nb);
}

/* ── the tier's ONE exit. Every handle this create returns passes through
 * here, so a new early exit cannot skip the shared post-step (the audit
 * found two that did: the padded-batch exit and the IL-prime exit both
 * shipped mt_unsafe=0 — calloc's default, which spells "proven safe" —
 * without running the proof). The gate is cheap for ST creates (skipped)
 * and engine handles (no cplan: nothing K-splits). */
static vfft_plan _c2c_ip_finish(struct vfft_plan_s *h)
{
    /* MT-safety: flag plans whose codelet ignores the partial-lane count (so
     * _c2c_mt runs them whole-batch instead of K-splitting). Checked once on
     * the FINAL cplan (after any natural rebuild). Safety net now that the
     * DIF/LOG3 K-split twiddle bug is fixed at codegen; only MT plans
     * K-split, so single-threaded creates skip the check and its cost. */
    if (h->cplan)
        h->mt_unsafe = (h->nthreads > 1) ? !_c2c_mt_safe(h->cplan, h->exec_fwd) : 0;
    return h;
}

static vfft_plan _vfft_create_c2c_ip(const vfft_config_t *cfg,
                                     vfft_batch ob,
                                     struct vfft_wisdom_s *W,
                                     const vfft_proto_registry_t *reg,
                                     int N,
                                     size_t K)
{
    if (cfg->transform == VFFT_C2C && cfg->placement == VFFT_INPLACE && ob)
    {
        vfft_batch b = ob;
        if (b->xform != (int)VFFT_C2C || b->oop || b->K != K || b->N != N) /* handle must match exactly */
        {                                                                  /* (an r2c handle's re/im are (N/2+1)*Kp; an OOP handle is 4-plane) */
            _vfft_warn("vfft_create: config.batch does not match this in-place C2C descriptor "
                       "(batch: %s%s N=%d K=%zu; config: C2C in-place N=%d K=%zu) — allocate "
                       "— INTERNAL INVARIANT (the plan allocates its own buffers); please report",
                       _vfft_tname(b->xform), b->oop ? " out-of-place" : "", b->N, b->K, N, K);
            return NULL;
        }
        size_t Kp = b->Kp;

        /* UNIFIED wisdom (single spike_wisdom.txt): the padded verdict is the (N,K) entry's
         * exec_me, and the pad plan IS the aligned (N,Kp) entry — both ordinary c2c cells. */
        const vfft_proto_wisdom_entry_t *te = vfft_proto_wisdom_lookup(&W->c2c, N, K);  /* tail leg = factK  */
        const vfft_proto_wisdom_entry_t *ae = vfft_proto_wisdom_lookup(&W->c2c, N, Kp); /* pad leg = aligned (N,Kp) */
        int misaligned = (Kp != K);
        /* wave-4: seed the process cache from the STORE (both legs); te/ae
         * re-looked-up after every set (wisdom_set may realloc). */
        if (!W->vw2_off_stride)
        {
            /* store-hit OVERWRITES the table (the frozen-file preload may
             * be stale vs post-freeze store rows — the store wins) */
            vfft_proto_wisdom_entry_t sb;
            if (vw2_stride_lookup(&W->vw2, 0, N, K, &sb))
                vfft_proto_wisdom_set(&W->c2c, &sb);
            if (vw2_stride_lookup(&W->vw2, 0, N, Kp, &sb))
                vfft_proto_wisdom_set(&W->c2c, &sb);
            te = vfft_proto_wisdom_lookup(&W->c2c, N, K);
            ae = vfft_proto_wisdom_lookup(&W->c2c, N, Kp);
        }

        /* CALIBRATE-ON-MISS (planner primitive). Ensure the (N,K) tight cell is calibrated
         * (tail leg / — for aligned K — the plan itself). Same on-miss contract as tight c2c. */
        if ((!te || cfg->recalibrate) && !_vfft_is_prime(N))
        {
            vfft_proto_wisdom_entry_t ne;
            if (_calibrate_c2c(N, K, cfg->rigor, reg, &ne) == 0)
            {
                vfft_proto_wisdom_add(&W->c2c, &ne, 1);
                vw2_stride_bank_entry(&W->vw2, &ne, 0);
                _vw2_persist(W, cfg);
                te = vfft_proto_wisdom_lookup(&W->c2c, N, K);
            }
        }
        /* Misaligned: the aligned (N,Kp) cell IS the pad plan — an ORDINARY c2c cell. Padding
         * stores ONLY the verdict (exec_me), never a copy of the aligned plan; the plan is
         * calibrated normally, ON DEMAND. So ensure (N,Kp) exists when we must MEASURE
         * (unmeasured / recalibrate) OR when the verdict is already PAD (a verdict-only cell —
         * e.g. shipped wisdom — whose aligned plan isn't present yet would otherwise fall
         * silently to the tail). When measuring, A/B tail-vs-pad and stamp exec_me. Aligned K
         * needs no A/B (Kp==K). Prime skips. */
        if (misaligned && te && !_vfft_is_prime(N))
        {
            int measure = (cfg->recalibrate || te->exec_me == 0);
            int need_aligned = measure || te->exec_me == (int)Kp;
            int dirty = 0;
            if (need_aligned && (!ae || cfg->recalibrate))
            {
                vfft_proto_wisdom_entry_t ne;
                if (_calibrate_c2c(N, (size_t)Kp, cfg->rigor, reg, &ne) == 0)
                {
                    vfft_proto_wisdom_add(&W->c2c, &ne, 1);
                    vw2_stride_bank_entry(&W->vw2, &ne, 0);
                    dirty = 1;
                }
            }
            te = vfft_proto_wisdom_lookup(&W->c2c, N, K); /* re-lookup: wisdom_add may realloc */
            ae = vfft_proto_wisdom_lookup(&W->c2c, N, Kp);
            if (measure && te && ae)
            {
                int verdict = _calibrate_pad(N, K, cfg->rigor, reg, te, ae); /* Kp / K / 0 */
                if (verdict > 0)
                {
                    vfft_proto_wisdom_entry_t upd = *te; /* keep factK, stamp the verdict */
                    upd.exec_me = verdict;
                    vfft_proto_wisdom_add(&W->c2c, &upd, 1);
                    vw2_stride_bank_entry(&W->vw2, &upd, 0); /* pad_me= rides the record */
                    dirty = 1;
                    te = vfft_proto_wisdom_lookup(&W->c2c, N, K);
                    ae = vfft_proto_wisdom_lookup(&W->c2c, N, Kp);
                }
            }
            if (dirty)
                _vw2_persist(W, cfg);
        }

        /* Select: PAD verdict -> the aligned (N,Kp) factorization @me=Kp ; else the (N,K) tight
         * factorization @me=K (tail on the padded buffer, always correct). */
        const int *facs = NULL, *vars = NULL;
        int nf = 0, use_dif = 0, exec_me = (int)K;
        if (misaligned && te && te->exec_me == (int)Kp && ae && ae->nf > 0)
        {
            facs = ae->factors;
            vars = ae->variants;
            nf = ae->nf;
            use_dif = ae->use_dif_forward;
            exec_me = (int)Kp;
        }
        else if (te && te->nf > 0)
        {
            facs = te->factors;
            vars = te->variants;
            nf = te->nf;
            use_dif = te->use_dif_forward;
            exec_me = (int)K;
        }
        else
            return NULL; /* no factorization available (e.g. prime N) */

        /* Backstop: verify the chosen factorization actually covers N (a wire-able-but-under-
         * covering factorization would silently compute the wrong-length transform). */
        {
            long long prod = 1;
            for (int i = 0; i < nf; i++)
                prod *= facs[i];
            if (prod != (long long)N)
                return NULL;
        }

        stride_plan_t *p = vfft_proto_plan_create_ex(N, Kp, facs, vars, nf, use_dif, reg);
        if (!p)
            return NULL;
        if (p->K != Kp) /* stride-match invariant: plan stride must equal buffer stride */
        {
            vfft_proto_plan_destroy(p);
            return NULL;
        }
        struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
        if (!h)
        {
            vfft_proto_plan_destroy(p);
            return NULL;
        }
        h->transform = VFFT_C2C;
        h->placement = VFFT_INPLACE;
        h->layout = (int)cfg->layout; /* SPLIT by the batch+IL gate above */
        h->N = N;
        h->K = K;
        h->nthreads = _vfft_plan_threads(cfg);
        h->cplan = p;
        h->padded = 1;
        h->exec_me = exec_me;
#ifdef VFFT_USE_JIT
        /* Wrinkle C: only the ALIGNED pad leg (me=Kp) is eligible for the baked/JIT fast
         * path. The tail leg (exec_me==K, odd) MUST use the generic tail-capable executor,
         * so leave exec_*=NULL there (execute falls back to vfft_proto_execute_fwd/bwd). */
        if (exec_me == (int)Kp && p->num_stages > 0)
        {
            h->exec_fwd = vfft_proto_plan_jit_fwd(p);
            h->exec_bwd = vfft_proto_plan_jit_bwd(p);
        }
#endif
        return _c2c_ip_finish(h);
    }

    /* ── c2c IN-PLACE ── */
    if (cfg->transform == VFFT_C2C && cfg->placement == VFFT_INPLACE)
    {
        vfft_proto_dispatch_set_bluestein_wisdom(&W->bluestein);
        if (_vfft_is_prime(N))
        {
            /* Prime N routes through Rader (radix-smooth N-1: M=N-1 + heuristic B,
             * no wisdom) or Bluestein (else: (M,B) FROM the bluestein wisdom). Only
             * the Bluestein cell consults wisdom, so calibrate-on-miss only there. */
            if (!_vfft_is_radix_smooth(N - 1) &&
                (cfg->recalibrate || !bluestein_wisdom_lookup(&W->bluestein, N, K)))
            {
                size_t tot = (size_t)N * K;
                double *cre = (double *)malloc(tot * sizeof(double));
                double *cim = (double *)malloc(tot * sizeof(double));
                if (cre && cim)
                {
                    for (size_t i = 0; i < tot; i++)
                    {
                        cre[i] = (double)rand() / RAND_MAX - 0.5;
                        cim[i] = (double)rand() / RAND_MAX - 0.5;
                    }
                    double budget = (cfg->rigor == VFFT_MEASURE) ? 0.02 : 0.05;
                    int trials = (cfg->rigor == VFFT_MEASURE) ? 2 : 3;
                    bluestein_calibrate_one(&W->bluestein, N, K, reg, &W->c2c,
                                            cre, cim, budget, trials, NULL);
                    if (W->path_bluestein[0])
                        bluestein_wisdom_save(&W->bluestein, W->path_bluestein);
                }
                free(cre);
                free(cim);
            }
        }
        else
        {
            const vfft_proto_wisdom_entry_t *e = vfft_proto_wisdom_lookup(&W->c2c, N, K);
            /* REGIME SEPARATION: order=NATURAL never re-measures/overwrites the SCRAMBLED entry — its
             * recalibrate governs only the natural verdict (below). It calibrates the scrambled cell just
             * once, when absent, to build the base plan p (the PURE-floor + race seed). So the natural
             * floor rides the STABLE banked scrambled chain, not a noisy fresh re-measure. order=DEFAULT/
             * SCRAMBLED keeps the old recalibrate-overwrites semantics. */
            int scr_recalib = cfg->recalibrate && cfg->order != VFFT_ORDER_NATURAL;
            /* wave-4: store-first; the in-memory table stays the process
             * cache auto_plan_dispatch walks below. */
            {
                vfft_proto_wisdom_entry_t seb;
                if (!scr_recalib && !W->vw2_off_stride &&
                    vw2_stride_lookup(&W->vw2, 0, N, K, &seb))
                {
                    vfft_proto_wisdom_set(&W->c2c, &seb); /* store wins */
                    e = vfft_proto_wisdom_lookup(&W->c2c, N, K);
                }
            }
            if (!e || scr_recalib)
            {
                vfft_proto_wisdom_entry_t ne;
                if (_calibrate_c2c(N, K, cfg->rigor, reg, &ne) == 0)
                {
                    vfft_proto_wisdom_add(&W->c2c, &ne, 1);
                    vw2_stride_bank_entry(&W->vw2, &ne, 0);
                    _vw2_persist(W, cfg);
                }
            }
        }
        /* prime-aware: factorable -> CT/wisdom; prime -> Rader/Bluestein (override). */
        stride_plan_t *p = vfft_proto_auto_plan_dispatch(N, K, reg, &W->c2c);
        if (!p)
        {
            /* AWKWARD-COMPOSITE coverage (2026-08-27, the last hole in
             * the K=1 IL grid): CT needs smooth factors and
             * prime_dispatch requires primality, so an odd N with a
             * prime factor past the radix set (129 = 3*43) had NO
             * in-place route at all — and the refusal was SILENT.
             * il_prime documents zin == zout safe in both methods, so
             * the K=1 INTERLEAVED cell adopts it directly (the forced-
             * route precedent: nothing exists to race against). The
             * handle carries ONLY k1ilpr — execute dispatches it before
             * any cplan path. Everything else now refuses LOUDLY. */
            if (K == 1 && cfg->layout == VFFT_LAYOUT_INTERLEAVED)
            {
                vfft_ilprime_plan_t *ilpr = vfft_ilprime_create(N);
                if (ilpr)
                {
                    struct vfft_plan_s *hh = (struct vfft_plan_s *)
                        calloc(1, sizeof *hh);
                    if (!hh)
                    {
                        vfft_ilprime_destroy(ilpr);
                        return NULL;
                    }
                    hh->transform = VFFT_C2C;
                    hh->placement = VFFT_INPLACE;
                    hh->layout = (int)cfg->layout;
                    hh->N = N;
                    hh->K = K;
                    hh->nthreads = _vfft_plan_threads(cfg);
                    hh->k1ilpr = ilpr;
                    return _c2c_ip_finish(hh);
                }
            }
            _vfft_warn("vfft_create: in-place C2C N=%d K=%zu — no CT "
                       "factorization, not prime, and the IL "
                       "prime/Bluestein engine cannot serve it "
                       "(K==1 INTERLEAVED only)",
                       N, K);
            return NULL;
        }
        /* (Self-contained natural design: the old C1 scrambled-entry bank-from-plan is GONE — the natural
         * block below no longer reads the scrambled entry, so it can't hard-fail for want of one. Its base
         * plan is `p` itself, which auto_plan_dispatch always built here.) */
        struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
        if (!h)
        {
            vfft_proto_plan_destroy(p);
            return NULL;
        }
        h->transform = VFFT_C2C;
        h->placement = VFFT_INPLACE;
        h->layout = (int)cfg->layout;
        h->N = N;
        h->K = K;
        h->nthreads = _vfft_plan_threads(cfg);
        h->cplan = p;
#ifdef VFFT_USE_JIT
        if (p->num_stages > 0)
        {
            /* staged plan -> resolve the whole-plan JIT/baked executor */
            h->exec_fwd = vfft_proto_plan_jit_fwd(p);
            h->exec_bwd = vfft_proto_plan_jit_bwd(p);
        }
        else
        {
            /* prime override plan (Rader/Bluestein): JIT the inner CT FFT — the override
             * executor calls inner_jit_* instead of the generic inner. exec_*=NULL keeps
             * the override-aware path. Accessors/setters are no-ops on the wrong kind. */
            stride_plan_t *in = stride_rader_inner_plan(p);
            if (!in)
                in = stride_bluestein_inner_plan(p);
            if (in)
            {
                vfft_proto_exec_fn ifwd = vfft_proto_plan_jit_fwd(in);
                vfft_proto_exec_fn ibwd = vfft_proto_plan_jit_bwd(in);
                if (ifwd && ibwd)
                {
                    stride_rader_set_inner_jit(p, ifwd, ibwd);
                    stride_bluestein_set_inner_jit(p, ifwd, ibwd);
                }
            }
        }
#endif
        /* ── VFFT_ORDER_NATURAL (P1b: FREE + PURE_CYCLE + PSWAP w/ injected chains; the
         * calibrator race stamps the verdict into wisdom v7. SCR/LEAF-IP still degrade to
         * PURE until their executors land). order==DEFAULT leaves everything below
         * untouched — byte-identical kill switch. */
        if (cfg->order == VFFT_ORDER_NATURAL)
        {
            vfft_proto_nat_entry_t neb;
            const vfft_proto_nat_entry_t *ne =
                W->vw2_off_stride ? vfft_proto_nat_lookup(&W->c2c, N, K)
                                  : (vw2_stride_lookup_nat(&W->vw2, _vw2_lay_of(cfg), N, K, &neb) ? &neb : NULL);
            int mode = (ne && !cfg->recalibrate) ? ne->mode : VFFT_NAT_UNSET;
            if (p->num_stages <= 1)
                mode = VFFT_NAT_FREE; /* single-stage / prime override: already natural, no tape */
            /* Natural-terminator cascade, built as a CANDIDATE for the race below from the kind-4
             * chain with recalibrate cleared. Kill switch: VFFT_NO_NAT_ZCASC.
             * See docs/design/vfft_front_door.md. */
            vfft_zturn2_plan_t *zct = NULL;
            if (K == 1 && !ob && cfg->layout == VFFT_LAYOUT_INTERLEAVED &&
                N >= _vfft_zcasc_min_n() && !getenv("VFFT_NO_NAT_ZCASC"))
            {
                vfft_config_t rcfg = *cfg;
                rcfg.recalibrate = 0;
                vfft_zsplit_plan_t *zcs = NULL;
                int zcr = 0;
                /* COLD-STORE candidate (census tail, 2026-08-25): with no
                 * kind-4 row banked yet the replay misses and the natural
                 * race used to run WITHOUT its cascade arm — the tape won
                 * by default (the same single-writer disease, natural
                 * flavor). Build the candidate instead: aliased t2q
                 * timing, no kind-4 bank (ip=1) — the race below still
                 * decides, and only its verdict banks (@nat). */
                if (_k1z_wisdom_replay(&rcfg, W, N, &zcs, &zct, &zcr) ||
                    _k1z_race_and_bank(&rcfg, W, N, /*ip=*/1, &zcs, &zct,
                                       &zcr))
                {
                    if (zcs)
                        vfft_zsplit_destroy(zcs);
                    if (zct && !vfft_zturn2_set_natord(zct, 1))
                    {
                        vfft_zturn2_destroy(zct);
                        zct = NULL;
                    }
                }
            }
            /* CONSUME ZCASC: attach and skip the whole tape build. A banked
             * ZCASC whose kind-4 line has since vanished (or been refused)
             * degrades to UNSET — re-measure, never hard-fail. */
            if (mode == VFFT_NAT_ZCASC)
            {
                if (zct)
                {
                    h->zturn = zct;
                    h->zroute = 1;
                    zct = NULL;
                    if (getenv("VFFT_NAT_LOG"))
                        fprintf(stderr, "[natorder] N=%d K=%zu replay ZCASC\n",
                                N, K);
                }
                else
                    mode = VFFT_NAT_UNSET;
            }
            /* ── ILP candidate (Phase B): the sub-2048 tier of the same
             * idea — il2p/il3p serve natural in-place interleaved natively
             * (alias-gated; two-stage through internal scratch, zout
             * written only by the last stage). Raced end-to-end vs the
             * convert incumbent, banked in the same @nat slot. */
            vfft_il2p_plan_t *ilc2 = NULL;
            vfft_il3p_plan_t *ilc3 = NULL;
            if (K == 1 && !ob && cfg->layout == VFFT_LAYOUT_INTERLEAVED &&
                N < 2048 && !getenv("VFFT_NO_NAT_ILP"))
                _k1_il_candidate(W, N, &ilc2, &ilc3);
            if (mode == VFFT_NAT_ILP)
            {
                if (ilc2 || ilc3)
                {
                    h->k1il2p = ilc2;
                    h->k1il3p = ilc3;
                    ilc2 = NULL;
                    ilc3 = NULL;
                    if (getenv("VFFT_NAT_LOG"))
                        fprintf(stderr, "[natorder] N=%d K=%zu replay ILP\n",
                                N, K);
                }
                else
                    mode = VFFT_NAT_UNSET;
            }
            if (mode != VFFT_NAT_FREE && mode != VFFT_NAT_ZCASC &&
                mode != VFFT_NAT_ILP)
            {
                /* Self-contained natural — the DEPLOYED plan + its OWN chain drive everything; this path
                 * NEVER reads the scrambled entry. CONSUME (warm ne) rebuilds the deployed plan from ne's
                 * own chain (may differ from the scrambled p — a palindrome, or a single-radix leaf) and
                 * swaps it into cplan. MEASURE reuses the handle's scrambled plan p as the PURE-floor base
                 * + race seed (the PLAN object — not the wisdom entry). */
                int consume = (ne && !cfg->recalibrate);
                int dnf, ddif, dfac[STRIDE_MAX_STAGES], dvar[STRIDE_MAX_STAGES];
                if (consume)
                {
                    dnf = ne->nf;
                    ddif = ne->use_dif;
                    for (int s = 0; s < dnf && s < STRIDE_MAX_STAGES; s++)
                    {
                        dfac[s] = ne->factors[s];
                        dvar[s] = ne->variants[s];
                    }
                }
                else
                {
                    dnf = p->num_stages;
                    ddif = p->use_dif_forward;
                    for (int s = 0; s < dnf && s < STRIDE_MAX_STAGES; s++)
                    {
                        dfac[s] = p->factors[s];
                        dvar[s] = p->variants[s];
                    }
                }
                /* per-worker cycle scratch: h->nthreads slots of 2*K doubles (MT split).
                 * Sized by the PLAN'S SNAPSHOT, not the live pool: the pool is grow-only,
                 * so a live count taken here can be smaller than the one execute sees
                 * later, and the reorder slices tmp + slot*2*K per worker. The execute
                 * side clamps by the same h->nthreads (natorder_mt.h), so the two numbers
                 * cannot disagree. natorder_scratch_gate asserts this. */
                h->nat_tmp = (double *)malloc((size_t)(h->nthreads < 1 ? 1 : h->nthreads) * 2 * K * sizeof(double));
                if (!h->nat_tmp)
                {
                    vfft_destroy(h);
                    return NULL;
                }

                /* CONSUME SCR (parked; rebuild the DIT scatter from the stored chain). */
                if (consume && mode == VFFT_NAT_SCR)
                {
                    natorder_scr_t sc;
                    stride_plan_t *sp = NULL;
                    int *scyc = NULL;
                    if (natorder_scr_build_dit(N, K, dfac, dnf, reg, &sc, &sp, &scyc))
                    {
                        h->nat_scr = (natorder_scr_t *)malloc(sizeof(natorder_scr_t));
                        if (h->nat_scr)
                        {
                            *h->nat_scr = sc;
                            vfft_proto_plan_destroy(h->cplan);
                            h->cplan = sp;
                            h->nat_list = scyc;
                            h->exec_fwd = NULL;
                            h->exec_bwd = NULL;
#ifdef VFFT_USE_JIT
                            h->exec_bwd = vfft_proto_plan_jit_bwd(h->cplan);
#endif
                        }
                        else
                        {
                            natorder_scr_free(&sc);
                            vfft_proto_plan_destroy(sp);
                            free(scyc);
                            mode = VFFT_NAT_PURE_CYCLE;
                        }
                    }
                    else
                        mode = VFFT_NAT_PURE_CYCLE;
                }
                /* CONSUME PURE/PSWAP (or SCR-demoted): rebuild the deployed plan from the stored chain. */
                if (consume && mode != VFFT_NAT_SCR && !h->nat_scr)
                {
                    stride_plan_t *dp = vfft_proto_plan_create_ex(N, K, dfac, dvar, dnf, ddif, reg);
                    if (dp)
                    {
                        vfft_proto_plan_destroy(h->cplan);
                        h->cplan = dp;
                        p = dp; /* probe + tape now follow the DEPLOYED plan */
                        h->exec_fwd = NULL;
                        h->exec_bwd = NULL;
#ifdef VFFT_USE_JIT
                        h->exec_fwd = vfft_proto_plan_jit_fwd(dp);
                        h->exec_bwd = vfft_proto_plan_jit_bwd(dp);
#endif
                    }
                    else
                    {
                        /* rebuild failed (OOM) -> honorable PURE on the ORIGINAL scrambled p. Reset the chain
                         * to p's OWN so the perm probe below detects on the plan actually deployed — otherwise
                         * detect runs the banked (possibly injected/leaf) chain dfac against p's spectrum,
                         * matches none, returns NULL, and the create hard-fails instead of degrading. */
                        mode = VFFT_NAT_PURE_CYCLE;
                        dnf = p->num_stages;
                        ddif = p->use_dif_forward;
                        for (int s = 0; s < dnf && s < STRIDE_MAX_STAGES; s++)
                        {
                            dfac[s] = p->factors[s];
                            dvar[s] = p->variants[s];
                        }
                    }
                }

                /* Perm + reorder tape from the deployed plan p (SCR already holds its cycle tape = scyc). */
                if (!h->nat_scr)
                {
                    size_t tot = (size_t)N * K;
                    double *cre = (double *)calloc(tot, sizeof(double));
                    double *cim = (double *)calloc(tot, sizeof(double));
                    int *M = NULL;
                    if (cre && cim)
                    {
                        cre[K] = 1.0; /* impulse at n0=1, lane 0 */
                        vfft_proto_execute_fwd(p, cre, cim, K);
                        M = vfft_natorder_detect(N, dfac, dnf, K, cre, cim, 1);
                    }
                    free(cre);
                    free(cim);
                    if (!M)
                    {
                        vfft_destroy(h);
                        return NULL;
                    }

                    if (mode == VFFT_NAT_PSWAP)
                        h->nat_list = vfft_natorder_mk_pairs(N, M); /* CONSUME PSWAP (single-leaf => empty tape = FREE) */
                    else if (mode == VFFT_NAT_PURE_CYCLE)
                        h->nat_list = vfft_natorder_mk_cycles(N, M); /* CONSUME PURE */
                    else                                             /* mode == VFFT_NAT_UNSET: MEASURE */
                    {
                        h->nat_list = vfft_natorder_mk_cycles(N, M); /* race PURE-floor baseline */
                        if (h->nat_list)
                        {
                            /* OPPORTUNISTIC PSWAP: p's perm is an involution => pairs beat cycles on the SAME
                             * plan (deterministic free win). GATE: if a single-stage [N] leaf exists, DON'T
                             * short-circuit — fall to the race so it also weighs the FREE-reorder single-radix
                             * candidate (the 2D 64x16 lesson). The race re-injects the calibrated palindrome. */
                            int has_leaf = (N > 1 && N < VFFT_PROTO_REG_MAX_RADIX && reg->n1_fwd[N]);
                            int *opp = (!has_leaf) ? vfft_natorder_mk_pairs(N, M) : NULL;
                            if (opp)
                            {
                                free(h->nat_list);
                                h->nat_list = opp; /* deployed plan p unchanged */
                                mode = VFFT_NAT_PSWAP;
                                _bank_nat_1d(W, cfg, N, K, mode, 0.0, dfac, dvar, dnf, ddif);
                            }
                            else
                            {
                                /* RACE (PURE vs injected-palindrome/single-leaf PSWAP vs DIT-SCR; 5% margin),
                                 * seeded from the deployed chain dfac (the PLAN object, never the scr entry). */
                                vfft_natorder_verdict_t v;
                                /* HARNESS: this racer is about to time. It lives in
                                 * natorder_calibrate.h, NOT in this file, which is why the
                                 * original census missed it - that census enumerated clock
                                 * calls in vfft.c plus their local callers, and a racer
                                 * DEFINED in another header is invisible to both passes.
                                 * The cost was concrete: c2c.split.ip.nat has no banked nat
                                 * entry, so it takes this branch every time and picked
                                 * nat=5/natcyc=96 in 8 of 10 runs and nat=4/natcyc=34 in the
                                 * other 2 - while reporting races=0 and therefore claiming to
                                 * be safe to diff. A fingerprint that flaps 20% of the time
                                 * under a purity flag is worse than no fingerprint. */
                                _vfft_create_race_count++;
                                vfft_natorder_race(N, K, reg, p, h->nat_list, h->nat_tmp, dfac, dnf, &v);
                                mode = v.mode;
                                if (mode == VFFT_NAT_PSWAP)
                                {
                                    vfft_proto_plan_destroy(h->cplan);
                                    h->cplan = v.planB;
                                    free(h->nat_list);
                                    h->nat_list = v.pairs;
                                    h->exec_fwd = NULL;
                                    h->exec_bwd = NULL;
#ifdef VFFT_USE_JIT
                                    h->exec_fwd = vfft_proto_plan_jit_fwd(h->cplan);
                                    h->exec_bwd = vfft_proto_plan_jit_bwd(h->cplan);
#endif
                                    /* deployed = injected chain: v.factors, uniform-prof variants (stage0 FLAT), dif=0. */
                                    int fac2[STRIDE_MAX_STAGES], var2[STRIDE_MAX_STAGES];
                                    for (int s = 0; s < v.nf && s < STRIDE_MAX_STAGES; s++)
                                    {
                                        fac2[s] = v.factors[s];
                                        var2[s] = s ? v.prof : 0;
                                    }
                                    _bank_nat_1d(W, cfg, N, K, mode, v.ns, fac2, var2, v.nf, 0);
                                }
                                else if (mode == VFFT_NAT_SCR)
                                {
                                    h->nat_scr = (natorder_scr_t *)malloc(sizeof(natorder_scr_t));
                                    if (!h->nat_scr)
                                    {
                                        free(M);
                                        vfft_proto_plan_destroy(v.scr_plan);
                                        natorder_scr_free(&v.scr);
                                        free(v.scr_cycles);
                                        vfft_destroy(h);
                                        return NULL;
                                    }
                                    *h->nat_scr = v.scr;
                                    vfft_proto_plan_destroy(h->cplan);
                                    h->cplan = v.scr_plan;
                                    free(h->nat_list);
                                    h->nat_list = v.scr_cycles;
                                    h->exec_fwd = NULL;
                                    h->exec_bwd = NULL;
#ifdef VFFT_USE_JIT
                                    h->exec_bwd = vfft_proto_plan_jit_bwd(h->cplan);
#endif
                                    _bank_nat_1d(W, cfg, N, K, mode, v.ns, dfac, dvar, dnf, ddif); /* the DIT base chain */
                                }
                                else /* PURE floor: deployed = p, bank p's chain */
                                    _bank_nat_1d(W, cfg, N, K, VFFT_NAT_PURE_CYCLE, v.ns, dfac, dvar, dnf, ddif);
                            }
                        }
                    }
                    free(M);
                    if (!h->nat_list && mode != VFFT_NAT_SCR)
                    {
                        vfft_destroy(h);
                        return NULL;
                    }
                }
            }
            /* MT metadata for the reorder pass: PURE + SCR-backward split cycles (need offsets);
             * PSWAP splits pairs (count only). nat_list is now final for the chosen mode. */
            if (mode == VFFT_NAT_PSWAP)
                h->nat_ncyc = vfft_natorder_pair_count(h->nat_list);
            else if (mode == VFFT_NAT_PURE_CYCLE || mode == VFFT_NAT_SCR)
            {
                h->nat_cyc_off = vfft_natorder_cycle_offsets(h->nat_list, &h->nat_ncyc);
                if (!h->nat_cyc_off)
                {
                    vfft_destroy(h);
                    return NULL;
                }
            }
#ifdef VFFT_USE_JIT
            /* SCR: JIT/bake the OOP scratch-fill's stages 1.. (stage 0 is a bare n1 loop). Only
             * meaningful for nf>=3; execute_fwd_oop_jit skips the call when sub has 1 stage. */
            if (mode == VFFT_NAT_SCR && h->nat_scr && h->nat_scr->sub.num_stages > 1)
                h->nat_scr->sub_jit_fwd = vfft_proto_plan_jit_fwd(&h->nat_scr->sub);
#endif
            h->nat_mode = mode;
            /* ── ZCASC MEASURE race (B5): the incumbent handle EXACTLY as
             * built (its real execute path, tape and all) vs the natord
             * cascade, in-place interleaved on the same scratch. End-to-end
             * on purpose — the engines share nothing, so any partial-cost
             * comparison would be a hand heuristic. 5 rounds, alternated
             * order, medians; buffer re-seeded per round (repeated in-place
             * fwd amplifies magnitudes — unchecked it walks into inf and
             * the timing measures denormal/inf handling, not the FFT).
             * Winner banked in the SAME @nat verdict slot. Loss path: the
             * earlier bank stands, candidate destroyed. */
            if (zct && h->nat_mode != VFFT_NAT_ZCASC &&
                h->nat_mode != VFFT_NAT_FREE)
            {
                double *rz = (double *)malloc(2 * (size_t)N * sizeof(double));
                double *r0 = (double *)malloc(2 * (size_t)N * sizeof(double));
                if (rz && r0)
                {
                    for (long i = 0; i < 2L * N; i++)
                        r0[i] = (double)rand() / RAND_MAX - 0.5;
                    const int reps = N <= 4096 ? 24 : (N <= 16384 ? 10 : 6);
                    double ns[2]; /* [0] incumbent, [1] zcasc */
                    _c2c_race_ctx_t rc = { h, 0, zct, NULL, 1, NULL, NULL, NULL, rz, r0,
                                            2 * (size_t)N * sizeof(double) };
                    const vfft_race_arm_t arms[2] = {
                        { "incumbent", _c2c_race_inc, &rc }, { "zcasc", _c2c_race_chal, &rc } };
                    /* 5 rounds, odd rounds reversed, median-of-5; reseed per burst */
                    const vfft_race_proto_t proto = { 5, reps, VFFT_RACE_MEDIAN, 1, 0, _c2c_race_reseed, &rc };
                    _vfft_pool_arm(h->nthreads);
                    vfft_race_run(&proto, arms, 2, ns);
                    if (ns[1] < ns[0])
                    {
                        h->zturn = zct;
                        h->zroute = 1;
                        zct = NULL;
                        h->nat_mode = VFFT_NAT_ZCASC;
                        /* chain fields informational (replay reads kind-4).
                         * 🔴 Read them from h->cplan, NOT the local p: when
                         * the tape race installed a PSWAP/SCR plan it
                         * destroyed the plan p still points at (found
                         * 2026-08-04 — freed-heap nf made the saver's
                         * factor loop walk off the entry: nondeterministic
                         * segfault + garbage @nat lines). h->cplan is the
                         * live deployed plan on every path. */
                        _bank_nat_1d(W, cfg, N, K, VFFT_NAT_ZCASC, ns[1],
                                     h->cplan->factors, h->cplan->variants,
                                     h->cplan->num_stages,
                                     h->cplan->use_dif_forward);
                        /* NOTE: the tape artifacts (nat_list/nat_cyc_off/
                         * nat_tmp/nat_scr) stay allocated — destroy frees
                         * them; selective freeing here would duplicate
                         * destroy's invariants for ~O(N) ints of dead
                         * weight. Flagged, accepted for v1. */
                    }
                    if (getenv("VFFT_NAT_LOG"))
                        fprintf(stderr,
                                "[natorder] N=%d K=%zu zcasc=%.0fns "
                                "incumbent=%.0fns -> %s\n",
                                N, K, ns[1], ns[0],
                                h->nat_mode == VFFT_NAT_ZCASC ? "ZCASC"
                                                              : "tape");
                }
                free(rz);
                free(r0);
            }
            if (zct)
            {
                vfft_zturn2_destroy(zct); /* candidate lost or was unused */
                zct = NULL;
            }
            /* ── ILP MEASURE race (Phase B): same protocol as ZCASC — the
             * finished incumbent's real execute vs the aliased IL engine,
             * 5 rounds alternated, medians, buffer re-seeded per round.
             * NATURAL creates only measure; scrambled rides the verdict
             * hit-only (single @nat writer). */
            if ((ilc2 || ilc3) && h->nat_mode != VFFT_NAT_ILP &&
                h->nat_mode != VFFT_NAT_FREE &&
                h->nat_mode != VFFT_NAT_ZCASC)
            {
                double *rz = (double *)malloc(2 * (size_t)N * sizeof(double));
                double *r0 = (double *)malloc(2 * (size_t)N * sizeof(double));
                if (rz && r0)
                {
                    for (long i = 0; i < 2L * N; i++)
                        r0[i] = (double)rand() / RAND_MAX - 0.5;
                    const int reps = N <= 256 ? 200 : (N <= 1024 ? 80 : 32);
                    double ns[2]; /* [0] incumbent, [1] ilp */
                    _c2c_race_ctx_t rc = { h, 0, NULL, NULL, 0, ilc2, ilc3, NULL, rz, r0,
                                            2 * (size_t)N * sizeof(double) };
                    const vfft_race_arm_t arms[2] = {
                        { "incumbent", _c2c_race_inc, &rc }, { "ilp", _c2c_race_chal, &rc } };
                    /* 5 rounds, odd rounds reversed, median-of-5; reseed per burst */
                    const vfft_race_proto_t proto = { 5, reps, VFFT_RACE_MEDIAN, 1, 0, _c2c_race_reseed, &rc };
                    _vfft_pool_arm(h->nthreads);
                    vfft_race_run(&proto, arms, 2, ns);
                    if (ns[1] < ns[0])
                    {
                        h->k1il2p = ilc2;
                        h->k1il3p = ilc3;
                        ilc2 = NULL;
                        ilc3 = NULL;
                        h->nat_mode = VFFT_NAT_ILP;
                        /* h->cplan, not p — same dangling-p hazard as the
                         * ZCASC bank above (chain is informational here). */
                        _bank_nat_1d(W, cfg, N, K, VFFT_NAT_ILP, ns[1],
                                     h->cplan->factors, h->cplan->variants,
                                     h->cplan->num_stages,
                                     h->cplan->use_dif_forward);
                    }
                    if (getenv("VFFT_NAT_LOG"))
                        fprintf(stderr,
                                "[natorder] N=%d K=%zu ilp=%.0fns "
                                "incumbent=%.0fns -> %s\n",
                                N, K, ns[1], ns[0],
                                h->nat_mode == VFFT_NAT_ILP ? "ILP" : "tape");
                }
                free(rz);
                free(r0);
            }
            if (ilc2)
                vfft_il2p_destroy(ilc2);
            if (ilc3)
                vfft_il3p_destroy(ilc3);
        }
        /* the MT-safety gate moved to _c2c_ip_finish — the tier's one exit —
         * so the early exits above cannot skip it. */

        /* ── K=1 SCRAMBLED interleaved IN-PLACE: attach the cascade on a
         * wisdom HIT (Phase A of docs/roadmap/cascade_natural_inplace_plan.md).
         *
         * P0a (zturn_inplace_probe.c): the cascade is alias-safe in==out,
         * memcmp-proven BOTH directions including tiled and fused-terminator
         * arms — the same shadow-plane shape MKL uses for its in-place.
         * HIT-ONLY on purpose: the OOP branch stays the only racer/banker; a
         * miss serves the classic in-place path exactly as before, so this is
         * strictly additive. Layout-gated at CREATE (unlike the OOP attach)
         * because the in-place execute dispatch only consults the cascade
         * under the interleaved z contract — building it for a split-layout
         * handle would be dead weight. Mono/Bailey IL tiers stay OOP-only
         * until their alias-safety is verified per family (A3) — the classic
         * path keeps serving their in-place cells as today. */
        if (K == 1 && !ob &&
            (cfg->order == VFFT_ORDER_SCRAMBLED ||
             cfg->order == VFFT_ORDER_DEFAULT ||
             (cfg->order == VFFT_ORDER_NATURAL && h->cplan &&
              h->cplan->num_stages <= 1)) &&
            cfg->layout == VFFT_LAYOUT_INTERLEAVED)
        {
            /* NATURAL admission is single-stage/prime ONLY (mode FREE:
             * the cell is already natural, all three order spellings are
             * one contract there — census classes 2+4). Multi-stage
             * NATURAL has its own tape/ZCASC machinery above. */
            /* >=2048 MODE-CELL flow (owner-approved class-3 fix,
             * 2026-08-25): the in-place caller consults its OWN
             * ord=scr lay=il mode cell — the same cell the sub-2048 ILP
             * race banks into, mode=zcasc as the third verdict. Single
             * writer per key: sub-2048 writes ilp|conv, >=2048 writes
             * zcasc|conv, the kind-4 place=oop cell stays the OOP
             * create's alone (recipe source here). On a MISS the cascade
             * candidate (aliased t2q pick) races THIS caller's convert
             * incumbent — the cascade is alias-safe in==out (P0a). BOTH
             * spellings: DEFAULT-order in-place is the scrambled-output
             * contract (identity rule). */
            if (N >= 2048 && !W->vw2_off_stride && h->cplan &&
                !getenv("VFFT_NO_K1Z_IP"))
            {
                vfft_proto_nat_entry_t zieb;
                const vfft_proto_nat_entry_t *zie =
                    vw2_stride_lookup_scrmode(&W->vw2, _vw2_lay_of(cfg), N,
                                              K, &zieb)
                        ? &zieb
                        : NULL;
                const int zmode = (zie && !cfg->recalibrate)
                                      ? zie->mode
                                      : VFFT_NAT_UNSET;
                vfft_zsplit_plan_t *ipzs = NULL;
                vfft_zturn2_plan_t *ipzt = NULL;
                int ipzr = 0;
                if (zmode == VFFT_NAT_ZCASC)
                {
                    /* the banked win: rebuild — recipe from the kind-4
                     * OOP row when banked, else the default construction
                     * (aliased t2q pick, no kind-4 bank) */
                    if (_k1z_wisdom_replay(cfg, W, N, &ipzs, &ipzt,
                                           &ipzr) ||
                        _k1z_race_and_bank(cfg, W, N, /*ip=*/1, &ipzs,
                                           &ipzt, &ipzr))
                    {
                        h->zsplit = ipzs; /* one non-NULL (atomicity) */
                        h->zturn = ipzt;
                        h->zroute = ipzr;
                    }
                }
                else if (zmode == VFFT_NAT_UNSET &&
                         _k1z_race_and_bank(cfg, W, N, /*ip=*/1, &ipzs,
                                            &ipzt, &ipzr))
                {
                    /* MISS: cascade vs THIS caller's convert incumbent —
                     * the ILP race protocol (5 rounds alternated,
                     * medians, aliased buffer re-seeded per burst). */
                    double *rz = (double *)malloc(2 * (size_t)N
                                                  * sizeof(double));
                    double *r0 = (double *)malloc(2 * (size_t)N
                                                  * sizeof(double));
                    if (rz && r0)
                    {
                        const int reps = N <= 4096 ? 32 : 8;
                        double ns[2]; /* [0] incumbent, [1] zcasc */
                        _c2c_race_ctx_t rc = { h, 0, ipzt, ipzs, ipzr, NULL, NULL, NULL, rz, r0,
                                                2 * (size_t)N * sizeof(double) };
                        const vfft_race_arm_t arms[2] = {
                            { "incumbent", _c2c_race_inc, &rc }, { "zcasc", _c2c_race_chal, &rc } };
                        /* 5 rounds, odd rounds reversed, median-of-5; reseed per burst */
                        const vfft_race_proto_t proto = { 5, reps, VFFT_RACE_MEDIAN, 1, 0, _c2c_race_reseed, &rc };
                        for (long i2 = 0; i2 < 2L * N; i2++)
                            r0[i2] = (double)rand() / RAND_MAX - 0.5;
                        _vfft_pool_arm(h->nthreads);
                        vfft_race_run(&proto, arms, 2, ns);
                        if (getenv("VFFT_NAT_LOG"))
                            fprintf(stderr,
                                    "[scrmode] N=%d K=%zu conv=%.0fns "
                                    "zcasc=%.0fns -> %s\n",
                                    N, K, ns[0], ns[1],
                                    ns[1] < ns[0] ? "ZCASC" : "conv");
                        if (ns[1] < ns[0])
                        {
                            h->zsplit = ipzs;
                            h->zturn = ipzt;
                            h->zroute = ipzr;
                            ipzs = NULL;
                            ipzt = NULL;
                            _bank_scrmode_1d(W, cfg, N, K,
                                             VFFT_NAT_ZCASC, ns[1],
                                             h->cplan->factors,
                                             h->cplan->variants,
                                             h->cplan->num_stages,
                                             h->cplan->use_dif_forward);
                        }
                        else
                            _bank_scrmode_1d(W, cfg, N, K, VFFT_NAT_CONV,
                                             ns[0], h->cplan->factors,
                                             h->cplan->variants,
                                             h->cplan->num_stages,
                                             h->cplan->use_dif_forward);
                    }
                    free(rz);
                    free(r0);
                    if (ipzs)
                        vfft_zsplit_destroy(ipzs);
                    if (ipzt)
                        vfft_zturn2_destroy(ipzt);
                }
                /* zmode == CONV: the banked loss — convert serves. */
            }
            /* THE ILP-ATTACH FIX (owner law 2026-08-25: everything is
             * measured — a scrambled caller never waits on a natural
             * caller to have raced first). The old design served the
             * @nat verdict HIT-ONLY ("single @nat writer"): a
             * scrambled-only user fell to convert FOREVER, a measured
             * 4-5.5x tax with the native engines one attach away. Now:
             * consult the caller's OWN ord=scr mode cell; on a miss RUN
             * THE RACE (the natural race's exact protocol, against THIS
             * caller's convert incumbent) and bank BOTH outcomes
             * (mode=ilp | mode=conv — the banked loss, no re-race). */
            if (!h->zsplit && !h->zturn && N < 2048 &&
                !getenv("VFFT_NO_NAT_ILP"))
            {
                vfft_proto_nat_entry_t nieb;
                const vfft_proto_nat_entry_t *nie =
                    W->vw2_off_stride
                        ? NULL
                        : (vw2_stride_lookup_scrmode(
                               &W->vw2, _vw2_lay_of(cfg), N, K, &nieb)
                               ? &nieb
                               : NULL);
                if (nie && !cfg->recalibrate &&
                    nie->mode == VFFT_NAT_ILP)
                {
                    _k1_il_candidate(W, N, &h->k1il2p, &h->k1il3p);
                    if (!h->k1il2p && !h->k1il3p)
                        h->k1ilpr = vfft_ilprime_create(N); /* prime cell */
                }
                else if ((!nie || cfg->recalibrate) &&
                         !W->vw2_off_stride)
                {
                    vfft_il2p_plan_t *ilc2 = NULL;
                    vfft_il3p_plan_t *ilc3 = NULL;
                    vfft_ilprime_plan_t *ilcp = NULL;
                    _k1_il_candidate(W, N, &ilc2, &ilc3);
                    if (!ilc2 && !ilc3)
                        ilcp = vfft_ilprime_create(N); /* self-validates */
                    if (ilc2 || ilc3 || ilcp)
                    {
                        double *rz = (double *)malloc(
                            2 * (size_t)N * sizeof(double));
                        double *r0 = (double *)malloc(
                            2 * (size_t)N * sizeof(double));
                        if (rz && r0)
                        {
                            const int reps =
                                N <= 256 ? 200
                                         : (N <= 1024 ? 80 : 32);
                            double ns[2]; /* [0] incumbent, [1] ilp */
                            _c2c_race_ctx_t rc = { h, 0, NULL, NULL, 0, ilc2, ilc3, ilcp, rz, r0,
                                                    2 * (size_t)N * sizeof(double) };
                            const vfft_race_arm_t arms[2] = {
                                { "incumbent", _c2c_race_inc, &rc }, { "ilp", _c2c_race_chal, &rc } };
                            /* 5 rounds, odd rounds reversed, median-of-5; reseed per burst */
                            const vfft_race_proto_t proto = { 5, reps, VFFT_RACE_MEDIAN, 1, 0, _c2c_race_reseed, &rc };
                            for (long i2 = 0; i2 < 2L * N; i2++)
                                r0[i2] = (double)rand() / RAND_MAX - 0.5;
                            _vfft_pool_arm(h->nthreads);
                            vfft_race_run(&proto, arms, 2, ns);
                            if (getenv("VFFT_NAT_LOG"))
                                fprintf(stderr,
                                        "[scrmode] N=%d K=%zu conv=%.0fns "
                                        "ilp=%.0fns -> %s\n",
                                        N, K, ns[0], ns[1],
                                        ns[1] < ns[0] ? "ILP" : "conv");
                            if (ns[1] < ns[0])
                            {
                                h->k1il2p = ilc2;
                                h->k1il3p = ilc3;
                                h->k1ilpr = ilcp;
                                ilc2 = NULL;
                                ilc3 = NULL;
                                ilcp = NULL;
                                _bank_scrmode_1d(
                                    W, cfg, N, K, VFFT_NAT_ILP, ns[1],
                                    h->cplan->factors,
                                    h->cplan->variants,
                                    h->cplan->num_stages,
                                    h->cplan->use_dif_forward);
                            }
                            else
                                _bank_scrmode_1d(
                                    W, cfg, N, K, VFFT_NAT_CONV, ns[0],
                                    h->cplan->factors,
                                    h->cplan->variants,
                                    h->cplan->num_stages,
                                    h->cplan->use_dif_forward);
                        }
                        free(rz);
                        free(r0);
                        if (ilc2)
                            vfft_il2p_destroy(ilc2);
                        if (ilc3)
                            vfft_il3p_destroy(ilc3);
                        if (ilcp)
                            vfft_ilprime_destroy(ilcp);
                    }
                }
                /* mode==CONV: the banked loss — convert serves, no
                 * re-race. */
            }
        }
        /* The pad-vs-tail decision serves the LANE-MAJOR interleaved batch;
         * the transform-contiguous geometry wraps a K=1 plan instead
         * (vfft.c ~2962) and never arrives here with K>1. */
        if (cfg->layout == VFFT_LAYOUT_INTERLEAVED && K > 1)
            _il_me_decide(W, cfg, h); /* D6: the fused-vs-padded A/B at create */
        return _c2c_ip_finish(h);
    }
    return NULL; /* unreachable: the one call site guards on the same
                  * condition, and every path in the block above returns. */
}

#endif /* VFFT_OOP_C2C_IP_CREATE_H */
