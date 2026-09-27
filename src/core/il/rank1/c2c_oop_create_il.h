/* c2c_oop_create_il.h — the c2c OUT-OF-PLACE create, interleaved tier.
 *
 * The K=1 engines (mono, pair, chain3, flat, ZTURN-T, four-step, prime): the
 * kind-3 row's IL axis replayed, or raced and banked (_k1_il_plan_race). A
 * cell with no IL engine REFUSES — split is not a fallback of IL; K>1 (the
 * explicit lane-major batch) refuses too. Depends on common/ and il/ only.
 *
 * POSITION IN vfft.c IS LOAD-BEARING: calls file-scope statics of vfft.c, so
 * it is included (via the dispatcher) after those are defined.
 */
#ifndef VFFT_IL_C2C_OOP_CREATE_IL_H
#define VFFT_IL_C2C_OOP_CREATE_IL_H

/* ── the IL tier's ONE exit. Every handle it returns passes through
 * here; a shared post-step cannot be skipped by a new early exit without
 * the skip being spelled at the call: the threaded verdicts of the K=1
 * engines that have one. zt_mt is unused (callers pass 0). */
static vfft_plan _c2c_oop_finish_il(struct vfft_plan_s *h, int zt_mt,
                                 struct vfft_wisdom_s *W,
                                 const vfft_config_t *cfg, int N)
{
    (void)zt_mt;
    if (h->k1ilfd && h->K == 1 && h->nthreads > 1)
        _ilfd_mt_replay_or_race(h, W, cfg, N); /* the flat DIT's, per-T banked */
    if (h->k1ztt && h->K == 1 && h->nthreads > 1)
        _ztt_mt_replay_or_race(h, W, cfg, N);  /* ZTURN-T's, per-T banked */
    if (h->k1fs && h->K == 1 && h->nthreads > 1)
        _k1fs_mt_replay_or_race(h, W, cfg, N); /* the four-step's split at T, per-T banked */
    return h;
}

static struct vfft_plan_s *_c2c_oop_create_k1_il(const vfft_config_t *cfg,
                                                 struct vfft_wisdom_s *W,
                                                 int N)
{
    int ilr = VFFT_K1_IL_2P;
    int iR1 = 0, iR2 = 0;
    /* the kind-3 cell's INTERLEAVED axis (il/wisdom/wisdom2_oop_il.h):
     * kin the natural cell, ki the request's order cell. The vw2_off_oop
     * kill switch serves it from the one legacy line. */
    vfft_oop_il_entry_t kib, kinb;
    const vfft_oop_wisdom_entry_t *kle =
        W->vw2_off_oop ? vfft_oop_wisdom_lookup_k1(&W->oop, N) : NULL;
    if (kle)
        vfft_oop_il_from_legacy(&kinb, kle);
    const vfft_oop_il_entry_t *kin =
        W->vw2_off_oop ? (kle ? &kinb : NULL)
                       : (vw2_oop_lookup_k1_il_cell(&W->vw2, N, 0, 0, _vfft_plan_threads(cfg), &kinb) ? &kinb : NULL);
    /* the IL axis reads the request's ORDER CELL: an
     * explicit SCRAMBLED request takes the ord=scr row — the
     * scrambled pool's own verdict (a natural-output engine, or the
     * flat DIT's scrambled class) — DEFAULT and NATURAL the ord=nat
     * row (kin). Two cells, never compared. */
    const int scr_req = (vfft_policy_ord_k1(cfg, N, /*inplace=*/0) == VW2_ORD_SCR &&
                         cfg->layout == VFFT_LAYOUT_INTERLEAVED && !W->vw2_off_oop);
    const vfft_oop_il_entry_t *ki =
        scr_req ? (vw2_oop_lookup_k1_il_cell(&W->vw2, N, 1, 0, _vfft_plan_threads(cfg), &kib) ? &kib : NULL) : kin;
    /* the threaded plan's row (v1.3): the K=1 route is thread-independent,
     * so a miss at T > 1 takes the one-thread row's route as its own row */
    if (cfg->layout == VFFT_LAYOUT_INTERLEAVED && !W->vw2_off_oop && !cfg->recalibrate &&
        _vfft_plan_threads(cfg) > 1 && !ki &&
        vw2_oop_k1_row_at_T(&W->vw2, N, scr_req, 0, _vfft_plan_threads(cfg)))
    {
        _vw2_persist(W, cfg);
        kin = vw2_oop_lookup_k1_il_cell(&W->vw2, N, 0, 0, _vfft_plan_threads(cfg), &kinb) ? &kinb : NULL;
        ki = scr_req ? (vw2_oop_lookup_k1_il_cell(&W->vw2, N, 1, 0, _vfft_plan_threads(cfg), &kib) ? &kib : NULL) : kin;
    }
    /* Per-layout wisdom: the IL axis is read from the store on its own;
     * the split axis of the same cell (split/rank1/c2c_oop_create_split.h)
     * never degrades it. */
    /* WISDOM OR RACE: an interleaved kind-3 miss (or recalibrate)
     * races here and banks before this block reads the row, so the
     * pair default below is never the source of a served IL plan
     * (its form-less pair would differ from the planner's in bits).
     * _k1_il_plan_race carries the N gate (vfft_policy_races). */
    if (cfg->layout == VFFT_LAYOUT_INTERLEAVED &&
        !W->vw2_off_oop &&
        (cfg->recalibrate || !ki || !ki->il_kv_raced))   /* a pair-only row (forms unraced) plans too */
    {
        if (_k1_il_plan_race(W, cfg, N) > 0)
        {
            if (_vfft_plan_threads(cfg) > 1 && vw2_oop_k1_row_at_T(&W->vw2, N, scr_req, 0, _vfft_plan_threads(cfg)))
                _vw2_persist(W, cfg);   /* the race banked the one-thread row: its copy at T */
            kin = vw2_oop_lookup_k1_il_cell(&W->vw2, N, 0, 0, _vfft_plan_threads(cfg), &kinb) ? &kinb : NULL;
            ki = scr_req ? (vw2_oop_lookup_k1_il_cell(&W->vw2, N, 1, 0, _vfft_plan_threads(cfg), &kib) ? &kib : NULL) : kin;
        }
    }
    /* il_banked: k1_il_route = -1 means the IL axis
     * was never raced at this cell — run the IL default, exactly as
     * an unbanked cell would. IL_NONE (0) is a VERDICT ("raced: no IL
     * route available") and is consumed as one. */
    const int il_banked = (ki && ki->k1_il_route >= 0);
    if (il_banked)
    {
        ilr = ki->k1_il_route;
        iR1 = ki->il_R1;
        iR2 = ki->il_R2;
    }
    if (!il_banked)
    {
        /* IL runs its OWN pair search — it must NOT inherit sR1/sR2:
         *  (a) COVERAGE. The split loop filters on SPLIT availability
         *      (vfft_oop_leaf_fn / vfft_oop_t1_fn, which reach R=128),
         *      but the il2p registries (vfft_il2p_leaf_fn /
         *      vfft_il2p_mid_fn) stop at R=64: at N=16384 the split
         *      pick is 128x128 and BOTH IL halves are NULL. A 2-pass IL
         *      route needs R1*R2 = N with both <= 64, so it tops out at
         *      N=4096; above that the answer is IL_NONE.
         *  (b) INDEPENDENCE. Even where a split pair is legal for IL,
         *      nothing makes it the IL optimum -- the two arms run
         *      different codelets over different layouts.
         * This is only the default for a cell with no banked verdict;
         * the IL planner (planning/dp_planner_il.h) measures the axis. */
        if (vfft_k1_mono_il_fn(N, 0))
        {
            ilr = VFFT_K1_IL_MONO; /* mono is whole-N; pair unused */
        }
        else
        {
            for (int R2c = (N < 64 ? N : 64); R2c >= 4; R2c--)
            {
                if (N % R2c)
                    continue;
                int R1c = N / R2c;
                /* No parity constraint: every monolithic cil kernel
                 * carries the inline VEX-128 odd-count tail, so odd
                 * factors are legal — all-odd pairs (45 = 9x5) and
                 * 2·odd pairs (50 = 5x10) route natively. The
                 * registry probes below are the only availability
                 * filter. */
                if (R1c < 3 || R1c > 64)
                    continue;
                if (!vfft_il2p_leaf_fn(R2c, 0) || !vfft_il2p_mid_fn(R1c, 0))
                    continue;
                if (!iR1 || abs(R1c - R2c) < abs(iR1 - iR2))
                {
                    iR1 = R1c;
                    iR2 = R2c;
                }
            }
            ilr = iR1 ? VFFT_K1_IL_2P_PURE : VFFT_K1_IL_NONE;
        }
    }
    vfft_il2p_plan_t *il2p = NULL;
    /* WHITELIST, not a blacklist: only the pair-based IL routes build
     * a plan from iR1/iR2. MONO is whole-N, NONE has no route, and
     * the cascade's route value 4 is retired (oop_plan.h) --
     * a growing "!= this && != that" chain would silently start
     * building plans for any IL route added later.
     *
     * il2p is the ONLY pair-based IL machinery, so the legacy wisdom aliases
     * (3P=1, 2P=2) and the canonical 2P_PURE all normalize to ONE
     * il2p attempt on the same (iR1,iR2) pair. Route stays TRUTHFUL:
     * it names 2P_PURE iff the plan exists, else NONE — execute never
     * dereferences a NULL k1il2p. Kill-switch: env VFFT_NO_IL2P
     * disables the whole pair-based IL axis (mono is unaffected). */
    if ((ilr == VFFT_K1_IL_2P || ilr == VFFT_K1_IL_3P ||
         ilr == VFFT_K1_IL_2P_PURE))
    {
        if (iR1 && !getenv("VFFT_NO_IL2P"))
        {   /* braces are load-bearing: apply_kv must not run when the
             * pair axis was skipped (it survived unbraced only because
             * it null-checks — a latent trap, not a working shortcut) */
            il2p = vfft_il2p_create(N, iR1, iR2);
            _k1_il2p_apply_kv(il2p, ki, &W->vw2, N, 0);   /* banked variant verdict (the out-of-place cell) */
        }
        ilr = il2p ? VFFT_K1_IL_2P_PURE : VFFT_K1_IL_NONE;
    }
    /* 3-STAGE CHAIN (route 6): attempted when the pair axis came up
     * empty, and only for INTERLEAVED-committed plans (an IL-only
     * handle may carry spr == -1; the split dispatch must never see
     * one). */
    vfft_il3p_plan_t *il3p = NULL;
    /* a banked chain3 row names the route up front (ilr == CHAIN3
     * from ki): it must reach this block, not the degrade below */
    if ((ilr == VFFT_K1_IL_NONE || ilr == VFFT_K1_IL_CHAIN3) && !il2p &&
        !getenv("VFFT_NO_IL2P") &&
        cfg->layout == VFFT_LAYOUT_INTERLEAVED)
    {
        int cR2, cA, cB;
        /* the BANKED chain first (the planner races every legal
         * 3-stage chain and banks il_chain=R2.A.B); the legal default
         * only for an uncalibrated cell */
        if (ki && ki->k1_il_route == VFFT_K1_IL_CHAIN3 && ki->il_c3[0])
        {
            il3p = vfft_il3p_create(N, ki->il_c3[0], ki->il_c3[1],
                                    ki->il_c3[2]);
            _k1_il3p_apply_kv(il3p, ki, &W->vw2, N, 0);   /* banked forms > default (the out-of-place cell) */
            if (il3p && getenv("VFFT_NAT_LOG"))
                fprintf(stderr, "[k1c3] N=%d: replay chain %d.%d.%d "
                                "src=wisdom (oop)\n", N, ki->il_c3[0],
                        ki->il_c3[1], ki->il_c3[2]);
        }
        if (!il3p && vfft_il3p_default_chain(N, &cR2, &cA, &cB))
            il3p = vfft_il3p_create(N, cR2, cA, cB);
        ilr = il3p ? VFFT_K1_IL_CHAIN3 : VFFT_K1_IL_NONE;
    }
    /* FLAT DIT (route 8): the odd-N flat mixed-radix DIT
     * (oop/il_flatdit.h). A banked verdict replays its chain and
     * per-stage forms; there is NO default build here — the K=1 plan
     * race above is the only source of a flat plan (never a rule). */
    vfft_ilfd_plan_t *ilfd = NULL;
    if (ilr == VFFT_K1_IL_FLAT && !il2p && !il3p && !getenv("VFFT_NO_IL2P") &&
        cfg->layout == VFFT_LAYOUT_INTERLEAVED && ki && ki->il_fl_n >= 2)
    {
        if (scr_req)
            ilfd = vfft_ilfd_create_scr_of(N, ki->il_fl, ki->il_fl_n, ki->il_flf, ki->il_tw);
        else
        {
            ilfd = vfft_ilfd_create_chain(N, ki->il_fl, ki->il_fl_n);
            if (ilfd && (!ilfd->bwd_ok ||
                         (ki->il_flf[0] && !vfft_ilfd_apply_forms(ilfd, ki->il_flf)) ||
                         (ki->il_tw > 0 && !vfft_ilfd_apply_tw(ilfd, ki->il_tw))))
            {
                vfft_ilfd_destroy(ilfd);
                ilfd = NULL;
            }
        }
        if (ilfd && getenv("VFFT_NAT_LOG"))
        {
            char chs[48];
            int q, off = 0;
            for (q = 0; q < ki->il_fl_n && off < (int)sizeof chs - 4; q++)
                off += snprintf(chs + off, sizeof chs - (size_t)off, "%s%d", q ? "." : "", ki->il_fl[q]);
            fprintf(stderr, "[k1fd] N=%d: replay flat chain %s (forms %s, tw %d, %s) "
                            "src=wisdom (oop)\n",
                    N, chs, ki->il_flf[0] ? ki->il_flf : "-", ki->il_tw,
                    scr_req ? "SCRAMBLED class" : "natural");
        }
    }
    if (ilr == VFFT_K1_IL_FLAT && !ilfd)
        ilr = VFFT_K1_IL_NONE;      /* truthful: the route names a plan that exists */
    /* ZTURN-T (route 9): a banked verdict replays its chain
     * (il_ztt=) through the create — the registry cell's fused codelets;
     * NO default build (the planner is the only source). The ORDER
     * CLASS is the row's: ord=nat replays the natural
     * drivers, ord=scr the PLAIN schedule (ztt_scrambled_design.md) —
     * one plan, one order, never mixed. */
    vfft_ztt_plan_t *ztt = NULL;
    if (ilr == VFFT_K1_IL_ZTT && !il2p && !il3p && !ilfd && !getenv("VFFT_NO_IL2P") &&
        cfg->layout == VFFT_LAYOUT_INTERLEAVED && ki && ki->il_zt_n >= 2)
    {
        ztt = vfft_ztt_create_chain_ord(N, ki->il_zt, ki->il_zt_n, scr_req);
        if (ztt && ki->il_tw > 0 && !vfft_ztt_set_tile(ztt, (size_t)ki->il_tw))
        {   /* the row names a tile the cell refuses: not a plan that exists */
            vfft_ztt_destroy(ztt);
            ztt = NULL;
        }
        if (ztt && getenv("VFFT_NAT_LOG"))
        {
            char chs[48];
            vfft_ztt_chain_str(ztt, chs, sizeof chs);
            fprintf(stderr, "[k1ztt] N=%d: replay ZTURN-T chain %s tile=%zu src=wisdom (oop)\n", N, chs, ztt->tile);
        }
    }
    if (ilr == VFFT_K1_IL_ZTT && !ztt)
        ilr = VFFT_K1_IL_NONE;      /* truthful: the route names a plan that exists */
    /* the FOUR-STEP (route 10): a banked split (il_pair =
     * N1.N2) replays through the create — the 2D child out of place at
     * the plan's thread count, the order class the row's */
    vfft_k1fs_plan_t *fs = NULL;
    if (ilr == VFFT_K1_IL_FS && !il2p && !il3p && !ilfd && !ztt && !getenv("VFFT_NO_IL2P") &&
        cfg->layout == VFFT_LAYOUT_INTERLEAVED && ki && ki->il_R1 > 0 && ki->il_R2 > 0 &&
        (long)ki->il_R1 * (long)ki->il_R2 == (long)N)
    {
        int pn1 = ki->il_R1, pn2 = ki->il_R2, sbc[8], sbn = 0, form = 0;
        const int pinned = _k1fs_pin(N, &pn1, &pn2);
        if (!scr_req) sbn = _k1fs_row_sb(W, N, ki->il_kv, sbc, &form);
        fs = vfft_k1fs_create(N, pn1, pn2, scr_req, W, cfg, 0, _vfft_plan_threads(cfg), form, sbc, sbn);
        if (fs && getenv("VFFT_NAT_LOG"))
            fprintf(stderr, "[k1fs] N=%d: replay FOUR-STEP %dx%d form=%d src=%s (oop)\n", N, fs->N1, fs->N2,
                    fs->form, pinned ? "pin" : "wisdom");
    }
    if (ilr == VFFT_K1_IL_FS && !fs)
        ilr = VFFT_K1_IL_NONE;      /* truthful: the route names a plan that exists */
    /* PRIME N (route 7): Rader/Bluestein on the IL machinery
     * (il_prime.h) — the OOP INTERLEAVED prime coverage the split
     * OOP path refuses. Same IL-only-handle rules as the chain. */
    vfft_ilprime_plan_t *ilpr = NULL;
    /* the prime cell is a route, not a fallback: a power of two is
     * never its cell (the pow2 tiers race on a miss, above). It is a
     * RACED ARM: a banked il_route=prime row replays it (the race's own
     * plan when the race just ran, else the prime shard's), and a cell
     * with no verdict at all -- above the race ceiling -- builds it
     * unraced. */
    if ((ilr == VFFT_K1_IL_NONE || ilr == VFFT_K1_IL_PRIME) &&
        !il2p && !il3p && !ilfd && !ztt && !fs &&
        (N & (N - 1)) != 0 &&
        !getenv("VFFT_NO_IL2P") &&
        cfg->layout == VFFT_LAYOUT_INTERLEAVED)
    {
        if (ilr == VFFT_K1_IL_PRIME && _k1pr_ctx.plan && _k1pr_ctx.N == N)
        {
            ilpr = _k1pr_ctx.plan;
            _k1pr_ctx.plan = NULL;
            _k1pr_ctx.N = 0;
        }
        else
            ilpr = _ilprime_create_banked(W, cfg, N);
        _k1pr_release();
        ilr = ilpr ? VFFT_K1_IL_PRIME : VFFT_K1_IL_NONE;   /* truthful: the route names a plan that exists */
        if (ilpr && getenv("VFFT_NAT_LOG"))
            fprintf(stderr, "[k1pr] N=%d: out of place prime cell (%s, M=%d) src=%s\n", N,
                    ilpr->method ? "RADER" : "BLUESTEIN", ilpr->M,
                    (ki && ki->k1_il_route == VFFT_K1_IL_PRIME) ? "wisdom" : "door");
    }
    /* (2P/3P/2P_PURE availability is settled by the normalize block
     * above — the route already names 2P_PURE iff il2p exists.) */
    /* validate the form the handle will RESOLVE, not form 0. The
     * resolution below takes the row's
     * banked il_kv, and form 1 exists only at N = 64 -- so a row
     * carrying il_route=MONO il_kv=1 at any other N passed a form-0
     * check here and then resolved to NULL pointers that
     * vfft_execute.h calls unguarded. The planner never banks that
     * pair, so reaching it means a hand-edited or foreign store,
     * which is exactly what a validator is for. The expression is the
     * resolution's own, character for character. */
    if (ilr == VFFT_K1_IL_MONO)
    {
        const int mf = (ki && ki->k1_il_route == VFFT_K1_IL_MONO)
                           ? ki->il_kv : 0;
        if (!vfft_k1_mono_il_form_fn(N, mf, 0) ||
            !vfft_k1_mono_il_form_fn(N, mf, 1))
            ilr = VFFT_K1_IL_NONE;
    }
    /* Handle exists when ANY IL route does. 🔴 Every IL engine MUST be in
     * this guard, or a cell with an IL plan would drop to the "no
     * interleaved engine" refusal. IL handles are INTERLEAVED-committed
     * (k1_sp_route == -1), so the split dispatch never sees one. */
    if (vfft_policy_k1_engine_present(
            /* mono   */ ilr == VFFT_K1_IL_MONO && cfg->layout == VFFT_LAYOUT_INTERLEAVED,
            /* pair   */ il2p && cfg->layout == VFFT_LAYOUT_INTERLEAVED,
            /* chain3 */ il3p != NULL, /* flat */ ilfd != NULL,
            /* ztt    */ ztt != NULL,  /* fs   */ fs != NULL,
            /* prime  */ ilpr != NULL))
        /* MONO is ROUTE presence, not pointer presence: the solo tier has no plan
         * object and its pointers are resolved ~30 lines below. Do NOT "correct"
         * this to hk->k1_mono_ilf -- that is a different question, asked later. */
    {
        struct vfft_plan_s *hk =
            (struct vfft_plan_s *)calloc(1, sizeof *hk);
        if (hk)
        {
            hk->transform = VFFT_C2C;
            hk->placement = VFFT_OUTOFPLACE;
            hk->layout = (int)cfg->layout;
            hk->N = N;
            hk->K = 1;
            hk->nthreads = _vfft_plan_threads(cfg);
            hk->k1_on = 1;
            hk->k1_sp_route = -1;   /* an IL handle has no split route */
            hk->k1_il_route = ilr;
            /* PURE-IL pair route, both directions (created and
             * route-normalized above; non-NULL iff ilr==2P_PURE). */
            hk->k1il2p = il2p;
            /* 3-stage chain route (non-NULL iff ilr==CHAIN3). */
            hk->k1il3p = il3p;
            /* prime route (non-NULL iff ilr==IL_PRIME). */
            hk->k1ilpr = ilpr;
            /* flat DIT route (non-NULL iff ilr==IL_FLAT). */
            hk->k1ilfd = ilfd;
            /* ZTURN-T route (non-NULL iff ilr==IL_ZTT). */
            hk->k1ztt = ztt;
            hk->k1fs = fs;
            {   /* MONO form = the banked il_kv on a MONO verdict
                 * (0 = solo n1, 1 = mono64); form 0 otherwise */
                const int mf = (ki && ki->k1_il_route == VFFT_K1_IL_MONO)
                                   ? ki->il_kv : 0;
                hk->k1_mono_ilf = vfft_k1_mono_il_form_fn(N, mf, 0);
                hk->k1_mono_ilb = vfft_k1_mono_il_form_fn(N, mf, 1);
            }
            return hk;
        }
    }
    vfft_il2p_destroy(il2p);
    vfft_il3p_destroy(il3p);
    vfft_ilprime_destroy(ilpr);
    vfft_ilfd_destroy(ilfd);
    vfft_ztt_destroy(ztt);
    vfft_k1fs_destroy(fs);   /* every engine this tier can build is released here */
    return NULL;
}

static vfft_plan _vfft_create_c2c_oop_il(const vfft_config_t *cfg,
                                        struct vfft_wisdom_s *W,
                                        int N,
                                        size_t K)
{
    if (K == 1)
    {
        struct vfft_plan_s *hk = _c2c_oop_create_k1_il(cfg, W, N);
        if (hk)
            return _c2c_oop_finish_il(hk, 0, W, cfg, N);
    }
    /* No split champion behind a convert for an interleaved caller.
     * K=1 arrives here only when the IL route selection above found
     * no engine; K>1 is the explicit lane-major batch (DEFAULT
     * geometry is the transform-contiguous wrapper), which lost to
     * transform-contiguous at every measured cell. */
    _vfft_warn("vfft_create: out-of-place C2C N=%d howmany=%zu with "
               "layout=INTERLEAVED has no interleaved engine (%s); nothing to "
               "fall back to by design",
               N, K,
               K > 1 ? "lane-major batches are not an IL route - use "
                        "VFFT_BATCH_DEFAULT / TRANSFORM_CONTIGUOUS"
                      : "no mono/pair/chain3/prime kernel serves this N");
    return NULL;
}

#endif /* VFFT_IL_C2C_OOP_CREATE_IL_H */
