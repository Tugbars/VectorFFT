/* c2c_oop_create.h — the c2c OUT-OF-PLACE create tier.
 *
 * The c2c out-of-place arm of _vfft_create_inner; it returns on every path.
 *
 * OOP IS NOT IN-PLACE WITH A COPY BOLTED ON
 * -----------------------------------------
 * It is a separate serving with its own wisdom family (W->oop / vw2 oop
 * records) and its own K=1 route. That is why this tier is a sibling of
 * c2c_ip_create.h rather than a branch inside it: the two consult different
 * banked verdicts and build different plans.
 *
 * THE K=1 CASE
 * ------------
 * `K == 1 && !ob` is the K=1 tier: the interleaved routes (mono, pair,
 * chain3, flat, ZTURN-T, four-step, prime) replay the banked kind-3 row or
 * race and bank it (_k1_il_plan_race); the split routes replay their own
 * lay=split row. The tier does not choose an engine by rule.
 *
 * `ob` splits the same way it does in-place: a caller-supplied batch handle is
 * checked and served exactly, otherwise the plan owns its buffers.
 *
 * WISDOM, NOT HEURISTIC
 * ---------------------
 * Every interleaved choice here is replayed from a banked verdict or raced
 * and then banked. The split side's uncalibrated default is structural until
 * calibrate_k1_split.c banks the cell.
 *
 * POSITION IN vfft.c IS LOAD-BEARING
 * ----------------------------------
 * Not a standalone header. It calls file-scope statics that live in vfft.c, so
 * it must be included after those are defined and before _vfft_create_inner.
 */
#ifndef VFFT_OOP_C2C_OOP_CREATE_H
#define VFFT_OOP_C2C_OOP_CREATE_H

/* ── the tier's ONE exit. Every handle this create returns passes through
 * here; a shared post-step cannot be skipped by a new early exit without
 * the skip being spelled at the call: the threaded verdicts of the K=1
 * engines that have one. zt_mt is unused (callers pass 0). */
static vfft_plan _c2c_oop_finish(struct vfft_plan_s *h, int zt_mt,
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

static vfft_plan _vfft_create_c2c_oop(const vfft_config_t *cfg,
                                      vfft_batch ob,
                                      struct vfft_wisdom_s *W,
                                      const vfft_proto_registry_t *reg,
                                      int N,
                                      size_t K)
{
    if (cfg->transform == VFFT_C2C && cfg->placement == VFFT_OUTOFPLACE)
    {
        /* ── the K=1 engine: routes from the kind-3 row (replayed) or the
         * race; execute dispatches on the COMMITTED layout axis
         * (config.layout, stamped on the handle). An interleaved request that
         * finds no K=1 engine is refused below; a split request falls through
         * to the classic OOP path when its K=1 plan cannot be built.
         *
         * ORDER IS A CONTRACT: an explicit SCRAMBLED request reads the ord=scr
         * row — the scrambled pool's own verdict (planning/policy.h), raced on
         * a miss; DEFAULT and NATURAL read the ord=nat row. */
        {
        if (K == 1 && !ob)
        {
            int spr = VFFT_K1_SP_2PB, ilr = VFFT_K1_IL_2P;
            int sR1 = 0, sR2 = 0, iR1 = 0, iR2 = 0;
            vfft_oop_wisdom_entry_t keb, kib;
            const vfft_oop_wisdom_entry_t *ke =
                W->vw2_off_oop ? vfft_oop_wisdom_lookup_k1(&W->oop, N)
                               : (vw2_oop_lookup_k1_cell(&W->vw2, N, 0, 0, _vfft_plan_threads(cfg), &keb) ? &keb : NULL);
            /* the IL axis reads the request's ORDER CELL: an
             * explicit SCRAMBLED request takes the ord=scr row — the
             * scrambled pool's own verdict (a natural-output engine, or the
             * flat DIT's scrambled class) — DEFAULT and NATURAL the ord=nat
             * row (ke). Two cells, never compared. The split axis keeps ke. */
            const int scr_req = (vfft_policy_ord_k1(cfg, N, /*inplace=*/0) == VW2_ORD_SCR &&
                                 cfg->layout == VFFT_LAYOUT_INTERLEAVED && !W->vw2_off_oop);
            const vfft_oop_wisdom_entry_t *ki =
                scr_req ? (vw2_oop_lookup_k1_cell(&W->vw2, N, 1, 0, _vfft_plan_threads(cfg), &kib) ? &kib : NULL) : ke;
            /* Per-layout wisdom: each axis is taken from the store
             * INDEPENDENTLY. A cell with only an IL verdict (k1_sp_route < 0 —
             * e.g. non-pow2 N, where split cannot factor) keeps the banked IL
             * route while the split side runs the unbanked default; neither
             * layout's absence degrades the other. */
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
                    ke = vw2_oop_lookup_k1_cell(&W->vw2, N, 0, 0, _vfft_plan_threads(cfg), &keb) ? &keb : NULL;
                    ki = scr_req ? (vw2_oop_lookup_k1_cell(&W->vw2, N, 1, 0, _vfft_plan_threads(cfg), &kib) ? &kib : NULL) : ke;
                }
            }
            /* TWO LIBRARIES: a request names ONE layout and this door
             * resolves, builds and commits that layout's axis only. An
             * interleaved request never builds a split K=1 plan (psp) and a
             * split request never builds an interleaved one; the other axis
             * reads as absent from here on (no route, no pair, no plan). */
            const int want_il = (cfg->layout == VFFT_LAYOUT_INTERLEAVED);
            if (want_il) { spr = -1; sR1 = sR2 = 0; }
            else         { ilr = VFFT_K1_IL_NONE; iR1 = iR2 = 0; }
            const int sp_banked = (!want_il && ke && ke->k1_sp_route >= 0);
            /* il_banked mirrors sp_banked: k1_il_route = -1 means the IL axis
             * was never raced at this cell — run the IL default, exactly as
             * an unbanked cell would. IL_NONE (0) is a VERDICT ("raced: no IL
             * route available") and is consumed as one. */
            const int il_banked = (want_il && ki && ki->k1_il_route >= 0);
            if (sp_banked)
            {
                spr = ke->k1_sp_route;
                sR1 = ke->R1;
                sR2 = ke->R2;
            }
            if (il_banked)
            {
                ilr = ki->k1_il_route;
                iR1 = ki->il_R1;
                iR2 = ki->il_R2;
            }
            if (!want_il && !sp_banked)
            {
                /* structural default (uncalibrated cell): mono when emitted,
                 * else 2pb on the most balanced valid pair. The offline
                 * calibrator (benches/calibrate_k1_split.c) banks the
                 * measured lay=split row per cell. */
                if (vfft_k1_mono_fn(N) && N <= 64)
                    spr = VFFT_K1_SP_MONO;
                for (int R2c = (N < 128 ? N : 128); R2c >= 4; R2c--)
                {
                    if (N % R2c)
                        continue;
                    int R1c = N / R2c;
                    if (R1c < 4 || R1c > 128 || (R1c % 4) || (R2c % 4))
                        continue;
                    if (!vfft_oop_leaf_fn(R2c) || !vfft_oop_t1_fn(R1c))
                        continue;
                    if (!sR1 || abs(R1c - R2c) < abs(sR1 - sR2))
                    {
                        sR1 = R1c;
                        sR2 = R2c;
                    }
                }
                if (!sR1 && (N % 64) == 0 && vfft_oop_t1_fn(64))
                {
                    /* no classic pair (past the leaf/t1 reach, N >= 16384):
                     * composed column is the ONLY K=1 route up there */
                    int ccf_[VFFT_K1_CC_MAX_NF];
                    if (vfft_k1_cc_default_chain(N / 64, ccf_))
                    {
                        spr = VFFT_K1_SP_CCOL;
                        sR1 = 64;
                        sR2 = N / 64;
                    }
                }
            }
            if (want_il && !il_banked)
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
                    iR1 = sR1;
                    iR2 = sR2;
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
            vfft_oop_plan_t *psp = NULL;
            vfft_il2p_plan_t *il2p = NULL;
            if (spr == VFFT_K1_SP_CCOL && sR1)
            {
                /* composed column: chain from the wisdom line, else the
                 * per-R2 default. Create is self-validating (perm
                 * discovery); failure falls through to the classic path. */
                int ccf[VFFT_K1_CC_MAX_NF];
                int ccn = (ke && ke->cc_chain)
                              ? vfft_k1_cc_chain_decode(ke->cc_chain, ccf)
                              : vfft_k1_cc_default_chain(N / sR1, ccf);
                /* column-plan VARIANTS from the kind-3 line's own cc_vars
                 * token — the CCOL verdict is
                 * SELF-CONTAINED in OOP wisdom (an OOP operation never
                 * reads the in-place spike file at create). Decode must
                 * match the chain's nf; absent/mismatch => NULL = the T1S
                 * default. */
                const int *ccv = NULL;
                int ccv_[VFFT_K1_CC_MAX_NF];
                if (ccn && ke && ke->cc_vars &&
                    vfft_k1_cc_vars_decode(ke->cc_vars, ccn, ccv_))
                    ccv = ccv_;
                if (ccn)
                    psp = vfft_oop_plan_create_k1_cc_v(N, sR1, ccf, ccn, ccv,
                                                       _registry());
            }
            else if (spr != VFFT_K1_SP_MONO && sR1)
                psp = vfft_oop_plan_create_k1(N, sR1, sR2);
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
            if (want_il && (ilr == VFFT_K1_IL_2P || ilr == VFFT_K1_IL_3P ||
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
             * from ke): it must reach this block, not the degrade below */
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
            /* availability degrade (wisdom may name routes this build lacks).
             * Runs BEFORE spr0 is captured: spr0 keys the JIT (and the TWL
             * table pick), and the JIT must never bake a route this build
             * cannot execute (e.g. 2PB 64x128: leaf_ugul stops at 64, so
             * execute degrades to 2PA and a 2PB bake could never compile).
             * The L3-missing cases degrade to their flat base here so spr0
             * never names an l3 twin this build lacks; the fold below then
             * only ever swaps pointers that exist. */
            if (spr == VFFT_K1_SP_MONO && !vfft_k1_mono_pair_fn(N, sR1))
                spr = VFFT_K1_SP_2PB;
            if (spr != VFFT_K1_SP_MONO)
            {
                if (!psp)
                    spr = -1;
                else
                {
                    if (spr == VFFT_K1_SP_3P_L3 && !psp->t1_l3)
                        spr = VFFT_K1_SP_3P;
                    if (spr == VFFT_K1_SP_2PA_L3 && !psp->t1_ul_l3)
                        spr = VFFT_K1_SP_2PA;
                    if (spr == VFFT_K1_SP_TWL && !psp->t1_ul_twl)
                        spr = VFFT_K1_SP_2PA;
                    if (spr == VFFT_K1_SP_2PB && !psp->leaf_ul)
                        spr = VFFT_K1_SP_2PA;
                    if (spr == VFFT_K1_SP_2PA && !psp->t1_ul)
                        spr = VFFT_K1_SP_3P;
                }
            }
            int spr0 = spr; /* the EXECUTABLE wisdom route, pre-L3-fold
                             * (JIT sources + the Qlr-vs-Qr pick key on it) */
            /* log3 routes resolve to a create-time fn swap + the base route
             * (same Qr/Qi; the l3 twins are drop-in pointers — guaranteed
             * present here by the degrade above) */
            if (spr == VFFT_K1_SP_3P_L3)
            {
                psp->t1p = psp->t1_l3;
                spr = VFFT_K1_SP_3P;
            }
            if (spr == VFFT_K1_SP_2PA_L3)
            {
                psp->t1_ul = psp->t1_ul_l3;
                spr = VFFT_K1_SP_2PA;
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
            /* Handle exists when the SPLIT axis has a route, OR when ANY
             * IL-only route does. 🔴 Every IL engine MUST be in this guard: a
             * cell with an IL plan but NO split K=1 route (spr == -1 — 50 =
             * 5x10, or any N past the split routes' reach) would otherwise
             * drop to the classic path or the "no interleaved engine" refusal.
             * IL-only handles are INTERLEAVED-committed by construction
             * (every IL attempt above is layout-gated for the spr < 0 case),
             * so the split dispatch never sees k1_sp_route == -1. */
            if (spr >= 0 ||   /* the SPLIT axis's route: not engine presence, so it stays here */
                vfft_policy_k1_engine_present(
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
                    hk->k1_sp_route = spr;
                    hk->k1_il_route = ilr;
                    hk->k1sp = psp;
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
                    hk->k1_mono = vfft_k1_mono_pair_fn(N, sR1);
                    {   /* MONO form = the banked il_kv on a MONO verdict
                         * (0 = solo n1, 1 = mono64); form 0 otherwise */
                        const int mf = (ki && ki->k1_il_route == VFFT_K1_IL_MONO)
                                           ? ki->il_kv : 0;
                        hk->k1_mono_ilf = vfft_k1_mono_il_form_fn(N, mf, 0);
                        hk->k1_mono_ilb = vfft_k1_mono_il_form_fn(N, mf, 1);
                    }
#ifdef VFFT_USE_JIT
                    /* stride-baking JIT for the split route: compile
                     * cost locked to create, cached on disk forever; NULL ->
                     * the normal route fns below. TWL bakes against the
                     * linear tables. */
                    if (psp)
                    {
                        hk->k1_jit_qr = (spr0 == VFFT_K1_SP_TWL) ? psp->Qlr : psp->Qr;
                        hk->k1_jit_qi = (spr0 == VFFT_K1_SP_TWL) ? psp->Qli : psp->Qi;
                        if (hk->k1_jit_qr)
                            hk->k1_jit = vfft_k1_jit_resolve(N, sR1, sR2, spr0);
                    }
#endif
                    return _c2c_oop_finish(hk, 0, W, cfg, N);
                }
            }
            vfft_il2p_destroy(il2p);
            vfft_il3p_destroy(il3p);
            vfft_ilprime_destroy(ilpr);
            vfft_ilfd_destroy(ilfd);
            vfft_ztt_destroy(ztt);
            vfft_k1fs_destroy(fs);   /* every engine this tier can build is released here */
            if (psp)
                vfft_oop_plan_destroy(psp);
            /* fall through to the classic OOP path */
        }
        } /* the K=1 tier's scope */
        /* PADDED (opt-in): build at Kp so the OOP plan strides the caller's Kp-wide 4 planes
         * exactly. Pad-only (OOP bakes K, no runtime me). Kp = the handle's roundup(K,8), which
         * keeps all 3 kinds AND lets the (N,Kp) OOP wisdom cell cache (BAILEY2 + the wisdom
         * reader both hard-gate on K%8). Pad lanes [K,Kp) are zeroed junk, discarded. */
        size_t bK = K;
        int padded = 0;
        if (ob)
        {
            vfft_batch b = ob;
            if (b->xform != (int)VFFT_C2C || !b->oop || b->N != N || b->K != K)
            {
                _vfft_warn("vfft_create: config.batch does not match this out-of-place C2C "
                           "descriptor (batch: %s%s N=%d K=%zu; config: C2C out-of-place "
                           "N=%d K=%zu) — INTERNAL INVARIANT (the plan allocates its own buffers); please report",
                           _vfft_tname(b->xform), b->oop ? " out-of-place" : " in-place",
                           b->N, b->K, N, K);
                return NULL;
            }
            bK = b->Kp;
            padded = 1;
        }
        if (cfg->layout == VFFT_LAYOUT_INTERLEAVED)
        {   /* No split champion behind a convert for an interleaved caller.
             * K=1 arrives here only when the IL route selection above found
             * no engine; K>1 is the explicit lane-major batch (DEFAULT
             * geometry is the transform-contiguous wrapper), which lost to
             * transform-contiguous at every measured cell. */
            _vfft_warn("vfft_create: out-of-place C2C N=%d howmany=%zu with "
                       "layout=INTERLEAVED has no interleaved engine (%s); nothing to "
                       "fall back to by design",
                       N, bK,
                       bK > 1 ? "lane-major batches are not an IL route - use "
                                "VFFT_BATCH_DEFAULT / TRANSFORM_CONTIGUOUS"
                              : "no mono/pair/chain3/prime kernel serves this N");
            return NULL;
        }
        vfft_oop_plan_t *op = NULL;
        int ord = cfg->order; /* 0=DEFAULT 1=NATURAL(LEAF/BAILEY2) 2=SCRAMBLED(MODEB) */
        /* Order-aware lookup: the cell can hold BOTH a natural and a MODEB champion as separate
         * (N,K,kind-class) entries, so the requested order is served straight from wisdom. */
        vfft_oop_wisdom_entry_t eb;
        const vfft_oop_wisdom_entry_t *e =
            W->vw2_off_oop ? vfft_oop_wisdom_lookup_ord(&W->oop, N, bK, ord)
                           : (vw2_oop_lookup_ord(&W->vw2, N, bK, ord, &eb) ? &eb : NULL);
        if (e && !cfg->recalibrate)
            op = vfft_oop_plan_from_entry(e, reg); /* the cached champion of the requested class */
        if (!op)
        {
            /* Calibrate-on-miss: build BOTH champions (native=LEAF/BAILEY2, MODEB), time each, and
             * persist BOTH as separate (N,K,kind-class) wisdom cells — so every config.order is cached
             * with no re-tune. Then return the requested order's champion (DEFAULT = the faster by ns).
             * Persisting both is exactly what makes MODEB and LEAF/BAILEY2 coexist per cell. */
            vfft_proto_dp_context_t ctx;
            vfft_proto_dp_init(&ctx, bK, N);
            if (cfg->rigor != VFFT_MEASURE)
                vfft_proto_dp_set_patient(&ctx);
            vfft_oop_plan_t *nat = NULL, *mb = NULL;
            double nns = 1e30, mns = 1e30;
            vfft_oop_plan_create_champions(N, bK, &ctx, reg, &nat, &nns, &mb, &mns);
            vfft_proto_dp_destroy(&ctx);
            /* Bank only servable cells: vfft_oop_plan_from_entry hard-gates
             * K%8, so a K%8!=0 champion row could never replay. It also
             * skips the K=1 MODEB champion, whose plan carries unraced
             * variant slots the wisdom2 codec would refuse. */
            if (bK > 0 && (bK % 8u) == 0)
            {
                if (nat)
                {
                    vfft_oop_wisdom_entry_t ne;
                    vfft_oop_wisdom_entry_from_plan(&ne, nat, N, bK, nns);
                    vw2_oop_bank_entry(&W->vw2, &ne);
                }
                if (mb)
                {
                    vfft_oop_wisdom_entry_t ne;
                    vfft_oop_wisdom_entry_from_plan(&ne, mb, N, bK, mns);
                    vw2_oop_bank_entry(&W->vw2, &ne);
                }
                if (nat || mb)
                    _vw2_persist(W, cfg);
            }
            if (ord == VFFT_ORDER_NATURAL)
            {
                op = nat;
                if (mb)
                    vfft_oop_plan_destroy(mb);
            }
            else if (ord == VFFT_ORDER_SCRAMBLED)
            {
                op = mb;
                if (nat)
                    vfft_oop_plan_destroy(nat);
            }
            else if (nat && mb)
            {
                if (nns <= mns)
                {
                    op = nat;
                    vfft_oop_plan_destroy(mb);
                }
                else
                {
                    op = mb;
                    vfft_oop_plan_destroy(nat);
                }
            }
            else
                op = nat ? nat : mb;
        }
        if (!op)
        {
            if (ord == VFFT_ORDER_NATURAL)
                _vfft_warn("vfft_create: no natural-order out-of-place C2C champion for "
                           "N=%d K=%zu (the natural kinds are gated on this cell) — use "
                           "order=DEFAULT/SCRAMBLED, or calibrate a natural champion into "
                           "the wisdom",
                           N, bK);
            else if (ob)
                _vfft_warn("vfft_create: no out-of-place C2C champion for the padded cell "
                           "N=%d Kp=%zu — drop config.batch or use in-place padding",
                           N, bK);
            else
                _vfft_warn("vfft_create: no out-of-place C2C engine covers N=%d K=%zu — "
                           "the OOP kinds need a radix factorization of N; prime and other "
                           "Rader/Bluestein-class sizes are served IN-PLACE only (create "
                           "with placement=VFFT_INPLACE)",
                           N, bK);
            return NULL;
        }
        struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
        if (!h)
        {
            vfft_oop_plan_destroy(op);
            return NULL;
        }
        h->transform = VFFT_C2C;
        h->placement = VFFT_OUTOFPLACE;
        h->layout = (int)cfg->layout;
        h->N = N;
        h->K = K;
        h->nthreads = _vfft_plan_threads(cfg);
        h->oplan = op;
        h->padded = padded;
        h->exec_me = (int)bK;
#ifdef VFFT_USE_JIT
        /* MODEB rides a staged inner plan -> JIT it (fwd: stages 1.. at start_stage=1;
         * bwd: whole in-place DIF at start_stage=0). LEAF/BAILEY2 have no staged plan. */
        if (op->kind == VFFT_OOP_KIND_MODEB && op->mb)
        {
            op->mb_jit_fwd = vfft_proto_plan_jit_fwd(op->mb);
            op->mb_jit_bwd = vfft_proto_plan_jit_bwd(op->mb);
        }
#endif
        return _c2c_oop_finish(h, 0, W, cfg, N);
    }
    return NULL; /* unreachable: the one call site guards on the same
                  * condition, and every path in the block above returns. */
}

#endif /* VFFT_OOP_C2C_OOP_CREATE_H */
