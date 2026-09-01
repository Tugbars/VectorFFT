/* c2c_oop_create.h — the c2c OUT-OF-PLACE create tier (migration step 25).
 *
 * WHAT THIS IS
 * ------------
 * The c2c out-of-place arm of _vfft_create_inner. It returns on every path, so
 * it lifts out behind its own guard, leaving the real and trig tiers that
 * follow untouched.
 *
 * OOP IS NOT IN-PLACE WITH A COPY BOLTED ON
 * -----------------------------------------
 * It is a separate serving with its own wisdom family (W->oop / vw2 oop
 * records) and its own K=1 route. That is why this tier is a sibling of
 * c2c_ip_create.h rather than a branch inside it: the two consult different
 * banked verdicts and build different plans.
 *
 * THE K=1 SPECIAL CASE
 * --------------------
 * `K == 1 && !ob` is where the K=1 machinery lives — the zsplit/zturn cascade
 * replay (_k1z_wisdom_replay) and, on a miss, the race that banks it
 * (_k1z_race_and_bank). Those two are the front door to the three-tier K=1
 * strategy (mono <=64, Bailey 128-1024, cascade >=2048); the tier itself does
 * not choose a tier, it replays or races for one.
 *
 * `ob` splits the same way it does in-place: a caller-supplied batch handle is
 * checked and served exactly, otherwise the plan owns its buffers.
 *
 * WISDOM, NOT HEURISTIC
 * ---------------------
 * Every open choice here is either replayed from a banked verdict or raced and
 * then banked. A banked line reads back as a verdict; nothing in this file may
 * grow a hand-written cutoff.
 *
 * POSITION IN vfft.c IS LOAD-BEARING
 * ----------------------------------
 * Not a standalone header. It calls file-scope statics that live in vfft.c, so
 * it must be included after those are defined and before _vfft_create_inner.
 *
 * The six parameters are the block's complete free-variable set, derived
 * rather than guessed: cfg, ob, W, reg, N, K.
 */
#ifndef VFFT_OOP_C2C_OOP_CREATE_H
#define VFFT_OOP_C2C_OOP_CREATE_H

/* the two arms of the odd-mid route race: one handle, zroute toggled
 * (the k1 arm runs with the cascade detached) */
typedef struct { struct vfft_plan_s *hk; vfft_zturn2_plan_t *zt; double *zi, *zo; } _ztodd_arm_t;
static void _ztodd_arm_cascade(void *v)
{
    _ztodd_arm_t *c = (_ztodd_arm_t *)v;
    c->hk->zroute = 1;
    c->hk->zturn = c->zt;
    vfft_execute((vfft_plan)c->hk, VFFT_FORWARD, c->zi, NULL, c->zo, NULL);
}
static void _ztodd_arm_k1(void *v)
{
    _ztodd_arm_t *c = (_ztodd_arm_t *)v;
    c->hk->zroute = 0;
    c->hk->zturn = NULL;
    vfft_execute((vfft_plan)c->hk, VFFT_FORWARD, c->zi, NULL, c->zo, NULL);
    c->hk->zturn = c->zt;
}
/* ── the tier's ONE exit. Every handle this create returns passes through
 * here; a shared post-step cannot be skipped by a new early exit without
 * the skip being spelled at the call. zt_mt says whether this exit races
 * the cascade MT verdict (INC-Z: K=1 zturn, live pool; serial default
 * everywhere the race does not run) — the K=1/odd-mid exit passes 0, its
 * historical behaviour. */
static vfft_plan _c2c_oop_finish(struct vfft_plan_s *h, int zt_mt)
{
    if (zt_mt && h->zroute && h->zturn && h->K == 1 && h->nthreads > 1)
        _zt_mt_race(h);
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
        /* ── K=1 engine (row_major_engine.md §13): natural-order routes from
         * kind-3 wisdom or the default heuristic; execute dispatches on the
         * COMMITTED layout axis (config.layout, stamped on the handle).
         * This IS the K=1 path (no kill-switch — user decision 2026-07-22:
         * K=1 is the headline feature; the classic champions path below was
         * never K=1-safe). Classic path still serves SCRAMBLED-order
         * requests and is the fallback if engine create fails. Construction
         * is layout-independent (both axes' routes are built as before). */
        vfft_zsplit_plan_t *zs_pending = NULL;
        vfft_zturn2_plan_t *zt_pending = NULL;
        int zroute_pending = 0; /* 0 = legacy zsplit, 1 = ZTURN-S */
        if (K == 1 && !ob && cfg->order == VFFT_ORDER_SCRAMBLED)
        {
            /* SCRAMBLED K=1: wisdom replay (>=2048 only, _k1z_wisdom_replay) else default chain + the stf/stf2 t2q race; the winning cascade attaches to the classic handle below.
             * t2q picks must be MEASURED on the installed binary — stf/stf2 are bit-identical, so the delta is code-placement order, never a hand-set constant.
             * See docs/design/vfft_front_door.md. */
            if (!_k1z_wisdom_replay(cfg, W, N, &zs_pending, &zt_pending,
                                    &zroute_pending))
            {
                /* MISS / recalibrate -> _k1z_race_and_bank (the single
                 * definition, shared with the IN-PLACE create). The HIT
                 * path above is the shared definition of replay
                 * semantics. */
                (void)_k1z_race_and_bank(cfg, W, N, /*ip=*/0,
                                         &zs_pending, &zt_pending,
                                         &zroute_pending); /* oop: banks kind-4 */
            }
        }
        /* K=1 engine admission (il_coverage_plan.md Phase A, 2026-08-03):
         * DEFAULT and NATURAL as always — and now explicit SCRAMBLED too,
         * WHEN no cascade plan attached above. The scrambled contract is
         * "any self-consistent permutation; a route's own bwd consumes its
         * own fwd comb" — the IDENTITY permutation qualifies, so the
         * natural-native K=1 engines serve an explicit-SCRAMBLED request
         * legally. Before this, asking for the CHEAPER contract below 2048
         * got the SLOWER route (convert fallback) while order=DEFAULT got
         * the native engine — a routing anomaly, nothing more. The
         * no-cascade guard keeps ≥2048 scrambled on the cascade dispatch
         * without building a dead-weight k1 engine beside it. */
        {
            /* an ODD-mid cascade candidate must not suppress the k1
             * admission: it has no fiat attach — it races the finished
             * handle at the commit, and the k1 routes ARE that
             * incumbent (without them the race timed the bare op
             * route, ~3x slower than the true serving — the strawman
             * caught 2026-08-27). */
            int ztodd = 0, s2o;
            if (zt_pending)
                for (s2o = 0; s2o < zt_pending->nf; s2o++)
                    if (zt_pending->chain[s2o] & 1)
                        ztodd = 1;
        if (K == 1 && !ob &&
            (cfg->order != VFFT_ORDER_SCRAMBLED || ztodd ||
             (!zs_pending && !zt_pending)))
        {
            int spr = VFFT_K1_SP_2PB, ilr = VFFT_K1_IL_2P;
            int sR1 = 0, sR2 = 0, iR1 = 0, iR2 = 0;
            vfft_oop_wisdom_entry_t keb;
            const vfft_oop_wisdom_entry_t *ke =
                W->vw2_off_oop ? vfft_oop_wisdom_lookup_k1(&W->oop, N)
                               : (vw2_oop_lookup_k1(&W->vw2, N, &keb) ? &keb : NULL);
            /* Per-layout wisdom (v1.2, 2026-08-24): each axis is taken
             * from the store INDEPENDENTLY. A cell with only an IL verdict
             * (k1_sp_route < 0 — e.g. non-pow2 N, where split cannot
             * factor) keeps the banked IL route while the split side runs
             * the same heuristic an unbanked cell always ran; neither
             * layout's absence degrades the other. */
            const int sp_banked = (ke && ke->k1_sp_route >= 0);
            /* il_banked mirrors sp_banked (review fix): k1_il_route = -1
             * means the IL axis was never raced at this cell — run the IL
             * heuristic, exactly as an unbanked cell would. IL_NONE (0) is
             * a VERDICT ("raced: no IL route available", the B2.1 meaning)
             * and is consumed as one. */
            const int il_banked = (ke && ke->k1_il_route >= 0);
            if (sp_banked)
            {
                spr = ke->k1_sp_route;
                sR1 = ke->R1;
                sR2 = ke->R2;
            }
            if (il_banked)
            {
                ilr = ke->k1_il_route;
                iR1 = ke->il_R1;
                iR2 = ke->il_R2;
            }
            if (!sp_banked)
            {
                /* heuristic default (uncalibrated cell): mono when emitted,
                 * else 2pb on the most balanced valid pair. The offline
                 * calibrator (benches/calibrate_k1.c, multi-run median)
                 * refines this into a kind-3 wisdom line per cell. */
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
            if (!il_banked)
            {
                /* IL runs its OWN pair search — it must NOT inherit sR1/sR2.
                 *
                 * Two independent reasons, both measured:
                 *  (a) COVERAGE. The loop above filters on SPLIT availability
                 *      (vfft_oop_leaf_fn / vfft_oop_t1_fn, which reach R=128),
                 *      but the il2p registries (vfft_il2p_leaf_fn /
                 *      vfft_il2p_mid_fn) stop at R=64. So the balanced split
                 *      pick can name radices IL has no kernel for — at
                 *      N=16384 it picks 128x128 and BOTH IL halves come back
                 *      NULL, while the route once claimed IL anyway (recorded
                 *      bug). Since a 2-pass IL route needs R1*R2 = N with both
                 *      <= 64, IL 2-pass genuinely tops out at N=4096; above
                 *      that the honest answer is IL_NONE, not a route that
                 *      cannot execute.
                 *  (b) INDEPENDENCE. Even where a split pair is legal for IL,
                 *      nothing guarantees it is the IL optimum -- the two arms
                 *      run different codelets over different layouts. The IL
                 *      planner (planning/dp_planner_il.h) searches this axis by
                 *      measurement; this loop only has to produce a LEGAL,
                 *      reasonable default for an uncalibrated cell.
                 *      (Note: a 2026-07-25 race showing 32x8 beating 4x64 at
                 *      N=256 was measured on the FUSED emit_k1 family, NOT this
                 *      staged 2P route -- do not cite it here. Measured on the
                 *      staged route, 4x64 wins at N=256, agreeing with split.)
                 *
                 * Calibrated cells are unaffected: calibrate_k1.c already picks
                 * an independent IL winner (win[2]) and writes its own iR1/iR2.
                 * This is only the uncalibrated default. */
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
                        /* NO parity constraint (2026-07-29): every monolithic
                         * cil kernel carries the inline VEX-128 odd-count
                         * tail, so odd factors are legal — all-odd pairs
                         * (45 = 9x5) and 2·odd pairs (50 = 5x10) route
                         * natively. The registry probes below are the only
                         * availability filter. (History: %4 was split's
                         * transpose contract; %2 was the pre-tail evenness
                         * contract.) */
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
                /* composed column (§12.4 item 5): chain from the wisdom line,
                 * else the per-R2 default. Create is self-validating (perm
                 * discovery); failure falls through to the classic path. */
                int ccf[VFFT_K1_CC_MAX_NF];
                int ccn = (ke && ke->cc_chain)
                              ? vfft_k1_cc_chain_decode(ke->cc_chain, ccf)
                              : vfft_k1_cc_default_chain(N / sR1, ccf);
                /* B4/B2.2 (2026-08-18): column-plan VARIANTS from the
                 * kind-3 line's own cc_vars token — the CCOL verdict is
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
             * CASCADE is record-only (see VFFT_K1_IL_CASCADE in oop_plan.h) --
             * a growing "!= this && != that" chain would silently start
             * building plans for any IL route added later.
             *
             * il2p is the ONLY pair-based IL machinery (the il_in/il_out
             * hybrids were deleted 2026-07-29), so the legacy wisdom aliases
             * (3P=1, 2P=2) and the canonical 2P_PURE all normalize to ONE
             * il2p attempt on the same (iR1,iR2) pair. Route stays TRUTHFUL:
             * it names 2P_PURE iff the plan exists, else NONE — execute never
             * dereferences a NULL k1il2p. Kill-switch: env VFFT_NO_IL2P
             * disables the whole pair-based IL axis (mono is unaffected). */
            if (ilr == VFFT_K1_IL_2P || ilr == VFFT_K1_IL_3P ||
                ilr == VFFT_K1_IL_2P_PURE)
            {
                if (iR1 && !getenv("VFFT_NO_IL2P"))
                {   /* braces are load-bearing: apply_kv must not run when the
                     * pair axis was skipped (it survived unbraced only because
                     * it null-checks — a latent trap, not a working shortcut) */
                    il2p = vfft_il2p_create(N, iR1, iR2);
                    _k1_il2p_apply_kv(il2p, ke, &W->vw2, N);   /* banked variant verdict */
                }
                ilr = il2p ? VFFT_K1_IL_2P_PURE : VFFT_K1_IL_NONE;
            }
            /* 3-STAGE CHAIN (route 6): the odd·2^k cells the pair search can
             * never serve (a 2-stage plan needs BOTH factors even — count
             * parity, il2p.h). Only attempted when the pair axis came up
             * empty and only for INTERLEAVED-committed plans (an IL-only
             * handle may carry spr == -1; the split dispatch must never see
             * one). Chain = LEGAL DEFAULT for the uncalibrated cell; the
             * measured per-cell pick is the wisdom campaign's job. */
            vfft_il3p_plan_t *il3p = NULL;
            if (ilr == VFFT_K1_IL_NONE && !il2p && !getenv("VFFT_NO_IL2P") &&
                cfg->layout == VFFT_LAYOUT_INTERLEAVED)
            {
                int cR2, cA, cB;
                if (vfft_il3p_default_chain(N, &cR2, &cA, &cB))
                    il3p = vfft_il3p_create(N, cR2, cA, cB);
                if (il3p)
                    ilr = VFFT_K1_IL_CHAIN3;
            }
            /* PRIME N (route 7): Rader/Bluestein on the IL machinery
             * (il_prime.h) — the OOP INTERLEAVED prime coverage the split
             * OOP path refuses. Same IL-only-handle rules as the chain. */
            vfft_ilprime_plan_t *ilpr = NULL;
            if (ilr == VFFT_K1_IL_NONE && !il2p && !il3p &&
                !getenv("VFFT_NO_IL2P") &&
                cfg->layout == VFFT_LAYOUT_INTERLEAVED)
            {
                ilpr = vfft_ilprime_create(N);
                if (ilpr)
                    ilr = VFFT_K1_IL_PRIME;
            }
            /* availability degrade (wisdom may name routes this build lacks).
             * Runs BEFORE spr0 is captured — P0c: spr0 keys the JIT (and the
             * TWL table pick), and keying it on the PRE-degrade route made
             * every create at N=8192 shell gcc for a 2PB bake whose
             * radix-128 UG_UL source does not exist (wisdom names 2PB 64x128;
             * leaf_ugul stops at 64 so execute degrades to 2PA, but the JIT
             * kept baking the route that could never compile — and with no
             * negative cache it retried per create). The L3-missing cases
             * degrade to their flat base here so spr0 never names an l3 twin
             * this build lacks; the fold below then only ever swaps pointers
             * that exist. */
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
            if (ilr == VFFT_K1_IL_MONO && !vfft_k1_mono_il_fn(N, 0))
                ilr = VFFT_K1_IL_NONE;
            /* Handle exists when the SPLIT axis has a route, OR when ANY
             * IL-only route does — pair, chain, or prime. 🔴 il2p MUST be in
             * this guard: with the odd-count tail, cells like 50 = 5x10 have
             * an IL pair but NO split K=1 route (spr == -1); omitting il2p
             * here silently dropped them to the classic path, whose DEFAULT-
             * order kind at such N is SCRAMBLED — natural-order callers got
             * a scrambled spectrum (caught by the public gate, 2026-07-29).
             * IL-only handles are INTERLEAVED-committed by construction
             * (every IL attempt above is layout-gated for the spr < 0 case),
             * so the split dispatch never sees k1_sp_route == -1. */
            if (spr >= 0 || (il2p && cfg->layout == VFFT_LAYOUT_INTERLEAVED) || il3p || ilpr)
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
                    hk->k1_mono = vfft_k1_mono_pair_fn(N, sR1);
                    hk->k1_mono_ilf = vfft_k1_mono_il_fn(N, 0);
                    hk->k1_mono_ilb = vfft_k1_mono_il_fn(N, 1);
#ifdef VFFT_USE_JIT
                    /* stride-baking JIT for the split route (§13.3): compile
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
                    /* OOP order=NATURAL at N >= _vfft_zcasc_min_n(): race this handle's real execute against a natord zturn cascade candidate; the winner attaches via hk->zturn on the existing zsplit||zturn-first dispatch.
                     * Both outcomes bank to @natoop, its own table — @nat stays the in-place single writer. Kill switch VFFT_NO_NAT_ZCASC; under it nothing is banked.
                     * See docs/design/vfft_front_door.md. */
                    if (cfg->order == VFFT_ORDER_NATURAL &&
                        N >= _vfft_zcasc_min_n() &&
                        cfg->layout == VFFT_LAYOUT_INTERLEAVED &&
                        !getenv("VFFT_NO_NAT_ZCASC"))
                    {
                        vfft_proto_nat_entry_t noeb;
                        const vfft_proto_nat_entry_t *noe =
                            W->vw2_off_stride ? vfft_proto_natoop_lookup(&W->c2c, N, K)
                                              : (vw2_stride_lookup_natoop(&W->vw2, _vw2_lay_of(cfg), N, K, &noeb) ? &noeb : NULL);
                        int nmode = (noe && !cfg->recalibrate)
                                        ? noe->mode : VFFT_NAT_UNSET;
                        vfft_zturn2_plan_t *zct = NULL;
                        if (nmode != VFFT_NAT_FREE)
                        {
                            vfft_config_t rcfg = *cfg;
                            rcfg.recalibrate = 0;
                            vfft_zsplit_plan_t *zcs = NULL;
                            int zcr = 0;
                            if (_k1z_wisdom_replay(&rcfg, W, N, &zcs,
                                                   &zct, &zcr))
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
                        if (nmode == VFFT_NAT_ZCASC)
                        {
                            if (zct)
                            {
                                hk->zturn = zct;
                                hk->zroute = 1;
                                zct = NULL;
                                if (getenv("VFFT_NAT_LOG"))
                                    fprintf(stderr, "[natorder] N=%d K=%zu "
                                            "replay ZCASC-OOP\n", N, K);
                            }
                            else /* banked chain vanished/refused: degrade to
                                  * re-measure — but with no candidate the
                                  * measure below is a no-op and the handle
                                  * serves as built (never hard-fail). */
                                nmode = VFFT_NAT_UNSET;
                        }
                        if (nmode == VFFT_NAT_UNSET && zct)
                        {
                            /* MEASURE: this handle's real OOP execute vs the
                             * natord cascade, src->dst distinct (src is
                             * read-only in OOP fwd — no reseed hazard).
                             * 5 rounds, alternated order, medians (B5). */
                            double *rz = (double *)malloc(
                                2 * (size_t)N * sizeof(double));
                            double *r0 = (double *)malloc(
                                2 * (size_t)N * sizeof(double));
                            if (rz && r0)
                            {
                                for (long i = 0; i < 2L * N; i++)
                                    r0[i] = (double)rand() / RAND_MAX - 0.5;
                                const int reps =
                                    N <= 4096 ? 24 : (N <= 16384 ? 10 : 6);
                                double ns[2]; /* [0] incumbent, [1] zcasc */
                                _c2c_race_ctx_t rc = { hk, 1, zct, NULL, 1, NULL, NULL, NULL, rz, r0,
                                                        2 * (size_t)N * sizeof(double) };
                                const vfft_race_arm_t arms[2] = {
                                    { "incumbent", _c2c_race_inc, &rc }, { "zcasc", _c2c_race_chal, &rc } };
                                /* 5 rounds, odd rounds reversed, median-of-5; no reseed: src is
                                 * read-only in OOP fwd (r0 -> rz) */
                                const vfft_race_proto_t proto = { 5, reps, VFFT_RACE_MEDIAN, 1, 0, NULL, NULL };
                                /* GROW-ONLY, like the five sibling copies of this
                                 * race body (c2c_ip_create.h). The public setter
                                 * SHRINKS: with the house spelling nthreads=1 for
                                 * "this child is serial" it tore an 8-worker pool
                                 * down to 1 for the whole process on every IL-2D
                                 * OOP row child. pool_preserve_gate asserts this. */
                                _vfft_pool_arm(hk->nthreads);
                                vfft_race_run(&proto, arms, 2, ns);
                                if (ns[1] < ns[0])
                                {
                                    hk->zturn = zct;
                                    hk->zroute = 1;
                                    zct = NULL;
                                    _bank_natoop_1d(W, cfg, N, K, VFFT_NAT_ZCASC,
                                                    ns[1]);
                                }
                                else
                                    _bank_natoop_1d(W, cfg, N, K, VFFT_NAT_FREE,
                                                    ns[0]);
                                if (getenv("VFFT_NAT_LOG"))
                                    fprintf(stderr,
                                            "[natorder] N=%d K=%zu OOP "
                                            "zcasc=%.0fns engine=%.0fns -> "
                                            "%s\n", N, K, ns[1], ns[0],
                                            hk->zturn ? "ZCASC-OOP"
                                                      : "engine");
                            }
                            free(rz);
                            free(r0);
                        }
                        if (zct)
                            vfft_zturn2_destroy(zct);
                    }
                    /* ── ODD-MID cascade, SCRAMBLED/DEFAULT (2026-08-27):
                     * zt_pending (built + t2q'd by the race helper) has
                     * NO fiat attach — it races THIS handle's real
                     * serving (the k1 IL routes included; racing the
                     * bare op route was the strawman caught today) and
                     * attaches only by winning. min-of-3 alternated on
                     * scratch; the loser dies here. The pow2 flow is
                     * untouched (its admission gate keeps it on the
                     * fiat commit, backed by calibration history). */
                    if (zt_pending && cfg->order != VFFT_ORDER_NATURAL)
                    {
                        int s2o, zodd2 = 0;
                        for (s2o = 0; s2o < zt_pending->nf; s2o++)
                            if (zt_pending->chain[s2o] & 1)
                                zodd2 = 1;
                        if (zodd2)
                        {
                            double *zi2 = (double *)malloc(
                                2 * (size_t)N * sizeof(double));
                            double *zo2b = (double *)malloc(
                                2 * (size_t)N * sizeof(double));
                            double tzc = 1e300, tkc = 1e300;
                            if (zi2 && zo2b)
                            {
                                size_t i2;
                                int r2;
                                for (i2 = 0; i2 < 2 * (size_t)N; i2++)
                                    zi2[i2] = 1.0 +
                                              1e-6 * (double)(i2 & 511);
                                vfft_execute((vfft_plan)hk,
                                             VFFT_FORWARD, zi2, NULL,
                                             zo2b, NULL); /* warm k1 */
                                hk->zturn = zt_pending;
                                hk->zroute = 1;
                                vfft_execute((vfft_plan)hk,
                                             VFFT_FORWARD, zi2, NULL,
                                             zo2b, NULL); /* warm zt */
                                {
                                    _ztodd_arm_t c = { hk, zt_pending, zi2, zo2b };
                                    const vfft_race_arm_t arms[2] = {
                                        { "cascade", _ztodd_arm_cascade, &c },
                                        { "k1", _ztodd_arm_k1, &c } };
                                    const vfft_race_proto_t proto = { 3, 1, VFFT_RACE_MIN, 0, 0, NULL, NULL }; /* min-of-3, A then B */
                                    double ns[2];
                                    (void)r2;
                                    vfft_race_run(&proto, arms, 2, ns);
                                    tzc = ns[0];
                                    tkc = ns[1];
                                }
                                if (getenv("VFFT_ZT_LOG") ||
                                    getenv("VFFT_NAT_LOG"))
                                    fprintf(stderr,
                                            "[zt-odd] route race N=%d: "
                                            "cascade=%.0f k1=%.0f -> "
                                            "%s\n",
                                            N, tzc, tkc,
                                            tzc < tkc ? "CASCADE"
                                                      : "k1");
                                if (tzc < tkc)
                                {
                                    hk->zroute = 1;
                                    zt_pending = NULL; /* owned by hk */
                                }
                                else
                                {
                                    hk->zroute = 0;
                                    hk->zturn = NULL;
                                }
                            }
                            free(zi2);
                            free(zo2b);
                        }
                    }
                    if (zt_pending)
                    { /* not attached (lost, pow2-stray, or NATURAL):
                       * consume — the hk return path used to LEAK it. */
                        vfft_zturn2_destroy(zt_pending);
                        zt_pending = NULL;
                    }
                    if (zs_pending)
                    {
                        vfft_zsplit_destroy(zs_pending);
                        zs_pending = NULL;
                    }
                    /* zt_mt=0: an odd-mid cascade attached here has NEVER
                     * had an MT verdict raced (the historical skip). Racing
                     * it is a new feature — threaded odd-mid cascades —
                     * priced separately, not flipped on in a refactor. */
                    return _c2c_oop_finish(hk, /*zt_mt=*/0);
                }
            }
            vfft_il2p_destroy(il2p);
            vfft_il3p_destroy(il3p);
            vfft_ilprime_destroy(ilpr);
            if (psp)
                vfft_oop_plan_destroy(psp);
            /* fall through to the classic OOP path */
        }
        } /* ztodd scope (the odd-cascade admission wrapper) */
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
             * K%8, so a K%8!=0 champion row could never replay — legacy
             * banked those anyway (the write-only "wart" lines, quarantined
             * as garbage at migration) and this guard is their sunset. It
             * also skips the K=1 MODEB champion, whose plan carries
             * unraced variant slots the wisdom2 codec would refuse. */
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
            vfft_zsplit_destroy(zs_pending);
            vfft_zturn2_destroy(zt_pending);
            return NULL;
        }
        struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
        if (!h)
        {
            vfft_zsplit_destroy(zs_pending);
            vfft_zturn2_destroy(zt_pending);
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
        h->zsplit = zs_pending; /* exactly one of zsplit/zturn is non-NULL */
        h->zturn = zt_pending;
        h->zroute = zroute_pending;
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
        return _c2c_oop_finish(h, /*zt_mt=*/1);
    }
    return NULL; /* unreachable: the one call site guards on the same
                  * condition, and every path in the block above returns. */
}

#endif /* VFFT_OOP_C2C_OOP_CREATE_H */
