/* k1_commit.h - the K=1 plan's replay, race, and commit.
 *
 * How a single-transform cell gets its plan: consult wisdom, and on a miss race
 * the candidates and bank the winner. Extracted from vfft.c as migration
 * step 19; see docs/design/refactor_migration_plan.md.
 *
 * THE PRECEDENCE LADDER, WHICH IS THE WHOLE POINT
 * ----------------------------------------------
 * Every decision in this file follows the same order, and the order is the
 * design:
 *
 *   1. an ENV pin      - beats everything, and NEVER banks. An override is an
 *                        experiment; if experiments wrote to the store, one
 *                        debugging session would poison every later run. That
 *                        is the tcut law.
 *   2. a BANKED verdict - honoured at every rigor tier. Banking it is the point
 *                        of having measured it once.
 *   3. RACE and BANK    - on a miss, build the candidates as finished plans,
 *                        time them, keep the winner, write it down.
 *   4. a STRUCTURAL default - only when nothing above applies.
 *
 * REPLAY AND RACE ARE SEPARATE FUNCTIONS ON PURPOSE
 * -------------------------------------------------
 * _k1z_wisdom_replay reconstructs a plan from a banked record and MEASURES
 * NOTHING. _k1z_race_and_bank measures. Keeping them apart is what makes a
 * warmed store cheap - a create on a hit never touches a clock - and it is also
 * what makes the harness's replay-purity assertion meaningful: a cell that
 * races during what should be a replay has the clock inside its own baseline.
 *
 * _k1_il2p_apply_kv IS NOT A RACE
 * -------------------------------
 * It applies a banked kind-3 il_kv verdict - the leaf/mid kernel FORM pair -
 * and measures nothing. The race that produced that verdict lives offline in
 * dp_planner_il.h and never runs at create. An il_kv of 0 keeps create's own
 * structural default (blocked at R>=32).
 *
 * Worth knowing: il_kv has NO fingerprint field, so two plans differing only in
 * kernel form fingerprint identically and the migration harness cannot see a
 * change here. This is one of the few places in the tree where obj_equiv is the
 * only guard.
 *
 * THREE BANKERS, THREE DIFFERENT CELLS
 * ------------------------------------
 * The order axis does not share a cell with the scrambled one:
 *   _bank_nat_1d      in-place natural   (ord=nat)
 *   _bank_natoop_1d   out-of-place natural
 *   _bank_scrmode_1d  the ord=scr mode cell (mode=ilp|zcasc|conv)
 * They are separate so that a natural-order create can never perturb the
 * scrambled plan, and vice versa - the regimes are calibrated independently
 * because they are genuinely different engines, not one engine with a flag.
 *
 * INCLUSION CONTRACT
 * ------------------
 * Include after the engine prelude, after vfft_internal.h, and AFTER
 * _vw2_persist - the bankers call it, and it stays in vfft.c. Same back-edge as
 * zr2c_build.h has; both would be freed by moving _vw2_persist to support/.
 */
#ifndef VFFT_OOP_K1_COMMIT_H
#define VFFT_OOP_K1_COMMIT_H

#include <stdlib.h>
#include <string.h>

#include "vfft_internal.h"                  /* struct vfft_plan_s / vfft_wisdom_s */
#include "il2p.h"                           /* the Bailey pair plan + kernel resolvers */
#include "il_prime.h"                       /* the prime IL engine */
#include "zsplit.h"                         /* the legacy cascade route */
#include "zturn.h"                          /* the ZTURN cascade route */
#include "planning/cascade_calibrate.h"     /* the t2q terminator calibrators */
#include "wisdom2/wisdom2_oop_reader.h"     /* the kind-3/kind-4 codecs */
#include "wisdom2/wisdom2_stride_reader.h"  /* the @nat / @natoop / mode cells */
#include "support/race.h"                   /* the shared race body */

/* Applies a banked kind-3 il_kv verdict; measures nothing (dp_planner_il.h
 * owns that race). il_kv==0 keeps create's default — blocked at R>=32. */
static void _k1_il2p_apply_kv(vfft_il2p_plan_t *p,
                              const vfft_oop_wisdom_entry_t *ke,
                              const vw2_store_t *st, int N)
{
    /* Wisdom variant verdict — runs AFTER create, so it OVERRIDES the
     * structural blocked default (il2p.h): a banked per-cell measurement
     * always outranks the structural rule. Nibble VFFT_IL_KV_MONO (0xF)
     * forces the monolithic kernel back — required since blocked became
     * the R>=32 default, so a platform where blocked measures slower
     * stays expressible as a verdict rather than only as an env. */
    if (!p)
        return;
    if (ke)
        vfft_il2p_apply_kv_forms(p, ke->il_kv); /* shared nibble semantics —
                                                 * one definition (il2p.h),
                                                 * planner uses the same fn */
    /* BACKWARD arm (2026-08-21). The backward kernel-variant verdict is its
     * OWN CELL, keyed `dir=bwd`, rather than more il_kv bits: wisdom2 keys
     * DIRECTION and does not key kernel forms. Deliberately outside the `ke`
     * guard - the backward pick does not depend on a forward wisdom hit. No
     * record => no-op, and il2p.h's apply_blocked_default_bwd structural
     * pick stands.
     *
     * The two directions genuinely disagree, which is why this is a separate
     * verdict and not a shared one: at N=1024 the raced forward and backward
     * winners for the same 32.32 plan are different variant codes. */
    if (st)
    {
        /* 🔴 PAIR CHECK, not just a lookup. A variant code names kernels
         * for ONE radix pair; the forward winner can move (a re-race, a
         * different machine, a hand-edited line) without this record being
         * re-raced, and applying a 32x32 verdict to a 64x16 plan would
         * install kernels whose counts do not match the plan's slots.
         * Mismatch => ignore the record and keep the structural default,
         * which is always correct if slower. */
        int bR1 = 0, bR2 = 0;
        int bkv = vw2_oop_lookup_k1_bwd(st, N, &bR1, &bR2);
        if (bkv && bR1 == p->R1 && bR2 == p->R2)
            vfft_il2p_apply_kv_forms_bwd(p, bkv);
    }
    /* Env applied LAST — it beats the banked verdict (racing hook). Packed
     * nibbles: VFFT_IL_KV=0x25 => mid 5, leaf 2 (VFFT_IL_KV_PACK, il2p.h).
     * See docs/design/vfft_front_door.md. */
    {
        const char *e = getenv("VFFT_IL_KV");
        if (e && e[0])
            vfft_il2p_apply_kv_forms(p, (int)strtol(e, NULL, 0));
    }
    {
        const char *e = getenv("VFFT_IL_BKV");
        if (e && e[0])
            vfft_il2p_apply_kv_forms_bwd(p, (int)strtol(e, NULL, 0));
    }
}

/* ── K=1 IL-engine candidate for the IN-PLACE tiers (il_coverage_plan.md
 * Phase B). Resolves N to exactly one of il2p/il3p (or neither): kind-3
 * pair when banked, else the balanced-pair heuristic, else the il3p chain
 * default. MONO is deliberately absent — its kernels are `__restrict__`
 * and refuse aliasing (A3 record). PRIME cells return neither (the
 * incumbent keeps serving them; il_prime aliasing is ungated).
 * ⚠ The pair heuristic MIRRORS the OOP K=1 block's IL search (the
 * "IL runs its OWN pair search" rules: il2p registries stop at R=64, no
 * parity constraint since the odd-count tail) — if you touch one, touch
 * both; they are cross-referenced. Planning side only. */
/* the two arms of the (R1,R2) ordering race: two il2p plans on one
 * aliased buffer, re-seeded before every burst */
typedef struct { vfft_il2p_plan_t *p; double *rz, *r0; size_t nb; } _k1ord_arm_t;
static void _k1ord_arm_run(void *v)
{
    _k1ord_arm_t *c = (_k1ord_arm_t *)v;
    vfft_il2p_execute_fwd(c->p, c->rz, c->rz);
}
static void _k1ord_reseed(void *v)
{
    _k1ord_arm_t *c = (_k1ord_arm_t *)v;
    memcpy(c->rz, c->r0, c->nb);
}
static void _k1_il_candidate(struct vfft_wisdom_s *W, int N,
                             vfft_il2p_plan_t **il2p_out,
                             vfft_il3p_plan_t **il3p_out)
{
    *il2p_out = NULL;
    *il3p_out = NULL;
    if (getenv("VFFT_NO_IL2P"))
        return;
    int iR1 = 0, iR2 = 0;
    vfft_oop_wisdom_entry_t keb;
    const vfft_oop_wisdom_entry_t *ke =
        W->vw2_off_oop ? vfft_oop_wisdom_lookup_k1(&W->oop, N)
                       : (vw2_oop_lookup_k1(&W->vw2, N, &keb) ? &keb : NULL);
    if (ke && ke->il_R1)
    {
        iR1 = ke->il_R1;
        iR2 = ke->il_R2;
    }
    else
    {
        for (int R2c = (N < 64 ? N : 64); R2c >= 4; R2c--)
        {
            if (N % R2c)
                continue;
            int R1c = N / R2c;
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
    }
    if (iR1)
    {   /* braces load-bearing (same latent trap fixed at the OOP site):
         * apply_kv must not run when the pair axis was skipped. */
        *il2p_out = vfft_il2p_create(N, iR1, iR2);
        _k1_il2p_apply_kv(*il2p_out, ke, &W->vw2, N);   /* wisdom verdict > default */
    }
    /* Ordering is a measured axis: (R1,R2) and (R2,R1) install different mid
     * kernels. Heuristic pairs only — a wisdom pair is the calibrator's.
     * See docs/design/vfft_front_door.md. */
    /* Per-process MEMO of the ordering pick, keyed by N: the race must
     * run at most ONCE per process per cell — without this, the natural
     * and scrambled handles (and measure vs consume) each re-race, and a
     * margin near the hysteresis flips on noise, breaking the
     * bitwise-identity contracts between them (caught by
     * vfft_ilp_front_gate's scrambled arm at 512, margin 4.5% vs 3%
     * hysteresis). Planning-side, no locks: worst case a benign double
     * race on concurrent first creates. */
    static int _ord_n[8];
    static signed char _ord_pick[8]; /* 0 = heuristic order, 1 = swapped */
    int ord_slot = -1, ord_known = -1;
    for (int ci = 0; ci < 8; ci++)
    {
        if (_ord_n[ci] == N) { ord_slot = ci; ord_known = _ord_pick[ci]; }
        else if (_ord_n[ci] == 0 && ord_slot < 0) ord_slot = ci;
    }
    if (ord_known == 1 && *il2p_out && !(ke && ke->il_R1) && iR1 != iR2)
    {
        vfft_il2p_plan_t *sw = vfft_il2p_create(N, iR2, iR1);
        if (sw)
        {
            vfft_il2p_destroy(*il2p_out);
            *il2p_out = sw;
        }
    }
    if (ord_known < 0 && *il2p_out && !(ke && ke->il_R1) && iR1 != iR2 &&
        !getenv("VFFT_NO_T2B"))
    {
        vfft_il2p_plan_t *alt = vfft_il2p_create(N, iR2, iR1);
        int picked_swap = 0;
        if (alt)
        {
            double *rz = (double *)malloc(2 * (size_t)N * sizeof(double));
            double *r0 = (double *)malloc(2 * (size_t)N * sizeof(double));
            if (rz && r0)
            {
                for (long i = 0; i < 2L * N; i++)
                    r0[i] = (double)(i % 17) * 0.0625 - 0.5;
                const int reps = N <= 256 ? 64 : (N <= 1024 ? 24 : 8);
                const size_t nb = 2 * (size_t)N * sizeof(double);
                double ta, tb;
                {
                    _k1ord_arm_t ca = { *il2p_out, rz, r0, nb };
                    _k1ord_arm_t cb = { alt, rz, r0, nb };
                    const vfft_race_arm_t arms[2] = {
                        { "heuristic", _k1ord_arm_run, &ca },
                        { "swapped", _k1ord_arm_run, &cb } };
                    /* 5 rounds, A then B, min; reseed before every burst:
                     * repeated in-place fwd amplifies magnitudes toward inf
                     * (the ZCASC-race hazard) */
                    const vfft_race_proto_t proto = { 5, reps, VFFT_RACE_MIN, 0, 0,
                                                      _k1ord_reseed, &ca };
                    double ns[2];
                    vfft_race_run(&proto, arms, 2, ns);
                    ta = ns[0];
                    tb = ns[1];
                }
                /* 3% hysteresis, incumbent (heuristic) keeps ties —
                 * the t2q/t2b precedent exactly. */
                if (vfft_race_beats(tb, ta, 0.97))
                {
                    vfft_il2p_destroy(*il2p_out);
                    *il2p_out = alt;
                    alt = NULL;
                    picked_swap = 1;
                }
            }
            free(rz);
            free(r0);
            if (alt)
                vfft_il2p_destroy(alt);
        }
        /* record the pick (even when the race could not run — alt-create
         * failure defaults to heuristic) so every later create in this
         * process agrees. */
        if (ord_slot >= 0)
        {
            _ord_n[ord_slot] = N;
            _ord_pick[ord_slot] = (signed char)picked_swap;
        }
    }
    if (!*il2p_out)
    {
        int cR2, cA, cB;
        if (vfft_il3p_default_chain(N, &cR2, &cA, &cB))
            *il3p_out = vfft_il3p_create(N, cR2, cA, cB);
    }
}

/* the mode-row RECIPE rule (owner, 2026-09-02): fac/var are the classic
 * plan of the CALLER (the convert incumbent) — the served recipe only for
 * mode=conv and the tape modes. A mode=zcasc row must not carry them: the
 * writer emits a signpost to the kind-4 recipe instead (comp when the
 * in-place race banked one, else the OOP verdict); mode=ilp emits neither. */
static int _zcasc_ref_is_comp(struct vfft_wisdom_s *W, int N, int mode)
{
    vfft_oop_wisdom_entry_t tmp;
    /* the row that SERVES (mirrors _k1z_wisdom_replay's order): the searched
     * verdict when a cascade engine holds it, else the comp recipe */
    return mode == VFFT_NAT_ZCASC && !W->vw2_off_oop &&
           !vw2_oop_lookup_zsplit(&W->vw2, N, &tmp) &&
           vw2_oop_lookup_zsplit_role(&W->vw2, N, VW2_ROLE_COMP, &tmp);
}

/* ── C1.9 zt_mt: the cascade MT verdict, banked PER THREAD COUNT ──────────
 * (arm audit 2026-09-02: it was raced on every OOP create and never
 * persisted). The verdict rides the recipe row that served the cascade —
 * the searched verdict, else the comp recipe — as zt_mt_t=<T> zt_mt=<0|1>;
 * a T match replays, a mismatch re-races and re-banks (validity-condition
 * banking, measurement_arms 'cores sharing one transform'). A re-raced
 * recipe row is rebuilt fresh, so it drops the MT verdict with the recipe.
 * VFFT_ZT_NO_MT (env) beats wisdom: the race helper applies it, so an env
 * pin never replays and never banks (the tcut law). */
static int _zt_mt_served_key(struct vfft_wisdom_s *W, int N, vw2_key_t *k)
{
    vfft_oop_wisdom_entry_t tmp;
    if (!W || W->vw2_off_oop) return 0;
    if (vw2_oop_lookup_zsplit(&W->vw2, N, &tmp)) { vw2_oop_zsplit_key(N, VW2_ROLE_NONE, k); return 1; }
    if (vw2_oop_lookup_zsplit_role(&W->vw2, N, VW2_ROLE_COMP, &tmp)) { vw2_oop_zsplit_key(N, VW2_ROLE_COMP, k); return 1; }
    return 0;
}

static void _zt_mt_replay_or_race(struct vfft_plan_s *h,
                                  struct vfft_wisdom_s *W,
                                  const vfft_config_t *cfg, int N)
{
    vw2_key_t k;
    const int T = h->nthreads;
    const int ip = (h->placement == VFFT_INPLACE);
    /* one pair per PLACEMENT: the in-place arms are aliased z->z, the OOP
     * arms z->z', so the two verdicts are different measurements and must
     * never overwrite each other (2026-09-02, the in-place exit joined). */
    const char *tok_t = ip ? "zt_mt_ip_t" : "zt_mt_t";
    const char *tok_v = ip ? "zt_mt_ip"   : "zt_mt";
    const vw2_rec_t *r = NULL;
    if (!getenv("VFFT_ZT_NO_MT") && !cfg->recalibrate &&
        _zt_mt_served_key(W, N, &k) && (r = vw2_lookup(&W->vw2, &k)) != NULL)
    {
        const int bt = vw2__oop_geti(r, tok_t, 0);
        if (bt == T)
        {
            h->zt_mt = vw2__oop_geti(r, tok_v, 0) ? 1 : 0;
            if (getenv("VFFT_ZT_LOG") || getenv("VFFT_IL2D_LOG"))
                fprintf(stderr, "[zt-mt] N=%d T=%d %s replay zt_mt=%d src=wisdom\n",
                        N, T, ip ? "ip" : "oop", h->zt_mt);
            return;
        }
    }
    _zt_mt_race(h);
    if (!getenv("VFFT_ZT_NO_MT") && _zt_mt_served_key(W, N, &k))
    {
        char tb[16];
        snprintf(tb, sizeof tb, "%d", T);
        if (vw2_update_field(&W->vw2, &k, tok_t, tb) == VW2_OK &&
            vw2_update_field(&W->vw2, &k, tok_v, h->zt_mt ? "1" : "0") == VW2_OK)
            _vw2_persist(W, cfg);
    }
}

static void _bank_nat_1d(struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                         int N, size_t K, int mode, double ns,
                         const int *fac, const int *var, int nf, int use_dif)
{
    vfft_proto_nat_entry_t nn;
    memset(&nn, 0, sizeof nn);
    nn.N = N;
    nn.K = K;
    nn.mode = mode;
    nn.nat_ns = ns;
    nn.nf = nf;
    nn.use_dif = use_dif;
    nn.ref_comp = _zcasc_ref_is_comp(W, N, mode);
    for (int s = 0; s < nf && s < STRIDE_MAX_STAGES; s++)
    {
        nn.factors[s] = fac[s];
        nn.variants[s] = var[s];
    }
    /* wave-4 flip: @nat verdicts bank into the wisdom2 store (memory;
     * persistence behind config.wisdom_write). spike_wisdom.txt freezes. */
    vw2_stride_bank_nat(&W->vw2, &nn, /*is_oop=*/0, _vw2_lay_of(cfg));
    _vw2_persist(W, cfg);
}

/* The banked LOSS (2026-09-02): the ZCASC/ILP challenger raced this @nat
 * cell and the tape won. Mark the EXISTING record with zr=1 in place
 * (vw2_update_field) — the tape's mode/chain line stays byte-for-byte as
 * the tape race banked it (re-banking here would re-encode a chain whose
 * provenance differs by mode — the SCR/dfac subtlety). A later re-bank of
 * the cell (recalibrate) drops the token: race once, mark again. Without
 * this, a losing race re-ran on EVERY create, forever — the exact disease
 * VFFT_NAT_CONV was minted to cure on the ord=scr cell. */
static void _bank_nat_raced(struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                            int N, size_t K)
{
    vw2_key_t k;
    if (W->vw2_off_stride)
        return; /* kill switch: legacy tables keep the old re-race behaviour */
    vw2__stride_key(&k, VW2_T_C2C, N, K, VW2_ORD_NAT, VW2_PL_IP);
    k.lay = _vw2_lay_of(cfg);
    if (vw2_update_field(&W->vw2, &k, "zr", "1") == VW2_OK)
        _vw2_persist(W, cfg);
}

/* OOP-natural verdict: same (N,K) cell as @nat but keyed place=oop, so the
 * placements cannot clobber each other — and keyed lay= (v1.2) so the
 * LAYOUTS cannot either, for exactly the same reason the placement split
 * existed. nf=1/factors[0]=N => ref= signpost.
 * See docs/design/vfft_front_door.md. */
static void _bank_natoop_1d(struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                            int N, size_t K, int mode, double ns)
{
    vfft_proto_nat_entry_t nn;
    memset(&nn, 0, sizeof nn);
    nn.N = N;
    nn.K = K;
    nn.mode = mode;
    nn.nat_ns = ns;
    nn.nf = 1;
    nn.factors[0] = N;
    /* wave-4 flip: the dummy-chain shape becomes the ref= SIGNPOST record
     * in the store (the family codec detects nf==1 && factors[0]==N). */
    vw2_stride_bank_nat(&W->vw2, &nn, /*is_oop=*/1, _vw2_lay_of(cfg));
    _vw2_persist(W, cfg);
}

/* the ord=scr mode-cell bank (the ILP-attach fix, 2026-08-25): the
 * scrambled in-place IL race's verdict — mode=ILP | ZCASC (win) or
 * mode=CONV (the banked loss, so a losing race never re-runs). The chain
 * is the caller's classic plan: served recipe for CONV only — see the
 * mode-row RECIPE rule above _bank_nat_1d. */
static void _bank_scrmode_1d(struct vfft_wisdom_s *W,
                             const vfft_config_t *cfg, int N, size_t K,
                             int mode, double ns, const int *fac,
                             const int *var, int nf, int use_dif)
{
    vfft_proto_nat_entry_t nn;
    memset(&nn, 0, sizeof nn);
    nn.N = N;
    nn.K = K;
    nn.mode = mode;
    nn.nat_ns = ns;
    nn.nf = nf;
    nn.use_dif = use_dif;
    nn.ref_comp = _zcasc_ref_is_comp(W, N, mode);
    for (int s = 0; s < nf && s < STRIDE_MAX_STAGES; s++)
    {
        nn.factors[s] = fac[s];
        nn.variants[s] = var[s];
    }
    vw2_stride_bank_scrmode(&W->vw2, &nn, _vw2_lay_of(cfg));
    _vw2_persist(W, cfg);
}

/* ════════════════════════════════════════════════════════════════════════
 * PUBLIC API
 * ════════════════════════════════════════════════════════════════════════ */

/* ── K=1 SCRAMBLED cascade: WISDOM-HIT replay — THE one definition ──────────
 *
 * Resolves the banked kind-4 verdict (route + chain + t2q + tcut width with
 * its L1 fence and the env-beats-wisdom rule) into exactly one live cascade
 * plan. Shared by the OOP create branch AND the in-place front door — the
 * calibrate_zchain incident (two writers, one taught about a new field, a
 * tiled winner banked as untiled with nothing complaining) is why replay
 * semantics live in ONE place. Returns 1 with outputs set on a full hit;
 * 0 (outputs untouched) on miss/recalibrate/create-failure — the caller
 * decides what a miss means (OOP: race + bank; in-place: classic path).
 * PLANNING side only; the exec purity audit watches this. */
static int _k1z_wisdom_replay(const vfft_config_t *cfg,
                              struct vfft_wisdom_s *W, int N,
                              vfft_zsplit_plan_t **zs_out,
                              vfft_zturn2_plan_t **zt_out, int *zroute_out)
{
    /* The cascade is the ≥2048 tier, period. A kind-4 row BELOW that is a
     * wrong-slot verdict (the sub-2048 SCRAMBLED champion is the identity
     * ILP engine — Phase A doctrine, k1scr-gated) and replaying it would
     * flip explicit-SCRAMBLED cells onto a cascade comb, silently breaking
     * the scr==nat identity contract while every correctness column stays
     * green — exactly how it was caught (2026-08-06): calibrate_k1's
     * plan_and_bank side-banked sub-2048 kind-4 rows and the k1scr gate
     * went DIFF at 128..1024 with the cascade 2.2× SLOWER than the engine
     * it displaced. The driver no longer banks them; this guard makes any
     * such row in a user's wisdom file inert as well. */
    if (N < 2048)
        return 0;
    vfft_zsplit_plan_t *zs_pending = NULL;
    vfft_zturn2_plan_t *zt_pending = NULL;
    int zroute_pending = 0;
    int zch[VFFT_ZSPLIT_MAX_NF];
    int znf = 0;
    vfft_oop_wisdom_entry_t zeb;
    /* RECIPE source (owner, 2026-09-02): an in-place caller replays the
     * role=comp kind-4 row its own race banked (chain + t2q terminator
     * pick + tcut width + L1 fence, raced IN PLACE), falling back to the
     * OOP problem verdict. An OOP caller reads its verdict first; with no
     * verdict it may replay a comp recipe ONLY for an ODD chain — the odd
     * candidate never attaches by fiat (it races the finished handle at
     * the commit, the hk block), so the comp row just spares the per-
     * create t2q re-race. A pow2 comp row came from an in-place race and
     * must not decide the OOP route. The mode row's ref= names whichever
     * served. */
    const int ip_call = (cfg->placement == VFFT_INPLACE);
    const vfft_oop_wisdom_entry_t *ze = NULL;
    if (W->vw2_off_oop)
        ze = vfft_oop_wisdom_lookup_zsplit(&W->oop, N);
    else if (vw2_oop_lookup_zsplit(&W->vw2, N, &zeb))
        ze = &zeb;                 /* the SEARCHED verdict (planner / OOP race) */
    else if (ip_call &&
             vw2_oop_lookup_zsplit_role(&W->vw2, N, VW2_ROLE_COMP, &zeb))
        ze = &zeb;                 /* in-place: the comp recipe (default chain,
                                    * raced t2q) when nothing searched exists */
    else if (vw2_oop_lookup_zsplit_role(&W->vw2, N, VW2_ROLE_COMP, &zeb))
    {
        int cch[VFFT_K1_CC_MAX_NF], cnf = 0, ci, codd = 0;
        if (zeb.cc_chain)
            cnf = vfft_k1_cc_chain_decode(zeb.cc_chain, cch);
        for (ci = 0; ci < cnf; ci++)
            if (cch[ci] & 1)
                codd = 1;
        if (codd)
            ze = &zeb;
    }
    int ze_hit = (ze && !cfg->recalibrate);
    /* Route forcing, read at CREATE (both directions follow — the
     * route is one plan field): VFFT_NO_ZTURN pins legacy (kill
     * switch; VFFT_NO_IL2P precedent) and wins over everything;
     * VFFT_FORCE_ZROUTE=legacy|zturn (or 0|1) is the gate/test hook
     * (VFFT_IL_PAD precedent). Unforced DEFAULT on a MISS is ZTURN
     * (2026-07-27 cutover); on a HIT the banked route verdict is
     * honored, whichever way it points. */
    int zforce = 0; /* 0 = none, 1 = legacy, 2 = zturn */
    {
        const char *fz = getenv("VFFT_FORCE_ZROUTE");
        if (fz && fz[0])
            zforce = (fz[0] == 'z' || fz[0] == 'Z' || fz[0] == '1')
                         ? 2
                         : 1;
        if (getenv("VFFT_NO_ZTURN"))
            zforce = 1;
    }
    /* The BANKED chain survives zsplit's rejection: a route-1 line
     * may carry a last==4 chain (the ZTURN radix-4 terminator) that
     * ONLY vfft_zturn2_create_chain can build — zsplit's create
     * rejects it (last==8-only), which previously zeroed znf and
     * OVERWROTE zch with the legacy default before the zturn replay
     * ever saw the banked bytes. zwch/zwnf keep them; zch/znf stay
     * the LEGACY arm's working copy (validator-is-the-law, per arm). */
    int zwch[VFFT_ZSPLIT_MAX_NF];
    int zwnf = 0;
    if (ze_hit && ze->cc_chain)
        zwnf = vfft_k1_cc_chain_decode(ze->cc_chain, zwch);
    if (zwnf)
    {
        memcpy(zch, zwch, sizeof zch);
        znf = zwnf;
        zs_pending = vfft_zsplit_create(N, zch, znf);
        if (!zs_pending)
            znf = 0; /* not legacy-legal (e.g. last==4): fall back */
    }
    if (!zs_pending)
    {
        znf = vfft_zsplit_default_chain(N, zch);
        if (znf)
            zs_pending = vfft_zsplit_create(N, zch, znf);
    }
    if (!ze_hit)
    {
        if (zs_pending)
            vfft_zsplit_destroy(zs_pending);
        return 0;
    }
    /* 🔴 zs_pending MAY BE NULL past this point, and that is a FIX, not an
     * accident (found 2026-08-02 by the replay probe): a route-1 line can
     * carry a legacy-illegal chain (last==4) at an N with NO
     * vfft_zsplit_default_chain entry (32768+). The old code required the
     * legacy fallback plan to exist before it would even ATTEMPT the zturn
     * replay, so the whole cascade silently dropped to the classic path —
     * at 32768 that served ~394us where the cascade serves ~44us, and the
     * bench labeled the row with the banked chain it never ran. A banked
     * ZTURN verdict must be replayable WITHOUT a legacy escort; the only
     * cost is that a later zturn-create failure then has no cascade
     * fallback (classic path, same as before this feature existed). */
    /* pure read: honor route + CHAIN + the winning route's
     * pick. cc_chain is the WINNING route's chain (Phase-5
     * planner tranche: dp_planner_il.h's route axis banks it
     * that way), so a route-1 line replays its chain through
     * vfft_zturn2_create_chain. Old race-banked route lines
     * carried the legacy default — which IS the chain zturn
     * shipped on, so replaying it is behavior-identical. A
     * fence-invalid banked chain (hand-edited / stale) falls
     * back to the calibrated-default create — skipped, never
     * force-fit (and zch/znf already fell back to the default
     * chain above if zsplit rejected it too). */
    if (zs_pending)
        zs_pending->t2q = ze->zs_t2q ? 1 : 0;
    if (!zs_pending && !((ze->zs_route == 1 && zforce != 1) || zforce == 2))
        return 0;   /* legacy verdict with no buildable legacy plan */
    if ((ze->zs_route == 1 && zforce != 1) || zforce == 2)
    {
        /* replay the BANKED chain (zwch — survives a legacy
         * rejection above, e.g. a last==4 chain), not the
         * legacy arm's working copy */
        if (zwnf)
            zt_pending = vfft_zturn2_create_chain(N, zwch, zwnf);
        if (!zt_pending)
            zt_pending = vfft_zturn2_create(N);
    }
    if (zt_pending)
    {
        zt_pending->t2q = ze->zt_t2q ? 1 : 0;
        zroute_pending = 1;
        /* tcut WIDTH replay. Absent field (zt_tw == 0) leaves
         * the plan calloc-untiled, i.e. exactly today's driver.
         *
         * 🔴 The banked width is only valid on the cache it was
         * tuned against. A width tuned on a 48 KB P-core and
         * replayed on a 32 KB E-core overshoots by 50%, and
         * overshoot is the failure mode that costs the whole
         * benefit at once instead of degrading. So a mismatch
         * means UNTILED (safe, today's behaviour) and a loud
         * line — never "use it anyway". */
        /* 🔴 EXPLICIT ENV BEATS WISDOM — same convention as
         * VFFT_FORCE_ZROUTE / VFFT_NO_ZTURN. If VFFT_TCUT is
         * set to ANYTHING (including "off"), the env gate's
         * verdict stands and the banked width is NOT applied.
         * Without this, `bench_1d_vs_mkl --tcut=off` against a
         * width-carrying wisdir would silently run TILED and
         * every off-vs-tiled A/B would compare tiled vs tiled
         * and read ~0%%. An arm that is not what its label says
         * is the exact failure class the engagement taps were
         * built to catch — this closes it at the source. */
        const char *tcenv = getenv("VFFT_TCUT");
        if (ze->zt_tw > 0 && tcenv && tcenv[0])
        {
            if (getenv("VFFT_TCUT_VERBOSE"))
                fprintf(stderr,
                        "[tcut] N=%d: banked width %d cplx "
                        "SUPPRESSED by explicit VFFT_TCUT=%s "
                        "(env override beats wisdom)\n",
                        N, ze->zt_tw, tcenv);
        }
        else if (ze->zt_tw > 0)
        {
            if (!vfft_cpu_l1d_matches(ze->zt_l1))
                fprintf(stderr,
                        "[tcut] N=%d: banked width %d cplx was "
                        "tuned for L1d=%d B, this machine has "
                        "%ld B -> running UNTILED, re-measure "
                        "this cell\n",
                        N, ze->zt_tw, ze->zt_l1,
                        vfft_cpu_l1d_bytes());
            else if (!vfft_zturn2_set_tile_w(zt_pending, 1,
                                            ze->zt_tw, 0, 0))
                fprintf(stderr,
                        "[tcut] N=%d: banked width %d cplx is "
                        "ILLEGAL for the banked chain -> "
                        "running UNTILED\n", N, ze->zt_tw);
            else if (getenv("VFFT_TCUT_VERBOSE"))
                /* Same shape as the env gate's line, so one
                 * parser reads both. Without it a banked width
                 * is INVISIBLE — the env path announces itself
                 * and the wisdom path would not, which is the
                 * asymmetry that lets a replay silently do
                 * something other than what was banked. */
                fprintf(stderr,
                        "[tcut] N=%d nf=%d tiled=%d tcut=%d "
                        "tfuse=%d tw=%s w=%ld NT=%ld "
                        "src=wisdom l1=%d\n",
                        N, zt_pending->nf, zt_pending->tiled,
                        zt_pending->tcut, zt_pending->tfuse,
                        zt_pending->thonest ? "honest" : "reset",
                        zt_pending->tw,
                        ((long)N / 4) / zt_pending->tw,
                        ze->zt_l1);
        }
    }
    if (getenv("VFFT_ZRACE_VERBOSE"))
        fprintf(stderr, "[zroute] N=%d wisdom hit: banked "
                        "route=%d zs_t2q=%d zt_t2q=%d force=%d -> "
                        "serving route=%d t2q=%d\n",
                N, ze->zs_route,
                ze->zs_t2q, ze->zt_t2q, zforce, zroute_pending,
                zroute_pending ? zt_pending->t2q
                               : (zs_pending ? zs_pending->t2q : -1));
    /* ROUTE ATOMICITY (structural): exactly ONE cascade plan
     * survives to the handle — the loser dies here, before the
     * handle exists — so fwd and bwd cannot pair across routes. */
    if (zroute_pending && zt_pending)
    {
        vfft_zsplit_destroy(zs_pending);
        zs_pending = NULL;
    }
    else
    {
        zroute_pending = 0;
        if (zt_pending)
        {
            vfft_zturn2_destroy(zt_pending);
            zt_pending = NULL;
        }
    }
    if (!zs_pending && !zt_pending)
        return 0;   /* route-1 create failed and no legacy escort — a miss */
    (void)znf;
    *zs_out = zs_pending;
    *zt_out = zt_pending;
    *zroute_out = zroute_pending;
    return 1;
}

/* K=1 SCRAMBLED-contract cascade MISS race (>=2048): default chain + the
 * stf/stf2 t2q race + bank kind-4 + route atomicity. THE single definition,
 * factored out of the OOP create (2026-08-25) so the IN-PLACE create can
 * run it too — before this, only the OOP create raced/banked and an
 * in-place caller on a cold store replayed nothing and fell to convert
 * forever (the same hit-only disease the ord=scr ILP fix cured sub-2048;
 * convert-arm census class 3). t2q picks must be MEASURED on the installed
 * binary — stf/stf2 are bit-identical, so the delta is code-placement
 * order, never a hand-set constant. See docs/design/vfft_front_door.md.
 * Returns 1 with exactly one plan attached (route atomicity), 0 = no
 * cascade for this N (caller keeps its previous serving).
 * ip=1: the IN-PLACE caller's CANDIDATE build — the t2q pick is timed on
 * the ALIASED call form (in-place has its own memory-access structure;
 * owner 2026-08-25), and the kind-4 cell is NOT banked: that cell is the
 * OOP create's verdict (single writer). The in-place verdict lives in the
 * ord=scr lay=il MODE cell (mode=zcasc|conv), banked by the caller after
 * its cascade-vs-convert race. */
static int _k1z_race_and_bank(const vfft_config_t *cfg,
                              struct vfft_wisdom_s *W, int N, int ip,
                              vfft_zsplit_plan_t **zs_out,
                              vfft_zturn2_plan_t **zt_out, int *zroute_out)
{
    vfft_zsplit_plan_t *zs_pending = NULL;
    vfft_zturn2_plan_t *zt_pending = NULL;
    int zroute_pending = 0;
    int zch[VFFT_ZSPLIT_MAX_NF];
    int znf;
    if (N < 2048)
        return 0; /* the cascade tier boundary — same guard as replay */
    znf = vfft_zsplit_default_chain(N, zch);
    /* Route forcing for the MISS race (the HIT path reads it inside
     * _k1z_wisdom_replay): VFFT_NO_ZTURN pins legacy,
     * VFFT_FORCE_ZROUTE=legacy|zturn is the test hook. An env PARSE is
     * not replay semantics, so this small read may live in both places
     * without the two-writers hazard. */
    int zforce = 0;
    {
        const char *fz = getenv("VFFT_FORCE_ZROUTE");
        if (fz && fz[0])
            zforce = (fz[0] == 'z' || fz[0] == 'Z' || fz[0] == '1') ? 2 : 1;
        if (getenv("VFFT_NO_ZTURN"))
            zforce = 1;
    }
    int zodd = 0;
    {
        /* ODD-MID chains (2026-08-27): the LEGACY zsplit engine's kind
         * set is radix 4/8 only, so an odd-factor chain has NO legacy
         * twin — the cell is ZTURN-ONLY. Skip the legacy build (its
         * create on an odd chain would refuse anyway) and let the
         * zturn arm carry the route alone. */
        int s2;
        for (s2 = 0; znf && s2 < znf; s2++)
            if (zch[s2] & 1)
                zodd = 1;
        if (znf && !zodd)
            zs_pending = vfft_zsplit_create(N, zch, znf);
        if (!zs_pending && (!znf || !zodd))
            return 0;
        if (zodd && zforce == 1)
            return 0; /* legacy pinned, but no legacy twin exists */
    }
    {
        double zns = 0.0;
        if (zforce != 1)
            zt_pending = vfft_zturn2_create(N);
        if (zt_pending)
        {
            zns = _calibrate_zturn_t2q(zt_pending, cfg->rigor, ip);
            if (zns > 0.0)
                zroute_pending = 1;
        }
        if (!zroute_pending && zs_pending)
            zns = _calibrate_zsplit_t2q(zs_pending, cfg->rigor, ip);
        else if (!zroute_pending)
            zns = 0.0; /* zturn-only cell and the zturn arm failed */
        if (zns > 0.0)
        {
            vfft_oop_wisdom_entry_t ne;
            memset(&ne, 0, sizeof ne);
            ne.N = N;
            ne.K = 1;
            ne.kind = VFFT_OOP_KIND_ZSPLIT;
            ne.zs_t2q = zs_pending ? zs_pending->t2q : 0;
            /* cc_chain = the WINNING route's chain (the reader contract).
             * At this create-time race both routes still run the same
             * default chain, so the encode is byte-identical either way
             * today — the chain-searched winners come from the offline
             * planner (dp_planner_il.h route axis / the calibrate_zchain
             * driver), not this race. */
            if (zroute_pending && zt_pending)
                ne.cc_chain = vfft_k1_cc_chain_encode(zt_pending->chain,
                                                      zt_pending->nf);
            else if (zs_pending)
                ne.cc_chain = vfft_k1_cc_chain_encode(zs_pending->chain,
                                                      zs_pending->nf);
            ne.zs_route = zroute_pending;
            ne.zt_t2q = zt_pending ? zt_pending->t2q : 0;
            /* tcut width + the cache it was tuned against. 0 when untiled,
             * which keeps the banked line byte-identical to the pre-width
             * format. This race does not SEARCH widths (that is the
             * planner's job); it records whatever width the plan is
             * carrying so a verdict is never banked as untiled when it
             * was not. */
            ne.zt_tw = (zt_pending && zt_pending->tiled == 1)
                           ? (int)zt_pending->tw : 0;
            ne.zt_l1 = ne.zt_tw ? (int)vfft_cpu_l1d_bytes() : 0;
            /* MEASURE-LESS bank (ns=0): this race's median is fwd-only
             * placement luck (§4.9993), not the cell's joint2 verdict —
             * kind-4 carries ns only from the dp planner. A measure-less
             * row can always be replaced by the planner's measured one;
             * the reverse is refused by the merge law, exactly the
             * intended authority order. */
            ne.ns = 0.0;
            /* an ODD chain must WIN the commit-site route race (vs the
             * true incumbent incl. the k1 IL routes) before any banking
             * — banking here would make replay attach it by fiat. The
             * sweep owns banking those winners. */
            if (!ip && !zodd) /* kind-4 = the OOP create's cell */
                vw2_oop_bank_entry(&W->vw2, &ne);
            else
                /* the in-place / odd race's RECIPE, as a COMPONENT row
                 * (role=comp): the terminator pick and tile width it just
                 * raced are banked, never re-raced per create, and the
                 * mode row signposts it (owner, 2026-09-02). Not a
                 * verdict: the OOP create never replays it. */
                vw2_oop_bank_entry_role(&W->vw2, &ne, VW2_ROLE_COMP);
            _vw2_persist(W, cfg);
        }
        /* ROUTE ATOMICITY (structural): exactly ONE cascade plan survives
         * to the handle — the loser dies here, before the handle exists —
         * so fwd and bwd cannot pair across routes. */
        if (zroute_pending && zt_pending)
        {
            vfft_zsplit_destroy(zs_pending);
            zs_pending = NULL;
        }
        else
        {
            zroute_pending = 0;
            if (zt_pending)
            {
                vfft_zturn2_destroy(zt_pending);
                zt_pending = NULL;
            }
        }
    }
    *zs_out = zs_pending;
    *zt_out = zt_pending;
    *zroute_out = zroute_pending;
    /* a zturn-only (odd-mid) cell whose zturn arm failed has NOTHING —
     * the caller must fall through to the classic OOP kinds. */
    return (zs_pending || zt_pending) ? 1 : 0;
}

#endif /* VFFT_OOP_K1_COMMIT_H */
