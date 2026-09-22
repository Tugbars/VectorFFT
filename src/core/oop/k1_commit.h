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
 * THE STRIDE-ROW BANKER
 * ---------------------
 * One is left (2026-09-18): _bank_nat_1d, the in-place natural cell
 * (ord=nat), called from the in-place create. Its three siblings --
 * _bank_natoop_1d (out-of-place natural), _bank_scrmode_oop_1d (the
 * out-of-place ord=scr mode cell) and _bank_nat_raced (the banked-loss
 * marker) -- had ZERO callers anywhere in src/ or benches/ and were deleted
 * whole (clean library, not history; survey section D). Rows a shipped store
 * already carries are still SERVED; nothing writes new ones.
 * The order axis does not share a cell with the scrambled one, which is why
 * the surviving banker is per-cell: a natural-order create can never perturb
 * the scrambled plan, and vice versa - the regimes are calibrated
 * independently because they are genuinely different engines, not one engine
 * with a flag.
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
#include "wisdom2/wisdom2_oop_reader.h"     /* the kind-3/kind-4 codecs */
#include "wisdom2/wisdom2_stride_reader.h"  /* the @nat / @natoop / mode cells */
#include "support/race.h"                   /* the shared race body */

/* Applies a banked kind-3 il_kv verdict; measures nothing (dp_planner_il.h
 * owns that race). il_kv==0 keeps create's default — blocked at R>=32. */
/* the CHAIN3 twin (2026-09-03): the banked three-slot il_kv on the chain3
 * row overrides the create's structural defaults; env VFFT_IL_KV pins. No
 * backward cell yet (the chain's backward leaf slot is a bwd-axis item). */
static void _k1_il3p_apply_kv(vfft_il3p_plan_t *p,
                              const vfft_oop_wisdom_entry_t *ke,
                              const vw2_store_t *st, int N, int ip)
{
    if (!p)
        return;
    if (ke)
        vfft_il3p_apply_kv_forms(p, ke->il_kv);
    /* the BACKWARD cell (2026-09-03): its own dir=bwd row, validated against
     * the chain it was raced at; outside the ke guard like the pair's */
    if (st)
    {
        int c3[3] = { 0, 0, 0 };
        int bkv = vw2_oop_lookup_k1_bwd_chain_pl(st, N, c3, ip ? VW2_PL_IP : VW2_PL_OOP);
        if (bkv >= 0 && c3[0] == p->R2 && c3[1] == p->A && c3[2] == p->B)
            vfft_il3p_apply_kv_forms_bwd(p, bkv);
    }
    {
        const char *e = getenv("VFFT_IL_KV");
        if (e && e[0])
            vfft_il3p_apply_kv_forms(p, (int)strtol(e, NULL, 0));
    }
    {
        const char *e = getenv("VFFT_IL_BKV");
        if (e && e[0])
            vfft_il3p_apply_kv_forms_bwd(p, (int)strtol(e, NULL, 0));
    }
}

static void _k1_il2p_apply_kv(vfft_il2p_plan_t *p,
                              const vfft_oop_wisdom_entry_t *ke,
                              const vw2_store_t *st, int N, int ip)
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
        int bkv = vw2_oop_lookup_k1_bwd_pl(st, N, &bR1, &bR2, ip ? VW2_PL_IP : VW2_PL_OOP);   /* -1 = no row */
        if (bkv >= 0 && bR1 == p->R1 && bR2 == p->R2)
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
static int _k1fs_row_sb(struct vfft_wisdom_s *W, int N, int il_kv, int *chain, int *form); /* below, with the threaded arm */
static void _k1_il_candidate(struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                             int N, vfft_il2p_plan_t **il2p_out,
                             vfft_il3p_plan_t **il3p_out,
                             vfft_ilfd_plan_t **ilfd_out,
                             vfft_ztt_plan_t **ztt_out,
                             vfft_k1fs_plan_t **fs_out,
                             vfft_ilprime_plan_t **ilp_out); /* defined below */

/* ── THE PRIME CELL'S INNER, RACED (2026-09-18; owner 2026-09-17: "prime
 * cells should have their own inner race"; ilprime_inner_race_design.md) ──
 * A prime N is a convolution done with an FFT of length M (Rader: N - 1;
 * Bluestein: the next power of two >= 2N - 1). Until today the inner was
 * BORROWED, in three layers: the K=1 tier's banked scrambled row at M (read
 * with no recalibrate term), else the K=1 candidate at M below 4096, else
 * the engine's structural rule -- the most balanced pair, or a default
 * chain. Chosen by proxy (M standalone is not M inside a convolution),
 * unreachable by recalibrate without racing M's cell nested inside N's
 * create (which _k1_il_dp_busy forbids), and a heuristic at the end. Now
 * the prime cell RACES its inner: every buildable (method, inner) pair,
 * built directly from a descriptor (no planner call, so the lock is never
 * touched), timed on the WHOLE convolution, the winner banked on the prime
 * cell's OWN row and replayed from there. */
typedef struct {
    int kind;                       /* 1 = il2p pair, 2 = il3p chain, 3 = ZTURN-T */
    int R1, R2;                     /* 1 */
    int cR2, cA, cB;                /* 2 */
    int zt[VFFT_ZTT_MAX_NF], ztn;   /* 3: the chain */
    int tw;                         /* 3: the tile, 0 = untiled */
    int failed;                     /* THIS descriptor could not build: the create's
                                     * structural fallback then served, and the plan
                                     * must be discarded -- never a silent substitute */
} _ilprime_inner_desc_t;

static void _ilprime_desc_str(const _ilprime_inner_desc_t *d, char *kind, size_t ksz, char *shape, size_t ssz)
{
    if (d->kind == 1) { snprintf(kind, ksz, "2p"); snprintf(shape, ssz, "%d.%d", d->R1, d->R2); }
    else if (d->kind == 2) { snprintf(kind, ksz, "3p"); snprintf(shape, ssz, "%d.%d.%d", d->cR2, d->cA, d->cB); }
    else
    {
        int q, off = 0;
        snprintf(kind, ksz, "ztt");
        shape[0] = 0;
        for (q = 0; q < d->ztn && off < (int)ssz - 4; q++)
            off += snprintf(shape + off, ssz - (size_t)off, "%s%d", q ? "." : "", d->zt[q]);
    }
}
static int _ilprime_desc_parse(_ilprime_inner_desc_t *d, const char *kind, const char *shape, int tw)
{
    memset(d, 0, sizeof *d);
    if (!strcmp(kind, "2p"))
        return sscanf(shape, "%d.%d", &d->R1, &d->R2) == 2 && (d->kind = 1);
    if (!strcmp(kind, "3p"))
        return sscanf(shape, "%d.%d.%d", &d->cR2, &d->cA, &d->cB) == 3 && (d->kind = 2);
    if (!strcmp(kind, "ztt"))
    {
        const char *p = shape;
        while (*p && d->ztn < VFFT_ZTT_MAX_NF)
        {
            char *end;
            long r = strtol(p, &end, 10);
            if (end == p || r < 2) return 0;
            d->zt[d->ztn++] = (int)r;
            p = (*end == '.') ? end + 1 : end;
            if (*end != '.' && *end != '\0') return 0;
        }
        d->tw = tw;
        return d->ztn >= 2 && (d->kind = 3);
    }
    return 0;
}

/* the provider: exactly ONE inner, from a descriptor -- never a guess */
static int _ilprime_inner_from_desc(int M, _ilprime_inner_t *in, void *v)
{
    _ilprime_inner_desc_t *d = (_ilprime_inner_desc_t *)v;
    d->failed = 0;
    if (d->kind == 1)
    {
        in->p2 = vfft_il2p_create(M, d->R1, d->R2);
        if (in->p2) return 1;
    }
    else if (d->kind == 2)
    {
        in->p3 = vfft_il3p_create(M, d->cR2, d->cA, d->cB);
        if (in->p3) return 1;
    }
#ifdef VFFT_ZTT_H
    else if (d->kind == 3)
    {
        vfft_ztt_plan_t *zp = vfft_ztt_create_chain_ord(M, d->zt, d->ztn, 1);
        if (zp && d->tw > 0 && !vfft_ztt_set_tile(zp, (size_t)d->tw)) { vfft_ztt_destroy(zp); zp = NULL; }
        if (zp) { in->pt = zp; return 1; }
    }
#endif
    d->failed = 1;
    return 0;
}

/* THE POOL at length M: every legal il2p pair (the structural rule's own
 * loop, minus its "pick the most balanced"), the il3p default chain, and
 * ZTURN-T's registry chains at M untiled and at each legal tile -- the K=1
 * planner's own ZTURN-T pool (_il_dp_enumerate_ztt_ord), scrambled class:
 * the convolution is a matched roundtrip in any order. A pool, not a rule:
 * nothing in it is a default. */
static int _ilprime_inner_cands(int M, _ilprime_inner_desc_t *out, int max)
{
    int n = 0, dropped = 0;
    if (M <= 4096)
    {
        int R2;
        for (R2 = (M < 64 ? M : 64); R2 >= 3; R2--)
        {
            int R1;
            if (M % R2) continue;
            R1 = M / R2;
            if (R1 < 3 || R1 > 64) continue;
            if (!vfft_il2p_leaf_fn(R2, 0) || !vfft_il2p_mid_fn(R1, 0)) continue;
            if (n >= max) { dropped++; continue; }
            memset(&out[n], 0, sizeof out[n]);
            out[n].kind = 1; out[n].R1 = R1; out[n].R2 = R2; n++;
        }
        {
            int cR2, cA, cB;
            if (vfft_il3p_default_chain(M, &cR2, &cA, &cB))
            {
                if (n >= max) dropped++;
                else { memset(&out[n], 0, sizeof out[n]); out[n].kind = 2; out[n].cR2 = cR2; out[n].cA = cA; out[n].cB = cB; n++; }
            }
        }
    }
#ifdef VFFT_ZTT_H
    /* ZTURN-T from the K=1 PLANNER'S OWN enumerators (2026-09-19) rather than
     * a second copy of the registry walk and the tile ladder, which is what
     * stood here for a day. Two grammars, each with its own ladder: the pow2
     * registry, and the 2^a*odd chain grammar. The second one is the point --
     * Rader's inner sits at M = N - 1, which is a power of two only at a
     * Fermat-shaped prime, so without the odd grammar Rader offered NOTHING
     * at 8191, 12289 and 40961 and the cell banked Bluestein unopposed.
     * Scrambled class: a convolution is a matched roundtrip in any order. */
    {
        static vfft_il_cand_t buf[VFFT_IL_DP_MAX_CAND];
        vfft_il_cand_sink_t sink;
        int i, q;
        memset(&sink, 0, sizeof sink);
        sink.out = buf;
        _il_dp_enumerate_ztt_ord(M, &sink, 1);
        _il_dp_enumerate_ztt_odd(M, &sink, 1);
        for (i = 0; i < sink.n; i++)
        {
            if (buf[i].il_zt_n < 2 || buf[i].il_zt_n > VFFT_ZTT_MAX_NF) continue;
            if (n >= max) { dropped++; continue; }
            memset(&out[n], 0, sizeof out[n]);
            out[n].kind = 3;
            out[n].ztn = buf[i].il_zt_n;
            out[n].tw = buf[i].il_tw;
            for (q = 0; q < buf[i].il_zt_n; q++) out[n].zt[q] = buf[i].il_zt[q];
            n++;
        }
        dropped += sink.dropped;   /* the planner's own cap, reported here too */
    }
#endif
    if (dropped)   /* the no-silent-caps law */
        _vfft_warn("ilprime inner pool capped at %d (%d candidate(s) dropped) at M=%d", max, dropped, M);
    return n;
}

/* build the prime plan for one (method, inner); a descriptor that cannot
 * build discards the plan the structural fallback would have served */
static vfft_ilprime_plan_t *_ilprime_build_with(int N, int rader, _ilprime_inner_desc_t *d)
{
    vfft_ilprime_plan_t *p;
    _ilprime_inner_provider = _ilprime_inner_from_desc;
    _ilprime_inner_provider_ctx = d;
    p = rader ? _ilprime_create_rader(N) : _ilprime_create_bluestein(N);
    _ilprime_inner_provider = 0;
    _ilprime_inner_provider_ctx = 0;
    if (p && d->failed) { vfft_ilprime_destroy(p); p = 0; }
    return p;
}
typedef struct { vfft_ilprime_plan_t *p; double *zi, *zo; } _ilprime_iarm_t;
static void _ilprime_iarm_run(void *v)
{
    _ilprime_iarm_t *a = (_ilprime_iarm_t *)v;
    if (a->p->method == 1) _ilprime_exec_rader(a->p, a->zi, a->zo, 0);
    else _ilprime_exec_bluestein(a->p, a->zi, a->zo, 0);
}
/* one same-run race over np live plans: the index of the fastest */
static int _ilprime_race_plans(vfft_ilprime_plan_t **plans, int np, double *zi, double *zo)
{
    _ilprime_iarm_t ctx[VFFT_RACE_MAX_ARMS];
    vfft_race_arm_t arms[VFFT_RACE_MAX_ARMS];
    double ns[VFFT_RACE_MAX_ARMS];
    const vfft_race_proto_t proto = { 3, 1, VFFT_RACE_MIN, 1, 1, NULL, NULL, 1 };   /* min-of-3, alternated, one warm pass */ /* single-thread arms: paced (VFFT_RACE_PACE_MS) */
    int a, w = 0;
    if (np <= 1) return 0;
    for (a = 0; a < np; a++)
    {
        ctx[a].p = plans[a]; ctx[a].zi = zi; ctx[a].zo = zo;
        arms[a].name = plans[a]->method == 1 ? "rader" : "bluestein";
        arms[a].run = _ilprime_iarm_run;
        arms[a].ctx = &ctx[a];
    }
    vfft_race_run(&proto, arms, np, ns);
    for (a = 1; a < np; a++)
        if (ns[a] < ns[w]) w = a;
    return w;
}

/* THE POOL is complete (the K=1 planner's own at M, both methods); a plan at
 * M = 262144 costs ~22 MB, so the pool is built and raced in HEATS of
 * sixteen -- each heat's winner stays alive, the rest are destroyed -- and
 * the heat winners meet in one same-run FINAL. */
/* 480 / 16 = 30 heat winners, inside the race body's 32-arm final. The
 * 2^a*odd grammar alone yields a few hundred chains at some M. */
#define _ILPR_MAX_CANDS 480
#define _ILPR_HEAT      16
typedef struct { int rader; _ilprime_inner_desc_t d; } _ilprime_cand_t;

/* the prime cell, banked: replay its OWN verdict (method + inner), else
 * race every buildable (method, inner) pair on the whole convolution and
 * bank the winner. Env pin VFFT_ILPR_METHOD never replays or banks. */
static vfft_ilprime_plan_t *_ilprime_create_banked(struct vfft_wisdom_s *W,
                                                   const vfft_config_t *cfg,
                                                   int N)
{
    int hint = 0;
    /* COMPOSITES COME IN TOO (2026-09-19). `!_ilprime_is_prime(N)` stood in
     * this condition, so a composite left on the first line and fell to
     * vfft_ilprime_create -- no store, no pool, and a structural inner that
     * is a balanced pair and therefore stops at M = 4096. That is why every
     * composite above 2048 with no chain was REFUSED: not a missing
     * algorithm, since Bluestein needs no primality, but a missing inner.
     * Through here it gets the same raced pool the primes get, which reaches
     * as far as the ZTURN-T grammars do. The Rader ARM is excluded below. */
    if (!W || W->vw2_off_oop || getenv("VFFT_ILPR_METHOD"))
        return vfft_ilprime_create(N);
    if (!cfg->recalibrate)
    {
        char kind[8], shape[64];
        int tw = 0;
        hint = vw2_prime_method_lookup(&W->vw2, N);
        if (hint && vw2_prime_inner_lookup(&W->vw2, N, kind, sizeof kind, shape, sizeof shape, &tw))
        {
            _ilprime_inner_desc_t d;
            if (_ilprime_desc_parse(&d, kind, shape, tw))
            {
                vfft_ilprime_plan_t *p = _ilprime_build_with(N, hint == 1, &d);
                if (p)
                {
                    if (getenv("VFFT_ILPR_LOG"))
                        fprintf(stderr, "[ilprime] N=%d: replay %s inner %s %s tw=%d src=wisdom\n",
                                N, hint == 1 ? "RADER" : "BLUESTEIN", kind, shape, tw);
                    return p;
                }
            }
            /* a banked inner that no longer builds: race below */
        }
    }
    {   /* THE RACE */
        static _ilprime_cand_t cands[_ILPR_MAX_CANDS];   /* 256 descriptors: off the stack */
        vfft_ilprime_plan_t *fin[_ILPR_MAX_CANDS / _ILPR_HEAT + 1];
        int fin_ci[_ILPR_MAX_CANDS / _ILPR_HEAT + 1];
        int nc = 0, nfin = 0, mi, h, w, wci, nbuilt = 0;
        int ncm[2] = { 0, 0 }, builtm[2] = { 0, 0 };   /* [0] = rader, [1] = bluestein */
        double *zi, *zo;
        for (mi = 0; mi < 2; mi++)
        {
            const int rader = (mi == 0);
            /* Rader is prime-only (see _ilprime_create_rader): skip the arm
             * rather than let it offer inners that can never build, which
             * would also trip the built-none warning on every composite. */
            if (rader && !_ilprime_is_prime(N)) continue;
            static _ilprime_inner_desc_t pool[_ILPR_MAX_CANDS];   /* off the stack */
            int M, n, q;
            /* The banked method is NOT a filter here (2026-09-19). It used
             * to narrow the race to its own inners, and then a binary that
             * cannot build that method -- a store calibrated against a
             * wider pool, or an engine whose reach changed -- was left with
             * an empty race and the cell REFUSED. Reaching this point at
             * all means the row could not be replayed, so its method is a
             * preference, not a measurement of what this build can do.
             * A verdict that cannot be built is a miss; a miss races. */
            if (rader) M = N - 1;
            else { M = 16; while (M < 2 * N - 1) M <<= 1; }
            n = _ilprime_inner_cands(M, pool, _ILPR_MAX_CANDS - nc);
            ncm[mi] = n;   /* this method's pool: zero means it never raced */
            for (q = 0; q < n; q++) { cands[nc].rader = rader; cands[nc].d = pool[q]; nc++; }
        }
        if (nc == 0)
            return 0;   /* nothing to build: nothing to serve, nothing to fall back to */
        zi = _ilprime_alloc((size_t)2 * N);
        zo = _ilprime_alloc((size_t)2 * N);
        if (!zi || !zo) { VFFT_IL2P_FREE(zi); VFFT_IL2P_FREE(zo); return 0; }
        for (h = 0; h < 2 * N; h++) zi[h] = 1.0 + 1e-6 * (double)(h & 255);
        /* HEATS */
        for (h = 0; h < nc; h += _ILPR_HEAT)
        {
            vfft_ilprime_plan_t *plans[_ILPR_HEAT];
            int ci[_ILPR_HEAT], np = 0, q, a;
            for (q = h; q < nc && q < h + _ILPR_HEAT; q++)
            {
                vfft_ilprime_plan_t *p = _ilprime_build_with(N, cands[q].rader, &cands[q].d);
                if (!p) continue;
                plans[np] = p; ci[np] = q; np++;
                builtm[cands[q].rader ? 0 : 1]++;
            }
            if (np == 0) continue;
            nbuilt += np;
            w = _ilprime_race_plans(plans, np, zi, zo);
            for (a = 0; a < np; a++)
                if (a != w) vfft_ilprime_destroy(plans[a]);
            fin[nfin] = plans[w]; fin_ci[nfin] = ci[w]; nfin++;
        }
        if (nfin == 0) { VFFT_IL2P_FREE(zi); VFFT_IL2P_FREE(zo); return 0; }
        /* THE FINAL */
        w = _ilprime_race_plans(fin, nfin, zi, zo);
        VFFT_IL2P_FREE(zi); VFFT_IL2P_FREE(zo);
        for (h = 0; h < nfin; h++)
            if (h != w) vfft_ilprime_destroy(fin[h]);
        wci = fin_ci[w];
        {
            char kind[8], shape[64];
            _ilprime_desc_str(&cands[wci].d, kind, sizeof kind, shape, sizeof shape);
            /* Rader's inner sits at M = N - 1, which is never a power of two,
             * so it has no ZTURN-T chain -- and above M = 4096 the pair and
             * the default chain3 run out too. When that happens Rader offers
             * arms that do not build, they vanish, and the cell banks
             * "bluestein" as though it had won a race it was alone in. Say so. */
            if (ncm[0] > 0 && builtm[0] == 0)
                _vfft_warn("ilprime N=%d: RADER offered %d inner(s) at M=%d and built NONE -- "
                           "the banked verdict is Bluestein BY DEFAULT, not by race",
                           N, ncm[0], N - 1);
            if (ncm[1] > 0 && builtm[1] == 0)
                _vfft_warn("ilprime N=%d: BLUESTEIN offered %d inner(s) and built NONE",
                           N, ncm[1]);
            if (getenv("VFFT_ILPR_LOG"))
                fprintf(stderr, "[ilprime] N=%d: inner race %d arm(s) in %d heat(s) "
                                "[rader %d/%d, blue %d/%d] -> %s inner %s %s tw=%d, banked\n",
                        N, nbuilt, nfin, builtm[0], ncm[0], builtm[1], ncm[1],
                        fin[w]->method == 1 ? "RADER" : "BLUESTEIN", kind, shape, cands[wci].d.tw);
            if (vw2_prime_method_bank(&W->vw2, N, fin[w]->method == 1 ? 1 : 2,
                                      kind, shape, cands[wci].d.tw) == VW2_OK)
                _vw2_persist(W, cfg);
        }
        return fin[w];
    }
}

/* ── THE IL PLAN RACE AT CREATE (2026-09-03, owner: "why don't we try
 * different factorizations for IL and see what wins") ─────────────────────
 * A kind-3 MISS (or recalibrate) below 2048 runs the IL dp planner — the
 * same search calibrate_k1 runs offline: every legal pair x its kernel
 * forms, every legal 3-stage chain x forms, the order swap, the backward
 * forms — and banks its verdicts (the kind-3 lay=il row, the dir=bwd row)
 * before the create replays them. There is no heuristic plan any more at
 * this tier: what serves was measured. A cold cell takes seconds; the
 * planner logs on entry. VFFT_NO_K1PLAN=1 skips it (probe hook). */
static vfft_il_dp_context_t _k1_il_dp_ctx;      /* planning side, one create at a time */
static int _k1_il_dp_ctx_ready = 0;
static int _k1_il_dp_busy = 0;                  /* a race in progress: nested calls refuse */
/* the race CEILINGS live in planning/policy.h (L9, 2026-09-16) */
static int _k1_il_plan_race(struct vfft_wisdom_s *W, const vfft_config_t *cfg, int N)
{
    vfft_il_cand_t top;
    int lines;
    /* WISDOM OR RACE, never a fallback (owner's law, 2026-09-09): a request
     * names (N, layout, order, placement); the door looks that cell up and
     * on a miss RACES the interleaved planner's pool, banks the winner and
     * serves it. Every N the planner covers races here: below 2048, the odd
     * cells above it (no factor of 4), and — since 2026-09-09 — the pow2
     * cells of ZTURN-T's band up to its ceiling. Until then a pow2 N >= 2048
     * returned 0 here and a cold band cell fell through to the prime engine
     * (Bluestein at 32768, seen in the natural front gate's tap). The
     * composite cells with a factor of 4 above 2048 outside ZTURN-T's odd
     * band stayed out as "the cascade's" until 2026-09-22; the cascade is
     * gone and they race here like every other cell (policy.h). */
    if (!W || W->vw2_off_oop || N < 2 || getenv("VFFT_NO_K1PLAN"))
        return 0;
    {   /* ownership + budget, one question (planning/policy.h, L1/L9) */
        vfft_cell_t pc = vfft_policy_cell(cfg, N, 1, 1, 0, 1);
        if (!vfft_policy_races(&pc))
            return 0;
    }
    /* the planner context is ONE static; the four-step's candidates create
     * 2D children whose row plans come back through this door — a nested
     * race would corrupt the outer one, so (a) the row cells are warmed
     * BEFORE the race (created and destroyed once: a cold one races and
     * banks its own row) and (b) a nested call refuses (2026-09-15) */
    if (_k1_il_dp_busy)
        return 0;
    if ((N & (N - 1)) == 0 && vfft_k1fs_band(N))
    {
        int n1[8], n2[8], ns = vfft_k1fs_splits(N, n1, n2, 8), i, j;
        for (i = 0; i < ns; i++)
        {
            int seen = 0;
            for (j = 0; j < i; j++) if (n2[j] == n2[i]) seen = 1;
            if (seen) continue;
            {
                vfft_config_t rc;
                vfft_plan rp;
                memset(&rc, 0, sizeof rc);
                rc.transform = VFFT_C2C; rc.placement = VFFT_INPLACE; rc.rigor = cfg->rigor;
                rc.dims = 1; rc.n[0] = n2[i]; rc.howmany = 1; rc.order = VFFT_ORDER_NATURAL;
                rc.layout = VFFT_LAYOUT_INTERLEAVED; rc.nthreads = 1;
                rc.wisdom = (vfft_wisdom *)W; rc.wisdom_write = cfg->wisdom_write;
                rp = vfft_create(&rc);
                if (rp) vfft_destroy(rp);
            }
        }
    }
    /* the PRIME arm (2026-09-21): the prime cell built ONCE here -- a cold
     * cell races its inner pool and banks the prime shard's row -- and lent
     * to the race through _k1pr_ctx (dp_planner_il.h); every non-pow2 cell
     * where it builds (the band map admits it at every non-pow2 N). The
     * verdict's plan is handed to the candidate after the race; a losing
     * plan dies below. */
    _k1pr_release();
    if ((N & (N - 1)) != 0 && cfg->layout == VFFT_LAYOUT_INTERLEAVED)
    {
        _k1pr_ctx.plan = _ilprime_create_banked(W, cfg, N);
        _k1pr_ctx.N = _k1pr_ctx.plan ? N : 0;
    }
    _k1fs_ctx.W = W;
    _k1fs_ctx.cfg = cfg;
    if (!_k1_il_dp_ctx_ready)
    {
        vfft_il_dp_init(&_k1_il_dp_ctx, VFFT_K1_IL_PLAN_MAX_N);
        _k1_il_dp_ctx_ready = 1;
    }
    if (N > _k1_il_dp_ctx.max_N)
    {   /* the flat DIT's cells (2026-09-05) and the four-step's (2026-09-15):
         * grow the scratch planes on demand — the candidate cache restarts,
         * wisdom is the memory */
        vfft_il_dp_destroy(&_k1_il_dp_ctx);
        vfft_il_dp_init(&_k1_il_dp_ctx, N > VFFT_K1_IL_PLAN_ODD_MAX_N ? N : VFFT_K1_IL_PLAN_ODD_MAX_N);
    }
    if (cfg->rigor != VFFT_MEASURE)
        vfft_il_dp_set_patient(&_k1_il_dp_ctx);
    else
        _k1_il_dp_ctx.beam = VFFT_IL_DP_BEAM_MEASURE;
    if (getenv("VFFT_NAT_LOG"))
        fprintf(stderr, "[k1plan] N=%d: IL plan race (solos, pairs x forms, ZTURN-T "
                        "chains x widths, chain3 x forms, bwd forms) — a cold cell "
                        "takes seconds\n", N);
    _k1_il_dp_busy = 1;
    lines = vfft_il_dp_plan_and_bank(&_k1_il_dp_ctx, &W->vw2, N,
                                     cfg->placement == VFFT_INPLACE,   /* the cell's placement (2026-09-21) */
                                     getenv("VFFT_IL_DP_VERBOSE") != NULL);
    _k1_il_dp_busy = 0;
    if (lines > 0)
        _vw2_persist(W, cfg);
    if (_k1pr_ctx.plan)
    {   /* the prime plan outlives the race only as the request's verdict */
        vfft_oop_wisdom_entry_t pe;
        const int ip_req = (cfg->placement == VFFT_INPLACE);
        const int scr_req = (vfft_policy_ord_k1(cfg, N, ip_req) == VW2_ORD_SCR);
        if (!(vw2_oop_lookup_k1_cell(&W->vw2, N, scr_req, ip_req, &pe) &&
              pe.k1_il_route == VFFT_K1_IL_PRIME))
            _k1pr_release();
    }
    if (getenv("VFFT_NAT_LOG") &&
        vfft_il_dp_rank(&_k1_il_dp_ctx, N,
                        cfg->order == VFFT_ORDER_SCRAMBLED ? VFFT_IL_ORD_SCRAMBLED
                                                           : VFFT_IL_ORD_NATURAL,
                        &top, 1) == 1)
        fprintf(stderr, "[k1plan] N=%d: ilp=%.0fns -> ILP (route %d, %d.%d, %d line(s) banked, %s cell)\n",
                N, top.cost_ns, top.route, top.R1, top.R2, lines,
                cfg->order == VFFT_ORDER_SCRAMBLED ? "ord=scr" : "ord=nat");
    return lines;
}

static void _k1_il_candidate(struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                             int N,
                             vfft_il2p_plan_t **il2p_out,
                             vfft_il3p_plan_t **il3p_out,
                             vfft_ilfd_plan_t **ilfd_out,   /* NULL = caller cannot take a flat plan */
                             vfft_ztt_plan_t **ztt_out,     /* NULL = caller cannot take a ZTURN-T plan */
                             vfft_k1fs_plan_t **fs_out,     /* NULL = caller cannot take a four-step plan */
                             vfft_ilprime_plan_t **ilp_out) /* NULL = caller cannot take the prime cell */
{
    *il2p_out = NULL;
    *il3p_out = NULL;
    if (ilfd_out) *ilfd_out = NULL;
    if (ztt_out) *ztt_out = NULL;
    if (fs_out) *fs_out = NULL;
    if (ilp_out) *ilp_out = NULL;
    if (getenv("VFFT_NO_IL2P"))
        return;
    int iR1 = 0, iR2 = 0;
    vfft_oop_wisdom_entry_t keb;
    /* the request's ORDER CELL (2026-09-05): an explicit SCRAMBLED request
     * reads the ord=scr row — the scrambled pool's own verdict — and nothing
     * else; DEFAULT and NATURAL read the ord=nat row. */
    /* the request's PLACEMENT CELL (2026-09-21): an in-place request reads
     * and races the place=ip row -- its own verdict, every arm executed
     * in place -- never the out-of-place cell's. Until now both doors asked
     * with inplace=0 and the in-place door served the out-of-place verdict
     * through a reference row (owner: wrong; one contract per request). */
    const int ip_req = (cfg->placement == VFFT_INPLACE);
    const int scr_req = (vfft_policy_ord_k1(cfg, N, ip_req) == VW2_ORD_SCR);
    const vfft_oop_wisdom_entry_t *ke =
        W->vw2_off_oop ? vfft_oop_wisdom_lookup_k1(&W->oop, N)
                       : (vw2_oop_lookup_k1_cell(&W->vw2, N, scr_req, ip_req, &keb) ? &keb : NULL);
    /* the IL plan race: a MISS (no IL verdict on the row) or recalibrate
     * below 2048 races the planner's pools and banks, then replays */
    /* ... and ABOVE 2048 for any N without a factor of 4 (2026-09-04):
     * the cascade's ingest is radix 4, so such an N has no cascade route
     * and would otherwise fall to Bluestein unraced — the Bailey tier's
     * race is the only measurement it can get. (N with a factor of 4 stayed
     * the cascade's until 2026-09-22; they race too now.) */
    /* WISDOM OR RACE (owner's law, 2026-09-09): every interleaved miss races,
     * the pow2 band included — _k1_il_plan_race carries the N gate. Until
     * 2026-09-09 this call was fenced to N < 2048 or odd N and a cold in-place
     * band cell refused with "no interleaved engine". */
    /* the scrambled pow2 band is the K=1 tier's since 2026-09-14: its writer is
     * the PLAIN ZTURN-T schedule (ztt_scrambled_design.md), raced and banked
     * on the ord=scr row like every other cell; the cascade's fence that stood
     * here is gone with the cascade's last pow2 role */
    if (!W->vw2_off_oop &&
        (cfg->recalibrate || !ke || !ke->il_kv_raced))   /* a pair-only row (forms unraced) plans too */
    {
        if (_k1_il_plan_race(W, cfg, N) > 0)
            ke = vw2_oop_lookup_k1_cell(&W->vw2, N, scr_req, ip_req, &keb) ? &keb : NULL;
    }
    /* a SCRAMBLED request at a pow2 cell with no scrambled row after the race
     * builds NOTHING here — no default pair, no heuristic (NO FALLBACKS): the
     * race is the only source of a scrambled plan, and a natural-writing pair
     * is not one. Seen 2026-09-09: the in-place scrambled create at 2048
     * attached a natural-writing pair 64.32. */
    {   /* the writer-band law (planning/policy.h): no fallback here */
        const vfft_cell_t sc = vfft_policy_cell(cfg, N, 1, 1, 0, 1);
        if (scr_req && vfft_policy_scr_writer_band(&sc) && !ke)
            return;
    }
    /* MONO verdict (2026-09-04): the cell's plan is ONE solo kernel; no pair
     * is built here — the caller serves the mono door (the OOP block reads
     * the form itself; in place, _k1_il_mono_candidate). Without this an
     * in-place create replayed a MONO row as the balanced pair. */
    if (ke && ke->k1_il_route == VFFT_K1_IL_MONO && vfft_k1_mono_il_fn(N, 0))
        return;
    /* PRIME verdict (2026-09-21): the cell's plan is the prime cell, an arm
     * of the race since today. The race's warm plan is handed over when the
     * race just ran (never rebuilt: under recalibrate a rebuild would race
     * the inner a second time and could bank a different one); a replay
     * builds it from the prime shard's row. */
    if (ke && ke->k1_il_route == VFFT_K1_IL_PRIME)
    {
        if (ilp_out)
        {
            if (_k1pr_ctx.plan && _k1pr_ctx.N == N)
            {
                *ilp_out = _k1pr_ctx.plan;
                _k1pr_ctx.plan = NULL;
                _k1pr_ctx.N = 0;
            }
            else
                *ilp_out = _ilprime_create_banked(W, cfg, N);
            if (getenv("VFFT_NAT_LOG") && *ilp_out)
                fprintf(stderr, "[k1pr] N=%d: %s prime cell (%s, M=%d) src=wisdom\n", N,
                        ip_req ? "in place" : "out of place",
                        (*ilp_out)->method ? "RADER" : "BLUESTEIN", (*ilp_out)->M);
        }
        _k1pr_release();
        return;
    }
    /* CHAIN3 verdict (2026-09-02): the banked 3-stage chain replays as
     * written; a build refusal falls through to the pair/default path */
    if (ke && ke->k1_il_route == VFFT_K1_IL_CHAIN3 && ke->il_c3[0])
    {
        *il3p_out = vfft_il3p_create(N, ke->il_c3[0], ke->il_c3[1], ke->il_c3[2]);
        if (*il3p_out)
        {
            _k1_il3p_apply_kv(*il3p_out, ke, &W->vw2, N, ip_req);   /* banked forms > default */
            if (getenv("VFFT_NAT_LOG"))
                fprintf(stderr, "[k1c3] N=%d: replay chain %d.%d.%d src=wisdom\n",
                        N, ke->il_c3[0], ke->il_c3[1], ke->il_c3[2]);
            return;
        }
    }
    /* FLAT DIT verdict (2026-09-05): the banked chain + per-stage forms
     * replay as written (validated by the engine's create/apply). Under a
     * SCRAMBLED request ke is the ord=scr row and the plan is the flat DIT's
     * scrambled class. A refusal falls through to the pair/default path. */
    if (ke && ke->k1_il_route == VFFT_K1_IL_FLAT && ke->il_fl_n >= 2 && ilfd_out)
    {
        vfft_ilfd_plan_t *fp;
        if (scr_req)
            fp = vfft_ilfd_create_scr_of(N, ke->il_fl, ke->il_fl_n, ke->il_flf, ke->il_tw);
        else
        {
            fp = vfft_ilfd_create_chain(N, ke->il_fl, ke->il_fl_n);
            if (fp && (!fp->bwd_ok || (ke->il_flf[0] && !vfft_ilfd_apply_forms(fp, ke->il_flf)) ||
                       (ke->il_tw > 0 && !vfft_ilfd_apply_tw(fp, ke->il_tw))))
            {
                vfft_ilfd_destroy(fp);
                fp = NULL;
            }
        }
        if (fp)
        {
            *ilfd_out = fp;
            if (getenv("VFFT_NAT_LOG"))
            {
                char chs[48];
                int q, off = 0;
                for (q = 0; q < ke->il_fl_n && off < (int)sizeof chs - 4; q++)
                    off += snprintf(chs + off, sizeof chs - (size_t)off, "%s%d", q ? "." : "", ke->il_fl[q]);
                fprintf(stderr, "[k1fd] N=%d: replay flat chain %s (forms %s, tw %d, %s) src=wisdom\n",
                        N, chs, ke->il_flf[0] ? ke->il_flf : "-", ke->il_tw,
                        scr_req ? "SCRAMBLED class" : "natural");
            }
            return;
        }
    }
    /* ZTURN-T verdict (2026-09-09): the banked chain replays as written
     * (validated by the create: legality, the quarter-wave's octave, the
     * registry cell). The ORDER CLASS is the row's: an ord=nat row replays
     * the natural drivers, an ord=scr row the PLAIN schedule's (2026-09-14,
     * ztt_scrambled_design.md) — one plan, one order, never mixed. A refusal
     * falls through to the pair/default path. */
    if (ke && ke->k1_il_route == VFFT_K1_IL_ZTT && ke->il_zt_n >= 2 && ztt_out)
    {
        vfft_ztt_plan_t *zp = vfft_ztt_create_chain_ord(N, ke->il_zt, ke->il_zt_n, scr_req);
        if (zp && ke->il_tw > 0 && !vfft_ztt_set_tile(zp, (size_t)ke->il_tw))
        {   /* the row names a tile the cell refuses: not a plan that exists */
            vfft_ztt_destroy(zp);
            zp = NULL;
        }
        if (zp)
        {
            *ztt_out = zp;
            if (getenv("VFFT_NAT_LOG"))
            {
                char chs[48];
                vfft_ztt_chain_str(zp, chs, sizeof chs);
                fprintf(stderr, "[k1ztt] N=%d: replay ZTURN-T chain %s tile=%zu src=wisdom\n", N, chs, zp->tile);
            }
            return;
        }
    }
    /* the FOUR-STEP (route 10, 2026-09-15): a banked verdict replays its split
     * (il_pair = N1.N2) through the create — the 2D child at the request's
     * placement and thread count, the order class the row's */
    if (ke && ke->k1_il_route == VFFT_K1_IL_FS && fs_out && ke->il_R1 > 0 && ke->il_R2 > 0 &&
        (long)ke->il_R1 * (long)ke->il_R2 == (long)N)
    {
        int pn1 = ke->il_R1, pn2 = ke->il_R2, sbc[8], sbn = 0, form = 0;
        const int pinned = _k1fs_pin(N, &pn1, &pn2);
        vfft_k1fs_plan_t *fp;
        if (!scr_req) sbn = _k1fs_row_sb(W, N, ke->il_kv, sbc, &form);
        fp = vfft_k1fs_create(N, pn1, pn2, scr_req, W, cfg,
                              cfg->placement == VFFT_INPLACE, _vfft_plan_threads(cfg), form, sbc, sbn);
        if (fp)
        {
            *fs_out = fp;
            if (getenv("VFFT_NAT_LOG"))
                fprintf(stderr, "[k1fs] N=%d: replay FOUR-STEP %dx%d form=%d src=%s (%s)\n", N, fp->N1, fp->N2,
                        fp->form, pinned ? "pin" : "wisdom", cfg->placement == VFFT_INPLACE ? "ip" : "oop");
            return;
        }
    }
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
        _k1_il2p_apply_kv(*il2p_out, ke, &W->vw2, N, ip_req);   /* wisdom verdict > default */
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
                                                      _k1ord_reseed, &ca, 1 } /* single-thread arms: paced (VFFT_RACE_PACE_MS) */;
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
                if (getenv("VFFT_NAT_LOG") || getenv("VFFT_ILPR_LOG"))
                    fprintf(stderr, "[k1ord] N=%d pair race: heuristic %d.%d=%.0f "
                                    "swapped %d.%d=%.0f -> %s\n",
                            N, iR1, iR2, ta, iR2, iR1, tb,
                            picked_swap ? "SWAPPED" : "heuristic");
                /* BANK the winner as the cell's kind-3 pair verdict (B1.4,
                 * 2026-09-02): the pair ORDER is exactly what il_pair= says,
                 * so the existing replay (ke->il_R1 above) serves it and this
                 * race never runs again for the cell. Measure-less (ns=0):
                 * the offline planner's measured row replaces it. */
                if (W && !W->vw2_off_oop && cfg)
                {
                    vfft_oop_wisdom_entry_t ne;
                    memset(&ne, 0, sizeof ne);
                    ne.N = N;
                    ne.K = 1;
                    ne.k1_il_route = VFFT_K1_IL_2P_PURE;
                    ne.il_R1 = picked_swap ? iR2 : iR1;
                    ne.il_R2 = picked_swap ? iR1 : iR2;
                    ne.ord_scr = scr_req;   /* the request's own order cell (2026-09-05) */
                    if (vw2_oop_bank_k1_lay(&W->vw2, &ne, VW2_LAY_IL) == VW2_OK)
                        _vw2_persist(W, cfg);
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
/* the ILP recipe row, AS KEYED: the kind-3 row at N (lay=il / split /
 * lay-less, exact keys) else the PRIME shard row; 0 = none (mono) */
static int _ilp_ref_of(struct vfft_wisdom_s *W, int N, int mode, int scr_req)
{
    int lay;
    if (mode != VFFT_NAT_ILP || W->vw2_off_oop) return 0;
    /* an explicit SCRAMBLED request is served from its own order cell
     * (2026-09-05): the signpost names the ord=scr kind-3 IL row */
    if (scr_req && vw2_oop_k1_row_lay_ord(&W->vw2, N, 1) == VW2_LAY_IL) return 5;
    lay = vw2_oop_k1_row_lay(&W->vw2, N);
    if (lay == VW2_LAY_IL) return 1;
    if (lay == VW2_LAY_SPLIT) return 2;
    if (lay == VW2_LAY_ANY) return 3;
    if (vw2_prime_method_lookup(&W->vw2, N)) return 4;
    return 0;
}

/* ── the FLAT DIT's threading verdict (2026-09-07, il_flatdit_mt.h; the
 * same law as the cascade's above): env pin > the banked il_mt at THIS T
 * on the cell's kind-3 IL row (ord=nat or ord=scr — the plan's own class)
 * > the race at T (serial vs blocks vs tiles at every legal width, steady-
 * state samples), banked il_mt= il_mt_t= il_mt_tw=. The one-thread width
 * il_tw= stays what a T=1 plan replays. */
static void _ilfd_mt_replay_or_race(struct vfft_plan_s *h,
                                    struct vfft_wisdom_s *W,
                                    const vfft_config_t *cfg, int N)
{
    vfft_ilfd_plan_t *p = h->k1ilfd;
    const int T = h->nthreads;
    const int tw0 = p->tw;
    const vw2_rec_t *r = NULL;
    const char *pin = getenv("VFFT_ILFD_MT");
    int mt_tw = 0;
    if (!p || T < 2)
        return;
    if (W && !W->vw2_off_oop)
        r = vw2__oop_k1_scan_ord(&W->vw2, N, VW2_LAY_IL, p->scr);
    if (pin)
    {
        const int v = atoi(pin);
        p->mt = (v >= 0 && v <= 2) ? v : 0;
        p->mt_t = T;
        p->mt_tw = tw0;
        if (p->mt > 0 && !vfft_ilfd_mt_bind(p, T))
            p->mt = 0;
        if (getenv("VFFT_NAT_LOG"))
            fprintf(stderr, "[k1fd-mt] N=%d T=%d %s: mt=%d src=env\n", N, T, p->scr ? "scr" : "nat", p->mt);
        return;
    }
    if (r && !cfg->recalibrate &&
        vfft_policy_replays_at_T(vw2__oop_geti(r, "il_mt_t", 0), T))
    {
        const int v = vw2__oop_geti(r, "il_mt", 0);
        const int w = vw2__oop_geti(r, "il_mt_tw", 0);
        p->mt = (v >= 0 && v <= 2) ? v : 0;
        p->mt_t = T;
        p->mt_tw = w;
        if (p->mt == 2 && w > 0 && !vfft_ilfd_apply_tw(p, w))
        {
            _vfft_warn("banked il_mt_tw=%d does not fit the flat chain at N=%d — serial", w, N);
            p->mt = 0;
        }
        if (p->mt > 0 && !vfft_ilfd_mt_bind(p, T))
            p->mt = 0;
        if (getenv("VFFT_NAT_LOG"))
            fprintf(stderr, "[k1fd-mt] N=%d T=%d %s: replay mt=%d tw=%d src=wisdom\n",
                    N, T, p->scr ? "scr" : "nat", p->mt, p->tw);
        return;
    }
    {   /* the race on scratch, out of place (the in-place serving is the same lists) */
        double *zi = (double *)malloc(2 * (size_t)N * sizeof(double));
        double *zo = (double *)malloc(2 * (size_t)N * sizeof(double));
        size_t i;
        if (!zi || !zo)
        {
            free(zi); free(zo);
            p->mt = 0;
            return;
        }
        for (i = 0; i < 2 * (size_t)N; i++)
            zi[i] = 1.0 + 1e-6 * (double)(i & 1023);
        vfft_ilfd_mt_race(p, T, tw0, zi, zo, &mt_tw);
        p->mt_tw = mt_tw;
        free(zi); free(zo);
    }
    if (r && !W->vw2_off_oop)
    {
        char b[16];
        int ok = 1;
        snprintf(b, sizeof b, "%d", p->mt);
        ok = ok && vw2_update_field(&W->vw2, &r->key, "il_mt", b) == VW2_OK;
        snprintf(b, sizeof b, "%d", T);
        ok = ok && vw2_update_field(&W->vw2, &r->key, "il_mt_t", b) == VW2_OK;
        snprintf(b, sizeof b, "%d", mt_tw);
        ok = ok && vw2_update_field(&W->vw2, &r->key, "il_mt_tw", b) == VW2_OK;
        if (ok)
            _vw2_persist(W, cfg);
    }
}

/* ── ZTURN-T's threading verdict (2026-09-15, ztt_mt.h, ztt_mt_design.md;
 * the flat DIT's law above): env pin VFFT_ZTT_MT=0|1|2 (never banked) > the
 * banked arm at THIS T on the cell's il_route=ztt row of the plan's own
 * order class > the race at T (serial vs blocks vs tiles, steady-state
 * samples, hot). Out of place banks il_mt= il_mt_t=; a plan bound IN PLACE
 * races aliased arms through the plane — a different measurement — and
 * banks its own pair il_mt_ip= il_mt_ip_t=. The one-thread tile il_tw= is
 * untouched: the arm sections the walk the row already names. */
static void _ztt_mt_replay_or_race(struct vfft_plan_s *h,
                                   struct vfft_wisdom_s *W,
                                   const vfft_config_t *cfg, int N)
{
    vfft_ztt_plan_t *p = h->k1ztt;
    const int T = h->nthreads;
    const int ip = (h->placement == VFFT_INPLACE);
    const char *tok_v = ip ? "il_mt_ip" : "il_mt", *tok_t = ip ? "il_mt_ip_t" : "il_mt_t";
    const vw2_rec_t *r = NULL;
    const char *pin = getenv("VFFT_ZTT_MT");
    if (!p || T < 2)
        return;
    if (W && !W->vw2_off_oop)
        r = vw2__oop_k1_scan_ord(&W->vw2, N, VW2_LAY_IL, p->scr);
    if (pin)
    {
        const int v = atoi(pin);
        if (!vfft_ztt_mt_bind(p, T, (v >= 0 && v <= 2) ? v : 0)) p->mt = 0;
        if (getenv("VFFT_NAT_LOG"))
            fprintf(stderr, "[ztt-mt] N=%d T=%d %s%s: mt=%d src=env\n", N, T, p->scr ? "scr" : "nat", ip ? " ip" : "", p->mt);
        return;
    }
    if (r && !cfg->recalibrate &&
        vfft_policy_replays_at_T(vw2__oop_geti(r, tok_t, 0), T))
    {
        const int v = vw2__oop_geti(r, tok_v, 0);
        if (!vfft_ztt_mt_bind(p, T, (v >= 0 && v <= 2) ? v : 0)) p->mt = 0;
        if (getenv("VFFT_NAT_LOG"))
            fprintf(stderr, "[ztt-mt] N=%d T=%d %s%s: replay mt=%d src=wisdom\n",
                    N, T, p->scr ? "scr" : "nat", ip ? " ip" : "", p->mt);
        return;
    }
    {   /* the race on 64-B aligned scratch, in the plan's own placement */
        const size_t nb = (size_t)2 * N * sizeof(double);
        double *zi = (double *)VFFT_ZS_ALLOC(nb);
        double *zo = (double *)VFFT_ZS_ALLOC(nb);
        size_t i;
        if (!zi || !zo) { VFFT_ZS_FREE(zi); VFFT_ZS_FREE(zo); p->mt = 0; return; }
        for (i = 0; i < 2 * (size_t)N; i++) zi[i] = 1.0 + 1e-6 * (double)(i & 1023);
        _vfft_pool_arm(T);
        if (ip) { memcpy(zo, zi, nb); vfft_ztt_mt_race(p, T, zo, zo, NULL); }
        else vfft_ztt_mt_race(p, T, zi, zo, NULL);
        VFFT_ZS_FREE(zi); VFFT_ZS_FREE(zo);
    }
    if (r && !W->vw2_off_oop)
    {
        char b[16];
        int ok = 1;
        snprintf(b, sizeof b, "%d", p->mt);
        ok = ok && vw2_update_field(&W->vw2, &r->key, tok_v, b) == VW2_OK;
        snprintf(b, sizeof b, "%d", T);
        ok = ok && vw2_update_field(&W->vw2, &r->key, tok_t, b) == VW2_OK;
        if (ok)
            _vw2_persist(W, cfg);
    }
}

/* ── the FOUR-STEP's threaded arm (2026-09-15): the SPLIT (and the natural
 * class's FORM, 2026-09-16) is the per-T verdict ──
 * The 1D race picks the split at one thread; at T > 1 the children's own
 * threaded verdicts reorder the ladder (4194304 at T=8: the serial winner
 * 1024x4096 runs 7.1 ms, 2048x2048 4.9 ms), so a plan at T races the
 * ladder's splits AT T — every child a 2D cell created at T, its threaded
 * verdict raced and banked on its own row — forward, in the plan's own
 * placement; the natural class races each split's form 0 and form 1 with
 * the residency sub-ladder of super-band chains. Banks il_mt=N1 il_mt_t=T
 * il_mtsb=<chain|0> (il_mt_ip / il_mt_ip_t / il_mtsb_ip in place) on the
 * cell's row: ZTURN-T's tokens, each route reading them as its own arm.
 * Replay rebuilds the banked plan when it differs from the row's serial
 * one. VFFT_K1_FS / VFFT_K1_FSSB (the probe pins) skip both. */
typedef struct { vfft_k1fs_plan_t *p; const double *zi; double *zo; int ip; } _k1fs_mt_ctx_t;
typedef struct { double *dst; const double *src; size_t nb; } _k1fs_mt_rst_t;
static void _k1fs_mt_arm_run(void *v)
{
    const _k1fs_mt_ctx_t *c = (const _k1fs_mt_ctx_t *)v;
    vfft_k1fs_execute(c->p, VFFT_FORWARD, c->ip ? c->zo : c->zi, c->zo);
}
static void _k1fs_mt_reseed(void *v) { _k1fs_mt_rst_t *r = (_k1fs_mt_rst_t *)v; memcpy(r->dst, r->src, r->nb); }
static int _k1fs_parse_chain(const char *v, int *out)
{
    int n = 0;
    while (v && *v && n < 8)
    {
        char *end;
        const long r = strtol(v, &end, 10);
        if (end == v || r < 2) return 0;
        out[n++] = (int)r;
        v = (*end == '.') ? end + 1 : end;
        if (*end && *end != '.') return 0;
    }
    return n;
}
static void _k1fs_chain_str(const vfft_k1fs_plan_t *p, char *b, size_t n)
{
    int k, off = 0;
    if (!p || p->form != 1) { snprintf(b, n, "0"); return; }
    for (k = 0; k < p->sbnst && off < (int)n - 4; k++)
        off += snprintf(b + off, n - (size_t)off, "%s%d", k ? "." : "", p->sbR[k]);
}
/* the natural class's serial form from the row (il_kv=1 + il_sb) or the
 * probe pin VFFT_K1_FSSB; returns the chain length (0 = form 0) */
static int _k1fs_row_sb(struct vfft_wisdom_s *W, int N, int il_kv, int *chain, int *form)
{
    const char *e = getenv("VFFT_K1_FSSB");
    *form = 0;
    if (e && *e) { const int n = _k1fs_parse_chain(e, chain); if (n >= 2) { *form = 1; return n; } }
    if (il_kv == 1 && W && !W->vw2_off_oop)
    {
        const vw2_rec_t *r = vw2__oop_k1_scan_ord(&W->vw2, N, VW2_LAY_IL, 0);
        const char *v = r ? vw2_rec_get(r, "il_sb") : NULL;
        if (v) { const int n = _k1fs_parse_chain(v, chain); if (n >= 2) { *form = 1; return n; } }
    }
    return 0;
}
static void _k1fs_mt_replay_or_race(struct vfft_plan_s *h,
                                    struct vfft_wisdom_s *W,
                                    const vfft_config_t *cfg, int N)
{
    vfft_k1fs_plan_t *p = h->k1fs;
    const int T = h->nthreads;
    const int ip = (h->placement == VFFT_INPLACE);
    const char *tok_v = ip ? "il_mt_ip" : "il_mt", *tok_t = ip ? "il_mt_ip_t" : "il_mt_t";
    const char *tok_s = ip ? "il_mtsb_ip" : "il_mtsb";
    const vw2_rec_t *r = NULL;
    int n1[8], n2[8], ns, i;
    if (!p || T < 2 || getenv("VFFT_K1_FS") || getenv("VFFT_K1_FSSB"))
        return;
    if (W && !W->vw2_off_oop)
        r = vw2__oop_k1_scan_ord(&W->vw2, N, VW2_LAY_IL, p->scr);
    ns = vfft_k1fs_splits(N, n1, n2, 8);
    if (r && !cfg->recalibrate &&
        vfft_policy_replays_at_T(vw2__oop_geti(r, tok_t, 0), T))
    {
        const int v = vw2__oop_geti(r, tok_v, 0);
        const char *sv = vw2_rec_get(r, tok_s);
        int ch[8], cn = 0, form = 0;
        char cur[48];
        if (sv && strcmp(sv, "0") != 0) { cn = _k1fs_parse_chain(sv, ch); form = cn >= 2; }
        for (i = 0; i < ns; i++) if (n1[i] == v) break;
        _k1fs_chain_str(p, cur, sizeof cur);
        if (i < ns && (v != p->N1 || form != p->form || (form && strcmp(sv, cur) != 0)))
        {
            vfft_k1fs_plan_t *q = vfft_k1fs_create(N, n1[i], n2[i], p->scr, W, cfg, ip, T, form, ch, cn);
            if (q) { vfft_k1fs_destroy(p); h->k1fs = p = q; }
        }
        if (getenv("VFFT_NAT_LOG"))
        {
            _k1fs_chain_str(p, cur, sizeof cur);
            fprintf(stderr, "[k1fs-mt] N=%d T=%d %s%s: replay split %dx%d sb=%s src=wisdom\n",
                    N, T, p->scr ? "scr" : "nat", ip ? " ip" : "", p->N1, p->N2, cur);
        }
        return;
    }
    {   /* the race: every split (x form x chain for the natural class) at T
         * on 64-B aligned scratch, the plan's placement */
        const size_t nb = (size_t)2 * N * sizeof(double);
        double *zi = (double *)VFFT_ZS_ALLOC(nb);
        double *zo = (double *)VFFT_ZS_ALLOC(nb);
        vfft_k1fs_plan_t *cand[VFFT_RACE_MAX_ARMS];
        _k1fs_mt_ctx_t cx[VFFT_RACE_MAX_ARMS];
        vfft_race_arm_t arms[VFFT_RACE_MAX_ARMS];
        char names[VFFT_RACE_MAX_ARMS][40];
        double tns[VFFT_RACE_MAX_ARMS];
        _k1fs_mt_rst_t rs;
        int na = 0, best = 0, reps, a;
        size_t k;
        if (!zi || !zo) { VFFT_ZS_FREE(zi); VFFT_ZS_FREE(zo); return; }
        for (k = 0; k < 2 * (size_t)N; k++) zi[k] = 1.0 + 1e-6 * (double)(k & 1023);
        _vfft_pool_arm(T);
        for (i = 0; i < ns && na < VFFT_RACE_MAX_ARMS; i++)
        {
            int ch[24][8], cl[24], nch = 0, c;
            /* form 0 (the current plan when it is this split and form) */
            cand[na] = (n1[i] == p->N1 && p->form == 0) ? p
                     : vfft_k1fs_create(N, n1[i], n2[i], p->scr, W, cfg, ip, T, 0, NULL, 0);
            if (cand[na])
            {
                cx[na].p = cand[na]; cx[na].zi = zi; cx[na].zo = zo; cx[na].ip = ip;
                snprintf(names[na], sizeof names[na], "%dx%d", n1[i], n2[i]);
                arms[na].name = names[na]; arms[na].run = _k1fs_mt_arm_run; arms[na].ctx = &cx[na];
                na++;
            }
            if (p->scr || !_k1fs_sb_admit(N)) continue;
            nch = _k1fs_sb_chains(n1[i], ch, cl, 24, 1);
            for (c = 0; c < nch && na < VFFT_RACE_MAX_ARMS; c++)
            {
                char cs[48];
                int q, off = 0, same = 0;
                for (q = 0; q < cl[c] && off < (int)sizeof cs - 4; q++)
                    off += snprintf(cs + off, sizeof cs - (size_t)off, "%s%d", q ? "." : "", ch[c][q]);
                if (n1[i] == p->N1 && p->form == 1 && p->sbnst == cl[c] && memcmp(p->sbR, ch[c], (size_t)cl[c] * sizeof(int)) == 0)
                    same = 1;
                cand[na] = same ? p : vfft_k1fs_create(N, n1[i], n2[i], 0, W, cfg, ip, T, 1, ch[c], cl[c]);
                if (!cand[na]) continue;
                cx[na].p = cand[na]; cx[na].zi = zi; cx[na].zo = zo; cx[na].ip = ip;
                snprintf(names[na], sizeof names[na], "%dx%d/%s", n1[i], n2[i], cs);
                arms[na].name = names[na]; arms[na].run = _k1fs_mt_arm_run; arms[na].ctx = &cx[na];
                na++;
            }
        }
        if (na == 0) { VFFT_ZS_FREE(zi); VFFT_ZS_FREE(zo); return; }
        {   /* reps from one timing of the serial verdict's plan at T */
            double t0;
            if (ip) memcpy(zo, zi, nb);
            vfft_k1fs_execute(p, VFFT_FORWARD, ip ? zo : zi, zo);
            if (ip) memcpy(zo, zi, nb);
            t0 = _il_ab_now(); vfft_k1fs_execute(p, VFFT_FORWARD, ip ? zo : zi, zo); t0 = _il_ab_now() - t0;
            reps = (int)(20e6 / (t0 > 1.0 ? t0 : 1.0));
            if (reps < 2) reps = 2;
            if (reps > 64) reps = 64;
        }
        rs.dst = zo; rs.src = zi; rs.nb = nb;
        {
            const vfft_race_proto_t proto = { 3, reps, VFFT_RACE_MIN, 1, 2, ip ? _k1fs_mt_reseed : NULL, ip ? &rs : NULL, 0 }; /* THREADED arms: never paused (mt_measurement_parking_trap) */
            vfft_race_run(&proto, arms, na, tns);
        }
        for (a = 1; a < na; a++) if (tns[a] < tns[best]) best = a;
        h->k1fs = cx[best].p;
        for (a = 0; a < na; a++) if (cand[a] && cand[a] != h->k1fs && cand[a] != p) vfft_k1fs_destroy(cand[a]);
        if (p != h->k1fs) vfft_k1fs_destroy(p);
        p = h->k1fs;
        if (getenv("VFFT_NAT_LOG"))
        {
            fprintf(stderr, "[k1fs-mt] N=%d T=%d %s%s: split race", N, T, p->scr ? "scr" : "nat", ip ? " ip" : "");
            for (a = 0; a < na; a++) fprintf(stderr, " %s=%.0f", arms[a].name, tns[a]);
            fprintf(stderr, " -> %s\n", arms[best].name);
        }
        VFFT_ZS_FREE(zi); VFFT_ZS_FREE(zo);
    }
    if (r && W && !W->vw2_off_oop)
    {
        char b[48];
        int ok = 1;
        snprintf(b, sizeof b, "%d", p->N1);
        ok = ok && vw2_update_field(&W->vw2, &r->key, tok_v, b) == VW2_OK;
        snprintf(b, sizeof b, "%d", T);
        ok = ok && vw2_update_field(&W->vw2, &r->key, tok_t, b) == VW2_OK;
        _k1fs_chain_str(p, b, sizeof b);
        ok = ok && vw2_update_field(&W->vw2, &r->key, tok_s, b) == VW2_OK;
        if (ok)
            _vw2_persist(W, cfg);
    }
}

/* ── the IN-PLACE mono candidate (2026-09-04) ──
 * Served when the cell's kind-3 row (already planned by _k1_il_candidate's
 * race on a miss) says MONO: the alias-tolerant n1c solo, both directions.
 * The row's form axis names the OOP kernel family (solo n1 vs mono64); in
 * place both forms map onto the ONE alias-safe solo, n1c(N). Returns 1 and
 * fills the pair when served, 0 otherwise (nothing built, nothing to free). */
static int _k1_il_mono_candidate(struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                                 int N, vfft_oop11_fn *ilf, vfft_oop11_fn *ilb)
{
    vfft_oop_wisdom_entry_t keb;
    const vfft_oop_wisdom_entry_t *ke;
    /* the REQUEST's order cell (2026-09-17) and, since 2026-09-21, its
     * PLACEMENT cell: this is the in-place door's candidate, so it reads the
     * place=ip row the in-place race banked (an explicit SCRAMBLED cell reads
     * its ord=scr row -- the 2026-09-17 fix for a MONO verdict refused on the
     * wrong row stands). */
    const int scr_req = (vfft_policy_ord_k1(cfg, N, 1) == VW2_ORD_SCR);
    *ilf = *ilb = 0;
    if (!W || W->vw2_off_oop) return 0;
    ke = vw2_oop_lookup_k1_cell(&W->vw2, N, scr_req, 1, &keb) ? &keb : NULL;
    if (!ke || ke->k1_il_route != VFFT_K1_IL_MONO) return 0;
    *ilf = vfft_k1_mono_ilc_fn(N, 0);
    *ilb = vfft_k1_mono_ilc_fn(N, 1);
    if (!*ilf || !*ilb) { *ilf = *ilb = 0; return 0; }
    return 1;
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
    nn.ref_comp = 0 /* no cascade recipe rows since 2026-09-15 */;
    nn.ref_ilp = _ilp_ref_of(W, N, mode, 0);   /* the @nat cell: the ord=nat recipe */
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
#endif /* VFFT_OOP_K1_COMMIT_H */
