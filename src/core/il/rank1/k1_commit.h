/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Tuğbars Heptaşkın */
/* k1_commit.h - the K=1 plan's replay, race, and commit.
 *
 * How a single-transform cell gets its plan: consult wisdom, and on a miss race
 * the candidates and bank the winner.
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
 * A REPLAY MEASURES NOTHING
 * -------------------------
 * A hit reconstructs the plan from the banked record without touching a
 * clock; only a miss (or recalibrate) races. That is what makes a warmed store
 * cheap, and what makes the harness's replay-purity assertion meaningful: a
 * cell that races during what should be a replay has the clock inside its own
 * baseline.
 *
 * _k1_il2p_apply_kv IS NOT A RACE
 * -------------------------------
 * It applies a banked kind-3 il_kv verdict - the leaf/mid kernel FORM pair -
 * and measures nothing. The race that produced that verdict lives offline in
 * dp_planner_il.h and never runs at create. An il_kv of 0 keeps create's own
 * structural default (blocked at R>=32).
 *
 * Worth knowing: il_kv has NO fingerprint field, so two plans differing only in
 * kernel form fingerprint identically and the baseline harness cannot see a
 * change here. This is one of the few places in the tree where obj_equiv is the
 * only guard.
 *
 * INCLUSION CONTRACT
 * ------------------
 * Include after the engine prelude, after vfft_internal.h, and AFTER
 * _vw2_persist - the bankers call it, and it stays in vfft.c. Same back-edge as
 * zr2c_build.h has; both would be freed by moving _vw2_persist to support/.
 */
#ifndef VFFT_IL_K1_COMMIT_H
#define VFFT_IL_K1_COMMIT_H
#include "vfft_prec.h" /* vfft_real: the element type of this unit (FP32: vfftf.c) */

#include <stdlib.h>
#include <string.h>

#include "vfft_internal.h"                  /* struct vfft_plan_s / vfft_wisdom_s */
#include "il2p.h"                           /* the Bailey pair plan + kernel resolvers */
#include "il_prime.h"                       /* the prime IL engine */
#include "il/wisdom/wisdom2_oop_il.h"      /* the kind-3 IL codecs */
#include "common/wisdom/wisdom2_oop_legacy.h" /* the kill switch's legacy line */
#include "common/support/race.h"                   /* the shared race body */

/* the CHAIN3 twin of _k1_il2p_apply_kv: the banked three-slot il_kv on the
 * chain3 row overrides the create's structural defaults; env VFFT_IL_KV /
 * VFFT_IL_BKV pin. Returns 1 when a BANKED form names a kernel this build
 * does not have (the row is empty: the door re-races the cell), else 0; an
 * env pin never counts. */
static int _k1_il3p_apply_kv(vfft_il3p_plan_t *p,
                             const vfft_oop_il_entry_t *ke,
                             const vw2_store_t *st, int N, int ip)
{
    int bad = 0;
    if (!p)
        return 0;
    if (ke && vfft_il3p_apply_kv_forms(p, ke->il_kv) != 0)
        bad = 1;
    /* the BACKWARD cell: its own dir=bwd row, validated against the chain it
     * was raced at; outside the ke guard like the pair's */
    if (st)
    {
        int c3[3] = { 0, 0, 0 };
        int bkv = vw2_oop_lookup_k1_bwd_chain_pl(st, N, c3, ip ? VW2_PL_IP : VW2_PL_OOP);
        if (bkv >= 0 && c3[0] == p->R2 && c3[1] == p->A && c3[2] == p->B &&
            vfft_il3p_apply_kv_forms_bwd(p, bkv) != 0)
            bad = 1;
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
    return bad;
}

/* Returns 1 when a BANKED form (the row's il_kv, or its matching dir=bwd
 * row's) names a kernel this build does not have -- the row is empty and the
 * door re-races the cell -- else 0. An env pin never counts. */
static int _k1_il2p_apply_kv(vfft_il2p_plan_t *p,
                             const vfft_oop_il_entry_t *ke,
                             const vw2_store_t *st, int N, int ip)
{
    int bad = 0;
    /* Wisdom variant verdict — runs AFTER create, so it OVERRIDES the
     * structural blocked default (il2p.h): a banked per-cell measurement
     * always outranks the structural rule. Nibble VFFT_IL_KV_MONO (0xF)
     * forces the monolithic kernel back — required since blocked became
     * the R>=32 default, so a platform where blocked measures slower
     * stays expressible as a verdict rather than only as an env. */
    if (!p)
        return 0;
    if (ke && vfft_il2p_apply_kv_forms(p, ke->il_kv) != 0) /* shared nibble semantics —
                                                            * one definition (il2p.h),
                                                            * planner uses the same fn */
        bad = 1;
    /* BACKWARD arm. The backward kernel-variant verdict is its
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
        if (bkv >= 0 && bR1 == p->R1 && bR2 == p->R2 &&
            vfft_il2p_apply_kv_forms_bwd(p, bkv) != 0)
            bad = 1;
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
    return bad;
}

/* the two arms of the (R1,R2) ordering race: two il2p plans on one
 * aliased buffer, re-seeded before every burst */
typedef struct { vfft_il2p_plan_t *p; vfft_real *rz, *r0; size_t nb; } _k1ord_arm_t;
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
static int _k1fs_row_sb(struct vfft_wisdom_s *W, int N, int ip, int il_kv, int *chain, int *form); /* below, with the threaded arm */
static int _k1_il_candidate(struct vfft_wisdom_s *W, const vfft_cfg_t *cfg,
                             int N, vfft_il2p_plan_t **il2p_out,
                             vfft_il3p_plan_t **il3p_out,
                             vfft_ilfd_plan_t **ilfd_out,
                             vfft_ztt_plan_t **ztt_out,
                             vfft_k1fs_plan_t **fs_out,
                             vfft_ilprime_plan_t **ilp_out); /* defined below */

/* ── THE PRIME CELL'S INNER, RACED ──
 * A prime N is a convolution done with an FFT of length M (Rader: N - 1;
 * Bluestein: any even M >= 2N - 1 -- the next power of two and the smallest
 * smooth sizes, vfft_policy_ilprime_blue_m). The prime cell RACES its
 * inner rather than borrowing M's own verdict (M standalone is not M inside
 * a convolution, and racing M's cell nested inside N's create is what
 * _k1_il_dp_busy forbids): every buildable (method, inner) pair, built
 * directly from a descriptor (no planner call, so the lock is never
 * touched), timed on the WHOLE convolution, the winner banked on the prime
 * cell's OWN row and replayed from there. */
typedef struct {
    int kind;                       /* 1 = il2p pair, 2 = il3p chain, 3 = ZTURN-T, 4 = the four-step at M (2026-10-12),
                                     * 5 = the flat DIT at M with a Rader stage (2026-10-13) */
    int R1, R2;                     /* 1; 4: the split N1 x N2 */
    int fl[VFFT_ILFD_MAX_K], fln;   /* 5: the flat chain, its prime above the pool a Rader stage */
    int cR2, cA, cB;                /* 2 */
    int zt[VFFT_ZTT_MAX_NF], ztn;   /* 3: the chain */
    int tw;                         /* 3: the tile, 0 = untiled */
    int failed;                     /* THIS descriptor could not build: the create's
                                     * structural fallback then served, and the plan
                                     * must be discarded -- never a silent substitute */
    /* kind 4, THE FOUR-STEP INNER (field trial items 1/13): its 2D child is
     * a raced plan of its own (k1_fourstep.h, THE CHILD'S OWN STORE). In the
     * race every candidate's child builds on ONE private store (S, the K=1
     * race's own recipe: the row cells warmed into it first); the winner's
     * child rides on the cell's row under fs_ and a replay seeds a private
     * store from that row (row). An own build with neither (grew) raced the
     * child in the plan's private store and says so. */
    const vw2_rec_t *row;           /* 4: the banked row carrying the child's fs_ tokens, or NULL */
    struct vfft_wisdom_s *S;        /* 4: the race's private store, or NULL (an own store) */
    struct vfft_wisdom_s *seed;     /* 4: a seeded private store the plan takes over (a composite's replay); consumed */
    int grew;                       /* 4: the child did not replay from `row` / `seed` */
    int T;                          /* 4: the thread count the four-step is built at (0 or 1: one thread). A c2c cell at
                                     * T > 1 builds it at T, its child's rows keyed at T: the four-step's threaded form
                                     * (2026-11-18; one thread, 1000003 and 4000037 ran 1.0x at every T) */
} _ilprime_inner_desc_t;

static void _ilprime_desc_str(const _ilprime_inner_desc_t *d, char *kind, size_t ksz, char *shape, size_t ssz)
{
    if (d->kind == 1) { snprintf(kind, ksz, "2p"); snprintf(shape, ssz, "%d.%d", d->R1, d->R2); }
    else if (d->kind == 2) { snprintf(kind, ksz, "3p"); snprintf(shape, ssz, "%d.%d.%d", d->cR2, d->cA, d->cB); }
    else if (d->kind == 4) { snprintf(kind, ksz, "fs"); snprintf(shape, ssz, "%d.%d", d->R1, d->R2); }
    else if (d->kind == 5)
    {
        int q, off = 0;
        snprintf(kind, ksz, "fd");
        shape[0] = 0;
        for (q = 0; q < d->fln && off < (int)ssz - 4; q++)
            off += snprintf(shape + off, ssz - (size_t)off, "%s%d", q ? "." : "", d->fl[q]);
    }
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
    if (!strcmp(kind, "fs"))
        return sscanf(shape, "%d.%d", &d->R1, &d->R2) == 2 && d->R1 >= 4 && d->R2 >= 4 && (d->kind = 4);
    if (!strcmp(kind, "fd"))
    {
        const char *p = shape;
        while (*p && d->fln < VFFT_ILFD_MAX_K)
        {
            char *end;
            long r = strtol(p, &end, 10);
            if (end == p || r < 2) return 0;
            d->fl[d->fln++] = (int)r;
            p = (*end == '.') ? end + 1 : end;
            if (*end != '.' && *end != '\0') return 0;
        }
        return d->fln >= 2 && (d->kind = 5);
    }
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

/* THE FOUR-STEP INNER at M (kind 4): the SCRAMBLED class, out of place, at
 * d->T threads (one by default), form 0, on the race's shared private store
 * (d->S: the plan does not own it) or on an own store seeded from the banked
 * row's fs_ tokens (_k1fs_build_own; with no row, or stale tokens, the child
 * races there and d->grew says so). The request the child sees is the inner's
 * own: c2c, MEASURE, nothing persisted. */
static vfft_k1fs_plan_t *_ilprime_fs_inner_build(int M, _ilprime_inner_desc_t *d)
{
    vfft_cfg_t cc;
    int grew = 0;
    const int T = d->T > 1 ? d->T : 1;
    vfft_k1fs_plan_t *p;
    if ((long)d->R1 * (long)d->R2 != (long)M)
        return NULL;
    memset(&cc, 0, sizeof cc);
    cc.transform = VFFT_C2C; cc.placement = VFFT_OUTOFPLACE; cc.rigor = VFFT_MEASURE;
    cc.dims = 1; cc.n[0] = M; cc.howmany = 1; cc.order = VFI_ORDER_SCRAMBLED;
    cc.layout = VFI_LAYOUT_IL; cc.nthreads = T;
    if (d->S)
    {
        cc.wisdom = (vfft_wisdom *)d->S;
        return vfft_k1fs_create(M, d->R1, d->R2, 1, d->S, &cc, 0, T, 0, NULL, 0);
    }
    if (d->seed)
    {   /* a composite's replay (zr2c): the child's rows seeded by the caller; the plan owns the store from here */
        cc.wisdom = (vfft_wisdom *)d->seed;
        p = vfft_k1fs_create(M, d->R1, d->R2, 1, d->seed, &cc, 0, T, 0, NULL, 0);
        if (p)
        {
            p->own = d->seed;
            d->grew = vfft_child_store_raced(d->seed);
            d->seed = NULL;
        }
        return p;
    }
    p = _k1fs_build_own(M, d->R1, d->R2, 1, &cc, 0, T, 0, NULL, 0, d->row, &grew);
    d->grew = (p != NULL) && (grew || !d->row);
    return p;
}
/* the provider: exactly ONE inner, from a descriptor -- never a guess */
static int _ilprime_inner_from_desc(int M, _ilprime_inner_t *in, void *v)
{
    _ilprime_inner_desc_t *d = (_ilprime_inner_desc_t *)v;
    d->failed = 0;
    if (d->kind == 4)
    {
        in->fs = _ilprime_fs_inner_build(M, d);
        in->fs_t = d->T > 1 ? d->T : 1;
        if (in->fs) return 1;
    }
    else if (d->kind == 5)
    {   /* the flat chain at M, natural class, both directions; the Rader stage's own
         * inner is the engine's pick (il_rader_stage.h) */
        vfft_ilfd_plan_t *fp = vfft_ilfd_create_chain(M, d->fl, d->fln);
        if (fp && !fp->bwd_ok) { vfft_ilfd_destroy(fp); fp = NULL; }
        if (fp) { in->fd = fp; return 1; }
    }
    else if (d->kind == 1)
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
/* THE FLAT CHAINS WITH A RADER STAGE at M (2026-10-13): where M carries exactly
 * one prime factor above the chain pool, every radix set of M over the flat
 * pool with that prime fixed first (after the lone 2 where one leads: then it
 * is the second, a Rader stage with pre-twiddles), each set in the chain
 * search's two orders. This is what a Rader inner at N - 1 = 2 . 3 . 683 is. */
static int _ilprime_fd_chains(int M, _ilprime_inner_desc_t *out, int max)
{
    vfft_chain_engine_t e;
    vfft_chain_t *sets;
    int cur[VFFT_CHAIN_MAX_K], ord[32], ns = 0, over = 0, i, j, n = 0, big = 0, nbig = 0, m = M, q = 2, np;
    while ((long)q * q <= m) { if (m % q == 0) { if (q > VFFT_ILRD_POOL_MAX) { big = q; nbig++; } while (m % q == 0) m /= q; } q++; }
    if (m > VFFT_ILRD_POOL_MAX) { big = m; nbig++; }
    if (nbig != 1 || (M / big) % big == 0 || max <= 0) return 0;
    memset(&e, 0, sizeof e);
    e.pool = _il_dp_flat_pool;
    np = (int)(sizeof _il_dp_flat_pool / sizeof _il_dp_flat_pool[0]);
    e.npool = np;
    for (i = 0; i < np; i++) ord[i] = e.pool[i];
    for (i = 1; i < np; i++)
        for (j = i; j > 0 && ord[j] < ord[j - 1]; j--) { const int t = ord[j]; ord[j] = ord[j - 1]; ord[j - 1] = t; }
    sets = (vfft_chain_t *)malloc(sizeof(vfft_chain_t) * VFFT_CHAIN_SETS_MAX);
    if (!sets) return 0;
    memset(cur, 0, sizeof cur);
    {
        const int Mr = M / big;
        if ((Mr & 1) == 0 && ((Mr >> 1) & 1))
        {
            cur[0] = 2; cur[1] = big;
            _vfft_chain_sets_rec(&e, Mr >> 1, 0, 2, cur, 2, ord, sets, &ns, &over);
        }
        else
        {
            cur[0] = big;
            _vfft_chain_sets_rec(&e, Mr, 0, 1, cur, 1, ord, sets, &ns, &over);
        }
    }
    for (i = 0; i < ns && n < max; i++)
    {
        vfft_chain_t c = sets[i];
        vfft_chain_sortval(&c);
        memset(&out[n], 0, sizeof out[n]);
        out[n].kind = 5; memcpy(out[n].fl, c.R, sizeof(int) * (size_t)c.n); out[n].fln = c.n; n++;
        if (n < max && vfft_chain_second_first(&c))
        {
            memset(&out[n], 0, sizeof out[n]);
            out[n].kind = 5; memcpy(out[n].fl, c.R, sizeof(int) * (size_t)c.n); out[n].fln = c.n; n++;
        }
    }
    free(sets);
    return n;
}

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
    /* ZTURN-T from the K=1 PLANNER'S OWN enumerators (never a second copy of
     * the registry walk and the tile ladder). Two grammars, each with its own
     * ladder: the pow2 registry, and the 2^a*odd chain grammar. The second one
     * is the point -- Rader's inner sits at M = N - 1, which is a power of two
     * only at a Fermat-shaped prime, so without the odd grammar Rader offers
     * NOTHING at 8191, 12289 and 40961 and the cell banks Bluestein unopposed.
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
    if (!getenv("VFFT_RD_ONE_LEVEL"))   /* the test of 2026-10-17: no second Rader level under the prime cell */
        n += _ilprime_fd_chains(M, out + n, max - n);   /* the flat chains with a Rader stage (2026-10-13) */
    if (vfft_k1fs_band_t(M, 1))
    {   /* THE FOUR-STEP's splits at M (2026-10-12): the lengths past ZTURN-T's
         * ceiling, pow2 and not -- the engine's own ladder, each split an arm
         * (a Rader inner N - 1 with a prime factor past the chain radices takes
         * the prime sides since 2026-11-20: 8388616 = 136 x 61681 for 8388617) */
        int n1[8], n2[8], ns = vfft_k1fs_splits(M, n1, n2, 8), i;
        for (i = 0; i < ns; i++)
        {
            if (n >= max) { dropped++; continue; }
            memset(&out[n], 0, sizeof out[n]);
            out[n].kind = 4; out[n].R1 = n1[i]; out[n].R2 = n2[i]; n++;
        }
    }
    if (dropped)   /* the no-silent-caps law */
        _vfft_warn("ilprime inner pool capped at %d (%d candidate(s) dropped) at M=%d", max, dropped, M);
    return n;
}

/* the length an inner descriptor transforms: the product of its shape (Bluestein's M, which the
 * banked shape carries, so a row replays at the size it was raced at) */
static int _ilprime_desc_len(const _ilprime_inner_desc_t *d)
{
    int q, m = 1;
    if (d->kind == 1 || d->kind == 4) return d->R1 * d->R2;   /* the pair, the four-step's split */
    if (d->kind == 2) return d->cR2 * d->cA * d->cB;
    if (d->kind == 5) { for (q = 0; q < d->fln; q++) m *= d->fl[q]; return d->fln ? m : 0; }
    for (q = 0; q < d->ztn; q++) m *= d->zt[q];
    return d->ztn ? m : 0;
}
/* build the prime plan for one (method, inner); a descriptor that cannot
 * build discards the plan the structural fallback would have served */
static vfft_ilprime_plan_t *_ilprime_build_with(int N, int rader, _ilprime_inner_desc_t *d)
{
    vfft_ilprime_plan_t *p;
    _ilprime_inner_provider = _ilprime_inner_from_desc;
    _ilprime_inner_provider_ctx = d;
    p = rader ? _ilprime_create_rader(N) : _ilprime_create_bluestein_m(N, _ilprime_desc_len(d));
    _ilprime_inner_provider = 0;
    _ilprime_inner_provider_ctx = 0;
    if (p && d->failed) { vfft_ilprime_destroy(p); p = 0; }
    return p;
}
typedef struct { vfft_ilprime_plan_t *p; vfft_real *zi, *zo; } _ilprime_iarm_t;
static void _ilprime_iarm_run(void *v)
{
    _ilprime_iarm_t *a = (_ilprime_iarm_t *)v;
    if (a->p->mt > 0 && vfft_ilprime_execute_mt(a->p, a->zi, a->zo, 0))   /* a plan the race at T bound threaded */
        return;
    if (a->p->method == 1) _ilprime_exec_rader(a->p, a->zi, a->zo, 0);
    else _ilprime_exec_bluestein(a->p, a->zi, a->zo, 0);
}
/* one same-run race over np live plans: the index of the fastest. T > 1: the plans run as bound at T (the race
 * at T binds each candidate's threaded form first, vfft_ilprime_mt_race), unpaced on the armed pool */
static int _ilprime_race_plans(vfft_ilprime_plan_t **plans, int np, vfft_real *zi, vfft_real *zo, int T)
{
    _ilprime_iarm_t ctx[VFFT_RACE_MAX_ARMS];
    vfft_race_arm_t arms[VFFT_RACE_MAX_ARMS];
    double ns[VFFT_RACE_MAX_ARMS];
    vfft_race_proto_t proto = { 3, 1, VFFT_RACE_MIN, 1, 1, NULL, NULL, 1 };   /* min-of-3, alternated, one warm pass */ /* single-thread arms: paced (VFFT_RACE_PACE_MS) */
    int a, w = 0;
    if (np <= 1) return 0;
    if (T > 1)
    {   /* threaded arms: no pause (it parks the team) */
        proto.pace = 0;
        _vfft_pool_arm(T);
    }
    for (a = 0; a < np; a++)
    {
        ctx[a].p = plans[a]; ctx[a].zi = zi; ctx[a].zo = zo;
        arms[a].name = plans[a]->method == 1 ? "rader" : "bluestein";
        arms[a].run = _ilprime_iarm_run;
        arms[a].ctx = &ctx[a];
    }
    {   /* THE SAMPLE'S REPS: vfft_now_ns reads a 100 ns counter on Windows, so one call of a short
         * convolution (0.5-5 us) is a few ticks; a sample is ~0.3 ms of executes from one timed call,
         * as the 2D races size theirs (_il2d_race_reps). The arms run zi -> zo: zi never changes. */
        double t0, est;
        _ilprime_iarm_run(&ctx[0]);
        t0 = vfft_now_ns();
        _ilprime_iarm_run(&ctx[0]);
        est = vfft_now_ns() - t0;
        proto.reps = (int)(3.0e5 / (est > 1.0 ? est : 1.0));
        if (proto.reps > 4096) proto.reps = 4096;
        if (proto.reps < 1) proto.reps = 1;
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
static void _k1_il_fs_warm(struct vfft_wisdom_s *S, const struct vfft_wisdom_s *W, const vfft_cfg_t *cfg, int N);   /* below: the four-step's row cells into a store */

/* the prime cell, banked: replay its OWN verdict (method + inner), else
 * race every buildable (method, inner) pair on the whole convolution and
 * bank the winner. The store off (W NULL or vw2_off_oop) races and banks
 * nothing; the env pin VFFT_ILPR_METHOD=rader|blue races only that method's
 * pool and never replays or banks. Nothing buildable: NULL, never a
 * structural default. `won` (may be NULL) receives the served inner's
 * descriptor, replayed or raced: a composite that banks the prime cell on
 * its own row (zr2c) records it from there. T = the plan's thread count:
 * at T > 1 the verdict is the one at T (below). */
/* THE VERDICT AT T (2026-11-18, the threading census; owner: race the inners at the plan's T, bank on its row at T):
 * a cell created at T > 1 races every (method, inner) at T -- each candidate bound to its threaded form first
 * (vfft_ilprime_mt_race: serial, blocks or tiles), then timed against the others as bound -- since an inner with no
 * threaded form (the four-step, a pair) can win on one thread and lose several times over at T (r2c 205762 at T = 4:
 * the inner four-step 0.91x, the ZTURN-T inner at T = 2 and 8 1.9x and 5.9x). Its verdict replays from the cell's
 * own row at T (il_prime, il_prime_in, il_prime_sh, il_prime_tw, a four-step inner's child as fs_*: the real row's
 * tokens, wisdom2_real_il.h); the prime shard's row keeps the one-thread verdict. A race at T banks through
 * _ilpr_mt_replay_or_race (the row at T exists once the cell is built): the verdict waits here for it. */
static struct
{
    const vfft_ilprime_plan_t *p;   /* the plan it was raced for (a plan the route race drops leaves it stale) */
    int N, T, method;               /* method 1 = Rader, 2 = Bluestein */
    _ilprime_inner_desc_t d;
    struct vfft_wisdom_s *S;        /* a four-step inner's child store (a copy of the race's), or NULL */
} _ilpr_tv;
static void _ilpr_tv_clear(void)
{
    _vfft_wisdom_free((vfft_wisdom *)_ilpr_tv.S);
    memset(&_ilpr_tv, 0, sizeof _ilpr_tv);
}
static vfft_ilprime_plan_t *_ilprime_create_banked(struct vfft_wisdom_s *W,
                                                   const vfft_cfg_t *cfg,
                                                   int N, _ilprime_inner_desc_t *won, int T)
{
    int hint = 0;
    const int store = W && !W->vw2_off_oop;
    const char *pin = getenv("VFFT_ILPR_METHOD");
    const int pin_m = (pin && pin[0] == 'r') ? 1 : (pin && pin[0] == 'b') ? 2 : 0;   /* 1 rader, 2 bluestein */
    const int bank = store && !pin_m;
    if (T < 1) T = 1;
    /* COMPOSITES COME IN TOO: Bluestein needs no primality, and the raced
     * pool reaches as far as the ZTURN-T grammars do. The Rader ARM is
     * excluded below. */
    if (bank && !cfg->recalibrate && T > 1)
    {   /* the verdict at T, on the cell's own row at T; none: the race at T below (never the one-thread verdict) */
        const int ip = (cfg->placement == VFFT_INPLACE);
        const int scr = (vfft_policy_ord_k1(cfg, N, ip) == VW2_ORD_SCR);
        const vw2_rec_t *r = vw2__oop_k1_scan_pl(&W->vw2, N, VW2_LAY_IL, scr, ip ? VW2_PL_IP : VW2_PL_OOP, T);
        const char *pm = r ? vw2_rec_get(r, "il_prime") : NULL;
        const char *pk = r ? vw2_rec_get(r, "il_prime_in") : NULL;
        const char *ps = r ? vw2_rec_get(r, "il_prime_sh") : NULL;
        _ilprime_inner_desc_t d;
        if (pm && pk && ps && _ilprime_desc_parse(&d, pk, ps, vw2__oop_geti(r, "il_prime_tw", 0)))
        {
            vfft_ilprime_plan_t *p;
            d.row = r;   /* a four-step inner's child replays from this row's fs_ tokens */
            if (d.kind == 4)
                d.T = T;   /* built at T, its child's rows keyed at T (the race at T below) */
            p = _ilprime_build_with(N, !strcmp(pm, "rader"), &d);
            if (p)
            {
                if (d.kind == 4 && d.grew && p->inner.fs && p->inner.fs->own)
                {   /* the child did not replay (a row banked before the four-step inner was built at T carries its
                     * child at one thread): it raced at T in the plan's private store; the row takes it now */
                    const vw2_key_t key = r->key;
                    const vw2_rec_t *r2 = vw2_lookup(&W->vw2, &key);
                    if (r2 && _k1fs_row_put_child(&W->vw2, r2, p->inner.fs->own, d.R1, d.R2, 0, T) == VW2_OK)
                        _vw2_persist(W, cfg);
                }
                if (getenv("VFFT_ILPR_LOG"))
                    fprintf(stderr, "[ilprime] N=%d T=%d: replay %s inner %s %s tw=%d src=wisdom (the row at T)\n",
                            N, T, p->method == 1 ? "RADER" : "BLUESTEIN", pk, ps, d.tw);
                d.row = NULL;
                if (won) *won = d;
                return p;
            }
        }
    }
    if (bank && !cfg->recalibrate && T <= 1)
    {
        char kind[8], shape[64];
        int tw = 0;
        hint = vw2_prime_method_lookup(&W->vw2, N);
        if (hint && vw2_prime_inner_lookup(&W->vw2, N, kind, sizeof kind, shape, sizeof shape, &tw))
        {
            _ilprime_inner_desc_t d;
            if (_ilprime_desc_parse(&d, kind, shape, tw))
            {
                vfft_ilprime_plan_t *p;
                vw2_key_t pk;
                vw2__prime_method_key(N, &pk);
                d.row = vw2_lookup(&W->vw2, &pk);   /* a four-step inner's child replays from this row's fs_ tokens */
                p = _ilprime_build_with(N, hint == 1, &d);
                if (p)
                {
                    if (d.kind == 4 && d.grew && p->inner.fs && p->inner.fs->own)
                    {   /* the child did not replay: it raced in the plan's private store; the row takes it now */
                        const vw2_rec_t *r = vw2_lookup(&W->vw2, &pk);
                        fprintf(stderr, "vfft: the prime cell's four-step inner at N=%d (M=%d, %dx%d) did not replay -- "
                                        "its child raced in the plan's private store\n", N, p->M, d.R1, d.R2);
                        if (r && _k1fs_row_put_child(&W->vw2, r, p->inner.fs->own, d.R1, d.R2, 0, 1) == VW2_OK)
                            _vw2_persist(W, cfg);
                    }
                    if (getenv("VFFT_ILPR_LOG"))
                        fprintf(stderr, "[ilprime] N=%d: replay %s inner %s %s tw=%d src=wisdom\n",
                                N, hint == 1 ? "RADER" : "BLUESTEIN", kind, shape, tw);
                    d.row = NULL;
                    if (won) *won = d;
                    return p;
                }
            }
            /* a banked inner that no longer builds: race below */
        }
    }
    {   /* THE RACE */
        static _ilprime_cand_t cands[_ILPR_MAX_CANDS];   /* off the stack */
        vfft_ilprime_plan_t *fin[_ILPR_MAX_CANDS / _ILPR_HEAT + 1];
        int fin_ci[_ILPR_MAX_CANDS / _ILPR_HEAT + 1];
        int nc = 0, nfin = 0, mi, h, w, wci, nbuilt = 0, hs = _ILPR_HEAT;
        int ncm[2] = { 0, 0 }, builtm[2] = { 0, 0 };   /* [0] = rader, [1] = bluestein */
        vfft_real *zi, *zo;
        struct vfft_wisdom_s *pS = NULL;   /* the four-step inners' race store */
        vfft_cfg_t pcfg;
        for (mi = 0; mi < 2; mi++)
        {
            const int rader = (mi == 0);
            /* Rader is prime-only (see _ilprime_create_rader): skip the arm
             * rather than let it offer inners that can never build, which
             * would also trip the built-none warning on every composite. */
            if (rader && !vfft_is_prime(N)) continue;
            if (pin_m && pin_m != (rader ? 1 : 2)) continue;   /* the env pin: that method's pool only */
            static _ilprime_inner_desc_t pool[_ILPR_MAX_CANDS];   /* off the stack */
            int Ms[1 + VFFT_POLICY_ILPRIME_SMOOTH_M], nM, k, n, q;
            /* The banked method is NOT a filter here: reaching this point
             * means the row could not be replayed, so its method is a
             * preference, not a measurement of what this build can do.
             * A verdict that cannot be built is a miss; a miss races. */
            if (rader) { Ms[0] = N - 1; nM = 1; }
            else nM = vfft_policy_ilprime_blue_m(N, Ms, 1 + VFFT_POLICY_ILPRIME_SMOOTH_M);   /* the pow2 and the smallest smooth sizes */
            for (k = 0; k < nM; k++)
            {
                n = _ilprime_inner_cands(Ms[k], pool, _ILPR_MAX_CANDS - nc);
                ncm[mi] += n;   /* this method's pool: zero means it never raced */
                for (q = 0; q < n; q++) { cands[nc].rader = rader; cands[nc].d = pool[q]; nc++; }
            }
        }
        if (nc == 0)
            return 0;   /* nothing to build: nothing to serve, nothing to fall back to */
        {   /* THE FOUR-STEP INNERS' CHILDREN race on ONE private store (the K=1
             * race's own recipe, k1_fourstep.h): the row cells at every split's
             * N2 warmed into it first, each candidate's 2D child raced there
             * once; the winner's child is banked on the prime row below. The
             * heat's size follows the plans' memory: a four-step inner at M
             * holds ~64 B per point (the convolution's four M-vectors and the
             * child's tables), a heat at most ~1 GB. */
            long maxM = 0;
            int q;
            for (q = 0; q < nc; q++)
                if (cands[q].d.kind == 4)
                {
                    const long m = (long)cands[q].d.R1 * (long)cands[q].d.R2;
                    int seen = 0, j;
                    if (m > maxM) maxM = m;
                    if (!pS && (pS = _k1fs_store_new()) != NULL)
                    {
                        pcfg = *cfg; pcfg.wisdom = (vfft_wisdom *)pS; pcfg.wisdom_write = 0; pcfg.recalibrate = 0;
                    }
                    for (j = 0; j < q; j++)
                        if (cands[j].d.kind == 4 && (long)cands[j].d.R1 * (long)cands[j].d.R2 == m) seen = 1;
                    if (pS && !seen) _k1_il_fs_warm(pS, W, &pcfg, (int)m);
                    cands[q].d.S = pS;
                    if (T > 1)
                        cands[q].d.T = T;   /* its threaded form: built at T, its child's rows keyed at T -- the c2c
                                             * cell's row at T and a composite's in-role race alike (the real row at T
                                             * carries the child at T, zr2c_build.h _zr2c_child_of_kid; 2026-11-19) */
                }
            if (maxM > 0)
            {
                hs = (int)((1L << 30) / (64L * maxM));
                if (hs < 2) hs = 2;
                if (hs > _ILPR_HEAT) hs = _ILPR_HEAT;
            }
        }
        zi = _ilprime_alloc((size_t)2 * N);
        zo = _ilprime_alloc((size_t)2 * N);
        if (!zi || !zo) { vfft_aligned_free(zi); vfft_aligned_free(zo); _vfft_wisdom_free((vfft_wisdom *)pS); return 0; }
        for (h = 0; h < 2 * N; h++) zi[h] = 1.0 + 1e-6 * (double)(h & 255);
        if (T > 1)
            _vfft_pool_arm(T);   /* the race at T: every candidate's threaded form on the team */
        /* HEATS */
        for (h = 0; h < nc; h += hs)
        {
            vfft_ilprime_plan_t *plans[_ILPR_HEAT];
            int ci[_ILPR_HEAT], np = 0, q, a;
            if (nfin >= (int)(sizeof fin / sizeof fin[0]))
            {   /* more heats than the final holds (the no-silent-caps law) */
                _vfft_warn("ilprime N=%d: %d candidate(s) past heat %d not raced (the final holds %d heat winners)",
                           N, nc - h, h / hs, (int)(sizeof fin / sizeof fin[0]));
                break;
            }
            for (q = h; q < nc && q < h + hs; q++)
            {
                vfft_ilprime_plan_t *p = _ilprime_build_with(N, cands[q].rader, &cands[q].d);
                if (!p) continue;
                if (T > 1)
                    vfft_ilprime_mt_race(p, T, NULL);   /* bound to its threaded form at T (serial where it has none) */
                plans[np] = p; ci[np] = q; np++;
                builtm[cands[q].rader ? 0 : 1]++;
            }
            if (np == 0) continue;
            nbuilt += np;
            w = _ilprime_race_plans(plans, np, zi, zo, T);
            for (a = 0; a < np; a++)
                if (a != w) vfft_ilprime_destroy(plans[a]);
            fin[nfin] = plans[w]; fin_ci[nfin] = ci[w]; nfin++;
        }
        if (nfin == 0) { vfft_aligned_free(zi); vfft_aligned_free(zo); _vfft_wisdom_free((vfft_wisdom *)pS); return 0; }
        /* THE FINAL */
        w = _ilprime_race_plans(fin, nfin, zi, zo, T);
        vfft_aligned_free(zi); vfft_aligned_free(zo);
        for (h = 0; h < nfin; h++)
            if (h != w) vfft_ilprime_destroy(fin[h]);
        wci = fin_ci[w];
        for (h = 0; h < nc; h++) { cands[h].d.S = NULL; cands[h].d.row = NULL; }   /* the race's store dies here: no descriptor keeps it */
        if (cands[wci].d.kind == 4 && pS && fin[w]->inner.fs && !fin[w]->inner.fs->own && !bank)
            fin[w]->inner.fs->own = _k1fs_store_dup(pS);   /* unbanked (the store off, a pin, a composite's in-role race): the plan keeps its child's store */
        {
            char kind[8], shape[64];
            _ilprime_desc_str(&cands[wci].d, kind, sizeof kind, shape, sizeof shape);
            /* Rader's inner sits at M = N - 1; where none of its inners
             * builds, Rader's arms vanish and the cell banks "bluestein" as
             * though it had won a race it was alone in. Say so. */
            if (ncm[0] > 0 && builtm[0] == 0)
                _vfft_warn("ilprime N=%d: RADER offered %d inner(s) at M=%d and built NONE -- "
                           "the banked verdict is Bluestein BY DEFAULT, not by race",
                           N, ncm[0], N - 1);
            if (ncm[1] > 0 && builtm[1] == 0)
                _vfft_warn("ilprime N=%d: BLUESTEIN offered %d inner(s) and built NONE",
                           N, ncm[1]);
            if (getenv("VFFT_ILPR_LOG"))
                fprintf(stderr, "[ilprime] N=%d T=%d: inner race %d arm(s) in %d heat(s) "
                                "[rader %d/%d, blue %d/%d] -> %s inner %s %s tw=%d mt=%d, %s\n",
                        N, T, nbuilt, nfin, builtm[0], ncm[0], builtm[1], ncm[1],
                        fin[w]->method == 1 ? "RADER" : "BLUESTEIN", kind, shape, cands[wci].d.tw, fin[w]->mt,
                        !bank ? "not banked" : T > 1 ? "banked on the row at T" : "banked");
            if (bank && T > 1)
            {   /* the verdict at T waits for the cell's row at T (_ilpr_mt_replay_or_race banks it with the arm) */
                _ilpr_tv_clear();
                _ilpr_tv.p = fin[w];
                _ilpr_tv.N = N;
                _ilpr_tv.T = T;
                _ilpr_tv.method = fin[w]->method == 1 ? 1 : 2;
                _ilpr_tv.d = cands[wci].d;
                if (cands[wci].d.kind == 4 && pS)
                    _ilpr_tv.S = _k1fs_store_dup(pS);   /* the four-step inner's child, for the row's fs_ tokens */
            }
            else if (bank && vw2_prime_method_bank(&W->vw2, N, fin[w]->method == 1 ? 1 : 2,
                                                   kind, shape, cands[wci].d.tw) == VW2_OK)
            {
                if (cands[wci].d.kind == 4 && pS)
                {   /* the four-step inner's child onto the prime row (fs_ tokens): a replay seeds it from there */
                    vw2_key_t pk;
                    const vw2_rec_t *r;
                    vw2__prime_method_key(N, &pk);
                    r = vw2_lookup(&W->vw2, &pk);
                    if (!r || _k1fs_row_put_child(&W->vw2, r, pS, cands[wci].d.R1, cands[wci].d.R2, 0, 1) != VW2_OK)
                        fprintf(stderr, "vfft: the prime cell's four-step child at N=%d NOT banked -- the inner's child will race again\n", N);
                }
                _vw2_persist(W, cfg);
            }
        }
        _vfft_wisdom_free((vfft_wisdom *)pS);
        if (won) *won = cands[wci].d;
        return fin[w];
    }
}

/* the four-step's row cells into the race's private store S BEFORE a race
 * that holds the planner (a nested race refuses, _k1_il_dp_busy): under
 * VFFT_HIGHN_PLAN=borrow (policy_il.h) the store W's banked 1D rows at each
 * split's N2, read only; a row W lacks, and every row under =scratch, races
 * as a plain 1D cell into S. The candidates' 2D children take their row
 * plans from those rows (_k1fs_child_row_store, k1_fourstep.h), so they
 * replay instead of running the door's default. Every race whose
 * pool can hold the four-step at N calls it: the K=1 race below, the prime
 * cell's inner race and the real Bluestein's. */
static void _k1_il_fs_warm_t(struct vfft_wisdom_s *S, const struct vfft_wisdom_s *W, const vfft_cfg_t *cfg, int N,
                             int T);
static void _k1_il_fs_warm(struct vfft_wisdom_s *S, const struct vfft_wisdom_s *W, const vfft_cfg_t *cfg, int N)
{
    _k1_il_fs_warm_t(S, W, cfg, N, 1);
}
/* the same for the splits a plan at T races (the band at T: the prime sides below VFFT_K1FS_MIN_N are a threaded
 * plan's only, 2026-11-20); the row cells themselves are one-thread cells either way */
static void _k1_il_fs_warm_t(struct vfft_wisdom_s *S, const struct vfft_wisdom_s *W, const vfft_cfg_t *cfg, int N,
                             int T)
{
    if (vfft_k1fs_band_t(N, T))   /* the prime sides too (2026-11-20) */
    {
        int n1[8], n2[8], ns = vfft_k1fs_splits(N, n1, n2, 8), i, j;
        int have[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        const int borrow = vfft_policy_highn_borrow();
        if (borrow && W)
            for (i = 0; i < ns; i++)
            {   /* borrowed: the row the child's row plan replays (in place, natural, one thread) must be among them */
                vw2_key_t k2d, krow, kbwd;
                int seen = 0;
                for (j = 0; j < i; j++) if (n2[j] == n2[i]) { seen = 1; have[i] = have[j]; }
                if (seen) continue;
                _k1fs_borrow_rows(S, W, n2[i]);
                have[i] = _k1fs_child_keys(n1[i], n2[i], 0, 1, &k2d, &krow, &kbwd) && vw2_lookup(&S->vw2, &krow) != NULL;
            }
        for (i = 0; i < ns; i++)
        {
            int seen = 0;
            for (j = 0; j < i; j++) if (n2[j] == n2[i]) seen = 1;
            if (seen || have[i]) continue;
            {
                vfft_cfg_t rc;
                vfft_plan rp;
                memset(&rc, 0, sizeof rc);
                rc.transform = VFFT_C2C; rc.placement = VFFT_INPLACE; rc.rigor = cfg->rigor;
                rc.dims = 1; rc.n[0] = n2[i]; rc.howmany = 1; rc.order = VFI_ORDER_NATURAL;
                rc.layout = VFI_LAYOUT_IL; rc.nthreads = 1;
                rc.wisdom = (vfft_wisdom *)S; rc.wisdom_write = cfg->wisdom_write;
                rp = _vfft_create_cfg(&rc);
                if (rp) vfft_destroy(rp);
            }
        }
        if (getenv("VFFT_NAT_LOG"))
            fprintf(stderr, "[k1fs] N=%d: %d split(s); row cells %s\n", N, ns,
                    borrow && W ? "borrowed where banked (VFFT_HIGHN_PLAN=borrow)" : "raced from scratch");
    }
}

/* ── THE IL PLAN RACE AT CREATE ─────────────────────────────────────────────
 * A kind-3 MISS (or recalibrate) runs the IL dp planner — the whole search:
 * the solos, every legal pair x its kernel
 * forms, every legal 3-stage chain x forms, the flat / ZTURN-T / four-step /
 * prime arms, the order swap, the backward forms — and banks its verdicts
 * (the kind-3 lay=il row, the dir=bwd row) before the create replays them.
 * What serves was measured. A cold cell takes seconds; the planner logs on
 * entry. VFFT_NO_K1PLAN=1 skips it (probe hook). Returns the lines banked,
 * 0 when no race ran, or -1 OUT OF MEMORY: the planner's four planes could
 * not be held. They are secured on entry, before the row warm and the prime
 * cell, so then nothing was warmed, raced or banked; the reason is recorded
 * and the caller refuses the create (never the default plan). */
static vfft_il_dp_context_t _k1_il_dp_ctx;      /* planning side, one create at a time */
static int _k1_il_dp_ctx_ready = 0;
static int _k1_il_dp_busy = 0;                  /* a race in progress: nested calls refuse */
/* the planner's planes for a race at N: held in *h unless the context already
 * covers N, at the size this door gives the context (the first create's, or
 * grown for the flat DIT's cells and the four-step's). 0, or -1 OUT OF MEMORY
 * (the reason recorded; nothing held). Allocates nothing when *h already
 * holds them. */
static int _k1_il_dp_hold(vfft_il_dp_planes_t *h, int N)
{
    int pn;
    if ((_k1_il_dp_ctx_ready && N <= _k1_il_dp_ctx.max_N) || h->max_N)
        return 0;
    pn = N <= VFFT_K1_IL_PLAN_MAX_N ? VFFT_K1_IL_PLAN_MAX_N
       : N > VFFT_K1_IL_PLAN_ODD_MAX_N ? N : VFFT_K1_IL_PLAN_ODD_MAX_N;
    if (vfft_il_dp_planes_alloc(h, pn) == 0)
        return 0;
    _vfft_warn("vfft_create: C2C N=%d: out of memory (measuring needs %.1f MB of planes)",
               N, (double)vfft_il_dp_planes_bytes(pn) / 1048576.0);
    return -1;
}
/* the race CEILINGS live in planning/policy.h (L9) */
/* bank = 0 (the race at T's own ranking, _k1_il_t_rank): the request's NATURAL class raced at one thread into the
 * context -- its routes' best left in the context's slot (dp_planner_il.h fam) -- and nothing banked; returns 1 when
 * a live candidate ranked */
static int _k1_il_plan_race_ex(struct vfft_wisdom_s *W, const vfft_cfg_t *cfg, int N, int bank);
static int _k1_il_plan_race(struct vfft_wisdom_s *W, const vfft_cfg_t *cfg, int N)
{
    return _k1_il_plan_race_ex(W, cfg, N, 1);
}
static int _k1_il_lean_off = 0;   /* the lean race's second pass, with the whole-length cell's arm (below) */
static int _k1_il_plan_race_ex(struct vfft_wisdom_s *W, const vfft_cfg_t *cfg, int N, int bank)
{
    vfft_il_cand_t top;
    int lines;
    /* THE LEAN PLANNER above VFFT_K1FS_MIN_N (owner, 2026-11-20; policy_il.h, the prime sides): a composite with a
     * prime factor past the chain radices that splits races the four-step and the flat chain, never the whole-length
     * cell -- its arm is not built, so the race holds no prime candidate (dp_planner_il.h _il_dp_enumerate_prime).
     * Where nothing of that race banks, the race runs again with the arm (the end of this function). */
    const int lean = !_k1_il_lean_off && bank && vfft_k1fs_prime_composite(N) && vfft_k1fs_band_t(N, 1);
    struct vfft_wisdom_s *fsS;   /* the four-step candidates' children race into it (k1_fourstep.h) */
    vfft_cfg_t fscfg;
    vfft_il_dp_planes_t planes;  /* the planner's planes, held from entry until the context takes them */
    /* WISDOM OR RACE, never a fallback: a request names (N, layout, order,
     * placement); the door looks that cell up and on a miss RACES the
     * interleaved planner's pool, banks the winner and serves it. Which N
     * race is policy.h's (ownership + budget, below). */
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
     * banks its own row) and (b) a nested call refuses. THE CHILD'S OWN
     * STORE (k1_fourstep.h): the row cells and every four-step candidate's
     * 2D child race into the race's PRIVATE store, never the caller's; the
     * winner's child is banked on its K=1 row (vfft_il_dp_emit_wisdom). */
    if (_k1_il_dp_busy)
        return 0;
    /* THE PLANES FIRST: out of memory refuses here, before anything is
     * warmed, raced or banked. They are held apart from the context: the
     * nested creates below share it, and it is not touched until the check
     * after them, so they find it (and its candidate cache) as the last create left it. */
    memset(&planes, 0, sizeof planes);
    if (_k1_il_dp_hold(&planes, N) != 0)
        return -1;
    fsS = _k1fs_store_new();
    if (!fsS)
    {   /* OUT OF MEMORY: the four-step candidates' private store -- refused with that cause, never the default plan */
        vfft_il_dp_planes_free(&planes);
        _vfft_warn("vfft_create: C2C N=%d: out of memory (the four-step candidates' store)", N);
        return -1;
    }
    fscfg = *cfg;
    fscfg.wisdom = (vfft_wisdom *)fsS;
    fscfg.wisdom_write = 0;
    fscfg.recalibrate = 0;
    _k1_il_fs_warm(fsS, W, &fscfg, N);
    /* the PRIME arm: the prime cell built ONCE here -- a cold
     * cell races its inner pool and banks the prime shard's row -- and lent
     * to the race through _k1pr_ctx (dp_planner_il.h), at the lengths the
     * band map admits it: those no chain carries (vfft_policy_prime_cell).
     * Nowhere else is it built, so a chain-carried cell never pays its inner
     * race. The verdict's plan is handed to the candidate after the race; a
     * losing plan dies below. */
    _k1pr_release();
    if (vfft_policy_prime_cell(N) && cfg->layout == VFI_LAYOUT_IL && !lean && bank)
    {   /* (bank 0, the ranking for the race at T: a prime cell reaches it only by the prime sides at T, and that race
         * builds the whole-length cell at T itself -- _k1_il_t_race) */
        _k1pr_ctx.plan = _ilprime_create_banked(W, cfg, N, NULL, _vfft_plan_threads(cfg));
        _k1pr_ctx.N = _k1pr_ctx.plan ? N : 0;
    }
    /* CHECK AGAIN: the row warm and the prime cell ran creates through this
     * door on the same context. A create only grows it (it installs planes
     * larger than the context it found, and a create that cannot hold them
     * leaves the context as it was), so a context that covered N on entry
     * still does and this check allocates nothing: the planes held since
     * entry are kept for the context below, or freed. */
    if (_k1_il_dp_hold(&planes, N) != 0)
    {
        _vfft_wisdom_free((vfft_wisdom *)fsS);
        _k1pr_release();
        return -1;
    }
    if (_k1_il_dp_ctx_ready && N <= _k1_il_dp_ctx.max_N)
        vfft_il_dp_planes_free(&planes);   /* the context covers N: entry's, or grown past N by a nested create */
    else
    {   /* the first create's context, or grown for the flat DIT's cells and
         * the four-step's (the candidate cache restarts, wisdom is the memory) */
        if (_k1_il_dp_ctx_ready)
            vfft_il_dp_destroy(&_k1_il_dp_ctx);
        vfft_il_dp_init_held(&_k1_il_dp_ctx, &planes);
        _k1_il_dp_ctx_ready = 1;
    }
    _k1fs_ctx.W = fsS;
    _k1fs_ctx.cfg = &fscfg;
    if (cfg->rigor != VFFT_MEASURE)
        vfft_il_dp_set_patient(&_k1_il_dp_ctx);
    else
        vfft_il_dp_set_measure(&_k1_il_dp_ctx); /* every rigor field of the shared context, each create (a PATIENT
                                                 * create before this one must not leak into it) */
    if (getenv("VFFT_NAT_LOG"))
        fprintf(stderr, "[k1plan] N=%d: IL plan race (solos, pairs x forms, ZTURN-T "
                        "chains x widths, chain3 x forms, bwd forms) — a cold cell "
                        "takes seconds\n", N);
    _k1_il_dp_busy = 1;
    if (bank)
        lines = vfft_il_dp_plan_and_bank(&_k1_il_dp_ctx, &W->vw2, N,
                                         cfg->placement == VFFT_INPLACE,   /* the cell's placement */
                                         getenv("VFFT_IL_DP_VERBOSE") != NULL);
    else
    {   /* the ranking alone, the natural class at the cell's placement, the cached verdict dropped so it races in full */
        vfft_il_cand_t nat;
        _k1_il_dp_ctx.inplace = cfg->placement == VFFT_INPLACE;
        vfft_il_dp_forget(&_k1_il_dp_ctx, N, VFFT_IL_ORD_NATURAL);
        lines = vfft_il_dp_plan(&_k1_il_dp_ctx, N, VFFT_IL_ORD_NATURAL, &nat,
                                getenv("VFFT_IL_DP_VERBOSE") != NULL) < 1e17 ? 1 : 0;
    }
    _k1_il_dp_busy = 0;
    _k1fs_ctx.W = NULL;
    _k1fs_ctx.cfg = NULL;
    _vfft_wisdom_free((vfft_wisdom *)fsS);
    if (!bank)
    {
        _k1pr_release();
        return lines;
    }
    if (lines > 0)
        _vw2_persist(W, cfg);
    if (_k1pr_ctx.plan)
    {   /* the prime plan outlives the race only as the request's verdict; at the prime sides at T the race at T decides
         * that verdict (_k1_il_t_race, which the door runs next), so the plan built at T waits for it there */
        vfft_oop_il_entry_t pe;
        const int ip_req = (cfg->placement == VFFT_INPLACE);
        const int scr_req = (vfft_policy_ord_k1(cfg, N, ip_req) == VW2_ORD_SCR);
        const int Tq = _vfft_plan_threads(cfg);
        const int primeT = Tq > 1 && !scr_req && vfft_k1fs_prime_composite(N) && vfft_k1fs_band_t(N, Tq) &&
                           !vfft_k1fs_band_t(N, 1);
        if (!primeT && !(vw2_oop_lookup_k1_il_cell(&W->vw2, N, scr_req, ip_req, 1, &pe) &&   /* the race banks the one-thread row */
                         pe.k1_il_route == VFFT_K1_IL_PRIME))
            _k1pr_release();
    }
    if (getenv("VFFT_NAT_LOG"))
    {   /* the race's verdict as the context holds it: READ, never planned again. vfft_il_dp_rank plans; a patient
         * create (every create) re-measures its cached top there, which ran every candidate a second time -- the
         * four-step's after its store is gone (_k1fs_ctx.W = NULL above), so its children raced again against the
         * library's own store (2026-11-20: 67 of 209 s at 2097151) */
        const vfft_il_dp_entry_t *le = _il_dp_lookup(&_k1_il_dp_ctx, N,
                                                     cfg->order == VFI_ORDER_SCRAMBLED ? VFFT_IL_ORD_SCRAMBLED
                                                                                        : VFFT_IL_ORD_NATURAL);
        if (le && le->n_top > 0)
        {
            top = le->top[0];
            fprintf(stderr, "[k1plan] N=%d: ilp=%.0fns -> ILP (route %d, %d.%d, %d line(s) banked, %s cell)\n",
                    N, top.cost_ns, top.route, top.R1, top.R2, lines,
                    cfg->order == VFI_ORDER_SCRAMBLED ? "ord=scr" : "ord=nat");
        }
    }
    if (lean && lines == 0)
    {   /* THE LEAN RACE built nothing (every split refused: a 2D child whose column the four-step cannot take, and no
         * flat chain): the race again, with the whole-length cell's arm */
        if (getenv("VFFT_NAT_LOG"))
            fprintf(stderr, "[k1plan] N=%d: no split and no flat chain built -- the race again with the whole-length "
                            "prime cell\n", N);
        if (_k1_il_dp_ctx_ready)
        {   /* the empty verdicts out of the context, so the second pass races in full */
            vfft_il_dp_forget(&_k1_il_dp_ctx, N, VFFT_IL_ORD_NATURAL);
            vfft_il_dp_forget(&_k1_il_dp_ctx, N, VFFT_IL_ORD_SCRAMBLED);
        }
        _k1_il_lean_off = 1;
        lines = _k1_il_plan_race_ex(W, cfg, N, bank);
        _k1_il_lean_off = 0;
    }
    return lines;
}

static vfft_k1fs_plan_t *_k1fs_replay(struct vfft_wisdom_s *W, const vfft_cfg_t *cfg, int N, int scr, int ip,
                                      const vfft_oop_il_entry_t *ke);
static int _k1fs_row_stale(struct vfft_wisdom_s *W, int N, int scr, int ip, int T, const vfft_oop_il_entry_t *ke);
static int _k1_il_t_race(struct vfft_wisdom_s *W, const vfft_cfg_t *cfg, int N, int T);   /* below: the 1D plan at T */
/* ── the K=1 IL-engine candidate: the request's cell (order x placement),
 * raced and banked on a miss, resolved to exactly one plan — the banked
 * route's (prime / chain3 / flat / ZTURN-T / four-step / pair), else the
 * balanced pair, else the il3p default chain. A MONO verdict builds nothing
 * here: the caller serves the mono door.
 * ⚠ The pair heuristic MIRRORS the OOP K=1 block's IL search (the
 * "IL runs its OWN pair search" rules in c2c_oop_create.h: il2p registries
 * stop at R=64, no parity constraint) — if you touch one, touch both.
 * Planning side only. */
/* Returns 1 when the cell's BANKED row did not build as written -- its engine,
 * chain or tile is refused, or a form names a kernel this build does not have.
 * Such a row is EMPTY (owner, 2026-10-04): what was built here from the
 * pair/default path is not the cell's plan, and the door races the cell from
 * scratch (a recalibrating second call). 0 = built as the row says, or the
 * cell has no row. -1 = OUT OF MEMORY in the plan race (_k1_il_plan_race):
 * nothing built, the reason recorded; the caller refuses the create. */
static int _k1_il_candidate(struct vfft_wisdom_s *W, const vfft_cfg_t *cfg,
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
        return 0;
    int iR1 = 0, iR2 = 0;
    int stale = 0;
    vfft_oop_il_entry_t keb;
    /* the request's ORDER CELL: an explicit SCRAMBLED request reads the
     * ord=scr row — the scrambled pool's own verdict — and nothing else;
     * DEFAULT and NATURAL read the ord=nat row. */
    /* the request's PLACEMENT CELL: an in-place request reads and races the
     * place=ip row -- its own verdict, every arm executed in place -- never
     * the out-of-place cell's (one contract per request). */
    const int ip_req = (cfg->placement == VFFT_INPLACE);
    const int scr_req = (vfft_policy_ord_k1(cfg, N, ip_req) == VW2_ORD_SCR);
    const int T = _vfft_plan_threads(cfg);   /* the plan's thread count: its row's key (v1.3) */
    /* the cell's INTERLEAVED axis (il/wisdom/wisdom2_oop_il.h); the
     * vw2_off_oop kill switch reads the legacy line's IL fields */
    const vfft_oop_wisdom_entry_t *kle =
        W->vw2_off_oop ? vfft_oop_wisdom_lookup_k1(&W->oop, N) : NULL;
    if (kle)
        vfft_oop_il_from_legacy(&keb, kle);
    const vfft_oop_il_entry_t *ke =
        W->vw2_off_oop ? (kle ? &keb : NULL)
                       : (vw2_oop_lookup_k1_il_cell(&W->vw2, N, scr_req, ip_req, T, &keb) ? &keb : NULL);
    /* the threaded plan's row (v1.3): a miss at T > 1 whose one-thread row exists. Natural order: the recipe is
     * chosen AT T (_k1_il_t_race, 2026-11-18: every candidate that threads raced in its threaded form); where that
     * race banks nothing (nothing in the pool threads, or a serial plan wins), and for the scrambled class, the
     * one-thread row's recipe becomes the row at T, keyed nthreads=T, and the MT commits below race its threaded
     * verdict there */
    if (!ke && T > 1 && !W->vw2_off_oop && !cfg->recalibrate &&
        vw2__oop_k1_scan_pl(&W->vw2, N, VW2_LAY_IL, scr_req, ip_req ? VW2_PL_IP : VW2_PL_OOP, 1))
    {
        const int tr = scr_req ? 0 : _k1_il_t_race(W, cfg, N, T);
        if (tr < 0)
            return -1;   /* out of memory: nothing built, the caller refuses the create */
        if (tr == 0 && vw2_oop_k1_row_at_T(&W->vw2, N, scr_req, ip_req, T))
            _vw2_persist(W, cfg);
        ke = vw2_oop_lookup_k1_il_cell(&W->vw2, N, scr_req, ip_req, T, &keb) ? &keb : NULL;
    }
    /* the IL plan race: a MISS (no IL verdict on the row), a pair-only row
     * whose forms were never raced, or recalibrate races the planner's pools
     * and banks the one-thread row, then replays; _k1_il_plan_race carries
     * the N gate. The scrambled pow2 band races here too: its writer is the
     * PLAIN ZTURN-T schedule, banked on the ord=scr row like every other cell. */
    if (!W->vw2_off_oop &&
        (cfg->recalibrate || !ke || !ke->il_kv_raced ||   /* a pair-only row (forms unraced) plans too */
         (ke->k1_il_route == VFFT_K1_IL_PRIME && !vfft_policy_prime_cell(N)) ||   /* a prime verdict where a chain carries N: stale, the chains race */
         _k1fs_row_stale(W, N, scr_req, ip_req, T, ke)))   /* a four-step verdict without its child: stale */
    {
        const int lines = _k1_il_plan_race(W, cfg, N);
        if (lines < 0)
            return -1;   /* out of memory: nothing built, the caller refuses the create */
        if (lines > 0)
        {
            if (T > 1)
            {   /* the row at T: the recipe chosen at T on the ranking just raced (above) */
                const int tr = scr_req ? 0 : _k1_il_t_race(W, cfg, N, T);
                if (tr < 0)
                    return -1;
                if (tr == 0 && vw2_oop_k1_row_at_T(&W->vw2, N, scr_req, ip_req, T))
                    _vw2_persist(W, cfg);
            }
            ke = vw2_oop_lookup_k1_il_cell(&W->vw2, N, scr_req, ip_req, T, &keb) ? &keb : NULL;
        }
    }
    /* a SCRAMBLED request at a pow2 cell with no scrambled row after the race
     * builds NOTHING here — no default pair, no heuristic (NO FALLBACKS): the
     * race is the only source of a scrambled plan, and a natural-writing pair
     * is not one. */
    {   /* the writer-band law (planning/policy.h): no fallback here */
        const vfft_cell_t sc = vfft_policy_cell(cfg, N, 1, 1, 0, 1);
        if (scr_req && vfft_policy_scr_writer_band(&sc) && !ke)
            return 0;
    }
    /* MONO verdict: the cell's plan is ONE solo kernel; no pair is built
     * here — the caller serves the mono door (the OOP block reads the form
     * itself; in place, _k1_il_mono_candidate). Without this return an
     * in-place create would replay a MONO row as the balanced pair. */
    if (ke && ke->k1_il_route == VFFT_K1_IL_MONO &&
        (vfft_k1_mono_il_fn(N, 0) || vfft_k1_mono_il_form_fn(N, ke->il_kv, 0)))   /* a length whose only solo kernels are the fused forms (FP32, 128 and up) has no form 0 */
        return 0;
    /* PRIME verdict: the cell's plan is the prime cell. The race's warm plan
     * is handed over when the race just ran (never rebuilt: under
     * recalibrate a rebuild would race the inner a second time and could
     * bank a different one); a replay builds it from the prime shard's row. */
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
                *ilp_out = _ilprime_create_banked(W, cfg, N, NULL, _vfft_plan_threads(cfg));
            if (getenv("VFFT_NAT_LOG") && *ilp_out)
                fprintf(stderr, "[k1pr] N=%d: %s prime cell (%s, M=%d) src=wisdom\n", N,
                        ip_req ? "in place" : "out of place",
                        (*ilp_out)->method ? "RADER" : "BLUESTEIN", (*ilp_out)->M);
        }
        _k1pr_release();
        return (ilp_out && !*ilp_out) ? 1 : 0;   /* a prime verdict whose cell does not build */
    }
    /* CHAIN3 verdict: the banked 3-stage chain replays as
     * written; a build refusal falls through to the pair/default path */
    if (ke && ke->k1_il_route == VFFT_K1_IL_CHAIN3 && ke->il_c3[0])
    {
        *il3p_out = vfft_il3p_create(N, ke->il_c3[0], ke->il_c3[1], ke->il_c3[2]);
        if (*il3p_out)
        {
            stale = _k1_il3p_apply_kv(*il3p_out, ke, &W->vw2, N, ip_req);   /* banked forms > default */
            if (getenv("VFFT_NAT_LOG"))
                fprintf(stderr, "[k1c3] N=%d: replay chain %d.%d.%d src=wisdom\n",
                        N, ke->il_c3[0], ke->il_c3[1], ke->il_c3[2]);
            return stale;
        }
    }
    /* FLAT DIT verdict: the banked chain + per-stage forms
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
            if (fp && ke->il_rd_n > 0 && fp->has_rd)
            {   /* the Rader stage's banked inner and width */
                int rs = 0;
                if (!vfft_ilrd_get_inner(fp, &rs, NULL, NULL, NULL) ||
                    !vfft_ilrd_set_inner(fp, rs, ke->il_rd, ke->il_rd_n, (size_t)(ke->il_rdw > 0 ? ke->il_rdw : 0)))
                {
                    vfft_ilfd_destroy(fp);
                    fp = NULL;
                }
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
            return 0;
        }
    }
    /* ZTURN-T verdict: the banked chain replays as written (validated by the
     * create: legality, the quarter-wave's octave, the registry cell). The
     * ORDER CLASS is the row's: an ord=nat row replays the natural drivers,
     * an ord=scr row the PLAIN schedule's — one plan, one order, never mixed.
     * A refusal falls through to the pair/default path. */
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
            return 0;
        }
    }
    /* the FOUR-STEP (route 10): a banked verdict replays its split with its
     * child from the row (_k1fs_replay) -- the request's placement and
     * thread count, the order class the row's */
    if (ke && ke->k1_il_route == VFFT_K1_IL_FS && fs_out && ke->il_R1 > 0 && ke->il_R2 > 0 &&
        (long)ke->il_R1 * (long)ke->il_R2 == (long)N)
    {
        vfft_k1fs_plan_t *fp = _k1fs_replay(W, cfg, N, scr_req, ip_req, ke);
        if (fp)
        {
            *fs_out = fp;
            return 0;
        }
    }
    /* a banked MONO, CHAIN3, FLAT, ZTURN-T or four-step verdict that reaches
     * this point was not built: what follows is the pair/default path, not the
     * row's plan */
    if (ke && (ke->k1_il_route == VFFT_K1_IL_MONO || ke->k1_il_route == VFFT_K1_IL_CHAIN3 ||
               (ke->k1_il_route == VFFT_K1_IL_FLAT && ilfd_out) ||
               (ke->k1_il_route == VFFT_K1_IL_ZTT && ztt_out) ||
               (ke->k1_il_route == VFFT_K1_IL_FS && fs_out)))
        stale = 1;
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
    {   /* braces load-bearing: apply_kv must not run when the pair axis
         * was skipped. */
        *il2p_out = vfft_il2p_create(N, iR1, iR2);
        if (_k1_il2p_apply_kv(*il2p_out, ke, &W->vw2, N, ip_req))   /* wisdom verdict > default */
            stale = 1;
    }
    /* a banked pair that the pair engine refuses */
    if (ke && ke->il_R1 && !*il2p_out &&
        (ke->k1_il_route == VFFT_K1_IL_2P_PURE || ke->k1_il_route == VFFT_K1_IL_2P ||
         ke->k1_il_route == VFFT_K1_IL_3P))
        stale = 1;
    /* Ordering is a measured axis: (R1,R2) and (R2,R1) install different mid
     * kernels. Heuristic pairs only — a wisdom pair is the calibrator's.
     * See docs/design/vfft_front_door.md. */
    /* Per-process MEMO of the ordering pick, keyed by N: the race must
     * run at most ONCE per process per cell — without this, the natural
     * and scrambled handles (and measure vs consume) each re-race, and a
     * margin near the hysteresis flips on noise, breaking the
     * bitwise-identity contracts between them. Planning-side, no locks:
     * worst case a benign double race on concurrent first creates. */
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
            vfft_real *rz = (vfft_real *)vfft_aligned_alloc(2 * (size_t)N * sizeof(vfft_real));
            vfft_real *r0 = (vfft_real *)vfft_aligned_alloc(2 * (size_t)N * sizeof(vfft_real));
            if (rz && r0)
            {
                for (long i = 0; i < 2L * N; i++)
                    r0[i] = (double)(i % 17) * 0.0625 - 0.5;
                const int reps = N <= 256 ? 64 : (N <= 1024 ? 24 : 8);
                const size_t nb = 2 * (size_t)N * sizeof(vfft_real);
                double ta, tb;
                {
                    _k1ord_arm_t ca = { *il2p_out, rz, r0, nb };
                    _k1ord_arm_t cb = { alt, rz, r0, nb };
                    const vfft_race_arm_t arms[2] = {
                        { "heuristic", _k1ord_arm_run, &ca },
                        { "swapped", _k1ord_arm_run, &cb } };
                    /* 5 rounds, A then B, min; reseed before every burst:
                     * repeated in-place fwd amplifies magnitudes toward inf */
                    const vfft_race_proto_t proto = { 5, reps, VFFT_RACE_MIN, 0, 0,
                                                      _k1ord_reseed, &ca, 1 } /* single-thread arms: paced (VFFT_RACE_PACE_MS) */;
                    double ns[2];
                    vfft_race_run(&proto, arms, 2, ns);
                    ta = ns[0];
                    tb = ns[1];
                }
                /* 3% hysteresis, incumbent (heuristic) keeps ties. */
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
                /* NOTHING IS BANKED (owner, 2026-10-04): the two arms are the two
                 * orders of a pair a rule picked, not the cell's pool, so the
                 * pick is no verdict. The per-process memo below keeps every
                 * create in this process on one order. */
            }
            vfft_aligned_free(rz);
            vfft_aligned_free(r0);
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
    return stale;
}


/* ── the FLAT DIT's threading verdict (il_flatdit_mt.h): env pin > the
 * banked il_mt at THIS T
 * on the cell's kind-3 IL row (ord=nat or ord=scr — the plan's own class)
 * > the race at T (serial vs blocks vs tiles at every legal width, steady-
 * state samples), banked il_mt= il_mt_tw= on the plan's own row (its thread
 * count is the key, v1.3). The one-thread width
 * il_tw= stays what a T=1 plan replays. */
static void _ilfd_mt_replay_or_race(struct vfft_plan_s *h,
                                    struct vfft_wisdom_s *W,
                                    const vfft_cfg_t *cfg, int N)
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
        r = vw2__oop_k1_scan_pl(&W->vw2, N, VW2_LAY_IL, p->scr,
                                h->placement == VFFT_INPLACE ? VW2_PL_IP : VW2_PL_OOP, T);   /* the plan's own row (v1.3) */
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
    if (r && !cfg->recalibrate && vw2_rec_get(r, "il_mt"))
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
        vfft_real *zi = (vfft_real *)vfft_aligned_alloc(2 * (size_t)N * sizeof(vfft_real));
        vfft_real *zo = (vfft_real *)vfft_aligned_alloc(2 * (size_t)N * sizeof(vfft_real));
        size_t i;
        if (!zi || !zo)
        {
            vfft_aligned_free(zi); vfft_aligned_free(zo);
            p->mt = 0;
            return;
        }
        for (i = 0; i < 2 * (size_t)N; i++)
            zi[i] = 1.0 + 1e-6 * (double)(i & 1023);
        vfft_ilfd_mt_race(p, T, tw0, zi, zo, &mt_tw);
        p->mt_tw = mt_tw;
        vfft_aligned_free(zi); vfft_aligned_free(zo);
    }
    if (r && !W->vw2_off_oop)
    {
        char b[16];
        int ok = 1;
        snprintf(b, sizeof b, "%d", p->mt);
        ok = ok && vw2_update_field(&W->vw2, &r->key, "il_mt", b) == VW2_OK;
        snprintf(b, sizeof b, "%d", mt_tw);
        ok = ok && vw2_update_field(&W->vw2, &r->key, "il_mt_tw", b) == VW2_OK;
        if (ok)
            _vw2_persist(W, cfg);
    }
}

/* ── ZTURN-T's threading verdict (ztt_mt.h; the flat DIT's law above):
 * env pin VFFT_ZTT_MT=0|1|2 (never banked) > the
 * banked arm at THIS T on the cell's il_route=ztt row of the plan's own
 * order class > the race at T (serial vs blocks vs tiles, steady-state
 * samples, hot), banked il_mt= on the plan's own row: placement and thread
 * count are its key (v1.3), so a plan bound IN PLACE, which races aliased
 * arms through the plane — a different measurement — banks on the place=ip
 * row at its T. The one-thread tile il_tw= is
 * untouched; the TILES arm races at every stage span the tile law admits
 * (vfft_ztt_mt_tile_cands, 2026-11-18) and its width banks as il_mt_tw= on
 * the row at T (the flat DIT's law): a 2^a x odd chain, untiled at one
 * thread, gets its tiles arm. */
static void _ztt_mt_replay_or_race(struct vfft_plan_s *h,
                                   struct vfft_wisdom_s *W,
                                   const vfft_cfg_t *cfg, int N)
{
    vfft_ztt_plan_t *p = h->k1ztt;
    const int T = h->nthreads;
    const int ip = (h->placement == VFFT_INPLACE);
    const char *tok_v = "il_mt";   /* the plan's own row: placement and thread count are its key (v1.3) */
    const vw2_rec_t *r = NULL;
    const char *pin = getenv("VFFT_ZTT_MT");
    if (!p || T < 2)
        return;
    if (W && !W->vw2_off_oop)
        r = vw2__oop_k1_scan_pl(&W->vw2, N, VW2_LAY_IL, p->scr,
                                h->placement == VFFT_INPLACE ? VW2_PL_IP : VW2_PL_OOP, T);   /* the plan's own row (v1.3) */
    if (pin)
    {
        const int v = atoi(pin);
        if (!vfft_ztt_mt_bind(p, T, (v >= 0 && v <= 2) ? v : 0)) p->mt = 0;
        if (getenv("VFFT_NAT_LOG"))
            fprintf(stderr, "[ztt-mt] N=%d T=%d %s%s: mt=%d src=env\n", N, T, p->scr ? "scr" : "nat", ip ? " ip" : "", p->mt);
        return;
    }
    size_t cands[VFFT_ZTT_MAX_NF + 1], tw = 0;
    int nc = vfft_ztt_mt_tile_cands(p, cands, VFFT_ZTT_MAX_NF);
    {   /* the one-thread width among them (the planner's ladder width at a power of two) */
        int q, have = 0;
        for (q = 0; q < nc; q++) have |= (cands[q] == p->tile);
        if (p->tile && !have && (size_t)N / p->tile >= 2) cands[nc++] = p->tile;
    }
    /* the banked verdict at THIS T, with its tile width (il_mt_tw, 2026-11-18); a row raced before the widths
     * (il_mt and no il_mt_tw) races again where the plan offers widths: its tiles arm was the one-thread width only */
    if (r && !cfg->recalibrate && vw2_rec_get(r, tok_v) && (vw2_rec_get(r, "il_mt_tw") || nc == 0))
    {
        int v = vw2__oop_geti(r, tok_v, 0);
        const int w = vw2__oop_geti(r, "il_mt_tw", 0);
        if (v == 2 && w > 0 && !vfft_ztt_set_tile(p, (size_t)w))
        {
            _vfft_warn("banked il_mt_tw=%d does not fit the ZTURN-T chain at N=%d - serial", w, N);
            v = 0;
        }
        if (!vfft_ztt_mt_bind(p, T, (v >= 0 && v <= 2) ? v : 0)) p->mt = 0;
        if (getenv("VFFT_NAT_LOG"))
            fprintf(stderr, "[ztt-mt] N=%d T=%d %s%s: replay mt=%d tw=%zu src=wisdom\n",
                    N, T, p->scr ? "scr" : "nat", ip ? " ip" : "", p->mt, p->tile);
        return;
    }
    {   /* the race on 64-B aligned scratch, in the plan's own placement: serial, blocks, the tiles at every width */
        const size_t nb = (size_t)2 * N * sizeof(vfft_real);
        vfft_real *zi = (vfft_real *)vfft_aligned_alloc(nb);
        vfft_real *zo = (vfft_real *)vfft_aligned_alloc(nb);
        size_t i;
        if (!zi || !zo) { vfft_aligned_free(zi); vfft_aligned_free(zo); p->mt = 0; return; }
        for (i = 0; i < 2 * (size_t)N; i++) zi[i] = 1.0 + 1e-6 * (double)(i & 1023);
        _vfft_pool_arm(T);
        if (ip) { memcpy(zo, zi, nb); vfft_ztt_mt_race(p, T, zo, zo, NULL, cands, nc, &tw); }
        else vfft_ztt_mt_race(p, T, zi, zo, NULL, cands, nc, &tw);
        vfft_aligned_free(zi); vfft_aligned_free(zo);
    }
    if (r && !W->vw2_off_oop)
    {
        const vw2_key_t key = r->key;   /* an update may move the record */
        char b[24];
        int ok = 1;
        snprintf(b, sizeof b, "%d", p->mt);
        ok = ok && vw2_update_field(&W->vw2, &key, tok_v, b) == VW2_OK;
        snprintf(b, sizeof b, "%zu", tw);
        ok = ok && vw2_update_field(&W->vw2, &key, "il_mt_tw", b) == VW2_OK;
        if (ok)
            _vw2_persist(W, cfg);
    }
}

/* ── the PRIME CELL's threading verdict (il_prime_mt.h; ZTURN-T's law above):
 * the banked arm at THIS T on the cell's own row (il_route=prime, the plan's
 * placement and order class, nthreads=T) > the race at T (serial vs blocks
 * vs tiles on the whole convolution, hot), banked il_mt= on that row -- the
 * K=1 record's word for a threaded arm. A cell whose inner is not ZTURN-T
 * has no threaded form and banks il_mt=0. */
static void _ilpr_mt_replay_or_race(struct vfft_plan_s *h,
                                    struct vfft_wisdom_s *W,
                                    const vfft_cfg_t *cfg, int N)
{
    vfft_ilprime_plan_t *p = h->k1ilpr;
    const int T = h->nthreads;
    const int ip = (h->placement == VFFT_INPLACE);
    const int scr = (vfft_policy_ord_k1(cfg, N, ip) == VW2_ORD_SCR);
    const vw2_rec_t *r = NULL;
    if (!p || T < 2)
        return;
    if (W && !W->vw2_off_oop)
        r = vw2__oop_k1_scan_pl(&W->vw2, N, VW2_LAY_IL, scr, ip ? VW2_PL_IP : VW2_PL_OOP, T);   /* the plan's own row (v1.3) */
    if (_ilpr_tv.p == p && _ilpr_tv.N == N && _ilpr_tv.T == T)
    {   /* THE VERDICT AT T raced in this create (_ilprime_create_banked): the plan is bound to its arm already; the
         * inner, the arm and a four-step inner's child onto the row at T (a row's earlier il_mt named the one-thread
         * inner's arm: the race at T replaces it). The key is copied: an update may move the record. */
        if (r && !W->vw2_off_oop)
        {
            const vw2_key_t key = r->key;
            char kind[8], shape[64], b[16];
            int ok;
            _ilprime_desc_str(&_ilpr_tv.d, kind, sizeof kind, shape, sizeof shape);
            ok = vw2_update_field(&W->vw2, &key, "il_prime", _ilpr_tv.method == 1 ? "rader" : "bluestein") == VW2_OK;
            ok = ok && vw2_update_field(&W->vw2, &key, "il_prime_in", kind) == VW2_OK;
            ok = ok && vw2_update_field(&W->vw2, &key, "il_prime_sh", shape) == VW2_OK;
            snprintf(b, sizeof b, "%d", _ilpr_tv.d.tw);
            ok = ok && vw2_update_field(&W->vw2, &key, "il_prime_tw", b) == VW2_OK;
            snprintf(b, sizeof b, "%d", p->mt);
            ok = ok && vw2_update_field(&W->vw2, &key, "il_mt", b) == VW2_OK;
            if (ok && _ilpr_tv.d.kind == 4 && _ilpr_tv.S)
            {
                const vw2_rec_t *r2 = vw2_lookup(&W->vw2, &key);
                ok = r2 && _k1fs_row_put_child(&W->vw2, r2, _ilpr_tv.S, _ilpr_tv.d.R1, _ilpr_tv.d.R2, 0,
                                               _ilpr_tv.d.T > 1 ? _ilpr_tv.d.T : 1) == VW2_OK;   /* the child at the T it was built at */
            }
            if (ok)
                _vw2_persist(W, cfg);
            else
                fprintf(stderr, "vfft: the prime cell's verdict at T=%d (N=%d) NOT banked -- it races again\n", T, N);
        }
        if (getenv("VFFT_NAT_LOG"))
            fprintf(stderr, "[ilpr-mt] N=%d T=%d%s: raced with the inner at T, mt=%d\n", N, T, ip ? " ip" : "", p->mt);
        _ilpr_tv_clear();
        return;
    }
    if (r && !cfg->recalibrate && vw2_rec_get(r, "il_mt"))
    {
        const int v = vw2__oop_geti(r, "il_mt", 0);
        vfft_ilprime_mt_bind(p, T, (v >= 0 && v <= 2) ? v : 0);
        if (getenv("VFFT_NAT_LOG"))
            fprintf(stderr, "[ilpr-mt] N=%d T=%d%s: replay mt=%d src=wisdom\n", N, T, ip ? " ip" : "", p->mt);
        return;
    }
    _vfft_pool_arm(T);
    vfft_ilprime_mt_race(p, T, NULL);
    if (r && !W->vw2_off_oop)
    {
        char b[16];
        snprintf(b, sizeof b, "%d", p->mt);
        if (vw2_update_field(&W->vw2, &r->key, "il_mt", b) == VW2_OK)
            _vw2_persist(W, cfg);
    }
}

/* ── the FOUR-STEP's threaded arm: the SPLIT (and the natural class's FORM)
 * is the per-T verdict ──
 * The 1D race picks the split at one thread; at T > 1 the children's own
 * threaded verdicts reorder the ladder (4194304 at T=8: the serial winner
 * 1024x4096 runs 7.1 ms, 2048x2048 4.9 ms), so a plan at T races the
 * ladder's splits AT T — every child a 2D cell created at T, its threaded
 * verdict raced and banked on its own row — forward, in the plan's own
 * placement; the natural class races each split's form 0 and form 1 with
 * the residency sub-ladder of super-band chains. Banks il_mt=N1
 * il_mtsb=<chain|0> on the plan's own row (placement and thread count are
 * its key, v1.3): ZTURN-T's tokens, each route reading them as its own arm.
 * Replay rebuilds the banked plan when it differs from the row's serial
 * one. VFFT_K1_FS / VFFT_K1_FSSB (the probe pins) skip both. */
typedef struct { vfft_k1fs_plan_t *p; const vfft_real *zi; vfft_real *zo; int ip; } _k1fs_mt_ctx_t;
typedef struct { vfft_real *dst; const vfft_real *src; size_t nb; } _k1fs_mt_rst_t;
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
 * probe pin VFFT_K1_FSSB; returns the chain length (0 = form 0). The row is
 * the one-thread row of the request's PLACEMENT, where the planner banks
 * il_sb beside the il_kv the caller read (vfft_il_dp_emit_wisdom) */
static int _k1fs_row_sb(struct vfft_wisdom_s *W, int N, int ip, int il_kv, int *chain, int *form)
{
    const char *e = getenv("VFFT_K1_FSSB");
    *form = 0;
    if (e && *e) { const int n = _k1fs_parse_chain(e, chain); if (n >= 2) { *form = 1; return n; } }
    if (il_kv == 1 && W && !W->vw2_off_oop)
    {
        const vw2_rec_t *r = vw2__oop_k1_scan_pl(&W->vw2, N, VW2_LAY_IL, 0, ip ? VW2_PL_IP : VW2_PL_OOP, 1);
        const char *v = r ? vw2_rec_get(r, "il_sb") : NULL;
        if (v) { const int n = _k1fs_parse_chain(v, chain); if (n >= 2) { *form = 1; return n; } }
    }
    return 0;
}
/* THE BANKED FOUR-STEP (owner, 2026-10-04: its child on its own row,
 * k1_fourstep.h). The request's row (order, placement, thread count) names
 * the split -- il_mt where a threaded row's split race banked one, il_pair
 * otherwise -- and carries the child (fs_ tokens) the build replays from a
 * private store. A row without its child (a threaded row before its split
 * race; a pinned split) builds on an empty private store: the child races
 * into it. A banked child that did not replay is said loudly; the plan is
 * still the row's split. Said only when the child MEASURED (a race counted
 * or a measuring scope entered across its build): a child store that
 * changed without it -- the 2D row put its row plan's recipe on itself
 * (rp_, wisdom2_child.h) where the row carried the older fs_row_ tokens --
 * is re-banked in silence. NULL = the split does not build. */
static vfft_k1fs_plan_t *_k1fs_replay(struct vfft_wisdom_s *W, const vfft_cfg_t *cfg, int N, int scr, int ip,
                                      const vfft_oop_il_entry_t *ke)
{
    const int T = _vfft_plan_threads(cfg);
    const vw2_rec_t *r = (W && !W->vw2_off_oop)
                             ? vw2__oop_k1_scan_pl(&W->vw2, N, VW2_LAY_IL, scr, ip ? VW2_PL_IP : VW2_PL_OOP, T)
                             : NULL;
    int pn1 = ke->il_R1, pn2 = ke->il_R2, sbc[8], sbn = 0, form = 0, grew = 0;
    const int pinned = _k1fs_pin(N, &pn1, &pn2);
    const int child = !pinned && _k1fs_row_has_child(r);
    long races0, scopes0;
    vfft_k1fs_plan_t *fp;
    if (!pinned && T > 1 && child && vw2_rec_get(r, "il_mt"))
    {   /* the threaded row's own split and form: its split race's verdict, the child its own */
        const int v = vw2__oop_geti(r, "il_mt", 0);
        const char *sv = vw2_rec_get(r, "il_mtsb");
        if (v >= 4 && N % v == 0) { pn1 = v; pn2 = N / v; }
        if (sv && strcmp(sv, "0") != 0)
        {
            sbn = _k1fs_parse_chain(sv, sbc);
            form = sbn >= 2;
            if (!form) sbn = 0;
        }
    }
    else if (!scr)
        sbn = _k1fs_row_sb(W, N, ip, ke->il_kv, sbc, &form);
    races0 = _vfft_create_race_count;   /* did the child measure: every race counts (race.h), an outermost */
    scopes0 = vfft_measure_scopes();    /* clock read enters a scope (race_scope.h); c2c_estimate_il.h's detectors */
    fp = _k1fs_build_own(N, pn1, pn2, scr, cfg, ip, T, form, sbc, sbn, child ? r : NULL, &grew);
    if (fp && grew)
    {
        if (_vfft_create_race_count != races0 || vfft_measure_scopes() != scopes0)
            fprintf(stderr, "vfft: the four-step's banked child at N=%d (%dx%d) did not replay -- it raced in the plan's "
                            "private store (the row's fs_ tokens are incomplete)\n", N, pn1, pn2);
        /* the completed child back onto the row (re-banks it; r is stale after), so the cell does not race again */
        if (r && W && !W->vw2_off_oop && _k1fs_row_put_child(&W->vw2, r, fp->own, fp->N1, fp->N2, ip, T) == VW2_OK)
            _vw2_persist(W, cfg);
    }
    if (fp && getenv("VFFT_NAT_LOG"))
        fprintf(stderr, "[k1fs] N=%d: replay FOUR-STEP %dx%d form=%d src=%s (%s, T=%d)\n", N, fp->N1, fp->N2, fp->form,
                pinned ? "pin" : child ? "wisdom" : "wisdom, child raced", ip ? "ip" : "oop", T);
    return fp;
}
/* a one-thread FOUR-STEP row without its child is stale: the cell races (a
 * threaded row's child is its split race's, _k1fs_mt_replay_or_race) */
static int _k1fs_row_stale(struct vfft_wisdom_s *W, int N, int scr, int ip, int T, const vfft_oop_il_entry_t *ke)
{
    return ke && ke->k1_il_route == VFFT_K1_IL_FS && T == 1 && W && !W->vw2_off_oop &&
           !_k1fs_row_has_child(vw2__oop_k1_scan_pl(&W->vw2, N, VW2_LAY_IL, scr, ip ? VW2_PL_IP : VW2_PL_OOP, 1));
}
/* THE SPLITS AT T (the threaded split race's body; shared since 2026-11-18 with the zr2c child's threaded form,
 * il/real/zr2c_build.h): every split of N -- x form x chain for the natural class -- built AT T on 64-B aligned
 * scratch in the placement `ip`, the candidates' children racing into ONE private store mS (taken: the winner keeps
 * it, else it is freed). `p` (may be NULL) is a plan already at T, its own split's form-0 arm (the c2c door's served
 * plan) and the plan the reps are timed on; W lends the row cells under VFFT_HIGHN_PLAN=borrow. Returns the winner --
 * p, or a new plan owning mS -- every other candidate destroyed, p too when it lost; NULL when nothing raced (p
 * untouched, mS freed). */
static vfft_k1fs_plan_t *_k1fs_mt_split_race(vfft_k1fs_plan_t *p, struct vfft_wisdom_s *mS,
                                             const struct vfft_wisdom_s *W, const vfft_cfg_t *cfg, int N, int scr,
                                             int ip, int T)
{
    const size_t nb = (size_t)2 * N * sizeof(vfft_real);
    vfft_real *zi = (vfft_real *)vfft_aligned_alloc(nb);
    vfft_real *zo = (vfft_real *)vfft_aligned_alloc(nb);
    vfft_cfg_t mcfg = *cfg;
    vfft_k1fs_plan_t *cand[VFFT_RACE_MAX_ARMS], *win;
    _k1fs_mt_ctx_t cx[VFFT_RACE_MAX_ARMS];
    vfft_race_arm_t arms[VFFT_RACE_MAX_ARMS];
    char names[VFFT_RACE_MAX_ARMS][40];
    double tns[VFFT_RACE_MAX_ARMS];
    _k1fs_mt_rst_t rs;
    int n1[8], n2[8], ns, na = 0, best = 0, reps, a, i;
    size_t k;
    if (!zi || !zo || !mS)
    {
        vfft_aligned_free(zi); vfft_aligned_free(zo);
        _vfft_wisdom_free((vfft_wisdom *)mS);
        return NULL;
    }
    ns = vfft_k1fs_splits(N, n1, n2, 8);
    mcfg.wisdom = (vfft_wisdom *)mS;
    mcfg.wisdom_write = 0;
    mcfg.recalibrate = 0;
    for (k = 0; k < 2 * (size_t)N; k++) zi[k] = 1.0 + 1e-6 * (double)(k & 1023);
    for (i = 0; i < ns; i++)
    {   /* the candidates' row cells at T: the store's 1D rows at N2 (borrow); else the
         * one-thread child's rp_ recipe in mS serves them (_k1fs_child_row_store) */
        if (vfft_policy_highn_borrow() && W) _k1fs_borrow_rows(mS, W, n2[i]);
    }
    _vfft_pool_arm(T);
    for (i = 0; i < ns && na < VFFT_RACE_MAX_ARMS; i++)
    {
        int ch[24][8], cl[24], nch = 0, c;
        /* form 0 (the current plan when it is this split and form) */
        cand[na] = (p && n1[i] == p->N1 && p->form == 0) ? p
                 : vfft_k1fs_create(N, n1[i], n2[i], scr, mS, &mcfg, ip, T, 0, NULL, 0);
        if (cand[na])
        {
            cx[na].p = cand[na]; cx[na].zi = zi; cx[na].zo = zo; cx[na].ip = ip;
            snprintf(names[na], sizeof names[na], "%dx%d", n1[i], n2[i]);
            arms[na].name = names[na]; arms[na].run = _k1fs_mt_arm_run; arms[na].ctx = &cx[na];
            na++;
        }
        if (scr || !_k1fs_sb_admit(N, n2[i])) continue;
        nch = _k1fs_sb_chains(n1[i], ch, cl, 24, 1);
        for (c = 0; c < nch && na < VFFT_RACE_MAX_ARMS; c++)
        {
            char cs[48];
            int q, off = 0, same = 0;
            for (q = 0; q < cl[c] && off < (int)sizeof cs - 4; q++)
                off += snprintf(cs + off, sizeof cs - (size_t)off, "%s%d", q ? "." : "", ch[c][q]);
            if (p && n1[i] == p->N1 && p->form == 1 && p->sbnst == cl[c] && memcmp(p->sbR, ch[c], (size_t)cl[c] * sizeof(int)) == 0)
                same = 1;
            cand[na] = same ? p : vfft_k1fs_create(N, n1[i], n2[i], 0, mS, &mcfg, ip, T, 1, ch[c], cl[c]);
            if (!cand[na]) continue;
            cx[na].p = cand[na]; cx[na].zi = zi; cx[na].zo = zo; cx[na].ip = ip;
            snprintf(names[na], sizeof names[na], "%dx%d/%s", n1[i], n2[i], cs);
            arms[na].name = names[na]; arms[na].run = _k1fs_mt_arm_run; arms[na].ctx = &cx[na];
            na++;
        }
    }
    if (na == 0)
    {
        vfft_aligned_free(zi); vfft_aligned_free(zo);
        _vfft_wisdom_free((vfft_wisdom *)mS);
        return NULL;
    }
    {   /* reps from one timing of the serial verdict's plan at T (the first candidate when none is given) */
        const vfft_k1fs_plan_t *tp = p ? p : cx[0].p;
        double t0;
        if (ip) memcpy(zo, zi, nb);
        vfft_k1fs_execute(tp, VFFT_FORWARD, ip ? zo : zi, zo);
        if (ip) memcpy(zo, zi, nb);
        t0 = vfft_now_ns(); vfft_k1fs_execute(tp, VFFT_FORWARD, ip ? zo : zi, zo); t0 = vfft_now_ns() - t0;
        reps = (int)(20e6 / (t0 > 1.0 ? t0 : 1.0));
        if (reps < 2) reps = 2;
        if (reps > 64) reps = 64;
    }
    rs.dst = zo; rs.src = zi; rs.nb = nb;
    {
        const vfft_race_proto_t proto = { 3, reps, VFFT_RACE_MIN, 1, 2, ip ? _k1fs_mt_reseed : NULL, ip ? &rs : NULL, 0 }; /* THREADED arms: never paused (VFFT_RACE_PACE_MS) */
        vfft_race_run(&proto, arms, na, tns);
    }
    for (a = 1; a < na; a++) if (tns[a] < tns[best]) best = a;
    win = cx[best].p;
    if (win != p)
    {   /* the winner keeps the race's store: its child is there */
        win->own = mS;
        mS = NULL;
    }
    _vfft_wisdom_free((vfft_wisdom *)mS);
    for (a = 0; a < na; a++) if (cand[a] && cand[a] != win && cand[a] != p) vfft_k1fs_destroy(cand[a]);
    if (p != win) vfft_k1fs_destroy(p);
    if (getenv("VFFT_NAT_LOG"))
    {
        fprintf(stderr, "[k1fs-mt] N=%d T=%d %s%s: split race", N, T, scr ? "scr" : "nat", ip ? " ip" : "");
        for (a = 0; a < na; a++) fprintf(stderr, " %s=%.0f", arms[a].name, tns[a]);
        fprintf(stderr, " -> %s\n", arms[best].name);
    }
    vfft_aligned_free(zi); vfft_aligned_free(zo);
    return win;
}

/* ── THE 1D PLAN AT T (2026-11-18, the threading census; the owner: "I agree with the changes") ──────────────────
 * A K=1 create at T > 1 that finds no row at its T (natural order, not a prime cell) no longer copies the one-thread
 * row's recipe and races only that recipe's threaded arms: the recipe is chosen AT T. A near-tie at one thread decides
 * a gap at T, between routes (15625: chain3 ahead of the flat DIT by 9 % on one thread and 2.08x behind it at 8 --
 * chain3 has no threaded form) and between chains of a route (65536: a ZTURN-T chain 0.3 % behind the pick on one
 * thread and 13 % ahead of it at 4 and 8; 177147, 59049, 45045 the flat DIT's, 6 - 13 %); and a candidate with fewer
 * passes over memory or a smaller footprint can lose on one thread and win threaded (the owner). So:
 *   1. the pool ranked at one thread by the planner's own race (_k1_il_plan_race_ex, nothing banked; this process's
 *      last full race of the cell when it just ran): every live candidate of a route that threads and the best of
 *      each route that cannot (dp_planner_il.h fam) -- the one-thread rank filters nothing that threads;
 *   2. each built and bound to its own best threaded form at T -- ZTURN-T serial / blocks / tiles at every width the
 *      tile law admits (vfft_ztt_mt_race), the flat DIT likewise (vfft_ilfd_mt_race) -- gated against the planner's
 *      reference and timed at T as bound; the four-step: every split at T on its own private store, its row cells
 *      warmed into it (_k1fs_mt_split_race); chain3, the pair, the mono: their one-thread form;
 *   3. the fastest VFFT_K1T_FINAL of each route by that time raced against each other at T (same-run, alternated,
 *      unpaced: threaded arms never park);
 *   4. the winner's whole recipe banked on the row at T with its threaded verdict (il_mt, il_mt_tw; the four-step's
 *      split il_mt, il_mtsb and its child at T), which the MT commits then replay. A winner that runs serial banks
 *      nothing: the one-thread row's recipe is the plan at T then, as before (vw2_oop_k1_row_at_T), so serial plans
 *      are not reshuffled by this race's noise.
 * Returns 1 = the row at T banked, 0 = not (the caller copies the one-thread row), -1 = OUT OF MEMORY. */
#define VFFT_K1T_FINAL 3
typedef struct
{
    vfft_il_cand_t c;            /* the recipe (ZTURN-T, flat DIT, chain3, pair, mono) */
    int mt;                      /* its arm at T (ZTURN-T, flat DIT: 0 serial, 1 BLOCKS, 2 TILES) */
    size_t tw;                   /* the TILES arm's width */
    double ns;                   /* its time at T as bound */
    _il_dp_built_t b;            /* built (the bound plan) */
    vfft_k1fs_plan_t *fs;        /* route FS: the four-step at T (owns its store) */
    vfft_ilprime_plan_t *ilp;    /* route PRIME (the prime sides at T, 2026-11-20): the whole-length cell built at T,
                                  * its own race at T, bound to its arm (_ilprime_create_banked) */
    const vfft_real *zi;
    vfft_real *zo;
    int ip;
} _k1t_arm_t;
static void _k1t_arm_run(void *v)
{
    _k1t_arm_t *a = (_k1t_arm_t *)v;
    const vfft_real *in = a->ip ? a->zo : a->zi;
    if (a->fs) { vfft_k1fs_execute(a->fs, VFFT_FORWARD, in, a->zo); return; }
    if (a->ilp)
    {
        if (!(a->ilp->mt > 0 && vfft_ilprime_execute_mt(a->ilp, in, a->zo, 0)))
            vfft_ilprime_execute_fwd(a->ilp, in, a->zo);
        return;
    }
    if (a->b.ztt && a->b.ztt->mt > 0 && vfft_ztt_execute_mt(a->b.ztt, in, a->zo, 0)) return;
    if (a->b.ifd && a->b.ifd->mt > 0 && vfft_ilfd_execute_mt(a->b.ifd, in, a->zo, 0)) return;
    (void)_il_dp_exec_io(&a->c, &a->b, in, a->zo, 0);
}
static void _k1t_arm_free(_k1t_arm_t *a)
{
    if (a->fs) vfft_k1fs_destroy(a->fs);
    else if (a->ilp)
    {   /* the verdict its race at T left for the door (_ilpr_tv) goes with it */
        if (_ilpr_tv.p == a->ilp)
            _ilpr_tv_clear();
        vfft_ilprime_destroy(a->ilp);
    }
    else _il_dp_free(&a->b);
    a->fs = NULL;
    a->ilp = NULL;
    memset(&a->b, 0, sizeof a->b);
}
/* build the arm's recipe in the placement and bind it at T: race = 1 races its threaded form (a->mt, a->tw set);
 * race = 0 binds the form already raced. 0 = it does not build */
static int _k1t_build(_k1t_arm_t *a, int N, int T, int ip, vfft_real *zi, vfft_real *zo, int race)
{
    const size_t nb = (size_t)2 * N * sizeof(vfft_real);
    memset(&a->b, 0, sizeof a->b);
    if (_il_dp_build(N, &a->c, &a->b, ip) != 0)
        return 0;
    if (a->b.ztt)
    {
        vfft_ztt_plan_t *p = a->b.ztt;
        if (race)
        {   /* the tile widths at T with the one-thread width among them (_ztt_mt_replay_or_race's set) */
            size_t cands[VFFT_ZTT_MAX_NF + 1], tw = 0;
            int nc = vfft_ztt_mt_tile_cands(p, cands, VFFT_ZTT_MAX_NF), q, have = 0;
            for (q = 0; q < nc; q++)
                have |= (cands[q] == p->tile);
            if (p->tile && !have && (size_t)N / p->tile >= 2)
                cands[nc++] = p->tile;
            if (ip)
            {
                memcpy(zo, zi, nb);
                vfft_ztt_mt_race(p, T, zo, zo, NULL, cands, nc, &tw);
            }
            else
                vfft_ztt_mt_race(p, T, zi, zo, NULL, cands, nc, &tw);
            a->mt = p->mt;
            a->tw = tw;
        }
        else
        {
            if (a->mt == 2 && a->tw > 0 && !vfft_ztt_set_tile(p, a->tw))
                a->mt = 0;
            if (!vfft_ztt_mt_bind(p, T, a->mt))
                p->mt = 0;
        }
    }
    else if (a->b.ifd)
    {
        vfft_ilfd_plan_t *p = a->b.ifd;
        if (race)
        {   /* out of place: the in-place serving runs the same lists (_ilfd_mt_replay_or_race) */
            int mt_tw = 0;
            vfft_ilfd_mt_race(p, T, p->tw, zi, zo, &mt_tw);
            p->mt_tw = mt_tw;
            a->mt = p->mt;
            a->tw = (size_t)mt_tw;
        }
        else
        {
            p->mt = a->mt;
            p->mt_t = T;
            p->mt_tw = (int)a->tw;
            if (a->mt == 2 && a->tw > 0 && !vfft_ilfd_apply_tw(p, (int)a->tw))
                p->mt = 0;
            if (p->mt > 0 && !vfft_ilfd_mt_bind(p, T))
                p->mt = 0;
        }
    }
    else
        a->mt = 0;
    return 1;
}
/* the arm's forward from the pristine input against the reference spectrum (natural bins): the planner's gate */
static int _k1t_gate(_k1t_arm_t *a, int N, const vfft_real *zorig, const vfft_real *zref, double scale)
{
    const size_t nb = (size_t)2 * N * sizeof(vfft_real);
    double worst = 0.0;
    long k;
    memcpy(a->ip ? a->zo : (vfft_real *)a->zi, zorig, nb);
    _k1t_arm_run(a);
    for (k = 0; k < 2 * (long)N; k += 2)
    {
        const double d = fabs(a->zo[k] - zref[k]) + fabs(a->zo[k + 1] - zref[k + 1]);
        if (!(d < 1e300)) return 0;
        if (d > worst) worst = d;
    }
    return worst / scale <= VFFT_IL_DP_GATE_TOL;
}
/* the arm's time at T as bound: the median of 5 samples of at least 2 ms (the heats' sample), threaded arms unpaced */
static double _k1t_time(_k1t_arm_t *a)
{
    double t[5], t0, one;
    int s, k, reps, i, j;
    _k1t_arm_run(a);
    _k1t_arm_run(a);
    t0 = vfft_now_ns();
    _k1t_arm_run(a);
    one = vfft_now_ns() - t0;
    reps = (int)(VFFT_IL_DP_HEAT_SAMPLE_NS / (one > 1.0 ? one : 1.0));
    if (reps < 2) reps = 2;
    if (reps > 1 << 16) reps = 1 << 16;
    for (s = 0; s < 5; s++)
    {
        t0 = vfft_now_ns();
        for (k = 0; k < reps; k++)
            _k1t_arm_run(a);
        t[s] = (vfft_now_ns() - t0) / reps;
    }
    for (i = 1; i < 5; i++)
        for (j = i; j > 0 && t[j] < t[j - 1]; j--) { const double x = t[j]; t[j] = t[j - 1]; t[j - 1] = x; }
    return t[2];
}
static int _k1_il_t_race(struct vfft_wisdom_s *W, const vfft_cfg_t *cfg, int N, int T)
{
    const int ip = (cfg->placement == VFFT_INPLACE);
    vfft_il_dp_context_t *ctx = &_k1_il_dp_ctx;
    const size_t nb = (size_t)2 * N * sizeof(vfft_real);
    vfft_il_cand_t *list = NULL;
    _k1t_arm_t *cand = NULL, fin[3 * VFFT_K1T_FINAL + 4], *win = NULL;
    vfft_race_arm_t arms[3 * VFFT_K1T_FINAL + 4];
    double ns[3 * VFFT_K1T_FINAL + 4], scale;
    vfft_real *zorig = NULL, *zref = NULL, *zi = NULL, *zo = NULL;
    int nl, nl2, nc = 0, nfin = 0, i, j, threads = 0, best = 0, rc = 0, verbose = getenv("VFFT_NAT_LOG") != NULL;
    int have_prime = 0;
    /* THE PRIME SIDES AT T (owner, 2026-11-20): a composite with a prime factor past the chain radices that splits at T
     * and not at one thread -- [VFFT_K1FS_PRIME_MIN_N, VFFT_K1FS_MIN_N] -- races its four-step's splits at T against
     * the one-thread pool's threaded forms and its whole-length cell at T. Every other prime cell keeps its own arm at
     * T: the four-step's split race above VFFT_K1FS_MIN_N (the lean race banks the four-step there), the prime cell's
     * race at T (_ilprime_create_banked) for a prime. */
    const int primeT = T > 1 && vfft_k1fs_prime_composite(N) && vfft_k1fs_band_t(N, T) && !vfft_k1fs_band_t(N, 1);
    if (T < 2 || !W || W->vw2_off_oop || (vfft_policy_prime_cell(N) && !primeT))
        return 0;
    /* 1. the ranking: this process's last full race of the cell, else the race now */
    if (!(_k1_il_dp_ctx_ready && ctx->fam_N[0] == N && ctx->fam_ip[0] == ip && ctx->fam_role[0] == 0 && ctx->fam_n[0] > 0))
    {
        const int r = _k1_il_plan_race_ex(W, cfg, N, 0);
        if (r < 0) return -1;
        if (!primeT && (r == 0 || !(ctx->fam_N[0] == N && ctx->fam_ip[0] == ip && ctx->fam_role[0] == 0 && ctx->fam_n[0] > 0)))
            return 0;
    }
    /* the prime sides at T go on with an empty ranking (no flat chain at one thread): their own two candidates race */
    nl = (_k1_il_dp_ctx_ready && ctx->fam_N[0] == N && ctx->fam_ip[0] == ip && ctx->fam_role[0] == 0) ? ctx->fam_n[0] : 0;
    for (i = 0; i < nl; i++)
        threads |= (ctx->fam[0][i].route == VFFT_K1_IL_ZTT || ctx->fam[0][i].route == VFFT_K1_IL_FLAT ||
                    ctx->fam[0][i].route == VFFT_K1_IL_FS);
    nl2 = nl + (primeT ? 2 : 0);   /* the prime sides at T: the four-step and the whole-length cell, added */
    if (primeT)
        threads = 1;
    if (!threads)
        return 0;   /* nothing in the pool threads: the one-thread row's recipe is the plan at T */
    /* the candidates and the planner's input and reference, copied before any nested create touches the context */
    if (ctx->ref_N != N && _il_dp_ref_build(ctx, N) != 0)
        return 0;
    scale = ctx->ref_scale;
    list = (vfft_il_cand_t *)calloc((size_t)nl2, sizeof *list);
    cand = (_k1t_arm_t *)calloc((size_t)nl2, sizeof *cand);
    zorig = (vfft_real *)vfft_aligned_alloc(nb);
    zref = (vfft_real *)vfft_aligned_alloc(nb);
    zi = (vfft_real *)vfft_aligned_alloc(nb);
    zo = (vfft_real *)vfft_aligned_alloc(nb);
    if (!list || !cand || !zorig || !zref || !zi || !zo)
    {
        rc = -1;
        goto out;
    }
    memcpy(list, ctx->fam[0], (size_t)nl * sizeof *list);
    if (primeT)
    {   /* the four-step (every split of its ladder at T, below) and the whole-length cell at T */
        list[nl].route = VFFT_K1_IL_FS;
        list[nl + 1].route = VFFT_K1_IL_PRIME;
    }
    memcpy(zorig, ctx->z_orig, nb);
    memcpy(zref, ctx->z_ref, nb);
    memcpy(zi, zorig, nb);
    _vfft_pool_arm(T);
    /* 2. each candidate bound to its threaded form at T, gated, timed; the four-step once, over every split */
    for (i = 0; i < nl2; i++)
    {
        _k1t_arm_t *a = &cand[nc];
        a->c = list[i];
        a->zi = zi;
        a->zo = zo;
        a->ip = ip;
        if (list[i].route == VFFT_K1_IL_PRIME)
        {   /* the whole-length cell (the prime sides at T only; elsewhere a prime cell never enters this race): built
             * at T, its own race at T binding its method, inner and arm, once */
            if (!primeT || have_prime) continue;
            have_prime = 1;
            if (_k1pr_ctx.plan && _k1pr_ctx.N == N)
            {   /* the cell the one-thread race of this create built at T (_k1_il_plan_race_ex keeps it for this race):
                 * taken, never raced at T twice */
                a->ilp = _k1pr_ctx.plan;
                _k1pr_ctx.plan = NULL;
                _k1pr_ctx.N = 0;
            }
            else
                a->ilp = _ilprime_create_banked(W, cfg, N, NULL, T);
            if (!a->ilp) continue;
        }
        else if (list[i].route == VFFT_K1_IL_FS)
        {
            struct vfft_wisdom_s *mS = _k1fs_store_new();
            vfft_cfg_t fscfg = *cfg;
            if (!mS) continue;
            fscfg.wisdom = (vfft_wisdom *)mS;
            fscfg.wisdom_write = 0;
            fscfg.recalibrate = 0;
            fscfg.nthreads = 1;
            _k1_il_fs_warm_t(mS, W, &fscfg, N, T);   /* the candidates' row cells, before the race (a nested create races them) */
            a->fs = _k1fs_mt_split_race(NULL, mS, W, cfg, N, 0, ip, T);
            if (!a->fs) continue;
        }
        else if (!_k1t_build(a, N, T, ip, zi, zo, 1))
            continue;
        memcpy(zi, zorig, nb);
        if (!_k1t_gate(a, N, zorig, zref, scale))
        {
            if (verbose) fprintf(stderr, "[k1-T] N=%d T=%d: candidate %d (route %d) FAILS the gate at T -- dropped\n",
                                 N, T, i, list[i].route);
            _k1t_arm_free(a);
            continue;
        }
        memcpy(zi, zorig, nb);
        a->ns = _k1t_time(a);
        if (!a->fs && !a->ilp)
            _k1t_arm_free(a);   /* rebuilt for the final from its raced form; the four-step and the prime cell stay built */
        nc++;
    }
    /* 3. the fastest VFFT_K1T_FINAL of each route by its time at T, raced against each other */
    {
        int used[VFFT_IL_DP_FAM_MAX + 2];   /* + the prime sides' two added candidates */
        memset(used, 0, sizeof used);
        for (j = 0; j < nc && nfin < (int)(sizeof fin / sizeof fin[0]); j++)
        {
            int k, bi = -1, r;
            for (k = 0; k < nc; k++)
                if (!used[k] && (bi < 0 || cand[k].ns < cand[bi].ns))
                    bi = k;
            if (bi < 0) break;
            used[bi] = 1;
            r = cand[bi].c.route;
            for (k = 0, i = 0; k < nfin; k++) i += (fin[k].c.route == r);
            if (i >= VFFT_K1T_FINAL)
            {
                if (cand[bi].fs || cand[bi].ilp) _k1t_arm_free(&cand[bi]);
                continue;
            }
            fin[nfin] = cand[bi];
            memset(&cand[bi].b, 0, sizeof cand[bi].b);
            cand[bi].fs = NULL;   /* moved into the final */
            cand[bi].ilp = NULL;
            if (!fin[nfin].fs && !fin[nfin].ilp && !_k1t_build(&fin[nfin], N, T, ip, zi, zo, 0))
                continue;
            nfin++;
        }
        for (j = 0; j < nc; j++)
            if (cand[j].fs || cand[j].ilp) _k1t_arm_free(&cand[j]);   /* a four-step or prime cell past the final's places */
    }
    if (nfin == 0)
        goto out;
    for (j = 0; j < nfin; j++)
    {
        arms[j].name = "k1-T";
        arms[j].run = _k1t_arm_run;
        arms[j].ctx = &fin[j];
    }
    {
        _k1fs_mt_rst_t rs;
        double t0, one;
        int reps;
        memcpy(zi, zorig, nb);
        if (ip) memcpy(zo, zorig, nb);
        _k1t_arm_run(&fin[0]);
        t0 = vfft_now_ns();
        _k1t_arm_run(&fin[0]);
        one = vfft_now_ns() - t0;
        reps = (int)(VFFT_IL_DP_HEAT_SAMPLE_NS / (one > 1.0 ? one : 1.0));
        if (reps < VFFT_IL_DP_HEAT_MIN_REPS) reps = VFFT_IL_DP_HEAT_MIN_REPS;
        if (reps > 1 << 16) reps = 1 << 16;
        rs.dst = zo; rs.src = zorig; rs.nb = nb;
        {
            const vfft_race_proto_t proto = { 7, reps, VFFT_RACE_MEDIAN, 1, 2, ip ? _k1fs_mt_reseed : NULL,
                                              ip ? &rs : NULL, 0 };   /* THREADED arms: never paused */
            vfft_race_run(&proto, arms, nfin, ns);
        }
    }
    for (j = 1; j < nfin; j++)
        if (ns[j] < ns[best]) best = j;
    win = &fin[best];
    if (verbose)
    {
        char s[96];
        fprintf(stderr, "[k1-T] N=%d T=%d %s: %d candidate(s) timed at T, final", N, T, ip ? "ip" : "oop", nc);
        for (j = 0; j < nfin; j++)
        {
            if (fin[j].fs) snprintf(s, sizeof s, "fs %dx%d%s", fin[j].fs->N1, fin[j].fs->N2, fin[j].fs->form ? "/sb" : "");
            else if (fin[j].ilp) snprintf(s, sizeof s, "prime %s M=%d", fin[j].ilp->method ? "rader" : "bluestein", fin[j].ilp->M);
            else _il_dp_cand_str(&fin[j].c, s, sizeof s);
            fprintf(stderr, " | %s mt%d=%.0f", s, fin[j].fs ? 0 : fin[j].mt, ns[j]);
        }
        fprintf(stderr, " -> %d\n", best);
    }
    /* 4. the winner's recipe on the row at T, when it threads; the whole-length cell (the prime sides at T) banks its
     * route at T here and goes to the door as the race's own plan (_k1pr_ctx), whose finisher banks its verdict at T
     * (_ilpr_mt_replay_or_race reads _ilpr_tv) */
    if (win->fs || win->ilp || win->mt > 0)
    {
        vfft_oop_il_entry_t e;
        const vw2_rec_t *r;
        const vfft_il_cand_t *c = &win->c;
        memset(&e, 0, sizeof e);
        e.N = N;
        e.K = 1;
        e.kind = VFFT_OOP_KIND_BAILEY2V;
        e.place_ip = ip;
        e.ord_scr = 0;
        e.nthreads = T;
        e.k1_il_route = win->fs ? VFFT_K1_IL_FS : win->ilp ? VFFT_K1_IL_PRIME : c->route;
        e.il_R1 = win->fs ? win->fs->N1 : win->ilp ? 0 : c->R1;
        e.il_R2 = win->fs ? win->fs->N2 : win->ilp ? 0 : c->R2;
        e.il_kv = win->fs ? win->fs->form : win->ilp ? 0 : c->il_kv;
        e.il_kv_raced = 1;
        if (!win->fs && c->route == VFFT_K1_IL_FLAT)
        {
            memcpy(e.il_fl, c->il_fl, sizeof e.il_fl);
            e.il_fl_n = c->il_fl_n;
            memcpy(e.il_flf, c->il_flf, sizeof e.il_flf);
            e.il_tw = c->il_tw;
            memcpy(e.il_rd, c->il_rd, sizeof e.il_rd);
            e.il_rd_n = c->il_rd_n;
            e.il_rdw = c->il_rdw;
        }
        if (!win->fs && c->route == VFFT_K1_IL_ZTT)
        {
            memcpy(e.il_zt, c->il_zt, sizeof e.il_zt);
            e.il_zt_n = c->il_zt_n;
            e.il_tw = c->il_tw;
        }
        e.ns = ns[best];
        if (vw2_oop_bank_k1_il(&W->vw2, &e) == VW2_OK &&
            (r = vw2__oop_k1_scan_pl(&W->vw2, N, VW2_LAY_IL, 0, ip ? VW2_PL_IP : VW2_PL_OOP, T)) != NULL)
        {
            const vw2_key_t key = r->key;   /* an update may move the record */
            char b[48];
            int ok = 1;
            if (win->ilp)
            {   /* the plan to the door, its verdict at T banked there */
                _k1pr_release();
                _k1pr_ctx.plan = win->ilp;
                _k1pr_ctx.N = N;
                win->ilp = NULL;
            }
            else if (win->fs)
            {
                snprintf(b, sizeof b, "%d", win->fs->N1);
                ok = ok && vw2_update_field(&W->vw2, &key, "il_mt", b) == VW2_OK;
                _k1fs_chain_str(win->fs, b, sizeof b);
                ok = ok && vw2_update_field(&W->vw2, &key, "il_mtsb", b) == VW2_OK;
                r = vw2__oop_k1_scan_pl(&W->vw2, N, VW2_LAY_IL, 0, ip ? VW2_PL_IP : VW2_PL_OOP, T);
                ok = ok && r && _k1fs_row_put_child(&W->vw2, r, win->fs->own, win->fs->N1, win->fs->N2, ip, T) == VW2_OK;
            }
            else
            {
                snprintf(b, sizeof b, "%d", win->mt);
                ok = ok && vw2_update_field(&W->vw2, &key, "il_mt", b) == VW2_OK;
                snprintf(b, sizeof b, "%zu", win->mt == 2 ? win->tw : (size_t)0);
                ok = ok && vw2_update_field(&W->vw2, &key, "il_mt_tw", b) == VW2_OK;
            }
            if (ok)
            {
                _vw2_persist(W, cfg);
                rc = 1;
            }
            else
                fprintf(stderr, "vfft: the 1D plan at T=%d for N=%d NOT banked -- the one-thread recipe serves\n", T, N);
        }
    }
out:
    for (j = 0; j < nfin; j++)
        _k1t_arm_free(&fin[j]);
    if (cand)
        for (j = 0; j < nl2; j++)
            if (cand[j].fs || cand[j].ilp) _k1t_arm_free(&cand[j]);
    free(list);
    free(cand);
    vfft_aligned_free(zorig); vfft_aligned_free(zref); vfft_aligned_free(zi); vfft_aligned_free(zo);
    return rc;
}

static void _k1fs_mt_replay_or_race(struct vfft_plan_s *h,
                                    struct vfft_wisdom_s *W,
                                    const vfft_cfg_t *cfg, int N)
{
    vfft_k1fs_plan_t *p = h->k1fs;
    const int T = h->nthreads;
    const int ip = (h->placement == VFFT_INPLACE);
    const char *tok_v = "il_mt";   /* the plan's own row: placement and thread count are its key (v1.3) */
    const char *tok_s = "il_mtsb";
    const vw2_rec_t *r = NULL;
    int n1[8], n2[8], ns, i;
    if (!p || T < 2 || getenv("VFFT_K1_FS") || getenv("VFFT_K1_FSSB"))
        return;
    if (W && !W->vw2_off_oop)
        r = vw2__oop_k1_scan_pl(&W->vw2, N, VW2_LAY_IL, p->scr,
                                h->placement == VFFT_INPLACE ? VW2_PL_IP : VW2_PL_OOP, T);   /* the plan's own row (v1.3) */
    ns = vfft_k1fs_splits(N, n1, n2, 8);
    if (r && !cfg->recalibrate && vw2_rec_get(r, tok_v) && _k1fs_row_has_child(r))   /* the verdict AND its child at T */
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
            int grew = 0;
            vfft_k1fs_plan_t *q = _k1fs_build_own(N, n1[i], n2[i], p->scr, cfg, ip, T, form, ch, cn, r, &grew);
            if (q && grew)
                fprintf(stderr, "vfft: the four-step's banked child at N=%d T=%d (%dx%d) did not replay -- it raced in "
                                "the plan's private store\n", N, T, n1[i], n2[i]);
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
         * on 64-B aligned scratch, the plan's placement; the candidates'
         * children race into ONE private store (a copy of the plan's own,
         * k1_fourstep.h) that the winner keeps (_k1fs_mt_split_race) */
        struct vfft_wisdom_s *mS = p->own ? _k1fs_store_dup(p->own) : _k1fs_store_new();
        vfft_k1fs_plan_t *w = _k1fs_mt_split_race(p, mS, W, cfg, N, p->scr, ip, T);
        if (!w)
            return;
        h->k1fs = p = w;
    }
    if (r && W && !W->vw2_off_oop)
    {
        char b[48];
        int ok = 1;
        snprintf(b, sizeof b, "%d", p->N1);
        ok = ok && vw2_update_field(&W->vw2, &r->key, tok_v, b) == VW2_OK;
        _k1fs_chain_str(p, b, sizeof b);
        ok = ok && vw2_update_field(&W->vw2, &r->key, tok_s, b) == VW2_OK;
        /* and the served plan's child at T (re-banks the row: r is stale after) */
        ok = ok && _k1fs_row_put_child(&W->vw2, r, p->own, p->N1, p->N2, ip, T) == VW2_OK;
        if (ok)
            _vw2_persist(W, cfg);
        else
            fprintf(stderr, "vfft: the four-step's threaded verdict at N=%d T=%d NOT banked -- the plan races again\n", N, T);
    }
}

/* ── the IN-PLACE mono candidate ──
 * Served when the cell's kind-3 row (already planned by _k1_il_candidate's
 * race on a miss) says MONO: the row's form's in-place kernel, both
 * directions (vfft_k1_mono_il_ip_fn: form 0 and mono64 map onto the ONE
 * alias-safe solo, n1c(N); an FP32 fused form runs the in-place kernel
 * its registry entry names, the one the in-place race timed). Returns 1 and
 * fills the pair when served, 0 when the row is not a MONO verdict, -1 when
 * it is one and its kernel is missing (the row is empty); nothing is built
 * or left to free in either case. */
static int _k1_il_mono_candidate(struct vfft_wisdom_s *W, const vfft_cfg_t *cfg,
                                 int N, vfft_oop11_fn *ilf, vfft_oop11_fn *ilb)
{
    vfft_oop_il_entry_t keb;
    const vfft_oop_il_entry_t *ke;
    /* the REQUEST's order and placement cell: this is the in-place door's
     * candidate, so it reads the place=ip row the in-place race banked (an
     * explicit SCRAMBLED request reads its ord=scr row). */
    const int scr_req = (vfft_policy_ord_k1(cfg, N, 1) == VW2_ORD_SCR);
    *ilf = *ilb = 0;
    if (!W || W->vw2_off_oop) return 0;
    ke = vw2_oop_lookup_k1_il_cell(&W->vw2, N, scr_req, 1, _vfft_plan_threads(cfg), &keb) ? &keb : NULL;
    if (!ke || ke->k1_il_route != VFFT_K1_IL_MONO) return 0;
    *ilf = vfft_k1_mono_il_ip_fn(N, ke->il_kv, 0);
    *ilb = vfft_k1_mono_il_ip_fn(N, ke->il_kv, 1);
    if (!*ilf || !*ilb) { *ilf = *ilb = 0; return -1; }
    return 1;
}


#endif /* VFFT_IL_K1_COMMIT_H */
