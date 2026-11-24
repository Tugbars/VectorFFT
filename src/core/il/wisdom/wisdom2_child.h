/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Tuğbars Heptaşkın */
/* wisdom2_child.h — A CHILD PLAN IN ROLE (owner, 2026-10-02; the 2D and 3D
 * tiers 2026-10-05).
 *
 * A plan a 2D or 3D create builds for one of its own passes -- the row plan
 * at N2, the turn plan at N1, the skewed pass's row plan, the turned prime
 * column plan, the 3D tier's 2D plane plan and its row plan -- is a COMPONENT
 * of that cell, not a cell of its own: it is created against a PRIVATE store
 * (in memory, never saved), races there when the parent's row carries no
 * recipe for it, and every row it banks rides on the parent's row under a
 * prefix. A 2D or 3D create never reads or writes a 1D row, and a 1D row never
 * serves a 2D plan's pass (docs/design/wisdom_system.md §7).
 *
 * THE TOKENS. Row i (1-based, the store's order) of the child's store lands on
 * the parent row as
 *     <pre>k<i>=<its key, spaces as commas>   rp_k1=t=c2c,n=48,q=1,ord=nat,place=ip,role=comp,lay=il
 *     <pre><i>_<name>=<value>                 one per payload token, e.g. rp_1_il_route=ztt
 * so whatever the child banked -- its route row, a backward twin, a prime
 * method's row, a four-step child's fs_ tokens -- comes back whole, and a 2D
 * row that carries its own children nests under a 3D row's plane_ prefix with
 * no special case. The prefixes: rp_ (the row plan; "row_" would collide with
 * the four-step child's fs_row_ tokens), turn_, csk_, tpc_, plane_; and dst_ and
 * dct_ (2026-11-14): the 1D real FFT inside a real-to-real cell -- dst_ in a
 * DST-II or DST-III cell (il/real/dst.h) and in a DST-I cell, dct_ in a DCT-I
 * cell (il/real/dct1_dst1.h) -- the same law one rank down: a real-to-real
 * create never reads or writes an r2c or a c2r row (il/real/real_child.h).
 *
 * THE FLOW. The parent create seeds the child's store from its own row
 * (nothing when the row carries no tokens under the prefix), creates the child
 * against it with nothing persisted and the parent's recalibrate flag, and at
 * its end puts the store's rows back on its row when a row of the store was
 * banked by this process (the child raced) or the row carried no recipe yet;
 * the parent row is re-banked and saved like any winner. A child's clones are
 * created against the same store, so they replay the child's recipe.
 */
#ifndef VFFT_IL_WISDOM2_CHILD_H
#define VFFT_IL_WISDOM2_CHILD_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common/wisdom/wisdom2.h"
#include "common/wisdom/wisdom2_rank_rows.h"   /* vw2__2d_stamp_date: the shortlist record's date */
#include "vfft_internal.h"

static struct vfft_wisdom_s *vfft_child_store_new(void)
{
    struct vfft_wisdom_s *S = (struct vfft_wisdom_s *)calloc(1, sizeof *S);
    if (S)
        S->vw2.writable = 1;   /* banks land in memory: no directory, no save */
    return S;
}
static void vfft_child_store_free(struct vfft_wisdom_s *S)
{
    if (S)
        _vfft_wisdom_free((vfft_wisdom *)S);
}

/* is `name` a child token under pre: <pre>k<i> (rest = NULL) or <pre><i>_<rest> */
static int vfft__child_tok(const char *name, const char *pre, size_t lp, int *idx, const char **rest)
{
    const char *p;
    char *end;
    long i;
    if (strncmp(name, pre, lp))
        return 0;
    p = name + lp;
    if (*p == 'k' && p[1] >= '1' && p[1] <= '9')
    {
        i = strtol(p + 1, &end, 10);
        if (*end || i <= 0 || i > 4096)
            return 0;
        *idx = (int)i;
        *rest = NULL;
        return 1;
    }
    if (*p < '1' || *p > '9')
        return 0;
    i = strtol(p, &end, 10);
    if (*end != '_' || !end[1] || i <= 0 || i > 4096)
        return 0;
    *idx = (int)i;
    *rest = end + 1;
    return 1;
}

/* 1 = the parent row carries a recipe under pre */
static int vfft_child_row_has(const vw2_rec_t *parent, const char *pre)
{
    const size_t lp = strlen(pre);
    const char *rest;
    int i, idx;
    if (!parent)
        return 0;
    for (i = 0; i < parent->ntok; i++)
        if (parent->tok[i].sect == 1 && vfft__child_tok(parent->tok[i].name, pre, lp, &idx, &rest) && !rest)
            return 1;
    return 0;
}

/* The child's store, seeded from the parent row's tokens under pre: an empty
 * store when the row is NULL or carries none (the child races), and when the
 * recipe does not parse (dropped whole: the child races). Seeded rows are not
 * "raced". NULL only when memory runs out. */
static struct vfft_wisdom_s *vfft_child_store_from_row(const vw2_rec_t *parent, const char *pre)
{
    const size_t lp = strlen(pre);
    struct vfft_wisdom_s *S = vfft_child_store_new();
    const char *rest;
    int i, k, n = 0, idx, ok = 1;
    if (!S || !parent)
        return S;
    for (i = 0; i < parent->ntok; i++)
        if (parent->tok[i].sect == 1 && vfft__child_tok(parent->tok[i].name, pre, lp, &idx, &rest) && idx > n)
            n = idx;
    for (k = 1; k <= n && ok; k++)
    {
        vw2_rec_t r;
        char kb[320];
        int have_key = 0;
        memset(&r, 0, sizeof r);
        for (i = 0; i < parent->ntok && ok; i++)
        {
            if (parent->tok[i].sect != 1 || !vfft__child_tok(parent->tok[i].name, pre, lp, &idx, &rest) || idx != k)
                continue;
            if (!rest)
            {   /* the key: its commas back to spaces */
                size_t j;
                snprintf(kb, sizeof kb, "%s", parent->tok[i].val);
                for (j = 0; kb[j]; j++)
                    if (kb[j] == ',') kb[j] = ' ';
                have_key = (vw2__key_parse(kb, &r.key) == 1);
            }
            else if (vw2_rec_set(&r, 1, rest, parent->tok[i].val) != VW2_OK)
                ok = 0;
        }
        if (!have_key || !ok || vw2_bank(&S->vw2, &r) != VW2_OK)
        {
            vw2_rec_free(&r);
            ok = 0;
        }
    }
    if (!ok)
    {
        vfft_child_store_free(S);
        return vfft_child_store_new();
    }
    vw2_disown(&S->vw2);   /* seeded, not raced */
    return S;
}

/* the parent row at key (exact), NULL when there is none */
static const vw2_rec_t *vfft_child_parent_row(const vw2_store_t *st, const vw2_key_t *key)
{
    int i;
    if (!st)
        return NULL;
    for (i = 0; i < st->nrec; i++)
        if (vw2_key_eq(&st->rec[i].key, key))
            return &st->rec[i];
    return NULL;
}
/* the child's store for the parent at key, seeded from that row */
static struct vfft_wisdom_s *vfft_child_store_for(const vw2_store_t *st, const vw2_key_t *key, const char *pre)
{
    return vfft_child_store_from_row(vfft_child_parent_row(st, key), pre);
}

/* 1 = a row of the store was banked by this process: the child raced */
static int vfft_child_store_raced(const struct vfft_wisdom_s *S)
{
    int i;
    if (!S)
        return 0;
    for (i = 0; i < S->vw2.nrec; i++)
        if (S->vw2.rec[i].own)
            return 1;
    return 0;
}

/* the store's rows as tokens of nr under pre (the child tokens above); 1 = all set */
static int vfft__child_rows_onto(vw2_rec_t *nr, const char *pre, const struct vfft_wisdom_s *S)
{
    char nm[128], kb[320];
    int i, t;
    for (i = 0; i < S->vw2.nrec; i++)
    {
        const vw2_rec_t *r = &S->vw2.rec[i];
        size_t j;
        vw2__key_format(&r->key, kb, sizeof kb);
        for (j = 0; kb[j]; j++)
            if (kb[j] == ' ') kb[j] = ',';
        if (snprintf(nm, sizeof nm, "%sk%d", pre, i + 1) >= (int)sizeof nm || vw2_rec_set(nr, 1, nm, kb) != VW2_OK)
            return 0;
        for (t = 0; t < r->ntok; t++)
        {
            if (r->tok[t].sect != 1)
                continue;
            if (snprintf(nm, sizeof nm, "%s%d_%s", pre, i + 1, r->tok[t].name) >= (int)sizeof nm ||
                vw2_rec_set(nr, 1, nm, r->tok[t].val) != VW2_OK)
                return 0;
        }
    }
    return 1;
}

/* The store's rows onto the parent row under pre, the row's older tokens
 * under pre replaced; the parent row is re-banked (pointers into the store
 * are stale after). Done only when the child raced or the row carried no
 * recipe; 1 = the row changed. */
static int vfft_child_row_update(vw2_store_t *st, const vw2_key_t *key, const char *pre,
                                 const struct vfft_wisdom_s *S)
{
    const size_t lp = strlen(pre);
    const vw2_rec_t *row;
    const char *rest;
    vw2_rec_t nr;
    int i, idx;
    if (!S || !st || S->vw2.nrec == 0)
        return 0;
    row = vfft_child_parent_row(st, key);
    if (!row)
        return 0;
    if (!vfft_child_store_raced(S) && vfft_child_row_has(row, pre))
        return 0;
    memset(&nr, 0, sizeof nr);
    nr.key = row->key;
    for (i = 0; i < row->ntok; i++)
    {
        if (row->tok[i].sect == 1 && vfft__child_tok(row->tok[i].name, pre, lp, &idx, &rest))
            continue;
        if (vw2_rec_set(&nr, row->tok[i].sect, row->tok[i].name, row->tok[i].val) != VW2_OK)
            goto fail;
    }
    if (!vfft__child_rows_onto(&nr, pre, S))
        goto fail;
    if (vw2_bank(st, &nr) != VW2_OK)
        goto fail;
    return 1;
fail:
    vw2_rec_free(&nr);
    fprintf(stderr, "[wisdom2] a child's recipe (%s) could not be put on its parent's row: "
                    "that child races again at the next create\n", pre);
    return 0;
}

/* ── THE SHORTLIST (owner, 2026-11-24) ────────────────────────────────────
 * The 1D plan a 2D plan runs as one of its passes -- the row plan at N2, the
 * turn plan at N1, the skewed pass's row plan, the turned prime column plan --
 * is raced as a 1D transform, and that race ranks its candidates the same on
 * every plane with that length. It runs ONCE per length and placement: the
 * first plans of its ranking are kept as one record, and each plane then
 * times those candidates as its own pass, on its whole transform at its
 * thread count (il/rank2/il2d_tier.h, _il2d_child_final), and banks the
 * winner on its row under the child's prefix as before. Measured on 13 planes
 * (rows of 150 to 4374 points): the 1D race's winner is 0-3 % behind the best
 * candidate in the plane at one thread and 0-6 % at eight; the best is among
 * the ranking's first 4 at one thread and within 1 % of its first 8 at eight.
 *
 * THE RECORD. Key: t=c2c n=<N> q=1 ord=nat place=ip|oop role=shortlist lay=il
 * (nthreads=<T> on a real row plan's, a batch created at the plane's T)
 * -- the 1D cell the candidates were raced as, under a role of its own and in
 * the 2D shard (wisdom2.h vw2_shard_route): no 1D request reads it, and a 2D
 * create still reads no 1D row. t=r2c / t=c2r: the 2D real tier's row plan,
 * the batch of N1 real transforms at N2 -- its one candidate is the 1D real
 * cell's recipe (the store's q = 1 rows; the batch's own row is each plane's).
 * x0_ / x1_ on a real record: the recipe of the 2D real tier's zr2c ROW ENGINE at
 * route 0 / route 1 -- its complex child raced in the real role
 * (il/rank2/il2d_real_plan.h, _il2d_rowx_zr2c): the row engine race times its
 * engines in every plane, and builds them from these. Payload: nc=<count> and, for candidate
 * c = 1..nc in the race's order, that candidate's whole recipe as a child's
 * rows under the prefix c<c>_ (c1_k1=<key>, c1_1_il_route=...). Candidate 1 is
 * the race's winner with everything its create banked; the others carry their
 * route row and, for a pair or a three-stage chain, their own raced backward
 * forms (il/rank1/k1_commit.h, _k1_il_shortlist_stores).
 *
 * Only a store that names its build keeps one (a real store: vw2.isa != 0).
 * A part plan's private store -- a 2D plane inside a 3D plan, a four-step's
 * child -- has none, and its children race as they did. */
#define VFFT_SHORTLIST_MAX 8

/* 1 = a row of the store's ONE-TRANSFORM cells (q = 1) was banked by this process: the 1D plan raced. A batch child's
 * store also holds its batch row (q = the batch), raced per plane and no part of the shortlist. */
static int vfft_child_store_raced_q1(const struct vfft_wisdom_s *S)
{
    int i;
    if (!S)
        return 0;
    for (i = 0; i < S->vw2.nrec; i++)
        if (S->vw2.rec[i].own && S->vw2.rec[i].key.q == 1)
            return 1;
    return 0;
}
/* the store's q = 1 rows as a store of their own (seeded, not raced); NULL when it has none or memory runs out */
static struct vfft_wisdom_s *vfft_child_store_q1(const struct vfft_wisdom_s *S)
{
    struct vfft_wisdom_s *C;
    int i, t, n = 0;
    if (!S || !(C = vfft_child_store_new()))
        return NULL;
    for (i = 0; i < S->vw2.nrec; i++)
    {
        const vw2_rec_t *r = &S->vw2.rec[i];
        vw2_rec_t nr;
        int ok = 1;
        if (r->key.q != 1)
            continue;
        memset(&nr, 0, sizeof nr);
        nr.key = r->key;
        for (t = 0; t < r->ntok && ok; t++)
            if (r->tok[t].sect == 1 && vw2_rec_set(&nr, 1, r->tok[t].name, r->tok[t].val) != VW2_OK)
                ok = 0;
        if (!ok || vw2_bank(&C->vw2, &nr) != VW2_OK)
        {
            vw2_rec_free(&nr);
            vfft_child_store_free(C);
            return NULL;
        }
        n++;
    }
    if (!n)
    {
        vfft_child_store_free(C);
        return NULL;
    }
    vw2_disown(&C->vw2);
    return C;
}

/* t: VW2_T_C2C, or VW2_T_R2C / VW2_T_C2R (a real row plan); T: the thread count the child plan is created at (a c2c
 * child: one; the real row batch: the plane's, whose one-transform rows are raced and keyed at that count) */
static void vfft_shortlist_key(int t, int N, int inplace, int T, vw2_key_t *k)
{
    memset(k, 0, sizeof *k);
    k->t = (uint8_t)t;
    k->nthreads = (uint8_t)(T > 1 ? T : 0);
    k->rank = 1;
    k->n[0] = N;
    k->q = 1;
    k->ord = VW2_ORD_NAT;
    k->pl = inplace ? VW2_PL_IP : VW2_PL_OOP;
    k->role = VW2_ROLE_SHORTLIST;
    k->lay = VW2_LAY_IL;
}
/* how many candidates the record at (N, placement) names; 0 = no record */
static int vfft_shortlist_count(const vw2_store_t *st, int t, int N, int inplace, int T)
{
    const vw2_rec_t *r;
    const char *v;
    vw2_key_t k;
    int n;
    if (!st)
        return 0;
    vfft_shortlist_key(t, N, inplace, T, &k);
    r = vfft_child_parent_row(st, &k);
    v = r ? vw2_rec_get(r, "nc") : NULL;
    n = v ? atoi(v) : 0;
    return n < 0 ? 0 : n > VFFT_SHORTLIST_MAX ? VFFT_SHORTLIST_MAX : n;
}
/* candidate c (1-based) as a private store a create replays; NULL when the record names no such candidate, or its
 * recipe does not parse */
static struct vfft_wisdom_s *vfft_shortlist_get(const vw2_store_t *st, int t, int N, int inplace, int T, int c)
{
    struct vfft_wisdom_s *S;
    vw2_key_t k;
    char pre[16];
    if (c < 1 || c > vfft_shortlist_count(st, t, N, inplace, T))
        return NULL;
    vfft_shortlist_key(t, N, inplace, T, &k);
    snprintf(pre, sizeof pre, "c%d_", c);
    S = vfft_child_store_from_row(vfft_child_parent_row(st, &k), pre);
    if (S && S->vw2.nrec == 0)
    {
        vfft_child_store_free(S);
        S = NULL;
    }
    return S;
}
/* the record at (t, N, placement) exists: an empty one (nc=0) is banked where the store has none; 1 = it is there */
static int vfft_shortlist_ensure(vw2_store_t *st, int t, int N, int inplace, int T)
{
    vw2_rec_t nr;
    vw2_key_t k;
    if (!st)
        return 0;
    vfft_shortlist_key(t, N, inplace, T, &k);
    if (vfft_child_parent_row(st, &k))
        return 1;
    memset(&nr, 0, sizeof nr);
    nr.key = k;
    if (vw2_rec_set(&nr, 1, "nc", "0") != VW2_OK || vw2_rec_set(&nr, 2, "src", "race") != VW2_OK)
    {
        vw2_rec_free(&nr);
        return 0;
    }
    vw2__2d_stamp_date(&nr);
    if (vw2_bank(st, &nr) != VW2_OK)
    {
        vw2_rec_free(&nr);
        return 0;
    }
    return 1;
}
/* the candidates' stores S[0..n) as the record at (N, placement), an older record replaced; 1 = banked (the caller
 * saves the store) */
static int vfft_shortlist_put(vw2_store_t *st, int t, int N, int inplace, int T, struct vfft_wisdom_s *const *S, int n)
{
    vw2_rec_t nr;
    char pre[16], v[16];
    int c;
    if (!st || n < 1)
        return 0;
    if (n > VFFT_SHORTLIST_MAX)
        n = VFFT_SHORTLIST_MAX;
    memset(&nr, 0, sizeof nr);
    vfft_shortlist_key(t, N, inplace, T, &nr.key);
    snprintf(v, sizeof v, "%d", n);
    if (vw2_rec_set(&nr, 1, "nc", v) != VW2_OK)
        goto fail;
    for (c = 0; c < n; c++)
    {
        snprintf(pre, sizeof pre, "c%d_", c + 1);
        if (!S[c] || S[c]->vw2.nrec == 0 || !vfft__child_rows_onto(&nr, pre, S[c]))
            goto fail;
    }
    if (vw2_rec_set(&nr, 2, "src", "race") != VW2_OK)
        goto fail;
    vw2__2d_stamp_date(&nr);
    if (vw2_bank(st, &nr) != VW2_OK)
        goto fail;
    return 1;
fail:
    vw2_rec_free(&nr);
    fprintf(stderr, "[wisdom2] the shortlist at N=%d could not be banked: the next plane with that length races it again\n", N);
    return 0;
}

#endif /* VFFT_IL_WISDOM2_CHILD_H */
