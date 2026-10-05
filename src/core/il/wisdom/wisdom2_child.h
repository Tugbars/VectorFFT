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
 * the four-step child's fs_row_ tokens), turn_, csk_, tpc_, plane_.
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
        vfft_wisdom_free((vfft_wisdom *)S);
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
    char nm[128], kb[320];
    int i, t, idx;
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
    for (i = 0; i < S->vw2.nrec; i++)
    {
        const vw2_rec_t *r = &S->vw2.rec[i];
        size_t j;
        vw2__key_format(&r->key, kb, sizeof kb);
        for (j = 0; kb[j]; j++)
            if (kb[j] == ' ') kb[j] = ',';
        if (snprintf(nm, sizeof nm, "%sk%d", pre, i + 1) >= (int)sizeof nm || vw2_rec_set(&nr, 1, nm, kb) != VW2_OK)
            goto fail;
        for (t = 0; t < r->ntok; t++)
        {
            if (r->tok[t].sect != 1)
                continue;
            if (snprintf(nm, sizeof nm, "%s%d_%s", pre, i + 1, r->tok[t].name) >= (int)sizeof nm ||
                vw2_rec_set(&nr, 1, nm, r->tok[t].val) != VW2_OK)
                goto fail;
        }
    }
    if (vw2_bank(st, &nr) != VW2_OK)
        goto fail;
    return 1;
fail:
    vw2_rec_free(&nr);
    fprintf(stderr, "[wisdom2] a child's recipe (%s) could not be put on its parent's row: "
                    "that child races again at the next create\n", pre);
    return 0;
}

#endif /* VFFT_IL_WISDOM2_CHILD_H */
