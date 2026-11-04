/* wisdom2_zts_split.h — the SPLIT library's zturn-ts wisdom: the cell rows a
 * split 1D c2c natural-order request is served from, and their codec.
 *
 * THE CELL is the split natural row the stride verdicts already use:
 *   @cell t=c2c n=N q=K ord=nat place=ip|oop lay=split
 * and the legend's eng-on-nat rule names the WINNING family on it:
 *   eng=zts chain=8.8.4.4 [tile=1024]   zturn-ts serves (lanes by q: 1 or 4)
 *   eng=stride mode=... chain=...       the stride natural engine (its codec,
 *                                       wisdom2_stride_reader.h, reads it)
 *   eng=k1                              the kind-3 split route (q=1 place=oop;
 *                                       the route itself lives on the kind-3
 *                                       role=comp row, this row only records
 *                                       that zturn-ts lost the cell's race)
 * A contiguous batch of K transforms is served from the q=1 cell of its
 * placement (K independent K=1 transforms); the lane-major K=4 batch is the
 * q=4 cell. Every family's reader gates on its own eng=, so a cell held by
 * another family reads as absent to the others and as HELD here: the create
 * races only an UNBANKED cell (feedback: race only when the banked cell is
 * missing, or under recalibrate).
 *
 * Reads and writes only lay=split rows. The split library's own; the IL
 * library's ZTURN-T rows (lay=il, il_ztt=) are a different cell. */
#ifndef VFFT_WISDOM2_ZTS_SPLIT_H
#define VFFT_WISDOM2_ZTS_SPLIT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common/wisdom/wisdom2.h"
#include "split/rank1/zts.h"

typedef struct
{
    int N;
    size_t K;
    int place_ip;
    int zts;                          /* 1 = eng=zts: chain / tile decoded; 0 = the cell is HELD
                                       * by another family (eng below)                            */
    char eng[16];
    int nf;
    int chain[VFFT_ZTS_MAX_NF];
    size_t tile;
    double ns;
} vfft_zts_entry_t;

static inline void _vw2_zts_key(vw2_key_t *k, int N, size_t K, int place_ip)
{
    memset(k, 0, sizeof *k);
    k->t = VW2_T_C2C;
    k->rank = 1;
    k->n[0] = N;
    k->q = (int64_t)K;
    k->ord = VW2_ORD_NAT;
    k->pl = place_ip ? VW2_PL_IP : VW2_PL_OOP;
    k->lay = VW2_LAY_SPLIT;
}

/* "8.8.4.4" -> ints in {4, 8}; the count, or 0 when the token is not a chain */
static inline int _vw2_zts_chain(const char *s, int *out, int cap)
{
    int n = 0;
    if (!s || !*s) return 0;
    while (*s)
    {
        char *end;
        long v = strtol(s, &end, 10);
        if (end == s || n >= cap) return 0;
        out[n++] = (int)v;
        s = end;
        if (*s == '.') s++;
        else if (*s) return 0;
    }
    return n;
}

/* 1 = a row holds the cell (e->zts tells whose), 0 = the cell is unbanked.
 * The lay=split row only: a lay-less vintage row is a stride verdict of the
 * pre-1.2 store, which the stride reader serves through its own fallback. */
static inline int vw2_zts_lookup(const vw2_store_t *s, int N, size_t K, int place_ip,
                                 vfft_zts_entry_t *e)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    const char *eng, *ns;
    _vw2_zts_key(&k, N, K, place_ip);
    r = vw2_lookup(s, &k);
    if (!r || r->key.lay != VW2_LAY_SPLIT) return 0;
    eng = vw2_rec_get(r, "eng");
    if (!eng) return 0;
    memset(e, 0, sizeof *e);
    e->N = N;
    e->K = K;
    e->place_ip = place_ip;
    snprintf(e->eng, sizeof e->eng, "%s", eng);
    ns = vw2_rec_get(r, "ns");
    e->ns = ns ? atof(ns) : 0.0;
    if (strcmp(eng, "zts")) return 1;              /* held by another family */
    e->nf = _vw2_zts_chain(vw2_rec_get(r, "chain"), e->chain, VFFT_ZTS_MAX_NF);
    if (e->nf < 2) return 1;                       /* an undecodable zts row: held, not served */
    {
        const char *t = vw2_rec_get(r, "tile");
        e->tile = t ? (size_t)atol(t) : 0;
    }
    e->zts = 1;
    return 1;
}

static inline int _vw2_zts_tail(vw2_rec_t *r, size_t ran, double ns)
{
    char b[48];
    snprintf(b, sizeof b, "%lld", (long long)ran);
    if (vw2_rec_set(r, 2, "ran", b) != VW2_OK) return -1;
    if (ns > 0.0)
    {
        snprintf(b, sizeof b, "%.1f", ns);
        if (vw2_rec_set(r, 2, "ns", b) != VW2_OK) return -1;
        if (vw2_rec_set(r, 2, "metric", "fwd1") != VW2_OK) return -1;
        if (vw2_rec_set(r, 2, "units", "ns") != VW2_OK) return -1;
    }
    if (vw2_rec_set(r, 2, "src", "race") != VW2_OK) return -1;
    {   /* the day of the race, the merge law's date rule */
        time_t now = time(NULL);
        struct tm *tmv = localtime(&now);
        if (tmv)
        {
            snprintf(b, sizeof b, "%04d-%02d-%02d", tmv->tm_year + 1900, tmv->tm_mon + 1, tmv->tm_mday);
            if (vw2_rec_set(r, 2, "date", b) != VW2_OK) return -1;
        }
    }
    return 0;
}

/* bank the cell as zturn-ts's: the chain, the tile, the measured forward */
static inline int vw2_zts_bank(vw2_store_t *s, int N, size_t K, int place_ip,
                               const int *chain, int nf, size_t tile, double ns)
{
    vw2_rec_t r;
    char b[64];
    int i, off = 0;
    memset(&r, 0, sizeof r);
    _vw2_zts_key(&r.key, N, K, place_ip);
    if (nf < 2 || nf > VFFT_ZTS_MAX_NF) return -1;
    for (i = 0; i < nf; i++)
        off += snprintf(b + off, sizeof b - (size_t)off, "%s%d", i ? "." : "", chain[i]);
    if (vw2_rec_set(&r, 1, "eng", "zts") != VW2_OK ||
        vw2_rec_set(&r, 1, "chain", b) != VW2_OK)
        goto refuse;
    if (tile)
    {
        snprintf(b, sizeof b, "%zu", tile);
        if (vw2_rec_set(&r, 1, "tile", b) != VW2_OK) goto refuse;
    }
    if (_vw2_zts_tail(&r, K, ns)) goto refuse;
    if (vw2_bank(s, &r) != VW2_OK) goto refuse;
    return 0;
refuse:
    vw2_rec_free(&r);
    fprintf(stderr, "[wisdom2] zts bank refused (N=%d q=%zu %s)\n", N, K, place_ip ? "ip" : "oop");
    return -1;
}

/* bank the cell as HELD by another family after zturn-ts lost its race there
 * (eng = that family's name; the family's own verdict lives in its own
 * grammar — for eng=stride on the same key, the stride bank that already ran
 * is the record and this is never called) */
static inline int vw2_zts_bank_held(vw2_store_t *s, int N, size_t K, int place_ip,
                                    const char *eng, double ns)
{
    vw2_rec_t r;
    memset(&r, 0, sizeof r);
    _vw2_zts_key(&r.key, N, K, place_ip);
    if (vw2_rec_set(&r, 1, "eng", eng) != VW2_OK) goto refuse;
    if (_vw2_zts_tail(&r, K, ns)) goto refuse;
    if (vw2_bank(s, &r) != VW2_OK) goto refuse;
    return 0;
refuse:
    vw2_rec_free(&r);
    fprintf(stderr, "[wisdom2] zts held-bank refused (N=%d q=%zu %s)\n", N, K, place_ip ? "ip" : "oop");
    return -1;
}

#endif /* VFFT_WISDOM2_ZTS_SPLIT_H */
