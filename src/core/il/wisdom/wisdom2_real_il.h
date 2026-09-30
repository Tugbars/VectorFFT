/* wisdom2_real_il.h - the interleaved 1D real cell's ENGINE verdict.
 *
 * One record per (transform, N, placement, thread count) at q=1 ord=nat in the
 * real shard (nthreads= on a threaded plan's row; nothing serves across thread counts),
 * exactly the key the zr2c route records use; the level-1 tokens name the
 * engine and its plan input:
 *   eng=zr2c route=child_oop_il|child_nat_ip      (kind-5, wisdom2_oop_il.h)
 *   eng=zrp  pair=R1.R2 leaf=n1t|r2z              (the real pair, il/real/zrp.h:
 *                                                  n1t = form A, r2z = form B)
 *   eng=zttr chain=4.8.8.4 tile=512 stk=3         (ZTT-r, il/real/zttr.h: the chain,
 *                                                  the tile width, the stack state)
 *   eng=zfsr split=1024x2048                      (the real four-step, il/real/zfsr.h:
 *                                                  the split of N/2)
 *   eng=zrm                                       (the real mono, il/real/zrm.h: one
 *                                                  rn1 kernel, N <= 64, no plan input)
 *   eng=oddr                                      (odd N: the odd-real routes stood
 *                                                  against the mono; their own route
 *                                                  record is wisdom2_oddr.h's)
 * The door (il/real/real_create_il.h) reads the engine first and lets the
 * engine read its own plan input; a miss races the engines and banks the
 * winner here. The zr2c route banker keeps its own-engine guard, so a cell
 * the pair owns does not take a zr2c route bank until the door re-banks it.
 */
#ifndef VFFT_WISDOM2_REAL_IL_H
#define VFFT_WISDOM2_REAL_IL_H

#include <stdio.h>
#include <string.h>

#include "wisdom2.h"
#include "wisdom2_oop_rows.h" /* vw2__oop_stamp_date */

static inline void vw2_real_il_key(vw2_key_t *k, int realN, int is_c2r, int is_inplace, int T)
{
    memset(k, 0, sizeof *k);
    k->t = is_c2r ? VW2_T_C2R : VW2_T_R2C;
    k->rank = 1; k->n[0] = realN;
    k->q = 1; k->ord = VW2_ORD_NAT;
    k->pl = is_inplace ? VW2_PL_IP : VW2_PL_OOP;
    k->nthreads = (uint8_t)(T > 1 ? T : 0);   /* a threaded plan's row is its own (v1.3) */
}

/* The cell's banked engine ("zr2c", "zrp", "zttr", "zrm", "oddr") or NULL
 * (no measured record; a seed row counts as none). For eng=zrp the pair and the form are decoded
 * (*R1 = 0 when the pair token is missing or malformed: a miss; *form = 0
 * for leaf=n1t or no leaf token, 1 for leaf=r2z). */
static inline const char *vw2_real_il_lookup(const vw2_store_t *s, int realN,
                                             int is_c2r, int is_inplace, int T,
                                             int *R1, int *R2, int *form)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    const char *eng;
    *R1 = *R2 = *form = 0;
    vw2_real_il_key(&k, realN, is_c2r, is_inplace, T);
    r = vw2_lookup(s, &k);
    if (!r || vw2__is_seed(r)) return NULL;
    eng = vw2_rec_get(r, "eng");
    if (!eng) return NULL;
    if (!strcmp(eng, "zrp")) {
        const char *pair = vw2_rec_get(r, "pair"), *leaf = vw2_rec_get(r, "leaf");
        int a = 0, b = 0;
        if (!pair || sscanf(pair, "%d.%d", &a, &b) != 2 || a <= 0 || b <= 0) return NULL;
        *R1 = a; *R2 = b;
        *form = (leaf && !strcmp(leaf, "r2z")) ? 1 : 0;
    }
    return eng;
}

/* Bank the real pair verdict at the cell (replacing whatever engine held it:
 * the pinned upsert says so on stderr when the engine changes). ns = the
 * winner's per-shot median; <= 0 omits the measurement. */
static inline int vw2_real_il_bank_zrp(vw2_store_t *s, int realN, int is_c2r,
                                       int is_inplace, int T, int R1, int R2, int form, double ns)
{
    vw2_rec_t r;
    char b[48];
    int rc;
    memset(&r, 0, sizeof r);
    vw2_real_il_key(&r.key, realN, is_c2r, is_inplace, T);
    snprintf(b, sizeof b, "%d.%d", R1, R2);
    if (vw2_rec_set(&r, 1, "eng", "zrp") != VW2_OK ||
        vw2_rec_set(&r, 1, "pair", b) != VW2_OK ||
        vw2_rec_set(&r, 1, "leaf", form ? "r2z" : "n1t") != VW2_OK ||
        vw2_rec_set(&r, 2, "ran", "1") != VW2_OK ||
        vw2_rec_set(&r, 2, "src", "race") != VW2_OK) { vw2_rec_free(&r); return -1; }
    if (ns > 0.0) {
        snprintf(b, sizeof b, "%.1f", ns);
        if (vw2_rec_set(&r, 2, "ns", b) != VW2_OK ||
            vw2_rec_set(&r, 2, "metric", is_c2r ? "bwd1" : "fwd1") != VW2_OK ||
            vw2_rec_set(&r, 2, "units", "ns") != VW2_OK) { vw2_rec_free(&r); return -1; }
    }
    vw2__oop_stamp_date(&r);
    rc = vw2_bank(s, &r);
    if (rc != VW2_OK) { vw2_rec_free(&r); return rc; }
    return VW2_OK;
}

/* Bank an engine with no plan input at the cell (replacing whatever engine
 * held it): eng=zrm (the real mono; the kernel follows from the transform
 * and N) or eng=oddr (odd N: the odd-real routes stood). */
static inline int vw2_real_il_bank_eng(vw2_store_t *s, int realN, int is_c2r,
                                       int is_inplace, int T, const char *eng, double ns)
{
    vw2_rec_t r;
    char b[48];
    int rc;
    memset(&r, 0, sizeof r);
    vw2_real_il_key(&r.key, realN, is_c2r, is_inplace, T);
    if (vw2_rec_set(&r, 1, "eng", eng) != VW2_OK ||
        vw2_rec_set(&r, 2, "ran", "1") != VW2_OK ||
        vw2_rec_set(&r, 2, "src", "race") != VW2_OK) { vw2_rec_free(&r); return -1; }
    if (ns > 0.0) {
        snprintf(b, sizeof b, "%.1f", ns);
        if (vw2_rec_set(&r, 2, "ns", b) != VW2_OK ||
            vw2_rec_set(&r, 2, "metric", is_c2r ? "bwd1" : "fwd1") != VW2_OK ||
            vw2_rec_set(&r, 2, "units", "ns") != VW2_OK) { vw2_rec_free(&r); return -1; }
    }
    vw2__oop_stamp_date(&r);
    rc = vw2_bank(s, &r);
    if (rc != VW2_OK) { vw2_rec_free(&r); return rc; }
    return VW2_OK;
}
static inline int vw2_real_il_bank_zrm(vw2_store_t *s, int realN, int is_c2r, int is_inplace, int T, double ns)
{
    return vw2_real_il_bank_eng(s, realN, is_c2r, is_inplace, T, "zrm", ns);
}

/* zr2c on a THREADED plan's row (T > 1): the record is complete on its own --
 * the child route and the fold's serving (fold=mt: cut over the plan's
 * threads) -- because the thread-free route record may belong to another
 * engine. 1 with *route, *fold_mt filled, 0 on a miss. */
static inline int vw2_real_il_lookup_zr2c_t(const vw2_store_t *s, int realN, int is_c2r,
                                            int is_inplace, int T, int *route, int *fold_mt)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    const char *eng, *rt, *fd;
    *route = 0; *fold_mt = 0;
    vw2_real_il_key(&k, realN, is_c2r, is_inplace, T);
    r = vw2_lookup(s, &k);
    if (!r || vw2__is_seed(r)) return 0;
    eng = vw2_rec_get(r, "eng");
    if (!eng || strcmp(eng, "zr2c")) return 0;
    rt = vw2_rec_get(r, "route"); fd = vw2_rec_get(r, "fold");
    if (!rt) return 0;
    *route = !strcmp(rt, "child_nat_ip") ? 1 : 0;
    *fold_mt = (fd && !strcmp(fd, "mt")) ? 1 : 0;
    return 1;
}
static inline int vw2_real_il_bank_zr2c_t(vw2_store_t *s, int realN, int is_c2r, int is_inplace,
                                          int T, int route, int fold_mt, double ns)
{
    vw2_rec_t r;
    char b[48];
    int rc;
    memset(&r, 0, sizeof r);
    vw2_real_il_key(&r.key, realN, is_c2r, is_inplace, T);
    if (vw2_rec_set(&r, 1, "eng", "zr2c") != VW2_OK ||
        vw2_rec_set(&r, 1, "route", route ? "child_nat_ip" : "child_oop_il") != VW2_OK ||
        vw2_rec_set(&r, 1, "fold", fold_mt ? "mt" : "st") != VW2_OK ||
        vw2_rec_set(&r, 2, "ran", "1") != VW2_OK ||
        vw2_rec_set(&r, 2, "src", "race") != VW2_OK) { vw2_rec_free(&r); return -1; }
    if (ns > 0.0) {
        snprintf(b, sizeof b, "%.1f", ns);
        if (vw2_rec_set(&r, 2, "ns", b) != VW2_OK ||
            vw2_rec_set(&r, 2, "metric", is_c2r ? "bwd1" : "fwd1") != VW2_OK ||
            vw2_rec_set(&r, 2, "units", "ns") != VW2_OK) { vw2_rec_free(&r); return -1; }
    }
    vw2__oop_stamp_date(&r);
    rc = vw2_bank(s, &r);
    if (rc != VW2_OK) { vw2_rec_free(&r); return rc; }
    return VW2_OK;
}

/* The banked real four-step split at the cell: 1 with *n1, *n2 filled, 0 when
 * the cell's engine is not zfsr or the token is malformed (a miss). */
static inline int vw2_real_il_lookup_zfsr(const vw2_store_t *s, int realN, int is_c2r,
                                          int is_inplace, int T, int *n1, int *n2)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    const char *eng, *sp;
    int a = 0, b = 0;
    *n1 = *n2 = 0;
    vw2_real_il_key(&k, realN, is_c2r, is_inplace, T);
    r = vw2_lookup(s, &k);
    if (!r || vw2__is_seed(r)) return 0;
    eng = vw2_rec_get(r, "eng");
    if (!eng || strcmp(eng, "zfsr")) return 0;
    sp = vw2_rec_get(r, "split");
    if (!sp || sscanf(sp, "%dx%d", &a, &b) != 2 || a <= 0 || b <= 0) return 0;
    *n1 = a; *n2 = b;
    return 1;
}

/* Bank the real four-step verdict at the cell (replacing whatever engine held it). */
static inline int vw2_real_il_bank_zfsr(vw2_store_t *s, int realN, int is_c2r, int is_inplace, int T,
                                        int n1, int n2, double ns)
{
    vw2_rec_t r;
    char b[48];
    int rc;
    memset(&r, 0, sizeof r);
    vw2_real_il_key(&r.key, realN, is_c2r, is_inplace, T);
    snprintf(b, sizeof b, "%dx%d", n1, n2);
    if (vw2_rec_set(&r, 1, "eng", "zfsr") != VW2_OK ||
        vw2_rec_set(&r, 1, "split", b) != VW2_OK ||
        vw2_rec_set(&r, 2, "ran", "1") != VW2_OK ||
        vw2_rec_set(&r, 2, "src", "race") != VW2_OK) { vw2_rec_free(&r); return -1; }
    if (ns > 0.0) {
        snprintf(b, sizeof b, "%.1f", ns);
        if (vw2_rec_set(&r, 2, "ns", b) != VW2_OK ||
            vw2_rec_set(&r, 2, "metric", is_c2r ? "bwd1" : "fwd1") != VW2_OK ||
            vw2_rec_set(&r, 2, "units", "ns") != VW2_OK) { vw2_rec_free(&r); return -1; }
    }
    vw2__oop_stamp_date(&r);
    rc = vw2_bank(s, &r);
    if (rc != VW2_OK) { vw2_rec_free(&r); return rc; }
    return VW2_OK;
}

/* The banked ZTT-r plan input at the cell: 1 with chain[0..*nf-1], *tile,
 * *stk filled, 0 when the cell's engine is not zttr or the tokens are
 * malformed (a miss). */
static inline int vw2_real_il_lookup_zttr(const vw2_store_t *s, int realN, int is_c2r,
                                          int is_inplace, int T, int chain[8], int *nf,
                                          size_t *tile, int *stk)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    const char *eng, *ch, *tl, *st;
    int n = 0;
    *nf = 0; *tile = 0; *stk = 3;
    vw2_real_il_key(&k, realN, is_c2r, is_inplace, T);
    r = vw2_lookup(s, &k);
    if (!r || vw2__is_seed(r)) return 0;
    eng = vw2_rec_get(r, "eng");
    if (!eng || strcmp(eng, "zttr")) return 0;
    ch = vw2_rec_get(r, "chain"); tl = vw2_rec_get(r, "tile"); st = vw2_rec_get(r, "stk");
    if (!ch) return 0;
    while (*ch && n < 8) {
        char *end;
        long v = strtol(ch, &end, 10);
        if (end == ch || !(v == 4 || v == 8 || v == 3 || v == 5 || v == 7 || v == 9 || v == 15)) return 0;
        chain[n++] = (int)v;
        ch = end;
        if (*ch == '.') ch++;
        else if (*ch) return 0;
    }
    if (n < 2) return 0;
    *nf = n;
    if (tl) *tile = (size_t)strtoul(tl, NULL, 10);
    if (st) { int v = atoi(st); if (v >= 0 && v <= 3) *stk = v; }
    return 1;
}

/* Bank the ZTT-r verdict at the cell (replacing whatever engine held it). */
static inline int vw2_real_il_bank_zttr(vw2_store_t *s, int realN, int is_c2r, int is_inplace, int T,
                                        const int *chain, int nf, size_t tile, int stk, double ns)
{
    vw2_rec_t r;
    char b[64];
    int rc, off = 0;
    memset(&r, 0, sizeof r);
    vw2_real_il_key(&r.key, realN, is_c2r, is_inplace, T);
    for (int i = 0; i < nf && off < (int)sizeof b - 4; i++)
        off += snprintf(b + off, sizeof b - (size_t)off, "%s%d", i ? "." : "", chain[i]);
    if (vw2_rec_set(&r, 1, "eng", "zttr") != VW2_OK ||
        vw2_rec_set(&r, 1, "chain", b) != VW2_OK) { vw2_rec_free(&r); return -1; }
    snprintf(b, sizeof b, "%zu", tile);
    if (vw2_rec_set(&r, 1, "tile", b) != VW2_OK) { vw2_rec_free(&r); return -1; }
    snprintf(b, sizeof b, "%d", stk);
    if (vw2_rec_set(&r, 1, "stk", b) != VW2_OK ||
        vw2_rec_set(&r, 2, "ran", "1") != VW2_OK ||
        vw2_rec_set(&r, 2, "src", "race") != VW2_OK) { vw2_rec_free(&r); return -1; }
    if (ns > 0.0) {
        snprintf(b, sizeof b, "%.1f", ns);
        if (vw2_rec_set(&r, 2, "ns", b) != VW2_OK ||
            vw2_rec_set(&r, 2, "metric", is_c2r ? "bwd1" : "fwd1") != VW2_OK ||
            vw2_rec_set(&r, 2, "units", "ns") != VW2_OK) { vw2_rec_free(&r); return -1; }
    }
    vw2__oop_stamp_date(&r);
    rc = vw2_bank(s, &r);
    if (rc != VW2_OK) { vw2_rec_free(&r); return rc; }
    return VW2_OK;
}

#endif /* VFFT_WISDOM2_REAL_IL_H */
