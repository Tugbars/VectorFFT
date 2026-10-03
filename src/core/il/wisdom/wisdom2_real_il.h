/* wisdom2_real_il.h - the interleaved 1D real cell's ENGINE verdict.
 *
 * One record per (transform, N, placement, thread count) at q=1 ord=nat in the
 * real shard (nthreads= on a threaded plan's row; nothing serves across thread counts),
 * exactly the key the zr2c route records use; the level-1 tokens name the
 * engine and its plan input:
 *   eng=zr2c route=child_oop_il|child_nat_ip      (il/real/zr2c_build.h: the child's
 *            [fold=st|mt] il_route=... il_*=...   placement; on a threaded plan's row the
 *                                                  fold's serving; and the CHILD -- the
 *                                                  c2c(N/2) plan raced in the real role --
 *                                                  in the c2c K=1 record's own words:
 *                                                  il_route il_pair il_chain il_flat il_forms
 *                                                  il_tw il_ztt il_sb il_kv, and il_bkv =
 *                                                  the backward forms a c2r child runs)
 *   eng=zrp  pair=R1.R2 leaf=n1t|r2z              (the real pair, il/real/zrp.h:
 *                                                  n1t = form A, r2z = form B)
 *   eng=zttr chain=4.8.8.4 tile=512 stk=3 [mt=1|2] (ZTT-r, il/real/zttr.h: the chain,
 *                                                  the tile width, the stack state; on a
 *                                                  threaded plan's row the threaded arm)
 *   eng=zfsr split=1024x2048 fs_*=.. fs_row_*=..  (the real four-step, il/real/zfsr.h:
 *                                                  the split of N/2, and its CHILD in its
 *                                                  own rows' words: the 2D plan's payload
 *                                                  under fs_ (fs_chain fs_wl fs_tf fs_ro
 *                                                  ...), the row plan's at N2 under
 *                                                  fs_row_ (fs_row_il_route ...), its
 *                                                  backward twin's under fs_row_bwd_
 *                                                  where it has one)
 *   eng=zrm                                       (the real mono, il/real/zrm.h: one
 *                                                  rn1 kernel, N <= 64, no plan input)
 *   eng=zrf  chain=9.9.5 msz=0|1 tile=256 [mt=1|2] (the real flat DIT, il/real/zrf.h, odd N:
 *                                                  the chain, whether the split-body stage
 *                                                  form runs where it exists, the tile
 *                                                  budget in complex, 0 = untiled; on a
 *                                                  threaded plan's row the threaded arm:
 *                                                  1 FIRST, 2 LEVELS)
 *   eng=zrb  m=1152 in=ztt in_sh=4.4.8.9 in_tw=0  (the real Bluestein, il/real/zrb.h, odd N
 *                                                  without a chain: the convolution length
 *                                                  and the inner pair's descriptor, the
 *                                                  prime route's own spelling)
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
#include "wisdom2_oop_il.h"   /* the c2c K=1 vocabulary: the zr2c child's recipe */

static inline void vw2_real_il_key(vw2_key_t *k, int realN, int is_c2r, int is_inplace, int T)
{
    memset(k, 0, sizeof *k);
    k->t = is_c2r ? VW2_T_C2R : VW2_T_R2C;
    k->rank = 1; k->n[0] = realN;
    k->q = 1; k->ord = VW2_ORD_NAT;
    k->pl = is_inplace ? VW2_PL_IP : VW2_PL_OOP;
    k->nthreads = (uint8_t)(T > 1 ? T : 0);   /* a threaded plan's row is its own (v1.3) */
}

/* The cell's banked engine ("zr2c", "zrp", "zttr", "zfsr", "zrm", "zrf", "zrb") or NULL
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
 * and N). */
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

/* The banked real flat DIT plan input at the cell: 1 with chain[0..*K-1],
 * *nomsz and *tile filled, 0 when the cell's engine is not zrf or the tokens
 * are malformed (a miss). max = the capacity of chain[]. */
static inline int vw2_real_il_lookup_zrf(const vw2_store_t *s, int realN, int is_c2r,
                                         int is_inplace, int T, int *chain, int max, int *K, int *nomsz,
                                         int *tile, int *mt)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    const char *eng, *ch, *ms, *tl;
    long prod = 1;
    int n = 0;
    *K = 0; *nomsz = 0; *tile = 0; *mt = 0;
    vw2_real_il_key(&k, realN, is_c2r, is_inplace, T);
    r = vw2_lookup(s, &k);
    if (!r || vw2__is_seed(r)) return 0;
    eng = vw2_rec_get(r, "eng");
    if (!eng || strcmp(eng, "zrf")) return 0;
    ch = vw2_rec_get(r, "chain"); ms = vw2_rec_get(r, "msz"); tl = vw2_rec_get(r, "tile");
    if (!ch) return 0;
    while (*ch && n < max) {
        char *end;
        long v = strtol(ch, &end, 10);
        if (end == ch || v < 3 || !(v & 1)) return 0;
        chain[n++] = (int)v;
        prod *= v;
        ch = end;
        if (*ch == '.') ch++;
        else if (*ch) return 0;
    }
    if (n < 2 || *ch || prod != (long)realN) return 0;
    *K = n;
    *nomsz = (ms && ms[0] == '0') ? 1 : 0;
    if (tl) { const int v = atoi(tl); if (v < 0) return 0; *tile = v; }
    {   /* the threaded arm of a threaded plan's row: 1 FIRST, 2 LEVELS; absent = serial */
        const char *m = vw2_rec_get(r, "mt");
        if (m) { const int v = atoi(m); if (v >= 1 && v <= 2) *mt = v; }
    }
    return 1;
}

/* Bank the real flat DIT verdict at the cell (replacing whatever engine held it). */
static inline int vw2_real_il_bank_zrf(vw2_store_t *s, int realN, int is_c2r, int is_inplace, int T,
                                       const int *chain, int K, int nomsz, int tile, int mt, double ns)
{
    vw2_rec_t r;
    char b[64];
    int rc, off = 0;
    memset(&r, 0, sizeof r);
    vw2_real_il_key(&r.key, realN, is_c2r, is_inplace, T);
    for (int i = 0; i < K && off < (int)sizeof b - 4; i++)
        off += snprintf(b + off, sizeof b - (size_t)off, "%s%d", i ? "." : "", chain[i]);
    if (vw2_rec_set(&r, 1, "eng", "zrf") != VW2_OK ||
        vw2_rec_set(&r, 1, "chain", b) != VW2_OK ||
        vw2_rec_set(&r, 1, "msz", nomsz ? "0" : "1") != VW2_OK) { vw2_rec_free(&r); return -1; }
    snprintf(b, sizeof b, "%d", tile > 0 ? tile : 0);
    if (vw2_rec_set(&r, 1, "tile", b) != VW2_OK ||
        (mt > 0 && vw2_rec_set(&r, 1, "mt", mt == 2 ? "2" : "1") != VW2_OK) ||
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

/* The banked real Bluestein plan input at the cell: 1 with *M and the inner's
 * kind / shape / tile filled, 0 when the cell's engine is not zrb or the
 * tokens are malformed (a miss). */
static inline int vw2_real_il_lookup_zrb_q(const vw2_store_t *s, int realN, int K, int is_c2r,
                                           int is_inplace, int T, int *M, char *kind, size_t ksz,
                                           char *shape, size_t ssz, int *tw)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    const char *eng, *m, *in, *sh, *t;
    *M = 0; *tw = 0; kind[0] = 0; shape[0] = 0;
    vw2_real_il_key(&k, realN, is_c2r, is_inplace, T);
    k.q = K;   /* the q=1 row is the one-row cell; a batch's lane-major verdict is its q=K row */
    r = vw2_lookup(s, &k);
    if (!r || vw2__is_seed(r)) return 0;
    eng = vw2_rec_get(r, "eng");
    if (!eng || strcmp(eng, "zrb")) return 0;
    m = vw2_rec_get(r, "m"); in = vw2_rec_get(r, "in"); sh = vw2_rec_get(r, "in_sh"); t = vw2_rec_get(r, "in_tw");
    if (!m || !in || !sh) return 0;
    *M = atoi(m);
    if (*M < realN + (realN - 1) / 2) return 0;
    snprintf(kind, ksz, "%s", in);
    snprintf(shape, ssz, "%s", sh);
    if (t) { const int v = atoi(t); if (v < 0) return 0; *tw = v; }
    return 1;
}
static inline int vw2_real_il_lookup_zrb(const vw2_store_t *s, int realN, int is_c2r,
                                         int is_inplace, int T, int *M, char *kind, size_t ksz,
                                         char *shape, size_t ssz, int *tw)
{
    return vw2_real_il_lookup_zrb_q(s, realN, 1, is_c2r, is_inplace, T, M, kind, ksz, shape, ssz, tw);
}

/* Bank the real Bluestein verdict at the cell (replacing whatever engine held it);
 * K > 1 = the lane-major batch's q=K row. */
static inline int vw2_real_il_bank_zrb_q(vw2_store_t *s, int realN, int K, int is_c2r, int is_inplace, int T,
                                         int M, const char *kind, const char *shape, int tw, double ns)
{
    vw2_rec_t r;
    char b[64];
    int rc;
    memset(&r, 0, sizeof r);
    vw2_real_il_key(&r.key, realN, is_c2r, is_inplace, T);
    r.key.q = K;
    snprintf(b, sizeof b, "%d", M);
    if (vw2_rec_set(&r, 1, "eng", "zrb") != VW2_OK ||
        vw2_rec_set(&r, 1, "m", b) != VW2_OK ||
        vw2_rec_set(&r, 1, "in", kind) != VW2_OK ||
        vw2_rec_set(&r, 1, "in_sh", shape) != VW2_OK) { vw2_rec_free(&r); return -1; }
    snprintf(b, sizeof b, "%d", tw > 0 ? tw : 0);
    if (vw2_rec_set(&r, 1, "in_tw", b) != VW2_OK ||
        vw2_rec_set(&r, 2, "ran", "1") != VW2_OK ||
        vw2_rec_set(&r, 2, "src", "race") != VW2_OK) { vw2_rec_free(&r); return -1; }
    if (ns > 0.0) {
        snprintf(b, sizeof b, "%.1f", ns);
        if (vw2_rec_set(&r, 2, "ns", b) != VW2_OK ||
            vw2_rec_set(&r, 2, "metric", K > 1 ? (is_c2r ? "bwdK" : "fwdK") : (is_c2r ? "bwd1" : "fwd1")) != VW2_OK ||
            vw2_rec_set(&r, 2, "units", "ns") != VW2_OK) { vw2_rec_free(&r); return -1; }
    }
    vw2__oop_stamp_date(&r);
    rc = vw2_bank(s, &r);
    if (rc != VW2_OK) { vw2_rec_free(&r); return rc; }
    return VW2_OK;
}
static inline int vw2_real_il_bank_zrb(vw2_store_t *s, int realN, int is_c2r, int is_inplace, int T,
                                       int M, const char *kind, const char *shape, int tw, double ns)
{
    return vw2_real_il_bank_zrb_q(s, realN, 1, is_c2r, is_inplace, T, M, kind, shape, tw, ns);
}

/* The lane Bluestein's cell is the q=K row (the batch count is the key's
 * quantity): eng=zrbl m= chain= forms= wc=. */
static inline void vw2_real_il_key_k(vw2_key_t *k, int realN, int K, int is_c2r, int is_inplace, int T)
{
    vw2_real_il_key(k, realN, is_c2r, is_inplace, T);
    k->q = K;
}
static inline int vw2_real_il_lookup_zrbl(const vw2_store_t *s, int realN, int K, int is_c2r,
                                          int is_inplace, int T, int *M, int *chain, int max, int *nst,
                                          char *forms, size_t fsz, int *wc)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    const char *eng, *m, *ch, *fo, *w;
    long prod = 1;
    int n = 0;
    *M = 0; *nst = 0; *wc = 0; forms[0] = 0;
    vw2_real_il_key_k(&k, realN, K, is_c2r, is_inplace, T);
    r = vw2_lookup(s, &k);
    if (!r || vw2__is_seed(r)) return 0;
    eng = vw2_rec_get(r, "eng");
    if (!eng || strcmp(eng, "zrbl")) return 0;
    m = vw2_rec_get(r, "m"); ch = vw2_rec_get(r, "chain"); fo = vw2_rec_get(r, "forms"); w = vw2_rec_get(r, "wc");
    if (!m || !ch) return 0;
    *M = atoi(m);
    if (*M < realN + (realN - 1) / 2) return 0;
    while (*ch && n < max) {
        char *end;
        long v = strtol(ch, &end, 10);
        if (end == ch || v < 2) return 0;
        chain[n++] = (int)v;
        prod *= v;
        ch = end;
        if (*ch == '.') ch++;
        else if (*ch) return 0;
    }
    if (n < 1 || *ch || prod != (long)*M) return 0;
    *nst = n;
    if (fo && strcmp(fo, "-")) snprintf(forms, fsz, "%s", fo);
    if (w) { const int v = atoi(w); if (v < 0) return 0; *wc = v; }
    return 1;
}
static inline int vw2_real_il_bank_zrbl(vw2_store_t *s, int realN, int K, int is_c2r, int is_inplace, int T,
                                        int M, const int *chain, int nst, const char *forms, int wc, double ns)
{
    vw2_rec_t r;
    char b[64];
    int rc, off = 0;
    memset(&r, 0, sizeof r);
    vw2_real_il_key_k(&r.key, realN, K, is_c2r, is_inplace, T);
    snprintf(b, sizeof b, "%d", M);
    if (vw2_rec_set(&r, 1, "eng", "zrbl") != VW2_OK || vw2_rec_set(&r, 1, "m", b) != VW2_OK) { vw2_rec_free(&r); return -1; }
    for (int i = 0; i < nst && off < (int)sizeof b - 4; i++)
        off += snprintf(b + off, sizeof b - (size_t)off, "%s%d", i ? "." : "", chain[i]);
    if (vw2_rec_set(&r, 1, "chain", b) != VW2_OK ||
        vw2_rec_set(&r, 1, "forms", forms && forms[0] ? forms : "-") != VW2_OK) { vw2_rec_free(&r); return -1; }
    snprintf(b, sizeof b, "%d", wc > 0 ? wc : 0);
    if (vw2_rec_set(&r, 1, "wc", b) != VW2_OK ||
        vw2_rec_set(&r, 2, "ran", "1") != VW2_OK ||
        vw2_rec_set(&r, 2, "src", "race") != VW2_OK) { vw2_rec_free(&r); return -1; }
    if (ns > 0.0) {
        snprintf(b, sizeof b, "%.1f", ns);
        if (vw2_rec_set(&r, 2, "ns", b) != VW2_OK ||
            vw2_rec_set(&r, 2, "metric", is_c2r ? "bwdK" : "fwdK") != VW2_OK ||
            vw2_rec_set(&r, 2, "units", "ns") != VW2_OK) { vw2_rec_free(&r); return -1; }
    }
    vw2__oop_stamp_date(&r);
    rc = vw2_bank(s, &r);
    if (rc != VW2_OK) { vw2_rec_free(&r); return rc; }
    return VW2_OK;
}

/* THE zr2c ROW. The child -- the c2c(N/2) plan raced IN THE REAL ROLE (with
 * the fold, at the route's placement) -- is the cell's own verdict, so its
 * recipe rides on the cell's row, never in the c2c cell's: a zr2c create
 * reads and writes no c2c row. The recipe is written in the c2c K=1 record's
 * vocabulary (wisdom2_oop_il.h), plus il_bkv (the backward forms, the only
 * forms a c2r child runs). One record per (transform, N, placement, T); on a
 * threaded plan's row fold= says how the fold is served. A PRIME child's
 * prime cell rides here too, in the prime row's vocabulary under il_prime*:
 * the method and the inner, raced on the cell's own convolution with no
 * store -- a zr2c create reads and writes no prime row either. */
typedef struct
{
    int  route;            /* il_route: VFFT_K1_IL_* */
    int  R1, R2;           /* il_pair (the pair; the four-step's N1 x N2) */
    int  c3[3];            /* il_chain (chain3: R2.A.B) */
    int  fl[10], fl_n;     /* il_flat (the flat DIT's chain) */
    char flf[24];          /* il_forms (its per-stage forms) */
    int  tw;               /* il_tw (the flat DIT's / ZTURN-T's tile) */
    int  zt[7], zt_n;      /* il_ztt (ZTURN-T's chain) / il_sb (the four-step's super-band chain) */
    int  kv, bkv;          /* il_kv (forward forms), il_bkv (backward forms) */
    int  pm;               /* il_prime: 1 rader, 2 bluestein (route PRIME only) */
    char pin[8];           /* il_prime_in (the inner's kind: 2p | 3p | ztt) */
    char psh[64];          /* il_prime_sh (its shape) */
    int  ptw;              /* il_prime_tw (its tile, 0 = untiled) */
} vw2_zr2c_child_t;

static inline int vw2__zr2c_ints_str(char *b, size_t cap, const int *v, int n)
{
    size_t off = 0;
    int i;
    b[0] = 0;
    for (i = 0; i < n; i++)
    {
        int rr = snprintf(b + off, cap - off, "%s%d", i ? "." : "", v[i]);
        if (rr < 0 || (size_t)rr >= cap - off) return -1;
        off += (size_t)rr;
    }
    return 0;
}

static inline int vw2_real_il_bank_zr2c(vw2_store_t *s, int realN, int is_c2r, int is_inplace, int T,
                                        int route, int fold_mt, const vw2_zr2c_child_t *ch, double ns)
{
    vw2_rec_t r;
    char b[64];
    int rc;
    if (!ch || ch->route <= VFFT_K1_IL_NONE || ch->route > VW2_OOP_IL_ROUTE_MAX) return -1;
    memset(&r, 0, sizeof r);
    vw2_real_il_key(&r.key, realN, is_c2r, is_inplace, T);
#define VW2__ZS(n, v) do { if (vw2_rec_set(&r, 1, (n), (v)) != VW2_OK) { vw2_rec_free(&r); return -1; } } while (0)
    VW2__ZS("eng", "zr2c");
    VW2__ZS("route", route ? "child_nat_ip" : "child_oop_il");
    if (T > 1) VW2__ZS("fold", fold_mt ? "mt" : "st");
    VW2__ZS("il_route", vw2_oop_il_name[ch->route]);
    if (ch->R1 || ch->R2) { snprintf(b, sizeof b, "%d.%d", ch->R1, ch->R2); VW2__ZS("il_pair", b); }
    if (ch->route == VFFT_K1_IL_CHAIN3) { if (vw2__zr2c_ints_str(b, sizeof b, ch->c3, 3)) goto bad; VW2__ZS("il_chain", b); }
    if (ch->route == VFFT_K1_IL_FLAT && ch->fl_n >= 2)
    {
        if (vw2__zr2c_ints_str(b, sizeof b, ch->fl, ch->fl_n)) goto bad;
        VW2__ZS("il_flat", b);
        if (ch->flf[0]) VW2__ZS("il_forms", ch->flf);
    }
    if (ch->zt_n >= 2 && (ch->route == VFFT_K1_IL_ZTT || ch->route == VFFT_K1_IL_FS))
    {
        if (vw2__zr2c_ints_str(b, sizeof b, ch->zt, ch->zt_n)) goto bad;
        VW2__ZS(ch->route == VFFT_K1_IL_ZTT ? "il_ztt" : "il_sb", b);
    }
    if (ch->tw > 0) { snprintf(b, sizeof b, "%d", ch->tw); VW2__ZS("il_tw", b); }
    if (ch->route == VFFT_K1_IL_PRIME)
    {   /* the prime cell: no method and inner, no recipe */
        if ((ch->pm != 1 && ch->pm != 2) || !ch->pin[0] || !ch->psh[0]) goto bad;
        VW2__ZS("il_prime", ch->pm == 1 ? "rader" : "bluestein");
        VW2__ZS("il_prime_in", ch->pin);
        VW2__ZS("il_prime_sh", ch->psh);
        snprintf(b, sizeof b, "%d", ch->ptw); VW2__ZS("il_prime_tw", b);
    }
    snprintf(b, sizeof b, "%d", ch->kv);  VW2__ZS("il_kv", b);
    snprintf(b, sizeof b, "%d", ch->bkv); VW2__ZS("il_bkv", b);
#undef VW2__ZS
    if (vw2_rec_set(&r, 2, "ran", "1") != VW2_OK || vw2_rec_set(&r, 2, "src", "race") != VW2_OK) goto bad;
    if (ns > 0.0)
    {
        snprintf(b, sizeof b, "%.1f", ns);
        if (vw2_rec_set(&r, 2, "ns", b) != VW2_OK ||
            /* the c2r composite runs the fold and then the child backward: bwd1 */
            vw2_rec_set(&r, 2, "metric", is_c2r ? "bwd1" : "fwd1") != VW2_OK ||
            vw2_rec_set(&r, 2, "units", "ns") != VW2_OK) goto bad;
    }
    vw2__oop_stamp_date(&r);
    rc = vw2_bank(s, &r);
    if (rc != VW2_OK) { vw2_rec_free(&r); return rc; }
    return VW2_OK;
bad:
    vw2_rec_free(&r);
    return -1;
}

/* 1 with *route, *fold_mt and the child's recipe filled; 0 on a miss -- and a
 * zr2c row WITHOUT a child recipe is a miss: it predates the in-role child
 * (its child was the c2c cell's verdict), so the cell races again. */
static inline int vw2_real_il_lookup_zr2c(const vw2_store_t *s, int realN, int is_c2r, int is_inplace,
                                          int T, int *route, int *fold_mt, vw2_zr2c_child_t *ch)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    const char *eng, *rt, *fd, *il, *ff, *pm, *pin, *psh;
    int v[10], n;
    *route = 0; *fold_mt = 0;
    memset(ch, 0, sizeof *ch);
    vw2_real_il_key(&k, realN, is_c2r, is_inplace, T);
    r = vw2_lookup(s, &k);
    if (!r || vw2__is_seed(r)) return 0;
    eng = vw2_rec_get(r, "eng");
    if (!eng || strcmp(eng, "zr2c")) return 0;
    rt = vw2_rec_get(r, "route"); fd = vw2_rec_get(r, "fold"); il = vw2_rec_get(r, "il_route");
    if (!rt || !il) return 0;
    if (!strcmp(rt, "child_nat_ip")) *route = 1;
    else if (strcmp(rt, "child_oop_il")) return 0;
    *fold_mt = (fd && !strcmp(fd, "mt")) ? 1 : 0;
    ch->route = vw2__oop_name_idx(vw2_oop_il_name, VW2_OOP_IL_ROUTE_MAX + 1, il);
    if (ch->route <= VFFT_K1_IL_NONE) return 0;
    if (vw2__oop_split_ints(vw2_rec_get(r, "il_pair"), v, 2) == 2) { ch->R1 = v[0]; ch->R2 = v[1]; }
    if (vw2__oop_split_ints(vw2_rec_get(r, "il_chain"), v, 3) == 3) { ch->c3[0] = v[0]; ch->c3[1] = v[1]; ch->c3[2] = v[2]; }
    n = vw2__oop_split_ints(vw2_rec_get(r, "il_flat"), v, 10);
    if (n >= 2) { memcpy(ch->fl, v, sizeof(int) * (size_t)n); ch->fl_n = n; }
    ff = vw2_rec_get(r, "il_forms");
    if (ff) { strncpy(ch->flf, ff, sizeof ch->flf - 1); ch->flf[sizeof ch->flf - 1] = 0; }
    n = vw2__oop_split_ints(vw2_rec_get(r, ch->route == VFFT_K1_IL_FS ? "il_sb" : "il_ztt"), v, 7);
    if (n >= 2) { memcpy(ch->zt, v, sizeof(int) * (size_t)n); ch->zt_n = n; }
    ch->tw = vw2__oop_geti(r, "il_tw", 0);
    ch->kv = vw2__oop_geti(r, "il_kv", 0);
    ch->bkv = vw2__oop_geti(r, "il_bkv", 0);
    if (ch->route == VFFT_K1_IL_PRIME)
    {   /* a prime child without its method and inner is no recipe */
        pm = vw2_rec_get(r, "il_prime"); pin = vw2_rec_get(r, "il_prime_in"); psh = vw2_rec_get(r, "il_prime_sh");
        if (!pm || !pin || !psh || strlen(pin) >= sizeof ch->pin || strlen(psh) >= sizeof ch->psh) return 0;
        if (!strcmp(pm, "rader")) ch->pm = 1;
        else if (!strcmp(pm, "bluestein")) ch->pm = 2;
        else return 0;
        strcpy(ch->pin, pin);
        strcpy(ch->psh, psh);
        ch->ptw = vw2__oop_geti(r, "il_prime_tw", 0);
    }
    /* the recipe's own shape: a route without its payload is no recipe */
    if ((ch->route == VFFT_K1_IL_2P_PURE || ch->route == VFFT_K1_IL_FS) && !(ch->R1 > 0 && ch->R2 > 0)) return 0;
    if (ch->route == VFFT_K1_IL_CHAIN3 && !ch->c3[0]) return 0;
    if (ch->route == VFFT_K1_IL_FLAT && ch->fl_n < 2) return 0;
    if (ch->route == VFFT_K1_IL_ZTT && ch->zt_n < 2) return 0;
    return 1;
}

/* THE FOUR-STEP CHILD on the real row (owner, 2026-10-03). The real
 * four-step's child -- the 2D plan at N1 x N2 and its row plan at N2, raced
 * into the plan's private store (il/rank1/k1_fourstep.h) -- rides here in its
 * own rows' words: every payload token of the 2D row under fs_, of the row
 * plan's under fs_row_, of the row plan's backward twin (dir=bwd: its
 * backward kernel forms, where its route has them) under fs_row_bwd_. Replay
 * rebuilds the rows from them (their keys follow from the split and the
 * thread count). */
static inline int vw2__fs_put(vw2_rec_t *dst, const char *pre, const vw2_rec_t *src)
{
    char nm[96];
    int i;
    for (i = 0; i < src->ntok; i++)
    {
        if (src->tok[i].sect != 1) continue;
        if (snprintf(nm, sizeof nm, "%s%s", pre, src->tok[i].name) >= (int)sizeof nm) return -1;
        if (vw2_rec_set(dst, 1, nm, src->tok[i].val) != VW2_OK) return -1;
    }
    return 0;
}
/* the payload tokens under `pre` (and not under `skip`) into dst, the prefix
 * stripped; the count, -1 on failure */
static inline int vw2__fs_get(vw2_rec_t *dst, const vw2_rec_t *src, const char *pre, const char *skip)
{
    const size_t lp = strlen(pre), ls = skip ? strlen(skip) : 0;
    int i, n = 0;
    for (i = 0; i < src->ntok; i++)
    {
        const char *nm = src->tok[i].name;
        if (src->tok[i].sect != 1 || strncmp(nm, pre, lp)) continue;
        if (skip && !strncmp(nm, skip, ls)) continue;
        if (vw2_rec_set(dst, 1, nm + lp, src->tok[i].val) != VW2_OK) return -1;
        n++;
    }
    return n;
}

/* The banked real four-step at the cell: 1 with *n1, *n2 and the child's
 * rows (payload only, *cbwd empty where the row plan has no backward twin;
 * the caller keys them and frees them) filled; 0 when the cell's engine is
 * not zfsr, a token is malformed, or the row carries no child -- a zfsr row
 * from before the child rode on it is a miss, so the cell races again. */
static inline int vw2_real_il_lookup_zfsr(const vw2_store_t *s, int realN, int is_c2r,
                                          int is_inplace, int T, int *n1, int *n2,
                                          vw2_rec_t *c2d, vw2_rec_t *crow, vw2_rec_t *cbwd)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    const char *eng, *sp;
    int a = 0, b = 0;
    *n1 = *n2 = 0;
    memset(c2d, 0, sizeof *c2d);
    memset(crow, 0, sizeof *crow);
    memset(cbwd, 0, sizeof *cbwd);
    vw2_real_il_key(&k, realN, is_c2r, is_inplace, T);
    r = vw2_lookup(s, &k);
    if (!r || vw2__is_seed(r)) return 0;
    eng = vw2_rec_get(r, "eng");
    if (!eng || strcmp(eng, "zfsr")) return 0;
    sp = vw2_rec_get(r, "split");
    if (!sp || sscanf(sp, "%dx%d", &a, &b) != 2 || a <= 0 || b <= 0) return 0;
    if (vw2__fs_get(c2d, r, "fs_", "fs_row_") <= 0 || vw2__fs_get(crow, r, "fs_row_", "fs_row_bwd_") <= 0 ||
        vw2__fs_get(cbwd, r, "fs_row_bwd_", NULL) < 0)
    {
        vw2_rec_free(c2d);
        vw2_rec_free(crow);
        vw2_rec_free(cbwd);
        return 0;
    }
    *n1 = a; *n2 = b;
    return 1;
}

/* Bank the real four-step verdict at the cell (replacing whatever engine held
 * it): the split and the child's rows (cbwd NULL where the row plan has no
 * backward twin) -- a four-step without its child is no recipe, and is not
 * banked. */
static inline int vw2_real_il_bank_zfsr(vw2_store_t *s, int realN, int is_c2r, int is_inplace, int T,
                                        int n1, int n2, const vw2_rec_t *c2d, const vw2_rec_t *crow,
                                        const vw2_rec_t *cbwd, double ns)
{
    vw2_rec_t r;
    char b[48];
    int rc;
    if (!c2d || !crow) return -1;
    memset(&r, 0, sizeof r);
    vw2_real_il_key(&r.key, realN, is_c2r, is_inplace, T);
    snprintf(b, sizeof b, "%dx%d", n1, n2);
    if (vw2_rec_set(&r, 1, "eng", "zfsr") != VW2_OK ||
        vw2_rec_set(&r, 1, "split", b) != VW2_OK ||
        vw2__fs_put(&r, "fs_", c2d) != 0 || vw2__fs_put(&r, "fs_row_", crow) != 0 ||
        (cbwd && vw2__fs_put(&r, "fs_row_bwd_", cbwd) != 0) ||
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
                                          size_t *tile, int *stk, int *mt)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    const char *eng, *ch, *tl, *st;
    int n = 0;
    *nf = 0; *tile = 0; *stk = 3; *mt = 0;
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
    {   /* the threaded arm of a threaded plan's row: 1 BLOCKS, 2 TILES; absent = serial */
        const char *m = vw2_rec_get(r, "mt");
        if (m) { int v = atoi(m); if (v >= 1 && v <= 2) *mt = v; }
    }
    return 1;
}

/* Bank the ZTT-r verdict at the cell (replacing whatever engine held it). */
static inline int vw2_real_il_bank_zttr(vw2_store_t *s, int realN, int is_c2r, int is_inplace, int T,
                                        const int *chain, int nf, size_t tile, int stk, int mt, double ns)
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
    if (mt > 0)
    {
        snprintf(b, sizeof b, "%d", mt);
        if (vw2_rec_set(&r, 1, "mt", b) != VW2_OK) { vw2_rec_free(&r); return -1; }
    }
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
