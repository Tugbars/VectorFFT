/* wisdom2_2d_il_reader.h — the INTERLEAVED rank>=2 cell codec: the lay=il
 * rows of the native IL 2D tier (column chains, forms, axis races), the IL
 * 2D real rows and the rank-3 IL (ilnd) tokens. Carved verbatim out of
 * wisdom2/wisdom2_2d_reader.h (layout separation phase 5). */
#ifndef VFFT_WISDOM2_2D_IL_READER_H
#define VFFT_WISDOM2_2D_IL_READER_H

#include "wisdom2.h"
#include "wisdom2_rank_rows.h" /* the rank>=2 row key / tail / bank helpers (common) */

/* ═══ native IL 2D c2c tier cells (lay=il —
 * docs/roadmap/fft2d_il_c2c_design.md) ═══
 * The tier's raced verdicts live in their OWN lay=il cells: key {t=c2c
 * rank=2 n=N1xN2 q=1 ord pl=OOP lay=il}, payload chain= (dot-separated
 * radices, the COLUMN-pass factorization — raced: 1.30x at 4096x64, 1.18x
 * at 1024x1024 over a greedy largest-radix chain). ord = the cell's order
 * class: scr and nat cells race and bank separately (a natural cell's
 * chain is raced under the natural pass). The lookup goes through
 * vw2_lookup, so a lay-less/split row can come back on the ANY fallback
 * phase — the chain-token check refuses it (split rows carry
 * rowplan/colplan, never chain=). Old binaries: cells invisible + opaque
 * carry (v1.2). */
/* blu: the N1-arm verdict — 0 = the odd chain won, M > 0 = the
 * column-axis Bluestein of length M won (chain= is then the M chain that
 * serves), absent = -1 = unraced. Same token on the real row. */
/* ═══ the COLUMN-AXIS ROW of an interleaved c2c plan, keyed by rank and
 * axis. The 2D tier's row is rank 2, axis 0
 * with the historical token names (chain= wl= tf= ro= cmt= cmtt= blu=
 * forms=); a rank-N plan keys rank 3 and names axis a >= 1's tokens with
 * the axis as a suffix (chain1= wl1= ...) on the SAME row, so one cell is
 * one row. axis 0 creates the row, a later axis updates fields on it. */
typedef struct {
    int rank;          /* 2 or 3 */
    int n0, n1, n2;    /* the cell's dims (n2 = 0 at rank 2) */
    int ord;           /* VW2_ORD_SCR / VW2_ORD_NAT */
    int axis;          /* 0 = the historical tokens; a >= 1 = suffixed */
    int real;          /* 1 = the real tier's row (t=r2c) */
    int nthreads;      /* the plan's thread count (v1.3): a threaded plan's
                        * row is its own, complete on its own; 0/1 = one */
    int ip;            /* 1 = the IN-PLACE real cell's row (pl=ip): its own race,
                        * its own cells, never the out-of-place cell's
                        * (docs/roadmap/real_inplace_design.md, 2026-10-08);
                        * 2 = the cell's PITCH TWIN (door 2, the plan's own plane
                        * off the caller's pitch): the same pl=ip row, its own
                        * pitch-sensitive verdicts as the _p token set;
                        * 0 = pl=oop: the out-of-place real row, and the c2c
                        * tier's placement-blind row */
} vw2_ilcol_key_t;

static inline const char *vw2__ilcol_tok(const vw2_ilcol_key_t *k, const char *base,
                                         char *buf, size_t sz)
{
    if (k->axis <= 0 && k->ip != 2) return base;
    if (k->axis <= 0)
        snprintf(buf, sz, "%s_p", base);                            /* door 2's own verdicts */
    else
        snprintf(buf, sz, "%s%d%s", base, k->axis, k->ip == 2 ? "_p" : "");
    return buf;
}
static inline void vw2__ilcol_key(const vw2_ilcol_key_t *ck, vw2_key_t *k)
{
    vw2__2d_key(k, ck->real ? VW2_T_R2C : VW2_T_C2C, ck->rank, ck->n0, ck->n1,
                ck->n2, ck->ord, VW2_LAY_IL, ck->nthreads);
    if (ck->ip)
        k->pl = VW2_PL_IP;   /* the in-place real cell's own row */
}

/* one integer verdict on a column row, by base name (axis-suffixed like the
 * chain tokens): ABSENT = dflt. tpc= (the turned prime pass) is
 * the first; a later axis verdict that is not part of the chain bank rides
 * the same pair. The set refuses (-1) when the row does not exist. */
static inline int vw2_ilcol_tok_geti(const vw2_store_t *s, const vw2_ilcol_key_t *ck,
                                     const char *base, int dflt)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    const char *v;
    char tb[16];
    vw2__ilcol_key(ck, &k);
    r = vw2_lookup(s, &k);
    if (!r) return dflt;
    v = vw2_rec_get(r, vw2__ilcol_tok(ck, base, tb, sizeof tb));
    return v ? atoi(v) : dflt;
}
static inline int vw2_ilcol_tok_seti(vw2_store_t *st, const vw2_ilcol_key_t *ck,
                                     const char *base, int val)
{
    vw2_key_t k;
    char tb[16], vb[24];
    vw2__ilcol_key(ck, &k);
    if (!vw2_lookup(st, &k)) return -1;
    snprintf(vb, sizeof vb, "%d", val);
    return vw2_update_field(st, &k, vw2__ilcol_tok(ck, base, tb, sizeof tb), vb) == VW2_OK ? VW2_OK : -1;
}

static inline int vw2_ilcol_chain_lookup(const vw2_store_t *s, const vw2_ilcol_key_t *ck,
                                         int *Rs, int *nst,
                                         int *wl, int *tf, int *ro,
                                         int *cmt, int *cmtt, int *blu)
{
    /* ord: VW2_ORD_SCR = the scrambled-comb serving; VW2_ORD_NAT = the
     * natural cell — its chain is raced under the natural pass and MUST
     * NOT share the scr row. */
    vw2_key_t k;
    const vw2_rec_t *r;
    const char *cv;
    char tb[16];
    int m = 0;
    vw2__ilcol_key(ck, &k);
    r = vw2_lookup(s, &k);
    if (!r) return 0;
    cv = vw2_rec_get(r, vw2__ilcol_tok(ck, "chain", tb, sizeof tb));
    if (!cv) return 0;                       /* an ANY/split row: refuse */
    /* the axis verdicts; ABSENT = -1 (chain-only vintage / unraced) */
    if (wl) { const char *v = vw2_rec_get(r, vw2__ilcol_tok(ck, "wl", tb, sizeof tb)); *wl = v ? atoi(v) : -1; }
    if (tf) { const char *v = vw2_rec_get(r, vw2__ilcol_tok(ck, "tf", tb, sizeof tb)); *tf = v ? atoi(v) : -1; }
    if (ro) { const char *v = vw2_rec_get(r, vw2__ilcol_tok(ck, "ro", tb, sizeof tb)); *ro = v ? atoi(v) : -1; }
    /* the MT verdict; the thread count it was raced at is the row's key
     * (v1.3) -- cmtt is read only from a pre-1.3 row that load did not split */
    if (cmt) { const char *v = vw2_rec_get(r, vw2__ilcol_tok(ck, "cmt", tb, sizeof tb)); *cmt = v ? atoi(v) : -1; }
    if (cmtt) { const char *v = vw2_rec_get(r, vw2__ilcol_tok(ck, "cmtt", tb, sizeof tb)); *cmtt = v ? atoi(v) : -1; }
    if (blu) { const char *v = vw2_rec_get(r, vw2__ilcol_tok(ck, "blu", tb, sizeof tb)); *blu = v ? atoi(v) : -1; }
    while (*cv && m < 8) {
        int v = 0;
        if (*cv < '0' || *cv > '9') return 0;
        while (*cv >= '0' && *cv <= '9') v = v * 10 + (*cv++ - '0');
        if (v < 2) return 0;
        Rs[m++] = v;
        if (*cv == '.') cv++;
        else break;
    }
    if (!m || *cv) return 0;
    *nst = m;
    return 1;
}
static inline int vw2_2d_il_chain_lookup(const vw2_store_t *s, int N1,
                                         int N2, int *Rs, int *nst,
                                         int *wl, int *tf, int *ro,
                                         int *cmt, int *cmtt, int *blu,
                                         int ord, int T)
{
    vw2_ilcol_key_t ck = { 2, N1, N2, 0, ord, 0, 0, T };
    return vw2_ilcol_chain_lookup(s, &ck, Rs, nst, wl, tf, ro, cmt, cmtt, blu);
}

/* the row's MEASURE refreshed in place (a chain race that confirmed the
 * row's chain): ns= metric= units= src=race, the date and the build stamp;
 * the payload untouched, an import's from= dropped (the row is this
 * machine's measurement now) */
static inline void vw2__ilcol_measure(vw2_store_t *st, const vw2_key_t *k, double ns)
{
    int i;
    for (i = 0; i < st->nrec; i++)
        if (vw2_key_eq(&st->rec[i].key, k)) {
            vw2_rec_t *r = &st->rec[i];
            char v[48];
            if (st->poisoned[r->shard]) return;
            snprintf(v, sizeof v, "%.1f", ns);
            if (vw2_rec_set(r, 2, "ns", v) != VW2_OK) return;
            (void)vw2_rec_set(r, 2, "metric", "fwd1");
            (void)vw2_rec_set(r, 2, "units", "ns");
            (void)vw2_rec_set(r, 2, "src", "race");
            vw2__rec_del(r, "from");
            vw2__2d_stamp_date(r);
            st->dirty[r->shard] = 1;
            r->own = 1;
            if (st->build[0]) (void)vw2_rec_set(r, 2, "bld", st->build);
            return;
        }
}

/* bank one axis's chain and verdicts: axis 0 writes the row (the 2D
 * tier's record, unchanged), a later axis updates its suffixed tokens on
 * the row axis 0 wrote — the row must exist. Negative verdicts are not
 * emitted (unraced). Returns VW2_OK or -1. */
static inline int vw2_ilcol_chain_bank(vw2_store_t *st, const vw2_ilcol_key_t *ck,
                                       const int *Rs, int nst,
                                       int wl, int tf, int ro,
                                       int cmt, int cmtt, int blu, double ns)
{
    vw2_rec_t rec;
    vw2_rec_t *r = &rec;
    const char *why = NULL;
    char b[64];
    int i, off = 0;
    for (i = 0; i < nst && off < (int)sizeof b - 8; i++)
        off += snprintf(b + off, sizeof b - off, "%s%d", i ? "." : "",
                        Rs[i]);
    /* the field-update path: a later axis always (its tokens live on the
     * row axis 0 wrote), and axis 0 itself when the row EXISTS and this
     * bank carries no measurement (ns <= 0: the N-arm verdict banked after
     * the chain race, or after a replayed chain) — a fresh measure-less
     * record would be refused against the measured row (the metric law),
     * and the N-arm verdict would never land — or MEASURES THE ROW'S OWN
     * CHAIN (a chain race, a recalibrate's, that landed where the row
     * stands): the row is updated in place, every verdict it carries that
     * this bank does not survives — the other direction's on the
     * direction-shared real row, the children, the forms — and the
     * measurement is refreshed. A chain race that lands elsewhere replaces
     * the row, the verdicts raced against the old chain with it. */
    {
        vw2_key_t k;
        char tb[16], vb[24];
        const vw2_rec_t *have;
        int same = 0;
        vw2__ilcol_key(ck, &k);
        have = vw2_lookup(st, &k);
        if (ck->axis > 0 && !have) {
            fprintf(stderr, "[wisdom2] il column axis %d bank refused (no row)\n", ck->axis);
            return -1;
        }
        if (have) {
            const char *hc = vw2_rec_get(have, vw2__ilcol_tok(ck, "chain", tb, sizeof tb));
            same = hc && !strcmp(hc, b);
        }
        if (ck->axis > 0 || (have && (ns <= 0.0 || same || ck->ip == 2))) {   /* the pitch twin: its own tokens on the cell's row */
        if (vw2_update_field(st, &k, vw2__ilcol_tok(ck, "chain", tb, sizeof tb), b) != VW2_OK) return -1;
#define VW2__ILCOL_UPD(base, val) do { \
        snprintf(vb, sizeof vb, "%d", (val)); \
        if (vw2_update_field(st, &k, vw2__ilcol_tok(ck, base, tb, sizeof tb), vb) != VW2_OK) return -1; \
    } while (0)
        if (wl >= 0) VW2__ILCOL_UPD("wl", wl);
        if (tf >= 0) VW2__ILCOL_UPD("tf", tf);
        if (ro >= 0) VW2__ILCOL_UPD("ro", ro);
        if (cmt >= 0 && cmtt > 0) VW2__ILCOL_UPD("cmt", cmt);   /* the T is the row's key (v1.3) */
        if (blu >= 0) VW2__ILCOL_UPD("blu", blu);
#undef VW2__ILCOL_UPD
        if (ns > 0.0) vw2__ilcol_measure(st, &k, ns);   /* the re-race's time onto the row it confirmed */
        return VW2_OK;
        }
    }
    memset(r, 0, sizeof *r);
    vw2__2d_rec_key(r, ck->real ? VW2_T_R2C : VW2_T_C2C, ck->rank, ck->n0, ck->n1, ck->n2, ck->ord,
                    /*migrated=*/0, /*ord_blind=*/0, VW2_LAY_IL, ck->nthreads);
    if (ck->ip)
        r->key.pl = VW2_PL_IP;   /* the in-place real cell's own row (the key's placement) */
    if (vw2_rec_set(r, 1, "chain", b) != VW2_OK) {
        vw2_rec_free(r);
        fprintf(stderr, "[wisdom2] il2d chain bank refused (token)\n");
        return -1;
    }
    /* the axis verdicts (negative = unraced: token not emitted) */
    if (wl >= 0) {
        snprintf(b, sizeof b, "%d", wl);
        if (vw2_rec_set(r, 1, "wl", b) != VW2_OK) goto tokfail;
    }
    if (tf >= 0) {
        snprintf(b, sizeof b, "%d", tf);
        if (vw2_rec_set(r, 1, "tf", b) != VW2_OK) goto tokfail;
    }
    if (ro >= 0) {
        snprintf(b, sizeof b, "%d", ro);
        if (vw2_rec_set(r, 1, "ro", b) != VW2_OK) goto tokfail;
    }
    if (cmt >= 0 && cmtt > 0) {   /* the MT verdict; its T is the row's key (v1.3) */
        snprintf(b, sizeof b, "%d", cmt);
        if (vw2_rec_set(r, 1, "cmt", b) != VW2_OK) goto tokfail;
    }
    if (blu >= 0) {               /* the N1-arm verdict */
        snprintf(b, sizeof b, "%d", blu);
        if (vw2_rec_set(r, 1, "blu", b) != VW2_OK) goto tokfail;
    }
    if (0) {
    tokfail:
        vw2_rec_free(r);
        fprintf(stderr, "[wisdom2] il2d axis bank refused (token)\n");
        return -1;
    }
    if (vw2__2d_tail(r, ns, "race", NULL, &why)) {
        fprintf(stderr, "[wisdom2] il2d chain bank refused (%s)\n",
                why ? why : "?");
        return -1;
    }
    return vw2__2d_bank(st, r, 0);
}
static inline int vw2_2d_il_chain_bank(vw2_store_t *st, int N1, int N2,
                                       const int *Rs, int nst,
                                       int wl, int tf, int ro,
                                       int cmt, int cmtt, int blu, double ns,
                                       int ord, int T)
{
    vw2_ilcol_key_t ck = { 2, N1, N2, 0, ord, 0, 0, T };
    return vw2_ilcol_chain_bank(st, &ck, Rs, nst, wl, tf, ro, cmt, cmtt, blu, ns);
}

/* one integer token on the 2D il c2c chain row — the strip width (sw=, the
 * serial unbanded walk's tile) and the threaded arm's shape (mtarm=, msw=,
 * on the plan's own T row): il2d_large_plane_design.md. Absent = dflt; a
 * set on a missing row is refused (the chain bank makes the row). */
static inline int vw2_2d_il_tok_geti(const vw2_store_t *s, int N1, int N2, int ord, int T,
                                     const char *name, int dflt)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    const char *v;
    vw2_ilcol_key_t ck = { 2, N1, N2, 0, ord, 0, 0, T };
    vw2__ilcol_key(&ck, &k);
    r = vw2_lookup(s, &k);
    if (!r) return dflt;
    v = vw2_rec_get(r, name);
    return v ? atoi(v) : dflt;
}
static inline int vw2_2d_il_tok_seti(vw2_store_t *st, int N1, int N2, int ord, int T,
                                     const char *name, int val)
{
    vw2_key_t k;
    char vb[24];
    vw2_ilcol_key_t ck = { 2, N1, N2, 0, ord, 0, 0, T };
    vw2__ilcol_key(&ck, &k);
    if (!vw2_lookup(st, &k)) return -1;
    snprintf(vb, sizeof vb, "%d", val);
    return vw2_update_field(st, &k, name, vb) == VW2_OK ? 0 : -1;
}

/* ═══ native IL 2D REAL tier cells (lay=il —
 * docs/roadmap/fft2d_real_il_design.md). Key {t=r2c rank=2 n=N1xN2 q=1
 * ord pl=OOP lay=il} — DIRECTION-SHARED: the c2r create reads the
 * r2c-keyed row (the pair law requires ONE chain for both directions).
 * The AXES are raced per direction (r2c and c2r have different row
 * kernels and a different column pass), so the shared row carries ONE
 * TOKEN SET PER DIRECTION — r2c = wl cmt cmtt, c2r = wl_c2r cmt_c2r
 * cmtt_c2r — or the two directions would overwrite each other's tokens.
 * COLLISION-FREE with the split tier's real cells (vw2_2d_r2c_lookup): the
 * lay axis separates them, and their rowplan/colplan payload (never
 * chain=) makes the chain-token check refuse a vintage lay=ANY row on the
 * fallback phase. Payload: chain= (the raced column-pass factorization) +
 * wl= (the banded column walk's band width in ROWS; 0 = unbanded — rows
 * sit OUTSIDE the walk per §2.5, tfuse structurally absent for real).
 * ABSENT axis -> -1 = unraced. (The rw= row-route token -- the ROWSPLIT
 * band on the split engines -- was retired 2026-10-03 with that route; an
 * old row's rw= is dead text.) */
/* the direction's token names on the shared real IL row */
static inline const char *vw2__rl_tok(int is_c2r, int which)
{
    static const char *const R2C[3] = { "wl", "cmt", "cmtt" };
    static const char *const C2R[3] = { "wl_c2r", "cmt_c2r", "cmtt_c2r" };
    return is_c2r ? C2R[which] : R2C[which];
}
/* the same at the plan's door: door 2 (ip == 2) keeps its own set, the _p suffix */
static inline const char *vw2__rl_tokp(int is_c2r, int which, int ip, char *buf, size_t sz)
{
    const char *b = vw2__rl_tok(is_c2r, which);
    if (ip != 2) return b;
    snprintf(buf, sz, "%s_p", b);
    return buf;
}
static inline const char *vw2__rl_name(const char *base, int ip, char *buf, size_t sz)
{
    if (ip != 2) return base;
    snprintf(buf, sz, "%s_p", base);
    return buf;
}

/* one string token on the shared real IL row -- the r2c row engine's verdict
 * (rx=, rxs=: il/rank2/il2d_real_plan.h). Absent = NULL; a set on a missing
 * row is refused (the chain bank makes the row). */
/* the real row's key: ip = 1 is the in-place cell's own row (pl=ip), 0 the
 * out-of-place cell's (real_inplace_design.md, 2026-10-08) */
static inline void vw2__2d_rl_key(vw2_key_t *k, int N1, int N2, int ord, int T, int ip)
{
    vw2__2d_key(k, VW2_T_R2C, 2, N1, N2, 0, ord, VW2_LAY_IL, T);
    if (ip)
        k->pl = VW2_PL_IP;
}
static inline const char *vw2_2d_rl_tok_gets(const vw2_store_t *s, int N1, int N2, int ord, int T,
                                             const char *name, int ip)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    vw2__2d_rl_key(&k, N1, N2, ord, T, ip);
    r = vw2_lookup(s, &k);
    return r ? vw2_rec_get(r, name) : NULL;
}
static inline int vw2_2d_rl_tok_sets(vw2_store_t *st, int N1, int N2, int ord, int T,
                                     const char *name, const char *val, int ip)
{
    vw2_key_t k;
    vw2__2d_rl_key(&k, N1, N2, ord, T, ip);
    if (!vw2_lookup(st, &k)) return -1;
    return vw2_update_field(st, &k, name, val) == VW2_OK ? 0 : -1;
}

static inline int vw2_2d_rl_lookup(const vw2_store_t *s, int N1, int N2,
                                   int is_c2r,
                                   int *Rs, int *nst, int *wl,
                                   int *cmt, int *cmtt, int *blu, int ord, int T, int ip)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    const char *cv;
    char tb[24];
    int m = 0;
    vw2__2d_rl_key(&k, N1, N2, ord, T, ip);
    r = vw2_lookup(s, &k);
    if (!r) return 0;
    cv = vw2_rec_get(r, vw2__rl_name("chain", ip, tb, sizeof tb));
    if (!cv) return 0;                       /* a veneer/ANY row, or no verdict at this door: refuse */
    if (wl) { const char *v = vw2_rec_get(r, vw2__rl_tokp(is_c2r, 0, ip, tb, sizeof tb)); *wl = v ? atoi(v) : -1; }
    /* cmt = the COLUMN-PASS MT verdict (1 = thread it, 0 = serial) at the
     * row's own thread count (the key's nthreads, v1.3); cmtt is read only
     * from a pre-1.3 row that load did not split. */
    if (cmt) { const char *v = vw2_rec_get(r, vw2__rl_tokp(is_c2r, 1, ip, tb, sizeof tb)); *cmt = v ? atoi(v) : -1; }
    if (cmtt) { const char *v = vw2_rec_get(r, vw2__rl_tokp(is_c2r, 2, ip, tb, sizeof tb)); *cmtt = v ? atoi(v) : -1; }
    if (blu) { const char *v = vw2_rec_get(r, vw2__rl_name("blu", ip, tb, sizeof tb)); *blu = v ? atoi(v) : -1; }  /* direction-shared */
    while (*cv && m < 8) {
        int v = 0;
        if (*cv < '0' || *cv > '9') return 0;
        while (*cv >= '0' && *cv <= '9') v = v * 10 + (*cv++ - '0');
        if (v < 2) return 0;
        Rs[m++] = v;
        if (*cv == '.') cv++;
        else break;
    }
    if (!m || *cv) return 0;
    *nst = m;
    return 1;
}

static inline int vw2_2d_rl_bank(vw2_store_t *st, int N1, int N2,
                                 int is_c2r,
                                 const int *Rs, int nst, int wl,
                                 int cmt, int cmtt, int blu, double ns,
                                 int ord, int T, int ip)
{
    vw2_rec_t rec;
    vw2_rec_t *r = &rec;
    const char *why = NULL;
    char b[64], t0[24], t1[24], tc[24], tbl[24];
    const char *ctk = vw2__rl_name("chain", ip, tc, sizeof tc), *btk = vw2__rl_name("blu", ip, tbl, sizeof tbl);
    const char *wtk = vw2__rl_tokp(is_c2r, 0, ip, t0, sizeof t0), *mtk = vw2__rl_tokp(is_c2r, 1, ip, t1, sizeof t1);
    int i, off = 0;
    for (i = 0; i < nst && off < (int)sizeof b - 8; i++)
        off += snprintf(b + off, sizeof b - off, "%s%d", i ? "." : "",
                        Rs[i]);
    /* MERGE into the shared row when its chain is the same: only THIS
     * direction's tokens move, the other direction's verdicts survive.
     * Unraced axes (-1 / cmtt 0) never erase a banked token. (The chain
     * bank, vw2_ilcol_chain_bank, obeys the same law: a chain race that
     * lands on the row's chain updates it in place.) The pitch twin (ip 2)
     * always merges: its _p set lives on the cell's row beside door 1's. */
    {
        vw2_key_t k;
        const vw2_rec_t *have;
        vw2__2d_rl_key(&k, N1, N2, ord, T, ip);
        have = vw2_lookup(st, &k);
        if (have && (ip == 2 || (vw2_rec_get(have, ctk) && !strcmp(vw2_rec_get(have, ctk), b)))) {
            char v[24];
            int rc = VW2_OK;
            if (ip == 2) rc |= vw2_update_field(st, &k, ctk, b);
            if (wl >= 0) { snprintf(v, sizeof v, "%d", wl); rc |= vw2_update_field(st, &k, wtk, v); }
            if (cmt >= 0 && cmtt > 0) {   /* the T is the row's key (v1.3) */
                snprintf(v, sizeof v, "%d", cmt);  rc |= vw2_update_field(st, &k, mtk, v);
            }
            if (blu >= 0) { snprintf(v, sizeof v, "%d", blu); rc |= vw2_update_field(st, &k, btk, v); }
            if (rc != VW2_OK)
                fprintf(stderr, "[wisdom2] il2d real merge refused (%s)\n",
                        is_c2r ? "c2r" : "r2c");
            return rc == VW2_OK ? VW2_OK : -1;
        }
    }
    memset(r, 0, sizeof *r);
    vw2__2d_rec_key(r, VW2_T_R2C, 2, N1, N2, 0, ord,
                    /*migrated=*/0, /*ord_blind=*/0, VW2_LAY_IL, T);
    if (ip)
        r->key.pl = VW2_PL_IP;   /* the in-place cell's own row */
    if (vw2_rec_set(r, 1, ctk, b) != VW2_OK) {
        vw2_rec_free(r);
        fprintf(stderr, "[wisdom2] il2d real bank refused (token)\n");
        return -1;
    }
    if (wl >= 0) {
        snprintf(b, sizeof b, "%d", wl);
        if (vw2_rec_set(r, 1, wtk, b) != VW2_OK) {
            vw2_rec_free(r);
            fprintf(stderr, "[wisdom2] il2d real wl bank refused (token)\n");
            return -1;
        }
    }
    if (cmt >= 0 && cmtt > 0) {   /* the column-MT verdict; its T is the row's key (v1.3) */
        snprintf(b, sizeof b, "%d", cmt);
        if (vw2_rec_set(r, 1, mtk, b) != VW2_OK) {
            vw2_rec_free(r);
            fprintf(stderr, "[wisdom2] il2d real cmt bank refused (token)\n");
            return -1;
        }
    }
    if (blu >= 0) {               /* the N1-arm verdict, direction-shared */
        snprintf(b, sizeof b, "%d", blu);
        if (vw2_rec_set(r, 1, btk, b) != VW2_OK) {
            vw2_rec_free(r);
            fprintf(stderr, "[wisdom2] il2d real blu bank refused (token)\n");
            return -1;
        }
    }
    if (vw2__2d_tail(r, ns, "race", NULL, &why)) {
        fprintf(stderr, "[wisdom2] il2d real bank refused (%s)\n",
                why ? why : "?");
        return -1;
    }
    return vw2__2d_bank(st, r, 0);
}

/* per-stage kernel FORMS on the IL chain rows - the c2c chain row (t=c2c
 * lay=il) and the direction-shared real row (t=r2c lay=il):
 * forms=<name>.<name>... one per chain stage
 * ("-" = the stage's single form; r32 b48|b84, r64 b88|b416). Merged onto
 * the existing row (vw2_update_field) after the chain is banked; a row
 * without a chain carries no forms. */
/* THE FORM TOKEN'S BASE NAME: "forms" describes the row's own chain=;
 * "bluforms" describes the column-axis Bluestein's INNER chain at M, which
 * shares the row but not the chain. One row, two independent stage lists,
 * neither able to overwrite the other -- a second row keyed (M, N2) would
 * be a row a user's own M x N2 cell owns. */
static inline int vw2_ilcol_forms_lookup_base(vw2_store_t *s, const vw2_ilcol_key_t *ck,
                                              const char *base, char *out, size_t osz)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    const char *v;
    char tb[16];
    vw2__ilcol_key(ck, &k);
    r = vw2_lookup(s, &k);
    if (!r || !vw2_rec_get(r, vw2__ilcol_tok(ck, "chain", tb, sizeof tb))) return 0;
    v = vw2_rec_get(r, vw2__ilcol_tok(ck, base, tb, sizeof tb));
    if (!v || !*v) return 0;
    snprintf(out, osz, "%s", v);
    return 1;
}
static inline int vw2_ilcol_forms_lookup(vw2_store_t *s, const vw2_ilcol_key_t *ck,
                                         char *out, size_t osz)
{
    return vw2_ilcol_forms_lookup_base(s, ck, "forms", out, osz);
}
static inline int vw2_ilcol_forms_bank_base(vw2_store_t *s, const vw2_ilcol_key_t *ck,
                                            const char *base, const char *forms)
{
    vw2_key_t k;
    char tb[16];
    vw2__ilcol_key(ck, &k);
    return vw2_update_field(s, &k, vw2__ilcol_tok(ck, base, tb, sizeof tb), forms) == VW2_OK;
}
static inline int vw2_ilcol_forms_bank(vw2_store_t *s, const vw2_ilcol_key_t *ck,
                                       const char *forms)
{
    return vw2_ilcol_forms_bank_base(s, ck, "forms", forms);
}
/* THE ROW BEFORE ITS VERDICTS. Every bank after the chain step
 * is a field update on the axis-0 row (vw2_update_field), and only the chain
 * RACE writes that row: a pinned or a served chain never does. A tier that
 * banks a structure, width, form or forms verdict calls this first; it
 * writes the row with the chain and no verdict tokens when none exists.
 * 1 = written (persist), 0 = the row was there. */
static inline int vw2_ilcol_row_ensure(vw2_store_t *s, const vw2_ilcol_key_t *ck,
                                       const int *Rs, int nst)
{
    vw2_key_t k;
    vw2__ilcol_key(ck, &k);
    if (vw2_lookup(s, &k)) return 0;
    return vw2_ilcol_chain_bank(s, ck, Rs, nst, -1, -1, -1, -1, -1, -1, 0.0) == VW2_OK;
}
/* A FORMS VERDICT RACED BEFORE ITS ROW EXISTED. The column
 * builder races the per-stage forms right after the chain step and banks
 * them there; with no row yet that bank fails and the verdict lives only in
 * the builder's out-buffer. The create re-banks it here once the row is
 * written. Nothing is written when the row already carries this verdict (a
 * served one: a warm create writes nothing) or when there is no row (an
 * env-pinned axis: pins never bank). 1 = written (persist). */
static inline int vw2_ilcol_forms_rebank_base(vw2_store_t *s, const vw2_ilcol_key_t *ck,
                                              const char *base, const char *forms)
{
    char have[64];
    if (!forms || !forms[0]) return 0;
    if (vw2_ilcol_forms_lookup_base(s, ck, base, have, sizeof have) && !strcmp(have, forms))
        return 0;
    return vw2_ilcol_forms_bank_base(s, ck, base, forms);
}
static inline int vw2_ilcol_forms_rebank(vw2_store_t *s, const vw2_ilcol_key_t *ck,
                                         const char *forms)
{
    return vw2_ilcol_forms_rebank_base(s, ck, "forms", forms);
}
/* the rank-N IL tier's STRUCTURE verdict (fftnd_il.h): s= on the rank-3
 * lay=il row that axis 0's chain bank created — 1 = the child per plane,
 * 2 = the flat tier; 0 = absent */
static inline int vw2_ilnd_int_lookup(const vw2_store_t *s, const vw2_ilcol_key_t *ck,
                                      const char *name)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    const char *v;
    vw2__ilcol_key(ck, &k);
    r = vw2_lookup(s, &k);
    if (!r) return 0;
    v = vw2_rec_get(r, name);
    return v ? atoi(v) : 0;
}
static inline int vw2_ilnd_int_bank(vw2_store_t *s, const vw2_ilcol_key_t *ck,
                                    const char *name, int val)
{
    vw2_key_t k;
    char b[16];
    vw2__ilcol_key(ck, &k);
    snprintf(b, sizeof b, "%d", val);
    return vw2_update_field(s, &k, name, b) == VW2_OK;
}
static inline int vw2_ilnd_arm_lookup(const vw2_store_t *s, const vw2_ilcol_key_t *ck)
{
    return vw2_ilnd_int_lookup(s, ck, "s");
}
static inline int vw2_ilnd_arm_bank(vw2_store_t *s, const vw2_ilcol_key_t *ck, int arm)
{
    return vw2_ilnd_int_bank(s, ck, "s", arm);
}
/* cmts= : the STRUCTURE the MT verdict (cmt=) runs with — raced jointly
 * with the partition arm at the plan's T; it may differ from s=, the
 * one-thread verdict, and serves only with cmt */
static inline int vw2_ilnd_mts_lookup(const vw2_store_t *s, const vw2_ilcol_key_t *ck)
{
    return vw2_ilnd_int_lookup(s, ck, "cmts");
}
static inline int vw2_ilnd_mts_bank(vw2_store_t *s, const vw2_ilcol_key_t *ck, int arm)
{
    return vw2_ilnd_int_bank(s, ck, "cmts", arm);
}
/* cmtp= : the PLANE TEAM of the plane arm (cmt=2), the workers its plane
 * phase runs on — raced at the plan's T beside the full team wherever a
 * worker of the full team would hold a single plane, and banked with the
 * verdict whenever it was raced (the full team's width included); 0 =
 * absent = the full team */
static inline int vw2_ilnd_ptw_lookup(const vw2_store_t *s, const vw2_ilcol_key_t *ck)
{
    return vw2_ilnd_int_lookup(s, ck, "cmtp");
}
static inline int vw2_ilnd_ptw_bank(vw2_store_t *s, const vw2_ilcol_key_t *ck, int workers)
{
    return vw2_ilnd_int_bank(s, ck, "cmtp", workers);
}

static inline int vw2_2d_forms_lookup(vw2_store_t *s, int is_real, int N1,
                                      int N2, char *out, size_t osz,
                                      int ord, int T)
{
    vw2_ilcol_key_t ck = { 2, N1, N2, 0, ord, 0, is_real, T };
    return vw2_ilcol_forms_lookup(s, &ck, out, osz);
}
static inline int vw2_2d_forms_bank(vw2_store_t *s, int is_real, int N1,
                                    int N2, const char *forms, int ord, int T, int ip)
{
    vw2_ilcol_key_t ck = { 2, N1, N2, 0, ord, 0, is_real, T, ip };
    return vw2_ilcol_forms_bank(s, &ck, forms);
}
static inline int vw2_2d_forms_rebank(vw2_store_t *s, int is_real, int N1,
                                      int N2, const char *forms, int ord, int T, int ip)
{
    vw2_ilcol_key_t ck = { 2, N1, N2, 0, ord, 0, is_real, T, ip };
    return vw2_ilcol_forms_rebank(s, &ck, forms);
}

#endif /* VFFT_WISDOM2_2D_IL_READER_H */
