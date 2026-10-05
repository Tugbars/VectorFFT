/* wisdom2_oop_il.h — the INTERLEAVED library's OOP-family wisdom: its K=1
 * record and codec.
 *
 * vfft_oop_il_entry_t carries what an interleaved K=1 request is served from:
 * the IL axis of the kind-3 cell (il_route, the pair, chain3's chain, the flat
 * DIT's chain + forms + tile, ZTURN-T's chain, the forms verdict il_kv). The
 * codec reads and writes only interleaved rows: the lay=il kind-3 rows and the
 * IL fields of the lay-less vintage row, the dir=bwd backward sibling, the IL
 * prime METHOD row, the zr2c (kind-5) real-composite rows, and the per-T copy.
 *
 * Layout separation phase 5, the kind-3 record split (owner's option (a),
 * 2026-09-27): every function here is the interleaved half of the pre-split
 * wisdom2_oop_reader.h / wisdom2_oop.h code, verbatim but for the types. The
 * split half is split/wisdom/wisdom2_oop_split.h. */
#ifndef VFFT_WISDOM2_OOP_IL_H
#define VFFT_WISDOM2_OOP_IL_H

#include <stdio.h>
#include "wisdom2_oop_rows.h"     /* the row mechanics (common) */
#include "wisdom2_oop_legacy.h"   /* the frozen legacy entry (common): the converter below */
#include "common/abi/route_ids.h" /* VFFT_K1_IL_* */

/* ---------------------------------------------------------------- the record */
typedef struct {
    int    N;
    size_t K;                             /* the run the verdict measured (ran=) */
    int    kind;                          /* VFFT_OOP_KIND_BAILEY2V (kind-3 cells) */
    int    k1_il_route;                   /* VFFT_K1_IL_*; -1 = unraced */
    int    il_R1, il_R2;                  /* the pair (the four-step's N1 x N2) */
    int    il_c3[3];                      /* chain3: il_chain=R2.A.B; zeros = absent */
    int    il_fl[10];                     /* the flat DIT's chain (il_flat=) */
    int    il_fl_n;
    char   il_flf[24];                    /* its per-stage form letters (il_forms=) */
    int    il_tw;                         /* the raced tile (flat DIT, ZTURN-T); 0 = untiled */
    int    il_zt[7];                      /* ZTURN-T's chain (il_ztt=) */
    int    il_zt_n;
    int    place_ip;                      /* the place=ip cell's own row */
    int    nthreads;                      /* the plan's thread count (v1.3) */
    int    ord_scr;                       /* the ord=scr cell's own row */
    int    il_kv;                         /* the raced kernel-forms verdict (see
                                           * wisdom2_oop_legacy.h for the packing) */
    int    il_kv_raced;                   /* 1 = the forms were RACED (explicit 0 counts) */
    int    role;
    double ns;                            /* measured (informational) */
} vfft_oop_il_entry_t;

/* a legacy (dual) entry's INTERLEAVED fields */
static inline void vfft_oop_il_from_legacy(vfft_oop_il_entry_t *o,
                                           const vfft_oop_wisdom_entry_t *e)
{
    memset(o, 0, sizeof *o);
    o->N = e->N; o->K = e->K; o->kind = e->kind;
    o->k1_il_route = e->k1_il_route; o->il_R1 = e->il_R1; o->il_R2 = e->il_R2;
    memcpy(o->il_c3, e->il_c3, sizeof o->il_c3);
    memcpy(o->il_fl, e->il_fl, sizeof o->il_fl); o->il_fl_n = e->il_fl_n;
    memcpy(o->il_flf, e->il_flf, sizeof o->il_flf); o->il_tw = e->il_tw;
    memcpy(o->il_zt, e->il_zt, sizeof o->il_zt); o->il_zt_n = e->il_zt_n;
    o->place_ip = e->place_ip; o->nthreads = e->nthreads; o->ord_scr = e->ord_scr;
    o->il_kv = e->il_kv; o->il_kv_raced = e->il_kv_raced;
    o->role = e->role; o->ns = e->ns;
}

/* ── kind-5 zr_kv codec — THE one definition of the packing (vfft.c banks
 * with it, _zr2c_build reads with it; a bench that wants to inspect a
 * verdict uses these, never raw shifts).
 *   slot: 0 = r2c OOP · 1 = r2c IN-PLACE · 2 = c2r OOP · 3 = c2r IN-PLACE
 *   field: 0 = unmeasured · 1 = child route 0 (OOP-IL) · 2 = route 1
 *          (the NAT-IP in-place child, zr2c_build.h). vfft_zr2c_kv_set
 *          takes the ROUTE (0/1). */
static inline int vfft_zr2c_kv_slot(int is_c2r, int is_inplace)
{ return ((is_c2r ? 1 : 0) << 1) | (is_inplace ? 1 : 0); }
static inline int vfft_zr2c_kv_get(int kv, int slot)
{ return (kv >> (2 * slot)) & 3; }
static inline int vfft_zr2c_kv_set(int kv, int slot, int route)
{ return (kv & ~(3 << (2 * slot))) | (((route ? 2 : 1)) << (2 * slot)); }


/* ------------------------------------------------------------ name maps */
/* THE IL ROUTE SET, declared once. Route ids come from VFFT_K1_IL_* in
 * oop/oop_plan.h; the largest is the only number this file may know, and
 * the table's length is checked against it at COMPILE time: a table shorter
 * than the enum banks every new route's verdict as `il_route=?`, which the
 * store refuses — silently.
 *
 * The table is declared UNSIZED on purpose: its length comes from the
 * NAMES, so the assertion below compares the names against the enum. Give
 * the array an explicit [MAX + 1] and the check becomes circular — it then
 * measures the macro against itself and passes with a NULL hole, which is
 * precisely the bug it is meant to catch (verified by negative test). */
#define VW2_OOP_IL_ROUTE_MAX VFFT_K1_IL_FS
static const char *vw2_oop_il_name[] = {   /* length from the NAMES, checked below */
    "none", "legacy3p", "legacy2p", "mono", "-", "2p", "chain3", "prime", "flat", "ztt", "fs"   /* 4 = the deleted cascade's slot */
};
typedef char vw2__il_name_table_is_complete[
    (sizeof vw2_oop_il_name / sizeof vw2_oop_il_name[0]) == VW2_OOP_IL_ROUTE_MAX + 1 ? 1 : -1];

/* --------------------------------------------------------- kind-3 (k1) */
/* the INTERLEAVED axis of the kind-3 (K=1) cell: the lay=il row first, the
 * lay-less vintage row second, tiered by decode success. THREE outcomes per
 * source: a decoded route (> NONE) commits; a vintage row WITHOUT an il_route
 * token is the raced-none verdict (IL_NONE); an undecodable token falls to
 * the next tier. Absent everywhere leaves k1_il_route = -1 = unraced. The
 * pre-split reader's IL loop, verbatim; K and ns are this axis's own (the
 * defaults K=1, ns=0 when no IL route decoded).
 *
 * Returns 1 when the IL axis decided - and, COMPAT (F3), when any kind-3 row
 * exists at the cell (vw2_oop_k1_row_present): the pre-split lookup reported
 * a cell present when EITHER layout's axis decoded, and the IL consumers
 * test the entry pointer (the writer-band return, the T-row copy), so a
 * split-only cell must keep reading as present with its IL axis unraced. */
static inline int vw2_oop_lookup_k1_il_cell(const vw2_store_t *s, int N, int want_scr,
                                            int inplace, int T, vfft_oop_il_entry_t *e)
{
    const int pl = inplace ? VW2_PL_IP : VW2_PL_OOP;
    const vw2_rec_t *ril = vw2__oop_k1_scan_pl(s, N, VW2_LAY_IL, want_scr, pl, T);
    const vw2_rec_t *rlg = vw2__oop_k1_scan_pl(s, N, VW2_LAY_ANY, want_scr, pl, T);
    int pair[2], np, got = 0, si;
    memset(e, 0, sizeof *e);
    e->place_ip = inplace;
    e->nthreads = T > 1 ? T : 0;
    e->ord_scr = want_scr;
    e->kind = VFFT_OOP_KIND_BAILEY2V;
    e->N = N;
    e->K = 1;
    e->k1_il_route = -1;                       /* UNRACED — distinct from
                                                * IL_NONE = "raced: none
                                                * available"                */
    /* IL axis, same tiering. THREE outcomes per source: a decoded route
     * (> NONE) commits; a legacy row WITHOUT an il_route token is the
     * raced-none verdict (IL_NONE — that is what its writer meant); an
     * undecodable token falls to the next tier. Absent from every source
     * leaves -1 = unraced, so the consumer can run its IL heuristic
     * instead of mistaking absence for a NONE verdict. */
    for (si = 0; si < 2 && e->k1_il_route < 0; si++) {
        const vw2_rec_t *ri = (si == 0) ? ril : rlg;
        const char *il;
        int ilr;
        if (!ri) continue;
        il = vw2_rec_get(ri, "il_route");
        if (!il) {
            if (si == 1) { e->k1_il_route = VFFT_K1_IL_NONE; got = 1; }
            continue;                          /* il CELLS always carry one */
        }
        ilr = vw2__oop_name_idx(vw2_oop_il_name, VW2_OOP_IL_ROUTE_MAX + 1, il);
        if (ilr < 0) continue;                 /* undecodable: next tier    */
        e->k1_il_route = ilr;
        if (ilr > VFFT_K1_IL_NONE) {
            np = vw2__oop_split_ints(vw2_rec_get(ri, "il_pair"), pair, 2);
            if (np == 2) { e->il_R1 = pair[0]; e->il_R2 = pair[1]; }
            e->il_kv = vw2__oop_geti(ri, "il_kv", 0);
            e->il_kv_raced = vw2_rec_get(ri, "il_kv") != NULL;   /* explicit 0 counts */
            {                                  /* chain3 payload */
                int c3[3];
                if (vw2__oop_split_ints(vw2_rec_get(ri, "il_chain"), c3, 3) == 3) {
                    e->il_c3[0] = c3[0]; e->il_c3[1] = c3[1]; e->il_c3[2] = c3[2];
                }
            }
            {                                  /* flat DIT payload */
                int fl[10];
                const int nfl = vw2__oop_split_ints(vw2_rec_get(ri, "il_flat"), fl, 10);
                const char *ff = vw2_rec_get(ri, "il_forms");
                if (nfl >= 2) { memcpy(e->il_fl, fl, sizeof(int) * (size_t)nfl); e->il_fl_n = nfl; }
                if (ff) { strncpy(e->il_flf, ff, sizeof e->il_flf - 1); e->il_flf[sizeof e->il_flf - 1] = 0; }
                e->il_tw = vw2__oop_geti(ri, "il_tw", 0);
            }
            {                                  /* ZTURN-T payload */
                int zt[7];
                const int nzt = vw2__oop_split_ints(vw2_rec_get(ri, "il_ztt"), zt, 7);
                if (nzt >= 2) { memcpy(e->il_zt, zt, sizeof(int) * (size_t)nzt); e->il_zt_n = nzt;
                                e->il_tw = vw2__oop_geti(ri, "il_tw", 0); }   /* the raced tile */
            }
            /* K/ns keep the pre-1.2 dual-line convention: when an IL
             * verdict is present they are the IL natural champion's
             * numbers (ran=1). */
            e->K = (size_t)vw2__oop_geti(ri, "ran", 1);
            { const char *ns = vw2_rec_get(ri, "ns"); e->ns = ns ? atof(ns) : 0.0; }
        }
        got = 1;
    }
    if (!got && vw2_oop_k1_row_present(s, N, want_scr, pl, T))
        got = 1;                               /* COMPAT (F3): present, IL unraced */
    return got;
}

/* THE FOUR-STEP CHILD's codec (owner, 2026-10-03; the c2c row 2026-10-04).
 * A four-step's child -- the 2D plan at N1 x N2 and its row plan at N2, raced
 * into a private store (il/rank1/k1_fourstep.h) -- rides on the row that
 * banks the four-step (the c2c K=1 row, the real row) in its own row's
 * words: every payload token of the 2D row under fs_. The 2D row carries its
 * row plan itself (rp_*, il/wisdom/wisdom2_child.h, 2026-10-05); rows banked
 * before that carry the row plan under fs_row_ and its backward twin under
 * fs_row_bwd_, read as optional. Replay rebuilds the rows from them (their
 * keys follow from the split, the placement and the thread count). */
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

/* THE K=1 ROW AT T > 1 (v1.3). The K=1 route is thread-independent (serial
 * kernels; threading is a flag on the route), so the route race banks the
 * one-thread row and a plan at T gets its own row as a COPY of that route,
 * keyed nthreads=T, on which the MT commits then race and bank the threaded
 * verdict (il_mt, il_mt_tw, il_mtsb). Complete on its own, never a re-race
 * of the route set per T. Replaces an existing T row (recalibrate). 1 = the
 * row exists now, 0 = no one-thread row to copy or the bank refused. */
static inline int vw2_oop_k1_row_at_T(vw2_store_t *s, int N, int want_scr, int inplace, int T)
{
    static const char *const THREADED[] = { "il_mt", "il_mt_tw", "il_mtsb", NULL };
    /* and the four-step's child (fs_*): the one-thread row's is the child at
     * one thread; the threaded split race banks the child at T beside il_mt
     * (il/rank1/k1_fourstep.h) */
    const vw2_rec_t *r;
    vw2_rec_t nr;
    int i, k;
    if (T < 2) return 0;
    r = vw2__oop_k1_scan_pl(s, N, VW2_LAY_IL, want_scr, inplace ? VW2_PL_IP : VW2_PL_OOP, 1);
    if (!r) return 0;
    memset(&nr, 0, sizeof nr);
    nr.key = r->key;
    nr.key.nthreads = (uint8_t)T;
    for (i = 0; i < r->ntok; i++) {
        int skip = 0;
        for (k = 0; THREADED[k]; k++) if (!strcmp(r->tok[i].name, THREADED[k])) skip = 1;
        if (r->tok[i].sect == 1 && !strncmp(r->tok[i].name, "fs_", 3)) skip = 1;
        if (skip) continue;
        if (vw2_rec_set(&nr, r->tok[i].sect, r->tok[i].name, r->tok[i].val) != VW2_OK) { vw2_rec_free(&nr); return 0; }
    }
    if (vw2_bank(s, &nr) != VW2_OK) { vw2_rec_free(&nr); return 0; }
    return 1;
}

/* kind-3 BACKWARD sibling (dir=bwd).
 *
 * The backward kernel-variant verdict is its OWN CELL rather than more il_kv
 * bits, because wisdom2 keys direction (`dir=`) and does not key kernel
 * forms. Same cell as the forward verdict in every other component, so the
 * two travel together and a reader that does not know about the backward
 * axis simply never asks for it.
 *
 * Returns the packed variant code (0 = no verdict / nothing to apply) and,
 * when non-zero, writes the pair the verdict was MEASURED ON into *R1/*R2.
 * The caller must check that pair against the plan it actually built: a
 * variant code is only meaningful for the radix pair it was raced at, and
 * the forward winner can move without this record being re-raced. */
static inline const vw2_rec_t *vw2__oop_find_k1_bwd_pl(const vw2_store_t *s, int N, int pl)
{
    int i, tier;
    const vw2_rec_t *r = NULL;
    /* v1.2: this reader owns the IL backward verdict. TWO TIERS, lay=il
     * before lay-less vintage — the same cell-before-legacy precedence the
     * forward reader implements: eq compares lay, so a re-raced verdict
     * banks as a NEW lay=il record, which a single first-match scan would
     * let a pre-1.2 row (loaded earlier) shadow forever — banked, persisted,
     * never served. A lay=split backward cell belongs to a split-side
     * reader, not here. (The finder is shared with the chain3 reader.) */
    for (tier = 0; tier < 2 && !r; tier++)
        for (i = 0; i < s->nrec; i++) {
            const vw2_rec_t *c = &s->rec[i];
            if (c->key.t != VW2_T_C2C || c->key.rank != 1 || c->key.n[0] != N) continue;
            if (c->key.dir != VW2_DIR_BWD) continue;
            if (c->key.pl != pl) continue;   /* the placement's own backward row */
            if (c->key.role != VW2_ROLE_COMP) continue;
            if (c->key.lay != (tier == 0 ? VW2_LAY_IL : VW2_LAY_ANY)) continue;
            if (strcmp(vw2__oop_eng(c), "k1")) continue;
            if (vw2__is_seed(c)) continue;
            r = c;
            break;
        }
    return r;
}
static inline const vw2_rec_t *vw2__oop_find_k1_bwd(const vw2_store_t *s, int N)
{
    return vw2__oop_find_k1_bwd_pl(s, N, VW2_PL_OOP);
}
/* the CHAIN3 backward verdict: the same cell, read through the chain it was
 * raced at (il_chain=R2.A.B); -1 when the row is absent, has no verdict, or
 * is a pair row. The three-nibble code is A | B<<4 | leaf<<8. */
static inline int vw2_oop_lookup_k1_bwd_chain_pl(const vw2_store_t *s, int N,
                                                 int *c3 /* [3] */, int pl)
{
    const vw2_rec_t *r = vw2__oop_find_k1_bwd_pl(s, N, pl);
    int ch[3];
    if (!r) return -1;
    if (!vw2_rec_get(r, "il_kv")) return -1;
    if (vw2__oop_split_ints(vw2_rec_get(r, "il_chain"), ch, 3) != 3) return -1;
    if (c3) { c3[0] = ch[0]; c3[1] = ch[1]; c3[2] = ch[2]; }
    return vw2__oop_geti(r, "il_kv", 0);
}
static inline int vw2_oop_lookup_k1_bwd_chain(const vw2_store_t *s, int N,
                                              int *c3 /* [3] */)
{
    return vw2_oop_lookup_k1_bwd_chain_pl(s, N, c3, VW2_PL_OOP);
}
static inline int vw2_oop_lookup_k1_bwd_pl(const vw2_store_t *s, int N,
                                           int *R1, int *R2, int pl)
{
    int pair[2], np, kv;
    const vw2_rec_t *r = vw2__oop_find_k1_bwd_pl(s, N, pl);
    /* returns the banked backward form code (>= 0; 0 = "the defaults won",
     * a real verdict) or -1 when no usable row exists */
    if (!r) return -1;
    if (!vw2_rec_get(r, "il_kv")) return -1;   /* vintage row without a verdict */
    kv = vw2__oop_geti(r, "il_kv", 0);
    np = vw2__oop_split_ints(vw2_rec_get(r, "il_pair"), pair, 2);
    if (np != 2) return -1;         /* unusable without the pair it was raced at */
    if (R1) *R1 = pair[0];
    if (R2) *R2 = pair[1];
    return kv;
}
static inline int vw2_oop_lookup_k1_bwd(const vw2_store_t *s, int N,
                                        int *R1, int *R2)
{
    return vw2_oop_lookup_k1_bwd_pl(s, N, R1, R2, VW2_PL_OOP);
}

/* ================================================================ WRITE */
/* Build the kind-3 BACKWARD record (dir=bwd). Wisdom2-NATIVE, like the
 * kind-5 builder below and unlike vw2_oop_rec_from_entry: there is no legacy
 * text line that can carry a backward variant verdict, so routing it through
 * the legacy entry struct would only widen a frozen format.
 *
 * Payload is deliberately minimal - the plan identity (il_route, il_pair)
 * plus the verdict (il_kv). Everything else about the cell is stated by the
 * forward record it shares a key with. */
static inline int vw2_oop_rec_k1_bwd(vw2_rec_t *r, int N, int il_route,
                                     int R1, int R2, int kv, double ns,
                                     const char *src, const char **why)
{
    char pair[48], kvb[16], nsbuf[48];
    *why = NULL;
    memset(r, 0, sizeof *r);
    if (kv < 0)                       { *why = "no-bwd-verdict";        return -1; }
    /* kv == 0 is a VERDICT ("the default forms won") and banks as an
     * explicit il_kv=0; only a negative kv means unraced */
    /* NOT VW2_OOP_IL_ROUTE_MAX, and not stale: the dir=bwd sibling row
     * exists only for the routes that HAVE a backward form axis (the pair,
     * the chain, prime). The flat DIT, ZTURN-T and the four-step never bank
     * one, and admitting them here would create a row nothing reads. */
    if (il_route < 0 || il_route > VFFT_K1_IL_PRIME) { *why = "il-route-out-of-range"; return -1; }
    if (R1 <= 0 || R2 <= 0)           { *why = "bwd-pair-missing";      return -1; }

    r->key.t = VW2_T_C2C; r->key.rank = 1; r->key.n[0] = N;
    r->key.q = 1; r->key.ord = VW2_ORD_NAT; r->key.pl = VW2_PL_OOP;
    r->key.role = VW2_ROLE_COMP;
    r->key.dir  = VW2_DIR_BWD;       /* separates this record from the
                                      * forward verdict */
    r->key.lay  = VW2_LAY_IL;        /* v1.2: the backward variant verdict is
                                      * INTERLEAVED-only payload (il_route/
                                      * il_kv), so it carries the IL tag —
                                      * a split backward verdict, if one ever
                                      * exists, gets its own lay=split cell
                                      * instead of colliding here. */
    snprintf(pair, sizeof pair, "%d.%d", R1, R2);
    snprintf(kvb,  sizeof kvb,  "%d", kv);
    snprintf(nsbuf, sizeof nsbuf, "%.1f", ns);
    if (vw2_rec_set(r, 1, "eng", "k1") != VW2_OK ||
        vw2_rec_set(r, 1, "il_route", vw2_oop_il_name[il_route]) != VW2_OK ||
        vw2_rec_set(r, 1, "il_pair", pair) != VW2_OK ||
        vw2_rec_set(r, 1, "il_kv", kvb) != VW2_OK ||
        vw2_rec_set(r, 2, "ran", "1") != VW2_OK ||
        (ns > 0.0 &&
         (vw2_rec_set(r, 2, "ns", nsbuf) != VW2_OK ||
          /* a THIRD metric identity: neither fwd1 nor joint2. The compare
           * helper refuses across metrics, which is exactly right here - a
           * backward-only number must never be ranked against a forward one. */
          vw2_rec_set(r, 2, "metric", "bwd1") != VW2_OK ||
          vw2_rec_set(r, 2, "units", "ns") != VW2_OK)) ||
        vw2_rec_set(r, 2, "src", src) != VW2_OK) {
        vw2_rec_free(r);
        *why = "token-refused";
        return -1;
    }
    return VW2_OK;
}

/* Build ONE per-layout kind-3 record. Wisdom2-NATIVE, like the bwd builder:
 * fresh banks only — no legacy text line has per-layout shape. lay names
 * the CALLER LAYOUT this verdict serves (layout is an integration
 * property, AoS/SoA, never a strategy output). Each record carries ONLY
 * its own layout's fields, so neither layout's re-race can erase the
 * other's verdict, and neither layout's planning failure can veto the
 * other's bank — the two collision classes of the pre-1.2 dual line. */
static inline int vw2_oop_rec_k1_il(vw2_rec_t *r,
                                    const vfft_oop_il_entry_t *e,
                                    const char **why)
{
    char nsbuf[48], pair[48];
    int i;
    *why = NULL;
    memset(r, 0, sizeof *r);
    snprintf(nsbuf, sizeof nsbuf, "%.1f", e->ns);
    r->key.t = VW2_T_C2C; r->key.rank = 1; r->key.n[0] = e->N;
    r->key.q = 1;
    r->key.pl = e->place_ip ? VW2_PL_IP : VW2_PL_OOP;   /* the in-place cell's own row */
    r->key.ord = e->ord_scr ? VW2_ORD_SCR : VW2_ORD_NAT;   /* the scrambled class's own cell */
    r->key.role = VW2_ROLE_COMP;
    r->key.lay  = VW2_LAY_IL;
    r->key.nthreads = (uint8_t)(e->nthreads > 1 ? e->nthreads : 0);   /* the threaded plan's own row (v1.3) */
#define VW2__OB_SET(sect, n, v) do { \
    if (vw2_rec_set(r, sect, n, v) != VW2_OK) { vw2_rec_free(r); *why = "token-refused"; return -1; } \
} while (0)
    VW2__OB_SET(1, "eng", "k1");
    if (e->k1_il_route <= VFFT_K1_IL_NONE || e->k1_il_route > VW2_OOP_IL_ROUTE_MAX) {
        vw2_rec_free(r); *why = "no-il-verdict"; return -1;
    }
    VW2__OB_SET(1, "il_route", vw2_oop_il_name[e->k1_il_route]);
    if (e->il_R1 || e->il_R2) {
        snprintf(pair, sizeof pair, "%d.%d", e->il_R1, e->il_R2);
        VW2__OB_SET(1, "il_pair", pair);
    }
    if (e->k1_il_route == VFFT_K1_IL_CHAIN3 && e->il_c3[0]) {
        char c3b[48];              /* the chain IS the verdict */
        snprintf(c3b, sizeof c3b, "%d.%d.%d", e->il_c3[0], e->il_c3[1], e->il_c3[2]);
        VW2__OB_SET(1, "il_chain", c3b);
    }
    if (e->k1_il_route == VFFT_K1_IL_FLAT && e->il_fl_n >= 2) {
        char flb[64];              /* the flat chain + its per-stage forms ARE the verdict */
        size_t off = 0;
        for (i = 0; i < e->il_fl_n; i++) {
            int rr = snprintf(flb + off, sizeof flb - off, "%s%d", i ? "." : "", e->il_fl[i]);
            if (rr < 0 || (size_t)rr >= sizeof flb - off) break;
            off += (size_t)rr;
        }
        VW2__OB_SET(1, "il_flat", flb);
        if (e->il_flf[0]) VW2__OB_SET(1, "il_forms", e->il_flf);
        if (e->il_tw > 0) { char twb[16]; snprintf(twb, sizeof twb, "%d", e->il_tw); VW2__OB_SET(1, "il_tw", twb); }
    }
    if (e->k1_il_route == VFFT_K1_IL_ZTT && e->il_zt_n >= 2) {
        char ztb[48];              /* ZTURN-T: the chain IS the verdict */
        size_t off = 0;
        for (i = 0; i < e->il_zt_n; i++) {
            int rr = snprintf(ztb + off, sizeof ztb - off, "%s%d", i ? "." : "", e->il_zt[i]);
            if (rr < 0 || (size_t)rr >= sizeof ztb - off) break;
            off += (size_t)rr;
        }
        VW2__OB_SET(1, "il_ztt", ztb);
        if (e->il_tw > 0) { char twb[16]; snprintf(twb, sizeof twb, "%d", e->il_tw); VW2__OB_SET(1, "il_tw", twb); }   /* the raced tile */
    }
    if (e->il_kv || e->il_kv_raced) {   /* a raced verdict emits il_kv even when 0 */
        char kvb[16];
        snprintf(kvb, sizeof kvb, "%d", e->il_kv);
        VW2__OB_SET(1, "il_kv", kvb);
    }
    {
        char ranb[24];
        snprintf(ranb, sizeof ranb, "%lld", (long long)e->K);
        VW2__OB_SET(2, "ran", ranb);
    }
    if (e->ns > 0.0) {
        VW2__OB_SET(2, "ns", nsbuf);
        VW2__OB_SET(2, "metric", "fwd1");
        VW2__OB_SET(2, "units", "ns");
    }
    VW2__OB_SET(2, "src", "race");
#undef VW2__OB_SET
    return VW2_OK;
}

static inline int vw2_oop_bank_k1_il(vw2_store_t *s,
                                     const vfft_oop_il_entry_t *e)
{
    vw2_rec_t r;
    const char *why = NULL;
    int rc;
    if (vw2_oop_rec_k1_il(&r, e, &why) != VW2_OK) {
        fprintf(stderr, "[wisdom2] oop k1 %s bank refused (%s)\n",
                "lay=il", why ? why : "?");
        return -1;
    }
    if (e->k1_il_route == VFFT_K1_IL_PRIME) {
        /* THE PRIME CELL'S ROW IS ITS METHOD VERDICT: the prime engine
         * banks eng=rader|bluestein + the raced inner at this very key
         * (vw2__prime_method_key: ord=scr place=ip role=comp lay=il), and
         * vw2_bank replaces on an equal key. A route row (eng=k1
         * il_route=prime) banked over it would leave the next create without
         * a method (the lookup reads eng): the inner re-races and the method
         * row is re-banked over the route row, so every other create races
         * -- and under load picks a different inner (il2d_blu_row_gate's
         * 53x64 cold-vs-warm bitwise flap). The method row stays and the
         * route row is not written. */
        const vw2_rec_t *old = vw2_lookup(s, &r.key);
        const char *eng = old ? vw2_rec_get(old, "eng") : NULL;
        if (eng && (!strcmp(eng, "rader") || !strcmp(eng, "bluestein"))) {
            vw2_rec_free(&r);
            return VW2_OK;
        }
    }
    vw2__oop_stamp_date(&r);
    rc = vw2_bank(s, &r);
    if (rc != VW2_OK) { vw2_rec_free(&r); return rc; }
    return VW2_OK;
}

/* ── the IL prime METHOD verdict ─────────────────────────────────────────
 * Rader vs Bluestein for a prime N, raced once per cell instead of on every
 * create. A COMPONENT row (the ilprime plan is a component of both the
 * in-place ilp mode and the OOP k1 PRIME route), homed in the PRIME shard
 * by its eng= token: t=c2c n=N q=1 ord=scr place=ip role=comp lay=il |
 * eng=rader|bluestein. 0 = no verdict, 1 = Rader, 2 = Bluestein. */
static inline void vw2__prime_method_key(int N, vw2_key_t *k)
{
    memset(k, 0, sizeof *k);
    k->t = VW2_T_C2C; k->rank = 1; k->n[0] = N;
    k->q = 1; k->ord = VW2_ORD_SCR; k->pl = VW2_PL_IP;
    k->role = VW2_ROLE_COMP; k->lay = VW2_LAY_IL;
}
static inline int vw2_prime_method_lookup(const vw2_store_t *s, int N)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    const char *eng;
    vw2__prime_method_key(N, &k);
    r = vw2_lookup(s, &k);
    if (!r) return 0;
    eng = vw2_rec_get(r, "eng");
    if (!eng) return 0;
    if (!strcmp(eng, "rader")) return 1;
    if (!strcmp(eng, "bluestein")) return 2;
    return 0;
}
/* THE PRIME CELL'S OWN VERDICT: the METHOD and the INNER, raced together
 * on the whole convolution and banked on this row. in= names the inner's
 * kind (2p | 3p | ztt), in_sh= its shape (R1.R2 | R2.A.B | the chain),
 * in_tw= ZTURN-T's tile (0 = none). */
static inline int vw2_prime_method_bank(vw2_store_t *s, int N, int method,
                                        const char *in_kind, const char *in_shape,
                                        int in_tw)
{
    vw2_rec_t r;
    int rc;
    char b[24];
    memset(&r, 0, sizeof r);
    vw2__prime_method_key(N, &r.key);
    if (vw2_rec_set(&r, 1, "eng", method == 1 ? "rader" : "bluestein") != VW2_OK ||
        vw2_rec_set(&r, 2, "ran", "1") != VW2_OK ||
        vw2_rec_set(&r, 2, "src", "race") != VW2_OK) {
        vw2_rec_free(&r);
        return -1;
    }
    if (in_kind && in_shape) {
        snprintf(b, sizeof b, "%d", in_tw);
        if (vw2_rec_set(&r, 1, "in", in_kind) != VW2_OK ||
            vw2_rec_set(&r, 1, "in_sh", in_shape) != VW2_OK ||
            vw2_rec_set(&r, 1, "in_tw", b) != VW2_OK) {
            vw2_rec_free(&r);
            return -1;
        }
    }
    vw2__oop_stamp_date(&r);
    rc = vw2_bank(s, &r);
    if (rc != VW2_OK) vw2_rec_free(&r);
    return rc;
}
/* the inner's verdict on the prime row: 1 with the tokens filled, 0 when
 * the row has none (a row without the tokens, or a miss) */
static inline int vw2_prime_inner_lookup(const vw2_store_t *s, int N,
                                         char *kind, size_t ksz,
                                         char *shape, size_t ssz, int *tw)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    const char *v;
    vw2__prime_method_key(N, &k);
    r = vw2_lookup(s, &k);
    if (!r) return 0;
    v = vw2_rec_get(r, "in");
    if (!v) return 0;
    snprintf(kind, ksz, "%s", v);
    v = vw2_rec_get(r, "in_sh");
    if (!v) return 0;
    snprintf(shape, ssz, "%s", v);
    v = vw2_rec_get(r, "in_tw");
    *tw = v ? atoi(v) : 0;
    return 1;
}

/* ------------------------------------------------------- kind-5 (zr2c) */
/* Build the up-to-4 per-slot records of a kind-5 packed verdict. Fills
 * out[] (caller-provided, size 4), returns the count (0 = all unmeasured). */
static inline int vw2_oop_recs_from_kind5(int N, int zr_kv,
                                          const char *src, const char *from,
                                          vw2_rec_t out[4], const char **why)
{
    int slot, n = 0;
    *why = NULL;
    for (slot = 0; slot < 4; slot++) {
        int v = vfft_zr2c_kv_get(zr_kv, slot);
        vw2_rec_t *s;
        if (!v) continue;
        s = &out[n];
        memset(s, 0, sizeof *s);
        s->key.t = (slot >> 1) ? VW2_T_C2R : VW2_T_R2C;
        s->key.rank = 1; s->key.n[0] = N;
        s->key.q = 1; s->key.ord = VW2_ORD_NAT;
        s->key.pl = (slot & 1) ? VW2_PL_IP : VW2_PL_OOP;
        if (vw2_rec_set(s, 1, "eng", "zr2c") != VW2_OK ||
            vw2_rec_set(s, 1, "route", v == 1 ? "child_oop_il" : "child_nat_ip") != VW2_OK ||
            vw2_rec_set(s, 2, "ran", "1") != VW2_OK ||
            vw2_rec_set(s, 2, "src", src) != VW2_OK ||
            (from && vw2_rec_set(s, 2, "from", from) != VW2_OK)) {
            int t;
            for (t = 0; t <= n; t++) vw2_rec_free(&out[t]);
            *why = "token-refused";
            return -1;
        }
        n++;
    }
    if (!n) { *why = "zr-kv-all-unmeasured"; return -1; }
    return n;
}

/* 1 when this problem cell is already owned by a DIFFERENT engine (the
 * split family's eng=route). The reciprocal of vw2_real_cell_taken.
 *
 * The route race is a LANE-BATCH race and the split engine's executed
 * batch is never 1, so q=1 real cells belong to the interleaved zr2c
 * verdicts ALONE: vw2_real_route_bank refuses K <= 1 loudly, so the shipped
 * writers never collide at this key. The guard is a backstop against a
 * hand-written or foreign-vintage row — refusing loudly is still right —
 * not the ownership mechanism. */
static inline int vw2_oop_zr2c_cell_taken(const vw2_store_t *s, int realN,
                                          int is_c2r, int is_inplace)
{
    vw2_key_t req;
    int i;
    memset(&req, 0, sizeof req);
    req.t = is_c2r ? VW2_T_C2R : VW2_T_R2C;
    req.rank = 1; req.n[0] = realN;
    req.q = 1; req.ord = VW2_ORD_NAT;
    req.pl = is_inplace ? VW2_PL_IP : VW2_PL_OOP;
    for (i = 0; i < s->nrec; i++) {
        const vw2_rec_t *c = &s->rec[i];
        if (!vw2_key_serves(&c->key, &req)) continue;
        if (strcmp(vw2__oop_eng(c), "zr2c")) return 1;
    }
    return 0;
}

/* zr2c slot bank: banks ONE (transform, placement) slot verdict directly —
 * per-slot records need no read-modify-write of a packed verdict, the other
 * slots' records are untouched by construction. ns = the slot's own race
 * median (attributable per slot); <= 0 omits the measurement. */
static inline int vw2_oop_bank_zr2c_slot(vw2_store_t *s, int realN,
                                         int is_c2r, int is_inplace, int route,
                                         double ns)
{
    vw2_rec_t r;
    char b[48];
    int rc;
    if (vw2_oop_zr2c_cell_taken(s, realN, is_c2r, is_inplace)) {
        fprintf(stderr, "[wisdom2] zr2c bank refused: t=%s n=%d place=%s is owned "
                        "by another engine\n",
                is_c2r ? "c2r" : "r2c", realN, is_inplace ? "ip" : "oop");
        /* NOT 0: VW2_OK == 0, so returning 0 here would make "declined"
         * indistinguishable from "banked" to every caller and to the gate. */
        return VW2_EOWNED;
    }
    memset(&r, 0, sizeof r);
    r.key.t = is_c2r ? VW2_T_C2R : VW2_T_R2C;
    r.key.rank = 1; r.key.n[0] = realN;
    r.key.q = 1; r.key.ord = VW2_ORD_NAT;
    r.key.pl = is_inplace ? VW2_PL_IP : VW2_PL_OOP;
    if (vw2_rec_set(&r, 1, "eng", "zr2c") != VW2_OK ||
        vw2_rec_set(&r, 1, "route", route ? "child_nat_ip" : "child_oop_il") != VW2_OK ||
        vw2_rec_set(&r, 2, "ran", "1") != VW2_OK ||
        vw2_rec_set(&r, 2, "src", "race") != VW2_OK) { vw2_rec_free(&r); return -1; }
    if (ns > 0.0) {
        snprintf(b, sizeof b, "%.1f", ns);
        if (vw2_rec_set(&r, 2, "ns", b) != VW2_OK ||
            /* The c2r race times a BACKWARD composite (fold_bwd then the
             * child run backward), so a c2r slot must not claim fwd1: that
             * would assert comparability with forward numbers against the
             * store's metric law, and against a correctly-stamped bwd1
             * incumbent the merge refuses with VW2_EMETRIC -- invisibly,
             * the caller discards the return -- so the cell would re-race
             * on every create. (No shipped zr2c row carries a metric=
             * token.) */
            vw2_rec_set(&r, 2, "metric", is_c2r ? "bwd1" : "fwd1") != VW2_OK ||
            vw2_rec_set(&r, 2, "units", "ns") != VW2_OK) { vw2_rec_free(&r); return -1; }
    }
    vw2__oop_stamp_date(&r);
    rc = vw2_bank(s, &r);
    if (rc != VW2_OK) { vw2_rec_free(&r); return rc; }
    return VW2_OK;
}

/* Reassembles the packed 4-slot verdict from the per-slot real records via
 * the SHIPPED kv codec. Returns 1 when any slot is measured. */
static inline int vw2_oop_lookup_zr2c(const vw2_store_t *s, int realN, int *zr_kv)
{
    int slot, any = 0, kv = 0;
    for (slot = 0; slot < 4; slot++) {
        vw2_key_t k;
        const vw2_rec_t *r;
        memset(&k, 0, sizeof k);
        k.t = (slot >> 1) ? VW2_T_C2R : VW2_T_R2C;
        k.rank = 1; k.n[0] = realN;
        k.q = 1; k.ord = VW2_ORD_NAT;
        k.pl = (slot & 1) ? VW2_PL_IP : VW2_PL_OOP;
        r = vw2_lookup(s, &k);
        if (!r) continue;
        if (strcmp(vw2__oop_eng(r), "zr2c")) {
            /* Foreign engine owns this cell. Fall through to the route race
             * rather than pretend there is no verdict -- but SAY so, because
             * a silent skip here and a silent clobber on the write side is
             * how two engines quietly fight over one key. */
            fprintf(stderr, "[wisdom2] zr2c: cell t=%s n=%d place=%s owned by eng=%s "
                            "-- ignoring for the zr2c route pick\n",
                    (slot >> 1) ? "c2r" : "r2c", realN,
                    (slot & 1) ? "ip" : "oop", vw2__oop_eng(r));
            continue;
        }
        /* SEED SKIP — the law vw2_oop_lookup_k1 applies too. It matters
         * more for zr2c than for k1: _zr2c_build
         * RETURNS on any banked verdict, so a bank-only row makes the racer
         * at step 3 permanently unreachable at that cell. A seed is a row
         * nothing measured; it must not preempt the measurement. */
        if (vw2__is_seed(r)) continue;
        {
            const char *route = vw2_rec_get(r, "route");
            if (!route) continue;
            if (!strcmp(route, "child_oop_il"))      kv = vfft_zr2c_kv_set(kv, slot, 0);
            else if (!strcmp(route, "child_nat_ip")) kv = vfft_zr2c_kv_set(kv, slot, 1);
            else continue;
            any = 1;
        }
    }
    if (any) *zr_kv = kv;
    return any;
}

#endif /* VFFT_WISDOM2_OOP_IL_H */
