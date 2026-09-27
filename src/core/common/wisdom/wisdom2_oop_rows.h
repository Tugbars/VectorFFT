/* wisdom2_oop_rows.h — the OOP family's row MECHANICS, shared by both layouts'
 * codecs: token parsing, the engine tag, the date stamp, the kind-3 (K=1)
 * row scan by layout, and the existence queries.
 *
 * Carved out of wisdom2/wisdom2_oop_reader.h in the kind-3 record split
 * (layout separation phase 5, owner's option (a), 2026-09-27). Nothing here
 * knows a layout's verdict: the split codec is split/wisdom/wisdom2_oop_split.h,
 * the interleaved one il/wisdom/wisdom2_oop_il.h. */
#ifndef VFFT_WISDOM2_OOP_ROWS_H
#define VFFT_WISDOM2_OOP_ROWS_H

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "wisdom2.h"

static inline int vw2__oop_name_idx(const char **tab, int n, const char *v)
{
    int i;
    if (!v) return -1;
    for (i = 0; i < n; i++)
        if (!strcmp(tab[i], v)) return i;
    return -1;
}

/* "a.b.c" -> ints; returns count or 0 on any malformed piece */
static inline int vw2__oop_split_ints(const char *s, int *out, int cap)
{
    int n = 0;
    if (!s) return 0;
    while (*s) {
        char *end;
        long v = strtol(s, &end, 10);
        if (end == s || v <= 0 || n >= cap) return 0;
        out[n++] = (int)v;
        if (*end == '\0') break;
        if (*end != '.') return 0;
        s = end + 1;
    }
    return n;
}

static inline int vw2__oop_geti(const vw2_rec_t *r, const char *name, int dflt)
{
    const char *v = vw2_rec_get(r, name);
    return v ? atoi(v) : dflt;
}

/* record -> is this the k1-engine family / cascade family / classic family?
 * The engine field IS the verdict; reading it to recognize the record's
 * family is reading the verdict, not smuggling semantics into sharding. */
static inline const char *vw2__oop_eng(const vw2_rec_t *r)
{
    const char *e = vw2_rec_get(r, "eng");
    return e ? e : "";
}

static inline void vw2__oop_stamp_date(vw2_rec_t *r)
{
    /* create-time banking timestamp (never on an execute path) */
    time_t t = time(NULL);
    struct tm *tmv = localtime(&t);
    char d[16];
    if (tmv && strftime(d, sizeof d, "%Y-%m-%d", tmv) > 0)
        vw2_rec_set(r, 2, "date", d);
}

/* kind-3 K=1 lookup — PER-LAYOUT (v1.2). The verdict may
 * live in three record shapes:
 *   lay=il     the interleaved caller's own cell (il_route/il_pair/il_kv)
 *   lay=split  the split caller's own cell       (sp_route/sp_pair/chain)
 *   lay-less   the pre-1.2 dual row — the fallback tier
 * Each axis composes INDEPENDENTLY: per-layout cell first, legacy row
 * second, absent third. sp-absent is k1_sp_route = -1 — an EXPLICIT
 * sentinel, because 0 is a VALID route (VFFT_K1_SP_3P); il-absent is
 * IL_NONE. A decode failure on one axis refuses ONLY that axis (a whole-row
 * refusal would let one layout's unknown token silently erase the other
 * layout's banked verdict). Exact-beats-wildcard is preserved inside
 * each tier. Returns 1 + fills e when ANY axis was found. */
static inline const vw2_rec_t *vw2__oop_k1_scan_pl(const vw2_store_t *s, int N,
                                               uint8_t lay, int want_scr, int pl, int T)
{
    int i, pass;
    for (pass = 0; pass < 2; pass++)
        for (i = 0; i < s->nrec; i++) {
            const vw2_rec_t *c = &s->rec[i];
            if (c->key.t != VW2_T_C2C || c->key.rank != 1 || c->key.n[0] != N) continue;
            if (c->key.lay != lay) continue;
            /* the THREAD-COUNT axis (v1.3): a threaded plan's row is its own */
            if (VW2__NT(&c->key) != (T > 1 ? T : 1)) continue;
            /* the PLACEMENT axis: the in-place cell has its own
             * kind-3 row keyed place=ip, raced executed in place; neither
             * placement ever reads the other's row */
            if (c->key.pl != pl) continue;
            /* the ORDER axis: the flat DIT's scrambled class
             * banks its own kind-3 IL row keyed ord=scr; the natural lookup
             * must never read it and the scrambled lookup reads only it */
            if (want_scr ? (c->key.ord != VW2_ORD_SCR) : (c->key.ord == VW2_ORD_SCR)) continue;
            /* pin the axes this scan relies on (the vw2_key_serves law:
             * a hand scan silently mis-scopes on axes it does not name) —
             * every kind-3 writer stamps role=comp, fresh and migrated */
            if (c->key.role != VW2_ROLE_COMP) continue;
            if (strcmp(vw2__oop_eng(c), "k1")) continue;
            /* The kind-3 family has a dir=bwd SIBLING (the backward
             * kernel-variant verdict). It shares eng=k1 and the cell key;
             * anything directional belongs to its own reader
             * (vw2_oop_lookup_k1_bwd). */
            if (c->key.dir != VW2_DIR_NONE) continue;
            if (vw2__is_seed(c)) continue;
            if ((pass == 0) == vw2_key_has_wildcard(&c->key)) continue;
            return c;
        }
    return NULL;
}

static inline const vw2_rec_t *vw2__oop_k1_scan_ord(const vw2_store_t *s, int N,
                                                uint8_t lay, int want_scr)
{
    return vw2__oop_k1_scan_pl(s, N, lay, want_scr, VW2_PL_OOP, 1);
}
static inline const vw2_rec_t *vw2__oop_k1_scan(const vw2_store_t *s, int N, uint8_t lay)
{
    return vw2__oop_k1_scan_ord(s, N, lay, 0);
}

/* COMPAT (kind-3 record split): 1 when ANY kind-3 row - lay=il, lay=split or
 * the lay-less vintage row - exists at this (order, placement, thread count)
 * cell. The pre-split reader filled ONE entry from all three rows and
 * returned "found" when either layout's axis decoded, so an interleaved
 * consumer holding a cell with only a SPLIT row saw a present entry (its IL
 * axis unraced). The interleaved lookup keeps that presence with this test
 * (deferred finding F3, docs/roadmap/layout_separation_plan.md): existence,
 * not decode - the one difference from the old reader is a cell whose rows
 * are ALL undecodable, which the old reader called absent. */
static inline int vw2_oop_k1_row_present(const vw2_store_t *s, int N, int want_scr,
                                         int pl, int T)
{
    return vw2__oop_k1_scan_pl(s, N, VW2_LAY_IL, want_scr, pl, T) != NULL ||
           vw2__oop_k1_scan_pl(s, N, VW2_LAY_SPLIT, want_scr, pl, T) != NULL ||
           vw2__oop_k1_scan_pl(s, N, VW2_LAY_ANY, want_scr, pl, T) != NULL;
}

/* which kind-3 row exists at length M, AS KEYED (lay=il, lay=split, or
 * lay-less): returns VW2_LAY_IL / VW2_LAY_SPLIT / VW2_LAY_ANY, or -1 when
 * none. EXACT key equality on purpose: the serving lookup answers a lay=il
 * request from a lay-less row (the layout fallback tier), and a signpost
 * spelled after the REQUEST rather than the ROW dangles under ref_ok's
 * exact match (the Bluestein inner at M=256 is the known case). */
static inline int vw2_oop_k1_row_lay_ord(const vw2_store_t *s, int M, int want_scr)
{
    static const int lays[3] = { VW2_LAY_IL, VW2_LAY_SPLIT, VW2_LAY_ANY };
    int i, j;
    for (i = 0; i < 3; i++) {
        vw2_key_t k;
        memset(&k, 0, sizeof k);
        k.t = VW2_T_C2C; k.rank = 1; k.n[0] = M;
        k.q = 1; k.ord = want_scr ? VW2_ORD_SCR : VW2_ORD_NAT; k.pl = VW2_PL_OOP;
        k.role = VW2_ROLE_COMP; k.lay = (uint8_t)lays[i];
        for (j = 0; j < s->nrec; j++)
            if (vw2_key_eq(&s->rec[j].key, &k)) return lays[i];
    }
    return -1;
}
static inline int vw2_oop_k1_row_lay(const vw2_store_t *s, int M)
{
    return vw2_oop_k1_row_lay_ord(s, M, 0);
}

#endif /* VFFT_WISDOM2_OOP_ROWS_H */
