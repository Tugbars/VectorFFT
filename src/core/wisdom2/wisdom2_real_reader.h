/* wisdom2_real_reader.h — the REAL-ROUTE family codec: which top-level route
 * an r2c / c2r problem takes at a given cell.
 *
 * This is a ROUTE verdict, not a plan. The chain each route then uses is a
 * separate record owned by that route's own engine (stride chains live in
 * wisdom2_stride_reader.h; the zr2c payload lives in wisdom2_oop_reader.h) —
 * this record only names the winner of the two-arm race.
 *
 *   r2c:  route=rfft    the packed rfft cascade
 *         route=stride  the decoupled stride path (threads; owns high K)
 *   c2r:  route=natural the packed cascade driven from split input
 *         route=split   the decoupled stride path
 *
 * KEY: t=r2c|c2r, n=REAL N, q=K, ord=nat, place=<as created>. Real transforms
 * are natural-ordered, so ord carries no choice here and is pinned nat — the
 * same canonical-key law the 2D real families use.
 *
 * ENGINE DISCRIMINATOR: eng=route. The problem cell is shared with the zr2c
 * verdict (kind-5 banks eng=zr2c at t=r2c|c2r, q=1, ord=nat), which is a route
 * decision in its own right and is taken ABOVE this one in create. A cell
 * already holding zr2c is therefore already decided: the reader refuses it and
 * the banker declines to overwrite it, so the two never clobber each other.
 *
 * WILDCARDS: c2r_path.txt had no ord/place columns, so its rows migrate as
 * ord=* place=* with from=. The lookup runs the standard two-pass scan —
 * exact beats wildcard — so the first concrete race at a cell retires the
 * migrated row without deleting it.
 *
 * METRIC: one whole-plan execute in the transform's own direction, single
 * threaded — fwd1 for r2c, bwd1 for c2r. arms= records BOTH timings so a
 * banked line reads back as a race, not just a name.
 *
 * DIR: absent on both tags, deliberately. c2r IS the backward transform, but
 * its direction is already carried by t=c2r — dir= exists to separate
 * DIRECTIONAL SIBLINGS sharing one cell (as t=c2c does, where forward and
 * backward pick independently at the same key). r2c and c2r are distinct tags
 * with no sibling, so stamping dir=bwd on c2r would key a cell nothing ever
 * requests. The bwd1 metric label is the direction record here; dir= is not.
 * Both scanners below go through vw2_key_serves, which equality-matches dir
 * and role, so a directional sibling in this shard can never be matched by
 * accident — the hazard that hand-rolled field-by-field scans carry.
 */
#ifndef VFFT_WISDOM2_REAL_READER_H
#define VFFT_WISDOM2_REAL_READER_H

#include <stdio.h>
#include <string.h>
#include <time.h>
#include "wisdom2.h"

/* Family-local, matching the stride/2d codecs — there is no shared stamp. */
static inline void vw2__real_stamp_date(vw2_rec_t *r)
{
    char d[16];
    time_t t = time(NULL);
    struct tm *tm = localtime(&t);
    if (tm && strftime(d, sizeof d, "%Y-%m-%d", tm))
        vw2_rec_set(r, 2, "date", d);
}

typedef enum {
    VW2_RROUTE_NONE    = 0,
    VW2_RROUTE_RFFT    = 1,   /* r2c */
    VW2_RROUTE_STRIDE  = 2,   /* r2c */
    VW2_RROUTE_NATURAL = 3,   /* c2r */
    VW2_RROUTE_SPLIT   = 4    /* c2r */
} vw2_real_route_t;

static const char *vw2_real_route_name[5] = {
    "none", "rfft", "stride", "natural", "split"
};

/* The two legal arms per transform tag — a route that does not belong to the
 * tag is a refusal, never a silent accept. */
static inline int vw2__real_route_fits(int t, int route)
{
    if (t == VW2_T_R2C) return route == VW2_RROUTE_RFFT || route == VW2_RROUTE_STRIDE;
    if (t == VW2_T_C2R) return route == VW2_RROUTE_NATURAL || route == VW2_RROUTE_SPLIT;
    return 0;
}

static inline int vw2__real_route_idx(const char *v)
{
    int i;
    if (!v) return -1;
    for (i = 1; i < 5; i++)
        if (!strcmp(vw2_real_route_name[i], v)) return i;
    return -1;
}

static inline const char *vw2__real_eng(const vw2_rec_t *r)
{
    const char *e = vw2_rec_get(r, "eng");
    return e ? e : "";
}

static inline void vw2__real_key(vw2_key_t *k, int t, int N, size_t K, int ord, int pl)
{
    memset(k, 0, sizeof *k);
    k->t = (uint8_t)t;
    k->rank = 1;
    k->n[0] = N;
    k->q = (int64_t)K;
    k->ord = (int8_t)ord;
    k->pl = (int8_t)pl;
}

/* ------------------------------------------------------------------ read */

/* Returns the banked route for (t,N,K,place), or VW2_RROUTE_NONE on a miss.
 * Two-pass: a concrete verdict beats a migrated wildcard row. */
static inline int vw2_real_route_lookup(const vw2_store_t *s, int t,
                                        int N, size_t K, int pl)
{
    vw2_key_t req;
    int i, pass, route;
    const vw2_rec_t *r = NULL;
    if (N < 2 || K < 1) return VW2_RROUTE_NONE;      /* junk cell: refuse */
    if (t != VW2_T_R2C && t != VW2_T_C2R) return VW2_RROUTE_NONE;
    vw2__real_key(&req, t, N, K, VW2_ORD_NAT, pl);
    for (pass = 0; pass < 2 && !r; pass++)
        for (i = 0; i < s->nrec; i++) {
            const vw2_rec_t *c = &s->rec[i];
            if (!vw2_key_serves(&c->key, &req)) continue;
            if (strcmp(vw2__real_eng(c), "route")) continue;   /* zr2c/other: not ours */
            if (vw2__is_seed(c)) continue;
            if ((pass == 0) == vw2_key_has_wildcard(&c->key)) continue;
            r = c;
            break;
        }
    if (!r) return VW2_RROUTE_NONE;
    route = vw2__real_route_idx(vw2_rec_get(r, "route"));
    if (route < 0 || !vw2__real_route_fits(t, route)) return VW2_RROUTE_NONE;
    return route;
}

/* 1 when the problem cell is already owned by a different engine (today:
 * zr2c). Such a cell is decided above this race and must not be overwritten. */
static inline int vw2_real_cell_taken(const vw2_store_t *s, int t,
                                      int N, size_t K, int pl)
{
    vw2_key_t req;
    int i;
    vw2__real_key(&req, t, N, K, VW2_ORD_NAT, pl);
    for (i = 0; i < s->nrec; i++) {
        const vw2_rec_t *c = &s->rec[i];
        if (!vw2_key_serves(&c->key, &req)) continue;
        if (strcmp(vw2__real_eng(c), "route")) return 1;
    }
    return 0;
}

/* ----------------------------------------------------------------- write */

#define VW2__RB_SET(sect, n, v) do { \
    if (vw2_rec_set(r, sect, n, v) != VW2_OK) { vw2_rec_free(r); *why = "token-refused"; return -1; } \
} while (0)

/* Build a route record. ns_win/ns_lose <= 0 => measure-less (a migrated row,
 * or a verdict taken without timing); arms= is emitted only when both arms
 * were actually timed. ord/pl of VW2_*_ANY produce the migration wildcard,
 * which the store refuses without a from=. */
static inline int vw2_real_rec_from_route(vw2_rec_t *r, int t, int N, size_t K,
                                          int ord, int pl, int route,
                                          double ns_win, double ns_lose,
                                          const char *src, const char *from,
                                          const char **why)
{
    char b[64];
    *why = NULL;
    memset(r, 0, sizeof *r);
    if (N < 2 || K < 1) { *why = "junk-cell"; return -1; }
    if (!vw2__real_route_fits(t, route)) { *why = "route-not-legal-for-transform"; return -1; }
    vw2__real_key(&r->key, t, N, K, ord, pl);
    VW2__RB_SET(1, "eng", "route");
    VW2__RB_SET(1, "route", vw2_real_route_name[route]);
    snprintf(b, sizeof b, "%zu", K);
    VW2__RB_SET(2, "ran", b);
    if (ns_win > 0.0) {
        snprintf(b, sizeof b, "%.2f", ns_win);
        VW2__RB_SET(2, "ns", b);
        VW2__RB_SET(2, "metric", t == VW2_T_C2R ? "bwd1" : "fwd1");
        VW2__RB_SET(2, "units", "ns");
        if (ns_lose > 0.0) {
            /* both arms, winner first — the line reads back as the race it was */
            snprintf(b, sizeof b, "%.2f/%.2f", ns_win, ns_lose);
            VW2__RB_SET(2, "arms", b);
        }
    }
    VW2__RB_SET(2, "src", src);
    if (from) VW2__RB_SET(2, "from", from);
    else vw2__real_stamp_date(r);
    return 0;
}

/* Bank a raced route verdict. Declines (quietly, returning 0) when the cell
 * belongs to another engine — that is a correct outcome, not an error. */
static inline int vw2_real_route_bank(vw2_store_t *st, int t, int N, size_t K,
                                      int pl, int route,
                                      double ns_win, double ns_lose)
{
    vw2_rec_t rec;
    const char *why = NULL;
    int rc;
    if (vw2_real_cell_taken(st, t, N, K, pl)) return 0;
    if (vw2_real_rec_from_route(&rec, t, N, K, VW2_ORD_NAT, pl, route,
                                ns_win, ns_lose, "race", NULL, &why)) {
        fprintf(stderr, "[wisdom2] route bank refused (%s)\n", why ? why : "?");
        return -1;
    }
    rc = vw2_bank(st, &rec);
    if (rc != VW2_OK) vw2_rec_free(&rec);
    return rc;
}

#endif /* VFFT_WISDOM2_REAL_READER_H */
