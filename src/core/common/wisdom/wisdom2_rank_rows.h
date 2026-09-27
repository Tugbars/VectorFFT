/* wisdom2_rank_rows.h — the rank>=2 row helpers both layouts' 2D/3D codecs
 * use: the canonical request key, the record key, the MEASURE/provenance
 * tail and the fill-only bank. Carved verbatim out of wisdom2_2d_reader.h
 * (layout separation phase 5): the split codec is
 * split/wisdom/wisdom2_2d_split_reader.h, the interleaved one
 * il/wisdom/wisdom2_2d_il_reader.h. */
#ifndef VFFT_WISDOM2_RANK_ROWS_H
#define VFFT_WISDOM2_RANK_ROWS_H

#include <string.h>
#include <time.h>
#include "wisdom2.h"

/* canonical request key (see the header law)
 *
 * lay= (v1.2): the CALLER's layout, on requests and fresh banks alike —
 * the split rank>=2 plans bank lay=split, the native IL tier banks its own
 * lay=il cells (below); pre-1.2 lay=ANY rows serve both layouts through
 * vw2_lookup's fallback phase. */
static inline void vw2__2d_key(vw2_key_t *k, int t, int rank,
                               int n0, int n1, int n2, int ord, uint8_t lay,
                               int nthreads)
{
    memset(k, 0, sizeof *k);
    k->t = (uint8_t)t;
    k->rank = (uint8_t)rank;
    k->n[0] = n0; k->n[1] = n1; k->n[2] = n2;
    k->q = 1;
    k->ord = (int8_t)ord;
    k->pl = VW2_PL_OOP;                       /* canonical: placement-blind */
    k->lay = lay;
    k->nthreads = (uint8_t)(nthreads > 1 ? nthreads : 0);   /* the plan's thread count (v1.3) */
}

static inline void vw2__2d_stamp_date(vw2_rec_t *r)
{
    char d[16];
    time_t t = time(NULL);
    struct tm *tm = localtime(&t);
    if (tm && strftime(d, sizeof d, "%Y-%m-%d", tm))
        vw2_rec_set(r, 2, "date", d);
}

#define VW2__2D_SET(sect, n, v) do { \
    if (vw2_rec_set(r, sect, n, v) != VW2_OK) { vw2_rec_free(r); *why = "token-refused"; return -1; } \
} while (0)

/* shared tail: MEASURE section + provenance. ns<=0 => measure-less. */
static inline int vw2__2d_tail(vw2_rec_t *r, double ns, const char *src,
                               const char *from, const char **why)
{
    VW2__2D_SET(2, "ran", "1");
    if (ns > 0.0) {
        char nsbuf[48];
        snprintf(nsbuf, sizeof nsbuf, "%.1f", ns);
        VW2__2D_SET(2, "ns", nsbuf);
        VW2__2D_SET(2, "metric", "fwd1");
        VW2__2D_SET(2, "units", "ns");
    }
    VW2__2D_SET(2, "src", src);
    if (from) VW2__2D_SET(2, "from", from);
    else vw2__2d_stamp_date(r);
    return 0;
}

/* shared key emit; migrated rows wildcard the axes the legacy tables never
 * keyed (place always; ord too for the order-blind real families). */
static inline void vw2__2d_rec_key(vw2_rec_t *r, int t, int rank,
                                   int n0, int n1, int n2, int ord,
                                   int migrated, int ord_blind, uint8_t lay,
                                   int nthreads)
{
    memset(&r->key, 0, sizeof r->key);
    r->key.nthreads = (uint8_t)(nthreads > 1 ? nthreads : 0);   /* the plan's thread count (v1.3) */
    r->key.t = (uint8_t)t;
    r->key.rank = (uint8_t)rank;
    r->key.n[0] = n0; r->key.n[1] = n1; r->key.n[2] = n2;
    r->key.q = 1;
    if (migrated) {
        r->key.ord = ord_blind ? VW2_ORD_ANY : (int8_t)ord;
        r->key.pl = VW2_PL_ANY;
        /* legacy files never recorded layout: VW2_LAY_ANY vintage */
    } else {
        r->key.ord = (int8_t)ord;             /* canonical concrete */
        r->key.pl = VW2_PL_OOP;
        r->key.lay = lay;
    }
}

static inline int vw2__2d_bank(vw2_store_t *st, vw2_rec_t *rec, int fill_only)
{
    int rc;
    if (fill_only && vw2_lookup(st, &rec->key)) {
        vw2_rec_free(rec);
        return VW2_OK;                        /* warm cell: keep it */
    }
    rc = vw2_bank(st, rec);
    if (rc != VW2_OK) vw2_rec_free(rec);      /* tokens move only on success */
    return rc;
}

#endif /* VFFT_WISDOM2_RANK_ROWS_H */
