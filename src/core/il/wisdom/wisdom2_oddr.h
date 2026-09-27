/* wisdom2_oddr.h — the ODD-REAL route verdict row (lay=il component row).
 *
 * Raced only for interleaved requests (K=1 OOP IL r2c at odd N: the rfft handle
 * against the c2c bridge). Carved verbatim out of
 * split/wisdom/wisdom2_real_reader.h (layout separation phase 5): an IL row in
 * the split real-route codec. */
#ifndef VFFT_WISDOM2_ODDR_H
#define VFFT_WISDOM2_ODDR_H

#include <string.h>
#include <time.h>
#include "wisdom2.h"

/* ── the ODD-REAL route verdict ───────────────────────────────────────────
 * K=1 OOP IL r2c at odd N races its rfft handle against the c2c bridge and
 * banks the answer here. The q=1 real key belongs to the zr2c verdicts, so
 * this is a COMPONENT row beside them:
 * t=r2c n=N q=1 ord=nat place=oop role=comp lay=il | eng=oddr route=rfft|bridge.
 * 0 = no verdict, 1 = rfft, 2 = bridge. */
static inline void vw2__oddr_key(int N, vw2_key_t *k)
{
    memset(k, 0, sizeof *k);
    k->t = VW2_T_R2C; k->rank = 1; k->n[0] = N;
    k->q = 1; k->ord = VW2_ORD_NAT; k->pl = VW2_PL_OOP;
    k->role = VW2_ROLE_COMP; k->lay = VW2_LAY_IL;
}
static inline int vw2_oddr_route_lookup(const vw2_store_t *s, int N)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    const char *eng, *route;
    vw2__oddr_key(N, &k);
    r = vw2_lookup(s, &k);
    if (!r) return 0;
    eng = vw2_rec_get(r, "eng");
    route = vw2_rec_get(r, "route");
    if (!eng || !route || strcmp(eng, "oddr")) return 0;
    if (!strcmp(route, "rfft")) return 1;
    if (!strcmp(route, "bridge")) return 2;
    return 0;
}
static inline int vw2_oddr_route_bank(vw2_store_t *s, int N, int route)
{
    vw2_rec_t r;
    int rc;
    memset(&r, 0, sizeof r);
    vw2__oddr_key(N, &r.key);
    if (vw2_rec_set(&r, 1, "eng", "oddr") != VW2_OK ||
        vw2_rec_set(&r, 1, "route", route == 2 ? "bridge" : "rfft") != VW2_OK ||
        vw2_rec_set(&r, 2, "ran", "1") != VW2_OK ||
        vw2_rec_set(&r, 2, "src", "race") != VW2_OK) {
        vw2_rec_free(&r);
        return -1;
    }
    {
        char d[16];
        time_t t = time(NULL);
        struct tm *tm = localtime(&t);
        if (tm && strftime(d, sizeof d, "%Y-%m-%d", tm))
            vw2_rec_set(&r, 2, "date", d);
    }
    rc = vw2_bank(s, &r);
    if (rc != VW2_OK) vw2_rec_free(&r);
    return rc;
}

#endif /* VFFT_WISDOM2_ODDR_H */
