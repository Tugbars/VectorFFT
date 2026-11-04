/* wisdom2_tcmt.h — the contiguous batch's THREADING verdict: its row and codec.
 *
 * A K > 1 transform-contiguous batch has one threaded arm, the serial loop vs
 * slabs of ceil(K/T) whole transforms over the pool's slots (the interleaved
 * wrapper's clones, vfft.c _tc_mt_decide; zturn-ts's worker planes,
 * split/rank1/c2c_create_split.h). The verdict is raced at create on the
 * batch's OWN cell at the plan's thread count and banked as
 *   @cell t=... n=N q=K ord= place= lay=il|split nthreads=T | eng=tcb tcmt=0|1
 * The thread count is a key axis like every threaded verdict's (v1.3): the
 * slabs are ceil(K/T) transforms a worker, so which way wins moves with T
 * (K = 4 leaves half of eight workers idle). eng=tcb keeps the row out of
 * every plan reader (their family gate rejects it), so a batch row can never
 * be served as a plan. Shared by both layouts: common/, never a library's.
 */
#ifndef VFFT_WISDOM2_TCMT_H
#define VFFT_WISDOM2_TCMT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "common/wisdom/wisdom2.h"

static inline void vw2__tcmt_key(vw2_key_t *k, int t, int N, size_t K,
                                 int ord, int pl, uint8_t lay, int T)
{
    memset(k, 0, sizeof *k);
    k->t = (uint8_t)t;
    k->rank = 1;
    k->n[0] = N;
    k->q = (int64_t)K;
    k->ord = (int8_t)ord;
    k->pl = (int8_t)pl;
    k->lay = lay;
    k->nthreads = (uint8_t)(T > 1 ? T : 0);
}

static inline int vw2_tcmt_lookup(const vw2_store_t *s, int t, int N,
                                  size_t K, int ord, int pl,
                                  uint8_t lay, int T, int *tcmt)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    const char *eng, *v;
    vw2__tcmt_key(&k, t, N, K, ord, pl, lay, T);
    r = vw2_lookup(s, &k);
    if (!r) return 0;
    eng = vw2_rec_get(r, "eng");
    if (!eng || strcmp(eng, "tcb")) return 0;
    v = vw2_rec_get(r, "tcmt");
    if (!v) return 0;
    *tcmt = atoi(v) ? 1 : 0;
    return 1;
}

/* the verdict banked: eng=tcb tcmt= | ran= ns= metric=fwd1 units=ns src=race date= */
static inline int vw2_tcmt_bank(vw2_store_t *st, int t, int N,
                                size_t K, int ord, int pl, uint8_t lay,
                                int tcmt, int T, double ns)
{
    vw2_rec_t rec;
    char b[32];
    int rc;
    memset(&rec, 0, sizeof rec);
    vw2__tcmt_key(&rec.key, t, N, K, ord, pl, lay, T);
    if (vw2_rec_set(&rec, 1, "eng", "tcb") != VW2_OK ||
        vw2_rec_set(&rec, 1, "tcmt", tcmt ? "1" : "0") != VW2_OK)
        goto refuse;
    snprintf(b, sizeof b, "%zu", K);
    if (vw2_rec_set(&rec, 2, "ran", b) != VW2_OK) goto refuse;
    if (ns > 0.0)
    {
        snprintf(b, sizeof b, "%.2f", ns);
        if (vw2_rec_set(&rec, 2, "ns", b) != VW2_OK ||
            vw2_rec_set(&rec, 2, "metric", "fwd1") != VW2_OK ||
            vw2_rec_set(&rec, 2, "units", "ns") != VW2_OK)
            goto refuse;
    }
    if (vw2_rec_set(&rec, 2, "src", "race") != VW2_OK) goto refuse;
    {
        char d[16];
        time_t now = time(NULL);
        struct tm *tmv = localtime(&now);
        if (tmv && strftime(d, sizeof d, "%Y-%m-%d", tmv))
            vw2_rec_set(&rec, 2, "date", d);
    }
    rc = vw2_bank(st, &rec);
    if (rc != VW2_OK) vw2_rec_free(&rec);
    return rc;
refuse:
    vw2_rec_free(&rec);
    fprintf(stderr, "[wisdom2] tcmt bank refused (token-refused)\n");
    return -1;
}

#endif /* VFFT_WISDOM2_TCMT_H */
