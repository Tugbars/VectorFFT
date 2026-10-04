/* chain_search.h - the measured CHAIN SEARCH over a radix pool (owner,
 * 2026-10-04): the flat DITs' chain pick, c2c (il/planning/dp_planner_il.h)
 * and real (il/real/odd_build.h, zrf).
 *
 * A flat DIT plan is a chain: stage radices whose product is N, leaf first.
 * Its pool is every ordered composition of N over the engine's radices (up to
 * VFFT_CHAIN_MAX_K stages): 92 chains at 945, 1385 at 50625, 41007 at 893025
 * -- almost all of them ORDERINGS of a few radix SETS (8 to 53 below 2^20) --
 * and the engine's own bench of a chain (its forms, its tile, its threaded
 * arms) is expensive. So the pick runs in measured levels, nothing cut
 * unmeasured:
 *   1. RADIX SETS: every set in two orders -- its lowest radix as the first
 *      stage, and its second-lowest as the first stage, the rest from small
 *      to large either way (owner, 2026-10-04: one fixed order can be the
 *      cache-unfriendly one and sink a set that wins in another; where an
 *      order does not build, the set's next one that does) -- raced in
 *      heats; a set counts by its better order;
 *   2. ORDERINGS of the best three sets: every distinct ordering raced in
 *      heats where a set has at most VFFT_CHAIN_ORD_EXH of them; a larger set
 *      is ordered STAGE BY STAGE (each position takes the radix whose chain
 *      -- the rest from small to large -- runs fastest): measured, not
 *      exhaustive, and said so on stderr;
 *   3. the best two orderings of each of those sets are the FINALISTS: the
 *      engine benches them its own way (forms, tile, threaded arms) and its
 *      planner ranks them.
 * A heat holds at most VFFT_CHAIN_HEAT chains, fewer where their plans
 * together would pass the engine's memory budget; each heat's best (its top
 * three where heats are wide enough to shrink by them, its winner otherwise)
 * race again until one heat stands (the tournament: heats.h). A lone factor
 * 2 (the c2c engine at N = 2 mod 4) is a fixed leaf.
 *
 * THE ENGINE supplies the HEAT: build every chain of a list, gate it against
 * the engine's reference, race the ones that pass in ONE same-run race
 * (arms alternating, round by round), and report each chain's time (1e18 =
 * refused or wrong); with `fix` set, a chain that does not build moves to its
 * set's next ordering (vfft_chain_next) and is raced there. This header
 * enumerates, orders and runs the tournaments.
 */
#ifndef VFFT_IL_CHAIN_SEARCH_H
#define VFFT_IL_CHAIN_SEARCH_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "il/planning/heats.h"   /* the tournament in heats */

#define VFFT_CHAIN_MAX_K    10      /* stages, at most (the flat DITs' VFFT_ILFD_MAX_K) */
#define VFFT_CHAIN_HEAT     VFFT_HEATS_MAX   /* chains per heat, at most */
#define VFFT_CHAIN_ORD_EXH  256     /* a set's orderings raced exhaustively, at most */
#define VFFT_CHAIN_SETS_MAX 4096    /* the sets' storage (the most below 2^20 is 53) */
#define VFFT_CHAIN_FIN_MAX  6       /* finalists: two orderings of each of three sets */

typedef struct
{
    int R[VFFT_CHAIN_MAX_K];   /* the stages, leaf first */
    int n;                     /* how many */
    int lead2;                 /* 1 = R[0] is the fixed lone 2 */
} vfft_chain_t;

/* the engine's heat: chains f[idx[0..n)] (n <= VFFT_CHAIN_HEAT) -> ns[k] */
typedef void (*vfft_chain_heat_fn)(void *hctx, vfft_chain_t *f, const int *idx, int n, int fix, double *ns);

typedef struct
{
    const int *pool;           /* the engine's radices */
    int npool;
    int lead2;                 /* 1 = a lone factor 2 may lead as the fixed leaf (N = 2 mod 4) */
    vfft_chain_heat_fn heat;
    void *hctx;
    size_t budget;             /* a heat's plans together, at most (bytes) ... */
    size_t per_stage_point;    /* ... a plan estimated at N * stages * this */
    const char *tag;           /* the engine's stderr tag */
} vfft_chain_engine_t;

typedef struct { int nsets, nord, cap, heats; long chains; } vfft_chain_stats_t;

/* every radix set of L over the pool, the stages small to large, after
 * `depth` fixed stages */
static void _vfft_chain_sets_rec(const vfft_chain_engine_t *e, int L, int i0, int depth, int *cur, int lead2,
                                 const int *ord, vfft_chain_t *out, int *n, int *over)
{
    int i;
    if (L == 1)
    {
        if (depth < 2) return;
        if (*n >= VFFT_CHAIN_SETS_MAX) { (*over)++; return; }
        memset(&out[*n], 0, sizeof out[*n]);
        memcpy(out[*n].R, cur, sizeof(int) * (size_t)depth);
        out[*n].n = depth;
        out[*n].lead2 = lead2;
        (*n)++;
        return;
    }
    if (depth >= VFFT_CHAIN_MAX_K) return;
    for (i = i0; i < e->npool; i++)
        if (L % ord[i] == 0)
        {
            cur[depth] = ord[i];
            _vfft_chain_sets_rec(e, L / ord[i], i, depth + 1, cur, lead2, ord, out, n, over);
        }
}
/* the stages after the fixed leaf small to large */
static void vfft_chain_sortval(vfft_chain_t *f)
{
    int i, j, t;
    for (i = f->lead2 + 1; i < f->n; i++)
        for (j = i; j > f->lead2 && f->R[j] < f->R[j - 1]; j--)
        { t = f->R[j]; f->R[j] = f->R[j - 1]; f->R[j - 1] = t; }
}
/* from small-to-large: the second-lowest radix moved to the first stage
 * (after the fixed leaf), the rest kept small to large; 0 when the set has
 * one radix value */
static int vfft_chain_second_first(vfft_chain_t *f)
{
    int q = f->lead2 + 1, r, v;
    while (q < f->n && f->R[q] == f->R[f->lead2]) q++;
    if (q >= f->n) return 0;
    v = f->R[q];
    for (r = q; r > f->lead2; r--) f->R[r] = f->R[r - 1];
    f->R[f->lead2] = v;
    return 1;
}
/* the next distinct ordering of the stages after the fixed leaf (ascending
 * value order, as std::next_permutation); 0 after the last */
static int vfft_chain_next(vfft_chain_t *f)
{
    int i = f->n - 2, j, t;
    while (i >= f->lead2 && f->R[i] >= f->R[i + 1]) i--;
    if (i < f->lead2) return 0;
    j = f->n - 1;
    while (f->R[j] <= f->R[i]) j--;
    t = f->R[i]; f->R[i] = f->R[j]; f->R[j] = t;
    for (i = i + 1, j = f->n - 1; i < j; i++, j--) { t = f->R[i]; f->R[i] = f->R[j]; f->R[j] = t; }
    return 1;
}
/* how many distinct orderings the set has */
static long vfft_chain_norders(const vfft_chain_t *f)
{
    vfft_chain_t s = *f;
    long r = 1;
    int i, k = f->n - f->lead2, run = 1;
    vfft_chain_sortval(&s);
    for (i = 2; i <= k; i++) r *= i;
    for (i = f->lead2 + 1; i <= f->n; i++)
    {
        if (i < f->n && s.R[i] == s.R[i - 1]) { run++; continue; }
        for (int q = 2; q <= run; q++) r /= q;
        run = 1;
    }
    return r;
}
static void vfft_chain_str(const vfft_chain_t *f, char *b, size_t sz)
{
    int i, off = 0;
    b[0] = 0;
    for (i = 0; i < f->n && off < (int)sz - 4; i++)
        off += snprintf(b + off, sz - (size_t)off, "%s%d", i ? "." : "", f->R[i]);
}

/* A TOURNAMENT over f[idx[0..n)] (heats.h) on the engine's heat; that
 * heat's ranking (fastest first) fills top[0..keep). Returns how many it
 * filled. */
typedef struct { const vfft_chain_engine_t *e; vfft_chain_t *f; int fix; } _vfft_chain_hc_t;
static void _vfft_chain_heat(void *v, const int *idx, int n, double *ns)
{
    _vfft_chain_hc_t *h = (_vfft_chain_hc_t *)v;
    h->e->heat(h->e->hctx, h->f, idx, n, h->fix, ns);
}
static int vfft_chain_tourney(const vfft_chain_engine_t *e, vfft_chain_t *f, const int *idx, int n, int fix,
                              int cap, int keep, int carry, int *top, int *heats)
{
    _vfft_chain_hc_t h;
    h.e = e; h.f = f; h.fix = fix;
    return vfft_heats_run(_vfft_chain_heat, &h, idx, n, cap, keep, carry, top, NULL, heats);
}
/* a set ordered STAGE BY STAGE: at each position the remaining radices are
 * offered one at a time (the rest small to large behind it) and the fastest
 * chain fixes that position; *f is left at the ordering */
static void vfft_chain_stagewise(const vfft_chain_engine_t *e, vfft_chain_t *f, int cap, int *heats)
{
    vfft_chain_t c[VFFT_CHAIN_MAX_K];
    int idx[VFFT_CHAIN_MAX_K], p, q, nc, top;
    vfft_chain_sortval(f);
    for (p = f->lead2; p < f->n - 1; p++)
    {
        nc = 0;
        for (q = p; q < f->n; q++)
        {
            int r;
            if (q > p && f->R[q] == f->R[q - 1]) continue;   /* the same radix: the same chain */
            c[nc] = *f;
            for (r = q; r > p; r--) c[nc].R[r] = c[nc].R[r - 1];
            c[nc].R[p] = f->R[q];
            idx[nc] = nc;
            nc++;
        }
        if (nc < 2) continue;
        if (vfft_chain_tourney(e, c, idx, nc, 0, cap, 1, 1, &top, heats) == 1)
            *f = c[top];
    }
}

/* THE SEARCH at N: the finalists (at most VFFT_CHAIN_FIN_MAX, each once)
 * into fin[]; returns how many. st (may be NULL) receives the counts. */
static int vfft_chain_search(const vfft_chain_engine_t *e, int N, vfft_chain_t *fin, vfft_chain_stats_t *st)
{
    vfft_chain_t *sets = (vfft_chain_t *)malloc(sizeof(vfft_chain_t) * VFFT_CHAIN_SETS_MAX), *ent = NULL, rep[3], cand[VFFT_CHAIN_FIN_MAX];
    int *ord = (int *)malloc(sizeof(int) * (size_t)(e->npool > 0 ? e->npool : 1));
    int cur[VFFT_CHAIN_MAX_K], ns = 0, over = 0, i, nmax = 2, cap, top[6], best[3], nt, nb = 0, ne = 0, nc = 0,
        heats = 0, nf = 0;
    long chains = 0;
    size_t per;
    if (st) memset(st, 0, sizeof *st);
    if (!sets || !ord) { free(sets); free(ord); return 0; }
    for (i = 0; i < e->npool; i++) ord[i] = e->pool[i];
    for (i = 1; i < e->npool; i++)   /* the pool small to large: a set's stages come out sorted */
        for (int j = i; j > 0 && ord[j] < ord[j - 1]; j--) { const int t = ord[j]; ord[j] = ord[j - 1]; ord[j - 1] = t; }
    memset(cur, 0, sizeof cur);
    if (e->lead2 && (N & 1) == 0 && ((N >> 1) & 1))
    {   /* the lone factor 2: the leaf */
        cur[0] = 2;
        _vfft_chain_sets_rec(e, N >> 1, 0, 1, cur, 1, ord, sets, &ns, &over);
    }
    _vfft_chain_sets_rec(e, N, 0, 0, cur, 0, ord, sets, &ns, &over);
    free(ord);
    if (over)
        fprintf(stderr, "%s N=%d: chain radix-set storage (%d) exceeded, %d set(s) not raced\n",
                e->tag, N, VFFT_CHAIN_SETS_MAX, over);
    if (ns == 0) { free(sets); return 0; }
    for (i = 0; i < ns; i++)
    {
        chains += vfft_chain_norders(&sets[i]);
        if (sets[i].n > nmax) nmax = sets[i].n;
    }
    per = (size_t)N * e->per_stage_point * (size_t)nmax;
    cap = (int)(e->budget / (per ? per : 1));
    if (cap > VFFT_CHAIN_HEAT) cap = VFFT_CHAIN_HEAT;
    if (cap < 2) cap = 2;
    {   /* 1. the radix sets, each in two orders: the lowest radix first, the second-lowest first */
        int *idx = (int *)malloc(sizeof(int) * 2u * (size_t)ns), *eset = (int *)malloc(sizeof(int) * 2u * (size_t)ns);
        ent = (vfft_chain_t *)malloc(sizeof(vfft_chain_t) * 2u * (size_t)ns);
        if (!idx || !eset || !ent) { free(idx); free(eset); free(ent); free(sets); return 0; }
        for (i = 0; i < ns; i++)
        {
            ent[ne] = sets[i];
            vfft_chain_sortval(&ent[ne]);
            eset[ne] = i; idx[ne] = ne; ne++;
            ent[ne] = ent[ne - 1];
            if (vfft_chain_second_first(&ent[ne])) { eset[ne] = i; idx[ne] = ne; ne++; }
        }
        nt = vfft_chain_tourney(e, ent, idx, ne, 1, cap, 6, cap >= 12 ? 3 : 1, top, &heats);
        for (i = 0; i < nt && nb < 3; i++)
        {   /* the three best sets, each by its better order */
            int k, seen = 0;
            for (k = 0; k < nb; k++) if (best[k] == eset[top[i]]) seen = 1;
            if (seen) continue;
            best[nb] = eset[top[i]];
            rep[nb] = ent[top[i]];
            nb++;
        }
        free(idx); free(eset);
    }
    for (i = 0; i < nb; i++)
    {   /* 2. the orderings of each of the best sets */
        vfft_chain_t s = rep[i];
        const long no = vfft_chain_norders(&s);
        if (no <= 1)
        {
            if (nc < VFFT_CHAIN_FIN_MAX) cand[nc++] = s;
        }
        else if (no <= VFFT_CHAIN_ORD_EXH)
        {
            vfft_chain_t *ords = (vfft_chain_t *)malloc(sizeof(vfft_chain_t) * (size_t)no);
            int *idx = (int *)malloc(sizeof(int) * (size_t)no), no2 = 0, t2[2], g;
            if (ords && idx)
            {
                vfft_chain_sortval(&s);
                do { ords[no2] = s; idx[no2] = no2; no2++; } while (no2 < no && vfft_chain_next(&s));
                g = vfft_chain_tourney(e, ords, idx, no2, 0, cap, 2, cap >= 8 ? 2 : 1, t2, &heats);
                for (int k = 0; k < g && nc < VFFT_CHAIN_FIN_MAX; k++) cand[nc++] = ords[t2[k]];
            }
            free(ords); free(idx);
        }
        else
        {
            vfft_chain_t g = s;
            char cs[64];
            vfft_chain_str(&s, cs, sizeof cs);
            fprintf(stderr, "%s N=%d: chain set %s has %ld orderings (> %d): ordered stage by stage "
                            "(measured, not exhaustive)\n", e->tag, N, cs, no, VFFT_CHAIN_ORD_EXH);
            vfft_chain_stagewise(e, &g, cap, &heats);
            if (nc < VFFT_CHAIN_FIN_MAX) cand[nc++] = g;
            if (nc < VFFT_CHAIN_FIN_MAX && memcmp(g.R, s.R, sizeof g.R)) cand[nc++] = s;   /* and the order level 1 raced */
        }
    }
    for (i = 0; i < nc; i++)
    {   /* 3. the finalists, once each */
        int k, dup = 0;
        for (k = 0; k < i; k++)
            if (cand[k].n == cand[i].n && !memcmp(cand[k].R, cand[i].R, sizeof(int) * (size_t)cand[i].n)) dup = 1;
        if (!dup) fin[nf++] = cand[i];
    }
    if (st) { st->nsets = ns; st->nord = ne; st->cap = cap; st->heats = heats; st->chains = chains; }
    free(ent);
    free(sets);
    return nf;
}

#endif /* VFFT_IL_CHAIN_SEARCH_H */
