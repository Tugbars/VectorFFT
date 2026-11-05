/* zts_race.h — zturn-ts's create-time race on the shared race body
 * (common/support/race.h): the five parts, docs/design/planning_model.md §4.
 *
 * ARMS      the pool at N: every registry cell at N (the chains the fused
 *           drivers exist for — the enumeration IS the registry walk, as in
 *           the IL planner's _il_dp_enumerate_ztt_ord) x {untiled, each
 *           ladder width the tile law admits}, in the request's lane mode;
 *           plus the INCUMBENT when the caller has one: the engine that holds
 *           the cell today, run as deployed (the stride natural execute, the
 *           kind-3 split route) — never a model of it.
 * PROTOCOL  rounds 7, reps by N (short cells repeat), MEDIAN, alternating,
 *           paced (single-thread arms), every sample re-seeded (the in-place
 *           arms alias their buffers; repeated forwards walk into inf).
 * VERDICT   the fastest zturn-ts arm; with an incumbent it serves only when
 *           it beats the incumbent by VFFT_RACE_HYST (the house 3 %) —
 *           re-creates must not flap. No incumbent (a cell nothing serves
 *           today: the lane-major K=4 out-of-place batch) -> the fastest.
 * HEATS     the candidates of a heat coexist in memory (forward streams only,
 *           one shared scratch) under VFFT_ZTS_RACE_BUDGET; a cell with more
 *           chains than fit runs several heats, each with the incumbent as
 *           its yardstick, and the heat winners meet in a final heat with the
 *           incumbent. One heat is the common case (a 4096 cell: 12 chains).
 * KEY/BANK  the caller's: the split natural cell (wisdom2_zts_split.h).
 *
 * Returns the winning plan, complete (backward stream, own scratch, tile
 * set), or NULL when the incumbent holds or the registry has no cell at N;
 * *win_ns / *inc_ns report the aggregates (0 when absent). VFFT_NAT_LOG
 * prints the verdict line. */
#ifndef VFFT_ZTS_RACE_H
#define VFFT_ZTS_RACE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "split/rank1/zts.h"
#include "common/support/race.h"
#include "common/support/zalloc.h"

#define VFFT_ZTS_RACE_ROUNDS 7
#define VFFT_ZTS_RACE_BUDGET (48u * 1024u * 1024u)   /* bytes of candidate streams per heat */
#define VFFT_ZTS_RACE_MAX_PER_PLAN (1 + VFFT_ZTS_TILE_LADDER_N)

typedef struct
{
    vfft_zts_plan_t *p;
    size_t tile;
    const double *ire, *iim;
    double *ore, *oim;
    size_t K;
} _zts_arm_t;

static void _zts_arm_run(void *v)
{
    _zts_arm_t *a = (_zts_arm_t *)v;
    a->p->tile = a->tile;
    vfft_zts_execute_fwd(a->p, a->ire, a->iim, a->ore, a->oim, a->K);
}

/* the reseed before every timed sample: every buffer the arms alias */
typedef struct
{
    double *b[8];
    size_t n;
    int nb;
} _zts_reseed_t;

static void _zts_reseed(void *v)
{
    _zts_reseed_t *r = (_zts_reseed_t *)v;
    int k;
    size_t i;
    for (k = 0; k < r->nb; k++)
        for (i = 0; i < r->n; i++)
            r->b[k][i] = (double)((i * (k & 1 ? 40503u : 2654435761u)) & 1023) / 1024.0 - 0.5;
}

typedef struct
{
    int chain[VFFT_ZTS_MAX_NF];
    int nf;
    size_t bytes;        /* the forward stream's bytes */
    size_t tile;         /* the heat winner's tile (the final heat) */
    double ns;
} _zts_cand_t;

static size_t _zts_chain_bytes(const int *chain, int nf)
{
    long L = chain[0];
    size_t d = 0;
    int s;
    for (s = 1; s < nf; s++)
    {
        d += (size_t)2 * (size_t)(chain[s] - 1) * (size_t)L;
        L *= chain[s];
    }
    return d * sizeof(double);
}
/* the odd band's pool (zts.h vfft_zts_odd_band): the chain GRAMMAR — the
 * odd part of N decomposed greedily largest-first over {15, 9, 7, 5, 3}, its
 * mids at every interior position, the pow2 slots over ordered {4, 8}, the
 * ends in {4, 8}, nf <= 7, (N / R0) % VW == 0 (the port of the interleaved
 * planner's _il_dp_enumerate_ztt_odd) */
static int _zts_pool_odd(int N, _zts_cand_t *out, int max)
{
    int mids[VFFT_ZTS_MAX_NF], nm = 0, m = N, i, nf, n = 0;
    long pw;
    if (!vfft_zts_odd_band(N)) return 0;
    while ((m & 1) == 0) m >>= 1;
    for (i = 0; i < VFFT_ZTS_ODD_MIDS_N; i++)
        while (m % VFFT_ZTS_ODD_MIDS[i] == 0) { mids[nm++] = VFFT_ZTS_ODD_MIDS[i]; m /= VFFT_ZTS_ODD_MIDS[i]; }
    pw = N;
    for (i = 0; i < nm; i++) pw /= mids[i];
    for (nf = nm + 2; nf <= VFFT_ZTS_MAX_NF && n < max; nf++)
    {
        const int np = nf - nm;                /* power-of-two slots */
        long mask;
        for (mask = 0; mask < (1L << np) && n < max; mask++)
        {
            int pchain[VFFT_ZTS_MAX_NF], pos[VFFT_ZTS_MAX_NF];
            long prod = 1;
            for (i = 0; i < np; i++) { pchain[i] = ((mask >> i) & 1) ? 8 : 4; prod *= pchain[i]; }
            if (prod != pw) continue;
            for (i = 0; i < nm; i++) pos[i] = i + 1;
            for (;;)
            {
                int chain[VFFT_ZTS_MAX_NF], pi = 0, mi = 0, q, k;
                for (i = 0; i < nf; i++)
                    chain[i] = (mi < nm && pos[mi] == i) ? mids[mi++] : pchain[pi++];
                if ((N / chain[0]) % VFFT_ZTS_VW == 0 && chain[0] % VFFT_ZTS_VW == 0 && n < max)
                {
                    memset(&out[n], 0, sizeof out[n]);
                    for (q = 0; q < nf; q++) out[n].chain[q] = chain[q];
                    out[n].nf = nf;
                    out[n].bytes = _zts_chain_bytes(chain, nf);
                    n++;
                }
                k = nm - 1;
                while (k >= 0 && pos[k] == nf - 2 - (nm - 1 - k)) k--;
                if (k < 0) break;
                pos[k]++;
                for (q = k + 1; q < nm; q++) pos[q] = pos[q - 1] + 1;
            }
        }
    }
    return n;
}
/* the pool at N: the registry's cells, in registry order; a cell without a
 * fused driver draws its pool from the odd band's grammar */
static int _zts_pool(int N, _zts_cand_t *out, int max)
{
    int i, n = 0;
    for (i = 0; i < VFFT_ZTS_NCELLS && n < max; i++)
    {
        const vfft_zts_cell_t *c = &vfft_zts_cells[i];
        if (c->n != N) continue;
        memset(&out[n], 0, sizeof out[n]);
        memcpy(out[n].chain, c->chain, sizeof out[n].chain);
        out[n].nf = c->nf;
        out[n].bytes = _zts_chain_bytes(c->chain, c->nf);
        n++;
    }
    if (n == 0) n = _zts_pool_odd(N, out, max);
    return n;
}

static int _zts_race_reps(int N)
{
    int r = 16384 / N;
    return r < 4 ? 4 : r > 64 ? 64 : r;
}

/* one heat: cand[0..nc) at their tiles (tile_fixed < 0: untiled + the ladder
 * per candidate; >= 0: each candidate at cand[i].tile only), plus the
 * incumbent. Fills cand[i].ns / .tile with each candidate's best; returns the
 * index of the fastest candidate (-1: nothing built), *inc_ns the incumbent's. */
static int _zts_heat(int N, int lanes, int inplace, _zts_cand_t *cand, int nc, int ladder,
                     const vfft_race_arm_t *inc, double *inc_ns,
                     double *scr_a, double *scr_b,
                     double *ire, double *iim, double *ore, double *oim,
                     _zts_reseed_t *rs)
{
    vfft_zts_plan_t *plans[VFFT_RACE_MAX_ARMS];
    _zts_arm_t ac[VFFT_RACE_MAX_ARMS];
    vfft_race_arm_t arms[VFFT_RACE_MAX_ARMS];
    int owner[VFFT_RACE_MAX_ARMS];
    double ns[VFFT_RACE_MAX_ARMS];
    int i, q, na = 0, best = -1;
    const size_t K = 1;
    if (inc)
    {
        arms[na] = *inc;
        owner[na++] = -1;
    }
    for (i = 0; i < nc; i++)
    {
        plans[i] = NULL;
        cand[i].ns = 0.0;
    }
    for (i = 0; i < nc; i++)
    {
        vfft_zts_plan_t *p = _vfft_zts_create_ex(N, cand[i].chain, cand[i].nf, lanes, 0, 0);
        size_t tiles[VFFT_ZTS_RACE_MAX_PER_PLAN];
        int nt = 0;
        if (!p) continue;
        vfft_zts_share_planes(p, scr_a, scr_b);
        plans[i] = p;
        if (ladder)
        {
            tiles[nt++] = 0;
            for (q = 0; q < VFFT_ZTS_TILE_LADDER_N; q++)
                if (vfft_zts_tile_legal(N, cand[i].chain, cand[i].nf, (size_t)VFFT_ZTS_TILE_LADDER[q]))
                    tiles[nt++] = (size_t)VFFT_ZTS_TILE_LADDER[q];
        }
        else
            tiles[nt++] = cand[i].tile;
        for (q = 0; q < nt && na < VFFT_RACE_MAX_ARMS; q++)
        {
            ac[na].p = p;
            ac[na].tile = tiles[q];
            ac[na].ire = ire;
            ac[na].iim = iim;
            ac[na].ore = inplace ? ire : ore;
            ac[na].oim = inplace ? iim : oim;
            ac[na].K = K;
            arms[na].name = "zts";
            arms[na].run = _zts_arm_run;
            arms[na].ctx = &ac[na];
            owner[na++] = i;
        }
    }
    if (na > (inc ? 1 : 0))
    {
        vfft_race_proto_t proto;
        memset(&proto, 0, sizeof proto);
        proto.rounds = VFFT_ZTS_RACE_ROUNDS;
        proto.reps = _zts_race_reps(N);
        proto.agg = VFFT_RACE_MEDIAN;
        proto.alternate = 1;
        proto.warm = 1;
        proto.reset = _zts_reseed;
        proto.reset_ctx = rs;
        proto.pace = 1;
        if (vfft_race_run(&proto, arms, na, ns) >= 0)
        {
            for (q = 0; q < na; q++)
            {
                const int o = owner[q];
                if (o < 0) { if (inc_ns) *inc_ns = ns[q]; continue; }
                if (cand[o].ns <= 0.0 || ns[q] < cand[o].ns)
                {
                    cand[o].ns = ns[q];
                    cand[o].tile = ac[q].tile;
                }
                if (best < 0 || ns[q] < cand[best].ns || (ns[q] == cand[best].ns && o < best))
                    best = o;
            }
        }
    }
    for (i = 0; i < nc; i++)
        if (plans[i]) vfft_zts_destroy(plans[i]);
    return best;
}

/* The race. inc = the incumbent arm (NULL = none); inc_bufs = the buffers it
 * aliases, re-seeded before every sample (n doubles each, up to 4). */
static vfft_zts_plan_t *vfft_zts_race(int N, int lanes, int inplace,
                                      const vfft_race_arm_t *inc,
                                      double *const *inc_bufs, int inc_nbufs,
                                      double *win_ns, double *inc_ns)
{
    static _zts_cand_t pool[VFFT_ZTS_NCELLS > 256 ? VFFT_ZTS_NCELLS : 256];   /* the registry's cells, or the odd band's chains */
    _zts_cand_t finals[64];
    _zts_reseed_t rs;
    const size_t n = (size_t)N * (size_t)lanes;
    double *ire, *iim, *ore = NULL, *oim = NULL, *scr_a, *scr_b = NULL;
    vfft_zts_plan_t *win = NULL;
    size_t maxb = 0;
    int np, i, g, nfin = 0, best = -1;
    double bns = 0.0, ins = 0.0;
    if (win_ns) *win_ns = 0.0;
    if (inc_ns) *inc_ns = 0.0;
    np = _zts_pool(N, pool, (int)(sizeof pool / sizeof pool[0]));
    if (np <= 0) return NULL;
    for (i = 0; i < np; i++) if (pool[i].bytes > maxb) maxb = pool[i].bytes;
    /* the buffers: the zts arms' planes (+ the out planes out of place), the
     * shared scratch */
    ire = (double *)vfft_aligned_alloc(n * sizeof(double));
    iim = (double *)vfft_aligned_alloc(n * sizeof(double));
    if (!inplace)
    {
        ore = (double *)vfft_aligned_alloc(n * sizeof(double));
        oim = (double *)vfft_aligned_alloc(n * sizeof(double));
    }
    scr_a = (double *)vfft_aligned_alloc(vfft_zts_plane_bytes(N, lanes, 1));
    if (lanes != 1) scr_b = (double *)vfft_aligned_alloc(vfft_zts_plane_bytes(N, lanes, 1));
    if (!ire || !iim || (!inplace && (!ore || !oim)) || !scr_a || (lanes != 1 && !scr_b))
        goto out;
    memset(&rs, 0, sizeof rs);
    rs.n = n;
    rs.b[rs.nb++] = ire;
    rs.b[rs.nb++] = iim;
    for (i = 0; i < inc_nbufs && rs.nb < 8; i++)
        if (inc_bufs[i]) rs.b[rs.nb++] = inc_bufs[i];
    {   /* heats under the budget (and the race body's arm cap) */
        int G = maxb ? (int)(VFFT_ZTS_RACE_BUDGET / maxb) : np;
        const int cap = (VFFT_RACE_MAX_ARMS - 1) / VFFT_ZTS_RACE_MAX_PER_PLAN;
        if (G < 1) G = 1;
        if (G > cap) G = cap;
        for (g = 0; g < np && nfin < 64; g += G)
        {
            const int nc = (np - g) < G ? (np - g) : G;
            const int b = _zts_heat(N, lanes, inplace, pool + g, nc, 1, inc, &ins,
                                    scr_a, scr_b, ire, iim, ore, oim, &rs);
            if (b >= 0) finals[nfin++] = pool[g + b];
        }
        if (nfin == 0) goto out;
        if (nfin == 1)
        {
            best = 0;
            bns = finals[0].ns;
        }
        else
        {   /* the final heat: the heat winners at their tiles, the incumbent again */
            const int b = _zts_heat(N, lanes, inplace, finals, nfin, 0, inc, &ins,
                                    scr_a, scr_b, ire, iim, ore, oim, &rs);
            if (b < 0) goto out;
            best = b;
            bns = finals[b].ns;
        }
    }
    if (inc_ns) *inc_ns = ins;
    if (win_ns) *win_ns = bns;
    if (inc && !(ins > 0.0 && vfft_race_beats(bns, ins, VFFT_RACE_HYST)))
        best = -1;   /* the incumbent holds */
    if (getenv("VFFT_NAT_LOG"))
    {
        char cs[48];
        int s, off = 0;
        const _zts_cand_t *c = &finals[best < 0 ? 0 : best];
        for (s = 0; s < c->nf; s++) off += snprintf(cs + off, sizeof cs - (size_t)off, "%s%d", s ? "." : "", c->chain[s]);
        fprintf(stderr, "[zts-race] N=%d lanes=%d %s: chains=%d best %s tile=%zu %.0f ns%s%.0f ns -> %s\n",
                N, lanes, inplace ? "ip" : "oop", np, cs, c->tile, c->ns,
                inc ? " vs incumbent " : " (no incumbent) ", ins,
                best >= 0 ? "zts" : "incumbent holds");
    }
    if (best >= 0)
    {
        win = vfft_zts_create_chain(N, finals[best].chain, finals[best].nf, lanes);
        if (win && !vfft_zts_set_tile(win, finals[best].tile))
        {
            vfft_zts_destroy(win);
            win = NULL;
        }
    }
out:
    vfft_aligned_free(ire);
    vfft_aligned_free(iim);
    vfft_aligned_free(ore);
    vfft_aligned_free(oim);
    vfft_aligned_free(scr_a);
    vfft_aligned_free(scr_b);
    return win;
}

/* ── THE BLOCK RACE of a lane-major plan (2026-11-05): the winner of the
 * chain race at its tile, run at every legal block of the ladder (1, 2, 4, 8,
 * 16 lane groups per blocked ingest/terminator pass), same protocol as the
 * chain race, on the plan's OWN planes (sized for the ladder's top first).
 * The blocked form pays off where the lane-major plane set exceeds L2 (the
 * TLB law of the column pass); below it the race keeps blk = 1. Sets the
 * plan's blk to the winner, returns it (1 when nothing could run); *win_ns
 * = the winner's aggregate. */
static const int VFFT_ZTS_BLK_LADDER[] = { 1, 2, 4, 8, 16 };
#define VFFT_ZTS_BLK_LADDER_N 5
typedef struct
{
    vfft_zts_plan_t *p;
    int blk;
    const double *ire, *iim;
    double *ore, *oim;
} _zts_blk_arm_t;
static void _zts_blk_arm_run(void *v)
{
    _zts_blk_arm_t *a = (_zts_blk_arm_t *)v;
    a->p->blk = a->blk;
    vfft_zts_execute_fwd(a->p, a->ire, a->iim, a->ore, a->oim, 1);
}
static int vfft_zts_race_blk(vfft_zts_plan_t *p, int inplace, double *win_ns)
{
    const int N = p->N, lanes = p->lanes;
    const size_t n = (size_t)N * (size_t)lanes;
    int ladder[VFFT_ZTS_BLK_LADDER_N], nl = 0, i, best = -1, top = 1;
    double *ire, *iim, *ore = NULL, *oim = NULL;
    _zts_blk_arm_t ac[VFFT_ZTS_BLK_LADDER_N];
    vfft_race_arm_t arms[VFFT_ZTS_BLK_LADDER_N];
    double ns[VFFT_ZTS_BLK_LADDER_N];
    _zts_reseed_t rs;
    if (win_ns) *win_ns = 0.0;
    if (lanes <= 1) return 1;
    for (i = 0; i < VFFT_ZTS_BLK_LADDER_N; i++)
        if (vfft_zts_blk_legal(lanes, VFFT_ZTS_BLK_LADDER[i]))
        {
            ladder[nl++] = VFFT_ZTS_BLK_LADDER[i];
            top = VFFT_ZTS_BLK_LADDER[i];
        }
    if (nl <= 1) { p->blk = 1; return 1; }
    if (!vfft_zts_set_blk(p, top)) { p->blk = 1; return 1; }   /* the planes hold the ladder's top */
    ire = (double *)vfft_aligned_alloc(n * sizeof(double));
    iim = (double *)vfft_aligned_alloc(n * sizeof(double));
    if (!inplace)
    {
        ore = (double *)vfft_aligned_alloc(n * sizeof(double));
        oim = (double *)vfft_aligned_alloc(n * sizeof(double));
    }
    if (!ire || !iim || (!inplace && (!ore || !oim)))
    {
        vfft_aligned_free(ire); vfft_aligned_free(iim); vfft_aligned_free(ore); vfft_aligned_free(oim);
        p->blk = 1;
        return 1;
    }
    memset(&rs, 0, sizeof rs);
    rs.n = n;
    rs.b[rs.nb++] = ire;
    rs.b[rs.nb++] = iim;
    for (i = 0; i < nl; i++)
    {
        ac[i].p = p;
        ac[i].blk = ladder[i];
        ac[i].ire = ire;
        ac[i].iim = iim;
        ac[i].ore = inplace ? ire : ore;
        ac[i].oim = inplace ? iim : oim;
        arms[i].name = "blk";
        arms[i].run = _zts_blk_arm_run;
        arms[i].ctx = &ac[i];
    }
    {
        vfft_race_proto_t proto;
        memset(&proto, 0, sizeof proto);
        proto.rounds = VFFT_ZTS_RACE_ROUNDS;
        proto.reps = _zts_race_reps(N);
        proto.agg = VFFT_RACE_MEDIAN;
        proto.alternate = 1;
        proto.warm = 1;
        proto.reset = _zts_reseed;
        proto.reset_ctx = &rs;
        proto.pace = 1;
        if (vfft_race_run(&proto, arms, nl, ns) >= 0)
            for (i = 0; i < nl; i++)
                if (best < 0 || ns[i] < ns[best]) best = i;
    }
    vfft_aligned_free(ire); vfft_aligned_free(iim); vfft_aligned_free(ore); vfft_aligned_free(oim);
    if (best < 0) { p->blk = 1; return 1; }
    /* blk = 1 keeps the cell unless a block beats it by the house hysteresis */
    if (ladder[best] != 1 && !vfft_race_beats(ns[best], ns[0], VFFT_RACE_HYST)) best = 0;
    p->blk = ladder[best];
    if (win_ns) *win_ns = ns[best];
    if (getenv("VFFT_NAT_LOG"))
    {
        char line[256];
        int off = 0;
        for (i = 0; i < nl; i++) off += snprintf(line + off, sizeof line - (size_t)off, "%s%d:%.0f", i ? " " : "", ladder[i], ns[i]);
        fprintf(stderr, "[zts-blk] N=%d lanes=%d %s: %s -> blk=%d\n", N, lanes, inplace ? "ip" : "oop", line, p->blk);
    }
    return p->blk;
}

#endif /* VFFT_ZTS_RACE_H */
