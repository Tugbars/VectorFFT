/* ztt.h — ZTURN-T: the RUN-CONTIGUOUS DIT engine for K=1 interleaved c2c,
 * 16 <= N <= 262144, route VFFT_K1_IL_ZTT (docs/design/zturn_t_ship_plan.md;
 * the probe's contract: docs/research/sub2048_mkl_method/campaign_state/
 * probes/ZT/CONTRACT.md).
 *
 *   N = R[0] * ... * R[nf-1], R in {4, 8}, (N / R[0]) % 4 == 0.
 *   ingest  t0tp : natural packed z legs at stride N/R0 -> the PLANE, one
 *                  run of R0 per column at run rb[c] (64-B [re x4][im x4]
 *                  blocks); twiddle-free.
 *   mids    tmg  : combine R adjacent runs of length L into ONE run of R*L
 *                  IN PLACE with the column-varying pre-twiddle w_{RL}^{r*b};
 *                  ONE stream per stage serves every group.
 *   last    tlf  : the same combine, REINT packed store: NATURAL output, no
 *                  reordering pass anywhere.
 *   inverse      : the same pipeline with conjugate roots (the bwd kinds +
 *                  the s-negated streams); unnormalised, roundtrip = N * x.
 *
 * IN PLACE (2026-09-09, zturn_t_ship_plan.md 9): the `plane` drivers end in
 * tlfi, the in-place terminator — tlf with its output streams prefetched. The
 * caller's buffer goes cold under the plane, and a store into it waited on
 * its line fill (+17..25% over out of place); the prefetch issues those
 * fills ahead. Same arithmetic: in place stays bitwise the dest result.
 *
 * EXECUTION is one indirect call: the plan binds ONE FUSED DRIVER per
 * direction (generated/ztt_drivers_avx2.c — the three kind bodies inlined
 * with literal trip counts and the twiddle cursor carried in a register, zero
 * calls inside; the form that measured 9-12% over per-stage calls at N=128,
 * cascade_stage_fusion.md). Two buffer modes: `dest` runs the whole pipeline
 * in the destination (out of place), `plane` in the plan's scratch (in place,
 * or zin == zout at the call). Bound at create by placement; the execute
 * keeps one predictable compare so an aliased call on an out-of-place plan
 * is still correct.
 *
 * TILING (docs/design/zturn_t_2048plus_plan.md step 2): the mid stages whose
 * run length R*L <= tile run PER TILE — a contiguous plane span of `tile`
 * complexes holds WHOLE groups of every such stage, so the tile stays L1-hot
 * across those stages — the rest sweep the plane; the ingest is untiled (a
 * per-tile ingest would re-read every input line once per tile). The tile
 * changes group ORDER only: every width is bitwise the untiled result. It is
 * a RACED plan parameter (the planner's ladder, banked as il_tw=), never a
 * rule; vfft_ztt_tile_legal is the one law for it.
 *
 * CREATE builds (almost) no trig: up to the table's octave (RL <= 16384)
 * every stream is expanded from the baked quarter-wave (ztt_qw16384.h) by
 * index shift + reflection + sign flip. Above it (S4, zcascade_sunset_plan.md,
 * 2026-09-09) a stage's record is the TWO-LEVEL product: the angle
 * 2*pi*pw/RL splits as pw = a*2^u + b (u = log2 RL - 14, b < 2^u <= 16 at
 * 262144), w = table(a) * fine(b) with fine(b) = exp(-2*pi*i*b/RL) — one
 * 2^u-entry cos/sin table per stage at create, one complex product per
 * record, ~1 ulp. The ceiling is the cascade's, 262144. The validator is the law: an illegal
 * chain, a size out of range, or a cell without a registry driver returns
 * NULL loudly — no fallback, no default chain (the planner is the only
 * source of a chain). */
#ifndef VFFT_ZTT_H
#define VFFT_ZTT_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>              /* the fine table above the octave (S4, 2026-09-09)  */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "ztt_registry_avx2.h"   /* the cells and their four drivers (generated) */
#include "ztt_qw16384.h"         /* the baked quarter-wave (generated)            */

#define VFFT_ZTT_MAX_NF 7        /* == the registry's chain[7]; VFFT_ZSPLIT_MAX_NF */
#define VFFT_ZTT_MAX_N 262144    /* the cascade's ceiling; above the octave the
                                  * streams are the TWO-LEVEL product (below)     */
#define VFFT_ZTT_QW_M 16384
#define VFFT_ZTT_QW_LGM 14       /* log2 M */
#define VFFT_ZTT_QW_Q 4096       /* M / 4 */
#define VFFT_ZTT_QW_LGQ 12

#if defined(_WIN32)
#include <malloc.h>
#define VFFT_ZTT_ALLOC(sz) _aligned_malloc((sz), 64)
#define VFFT_ZTT_FREE(p) _aligned_free(p)
#else
#define VFFT_ZTT_ALLOC(sz) aligned_alloc(64, ((((sz)) + 63u) / 64u) * 64u)
#define VFFT_ZTT_FREE(p) free(p)
#endif

typedef struct
{
    int N, nf;
    int chain[VFFT_ZTT_MAX_NF];
    long ncol;                        /* N / chain[0]: ingest columns = rb entries   */
    long L[VFFT_ZTT_MAX_NF];          /* run length BEFORE stage s (s >= 1)          */
    long Gs[VFFT_ZTT_MAX_NF];         /* group count of stage s                      */
    double *tw, *twb;                 /* ONE contiguous stream each, stage order:
                                       * fwd, and bwd (the s half negated)          */
    size_t twdoubles;
    size_t *rb;                       /* the run-base table, ncol entries            */
    double *plane;                    /* 2N doubles + 4 KB of slack, 64-B: the `plane`
                                       * mode scratch; each in-place call starts it 2 KB
                                       * (mod 4 KB) from the caller's buffer (_ztt_plane_for) */
    const vfft_ztt_cell_t *cell;      /* the registry row: the four drivers          */
    vfft_ztt_fn fwd, bwd;             /* BOUND by placement (vfft_ztt_bind)          */
    int inplace;
    size_t tile;                      /* TILE WIDTH in complexes, 0 = untiled (raced,
                                       * il_tw=; vfft_ztt_set_tile)                  */
} vfft_ztt_plan_t;

/* the registry row for (N, chain), NULL when the cell was not emitted */
static inline const vfft_ztt_cell_t *vfft_ztt_lookup(int N, const int *chain, int nf)
{
    int i, s;
    for (i = 0; i < VFFT_ZTT_NCELLS_AVX2; i++)
    {
        const vfft_ztt_cell_t *c = &vfft_ztt_cells_avx2[i];
        if (c->n != N || c->nf != nf) continue;
        for (s = 0; s < nf; s++)
            if (c->chain[s] != chain[s]) break;
        if (s == nf) return c;
    }
    return NULL;
}

static inline void vfft_ztt_destroy(vfft_ztt_plan_t *p)
{
    if (!p) return;
    VFFT_ZTT_FREE(p->tw);
    VFFT_ZTT_FREE(p->twb);
    VFFT_ZTT_FREE(p->rb);
    VFFT_ZTT_FREE(p->plane);
    free(p);
}

/* mixed-radix digit reversal over R[0..nR-1]: n = d0 + R0*(d1 + ...), d0
 * fastest; digit d_s carries weight prod_{u>s} R[u] (CONTRACT.md 5) */
static inline long _ztt_digitrev(long n, const int *R, int nR)
{
    long out = 0;
    int s, u;
    for (s = 0; s < nR; s++)
    {
        const int d = (int)(n % R[s]);
        long wt = 1;
        n /= R[s];
        for (u = s + 1; u < nR; u++) wt *= R[u];
        out += (long)d * wt;
    }
    return out;
}

/* sin(2*pi*u/M) from the quarter wave, u in [0, M): quadrant q = u / Q,
 * rem = u % Q: q=0 T[rem], q=1 T[Q-rem] (mirror about pi/2), q>=2 negated
 * (a sign-bit flip, so the -0.0 the contract requires at r*b == 0 comes out
 * bit for bit) */
static inline double _ztt_qsin(long u)
{
    const long q = u >> VFFT_ZTT_QW_LGQ;
    const long rem = u & (VFFT_ZTT_QW_Q - 1);
    const double v = (q & 1) ? VFFT_ZTT_QW16384_SIN[VFFT_ZTT_QW_Q - rem]
                             : VFFT_ZTT_QW16384_SIN[rem];
    return (q & 2) ? -v : v;
}

static inline int _ztt_log2(long v)
{
    int k = 0;
    while ((1L << k) < v) k++;
    return k;
}

/* one stage's stream (CONTRACT.md 4): record for (column quad k, leg r>=1)
 * at tw + ((k/4)*(R-1) + (r-1))*8 doubles = [c(k..k+3)][s(k..k+3)] of
 * w_{RL}^{r*b}, b = k + lane; fwd s = -sin (times c + i s = exp(-i th)),
 * bwd s = +sin. Returns the doubles written = 2*(R-1)*L. */
static inline size_t _ztt_fill_stage(double *tw, long L, int R, long RL, int bwd)
{
    const int lg = _ztt_log2(RL);
    /* up = the bits of pw ABOVE the table's octave (0 within it): the fine
     * table has 2^up entries, cos/sin of 2*pi*b/RL, b < 2^up (<= 16 at the
     * ceiling) — the only trig the create ever runs */
    const int up = lg > VFFT_ZTT_QW_LGM ? lg - VFFT_ZTT_QW_LGM : 0;
    const int sh = up ? 0 : VFFT_ZTT_QW_LGM - lg;    /* M = 2^LGM */
    double fc[64], fs[64];
    long k, bf;
    int r, lane;
    if (up)
        for (bf = 0; bf < (1L << up); bf++)
        {
            const double a = 2.0 * M_PI * (double)bf / (double)RL;
            fc[bf] = cos(a);
            fs[bf] = sin(a);
        }
    for (k = 0; k < L; k += 4)
        for (r = 1; r < R; r++)
        {
            double *rec = tw + ((size_t)(k / 4) * (size_t)(R - 1) + (size_t)(r - 1)) * 8;
            for (lane = 0; lane < 4; lane++)
            {
                const long b = k + lane;
                const long pw = ((long)r * b) % RL;
                double c, s;
                if (!up)
                {
                    const long idx = pw << sh;
                    s = _ztt_qsin(idx);
                    c = _ztt_qsin((idx + VFFT_ZTT_QW_Q) & (VFFT_ZTT_QW_M - 1));
                }
                else
                {   /* pw = a*2^up + bb: w = table(a) * fine(bb) */
                    const long a = pw >> up, bb = pw & ((1L << up) - 1);
                    const double sa = _ztt_qsin(a);
                    const double ca = _ztt_qsin((a + VFFT_ZTT_QW_Q) & (VFFT_ZTT_QW_M - 1));
                    c = ca * fc[bb] - sa * fs[bb];
                    s = sa * fc[bb] + ca * fs[bb];
                }
                rec[lane] = c;
                rec[4 + lane] = bwd ? s : -s;
            }
        }
    return (size_t)2 * (size_t)(R - 1) * (size_t)L;
}

/* create: the validator is the law (NULL = not a ZTURN-T cell, loudly under
 * VFFT_NAT_LOG). Bound out of place; vfft_ztt_bind(p, 1) rebinds in place. */
static inline vfft_ztt_plan_t *vfft_ztt_create_chain(int N, const int *chain, int nf)
{
    vfft_ztt_plan_t *p;
    const vfft_ztt_cell_t *cell;
    long prod = 1, j;
    int s;
    size_t off;
    const char *why = NULL;
    if (nf < 2 || nf > VFFT_ZTT_MAX_NF) why = "nf outside 2..7";
    else if (N < 16 || N > VFFT_ZTT_MAX_N) why = "N outside 16..262144";
    for (s = 0; !why && s < nf; s++)
    {
        if (chain[s] != 4 && chain[s] != 8) why = "radix not in {4, 8}";
        prod *= chain[s];
    }
    if (!why && prod != (long)N) why = "chain product != N";
    if (!why && (N / chain[0]) % 4) why = "(N / R0) % 4 != 0";
    cell = why ? NULL : vfft_ztt_lookup(N, chain, nf);
    if (!why && !cell) why = "no fused driver for this cell (ztt_registry_avx2.h)";
    if (why)
    {
        if (getenv("VFFT_NAT_LOG"))
            fprintf(stderr, "[ztt] N=%d: refused: %s\n", N, why);
        return NULL;
    }
    p = (vfft_ztt_plan_t *)calloc(1, sizeof *p);
    if (!p) return NULL;
    p->N = N;
    p->nf = nf;
    for (s = 0; s < nf; s++) p->chain[s] = chain[s];
    p->cell = cell;
    p->ncol = N / chain[0];
    /* geometry: L[1] = R0, L[s+1] = L[s]*R[s], Gs[s] = N / (R[s]*L[s]) */
    p->L[1] = chain[0];
    p->twdoubles = 0;
    for (s = 1; s < nf; s++)
    {
        const long RL = p->L[s] * chain[s];
        p->Gs[s] = N / RL;
        p->twdoubles += (size_t)2 * (size_t)(chain[s] - 1) * (size_t)p->L[s];
        if (s + 1 < nf) p->L[s + 1] = RL;
    }
    p->tw = (double *)VFFT_ZTT_ALLOC(p->twdoubles * sizeof(double));
    p->twb = (double *)VFFT_ZTT_ALLOC(p->twdoubles * sizeof(double));
    p->rb = (size_t *)VFFT_ZTT_ALLOC((size_t)p->ncol * sizeof(size_t));
    p->plane = (double *)VFFT_ZTT_ALLOC((size_t)2 * (size_t)N * sizeof(double) + 4096u);
    if (!p->tw || !p->twb || !p->rb || !p->plane) { vfft_ztt_destroy(p); return NULL; }
    /* the streams, stage order, ONE allocation each (the fused driver's
     * carried cursor walks straight from one stage's end into the next) */
    for (s = 1, off = 0; s < nf; s++)
    {
        const size_t n = _ztt_fill_stage(p->tw + off, p->L[s], chain[s], p->L[s] * chain[s], 0);
        (void)_ztt_fill_stage(p->twb + off, p->L[s], chain[s], p->L[s] * chain[s], 1);
        off += n;
    }
    /* the run-base table: rb[c] = the run j whose digit reversal (over the
     * middle+last radices) is column c; refused unless a bijection */
    for (j = 0; j < p->ncol; j++) p->rb[j] = (size_t)-1;
    for (j = 0; j < p->ncol; j++)
    {
        const long c = _ztt_digitrev(j, chain + 1, nf - 1);
        if (c < 0 || c >= p->ncol || p->rb[c] != (size_t)-1)
        {
            if (getenv("VFFT_NAT_LOG"))
                fprintf(stderr, "[ztt] N=%d: run-base table is not a bijection (c=%ld)\n", N, c);
            vfft_ztt_destroy(p);
            return NULL;
        }
        p->rb[c] = (size_t)j;
    }
    p->inplace = 0;
    p->tile = 0;
    p->fwd = cell->fwd_dest;
    p->bwd = cell->bwd_dest;
    return p;
}

/* placement binding: out of place = the `dest` drivers (the pipeline runs
 * in zout), in place = the `plane` drivers (zin is consumed by the ingest
 * before the last stage writes zout, so zin == zout is legal) */
static inline void vfft_ztt_bind(vfft_ztt_plan_t *p, int inplace)
{
    p->inplace = inplace ? 1 : 0;
    p->fwd = inplace ? p->cell->fwd_plane : p->cell->fwd_dest;
    p->bwd = inplace ? p->cell->bwd_plane : p->cell->bwd_dest;
}

/* the tile law: 0 (untiled) is always legal; else a power of two, at least
 * the first mid's R*L (below it no stage tiles) and below N; a chain without
 * mids (nf == 2) tiles nothing and refuses every width. Shared by the create,
 * the planner's ladder and the gate. */
static inline int vfft_ztt_tile_legal(int N, const int *chain, int nf, size_t tile)
{
    if (tile == 0) return 1;
    if (nf < 3) return 0;
    if (tile & (tile - 1)) return 0;
    if (tile < (size_t)chain[0] * (size_t)chain[1]) return 0;
    return tile < (size_t)N;
}

static inline int vfft_ztt_set_tile(vfft_ztt_plan_t *p, size_t tile)
{
    if (!vfft_ztt_tile_legal(p->N, p->chain, p->nf, tile)) return 0;
    p->tile = tile;
    return 1;
}

/* the plane's PAGE OFFSET against the caller's buffer (measured 2026-09-09,
 * probes/ZT/zt_plane_skew.c, and again with tlfi in the tree): with the plane
 * just below zout in its 4 KB slot, the terminator's stores to zout sit in the
 * same 4K slot as its loads from the plane a few column quads ahead — a
 * 4K-alias stall worth +20..30% in place at 4096..16384, SEPARATE from the
 * store-miss latency tlfi's prefetch removes (both are needed). The plane
 * carries 4 KB of slack and each in-place call starts it 2 KB (mod 4 KB) from
 * zout, 64-B aligned: the alias distance becomes 128 columns. Arithmetic is
 * untouched (the gate's in-place bitwise law holds). */
static inline double *_ztt_plane_for(const vfft_ztt_plan_t *p, const double *zout)
{
    const uintptr_t base = (uintptr_t)p->plane;
    const uintptr_t want = ((uintptr_t)zout + 2048u) & 4095u;
    const uintptr_t off = ((want - (base & 4095u)) & 4095u) & ~(uintptr_t)63u;
    return (double *)(base + off);
}

static inline void vfft_ztt_execute_fwd(const vfft_ztt_plan_t *p,
                                        const double *zin, double *zout)
{
    if (zin == zout || p->inplace)
        p->cell->fwd_plane(zin, zout, _ztt_plane_for(p, zout), p->tw, p->rb, p->tile);
    else
        p->fwd(zin, zout, p->plane, p->tw, p->rb, p->tile);
}

static inline void vfft_ztt_execute_bwd(const vfft_ztt_plan_t *p,
                                        const double *zin, double *zout)
{
    if (zin == zout || p->inplace)
        p->cell->bwd_plane(zin, zout, _ztt_plane_for(p, zout), p->twb, p->rb, p->tile);
    else
        p->bwd(zin, zout, p->plane, p->twb, p->rb, p->tile);
}

/* "4.4.8" for logs and the wisdom token il_ztt= */
static inline int vfft_ztt_chain_str(const vfft_ztt_plan_t *p, char *buf, size_t n)
{
    int s, off = 0;
    for (s = 0; s < p->nf && (size_t)off < n; s++)
        off += snprintf(buf + off, n - (size_t)off, "%s%d", s ? "." : "", p->chain[s]);
    return off;
}

#endif /* VFFT_ZTT_H */
