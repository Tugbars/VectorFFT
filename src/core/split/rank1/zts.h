/* zts.h — zturn-ts: ZTURN-T's RUN-CONTIGUOUS DIT engine on the SPLIT layout,
 * 1D c2c, natural order, 16 <= N <= 262144, N = R[0] * ... * R[nf-1] with
 * every R in {4, 8}, nf >= 2, (N / R[0]) % 4 == 0 (the natural-class cells of
 * ZTURN-T; the registry is derived from the same list).
 *
 * Two lane choices, one plan type:
 *   lanes = 1  the K=1 engine (fused codelets zts_*): the caller's re[] / im[]
 *              at the EDGES, ZTURN-T's 64-B [re x4][im x4] block plane INSIDE
 *              (the plan's scratch). ingest = split legs at stride N/R0,
 *              radix-R0, a TR4 per plane per quartet into block runs through
 *              rb[]; mids = the block combine; last = block loads, natural
 *              split-plane stores, no shuffle. K contiguous transforms
 *              (transform t at offset t*N) run transform-major through the
 *              _batch twin on the one scratch plane.
 *   lanes = 4  the lane-major engine (fused codelets ztv_*): data[e*4 + t],
 *              element e of transform t; every column is ONE vector per plane
 *              (lane = transform), no turn anywhere, twiddles broadcast from
 *              (c, s) pairs; two scratch planes of 4N doubles.
 * Both run the pipeline IN THE SCRATCH and the terminator writes the caller's
 * planes: in place when in == out. There is no destination mode for split
 * (two caller planes cannot host the block interior; a terminator writing the
 * split output over the blocks it reads would clobber unconsumed data).
 *
 * TWIDDLES come from the baked quarter-wave (common/math/qw_fill.h): the quad
 * records for lanes = 1, the pairs for lanes = 4; ONE contiguous stream each
 * way (fwd / bwd), stage order, the fused driver's cursor walks it.
 * rb[] = ZTURN-T's run-base table. TILE = ZTURN-T's law (vfft_zts_tile_legal):
 * 0 = untiled; a width that divides N, that the first mid's group R0*R1
 * divides, below N — the mids with R*L <= tile run per tile.
 *
 * PLANE PLACEMENT (measured 2026-11-04): the block plane starts, per call, at
 * the 512-B slot (mod 4 KB) farthest from all four caller pointers, so the
 * terminator's block loads and split stores, and the ingest's split loads and
 * block stores, never share an L1 set (a congruent plane cost up to 1.7x on a
 * stage); the lane-major planes sit 1 KB and 3 KB (mod 4 KB) from their
 * allocation, never congruent with each other.
 *
 * The create is the validator: an illegal chain, a size out of range, a lane
 * count the registry does not serve, or a cell without a driver returns NULL
 * loudly (VFFT_NAT_LOG) — no fallback, no default chain (the planner is the
 * only source of a chain). Serial (T=1) only. */
#ifndef VFFT_ZTS_H
#define VFFT_ZTS_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "build_isa.h"                /* VFFT_BUILD_ISA_AVX512: the build's ISA     */
#include "common/math/qw_fill.h"      /* the streams from the baked quarter-wave   */
#include "common/support/zalloc.h"    /* vfft_aligned_alloc / vfft_aligned_free    */

#if !defined(VFFT_BUILD_ISA_AVX512)
#include "zts_registry_avx2.h"        /* the cells and their fused codelets (generated) */
#define VFFT_ZTS_NCELLS VFFT_ZTS_NCELLS_AVX2
#define vfft_zts_cells  vfft_zts_cells_avx2
#define VFFT_ZTS_VW 4
#else
/* zturn-ts is emitted at avx2 only so far (2026-11-04): no cells at avx512 */
typedef void (*vfft_zts_fn)(const double *, const double *, double *, double *, double *, const double *, const size_t *, size_t);
typedef void (*vfft_zts_bfn)(const double *, const double *, double *, double *, double *, const double *, const size_t *, size_t, size_t);
typedef void (*vfft_ztv_fn)(const double *, const double *, double *, double *, double *, double *, const double *, const size_t *, size_t);
typedef struct { int n, nf; int chain[7]; vfft_zts_fn s_fwd, s_bwd; vfft_zts_bfn s_fwd_batch, s_bwd_batch; vfft_ztv_fn v_fwd, v_bwd; } vfft_zts_cell_t;
#define VFFT_ZTS_NCELLS 0
static const vfft_zts_cell_t vfft_zts_cells[1] = { { 0, 0, { 0 }, 0, 0, 0, 0, 0, 0 } };
#define VFFT_ZTS_VW 8
#endif

#define VFFT_ZTS_MAX_NF 7
#define VFFT_ZTS_MAX_N 262144
#define VFFT_ZTS_SLACK 8192u          /* bytes of slack behind each scratch plane */

typedef struct
{
    int N, nf;
    int chain[VFFT_ZTS_MAX_NF];
    int lanes;                        /* 1 = the K=1 engine, VFFT_ZTS_VW = lane-major */
    long ncol;                        /* N / chain[0]: ingest columns = rb entries    */
    long L[VFFT_ZTS_MAX_NF];          /* run length BEFORE stage s (s >= 1)           */
    long Gs[VFFT_ZTS_MAX_NF];         /* group count of stage s                       */
    double *tw, *twb;                 /* ONE contiguous stream each, stage order       */
    size_t twdoubles;
    size_t twoff[VFFT_ZTS_MAX_NF];
    size_t *rb;                       /* the run-base table, ncol entries              */
    double *plane;                    /* lanes = 1: the block plane, 2N doubles + slack */
    double *plane_re, *plane_im;      /* lanes = VW: the two lane-batched planes       */
    const vfft_zts_cell_t *cell;      /* the registry row                              */
    size_t tile;
} vfft_zts_plan_t;

static inline const vfft_zts_cell_t *vfft_zts_lookup(int N, const int *chain, int nf)
{
    int i, s;
    for (i = 0; i < VFFT_ZTS_NCELLS; i++)
    {
        const vfft_zts_cell_t *c = &vfft_zts_cells[i];
        if (c->n != N || c->nf != nf) continue;
        for (s = 0; s < nf; s++)
            if (c->chain[s] != chain[s]) break;
        if (s == nf) return c;
    }
    return NULL;
}

static inline void vfft_zts_destroy(vfft_zts_plan_t *p)
{
    if (!p) return;
    vfft_aligned_free(p->tw);
    vfft_aligned_free(p->twb);
    vfft_aligned_free(p->rb);
    vfft_aligned_free(p->plane);
    vfft_aligned_free(p->plane_re);
    vfft_aligned_free(p->plane_im);
    free(p);
}

/* the tile law, ZTURN-T's: 0 always legal; else a width that divides N, that
 * the first mid's group R0*R1 divides, below N; nf == 2 tiles nothing */
static inline int vfft_zts_tile_legal(int N, const int *chain, int nf, size_t tile)
{
    if (tile == 0) return 1;
    if (nf < 3) return 0;
    if ((size_t)N % tile) return 0;
    if (tile % ((size_t)chain[0] * (size_t)chain[1])) return 0;
    return tile < (size_t)N;
}

static inline int vfft_zts_set_tile(vfft_zts_plan_t *p, size_t tile)
{
    if (!vfft_zts_tile_legal(p->N, p->chain, p->nf, tile)) return 0;
    p->tile = tile;
    return 1;
}

/* create: lanes = 1 (K=1 / contiguous K) or VFFT_ZTS_VW (lane-major). NULL = not
 * a zturn-ts cell, loudly under VFFT_NAT_LOG. */
static inline vfft_zts_plan_t *vfft_zts_create_chain(int N, const int *chain, int nf, int lanes)
{
    vfft_zts_plan_t *p;
    const vfft_zts_cell_t *cell;
    long prod = 1, j;
    int s;
    size_t off;
    const char *why = NULL;
    if (nf < 2 || nf > VFFT_ZTS_MAX_NF) why = "nf outside 2..7";
    else if (N < 16 || N > VFFT_ZTS_MAX_N) why = "N outside 16..262144";
    else if (lanes != 1 && lanes != VFFT_ZTS_VW) why = "lanes is neither 1 nor the vector width";
    for (s = 0; !why && s < nf; s++)
    {
        if (chain[s] != 4 && chain[s] != 8) why = "radix not in {4, 8}";
        prod *= chain[s];
    }
    if (!why && prod != (long)N) why = "chain product != N";
    if (!why && (N / chain[0]) % VFFT_ZTS_VW) why = "(N / R0) % VW != 0";
    if (!why && chain[0] % VFFT_ZTS_VW) why = "R0 % VW != 0 (the ingest writes whole blocks)";
    cell = why ? NULL : vfft_zts_lookup(N, chain, nf);
    if (!why && !cell) why = "no fused driver for this cell (zts_registry)";
    if (why)
    {
        if (getenv("VFFT_NAT_LOG"))
            fprintf(stderr, "[zts] N=%d lanes=%d: refused: %s\n", N, lanes, why);
        return NULL;
    }
    p = (vfft_zts_plan_t *)calloc(1, sizeof *p);
    if (!p) return NULL;
    p->N = N;
    p->nf = nf;
    p->lanes = lanes;
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
    p->tw = (double *)vfft_aligned_alloc(p->twdoubles * sizeof(double));
    p->twb = (double *)vfft_aligned_alloc(p->twdoubles * sizeof(double));
    p->rb = (size_t *)vfft_aligned_alloc((size_t)p->ncol * sizeof(size_t));
    if (lanes == 1)
        p->plane = (double *)vfft_aligned_alloc((size_t)2 * (size_t)N * sizeof(double) + VFFT_ZTS_SLACK);
    else
    {
        p->plane_re = (double *)vfft_aligned_alloc((size_t)N * (size_t)lanes * sizeof(double) + VFFT_ZTS_SLACK);
        p->plane_im = (double *)vfft_aligned_alloc((size_t)N * (size_t)lanes * sizeof(double) + VFFT_ZTS_SLACK);
    }
    if (!p->tw || !p->twb || !p->rb || (lanes == 1 ? !p->plane : (!p->plane_re || !p->plane_im)))
    {
        vfft_zts_destroy(p);
        return NULL;
    }
    /* the streams, stage order, ONE allocation each */
    for (s = 1, off = 0; s < nf; s++)
    {
        const long RL = p->L[s] * chain[s];
        size_t n;
        if (lanes == 1)
        {
            n = vfft_qw_fill_stage_quad(p->tw + off, p->L[s], chain[s], RL, 0, VFFT_ZTS_VW);
            (void)vfft_qw_fill_stage_quad(p->twb + off, p->L[s], chain[s], RL, 1, VFFT_ZTS_VW);
        }
        else
        {
            n = vfft_qw_fill_stage_pair(p->tw + off, p->L[s], chain[s], RL, 0);
            (void)vfft_qw_fill_stage_pair(p->twb + off, p->L[s], chain[s], RL, 1);
        }
        p->twoff[s] = off;
        off += n;
    }
    /* the run-base table: rb[c] = the run j whose digit reversal (over the
     * middle+last radices) is column c; refused unless a bijection */
    for (j = 0; j < p->ncol; j++) p->rb[j] = (size_t)-1;
    for (j = 0; j < p->ncol; j++)
    {
        const long c = vfft_qw_digitrev(j, chain + 1, nf - 1);
        if (c < 0 || c >= p->ncol || p->rb[c] != (size_t)-1)
        {
            if (getenv("VFFT_NAT_LOG"))
                fprintf(stderr, "[zts] N=%d: run-base table is not a bijection (c=%ld)\n", N, c);
            vfft_zts_destroy(p);
            return NULL;
        }
        p->rb[c] = (size_t)j;
    }
    p->tile = 0;
    return p;
}

/* the block plane's start for this call (lanes = 1): the 512-B slot mod 4 KB
 * farthest from all four caller pointers; the allocation carries the slack */
static inline double *_zts_plane_for(const vfft_zts_plan_t *p, const double *ire, const double *iim,
                                     const double *ore, const double *oim)
{
    const uintptr_t base = (uintptr_t)p->plane;
    const uintptr_t o[4] = { (uintptr_t)ire & 4095u, (uintptr_t)iim & 4095u, (uintptr_t)ore & 4095u, (uintptr_t)oim & 4095u };
    uintptr_t best = 2048u, bestm = 0, t;
    int i;
    for (t = 0; t < 4096u; t += 512u)
    {
        uintptr_t m = 4096u;
        for (i = 0; i < 4; i++)
        {
            const uintptr_t d = (t - o[i]) & 4095u;
            const uintptr_t md = d < 2048u ? d : 4096u - d;
            if (md < m) m = md;
        }
        if (m > bestm) { bestm = m; best = t; }
    }
    return (double *)(base + (((best - (base & 4095u)) & 4095u) & ~(uintptr_t)63u));
}

/* the lane-major planes: 1 KB and 3 KB (mod 4 KB) from their allocations */
static inline double *_zts_vplane(const double *alloc, unsigned want)
{
    const uintptr_t base = (uintptr_t)alloc;
    return (double *)(base + (((want - (base & 4095u)) & 4095u) & ~(uintptr_t)63u));
}

/* execute. lanes = 1: K contiguous transforms (K >= 1), transform t at offset
 * t*N in every plane; lanes = VW: one lane-major batch of VW transforms. In
 * place when in == out. */
static inline void _zts_run(const vfft_zts_plan_t *p, int bwd,
                            const double *ire, const double *iim, double *ore, double *oim, size_t K)
{
    const double *tw = bwd ? p->twb : p->tw;
    if (p->lanes == 1)
    {
        double *W = _zts_plane_for(p, ire, iim, ore, oim);
        if (K == 1)
            (bwd ? p->cell->s_bwd : p->cell->s_fwd)(ire, iim, ore, oim, W, tw, p->rb, p->tile);
        else
            (bwd ? p->cell->s_bwd_batch : p->cell->s_fwd_batch)(ire, iim, ore, oim, W, tw, p->rb, p->tile, K);
    }
    else
    {
        double *Wr = _zts_vplane(p->plane_re, 1024u), *Wi = _zts_vplane(p->plane_im, 3072u);
        (bwd ? p->cell->v_bwd : p->cell->v_fwd)(ire, iim, ore, oim, Wr, Wi, tw, p->rb, p->tile);
    }
}

static inline void vfft_zts_execute_fwd(const vfft_zts_plan_t *p, const double *ire, const double *iim,
                                        double *ore, double *oim, size_t K)
{
    _zts_run(p, 0, ire, iim, ore, oim, K);
}

static inline void vfft_zts_execute_bwd(const vfft_zts_plan_t *p, const double *ire, const double *iim,
                                        double *ore, double *oim, size_t K)
{
    _zts_run(p, 1, ire, iim, ore, oim, K);
}

/* "8.8.4.4.4" for logs and the wisdom token */
static inline int vfft_zts_chain_str(const vfft_zts_plan_t *p, char *buf, size_t n)
{
    int s, off = 0;
    for (s = 0; s < p->nf && (size_t)off < n; s++)
        off += snprintf(buf + off, n - (size_t)off, "%s%d", s ? "." : "", p->chain[s]);
    return off;
}

#endif /* VFFT_ZTS_H */
