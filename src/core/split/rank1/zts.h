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
 *   lanes = K  the lane-major engine (fused codelets ztv_*), K a multiple of
 *              the vector width: data[e*K + t], element e of transform t. The
 *              driver runs the pipeline per LANE GROUP of VW lanes (the edges
 *              at element pitch K, the group's lane offset on the pointers);
 *              every column is ONE vector per plane (lane = transform), no
 *              turn anywhere, twiddles broadcast from (c, s) pairs; two
 *              scratch planes of VW*N doubles (one group at a time).
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
 * only source of a chain: split/rank1/zts_race.h, the create-time race, and
 * the cell's wisdom row, split/wisdom/wisdom2_zts_split.h). Serial: the
 * engine has no threaded form; at K <= 4 the stride engine it replaces runs
 * its FFT serially too (the pool K-split needs K >= 8). */
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
typedef void (*vfft_ztv_fn)(const double *, const double *, double *, double *, double *, double *, const double *, const size_t *, size_t, size_t, size_t, size_t, size_t, size_t);   /* (…, tile, K, blk, gp, g0, g1): the registry's V driver ABI (blocked edges, 2026-11-05) */
typedef struct { int n, nf; int chain[7]; vfft_zts_fn s_fwd, s_bwd; vfft_zts_bfn s_fwd_batch, s_bwd_batch; vfft_ztv_fn v_fwd, v_bwd; } vfft_zts_cell_t;
#define VFFT_ZTS_NCELLS 0
static const vfft_zts_cell_t vfft_zts_cells[1] = { { 0, 0, { 0 }, 0, 0, 0, 0, 0, 0 } };
#define VFFT_ZTS_VW 8
#endif

#define VFFT_ZTS_MAX_NF 7
#define VFFT_ZTS_MAX_N 262144
#define VFFT_ZTS_MIN_N 16         /* the registry's smallest cell (c2c_create_split.h names the band) */
#define VFFT_ZTS_SLACK 8192u          /* bytes of slack behind each scratch plane */
#define VFFT_ZTS_MT_MAX 63            /* worker planes: THREAD_POOL_MAX_DISPATCH - 1 (the caller is slot 0) */
#define VFFT_ZTS_LANES_MAX 262144     /* the lane-major batch: K a multiple of VW up to this (the
                                       * group planes do not grow with K; a 2D column pass is K = N2) */

/* the tile ladder the race walks beside untiled (complex columns per tile):
 * ZTURN-T's 1024 / 2048 (16 KB / 32 KB of block plane) and, for the lane-major
 * engine whose planes are four times wider per column, 256 / 512. The tile
 * law filters each; the race decides. */
static const int VFFT_ZTS_TILE_LADDER[] = { 256, 512, 1024, 2048, 3072 };   /* 3072 (48 KB): the odd cells' width */
#define VFFT_ZTS_TILE_LADDER_N 5

#include "split/rank1/zts_stages.h"   /* the stage kernels: the staged executor's vocabulary */

/* THE ODD BAND on the split side (ztt_odd_design.md's grammar, 2026-11-06):
 *   N = 2^a * m,  a >= 4,  m > 1 a product over {3, 5, 7, 9, 15} with at most
 *   five odd mids (nf <= 7 with the two pow2 ends),  48 <= N <= the ceiling.
 * A cell of the band has no fused driver: the STAGED executor walks its chain's
 * stage kernels (zts_stages.h), one call per stage and block, in both families.
 * No floor at 2048 as on the interleaved side: the split side is the low-N
 * library. The chain grammar: the ends in {4, 8} (the ingest's turn lattice,
 * the terminator's lane transposes), a mid in {4, 8, 3, 5, 7, 9, 15}, the
 * product N, (N / R0) % VW == 0; the odd part decomposed greedily
 * largest-first (9 is a radix-9 mid, never 3.3; 45 is 15.3), its mids at
 * every interior position, the pow2 slots over ordered {4, 8}: the race's
 * pool (zts_race.h _zts_pool_odd). */
static const int VFFT_ZTS_ODD_MIDS[] = { 15, 9, 7, 5, 3 };
#define VFFT_ZTS_ODD_MIDS_N 5
static inline int vfft_zts_odd_band(int N)
{
    int m = N, a = 0, nm = 0, i;
    if (N < 48 || N > VFFT_ZTS_MAX_N) return 0;
    while ((m & 1) == 0) { m >>= 1; a++; }
    if (m == 1 || a < 4) return 0;
    for (i = 0; i < VFFT_ZTS_ODD_MIDS_N; i++)
        while (m % VFFT_ZTS_ODD_MIDS[i] == 0) { m /= VFFT_ZTS_ODD_MIDS[i]; nm++; }
    return m == 1 && nm <= VFFT_ZTS_MAX_NF - 2;
}
static inline int vfft_zts_mid_ok(int R)
{
    return R == 4 || R == 8 || R == 3 || R == 5 || R == 7 || R == 9 || R == 15;
}
/* THE LANE-MAJOR BAND (the split side's lane-major flat DIT, 2026-11-06): the
 * V family's kinds carry no lane lattice (one vector per element: the ingest
 * stores R0 vectors per column through rb[], the terminator loads any leg
 * count), so its ENDS take the odd radices too and a lane-major chain is any
 * 2..7 radices over {3, 4, 5, 7, 8, 9, 15} with product N: N = 2^a * m, m
 * smooth over the odd mids (greedy decomposition), the pow2 part in 4s and 8s
 * (a != 1), nf <= 7; N >= 9. A pure-odd N is served lane-major ONLY (the S
 * family's ends are the lattice). The chain pool: zts_race.h _zts_pool_odd. */
static inline int vfft_zts_v_band(int N)
{
    int m = N, a = 0, nm = 0, i, np;
    if (N < 9 || N > VFFT_ZTS_MAX_N) return 0;
    while ((m & 1) == 0) { m >>= 1; a++; }
    if (a == 1) return 0;
    for (i = 0; i < VFFT_ZTS_ODD_MIDS_N; i++)
        while (m % VFFT_ZTS_ODD_MIDS[i] == 0) { m /= VFFT_ZTS_ODD_MIDS[i]; nm++; }
    if (m != 1) return 0;
    np = a == 0 ? 0 : (a + 2) / 3;   /* the fewest pow2 slots: 8s then one 4 */
    if (nm + np < 2) return N == 9 || N == 15;   /* a lone 9 or 15 runs as 3.3 / 3.5 (the pool splits it) */
    return nm + np <= VFFT_ZTS_MAX_NF;
}

typedef struct vfft_zts_plan_s
{
    int N, nf;
    int chain[VFFT_ZTS_MAX_NF];
    int lanes;                        /* 1 = the K=1 engine; K (a multiple of VW) = lane-major:
                                       * the driver runs the pipeline per lane group of VW */
    long ncol;                        /* N / chain[0]: ingest columns = rb entries    */
    long L[VFFT_ZTS_MAX_NF];          /* run length BEFORE stage s (s >= 1)           */
    long Gs[VFFT_ZTS_MAX_NF];         /* group count of stage s                       */
    double *tw, *twb;                 /* ONE contiguous stream each, stage order; twb
                                       * NULL until vfft_zts_fill_bwd (the race builds
                                       * forward streams only)                        */
    size_t twdoubles;
    size_t twoff[VFFT_ZTS_MAX_NF];
    size_t *rb;                       /* the run-base table, ncol entries              */
    double *plane;                    /* lanes = 1: the block plane, 2N doubles + slack */
    double *plane_re, *plane_im;      /* lanes > 1: the two lane-group plane sets, blk
                                       * group planes each at pitch gp doubles           */
    double *tail_re, *tail_im;        /* lanes % VW != 0: the TAIL's scratch group, N x VW doubles
                                       * each at pitch VW (the tail lanes zero-padded to a group;
                                       * the full groups run on the caller's planes as before) */
    int blk;                          /* lanes > 1: lane groups per BLOCK of the blocked
                                       * ingest/terminator (one row visit reads blk*VW
                                       * lanes); 1 = one group at a time. A raced axis
                                       * of the lane-major cell (blk= on its row).      */
    size_t gp;                        /* the group plane pitch in doubles (N*VW + a set
                                       * stagger)                                        */
    int planes_owned;                 /* 0 = the race's shared scratch (never freed here) */
    const vfft_zts_cell_t *cell;      /* the registry row                              */
    size_t tile;
    /* THE THREADED FORM of a contiguous batch (lanes = 1, K > 1): the K
     * transforms sliced over the pool's slots, slot 0 the caller, each slot
     * running the _batch twin on its OWN block plane (the streams and rb are
     * read-only and shared) — the IL contiguous wrapper's shape, bitwise the
     * serial loop. mt = the verdict (1 = slabs; raced at create, banked
     * eng=tcb tcmt= at the plan's thread count, split/rank1/c2c_ip_create_split.h
     * _zts_tcmt_decide); mt_n = the worker planes built (0 = no arm). */
    int mt;                           /* lanes 1, K > 1: 1 = slabs; K = 1 (and the lane-major
                                       * single group): 1 = the sectioned walk's BLOCKS arm,
                                       * 2 = its TILES arm (zts_mt.h) */
    int mt_n;
    int mt_sect;                      /* 1 = mt names the SECTIONED walk (zts_mt.h), 0 = the
                                       * slab / lane-block form */
    int mt_grp;                       /* lanes > 1, the lane-range form: 1 = ranges of one cache LINE of
                                       * lanes (two groups on a 64-byte line), 0 = ranges of whole blocks */
    size_t mt_unit;                   /* lanes per range unit (set with the verdict; 0 = a whole block) */
    double *mtplane[VFFT_ZTS_MT_MAX];
    /* THE STAGED FORM (the odd band): no fused driver, the stage kernels
     * resolved at create (zts_stages.h), the executor _zts_staged_s / _v */
    int staged;
    int jit;                          /* 1 = cell is a driver compiled at create (jit/zts_jit_runtime.h) */
    vfft_zts_s0_fn s0f, s0b;
    vfft_zts_sm_fn smf[VFFT_ZTS_MAX_NF], smb[VFFT_ZTS_MAX_NF];
    vfft_zts_sl_fn slf, slb;
    vfft_zts_v0_fn v0f, v0b;
    vfft_zts_vm_fn vmf[VFFT_ZTS_MAX_NF], vmb[VFFT_ZTS_MAX_NF];
    vfft_zts_vl_fn vlf, vlb;
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

/* 1 = the registry has at least one fused cell at N */
static inline int vfft_zts_fused_at(int N)
{
    int i;
    for (i = 0; i < VFFT_ZTS_NCELLS; i++)
        if (vfft_zts_cells[i].n == N) return 1;
    return 0;
}
/* 1 = zturn-ts serves N in the S family (K = 1, the contiguous batch, the
 * rows of a plane): a fused cell, or the odd band's staged form */
static inline int vfft_zts_serves(int N)
{
    return vfft_zts_fused_at(N) || vfft_zts_odd_band(N);
}
/* 1 = the V family serves N (a lane-major batch, the column axes of a plane
 * or volume): a fused cell, or the lane-major band's staged form */
static inline int vfft_zts_serves_v(int N)
{
    return vfft_zts_fused_at(N) || vfft_zts_v_band(N);
}

static inline void vfft_zts_destroy(vfft_zts_plan_t *p)
{
    int i;
    if (!p) return;
    vfft_aligned_free(p->tw);
    vfft_aligned_free(p->twb);
    vfft_aligned_free(p->rb);
    if (p->planes_owned)
    {
        vfft_aligned_free(p->plane);
        vfft_aligned_free(p->plane_re);
        vfft_aligned_free(p->plane_im);
    }
    for (i = 0; i < p->mt_n; i++) vfft_aligned_free(p->mtplane[i]);
    vfft_aligned_free(p->tail_re);
    vfft_aligned_free(p->tail_im);
    free(p);
}

/* the worker planes of the threaded contiguous form (lanes = 1): n of them,
 * one per pool slot beyond the caller; returns the count built */
static inline size_t vfft_zts_plane_bytes(int N, int lanes, int blk); /* below */
static inline void _zts_staged_s(const vfft_zts_plan_t *p, int bwd, const double *ire, const double *iim, double *ore, double *oim, double *W); /* below */
static inline void _zts_staged_v(const vfft_zts_plan_t *p, int bwd, const double *ire, const double *iim, double *ore, double *oim, size_t g0, size_t g1, double *W0, double *Wi0, size_t pitch); /* below */
static inline int vfft_zts_mt_planes(vfft_zts_plan_t *p, int n)
{
    int i;
    /* lanes = 1: a block plane per worker; lane-major: a worker's re and im
     * plane sets back to back in ONE allocation (vfft_zts_run_lanes splits
     * it), sized for the plan's block -- set the block before the workers */
    const size_t bytes = vfft_zts_plane_bytes(p->N, p->lanes, p->blk);
    if (n > VFFT_ZTS_MT_MAX) n = VFFT_ZTS_MT_MAX;
    for (i = p->mt_n; i < n; i++)
    {
        p->mtplane[i] = (double *)vfft_aligned_alloc(p->lanes == 1 ? bytes : 2 * bytes);
        if (!p->mtplane[i]) break;
        p->mt_n = i + 1;
    }
    return p->mt_n;
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

/* the group plane pitch (doubles) of a lane-major plan: one lane group of VW
 * per element, plus a 512-B stagger so the blk planes of a block do not share
 * their cache sets at equal offsets */
static inline size_t vfft_zts_group_pitch(int N)
{
    return (size_t)N * (size_t)VFFT_ZTS_VW + 64u;
}

/* the bytes of ONE scratch plane for this lane mode (lanes = 1: the block
 * plane; lane-major: each of the two lane planes holds blk lane groups at
 * the group pitch) */
static inline size_t vfft_zts_plane_bytes(int N, int lanes, int blk)
{
    return lanes == 1 ? (size_t)2 * (size_t)N * sizeof(double) + VFFT_ZTS_SLACK
                      : (size_t)(blk < 1 ? 1 : blk) * vfft_zts_group_pitch(N) * sizeof(double) + VFFT_ZTS_SLACK;
}

/* own scratch planes for the plan (the create's default; the race's winner) */
static inline int vfft_zts_own_planes(vfft_zts_plan_t *p)
{
    const size_t bytes = vfft_zts_plane_bytes(p->N, p->lanes, p->blk);
    p->plane = p->plane_re = p->plane_im = NULL;
    if (p->lanes == 1)
        p->plane = (double *)vfft_aligned_alloc(bytes);
    else
    {
        p->plane_re = (double *)vfft_aligned_alloc(bytes);
        p->plane_im = (double *)vfft_aligned_alloc(bytes);
    }
    p->planes_owned = 1;
    return p->lanes == 1 ? p->plane != NULL : (p->plane_re != NULL && p->plane_im != NULL);
}

/* the race's shared scratch: one block plane, or two lane planes, not owned */
static inline void vfft_zts_share_planes(vfft_zts_plan_t *p, double *a, double *b)
{
    p->planes_owned = 0;
    if (p->lanes == 1) { p->plane = a; p->plane_re = p->plane_im = NULL; }
    else { p->plane = NULL; p->plane_re = a; p->plane_im = b; }
}

/* the BLOCK of a lane-major plan: blk lane groups per blocked ingest /
 * terminator pass. Legal when blk >= 1 and blk*VW <= lanes. Owned planes
 * are re-sized for it; shared planes must already hold blk groups (the
 * race sizes its scratch for the ladder's top). 1 = ok. */
static inline int vfft_zts_blk_legal(int lanes, int blk)
{
    return lanes > 1 && blk >= 1 && (size_t)blk * (size_t)VFFT_ZTS_VW <= (size_t)lanes;
}
static inline int vfft_zts_set_blk(vfft_zts_plan_t *p, int blk)
{
    if (!vfft_zts_blk_legal(p->lanes, blk)) return 0;
    if (p->blk == blk) return 1;
    p->blk = blk;
    if (p->planes_owned)
    {
        vfft_aligned_free(p->plane_re);
        vfft_aligned_free(p->plane_im);
        p->plane_re = p->plane_im = NULL;
        return vfft_zts_own_planes(p);
    }
    return 1;
}

/* the backward stream (the race builds candidates forward-only) */
static inline int vfft_zts_fill_bwd(vfft_zts_plan_t *p)
{
    size_t off = 0;
    int s;
    if (p->twb) return 1;
    p->twb = (double *)vfft_aligned_alloc(p->twdoubles * sizeof(double));
    if (!p->twb) return 0;
    for (s = 1; s < p->nf; s++)
    {
        const long RL = p->L[s] * p->chain[s];
        off += (p->lanes == 1)
                   ? vfft_qw_fill_stage_quad(p->twb + off, p->L[s], p->chain[s], RL, 1, VFFT_ZTS_VW)
                   : vfft_qw_fill_stage_pair(p->twb + off, p->L[s], p->chain[s], RL, 1);
    }
    return 1;
}

/* create: lanes = 1 (K=1 / contiguous K) or VFFT_ZTS_VW (lane-major). NULL = not
 * a zturn-ts cell, loudly under VFFT_NAT_LOG. want_bwd / want_planes = 0: the
 * race's candidate shape (forward stream only, planes shared by the caller). */
static inline vfft_zts_plan_t *_vfft_zts_create_ex(int N, const int *chain, int nf, int lanes,
                                                   int want_bwd, int want_planes)
{
    vfft_zts_plan_t *p;
    const vfft_zts_cell_t *cell;
    vfft_zts_plan_t st;   /* the staged form's stage table, resolved before the plan exists */
    long prod = 1, j;
    int s, staged = 0;
    size_t off;
    const char *why = NULL;
    memset(&st, 0, sizeof st);
    if (nf < 2 || nf > VFFT_ZTS_MAX_NF) why = "nf outside 2..7";
    else if (N < (lanes > 1 ? 9 : 16) || N > VFFT_ZTS_MAX_N) why = "N outside 16..262144 (lane-major: 9..)";
    else if (lanes != 1 && (lanes < 2 || lanes > VFFT_ZTS_LANES_MAX))
        why = "lanes is neither 1 nor 2..VFFT_ZTS_LANES_MAX";
    for (s = 0; !why && s < nf; s++)
    {
        if ((s == 0 || s == nf - 1) && lanes == 1)
        {   /* the S family's ends are the lane lattice (the ingest's turn, the
             * terminator's section transposes): 4 or 8; the V family's ends
             * carry no lattice and take every radix of the pool */
            if (chain[s] != 4 && chain[s] != 8) why = "an end radix not in {4, 8}";
        }
        else if (!vfft_zts_mid_ok(chain[s])) why = "a radix not in {4, 8, 3, 5, 7, 9, 15}";
        prod *= chain[s];
    }
    if (!why && prod != (long)N) why = "chain product != N";
    if (!why && lanes == 1 && (N / chain[0]) % VFFT_ZTS_VW) why = "(N / R0) % VW != 0";
    if (!why && lanes == 1 && chain[0] % VFFT_ZTS_VW) why = "R0 % VW != 0 (the ingest writes whole blocks)";
    cell = why ? NULL : vfft_zts_lookup(N, chain, nf);
    staged = !why && !cell;
    if (!why && staged && (N & (N - 1)) == 0) why = "no fused driver for this pow2 cell (zts_registry)";
    if (!why && staged && lanes == 1 && !vfft_zts_odd_band(N)) why = "N outside the odd band (2^a * {3,5,7,9,15}-smooth, a >= 4)";
    if (!why && staged && lanes > 1 && !vfft_zts_v_band(N)) why = "N outside the lane-major band ({3,4,5,7,8,9,15}-smooth)";
    if (!why)
    {   /* the stage table: every kernel resolved now, none looked up at run
         * time — for every plan, the fused cells too: the sectioned walk
         * (zts_mt.h) runs this table at any cell; a fused cell without a
         * stage kernel keeps its driver (the table stays NULL) and cannot
         * section */
        int ok = 1;
        for (s = 0; s < nf; s++)
        {
            const int R = chain[s];
            if (lanes == 1)
            {
                if (s == 0)           { st.s0f = _zts_s0_fn(R, 0); st.s0b = _zts_s0_fn(R, 1); ok = ok && st.s0f && st.s0b; }
                else if (s == nf - 1) { st.slf = _zts_sl_fn(R, 0); st.slb = _zts_sl_fn(R, 1); ok = ok && st.slf && st.slb; }
                else                  { st.smf[s] = _zts_sm_fn(R, 0); st.smb[s] = _zts_sm_fn(R, 1); ok = ok && st.smf[s] && st.smb[s]; }
            }
            else
            {
                if (s == 0)           { st.v0f = _zts_v0_fn(R, 0); st.v0b = _zts_v0_fn(R, 1); ok = ok && st.v0f && st.v0b; }
                else if (s == nf - 1) { st.vlf = _zts_vl_fn(R, 0); st.vlb = _zts_vl_fn(R, 1); ok = ok && st.vlf && st.vlb; }
                else                  { st.vmf[s] = _zts_vm_fn(R, 0); st.vmb[s] = _zts_vm_fn(R, 1); ok = ok && st.vmf[s] && st.vmb[s]; }
            }
        }
        if (!ok && staged) why = "no stage kernel for a radix in this chain (zts_stages.h)";
        if (!ok) memset(&st, 0, sizeof st);   /* a fused cell: no sectioned form */
    }
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
    p->blk = 1;
    p->gp = vfft_zts_group_pitch(N);
    p->staged = staged;
    {
        p->s0f = st.s0f; p->s0b = st.s0b; p->slf = st.slf; p->slb = st.slb;
        p->v0f = st.v0f; p->v0b = st.v0b; p->vlf = st.vlf; p->vlb = st.vlb;
        for (s = 0; s < VFFT_ZTS_MAX_NF; s++)
        {
            p->smf[s] = st.smf[s]; p->smb[s] = st.smb[s];
            p->vmf[s] = st.vmf[s]; p->vmb[s] = st.vmb[s];
        }
    }
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
    p->rb = (size_t *)vfft_aligned_alloc((size_t)p->ncol * sizeof(size_t));
    if (lanes > 1 && lanes % VFFT_ZTS_VW)
    {   /* the tail's scratch group (every plan that executes, the race's candidates too) */
        p->tail_re = (double *)vfft_aligned_alloc((size_t)N * VFFT_ZTS_VW * sizeof(double));
        p->tail_im = (double *)vfft_aligned_alloc((size_t)N * VFFT_ZTS_VW * sizeof(double));
    }
    if (!p->tw || !p->rb || (lanes > 1 && lanes % VFFT_ZTS_VW && (!p->tail_re || !p->tail_im)) ||
        (want_planes && !vfft_zts_own_planes(p)) ||
        (want_bwd && !vfft_zts_fill_bwd(p)))
    {
        vfft_zts_destroy(p);
        return NULL;
    }
    /* the forward stream, stage order, ONE allocation */
    for (s = 1, off = 0; s < nf; s++)
    {
        const long RL = p->L[s] * chain[s];
        p->twoff[s] = off;
        off += (lanes == 1)
                   ? vfft_qw_fill_stage_quad(p->tw + off, p->L[s], chain[s], RL, 0, VFFT_ZTS_VW)
                   : vfft_qw_fill_stage_pair(p->tw + off, p->L[s], chain[s], RL, 0);
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

static inline vfft_zts_plan_t *vfft_zts_create_chain(int N, const int *chain, int nf, int lanes)
{
    return _vfft_zts_create_ex(N, chain, nf, lanes, 1, 1);
}

/* the block plane's start for this call (lanes = 1): the 512-B slot mod 4 KB
 * farthest from all four caller pointers; the allocation (`alloc`: the
 * plan's plane, or a worker's) carries the slack */
static inline double *_zts_plane_at(const double *alloc, const double *ire, const double *iim,
                                    const double *ore, const double *oim)
{
    const uintptr_t base = (uintptr_t)alloc;
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
static inline double *_zts_plane_for(const vfft_zts_plan_t *p, const double *ire, const double *iim,
                                     const double *ore, const double *oim)
{
    return _zts_plane_at(p->plane, ire, iim, ore, oim);
}

/* one slot of the threaded contiguous form: transforms [t0, t0 + cnt) on the
 * slot's own block plane (the serial loop's exact arithmetic per transform) */
static inline void vfft_zts_run_slab(const vfft_zts_plan_t *p, int bwd,
                                     const double *ire, const double *iim, double *ore, double *oim,
                                     size_t t0, size_t cnt, const double *plane_alloc)
{
    const double *tw = bwd ? p->twb : p->tw;
    const size_t off = t0 * (size_t)p->N;
    const double *ir = ire + off, *ii = iim + off;
    double *orr = ore + off, *oi = oim + off;
    double *W = _zts_plane_at(plane_alloc, ir, ii, orr, oi);
    if (p->staged)
    {
        const size_t N = (size_t)p->N;
        size_t t;
        for (t = 0; t < cnt; t++)
            _zts_staged_s(p, bwd, ir + t * N, ii + t * N, orr + t * N, oi + t * N, W);
    }
    else if (cnt == 1)
        (bwd ? p->cell->s_bwd : p->cell->s_fwd)(ir, ii, orr, oi, W, tw, p->rb, p->tile);
    else
        (bwd ? p->cell->s_bwd_batch : p->cell->s_fwd_batch)(ir, ii, orr, oi, W, tw, p->rb, p->tile, cnt);
}

/* the lane-major planes: 1 KB and 3 KB (mod 4 KB) from their allocations */
static inline double *_zts_vplane(const double *alloc, unsigned want)
{
    const uintptr_t base = (uintptr_t)alloc;
    return (double *)(base + (((want - (base & 4095u)) & 4095u) & ~(uintptr_t)63u));
}

/* ── THE STAGED EXECUTOR (the odd band): the fused driver's walk as calls ──
 * S: the ingest (split planes -> the block plane W), the mids in place on W
 * (the tiled prefix per tile, the rest whole-plane), the terminator (W -> the
 * caller's planes). One transform. */
static inline void _zts_staged_s(const vfft_zts_plan_t *p, int bwd,
                                 const double *ire, const double *iim, double *ore, double *oim, double *W)
{
    const double *tw = bwd ? p->twb : p->tw;
    const int nf = p->nf;
    const size_t N = (size_t)p->N, tile = p->tile, L = (size_t)p->L[nf - 1];
    int s;
    size_t t;
    (bwd ? p->s0b : p->s0f)(ire, iim, W, NULL, p->rb, (size_t)p->ncol, 0, 0, (size_t)p->ncol);
    if (tile)
        for (t = 0; t < N / tile; t++)
        {
            double *B = W + t * tile * 2;
            for (s = 1; s < nf - 1; s++)
            {
                const size_t RL = (size_t)p->L[s] * (size_t)p->chain[s];
                if (tile % RL == 0)
                    (bwd ? p->smb[s] : p->smf[s])(B, B, tw + p->twoff[s], NULL, (size_t)p->L[s], tile / RL, 0, (size_t)p->L[s]);
            }
        }
    for (s = 1; s < nf - 1; s++)
    {
        const size_t RL = (size_t)p->L[s] * (size_t)p->chain[s];
        if (!tile || tile % RL)
            (bwd ? p->smb[s] : p->smf[s])(W, W, tw + p->twoff[s], NULL, (size_t)p->L[s], (size_t)p->Gs[s], 0, (size_t)p->L[s]);
    }
    (bwd ? p->slb : p->slf)(W, ore, oim, tw + p->twoff[nf - 1], NULL, L, 1, L, L);
}
/* V: the blocked loop of the lane-major driver over the lane range [g0, g1):
 * the ingest of nb groups, the mids per group on its plane, the terminator
 * of the block */
static inline void _zts_staged_v(const vfft_zts_plan_t *p, int bwd,
                                 const double *ire, const double *iim, double *ore, double *oim,
                                 size_t g0, size_t g1, double *W0, double *Wi0, size_t pitch)
{
    const double *tw = bwd ? p->twb : p->tw;
    const int nf = p->nf;
    const size_t N = (size_t)p->N, tile = p->tile, L = (size_t)p->L[nf - 1];
    const size_t K = pitch, blk = (size_t)p->blk, gp = p->gp, VW = (size_t)VFFT_ZTS_VW;
    size_t gl;
    g1 -= (g1 - g0) % VW;   /* whole groups only: a lane count off the width ends at its full groups (the tail is run apart) */
    for (gl = g0; gl < g1; gl += VW * blk)
    {
        const size_t nb = ((g1 - gl) / VW < blk) ? (g1 - gl) / VW : blk;
        size_t gb, t;
        int s;
        (bwd ? p->v0b : p->v0f)(ire + gl, iim + gl, W0, Wi0, NULL, p->rb, (size_t)p->ncol, 0, 0, (size_t)p->ncol, K, nb, gp);
        for (gb = 0; gb < nb; gb++)
        {
            double *W = W0 + gb * gp, *Wi = Wi0 + gb * gp;
            if (tile)
                for (t = 0; t < N / tile; t++)
                {
                    double *B = W + t * tile * VW, *Bi = Wi + t * tile * VW;
                    for (s = 1; s < nf - 1; s++)
                    {
                        const size_t RL = (size_t)p->L[s] * (size_t)p->chain[s];
                        if (tile % RL == 0)
                            (bwd ? p->vmb[s] : p->vmf[s])(B, Bi, B, Bi, tw + p->twoff[s], NULL, (size_t)p->L[s], tile / RL, 0, (size_t)p->L[s]);
                    }
                }
            for (s = 1; s < nf - 1; s++)
            {
                const size_t RL = (size_t)p->L[s] * (size_t)p->chain[s];
                if (!tile || tile % RL)
                    (bwd ? p->vmb[s] : p->vmf[s])(W, Wi, W, Wi, tw + p->twoff[s], NULL, (size_t)p->L[s], (size_t)p->Gs[s], 0, (size_t)p->L[s]);
            }
        }
        (bwd ? p->vlb : p->vlf)(W0, Wi0, ore + gl, oim + gl, tw + p->twoff[nf - 1], NULL, L, 1, L, L, K, nb, gp);
    }
}

/* THE TAIL (2026-11-07): the r = lanes % VW lanes past the full groups, run
 * as one zero-padded group of VW lanes on the plan's scratch (pitch VW) and
 * copied back — internal padding, the caller's tight layout untouched. The
 * V kernels compute each lane independently, so the tail's lanes come out
 * bitwise what a caller-padded batch would give. On the plane set given
 * (the caller's, or a slot's). No-op when the lane count is a multiple of VW. */
static inline void _zts_run_tail(const vfft_zts_plan_t *p, int bwd,
                                 const double *ire, const double *iim, double *ore, double *oim,
                                 double *Wr, double *Wi)
{
    const size_t VW = (size_t)VFFT_ZTS_VW, K = (size_t)p->lanes, r = K % VW, g0 = K - r, N = (size_t)p->N;
    const double *tw = bwd ? p->twb : p->tw;
    size_t n, j;
    if (r == 0) return;
    for (n = 0; n < N; n++)
    {
        for (j = 0; j < r; j++)
        {
            p->tail_re[n * VW + j] = ire[n * K + g0 + j];
            p->tail_im[n * VW + j] = iim[n * K + g0 + j];
        }
        for (j = r; j < VW; j++)
        {
            p->tail_re[n * VW + j] = 0.0;
            p->tail_im[n * VW + j] = 0.0;
        }
    }
    if (p->staged)
        _zts_staged_v(p, bwd, p->tail_re, p->tail_im, p->tail_re, p->tail_im, 0, VW, Wr, Wi, VW);
    else
        (bwd ? p->cell->v_bwd : p->cell->v_fwd)(p->tail_re, p->tail_im, p->tail_re, p->tail_im, Wr, Wi, tw, p->rb, p->tile,
                                                VW, (size_t)p->blk, p->gp, 0, VW);
    for (n = 0; n < N; n++)
        for (j = 0; j < r; j++)
        {
            ore[n * K + g0 + j] = p->tail_re[n * VW + j];
            oim[n * K + g0 + j] = p->tail_im[n * VW + j];
        }
}

/* execute. lanes = 1: K contiguous transforms (K >= 1), transform t at offset
 * t*N in every plane; lanes = VW: one lane-major batch of VW transforms. In
 * place when in == out. The backward call needs the backward stream
 * (vfft_zts_fill_bwd; the create builds it unless asked not to). */
static inline void _zts_run(const vfft_zts_plan_t *p, int bwd,
                            const double *ire, const double *iim, double *ore, double *oim, size_t K)
{
    const double *tw = bwd ? p->twb : p->tw;
    if (p->lanes == 1)
    {
        double *W = _zts_plane_for(p, ire, iim, ore, oim);
        if (p->staged)
        {
            const size_t N = (size_t)p->N;
            size_t t;
            for (t = 0; t < K; t++)
                _zts_staged_s(p, bwd, ire + t * N, iim + t * N, ore + t * N, oim + t * N, W);
        }
        else if (K == 1)
            (bwd ? p->cell->s_bwd : p->cell->s_fwd)(ire, iim, ore, oim, W, tw, p->rb, p->tile);
        else
            (bwd ? p->cell->s_bwd_batch : p->cell->s_fwd_batch)(ire, iim, ore, oim, W, tw, p->rb, p->tile, K);
    }
    else
    {
        double *Wr = _zts_vplane(p->plane_re, 1024u), *Wi = _zts_vplane(p->plane_im, 3072u);
        const size_t full = (size_t)p->lanes - (size_t)p->lanes % VFFT_ZTS_VW;   /* the drivers take whole groups only */
        if (p->staged)
            _zts_staged_v(p, bwd, ire, iim, ore, oim, 0, full, Wr, Wi, (size_t)p->lanes);
        else
            (bwd ? p->cell->v_bwd : p->cell->v_fwd)(ire, iim, ore, oim, Wr, Wi, tw, p->rb, p->tile, (size_t)p->lanes,
                                                    (size_t)p->blk, p->gp, 0, full);
        _zts_run_tail(p, bwd, ire, iim, ore, oim, Wr, Wi);   /* the lanes past the full groups */
    }
}

/* the lane-major pass over the lane range [g0, g1) (multiples of VW) on a
 * plane set: the plan's own (pre = plane_re, pim = plane_im) or a worker's
 * allocation (pre = mtplane[t], pim = NULL: the im set follows the re set) */
static inline void vfft_zts_run_lanes(const vfft_zts_plan_t *p, int bwd,
                                      const double *ire, const double *iim, double *ore, double *oim,
                                      size_t g0, size_t g1, double *pre, double *pim)
{
    const double *tw = bwd ? p->twb : p->tw;
    double *Wr, *Wi;
    if (!pim) pim = pre + vfft_zts_plane_bytes(p->N, p->lanes, p->blk) / sizeof(double);
    const size_t full = (size_t)p->lanes - (size_t)p->lanes % VFFT_ZTS_VW;   /* the drivers take whole groups only */
    const size_t ge = g1 > full ? full : g1;
    Wr = _zts_vplane(pre, 1024u);
    Wi = _zts_vplane(pim, 3072u);
    if (g0 < ge)
    {
        if (p->staged)
            _zts_staged_v(p, bwd, ire, iim, ore, oim, g0, ge, Wr, Wi, (size_t)p->lanes);
        else
            (bwd ? p->cell->v_bwd : p->cell->v_fwd)(ire, iim, ore, oim, Wr, Wi, tw, p->rb, p->tile, (size_t)p->lanes,
                                                    (size_t)p->blk, p->gp, g0, ge);
    }
    if (g1 >= (size_t)p->lanes)
        _zts_run_tail(p, bwd, ire, iim, ore, oim, Wr, Wi);   /* the range that reaches the end runs the tail */
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
