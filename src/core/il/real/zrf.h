/* zrf.h - the real FLAT DIT: the odd-N real transform on the c2c flat DIT's
 * own stages (il/rank1/il_flatdit.h), real input never widened to complex.
 *
 * The flat DIT's first stage transforms across the most significant digit:
 * R0 legs at stride D0 = N/R0, D0 contiguous columns. On REAL input that
 * stage is the real R0-point DFT of each column, and its output is Hermitian
 * in the digit p: only p = 0..h (h = (R0-1)/2) exist.
 *   digit 0   is a REAL run of D0: the same problem, R0 times smaller. It
 *             goes down one LEVEL (the next radix of the chain as its leaf),
 *             until the run has a real mono kernel (il/real/zrm.h), which
 *             writes its bins straight to their natural places (stride
 *             N/run).
 *   digit p   (1..h) is a COMPLEX run of D0: exactly the state the c2c flat
 *             DIT is in after its own leaf, for block p. The c2c plan's
 *             stages 1.. run on blocks 1..h as they stand -- a contiguous
 *             range of every stage's units -- in the SCRAMBLED class, in
 *             place on the level's plane.
 * The one new kernel is the leaf, r1c (codelets/zil/<isa>/real/flat/,
 * generator/lib/gen/real_il.ml): real legs in, digit 0 real at plane[k],
 * digit p complex at block p of the c2c plane (complex p*D + k), so no
 * pointer is shifted and no block is copied. A level's plane is (R+1)*D
 * doubles; the second half of block 0 is unused.
 *
 * Block p holds the bins congruent to p mod R0, in the scrambled class's
 * block order. The r2c wants bins 0..N/2: a bin f above N/2 is stored as
 * conj at N - f. One ORDER SWEEP per level moves the blocks to the caller's
 * half spectrum. It walks the last stage's blocks: a block of Rl legs holds
 * the bins low + l*Nj/Rl, so legs below Rl/2 go out in natural order at a
 * constant stride from the block's base, legs above it conjugated at the
 * mirrored stride, and the middle leg by the block's flag -- one table
 * entry per block, none per element. c2r starts each level with the same
 * walk as a gather, runs the scrambled class's TRANSPOSED stages, and ends
 * with the leaf's inverse. Every stage works on (N - D0)/2 complex where
 * the c2c transform of the widened input works on N.
 *
 * THE TILE AXIS is the c2c flat DIT's (its validator is the law): a level
 * with a tile width runs its stages from the cut on, and its sweep,
 * depth-first per tile -- one block of the cut stage at a time -- so a
 * tile's stages and its sweep meet it in L1. The plan input is a width
 * BUDGET; each level takes its widest legal stage span within it.
 * A tile holds the bins congruent to its slow digits' index Q modulo
 * P = the tiles of the whole plane: a comb across the half spectrum, its
 * direct legs at residue Q and its mirrored legs at residue P - Q. The
 * tiles are walked in ascending min(Q, P - Q), and a contiguous range of
 * the walk owns a contiguous set of residues (the threaded form's
 * workers, zrf_mt.h, write disjoint lines). The sweep is by GROUPS of
 * consecutive tiles: a group's stages run tile by tile (each tile L1-hot),
 * then the group's bins go out in OUTPUT order -- for every period P and
 * leg, one run of consecutive residues, each bin fetched from its tile's
 * block (the direct-class tile of that residue, or the mirror-class tile of
 * the opposite residue, conjugated from its mirrored leg). A tile swept
 * alone writes its bins as a comb over the whole half spectrum (one page
 * touch and one line per bin); a group writes runs. The group is sized
 * so its tiles stay in L2 (VFFT_ZRF_GROUP_BYTES). The threaded form sweeps
 * by groups; the serial plan sweeps each tile from L1 (faster on one core).
 *
 * Both placements are one pipeline: the first leaf reads the whole input
 * before any sweep writes (r2c), and every gather reads before the last leaf
 * writes (c2r). Unnormalized: c2r(r2c(x)) = N x.
 *
 * The plan inputs are the chain, the split-body form switch and the tile
 * budget; the odd real race sweeps them at create and banks eng=zrf
 * (bridge/real_bridge.h, wisdom2_real_il.h). The level at which the mono
 * takes over is derived: the first run with an rn1 kernel. The threaded
 * form is zrf_mt.h.
 */
#ifndef VFFT_ZRF_H
#define VFFT_ZRF_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#include "il_flatdit.h" /* the c2c flat DIT: plan, create, the bound stage records, the executor */
#include "zrm.h"        /* the real mono: the terminal level */

#define VFFT_ZRF_MAX_LV VFFT_ILFD_MAX_K
#define VFFT_ZRF_GROUP_BYTES (1u << 20)   /* a sweep group's tiles: half of L2 */

/* the leaf at radix R in one direction, or 0 when the kind has no such radix */
static inline vfft_il2p_fn vfft_zrf_leaf_fn(int R, int bwd)
{
    switch (R)
    {
#if defined(VFFT_IL_R1C_PAIR_RADICES) && VFFT_IL_VW == 4
#define C(R_) case R_: return bwd ? VFFT_IL_SYM(radix##R_##_z_r1c_bwd) : VFFT_IL_SYM(radix##R_##_z_r1c_fwd);
    VFFT_IL_R1C_PAIR_RADICES(C)
#undef C
#endif
    default: (void)bwd; return 0;
    }
}

/* one LEVEL: a real run of Nj = R * D through the leaf, then the c2c stages
 * on blocks 1..h of its plane */
typedef struct {
    int R, h, ns;                 /* the leaf radix, its digits 1..h, the c2c stages (the level's chain - 1) */
    size_t Nj, D;
    vfft_il2p_fn lf, lb;          /* r1c fwd / bwd */
    vfft_ilfd_plan_t *fd;         /* the c2c flat DIT of length Nj, scrambled class: tables and records */
    /* its stage records on the level's plane: cf in stage order, ct the
     * transposed backward in its order. A WIDE record is cut to blocks 1..h
     * and runs once (t = 1); a TILED one is the c2c plan's per-tile record
     * and runs per tile t of [t0, t1). nwide = the forward's leading wide
     * records (all of them when untiled); the backward's first ns - nwide
     * are the tiled ones. */
    vfft_ilfd_call_t cf[VFFT_ILFD_MAX_K], ct[VFFT_ILFD_MAX_K];
    int nwide;
    size_t tw, t0, t1, bpt;       /* the tile width (complex; 0 = untiled), its tiles in blocks 1..h
                                   * (the plane's tile numbering), last-stage blocks per tile */
    double *t2;                   /* a tail stage 1: the group base of block 1, fwd then conj */
    double *plane;                /* (R + 1) * D doubles */
    size_t Rl, ost, top, nlb;     /* the sweep: the last stage's radix, a leg's stride in the half
                                   * spectrum (doubles), 2N, the last-stage blocks in blocks 1..h */
    uint32_t *low;                /* per such block: its base (doubles) | the middle leg mirrored << 31 */
    uint32_t *tord;               /* the tiles of [t0, t1) in walk order: ascending min(Q, P - Q) */
    /* the group sweep (tiled levels): P = the plane's tiles, nst = Nj/Rl, sc = the
     * caller's doubles per level bin (2N/Nj), gsz = tiles per group; per walk entry
     * its key, its Q and its first block's complex index; jofm = the block of a
     * tile holding natural offset m (the same for every tile) */
    size_t P, nst, sc, gsz;
    uint32_t *tkey, *tQ, *tbase, *jofm;
} _zrf_level_t;

typedef struct vfft_zrf_s {
    int N, K, J;                  /* J flat levels, then the mono */
    int R[VFFT_ILFD_MAX_K];
    int nomsz;                    /* 1 = the split-body stage form is off (plan input) */
    int tile;                     /* the tile width budget in complex, 0 = untiled (plan input) */
    int mt, mt_t;                 /* the threaded arm (zrf_mt.h): 1 FIRST, 2 LEVELS, bound for mt_t threads,
                                   * raced at the plan's T and banked on the threaded plan's row */
    int mt_lv;                    /* its threaded levels (the leading ones), the rest serial on the caller */
    size_t mt_lo[64][VFFT_ZRF_MAX_LV], mt_hi[64][VFFT_ZRF_MAX_LV]; /* per worker, per threaded level: its walk range */
    _zrf_level_t lv[VFFT_ZRF_MAX_LV];
    vfft_oop11_fn mf, mb;         /* rn1 at the last run */
    size_t NJ, MJ;                /* the mono's length and its bin stride N / NJ */
} vfft_zrf_plan_t;

static inline void vfft_zrf_destroy(vfft_zrf_plan_t *p)
{
    int j;
    if (!p) return;
    for (j = 0; j < VFFT_ZRF_MAX_LV; j++)
    {
        vfft_ilfd_destroy(p->lv[j].fd);
        vfft_aligned_free(p->lv[j].t2);
        vfft_aligned_free(p->lv[j].plane);
        free(p->lv[j].low);
        free(p->lv[j].tord);
        free(p->lv[j].tkey); free(p->lv[j].tQ); free(p->lv[j].tbase); free(p->lv[j].jofm);
    }
    free(p);
}

/* a bound record of the c2c plan rebased on the level's plane (in = out =
 * the plane, passed as zin / zout) */
static inline void _zrf_rebase_rec(vfft_ilfd_call_t *c)
{
    if (c->in_sel != _ILFD_NUL) c->in_sel = _ILFD_ZIN;
    c->out_sel = _ILFD_ZOUT;
}

/* A bound stage record of the c2c plan, cut to the blocks of first digit
 * 1..h (the record runs with t = 1, which steps its bases over block 0).
 * bpt = the stage's blocks per first digit. A tail form at stage 1 packs
 * the R0 blocks of its one group into lanes: there the cut is a COLUMN
 * range (count = h from block 1) and the group base record becomes block
 * 1's (t2one); the pair stream is relative to the first column and stays. */
static inline int _zrf_range_rec(const vfft_ilfd_plan_t *p, int s, size_t h, const double *t2one,
                                 vfft_ilfd_call_t *c)
{
    const size_t bpt = p->nblk[s] / (size_t)p->R[0];
    const size_t recs = (size_t)(p->R[s] - 1) * VFFT_IL_TWREC;
    const size_t w = 2 * p->D[0]; /* one first-digit block, doubles */
    _zrf_rebase_rec(c);
    c->in_tstep = c->out_tstep = c->tw_tstep = c->t2_tstep = c->a1_tstep = c->g_tstep = 0;
    if (c->op == _ILFD_ONE)
    {
        if (c->a1)
        {   /* t2csgn: reads by the relative group, writes by the absolute base table */
            if (s == 1) { c->count = h; c->Gs = 1; c->in_tstep = c->out_tstep = w; c->t2 = t2one; }
            else
            {
                const size_t gpt = bpt / c->count;
                c->Gs = h * gpt; c->in_tstep = w; c->a1_tstep = gpt * c->count; c->t2_tstep = gpt * VFFT_IL_TWREC;
            }
        }
        else if (c->in_sel == _ILFD_NUL) { c->Gs = h * bpt; c->out_tstep = w; c->tw_tstep = bpt * recs; }     /* msz */
        else { c->OGs = h * bpt; c->in_tstep = c->out_tstep = w; c->tw_tstep = bpt * recs; }                  /* t2cp */
        return 1;
    }
    if (c->op == _ILFD_COL)
    {
        if (s == 1) { c->count = h; c->ngrp = 1; c->in_tstep = c->out_tstep = w; c->t2 = t2one; c->t2_step = 0; }
        else { const size_t gpt = bpt / c->G; c->ngrp = h * gpt; c->g_tstep = gpt; }
        return 1;
    }
    return 0; /* a per-block redirected last stage: the natural class only */
}

/* the chain as text: "9.9.5" */
static inline void vfft_zrf_chain_str(const int *R, int K, char *buf, size_t n)
{
    size_t off = 0;
    int s;
    if (n) buf[0] = 0;
    for (s = 0; s < K && off + 1 < n; s++)
    {
        const int r = snprintf(buf + off, n - off, "%s%d", s ? "." : "", R[s]);
        if (r < 0) break;
        off += (size_t)r;
    }
}

/* Create from a chain (N = the product, every radix odd), the split-body
 * switch and the tile budget. NULL when a level has no leaf, a c2c stage has
 * no transposed twin, or N itself is the mono's (the first level must be
 * flat). */
static inline vfft_zrf_plan_t *vfft_zrf_create(int N, const int *R, int K, int nomsz, int tile)
{
#if VFFT_IL_VW == 4
    vfft_zrf_plan_t *p;
    long prod = 1;
    size_t Nj;
    int s, j;
    if (N < 9 || !(N & 1) || K < 2 || K > VFFT_ILFD_MAX_K) return 0;
    for (s = 0; s < K; s++) { if (R[s] < 3 || !(R[s] & 1)) return 0; prod *= R[s]; }
    if (prod != (long)N) return 0;
    p = (vfft_zrf_plan_t *)calloc(1, sizeof *p);
    if (!p) return 0;
    p->N = N; p->K = K; p->nomsz = nomsz != 0; p->tile = tile > 0 ? tile : 0;
    for (s = 0; s < K; s++) p->R[s] = R[s];
    Nj = (size_t)N;
    for (j = 0; j < K; j++)
    {
        _zrf_level_t *lv = &p->lv[j];
        vfft_ilfd_plan_t *fd;
        const int Kj = K - j;
        if (j >= 1 && Nj <= (size_t)VFFT_ZRM_MAX_N && vfft_zrm_fn((int)Nj, 0) && vfft_zrm_fn((int)Nj, 1)) break;
        if (Kj < 2) { vfft_zrf_destroy(p); return 0; }
        lv->R = R[j]; lv->h = R[j] / 2; lv->Nj = Nj; lv->D = Nj / (size_t)R[j]; lv->ns = Kj - 1;
        lv->lf = vfft_zrf_leaf_fn(R[j], 0);
        lv->lb = vfft_zrf_leaf_fn(R[j], 1);
        if (!lv->lf || !lv->lb) { vfft_zrf_destroy(p); return 0; }
        fd = lv->fd = vfft_ilfd_create_chain((int)Nj, R + j, Kj);
        if (!fd || !fd->bwd_ok || !fd->scr_ok) { vfft_zrf_destroy(p); return 0; }
        fd->scr = 1;
        if (nomsz) for (s = 0; s < Kj; s++) fd->msz[s] = 0;
        vfft_ilfd_bind(fd);
        vfft_aligned_free(fd->stg); /* the stages run on the level's plane */
        fd->stg = 0;
        if (p->tile > 0)
        {   /* the widest legal stage span within the budget (spans shrink with the stage) */
            for (s = 1; s <= Kj - 2; s++)
                if (!fd->tail[s] && (long)fd->R[s] * (long)fd->D[s] <= (long)p->tile)
                {
                    vfft_ilfd_apply_tw(fd, (int)((long)fd->R[s] * (long)fd->D[s]));
                    break;
                }
        }
        if (fd->tail[1] && !fd->msz[1])
        {   /* the group base of block 1 at stage 1: w^Q with Q = 1, modulus Nj / D_1 */
            const size_t L = Nj / fd->D[1];
            double c, sn;
            int lane;
            lv->t2 = (double *)vfft_aligned_alloc(16 * sizeof(double));
            if (!lv->t2) { vfft_zrf_destroy(p); return 0; }
            vfft_cs2pi_exact(1, (long long)L, &c, &sn);
            sn = -sn;
            for (lane = 0; lane < 4; lane++)
            {
                lv->t2[lane] = c; lv->t2[4 + lane] = (lane & 1) ? sn : -sn;
                lv->t2[8 + lane] = c; lv->t2[12 + lane] = (lane & 1) ? -sn : sn; /* bwd: conj */
            }
        }
        lv->tw = fd->tw > 0 ? (size_t)fd->tw : 0;
        lv->nwide = lv->tw ? fd->tcut - 1 : lv->ns;
        for (s = 1; s < Kj; s++)
        {
            vfft_ilfd_call_t *cf = &lv->cf[s - 1], *ct = &lv->ct[Kj - 1 - s];
            *cf = fd->cf[s];
            *ct = fd->ct[Kj - 1 - s];
            if (lv->tw && s >= fd->tcut) { _zrf_rebase_rec(cf); _zrf_rebase_rec(ct); continue; } /* the c2c plan's tile records */
            if (!_zrf_range_rec(fd, s, (size_t)lv->h, lv->t2, cf) ||
                !_zrf_range_rec(fd, s, (size_t)lv->h, lv->t2 ? lv->t2 + 8 : 0, ct))
            { vfft_zrf_destroy(p); return 0; }
        }
        lv->Rl = (size_t)R[K - 1];
        lv->nlb = (size_t)lv->h * lv->D / lv->Rl;
        if (lv->tw) { lv->t0 = lv->D / lv->tw; lv->t1 = ((size_t)lv->h + 1) * lv->D / lv->tw; lv->bpt = lv->tw / lv->Rl; }
        lv->plane = (double *)vfft_aligned_alloc((((size_t)lv->R + 1) * lv->D + 8) * sizeof(double));
        lv->low = (uint32_t *)malloc(lv->nlb * sizeof(uint32_t));
        if (!lv->plane || !lv->low) { vfft_zrf_destroy(p); return 0; }
        memset(lv->plane, 0, (((size_t)lv->R + 1) * lv->D + 8) * sizeof(double));
        {   /* block order -> the half spectrum: block b of the last stage holds the level's
             * bins natbase[b] + l*Nj/Rl, the transform's bins those times N/Nj. In doubles:
             * leg l direct at base + l*ost, mirrored at 2N - base - l*ost. */
            const size_t M = (size_t)N / Nj, b0 = lv->D / lv->Rl;
            size_t bi;
            lv->ost = 2 * M * (Nj / lv->Rl);
            lv->top = 2 * (size_t)N;
            for (bi = 0; bi < lv->nlb; bi++)
            {
                const size_t base = 2 * M * fd->natbase[b0 + bi];
                const int mir = base + (lv->Rl / 2) * lv->ost > (size_t)N; /* the middle leg's bin is above N/2 */
                lv->low[bi] = (uint32_t)base | (mir ? 0x80000000u : 0u);
            }
        }
        if (lv->tw)
        {   /* the walk order: a counting sort on min(Q, P - Q) (distinct: P - Q is a tile of the
             * mirrored blocks, never of blocks 1..h) */
            const size_t P = Nj / lv->tw, nt = lv->t1 - lv->t0;
            uint32_t *at = (uint32_t *)malloc((P + 1) * sizeof(uint32_t));
            size_t k = 0, q, t;
            lv->tord = (uint32_t *)malloc(nt * sizeof(uint32_t));
            if (!at || !lv->tord) { free(at); vfft_zrf_destroy(p); return 0; }
            memset(at, 0xff, (P + 1) * sizeof(uint32_t));
            for (t = lv->t0; t < lv->t1; t++)
            {
                const size_t Q = _ilfd_block_Q(fd, fd->tcut, t);
                at[Q < P - Q ? Q : P - Q] = (uint32_t)t;
            }
            for (q = 0; q <= P; q++) if (at[q] != 0xffffffffu) lv->tord[k++] = at[q];
            free(at);
            if (k != nt) { vfft_zrf_destroy(p); return 0; }
            /* the group sweep's tables */
            lv->P = P; lv->nst = Nj / lv->Rl; lv->sc = 2 * ((size_t)N / Nj);
            lv->gsz = VFFT_ZRF_GROUP_BYTES / (lv->tw * 16); if (lv->gsz < 1) lv->gsz = 1;
            lv->tkey = (uint32_t *)malloc(nt * sizeof(uint32_t));
            lv->tQ = (uint32_t *)malloc(nt * sizeof(uint32_t));
            lv->tbase = (uint32_t *)malloc(nt * sizeof(uint32_t));
            lv->jofm = (uint32_t *)malloc(lv->bpt * sizeof(uint32_t));
            if (!lv->tkey || !lv->tQ || !lv->tbase || !lv->jofm) { vfft_zrf_destroy(p); return 0; }
            for (k = 0; k < nt; k++)
            {
                const size_t tt = lv->tord[k], Q = _ilfd_block_Q(fd, fd->tcut, tt);
                lv->tQ[k] = (uint32_t)Q; lv->tkey[k] = (uint32_t)(Q < P - Q ? Q : P - Q);
                lv->tbase[k] = (uint32_t)(tt * lv->bpt * lv->Rl);
            }
            {   /* block jb of the first tile holds natural offset (natbase - Q) / P */
                const size_t Q0 = _ilfd_block_Q(fd, fd->tcut, lv->t0);
                size_t jb;
                memset(lv->jofm, 0xff, lv->bpt * sizeof(uint32_t));
                for (jb = 0; jb < lv->bpt; jb++)
                {
                    const size_t d = fd->natbase[lv->t0 * lv->bpt + jb] - Q0;
                    if (d % P || d / P >= lv->bpt) { vfft_zrf_destroy(p); return 0; }
                    lv->jofm[d / P] = (uint32_t)jb;
                }
                for (jb = 0; jb < lv->bpt; jb++) if (lv->jofm[jb] == 0xffffffffu) { vfft_zrf_destroy(p); return 0; }
            }
        }
        Nj = lv->D;
    }
    if (j >= K || Nj > (size_t)VFFT_ZRM_MAX_N) { vfft_zrf_destroy(p); return 0; }
    p->J = j; p->NJ = Nj; p->MJ = (size_t)N / Nj;
    p->mf = vfft_zrm_fn((int)Nj, 0);
    p->mb = vfft_zrm_fn((int)Nj, 1);
    if (!p->mf || !p->mb) { vfft_zrf_destroy(p); return 0; }
    return p;
#else
    (void)N; (void)R; (void)K; (void)nomsz; (void)tile;
    return 0;
#endif
}

/* 1 when any level of the plan is tiled (a budget no level can take builds
 * the untiled plan) */
static inline int vfft_zrf_tiled(const vfft_zrf_plan_t *p)
{
    int j;
    for (j = 0; j < p->J; j++) if (p->lv[j].tw) return 1;
    return 0;
}

/* The chain candidates: ordered compositions of N over the leaf's radices in
 * the flat DIT's seed order, at most `max`; *dropped counts the ones past
 * the cap. The create is the validator. */
static inline void _zrf_chains_rec(int L, int depth, int *cur, int (*out)[VFFT_ILFD_MAX_K], int *lens,
                                   int *n, int max, int *dropped)
{
    static const int POOL[] = { 9, 7, 5, 3, 25, 27, 21, 23, 19, 17, 15, 13, 11, 29, 31, 37, 41, 43, 47 };
    int i;
    if (L == 1)
    {
        if (depth < 2) return;
        if (*n >= max) { (*dropped)++; return; }
        memcpy(out[*n], cur, sizeof(int) * VFFT_ILFD_MAX_K);
        lens[(*n)++] = depth;
        return;
    }
    if (depth >= VFFT_ILFD_MAX_K) return;
    for (i = 0; i < (int)(sizeof POOL / sizeof POOL[0]); i++)
        if (L % POOL[i] == 0)
        {
            cur[depth] = POOL[i];
            _zrf_chains_rec(L / POOL[i], depth + 1, cur, out, lens, n, max, dropped);
        }
}
static inline int vfft_zrf_chains(int N, int (*out)[VFFT_ILFD_MAX_K], int *lens, int max, int *dropped)
{
    int cur[VFFT_ILFD_MAX_K], n = 0, d = 0;
    memset(cur, 0, sizeof cur);
    if (N >= 9 && (N & 1)) _zrf_chains_rec(N, 0, cur, out, lens, &n, max, &d);
    if (dropped) *dropped = d;
    return n;
}

/* 1 when N has at least one chain to sweep (the create still validates it) */
static inline int _zrf_has_chain(int N)
{
    int ch[1][VFFT_ILFD_MAX_K], len[1];
    return VFFT_IL_VW == 4 && vfft_zrf_chains(N, ch, len, 1, NULL) > 0;
}

#if defined(__AVX2__) || defined(__SSE2__)
/* the order sweep: last-stage blocks [b0, b1) of blocks 1..h -> the half
 * spectrum. Rl is a constant at the dispatch below, so the leg loops unroll. */
static inline __attribute__((always_inline)) void _zrf_sweep_r(const _zrf_level_t *lv, double *out,
                                                               size_t b0, size_t b1, const size_t Rl)
{
    const __m128d cj = _mm_castsi128_pd(_mm_set_epi64x((long long)0x8000000000000000ull, 0));
    const size_t hl = Rl / 2, st = lv->ost;
    const double *src = lv->plane + 2 * lv->D + 2 * Rl * b0;
    const uint32_t *lo = lv->low;
    size_t b, l;
    for (b = b0; b < b1; b++, src += 2 * Rl)
    {
        const uint32_t e = lo[b];
        const size_t mir = e >> 31;
        double *d = out + (e & 0x7fffffffu);            /* leg l direct at d + l*st */
        double *m = out + lv->top - (e & 0x7fffffffu);  /* leg l mirrored at m - l*st */
        for (l = 0; l < hl; l++)
            _mm_storeu_pd(d + l * st, _mm_load_pd(src + 2 * l));
        {   /* the middle leg: by the block's flag, no branch */
            const __m128d v = _mm_load_pd(src + 2 * hl);
            const __m128d sg = _mm_and_pd(cj, _mm_castsi128_pd(_mm_set1_epi64x(-(long long)mir)));
            _mm_storeu_pd(mir ? m - hl * st : d + hl * st, _mm_xor_pd(v, sg));
        }
        for (l = hl + 1; l < Rl; l++)
            _mm_storeu_pd(m - l * st, _mm_xor_pd(_mm_load_pd(src + 2 * l), cj));
    }
}
static inline void _zrf_sweep(const _zrf_level_t *lv, double *out, size_t b0, size_t b1)
{
    switch (lv->Rl)
    {
    case 3: _zrf_sweep_r(lv, out, b0, b1, 3); return;
    case 5: _zrf_sweep_r(lv, out, b0, b1, 5); return;
    case 7: _zrf_sweep_r(lv, out, b0, b1, 7); return;
    case 9: _zrf_sweep_r(lv, out, b0, b1, 9); return;
    default: _zrf_sweep_r(lv, out, b0, b1, lv->Rl); return;
    }
}
/* its inverse: the half spectrum -> last-stage blocks [b0, b1) in block order */
static inline __attribute__((always_inline)) void _zrf_gather_r(const _zrf_level_t *lv, const double *in,
                                                                size_t b0, size_t b1, const size_t Rl)
{
    const __m128d cj = _mm_castsi128_pd(_mm_set_epi64x((long long)0x8000000000000000ull, 0));
    const size_t hl = Rl / 2, st = lv->ost;
    double *dst = lv->plane + 2 * lv->D + 2 * Rl * b0;
    const uint32_t *lo = lv->low;
    size_t b, l;
    for (b = b0; b < b1; b++, dst += 2 * Rl)
    {
        const uint32_t e = lo[b];
        const size_t mir = e >> 31;
        const double *d = in + (e & 0x7fffffffu);
        const double *m = in + lv->top - (e & 0x7fffffffu);
        for (l = 0; l < hl; l++)
            _mm_store_pd(dst + 2 * l, _mm_loadu_pd(d + l * st));
        {
            const __m128d v = _mm_loadu_pd(mir ? m - hl * st : d + hl * st);
            const __m128d sg = _mm_and_pd(cj, _mm_castsi128_pd(_mm_set1_epi64x(-(long long)mir)));
            _mm_store_pd(dst + 2 * hl, _mm_xor_pd(v, sg));
        }
        for (l = hl + 1; l < Rl; l++)
            _mm_store_pd(dst + 2 * l, _mm_xor_pd(_mm_loadu_pd(m - l * st), cj));
    }
}
static inline void _zrf_gather(const _zrf_level_t *lv, const double *in, size_t b0, size_t b1)
{
    switch (lv->Rl)
    {
    case 3: _zrf_gather_r(lv, in, b0, b1, 3); return;
    case 5: _zrf_gather_r(lv, in, b0, b1, 5); return;
    case 7: _zrf_gather_r(lv, in, b0, b1, 7); return;
    case 9: _zrf_gather_r(lv, in, b0, b1, 9); return;
    default: _zrf_gather_r(lv, in, b0, b1, lv->Rl); return;
    }
}

/* THE GROUP SWEEP: the tiles at walk entries [k0, k1) -> the half spectrum in
 * output order. The group owns two residue bands: its keys, and P minus
 * them. For a residue c the direct-class tile is the one with Q = c (its
 * legs below Rl/2 sit at c + P*m + l*ns), the mirror-class tile the one with
 * Q = P - c (its mirrored leg Rl-1-l at block bpt-1-m lands at the same bin,
 * conjugated); every walk entry is one or the other for each band. The
 * middle leg exists only while its bin is below N/2. Rl is a constant at
 * the dispatch below. */
static inline __attribute__((always_inline)) void _zrf_sweep_group_r(const _zrf_level_t *lv, double *out,
                                                                     size_t k0, size_t k1, const size_t Rl)
{
    const __m128d cj = _mm_castsi128_pd(_mm_set_epi64x((long long)0x8000000000000000ull, 0));
    const size_t hl = Rl / 2, P = lv->P, bpt = lv->bpt, ns = lv->nst, sc = lv->sc, nG = k1 - k0;
    const double *pl = lv->plane;
    int band;
    for (band = 0; band < 2; band++)
    {
        size_t m, l, i;
        for (m = 0; m < bpt; m++)
        {
            const size_t jd = (size_t)lv->jofm[m] * Rl, jm = (size_t)lv->jofm[bpt - 1 - m] * Rl;
            for (l = 0; l <= hl; l++)
                for (i = 0; i < nG; i++)
                {
                    const size_t k = band ? k1 - 1 - i : k0 + i;
                    const size_t key = lv->tkey[k], c = band ? P - key : key, base = c + P * m;
                    const size_t dir = (size_t)((lv->tQ[k] == key) ^ (size_t)band);   /* 1 = the direct-class tile of c */
                    const size_t src = lv->tbase[k] + (dir ? jd + l : jm + Rl - 1 - l);
                    const __m128d sg = _mm_and_pd(cj, _mm_castsi128_pd(_mm_set1_epi64x(-(long long)(dir ^ 1))));
                    if (l == hl && 2 * base >= ns) continue;   /* the middle leg's bin above N/2 */
                    _mm_storeu_pd(out + sc * (base + l * ns), _mm_xor_pd(_mm_load_pd(pl + 2 * src), sg));
                }
        }
    }
}
static inline void _zrf_sweep_group(const _zrf_level_t *lv, double *out, size_t k0, size_t k1)
{
    switch (lv->Rl)
    {
    case 3: _zrf_sweep_group_r(lv, out, k0, k1, 3); return;
    case 5: _zrf_sweep_group_r(lv, out, k0, k1, 5); return;
    case 7: _zrf_sweep_group_r(lv, out, k0, k1, 7); return;
    case 9: _zrf_sweep_group_r(lv, out, k0, k1, 9); return;
    default: _zrf_sweep_group_r(lv, out, k0, k1, lv->Rl); return;
    }
}

/* the tiles at walk entries [k0, k1), forward. group = 0: each tile's stages
 * then its own sweep, read where its stages left it in L1 (the serial
 * plan: measured faster than groups on one core, 2026-09-30). group = 1:
 * groups of gsz tiles, each tile's stages then the group's sweep in output
 * order (the threaded form: the workers' runs stay disjoint and
 * contiguous). Bitwise the same output either way. */
static inline void _zrf_tiles_fwd(const _zrf_level_t *lv, double *out, size_t k0, size_t k1, int group)
{
    const size_t gs = group ? lv->gsz : 1;
    size_t g0;
    for (g0 = k0; g0 < k1; g0 += gs)
    {
        const size_t g1 = g0 + gs < k1 ? g0 + gs : k1;
        size_t k;
        int i;
        for (k = g0; k < g1; k++)
        {
            const size_t t = lv->tord[k];
            for (i = lv->nwide; i < lv->ns; i++) _ilfd_call(lv->fd, &lv->cf[i], t, lv->plane, lv->plane);
            if (!group) _zrf_sweep(lv, out, (t - lv->t0) * lv->bpt, (t - lv->t0 + 1) * lv->bpt);
        }
        if (group) _zrf_sweep_group(lv, out, g0, g1);
    }
}

/* one level's complex side, forward: the wide stages over blocks 1..h, then
 * (tiled) the tiles, or the rest and one sweep */
static inline void _zrf_level_fwd(const _zrf_level_t *lv, double *out)
{
    int i;
    for (i = 0; i < lv->nwide; i++) _ilfd_call(lv->fd, &lv->cf[i], 1, lv->plane, lv->plane);
    if (lv->tw)
        _zrf_tiles_fwd(lv, out, 0, lv->t1 - lv->t0, 0);
    else
        _zrf_sweep(lv, out, 0, lv->nlb);
}
/* and backward: the gather and the transposed stages, tile by tile, then the
 * wide stages */
static inline void _zrf_level_bwd(const _zrf_level_t *lv, const double *in)
{
    const int ntl = lv->ns - lv->nwide;
    int i;
    if (lv->tw)
    {
        const size_t nt = lv->t1 - lv->t0;
        size_t k;
        for (k = 0; k < nt; k++)
        {
            const size_t t = lv->tord[k];
            _zrf_gather(lv, in, (t - lv->t0) * lv->bpt, (t - lv->t0 + 1) * lv->bpt);
            for (i = 0; i < ntl; i++) _ilfd_call(lv->fd, &lv->ct[i], t, lv->plane, lv->plane);
        }
    }
    else
        _zrf_gather(lv, in, 0, lv->nlb);
    for (i = ntl; i < lv->ns; i++) _ilfd_call(lv->fd, &lv->ct[i], 1, lv->plane, lv->plane);
}

/* r2c: x[N] -> bins 0..N/2 (N + 1 doubles); x == out is legal */
static inline void vfft_zrf_execute_fwd(const vfft_zrf_plan_t *p, const double *x, double *out)
{
    const double *src = x;
    int j;
    for (j = 0; j < p->J; j++)
    {
        const _zrf_level_t *lv = &p->lv[j];
        lv->lf(src, NULL, lv->plane, NULL, NULL, NULL, lv->D, 0, lv->D, 0, lv->D);
        _zrf_level_fwd(lv, out);
        src = lv->plane;
    }
    p->mf(src, NULL, out, NULL, NULL, NULL, 1, 0, p->MJ, 0, 1);
}

/* c2r: bins 0..N/2 -> N reals, unnormalized; in == out is legal */
static inline void vfft_zrf_execute_bwd(const vfft_zrf_plan_t *p, const double *in, double *out)
{
    int j;
    p->mb(in, NULL, p->lv[p->J - 1].plane, NULL, NULL, NULL, p->MJ, 0, 1, 0, 1);
    for (j = p->J - 1; j >= 0; j--)
    {
        const _zrf_level_t *lv = &p->lv[j];
        _zrf_level_bwd(lv, in);
        lv->lb(lv->plane, NULL, j ? p->lv[j - 1].plane : out, NULL, NULL, NULL, lv->D, 0, lv->D, 0, lv->D);
    }
}
#endif

#endif /* VFFT_ZRF_H */
