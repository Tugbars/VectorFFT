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
 * Two kernels are the engine's own (codelets/zil/<isa>/real/flat/):
 *   r1c       the leaf (generator/lib/gen/real_il.ml): real legs in, digit 0
 *             real at plane[k], digit p complex at block p of the c2c plane
 *             (complex p*D + k), so no pointer is shifted and no block is
 *             copied. A level's plane is (R+1)*D doubles; the second half of
 *             block 0 is unused.
 *   t2csgh    the LAST stage (c2c_il.ml, the t2csgn group-loop tail with a
 *             Hermitian store edge). Block p holds the bins congruent to p mod
 *             R0; the r2c wants bins 0..N/2, a bin above N/2 as conj at N - f.
 *             The stage's legs are one block's bins (leg l = natural base +
 *             l*Nj/Rl, scaled by N/Nj to the caller's bins), so the kernel
 *             writes legs below Rl/2 to their places, legs above Rl/2
 *             conjugated at the mirror, and the middle leg by its bin --
 *             straight into the half spectrum, no order pass. Its backward
 *             twin t2csght reads the half spectrum the same way into the
 *             plane's block order (the scrambled class's transposed first
 *             stage). Every stage works on (N - D0)/2 complex where the c2c
 *             transform of the widened input works on N; the order pass
 *             that the sweep was (12% of the transform at N <= 2048,
 *             2026-09-30) is gone.
 *
 * THE TILE AXIS is the c2c flat DIT's (its validator is the law): a level
 * with a tile width runs its stages from the cut on depth-first per tile --
 * one block of the cut stage at a time -- so a tile's stages meet it in L1.
 * The plan input is a width BUDGET; each level takes its widest legal stage
 * span within it. A tile holds the bins congruent to its slow digits' index
 * Q modulo P = the tiles of the whole plane: a comb across the half spectrum,
 * its direct legs at residue Q and its mirrored legs at residue P - Q. The
 * tiles are walked in ascending min(Q, P - Q), so consecutive tiles fill the
 * same output lines while those are still in cache, and a contiguous range
 * of the walk owns a contiguous set of residues (the threaded form's workers,
 * zrf_mt.h, write disjoint lines).
 *
 * Both placements are one pipeline: the first leaf reads the whole input
 * before any last stage writes (r2c), and every level's first stage reads
 * the spectrum before the last leaf writes (c2r). Unnormalized: c2r(r2c(x))
 * = N x.
 *
 * The plan inputs are the chain, the split-body form switch and the tile
 * budget; the odd real race sweeps them at create and banks eng=zrf
 * (bridge/real_bridge.h, wisdom2_real_il.h). The level at which the mono
 * takes over is derived: the first run with an rn1 kernel. The threaded
 * forms are zrf_mt.h.
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
/* the Hermitian last stage at radix R: fwd = t2csgh, bwd = t2csght */
static inline vfft_il2p_fn vfft_zrf_last_fn(int R, int bwd)
{
    switch (R)
    {
#if defined(VFFT_IL_T2CSGH_FWD_RADICES) && defined(VFFT_IL_T2CSGHT_BWD_RADICES) && VFFT_IL_VW == 4
#define C(R_) case R_: return bwd ? VFFT_IL_SYM(radix##R_##_z_t2csght_bwd) : VFFT_IL_SYM(radix##R_##_z_t2csgh_fwd);
    VFFT_IL_T2CSGH_FWD_RADICES(C)
#undef C
#endif
    default: (void)bwd; return 0;
    }
}

/* one LEVEL: a real run of Nj = R * D through the leaf, then the c2c stages
 * on blocks 1..h of its plane, the last of them into the half spectrum */
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
     * are the tiled ones. cf[ns-1] and ct[0] are the HERMITIAN last stage:
     * their spectrum side is the caller's buffer, absolute through hb, and
     * the mirror base is passed at the call (_zrf_call_herm). */
    vfft_ilfd_call_t cf[VFFT_ILFD_MAX_K], ct[VFFT_ILFD_MAX_K];
    int nwide;
    size_t tw, t0, t1;            /* the tile width (complex; 0 = untiled), its tiles in blocks 1..h
                                   * (the plane's tile numbering) */
    double *t2;                   /* a tail stage 1: the group base of block 1, fwd then conj */
    double *plane;                /* (R + 1) * D doubles */
    size_t *hb;                   /* per last-stage block: its natural base scaled to the caller's bins */
    uint32_t *tord;               /* the tiles of [t0, t1) in walk order: ascending min(Q, P - Q) */
    size_t n2;                    /* 2N: the mirror base's offset from the caller's buffer, doubles */
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
        free(p->lv[j].hb);
        free(p->lv[j].tord);
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
 * no transposed twin, the last stage is not the group-loop form, or N itself
 * is the mono's (the first level must be flat). */
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
        lv->n2 = 2 * (size_t)N;
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
        {   /* THE HERMITIAN LAST STAGE: the group-loop tail's records, their spectrum side
             * turned absolute -- the base table scaled to the caller's bins, the leg
             * stride and column pitch there (a level's bin f is the transform's bin
             * M f, M = N/Nj), the mirror passed at the call. */
            const int sl = Kj - 1;
            const size_t Rl = (size_t)fd->R[sl], nb = fd->nblk[sl], M = (size_t)N / Nj, nst = Nj / Rl;
            vfft_ilfd_call_t *cf = &lv->cf[sl - 1], *ct = &lv->ct[0];
            size_t W = 1, b;
            int q;
            for (q = 0; q < sl - 1; q++) W *= (size_t)fd->R[q];   /* the weight of digit q_{sl-1} */
            if (!fd->gl[sl] || cf->op != _ILFD_ONE || !cf->a1 || ct->op != _ILFD_ONE || !ct->a1)
            { vfft_zrf_destroy(p); return 0; }
            cf->fn = vfft_zrf_last_fn((int)Rl, 0);
            ct->fn = vfft_zrf_last_fn((int)Rl, 1);
            if (!cf->fn || !ct->fn) { vfft_zrf_destroy(p); return 0; }
            lv->hb = (size_t *)malloc(nb * sizeof(size_t));
            if (!lv->hb) { vfft_zrf_destroy(p); return 0; }
            for (b = 0; b < nb; b++) lv->hb[b] = M * fd->natbase[b];
            cf->a1 = (const double *)(sl == 1 ? lv->hb + 1 : lv->hb);   /* the column range at stage 1 starts at block 1 */
            cf->a3 = 0;
            cf->OLs = M * nst; cf->OGs = M * W;
            cf->out_sel = _ILFD_ZOUT; cf->out_tstep = 0;                /* the spectrum: absolute */
            ct->a1 = cf->a1;
            ct->a3 = 0;
            ct->Ls = M * nst; ct->OLs = M * W;                          /* the spectrum's strides on the load side */
            ct->out_tstep = ct->in_tstep; ct->in_tstep = 0;             /* the plane keeps the relative step */
            ct->in_sel = _ILFD_ZIN; ct->out_sel = _ILFD_ZOUT;
        }
        if (lv->tw)
        {
            lv->t0 = lv->D / lv->tw; lv->t1 = ((size_t)lv->h + 1) * lv->D / lv->tw;
        }
        lv->plane = (double *)vfft_aligned_alloc((((size_t)lv->R + 1) * lv->D + 8) * sizeof(double));
        if (!lv->plane) { vfft_zrf_destroy(p); return 0; }
        memset(lv->plane, 0, (((size_t)lv->R + 1) * lv->D + 8) * sizeof(double));
        if (lv->tw)
        {   /* the walk order: a counting sort on min(Q, P - Q) (distinct: P - Q is a tile of the
             * mirrored blocks, never of blocks 1..h) */
            const size_t P = Nj / lv->tw, nt = lv->t1 - lv->t0;
            uint32_t *at = (uint32_t *)malloc((P + 1) * sizeof(uint32_t));
            size_t k = 0, q2, t;
            lv->tord = (uint32_t *)malloc(nt * sizeof(uint32_t));
            if (!at || !lv->tord) { free(at); vfft_zrf_destroy(p); return 0; }
            memset(at, 0xff, (P + 1) * sizeof(uint32_t));
            for (t = lv->t0; t < lv->t1; t++)
            {
                const size_t Q = _ilfd_block_Q(fd, fd->tcut, t);
                at[Q < P - Q ? Q : P - Q] = (uint32_t)t;
            }
            for (q2 = 0; q2 <= P; q2++) if (at[q2] != 0xffffffffu) lv->tord[k++] = at[q2];
            free(at);
            if (k != nt) { vfft_zrf_destroy(p); return 0; }
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
/* the Hermitian last stage's call: as _ilfd_call's one-call form, the
 * spectrum side (zout forward, zin backward) the caller's buffer and its
 * mirror base passed in the zout_unused slot */
static inline void _zrf_call_herm(const vfft_ilfd_call_t *c, size_t t, const double *zin, double *zout,
                                  double *mirror)
{
    const double *in = zin + t * c->in_tstep;
    double *out = zout + t * c->out_tstep;
    const double *tw = c->tw ? c->tw + t * c->tw_tstep : 0;
    const double *t2 = c->t2 ? c->t2 + t * c->t2_tstep : 0;
    const double *a1 = c->a1 + t * c->a1_tstep;
    c->fn(in, a1, out, mirror, tw, t2, c->Ls, c->Gs, c->OLs, c->OGs, c->count);
}

/* the tiles at walk entries [k0, k1), forward: each tile's stages depth-first,
 * the last of them into the half spectrum */
static inline void _zrf_tiles_fwd(const _zrf_level_t *lv, double *out, size_t k0, size_t k1)
{
    size_t k;
    int i;
    for (k = k0; k < k1; k++)
    {
        const size_t t = lv->tord[k];
        for (i = lv->nwide; i < lv->ns - 1; i++) _ilfd_call(lv->fd, &lv->cf[i], t, lv->plane, lv->plane);
        _zrf_call_herm(&lv->cf[lv->ns - 1], t, lv->plane, out, out + lv->n2);
    }
}
/* and backward: each tile's first stage from the half spectrum, then its
 * transposed stages */
static inline void _zrf_tiles_bwd(const _zrf_level_t *lv, const double *in, size_t k0, size_t k1)
{
    const int ntl = lv->ns - lv->nwide;
    size_t k;
    int i;
    for (k = k0; k < k1; k++)
    {
        const size_t t = lv->tord[k];
        _zrf_call_herm(&lv->ct[0], t, in, lv->plane, (double *)in + lv->n2);
        for (i = 1; i < ntl; i++) _ilfd_call(lv->fd, &lv->ct[i], t, lv->plane, lv->plane);
    }
}

/* one level's complex side, forward: the wide stages over blocks 1..h, then
 * the tiles (or, untiled, the last stage over the range) */
static inline void _zrf_level_fwd(const _zrf_level_t *lv, double *out)
{
    const int nw = lv->nwide < lv->ns ? lv->nwide : lv->ns - 1;
    int i;
    for (i = 0; i < nw; i++) _ilfd_call(lv->fd, &lv->cf[i], 1, lv->plane, lv->plane);
    if (lv->tw)
        _zrf_tiles_fwd(lv, out, 0, lv->t1 - lv->t0);
    else
        _zrf_call_herm(&lv->cf[lv->ns - 1], 1, lv->plane, out, out + lv->n2);
}
/* and backward: the first stage from the half spectrum and the transposed
 * stages, tile by tile (or over the range), then the wide stages */
static inline void _zrf_level_bwd(const _zrf_level_t *lv, const double *in)
{
    const int ntl = lv->ns - lv->nwide;
    int i;
    if (lv->tw)
    {
        _zrf_tiles_bwd(lv, in, 0, lv->t1 - lv->t0);
        for (i = ntl; i < lv->ns; i++) _ilfd_call(lv->fd, &lv->ct[i], 1, lv->plane, lv->plane);
    }
    else
    {
        _zrf_call_herm(&lv->ct[0], 1, in, lv->plane, (double *)in + lv->n2);
        for (i = 1; i < lv->ns; i++) _ilfd_call(lv->fd, &lv->ct[i], 1, lv->plane, lv->plane);
    }
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
