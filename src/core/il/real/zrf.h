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
 * half spectrum through a position -> destination table (the conjugation is
 * the table's top bit); c2r starts each level with the same table as a
 * gather, runs the scrambled class's TRANSPOSED stages, and ends with the
 * leaf's inverse. Every stage works on (N - D0)/2 complex where the c2c
 * transform of the widened input works on N.
 *
 * Both placements are one pipeline: the first leaf reads the whole input
 * before any sweep writes (r2c), and every gather reads before the last leaf
 * writes (c2r). Unnormalized: c2r(r2c(x)) = N x.
 *
 * The plan inputs are the chain and the split-body form switch; the real
 * door races the chains at create and banks eng=zrf (bridge/real_bridge.h,
 * wisdom2_real_il.h). The level at which the mono takes over is derived: the
 * first run with an rn1 kernel.
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

/* one LEVEL: a real run of Nj = R * D through the leaf, then the c2c stages
 * on blocks 1..h of its plane */
typedef struct {
    int R, h, ns;                 /* the leaf radix, its digits 1..h, the c2c stages (the level's chain - 1) */
    size_t Nj, D;
    vfft_il2p_fn lf, lb;          /* r1c fwd / bwd */
    vfft_ilfd_plan_t *fd;         /* the c2c flat DIT of length Nj, scrambled class: tables and records */
    vfft_ilfd_call_t cf[VFFT_ILFD_MAX_K], ct[VFFT_ILFD_MAX_K]; /* its stage records cut to blocks 1..h:
                                   * forward in stage order, transposed backward in its order */
    double *t2;                   /* a tail stage 1: the group base of block 1, fwd then conj */
    double *plane;                /* (R + 1) * D doubles */
    uint32_t *ord;                /* h * D entries, block order: 2 * destination bin | conj << 31 */
} _zrf_level_t;

typedef struct vfft_zrf_s {
    int N, K, J;                  /* J flat levels, then the mono */
    int R[VFFT_ILFD_MAX_K];
    int nomsz;                    /* 1 = the split-body stage form is off (plan input) */
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
        free(p->lv[j].ord);
    }
    free(p);
}

/* A bound stage record of the c2c plan, cut to the blocks of first digit
 * 1..h and rebased on the level's plane (in = out = the plane, passed as
 * zin / zout; the record runs with t = 1, which steps its bases over block
 * 0). bpt = the stage's blocks per first digit. A tail form at stage 1 packs
 * the R0 blocks of its one group into lanes: there the cut is a COLUMN
 * range (count = h from block 1) and the group base record becomes block
 * 1's (t2one); the pair stream is relative to the first column and stays. */
static inline int _zrf_range_rec(const vfft_ilfd_plan_t *p, int s, size_t h, const double *t2one,
                                 vfft_ilfd_call_t *c)
{
    const size_t bpt = p->nblk[s] / (size_t)p->R[0];
    const size_t recs = (size_t)(p->R[s] - 1) * VFFT_IL_TWREC;
    const size_t w = 2 * p->D[0]; /* one first-digit block, doubles */
    if (c->in_sel != _ILFD_NUL) c->in_sel = _ILFD_ZIN;
    c->out_sel = _ILFD_ZOUT;
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

/* Create from a chain (N = the product, every radix odd). NULL when a level
 * has no leaf, a c2c stage has no transposed twin, or N itself is the
 * mono's (the first level must be flat). */
static inline vfft_zrf_plan_t *vfft_zrf_create(int N, const int *R, int K, int nomsz)
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
    p->N = N; p->K = K; p->nomsz = nomsz != 0;
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
        for (s = 1; s < Kj; s++)
        {
            lv->cf[s - 1] = fd->cf[s];
            lv->ct[Kj - 1 - s] = fd->ct[Kj - 1 - s];
            if (!_zrf_range_rec(fd, s, (size_t)lv->h, lv->t2, &lv->cf[s - 1]) ||
                !_zrf_range_rec(fd, s, (size_t)lv->h, lv->t2 ? lv->t2 + 8 : 0, &lv->ct[Kj - 1 - s]))
            { vfft_zrf_destroy(p); return 0; }
        }
        lv->plane = (double *)vfft_aligned_alloc((((size_t)lv->R + 1) * lv->D + 8) * sizeof(double));
        lv->ord = (uint32_t *)malloc((size_t)lv->h * lv->D * sizeof(uint32_t));
        if (!lv->plane || !lv->ord) { vfft_zrf_destroy(p); return 0; }
        memset(lv->plane, 0, (((size_t)lv->R + 1) * lv->D + 8) * sizeof(double));
        {   /* block order -> the half spectrum: position b*Rl + l of the plane holds the
             * level's bin natbase[b] + l*Nj/Rl, the transform's bin that times N/Nj */
            const size_t Rl = (size_t)R[K - 1], nstride = Nj / Rl, M = (size_t)N / Nj;
            size_t pos;
            for (pos = lv->D; pos < ((size_t)lv->h + 1) * lv->D; pos++)
            {
                const size_t f = (fd->natbase[pos / Rl] + (pos % Rl) * nstride) * M;
                lv->ord[pos - lv->D] = 2 * f <= (size_t)N ? (uint32_t)(2 * f)
                                                          : ((uint32_t)(2 * ((size_t)N - f)) | 0x80000000u);
            }
        }
        Nj = lv->D;
    }
    if (j < 1 || j >= K + 1 || Nj > (size_t)VFFT_ZRM_MAX_N) { vfft_zrf_destroy(p); return 0; }
    p->J = j; p->NJ = Nj; p->MJ = (size_t)N / Nj;
    p->mf = vfft_zrm_fn((int)Nj, 0);
    p->mb = vfft_zrm_fn((int)Nj, 1);
    if (!p->mf || !p->mb) { vfft_zrf_destroy(p); return 0; }
    return p;
#else
    (void)N; (void)R; (void)K; (void)nomsz;
    return 0;
#endif
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
/* the order sweep: blocks 1..h of the plane -> the half spectrum */
static inline void _zrf_sweep(const _zrf_level_t *lv, double *out)
{
    static const union { uint64_t u[4]; double d[4]; } cj = { { 0, 0, 0, 0x8000000000000000ull } };
    const double *src = lv->plane + 2 * lv->D;
    const uint32_t *t = lv->ord;
    const size_t n = (size_t)lv->h * lv->D;
    size_t i;
    for (i = 0; i < n; i++)
    {
        const uint32_t e = t[i];
        const __m128d v = _mm_xor_pd(_mm_load_pd(src + 2 * i), _mm_loadu_pd(cj.d + 2 * (e >> 31)));
        _mm_storeu_pd(out + (e & 0x7fffffffu), v);
    }
}
/* its inverse: the half spectrum -> blocks 1..h in block order */
static inline void _zrf_gather(const _zrf_level_t *lv, const double *in)
{
    static const union { uint64_t u[4]; double d[4]; } cj = { { 0, 0, 0, 0x8000000000000000ull } };
    double *dst = lv->plane + 2 * lv->D;
    const uint32_t *t = lv->ord;
    const size_t n = (size_t)lv->h * lv->D;
    size_t i;
    for (i = 0; i < n; i++)
    {
        const uint32_t e = t[i];
        const __m128d v = _mm_xor_pd(_mm_loadu_pd(in + (e & 0x7fffffffu)), _mm_loadu_pd(cj.d + 2 * (e >> 31)));
        _mm_store_pd(dst + 2 * i, v);
    }
}

/* r2c: x[N] -> bins 0..N/2 (N + 1 doubles); x == out is legal */
static inline void vfft_zrf_execute_fwd(const vfft_zrf_plan_t *p, const double *x, double *out)
{
    const double *src = x;
    int j, i;
    for (j = 0; j < p->J; j++)
    {
        const _zrf_level_t *lv = &p->lv[j];
        lv->lf(src, NULL, lv->plane, NULL, NULL, NULL, lv->D, 0, lv->D, 0, lv->D);
        for (i = 0; i < lv->ns; i++) _ilfd_call(lv->fd, &lv->cf[i], 1, lv->plane, lv->plane);
        _zrf_sweep(lv, out);
        src = lv->plane;
    }
    p->mf(src, NULL, out, NULL, NULL, NULL, 1, 0, p->MJ, 0, 1);
}

/* c2r: bins 0..N/2 -> N reals, unnormalized; in == out is legal */
static inline void vfft_zrf_execute_bwd(const vfft_zrf_plan_t *p, const double *in, double *out)
{
    int j, i;
    p->mb(in, NULL, p->lv[p->J - 1].plane, NULL, NULL, NULL, p->MJ, 0, 1, 0, 1);
    for (j = p->J - 1; j >= 0; j--)
    {
        const _zrf_level_t *lv = &p->lv[j];
        _zrf_gather(lv, in);
        for (i = 0; i < lv->ns; i++) _ilfd_call(lv->fd, &lv->ct[i], 1, lv->plane, lv->plane);
        lv->lb(lv->plane, NULL, j ? p->lv[j - 1].plane : out, NULL, NULL, NULL, lv->D, 0, lv->D, 0, lv->D);
    }
}
#endif

#endif /* VFFT_ZRF_H */
