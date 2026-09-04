/* il_flatdit.h — the FLAT mixed-radix DIT chain, un-turned, v0 STRUCTURE
 * CHECK (2026-09-04): MKL's generic-N shape on shipped pure-IL kinds,
 * built to validate the algebra BEFORE the emitter kind it motivates.
 *
 *   N = R[0]*...*R[K-1]. Natural input, same-slot stages, natural output.
 *   stage 0: the plain leaf (n1c) — R[0] legs at stride D0 = N/R[0],
 *            count D0 (full lanes), UN-turned store: output digit p keeps
 *            the leg slot p*D0.
 *   stage s: legs at stride D_s = N/(R[0]..R[s]), span L_s = R[s]*D_s,
 *            count = D_s — the run SHRINKS (long early, short late). The
 *            PRE-twiddle w_{L_s}^(l*Q_s) depends only on the SLOW digits
 *            already produced, Q_s = p + R0*(q1 + R1*(...)): constant
 *            across the run. v0 drives it with the t2 kind (pre-twiddle,
 *            per column-pair records) and REPEATS the block's record set
 *            for every pair — ~4N doubles per stage, a probe-only cost.
 *            The real stage kind is t2c with the pre-twiddle placement
 *            (per-block broadcast records: (R-1) per block), an emitter
 *            gate away; the late stages then take the gen2 policy.
 *   last stage (D = 1): legs adjacent, count 1 (the VEX-128 tail; the
 *            two-group kind retires it). Its stores are REDIRECTED to
 *            natural order: OLs = N/R[K-1], out base = the block's natural
 *            index (block-affine) — from the staging plane into zout, so
 *            the output is natural with no ordering pass.
 *   v0 = forward only, out of place, single thread. */
#ifndef VFFT_IL_FLATDIT_H
#define VFFT_IL_FLATDIT_H

#include "il2p.h"
#include "il2d_cols.h"

#define VFFT_ILFD_MAX_K 10

typedef struct {
    int N, K;
    int R[VFFT_ILFD_MAX_K];
    size_t D[VFFT_ILFD_MAX_K];        /* leg stride / run at stage s */
    size_t nblk[VFFT_ILFD_MAX_K];     /* blocks at stage s */
    vfft_il2p_fn lf;                  /* n1c fwd: the plain leaf */
    vfft_il2p_fn f[VFFT_ILFD_MAX_K];  /* t2 fwd (pre-twiddle) */
    double *tf[VFFT_ILFD_MAX_K];      /* per block: ceil(D/2) x (R-1) records */
    size_t *natbase;                  /* last stage: block -> natural index */
    double *stg;                      /* 2N staging plane */
} vfft_ilfd_plan_t;

static inline void vfft_ilfd_destroy(vfft_ilfd_plan_t *p)
{
    int s;
    if (!p) return;
    for (s = 0; s < VFFT_ILFD_MAX_K; s++) VFFT_IL2P_FREE(p->tf[s]);
    free(p->natbase);
    VFFT_IL2P_FREE(p->stg);
    free(p);
}

/* block b at stage s enumerates the slow slots (p, q1, .., q_{s-1}) with p
 * most significant; Q = p + R0*(q1 + R1*(q2 + ...)) is its natural index */
static inline size_t _ilfd_block_Q(const vfft_ilfd_plan_t *p, int s, size_t b)
{
    size_t dig[VFFT_ILFD_MAX_K], Q = 0, W = 1, rem = b;
    int j;
    for (j = s - 1; j >= 0; j--) { dig[j] = rem % (size_t)p->R[j]; rem /= (size_t)p->R[j]; }
    for (j = 0; j < s; j++) { Q += dig[j] * W; W *= (size_t)p->R[j]; }
    return Q;
}

static inline vfft_ilfd_plan_t *vfft_ilfd_create_chain(int N, const int *R, int K)
{
    vfft_ilfd_plan_t *p;
    long prod = 1;
    int s;
    size_t D;
    if (N < 4 || K < 2 || K > VFFT_ILFD_MAX_K) return 0;
    for (s = 0; s < K; s++) { if (R[s] < 2) return 0; prod *= R[s]; }
    if (prod != (long)N) return 0;
    p = (vfft_ilfd_plan_t *)calloc(1, sizeof(*p));
    if (!p) return 0;
    p->N = N; p->K = K;
    for (s = 0; s < K; s++) p->R[s] = R[s];
    {   /* the plain leaf: n1c (natural in/out, alias-tolerant) */
        vfft_il2p_fn ff[1], fb[1];
        int r1[1]; r1[0] = R[0];
        if (!_il2d_resolve(r1, 1, ff, fb)) { vfft_ilfd_destroy(p); return 0; }
        p->lf = ff[0];
    }
    D = (size_t)N;
    for (s = 0; s < K; s++) {
        D /= (size_t)R[s];
        p->D[s] = D;
        p->nblk[s] = (size_t)N / ((size_t)R[s] * D);
        if (s >= 1) {
            const size_t nb = p->nblk[s];
            /* the twiddle modulus is the product of the radices processed so
             * far INCLUDING this stage = N / D_s (il3p's stage-B convention:
             * B*R2), not the block span R_s*D_s. */
            const size_t L = (size_t)N / D;
            const size_t npair = (D + 1) / 2;                  /* ceiling pairs */
            const size_t recs_blk = npair * (size_t)(R[s] - 1);
            double *tf;
            size_t bi, pp;
            int l, j;
            p->f[s] = vfft_il2p_mid_fn(R[s], 0);               /* t2 fwd: PRE-twiddle */
            if (!p->f[s]) { vfft_ilfd_destroy(p); return 0; }
            tf = (double *)VFFT_IL2P_ALLOC(nb * recs_blk * 8 * sizeof(double));
            if (!tf) { vfft_ilfd_destroy(p); return 0; }
            for (bi = 0; bi < nb; bi++) {
                const size_t Q = _ilfd_block_Q(p, s, bi);
                for (l = 1; l < R[s]; l++) {
                    const double a = -2.0 * VFFT_IL2P_PI * (double)((size_t)l * Q % L) / (double)L;
                    const double c = cos(a), sn = sin(a);
                    for (pp = 0; pp < npair; pp++) {
                        double *rf = tf + (bi * recs_blk + pp * (size_t)(R[s] - 1) + (size_t)(l - 1)) * 8;
                        for (j = 0; j < 4; j++) rf[j] = c;
                        rf[4] = -sn; rf[5] = sn; rf[6] = -sn; rf[7] = sn;  /* VTW2 sign-folded */
                    }
                }
            }
            p->tf[s] = tf;
        }
    }
    {   /* the last stage's natural redirection */
        const int s = K - 1;
        size_t bi;
        p->natbase = (size_t *)malloc(p->nblk[s] * sizeof(size_t));
        if (!p->natbase) { vfft_ilfd_destroy(p); return 0; }
        for (bi = 0; bi < p->nblk[s]; bi++) p->natbase[bi] = _ilfd_block_Q(p, s, bi);
    }
    p->stg = (double *)VFFT_IL2P_ALLOC(2u * (size_t)N * sizeof(double));
    if (!p->stg) { vfft_ilfd_destroy(p); return 0; }
    return p;
}

/* SEED: small radices, 9 first (the raced winners' shape) — a seed. */
static inline int vfft_ilfd_default_chain(int N, int *R, int *K)
{
    static const int POOL[] = { 9, 7, 5, 3, 25, 27, 21, 15, 13, 11, 8, 4, 16 };
    int rem = N, k = 0, i;
    while (rem > 1 && k < VFFT_ILFD_MAX_K) {
        int hit = 0;
        for (i = 0; i < (int)(sizeof POOL / sizeof POOL[0]); i++)
            if (rem % POOL[i] == 0 && vfft_il2p_mid_fn(POOL[i], 0)) { R[k++] = POOL[i]; rem /= POOL[i]; hit = 1; break; }
        if (!hit) return 0;
    }
    if (rem != 1 || k < 2) return 0;
    *K = k;
    return 1;
}
static inline vfft_ilfd_plan_t *vfft_ilfd_create(int N)
{
    int R[VFFT_ILFD_MAX_K], K = 0;
    if (!vfft_ilfd_default_chain(N, R, &K)) return 0;
    return vfft_ilfd_create_chain(N, R, K);
}

/* fwd: leaf zin -> stg; stages 1..K-2 in place on stg; the last stage
 * scatters stg -> zout in natural order. */
static inline void vfft_ilfd_execute_fwd(const vfft_ilfd_plan_t *p,
                                         const double *zin, double *zout)
{
    const size_t N = (size_t)p->N;
    double *stg = p->stg;
    int s;
    p->lf(zin, 0, stg, 0, 0, 0, p->D[0], 0, p->D[0], 0, p->D[0]);
    for (s = 1; s < p->K; s++) {
        const size_t D = p->D[s], L = (size_t)p->R[s] * D, nb = p->nblk[s];
        const size_t recs_blk = ((D + 1) / 2) * (size_t)(p->R[s] - 1);
        const size_t nstride = N / (size_t)p->R[s];
        size_t bi;
        for (bi = 0; bi < nb; bi++) {
            const double *blk = stg + 2 * bi * L;
            const double *tw = p->tf[s] + bi * recs_blk * 8;
            if (s < p->K - 1)
                p->f[s](blk, 0, stg + 2 * bi * L, 0, tw, 0, D, 0, D, 0, D);
            else
                p->f[s](blk, 0, zout + 2 * p->natbase[bi], 0, tw, 0, D, 0, nstride, 0, D);
        }
    }
}

#endif /* VFFT_IL_FLATDIT_H */
