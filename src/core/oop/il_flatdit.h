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
/* stages whose run D is at most this many columns run on the column-stride
 * kind: a "column" becomes one block, so the lanes fill from blocks. */
#ifndef VFFT_ILFD_TAIL_D
#define VFFT_ILFD_TAIL_D 4
#endif

typedef struct {
    int N, K;
    int R[VFFT_ILFD_MAX_K];
    size_t D[VFFT_ILFD_MAX_K];        /* leg stride / run at stage s */
    size_t nblk[VFFT_ILFD_MAX_K];     /* blocks at stage s */
    vfft_il2p_fn lf;                  /* n1c fwd: the plain leaf */
    vfft_il2p_fn f[VFFT_ILFD_MAX_K];  /* t2cp fwd (pre-twiddle, per-digit records) */
    vfft_il2p_fn fcs[VFFT_ILFD_MAX_K];/* t2cs / t2csg fwd for the TAIL stages (D <= VFFT_ILFD_TAIL_D) */
    int tail[VFFT_ILFD_MAX_K];        /* 1 = t2cs (per-pair stream), 2 = t2csg (generated) */
    double *t2g[VFFT_ILFD_MAX_K];     /* gen2: per-group broadcast records (T2), 8 doubles each */
    vfft_il2p_fn fgl[VFFT_ILFD_MAX_K];/* t2csgn fwd: the LAST stage's group loop in-kernel (non-null = available) */
    int gl[VFFT_ILFD_MAX_K];          /* 1 = run the last stage on t2csgn (default when available;
                                       * VFFT_ILFD_NO_GL=1 at create turns it off; A/B flips it) */
    size_t *gorder;                   /* last stage: the groups in ascending natural-base order
                                       * (t2csgn walks them so consecutive groups fill adjacent
                                       * output lines); NULL = block order */
    int gord;                         /* 1 = pass gorder to t2csgn (default; VFFT_ILFD_NO_GORD=1
                                       * turns it off; A/B flips it) */
    vfft_il2p_fn fz[VFFT_ILFD_MAX_K]; /* msz fwd (split body, IL edges, unordered lanes) */
    double *tz[VFFT_ILFD_MAX_K];      /* msz: per block (R-1) [c x4][s x4] records (plain sin);
                                       * non-null = the stage is msz-ELIGIBLE (s < K-1; any run
                                       * since the §3 odd-count arms, 2026-09-05) */
    int msz[VFFT_ILFD_MAX_K];         /* 1 = run the stage on msz (default where eligible;
                                       * VFFT_ILFD_NO_MSZ=1 at create turns it off; A/B flips it) */
    double *tf[VFFT_ILFD_MAX_K];      /* t2cp: per block (R-1) broadcast records;
                                       * t2cs: per GROUP, per block-pair, (R-1) VTW2 records */
    size_t *natbase;                  /* last stage: block -> natural index */
    double *stg;                      /* 2N staging plane */
} vfft_ilfd_plan_t;

static inline void vfft_ilfd_destroy(vfft_ilfd_plan_t *p)
{
    int s;
    if (!p) return;
    for (s = 0; s < VFFT_ILFD_MAX_K; s++) { VFFT_IL2P_FREE(p->tf[s]); VFFT_IL2P_FREE(p->t2g[s]); VFFT_IL2P_FREE(p->tz[s]); }
    free(p->gorder);
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
            const size_t recs_blk = (size_t)(R[s] - 1);
            double *tf;
            size_t bi;
            int l, lane;
            /* msz (2026-09-05): every non-last stage can run on the split-body
             * kernel (its il_odd_count_tail §3 arms take any run); the records
             * are built alongside the t2cp/tail ones so a probe can A/B the
             * two forms on ONE plan by flipping p->msz[s]. */
            p->msz[s] = 0;
            if (s < K - 1 && vfft_il2p_msz_fn(R[s])) {
                double *tz = (double *)VFFT_IL2P_ALLOC(nb * recs_blk * 8 * sizeof(double));
                if (!tz) { vfft_ilfd_destroy(p); return 0; }
                for (bi = 0; bi < nb; bi++) {
                    const size_t Q = _ilfd_block_Q(p, s, bi);
                    for (l = 1; l < R[s]; l++) {
                        const double a = -2.0 * VFFT_IL2P_PI * (double)((size_t)l * Q % L) / (double)L;
                        const double c = cos(a), sn = sin(a);
                        double *rf = tz + (bi * recs_blk + (size_t)(l - 1)) * 8;
                        for (lane = 0; lane < 4; lane++) { rf[lane] = c; rf[4 + lane] = sn; }
                    }
                }
                p->tz[s] = tz;
                p->fz[s] = vfft_il2p_msz_fn(R[s]);
                p->msz[s] = !getenv("VFFT_ILFD_NO_MSZ");
            }
            p->tail[s] = 0;
            if (D <= (size_t)VFFT_ILFD_TAIL_D) {
                if (vfft_il2p_t2csg_fn(R[s]) && !getenv("VFFT_ILFD_NO_GEN2")) p->tail[s] = 2;
                else if (vfft_il2p_t2cs_fn(R[s])) p->tail[s] = 1;
            }
            if (p->tail[s] == 2) {
                /* t2csg: the stream is GENERATED. T1 (tw_re, per stage): one
                 * VTW2 pair record per block pair of a group — lane j =
                 * w_L^(W*(2pp+j)), the group-internal step (Q steps by W
                 * from block to block). T2 (tw_im, per group): the group's
                 * base w_L^(Q(g*G)) as one broadcast record. The kernel
                 * forms W^1 = T1 x T2 and derives legs 2..R-1 itself. */
                const size_t G = (size_t)R[s - 1], ngrp = nb / G, npair = (G + 1) / 2;
                size_t g, pp, W = 1;
                int j;
                for (j = 0; j < s - 1; j++) W *= (size_t)R[j];
                p->fcs[s] = vfft_il2p_t2csg_fn(R[s]);
                p->fgl[s] = (s == K - 1) ? vfft_il2p_t2csgn_fn(R[s]) : 0;
                p->gl[s] = (p->fgl[s] != 0) && !getenv("VFFT_ILFD_NO_GL");
                tf = (double *)VFFT_IL2P_ALLOC(npair * 8 * sizeof(double));
                p->t2g[s] = (double *)VFFT_IL2P_ALLOC(ngrp * 8 * sizeof(double));
                if (!tf || !p->t2g[s]) { VFFT_IL2P_FREE(tf); vfft_ilfd_destroy(p); return 0; }
                for (pp = 0; pp < npair; pp++) {
                    double *rf = tf + pp * 8;
                    for (j = 0; j < 2; j++) {
                        const size_t jj = 2 * pp + (size_t)j;
                        const double a = -2.0 * VFFT_IL2P_PI * (double)((W * jj) % L) / (double)L;
                        const double c = cos(a), sn = sin(a);
                        rf[2 * j] = c; rf[2 * j + 1] = c;
                        rf[4 + 2 * j] = -sn; rf[4 + 2 * j + 1] = sn;
                    }
                }
                for (g = 0; g < ngrp; g++) {
                    const size_t Q = _ilfd_block_Q(p, s, g * G);
                    const double a = -2.0 * VFFT_IL2P_PI * (double)(Q % L) / (double)L;
                    const double c = cos(a), sn = sin(a);
                    double *rg = p->t2g[s] + g * 8;
                    for (lane = 0; lane < 4; lane++) { rg[lane] = c; rg[4 + lane] = (lane & 1) ? sn : -sn; }
                }
                p->tf[s] = tf;
            } else if (!p->tail[s]) {
                /* t2cp: ONE digit per call => (R-1) broadcast records per
                 * block (the 2D per-digit record: [c x4][sign-folded s x4]) */
                p->f[s] = vfft_il2p_t2cp_fn(R[s]);
                if (!p->f[s]) { vfft_ilfd_destroy(p); return 0; }
                tf = (double *)VFFT_IL2P_ALLOC(nb * recs_blk * 8 * sizeof(double));
                if (!tf) { vfft_ilfd_destroy(p); return 0; }
                for (bi = 0; bi < nb; bi++) {
                    const size_t Q = _ilfd_block_Q(p, s, bi);
                    for (l = 1; l < R[s]; l++) {
                        const double a = -2.0 * VFFT_IL2P_PI * (double)((size_t)l * Q % L) / (double)L;
                        const double c = cos(a), sn = sin(a);
                        double *rf = tf + (bi * recs_blk + (size_t)(l - 1)) * 8;
                        for (lane = 0; lane < 4; lane++) {
                            rf[lane] = c;
                            rf[4 + lane] = (lane & 1) ? sn : -sn;
                        }
                    }
                }
                p->tf[s] = tf;
            } else {
                /* t2cs: columns = blocks. A GROUP = the R[s-1] consecutive
                 * blocks sharing every slow digit but q_{s-1} (so a group's
                 * natural bases step by a constant W). Per group: ceil(G/2)
                 * block PAIRS x (R-1) VTW2 pair records [c0,c0,c1,c1]
                 * [-s0,+s0,-s1,+s1] (lane j = block 2pp+j of the group). */
                const size_t G = (size_t)R[s - 1], ngrp = nb / G, npair = (G + 1) / 2;
                const size_t recs_grp = npair * recs_blk;
                size_t g, pp;
                int j;
                p->fcs[s] = vfft_il2p_t2cs_fn(R[s]);
                tf = (double *)VFFT_IL2P_ALLOC(ngrp * recs_grp * 8 * sizeof(double));
                if (!tf) { vfft_ilfd_destroy(p); return 0; }
                for (g = 0; g < ngrp; g++)
                    for (pp = 0; pp < npair; pp++)
                        for (l = 1; l < R[s]; l++) {
                            double *rf = tf + (g * recs_grp + pp * recs_blk + (size_t)(l - 1)) * 8;
                            for (j = 0; j < 2; j++) {
                                const size_t b2 = g * G + 2 * pp + (size_t)j;
                                const size_t Q = (b2 < nb) ? _ilfd_block_Q(p, s, b2) : 0;
                                const double a = -2.0 * VFFT_IL2P_PI * (double)((size_t)l * Q % L) / (double)L;
                                const double c = cos(a), sn = sin(a);
                                rf[2 * j] = c; rf[2 * j + 1] = c;
                                rf[4 + 2 * j] = -sn; rf[4 + 2 * j + 1] = sn;
                            }
                        }
                p->tf[s] = tf;
            }
        }
    }
    {   /* the last stage's natural redirection */
        const int s = K - 1;
        size_t bi;
        p->natbase = (size_t *)malloc(p->nblk[s] * sizeof(size_t));
        if (!p->natbase) { vfft_ilfd_destroy(p); return 0; }
        for (bi = 0; bi < p->nblk[s]; bi++) p->natbase[bi] = _ilfd_block_Q(p, s, bi);
        /* t2csgn group order = groups sorted by their natural base (a counting
         * sort over the base values: they are distinct and bounded by N). */
        if (s >= 1 && p->fgl[s]) {
            const size_t G = (size_t)p->R[s - 1], ngrp = p->nblk[s] / G;
            size_t *pos = (size_t *)calloc((size_t)N + 1, sizeof(size_t));
            size_t g, n;
            p->gorder = (size_t *)malloc(ngrp * sizeof(size_t));
            if (!pos || !p->gorder) { free(pos); vfft_ilfd_destroy(p); return 0; }
            for (g = 0; g < ngrp; g++) pos[p->natbase[g * G] + 1] = 1;
            for (n = 1; n <= (size_t)N; n++) pos[n] += pos[n - 1];
            for (g = 0; g < ngrp; g++) p->gorder[pos[p->natbase[g * G]]] = g;
            free(pos);
            p->gord = !getenv("VFFT_ILFD_NO_GORD");
        }
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
            if (rem % POOL[i] == 0 && vfft_il2p_t2cp_fn(POOL[i])) { R[k++] = POOL[i]; rem /= POOL[i]; hit = 1; break; }
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

/* ONE stage: s == 0 is the leaf (zin -> stg); 1 <= s < K-1 in place on stg;
 * s == K-1 scatters stg -> zout in natural order. Exposed so a probe can
 * time stages one at a time through the exact serving path. */
static inline void vfft_ilfd_stage(const vfft_ilfd_plan_t *p, int s,
                                   const double *zin, double *zout)
{
    const size_t N = (size_t)p->N;
    double *stg = p->stg;
    if (s == 0) {
        p->lf(zin, 0, stg, 0, 0, 0, p->D[0], 0, p->D[0], 0, p->D[0]);
        return;
    }
    {
        const size_t D = p->D[s], L = (size_t)p->R[s] * D, nb = p->nblk[s]; /* L = block span */
        const size_t recs_blk = (size_t)(p->R[s] - 1);
        const size_t nstride = N / (size_t)p->R[s];
        size_t bi;
        if (p->msz[s]) {
            /* msz: ONE call, Gs = blocks (in-kernel group loop: bp += 2*R*Ls,
             * twg += (R-1)*8 per block), Ls = count = D, in place on stg. */
            p->fz[s](0, 0, stg, 0, p->tz[s], 0, D, nb, 0, 0, D);
            return;
        }
        if (p->tail[s]) {
            /* t2cs: per group of G = R[s-1] blocks, per inner column c:
             * columns = the G blocks (in stride Gs = L, out stride OGs =
             * L in place, or the natural weight W of q_{s-1} on the last
             * stage), legs at Ls = D, count = G. */
            const size_t G = (size_t)p->R[s - 1], ngrp = nb / G;
            const size_t recs_grp = ((G + 1) / 2) * recs_blk;
            const int gen2 = (p->tail[s] == 2);
            size_t g, c, W = 1;
            int j;
            for (j = 0; j < s - 1; j++) W *= (size_t)p->R[j];
            if (gen2 && s == p->K - 1 && p->gl[s]) {
                /* t2csgn: the whole last stage in ONE call — the group loop
                 * runs in-kernel over natbase (stride G); Gs = the groups. */
                p->fgl[s](stg, (const double *)p->natbase, zout,
                          (double *)(p->gord ? p->gorder : 0), p->tf[s], p->t2g[s],
                          D, ngrp, nstride, W, G);
                return;
            }
            for (g = 0; g < ngrp; g++) {
                const double *tw = gen2 ? p->tf[s] : p->tf[s] + g * recs_grp * 8;
                const double *t2 = gen2 ? p->t2g[s] + g * 8 : 0;
                for (c = 0; c < D; c++) {
                    const double *in = stg + 2 * (g * G * L + c);
                    if (s < p->K - 1)
                        p->fcs[s](in, 0, stg + 2 * (g * G * L + c), 0, tw, t2, D, L, D, L, G);
                    else
                        p->fcs[s](in, 0, zout + 2 * (p->natbase[g * G] + c), 0, tw, t2,
                                  D, L, nstride, W, G);
                }
            }
            return;
        }
        /* t2cp call: Ls = D (legs), Gs unused (one digit), OLs, OGs = 1, count = D */
        for (bi = 0; bi < nb; bi++) {
            const double *blk = stg + 2 * bi * L;
            const double *tw = p->tf[s] + bi * recs_blk * 8;
            if (s < p->K - 1)
                p->f[s](blk, 0, stg + 2 * bi * L, 0, tw, 0, D, 0, D, 1, D);
            else
                p->f[s](blk, 0, zout + 2 * p->natbase[bi], 0, tw, 0, D, 0, nstride, 1, D);
        }
    }
}

static inline void vfft_ilfd_execute_fwd(const vfft_ilfd_plan_t *p,
                                         const double *zin, double *zout)
{
    int s;
    for (s = 0; s < p->K; s++) vfft_ilfd_stage(p, s, zin, zout);
}

/* Per-stage FORM race (2026-09-05): on each msz-eligible stage, time the
 * t2cp/tail form against msz on real data in pipeline order (a stage's
 * input is the previous stages' output) and keep the faster in
 * p->msz[s]. The probes' stand-in for the wisdom axis the front door will
 * bank; never a rule. Leaves the staging plane transformed (call the
 * plain execute afterwards for a spectrum). now_ns = the caller's clock. */
static inline void vfft_ilfd_race_forms(vfft_ilfd_plan_t *p, const double *zin,
                                        double *zout, double (*now_ns)(void))
{
    int s;
    vfft_ilfd_stage(p, 0, zin, zout);
    for (s = 1; s < p->K; s++) {
        if (p->tz[s]) {
            double ta = 1e300, tb = 1e300;
            int r;
            for (r = 0; r < 3; r++) {
                double t0;
                p->msz[s] = 0;
                t0 = now_ns(); vfft_ilfd_stage(p, s, zin, zout); t0 = now_ns() - t0;
                if (t0 < ta) ta = t0;
                p->msz[s] = 1;
                t0 = now_ns(); vfft_ilfd_stage(p, s, zin, zout); t0 = now_ns() - t0;
                if (t0 < tb) tb = t0;
            }
            p->msz[s] = (tb < ta);
        }
        if (p->fgl[s]) {
            /* the last stage's three forms: t2csg per group, t2csgn in block
             * order, t2csgn in natural-base order */
            double tt[3] = { 1e300, 1e300, 1e300 };
            int r, arm, best = 0;
            for (r = 0; r < 7; r++)
                for (arm = 0; arm < 3; arm++) {
                    double t0;
                    p->gl[s] = (arm != 0); p->gord = (arm == 2);
                    t0 = now_ns(); vfft_ilfd_stage(p, s, zin, zout); t0 = now_ns() - t0;
                    if (t0 < tt[arm]) tt[arm] = t0;
                }
            for (arm = 1; arm < 3; arm++) if (tt[arm] < tt[best]) best = arm;
            p->gl[s] = (best != 0); p->gord = (best == 2);
        }
        vfft_ilfd_stage(p, s, zin, zout);
    }
}

#endif /* VFFT_IL_FLATDIT_H */
