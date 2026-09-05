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
 *   Both directions, both order classes (natural / scrambled), in place
 *   legal, single thread; execution = the bound call lists (see below). */
#ifndef VFFT_IL_FLATDIT_H
#define VFFT_IL_FLATDIT_H

#include "il2p.h"

#define VFFT_ILFD_MAX_K 10
/* stages whose run D is at most this many columns run on the column-stride
 * kind: a "column" becomes one block, so the lanes fill from blocks. */
#ifndef VFFT_ILFD_TAIL_D
#define VFFT_ILFD_TAIL_D 4
#endif

/* ONE BOUND CALL of a stage — the executor's whole vocabulary (2026-09-05):
 * the kernel, the buffers, the tables, the strides and the counts, resolved
 * from the plan's form fields at bind time (vfft_ilfd_bind). The served
 * path walks a list of these and calls: no division, no digit-weight loop,
 * no form or direction branch, no resolver, per call. */
enum { _ILFD_ONE = 0,   /* one kernel call: the leaf, msz, t2csgn, t2cp (its in-kernel
                         * block loop: OGs = blocks, Gs = the block pitch R*D) */
       _ILFD_COL = 1,   /* the t2csg per-column tail form 't': ngrp x D calls */
       _ILFD_BLK = 2 }; /* t2cp per block, redirected (a natural last stage without a
                         * tail form — no POOL radix lacks one) */
enum { _ILFD_STG = 0, _ILFD_ZIN = 1, _ILFD_ZOUT = 2, _ILFD_NUL = 3 };  /* in_sel / out_sel */
typedef struct {
    vfft_il2p_fn fn;
    int op, in_sel, out_sel;
    const double *a1;             /* zin_unused: the base table (t2csgn) */
    double *a3;                   /* zout_unused: the group order (t2csgn, natural last stage) */
    const double *tw, *t2;
    size_t Ls, Gs, OLs, OGs, count;
    size_t D, L, G, ngrp, nb, tw_step, t2_step;   /* _ILFD_COL / _ILFD_BLK loops */
    const size_t *obase;          /* COL: the group's first block's natural base; BLK: per
                                   * block; NULL = in place (block order) */
    /* THE TILE AXIS (2026-09-05, the cascade's tcut in flat form): a record
     * inside the tiled suffix holds PER-TILE counts above and advances its
     * bases by these per tile t (doubles; a1 is a size_t table cast to
     * double*, both 8 bytes, so its step is in entries). COL: g_tstep = the
     * tile's first group. All zero = the record runs once, whole plane. */
    size_t in_tstep, out_tstep, tw_tstep, t2_tstep, a1_tstep, g_tstep;
} vfft_ilfd_call_t;

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
    size_t *ipb[VFFT_ILFD_MAX_K];     /* in-place tail stages (s < K-1) on t2csgn: the group base
                                       * table (entry g*G = g*G*L, complex units), block order */
    size_t *gorder;                   /* last stage: the groups in ascending natural-base order
                                       * (t2csgn walks them so consecutive groups fill adjacent
                                       * output lines); NULL = block order */
    int gord;                         /* 1 = pass gorder to t2csgn (default; VFFT_ILFD_NO_GORD=1
                                       * turns it off; A/B flips it) */
    int scr;                          /* 1 = the SCRAMBLED class: the forward's last stage writes
                                       * zout in the plane's own block order (position b*R + l
                                       * holds bin natbase[b] + l*N/R = the mixed-radix digit
                                       * reversal — no scatter, no order table) and the backward
                                       * CONSUMES that comb: the TRANSPOSED pipeline (stages in
                                       * reverse, each IDFT + conj POST-twiddle, the leaf's
                                       * transpose last). Requires scr_ok. */
    /* the transposed kernel set (the scrambled class's backward): t2cp's
     * transpose, msz's transpose, the transposed tails; conj tables shared
     * with the conjugate pipeline (post-multiplying by conj(w) reads the
     * same records). scr_ok = every stage has its transposed twin. */
    vfft_il2p_fn fbt[VFFT_ILFD_MAX_K], fcsbt[VFFT_ILFD_MAX_K], fglbt[VFFT_ILFD_MAX_K], fzbt[VFFT_ILFD_MAX_K];
    int scr_ok;
    vfft_il2p_fn fz[VFFT_ILFD_MAX_K]; /* msz fwd (split body, IL edges, unordered lanes) */
    double *tz[VFFT_ILFD_MAX_K];      /* msz: per block (R-1) [c x4][s x4] records (plain sin);
                                       * non-null = the stage is msz-ELIGIBLE (s < K-1; any run
                                       * since the §3 odd-count arms, 2026-09-05) */
    int msz[VFFT_ILFD_MAX_K];         /* 1 = run the stage on msz (default where eligible;
                                       * VFFT_ILFD_NO_MSZ=1 at create turns it off; A/B flips it) */
    double *tf[VFFT_ILFD_MAX_K];      /* t2cp: per block (R-1) broadcast records;
                                       * t2cs: per GROUP, per block-pair, (R-1) VTW2 records */
    /* BACKWARD (2026-09-05): the CONJUGATE pipeline — same stage order and
     * forms (msz / gl / gord are shared), backward kernels (n1c, t2c, msz,
     * t2csg, t2csgn _bwd: IDFT blocks, PRE-twiddle) and conjugated tables.
     * bwd_ok = 0 when a stage has no backward form (a t2cs tail): such a
     * plan serves the forward direction only — the front door refuses it. */
    vfft_il2p_fn lb;
    vfft_il2p_fn fb[VFFT_ILFD_MAX_K], fcsb[VFFT_ILFD_MAX_K], fglb[VFFT_ILFD_MAX_K], fzb[VFFT_ILFD_MAX_K];
    double *tfb[VFFT_ILFD_MAX_K], *t2gb[VFFT_ILFD_MAX_K], *tzb[VFFT_ILFD_MAX_K];
    int bwd_ok;
    size_t *natbase;                  /* last stage: block -> natural index */
    double *stg;                      /* 2N staging plane */
    /* THE BOUND CALL LISTS (2026-09-05): what execute walks. cf = the
     * forward, cb = the conjugate backward (natural class), ct = the
     * transposed backward (scrambled class), each in execution order, one
     * record per stage. The form fields above (msz / gl / gord / scr / tail)
     * are the SOURCE; vfft_ilfd_bind derives the lists from them, and every
     * writer of those fields rebinds (create_chain, apply_forms, race_forms,
     * create_scr_of, the planner's scr flip, the probes). */
    vfft_ilfd_call_t cf[VFFT_ILFD_MAX_K], cb[VFFT_ILFD_MAX_K], ct[VFFT_ILFD_MAX_K];
    /* THE TILE AXIS (2026-09-05): tw = the tile width in complex = one block
     * of stage tcut (a non-tail stage in [1, K-2]; the width is the INPUT,
     * the cut is DERIVED — the cascade's tcut law), 0 = untiled. The stages
     * tcut.. run depth-first per tile: the natural class to K-2 (its last
     * stage stays global — the scatter's natural-base order fills output
     * lines contiguously only across the whole plane), the scrambled class
     * to K-1 (tile-local); both backward pipelines mirror it. Raced by the
     * planner (vfft_ilfd_race_tw), banked as il_tw= on the kind-3 row,
     * validated by vfft_ilfd_apply_tw. tlo/thi = the tiled range of cf/cb;
     * ct tiles its first K-tcut records; ntile = N / tw. */
    int tw, tcut, ntile, tlo, thi;
} vfft_ilfd_plan_t;

static inline void vfft_ilfd_bind(vfft_ilfd_plan_t *p);

static inline void vfft_ilfd_destroy(vfft_ilfd_plan_t *p)
{
    int s;
    if (!p) return;
    for (s = 0; s < VFFT_ILFD_MAX_K; s++) {
        VFFT_IL2P_FREE(p->tf[s]); VFFT_IL2P_FREE(p->t2g[s]); VFFT_IL2P_FREE(p->tz[s]);
        VFFT_IL2P_FREE(p->tfb[s]); VFFT_IL2P_FREE(p->t2gb[s]); VFFT_IL2P_FREE(p->tzb[s]);
    }
    free(p->gorder);
    for (s = 0; s < VFFT_ILFD_MAX_K; s++) free(p->ipb[s]);
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
    {   /* the plain leaf: n1c (natural in/out, alias-tolerant), both directions */
        p->lf = vfft_il2p_n1c_fn(R[0], 0);
        p->lb = vfft_il2p_n1c_fn(R[0], 1);
        if (!p->lf || !p->lb) { vfft_ilfd_destroy(p); return 0; }
    }
    p->bwd_ok = 1;
    p->scr_ok = 1;
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
            if (s < K - 1 && vfft_il2p_msz_fn(R[s]) && vfft_il2p_msz_bwd_fn(R[s]) &&
                vfft_il2p_mszt_bwd_fn(R[s])) {
                double *tz = (double *)VFFT_IL2P_ALLOC(nb * recs_blk * 8 * sizeof(double));
                double *tzb = (double *)VFFT_IL2P_ALLOC(nb * recs_blk * 8 * sizeof(double));
                if (!tz || !tzb) { VFFT_IL2P_FREE(tz); VFFT_IL2P_FREE(tzb); vfft_ilfd_destroy(p); return 0; }
                for (bi = 0; bi < nb; bi++) {
                    const size_t Q = _ilfd_block_Q(p, s, bi);
                    for (l = 1; l < R[s]; l++) {
                        const double a = -2.0 * VFFT_IL2P_PI * (double)((size_t)l * Q % L) / (double)L;
                        const double c = cos(a), sn = sin(a);
                        double *rf = tz + (bi * recs_blk + (size_t)(l - 1)) * 8;
                        double *rb = tzb + (bi * recs_blk + (size_t)(l - 1)) * 8;
                        for (lane = 0; lane < 4; lane++) {
                            rf[lane] = c; rf[4 + lane] = sn;
                            rb[lane] = c; rb[4 + lane] = -sn;   /* bwd: conj */
                        }
                    }
                }
                p->tz[s] = tz; p->tzb[s] = tzb;
                p->fz[s] = vfft_il2p_msz_fn(R[s]);
                p->fzb[s] = vfft_il2p_msz_bwd_fn(R[s]);
                p->fzbt[s] = vfft_il2p_mszt_bwd_fn(R[s]);
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
                p->fgl[s] = vfft_il2p_t2csgn_fn(R[s]);
                p->gl[s] = (p->fgl[s] != 0) && !getenv("VFFT_ILFD_NO_GL");
                p->fcsb[s] = vfft_il2p_t2csg_bwd_fn(R[s]);
                p->fglb[s] = p->fgl[s] ? vfft_il2p_t2csgn_bwd_fn(R[s]) : 0;
                if (!p->fcsb[s] || (p->fgl[s] && !p->fglb[s])) p->bwd_ok = 0;
                p->fcsbt[s] = vfft_il2p_t2csgt_bwd_fn(R[s]);
                p->fglbt[s] = p->fgl[s] ? vfft_il2p_t2csgnt_bwd_fn(R[s]) : 0;
                if (!p->fcsbt[s] || (p->fgl[s] && !p->fglbt[s])) p->scr_ok = 0;
                if (p->fgl[s]) {
                    /* the identity base table in complex units (group g starts
                     * at block g*G = g*G*L): in place on the tail stages, and
                     * the SCRAMBLED last stage's block-order output */
                    size_t gg;
                    p->ipb[s] = (size_t *)malloc(nb * sizeof(size_t));
                    if (!p->ipb[s]) { vfft_ilfd_destroy(p); return 0; }
                    /* block span = R*D here (create's L is the twiddle modulus N/D) */
                    for (gg = 0; gg < ngrp; gg++) p->ipb[s][gg * G] = gg * G * (size_t)R[s] * D;
                }
                tf = (double *)VFFT_IL2P_ALLOC(npair * 8 * sizeof(double));
                p->tfb[s] = (double *)VFFT_IL2P_ALLOC(npair * 8 * sizeof(double));
                p->t2g[s] = (double *)VFFT_IL2P_ALLOC(ngrp * 8 * sizeof(double));
                p->t2gb[s] = (double *)VFFT_IL2P_ALLOC(ngrp * 8 * sizeof(double));
                if (!tf || !p->tfb[s] || !p->t2g[s] || !p->t2gb[s]) { VFFT_IL2P_FREE(tf); vfft_ilfd_destroy(p); return 0; }
                for (pp = 0; pp < npair; pp++) {
                    double *rf = tf + pp * 8, *rb = p->tfb[s] + pp * 8;
                    for (j = 0; j < 2; j++) {
                        const size_t jj = 2 * pp + (size_t)j;
                        const double a = -2.0 * VFFT_IL2P_PI * (double)((W * jj) % L) / (double)L;
                        const double c = cos(a), sn = sin(a);
                        rf[2 * j] = c; rf[2 * j + 1] = c;
                        rf[4 + 2 * j] = -sn; rf[4 + 2 * j + 1] = sn;
                        rb[2 * j] = c; rb[2 * j + 1] = c;
                        rb[4 + 2 * j] = sn; rb[4 + 2 * j + 1] = -sn;     /* bwd: conj */
                    }
                }
                for (g = 0; g < ngrp; g++) {
                    const size_t Q = _ilfd_block_Q(p, s, g * G);
                    const double a = -2.0 * VFFT_IL2P_PI * (double)(Q % L) / (double)L;
                    const double c = cos(a), sn = sin(a);
                    double *rg = p->t2g[s] + g * 8, *rgb = p->t2gb[s] + g * 8;
                    for (lane = 0; lane < 4; lane++) {
                        rg[lane] = c; rg[4 + lane] = (lane & 1) ? sn : -sn;
                        rgb[lane] = c; rgb[4 + lane] = (lane & 1) ? -sn : sn;   /* bwd: conj */
                    }
                }
                p->tf[s] = tf;
            } else if (!p->tail[s]) {
                /* t2cp: ONE digit per call => (R-1) broadcast records per
                 * block (the 2D per-digit record: [c x4][sign-folded s x4]) */
                p->f[s] = vfft_il2p_t2cp_fn(R[s]);
                p->fb[s] = vfft_il2p_t2c_fn(R[s], 1);   /* t2c bwd = PRE-twiddle conj + IDFT */
                p->fbt[s] = vfft_il2p_t2cp_bwd_fn(R[s]); /* t2cp's transpose: IDFT + POST conj */
                if (!p->f[s] || !p->fb[s]) { vfft_ilfd_destroy(p); return 0; }
                if (!p->fbt[s]) p->scr_ok = 0;
                tf = (double *)VFFT_IL2P_ALLOC(nb * recs_blk * 8 * sizeof(double));
                p->tfb[s] = (double *)VFFT_IL2P_ALLOC(nb * recs_blk * 8 * sizeof(double));
                if (!tf || !p->tfb[s]) { VFFT_IL2P_FREE(tf); vfft_ilfd_destroy(p); return 0; }
                for (bi = 0; bi < nb; bi++) {
                    const size_t Q = _ilfd_block_Q(p, s, bi);
                    for (l = 1; l < R[s]; l++) {
                        const double a = -2.0 * VFFT_IL2P_PI * (double)((size_t)l * Q % L) / (double)L;
                        const double c = cos(a), sn = sin(a);
                        double *rf = tf + (bi * recs_blk + (size_t)(l - 1)) * 8;
                        double *rb = p->tfb[s] + (bi * recs_blk + (size_t)(l - 1)) * 8;
                        for (lane = 0; lane < 4; lane++) {
                            rf[lane] = c;
                            rf[4 + lane] = (lane & 1) ? sn : -sn;
                            rb[lane] = c;
                            rb[4 + lane] = (lane & 1) ? -sn : sn;   /* bwd: conj */
                        }
                    }
                }
                p->tf[s] = tf;
            } else {
                p->bwd_ok = 0;   /* t2cs has no backward twin: forward-only plan */
                p->scr_ok = 0;
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
        if (s == K - 1 && s >= 1 && p->fgl[s]) {
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
    vfft_ilfd_bind(p);
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

/* ═══ BINDING: the form fields -> one call record per stage ══════════════
 * _ilfd_bind_stage_dir mirrors the stage geometry for the forward (bwd = 0)
 * and the conjugate backward (bwd = 1: backward kernels, conjugated tables,
 * same shapes); _ilfd_bind_stage_T for the transposed backward. */
static inline void _ilfd_bind_stage_dir(const vfft_ilfd_plan_t *p, int s, int bwd,
                                        vfft_ilfd_call_t *c)
{
    const size_t N = (size_t)p->N;
    const vfft_il2p_fn lf = bwd ? p->lb : p->lf;
    const vfft_il2p_fn f = bwd ? p->fb[s] : p->f[s];
    const vfft_il2p_fn fcs = bwd ? p->fcsb[s] : p->fcs[s];
    const vfft_il2p_fn fgl = bwd ? p->fglb[s] : p->fgl[s];
    const vfft_il2p_fn fz = bwd ? p->fzb[s] : p->fz[s];
    const double *tf = bwd ? p->tfb[s] : p->tf[s];
    const double *t2g = bwd ? p->t2gb[s] : p->t2g[s];
    const double *tz = bwd ? p->tzb[s] : p->tz[s];
    const int last = (s == p->K - 1);
    memset(c, 0, sizeof *c);
    if (s == 0) {   /* the leaf: zin -> the plane, one call */
        c->op = _ILFD_ONE; c->fn = lf; c->in_sel = _ILFD_ZIN; c->out_sel = _ILFD_STG;
        c->Ls = c->OLs = c->count = p->D[0];
        return;
    }
    {
        const size_t D = p->D[s], L = (size_t)p->R[s] * D, nb = p->nblk[s]; /* L = block span */
        const size_t recs_blk = (size_t)(p->R[s] - 1);
        const size_t nstride = N / (size_t)p->R[s];
        c->in_sel = _ILFD_STG; c->out_sel = last ? _ILFD_ZOUT : _ILFD_STG;
        c->D = D; c->L = L; c->nb = nb;
        if (p->msz[s]) {
            /* msz: ONE call, Gs = blocks (in-kernel group loop: bp += 2*R*Ls,
             * twg += (R-1)*8 per block), Ls = count = D, in place on the plane;
             * never the last stage. zin unused (NULL, as ever). */
            c->op = _ILFD_ONE; c->fn = fz; c->tw = tz; c->in_sel = _ILFD_NUL; c->out_sel = _ILFD_STG;
            c->Ls = D; c->Gs = nb; c->count = D;
            return;
        }
        if (p->tail[s]) {
            /* the tail kinds: per group of G = R[s-1] blocks, columns = the G
             * blocks (in stride Gs = L; out stride OGs = L in place, or the
             * natural weight W of q_{s-1} on the natural last stage), legs at
             * Ls = D, count = G. */
            const size_t G = (size_t)p->R[s - 1], ngrp = nb / G;
            const size_t recs_grp = ((G + 1) / 2) * recs_blk;
            const int gen2 = (p->tail[s] == 2);
            size_t W = 1;
            int j;
            for (j = 0; j < s - 1; j++) W *= (size_t)p->R[j];
            c->Ls = D; c->count = G; c->G = G; c->ngrp = ngrp;
            if (gen2 && p->gl[s]) {
                /* t2csgn: the whole stage in ONE call — the group loop runs
                 * in-kernel over the base table (stride G); Gs = the groups.
                 * Natural last stage: natbase -> zout, optionally in
                 * natural-base order; scrambled last stage: zout in block
                 * order; earlier tail stages: in place, block order. */
                c->op = _ILFD_ONE; c->fn = fgl; c->tw = tf; c->t2 = t2g; c->Gs = ngrp;
                if (last && !p->scr) {
                    c->a1 = (const double *)p->natbase;
                    c->a3 = (double *)(p->gord ? p->gorder : 0);
                    c->OLs = nstride; c->OGs = W;
                } else {
                    c->a1 = (const double *)p->ipb[s];
                    c->OLs = D; c->OGs = L;
                }
                return;
            }
            /* t2cs / t2csg per column (the tail form 't'): ngrp x D calls; the
             * gen1 table advances per group, the gen2 T2 record does */
            c->op = _ILFD_COL; c->fn = fcs; c->Gs = L;
            c->tw = tf; c->tw_step = gen2 ? 0 : recs_grp * 8;
            c->t2 = gen2 ? t2g : 0; c->t2_step = gen2 ? 8 : 0;
            if (last && !p->scr) { c->obase = p->natbase; c->OLs = nstride; c->OGs = W; }
            else { c->OLs = D; c->OGs = L; }      /* in place / the scrambled comb */
            return;
        }
        /* t2cp: Ls = D (legs), count = D. In place, and on the scrambled last
         * stage (block order into zout): ONE call — the kernel's own block loop
         * (OGs = blocks, Gs = the block pitch L, the records advance R-1 per
         * block). The natural last stage: per block, redirected to natbase. */
        c->fn = f; c->tw = tf; c->Ls = D; c->count = D;
        if (!last || p->scr) { c->op = _ILFD_ONE; c->Gs = L; c->OLs = D; c->OGs = nb; return; }
        c->op = _ILFD_BLK; c->tw_step = recs_blk * 8; c->obase = p->natbase;
        c->OLs = nstride; c->OGs = 1;
    }
}

/* the scrambled class's backward, stage s transposed: the IDFT block then
 * conj(w) on the output legs, same geometry; the last stage's transpose
 * reads the comb (zin, block order) into the plane, every other stage runs
 * in place on the plane. */
static inline void _ilfd_bind_stage_T(const vfft_ilfd_plan_t *p, int s, vfft_ilfd_call_t *c)
{
    const size_t D = p->D[s], L = (size_t)p->R[s] * D, nb = p->nblk[s];
    const int last = (s == p->K - 1);
    memset(c, 0, sizeof *c);
    c->in_sel = last ? _ILFD_ZIN : _ILFD_STG; c->out_sel = _ILFD_STG;
    c->D = D; c->L = L; c->nb = nb;
    if (p->msz[s]) {            /* never the last stage; in place */
        c->op = _ILFD_ONE; c->fn = p->fzbt[s]; c->tw = p->tzb[s]; c->in_sel = _ILFD_NUL;
        c->Ls = D; c->Gs = nb; c->count = D;
        return;
    }
    if (p->tail[s]) {
        const size_t G = (size_t)p->R[s - 1], ngrp = nb / G;
        c->Ls = D; c->count = G; c->G = G; c->ngrp = ngrp; c->tw = p->tfb[s]; c->t2 = p->t2gb[s];
        c->OLs = D; c->OGs = L;
        if (p->gl[s]) {
            c->op = _ILFD_ONE; c->fn = p->fglbt[s]; c->a1 = (const double *)p->ipb[s]; c->Gs = ngrp;
            return;
        }
        c->op = _ILFD_COL; c->fn = p->fcsbt[s]; c->Gs = L; c->t2_step = 8;
        return;
    }
    c->op = _ILFD_ONE; c->fn = p->fbt[s]; c->tw = p->tfb[s];
    c->Ls = D; c->Gs = L; c->OLs = D; c->OGs = nb; c->count = D;
}

/* THE EXECUTOR: one record on tile t (t = 0 and zero steps = the whole
 * plane); the buffers are picked by index, the rest is the record. */
static inline void _ilfd_call(const vfft_ilfd_plan_t *p, const vfft_ilfd_call_t *c, size_t t,
                              const double *zin, double *zout)
{
    const double *base[4];
    const double *in, *tw, *t2, *a1;
    double *out;
    base[_ILFD_STG] = p->stg; base[_ILFD_ZIN] = zin; base[_ILFD_ZOUT] = zout; base[_ILFD_NUL] = 0;
    in = (c->in_sel == _ILFD_NUL) ? 0 : base[c->in_sel] + t * c->in_tstep;
    out = (double *)base[c->out_sel] + t * c->out_tstep;
    tw = c->tw ? c->tw + t * c->tw_tstep : 0;
    t2 = c->t2 ? c->t2 + t * c->t2_tstep : 0;
    a1 = c->a1 ? c->a1 + t * c->a1_tstep : 0;
    if (c->op == _ILFD_ONE) {
        c->fn(in, a1, out, c->a3, tw, t2, c->Ls, c->Gs, c->OLs, c->OGs, c->count);
    } else if (c->op == _ILFD_COL) {
        const size_t GL = c->G * c->L, g0 = t * c->g_tstep;
        size_t g, k;
        for (g = g0; g < g0 + c->ngrp; g++) {
            const double *twg = c->tw + g * c->tw_step;
            const double *t2g = c->t2 ? c->t2 + g * c->t2_step : 0;
            const size_t ib = g * GL, ob = c->obase ? c->obase[g * c->G] : ib;
            for (k = 0; k < c->D; k++)
                c->fn(in + 2 * (ib + k), 0, out + 2 * (ob + k), 0, twg, t2g,
                      c->Ls, c->Gs, c->OLs, c->OGs, c->count);
        }
    } else {                                        /* _ILFD_BLK: never tiled */
        size_t b;
        for (b = 0; b < c->nb; b++)
            c->fn(in + 2 * b * c->L, 0, out + 2 * c->obase[b], 0, c->tw + b * c->tw_step, 0,
                  c->Ls, 0, c->OLs, c->OGs, c->count);
    }
}
/* n records, whole plane, in order */
static inline void _ilfd_run(const vfft_ilfd_plan_t *p, const vfft_ilfd_call_t *c, int n,
                             const double *zin, double *zout)
{
    int i;
    for (i = 0; i < n; i++) _ilfd_call(p, c + i, 0, zin, zout);
}
/* n records depth-first per tile: every record on tile 0, then on tile 1, ... */
static inline void _ilfd_run_tiled(const vfft_ilfd_plan_t *p, const vfft_ilfd_call_t *c, int n,
                                   size_t ntile, const double *zin, double *zout)
{
    size_t t;
    int i;
    for (t = 0; t < ntile; t++)
        for (i = 0; i < n; i++) _ilfd_call(p, c + i, t, zin, zout);
}

/* the tile view of one stage record: per-tile counts and steps. bpt =
 * blocks of stage s per tile (tw / L_s, exact by construction). */
static inline void _ilfd_tile_rec(const vfft_ilfd_plan_t *p, int s, vfft_ilfd_call_t *c)
{
    const size_t Ls = (size_t)p->R[s] * p->D[s];
    const size_t bpt = (size_t)p->tw / Ls;
    const size_t recs = (size_t)(p->R[s] - 1) * 8;
    c->in_tstep = c->out_tstep = c->tw_tstep = c->t2_tstep = c->a1_tstep = c->g_tstep = 0;
    if (c->op == _ILFD_ONE) {
        if (c->a1) {                  /* t2csgn: the wrapper reads zin by the RELATIVE group
                                       * index (g*count*R*Ls) and writes by the ABSOLUTE base
                                       * table (obase[g*count]): the input base steps per
                                       * tile, the table pointer and the T2 records shift */
            const size_t gpt = bpt / c->count;          /* count = G blocks per group */
            c->Gs = gpt; c->in_tstep = 2 * (size_t)p->tw;
            c->a1_tstep = gpt * c->count; c->t2_tstep = gpt * 8;
        } else if (c->in_sel == _ILFD_NUL) {            /* msz: Gs = blocks, in place */
            c->Gs = bpt; c->out_tstep = 2 * (size_t)p->tw; c->tw_tstep = bpt * recs;
        } else {                                        /* t2cp: OGs = blocks, Gs = the pitch */
            c->OGs = bpt; c->in_tstep = c->out_tstep = 2 * (size_t)p->tw; c->tw_tstep = bpt * recs;
        }
    } else if (c->op == _ILFD_COL) {
        const size_t gpt = bpt / c->G;
        c->ngrp = gpt; c->g_tstep = gpt;
    }
    /* _ILFD_BLK (a natural last stage without a tail form) is never inside a tile */
}

/* derive the tiled range from tw (the validator: a non-tail stage in
 * [1, K-2] whose block span is tw; anything else = untiled) and stamp the
 * records of the three lists */
static inline void _ilfd_bind_tiles(vfft_ilfd_plan_t *p)
{
    int s, cut = -1;
    p->tcut = 0; p->ntile = 1; p->tlo = 0; p->thi = 0;
    if (p->tw <= 0) { p->tw = 0; return; }
    for (s = 1; s <= p->K - 2; s++)
        if (!p->tail[s] && (long)p->R[s] * (long)p->D[s] == (long)p->tw) { cut = s; break; }
    if (cut < 0) { p->tw = 0; return; }
    p->tcut = cut;
    p->ntile = p->N / p->tw;
    p->tlo = cut;
    p->thi = p->scr ? p->K : p->K - 1;
    for (s = p->tlo; s < p->thi; s++) {
        _ilfd_tile_rec(p, s, &p->cf[s]);
        _ilfd_tile_rec(p, s, &p->cb[s]);
        _ilfd_tile_rec(p, s, &p->ct[p->K - 1 - s]);
    }
}

/* bind all three lists from the form fields (see the plan struct) */
static inline void vfft_ilfd_bind(vfft_ilfd_plan_t *p)
{
    int s;
    for (s = 0; s < p->K; s++) {
        _ilfd_bind_stage_dir(p, s, 0, &p->cf[s]);
        _ilfd_bind_stage_dir(p, s, 1, &p->cb[s]);
    }
    for (s = p->K - 1; s >= 1; s--) _ilfd_bind_stage_T(p, s, &p->ct[p->K - 1 - s]);
    {   /* the leaf's transpose last: the plane -> natural zout */
        vfft_ilfd_call_t *c = &p->ct[p->K - 1];
        memset(c, 0, sizeof *c);
        c->op = _ILFD_ONE; c->fn = p->lb; c->in_sel = _ILFD_STG; c->out_sel = _ILFD_ZOUT;
        c->Ls = c->OLs = c->count = p->D[0];
    }
    _ilfd_bind_tiles(p);
}

/* ONE stage through the exact serving record — bound from the CURRENT form
 * fields on the spot, so a probe or the form race can flip a field and time
 * the stage without rebinding the plan (the served lists stay as bound;
 * flip sites rebind when done). s == 0 is the leaf (zin -> plane), the
 * last stage writes zout. */
static inline void _ilfd_stage_dir(const vfft_ilfd_plan_t *p, int s,
                                   const double *zin, double *zout, int bwd)
{
    vfft_ilfd_call_t c;
    _ilfd_bind_stage_dir(p, s, bwd, &c);
    _ilfd_run(p, &c, 1, zin, zout);
}

static inline void vfft_ilfd_stage(const vfft_ilfd_plan_t *p, int s,
                                   const double *zin, double *zout)
{
    _ilfd_stage_dir(p, s, zin, zout, 0);
}
static inline void vfft_ilfd_stage_bwd(const vfft_ilfd_plan_t *p, int s,
                                       const double *zin, double *zout)
{
    _ilfd_stage_dir(p, s, zin, zout, 1);
}

/* the forward: the bound list — K calls for a plan whose stages are all
 * one-call forms; with a tile width, the wide prefix, the suffix
 * depth-first per tile, then (natural) the global last stage */
static inline void vfft_ilfd_execute_fwd(const vfft_ilfd_plan_t *p,
                                         const double *zin, double *zout)
{
    if (p->tw > 0) {
        _ilfd_run(p, p->cf, p->tlo, zin, zout);
        _ilfd_run_tiled(p, p->cf + p->tlo, p->thi - p->tlo, (size_t)p->ntile, zin, zout);
        _ilfd_run(p, p->cf + p->thi, p->K - p->thi, zin, zout);
        return;
    }
    _ilfd_run(p, p->cf, p->K, zin, zout);
}

/* the inverse (unnormalized, N * x on a roundtrip).
 * NATURAL class: the conjugate pipeline in the SAME stage order (bwd_ok).
 * SCRAMBLED class (p->scr): the TRANSPOSED pipeline — the last stage's
 * transpose consumes the comb first, the mids and tails follow in reverse
 * order in place, the leaf's transpose writes natural zout last (scr_ok).
 * In place is legal in both: zin is fully consumed before zout is written. */
static inline void vfft_ilfd_execute_bwd(const vfft_ilfd_plan_t *p,
                                         const double *zin, double *zout)
{
    if (p->tw > 0) {
        if (p->scr) {   /* transposed: the tiled stages K-1..tcut first, then the wide rest */
            const int nt = p->K - p->tcut;
            _ilfd_run_tiled(p, p->ct, nt, (size_t)p->ntile, zin, zout);
            _ilfd_run(p, p->ct + nt, p->K - nt, zin, zout);
            return;
        }
        _ilfd_run(p, p->cb, p->tlo, zin, zout);
        _ilfd_run_tiled(p, p->cb + p->tlo, p->thi - p->tlo, (size_t)p->ntile, zin, zout);
        _ilfd_run(p, p->cb + p->thi, p->K - p->thi, zin, zout);
        return;
    }
    _ilfd_run(p, p->scr ? p->ct : p->cb, p->K, zin, zout);
}

/* The per-stage FORM verdict as text — the wisdom token il_forms= (one letter
 * per stage s >= 1, '.'-joined): t = the default column form (t2cp, or the
 * t2csg tail), m = msz, n = t2csgn in block order, o = t2csgn in natural-base
 * order (last stage only). */
static inline int vfft_ilfd_forms_str(const vfft_ilfd_plan_t *p, char *buf, size_t n)
{
    size_t off = 0;
    int s;
    if (n == 0) return 0;
    buf[0] = 0;
    for (s = 1; s < p->K; s++) {
        const char c = p->msz[s] ? 'm' : p->gl[s] ? ((s == p->K - 1 && p->gord && !p->scr) ? 'o' : 'n') : 't';
        const int r = snprintf(buf + off, n - off, "%s%c", s > 1 ? "." : "", c);
        if (r < 0 || (size_t)r >= n - off) return 0;
        off += (size_t)r;
    }
    return 1;
}

static inline int vfft_ilfd_apply_forms(vfft_ilfd_plan_t *p, const char *forms);
/* Build the SCRAMBLED class from a banked verdict (chain + forms token):
 * scr = 1 before the forms, a natural-base-order letter ('o') on the last
 * stage becoming the plain group loop ('n'). NULL when the chain has no
 * transposed backward or the token refuses. */
static inline int vfft_ilfd_apply_tw(vfft_ilfd_plan_t *p, int w);
static inline vfft_ilfd_plan_t *vfft_ilfd_create_scr_of(int N, const int *R, int K,
                                                        const char *forms, int tw)
{
    vfft_ilfd_plan_t *p = vfft_ilfd_create_chain(N, R, K);
    char flf[32];
    size_t i;
    if (!p) return 0;
    p->scr = 1;
    if (!p->bwd_ok || !p->scr_ok) { vfft_ilfd_destroy(p); return 0; }
    if (forms && *forms) {
        strncpy(flf, forms, sizeof flf - 1);
        flf[sizeof flf - 1] = 0;
        for (i = 0; flf[i]; i++) if (flf[i] == 'o') flf[i] = 'n';
        if (!vfft_ilfd_apply_forms(p, flf)) { vfft_ilfd_destroy(p); return 0; }
    }
    if (tw > 0 && !vfft_ilfd_apply_tw(p, tw)) { vfft_ilfd_destroy(p); return 0; }
    vfft_ilfd_bind(p);
    return p;
}

/* Apply a forms token — the validator is the law: a letter the stage cannot
 * serve, a wrong length or an unknown letter refuses the WHOLE token (0) and
 * leaves the plan's defaults untouched. */
static inline int vfft_ilfd_apply_forms(vfft_ilfd_plan_t *p, const char *forms)
{
    int msz[VFFT_ILFD_MAX_K], gl[VFFT_ILFD_MAX_K], gord = 0, s;
    const char *q = forms;
    if (!forms || !*forms) return 1;
    for (s = 1; s < p->K; s++) {
        const char c = *q++;
        msz[s] = 0; gl[s] = 0;
        switch (c) {
        case 't': break;
        case 'm': if (!p->tz[s]) return 0; msz[s] = 1; break;
        case 'n': if (!p->fgl[s]) return 0; gl[s] = 1; break;
        case 'o': if (!p->fgl[s] || s != p->K - 1 || !p->gorder || p->scr) return 0; gl[s] = 1; gord = 1; break;
        default: return 0;
        }
        if (s < p->K - 1) { if (*q != '.') return 0; q++; }
    }
    if (*q) return 0;
    for (s = 1; s < p->K; s++) { p->msz[s] = msz[s]; p->gl[s] = gl[s]; }
    p->gord = gord;
    vfft_ilfd_bind(p);
    return 1;
}

/* THE TILE AXIS. Apply a width — the validator is the law: 0 = untiled;
 * otherwise the block span of a non-tail stage in [1, K-2], anything else
 * refuses (0) and leaves the plan untouched. */
static inline int vfft_ilfd_apply_tw(vfft_ilfd_plan_t *p, int w)
{
    int s;
    if (w <= 0) { p->tw = 0; vfft_ilfd_bind(p); return 1; }
    for (s = 1; s <= p->K - 2; s++)
        if (!p->tail[s] && (long)p->R[s] * (long)p->D[s] == (long)w) {
            p->tw = w;
            vfft_ilfd_bind(p);
            return 1;
        }
    return 0;
}

/* the width candidates: 0 (untiled) and every legal stage span whose tile
 * (16 bytes per complex) fits the given cache budget (<= 0 = no gate).
 * Returns the count. The race decides; these are candidates, never a rule. */
static inline int vfft_ilfd_tw_candidates(const vfft_ilfd_plan_t *p, long cache_bytes,
                                          int *out, int max)
{
    int n = 0, s;
    if (max < 1) return 0;
    out[n++] = 0;
    for (s = 1; s <= p->K - 2 && n < max; s++) {
        const long w = (long)p->R[s] * (long)p->D[s];
        if (p->tail[s]) continue;
        if (cache_bytes > 0 && 16L * w > cache_bytes) continue;
        out[n++] = (int)w;
    }
    return n;
}

/* the TILE race (2026-09-05): every candidate width timed on the whole
 * forward (tiling is a cross-stage locality property, so a per-stage clock
 * cannot see it), rounds alternating direction, min; the winner applied
 * and returned (0 = untiled). Runs AFTER the form race: the forms are the
 * kernels, the width is the walk. Leaves zout transformed. */
static inline int vfft_ilfd_race_tw(vfft_ilfd_plan_t *p, const double *zin, double *zout,
                                    long cache_bytes, double (*now_ns)(void))
{
    int cand[VFFT_ILFD_MAX_K + 1], n, i, r, best = 0;
    double tt[VFFT_ILFD_MAX_K + 1];
    n = vfft_ilfd_tw_candidates(p, cache_bytes, cand, VFFT_ILFD_MAX_K + 1);
    if (n <= 1) { vfft_ilfd_apply_tw(p, 0); return 0; }
    for (i = 0; i < n; i++) tt[i] = 1e300;
    for (r = 0; r < 4; r++)
        for (i = 0; i < n; i++) {
            const int a = (r & 1) ? n - 1 - i : i;
            double t0;
            if (!vfft_ilfd_apply_tw(p, cand[a])) continue;
            t0 = now_ns(); vfft_ilfd_execute_fwd(p, zin, zout); t0 = now_ns() - t0;
            if (t0 < tt[a]) tt[a] = t0;
        }
    for (i = 1; i < n; i++) if (tt[i] < tt[best]) best = i;
    vfft_ilfd_apply_tw(p, cand[best]);
    return cand[best];
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
        if (p->fgl[s]) {
            /* the tail forms: t2csg per group, t2csgn in block order, and on
             * the last stage t2csgn in natural-base order (msz off meanwhile) */
            const int narm = (s == p->K - 1 && !p->scr) ? 3 : 2;
            double tt[3] = { 1e300, 1e300, 1e300 };
            int r, arm, best = 0;
            p->msz[s] = 0;
            for (r = 0; r < 7; r++)
                for (arm = 0; arm < narm; arm++) {
                    double t0;
                    p->gl[s] = (arm != 0);
                    if (s == p->K - 1) p->gord = (arm == 2);
                    t0 = now_ns(); vfft_ilfd_stage(p, s, zin, zout); t0 = now_ns() - t0;
                    if (t0 < tt[arm]) tt[arm] = t0;
                }
            for (arm = 1; arm < narm; arm++) if (tt[arm] < tt[best]) best = arm;
            p->gl[s] = (best != 0);
            if (s == p->K - 1) p->gord = (best == 2);
        }
        if (p->tz[s]) {
            /* msz against the best non-msz form of this stage */
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
        vfft_ilfd_stage(p, s, zin, zout);
    }
    vfft_ilfd_bind(p);
}

#endif /* VFFT_IL_FLATDIT_H */
