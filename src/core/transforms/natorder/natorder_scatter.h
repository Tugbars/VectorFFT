/* natorder_scatter.h — SCR (scatter terminator) mode for VFFT_ORDER_NATURAL (in-place 1D c2c).
 *
 * Fuses the digit-reversal into the FORWARD's last butterfly stage, so the permutation costs no
 * extra pass (nf passes total, vs PURE's nf+1). Design §2b/§4 + T6-T11 (wins the K=4 band, ~+16-27%
 * vs PURE's +45%). Footprint = OOP (a plan-owned N*K scratch plane pair) — pointer-in-place, not
 * memory-in-place (§1 ambiguity).
 *
 * DATAFLOW (forward): MODEB stage-0 redirect user->scratch + stages 1..nf-2 in-place on scratch
 * (execute_fwd_oop), then a TERMINATOR: for each group, pre-twiddle its R scratch legs by the last
 * stage's combined twiddle (leg0*=cf0, leg j*=tw_scalar[j-1] — twiddle.h:114-118) and run the
 * OOP-capable n1_fwd (plain radix-R DFT: n1(pretwiddled)==t1(raw)) with in_stride=last->stride,
 * OUT_STRIDE=P*K so leg j lands at natural row q+j*P. Groups iterated in q-ascending order (via the
 * reverse row-base map) => writes are R sequential streams (the j-outer pattern, 0.40x vs 0.10x
 * q-outer — natorder §2b). BACKWARD needs nothing here: a natural spectrum inverts identically for
 * every mode, so vfft.c uses the PURE cycle-inverse + DIF backward.
 *
 * Applicability (else build returns 0 -> caller keeps PURE, honorable): DIT only (MODEB needs an
 * untwiddled OOP stage 0; DIF rejected), nf>=2, last stage not LOG3 (its per-element grp_tw isn't a
 * scalar pre-twiddle), and every group's natural homes must form the stride-P comb (verified). */
#ifndef VFFT_NATORDER_SCATTER_H
#define VFFT_NATORDER_SCATTER_H

#include "executor.h"
#include "oop_execute.h"     /* vfft_proto_execute_fwd_oop (MODEB redirect) */
#include "planner.h"
#include <stdlib.h>

typedef struct {
    stride_plan_t sub;          /* shallow copy of the full plan, num_stages = nf-1 */
    const stride_stage_t *last; /* &fullplan->stages[nf-1] (twiddles + n1_fwd codelet) */
    double *scr_re, *scr_im;    /* plan-owned scratch, N*K each */
    int R, P, N;
    size_t K;
    size_t *src;                /* [P] scratch base (doubles) for the group at natural q */
    int *twg;                   /* [P] full-plan stage-group index for that group          */
    int ok;
} natorder_scr_t;

/* Build. `full` = the calibrated in-place plan (owned by the caller — sub aliases its stage table).
 * M/IM = the orientation-detected perm and its inverse (natural[k]=scrambled[M[k]]). */
static inline int natorder_scr_build(natorder_scr_t *s, const stride_plan_t *full,
                                     int N, size_t K, const int *M, const int *IM)
{
    memset(s, 0, sizeof *s);
    int nf = full->num_stages;
    if (full->use_dif_forward || nf < 2) return 0;
    const stride_stage_t *last = &full->stages[nf - 1];
    if (last->use_log3) return 0;                    /* per-element twiddle, not a scalar pre-twiddle */
    int R = last->radix, P = N / R;
    if ((size_t)last->stride != K) return 0;         /* last stage must be adjacent-row (stride==K) */
    if (last->num_groups != P) return 0;

    int *rb2g = (int *)malloc((size_t)N * 4);
    for (int r = 0; r < N; r++) rb2g[r] = -1;
    for (int g = 0; g < P; g++) rb2g[last->group_base[g] / K] = g;

    s->src = (size_t *)malloc((size_t)P * sizeof(size_t));
    s->twg = (int *)malloc((size_t)P * sizeof(int));
    int good = 1;
    for (int q = 0; q < P && good; q++) {
        int gbr = M[q];                              /* bin q sits at scrambled row M[q] = a group base */
        int g = (gbr >= 0 && gbr < N) ? rb2g[gbr] : -1;
        if (g < 0) { good = 0; break; }
        for (int j = 1; j < R; j++)                  /* comb check: natural homes = {q + j*P} */
            if (gbr + j >= N || IM[gbr + j] != q + j * P) { good = 0; break; }
        s->src[q] = (size_t)gbr * K;
        s->twg[q] = g;
    }
    free(rb2g);
    if (!good) { free(s->src); free(s->twg); s->src = NULL; s->twg = NULL; return 0; }

    s->scr_re = (double *)malloc((size_t)N * K * sizeof(double));
    s->scr_im = (double *)malloc((size_t)N * K * sizeof(double));
    if (!s->scr_re || !s->scr_im) {
        free(s->scr_re); free(s->scr_im); free(s->src); free(s->twg);
        memset(s, 0, sizeof *s); return 0;
    }
    s->sub = *full;                                  /* shallow: shares stage table */
    s->sub.num_stages = nf - 1;
    s->last = last;
    s->R = R; s->P = P; s->N = N; s->K = K;
    s->ok = 1;
    return 1;
}

/* Forward: natural spectrum into (ure,uim). Terminator groups [q0,q1) — the whole range for ST;
 * a worker's slice for MT (disjoint output combs + disjoint scratch groups => race-free). */
static inline void natorder_scr_term_range(natorder_scr_t *s, double *ure, double *uim, int q0, int q1)
{
    const stride_stage_t *L = s->last;
    int R = s->R; size_t K = s->K, stride = (size_t)L->stride, ostride = (size_t)s->P * K;
    for (int q = q0; q < q1; q++) {
        size_t gb = s->src[q]; int g = s->twg[q];
        double *sr = s->scr_re + gb, *si = s->scr_im + gb;
        if (L->needs_tw[g]) {
            if (L->cf0_re[g] != 1.0 || L->cf0_im[g] != 0.0)
                _stride_cmul_scalar_inplace(sr, si, K, L->cf0_re[g], L->cf0_im[g]);
            for (int j = 1; j < R; j++)
                _stride_cmul_scalar_inplace(sr + (size_t)j * stride, si + (size_t)j * stride, K,
                                            L->tw_scalar_re[g][j - 1], L->tw_scalar_im[g][j - 1]);
        }
        L->n1_fwd(sr, si, ure + (size_t)q * K, uim + (size_t)q * K, stride, ostride, K);
    }
}

/* Single-thread forward (MODEB body stays ST here; MT split is done by the vfft.c orchestrator). */
static inline void natorder_scr_fwd(natorder_scr_t *s, double *ure, double *uim, size_t K)
{
    vfft_proto_execute_fwd_oop(&s->sub, ure, uim, s->scr_re, s->scr_im, K); /* stages [0,nf-1) */
    natorder_scr_term_range(s, ure, uim, 0, s->P);
}

static inline void natorder_scr_free(natorder_scr_t *s)
{
    if (!s) return;
    free(s->scr_re); free(s->scr_im); free(s->src); free(s->twg);
    memset(s, 0, sizeof *s);
}

#endif /* VFFT_NATORDER_SCATTER_H */
