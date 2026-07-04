/* natorder_calibrate.h — the ORDER_NATURAL per-cell verdict race (P1b: PURE vs injected-chain
 * PSWAP; SCR and LEAF-IP join as their executors land).
 *
 * Why injection: the DP/beam planner scores chains under scrambled/T1S economics, so palindromic
 * factorizations (whose digit reversal is an involution => cheapest possible reorder) are pruned
 * before natural-order costs exist. T6-T11 measured an injected 4·8·4 at 128/64 delivering natural
 * order BELOW the scrambled baseline. So the race builds its own candidates.
 *
 * Methodology (T8-derived, create-time budget): warm-up, then ROUNDS interleaved across candidates
 * (A,B,A,B,...) — the order-neutralization that matters most for thermal bias — averaged, 4 executes
 * per timed chunk (values grow ×N^4 max: no rescale needed for N ≤ 2^20). WIN-MARGIN: a challenger
 * must beat PURE by >5% (T8/T11: several cells are noise-tied; verdicts must not flap).
 * The caller stamps the verdict into wisdom v7. natural_order_inplace_design.md §2e. */
#ifndef VFFT_NATORDER_CALIBRATE_H
#define VFFT_NATORDER_CALIBRATE_H

#include "natorder_perm.h"
#include "natorder_exec.h"

#define VFFT_NATORDER_MARGIN 0.95     /* challenger wins at < 95% of PURE's time */
#define VFFT_NATORDER_ROUNDS 3
#define VFFT_NATORDER_CHUNK  4

typedef struct
{
    int mode;               /* VFFT_NAT_PURE_CYCLE or VFFT_NAT_PSWAP            */
    double ns;              /* winner's measured natural-total (wisdom nat_ns)  */
    stride_plan_t *planB;   /* PSWAP: the injected plan (ownership -> caller)   */
    int *pairs;             /* PSWAP: involution pair tape (ownership -> caller)*/
    int nf;                 /* PSWAP: injected chain, for the wisdom stamp      */
    int factors[STRIDE_MAX_STAGES];
    int prof;               /* uniform variant profile used (2 = T1S)           */
} vfft_natorder_verdict_t;

/* Palindromic candidate chains for N: (a,a), (a,b,a), (a,a,a,a), uniform a^m. Radix must
 * have an in-place codelet (reg->n1_fwd). Returns count (<= max). */
static inline int vfft_natorder_palindromes(int N, const vfft_proto_registry_t *reg,
                                            int chains[][STRIDE_MAX_STAGES], int *nfs, int max)
{
    int cnt = 0;
    for (int a = 2; a <= 64 && cnt < max; a++) {
        if (!(a < VFFT_PROTO_REG_MAX_RADIX && reg->n1_fwd[a])) continue;
        long long aa = (long long)a * a;
        if (aa == N) { chains[cnt][0] = a; chains[cnt][1] = a; nfs[cnt++] = 2; continue; }
        if (aa < N && N % aa == 0) {
            int b = (int)(N / aa);
            if (b > 1 && b < VFFT_PROTO_REG_MAX_RADIX && reg->n1_fwd[b] && cnt < max) {
                chains[cnt][0] = a; chains[cnt][1] = b; chains[cnt][2] = a; nfs[cnt++] = 3;
            }
            /* uniform a^m (m>=4): palindromic by construction */
            if (cnt < max) {
                long long p = aa; int m = 2;
                while (p < N && m < STRIDE_MAX_STAGES) { p *= a; m++; }
                if (p == N && m >= 4) {
                    for (int s = 0; s < m; s++) chains[cnt][s] = a;
                    nfs[cnt++] = m;
                }
            }
        }
    }
    return cnt;
}

/* One timed sample: CHUNK x (forward + reorder pass). Buffers are caller scratch (N*K each). */
static inline double _natorder_sample(stride_plan_t *p, double *re, double *im, size_t K,
                                      const int *cycles, const int *pairs, double *tmp)
{
    size_t n = (size_t)p->N * K;
    for (size_t i = 0; i < n; i++) {
        re[i] = (double)((i * 2654435761u) & 1023) / 1024.0 - 0.5;
        im[i] = (double)((i * 40503u) & 1023) / 1024.0 - 0.5;
    }
    double t0 = vfft_proto_now_ns();
    for (int c = 0; c < VFFT_NATORDER_CHUNK; c++) {
        vfft_proto_execute_fwd(p, re, im, K);
        if (cycles) vfft_natorder_cycle_pass(re, im, K, cycles, tmp);
        else if (pairs) vfft_natorder_pair_pass(re, im, K, pairs);
    }
    return (vfft_proto_now_ns() - t0) / VFFT_NATORDER_CHUNK;
}

/* The race. pA/cyclesA = the PURE candidate (calibrated scrambled plan + its cycle tape,
 * already built by the caller). On PSWAP win, v->planB/pairs are live and the caller owns
 * them; on PURE win they are NULL. Never fails: PURE is the floor. */
static inline void vfft_natorder_race(int N, size_t K, const vfft_proto_registry_t *reg,
                                      stride_plan_t *pA, const int *cyclesA, double *tmp2K,
                                      vfft_natorder_verdict_t *v)
{
    memset(v, 0, sizeof *v);
    v->mode = VFFT_NAT_PURE_CYCLE;
    v->prof = 2; /* T1S */

    int chains[3][STRIDE_MAX_STAGES], nfs[3];
    int nc = vfft_natorder_palindromes(N, reg, chains, nfs, 3);

    /* build challenger plans + their pair tapes (drop any that fail the involution check) */
    stride_plan_t *pb[3] = {0};
    int *prs[3] = {0};
    for (int c = 0; c < nc; c++) {
        int vb[STRIDE_MAX_STAGES];
        for (int s = 0; s < nfs[c]; s++) vb[s] = 2; /* uniform T1S; stage 0 ignored */
        vb[0] = 0;
        stride_plan_t *p = vfft_proto_plan_create_ex(N, K, chains[c], vb, nfs[c], 0, reg);
        if (!p) continue;
        /* orientation via impulse probe on THIS plan (fail => drop candidate) */
        size_t tot = (size_t)N * K;
        double *cre = (double *)calloc(tot, sizeof(double));
        double *cim = (double *)calloc(tot, sizeof(double));
        int *M = NULL;
        if (cre && cim) {
            cre[K] = 1.0; /* impulse at n0=1, lane 0 */
            vfft_proto_execute_fwd(p, cre, cim, K);
            M = vfft_natorder_detect(N, chains[c], nfs[c], K, cre, cim, 1);
        }
        free(cre); free(cim);
        int *pl = M ? vfft_natorder_mk_pairs(N, M) : NULL;
        free(M);
        if (!pl) { vfft_proto_plan_destroy(p); continue; }
        pb[c] = p; prs[c] = pl;
    }

    /* interleaved timing rounds (order-neutralized), averaged */
    size_t n = (size_t)N * K;
    double *re = (double *)malloc(n * 8), *im = (double *)malloc(n * 8);
    if (!re || !im) { free(re); free(im); goto cleanup_losers; }
    _natorder_sample(pA, re, im, K, cyclesA, NULL, tmp2K); /* warm-up */
    {
        double sA = 0, sB[3] = {0, 0, 0};
        for (int r = 0; r < VFFT_NATORDER_ROUNDS; r++) {
            sA += _natorder_sample(pA, re, im, K, cyclesA, NULL, tmp2K);
            for (int c = 0; c < nc; c++)
                if (pb[c]) sB[c] += _natorder_sample(pb[c], re, im, K, NULL, prs[c], tmp2K);
        }
        v->ns = sA / VFFT_NATORDER_ROUNDS;
        int best = -1;
        double bns = v->ns * VFFT_NATORDER_MARGIN; /* must beat PURE by the margin */
        for (int c = 0; c < nc; c++) {
            if (!pb[c]) continue;
            double t = sB[c] / VFFT_NATORDER_ROUNDS;
            if (t < bns) { bns = t; best = c; }
        }
        if (best >= 0) {
            v->mode = VFFT_NAT_PSWAP;
            v->ns = bns;
            v->planB = pb[best]; pb[best] = NULL;
            v->pairs = prs[best]; prs[best] = NULL;
            v->nf = nfs[best];
            for (int s = 0; s < nfs[best]; s++) v->factors[s] = chains[best][s];
        }
    }
    free(re); free(im);
cleanup_losers:
    for (int c = 0; c < nc; c++) {
        if (pb[c]) vfft_proto_plan_destroy(pb[c]);
        free(prs[c]);
    }
}

#endif /* VFFT_NATORDER_CALIBRATE_H */
