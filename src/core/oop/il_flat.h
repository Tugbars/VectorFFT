/* il_flat.h — the odd/any-N K=1 IL engine to arbitrary depth (2026-09-04,
 * docs/design/odd_n_cascade_geometry.md, docs/roadmap/odd_large_n_engine.md).
 *
 *   N = R0 * M.   fwd:  LEAF (n1t, R0 legs at stride M, count M — full
 *   lanes, the corner-turned store puts (k, p) at k*R0 + p)
 *              -> ROTOR (one streaming sweep: (k, p) *= w_N^(k*p), the
 *                 four-step's outer twiddle, generated per row from a
 *                 table of M entries — no N-sized table)
 *              -> the 2D COLUMN CHAIN over M rows x R0 lanes (t2c/n1c,
 *                 per-DIGIT compact tables, any depth; the M4-lite natural
 *                 leaf scatter lands the rows in natural order, so memory
 *                 index q*R0 + p == X[p + R0*q] == natural output).
 *   bwd = the Hermitian transpose run backwards: the column chain's
 *   reverse pass (consumes natural X, produces the (k, p) comb layout),
 *   the conjugate rotor, an un-turn sweep (k, p) -> p*M + k, then the
 *   plain n1 bwd leaf (legs at stride M, natural y). v1 pays that un-turn
 *   sweep on bwd only; the un-turn leaf kind is an emitter item.
 *
 *   Why this and not a flat chain of t2 stages: t2's twiddle is per
 *   column PAIR, and in a flat chain the deep stages' twiddles depend on
 *   the block AND the fast index — ~8N doubles of table per stage. The
 *   column chain's t2c twiddle is per DIGIT (rows share it across the R0
 *   lanes): D*(R-1) records per stage. The truly flat interior needs the
 *   two-group "arrange halves" kinds (MKL's Fact kernels) — the emitter
 *   item that retires the leaf turn and the rotor; this engine is the
 *   bridge to it and already covers the size range.
 *
 *   Every piece is shipped and gated: n1t/n1 leaves (il2p), t2c/n1c
 *   stages + tables + the natural pass (il2d_cols.h). Interior count = R0
 *   (odd-count tail: 3.6% at 27), so the LEAF radix is the lever.
 *   v1 = out of place, single thread, seed chain; the plan race, MT and
 *   banding ride the driver next. */
#ifndef VFFT_IL_FLAT_H
#define VFFT_IL_FLAT_H

#include "il2p.h"
#include "il2d_cols.h"

typedef struct {
    int N, R0, M, nst;
    int R[8], L[8];
    vfft_il2p_fn lf;               /* n1t fwd: the corner-turned leaf */
    vfft_il2p_fn lb;               /* n1 bwd: plain leaf, natural out */
    vfft_il2p_fn f[8], b[8];       /* t2c/n1c chain, both directions */
    double *tf[8], *tb[8];         /* per-digit stage tables */
    int *perm;                     /* the natural leaf permutation (nst >= 2) */
    double *scr;                   /* 2N: the natural pass's pre-leaf plane */
    double *rot;                   /* 2M: w_N^k, k in [0, M) */
} vfft_ilflat_plan_t;

static inline void vfft_ilflat_destroy(vfft_ilflat_plan_t *p)
{
    int s;
    if (!p) return;
    for (s = 0; s < p->nst; s++) { free(p->tf[s]); free(p->tb[s]); }
    free(p->perm);
    VFFT_IL2P_FREE(p->scr);
    VFFT_IL2P_FREE(p->rot);
    free(p);
}

/* chain = { R0, interior radices over M = N / R0 ... }, K entries.
 * NULL = outside the corpus / does not multiply to N (the validator). */
static inline vfft_ilflat_plan_t *vfft_ilflat_create_chain(int N, const int *chain, int K)
{
    vfft_ilflat_plan_t *p;
    long prod = 1;
    int s;
    if (N < 4 || K < 2 || K > 9) return 0;
    for (s = 0; s < K; s++) { if (chain[s] < 2) return 0; prod *= chain[s]; }
    if (prod != (long)N) return 0;
    p = (vfft_ilflat_plan_t *)calloc(1, sizeof(*p));
    if (!p) return 0;
    p->N = N; p->R0 = chain[0]; p->M = N / chain[0]; p->nst = K - 1;
    for (s = 1; s < K; s++) p->R[s - 1] = chain[s];
    p->lf = vfft_il2p_leaf_fn(p->R0, 0);
    p->lb = vfft_il2p_n1_bwd_fn(p->R0);
    if (!p->lf || !p->lb || !_il2d_resolve(p->R, p->nst, p->f, p->b) ||
        _il2d_build_tables(p->M, p->nst, p->R, p->L, p->tf, p->tb)) {
        vfft_ilflat_destroy(p);
        return 0;
    }
    if (p->nst >= 2) {
        p->perm = _il2d_nat_perm(p->R, p->nst, p->M);
        if (!p->perm) { vfft_ilflat_destroy(p); return 0; }
    }
    p->scr = (double *)VFFT_IL2P_ALLOC(2u * (size_t)N * sizeof(double));
    p->rot = (double *)VFFT_IL2P_ALLOC(2u * (size_t)p->M * sizeof(double));
    if (!p->scr || !p->rot) { vfft_ilflat_destroy(p); return 0; }
    for (s = 0; s < p->M; s++) {
        const double a = -2.0 * VFFT_IL2P_PI * (double)s / (double)N;
        p->rot[2 * s] = cos(a);
        p->rot[2 * s + 1] = sin(a);
    }
    return p;
}

/* SEED: the largest corpus leaf radix dividing N whose cofactor has a
 * column chain (the 2D builder's greedy). The race decides later. */
static inline vfft_ilflat_plan_t *vfft_ilflat_create(int N)
{
    static const int POOL[] = { 27, 25, 21, 19, 17, 16, 15, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3 };
    int i;
    for (i = 0; i < (int)(sizeof POOL / sizeof POOL[0]); i++) {
        const int r0 = POOL[i];
        int Rs[8], nst = 0, chain[9], s;
        vfft_il2p_fn ff[8], fb[8];
        if (N % r0 || N / r0 < 3) continue;
        if (!vfft_il2p_leaf_fn(r0, 0) || !vfft_il2p_n1_bwd_fn(r0)) continue;
        if (!_il2d_build_chain(N / r0, Rs, ff, fb, &nst) || nst < 1) continue;
        chain[0] = r0;
        for (s = 0; s < nst; s++) chain[s + 1] = Rs[s];
        {
            vfft_ilflat_plan_t *p = vfft_ilflat_create_chain(N, chain, nst + 1);
            if (p) return p;
        }
    }
    return 0;
}

/* the outer twiddle: row k, lane p: z *= w_N^(k p) (conj for bwd). Two
 * lanes per vector; the rotor pair [w^p, w^(p+1)] steps by w^2. */
static inline void _ilflat_rotor(double *z, int M, int R0, const double *rot, int conj)
{
    /* i*w = (-wi, wr): from the swapped pair (wi, wr) negate the EVEN
     * lanes (lane order lo->hi: set_pd lists hi->lo). The rotor pair
     * [w^p, w^(p+1)] and its step [w^2, w^2] live in ymm; advancing is
     * one vector complex multiply (mul + fma + 2 permutes). */
    const __m256d mim = _mm256_set_pd(0.0, -0.0, 0.0, -0.0);
    int k;
    for (k = 1; k < M; k++) {
        double *row = z + 2 * (size_t)k * (size_t)R0;
        const double wr = rot[2 * k], wi = conj ? -rot[2 * k + 1] : rot[2 * k + 1];
        const double s2r = wr * wr - wi * wi, s2i = 2.0 * wr * wi;
        const __m256d s2 = _mm256_set_pd(s2i, s2r, s2i, s2r);          /* [w^2, w^2] */
        const __m256d s2i_ = _mm256_xor_pd(_mm256_permute_pd(s2, 0x5), mim); /* i*w^2 */
        __m256d rv = _mm256_set_pd(wi, wr, 0.0, 1.0);                  /* [1, w] */
        int p = 0;
        for (; p + 2 <= R0; p += 2) {
            const __m256d x = _mm256_loadu_pd(row + 2 * p);
            const __m256d xr = _mm256_movedup_pd(x);
            const __m256d xi = _mm256_permute_pd(x, 0xF);
            const __m256d iw = _mm256_xor_pd(_mm256_permute_pd(rv, 0x5), mim); /* i*rv */
            _mm256_storeu_pd(row + 2 * p, _mm256_fmadd_pd(xi, iw, _mm256_mul_pd(xr, rv)));
            {   /* rv *= [w^2, w^2] */
                const __m256d rr = _mm256_movedup_pd(rv);
                const __m256d ri = _mm256_permute_pd(rv, 0xF);
                rv = _mm256_fmadd_pd(ri, s2i_, _mm256_mul_pd(rr, s2));
            }
        }
        if (p < R0) { /* odd R0: the last lane uses rv's lane 0 = w^p */
            double rb[4];
            _mm256_storeu_pd(rb, rv);
            {
                const double xr = row[2 * p], xi = row[2 * p + 1];
                row[2 * p] = xr * rb[0] - xi * rb[1];
                row[2 * p + 1] = xr * rb[1] + xi * rb[0];
            }
        }
    }
}

static inline void vfft_ilflat_execute_fwd(const vfft_ilflat_plan_t *p,
                                           const double *zin, double *zout)
{
    const size_t M = (size_t)p->M, R0 = (size_t)p->R0;
    p->lf(zin, 0, zout, 0, 0, 0, M, 0, R0, 0, M);          /* turn: (k,p) at k*R0+p */
    _ilflat_rotor(zout, p->M, p->R0, p->rot, 0);
    if (p->nst >= 2)
        _il2d_col_pass_nat(zout, zout, p->M, R0, p->nst, p->R, p->L, p->f,
                           p->tf, 0, p->perm, p->scr);
    else
        _il2d_col_pass(zout, zout, p->M, R0, R0, p->nst, p->R, p->L, p->f,
                       p->tf, 0);
}

static inline void vfft_ilflat_execute_bwd(const vfft_ilflat_plan_t *p,
                                           const double *zin, double *zout)
{
    const size_t M = (size_t)p->M, R0 = (size_t)p->R0;
    size_t k, q;
    /* the column chain's transpose: natural X -> the (k,p) comb layout */
    if (p->nst >= 2)
        _il2d_col_pass_nat(zin, zout, p->M, R0, p->nst, p->R, p->L, p->b,
                           p->tb, 1, p->perm, p->scr);
    else
        _il2d_col_pass(zin, zout, p->M, R0, R0, p->nst, p->R, p->L, p->b,
                       p->tb, 1);
    _ilflat_rotor(zout, p->M, p->R0, p->rot, 1);
    /* un-turn into scr: (k,p) -> p*M + k, as 2x2 complex blocks in
     * registers (rows k,k+1 x lanes q,q+1: one unpack pair per block);
     * the odd row / odd lane edges scalar. Then the plain bwd leaf. */
    for (k = 0; k + 2 <= M; k += 2) {
        const double *r0 = zout + 2 * k * R0, *r1 = r0 + 2 * R0;
        for (q = 0; q + 2 <= R0; q += 2) {
            const __m256d a = _mm256_loadu_pd(r0 + 2 * q);   /* (k,q) (k,q+1) */
            const __m256d b = _mm256_loadu_pd(r1 + 2 * q);   /* (k+1,q) (k+1,q+1) */
            _mm256_storeu_pd(p->scr + 2 * (q * M + k), _mm256_permute2f128_pd(a, b, 0x20));
            _mm256_storeu_pd(p->scr + 2 * ((q + 1) * M + k), _mm256_permute2f128_pd(a, b, 0x31));
        }
        for (; q < R0; q++) {
            p->scr[2 * (q * M + k)]         = r0[2 * q];
            p->scr[2 * (q * M + k) + 1]     = r0[2 * q + 1];
            p->scr[2 * (q * M + k + 1)]     = r1[2 * q];
            p->scr[2 * (q * M + k + 1) + 1] = r1[2 * q + 1];
        }
    }
    for (; k < M; k++)
        for (q = 0; q < R0; q++) {
            p->scr[2 * (q * M + k)]     = zout[2 * (k * R0 + q)];
            p->scr[2 * (q * M + k) + 1] = zout[2 * (k * R0 + q) + 1];
        }
    p->lb(p->scr, 0, zout, 0, 0, 0, M, 0, M, 0, M);
}

#endif /* VFFT_IL_FLAT_H */
