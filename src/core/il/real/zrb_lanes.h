/* zrb_lanes.h - the real Bluestein over K LANES: the interleaved real batch
 * in its DEFAULT geometry (lane-major: element e of lane t at x[e K + t], bin
 * f of lane t at z[2 (f K + t)]) at an odd N without a chain, as a native IL
 * engine -- the same chirp-z at M >= (3N-1)/2 as the one-row engine (zrb.h),
 * with every edge a lane-wide ROW operation and the convolution the IL
 * column pass at M over the K lanes (il/rank2/il2d_cols.h: the t2c / n1c
 * column chain, its d-major tables, the banded column walk).
 *
 * r2c: row n of the M x K plane = x row n times c[n] (a real row, two
 * multiplies per lane), rows N..M-1 zero; the column chain forward; row r
 * times the kernel's r-th value (the kernel in the chain's own comb order,
 * FFT'd once at create through the same column pass on one lane); the chain
 * backward (the Hermitian transpose, natural out); rows 0..h times c[r] into
 * the half spectrum. c2r: rows 0..h = (X[0] real, 2 X[k]) times conj(c[k]),
 * the two passes, rows 0..N-1 as Re(conj(c[n]) y[n]) into the K real lanes.
 * Both placements one pipeline (every read of the caller's plane precedes the
 * first write). Unnormalized.
 *
 * The plan inputs are M, the column chain at M (with its stage forms) and the
 * column window wc (lanes per pass through the chain: the column pass's own
 * tiling axis); the lane race sweeps them at create and banks eng=zrbl on the
 * cell's q=K row (bridge/real_bridge.h, wisdom2_real_il.h).
 *
 * Needs il/rank2/il2d_cols.h before it (the column pass and its builders are
 * that header's statics); vfft.c includes it right after.
 */
#ifndef VFFT_ZRB_LANES_H
#define VFFT_ZRB_LANES_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#include "zrb.h"   /* vfft_zrb_min_m, the exact chirp trig (through il_prime.h) */

typedef struct vfft_zrbl_s
{
    int N, h;                   /* h = (N-1)/2 */
    int M, K;                   /* the convolution length, the lanes */
    int nst, Rs[8], Ls[8];      /* the column chain at M */
    vfft_il2p_fn ff[8], fb[8];
    double *tf[8], *tb[8];
    double *c;                  /* the chirp, 2N */
    double *kf, *kb;            /* the kernels in the chain's comb order, 1/M baked, 2M each */
    double *scr;                /* the M x K plane */
    int wc;                     /* lanes per column pass, 0 = all K */
    char forms[64];             /* the stage forms as applied ("" = the defaults) */
} vfft_zrbl_plan_t;

static inline void vfft_zrbl_destroy(vfft_zrbl_plan_t *p)
{
    int s;
    if (!p) return;
    for (s = 0; s < p->nst; s++) { vfft_aligned_free(p->tf[s]); vfft_aligned_free(p->tb[s]); }
    vfft_aligned_free(p->c);
    vfft_aligned_free(p->kf); vfft_aligned_free(p->kb);
    vfft_aligned_free(p->scr);
    free(p);
}

/* "M/R.R.R[/f<forms>]/wW" */
static inline void vfft_zrbl_str(const vfft_zrbl_plan_t *p, char *b, size_t sz)
{
    int s, off = snprintf(b, sz, "%d/", p->M);
    for (s = 0; s < p->nst && off < (int)sz - 4; s++)
        off += snprintf(b + off, sz - (size_t)off, "%s%d", s ? "." : "", p->Rs[s]);
    if (p->forms[0] && off < (int)sz - 4) off += snprintf(b + off, sz - (size_t)off, "/f%s", p->forms);
    if (off < (int)sz - 4) snprintf(b + off, sz - (size_t)off, "/w%d", p->wc);
}

/* The plan at (N, K, M, chain). NULL when M is below the bound, the chain
 * does not multiply to M, or a stage has no kernel pair. */
static inline vfft_zrbl_plan_t *vfft_zrbl_create(int N, int K, int M, const int *Rs, int nst,
                                                 const char *forms, int wc)
{
    vfft_zrbl_plan_t *p;
    double *za, *zb;
    int n, j, s;
    if (N < 3 || !(N & 1) || K < 1 || M < vfft_zrb_min_m(N) || nst < 1 || nst > 8) return NULL;
    if (_il2d_chain_prod(Rs, nst) != (long)M) return NULL;
    p = (vfft_zrbl_plan_t *)calloc(1, sizeof *p);
    if (!p) return NULL;
    p->N = N; p->h = (N - 1) / 2; p->M = M; p->K = K; p->nst = nst; p->wc = wc > 0 && wc < K ? wc : 0;
    memcpy(p->Rs, Rs, sizeof(int) * (size_t)nst);
    if (forms) snprintf(p->forms, sizeof p->forms, "%s", forms);
    if (!_il2d_resolve(p->Rs, nst, p->ff, p->fb) || !_il2d_apply_forms(p->Rs, nst, p->forms, p->ff, p->fb))
    { free(p); return NULL; }
    if (_il2d_build_tables(M, nst, p->Rs, p->Ls, p->tf, p->tb)) { p->nst = 0; vfft_zrbl_destroy(p); return NULL; }
    p->c = _ilprime_alloc((size_t)2 * N);
    p->kf = _ilprime_alloc((size_t)2 * M);
    p->kb = _ilprime_alloc((size_t)2 * M);
    p->scr = _ilprime_alloc((size_t)2 * M * (size_t)K);
    za = (double *)vfft_aligned_calloc((size_t)2 * M, sizeof(double));
    zb = (double *)vfft_aligned_alloc((size_t)2 * M * sizeof(double));
    if (!p->c || !p->kf || !p->kb || !p->scr || !za || !zb) { vfft_aligned_free(za); vfft_aligned_free(zb); vfft_zrbl_destroy(p); return NULL; }
    for (n = 0; n < N; n++)
    {
        const long long m2 = ((long long)n * n) % (2LL * N);
        double co, si;
        vfft_cs2pi_exact(m2, 2LL * N, &co, &si);
        p->c[2 * n] = co; p->c[2 * n + 1] = -si;
    }
    /* the forward kernel: conj(c[j]) on j in [-(N-1), h]; one lane through the chain */
    for (j = 0; j <= p->h; j++) { za[2 * j] = p->c[2 * j]; za[2 * j + 1] = -p->c[2 * j + 1]; }
    for (j = 1; j < N; j++) { za[2 * (M - j)] = p->c[2 * j]; za[2 * (M - j) + 1] = -p->c[2 * j + 1]; }
    _il2d_col_pass(za, zb, M, 1, 1, nst, p->Rs, p->Ls, p->ff, p->tf, 0);
    for (j = 0; j < 2 * M; j++) p->kf[j] = zb[j] / (double)M;
    /* the backward kernel: c[j] on j in [-h, N-1] */
    memset(za, 0, (size_t)2 * M * sizeof(double));
    for (j = 0; j < N; j++) { za[2 * j] = p->c[2 * j]; za[2 * j + 1] = p->c[2 * j + 1]; }
    for (j = 1; j <= p->h; j++) { za[2 * (M - j)] = p->c[2 * j]; za[2 * (M - j) + 1] = p->c[2 * j + 1]; }
    _il2d_col_pass(za, zb, M, 1, 1, nst, p->Rs, p->Ls, p->ff, p->tf, 0);
    for (j = 0; j < 2 * M; j++) p->kb[j] = zb[j] / (double)M;
    vfft_aligned_free(za); vfft_aligned_free(zb);
    (void)s;
    return p;
}

/* ── the lane-wide edges ───────────────────────────────────────────────── */

/* dst[t] = x[t] * (cr + i ci) over K real lanes: a real row times one complex */
static inline void _zrbl_row_rmul(double *dst, const double *x, double cr, double ci, int K)
{
    int t = 0;
#if defined(__AVX2__)
    const __m256d cv = _mm256_setr_pd(cr, ci, cr, ci);
    for (; t + 4 <= K; t += 4)
    {
        const __m256d xv = _mm256_loadu_pd(x + t);
        _mm256_storeu_pd(dst + 2 * t, _mm256_mul_pd(_mm256_permute4x64_pd(xv, 0x50), cv));
        _mm256_storeu_pd(dst + 2 * t + 4, _mm256_mul_pd(_mm256_permute4x64_pd(xv, 0xFA), cv));
    }
#endif
    for (; t < K; t++) { dst[2 * t] = x[t] * cr; dst[2 * t + 1] = x[t] * ci; }
}

/* x[t] = Re(conj(c) * y[t]) = cr*yr + ci*yi over K lanes: a complex row into reals */
static inline void _zrbl_row_reout(double *x, const double *y, double cr, double ci, int K)
{
    int t = 0;
#if defined(__AVX2__)
    const __m256d cv = _mm256_setr_pd(cr, ci, cr, ci);
    for (; t + 4 <= K; t += 4)
    {
        const __m256d p0 = _mm256_mul_pd(_mm256_loadu_pd(y + 2 * t), cv);
        const __m256d p1 = _mm256_mul_pd(_mm256_loadu_pd(y + 2 * t + 4), cv);
        _mm256_storeu_pd(x + t, _mm256_permute4x64_pd(_mm256_hadd_pd(p0, p1), 0xD8));
    }
#endif
    for (; t < K; t++) x[t] = y[2 * t] * cr + y[2 * t + 1] * ci;
}

/* ── the transforms ────────────────────────────────────────────────────── */

/* r2c: N x K reals (lane-major) in, (h+1) x K bins out. x == X is safe. */
static inline void vfft_zrbl_execute_fwd(const vfft_zrbl_plan_t *p, const double *x, double *X)
{
    const int N = p->N, M = p->M, K = p->K;
    const size_t rk = (size_t)K;
    double *scr = p->scr;
    int r;
    for (r = 0; r < N; r++)
        _zrbl_row_rmul(scr + 2 * (size_t)r * rk, x + (size_t)r * rk, p->c[2 * r], p->c[2 * r + 1], K);
    memset(scr + 2 * (size_t)N * rk, 0, 2 * (size_t)(M - N) * rk * sizeof(double));
    _il2d_col_pass(scr, scr, M, rk, (size_t)p->wc, p->nst, p->Rs, p->Ls, p->ff, p->tf, 0);
    for (r = 0; r < M; r++)
        _il2d_row_cmul(scr + 2 * (size_t)r * rk, scr + 2 * (size_t)r * rk, p->kf[2 * r], p->kf[2 * r + 1], rk);
    _il2d_col_pass(scr, scr, M, rk, (size_t)p->wc, p->nst, p->Rs, p->Ls, p->fb, p->tb, 1);
    for (r = 0; r <= p->h; r++)
        _il2d_row_cmul(X + 2 * (size_t)r * rk, scr + 2 * (size_t)r * rk, p->c[2 * r], p->c[2 * r + 1], rk);
}

/* c2r: (h+1) x K bins in (the DC taken real), N x K reals out. X == x is safe. */
static inline void vfft_zrbl_execute_bwd(const vfft_zrbl_plan_t *p, const double *X, double *x)
{
    const int N = p->N, M = p->M, K = p->K, h = p->h;
    const size_t rk = (size_t)K;
    double *scr = p->scr;
    int r, t;
    for (t = 0; t < K; t++) { scr[2 * t] = X[2 * t]; scr[2 * t + 1] = 0.0; }
    for (r = 1; r <= h; r++)   /* 2 X[r] conj(c[r]) */
        _il2d_row_cmul(scr + 2 * (size_t)r * rk, X + 2 * (size_t)r * rk, 2.0 * p->c[2 * r], -2.0 * p->c[2 * r + 1], rk);
    memset(scr + 2 * (size_t)(h + 1) * rk, 0, 2 * (size_t)(M - h - 1) * rk * sizeof(double));
    _il2d_col_pass(scr, scr, M, rk, (size_t)p->wc, p->nst, p->Rs, p->Ls, p->ff, p->tf, 0);
    for (r = 0; r < M; r++)
        _il2d_row_cmul(scr + 2 * (size_t)r * rk, scr + 2 * (size_t)r * rk, p->kb[2 * r], p->kb[2 * r + 1], rk);
    _il2d_col_pass(scr, scr, M, rk, (size_t)p->wc, p->nst, p->Rs, p->Ls, p->fb, p->tb, 1);
    for (r = 0; r < N; r++)
        _zrbl_row_reout(x + (size_t)r * rk, scr + 2 * (size_t)r * rk, p->c[2 * r], p->c[2 * r + 1], K);
}

#endif /* VFFT_ZRB_LANES_H */
