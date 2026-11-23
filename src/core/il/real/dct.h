/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Tuğbars Heptaşkın */
/* dct.h -- DCT-II and DCT-III (VFFT_DCT2, VFFT_DCT3): 1D, N reals in and N reals out, any N, one transform,
 * on the interleaved real FFT of the same length.
 *
 * THE DEFINITIONS (include/vfft.h; unnormalized, each the other's inverse up to 2N):
 *   DCT-II   Y[k] = 2 sum_{n=0..N-1} x[n] cos(pi k (2n+1) / 2N)
 *   DCT-III  Y[k] = X[0] + 2 sum_{n=1..N-1} X[n] cos(pi n (2k+1) / 2N)
 *
 * THE METHOD: one real FFT of length N between two passes (J. Makhoul, "A fast cosine transform in one and
 * two dimensions", IEEE Trans. ASSP-28, 1980). It is the machine of the sine pair (il/real/dst.h) without
 * the sign flip and the reversal that turn a cosine transform into a sine one. With c_k = cos(pi k / 2N),
 * s_k = sin(pi k / 2N):
 *   DCT-II   reorder   v[m] = x[2m], v[N-1-m] = x[2m+1]          (odd N: v[(N-1)/2] = x[N-1])
 *            R2C at N  V[0 .. N/2]
 *            rotate    Y[0] = 2 Re V[0]
 *                      Y[k] = 2 (c_k Re V[k] + s_k Im V[k]),  Y[N-k] = 2 (s_k Re V[k] - c_k Im V[k]),
 *                      k = 1 .. (N-1)/2;  even N: Y[N/2] = sqrt 2 Re V[N/2]
 *   DCT-III  the same steps transposed:
 *            rotate    V[0] = X[0];  V[k] = (c_k X[k] + s_k X[N-k]) + i (s_k X[k] - c_k X[N-k]);
 *                      even N: V[N/2] = sqrt 2 X[N/2]
 *            C2R at N  v[0 .. N-1]
 *            reorder   Y[2m] = v[m], Y[2m+1] = v[N-1-m]          (odd N: Y[N-1] = v[(N-1)/2])
 * Every step is a permutation or a rotation by a unit complex number, so the error stays that of the FFT
 * at every length. Each pass reads and writes whole vectors, 4 doubles at AVX2 and 8 at AVX-512, with a
 * scalar tail.
 *
 * THE PLAN is the sine pair's (vfft_dst_t, il/real/dst.h): the two real FFTs at N, out of place (an R2C and
 * a C2R plan of the real door), the rotation tables and two work buffers allocated at create: W, N + 2
 * doubles, the half spectrum; V, N doubles, the reordered reals. A DCT-II whose output is not its input
 * reorders into the output buffer and leaves V alone. Either transform takes out == in. A plan of T threads
 * threads inside its real FFTs; the passes run on the calling thread.
 *
 * A VFFT_DCT2 plan runs DCT-II forward and DCT-III backward; a VFFT_DCT3 plan the other way round.
 *
 * THE WISDOM: a DCT cell has its own rows in the real shard, one per mathematical transform --
 * t=dct2|dct3 n=N q=1 ord=nat place=oop [nthreads=T] | eng=dct and its real FFT under dct_ (vw2_r2r_key,
 * il/wisdom/wisdom2_real_il.h), as the sine pair's rows. The real FFT is a CHILD IN ROLE
 * (il/real/real_child.h): created against a private store seeded from the DCT's row, it replays the recipe
 * there or races, and what it banked goes back onto the DCT's row. A DCT never reads or writes an r2c or a
 * c2r row. Its race runs in the DCT's role (il/real/real_role.h): every candidate FFT is timed as one whole
 * DCT -- the reorder pass, that FFT, the rotation pass -- and the fastest whole DCT wins. The plan is
 * placement-blind, so one row serves both placements; a plan of either kind reads both rows. Under
 * VFFT_FLAG_ESTIMATE the two FFTs are planned by formulas, nothing is measured and no store is read or
 * written.
 *
 * POSITION IN vfft.c: after il/real/dst.h (the plan and its tables) and before the front door's create. */
#ifndef VFFT_IL_REAL_DCT_H
#define VFFT_IL_REAL_DCT_H
#ifndef VFFT_FLOAT   /* FP64 only: the FP32 unit serves 1D C2C */

#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

#include "il/real/dst.h"   /* vfft_dst_t: the two real FFTs at N, the rotation tables, the work buffers */

/* ── the reorder passes ─────────────────────────────────────────────────── */

/* v[m] = x[2m], v[N-1-m] = x[2m+1], m = 0 .. N/2 - 1; odd N: v[(N-1)/2] = x[N-1]. x and v apart. */
static inline void _dct_reorder(const double *x, double *v, int N)
{
    const int np = N / 2;
    int m = 0;
#if defined(__AVX512F__)
    {
        const __m512i EV = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14), OD = _mm512_setr_epi64(15, 13, 11, 9, 7, 5, 3, 1);
        for (; m + 8 <= np; m += 8)
        {
            const __m512d a = _mm512_loadu_pd(x + 2 * m), b = _mm512_loadu_pd(x + 2 * m + 8);
            _mm512_storeu_pd(v + m, _mm512_permutex2var_pd(a, EV, b));
            _mm512_storeu_pd(v + N - 8 - m, _mm512_permutex2var_pd(a, OD, b));
        }
    }
#endif
#if defined(__AVX2__)
    for (; m + 4 <= np; m += 4)
    {
        const __m256d a = _mm256_loadu_pd(x + 2 * m), b = _mm256_loadu_pd(x + 2 * m + 4);
        _mm256_storeu_pd(v + m, _mm256_permute4x64_pd(_mm256_unpacklo_pd(a, b), 0xD8));
        _mm256_storeu_pd(v + N - 4 - m, _mm256_permute4x64_pd(_mm256_unpackhi_pd(a, b), 0x27));
    }
#endif
    for (; m < np; m++)
    {
        v[m] = x[2 * m];
        v[N - 1 - m] = x[2 * m + 1];
    }
    if (N & 1)
        v[np] = x[N - 1];
}

/* y[2m] = v[m], y[2m+1] = v[N-1-m], m = 0 .. N/2 - 1; odd N: y[N-1] = v[(N-1)/2]. v and y apart. */
static inline void _dct_unreorder(const double *v, double *y, int N)
{
    const int np = N / 2;
    int m = 0;
#if defined(__AVX512F__)
    {
        const __m512i LO = _mm512_setr_epi64(0, 15, 1, 14, 2, 13, 3, 12), HI = _mm512_setr_epi64(4, 11, 5, 10, 6, 9, 7, 8);
        for (; m + 8 <= np; m += 8)
        {
            const __m512d a = _mm512_loadu_pd(v + m), b = _mm512_loadu_pd(v + N - 8 - m);
            _mm512_storeu_pd(y + 2 * m, _mm512_permutex2var_pd(a, LO, b));
            _mm512_storeu_pd(y + 2 * m + 8, _mm512_permutex2var_pd(a, HI, b));
        }
    }
#endif
#if defined(__AVX2__)
    for (; m + 4 <= np; m += 4)
    {
        const __m256d a = _mm256_loadu_pd(v + m);
        const __m256d o = _mm256_permute4x64_pd(_mm256_loadu_pd(v + N - 4 - m), 0x1B);
        const __m256d lo = _mm256_unpacklo_pd(a, o), hi = _mm256_unpackhi_pd(a, o);
        _mm256_storeu_pd(y + 2 * m, _mm256_permute2f128_pd(lo, hi, 0x20));
        _mm256_storeu_pd(y + 2 * m + 4, _mm256_permute2f128_pd(lo, hi, 0x31));
    }
#endif
    for (; m < np; m++)
    {
        y[2 * m] = v[m];
        y[2 * m + 1] = v[N - 1 - m];
    }
    if (N & 1)
        y[N - 1] = v[np];
}

/* ── the rotation passes ────────────────────────────────────────────────── */

/* DCT-II: the half spectrum V (interleaved, N/2 + 1 bins) into the N outputs; t2c, t2s the doubled tables.
 * V and y apart. */
static inline void _dct2_rotate(const double *V, double *y, const double *t2c, const double *t2s, int N)
{
    const int km = (N - 1) / 2;
    int k = 1;
    y[0] = 2.0 * V[0];
#if defined(__AVX512F__)
    {
        const __m512i RE = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14), IM = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
        const __m512i REV = _mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
        for (; k + 7 <= km; k += 8)
        {
            const __m512d z0 = _mm512_loadu_pd(V + 2 * k), z1 = _mm512_loadu_pd(V + 2 * k + 8);
            const __m512d re = _mm512_permutex2var_pd(z0, RE, z1), im = _mm512_permutex2var_pd(z0, IM, z1);
            const __m512d c = _mm512_loadu_pd(t2c + k), s = _mm512_loadu_pd(t2s + k);
            _mm512_storeu_pd(y + k, _mm512_fmadd_pd(c, re, _mm512_mul_pd(s, im)));
            _mm512_storeu_pd(y + N - 7 - k, _mm512_permutexvar_pd(REV, _mm512_fmsub_pd(s, re, _mm512_mul_pd(c, im))));
        }
    }
#endif
#if defined(__AVX2__)
    for (; k + 3 <= km; k += 4)
    {
        const __m256d z0 = _mm256_loadu_pd(V + 2 * k), z1 = _mm256_loadu_pd(V + 2 * k + 4);
        const __m256d re = _mm256_permute4x64_pd(_mm256_unpacklo_pd(z0, z1), 0xD8);
        const __m256d im = _mm256_permute4x64_pd(_mm256_unpackhi_pd(z0, z1), 0xD8);
        const __m256d c = _mm256_loadu_pd(t2c + k), s = _mm256_loadu_pd(t2s + k);
        _mm256_storeu_pd(y + k, _mm256_fmadd_pd(c, re, _mm256_mul_pd(s, im)));
        _mm256_storeu_pd(y + N - 3 - k, _mm256_permute4x64_pd(_mm256_fmsub_pd(s, re, _mm256_mul_pd(c, im)), 0x1B));
    }
#endif
    for (; k <= km; k++)
    {
        const double re = V[2 * k], im = V[2 * k + 1];
        y[k] = t2c[k] * re + t2s[k] * im;
        y[N - k] = t2s[k] * re - t2c[k] * im;
    }
    if (!(N & 1))
        y[N / 2] = t2c[N / 2] * V[N];
}

/* DCT-III: the N inputs into the half spectrum V the C2R reads (the imaginary parts of bin 0 and, at an even
 * N, bin N/2 written zero). X and V apart. */
static inline void _dct3_rotate(const double *X, double *V, const double *tc, const double *ts, const double *t2c, int N)
{
    const int km = (N - 1) / 2;
    int k = 1;
    V[0] = X[0];
    V[1] = 0.0;
#if defined(__AVX512F__)
    {
        const __m512i REV = _mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
        const __m512i LO = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11), HI = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
        for (; k + 7 <= km; k += 8)
        {
            const __m512d a = _mm512_loadu_pd(X + k);                                            /* X[k] */
            const __m512d d = _mm512_permutexvar_pd(REV, _mm512_loadu_pd(X + N - 7 - k));         /* X[N-k] */
            const __m512d c = _mm512_loadu_pd(tc + k), s = _mm512_loadu_pd(ts + k);
            const __m512d vr = _mm512_fmadd_pd(c, a, _mm512_mul_pd(s, d)), vi = _mm512_fmsub_pd(s, a, _mm512_mul_pd(c, d));
            _mm512_storeu_pd(V + 2 * k, _mm512_permutex2var_pd(vr, LO, vi));
            _mm512_storeu_pd(V + 2 * k + 8, _mm512_permutex2var_pd(vr, HI, vi));
        }
    }
#endif
#if defined(__AVX2__)
    for (; k + 3 <= km; k += 4)
    {
        const __m256d a = _mm256_loadu_pd(X + k);
        const __m256d d = _mm256_permute4x64_pd(_mm256_loadu_pd(X + N - 3 - k), 0x1B);
        const __m256d c = _mm256_loadu_pd(tc + k), s = _mm256_loadu_pd(ts + k);
        const __m256d vr = _mm256_fmadd_pd(c, a, _mm256_mul_pd(s, d)), vi = _mm256_fmsub_pd(s, a, _mm256_mul_pd(c, d));
        const __m256d lo = _mm256_unpacklo_pd(vr, vi), hi = _mm256_unpackhi_pd(vr, vi);
        _mm256_storeu_pd(V + 2 * k, _mm256_permute2f128_pd(lo, hi, 0x20));
        _mm256_storeu_pd(V + 2 * k + 4, _mm256_permute2f128_pd(lo, hi, 0x31));
    }
#endif
    for (; k <= km; k++)
    {
        const double a = X[k], d = X[N - k];
        V[2 * k] = tc[k] * a + ts[k] * d;
        V[2 * k + 1] = ts[k] * a - tc[k] * d;
    }
    if (!(N & 1))
    {
        V[N] = t2c[N / 2] * X[N / 2];
        V[N + 1] = 0.0;
    }
}

/* ── execute ────────────────────────────────────────────────────────────── */

/* DCT-II of x into y (y may be x) */
static inline void _dct2_run(const vfft_dst_t *d, const double *x, double *y)
{
    if (d->N == 1)
    {
        y[0] = 2.0 * x[0];
        return;
    }
    {
        double *v = (y != x) ? y : d->V;   /* the caller's output holds the reordered reals when it is free */
        _dct_reorder(x, v, d->N);
        _vfft_exec6((vfft_plan)d->r2c, VFFT_FORWARD, v, NULL, d->W, NULL);
        _dct2_rotate(d->W, y, d->t2c, d->t2s, d->N);
    }
}
/* DCT-III of X into y (y may be X) */
static inline void _dct3_run(const vfft_dst_t *d, const double *X, double *y)
{
    if (d->N == 1)
    {
        y[0] = X[0];
        return;
    }
    _dct3_rotate(X, d->W, d->tc, d->ts, d->t2c, d->N);
    _vfft_exec6((vfft_plan)d->c2r, VFFT_BACKWARD, d->W, NULL, d->V, NULL);
    _dct_unreorder(d->V, y, d->N);
}

/* the bound executors (vfft_plan_s.k1_exec, k1_go): never decline, either pointer pattern */
static int _k1x_dct2(struct vfft_plan_s *h, vfft_dir_t dir, const vfft_real *zin, vfft_real *zout)
{
    (void)dir;
    _dct2_run(h->dst, zin, zout);
    return 0;
}
static int _k1x_dct3(struct vfft_plan_s *h, vfft_dir_t dir, const vfft_real *zin, vfft_real *zout)
{
    (void)dir;
    _dct3_run(h->dst, zin, zout);
    return 0;
}
/* by direction: the plan's own transform forward, its inverse backward */
static int _k1x_dct(struct vfft_plan_s *h, vfft_dir_t dir, const vfft_real *zin, vfft_real *zout)
{
    if ((h->transform == VFFT_DCT2) == (dir == VFFT_FORWARD))
        _dct2_run(h->dst, zin, zout);
    else
        _dct3_run(h->dst, zin, zout);
    return 0;
}

/* ── the role and the create ────────────────────────────────────────────── */

/* THE ROLE (il/real/real_role.h): one whole DCT with the candidate FFT h inside, on the create's own buffers
 * x (the input) and y (the output), the plan's tables and work buffers. What every candidate of the real
 * door's races is timed as while the DCT's FFT is created. */
static void _dct2_role_run(void *ctx, struct vfft_plan_s *h)
{
    const _dst_role_t *r = (const _dst_role_t *)ctx;
    const vfft_dst_t *d = r->d;
    _dct_reorder(r->x, r->y, d->N);
    _real_il_exec_any(h, r->y, d->W);
    _dct2_rotate(d->W, r->y, d->t2c, d->t2s, d->N);
}
static void _dct3_role_run(void *ctx, struct vfft_plan_s *h)
{
    const _dst_role_t *r = (const _dst_role_t *)ctx;
    const vfft_dst_t *d = r->d;
    _dct3_rotate(r->x, d->W, d->tc, d->ts, d->t2c, d->N);
    _real_il_exec_any(h, d->W, d->V);
    _dct_unreorder(d->V, r->y, d->N);
}

/* One of the DCT's two real FFTs at N, out of place: the R2C inside a DCT-II (is_c2r = 0) or the C2R inside a
 * DCT-III, on its row t=dct2 or t=dct3 under dct_ (il/real/real_child.h); a measured create times every
 * candidate as the whole DCT on x and y. NULL: the real door's reason is on the record. */
static struct vfft_plan_s *_dct_child(const vfft_cfg_t *cfg, struct vfft_wisdom_s *W, vfft_dst_t *d, int is_c2r,
                                      int est, double *x, double *y, int *changed)
{
    _dst_role_t role;
    _real_child_t rc;
    role.d = d;
    role.x = x;
    role.y = y;
    rc.t = is_c2r ? VW2_T_DCT3 : VW2_T_DCT2;
    rc.N = d->N;
    rc.pre = "dct_";
    rc.eng = "dct";
    rc.n = d->N;
    rc.is_c2r = is_c2r;
    rc.run = is_c2r ? _dct3_role_run : _dct2_role_run;
    rc.ctx = &role;
    return _real_child_create(cfg, W, &rc, est, is_c2r ? &d->S3 : &d->S2, changed);
}

/* THE DCT DOOR: a VFFT_DCT2 or VFFT_DCT3 request, 1D, one transform. est = 1 (VFFT_FLAG_ESTIMATE): the two
 * real FFTs planned by formulas, nothing measured, no store (W unused); est = 0: each a child in role of the
 * DCT cell's row in W. NULL with the reason on the error record. */
static vfft_plan _vfft_create_dct_il(const vfft_cfg_t *cfg, struct vfft_wisdom_s *W, int est)
{
    const int N = cfg->n[0];
    const char *tn = _vfft_tname(cfg->transform);
    struct vfft_plan_s *h;
    vfft_dst_t *d;
    if (cfg->dims >= 2)
    {
        _vfft_warn("vfft_create: %dD %s is not served: DCT-II and DCT-III plans are 1D", cfg->dims, tn);
        return NULL;
    }
    if (cfg->howmany > 1)
    {
        _vfft_warn("vfft_create: %s N=%d howmany=%zu is not served: a DCT-II or DCT-III plan runs one transform "
                   "(create it with howmany = 1 and call it once per transform)", tn, N, cfg->howmany);
        return NULL;
    }
    if (N < 1)
    {
        _vfft_warn("vfft_create: %s N=%d invalid (the length must be 1 or more)", tn, N);
        return NULL;
    }
    d = (vfft_dst_t *)calloc(1, sizeof *d);
    h = (struct vfft_plan_s *)calloc(1, sizeof *h);
    if (!d || !h)
    {
        free(d);
        free(h);
        _vfft_warn("vfft_create: %s N=%d: out of memory", tn, N);
        return NULL;
    }
    d->N = N;
    if (N > 1)
    {
        const size_t nt = (size_t)N / 2 + 1;
        double *x = NULL, *y = NULL;   /* the races' input and output (measured plans) */
        int changed = 0;
        d->W = (double *)vfft_aligned_alloc(((size_t)N + 2) * sizeof(double));
        d->V = (double *)vfft_aligned_alloc((size_t)N * sizeof(double));
        d->tab = (double *)vfft_aligned_alloc(4 * nt * sizeof(double));
        if (!est)
        {
            x = (double *)vfft_aligned_alloc((size_t)N * sizeof(double));
            y = (double *)vfft_aligned_alloc((size_t)N * sizeof(double));
        }
        if (!d->W || !d->V || !d->tab || (!est && (!x || !y)))
        {
            vfft_aligned_free(x);
            vfft_aligned_free(y);
            _vfft_dst_destroy(d);
            free(h);
            _vfft_warn("vfft_create: %s N=%d: out of memory (the work buffers)", tn, N);
            return NULL;
        }
        d->tc = d->tab;
        d->ts = d->tab + nt;
        d->t2c = d->tab + 2 * nt;
        d->t2s = d->tab + 3 * nt;
        vfft_cs2pi_table(4LL * N, (long long)nt, d->tc, d->ts);   /* cos, sin(2 pi k / 4N) */
        for (size_t k = 0; k < nt; k++)
        {
            d->t2c[k] = 2.0 * d->tc[k];
            d->t2s[k] = 2.0 * d->ts[k];
        }
        if (!est)
        {
            unsigned sd = 0x9e3779b9u ^ (unsigned)N;
            for (int i = 0; i < N; i++)
            {
                sd = sd * 1664525u + 1013904223u;
                x[i] = (double)(sd >> 8) / (double)(1u << 24) - 0.5;
                y[i] = 0.0;
            }
        }
        d->r2c = _dct_child(cfg, W, d, 0, est, x, y, &changed);
        if (d->r2c)
            d->c2r = _dct_child(cfg, W, d, 1, est, x, y, &changed);
        vfft_aligned_free(x);
        vfft_aligned_free(y);
        if (changed)
            _vw2_persist(W, cfg);   /* the rows banked so far are the cell's, whatever follows */
        if (!d->r2c || !d->c2r)
        {   /* the real door's own reason is on the record: it names the cell that could not be built */
            _vfft_dst_destroy(d);
            free(h);
            return NULL;
        }
    }
    h->transform = cfg->transform;
    h->placement = cfg->placement;
    h->layout = (int)VFI_LAYOUT_IL;
    h->N = N;
    h->K = 1;
    h->nthreads = _vfft_plan_threads(cfg);
    h->dst = d;
    h->k1_exec = _k1x_dct;
    h->k1_go[VFFT_FORWARD] = cfg->transform == VFFT_DCT2 ? _k1x_dct2 : _k1x_dct3;
    h->k1_go[VFFT_BACKWARD] = cfg->transform == VFFT_DCT2 ? _k1x_dct3 : _k1x_dct2;
    return (vfft_plan)h;
}

#endif /* !VFFT_FLOAT */
#endif /* VFFT_IL_REAL_DCT_H */
