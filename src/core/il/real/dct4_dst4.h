/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Tuğbars Heptaşkın */
/* dct4_dst4.h -- DCT-IV and DST-IV (VFFT_DCT4, VFFT_DST4): 1D, N reals in and N reals out, any N, one
 * transform, on the interleaved real FFT of the same length.
 *
 * THE DEFINITIONS (include/vfft.h; unnormalized, each its own inverse up to 2N):
 *   DCT-IV   Y[k] = 2 sum_{n=0..N-1} x[n] cos(pi (2k+1)(2n+1) / 4N)
 *   DST-IV   Y[k] = 2 sum_{n=0..N-1} x[n] sin(pi (2k+1)(2n+1) / 4N)
 * The sine transform is the cosine transform of the input read backwards, with every other output negated:
 * DST-IV(x)[k] = (-1)^k DCT-IV(x reversed)[k]. Both of those ride in the passes below.
 *
 * THE METHOD: one real FFT of length N between two passes. Every step is a permutation, a sign, or a rotation
 * by a unit complex number, so the error stays that of the FFT at every length; no step multiplies or divides
 * by a cosine that passes through zero (the route through a same-length DCT-II or DCT-III and a running
 * difference does, and loses digits as N grows).
 *
 * EVEN N, M = N/2. The kernel factors over the pairs z[m] = x[2m] + i x[N-1-2m]:
 *   U[k] = sum_m z[m] e^(-i pi (4m+1)(4k+1) / 4N),   Y[2k] = 2 Re U[k],   Y[N-1-2k] = -2 Im U[k],
 * and (4m+1)(4k+1) = 16mk + 4m + 4k + 1 makes U a length-M complex DFT between two rotations:
 *   U[k] = e^(-i pi (4k+1) / 4N) Z[k],   Z = DFT_M( z[m] e^(-i pi m / N) ).
 * That complex DFT is taken from the real FFT of the same N reals: with r[2m] + i r[2m+1] = z[m] e^(-i pi m/N)
 * and R = R2C_N(r), the two halves of the spectrum give Z[k] = (R[k] + conj R[M-k]) / 2
 * + i e^(2 pi i k / N) (R[k] - conj R[M-k]) / 2. In reals, with P = R[k], Q = R[M-k],
 * alpha_k = pi (4k+1) / 4N and gamma_k = pi (4k-1) / 4N:
 *   pre      r[2m]   = x[2m] cos(pi m/N) + x[N-1-2m] sin(pi m/N)
 *            r[2m+1] = x[N-1-2m] cos(pi m/N) - x[2m] sin(pi m/N)
 *   R2C at N R[0 .. M]
 *   rotate   A1 = Re P + Re Q, A2 = Im P - Im Q, B1 = Re P - Re Q, B2 = Im P + Im Q
 *            v[k]     =  A1 cos alpha + A2 sin alpha - B1 sin gamma - B2 cos gamma       (= Y[2k])
 *            v[M + k] = -A2 cos alpha + A1 sin alpha - B1 cos gamma + B2 sin gamma       (= Y[N-1-2k])
 *   place    Y[2m] = v[m], Y[2m+1] = v[N-1-m]          (the cosine pair's reorder, il/real/dct.h)
 * DST-IV: the pairs are x[N-1-2m] + i x[2m], and v[M + k] changes sign.
 *
 * ODD N. The kernel cos(2 pi a b / 8N), a = 2n+1 and b = 2k+1 both odd, splits by the Chinese remainder
 * theorem, 8 and N being coprime: 1/8N = s/8 + t/N (mod 1) with s = N^-1 mod 8 (= N mod 8) and t = 8^-1 mod N.
 * The mod-8 part is an odd multiple of pi/4, so its cosine and sine are +-1/sqrt 2, and each is a character
 * of the odd residues mod 8 (it factors over a, b and s). What remains is a length-N DFT of the input in
 * the order j = a t mod N, at the frequency l = b mod N:
 *   Y[k] = sqrt 2 ( +-Re W[l] +- Im W[l] ),   W = R2C_N(w),   w[j] = +-x[n] or +-x[N-1-n].
 * The cosine sum needs the even part of the input so ordered with one sign pattern, the sine sum the odd part
 * with another; n and N-1-n land on j and -j, and the two patterns agree on one of the two samples and cancel
 * on the other, so w is the input permuted with signs: one real sequence, one real FFT. In the loops:
 *   pre      j ascending, a = 8j mod N made odd by adding N: the source index (a-1)/2 steps by 4 and the
 *            sign and the choice between x[n] and x[N-1-n] depend on a mod 8 alone, so the pass is a few
 *            strided copies, each with one sign
 *   R2C at N W[0 .. (N-1)/2]
 *   post     k ascending, l = (2k+1) mod N folded to the half spectrum (Im negated past the fold), the two
 *            signs from (2k+1) s mod 8
 * DST-IV: the source index reversed, and the odd outputs negated.
 *
 * THE PLAN holds the R2C at N, out of place (a plan of the real door), the tables of an even N and two work
 * buffers allocated at create: W, N + 2 doubles, the half spectrum; V, N doubles, the FFT's input when the
 * output is the input, and at an even N the outputs before they are placed. Either transform takes
 * out == in. A plan of T threads threads inside its real FFT; the passes run on the calling thread. Forward
 * and backward are the same transform.
 *
 * THE WISDOM: the cell has its own row in the real shard --
 *   t=dct4 n=N q=1 ord=nat place=oop [nthreads=T] | eng=dct and its real FFT under dct_
 *   t=dst4 n=N q=1 ord=nat place=oop [nthreads=T] | eng=dst and its real FFT under dst_
 * (vw2_r2r_key, il/wisdom/wisdom2_real_il.h). The real FFT is a CHILD IN ROLE (il/real/real_child.h): created
 * against a private store seeded from the cell's row, it replays the recipe there or races, and what it banked
 * goes back onto the cell's row. Its race runs in the cell's role (il/real/real_role.h): every candidate FFT
 * is timed as one whole transform, the passes with that FFT between them. The cell never reads or writes an
 * r2c or a c2r row. The plan is placement-blind, so one row serves both placements. Under VFFT_FLAG_ESTIMATE
 * the FFT is planned by formulas, nothing is measured and no store is read or written.
 *
 * POSITION IN vfft.c: after il/real/dct.h (the placing pass) and before the front door's create. */
#ifndef VFFT_IL_REAL_DCT4_DST4_H
#define VFFT_IL_REAL_DCT4_DST4_H
#ifndef VFFT_FLOAT   /* FP64 only: the FP32 unit serves 1D C2C */

#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

#include "common/math/tw_exact.h"   /* the rotation tables, each entry rounded once */
#include "il/real/real_child.h"     /* the real FFT as a child in role of the cell's row */
#include "il/real/dct.h"            /* _dct_unreorder: the placing pass of an even N */

typedef struct vfft_type4_s
{
    int N;
    int sine;                    /* 1: DST-IV; 0: DCT-IV */
    struct vfft_plan_s *r2c;     /* the real FFT at N, out of place (owned); NULL at N = 1 */
    struct vfft_wisdom_s *S;     /* its private store: owned, never saved; NULL under VFFT_FLAG_ESTIMATE and
                                  * at N = 1 */
    double *W;                   /* N + 2 doubles: the half spectrum */
    double *V;                   /* N doubles: the FFT's input; at an even N the outputs before they are placed */
    double *tab;                 /* even N: the six tables below, one allocation, N/2 entries each */
    double *pc, *ps;             /* cos, sin(pi m / N) */
    double *ac, *as;             /* cos, sin(pi (4k+1) / 4N) */
    double *gc, *gs;             /* cos, sin(pi (4k-1) / 4N) */
    int s8;                      /* odd N: N^-1 mod 8 */
    double psign;                /* odd N: the sign of the picks of x[N-1-n] at a = 3 mod 8 negated, at a = 7
                                  * mod 8 as it is: -1 at N = 1 mod 4, +1 at N = 3 mod 4 */
} vfft_type4_t;

/* ── an even N: the passes ──────────────────────────────────────────────── */

/* r[2m] = e c + o s, r[2m+1] = o c - e s, m = 0 .. N/2 - 1, with (e, o) = (x[2m], x[N-1-2m]) for the cosine
 * transform and (x[N-1-2m], x[2m]) for the sine one; c, s = cos, sin(pi m / N). x and r apart. */
static inline void _t4_pre_even(const double *x, double *r, const double *pc, const double *ps, int N, int sine)
{
    const int M = N / 2;
    int m = 0;
#if defined(__AVX512F__)
    {
        const __m512i EV = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14), OD = _mm512_setr_epi64(15, 13, 11, 9, 7, 5, 3, 1);
        const __m512i LO = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11), HI = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
        for (; m + 8 <= M; m += 8)
        {
            const __m512d E = _mm512_permutex2var_pd(_mm512_loadu_pd(x + 2 * m), EV, _mm512_loadu_pd(x + 2 * m + 8));
            const __m512d O = _mm512_permutex2var_pd(_mm512_loadu_pd(x + N - 16 - 2 * m), OD, _mm512_loadu_pd(x + N - 8 - 2 * m));
            const __m512d e = sine ? O : E, o = sine ? E : O;
            const __m512d c = _mm512_loadu_pd(pc + m), s = _mm512_loadu_pd(ps + m);
            const __m512d rr = _mm512_fmadd_pd(e, c, _mm512_mul_pd(o, s)), ri = _mm512_fmsub_pd(o, c, _mm512_mul_pd(e, s));
            _mm512_storeu_pd(r + 2 * m, _mm512_permutex2var_pd(rr, LO, ri));
            _mm512_storeu_pd(r + 2 * m + 8, _mm512_permutex2var_pd(rr, HI, ri));
        }
    }
#endif
#if defined(__AVX2__)
    for (; m + 4 <= M; m += 4)
    {
        const __m256d E = _mm256_permute4x64_pd(_mm256_unpacklo_pd(_mm256_loadu_pd(x + 2 * m), _mm256_loadu_pd(x + 2 * m + 4)), 0xD8);
        const __m256d O = _mm256_permute4x64_pd(_mm256_unpackhi_pd(_mm256_loadu_pd(x + N - 8 - 2 * m),
                                                                   _mm256_loadu_pd(x + N - 4 - 2 * m)), 0x27);
        const __m256d e = sine ? O : E, o = sine ? E : O;
        const __m256d c = _mm256_loadu_pd(pc + m), s = _mm256_loadu_pd(ps + m);
        const __m256d rr = _mm256_fmadd_pd(e, c, _mm256_mul_pd(o, s)), ri = _mm256_fmsub_pd(o, c, _mm256_mul_pd(e, s));
        const __m256d lo = _mm256_unpacklo_pd(rr, ri), hi = _mm256_unpackhi_pd(rr, ri);
        _mm256_storeu_pd(r + 2 * m, _mm256_permute2f128_pd(lo, hi, 0x20));
        _mm256_storeu_pd(r + 2 * m + 4, _mm256_permute2f128_pd(lo, hi, 0x31));
    }
#endif
    for (; m < M; m++)
    {
        const double e = sine ? x[N - 1 - 2 * m] : x[2 * m], o = sine ? x[2 * m] : x[N - 1 - 2 * m];
        r[2 * m] = e * pc[m] + o * ps[m];
        r[2 * m + 1] = o * pc[m] - e * ps[m];
    }
}

/* the half spectrum R (interleaved, N/2 + 1 bins) into v: v[k] the even outputs Y[2k], v[N/2 + k] the odd
 * ones Y[N-1-2k] (the header's formulas; the sine transform's odd outputs negated). R and v apart. */
static inline void _t4_rotate_even(const double *R, double *v, const double *ac, const double *as, const double *gc,
                                   const double *gs, int N, int sine)
{
    const int M = N / 2;
    int k = 0;
#if defined(__AVX512F__)
    {
        const __m512i RE = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14), IM = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
        const __m512i RER = _mm512_setr_epi64(14, 12, 10, 8, 6, 4, 2, 0), IMR = _mm512_setr_epi64(15, 13, 11, 9, 7, 5, 3, 1);
        const __m512d SG = _mm512_set1_pd(sine ? 0.0 : -0.0);
        for (; k + 8 <= M; k += 8)
        {
            const __m512d z0 = _mm512_loadu_pd(R + 2 * k), z1 = _mm512_loadu_pd(R + 2 * k + 8);
            const __m512d w0 = _mm512_loadu_pd(R + 2 * (M - k - 7)), w1 = _mm512_loadu_pd(R + 2 * (M - k - 7) + 8);
            const __m512d pr = _mm512_permutex2var_pd(z0, RE, z1), pi = _mm512_permutex2var_pd(z0, IM, z1);
            const __m512d qr = _mm512_permutex2var_pd(w0, RER, w1), qi = _mm512_permutex2var_pd(w0, IMR, w1);   /* R[M-k-i] */
            const __m512d a1 = _mm512_add_pd(pr, qr), a2 = _mm512_sub_pd(pi, qi), b1 = _mm512_sub_pd(pr, qr), b2 = _mm512_add_pd(pi, qi);
            const __m512d ca = _mm512_loadu_pd(ac + k), sa = _mm512_loadu_pd(as + k);
            const __m512d cg = _mm512_loadu_pd(gc + k), sg = _mm512_loadu_pd(gs + k);
            const __m512d re = _mm512_sub_pd(_mm512_fmadd_pd(a1, ca, _mm512_mul_pd(a2, sa)), _mm512_fmadd_pd(b1, sg, _mm512_mul_pd(b2, cg)));
            const __m512d im = _mm512_add_pd(_mm512_fmsub_pd(a2, ca, _mm512_mul_pd(a1, sa)), _mm512_fmsub_pd(b1, cg, _mm512_mul_pd(b2, sg)));
            _mm512_storeu_pd(v + k, re);
            _mm512_storeu_pd(v + M + k, _mm512_xor_pd(im, SG));
        }
    }
#endif
#if defined(__AVX2__)
    {
        const __m256d SG = _mm256_set1_pd(sine ? 0.0 : -0.0);
        for (; k + 4 <= M; k += 4)
        {
            const __m256d z0 = _mm256_loadu_pd(R + 2 * k), z1 = _mm256_loadu_pd(R + 2 * k + 4);
            const __m256d w0 = _mm256_loadu_pd(R + 2 * (M - k - 3)), w1 = _mm256_loadu_pd(R + 2 * (M - k - 3) + 4);
            const __m256d pr = _mm256_permute4x64_pd(_mm256_unpacklo_pd(z0, z1), 0xD8);
            const __m256d pi = _mm256_permute4x64_pd(_mm256_unpackhi_pd(z0, z1), 0xD8);
            const __m256d qr = _mm256_permute4x64_pd(_mm256_unpacklo_pd(w0, w1), 0x27);   /* R[M-k-i], i = 0 .. 3 */
            const __m256d qi = _mm256_permute4x64_pd(_mm256_unpackhi_pd(w0, w1), 0x27);
            const __m256d a1 = _mm256_add_pd(pr, qr), a2 = _mm256_sub_pd(pi, qi), b1 = _mm256_sub_pd(pr, qr), b2 = _mm256_add_pd(pi, qi);
            const __m256d ca = _mm256_loadu_pd(ac + k), sa = _mm256_loadu_pd(as + k);
            const __m256d cg = _mm256_loadu_pd(gc + k), sg = _mm256_loadu_pd(gs + k);
            const __m256d re = _mm256_sub_pd(_mm256_fmadd_pd(a1, ca, _mm256_mul_pd(a2, sa)), _mm256_fmadd_pd(b1, sg, _mm256_mul_pd(b2, cg)));
            const __m256d im = _mm256_add_pd(_mm256_fmsub_pd(a2, ca, _mm256_mul_pd(a1, sa)), _mm256_fmsub_pd(b1, cg, _mm256_mul_pd(b2, sg)));
            _mm256_storeu_pd(v + k, re);
            _mm256_storeu_pd(v + M + k, _mm256_xor_pd(im, SG));
        }
    }
#endif
    for (; k < M; k++)
    {
        const double pr = R[2 * k], pi = R[2 * k + 1], qr = R[2 * (M - k)], qi = R[2 * (M - k) + 1];
        const double a1 = pr + qr, a2 = pi - qi, b1 = pr - qr, b2 = pi + qi;
        const double re = a1 * ac[k] + a2 * as[k] - b1 * gs[k] - b2 * gc[k];
        const double im = a2 * ac[k] - a1 * as[k] + b1 * gc[k] - b2 * gs[k];
        v[k] = re;
        v[M + k] = sine ? im : -im;
    }
}

/* ── an odd N: the passes ───────────────────────────────────────────────── */

/* w[j] = +-x[n] or +-x[N-1-n], j = 0 .. N-1 ascending (the header: a = 8j mod N made odd, n = (a-1)/2). The j
 * of one run of a below N are a strided copy: the source steps by 4, up or down, with one sign. x and w
 * apart. */
static inline void _t4_pre_odd(const double *x, double *w, int N, int sine, double psign)
{
    int j = 0, a = 0;
    while (j < N)
    {
        const int cnt = (N - a + 7) / 8;                   /* the j until a reaches N */
        const int a2 = (a & 1) ? a : a + N;                /* a made odd: the same residue mod N */
        const int r8 = a2 & 7;
        const int partner = (r8 == 3 || r8 == 7);          /* the sample is x[N-1-n] */
        const double sign = r8 == 1 ? 1.0 : r8 == 5 ? -1.0 : r8 == 3 ? -psign : psign;
        const double *src = x + (a2 - 1) / 2;
        int step = 4, i;
        if (partner != sine)
        {   /* the index reversed: the partner's pick, or the sine transform's reversed input (both: neither) */
            src = x + (N - 1 - (a2 - 1) / 2);
            step = -4;
        }
        if (sign > 0.0)
            for (i = 0; i < cnt; i++)
                w[j + i] = src[(ptrdiff_t)step * i];
        else
            for (i = 0; i < cnt; i++)
                w[j + i] = -src[(ptrdiff_t)step * i];
        j += cnt;
        a += 8 * cnt - N;
    }
}

/* Y[k] = sqrt 2 (+-Re W[l] +- Im W[l]), k = 0 .. N-1, from the half spectrum W (interleaved, (N+1)/2 bins):
 * l = (2k+1) mod N folded to the half spectrum, the signs from (2k+1) s8 mod 8 (the header). W and y apart. */
static inline void _t4_post_odd(const double *W, double *y, int N, int sine, int s8)
{
    static const double SR[4] = { 1.0, -1.0, -1.0, 1.0 }, SI[4] = { 1.0, 1.0, -1.0, -1.0 };   /* q = 1, 3, 5, 7 */
    const double SQ2 = 1.41421356237309504880168872420969808;
    const int h = (N - 1) / 2;
    int k;
    for (k = 0; k < N; k++)
    {
        const int b = 2 * k + 1, q = ((b & 7) * s8) & 7;
        double sr = SQ2 * SR[q >> 1], si = SQ2 * SI[q >> 1];
        int l = b < N ? b : b - N;
        if (l > h)
        {
            l = N - l;
            si = -si;
        }
        if (sine && (k & 1))
        {
            sr = -sr;
            si = -si;
        }
        y[k] = l ? sr * W[2 * l] + si * W[2 * l + 1] : sr * W[0];   /* bin 0 is real */
    }
}

/* ── execute ────────────────────────────────────────────────────────────── */

/* the transform of x into y (y may be x); `in` the buffer the real FFT reads, apart from x */
static inline void _t4_run(const vfft_type4_t *d, const double *x, double *y)
{
    const int N = d->N;
    if (N == 1)
    {
        y[0] = 1.41421356237309504880168872420969808 * x[0];   /* 2 cos(pi/4) = 2 sin(pi/4) */
        return;
    }
    {
        double *in = (y != x) ? y : d->V;   /* the caller's output holds the FFT's input when it is free */
        if (N & 1)
        {
            _t4_pre_odd(x, in, N, d->sine, d->psign);
            _vfft_exec6((vfft_plan)d->r2c, VFFT_FORWARD, in, NULL, d->W, NULL);
            _t4_post_odd(d->W, y, N, d->sine, d->s8);
        }
        else
        {
            _t4_pre_even(x, in, d->pc, d->ps, N, d->sine);
            _vfft_exec6((vfft_plan)d->r2c, VFFT_FORWARD, in, NULL, d->W, NULL);
            _t4_rotate_even(d->W, d->V, d->ac, d->as, d->gc, d->gs, N, d->sine);
            _dct_unreorder(d->V, y, N);
        }
    }
}

/* the bound executor (vfft_plan_s.k1_exec, k1_go): never declines, either pointer pattern, either direction
 * (each transform is its own inverse) */
static int _k1x_type4(struct vfft_plan_s *h, vfft_dir_t dir, const vfft_real *zin, vfft_real *zout)
{
    (void)dir;
    _t4_run(h->type4, zin, zout);
    return 0;
}

/* ── the role, create and destroy ───────────────────────────────────────── */

/* THE ROLE (il/real/real_role.h): one whole transform with the candidate FFT h inside, on the create's own
 * buffers x (the input) and y (the output), the plan's tables and work buffers. What every candidate of the
 * real door's races is timed as while the transform's FFT is created. */
typedef struct
{
    const vfft_type4_t *d;
    double *x, *y;
} _t4_role_t;
static void _t4_role_run(void *ctx, struct vfft_plan_s *h)
{
    const _t4_role_t *r = (const _t4_role_t *)ctx;
    const vfft_type4_t *d = r->d;
    const int N = d->N;
    if (N & 1)
    {
        _t4_pre_odd(r->x, r->y, N, d->sine, d->psign);
        _real_il_exec_any(h, r->y, d->W);
        _t4_post_odd(d->W, r->y, N, d->sine, d->s8);
    }
    else
    {
        _t4_pre_even(r->x, r->y, d->pc, d->ps, N, d->sine);
        _real_il_exec_any(h, r->y, d->W);
        _t4_rotate_even(d->W, d->V, d->ac, d->as, d->gc, d->gs, N, d->sine);
        _dct_unreorder(d->V, r->y, N);
    }
}

static void _vfft_type4_destroy(vfft_type4_t *d)
{
    if (!d)
        return;
    if (d->r2c)
        vfft_destroy((vfft_plan)d->r2c);
    vfft_child_store_free(d->S);
    vfft_aligned_free(d->W);
    vfft_aligned_free(d->V);
    vfft_aligned_free(d->tab);
    free(d);
}

/* THE DCT-IV / DST-IV DOOR: a VFFT_DCT4 or VFFT_DST4 request, 1D, one transform. est = 1
 * (VFFT_FLAG_ESTIMATE): the real FFT planned by formulas, nothing measured, no store (W unused); est = 0: a
 * child in role of the cell's row in W. NULL with the reason on the error record. */
static vfft_plan _vfft_create_type4_il(const vfft_cfg_t *cfg, struct vfft_wisdom_s *W, int est)
{
    const int N = cfg->n[0], sine = cfg->transform == VFFT_DST4;
    const char *tn = _vfft_tname(cfg->transform);
    struct vfft_plan_s *h;
    vfft_type4_t *d;
    if (cfg->dims >= 2)
    {
        _vfft_warn("vfft_create: %dD %s is not served: DCT-IV and DST-IV plans are 1D", cfg->dims, tn);
        return NULL;
    }
    if (cfg->howmany > 1)
    {
        _vfft_warn("vfft_create: %s N=%d howmany=%zu is not served: a DCT-IV or DST-IV plan runs one transform "
                   "(create it with howmany = 1 and call it once per transform)", tn, N, cfg->howmany);
        return NULL;
    }
    if (N < 1)
    {
        _vfft_warn("vfft_create: %s N=%d invalid (the length must be 1 or more)", tn, N);
        return NULL;
    }
    d = (vfft_type4_t *)calloc(1, sizeof *d);
    h = (struct vfft_plan_s *)calloc(1, sizeof *h);
    if (!d || !h)
    {
        free(d);
        free(h);
        _vfft_warn("vfft_create: %s N=%d: out of memory", tn, N);
        return NULL;
    }
    d->N = N;
    d->sine = sine;
    if (N > 1)
    {
        const size_t M = (size_t)N / 2;
        const int even = !(N & 1);
        double *x = NULL, *y = NULL;   /* the race's input and output (measured plans) */
        _t4_role_t role;
        _real_child_t rc;
        int changed = 0;
        d->W = (double *)vfft_aligned_alloc(((size_t)N + 2) * sizeof(double));
        d->V = (double *)vfft_aligned_alloc((size_t)N * sizeof(double));
        if (even)
            d->tab = (double *)vfft_aligned_alloc(6 * M * sizeof(double));
        if (!est)
        {
            x = (double *)vfft_aligned_alloc((size_t)N * sizeof(double));
            y = (double *)vfft_aligned_alloc((size_t)N * sizeof(double));
        }
        if (!d->W || !d->V || (even && !d->tab) || (!est && (!x || !y)))
        {
            vfft_aligned_free(x);
            vfft_aligned_free(y);
            _vfft_type4_destroy(d);
            free(h);
            _vfft_warn("vfft_create: %s N=%d: out of memory (the work buffers)", tn, N);
            return NULL;
        }
        if (even)
        {
            d->pc = d->tab;
            d->ps = d->tab + M;
            d->ac = d->tab + 2 * M;
            d->as = d->tab + 3 * M;
            d->gc = d->tab + 4 * M;
            d->gs = d->tab + 5 * M;
            vfft_cs2pi_table(2LL * N, (long long)M, d->pc, d->ps);   /* cos, sin(2 pi m / 2N) */
            for (size_t k = 0; k < M; k++)
            {   /* alpha_k = 2 pi (4k+1) / 8N; gamma_k = 2 pi (4k-1) / 8N, at k = 0 the angle -pi/4N as 8N - 1 */
                vfft_cs2pi_exact(4LL * (long long)k + 1, 8LL * N, d->ac + k, d->as + k);
                vfft_cs2pi_exact(k ? 4LL * (long long)k - 1 : 8LL * N - 1, 8LL * N, d->gc + k, d->gs + k);
            }
        }
        else
        {
            d->s8 = N & 7;                             /* an odd N is its own inverse mod 8 */
            d->psign = (N & 3) == 1 ? -1.0 : 1.0;
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
        role.d = d;
        role.x = x;
        role.y = y;
        rc.t = sine ? VW2_T_DST4 : VW2_T_DCT4;
        rc.N = N;
        rc.pre = sine ? "dst_" : "dct_";
        rc.eng = sine ? "dst" : "dct";
        rc.n = N;
        rc.is_c2r = 0;
        rc.run = _t4_role_run;
        rc.ctx = &role;
        d->r2c = _real_child_create(cfg, W, &rc, est, &d->S, &changed);
        vfft_aligned_free(x);
        vfft_aligned_free(y);
        if (changed)
            _vw2_persist(W, cfg);   /* the row banked is the cell's, whatever follows */
        if (!d->r2c)
        {   /* the real door's own reason is on the record: it names the cell that could not be built */
            _vfft_type4_destroy(d);
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
    h->type4 = d;
    h->k1_exec = _k1x_type4;
    h->k1_go[VFFT_FORWARD] = _k1x_type4;
    h->k1_go[VFFT_BACKWARD] = _k1x_type4;
    return (vfft_plan)h;
}

#endif /* !VFFT_FLOAT */
#endif /* VFFT_IL_REAL_DCT4_DST4_H */
