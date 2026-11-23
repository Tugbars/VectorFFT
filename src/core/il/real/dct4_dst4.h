/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Tuğbars Heptaşkın */
/* dct4_dst4.h -- DCT-IV and DST-IV (VFFT_DCT4, VFFT_DST4): 1D, N reals in and N reals out, any N, one
 * transform: at an even N on the interleaved complex FFT of half the length, at an odd N on the interleaved
 * real FFT of the same length.
 *
 * THE DEFINITIONS (include/vfft.h; unnormalized, each its own inverse up to 2N):
 *   DCT-IV   Y[k] = 2 sum_{n=0..N-1} x[n] cos(pi (2k+1)(2n+1) / 4N)
 *   DST-IV   Y[k] = 2 sum_{n=0..N-1} x[n] sin(pi (2k+1)(2n+1) / 4N)
 * The sine transform is the cosine transform of the input read backwards, with every other output negated:
 * DST-IV(x)[k] = (-1)^k DCT-IV(x reversed)[k]. Both of those ride in the passes below.
 *
 * THE METHOD: one FFT between two passes. Every step is a permutation, a sign, or a rotation
 * by a unit complex number, so the error stays that of the FFT at every length; no step multiplies or divides
 * by a cosine that passes through zero (the route through a same-length DCT-II or DCT-III and a running
 * difference does, and loses digits as N grows).
 *
 * EVEN N, M = N/2. The kernel factors over the pairs z[m] = x[2m] + i x[N-1-2m]:
 *   U[k] = sum_m z[m] e^(-i pi (4m+1)(4k+1) / 4N),   Y[2k] = 2 Re U[k],   Y[N-1-2k] = -2 Im U[k],
 * and (4m+1)(4k+1) = 16mk + 4m + 4k + 1 makes U a length-M complex DFT between two rotations:
 *   U[k] = e^(-i pi (4k+1) / 4N) Z[k],   Z = DFT_M( z[m] e^(-i pi m / N) ).
 * In reals, with alpha_k = pi (4k+1) / 4N:
 *   pre      t[m] = z[m] e^(-i pi m/N):  Re t[m] = x[2m] cos(pi m/N) + x[N-1-2m] sin(pi m/N)
 *                                        Im t[m] = x[N-1-2m] cos(pi m/N) - x[2m] sin(pi m/N)
 *   C2C at M Z[0 .. M-1], forward
 *   post     Y[2k]     = 2 (cos alpha_k Re Z[k] + sin alpha_k Im Z[k])
 *            Y[N-1-2k] = 2 (sin alpha_k Re Z[k] - cos alpha_k Im Z[k])
 * DST-IV: the pairs are x[N-1-2m] + i x[2m], and Y[N-1-2k] changes sign.
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
 * THE PLAN holds its FFT, out of place -- at an even N a C2C at N/2 (a plan of the complex door, natural order),
 * at an odd N an R2C at N (a plan of the real door) -- the tables of an even N and two work buffers allocated
 * at create: V, N doubles, the FFT's input; W, N + 2 doubles, its output. Either transform takes out == in.
 * A plan of T threads threads inside its FFT; the passes run on the calling thread. Forward and backward are
 * the same transform.
 *
 * THE WISDOM: the cell has its own row in the real shard --
 *   t=dct4 n=N q=1 ord=nat place=oop [nthreads=T] | eng=dct and its FFT under dct_
 *   t=dst4 n=N q=1 ord=nat place=oop [nthreads=T] | eng=dst and its FFT under dst_
 * (vw2_r2r_key, il/wisdom/wisdom2_real_il.h). The FFT is a CHILD IN ROLE (il/wisdom/wisdom2_child.h): created
 * against a private store seeded from the cell's row, it replays the recipe there or races, and what it banked
 * goes back onto the cell's row; the cell never reads or writes a c2c, an r2c or a c2r row. The R2C of an odd
 * N races in the cell's role (il/real/real_child.h, real_role.h): every candidate FFT is timed as one whole
 * transform, the passes with that FFT between them. The C2C of an even N races in its private store as the
 * complex door races a component, the way the 2D tier's row plan does. The plan is placement-blind, so one
 * row serves both placements. Under VFFT_FLAG_ESTIMATE the FFT is planned by formulas, nothing is measured
 * and no store is read or written.
 *
 * POSITION IN vfft.c: after il/real/real_create_il.h (the real door) and before the front door's create. */
#ifndef VFFT_IL_REAL_DCT4_DST4_H
#define VFFT_IL_REAL_DCT4_DST4_H
#ifndef VFFT_FLOAT   /* FP64 only: the FP32 unit serves 1D C2C */

#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

#include "common/math/tw_exact.h"   /* the rotation tables, each entry rounded once */
#include "il/real/real_child.h"     /* the real FFT as a child in role of the cell's row */

typedef struct vfft_type4_s
{
    int N;
    int sine;                    /* 1: DST-IV; 0: DCT-IV */
    struct vfft_plan_s *fft;     /* the FFT, out of place (owned): the C2C at N/2 of an even N, the R2C at N of
                                  * an odd one; NULL at N = 1 and N = 2 */
    struct vfft_wisdom_s *S;     /* its private store: owned, never saved; NULL under VFFT_FLAG_ESTIMATE and
                                  * where there is no FFT */
    double *W;                   /* N + 2 doubles: the FFT's output */
    double *V;                   /* N doubles: the FFT's input */
    double *tab;                 /* even N: the four tables below, one allocation, N/2 entries each */
    double *pc, *ps;             /* cos, sin(pi m / N) */
    double *ac, *as;             /* 2 cos, 2 sin(pi (4k+1) / 4N) */
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

/* the complex spectrum Z (interleaved, N/2 bins) into the N outputs: Y[2k] = c Re Z[k] + s Im Z[k] and
 * Y[N-1-2k] = s Re Z[k] - c Im Z[k] (the sine transform: negated), c, s = ac[k], as[k], the doubled tables. Bin k
 * and bin N/2 - 1 - k fill the outputs 2k, 2k + 1 and their mirror, so the two ends are taken together and every
 * store is a whole vector. Z and y apart. */
static inline void _t4_post_even(const double *Z, double *y, const double *ac, const double *as, int N, int sine)
{
    const int M = N / 2;
    int k = 0, m;
#if defined(__AVX512F__)
    {
        const __m512i RE = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14), IM = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
        const __m512i REV = _mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
        const __m512i LO = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11), HI = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
        const __m512d SG = _mm512_set1_pd(sine ? -0.0 : 0.0);
        for (; 2 * k + 16 <= M; k += 8)
        {
            const int j = M - 8 - k;   /* the mirror block's first bin */
            const __m512d a0 = _mm512_loadu_pd(Z + 2 * k), a1 = _mm512_loadu_pd(Z + 2 * k + 8);
            const __m512d b0 = _mm512_loadu_pd(Z + 2 * j), b1 = _mm512_loadu_pd(Z + 2 * j + 8);
            const __m512d ar = _mm512_permutex2var_pd(a0, RE, a1), ai = _mm512_permutex2var_pd(a0, IM, a1);
            const __m512d br = _mm512_permutex2var_pd(b0, RE, b1), bi = _mm512_permutex2var_pd(b0, IM, b1);
            const __m512d ca = _mm512_loadu_pd(ac + k), sa = _mm512_loadu_pd(as + k);
            const __m512d cb = _mm512_loadu_pd(ac + j), sb = _mm512_loadu_pd(as + j);
            const __m512d ea = _mm512_fmadd_pd(ca, ar, _mm512_mul_pd(sa, ai));                       /* Y[2 (k + i)] */
            const __m512d eb = _mm512_fmadd_pd(cb, br, _mm512_mul_pd(sb, bi));                       /* Y[2 (j + i)] */
            const __m512d oa = _mm512_xor_pd(_mm512_fmsub_pd(sa, ar, _mm512_mul_pd(ca, ai)), SG);    /* Y[2 (j + 7 - i) + 1] */
            const __m512d ob = _mm512_xor_pd(_mm512_fmsub_pd(sb, br, _mm512_mul_pd(cb, bi)), SG);    /* Y[2 (k + 7 - i) + 1] */
            const __m512d oar = _mm512_permutexvar_pd(REV, oa), obr = _mm512_permutexvar_pd(REV, ob);
            _mm512_storeu_pd(y + 2 * k, _mm512_permutex2var_pd(ea, LO, obr));
            _mm512_storeu_pd(y + 2 * k + 8, _mm512_permutex2var_pd(ea, HI, obr));
            _mm512_storeu_pd(y + 2 * j, _mm512_permutex2var_pd(eb, LO, oar));
            _mm512_storeu_pd(y + 2 * j + 8, _mm512_permutex2var_pd(eb, HI, oar));
        }
    }
#endif
#if defined(__AVX2__)
    {
        const __m256d SG = _mm256_set1_pd(sine ? -0.0 : 0.0);
        for (; 2 * k + 8 <= M; k += 4)
        {
            const int j = M - 4 - k;   /* the mirror block's first bin */
            const __m256d a0 = _mm256_loadu_pd(Z + 2 * k), a1 = _mm256_loadu_pd(Z + 2 * k + 4);
            const __m256d b0 = _mm256_loadu_pd(Z + 2 * j), b1 = _mm256_loadu_pd(Z + 2 * j + 4);
            const __m256d ar = _mm256_permute4x64_pd(_mm256_unpacklo_pd(a0, a1), 0xD8);
            const __m256d ai = _mm256_permute4x64_pd(_mm256_unpackhi_pd(a0, a1), 0xD8);
            const __m256d br = _mm256_permute4x64_pd(_mm256_unpacklo_pd(b0, b1), 0xD8);
            const __m256d bi = _mm256_permute4x64_pd(_mm256_unpackhi_pd(b0, b1), 0xD8);
            const __m256d ca = _mm256_loadu_pd(ac + k), sa = _mm256_loadu_pd(as + k);
            const __m256d cb = _mm256_loadu_pd(ac + j), sb = _mm256_loadu_pd(as + j);
            const __m256d ea = _mm256_fmadd_pd(ca, ar, _mm256_mul_pd(sa, ai));                       /* Y[2 (k + i)] */
            const __m256d eb = _mm256_fmadd_pd(cb, br, _mm256_mul_pd(sb, bi));                       /* Y[2 (j + i)] */
            const __m256d oa = _mm256_xor_pd(_mm256_fmsub_pd(sa, ar, _mm256_mul_pd(ca, ai)), SG);    /* Y[2 (j + 3 - i) + 1] */
            const __m256d ob = _mm256_xor_pd(_mm256_fmsub_pd(sb, br, _mm256_mul_pd(cb, bi)), SG);    /* Y[2 (k + 3 - i) + 1] */
            const __m256d oar = _mm256_permute4x64_pd(oa, 0x1B), obr = _mm256_permute4x64_pd(ob, 0x1B);
            const __m256d l0 = _mm256_unpacklo_pd(ea, obr), h0 = _mm256_unpackhi_pd(ea, obr);
            const __m256d l1 = _mm256_unpacklo_pd(eb, oar), h1 = _mm256_unpackhi_pd(eb, oar);
            _mm256_storeu_pd(y + 2 * k, _mm256_permute2f128_pd(l0, h0, 0x20));
            _mm256_storeu_pd(y + 2 * k + 4, _mm256_permute2f128_pd(l0, h0, 0x31));
            _mm256_storeu_pd(y + 2 * j, _mm256_permute2f128_pd(l1, h1, 0x20));
            _mm256_storeu_pd(y + 2 * j + 4, _mm256_permute2f128_pd(l1, h1, 0x31));
        }
    }
#endif
    for (m = k; m < M - k; m++)
    {   /* the bins between the two ends */
        const double zr = Z[2 * m], zi = Z[2 * m + 1];
        const double o = as[m] * zr - ac[m] * zi;
        y[2 * m] = ac[m] * zr + as[m] * zi;
        y[N - 1 - 2 * m] = sine ? -o : o;
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

/* the transform of x into y (y may be x) */
static inline void _t4_run(const vfft_type4_t *d, const double *x, double *y)
{
    const int N = d->N;
    if (N == 1)
    {
        y[0] = 1.41421356237309504880168872420969808 * x[0];   /* 2 cos(pi/4) = 2 sin(pi/4) */
        return;
    }
    if (N & 1)
    {
        double *in = (y != x) ? y : d->V;   /* the caller's output holds the FFT's input when it is free */
        _t4_pre_odd(x, in, N, d->sine, d->psign);
        _vfft_exec6((vfft_plan)d->fft, VFFT_FORWARD, in, NULL, d->W, NULL);
        _t4_post_odd(d->W, y, N, d->sine, d->s8);
    }
    else
    {
        const double *Z = d->V;             /* N = 2: the DFT of one point is the point */
        _t4_pre_even(x, d->V, d->pc, d->ps, N, d->sine);
        if (d->fft)
        {
            _vfft_exec6((vfft_plan)d->fft, VFFT_FORWARD, d->V, NULL, d->W, NULL);
            Z = d->W;
        }
        _t4_post_even(Z, y, d->ac, d->as, N, d->sine);
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

/* ── the role, the children, create and destroy ───────────────────────── */

/* THE ROLE OF AN ODD N (il/real/real_role.h): one whole transform with the candidate R2C h inside, on the
 * create's own buffers x (the input) and y (the output) and the plan's work buffer. What every candidate of the
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
    _t4_pre_odd(r->x, r->y, d->N, d->sine, d->psign);
    _real_il_exec_any(h, r->y, d->W);
    _t4_post_odd(d->W, r->y, d->N, d->sine, d->s8);
}

static void _vfft_type4_destroy(vfft_type4_t *d)
{
    if (!d)
        return;
    if (d->fft)
        vfft_destroy((vfft_plan)d->fft);
    vfft_child_store_free(d->S);
    vfft_aligned_free(d->W);
    vfft_aligned_free(d->V);
    vfft_aligned_free(d->tab);
    free(d);
}

/* The C2C at N/2 of an even N, out of place, natural order, for the request cfg. est: planned by formulas. Else a
 * child in role (il/wisdom/wisdom2_child.h): its private store d->S seeded from the cell's row in W (empty when
 * the row carries no recipe, or there is no store), the create against it with nothing persisted; where the child
 * raced, or the row did not carry it yet, the cell's row is banked afresh with the child's rows under its prefix
 * (*changed = 1: the caller saves). NULL: the complex door's reason is on the record. */
static struct vfft_plan_s *_t4_c2c_child(const vfft_cfg_t *cfg, struct vfft_wisdom_s *W, vfft_type4_t *d, int est,
                                         int *changed)
{
    const char *pre = d->sine ? "dst_" : "dct_", *eng = d->sine ? "dst" : "dct";
    const int t = d->sine ? VW2_T_DST4 : VW2_T_DCT4;
    vfft_cfg_t c = *cfg;
    c.transform = VFFT_C2C;
    c.dims = 1;
    c.n[0] = d->N / 2;
    c.howmany = 1;
    c.layout = VFI_LAYOUT_IL;
    c.placement = VFFT_OUTOFPLACE;
    c.order = VFI_ORDER_NATURAL;
    c.batch_geom = VFFT_BATCH_DEFAULT;
    c.owned_buffers = 0;
    c.destroy_input = 0;
    if (est)
        return (struct vfft_plan_s *)_vfft_create_estimate(&c);
    {
        const int T = _vfft_plan_threads(cfg), usable = W && !W->vw2_off_oop;
        struct vfft_plan_s *ch;
        vw2_key_t pk;
        vw2_r2r_key(&pk, t, d->N, T);
        d->S = vfft_child_store_for(usable ? &W->vw2 : NULL, &pk, pre);
        if (!d->S)
        {
            _vfft_warn("vfft_create: %s N=%d: out of memory (the complex FFT's store)", _vfft_tname(cfg->transform), d->N);
            return NULL;
        }
        c.wisdom = (vfft_wisdom *)d->S;
        c.wisdom_write = 0;   /* a private store's plan never saves (il/wisdom/wisdom2_child.h) */
        ch = (struct vfft_plan_s *)_vfft_create_cfg(&c);
        if (ch && usable && d->S->vw2.nrec > 0 &&
            (vfft_child_store_raced(d->S) || !vfft_child_row_has(vfft_child_parent_row(&W->vw2, &pk), pre)))
        {
            if (vw2_r2r_bank(&W->vw2, &pk, eng, 0.0) == VW2_OK && vfft_child_row_update(&W->vw2, &pk, pre, d->S))
                *changed = 1;
            else
                fprintf(stderr, "vfft: %s verdict NOT banked at N=%d -- the cell will re-race\n", vw2_t_name[t], d->N);
        }
        return ch;
    }
}

/* THE DCT-IV / DST-IV DOOR: a VFFT_DCT4 or VFFT_DST4 request, 1D, one transform. est = 1
 * (VFFT_FLAG_ESTIMATE): the FFT planned by formulas, nothing measured, no store (W unused); est = 0: a child in
 * role of the cell's row in W. NULL with the reason on the error record. */
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
        int changed = 0;
        d->W = (double *)vfft_aligned_alloc(((size_t)N + 2) * sizeof(double));
        d->V = (double *)vfft_aligned_alloc((size_t)N * sizeof(double));
        if (even)
            d->tab = (double *)vfft_aligned_alloc(4 * M * sizeof(double));
        if (!d->W || !d->V || (even && !d->tab))
        {
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
            vfft_cs2pi_table(2LL * N, (long long)M, d->pc, d->ps);   /* cos, sin(2 pi m / 2N) */
            for (size_t k = 0; k < M; k++)
            {   /* alpha_k = 2 pi (4k+1) / 8N, the two doubled */
                vfft_cs2pi_exact(4LL * (long long)k + 1, 8LL * N, d->ac + k, d->as + k);
                d->ac[k] *= 2.0;
                d->as[k] *= 2.0;
            }
            if (M > 1)
                d->fft = _t4_c2c_child(cfg, W, d, est, &changed);
        }
        else
        {
            double *x = NULL, *y = NULL;   /* the race's input and output (measured plans) */
            _t4_role_t role;
            _real_child_t rc;
            d->s8 = N & 7;                             /* an odd N is its own inverse mod 8 */
            d->psign = (N & 3) == 1 ? -1.0 : 1.0;
            if (!est)
            {
                x = (double *)vfft_aligned_alloc((size_t)N * sizeof(double));
                y = (double *)vfft_aligned_alloc((size_t)N * sizeof(double));
                if (!x || !y)
                {
                    vfft_aligned_free(x);
                    vfft_aligned_free(y);
                    _vfft_type4_destroy(d);
                    free(h);
                    _vfft_warn("vfft_create: %s N=%d: out of memory (the race's buffers)", tn, N);
                    return NULL;
                }
                {
                    unsigned sd = 0x9e3779b9u ^ (unsigned)N;
                    for (int i = 0; i < N; i++)
                    {
                        sd = sd * 1664525u + 1013904223u;
                        x[i] = (double)(sd >> 8) / (double)(1u << 24) - 0.5;
                        y[i] = 0.0;
                    }
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
            d->fft = _real_child_create(cfg, W, &rc, est, &d->S, &changed);
            vfft_aligned_free(x);
            vfft_aligned_free(y);
        }
        if (changed)
            _vw2_persist(W, cfg);   /* the row banked is the cell's, whatever follows */
        if (!d->fft && (!even || M > 1))
        {   /* the door's own reason is on the record: it names the cell that could not be built */
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
