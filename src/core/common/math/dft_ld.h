/* dft_ld.h -- THE LONG-DOUBLE DFT OF CREATE TIME (2026-10-17).
 *
 * Some create-time tables are the DFT of a known sequence: the Rader kernel is the DFT of length
 * M = p - 1 of w[m] = e^(sign 2 pi i g^(-m) / p). Built by the engine's own double transform, such a table
 * carries that transform's rounding into every product of the convolution it feeds (one of a Rader
 * level's three passes of rounding). Here the same DFT runs in long double and each entry is rounded
 * once to double.
 *
 * The transform: Cooley-Tukey (decimation in time) by radix 4 where 4 divides the length, else over its
 * prime factors up to 64: radix 2 and 4 by their butterflies, a larger factor's by the definition (O(n r)
 * per level); a remaining part whose every factor is above 64 by the
 * definition up to 256 points, else by Bluestein over a radix-2 transform of a power of two; every
 * twiddle and chirp from tw_exact.h's fold, kept in long double. O(n log n) except the definition's small
 * pieces; memory a few arrays of n long double complex. Create-time only: nothing here runs on an
 * execute path. */
#ifndef VFFT_DFT_LD_H
#define VFFT_DFT_LD_H
#include <stdlib.h>
#include <string.h>
#include "common/math/tw_exact.h"

typedef struct
{
    long double re, im;
} vfft_cld_t;

/* the table e^(2 pi i j / n), j in [0, count): each entry as vfft_cs2pi_ld(j, n) gives it, the first octant
 * evaluated and the rest taken by the fold's swap and flips (vfft_cs2pi_table's rule, in long double) */
static inline void vfft_cs2pi_ld_table(long long n, long long count, vfft_cld_t *w)
{
    const long long nb = n / 8 + 1;
    vfft_cld_t *b = (vfft_cld_t *)malloc((size_t)nb * sizeof(vfft_cld_t));
    long long f;
    if (!b)
    {
        for (f = 0; f < count; f++)
            vfft_cs2pi_ld(f, n, &w[f].re, &w[f].im);
        return;
    }
    for (f = 0; f < nb; f++)
        vfft_cs2pi_ld(f, n, &b[f].re, &b[f].im); /* t = 8f <= n: no fold */
    for (f = 0; f < count; f++)
    {
        long long t = 8 * (f % n);
        int flip_s = 0, flip_c = 0, swap = 0;
        long double cv, sv;
        if (t > 4 * n) { t = 8 * n - t; flip_s = 1; }
        if (t > 2 * n) { t = 4 * n - t; flip_c = 1; }
        if (t > n)     { t = 2 * n - t; swap = 1; }
        if (t % 8 != 0)
        {
            vfft_cs2pi_ld(f, n, &w[f].re, &w[f].im);
            continue;
        }
        cv = b[t / 8].re;
        sv = b[t / 8].im;
        if (swap) { const long double tmp = cv; cv = sv; sv = tmp; }
        if (flip_c) cv = -cv;
        if (flip_s) sv = -sv;
        w[f].re = cv;
        w[f].im = sv;
    }
    free(b);
}

/* radix 2 in place over a power of two L: x[k] = sum_j x[j] e^(sign 2 pi i j k / L); w = e^(2 pi i j / L), j < L/2 */
static inline void _vfft_ld_fft2(long long L, vfft_cld_t *x, int sign, const vfft_cld_t *w)
{
    long long i, j = 0, len, k;
    for (i = 1; i < L; i++)
    {   /* bit reversal */
        long long bit = L >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) { const vfft_cld_t t = x[i]; x[i] = x[j]; x[j] = t; }
    }
    for (len = 2; len <= L; len <<= 1)
    {
        const long long half = len >> 1, step = L / len;
        for (i = 0; i < L; i += len)
            for (k = 0; k < half; k++)
            {
                const long double wr = w[k * step].re, wi = sign * w[k * step].im;
                vfft_cld_t *u = &x[i + k], *v = &x[i + k + half];
                const long double tr = v->re * wr - v->im * wi, ti = v->re * wi + v->im * wr;
                v->re = u->re - tr;
                v->im = u->im - ti;
                u->re += tr;
                u->im += ti;
            }
    }
}

/* Bluestein: X[k] = sum_j x[j xs] e^(sign 2 pi i j k / n) for any n, over a radix-2 transform of L >= 2n - 1;
 * the chirp c_j = e^(sign pi i j^2 / n) from the integers (j^2 mod 2n). 0 when memory runs out. */
static inline int _vfft_ld_bluestein(long long n, const vfft_cld_t *x, long long xs, vfft_cld_t *X, int sign)
{
    long long L = 1, j;
    vfft_cld_t *a, *b, *c, *w;
    while (L < 2 * n - 1)
        L <<= 1;
    a = (vfft_cld_t *)calloc((size_t)(2 * L + n + L / 2), sizeof(vfft_cld_t));
    if (!a)
        return 0;
    b = a + L;
    c = b + L;
    w = c + n;
    vfft_cs2pi_ld_table(L, L / 2, w);
    for (j = 0; j < n; j++)
    {
        vfft_cs2pi_ld((j * j) % (2 * n), 2 * n, &c[j].re, &c[j].im);
        c[j].im *= sign;
        a[j].re = x[j * xs].re * c[j].re - x[j * xs].im * c[j].im;
        a[j].im = x[j * xs].re * c[j].im + x[j * xs].im * c[j].re;
    }
    b[0].re = c[0].re;
    b[0].im = -c[0].im;
    for (j = 1; j < n; j++)
    {
        b[j].re = b[L - j].re = c[j].re;
        b[j].im = b[L - j].im = -c[j].im;
    }
    _vfft_ld_fft2(L, a, -1, w);
    _vfft_ld_fft2(L, b, -1, w);
    for (j = 0; j < L; j++)
    {
        const long double r = a[j].re * b[j].re - a[j].im * b[j].im, i = a[j].re * b[j].im + a[j].im * b[j].re;
        a[j].re = r;
        a[j].im = i;
    }
    _vfft_ld_fft2(L, a, 1, w);
    for (j = 0; j < n; j++)
    {   /* X_k = c_k conv_k / L */
        const long double r = a[j].re / (long double)L, i = a[j].im / (long double)L;
        X[j].re = c[j].re * r - c[j].im * i;
        X[j].im = c[j].re * i + c[j].im * r;
    }
    free(a);
    return 1;
}

/* the smallest prime factor of n (n itself when prime) */
static inline long long _vfft_ld_spf(long long n)
{
    long long r;
    if (n % 2 == 0)
        return 2;
    for (r = 3; r * r <= n; r += 2)
        if (n % r == 0)
            return r;
    return n;
}

/* the radix of a level: 4 where 4 divides n, else the smallest prime factor */
static inline long long _vfft_ld_radix(long long n)
{
    return n % 4 == 0 ? 4 : _vfft_ld_spf(n);
}

/* one level's twiddles: e^(2 pi i e / n) = w[e st], e < n */
typedef struct
{
    const vfft_cld_t *w;
    long long st;
} _vfft_ld_lvl_t;

/* y[k] = sum_j x[j xs] e^(sign 2 pi i j k / n), k < n; L = this level's twiddles, L + 1 the next level's.
 * Radix 2 and 4 combine by their butterflies, the others by the definition. A twiddle index q k stays
 * below n (q < r, k < n / r). */
static inline int _vfft_ld_rec(long long n, const vfft_cld_t *x, long long xs, vfft_cld_t *y, int sign,
                               const _vfft_ld_lvl_t *L)
{
    const vfft_cld_t *tw = L->w;
    const long long st = L->st, r = _vfft_ld_radix(n);
    long long m, q, k, s;
    vfft_cld_t t[64];
    if (n == 1)
    {
        y[0] = x[0];
        return 1;
    }
    if (r > 64)
    {   /* every factor above 64: the definition up to 256 points, else Bluestein */
        if (n > 256)
            return _vfft_ld_bluestein(n, x, xs, y, sign);
        for (k = 0; k < n; k++)
        {
            long double sr = 0, si = 0;
            long long e = 0;
            for (q = 0; q < n; q++)
            {
                const long double wr = tw[e * st].re, wi = sign * tw[e * st].im;
                sr += x[q * xs].re * wr - x[q * xs].im * wi;
                si += x[q * xs].re * wi + x[q * xs].im * wr;
                e += k;
                if (e >= n) e -= n;
            }
            y[k].re = sr;
            y[k].im = si;
        }
        return 1;
    }
    m = n / r;
    for (q = 0; q < r; q++)
        if (!_vfft_ld_rec(m, x + q * xs, xs * r, y + q * m, sign, L + 1))
            return 0;
    if (r == 2)
    {   /* X[k] = Y_0[k] + w_n^k Y_1[k], X[k + m] = Y_0[k] - w_n^k Y_1[k] */
        for (k = 0; k < m; k++)
        {
            const long double wr = tw[k * st].re, wi = sign * tw[k * st].im;
            vfft_cld_t *u = &y[k], *v = &y[k + m];
            const long double tr = v->re * wr - v->im * wi, ti = v->re * wi + v->im * wr;
            v->re = u->re - tr;
            v->im = u->im - ti;
            u->re += tr;
            u->im += ti;
        }
        return 1;
    }
    if (r == 4)
    {   /* t_q = w_n^(q k) Y_q[k]; X[k + s m] = sum_q t_q (sign i)^(q s) */
        for (k = 0; k < m; k++)
        {
            long double ar, ai, br, bi, cr, ci, dr, di;
            t[0] = y[k];
            for (q = 1; q < 4; q++)
            {
                const long long e = q * k;
                const long double wr = tw[e * st].re, wi = sign * tw[e * st].im;
                const vfft_cld_t yq = y[q * m + k];
                t[q].re = yq.re * wr - yq.im * wi;
                t[q].im = yq.re * wi + yq.im * wr;
            }
            ar = t[0].re + t[2].re; ai = t[0].im + t[2].im;
            br = t[0].re - t[2].re; bi = t[0].im - t[2].im;
            cr = t[1].re + t[3].re; ci = t[1].im + t[3].im;
            dr = t[1].re - t[3].re; di = t[1].im - t[3].im;   /* (sign i) d = (-sign di, sign dr) */
            y[k].re = ar + cr;             y[k].im = ai + ci;
            y[k + 2 * m].re = ar - cr;     y[k + 2 * m].im = ai - ci;
            y[k + m].re = br - sign * di;  y[k + m].im = bi + sign * dr;
            y[k + 3 * m].re = br + sign * di; y[k + 3 * m].im = bi - sign * dr;
        }
        return 1;
    }
    for (k = 0; k < m; k++)
    {   /* X[k + s m] = sum_q (w_n^(q k) Y_q[k]) w_r^(q s) */
        for (q = 0; q < r; q++)
        {
            const long long e = q * k;
            const long double wr = tw[e * st].re, wi = sign * tw[e * st].im;
            const vfft_cld_t yq = y[q * m + k];
            t[q].re = yq.re * wr - yq.im * wi;
            t[q].im = yq.re * wi + yq.im * wr;
        }
        for (s = 0; s < r; s++)
        {
            long double sr = 0, si = 0;
            for (q = 0; q < r; q++)
            {
                const long long e = ((q * s) % r) * m;
                const long double wr = tw[e * st].re, wi = sign * tw[e * st].im;
                sr += t[q].re * wr - t[q].im * wi;
                si += t[q].re * wi + t[q].im * wr;
            }
            y[k + s * m].re = sr;
            y[k + s * m].im = si;
        }
    }
    return 1;
}

/* the levels up to this many points read a contiguous copy of their twiddles (at most twice this many
 * entries in all); the larger levels stride the top table */
#define VFFT_DFT_LD_LVL_MAX 65536

/* X[k] = sum_j x[j] e^(sign 2 pi i j k / n), in long double; 0 when memory runs out */
static inline int vfft_dft_ld(long long n, const vfft_cld_t *x, vfft_cld_t *X, int sign)
{
    _vfft_ld_lvl_t L[64];
    long long sz[64], c = n, tot = 0, off = 0, j;
    int nl = 0, i, ok;
    vfft_cld_t *tw = (vfft_cld_t *)malloc((size_t)n * sizeof(vfft_cld_t)), *sub = 0;
    if (!tw)
        return 0;
    vfft_cs2pi_ld_table(n, n, tw);
    for (;;)
    {   /* the levels' lengths, as _vfft_ld_rec divides them */
        sz[nl++] = c;
        if (c == 1 || _vfft_ld_radix(c) > 64)
            break;
        c /= _vfft_ld_radix(c);
    }
    for (i = 1; i < nl; i++)
        if (sz[i] <= VFFT_DFT_LD_LVL_MAX)
            tot += sz[i];
    if (tot)
        sub = (vfft_cld_t *)malloc((size_t)tot * sizeof(vfft_cld_t));
    for (i = 0; i < nl; i++)
        if (i > 0 && sub && sz[i] <= VFFT_DFT_LD_LVL_MAX)
        {
            for (j = 0; j < sz[i]; j++)
                sub[off + j] = tw[j * (n / sz[i])];
            L[i].w = sub + off;
            L[i].st = 1;
            off += sz[i];
        }
        else
        {
            L[i].w = tw;
            L[i].st = n / sz[i];
        }
    ok = _vfft_ld_rec(n, x, 1, X, sign, L);
    free(sub);
    free(tw);
    return ok;
}

/* ── THE RADER KERNEL IN EXTENDED PRECISION ────────────────────────────────
 * A Rader engine bakes the forward DFT of length M of w[m] = e^(sign 2 pi i perm[m] / P), times 1/M, in
 * the order its own inner transform writes. The order is read off the transform itself: the labelled
 * input x[n] = 1/(e^(2 pi i n / M) - 1), x[0] = (M - 1)/2, has the DFT k, so the transform's output at
 * position c names the frequency pos[c] it holds (vfft_rader_positions rounds and checks it is a
 * permutation). The kernel is then vfft_dft_ld of the sequence, rounded once, placed by pos. */
static inline void vfft_rader_label(int M, double *x)
{
    int n;
    x[0] = 0.5 * (double)(M - 1);
    x[1] = 0.0;
    for (n = 1; n < M; n++)
    {   /* 1 / (e^(2 pi i n / M) - 1) = -1/2 - (i/2) cot(pi n / M) */
        double c, s;
        vfft_cs2pi_exact(n, 2LL * M, &c, &s);
        x[2 * n] = -0.5;
        x[2 * n + 1] = -0.5 * c / s;
    }
}
static inline int vfft_rader_positions(int M, const double *y, int *pos)
{
    char *seen = (char *)calloc((size_t)M, 1);
    int c, ok = seen != NULL;
    for (c = 0; ok && c < M; c++)
    {
        const long k = lround(y[2 * c]);
        if (k < 0 || k >= M || seen[k])
            ok = 0;
        else
        {
            seen[k] = 1;
            pos[c] = (int)k;
        }
    }
    free(seen);
    return ok;
}
/* 1 when the kernel was written; 0 (om untouched) when memory runs out */
static inline int vfft_rader_kernel_ld(int M, const int *perm, long long P, int sign, const int *pos, double *om)
{
    vfft_cld_t *x = (vfft_cld_t *)malloc(2 * (size_t)M * sizeof(vfft_cld_t)), *X = x ? x + M : NULL;
    const long double inv = 1.0L / (long double)M;
    int m, c;
    if (!x)
        return 0;
    for (m = 0; m < M; m++)
    {
        vfft_cs2pi_ld(perm[m], P, &x[m].re, &x[m].im);
        x[m].im *= sign;
    }
    if (!vfft_dft_ld(M, x, X, -1))
    {
        free(x);
        return 0;
    }
    for (c = 0; c < M; c++)
    {
        om[2 * c] = (double)(X[pos[c]].re * inv);
        om[2 * c + 1] = (double)(X[pos[c]].im * inv);
    }
    free(x);
    return 1;
}

/* both of an engine's kernels from ONE long-double DFT: the forward sequence is the conjugate of the backward
 * one reversed (g^(-m) = g^(M - m)), so the forward kernel is the conjugate of the backward kernel. gpow =
 * g^m mod P; omf / omb as the engine bakes them (forward perm g^(-m) sign -1, backward perm g^m sign +1),
 * times 1/M, placed by pos. 1 when written; 0 (untouched) when memory runs out. */
static inline int vfft_rader_kernels_ld(int M, const int *gpow, long long P, const int *pos, double *omf, double *omb)
{
    vfft_cld_t *x = (vfft_cld_t *)malloc(2 * (size_t)M * sizeof(vfft_cld_t)), *X = x ? x + M : NULL;
    const long double inv = 1.0L / (long double)M;
    int m, c;
    if (!x)
        return 0;
    for (m = 0; m < M; m++)
        vfft_cs2pi_ld(gpow[m], P, &x[m].re, &x[m].im);   /* e^(+2 pi i g^m / P) */
    if (!vfft_dft_ld(M, x, X, -1))
    {
        free(x);
        return 0;
    }
    for (c = 0; c < M; c++)
    {
        const long double re = X[pos[c]].re * inv, im = X[pos[c]].im * inv;
        omb[2 * c] = (double)re;
        omb[2 * c + 1] = (double)im;
        omf[2 * c] = (double)re;
        omf[2 * c + 1] = (double)-im;
    }
    free(x);
    return 1;
}

#endif /* VFFT_DFT_LD_H */
