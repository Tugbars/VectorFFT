/* zts_bluestein.h — Bluestein over zturn-ts: the split library's engine for
 * every length its bands do not reach (a prime, 2·odd, a pure-odd K = 1,
 * anything up to 131072), built 2026-11-06 (the owner's odd plan, item 4).
 *
 * THE METHOD. X[k] = c*[k] · sum_n (x[n] c*[n]) · c[k-n], c[n] = exp(i pi n^2 / N):
 * a modulation by the chirp, a convolution of length M >= 2N-1 (a power of
 * two: zturn-ts's fused cell at M, forward then backward, in place on the
 * plan's own planes), a demodulation. The convolution kernel's spectrum
 * (FFT_M of the wrapped chirp, divided by M) is built once per direction at
 * create with the M cell's own forward. The chirp uses n^2 mod 2N in
 * integer arithmetic and once-rounded cos/sin (tw_exact.h): valid for any N.
 *
 * TWO FORMS, the engine's own: lanes = 1 (K = 1, and the contiguous batch
 * as K independent transforms) on the S cell at M; lanes = K (the lane-major
 * batch, a column axis) on the V cell at M with K lanes, the chirp and the
 * kernel broadcast over the lanes, the M cell's block (blk=) as banked.
 * M = the next power of two at or above max(16, 2N-1): the one cell the
 * registry has at that size, no race of M (the inner's chain, tile and block
 * are the M cell's own, replayed or raced and banked by the 1D create).
 *
 * No wisdom row of its own: the Bluestein cell is one method, its verdicts
 * are the M cell's. Not threaded (a contiguous batch runs serially; the slab
 * form would need per-worker work planes — not built). */
#ifndef VFFT_ZTS_BLUESTEIN_H
#define VFFT_ZTS_BLUESTEIN_H

#include <stdlib.h>
#include <string.h>
#include "split/rank1/zts.h"
#include "common/math/tw_exact.h"
#include "common/support/zalloc.h"

typedef struct vfft_zts_blu_s
{
    int N, M;
    int lanes;                       /* 1, or the lane-major K */
    vfft_zts_plan_t *inner;          /* the M cell: S (lanes 1) or V (lanes K), in place */
    double *c_re, *c_im;             /* the forward chirp c[n], n < N (the backward is its conjugate) */
    double *kf_re, *kf_im;           /* the forward kernel spectrum, M entries */
    double *kb_re, *kb_im;           /* the backward kernel spectrum */
    double *za_re, *za_im;           /* the work planes: M (lanes 1) or M*lanes doubles each */
} vfft_zts_blu_t;

/* the length law: any N >= 2 whose convolution cell exists (2N-1 <= the
 * ceiling, the next power of two a fused cell of the registry) */
static inline int vfft_zts_blu_m(int N)
{
    int M = 16;
    if (N < 2 || 2 * N - 1 > VFFT_ZTS_MAX_N) return 0;
    while (M < 2 * N - 1) M <<= 1;
    return vfft_zts_fused_at(M) ? M : 0;
}

static inline void vfft_zts_blu_destroy(vfft_zts_blu_t *b)
{
    if (!b) return;
    vfft_zts_destroy(b->inner);
    vfft_aligned_free(b->c_re); vfft_aligned_free(b->c_im);
    vfft_aligned_free(b->kf_re); vfft_aligned_free(b->kf_im);
    vfft_aligned_free(b->kb_re); vfft_aligned_free(b->kb_im);
    vfft_aligned_free(b->za_re); vfft_aligned_free(b->za_im);
    free(b);
}

/* the kernel spectra, with a K = 1 plan at M (ks: the S cell, built by the
 * caller and destroyed after): kern = FFT_M(b) / M, b[0] = c*[0],
 * b[n] = b[M-n] = c*[n] for 0 < n < N, 0 elsewhere; the backward kernel from
 * the conjugate chirp */
static inline int _zts_blu_kernels(vfft_zts_blu_t *b, const vfft_zts_plan_t *ks)
{
    const int N = b->N, M = b->M;
    double *wr = (double *)vfft_aligned_alloc((size_t)M * sizeof(double));
    double *wi = (double *)vfft_aligned_alloc((size_t)M * sizeof(double));
    int d, n, i;
    if (!wr || !wi) { vfft_aligned_free(wr); vfft_aligned_free(wi); return 0; }
    for (d = 0; d < 2; d++)
    {
        double *kr = d ? b->kb_re : b->kf_re, *ki = d ? b->kb_im : b->kf_im;
        const double sgn = d ? -1.0 : 1.0;   /* the backward chirp is the conjugate; b = conj(c) */
        const double inv = 1.0 / (double)M;
        memset(wr, 0, (size_t)M * sizeof(double));
        memset(wi, 0, (size_t)M * sizeof(double));
        wr[0] = b->c_re[0];
        wi[0] = -sgn * b->c_im[0];
        for (n = 1; n < N; n++)
        {
            const double br = b->c_re[n], bi = -sgn * b->c_im[n];
            wr[n] = br; wi[n] = bi;
            wr[M - n] = br; wi[M - n] = bi;
        }
        vfft_zts_execute_fwd(ks, wr, wi, wr, wi, 1);
        for (i = 0; i < M; i++) { kr[i] = wr[i] * inv; ki[i] = wi[i] * inv; }
    }
    vfft_aligned_free(wr);
    vfft_aligned_free(wi);
    return 1;
}

/* create: N, the form (lanes), the M cell (inner, owned from here) and a
 * K = 1 plan at M for the kernels (ks, the caller's) */
static inline vfft_zts_blu_t *vfft_zts_blu_create(int N, int lanes, vfft_zts_plan_t *inner,
                                                  const vfft_zts_plan_t *ks)
{
    const int M = vfft_zts_blu_m(N);
    vfft_zts_blu_t *b;
    const size_t wn = (size_t)M * (size_t)(lanes > 1 ? lanes : 1);
    int n;
    if (!M || !inner || inner->N != M || inner->lanes != lanes || !ks || ks->N != M || ks->lanes != 1)
        return NULL;
    b = (vfft_zts_blu_t *)calloc(1, sizeof *b);
    if (!b) return NULL;
    b->N = N; b->M = M; b->lanes = lanes; b->inner = inner;
    b->c_re = (double *)vfft_aligned_alloc((size_t)N * sizeof(double));
    b->c_im = (double *)vfft_aligned_alloc((size_t)N * sizeof(double));
    b->kf_re = (double *)vfft_aligned_alloc((size_t)M * sizeof(double));
    b->kf_im = (double *)vfft_aligned_alloc((size_t)M * sizeof(double));
    b->kb_re = (double *)vfft_aligned_alloc((size_t)M * sizeof(double));
    b->kb_im = (double *)vfft_aligned_alloc((size_t)M * sizeof(double));
    b->za_re = (double *)vfft_aligned_alloc(wn * sizeof(double));
    b->za_im = (double *)vfft_aligned_alloc(wn * sizeof(double));
    if (!b->c_re || !b->c_im || !b->kf_re || !b->kf_im || !b->kb_re || !b->kb_im || !b->za_re || !b->za_im)
    {
        b->inner = NULL; /* the caller's until the create returns it */
        vfft_zts_blu_destroy(b);
        return NULL;
    }
    for (n = 0; n < N; n++)
    {   /* c[n] = exp(-i pi n^2 / N): the angle -2 pi (n^2 mod 2N) / 2N */
        const long long m2 = ((long long)n * n) % (2LL * N);
        double c, s;
        vfft_cs2pi_exact(m2, 2LL * N, &c, &s);
        b->c_re[n] = c;
        b->c_im[n] = -s;
    }
    if (!_zts_blu_kernels(b, ks))
    {
        b->inner = NULL;
        vfft_zts_blu_destroy(b);
        return NULL;
    }
    return b;
}

/* ── the execute, lanes = 1: one transform (the contiguous batch loops it) ── */
static inline void _zts_blu_one(const vfft_zts_blu_t *b, int bwd,
                                const double *ire, const double *iim, double *ore, double *oim)
{
    const int N = b->N, M = b->M;
    const double sgn = bwd ? -1.0 : 1.0;
    const double *kr = bwd ? b->kb_re : b->kf_re, *ki = bwd ? b->kb_im : b->kf_im;
    double *zr = b->za_re, *zi = b->za_im;
    int n;
    for (n = 0; n < N; n++)
    {   /* modulate by c (bwd: by c*) */
        const double cr = b->c_re[n], ci = sgn * b->c_im[n];
        const double xr = ire[n], xi = iim[n];
        zr[n] = xr * cr - xi * ci;
        zi[n] = xr * ci + xi * cr;
    }
    memset(zr + N, 0, (size_t)(M - N) * sizeof(double));
    memset(zi + N, 0, (size_t)(M - N) * sizeof(double));
    vfft_zts_execute_fwd(b->inner, zr, zi, zr, zi, 1);
    for (n = 0; n < M; n++)
    {
        const double ar = zr[n], ai = zi[n];
        zr[n] = ar * kr[n] - ai * ki[n];
        zi[n] = ar * ki[n] + ai * kr[n];
    }
    vfft_zts_execute_bwd(b->inner, zr, zi, zr, zi, 1);
    for (n = 0; n < N; n++)
    {   /* demodulate by c (bwd: by c*); the inner's backward is unnormalised,
         * the kernel carries the 1/M */
        const double cr = b->c_re[n], ci = sgn * b->c_im[n];
        const double vr = zr[n], vi = zi[n];
        ore[n] = vr * cr - vi * ci;
        oim[n] = vr * ci + vi * cr;
    }
}

/* ── the execute, lanes = K: the lane-major batch, the chirp and the kernel
 * broadcast over the lanes of each element (plain C over the lane vector:
 * the compiler vectorises the inner loop) ── */
static inline void _zts_blu_lanes(const vfft_zts_blu_t *b, int bwd,
                                  const double *ire, const double *iim, double *ore, double *oim)
{
    const int N = b->N, M = b->M;
    const size_t K = (size_t)b->lanes;
    const double sgn = bwd ? -1.0 : 1.0;
    const double *kr = bwd ? b->kb_re : b->kf_re, *ki = bwd ? b->kb_im : b->kf_im;
    double *zr = b->za_re, *zi = b->za_im;
    int n;
    size_t t;
    for (n = 0; n < N; n++)
    {
        const double cr = b->c_re[n], ci = sgn * b->c_im[n];
        const double *xr = ire + (size_t)n * K, *xi = iim + (size_t)n * K;
        double *wr = zr + (size_t)n * K, *wi = zi + (size_t)n * K;
        for (t = 0; t < K; t++)
        {
            wr[t] = xr[t] * cr - xi[t] * ci;
            wi[t] = xr[t] * ci + xi[t] * cr;
        }
    }
    memset(zr + (size_t)N * K, 0, (size_t)(M - N) * K * sizeof(double));
    memset(zi + (size_t)N * K, 0, (size_t)(M - N) * K * sizeof(double));
    vfft_zts_execute_fwd(b->inner, zr, zi, zr, zi, 1);
    for (n = 0; n < M; n++)
    {
        const double gr = kr[n], gi = ki[n];
        double *wr = zr + (size_t)n * K, *wi = zi + (size_t)n * K;
        for (t = 0; t < K; t++)
        {
            const double ar = wr[t], ai = wi[t];
            wr[t] = ar * gr - ai * gi;
            wi[t] = ar * gi + ai * gr;
        }
    }
    vfft_zts_execute_bwd(b->inner, zr, zi, zr, zi, 1);
    for (n = 0; n < N; n++)
    {
        const double cr = b->c_re[n], ci = sgn * b->c_im[n];
        const double *wr = zr + (size_t)n * K, *wi = zi + (size_t)n * K;
        double *yr = ore + (size_t)n * K, *yi = oim + (size_t)n * K;
        for (t = 0; t < K; t++)
        {
            yr[t] = wr[t] * cr - wi[t] * ci;
            yi[t] = wr[t] * ci + wi[t] * cr;
        }
    }
}

/* K = the contiguous count (lanes = 1), ignored lane-major; in place when
 * out == in */
static inline void vfft_zts_blu_execute(const vfft_zts_blu_t *b, int bwd,
                                        const double *ire, const double *iim, double *ore, double *oim, size_t K)
{
    if (b->lanes > 1)
        _zts_blu_lanes(b, bwd, ire, iim, ore, oim);
    else
    {
        const size_t N = (size_t)b->N;
        size_t t;
        for (t = 0; t < K; t++)
            _zts_blu_one(b, bwd, ire + t * N, iim + t * N, ore + t * N, oim + t * N);
    }
}

#endif /* VFFT_ZTS_BLUESTEIN_H */
