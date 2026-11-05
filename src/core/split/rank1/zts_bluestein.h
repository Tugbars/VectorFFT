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
 * THE FORMS: lanes = 1 (K = 1, and the contiguous batch as K independent
 * transforms) on the S cell at M; lanes = K the lane-major batch, TURNED
 * (2026-11-06): the modulate pass reads the lane-major planes and writes
 * each lane as a ROW of M (the turn in, free: the pass exists anyway), the
 * convolution runs per row on the S cell at M, in cache, and the demodulate
 * pass reads the rows and writes the lane-major output (the turn out), in
 * blocks of VFFT_ZTS_BLU_ROWS lanes on a row scratch of that height. The
 * lane-major V cell at M with K lanes (the chirp broadcast over the lanes)
 * is the other lane-major form, kept for the measurement: its inner streams
 * the whole M x K plane per stage, 2.4x the row form's cost past L2
 * (docs/research/split_odd_band_2026_11_06.md).
 * M = the next power of two at or above max(16, 2N-1): the one cell the
 * registry has at that size, no race of M (the inner's chain, tile and block
 * are the M cell's own, replayed or raced and banked by the 1D create).
 *
 * No wisdom row of its own: the Bluestein cell is one method, its verdicts
 * are the M cell's. THREADED (zts_blu_mt.h, 2026-11-06): the three forms of
 * zturn-ts around the inner M cell — the passes cut around the inner's
 * sectioned walk (one transform, the lane-major single group), slabs of
 * whole transforms (the contiguous batch), lane blocks (the lane-major
 * batch) — each a verdict raced at create and banked eng=tcb tcmt= on the
 * cell's threading row at the plan's thread count. */
#ifndef VFFT_ZTS_BLUESTEIN_H
#define VFFT_ZTS_BLUESTEIN_H

#include <stdlib.h>
#include <string.h>
#include "split/rank1/zts.h"
#include "common/math/tw_exact.h"
#include "common/support/zalloc.h"

#define VFFT_ZTS_BLU_ROWS 8              /* the turned form's block: lanes per row scratch (a cache line of lanes) */
#define VFFT_ZTS_BLU_TURN_BLOCK 128      /* the turns' element block: 128 input lines (8 KB) + 8 rows x 128 x 2 planes (16 KB) in L1 */
typedef struct vfft_zts_blu_s
{
    int N, M;
    int lanes;                       /* 1, or the lane-major K */
    int turned;                      /* lanes > 1: 1 = the turned form on the S cell (rows), 0 = the V cell */
    vfft_zts_plan_t *inner;          /* the M cell: S (lanes 1, or the turned form) or V (lanes K), in place */
    double *c_re, *c_im;             /* the forward chirp c[n], n < N (the backward is its conjugate) */
    double *kf_re, *kf_im;           /* the forward kernel spectrum, M entries */
    double *kb_re, *kb_im;           /* the backward kernel spectrum */
    double *za_re, *za_im;           /* the work planes: M (lanes 1), ROWS*M (turned) or M*lanes (V) doubles each */
    /* THE THREADED FORMS (zts_blu_mt.h): mt = the verdict (0 serial; 1 the
     * sectioned BLOCKS, the slabs or the lane blocks; 2 the sectioned
     * TILES), mt_sect = 1 when the form is the inner's sectioned walk with
     * the passes cut (K = 1, the lane-major single group), mt_n = the worker
     * planes built (the inner's mtplane; for the slab form mtza too: a
     * worker's own za planes, re then im, M doubles each) */
    int mt, mt_sect, mt_grp, mt_n;   /* mt_grp: the lane-range form at line granularity */
    size_t mt_unit;                  /* lanes per range unit (0 = a whole block) */
    double *mtza[VFFT_ZTS_MT_MAX];
} vfft_zts_blu_t;

/* the verdict onto the plan and its inner (the inner's sectioned walk reads
 * its own mt) */
static inline void _zts_blu_set_mt(vfft_zts_blu_t *b, int mt, int sect, int grp, size_t unit)
{
    b->mt = mt;
    b->mt_sect = sect;
    b->mt_grp = grp;
    b->mt_unit = unit;
    if (b->inner)
    {
        b->inner->mt = sect ? mt : 0;
        b->inner->mt_sect = sect;
        b->inner->mt_grp = grp;
        b->inner->mt_unit = unit;
    }
}

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
    int i;
    if (!b) return;
    for (i = 0; i < b->mt_n; i++) vfft_aligned_free(b->mtza[i]);
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
    const int turned = lanes > 1 && inner && inner->lanes == 1;   /* the S cell under a lane-major batch: the turned form */
    vfft_zts_blu_t *b;
    const size_t wn = (size_t)M * (size_t)(lanes > 1 ? (turned ? VFFT_ZTS_BLU_ROWS : lanes) : 1);
    int n;
    if (!M || !inner || inner->N != M || (inner->lanes != lanes && !turned) || !ks || ks->N != M || ks->lanes != 1)
        return NULL;
    b = (vfft_zts_blu_t *)calloc(1, sizeof *b);
    if (!b) return NULL;
    b->N = N; b->M = M; b->lanes = lanes; b->inner = inner; b->turned = turned;
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

/* ── THE PASSES, over ranges: the serial run is the whole range, the
 * threaded forms (zts_blu_mt.h) cut them. Every element takes one
 * arithmetic whatever the range: the forms are bitwise. ── */
/* lanes = 1: the modulation by c (bwd: by c*) with the zero pad, over the
 * index range [lo, hi) of [0, M) */
static inline void _zts_blu_mod_s(const vfft_zts_blu_t *b, int bwd, const double *ire, const double *iim,
                                  double *zr, double *zi, size_t lo, size_t hi)
{
    const size_t N = (size_t)b->N;
    const double sgn = bwd ? -1.0 : 1.0;
    size_t n;
    for (n = lo; n < hi && n < N; n++)
    {
        const double cr = b->c_re[n], ci = sgn * b->c_im[n];
        const double xr = ire[n], xi = iim[n];
        zr[n] = xr * cr - xi * ci;
        zi[n] = xr * ci + xi * cr;
    }
    for (n = lo < N ? N : lo; n < hi; n++)
    {
        zr[n] = 0.0;
        zi[n] = 0.0;
    }
}
/* the pointwise multiply by the kernel spectrum over [lo, hi) of [0, M) */
static inline void _zts_blu_mul_s(const vfft_zts_blu_t *b, int bwd, double *zr, double *zi, size_t lo, size_t hi)
{
    const double *kr = bwd ? b->kb_re : b->kf_re, *ki = bwd ? b->kb_im : b->kf_im;
    size_t n;
    for (n = lo; n < hi; n++)
    {
        const double ar = zr[n], ai = zi[n];
        zr[n] = ar * kr[n] - ai * ki[n];
        zi[n] = ar * ki[n] + ai * kr[n];
    }
}
/* the demodulation by c (bwd: by c*) over [lo, hi) of [0, N); the inner's
 * backward is unnormalised, the kernel carries the 1/M */
static inline void _zts_blu_dem_s(const vfft_zts_blu_t *b, int bwd, const double *zr, const double *zi,
                                  double *ore, double *oim, size_t lo, size_t hi)
{
    const double sgn = bwd ? -1.0 : 1.0;
    size_t n;
    for (n = lo; n < hi; n++)
    {
        const double cr = b->c_re[n], ci = sgn * b->c_im[n];
        const double vr = zr[n], vi = zi[n];
        ore[n] = vr * cr - vi * ci;
        oim[n] = vr * ci + vi * cr;
    }
}
/* lanes = K: the same three over the elements [n0, n1) and the lanes
 * [g0, g1) (the chirp and the kernel broadcast over the lanes; plain C over
 * the lane vector, the compiler vectorises the inner loop) */
static inline void _zts_blu_mod_v(const vfft_zts_blu_t *b, int bwd, const double *ire, const double *iim,
                                  double *zr, double *zi, size_t n0, size_t n1, size_t g0, size_t g1)
{
    const size_t N = (size_t)b->N, K = (size_t)b->lanes;
    const double sgn = bwd ? -1.0 : 1.0;
    size_t n, t;
    for (n = n0; n < n1; n++)
    {
        double *wr = zr + n * K, *wi = zi + n * K;
        if (n < N)
        {
            const double cr = b->c_re[n], ci = sgn * b->c_im[n];
            const double *xr = ire + n * K, *xi = iim + n * K;
            for (t = g0; t < g1; t++)
            {
                wr[t] = xr[t] * cr - xi[t] * ci;
                wi[t] = xr[t] * ci + xi[t] * cr;
            }
        }
        else
            for (t = g0; t < g1; t++)
            {
                wr[t] = 0.0;
                wi[t] = 0.0;
            }
    }
}
static inline void _zts_blu_mul_v(const vfft_zts_blu_t *b, int bwd, double *zr, double *zi,
                                  size_t n0, size_t n1, size_t g0, size_t g1)
{
    const size_t K = (size_t)b->lanes;
    const double *kr = bwd ? b->kb_re : b->kf_re, *ki = bwd ? b->kb_im : b->kf_im;
    size_t n, t;
    for (n = n0; n < n1; n++)
    {
        const double gr = kr[n], gi = ki[n];
        double *wr = zr + n * K, *wi = zi + n * K;
        for (t = g0; t < g1; t++)
        {
            const double ar = wr[t], ai = wi[t];
            wr[t] = ar * gr - ai * gi;
            wi[t] = ar * gi + ai * gr;
        }
    }
}
static inline void _zts_blu_dem_v(const vfft_zts_blu_t *b, int bwd, const double *zr, const double *zi,
                                  double *ore, double *oim, size_t n0, size_t n1, size_t g0, size_t g1)
{
    const size_t K = (size_t)b->lanes;
    const double sgn = bwd ? -1.0 : 1.0;
    size_t n, t;
    for (n = n0; n < n1; n++)
    {
        const double cr = b->c_re[n], ci = sgn * b->c_im[n];
        const double *wr = zr + n * K, *wi = zi + n * K;
        double *yr = ore + n * K, *yi = oim + n * K;
        for (t = g0; t < g1; t++)
        {
            yr[t] = wr[t] * cr - wi[t] * ci;
            yi[t] = wr[t] * ci + wi[t] * cr;
        }
    }
}

/* ── the execute, lanes = 1: one transform on the given za planes; the inner
 * on its own plane (plane_alloc NULL) or on a slot's (the slab form) ── */
static inline void _zts_blu_one_on(const vfft_zts_blu_t *b, int bwd,
                                   const double *ire, const double *iim, double *ore, double *oim,
                                   double *zr, double *zi, const double *plane_alloc)
{
    const size_t N = (size_t)b->N, M = (size_t)b->M;
    _zts_blu_mod_s(b, bwd, ire, iim, zr, zi, 0, M);
    if (plane_alloc) vfft_zts_run_slab(b->inner, 0, zr, zi, zr, zi, 0, 1, plane_alloc);
    else             vfft_zts_execute_fwd(b->inner, zr, zi, zr, zi, 1);
    _zts_blu_mul_s(b, bwd, zr, zi, 0, M);
    if (plane_alloc) vfft_zts_run_slab(b->inner, 1, zr, zi, zr, zi, 0, 1, plane_alloc);
    else             vfft_zts_execute_bwd(b->inner, zr, zi, zr, zi, 1);
    _zts_blu_dem_s(b, bwd, zr, zi, ore, oim, 0, N);
}
static inline void _zts_blu_one(const vfft_zts_blu_t *b, int bwd,
                                const double *ire, const double *iim, double *ore, double *oim)
{
    _zts_blu_one_on(b, bwd, ire, iim, ore, oim, b->za_re, b->za_im, NULL);
}

/* ── the execute, lanes = K: the lane-major batch, all lanes ── */
static inline void _zts_blu_lanes(const vfft_zts_blu_t *b, int bwd,
                                  const double *ire, const double *iim, double *ore, double *oim)
{
    const size_t N = (size_t)b->N, M = (size_t)b->M, K = (size_t)b->lanes;
    double *zr = b->za_re, *zi = b->za_im;
    _zts_blu_mod_v(b, bwd, ire, iim, zr, zi, 0, M, 0, K);
    vfft_zts_execute_fwd(b->inner, zr, zi, zr, zi, 1);
    _zts_blu_mul_v(b, bwd, zr, zi, 0, M, 0, K);
    vfft_zts_execute_bwd(b->inner, zr, zi, zr, zi, 1);
    _zts_blu_dem_v(b, bwd, zr, zi, ore, oim, 0, N, 0, K);
}

/* ── the execute, lanes = K TURNED: the lanes [g0, g1) in blocks of
 * VFFT_ZTS_BLU_ROWS on the row scratch (zr, zi: ROWS x M each), the
 * convolution per row on the S cell (its own plane, or a slot's when
 * plane_alloc is given). The turn in reads one line of lanes per element
 * and writes the block's rows; the turn out reads the rows and writes one
 * line of lanes per element. In place when out == in: a block's lanes are
 * read whole before its output is written. ── */
static inline void _zts_blu_turned(const vfft_zts_blu_t *b, int bwd,
                                   const double *ire, const double *iim, double *ore, double *oim,
                                   size_t g0, size_t g1, double *zr, double *zi, const double *plane_alloc)
{
    const size_t N = (size_t)b->N, M = (size_t)b->M, K = (size_t)b->lanes, RB = VFFT_ZTS_BLU_ROWS;
    const double sgn = bwd ? -1.0 : 1.0;
    const double *kr = bwd ? b->kb_re : b->kf_re, *ki = bwd ? b->kb_im : b->kf_im;
    size_t t0, n, j;
    const size_t NB = VFFT_ZTS_BLU_TURN_BLOCK;
    double sr[VFFT_ZTS_BLU_ROWS][VFFT_ZTS_BLU_TURN_BLOCK], si[VFFT_ZTS_BLU_ROWS][VFFT_ZTS_BLU_TURN_BLOCK];   /* the staging tile, 16 KB */
    size_t n0;
    for (t0 = g0; t0 < g1; t0 += RB)
    {
        const size_t nr = (g1 - t0 < RB) ? g1 - t0 : RB;
        for (n0 = 0; n0 < N; n0 += NB)
        {   /* the turn in, modulated by c (bwd: by c*), as a tiled transpose: each element's
             * line of lanes read once into a staging tile of nr rows (in L1), then each staged
             * row run copied contiguously into its far row */
            const size_t n1 = n0 + NB < N ? n0 + NB : N, w = n1 - n0;
            for (n = n0; n < n1; n++)
            {
                const double cr = b->c_re[n], ci = sgn * b->c_im[n];
                const double *xr = ire + n * K + t0, *xi = iim + n * K + t0;
                for (j = 0; j < nr; j++)
                {
                    sr[j][n - n0] = xr[j] * cr - xi[j] * ci;
                    si[j][n - n0] = xr[j] * ci + xi[j] * cr;
                }
            }
            for (j = 0; j < nr; j++)
            {
                memcpy(zr + j * M + n0, sr[j], w * sizeof(double));
                memcpy(zi + j * M + n0, si[j], w * sizeof(double));
            }
        }
        for (j = 0; j < nr; j++)
        {   /* the zero pad, then the convolution on this row */
            double *r = zr + j * M, *i = zi + j * M;
            memset(r + N, 0, (M - N) * sizeof(double));
            memset(i + N, 0, (M - N) * sizeof(double));
            if (plane_alloc) vfft_zts_run_slab(b->inner, 0, r, i, r, i, 0, 1, plane_alloc);
            else             vfft_zts_execute_fwd(b->inner, r, i, r, i, 1);
            for (n = 0; n < M; n++)
            {
                const double ar = r[n], ai = i[n];
                r[n] = ar * kr[n] - ai * ki[n];
                i[n] = ar * ki[n] + ai * kr[n];
            }
            if (plane_alloc) vfft_zts_run_slab(b->inner, 1, r, i, r, i, 0, 1, plane_alloc);
            else             vfft_zts_execute_bwd(b->inner, r, i, r, i, 1);
        }
        for (n0 = 0; n0 < N; n0 += NB)
        {   /* the turn out, demodulated by c (bwd: by c*), the mirror: each row's run copied into
             * the staging tile, then each element's line of lanes written once; the kernel
             * carries the 1/M */
            const size_t n1 = n0 + NB < N ? n0 + NB : N, w = n1 - n0;
            for (j = 0; j < nr; j++)
            {
                memcpy(sr[j], zr + j * M + n0, w * sizeof(double));
                memcpy(si[j], zi + j * M + n0, w * sizeof(double));
            }
            for (n = n0; n < n1; n++)
            {
                const double cr = b->c_re[n], ci = sgn * b->c_im[n];
                double *yr = ore + n * K + t0, *yi = oim + n * K + t0;
                for (j = 0; j < nr; j++)
                {
                    const double vr = sr[j][n - n0], vi = si[j][n - n0];
                    yr[j] = vr * cr - vi * ci;
                    yi[j] = vr * ci + vi * cr;
                }
            }
        }
    }
}

/* K = the contiguous count (lanes = 1), ignored lane-major; in place when
 * out == in */
static inline void vfft_zts_blu_execute(const vfft_zts_blu_t *b, int bwd,
                                        const double *ire, const double *iim, double *ore, double *oim, size_t K)
{
    if (b->lanes > 1 && b->turned)
        _zts_blu_turned(b, bwd, ire, iim, ore, oim, 0, (size_t)b->lanes, b->za_re, b->za_im, NULL);
    else if (b->lanes > 1)
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
