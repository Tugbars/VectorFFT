/* dct1_dst1.h -- DCT-I and DST-I (VFFT_DCT1, VFFT_DST1): 1D, N reals in and N reals out, any N, one transform,
 * on the interleaved real FFT of the input extended to twice its length.
 *
 * THE DEFINITIONS (include/vfft.h; unnormalized, each its own inverse):
 *   DCT-I   Y[k] = x[0] + (-1)^k x[N-1] + 2 sum_{n=1..N-2} x[n] cos(pi n k / (N-1))     scale 2 (N - 1)
 *   DST-I   Y[k] = 2 sum_{n=0..N-1} x[n] sin(pi (n+1)(k+1) / (N+1))                     scale 2 (N + 1)
 *
 * THE METHOD: the transform is the real FFT of the input extended by its own mirror image, exactly.
 *   DCT-I   M = N - 1, L = 2M. The input extended even:
 *             e[0 .. M] = x[0 .. M],  e[M + j] = x[M - j], j = 1 .. M - 1
 *           R2C at L:  E[0 .. M], real;  Y[k] = Re E[k], k = 0 .. M
 *   DST-I   M = N + 1, L = 2M. The input extended odd:
 *             o[0] = 0,  o[j] = x[j - 1],  o[M] = 0,  o[M + j] = -x[N - j], j = 1 .. N
 *           R2C at L:  O[0 .. M], imaginary;  Y[k] = -Im O[k + 1], k = 0 .. N - 1
 * So a transform is one pass that extends, one R2C at L, one pass that takes the outputs from the half
 * spectrum. Each pass reads and writes whole vectors, 4 doubles at AVX2 and 8 at AVX-512, with a scalar tail.
 * The two shortest lengths have no FFT: DCT-I at N = 2 is (x0 + x1, x0 - x1), DST-I at N = 1 is 2 x0.
 *
 * THE PLAN holds the R2C at L, out of place (a plan of the real door), and two work buffers allocated at
 * create: E, L doubles, the extended input; W, L + 2 doubles, its half spectrum. Either placement takes
 * out == in. A plan of T threads threads inside its real FFT; the passes run on the calling thread. Forward
 * and backward are the same transform.
 *
 * THE WISDOM (owner, 2026-11-14): the cell has its own row in the real shard --
 *   t=dct1 n=N q=1 ord=nat place=oop [nthreads=T] | eng=dct and its real FFT under dct_
 *   t=dst1 n=N q=1 ord=nat place=oop [nthreads=T] | eng=dst and its real FFT under dst_
 * (vw2_r2r_key, il/wisdom/wisdom2_real_il.h). The real FFT is a CHILD IN ROLE (il/real/real_child.h): created
 * against a private store seeded from the cell's row, it replays the recipe there or races, and what it banked
 * goes back onto the cell's row. Its race runs in the transform's role (il/real/real_role.h): every candidate
 * FFT is timed as one whole DCT-I or DST-I -- the extending pass, that FFT, the taking pass -- and the fastest
 * whole transform wins. The plan is placement-blind, so one row serves both placements. Under
 * VFFT_FLAG_ESTIMATE the FFT is planned by formulas, nothing is measured and no store is read or written.
 *
 * POSITION IN vfft.c: after il/real/dst.h and before the front door's create. */
#ifndef VFFT_IL_REAL_DCT1_DST1_H
#define VFFT_IL_REAL_DCT1_DST1_H
#ifndef VFFT_FLOAT   /* FP64 only: the FP32 unit serves 1D C2C */

#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

#include "il/real/real_child.h"   /* the real FFT as a child in role of the cell's row */

#define VFFT_TYPE1_NMAX 0x3FFFFFF0   /* the real FFT's length, 2 (N + 1), stays an int */

typedef struct vfft_type1_s
{
    int N;                     /* the transform's length */
    int L;                     /* the real FFT's: 2 (N - 1) for DCT-I, 2 (N + 1) for DST-I */
    int sine;                  /* 1 = DST-I, 0 = DCT-I */
    struct vfft_plan_s *r2c;   /* the R2C at L, out of place (owned); NULL at the length served without one */
    struct vfft_wisdom_s *S;   /* its private store: owned, never saved; NULL under VFFT_FLAG_ESTIMATE */
    double *E;                 /* L doubles: the extended input */
    double *W;                 /* L + 2 doubles: its half spectrum */
} vfft_type1_t;

/* ── the extending passes ───────────────────────────────────────────────── */

/* DCT-I, M = N - 1: e[0 .. M] = x[0 .. M], e[M + j] = x[M - j], j = 1 .. M - 1. x and e apart. */
static inline void _dct1_extend(const double *x, double *e, int N)
{
    const int M = N - 1;
    int j = 1;
    memcpy(e, x, (size_t)N * sizeof(double));
#if defined(__AVX512F__)
    {
        const __m512i REV = _mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
        for (; j + 8 <= M; j += 8)
            _mm512_storeu_pd(e + M + j, _mm512_permutexvar_pd(REV, _mm512_loadu_pd(x + M - j - 7)));
    }
#endif
#if defined(__AVX2__)
    for (; j + 4 <= M; j += 4)
        _mm256_storeu_pd(e + M + j, _mm256_permute4x64_pd(_mm256_loadu_pd(x + M - j - 3), 0x1B));
#endif
    for (; j < M; j++)
        e[M + j] = x[M - j];
}

/* DST-I, M = N + 1: o[0] = 0, o[j] = x[j - 1], o[M] = 0, o[M + j] = -x[N - j], j = 1 .. N. x and o apart. */
static inline void _dst1_extend(const double *x, double *o, int N)
{
    const int M = N + 1;
    int j = 1;
    o[0] = 0.0;
    memcpy(o + 1, x, (size_t)N * sizeof(double));
    o[M] = 0.0;
#if defined(__AVX512F__)
    {
        const __m512i REV = _mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
        const __m512d NEG = _mm512_set1_pd(-0.0);
        for (; j + 7 <= N; j += 8)
            _mm512_storeu_pd(o + M + j, _mm512_xor_pd(_mm512_permutexvar_pd(REV, _mm512_loadu_pd(x + N - j - 7)), NEG));
    }
#endif
#if defined(__AVX2__)
    {
        const __m256d NEG = _mm256_set1_pd(-0.0);
        for (; j + 3 <= N; j += 4)
            _mm256_storeu_pd(o + M + j, _mm256_xor_pd(_mm256_permute4x64_pd(_mm256_loadu_pd(x + N - j - 3), 0x1B), NEG));
    }
#endif
    for (; j <= N; j++)
        o[M + j] = -x[N - j];
}

/* ── the taking passes ──────────────────────────────────────────────────── */

/* DCT-I: y[k] = Re V[k], k = 0 .. N - 1, from the half spectrum V (interleaved, N bins). V and y apart. */
static inline void _dct1_take(const double *V, double *y, int N)
{
    int k = 0;
#if defined(__AVX512F__)
    {
        const __m512i RE = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
        for (; k + 8 <= N; k += 8)
            _mm512_storeu_pd(y + k, _mm512_permutex2var_pd(_mm512_loadu_pd(V + 2 * k), RE, _mm512_loadu_pd(V + 2 * k + 8)));
    }
#endif
#if defined(__AVX2__)
    for (; k + 4 <= N; k += 4)
    {
        const __m256d a = _mm256_loadu_pd(V + 2 * k), b = _mm256_loadu_pd(V + 2 * k + 4);
        _mm256_storeu_pd(y + k, _mm256_permute4x64_pd(_mm256_unpacklo_pd(a, b), 0xD8));
    }
#endif
    for (; k < N; k++)
        y[k] = V[2 * k];
}

/* DST-I: y[k] = -Im V[k + 1], k = 0 .. N - 1, from the half spectrum V (interleaved, N + 2 bins). V and y apart. */
static inline void _dst1_take(const double *V, double *y, int N)
{
    const double *B = V + 2;   /* bin 1 */
    int k = 0;
#if defined(__AVX512F__)
    {
        const __m512i IM = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
        const __m512d NEG = _mm512_set1_pd(-0.0);
        for (; k + 8 <= N; k += 8)
            _mm512_storeu_pd(y + k, _mm512_xor_pd(_mm512_permutex2var_pd(_mm512_loadu_pd(B + 2 * k), IM,
                                                                           _mm512_loadu_pd(B + 2 * k + 8)), NEG));
    }
#endif
#if defined(__AVX2__)
    {
        const __m256d NEG = _mm256_set1_pd(-0.0);
        for (; k + 4 <= N; k += 4)
        {
            const __m256d a = _mm256_loadu_pd(B + 2 * k), b = _mm256_loadu_pd(B + 2 * k + 4);
            _mm256_storeu_pd(y + k, _mm256_xor_pd(_mm256_permute4x64_pd(_mm256_unpackhi_pd(a, b), 0xD8), NEG));
        }
    }
#endif
    for (; k < N; k++)
        y[k] = -B[2 * k + 1];
}

/* ── execute ────────────────────────────────────────────────────────────── */

/* DCT-I of x into y (y may be x) */
static inline void _dct1_run(const vfft_type1_t *d, const double *x, double *y)
{
    if (d->N == 2)
    {
        const double a = x[0], b = x[1];
        y[0] = a + b;
        y[1] = a - b;
        return;
    }
    _dct1_extend(x, d->E, d->N);
    _vfft_exec6((vfft_plan)d->r2c, VFFT_FORWARD, d->E, NULL, d->W, NULL);
    _dct1_take(d->W, y, d->N);
}
/* DST-I of x into y (y may be x) */
static inline void _dst1_run(const vfft_type1_t *d, const double *x, double *y)
{
    if (d->N == 1)
    {
        y[0] = 2.0 * x[0];
        return;
    }
    _dst1_extend(x, d->E, d->N);
    _vfft_exec6((vfft_plan)d->r2c, VFFT_FORWARD, d->E, NULL, d->W, NULL);
    _dst1_take(d->W, y, d->N);
}

/* the bound executors (vfft_plan_s.k1_exec, k1_go): never decline, either pointer pattern, either direction
 * (each transform is its own inverse) */
static int _k1x_dct1(struct vfft_plan_s *h, vfft_dir_t dir, const vfft_real *zin, vfft_real *zout)
{
    (void)dir;
    _dct1_run(h->type1, zin, zout);
    return 0;
}
static int _k1x_dst1(struct vfft_plan_s *h, vfft_dir_t dir, const vfft_real *zin, vfft_real *zout)
{
    (void)dir;
    _dst1_run(h->type1, zin, zout);
    return 0;
}

/* ── the role, create and destroy ───────────────────────────────────────── */

/* THE ROLE (il/real/real_role.h): one whole transform with the candidate FFT h inside, on the create's own
 * buffers x (the input) and y (the output) and the plan's work buffers. What every candidate of the real
 * door's races is timed as while the transform's FFT is created. */
typedef struct
{
    const vfft_type1_t *d;
    double *x, *y;
} _type1_role_t;
static void _dct1_role_run(void *ctx, struct vfft_plan_s *h)
{
    const _type1_role_t *r = (const _type1_role_t *)ctx;
    const vfft_type1_t *d = r->d;
    _dct1_extend(r->x, d->E, d->N);
    _real_il_exec_any(h, d->E, d->W);
    _dct1_take(d->W, r->y, d->N);
}
static void _dst1_role_run(void *ctx, struct vfft_plan_s *h)
{
    const _type1_role_t *r = (const _type1_role_t *)ctx;
    const vfft_type1_t *d = r->d;
    _dst1_extend(r->x, d->E, d->N);
    _real_il_exec_any(h, d->E, d->W);
    _dst1_take(d->W, r->y, d->N);
}

static void _vfft_type1_destroy(vfft_type1_t *d)
{
    if (!d)
        return;
    if (d->r2c)
        vfft_destroy((vfft_plan)d->r2c);
    vfft_child_store_free(d->S);
    vfft_aligned_free(d->E);
    vfft_aligned_free(d->W);
    free(d);
}

/* THE DCT-I / DST-I DOOR: a VFFT_DCT1 or VFFT_DST1 request, 1D, one transform. est = 1 (VFFT_FLAG_ESTIMATE):
 * the real FFT planned by formulas, nothing measured, no store (W unused); est = 0: a child in role of the
 * cell's row in W. NULL with the reason on the error record. */
static vfft_plan _vfft_create_type1_il(const vfft_cfg_t *cfg, struct vfft_wisdom_s *W, int est)
{
    const int N = cfg->n[0], sine = cfg->transform == VFFT_DST1, nmin = sine ? 1 : 2;
    const char *tn = _vfft_tname(cfg->transform);
    struct vfft_plan_s *h;
    vfft_type1_t *d;
    if (cfg->dims >= 2)
    {
        _vfft_warn("vfft_create: %dD %s is not served: DCT-I and DST-I plans are 1D", cfg->dims, tn);
        return NULL;
    }
    if (cfg->howmany > 1)
    {
        _vfft_warn("vfft_create: %s N=%d howmany=%zu is not served: a DCT-I or DST-I plan runs one transform "
                   "(create it with howmany = 1 and call it once per transform)", tn, N, cfg->howmany);
        return NULL;
    }
    if (N < nmin || N > VFFT_TYPE1_NMAX)
    {
        _vfft_warn("vfft_create: %s N=%d invalid (the length must be %d or more%s)", tn, N, nmin,
                   N > VFFT_TYPE1_NMAX ? " and at most 2^30 - 16" : "");
        return NULL;
    }
    d = (vfft_type1_t *)calloc(1, sizeof *d);
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
    d->L = sine ? 2 * (N + 1) : 2 * (N - 1);
    if (N > nmin)
    {
        double *x = NULL, *y = NULL;   /* the races' input and output (measured plans) */
        _type1_role_t role;
        _real_child_t rc;
        int changed = 0;
        d->E = (double *)vfft_aligned_alloc((size_t)d->L * sizeof(double));
        d->W = (double *)vfft_aligned_alloc(((size_t)d->L + 2) * sizeof(double));
        if (!est)
        {
            x = (double *)vfft_aligned_alloc((size_t)N * sizeof(double));
            y = (double *)vfft_aligned_alloc((size_t)N * sizeof(double));
        }
        if (!d->E || !d->W || (!est && (!x || !y)))
        {
            vfft_aligned_free(x);
            vfft_aligned_free(y);
            _vfft_type1_destroy(d);
            free(h);
            _vfft_warn("vfft_create: %s N=%d: out of memory (the work buffers)", tn, N);
            return NULL;
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
        rc.t = sine ? VW2_T_DST1 : VW2_T_DCT1;
        rc.N = N;
        rc.pre = sine ? "dst_" : "dct_";
        rc.eng = sine ? "dst" : "dct";
        rc.n = d->L;
        rc.is_c2r = 0;
        rc.run = sine ? _dst1_role_run : _dct1_role_run;
        rc.ctx = &role;
        d->r2c = _real_child_create(cfg, W, &rc, est, &d->S, &changed);
        vfft_aligned_free(x);
        vfft_aligned_free(y);
        if (changed)
            _vw2_persist(W, cfg);
        if (!d->r2c)
        {   /* the real door's own reason is on the record: it names the cell that could not be built */
            _vfft_type1_destroy(d);
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
    h->type1 = d;
    h->k1_exec = sine ? _k1x_dst1 : _k1x_dct1;
    h->k1_go[VFFT_FORWARD] = h->k1_exec;
    h->k1_go[VFFT_BACKWARD] = h->k1_exec;
    return (vfft_plan)h;
}

#endif /* !VFFT_FLOAT */
#endif /* VFFT_IL_REAL_DCT1_DST1_H */
