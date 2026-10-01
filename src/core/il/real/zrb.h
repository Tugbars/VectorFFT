/* zrb.h - the real BLUESTEIN: the odd-N real transform that has no chain (a
 * prime, or a composite with a factor outside the flat DIT's pool) as a
 * chirp-z convolution at length M on the IL machinery -- the half spectrum
 * from the convolution's own structure, never from a complex transform of the
 * widened input.
 *
 * THE FORWARD (r2c). With c[n] = e^{-i pi n^2/N}, w^{nk} = c[n] c[k] conj(c[k-n]):
 *   X[k] = c[k] * sum_n (x[n] c[n]) * conj(c[k-n]),   k = 0..h,  h = (N-1)/2.
 * The chirped input a[n] = x[n] c[n] is a REAL load edge (two multiplies per
 * sample), the convolution runs at length M through a matched fwd/bwd inner
 * pair (the kernel FFT'd once at create through the same forward, so the
 * pointwise multiply happens in that forward's output order and the backward
 * consumes it back), and the output chirp is applied to bins 0..h only.
 * Only those bins are wanted, so the wrapped convolution has to keep only
 * j = k - n in [-(N-1), h] free of aliasing:
 *   M >= (3N-1)/2, not 2N-1.
 * The length is a plan input, any M the inner pool builds (a power of two, a
 * 2^a*odd ZTURN-T length, an il2p pair): for N = 1031 M = 1920 against the
 * complex Bluestein's 4096, for 749 M = 1152 against 2048.
 *
 * THE BACKWARD (c2r). A Hermitian spectrum makes the inverse a chirp-z of the
 * HALF spectrum: with g[0] = X[0], g[k] = 2 X[k] (k = 1..h),
 *   x[n] = Re( conj(c[n]) * sum_{k<=h} (g[k] conj(c[k])) * c[n-k] ),  n = 0..N-1,
 * (N+1)/2 inputs to N outputs: j = n - k in [-h, N-1], the same bound. The
 * output is a real store edge (the real part of one complex product).
 * Unnormalized: c2r(r2c(x)) = N x. Both placements are one pipeline (every
 * read of the caller's array precedes the first write).
 *
 * The plan inputs are M and the inner (its kind, shape and tile, the prime
 * route's own descriptors); the odd real race sweeps them at create and banks
 * eng=zrb (bridge/real_bridge.h, wisdom2_real_il.h). K rows run as a loop
 * over the one-row pipeline (vfft_zrb_execute_*_rows).
 */
#ifndef VFFT_ZRB_H
#define VFFT_ZRB_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#include "il_prime.h" /* the inner pair at M (_ilprime_inner_t, its provider), the exact chirp trig, the packed multiply */

typedef struct vfft_zrb_s
{
    int N, h;                   /* h = (N-1)/2: the bins beside the DC */
    int M;                      /* the convolution length, >= (3N-1)/2 */
    _ilprime_inner_t inner;     /* the matched fwd/bwd pair at M */
    double *c;                  /* the chirp c[n], n < N (2N doubles) */
    double *kf, *kb;            /* the kernels in the inner's output order, 1/M baked (2M each) */
    double *za, *zb;            /* two packed planes of M */
    char ikind[8], ishape[64];  /* the inner's name, as the prime route spells it (banked, fingerprinted) */
    int itw;                    /* the inner's tile, 0 = untiled */
} vfft_zrb_plan_t;

/* the smallest legal M at odd N: (3N-1)/2 */
static inline int vfft_zrb_min_m(int N) { return N + (N - 1) / 2; }

static inline void vfft_zrb_destroy(vfft_zrb_plan_t *p)
{
    if (!p) return;
    _ilprime_inner_free(&p->inner);
    vfft_aligned_free(p->c);
    vfft_aligned_free(p->kf); vfft_aligned_free(p->kb);
    vfft_aligned_free(p->za); vfft_aligned_free(p->zb);
    free(p);
}

/* "M/kind:shape[/tW]" */
static inline void vfft_zrb_str(const vfft_zrb_plan_t *p, char *b, size_t sz)
{
    if (p->itw > 0) snprintf(b, sz, "%d/%s:%s/t%d", p->M, p->ikind, p->ishape, p->itw);
    else            snprintf(b, sz, "%d/%s:%s", p->M, p->ikind, p->ishape);
}

/* The plan at (N, M); the inner from the provider (exactly one descriptor,
 * il/rank1/k1_commit.h's _ilprime_inner_from_desc) or, with none, from
 * il_prime's structural rule. The caller names the inner afterwards
 * (vfft_zrb_name_inner). NULL when M is below the bound or the inner does
 * not build. */
static inline vfft_zrb_plan_t *vfft_zrb_create(int N, int M, _ilprime_inner_provider_fn prov, void *ctx)
{
    vfft_zrb_plan_t *p;
    int ok, n, j;
    if (N < 3 || !(N & 1) || M < vfft_zrb_min_m(N)) return NULL;
    p = (vfft_zrb_plan_t *)calloc(1, sizeof *p);
    if (!p) return NULL;
    p->N = N; p->h = (N - 1) / 2; p->M = M;
    snprintf(p->ikind, sizeof p->ikind, "?");
    _ilprime_inner_provider = prov;
    _ilprime_inner_provider_ctx = ctx;
    ok = _ilprime_inner_make(M, &p->inner);
    _ilprime_inner_provider = 0;
    _ilprime_inner_provider_ctx = 0;
    if (!ok) { vfft_zrb_destroy(p); return NULL; }
    p->c  = _ilprime_alloc((size_t)2 * N);
    p->kf = _ilprime_alloc((size_t)2 * M);
    p->kb = _ilprime_alloc((size_t)2 * M);
    p->za = _ilprime_alloc((size_t)2 * M);
    p->zb = _ilprime_alloc((size_t)2 * M);
    if (!p->c || !p->kf || !p->kb || !p->za || !p->zb) { vfft_zrb_destroy(p); return NULL; }
    for (n = 0; n < N; n++)
    {   /* c[n] = e^{-i pi n^2/N}: the angle's numerator reduced mod 2N first */
        const long long m2 = ((long long)n * n) % (2LL * N);
        double co, si;
        vfft_cs2pi_exact(m2, 2LL * N, &co, &si);
        p->c[2 * n] = co; p->c[2 * n + 1] = -si;
    }
    /* the forward kernel: conj(c[j]) on j in [-(N-1), h], the negative j at M + j */
    memset(p->za, 0, (size_t)2 * M * sizeof(double));
    for (j = 0; j <= p->h; j++) { p->za[2 * j] = p->c[2 * j]; p->za[2 * j + 1] = -p->c[2 * j + 1]; }
    for (j = 1; j < N; j++) { p->za[2 * (M - j)] = p->c[2 * j]; p->za[2 * (M - j) + 1] = -p->c[2 * j + 1]; }
    _ilprime_inner_fwd(&p->inner, p->za, p->zb);
    for (j = 0; j < 2 * M; j++) p->kf[j] = p->zb[j] / (double)M;
    /* the backward kernel: c[j] on j in [-h, N-1] */
    memset(p->za, 0, (size_t)2 * M * sizeof(double));
    for (j = 0; j < N; j++) { p->za[2 * j] = p->c[2 * j]; p->za[2 * j + 1] = p->c[2 * j + 1]; }
    for (j = 1; j <= p->h; j++) { p->za[2 * (M - j)] = p->c[2 * j]; p->za[2 * (M - j) + 1] = p->c[2 * j + 1]; }
    _ilprime_inner_fwd(&p->inner, p->za, p->zb);
    for (j = 0; j < 2 * M; j++) p->kb[j] = p->zb[j] / (double)M;
    return p;
}

static inline void vfft_zrb_name_inner(vfft_zrb_plan_t *p, const char *kind, const char *shape, int tw)
{
    snprintf(p->ikind, sizeof p->ikind, "%s", kind);
    snprintf(p->ishape, sizeof p->ishape, "%s", shape);
    p->itw = tw;
}

/* ── the edges ─────────────────────────────────────────────────────────── */

/* za[n] = x[n] * c[n] for n < N: a real load, two multiplies per sample */
static inline void _zrb_chirp_in(const double *x, const double *c, double *za, int N)
{
    int n = 0;
#if defined(__AVX2__)
    for (; n + 4 <= N; n += 4)
    {
        const __m256d xv = _mm256_loadu_pd(x + n);
        const __m256d c0 = _mm256_loadu_pd(c + 2 * n), c1 = _mm256_loadu_pd(c + 2 * n + 4);
        _mm256_storeu_pd(za + 2 * n, _mm256_mul_pd(_mm256_permute4x64_pd(xv, 0x50), c0));     /* [x0 x0 x1 x1] */
        _mm256_storeu_pd(za + 2 * n + 4, _mm256_mul_pd(_mm256_permute4x64_pd(xv, 0xFA), c1)); /* [x2 x2 x3 x3] */
    }
#endif
    for (; n < N; n++) { za[2 * n] = x[n] * c[2 * n]; za[2 * n + 1] = x[n] * c[2 * n + 1]; }
}

/* out[k] = a[k] * c[k] for k < cnt (packed complex) */
static inline void _zrb_cmul_out(const double *a, const double *c, double *out, int cnt)
{
    int k = 0;
#if defined(__AVX2__)
    for (; k + 2 <= cnt; k += 2)
    {
        const __m256d av = _mm256_loadu_pd(a + 2 * k), cv = _mm256_loadu_pd(c + 2 * k);
        const __m256d cr = _mm256_movedup_pd(cv), ci = _mm256_permute_pd(cv, 0xF);
        const __m256d t = _mm256_mul_pd(_mm256_permute_pd(av, 0x5), ci);   /* [ai*ci, ar*ci] */
        _mm256_storeu_pd(out + 2 * k, _mm256_fmaddsub_pd(av, cr, t));      /* [ar*cr - ai*ci, ai*cr + ar*ci] */
    }
#endif
    for (; k < cnt; k++)
    {
        const double ar = a[2 * k], ai = a[2 * k + 1], cr = c[2 * k], ci = c[2 * k + 1];
        out[2 * k] = ar * cr - ai * ci;
        out[2 * k + 1] = ar * ci + ai * cr;
    }
}

/* a[i] *= b[i] over cnt packed complex (an odd count has a scalar tail) */
static inline void _zrb_cmul(double *a, const double *b, int cnt)
{
    const int ev = cnt & ~1;
    _ilprime_cmul_vec(a, b, (size_t)ev);
    if (ev < cnt)
    {
        const double ar = a[2 * ev], ai = a[2 * ev + 1], br = b[2 * ev], bi = b[2 * ev + 1];
        a[2 * ev] = ar * br - ai * bi;
        a[2 * ev + 1] = ar * bi + ai * br;
    }
}

/* x[n] = Re(conj(c[n]) * a[n]) = cr*ar + ci*ai for n < N: a real store */
static inline void _zrb_real_out(const double *a, const double *c, double *x, int N)
{
    int n = 0;
#if defined(__AVX2__)
    for (; n + 4 <= N; n += 4)
    {
        const __m256d p0 = _mm256_mul_pd(_mm256_loadu_pd(a + 2 * n), _mm256_loadu_pd(c + 2 * n));
        const __m256d p1 = _mm256_mul_pd(_mm256_loadu_pd(a + 2 * n + 4), _mm256_loadu_pd(c + 2 * n + 4));
        const __m256d s = _mm256_hadd_pd(p0, p1);   /* [x0 x2 x1 x3] */
        _mm256_storeu_pd(x + n, _mm256_permute4x64_pd(s, 0xD8));
    }
#endif
    for (; n < N; n++) x[n] = a[2 * n] * c[2 * n] + a[2 * n + 1] * c[2 * n + 1];
}

/* ── the transforms ────────────────────────────────────────────────────── */

/* r2c: N reals in, bins 0..h out (2(h+1) doubles). x == X is safe. */
static inline void vfft_zrb_execute_fwd(const vfft_zrb_plan_t *p, const double *x, double *X)
{
    const int N = p->N, M = p->M;
    _zrb_chirp_in(x, p->c, p->za, N);
    memset(p->za + 2 * N, 0, (size_t)2 * (M - N) * sizeof(double));
    _ilprime_inner_fwd(&p->inner, p->za, p->zb);
    _zrb_cmul(p->zb, p->kf, M);
    _ilprime_inner_bwd(&p->inner, p->zb, p->za);
    _zrb_cmul_out(p->za, p->c, X, p->h + 1);
}

/* c2r: bins 0..h in (the DC taken real), N x out. X == x is safe. */
static inline void vfft_zrb_execute_bwd(const vfft_zrb_plan_t *p, const double *X, double *x)
{
    const int N = p->N, M = p->M, h = p->h;
    double *za = p->za;
    int k;
    za[0] = X[0]; za[1] = 0.0;
    for (k = 1; k <= h; k++)
    {   /* g[k] conj(c[k]), g[k] = 2 X[k] */
        const double gr = 2.0 * X[2 * k], gi = 2.0 * X[2 * k + 1], cr = p->c[2 * k], ci = p->c[2 * k + 1];
        za[2 * k] = gr * cr + gi * ci;
        za[2 * k + 1] = gi * cr - gr * ci;
    }
    memset(za + 2 * (h + 1), 0, (size_t)2 * (M - h - 1) * sizeof(double));
    _ilprime_inner_fwd(&p->inner, za, p->zb);
    _zrb_cmul(p->zb, p->kb, M);
    _ilprime_inner_bwd(&p->inner, p->zb, za);
    _zrb_real_out(za, p->c, x, N);
}

/* K LANES, lane-major (element e of lane t at x[e K + t], bin f of lane t at
 * X[2 (f K + t)]): the one-row pipeline per lane with its two edges at the
 * lane stride -- the edges are O(N) scalar reads and writes against the
 * convolution's O(M log M). The lane race's second arm beside the column
 * form (zrb_lanes.h). Out of place. */
static inline void vfft_zrb_execute_fwd_lanes(const vfft_zrb_plan_t *p, const double *x, double *X, int K)
{
    const int N = p->N, M = p->M, h = p->h;
    const size_t ks = (size_t)K;
    double *za = p->za;
    for (int t = 0; t < K; t++)
    {
        const double *xt = x + t;
        int n, k;
        for (n = 0; n < N; n++)
        {
            const double v = xt[(size_t)n * ks];
            za[2 * n] = v * p->c[2 * n]; za[2 * n + 1] = v * p->c[2 * n + 1];
        }
        memset(za + 2 * N, 0, (size_t)2 * (M - N) * sizeof(double));
        _ilprime_inner_fwd(&p->inner, za, p->zb);
        _zrb_cmul(p->zb, p->kf, M);
        _ilprime_inner_bwd(&p->inner, p->zb, za);
        for (k = 0; k <= h; k++)
        {
            const double ar = za[2 * k], ai = za[2 * k + 1], cr = p->c[2 * k], ci = p->c[2 * k + 1];
            double *o = X + 2 * ((size_t)k * ks + (size_t)t);
            o[0] = ar * cr - ai * ci;
            o[1] = ar * ci + ai * cr;
        }
    }
}
static inline void vfft_zrb_execute_bwd_lanes(const vfft_zrb_plan_t *p, const double *X, double *x, int K)
{
    const int N = p->N, M = p->M, h = p->h;
    const size_t ks = (size_t)K;
    double *za = p->za;
    for (int t = 0; t < K; t++)
    {
        int n, k;
        za[0] = X[2 * t]; za[1] = 0.0;
        for (k = 1; k <= h; k++)
        {
            const double *i = X + 2 * ((size_t)k * ks + (size_t)t);
            const double gr = 2.0 * i[0], gi = 2.0 * i[1], cr = p->c[2 * k], ci = p->c[2 * k + 1];
            za[2 * k] = gr * cr + gi * ci;
            za[2 * k + 1] = gi * cr - gr * ci;
        }
        memset(za + 2 * (h + 1), 0, (size_t)2 * (M - h - 1) * sizeof(double));
        _ilprime_inner_fwd(&p->inner, za, p->zb);
        _zrb_cmul(p->zb, p->kb, M);
        _ilprime_inner_bwd(&p->inner, p->zb, za);
        for (n = 0; n < N; n++)
            x[(size_t)n * ks + (size_t)t] = za[2 * n] * p->c[2 * n] + za[2 * n + 1] * p->c[2 * n + 1];
    }
}

/* K rows, each at its pitch (doubles) */
static inline void vfft_zrb_execute_fwd_rows(const vfft_zrb_plan_t *p, const double *x, size_t xp,
                                             double *X, size_t Xp, size_t K)
{
    for (size_t r = 0; r < K; r++) vfft_zrb_execute_fwd(p, x + r * xp, X + r * Xp);
}
static inline void vfft_zrb_execute_bwd_rows(const vfft_zrb_plan_t *p, const double *X, size_t Xp,
                                             double *x, size_t xp, size_t K)
{
    for (size_t r = 0; r < K; r++) vfft_zrb_execute_bwd(p, X + r * Xp, x + r * xp);
}

#endif /* VFFT_ZRB_H */
