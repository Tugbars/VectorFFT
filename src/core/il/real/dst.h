/* dst.h -- DST-II and DST-III (VFFT_DST2, VFFT_DST3): 1D, N reals in and N reals out, any N, one transform,
 * on the interleaved real FFT of the same length.
 *
 * THE DEFINITIONS (include/vfft.h; unnormalized, each the other's inverse up to 2N):
 *   DST-II   Y[k] = 2 sum_{n=0..N-1} x[n] sin(pi (k+1)(2n+1) / 2N)
 *   DST-III  Y[k] = (-1)^k X[N-1] + 2 sum_{n=0..N-2} X[n] sin(pi (n+1)(2k+1) / 2N)
 *
 * THE METHOD: one real FFT of length N between two passes (J. Makhoul, "A fast cosine transform in one and
 * two dimensions", IEEE Trans. ASSP-28, 1980, for the cosine transform; the sine transform is the cosine
 * transform of the input with every other sign flipped, read backwards, and both of those ride in the two
 * passes). With c_k = cos(pi k / 2N), s_k = sin(pi k / 2N):
 *   DST-II   reorder   v[m] = x[2m], v[N-1-m] = -x[2m+1]         (odd N: v[(N-1)/2] = x[N-1])
 *            R2C at N  V[0 .. N/2]
 *            rotate    Y[N-1] = 2 Re V[0]
 *                      Y[N-1-k] = 2 (c_k Re V[k] + s_k Im V[k]),  Y[k-1] = 2 (s_k Re V[k] - c_k Im V[k]),
 *                      k = 1 .. (N-1)/2;  even N: Y[N/2-1] = sqrt 2 Re V[N/2]
 *   DST-III  the same steps transposed:
 *            rotate    V[0] = X[N-1];  V[k] = (c_k X[N-1-k] + s_k X[k-1]) + i (s_k X[N-1-k] - c_k X[k-1]);
 *                      even N: V[N/2] = sqrt 2 X[N/2-1]
 *            C2R at N  v[0 .. N-1]
 *            reorder   Y[2m] = v[m], Y[2m+1] = -v[N-1-m]         (odd N: Y[N-1] = v[(N-1)/2])
 * Each pass reads and writes whole vectors, 4 doubles at AVX2 and 8 at AVX-512, with a scalar tail.
 *
 * THE PLAN holds the two real FFTs at N, out of place (an R2C and a C2R plan of the real door), the rotation
 * tables and two work buffers allocated at create: W, N + 2 doubles, the half spectrum; V, N doubles, the
 * reordered reals. A DST-II whose output is not its input reorders into the output buffer and leaves V alone.
 * Either transform takes out == in. A plan of T threads threads inside its real FFTs; the passes run on the
 * calling thread.
 *
 * A VFFT_DST2 plan runs DST-II forward and DST-III backward; a VFFT_DST3 plan the other way round.
 *
 * THE WISDOM (owner, 2026-11-14): a DST cell has its own rows in the real shard, one per mathematical
 * transform -- t=dst2|dst3 n=N q=1 ord=nat place=oop [nthreads=T] | eng=dst and its real FFT under dst_
 * (vw2_dst_key, il/wisdom/wisdom2_real_il.h). The real FFT is a CHILD IN ROLE (il/wisdom/wisdom2_child.h):
 * created against a private store seeded from the DST's row, it replays the recipe there or races, and what
 * it banked goes back onto the DST's row. A DST never reads or writes an r2c or a c2r row. Its race runs in
 * the DST's role (il/real/real_role.h): every candidate FFT is timed as one whole DST -- the reorder pass,
 * that FFT, the rotation pass -- and the fastest whole DST wins. The plan is placement-blind, so one row
 * serves both placements; a plan of either kind reads both rows. Under VFFT_FLAG_ESTIMATE the two FFTs are
 * planned by formulas, nothing is measured and no store is read or written.
 *
 * POSITION IN vfft.c: after il/real/real_create_il.h (the real door) and before the front door's create. */
#ifndef VFFT_IL_REAL_DST_H
#define VFFT_IL_REAL_DST_H
#ifndef VFFT_FLOAT   /* FP64 only: the FP32 unit serves 1D C2C */

#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

#include "common/math/tw_exact.h"     /* the rotation tables, each entry rounded once */
#include "il/real/real_role.h"        /* the role the real FFT is raced in: the whole DST */
#include "il/wisdom/wisdom2_real_il.h" /* the DST cell's row */
#include "il/wisdom/wisdom2_child.h"  /* the real FFT as a child in role: its private store, its rows on the DST's row */

typedef struct vfft_dst_s
{
    int N;
    struct vfft_plan_s *r2c, *c2r;   /* the real FFTs at N, out of place (owned); NULL at N = 1 */
    struct vfft_wisdom_s *S2, *S3;   /* their private stores (the R2C's, the C2R's): owned, never saved; NULL
                                      * under VFFT_FLAG_ESTIMATE and at N = 1 */
    double *W;                       /* N + 2 doubles: the half spectrum */
    double *V;                       /* N doubles: the reordered reals */
    double *tab;                     /* the four tables below, one allocation */
    double *tc, *ts;                 /* cos and sin of pi k / 2N, k = 0 .. N/2 */
    double *t2c, *t2s;               /* twice those */
} vfft_dst_t;

/* ── the reorder passes ─────────────────────────────────────────────────── */

/* v[m] = x[2m], v[N-1-m] = -x[2m+1], m = 0 .. N/2 - 1; odd N: v[(N-1)/2] = x[N-1]. x and v apart. */
static inline void _dst_reorder(const double *x, double *v, int N)
{
    const int np = N / 2;
    int m = 0;
#if defined(__AVX512F__)
    {
        const __m512i EV = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14), OD = _mm512_setr_epi64(15, 13, 11, 9, 7, 5, 3, 1);
        const __m512d NEG = _mm512_set1_pd(-0.0);
        for (; m + 8 <= np; m += 8)
        {
            const __m512d a = _mm512_loadu_pd(x + 2 * m), b = _mm512_loadu_pd(x + 2 * m + 8);
            _mm512_storeu_pd(v + m, _mm512_permutex2var_pd(a, EV, b));
            _mm512_storeu_pd(v + N - 8 - m, _mm512_xor_pd(_mm512_permutex2var_pd(a, OD, b), NEG));
        }
    }
#endif
#if defined(__AVX2__)
    {
        const __m256d NEG = _mm256_set1_pd(-0.0);
        for (; m + 4 <= np; m += 4)
        {
            const __m256d a = _mm256_loadu_pd(x + 2 * m), b = _mm256_loadu_pd(x + 2 * m + 4);
            _mm256_storeu_pd(v + m, _mm256_permute4x64_pd(_mm256_unpacklo_pd(a, b), 0xD8));
            _mm256_storeu_pd(v + N - 4 - m, _mm256_xor_pd(_mm256_permute4x64_pd(_mm256_unpackhi_pd(a, b), 0x27), NEG));
        }
    }
#endif
    for (; m < np; m++)
    {
        v[m] = x[2 * m];
        v[N - 1 - m] = -x[2 * m + 1];
    }
    if (N & 1)
        v[np] = x[N - 1];
}

/* y[2m] = v[m], y[2m+1] = -v[N-1-m], m = 0 .. N/2 - 1; odd N: y[N-1] = v[(N-1)/2]. v and y apart. */
static inline void _dst_unreorder(const double *v, double *y, int N)
{
    const int np = N / 2;
    int m = 0;
#if defined(__AVX512F__)
    {
        const __m512i LO = _mm512_setr_epi64(0, 15, 1, 14, 2, 13, 3, 12), HI = _mm512_setr_epi64(4, 11, 5, 10, 6, 9, 7, 8);
        const __m512d NEG = _mm512_set1_pd(-0.0);
        for (; m + 8 <= np; m += 8)
        {
            const __m512d a = _mm512_loadu_pd(v + m), b = _mm512_xor_pd(_mm512_loadu_pd(v + N - 8 - m), NEG);
            _mm512_storeu_pd(y + 2 * m, _mm512_permutex2var_pd(a, LO, b));
            _mm512_storeu_pd(y + 2 * m + 8, _mm512_permutex2var_pd(a, HI, b));
        }
    }
#endif
#if defined(__AVX2__)
    {
        const __m256d NEG = _mm256_set1_pd(-0.0);
        for (; m + 4 <= np; m += 4)
        {
            const __m256d a = _mm256_loadu_pd(v + m);
            const __m256d o = _mm256_xor_pd(_mm256_permute4x64_pd(_mm256_loadu_pd(v + N - 4 - m), 0x1B), NEG);
            const __m256d lo = _mm256_unpacklo_pd(a, o), hi = _mm256_unpackhi_pd(a, o);
            _mm256_storeu_pd(y + 2 * m, _mm256_permute2f128_pd(lo, hi, 0x20));
            _mm256_storeu_pd(y + 2 * m + 4, _mm256_permute2f128_pd(lo, hi, 0x31));
        }
    }
#endif
    for (; m < np; m++)
    {
        y[2 * m] = v[m];
        y[2 * m + 1] = -v[N - 1 - m];
    }
    if (N & 1)
        y[N - 1] = v[np];
}

/* ── the rotation passes ────────────────────────────────────────────────── */

/* DST-II: the half spectrum V (interleaved, N/2 + 1 bins) into the N outputs; t2c, t2s the doubled tables.
 * V and y apart. */
static inline void _dst2_rotate(const double *V, double *y, const double *t2c, const double *t2s, int N)
{
    const int km = (N - 1) / 2;
    int k = 1;
    y[N - 1] = 2.0 * V[0];
#if defined(__AVX512F__)
    {
        const __m512i RE = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14), IM = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
        const __m512i REV = _mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
        for (; k + 7 <= km; k += 8)
        {
            const __m512d z0 = _mm512_loadu_pd(V + 2 * k), z1 = _mm512_loadu_pd(V + 2 * k + 8);
            const __m512d re = _mm512_permutex2var_pd(z0, RE, z1), im = _mm512_permutex2var_pd(z0, IM, z1);
            const __m512d c = _mm512_loadu_pd(t2c + k), s = _mm512_loadu_pd(t2s + k);
            _mm512_storeu_pd(y + k - 1, _mm512_fmsub_pd(s, re, _mm512_mul_pd(c, im)));
            _mm512_storeu_pd(y + N - 8 - k, _mm512_permutexvar_pd(REV, _mm512_fmadd_pd(c, re, _mm512_mul_pd(s, im))));
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
        _mm256_storeu_pd(y + k - 1, _mm256_fmsub_pd(s, re, _mm256_mul_pd(c, im)));
        _mm256_storeu_pd(y + N - 4 - k, _mm256_permute4x64_pd(_mm256_fmadd_pd(c, re, _mm256_mul_pd(s, im)), 0x1B));
    }
#endif
    for (; k <= km; k++)
    {
        const double re = V[2 * k], im = V[2 * k + 1];
        y[k - 1] = t2s[k] * re - t2c[k] * im;
        y[N - 1 - k] = t2c[k] * re + t2s[k] * im;
    }
    if (!(N & 1))
        y[N / 2 - 1] = t2c[N / 2] * V[N];
}

/* DST-III: the N inputs into the half spectrum V the C2R reads (the imaginary parts of bin 0 and, at an even
 * N, bin N/2 written zero). X and V apart. */
static inline void _dst3_rotate(const double *X, double *V, const double *tc, const double *ts, const double *t2c, int N)
{
    const int km = (N - 1) / 2;
    int k = 1;
    V[0] = X[N - 1];
    V[1] = 0.0;
#if defined(__AVX512F__)
    {
        const __m512i REV = _mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
        const __m512i LO = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11), HI = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
        for (; k + 7 <= km; k += 8)
        {
            const __m512d a = _mm512_loadu_pd(X + k - 1);                                        /* X[k-1] */
            const __m512d d = _mm512_permutexvar_pd(REV, _mm512_loadu_pd(X + N - 8 - k));         /* X[N-1-k] */
            const __m512d c = _mm512_loadu_pd(tc + k), s = _mm512_loadu_pd(ts + k);
            const __m512d vr = _mm512_fmadd_pd(c, d, _mm512_mul_pd(s, a)), vi = _mm512_fmsub_pd(s, d, _mm512_mul_pd(c, a));
            _mm512_storeu_pd(V + 2 * k, _mm512_permutex2var_pd(vr, LO, vi));
            _mm512_storeu_pd(V + 2 * k + 8, _mm512_permutex2var_pd(vr, HI, vi));
        }
    }
#endif
#if defined(__AVX2__)
    for (; k + 3 <= km; k += 4)
    {
        const __m256d a = _mm256_loadu_pd(X + k - 1);
        const __m256d d = _mm256_permute4x64_pd(_mm256_loadu_pd(X + N - 4 - k), 0x1B);
        const __m256d c = _mm256_loadu_pd(tc + k), s = _mm256_loadu_pd(ts + k);
        const __m256d vr = _mm256_fmadd_pd(c, d, _mm256_mul_pd(s, a)), vi = _mm256_fmsub_pd(s, d, _mm256_mul_pd(c, a));
        const __m256d lo = _mm256_unpacklo_pd(vr, vi), hi = _mm256_unpackhi_pd(vr, vi);
        _mm256_storeu_pd(V + 2 * k, _mm256_permute2f128_pd(lo, hi, 0x20));
        _mm256_storeu_pd(V + 2 * k + 4, _mm256_permute2f128_pd(lo, hi, 0x31));
    }
#endif
    for (; k <= km; k++)
    {
        const double a = X[k - 1], d = X[N - 1 - k];
        V[2 * k] = tc[k] * d + ts[k] * a;
        V[2 * k + 1] = ts[k] * d - tc[k] * a;
    }
    if (!(N & 1))
    {
        V[N] = t2c[N / 2] * X[N / 2 - 1];
        V[N + 1] = 0.0;
    }
}

/* ── execute ────────────────────────────────────────────────────────────── */

/* DST-II of x into y (y may be x) */
static inline void _dst2_run(const vfft_dst_t *d, const double *x, double *y)
{
    if (d->N == 1)
    {
        y[0] = 2.0 * x[0];
        return;
    }
    {
        double *v = (y != x) ? y : d->V;   /* the caller's output holds the reordered reals when it is free */
        _dst_reorder(x, v, d->N);
        _vfft_exec6((vfft_plan)d->r2c, VFFT_FORWARD, v, NULL, d->W, NULL);
        _dst2_rotate(d->W, y, d->t2c, d->t2s, d->N);
    }
}
/* DST-III of X into y (y may be X) */
static inline void _dst3_run(const vfft_dst_t *d, const double *X, double *y)
{
    if (d->N == 1)
    {
        y[0] = X[0];
        return;
    }
    _dst3_rotate(X, d->W, d->tc, d->ts, d->t2c, d->N);
    _vfft_exec6((vfft_plan)d->c2r, VFFT_BACKWARD, d->W, NULL, d->V, NULL);
    _dst_unreorder(d->V, y, d->N);
}

/* the bound executors (vfft_plan_s.k1_exec, k1_go): never decline, either pointer pattern */
static int _k1x_dst2(struct vfft_plan_s *h, vfft_dir_t dir, const vfft_real *zin, vfft_real *zout)
{
    (void)dir;
    _dst2_run(h->dst, zin, zout);
    return 0;
}
static int _k1x_dst3(struct vfft_plan_s *h, vfft_dir_t dir, const vfft_real *zin, vfft_real *zout)
{
    (void)dir;
    _dst3_run(h->dst, zin, zout);
    return 0;
}
/* by direction: the plan's own transform forward, its inverse backward */
static int _k1x_dst(struct vfft_plan_s *h, vfft_dir_t dir, const vfft_real *zin, vfft_real *zout)
{
    if ((h->transform == VFFT_DST2) == (dir == VFFT_FORWARD))
        _dst2_run(h->dst, zin, zout);
    else
        _dst3_run(h->dst, zin, zout);
    return 0;
}

/* ── the role, create and destroy ───────────────────────────────────────── */

/* THE ROLE (il/real/real_role.h): one whole DST with the candidate FFT h inside, on the create's own buffers
 * x (the input) and y (the output), the plan's tables and work buffers. What every candidate of the real
 * door's races is timed as while the DST's FFT is created. */
typedef struct
{
    const vfft_dst_t *d;
    double *x, *y;
} _dst_role_t;
static void _dst2_role_run(void *ctx, struct vfft_plan_s *h)
{
    const _dst_role_t *r = (const _dst_role_t *)ctx;
    const vfft_dst_t *d = r->d;
    _dst_reorder(r->x, r->y, d->N);
    _real_il_exec_any(h, r->y, d->W);
    _dst2_rotate(d->W, r->y, d->t2c, d->t2s, d->N);
}
static void _dst3_role_run(void *ctx, struct vfft_plan_s *h)
{
    const _dst_role_t *r = (const _dst_role_t *)ctx;
    const vfft_dst_t *d = r->d;
    _dst3_rotate(r->x, d->W, d->tc, d->ts, d->t2c, d->N);
    _real_il_exec_any(h, d->W, d->V);
    _dst_unreorder(d->V, r->y, d->N);
}

static void _vfft_dst_destroy(vfft_dst_t *d)
{
    if (!d)
        return;
    if (d->r2c)
        vfft_destroy((vfft_plan)d->r2c);
    if (d->c2r)
        vfft_destroy((vfft_plan)d->c2r);
    vfft_child_store_free(d->S2);
    vfft_child_store_free(d->S3);
    vfft_aligned_free(d->W);
    vfft_aligned_free(d->V);
    vfft_aligned_free(d->tab);
    free(d);
}

static vfft_plan _vfft_create_estimate(const vfft_cfg_t *cfg);   /* il/estimate/c2c_estimate_il.h */

/* One of the DST's two real FFTs at N, out of place: the R2C inside a DST-II (is_c2r = 0) or the C2R inside a
 * DST-III. est: planned by formulas. Else a child in role: its private store seeded from the DST cell's row
 * (empty when the row carries no recipe, or there is no store), the create against it with nothing persisted,
 * every candidate of a race timed as the whole DST on x and y; where the child raced, or the row did not carry
 * it yet, the cell's row is banked afresh -- the time of one DST call from the child's winner -- with the
 * child's rows under dst_ (*changed = 1: the caller saves). NULL: the real door's reason is on the record. */
static struct vfft_plan_s *_dst_child(const vfft_cfg_t *cfg, struct vfft_wisdom_s *W, vfft_dst_t *d, int is_c2r,
                                      int est, double *x, double *y, int *changed)
{
    vfft_cfg_t c = *cfg;
    c.transform = is_c2r ? VFFT_C2R : VFFT_R2C;
    c.dims = 1;
    c.howmany = 1;
    c.layout = VFI_LAYOUT_IL;
    c.placement = VFFT_OUTOFPLACE;
    c.order = VFI_ORDER_DEFAULT;
    c.batch_geom = VFFT_BATCH_DEFAULT;
    c.owned_buffers = 0;
    c.destroy_input = 0;
    if (est)
        return (struct vfft_plan_s *)_vfft_create_estimate(&c);
    {
        const int T = _vfft_plan_threads(cfg), usable = W && !W->vw2_off_oop;
        struct vfft_wisdom_s **Sp = is_c2r ? &d->S3 : &d->S2;
        _dst_role_t role;
        struct vfft_plan_s *ch;
        vw2_key_t pk;
        vw2_dst_key(&pk, d->N, is_c2r, T);
        *Sp = vfft_child_store_for(usable ? &W->vw2 : NULL, &pk, "dst_");
        if (!*Sp)
        {
            _vfft_warn("vfft_create: %s N=%d: out of memory (the real FFT's store)", _vfft_tname(cfg->transform), d->N);
            return NULL;
        }
        c.wisdom = (vfft_wisdom *)*Sp;
        c.wisdom_write = 0;   /* a private store's plan never saves (il/wisdom/wisdom2_child.h) */
        role.d = d;
        role.x = x;
        role.y = y;
        _real_il_role.run = is_c2r ? _dst3_role_run : _dst2_role_run;
        _real_il_role.ctx = &role;
        _real_il_role.transform = (int)c.transform;
        _real_il_role.N = d->N;
        ch = (struct vfft_plan_s *)_vfft_create_cfg(&c);
        memset(&_real_il_role, 0, sizeof _real_il_role);
        if (ch && usable && (*Sp)->vw2.nrec > 0 &&
            (vfft_child_store_raced(*Sp) || !vfft_child_row_has(vfft_child_parent_row(&W->vw2, &pk), "dst_")))
        {
            vw2_key_t ck;
            const vw2_rec_t *cr;
            const char *nsv;
            vw2_real_il_key(&ck, d->N, is_c2r, 0, T);
            cr = vw2_lookup(&(*Sp)->vw2, &ck);
            nsv = cr ? vw2_rec_get(cr, "ns") : NULL;
            if (vw2_dst_bank(&W->vw2, d->N, is_c2r, T, nsv ? atof(nsv) : 0.0) == VW2_OK &&
                vfft_child_row_update(&W->vw2, &pk, "dst_", *Sp))
                *changed = 1;
            else
                fprintf(stderr, "vfft: DST verdict NOT banked at N=%d (%s) -- the cell will re-race\n", d->N,
                        is_c2r ? "dst3" : "dst2");
        }
        return ch;
    }
}

/* THE DST DOOR: a VFFT_DST2 or VFFT_DST3 request, 1D, one transform. est = 1 (VFFT_FLAG_ESTIMATE): the two
 * real FFTs planned by formulas, nothing measured, no store (W unused); est = 0: each a child in role of the
 * DST cell's row in W. NULL with the reason on the error record. */
static vfft_plan _vfft_create_dst_il(const vfft_cfg_t *cfg, struct vfft_wisdom_s *W, int est)
{
    const int N = cfg->n[0];
    const char *tn = _vfft_tname(cfg->transform);
    struct vfft_plan_s *h;
    vfft_dst_t *d;
    if (cfg->dims >= 2)
    {
        _vfft_warn("vfft_create: %dD %s is not served: DST-II and DST-III plans are 1D", cfg->dims, tn);
        return NULL;
    }
    if (cfg->howmany > 1)
    {
        _vfft_warn("vfft_create: %s N=%d howmany=%zu is not served: a DST-II or DST-III plan runs one transform "
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
        d->r2c = _dst_child(cfg, W, d, 0, est, x, y, &changed);
        if (d->r2c)
            d->c2r = _dst_child(cfg, W, d, 1, est, x, y, &changed);
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
    h->k1_exec = _k1x_dst;
    h->k1_go[VFFT_FORWARD] = cfg->transform == VFFT_DST2 ? _k1x_dst2 : _k1x_dst3;
    h->k1_go[VFFT_BACKWARD] = cfg->transform == VFFT_DST2 ? _k1x_dst3 : _k1x_dst2;
    return (vfft_plan)h;
}

#endif /* !VFFT_FLOAT */
#endif /* VFFT_IL_REAL_DST_H */
