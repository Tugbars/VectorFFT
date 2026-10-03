/* zrb_mt.h - the real Bluestein's THREADED form (il/real/zrb.h; owner,
 * 2026-10-03: the convolutions take c2c's threading).
 *
 * One row of zrb is a chirp-z convolution at M around two edges. The
 * threaded form runs the inner ZTURN-T's own threaded walk (ztt_mt.h:
 * mt=1 BLOCKS, mt=2 TILES -- the arm is the plan's) and cuts the passes
 * around it into ranges on the same pool, one fork-join each:
 *   r2c   the chirp-in with the zero pad, the pointwise multiply (over M),
 *         the chirp-out of bins 0..h;
 *   c2r   the half-spectrum spread with the zero pad, the pointwise multiply,
 *         the real store of N samples.
 * The edge kernels are vectorised (four reals, or two complex, per step,
 * with a scalar tail), so every cut sits on their step: each element takes
 * the code path the serial run gives it, and the threaded output is BITWISE
 * the serial plan's. A plan whose inner is not ZTURN-T has no threaded form.
 * K = 1 only (a lane-major batch runs through the bridge, serial).
 *
 * Banked per thread count by the odd door (odd_build.h: eng=zrb ... mt=);
 * this header is the execution only. Included by vfft.c ONLY, after
 * ztt_mt.h. Nesting is forbidden by the pool's law: this runs on the caller
 * thread of a plan whose own nthreads > 1, never from a worker.
 */
#ifndef VFFT_ZRB_MT_H
#define VFFT_ZRB_MT_H

#include "zrb.h"
#include "il/rank1/ztt_mt.h"

extern long _vfft_zrb_mt_count;   /* vfft.c: the engagement counter */

/* bind the arm for T: the inner ZTURN-T's walk (1 BLOCKS, 2 TILES); 0 =
 * declined or unbound -- the plan then runs serial */
static inline int vfft_zrb_mt_bind(vfft_zrb_plan_t *p, int T, int arm)
{
    p->mt = 0;
    p->mt_t = 0;
    if (!p->inner.pt)
        return 0;
    if (!vfft_ztt_mt_bind(p->inner.pt, T, arm))
        return 0;
    p->mt = arm;
    p->mt_t = T;
    return 1;
}

/* one thread's ranges */
typedef struct
{
    int pass;                 /* 0 r2c in, 1 multiply, 2 r2c out, 3 c2r in, 4 c2r out */
    const vfft_zrb_plan_t *p;
    const double *in, *k;
    double *out;
    size_t lo, hi;            /* the edge's range */
    size_t zlo, zhi;          /* the zero pad's range (passes 0 and 3) */
} _zrb_mt_arg;
static void _zrb_mt_tramp(void *v)
{
    const _zrb_mt_arg *a = (const _zrb_mt_arg *)v;
    const vfft_zrb_plan_t *p = a->p;
    double *za = p->za;
    size_t k;
    switch (a->pass)
    {
    case 0:
        if (a->lo < a->hi) _zrb_chirp_in(a->in + a->lo, p->c + 2 * a->lo, za + 2 * a->lo, (int)(a->hi - a->lo));
        break;
    case 1:
        if (a->lo < a->hi) _zrb_cmul(p->zb + 2 * a->lo, a->k + 2 * a->lo, (int)(a->hi - a->lo));
        break;
    case 2:
        if (a->lo < a->hi) _zrb_cmul_out(za + 2 * a->lo, p->c + 2 * a->lo, a->out + 2 * a->lo, (int)(a->hi - a->lo));
        break;
    case 3:
        for (k = a->lo; k < a->hi; k++)
        {   /* g[k] conj(c[k]), g[k] = 2 X[k]; the DC taken real */
            if (k == 0) { za[0] = a->in[0]; za[1] = 0.0; continue; }
            {
                const double gr = 2.0 * a->in[2 * k], gi = 2.0 * a->in[2 * k + 1], cr = p->c[2 * k], ci = p->c[2 * k + 1];
                za[2 * k] = gr * cr + gi * ci;
                za[2 * k + 1] = gi * cr - gr * ci;
            }
        }
        break;
    default:
        if (a->lo < a->hi) _zrb_real_out(za + 2 * a->lo, p->c + 2 * a->lo, a->out + a->lo, (int)(a->hi - a->lo));
        break;
    }
    if ((a->pass == 0 || a->pass == 3) && a->zlo < a->zhi)
        memset(za + 2 * a->zlo, 0, 2 * (a->zhi - a->zlo) * sizeof(double));
}
/* one pass: the edge over [0, n) cut at multiples of `step`, and (passes 0
 * and 3) the zero pad over [z0, z1) cut evenly */
static void _zrb_mt_pass(int T, const _zrb_mt_arg *proto, size_t n, size_t step, size_t z0, size_t z1)
{
    _zrb_mt_arg a[THREAD_POOL_MAX_DISPATCH];
    int w;
    for (w = 0; w < T; w++)
    {
        a[w] = *proto;
        a[w].lo = (n * (size_t)w / (size_t)T) / step * step;
        a[w].hi = (w == T - 1) ? n : (n * (size_t)(w + 1) / (size_t)T) / step * step;
        a[w].zlo = z0 + (z1 - z0) * (size_t)w / (size_t)T;
        a[w].zhi = z0 + (z1 - z0) * (size_t)(w + 1) / (size_t)T;
    }
    thread_pool_run(T, _zrb_mt_tramp, a, sizeof a[0]);
}
static inline void _zrb_mt_inner(const vfft_zrb_plan_t *p, const double *zi, double *zo, int bwd)
{
    if (vfft_ztt_execute_mt(p->inner.pt, zi, zo, bwd))
        return;
    if (bwd) _ilprime_inner_bwd(&p->inner, zi, zo);
    else     _ilprime_inner_fwd(&p->inner, zi, zo);
}

/* Returns 1 when it ran threaded, 0 when the caller must run serial (no arm
 * bound, no ZTURN-T inner, T < 2, or a pool clamped below the bound T).
 * x == X is safe both ways, as in the serial run. */
static inline int vfft_zrb_execute_mt(const vfft_zrb_plan_t *p, const double *in, double *out, int c2r)
{
    const int T = thread_pool_workers_for(p->mt_t);
    const size_t N = (size_t)p->N, M = (size_t)p->M, h = (size_t)p->h;
    _zrb_mt_arg a;
    if (p->mt <= 0 || T < 2 || T != p->mt_t || !p->inner.pt)
        return 0;
    memset(&a, 0, sizeof a);
    a.p = p;
    a.in = in;
    a.out = out;
    if (!c2r)
    {
        a.k = p->kf;
        a.pass = 0; _zrb_mt_pass(T, &a, N, 4, N, M);
        _zrb_mt_inner(p, p->za, p->zb, 0);
        a.pass = 1; _zrb_mt_pass(T, &a, M, 2, 0, 0);
        _zrb_mt_inner(p, p->zb, p->za, 1);
        a.pass = 2; _zrb_mt_pass(T, &a, h + 1, 2, 0, 0);
    }
    else
    {
        a.k = p->kb;
        a.pass = 3; _zrb_mt_pass(T, &a, h + 1, 1, h + 1, M);
        _zrb_mt_inner(p, p->za, p->zb, 0);
        a.pass = 1; _zrb_mt_pass(T, &a, M, 2, 0, 0);
        _zrb_mt_inner(p, p->zb, p->za, 1);
        a.pass = 4; _zrb_mt_pass(T, &a, N, 4, 0, 0);
    }
    _vfft_zrb_mt_count++;
    return 1;
}

#endif /* VFFT_ZRB_MT_H */
