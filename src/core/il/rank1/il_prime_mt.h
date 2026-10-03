/* il_prime_mt.h - the prime cell's THREADED form (Rader / Bluestein,
 * il_prime.h; owner, 2026-10-03: the convolutions take c2c's threading).
 *
 * A convolution is two FFTs at the inner length M and O(N) / O(M) passes
 * around them. The threaded form runs the inner ZTURN-T's own threaded walk
 * (ztt_mt.h: mt=1 BLOCKS, mt=2 TILES -- the arm is the plan's) and cuts the
 * convolution's passes into ranges on the same pool, one fork-join each:
 *   Bluestein   the modulate with its zero pad (over M), the pointwise
 *               multiply (over M), the demodulate (over N);
 *   Rader       the gather (over N-1), the pointwise multiply (over N-1),
 *               the scatter (over N-1). The DC sum stays on the caller,
 *               serial, before the gather: a split sum would round apart.
 * Every pass is elementwise, cut at even boundaries (the packed multiply
 * pairs complex from its range's start), so every element takes the code
 * path the serial run gives it: the threaded output is BITWISE the serial
 * plan's. A plan whose inner is not ZTURN-T has no threaded form.
 *
 * Banked per thread count by the planners (k1_commit.h for the c2c cell,
 * zr2c_build.h / zrp_build.h for the zr2c child), raced at T by
 * vfft_ilprime_mt_race; this header is the execution and the race only.
 * Included by vfft.c ONLY, after ztt_mt.h (the pool's state is per
 * translation unit). Nesting is forbidden by the pool's law: this runs on
 * the caller thread of a plan whose own nthreads > 1, never from a worker.
 */
#ifndef VFFT_IL_PRIME_MT_H
#define VFFT_IL_PRIME_MT_H

#include "il_prime.h"
#include "il/rank1/ztt_mt.h"

extern long _vfft_ilpr_mt_count;   /* vfft.c: the engagement counter */

/* bind the arm for T: the inner ZTURN-T's walk (1 BLOCKS, 2 TILES); 0 =
 * declined or unbound (T < 2, arm 0, no ZTURN-T inner, or an arm the inner
 * cannot run) -- the plan then runs serial */
static inline int vfft_ilprime_mt_bind(vfft_ilprime_plan_t *p, int T, int arm)
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

/* one thread's range of one pass */
typedef struct
{
    int pass;                 /* 0 bs_in, 1 multiply, 2 bs_out, 3 rd_in, 4 rd_out */
    const vfft_ilprime_plan_t *p;
    const double *zin, *c, *k;
    double *zout;
    const int *perm;
    double x0r, x0i;
    size_t lo, hi;
} _ilpr_mt_arg;
static void _ilpr_mt_tramp(void *v)
{
    const _ilpr_mt_arg *a = (const _ilpr_mt_arg *)v;
    const vfft_ilprime_plan_t *p = a->p;
    if (a->lo >= a->hi)
        return;
    switch (a->pass)
    {
    case 0: _ilprime_bs_in(a->zin, a->c, p->za, (size_t)p->N, a->lo, a->hi); break;
    case 1: _ilprime_cmul_vec(p->zb + 2 * a->lo, a->k + 2 * a->lo, a->hi - a->lo); break;
    case 2: _ilprime_bs_out(p->za, a->c, a->zout, a->lo, a->hi); break;
    case 3: _ilprime_rd_in(a->zin, a->perm, p->za, a->lo, a->hi); break;
    default: _ilprime_rd_out(p->za, a->perm, a->zout, a->x0r, a->x0i, a->lo, a->hi); break;
    }
}
/* one pass over [0, n) cut into T ranges at even boundaries */
static void _ilpr_mt_pass(int T, const _ilpr_mt_arg *proto, size_t n)
{
    _ilpr_mt_arg a[THREAD_POOL_MAX_DISPATCH];
    int w;
    for (w = 0; w < T; w++)
    {
        a[w] = *proto;
        a[w].lo = (n * (size_t)w / (size_t)T) & ~(size_t)1;
        a[w].hi = (w == T - 1) ? n : (n * (size_t)(w + 1) / (size_t)T) & ~(size_t)1;
    }
    thread_pool_run(T, _ilpr_mt_tramp, a, sizeof a[0]);
}
/* the inner FFT through its threaded walk (it declines only on a clamped
 * pool, and then the serial walk runs: the same bits) */
static inline void _ilpr_mt_inner(const vfft_ilprime_plan_t *p, const double *zi, double *zo, int bwd)
{
    if (vfft_ztt_execute_mt(p->inner.pt, zi, zo, bwd))
        return;
    if (bwd) _ilprime_inner_bwd(&p->inner, zi, zo);
    else     _ilprime_inner_fwd(&p->inner, zi, zo);
}

/* Returns 1 when it ran threaded, 0 when the caller must run serial: no arm
 * bound, no ZTURN-T inner, T < 2, or a live pool clamped below the bound T
 * (the inner's cut is for exactly mt_t threads). zin == zout is safe, as in
 * the serial run: every read of zin completes before the first zout write. */
static inline int vfft_ilprime_execute_mt(const vfft_ilprime_plan_t *p, const double *zin, double *zout, int bwd)
{
    const int T = thread_pool_workers_for(p->mt_t);
    _ilpr_mt_arg a;
    if (p->mt <= 0 || T < 2 || T != p->mt_t || !p->inner.pt)
        return 0;
    memset(&a, 0, sizeof a);
    a.p = p;
    a.zin = zin;
    a.zout = zout;
    if (!p->method)
    {   /* Bluestein */
        a.c = bwd ? p->chb : p->chf;
        a.k = bwd ? p->kb : p->kf;
        a.pass = 0; _ilpr_mt_pass(T, &a, (size_t)p->M);
        _ilpr_mt_inner(p, p->za, p->zb, 0);
        a.pass = 1; _ilpr_mt_pass(T, &a, (size_t)p->M);
        _ilpr_mt_inner(p, p->zb, p->za, 1);
        a.pass = 2; _ilpr_mt_pass(T, &a, (size_t)p->N);
    }
    else
    {   /* Rader */
        const int N = p->N, nm1 = p->M;
        const int *gat = bwd ? p->ginvpow : p->gpow, *sct = bwd ? p->gpow : p->ginvpow;
        double dcr = 0.0, dci = 0.0;
        int n;
        a.x0r = zin[0];
        a.x0i = zin[1];
        for (n = 0; n < N; n++)
        {
            dcr += zin[2 * n];
            dci += zin[2 * n + 1];
        }
        a.k = bwd ? p->omb : p->omf;
        a.perm = gat; a.pass = 3; _ilpr_mt_pass(T, &a, (size_t)nm1);
        _ilpr_mt_inner(p, p->za, p->zb, 0);
        a.pass = 1; _ilpr_mt_pass(T, &a, (size_t)nm1);
        _ilpr_mt_inner(p, p->zb, p->za, 1);
        a.perm = sct; a.pass = 4; _ilpr_mt_pass(T, &a, (size_t)nm1);
        zout[0] = dcr;
        zout[1] = dci;
    }
    _vfft_ilpr_mt_count++;
    return 1;
}

/* ── the race at T: serial vs BLOCKS vs TILES on the whole convolution ───
 * (ZTURN-T's protocol, ztt_mt.h: reps to ~20 ms of serial work per sample,
 * min of 3, 2 warm passes, unpaced -- threaded arms are measured hot). Out of
 * place on scratch; the convolution is alias-safe, so one verdict serves both
 * placements. Leaves the plan bound to the winner for T; returns mt (0 when
 * the plan has no threaded form). The caller arms the pool. */
typedef struct { vfft_ilprime_plan_t *p; const double *zi; double *zo; int mt; } _ilpr_mt_ctx_t;
static void _ilpr_mt_arm_run(void *v)
{
    _ilpr_mt_ctx_t *c = (_ilpr_mt_ctx_t *)v;
    c->p->mt = c->mt;
    c->p->inner.pt->mt = c->mt;
    if (c->mt == 0 || !vfft_ilprime_execute_mt(c->p, c->zi, c->zo, 0))
        vfft_ilprime_execute_fwd(c->p, c->zi, c->zo);
}
static inline int vfft_ilprime_mt_race(vfft_ilprime_plan_t *p, int T, double *ns_out)
{
    static const char *names[3] = { "serial", "blocks", "tiles" };
    _ilpr_mt_ctx_t cx[3];
    vfft_race_arm_t arms[3];
    double ns[3];
    const size_t nb = (size_t)2 * p->N * sizeof(double);
    double *zi, *zo, t0;
    int na = 0, a, best = 0, reps;
    size_t i;
    if (T < 2 || !p->inner.pt || !vfft_ilprime_mt_bind(p, T, 1))
    {
        vfft_ilprime_mt_bind(p, T, 0);
        return 0;
    }
    zi = (double *)vfft_aligned_alloc(nb);
    zo = (double *)vfft_aligned_alloc(nb);
    if (!zi || !zo)
    {
        vfft_aligned_free(zi);
        vfft_aligned_free(zo);
        vfft_ilprime_mt_bind(p, T, 0);
        return 0;
    }
    for (i = 0; i < 2 * (size_t)p->N; i++)
        zi[i] = 1.0 + 1e-6 * (double)(i & 1023);
    p->mt = 0;
    p->inner.pt->mt = 0;
    vfft_ilprime_execute_fwd(p, zi, zo);
    t0 = vfft_now_ns();
    vfft_ilprime_execute_fwd(p, zi, zo);
    t0 = vfft_now_ns() - t0;
    reps = (int)(20e6 / (t0 > 1.0 ? t0 : 1.0));
    if (reps < 2) reps = 2;
    if (reps > (1 << 19)) reps = 1 << 19;
    for (a = 0; a < 3; a++)
    {
        if (a == 2 && (p->inner.pt->tile == 0 || (size_t)p->inner.pt->N / p->inner.pt->tile < 2))
            break;
        cx[na].p = p; cx[na].zi = zi; cx[na].zo = zo; cx[na].mt = a;
        arms[na].name = names[a]; arms[na].run = _ilpr_mt_arm_run; arms[na].ctx = &cx[na];
        na++;
    }
    {
        const vfft_race_proto_t proto = { 3, reps, VFFT_RACE_MIN, 1, 2, NULL, NULL, 0 }; /* THREADED arms: never paused (VFFT_RACE_PACE_MS) */
        vfft_race_run(&proto, arms, na, ns);
    }
    for (a = 1; a < na; a++)
        if (ns[a] < ns[best]) best = a;
    vfft_aligned_free(zi);
    vfft_aligned_free(zo);
    vfft_ilprime_mt_bind(p, T, cx[best].mt);
    if (ns_out) *ns_out = ns[best];
    if (getenv("VFFT_NAT_LOG"))
    {
        fprintf(stderr, "[ilpr-mt] N=%d %s M=%d T=%d race:", p->N, p->method ? "RADER" : "BLUESTEIN", p->M, T);
        for (a = 0; a < na; a++) fprintf(stderr, " %s=%.0f", names[a], ns[a]);
        fprintf(stderr, " -> mt=%d\n", p->mt);
    }
    return p->mt;
}

#endif /* VFFT_IL_PRIME_MT_H */
