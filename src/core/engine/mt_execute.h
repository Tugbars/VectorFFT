/* mt_execute.h - the generic K-split multithreaded executor.
 *
 * A batch of K transforms is K independent problems, so the pool cuts [0,K)
 * into slabs and runs the same plan on each, after a create-time check that
 * the plan survives the split.
 *
 * "Each lane is an independent transform" is true of the MATH and not
 * automatically true of the KERNELS. Two codelet families bake assumptions
 * about the whole batch into their code:
 *
 *   (a) radix-8 LOG3 last-stage - its twiddle blocking bakes the full K, so a
 *       partial batch is wrong for ANY input, including symmetric ones;
 *   (b) DIF chains - wrong for a partial batch on ASYMMETRIC input only.
 *
 * A poorly mixed probe MASKS (b), so the self-check probe is well-mixed, and it
 * replays EVERY slab size the executor can pick rather than sampling one.
 *
 * _c2c_mt_safe is a correctness gate, not a race: a deterministic, sequential
 * replay, no clock, nothing banked. A plan that fails it runs whole-batch
 * under MT (the reorder pass still threads).
 *
 * Takes the proto plan by pointer, never a vfft_plan_s, so it does not depend
 * on the front door's opaque types. No mutable file-scope state, no wisdom.
 */
#ifndef VFFT_ENGINE_MT_EXECUTE_H
#define VFFT_ENGINE_MT_EXECUTE_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "executor.h"          /* stride_plan_t, vfft_proto_exec_fn, execute_fwd/bwd */
#include "support/threads.h"   /* the pool: dispatch, wait_all, get_num_threads */

/* ════════════════════════════════════════════════════════════════════════
 * MT EXECUTE — pool K-split over the in-place executor
 * ════════════════════════════════════════════════════════════════════════ */
typedef struct
{
    const stride_plan_t *p;
    vfft_proto_exec_fn fn; /* resolved executor for this direction (NULL = generic) */
    double *re, *im;
    size_t k0, S;
    int dir;
} _ip_arg;
static void _ip_tramp(void *a)
{
    _ip_arg *x = (_ip_arg *)a;
    if (x->fn)
        x->fn(x->p, x->re + x->k0, x->im + x->k0, x->S, x->p->K, 0);
    else if (x->dir)
        vfft_proto_execute_fwd(x->p, x->re + x->k0, x->im + x->k0, x->S);
    else
        vfft_proto_execute_bwd(x->p, x->re + x->k0, x->im + x->k0, x->S);
}
/* SLAB-SPLIT self-check: does the plan reproduce the WHOLE-batch result when
 * run as _c2c_mt's per-slab partial batches (see the header for the two
 * codelet families that break this)? The failures are structural, not
 * concurrency, so a sequential replay reproduces them. Replays EVERY slab
 * size _c2c_mt can pick (S = 8,16,..,K: S = ceil(K/T) rounded to 8 for some
 * T, boundaries k0 = t*S exactly) on a well-mixed xorshift input and compares
 * to the whole. One-time at create. Returns 1 = safe (K-split OK),
 * 0 = unsafe (whole-batch). */
static int _c2c_mt_safe(const stride_plan_t *p, vfft_proto_exec_fn fn)
{
    size_t K = p->K;
    if (K < 16)
        return 1; /* _c2c_mt runs ST for K<8; K<16 never splits into >=2 slabs of 8 */
    size_t tot = (size_t)p->N * K;
    double *xr = (double *)malloc(tot * 8), *xi = (double *)malloc(tot * 8);
    double *ar = (double *)malloc(tot * 8), *ai = (double *)malloc(tot * 8);
    double *br = (double *)malloc(tot * 8), *bi = (double *)malloc(tot * 8);
    if (!xr || !xi || !ar || !ai || !br || !bi)
    {
        free(xr);
        free(xi);
        free(ar);
        free(ai);
        free(br);
        free(bi);
        return 1;
    }
    unsigned long long st = 0x243F6A8885A308D3ULL; /* xorshift64: well-mixed, non-periodic -> exposes (b) */
    for (size_t i = 0; i < tot; i++)
    {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        xr[i] = (double)(st >> 40) / 16777216.0 - 0.5;
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        xi[i] = (double)(st >> 40) / 16777216.0 - 0.5;
    }
    memcpy(ar, xr, tot * 8);
    memcpy(ai, xi, tot * 8);
    if (fn)
        fn(p, ar, ai, K, p->K, 0);
    else
        vfft_proto_execute_fwd(p, ar, ai, K); /* whole-batch reference */
    int unsafe = 0;
    for (size_t S = 8; S <= K && !unsafe; S += 8)
    { /* every slab size _c2c_mt can choose */
        memcpy(br, xr, tot * 8);
        memcpy(bi, xi, tot * 8);
        for (size_t k0 = 0; k0 < K; k0 += S)
        { /* _c2c_mt's exact slab boundaries, replayed sequentially */
            size_t me = (k0 + S > K) ? K - k0 : S;
            if (fn)
                fn(p, br + k0, bi + k0, me, p->K, 0);
            else
                vfft_proto_execute_fwd(p, br + k0, bi + k0, me);
        }
        for (size_t i = 0; i < tot; i++)
            if (fabs(ar[i] - br[i]) + fabs(ai[i] - bi[i]) > 1e-9)
            {
                unsafe = 1;
                break;
            }
    }
    free(xr);
    free(xi);
    free(ar);
    free(ai);
    free(br);
    free(bi);
    return !unsafe;
}
/* In-place c2c, pool K-split. `fn` is the transparent JIT/baked-resolved executor
 * for `dir` (NULL = fall back to the generic executor) — set once at create. */
/* `me` = number of batch lanes to process (tight: p->K ; padded: exec_me = Kp pad / K tail).
 * The pool splits [0,me) into VW-aligned blocks run at the plan's baked stride p->K. For a
 * padded (Kp-wide) buffer with me=Kp, blocks are 4-aligned so the (Kp-K) zero pad lanes ride
 * in the last block full-SIMD (no per-block tail); with me=K the last block carries the tail. */
static void _c2c_mt(const stride_plan_t *p, double *re, double *im, int dir,
                    vfft_proto_exec_fn fn, size_t me)
{
    size_t K = me;
    /* The pool owns the clamp (support/threads.h). This helper has no plan
     * handle, so no snapshot is passed: the caller's plan decided whether to
     * come here at all (see _c2c_mt_safe / the create-time engage decision). */
    int T = stride_pool_workers_for(0);
    if (T <= 1 || K < 8)
    {
        if (fn)
            fn(p, re, im, K, p->K, 0);
        else if (dir)
            vfft_proto_execute_fwd(p, re, im, K);
        else
            vfft_proto_execute_bwd(p, re, im, K);
        return;
    }
    /* The engine's own part: the slicing. CEIL(K/T) rounded up to 8, so no
     * tail lanes are dropped (floor would lose K%T lanes, e.g. T=8, K=65).
     * Slot 0 is the caller's slice by the pool's convention. */
    size_t S = (((K + (size_t)T - 1) / (size_t)T) + 7) & ~(size_t)7;
    _ip_arg a[STRIDE_POOL_MAX_DISPATCH];
    int n = 0;
    for (int t = 0; t < T; t++)
    {
        size_t k0 = (size_t)t * S;
        if (k0 >= K)
            break;
        size_t ke = k0 + S;
        if (ke > K)
            ke = K;
        a[n++] = (_ip_arg){p, fn, re, im, k0, ke - k0, dir};
    }
    stride_pool_run(n, _ip_tramp, a, sizeof a[0]);
}

#endif /* VFFT_ENGINE_MT_EXECUTE_H */
