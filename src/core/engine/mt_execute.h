/* mt_execute.h - the generic K-split multithreaded executor.
 *
 * A batch of K transforms is K independent problems, so the pool can simply cut
 * [0,K) into slabs and run the same plan on each. That is the whole idea; this
 * header is that idea plus the safety proof it requires. Extracted from vfft.c
 * as migration step 7; see docs/design/refactor_migration_plan.md.
 *
 * WHY THE SPLIT NEEDS A PROOF, NOT AN ASSUMPTION
 * ----------------------------------------------
 * "Each lane is an independent transform" is true of the MATH and not
 * automatically true of the KERNELS. Two codelet families bake assumptions
 * about the whole batch into their code:
 *
 *   (a) radix-8 LOG3 last-stage - its twiddle blocking bakes the full K, so a
 *       partial batch is wrong for ANY input, including symmetric ones;
 *   (b) DIF chains - wrong for a partial batch on ASYMMETRIC input only.
 *
 * (b) is the dangerous one, because a poorly-mixed probe MASKS it: an early
 * low-bit index hash passed while random input failed at 1.2. So the self-check
 * probe must be well-mixed, and it replays EVERY slab size the executor can
 * pick rather than sampling one.
 *
 * THIS IS A CORRECTNESS GATE, NOT A RACE
 * --------------------------------------
 * _c2c_mt_safe answers "does this plan reproduce the whole-batch result when
 * run on partial batches?" - a deterministic, sequential replay. No clock is
 * involved and nothing is banked. It belongs in no catalogue of measurement
 * arms; a plan that fails it runs whole-batch under MT (the reorder pass still
 * threads), it does not run slower-but-correct.
 *
 * FLOOR-LEGAL BY CONSTRUCTION
 * ---------------------------
 * Takes the proto plan by pointer and never a vfft_plan_s, so it carries no
 * dependency on the front door's opaque types. No mutable file-scope state, no
 * wisdom. In particular it does NOT pull engine/stride_executor.h - that header
 * redefines executor symbols and is excluded from the build by design.
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
/* In-place c2c, pool K-split. `fn` is the transparent JIT/baked-resolved executor
 * for `dir` (NULL = fall back to the generic executor) — set once at create. */
/* `me` = number of batch lanes to process (tight: p->K ; padded: exec_me = Kp pad / K tail).
 * The pool splits [0,me) into VW-aligned blocks run at the plan's baked stride p->K. For a
 * padded (Kp-wide) buffer with me=Kp, blocks are 4-aligned so the (Kp-K) zero pad lanes ride
 * in the last block full-SIMD (no per-block tail); with me=K the last block carries the tail. */
/* SLAB-SPLIT self-check: does the plan reproduce the WHOLE-batch result when run as _c2c_mt's per-slab
 * partial batches? Each lane is an INDEPENDENT transform, so splitting [0,K) into slabs [k0,k0+me) and
 * running fn(me) on each MUST equal fn(K) on the whole. Two codelet families break this, both structural
 * (NOT concurrency — a SEQUENTIAL replay reproduces them; deterministic given the input):
 *   (a) radix-8 LOG3 last-stage — its twiddle blocking bakes the full K, so any me<K is wrong (visible on
 *       ANY input, incl. symmetric);
 *   (b) DIF chains (use_dif=1) — wrong for a partial batch on ASYMMETRIC input (a symmetric/periodic probe
 *       like a low-bit index hash MASKS it — that is exactly why an earlier det-input check passed 4·32 DIF
 *       while rand failed 1.2). So the probe MUST be well-mixed (xorshift, non-periodic).
 * We replay EVERY slab size _c2c_mt can pick (S = 8,16,..,K — S = ceil(K/T) rounded to 8 for some T, its
 * slab boundaries k0 = t*S exactly) and compare to the whole. Unsafe if ANY differs -> _c2c_mt runs the
 * plan WHOLE-batch under MT (the reorder pass still threads). Lock-free, one-time at create. Returns
 * 1 = safe (K-split OK), 0 = unsafe (whole-batch). */
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
static void _c2c_mt(const stride_plan_t *p, double *re, double *im, int dir,
                    vfft_proto_exec_fn fn, size_t me)
{
    size_t K = me;
    int T = stride_get_num_threads();
    if (T > _stride_pool_size + 1)
        T = _stride_pool_size + 1;
    if (T > 64)
        T = 64; /* a[64] MT arg-array bound: cap dispatched workers to a[..<64] (EPYC-port hardening;
                 * the i9 pool is well below 64, so this is a no-op there). */
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
    size_t S = (((K + (size_t)T - 1) / (size_t)T) + 7) & ~(size_t)7; /* CEIL(K/T) then round to 8: floor dropped the last K%T lanes when floor(K/T)%8==0 (e.g. T=8,K=65) */
    _ip_arg a[64];
    int nd = 0;
    for (int t = 1; t < T && t <= _stride_pool_size; t++)
    {
        size_t k0 = (size_t)t * S;
        if (k0 >= K)
            break;
        size_t ke = k0 + S;
        if (ke > K)
            ke = K;
        a[nd] = (_ip_arg){p, fn, re, im, k0, ke - k0, dir};
        _stride_pool_dispatch(&_stride_workers[nd], _ip_tramp, &a[nd]);
        nd++;
    }
    size_t s0 = S < K ? S : K;
    if (fn)
        fn(p, re, im, s0, p->K, 0);
    else if (dir)
        vfft_proto_execute_fwd(p, re, im, s0);
    else
        vfft_proto_execute_bwd(p, re, im, s0);
    if (nd)
        _stride_pool_wait_all();
}

#endif /* VFFT_ENGINE_MT_EXECUTE_H */
