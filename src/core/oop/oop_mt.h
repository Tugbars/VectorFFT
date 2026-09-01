/* oop_mt.h - out-of-place c2c, multithreaded by lane slice.
 *
 * A [k0, k0+S) slice of the batch, run independently by each worker. Extracted
 * from vfft.c as migration step 9; see docs/design/refactor_migration_plan.md.
 *
 * WHY ONLY TWO OF THE THREE KINDS THREAD
 * --------------------------------------
 * K-split is exact only when a lane slice is independent END TO END:
 *
 *   LEAF   - one codelet per lane. Independent.
 *   MODEB  - in-place dataflow run into the destination. Independent.
 *   BAILEY2 - NOT independent: its s1->s2 transpose reads ACROSS the R1
 *             n1-blocks, so a lane slice is not a closed unit of work.
 *
 * BAILEY2 therefore runs single-threaded here, and that is a correctness
 * requirement rather than a tuning choice - a lane-split BAILEY2 would read
 * data another worker has not written yet. Threading it properly needs a
 * barrier on a different split axis, which is a separate piece of work, not a
 * flag on this one.
 *
 * T <= 1 also runs whole-batch, and an odd K rides the last slab's tail (the
 * codelet-internal rem-aware path), so no lane is ever dropped.
 *
 * THE CALLER MUST PIN TO CORE 0
 * -----------------------------
 * Same contract as the generic K-split executor: the pool's workers spin
 * rather than sleep, so an unpinned caller competes with its own pool for the
 * core it is dispatching from.
 *
 * FLOOR-LEGAL BY CONSTRUCTION
 * ---------------------------
 * Takes the OOP plan by pointer and never a vfft_plan_s. No mutable file-scope
 * state, no wisdom. Does NOT pull engine/stride_executor.h.
 */
#ifndef VFFT_OOP_OOP_MT_H
#define VFFT_OOP_OOP_MT_H

#include <stdlib.h>

#include "oop_auto.h"          /* vfft_oop_plan_t and the kind enum */
#include "support/threads.h"   /* the pool: dispatch, wait_all, size, get_num_threads */

/* ── OOP c2c multithreading (pool K-split). A lane-slice [k0,k0+S) is executed
 * independently by each worker. LEAF (one codelet) and MODEB (in-place dataflow on
 * the dst) are lane-independent END-TO-END, so K-split is exact. BAILEY2 is NOT: its
 * s1->s2 transpose reads across the R1 n1-blocks, so a lane-slice isn't independent —
 * it stays single-thread (proper MT needs a barrier on a different split dim). K<8 and
 * T<=1 also run whole-batch. Odd K rides the last slab's tail (the codelet is rem-aware).
 * GOTCHA (as with _c2c_mt): the CALLER must pin to core 0 — workers pin 1..T-1. ── */
static void _oop_slice_fwd(const vfft_oop_plan_t *p, const double *sr, const double *si,
                           double *dr, double *di, size_t k0, size_t S)
{
    size_t K = p->K;
    if (p->kind == VFFT_OOP_KIND_LEAF)
        p->leaf(sr + k0, si + k0, dr + k0, di + k0, 0, 0, K, 1, K, 1, S);
    else /* MODEB: OOP inner on the dst slice (JIT if resolved, else generic) */
        vfft_proto_execute_fwd_oop_jit(p->mb, sr + k0, si + k0, dr + k0, di + k0, S, p->mb_jit_fwd);
}
static void _oop_slice_bwd(const vfft_oop_plan_t *p, const double *sr, const double *si,
                           double *dr, double *di, size_t k0, size_t S)
{
    size_t K = p->K;
    if (p->kind == VFFT_OOP_KIND_MODEB)
    {
        /* copy the slice's spectrum lanes to dst, then DIF-bwd in place on the slice. */
        for (int e = 0; e < p->N; e++)
        {
            memcpy(dr + (size_t)e * K + k0, sr + (size_t)e * K + k0, S * sizeof(double));
            memcpy(di + (size_t)e * K + k0, si + (size_t)e * K + k0, S * sizeof(double));
        }
        if (p->mb_jit_bwd)
            p->mb_jit_bwd(p->mb, dr + k0, di + k0, S, p->mb->K, 0);
        else
            vfft_proto_execute_bwd_generic(p->mb, dr + k0, di + k0, S);
    }
    else /* LEAF: natural-order swap identity — bwd = fwd with re/im swapped */
        p->leaf(si + k0, sr + k0, di + k0, dr + k0, 0, 0, K, 1, K, 1, S);
}
typedef struct
{
    const vfft_oop_plan_t *p;
    const double *sr, *si;
    double *dr, *di;
    size_t k0, S;
    int dir;
} _oop_mt_arg_t;
static void _oop_mt_tramp(void *a)
{
    _oop_mt_arg_t *x = (_oop_mt_arg_t *)a;
    if (x->dir)
        _oop_slice_fwd(x->p, x->sr, x->si, x->dr, x->di, x->k0, x->S);
    else
        _oop_slice_bwd(x->p, x->sr, x->si, x->dr, x->di, x->k0, x->S);
}
static void _oop_mt(const vfft_oop_plan_t *p, const double *sr, const double *si,
                    double *dr, double *di, int dir)
{
    size_t K = p->K;
    /* The pool owns the clamp (support/threads.h); the OOP plan carries no
     * thread snapshot of its own, so none is passed. */
    int T = stride_pool_workers_for(0);
    if (T <= 1 || K < 8 || p->kind == VFFT_OOP_KIND_BAILEY2)
    {
        if (dir)
            vfft_oop_execute_fwd(p, sr, si, dr, di);
        else
            vfft_oop_execute_bwd(p, sr, si, dr, di);
        return;
    }
    /* THE ENGINE'S OWN PART: the slicing. CEIL(K/T) then round to 8: floor
     * dropped the last K%T lanes when floor(K/T)%8==0 (e.g. T=8,K=65). Slot 0
     * is the caller's slice by the pool's convention. */
    size_t S = (((K + (size_t)T - 1) / (size_t)T) + 7) & ~(size_t)7;
    _oop_mt_arg_t a[STRIDE_POOL_MAX_DISPATCH];
    int n = 0;
    for (int t = 0; t < T; t++)
    {
        size_t k0 = (size_t)t * S;
        if (k0 >= K)
            break;
        size_t ke = k0 + S;
        if (ke > K)
            ke = K;
        a[n++] = (_oop_mt_arg_t){p, sr, si, dr, di, k0, ke - k0, dir};
    }
    stride_pool_run(n, _oop_mt_tramp, a, sizeof a[0]);
}

#endif /* VFFT_OOP_OOP_MT_H */
