/* natorder_mt.h - the natural-order reorder passes, multithreaded.
 *
 * Two groups, both extracted from vfft.c as migration step 8; see
 * docs/design/refactor_migration_plan.md.
 *
 *   (1) the CYCLE/PAIR reorder pass - what turns a scrambled spectrum into a
 *       natural one after a forward, or unwinds it before a backward.
 *   (2) the SCR forward, which reaches natural order by a different route
 *       entirely: an out-of-place scratch fill followed by a terminator.
 *
 * WHY THE TWO SPLIT DIFFERENTLY, AND WHY THAT IS NOT ARBITRARY
 * -----------------------------------------------------------
 * The reorder pass splits by CYCLE/PAIR RANGE, never by K. A permutation
 * decomposes into disjoint cycles, so handing each worker its own cycles gives
 * disjoint row sets and needs no barrier. Splitting by K instead would make
 * 64-byte sub-rows - the measured catastrophic regime for this access pattern.
 *
 * SCR has two DEPENDENT phases and therefore a barrier between them:
 *   phase 1, the scratch fill, is K-split across lanes exactly like the generic
 *            executor, because lanes are independent there;
 *   phase 2, the terminator, is GROUP-split and never K-split - each worker
 *            reads disjoint scratch and writes a disjoint output comb, which is
 *            what makes it race-free.
 * Same transform, opposite split axes, for reasons specific to each phase.
 *
 * WHAT STAYED BEHIND
 * ------------------
 * _natorder_mt, the five-line adapter that unpacks vfft_plan_s and calls
 * _natorder_reorder_mt, is still in vfft.c: it dereferences the plan struct, so
 * it waits for step 15. The division is the right one regardless - this header
 * owns the algorithm, the front door owns the plan-to-arguments adaptation.
 *
 * FLOOR-LEGAL BY CONSTRUCTION
 * ---------------------------
 * Every function here takes its inputs explicitly. No vfft_plan_s, no mutable
 * file-scope state, no wisdom. It does NOT pull engine/stride_executor.h.
 *
 * THE WORKER COUNT IS A PARAMETER, NOT A GLOBAL READ (2026-09-01)
 * ---------------------------------------------------------------
 * _natorder_reorder_mt takes `nthreads` -- the plan's create-time snapshot
 * (h->nthreads), the number of per-worker scratch slots `tmp` was sized for --
 * and clamps its dispatched worker count by it. It used to read only the live
 * pool, which is grow-only and can therefore be larger at execute than the
 * scratch allocated at create: T workers sliced a smaller buffer. The result was
 * a wrong answer on one run and heap corruption at destroy on the next
 * (natorder_scratch_gate). Every engine in this tree clamps by its own
 * snapshot; this one now does too.
 */
#ifndef VFFT_TRANSFORMS_NATORDER_NATORDER_MT_H
#define VFFT_TRANSFORMS_NATORDER_NATORDER_MT_H

#include <stdlib.h>
#include <string.h>

#include "natorder_exec.h"     /* vfft_natorder_cycle_range / _pair_range */
#include "natorder_scatter.h"  /* natorder_scr_t, natorder_scr_fwd, term_range */
#include "oop_execute.h"       /* vfft_proto_execute_fwd_oop_jit (SCR phase 1) */
#include "support/threads.h"   /* the pool: dispatch, wait_all, get_num_threads */

/* ── ORDER_NATURAL reorder pass, MT by CYCLE/PAIR ranges (full K-wide rows — NEVER K-split;
 * K-split makes 64B sub-rows, the measured catastrophic regime). Runs AFTER the forward FFT
 * (dir!=0) or BEFORE the backward (dir==0, inverse shift). Each worker owns a disjoint set of
 * cycles/pairs + its own 2K temp slot; disjoint row sets => race-free. natural_order §2e. */
typedef struct
{
    double *re, *im, *tmp;
    const int *list, *cyc_off;
    size_t K;
    int c0, c1, slot, inv, is_pairs;
} _nat_arg;
static void _nat_range_tramp(void *a)
{
    _nat_arg *x = (_nat_arg *)a;
    if (x->is_pairs)
        vfft_natorder_pair_range(x->re, x->im, x->K, x->list, x->c0, x->c1);
    else
        vfft_natorder_cycle_range(x->re, x->im, x->K, x->list, x->cyc_off,
                                  x->c0, x->c1, x->tmp + (size_t)x->slot * 2 * x->K, x->inv);
}
/* MT split of a whole-row reorder (N rows x K lanes) by unit COUNT (cycles or pairs). Each worker owns a
 * disjoint unit range + its OWN 2K temp slot (tmp = (pool+1) slots) => disjoint row sets, race-free.
 * SHARED by the 1D natorder pass and the 2D dim1 (whole-row) pass — same shape — so the 2D dim1 reorder
 * is no longer single-threaded (it was the whole ~1.2-1.6x tax on one core at 256^2/512^2). inv: 1 =
 * inverse cycle (backward), 0 = forward; ignored for a self-inverse pair tape. */
static void _natorder_reorder_mt(double *re, double *im, size_t N, size_t K,
                                 const int *list, const int *cyc_off, int nunits,
                                 int is_pairs, double *tmp, int inv, int nthreads)
{
    int T = stride_get_num_threads();
    if (T > _stride_pool_size + 1)
        T = _stride_pool_size + 1;
    /* THE PLAN'S OWN SNAPSHOT IS THE CEILING. `tmp` was sized at create for
     * exactly `nthreads` per-worker slots (the plan's h->nthreads). The pool is
     * grow-only, so the live count read above can EXCEED that later, and every
     * worker slices `tmp + slot*2*K` -- reading the live pool alone therefore
     * indexed past the buffer (natorder_scratch_gate: wrong output on one run,
     * heap corruption at destroy on the next). Same rule as every other engine:
     * clamp by the plan-time snapshot, never by the live pool alone. */
    if (nthreads >= 1 && T > nthreads)
        T = nthreads;
    if (T > 64)
        T = 64; /* a[64] MT arg-array bound: cap dispatched workers to a[..<64] (EPYC-port hardening;
                 * the i9 pool is well below 64, so this is a no-op there). */
    if (T <= 1 || nunits < T || N * K < 8192)
    {
        if (is_pairs)
            vfft_natorder_pair_range(re, im, K, list, 0, nunits);
        else
            vfft_natorder_cycle_range(re, im, K, list, cyc_off, 0, nunits, tmp, inv);
        return;
    }
    int per = (nunits + T - 1) / T; /* count-balanced (pairs exact; cycles approx) */
    _nat_arg a[64];
    int nd = 0, c = per;
    for (int t = 1; t < T && t <= _stride_pool_size; t++)
    {
        if (c >= nunits)
            break;
        int c1 = c + per;
        if (c1 > nunits)
            c1 = nunits;
        a[nd] = (_nat_arg){re, im, tmp, list, cyc_off, K, c, c1, nd, inv, is_pairs};
        _stride_pool_dispatch(&_stride_workers[nd], _nat_range_tramp, &a[nd]);
        nd++;
        c = c1;
    }
    int m1 = per < nunits ? per : nunits; /* main thread does [0,per) */
    if (is_pairs)
        vfft_natorder_pair_range(re, im, K, list, 0, m1);
    else
        vfft_natorder_cycle_range(re, im, K, list, cyc_off, 0, m1, tmp + (size_t)nd * 2 * K, inv);
    if (nd)
        _stride_pool_wait_all();
}

/* ── SCR forward, MT. Two dependent phases with a barrier between:
 *   (1) OOP scratch-fill user->scratch (execute_fwd_oop; NOT the OOP MODEB kind — just its
 *       stage-0-redirect technique): K-split across lanes (each lane an independent transform,
 *       exactly like _c2c_mt); odd tail rides the last slab's rem-aware codelets.
 *   (2) terminator scratch->user: GROUP(q)-split (never K-split — full K-wide scattered rows);
 *       disjoint scratch reads + disjoint output combs => race-free. Each worker pre-twiddles only
 *       its own groups' scratch. Caller pins core 0 (workers 1..T-1). ── */
typedef struct
{
    natorder_scr_t *s;
    double *ur, *ui;
    size_t k0, S;
} _scr_modeb_arg;
static void _scr_modeb_tramp(void *a)
{
    _scr_modeb_arg *x = (_scr_modeb_arg *)a;
    vfft_proto_execute_fwd_oop_jit(&x->s->sub, x->ur + x->k0, x->ui + x->k0,
                                   x->s->scr_re + x->k0, x->s->scr_im + x->k0, x->S,
                                   x->s->sub_jit_fwd);
}
typedef struct
{
    natorder_scr_t *s;
    double *ur, *ui;
    int q0, q1;
} _scr_term_arg;
static void _scr_term_tramp(void *a)
{
    _scr_term_arg *x = (_scr_term_arg *)a;
    natorder_scr_term_range(x->s, x->ur, x->ui, x->q0, x->q1);
}
static void _scr_fwd_mt(natorder_scr_t *s, double *ur, double *ui, size_t K)
{
    int T = stride_get_num_threads();
    if (T > _stride_pool_size + 1)
        T = _stride_pool_size + 1;
    if (T > 64)
        T = 64; /* a[64] MT arg-array bound: cap dispatched workers to a[..<64] (EPYC-port hardening;
                 * the i9 pool is well below 64, so this is a no-op there). */
    if (T <= 1 || K < 8 || (size_t)s->N * K < 8192)
    {
        natorder_scr_fwd(s, ur, ui, K);
        return;
    }
    /* phase 1: OOP scratch-fill, K-split (lanes) */
    size_t Sv = (((K + (size_t)T - 1) / (size_t)T) + 7) & ~(size_t)7; /* CEIL(K/T) then round to 8 (floor dropped last K%T lanes when floor(K/T)%8==0) */
    _scr_modeb_arg a1[64];
    int nd = 0;
    for (int t = 1; t < T && t <= _stride_pool_size; t++)
    {
        size_t k0 = (size_t)t * Sv;
        if (k0 >= K)
            break;
        size_t ke = k0 + Sv;
        if (ke > K)
            ke = K;
        a1[nd] = (_scr_modeb_arg){s, ur, ui, k0, ke - k0};
        _stride_pool_dispatch(&_stride_workers[nd], _scr_modeb_tramp, &a1[nd]);
        nd++;
    }
    {
        size_t s0 = Sv < K ? Sv : K;
        vfft_proto_execute_fwd_oop_jit(&s->sub, ur, ui, s->scr_re, s->scr_im, s0,
                                       s->sub_jit_fwd); /* B6: main slice on JIT too (was generic ->
                                       straggler at the phase-1 barrier); matches workers + ST path. */
    }
    if (nd)
        _stride_pool_wait_all(); /* BARRIER: scratch complete */
    /* phase 2: terminator, group(q)-split */
    int P = s->P, per = (P + T - 1) / T;
    _scr_term_arg a2[64];
    int nd2 = 0;
    for (int t = 1; t < T && t <= _stride_pool_size; t++)
    {
        int q0 = t * per;
        if (q0 >= P)
            break;
        int q1 = q0 + per;
        if (q1 > P)
            q1 = P;
        a2[nd2] = (_scr_term_arg){s, ur, ui, q0, q1};
        _stride_pool_dispatch(&_stride_workers[nd2], _scr_term_tramp, &a2[nd2]);
        nd2++;
    }
    natorder_scr_term_range(s, ur, ui, 0, per < P ? per : P);
    if (nd2)
        _stride_pool_wait_all();
}

#endif /* VFFT_TRANSFORMS_NATORDER_NATORDER_MT_H */
