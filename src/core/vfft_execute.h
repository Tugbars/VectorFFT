/* vfft_execute.h - THE execute entry point.
 *
 * vfft_execute and the transform-contiguous batch MT it dispatches through.
 * Extracted from vfft.c as migration step 16; see
 * docs/design/refactor_migration_plan.md. Step 28 extends this header.
 *
 * SCOPE: EVERYTHING - every transform, both layouts
 * ------------------------------------------------
 * The single dispatcher behind the public API. It validates the call
 * (_vfft_sig_bad), tries the bound K=1 interleaved dispatch (k1_exec), sends
 * 1D real to bridge/real_bridge_exec.h (the D1 crossing, temporary), and then
 * forks ONCE on the committed layout (layout separation, phase 7.2):
 *
 *   INTERLEAVED   il/il_execute.h       _vfft_il_execute
 *   SPLIT         split/split_execute.h _vfft_split_execute
 *
 * The two families share no codelets, no planner and no executor (see
 * docs/design/planning_model.md, Parts II and III); this header is the one
 * place they meet, and it tests the layout exactly once.
 *
 * ONE TRANSLATION UNIT MUST OWN THE DEFINITION
 * --------------------------------------------
 * Unlike every other module header in this tree, this one carries a function
 * with EXTERNAL linkage - vfft_execute is the public entry point. A header that
 * defines a non-static function is a duplicate symbol waiting for a second
 * includer, and step 21 is expected to give the four bench TUs that currently
 * #include "vfft.c" an alternative spelling, so a second includer is a matter
 * of when rather than whether.
 *
 * So the body is guarded: define VFFT_EXECUTE_IMPL before including, in exactly
 * one TU. vfft.c does. Anyone else including this header gets nothing, which is
 * correct - the DECLARATION they need already lives in the public vfft.h, and
 * what is here is the implementation.
 *
 * WHY THE INCLUDE SITS WHERE IT DOES IN vfft.c
 * -------------------------------------------
 * The execute sides call helpers that must be declared before them:
 * _pq_execute (il/rank2/plane_queue.h, which vfft.c includes just above this
 * header) and the execute-side statics of vfft.c. The include therefore
 * replaces the definition in place rather than moving to the top of the file;
 * its position is load-bearing. The transform-contiguous MT trampoline
 * (_tc_mt_arg / _tc_mt_tramp) lives with its only caller, in il/il_execute.h.
 *
 * (The 2048-point scalar engage floor that once lived here is retired:
 * the wrapper now carries a raced, banked verdict, h->tc_mt - see
 * _tc_mt_decide in vfft.c.)
 */
#ifndef VFFT_EXECUTE_H
#define VFFT_EXECUTE_H

#include "vfft_internal.h" /* struct vfft_plan_s - the dispatch reads it */

#ifdef VFFT_EXECUTE_IMPL

/* ---- execute-side helpers ----
 * The front door's own helper (_vfft_sig_bad) is below; each layout's
 * helpers live with its execute side (il/il_execute.h, split/split_execute.h).
 * _pq_execute stays above this header's include point, in
 * il/rank2/plane_queue.h: the plane queue's create calls it too. */

/* ── EXECUTE-SIDE SIGNATURE ENFORCEMENT ──
 * The pointer pattern must MATCH the plan's committed layout; the historical
 * NULL-pointer inference ("sim==dim==NULL means interleaved") is REMOVED.
 * Returns 1 (and prints an actionable stderr line) when the call must be
 * REFUSED — the caller returns without computing ANYTHING, so a mismatch can
 * never silently reinterpret buffers or produce garbage. */
static int _vfft_sig_bad(struct vfft_plan_s *h, vfft_dir_t dir, double *sre,
                         double *sim, double *dre, double *dim)
{
    const int il = (h->layout == (int)VFI_LAYOUT_IL);
    const char *tn = _vfft_tname(h->transform);
    if (_VFFT_IS_TRIG(h->transform))
    {
        if (!sre || !dre)
        {
            _vfft_warn("vfft_execute: %s needs in=real_in and out=real_out non-NULL "
                       "(got in=%s, out=%s) — nothing executed",
                       tn, sre ? "ok" : "NULL", dre ? "ok" : "NULL");
            return 1;
        }
        if (sim || dim)
        {
            _vfft_warn("vfft_execute: %s is real->real (in=real_in, out=real_out); "
                       "in_im/out_im must be NULL — nothing executed",
                       tn);
            return 1;
        }
        return 0;
    }
    if (h->transform == VFFT_R2C)
    {
        if (dir != VFFT_FORWARD)
        {
            _vfft_warn("vfft_execute: R2C plans are forward-only (real -> spectrum); the "
                       "unnormalized inverse is a separate VFFT_C2R plan (executed with "
                       "VFFT_BACKWARD) — nothing executed");
            return 1;
        }
        if (sim)
        {
            _vfft_warn("vfft_execute: R2C takes its real input in `in` only; in_im must be NULL "
                       "— nothing executed");
            return 1;
        }
        if (!sre || !dre)
        {
            _vfft_warn("vfft_execute: R2C needs in=real_in and out=%s non-NULL — "
                       "nothing executed",
                       il ? "z_CCE_out" : "spectrum re");
            return 1;
        }
        /* 🔴 PLACEMENT IS A COMMITMENT. An in-place real plan owns ONE
         * padded plane: 2*(N/2+1) doubles, dre == sre. Passing a distinct
         * dre is undocumented misuse that used to be ACCEPTED and silently
         * miscomputed, and which of the two zr2c routes served the call --
         * i.e. a MEASURED wisdom verdict -- decided whether the result was
         * right. Refuse it here instead, mirroring the split-C2C rule.
         *
         * The OOP-aliased case (dre == sre on an OUT-OF-PLACE plan) is not
         * refused: the full door serves it from a copy of the input (field
         * trial V11: the engines are not alias-safe; owner 2026-10-13). */
        if (h->placement == VFFT_INPLACE && dre != sre)
        {
            _vfft_warn("vfft_execute: this %s plan is IN-PLACE (one padded CCE plane of "
                       "2*(N/2+1) doubles) and must be called with out == in; got "
                       "distinct pointers -- nothing executed",
                       tn);
            return 1;
        }
        if (il && dim)
        {
            _vfft_warn("vfft_execute: this R2C plan is committed to layout=INTERLEAVED "
                       "(out = packed CCE spectrum, out_im=NULL) but got a non-NULL out_im; for "
                       "split spectrum output create the plan with layout=VFFT_LAYOUT_SPLIT "
                       "— nothing executed");
            return 1;
        }
        if (!il && !dim)
        {
            _vfft_warn("vfft_execute_split: this R2C plan is SPLIT (out_re, out_im = the "
                       "spectrum planes) but out_im is NULL; an interleaved spectrum is a "
                       "VFFT_LAYOUT_INTERLEAVED plan — nothing executed");
            return 1;
        }
        return 0;
    }
    if (h->transform == VFFT_C2R)
    {
        if (dir != VFFT_BACKWARD)
        {
            _vfft_warn("vfft_execute: C2R plans are backward-only (spectrum -> real, the "
                       "unnormalized inverse); the forward transform is a separate "
                       "VFFT_R2C plan (executed with VFFT_FORWARD) — nothing executed");
            return 1;
        }
        if (dim)
        {
            _vfft_warn("vfft_execute: C2R writes its real output to `out` only; out_im must be NULL "
                       "— nothing executed");
            return 1;
        }
        if (!sre || !dre)
        {
            _vfft_warn("vfft_execute: C2R needs in=%s and out=real_out non-NULL — "
                       "nothing executed",
                       il ? "z_CCE_in" : "spectrum re");
            return 1;
        }
        /* 🔴 PLACEMENT IS A COMMITMENT. An in-place real plan owns ONE
         * padded plane: 2*(N/2+1) doubles, dre == sre. Passing a distinct
         * dre is undocumented misuse that used to be ACCEPTED and silently
         * miscomputed, and which of the two zr2c routes served the call --
         * i.e. a MEASURED wisdom verdict -- decided whether the result was
         * right. Refuse it here instead, mirroring the split-C2C rule.
         *
         * The OOP-aliased case (dre == sre on an OUT-OF-PLACE plan) is not
         * refused: the full door serves it from a copy of the input (field
         * trial V11: the engines are not alias-safe; owner 2026-10-13). */
        if (h->placement == VFFT_INPLACE && dre != sre)
        {
            _vfft_warn("vfft_execute: this %s plan is IN-PLACE (one padded CCE plane of "
                       "2*(N/2+1) doubles) and must be called with out == in; got "
                       "distinct pointers -- nothing executed",
                       tn);
            return 1;
        }
        if (il && sim)
        {
            _vfft_warn("vfft_execute: this C2R plan is committed to layout=INTERLEAVED "
                       "(in = packed CCE spectrum input, in_im=NULL) but got a non-NULL in_im; "
                       "for split spectrum input create the plan with layout=VFFT_LAYOUT_SPLIT "
                       "— nothing executed");
            return 1;
        }
        if (!il && !sim)
        {
            _vfft_warn("vfft_execute_split: this C2R plan is SPLIT (in_re, in_im = the "
                       "spectrum planes) but in_im is NULL; an interleaved spectrum is a "
                       "VFFT_LAYOUT_INTERLEAVED plan — nothing executed");
            return 1;
        }
        return 0;
    }
    /* C2C (1D..4D) */
    if (il)
    {
        if (sim || dim)
        {
            _vfft_warn("vfft_execute: this C2C plan is committed to layout=INTERLEAVED "
                       "(in=z_in, out=z_out, in_im=out_im=NULL) but got non-NULL in_im/out_im; for "
                       "split re/im planes create the plan with layout=VFFT_LAYOUT_SPLIT — "
                       "nothing executed");
            return 1;
        }
        if (!sre || !dre)
        {
            _vfft_warn("vfft_execute: INTERLEAVED C2C needs in=z_in and out=z_out non-NULL "
                       "(out may equal in) — nothing executed");
            return 1;
        }
        return 0;
    }
    if (!sre || !sim)
    {
        if (!sim && sre && !dim && dre)
            _vfft_warn("vfft_execute_split: this C2C plan is SPLIT (in_re/in_im and out_re/out_im "
                       "planes) but in_im and out_im are NULL; interleaved z buffers take a "
                       "VFFT_LAYOUT_INTERLEAVED plan — nothing executed");
        else
            _vfft_warn("vfft_execute_split: SPLIT C2C needs in_re and in_im non-NULL — nothing "
                       "executed");
        return 1;
    }
    if (h->N2 > 0)
    { /* 2D..4D: the executor memcpys src->dst when they differ (both
       * placements); a NULL dst pair means in-place-on-src. */
        if ((dre == NULL) != (dim == NULL))
        {
            _vfft_warn("vfft_execute_split: 2D+ SPLIT C2C got a half-NULL destination pair "
                       "(out_re=%s, out_im=%s) — pass both or neither — nothing executed",
                       dre ? "ok" : "NULL", dim ? "ok" : "NULL");
            return 1;
        }
        return 0;
    }
    if (h->placement == VFFT_INPLACE)
    { /* in-place engine: the destination arguments are NOT read. Accept the
       * documented forms only, so an out-of-place-style call cannot silently
       * leave the result in the source buffers. */
        if (!(((dre == NULL) && (dim == NULL)) || (dre == sre && dim == sim)))
        {
            _vfft_warn("vfft_execute_split: an in-place SPLIT C2C plan takes out_re == in_re "
                       "and out_im == in_im (or both NULL); for out of place create the plan "
                       "with placement=VFFT_PLACEMENT_OUT — nothing executed");
            return 1;
        }
        return 0;
    }
    if (!dre || !dim)
    {
        _vfft_warn("vfft_execute_split: out-of-place SPLIT C2C needs out_re and out_im non-NULL — "
                   "nothing executed");
        return 1;
    }
    if (dre == sre || dim == sim || dre == sim || dim == sre)
    {
        _vfft_warn("vfft_execute_split: out-of-place SPLIT C2C requires destination planes "
                   "disjoint from the sources (got an aliased pointer) — the OOP kernels "
                   "stream the sources while writing the destination, so aliasing corrupts "
                   "the data; for in-place transforms create the plan with "
                   "placement=VFFT_PLACEMENT_IN — nothing executed");
        return 1;
    }
    return 0;
}

#include "il/il_execute.h"           /* the INTERLEAVED execute (IL side of the fork) */
#include "split/split_execute.h"     /* the SPLIT execute */
#include "bridge/real_bridge_exec.h" /* 1D real: the D1 crossing (temporary) */

/* THE FULL DOOR: the calls the fast path (vfft_execute, below) does not
 * finish -- no bound executor, a failed compare, an executor's decline --
 * through the general signature walk (a transform-name lookup, then the
 * real, C2R and C2C branches), which refuses a bad call and says why, and
 * the general dispatch for the rest. Out of line, so the fast path keeps no
 * frame of its own. */
static __attribute__((noinline)) void _vfft_execute_full(vfft_plan h, vfft_dir_t dir,
                                                         double *sre, double *sim, double *dre, double *dim)
{
    if (!h)
    {
        _vfft_warn("vfft_execute: NULL plan (vfft_create failed, or the plan was "
                   "destroyed) — nothing executed");
        return;
    }
    if (dir != VFFT_FORWARD && dir != VFFT_BACKWARD)
    {
        _vfft_warn("vfft_execute: invalid dir value %d (valid: VFFT_FORWARD, "
                   "VFFT_BACKWARD) — nothing executed",
                   (int)dir);
        return;
    }
    if (_vfft_sig_bad(h, dir, sre, sim, dre, dim))
        return;
    if (h->alias_scr && dre == sre)
    {   /* THE ALIASED OUT-OF-PLACE REAL CALL (field trial V11; owner 2026-10-13: served, as in place):
         * the engines read their input while they write the output, so the input is copied aside into
         * the plan's own buffer (allocated at create, the input's size) and the transform runs from the
         * copy into the caller's buffer, which must hold the larger of the input and the output */
        memcpy(h->alias_scr, sre, h->alias_doubles * sizeof(double));
        sre = h->alias_scr;
    }
    if (h->k1_exec && h->k1_exec(h, dir, sre, dre ? dre : sre) == 0)
        return; /* THE BOUND K=1 IL DISPATCH: one indirect call, bound at create (after a decline it declines again) */
    /* 1D real: the one place the layouts still meet (bridge/, owner
     * decision D1 - temporary until the IL real engine lands). */
    if (!h->pq_inner && !h->tcb && h->N2 == 0 &&
        (h->transform == VFFT_R2C || h->transform == VFFT_C2R))
    {
        _vfft_real_bridge_execute(h, dir, sre, sim, dre, dim);
        return;
    }
    /* ── THE LAYOUT FORK: the committed layout picks one side, once ── */
    if (h->layout == (int)VFI_LAYOUT_IL)
        _vfft_il_execute(h, dir, sre, sim, dre, dim);
    else
        _vfft_split_execute(h, dir, sre, sim, dre, dim);
}

/* A BOUND EXECUTOR THAT CAN DECLINE (the 2D real forms: an aliased call of an
 * out-of-place plan): tried first, the full door on a decline. A real plan's
 * executor is its one direction's math and never reads dir, so a call in the
 * other direction is not tried: the full door refuses it and says why. */
static __attribute__((noinline)) void _vfft_execute_try(vfft_plan h, vfft_dir_t dir,
                                                        double *sre, double *sim, double *dre, double *dim)
{
    const int wrong_dir = (h->transform == VFFT_R2C && dir != VFFT_FORWARD) ||
                          (h->transform == VFFT_C2R && dir != VFFT_BACKWARD);
    if (!wrong_dir && h->k1_exec(h, dir, sre, dre) == 0)
        return;
    _vfft_execute_full(h, dir, sre, sim, dre, dim);
}

static inline void _vfft_execute_door(vfft_plan h, vfft_dir_t dir,
                                      double *sre, double *sim, double *dre, double *dim);

/* THE CALLER'S CORE. The pool pins its workers, one per P-core, and leaves
 * one P-core to its caller (threads.h, thread_pool_caller_cpu), but the
 * calling thread runs where the OS puts it, and it runs a full 1/T share of
 * every phase (slot 0 of thread_pool_run), so every fork-join ends at the
 * caller's pace: on a worker's hyperthread sibling 1.3x the pinned time, on
 * an E-core 2.2x, and a bursty caller drifts between executes (64x64x64 c2r
 * T=8: 47 us pinned, 65, 111, 240-590; 1D c2c 262144 T=8: 99 / 150 / 208 us;
 * measured 2026-10-19, caller_probe). So a threaded plan's execute pins the
 * calling thread to that core for its duration and puts its affinity mask
 * back after -- ~1 us the pair, the move immediate -- on every path through
 * the door, a refused call included. A thread whose own mask does not hold
 * that core (an application's narrower pin, a pool worker) is left where it
 * is; one already pinned to it alone (the race scope, a bench) is not moved.
 * No lock is taken here: the fork-joins take the pool's own (thread_pool_run)
 * as before, so a second thread executing a threaded plan waits there, pinned
 * to the same core for its own execute's duration. Windows; elsewhere the
 * caller runs where it is (sched_setaffinity is the later twin). */
typedef struct
{
#ifdef _WIN32
    DWORD_PTR old;   /* the mask to put back; 0 = the thread was not moved */
#else
    int old;
#endif
} _vfft_caller_pin_t;
static inline void _vfft_caller_pin(_vfft_caller_pin_t *p)
{
#ifdef _WIN32
    const int cpu = thread_pool_caller_cpu();
    GROUP_AFFINITY ga;
    p->old = 0;
    if (cpu < 0 || cpu >= (int)(8 * sizeof(DWORD_PTR)))
        return;   /* no worker runs: no core is reserved */
    if (!GetThreadGroupAffinity(GetCurrentThread(), &ga))
        return;
    if (!((ga.Mask >> cpu) & 1) || ga.Mask == ((KAFFINITY)1 << cpu))
        return;   /* the core is not this thread's to take, or the thread is there alone already */
    p->old = SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << cpu);
#else
    p->old = 0;   /* not built: the caller runs where it is */
#endif
}
static inline void _vfft_caller_unpin(const _vfft_caller_pin_t *p)
{
#ifdef _WIN32
    if (p->old)
        SetThreadAffinityMask(GetCurrentThread(), p->old);
#else
    (void)p;
#endif
}

/* THE FP MODE (owner, 2026-10-13): the kernels run FTZ|DAZ; the create set
 * it on the calling thread and left it so, the workers keep it for life, and
 * nothing about MXCSR happens here -- execute is the door and no more. A
 * threaded plan's execute is wrapped by the caller's pin (above): such a plan
 * has no lean entry (vfft_create), so every call of it comes through here. */
static void _vfft_exec6(vfft_plan h, vfft_dir_t dir,
                        double *sre, double *sim, double *dre, double *dim)
{
    _vfft_caller_pin_t pin;
    if (!h || h->nthreads <= 1)
    {
        _vfft_execute_door(h, dir, sre, sim, dre, dim);
        return;
    }
    _vfft_caller_pin(&pin);
    _vfft_execute_door(h, dir, sre, sim, dre, dim);
    _vfft_caller_unpin(&pin);
}
static inline void _vfft_execute_door(vfft_plan h, vfft_dir_t dir,
                                      double *sre, double *sim, double *dre, double *dim)
{
    /* THE BOUND K=1 FAST PATH (2026-09-21). A plan that carries a bound
     * executor (k1_exec, set at create) can be wrong in exactly four ways at
     * this door -- a split plane offered (sim/dim), a NULL buffer, an
     * in-place plan called with two buffers, a bad direction -- and each is
     * one compare. The full door's signature walk cost 4-5 ns per call: 5.6 ns
     * at N = 2 through the door for ~1 ns of kernel, 25% of the cell at
     * N = 32, 12% at 64 (measured 2026-09-21, benches/k1_fwd_ref_probe --time).
     * THE LEAN DOOR (2026-10-11): an executor that never declines (k1_go,
     * per direction: every 1D c2c and 1D real executor) is entered by a jump,
     * with no frame here and no return test; it returns straight to the
     * caller: 0.5-0.7 ns off a small 1D call (N = 8: c2c 6.1 -> 5.4 ns, r2c
     * 5.2 -> 4.6; measured 2026-10-11 against the door before it, same run).
     * One that can decline goes through _vfft_execute_try. Any failed
     * compare goes to the full door, which says why. An OUT-OF-PLACE real
     * plan called with one buffer (dre == sre) is the full door's: it is
     * served from a copy of the input (V11, below); a c2c plan's executors
     * take an aliased call as they are. A THREADED plan carries no lean
     * entry (vfft_create): it comes through _vfft_exec6, which pins its
     * caller around this door, and takes the k1_exec path below. */
    if (h && (unsigned)dir <= (unsigned)VFFT_BACKWARD && sre && dre && !sim && !dim &&
        (h->placement == VFFT_INPLACE ? dre == sre : (dre != sre || h->transform == VFFT_C2C)))
    {
        if (h->k1_go[dir])
        {
            h->k1_go[dir](h, dir, sre, dre);
            return;
        }
        if (h->k1_exec)
        {
            _vfft_execute_try(h, dir, sre, sim, dre, dim);
            return;
        }
    }
    _vfft_execute_full(h, dir, sre, sim, dre, dim);
}

/* ---- destroy (migration step 28) ----
 * The mirror of create, and it must free EVERY plane the plan owns --
 * including the owned batch, whose allocator now lives in vfft_batch.h. */
void vfft_destroy(vfft_plan h)
{
    if (h)
    {
        thread_pool_plan_release(h->nthreads); /* paired with the retain in vfft_create; a child or clone releases its own */
        vfft_aligned_free(h->alias_scr);       /* the aliased out-of-place real call's copy buffer (V11) */
        h->alias_scr = NULL;
        if (h->lm)
        {   /* the lane-major batch: its column pass, and the real batch's packed plane, permutation and twiddles */
            if (h->lm_row)
            {   /* the 2D batch's row child, its clones and its store */
                _lm2d_clones_drop(h);
                vfft_destroy((vfft_plan)h->lm_row);
                vfft_child_store_free(h->lm_rowS);
            }
            _il2d_col_free(&h->il2d_col);
            vfft_aligned_free(h->lm_W);
            vfft_aligned_free(h->lm_tw);
            free(h->lm_perm);
            free(h);
            return;
        }
        if (h->pq_inner)
        { /* plane-queue wrapper: the inner + clones own everything */
            int t;
            vfft_destroy((vfft_plan)h->pq_inner);
            for (t = 0; t < h->pq_wn; t++)
                vfft_destroy((vfft_plan)h->pq_w[t]);
            free(h->pq_w);
            for (t = 0; t < h->pq_gn; t++)   /* the groups arm's clones and barriers (2026-10-19) */
                vfft_destroy((vfft_plan)h->pq_g[t]);
            free(h->pq_g);
            vfft_aligned_free(h->pq_bar);
            free(h);
            return;
        }
        if (h->ilnd)
        { /* the rank-N interleaved tier owns its axes, child and row plan */
            vfft_ilnd_destroy(h->ilnd);
            free(h);
            return;
        }
        if (h->ilndr)
        { /* the rank-3 interleaved real tier owns its axes, child, volume and scratch */
            vfft_ilndr_destroy(h->ilndr);
            free(h);
            return;
        }
        if (h->il2d_row)
        {
            int s2;
            vfft_destroy(h->il2d_row);           /* native IL 2D tier owns its row child */
            vfft_child_store_free(h->il2d_rowS); /* the children's private stores (il/wisdom/wisdom2_child.h) */
            vfft_child_store_free(h->il2d_turnS);
            vfft_child_store_free(h->il2d_cskS);
            vfft_child_store_free(h->il2d_col.tpcS);
            for (s2 = 0; s2 < h->il2d_roww_n; s2++)
                vfft_destroy(h->il2d_roww[s2]); /* the MT row clones */
            free(h->il2d_roww);
            for (s2 = 0; s2 < h->il2d_cskw_n; s2++)
                vfft_destroy(h->il2d_cskw[s2]); /* the threaded skewed pass's row clones (2026-09-24) */
            free(h->il2d_cskw);
            for (s2 = 0; s2 < h->il2d_turnw_n; s2++)
                vfft_destroy(h->il2d_turnw[s2]); /* the threaded turn's N1 plan clones */
            free(h->il2d_turnw);
            vfft_aligned_free(h->il2d_orbuf); /* the odd-N2 row pair buffer */
            free(h->il2d_col.natperm);
            vfft_aligned_free(h->il2d_col.natscr); /* aligned since 2026-09-24 */
            vfft_aligned_free(h->il2d_col.natstage);
            _il2d_nat_sscr_free(&h->il2d_col); /* the strips' dense scratch (2026-09-24) */
            vfft_aligned_free(h->il2d_col.bluchf);
            vfft_aligned_free(h->il2d_col.bluchb);
            vfft_aligned_free(h->il2d_col.blukf);
            vfft_aligned_free(h->il2d_col.blukb);
            vfft_aligned_free(h->il2d_col.bluscr);
            if (h->il2d_col.tpcplan)
                vfft_destroy(h->il2d_col.tpcplan); /* the turned prime pass's 1D plan */
            if (h->il2d_col.tpcscr)
                vfft_aligned_free(h->il2d_col.tpcscr);
            vfft_aligned_free(h->il2d_rowb2_scr); /* the two-pass rows' chunk scratch (route 3) */
            if (h->il2d_turn_plan)
                vfft_destroy(h->il2d_turn_plan); /* the turn route's N1 plan */
            vfft_aligned_free(h->il2d_turn_scr);
            if (h->il2d_csk_row)
                vfft_destroy(h->il2d_csk_row); /* the skewed column pass's OOP row plan */
            vfft_aligned_free(h->il2d_csk_scr);
            vfft_aligned_free(h->il2d_col.bandscr);
            vfft_aligned_free(h->il2d_rscr); /* the real tier's c2r column-inverse plane */
            if (h->il2d_rx_eng)
                vfft_destroy((vfft_plan)h->il2d_rx_eng); /* the real tier's row engine */
            if (h->il2d_rxw)
            { /* its worker clones (the threaded row pass) */
                for (s2 = 0; s2 < h->il2d_rxw_n; s2++)
                    if (h->il2d_rxw[s2])
                        vfft_destroy((vfft_plan)h->il2d_rxw[s2]);
                free(h->il2d_rxw);
            }
            vfft_child_store_free(h->il2d_rxS); /* the row engines' private store */
            _il2d_rax_free(h);                  /* the real axis on N1: its planes, chain, batch and stores */
            _il2d_tf_free(h);                   /* the fused walk's staging */
            _il2d_rcsk_free(h);                 /* the skewed plane */
            if (h->il2d_col.rd0)
            {   /* the Rader stage's engine; its tags in the stage-0 slots go with the tables */
                if (_il2d_rd0_free_hook)
                    _il2d_rd0_free_hook(h->il2d_col.rd0);
                h->il2d_col.rd0 = NULL;
            }
            for (s2 = 0; s2 < h->il2d_col.nst; s2++)
            {
                vfft_aligned_free(h->il2d_col.tf[s2]);
                vfft_aligned_free(h->il2d_col.tb[s2]);
            }
        }
    }
    if (!h)
        return;
    if (h->own_batch)
        _own_batch_free(h->own_batch); /* config.owned_buffers planes */
    vfft_aligned_free(h->own_plane);   /* door 2's plane (real_inplace_design.md) */
    if (h->cplan)
        vfft_proto_plan_destroy(h->cplan);
    if (h->oplan)
        vfft_oop_plan_destroy(h->oplan);
    if (h->tcb)
        vfft_destroy(h->tcb); /* transform-contiguous wrapper owns its K=1 plan */
    if (h->tcbw)
    { /* ...and its MT worker clones (depth-1 recursion: clones have no tcb) */
        for (int t = 0; t < h->tcbw_n; t++)
            vfft_destroy(h->tcbw[t]);
        free(h->tcbw);
    }
    if (h->tcb0)
        vfft_destroy(h->tcb0); /* the primary's serial twin, the caller's slab */
    vfft_il2p_destroy(h->k1il2p);
    vfft_il3p_destroy(h->k1il3p);
    vfft_ilprime_destroy(h->k1ilpr);
    vfft_ilfd_destroy(h->k1ilfd);
    vfft_ztt_destroy(h->k1ztt);
    vfft_k1fs_destroy(h->k1fs);
    if (h->k1sp)
        vfft_oop_plan_destroy(h->k1sp);
    _zr2c_kid_destroy(h->zr2c_kid); /* the zr2c child: its engines (il/real/zr2c_build.h) */
    vfft_zrp_destroy(h->zrp);       /* the real pair (il/real/zrp.h) */
    vfft_zttr_destroy(h->zttr);     /* ZTT-r (il/real/zttr.h) */
    vfft_zfsr_destroy(h->zfsr);     /* the real four-step (il/real/zfsr.h) */
    vfft_zrf_destroy(h->zrf);       /* the real flat DIT (il/real/zrf.h) */
    vfft_zrb_destroy(h->zrb);       /* the real Bluestein (il/real/zrb.h) */
    vfft_zrbl_destroy(h->zrbl);     /* the lane Bluestein (il/real/zrb_lanes.h) */
    vfft_aligned_free(h->zr2c_aff); /* posix_memalign-backed */
    vfft_aligned_free(h->zr2c_scratch);
    if (h->rplan)
        vfft_r2c_plan_destroy(h->rplan);
    if (h->c2rdisp)
        vfft_c2r_disp_destroy(h->c2rdisp);
    if (h->rfft_row)
        vfft_r2c_plan_destroy(h->rfft_row);
    if (h->c2r_row)
        vfft_c2r_disp_destroy(h->c2r_row);
    if (h->tplan)
        stride_plan_destroy(h->tplan); /* frees inner r2c/c2c via override_destroy */
    free(h->nat_list);
    vfft_aligned_free(h->nat_tmp);
    free(h->nat_cyc_off);
    if (h->nat_scr)
    {
        natorder_scr_free(h->nat_scr);
        free(h->nat_scr);
    }
    free(h->nat2d_row_list);
    free(h->nat2d_col_list);
    vfft_aligned_free(h->nat2d_tmp);
    free(h->nat2d_cyc_off);
    free(h);
}

#endif /* VFFT_EXECUTE_IMPL */
#endif /* VFFT_EXECUTE_H */
