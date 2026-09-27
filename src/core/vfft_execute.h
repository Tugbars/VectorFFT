/* vfft_execute.h - THE execute entry point.
 *
 * vfft_execute and the transform-contiguous batch MT it dispatches through.
 * Extracted from vfft.c as migration step 16; see
 * docs/design/refactor_migration_plan.md. Step 28 extends this header.
 *
 * SCOPE: EVERYTHING. NOT c2c, NOT interleaved.
 * --------------------------------------------
 * The migration plan's table labels this step "execute-side c2c dispatch", and
 * that label is too narrow - recorded here because it misled once already. This
 * is the single universal dispatcher behind the public API. It branches on:
 *
 *   transform   C2C, R2C, C2R, the trig family (DCT/DST/DHT via _VFFT_IS_TRIG),
 *               and the rank-3 paths (N3 > 0).
 *   layout      at essentially every one of those branches. SPLIT IS SERVED
 *               HERE TOO - it is the fall-through arm of each
 *               `h->layout == VFFT_LAYOUT_INTERLEAVED` test, which is why the
 *               token VFFT_LAYOUT_SPLIT never literally appears in this file.
 *               Reading "no LAYOUT_SPLIT" as "interleaved only" is exactly
 *               backwards.
 *   placement   in-place and out-of-place take different arms throughout.
 *
 * The two families share no codelets, no planner and no executor (see
 * docs/design/planning_model.md, Parts II and III) - but they share this one
 * front door, and the layout test is where they part company.
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
 * vfft_execute calls helpers that must be declared before it: _pq_execute
 * (plane_queue.h, which vfft.c includes just above this header) and the
 * create-side statics of vfft.c. The include therefore replaces the definition
 * in place rather than moving to the top of the file; its position is
 * load-bearing. (_exec_c2c_interleaved, _exec_c2c_oop_convert and
 * _exec_zcascade, named here before, were retired with the convert machinery
 * on 2026-09-03.)
 *
 * THE TRAMPOLINES CAME WITH IT, AND _zc_* DID NOT
 * -----------------------------------------------
 * _tc_mt_arg and _tc_mt_tramp are used ONLY by vfft_execute, so
 * they belong with it. The _zc_* trampoline pair looks similar but serves
 * _exec_c2c_interleaved, which stays in vfft.c - so it stayed too. Grouping by
 * "looks like a trampoline" would have coupled this header to a function that
 * is not moving.
 *
 * (The 2048-point scalar engage floor that once lived here is retired:
 * the wrapper now carries a raced, banked verdict, h->tc_mt - see
 * _tc_mt_decide in vfft.c.)
 */
#ifndef VFFT_EXECUTE_H
#define VFFT_EXECUTE_H

#include "vfft_internal.h"   /* struct vfft_plan_s - the dispatch reads it */

#ifdef VFFT_EXECUTE_IMPL


/* ---- execute-side helpers (migration step 28) ----
 * These sat in vfft.c immediately above the point this header is included,
 * for one reason: vfft_execute calls them. Moving them here puts them beside
 * their only caller. _pq_execute did NOT come with them -- the create side
 * (fft2d_create.h) calls it too -- so it stays above this header's include
 * point, in plane_queue.h. */



/* ── EXECUTE-SIDE SIGNATURE ENFORCEMENT ──
 * The pointer pattern must MATCH the plan's committed layout; the historical
 * NULL-pointer inference ("sim==dim==NULL means interleaved") is REMOVED.
 * Returns 1 (and prints an actionable stderr line) when the call must be
 * REFUSED — the caller returns without computing ANYTHING, so a mismatch can
 * never silently reinterpret buffers or produce garbage. */
static int _vfft_sig_bad(struct vfft_plan_s *h, vfft_dir_t dir, double *sre,
                         double *sim, double *dre, double *dim)
{
    const int il = (h->layout == (int)VFFT_LAYOUT_INTERLEAVED);
    const char *tn = _vfft_tname(h->transform);
    if (_VFFT_IS_TRIG(h->transform))
    {
        if (!sre || !dre)
        {
            _vfft_warn("vfft_execute: %s needs sre=real_in and dre=real_out non-NULL "
                       "(got sre=%s, dre=%s) — nothing executed",
                       tn, sre ? "ok" : "NULL", dre ? "ok" : "NULL");
            return 1;
        }
        if (sim || dim)
        {
            _vfft_warn("vfft_execute: %s is real->real (sre=real_in, dre=real_out); "
                       "sim/dim must be NULL — nothing executed",
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
            _vfft_warn("vfft_execute: R2C takes real input in sre only; sim must be NULL "
                       "— nothing executed");
            return 1;
        }
        if (!sre || !dre)
        {
            _vfft_warn("vfft_execute: R2C needs sre=real_in and dre=%s non-NULL — "
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
         * The OOP-aliased case (dre == sre on an OUT-OF-PLACE plan) is
         * deliberately NOT refused: it currently works on both routes and on
         * c2r, and turning working behaviour into an error is a separate
         * decision from closing a miscomputation. */
        if (h->placement == VFFT_INPLACE && dre != sre)
        {
            _vfft_warn("vfft_execute: this %s plan is IN-PLACE (one padded CCE plane of "
                       "2*(N/2+1) doubles) and must be called with dre == sre; got "
                       "distinct pointers -- nothing executed", tn);
            return 1;
        }
        if (il && dim)
        {
            _vfft_warn("vfft_execute: this R2C plan is committed to layout=INTERLEAVED "
                       "(dre = packed CCE spectrum, dim=NULL) but got a non-NULL dim; for "
                       "split spectrum output create the plan with layout=VFFT_LAYOUT_SPLIT "
                       "— nothing executed");
            return 1;
        }
        if (!il && !dim)
        {
            _vfft_warn("vfft_execute: this R2C plan is committed to layout=SPLIT "
                       "(dre/dim = split spectrum planes) but dim is NULL. The old "
                       "\"dim==NULL means CCE\" inference is REMOVED — create the plan with "
                       "layout=VFFT_LAYOUT_INTERLEAVED for the packed z spectrum — nothing "
                       "executed");
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
            _vfft_warn("vfft_execute: C2R writes real output to dre only; dim must be NULL "
                       "— nothing executed");
            return 1;
        }
        if (!sre || !dre)
        {
            _vfft_warn("vfft_execute: C2R needs sre=%s and dre=real_out non-NULL — "
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
         * The OOP-aliased case (dre == sre on an OUT-OF-PLACE plan) is
         * deliberately NOT refused: it currently works on both routes and on
         * c2r, and turning working behaviour into an error is a separate
         * decision from closing a miscomputation. */
        if (h->placement == VFFT_INPLACE && dre != sre)
        {
            _vfft_warn("vfft_execute: this %s plan is IN-PLACE (one padded CCE plane of "
                       "2*(N/2+1) doubles) and must be called with dre == sre; got "
                       "distinct pointers -- nothing executed", tn);
            return 1;
        }
        if (il && sim)
        {
            _vfft_warn("vfft_execute: this C2R plan is committed to layout=INTERLEAVED "
                       "(sre = packed CCE spectrum input, sim=NULL) but got a non-NULL sim; "
                       "for split spectrum input create the plan with layout=VFFT_LAYOUT_SPLIT "
                       "— nothing executed");
            return 1;
        }
        if (!il && !sim)
        {
            _vfft_warn("vfft_execute: this C2R plan is committed to layout=SPLIT "
                       "(sre/sim = split spectrum planes) but sim is NULL. The old "
                       "\"sim==NULL means CCE\" inference is REMOVED — create the plan with "
                       "layout=VFFT_LAYOUT_INTERLEAVED for the packed z spectrum — nothing "
                       "executed");
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
                       "(sre=z_in, dre=z_out, sim=dim=NULL) but got non-NULL sim/dim; for "
                       "split re/im planes create the plan with layout=VFFT_LAYOUT_SPLIT — "
                       "nothing executed");
            return 1;
        }
        if (!sre || !dre)
        {
            _vfft_warn("vfft_execute: INTERLEAVED C2C needs sre=z_in and dre=z_out non-NULL "
                       "(dre may equal sre) — nothing executed");
            return 1;
        }
        return 0;
    }
    if (!sre || !sim)
    {
        if (!sim && sre && !dim && dre)
            _vfft_warn("vfft_execute: this C2C plan is committed to layout=SPLIT (sre/sim + "
                       "dre/dim planes) but the call passed the interleaved-style signature "
                       "(sim==dim==NULL). The old NULL-pointer layout inference is REMOVED — "
                       "create the plan with layout=VFFT_LAYOUT_INTERLEAVED for z buffers — "
                       "nothing executed");
        else
            _vfft_warn("vfft_execute: SPLIT C2C needs sre and sim non-NULL — nothing "
                       "executed");
        return 1;
    }
    if (h->N2 > 0)
    { /* 2D..4D: the executor memcpys src->dst when they differ (both
       * placements); a NULL dst pair means in-place-on-src. */
        if ((dre == NULL) != (dim == NULL))
        {
            _vfft_warn("vfft_execute: 2D+ SPLIT C2C got a half-NULL destination pair "
                       "(dre=%s, dim=%s) — pass both or neither — nothing executed",
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
            _vfft_warn("vfft_execute: in-place SPLIT C2C takes dre==sre && dim==sim (or "
                       "dre=dim=NULL); a different destination is ignored by the in-place "
                       "engine — for true out-of-place create with "
                       "placement=VFFT_OUTOFPLACE — nothing executed");
            return 1;
        }
        return 0;
    }
    if (!dre || !dim)
    {
        _vfft_warn("vfft_execute: out-of-place SPLIT C2C needs dre and dim non-NULL — "
                   "nothing executed");
        return 1;
    }
    if (dre == sre || dim == sim || dre == sim || dim == sre)
    {
        _vfft_warn("vfft_execute: out-of-place SPLIT C2C requires destination planes "
                   "disjoint from the sources (got an aliased pointer) — the OOP kernels "
                   "stream the sources while writing the destination, so aliasing corrupts "
                   "the data; for in-place transforms create the plan with "
                   "placement=VFFT_INPLACE — nothing executed");
        return 1;
    }
    return 0;
}

#include "il/il_execute.h"          /* the INTERLEAVED execute (IL side of the fork) */
#include "split/split_execute.h"    /* the SPLIT execute */
#include "bridge/real_bridge_exec.h" /* 1D real: the D1 crossing (temporary) */

void vfft_execute(vfft_plan h, vfft_dir_t dir,
                  double *sre, double *sim, double *dre, double *dim)
{
    /* THE BOUND K=1 IL FAST PATH (2026-09-21). A plan that carries the bound
     * K=1 interleaved dispatch (k1_exec, set at both c2c create exits) can be
     * wrong in exactly four ways at this door -- a split plane offered
     * (sim/dim), a NULL buffer, an in-place plan called with two buffers, a
     * bad direction -- and each is one compare. The general signature walk
     * below (a transform-name lookup, then the real, C2R and C2C branches)
     * cost 4-5 ns per call: 5.6 ns at N = 2 through the door for ~1 ns of
     * kernel, 25% of the cell at N = 32, 12% at 64 (measured 2026-09-21,
     * benches/k1_fwd_ref_probe --time). Any failed compare falls through to
     * the general path, which is unchanged and says why. */
    int k1_tried = 0;
    if (h && h->k1_exec && sre && dre && !sim && !dim &&
        (dir == VFFT_FORWARD || dir == VFFT_BACKWARD) &&
        (h->placement != VFFT_INPLACE || dre == sre))
    {
        if (h->k1_exec(h, dir, sre, dre) == 0)
            return;
        k1_tried = 1;   /* the trampoline declined (il2p's unresolvable bwd arm): the general path decides */
    }
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
    if (!k1_tried && h->k1_exec && h->k1_exec(h, dir, sre, dre ? dre : sre) == 0)
        return; /* THE BOUND K=1 IL DISPATCH: one indirect call, bound at create */
    /* 1D real: the one place the layouts still meet (bridge/, owner
     * decision D1 - temporary until the IL real engine lands). */
    if (h->oddr_child ||
        (!h->pq_inner && !h->tcb && h->N2 == 0 &&
         (h->transform == VFFT_R2C || h->transform == VFFT_C2R)))
    {
        _vfft_real_bridge_execute(h, dir, sre, sim, dre, dim);
        return;
    }
    /* ── THE LAYOUT FORK: the committed layout picks one side, once ── */
    if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
        _vfft_il_execute(h, dir, sre, sim, dre, dim);
    else
        _vfft_split_execute(h, dir, sre, sim, dre, dim);
}

/* ---- destroy (migration step 28) ----
 * The mirror of create, and it must free EVERY plane the plan owns --
 * including the owned batch, whose allocator now lives in vfft_batch.h. */
void vfft_destroy(vfft_plan h)
{
    if (h)
    {
        if (h->pq_inner)
        { /* plane-queue wrapper: the inner + clones own everything */
            int t;
            vfft_destroy((vfft_plan)h->pq_inner);
            for (t = 0; t < h->pq_wn; t++)
                vfft_destroy((vfft_plan)h->pq_w[t]);
            free(h->pq_w);
            free(h);
            return;
        }
        if (h->oddr_child)
        { /* the odd-real bridge: the child + one buffer */
            vfft_destroy((vfft_plan)h->oddr_child);
            free(h->oddr_buf);
            free(h);
            return;
        }
        if (h->ilnd)
        { /* the rank-N interleaved tier owns its axes, child and row plan */
            vfft_ilnd_destroy(h->ilnd);
            free(h);
            return;
        }
        if (h->il2d_row)
        {
            int s2;
            vfft_destroy(h->il2d_row); /* native IL 2D tier owns its row child */
            for (s2 = 0; s2 < h->il2d_roww_n; s2++)
                vfft_destroy(h->il2d_roww[s2]); /* the MT row clones */
            free(h->il2d_roww);
            for (s2 = 0; s2 < h->il2d_cskw_n; s2++)
                vfft_destroy(h->il2d_cskw[s2]); /* the threaded skewed pass's row clones (2026-09-24) */
            free(h->il2d_cskw);
            for (s2 = 0; s2 < h->il2d_turnw_n; s2++)
                vfft_destroy(h->il2d_turnw[s2]); /* the threaded turn's N1 plan clones */
            free(h->il2d_turnw);
            free(h->il2d_orbuf); /* the odd-N2 row pair buffer */
            free(h->il2d_col.natperm);
            VFFT_ZS_FREE(h->il2d_col.natscr);   /* aligned since 2026-09-24 */
            VFFT_ZS_FREE(h->il2d_col.natstage);
            _il2d_nat_sscr_free(&h->il2d_col);   /* the strips' dense scratch (2026-09-24) */
            free(h->il2d_col.bluchf);
            free(h->il2d_col.bluchb);
            free(h->il2d_col.blukf);
            free(h->il2d_col.blukb);
            free(h->il2d_col.bluscr);
            if (h->il2d_col.tpcplan)
                vfft_destroy(h->il2d_col.tpcplan); /* the turned prime pass's 1D plan */
            if (h->il2d_col.tpcscr)
                VFFT_ZS_FREE(h->il2d_col.tpcscr);
            VFFT_ZS_FREE(h->il2d_rowb2_scr);   /* the two-pass rows' chunk scratch (route 3) */
            if (h->il2d_turn_plan)
                vfft_destroy(h->il2d_turn_plan); /* the turn route's N1 plan */
            VFFT_ZS_FREE(h->il2d_turn_scr);
            if (h->il2d_csk_row)
                vfft_destroy(h->il2d_csk_row); /* the skewed column pass's OOP row plan */
            VFFT_ZS_FREE(h->il2d_csk_scr);
            free(h->il2d_col.bandscr);
            free(h->il2d_rscr); /* the real tier's c2r column-inverse plane */
            if (h->il2d_rows)
                vfft_destroy(h->il2d_rows); /* the rowsplit band engine */
            free(h->il2d_lx);
            free(h->il2d_lre);
            free(h->il2d_lim);
            free(h->il2d_tre);
            free(h->il2d_tim);
            for (s2 = 0; s2 < h->il2d_col.nst; s2++)
            {
                free(h->il2d_col.tf[s2]);
                free(h->il2d_col.tb[s2]);
            }
        }
    }
    if (!h)
        return;
    if (h->own_batch)
        _own_batch_free(h->own_batch); /* config.owned_buffers planes */
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
    vfft_il2p_destroy(h->k1il2p);
    vfft_il3p_destroy(h->k1il3p);
    vfft_ilprime_destroy(h->k1ilpr);
    vfft_ilfd_destroy(h->k1ilfd);
    vfft_ztt_destroy(h->k1ztt);
    vfft_k1fs_destroy(h->k1fs);
    if (h->k1sp)
        vfft_oop_plan_destroy(h->k1sp);
    if (h->zr2c_child)
        vfft_destroy((vfft_plan)h->zr2c_child); /* §D2: recursive child */
    vfft_proto_aligned_free(h->zr2c_aff);      /* posix_memalign-backed */
    vfft_proto_aligned_free(h->zr2c_scratch);
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
    free(h->nat_tmp);
    free(h->nat_cyc_off);
    if (h->nat_scr)
    {
        natorder_scr_free(h->nat_scr);
        free(h->nat_scr);
    }
    free(h->nat2d_row_list);
    free(h->nat2d_col_list);
    free(h->nat2d_tmp);
    free(h->nat2d_cyc_off);
    free(h);
}

#endif /* VFFT_EXECUTE_IMPL */
#endif /* VFFT_EXECUTE_H */
