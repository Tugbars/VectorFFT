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
 * vfft_execute calls six helpers defined ABOVE it in vfft.c
 * (_exec_c2c_interleaved, _exec_c2c_oop_convert, _exec_k1_split, _exec_zcascade,
 * _pq_execute, _vfft_sig_bad). The include therefore replaces the definition in
 * place rather than moving to the top of the file. Those six are later steps'
 * work; until they move, this header's position is load-bearing.
 *
 * THE TRAMPOLINES CAME WITH IT, AND _zc_* DID NOT
 * -----------------------------------------------
 * _tc_mt_floor, _tc_mt_arg and _tc_mt_tramp are used ONLY by vfft_execute, so
 * they belong with it. The _zc_* trampoline pair looks similar but serves
 * _exec_c2c_interleaved, which stays in vfft.c - so it stayed too. Grouping by
 * "looks like a trampoline" would have coupled this header to a function that
 * is not moving.
 *
 * A NOTE ON _tc_mt_floor's CACHE
 * ------------------------------
 * It memoises its result in a function-local static. That is one cache per
 * includer rather than one per program - harmless here, because the value is a
 * pure function of the environment and every includer would compute the same
 * number, and moot in practice because the guard above means only one TU
 * instantiates it at all. Recorded because the general rule in this migration is
 * that mutable state does not go in headers, and this is the one place a
 * (benign) exception exists.
 *
 * The floor itself is a SCALAR DEFAULT, not a wisdom verdict - it decides
 * whether threading is worth engaging at all, and VFFT_TCMT_FLOOR re-maps it.
 */
#ifndef VFFT_EXECUTE_H
#define VFFT_EXECUTE_H

#include "vfft_internal.h"   /* struct vfft_plan_s - the dispatch reads it */

#ifdef VFFT_EXECUTE_IMPL

/* TRANSFORM-CONTIGUOUS batch MT: worker t runs transforms [t0, t0+tc) of
 * the batch through vfft_execute on its OWN clone handle (tcbw comment on
 * the struct) — full independence, no barriers, disjoint blocks. The clone's
 * route is pool-free by _tc_inner_mt_safe, so this re-entry into
 * vfft_execute from a pool thread can never touch the pool. */
/* MT engage floor, in COMPLEX POINTS (callers convert; h->N is not always that).
 * A scalar default, not a wisdom verdict — VFFT_TCMT_FLOOR re-maps the crossover. */
static size_t _tc_mt_floor(void)
{
    static size_t f = 0;
    if (!f)
    {
        const char *e = getenv("VFFT_TCMT_FLOOR");
        long v = e ? atol(e) : 0;
        f = (v > 0) ? (size_t)v : 2048;
    }
    return f;
}

typedef struct
{
    struct vfft_plan_s *p;
    vfft_dir_t dir;
    double *s, *d;
    size_t t0, tc, sn, dn; /* sn/dn: source and destination block strides --
                            * EQUAL for C2C, DIFFERENT for r2c/c2r (see
                            * h->tcb_sn/tcb_dn at create) */
} _tc_mt_arg;
static void _tc_mt_tramp(void *v)
{
    _tc_mt_arg *a = (_tc_mt_arg *)v;
    for (size_t t = 0; t < a->tc; t++)
        vfft_execute(a->p, a->dir, a->s + (a->t0 + t) * a->sn, NULL,
                     a->d + (a->t0 + t) * a->dn, NULL);
}

/* ---- execute-side helpers (migration step 28) ----
 * These four sat in vfft.c immediately above the point this header is
 * included, for one reason: vfft_execute calls them. Moving them here puts
 * them beside their only caller. _exec_c2c_interleaved and _pq_execute did
 * NOT come with them -- both are also called from the CREATE side
 * (c2c_ip_create.h measures with _exec_c2c_interleaved at plan time), so
 * they must stay above this header's include point. */
/* K=1 SCRAMBLED cascade: the single dispatch consumer of h->zroute, both directions.
 * Invariant and route axis are documented at the zroute field. */
static void _exec_zcascade(struct vfft_plan_s *h, vfft_dir_t dir,
                           const double *sre, double *dre)
{
    if (h->zroute)
    {
        if (h->zt_mt && h->nthreads > 1 &&
            _zt_execute_mt(h, dir, sre, dre, h->nthreads))
            return;
        if (dir == VFFT_FORWARD)
            vfft_zturn2_execute_fwd(h->zturn, sre, dre);
        else
            vfft_zturn2_execute_bwd(h->zturn, sre, dre);
    }
    else
    {
        if (dir == VFFT_FORWARD)
            vfft_zsplit_execute_fwd(h->zsplit, sre, dre);
        else
            vfft_zsplit_execute_bwd(h->zsplit, sre, dre);
    }
}

/* K=1 engine, SPLIT-plane side (natural order both directions; split bwd =
 * the pointer-swap identity on the forward route). Extracted verbatim from
 * the dispatch so the OOP INTERLEAVED convert fallback can reuse it. */
static void _exec_k1_split(struct vfft_plan_s *h, int fwd,
                           double *sre, double *sim, double *dre, double *dim)
{
    const double *ar = fwd ? sre : sim, *ai = fwd ? sim : sre;
    double *br = fwd ? dre : dim, *bi = fwd ? dim : dre;
#ifdef VFFT_USE_JIT
    if (h->k1_jit)
    { /* stride-baked whole-route kernel; bwd rides the same
       * pointer-swap identity (natural order) */
        h->k1_jit(ar, ai, br, bi, h->k1sp->col_re, h->k1sp->col_im,
                  h->k1_jit_qr, h->k1_jit_qi);
        return;
    }
#endif
    switch (h->k1_sp_route)
    {
    case VFFT_K1_SP_MONO:
        h->k1_mono(ar, ai, br, bi, 0, 0, 0, 0, 0, 0, 0);
        return;
    case VFFT_K1_SP_2PA:
        vfft_oop_execute_fwd_2pa(h->k1sp, ar, ai, br, bi);
        return;
    case VFFT_K1_SP_2PB:
        vfft_oop_execute_fwd_2pb(h->k1sp, ar, ai, br, bi);
        return;
    case VFFT_K1_SP_TWL:
        vfft_oop_execute_fwd_2pa_twl(h->k1sp, ar, ai, br, bi);
        return;
    case VFFT_K1_SP_CCOL:
        vfft_oop_execute_fwd_ccol(h->k1sp, ar, ai, br, bi);
        return;
    default:
        vfft_oop_execute_fwd(h->k1sp, ar, ai, br, bi);
        return;
    }
}

/* OOP INTERLEAVED convert fallback: dein z -> split OOP engines -> inter z.
 * Serves every OOP cell with NO native z route (K>1; K=1 SCRAMBLED at
 * cascade-uncovered N; K=1 engine cells whose IL route is NONE; k1-create
 * fallbacks) — the cells that were historically a NULL-deref or a silent
 * no-op. Always correct, documented convert cost (vfft.h support matrix). */
static void _exec_c2c_oop_convert(struct vfft_plan_s *h, vfft_dir_t dir,
                                  const double *z_in, double *z_out)
{
    const size_t NK = (size_t)h->N * h->K;
    const size_t bytes = (NK * 8 + 63) & ~(size_t)63;
    /* census knob, cached ONCE (see the ip-site comment: per-execute
     * getenv ~1.3us on Windows dominated tiny-N convert executes). */
    static int _clog_oop = -1;
    if (_clog_oop < 0)
        _clog_oop = getenv("VFFT_CONV_LOG") != NULL;
    if (_clog_oop)
        fprintf(stderr, "[conv] oop N=%d K=%zu dir=%s k1=%d route=%d\n",
                h->N, h->K, dir == VFFT_FORWARD ? "fwd" : "bwd", h->k1_on,
                h->k1_il_route);
    if (!h->il_wr)
    {
        h->il_wr = (double *)STRIDE_ALIGNED_ALLOC(64, bytes);
        h->il_wi = (double *)STRIDE_ALIGNED_ALLOC(64, bytes);
    }
    if (!h->il_wr2)
    {
        h->il_wr2 = (double *)STRIDE_ALIGNED_ALLOC(64, bytes);
        h->il_wi2 = (double *)STRIDE_ALIGNED_ALLOC(64, bytes);
    }
    if (!h->il_wr || !h->il_wi || !h->il_wr2 || !h->il_wi2)
        return;
    _vfft_z_dein(z_in, h->il_wr, h->il_wi, NK);
    if (h->k1_on && h->k1_sp_route < 0)
    {
        /* IL-only K=1 handle (chain cells at odd·2^k N carry NO split
         * route). Unreachable by construction — the IL switch serves such
         * handles and its route always names a runnable plan — but if a
         * future edit breaks that invariant, refuse LOUDLY rather than
         * dispatch _exec_k1_split on route -1. */
        _vfft_warn("vfft_execute: IL-only K=1 handle (N=%d) reached the "
                   "convert fallback — no split route exists; output NOT "
                   "computed. This is a routing bug; please report.",
                   h->N);
        return;
    }
    if (h->k1_on)
        _exec_k1_split(h, dir == VFFT_FORWARD, h->il_wr, h->il_wi,
                       h->il_wr2, h->il_wi2);
    else
    {
        _vfft_pool_arm(h->nthreads);
        _oop_mt(h->oplan, h->il_wr, h->il_wi, h->il_wr2, h->il_wi2,
                dir == VFFT_FORWARD ? 1 : 0);
    }
    _vfft_z_inter(h->il_wr2, h->il_wi2, z_out, NK);
}

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

void vfft_execute(vfft_plan h, vfft_dir_t dir,
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
    if (h->pq_inner)
    { /* 2D PLANE QUEUE (howmany > 1): loop or atomic-counter queue per
       * the raced verdict — see _pq_execute. */
        _pq_execute(h, dir, sre, dre);
        return;
    }
    if (h->oddr_child)
    { /* the ODD-REAL BRIDGE (struct comment at oddr_child): 1D K==1
       * odd-N real transforms through the c2c child. Both layouts —
       * the split spellings pack/unpack around the same child. */
        const size_t n = (size_t)h->N, hp1 = n / 2 + 1;
        double *b1 = h->oddr_buf, *b2 = h->oddr_buf + 2 * n;
        size_t k;
        if (h->transform == VFFT_R2C)
        {
            _il2d_row_promote(sre, b1, n);
            vfft_execute((vfft_plan)h->oddr_child, VFFT_FORWARD, b1,
                         NULL, b2, NULL);
            if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
                memcpy(dre, b2, 2 * hp1 * sizeof(double));
            else
                for (k = 0; k < hp1; k++)
                {
                    dre[k] = b2[2 * k];
                    dim[k] = b2[2 * k + 1];
                }
        }
        else
        {
            if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
                _il2d_row_extend(sre, b1, n, hp1);
            else
            {
                for (k = 0; k < hp1; k++)
                {
                    b1[2 * k] = sre[k];
                    b1[2 * k + 1] = sim[k];
                }
                for (k = 1; k < hp1; k++)
                {
                    b1[2 * (n - k)] = sre[k];
                    b1[2 * (n - k) + 1] = -sim[k];
                }
            }
            vfft_execute((vfft_plan)h->oddr_child, VFFT_BACKWARD, b1,
                         NULL, b2, NULL);
            _il2d_row_re(b2, dre, n);
        }
        return;
    }
    if (h->tcb)
        { /* TRANSFORM-CONTIGUOUS batch: K independent K=1 transforms. Block strides are h->tcb_sn/tcb_dn
     * (equal for C2C, different for r2c/c2r); the inner handle carries route, placement and order.
     * See docs/design/vfft_front_door.md. */
        double *d = dre;
        const size_t sn = h->tcb_sn, dn = h->tcb_dn;
        int T = 1 + h->tcbw_n;
        /* Engage floor is in COMPLEX POINTS, and h->N is not always that.
         * For C2C, N IS the complex length. For R2C/C2R, N counts REAL
         * samples and the transform actually performed is the N/2-point
         * complex child plus a linear fold -- so testing N*K engages
         * threading at HALF the work the floor was calibrated on.
         *
         * Measured 2026-08-22 (8 threads, P-cores, medians of 7), the cells
         * that sit exactly in that gap -- N*K == 2048 real points but only
         * 1024 complex -- all LOSE:
         *     r2c 256x8  0.80x    c2r 256x8  0.74x
         *     r2c 512x4  0.89x    c2r 512x4  0.95x
         * while every cell at 2048 genuine complex points wins (r2c 512x8
         * 1.51x, r2c 1024x4 1.61x, c2r 512x8 1.41x). Converting to complex
         * points turns each of those losses back into the serial path, which
         * is what the floor exists to do. */
        {
        const size_t work = (h->transform == VFFT_C2C)
                                ? (size_t)h->N * h->K
                                : ((size_t)h->N / 2u) * h->K;
        if (T > 1 && work >= _tc_mt_floor())
        { /* engage floor in complex points — MEASURED, see _tc_mt_floor. */
            _vfft_pool_arm(h->nthreads); /* re-assert snapshot pool */
            /* T = 1 + clones built at create (the plan's own snapshot); the
             * pool's one clamp also bounds it by the live pool and the
             * arg-array size, and never above the clone count. */
            T = stride_pool_workers_for(T);
        }
        else
            T = 1;
        }
        if (T > 1)
        {
            /* 🔴 NO TAIL, BY CONSTRUCTION — and note the contrast with the
             * lane-major arm right below (_il_mt_arg), whose slab size is
             * `(ceil(K/T) + 7) & ~7`: there a slab is a set of SIMD LANES,
             * so it must stay a whole multiple of the vector width and the
             * leftover lanes need padded/SSE2 tail machinery. Here the unit
             * of work is ONE WHOLE K=1 TRANSFORM, so ceil(K/T) needs no
             * rounding at all: a ragged K just gives the last worker fewer
             * complete transforms, each running the identical kernel. This
             * is the "loop the K=1 solution for any K" contract — no `me`,
             * no partial-lane count, no padding, nothing to get wrong.
             * Gated at K=43 over 8 threads (slabs 6,6,6,6,6,6,6,1).
             *
             * Slot 0 is the caller on the PRIMARY plan h->tcb; slot t>=1 is
             * worker t-1 on its own clone h->tcbw[t-1] (a clone per worker
             * is what makes the pool-free inner route safe to run
             * concurrently). The pool's fork-join dispatches exactly that. */
            const size_t S = (h->K + (size_t)T - 1) / (size_t)T;
            _tc_mt_arg a[STRIDE_POOL_MAX_DISPATCH];
            int n = 0;
            for (int t = 0; t < T; t++)
            {
                size_t t0 = (size_t)t * S;
                if (t0 >= h->K)
                    break;
                size_t te = t0 + S;
                if (te > h->K)
                    te = h->K;
                a[n++] = (_tc_mt_arg){t == 0 ? h->tcb : h->tcbw[t - 1], dir, sre, d,
                                      t0, te - t0, sn, dn};
            }
            stride_pool_run(n, _tc_mt_tramp, a, sizeof a[0]);
            _vfft_tc_mt_dispatch_count += n - 1; /* one per worker dispatched, see vfft.h */
            return;
        }
        for (size_t t = 0; t < h->K; t++)
            vfft_execute(h->tcb, dir, sre + t * sn, NULL, d + t * dn, NULL);
        return;
    }
    if (h->N2 > 0)
    { /* ── 2D (dispatch before the same-named 1D transforms) ── */
        _vfft_pool_arm(h->nthreads);
        if (h->transform == VFFT_C2C)
        {
            /* tiled-row + native-col, in-place. OOP = copy src->dst then in-place. */
            size_t plane = (size_t)h->N * h->N2 * (h->N3 ? (size_t)h->N3 : 1) * (h->N4 ? (size_t)h->N4 : 1);
            if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
            {
                if (h->il2d_row)
                {
                    /* ── native IL 2D tier (M1/M2): the column chain —
                     * t2c stages then the n1c leaf, block-looped, same
                     * slots (simulator-proven maps, il2d_proto.h) — then
                     * per-row K=1 IL children. The column and row passes
                     * COMMUTE (no inter-pass twiddle) so BWD runs the same
                     * stage order with the bwd pair + conjugated tables.
                     * OOP: stage 0 performs the src->dst move (the kinds
                     * are alias-tolerant both ways). nst == 1 leaves i in
                     * natural order; nst > 1 leaves i digit-reversed by
                     * the chain (the scrambled contract). */
                    const int fwd = (dir == VFFT_FORWARD);
                    size_t i, rn = (size_t)h->N2;
                    const size_t wc = (h->il2d_wc > 0)
                                          ? (size_t)h->il2d_wc
                                          : rn;
                    if (!dre)
                        dre = sre; /* in-place convenience */
                    /* INC-C: the raced MT walk (bands are self-contained
                     * [suffix + fused rows] units because rows commute —
                     * the same fact that legalizes tfuse). Declines back
                     * to the serial walk below when it cannot engage. */
                    if (h->il2d_colmt && h->nthreads > 1 &&
                        _il2d_c2c_mt(h, sre, dre, dir, h->nthreads))
                        return;
                    if (h->il2d_blu)
                    { /* ODD/PRIME N1: the column-axis Bluestein — the
                       * shared pipeline (_il2d_blu_cols), then the rows
                       * (commute). n1 NATURAL on this route. */
                        _il2d_blu_cols(sre, dre, h->N, rn, h->il2d_blu,
                                       h->il2d_nst, h->il2d_R, h->il2d_L,
                                       h->il2d_f, h->il2d_b, h->il2d_tf,
                                       h->il2d_tb,
                                       fwd ? h->il2d_bluchf
                                           : h->il2d_bluchb,
                                       fwd ? h->il2d_blukf
                                           : h->il2d_blukb,
                                       h->il2d_bluscr);
                        for (i = 0; i < (size_t)h->N; i++)
                            _il2d_row_exec(h, dir, dre + 2 * i * rn,
                                           rn);
                        return;
                    }
                    /* strip loop-interchange: all stages depth-first per
                     * column strip — the strip stays cache-resident
                     * across stages (one DRAM sweep, not nst). Legal
                     * because columns are independent within the column
                     * pass; Gs stays the FULL row pitch. wc = rn is the
                     * untiled M2 walk, path-identical. */
                    if (h->il2d_wl > 0)
                    {
                        /* ── BANDED walk (the cascade's tcut, 2D form):
                         * fwd = wide prefix stages 0..cut-1, then per
                         * band of wl rows the stage SUFFIX depth-first
                         * (+ tfuse: that band's row pass, while hot).
                         * bwd mirrors the Hermitian chain: per band
                         * rows-bwd then the REVERSED suffix, then the
                         * reversed wide prefix. Same kernel calls, same
                         * tables, same count — only loop order and base
                         * pointers differ (F0: memcmp-identical). */
                        /* Rows commute with every column stage (disjoint
                         * axes), so BOTH directions keep rows LAST in the
                         * band: the band's first op is the OOP-capable
                         * suffix kernel — no copy for OOP bwd. Execution:
                         * fwd = prefix wide, then per band [suffix fwd,
                         * rows]; bwd = per band [suffix REVERSED (the
                         * Hermitian chain), rows-bwd], then prefix
                         * reversed wide (in place on dre by then). */
                        const int cut = h->il2d_cut, nst = h->il2d_nst;
                        const size_t wl = (size_t)h->il2d_wl;
                        vfft_il2p_fn const *fns = fwd ? h->il2d_f
                                                      : h->il2d_b;
                        double *const *tabs = fwd ? h->il2d_tf
                                                  : h->il2d_tb;
                        size_t b0;
                        if (fwd && cut > 0)
                            _il2d_col_stages(sre, dre, h->N, rn, 0, cut,
                                             h->il2d_R, h->il2d_L, fns,
                                             tabs, 0);
                        for (b0 = 0; b0 < (size_t)h->N; b0 += wl)
                        {
                            const double *bs =
                                (fwd && cut > 0) ? dre + 2 * b0 * rn
                                                 : sre + 2 * b0 * rn;
                            double *bd = dre + 2 * b0 * rn;
                            if (h->il2d_staged)
                            {
                                /* §10b staged: band -> skewed scratch
                                 * (kills the 4KB set-group aliasing,
                                 * priced 2.4-3x on wide stages), suffix
                                 * + rows there, copy back. count stays
                                 * rn: identical arithmetic (F0). */
                                const size_t pit =
                                    (size_t)h->il2d_pitch;
                                double *sc = h->il2d_bandscr;
                                for (i = 0; i < wl; i++)
                                    memcpy(sc + 2 * i * pit,
                                           bs + 2 * i * rn,
                                           2 * rn * sizeof(double));
                                _il2d_col_stages2(sc, sc, (int)wl,
                                                  pit, rn, cut, nst,
                                                  h->il2d_R, h->il2d_L,
                                                  fns, tabs, !fwd);
                                if (h->il2d_tfuse)
                                    for (i = 0; i < wl; i++)
                                        _il2d_row_exec(h, dir,
                                                       sc + 2 * i * pit,
                                                       rn);
                                for (i = 0; i < wl; i++)
                                    memcpy(bd + 2 * i * rn,
                                           sc + 2 * i * pit,
                                           2 * rn * sizeof(double));
                                continue;
                            }
                            _il2d_col_stages(bs, bd, (int)wl, rn, cut,
                                             nst, h->il2d_R, h->il2d_L,
                                             fns, tabs, !fwd);
                            if (h->il2d_tfuse)
                                for (i = 0; i < wl; i++)
                                    _il2d_row_exec(h, dir,
                                                   bd + 2 * i * rn, rn);
                        }
                        if (!fwd && cut > 0)
                            _il2d_col_stages(dre, dre, h->N, rn, 0, cut,
                                             h->il2d_R, h->il2d_L, fns,
                                             tabs, 1);
                        if (!h->il2d_tfuse)
                            for (i = 0; i < (size_t)h->N; i++)
                                _il2d_row_exec(h, dir, dre + 2 * i * rn,
                                               rn);
                        return;
                    }
                    if (h->il2d_nat)
                        _il2d_col_pass_nat(sre, dre, h->N, rn,
                                           h->il2d_nst, h->il2d_R,
                                           h->il2d_L,
                                           fwd ? h->il2d_f : h->il2d_b,
                                           fwd ? h->il2d_tf
                                               : h->il2d_tb,
                                           /*reverse=*/!fwd,
                                           h->il2d_natperm,
                                           h->il2d_natscr);
                    else
                        _il2d_col_pass(sre, dre, h->N, rn, wc,
                                       h->il2d_nst, h->il2d_R,
                                       h->il2d_L,
                                       fwd ? h->il2d_f : h->il2d_b,
                                       fwd ? h->il2d_tf : h->il2d_tb,
                                       /*reverse=*/!fwd);
                    for (i = 0; i < (size_t)h->N; i++)
                        _il2d_row_exec(h, dir, dre + 2 * i * rn, rn);
                    return;
                }
                /* OWNER LAW (2026-08-25): the convert wrapper is
                 * GONE — an IL 2D c2c plan is native or was refused at
                 * create; reaching here without il2d_row is a bug. */
                _vfft_warn("vfft_execute: IL 2D c2c plan without the "
                           "native tier — create/execute wiring bug");
                return;
            }
            if (!dre && !dim)
            { /* validated in-place convenience: result stays in sre/sim */
                dre = sre;
                dim = sim;
            }
            if (dre != sre)
                memcpy(dre, sre, plane * sizeof(double));
            if (dim != sim)
                memcpy(dim, sim, plane * sizeof(double));
            if (dir == VFFT_FORWARD)
            {
                stride_execute_fwd(h->tplan, dre, dim);
                if (h->nat2d)
                    _natorder_2d(h, dre, dim, 0); /* scrambled -> natural (per-axis) */
            }
            else
            {
                if (h->nat2d)
                    _natorder_2d(h, dre, dim, 1); /* natural -> scrambled before the inverse FFT */
                stride_execute_bwd(h->tplan, dre, dim);
            }
        }
        else if (h->transform == VFFT_R2C && h->N3 > 0)
        { /* §6a47/Q1: 3D real fwd — rows, axes, unpack; il per the layout axis. */
            stride_fftnd_r2c_data_t *d3 =
                (stride_fftnd_r2c_data_t *)h->tplan->override_data;
            d3->il_out = (h->layout == (int)VFFT_LAYOUT_INTERLEAVED);
            _fndr_rows_mt(d3, sre, NULL, 0);
            for (int m = 0; m < d3->rank - 1; m++)
                _fndr_axis_mt(d3, m, 0);
            _fndr_unpack(d3, dre, dim);
        }
        else if (h->transform == VFFT_C2R && h->N3 > 0)
        {
            stride_fftnd_r2c_data_t *d3 =
                (stride_fftnd_r2c_data_t *)h->tplan->override_data;
            d3->il_out = (h->layout == (int)VFFT_LAYOUT_INTERLEAVED);
            _fndr_pack(d3, sre, sim);
            for (int m = 0; m < d3->rank - 1; m++)
                _fndr_axis_mt(d3, m, 1);
            _fndr_rows_mt(d3, NULL, dre, 1);
        }
        else if (h->transform == VFFT_R2C && h->il2d_row)
        {
            /* ── native IL 2D REAL fwd (fft2d_real_il_design.md): the
             * batched TC K=N1 zr2c row door does the OOP move (real rows
             * at pitch N2 -> CCE half-spectrum plane at pitch hp1), then
             * the column chain runs IN PLACE over the hp1 columns.
             * Two-phase law (§2.5): the Hermitian fold is R-linear and
             * does not commute with the column stages — ALL rows fold
             * before column stage 0. nst>1 leaves the N1 axis
             * digit-reversed (the scrambled contract); rows (CCE bins)
             * stay natural. dir is ignored (r2c = forward math, the 1D
             * contract). */
            _il2d_real_rows_fwd(h, sre, dre);
            /* INC-3: threaded column pass (band or strip arm, both pure
             * loop restrictions => bitwise identical); falls through to
             * the serial pass when there is not enough independent work
             * or the pool is absent. */
            if (!h->il2d_colmt ||
                !_il2d_real_cols_mt(h, dre, dre, 0, h->nthreads))
                _il2d_real_cols(h, dre, dre, /*reverse=*/0);
        }
        else if (h->transform == VFFT_C2R && h->il2d_row)
        {
            /* ── native IL 2D REAL bwd: the reversed column chain (the
             * Hermitian-transpose pair — conjugated tables pre-butterfly,
             * consuming the r2c pair's scrambled-N1 comb) moves the
             * caller's z into the il2d_rscr plane on its FIRST executed
             * stage (§2.6 input-preserving contract; FFTW destroys its
             * input here — we don't), then the batched TC K=N1 c2r row
             * door folds rows scratch -> the caller's real plane. dir is
             * ignored (c2r = inverse math, unnormalized: caller divides
             * by N1*N2). */
            if (!h->il2d_colmt ||
                !_il2d_real_cols_mt(h, sre, h->il2d_rscr, 1,
                                    h->nthreads))
                _il2d_real_cols(h, sre, h->il2d_rscr, /*reverse=*/1);
            _il2d_real_rows_bwd(h, h->il2d_rscr, dre);
        }
        else if (h->transform == VFFT_R2C)
        {
            if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
                /* OWNER LAW (M3, 2026-08-26): the §6a30 z-veneer no
                 * longer serves IL callers — an IL real 2D plan is
                 * native (il2d_row) or was refused at create. */
                _vfft_warn("vfft_execute: IL 2D r2c plan without the "
                           "native tier — create/execute wiring bug");
            else
                stride_execute_2d_r2c(h->tplan, sre, dre, dim); /* real plane -> split spectrum */
        }
        else if (h->transform == VFFT_C2R)
        {
            if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
                _vfft_warn("vfft_execute: IL 2D c2r plan without the "
                           "native tier — create/execute wiring bug");
            else
                stride_execute_2d_c2r(h->tplan, sre, sim, dre); /* split spectrum -> real plane */
        }
        return;
    }
    if (h->transform == VFFT_C2C && h->placement == VFFT_INPLACE)
    {
        if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
        { /* interleaved z contract — see _exec_c2c_interleaved. (padded
           * plans can't get here: batch+INTERLEAVED is rejected at create.) */
            if (h->zsplit || h->zturn)
            { /* K=1 SCRAMBLED cascade, ALIASED in==out — P0a memcmp-proven
               * both directions incl tiled/tfuse. The documented in-place
               * call form allows dre==NULL; normalize to the aliased buffer
               * (dre==sre is the only other accepted form). */
                _exec_zcascade(h, dir, sre, dre ? dre : sre);
                return;
            }
            if (h->k1il2p || h->k1il3p || h->k1ilpr)
            { /* Phase B (il_coverage_plan.md): sub-2048 native IL tier,
               * ALIASED — two-stage engines through internal scratch, zout
               * written only by the last stage (alias-gated, A3 record);
               * ilprime documents zin==zout safe in both methods.
               * Attach implies verdict (the ord=scr mode cell, mode=ILP);
               * all order spellings land here (identity under SCRAMBLED —
               * Phase A; primes/single-stage are natural = FREE). */
                double *zo = dre ? dre : (double *)sre;
                if (h->k1il2p)
                {
                    if (dir == VFFT_FORWARD)
                        vfft_il2p_execute_fwd(h->k1il2p, sre, zo);
                    else
                        (void)vfft_il2p_execute_bwd(h->k1il2p, sre, zo);
                }
                else if (h->k1il3p)
                {
                    if (dir == VFFT_FORWARD)
                        vfft_il3p_execute_fwd(h->k1il3p, sre, zo);
                    else
                        vfft_il3p_execute_bwd(h->k1il3p, sre, zo);
                }
                else
                {
                    if (dir == VFFT_FORWARD)
                        vfft_ilprime_execute_fwd(h->k1ilpr, sre, zo);
                    else
                        vfft_ilprime_execute_bwd(h->k1ilpr, sre, zo);
                }
                return;
            }
            _vfft_pool_arm(h->nthreads);
            _exec_c2c_interleaved(h, dir, sre, dre);
            return;
        }
        _exec_c2c_inplace(h, dir, sre, sim);
        return;
    }
    if (h->transform == VFFT_C2C && h->placement == VFFT_OUTOFPLACE)
    {
        if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
        { /* z -> z, by the committed axis (signature already validated). */
            if (h->zsplit || h->zturn)
            { /* K=1 SCRAMBLED: the cascade (legacy zsplit or ZTURN-S).
               * fwd: natural -> the route's scrambled comb; bwd consumes
               * the SAME route's comb -> N*natural (matched-permutation
               * roundtrip). BOTH directions go through the one route
               * dispatcher — see _exec_zcascade. */
                _exec_zcascade(h, dir, sre, dre);
                return;
            }
            if (h->k1_on)
            { /* K=1 engine (§13), IL routes; natural order both directions. */
                int fwd = (dir == VFFT_FORWARD);
                switch (h->k1_il_route)
                {
                case VFFT_K1_IL_MONO:
                    (fwd ? h->k1_mono_ilf : h->k1_mono_ilb)(sre, 0, dre, 0,
                                                            0, 0, 0, 0, 0, 0, 0);
                    return;
                case VFFT_K1_IL_2P_PURE:
                    /* Route truthfulness at create makes k1il2p non-NULL for this route; the guard
                     * is defensive — an unresolvable bwd arm breaks to convert, never to silence. */
                    if (h->k1il2p)
                    {
                        if (fwd)
                        {
                            vfft_il2p_execute_fwd(h->k1il2p, sre, dre);
                            return;
                        }
                        if (vfft_il2p_execute_bwd(h->k1il2p, sre, dre) == 0)
                            return;
                    }
                    break; /* -> convert fallback (NEVER a silent no-op) */
                case VFFT_K1_IL_CHAIN3:
                    /* 3-STAGE PURE-IL CHAIN (odd·2^k N): both directions
                     * gated (fwd 12/12, bwd 13/13 — il_odd_chain.md). Route
                     * truthfulness guarantees k1il3p != NULL here; the guard
                     * is defensive, falling to convert, never to silence. */
                    if (h->k1il3p)
                    {
                        if (fwd)
                            vfft_il3p_execute_fwd(h->k1il3p, sre, dre);
                        else
                            vfft_il3p_execute_bwd(h->k1il3p, sre, dre);
                        return;
                    }
                    break; /* -> convert fallback (NEVER a silent no-op) */
                case VFFT_K1_IL_PRIME:
                    /* PRIME N via Rader/Bluestein on IL inner plans
                     * (il_prime.h); both directions, natural order,
                     * unnormalized inverse like every IL bwd. */
                    if (h->k1ilpr)
                    {
                        if (fwd)
                            vfft_ilprime_execute_fwd(h->k1ilpr, sre, dre);
                        else
                            vfft_ilprime_execute_bwd(h->k1ilpr, sre, dre);
                        return;
                    }
                    break; /* -> convert fallback (NEVER a silent no-op) */
                default:
                    break; /* no IL route emitted for this N -> convert
                            * fallback below (NEVER a silent no-op) */
                }
            }
            /* No native z route on this cell (K>1, cascade-uncovered N, or
             * no K=1 IL route): convert around the split engines. */
            _exec_c2c_oop_convert(h, dir, sre, dre);
            return;
        }
        if (h->k1_on)
        { /* K=1 engine, SPLIT planes: natural order; bwd = pointer-swap
           * identity on the forward route. */
            _exec_k1_split(h, dir == VFFT_FORWARD, sre, sim, dre, dim);
            return;
        }
        /* MT via the pool K-split (LEAF/MODEB lane-independent; BAILEY2 + small K run
         * whole-batch — see _oop_mt). vfft_oop_execute_fwd/bwd are kind-correct (natural-
         * order swap for LEAF/BAILEY2; in-place DIF-bwd-on-copy for MODEB) and are the
         * single-thread fallback inside _oop_mt. Caller pins core 0 (workers pin 1..T-1). */
        _vfft_pool_arm(h->nthreads);
        _oop_mt(h->oplan, sre, sim, dre, dim, dir == VFFT_FORWARD ? 1 : 0);
        return;
    }
    if (h->transform == VFFT_R2C)
    {
        /* forward only: real in (sre); spectrum out per the committed layout
         * (SPLIT dre/dim planes, or INTERLEAVED packed CCE z in dre — §6a24).
         * MT internal.
         *
         * 🔴 THE ZR2C COMPOSITE IS POOL-FREE, AND MUST STAY THAT WAY.
         * The pool re-assert below is deliberately AFTER the zr2c branch, not
         * before it. _exec_zr2c is a pure fold plus vfft_execute on the child,
         * and the child was created with c2.nthreads = cfg->nthreads, so it
         * re-asserts the identical snapshot itself -- the outer call was pure
         * duplication. Removing it is what lets a zr2c plan serve as a
         * TRANSFORM-CONTIGUOUS worker clone (_tc_inner_mt_safe): a clone runs
         * on a POOL THREAD, and vfft_set_num_threads from a worker
         * creates/destroys the very pool it is running on. Same edit in the
         * C2R branch below; keep the two in step. */
        if (h->zr2c_child)
        {
            _exec_zr2c(h, sre, dre); /* §D2 composite (incl. in place) */
            return;
        }
        _vfft_pool_arm(h->nthreads);
        if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
            vfft_r2c_execute_fwd_z(h->rplan, sre, dre); /* dre = packed CCE spectrum */
        else
            vfft_r2c_execute_fwd(h->rplan, sre, dre, dim); /* split out */
        return;
    }
    if (h->transform == VFFT_C2R)
    {
        /* the inverse: spectrum in per the committed layout (SPLIT sre/sim, or
         * INTERLEAVED packed CCE z in sre — §6a24) -> real out (dre). dir
         * ignored. NATURAL or STRIDE per the bakeoff/wisdom.
         *
         * 🔴 Pool-free zr2c: the mirror of the R2C branch above -- read
         * that comment before moving either call. */
        if (h->zr2c_child)
        {
            _exec_zr2c(h, sre, dre); /* §D2 composite (incl. in place) */
            return;
        }
        _vfft_pool_arm(h->nthreads);
        if (h->layout == (int)VFFT_LAYOUT_INTERLEAVED)
            vfft_c2r_disp_execute_z(h->c2rdisp, sre, dre); /* sre = packed CCE spectrum in */
        else
            vfft_c2r_disp_execute(h->c2rdisp, sre, sim, dre);
        return;
    }
    if (_VFFT_IS_TRIG(h->transform))
    {
        /* real in (sre) -> real out (dre). Involutory kinds (DCT-I/IV, DST-I, DHT)
         * ignore `dir`; for II<->III the forward enum picks the matching member and
         * BACKWARD runs its inverse (DCT-III for a DCT-II plan, etc.). */
        _vfft_pool_arm(h->nthreads);
        /* HARNESS: the trig family's only engagement signal. A DCT/DST/DHT plan
         * sets tplan and never touches tcb, so none of the four pre-existing
         * counters can move for it and an MT==ST bitwise pass here is vacuous -
         * it passes just as happily when no thread ever ran.
         *
         * HONEST LIMIT: this counts trig executes issued with a THREADED POOL,
         * not work proven dispatched. Dispatch happens inside the
         * stride_execute_dctN entry points, below this file. Closing that last
         * gap needs a counter inside the trig executor; until then a non-zero
         * value proves the pool was armed, which is strictly more than the
         * nothing that was observable before. */
        if (h->nthreads > 1)
            _vfft_trig_mt_count++;
        const stride_plan_t *p = h->tplan;
        int f = (dir == VFFT_FORWARD);
        switch (h->transform)
        {
        case VFFT_DCT1:
            stride_execute_dct1(p, sre, dre);
            break;
        case VFFT_DCT2:
            if (f)
                stride_execute_dct2(p, sre, dre);
            else
                stride_execute_dct3(p, sre, dre);
            break;
        case VFFT_DCT3:
            if (f)
                stride_execute_dct3(p, sre, dre);
            else
                stride_execute_dct2(p, sre, dre);
            break;
        case VFFT_DCT4:
            stride_execute_dct4(p, sre, dre);
            break;
        case VFFT_DST1:
            stride_execute_dst1(p, sre, dre);
            break;
        case VFFT_DST2:
            if (f)
                stride_execute_dst2(p, sre, dre);
            else
                stride_execute_dst3(p, sre, dre);
            break;
        case VFFT_DST3:
            if (f)
                stride_execute_dst3(p, sre, dre);
            else
                stride_execute_dst2(p, sre, dre);
            break;
        case VFFT_DHT:
            stride_execute_dht(p, sre, dre);
            break;
        default:
            break;
        }
        return;
    }
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
        if (h->cplan_il)
            stride_plan_destroy(h->cplan_il);
        STRIDE_ALIGNED_FREE(h->il_wr);
        STRIDE_ALIGNED_FREE(h->il_wi);
        STRIDE_ALIGNED_FREE(h->il_wr2);
        STRIDE_ALIGNED_FREE(h->il_wi2);
        if (h->il2d_row)
        {
            int s2;
            if (h->il2d_row != h->il2d_rowo)
                vfft_destroy(h->il2d_row); /* native IL 2D tier owns its row child */
            if (h->il2d_rowo)
                vfft_destroy(h->il2d_rowo); /* (the forced-oop route aliases
                                             * il2d_row to rowo — freed once) */
            for (s2 = 0; s2 < h->il2d_roww_n; s2++)
                vfft_destroy(h->il2d_roww[s2]); /* the MT row clones */
            free(h->il2d_roww);
            free(h->il2d_rowscr_w);
            free(h->il2d_orbuf); /* the odd-N2 row pair buffer */
            free(h->il2d_natperm);
            free(h->il2d_natscr);
            free(h->il2d_bluchf);
            free(h->il2d_bluchb);
            free(h->il2d_blukf);
            free(h->il2d_blukb);
            free(h->il2d_bluscr);
            free(h->il2d_rowscr);
            free(h->il2d_bandscr);
            free(h->il2d_rscr); /* the real tier's c2r column-inverse plane */
            if (h->il2d_rows)
                vfft_destroy(h->il2d_rows); /* the rowsplit band engine */
            free(h->il2d_lx);
            free(h->il2d_lre);
            free(h->il2d_lim);
            free(h->il2d_tre);
            free(h->il2d_tim);
            for (s2 = 0; s2 < h->il2d_nst; s2++)
            {
                free(h->il2d_tf[s2]);
                free(h->il2d_tb[s2]);
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
    if (h->zsplit)
        vfft_zsplit_destroy(h->zsplit);
    if (h->tcb)
        vfft_destroy(h->tcb); /* transform-contiguous wrapper owns its K=1 plan */
    if (h->tcbw)
    { /* ...and its MT worker clones (depth-1 recursion: clones have no tcb) */
        for (int t = 0; t < h->tcbw_n; t++)
            vfft_destroy(h->tcbw[t]);
        free(h->tcbw);
    }
    if (h->zturn)
        vfft_zturn2_destroy(h->zturn);
    vfft_il2p_destroy(h->k1il2p);
    vfft_il3p_destroy(h->k1il3p);
    vfft_ilprime_destroy(h->k1ilpr);
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
