/* c2c_estimate_nd_il.h — the ESTIMATE create at rank 2 and 3 (VFFT_FLAG_ESTIMATE): a 2D or 3D
 * interleaved c2c plan built from the formulas' recipe (il/estimate/estimate_nd_il.h) through the
 * doors' own replay, with nothing measured (docs/design/estimate_2d3d_sketch.md §11).
 *
 * The 1D build's steps one rank up (c2c_estimate_il.h), on its pieces:
 *   1. THE PRE-FLIGHT. What the door would refuse, or race behind, checked before the door sees the
 *      row: each column chain with the door's resolver, its forms, its natural permutation, its Rader
 *      stage and Bluestein M chain; the row route's kernels (the n1ccs pair at N2 for ro=2, the row
 *      plan's four row-loop twins for ro=3, the turn's and the skew's); the band width's legality;
 *      the 3D strip width in the door's pool; every 1D child by the 1D pre-flight (est__preflight).
 *      A child the door builds is a nested create, so it is checked without one.
 *   2. THE ROWS. The 2D row, or the 3D row with the 2D row nested under plane_, written with the
 *      store's own writers into a store held in memory (est_put_rows_2d / est_put_rows_3d).
 *   3. THE DOOR. The library's own create runs against it (est__replay: nested, wisdom_write = 0).
 *   4. NOTHING MEASURED. est__replay's detectors, unchanged: races, race scopes, the store's rows,
 *      the library's store, the directory stores; a row re-banked with another payload.
 *   5. THE RECIPE'S PLAN. The handle must carry what the row says -- a door that declined a form
 *      serves another in silence (an illegal band unbanded, a missing kernel's route re-raced, a
 *      clone set dropped): the column chain, Bluestein, the turned pass, the Rader engine, the
 *      natural leaf, the band, the tile, the row route, the turn, the skew, each child's engine;
 *      at rank 3 the structure, the serial verdict, the natural form and its strip width, and the
 *      plane plan as a 2D plan (s=1) or axis 1 and the row plan (s=2).
 * An estimate create never hands back a plan that measured.
 *
 * POSITION IN vfft.c: included by c2c_estimate_il.h after the 1D build, whose pre-flight, plan check
 * and replay it reuses. */
#ifndef VFFT_IL_C2C_ESTIMATE_ND_IL_H
#define VFFT_IL_C2C_ESTIMATE_ND_IL_H

#include "il/estimate/estimate_nd_il.h"

/* ── the pre-flight ─────────────────────────────────────────────────────── */

/* the 1D child's request: natural, one thread */
static inline void est__rq1(est_request_t *rq, int N, int placement)
{
    memset(rq, 0, sizeof *rq);
    rq->N = N;
    rq->transform = VFFT_C2C;
    rq->order = VFFT_ORDER_NATURAL;
    rq->placement = placement;
    rq->nthreads = 1;
}
/* a 1D child through the 1D pre-flight */
static inline int est__preflight_child(const est_recipe_t *rc, int N, int placement, const char *what, char *why,
                                       size_t whylen)
{
    est_request_t rq;
    char w2[400];
    est__rq1(&rq, N, placement);
    if (est__preflight(rc, &rq, w2, sizeof w2))
        return est_refuse(why, whylen, "%s at N = %d does not build: %s", what, N, w2);
    return 0;
}
/* The column pass as _il2d_col_build replays it (il2d_tier.h:3425-3912), without a create: the chain
 * multiplies to N (Bluestein: to the M the door computes, il2d_cols.h:937-938, here in 64 bits) -- else the lookup
 * drops the row and the door races its pool (il2d_tier.h:3450-3457); it resolves and takes its forms
 * (else the forms race, :3238-3250); the natural pass has its leaf permutation (else the create
 * refuses, :3573-3588); a Rader stage builds with the written inner and window and keeps them (else
 * the door takes its default and races, :3047-3052, 3591-3606); a band width is legal (else the door
 * serves unbanded, fft2d_create_il.h:664-675, fftnd_il.h:1516-1519). Returns 0, or -1 with the
 * reason. */
static inline int est__preflight_col(const est_col_t *c, char *why, size_t whylen)
{
    vfft_il2p_fn ff[8], fb[8];
    int L[8];
    long long M = 16;
    if (c->nst < 1 || c->nst > 8)
        return est_refuse(why, whylen, "the column chain at N = %d has %d stages", c->N, c->nst);
    while (M < 2LL * c->N - 1)
        M <<= 1;
    if (c->blu > 0 && c->blu != M)
        return est_refuse(why, whylen, "the column Bluestein at N = %d is M = %lld, not %d", c->N, M, c->blu);
    if (_il2d_chain_prod(c->R, c->nst) != (c->blu > 0 ? c->blu : c->N))
        return est_refuse(why, whylen, "the column chain does not multiply to %d", c->blu > 0 ? c->blu : c->N);
    if (!_il2d_resolve(c->R, c->nst, ff, fb))
        return est_refuse(why, whylen, "the column chain at %d does not resolve (il2d_cols.h:93-113)",
                          c->blu > 0 ? c->blu : c->N);
    {   /* a stage with rival forms needs the token, and the token must fit */
        char f[64];
        est__col_forms(c->R, c->nst, f, sizeof f);
        if (strcmp(f, c->forms) || (c->forms[0] && !_il2d_apply_forms(c->R, c->nst, c->forms, ff, fb)))
            return est_refuse(why, whylen, "the column forms \"%s\" do not fit the chain at %d", c->forms,
                              c->blu > 0 ? c->blu : c->N);
    }
    if (c->blu > 0)
    {
        if (c->tpc != 1 || c->rd_n || c->wl)
            return est_refuse(why, whylen, "the column Bluestein at N = %d with a Rader stage or a band", c->N);
        return 0;
    }
    if (c->tpc >= 0)
        return est_refuse(why, whylen, "a turned prime pass at N = %d without the Bluestein marker", c->N);
    if (c->blu != (est_is_pow2(c->N) ? -1 : 0))
        return est_refuse(why, whylen, "the N-arm verdict blu=%d at N = %d", c->blu, c->N);
    if (c->nat && c->nst > 1)
    {
        int *perm = _il2d_nat_perm(c->R, c->nst, c->N);
        if (!perm)
            return est_refuse(why, whylen, "the natural leaf permutation of the column chain at N = %d does not build", c->N);
        free(perm);
    }
    if (VFFT_IL2D_RADER_RADIX(c->R[0]))
    {
        char d[64], b[64];
        int W = 0;
        est_sb_t s;
        void *rd;
        if (c->rd_n < 1)
            return est_refuse(why, whylen, "the Rader stage at %d carries no inner chain", c->R[0]);
        rd = _ilrd_col0_build(c->R[0], c->N, (size_t)c->w, 1, c->rd, c->rd_n, (size_t)c->rdw);
        if (!rd)
            return est_refuse(why, whylen, "the Rader stage at %d over %d x %lld does not build with the inner and "
                              "window written (_ilrd_col0_build)", c->R[0], c->N, c->w);
        (void)_ilrd_col0_desc(rd, d, sizeof d, &W);
        _ilrd_free((vfft_ilrd_t *)rd);
        est_sb_init(&s, b, sizeof b);
        est_sb_chain(&s, c->rd, c->rd_n);
        if (strcmp(d, b) || W != c->rdw)
            return est_refuse(why, whylen, "the Rader stage at %d built inner %s window %d, not %s window %d", c->R[0],
                              d, W, b, c->rdw);
    }
    else if (c->rd_n)
        return est_refuse(why, whylen, "a Rader stage's tokens on a chain at N = %d without one", c->N);
    est__spans(c->N, c->R, c->nst, L);
    if (c->wl > 0 && vfft_policy_il2d_wl_cut(c->N, c->nst, L, c->wl) < 0)
        return est_refuse(why, whylen, "the band width %d is illegal for the column chain at N = %d", c->wl, c->N);
    return 0;
}
/* the batched two-pass rows bind: the row plan a pair whose four stage kernels have row-loop twins,
 * built as the door builds it (vfft_il2p_create and the forms, k1_commit.h:100-156) and asked as
 * fft2d_create_il.h:510-525 asks it */
static inline int est__rowb2_twins(const est_recipe_t *rp, int N2)
{
    vfft_il2p_plan_t *pp;
    int ok;
    if (rp->k1_il_route != VFFT_K1_IL_2P_PURE)
        return 0;
    if (!(pp = vfft_il2p_create(N2, rp->il_R1, rp->il_R2)))
        return 0;
    ok = vfft_il2p_apply_kv_forms(pp, rp->il_kv) == 0 && (!rp->il_bkv || vfft_il2p_apply_kv_forms_bwd(pp, rp->il_bkv) == 0) &&
         _il2d_rowloop_twin(pp->leaf_f) && _il2d_rowloop_twin(pp->mid_f) && _il2d_rowloop_twin(pp->t2t_b) &&
         _il2d_rowloop_twin(pp->n1_b_r2);
    vfft_il2p_destroy(pp);
    return ok;
}
/* The 2D row as the 2D door replays it (fft2d_create_il.h:327-1318) at one thread: the column pass,
 * the children it builds and their recipes, the row route's kernels -- a banked ro=2 without the
 * n1ccs pair at N2 or ro=3 without the twins is re-raced (:494-500, 526-532) -- the turn's and the
 * skew's. Returns 0, or -1 with the reason. */
static inline int est__preflight_2d(const est_plan2d_t *p, char *why, size_t whylen)
{
    const int mono = vfft_k1_mono_ilc_fn(p->N1, 0) && vfft_k1_mono_ilc_fn(p->N1, 1);
    const int n1ccs = vfft_il_n1ccs_fn(p->N2, 0) && vfft_il_n1ccs_fn(p->N2, 1);
    char w2[400];
    if (p->N1 < 2 || p->N2 < 2 || vfft_policy_c2c_row_refused(p->N2))
        return est_refuse(why, whylen, "%dx%d is not a 2D cell the door serves", p->N1, p->N2);
    if (est__preflight_col(&p->col, w2, sizeof w2))
        return est_refuse(why, whylen, "%dx%d: %s", p->N1, p->N2, w2);
    if (p->col.w != p->N2 || !p->col.nat || p->col.N != p->N1)
        return est_refuse(why, whylen, "%dx%d: the column pass is not the 2D plan's natural pass", p->N1, p->N2);
    if (p->has_turn != (p->col.blu <= 0 && n1ccs) || p->has_csk != (p->col.blu <= 0 && mono))
        return est_refuse(why, whylen, "%dx%d: the children the recipe carries are not the ones the door builds", p->N1,
                          p->N2);
    if (est__preflight_child(&p->rp, p->N2, VFFT_PLACEMENT_IN, "the row plan", why, whylen) ||
        (p->has_turn && est__preflight_child(&p->turnp, p->N1, VFFT_PLACEMENT_IN, "the turn plan", why, whylen)) ||
        (p->has_csk && est__preflight_child(&p->cskp, p->N2, VFFT_PLACEMENT_OUT, "the skewed pass's row plan", why, whylen)) ||
        (p->col.tpc == 1 && est__preflight_child(&p->tpcp, p->N1, VFFT_PLACEMENT_IN, "the turned prime pass", why, whylen)))
        return -1;
    if ((p->ro == 2 && !n1ccs) || (p->ro == 3 && !est__rowb2_twins(&p->rp, p->N2)) || (p->ro != 0 && p->ro != 2 && p->ro != 3))
        return est_refuse(why, whylen, "%dx%d: the row route ro=%d has no kernels for the row plan", p->N1, p->N2, p->ro);
    if ((p->turn && (!p->has_turn || p->csk || p->ro)) || (p->csk && !p->has_csk) || ((p->turn || p->csk) && p->col.wl))
        return est_refuse(why, whylen, "%dx%d: turn=%d csk=%d ro=%d wl=%d is no configuration the door serves", p->N1,
                          p->N2, p->turn, p->csk, p->ro, p->col.wl);
    if (p->sw || p->rbk < 1)
        return est_refuse(why, whylen, "%dx%d: sw=%d rbk=%d", p->N1, p->N2, p->sw, p->rbk);
    if (p->T > 1)
    {   /* the threaded verdict as the door serves it (fft2d_create_il.h:1144-1152, _il2d_c2c_mt il2d_tier.h:1492):
         * the skew and the turn one walk, the chain walk the strips (unbanded) at a width on the door's ladder,
         * the turned prime pass serial; nls beside a natural leaf only */
        const int want_nat = p->col.nat && p->col.blu <= 0 && p->col.nst > 1;
        int lad[6], n, i, in = (p->msw == 0);
        n = _il2d_sw_ladder(p->N1, p->N2, p->T, lad, 6);
        for (i = 0; i < n; i++)
            in |= lad[i] == p->msw;
        if ((p->cmt != 0 && p->cmt != 1) || (!p->cmt && (p->mtarm || p->msw)) || (p->cmt && p->col.tpc == 1) ||
            (p->cmt && (p->turn || p->csk) && (p->mtarm || p->msw)) ||
            (p->cmt && !p->turn && !p->csk && (p->mtarm != 1 || p->col.wl || !in)) || (p->nls >= 0) != want_nat ||
            p->nls > 1)
            return est_refuse(why, whylen, "%dx%d T=%d: cmt=%d mtarm=%d msw=%d nls=%d is no threaded verdict the door serves "
                              "here", p->N1, p->N2, p->T, p->cmt, p->mtarm, p->msw, p->nls);
    }
    else if (p->cmt || p->mtarm || p->msw)
        return est_refuse(why, whylen, "%dx%d: a threaded verdict on a one-thread plan", p->N1, p->N2);
    return 0;
}
/* The 3D row as the 3D door replays it at one thread (fftnd_il.h:1410-1940): axis 0, the structure
 * written alone (the other arm is not built), the natural form -- the strips take the written width
 * only where the door's strip pool is not empty (fftnd_il.h:1566-1575; else it drops the form and
 * races the band), and serve axis 0 unbanded -- then s=1: the plane plan as a 2D plan; s=2: axis 1 as
 * the natural pass at (N2, w = N3) unbanded and the row plan at N3; the turned pass's child. Returns
 * 0, or -1 with the reason. */
static inline int est__preflight_3d(const est_plan3d_t *p, char *why, size_t whylen)
{
    char w2[400];
    if (p->N1 < 2 || p->N2 < 2 || p->N3 < 2 || vfft_policy_c2c_row_refused(p->N3))
        return est_refuse(why, whylen, "%dx%dx%d is not a 3D cell the door serves", p->N1, p->N2, p->N3);
    if ((p->s != 1 && p->s != 2) || (p->nf != 1 && p->nf != 2) || (p->nf == 1 && p->nsw) || (p->nf == 2 && p->col.wl))
        return est_refuse(why, whylen, "%dx%dx%d: s=%d nf=%d nsw=%d wl=%d", p->N1, p->N2, p->N3, p->s, p->nf, p->nsw,
                          p->col.wl);
    if (p->nf == 2)
    {   /* the width one of the door's strip pool at this axis 0 */
        int sp[8], n, i, in = 0;
        n = est__sw_pool(&p->col, p->N2, p->N3, sp, 8);
        for (i = 0; i < n; i++)
            in |= sp[i] == p->nsw;
        if (!in)
            return est_refuse(why, whylen, "%dx%dx%d: the strip width %d is not in the door's strip pool", p->N1, p->N2,
                              p->N3, p->nsw);
    }
    if (est__preflight_col(&p->col, w2, sizeof w2))
        return est_refuse(why, whylen, "%dx%dx%d axis 0: %s", p->N1, p->N2, p->N3, w2);
    if (p->col.nat || p->col.w != (long long)p->N2 * p->N3 || p->col.N != p->N1)
        return est_refuse(why, whylen, "%dx%dx%d: axis 0 is not the scrambled pass over the planes", p->N1, p->N2, p->N3);
    if (p->plane.N1 != p->N2 || p->plane.N2 != p->N3)
        return est_refuse(why, whylen, "%dx%dx%d: the plane plan is not %dx%d", p->N1, p->N2, p->N3, p->N2, p->N3);
    if (p->s == 2)
    {
        if (est__preflight_col(&p->col1, w2, sizeof w2))
            return est_refuse(why, whylen, "%dx%dx%d axis 1: %s", p->N1, p->N2, p->N3, w2);
        if (!p->col1.nat || p->col1.w != p->N3 || p->col1.N != p->N2 || p->col1.wl || p->col1.blu > 0 || p->col1.rd_n)
            return est_refuse(why, whylen, "%dx%dx%d: axis 1 is not the natural chain pass at %d over %d, unbanded", p->N1,
                              p->N2, p->N3, p->N2, p->N3);
        if (est__preflight_child(&p->plane.rp, p->N3, VFFT_PLACEMENT_IN, "the row plan", why, whylen))
            return -1;
    }
    else if (est__preflight_2d(&p->plane, w2, sizeof w2))
        return est_refuse(why, whylen, "%dx%dx%d, the plane plan: %s", p->N1, p->N2, p->N3, w2);
    if (p->col.tpc == 1 && est__preflight_child(&p->tpcp, p->N1, VFFT_PLACEMENT_IN, "the turned prime pass", why, whylen))
        return -1;
    return 0;
}

/* ── the plan the door built ────────────────────────────────────────────── */

/* the checks below return 0 when the handle is the recipe's plan, -1 with the reason */

/* a built child against its recipe: the 1D plan check (est__check_plan), one thread */
static inline int est__check_child(const struct vfft_plan_s *c, const est_recipe_t *rc, int N, int placement,
                                   const char *what, char *why, size_t whylen)
{
    est_request_t rq;
    char w2[256];
    int serial = 0;
    if (!c)
        return est_refuse(why, whylen, "the door built no %s", what);
    est__rq1(&rq, N, placement);
    if (!est__check_plan(c, rc, &rq, &serial, w2, sizeof w2))
        return est_refuse(why, whylen, "%s at N = %d: %s", what, N, w2);
    return 0;
}
/* the column descriptor the door built against the recipe: the chain, Bluestein, the turned pass,
 * the natural leaf, the band, the Rader engine with its inner and window */
static inline int est__check_col(const vfft_ilcol_t *d, const est_col_t *c, char *why, size_t whylen)
{
    const int want_nat = c->nat && c->blu <= 0 && c->nst > 1;
    if (d->nst != c->nst || memcmp(d->R, c->R, sizeof(int) * (size_t)c->nst))
        return est_refuse(why, whylen, "the door built another column chain at N = %d than the recipe's", c->N);
    if (d->blu != (c->blu > 0 ? c->blu : 0) || d->tpc != (c->tpc == 1) || (c->tpc == 1 && !d->tpcplan))
        return est_refuse(why, whylen, "the column pass at N = %d: blu=%d tpc=%d built, blu=%d tpc=%d written", c->N,
                          d->blu, d->tpc, c->blu, c->tpc);
    if (d->nat != want_nat || d->wl != c->wl || d->tfuse != (c->wl > 0))
        return est_refuse(why, whylen, "the column pass at N = %d: natural leaf %d band %d built, %d and %d written",
                          c->N, d->nat, d->wl, want_nat, c->wl);
    if (c->rd_n)
    {
        char dd[64], b[64];
        int W = 0;
        est_sb_t s;
        if (!d->rd0 || !_ilrd_col0_desc(d->rd0, dd, sizeof dd, &W))
            return est_refuse(why, whylen, "the Rader stage at %d has no engine on the plan", c->R[0]);
        est_sb_init(&s, b, sizeof b);
        est_sb_chain(&s, c->rd, c->rd_n);
        if (strcmp(dd, b) || W != c->rdw)
            return est_refuse(why, whylen, "the Rader stage at %d serves inner %s window %d, not %s window %d", c->R[0], dd,
                              W, b, c->rdw);
    }
    return 0;
}
/* the 2D plan against its recipe: the column pass, the tile, the row route and its tile, the turn,
 * the skew, at T > 1 the threaded verdict, every child's engine */
static inline int est__check_2d(const struct vfft_plan_s *h, const est_plan2d_t *p, char *why, size_t whylen)
{
    if (!h || h->N != p->N1 || h->N2 != p->N2 || h->ilnd)
        return est_refuse(why, whylen, "the door built no 2D plan at %dx%d", p->N1, p->N2);
    if (est__check_col(&h->il2d_col, &p->col, why, whylen))
        return -1;
    if (p->T > 1 && (h->nthreads != p->T || h->il2d_col.colmt != p->cmt || h->il2d_col.natarm != p->mtarm ||
                     h->il2d_col.msw != p->msw || h->il2d_col.natst != ((p->cmt && p->nls >= 0) ? p->nls : 1)))
        return est_refuse(why, whylen, "%dx%d: the door serves T=%d cmt=%d mtarm=%d msw=%d nls=%d, the recipe T=%d cmt=%d "
                          "mtarm=%d msw=%d nls=%d", p->N1, p->N2, h->nthreads, h->il2d_col.colmt, h->il2d_col.natarm,
                          h->il2d_col.msw, h->il2d_col.natst, p->T, p->cmt, p->mtarm, p->msw, p->nls);
    if (h->il2d_col.wc != p->sw || h->il2d_rowb != (p->ro == 2) || h->il2d_rowb2 != (p->ro == 3) ||
        (p->ro == 3 && h->il2d_rowb2_ch != (int)_il2d_rb2_rows(p->rbk, (size_t)p->N2)) || h->il2d_turn != p->turn ||
        h->il2d_csk != p->csk)
        return est_refuse(why, whylen, "%dx%d: the door serves sw=%d rowb=%d rowb2=%d turn=%d csk=%d, the recipe sw=%d ro=%d "
                          "turn=%d csk=%d", p->N1, p->N2, h->il2d_col.wc, h->il2d_rowb, h->il2d_rowb2, h->il2d_turn,
                          h->il2d_csk, p->sw, p->ro, p->turn, p->csk);
    if ((!p->has_turn && h->il2d_turn_plan) || (!p->has_csk && h->il2d_csk_row))
        return est_refuse(why, whylen, "%dx%d: the door built a child the recipe does not carry", p->N1, p->N2);
    if (est__check_child(h->il2d_row, &p->rp, p->N2, VFFT_PLACEMENT_IN, "row plan", why, whylen) ||
        (p->has_turn && est__check_child(h->il2d_turn_plan, &p->turnp, p->N1, VFFT_PLACEMENT_IN, "turn plan", why, whylen)) ||
        (p->has_csk && est__check_child(h->il2d_csk_row, &p->cskp, p->N2, VFFT_PLACEMENT_OUT, "skewed pass's row plan", why,
                                        whylen)) ||
        (p->col.tpc == 1 &&
         est__check_child(h->il2d_col.tpcplan, &p->tpcp, p->N1, VFFT_PLACEMENT_IN, "turned prime pass's plan", why, whylen)))
        return -1;
    return 0;
}
/* the 3D plan against its recipe: the structure, serial, the natural form (the strips at the recipe's
 * width, or the cycle walk), axis 0, then s=1: the plane plan as a 2D plan; s=2: axis 1 and the row
 * plan; the turned pass's plan */
static inline int est__check_3d(const struct vfft_plan_s *h, const est_plan3d_t *p, char *why, size_t whylen)
{
    const vfft_ilnd_t *d = h ? h->ilnd : NULL;
    if (!d || d->N[0] != p->N1 || d->N[1] != p->N2 || d->N[2] != p->N3)
        return est_refuse(why, whylen, "the door built no 3D plan at %dx%dx%d", p->N1, p->N2, p->N3);
    if (d->arm != p->s || d->mt != 0 || !d->child != (p->s == 2) || !d->row != (p->s == 1) || d->nf != p->nf ||
        (p->nf == 2 && d->nsw != p->nsw))
        return est_refuse(why, whylen, "%dx%dx%d: the door serves structure %d, mt %d, form %d width %d; the recipe %d, "
                          "serial, %d width %d", p->N1, p->N2, p->N3, d->arm, d->mt, d->nf, d->nsw, p->s, p->nf, p->nsw);
    if (est__check_col(&d->ax0, &p->col, why, whylen))
        return -1;
    if (p->s == 2 ? est__check_col(&d->ax1, &p->col1, why, whylen) ||
                        est__check_child(d->row, &p->plane.rp, p->N3, VFFT_PLACEMENT_IN, "row plan", why, whylen)
                  : est__check_2d(d->child, &p->plane, why, whylen))
        return -1;
    if (p->col.tpc == 1 &&
        est__check_child(d->ax0.tpcplan, &p->tpcp, p->N1, VFFT_PLACEMENT_IN, "turned prime pass's plan", why, whylen))
        return -1;
    return 0;
}

/* ── the build ──────────────────────────────────────────────────────────── */

/* est_build_nd: the recipe of a 2D (p2) or 3D (p3; the other NULL) request -> a plan built with
 * nothing measured, or NULL with the reason in why and the outcome in rep (may be NULL). cfg is the
 * request as the front door translated it: interleaved c2c, rank 2 or 3, howmany 1, natural, rank 2 at
 * the recipe's thread count (est_mt_2d), rank 3 on one thread; its store and save flag are not used.
 * Holds the create lock from the pre-flight to the end.
 * The caller owns the plan (vfft_destroy). */
static inline struct vfft_plan_s *est_build_nd(const est_plan2d_t *p2, const est_plan3d_t *p3, const vfft_cfg_t *cfg,
                                               est_build_report_t *rep, char *why, size_t whylen)
{
    est_build_report_t lrep;
    struct vfft_wisdom_s *S = NULL;
    struct vfft_plan_s *h = NULL;
    est__meter_t m0;
    const int rank = p3 ? 3 : 2;
    if (!rep)
        rep = &lrep;
    memset(rep, 0, sizeof *rep);
    rep->status = EST_REFUSED;
    if (why && whylen)
        why[0] = 0;
    if ((!p2 && !p3) || (p2 && p3) || !cfg)
    {
        est_refuse(why, whylen, "no recipe or configuration");
        return NULL;
    }
    if (cfg->transform != VFFT_C2C || cfg->layout != VFI_LAYOUT_IL || cfg->dims != rank || cfg->howmany > 1 ||
        cfg->order == VFI_ORDER_SCRAMBLED || (p3 && cfg->nthreads > 1) ||
        (p2 && (cfg->n[0] != p2->N1 || cfg->n[1] != p2->N2 || (cfg->nthreads > 1 ? cfg->nthreads : 1) != (p2->T > 1 ? p2->T : 1))) ||
        (p3 && (cfg->n[0] != p3->N1 || cfg->n[1] != p3->N2 || cfg->n[2] != p3->N3)))
    {
        est_refuse(why, whylen, "estimate builds one %dD interleaved c2c transform in natural order, %s", rank,
                   p3 ? "on one thread" : "at the recipe's thread count");
        return NULL;
    }
    vfft_env_init();          /* env.h:96: the pre-flight's tables in the create's FP mode */
    _vfft_create_lock();      /* vfft.c:2485, recursive: the pre-flight touches the create's process statics */
    _vfft_create_depth++;     /* the whole build inside a create: a clock read enters a scope */
    est__meter_read(&m0);
    if (p2 ? est__preflight_2d(p2, why, whylen) : est__preflight_3d(p3, why, whylen))
        goto out;
    if (!(S = vfft_child_store_new()))
    {
        est_refuse(why, whylen, "out of memory");
        goto out;
    }
    if (p2 ? est_put_rows_2d(&S->vw2, p2, why, whylen) : est_put_rows_3d(&S->vw2, p3, why, whylen))
        goto out;
    h = est__replay(S, cfg, &m0, rep, why, whylen);
    if (h && (p2 ? est__check_2d(h, p2, why, whylen) : est__check_3d(h, p3, why, whylen)))
    {
        rep->status = EST_NOPLAN;
        vfft_destroy((vfft_plan)h);
        h = NULL;
    }
    else if (h)
        rep->status = EST_BUILT;
out:
    vfft_child_store_free(S);
    if (--_vfft_create_depth == 0)
        _vfft_scope_create_done();   /* race_scope.h:621: leave a scope a clock read entered */
    _vfft_create_unlock();
    return h;
}

/* THE FRONT DOOR'S ESTIMATE CREATE at rank 2 and 3 (_vfft_create_estimate): the recipe and the build.
 * NULL with the reason on the error record (vfft_error). */
static vfft_plan _vfft_create_estimate_nd(const est_machine_t *m, const vfft_cfg_t *cfg)
{
    char why[640];
    struct vfft_plan_s *h = NULL;
    if (cfg->dims == 2)
    {
        est_plan2d_t *p = (est_plan2d_t *)malloc(sizeof *p);
        if (!p)
            snprintf(why, sizeof why, "out of memory");
        else if (!est_plan_2d(m, cfg->n[0], cfg->n[1], p, why, sizeof why))
        {
            est_mt_2d(m, cfg->nthreads, p);
            h = est_build_nd(p, NULL, cfg, NULL, why, sizeof why);
        }
        free(p);
    }
    else if (cfg->dims == 3)
    {
        est_plan3d_t *p = (est_plan3d_t *)malloc(sizeof *p);
        if (!p)
            snprintf(why, sizeof why, "out of memory");
        else if (!est_plan_3d(m, cfg->n[0], cfg->n[1], cfg->n[2], p, why, sizeof why))
            h = est_build_nd(NULL, p, cfg, NULL, why, sizeof why);
        free(p);
    }
    else
        snprintf(why, sizeof why, "rank %d: estimate plans ranks 1 to 3", cfg->dims);
    if (!h)
        _vfft_warn("vfft_create: VFFT_FLAG_ESTIMATE: %s", why);
    return (vfft_plan)h;
}

#endif /* VFFT_IL_C2C_ESTIMATE_ND_IL_H */
