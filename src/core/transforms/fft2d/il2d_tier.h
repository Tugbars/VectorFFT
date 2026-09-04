/* il2d_tier.h - the native interleaved 2D tier.
 *
 * The row and column passes, their multithreaded forms, and the four racers
 * that decide a 2D IL cell's plan. Extracted from vfft.c as migration step 17;
 * see docs/design/refactor_migration_plan.md.
 *
 * THIS IS THE FIRST MOVE THE GATE MADE POSSIBLE
 * ---------------------------------------------
 * Almost every function here dereferences vfft_plan_s. Before step 15 lifted
 * that struct into vfft_internal.h, none of this could leave vfft.c at any
 * price - which is exactly why step 15 was the gate and why the ten moves
 * before it were confined to struct-free code.
 *
 * WHAT DECIDES WHAT, AND IN WHICH ORDER
 * -------------------------------------
 * The tier's plan is not one search. It is a chain of races where an earlier
 * verdict decides whether a later axis EXISTS at all:
 *
 *   chain      factorize N1 over the radix pool (depth <= 4, cap 24, and the
 *              cap LOGS when it bites - a truncated pool is a biased pool).
 *   wl         the banded column-walk width. A 2D column pass touches every
 *              row, which blows the cache in one sweep; banding keeps the
 *              working set resident. Measured, not computed, because the best
 *              band depends on cache size AND on the chain's stage spans.
 *   cut, tf    NOT independent. The tcut law: width is the INPUT, cut is the
 *              OUTPUT - cut is the first stage whose span divides wl, and tf
 *              is slaved to (wl > 0). Setting either by hand produces an
 *              illegal combination.
 *   rw         (real tier) the row route: an interleaved per-row door, or a
 *              SPLIT-layout child at width W. That race is the one bridge
 *              between the interleaved and split families in the whole
 *              library - see docs/design/planning_model.md Part IV.
 *   cmt        column MT. It EXISTS only when nb = N1/wl >= 2: with a single
 *              band there is nothing to distribute, so the race cannot run and
 *              cmt=0 is banked with the thread count it was decided at.
 *
 * That last one is why a 256x256 cell has no MT axis while 64x64 and
 * 1024x1024 do - wl lands on N1 there, giving one band.
 *
 * WHY cmt BANKS ITS THREAD COUNT (cmtt)
 * -------------------------------------
 * Column threading is the CORES-SHARE-ONE-TRANSFORM class: T decides how the
 * work is cut - band counts, worker clamps, legal row widths - so a cut that
 * wins at T=2 can lose at T=8. The verdict therefore carries the T it was
 * raced at, and a mismatch RE-RACES rather than serving a stale answer. (The
 * other MT class, one transform per core, banks T-free; nothing about those
 * plans depends on T.)
 *
 * ONE ASYMMETRY THAT LOOKS LIKE A BUG AND IS NOT
 * ----------------------------------------------
 * The row-route race is guarded with !il2d_oddn2; the column-MT guard eight
 * lines below is not. That is deliberate: an odd N2 has no ROWSPLIT arm to
 * race, but column threading stays valid. Measured consistent - 128x127 at T=8
 * engages cmt and is BIT-IDENTICAL to the single-threaded result.
 *
 * THE ENGAGEMENT COUNTER STAYS IN vfft.c
 * --------------------------------------
 * _vfft_il2d_col_mt_count is incremented from here but DEFINED there, with
 * external linkage (step 17 changed it from static for exactly this reason).
 * It cannot live in this header: a static in a header is one copy per
 * includer, and the public accessor would then read a different object than
 * the increment writes - reporting a confident zero while threading ran.
 *
 * INCLUSION CONTRACT
 * ------------------
 * Include after the engine prelude and after vfft_internal.h, as vfft.c does.
 */
#ifndef VFFT_TRANSFORMS_FFT2D_IL2D_TIER_H
#define VFFT_TRANSFORMS_FFT2D_IL2D_TIER_H

#include <stdlib.h>
#include <string.h>

#include "vfft_internal.h"                 /* struct vfft_plan_s / vfft_wisdom_s */
#include "il2d_cols.h"                     /* the column kernels + chain builders */
#include "fft2d_real_il.h"                 /* the real-tier row kernels */
#include "support/threads.h"               /* the pool */
#include "support/race_timing.h"           /* the shared clock */
#include "wisdom2/wisdom2_2d_reader.h"     /* the lay=il 2D cell codec */

/* Defined in vfft.c with external linkage; see the note above. */
extern long _vfft_il2d_col_mt_count;

/* one row of the row pass, by the plan's row route: in-place child
 * (default) or OOP child into the L1-hot scratch + copy back (the
 * small-N2 lever — the in-place K=1 IL service floor is ~6x the mono
 * math at tiny N). */
static void _il2d_row_exec(struct vfft_plan_s *h, vfft_dir_t dir,
                           double *row, size_t rn)
{
    if (h->il2d_rowoop)
    {
        vfft_execute(h->il2d_rowo, dir, row, NULL, h->il2d_rowscr, NULL);
        memcpy(row, h->il2d_rowscr, 2 * rn * sizeof(double));
    }
    else
        vfft_execute(h->il2d_row, dir, row, NULL, row, NULL);
}

/* ── native IL 2D REAL row passes (fft2d_real_il_design.md §2.4): ONE
 * function per direction, used by BOTH the execute and the create-time
 * row-route race (race == serving path). il2d_rows set = the ROWSPLIT
 * band route (transpose rows->lanes, split engine at (N2,K=rw),
 * fused transpose+zip back); NULL = the per-row TC door. */
static void _il2d_real_rows_fwd(struct vfft_plan_s *h, const double *sre,
                                double *dre)
{
    const size_t hp1 = (size_t)h->N2 / 2 + 1;
    if (h->il2d_oddn2)
    { /* odd N2: promote -> c2c(N2) -> keep the hp1 CCE bins */
        const size_t rn2 = (size_t)h->N2;
        double *b1 = h->il2d_orbuf, *b2 = h->il2d_orbuf + 2 * rn2;
        size_t r;
        for (r = 0; r < (size_t)h->N; r++)
        {
            _il2d_row_promote(sre + r * rn2, b1, rn2);
            vfft_execute((vfft_plan)h->il2d_row, VFFT_FORWARD, b1, NULL,
                         b2, NULL);
            memcpy(dre + r * 2 * hp1, b2, 2 * hp1 * sizeof(double));
        }
        return;
    }
    if (h->il2d_rows)
    {
        const int W2 = h->il2d_rw, rn2 = h->N2;
        size_t b;
        for (b = 0; b < (size_t)h->N / W2; b++)
        {
            const double *xb = sre + b * (size_t)W2 * rn2;
            double *zb = dre + b * (size_t)W2 * 2 * hp1;
            /* fused ROW-MODE door (r2c.h rowsplit fusion): rows in, rows
             * out, boundaries folded into the engine's own pack/store
             * passes. -1 = this plan can't serve it (non-stride path) —
             * the staged transpose route below stays the fallback. */
            if (!h->il2d_norowz && h->il2d_rows->rplan &&
                vfft_r2c_execute_fwd_rowz(h->il2d_rows->rplan, xb, rn2,
                                          zb, 2 * hp1) == 0)
                continue;
            if (!h->il2d_norowz && getenv("VFFT_IL2D_LOG"))
                fprintf(stderr, "[il2d-real] rowz fwd door FELL BACK "
                                "(staged route) at N2=%d W=%d\n",
                        rn2, W2);
            _vfft_k1_transpose(xb, h->il2d_lx, W2, rn2);
            vfft_execute(h->il2d_rows, VFFT_FORWARD, h->il2d_lx, NULL,
                         h->il2d_lre, h->il2d_lim);
            _il2d_transpose_zip(h->il2d_lre, h->il2d_lim, zb, W2,
                                (int)hp1);
        }
    }
    else
        vfft_execute(h->il2d_row, VFFT_FORWARD, (double *)sre, NULL,
                     dre, NULL);
}

static void _il2d_real_rows_bwd(struct vfft_plan_s *h, const double *zsrc,
                                double *dre)
{
    const size_t hp1 = (size_t)h->N2 / 2 + 1;
    if (h->il2d_oddn2)
    { /* odd N2: Hermitian-extend hp1 -> N2 -> inverse c2c -> Re. The
       * inverse is unnormalized (x N2), matching the even tier's c2r
       * scale contract. */
        const size_t rn2 = (size_t)h->N2;
        double *b1 = h->il2d_orbuf, *b2 = h->il2d_orbuf + 2 * rn2;
        size_t r;
        for (r = 0; r < (size_t)h->N; r++)
        {
            _il2d_row_extend(zsrc + r * 2 * hp1, b1, rn2, hp1);
            vfft_execute((vfft_plan)h->il2d_row, VFFT_BACKWARD, b1, NULL,
                         b2, NULL);
            _il2d_row_re(b2, dre + r * rn2, rn2);
        }
        return;
    }
    if (h->il2d_rows)
    {
        const int W2 = h->il2d_rw, rn2 = h->N2;
        size_t b;
        for (b = 0; b < (size_t)h->N / W2; b++)
        {
            const double *zs = zsrc + b * (size_t)W2 * 2 * hp1;
            double *xb = dre + b * (size_t)W2 * rn2;
            /* fused ROW-MODE door (mirror): unzip-once into the plan's
             * working planes, bwd without the split-door memcpys, hot
             * per-block transpose out. -1 = staged fallback below. */
            if (!h->il2d_norowz && h->il2d_rows->c2rdisp &&
                vfft_c2r_disp_execute_rowz(h->il2d_rows->c2rdisp, zs,
                                           2 * hp1, xb, rn2) == 0)
                continue;
            if (!h->il2d_norowz && getenv("VFFT_IL2D_LOG"))
                fprintf(stderr, "[il2d-real] rowz bwd door FELL BACK "
                                "(staged route) at N2=%d W=%d\n",
                        rn2, W2);
            /* fused de-zip+transpose reads FULL 4-wide e-blocks — legal
             * because zsrc is the tier's over-allocated rscr plane (+8
             * dbl pad at create; the c2r execute passes rscr here). */
            _il2d_unzip_transpose(zs, h->il2d_lre, h->il2d_lim, W2,
                                  (int)hp1);
            vfft_execute(h->il2d_rows, VFFT_BACKWARD, h->il2d_lre,
                         h->il2d_lim, h->il2d_lx, NULL);
            _vfft_k1_transpose(h->il2d_lx, xb, rn2, W2);
        }
    }
    else
        vfft_execute(h->il2d_row, VFFT_BACKWARD, (double *)zsrc, NULL,
                     dre, NULL);
}


/* ── native IL 2D REAL column pass (banded-aware; execute AND the wl
 * race serve through it — race == serving path). The banded walk is the
 * c2c cascade's column form MINUS the row fusion: §2.5 keeps rows
 * entirely OUTSIDE (fwd rows complete before any stage; bwd rows follow
 * the last stage), so a band is pure column loop interchange —
 * F0-bitwise vs unbanded. fwd = wide prefix 0..cut-1 full-plane, then
 * per band of wl rows the suffix depth-first; bwd (the Hermitian
 * transpose chain) = per band the REVERSED suffix (its first executed
 * stage does the OOP move for c2r's z->rscr), then the reversed prefix
 * in place on dst. */
static void _il2d_real_cols(struct vfft_plan_s *h, const double *src,
                            double *dst, int reverse)
{
    const size_t hp1 = (size_t)h->N2 / 2 + 1;
    if (h->il2d_nat)
    { /* NATURAL n1 (M4-lite): the leaf-redirected pass, unbanded by
       * construction (wl pinned 0 at create). */
        _il2d_col_pass_nat(src, dst, h->N, hp1, h->il2d_nst, h->il2d_R,
                           h->il2d_L,
                           reverse ? h->il2d_b : h->il2d_f,
                           reverse ? h->il2d_tb : h->il2d_tf, reverse,
                           h->il2d_natperm, h->il2d_natscr);
        return;
    }
    if (h->il2d_blu)
    { /* prime N1: the shared Bluestein pipeline over the CCE plane;
       * reverse = the inverse transform (conjugated chirp/kernel). */
        _il2d_blu_cols(src, dst, h->N, hp1, h->il2d_blu, h->il2d_nst,
                       h->il2d_R, h->il2d_L, h->il2d_f, h->il2d_b,
                       h->il2d_tf, h->il2d_tb,
                       reverse ? h->il2d_bluchb : h->il2d_bluchf,
                       reverse ? h->il2d_blukb : h->il2d_blukf,
                       h->il2d_bluscr);
        return;
    }
    if (h->il2d_wl > 0)
    {
        const int cut = h->il2d_cut, nst = h->il2d_nst;
        const size_t wl = (size_t)h->il2d_wl;
        vfft_il2p_fn const *fns = reverse ? h->il2d_b : h->il2d_f;
        double *const *tabs = reverse ? h->il2d_tb : h->il2d_tf;
        size_t b0;
        if (!reverse)
        {
            if (cut > 0)
                _il2d_col_stages(src, dst, h->N, hp1, 0, cut,
                                 h->il2d_R, h->il2d_L, fns, tabs, 0);
            for (b0 = 0; b0 < (size_t)h->N; b0 += wl)
            {
                const double *bs = (cut > 0) ? dst + 2 * b0 * hp1
                                             : src + 2 * b0 * hp1;
                _il2d_col_stages(bs, dst + 2 * b0 * hp1, (int)wl, hp1,
                                 cut, nst, h->il2d_R, h->il2d_L, fns,
                                 tabs, 0);
            }
        }
        else
        {
            for (b0 = 0; b0 < (size_t)h->N; b0 += wl)
                _il2d_col_stages(src + 2 * b0 * hp1,
                                 dst + 2 * b0 * hp1, (int)wl, hp1, cut,
                                 nst, h->il2d_R, h->il2d_L, fns, tabs,
                                 1);
            if (cut > 0)
                _il2d_col_stages(dst, dst, h->N, hp1, 0, cut,
                                 h->il2d_R, h->il2d_L, fns, tabs, 1);
        }
        return;
    }
    _il2d_col_pass(src, dst, h->N, hp1, 0, h->il2d_nst, h->il2d_R,
                   h->il2d_L, reverse ? h->il2d_b : h->il2d_f,
                   reverse ? h->il2d_tb : h->il2d_tf, reverse);
}

/* ══ MT column pass (INC-3, docs/design/il2d_real_mt.md) ═════════════
 * TWO partition arms, both pure loop restrictions of the SERVING loops
 * above (no arithmetic changes => MT == ST bitwise, gated):
 *   BAND arm  (wl > 0): workers take disjoint sets of wl-row BANDS of
 *     the suffix stages. Measured EXCHANGE-FREE (INC-2: reading rows
 *     you wrote scales 7.8-7.9x) because these are the same rows the
 *     row pass just produced. The wide prefix stages [0,cut) stay
 *     serial here — stage 0 spans the whole plane, and splitting it by
 *     DIGIT is INC-3b.
 *   STRIP arm (wl == 0): workers take disjoint COLUMN ranges and run
 *     the whole chain. This is the ONLY axis for single-stage chains
 *     (L[0] == N1 => one block, no row axis), and it pays the full
 *     cross-core exchange (INC-2: ~2.9-3.4x, not 8x). Boundaries are
 *     NOT rounded to cache lines: hp1 is always odd, so a row's start
 *     rotates and rounding neither removes the split lines nor pays
 *     for itself (it would collapse hp1=33 from 8 workers to 5). */
typedef struct
{
    struct vfft_plan_s *h;
    const double *src;
    double *dst;
    int reverse;
    size_t lo, hi;   /* band index range, or column range */
    int strip;
    int natleaf; /* natural x MT: this dispatch is the leaf block range */
} _il2d_cmt_arg;

/* ── INC-3b: the DIGIT axis of a wide (prefix) stage ─────────────────
 * A stage's kernel walks digits itself — per digit it advances
 * `twp += (R-1)*8` and `zin/zout += 2*Gs` (verified in the emitted
 * body) — so running only digits [d0, d0+nd) is THREE pointer edits:
 * base + 2*d0*pitch, table + d0*(R-1)*8, OGs = nd. Digit d owns rows
 * {b*L + d + j*D}, disjoint across d, WHOLE ROWS, `count` untouched,
 * and no new codelet. That makes the full-plane prefix stage — the
 * last serial chunk of the banded column pass — parallel over D. */
typedef struct
{
    const double *src;
    double *dst;
    size_t pitch, cnt;
    int nrows, R, L;
    vfft_il2p_fn fn;
    const double *tab;
    size_t d0, nd;
} _il2d_dmt_arg;

static void _il2d_dmt_tramp(void *v)
{
    _il2d_dmt_arg *a = (_il2d_dmt_arg *)v;
    const int D = a->L / a->R;
    int b;
    for (b = 0; b < a->nrows / a->L; b++)
    {
        const size_t off =
            2 * ((size_t)b * a->L * a->pitch + a->d0 * a->pitch);
        a->fn(a->src + off, NULL, a->dst + off, NULL,
              a->tab + a->d0 * (size_t)(a->R - 1) * 8, NULL,
              (size_t)D * a->pitch, a->pitch, (size_t)D * a->pitch,
              a->nd, a->cnt);
    }
}

/* Run ONE stage with its digits split across T workers. Returns 1 when
 * it threaded, 0 when the caller must run the stage serially. */
static int _il2d_stage_digits_mt(const double *src, double *dst,
                                 int nrows, size_t pitch, size_t cnt,
                                 int R, int L, vfft_il2p_fn fn,
                                 const double *tab, int T)
{
    const size_t D = (size_t)(L / R);
    _il2d_dmt_arg a[STRIDE_POOL_MAX_DISPATCH];
    int t;
    if (!tab || D < (size_t)T || T < 2)
        return 0; /* D == 1 stages carry no table and no digit axis */
    for (t = 0; t < T; t++)
    {
        a[t].src = src; a[t].dst = dst;
        a[t].pitch = pitch; a[t].cnt = cnt;
        a[t].nrows = nrows; a[t].R = R; a[t].L = L;
        a[t].fn = fn; a[t].tab = tab;
        a[t].d0 = D * (size_t)t / (size_t)T;
        a[t].nd = D * (size_t)(t + 1) / (size_t)T - a[t].d0;
    }
    stride_pool_run(T, _il2d_dmt_tramp, a, sizeof a[0]); /* caller = a[0] */
    return 1;
}

static void _il2d_cmt_tramp(void *v)
{
    _il2d_cmt_arg *a = (_il2d_cmt_arg *)v;
    struct vfft_plan_s *h = a->h;
    const size_t hp1 = (size_t)h->N2 / 2 + 1;
    vfft_il2p_fn const *fns = a->reverse ? h->il2d_b : h->il2d_f;
    double *const *tabs = a->reverse ? h->il2d_tb : h->il2d_tf;
    if (a->natleaf)
    {   /* natural x MT: the leaf scatter/gather over [lo,hi) blocks */
        _il2d_nat_leaf_range(a->src, a->dst, h->N, hp1,
                             h->il2d_R[h->il2d_nst - 1], fns[h->il2d_nst - 1],
                             h->il2d_natperm, a->lo, a->hi, a->reverse);
        return;
    }
    if (a->strip)
    {
        if (h->il2d_blu)
        {   /* Bluestein column axis: the window pipeline (2026-09-02) */
            _il2d_blu_cols_range(a->src, a->dst, h->N, hp1, a->lo, a->hi,
                                 h->il2d_blu, h->il2d_nst, h->il2d_R,
                                 h->il2d_L, h->il2d_f, h->il2d_b,
                                 h->il2d_tf, h->il2d_tb,
                                 a->reverse ? h->il2d_bluchb : h->il2d_bluchf,
                                 a->reverse ? h->il2d_blukb : h->il2d_blukf,
                                 h->il2d_bluscr);
            return;
        }
        _il2d_col_pass_range(a->src, a->dst, h->N, hp1, a->lo, a->hi,
                             h->il2d_nst, h->il2d_R, h->il2d_L, fns,
                             tabs, a->reverse);
        return;
    }
    {
        const size_t wl = (size_t)h->il2d_wl;
        size_t b;
        for (b = a->lo; b < a->hi; b++)
        {
            const size_t b0 = b * wl;
            const double *bs = a->src + 2 * b0 * hp1;
            _il2d_col_stages(bs, a->dst + 2 * b0 * hp1, (int)wl, hp1,
                             h->il2d_cut, h->il2d_nst, h->il2d_R,
                             h->il2d_L, fns, tabs, a->reverse);
        }
    }
}

/* Returns 1 when it ran threaded, 0 when the caller must run serial. */
static int _il2d_real_cols_mt(struct vfft_plan_s *h, const double *src,
                              double *dst, int reverse, int T)
{
    const size_t hp1 = (size_t)h->N2 / 2 + 1;
    const int strip = (h->il2d_wl <= 0);
    size_t units = strip ? hp1 : ((size_t)h->N / (size_t)h->il2d_wl);
    _il2d_cmt_arg a[STRIDE_POOL_MAX_DISPATCH];
    int t;
    /* T arrives as the plan's snapshot (h->nthreads); the pool's one clamp
     * bounds it by the live pool and the arg-array size. */
    T = stride_pool_workers_for(T);
    if (T >= 2 && h->il2d_nat)
    {
        /* NATURAL x MT (2026-09-04): the matched partition of the
         * natural pass — prefix stages digit-split (src -> scratch, then
         * in place), the leaf scatter by BLOCK RANGE (scratch -> dst),
         * mirrored for bwd (gather first, reversed prefix after, stage 0
         * scratch -> dst). No band arm: the scatter crosses bands. */
        const int Rl = h->il2d_R[h->il2d_nst - 1];
        const size_t nb = (size_t)h->N / (size_t)Rl;
        const int Tb = nb < (size_t)T ? (int)nb : T;
        double *scr = h->il2d_natscr;
        int s;
        if (Tb < 2 || h->il2d_nst < 2)
            return 0;
        if (!reverse)
        {
            for (s = 0; s < h->il2d_nst - 1; s++)
            {
                const double *ssrc = (s == 0) ? src : scr;
                if (!_il2d_stage_digits_mt(ssrc, scr, h->N, hp1, hp1,
                                           h->il2d_R[s], h->il2d_L[s],
                                           h->il2d_f[s], h->il2d_tf[s], T))
                    _il2d_col_stages(ssrc, scr, h->N, hp1, s, s + 1,
                                     h->il2d_R, h->il2d_L, h->il2d_f,
                                     h->il2d_tf, 0);
            }
        }
        for (t = 0; t < Tb; t++)
        {
            a[t].h = h;
            a[t].src = reverse ? src : scr;
            a[t].dst = reverse ? scr : dst;
            a[t].reverse = reverse;
            a[t].strip = 0;
            a[t].natleaf = 1;
            a[t].lo = nb * (size_t)t / (size_t)Tb;
            a[t].hi = nb * (size_t)(t + 1) / (size_t)Tb;
        }
        stride_pool_run(Tb, _il2d_cmt_tramp, a, sizeof a[0]);
        if (reverse)
        {
            for (s = h->il2d_nst - 2; s >= 0; s--)
            {
                double *out = (s == 0) ? dst : scr;
                if (!_il2d_stage_digits_mt(scr, out, h->N, hp1, hp1,
                                           h->il2d_R[s], h->il2d_L[s],
                                           h->il2d_b[s], h->il2d_tb[s], T))
                    _il2d_col_stages(scr, out, h->N, hp1, s, s + 1,
                                     h->il2d_R, h->il2d_L, h->il2d_b,
                                     h->il2d_tb, 0);
            }
        }
        _vfft_il2d_col_mt_count++;
        return 1;
    }
    if (T < 2 || units < (size_t)T)
        return 0; /* not enough independent units to be worth splitting */
    /* fwd: the wide prefix must complete before ANY band (stage 0's legs
     * span the whole plane). bwd: the reversed prefix runs after. */
    if (!strip && !reverse && h->il2d_cut > 0)
    {
        /* INC-3b: each prefix stage's DIGITS split over the workers
         * (whole rows, count untouched); stages stay ordered, one
         * dispatch each — a dispatch+wait is ~100 ns (INC-2), so the
         * per-stage join is free at these sizes. A stage that cannot
         * split (D < T, or the table-free D==1 leaf) runs serial. */
        int s;
        for (s = 0; s < h->il2d_cut; s++)
        {
            const double *ssrc = (s == 0) ? src : dst;
            if (!_il2d_stage_digits_mt(ssrc, dst, h->N, hp1, hp1,
                                       h->il2d_R[s], h->il2d_L[s],
                                       h->il2d_f[s], h->il2d_tf[s], T))
                _il2d_col_stages(ssrc, dst, h->N, hp1, s, s + 1,
                                 h->il2d_R, h->il2d_L, h->il2d_f,
                                 h->il2d_tf, 0);
        }
    }
    for (t = 0; t < T; t++)
    {
        a[t].h = h;
        /* after a fwd prefix the band source IS dst (in place) */
        a[t].src = (!strip && !reverse && h->il2d_cut > 0) ? dst : src;
        a[t].dst = dst;
        a[t].reverse = reverse;
        a[t].strip = strip;
        a[t].natleaf = 0;
        a[t].lo = units * (size_t)t / (size_t)T;
        a[t].hi = units * (size_t)(t + 1) / (size_t)T;
    }
    stride_pool_run(T, _il2d_cmt_tramp, a, sizeof a[0]); /* caller = a[0] */
    _vfft_il2d_col_mt_count++; /* engagement, see vfft.h */
    if (!strip && reverse && h->il2d_cut > 0)
    {
        /* the Hermitian-transpose chain: prefix stages in REVERSE order,
         * in place on dst, each digit-split the same way. */
        int s;
        for (s = h->il2d_cut - 1; s >= 0; s--)
            if (!_il2d_stage_digits_mt(dst, dst, h->N, hp1, hp1,
                                       h->il2d_R[s], h->il2d_L[s],
                                       h->il2d_b[s], h->il2d_tb[s], T))
                _il2d_col_stages(dst, dst, h->N, hp1, s, s + 1,
                                 h->il2d_R, h->il2d_L, h->il2d_b,
                                 h->il2d_tb, 0);
    }
    return 1;
}

/* ══ c2c MT (INC-C, the real tier's design ported) ═══════════════════
 * The structural difference from real: rows COMMUTE with column stages
 * (both C-linear — the same fact that makes tfuse legal here and banned
 * for real by §2.5). So a banded cell's unit of work is a SELF-CONTAINED
 * band [suffix stages + its own fused rows]: partition bands across
 * workers and there is no rows/columns wall and no cross-core exchange
 * for the fused part. Only the wide prefix needs the digit split.
 * Row execution mutates shared plan state (one child, one rowscr), so a
 * worker t > 0 runs its CLONE (il2d_roww[t-1], route-equivalence-checked
 * at build) and, on the rowoop route, its own rowscr slot. Every arm is
 * a loop restriction of the serving walk => MT == ST bitwise. */
static void _il2d_row_exec_t(struct vfft_plan_s *h, int tid,
                             vfft_dir_t dir, double *row, size_t rn)
{
    if (tid <= 0)
    {
        _il2d_row_exec(h, dir, row, rn);
        return;
    }
    {
        struct vfft_plan_s *c = h->il2d_roww[tid - 1];
        if (h->il2d_rowoop)
        {
            double *scr = h->il2d_rowscr_w + 2 * rn * (size_t)(tid - 1);
            vfft_execute((vfft_plan)c, dir, row, NULL, scr, NULL);
            memcpy(row, scr, 2 * rn * sizeof(double));
        }
        else
            vfft_execute((vfft_plan)c, dir, row, NULL, row, NULL);
    }
}

typedef struct
{
    struct vfft_plan_s *h;
    const double *src;
    double *dst;
    vfft_dir_t dir;
    int fwd;
    size_t lo, hi; /* band range (mode 0), column range (1), row range (2) */
    int mode, tid;
} _il2d_c2c_mt_arg;

static void _il2d_c2c_mt_tramp(void *v)
{
    _il2d_c2c_mt_arg *a = (_il2d_c2c_mt_arg *)v;
    struct vfft_plan_s *h = a->h;
    const size_t rn = (size_t)h->N2;
    vfft_il2p_fn const *fns = a->fwd ? h->il2d_f : h->il2d_b;
    double *const *tabs = a->fwd ? h->il2d_tf : h->il2d_tb;
    size_t i, b;
    switch (a->mode)
    {
    case 0: /* bands: suffix stages, then (tfuse) the band's rows —
             * rows LAST in the band in BOTH directions, the serving
             * order (bwd runs the reversed suffix via !fwd) */
        for (b = a->lo; b < a->hi; b++)
        {
            const size_t b0 = b * (size_t)h->il2d_wl;
            const double *bs = (a->fwd && h->il2d_cut > 0)
                                   ? a->dst + 2 * b0 * rn
                                   : a->src + 2 * b0 * rn;
            double *bd = a->dst + 2 * b0 * rn;
            _il2d_col_stages(bs, bd, h->il2d_wl, rn, h->il2d_cut,
                             h->il2d_nst, h->il2d_R, h->il2d_L, fns,
                             tabs, !a->fwd);
            if (h->il2d_tfuse)
                for (i = 0; i < (size_t)h->il2d_wl; i++)
                    _il2d_row_exec_t(h, a->tid, a->dir,
                                     bd + 2 * i * rn, rn);
        }
        break;
    case 1: /* column strip: the whole chain over [lo,hi) columns */
        _il2d_col_pass_range(a->src, a->dst, h->N, rn, a->lo, a->hi,
                             h->il2d_nst, h->il2d_R, h->il2d_L, fns,
                             tabs, !a->fwd);
        break;
    case 5: /* natural x MT, the STRIP arm: the whole natural pass over
             * [lo,hi) columns (shared scratch, disjoint columns) */
        _il2d_col_pass_nat_range(a->src, a->dst, h->N, rn, a->lo, a->hi,
                                 h->il2d_nst, h->il2d_R, h->il2d_L, fns,
                                 tabs, !a->fwd, h->il2d_natperm,
                                 h->il2d_natscr);
        break;
    case 4: /* natural x MT: the leaf scatter (fwd: src = scratch, dst =
             * plane) / gather (bwd: src = natural plane, dst = scratch)
             * over [lo,hi) blocks */
        _il2d_nat_leaf_range(a->src, a->dst, h->N, rn,
                             h->il2d_R[h->il2d_nst - 1], fns[h->il2d_nst - 1],
                             h->il2d_natperm, a->lo, a->hi, !a->fwd);
        break;
    case 3: /* Bluestein column axis: the window pipeline (2026-09-02) */
        _il2d_blu_cols_range(a->src, a->dst, h->N, rn, a->lo, a->hi,
                             h->il2d_blu, h->il2d_nst, h->il2d_R, h->il2d_L,
                             h->il2d_f, h->il2d_b, h->il2d_tf, h->il2d_tb,
                             a->fwd ? h->il2d_bluchf : h->il2d_bluchb,
                             a->fwd ? h->il2d_blukf : h->il2d_blukb,
                             h->il2d_bluscr);
        break;
    default: /* row slab on the destination plane */
        for (i = a->lo; i < a->hi; i++)
            _il2d_row_exec_t(h, a->tid, a->dir, a->dst + 2 * i * rn,
                             rn);
    }
}

/* Dispatch one phase across T workers (caller participates as tid 0). */
static void _il2d_c2c_mt_phase(struct vfft_plan_s *h, const double *src,
                               double *dst, vfft_dir_t dir, int fwd,
                               int mode, size_t units, int T)
{
    _il2d_c2c_mt_arg a[STRIDE_POOL_MAX_DISPATCH];
    int t;
    for (t = 0; t < T; t++)
    {
        a[t].h = h;
        a[t].src = src;
        a[t].dst = dst;
        a[t].dir = dir;
        a[t].fwd = fwd;
        a[t].mode = mode;
        a[t].tid = t;
        a[t].lo = units * (size_t)t / (size_t)T;
        a[t].hi = units * (size_t)(t + 1) / (size_t)T;
    }
    stride_pool_run(T, _il2d_c2c_mt_tramp, a, sizeof a[0]); /* caller = a[0] (tid 0) */
}

/* Returns 1 when it ran threaded, 0 when the caller must run serial. */
static int _il2d_c2c_mt(struct vfft_plan_s *h, const double *sre,
                        double *dre, vfft_dir_t dir, int T)
{
    const size_t rn = (size_t)h->N2;
    const int fwd = (dir == VFFT_FORWARD);
    int s;
    if (h->il2d_staged)
        return 0; /* env-experimental route: one shared band scratch —
                   * per-worker slots are not built for it */
    /* T arrives as the plan's snapshot (h->nthreads); the pool's one clamp
     * bounds it by the live pool and the arg-array size. */
    T = stride_pool_workers_for(T);
    if (T < 2 || h->il2d_roww_n < T - 1)
        return 0; /* every arm here runs rows => clones are mandatory */
    if (h->il2d_blu)
    {   /* Bluestein column axis (2026-09-02): column windows, then rows —
         * the same order the unbanded chain walk uses */
        const int Ts = rn < (size_t)T ? (int)rn : T;
        const int Tr = (size_t)h->N < (size_t)T ? h->N : T;
        if (Ts < 2 && Tr < 2)
            return 0;
        if (Ts >= 2)
            _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 3, rn, Ts);
        else
            _il2d_blu_cols(sre, dre, h->N, rn, h->il2d_blu, h->il2d_nst,
                           h->il2d_R, h->il2d_L, h->il2d_f, h->il2d_b,
                           h->il2d_tf, h->il2d_tb,
                           fwd ? h->il2d_bluchf : h->il2d_bluchb,
                           fwd ? h->il2d_blukf : h->il2d_blukb,
                           h->il2d_bluscr);
        _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 2, (size_t)h->N, Tr);
        _vfft_il2d_col_mt_count++;
        return 1;
    }
    if (h->il2d_nat)
    {
        /* NATURAL x MT (2026-09-04): digit-split prefix (sre -> scratch,
         * then in place), the leaf scatter by BLOCK RANGE (mode 4,
         * scratch -> dre), then row slabs on dre; bwd mirrors (gather
         * first, reversed prefix, stage 0 scratch -> dre). The band arm
         * is structurally out (the scatter crosses bands). */
        const int Rl = h->il2d_R[h->il2d_nst - 1];
        const size_t nb = (size_t)h->N / (size_t)Rl;
        const int Tb = nb < (size_t)T ? (int)nb : T;
        const int Tr = (size_t)h->N < (size_t)T ? h->N : T;
        double *scr = h->il2d_natscr;
        if (h->il2d_nst < 2 || (Tb < 2 && Tr < 2))
            return 0;
        if (h->il2d_natarm == 1)
        {   /* the STRIP arm (raced against the block arm at create) */
            const int Ts = rn < (size_t)T ? (int)rn : T;
            if (Ts < 2 && Tr < 2)
                return 0;
            if (Ts >= 2)
                _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 5, rn, Ts);
            else
                _il2d_col_pass_nat(sre, dre, h->N, rn, h->il2d_nst,
                                   h->il2d_R, h->il2d_L,
                                   fwd ? h->il2d_f : h->il2d_b,
                                   fwd ? h->il2d_tf : h->il2d_tb, !fwd,
                                   h->il2d_natperm, scr);
            _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 2, (size_t)h->N, Tr);
            _vfft_il2d_col_mt_count++;
            return 1;
        }
        if (fwd)
        {
            for (s = 0; s < h->il2d_nst - 1; s++)
            {
                const double *ssrc = (s == 0) ? sre : scr;
                if (!_il2d_stage_digits_mt(ssrc, scr, h->N, rn, rn,
                                           h->il2d_R[s], h->il2d_L[s],
                                           h->il2d_f[s], h->il2d_tf[s], T))
                    _il2d_col_stages(ssrc, scr, h->N, rn, s, s + 1,
                                     h->il2d_R, h->il2d_L, h->il2d_f,
                                     h->il2d_tf, 0);
            }
            if (Tb >= 2)
                _il2d_c2c_mt_phase(h, scr, dre, dir, fwd, 4, nb, Tb);
            else
                _il2d_nat_leaf_range(scr, dre, h->N, rn, Rl,
                                     h->il2d_f[h->il2d_nst - 1],
                                     h->il2d_natperm, 0, nb, 0);
        }
        else
        {
            if (Tb >= 2)
                _il2d_c2c_mt_phase(h, sre, scr, dir, fwd, 4, nb, Tb);
            else
                _il2d_nat_leaf_range(sre, scr, h->N, rn, Rl,
                                     h->il2d_b[h->il2d_nst - 1],
                                     h->il2d_natperm, 0, nb, 1);
            for (s = h->il2d_nst - 2; s >= 0; s--)
            {
                double *out = (s == 0) ? dre : scr;
                if (!_il2d_stage_digits_mt(scr, out, h->N, rn, rn,
                                           h->il2d_R[s], h->il2d_L[s],
                                           h->il2d_b[s], h->il2d_tb[s], T))
                    _il2d_col_stages(scr, out, h->N, rn, s, s + 1,
                                     h->il2d_R, h->il2d_L, h->il2d_b,
                                     h->il2d_tb, 0);
            }
        }
        _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 2, (size_t)h->N, Tr);
        _vfft_il2d_col_mt_count++;
        return 1;
    }
    if (h->il2d_wl > 0)
    {
        /* fewer bands than workers is a CLAMP, not a decline: 4 bands
         * across 4 workers still beats serial, and the prefix digit
         * split keeps the full T regardless (its axis is D, not nb). */
        const size_t nb = (size_t)h->N / (size_t)h->il2d_wl;
        const int Tb = nb < (size_t)T ? (int)nb : T;
        if (Tb < 2)
            return 0;
        if (fwd && h->il2d_cut > 0)
            for (s = 0; s < h->il2d_cut; s++)
            {
                const double *ssrc = (s == 0) ? sre : dre;
                if (!_il2d_stage_digits_mt(ssrc, dre, h->N, rn, rn,
                                           h->il2d_R[s], h->il2d_L[s],
                                           h->il2d_f[s], h->il2d_tf[s],
                                           T))
                    _il2d_col_stages(ssrc, dre, h->N, rn, s, s + 1,
                                     h->il2d_R, h->il2d_L, h->il2d_f,
                                     h->il2d_tf, 0);
            }
        _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 0, nb, Tb);
        if (!h->il2d_tfuse)
            _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 2, (size_t)h->N,
                               T);
        if (!fwd && h->il2d_cut > 0)
            for (s = h->il2d_cut - 1; s >= 0; s--)
                if (!_il2d_stage_digits_mt(dre, dre, h->N, rn, rn,
                                           h->il2d_R[s], h->il2d_L[s],
                                           h->il2d_b[s], h->il2d_tb[s],
                                           T))
                    _il2d_col_stages(dre, dre, h->N, rn, s, s + 1,
                                     h->il2d_R, h->il2d_L, h->il2d_b,
                                     h->il2d_tb, 0);
        _vfft_il2d_col_mt_count++; /* engagement, see vfft.h */
        return 1;
    }
    /* unbanded: column strips, then row slabs (rows follow the column
     * pass in the serving order for BOTH directions — rows commute) */
    {
        const int Ts = rn < (size_t)T ? (int)rn : T;
        const int Tr = (size_t)h->N < (size_t)T ? h->N : T;
        if (Ts < 2 && Tr < 2)
            return 0;
        if (Ts >= 2)
            _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 1, rn, Ts);
        else
            _il2d_col_pass(sre, dre, h->N, rn, rn, h->il2d_nst,
                           h->il2d_R, h->il2d_L,
                           fwd ? h->il2d_f : h->il2d_b,
                           fwd ? h->il2d_tf : h->il2d_tb, !fwd);
        _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 2, (size_t)h->N, Tr);
    }
    _vfft_il2d_col_mt_count++;
    return 1;
}

/* derive the banded walk's cut for a wl candidate: the first suffix
 * stage whose span divides wl. -1 = illegal (stay unbanded). */
static int _il2d_real_wl_cut(const struct vfft_plan_s *h, int wl)
{
    int s2;
    if (wl <= 0 || wl > h->N || h->N % wl != 0)
        return -1;
    for (s2 = 0; s2 < h->il2d_nst; s2++)
        if (wl % h->il2d_L[s2] == 0)
            return s2;
    return -1;
}

/* build one ROWSPLIT arm's engine + scratch (legality is the caller's:
 * W%8==0, W|N1, N2%4==0). Returns 1 on success with all six outputs
 * set; 0 with everything freed/NULL. */
static int _il2d_rowsplit_build(const vfft_config_t *cfg, int Wb, int N2,
                                struct vfft_plan_s **rows, double **lx,
                                double **lre, double **lim, double **tre,
                                double **tim)
{
    const int hp1i = N2 / 2 + 1;
    const int hp1p = (hp1i + 3) & ~3;
    vfft_config_t sc;
    memset(&sc, 0, sizeof sc);
    sc.transform = cfg->transform;
    sc.placement = VFFT_OUTOFPLACE;
    sc.rigor = cfg->rigor;
    sc.dims = 1;
    sc.n[0] = N2;
    sc.howmany = (size_t)Wb;
    sc.layout = VFFT_LAYOUT_SPLIT;
    sc.nthreads = 1;
    sc.wisdom = cfg->wisdom;
    sc.wisdom_write = cfg->wisdom_write;
    *rows = (struct vfft_plan_s *)vfft_create(&sc);
    if (!*rows)
        return 0;
    *lx = (double *)malloc((size_t)N2 * Wb * sizeof(double));
    *lre = (double *)malloc((size_t)hp1p * Wb * sizeof(double));
    *lim = (double *)malloc((size_t)hp1p * Wb * sizeof(double));
    *tre = NULL; /* fused boundaries (transpose_zip/unzip_transpose) —  */
    *tim = NULL; /* the row-major staging halves are GONE               */
    if (*lx && *lre && *lim)
    {
        memset(*lre, 0, (size_t)hp1p * Wb * sizeof(double));
        memset(*lim, 0, (size_t)hp1p * Wb * sizeof(double));
        return 1;
    }
    vfft_destroy(*rows);
    free(*lx); free(*lre); free(*lim);
    *rows = NULL;
    *lx = *lre = *lim = NULL;
    return 0;
}

/* ── the ROW-ROUTE race (owner 2026-08-26): per-row TC door vs ROWSPLIT
 * over the legal W pool, timed on the SAME row-pass helpers execute
 * serves with, min-of-3 each on scratch planes (r2c/c2r read-only
 * inputs — no compounding, no refills). Winner installed on h + banked
 * (chain + rw) in the direction-shared lay=il real cell. MUST run after
 * the h-> field commits (it executes h — the axis-race law). */
/* ── the arms of the il2d races (support/race.h): one context, the same
 * functions execute serves with. For the column-MT race the threaded arm
 * reports whether it could engage; the race runs to completion either way
 * and the site banks the "no" afterwards. */
typedef struct
{
    struct vfft_plan_s *h;
    double *a, *z;          /* real plane / complex scratch (in place) */
    int isr;                /* rows: r2c fwd (a -> z) or c2r bwd (z -> a) */
    int ok;                 /* colmt: the threaded arm engaged */
    /* chain candidate (the chain race) */
    int N1;
    size_t N2;
    int nst;
    const int *R;
    int *Ls;
    vfft_il2p_fn *ff;
    double **tf;
    /* the chain race under NATURAL order (2026-09-04): the candidate is
     * timed through the M4-lite leaf-redirected pass with ITS OWN perm —
     * the best chain for scrambled and for natural can differ (the
     * leaf radix sets the scatter width), so natural cells race under
     * the pass they serve and bank under their own ord=nat row. */
    int nat;
    const int *perm;
    double *nscr;
} _il2d_race_ctx_t;
static void _il2d_arm_cols(void *v)
{
    _il2d_race_ctx_t *c = (_il2d_race_ctx_t *)v;
    _il2d_real_cols(c->h, c->z, c->z, 0);
}
static void _il2d_arm_cols_mt(void *v)
{
    _il2d_race_ctx_t *c = (_il2d_race_ctx_t *)v;
    if (c->ok && !_il2d_real_cols_mt(c->h, c->z, c->z, 0, c->h->nthreads))
        c->ok = 0; /* the threaded arm cannot engage on this cell */
}
static void _il2d_arm_rows(void *v)
{
    _il2d_race_ctx_t *c = (_il2d_race_ctx_t *)v;
    if (c->isr)
        _il2d_real_rows_fwd(c->h, c->a, c->z);
    else
        _il2d_real_rows_bwd(c->h, c->z, c->a);
}
static void _il2d_arm_exec_st(void *v)
{
    _il2d_race_ctx_t *c = (_il2d_race_ctx_t *)v;
    c->h->il2d_colmt = 0;
    vfft_execute((vfft_plan)c->h, VFFT_FORWARD, c->z, NULL, c->z, NULL);
}
static void _il2d_arm_exec_mt(void *v)
{
    _il2d_race_ctx_t *c = (_il2d_race_ctx_t *)v;
    c->h->il2d_colmt = 1;
    c->h->il2d_natarm = 0;
    vfft_execute((vfft_plan)c->h, VFFT_FORWARD, c->z, NULL, c->z, NULL);
}
static void _il2d_arm_exec_mt_strip(void *v)
{   /* natural cells only: the strip partition of the natural pass */
    _il2d_race_ctx_t *c = (_il2d_race_ctx_t *)v;
    c->h->il2d_colmt = 1;
    c->h->il2d_natarm = 1;
    vfft_execute((vfft_plan)c->h, VFFT_FORWARD, c->z, NULL, c->z, NULL);
}
static void _il2d_arm_exec(void *v)
{
    _il2d_race_ctx_t *c = (_il2d_race_ctx_t *)v;
    vfft_execute((vfft_plan)c->h, VFFT_FORWARD, c->z, NULL, c->z, NULL);
}
static void _il2d_arm_chain(void *v)
{
    _il2d_race_ctx_t *c = (_il2d_race_ctx_t *)v;
    if (c->nat)
        _il2d_col_pass_nat(c->z, c->z, c->N1, c->N2, c->nst, c->R, c->Ls,
                           c->ff, c->tf, /*reverse=*/0, c->perm, c->nscr);
    else
        _il2d_col_pass(c->z, c->z, c->N1, c->N2, 0, c->nst, c->R, c->Ls,
                       c->ff, c->tf, /*reverse=*/0);
}
static void _il2d_real_rowrace(struct vfft_plan_s *h,
                               struct vfft_wisdom_s *W,
                               const vfft_config_t *cfg, int N1, int N2)
{
    static const int POOL[] = { 32, 64, 128, 256 };
    const size_t RN = (size_t)N1 * N2;
    const size_t CN = (size_t)N1 * ((size_t)N2 / 2 + 1);
    const int isr = (h->transform == VFFT_R2C);
    double *a = (double *)malloc(RN * sizeof(double));
    double *bz = (double *)malloc((2 * CN + 8) * sizeof(double));
    /* +8: the fused c2r unzip reads past the last row's tail (rscr law) */
    double bestns = 1e300;
    int bw = 0, pi, p;
    size_t i;
    /* current best's resources (arm 0 = the per-row door: all NULL) */
    struct vfft_plan_s *brows = NULL;
    double *blx = NULL, *blre = NULL, *blim = NULL;
    double *btre = NULL, *btim = NULL;
    if (!a || !bz)
    {
        free(a);
        free(bz);
        return;
    }
    for (i = 0; i < RN; i++)
        a[i] = 1.0 + 1e-6 * (double)(i & 1023);
    for (i = 0; i < 2 * CN + 8; i++)
        bz[i] = 1.0 + 1e-6 * (double)(i & 511);
    /* arm 0: the per-row TC door */
    _il2d_race_ctx_t rc = { h, a, bz, isr, 1, 0, 0, 0, NULL, NULL, NULL, NULL };
    const vfft_race_arm_t rows_arm = { "rows", _il2d_arm_rows, &rc };
    const vfft_race_arm_t cols_arm = { "cols", _il2d_arm_cols, &rc };
    const vfft_race_proto_t proto = { 3, 1, VFFT_RACE_MIN, 0, 0, NULL, NULL }; /* min-of-3, A then B */
    (void)p;
    h->il2d_rows = NULL;
    h->il2d_rw = 0;
    vfft_race_run(&proto, &rows_arm, 1, &bestns);
    for (pi = 0; pi < 4; pi++)
    {
        const int Wb = POOL[pi];
        struct vfft_plan_s *rows = NULL;
        double *lx = NULL, *lre = NULL, *lim = NULL;
        double *tre = NULL, *tim = NULL;
        double ns = 1e300;
        if (Wb > N1 || N1 % Wb != 0 || (N2 % 4) != 0)
            continue;
        if (!_il2d_rowsplit_build(cfg, Wb, N2, &rows, &lx, &lre, &lim,
                                  &tre, &tim))
            continue;
        h->il2d_rows = rows;
        h->il2d_rw = Wb;
        h->il2d_lx = lx;
        h->il2d_lre = lre;
        h->il2d_lim = lim;
        h->il2d_tre = tre;
        h->il2d_tim = tim;
        vfft_race_run(&proto, &rows_arm, 1, &ns);
        if (ns < bestns)
        {
            if (brows)
            {
                vfft_destroy(brows);
                free(blx); free(blre); free(blim);
                free(btre); free(btim);
            }
            bestns = ns;
            bw = Wb;
            brows = rows;
            blx = lx; blre = lre; blim = lim;
            btre = tre; btim = tim;
        }
        else
        {
            vfft_destroy(rows);
            free(lx); free(lre); free(lim); free(tre); free(tim);
        }
    }
    /* install the winner (NULLs = the per-row door) */
    h->il2d_rows = brows;
    h->il2d_rw = bw;
    h->il2d_lx = blx;
    h->il2d_lre = blre;
    h->il2d_lim = blim;
    h->il2d_tre = btre;
    h->il2d_tim = btim;
    /* ── the wl axis (the banded column walk; rows stay OUTSIDE per
     * §2.5): unbanded arm + the static pool + L2-admitted stage spans
     * (the c2c lever — row width here is hp1 complex), timed on the
     * column pass alone (the only thing wl changes), min-of-3 in place
     * on the z scratch (compounding is benign — the c2c chain-race
     * precedent). */
    {
        static const int WPOOL[] = { 8, 16, 32, 64, 128, 256 };
        const size_t hp1 = (size_t)N2 / 2 + 1;
        int wlc[14], nwl = 0, wi, s2;
        double cbest = 1e300;
        int bwl = 0, bcut = 0;
        h->il2d_wl = 0;
        h->il2d_cut = 0;
        vfft_race_run(&proto, &cols_arm, 1, &cbest);
        for (wi = 0; wi < 6 && nwl < 14; wi++)
            if (_il2d_real_wl_cut(h, WPOOL[wi]) >= 0 && WPOOL[wi] < N1)
                wlc[nwl++] = WPOOL[wi];
        for (s2 = 1; s2 < h->il2d_nst && nwl < 14; s2++)
        {
            const int w2 = h->il2d_L[s2];
            int dup = 0;
            if ((long)w2 * (long)hp1 * 16 > vfft_cpu_l2_bytes())
                continue;
            if (_il2d_real_wl_cut(h, w2) < 0 || w2 >= N1)
                continue;
            for (wi = 0; wi < nwl; wi++)
                if (wlc[wi] == w2)
                    dup = 1;
            if (!dup)
                wlc[nwl++] = w2;
        }
        for (wi = 0; wi < nwl; wi++)
        {
            const int cut = _il2d_real_wl_cut(h, wlc[wi]);
            double ns = 1e300;
            h->il2d_wl = wlc[wi];
            h->il2d_cut = cut;
            vfft_race_run(&proto, &cols_arm, 1, &ns);
            if (ns < cbest)
            {
                cbest = ns;
                bwl = wlc[wi];
                bcut = cut;
            }
        }
        h->il2d_wl = bwl;
        h->il2d_cut = bcut;
        free(a);
        free(bz);
        if (getenv("VFFT_IL2D_LOG"))
            fprintf(stderr, "[il2d-real] rowrace %s %dx%d -> rw=%d "
                            "wl=%d (%.0f ns rows / %.0f ns cols)\n",
                    isr ? "r2c" : "c2r", N1, N2, bw, bwl, bestns,
                    cbest);
        vw2_2d_rl_bank(&W->vw2, N1, N2, !isr, h->il2d_R, h->il2d_nst, bw,
                       bwl, -1, -1, (N1 & (N1 - 1)) ? h->il2d_blu : -1,
                       bestns + cbest, (h->il2d_nat ? VW2_ORD_NAT : VW2_ORD_SCR));
        _vw2_persist(W, cfg);
    }
}

/* ── INC-3: the COLUMN-MT verdict race. Times the column pass SERIAL vs
 * THREADED through the very functions the execute serves with (race ==
 * serving path), min-of-3 on a scratch plane, and banks {cmt, cmtt} in
 * the cell's lay=il real row. There is NO structural default and no
 * invented floor: at 512x32 (hp1=17 => the strip arm over 17 columns)
 * threading the column pass MEASURED SLOWER, and that "no" is banked
 * exactly like a "yes". A verdict is only served back at the SAME
 * thread count it was raced at (cmtt) — a T=4 verdict never serves a
 * T=8 request. MUST run after the plan's stage arrays are committed. */
static void _il2d_real_colmt_race(struct vfft_plan_s *h,
                                  struct vfft_wisdom_s *W,
                                  const vfft_config_t *cfg, int N1,
                                  int N2)
{
    const size_t hp1 = (size_t)N2 / 2 + 1;
    const size_t CN = (size_t)N1 * hp1;
    double *z = (double *)malloc((2 * CN + 8) * sizeof(double));
    double st = 1e300, mt = 1e300;
    int p;
    size_t i;
    if (!z)
        return;
    for (i = 0; i < 2 * CN + 8; i++)
        z[i] = 1.0 + 1e-6 * (double)(i & 511);
    {
        _il2d_race_ctx_t rc = { h, NULL, z, 0, 1, 0, 0, 0, NULL, NULL, NULL, NULL };
        const vfft_race_arm_t arms[2] = { { "serial", _il2d_arm_cols, &rc },
                                          { "threaded", _il2d_arm_cols_mt, &rc } };
        const vfft_race_proto_t proto = { 3, 1, VFFT_RACE_MIN, 0, 0, NULL, NULL }; /* min-of-3, A then B */
        double ns[2];
        (void)p;
        vfft_race_run(&proto, arms, 2, ns);
        st = ns[0];
        mt = ns[1];
        if (!rc.ok)
        {
            /* the threaded arm cannot even engage on this cell */
            free(z);
            h->il2d_colmt = 0;
            vw2_2d_rl_bank(&W->vw2, N1, N2, h->transform == VFFT_C2R,
                           h->il2d_R, h->il2d_nst,
                           h->il2d_rw, h->il2d_wl, 0, h->nthreads,
                           (N1 & (N1 - 1)) ? h->il2d_blu : -1, st, (h->il2d_nat ? VW2_ORD_NAT : VW2_ORD_SCR));
            _vw2_persist(W, cfg);
            return;
        }
    }
    h->il2d_colmt = (mt < st);
    free(z);
    if (getenv("VFFT_IL2D_LOG"))
        fprintf(stderr, "[il2d-real] colmt race %dx%d T=%d: st=%.0f "
                        "mt=%.0f -> %s\n",
                N1, N2, h->nthreads, st, mt,
                h->il2d_colmt ? "THREADED" : "serial");
    vw2_2d_rl_bank(&W->vw2, N1, N2, h->transform == VFFT_C2R,
                   h->il2d_R, h->il2d_nst, h->il2d_rw,
                   h->il2d_wl, h->il2d_colmt, h->nthreads,
                   (N1 & (N1 - 1)) ? h->il2d_blu : -1,
                   h->il2d_colmt ? mt : st, (h->il2d_nat ? VW2_ORD_NAT : VW2_ORD_SCR));
    _vw2_persist(W, cfg);
}


/* the chain RACE: time every candidate's column pass on scratch (min of
 * 3 passes), return the winner's index. -1 = race impossible. */
/* the Bluestein inner's chain at M: the (M, N2) 2D chain row — replayed
 * when banked (prod == M), else the E1.1 chain race at (M, N2), banked
 * there. Context = the create in progress (set at _vfft_create_2d's entry). */
static struct { struct vfft_wisdom_s *W; const vfft_config_t *cfg; int N2; } _il2d_blu_ctx;
static int _il2d_race_chains(int N1, int N2, int ncand, int (*cand)[8],
                             const int *lens, double *best_ns, int nat);
static int _il2d_race_forms(int N1, int N2, const int *Rs, int nst,
                            vfft_il2p_fn *ff, vfft_il2p_fn *fb, char *forms,
                            size_t fsz);
static void _il2d_forms_serve(struct vfft_wisdom_s *W,
                              const vfft_config_t *cfg, int is_real, int N1,
                              int N2, const int *Rs, int nst,
                              vfft_il2p_fn *ff, vfft_il2p_fn *fb,
                              char *forms, size_t fsz, int ord);
static int _il2d_blu_m_chain(int M, int *Rs, int *nst, char *forms,
                             size_t fsz)
{
    struct vfft_wisdom_s *W = _il2d_blu_ctx.W;
    const vfft_config_t *cfg = _il2d_blu_ctx.cfg;
    const int N2 = _il2d_blu_ctx.N2;
    int wl, tf, ro, cmt, cmtt, blu;
    vfft_il2p_fn ff[8], fb[8];
    forms[0] = 0;
    if (!W || W->vw2_off_2d || N2 <= 0) return 0;
    if (!cfg->recalibrate &&
        vw2_2d_il_chain_lookup(&W->vw2, M, N2, Rs, nst, &wl, &tf, &ro,
                               &cmt, &cmtt, &blu, VW2_ORD_SCR) &&
        _il2d_chain_prod(Rs, *nst) == M)
    {
        if (getenv("VFFT_IL2D_LOG"))
            fprintf(stderr, "[il2d] blu inner M=%d x %d: replay chain src=wisdom\n", M, N2);
        if (_il2d_resolve(Rs, *nst, ff, fb))
            _il2d_forms_serve(W, cfg, 0, M, N2, Rs, *nst, ff, fb, forms, fsz, VW2_ORD_SCR);
        return 1;
    }
    {
        int cand[VFFT_IL2D_MAXCAND][8], lens[VFFT_IL2D_MAXCAND];
        int cur[8], ncand = 0, dropped = 0, win;
        double bns = 0;
        _il2d_enum_rec(M, 0, cur, cand, lens, &ncand, &dropped);
        if (dropped)
            _vfft_warn("il2d blu inner chain race: pool capped at %d "
                       "(%d candidate(s) dropped) at M=%d x %d",
                       VFFT_IL2D_MAXCAND, dropped, M, N2);
        if (ncand < 1) return 0;
        win = (ncand > 1) ? _il2d_race_chains(M, N2, ncand, cand, lens, &bns, 0) : 0;
        if (win < 0) return 0;
        memcpy(Rs, cand[win], sizeof cand[win]);
        *nst = lens[win];
        if (getenv("VFFT_IL2D_LOG"))
            fprintf(stderr, "[il2d] blu inner M=%d x %d: chain race -> %d candidates, "
                            "winner banked\n", M, N2, ncand);
        vw2_2d_il_chain_bank(&W->vw2, M, N2, Rs, *nst, -1, -1, -1, -1, -1, -1, bns, VW2_ORD_SCR);
        _vw2_persist(W, cfg);
        if (_il2d_resolve(Rs, *nst, ff, fb))
            _il2d_forms_serve(W, cfg, 0, M, N2, Rs, *nst, ff, fb, forms, fsz, VW2_ORD_SCR);
        return 1;
    }
}

/* PER-STAGE FORM RACE (E1.11, 2026-09-02): coordinate descent over the
 * stages whose radix has rival forms (vfft_il2p_col_forms), each stage's
 * two arms timed on the WHOLE column pass with the other stages held at
 * their current pick (the pass is the only thing the form changes; same
 * harness as the chain race). The construction-table default is the
 * incumbent and keeps ties; the rival must beat it by 3%. Installs the
 * winners into ff/fb and spells them into `forms`. Returns 1 when any
 * stage had a choice, 0 otherwise (forms = ""). */
static int _il2d_race_forms(int N1, int N2, const int *Rs, int nst,
                            vfft_il2p_fn *ff, vfft_il2p_fn *fb, char *forms,
                            size_t fsz)
{
    const size_t T = (size_t)N1 * N2;
    const char *pick[8];
    int Ls[8], s, any = 0, off = 0;
    double *tf[8], *tb[8], *z;
    size_t i;
    forms[0] = 0;
    for (s = 0; s < nst; s++)
    {
        const char *nm[2];
        (void)vfft_il2p_col_forms(Rs[s], nm);
        pick[s] = nm[0];
        if (nm[1])
            any = 1;
    }
    if (!any)
        return 0;
    z = (double *)malloc(2 * T * sizeof(double));
    if (!z)
        return 0;
    for (i = 0; i < 2 * T; i++)
        z[i] = 1.0 + 1e-6 * (double)(i & 1023);
    if (_il2d_build_tables(N1, nst, Rs, Ls, tf, tb))
    {
        free(z);
        return 0;
    }
    for (s = 0; s < nst; s++)
    {
        const char *nm[2];
        const int last = (s == nst - 1);
        vfft_il2p_fn ffa[2][8];
        _il2d_race_ctx_t rc[2];
        vfft_race_arm_t arm[2];
        const vfft_race_proto_t proto = { 3, 1, VFFT_RACE_MIN, 0, 0, NULL, NULL };
        double ns[2] = { 1e300, 1e300 };
        int a, win;
        if (vfft_il2p_col_forms(Rs[s], nm) < 2)
            continue;
        for (a = 0; a < 2; a++)
        {
            memcpy(ffa[a], ff, sizeof ffa[a]);
            ffa[a][s] = last ? vfft_il2p_n1c_form_fn(Rs[s], nm[a], 0)
                             : vfft_il2p_t2c_form_fn(Rs[s], nm[a], 0);
            if (!ffa[a][s])
                break;
            memset(&rc[a], 0, sizeof rc[a]);
            rc[a].z = z;
            rc[a].ok = 1;
            rc[a].N1 = N1;
            rc[a].N2 = (size_t)N2;
            rc[a].nst = nst;
            rc[a].R = Rs;
            rc[a].Ls = Ls;
            rc[a].ff = ffa[a];
            rc[a].tf = tf;
            arm[a].name = nm[a];
            arm[a].run = _il2d_arm_chain;
            arm[a].ctx = &rc[a];
        }
        if (a < 2)
            continue;
        vfft_race_run(&proto, arm, 2, ns);
        win = (ns[1] < 0.97 * ns[0]) ? 1 : 0;
        pick[s] = nm[win];
        ff[s] = ffa[win][s];
        fb[s] = last ? vfft_il2p_n1c_form_fn(Rs[s], nm[win], 1)
                     : vfft_il2p_t2c_form_fn(Rs[s], nm[win], 1);
        if (getenv("VFFT_IL2D_LOG"))
            fprintf(stderr, "[il2d] forms %dx%d stage %d r%d: %s %.0f ns vs %s %.0f ns -> %s\n",
                    N1, N2, s, Rs[s], nm[0], ns[0], nm[1], ns[1], nm[win]);
    }
    for (s = 0; s < nst; s++)
    {
        free(tf[s]);
        free(tb[s]);
    }
    free(z);
    for (s = 0; s < nst && off < (int)fsz - 8; s++)
        off += snprintf(forms + off, fsz - off, "%s%s", s ? "." : "", pick[s]);
    return 1;
}

/* the FORM axis at create: env pin > the banked forms= on the chain row >
 * the per-stage race, banked on that row (wisdom off / no choice: the
 * construction-table defaults stand). ff/fb come in resolved (defaults). */
static void _il2d_forms_serve(struct vfft_wisdom_s *W,
                              const vfft_config_t *cfg, int is_real, int N1,
                              int N2, const int *Rs, int nst,
                              vfft_il2p_fn *ff, vfft_il2p_fn *fb,
                              char *forms, size_t fsz, int ord)
{
    const char *pin = getenv("VFFT_IL2D_FORMS");
    int s, any = 0;
    forms[0] = 0;
    for (s = 0; s < nst; s++)
        if (Rs[s] == 32 || Rs[s] == 64)
            any = 1;
    if (!any)
        return;
    if (pin && *pin)
    {
        if (_il2d_apply_forms(Rs, nst, pin, ff, fb))
            snprintf(forms, fsz, "%s", pin);
        else
            _vfft_warn("VFFT_IL2D_FORMS=%s does not fit chain at %dx%d - ignored",
                       pin, N1, N2);
        return;
    }
    if (!W || W->vw2_off_2d)
        return;
    if (!cfg->recalibrate &&
        vw2_2d_forms_lookup(&W->vw2, is_real, N1, N2, forms, fsz, ord))
    {
        if (_il2d_apply_forms(Rs, nst, forms, ff, fb))
        {
            if (getenv("VFFT_IL2D_LOG"))
                fprintf(stderr, "[il2d] forms %dx%d: replay %s src=wisdom\n", N1, N2, forms);
            return;
        }
        _vfft_warn("banked forms=%s does not fit chain at %dx%d - re-racing",
                   forms, N1, N2);
        (void)_il2d_resolve(Rs, nst, ff, fb);
    }
    if (_il2d_race_forms(N1, N2, Rs, nst, ff, fb, forms, fsz) && forms[0])
    {
        const int banked = vw2_2d_forms_bank(&W->vw2, is_real, N1, N2, forms, ord);
        if (banked)
            _vw2_persist(W, cfg);
        if (getenv("VFFT_IL2D_LOG"))
            fprintf(stderr, "[il2d] forms %dx%d: raced -> %s, %s\n", N1, N2, forms,
                    banked ? "banked" : "NOT banked yet (no chain row; the create re-banks once it lands)");
    }
}

static int _il2d_race_chains(int N1, int N2, int ncand, int (*cand)[8],
                             const int *lens, double *best_ns, int nat)
{
    const size_t T = (size_t)N1 * N2;
    double *z = (double *)malloc(2 * T * sizeof(double));
    double *nscr = nat ? (double *)malloc(2 * T * sizeof(double)) : NULL;
    int ci, win = -1;
    double wns = 1e300;
    size_t i;
    if (!z || (nat && !nscr))
    {
        free(z);
        free(nscr);
        return -1;
    }
    for (i = 0; i < 2 * T; i++)
        z[i] = 1.0 + 1e-6 * (double)(i & 1023);
    for (ci = 0; ci < ncand; ci++)
    {
        vfft_il2p_fn ff[8], fb[8];
        int Ls[8];
        double *tf[8], *tb[8];
        double ns = 1e300;
        int *perm = NULL;
        int p, s2;
        if (!_il2d_resolve(cand[ci], lens[ci], ff, fb))
            continue;
        if (nat && lens[ci] > 1)
        {
            perm = _il2d_nat_perm(cand[ci], lens[ci], N1);
            if (!perm)
                continue; /* no natural leaf for this chain: not a candidate */
        }
        if (_il2d_build_tables(N1, lens[ci], cand[ci], Ls, tf, tb))
        {
            free(perm);
            continue;
        }
        {
            _il2d_race_ctx_t rc = { NULL, NULL, z, 0, 1, N1, (size_t)N2,
                                    lens[ci], cand[ci], Ls, ff, tf,
                                    nat && perm != NULL, perm, nscr };
            const vfft_race_arm_t arm = { "chain", _il2d_arm_chain, &rc };
            const vfft_race_proto_t proto = { 3, 1, VFFT_RACE_MIN, 0, 0, NULL, NULL }; /* min-of-3, A then B */
            (void)p;
            vfft_race_run(&proto, &arm, 1, &ns);
        }
        for (s2 = 0; s2 < lens[ci]; s2++)
        {
            free(tf[s2]);
            free(tb[s2]);
        }
        free(perm);
        if (ns < wns)
        {
            wns = ns;
            win = ci;
        }
    }
    free(z);
    free(nscr);
    *best_ns = wns;
    return win;
}



/* the §10a axis race: time the FULL execute (column chain + rows) over
 * the wl candidates x the row routes, set the winner on the plan, bank
 * chain+wl+tf+ro as one lay=il verdict. Falsifier-grounded: wl wins
 * +15-21% at some cells and LOSES at others; rowoop wins 1.6-2x at
 * N2<=64 and loses at large N2 — per-cell only, never defaults. */
static void _il2d_axis_race(struct vfft_plan_s *h, struct vfft_wisdom_s *W,
                            const vfft_config_t *cfg, int N1, int N2)
{
    const size_t T = (size_t)N1 * N2;
    double *z = (double *)malloc(2 * T * sizeof(double));
    struct vfft_plan_s *rowo = NULL;
    double *rowscr = NULL;
    int wlc[14], nwl = 1, wi, ro, bwl = 0, bro = 0;
    double best = 1e300;
    size_t i;
    int reps = (int)(1e6 / (double)(T + 1));
    if (reps < 2) reps = 2;
    if (!z)
        return;
    for (i = 0; i < 2 * T; i++)
        z[i] = 1.0 + 1e-6 * (double)(i & 1023);
    /* wl candidates: 0 (unbanded) + legal widths */
    wlc[0] = 0;
    {
        static const int WPOOL[] = { 8, 16, 32, 64, 128, 256 };
        int p, s2;
        for (p = 0; p < 6 && nwl < 14; p++)
        {
            const int w = WPOOL[p];
            int cut = -1;
            if (w > N1 || N1 % w)
                continue;
            for (s2 = 0; s2 < h->il2d_nst; s2++)
                if (w % h->il2d_L[s2] == 0)
                {
                    cut = s2;
                    break;
                }
            if (cut >= 0)
                wlc[nwl++] = w;
        }
        /* the CASCADE widths (2026-08-25, owner-funded 2D cascade arc):
         * at huge N1 the static pool tops out at 256, pinning cut deep —
         * MULTIPLE wide stages stream the full plane per execute (the
         * measured L2 knee: per-point 1.9x off the memory floor at
         * 32768x64 while the memcpy floor moved 1.3x). The stage spans
         * L[s] themselves are the natural band widths: wl == L[s] pulls
         * every stage below s into the L2-resident depth-first suffix,
         * leaving s wide passes. Gate = live band residency
         * (w * N2 * 16 <= vfft_cpu_l2_bytes(), the hardware-derived
         * fence — never a platform-baked constant), and the RACE still
         * decides: these are candidates, not defaults. */
        for (s2 = 1; s2 < h->il2d_nst && nwl < 14; s2++)
        {
            const int w = h->il2d_L[s2];
            int dup = 0, p2;
            if (w > N1 || N1 % w || w < 8)
                continue;
            if ((long)w * N2 * 16 > vfft_cpu_l2_bytes())
                continue;
            for (p2 = 0; p2 < nwl; p2++)
                if (wlc[p2] == w)
                    dup = 1;
            if (!dup)
                wlc[nwl++] = w;
        }
    }
    /* the OOP row child for the rowoop arms (kept only if it wins) */
    {
        vfft_config_t rc;
        memset(&rc, 0, sizeof rc);
        rc.transform = VFFT_C2C;
        rc.placement = VFFT_OUTOFPLACE;
        rc.rigor = cfg->rigor;
        rc.dims = 1;
        rc.n[0] = N2;
        rc.howmany = 1;
        rc.order = VFFT_ORDER_NATURAL;
        rc.layout = VFFT_LAYOUT_INTERLEAVED;
        rc.nthreads = 1;
        rc.wisdom = cfg->wisdom;
        rc.wisdom_write = cfg->wisdom_write;
        rowo = (struct vfft_plan_s *)vfft_create(&rc);
        if (rowo)
        {
            rowscr = (double *)malloc(2 * (size_t)N2 * sizeof(double));
            if (!rowscr)
            {
                vfft_destroy(rowo);
                rowo = NULL;
            }
        }
    }
    for (ro = 0; ro <= (rowo ? 1 : 0); ro++)
        for (wi = 0; wi < nwl; wi++)
        {
            double ns = 1e300;
            int p2, s2, cut = 0;
            const int w = wlc[wi];
            if (w > 0)
                for (s2 = 0; s2 < h->il2d_nst; s2++)
                    if (w % h->il2d_L[s2] == 0)
                    {
                        cut = s2;
                        break;
                    }
            h->il2d_wl = w;
            h->il2d_cut = cut;
            h->il2d_tfuse = (w > 0);
            h->il2d_rowoop = ro;
            h->il2d_rowo = ro ? rowo : NULL;
            h->il2d_rowscr = ro ? rowscr : NULL;
            {
                _il2d_race_ctx_t rc = { h, NULL, z, 0, 1, 0, 0, 0, NULL, NULL, NULL, NULL };
                const vfft_race_arm_t arm = { "config", _il2d_arm_exec, &rc };
                /* min of 2 passes of reps full executes, this configuration */
                const vfft_race_proto_t proto = { 2, reps, VFFT_RACE_MIN, 0, 0, NULL, NULL };
                (void)p2;
                vfft_race_run(&proto, &arm, 1, &ns);
            }
            if (ns < best)
            {
                best = ns;
                bwl = w;
                bro = ro;
            }
        }
    /* set the winner, keep or drop the OOP child */
    {
        int s2, cut = 0;
        if (bwl > 0)
            for (s2 = 0; s2 < h->il2d_nst; s2++)
                if (bwl % h->il2d_L[s2] == 0)
                {
                    cut = s2;
                    break;
                }
        h->il2d_wl = bwl;
        h->il2d_cut = cut;
        h->il2d_tfuse = (bwl > 0);
        h->il2d_rowoop = bro;
        if (bro && rowo)
        {
            h->il2d_rowo = rowo;
            h->il2d_rowscr = rowscr;
        }
        else
        {
            h->il2d_rowo = NULL;
            h->il2d_rowscr = NULL;
            if (rowo)
                vfft_destroy(rowo);
            free(rowscr);
        }
    }
    vw2_2d_il_chain_bank(&W->vw2, N1, N2, h->il2d_R, h->il2d_nst,
                         h->il2d_wl, h->il2d_tfuse, h->il2d_rowoop,
                         -1, -1, (N1 & (N1 - 1)) ? h->il2d_blu : -1, best, (h->il2d_nat ? VW2_ORD_NAT : VW2_ORD_SCR));
    _vw2_persist(W, cfg);
    free(z);
}

/* ── c2c MT clones (INC-C). Worker t > 0 needs its own row child: the
 * serving path runs ONE plan through ONE rowscr, and two concurrent
 * bands interleaving that state produce garbage, not slowness. Clones
 * are built for the BANKED route only, verified route-equivalent
 * against the primary (_tc_clone_equiv — the TC army's structural
 * check, valid on any K=1 c2c plan), and required pool-free (a K=1
 * plan owns no TC batch, but the assert keeps the invariant explicit).
 * Any failure tears the set down: MT then declines and the engagement
 * counter shows it — never a half-cloned dispatch. */
static int _tc_clone_equiv(const struct vfft_plan_s *a,
                           const struct vfft_plan_s *b);
static void _il2d_c2c_build_clones(struct vfft_plan_s *h,
                                   const vfft_config_t *cfg, int T)
{
    const int n = (T > 64 ? 64 : T) - 1;
    const struct vfft_plan_s *prim =
        h->il2d_rowoop ? h->il2d_rowo : h->il2d_row;
    vfft_config_t rc;
    int t;
    if (n <= 0 || h->il2d_roww || !prim)
        return;
    memset(&rc, 0, sizeof rc);
    rc.transform = VFFT_C2C;
    rc.placement = h->il2d_rowoop ? VFFT_OUTOFPLACE : VFFT_INPLACE;
    rc.rigor = cfg->rigor;
    rc.dims = 1;
    rc.n[0] = h->N2;
    rc.howmany = 1;
    rc.order = VFFT_ORDER_NATURAL;
    rc.layout = VFFT_LAYOUT_INTERLEAVED;
    rc.nthreads = 1;
    rc.wisdom = cfg->wisdom;
    rc.wisdom_write = 0; /* clones read warm wisdom, never bank */
    h->il2d_roww = (struct vfft_plan_s **)calloc((size_t)n,
                                                 sizeof *h->il2d_roww);
    if (!h->il2d_roww)
        return;
    if (h->il2d_rowoop)
    {
        h->il2d_rowscr_w = (double *)malloc(
            2 * (size_t)h->N2 * (size_t)n * sizeof(double));
        if (!h->il2d_rowscr_w)
        {
            free(h->il2d_roww);
            h->il2d_roww = NULL;
            return;
        }
    }
    for (t = 0; t < n; t++)
    {
        struct vfft_plan_s *c =
            (struct vfft_plan_s *)vfft_create(&rc);
        h->il2d_roww[t] = c;
        if (!c || !_tc_clone_equiv(prim, c) || c->tcb || c->tcbw)
        {
            int u;
            _vfft_warn("il2d c2c MT: row clone %d %s at N2=%d — MT "
                       "declines for this plan",
                       t, c ? "route-mismatched" : "failed to create",
                       h->N2);
            for (u = 0; u <= t; u++)
                if (h->il2d_roww[u])
                    vfft_destroy(h->il2d_roww[u]);
            free(h->il2d_roww);
            h->il2d_roww = NULL;
            free(h->il2d_rowscr_w);
            h->il2d_rowscr_w = NULL;
            return;
        }
    }
    h->il2d_roww_n = n;
}

/* ── the c2c MT verdict race (same law as the real tier's): serial vs
 * threaded FULL walk through the very code execute serves with, min-of-3
 * alternated on a scratch plane, banked as cmt= + cmtt= (the T raced
 * at) in the cell's chain row. The "no" is banked exactly like the
 * "yes". If MT cannot engage at all (no clones, too few units), that IS
 * the verdict: cmt=0. */
static int _il2d_c2c_mt(struct vfft_plan_s *h, const double *sre,
                        double *dre, vfft_dir_t dir, int T);
static void _il2d_c2c_mt_race(struct vfft_plan_s *h,
                              struct vfft_wisdom_s *W,
                              const vfft_config_t *cfg, int N1, int N2)
{
    const size_t PN = (size_t)N1 * N2;
    double *z = (double *)malloc(2 * PN * sizeof(double));
    double st = 1e300, mt = 1e300;
    int p;
    size_t i;
    if (!z)
        return;
    for (i = 0; i < 2 * PN; i++)
        z[i] = 1.0 + 1e-6 * (double)(i & 511);
    if (!_il2d_c2c_mt(h, z, z, VFFT_FORWARD, h->nthreads))
    {
        h->il2d_colmt = 0; /* cannot engage — that IS the verdict */
        free(z);
        vw2_2d_il_chain_bank(&W->vw2, N1, N2, h->il2d_R, h->il2d_nst,
                             h->il2d_wl, h->il2d_tfuse, h->il2d_rowoop,
                             0, h->nthreads,
                             (N1 & (N1 - 1)) ? h->il2d_blu : -1, 0.0, (h->il2d_nat ? VW2_ORD_NAT : VW2_ORD_SCR));
        _vw2_persist(W, cfg);
        return;
    }
    {
        _il2d_race_ctx_t rc = { h, NULL, z, 0, 1, 0, 0, 0, NULL, NULL, NULL, NULL };
        const vfft_race_arm_t arms[3] = { { "serial", _il2d_arm_exec_st, &rc },
                                          { "threaded", _il2d_arm_exec_mt, &rc },
                                          { "strips", _il2d_arm_exec_mt_strip, &rc } };
        const vfft_race_proto_t proto = { 3, 1, VFFT_RACE_MIN, 0, 0, NULL, NULL }; /* min-of-3, A then B */
        double ns[3] = { 1e300, 1e300, 1e300 };
        (void)p;
        /* natural cells race THREE arms (serial, block, strips — the
         * natural pass has two legal partitions and no band arm); every
         * other cell keeps its two. */
        vfft_race_run(&proto, arms, h->il2d_nat ? 3 : 2, ns);
        st = ns[0];
        mt = ns[1];
        h->il2d_natarm = 0;
        if (h->il2d_nat && ns[2] < mt)
        {
            mt = ns[2];
            h->il2d_natarm = 1;
        }
        if (getenv("VFFT_IL2D_LOG") && h->il2d_nat)
            fprintf(stderr, "[il2d-c2c] natural arms %dx%d T=%d: block=%.0f "
                            "strips=%.0f\n",
                    N1, N2, h->nthreads, ns[1], ns[2]);
    }
    h->il2d_colmt = (mt < st);
    free(z);
    if (getenv("VFFT_IL2D_LOG"))
        fprintf(stderr, "[il2d-c2c] colmt race %dx%d T=%d: st=%.0f "
                        "mt=%.0f -> %s%s\n",
                N1, N2, h->nthreads, st, mt,
                h->il2d_colmt ? "THREADED" : "serial",
                (h->il2d_colmt && h->il2d_natarm) ? " (strips)" : "");
    vw2_2d_il_chain_bank(&W->vw2, N1, N2, h->il2d_R, h->il2d_nst,
                         h->il2d_wl, h->il2d_tfuse, h->il2d_rowoop,
                         h->il2d_colmt, h->nthreads,
                         (N1 & (N1 - 1)) ? h->il2d_blu : -1,
                         h->il2d_colmt ? mt : st, (h->il2d_nat ? VW2_ORD_NAT : VW2_ORD_SCR));
    _vw2_persist(W, cfg);
}

#endif /* VFFT_TRANSFORMS_FFT2D_IL2D_TIER_H */
