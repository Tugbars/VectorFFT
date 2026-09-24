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
 *              cmt=0 is banked with the thread count it was decided at. Since
 *              2026-09-24 every c2c route races it: the turn and the skewed
 *              column pass have row-slab walks of their own (one threaded arm).
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

/* the K=1 FOUR-STEP's inter-pass twiddle on one row (oop/k1_fourstep.h):
 * the row at plane position p times W_N^(k1(p) * n2), n2 = a*B + b, from
 * the per-position two-level record [coarse C[a] (N2/B)][fine F[b] (B)];
 * conj = the backward's conjugate. Straight C: the row is L2-hot and the
 * multiply is small against the row transform. */
static void _il2d_fs_twiddle(double *row, const double *rec, int B, size_t n2, int conj)
{
    const size_t na = n2 / (size_t)B;
    const double *C = rec, *F = rec + 2 * na;
    size_t a, b;
    for (a = 0; a < na; a++)
    {
        const double cr = C[2 * a], ci = conj ? -C[2 * a + 1] : C[2 * a + 1];
        double *r = row + 2 * a * (size_t)B;
        for (b = 0; b < (size_t)B; b++)
        {
            const double fr = F[2 * b], fi = conj ? -F[2 * b + 1] : F[2 * b + 1];
            const double wr = cr * fr - ci * fi, wi = cr * fi + ci * fr;
            const double xr = r[2 * b], xi = r[2 * b + 1];
            r[2 * b] = xr * wr - xi * wi;
            r[2 * b + 1] = xr * wi + xi * wr;
        }
    }
}
static const double *_il2d_fs_rec(const struct vfft_plan_s *h, size_t rn, size_t p)
{
    return h->il2d_fs_tw ? h->il2d_fs_tw + p * 2 * (rn / (size_t)h->il2d_fs_B + (size_t)h->il2d_fs_B) : NULL;
}

/* one row of the row pass through the in-place row child (route 0, one
 * door walk per row; the batched routes run rows through _il2d_rows_exec
 * and never come here). p = the row's plane position (the four-step's
 * twiddle hook keys on it; every other plan ignores it). */
static void _il2d_row_exec(struct vfft_plan_s *h, vfft_dir_t dir,
                           double *row, size_t rn, size_t p)
{
    const double *tw = _il2d_fs_rec(h, rn, p);
    if (tw && dir == VFFT_FORWARD)
        _il2d_fs_twiddle(row, tw, h->il2d_fs_B, rn, 0);
    vfft_execute(h->il2d_row, dir, row, NULL, row, NULL);
    if (tw && dir != VFFT_FORWARD)
        _il2d_fs_twiddle(row, tw, h->il2d_fs_B, rn, 1);
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
/* THE COLUMN-ONLY SERIAL EXECUTE (2026-09-06): the pass a rank-N IL plan
 * runs on each column axis and the real tier runs on its hp1-wide plane —
 * natural (leaf-redirected), Bluestein, banded (no row fusion: the owning
 * plan runs the rows) or unbanded, both directions, by the descriptor alone.
 * The c2c 2D serial walk with the fused row pass lives in vfft_execute.h. */
/* ── THE TURNED PRIME COLUMN PASS (tpc, 2026-09-24) ──────────────────────
 * At a prime N (no chain) the column Bluestein was the only engine and read
 * 0.5-0.9x the vendor library at N >= 131 over pow2 lanes (131x4096 0.51,
 * 257x256 0.69; the rank-3 cells 131x64x64 0.54, 257x16x16 0.70 are this
 * pass over N2*N3 lanes), while the 1D prime route at the same N wins
 * 1.1-3.1x. The turned pass runs the axis through that route: the lanes
 * [c0, c1) of the plane transposed into the scratch (row c = lane c, pitch
 * N + 8), the 1D natural in-place K=1 plan at N on every row of it, the rows
 * transposed back into the plane's lanes. Measured before it was built:
 * 131x4096 turn 426 + rows 1585 + back 527 us against the Bluestein pass's
 * ~8400. Both directions (the 1D plan's own backward). */
#define VFFT_IL2D_TPC_PITCH(N) ((size_t)(N) + 8)
/* S[c][r] = src[r][c] for lanes c in [c0, c1), rows r < N (src pitch rn, S
 * pitch P): 2 x 2 complex blocks through one 128-bit lane permute, the rows
 * blocked so a block's source lines stay in L1 across the lane pairs */
static void _il2d_tpc_turn_in(const double *src, size_t rn, size_t N, size_t c0, size_t c1,
                              double *S, size_t P)
{
    const size_t RB = 256;
    size_t r0, c, r;
    for (r0 = 0; r0 < N; r0 += RB)
    {
        const size_t r1 = (r0 + RB < N) ? r0 + RB : N;
        for (c = c0; c + 2 <= c1; c += 2)
        {
            double *sa = S + 2 * (c * P), *sb = S + 2 * ((c + 1) * P);
            for (r = r0; r + 2 <= r1; r += 2)
            {
                const __m256d a = _mm256_loadu_pd(src + 2 * (r * rn + c));
                const __m256d b = _mm256_loadu_pd(src + 2 * ((r + 1) * rn + c));
                _mm256_storeu_pd(sa + 2 * r, _mm256_permute2f128_pd(a, b, 0x20));
                _mm256_storeu_pd(sb + 2 * r, _mm256_permute2f128_pd(a, b, 0x31));
            }
            if (r < r1)
            {
                _mm_storeu_pd(sa + 2 * r, _mm_loadu_pd(src + 2 * (r * rn + c)));
                _mm_storeu_pd(sb + 2 * r, _mm_loadu_pd(src + 2 * (r * rn + c + 1)));
            }
        }
        if (c < c1)
        {
            double *sa = S + 2 * (c * P);
            for (r = r0; r < r1; r++)
                _mm_storeu_pd(sa + 2 * r, _mm_loadu_pd(src + 2 * (r * rn + c)));
        }
    }
}
/* dst[r][c] = S[c][r] for lanes c in [c0, c1): the turn route's back-turn
 * over a lane range (dst pitch rn, S pitch P) */
static void _il2d_tpc_turn_out(const double *S, size_t P, double *dst, size_t rn, size_t N,
                               size_t c0, size_t c1)
{
    const size_t RB = 256;
    size_t r0, c, r;
    for (r0 = 0; r0 < N; r0 += RB)
    {
        const size_t r1 = (r0 + RB < N) ? r0 + RB : N;
        for (c = c0; c + 2 <= c1; c += 2)
        {
            const double *ta = S + 2 * (c * P), *tb = S + 2 * ((c + 1) * P);
            for (r = r0; r + 2 <= r1; r += 2)
            {
                const __m256d a = _mm256_loadu_pd(ta + 2 * r), b = _mm256_loadu_pd(tb + 2 * r);
                _mm256_storeu_pd(dst + 2 * (r * rn + c), _mm256_permute2f128_pd(a, b, 0x20));
                _mm256_storeu_pd(dst + 2 * ((r + 1) * rn + c), _mm256_permute2f128_pd(a, b, 0x31));
            }
            if (r < r1)
            {
                _mm_storeu_pd(dst + 2 * (r * rn + c), _mm_loadu_pd(ta + 2 * r));
                _mm_storeu_pd(dst + 2 * (r * rn + c + 1), _mm_loadu_pd(tb + 2 * r));
            }
        }
        if (c < c1)
        {
            const double *ta = S + 2 * (c * P);
            for (r = r0; r < r1; r++)
                _mm_storeu_pd(dst + 2 * (r * rn + c), _mm_loadu_pd(ta + 2 * r));
        }
    }
}
/* the pass over lanes [c0, c1): turn in, the 1D plan per row, turn out */
static void _il2d_tpc_cols_range(const vfft_ilcol_t *c, const double *src, double *dst,
                                 size_t rn, size_t c0, size_t c1, int rev)
{
    const size_t N = (size_t)c->N, P = VFFT_IL2D_TPC_PITCH(N);
    double *S = c->tpcscr;
    size_t l;
    _il2d_tpc_turn_in(src, rn, N, c0, c1, S, P);
    for (l = c0; l < c1; l++)
        vfft_execute((vfft_plan)c->tpcplan, rev ? VFFT_BACKWARD : VFFT_FORWARD,
                     S + 2 * l * P, NULL, S + 2 * l * P, NULL);
    _il2d_tpc_turn_out(S, P, dst, rn, N, c0, c1);
}

static void _il2d_col_exec(const vfft_ilcol_t *c, const double *src,
                           double *dst, int reverse)
{
    const size_t hp1 = c->rn;
    if (c->nat)
    { /* NATURAL n1 (M4-lite): the leaf-redirected pass, unbanded by
       * construction (wl pinned 0 at create). */
        _il2d_col_pass_nat(src, dst, c->N, hp1, c->nst, c->R,
                           c->L,
                           reverse ? c->b : c->f,
                           reverse ? c->tb : c->tf, reverse,
                           c->natperm, c->natscr, NULL);
        return;
    }
    if (c->blu && c->tpc)
    {   /* prime N1: the TURNED pass (2026-09-24) */
        _il2d_tpc_cols_range(c, src, dst, hp1, 0, hp1, reverse);
        return;
    }
    if (c->blu)
    { /* prime N1: the shared Bluestein pipeline over the CCE plane;
       * reverse = the inverse transform (conjugated chirp/kernel). */
        _il2d_blu_cols(src, dst, c->N, hp1, c->blu, c->nst,
                       c->R, c->L, c->f, c->b,
                       c->tf, c->tb,
                       reverse ? c->bluchb : c->bluchf,
                       reverse ? c->blukb : c->blukf,
                       c->bluscr);
        return;
    }
    if (c->wl > 0)
    {
        const int cut = c->cut, nst = c->nst;
        const size_t wl = (size_t)c->wl;
        vfft_il2p_fn const *fns = reverse ? c->b : c->f;
        double *const *tabs = reverse ? c->tb : c->tf;
        size_t b0;
        if (!reverse)
        {
            if (cut > 0)
                _il2d_col_stages(src, dst, c->N, hp1, 0, cut,
                                 c->R, c->L, fns, tabs, 0);
            for (b0 = 0; b0 < (size_t)c->N; b0 += wl)
            {
                const double *bs = (cut > 0) ? dst + 2 * b0 * hp1
                                             : src + 2 * b0 * hp1;
                _il2d_col_stages(bs, dst + 2 * b0 * hp1, (int)wl, hp1,
                                 cut, nst, c->R, c->L, fns,
                                 tabs, 0);
            }
        }
        else
        {
            for (b0 = 0; b0 < (size_t)c->N; b0 += wl)
                _il2d_col_stages(src + 2 * b0 * hp1,
                                 dst + 2 * b0 * hp1, (int)wl, hp1, cut,
                                 nst, c->R, c->L, fns, tabs,
                                 1);
            if (cut > 0)
                _il2d_col_stages(dst, dst, c->N, hp1, 0, cut,
                                 c->R, c->L, fns, tabs, 1);
        }
        return;
    }
    _il2d_col_pass(src, dst, c->N, hp1, 0, c->nst, c->R,
                   c->L, reverse ? c->b : c->f,
                   reverse ? c->tb : c->tf, reverse);
}

/* the real tier's column pass: the descriptor's, over hp1 columns */
static void _il2d_real_cols(struct vfft_plan_s *h, const double *src,
                            double *dst, int reverse)
{
    _il2d_col_exec(&h->il2d_col, src, dst, reverse);
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
    vfft_il2p_fn const *fns = a->reverse ? h->il2d_col.b : h->il2d_col.f;
    double *const *tabs = a->reverse ? h->il2d_col.tb : h->il2d_col.tf;
    if (a->natleaf)
    {   /* natural x MT: the leaf scatter/gather over [lo,hi) blocks */
        _il2d_nat_leaf_range(a->src, a->dst, h->N, hp1,
                             h->il2d_col.R[h->il2d_col.nst - 1], fns[h->il2d_col.nst - 1],
                             h->il2d_col.natperm, a->lo, a->hi, a->reverse, NULL);
        return;
    }
    if (a->strip)
    {
        if (h->il2d_col.blu)
        {   /* Bluestein column axis: the window pipeline (2026-09-02) */
            _il2d_blu_cols_range(a->src, a->dst, h->N, hp1, a->lo, a->hi,
                                 h->il2d_col.blu, h->il2d_col.nst, h->il2d_col.R,
                                 h->il2d_col.L, h->il2d_col.f, h->il2d_col.b,
                                 h->il2d_col.tf, h->il2d_col.tb,
                                 a->reverse ? h->il2d_col.bluchb : h->il2d_col.bluchf,
                                 a->reverse ? h->il2d_col.blukb : h->il2d_col.blukf,
                                 h->il2d_col.bluscr);
            return;
        }
        _il2d_col_pass_range(a->src, a->dst, h->N, hp1, a->lo, a->hi,
                             h->il2d_col.nst, h->il2d_col.R, h->il2d_col.L, fns,
                             tabs, a->reverse);
        return;
    }
    {
        const size_t wl = (size_t)h->il2d_col.wl;
        size_t b;
        for (b = a->lo; b < a->hi; b++)
        {
            const size_t b0 = b * wl;
            const double *bs = a->src + 2 * b0 * hp1;
            _il2d_col_stages(bs, a->dst + 2 * b0 * hp1, (int)wl, hp1,
                             h->il2d_col.cut, h->il2d_col.nst, h->il2d_col.R,
                             h->il2d_col.L, fns, tabs, a->reverse);
        }
    }
}

/* Returns 1 when it ran threaded, 0 when the caller must run serial. */
static int _il2d_real_cols_mt(struct vfft_plan_s *h, const double *src,
                              double *dst, int reverse, int T)
{
    const size_t hp1 = (size_t)h->N2 / 2 + 1;
    const int strip = (h->il2d_col.wl <= 0);
    size_t units = strip ? hp1 : ((size_t)h->N / (size_t)h->il2d_col.wl);
    _il2d_cmt_arg a[STRIDE_POOL_MAX_DISPATCH];
    int t;
    /* T arrives as the plan's snapshot (h->nthreads); the pool's one clamp
     * bounds it by the live pool and the arg-array size. */
    T = stride_pool_workers_for(T);
    if (T >= 2 && h->il2d_col.nat)
    {
        /* NATURAL x MT (2026-09-04): the matched partition of the
         * natural pass — prefix stages digit-split (src -> scratch, then
         * in place), the leaf scatter by BLOCK RANGE (scratch -> dst),
         * mirrored for bwd (gather first, reversed prefix after, stage 0
         * scratch -> dst). No band arm: the scatter crosses bands. */
        const int Rl = h->il2d_col.R[h->il2d_col.nst - 1];
        const size_t nb = (size_t)h->N / (size_t)Rl;
        const int Tb = nb < (size_t)T ? (int)nb : T;
        double *scr = h->il2d_col.natscr;
        int s;
        if (Tb < 2 || h->il2d_col.nst < 2)
            return 0;
        if (!reverse)
        {
            for (s = 0; s < h->il2d_col.nst - 1; s++)
            {
                const double *ssrc = (s == 0) ? src : scr;
                if (!_il2d_stage_digits_mt(ssrc, scr, h->N, hp1, hp1,
                                           h->il2d_col.R[s], h->il2d_col.L[s],
                                           h->il2d_col.f[s], h->il2d_col.tf[s], T))
                    _il2d_col_stages(ssrc, scr, h->N, hp1, s, s + 1,
                                     h->il2d_col.R, h->il2d_col.L, h->il2d_col.f,
                                     h->il2d_col.tf, 0);
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
            for (s = h->il2d_col.nst - 2; s >= 0; s--)
            {
                double *out = (s == 0) ? dst : scr;
                if (!_il2d_stage_digits_mt(scr, out, h->N, hp1, hp1,
                                           h->il2d_col.R[s], h->il2d_col.L[s],
                                           h->il2d_col.b[s], h->il2d_col.tb[s], T))
                    _il2d_col_stages(scr, out, h->N, hp1, s, s + 1,
                                     h->il2d_col.R, h->il2d_col.L, h->il2d_col.b,
                                     h->il2d_col.tb, 0);
            }
        }
        _vfft_il2d_col_mt_count++;
        return 1;
    }
    if (T < 2 || units < (size_t)T)
        return 0; /* not enough independent units to be worth splitting */
    /* fwd: the wide prefix must complete before ANY band (stage 0's legs
     * span the whole plane). bwd: the reversed prefix runs after. */
    if (!strip && !reverse && h->il2d_col.cut > 0)
    {
        /* INC-3b: each prefix stage's DIGITS split over the workers
         * (whole rows, count untouched); stages stay ordered, one
         * dispatch each — a dispatch+wait is ~100 ns (INC-2), so the
         * per-stage join is free at these sizes. A stage that cannot
         * split (D < T, or the table-free D==1 leaf) runs serial. */
        int s;
        for (s = 0; s < h->il2d_col.cut; s++)
        {
            const double *ssrc = (s == 0) ? src : dst;
            if (!_il2d_stage_digits_mt(ssrc, dst, h->N, hp1, hp1,
                                       h->il2d_col.R[s], h->il2d_col.L[s],
                                       h->il2d_col.f[s], h->il2d_col.tf[s], T))
                _il2d_col_stages(ssrc, dst, h->N, hp1, s, s + 1,
                                 h->il2d_col.R, h->il2d_col.L, h->il2d_col.f,
                                 h->il2d_col.tf, 0);
        }
    }
    for (t = 0; t < T; t++)
    {
        a[t].h = h;
        /* after a fwd prefix the band source IS dst (in place) */
        a[t].src = (!strip && !reverse && h->il2d_col.cut > 0) ? dst : src;
        a[t].dst = dst;
        a[t].reverse = reverse;
        a[t].strip = strip;
        a[t].natleaf = 0;
        a[t].lo = units * (size_t)t / (size_t)T;
        a[t].hi = units * (size_t)(t + 1) / (size_t)T;
    }
    stride_pool_run(T, _il2d_cmt_tramp, a, sizeof a[0]); /* caller = a[0] */
    _vfft_il2d_col_mt_count++; /* engagement, see vfft.h */
    if (!strip && reverse && h->il2d_col.cut > 0)
    {
        /* the Hermitian-transpose chain: prefix stages in REVERSE order,
         * in place on dst, each digit-split the same way. */
        int s;
        for (s = h->il2d_col.cut - 1; s >= 0; s--)
            if (!_il2d_stage_digits_mt(dst, dst, h->N, hp1, hp1,
                                       h->il2d_col.R[s], h->il2d_col.L[s],
                                       h->il2d_col.b[s], h->il2d_col.tb[s], T))
                _il2d_col_stages(dst, dst, h->N, hp1, s, s + 1,
                                 h->il2d_col.R, h->il2d_col.L, h->il2d_col.b,
                                 h->il2d_col.tb, 0);
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
 * Row execution mutates shared plan state (one child), so a worker t > 0
 * runs its CLONE (il2d_roww[t-1], route-equivalence-checked at build).
 * Every arm is
 * a loop restriction of the serving walk => MT == ST bitwise. */
static void _il2d_row_exec_t(struct vfft_plan_s *h, int tid,
                             vfft_dir_t dir, double *row, size_t rn, size_t p)
{
    if (tid <= 0)
    {
        _il2d_row_exec(h, dir, row, rn, p);
        return;
    }
    {
        struct vfft_plan_s *c = h->il2d_roww[tid - 1];
        const double *tw = _il2d_fs_rec(h, rn, p);
        if (tw && dir == VFFT_FORWARD)
            _il2d_fs_twiddle(row, tw, h->il2d_fs_B, rn, 0);
        vfft_execute((vfft_plan)c, dir, row, NULL, row, NULL);
        if (tw && dir != VFFT_FORWARD)
            _il2d_fs_twiddle(row, tw, h->il2d_fs_B, rn, 1);
    }
}

/* the banked ROW-ROUTE value of a plan: 0 = the in-place child, 2 = the
 * batched rows, 3 = the batched two-pass rows (the ro= token, 2026-09-23) */
static int _il2d_ro_of(const struct vfft_plan_s *h)
{
    return h->il2d_rowb2 ? 3 : h->il2d_rowb ? 2 : 0;
}

/* the ROW-LOOP twin of a two-pass stage kernel (2026-09-23): the same body
 * with the in-kernel row loop (count = rows x Ls lanes, in pitch Gs, out pitch
 * OGs); NULL where the corpus has none -- then the two-pass rows have no arm */
#define VFFT_IL2D_RB2_CHUNK 64   /* the tile's ceiling in rows: the per-worker scratch is CHUNK x N2 */
/* the two-pass rows' TILE (2026-09-23): a RACED axis like ZTURN-T's tile --
 * the ladder is the chunk scratch in KB, rows = KB*1024 / (16*N2), 2..CHUNK;
 * banked as rbk= beside ro=3, replayed, VFFT_IL2D_RB2_KB pins it for probes */
static const int VFFT_IL2D_RB2_KB_LADDER[4] = { 4, 8, 16, 32 };
static size_t _il2d_rb2_rows(int kb, size_t rn)
{
    size_t r = (size_t)kb * 1024 / (16 * rn);
    return r < 2 ? 2 : (r > VFFT_IL2D_RB2_CHUNK ? VFFT_IL2D_RB2_CHUNK : r);
}
static vfft_il2p_fn _il2d_rowloop_twin(vfft_il2p_fn f)
{
    if (!f) return 0;
#define T_(a, b) if (f == (vfft_il2p_fn)a) return b;
    T_(radix4_z_n1t_fwd_avx2, radix4_z_n1tr_fwd_avx2)
    T_(radix8_z_n1t_fwd_avx2, radix8_z_n1tr_fwd_avx2)
    T_(radix16_z_n1t_fwd_avx2, radix16_z_n1tr_fwd_avx2)
    T_(radix4_z_t2_fwd_avx2, radix4_z_t2r_fwd_avx2)
    T_(radix8_z_t2_fwd_avx2, radix8_z_t2r_fwd_avx2)
    T_(radix16_z_t2_fwd_avx2, radix16_z_t2r_fwd_avx2)
    T_(radix4_z_t2t_bwd_avx2, radix4_z_t2tr_bwd_avx2)
    T_(radix8_z_t2t_bwd_avx2, radix8_z_t2tr_bwd_avx2)
    T_(radix16_z_t2t_bwd_avx2, radix16_z_t2tr_bwd_avx2)
    T_(radix4_z_n1_bwd_avx2, radix4_z_n1r_bwd_avx2)
    T_(radix8_z_n1_bwd_avx2, radix8_z_n1r_bwd_avx2)
    T_(radix16_z_n1_bwd_avx2, radix16_z_n1r_bwd_avx2)
    T_(radix8_z_n1ttan_fwd_avx2, radix8_z_n1trtan_fwd_avx2)
    T_(radix16_z_n1ttan_fwd_avx2, radix16_z_n1trtan_fwd_avx2)
    T_(radix8_z_t2tan_fwd_avx2, radix8_z_t2rtan_fwd_avx2)
    T_(radix16_z_t2tan_fwd_avx2, radix16_z_t2rtan_fwd_avx2)
    T_(radix8_z_t2ttan_bwd_avx2, radix8_z_t2trtan_bwd_avx2)
    T_(radix16_z_t2ttan_bwd_avx2, radix16_z_t2trtan_bwd_avx2)
    T_(radix8_z_n1tan_bwd_avx2, radix8_z_n1rtan_bwd_avx2)
    T_(radix16_z_n1tan_bwd_avx2, radix16_z_n1rtan_bwd_avx2)
#undef T_
    return 0;
}

/* the row pass over a RUN of rows (2026-09-23): nrows rows, row i at
 * base + 2*i*pitch (pitch in complex), plane position p0 + i*pstep (the
 * four-step's twiddle hook keys on it). Route 2 (il2d_rowb) runs ONE
 * kernel call over the run -- lane k = row k, two rows per vector, no
 * per-row door -- with the hook's twiddle, when a four-step owns this
 * plan, applied per row around that call exactly as _il2d_row_exec applies
 * it (forward: before; backward: after, conjugate). Routes 0/1 loop the
 * per-row child (worker tid's clone). Every row loop of the c2c tier --
 * serial walks, the natural leaf, the MT trampoline, the four-step's
 * super-band -- runs its rows through here, so the batched route serves
 * every arm the same way (MT == ST bitwise: the kernel is stateless). */
static void _il2d_rows_exec2(struct vfft_plan_s *h, int tid, vfft_dir_t dir,
                             const double *in, size_t pin, double *out, size_t pout,
                             size_t rn, size_t p0, size_t pstep, size_t nrows);
static void _il2d_rows_exec(struct vfft_plan_s *h, int tid, vfft_dir_t dir,
                            double *base, size_t rn, size_t pitch,
                            size_t p0, size_t pstep, size_t nrows)
{
    _il2d_rows_exec2(h, tid, dir, base, pitch, base, pitch, rn, p0, pstep, nrows);
}
/* the same run of rows from `in` (pitch pin) to `out` (pitch pout) -- the
 * skewed column pass's row move (in != out). Route 2 and 3 read and write at
 * their own pitches; route 0 runs the out-of-place K=1 plan (il2d_csk_row) when
 * in != out and the in-place child otherwise. The four-step's hook rides only
 * on in-place runs (its child never takes the skewed column pass). */
static void _il2d_rows_exec2(struct vfft_plan_s *h, int tid, vfft_dir_t dir,
                             const double *in, size_t pin, double *out, size_t pout,
                             size_t rn, size_t p0, size_t pstep, size_t nrows)
{
    size_t i;
    const int inplace = (in == out);
    double *base = out;
    const size_t pitch = pout;
    if (h->il2d_rowb2)
    {   /* route 3: the batched TWO-PASS rows -- per chunk, stage 1 rows ->
         * the worker's scratch (rows at pitch rn), stage 2 scratch -> rows;
         * backward t2t then n1 the same way; the four-step's hook per row
         * around the run as route 2 */
        const vfft_il2p_plan_t *p = h->il2d_row->k1il2p;
        const size_t R1 = (size_t)p->R1, R2 = (size_t)p->R2;
        double *scr = h->il2d_rowb2_scr + (size_t)(tid > 0 ? tid : 0) * 2 * (size_t)VFFT_IL2D_RB2_CHUNK * rn;
        const int fwd = (dir == VFFT_FORWARD);
        /* the tile: the raced rbk= (a 64-row chunk at N2 = 32/64 spilled L1 and
         * lost to the per-row child; 16 rows at 32 won 1.5x, 2026-09-23) */
        const size_t ch = h->il2d_rowb2_ch > 0 ? (size_t)h->il2d_rowb2_ch : 2;
        size_t i0;
        if (h->il2d_fs_tw && fwd)
            for (i = 0; i < nrows; i++)
            {
                const double *tw = _il2d_fs_rec(h, rn, p0 + i * pstep);
                if (tw)
                    _il2d_fs_twiddle(base + 2 * i * pitch, tw, h->il2d_fs_B, rn, 0);
            }
        for (i0 = 0; i0 < nrows; i0 += ch)
        {
            const size_t n = (nrows - i0 < ch) ? nrows - i0 : ch;
            const double *bi = in + 2 * i0 * pin;
            double *b = base + 2 * i0 * pitch;
            if (fwd)
            {
                h->il2d_rowb2_leaf_f(bi, NULL, scr, NULL, NULL, NULL, R1, pin, R2, rn, n * R1);
                h->il2d_rowb2_mid_f(scr, NULL, b, NULL, p->tw, NULL, R2, rn, R2, pitch, n * R2);
            }
            else
            {
                h->il2d_rowb2_t2t_b(bi, NULL, scr, NULL, p->twb, NULL, R2, pin, R1, rn, n * R2);
                h->il2d_rowb2_n1_b(scr, NULL, b, NULL, NULL, NULL, R1, rn, R1, pitch, n * R1);
            }
        }
        if (h->il2d_fs_tw && !fwd)
            for (i = 0; i < nrows; i++)
            {
                const double *tw = _il2d_fs_rec(h, rn, p0 + i * pstep);
                if (tw)
                    _il2d_fs_twiddle(base + 2 * i * pitch, tw, h->il2d_fs_B, rn, 1);
            }
        return;
    }
    if (h->il2d_rowb)
    {
        const int fwd = (dir == VFFT_FORWARD);
        if (h->il2d_fs_tw && fwd)
            for (i = 0; i < nrows; i++)
            {
                const double *tw = _il2d_fs_rec(h, rn, p0 + i * pstep);
                if (tw)
                    _il2d_fs_twiddle(base + 2 * i * pitch, tw, h->il2d_fs_B, rn, 0);
            }
        (fwd ? h->il2d_rowb_f : h->il2d_rowb_b)(in, NULL, base, NULL, NULL, NULL,
                                                 1, pin, 1, pitch, nrows);
        if (h->il2d_fs_tw && !fwd)
            for (i = 0; i < nrows; i++)
            {
                const double *tw = _il2d_fs_rec(h, rn, p0 + i * pstep);
                if (tw)
                    _il2d_fs_twiddle(base + 2 * i * pitch, tw, h->il2d_fs_B, rn, 1);
            }
        return;
    }
    if (!inplace)
    {   /* the per-row route from the scratch: the out-of-place K=1 plan at N2
         * (worker t > 0 runs its clone: the threaded skewed pass, 2026-09-24) */
        struct vfft_plan_s *rp = (tid > 0) ? h->il2d_cskw[tid - 1] : h->il2d_csk_row;
        for (i = 0; i < nrows; i++)
            vfft_execute((vfft_plan)rp, dir, in + 2 * i * pin, NULL, base + 2 * i * pitch, NULL);
        return;
    }
    for (i = 0; i < nrows; i++)
        _il2d_row_exec_t(h, tid, dir, base + 2 * i * pitch, rn, p0 + i * pstep);
}

/* the skewed column pass's pitch in complex: N2 + 8 -- the R output streams of
 * the column stage never sit a multiple of 4 KB apart */
#define VFFT_IL2D_CSK_PITCH(N2) ((size_t)(N2) + 8)
/* the SKEWED column pass's execute (2026-09-23): the single column stage from
 * the plane (leg stride N2) into the scratch (leg stride N2 + 8), then the rows
 * from the scratch into the destination -- the rows are the move. Both
 * directions (the passes commute), both placements (the plane is read whole
 * before the rows write it). */
static void _il2d_csk_exec(struct vfft_plan_s *h, vfft_dir_t dir, const double *sre, double *dre)
{
    const size_t N1 = (size_t)h->N, rn = (size_t)h->N2, P = VFFT_IL2D_CSK_PITCH(rn);
    const int fwd = (dir == VFFT_FORWARD);
    double *T = h->il2d_csk_scr;
    (fwd ? h->il2d_csk_f : h->il2d_csk_b)(sre, NULL, T, NULL, NULL, NULL, rn, 0, P, 0, rn);
    _il2d_rows_exec2(h, 0, dir, T, P, dre, rn, rn, 0, 1, N1);
}

/* the TURN route's back-turn (2026-09-23): the N2 x N1 scratch T (row c =
 * column c of the plane, N1 long) into the N1 x N2 plane dst, 2 x 2 complex
 * blocks through one 128-bit lane permute, the row range blocked so a block
 * of dst rows stays in L1 while every column pair lands in it (N2 is small
 * where this route wins: dst rows are 32..128 B and would otherwise be
 * re-fetched once per column pair). Odd N1 / odd N2 finish scalar. */
/* the scratch's row PITCH in complex: N1 + 8, so the N2 column streams never sit
 * a multiple of 4 KB apart (they did at every N1 >= 256: one L1 set, 12-way, for
 * 8 or 16 streams -- the route's margin shrank from 2.3x at N2 = 2 to nothing at
 * 16 before the skew) */
#define VFFT_IL2D_TURN_PITCH(N1) ((size_t)(N1) + 8)
static void _il2d_turn_back_range(const double *T, size_t P, double *dst, size_t N2, size_t ra, size_t rb)
{
    const size_t RB = 256;
    size_t r0, c, r;
    for (r0 = ra; r0 < rb; r0 += RB)
    {
        const size_t r1 = (r0 + RB < rb) ? r0 + RB : rb;
        for (c = 0; c + 2 <= N2; c += 2)
        {
            const double *ta = T + 2 * (c * P), *tb = T + 2 * ((c + 1) * P);
            for (r = r0; r + 2 <= r1; r += 2)
            {
                const __m256d a = _mm256_loadu_pd(ta + 2 * r), b = _mm256_loadu_pd(tb + 2 * r);
                _mm256_storeu_pd(dst + 2 * (r * N2 + c), _mm256_permute2f128_pd(a, b, 0x20));
                _mm256_storeu_pd(dst + 2 * ((r + 1) * N2 + c), _mm256_permute2f128_pd(a, b, 0x31));
            }
            if (r < r1)
            {
                _mm_storeu_pd(dst + 2 * (r * N2 + c), _mm_loadu_pd(ta + 2 * r));
                _mm_storeu_pd(dst + 2 * (r * N2 + c + 1), _mm_loadu_pd(tb + 2 * r));
            }
        }
        if (c < N2)
        {
            const double *ta = T + 2 * (c * P);
            for (r = r0; r < r1; r++)
                _mm_storeu_pd(dst + 2 * (r * N2 + c), _mm_loadu_pd(ta + 2 * r));
        }
    }
}
static void _il2d_turn_back(const double *T, size_t P, double *dst, size_t N1, size_t N2)
{
    _il2d_turn_back_range(T, P, dst, N2, 0, N1);
}

/* the TURN route's execute: rows through the batched mono kernel with turned
 * stores (leg l of row k -> T[l][k]: Ls = 1, Gs = N2, OLs = the skewed pitch P, OGs = 1), the
 * N2 columns as rows of T through the in-place K=1 plan at N1, the back-turn.
 * Both directions (the 2D passes commute), both placements (T is private:
 * the plane is read whole before dst is written). */
static void _il2d_turn_exec(struct vfft_plan_s *h, vfft_dir_t dir, const double *sre, double *dre)
{
    const size_t N1 = (size_t)h->N, rn = (size_t)h->N2, P = VFFT_IL2D_TURN_PITCH(N1);
    double *T = h->il2d_turn_scr;
    size_t c;
    (dir == VFFT_FORWARD ? h->il2d_rowb_f : h->il2d_rowb_b)(sre, NULL, T, NULL, NULL, NULL, 1, rn, P, 1, N1);
    for (c = 0; c < rn; c++)
        vfft_execute((vfft_plan)h->il2d_turn_plan, dir, T + 2 * c * P, NULL, T + 2 * c * P, NULL);
    _il2d_turn_back(T, P, dre, N1, rn);
}

/* the natural leaf over [blo, bhi) blocks through worker tid's staging,
 * the block's rows FUSED there forward when the walk fuses rows (each
 * natural row leaves finished, one sequential stream); backward the leaf
 * gathers only — the backward's rows stay after the column pass, the
 * order every backward arm shares (il2d_natural_leaf_design.md) */
static double *_il2d_nat_stage_of(struct vfft_plan_s *h, int tid)
{
    const int Rl = h->il2d_col.R[h->il2d_col.nst - 1];
    return (h->il2d_col.natstage && h->il2d_col.natst)
               ? h->il2d_col.natstage + (size_t)(tid > 0 ? tid : 0) * 2 * (size_t)Rl * h->il2d_col.rn
               : NULL;
}
static void _il2d_nat_leaf_blocks(struct vfft_plan_s *h, int tid, vfft_dir_t dir,
                                  const double *from, double *to, size_t blo, size_t bhi, int fuse)
{
    const int nst = h->il2d_col.nst, Rl = h->il2d_col.R[nst - 1];
    const size_t rn = h->il2d_col.rn, nstride_rows = (size_t)(h->N / Rl);
    const int fwd = (dir == VFFT_FORWARD);
    vfft_il2p_fn fn = fwd ? h->il2d_col.f[nst - 1] : h->il2d_col.b[nst - 1];
    double *stage = _il2d_nat_stage_of(h, tid);
    size_t b;
    if (!stage || !fwd || !fuse)
    {
        _il2d_nat_leaf_range(from, to, h->N, rn, Rl, fn, h->il2d_col.natperm, blo, bhi, !fwd, stage);
        if (fwd && fuse)
            for (b = blo; b < bhi; b++)
            {
                const size_t r0 = (size_t)h->il2d_col.natperm[b * (size_t)Rl];
                _il2d_rows_exec(h, tid, dir, to + 2 * r0 * rn, rn, nstride_rows * rn, r0, nstride_rows, (size_t)Rl);
            }
        return;
    }
    for (b = blo; b < bhi; b++)
    {
        const size_t coff = 2 * b * (size_t)Rl * rn;
        const size_t nrow0 = (size_t)h->il2d_col.natperm[b * (size_t)Rl];
        fn(from + coff, NULL, stage, NULL, NULL, NULL, rn, 0, rn, 0, rn);
        _il2d_rows_exec(h, tid, dir, stage, rn, rn, nrow0, nstride_rows, (size_t)Rl);
        _il2d_nat_stage_out(stage, to, nrow0, nstride_rows, Rl, rn, 0, rn, 1);   /* finished rows: streamed */
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
    vfft_il2p_fn const *fns = a->fwd ? h->il2d_col.f : h->il2d_col.b;
    double *const *tabs = a->fwd ? h->il2d_col.tf : h->il2d_col.tb;
    size_t b;
    switch (a->mode)
    {
    case 0: /* bands: suffix stages, then (tfuse) the band's rows —
             * rows LAST in the band in BOTH directions, the serving
             * order (bwd runs the reversed suffix via !fwd). The one
             * exception is the four-step's backward (il2d_fs_tw): its rows
             * carry the conjugate inter-pass twiddle and must precede every
             * column stage, so the band is moved to dst first and its rows
             * run BEFORE the reversed suffix. */
        for (b = a->lo; b < a->hi; b++)
        {
            const size_t b0 = b * (size_t)h->il2d_col.wl;
            const double *bs = (a->fwd && h->il2d_col.cut > 0)
                                   ? a->dst + 2 * b0 * rn
                                   : a->src + 2 * b0 * rn;
            double *bd = a->dst + 2 * b0 * rn;
            const int hookb = (h->il2d_fs_tw != NULL) && !a->fwd;
            if (hookb && h->il2d_col.tfuse)
            {
                if (bd != bs)
                    memcpy(bd, bs, 2 * (size_t)h->il2d_col.wl * rn * sizeof(double));
                _il2d_rows_exec(h, a->tid, a->dir, bd, rn, rn, b0, 1, (size_t)h->il2d_col.wl);
                bs = bd;
            }
            _il2d_col_stages(bs, bd, h->il2d_col.wl, rn, h->il2d_col.cut,
                             h->il2d_col.nst, h->il2d_col.R, h->il2d_col.L, fns,
                             tabs, !a->fwd);
            if (h->il2d_col.tfuse && !hookb)
                _il2d_rows_exec(h, a->tid, a->dir, bd, rn, rn, b0, 1, (size_t)h->il2d_col.wl);
        }
        break;
    case 1: /* column strip: the whole chain over [lo,hi) columns, in
             * sub-strips of msw columns when the verdict sized them (an
             * L2-resident strip is the column pass in ONE sweep:
             * il2d_large_plane_design.md, 2026-09-15) */
        {
            const size_t sw = h->il2d_col.msw > 0 ? (size_t)h->il2d_col.msw : a->hi - a->lo;
            size_t k;
            for (k = a->lo; k < a->hi; k += sw)
                _il2d_col_pass_range(a->src, a->dst, h->N, rn, k,
                                     (k + sw < a->hi) ? k + sw : a->hi,
                                     h->il2d_col.nst, h->il2d_col.R, h->il2d_col.L, fns,
                                     tabs, !a->fwd);
        }
        break;
    case 5: /* natural x MT, the STRIP arm: the whole natural pass over
             * [lo,hi) columns in sub-strips of msw when sized, each through
             * the worker's DENSE strip scratch (pitch = the strip width: one
             * contiguous N1 x w block in L2, the 3D tier's strip form) when
             * the width fits it (2026-09-24: 512x128 T=8 75 -> 37 us, 1024x128
             * 139 -> 64, the shared full-pitch scratch put every strip piece
             * in the input rows' L2 sets); VFFT_IL2D_DENSE=0 (the probe pin)
             * keeps the shared scratch */
        {
            const size_t sw = h->il2d_col.msw > 0 ? (size_t)h->il2d_col.msw : a->hi - a->lo;
            const int dense = h->il2d_col.natdense && h->il2d_col.natsscr && a->tid < h->il2d_col.nnatsscr;
            size_t k;
            for (k = a->lo; k < a->hi; k += sw)
            {
                const size_t w = (k + sw < a->hi) ? sw : a->hi - k;
                if (dense && w <= (size_t)h->il2d_col.natswcap)
                    _il2d_col_pass_nat_strip(a->src, a->dst, h->N, rn, k, w,
                                             h->il2d_col.nst, h->il2d_col.R, h->il2d_col.L, fns,
                                             tabs, !a->fwd, h->il2d_col.natperm,
                                             h->il2d_col.natsscr[a->tid > 0 ? a->tid : 0]);
                else
                    _il2d_col_pass_nat_range(a->src, a->dst, h->N, rn, k, k + w,
                                             h->il2d_col.nst, h->il2d_col.R, h->il2d_col.L, fns,
                                             tabs, !a->fwd, h->il2d_col.natperm,
                                             h->il2d_col.natscr, _il2d_nat_stage_of(h, a->tid));
            }
        }
        break;
    case 4: /* natural x MT: the leaf scatter (fwd: src = scratch, dst =
             * plane) / gather (bwd: src = natural plane, dst = scratch)
             * over [lo,hi) blocks */
        _il2d_nat_leaf_blocks(h, a->tid, a->dir, a->src, a->dst, a->lo, a->hi, /*fuse=*/a->fwd);
        break;
    case 3: /* Bluestein column axis: the window pipeline (2026-09-02) */
        _il2d_blu_cols_range(a->src, a->dst, h->N, rn, a->lo, a->hi,
                             h->il2d_col.blu, h->il2d_col.nst, h->il2d_col.R, h->il2d_col.L,
                             h->il2d_col.f, h->il2d_col.b, h->il2d_col.tf, h->il2d_col.tb,
                             a->fwd ? h->il2d_col.bluchf : h->il2d_col.bluchb,
                             a->fwd ? h->il2d_col.blukf : h->il2d_col.blukb,
                             h->il2d_col.bluscr);
        break;
    case 6: /* the threaded SKEWED column pass (2026-09-24): the single column
             * stage over columns [16 lo, 16 hi) of the plane into the skewed
             * scratch (16-lane blocks: the column kernels' lane grain) */
        (a->fwd ? h->il2d_csk_f : h->il2d_csk_b)(a->src + 2 * 16 * a->lo, NULL,
                                                  h->il2d_csk_scr + 2 * 16 * a->lo, NULL, NULL, NULL,
                                                  rn, 0, VFFT_IL2D_CSK_PITCH(rn), 0, 16 * (a->hi - a->lo));
        break;
    case 7: /* the skewed pass's rows [lo, hi) from the scratch into the
             * destination plane, worker tid's row state */
        _il2d_rows_exec2(h, a->tid, a->dir, h->il2d_csk_scr + 2 * a->lo * VFFT_IL2D_CSK_PITCH(rn),
                         VFFT_IL2D_CSK_PITCH(rn), a->dst + 2 * a->lo * rn, rn, rn, a->lo, 1, a->hi - a->lo);
        break;
    case 8: /* the threaded TURN (2026-09-24), phase 1: rows [4 lo, 4 hi) through
             * the batched mono kernel with turned stores into the N2 x P scratch */
        (a->fwd ? h->il2d_rowb_f : h->il2d_rowb_b)(a->src + 2 * 4 * a->lo * rn, NULL,
                                                    h->il2d_turn_scr + 2 * 4 * a->lo, NULL, NULL, NULL,
                                                    1, rn, VFFT_IL2D_TURN_PITCH((size_t)h->N), 1, 4 * (a->hi - a->lo));
        break;
    case 9: /* turn, phase 2: the scratch rows [lo, hi) (the plane's columns)
             * through worker tid's in-place K=1 plan at N1 */
        {
            struct vfft_plan_s *tp = (a->tid > 0) ? h->il2d_turnw[a->tid - 1] : h->il2d_turn_plan;
            const size_t P = VFFT_IL2D_TURN_PITCH((size_t)h->N);
            size_t c;
            for (c = a->lo; c < a->hi; c++)
                vfft_execute((vfft_plan)tp, a->dir, h->il2d_turn_scr + 2 * c * P, NULL,
                             h->il2d_turn_scr + 2 * c * P, NULL);
        }
        break;
    case 11: /* the TILE arm (2026-09-25), forward: tiles [lo,hi) = the first
              * stage's sub-problems (N/R0 rows each, the scratch): the middle
              * stages in place on the tile, then its leaf blocks with the
              * rows fused (and streamed when staged) */
        {
            const int R0 = h->il2d_col.R[0], Rl = h->il2d_col.R[h->il2d_col.nst - 1];
            const size_t D0 = (size_t)h->N / (size_t)R0;
            size_t t;
            for (t = a->lo; t < a->hi; t++)
            {
                double *tile = (double *)a->src + 2 * t * D0 * rn;   /* the scratch: writable */
                if (h->il2d_col.nst > 2)
                    _il2d_col_stages2(tile, tile, (int)D0, rn, rn, 1, h->il2d_col.nst - 1,
                                      h->il2d_col.R, h->il2d_col.L, fns, tabs, 0);
                _il2d_nat_leaf_blocks(h, a->tid, a->dir, a->src, a->dst,
                                      t * D0 / (size_t)Rl, (t + 1) * D0 / (size_t)Rl, 1);
            }
        }
        break;
    case 12: /* the tile arm, backward: the leaf gathers the tile's natural rows
              * into the scratch, then the middle stages reversed in place */
        {
            const int R0 = h->il2d_col.R[0], Rl = h->il2d_col.R[h->il2d_col.nst - 1];
            const size_t D0 = (size_t)h->N / (size_t)R0;
            size_t t;
            for (t = a->lo; t < a->hi; t++)
            {
                double *tile = a->dst + 2 * t * D0 * rn;
                _il2d_nat_leaf_blocks(h, a->tid, a->dir, a->src, a->dst,
                                      t * D0 / (size_t)Rl, (t + 1) * D0 / (size_t)Rl, 0);
                if (h->il2d_col.nst > 2)
                    _il2d_col_stages2(tile, tile, (int)D0, rn, rn, 1, h->il2d_col.nst - 1,
                                      h->il2d_col.R, h->il2d_col.L, fns, tabs, 1);
            }
        }
        break;
    case 10: /* turn, phase 3: the back-turn of destination rows [lo, hi) */
        _il2d_turn_back_range(h->il2d_turn_scr, VFFT_IL2D_TURN_PITCH((size_t)h->N), a->dst,
                              rn, a->lo, a->hi);
        break;
    default: /* row slab on the destination plane */
        _il2d_rows_exec(h, a->tid, a->dir, a->dst + 2 * a->lo * rn, rn, rn, a->lo, 1, a->hi - a->lo);
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

/* ── the THREADED skewed column pass and turn (2026-09-24). Both are loop
 * restrictions of their serial executes (the same kernels on disjoint
 * ranges: MT == ST bitwise). csk: the column stage over 16-lane column
 * blocks, then the rows over row slabs -- route 0 through worker t's clone
 * of the out-of-place row plan, the batched routes through stateless
 * kernels on their own slots. turn: the turned rows over row slabs, the
 * scratch rows (the plane's columns) through worker t's clone of the N1
 * plan, the back-turn over destination row slabs. Return 1 when threaded,
 * 0 when the caller must run serial (the engagement counter shows it). */
static int _il2d_csk_exec_mt(struct vfft_plan_s *h, const double *sre, double *dre, vfft_dir_t dir, int T)
{
    const size_t rn = (size_t)h->N2, N1 = (size_t)h->N;
    const int fwd = (dir == VFFT_FORWARD);
    int Tc, Tr;
    T = stride_pool_workers_for(T);
    if (T < 2 || (rn % 16) != 0)
        return 0;
    if (!h->il2d_rowb && !h->il2d_rowb2 && h->il2d_cskw_n < T - 1)
        return 0; /* route 0 runs the OOP row plan: clones are mandatory */
    Tc = (rn / 16 < (size_t)T) ? (int)(rn / 16) : T;
    Tr = (N1 < (size_t)T) ? (int)N1 : T;
    if (Tc < 2 && Tr < 2)
        return 0;
    if (Tc >= 2)
        _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 6, rn / 16, Tc);
    else
        (fwd ? h->il2d_csk_f : h->il2d_csk_b)(sre, NULL, h->il2d_csk_scr, NULL, NULL, NULL,
                                              rn, 0, VFFT_IL2D_CSK_PITCH(rn), 0, rn);
    if (Tr >= 2)
        _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 7, N1, Tr);
    else
        _il2d_rows_exec2(h, 0, dir, h->il2d_csk_scr, VFFT_IL2D_CSK_PITCH(rn), dre, rn, rn, 0, 1, N1);
    _vfft_il2d_col_mt_count++;
    return 1;
}
static int _il2d_turn_exec_mt(struct vfft_plan_s *h, const double *sre, double *dre, vfft_dir_t dir, int T)
{
    const size_t rn = (size_t)h->N2, N1 = (size_t)h->N, P = VFFT_IL2D_TURN_PITCH(N1);
    const int fwd = (dir == VFFT_FORWARD);
    int Tr, Tc;
    size_t c;
    T = stride_pool_workers_for(T);
    if (T < 2 || (N1 % 4) != 0)
        return 0;
    Tc = (rn < (size_t)T) ? (int)rn : T;
    if (h->il2d_turnw_n < Tc - 1)
        return 0; /* the scratch rows run the N1 plan: clones are mandatory */
    Tr = (N1 / 4 < (size_t)T) ? (int)(N1 / 4) : T;
    if (Tr >= 2)
        _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 8, N1 / 4, Tr);
    else
        (fwd ? h->il2d_rowb_f : h->il2d_rowb_b)(sre, NULL, h->il2d_turn_scr, NULL, NULL, NULL, 1, rn, P, 1, N1);
    if (Tc >= 2)
        _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 9, rn, Tc);
    else
        for (c = 0; c < rn; c++)
            vfft_execute((vfft_plan)h->il2d_turn_plan, dir, h->il2d_turn_scr + 2 * c * P, NULL,
                         h->il2d_turn_scr + 2 * c * P, NULL);
    if (Tr >= 2)
        _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 10, N1, T);
    else
        _il2d_turn_back(h->il2d_turn_scr, P, dre, N1, rn);
    _vfft_il2d_col_mt_count++;
    return 1;
}

/* VFFT_IL2D_PHASES read once: no getenv on the execute path (the CRT's
 * environment lock serialised the workers' phase starts, 2026-09-25) */
static int _il2d_phlog_flag = -1;
static inline int _il2d_phlog_on(void)
{
    if (_il2d_phlog_flag < 0)
        _il2d_phlog_flag = getenv("VFFT_IL2D_PHASES") != NULL;
    return _il2d_phlog_flag;
}
/* Returns 1 when it ran threaded, 0 when the caller must run serial. */
static int _il2d_c2c_mt(struct vfft_plan_s *h, const double *sre,
                        double *dre, vfft_dir_t dir, int T)
{
    const size_t rn = (size_t)h->N2;
    const int fwd = (dir == VFFT_FORWARD);
    int s;
    if (h->il2d_turn)
        return _il2d_turn_exec_mt(h, sre, dre, dir, T); /* the turn's own walk (2026-09-24) */
    if (h->il2d_csk)
        return _il2d_csk_exec_mt(h, sre, dre, dir, T);  /* the skewed pass's own walk (2026-09-24) */
    if (h->il2d_col.tpc)
        return 0; /* the turned prime pass: one 1D plan, serial (2026-09-24) */
    if (h->il2d_col.staged)
        return 0; /* env-experimental route: one shared band scratch —
                   * per-worker slots are not built for it */
    /* T arrives as the plan's snapshot (h->nthreads); the pool's one clamp
     * bounds it by the live pool and the arg-array size. */
    T = stride_pool_workers_for(T);
    if (T < 2 || h->il2d_roww_n < T - 1)
        return 0; /* every arm here runs rows => clones are mandatory */
    if (h->il2d_col.blu)
    {   /* Bluestein column axis (2026-09-02): column windows, then rows —
         * the same order the unbanded chain walk uses */
        const int Ts = rn < (size_t)T ? (int)rn : T;
        const int Tr = (size_t)h->N < (size_t)T ? h->N : T;
        if (Ts < 2 && Tr < 2)
            return 0;
        if (Ts >= 2)
            _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 3, rn, Ts);
        else
            _il2d_blu_cols(sre, dre, h->N, rn, h->il2d_col.blu, h->il2d_col.nst,
                           h->il2d_col.R, h->il2d_col.L, h->il2d_col.f, h->il2d_col.b,
                           h->il2d_col.tf, h->il2d_col.tb,
                           fwd ? h->il2d_col.bluchf : h->il2d_col.bluchb,
                           fwd ? h->il2d_col.blukf : h->il2d_col.blukb,
                           h->il2d_col.bluscr);
        _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 2, (size_t)h->N, Tr);
        _vfft_il2d_col_mt_count++;
        return 1;
    }
    if (h->il2d_col.nat)
    {
        /* NATURAL x MT (2026-09-04): digit-split prefix (sre -> scratch,
         * then in place), the leaf scatter by BLOCK RANGE (mode 4,
         * scratch -> dre), then row slabs on dre; bwd mirrors (gather
         * first, reversed prefix, stage 0 scratch -> dre). The band arm
         * is structurally out (the scatter crosses bands). */
        const int Rl = h->il2d_col.R[h->il2d_col.nst - 1];
        const size_t nb = (size_t)h->N / (size_t)Rl;
        const int Tb = nb < (size_t)T ? (int)nb : T;
        const int Tr = (size_t)h->N < (size_t)T ? h->N : T;
        double *scr = h->il2d_col.natscr;
        int rows_done = 0;
        if (h->il2d_col.nst < 2 || (Tb < 2 && Tr < 2))
            return 0;
        if (h->il2d_col.natarm == 2)
        {   /* the TILE arm (2026-09-25): stage 0 across the plane (its digit is
             * the row within a first-stage sub-problem), then every sub-problem
             * a tile one worker owns -- the middle stages in L2, the leaf's
             * natural rows with the row transform fused; the plane crosses
             * cores once each way */
            const int R0 = h->il2d_col.R[0];
            const size_t D0 = (size_t)h->N / (size_t)R0;
            const int Tt = (R0 < T) ? R0 : T;
            if (R0 < 2 || Tt < 2 || D0 % (size_t)Rl)
                return 0;
            if (fwd)
            {
                if (!_il2d_stage_digits_mt(sre, scr, h->N, rn, rn, R0, h->il2d_col.L[0],
                                           h->il2d_col.f[0], h->il2d_col.tf[0], T))
                    _il2d_col_stages(sre, scr, h->N, rn, 0, 1, h->il2d_col.R, h->il2d_col.L,
                                     h->il2d_col.f, h->il2d_col.tf, 0);
                _il2d_c2c_mt_phase(h, scr, dre, dir, fwd, 11, (size_t)R0, Tt);
            }
            else
            {
                _il2d_c2c_mt_phase(h, sre, scr, dir, fwd, 12, (size_t)R0, Tt);
                if (!_il2d_stage_digits_mt(scr, dre, h->N, rn, rn, R0, h->il2d_col.L[0],
                                           h->il2d_col.b[0], h->il2d_col.tb[0], T))
                    _il2d_col_stages(scr, dre, h->N, rn, 0, 1, h->il2d_col.R, h->il2d_col.L,
                                     h->il2d_col.b, h->il2d_col.tb, 0);
                _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 2, (size_t)h->N, Tr);   /* the backward's rows after its columns */
            }
            _vfft_il2d_col_mt_count++;
            return 1;
        }
        if (h->il2d_col.natarm == 1)
        {   /* the STRIP arm (raced against the block arm at create) */
            const int Ts = rn < (size_t)T ? (int)rn : T;
            const int phlog = _il2d_phlog_on();   /* per-phase ns, as the block arm prints */
            double p0 = 0, p1 = 0;
            if (Ts < 2 && Tr < 2)
                return 0;
            if (phlog) p0 = _il_ab_now();
            if (Ts >= 2)
                _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 5, rn, Ts);
            else
                _il2d_col_pass_nat(sre, dre, h->N, rn, h->il2d_col.nst,
                                   h->il2d_col.R, h->il2d_col.L,
                                   fwd ? h->il2d_col.f : h->il2d_col.b,
                                   fwd ? h->il2d_col.tf : h->il2d_col.tb, !fwd,
                                   h->il2d_col.natperm, scr, _il2d_nat_stage_of(h, 0));
            if (phlog) p1 = _il_ab_now();
            _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 2, (size_t)h->N, Tr);
            if (phlog)
                fprintf(stderr, "[il2d-phases] %dx%zu T=%d strips msw=%d nls=%d: cols=%.0f rows=%.0f ns\n",
                        h->N, rn, T, h->il2d_col.msw, h->il2d_col.natst, p1 - p0, _il_ab_now() - p1);
            _vfft_il2d_col_mt_count++;
            return 1;
        }
        if (fwd)
        {
            for (s = 0; s < h->il2d_col.nst - 1; s++)
            {
                const double *ssrc = (s == 0) ? sre : scr;
                if (!_il2d_stage_digits_mt(ssrc, scr, h->N, rn, rn,
                                           h->il2d_col.R[s], h->il2d_col.L[s],
                                           h->il2d_col.f[s], h->il2d_col.tf[s], T))
                    _il2d_col_stages(ssrc, scr, h->N, rn, s, s + 1,
                                     h->il2d_col.R, h->il2d_col.L, h->il2d_col.f,
                                     h->il2d_col.tf, 0);
            }
            if (Tb >= 2)
                _il2d_c2c_mt_phase(h, scr, dre, dir, fwd, 4, nb, Tb);
            else
                _il2d_nat_leaf_blocks(h, 0, dir, scr, dre, 0, nb, 1);
            rows_done = 1;   /* the block arm fused its rows in the leaf */
        }
        else
        {
            if (Tb >= 2)
                _il2d_c2c_mt_phase(h, sre, scr, dir, fwd, 4, nb, Tb);
            else
                _il2d_nat_leaf_blocks(h, 0, dir, sre, scr, 0, nb, 0);
            for (s = h->il2d_col.nst - 2; s >= 0; s--)
            {
                double *out = (s == 0) ? dre : scr;
                if (!_il2d_stage_digits_mt(scr, out, h->N, rn, rn,
                                           h->il2d_col.R[s], h->il2d_col.L[s],
                                           h->il2d_col.b[s], h->il2d_col.tb[s], T))
                    _il2d_col_stages(scr, out, h->N, rn, s, s + 1,
                                     h->il2d_col.R, h->il2d_col.L, h->il2d_col.b,
                                     h->il2d_col.tb, 0);
            }
        }
        if (!rows_done)
            _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 2, (size_t)h->N, Tr);
        _vfft_il2d_col_mt_count++;
        return 1;
    }
    if (h->il2d_col.wl > 0 && h->il2d_col.natarm != 1)
    {   /* (a strips verdict — natarm == 1, every class since 2026-09-15 —
         * takes the unbanded strip walk below whatever the serial band) */
        /* fewer bands than workers is a CLAMP, not a decline: 4 bands
         * across 4 workers still beats serial, and the prefix digit
         * split keeps the full T regardless (its axis is D, not nb). */
        const size_t nb = (size_t)h->N / (size_t)h->il2d_col.wl;
        const int Tb = nb < (size_t)T ? (int)nb : T;
        /* VFFT_IL2D_PHASES: per-phase ns of this walk on stderr (a probe
         * instrument for the large-plane question, 2026-09-15) */
        const int phlog = _il2d_phlog_on();
        double ph0 = 0, ph1 = 0, ph2 = 0, ph3 = 0;
        if (Tb < 2)
            return 0;
        if (phlog) ph0 = _il_ab_now();
        if (fwd && h->il2d_col.cut > 0)
            for (s = 0; s < h->il2d_col.cut; s++)
            {
                const double *ssrc = (s == 0) ? sre : dre;
                if (!_il2d_stage_digits_mt(ssrc, dre, h->N, rn, rn,
                                           h->il2d_col.R[s], h->il2d_col.L[s],
                                           h->il2d_col.f[s], h->il2d_col.tf[s],
                                           T))
                    _il2d_col_stages(ssrc, dre, h->N, rn, s, s + 1,
                                     h->il2d_col.R, h->il2d_col.L, h->il2d_col.f,
                                     h->il2d_col.tf, 0);
            }
        if (h->il2d_fs_tw && !fwd && !h->il2d_col.tfuse)
        {   /* the four-step's backward: the rows (conjugate twiddle) before
             * any column stage — on dst, moved there first out of place */
            if (dre != sre)
                memcpy(dre, sre, 2 * (size_t)h->N * rn * sizeof(double));
            _il2d_c2c_mt_phase(h, dre, dre, dir, fwd, 2, (size_t)h->N, T);
            sre = dre;
        }
        if (phlog) ph1 = _il_ab_now();
        _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 0, nb, Tb);
        if (phlog) ph2 = _il_ab_now();
        if (!h->il2d_col.tfuse && !(h->il2d_fs_tw && !fwd))
            _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 2, (size_t)h->N,
                               T);
        if (phlog)
        {
            ph3 = _il_ab_now();
            fprintf(stderr, "[il2d-phases] %dx%d T=%d wl=%d cut=%d nst=%d tfuse=%d: prefix=%.0f bands=%.0f rows=%.0f ns\n",
                    h->N, (int)rn, T, h->il2d_col.wl, h->il2d_col.cut, h->il2d_col.nst, h->il2d_col.tfuse,
                    ph1 - ph0, ph2 - ph1, ph3 - ph2);
        }
        if (!fwd && h->il2d_col.cut > 0)
            for (s = h->il2d_col.cut - 1; s >= 0; s--)
                if (!_il2d_stage_digits_mt(dre, dre, h->N, rn, rn,
                                           h->il2d_col.R[s], h->il2d_col.L[s],
                                           h->il2d_col.b[s], h->il2d_col.tb[s],
                                           T))
                    _il2d_col_stages(dre, dre, h->N, rn, s, s + 1,
                                     h->il2d_col.R, h->il2d_col.L, h->il2d_col.b,
                                     h->il2d_col.tb, 0);
        _vfft_il2d_col_mt_count++; /* engagement, see vfft.h */
        return 1;
    }
    /* unbanded: column strips, then row slabs (rows follow the column
     * pass in the serving order for BOTH directions — rows commute; the
     * four-step's backward runs its twiddled rows FIRST) */
    {
        const int Ts = rn < (size_t)T ? (int)rn : T;
        const int Tr = (size_t)h->N < (size_t)T ? h->N : T;
        if (Ts < 2 && Tr < 2)
            return 0;
        if (h->il2d_fs_tw && !fwd)
        {
            if (dre != sre)
                memcpy(dre, sre, 2 * (size_t)h->N * rn * sizeof(double));
            _il2d_c2c_mt_phase(h, dre, dre, dir, fwd, 2, (size_t)h->N, Tr);
            if (Ts >= 2)
                _il2d_c2c_mt_phase(h, dre, dre, dir, fwd, 1, rn, Ts);
            else
                _il2d_col_pass(dre, dre, h->N, rn, rn, h->il2d_col.nst,
                               h->il2d_col.R, h->il2d_col.L, h->il2d_col.b,
                               h->il2d_col.tb, 1);
            _vfft_il2d_col_mt_count++;
            return 1;
        }
        if (Ts >= 2)
            _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 1, rn, Ts);
        else
            _il2d_col_pass(sre, dre, h->N, rn, rn, h->il2d_col.nst,
                           h->il2d_col.R, h->il2d_col.L,
                           fwd ? h->il2d_col.f : h->il2d_col.b,
                           fwd ? h->il2d_col.tf : h->il2d_col.tb, !fwd);
        _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 2, (size_t)h->N, Tr);
    }
    _vfft_il2d_col_mt_count++;
    return 1;
}

/* derive the banded walk's cut for a wl candidate: the first suffix
 * stage whose span divides wl. -1 = illegal (stay unbanded). */
static int _il2d_real_wl_cut(const struct vfft_plan_s *h, int wl)
{   /* the tcut law lives in planning/policy.h (R3, 2026-09-17) */
    return vfft_policy_il2d_wl_cut(h->N, h->il2d_col.nst, h->il2d_col.L, wl);
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
    int sw;                   /* the strips arm's sub-strip width (0 = unsized) */
    double *nstage;           /* the natural leaf's staging for the chain arm */
    int lst;                  /* the natural leaf: staged (1) or strided (0) for this arm */
    double *zo;               /* the threading race's destination plane: the cell's own placement (2026-09-24) */
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
    c->h->il2d_col.colmt = 0;
    c->h->il2d_col.natst = 1;      /* the serial walk's form: staged */
    vfft_execute((vfft_plan)c->h, VFFT_FORWARD, c->z, NULL, c->zo ? c->zo : c->z, NULL);
}
static void _il2d_arm_exec_mt(void *v)
{
    _il2d_race_ctx_t *c = (_il2d_race_ctx_t *)v;
    c->h->il2d_col.colmt = 1;
    c->h->il2d_col.natarm = 0;
    c->h->il2d_col.natst = c->lst;
    vfft_execute((vfft_plan)c->h, VFFT_FORWARD, c->z, NULL, c->zo ? c->zo : c->z, NULL);
}
static void _il2d_arm_exec_mt_strip(void *v)
{   /* natural cells only: the strip partition of the natural pass */
    _il2d_race_ctx_t *c = (_il2d_race_ctx_t *)v;
    c->h->il2d_col.colmt = 1;
    c->h->il2d_col.natarm = 1;
    c->h->il2d_col.msw = 0;
    c->h->il2d_col.natst = c->lst;
    vfft_execute((vfft_plan)c->h, VFFT_FORWARD, c->z, NULL, c->zo ? c->zo : c->z, NULL);
}
static void _il2d_arm_exec_mt_tile(void *v)
{   /* natural cells: the TILE partition (2026-09-25) */
    _il2d_race_ctx_t *c = (_il2d_race_ctx_t *)v;
    c->h->il2d_col.colmt = 1;
    c->h->il2d_col.natarm = 2;
    c->h->il2d_col.msw = 0;
    c->h->il2d_col.natst = c->lst;
    vfft_execute((vfft_plan)c->h, VFFT_FORWARD, c->z, NULL, c->zo ? c->zo : c->z, NULL);
}
static void _il2d_arm_exec_mt_sw(void *v)
{   /* every class: strips of c->sw columns (il2d_large_plane_design.md) */
    _il2d_race_ctx_t *c = (_il2d_race_ctx_t *)v;
    c->h->il2d_col.colmt = 1;
    c->h->il2d_col.natarm = 1;
    c->h->il2d_col.msw = c->sw;
    c->h->il2d_col.natst = c->lst;
    vfft_execute((vfft_plan)c->h, VFFT_FORWARD, c->z, NULL, c->zo ? c->zo : c->z, NULL);
}
/* the strip-width ladder: N1 x sw x 16 B under L2 (the hardware fence),
 * sw a divisor of N2, and at T > 1 at least T sub-strips. 8 joined on
 * 2026-09-25: at N1 = 8192 the 16-column dense strip is the whole L2 and
 * streamed every stage through L3 (8192x128 0.76 at T=8); the column
 * kernels take any count >= 1. */
static int _il2d_sw_ladder(int N1, int N2, int T, int *out, int max)
{
    static const int SW[] = { 8, 16, 32, 64, 128, 256 };
    int i, n = 0;
    for (i = 0; i < 6 && n < max; i++)
    {
        const int w = SW[i];
        if (w > N2 || N2 % w) continue;
        if (!vfft_policy_fits_l2((long)N1 * w * 16)) continue;
        if (T > 1 && N2 / w < T) continue;
        out[n++] = w;
    }
    return n;
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
                           c->ff, c->tf, /*reverse=*/0, c->perm, c->nscr, c->nstage);
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
    const vfft_race_proto_t proto = { 3, 1, VFFT_RACE_MIN, 0, 0, NULL, NULL, 1 }; /* min-of-3, A then B */ /* single-thread arms: paced (VFFT_RACE_PACE_MS) */
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
        const size_t hp1 = (size_t)N2 / 2 + 1;
        int wlc[14], nwl = 0, wi, s2;
        double cbest = 1e300;
        int bwl = 0, bcut = 0;
        h->il2d_col.wl = 0;
        h->il2d_col.cut = 0;
        vfft_race_run(&proto, &cols_arm, 1, &cbest);
        for (wi = 0; wi < VFFT_IL2D_WL_LADDER_N && nwl < 14; wi++)
            if (_il2d_real_wl_cut(h, VFFT_IL2D_WL_LADDER[wi]) >= 0)   /* w == N1 admitted like c2c/3D (R2) */
                wlc[nwl++] = VFFT_IL2D_WL_LADDER[wi];
        for (s2 = 1; s2 < h->il2d_col.nst && nwl < 14; s2++)
        {
            const int w2 = h->il2d_col.L[s2];
            int dup = 0;
            if (!vfft_policy_il2d_band_ok(N1, h->il2d_col.nst, h->il2d_col.L, w2))
                continue;   /* R2: the real tier follows c2c/3D (floor 8; w == N1 admitted) */
            if (!vfft_policy_fits_l2((long)w2 * (long)hp1 * 16))
                continue;
            for (wi = 0; wi < nwl; wi++)
                if (wlc[wi] == w2)
                    dup = 1;
            if (!dup)
                wlc[nwl++] = w2;
        }
        if (getenv("VFFT_IL2D_LOG"))
        {   /* the LADDER, not just the winner: the census a pool change is
             * gated by (design R2, 2026-09-17) */
            fprintf(stderr, "[il2d-real] wl ladder %dx%d:", h->N, h->N2);
            for (wi = 0; wi < nwl; wi++) fprintf(stderr, " %d", wlc[wi]);
            fputc(10, stderr);
        }
        for (wi = 0; wi < nwl; wi++)
        {
            const int cut = _il2d_real_wl_cut(h, wlc[wi]);
            double ns = 1e300;
            h->il2d_col.wl = wlc[wi];
            h->il2d_col.cut = cut;
            vfft_race_run(&proto, &cols_arm, 1, &ns);
            if (ns < cbest)
            {
                cbest = ns;
                bwl = wlc[wi];
                bcut = cut;
            }
        }
        h->il2d_col.wl = bwl;
        h->il2d_col.cut = bcut;
        free(a);
        free(bz);
        if (getenv("VFFT_IL2D_LOG"))
            fprintf(stderr, "[il2d-real] rowrace %s %dx%d -> rw=%d "
                            "wl=%d (%.0f ns rows / %.0f ns cols)\n",
                    isr ? "r2c" : "c2r", N1, N2, bw, bwl, bestns,
                    cbest);
        vw2_2d_rl_bank(&W->vw2, N1, N2, !isr, h->il2d_col.R, h->il2d_col.nst, bw,
                       bwl, -1, -1, (N1 & (N1 - 1)) ? h->il2d_col.blu : -1,
                       bestns + cbest, vfft_policy_ord_rankn(cfg));
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
        const vfft_race_proto_t proto = { 3, 1, VFFT_RACE_MIN, 0, 2, NULL, NULL, 0 }; /* min-of-3, A then B, two untimed passes per arm first (2026-09-25: a cold threaded arm read 58 us for a 37-us plan) */ /* THREADED arms: never paused (mt_measurement_parking_trap) */
        double ns[2];
        (void)p;
        vfft_race_run(&proto, arms, 2, ns);
        st = ns[0];
        mt = ns[1];
        if (!rc.ok)
        {
            /* the threaded arm cannot even engage on this cell */
            free(z);
            h->il2d_col.colmt = 0;
            vw2_2d_rl_bank(&W->vw2, N1, N2, h->transform == VFFT_C2R,
                           h->il2d_col.R, h->il2d_col.nst,
                           h->il2d_rw, h->il2d_col.wl, 0, h->nthreads,
                           (N1 & (N1 - 1)) ? h->il2d_col.blu : -1, st, vfft_policy_ord_rankn(cfg));
            _vw2_persist(W, cfg);
            return;
        }
    }
    h->il2d_col.colmt = (mt < st);
    free(z);
    if (getenv("VFFT_IL2D_LOG"))
        fprintf(stderr, "[il2d-real] colmt race %dx%d T=%d: st=%.0f "
                        "mt=%.0f -> %s\n",
                N1, N2, h->nthreads, st, mt,
                h->il2d_col.colmt ? "THREADED" : "serial");
    vw2_2d_rl_bank(&W->vw2, N1, N2, h->transform == VFFT_C2R,
                   h->il2d_col.R, h->il2d_col.nst, h->il2d_rw,
                   h->il2d_col.wl, h->il2d_col.colmt, h->nthreads,
                   (N1 & (N1 - 1)) ? h->il2d_col.blu : -1,
                   h->il2d_col.colmt ? mt : st, vfft_policy_ord_rankn(cfg));
    _vw2_persist(W, cfg);
}


/* the chain RACE: time every candidate's column pass on scratch (min of
 * 3 passes), return the winner's index. -1 = race impossible. */
/* the Bluestein inner's chain at M: the (M, N2) 2D chain row — replayed
 * when banked (prod == M), else the E1.1 chain race at (M, N2), banked
 * there. Context = the create in progress (set at _vfft_create_2d's entry). */
static struct {
    struct vfft_wisdom_s *W;
    const vfft_config_t *cfg;
    int N2;
    /* D1 (2026-09-18): the column-axis Bluestein's INNER chain at M is the
     * cell's verdict and banks on THE CELL'S OWN row -- never on the
     * (M, N2) row, which is the row a user's own scrambled M x N2 cell owns.
     * key   = that row;
     * rep_R = the M chain the builder already read from it (replay);
     * commit= this build IS the cell's serving plan, so the chain and blu=M
     *         bank here (a speculative N-arm arm banks nothing). */
    const vw2_ilcol_key_t *key;
    const int *rep_R;
    int rep_nst;
    int commit;
} _il2d_blu_ctx;
static int _il2d_race_chains(int N1, int N2, int ncand, int (*cand)[8],
                             const int *lens, double *best_ns, int nat);
static int _il2d_race_forms(int N1, int N2, const int *Rs, int nst,
                            vfft_il2p_fn *ff, vfft_il2p_fn *fb, char *forms,
                            size_t fsz);
static void _il2d_forms_serve_key(struct vfft_wisdom_s *W,
                                  const vfft_config_t *cfg,
                                  const vw2_ilcol_key_t *key, const char *base,
                                  int N, size_t rn,
                                  const int *Rs, int nst,
                                  vfft_il2p_fn *ff, vfft_il2p_fn *fb,
                                  char *forms, size_t fsz);
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
    const vw2_ilcol_key_t *key = _il2d_blu_ctx.key;
    vfft_il2p_fn ff[8], fb[8];
    forms[0] = 0;
    if (!W || W->vw2_off_2d || N2 <= 0 || !key) return 0;
    /* REPLAY -- the M chain THE CELL'S OWN row carries, read by the builder
     * (which checked prod == blu) and handed over here. Until 2026-09-18
     * this was a lookup, and the race below a bank, on the (M, N2, scr) row:
     * the row a user's own scrambled M x N2 cell owns. The bank passed a
     * positive time with every axis verdict at -1, which is the REPLACE path
     * of vw2_2d_il_chain_bank, so a user cell's width, form and column-MT
     * verdicts were wiped and re-raced on its next create (D1). */
    if (_il2d_blu_ctx.rep_R && _il2d_blu_ctx.rep_nst > 0 &&
        _il2d_chain_prod(_il2d_blu_ctx.rep_R, _il2d_blu_ctx.rep_nst) == M)
    {
        memcpy(Rs, _il2d_blu_ctx.rep_R,
               (size_t)_il2d_blu_ctx.rep_nst * sizeof(int));
        *nst = _il2d_blu_ctx.rep_nst;
        if (getenv("VFFT_IL2D_LOG"))
            fprintf(stderr, "[il2d] blu inner M=%d x %d: replay chain src=wisdom\n", M, N2);
        if (_il2d_resolve(Rs, *nst, ff, fb))
            _il2d_forms_serve_key(W, cfg, key, "bluforms", M, (size_t)N2,
                                  Rs, *nst, ff, fb, forms, fsz);
        return 1;
    }
    {
        int cand[VFFT_IL2D_MAXCAND][8], lens[VFFT_IL2D_MAXCAND];
        int cur[8], ncand = 0, dropped = 0, win;
        double bns = 0;
        _il2d_enum_rec(M, 0, cur, cand, lens, &ncand, &dropped);
        if (ncand < 1) return 0;   /* (a capped pool warns from inside the enumerator) */
        win = (ncand > 1) ? _il2d_race_chains(M, N2, ncand, cand, lens, &bns, 0) : 0;
        if (win < 0) return 0;
        memcpy(Rs, cand[win], sizeof cand[win]);
        *nst = lens[win];
        /* the winner banks on THE CELL'S row as chain= + blu=M, ns = 0 so an
         * existing row is FIELD-UPDATED and its other verdicts survive. Only
         * when this build is the cell's plan: a speculative N-arm arm leaves
         * chain= naming the N chain until the race picks a winner. */
        if (_il2d_blu_ctx.commit)
        {
            vw2_ilcol_chain_bank(&W->vw2, key, Rs, *nst,
                                 -1, -1, -1, -1, -1, M, 0.0);
            _vw2_persist(W, cfg);
        }
        if (getenv("VFFT_IL2D_LOG"))
            fprintf(stderr, "[il2d] blu inner M=%d x %d: chain race -> %d candidates,"
                            " winner %s\n", M, N2, ncand,
                    _il2d_blu_ctx.commit ? "banked on the cell's row"
                                         : "(N-arm arm, unbanked)");
        if (_il2d_resolve(Rs, *nst, ff, fb))
            _il2d_forms_serve_key(W, cfg, key, "bluforms", M, (size_t)N2,
                                  Rs, *nst, ff, fb, forms, fsz);
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
    z = (double *)VFFT_ZS_ALLOC(2 * T * sizeof(double));   /* aligned like every plane the door serves (2026-09-23) */
    if (!z)
        return 0;
    for (i = 0; i < 2 * T; i++)
        z[i] = 1.0 + 1e-6 * (double)(i & 1023);
    if (_il2d_build_tables(N1, nst, Rs, Ls, tf, tb))
    {
        VFFT_ZS_FREE(z);
        return 0;
    }
    for (s = 0; s < nst; s++)
    {
        const char *nm[2];
        const int last = (s == nst - 1);
        vfft_il2p_fn ffa[2][8];
        _il2d_race_ctx_t rc[2];
        vfft_race_arm_t arm[2];
        const vfft_race_proto_t proto = { 3, 1, VFFT_RACE_MIN, 0, 0, NULL, NULL, 1 }; /* single-thread arms: paced (VFFT_RACE_PACE_MS) */
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
    VFFT_ZS_FREE(z);
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
{   /* ONE body (R5, 2026-09-17): vw2_2d_forms_lookup/bank are themselves
     * wrappers that build this key and call the ilcol pair, so the 2D-key
     * spelling of the precedence law (env pin > banked forms= > the
     * per-stage race) is the ilcol-key spelling with this key. The 45-line
     * twin that used to live here was byte-identical to _il2d_forms_serve_key
     * apart from the key. */
    const vw2_ilcol_key_t ck = { 2, N1, N2, 0, ord, 0, is_real };
    _il2d_forms_serve_key(W, cfg, &ck, "forms", N1, (size_t)N2, Rs, nst, ff, fb, forms, fsz);
}

static int _il2d_race_chains(int N1, int N2, int ncand, int (*cand)[8],
                             const int *lens, double *best_ns, int nat)
{
    const size_t T = (size_t)N1 * N2;
    double *z = (double *)VFFT_ZS_ALLOC(2 * T * sizeof(double));   /* aligned like every plane the door serves (2026-09-23) */
    double *nscr = nat ? (double *)VFFT_ZS_ALLOC(2 * T * sizeof(double)) : NULL;   /* aligned like every plane the door serves (2026-09-23) */
    double *nstage = nat ? (double *)VFFT_ZS_ALLOC(2 * 64 * (size_t)N2 * sizeof(double)) : NULL;   /* R_last <= 64 */
    int ci, win = -1;
    double wns = 1e300;
    size_t i;
    if (!z || (nat && !nscr))
    {
        VFFT_ZS_FREE(z);
        VFFT_ZS_FREE(nscr);
        VFFT_ZS_FREE(nstage);
        return -1;
    }
    for (i = 0; i < 2 * T; i++)
        z[i] = 1.0 + 1e-6 * (double)(i & 1023);
    /* Every buildable candidate is an ARM of ONE race (2026-09-06): the
     * tables and (natural) the permutation of all candidates are built up
     * front, then vfft_race_run samples every arm once per round with the
     * rounds alternating direction — the house protocol — so a drift of
     * the host (its throttled state lasts minutes) hits all chains alike.
     * The per-candidate loop it replaces timed each chain in its own burst
     * of three, one after another, and two cold runs at 1215x243 natural
     * banked different chains. Same samples per arm as before (min of 3
     * single executes); the arm count is VFFT_IL2D_MAXCAND <= the
     * template's VFFT_RACE_MAX_ARMS. */
    {
        struct
        {
            vfft_il2p_fn ff[8], fb[8];
            int Ls[8];
            double *tf[8], *tb[8];
            int *perm;
            int ci;
        } cb[VFFT_IL2D_MAXCAND];
        _il2d_race_ctx_t rc[VFFT_IL2D_MAXCAND];
        vfft_race_arm_t arms[VFFT_IL2D_MAXCAND];
        double ns[VFFT_IL2D_MAXCAND];
        int na = 0, a, s2;
        for (ci = 0; ci < ncand && na < VFFT_IL2D_MAXCAND; ci++)
        {
            int *perm = NULL;
            if (!_il2d_resolve(cand[ci], lens[ci], cb[na].ff, cb[na].fb))
                continue;
            if (nat && lens[ci] > 1)
            {
                perm = _il2d_nat_perm(cand[ci], lens[ci], N1);
                if (!perm)
                    continue; /* no natural leaf for this chain: not a candidate */
            }
            if (_il2d_build_tables(N1, lens[ci], cand[ci], cb[na].Ls, cb[na].tf, cb[na].tb))
            {
                free(perm);
                continue;
            }
            cb[na].perm = perm;
            cb[na].ci = ci;
            {
                _il2d_race_ctx_t c0 = { NULL, NULL, z, 0, 1, N1, (size_t)N2,
                                        lens[ci], cand[ci], cb[na].Ls, cb[na].ff, cb[na].tf,
                                        nat && perm != NULL, perm, nscr, 0, nstage };
                rc[na] = c0;
            }
            arms[na].name = "chain";
            arms[na].run = _il2d_arm_chain;
            arms[na].ctx = &rc[na];
            na++;
        }
        if (na > 0)
        {
            const vfft_race_proto_t proto = { 3, 1, VFFT_RACE_MIN, 1, 0, NULL, NULL, 1 }; /* min-of-3, alternated */ /* single-thread arms: paced (VFFT_RACE_PACE_MS) */
            const int best = vfft_race_run(&proto, arms, na, ns);
            if (best >= 0)
            {
                win = cb[best].ci;
                wns = ns[best];
            }
            if (getenv("VFFT_IL2D_LOG"))
            {
                fprintf(stderr, "[il2d] chain race %dx%d (%s): %d arm(s)", N1, N2,
                        nat ? "nat" : "scr", na);
                for (a = 0; a < na; a++)
                {
                    int q;
                    fprintf(stderr, " ");
                    for (q = 0; q < lens[cb[a].ci]; q++)
                        fprintf(stderr, "%s%d", q ? "." : "", cand[cb[a].ci][q]);
                    fprintf(stderr, "=%.0f", ns[a]);
                }
                fprintf(stderr, " -> arm %d\n", best);
            }
        }
        for (a = 0; a < na; a++)
        {
            for (s2 = 0; s2 < lens[cb[a].ci]; s2++)
            {
                free(cb[a].tf[s2]);
                free(cb[a].tb[s2]);
            }
            free(cb[a].perm);
        }
    }
    VFFT_ZS_FREE(z);
    VFFT_ZS_FREE(nscr);
    VFFT_ZS_FREE(nstage);
    *best_ns = wns;
    return win;
}



typedef struct
{
    double *sc;
    int N1;
    size_t N2;
    int nst;
    const int *R;
    int *L;
    vfft_il2p_fn *f;
    double **tf;
    int M2, bnst;
    int *bR, *bL;
    vfft_il2p_fn *bf, *bb;
    double **btf, **btb;
    double *bchf, *bkf, *bscr;
    /* NATURAL cells: the chain arm must be timed under the serving it
     * will run - the M4-lite leaf-redirected pass (scratch round trip +
     * strided scatter), never the scrambled pass (a strawman arm). */
    int nat;
    const int *perm;
    double *nscr;
} _il2d_n1arm_ctx_t;
static void _il2d_n1arm_chain(void *v)
{
    _il2d_n1arm_ctx_t *c = (_il2d_n1arm_ctx_t *)v;
    if (c->nat)
        _il2d_col_pass_nat(c->sc, c->sc, c->N1, c->N2, c->nst, c->R, c->L,
                           c->f, c->tf, 0, c->perm, c->nscr, NULL);
    else
        _il2d_col_pass(c->sc, c->sc, c->N1, c->N2, c->N2, c->nst, c->R,
                       c->L, c->f, c->tf, 0);
}
static void _il2d_n1arm_blu(void *v)
{
    _il2d_n1arm_ctx_t *c = (_il2d_n1arm_ctx_t *)v;
    _il2d_blu_cols(c->sc, c->sc, c->N1, c->N2, c->M2, c->bnst, c->bR, c->bL,
                   c->bf, c->bb, c->btf, c->btb, c->bchf, c->bkf, c->bscr);
}

/* free everything a column-axis pass owns (tables, natural, Bluestein,
 * the staged scratch); the descriptor is zero afterwards */
static void _il2d_tpc_drop(vfft_ilcol_t *c);
/* the strips' DENSE per-worker scratch (2026-09-24): T blocks of N x swcap
 * complexes, swcap = the widest strip a worker runs (the ladder's widest
 * admissible width, and the unsized whole range when it fits L2) */
static int _il2d_sw_ladder(int N1, int N2, int T, int *out, int max);
static void _il2d_nat_sscr_free(vfft_ilcol_t *c)
{
    int t;
    if (c->natsscr)
    {
        for (t = 0; t < c->nnatsscr; t++)
            VFFT_ZS_FREE(c->natsscr[t]);
        free(c->natsscr);
    }
    c->natsscr = NULL;
    c->nnatsscr = 0;
    c->natswcap = 0;
}
static int _il2d_nat_sscr_build(vfft_ilcol_t *c, int N1, int N2, int T)
{
    int lad[6], nl, k, swcap = 0, t;
    if (!c->nat || T < 2 || c->natsscr)
        return c->natsscr != NULL;
    nl = _il2d_sw_ladder(N1, N2, T, lad, 6);
    for (k = 0; k < nl; k++)
        if (lad[k] > swcap) swcap = lad[k];
    if (N2 / T >= 2 && vfft_policy_fits_l2((long)N1 * (N2 / T) * 16) && N2 / T > swcap)
        swcap = N2 / T;
    if (swcap < 2)
        return 0;
    c->natsscr = (double **)calloc((size_t)T, sizeof *c->natsscr);
    if (!c->natsscr)
        return 0;
    for (t = 0; t < T; t++)
    {
        c->natsscr[t] = (double *)VFFT_ZS_ALLOC(2 * (size_t)N1 * (size_t)swcap * sizeof(double));
        if (!c->natsscr[t])
        {
            c->nnatsscr = t;
            _il2d_nat_sscr_free(c);
            return 0;
        }
    }
    c->nnatsscr = T;
    c->natswcap = swcap;
    {   /* the probe pin, read ONCE here: eight workers calling getenv at every
         * phase start serialised on the CRT's environment lock (2026-09-25) */
        const char *dpin = getenv("VFFT_IL2D_DENSE");
        c->natdense = !(dpin && atoi(dpin) == 0);
    }
    return 1;
}
static void _il2d_col_free(vfft_ilcol_t *c)
{
    int s;
    for (s = 0; s < c->nst && s < 8; s++)
    {
        free(c->tf[s]);
        free(c->tb[s]);
    }
    free(c->natperm);
    VFFT_ZS_FREE(c->natscr);
    _il2d_nat_sscr_free(c);
    free(c->bluchf);
    free(c->bluchb);
    free(c->blukf);
    free(c->blukb);
    free(c->bluscr);
    free(c->bandscr);
    _il2d_tpc_drop(c);
    memset(c, 0, sizeof *c);
}

/* the forms axis by the column row's key (rank/axis-general twin of
 * _il2d_forms_serve, which keys rank 2 axis 0) */
static void _il2d_forms_serve_key(struct vfft_wisdom_s *W,
                                  const vfft_config_t *cfg,
                                  const vw2_ilcol_key_t *key, const char *base,
                                  int N, size_t rn,
                                  const int *Rs, int nst,
                                  vfft_il2p_fn *ff, vfft_il2p_fn *fb,
                                  char *forms, size_t fsz)
{
    const char *pin = getenv("VFFT_IL2D_FORMS");
    int s, any = 0;
    forms[0] = 0;
    for (s = 0; s < nst; s++)
    {   /* the AUTHORITY (vfft_il2p_col_forms), not a restatement of it (R4) */
        const char *nm[2];
        if (vfft_il2p_col_forms(Rs[s], nm) > 1)
            any = 1;
    }
    if (!any)
        return;
    if (pin && *pin)
    {
        if (_il2d_apply_forms(Rs, nst, pin, ff, fb))
            snprintf(forms, fsz, "%s", pin);
        else
            _vfft_warn("VFFT_IL2D_FORMS=%s does not fit chain at %dx%d - ignored",
                       pin, N, (int)rn);
        return;
    }
    if (!W || W->vw2_off_2d)
        return;
    if (!cfg->recalibrate &&
        vw2_ilcol_forms_lookup_base(&W->vw2, key, base, forms, fsz))
    {
        if (_il2d_apply_forms(Rs, nst, forms, ff, fb))
        {
            if (getenv("VFFT_IL2D_LOG"))
                fprintf(stderr, "[il2d] %s %dx%d: replay %s src=wisdom\n", base, N, (int)rn, forms);
            return;
        }
        _vfft_warn("banked %s=%s does not fit chain at %dx%d - re-racing",
                   base, forms, N, (int)rn);
        (void)_il2d_resolve(Rs, nst, ff, fb);
    }
    if (_il2d_race_forms(N, (int)rn, Rs, nst, ff, fb, forms, fsz) && forms[0])
    {
        const int banked = vw2_ilcol_forms_bank_base(&W->vw2, key, base, forms);
        if (banked)
            _vw2_persist(W, cfg);
        if (getenv("VFFT_IL2D_LOG"))
            fprintf(stderr, "[il2d] %s %dx%d: raced -> %s, %s\n", base, N, (int)rn, forms,
                    banked ? "banked" : "NOT banked yet (no chain row; the create re-banks once it lands)");
    }
}

/* ═══ THE COLUMN-AXIS PASS BUILD (2026-09-06, phase 2 of the rank-N IL
 * tier). One column axis of an interleaved c2c plan — N rows over rn
 * complex per row — under the cell's wisdom key: the chain (env > banked
 * > raced over the composition pool > greedy), its per-stage forms, the
 * column-axis Bluestein where no chain exists, the natural leaf
 * redirection when the caller asks for NATURAL, the N1-arm race (chain vs
 * Bluestein where a chain carries an odd radix), and the stage tables.
 * Lifted verbatim from the 2D create's c2c branch (2026-09-06), which now
 * calls it with a rank-2 key; a rank-N IL plan calls it per column axis
 * with a rank-3 key and the axis as the token suffix. The banked axis
 * verdicts (band width, fusion, row route, column MT + its T, the N1-arm
 * verdict) come back through the out-parameters, -1 = unraced. Returns 1;
 * 0 = REFUSED (loud), the descriptor freed. */
/* ── the tpc arm's build, race and replay (2026-09-24; struct comment at
 * c->tpc). c2c axis 0 only: the real tier's CCE columns and a rank-3 axis 1
 * run per-worker clones the one plan cannot serve. VFFT_IL2D_TPC=1|0 pins
 * (never banks). A row raced before the arm existed carries no tpc= token
 * and races it on its next create. */
static int _il2d_tpc_admits(const vw2_ilcol_key_t *key, const vfft_ilcol_t *c)
{
    return c->blu > 0 && !key->real && key->axis == 0;
}
static int _il2d_tpc_build(const vfft_config_t *cfg, vfft_ilcol_t *c, int N, size_t rn)
{
    vfft_config_t tc;
    c->N = N;   /* the builder's caller stamps it later; the pass reads it */
    memset(&tc, 0, sizeof tc);
    tc.transform = VFFT_C2C;
    tc.placement = VFFT_INPLACE;
    tc.rigor = cfg->rigor;
    tc.dims = 1;
    tc.n[0] = c->N;
    tc.howmany = 1;
    tc.order = VFFT_ORDER_NATURAL;
    tc.layout = VFFT_LAYOUT_INTERLEAVED;
    tc.nthreads = 1;
    tc.wisdom = cfg->wisdom;
    tc.wisdom_write = cfg->wisdom_write;
    c->tpcplan = (struct vfft_plan_s *)vfft_create(&tc);
    if (!c->tpcplan)
        return 0;
    c->tpcscr = (double *)VFFT_ZS_ALLOC(2 * VFFT_IL2D_TPC_PITCH(c->N) * rn * sizeof(double));
    if (!c->tpcscr)
    {
        vfft_destroy(c->tpcplan);
        c->tpcplan = NULL;
        return 0;
    }
    return 1;
}
static void _il2d_tpc_drop(vfft_ilcol_t *c)
{
    if (c->tpcplan)
        vfft_destroy(c->tpcplan);
    c->tpcplan = NULL;
    if (c->tpcscr)
        VFFT_ZS_FREE(c->tpcscr);
    c->tpcscr = NULL;
    c->tpc = 0;
}
static void _il2d_blu_drop_tables(vfft_ilcol_t *c)
{   /* the Bluestein's tables when the turned pass serves; blu = M stays */
    free(c->bluchf);
    free(c->bluchb);
    free(c->blukf);
    free(c->blukb);
    free(c->bluscr);
    c->bluchf = c->bluchb = c->blukf = c->blukb = c->bluscr = NULL;
}
static void _il2d_tpc_race(struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                           const vw2_ilcol_key_t *key, vfft_ilcol_t *c, int N, size_t rn)
{
    const char *pin = getenv("VFFT_IL2D_TPC");
    const size_t T = (size_t)N * rn;
    double *z, *zo, tb = 1e300, tt = 1e300;
    size_t i;
    int r;
    if (!_il2d_tpc_admits(key, c))
        return;
    if (pin && atoi(pin) == 0)
        return;
    if (!_il2d_tpc_build(cfg, c, N, rn))
    {
        if (getenv("VFFT_IL2D_LOG"))
            fprintf(stderr, "[il2d] tpc: the 1D in-place plan at N=%d could not be built\n", N);
        return;
    }
    if (pin)
    {
        c->tpc = 1;
        _il2d_blu_drop_tables(c);
        return;
    }
    z = (double *)VFFT_ZS_ALLOC(2 * T * sizeof(double));
    zo = (double *)VFFT_ZS_ALLOC(2 * T * sizeof(double));
    if (!z || !zo)
    {
        if (z) VFFT_ZS_FREE(z);
        if (zo) VFFT_ZS_FREE(zo);
        _il2d_tpc_drop(c);
        return;
    }
    for (i = 0; i < 2 * T; i++)
        z[i] = (double)(i % 977) / 977.0 - 0.5;
    /* min of 3, alternated, both arms through their serving functions */
    _il2d_blu_cols(z, zo, N, rn, c->blu, c->nst, c->R, c->L, c->f, c->b, c->tf, c->tb,
                   c->bluchf, c->blukf, c->bluscr);
    _il2d_tpc_cols_range(c, z, zo, rn, 0, rn, 0);
    for (r = 0; r < 3; r++)
    {
        const double t0 = _il_ab_now();
        _il2d_blu_cols(z, zo, N, rn, c->blu, c->nst, c->R, c->L, c->f, c->b, c->tf, c->tb,
                       c->bluchf, c->blukf, c->bluscr);
        const double t1 = _il_ab_now();
        _il2d_tpc_cols_range(c, z, zo, rn, 0, rn, 0);
        const double t2 = _il_ab_now();
        if (t1 - t0 < tb) tb = t1 - t0;
        if (t2 - t1 < tt) tt = t2 - t1;
    }
    VFFT_ZS_FREE(z);
    VFFT_ZS_FREE(zo);
    c->tpc = (tt < tb);
    if (getenv("VFFT_IL2D_LOG"))
        fprintf(stderr, "[il2d] tpc race at N=%d x %lu lanes: blu %.0f us, turned %.0f us -> %s\n",
                N, (unsigned long)rn, tb / 1e3, tt / 1e3, c->tpc ? "TURNED" : "blu");
    vw2_ilcol_tok_seti(&W->vw2, key, "tpc", c->tpc);
    _vw2_persist(W, cfg);
    if (c->tpc)
        _il2d_blu_drop_tables(c);
    else
        _il2d_tpc_drop(c);
}
static void _il2d_tpc_replay(struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                             const vw2_ilcol_key_t *key, vfft_ilcol_t *c, int N, size_t rn)
{
    const char *pin = getenv("VFFT_IL2D_TPC");
    int want;
    if (!_il2d_tpc_admits(key, c))
        return;
    want = pin ? atoi(pin) : vw2_ilcol_tok_geti(&W->vw2, key, "tpc", -1);
    if (want < 0)
    {   /* a row from before the arm existed: race it now */
        _il2d_tpc_race(W, cfg, key, c, N, rn);
        return;
    }
    if (want != 1)
        return;
    if (!_il2d_tpc_build(cfg, c, N, rn))
        return;
    c->tpc = 1;
    _il2d_blu_drop_tables(c);
}

static int _il2d_col_build(struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                           const vw2_ilcol_key_t *key, int N, size_t rn, int nat_req,
                           vfft_ilcol_t *c, char *forms, size_t fsz,
                           int *bwl, int *btf, int *bro, int *bcmt, int *bcmtt, int *bblu)
{
    int il2d_bwl = -1, il2d_btf = -1, il2d_bro = -1;
    int il2d_bcmt = -1, il2d_bcmtt = -1, il2d_bblu = -1;
    int il2d_tbl_done = 0;
    _il2d_blu_ctx.N2 = (int)rn;   /* the Bluestein inner's chain provider: this axis's row length */
    _il2d_blu_ctx.key = key;      /* ... and the row its verdict belongs on (D1, 2026-09-18) */
    _il2d_blu_ctx.rep_R = NULL;
    _il2d_blu_ctx.rep_nst = 0;
    _il2d_blu_ctx.commit = 0;
    int chain_ok = 0;
    {
        /* chain precedence: env > banked lay=il verdict > RACE
         * the full composition pool (multi-stage cells only;
         * component-pinned: the race times the column pass, the
         * only thing the axis changes) > greedy. */
        chain_ok = _il2d_env_chain(N, c->R, c->f, c->b, &c->nst);   /* the pin */
        if (!chain_ok && !cfg->recalibrate &&
                 /* the caller's explicit override reaches THIS tier too
                  * (2026-09-16): include/vfft.h promises recalibrate=1
                  * "re-measures and overwrites the cell even on" a hit, and
                  * this lookup used to ignore it — so a 2D c2c (and 3D, via
                  * fftnd_il.h) cell replayed its banked chain/wl/ro/cmt no
                  * matter what the caller asked. The real tier's twin
                  * (fft2d_create.h) and five sibling lookups in this file
                  * were already guarded; this one was the omission. */
                 vw2_ilcol_chain_lookup(&W->vw2, key, c->R,
                                        &c->nst, &il2d_bwl,
                                        &il2d_btf, &il2d_bro,
                                        &il2d_bcmt,
                                        &il2d_bcmtt, &il2d_bblu) &&
                 _il2d_chain_prod(c->R, c->nst) ==
                     (il2d_bblu > 0 ? il2d_bblu : N) &&
                 _il2d_resolve(c->R, c->nst, c->f, c->b))
            chain_ok = 1;
        /* a replayed Bluestein row (il2d_bblu > 0) rebuilds its inner in the
         * N-arm block below (E1.7, 2026-09-02) -- at every rank. A rank >= 3
         * copy of that rebuild lived here for one day (2026-09-17) on the
         * belief the builder had none; it was redundant and is gone. */
        if (!chain_ok)
        {
            int cand[VFFT_IL2D_MAXCAND][8], lens[VFFT_IL2D_MAXCAND];
            int cur[8], ncand = 0, dropped = 0;
            _il2d_enum_rec(N, 0, cur, cand, lens, &ncand,
                           &dropped);
            /* (a capped pool warns from inside the enumerator, 2026-09-17) */
            /* ncand >= 1, not > 1 (2026-09-17): a cell with exactly ONE
             * legal chain used to skip the race and drop to the greedy --
             * unmeasured, UNBANKED, re-derived on every create, with no row
             * for its forms and widths to hang off. One arm is still a
             * race: it is timed, and it banks. */
            if (ncand >= 1)
            {
                double bns = 0;
                int win = _il2d_race_chains(N, (int)rn, ncand, cand,
                                            lens, &bns, nat_req);
                /* nat_req, NOT key->ord (2026-09-17): key->ord is the ROW
                 * LABEL, nat_req is which PASS this build will run. They are
                 * equal at the 2D tier and at the 3D tier's axis 1; they are
                 * NOT equal at the 3D tier's axis 0, which is the scrambled
                 * class for both order classes by design (fftnd_il.h). Racing
                 * on the label there timed a pass the tier never runs and
                 * excluded every chain with no natural leaf from the pool. */
                if (win >= 0 &&
                    _il2d_resolve(cand[win], lens[win], c->f,
                                  c->b))
                {
                    memcpy(c->R, cand[win],
                           sizeof cand[win]);
                    c->nst = lens[win];
                    chain_ok = 1;
                    vw2_ilcol_chain_bank(&W->vw2, key,
                                         c->R, c->nst,
                                         -1, -1, -1, -1, -1, -1,
                                         bns);
                    _vw2_persist(W, cfg);
                }
            }
            /* no greedy fallback (2026-09-17): a race with no buildable arm
             * has no chain, and !chain_ok below is Bluestein or a refusal */
        }
    }
    if (chain_ok && il2d_bblu <= 0)
    {   /* E1.11 per-stage kernel forms (2026-09-02); a banked
         * Bluestein cell's forms live on its (M, (int)rn) row */
        _il2d_forms_serve_key(W, cfg, key, "forms", N, rn, c->R, c->nst,
                          c->f, c->b, forms, fsz);
    }
    if (!chain_ok)
    {
        /* ODD/PRIME N: the COLUMN-AXIS BLUESTEIN (struct
         * comment at c->blu; _il2d_blu_build). Reached only
         * when no chain exists — with the odd t2c/n1c kinds
         * emitted, that now means prime / unexpressible N.
         * n1 comes out NATURAL by construction, so ALL order
         * spellings are served (M4-lite closed the old
         * DEFAULT-only gate 2026-08-27). */
        _il2d_blu_ctx.commit = 1;   /* no chain exists: this build IS the cell's plan */
        c->blu = _il2d_blu_build(N, rn, c->R,
                                   c->L, c->f, c->b,
                                   c->tf, c->tb, &c->nst,
                                   &c->bluchf, &c->bluchb,
                                   &c->blukf, &c->blukb,
                                   &c->bluscr);
        _il2d_blu_ctx.commit = 0;
        if (c->blu)
        {
            chain_ok = 1;
            _il2d_tpc_race(W, cfg, key, c, N, rn);   /* the turned pass vs this Bluestein (2026-09-24) */
        }
        /* (the cell's own row is banked by the provider, at every rank and
         * before it serves the inner's forms -- vw2_ilcol_forms_bank_base is
         * a field update and needs the row to exist. The rank >= 3 bank that
         * stood here, added 2026-09-17 for the un-bankable 3D Bluestein axis,
         * moved there whole on 2026-09-18 with D1.) */
    }
    if (chain_ok && !c->blu && il2d_bblu <= 0 && c->nst > 1 &&
        nat_req)
    {
        /* (il2d_bblu > 0 = a banked Bluestein cell: c->R is the
         * length-M chain, n1 natural by construction, the perm
         * would divide by zero - pre-existing, fixed 2026-09-02) */
        /* M4-lite (struct comment at c->nat): natural n1 via
         * the LEAF REDIRECTION - driver-only, any chain. Built
         * BEFORE the N-arm race below so the chain arm is
         * timed under the serving it will actually run. The
         * perm builder refuses on any convention mismatch. */
        c->natperm = _il2d_nat_perm(c->R, c->nst, N);
        if (c->natperm)
            c->natscr = (double *)VFFT_ZS_ALLOC(   /* 64-B aligned: the strip workers' pieces of a row must not share a line (2026-09-24) */
            
                2 * (size_t)N * (int)rn * sizeof(double));
        if (!c->natperm || !c->natscr)
        {
            free(c->natperm);
            c->natperm = NULL;
            _vfft_warn("vfft_create: IL 2D c2c %dx%d "
                       "order=NATURAL - the natural leaf "
                       "permutation could not be built for "
                       "this chain; unsupported",
                       N, (int)rn);
            { _il2d_col_free(c); return 0; }
        }
        c->nat = 1;
    }
    if (chain_ok && !c->blu && !getenv("VFFT_IL2D_CHAIN"))
    {
        /* THE RACED CHAIN ARM (owner directive): for a chain
         * that carries an ODD radix (the newly emitted kinds),
         * race it against the Bluestein column route. DEFAULT
         * order: the two serve different n1 orders (chain =
         * scrambled comb, blu = natural), both self-consistent.
         * NATURAL order (2026-08-28): BOTH arms are natural -
         * the chain via the leaf redirection, blu by
         * construction - so the race is the only lawful pick;
         * the chain arm runs the natural pass. Pick is pure
         * speed either way. Env VFFT_IL2D_BLU=1 pins blu,
         * =0 pins the chain (env never banks); unset = race
         * min-of-3 alternated on scratch through the SERVING
         * functions. pow2 chains never race (blu is pointless
         * there). Verdict plan-local (the wisdom banking of a
         * blu marker rides the layout-audit wave). */
        int hasodd = 0, s3;
        const char *be = getenv("VFFT_IL2D_BLU");
        for (s3 = 0; s3 < c->nst; s3++)
            if (c->R[s3] & 1)
                hasodd = 1;
        /* the chain arm times the SERVING column pass, so the
         * N tables must exist BEFORE the race (they are
         * otherwise built at the row-child block below — timing
         * with empty tabs was a NULL-load crash, caught by the
         * cell sweep 2026-08-27). il2d_tbl_done stops the later
         * shared build from double-building the winner's. */
        /* REPLAY the banked N-arm verdict (E1.7, 2026-09-02):
         * blu > 0 = Bluestein won (the row's chain IS the M
         * chain, already resolved above — build the Bluestein
         * tables and adopt, no timing); blu == 0 = the chain won
         * (nothing to do). Only an unraced cell (-1) or an env
         * pin runs the race below. */
        /* blu > 0 REGARDLESS of hasodd (2026-09-02): a banked Bluestein
         * row carries the pow2 M chain, which has no odd factor — the
         * old guard skipped the adoption for a PRIME N and the create
         * ran the M chain as the N chain (wrong output, caught by the
         * naive-DFT probe at 127x100 the day prime rows first banked) */
        if (il2d_bblu > 0 && !be && !cfg->recalibrate)
        {
            int bR[8], bL[8], bnst = 0, M2;
            vfft_il2p_fn bf[8], bb[8];
            double *btf[8], *btb[8];
            double *bchf, *bchb, *bkf, *bkb, *bscr;
            memset(btf, 0, sizeof btf);
            memset(btb, 0, sizeof btb);
            /* the replayed row's chain IS the M chain (the lookup above
             * checked prod == blu): hand it to the provider rather than have
             * it re-read a row of its own (D1, 2026-09-18) */
            _il2d_blu_ctx.rep_R = c->R;
            _il2d_blu_ctx.rep_nst = c->nst;
            M2 = _il2d_blu_build(N, rn, bR, bL, bf, bb,
                                 btf, btb, &bnst, &bchf, &bchb,
                                 &bkf, &bkb, &bscr);
            _il2d_blu_ctx.rep_R = NULL;
            _il2d_blu_ctx.rep_nst = 0;
            if (M2 == il2d_bblu)
            {
                memcpy(c->R, bR, sizeof bR);
                memcpy(c->L, bL, sizeof bL);
                memcpy(c->f, bf, sizeof bf);
                memcpy(c->b, bb, sizeof bb);
                memcpy(c->tf, btf, sizeof btf);
                memcpy(c->tb, btb, sizeof btb);
                c->nst = bnst;
                c->blu = M2;
                c->bluchf = bchf;
                c->bluchb = bchb;
                c->blukf = bkf;
                c->blukb = bkb;
                c->bluscr = bscr;
                il2d_tbl_done = 1;
                _il2d_tpc_replay(W, cfg, key, c, N, rn);   /* tpc= on the row, or race it (2026-09-24) */
                if (c->nat)
                {
                    free(c->natperm);
                    VFFT_ZS_FREE(c->natscr);
                    c->natperm = NULL;
                    c->natscr = NULL;
                    c->nat = 0;
                }
                if (getenv("VFFT_IL2D_LOG"))
                    fprintf(stderr, "[il2d] N-arm %dx%d (c2c): "
                                    "replay BLUESTEIN M=%d src=wisdom\n",
                            N, (int)rn, M2);
                hasodd = 0;            /* verdict served */
            }
            else
            {
                for (s3 = 0; s3 < bnst; s3++)
                {
                    free(btf[s3]);
                    free(btb[s3]);
                }
                free(bchf); free(bchb); free(bkf); free(bkb); free(bscr);
            }
        }
        else if (hasodd && il2d_bblu == 0 && !be && !cfg->recalibrate)
        {
            if (getenv("VFFT_IL2D_LOG"))
                fprintf(stderr, "[il2d] N-arm %dx%d (c2c): replay "
                                "chain src=wisdom\n", N, (int)rn);
            hasodd = 0;                /* the chain won: no race */
        }
        if (hasodd && (!be || atoi(be) == 1) &&
            !_il2d_build_tables(N, c->nst, c->R, c->L,
                                c->tf, c->tb))
        {
            il2d_tbl_done = 1;
            int bR[8], bL[8], bnst = 0, M2;
            vfft_il2p_fn bf[8], bb[8];
            double *btf[8], *btb[8];
            double *bchf, *bchb, *bkf, *bkb, *bscr;
            memset(btf, 0, sizeof btf);
            memset(btb, 0, sizeof btb);
            M2 = _il2d_blu_build(N, rn, bR, bL, bf, bb,
                                 btf, btb, &bnst, &bchf, &bchb,
                                 &bkf, &bkb, &bscr);
            if (M2)
            {
                double *sc = (double *)malloc(
                    2 * (size_t)N * (int)rn * sizeof(double));
                double tc = 1e300, tbu = 1e300;
                int rr, use_blu = (be != NULL); /* env pin */
                size_t i3;
                if (sc && !use_blu)
                {
                    for (i3 = 0; i3 < 2 * (size_t)N * (int)rn; i3++)
                        sc[i3] = 1.0 + 1e-6 * (double)(i3 & 511);
                    {
                        _il2d_n1arm_ctx_t rc = {
                            sc, N, rn, c->nst, c->R,
                            c->L, c->f, c->tf, M2, bnst, bR,
                            bL, bf, bb, btf, btb, bchf, bkf, bscr,
                            c->nat, c->natperm, c->natscr };
                        const vfft_race_arm_t arms[2] = {
                            { "chain", _il2d_n1arm_chain, &rc },
                            { "bluestein", _il2d_n1arm_blu, &rc } };
                        const vfft_race_proto_t proto = { 3, 1, VFFT_RACE_MIN, 0, 0, NULL, NULL, 1 }; /* min-of-3, A then B */ /* single-thread arms: paced (VFFT_RACE_PACE_MS) */
                        double ns[2];
                        (void)rr;
                        vfft_race_run(&proto, arms, 2, ns);
                        tc = ns[0];
                        tbu = ns[1];
                    }
                }
                free(sc);
                if (!use_blu)
                    use_blu = (tbu < tc);
                if (getenv("VFFT_IL2D_LOG"))
                    fprintf(stderr, "[il2d] N-arm race %dx%d "
                                    "(c2c %s): chain=%.0f blu=%.0f "
                                    "-> %s\n",
                            N, (int)rn, c->nat ? "nat" : "scr",
                            tc, tbu,
                            use_blu ? "BLUESTEIN" : "chain");
                if (!be)
                {   /* bank the verdict with the chain that SERVES */
                    vw2_ilcol_chain_bank(&W->vw2, key,
                                         use_blu ? bR : c->R,
                                         use_blu ? bnst : c->nst,
                                         -1, -1, -1, -1, -1,
                                         use_blu ? M2 : 0, 0.0);
                    _vw2_persist(W, cfg);
                }
                if (use_blu)
                {
                    for (s3 = 0; s3 < c->nst; s3++)
                    {
                        free(c->tf[s3]);
                        free(c->tb[s3]);
                    }
                    memcpy(c->R, bR, sizeof bR);
                    memcpy(c->L, bL, sizeof bL);
                    memcpy(c->f, bf, sizeof bf);
                    memcpy(c->b, bb, sizeof bb);
                    memcpy(c->tf, btf, sizeof btf);
                    memcpy(c->tb, btb, sizeof btb);
                    c->nst = bnst;
                    c->blu = M2;
                    c->bluchf = bchf;
                    c->bluchb = bchb;
                    c->blukf = bkf;
                    c->blukb = bkb;
                    c->bluscr = bscr;
                    if (c->nat)
                    { /* blu is natural by construction */
                        free(c->natperm);
                        VFFT_ZS_FREE(c->natscr);
                        c->natperm = NULL;
                        c->natscr = NULL;
                        c->nat = 0;
                    }
                }
                else
                {
                    for (s3 = 0; s3 < bnst; s3++)
                    {
                        free(btf[s3]);
                        free(btb[s3]);
                    }
                    free(bchf); free(bchb);
                    free(bkf); free(bkb); free(bscr);
                }
            }
        }
        else if (hasodd && be && atoi(be) == 0)
            ; /* env pins the chain: nothing to do */
    }
    if (!chain_ok)
    {
        /* OWNER LAW: split is NOT a fallback of IL — no convert
         * wrapper. An inexpressible N refuses loudly. */
        _vfft_warn("vfft_create: IL 2D c2c %dx%d — N has no "
                   "native column chain (radices 4..64, no "
                   "leftover factor)%s",
                   N, (int)rn,
                   " and the Bluestein column route could "
                   "not be built");
        { _il2d_col_free(c); return 0; }
    }

    if (!c->blu && !il2d_tbl_done &&
        _il2d_build_tables(N, c->nst, c->R, c->L, c->tf, c->tb))
    {
        _vfft_warn("vfft_create: IL column pass %dx%d — stage tables failed; unsupported",
                   N, (int)rn);
        _il2d_col_free(c);
        return 0;
    }
    c->N = N;
    c->rn = rn;
    *bwl = il2d_bwl; *btf = il2d_btf; *bro = il2d_bro;
    *bcmt = il2d_bcmt; *bcmtt = il2d_bcmtt; *bblu = il2d_bblu;
    return 1;
}

/* the §10a axis race: time the FULL execute (column chain + rows) over
 * the wl candidates x the row routes (x the two-pass rows' tile), set the
 * winner on the plan, bank chain+wl+tf+ro(+rbk) as one lay=il verdict.
 * Falsifier-grounded: wl wins +15-21% at some cells and LOSES at others;
 * the batched rows win at N2 <= 16 (mono) and 32/64 (two-pass) and the
 * per-row child elsewhere -- per-cell only, never defaults. The raced
 * out-of-place child route (ro=1, the copy-back) was DELETED 2026-09-23:
 * beside the batched routes it never won, and the in-place K=1 tier serves
 * every N2 (its last candidate is the prime engine): no forced path either. */
/* one (row route, wl, tile) configuration of the axis race, as a race ARM:
 * sets the plan's band axes, then one full forward execute through the
 * serving path (the natural banded walk under NATURAL, the scrambled one
 * otherwise). */
typedef struct
{
    struct vfft_plan_s *h;
    double *z;
    double *zo;               /* the cell's own placement: z for in place, a second plane out of place (2026-09-23) */
    int wl, cut, ro;
    int wc;                   /* the unbanded walk's column tile (0 = the whole plane) */
    int kb;                   /* ro=3: the two-pass rows' tile, KB of chunk scratch */
    int csk;                  /* the SKEWED column pass (2026-09-23) */
    int mt;                   /* T > 1: the arm runs its route's threaded walk (2026-09-24) */
    char name[24];
} _il2d_axis_arm_t;
static void _il2d_arm_axis(void *v)
{
    _il2d_axis_arm_t *c = (_il2d_axis_arm_t *)v;
    struct vfft_plan_s *h = c->h;
    h->il2d_col.wl = c->wl;
    h->il2d_col.cut = c->cut;
    h->il2d_col.tfuse = (c->wl > 0);
    h->il2d_col.wc = c->wc;
    h->il2d_rowb = (c->ro == 2);   /* the batched rows (ro=2, 2026-09-23) */
    h->il2d_rowb2 = (c->ro == 3);  /* the batched two-pass rows (ro=3) */
    h->il2d_turn = (c->ro == 4);   /* the TURN route: the whole plane through the 1D engine */
    h->il2d_csk = c->csk;          /* the SKEWED column pass */
    if (c->ro == 3)
        h->il2d_rowb2_ch = (int)_il2d_rb2_rows(c->kb, (size_t)h->N2);
    if (c->mt)
    {   /* the T-aware race (2026-09-24): every arm through its route's threaded
         * walk -- the chain under the DENSE STRIPS (the form that dominated the
         * block arm at every cell measured: an arm raced under the block shape
         * picked tiled rows the strips then ran worse with, 128x256 0.58,
         * 4096x256 0.77); the threading race refines the winner's shape
         * (serial, block, the strips ladder) afterwards */
        h->il2d_col.colmt = 1;
        h->il2d_col.natarm = 1;
        h->il2d_col.msw = 0;
        h->il2d_col.natst = 0;
    }
    else
        h->il2d_col.colmt = 0;   /* the serial form (every arm at T = 1; the "+s" twins at T > 1) */
    vfft_execute((vfft_plan)h, VFFT_FORWARD, c->z, NULL, c->zo, NULL);
}

/* ORDER CELLS (2026-09-07): every bank below keys the order by the CELL's
 * order (cfg->order), never by the pass's natural flag — a natural cell
 * whose chain is natural by construction (a single stage, a Bluestein
 * axis) has il2d_col.nat == 0, and keying by that flag banked its axis and
 * MT verdicts on the SCRAMBLED row: the natural cell re-raced on every
 * create and the scrambled cell served a natural measurement (found by the
 * 3D natural class's child cells; the owner's law: order cells are never
 * compared, never mixed). */
static void _il2d_axis_race(struct vfft_plan_s *h, struct vfft_wisdom_s *W,
                            const vfft_config_t *cfg, int N1, int N2)
{
    const size_t T = (size_t)N1 * N2;
    double *z = (double *)VFFT_ZS_ALLOC(2 * T * sizeof(double));   /* aligned like every plane the door serves (2026-09-23) */
    /* THE RACE RUNS THE CELL'S OWN PLACEMENT (2026-09-23): an out-of-place cell
     * races x -> y like the door serves it; racing z -> z ranked a
     * placement-sensitive route (the skewed column pass) the other way round */
    double *zo = (h->placement == VFFT_OUTOFPLACE) ? (double *)VFFT_ZS_ALLOC(2 * T * sizeof(double)) : z;   /* aligned like every plane the door serves (2026-09-23) */
    int wlc[14], nwl = 1, wi, ro, bwl = 0, bro = 0, bwc = 0, bkb = 8, bcsk = 0, csk;
    int swl[6], nsw = 0;
    double best = 1e300;
    size_t i;
    int reps = (int)(1e6 / (double)(T + 1));
    /* THE T-AWARE RACE (2026-09-24): at T > 1 with a live pool every arm runs
     * its route's threaded walk, so a threaded chain competes with the
     * threaded turn and skewed pass (at T=8 the serial race sent 81 of 159
     * pow2 cells down a route with no threaded walk). The verdict banks
     * beside the serial one (axt= and the t-suffixed tokens), served at
     * that T only. On natural cells the threaded chain walk ignores the band
     * width and the sub-strip tile, so those arms are pruned to wl = 0. */
    const int mt = (h->nthreads > 1 && stride_pool_workers_for(h->nthreads) >= 2);
    const int prune = mt && h->il2d_col.nat;
    if (reps < 2) reps = 2;
    if (!z || !zo)
    {
        VFFT_ZS_FREE(z);
        if (zo != z) VFFT_ZS_FREE(zo);
        return;
    }
    for (i = 0; i < 2 * T; i++)
        z[i] = 1.0 + 1e-6 * (double)(i & 1023);
    /* wl candidates: 0 (unbanded) + legal widths */
    wlc[0] = 0;
    {
        int p, s2;
        for (p = 0; p < VFFT_IL2D_WL_LADDER_N && nwl < 14; p++)
        {
            const int w = VFFT_IL2D_WL_LADDER[p];
            int cut = -1;
            cut = vfft_policy_il2d_wl_cut(N1, h->il2d_col.nst, h->il2d_col.L, w);
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
         * (vfft_policy_fits_l2(w * N2 * 16), the hardware-derived
         * fence — never a platform-baked constant), and the RACE still
         * decides: these are candidates, not defaults. */
        for (s2 = 1; s2 < h->il2d_col.nst && nwl < 14; s2++)
        {
            const int w = h->il2d_col.L[s2];
            int dup = 0, p2;
            if (!vfft_policy_il2d_band_ok(N1, h->il2d_col.nst, h->il2d_col.L, w))
                continue;
            if (!vfft_policy_fits_l2((long)w * N2 * 16))
                continue;
            for (p2 = 0; p2 < nwl; p2++)
                if (wlc[p2] == w)
                    dup = 1;
            if (!dup)
                wlc[nwl++] = w;
        }
    }
    {   /* VFFT_IL2D_WL=w pins the band width for a PROBE (0 = unbanded):
         * the race keeps that one candidate; a width the ladder does not
         * hold is ignored. Beside VFFT_IL2D_CHAIN, on a scratch store. */
        const char *e = getenv("VFFT_IL2D_WL");
        if (e)
        {
            const int w = atoi(e);
            int p2, hit = 0;
            for (p2 = 0; p2 < nwl; p2++) if (wlc[p2] == w) hit = 1;
            if (hit) { wlc[0] = w; nwl = 1; }
        }
    }
    /* the unbanded walk's tile (il2d_large_plane_design.md, 2026-09-15):
     * every ladder width an arm beside wl = 0 */
    nsw = _il2d_sw_ladder(N1, N2, 1, swl, 6);
    /* every (row route, wl, tile) configuration is an ARM of ONE race, rounds
     * alternating direction — the house protocol (2026-09-05). The
     * sequential per-configuration loop it replaces timed each arm in
     * its own block, so drift favoured whichever ran in the cooler
     * moment: at 405x405 natural it banked wl=9 while a same-run
     * alternated A/B put wl=81 6.5% ahead. Same samples per arm as
     * before (min of 2 x reps), same candidates. */
    {
        _il2d_axis_arm_t ac[VFFT_RACE_MAX_ARMS];
        vfft_race_arm_t arms[VFFT_RACE_MAX_ARMS];
        double ns[VFFT_RACE_MAX_ARMS];
        int na = 0, a;
        for (csk = 0; csk <= (h->il2d_csk_scr ? 1 : 0); csk++)
        for (ro = 0; ro <= 4; ro++)
            for (wi = 0; wi < nwl && na < VFFT_RACE_MAX_ARMS; wi++)
            {
                int s2, cut = 0;
                const int w = wlc[wi];
                (void)s2;
                /* the SKEWED column pass (2026-09-23): unbanded, untiled, crossed
                 * with the row routes 0/2/3 (the per-row one through the OOP
                 * K=1 plan at N2, when it built); never with the turn route */
                if (csk && (w != 0 || ro == 4 || (ro == 0 && !h->il2d_csk_row)))
                    continue;
                /* the row routes: 0 = the in-place child; 2 = the BATCHED rows,
                 * when the radix has the n1ccs pair; 3 = the batched TWO-PASS
                 * rows, when the child's stage kernels have row-loop twins
                 * (2026-09-23). 1, the raced OOP child, is deleted. 4 = the
                 * TURN route (the whole plane through the 1D engine): one arm,
                 * no band, no tile, when its N1 plan built. */
                if (ro == 1 || (ro == 2 && !h->il2d_rowb_f) || (ro == 3 && !h->il2d_rowb2_leaf_f) ||
                    (ro == 4 && (!h->il2d_turn_plan || w != 0)))
                    continue;
                if (prune && w != 0)
                    continue;   /* the threaded natural walk has no bands */
                if (w > 0)
                {   /* an admitted width's cut; 0 kept where none (R3) */
                    const int r = vfft_policy_il2d_cut_of(h->il2d_col.nst, h->il2d_col.L, w);
                    if (r >= 0) cut = r;
                }
                {
                    int sub, ki, fm;
                    for (sub = 0; sub <= (w == 0 && ro != 4 && !csk && !prune ? nsw : 0) && na < VFFT_RACE_MAX_ARMS; sub++)
                        for (ki = 0; ki < (ro == 3 ? 4 : 1) && na < VFFT_RACE_MAX_ARMS; ki++)
                        for (fm = 0; fm <= (mt ? 1 : 0) && na < VFFT_RACE_MAX_ARMS; fm++)
                        {   /* fm = 1: the arm's SERIAL form beside its threaded one at T > 1
                             * (2026-09-25): a plane the threading race will keep serial
                             * must pick its route by the serial walks */
                            const int kb = ro == 3 ? VFFT_IL2D_RB2_KB_LADDER[ki] : 0;
                            if (ro == 3 && ki > 0 &&
                                _il2d_rb2_rows(kb, (size_t)N2) == _il2d_rb2_rows(VFFT_IL2D_RB2_KB_LADDER[ki - 1], (size_t)N2))
                                continue;   /* the same tile in rows: one arm */
                            ac[na].h = h;
                            ac[na].z = z;
                            ac[na].zo = zo;
                            ac[na].wl = w;
                            ac[na].cut = cut;
                            ac[na].ro = ro;
                            ac[na].kb = kb;
                            ac[na].csk = csk;
                            ac[na].mt = mt && !fm;
                            ac[na].wc = sub ? swl[sub - 1] : 0;
                            if (sub)
                                snprintf(ac[na].name, sizeof ac[na].name, "sw%d%s%s", ac[na].wc,
                                         ro == 2 ? "+rowb" : ro == 4 ? "+turn" : "", fm ? "+s" : "");
                            else
                                snprintf(ac[na].name, sizeof ac[na].name, "wl%d%s%s", w,
                                         ro == 2 ? "+rowb" : ro == 4 ? "+turn" : "", fm ? "+s" : "");
                            if (ro == 3)
                            {
                                const size_t used = strlen(ac[na].name);
                                snprintf(ac[na].name + used, sizeof ac[na].name - used, "+rb2k%d", kb);
                            }
                            if (csk)
                            {
                                const size_t used = strlen(ac[na].name);
                                snprintf(ac[na].name + used, sizeof ac[na].name - used, "+csk");
                            }
                            arms[na].name = ac[na].name;
                            arms[na].run = _il2d_arm_axis;
                            arms[na].ctx = &ac[na];
                            na++;
                        }
                }
            }
        {
            const vfft_race_proto_t proto = { 2, reps, VFFT_RACE_MIN, 1, mt ? 2 : 0, NULL, NULL, 0 };   /* threaded arms: two untimed passes first (2026-09-25) */
            vfft_race_run(&proto, arms, na, ns);
        }
        for (a = 0; a < na; a++)
            if (ns[a] < best)
            {
                best = ns[a];
                bwl = ac[a].wl;
                bro = ac[a].ro;
                bwc = ac[a].wc;
                bkb = ac[a].ro == 3 ? ac[a].kb : 8;
                bcsk = ac[a].csk;
            }
        if (getenv("VFFT_IL2D_LOG"))
        {
            fprintf(stderr, "[il2d] axis race %dx%d (%s, T=%d):", N1, N2,
                    h->il2d_col.nat ? "nat" : "scr", mt ? h->nthreads : 1);
            for (a = 0; a < na; a++)
                fprintf(stderr, " %s=%.0f", ac[a].name, ns[a]);
            fprintf(stderr, " -> wl=%d sw=%d ro=%d\n", bwl, bwc, bro);
        }
    }
    /* set the winner */
    {
        int s2, cut = 0;
        (void)s2;
        if (bwl > 0)
        {   /* the winner's cut; 0 kept where none (R3) */
            const int r = vfft_policy_il2d_cut_of(h->il2d_col.nst, h->il2d_col.L, bwl);
            if (r >= 0) cut = r;
        }
        h->il2d_col.wl = bwl;
        h->il2d_col.cut = cut;
        h->il2d_col.tfuse = (bwl > 0);
        h->il2d_col.wc = bwc;
        h->il2d_rowb = (bro == 2);
        h->il2d_rowb2 = (bro == 3);
        h->il2d_turn = (bro == 4);
        h->il2d_csk = bcsk;
        if (bro == 3)
            h->il2d_rowb2_ch = (int)_il2d_rb2_rows(bkb, (size_t)N2);
        if (mt)
        {   /* the threading race decides the winner's threaded shape next */
            h->il2d_col.colmt = 0;
            h->il2d_col.natarm = 0;
            h->il2d_col.msw = 0;
            h->il2d_col.natst = 1;
        }
    }
    if (mt)
    {   /* the T verdict, beside the serial one: the t-suffixed tokens and the
         * T raced at; the serial tokens are another race's (2026-09-24) */
        const int ord = vfft_policy_ord_rankn(cfg);
        if (vw2_2d_il_tok_seti(&W->vw2, N1, N2, ord, "axt", h->nthreads) != 0)
        {   /* no row yet (a recalibrating create races the axis before the chain
             * row is written): a chain-only row (negative axes omitted), then
             * the tokens (2026-09-25) */
            vw2_2d_il_chain_bank(&W->vw2, N1, N2, h->il2d_col.R, h->il2d_col.nst,
                                 -1, -1, -1, -1, -1, (N1 & (N1 - 1)) ? h->il2d_col.blu : -1, best, ord);
            if (vw2_2d_il_tok_seti(&W->vw2, N1, N2, ord, "axt", h->nthreads) != 0)
                _vfft_warn("il2d axis race at T=%d: no row to bank the T verdict at %dx%d", h->nthreads, N1, N2);
        }
        vw2_2d_il_tok_seti(&W->vw2, N1, N2, ord, "rot", _il2d_ro_of(h));
        vw2_2d_il_tok_seti(&W->vw2, N1, N2, ord, "wlt", h->il2d_col.wl);
        vw2_2d_il_tok_seti(&W->vw2, N1, N2, ord, "swt", bwc);
        vw2_2d_il_tok_seti(&W->vw2, N1, N2, ord, "rbkt", bkb);
        vw2_2d_il_tok_seti(&W->vw2, N1, N2, ord, "turnt", h->il2d_turn);
        vw2_2d_il_tok_seti(&W->vw2, N1, N2, ord, "cskt", h->il2d_csk);
        vw2_2d_il_tok_seti(&W->vw2, N1, N2, ord, "axns", (int)best);
    }
    else
    {
    vw2_2d_il_chain_bank(&W->vw2, N1, N2, h->il2d_col.R, h->il2d_col.nst,
                         h->il2d_col.wl, h->il2d_col.tfuse, _il2d_ro_of(h),
                         -1, -1, (N1 & (N1 - 1)) ? h->il2d_col.blu : -1, best, vfft_policy_ord_rankn(cfg));
    vw2_2d_il_tok_seti(&W->vw2, N1, N2, vfft_policy_ord_rankn(cfg), "sw", bwc);
    vw2_2d_il_tok_seti(&W->vw2, N1, N2, vfft_policy_ord_rankn(cfg), "rbk", bkb);   /* the two-pass rows' tile */
    vw2_2d_il_tok_seti(&W->vw2, N1, N2, vfft_policy_ord_rankn(cfg), "turn", h->il2d_turn);   /* the turn route */
    vw2_2d_il_tok_seti(&W->vw2, N1, N2, vfft_policy_ord_rankn(cfg), "csk", h->il2d_csk);     /* the skewed column pass */
    }
    _vw2_persist(W, cfg);
    if (zo != z) VFFT_ZS_FREE(zo);
    VFFT_ZS_FREE(z);
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
static int _il2d_clone_set(const struct vfft_plan_s *prim, const vfft_config_t *cfg,
                           int N, int placement, int T,
                           struct vfft_plan_s ***out, int *out_n, const char *what)
{
    const int n = (T > 64 ? 64 : T) - 1;
    vfft_config_t rc;
    struct vfft_plan_s **arr;
    int t;
    if (n <= 0 || *out || !prim)
        return 0;
    memset(&rc, 0, sizeof rc);
    rc.transform = VFFT_C2C;
    rc.placement = placement;
    rc.rigor = cfg->rigor;
    rc.dims = 1;
    rc.n[0] = N;
    rc.howmany = 1;
    rc.order = VFFT_ORDER_NATURAL;
    rc.layout = VFFT_LAYOUT_INTERLEAVED;
    rc.nthreads = 1;
    rc.wisdom = cfg->wisdom;
    rc.wisdom_write = 0; /* clones read warm wisdom, never bank */
    arr = (struct vfft_plan_s **)calloc((size_t)n, sizeof *arr);
    if (!arr)
        return 0;
    for (t = 0; t < n; t++)
    {
        struct vfft_plan_s *c = (struct vfft_plan_s *)vfft_create(&rc);
        arr[t] = c;
        if (!c || !_tc_clone_equiv(prim, c) || c->tcb || c->tcbw)
        {
            int u;
            _vfft_warn("il2d c2c MT: %s clone %d %s at N=%d — MT "
                       "declines for this plan",
                       what, t, c ? "route-mismatched" : "failed to create", N);
            for (u = 0; u <= t; u++)
                if (arr[u])
                    vfft_destroy(arr[u]);
            free(arr);
            return 0;
        }
    }
    *out = arr;
    *out_n = n;
    return 1;
}
/* the set the banked route needs (2026-09-24): the turn's N1 plan, the skewed
 * pass's OOP row plan on route 0 (its batched routes run stateless kernels on
 * their own slots), else the chain walk's in-place row child */
static void _il2d_c2c_build_clones(struct vfft_plan_s *h,
                                   const vfft_config_t *cfg, int T)
{
    if (h->il2d_turn)
    {
        _il2d_clone_set(h->il2d_turn_plan, cfg, h->N, VFFT_INPLACE, T,
                        &h->il2d_turnw, &h->il2d_turnw_n, "turn");
        return;
    }
    if (h->il2d_csk)
    {
        if (!h->il2d_rowb && !h->il2d_rowb2)
            _il2d_clone_set(h->il2d_csk_row, cfg, h->N2, VFFT_OUTOFPLACE, T,
                            &h->il2d_cskw, &h->il2d_cskw_n, "csk row");
        return;
    }
    _il2d_clone_set(h->il2d_row, cfg, h->N2, VFFT_INPLACE, T,
                    &h->il2d_roww, &h->il2d_roww_n, "row");
}
/* the T-aware axis race needs every route's set BEFORE it runs (2026-09-24):
 * an arm whose set is missing runs serial -- what execute would do, but not
 * the comparison the race is for. After the race the sets the banked route
 * does not run are dropped. */
static void _il2d_c2c_build_clone_sets_all(struct vfft_plan_s *h, const vfft_config_t *cfg, int T)
{
    _il2d_clone_set(h->il2d_row, cfg, h->N2, VFFT_INPLACE, T, &h->il2d_roww, &h->il2d_roww_n, "row");
    if (h->il2d_csk_row)
        _il2d_clone_set(h->il2d_csk_row, cfg, h->N2, VFFT_OUTOFPLACE, T, &h->il2d_cskw, &h->il2d_cskw_n, "csk row");
    if (h->il2d_turn_plan)
        _il2d_clone_set(h->il2d_turn_plan, cfg, h->N, VFFT_INPLACE, T, &h->il2d_turnw, &h->il2d_turnw_n, "turn");
}
static void _il2d_clone_set_drop(struct vfft_plan_s ***arr, int *n)
{
    int t;
    if (!*arr)
        return;
    for (t = 0; t < *n; t++)
        if ((*arr)[t])
            vfft_destroy((*arr)[t]);
    free(*arr);
    *arr = NULL;
    *n = 0;
}
static void _il2d_c2c_drop_unneeded_clones(struct vfft_plan_s *h)
{
    if (h->il2d_turn)
    {
        _il2d_clone_set_drop(&h->il2d_roww, &h->il2d_roww_n);
        _il2d_clone_set_drop(&h->il2d_cskw, &h->il2d_cskw_n);
        return;
    }
    if (h->il2d_csk)
    {
        _il2d_clone_set_drop(&h->il2d_roww, &h->il2d_roww_n);
        _il2d_clone_set_drop(&h->il2d_turnw, &h->il2d_turnw_n);
        if (h->il2d_rowb || h->il2d_rowb2)
            _il2d_clone_set_drop(&h->il2d_cskw, &h->il2d_cskw_n);
        return;
    }
    _il2d_clone_set_drop(&h->il2d_cskw, &h->il2d_cskw_n);
    _il2d_clone_set_drop(&h->il2d_turnw, &h->il2d_turnw_n);
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
    /* aligned like every plane the door serves, and the cell's own placement
     * (an out-of-place cell races x -> y): the axis race's law since
     * 2026-09-23, here since 2026-09-24 */
    double *z = (double *)VFFT_ZS_ALLOC(2 * PN * sizeof(double));
    double *zo = (h->placement == VFFT_OUTOFPLACE) ? (double *)VFFT_ZS_ALLOC(2 * PN * sizeof(double)) : z;
    double st = 1e300, mt = 1e300;
    int p;
    size_t i;
    if (!z || !zo)
    {
        VFFT_ZS_FREE(z);
        if (zo != z) VFFT_ZS_FREE(zo);
        return;
    }
    for (i = 0; i < 2 * PN; i++)
        z[i] = 1.0 + 1e-6 * (double)(i & 511);
    if (!_il2d_c2c_mt(h, z, zo, VFFT_FORWARD, h->nthreads))
    {
        h->il2d_col.colmt = 0; /* cannot engage — that IS the verdict */
        VFFT_ZS_FREE(z);
        if (zo != z) VFFT_ZS_FREE(zo);
        vw2_2d_il_chain_bank(&W->vw2, N1, N2, h->il2d_col.R, h->il2d_col.nst,
                             h->il2d_col.wl, h->il2d_col.tfuse, _il2d_ro_of(h),
                             0, h->nthreads,
                             (N1 & (N1 - 1)) ? h->il2d_col.blu : -1, 0.0, vfft_policy_ord_rankn(cfg));
        _vw2_persist(W, cfg);
        return;
    }
    {
        _il2d_race_ctx_t rc0 = { h, NULL, z, 0, 1, 0, 0, 0, NULL, NULL, NULL, NULL };
        _il2d_race_ctx_t rc[VFFT_RACE_MAX_ARMS];
        vfft_race_arm_t arms[VFFT_RACE_MAX_ARMS];
        char names[VFFT_RACE_MAX_ARMS][20];
        double ns[VFFT_RACE_MAX_ARMS];
        int lad[6], nl, a, na = 0, best = 1, k, nv;
        const vfft_race_proto_t proto = { 3, 1, VFFT_RACE_MIN, 0, 2, NULL, NULL, 0 }; /* min-of-3, A then B, two untimed passes per arm first (2026-09-25: a cold threaded arm read 58 us for a 37-us plan) */ /* THREADED arms: never paused (mt_measurement_parking_trap) */
        (void)p;
        for (a = 0; a < VFFT_RACE_MAX_ARMS; a++) { rc[a] = rc0; rc[a].sw = 0; rc[a].lst = 1; rc[a].zo = zo; ns[a] = 1e300; }
        arms[na].name = "serial"; arms[na].run = _il2d_arm_exec_st; arms[na].ctx = &rc[na]; na++;
        if (h->il2d_turn || h->il2d_csk)
        {   /* the turn and the skewed pass have ONE threaded walk each (2026-09-24) */
            arms[na].name = "threaded"; arms[na].run = _il2d_arm_exec_mt; arms[na].ctx = &rc[na]; na++;
            nl = 0;
        }
        else
            nl = _il2d_sw_ladder(N1, N2, h->nthreads, lad, 6);
        /* a natural cell races every threaded arm STAGED (nst = 1) and
         * STRIDED (nst = 0): the leaf's form is a raced plan parameter at
         * T > 1 (il2d_natural_leaf_design.md, 2026-09-16); a scrambled
         * cell has no leaf form */
        for (nv = 0; nv < ((h->il2d_turn || h->il2d_csk) ? 0 : (h->il2d_col.nat && h->il2d_col.natstage) ? 2 : 1); nv++)
        {
            const int nst = nv ? 0 : 1;
            const char *tag = (h->il2d_col.nat && !nst) ? "/str" : "";
            snprintf(names[na], sizeof names[na], "%s%s", h->il2d_col.nat ? "block" : "bands", tag);
            rc[na].lst = nst; arms[na].name = names[na]; arms[na].run = _il2d_arm_exec_mt; arms[na].ctx = &rc[na]; na++;
            if (h->il2d_col.nat)
            {
                snprintf(names[na], sizeof names[na], "strips%s", tag);
                rc[na].lst = nst; arms[na].name = names[na]; arms[na].run = _il2d_arm_exec_mt_strip; arms[na].ctx = &rc[na]; na++;
            }
            if (h->il2d_col.nat && h->il2d_col.R[0] >= 2 && na < VFFT_RACE_MAX_ARMS)
            {   /* the TILE partition (2026-09-25) */
                snprintf(names[na], sizeof names[na], "tile%s", tag);
                rc[na].lst = nst; arms[na].name = names[na]; arms[na].run = _il2d_arm_exec_mt_tile; arms[na].ctx = &rc[na]; na++;
            }
            for (k = 0; k < nl && na < VFFT_RACE_MAX_ARMS; k++)
            {
                rc[na].sw = lad[k]; rc[na].lst = nst;
                snprintf(names[na], sizeof names[na], "strips%d%s", lad[k], tag);
                arms[na].name = names[na]; arms[na].run = _il2d_arm_exec_mt_sw; arms[na].ctx = &rc[na]; na++;
            }
        }
        vfft_race_run(&proto, arms, na, ns);
        st = ns[0];
        for (a = 2; a < na; a++) if (ns[a] < ns[best]) best = a;
        mt = ns[best];
        h->il2d_col.natarm = (arms[best].run == _il2d_arm_exec_mt) ? 0
                           : (arms[best].run == _il2d_arm_exec_mt_tile) ? 2 : 1;
        h->il2d_col.msw = rc[best].sw;
        h->il2d_col.natst = rc[best].lst;
        if (getenv("VFFT_IL2D_LOG"))
        {
            fprintf(stderr, "[il2d-c2c] threaded arms %dx%d T=%d (%s):", N1, N2, h->nthreads, h->il2d_col.nat ? "nat" : "scr");
            for (a = 0; a < na; a++) fprintf(stderr, " %s=%.0f", arms[a].name, ns[a]);
            fprintf(stderr, "\n");
        }
    }
    h->il2d_col.colmt = (mt < st);
    if (!h->il2d_col.colmt) { h->il2d_col.natarm = 0; h->il2d_col.msw = 0; h->il2d_col.natst = 1; }
    VFFT_ZS_FREE(z);
    if (zo != z) VFFT_ZS_FREE(zo);
    if (getenv("VFFT_IL2D_LOG"))
        fprintf(stderr, "[il2d-c2c] colmt race %dx%d T=%d: st=%.0f "
                        "mt=%.0f -> %s%s\n",
                N1, N2, h->nthreads, st, mt,
                h->il2d_col.colmt ? "THREADED" : "serial",
                (h->il2d_col.colmt && h->il2d_col.natarm) ? " (strips)" : "");
    {   /* the verdict lands on the cell's EXISTING row through the field-update
         * path (ns = 0): a measured bank rebuilt the row and dropped the axis
         * race's tokens (sw, rbk, turn, csk) -- every T=8 replay of a tiled row
         * ran the default tile until 2026-09-24. The measured time is mtns=. */
        const int ord = vfft_policy_ord_rankn(cfg);
        const int have_row = vw2_2d_il_tok_geti(&W->vw2, N1, N2, ord, "chain", -1) >= 0;   /* chain= is on every row */
        vw2_2d_il_chain_bank(&W->vw2, N1, N2, h->il2d_col.R, h->il2d_col.nst,
                             h->il2d_col.wl, h->il2d_col.tfuse, _il2d_ro_of(h),
                             h->il2d_col.colmt, h->nthreads,
                             (N1 & (N1 - 1)) ? h->il2d_col.blu : -1,
                             have_row ? 0.0 : (h->il2d_col.colmt ? mt : st), ord);
        /* the threaded arm's shape beside cmt/cmtt, read back at that T only */
        vw2_2d_il_tok_seti(&W->vw2, N1, N2, ord, "mtarm", h->il2d_col.natarm);
        vw2_2d_il_tok_seti(&W->vw2, N1, N2, ord, "msw", h->il2d_col.msw);
        if (h->il2d_col.nat) vw2_2d_il_tok_seti(&W->vw2, N1, N2, ord, "nls", h->il2d_col.natst);
        vw2_2d_il_tok_seti(&W->vw2, N1, N2, ord, "mtns", (int)(h->il2d_col.colmt ? mt : st));
    }
    _vw2_persist(W, cfg);
}

#endif /* VFFT_TRANSFORMS_FFT2D_IL2D_TIER_H */
