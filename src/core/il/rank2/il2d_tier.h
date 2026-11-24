/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Tuğbars Heptaşkın */
/* il2d_tier.h - the native interleaved 2D tier.
 *
 * The row and column passes, their multithreaded forms, and the racers that
 * decide a 2D IL cell's plan. Almost every function here dereferences
 * vfft_plan_s (vfft_internal.h).
 *
 * WHAT DECIDES WHAT, AND IN WHICH ORDER
 * -------------------------------------
 * The tier's plan is not one search. It is a chain of races where an earlier
 * verdict decides whether a later axis EXISTS at all:
 *
 *   chain      factorize N1 over the radix pool (within the reach of
 *              VFFT_IL2D_CHAIN_REACH stages, past it at the fewest stages; the
 *              storage cap LOGS when it bites - a truncated pool is a biased pool).
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
 *              cmt=0 is banked with the thread count it was decided at. Every
 *              c2c route races it: the turn and the skewed column pass have
 *              row-slab walks of their own (one threaded arm).
 *
 * That last one is why a 256x256 cell has no MT axis while 64x64 and
 * 1024x1024 do - wl lands on N1 there, giving one band.
 *
 * WHY THE THREAD COUNT IS THE ROW'S KEY (nthreads=, wisdom2 v1.3)
 * ----------------------------------------------------------------
 * Column threading is the CORES-SHARE-ONE-TRANSFORM class: T decides how the
 * work is cut - band counts, worker clamps, legal row widths - so a cut that
 * wins at T=2 can lose at T=8, and the route itself changes with T (64x1024:
 * the chain at one thread, the skewed pass at eight). A threaded plan's row
 * is therefore its own, keyed by the plan's thread count and complete on its
 * own: the plain tokens are its verdict, raced at that T. Nothing serves
 * across thread counts. (The other MT class, one transform per core, keys
 * its verdict by T too: its plans do not depend on T, but how K transforms
 * share T workers does.)
 *
 * ONE ASYMMETRY THAT LOOKS LIKE A BUG AND IS NOT
 * ----------------------------------------------
 * The wl race is guarded with !il2d_oddn2; the column-MT guard is not. That
 * is deliberate: an odd N2's rows ride the c2c child and the banded walk
 * is not raced there, but column threading stays valid. Measured consistent - 128x127 at T=8 engages cmt and
 * is BIT-IDENTICAL to the single-threaded result.
 *
 * THE ENGAGEMENT COUNTER STAYS IN vfft.c
 * --------------------------------------
 * _vfft_il2d_col_mt_count is incremented from here but DEFINED there, with
 * external linkage. It cannot live in this header: a static in a header is one copy per
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
#include "common/support/build_isa.h"             /* VFFT_BUILD_ISA_AVX512: the build's ISA */
#include "common/support/threads.h"               /* the pool */
#include "common/support/race_timing.h"           /* the shared clock */
#include "il/wisdom/wisdom2_2d_il_reader.h"     /* the lay=il 2D cell codec */
#include "il/wisdom/wisdom2_child.h"            /* a child plan in role: its private store, its recipe on the cell's row */

/* Defined in vfft.c with external linkage; see the note above. */
extern long _vfft_il2d_col_mt_count;
extern long _vfft_il2d_row_mt_count;   /* the 2D real ROW plan's threaded passes (il2d_real_plan.h) */

/* the K=1 FOUR-STEP's inter-pass twiddle on one row (oop/k1_fourstep.h):
 * the row at plane position p times W_N^(k1(p) * n2), n2 = a*B + b, from
 * the per-position two-level record [coarse C[a] (N2/B)][fine F[b] (B)];
 * conj = the backward's conjugate.
 * By hand at both precisions, as (x F[b]) C[a]: the fine entry does not
 * change with a, so its two duplicated forms (fr, fr) and (fi, fi) are laid
 * out once per call, VFFT_IL2D_FS_HB entries at a time, and each of the two
 * multiplies is one swap, one product and one fused multiply-add/subtract.
 * The backward's conjugate is the sign of fi and of ci. Whole vectors
 * first, then point by point where a run is not a whole number of them. */
#ifdef VFFT_FLOAT
/* FP32: 4 points per ymm, 8 per zmm on an AVX-512 build. The arithmetic is
 * float throughout: written in double, the multiply converts every sample
 * to double and back and costs more than the row transform. */
#define VFFT_IL2D_FS_HB 128   /* fine entries laid out at a time: 2 KB of stack */
static void _il2d_fs_twiddle(vfft_real *row, const vfft_real *rec, int B, size_t n2, int conj)
{
    const size_t na = (n2 + (size_t)B - 1) / (size_t)B;   /* ceil: a non-pow2 row's last coarse entry is partial */
    const vfft_real *C = rec, *F = rec + 2 * na;
    const float sg = conj ? -1.0f : 1.0f;
    const __m256 sgv = _mm256_set1_ps(conj ? -0.0f : 0.0f);
    __attribute__((aligned(64))) float frd[2 * VFFT_IL2D_FS_HB], fid[2 * VFFT_IL2D_FS_HB];
    size_t a, b, b0;
    for (b0 = 0; b0 < (size_t)B; b0 += VFFT_IL2D_FS_HB)
    {
        const size_t hl = ((size_t)B - b0 < (size_t)VFFT_IL2D_FS_HB) ? (size_t)B - b0 : (size_t)VFFT_IL2D_FS_HB;
        const vfft_real *Fb = F + 2 * b0;
        for (b = 0; b + 4 <= hl; b += 4)
        {
            const __m256 f = _mm256_loadu_ps(Fb + 2 * b);
            _mm256_store_ps(frd + 2 * b, _mm256_moveldup_ps(f));
            _mm256_store_ps(fid + 2 * b, _mm256_xor_ps(_mm256_movehdup_ps(f), sgv));
        }
        for (; b < hl; b++)
        {
            frd[2 * b] = frd[2 * b + 1] = Fb[2 * b];
            fid[2 * b] = fid[2 * b + 1] = sg * Fb[2 * b + 1];
        }
        for (a = 0; a < na; a++)
        {
            const float cr = C[2 * a], ci = sg * C[2 * a + 1];
            vfft_real *r = row + 2 * (a * (size_t)B + b0);
            const size_t run = n2 - a * (size_t)B;   /* this coarse entry's points: B, fewer in a partial last one */
            const size_t bl = run <= b0 ? 0 : (run - b0 < hl ? run - b0 : hl);
            b = 0;
#if defined(VFFT_BUILD_ISA_AVX512)
            {
                const __m512 vcr = _mm512_set1_ps(cr), vci = _mm512_set1_ps(ci);
                for (; b + 8 <= bl; b += 8)
                {
                    const __m512 x = _mm512_loadu_ps(r + 2 * b);
                    const __m512 z = _mm512_fmaddsub_ps(x, _mm512_load_ps(frd + 2 * b),   /* x f: xr fr - xi fi, xi fr + xr fi */
                                                        _mm512_mul_ps(_mm512_permute_ps(x, 0xB1), _mm512_load_ps(fid + 2 * b)));
                    _mm512_storeu_ps(r + 2 * b, _mm512_fmaddsub_ps(z, vcr, _mm512_mul_ps(_mm512_permute_ps(z, 0xB1), vci)));
                }
            }
#endif
            {
                const __m256 vcr = _mm256_set1_ps(cr), vci = _mm256_set1_ps(ci);
                for (; b + 4 <= bl; b += 4)
                {
                    const __m256 x = _mm256_loadu_ps(r + 2 * b);
                    const __m256 z = _mm256_fmaddsub_ps(x, _mm256_load_ps(frd + 2 * b),   /* x f: xr fr - xi fi, xi fr + xr fi */
                                                        _mm256_mul_ps(_mm256_permute_ps(x, 0xB1), _mm256_load_ps(fid + 2 * b)));
                    _mm256_storeu_ps(r + 2 * b, _mm256_fmaddsub_ps(z, vcr, _mm256_mul_ps(_mm256_permute_ps(z, 0xB1), vci)));
                }
            }
            for (; b < bl; b++)
            {
                const float fr = frd[2 * b], fi = fid[2 * b];
                const float xr = r[2 * b], xi = r[2 * b + 1];
                const float zr = xr * fr - xi * fi, zi = xi * fr + xr * fi;
                r[2 * b] = zr * cr - zi * ci;
                r[2 * b + 1] = zi * cr + zr * ci;
            }
        }
    }
}
#else
/* FP64: 2 points per ymm, 4 per zmm on an AVX-512 build. */
#define VFFT_IL2D_FS_HB 64   /* fine entries laid out at a time: 2 KB of stack */
static void _il2d_fs_twiddle(vfft_real *row, const vfft_real *rec, int B, size_t n2, int conj)
{
    const size_t na = (n2 + (size_t)B - 1) / (size_t)B;   /* ceil: a non-pow2 row's last coarse entry is partial */
    const vfft_real *C = rec, *F = rec + 2 * na;
    const double sg = conj ? -1.0 : 1.0;
    const __m256d sgv = _mm256_set1_pd(conj ? -0.0 : 0.0);
    __attribute__((aligned(64))) double frd[2 * VFFT_IL2D_FS_HB], fid[2 * VFFT_IL2D_FS_HB];
    size_t a, b, b0;
    for (b0 = 0; b0 < (size_t)B; b0 += VFFT_IL2D_FS_HB)
    {
        const size_t hl = ((size_t)B - b0 < (size_t)VFFT_IL2D_FS_HB) ? (size_t)B - b0 : (size_t)VFFT_IL2D_FS_HB;
        const vfft_real *Fb = F + 2 * b0;
        b = 0;
#if defined(VFFT_BUILD_ISA_AVX512)
        {
            const __m512d sgz = _mm512_set1_pd(conj ? -0.0 : 0.0);
            for (; b + 4 <= hl; b += 4)
            {
                const __m512d f = _mm512_loadu_pd(Fb + 2 * b);
                _mm512_store_pd(frd + 2 * b, _mm512_movedup_pd(f));
                _mm512_store_pd(fid + 2 * b, _mm512_xor_pd(_mm512_permute_pd(f, 0xFF), sgz));
            }
        }
#endif
        for (; b + 2 <= hl; b += 2)
        {
            const __m256d f = _mm256_loadu_pd(Fb + 2 * b);
            _mm256_store_pd(frd + 2 * b, _mm256_movedup_pd(f));
            _mm256_store_pd(fid + 2 * b, _mm256_xor_pd(_mm256_permute_pd(f, 0xF), sgv));
        }
        for (; b < hl; b++)
        {
            frd[2 * b] = frd[2 * b + 1] = Fb[2 * b];
            fid[2 * b] = fid[2 * b + 1] = sg * Fb[2 * b + 1];
        }
        for (a = 0; a < na; a++)
        {
            const double cr = C[2 * a], ci = sg * C[2 * a + 1];
            vfft_real *r = row + 2 * (a * (size_t)B + b0);
            const size_t run = n2 - a * (size_t)B;   /* this coarse entry's points: B, fewer in a partial last one */
            const size_t bl = run <= b0 ? 0 : (run - b0 < hl ? run - b0 : hl);
            b = 0;
#if defined(VFFT_BUILD_ISA_AVX512)
            {
                const __m512d vcr = _mm512_set1_pd(cr), vci = _mm512_set1_pd(ci);
                for (; b + 4 <= bl; b += 4)
                {
                    const __m512d x = _mm512_loadu_pd(r + 2 * b);
                    const __m512d z = _mm512_fmaddsub_pd(x, _mm512_load_pd(frd + 2 * b),   /* x f: xr fr - xi fi, xi fr + xr fi */
                                                         _mm512_mul_pd(_mm512_permute_pd(x, 0x55), _mm512_load_pd(fid + 2 * b)));
                    _mm512_storeu_pd(r + 2 * b, _mm512_fmaddsub_pd(z, vcr, _mm512_mul_pd(_mm512_permute_pd(z, 0x55), vci)));
                }
            }
#endif
            {
                const __m256d vcr = _mm256_set1_pd(cr), vci = _mm256_set1_pd(ci);
                for (; b + 2 <= bl; b += 2)
                {
                    const __m256d x = _mm256_loadu_pd(r + 2 * b);
                    const __m256d z = _mm256_fmaddsub_pd(x, _mm256_load_pd(frd + 2 * b),   /* x f: xr fr - xi fi, xi fr + xr fi */
                                                         _mm256_mul_pd(_mm256_permute_pd(x, 0x5), _mm256_load_pd(fid + 2 * b)));
                    _mm256_storeu_pd(r + 2 * b, _mm256_fmaddsub_pd(z, vcr, _mm256_mul_pd(_mm256_permute_pd(z, 0x5), vci)));
                }
            }
            for (; b < bl; b++)
            {
                const double fr = frd[2 * b], fi = fid[2 * b];
                const double xr = r[2 * b], xi = r[2 * b + 1];
                const double zr = xr * fr - xi * fi, zi = xi * fr + xr * fi;
                r[2 * b] = zr * cr - zi * ci;
                r[2 * b + 1] = zi * cr + zr * ci;
            }
        }
    }
}
#endif
static const vfft_real *_il2d_fs_rec(const struct vfft_plan_s *h, size_t rn, size_t p)
{
    return h->il2d_fs_tw ? h->il2d_fs_tw + p * 2 * ((rn + (size_t)h->il2d_fs_B - 1) / (size_t)h->il2d_fs_B + (size_t)h->il2d_fs_B) : NULL;
}

/* one row of the row pass through the in-place row child (route 0, one
 * door walk per row; the batched routes run rows through _il2d_rows_exec
 * and never come here). p = the row's plane position (the four-step's
 * twiddle hook keys on it; every other plan ignores it). */
static void _il2d_row_exec(struct vfft_plan_s *h, vfft_dir_t dir,
                           vfft_real *row, size_t rn, size_t p)
{
    const vfft_real *tw = _il2d_fs_rec(h, rn, p);
    if (tw && dir == VFFT_FORWARD)
        _il2d_fs_twiddle(row, tw, h->il2d_fs_B, rn, 0);
    _vfft_exec6(h->il2d_row, dir, row, NULL, row, NULL);
    if (tw && dir != VFFT_FORWARD)
        _il2d_fs_twiddle(row, tw, h->il2d_fs_B, rn, 1);
}

/* ── native IL 2D REAL row passes (docs/roadmap/fft2d_real_il_design.md §2.4): ONE
 * function per direction, used by BOTH the execute and the create-time
 * races (race == serving path). The route: the per-row door (the K=1 1D
 * real engine at N2, il2d_row) over every row; at an odd N2 its c2c(N2)
 * child with the promote / extend edges. */
/* the real plane's row pitch in doubles: N2 out of place; in place the padded
 * row of the one plane, 2 il2d_ipP (docs/roadmap/real_inplace_design.md) */
static inline size_t _il2d_rp(const struct vfft_plan_s *h)
{
    return h->il2d_ip ? 2 * h->il2d_ipP : (size_t)h->N2;
}
/* the CCE plane's row pitch in doubles: 2 (N2/2 + 1) out of place; in place the
 * same padded row, 2 il2d_ipP */
static inline size_t _il2d_cpd(const struct vfft_plan_s *h)
{
    return h->il2d_ip ? 2 * h->il2d_ipP : 2 * ((size_t)h->N2 / 2 + 1);
}
static inline void _tc_one(struct vfft_plan_s *in, vfft_dir_t dir, vfft_real *s, vfft_real *d);   /* il/il_execute.h */
/* THE DOOR ROUTE IN PLACE: the K = 1 in-place inner on every row of the plane, a landing's row
 * (c2r from the column-inverse plane, at its pitch Pz) moved onto its own row first. The row
 * batch's threading verdict (tc_mt) decides as in its own execute (il_execute.h): slabs, the
 * rows split across the batch's own clones (each a serial in-place plan: the slab role, vfft.c),
 * the caller on the primary's serial twin -- an engagement of the row pass; else the serial loop,
 * every row on the primary (its threaded form, when it has one, on each row). */
typedef struct { struct vfft_plan_s *h; const vfft_real *s; vfft_real *d; int bwd, tid, mt; size_t lo, hi, Pz; } _il2d_door_ip_t;
static void _il2d_door_ip_range(void *v)
{
    const _il2d_door_ip_t *a = (const _il2d_door_ip_t *)v;
    const struct vfft_plan_s *h = a->h;
    struct vfft_plan_s *row = h->il2d_row;
    struct vfft_plan_s *in = a->tid > 0 ? row->tcbw[a->tid - 1]
                                        : (row->tcb ? ((a->mt && row->tcb0) ? row->tcb0 : row->tcb) : row);
    const size_t rp = _il2d_rp(h), hp1 = (size_t)h->N2 / 2 + 1;
    size_t r;
    for (r = a->lo; r < a->hi; r++)
    {
        vfft_real *dst = a->d + r * rp;
        const vfft_real *src = a->bwd ? a->s + r * 2 * a->Pz : a->s + r * rp;
        if (src != dst)
            memcpy(dst, src, (a->bwd ? 2 * hp1 : (size_t)h->N2) * sizeof(vfft_real));
        _tc_one(in, a->bwd ? VFFT_BACKWARD : VFFT_FORWARD, dst, dst);
    }
}
static void _il2d_door_ip(struct vfft_plan_s *h, const vfft_real *s, vfft_real *d, int bwd, size_t Pz)
{
    _il2d_door_ip_t a[THREAD_POOL_MAX_DISPATCH];
    const size_t n1 = (size_t)h->N;
    struct vfft_plan_s *row = h->il2d_row;
    int T = (h->nthreads > 1 && row->tcb && row->tc_mt) ? thread_pool_workers_for(h->nthreads) : 1, t;
    if (T > row->tcbw_n + 1) T = row->tcbw_n + 1;
    if ((size_t)T > n1) T = (int)n1;
    if (T < 1) T = 1;
    for (t = 0; t < T; t++)
    {
        a[t].h = h; a[t].s = s; a[t].d = d; a[t].bwd = bwd; a[t].tid = t; a[t].mt = T > 1; a[t].Pz = Pz;
        a[t].lo = n1 * (size_t)t / (size_t)T;
        a[t].hi = n1 * (size_t)(t + 1) / (size_t)T;
    }
    if (T < 2)
    {
        _il2d_door_ip_range(&a[0]);
        return;
    }
    thread_pool_run(T, _il2d_door_ip_range, a, sizeof a[0]);
    _vfft_il2d_row_mt_count++;   /* engagement, see vfft.c */
}
static void _il2d_rowx_fwd(struct vfft_plan_s *h, const vfft_real *sre, vfft_real *dre); /* il2d_real_plan.h, later in this TU */
static void _il2d_real_rows_fwd_route(struct vfft_plan_s *h, const vfft_real *sre,
                                      vfft_real *dre)
{
    const size_t hp1 = (size_t)h->N2 / 2 + 1, rp = _il2d_rp(h), cpd = _il2d_cpd(h);
    if (h->il2d_oddn2)
    { /* odd N2: promote -> c2c(N2) -> keep the hp1 CCE bins */
        const size_t rn2 = (size_t)h->N2;
        vfft_real *b1 = h->il2d_orbuf, *b2 = h->il2d_orbuf + 2 * rn2;
        size_t r;
        for (r = 0; r < (size_t)h->N; r++)
        {
            _il2d_row_promote(sre + r * rp, b1, rn2);
            _vfft_exec6((vfft_plan)h->il2d_row, VFFT_FORWARD, b1, NULL,
                         b2, NULL);
            memcpy(dre + r * cpd, b2, 2 * hp1 * sizeof(vfft_real));
        }
        return;
    }
    if (h->il2d_ip)
    {   /* in place: the door's K = 1 inner on every row of the one plane, the rows across the
         * batch's clones at T > 1 */
        _il2d_door_ip(h, sre, dre, 0, 0);
        return;
    }
    _vfft_exec6(h->il2d_row, VFFT_FORWARD, (vfft_real *)sre, NULL, dre, NULL);
}

/* the r2c row pass: the plan's own row plan when one is bound (the row
 * race's verdict -- the rows kernel, a real engine per row, or the route
 * above -- entered at the plan's stack state), else the route as it stands */
static void _il2d_real_rows_fwd(struct vfft_plan_s *h, const vfft_real *sre,
                                vfft_real *dre)
{
    if (h->il2d_rx_on)
    {
        _il2d_rowx_fwd(h, sre, dre);
        return;
    }
    _il2d_real_rows_fwd_route(h, sre, dre);
}

static void _il2d_rowx_bwd(struct vfft_plan_s *h, const vfft_real *zsrc, vfft_real *dre); /* il2d_real_plan.h, later in this TU */
static void _il2d_real_rows_bwd_route(struct vfft_plan_s *h, const vfft_real *zsrc,
                                      vfft_real *dre)
{
    const size_t hp1 = (size_t)h->N2 / 2 + 1;
    if (h->il2d_oddn2)
    { /* odd N2: Hermitian-extend hp1 -> N2 -> inverse c2c -> Re. The
       * inverse is unnormalized (x N2), matching the even tier's c2r
       * scale contract. */
        const size_t rn2 = (size_t)h->N2, rp = _il2d_rp(h);
        const size_t Pz = (zsrc == h->il2d_rscr && h->il2d_rscr_P) ? h->il2d_rscr_P : h->il2d_col.rn;   /* the source's pitch */
        vfft_real *b1 = h->il2d_orbuf, *b2 = h->il2d_orbuf + 2 * rn2;
        size_t r;
        for (r = 0; r < (size_t)h->N; r++)
        {
            _il2d_row_extend(zsrc + r * 2 * Pz, b1, rn2, hp1);
            _vfft_exec6((vfft_plan)h->il2d_row, VFFT_BACKWARD, b1, NULL,
                         b2, NULL);
            _il2d_row_re(b2, dre + r * rp, rn2);
        }
        return;
    }
    if (h->il2d_ip)
    {   /* in place: the door's K = 1 in-place inner on every row of the plane, a row coming from
         * the column-inverse plane (the private landing) moved onto its own row first; the rows
         * across the batch's clones at T > 1 */
        const size_t Pz = (zsrc == h->il2d_rscr && h->il2d_rscr_P) ? h->il2d_rscr_P : h->il2d_col.rn;
        _il2d_door_ip(h, zsrc, dre, 1, Pz);
        return;
    }
    _vfft_exec6(h->il2d_row, VFFT_BACKWARD, (vfft_real *)zsrc, NULL, dre, NULL);
}

/* the c2r row pass: the plan's own row plan when one is bound (the c2r row
 * race's verdict -- the backward rows kernel, a real engine's backward per
 * row, or the route above -- entered at the plan's stack state), else the
 * route as it stands */
static void _il2d_real_rows_bwd(struct vfft_plan_s *h, const vfft_real *zsrc,
                                vfft_real *dre)
{
    if (h->il2d_rx_on)
    {
        _il2d_rowx_bwd(h, zsrc, dre);
        return;
    }
    _il2d_real_rows_bwd_route(h, zsrc, dre);
}


/* ── THE TURNED PRIME COLUMN PASS (tpc) ──────────────────────────────────
 * At a prime N (no chain) the column Bluestein pass measured 0.51-0.70 over
 * pow2 lanes (131x4096, 257x256; the rank-3 cells 131x64x64 and 257x16x16
 * are this pass over N2*N3 lanes), while the 1D prime route at the same N
 * wins 1.1-3.1x. The turned pass runs the axis through that route: the lanes
 * [c0, c1) of the plane transposed into the scratch (row c = lane c, pitch
 * N + 8), the 1D natural in-place K=1 plan at N on every row of it, the rows
 * transposed back into the plane's lanes. At 131x4096: turn 426 + rows 1585
 * + back 527 us against the Bluestein pass's ~8400. Both directions (the 1D
 * plan's own backward). */
#define VFFT_IL2D_TPC_PITCH(N) ((size_t)(N) + 8)
#if defined(__AVX512F__) && !defined(VFFT_FLOAT)
/* THE 4 x 4 COMPLEX TURN at 512 bits (the turns below and the TURN route's back-turn): four
 * vectors of four points, a row each, into the four columns' vectors -- two rounds of the
 * 128-bit-lane deinterleave. It is its own inverse. */
#define VFFT_IL2D_TURN4 1
static inline __attribute__((always_inline)) void _il2d_turn4(const __m512d a0, const __m512d a1, const __m512d a2,
                                                             const __m512d a3, __m512d *o0, __m512d *o1,
                                                             __m512d *o2, __m512d *o3)
{
    const __m512d t0 = _mm512_shuffle_f64x2(a0, a1, 0x88), t1 = _mm512_shuffle_f64x2(a0, a1, 0xDD);
    const __m512d t2 = _mm512_shuffle_f64x2(a2, a3, 0x88), t3 = _mm512_shuffle_f64x2(a2, a3, 0xDD);
    *o0 = _mm512_shuffle_f64x2(t0, t2, 0x88);
    *o1 = _mm512_shuffle_f64x2(t1, t3, 0x88);
    *o2 = _mm512_shuffle_f64x2(t0, t2, 0xDD);
    *o3 = _mm512_shuffle_f64x2(t1, t3, 0xDD);
}
#endif
/* S[c][r] = src[r][c] for lanes c in [c0, c1), rows r < N (src pitch rn, S
 * pitch P): 2 x 2 complex blocks through one 128-bit lane permute (at 512 bits
 * 4 x 4 blocks first, _il2d_turn4), the rows blocked so a block's source lines
 * stay in L1 across the lane groups */
static void _il2d_tpc_turn_in(const vfft_real *src, size_t rn, size_t N, size_t c0, size_t c1,
                              vfft_real *S, size_t P)
{
    const size_t RB = 256;
    size_t r0, c, r;
    for (r0 = 0; r0 < N; r0 += RB)
    {
        const size_t r1 = (r0 + RB < N) ? r0 + RB : N;
        c = c0;
#ifdef VFFT_IL2D_TURN4
        for (; c + 4 <= c1; c += 4)
        {   /* four lanes a step; the rows a 4 x 4 block does not fill go point by point */
            vfft_real *s0 = S + 2 * (c * P), *s1 = S + 2 * ((c + 1) * P), *s2 = S + 2 * ((c + 2) * P),
                      *s3 = S + 2 * ((c + 3) * P);
            for (r = r0; r + 4 <= r1; r += 4)
            {
                __m512d o0, o1, o2, o3;
                _il2d_turn4(_mm512_loadu_pd(src + 2 * (r * rn + c)), _mm512_loadu_pd(src + 2 * ((r + 1) * rn + c)),
                            _mm512_loadu_pd(src + 2 * ((r + 2) * rn + c)), _mm512_loadu_pd(src + 2 * ((r + 3) * rn + c)),
                            &o0, &o1, &o2, &o3);
                _mm512_storeu_pd(s0 + 2 * r, o0);
                _mm512_storeu_pd(s1 + 2 * r, o1);
                _mm512_storeu_pd(s2 + 2 * r, o2);
                _mm512_storeu_pd(s3 + 2 * r, o3);
            }
            for (; r < r1; r++)
            {
                _mm_storeu_pd(s0 + 2 * r, _mm_loadu_pd(src + 2 * (r * rn + c)));
                _mm_storeu_pd(s1 + 2 * r, _mm_loadu_pd(src + 2 * (r * rn + c + 1)));
                _mm_storeu_pd(s2 + 2 * r, _mm_loadu_pd(src + 2 * (r * rn + c + 2)));
                _mm_storeu_pd(s3 + 2 * r, _mm_loadu_pd(src + 2 * (r * rn + c + 3)));
            }
        }
#endif
        for (; c + 2 <= c1; c += 2)
        {
            vfft_real *sa = S + 2 * (c * P), *sb = S + 2 * ((c + 1) * P);
            for (r = r0; r + 2 <= r1; r += 2)
            {
#ifdef VFFT_FLOAT
                vfftf__turn2(sa + 2 * r, sb + 2 * r, src + 2 * (r * rn + c), src + 2 * ((r + 1) * rn + c));
#else
                const __m256d a = _mm256_loadu_pd(src + 2 * (r * rn + c));
                const __m256d b = _mm256_loadu_pd(src + 2 * ((r + 1) * rn + c));
                _mm256_storeu_pd(sa + 2 * r, _mm256_permute2f128_pd(a, b, 0x20));
                _mm256_storeu_pd(sb + 2 * r, _mm256_permute2f128_pd(a, b, 0x31));
#endif
            }
            if (r < r1)
            {
#ifdef VFFT_FLOAT
                vfftf__cx1(sa + 2 * r, src + 2 * (r * rn + c));
                vfftf__cx1(sb + 2 * r, src + 2 * (r * rn + c + 1));
#else
                _mm_storeu_pd(sa + 2 * r, _mm_loadu_pd(src + 2 * (r * rn + c)));
                _mm_storeu_pd(sb + 2 * r, _mm_loadu_pd(src + 2 * (r * rn + c + 1)));
#endif
            }
        }
        if (c < c1)
        {
            vfft_real *sa = S + 2 * (c * P);
            for (r = r0; r < r1; r++)
#ifdef VFFT_FLOAT
                vfftf__cx1(sa + 2 * r, src + 2 * (r * rn + c));
#else
                _mm_storeu_pd(sa + 2 * r, _mm_loadu_pd(src + 2 * (r * rn + c)));
#endif
        }
    }
}
/* dst[r][c] = S[c][r] for lanes c in [c0, c1): the turn route's back-turn
 * over a lane range (dst pitch rn, S pitch P) */
static void _il2d_tpc_turn_out(const vfft_real *S, size_t P, vfft_real *dst, size_t rn, size_t N,
                               size_t c0, size_t c1)
{
    const size_t RB = 256;
    size_t r0, c, r;
    for (r0 = 0; r0 < N; r0 += RB)
    {
        const size_t r1 = (r0 + RB < N) ? r0 + RB : N;
        c = c0;
#ifdef VFFT_IL2D_TURN4
        for (; c + 4 <= c1; c += 4)
        {   /* four lanes a step; the rows a 4 x 4 block does not fill go point by point */
            const vfft_real *t0 = S + 2 * (c * P), *t1 = S + 2 * ((c + 1) * P), *t2 = S + 2 * ((c + 2) * P),
                            *t3 = S + 2 * ((c + 3) * P);
            for (r = r0; r + 4 <= r1; r += 4)
            {
                __m512d o0, o1, o2, o3;
                _il2d_turn4(_mm512_loadu_pd(t0 + 2 * r), _mm512_loadu_pd(t1 + 2 * r), _mm512_loadu_pd(t2 + 2 * r),
                            _mm512_loadu_pd(t3 + 2 * r), &o0, &o1, &o2, &o3);
                _mm512_storeu_pd(dst + 2 * (r * rn + c), o0);
                _mm512_storeu_pd(dst + 2 * ((r + 1) * rn + c), o1);
                _mm512_storeu_pd(dst + 2 * ((r + 2) * rn + c), o2);
                _mm512_storeu_pd(dst + 2 * ((r + 3) * rn + c), o3);
            }
            for (; r < r1; r++)
            {
                _mm_storeu_pd(dst + 2 * (r * rn + c), _mm_loadu_pd(t0 + 2 * r));
                _mm_storeu_pd(dst + 2 * (r * rn + c + 1), _mm_loadu_pd(t1 + 2 * r));
                _mm_storeu_pd(dst + 2 * (r * rn + c + 2), _mm_loadu_pd(t2 + 2 * r));
                _mm_storeu_pd(dst + 2 * (r * rn + c + 3), _mm_loadu_pd(t3 + 2 * r));
            }
        }
#endif
        for (; c + 2 <= c1; c += 2)
        {
            const vfft_real *ta = S + 2 * (c * P), *tb = S + 2 * ((c + 1) * P);
            for (r = r0; r + 2 <= r1; r += 2)
            {
#ifdef VFFT_FLOAT
                {   /* the turn back: [ta r, ta r+1], [tb r, tb r+1] -> rows r and r+1 at lanes c, c+1 */
                    float tmp[8];
                    vfftf__turn2(tmp, tmp + 4, ta + 2 * r, tb + 2 * r);
                    vfftf__cx1(dst + 2 * (r * rn + c), tmp);
                    vfftf__cx1(dst + 2 * (r * rn + c + 1), tmp + 2);
                    vfftf__cx1(dst + 2 * ((r + 1) * rn + c), tmp + 4);
                    vfftf__cx1(dst + 2 * ((r + 1) * rn + c + 1), tmp + 6);
                }
#else
                const __m256d a = _mm256_loadu_pd(ta + 2 * r), b = _mm256_loadu_pd(tb + 2 * r);
                _mm256_storeu_pd(dst + 2 * (r * rn + c), _mm256_permute2f128_pd(a, b, 0x20));
                _mm256_storeu_pd(dst + 2 * ((r + 1) * rn + c), _mm256_permute2f128_pd(a, b, 0x31));
#endif
            }
            if (r < r1)
            {
#ifdef VFFT_FLOAT
                vfftf__cx1(dst + 2 * (r * rn + c), ta + 2 * r);
                vfftf__cx1(dst + 2 * (r * rn + c + 1), tb + 2 * r);
#else
                _mm_storeu_pd(dst + 2 * (r * rn + c), _mm_loadu_pd(ta + 2 * r));
                _mm_storeu_pd(dst + 2 * (r * rn + c + 1), _mm_loadu_pd(tb + 2 * r));
#endif
            }
        }
        if (c < c1)
        {
            const vfft_real *ta = S + 2 * (c * P);
            for (r = r0; r < r1; r++)
#ifdef VFFT_FLOAT
                vfftf__cx1(dst + 2 * (r * rn + c), ta + 2 * r);
#else
                _mm_storeu_pd(dst + 2 * (r * rn + c), _mm_loadu_pd(ta + 2 * r));
#endif
        }
    }
}
/* the pass over lanes [c0, c1): turn in, the 1D plan per row, turn out -- worker w's
 * plan (0: the plan's own, w > 0: its clone) */
static void _il2d_tpc_cols_range_w(const vfft_ilcol_t *c, int w, const vfft_real *src, vfft_real *dst,
                                   size_t rn, size_t c0, size_t c1, int rev)
{
    const size_t N = (size_t)c->N, P = VFFT_IL2D_TPC_PITCH(N);
    vfft_real *S = c->tpcscr;
    struct vfft_plan_s *pl = (w > 0 && w <= c->tpcw_n) ? c->tpcw[w - 1] : c->tpcplan;
    size_t l;
    _il2d_tpc_turn_in(src, rn, N, c0, c1, S, P);
    for (l = c0; l < c1; l++)
        _vfft_exec6((vfft_plan)pl, rev ? VFFT_BACKWARD : VFFT_FORWARD,
                     S + 2 * l * P, NULL, S + 2 * l * P, NULL);
    _il2d_tpc_turn_out(S, P, dst, rn, N, c0, c1);
}
static void _il2d_tpc_cols_range(const vfft_ilcol_t *c, const vfft_real *src, vfft_real *dst,
                                 size_t rn, size_t c0, size_t c1, int rev)
{
    _il2d_tpc_cols_range_w(c, 0, src, dst, rn, c0, c1, rev);
}
/* THE TURNED PRIME PASS THREADED (2026-11-18; serial, it ran 1.0x at every T: 2D 4099 x 256, lane-major 4099:32,
 * 4483:64): every worker its lanes -- the turn into its rows of the scratch, the 1D plan on each through its own
 * clone (worker 0 the plan's own: the clones carry its recipe, route-checked at build), the turn back -- cut at
 * four-lane boundaries (no destination line shared between two workers' lanes but at the plane's edge). The
 * values are the serial pass's, bit for bit. 1 = it ran threaded, 0 = the caller runs the serial pass (T < 2,
 * fewer clones than workers, fewer than two lane groups) */
typedef struct
{
    const vfft_ilcol_t *c;
    const vfft_real *src;
    vfft_real *dst;
    size_t rn, lo, hi;
    int rev, w;
} _il2d_tpc_mt_arg;
static void _il2d_tpc_mt_tramp(void *v)
{
    const _il2d_tpc_mt_arg *a = (const _il2d_tpc_mt_arg *)v;
    if (a->lo < a->hi)
        _il2d_tpc_cols_range_w(a->c, a->w, a->src, a->dst, a->rn, a->lo, a->hi, a->rev);
}
static int _il2d_tpc_cols_mt(const vfft_ilcol_t *c, const vfft_real *src, vfft_real *dst, size_t rn, int rev, int T)
{
    _il2d_tpc_mt_arg a[THREAD_POOL_MAX_DISPATCH];
    int t;
    T = thread_pool_workers_for(T);
    if ((size_t)T > rn / 4)
        T = (int)(rn / 4);
    if (T < 2 || !c->tpcplan || c->tpcw_n < T - 1)   /* (the races time it before the verdict sets c->tpc) */
        return 0;
    for (t = 0; t < T; t++)
    {
        a[t].c = c;
        a[t].src = src;
        a[t].dst = dst;
        a[t].rn = rn;
        a[t].rev = rev;
        a[t].w = t;
        a[t].lo = (rn * (size_t)t / (size_t)T) & ~(size_t)3;
        a[t].hi = (t == T - 1) ? rn : (rn * (size_t)(t + 1) / (size_t)T) & ~(size_t)3;
    }
    thread_pool_run(T, _il2d_tpc_mt_tramp, a, sizeof a[0]);   /* caller = a[0] */
    return 1;
}
/* the turned pass as a race arm at the cell's T: threaded above 1 where it engages, else serial */
static void _il2d_tpc_arm(const vfft_ilcol_t *c, const vfft_real *src, vfft_real *dst, size_t rn, int T)
{
    if (T > 1 && _il2d_tpc_cols_mt(c, src, dst, rn, 0, T))
        return;
    _il2d_tpc_cols_range(c, src, dst, rn, 0, rn, 0);
}

/* THE COLUMN-ONLY SERIAL EXECUTE: the pass a rank-N IL plan runs on each
 * column axis and the real tier runs on its hp1-wide plane — natural
 * (leaf-redirected), Bluestein, turned, banded (no row fusion: the owning plan
 * runs the rows) or unbanded, both directions, by the descriptor alone. The
 * c2c 2D serial walk with the fused row pass lives in vfft_execute.h. */
static void _il2d_col_exec_st(const vfft_ilcol_t *c, const vfft_real *src,
                              vfft_real *dst, int reverse, vfft_real *stage)
{
    const size_t hp1 = c->rn;
    if (c->nat)
    { /* NATURAL n1: the leaf-redirected pass, unbanded by
       * construction (wl pinned 0 at create). */
        _il2d_col_pass_nat(src, dst, c->N, hp1, hp1, c->nst, c->R,
                           c->L,
                           reverse ? c->b : c->f,
                           reverse ? c->tb : c->tf, reverse,
                           c->natperm, c->natscr, stage);
        return;
    }
    if (c->blu && c->tpc)
    {   /* prime N1: the TURNED pass */
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
        vfft_real *const *tabs = reverse ? c->tb : c->tf;
        size_t b0;
        if (!reverse)
        {
            if (cut > 0)
                _il2d_col_stages(src, dst, c->N, hp1, 0, cut,
                                 c->R, c->L, fns, tabs, 0);
            for (b0 = 0; b0 < (size_t)c->N; b0 += wl)
            {
                const vfft_real *bs = (cut > 0) ? dst + 2 * b0 * hp1
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
/* the natural leaf at its natural stride (the standing serial form) */
static void _il2d_col_exec(const vfft_ilcol_t *c, const vfft_real *src,
                           vfft_real *dst, int reverse)
{
    _il2d_col_exec_st(c, src, dst, reverse, NULL);
}

/* ── native IL 2D REAL column pass (banded-aware; execute AND the wl
 * race serve through it — race == serving path): the descriptor's, over hp1
 * columns. The banded walk is the c2c tier's banded column form MINUS the row
 * fusion: fft2d_real_il_design.md §2.5 keeps rows entirely OUTSIDE (fwd rows
 * complete before any stage; bwd rows follow the last stage), so a band is
 * pure column loop interchange — F0-bitwise vs unbanded. fwd = wide prefix
 * 0..cut-1 full-plane, then per band of wl rows the suffix depth-first; bwd
 * (the Hermitian transpose chain) = per band the REVERSED suffix (its first
 * executed stage does the OOP move for c2r's z->rscr), then the reversed
 * prefix in place on dst. */
static void _il2d_colx_fwd(struct vfft_plan_s *h, const vfft_real *src, vfft_real *dst); /* il2d_real_plan.h, later in this TU */
static void _il2d_colx_bwd(struct vfft_plan_s *h, const vfft_real *src, vfft_real *dst); /* its c2r twin */
static void _il2d_cxd_cols(struct vfft_plan_s *h, vfft_real *z);   /* the destroying c2r's column pass, in place on the caller's plane */
static int _il2d_real_cols_mt(struct vfft_plan_s *h, const vfft_real *src, vfft_real *dst, int reverse, int T); /* below */
/* THE COLUMN PASS AS IT SERVES (2026-10-06, the thread guard lifted): the
 * plan's form when one is bound (it threads itself under the cell's colmt
 * verdict, il2d_real_plan.h), else the chain's pass -- threaded under the
 * verdict, serial when the threaded walk cannot engage or the verdict is
 * serial. The execute and every create-time race come through here. */
static void _il2d_real_cols(struct vfft_plan_s *h, const vfft_real *src,
                            vfft_real *dst, int reverse)
{
    if (!reverse && h->il2d_cx_on)
    { /* the r2c column plan: the pass at the plan's stack state, the natural leaf in its raced form */
        _il2d_colx_fwd(h, src, dst);
        return;
    }
    if (reverse && h->il2d_cx_on)
    { /* the c2r column plan (a c2r plan's own, cx_c2r=): the reverse chain at its stack states, or the backward leaf */
        _il2d_colx_bwd(h, src, dst);
        return;
    }
    if (h->il2d_col.colmt && h->nthreads > 1 && _il2d_real_cols_mt(h, src, dst, reverse, h->nthreads))
        return;
    _il2d_col_exec(&h->il2d_col, src, dst, reverse);
}

/* ══ MT column pass (docs/design/il2d_real_mt.md) ════════════════════
 * TWO partition arms, both pure loop restrictions of the SERVING loops
 * above (no arithmetic changes => MT == ST bitwise, gated):
 *   BAND arm  (wl > 0): workers take disjoint sets of wl-row BANDS of
 *     the suffix stages. Measured EXCHANGE-FREE (reading rows you wrote
 *     scales 7.8-7.9x) because these are the same rows the row pass just
 *     produced. The wide prefix stages [0,cut) — stage 0 spans the whole
 *     plane — run before the bands, each split by DIGIT (below).
 *   STRIP arm (wl == 0): workers take disjoint COLUMN ranges and run
 *     the whole chain. This is the ONLY axis for single-stage chains
 *     (L[0] == N1 => one block, no row axis), and it pays the full
 *     cross-core exchange (~2.9-3.4x, not 8x). Boundaries are
 *     NOT rounded to cache lines: hp1 is always odd, so a row's start
 *     rotates and rounding neither removes the split lines nor pays
 *     for itself (it would collapse hp1=33 from 8 workers to 5). */
typedef struct
{
    const vfft_ilcol_t *c;   /* the column descriptor the pass runs: the plan's, or a chain race candidate's view */
    int N1;
    vfft_real *stage;           /* the natural leaf's staging slots (T x 2 R_last rn doubles); NULL = the strided leaf */
    const vfft_real *src;
    vfft_real *dst;
    int reverse;
    size_t lo, hi;   /* band index range, or column range */
    int strip;
    int natleaf; /* natural x MT: this dispatch is the leaf block range */
    int tid;     /* this dispatch's worker: its staging slot (the column plan's staged form) */
    int stk;     /* the column plan's pass state: the body entered through the aligned call when >= 0 */
} _il2d_cmt_arg;
static vfft_real *_il2d_nat_stage_of(struct vfft_plan_s *h, int tid); /* below */
static inline void _il2d_rowx_call(void (*fn)(const void *, const vfft_real *, vfft_real *), const void *h,
                                   const vfft_real *s, vfft_real *d, int stk); /* il2d_real_plan.h: the aligned entry */

/* ── the DIGIT axis of a wide (prefix) stage ─────────────────────────
 * A stage's kernel walks digits itself — per digit it advances
 * `twp += (R-1)*8` and `zin/zout += 2*Gs` (verified in the emitted
 * body) — so running only digits [d0, d0+nd) is THREE pointer edits:
 * base + 2*d0*pitch, table + d0*(R-1)*8, OGs = nd. Digit d owns rows
 * {b*L + d + j*D}, disjoint across d, WHOLE ROWS, `count` untouched,
 * and no new codelet. That makes the full-plane prefix stage — the
 * last serial chunk of the banded column pass — parallel over D. */
typedef struct
{
    const vfft_real *src;
    vfft_real *dst;
    size_t spitch, dpitch, cnt;   /* the row pitch of src and of dst */
    int nrows, R, L;
    vfft_il2p_fn fn;
    const vfft_real *tab;
    size_t d0, nd;
} _il2d_dmt_arg;

static void _il2d_dmt_tramp(void *v)
{
    _il2d_dmt_arg *a = (_il2d_dmt_arg *)v;
    const int D = a->L / a->R;
    int b;
    if (!a->fn)
    {   /* the Rader stage: this worker's digits through the engine (its tag is the table) */
        _il2d_rd0_hook((const void *)a->tab, a->src, a->spitch, a->dst, a->dpitch, 0, 0, a->cnt, a->d0, a->nd);
        return;
    }
    for (b = 0; b < a->nrows / a->L; b++)
    {
        if (a->spitch == a->dpitch)
        {
            const size_t off =
                2 * ((size_t)b * a->L * a->spitch + a->d0 * a->spitch);
            a->fn(a->src + off, NULL, a->dst + off, NULL,
                  a->tab + a->d0 * (size_t)(a->R - 1) * VFFT_IL_T2CREC, NULL,
                  (size_t)D * a->spitch, a->spitch, (size_t)D * a->spitch,
                  a->nd, a->cnt);
        }
        else
        {   /* two pitches: the kernel advances input and output by one pitch a digit, so one call a digit
             * (il2d_cols.h _il2d_col_stage_x) */
            size_t d;
            for (d = a->d0; d < a->d0 + a->nd; d++)
            {
                const size_t row = (size_t)b * a->L + d;
                a->fn(a->src + 2 * row * a->spitch, NULL, a->dst + 2 * row * a->dpitch, NULL,
                      a->tab + d * (size_t)(a->R - 1) * VFFT_IL_T2CREC, NULL,
                      (size_t)D * a->spitch, a->spitch, (size_t)D * a->dpitch, 1, a->cnt);
            }
        }
    }
}

/* Run ONE stage with its digits split across T workers, src at row pitch
 * spitch and dst at dpitch. Returns 1 when it threaded, 0 when the caller
 * must run the stage serially. */
static int _il2d_stage_digits_mt_x(const vfft_real *src, size_t spitch, vfft_real *dst, size_t dpitch,
                                   int nrows, size_t cnt, int R, int L, vfft_il2p_fn fn,
                                   const vfft_real *tab, int T)
{
    const size_t D = (size_t)(L / R);
    _il2d_dmt_arg a[THREAD_POOL_MAX_DISPATCH];
    int t;
    if (!tab || D < (size_t)T || T < 2)
        return 0; /* D == 1 stages carry no table and no digit axis */
    for (t = 0; t < T; t++)
    {
        a[t].src = src; a[t].dst = dst;
        a[t].spitch = spitch; a[t].dpitch = dpitch; a[t].cnt = cnt;
        a[t].nrows = nrows; a[t].R = R; a[t].L = L;
        a[t].fn = fn; a[t].tab = tab;
        a[t].d0 = D * (size_t)t / (size_t)T;
        a[t].nd = D * (size_t)(t + 1) / (size_t)T - a[t].d0;
    }
    thread_pool_run(T, _il2d_dmt_tramp, a, sizeof a[0]); /* caller = a[0] */
    return 1;
}
static int _il2d_stage_digits_mt(const vfft_real *src, vfft_real *dst,
                                 int nrows, size_t pitch, size_t cnt,
                                 int R, int L, vfft_il2p_fn fn,
                                 const vfft_real *tab, int T)
{
    return _il2d_stage_digits_mt_x(src, pitch, dst, pitch, nrows, cnt, R, L, fn, tab, T);
}

/* the dispatched unit's body; the trampoline enters it at the column plan's
 * pass state when the plan is bound (every worker at the one state, as the
 * serial pass is entered), directly otherwise */
static void _il2d_cmt_body(const void *v, const vfft_real *src, vfft_real *dst)
{
    const _il2d_cmt_arg *a = (const _il2d_cmt_arg *)v;
    const vfft_ilcol_t *c = a->c;
    const size_t hp1 = c->rn;   /* the plane's pitch and lane count (door 2 past N2/2 + 1) */
    vfft_il2p_fn const *fns = a->reverse ? c->b : c->f;
    vfft_real *const *tabs = a->reverse ? c->tb : c->tf;
    if (a->natleaf)
    {   /* natural x MT: the leaf scatter/gather over [lo,hi) blocks -- the forward
         * scatter through this worker's own staging slot when the pass is staged
         * (bitwise the strided leaf: the same values in the same order); the reverse
         * leaf gathers at its stride (the c2r column plan has no staged form: the
         * gather is cheap at the plane's odd pitch) */
        const int nst = c->nst, Rl = c->R[nst - 1];
        _il2d_nat_leaf_range(src, dst, a->N1, hp1, hp1, Rl, fns[nst - 1], c->natperm, a->lo, a->hi, a->reverse,
                             a->stage ? _il2d_stage_slot(a->stage, a->tid, Rl, hp1) : NULL);
        return;
    }
    if (a->strip == 3)
    {   /* THE TILE (natural, mtarm = 2): the first stage's sub-problems [lo, hi), each D0 = N1 / R0 rows of the
         * scratch, this worker's alone -- forward: the middle stages in place on the tile, then its leaf blocks
         * (scratch -> dst); reverse: its leaf blocks gathered (src -> scratch), then the middle stages reversed
         * in place. _il2d_cols_mt_desc runs the first stage across the plane before (forward) or after (reverse) */
        const int nst = c->nst, R0 = c->R[0], Rl = c->R[nst - 1];
        const size_t D0 = (size_t)a->N1 / (size_t)R0;
        vfft_real *slot = a->stage ? _il2d_stage_slot(a->stage, a->tid, Rl, hp1) : NULL;
        size_t t;
        for (t = a->lo; t < a->hi; t++)
        {
            if (!a->reverse)
            {
                vfft_real *tile = (vfft_real *)src + 2 * t * D0 * hp1;   /* the scratch: writable */
                if (nst > 2)
                    _il2d_col_stages2(tile, tile, (int)D0, hp1, hp1, 1, nst - 1, c->R, c->L, fns, tabs, 0);
                _il2d_nat_leaf_range(src, dst, a->N1, hp1, hp1, Rl, fns[nst - 1], c->natperm, t * D0 / (size_t)Rl,
                                     (t + 1) * D0 / (size_t)Rl, 0, slot);
            }
            else
            {
                vfft_real *tile = dst + 2 * t * D0 * hp1;
                _il2d_nat_leaf_range(src, dst, a->N1, hp1, hp1, Rl, fns[nst - 1], c->natperm, t * D0 / (size_t)Rl,
                                     (t + 1) * D0 / (size_t)Rl, 1, slot);
                if (nst > 2)
                    _il2d_col_stages2(tile, tile, (int)D0, hp1, hp1, 1, nst - 1, c->R, c->L, fns, tabs, 1);
            }
        }
        return;
    }
    if (a->strip == 2)
    {   /* THE COLUMN BLOCKS (2026-10-08): the whole chain over this worker's columns
         * [lo, hi) in sub-strips of msw (0 = the whole range) -- a natural chain through the worker's
         * dense strip scratch (pitch = the strip width: the chain in L2, the plane read once and
         * written once; the shared pre-leaf plane where the width exceeds the scratch), a scrambled
         * chain through the same scratch where the plan built one (the lane-major batches,
         * _lm_sscr_build), else a scrambled or one-stage chain in place on the plane */
        const size_t sw = c->msw > 0 ? (size_t)c->msw : a->hi - a->lo;
        const int dense = c->nat && c->natdense && c->natsscr && a->tid < c->nnatsscr;
        size_t k;
        for (k = a->lo; k < a->hi; k += sw)
        {
            const size_t w = (k + sw < a->hi) ? sw : a->hi - k;
            if (!c->nat)
            {
                if (c->natdense && c->natsscr && a->tid < c->nnatsscr && w <= (size_t)c->natswcap)
                    _il2d_col_pass_scr_strip(src, dst, a->N1, hp1, k, w, c->nst, c->R, c->L, fns, tabs, a->reverse,
                                             c->natsscr[a->tid > 0 ? a->tid : 0]);
                else
                    _il2d_col_pass_range(src, dst, a->N1, hp1, k, k + w, c->nst, c->R, c->L, fns, tabs, a->reverse);
            }
            else if (dense && w <= (size_t)c->natswcap)
                _il2d_col_pass_nat_strip(src, dst, a->N1, hp1, k, w, c->nst, c->R, c->L, fns, tabs, a->reverse,
                                         c->natperm, c->natsscr[a->tid > 0 ? a->tid : 0]);
            else
                _il2d_col_pass_nat_range(src, dst, a->N1, hp1, k, k + w, c->nst, c->R, c->L, fns, tabs, a->reverse,
                                         c->natperm, c->natscr,
                                         a->stage ? _il2d_stage_slot(a->stage, a->tid, c->R[c->nst - 1], hp1) : NULL);
        }
        return;
    }
    if (a->strip)
    {
        if (c->blu)
        {   /* Bluestein column axis: the window pipeline */
            _il2d_blu_cols_range(src, dst, a->N1, hp1, a->lo, a->hi,
                                 c->blu, c->nst, c->R, c->L, c->f, c->b, c->tf, c->tb,
                                 a->reverse ? c->bluchb : c->bluchf,
                                 a->reverse ? c->blukb : c->blukf,
                                 c->bluscr);
            return;
        }
        _il2d_col_pass_range(src, dst, a->N1, hp1, a->lo, a->hi,
                             c->nst, c->R, c->L, fns, tabs, a->reverse);
        return;
    }
    {
        const size_t wl = (size_t)c->wl;
        size_t b;
        for (b = a->lo; b < a->hi; b++)
        {
            const size_t b0 = b * wl;
            const vfft_real *bs = src + 2 * b0 * hp1;
            _il2d_col_stages(bs, dst + 2 * b0 * hp1, (int)wl, hp1,
                             c->cut, c->nst, c->R, c->L, fns, tabs, a->reverse);
        }
    }
}
static void _il2d_cmt_tramp(void *v)
{
    _il2d_cmt_arg *a = (_il2d_cmt_arg *)v;
    if (a->stk >= 0)
        _il2d_rowx_call(_il2d_cmt_body, a, a->src, a->dst, a->stk);
    else
        _il2d_cmt_body(a, a->src, a->dst);
}

/* THE TILE SERVES (mtarm = 2, _il2d_cols_mt_desc): a natural chain of two stages or more whose first stage is no
 * Rader stage, on an axis with no Bluestein and no turned prime pass, the natural leaf's scratch built. The
 * lane-major door races it, and the estimate names it, as the threaded form of such a pass. */
static inline int _il2d_col_tile_serves(const vfft_ilcol_t *c)
{
    return c->nat && c->nst >= 2 && !c->blu && !c->tpc && c->f[0] && c->b[0] && c->natscr;
}

/* Returns 1 when it ran threaded, 0 when the caller must run serial. */
/* THE THREADED COLUMN PASS over a column descriptor c (N1 rows): the plan's own (il2d_col,
 * through _il2d_real_cols_mt) or a chain race candidate's view at T > 1 (_il2d_arm_chain_mt), so
 * the race times the pass the cell serves with. stage = the natural leaf's staging slots (T of
 * them), NULL = the strided leaf; stk = the column plan's pass state (-1 = none). 1 = it ran
 * threaded, 0 = the caller runs the serial pass. */
static int _il2d_cols_mt_desc(const vfft_ilcol_t *c, int N1, const vfft_real *src, vfft_real *dst, int reverse, int T,
                              vfft_real *stage, int stk)
{
    const size_t hp1 = c->rn;   /* the plane's pitch and lane count (door 2 past N2/2 + 1) */
    const int strip = (c->wl <= 0);
    size_t units = strip ? hp1 : ((size_t)N1 / (size_t)c->wl);
    _il2d_cmt_arg a[THREAD_POOL_MAX_DISPATCH];
    int t;
    /* T arrives as the plan's snapshot (the plan's nthreads); the pool's one clamp
     * bounds it by the live pool and the arg-array size. */
    T = thread_pool_workers_for(T);
    if (T >= 2 && c->tpc)
        return _il2d_tpc_cols_mt(c, src, dst, hp1, reverse, T);   /* the turned prime pass: every worker its lanes */
    if (T >= 2 && c->natarm == 1 && !c->blu && !c->tpc && (!c->nat || c->nst >= 2))
    {   /* mtarm = 1: the column blocks -- every worker the whole chain over its columns (the
         * sub-strip arm of the body); every class but the Bluestein and turned axes */
        const int Ts = hp1 < (size_t)T ? (int)hp1 : T;
        if (Ts < 2)
            return 0;
        for (t = 0; t < Ts; t++)
        {
            a[t].c = c; a[t].N1 = N1; a[t].stage = stage;
            a[t].src = src;
            a[t].dst = dst;
            a[t].reverse = reverse;
            a[t].strip = 2;
            a[t].natleaf = 0;
            a[t].tid = t;
            a[t].stk = stk;
            a[t].lo = hp1 * (size_t)t / (size_t)Ts;
            a[t].hi = hp1 * (size_t)(t + 1) / (size_t)Ts;
        }
        thread_pool_run(Ts, _il2d_cmt_tramp, a, sizeof a[0]); /* caller = a[0] */
        return 1;
    }
    if (T >= 2 && c->natarm == 2 && _il2d_col_tile_serves(c))
    {   /* mtarm = 2: THE TILE (2026-11-17; the 2D tier's tile, _il2d_c2c_mt, without its row transform): the first
         * stage across the plane (src -> scratch; its digit is the row within a sub-problem), then every
         * sub-problem, N1 / R0 rows of the scratch, one worker's: its middle stages and its leaf blocks; reverse
         * mirrored. The plane crosses the cores once each way where the walk crosses them at every stage: the
         * threaded form of a long chain over few lanes (a lane-major batch of K = 8) */
        const int R0 = c->R[0], Rl = c->R[c->nst - 1];
        const size_t D0 = (size_t)N1 / (size_t)R0;
        const int Tt = R0 < T ? R0 : T;
        vfft_real *scr = c->natscr;
        if (R0 >= 2 && Tt >= 2 && D0 % (size_t)Rl == 0)
        {
            if (!reverse && !_il2d_stage_digits_mt(src, scr, N1, hp1, hp1, R0, c->L[0], c->f[0], c->tf[0], T))
                _il2d_col_stages(src, scr, N1, hp1, 0, 1, c->R, c->L, c->f, c->tf, 0);
            for (t = 0; t < Tt; t++)
            {
                a[t].c = c; a[t].N1 = N1; a[t].stage = stage;
                a[t].src = reverse ? src : scr;
                a[t].dst = reverse ? scr : dst;
                a[t].reverse = reverse;
                a[t].strip = 3;
                a[t].natleaf = 0;
                a[t].tid = t;
                a[t].stk = stk;
                a[t].lo = (size_t)R0 * (size_t)t / (size_t)Tt;
                a[t].hi = (size_t)R0 * (size_t)(t + 1) / (size_t)Tt;
            }
            thread_pool_run(Tt, _il2d_cmt_tramp, a, sizeof a[0]);   /* caller = a[0] */
            if (reverse && !_il2d_stage_digits_mt(scr, dst, N1, hp1, hp1, R0, c->L[0], c->b[0], c->tb[0], T))
                _il2d_col_stages(scr, dst, N1, hp1, 0, 1, c->R, c->L, c->b, c->tb, 0);
            return 1;
        }
    }
    if (T >= 2 && c->nat)
    {
        /* NATURAL x MT: the matched partition of the
         * natural pass — prefix stages digit-split (src -> scratch, then
         * in place), the leaf scatter by BLOCK RANGE (scratch -> dst),
         * mirrored for bwd (gather first, reversed prefix after, stage 0
         * scratch -> dst). No band arm: the scatter crosses bands. */
        const int Rl = c->R[c->nst - 1];
        const size_t nb = (size_t)N1 / (size_t)Rl;
        const int Tb = nb < (size_t)T ? (int)nb : T;
        vfft_real *scr = c->natscr;
        int s;
        if (Tb < 2 || c->nst < 2)
            return 0;
        if (!reverse)
        {
            for (s = 0; s < c->nst - 1; s++)
            {
                const vfft_real *ssrc = (s == 0) ? src : scr;
                if (!_il2d_stage_digits_mt(ssrc, scr, N1, hp1, hp1,
                                           c->R[s], c->L[s],
                                           c->f[s], c->tf[s], T))
                    _il2d_col_stages(ssrc, scr, N1, hp1, s, s + 1,
                                     c->R, c->L, c->f,
                                     c->tf, 0);
            }
        }
        for (t = 0; t < Tb; t++)
        {
            a[t].c = c; a[t].N1 = N1; a[t].stage = stage;
            a[t].src = reverse ? src : scr;
            a[t].dst = reverse ? scr : dst;
            a[t].reverse = reverse;
            a[t].strip = 0;
            a[t].natleaf = 1;
            a[t].tid = t;
            a[t].stk = stk;
            a[t].lo = nb * (size_t)t / (size_t)Tb;
            a[t].hi = nb * (size_t)(t + 1) / (size_t)Tb;
        }
        thread_pool_run(Tb, _il2d_cmt_tramp, a, sizeof a[0]);
        if (reverse)
        {
            for (s = c->nst - 2; s >= 0; s--)
            {
                vfft_real *out = (s == 0) ? dst : scr;
                if (!_il2d_stage_digits_mt(scr, out, N1, hp1, hp1,
                                           c->R[s], c->L[s],
                                           c->b[s], c->tb[s], T))
                    _il2d_col_stages(scr, out, N1, hp1, s, s + 1,
                                     c->R, c->L, c->b,
                                     c->tb, 0);
            }
        }
        return 1;
    }
    if (T < 2 || units < (size_t)T)
        return 0; /* not enough independent units to be worth splitting */
    /* fwd: the wide prefix must complete before ANY band (stage 0's legs
     * span the whole plane). bwd: the reversed prefix runs after. */
    if (!strip && !reverse && c->cut > 0)
    {
        /* each prefix stage's DIGITS split over the workers
         * (whole rows, count untouched); stages stay ordered, one
         * dispatch each — a dispatch+wait is ~100 ns, so the
         * per-stage join is free at these sizes. A stage that cannot
         * split (D < T, or the table-free D==1 leaf) runs serial. */
        int s;
        for (s = 0; s < c->cut; s++)
        {
            const vfft_real *ssrc = (s == 0) ? src : dst;
            if (!_il2d_stage_digits_mt(ssrc, dst, N1, hp1, hp1,
                                       c->R[s], c->L[s],
                                       c->f[s], c->tf[s], T))
                _il2d_col_stages(ssrc, dst, N1, hp1, s, s + 1,
                                 c->R, c->L, c->f,
                                 c->tf, 0);
        }
    }
    for (t = 0; t < T; t++)
    {
        a[t].c = c; a[t].N1 = N1; a[t].stage = stage;
        /* after a fwd prefix the band source IS dst (in place) */
        a[t].src = (!strip && !reverse && c->cut > 0) ? dst : src;
        a[t].dst = dst;
        a[t].reverse = reverse;
        a[t].strip = strip;
        a[t].natleaf = 0;
        a[t].tid = t;
        a[t].stk = stk;
        a[t].lo = units * (size_t)t / (size_t)T;
        a[t].hi = units * (size_t)(t + 1) / (size_t)T;
    }
    thread_pool_run(T, _il2d_cmt_tramp, a, sizeof a[0]); /* caller = a[0] */
    if (!strip && reverse && c->cut > 0)
    {
        /* the Hermitian-transpose chain: prefix stages in REVERSE order,
         * in place on dst, each digit-split the same way. */
        int s;
        for (s = c->cut - 1; s >= 0; s--)
            if (!_il2d_stage_digits_mt(dst, dst, N1, hp1, hp1,
                                       c->R[s], c->L[s],
                                       c->b[s], c->tb[s], T))
                _il2d_col_stages(dst, dst, N1, hp1, s, s + 1,
                                 c->R, c->L, c->b,
                                 c->tb, 0);
    }
    return 1;
}
/* the serving entry: the plan's descriptor, its column plan's pass state and staged leaf; counts
 * the engagement (vfft_il2d_col_mt_passes) */
static int _il2d_real_cols_mt(struct vfft_plan_s *h, const vfft_real *src,
                              vfft_real *dst, int reverse, int T)
{
    const int stk = (h->il2d_cx_on && !h->il2d_cx_perk) ? h->il2d_cx_stk : -1;
    vfft_real *stage = (!reverse && h->il2d_cx_on && h->il2d_cx_st) ? _il2d_nat_stage_of(h, 0) : NULL;
    if (!_il2d_cols_mt_desc(&h->il2d_col, h->N, src, dst, reverse, T, stage, stk))
        return 0;
    _vfft_il2d_col_mt_count++; /* engagement, see vfft.h */
    return 1;
}


/* ══ c2c MT (the real tier's design, ported) ═════════════════════════
 * The structural difference from real: rows COMMUTE with column stages
 * (both C-linear — the same fact that makes tfuse legal here and banned
 * for real by fft2d_real_il_design.md §2.5). So a banded cell's unit of work is a SELF-CONTAINED
 * band [suffix stages + its own fused rows]: partition bands across
 * workers and there is no rows/columns wall and no cross-core exchange
 * for the fused part. Only the wide prefix needs the digit split.
 * Row execution mutates shared plan state (one child), so a worker t > 0
 * runs its CLONE (il2d_roww[t-1], route-equivalence-checked at build).
 * Every arm is
 * a loop restriction of the serving walk => MT == ST bitwise. */
static void _il2d_row_exec_t(struct vfft_plan_s *h, int tid,
                             vfft_dir_t dir, vfft_real *row, size_t rn, size_t p)
{
    if (tid <= 0)
    {
        _il2d_row_exec(h, dir, row, rn, p);
        return;
    }
    {
        struct vfft_plan_s *c = h->il2d_roww[tid - 1];
        const vfft_real *tw = _il2d_fs_rec(h, rn, p);
        if (tw && dir == VFFT_FORWARD)
            _il2d_fs_twiddle(row, tw, h->il2d_fs_B, rn, 0);
        _vfft_exec6((vfft_plan)c, dir, row, NULL, row, NULL);
        if (tw && dir != VFFT_FORWARD)
            _il2d_fs_twiddle(row, tw, h->il2d_fs_B, rn, 1);
    }
}

/* the banked ROW-ROUTE value of a plan: 0 = the in-place child, 2 = the
 * batched rows, 3 = the batched two-pass rows (the ro= token) */
static int _il2d_ro_of(const struct vfft_plan_s *h)
{
    return h->il2d_rowb2 ? 3 : h->il2d_rowb ? 2 : 0;
}

/* the ROW-LOOP twin of a two-pass stage kernel: the same body
 * with the in-kernel row loop (count = rows x Ls lanes, in pitch Gs, out pitch
 * OGs); NULL where the corpus has none -- then the two-pass rows have no arm */
#define VFFT_IL2D_RB2_CHUNK 64   /* the tile's ceiling in rows: the per-worker scratch is CHUNK x N2 */
/* the two-pass rows' TILE: a RACED axis like ZTURN-T's tile --
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
#ifdef VFFT_FLOAT
    return 0;   /* the row-loop twins are FP64's: the FP32 corpus carries none */
#else
#define T_(a, b) if (f == (vfft_il2p_fn)a) return b;
    T_(VFFT_IL_SYM(radix4_z_n1t_fwd), VFFT_IL_SYM(radix4_z_n1tr_fwd))
    T_(VFFT_IL_SYM(radix8_z_n1t_fwd), VFFT_IL_SYM(radix8_z_n1tr_fwd))
    T_(VFFT_IL_SYM(radix16_z_n1t_fwd), VFFT_IL_SYM(radix16_z_n1tr_fwd))
    T_(VFFT_IL_SYM(radix4_z_t2_fwd), VFFT_IL_SYM(radix4_z_t2r_fwd))
    T_(VFFT_IL_SYM(radix8_z_t2_fwd), VFFT_IL_SYM(radix8_z_t2r_fwd))
    T_(VFFT_IL_SYM(radix16_z_t2_fwd), VFFT_IL_SYM(radix16_z_t2r_fwd))
    T_(VFFT_IL_SYM(radix4_z_t2t_bwd), VFFT_IL_SYM(radix4_z_t2tr_bwd))
    T_(VFFT_IL_SYM(radix8_z_t2t_bwd), VFFT_IL_SYM(radix8_z_t2tr_bwd))
    T_(VFFT_IL_SYM(radix16_z_t2t_bwd), VFFT_IL_SYM(radix16_z_t2tr_bwd))
    T_(VFFT_IL_SYM(radix4_z_n1_bwd), VFFT_IL_SYM(radix4_z_n1r_bwd))
    T_(VFFT_IL_SYM(radix8_z_n1_bwd), VFFT_IL_SYM(radix8_z_n1r_bwd))
    T_(VFFT_IL_SYM(radix16_z_n1_bwd), VFFT_IL_SYM(radix16_z_n1r_bwd))
    T_(VFFT_IL_SYM(radix8_z_n1ttan_fwd), VFFT_IL_SYM(radix8_z_n1trtan_fwd))
    T_(VFFT_IL_SYM(radix16_z_n1ttan_fwd), VFFT_IL_SYM(radix16_z_n1trtan_fwd))
    T_(VFFT_IL_SYM(radix8_z_t2tan_fwd), VFFT_IL_SYM(radix8_z_t2rtan_fwd))
    T_(VFFT_IL_SYM(radix16_z_t2tan_fwd), VFFT_IL_SYM(radix16_z_t2rtan_fwd))
    T_(VFFT_IL_SYM(radix8_z_t2ttan_bwd), VFFT_IL_SYM(radix8_z_t2trtan_bwd))
    T_(VFFT_IL_SYM(radix16_z_t2ttan_bwd), VFFT_IL_SYM(radix16_z_t2trtan_bwd))
    T_(VFFT_IL_SYM(radix8_z_n1tan_bwd), VFFT_IL_SYM(radix8_z_n1rtan_bwd))
    T_(VFFT_IL_SYM(radix16_z_n1tan_bwd), VFFT_IL_SYM(radix16_z_n1rtan_bwd))
#undef T_
    return 0;
#endif
}

/* the row pass over a RUN of rows: nrows rows, row i at
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
                             const vfft_real *in, size_t pin, vfft_real *out, size_t pout,
                             size_t rn, size_t p0, size_t pstep, size_t nrows);
static void _il2d_rows_exec(struct vfft_plan_s *h, int tid, vfft_dir_t dir,
                            vfft_real *base, size_t rn, size_t pitch,
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
                             const vfft_real *in, size_t pin, vfft_real *out, size_t pout,
                             size_t rn, size_t p0, size_t pstep, size_t nrows)
{
    size_t i;
    const int inplace = (in == out);
    vfft_real *base = out;
    const size_t pitch = pout;
    if (h->il2d_rowb2)
    {   /* route 3: the batched TWO-PASS rows -- per chunk, stage 1 rows ->
         * the worker's scratch (rows at pitch rn), stage 2 scratch -> rows;
         * backward t2t then n1 the same way; the four-step's hook per row
         * around the run as route 2 */
        const vfft_il2p_plan_t *p = h->il2d_row->k1il2p;
        const size_t R1 = (size_t)p->R1, R2 = (size_t)p->R2;
        vfft_real *scr = h->il2d_rowb2_scr + (size_t)(tid > 0 ? tid : 0) * 2 * (size_t)VFFT_IL2D_RB2_CHUNK * rn;
        const int fwd = (dir == VFFT_FORWARD);
        /* the tile: the raced rbk= (a 64-row chunk at N2 = 32/64 spilled L1 and
         * lost to the per-row child; 16 rows at 32 won 1.5x) */
        const size_t ch = h->il2d_rowb2_ch > 0 ? (size_t)h->il2d_rowb2_ch : 2;
        size_t i0;
        if (h->il2d_fs_tw && fwd)
            for (i = 0; i < nrows; i++)
            {
                const vfft_real *tw = _il2d_fs_rec(h, rn, p0 + i * pstep);
                if (tw)
                    _il2d_fs_twiddle(base + 2 * i * pitch, tw, h->il2d_fs_B, rn, 0);
            }
        for (i0 = 0; i0 < nrows; i0 += ch)
        {
            const size_t n = (nrows - i0 < ch) ? nrows - i0 : ch;
            const vfft_real *bi = in + 2 * i0 * pin;
            vfft_real *b = base + 2 * i0 * pitch;
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
                const vfft_real *tw = _il2d_fs_rec(h, rn, p0 + i * pstep);
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
                const vfft_real *tw = _il2d_fs_rec(h, rn, p0 + i * pstep);
                if (tw)
                    _il2d_fs_twiddle(base + 2 * i * pitch, tw, h->il2d_fs_B, rn, 0);
            }
        (fwd ? h->il2d_rowb_f : h->il2d_rowb_b)(in, NULL, base, NULL, NULL, NULL,
                                                 1, pin, 1, pitch, nrows);
        if (h->il2d_fs_tw && !fwd)
            for (i = 0; i < nrows; i++)
            {
                const vfft_real *tw = _il2d_fs_rec(h, rn, p0 + i * pstep);
                if (tw)
                    _il2d_fs_twiddle(base + 2 * i * pitch, tw, h->il2d_fs_B, rn, 1);
            }
        return;
    }
    if (!inplace)
    {   /* the per-row route from the scratch: the out-of-place K=1 plan at N2
         * (worker t > 0 runs its clone: the threaded skewed pass) */
        struct vfft_plan_s *rp = (tid > 0) ? h->il2d_cskw[tid - 1] : h->il2d_csk_row;
        for (i = 0; i < nrows; i++)
            _vfft_exec6((vfft_plan)rp, dir, (vfft_real *)(in + 2 * i * pin), NULL, base + 2 * i * pitch, NULL);
        return;
    }
    for (i = 0; i < nrows; i++)
        _il2d_row_exec_t(h, tid, dir, base + 2 * i * pitch, rn, p0 + i * pstep);
}

/* the skewed column pass's pitch in complex: N2 + 8 -- the R output streams of
 * the column stage never sit a multiple of 4 KB apart */
#define VFFT_IL2D_CSK_PITCH(N2) ((size_t)(N2) + 8)
/* the SKEWED column pass's execute: the single column stage from
 * the plane (leg stride N2) into the scratch (leg stride N2 + 8), then the rows
 * from the scratch into the destination -- the rows are the move. Both
 * directions (the passes commute), both placements (the plane is read whole
 * before the rows write it). */
static void _il2d_csk_exec(struct vfft_plan_s *h, vfft_dir_t dir, const vfft_real *sre, vfft_real *dre)
{
    const size_t N1 = (size_t)h->N, rn = (size_t)h->N2, P = VFFT_IL2D_CSK_PITCH(rn);
    const int fwd = (dir == VFFT_FORWARD);
    vfft_real *T = h->il2d_csk_scr;
    (fwd ? h->il2d_csk_f : h->il2d_csk_b)(sre, NULL, T, NULL, NULL, NULL, rn, 0, P, 0, rn);
    _il2d_rows_exec2(h, 0, dir, T, P, dre, rn, rn, 0, 1, N1);
}

/* THE ROWS-FIRST WALK's execute (2026-11-24; vfft_internal.h il2d_rfw, the evidence at policy_il.h
 * vfft_policy_il2d_rowsfirst_widths): the rows from the source into the destination (the rows are the move; an
 * in-place plan runs them where they are), then the natural column chain IN PLACE on the destination, one strip of
 * il2d_rfw columns after another through the dense strip scratch -- the threaded strips' walker
 * (_il2d_col_pass_nat_strip) with source = destination, in place by construction (a strip is read whole into the
 * scratch before any row of it is written). Both directions: the passes commute, and the walker's backward is the
 * strips arm's. One thread (the scratch is one block). */
static void _il2d_rfw_exec(struct vfft_plan_s *h, vfft_dir_t dir, const vfft_real *sre, vfft_real *dre)
{
    const vfft_ilcol_t *c = &h->il2d_col;
    const size_t rn = (size_t)h->N2, w = (size_t)h->il2d_rfw;
    const int fwd = (dir == VFFT_FORWARD);
    size_t k;
    _il2d_rows_exec2(h, 0, dir, sre, rn, dre, rn, rn, 0, 1, (size_t)h->N);
    for (k = 0; k < rn; k += w)
        _il2d_col_pass_nat_strip(dre, dre, c->N, rn, k, (rn - k < w) ? rn - k : w, c->nst, c->R, c->L,
                                 fwd ? c->f : c->b, fwd ? c->tf : c->tb, !fwd, c->natperm, c->natsscr[0]);
}
/* the plan can run that walk: a natural chain of two stages or more at one thread, off the turn, the skewed pass and
 * the prime-column routes, with a way to move its rows (in place; out of place the batched kernels or the
 * out-of-place row plan) */
static int _il2d_rfw_ok(const struct vfft_plan_s *h)
{
    return h->transform == VFFT_C2C && h->nthreads <= 1 && h->il2d_row && h->il2d_col.nat && h->il2d_col.nst >= 2 &&
           h->il2d_col.natperm && !h->il2d_col.blu && !h->il2d_turn && !h->il2d_csk &&
           (h->placement == VFFT_INPLACE || h->il2d_rowb || h->il2d_rowb2 || h->il2d_csk_row);
}

/* the scratch's row PITCH in complex: N1 + 8, so the N2 column streams never sit
 * a multiple of 4 KB apart (unskewed, at every N1 >= 256 they share one L1 set,
 * 12-way, for 8 or 16 streams -- the route's margin shrank from 2.3x at N2 = 2
 * to nothing at 16) */
#define VFFT_IL2D_TURN_PITCH(N1) ((size_t)(N1) + 8)
/* the TURN route's back-turn: the N2 x N1 scratch T (row c = column c of the
 * plane, N1 long) into the N1 x N2 plane dst, 2 x 2 complex blocks through one
 * 128-bit lane permute (at 512 bits 4 x 4 blocks first, _il2d_turn4), the row
 * range blocked so a block of dst rows stays in
 * L1 while every column pair lands in it (N2 is small where this route wins:
 * dst rows are 32..128 B and would otherwise be re-fetched once per column
 * pair). Odd N1 / odd N2 finish scalar. */
static void _il2d_turn_back_range(const vfft_real *T, size_t P, vfft_real *dst, size_t N2, size_t ra, size_t rb)
{
    const size_t RB = 256;
    size_t r0, c, r;
    for (r0 = ra; r0 < rb; r0 += RB)
    {
        const size_t r1 = (r0 + RB < rb) ? r0 + RB : rb;
        c = 0;
#ifdef VFFT_IL2D_TURN4
        for (; c + 4 <= N2; c += 4)
        {   /* four columns a step (512 bits, _il2d_turn4); the rows a 4 x 4 block does not fill go point by point */
            const vfft_real *t0 = T + 2 * (c * P), *t1 = T + 2 * ((c + 1) * P), *t2 = T + 2 * ((c + 2) * P),
                            *t3 = T + 2 * ((c + 3) * P);
            for (r = r0; r + 4 <= r1; r += 4)
            {
                __m512d o0, o1, o2, o3;
                _il2d_turn4(_mm512_loadu_pd(t0 + 2 * r), _mm512_loadu_pd(t1 + 2 * r), _mm512_loadu_pd(t2 + 2 * r),
                            _mm512_loadu_pd(t3 + 2 * r), &o0, &o1, &o2, &o3);
                _mm512_storeu_pd(dst + 2 * (r * N2 + c), o0);
                _mm512_storeu_pd(dst + 2 * ((r + 1) * N2 + c), o1);
                _mm512_storeu_pd(dst + 2 * ((r + 2) * N2 + c), o2);
                _mm512_storeu_pd(dst + 2 * ((r + 3) * N2 + c), o3);
            }
            for (; r < r1; r++)
            {
                _mm_storeu_pd(dst + 2 * (r * N2 + c), _mm_loadu_pd(t0 + 2 * r));
                _mm_storeu_pd(dst + 2 * (r * N2 + c + 1), _mm_loadu_pd(t1 + 2 * r));
                _mm_storeu_pd(dst + 2 * (r * N2 + c + 2), _mm_loadu_pd(t2 + 2 * r));
                _mm_storeu_pd(dst + 2 * (r * N2 + c + 3), _mm_loadu_pd(t3 + 2 * r));
            }
        }
#endif
        for (; c + 2 <= N2; c += 2)
        {
            const vfft_real *ta = T + 2 * (c * P), *tb = T + 2 * ((c + 1) * P);
            for (r = r0; r + 2 <= r1; r += 2)
            {
#ifdef VFFT_FLOAT
                {   /* the turn back: [ta r, ta r+1], [tb r, tb r+1] -> rows r and r+1 at lanes c, c+1 */
                    float tmp[8];
                    vfftf__turn2(tmp, tmp + 4, ta + 2 * r, tb + 2 * r);
                    vfftf__cx1(dst + 2 * (r * N2 + c), tmp);
                    vfftf__cx1(dst + 2 * (r * N2 + c + 1), tmp + 2);
                    vfftf__cx1(dst + 2 * ((r + 1) * N2 + c), tmp + 4);
                    vfftf__cx1(dst + 2 * ((r + 1) * N2 + c + 1), tmp + 6);
                }
#else
                const __m256d a = _mm256_loadu_pd(ta + 2 * r), b = _mm256_loadu_pd(tb + 2 * r);
                _mm256_storeu_pd(dst + 2 * (r * N2 + c), _mm256_permute2f128_pd(a, b, 0x20));
                _mm256_storeu_pd(dst + 2 * ((r + 1) * N2 + c), _mm256_permute2f128_pd(a, b, 0x31));
#endif
            }
            if (r < r1)
            {
#ifdef VFFT_FLOAT
                vfftf__cx1(dst + 2 * (r * N2 + c), ta + 2 * r);
                vfftf__cx1(dst + 2 * (r * N2 + c + 1), tb + 2 * r);
#else
                _mm_storeu_pd(dst + 2 * (r * N2 + c), _mm_loadu_pd(ta + 2 * r));
                _mm_storeu_pd(dst + 2 * (r * N2 + c + 1), _mm_loadu_pd(tb + 2 * r));
#endif
            }
        }
        if (c < N2)
        {
            const vfft_real *ta = T + 2 * (c * P);
            for (r = r0; r < r1; r++)
#ifdef VFFT_FLOAT
                vfftf__cx1(dst + 2 * (r * N2 + c), ta + 2 * r);
#else
                _mm_storeu_pd(dst + 2 * (r * N2 + c), _mm_loadu_pd(ta + 2 * r));
#endif
        }
    }
}
static void _il2d_turn_back(const vfft_real *T, size_t P, vfft_real *dst, size_t N1, size_t N2)
{
    _il2d_turn_back_range(T, P, dst, N2, 0, N1);
}

/* the TURN route's execute: rows through the batched mono kernel with turned
 * stores (leg l of row k -> T[l][k]: Ls = 1, Gs = N2, OLs = the skewed pitch P, OGs = 1), the
 * N2 columns as rows of T through the in-place K=1 plan at N1, the back-turn.
 * Both directions (the 2D passes commute), both placements (T is private:
 * the plane is read whole before dst is written). */
static void _il2d_turn_exec(struct vfft_plan_s *h, vfft_dir_t dir, const vfft_real *sre, vfft_real *dre)
{
    const size_t N1 = (size_t)h->N, rn = (size_t)h->N2, P = VFFT_IL2D_TURN_PITCH(N1);
    vfft_real *T = h->il2d_turn_scr;
    size_t c;
    (dir == VFFT_FORWARD ? h->il2d_rowt_f : h->il2d_rowt_b)(sre, NULL, T, NULL, NULL, NULL, 1, rn, P, 1, N1);
    for (c = 0; c < rn; c++)
        _vfft_exec6((vfft_plan)h->il2d_turn_plan, dir, T + 2 * c * P, NULL, T + 2 * c * P, NULL);
    _il2d_turn_back(T, P, dre, N1, rn);
}

/* the natural leaf over [blo, bhi) blocks through worker tid's staging,
 * the block's rows FUSED there forward when the walk fuses rows (each
 * natural row leaves finished, one sequential stream); backward the leaf
 * gathers only — the backward's rows stay after the column pass, the
 * order every backward arm shares (il2d_natural_leaf_design.md) */
static vfft_real *_il2d_nat_stage_buf(struct vfft_plan_s *h, int tid)
{   /* worker tid's staging block, wherever the plan carries the buffer (every
     * natural plan: T slots, fft2d_create_il.h) */
    const int Rl = h->il2d_col.R[h->il2d_col.nst - 1];
    return h->il2d_col.natstage ? _il2d_stage_slot(h->il2d_col.natstage, tid, Rl, h->il2d_col.rn) : NULL;
}
/* the FORWARD leaf's staging: the buffer under the staged verdict (nls =
 * natst, raced against the strided leaf at create); NULL = strided */
static vfft_real *_il2d_nat_stage_of(struct vfft_plan_s *h, int tid)
{
    return h->il2d_col.natst ? _il2d_nat_stage_buf(h, tid) : NULL;
}
/* THE NATURAL BACKWARD THROUGH THE FORWARD'S WALK (2026-10-06). The inverse
 * DFT along the column axis is the forward DFT with its output index negated,
 * so a natural backward runs the forward's walk SHAPE -- the wide prefix with
 * the FORWARD kernels and tables, the band suffix, the forward leaf through
 * the staging with the BACKWARD row child fused after it, as the forward
 * fuses its rows -- and the staged scatter writes natural row i to row
 * (N1 - i) mod N1 (_il2d_nat_stage_out_rev). The same memory pattern as the
 * forward by construction. Wherever the leaf is staged, on the serial walks
 * (banded or not), the MT block arm and the MT tile arm (they apply the same
 * kernels at the same points: MT == ST stays bitwise). Not for a strips
 * verdict (a strip cannot carry whole rows: that arm, and the serial twin of
 * such a plan, keep the mirrored walk with rows last), nor for a four-step
 * child (its backward hook wants the rows before every column stage). The
 * mirrored walk -- the gather, the reversed suffix, the reversed prefix with
 * the Hermitian-transpose kernel set, rows last -- stays for those. Before
 * this the backward's rows were a separate sweep over the plane: 24% over
 * the forward with the same chain (8.88 vs 7.15 ms at 2048x1024); the
 * mirrored walk with its gather fused (a step on the way) still read the
 * plane as a bare copy at the DRAM roof, 8-15% over.
 * THE BACKWARD'S LEAF IS STAGED WHEREVER THE STAGING EXISTS (2026-10-06):
 * the strided-vs-staged verdict (natst) is the forward's -- at T = 8 the
 * natural race takes the tile arm with the strided leaf at 256x256 and
 * 512x256, and the mirrored backward of that plan cost 1.50-1.56x the
 * forward; through the staging it costs the forward (26.6 vs 34.4 us,
 * 55.7 vs 64.7). The buffer is every natural plan's, so the backward takes
 * it regardless of the verdict; the serial twin does the same, bitwise. */
static int _il2d_nat_bwd_fused(struct vfft_plan_s *h)
{
    return h->il2d_col.nat && h->il2d_col.nst >= 2 && h->il2d_col.natarm != 1 && !h->il2d_fs_tw &&
           _il2d_nat_stage_buf(h, 0) != NULL;
}
/* the natural leaf over [blo, bhi) blocks through worker tid's staging; fuse =
 * run the block's rows inside, after the leaf: forward, the block's rows
 * finished and streamed to their natural rows; backward (dir = BACKWARD with
 * fuse: the backward through the forward's walk), the FORWARD leaf from the
 * scratch comb, the backward rows, the block streamed to the rows
 * (N1 - i) mod N1. fuse = 0 backward is the mirrored walk's gather (natural
 * rows -> the scratch comb, no rows). Returns 1 when the rows ran inside, 0
 * when the caller must run them (the strided leaf, or the mirrored gather).
 * sp: the row pitch of the pre-leaf plane (from, or the mirrored gather's to)
 * as the caller's walk lays it, and of the staging here (rn, or the natural
 * pass's own pitch: il2d_col.h VFFT_IL2D_NAT_PITCH). */
static int _il2d_nat_leaf_blocks(struct vfft_plan_s *h, int tid, vfft_dir_t dir,
                                 const vfft_real *from, vfft_real *to, size_t blo, size_t bhi, int fuse, size_t sp)
{
    const int nst = h->il2d_col.nst, Rl = h->il2d_col.R[nst - 1];
    const size_t rn = h->il2d_col.rn, nstride_rows = (size_t)(h->N / Rl);
    const int fwd = (dir == VFFT_FORWARD);
    vfft_il2p_fn fn = fwd ? h->il2d_col.f[nst - 1] : h->il2d_col.b[nst - 1];
    /* the forward's staging is the verdict's; a fused backward takes the buffer wherever it exists */
    vfft_real *stage = (!fwd && fuse) ? _il2d_nat_stage_buf(h, tid) : _il2d_nat_stage_of(h, tid);
    size_t b;
    if (!stage || !fuse)
    {
        _il2d_nat_leaf_range(from, to, h->N, rn, sp, Rl, fn, h->il2d_col.natperm, blo, bhi, !fwd, stage);
        if (fwd && fuse)
        {
            for (b = blo; b < bhi; b++)
            {
                const size_t r0 = (size_t)h->il2d_col.natperm[b * (size_t)Rl];
                _il2d_rows_exec(h, tid, dir, to + 2 * r0 * rn, rn, nstride_rows * rn, r0, nstride_rows, (size_t)Rl);
            }
            return 1;
        }
        return 0;
    }
    if (!fwd)
    {   /* the backward through the forward's walk: the forward leaf from the
         * comb into the staging, the backward rows there, the block to the
         * rows (N1 - i) mod N1 */
        vfft_il2p_fn ff = h->il2d_col.f[nst - 1];
        for (b = blo; b < bhi; b++)
        {
            const size_t coff = 2 * b * (size_t)Rl * sp;
            const size_t nrow0 = (size_t)h->il2d_col.natperm[b * (size_t)Rl];
            ff(from + coff, NULL, stage, NULL, NULL, NULL, sp, 0, sp, 0, rn);
            _il2d_rows_exec(h, tid, dir, stage, rn, sp, nrow0, nstride_rows, (size_t)Rl);
            _il2d_nat_stage_out_rev(stage, to, nrow0, nstride_rows, Rl, rn, 0, rn, sp, !h->il2d_rows_cached, (size_t)h->N);
        }
        return 1;
    }
    for (b = blo; b < bhi; b++)
    {
        const size_t coff = 2 * b * (size_t)Rl * sp;
        const size_t nrow0 = (size_t)h->il2d_col.natperm[b * (size_t)Rl];
        fn(from + coff, NULL, stage, NULL, NULL, NULL, sp, 0, sp, 0, rn);
        _il2d_rows_exec(h, tid, dir, stage, rn, sp, nrow0, nstride_rows, (size_t)Rl);
        /* finished rows: streamed past the cache, or through it on a plane the caches hold (il2d_rows_cached) */
        _il2d_nat_stage_out(stage, to, nrow0, nstride_rows, Rl, rn, 0, rn, sp, !h->il2d_rows_cached);
    }
    return 1;
}

typedef struct
{
    struct vfft_plan_s *h;
    const vfft_real *src;
    vfft_real *dst;
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
    vfft_real *const *tabs = a->fwd ? h->il2d_col.tf : h->il2d_col.tb;
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
            const vfft_real *bs = (a->fwd && h->il2d_col.cut > 0)
                                   ? a->dst + 2 * b0 * rn
                                   : a->src + 2 * b0 * rn;
            vfft_real *bd = a->dst + 2 * b0 * rn;
            const int hookb = (h->il2d_fs_tw != NULL) && !a->fwd;
            if (hookb && h->il2d_col.tfuse)
            {
                if (bd != bs)
                    memcpy(bd, bs, 2 * (size_t)h->il2d_col.wl * rn * sizeof(vfft_real));
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
             * il2d_large_plane_design.md) */
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
             * the width fits it (measured 512x128 T=8 75 -> 37 us, 1024x128
             * 139 -> 64: the shared full-pitch scratch puts every strip piece
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
        (void)_il2d_nat_leaf_blocks(h, a->tid, a->dir, a->src, a->dst, a->lo, a->hi, /*fuse=*/a->fwd, VFFT_IL2D_NAT_PITCH(rn));
        break;
    case 3: /* Bluestein column axis: the window pipeline */
        _il2d_blu_cols_range(a->src, a->dst, h->N, rn, a->lo, a->hi,
                             h->il2d_col.blu, h->il2d_col.nst, h->il2d_col.R, h->il2d_col.L,
                             h->il2d_col.f, h->il2d_col.b, h->il2d_col.tf, h->il2d_col.tb,
                             a->fwd ? h->il2d_col.bluchf : h->il2d_col.bluchb,
                             a->fwd ? h->il2d_col.blukf : h->il2d_col.blukb,
                             h->il2d_col.bluscr);
        break;
    case 6: /* the threaded SKEWED column pass: the single column
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
    case 8: /* the threaded TURN, phase 1: rows [4 lo, 4 hi) through
             * the batched mono kernel with turned stores into the N2 x P scratch */
        (a->fwd ? h->il2d_rowt_f : h->il2d_rowt_b)(a->src + 2 * 4 * a->lo * rn, NULL,
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
                _vfft_exec6((vfft_plan)tp, a->dir, h->il2d_turn_scr + 2 * c * P, NULL,
                             h->il2d_turn_scr + 2 * c * P, NULL);
        }
        break;
    case 11: /* the TILE arm, forward: tiles [lo,hi) = the first
              * stage's sub-problems (N/R0 rows each, the scratch): the middle
              * stages in place on the tile, then its leaf blocks with the
              * rows fused (and streamed when staged) */
        {
            const int R0 = h->il2d_col.R[0], Rl = h->il2d_col.R[h->il2d_col.nst - 1];
            const size_t D0 = (size_t)h->N / (size_t)R0, np = VFFT_IL2D_NAT_PITCH(rn);
            size_t t;
            for (t = a->lo; t < a->hi; t++)
            {
                vfft_real *tile = (vfft_real *)a->src + 2 * t * D0 * np;   /* the scratch: writable */
                if (h->il2d_col.nst > 2)
                    _il2d_col_stages2(tile, tile, (int)D0, np, rn, 1, h->il2d_col.nst - 1,
                                      h->il2d_col.R, h->il2d_col.L, fns, tabs, 0);
                _il2d_nat_leaf_blocks(h, a->tid, a->dir, a->src, a->dst,
                                      t * D0 / (size_t)Rl, (t + 1) * D0 / (size_t)Rl, 1, np);
            }
        }
        break;
    case 12: /* the tile arm, backward: the leaf gathers the tile's natural rows
              * into the scratch, then the middle stages reversed in place */
        {
            const int R0 = h->il2d_col.R[0], Rl = h->il2d_col.R[h->il2d_col.nst - 1];
            const size_t D0 = (size_t)h->N / (size_t)R0, np = VFFT_IL2D_NAT_PITCH(rn);
            size_t t;
            for (t = a->lo; t < a->hi; t++)
            {
                vfft_real *tile = a->dst + 2 * t * D0 * np;
                (void)_il2d_nat_leaf_blocks(h, a->tid, a->dir, a->src, a->dst,
                                            t * D0 / (size_t)Rl, (t + 1) * D0 / (size_t)Rl, 0, np);
                if (h->il2d_col.nst > 2)
                    _il2d_col_stages2(tile, tile, (int)D0, np, rn, 1, h->il2d_col.nst - 1,
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
static void _il2d_c2c_mt_phase(struct vfft_plan_s *h, const vfft_real *src,
                               vfft_real *dst, vfft_dir_t dir, int fwd,
                               int mode, size_t units, int T)
{
    _il2d_c2c_mt_arg a[THREAD_POOL_MAX_DISPATCH];
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
    thread_pool_run(T, _il2d_c2c_mt_tramp, a, sizeof a[0]); /* caller = a[0] (tid 0) */
}

/* ── the THREADED skewed column pass and turn. Both are loop
 * restrictions of their serial executes (the same kernels on disjoint
 * ranges: MT == ST bitwise). csk: the column stage over 16-lane column
 * blocks, then the rows over row slabs -- route 0 through worker t's clone
 * of the out-of-place row plan, the batched routes through stateless
 * kernels on their own slots. turn: the turned rows over row slabs, the
 * scratch rows (the plane's columns) through worker t's clone of the N1
 * plan, the back-turn over destination row slabs. Return 1 when threaded,
 * 0 when the caller must run serial (the engagement counter shows it). */
static int _il2d_csk_exec_mt(struct vfft_plan_s *h, const vfft_real *sre, vfft_real *dre, vfft_dir_t dir, int T)
{
    const size_t rn = (size_t)h->N2, N1 = (size_t)h->N;
    const int fwd = (dir == VFFT_FORWARD);
    int Tc, Tr;
    T = thread_pool_workers_for(T);
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
static int _il2d_turn_exec_mt(struct vfft_plan_s *h, const vfft_real *sre, vfft_real *dre, vfft_dir_t dir, int T)
{
    const size_t rn = (size_t)h->N2, N1 = (size_t)h->N, P = VFFT_IL2D_TURN_PITCH(N1);
    const int fwd = (dir == VFFT_FORWARD);
    int Tr, Tc;
    size_t c;
    T = thread_pool_workers_for(T);
    if (T < 2 || (N1 % 4) != 0)
        return 0;
    Tc = (rn < (size_t)T) ? (int)rn : T;
    if (h->il2d_turnw_n < Tc - 1)
        return 0; /* the scratch rows run the N1 plan: clones are mandatory */
    Tr = (N1 / 4 < (size_t)T) ? (int)(N1 / 4) : T;
    if (Tr >= 2)
        _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 8, N1 / 4, Tr);
    else
        (fwd ? h->il2d_rowt_f : h->il2d_rowt_b)(sre, NULL, h->il2d_turn_scr, NULL, NULL, NULL, 1, rn, P, 1, N1);
    if (Tc >= 2)
        _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 9, rn, Tc);
    else
        for (c = 0; c < rn; c++)
            _vfft_exec6((vfft_plan)h->il2d_turn_plan, dir, h->il2d_turn_scr + 2 * c * P, NULL,
                         h->il2d_turn_scr + 2 * c * P, NULL);
    if (Tr >= 2)
        _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 10, N1, T);
    else
        _il2d_turn_back(h->il2d_turn_scr, P, dre, N1, rn);
    _vfft_il2d_col_mt_count++;
    return 1;
}

/* VFFT_IL2D_PHASES read once: no getenv on the execute path (the CRT's
 * environment lock serialises the workers' phase starts) */
static int _il2d_phlog_flag = -1;
static inline int _il2d_phlog_on(void)
{
    if (_il2d_phlog_flag < 0)
        _il2d_phlog_flag = getenv("VFFT_IL2D_PHASES") != NULL;
    return _il2d_phlog_flag;
}
/* Returns 1 when it ran threaded, 0 when the caller must run serial. */
static int _il2d_c2c_mt(struct vfft_plan_s *h, const vfft_real *sre,
                        vfft_real *dre, vfft_dir_t dir, int T)
{
    const size_t rn = (size_t)h->N2;
    const int fwd = (dir == VFFT_FORWARD);
    int s;
    if (h->il2d_turn)
        return _il2d_turn_exec_mt(h, sre, dre, dir, T); /* the turn's own walk */
    if (h->il2d_csk)
        return _il2d_csk_exec_mt(h, sre, dre, dir, T);  /* the skewed pass's own walk */
    if (h->il2d_col.staged)
        return 0; /* env-experimental route: one shared band scratch —
                   * per-worker slots are not built for it */
    /* T arrives as the plan's snapshot (h->nthreads); the pool's one clamp
     * bounds it by the live pool and the arg-array size. */
    T = thread_pool_workers_for(T);
    if (T < 2 || h->il2d_roww_n < T - 1)
        return 0; /* every arm here runs rows => clones are mandatory */
    if (h->il2d_col.tpc)
    {   /* the turned prime pass (2026-11-18): every worker its lanes through its own clone of the 1D plan, then the
         * rows over row slabs -- the serial walk's order (il_execute.h) */
        const int Tr = (size_t)h->N < (size_t)T ? h->N : T;
        if (!_il2d_tpc_cols_mt(&h->il2d_col, sre, dre, rn, !fwd, T))
            return 0;
        _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 2, (size_t)h->N, Tr);
        _vfft_il2d_col_mt_count++;
        return 1;
    }
    if (h->il2d_col.blu)
    {   /* Bluestein column axis: column windows, then rows —
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
        /* NATURAL x MT: digit-split prefix (sre -> scratch,
         * then in place), the leaf scatter by BLOCK RANGE (mode 4,
         * scratch -> dre), then row slabs on dre; bwd mirrors (gather
         * first, reversed prefix, stage 0 scratch -> dre). The band arm
         * is structurally out (the scatter crosses bands). */
        const int Rl = h->il2d_col.R[h->il2d_col.nst - 1];
        const size_t nb = (size_t)h->N / (size_t)Rl;
        const int Tb = nb < (size_t)T ? (int)nb : T;
        const int Tr = (size_t)h->N < (size_t)T ? h->N : T;
        const size_t np = VFFT_IL2D_NAT_PITCH(rn);   /* the scratch's and the staging's row pitch (il2d_col.h) */
        vfft_real *scr = h->il2d_col.natscr;
        int rows_done = 0;
        if (h->il2d_col.nst < 2 || (Tb < 2 && Tr < 2))
            return 0;
        if (h->il2d_col.natarm == 2)
        {   /* the TILE arm: stage 0 across the plane (its digit is
             * the row within a first-stage sub-problem), then every sub-problem
             * a tile one worker owns -- the middle stages in L2, the leaf's
             * natural rows with the row transform fused; the plane crosses
             * cores once each way */
            const int R0 = h->il2d_col.R[0];
            const size_t D0 = (size_t)h->N / (size_t)R0;
            const int Tt = (R0 < T) ? R0 : T;
            if (R0 < 2 || Tt < 2 || D0 % (size_t)Rl)
                return 0;
            if (fwd || _il2d_nat_bwd_fused(h))
            {   /* the forward's shape (a fused backward runs it with the backward rows) */
                if (!_il2d_stage_digits_mt_x(sre, rn, scr, np, h->N, rn, R0, h->il2d_col.L[0],
                                             h->il2d_col.f[0], h->il2d_col.tf[0], T))
                    _il2d_col_stage_x(sre, rn, scr, np, h->N, rn, 0, h->il2d_col.R, h->il2d_col.L,
                                      h->il2d_col.f, h->il2d_col.tf);
                _il2d_c2c_mt_phase(h, scr, dre, dir, /*the shape*/ 1, 11, (size_t)R0, Tt);
            }
            else
            {
                _il2d_c2c_mt_phase(h, sre, scr, dir, fwd, 12, (size_t)R0, Tt);
                if (!_il2d_stage_digits_mt_x(scr, np, dre, rn, h->N, rn, R0, h->il2d_col.L[0],
                                             h->il2d_col.b[0], h->il2d_col.tb[0], T))
                    _il2d_col_stage_x(scr, np, dre, rn, h->N, rn, 0, h->il2d_col.R, h->il2d_col.L,
                                      h->il2d_col.b, h->il2d_col.tb);
                _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 2, (size_t)h->N, Tr);   /* the mirrored walk: the rows after its columns */
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
            if (phlog) p0 = vfft_now_ns();
            if (Ts >= 2)
                _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 5, rn, Ts);
            else
                _il2d_col_pass_nat(sre, dre, h->N, rn, np, h->il2d_col.nst,
                                   h->il2d_col.R, h->il2d_col.L,
                                   fwd ? h->il2d_col.f : h->il2d_col.b,
                                   fwd ? h->il2d_col.tf : h->il2d_col.tb, !fwd,
                                   h->il2d_col.natperm, scr, _il2d_nat_stage_of(h, 0));
            if (phlog) p1 = vfft_now_ns();
            _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 2, (size_t)h->N, Tr);
            if (phlog)
                fprintf(stderr, "[il2d-phases] %dx%zu T=%d strips msw=%d nls=%d: cols=%.0f rows=%.0f ns\n",
                        h->N, rn, T, h->il2d_col.msw, h->il2d_col.natst, p1 - p0, vfft_now_ns() - p1);
            _vfft_il2d_col_mt_count++;
            return 1;
        }
        if (fwd || _il2d_nat_bwd_fused(h))
        {   /* the forward's shape (a fused backward runs it with the backward rows) */
            for (s = 0; s < h->il2d_col.nst - 1; s++)
            {   /* stage 0 moves the plane into the scratch (the two pitches), the others run in place there */
                const vfft_real *ssrc = (s == 0) ? sre : scr;
                const size_t sp = (s == 0) ? rn : np;
                if (!_il2d_stage_digits_mt_x(ssrc, sp, scr, np, h->N, rn,
                                             h->il2d_col.R[s], h->il2d_col.L[s],
                                             h->il2d_col.f[s], h->il2d_col.tf[s], T))
                    _il2d_col_stage_x(ssrc, sp, scr, np, h->N, rn, s,
                                      h->il2d_col.R, h->il2d_col.L, h->il2d_col.f,
                                      h->il2d_col.tf);
            }
            if (Tb >= 2)
                _il2d_c2c_mt_phase(h, scr, dre, dir, /*the shape*/ 1, 4, nb, Tb);
            else
                (void)_il2d_nat_leaf_blocks(h, 0, dir, scr, dre, 0, nb, 1, np);
            rows_done = 1;   /* the block arm fused its rows in the leaf */
        }
        else
        {   /* the mirrored walk */
            if (Tb >= 2)
                _il2d_c2c_mt_phase(h, sre, scr, dir, fwd, 4, nb, Tb);
            else
                (void)_il2d_nat_leaf_blocks(h, 0, dir, sre, scr, 0, nb, 0, np);
            for (s = h->il2d_col.nst - 2; s >= 0; s--)
            {
                vfft_real *out = (s == 0) ? dre : scr;
                const size_t op = (s == 0) ? rn : np;
                if (!_il2d_stage_digits_mt_x(scr, np, out, op, h->N, rn,
                                             h->il2d_col.R[s], h->il2d_col.L[s],
                                             h->il2d_col.b[s], h->il2d_col.tb[s], T))
                    _il2d_col_stage_x(scr, np, out, op, h->N, rn, s,
                                      h->il2d_col.R, h->il2d_col.L, h->il2d_col.b,
                                      h->il2d_col.tb);
            }
        }
        if (!rows_done)
            _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 2, (size_t)h->N, Tr);
        _vfft_il2d_col_mt_count++;
        return 1;
    }
    if (h->il2d_col.wl > 0 && h->il2d_col.natarm != 1)
    {   /* (a strips verdict — natarm == 1, any class — takes the
         * unbanded strip walk below whatever the serial band) */
        /* fewer bands than workers is a CLAMP, not a decline: 4 bands
         * across 4 workers still beats serial, and the prefix digit
         * split keeps the full T regardless (its axis is D, not nb). */
        const size_t nb = (size_t)h->N / (size_t)h->il2d_col.wl;
        const int Tb = nb < (size_t)T ? (int)nb : T;
        /* VFFT_IL2D_PHASES: per-phase ns of this walk on stderr (a probe
         * instrument) */
        const int phlog = _il2d_phlog_on();
        double ph0 = 0, ph1 = 0, ph2 = 0, ph3 = 0;
        if (Tb < 2)
            return 0;
        if (phlog) ph0 = vfft_now_ns();
        if (fwd && h->il2d_col.cut > 0)
            for (s = 0; s < h->il2d_col.cut; s++)
            {
                const vfft_real *ssrc = (s == 0) ? sre : dre;
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
                memcpy(dre, sre, 2 * (size_t)h->N * rn * sizeof(vfft_real));
            _il2d_c2c_mt_phase(h, dre, dre, dir, fwd, 2, (size_t)h->N, T);
            sre = dre;
        }
        if (phlog) ph1 = vfft_now_ns();
        _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 0, nb, Tb);
        if (phlog) ph2 = vfft_now_ns();
        if (!h->il2d_col.tfuse && !(h->il2d_fs_tw && !fwd))
            _il2d_c2c_mt_phase(h, sre, dre, dir, fwd, 2, (size_t)h->N,
                               T);
        if (phlog)
        {
            ph3 = vfft_now_ns();
            fprintf(stderr, "[il2d-phases] %dx%zu T=%d wl=%d cut=%d nst=%d tfuse=%d: prefix=%.0f bands=%.0f rows=%.0f ns\n",
                    h->N, rn, T, h->il2d_col.wl, h->il2d_col.cut, h->il2d_col.nst, h->il2d_col.tfuse,
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
                memcpy(dre, sre, 2 * (size_t)h->N * rn * sizeof(vfft_real));
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
{   /* the tcut law lives in planning/policy.h (R3) */
    return vfft_policy_il2d_wl_cut(h->N, h->il2d_col.nst, h->il2d_col.L, wl);
}

/* ── the arms of the il2d races (support/race.h): one context, the same
 * functions execute serves with. For the column-MT race the threaded arm
 * reports whether it could engage; the race runs to completion either way
 * and the site banks the "no" afterwards. */
typedef struct
{
    struct vfft_plan_s *h;
    vfft_real *a, *z;          /* real plane / complex scratch (in place) */
    int isr;                /* r2c fwd (a -> z) or c2r bwd (z -> a) */
    int ok;                 /* colmt: the threaded arm engaged */
    /* chain candidate (the chain race) */
    int N1;
    size_t N2;
    int nst;
    const int *R;
    int *Ls;
    vfft_il2p_fn *ff;
    vfft_real **tf;
    /* the chain race under NATURAL order: the candidate is
     * timed through the leaf-redirected natural pass with ITS OWN perm —
     * the best chain for scrambled and for natural can differ (the
     * leaf radix sets the scatter width), so natural cells race under
     * the pass they serve and bank under their own ord=nat row. */
    int nat;
    const int *perm;
    vfft_real *nscr;
    int sw;                   /* the strips arm's sub-strip width (0 = unsized) */
    vfft_real *nstage;           /* the natural leaf's staging for the chain arm */
    int lst;                  /* the natural leaf: staged (1) or strided (0) for this arm */
    vfft_real *zo;               /* the threading race's destination plane: the cell's own placement */
    int wl, cut;              /* the chain race's banded arm: the band width and its cut (wl = 0: unbanded) */
    int T;                    /* the chain race at T > 1: the candidate timed through the THREADED pass (_il2d_arm_chain_mt) */
    int wc;                   /* a lane-major chain arm (_lm_chain_arm): the lane tile it serves with (0 = none: the walk,
                               * or the real batch's whole plane) */
    int lmtile;               /* a lane-major chain arm at T > 1: the candidate as the TILE on T threads (_lm_chain_arm,
                               * mtarm = 2), 0 = a serial form */
    size_t np;                /* the row pitch of nscr and the staging in the serial natural arms: the pitch the cell's
                               * serial walk lays them at (the c2c 2D plan: il2d_col.h VFFT_IL2D_NAT_PITCH); 0 = N2 */
} _il2d_race_ctx_t;
/* the serial natural arms' pitch of their own buffers (above) */
static inline size_t _il2d_race_np(const _il2d_race_ctx_t *c)
{
    return c->np ? c->np : c->N2;
}
/* THE PITCH A CELL'S SERIAL WALK LAYS THE NATURAL PASS'S OWN BUFFERS AT (il2d_col.h VFFT_IL2D_NAT_PITCH): the c2c 2D
 * plan's walks (il_execute.h) the natural pass's own pitch; every other cell's (the real tier, a rank-3 axis, a
 * lane-major batch) the row. A race's serial natural arm takes it from here: the arm is the serving walk. */
static inline size_t _il2d_key_np(const vw2_ilcol_key_t *key, size_t rn)
{
    return (key->rank == 2 && !key->real && !key->lm) ? VFFT_IL2D_NAT_PITCH(rn) : rn;
}

/* A LANE-MAJOR CELL IN THE CHAIN RACE (il/rank1/lm_batch_il.h, rank2/lm_batch2d_il.h, lm_batch2dr_il.h,
 * rank3/lm_batch3d_il.h, lm_batch3dr_il.h): every candidate is timed in the serial forms its door serves it with,
 * through the door's own pass on a plan that carries the candidate (_lm_chain_plan): the c2c pass (the 1D batch,
 * the first axis of a 2D / 3D c2c batch) at each lane tile the serial-form law admits for the chain, or the walk;
 * the 1D real batch's executor (the fused strips at each width, the whole plane as one strip, the three-phase form
 * past the scratch); the first axis of a 2D / 3D real batch, the complex pass on the spectrum, at each width and the
 * walk. In the cell's placement and direction: in place where the door runs the pass in place (always for the r2c
 * spectrum), the backward for a c2r cell. The dense scratch is the one the door builds at the cell's T (one thread:
 * the doors'; more: _lm_race's for its threaded arms, the 1D real batch's own), so a width past it runs what the door
 * runs there. The column builder fills the request part from the key and the request (_il2d_lmform_of); the race's
 * shared pieces are lm_batch_il.h's (_lm_chain_open / _lm_chain_close); the race hands back the width the winning
 * chain served best with. */
typedef struct
{
    int kind;            /* 0 = the c2c pass; 1 = the 1D real batch; 2 = the first axis of a 2D / 3D real batch */
    int ip;              /* the pass runs in place */
    int rev;             /* the backward (a c2r cell) */
    int T;               /* the cell's thread count */
    vfft_ilcol_t sscr;   /* the dense scratch a scrambled chain's tiles run in (and every chain's at one thread) */
    vfft_ilcol_t snat;   /* a natural chain's at T > 1 (its own builder), empty otherwise */
    vfft_real *W, *tw;   /* the 1D real batch: the three-phase form's plane, the fold's twiddles */
    int wc;              /* out: the lane tile the winning chain served best with */
} _il2d_lmform_t;
/* lm_batch_il.h, later in this TU */
static int _lm_chain_open(_il2d_lmform_t *f, int N1, size_t rn, int nat_req);
static void _lm_chain_close(_il2d_lmform_t *f);
static int _lm_chain_forms(const _il2d_lmform_t *f, int N1, size_t rn, const int *R, int nat, int *wcs);
static struct vfft_plan_s *_lm_chain_plan(const _il2d_lmform_t *f, int N1, size_t rn, int nst, const int *R,
                                          const int *L, vfft_il2p_fn const *ff, vfft_il2p_fn const *fb,
                                          vfft_real *const *tf, vfft_real *const *tb, int nat, const int *perm,
                                          vfft_real *nscr);
static void _lm_chain_plan_free(struct vfft_plan_s *h);
static void _lm_chain_arm(void *v);
static void _il2d_arm_cols(void *v)
{
    _il2d_race_ctx_t *c = (_il2d_race_ctx_t *)v;
    _il2d_real_cols(c->h, c->z, c->z, 0);
}
static void _il2d_arm_cols_mt(void *v)
{
    _il2d_race_ctx_t *c = (_il2d_race_ctx_t *)v;
    c->h->il2d_col.natarm = 0;   /* the matched partition */
    if (c->ok && !_il2d_real_cols_mt(c->h, c->z, c->z, 0, c->h->nthreads))
        c->ok = 0; /* the threaded arm cannot engage on this cell */
}
static void _il2d_arm_cols_mt_sw(void *v)
{   /* the column blocks in sub-strips of c->sw columns (0 = the worker's whole range) */
    _il2d_race_ctx_t *c = (_il2d_race_ctx_t *)v;
    c->h->il2d_col.natarm = 1;
    c->h->il2d_col.msw = c->sw;
    if (c->ok && !_il2d_real_cols_mt(c->h, c->z, c->z, 0, c->h->nthreads))
        c->ok = 0;
}
/* THE WHOLE TRANSFORM as one arm of the column-verdict race: the column pass in the mode and
 * shape the context names (colmt / natarm / msw), the rows as the cell serves them at this point
 * of the create (the door route; the row plan is raced after, under this verdict) -- r2c the rows
 * then the columns on the CCE plane, c2r the columns onto the column-inverse plane then the rows
 * from it; in place on the one plane (a == z). A pass timed alone never pays the exchange between
 * the passes (§6.4): the verdict must. */
static void _il2d_arm_whole(void *v)
{
    _il2d_race_ctx_t *c = (_il2d_race_ctx_t *)v;
    struct vfft_plan_s *h = c->h;
    h->il2d_col.colmt = c->lst;      /* the arm's mode */
    h->il2d_col.natarm = c->nat;     /* the arm's shape */
    h->il2d_col.msw = c->sw;
    if (c->isr)
    {
        _il2d_real_rows_fwd(h, c->a, c->z);
        _il2d_real_cols(h, c->z, c->z, 0);
    }
    else
    {
        _il2d_real_cols(h, c->z, h->il2d_rscr, 1);
        _il2d_real_rows_bwd(h, h->il2d_rscr, c->a);
    }
}
static void _il2d_arm_exec_st(void *v)
{
    _il2d_race_ctx_t *c = (_il2d_race_ctx_t *)v;
    c->h->il2d_col.colmt = 0;
    c->h->il2d_col.natst = 1;      /* the serial walk's form: staged */
    _vfft_exec6((vfft_plan)c->h, VFFT_FORWARD, c->z, NULL, c->zo ? c->zo : c->z, NULL);
}
static void _il2d_arm_exec_mt(void *v)
{   /* the matched partition: no sub-strips -- at wl = 0 a scrambled plane's walk is the strips over each
     * worker's whole block, the form the verdict serves (msw 0); an msw left by the ladder arm before it
     * timed that arm's form under this one's name */
    _il2d_race_ctx_t *c = (_il2d_race_ctx_t *)v;
    c->h->il2d_col.colmt = 1;
    c->h->il2d_col.natarm = 0;
    c->h->il2d_col.msw = 0;
    c->h->il2d_col.natst = c->lst;
    _vfft_exec6((vfft_plan)c->h, VFFT_FORWARD, c->z, NULL, c->zo ? c->zo : c->z, NULL);
}
static void _il2d_arm_exec_mt_strip(void *v)
{   /* natural cells only: the strip partition of the natural pass */
    _il2d_race_ctx_t *c = (_il2d_race_ctx_t *)v;
    c->h->il2d_col.colmt = 1;
    c->h->il2d_col.natarm = 1;
    c->h->il2d_col.msw = 0;
    c->h->il2d_col.natst = c->lst;
    _vfft_exec6((vfft_plan)c->h, VFFT_FORWARD, c->z, NULL, c->zo ? c->zo : c->z, NULL);
}
static void _il2d_arm_exec_mt_tile(void *v)
{   /* natural cells: the TILE partition */
    _il2d_race_ctx_t *c = (_il2d_race_ctx_t *)v;
    c->h->il2d_col.colmt = 1;
    c->h->il2d_col.natarm = 2;
    c->h->il2d_col.msw = 0;
    c->h->il2d_col.natst = c->lst;
    _vfft_exec6((vfft_plan)c->h, VFFT_FORWARD, c->z, NULL, c->zo ? c->zo : c->z, NULL);
}
static void _il2d_arm_exec_mt_sw(void *v)
{   /* every class: strips of c->sw columns (il2d_large_plane_design.md) */
    _il2d_race_ctx_t *c = (_il2d_race_ctx_t *)v;
    c->h->il2d_col.colmt = 1;
    c->h->il2d_col.natarm = 1;
    c->h->il2d_col.msw = c->sw;
    c->h->il2d_col.natst = c->lst;
    _vfft_exec6((vfft_plan)c->h, VFFT_FORWARD, c->z, NULL, c->zo ? c->zo : c->z, NULL);
}
/* the strip-width ladder: N1 x sw x 16 B under L2 (the hardware fence).
 * At T > 1 every width up to a worker's column block (N2 / T): a worker
 * walks its block in sub-strips of sw and the last one is narrower where
 * sw does not divide the block (_il2d_cmt_body, mode 5), as the real
 * tier's ladder has it -- so a plane with no width dividing N2 (750, 125,
 * 1500 columns) still walks a narrow dense strip in L2 instead of each
 * worker's whole block (1500x750 T=8: the column pass 943 us whole, MKL's
 * 4-column groups 256). One thread (the serial walk's tile) keeps sw a
 * divisor of N2. 8 is on it because at N1 = 8192 the 16-column dense
 * strip is the whole L2 and streams every stage through L3 (8192x128
 * measured 0.76 at T=8); the column kernels take any count >= 1. N2 is
 * size_t: a lane-major batch's column count is its lane count. */
static int _il2d_sw_ladder(int N1, size_t N2, int T, int *out, int max)
{
    static const int SW[] = { 8, 16, 32, 64, 128, 256 };
    int i, n = 0;
    for (i = 0; i < 6 && n < max; i++)
    {
        const int w = SW[i];
        if ((size_t)w > N2 || (T > 1 ? (size_t)w > N2 / (size_t)T : N2 % (size_t)w != 0)) continue;
        if (!vfft_policy_fits_l2((long)N1 * w * VFFT_CX_BYTES)) continue;
        out[n++] = w;
    }
    return n;
}
static void _il2d_arm_exec(void *v)
{
    _il2d_race_ctx_t *c = (_il2d_race_ctx_t *)v;
    _vfft_exec6((vfft_plan)c->h, VFFT_FORWARD, c->z, NULL, c->z, NULL);
}
static void _il2d_arm_chain(void *v)
{
    _il2d_race_ctx_t *c = (_il2d_race_ctx_t *)v;
    if (c->nat)
        _il2d_col_pass_nat(c->z, c->z, c->N1, c->N2, _il2d_race_np(c), c->nst, c->R, c->Ls,
                           c->ff, c->tf, /*reverse=*/0, c->perm, c->nscr, c->nstage);
    else
        _il2d_col_pass(c->z, c->z, c->N1, c->N2, 0, c->nst, c->R, c->Ls,
                       c->ff, c->tf, /*reverse=*/0);
}
/* THE CHAIN RACE'S ARMS ARE THE SERVING WALK (2026-10-06). A chain used to be
 * timed as the unbanded column pass alone: on a plane past L2 every stage of
 * that pass is a DRAM sweep, so the race crowned the chain with the fewest
 * stages (32.8.8 at 2048x1024) -- while the walk that SERVES is banded, where
 * only the wide prefix stages sweep the plane and the suffix runs inside an
 * L2-resident band. There the cost is the wide stage's radix (a radix-32 wide
 * stage runs 64 streams 16 KB apart, past the prefetchers and the first-level
 * TLB: 2.7 copies of the plane where a radix-8 stage costs one), and a chain
 * of four stages beat the verdict by 24% (phase split, 2048x1024). So every
 * candidate is timed as the column walk it would serve with -- out of place
 * from z into zo, as the door runs it: unbanded, and banded at every width the
 * band law admits for it (the ladder, the stage spans that fit L2) -- and its
 * score is its best form. The rows are not in these arms (their cost does not
 * depend on the chain); the width itself is raced again on the full execute
 * by the axis race, for the chain that won. */
static void _il2d_arm_chain_oop(void *v)
{
    _il2d_race_ctx_t *c = (_il2d_race_ctx_t *)v;
    if (c->nat)
        _il2d_col_pass_nat(c->z, c->zo, c->N1, c->N2, _il2d_race_np(c), c->nst, c->R, c->Ls,
                           c->ff, c->tf, /*reverse=*/0, c->perm, c->nscr, c->nstage);
    else
        _il2d_col_pass(c->z, c->zo, c->N1, c->N2, 0, c->nst, c->R, c->Ls,
                       c->ff, c->tf, /*reverse=*/0);
}
static void _il2d_arm_chain_banded(void *v)
{
    _il2d_race_ctx_t *c = (_il2d_race_ctx_t *)v;
    const int cut = c->cut, nst = c->nst, N1 = c->N1;
    const size_t rn = c->N2, wl = (size_t)c->wl;
    size_t b0;
    if (c->nat)
    {   /* the natural banded walk's column part (il_execute.h): the wide prefix
         * z -> nscr, per band the suffix in place on nscr and the staged leaf's
         * blocks scattered to their natural rows of zo; nscr and the staging at
         * the walk's own pitch */
        const int Rl = c->R[nst - 1];
        const size_t np = _il2d_race_np(c);
        _il2d_col_stages_x(c->z, rn, c->nscr, np, N1, rn, 0, cut, c->R, c->Ls, c->ff, c->tf, 0);
        for (b0 = 0; b0 < (size_t)N1; b0 += wl)
        {
            const size_t blo = b0 / (size_t)Rl, bhi = (b0 + wl) / (size_t)Rl;
            if (cut > 0)
                _il2d_col_stages2(c->nscr + 2 * b0 * np, c->nscr + 2 * b0 * np, (int)wl, np, rn, cut, nst - 1,
                                  c->R, c->Ls, c->ff, c->tf, 0);
            else
                _il2d_col_stages_x(c->z + 2 * b0 * rn, rn, c->nscr + 2 * b0 * np, np, (int)wl, rn, cut, nst - 1,
                                   c->R, c->Ls, c->ff, c->tf, 0);
            _il2d_nat_leaf_range(c->nscr, c->zo, N1, rn, np, Rl, c->ff[nst - 1], c->perm, blo, bhi, 0, c->nstage);
        }
        return;
    }
    /* the scrambled banded walk: the wide prefix z -> zo, per band the suffix in place on zo */
    if (cut > 0)
        _il2d_col_stages(c->z, c->zo, N1, rn, 0, cut, c->R, c->Ls, c->ff, c->tf, 0);
    for (b0 = 0; b0 < (size_t)N1; b0 += wl)
        _il2d_col_stages((cut > 0 ? c->zo : c->z) + 2 * b0 * rn, c->zo + 2 * b0 * rn, (int)wl, rn, cut, nst,
                         c->R, c->Ls, c->ff, c->tf, 0);
}
/* THE CANDIDATE THROUGH THE THREADED PASS (2026-10-08). At T > 1 the cell serves its column pass
 * threaded under the colmt verdict, and that pass's cost is the chain's digit structure, not its
 * stage count: a prefix stage with fewer digits than workers (D = L/R < T) cannot split and runs
 * serial inside the threaded pass -- at 1024x1024 the serial race crowned 8.64.2 (its radix-64
 * stage has two digits) and the threaded pass took 650 us where 32.32 takes 245. So at T > 1 every
 * form of every candidate -- the unbanded walk (the natural partition, or the strips) and the
 * banded walk at every admitted width -- is timed through the very body the plan will serve with
 * (_il2d_cols_mt_desc over the candidate's view, z -> zo like the serial forms), the serial form
 * standing in where a width cannot engage. The colmt race afterwards still decides serial or
 * threaded for the winner. */
static void _il2d_arm_chain_mt(void *v)
{
    _il2d_race_ctx_t *c = (_il2d_race_ctx_t *)v;
    vfft_ilcol_t cv;
    memset(&cv, 0, sizeof cv);
    cv.N = c->N1;
    cv.rn = c->N2;
    cv.nst = c->nst;
    memcpy(cv.R, c->R, (size_t)c->nst * sizeof cv.R[0]);
    memcpy(cv.L, c->Ls, (size_t)c->nst * sizeof cv.L[0]);
    memcpy(cv.f, c->ff, (size_t)c->nst * sizeof cv.f[0]);
    memcpy(cv.tf, c->tf, (size_t)c->nst * sizeof cv.tf[0]);
    cv.wl = c->wl;
    cv.cut = c->cut;
    cv.nat = c->nat;
    cv.natperm = (int *)c->perm;
    cv.natscr = c->nscr;
    if (!_il2d_cols_mt_desc(&cv, c->N1, c->z, c->zo, 0, c->T, c->nat ? c->nstage : NULL, -1))
    {   /* this form cannot engage at T: the serial form, as the dispatcher would run it */
        if (c->wl > 0)
            _il2d_arm_chain_banded(v);
        else
            _il2d_arm_chain_oop(v);
    }
}
/* the band widths the chain race offers a candidate: the serving law's own
 * admission (the axis race's) -- the ladder under the tcut law, the stage
 * spans that fit L2; a VFFT_IL2D_WL pin narrows it to that width (a probe) */
static int _il2d_race_widths(int N1, size_t N2, int nst, const int *Ls, int *wlc, int cap)
{
    int nwl = 0, p, s2;
    for (p = 0; p < VFFT_IL2D_WL_LADDER_N && nwl < cap; p++)
        if (vfft_policy_il2d_wl_cut(N1, nst, Ls, VFFT_IL2D_WL_LADDER[p]) >= 0)
            wlc[nwl++] = VFFT_IL2D_WL_LADDER[p];
    for (s2 = 1; s2 < nst && nwl < cap; s2++)
    {
        const int w = Ls[s2];
        int dup = 0, p2;
        if (!vfft_policy_il2d_band_ok(N1, nst, Ls, w))
            continue;
        if (!vfft_policy_fits_l2_lanes((size_t)w, N2))   /* w x N2 complexes, formed in size_t: a band past 2 GiB, or a
                                                           * 3D axis-0 plane past INT_MAX lanes, never wraps */
            continue;
        for (p2 = 0; p2 < nwl; p2++)
            if (wlc[p2] == w)
                dup = 1;
        if (!dup)
            wlc[nwl++] = w;
    }
    {
        const char *e = getenv("VFFT_IL2D_WL");
        if (e)
        {
            const int w = atoi(e);
            int p2, hit = 0;
            for (p2 = 0; p2 < nwl; p2++) if (wlc[p2] == w) hit = 1;
            if (hit) { wlc[0] = w; nwl = 1; }
            else nwl = 0;   /* 0 = the pin asks for the unbanded form */
        }
    }
    return nwl;
}
/* THE IN-PLACE RACE's reset (support/race.h's hook): an in-place arm walks its
 * own output, so the plane is re-laid from its seed before every timed
 * sample (docs/roadmap/real_inplace_design.md) */
typedef struct
{
    vfft_real *p;
    const vfft_real *seed;
    size_t n;
} _il2d_ip_reset_t;
static void _il2d_ip_reset(void *v)
{
    const _il2d_ip_reset_t *r = (const _il2d_ip_reset_t *)v;
    memcpy(r->p, r->seed, r->n * sizeof(vfft_real));
}
/* THE SAMPLE'S REPS of the races that time a column pass or a plane (the width, forms, chain,
 * prime-column and threading races below). vfft_now_ns reads a 100 ns counter on Windows, so one
 * call of a small plane's pass (8x4: 10-25 ns) reads 0 or 100 ns and a single-call sample is a
 * coin flip (8x4 banked a two-stage strided column pass, 34 ns, where one stage serves 15;
 * measured 2026-10-11). A sample is sized to ~0.3 ms of work from one timed call, as the row and
 * column plan races size theirs. An arm that runs IN PLACE multiplies the plane by up to `grow` a
 * call (the length it transforms, unnormalized), so its sample stays finite: grow^reps under
 * 1e290 from a plane seeded near 1, re-laid before every sample by the race's reset
 * (_il2d_ip_reset). grow <= 1: an out-of-place arm, whose input never changes. */
static int _il2d_race_reps_ns(void (*run)(void *), void *ctx, double grow, double target_ns);
static int _il2d_race_reps(void (*run)(void *), void *ctx, double grow)
{
    return _il2d_race_reps_ns(run, ctx, grow, 3.0e5);
}
/* the same, sized to target_ns of the arm's work. THREADED arms are sized on a threaded arm to ~1 ms
 * (VFFT_IL2D_MT_SAMPLE_NS) and race with one untimed execute before every sample (warm_each): a worker's cache
 * partition settles over the first executes after another arm's form ran -- a sample of one execute times that
 * transient, never the steady state the plan serves (the 3D tier's law, fftnd_il.h; 2026-11-19, 512x512 T=8: the
 * threading race sized on the serial arm timed one ~85 us execute a sample and served the strips, 5-10 % behind the
 * tile the same chain runs in ~1 ms samples) */
#define VFFT_IL2D_MT_SAMPLE_NS 1.0e6
static int _il2d_race_reps_ns(void (*run)(void *), void *ctx, double grow, double target_ns)
{
    double t0, est;
    int reps, cap = 4096;
    run(ctx);
    t0 = vfft_now_ns();
    run(ctx);
    est = vfft_now_ns() - t0;
    reps = (int)(target_ns / (est > 1.0 ? est : 1.0));
    if (grow > 1.0)
    {
        const double g = 290.0 / log10(grow);
        if (g < (double)cap)
            cap = g < 1.0 ? 1 : (int)g;
    }
    if (reps > cap) reps = cap;
    if (reps < 1) reps = 1;
    return reps;
}
/* ── the WL race: the banded column walk's width (rows stay OUTSIDE per
 * fft2d_real_il_design.md §2.5): the unbanded arm + the static pool +
 * L2-admitted stage spans (the c2c lever — row width here is hp1 complex),
 * timed on the column pass alone (the only thing wl changes), min-of-3 in
 * place on a z scratch re-laid before every sample (_il2d_race_reps).
 * Winner installed on h + banked (chain + wl) in the
 * direction-shared lay=il real cell. MUST run after the h-> field commits
 * (it executes h — the axis-race law). (Until 2026-10-03 this race also
 * held the ROWSPLIT row arm -- the split engines hired for the rows; it
 * measured no better than the per-row door and is gone.) */
/* Returns the column verdict the race decided at T > 1 (0 serial, 1 threaded: every width timed
 * under BOTH modes, the winner's mode banked as cmt beside wl), -1 when it decided none (one
 * thread, or no race). */
static int _il2d_real_wlrace(struct vfft_plan_s *h,
                             struct vfft_wisdom_s *W,
                             const vfft_cfg_t *cfg, int N1, int N2)
{
    const size_t CN = (size_t)N1 * h->il2d_col.rn;   /* the plane at the plan's pitch */
    const int isr = (h->transform == VFFT_R2C);
    vfft_real *bz = (vfft_real *)vfft_aligned_alloc((2 * CN + 8) * sizeof(vfft_real));
    vfft_real *bseed = (vfft_real *)vfft_aligned_alloc((2 * CN + 8) * sizeof(vfft_real));   /* the arms run in place: the reset's seed */
    size_t i;
    int bmode = -1;
    if (!bz || !bseed)
    {
        vfft_aligned_free(bz);
        vfft_aligned_free(bseed);
        return -1;
    }
    for (i = 0; i < 2 * CN + 8; i++)
        bz[i] = 1.0 + 1e-6 * (double)(i & 511);
    memcpy(bseed, bz, (2 * CN + 8) * sizeof(vfft_real));
    _il2d_race_ctx_t rc = { h, NULL, bz, isr, 1, 0, 0, 0, NULL, NULL, NULL, NULL };
    _il2d_ip_reset_t brs = { bz, bseed, 2 * CN + 8 };
    const vfft_race_arm_t cols_arm = { "cols", _il2d_arm_cols, &rc };
    const int mt = h->nthreads > 1;   /* T > 1: every width timed through the THREADED pass alone (the width is the
                                       * column pass's own lever; the dispatcher runs a width serial where it cannot
                                       * engage); the column verdict is the whole-transform race's (2026-10-08) */
    vfft_race_proto_t proto = { 3, 1, VFFT_RACE_MIN, 0, 0, _il2d_ip_reset, &brs, 1 }; /* min-of-3, the plane re-laid before every sample */ /* single-thread arms: paced (VFFT_RACE_PACE_MS) */
    if (mt)
    {   /* THREADED arms: two untimed passes per arm first, never paused (a paced pool parks its workers) */
        proto.warm = 2;
        proto.pace = 0;
    }
    {
        const size_t hp1 = (size_t)N2 / 2 + 1;
        int wlc[14], nwl = 0, wi, s2;
        double cbest = 1e300;
        int bwl = 0, bcut = 0;
        h->il2d_col.wl = 0;
        h->il2d_col.cut = 0;
        h->il2d_col.colmt = mt;
        h->il2d_col.natarm = 0;
        proto.reps = _il2d_race_reps(_il2d_arm_cols, &rc, (double)N1);   /* sized on the unbanded pass: the widths cost alike */
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
             * gated by (policy R2) */
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
        h->il2d_col.colmt = 0;   /* the column verdict is the whole-transform race's */
        vfft_aligned_free(bz);
        vfft_aligned_free(bseed);
        if (getenv("VFFT_IL2D_LOG"))
            fprintf(stderr, "[il2d-real] wlrace %s %dx%d%s -> wl=%d (%.0f ns cols)\n",
                    isr ? "r2c" : "c2r", N1, N2, mt ? " (the threaded pass)" : "", bwl, cbest);
        vw2_2d_rl_bank(&W->vw2, N1, N2, !isr, h->il2d_col.R, h->il2d_col.nst,
                       bwl, -1, -1, (N1 & (N1 - 1)) ? h->il2d_col.blu : -1,
                       cbest, vfft_policy_ord_rankn(cfg), h->nthreads, h->il2d_ip);
        _vw2_persist(W, cfg);
    }
    (void)bmode;
    return -1;
}

/* ── the COLUMN-MT verdict race. Times the column pass SERIAL vs
 * THREADED through the very functions the execute serves with (race ==
 * serving path), min-of-3 on a scratch plane, and banks {cmt, cmtt} in
 * the cell's lay=il real row. There is NO structural default and no
 * invented floor: at 512x32 (hp1=17 => the strip arm over 17 columns)
 * threading the column pass MEASURED SLOWER, and that "no" is banked
 * exactly like a "yes". A verdict is only served back at the SAME
 * thread count it was raced at (cmtt) — a T=4 verdict never serves a
 * T=8 request. MUST run after the plan's stage arrays are committed. */
static const char *_il2d_tkp(const struct vfft_plan_s *h, const char *base, char *buf, size_t sz);   /* il2d_real_plan.h: door 2's _p namer */
static int _il2d_real_sscr_build(vfft_ilcol_t *c, int N1, size_t rn, int T);   /* below: the dense strip scratch */
static int _il2d_real_sw_ladder(int N1, size_t rn, int T, int *out, int max);
static void _il2d_nat_sscr_free(vfft_ilcol_t *c);
/* THE COLUMN-VERDICT RACE (the column blocks joined 2026-10-08): the serial pass, the matched
 * threaded partition (mtarm = 0: digit-split + leaf for a natural chain, band or strip per wl
 * otherwise) and THE STRIPS (mtarm = 1: every worker the whole chain over its column block) in
 * sub-strips of every width the ladder admits (msw; 0 = the worker's whole block), a natural chain
 * through the dense per-worker scratch built here. The winner is served and banked: cmt, and
 * mtarm= msw= (the direction's twins, _p at door 2). A dense scratch the verdict does not use is
 * freed. */
static void _il2d_real_colmt_race(struct vfft_plan_s *h,
                                  struct vfft_wisdom_s *W,
                                  const vfft_cfg_t *cfg, int N1,
                                  int N2)
{
    const size_t CN = 2 * (size_t)N1 * h->il2d_col.rn;   /* the CCE plane at the plan's pitch, doubles */
    const int c2r = h->transform == VFFT_C2R, ord = vfft_policy_ord_rankn(cfg), ip = h->il2d_ip;
    const size_t RN = ip ? CN : (size_t)N1 * (size_t)N2;   /* the real plane: in place the one padded plane */
    vfft_real *z = (vfft_real *)vfft_aligned_alloc((CN + 8) * sizeof(vfft_real));
    vfft_real *a = ip ? z : (vfft_real *)vfft_aligned_alloc((RN + 8) * sizeof(vfft_real));
    vfft_real *seed = ip ? (vfft_real *)vfft_aligned_alloc((CN + 8) * sizeof(vfft_real)) : NULL;   /* the plane re-laid before every sample */
    _il2d_ip_reset_t rs;
    _il2d_race_ctx_t rc[10];
    vfft_race_arm_t arms[10];
    char names[10][16], tkb_a[16], tkb_w[16], vb[16];
    const char *tk_mtarm = _il2d_tkp(h, c2r ? "mtarm_c2r" : "mtarm", tkb_a, sizeof tkb_a);
    const char *tk_msw = _il2d_tkp(h, c2r ? "msw_c2r" : "msw", tkb_w, sizeof tkb_w);
    double ns[10], t0, est;
    int lad[6], nl = 0, na = 0, i, best = 0, strips_ok, reps, lg = 1;
    size_t q;
    if (!z || !a || (ip && !seed))
    {
        vfft_aligned_free(z); if (!ip) vfft_aligned_free(a); vfft_aligned_free(seed);
        return;
    }
    {   /* the input: a CCE plane for c2r (the DC and Nyquist bins real), a real plane for r2c */
        unsigned sd = 0x9e3779b9u ^ (unsigned)N1 ^ ((unsigned)N2 << 12);
        vfft_real *in = c2r ? z : a;
        const size_t nin = c2r ? CN : RN;
        for (q = 0; q < nin + 8; q++) { sd = sd * 1664525u + 1013904223u; in[q] = (double)(sd >> 8) / (double)(1u << 24) - 0.5; }
        if (c2r)
            for (q = 0; q < (size_t)N1; q++)
                z[q * 2 * h->il2d_col.rn + 1] = z[q * 2 * h->il2d_col.rn + 2 * ((size_t)N2 / 2) + 1] = 0.0;
        if (ip)
        {
            memcpy(seed, z, (CN + 8) * sizeof(vfft_real));
            rs.p = z; rs.seed = seed; rs.n = CN + 8;
        }
    }
    strips_ok = !h->il2d_col.blu && !h->il2d_col.tpc && (!h->il2d_col.nat || h->il2d_col.nst >= 2);
    if (strips_ok)
    {
        nl = _il2d_real_sw_ladder(N1, h->il2d_col.rn, h->nthreads, lad, 6);
        if (h->il2d_col.nat)
            (void)_il2d_real_sscr_build(&h->il2d_col, N1, h->il2d_col.rn, thread_pool_workers_for(h->nthreads));
    }
    h->il2d_col.colmt = 0;
    h->il2d_col.natarm = 0;
    h->il2d_col.msw = 0;
    /* the arms: ctx.lst = the mode (colmt), ctx.nat = the shape (natarm), ctx.sw = the width */
#define _IL2D_CMT_ARM(NAME_, MODE_, SHAPE_, SW_) do { \
        _il2d_race_ctx_t c0 = { h, a, z, !c2r, 1, 0, 0, 0, NULL, NULL, NULL, NULL }; \
        c0.lst = (MODE_); c0.nat = (SHAPE_); c0.sw = (SW_); rc[na] = c0; \
        snprintf(names[na], sizeof names[na], "%s", NAME_); \
        arms[na].name = names[na]; arms[na].run = _il2d_arm_whole; arms[na].ctx = &rc[na]; na++; } while (0)
    _IL2D_CMT_ARM("serial", 0, 0, 0);
    _IL2D_CMT_ARM("threaded", 1, 0, 0);
    if (strips_ok)
    {
        _IL2D_CMT_ARM("strips", 1, 1, 0);
        for (i = 0; i < nl && na < 10; i++)
        {
            char nm[16];
            snprintf(nm, sizeof nm, "strips/sw%d", lad[i]);
            _IL2D_CMT_ARM(nm, 1, 1, lad[i]);
        }
    }
#undef _IL2D_CMT_ARM
    /* a threaded arm whose column pass does not engage on this cell is out (its time would be
     * the serial arm's); the engagement counter says */
    for (i = 1; i < na; i++)
    {
        const long e0 = _vfft_il2d_col_mt_count;
        if (ip) memcpy(z, seed, (CN + 8) * sizeof(vfft_real));
        _il2d_arm_whole(&rc[i]);
        rc[i].ok = _vfft_il2d_col_mt_count != e0;
    }
    /* reps from the serial arm's timing (~0.3 ms of work per sample); in place the plane grows a
     * factor N1 per pass: a sample stays finite */
    if (ip) memcpy(z, seed, (CN + 8) * sizeof(vfft_real));
    _il2d_arm_whole(&rc[0]);
    if (ip) memcpy(z, seed, (CN + 8) * sizeof(vfft_real));
    t0 = vfft_now_ns();
    _il2d_arm_whole(&rc[0]);
    est = vfft_now_ns() - t0;
    reps = (int)(3.0e5 / (est > 1.0 ? est : 1.0));
    if (reps < 1) reps = 1;
    if (reps > 4096) reps = 4096;
    if (ip)
    {
        while ((1 << lg) < N1) lg++;
        if (reps > 900 / lg) reps = 900 / lg;
        if (reps > 32) reps = 32;
        if (reps < 1) reps = 1;
    }
    {
        vfft_race_proto_t proto = { 3, 1, VFFT_RACE_MIN, 0, 2, NULL, NULL, 0 }; /* min-of-3, A then B, two untimed passes per arm first */ /* THREADED arms: never paused (VFFT_RACE_PACE_MS) */
        proto.reps = reps;
        if (ip)
        {   /* the one plane, re-laid before every sample */
            proto.reset = _il2d_ip_reset;
            proto.reset_ctx = &rs;
        }
        _vfft_create_race_count++;
        vfft_race_run(&proto, arms, na, ns);
    }
    for (i = 1; i < na; i++)
        if (rc[i].ok && ns[i] < ns[best])
            best = i;
    h->il2d_col.colmt = best > 0;
    h->il2d_col.natarm = best >= 2 ? 1 : 0;
    h->il2d_col.msw = best >= 2 ? rc[best].sw : 0;
    if (h->il2d_col.natarm != 1 || !h->il2d_col.nat)
        _il2d_nat_sscr_free(&h->il2d_col);   /* the dense scratch serves the strips only */
    vfft_aligned_free(z); if (!ip) vfft_aligned_free(a); vfft_aligned_free(seed);
    if (getenv("VFFT_IL2D_LOG"))
    {
        fprintf(stderr, "[il2d-real] colmt race %s%dx%d T=%d (the whole transform, reps=%d):", c2r ? "c2r " : "", N1, N2, h->nthreads, reps);
        for (i = 0; i < na; i++)
            fprintf(stderr, " %s=%.0f%s", names[i], ns[i], rc[i].ok ? "" : "(no)");
        fprintf(stderr, " -> %s\n", names[best]);
    }
    vw2_2d_rl_bank(&W->vw2, N1, N2, c2r,
                   h->il2d_col.R, h->il2d_col.nst,
                   h->il2d_col.wl, h->il2d_col.colmt, h->nthreads,
                   (N1 & (N1 - 1)) ? h->il2d_col.blu : -1,
                   ns[best], ord, h->nthreads, h->il2d_ip);
    snprintf(vb, sizeof vb, "%d", h->il2d_col.natarm);
    (void)vw2_2d_rl_tok_sets(&W->vw2, N1, N2, ord, h->nthreads, tk_mtarm, vb, h->il2d_ip);
    snprintf(vb, sizeof vb, "%d", h->il2d_col.msw);
    (void)vw2_2d_rl_tok_sets(&W->vw2, N1, N2, ord, h->nthreads, tk_msw, vb, h->il2d_ip);
    _vw2_persist(W, cfg);
}


/* the Bluestein inner's chain at M: replayed from the cell's own row when
 * banked (prod == M), else the chain race at (M, N2), banked on the cell's
 * row. Context = the create in progress (set at the 2D/3D create's entry). */
static struct {
    struct vfft_wisdom_s *W;
    const vfft_cfg_t *cfg;
    size_t N2;   /* the axis's row length in complexes: a lane-major batch's lanes, size_t */
    /* the column-axis Bluestein's INNER chain at M is the
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
#define VFFT_IL2D_RACE_OOM (-2)   /* _il2d_race_chains: its planes could not be allocated (out of memory, recorded) */
static int _il2d_race_chains(int N1, size_t N2, int ncand, int (*cand)[8],
                             const int *lens, double *best_ns, int nat, int banded, int *best_wl, int T,
                             _il2d_lmform_t *lm, size_t np);
static int _il2d_race_forms(int N1, size_t N2, const int *Rs, int nst,
                            vfft_il2p_fn *ff, vfft_il2p_fn *fb, char *forms,
                            size_t fsz, int half, int T, int nat, int lm);
static void _il2d_forms_serve_key(struct vfft_wisdom_s *W,
                                  const vfft_cfg_t *cfg,
                                  const vw2_ilcol_key_t *key, const char *base,
                                  int N, size_t rn,
                                  const int *Rs, int nst,
                                  vfft_il2p_fn *ff, vfft_il2p_fn *fb,
                                  char *forms, size_t fsz, int T, int nat);
static void _il2d_forms_serve(struct vfft_wisdom_s *W,
                              const vfft_cfg_t *cfg, int is_real, int N1,
                              int N2, const int *Rs, int nst,
                              vfft_il2p_fn *ff, vfft_il2p_fn *fb,
                              char *forms, size_t fsz, int ord);
static int _il2d_blu_m_chain(int M, int *Rs, int *nst, char *forms,
                             size_t fsz)
{
    struct vfft_wisdom_s *W = _il2d_blu_ctx.W;
    const vfft_cfg_t *cfg = _il2d_blu_ctx.cfg;
    const size_t N2 = _il2d_blu_ctx.N2;
    const vw2_ilcol_key_t *key = _il2d_blu_ctx.key;
    vfft_il2p_fn ff[8], fb[8];
    forms[0] = 0;
    if (!W || W->vw2_off_2d || N2 == 0 || !key) return 0;
    /* REPLAY -- the M chain THE CELL'S OWN row carries, read by the builder
     * (which checked prod == blu) and handed over here. Never the (M, N2,
     * scr) row: that is a user's own scrambled M x N2 cell, and a bank there
     * (a positive time with every axis verdict at -1 is the REPLACE path of
     * vw2_2d_il_chain_bank whenever the chain differs) would wipe its width,
     * form and column-MT verdicts. */
    if (_il2d_blu_ctx.rep_R && _il2d_blu_ctx.rep_nst > 0 &&
        _il2d_chain_prod(_il2d_blu_ctx.rep_R, _il2d_blu_ctx.rep_nst) == M)
    {
        memcpy(Rs, _il2d_blu_ctx.rep_R,
               (size_t)_il2d_blu_ctx.rep_nst * sizeof(int));
        *nst = _il2d_blu_ctx.rep_nst;
        if (getenv("VFFT_IL2D_LOG"))
            fprintf(stderr, "[il2d] blu inner M=%d x %zu: replay chain src=wisdom\n", M, N2);
        if (_il2d_resolve(Rs, *nst, ff, fb))
            _il2d_forms_serve_key(W, cfg, key, "bluforms", M, N2,
                                  Rs, *nst, ff, fb, forms, fsz, 1, 0);
        return 1;
    }
    {
        int cand[VFFT_IL2D_POOL_MAX][8], lens[VFFT_IL2D_POOL_MAX];
        int cur[8], ncand = 0, dropped = 0, win;
        double bns = 0;
        _il2d_enum_rec(M, 0, cur, cand, lens, &ncand, &dropped);
        if (ncand < 1) return 0;   /* (a capped pool warns from inside the enumerator) */
        win = (ncand > 1) ? _il2d_race_chains(M, N2, ncand, cand, lens, &bns, 0, /*banded=*/0, NULL, 1, NULL, 0) : 0;   /* the inner runs unbanded */
        if (win == VFFT_IL2D_RACE_OOM)
            return -1;   /* out of memory (the race recorded it): no verdict on M's chains */
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
            fprintf(stderr, "[il2d] blu inner M=%d x %zu: chain race -> %d candidates,"
                            " winner %s\n", M, N2, ncand,
                    _il2d_blu_ctx.commit ? "banked on the cell's row"
                                         : "(N-arm arm, unbanked)");
        if (_il2d_resolve(Rs, *nst, ff, fb))
            _il2d_forms_serve_key(W, cfg, key, "bluforms", M, N2,
                                  Rs, *nst, ff, fb, forms, fsz, 1, 0);
        return 1;
    }
}

/* PER-STAGE FORM RACE: coordinate descent over the
 * stages whose radix has rival forms (vfft_il2p_col_forms: the blocked
 * shapes at r32/r64, the leaf's half-store twins), each stage's arms timed
 * on the WHOLE column pass with the other stages held at their current pick
 * (the pass is the only thing the form changes; same harness as the chain
 * race; the plane at the pass's own pitch, so an odd CCE pitch is what the
 * arms meet). The construction-table default is the incumbent and keeps
 * ties; a rival must beat it by 3%. half = the leaf's half-store twins are
 * in the pool (the 2D real r2c plan only: _il2d_forms_serve_key). Installs
 * the winners into ff/fb and spells them into `forms`. Returns 1 when any
 * stage had a choice, 0 otherwise (forms = ""). */
/* THE FORMS THROUGH THE THREADED PASS (2026-10-08): at T > 1 the rank-2 real cell times every
 * form through the pass it serves with -- the natural partition for a natural chain (its own
 * perm, scratch and staging slots), the strips otherwise -- the serial pass standing in where it
 * cannot engage; at one thread the scrambled pass in place, as before. A lane-major batch's
 * cell (lm = 1) serves its natural leaf strided (it has no staging), so its natural arms run
 * without the staging slots. */
static void _il2d_arm_forms_mt(void *v)
{
    _il2d_race_ctx_t *c = (_il2d_race_ctx_t *)v;
    vfft_ilcol_t cv;
    memset(&cv, 0, sizeof cv);
    cv.N = c->N1;
    cv.rn = c->N2;
    cv.nst = c->nst;
    memcpy(cv.R, c->R, (size_t)c->nst * sizeof cv.R[0]);
    memcpy(cv.L, c->Ls, (size_t)c->nst * sizeof cv.L[0]);
    memcpy(cv.f, c->ff, (size_t)c->nst * sizeof cv.f[0]);
    memcpy(cv.tf, c->tf, (size_t)c->nst * sizeof cv.tf[0]);
    cv.nat = c->nat;
    cv.natperm = (int *)c->perm;
    cv.natscr = c->nscr;
    if (!_il2d_cols_mt_desc(&cv, c->N1, c->z, c->z, 0, c->T, c->nat ? c->nstage : NULL, -1))
        _il2d_arm_chain(v);
}
static int _il2d_race_forms(int N1, size_t N2, const int *Rs, int nst,
                            vfft_il2p_fn *ff, vfft_il2p_fn *fb, char *forms,
                            size_t fsz, int half, int Tw, int nat, int lm)
{
    const size_t T = (size_t)N1 * N2;
    const char *pick[8];
    int Ls[8], s, any = 0, off = 0;
    vfft_real *tf[8], *tb[8], *z, *zseed;
    int *perm = NULL;              /* the natural chain's leaf permutation, scratch and staging: the pass as served */
    vfft_real *nscr = NULL, *nstage = NULL;
    const int natpass = Tw > 1 && nat && nst > 1;
    _il2d_ip_reset_t zrs;
    size_t i;
    forms[0] = 0;
    for (s = 0; s < nst; s++)
    {
        const char *nm[VFFT_IL2P_COL_MAXFORMS];
        if (vfft_il2p_col_forms(Rs[s], half && s == nst - 1, nm) > 1)
            any = 1;
        pick[s] = nm[0];
    }
    if (!any)
        return 0;
    z = (vfft_real *)vfft_aligned_alloc(2 * T * sizeof(vfft_real));   /* aligned like every plane the door serves */
    zseed = (vfft_real *)vfft_aligned_alloc(2 * T * sizeof(vfft_real));   /* the arms run in place: the reset's seed */
    if (!z || !zseed)
    {
        vfft_aligned_free(z);
        vfft_aligned_free(zseed);
        return 0;
    }
    for (i = 0; i < 2 * T; i++)
        z[i] = 1.0 + 1e-6 * (double)(i & 1023);
    memcpy(zseed, z, 2 * T * sizeof(vfft_real));
    zrs.p = z;
    zrs.seed = zseed;
    zrs.n = 2 * T;
    if (_il2d_build_tables(N1, nst, Rs, Ls, tf, tb))
    {
        vfft_aligned_free(z);
        vfft_aligned_free(zseed);
        return 0;
    }
    if (natpass)
    {
        perm = _il2d_nat_perm(Rs, nst, N1);
        nscr = (vfft_real *)vfft_aligned_alloc(2 * T * sizeof(vfft_real));
        if (!lm)
            nstage = _il2d_stage_alloc(Tw, 64, N2);   /* R_last <= 64 */
        if (!perm || !nscr || (!lm && !nstage))
        {   /* no natural pass to time: the scrambled strips stand in */
            free(perm); vfft_aligned_free(nscr); _il2d_stage_free(nstage);
            perm = NULL; nscr = NULL; nstage = NULL;
        }
    }
    for (s = 0; s < nst; s++)
    {
        const char *nm[VFFT_IL2P_COL_MAXFORMS], *an[VFFT_IL2P_COL_MAXFORMS];
        const int last = (s == nst - 1);
        vfft_il2p_fn ffa[VFFT_IL2P_COL_MAXFORMS][8], fba[VFFT_IL2P_COL_MAXFORMS][8];
        _il2d_race_ctx_t rc[VFFT_IL2P_COL_MAXFORMS];
        vfft_race_arm_t arm[VFFT_IL2P_COL_MAXFORMS];
        vfft_race_proto_t proto = { 3, 1, VFFT_RACE_MIN, 0, 0, _il2d_ip_reset, &zrs, 1 }; /* the plane re-laid before every sample */ /* single-thread arms: paced (VFFT_RACE_PACE_MS) */
        double ns[VFFT_IL2P_COL_MAXFORMS];
        if (Tw > 1)
        {   /* THREADED arms: two untimed passes per arm first, never paused (a paced pool parks its workers) */
            proto.warm = 2;
            proto.pace = 0;
        }
        const int nf = vfft_il2p_col_forms(Rs[s], half && last, nm);
        int f, na = 0, win = 0;
        if (nf < 2)
            continue;
        for (f = 0; f < nf; f++)
        {   /* a form without a kernel pair at this stage is no arm (the
             * backward of a half-store leaf form is its full-store twin) */
            memcpy(ffa[na], ff, sizeof ffa[na]);
            memcpy(fba[na], fb, sizeof fba[na]);
            ffa[na][s] = last ? vfft_il2p_n1c_form_fn(Rs[s], nm[f], 0)
                              : vfft_il2p_t2c_form_fn(Rs[s], nm[f], 0);
            fba[na][s] = last ? vfft_il2p_n1c_form_fn(Rs[s], nm[f], 1)
                              : vfft_il2p_t2c_form_fn(Rs[s], nm[f], 1);
            if (!ffa[na][s] || !fba[na][s])
                continue;
            memset(&rc[na], 0, sizeof rc[na]);
            rc[na].z = z;
            rc[na].ok = 1;
            rc[na].N1 = N1;
            rc[na].N2 = N2;
            rc[na].nst = nst;
            rc[na].R = Rs;
            rc[na].Ls = Ls;
            rc[na].ff = ffa[na];
            rc[na].tf = tf;
            rc[na].nat = perm != NULL;
            rc[na].perm = perm;
            rc[na].nscr = nscr;
            rc[na].nstage = nstage;
            rc[na].T = Tw;
            an[na] = nm[f];
            arm[na].name = nm[f];
            arm[na].run = Tw > 1 ? _il2d_arm_forms_mt : _il2d_arm_chain;
            arm[na].ctx = &rc[na];
            na++;
        }
        if (na < 2 || strcmp(an[0], nm[0]))
            continue;   /* no rival, or the incumbent itself did not build */
        proto.reps = _il2d_race_reps(arm[0].run, arm[0].ctx, (double)N1);   /* sized on the incumbent */
        vfft_race_run(&proto, arm, na, ns);
        for (f = 1; f < na; f++)   /* a rival takes the stage past the race's margin of the incumbent and of the best so far */
            if (vfft_race_beats(ns[f], ns[0], VFFT_RACE_HYST) && ns[f] < ns[win])
                win = f;
        pick[s] = an[win];
        ff[s] = ffa[win][s];
        fb[s] = fba[win][s];
        if (getenv("VFFT_IL2D_LOG"))
        {
            fprintf(stderr, "[il2d] forms %dx%zu stage %d r%d%s:", N1, N2, s, Rs[s],
                    Tw > 1 ? (perm ? " (the threaded natural pass)" : " (the threaded pass)") : "");
            for (f = 0; f < na; f++)
                fprintf(stderr, " %s %.0f ns%s", an[f], ns[f], f == win ? "*" : "");
            fprintf(stderr, "\n");
        }
    }
    for (s = 0; s < nst; s++)
    {
        vfft_aligned_free(tf[s]);
        vfft_aligned_free(tb[s]);
    }
    vfft_aligned_free(z);
    vfft_aligned_free(zseed);
    free(perm);
    vfft_aligned_free(nscr);
    _il2d_stage_free(nstage);
    for (s = 0; s < nst && off < (int)fsz - 8; s++)
        off += snprintf(forms + off, fsz - off, "%s%s", s ? "." : "", pick[s]);
    return 1;
}

/* the FORM axis at create: env pin > the banked forms= on the chain row >
 * the per-stage race, banked on that row (wisdom off / no choice: the
 * construction-table defaults stand). ff/fb come in resolved (defaults). */
static void _il2d_forms_serve(struct vfft_wisdom_s *W,
                              const vfft_cfg_t *cfg, int is_real, int N1,
                              int N2, const int *Rs, int nst,
                              vfft_il2p_fn *ff, vfft_il2p_fn *fb,
                              char *forms, size_t fsz, int ord)
{   /* ONE body (policy R5): vw2_2d_forms_lookup/bank are themselves
     * wrappers that build this key and call the ilcol pair, so the 2D-key
     * spelling of the precedence law (env pin > banked forms= > the
     * per-stage race) is the ilcol-key spelling with this key. */
    const vw2_ilcol_key_t ck = { 2, N1, N2, 0, ord, 0, is_real };
    _il2d_forms_serve_key(W, cfg, &ck, "forms", N1, (size_t)N2, Rs, nst, ff, fb, forms, fsz, 1, 0);
}

/* one HEAT of the chain race: the candidates idx[0..n-1] (n <= the heat's
 * chain count) as the arms of ONE race. Every buildable candidate's tables and
 * (natural) permutation are built up front, then vfft_race_run samples every
 * arm once per round with the rounds alternating direction -- the house
 * protocol -- so a drift of the host (its throttled state lasts minutes) hits
 * all chains alike (timing each chain in its own burst, one after another, let
 * two cold runs at 1215x243 natural bank different chains). Min of 3 samples
 * per arm, each ~0.3 ms of executes (_il2d_race_reps). A candidate's arms are its SERVING forms (banded = 1): the
 * unbanded walk and the banded walk at every width the band law admits for it,
 * each out of place z -> zo; its score is its best form (banded = 0: the
 * unbanded in-place arm alone, the Bluestein inner's). The arms of a heat stay
 * within VFFT_RACE_MAX_ARMS: the caller sizes its heats by the widths. A
 * LANE-MAJOR batch's cell (lm != NULL, _il2d_lmform_t) is never served banded
 * or with a staged leaf: its arms are the serial forms its door serves the
 * candidate with, each through the door's own pass on a plan carrying the
 * candidate (_lm_chain_arm), in the cell's placement and direction (in place:
 * the plane re-laid from zseed before every sample); its score is its best
 * form, and lm->wc names the winner's serial form. At T > 1 a candidate the
 * tile serves is also timed as the tile on T threads (its cost turns on the
 * chain); the lane-major race (_lm_race) decides the threaded form after, on
 * the winning chain.
 * The winner's candidate index, -1 when none builds; *wns its ns, *wwl its
 * width. */
static int _il2d_race_heat(int N1, size_t N2, int (*cand)[8], const int *lens, const int *idx, int n,
                           int nat, vfft_real *z, const vfft_real *zseed, vfft_real *nscr, vfft_real *nstage, double *wns,
                           vfft_real *zo, int banded, int *wwl, int T, _il2d_lmform_t *lm, size_t zn, size_t np)
{   /* zseed: z's seed, re-laid before every sample where the arms run in place (banded = 0, or a lane-major cell in
     * place); zn: the planes' size in doubles; np: the pitch the serial natural arms lay nscr and the staging at
     * (_il2d_key_np; 0 = N2) */
    struct
    {
        vfft_il2p_fn ff[8], fb[8];
        int Ls[8];
        vfft_real *tf[8], *tb[8];
        int *perm;
        int ci;
        struct vfft_plan_s *lmh;   /* a lane-major cell: the plan carrying this candidate (_lm_chain_plan) */
    } cb[VFFT_IL2D_HEAT];
    _il2d_race_ctx_t rc[VFFT_RACE_MAX_ARMS];
    vfft_race_arm_t arms[VFFT_RACE_MAX_ARMS];
    double ns[VFFT_RACE_MAX_ARMS];
    int armc[VFFT_RACE_MAX_ARMS];   /* the arm's chain slot */
    int na = 0, nc = 0, a, s2, k, win = -1;
    *wns = 1e300;
    if (wwl) *wwl = 0;
    for (k = 0; k < n && nc < VFFT_IL2D_HEAT; k++)
    {
        const int ci = idx[k];
        int *perm = NULL;
        int wlc[14], nwl = 0, w, tarm = 0;
        if (!_il2d_resolve(cand[ci], lens[ci], cb[nc].ff, cb[nc].fb))
            continue;
        if (nat && lens[ci] > 1)
        {
            perm = _il2d_nat_perm(cand[ci], lens[ci], N1);
            if (!perm)
                continue; /* no natural leaf for this chain: not a candidate */
        }
        if (_il2d_build_tables(N1, lens[ci], cand[ci], cb[nc].Ls, cb[nc].tf, cb[nc].tb))
        {
            free(perm);
            continue;
        }
        cb[nc].lmh = NULL;
        if (lm)
        {   /* the serial forms the door serves this chain with, on a plan carrying it; a chain the door cannot
             * carry (the real batch's fold order does not derive for it) is no candidate */
            cb[nc].lmh = _lm_chain_plan(lm, N1, N2, lens[ci], cand[ci], cb[nc].Ls, cb[nc].ff, cb[nc].fb,
                                        cb[nc].tf, cb[nc].tb, nat && perm != NULL, perm, nscr);
            if (cb[nc].lmh)
                nwl = _lm_chain_forms(lm, N1, N2, cand[ci], nat && perm != NULL, wlc);
            /* AT T > 1 THE CANDIDATE AS THE TILE TOO (2026-11-17): the tile's cost turns on the chain (its first radix
             * is the sub-problems the workers share: 4096 x 8 at 8 threads, 4.4.32.8 the serial forms' pick, the tile
             * on it 24.9 us against 17.9 on 8.8.8.8), so a chain is timed in the threaded form the door serves it with
             * as well as in its serial forms; the lane-major race then decides the form on the winner */
            tarm = cb[nc].lmh && T > 1 && _il2d_col_tile_serves(&cb[nc].lmh->il2d_col);
        }
        else if (banded == 1)   /* banded = 2: the serving forms without widths (the pruning law, policy_il.h) */
            nwl = _il2d_race_widths(N1, N2, lens[ci], cb[nc].Ls, wlc, 14);
        if ((lm && !cb[nc].lmh) || na + (lm ? 0 : 1) + nwl + tarm > VFFT_RACE_MAX_ARMS)
        {   /* no plan carries it, or the heat is full: this chain waits for the next (the caller sizes heats so this
             * is rare) */
            const int full = !lm || cb[nc].lmh;
            for (s2 = 0; s2 < lens[ci]; s2++)
            {
                vfft_aligned_free(cb[nc].tf[s2]);
                vfft_aligned_free(cb[nc].tb[s2]);
            }
            free(perm);
            _lm_chain_plan_free(cb[nc].lmh);
            if (full)
                break;
            continue;
        }
        cb[nc].perm = perm;
        cb[nc].ci = ci;
        for (w = lm ? 0 : -1; w < nwl; w++)
        {   /* w = -1: the unbanded arm; then every admitted width (a lane-major cell: its serial forms) */
            _il2d_race_ctx_t c0 = { NULL, NULL, z, 0, 1, N1, (size_t)N2,
                                    lens[ci], cand[ci], cb[nc].Ls, cb[nc].ff, cb[nc].tf,
                                    nat && perm != NULL, perm, nscr, 0, nstage };
            c0.zo = zo;
            if (lm)
            {
                c0.h = cb[nc].lmh;
                c0.wc = wlc[w];
            }
            else
            {
                c0.wl = w < 0 ? 0 : wlc[w];
                c0.cut = w < 0 ? 0 : vfft_policy_il2d_wl_cut(N1, lens[ci], cb[nc].Ls, wlc[w]);
            }
            c0.T = T;
            c0.np = np;
            rc[na] = c0;
            arms[na].name = "chain";
            arms[na].run = lm ? _lm_chain_arm   /* the door's serial form */
                         : (T > 1 && banded) ? _il2d_arm_chain_mt   /* the threaded pass the cell serves with */
                         : w < 0 ? (banded ? _il2d_arm_chain_oop : _il2d_arm_chain) : _il2d_arm_chain_banded;
            arms[na].ctx = &rc[na];
            armc[na] = nc;
            na++;
        }
        if (tarm)
        {   /* the tile arm: the candidate's plan on T threads (its serial lane tile irrelevant there) */
            _il2d_race_ctx_t c0 = { NULL, NULL, z, 0, 1, N1, (size_t)N2,
                                    lens[ci], cand[ci], cb[nc].Ls, cb[nc].ff, cb[nc].tf,
                                    nat && perm != NULL, perm, nscr, 0, nstage };
            c0.zo = zo;
            c0.h = cb[nc].lmh;
            c0.wc = 0;
            c0.T = T;
            c0.lmtile = 1;
            rc[na] = c0;
            arms[na].name = "chain";
            arms[na].run = _lm_chain_arm;
            arms[na].ctx = &rc[na];
            armc[na] = nc;
            na++;
        }
        nc++;
    }
    if (na > 0)
    {
        vfft_race_proto_t proto = { 3, 1, VFFT_RACE_MIN, 1, 0, NULL, NULL, 1 }; /* min-of-3, alternated */ /* single-thread arms: paced (VFFT_RACE_PACE_MS) */
        _il2d_ip_reset_t hrs = { z, zseed, lm ? zn : 2 * (size_t)N1 * (size_t)N2 };
        int best, anytile = 0;
        for (a = 0; a < na; a++)
            anytile |= rc[a].lmtile;
        if ((T > 1 && banded && !lm) || anytile)
        {   /* THREADED arms: two untimed passes per arm first, never paused (a paced pool parks its workers) */
            proto.warm = 2;
            proto.pace = 0;
        }
        if ((!banded || lm) && zseed)
        {   /* the in-place arms walk their own output */
            proto.reset = _il2d_ip_reset;
            proto.reset_ctx = &hrs;
        }
        /* sized on the heat's first arm; an in-place arm grows the plane by the length it transforms (the 1D real
         * batch's N = 2 M) */
        proto.reps = _il2d_race_reps(arms[0].run, arms[0].ctx,
                                     lm ? (lm->ip ? (lm->kind == 1 ? 2.0 * N1 : (double)N1) : 0.0)
                                        : banded ? 0.0 : (double)N1);
        best = vfft_race_run(&proto, arms, na, ns);
        if (best >= 0)
        {
            win = cb[armc[best]].ci;
            *wns = ns[best];
            if (wwl) *wwl = rc[best].wl;
            if (lm)
            {   /* the winner's serial lane tile: the arm's, or where its tile arm won, that chain's best serial form */
                int bs = best;
                if (rc[best].lmtile)
                {
                    bs = -1;
                    for (a = 0; a < na; a++)
                        if (armc[a] == armc[best] && !rc[a].lmtile && (bs < 0 || ns[a] < ns[bs]))
                            bs = a;
                }
                lm->wc = bs >= 0 ? rc[bs].wc : 0;
            }
        }
        if (getenv("VFFT_IL2D_LOG"))
        {
            fprintf(stderr, "[il2d] chain race %dx%zu (%s%s): %d chain(s), %d arm(s)", N1, N2,
                    nat ? "nat" : "scr",
                    lm ? (lm->ip ? (anytile ? ", the lane-major serial forms and the tile, in place"
                                            : ", the lane-major serial forms, in place")
                                 : (anytile ? ", the lane-major serial forms and the tile" : ", the lane-major serial forms"))
                       : (T > 1 && banded) ? ", the threaded pass" : "", nc, na);
            for (a = 0; a < na; a++)
            {
                int q;
                fprintf(stderr, " ");
                for (q = 0; q < lens[cb[armc[a]].ci]; q++)
                    fprintf(stderr, "%s%d", q ? "." : "", cand[cb[armc[a]].ci][q]);
                if (lm && rc[a].lmtile)
                    fprintf(stderr, "/tile");
                else if (lm)
                    fprintf(stderr, "/sw%d", rc[a].wc);
                else if (rc[a].wl > 0)
                    fprintf(stderr, "/wl%d", rc[a].wl);
                fprintf(stderr, "=%.0f", ns[a]);
            }
            fprintf(stderr, " -> arm %d\n", best);
        }
    }
    for (a = 0; a < nc; a++)
    {
        for (s2 = 0; s2 < lens[cb[a].ci]; s2++)
        {
            vfft_aligned_free(cb[a].tf[s2]);
            vfft_aligned_free(cb[a].tb[s2]);
        }
        free(cb[a].perm);
        _lm_chain_plan_free(cb[a].lmh);
    }
    return win;
}

/* the chain RACE over the whole pool (owner, 2026-10-03: complete, in heats):
 * balanced heats of at most VFFT_IL2D_HEAT chains, the heat winners raced
 * again the same way until one stands -- for every pool in reach the heats
 * and one final. Returns the winner's index (its ns in *best_ns, from the
 * last race it ran), -1 = no arm built, VFFT_IL2D_RACE_OOM = its planes could
 * not be allocated: OUT OF MEMORY, the reason recorded, nothing raced. */
static int _il2d_race_chains(int N1, size_t N2, int ncand, int (*cand)[8],
                             const int *lens, double *best_ns, int nat, int banded, int *best_wl, int Tw,
                             _il2d_lmform_t *lm, size_t np)
{   /* lm: a lane-major batch's cell, every candidate in the serial forms its door serves it with, in the cell's
     * placement (_il2d_race_heat); the winner's lane tile comes back in lm->wc. NULL: every other cell. N2: the
     * lanes, size_t. np: the pitch the cell's serial walk lays the natural pass's own buffers at (_il2d_key_np;
     * 0 = N2): nscr's rows are sized for it */
    const size_t T = (size_t)N1 * N2;
    /* the planes, in doubles: a lane-major 1D real batch's arm runs its executor, whose spectrum side is M + 1
     * rows of K bins (2 K doubles past the N K reals) */
    const size_t zn = 2 * T + ((lm && lm->kind == 1) ? 2 * (size_t)N2 : 0);
    const int lmip = lm && lm->ip;   /* a lane-major cell in place: one plane, re-laid from its seed */
    const size_t nslot = (size_t)(Tw > 1 ? Tw : 1);   /* the threaded arms' staging: a slot per worker */
    vfft_real *z = (vfft_real *)vfft_aligned_alloc(zn * sizeof(vfft_real));   /* aligned like every plane the door serves */
    vfft_real *zo = !banded ? NULL : lmip ? z : (vfft_real *)vfft_aligned_alloc(zn * sizeof(vfft_real));   /* the serving forms' destination plane */
    vfft_real *zseed = (banded && !lmip) ? NULL : (vfft_real *)vfft_aligned_alloc(zn * sizeof(vfft_real));   /* the arms run in place: the reset's seed */
    vfft_real *nscr = nat ? (vfft_real *)vfft_aligned_alloc(2 * (size_t)N1 * VFFT_IL2D_NAT_CAP(N2) * sizeof(vfft_real)) : NULL;   /* aligned like every plane the door serves */
    vfft_real *nstage = (nat && !lm) ? _il2d_stage_alloc((int)nslot, 64, N2) : NULL;   /* R_last <= 64 */
    int *surv = (int *)malloc((size_t)(ncand > 0 ? ncand : 1) * sizeof(int));
    int *next = (int *)malloc((size_t)(ncand > 0 ? ncand : 1) * sizeof(int));
    /* the heat's chain count: VFFT_IL2D_HEAT, or what the arm budget holds when
     * every chain brings its widths (the unbanded arm + at most 14 widths; a
     * lane-major chain its serial forms: the walk or the whole plane + at most
     * the serial ladder's widths) */
    int lad[6];
    const int lmn = lm ? 1 + _il2d_sw_ladder(N1, N2, 1, lad, 6) + (Tw > 1) : 1;   /* + the tile arm at T > 1 */
    const int heat = lm ? (VFFT_RACE_MAX_ARMS / lmn < VFFT_IL2D_HEAT ? VFFT_RACE_MAX_ARMS / lmn : VFFT_IL2D_HEAT)
                   : banded == 1 ? (VFFT_RACE_MAX_ARMS / 15 < VFFT_IL2D_HEAT ? VFFT_RACE_MAX_ARMS / 15 : VFFT_IL2D_HEAT) : VFFT_IL2D_HEAT;
    int ns = ncand, win = -1, wwl = 0, i, rounds = 0, heats = 0;
    double wns = 1e300;
    size_t q;
    if (best_wl) *best_wl = 0;
    if (!z || (banded && !zo) || ((!banded || lmip) && !zseed) || (nat && !nscr) || (nat && !lm && !nstage) ||
        !surv || !next || (lm && !_lm_chain_open(lm, N1, N2, nat)))
    {   /* OUT OF MEMORY: the planes the arms run on (z, its destination or its seed, nscr when natural; a lane-major
         * cell's door scratch, and the 1D real batch's plane and twiddles) and the staging -- refused with that
         * cause, nothing raced (a race without the staging would time a form the cell does not serve) */
        const double mb = (2.0 * (double)zn + (nat ? 2.0 * (double)T : 0.0) + ((lm && lm->kind == 1) ? 2.0 * (double)T : 0.0)) *
                              (double)sizeof(vfft_real) / 1048576.0 +
                          ((nat && !lm) ? (double)nslot * (double)vfft_apart_stride(_il2d_stage_slot_bytes(64, N2)) / 1048576.0 : 0.0);
        vfft_aligned_free(z);
        if (zo != z) vfft_aligned_free(zo);
        vfft_aligned_free(zseed);
        vfft_aligned_free(nscr);
        _il2d_stage_free(nstage);
        free(surv);
        free(next);
        _vfft_warn("vfft_create: IL column pass %dx%zu: out of memory (measuring its chains needs %.1f MB of planes)",
                   N1, N2, mb);
        return VFFT_IL2D_RACE_OOM;
    }
    for (q = 0; q < zn; q++)
        z[q] = 1.0 + 1e-6 * (double)(q & 1023);
    if (zseed)
        memcpy(zseed, z, zn * sizeof(vfft_real));
    for (i = 0; i < ncand; i++)
        surv[i] = i;
    while (ns > 0)
    {
        const int nh = (ns + heat - 1) / heat;
        int h, nn = 0;
        rounds++;
        heats += nh;
        for (h = 0; h < nh; h++)
        {
            const int lo = (int)((long)ns * h / nh), hi = (int)((long)ns * (h + 1) / nh);
            double hns;
            int hwl = 0;
            const int w = _il2d_race_heat(N1, N2, cand, lens, surv + lo, hi - lo, nat, z, zseed, nscr, nstage, &hns,
                                          zo, banded, &hwl, Tw, lm, zn, np);
            if (w >= 0)
            {
                next[nn++] = w;
                win = w;
                wns = hns;
                wwl = hwl;
            }
        }
        if (nh == 1)
            break;   /* one heat: its winner is the race's */
        memcpy(surv, next, (size_t)nn * sizeof(int));
        ns = nn;
        win = -1;
        wns = 1e300;
    }
    if (getenv("VFFT_IL2D_LOG") && heats > 1)
        fprintf(stderr, "[il2d] chain race %dx%zu (%s): %d chain(s) in %d heat(s) over %d round(s)\n",
                N1, N2, nat ? "nat" : "scr", ncand, heats, rounds);
    if (best_wl) *best_wl = wwl;
    if (lm)
        _lm_chain_close(lm);
    vfft_aligned_free(z);
    if (zo != z) vfft_aligned_free(zo);
    vfft_aligned_free(zseed);
    vfft_aligned_free(nscr);
    _il2d_stage_free(nstage);
    free(surv);
    free(next);
    *best_ns = wns;
    return win;
}



typedef struct
{
    vfft_real *sc;
    int N1;
    size_t N2;
    int nst;
    const int *R;
    int *L;
    vfft_il2p_fn *f;
    vfft_real **tf;
    int M2, bnst;
    int *bR, *bL;
    vfft_il2p_fn *bf, *bb;
    vfft_real **btf, **btb;
    vfft_real *bchf, *bkf, *bscr;
    /* NATURAL cells: the chain arm must be timed under the serving it
     * will run - the leaf-redirected natural pass (scratch round trip +
     * strided scatter), never the scrambled pass (a strawman arm). */
    int nat;
    const int *perm;
    vfft_real *nscr;
    int T;   /* the cell's T: above 1 every arm is timed through the threaded pass (_il2d_n1arm_mt); 0 or 1 serial */
    size_t np;   /* the row pitch of nscr in the serial natural arm, as the cell's serial walk lays it (_il2d_key_np); 0 = N2 */
} _il2d_n1arm_ctx_t;
/* THE N-ARM ARMS AT T > 1 (2026-11-18, the threading census): the cell serves its column pass threaded at T, so
 * the chain and the Bluestein arms run through the threaded pass it serves with (_il2d_cols_mt_desc over a view of
 * the arm, as the chain race times its candidates, _il2d_arm_chain_mt); 0 = the arm cannot engage at T and the
 * caller runs its serial pass. The serial races crowned the turned prime pass, which has no threaded form, at
 * 4099 x 256 and 4483 x 64 lanes at every T, where the Rader chain threaded runs several times faster. */
static int _il2d_n1arm_mt(const _il2d_n1arm_ctx_t *c, int blu)
{
    vfft_ilcol_t cv;
    if (c->T < 2)
        return 0;
    memset(&cv, 0, sizeof cv);
    cv.N = c->N1;
    cv.rn = c->N2;
    if (blu)
    {   /* the Bluestein pipeline over each worker's lanes (_il2d_blu_cols_range: windows share the scratch
         * disjointly), the forward chirp and kernel */
        cv.blu = c->M2;
        cv.nst = c->bnst;
        memcpy(cv.R, c->bR, (size_t)c->bnst * sizeof cv.R[0]);
        memcpy(cv.L, c->bL, (size_t)c->bnst * sizeof cv.L[0]);
        memcpy(cv.f, c->bf, (size_t)c->bnst * sizeof cv.f[0]);
        memcpy(cv.b, c->bb, (size_t)c->bnst * sizeof cv.b[0]);
        memcpy(cv.tf, c->btf, (size_t)c->bnst * sizeof cv.tf[0]);
        memcpy(cv.tb, c->btb, (size_t)c->bnst * sizeof cv.tb[0]);
        cv.bluchf = c->bchf;
        cv.blukf = c->bkf;
        cv.bluscr = c->bscr;
    }
    else
    {
        cv.nst = c->nst;
        memcpy(cv.R, c->R, (size_t)c->nst * sizeof cv.R[0]);
        memcpy(cv.L, c->L, (size_t)c->nst * sizeof cv.L[0]);
        memcpy(cv.f, c->f, (size_t)c->nst * sizeof cv.f[0]);
        memcpy(cv.tf, c->tf, (size_t)c->nst * sizeof cv.tf[0]);
        cv.nat = c->nat;
        cv.natperm = (int *)c->perm;
        cv.natscr = c->nscr;
    }
    return _il2d_cols_mt_desc(&cv, c->N1, c->sc, c->sc, 0, c->T, NULL, -1);
}
static void _il2d_n1arm_chain(void *v)
{
    _il2d_n1arm_ctx_t *c = (_il2d_n1arm_ctx_t *)v;
    if (_il2d_n1arm_mt(c, 0))
        return;
    if (c->nat)
        _il2d_col_pass_nat(c->sc, c->sc, c->N1, c->N2, c->np ? c->np : c->N2, c->nst, c->R, c->L,
                           c->f, c->tf, 0, c->perm, c->nscr, NULL);
    else
        _il2d_col_pass(c->sc, c->sc, c->N1, c->N2, c->N2, c->nst, c->R,
                       c->L, c->f, c->tf, 0);
}
static void _il2d_n1arm_blu(void *v)
{
    _il2d_n1arm_ctx_t *c = (_il2d_n1arm_ctx_t *)v;
    if (_il2d_n1arm_mt(c, 1))
        return;
    _il2d_blu_cols(c->sc, c->sc, c->N1, c->N2, c->M2, c->bnst, c->bR, c->bL,
                   c->bf, c->bb, c->btf, c->btb, c->bchf, c->bkf, c->bscr);
}

/* free everything a column-axis pass owns (tables, natural, Bluestein,
 * the staged scratch); the descriptor is zero afterwards */
static void _il2d_tpc_drop(vfft_ilcol_t *c);
/* the strips' DENSE per-worker scratch: T blocks of N rows of _il2d_scr_cap(swcap)
 * complexes, swcap = the widest strip a worker runs (the ladder's widest
 * admissible width, and the unsized whole range when it fits L2) */
static int _il2d_sw_ladder(int N1, size_t N2, int T, int *out, int max);
static void _il2d_nat_sscr_free(vfft_ilcol_t *c)
{
    int t;
    if (c->natsscr)
    {
        for (t = 0; t < c->nnatsscr; t++)
            vfft_aligned_free(c->natsscr[t]);
        free(c->natsscr);
    }
    c->natsscr = NULL;
    c->nnatsscr = 0;
    c->natswcap = 0;
}
static int _il2d_nat_sscr_build(vfft_ilcol_t *c, int N1, size_t N2, int T)
{   /* N2: the columns, size_t (a lane-major batch's lanes); a width the L2 holds is an int */
    int lad[6], nl, k, swcap = 0, t;
    if (!c->nat || T < 2 || c->natsscr)
        return c->natsscr != NULL;
    nl = _il2d_sw_ladder(N1, N2, T, lad, 6);
    for (k = 0; k < nl; k++)
        if (lad[k] > swcap) swcap = lad[k];
    {   /* the unsized walk (msw 0) runs a worker's whole block: the blocks split N2 as N2 t / T, so the
         * widest is N2 / T rounded up (100 columns on 8 workers: 13; sized down, those workers left
         * the dense scratch for the in-place walk, 2-4x slower) */
        const size_t wmax = N2 / (size_t)T + (N2 % (size_t)T != 0);
        if (wmax >= 2 && vfft_policy_fits_l2_lanes((size_t)N1, wmax) && wmax > (size_t)swcap)
            swcap = (int)wmax;
    }
    if (swcap < 2)
        return 0;
    c->natsscr = (vfft_real **)calloc((size_t)T, sizeof *c->natsscr);
    if (!c->natsscr)
        return 0;
    for (t = 0; t < T; t++)
    {
        c->natsscr[t] = (vfft_real *)vfft_aligned_alloc(2 * (size_t)N1 * _il2d_scr_cap((size_t)swcap) * sizeof(vfft_real));
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
         * phase start would serialise on the CRT's environment lock */
        const char *dpin = getenv("VFFT_IL2D_DENSE");
        c->natdense = !(dpin && atoi(dpin) == 0);
    }
    return 1;
}
/* the ROWS-FIRST walk's strip scratch (one thread, _il2d_rfw_exec): ONE block of N1 rows of _il2d_scr_cap(w)
 * complexes, w the widest strip the plan runs; a block that already holds that width stays. 0 = out of memory. */
static int _il2d_rfw_scr_build(vfft_ilcol_t *c, int N1, int w)
{
    if (!c->nat || w < 1)
        return 0;
    if (c->natsscr && c->nnatsscr >= 1 && c->natswcap >= w)
        return 1;
    _il2d_nat_sscr_free(c);
    c->natsscr = (vfft_real **)calloc(1, sizeof *c->natsscr);
    if (!c->natsscr)
        return 0;
    c->natsscr[0] = (vfft_real *)vfft_aligned_alloc(2 * (size_t)N1 * _il2d_scr_cap((size_t)w) * sizeof(vfft_real));
    if (!c->natsscr[0])
    {
        free(c->natsscr);
        c->natsscr = NULL;
        return 0;
    }
    c->nnatsscr = 1;
    c->natswcap = w;
    c->natdense = 1;
    return 1;
}
/* the real tier's strip-width ladder: the column count is odd (hp1) or a padded width, so no
 * divisor law -- every width under a worker's column block that keeps N1 x w x 16 B in L2; the
 * last sub-strip is narrower */
static int _il2d_real_sw_ladder(int N1, size_t rn, int T, int *out, int max)
{
    static const int SW[] = { 8, 16, 32, 64, 128, 256 };
    const size_t per = rn / (size_t)(T > 1 ? T : 1);
    int i, n = 0;
    for (i = 0; i < 6 && n < max; i++)
        if ((size_t)SW[i] < per && vfft_policy_fits_l2((long)N1 * SW[i] * VFFT_CX_BYTES))
            out[n++] = SW[i];
    return n;
}
/* the real tier's dense strip scratch: T blocks of N1 rows of _il2d_scr_cap(swcap) complexes, swcap =
 * the widest ladder width, or a worker's whole block where that fits L2 */
static int _il2d_real_sscr_build(vfft_ilcol_t *c, int N1, size_t rn, int T)
{
    int lad[6], nl, k, swcap = 0, t;
    const size_t per = rn / (size_t)(T > 1 ? T : 1);
    if (!c->nat || T < 2 || c->natsscr)
        return c->natsscr != NULL;
    nl = _il2d_real_sw_ladder(N1, rn, T, lad, 6);
    for (k = 0; k < nl; k++)
        if (lad[k] > swcap) swcap = lad[k];
    if (per >= 2 && vfft_policy_fits_l2((long)N1 * (long)per * VFFT_CX_BYTES) && (int)per > swcap)
        swcap = (int)per;
    if (swcap < 2)
        return 0;
    c->natsscr = (vfft_real **)calloc((size_t)T, sizeof *c->natsscr);
    if (!c->natsscr)
        return 0;
    for (t = 0; t < T; t++)
    {
        c->natsscr[t] = (vfft_real *)vfft_aligned_alloc(2 * (size_t)N1 * _il2d_scr_cap((size_t)swcap) * sizeof(vfft_real));
        if (!c->natsscr[t])
        {
            c->nnatsscr = t;
            _il2d_nat_sscr_free(c);
            return 0;
        }
    }
    c->nnatsscr = T;
    c->natswcap = swcap;
    {   /* the probe pin, read ONCE here (no getenv on the execute path) */
        const char *dpin = getenv("VFFT_IL2D_DENSE");
        c->natdense = !(dpin && atoi(dpin) == 0);
    }
    return 1;
}
static void _il2d_col_free(vfft_ilcol_t *c)
{
    int s;
    if (c->rd0)
    {   /* the Rader stage's engine; its tags in the stage-0 slots are this set's blocks,
         * freed with the tables below */
        _ilrd_col0_ctx_clear(c->rd0);
        if (_il2d_rd0_free_hook) _il2d_rd0_free_hook(c->rd0);
        c->rd0 = NULL;
    }
    for (s = 0; s < c->nst && s < 8; s++)
    {
        vfft_aligned_free(c->tf[s]);
        vfft_aligned_free(c->tb[s]);
    }
    free(c->natperm);
    vfft_aligned_free(c->natscr);
    _il2d_nat_sscr_free(c);
    vfft_aligned_free(c->bluchf);
    vfft_aligned_free(c->bluchb);
    vfft_aligned_free(c->blukf);
    vfft_aligned_free(c->blukb);
    vfft_aligned_free(c->bluscr);
    vfft_aligned_free(c->bandscr);
    _il2d_tpc_drop(c);
    memset(c, 0, sizeof *c);
}

/* THE RADER STAGE on a column descriptor whose chain starts with a prime above
 * the pool (R[0] >= VFFT_IL2D_RADER_MIN, docs/roadmap/rader_leaf_design.md
 * section 13): the engine is built once the chain is settled and BEFORE any
 * table set for it (the descriptor's, the races'): every such set names it
 * through the tag blocks _il2d_build_tables takes from the build's engine
 * context. T scratch slots. 1 = ok (or no such stage), 0 = it could not be
 * built. */
static int _il2d_col_rd0_attach(struct vfft_wisdom_s *W, const vfft_cfg_t *cfg,
                                const vw2_ilcol_key_t *key, vfft_ilcol_t *c, int p, int N, size_t rn, int T,
                                int *replayed)
{
    int req[8], nreq = 0, Wr = 0;
    char rd[64] = "", rdw[16] = "";
    if (replayed) *replayed = 0;
    if (!VFFT_IL2D_RADER_RADIX(p) || c->rd0)
        return 1;
    /* the row's own verdict for the stage (rd= / rdw=, the owner's tokens, 2026-10-14):
     * never under recalibrate, never without the chain's row */
    if (W && !W->vw2_off_2d && !cfg->recalibrate &&
        vw2_ilcol_forms_lookup_base(&W->vw2, key, "rd", rd, sizeof rd))
    {
        const char *q = rd;
        long prod = 1;
        while (*q && nreq < 8)
        {
            char *e;
            const long r = strtol(q, &e, 10);
            if (e == q || r < 2) { nreq = 0; break; }
            req[nreq++] = (int)r;
            prod *= r;
            if (*e == '.') q = e + 1;
            else if (*e == '\0') q = e;
            else { nreq = 0; break; }
        }
        if (nreq && prod != p - 1) nreq = 0;
        if (nreq && vw2_ilcol_forms_lookup_base(&W->vw2, key, "rdw", rdw, sizeof rdw))
            Wr = atoi(rdw);
    }
    c->rd0 = _ilrd_col0_build(p, N, rn, T < 1 ? 1 : T, nreq ? req : NULL, nreq, Wr > 0 ? (size_t)Wr : 0);
    if (!c->rd0 && nreq)
    {   /* a banked recipe that no longer builds: the default, and the race re-banks */
        nreq = 0;
        c->rd0 = _ilrd_col0_build(p, N, rn, T < 1 ? 1 : T, NULL, 0, 0);
    }
    if (!c->rd0)
        return 0;
    if (replayed) *replayed = nreq > 0;
    if (getenv("VFFT_IL2D_LOG"))
    {
        char d[64]; int w = 0;
        _ilrd_col0_desc(c->rd0, d, sizeof d, &w);
        fprintf(stderr, "[il2d] rader %dx%zu: stage 0 p=%d inner %s W=%d %s\n", N, rn, p, d, w,
                nreq ? "src=wisdom" : "(the default; the race follows)");
    }
    _ilrd_col0_ctx_set(c->rd0, N);
    return 1;
}

/* THE COLUMN ENGINE'S RACE (section 13, phase A2): the inner chain of p - 1 and the
 * window width, every arm a live engine swapped into the stage's tags, timed on the
 * column pass the cell serves (the N-arm harness: the natural pass with the leaf
 * redirection, or the plain pass); the winner stays attached and banks as rd= / rdw= on
 * the column row. Once per cell: a row with rd= replays through the attach. Serial
 * timing at every T (the engine's inner is a per-column choice). */
/* the engine's verdict on the column row (rd= / rdw=); nothing without the row (a pinned
 * chain's row is created after the build: the cell's next create banks) */
static void _il2d_col_rd0_bank(struct vfft_wisdom_s *W, const vfft_cfg_t *cfg,
                               const vw2_ilcol_key_t *key, const void *rd0, int N, size_t rn, const char *how)
{
    char d[64], wb[16];
    int w = 0, b1;
    if (!W || W->vw2_off_2d || !_ilrd_col0_desc(rd0, d, sizeof d, &w)) return;
    snprintf(wb, sizeof wb, "%d", w);
    b1 = vw2_ilcol_forms_bank_base(&W->vw2, key, "rd", d);
    if (b1 && vw2_ilcol_forms_bank_base(&W->vw2, key, "rdw", wb))
        _vw2_persist(W, cfg);
    if (getenv("VFFT_IL2D_LOG"))
        fprintf(stderr, "[il2d] rader %dx%zu: %s -> inner %s W=%d, %s\n", N, rn, how, d, w,
                b1 ? "banked" : "not banked (no chain row)");
}
typedef struct { _il2d_n1arm_ctx_t *h; vfft_ilrd_t *rd; char name[40]; } _il2d_rd0_arm_t;
static void _il2d_rd0_arm_run(void *v)
{
    _il2d_rd0_arm_t *a = (_il2d_rd0_arm_t *)v;
    ((_ilrd_ctab_t *)a->h->tf[0])->rd = a->rd;
    _il2d_n1arm_chain(a->h);
}
static void _il2d_col_rd0_race(struct vfft_wisdom_s *W, const vfft_cfg_t *cfg,
                               const vw2_ilcol_key_t *key, vfft_ilcol_t *c, int N, size_t rn, int T)
{
    vfft_ilrd_t *cur = (vfft_ilrd_t *)c->rd0, *win;
    _ilrd_inner_t cand[VFFT_ILRD_MAX_INNERS];
    _il2d_rd0_arm_t cx[VFFT_ILRD_MAX_INNERS + 4];
    vfft_race_arm_t arms[VFFT_ILRD_MAX_INNERS + 4];
    double ns[VFFT_ILRD_MAX_INNERS + 4];
    _il2d_n1arm_ctx_t h;
    _il2d_ip_reset_t rs;
    vfft_race_proto_t proto = { 3, 1, VFFT_RACE_MIN, 0, 0, _il2d_ip_reset, &rs, 1 }; /* min-of-3, the plane re-laid before every sample */ /* single-thread arms: paced (VFFT_RACE_PACE_MS) */
    const size_t scn = 2 * (size_t)N * rn;
    vfft_real *sc, *scs;
    int n, na = 0, i, best, nst_w, Rs_w[VFFT_ILRD_MAX_NST];
    size_t W_w, wc[4], q;
    const int p = cur ? cur->p : 0;
    if (!cur || !c->tf[0] || !c->tb[0] || c->blu)
        return;
    sc = (vfft_real *)vfft_aligned_alloc(scn * sizeof(vfft_real));
    scs = (vfft_real *)vfft_aligned_alloc(scn * sizeof(vfft_real));
    if (!sc || !scs) { vfft_aligned_free(sc); vfft_aligned_free(scs); return; }
    for (q = 0; q < scn; q++) sc[q] = 1.0 + 1e-6 * (double)(q & 511);   /* the whole plane: scn is size_t */
    memcpy(scs, sc, scn * sizeof(vfft_real));
    memset(&h, 0, sizeof h);
    h.sc = sc; h.N1 = N; h.N2 = rn; h.nst = c->nst; h.R = c->R; h.L = c->L; h.f = c->f; h.tf = c->tf;
    h.nat = c->nat; h.perm = c->natperm; h.nscr = c->natscr; h.np = _il2d_key_np(key, rn);
    rs.p = sc; rs.seed = scs; rs.n = scn;
    n = _ilrd_enum_inners(p - 1, cand, VFFT_ILRD_MAX_INNERS);
    for (i = 0; i < n; i++)
    {   /* the inner chains at the default width; the incumbent's own is among them */
        vfft_ilrd_t *rd = (vfft_ilrd_t *)_ilrd_col0_build(p, N, rn, T < 1 ? 1 : T, cand[i].Rs, cand[i].nst, 0);
        int j, off = 0;
        if (!rd) continue;
        cx[na].h = &h; cx[na].rd = rd;
        cx[na].name[0] = '\0';
        for (j = 0; j < cand[i].nst && off < (int)sizeof cx[na].name - 6; j++)
            off += snprintf(cx[na].name + off, sizeof cx[na].name - (size_t)off, "%s%d", j ? "." : "", cand[i].Rs[j]);
        arms[na].name = cx[na].name; arms[na].run = _il2d_rd0_arm_run; arms[na].ctx = &cx[na];
        na++;
    }
    if (na == 0)
    {   /* no candidate built: the default serves, its verdict banked so the row replays */
        vfft_aligned_free(sc); vfft_aligned_free(scs);
        _il2d_col_rd0_bank(W, cfg, key, cur, N, rn, "no inner candidate; the default");
        return;
    }
    ((_ilrd_ctab_t *)c->tf[0])->rd = cx[0].rd;
    proto.reps = _il2d_race_reps(_il2d_n1arm_chain, &h, (double)N);
    best = vfft_race_run(&proto, arms, na, ns);
    if (best < 0) best = 0;
    if (getenv("VFFT_IL2D_LOG"))
    {
        fprintf(stderr, "[il2d] rader %dx%zu p=%d inner race (reps=%d):", N, rn, p, proto.reps);
        for (i = 0; i < na; i++) fprintf(stderr, " %s=%.0f", cx[i].name, ns[i]);
        fprintf(stderr, " -> %s\n", cx[best].name);
    }
    win = cx[best].rd;
    memcpy(Rs_w, win->Rs, sizeof Rs_w); nst_w = win->nst; W_w = win->W;
    for (i = 0; i < na; i++) if (cx[i].rd != win) _ilrd_free(cx[i].rd);
    /* the widths around the winner's: half, double, quadruple, the column space */
    na = 0;
    cx[na].h = &h; cx[na].rd = win;
    snprintf(cx[na].name, sizeof cx[na].name, "W%zu", W_w);
    arms[na].name = cx[na].name; arms[na].run = _il2d_rd0_arm_run; arms[na].ctx = &cx[na]; na++;
    wc[0] = W_w / 2; wc[1] = W_w * 2; wc[2] = W_w * 4; wc[3] = win->ncol & ~(size_t)1;
    for (i = 0; i < 4; i++)
    {
        vfft_ilrd_t *rd;
        int dup = 0, j;
        if (wc[i] < VFFT_ILRD_W_MIN || wc[i] > win->ncol || wc[i] == W_w) continue;
        for (j = 1; j < na; j++) if (cx[j].rd->W == wc[i]) dup = 1;
        if (dup) continue;
        rd = (vfft_ilrd_t *)_ilrd_col0_build(p, N, rn, T < 1 ? 1 : T, Rs_w, nst_w, wc[i]);
        if (!rd || rd->W == W_w) { if (rd) _ilrd_free(rd); continue; }
        cx[na].h = &h; cx[na].rd = rd;
        snprintf(cx[na].name, sizeof cx[na].name, "W%zu", rd->W);
        arms[na].name = cx[na].name; arms[na].run = _il2d_rd0_arm_run; arms[na].ctx = &cx[na]; na++;
    }
    if (na > 1)
    {
        best = vfft_race_run(&proto, arms, na, ns);
        if (best < 0) best = 0;
        if (getenv("VFFT_IL2D_LOG"))
        {
            fprintf(stderr, "[il2d] rader %dx%zu p=%d width race:", N, rn, p);
            for (i = 0; i < na; i++) fprintf(stderr, " %s=%.0f", cx[i].name, ns[i]);
            fprintf(stderr, " -> %s\n", cx[best].name);
        }
        for (i = 0; i < na; i++) if (i != best) _ilrd_free(cx[i].rd);
        win = cx[best].rd;
    }
    vfft_aligned_free(sc);
    vfft_aligned_free(scs);
    /* the winner is the stage's engine: the tags, the descriptor, the context; the incumbent goes */
    ((_ilrd_ctab_t *)c->tf[0])->rd = win;
    ((_ilrd_ctab_t *)c->tb[0])->rd = win;
    _ilrd_col0_ctx_clear(cur);
    if (cur != win) _ilrd_free(cur);
    c->rd0 = win;
    _ilrd_col0_ctx_set(win, N);
    _il2d_col_rd0_bank(W, cfg, key, win, N, rn, "raced");
}

/* the forms axis by the column row's key (rank/axis-general twin of
 * _il2d_forms_serve, which keys rank 2 axis 0) */
static void _il2d_forms_serve_key(struct vfft_wisdom_s *W,
                                  const vfft_cfg_t *cfg,
                                  const vw2_ilcol_key_t *key, const char *base,
                                  int N, size_t rn,
                                  const int *Rs, int nst,
                                  vfft_il2p_fn *ff, vfft_il2p_fn *fb,
                                  char *forms, size_t fsz, int T, int nat)
{
    const char *pin = getenv("VFFT_IL2D_FORMS");
    /* the leaf's half-store twins race for the 2D real r2c plan only: its
     * column pass runs in place at the odd CCE pitch, the leaf storing last
     * (shared/col/half/README.md). A c2r plan's pool is the one it had (its
     * backward leaf stores into the scratch plane, a different question: a
     * separate piece of work), and a c2r replaying a forward-banked half form
     * runs the full-store twin (vfft_il2p_n1c_form_fn, bwd). */
    const int half = key->rank == 2 && key->real && cfg->transform == VFFT_R2C;
    int s, any = 0;
    forms[0] = 0;
    for (s = 0; s < nst; s++)
    {   /* the AUTHORITY (vfft_il2p_col_forms), not a restatement of it (R4) */
        const char *nm[VFFT_IL2P_COL_MAXFORMS];
        if (vfft_il2p_col_forms(Rs[s], half && s == nst - 1, nm) > 1)
            any = 1;
    }
    if (!any)
        return;
    if (pin && *pin)
    {
        if (_il2d_apply_forms(Rs, nst, pin, ff, fb))
            snprintf(forms, fsz, "%s", pin);
        else
            _vfft_warn("VFFT_IL2D_FORMS=%s does not fit chain at %dx%zu - ignored",
                       pin, N, rn);
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
                fprintf(stderr, "[il2d] %s %dx%zu: replay %s src=wisdom\n", base, N, rn, forms);
            return;
        }
        _vfft_warn("banked %s=%s does not fit chain at %dx%zu - re-racing",
                   base, forms, N, rn);
        (void)_il2d_resolve(Rs, nst, ff, fb);
    }
    if (_il2d_race_forms(N, rn, Rs, nst, ff, fb, forms, fsz, half, T, nat, key->lm) && forms[0])
    {
        const int banked = vw2_ilcol_forms_bank_base(&W->vw2, key, base, forms);
        if (banked)
            _vw2_persist(W, cfg);
        if (getenv("VFFT_IL2D_LOG"))
            fprintf(stderr, "[il2d] %s %dx%zu: raced -> %s, %s\n", base, N, rn, forms,
                    banked ? "banked" : "NOT banked yet (no chain row; the create re-banks once it lands)");
    }
}

/* ── the tpc arm's build, race and replay (struct comment at c->tpc). c2c
 * axis 0 only: the real tier's CCE columns and a rank-3 axis 1 run
 * per-worker clones the one plan cannot serve. VFFT_IL2D_TPC=1|0 pins
 * (never banks). A row without a tpc= token races the arm on its next
 * create. */
static int _il2d_tpc_admits(const vw2_ilcol_key_t *key, const vfft_ilcol_t *c)
{
    return c->blu > 0 && !key->real && key->axis == 0;
}
static int _il2d_clone_set(const struct vfft_plan_s *prim, struct vfft_wisdom_s *S, const vfft_cfg_t *cfg,
                           int N, int placement, int T, struct vfft_plan_s ***out, int *out_n, const char *what);   /* below */
static void _il2d_clone_set_drop(struct vfft_plan_s ***arr, int *n);

/* ── A CHILD PLAN AND ITS SHORTLIST (owner, 2026-11-24; il/wisdom/wisdom2_child.h, the shortlist) ──────────────
 * The 1D plan of one of a 2D c2c plan's passes -- the row plan, the turn plan, the skewed pass's row plan, the
 * turned prime column plan -- created in role on its own store *S (seeded from the cell's row by
 * vfft_child_store_for; cc is its request, the store set here). Where the row carries no recipe for it:
 *   - the store holds a shortlist at (the length, the placement): the child is created from its first candidate,
 *     nothing raced;
 *   - it holds none: the child races as a 1D transform, as it always did, and the race's first candidates are
 *     banked as the shortlist -- the one race every plane with that length shares.
 * *due = 1 when the child's recipe is new to this cell (from the shortlist, or raced here): the plane's final
 * (_il2d_child_final) then times the shortlist's candidates in the plane. Returns the plan, NULL = none; *S is the
 * store it was created on. A store that names no build keeps no shortlist: a part plan's children race as before. */
static int _k1_il_shortlist_stores(int N, int inplace, struct vfft_wisdom_s **out, int max);   /* il/rank1/k1_commit.h */
static int _il2d_shortlist_on(const struct vfft_wisdom_s *W)
{
    return W && W->vw2.isa != 0 && !W->vw2_off_2d;
}
/* THE STORE THAT KEEPS THE SHORTLISTS for a create on W: W itself when it is a real store; else the store of the 3D
 * create this plan is a part of (il/rank3/fftnd_il.h sets it around its plane child's create: the 2D plane inside a
 * 3D plan plans on a private store, and its row races are the same ones every plane with that row length runs).
 * NULL = none: the child races as it always did (a four-step's 2D child; an estimate's replay). */
/* above 0 while the 2D plan under creation is a PART of a larger plan (a volume's plane: fftnd_il.h; a batch's plane:
 * the plane queue, fft2d_create_il.h): its plane fits the caches while the volume or the batch around it may not, so its
 * staged leaf keeps the streamed store for its finished rows (vfft_internal.h il2d_rows_cached; the measurements
 * behind the cached store are single planes, policy_il.h vfft_policy_il2d_rows_cached) */
static int _il2d_part_depth = 0;
static struct vfft_wisdom_s *_il2d_sl_root = NULL;
static const vfft_cfg_t *_il2d_sl_root_cfg = NULL;
static struct vfft_wisdom_s *_il2d_shortlist_store(struct vfft_wisdom_s *W, const vfft_cfg_t *cfg, const vfft_cfg_t **pcfg)
{
    if (_il2d_shortlist_on(W))
    {
        *pcfg = cfg;
        return W;
    }
    *pcfg = _il2d_sl_root_cfg;
    return _il2d_sl_root;
}
static int _il2d_shortlist_t(const vfft_cfg_t *cc)   /* the record's transform */
{
    return cc->transform == VFFT_R2C ? VW2_T_R2C : cc->transform == VFFT_C2R ? VW2_T_C2R : VW2_T_C2C;
}
static struct vfft_plan_s *_il2d_child_create(struct vfft_wisdom_s *W, const vfft_cfg_t *cfg, vfft_cfg_t *cc,
                                              struct vfft_wisdom_s **S, int *due)
{
    const int N = cc->n[0], ip = (cc->placement == VFFT_INPLACE), t = _il2d_shortlist_t(cc);
    const int Tc = cc->nthreads > 1 ? cc->nthreads : 1;   /* the child's own thread count: the record's */
    const vfft_cfg_t *scfg = NULL;
    struct vfft_wisdom_s *SW = _il2d_shortlist_store(W, cfg, &scfg);
    struct vfft_plan_s *p;
    int listed = 0;
    if (due)
        *due = 0;
    if (!*S)
        return NULL;
    if (SW && (*S)->vw2.nrec == 0 && !cfg->recalibrate)
    {
        struct vfft_wisdom_s *c1 = vfft_shortlist_get(&SW->vw2, t, N, ip, Tc, 1);
        if (c1)
        {
            vfft_child_store_free(*S);
            *S = c1;
            listed = 1;
        }
    }
    cc->wisdom = (vfft_wisdom *)*S;
    p = (struct vfft_plan_s *)_vfft_create_cfg(cc);
    if (!p || !SW)
        return p;
    if (vfft_child_store_raced_q1(*S))
    {   /* the 1D plan raced here (no shortlist yet; its first recipe did not build as written; a recalibrating
         * create): the race's first candidates are the shortlist from now on -- the child's own one-transform rows
         * its first; behind it, for a c2c plan, the planner's next candidates */
        struct vfft_wisdom_s *cand[VFFT_SHORTLIST_MAX];
        int n = 0, k;
        cand[0] = vfft_child_store_q1(*S);
        if (cand[0])
        {
            n = 1;
            if (t == VW2_T_C2C && cc->howmany <= 1)
                n += _k1_il_shortlist_stores(N, ip, cand + 1, VFFT_SHORTLIST_MAX - 1);
            if (vfft_shortlist_put(&SW->vw2, t, N, ip, Tc, cand, n) && scfg)
                _vw2_persist(SW, scfg);
            for (k = 0; k < n; k++)
                vfft_child_store_free(cand[k]);
        }
        if (getenv("VFFT_IL2D_LOG"))
            fprintf(stderr, "[il2d] shortlist %s N=%d (%s): raced here, %d candidate(s) banked\n",
                    t == VW2_T_R2C ? "r2c" : t == VW2_T_C2R ? "c2r" : "c2c", N, ip ? "in place" : "out of place", n);
        listed = 1;
    }
    if (due)
        *due = listed;
    return p;
}

static int _il2d_tpc_build(struct vfft_wisdom_s *W, const vfft_cfg_t *cfg, const vw2_ilcol_key_t *key,
                           vfft_ilcol_t *c, int N, size_t rn)
{
    vfft_cfg_t tc;
    c->N = N;   /* the builder's caller stamps it later; the pass reads it */
    if (!c->tpcS)
    {   /* the turned prime column plan in role: its own store, its recipe from the column row (tpc_*) */
        vw2_key_t pk;
        vw2__ilcol_key(key, &pk);
        c->tpcS = vfft_child_store_for(W ? &W->vw2 : NULL, &pk, "tpc_");
        if (!c->tpcS)
            return 0;
    }
    memset(&tc, 0, sizeof tc);
    tc.transform = VFFT_C2C;
    tc.placement = VFFT_INPLACE;
    tc.rigor = cfg->rigor;
    tc.dims = 1;
    tc.n[0] = c->N;
    tc.howmany = 1;
    tc.order = VFI_ORDER_NATURAL;
    tc.layout = VFI_LAYOUT_IL;
    tc.nthreads = 1;
    tc.wisdom = (vfft_wisdom *)c->tpcS;
    tc.wisdom_write = 0;
    tc.recalibrate = cfg->recalibrate;
    if (key->rank == 2 && !key->real && !key->lm)
        /* a 2D c2c plane's: the prime cell at N raced once for every plane with that column length (the shortlist) */
        c->tpcplan = _il2d_child_create(W, cfg, &tc, &c->tpcS, NULL);
    else
        c->tpcplan = (struct vfft_plan_s *)_vfft_create_cfg(&tc);
    if (!c->tpcplan)
        return 0;
    c->tpcscr = (vfft_real *)vfft_aligned_alloc(2 * VFFT_IL2D_TPC_PITCH(c->N) * rn * sizeof(vfft_real));
    if (!c->tpcscr)
    {
        vfft_destroy(c->tpcplan);
        c->tpcplan = NULL;
        return 0;
    }
    if (key->nthreads > 1 && !c->tpcw)
        /* the threaded form's clones at the cell's T (_il2d_tpc_cols_mt); a set that does not build leaves the pass
         * serial, and the warning names it */
        (void)_il2d_clone_set(c->tpcplan, c->tpcS, cfg, c->N, VFFT_INPLACE, key->nthreads, &c->tpcw, &c->tpcw_n,
                              "turned prime pass");
    return 1;
}
static void _il2d_tpc_drop(vfft_ilcol_t *c)
{
    _il2d_clone_set_drop(&c->tpcw, &c->tpcw_n);
    if (c->tpcplan)
        vfft_destroy(c->tpcplan);
    c->tpcplan = NULL;
    vfft_child_store_free(c->tpcS);
    c->tpcS = NULL;
    if (c->tpcscr)
        vfft_aligned_free(c->tpcscr);
    c->tpcscr = NULL;
    c->tpc = 0;
}
static void _il2d_blu_drop_tables(vfft_ilcol_t *c)
{   /* the Bluestein's tables when the turned pass serves; blu = M stays */
    vfft_aligned_free(c->bluchf);
    vfft_aligned_free(c->bluchb);
    vfft_aligned_free(c->blukf);
    vfft_aligned_free(c->blukb);
    vfft_aligned_free(c->bluscr);
    c->bluchf = c->bluchb = c->blukf = c->blukb = c->bluscr = NULL;
}
/* the tpc race's Bluestein arm: at the cell's T > 1 through the threaded pass the cell serves Bluestein with (each
 * worker its lanes: _il2d_blu_cols_range), else the serial pipeline (a one-thread race crowned the turned pass at
 * 12005 x 32 lanes at every T, 2026-11-18, the threading census); the turned arm likewise (_il2d_tpc_arm) */
static void _il2d_tpc_blu_arm(const vfft_ilcol_t *c, vfft_real *z, vfft_real *zo, int N, size_t rn, int Tc)
{
    if (Tc > 1)
    {
        vfft_ilcol_t cv;
        memset(&cv, 0, sizeof cv);
        cv.N = N;
        cv.rn = rn;
        cv.blu = c->blu;
        cv.nst = c->nst;
        memcpy(cv.R, c->R, sizeof cv.R);
        memcpy(cv.L, c->L, sizeof cv.L);
        memcpy(cv.f, c->f, sizeof cv.f);
        memcpy(cv.b, c->b, sizeof cv.b);
        memcpy(cv.tf, c->tf, sizeof cv.tf);
        memcpy(cv.tb, c->tb, sizeof cv.tb);
        cv.bluchf = c->bluchf;
        cv.blukf = c->blukf;
        cv.bluscr = c->bluscr;
        if (_il2d_cols_mt_desc(&cv, N, z, zo, 0, Tc, NULL, -1))
            return;
    }
    _il2d_blu_cols(z, zo, N, rn, c->blu, c->nst, c->R, c->L, c->f, c->b, c->tf, c->tb,
                   c->bluchf, c->blukf, c->bluscr);
}
static void _il2d_tpc_race(struct vfft_wisdom_s *W, const vfft_cfg_t *cfg,
                           const vw2_ilcol_key_t *key, vfft_ilcol_t *c, int N, size_t rn)
{
    const char *pin = getenv("VFFT_IL2D_TPC");
    const int Tc = key->nthreads;   /* the cell's T: the Bluestein arm threaded above 1 */
    const size_t T = (size_t)N * rn;
    vfft_real *z, *zo, tb = 1e300, tt = 1e300;
    size_t i;
    int r;
    if (!_il2d_tpc_admits(key, c))
        return;
    if (pin && atoi(pin) == 0)
        return;
    if (!_il2d_tpc_build(W, cfg, key, c, N, rn))
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
    z = (vfft_real *)vfft_aligned_alloc(2 * T * sizeof(vfft_real));
    zo = (vfft_real *)vfft_aligned_alloc(2 * T * sizeof(vfft_real));
    if (!z || !zo)
    {
        if (z) vfft_aligned_free(z);
        if (zo) vfft_aligned_free(zo);
        _il2d_tpc_drop(c);
        return;
    }
    for (i = 0; i < 2 * T; i++)
        z[i] = (double)(i % 977) / 977.0 - 0.5;
    /* min of 3, alternated, both arms through their serving functions (both threaded at the cell's T) */
    if (Tc > 1)
        _vfft_pool_arm(Tc);
    _il2d_tpc_blu_arm(c, z, zo, N, rn, Tc);
    _il2d_tpc_arm(c, z, zo, rn, Tc);
    for (r = 0; r < 3; r++)
    {
        const double t0 = vfft_now_ns();
        _il2d_tpc_blu_arm(c, z, zo, N, rn, Tc);
        const double t1 = vfft_now_ns();
        _il2d_tpc_arm(c, z, zo, rn, Tc);
        const double t2 = vfft_now_ns();
        if (t1 - t0 < tb) tb = t1 - t0;
        if (t2 - t1 < tt) tt = t2 - t1;
    }
    vfft_aligned_free(z);
    vfft_aligned_free(zo);
    c->tpc = (tt < tb);
    if (getenv("VFFT_IL2D_LOG"))
        fprintf(stderr, "[il2d] tpc race at N=%d x %zu lanes, T=%d: blu %.0f us, turned %.0f us -> %s\n",
                N, rn, Tc, tb / 1e3, tt / 1e3, c->tpc ? "TURNED" : "blu");
    vw2_ilcol_tok_seti(&W->vw2, key, "tpc", c->tpc);
    _vw2_persist(W, cfg);
    if (c->tpc)
        _il2d_blu_drop_tables(c);
    else
        _il2d_tpc_drop(c);
}
static void _il2d_tpc_replay(struct vfft_wisdom_s *W, const vfft_cfg_t *cfg,
                             const vw2_ilcol_key_t *key, vfft_ilcol_t *c, int N, size_t rn)
{
    const char *pin = getenv("VFFT_IL2D_TPC");
    int want;
    if (!_il2d_tpc_admits(key, c))
        return;
    want = pin ? atoi(pin) : vw2_ilcol_tok_geti(&W->vw2, key, "tpc", -1);
    if (want < 0)
    {   /* a row without the token: race it now */
        _il2d_tpc_race(W, cfg, key, c, N, rn);
        return;
    }
    if (want != 1)
        return;
    if (!_il2d_tpc_build(W, cfg, key, c, N, rn))
        return;
    c->tpc = 1;
    _il2d_blu_drop_tables(c);
}

/* a lane-major cell's serving form for the chain race (_il2d_lmform_t), from its key and request: the door by the
 * key's rank and class, the pass's placement (the r2c spectrum's pass always in place) and direction, its T */
static void _il2d_lmform_of(const vw2_ilcol_key_t *key, const vfft_cfg_t *cfg, _il2d_lmform_t *f)
{
    const int c2r = key->real && cfg->transform == VFFT_C2R;
    memset(f, 0, sizeof *f);
    f->kind = !key->real ? 0 : key->rank == 1 ? 1 : 2;
    f->rev = c2r;
    f->ip = (f->kind == 2 && !c2r) || cfg->placement == VFFT_INPLACE;
    f->T = key->nthreads;
}

/* ═══ THE COLUMN-AXIS PASS BUILD. One column axis of an interleaved c2c
 * plan — N rows over rn complex per row — under the cell's wisdom key: the
 * chain (env > banked > raced over the composition pool), its per-stage
 * forms, the column-axis Bluestein where no chain exists, the natural leaf
 * redirection when the caller asks for NATURAL, the N1-arm race (chain vs
 * Bluestein where a chain carries an odd radix), and the stage tables. The
 * 2D create's c2c branch calls it with a rank-2 key; a rank-N IL plan calls
 * it per column axis with a rank-3 key and the axis as the token suffix.
 * The banked axis verdicts (band width, fusion, row route, column MT + its
 * T, the N1-arm verdict) come back through the out-parameters, -1 = unraced.
 * Returns 1; 0 = REFUSED (loud), the descriptor freed. *oom (may be NULL) = 1
 * when the refusal is OUT OF MEMORY (the chain race's planes): the recorded
 * reason is that cause, and a caller keeps it rather than writing its own. */
static int _il2d_col_build(struct vfft_wisdom_s *W, const vfft_cfg_t *cfg,
                           const vw2_ilcol_key_t *key, int N, size_t rn, int nat_req,
                           vfft_ilcol_t *c, char *forms, size_t fsz,
                           int *bwl, int *btf, int *bro, int *bcmt, int *bcmtt, int *bblu, int *oom)
{
    int il2d_bwl = -1, il2d_btf = -1, il2d_bro = -1;
    int il2d_bcmt = -1, il2d_bcmtt = -1, il2d_bblu = -1;
    int il2d_tbl_done = 0;
    int il2d_rd_replayed = 0;   /* the Rader stage's inner and width came from the row (no race) */
    if (oom) *oom = 0;
    _il2d_blu_ctx.N2 = rn;        /* the Bluestein inner's chain provider: this axis's row length (size_t, the lanes) */
    _il2d_blu_ctx.key = key;      /* ... and the row its verdict belongs on */
    _il2d_blu_ctx.rep_R = NULL;
    _il2d_blu_ctx.rep_nst = 0;
    _il2d_blu_ctx.commit = 0;
    if (key->lm)
        c->lmsw = -1;   /* no chain race in this build yet (the lane-major race reads it) */
    int chain_ok = 0;
    {
        /* chain precedence: env > banked lay=il verdict > RACE
         * the full composition pool (component-pinned: the race times
         * the column pass, the only thing the axis changes). */
        chain_ok = _il2d_env_chain(N, c->R, c->f, c->b, &c->nst);   /* the pin */
        if (!chain_ok && !cfg->recalibrate &&
                 /* the caller's explicit override reaches this tier too:
                  * include/vfft.h promises recalibrate=1 re-measures and
                  * overwrites the cell even on a hit (2D c2c, and 3D via
                  * fftnd_il.h). */
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
         * N-arm block below -- at every rank. */
        if (!chain_ok)
        {
            int cand[VFFT_IL2D_POOL_MAX][8], lens[VFFT_IL2D_POOL_MAX];
            int cur[8], ncand = 0, dropped = 0;
            _il2d_enum_rec(N, 0, cur, cand, lens, &ncand,
                           &dropped);
            /* (a capped pool warns from inside the enumerator) */
            /* ncand >= 1, not > 1: one arm is still a race -- it is timed
             * and it banks, so the cell gets a row for its forms and widths
             * to hang off. */
            if (ncand >= 1 && VFFT_IL2D_RADER_RADIX(cand[0][0]) &&
                !_il2d_col_rd0_attach(W, cfg, key, c, cand[0][0], N, rn, key->nthreads, NULL))
            {   /* every candidate is a leaf chain at this one prime (the enumerator's law):
                 * the engine before the race, so every arm's table set names it */
                _vfft_warn("vfft_create: IL column pass %dx%zu - the Rader stage at %d could not be "
                           "built; unsupported", N, rn, cand[0][0]);
                _il2d_col_free(c);
                return 0;
            }
            if (ncand >= 1)
            {
                double bns = 0;
                int bwl = 0;
                _il2d_lmform_t lmf;
                if (key->lm)
                    _il2d_lmform_of(key, cfg, &lmf);
                /* banded = 2: the serving forms without the band widths -- the pruning law (policy_il.h) on a 2D c2c
                 * column, composite or odd, at one thread within the L2 */
                const int bnd = (key->rank == 2 && !key->real && !key->lm && key->axis == 0 &&
                                 !vfft_policy_il2d_bands_race(N, (long long)rn, key->nthreads)) ? 2 : 1;
                int win = _il2d_race_chains(N, rn, ncand, cand,
                                            lens, &bns, nat_req, /*banded=*/bnd, &bwl,
                                            key->lm ? key->nthreads : (key->rank == 2 && key->real) ? key->nthreads : 1,   /* the serving forms: the width is re-raced on the full execute (the axis race); the rank-2 real cell at T > 1 in its serial walks and every threaded form its door serves at T; a lane-major cell at T > 1 its tile too */
                                            key->lm ? &lmf : NULL,   /* a lane-major batch: its door's serial forms in its placement and at T > 1 the tile, the winner's lane tile kept (the threaded form is decided after, _lm_race) */
                                            _il2d_key_np(key, rn));   /* the natural pass's own buffers as the cell's serial walk lays them */
                if (win == VFFT_IL2D_RACE_OOM)
                {   /* OUT OF MEMORY: the race's planes could not be allocated (its reason recorded). That is no
                     * verdict on N's chains: refused with that cause, no other route tried */
                    if (oom) *oom = 1;
                    _il2d_col_free(c);
                    return 0;
                }
                /* nat_req, NOT key->ord: key->ord is the ROW LABEL, nat_req
                 * is which PASS this build will run. They are equal at the 2D
                 * tier and at the 3D tier's axis 1; they are NOT equal at the
                 * 3D tier's axis 0, which is the scrambled class for both
                 * order classes by design (fftnd_il.h). Racing on the label
                 * there would time a pass the tier never runs and exclude
                 * every chain with no natural leaf from the pool. */
                if (win >= 0 &&
                    _il2d_resolve(cand[win], lens[win], c->f,
                                  c->b))
                {
                    memcpy(c->R, cand[win],
                           sizeof cand[win]);
                    c->nst = lens[win];
                    chain_ok = 1;
                    if (key->lm)
                        c->lmsw = lmf.wc;   /* the lane tile it served best with: the lane-major race keeps it */
                    if (getenv("VFFT_IL2D_LOG"))
                    {
                        if (key->lm)
                            fprintf(stderr, "[il2d] chain race %dx%zu: the winner served best at sw=%d (%.0f ns; the lane-major race keeps the tile)\n",
                                    N, rn, lmf.wc, bns);
                        else
                            fprintf(stderr, "[il2d] chain race %dx%zu: the winner's serving form had wl=%d (%.0f ns; the axis race re-races the width)\n",
                                    N, rn, bwl, bns);
                    }
                    vw2_ilcol_chain_bank(&W->vw2, key,
                                         c->R, c->nst,
                                         -1, -1, -1, -1, -1, -1,
                                         bns);
                    _vw2_persist(W, cfg);
                }
            }
            /* no greedy fallback: a race with no buildable arm has no
             * chain, and !chain_ok below is Bluestein or a refusal */
            if (!chain_ok && c->rd0)
            {   /* the race's engine goes with the chain it had no winner for */
                _ilrd_col0_ctx_clear(c->rd0);
                if (_il2d_rd0_free_hook) _il2d_rd0_free_hook(c->rd0);
                c->rd0 = NULL;
            }
        }
    }
    if (chain_ok && il2d_bblu <= 0 && c->nst >= 1 && VFFT_IL2D_RADER_RADIX(c->R[0]) &&
        !_il2d_col_rd0_attach(W, cfg, key, c, c->R[0], N, rn, key->nthreads, &il2d_rd_replayed))
    {   /* the Rader stage's engine, before any table set for this chain */
        _vfft_warn("vfft_create: IL column pass %dx%zu - the Rader stage at %d could not be "
                   "built; unsupported", N, rn, c->R[0]);
        _il2d_col_free(c);
        return 0;
    }
    if (chain_ok && il2d_bblu <= 0)
    {   /* per-stage kernel forms; a banked Bluestein cell's inner forms
         * serve through the chain provider (bluforms=) */
        _il2d_forms_serve_key(W, cfg, key, "forms", N, rn, c->R, c->nst,
                          c->f, c->b, forms, fsz,
                          (key->rank == 2 && key->real) ? key->nthreads : 1, nat_req);   /* the rank-2 real cell at T > 1: through its threaded pass */
    }
    if (!chain_ok)
    {
        /* ODD/PRIME N: the COLUMN-AXIS BLUESTEIN (struct
         * comment at c->blu; _il2d_blu_build). Reached only
         * when no chain exists — with the odd t2c/n1c kinds
         * emitted, that means prime / unexpressible N.
         * n1 comes out NATURAL by construction, so ALL order
         * spellings are served. */
        int bluoom = 0;
        _il2d_blu_ctx.commit = 1;   /* no chain exists: this build IS the cell's plan */
        c->blu = _il2d_blu_build(N, rn, c->R,
                                   c->L, c->f, c->b,
                                   c->tf, c->tb, &c->nst,
                                   &c->bluchf, &c->bluchb,
                                   &c->blukf, &c->blukb,
                                   &c->bluscr, &bluoom);
        _il2d_blu_ctx.commit = 0;
        if (!c->blu && bluoom)
        {   /* OUT OF MEMORY building the Bluestein route (its reason recorded): refused with that cause */
            if (oom) *oom = 1;
            _il2d_col_free(c);
            return 0;
        }
        if (c->blu)
        {
            chain_ok = 1;
            _il2d_tpc_race(W, cfg, key, c, N, rn);   /* the turned pass vs this Bluestein */
        }
        /* (the cell's own row is banked by the provider, at every rank and
         * before it serves the inner's forms -- vw2_ilcol_forms_bank_base is
         * a field update and needs the row to exist.) */
    }
    if (chain_ok && !c->blu && il2d_bblu <= 0 && c->nst > 1 &&
        nat_req)
    {
        /* (il2d_bblu > 0 = a banked Bluestein cell: c->R is the
         * length-M chain, n1 natural by construction, the perm
         * would divide by zero) */
        /* the natural n1 (struct comment at c->nat) via
         * the LEAF REDIRECTION - driver-only, any chain. Built
         * BEFORE the N-arm race below so the chain arm is
         * timed under the serving it will actually run. The
         * perm builder refuses on any convention mismatch. */
        c->natperm = _il2d_nat_perm(c->R, c->nst, N);
        if (c->natperm)
            c->natscr = (vfft_real *)vfft_aligned_alloc(   /* 64-B aligned: the strip workers' pieces of a row must not share a line */
                2 * (size_t)N * VFFT_IL2D_NAT_CAP(rn) * sizeof(vfft_real));   /* its rows sized for the natural pass's own pitch (il2d_col.h) */
        if (c->natperm && !c->natscr)
        {   /* OUT OF MEMORY: the natural leaf's plane-sized scratch -- refused with that cause */
            free(c->natperm);
            c->natperm = NULL;
            _vfft_warn("vfft_create: IL column pass %dx%zu order=NATURAL: out of memory (the natural leaf's scratch "
                       "needs %.1f MB)", N, rn, 2.0 * (double)N * (double)VFFT_IL2D_NAT_CAP(rn) * sizeof(vfft_real) / 1048576.0);
            if (oom) *oom = 1;
            { _il2d_col_free(c); return 0; }
        }
        if (!c->natperm)
        {
            free(c->natperm);
            c->natperm = NULL;
            _vfft_warn("vfft_create: IL 2D c2c %dx%zu "
                       "order=NATURAL - the natural leaf "
                       "permutation could not be built for "
                       "this chain; unsupported",
                       N, rn);
            { _il2d_col_free(c); return 0; }
        }
        c->nat = 1;
    }
    if (chain_ok && !c->blu && il2d_bblu <= 0 && c->rd0 && !il2d_rd_replayed)
    {   /* THE COLUMN ENGINE'S RACE: its inner chain and width on the pass the cell serves,
         * before the N-arm race times the chain with it. The descriptor's tables (the tags)
         * first; the shared build below is skipped by il2d_tbl_done. */
        if (!il2d_tbl_done)
        {
            if (_il2d_build_tables(N, c->nst, c->R, c->L, c->tf, c->tb))
            {
                _vfft_warn("vfft_create: IL column pass %dx%zu - stage tables failed; unsupported", N, rn);
                _il2d_col_free(c);
                return 0;
            }
            il2d_tbl_done = 1;
        }
        _il2d_col_rd0_race(W, cfg, key, c, N, rn, key->nthreads);
    }
    if (chain_ok && !c->blu && !getenv("VFFT_IL2D_CHAIN"))
    {
        /* THE RACED CHAIN ARM: for a chain that carries an ODD
         * radix, race it against the Bluestein column route.
         * SCRAMBLED order: the two serve different n1 orders
         * (chain = scrambled comb, blu = natural), both
         * self-consistent. NATURAL order: BOTH arms are
         * natural - the chain via the leaf redirection, blu by
         * construction - so the race is the only lawful pick;
         * the chain arm runs the natural pass. Pick is pure
         * speed either way. Env VFFT_IL2D_BLU=1 pins blu,
         * =0 pins the chain (env never banks); unset = race
         * min-of-3 on scratch through the SERVING functions,
         * banked (blu= on the chain row) and replayed. pow2
         * chains never race (blu is pointless there), nor a
         * chain of kernel radices: Bluestein races a Rader chain
         * only (vfft_policy_il2d_blu_races); the env pin still
         * races it. */
        int hasodd = 0, s3;
        const char *be = getenv("VFFT_IL2D_BLU");
        for (s3 = 0; s3 < c->nst; s3++)
            if (c->R[s3] & 1)
                hasodd = 1;
        if (hasodd && !be && !vfft_policy_il2d_blu_races(c->nst >= 1 && VFFT_IL2D_RADER_RADIX(c->R[0])))
            hasodd = 0;   /* no Bluestein arm: the chain serves */
        /* the chain arm times the SERVING column pass, so the
         * N tables must exist BEFORE the race (they are
         * otherwise built at the shared build below — timing
         * with empty tables would NULL-load). il2d_tbl_done
         * stops the later shared build from double-building
         * the winner's. */
        /* REPLAY the banked N-arm verdict:
         * blu > 0 = Bluestein won (the row's chain IS the M
         * chain, already resolved above — build the Bluestein
         * tables and adopt, no timing); blu == 0 = the chain won
         * (nothing to do). Only an unraced cell (-1) or an env
         * pin runs the race below. */
        /* blu > 0 REGARDLESS of hasodd: a banked Bluestein row carries
         * the pow2 M chain, which has no odd factor — gating the
         * adoption on hasodd would run the M chain as the N chain at a
         * PRIME N (wrong output) */
        if (il2d_bblu > 0 && !be && !cfg->recalibrate)
        {
            int bR[8], bL[8], bnst = 0, M2;
            vfft_il2p_fn bf[8], bb[8];
            vfft_real *btf[8], *btb[8];
            vfft_real *bchf, *bchb, *bkf, *bkb, *bscr;
            memset(btf, 0, sizeof btf);
            memset(btb, 0, sizeof btb);
            /* the replayed row's chain IS the M chain (the lookup above
             * checked prod == blu): hand it to the provider rather than have
             * it re-read a row of its own */
            int bluoom = 0;
            _il2d_blu_ctx.rep_R = c->R;
            _il2d_blu_ctx.rep_nst = c->nst;
            M2 = _il2d_blu_build(N, rn, bR, bL, bf, bb,
                                 btf, btb, &bnst, &bchf, &bchb,
                                 &bkf, &bkb, &bscr, &bluoom);
            _il2d_blu_ctx.rep_R = NULL;
            _il2d_blu_ctx.rep_nst = 0;
            if (!M2 && bluoom)
            {   /* OUT OF MEMORY rebuilding the banked Bluestein (its reason recorded): refused with that cause.
                 * The descriptor holds the row's M chain here, never N's: nothing may run on past this */
                if (oom) *oom = 1;
                _il2d_col_free(c);
                return 0;
            }
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
                _il2d_tpc_replay(W, cfg, key, c, N, rn);   /* tpc= on the row, or race it */
                if (c->nat)
                {
                    free(c->natperm);
                    vfft_aligned_free(c->natscr);
                    c->natperm = NULL;
                    c->natscr = NULL;
                    c->nat = 0;
                }
                if (getenv("VFFT_IL2D_LOG"))
                    fprintf(stderr, "[il2d] N-arm %dx%zu (c2c): "
                                    "replay BLUESTEIN M=%d src=wisdom\n",
                            N, rn, M2);
                hasodd = 0;            /* verdict served */
            }
            else
            {
                for (s3 = 0; s3 < bnst; s3++)
                {
                    vfft_aligned_free(btf[s3]);
                    vfft_aligned_free(btb[s3]);
                }
                vfft_aligned_free(bchf); vfft_aligned_free(bchb); vfft_aligned_free(bkf); vfft_aligned_free(bkb); vfft_aligned_free(bscr);
            }
        }
        else if (hasodd && il2d_bblu == 0 && !be && !cfg->recalibrate)
        {
            if (getenv("VFFT_IL2D_LOG"))
                fprintf(stderr, "[il2d] N-arm %dx%zu (c2c): replay "
                                "chain src=wisdom\n", N, rn);
            hasodd = 0;                /* the chain won: no race */
        }
        if (hasodd && (!be || atoi(be) == 1) &&
            (il2d_tbl_done || !_il2d_build_tables(N, c->nst, c->R, c->L,
                                                  c->tf, c->tb)))
        {
            il2d_tbl_done = 1;
            int bR[8], bL[8], bnst = 0, M2;
            vfft_il2p_fn bf[8], bb[8];
            vfft_real *btf[8], *btb[8];
            vfft_real *bchf, *bchb, *bkf, *bkb, *bscr;
            int bluoom = 0;
            memset(btf, 0, sizeof btf);
            memset(btb, 0, sizeof btb);
            M2 = _il2d_blu_build(N, rn, bR, bL, bf, bb,
                                 btf, btb, &bnst, &bchf, &bchb,
                                 &bkf, &bkb, &bscr, &bluoom);
            if (!M2 && bluoom)
            {   /* OUT OF MEMORY building the Bluestein arm (its reason recorded): the race cannot run, and a chain kept
                 * because memory ran out is no verdict -- refused with that cause */
                if (oom) *oom = 1;
                _il2d_col_free(c);
                return 0;
            }
            if (M2)
            {
                vfft_real *sc = (vfft_real *)vfft_aligned_alloc(
                    2 * (size_t)N * rn * sizeof(vfft_real));
                double tc = 1e300, tbu = 1e300;
                int rr, use_blu = (be != NULL); /* env pin */
                size_t i3;
                if (!sc && !use_blu)
                {   /* OUT OF MEMORY: the race's plane -- refused with that cause, never a chain banked unmeasured */
                    _vfft_warn("vfft_create: IL column pass %dx%zu: out of memory (racing the chain against the "
                               "Bluestein needs %.1f MB of planes)", N, rn, 2.0 * 2.0 * (double)N * (double)rn *
                               sizeof(vfft_real) / 1048576.0);
                    for (s3 = 0; s3 < bnst; s3++)
                    {
                        vfft_aligned_free(btf[s3]);
                        vfft_aligned_free(btb[s3]);
                    }
                    vfft_aligned_free(bchf); vfft_aligned_free(bchb); vfft_aligned_free(bkf); vfft_aligned_free(bkb); vfft_aligned_free(bscr);
                    if (oom) *oom = 1;
                    _il2d_col_free(c);
                    return 0;
                }
                if (sc && !use_blu)
                {
                    const size_t scn = 2 * (size_t)N * rn;
                    vfft_real *scs = (vfft_real *)vfft_aligned_alloc(scn * sizeof(vfft_real));   /* the arms run in place: the reset's seed */
                    if (!scs)
                    {   /* OUT OF MEMORY: the reset's seed plane -- the same refusal */
                        _vfft_warn("vfft_create: IL column pass %dx%zu: out of memory (racing the chain against the "
                                   "Bluestein needs %.1f MB of planes)", N, rn, 2.0 * 2.0 * (double)N * (double)rn *
                                   sizeof(vfft_real) / 1048576.0);
                        vfft_aligned_free(sc);
                        for (s3 = 0; s3 < bnst; s3++)
                        {
                            vfft_aligned_free(btf[s3]);
                            vfft_aligned_free(btb[s3]);
                        }
                        vfft_aligned_free(bchf); vfft_aligned_free(bchb); vfft_aligned_free(bkf); vfft_aligned_free(bkb); vfft_aligned_free(bscr);
                        if (oom) *oom = 1;
                        _il2d_col_free(c);
                        return 0;
                    }
                    for (i3 = 0; i3 < scn; i3++)
                        sc[i3] = 1.0 + 1e-6 * (double)(i3 & 511);
                    if (scs)
                    {
                        _il2d_n1arm_ctx_t rc = {
                            sc, N, rn, c->nst, c->R,
                            c->L, c->f, c->tf, M2, bnst, bR,
                            bL, bf, bb, btf, btb, bchf, bkf, bscr,
                            c->nat, c->natperm, c->natscr, key->nthreads,   /* at T > 1 both arms threaded */
                            _il2d_key_np(key, rn) };
                        const vfft_race_arm_t arms[2] = {
                            { "chain", _il2d_n1arm_chain, &rc },
                            { "bluestein", _il2d_n1arm_blu, &rc } };
                        _il2d_ip_reset_t srs = { sc, scs, scn };
                        vfft_race_proto_t proto = { 3, 1, VFFT_RACE_MIN, 0, 0, _il2d_ip_reset, &srs, 1 }; /* min-of-3, A then B, the plane re-laid before every sample */ /* single-thread arms: paced (VFFT_RACE_PACE_MS) */
                        double ns[2];
                        (void)rr;
                        if (key->nthreads > 1)
                        {   /* threaded arms: no pause (it parks the team), the pool armed at T */
                            proto.pace = 0;
                            _vfft_pool_arm(key->nthreads);
                        }
                        memcpy(scs, sc, scn * sizeof(vfft_real));
                        proto.reps = _il2d_race_reps(_il2d_n1arm_chain, &rc, (double)N);
                        vfft_race_run(&proto, arms, 2, ns);
                        tc = ns[0];
                        tbu = ns[1];
                    }
                    vfft_aligned_free(scs);
                }
                vfft_aligned_free(sc);
                if (!use_blu)
                    use_blu = (tbu < tc);
                if (getenv("VFFT_IL2D_LOG"))
                    fprintf(stderr, "[il2d] N-arm race %dx%zu "
                                    "(c2c %s, T=%d): chain=%.0f blu=%.0f "
                                    "-> %s\n",
                            N, rn, c->nat ? "nat" : "scr", key->nthreads,
                            tc, tbu,
                            use_blu ? "BLUESTEIN" : "chain");
                if (!use_blu && !be && c->nst == 1 && VFFT_IL2D_RADER_RADIX(c->R[0]) && c->rd0 &&
                    !key->real && key->axis == 0 && !getenv("VFFT_IL2D_TPC"))
                {   /* a prime N1: the Rader chain beat Bluestein; the TURNED pass (the 1D prime
                     * plan per lane, _il2d_tpc_race's arm) is the third, timed as that race times
                     * it; when it wins the cell is the turned Bluestein-marked cell (blu = M, tpc = 1) */
                    vfft_real *tz = (vfft_real *)vfft_aligned_alloc(2 * (size_t)N * rn * sizeof(vfft_real));
                    vfft_real *tzo = (vfft_real *)vfft_aligned_alloc(2 * (size_t)N * rn * sizeof(vfft_real));
                    if (tz && tzo && _il2d_tpc_build(W, cfg, key, c, N, rn))
                    {
                        _il2d_n1arm_ctx_t rc2 = {
                            tz, N, rn, c->nst, c->R,
                            c->L, c->f, c->tf, M2, bnst, bR,
                            bL, bf, bb, btf, btb, bchf, bkf, bscr,
                            c->nat, c->natperm, c->natscr, key->nthreads,   /* the chain at T: threaded above 1, and the
                                                                             * turned pass too (_il2d_tpc_arm) */
                            _il2d_key_np(key, rn) };
                        double tch = 1e300, ttp = 1e300;
                        int r3;
                        if (key->nthreads > 1)
                            _vfft_pool_arm(key->nthreads);
                        for (i3 = 0; i3 < 2 * (size_t)N * rn; i3++)
                            tz[i3] = 1.0 + 1e-6 * (double)(i3 & 511);
                        _il2d_n1arm_chain(&rc2);
                        _il2d_tpc_arm(c, tz, tzo, rn, key->nthreads);
                        for (r3 = 0; r3 < 3; r3++)
                        {
                            const double t0 = vfft_now_ns();
                            double t1, t2;
                            _il2d_n1arm_chain(&rc2);
                            t1 = vfft_now_ns();
                            _il2d_tpc_arm(c, tz, tzo, rn, key->nthreads);
                            t2 = vfft_now_ns();
                            if (t1 - t0 < tch) tch = t1 - t0;
                            if (t2 - t1 < ttp) ttp = t2 - t1;
                        }
                        if (getenv("VFFT_IL2D_LOG"))
                            fprintf(stderr, "[il2d] tpc vs the Rader chain at %dx%zu, T=%d: chain %.0f us, turned %.0f us -> %s\n",
                                    N, rn, key->nthreads, tch / 1e3, ttp / 1e3, ttp < tch ? "TURNED" : "rader chain");
                        if (ttp < tch)
                            use_blu = 2;   /* the turned pass serves: the Bluestein marker, tpc = 1 */
                        else
                            _il2d_tpc_drop(c);
                    }
                    else
                        _il2d_tpc_drop(c);
                    vfft_aligned_free(tz);
                    vfft_aligned_free(tzo);
                }
                if (!be)
                {   /* bank the verdict with the chain that SERVES */
                    vw2_ilcol_chain_bank(&W->vw2, key,
                                         use_blu ? bR : c->R,
                                         use_blu ? bnst : c->nst,
                                         -1, -1, -1, -1, -1,
                                         use_blu ? M2 : 0, 0.0);
                    if (use_blu == 2)
                        vw2_ilcol_tok_seti(&W->vw2, key, "tpc", 1);
                    _vw2_persist(W, cfg);
                }
                if (use_blu)
                {
                    if (c->rd0)
                    {   /* the N chain's Rader stage goes with its chain (its tags with the tables) */
                        _ilrd_col0_ctx_clear(c->rd0);
                        if (_il2d_rd0_free_hook) _il2d_rd0_free_hook(c->rd0);
                        c->rd0 = NULL;
                    }
                    for (s3 = 0; s3 < c->nst; s3++)
                    {
                        vfft_aligned_free(c->tf[s3]);
                        vfft_aligned_free(c->tb[s3]);
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
                        vfft_aligned_free(c->natscr);
                        c->natperm = NULL;
                        c->natscr = NULL;
                        c->nat = 0;
                    }
                    if (use_blu == 2)
                    {   /* the turned pass won against the Rader chain: it serves, the Bluestein tables go */
                        c->tpc = 1;
                        _il2d_blu_drop_tables(c);
                    }
                    else if (!be)
                    {   /* Bluestein beat the chain: the turned pass races the Bluestein now, as the no-chain build
                         * races it, so tpc= lands on the row in the create that adopted Bluestein (the replay of a
                         * blu = M row reads tpc=, and races it only on a row without one) */
                        _il2d_tpc_race(W, cfg, key, c, N, rn);
                    }
                }
                else
                {
                    for (s3 = 0; s3 < bnst; s3++)
                    {
                        vfft_aligned_free(btf[s3]);
                        vfft_aligned_free(btb[s3]);
                    }
                    vfft_aligned_free(bchf); vfft_aligned_free(bchb);
                    vfft_aligned_free(bkf); vfft_aligned_free(bkb); vfft_aligned_free(bscr);
                }
            }
        }
        else if (hasodd && be && atoi(be) == 0)
            ; /* env pins the chain: nothing to do */
    }
    if (!chain_ok)
    {
        /* Split is NOT a fallback of IL — no convert wrapper.
         * An inexpressible N refuses loudly. */
        _vfft_warn("vfft_create: IL 2D c2c %dx%zu — N has no "
                   "native column chain (radices 4..64, no "
                   "leftover factor)%s",
                   N, rn,
                   " and the Bluestein column route could "
                   "not be built");
        { _il2d_col_free(c); return 0; }
    }

    if (!c->blu && !il2d_tbl_done &&
        _il2d_build_tables(N, c->nst, c->R, c->L, c->tf, c->tb))
    {
        _vfft_warn("vfft_create: IL column pass %dx%zu — stage tables failed; unsupported",
                   N, rn);
        _il2d_col_free(c);
        return 0;
    }
    if (!c->blu && c->nst >= 1 && VFFT_IL2D_RADER_RADIX(c->R[0]) && (!c->rd0 || !c->tf[0] || !c->tb[0]))
    {   /* the chain's Rader stage must have its engine and the descriptor's tags */
        _vfft_warn("vfft_create: IL column pass %dx%zu - the Rader stage at %d has no engine on the plan; unsupported",
                   N, rn, c->R[0]);
        _il2d_col_free(c);
        return 0;
    }
    _ilrd_col0_ctx_clear(c->rd0);   /* the build is over: no later table set borrows this engine */
    c->N = N;
    c->rn = rn;
    *bwl = il2d_bwl; *btf = il2d_btf; *bro = il2d_bro;
    *bcmt = il2d_bcmt; *bcmtt = il2d_bcmtt; *bblu = il2d_bblu;
    return 1;
}

/* one (row route, wl, tile) configuration of the axis race, as a race ARM:
 * sets the plan's band axes, then one full forward execute through the
 * serving path (the natural banded walk under NATURAL, the scrambled one
 * otherwise). */
typedef struct
{
    struct vfft_plan_s *h;
    vfft_real *z;
    vfft_real *zo;               /* the cell's own placement: z for in place, a second plane out of place */
    int wl, cut, ro;
    int wc;                   /* the unbanded walk's column tile (0 = the whole plane) */
    int kb;                   /* ro=3: the two-pass rows' tile, KB of chunk scratch */
    int csk;                  /* the SKEWED column pass */
    int mt;                   /* T > 1: 1 = the arm runs threaded, in the form natarm / msw / natst names (a form
                               * of the threading race, _il2d_c2c_mt_race); 0 the serial walk */
    int natarm, msw, natst;   /* mt: the threaded form -- 0 the block partition (scrambled: the matched one; at a
                               * width the bands; the turn's and the skewed pass's own walk), 1 the strips (msw
                               * their sub-strip width, 0 a worker's whole block), 2 the tile; natst the
                               * natural leaf staged (1) or strided (0) */
    int grp;                  /* mt: the form's group (_il2d_axis_race: the configurations raced under ONE form) */
    char name[48];
} _il2d_axis_arm_t;
static void _il2d_axis_arm_set(const _il2d_axis_arm_t *c);
static void _il2d_arm_axis(void *v)
{
    _il2d_axis_arm_t *c = (_il2d_axis_arm_t *)v;
    _il2d_axis_arm_set(c);
    _vfft_exec6((vfft_plan)c->h, VFFT_FORWARD, c->z, NULL, c->zo, NULL);
}
/* the arm's configuration onto the plan, nothing run */
static void _il2d_axis_arm_set(const _il2d_axis_arm_t *c)
{
    struct vfft_plan_s *h = c->h;
    h->il2d_col.wl = c->wl;
    h->il2d_col.cut = c->cut;
    h->il2d_col.tfuse = (c->wl > 0);
    h->il2d_col.wc = c->wc;
    h->il2d_rowb = (c->ro == 2);   /* the batched rows (ro=2) */
    h->il2d_rowb2 = (c->ro == 3);  /* the batched two-pass rows (ro=3) */
    h->il2d_turn = (c->ro == 4);   /* the TURN route: the whole plane through the 1D engine */
    h->il2d_csk = c->csk;          /* the SKEWED column pass */
    if (c->ro == 3)
        h->il2d_rowb2_ch = (int)_il2d_rb2_rows(c->kb, (size_t)h->N2);
    if (c->mt)
    {   /* the T-aware race: the arm through ONE threaded form of its route (_il2d_axis_race groups the
         * configurations by form) */
        h->il2d_col.colmt = 1;
        h->il2d_col.natarm = c->natarm;
        h->il2d_col.msw = c->msw;
        h->il2d_col.natst = c->natst;
    }
    else
    {   /* the serial form (every arm at T = 1; the "+s" twins at T > 1), its leaf staged as the serial verdict
         * serves it (_il2d_c2c_mt_race: colmt 0 -> natst 1) */
        h->il2d_col.colmt = 0;
        h->il2d_col.natarm = 0;
        h->il2d_col.msw = 0;
        h->il2d_col.natst = 1;
    }
}
/* THE AXIS RACE'S HEAT: the arms ac[idx[0..n)) on the full execute (rounds alternated, the minimum; warm untimed
 * passes per arm first), at most VFFT_RACE_MAX_ARMS arms a race -- a larger set in balanced interleaved chunks whose
 * winners race again the same way. Returns the winner's index into ac (-1 when n = 0), its time in *bns (from the
 * last race it ran); ns_out[0..n), when given, each arm's time in the race it ran in. */
static int _il2d_axis_heat(_il2d_axis_arm_t *ac, const int *idx, int n, int rounds, int reps, int warm, double *bns,
                           double *ns_out, _il2d_ip_reset_t *rs)
{   /* rs: an in-place cell's plane re-laid before every sample (at T > 1; NULL: none). warm > 0 = threaded arms:
     * unpaced, one untimed execute before every sample (_il2d_race_reps_ns) */
    vfft_race_arm_t arms[VFFT_RACE_MAX_ARMS];
    double ns[VFFT_RACE_MAX_ARMS], wns[VFFT_RACE_MAX_ARMS];
    int wins[VFFT_RACE_MAX_ARMS], nw = 0, c0, i;
    const int nch = (n + VFFT_RACE_MAX_ARMS - 1) / VFFT_RACE_MAX_ARMS;
    if (n <= 0)
        return -1;
    for (c0 = 0; c0 < nch; c0++)
    {   /* chunk c0: the arms c0, c0 + nch, c0 + 2 nch, ... */
        const vfft_race_proto_t proto = { rounds, reps, VFFT_RACE_MIN, 1, warm, rs ? _il2d_ip_reset : NULL, rs, 0,
                                          warm > 0 ? 1 : 0 };
        int m = 0, best = 0;
        for (i = c0; i < n; i += nch)
        {
            arms[m].name = ac[idx[i]].name;
            arms[m].run = _il2d_arm_axis;
            arms[m].ctx = &ac[idx[i]];
            m++;
        }
        vfft_race_run(&proto, arms, m, ns);
        for (i = 0; i < m; i++)
        {
            if (ns[i] < ns[best])
                best = i;
            if (ns_out)
                ns_out[c0 + i * nch] = ns[i];
        }
        wns[nw] = ns[best];
        wins[nw++] = idx[c0 + best * nch];
    }
    if (nw == 1)
    {
        *bns = wns[0];
        return wins[0];
    }
    return _il2d_axis_heat(ac, wins, nw, rounds, reps, warm, bns, NULL, rs);
}

/* the axis race: time the FULL execute (column chain + rows) over the wl
 * candidates x the row routes (x the two-pass rows' tile), set the winner on
 * the plan, bank chain+wl+tf+ro(+rbk) as one lay=il verdict. wl wins +15-21%
 * at some cells and LOSES at others -- per-cell only, never defaults. The ROW
 * ENGINE is raced only where its admission leaves a question
 * (vfft_policy_il2d_row_fixed, 2026-11-08): a one-pass row kernel that does
 * holds its row in the registers (a whole-row form up to half the register
 * file -- 16 on AVX2, 64 on AVX-512 -- but 25 and 27, a split form at any
 * radix) IS the row engine, and the per-row child and the two-pass rows are
 * not arms beside it; above those lengths the child and the two-pass rows
 * race as before. The TURN arm runs only where the skewed pass has no arm:
 * when both axes are one kernel, the column goes first (two walks, not three;
 * 21 of 21 banked planes). There is no out-of-place child route (ro=1): beside
 * the batched routes it never won, and the in-place K=1 tier serves every N2
 * (its last candidate is the prime engine).
 *
 * ORDER CELLS: every bank below keys the order by the CELL's order
 * (cfg->order), never by the pass's natural flag — a natural cell whose
 * chain is natural by construction (a single stage, a Bluestein axis) has
 * il2d_col.nat == 0, and keying by that flag would bank its axis and MT
 * verdicts on the SCRAMBLED row (the natural cell re-racing on every create,
 * the scrambled cell serving a natural measurement). Order cells are never
 * compared, never mixed. */
static int _il2d_axis_heat(_il2d_axis_arm_t *ac, const int *idx, int n, int rounds, int reps, int warm, double *bns,
                           double *ns_out, _il2d_ip_reset_t *rs);
/* the axis verdict onto the plan: band width and its cut, the serial walk's column tile, the row route (and the
 * two-pass rows' tile), the skewed pass */
static void _il2d_axis_set_cfg(struct vfft_plan_s *h, int N2, int bwl, int bro, int bwc, int bkb, int bcsk)
{
    int cut = 0;
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
}
/* the verdict on the cell's row at the plan's thread count (v1.3): a threaded plan's row is its own, the plain
 * tokens its verdict; at T > 1 (mt) the threading verdict with it, in the tokens the threading race banks
 * (_il2d_c2c_mt_race: cmt/cmtt, mtarm, msw, nls, mtns) */
static void _il2d_axis_bank(struct vfft_plan_s *h, struct vfft_wisdom_s *W, const vfft_cfg_t *cfg, int N1, int N2,
                            int mt, int bwc, int bkb, double best)
{
    const int ord = vfft_policy_ord_rankn(cfg);
    vw2_2d_il_chain_bank(&W->vw2, N1, N2, h->il2d_col.R, h->il2d_col.nst,
                         h->il2d_col.wl, h->il2d_col.tfuse, _il2d_ro_of(h),
                         mt ? h->il2d_col.colmt : -1, mt ? h->nthreads : -1,
                         (N1 & (N1 - 1)) ? h->il2d_col.blu : -1, best, ord, h->nthreads);
    vw2_2d_il_tok_seti(&W->vw2, N1, N2, ord, h->nthreads, "sw", bwc);
    vw2_2d_il_tok_seti(&W->vw2, N1, N2, ord, h->nthreads, "rbk", bkb);   /* the two-pass rows' tile */
    vw2_2d_il_tok_seti(&W->vw2, N1, N2, ord, h->nthreads, "turn", h->il2d_turn);   /* the turn route */
    vw2_2d_il_tok_seti(&W->vw2, N1, N2, ord, h->nthreads, "csk", h->il2d_csk);     /* the skewed column pass */
    if (mt)
    {
        vw2_2d_il_tok_seti(&W->vw2, N1, N2, ord, h->nthreads, "mtarm", h->il2d_col.natarm);
        vw2_2d_il_tok_seti(&W->vw2, N1, N2, ord, h->nthreads, "msw", h->il2d_col.msw);
        if (h->il2d_col.nat)
            vw2_2d_il_tok_seti(&W->vw2, N1, N2, ord, h->nthreads, "nls", h->il2d_col.natst);
        vw2_2d_il_tok_seti(&W->vw2, N1, N2, ord, h->nthreads, "mtns", (int)best);
    }
    _vw2_persist(W, cfg);
}
/* cand = 0: THE AXIS RACE of the plan's chain -- set and banked; at T > 1 returns 1 (the threading verdict decided
 * with the route), its winning arm in *win_out and time in *best_out when given. cand = 1 (T > 1 only, the chain race
 * at T, _il2d_chain_race_t): the descriptor's candidate chain in its THREADED forms only (the serial walk at T is the
 * one-thread walk, whose chain the one-thread race decided), nothing set or banked; returns 1 with the champion in
 * *win_out / *best_out, 0 when none engages. */
static int _il2d_axis_race(struct vfft_plan_s *h, struct vfft_wisdom_s *W,
                           const vfft_cfg_t *cfg, int N1, int N2, int cand, _il2d_axis_arm_t *win_out,
                           double *best_out)
{
    const size_t T = (size_t)N1 * N2;
    vfft_real *z = (vfft_real *)vfft_aligned_alloc(2 * T * sizeof(vfft_real));   /* aligned like every plane the door serves */
    /* THE RACE RUNS THE CELL'S OWN PLACEMENT: an out-of-place cell races
     * x -> y like the door serves it; racing z -> z would rank a
     * placement-sensitive route (the skewed column pass) the other way round */
    vfft_real *zo = (h->placement == VFFT_OUTOFPLACE) ? (vfft_real *)vfft_aligned_alloc(2 * T * sizeof(vfft_real)) : z;   /* aligned like every plane the door serves */
    int wlc[14], nwl = 1, wi, ro, bwl = 0, bro = 0, bwc = 0, bkb = 8, bcsk = 0, csk;
    int swl[6], nsw = 0;
    double best = 1e300;
    size_t i;
    int reps = (int)(1e6 / (double)(T + 1));
    /* THE T-AWARE RACE: at T > 1 with a live pool every arm runs its route's
     * threaded walk, so a threaded chain competes with the threaded turn and
     * skewed pass (racing serially at T=8 sent 81 of 159 pow2 cells down a
     * route with no threaded walk). The verdict banks
     * beside the serial one (axt= and the t-suffixed tokens), served at
     * that T only. Every configuration is timed in its SERIAL walk (the "+s"
     * twin: a plane the verdict keeps serial picks its width and tile by
     * them, natural or not) and in EVERY threaded form it has (the threading
     * race's: a scrambled width its bands; the unbanded configuration the
     * block, the strips whole and at each sub-strip width, the tile, a
     * natural leaf staged and strided; the turn's and the skewed pass's own
     * walk); the threaded walks ignore the serial walk's column tile, and a
     * natural cell's every band width, so those configurations run serial
     * only (2026-11-18: the width is the bands' lever, and timing the strips
     * at every width picked it by noise). Per-form champions meet in one
     * final (2026-11-19, below): the route and the threaded form are ONE
     * verdict. */
    const int mt = (h->nthreads > 1 && thread_pool_workers_for(h->nthreads) >= 2);
    vfft_real *zseed = NULL;          /* in place at T > 1: the plane's seed, re-laid before every sample */
    _il2d_ip_reset_t zrs;
    _il2d_ip_reset_t *zrsp = NULL;
    if (reps < 2) reps = 2;
    if (!z || !zo)
    {
        vfft_aligned_free(z);
        if (zo != z) vfft_aligned_free(zo);
        return 0;
    }
    for (i = 0; i < 2 * T; i++)
        z[i] = 1.0 + 1e-6 * (double)(i & 1023);
    if (mt && zo == z && (zseed = (vfft_real *)vfft_aligned_alloc(2 * T * sizeof(vfft_real))) != NULL)
    {   /* in place at T > 1: the arms walk their own output, the plane re-laid before every sample */
        memcpy(zseed, z, 2 * T * sizeof(vfft_real));
        zrs.p = z;
        zrs.seed = zseed;
        zrs.n = 2 * T;
        zrsp = &zrs;
    }
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
        /* the STAGE-SPAN widths: at huge N1 the static pool tops out at
         * 256, pinning cut deep —
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
            if (!vfft_policy_fits_l2((long)w * N2 * VFFT_CX_BYTES))
                continue;
            for (p2 = 0; p2 < nwl; p2++)
                if (wlc[p2] == w)
                    dup = 1;
            if (!dup)
                wlc[nwl++] = w;
        }
    }
    if (!vfft_policy_il2d_bands_race(N1, N2, h->nthreads))
        nwl = 1;   /* the pruning law (policy_il.h): a composite or odd column at one thread within the L2 races no band */
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
    /* the unbanded walk's tile (il2d_large_plane_design.md): every ladder
     * width an arm beside wl = 0 */
    nsw = _il2d_sw_ladder(N1, N2, 1, swl, 6);
    /* every (row route, wl, tile) configuration is an ARM of ONE race, rounds
     * alternating direction — the house protocol: timing each arm in its own
     * block lets drift favour whichever ran in the cooler moment (at 405x405
     * natural such a loop banked wl=9 where an alternated A/B put wl=81 6.5%
     * ahead). Min of 2 x reps per arm. */
    {
        /* the arms on the heap: at T > 1 a configuration runs in every threaded form its route serves */
        const int amax = 2048;
        _il2d_axis_arm_t *ac = (_il2d_axis_arm_t *)calloc((size_t)amax, sizeof *ac);
        int *idx = (int *)malloc((size_t)amax * sizeof(int));
        double *ans = (double *)malloc((size_t)amax * sizeof(double));
        int na = 0, a, win = -1, lad[6], nl = 0;
        /* the row engine's admission: the one-pass kernel is THE row engine
         * (no child, no two-pass arm) where it does not spill by construction */
        const int rofix = vfft_policy_il2d_row_fixed(N2, h->il2d_rowb_f != NULL, vfft_il_row_split(N2));
        if (!ac || !idx || !ans)
        {
            free(ac);
            free(idx);
            free(ans);
            if (zo != z) vfft_aligned_free(zo);
            vfft_aligned_free(z);
            vfft_aligned_free(zseed);
            return 0;
        }
        if (mt)
            nl = _il2d_sw_ladder(N1, N2, h->nthreads, lad, 6);   /* the threaded strips' sub-strip widths (the threading
                                                                   * race's; the turn and the skewed pass have none) */
        for (csk = 0; csk <= (h->il2d_csk_scr ? 1 : 0); csk++)
        for (ro = 0; ro <= 4; ro++)
            for (wi = 0; wi < nwl && na < amax; wi++)
            {
                int s2, cut = 0;
                const int w = wlc[wi];
                (void)s2;
                /* the SKEWED column pass: unbanded, untiled, crossed
                 * with the row routes 0/2/3 (the per-row one through the OOP
                 * K=1 plan at N2, when it built); never with the turn route */
                if (csk && (w != 0 || ro == 4 || (ro == 0 && !h->il2d_csk_row)))
                    continue;
                /* the row routes: 0 = the in-place child; 2 = the BATCHED rows,
                 * when the radix has a row kernel; 3 = the batched TWO-PASS
                 * rows, when the child's stage kernels have row-loop twins
                 * (there is no route 1); 0 and 3 only where the kernel's
                 * admission leaves the question open (rofix). 4 = the
                 * TURN route (the whole plane through the 1D engine): one arm,
                 * no band, no tile, when its N1 plan built and the skewed pass
                 * has no arm (a one-kernel column goes first). */
                if (ro == 1 || (ro == 2 && !h->il2d_rowb_f) || (ro == 3 && (!h->il2d_rowb2_leaf_f || rofix)) ||
                    (ro == 0 && rofix) || (ro == 4 && (!h->il2d_turn_plan || w != 0 || h->il2d_csk_scr)))
                    continue;
                if (w > 0)
                {   /* an admitted width's cut; 0 kept where none (R3) */
                    const int r = vfft_policy_il2d_cut_of(h->il2d_col.nst, h->il2d_col.L, w);
                    if (r >= 0) cut = r;
                }
                {
                    int sub, ki, f;
                    for (sub = 0; sub <= (w == 0 && ro != 4 && !csk ? nsw : 0) && na < amax; sub++)
                        for (ki = 0; ki < (ro == 3 ? 4 : 1) && na < amax; ki++)
                        {
                            const int kb = ro == 3 ? VFFT_IL2D_RB2_KB_LADDER[ki] : 0;
                            /* the configuration's forms {mt, natarm, msw, natst, group}: the SERIAL walk (at T > 1
                             * the "+s" twin: a plane the verdict keeps serial picks its route by the serial walks),
                             * and at T > 1 every threaded form the threading race serves it with
                             * (_il2d_c2c_mt_race), where it has a threaded walk of its own: the turn's and the
                             * skewed pass's one walk; a scrambled width's bands; the unbanded untiled
                             * configuration's block (scrambled: the matched partition), strips, tile (natural)
                             * and strips at every sub-strip width, a natural leaf staged and strided. The serial
                             * walk's column tile and a natural plane's band width are no threaded walk's: those
                             * configurations run serial only. */
                            int fv[1 + 2 * 9][5], nfv = 0;
                            if (ro == 3 && ki > 0 &&
                                _il2d_rb2_rows(kb, (size_t)N2) == _il2d_rb2_rows(VFFT_IL2D_RB2_KB_LADDER[ki - 1], (size_t)N2))
                                continue;   /* the same tile in rows: one arm */
#define _IL2D_AXF(M_, A_, S_, L_, G_) do { fv[nfv][0] = (M_); fv[nfv][1] = (A_); fv[nfv][2] = (S_); \
                                           fv[nfv][3] = (L_); fv[nfv][4] = (G_); nfv++; } while (0)
                            _IL2D_AXF(0, 0, 0, 1, 0);
                            if (mt && !(sub > 0 || (w > 0 && h->il2d_col.nat)))
                            {
                                if (ro == 4 || csk)
                                    _IL2D_AXF(1, 0, 0, 1, 1);
                                else if (w > 0)
                                    _IL2D_AXF(1, 0, 0, 1, 2);
                                else
                                {
                                    int nv, k;
                                    for (nv = 0; nv < ((h->il2d_col.nat && h->il2d_col.natstage) ? 2 : 1); nv++)
                                    {
                                        const int lst = nv ? 0 : 1, g0 = 3 + 9 * nv;
                                        if (!h->il2d_col.nat || vfft_policy_il2d_blocks_race(N1, N2, h->nthreads))
                                            _IL2D_AXF(1, 0, 0, lst, g0);   /* the block; pruned on a natural plane
                                                                            * where the law says (policy_il.h) */
                                        if (h->il2d_col.nat)
                                            _IL2D_AXF(1, 1, 0, lst, g0 + 1);
                                        if (h->il2d_col.nat && h->il2d_col.R[0] >= 2)
                                            _IL2D_AXF(1, 2, 0, lst, g0 + 2);
                                        for (k = 0; k < nl; k++)
                                            _IL2D_AXF(1, 1, lad[k], lst, g0 + 3 + k);
                                    }
                                }
                            }
#undef _IL2D_AXF
                            for (f = 0; f < nfv && na < amax; f++)
                            {
                                size_t used;
                                ac[na].h = h;
                                ac[na].z = z;
                                ac[na].zo = zo;
                                ac[na].wl = w;
                                ac[na].cut = cut;
                                ac[na].ro = ro;
                                ac[na].kb = kb;
                                ac[na].csk = csk;
                                ac[na].mt = fv[f][0];
                                ac[na].natarm = fv[f][1];
                                ac[na].msw = fv[f][2];
                                ac[na].natst = fv[f][3];
                                ac[na].grp = fv[f][4];
                                ac[na].wc = sub ? swl[sub - 1] : 0;
                                if (sub)
                                    snprintf(ac[na].name, sizeof ac[na].name, "sw%d%s", ac[na].wc,
                                             ro == 2 ? "+rowb" : ro == 4 ? "+turn" : "");
                                else
                                    snprintf(ac[na].name, sizeof ac[na].name, "wl%d%s", w,
                                             ro == 2 ? "+rowb" : ro == 4 ? "+turn" : "");
                                used = strlen(ac[na].name);
                                if (ro == 3)
                                    used += (size_t)snprintf(ac[na].name + used, sizeof ac[na].name - used, "+rb2k%d", kb);
                                if (csk && used < sizeof ac[na].name)
                                    used += (size_t)snprintf(ac[na].name + used, sizeof ac[na].name - used, "+csk");
                                if (mt && used < sizeof ac[na].name)
                                {   /* the form: "+s" serial, else its name */
                                    if (!ac[na].mt)
                                        snprintf(ac[na].name + used, sizeof ac[na].name - used, "+s");
                                    else if (ro == 4 || csk || w > 0)
                                        snprintf(ac[na].name + used, sizeof ac[na].name - used, "/%s",
                                                 w > 0 ? "bands" : "walk");
                                    else
                                        snprintf(ac[na].name + used, sizeof ac[na].name - used, "/%s%s",
                                                 ac[na].natarm == 2 ? "tile" : ac[na].natarm == 1 ? "strips"
                                                 : h->il2d_col.nat ? "block" : "matched",
                                                 (h->il2d_col.nat && !ac[na].natst) ? "/str" : "");
                                    if (ac[na].mt && ac[na].msw > 0)
                                    {
                                        used = strlen(ac[na].name);
                                        snprintf(ac[na].name + used, sizeof ac[na].name - used, "%d", ac[na].msw);
                                    }
                                }
                                na++;
                            }
                        }
                }
            }
        if (!mt)
        {   /* one thread: every configuration in ONE race (min of 2, alternated) */
            for (a = 0; a < na; a++)
                idx[a] = a;
            win = _il2d_axis_heat(ac, idx, na, 2, reps, 0, &best, ans, NULL);
        }
        else
        {   /* at T: PER-FORM CHAMPIONS, THEN THE FINAL (2026-11-19). The route, width and tiles were picked
             * under one threaded form and the threading race then chose the form for that pick alone: a
             * configuration best under another form was never timed in it (256x256 T=2: the two-pass rows
             * chosen under the strips, 6 % slower than the per-row child under the tile served). Each form
             * races every configuration it serves (min of 3, alternated); a threaded champion must engage at
             * T (the threading race's own first question); the champions and the serial walks' champion then
             * meet in one final on the full execute (min of 5) -- the route and the threaded form one verdict,
             * banked together (cmt, mtarm, msw, nls beside the axis tokens). */
            int gw[32], ng = 0, g, pass;
            double gns[32];
            /* the pruning law (policy_il.h): a composite or odd column above the L1d races no serial walk at T;
             * where no threaded form engages, the second pass races the serial walks alone (the plane cannot thread) */
            const int serial_ok = cand || vfft_policy_il2d_serial_races(N1, N2, h->nthreads);
            for (pass = 0; pass < 2 && !(pass && (serial_ok || ng)); pass++)
            for (g = pass ? 0 : (cand || !serial_ok) ? 1 : 0; g < (pass ? 1 : 21) && ng < 32; g++)
            {
                int n = 0, w2, gr;
                double t2 = 1e300;
                for (a = 0; a < na; a++)
                    if ((g == 0) ? !ac[a].mt : (ac[a].mt && ac[a].grp == g))
                        idx[n++] = a;
                if (n == 0)
                    continue;
                if (zrsp) _il2d_ip_reset(zrsp);
                gr = _il2d_race_reps_ns(_il2d_arm_axis, &ac[idx[0]], zrsp ? (double)T : 0.0, VFFT_IL2D_MT_SAMPLE_NS);
                w2 = _il2d_axis_heat(ac, idx, n, 3, gr, 2, &t2, NULL, zrsp);
                if (w2 < 0)
                    continue;
                if (ac[w2].mt)
                {
                    _il2d_axis_arm_set(&ac[w2]);
                    if (!_il2d_c2c_mt(h, z, zo, VFFT_FORWARD, h->nthreads))
                        continue;   /* cannot engage at T: no arm of the final */
                }
                gns[ng] = t2;
                gw[ng++] = w2;
            }
            {
                double fns[32];
                int fr = 1, fw;
                for (g = 0; g < ng; g++)
                {   /* the final's samples: ~1 ms of its FASTEST champion */
                    int r2;
                    if (zrsp) _il2d_ip_reset(zrsp);
                    r2 = _il2d_race_reps_ns(_il2d_arm_axis, &ac[gw[g]], zrsp ? (double)T : 0.0, VFFT_IL2D_MT_SAMPLE_NS);
                    if (r2 > fr) fr = r2;
                }
                fw = _il2d_axis_heat(ac, gw, ng, 5, fr, 2, &best, fns, zrsp);
                win = fw;
                if (getenv("VFFT_IL2D_LOG"))
                {
                    fprintf(stderr, "[il2d] axis race %dx%d (%s, T=%d%s): %d arm(s), %d form champion(s):", N1, N2,
                            h->il2d_col.nat ? "nat" : "scr", h->nthreads, cand ? ", a candidate chain" : "", na, ng);
                    for (g = 0; g < ng; g++)
                        fprintf(stderr, " %s=%.0f(%.0f)", ac[gw[g]].name, fns[g], gns[g]);
                    fprintf(stderr, "\n");
                }
            }
        }
        if (win >= 0)
        {
            bwl = ac[win].wl;
            bro = ac[win].ro;
            bwc = ac[win].wc;
            bkb = ac[win].ro == 3 ? ac[win].kb : 8;
            bcsk = ac[win].csk;
        }
        if (getenv("VFFT_IL2D_LOG"))
        {
            if (!mt)
            {
                fprintf(stderr, "[il2d] axis race %dx%d (%s, T=%d):", N1, N2, h->il2d_col.nat ? "nat" : "scr", 1);
                for (a = 0; a < na; a++)
                    fprintf(stderr, " %s=%.0f", ac[a].name, ans[a]);
                fprintf(stderr, " -> wl=%d sw=%d ro=%d\n", bwl, bwc, bro);
            }
            else
                fprintf(stderr, "[il2d] axis race %dx%d (%s, T=%d) -> wl=%d sw=%d ro=%d, %s (%.0f ns)\n", N1, N2,
                        h->il2d_col.nat ? "nat" : "scr", h->nthreads, bwl, bwc, bro, win >= 0 ? ac[win].name : "-",
                        best);
        }
        if (mt)
        {   /* the joint verdict's threaded form (the serial walk: colmt 0, the leaf staged) */
            const int wm = win >= 0 && ac[win].mt;
            h->il2d_col.colmt = wm;
            h->il2d_col.natarm = wm ? ac[win].natarm : 0;
            h->il2d_col.msw = wm ? ac[win].msw : 0;
            h->il2d_col.natst = wm ? ac[win].natst : 1;
        }
        if (win_out && win >= 0)
            *win_out = ac[win];
        if (best_out)
            *best_out = best;
        free(ac);
        free(idx);
        free(ans);
    }
    if (cand)
    {   /* a candidate chain's champion only: nothing set, nothing banked */
        if (zo != z) vfft_aligned_free(zo);
        vfft_aligned_free(z);
        vfft_aligned_free(zseed);
        return best < 1e299;
    }
    /* set the winner (at T > 1 the threaded form is the joint verdict's, set above: colmt, natarm, msw, natst) */
    _il2d_axis_set_cfg(h, N2, bwl, bro, bwc, bkb, bcsk);
    _il2d_axis_bank(h, W, cfg, N1, N2, mt, bwc, bkb, best);
    if (zo != z) vfft_aligned_free(zo);
    vfft_aligned_free(z);
    vfft_aligned_free(zseed);
    return mt;   /* 1: the threading verdict decided with the route (the create replays it from the row) */
}

/* ── THE CHAIN RACE AT T (2026-11-19; the owner, 11-18: "a correct version needs the whole transform in the
 * race"). The column chain was the one-thread race's (_il2d_col_build: the column pass in its serial forms), and a
 * chain that ties or loses there can win threaded by several per cent: same-run on the whole transform at T,
 * 512x512 T=8 8.8.8 under the tile 0.93 of 8.8.4.2, 1000x1000 T=8 5.5.4.10 0.92 of 25.8.5, 1024x1024 T=2/8 0.96.
 * So at T > 1, after the axis race crowned the one-thread chain's configuration, EVERY other chain of the pool
 * (the column build's own enumeration; no rank of the one-thread race filters it) is built on the plan's
 * descriptor -- its stage tables, its per-stage forms raced as the column build races them, its natural leaf --
 * and raced in its threaded forms by the same axis race (_il2d_axis_race cand = 1: per-form champions and their
 * final, nothing banked); the serial walk at T is the one-thread walk, whose chain the one-thread race decided. The
 * chains' champions and the one-thread chain's verdict meet in one final on the full execute (min of 5, ~1 ms
 * samples, one untimed execute before each); a new chain is installed on the plan (the old one's tables freed)
 * and its verdict banked on the cell's row at T in the axis race's tokens, its forms handed back for the create's
 * re-bank. Chains with a Rader stage, Bluestein and the turned pass keep their verdict (no pool to race). */
typedef struct
{
    int nst, R[8], L[8], nat;
    vfft_il2p_fn f[8], b[8];
    vfft_real *tf[8], *tb[8];
    int *natperm;
    vfft_real *natscr, *natstage;
    int own_natscr;           /* natscr is this state's (the plan's descriptor had none) */
    char forms[64];
} _il2d_chainst_t;
static void _il2d_chainst_get(const vfft_ilcol_t *c, _il2d_chainst_t *s)
{
    s->nst = c->nst;
    memcpy(s->R, c->R, sizeof s->R);
    memcpy(s->L, c->L, sizeof s->L);
    memcpy(s->f, c->f, sizeof s->f);
    memcpy(s->b, c->b, sizeof s->b);
    memcpy(s->tf, c->tf, sizeof s->tf);
    memcpy(s->tb, c->tb, sizeof s->tb);
    s->nat = c->nat;
    s->natperm = c->natperm;
    s->natscr = c->natscr;
    s->natstage = c->natstage;
}
static void _il2d_chainst_put(vfft_ilcol_t *c, const _il2d_chainst_t *s)
{
    c->nst = s->nst;
    memcpy(c->R, s->R, sizeof c->R);
    memcpy(c->L, s->L, sizeof c->L);
    memcpy(c->f, s->f, sizeof c->f);
    memcpy(c->b, s->b, sizeof c->b);
    memcpy(c->tf, s->tf, sizeof c->tf);
    memcpy(c->tb, s->tb, sizeof c->tb);
    c->nat = s->nat;
    c->natperm = s->natperm;
    c->natscr = s->natscr;
    c->natstage = s->natstage;
}
static void _il2d_chainst_free(_il2d_chainst_t *s)
{
    int k;
    for (k = 0; k < s->nst && k < 8; k++)
    {
        vfft_aligned_free(s->tf[k]);
        vfft_aligned_free(s->tb[k]);
    }
    free(s->natperm);
    if (s->own_natscr)
        vfft_aligned_free(s->natscr);
    _il2d_stage_free(s->natstage);
    memset(s, 0, sizeof *s);
}
/* a candidate chain's state as the column build makes it: the kernels resolved, the stage tables, the per-stage
 * forms raced (the column build's race, at one thread as it runs for a c2c cell), and where the cell is natural and
 * the chain has two stages or more the natural leaf (its permutation, the pre-leaf plane -- the descriptor's, of the
 * same size, or its own -- and the leaf's staging, T slots of R_last x N2). 0 = the chain does not build. */
static int _il2d_chainst_build(int N1, int N2, const int *R, int nst, int nat_req, int T, const vfft_ilcol_t *prim,
                               _il2d_chainst_t *s)
{
    memset(s, 0, sizeof *s);
    if (nst < 1 || nst > 8)
        return 0;
    s->nst = nst;
    memcpy(s->R, R, (size_t)nst * sizeof(int));
    if (!_il2d_resolve(s->R, nst, s->f, s->b))
        return 0;
    if (_il2d_build_tables(N1, nst, s->R, s->L, s->tf, s->tb))
    {
        memset(s, 0, sizeof *s);
        return 0;
    }
    (void)_il2d_race_forms(N1, (size_t)N2, s->R, nst, s->f, s->b, s->forms, sizeof s->forms, 0, 1, nat_req && nst > 1, 0);
    if (nat_req && nst > 1)
    {
        s->natperm = _il2d_nat_perm(s->R, nst, N1);
        s->natscr = prim->natscr;
        if (!s->natscr)
        {
            s->natscr = (vfft_real *)vfft_aligned_alloc(2 * (size_t)N1 * VFFT_IL2D_NAT_CAP(N2) * sizeof(vfft_real));
            s->own_natscr = 1;
        }
        s->natstage = _il2d_stage_alloc(T, s->R[nst - 1], (size_t)N2);
        if (!s->natperm || !s->natscr || !s->natstage)
        {
            _il2d_chainst_free(s);
            return 0;
        }
        s->nat = 1;
    }
    return 1;
}
typedef struct { _il2d_axis_arm_t a; const _il2d_chainst_t *st; } _il2d_chain_arm_t;
static void _il2d_arm_chain_t(void *v)
{   /* the arm's chain onto the descriptor, then its configuration's full execute */
    _il2d_chain_arm_t *c = (_il2d_chain_arm_t *)v;
    _il2d_chainst_put(&c->a.h->il2d_col, c->st);
    _il2d_arm_axis(&c->a);
}
/* the race of the arms a[0..n) under proto: at most VFFT_RACE_MAX_ARMS a race, a larger set in balanced interleaved
 * chunks whose winners race again the same way. The winner's index, its time in *bns. */
static int _il2d_heat_arms(const vfft_race_arm_t *a, int n, const vfft_race_proto_t *proto, double *bns)
{
    vfft_race_arm_t arms[VFFT_RACE_MAX_ARMS];
    double ns[VFFT_RACE_MAX_ARMS];
    int wins[VFFT_RACE_MAX_ARMS], nw = 0, c0, i, m, best;
    const int nch = (n + VFFT_RACE_MAX_ARMS - 1) / VFFT_RACE_MAX_ARMS;
    if (n <= 0)
        return -1;
    for (c0 = 0; c0 < nch; c0++)
    {
        for (m = 0, i = c0; i < n; i += nch)
            arms[m++] = a[i];
        vfft_race_run(proto, arms, m, ns);
        for (best = 0, i = 1; i < m; i++)
            if (ns[i] < ns[best]) best = i;
        wins[nw++] = c0 + best * nch;
        *bns = ns[best];
    }
    if (nw == 1)
        return wins[0];
    for (i = 0; i < nw; i++)
        arms[i] = a[wins[i]];
    vfft_race_run(proto, arms, nw, ns);
    for (best = 0, i = 1; i < nw; i++)
        if (ns[i] < ns[best]) best = i;
    *bns = ns[best];
    return wins[best];
}
/* prim / prim_ns: the axis race's verdict for the plan's chain (its winning arm); fm: the plan's forms in, the
 * served chain's out */
static void _il2d_chain_race_t(struct vfft_plan_s *h, struct vfft_wisdom_s *W, const vfft_cfg_t *cfg, int N1, int N2,
                               const _il2d_axis_arm_t *prim, double prim_ns, char *fm, size_t fmsz)
{
    static int cand[VFFT_IL2D_POOL_MAX][8], lens[VFFT_IL2D_POOL_MAX];   /* off the stack: the create is single-threaded */
    int cur[8], ncand = 0, dropped = 0, i, k, nk = 0, win = 0, built = 0;
    const int T = h->nthreads;
    const int nat_req = vfft_policy_rankn_axis_nat(2, 0, vfft_policy_ord_rankn(cfg));
    const size_t PN = (size_t)N1 * (size_t)N2;
    _il2d_chainst_t prim_st, *st = NULL;
    _il2d_chain_arm_t *ca = NULL;
    vfft_race_arm_t *arms = NULL;
    vfft_real *z = NULL, *zo = NULL, *zseed = NULL;
    _il2d_ip_reset_t zrs;
    double t0 = vfft_now_ns(), bns = 1e300;
    if (T < 2 || !prim || h->il2d_col.blu || h->il2d_col.tpc || h->il2d_col.rd0 || h->il2d_col.nst < 1)
        return;
    _il2d_enum_rec(N1, 0, cur, cand, lens, &ncand, &dropped);
    if (ncand < 2)
        return;
    st = (_il2d_chainst_t *)calloc((size_t)ncand + 1, sizeof *st);
    ca = (_il2d_chain_arm_t *)calloc((size_t)ncand + 1, sizeof *ca);
    arms = (vfft_race_arm_t *)calloc((size_t)ncand + 1, sizeof *arms);
    if (!st || !ca || !arms)
        goto done;
    memset(&prim_st, 0, sizeof prim_st);
    _il2d_chainst_get(&h->il2d_col, &prim_st);
    snprintf(prim_st.forms, sizeof prim_st.forms, "%s", fm ? fm : "");
    ca[0].a = *prim;
    ca[0].st = &prim_st;
    nk = 1;
    for (i = 0; i < ncand; i++)
    {
        _il2d_axis_arm_t w;
        double wns = 1e300;
        if (lens[i] == prim_st.nst && !memcmp(cand[i], prim_st.R, (size_t)lens[i] * sizeof(int)))
            continue;   /* the plan's own chain: its verdict is prim */
        if (!_il2d_chainst_build(N1, N2, cand[i], lens[i], nat_req, T, &h->il2d_col, &st[nk]))
            continue;
        built++;
        _il2d_chainst_put(&h->il2d_col, &st[nk]);
        if (st[nk].nat)
            (void)_il2d_nat_sscr_build(&h->il2d_col, N1, (size_t)N2, T);   /* the dense strips' scratch, chain-free */
        memset(&w, 0, sizeof w);
        k = _il2d_axis_race(h, W, cfg, N1, N2, 1, &w, &wns);
        _il2d_chainst_put(&h->il2d_col, &prim_st);
        if (!k)
        {
            _il2d_chainst_free(&st[nk]);
            continue;
        }
        ca[nk].a = w;
        ca[nk].st = &st[nk];
        nk++;
    }
    if (nk < 2)
        goto done;
    /* THE FINAL: every chain's champion and the plan's own verdict, one race on the full execute */
    z = (vfft_real *)vfft_aligned_alloc(2 * PN * sizeof(vfft_real));
    zo = (h->placement == VFFT_OUTOFPLACE) ? (vfft_real *)vfft_aligned_alloc(2 * PN * sizeof(vfft_real)) : z;
    if (!z || !zo)
        goto done;
    for (i = 0; i < (int)(2 * PN); i++)
        z[i] = 1.0 + 1e-6 * (double)(i & 1023);
    memset(&zrs, 0, sizeof zrs);
    if (zo == z && (zseed = (vfft_real *)vfft_aligned_alloc(2 * PN * sizeof(vfft_real))) != NULL)
    {   /* in place: the plane re-laid before every sample */
        memcpy(zseed, z, 2 * PN * sizeof(vfft_real));
        zrs.p = z;
        zrs.seed = zseed;
        zrs.n = 2 * PN;
    }
    {
        int reps = 1;
        for (k = 0; k < nk; k++)
        {
            int r2;
            ca[k].a.z = z;
            ca[k].a.zo = zo;
            arms[k].name = ca[k].a.name;
            arms[k].run = _il2d_arm_chain_t;
            arms[k].ctx = &ca[k];
            if (zseed) _il2d_ip_reset(&zrs);
            r2 = _il2d_race_reps_ns(_il2d_arm_chain_t, &ca[k], zseed ? (double)PN : 0.0, VFFT_IL2D_MT_SAMPLE_NS);
            if (r2 > reps) reps = r2;   /* ~1 ms of the FASTEST arm */
        }
        {
            const vfft_race_proto_t proto = { 5, reps, VFFT_RACE_MIN, 1, 2, zseed ? _il2d_ip_reset : NULL,
                                              zseed ? &zrs : NULL, 0, 1 };
            win = _il2d_heat_arms(arms, nk, &proto, &bns);
        }
    }
    if (win < 0)
        win = 0;
    if (getenv("VFFT_IL2D_LOG"))
    {
        int q;
        fprintf(stderr, "[il2d] chain race at T=%d %dx%d: %d chain(s) in the pool, %d built, %d champion(s), %.1f s ->",
                T, N1, N2, ncand, built + 1, nk, (vfft_now_ns() - t0) / 1e9);
        for (q = 0; q < ca[win].st->nst; q++)
            fprintf(stderr, "%s%d", q ? "." : " ", ca[win].st->R[q]);
        fprintf(stderr, " %s %.0f ns (the one-thread chain's verdict %.0f in its axis race)%s\n", ca[win].a.name, bns,
                prim_ns, win ? "" : " -- kept");
    }
    if (win == 0)
        _il2d_chainst_put(&h->il2d_col, &prim_st);   /* kept: the final left its last arm's chain on the descriptor */
    if (win > 0)
    {   /* the new chain onto the plan: the one-thread chain's tables, permutation and staging go, its plane stays when
         * the new chain shares it; its forms back to the create (the row's forms= re-bank) */
        _il2d_chainst_t old = prim_st;
        if (fm && fmsz)
            snprintf(fm, fmsz, "%s", st[win].forms);
        _il2d_chainst_put(&h->il2d_col, &st[win]);
        if (st[win].natscr == old.natscr)
            old.natscr = NULL;   /* shared: the plan keeps it */
        old.own_natscr = 1;      /* the descriptor's own: freed here unless shared */
        _il2d_chainst_free(&old);
        memset(&st[win], 0, sizeof st[win]);   /* the plan owns it now */
        ca[win].st = NULL;
    }
    /* the winner's configuration onto the plan (the final ran every arm on it), and its verdict on the row */
    {
        const _il2d_axis_arm_t *wa = &ca[win].a;
        const int kb = wa->ro == 3 ? wa->kb : 8;
        _il2d_axis_arm_set(wa);
        _il2d_axis_set_cfg(h, N2, wa->wl, wa->ro, wa->wc, kb, wa->csk);
        h->il2d_col.colmt = wa->mt;
        h->il2d_col.natarm = wa->mt ? wa->natarm : 0;
        h->il2d_col.msw = wa->mt ? wa->msw : 0;
        h->il2d_col.natst = wa->mt ? wa->natst : 1;
        if (win > 0)
            _il2d_axis_bank(h, W, cfg, N1, N2, 1, wa->wc, kb, bns);
    }
done:
    if (st)
        for (k = 1; k < nk; k++)   /* the champions' states (the winner's moved to the plan: zeroed) */
            if (st[k].nst)
                _il2d_chainst_free(&st[k]);
    free(st);
    free(ca);
    free(arms);
    if (zo && zo != z) vfft_aligned_free(zo);
    vfft_aligned_free(z);
    vfft_aligned_free(zseed);
}

/* the children's recipes onto the cell's row (wisdom2_child.h): the ones
 * that raced in this create, and any the row did not carry yet; saved like
 * any winner. The 2D create's last act. */
static void _il2d_children_put(struct vfft_wisdom_s *W, const vfft_cfg_t *cfg, struct vfft_plan_s *h,
                               int N1, int N2, int ord, int T)
{
    vw2_ilcol_key_t ck;
    vw2_key_t pk;
    int changed = 0;
    if (!W || W->vw2_off_2d)
        return;
    memset(&ck, 0, sizeof ck);
    ck.rank = 2; ck.n0 = N1; ck.n1 = N2; ck.ord = ord; ck.nthreads = T;
    vw2__ilcol_key(&ck, &pk);
    changed |= vfft_child_row_update(&W->vw2, &pk, "rp_", h->il2d_rowS);
    changed |= vfft_child_row_update(&W->vw2, &pk, "turn_", h->il2d_turnS);
    changed |= vfft_child_row_update(&W->vw2, &pk, "csk_", h->il2d_cskS);
    changed |= vfft_child_row_update(&W->vw2, &pk, "tpc_", h->il2d_col.tpcS);
    if (changed)
        _vw2_persist(W, cfg);
}

/* the real door's twin: the row child (rp_*, or rp_c2r_* for a c2r plan:
 * the two directions' rows differ) and the row engines (rx_*, or rx_c2r_*:
 * each direction's row plan has its own) onto the real row. */
static void _il2d_real_children_put(struct vfft_wisdom_s *W, const vfft_cfg_t *cfg, struct vfft_plan_s *h,
                                    int N1, int N2, int ord, int T)
{
    vw2_ilcol_key_t ck;
    vw2_key_t pk;
    int changed = 0;
    if (!W || W->vw2_off_2d)
        return;
    memset(&ck, 0, sizeof ck);
    ck.rank = 2; ck.n0 = N1; ck.n1 = N2; ck.ord = ord; ck.real = 1; ck.nthreads = T; ck.ip = h->il2d_ip;
    vw2__ilcol_key(&ck, &pk);
    changed |= vfft_child_row_update(&W->vw2, &pk, h->transform == VFFT_C2R ? "rp_c2r_" : "rp_", h->il2d_rowS);
    changed |= vfft_child_row_update(&W->vw2, &pk, h->transform == VFFT_C2R ? "rx_c2r_" : "rx_", h->il2d_rxS);
    if (h->il2d_rax_on)
    {   /* the real axis on N1 (il2d_real_axis.h): its column chain and its row batch, when the form serves */
        changed |= vfft_child_row_update(&W->vw2, &pk, "rax_", h->il2d_raxS);
        changed |= vfft_child_row_update(&W->vw2, &pk, "raxr_", h->il2d_rax_rowS);
    }
    if (changed)
        _vw2_persist(W, cfg);
}

/* ── c2c MT clones. Worker t > 0 needs its own row child: the
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
static int _il2d_clone_set(const struct vfft_plan_s *prim, struct vfft_wisdom_s *S, const vfft_cfg_t *cfg,
                           int N, int placement, int T,
                           struct vfft_plan_s ***out, int *out_n, const char *what)
{
    const int n = (T > 64 ? 64 : T) - 1;
    vfft_cfg_t rc;
    struct vfft_plan_s **arr;
    int t;
    if (n <= 0 || *out || !prim || !S)
        return 0;
    memset(&rc, 0, sizeof rc);
    rc.transform = VFFT_C2C;
    rc.placement = placement;
    rc.rigor = cfg->rigor;
    rc.dims = 1;
    rc.n[0] = N;
    rc.howmany = 1;
    rc.order = VFI_ORDER_NATURAL;
    rc.layout = VFI_LAYOUT_IL;
    rc.nthreads = 1;
    rc.wisdom = (vfft_wisdom *)S; /* the child's own store: a clone replays its recipe, never a 1D row */
    rc.wisdom_write = 0;
    arr = (struct vfft_plan_s **)calloc((size_t)n, sizeof *arr);
    if (!arr)
        return 0;
    for (t = 0; t < n; t++)
    {
        struct vfft_plan_s *c = (struct vfft_plan_s *)_vfft_create_cfg(&rc);
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
/* the set the banked route needs: the turn's N1 plan, the skewed
 * pass's OOP row plan on route 0 (its batched routes run stateless kernels on
 * their own slots), else the chain walk's in-place row child */
static void _il2d_c2c_build_clones(struct vfft_plan_s *h,
                                   const vfft_cfg_t *cfg, int T)
{
    if (h->il2d_turn)
    {
        _il2d_clone_set(h->il2d_turn_plan, h->il2d_turnS, cfg, h->N, VFFT_INPLACE, T,
                        &h->il2d_turnw, &h->il2d_turnw_n, "turn");
        return;
    }
    if (h->il2d_csk)
    {
        if (!h->il2d_rowb && !h->il2d_rowb2)
            _il2d_clone_set(h->il2d_csk_row, h->il2d_cskS, cfg, h->N2, VFFT_OUTOFPLACE, T,
                            &h->il2d_cskw, &h->il2d_cskw_n, "csk row");
        return;
    }
    _il2d_clone_set(h->il2d_row, h->il2d_rowS, cfg, h->N2, VFFT_INPLACE, T,
                    &h->il2d_roww, &h->il2d_roww_n, "row");
}
/* the T-aware axis race needs every route's set BEFORE it runs:
 * an arm whose set is missing runs serial -- what execute would do, but not
 * the comparison the race is for. After the race the sets the banked route
 * does not run are dropped. */
static void _il2d_c2c_build_clone_sets_all(struct vfft_plan_s *h, const vfft_cfg_t *cfg, int T)
{
    _il2d_clone_set(h->il2d_row, h->il2d_rowS, cfg, h->N2, VFFT_INPLACE, T, &h->il2d_roww, &h->il2d_roww_n, "row");
    if (h->il2d_csk_row)
        _il2d_clone_set(h->il2d_csk_row, h->il2d_cskS, cfg, h->N2, VFFT_OUTOFPLACE, T, &h->il2d_cskw, &h->il2d_cskw_n, "csk row");
    if (h->il2d_turn_plan)
        _il2d_clone_set(h->il2d_turn_plan, h->il2d_turnS, cfg, h->N, VFFT_INPLACE, T, &h->il2d_turnw, &h->il2d_turnw_n, "turn");
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

/* ── THE LEAF AT ONE THREAD (2026-11-24) ────────────────────────────────────────────────────────────────────
 * The natural leaf writes a block's rows into its staging and the rows run there (staged), or it stores them at
 * their natural stride and the rows run in the plane (strided). The threaded forms race the two (nls=); the
 * one-thread walk served the staged leaf unraced. Measured on the 14900KF (AVX2, one thread), the same plan with
 * the strided leaf, outputs bitwise equal: 18x1050 44.8 -> 41.2 us, 42x2058 245 -> 220, 18x4374 254 -> 235,
 * 42x8750 1211 -> 1096, 70x1050 186 -> 170, 150x1050 416 -> 375, 270x1050 862 -> 814 (6 to 10 % on short columns
 * with long rows; the staging was fitted on power-of-two planes, whose strided stores alias).
 * So a cold cell's one-thread plan times both on its whole forward transform, once its walk is settled: min of 7
 * alternated samples of about 1 ms, an untimed execute before each. The winner stays and is banked as nls= on the
 * cell's row, the token the door already reads at one thread. */
typedef struct
{
    struct vfft_plan_s *h;
    vfft_real *z, *zo;
    int natst;
} _il2d_leaf_arm_t;
static void _il2d_arm_leaf(void *v)
{
    _il2d_leaf_arm_t *c = (_il2d_leaf_arm_t *)v;
    c->h->il2d_col.natst = c->natst;
    _vfft_exec6((vfft_plan)c->h, VFFT_FORWARD, c->z, NULL, c->zo, NULL);
}
static void _il2d_leaf_race_1t(struct vfft_plan_s *h, struct vfft_wisdom_s *W, const vfft_cfg_t *cfg, int N1, int N2)
{
    const size_t PN = (size_t)N1 * (size_t)N2;
    const int ord = vfft_policy_ord_rankn(cfg);
    _il2d_leaf_arm_t la[2];
    vfft_race_arm_t arms[2];
    double ns[2] = { 0, 0 };
    vfft_real *z, *zo, *zseed = NULL;
    _il2d_ip_reset_t zrs;
    size_t q;
    int k, reps = 1, win = 0;
    if (h->transform != VFFT_C2C || h->nthreads > 1 || !W || W->vw2_off_2d || !h->il2d_col.nat || !h->il2d_col.natstage ||
        h->il2d_turn || h->il2d_csk || h->il2d_col.nst < 2)
        return;
    z = (vfft_real *)vfft_aligned_alloc(2 * PN * sizeof(vfft_real));
    zo = (h->placement == VFFT_OUTOFPLACE) ? (vfft_real *)vfft_aligned_alloc(2 * PN * sizeof(vfft_real)) : z;
    if (!z || !zo)
    {
        if (zo != z) vfft_aligned_free(zo);
        vfft_aligned_free(z);
        return;
    }
    for (q = 0; q < 2 * PN; q++)
        z[q] = 1.0 + 1e-6 * (double)(q & 1023);
    memset(&zrs, 0, sizeof zrs);
    if (zo == z && (zseed = (vfft_real *)vfft_aligned_alloc(2 * PN * sizeof(vfft_real))) != NULL)
    {   /* in place: the plane re-laid before every sample */
        memcpy(zseed, z, 2 * PN * sizeof(vfft_real));
        zrs.p = z;
        zrs.seed = zseed;
        zrs.n = 2 * PN;
    }
    if (zo != z || zseed)
    {
        for (k = 0; k < 2; k++)
        {
            int r2;
            la[k].h = h; la[k].z = z; la[k].zo = zo; la[k].natst = 1 - k;   /* arm 0 staged, arm 1 strided */
            arms[k].name = k ? "strided" : "staged";
            arms[k].run = _il2d_arm_leaf;
            arms[k].ctx = &la[k];
            if (zseed) _il2d_ip_reset(&zrs);
            r2 = _il2d_race_reps_ns(_il2d_arm_leaf, &la[k], zseed ? (double)PN : 0.0, VFFT_IL2D_MT_SAMPLE_NS);
            if (r2 > reps) reps = r2;
        }
        {
            const vfft_race_proto_t proto = { 7, reps, VFFT_RACE_MIN, 1, 2, zseed ? _il2d_ip_reset : NULL,
                                              zseed ? &zrs : NULL, 1, 1 };
            vfft_race_run(&proto, arms, 2, ns);
        }
        win = ns[1] < ns[0];
        if (getenv("VFFT_IL2D_LOG"))
            fprintf(stderr, "[il2d] leaf at one thread %dx%d: staged %.0f ns, strided %.0f ns -> %s\n", N1, N2, ns[0], ns[1],
                    win ? "strided" : "staged");
    }
    h->il2d_col.natst = win ? 0 : 1;
    if (vw2_2d_il_tok_seti(&W->vw2, N1, N2, ord, 1, "nls", h->il2d_col.natst) == 0)
        _vw2_persist(W, cfg);
    if (zo != z) vfft_aligned_free(zo);
    vfft_aligned_free(z);
    vfft_aligned_free(zseed);
}

/* ── THE ROWS-FIRST WALK AT ONE THREAD (2026-11-24) ─────────────────────────────────────────────────────────
 * The walk of _il2d_rfw_exec (the rows source -> destination, then the column chain in place on the destination in
 * strips) is an arm on a composite or odd column where a strip fits the L1d twice (policy_il.h,
 * vfft_policy_il2d_rowsfirst_widths: the measurements and the widths). A cold cell's one-thread plan, its axes and
 * its leaf settled, times the walk it stands with against the rows-first walk at each of those widths on its whole
 * forward transform: min of 7 alternated samples of about 1 ms, an untimed execute before each. The winner stays and
 * is banked as rfw= on the cell's row (the strip width; 0 = the walk the axes name), replayed by _il2d_rfw_set.
 * Out of place on the per-row route the rows run through the out-of-place K=1 plan at N2 (il2d_csk_row, csk_* on the
 * row: the create builds it for this arm where the skewed pass has none); when the walk loses, that plan and the
 * strip scratch go. */
typedef struct
{
    struct vfft_plan_s *h;
    vfft_real *z, *zo;
    int rfw;
} _il2d_rfw_arm_t;
static void _il2d_arm_rfw(void *v)
{
    _il2d_rfw_arm_t *c = (_il2d_rfw_arm_t *)v;
    c->h->il2d_rfw = c->rfw;
    _vfft_exec6((vfft_plan)c->h, VFFT_FORWARD, c->z, NULL, c->zo, NULL);
}
/* a banked width onto the plan: the walk when the plan can run it and its scratch builds, else off */
static void _il2d_rfw_set(struct vfft_plan_s *h, int w)
{
    h->il2d_rfw = 0;
    if (w > 0 && w < h->N2 && _il2d_rfw_ok(h) && _il2d_rfw_scr_build(&h->il2d_col, h->N, w))
        h->il2d_rfw = w;
}
static void _il2d_rfw_race_1t(struct vfft_plan_s *h, struct vfft_wisdom_s *W, const vfft_cfg_t *cfg, int N1, int N2)
{
    const size_t PN = (size_t)N1 * (size_t)N2;
    const int ord = vfft_policy_ord_rankn(cfg);
    _il2d_rfw_arm_t ra[4];
    vfft_race_arm_t arms[4];
    char nm[4][16];
    double ns[4] = { 0, 0, 0, 0 };
    vfft_real *z = NULL, *zo = NULL, *zseed = NULL;
    _il2d_ip_reset_t zrs;
    size_t q;
    int wd[3], nw, k, reps = 1, win = 0;
    h->il2d_rfw = 0;
    if (!W || W->vw2_off_2d)
        return;
    nw = _il2d_rfw_ok(h) ? vfft_policy_il2d_rowsfirst_widths(N1, N2, h->nthreads, wd, 3) : 0;
    if (nw > 0 && !_il2d_rfw_scr_build(&h->il2d_col, N1, wd[0]))   /* wd[0]: the widest */
        nw = 0;
    if (nw > 0)
    {
        z = (vfft_real *)vfft_aligned_alloc(2 * PN * sizeof(vfft_real));
        zo = (h->placement == VFFT_OUTOFPLACE) ? (vfft_real *)vfft_aligned_alloc(2 * PN * sizeof(vfft_real)) : z;
    }
    memset(&zrs, 0, sizeof zrs);
    if (nw > 0 && z && zo)
    {
        for (q = 0; q < 2 * PN; q++)
            z[q] = 1.0 + 1e-6 * (double)(q & 1023);
        if (zo == z && (zseed = (vfft_real *)vfft_aligned_alloc(2 * PN * sizeof(vfft_real))) != NULL)
        {   /* in place: the plane re-laid before every sample */
            memcpy(zseed, z, 2 * PN * sizeof(vfft_real));
            zrs.p = z;
            zrs.seed = zseed;
            zrs.n = 2 * PN;
        }
    }
    if (nw > 0 && z && zo && (zo != z || zseed))
    {
        for (k = 0; k <= nw; k++)
        {
            int r2;
            ra[k].h = h; ra[k].z = z; ra[k].zo = zo; ra[k].rfw = k ? wd[k - 1] : 0;   /* arm 0: the walk it stands with */
            if (k) snprintf(nm[k], sizeof nm[k], "w%d", wd[k - 1]);
            else snprintf(nm[k], sizeof nm[k], "served");
            arms[k].name = nm[k];
            arms[k].run = _il2d_arm_rfw;
            arms[k].ctx = &ra[k];
            if (zseed) _il2d_ip_reset(&zrs);
            r2 = _il2d_race_reps_ns(_il2d_arm_rfw, &ra[k], zseed ? (double)PN : 0.0, VFFT_IL2D_MT_SAMPLE_NS);
            if (r2 > reps) reps = r2;
        }
        {
            const vfft_race_proto_t proto = { 7, reps, VFFT_RACE_MIN, 1, 2, zseed ? _il2d_ip_reset : NULL,
                                              zseed ? &zrs : NULL, 1, 1 };
            vfft_race_run(&proto, arms, nw + 1, ns);
        }
        for (k = 1; k <= nw; k++)
            if (ns[k] < ns[win]) win = k;
        if (getenv("VFFT_IL2D_LOG"))
        {
            fprintf(stderr, "[il2d] rows first at one thread %dx%d:", N1, N2);
            for (k = 0; k <= nw; k++)
                fprintf(stderr, " %s=%.0f%s", nm[k], ns[k], k == win ? "*" : "");
            fprintf(stderr, "\n");
        }
    }
    h->il2d_rfw = win ? wd[win - 1] : 0;
    if (!h->il2d_rfw)
    {   /* the walk the axes name stays: the strip scratch goes, and the out-of-place row plan built for this arm alone */
        _il2d_nat_sscr_free(&h->il2d_col);
        if (h->il2d_csk_row && !h->il2d_csk_scr)
        {
            vfft_destroy(h->il2d_csk_row);
            h->il2d_csk_row = NULL;
            vfft_child_store_free(h->il2d_cskS);
            h->il2d_cskS = NULL;
        }
    }
    if (nw > 0 && vw2_2d_il_tok_seti(&W->vw2, N1, N2, ord, 1, "rfw", h->il2d_rfw) == 0)
        _vw2_persist(W, cfg);
    if (zo != z) vfft_aligned_free(zo);
    vfft_aligned_free(z);
    vfft_aligned_free(zseed);
}

/* ── THE CHILD'S FINAL, IN THE PLANE (owner, 2026-11-24; il/wisdom/wisdom2_child.h, the shortlist) ─────────────
 * The 1D plan the plan's banked route runs -- the row plan (the chain walk's rows), the skewed pass's row plan on
 * its per-row route, the turn plan -- was created from the shortlist's first candidate: the winner of the 1D race
 * at its length. The other candidates of that shortlist now run in its place, each on its own store with its own
 * clones for the workers, and all of them are timed on the plan's WHOLE forward transform as it stands (its chain,
 * walk and threaded form, at its thread count): min of 7 alternated samples of about 1 ms, an untimed execute before
 * each. The fastest stays on the plan; the cell's row gets its recipe under the child's prefix (_il2d_children_put).
 * due_*: the child's recipe is new to this cell (_il2d_child_create) -- a cell that replays its own row has no
 * final. No final where the batched rows serve (no per-row plan runs) or the shortlist has one candidate. */
typedef struct
{
    struct vfft_plan_s *h;
    vfft_real *z, *zo;
    struct vfft_plan_s **slot;      /* the plan's field the child sits in */
    struct vfft_plan_s ***wslot;    /* and its clones' */
    struct vfft_plan_s *p;          /* this arm's child */
    struct vfft_plan_s **w;         /* this arm's clones, wn of them */
    int wn;
    struct vfft_wisdom_s *S;        /* this arm's recipe */
    char name[48];
} _il2d_child_arm_t;
static void _il2d_arm_child(void *v)
{
    _il2d_child_arm_t *c = (_il2d_child_arm_t *)v;
    *c->slot = c->p;
    *c->wslot = c->w;
    _vfft_exec6((vfft_plan)c->h, VFFT_FORWARD, c->z, NULL, c->zo, NULL);
}
static void _il2d_child_final(struct vfft_plan_s *h, struct vfft_wisdom_s *W, const vfft_cfg_t *cfg, int N1, int N2,
                              int due_row, int due_turn, int due_csk)
{
    const int T = h->nthreads > 1 ? h->nthreads : 1;
    const size_t PN = (size_t)N1 * (size_t)N2;
    struct vfft_plan_s **slot, ***wslot;
    struct vfft_wisdom_s **Sslot;
    _il2d_child_arm_t ca[VFFT_SHORTLIST_MAX];
    vfft_race_arm_t arms[VFFT_SHORTLIST_MAX];
    double ns[VFFT_SHORTLIST_MAX], t0;
    vfft_real *z = NULL, *zo = NULL, *zseed = NULL;
    _il2d_ip_reset_t zrs;
    const char *what;
    int *wn, N, ip, due, nc, na = 0, k, win = 0, need_w;
    const vfft_cfg_t *scfg = NULL;
    struct vfft_wisdom_s *SW = _il2d_shortlist_store(W, cfg, &scfg);
    if (h->transform != VFFT_C2C || !SW)
        return;
    if (h->il2d_turn)
    {
        slot = &h->il2d_turn_plan; wslot = &h->il2d_turnw; wn = &h->il2d_turnw_n; Sslot = &h->il2d_turnS;
        N = N1; ip = 1; due = due_turn; what = "turn";
    }
    else if (h->il2d_rowb || h->il2d_rowb2)
        return;
    else if (h->il2d_csk || (h->il2d_rfw > 0 && h->placement == VFFT_OUTOFPLACE))
    {   /* the out-of-place row plan: the skewed pass's rows, and the rows-first walk's move */
        slot = &h->il2d_csk_row; wslot = &h->il2d_cskw; wn = &h->il2d_cskw_n; Sslot = &h->il2d_cskS;
        N = N2; ip = 0; due = due_csk; what = h->il2d_csk ? "csk row" : "rows-first row";
    }
    else
    {
        slot = &h->il2d_row; wslot = &h->il2d_roww; wn = &h->il2d_roww_n; Sslot = &h->il2d_rowS;
        N = N2; ip = 1; due = due_row; what = "row";
    }
    if (!due || !*slot || !*Sslot)
        return;
    nc = vfft_shortlist_count(&SW->vw2, VW2_T_C2C, N, ip, 1);
    if (nc < 2)
        return;
    t0 = vfft_now_ns();   /* from here the create measures: a cell with no final reads no clock */
    need_w = (T > 1 && *wn > 0);   /* the plan holds a clone a worker: every candidate brings its own */
    memset(ca, 0, sizeof ca);
    ca[0].p = *slot; ca[0].w = *wslot; ca[0].wn = *wn; ca[0].S = *Sslot;
    snprintf(ca[0].name, sizeof ca[0].name, "1:%s", _vfft_plan_route((vfft_plan)*slot));
    na = 1;
    for (k = 2; k <= nc && na < VFFT_SHORTLIST_MAX; k++)
    {   /* candidate k: its recipe replayed (a create that races it is not that candidate: left out) */
        _il2d_child_arm_t *a = &ca[na];
        vfft_cfg_t cc;
        a->S = vfft_shortlist_get(&SW->vw2, VW2_T_C2C, N, ip, 1, k);
        if (!a->S)
            continue;
        memset(&cc, 0, sizeof cc);
        cc.transform = VFFT_C2C;
        cc.placement = ip ? VFFT_INPLACE : VFFT_OUTOFPLACE;
        cc.rigor = cfg->rigor;
        cc.dims = 1;
        cc.n[0] = N;
        cc.howmany = 1;
        cc.order = VFI_ORDER_NATURAL;
        cc.layout = VFI_LAYOUT_IL;
        cc.nthreads = 1;
        cc.wisdom = (vfft_wisdom *)a->S;
        cc.wisdom_write = 0;
        a->p = (struct vfft_plan_s *)_vfft_create_cfg(&cc);
        if (a->p && !vfft_child_store_raced(a->S) && need_w &&
            !_il2d_clone_set(a->p, a->S, cfg, N, cc.placement, T, &a->w, &a->wn, what))
        {
            vfft_destroy(a->p);
            a->p = NULL;
        }
        if (!a->p || vfft_child_store_raced(a->S))
        {
            if (a->p)
                vfft_destroy(a->p);
            _il2d_clone_set_drop(&a->w, &a->wn);
            vfft_child_store_free(a->S);
            memset(a, 0, sizeof *a);
            continue;
        }
        snprintf(a->name, sizeof a->name, "%d:%s", k, _vfft_plan_route((vfft_plan)a->p));
        na++;
    }
    if (na >= 2)
    {
        z = (vfft_real *)vfft_aligned_alloc(2 * PN * sizeof(vfft_real));
        zo = (h->placement == VFFT_OUTOFPLACE) ? (vfft_real *)vfft_aligned_alloc(2 * PN * sizeof(vfft_real)) : z;
    }
    if (na >= 2 && z && zo)
    {
        size_t q;
        int reps = 1;
        for (q = 0; q < 2 * PN; q++)
            z[q] = 1.0 + 1e-6 * (double)(q & 1023);
        memset(&zrs, 0, sizeof zrs);
        if (zo == z && (zseed = (vfft_real *)vfft_aligned_alloc(2 * PN * sizeof(vfft_real))) != NULL)
        {   /* in place: the plane re-laid before every sample */
            memcpy(zseed, z, 2 * PN * sizeof(vfft_real));
            zrs.p = z;
            zrs.seed = zseed;
            zrs.n = 2 * PN;
        }
        for (k = 0; k < na; k++)
        {
            int r2;
            ca[k].h = h; ca[k].z = z; ca[k].zo = zo; ca[k].slot = slot; ca[k].wslot = wslot;
            arms[k].name = ca[k].name;
            arms[k].run = _il2d_arm_child;
            arms[k].ctx = &ca[k];
            if (zseed) _il2d_ip_reset(&zrs);
            r2 = _il2d_race_reps_ns(_il2d_arm_child, &ca[k], zseed ? (double)PN : 0.0, VFFT_IL2D_MT_SAMPLE_NS);
            if (r2 > reps) reps = r2;   /* ~1 ms of the FASTEST arm */
        }
        if (zo != z || zseed)
        {
            const vfft_race_proto_t proto = { 7, reps, VFFT_RACE_MIN, 1, 2, zseed ? _il2d_ip_reset : NULL,
                                              zseed ? &zrs : NULL, T > 1 ? 0 : 1, 1 };
            vfft_race_run(&proto, arms, na, ns);
            for (k = 1; k < na; k++)
                if (ns[k] < ns[win]) win = k;
            if (getenv("VFFT_IL2D_LOG"))
            {
                fprintf(stderr, "[il2d] %s final %dx%d T=%d: %d candidate(s) of the shortlist at N=%d, %.2f s:", what, N1, N2, T,
                        na, N, (vfft_now_ns() - t0) / 1e9);
                for (k = 0; k < na; k++)
                    fprintf(stderr, " %s=%.0f%s", ca[k].name, ns[k], k == win ? "*" : "");
                fprintf(stderr, "\n");
            }
        }
    }
    /* the winner onto the plan, its recipe the child's store; the others go */
    for (k = 0; k < na; k++)
    {
        if (k == win)
            continue;
        vfft_destroy(ca[k].p);
        _il2d_clone_set_drop(&ca[k].w, &ca[k].wn);
        vfft_child_store_free(ca[k].S);
    }
    *slot = ca[win].p;
    *wslot = ca[win].w;
    *wn = ca[win].wn;
    *Sslot = ca[win].S;
    for (k = 0; k < ca[win].S->vw2.nrec; k++)
        ca[win].S->vw2.rec[k].own = 1;   /* this cell's recipe now: onto its row (vfft_child_row_update) */
    if (zo && zo != z) vfft_aligned_free(zo);
    vfft_aligned_free(z);
    vfft_aligned_free(zseed);
}

/* ── the c2c MT verdict race (same law as the real tier's): serial vs
 * threaded FULL walk through the very code execute serves with, min-of-3
 * alternated on a scratch plane, banked as cmt= + cmtt= (the T raced
 * at) in the cell's chain row. The "no" is banked exactly like the
 * "yes". If MT cannot engage at all (no clones, too few units), that IS
 * the verdict: cmt=0. */
static int _il2d_c2c_mt(struct vfft_plan_s *h, const vfft_real *sre,
                        vfft_real *dre, vfft_dir_t dir, int T);
static void _il2d_c2c_mt_race(struct vfft_plan_s *h,
                              struct vfft_wisdom_s *W,
                              const vfft_cfg_t *cfg, int N1, int N2)
{
    const size_t PN = (size_t)N1 * N2;
    /* aligned like every plane the door serves, and the cell's own placement
     * (an out-of-place cell races x -> y), as in the axis race */
    vfft_real *z = (vfft_real *)vfft_aligned_alloc(2 * PN * sizeof(vfft_real));
    vfft_real *zo = (h->placement == VFFT_OUTOFPLACE) ? (vfft_real *)vfft_aligned_alloc(2 * PN * sizeof(vfft_real)) : z;
    vfft_real *zseed = NULL;
    double st = 1e300, mt = 1e300;
    int p;
    size_t i;
    if (!z || !zo)
    {
        vfft_aligned_free(z);
        if (zo != z) vfft_aligned_free(zo);
        return;
    }
    for (i = 0; i < 2 * PN; i++)
        z[i] = 1.0 + 1e-6 * (double)(i & 511);
    zseed = (zo == z) ? (vfft_real *)vfft_aligned_alloc(2 * PN * sizeof(vfft_real)) : NULL;   /* an in-place cell's arms walk their output */
    if (zseed)
        memcpy(zseed, z, 2 * PN * sizeof(vfft_real));
    if (!_il2d_c2c_mt(h, z, zo, VFFT_FORWARD, h->nthreads))
    {
        h->il2d_col.colmt = 0; /* cannot engage — that IS the verdict */
        vfft_aligned_free(z);
        if (zo != z) vfft_aligned_free(zo);
        vfft_aligned_free(zseed);
        vw2_2d_il_chain_bank(&W->vw2, N1, N2, h->il2d_col.R, h->il2d_col.nst,
                             h->il2d_col.wl, h->il2d_col.tfuse, _il2d_ro_of(h),
                             0, h->nthreads,
                             (N1 & (N1 - 1)) ? h->il2d_col.blu : -1, 0.0, vfft_policy_ord_rankn(cfg), h->nthreads);
        _vw2_persist(W, cfg);
        return;
    }
    {
        _il2d_race_ctx_t rc0 = { h, NULL, z, 0, 1, 0, 0, 0, NULL, NULL, NULL, NULL };
        _il2d_race_ctx_t rc[VFFT_RACE_MAX_ARMS];
        vfft_race_arm_t arms[VFFT_RACE_MAX_ARMS];
        char names[VFFT_RACE_MAX_ARMS][20];
        double ns[VFFT_RACE_MAX_ARMS];
        /* the pruning laws (policy_il.h): no serial arm on a composite or odd column above the L1d (MT engaged above:
         * the plane threads), no natural block arm where the blocks law prunes it */
        const int serial_ok = vfft_policy_il2d_serial_races(N1, N2, h->nthreads);
        const int blocks_ok = !h->il2d_col.nat || vfft_policy_il2d_blocks_race(N1, N2, h->nthreads);
        const int a0 = serial_ok ? 1 : 0;   /* the first threaded arm */
        int lad[6], nl, a, na = 0, best = a0, k, nv;
        vfft_race_proto_t proto = { 3, 1, VFFT_RACE_MIN, 0, 2, NULL, NULL, 0 }; /* min-of-3, A then B, two untimed passes per arm first (a cold threaded arm reads 58 us for a 37-us plan) */ /* THREADED arms: never paused (VFFT_RACE_PACE_MS) */
        _il2d_ip_reset_t mrs = { z, zseed, 2 * PN };
        if (zseed)
        {   /* in place: the plane re-laid before every sample */
            proto.reset = _il2d_ip_reset;
            proto.reset_ctx = &mrs;
        }
        (void)p;
        for (a = 0; a < VFFT_RACE_MAX_ARMS; a++) { rc[a] = rc0; rc[a].sw = 0; rc[a].lst = 1; rc[a].zo = zo; ns[a] = 1e300; }
        if (serial_ok)
        {
            arms[na].name = "serial"; arms[na].run = _il2d_arm_exec_st; arms[na].ctx = &rc[na]; na++;
        }
        if (h->il2d_turn || h->il2d_csk)
        {   /* the turn and the skewed pass have ONE threaded walk each */
            arms[na].name = "threaded"; arms[na].run = _il2d_arm_exec_mt; arms[na].ctx = &rc[na]; na++;
            nl = 0;
        }
        else
            nl = _il2d_sw_ladder(N1, N2, h->nthreads, lad, 6);
        /* a natural cell races every threaded arm STAGED (nst = 1) and
         * STRIDED (nst = 0): the leaf's form is a raced plan parameter at
         * T > 1 (il2d_natural_leaf_design.md); a scrambled
         * cell has no leaf form */
        for (nv = 0; nv < ((h->il2d_turn || h->il2d_csk) ? 0 : (h->il2d_col.nat && h->il2d_col.natstage) ? 2 : 1); nv++)
        {
            const int nst = nv ? 0 : 1;
            const char *tag = (h->il2d_col.nat && !nst) ? "/str" : "";
            if (blocks_ok)
            {
                snprintf(names[na], sizeof names[na], "%s%s", h->il2d_col.nat ? "block" : "bands", tag);
                rc[na].lst = nst; arms[na].name = names[na]; arms[na].run = _il2d_arm_exec_mt; arms[na].ctx = &rc[na]; na++;
            }
            if (h->il2d_col.nat)
            {
                snprintf(names[na], sizeof names[na], "strips%s", tag);
                rc[na].lst = nst; arms[na].name = names[na]; arms[na].run = _il2d_arm_exec_mt_strip; arms[na].ctx = &rc[na]; na++;
            }
            if (h->il2d_col.nat && h->il2d_col.R[0] >= 2 && na < VFFT_RACE_MAX_ARMS)
            {   /* the TILE partition */
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
        /* the samples: ~1 ms of a THREADED arm (the serial one runs T times longer), one untimed execute before
         * each (_il2d_race_reps_ns: a single-execute sample times the partition's transient) */
        proto.reps = _il2d_race_reps_ns(arms[na > a0 ? a0 : 0].run, arms[na > a0 ? a0 : 0].ctx, zseed ? (double)PN : 0.0,
                                        VFFT_IL2D_MT_SAMPLE_NS);
        proto.warm_each = 1;
        vfft_race_run(&proto, arms, na, ns);
        st = serial_ok ? ns[0] : 1e300;   /* no serial arm: the threaded champion is the verdict */
        for (a = a0 + 1; a < na; a++) if (ns[a] < ns[best]) best = a;
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
    vfft_aligned_free(z);
    if (zo != z) vfft_aligned_free(zo);
    vfft_aligned_free(zseed);
    if (getenv("VFFT_IL2D_LOG"))
        fprintf(stderr, "[il2d-c2c] colmt race %dx%d T=%d: st=%.0f "
                        "mt=%.0f -> %s%s\n",
                N1, N2, h->nthreads, st, mt,
                h->il2d_col.colmt ? "THREADED" : "serial",
                (h->il2d_col.colmt && h->il2d_col.natarm) ? " (strips)" : "");
    {   /* the verdict lands on the cell's EXISTING row through the field-update
         * path (ns = 0): a measured bank rebuilds the row and would drop the
         * axis race's tokens (sw, rbk, turn, csk). The measured time is mtns=. */
        const int ord = vfft_policy_ord_rankn(cfg);
        const int have_row = vw2_2d_il_tok_geti(&W->vw2, N1, N2, ord, h->nthreads, "chain", -1) >= 0;   /* chain= is on every row */
        vw2_2d_il_chain_bank(&W->vw2, N1, N2, h->il2d_col.R, h->il2d_col.nst,
                             h->il2d_col.wl, h->il2d_col.tfuse, _il2d_ro_of(h),
                             h->il2d_col.colmt, h->nthreads,
                             (N1 & (N1 - 1)) ? h->il2d_col.blu : -1,
                             have_row ? 0.0 : (h->il2d_col.colmt ? mt : st), ord, h->nthreads);
        /* the threaded arm's shape beside cmt/cmtt, read back at that T only */
        vw2_2d_il_tok_seti(&W->vw2, N1, N2, ord, h->nthreads, "mtarm", h->il2d_col.natarm);
        vw2_2d_il_tok_seti(&W->vw2, N1, N2, ord, h->nthreads, "msw", h->il2d_col.msw);
        if (h->il2d_col.nat) vw2_2d_il_tok_seti(&W->vw2, N1, N2, ord, h->nthreads, "nls", h->il2d_col.natst);
        vw2_2d_il_tok_seti(&W->vw2, N1, N2, ord, h->nthreads, "mtns", (int)(h->il2d_col.colmt ? mt : st));
    }
    _vw2_persist(W, cfg);
}

#endif /* VFFT_TRANSFORMS_FFT2D_IL2D_TIER_H */
