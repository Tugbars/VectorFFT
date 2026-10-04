/* zttr.h — ZTURN-T REAL (2026-09-29, docs/roadmap/il_real_engine_research.md
 * method 2): the r2c of an even N on the ZTURN-T machinery at M = N/2 with
 * the Hermitian fold FUSED INTO THE TERMINATOR. No fold pass, no scratch
 * plane: the pipeline runs in the caller's CCE plane (the ZTT's `dest`
 * mode), the split interior does its arithmetic without a shuffle, and the
 * untangle — pure add/mul on split lanes — rides in the last stage.
 *
 * THE PACKING. x[N] read as z[M] = x[2n] + i x[2n+1]; Z = DFT_M(z); the real
 * spectrum is X[f] = E[f] + w_N^f O[f] with E, O from Z[f] and Z[M-f]:
 *   t1 = Zr[f] - Zr[m]   t2 = Zi[f] + Zi[m]        (m = M - f)
 *   xr = S~ t1 + C~ t2   xi = S~ t2 - C~ t1         S~ = 1/2 - 1/2 sin(2 pi f/N)
 *   X[f] = (Zr[m] + xr, xi - Zi[m])                C~ = 1/2 cos(2 pi f/N)
 *   X[m] = (Zr[f] - xr, xi - Zi[f])
 * (zr2c.h's fold, whose f = 0 pair is Z[0] with itself: X[0], X[M] fall out
 * of the same lines, imaginary parts exactly zero.)
 *
 * THE TERMINATOR (tlfh). The ZTT last stage combines R runs of length L
 * (M = R L): column b's radix-R butterfly over the R legs, pre-twiddled by
 * w_M^(r b), yields Z[b + r L]; tlf stores that quad-by-quad (REINT) as the
 * natural output. The mirror of f = r L + b is (R-1-r) L + (L - b), so an
 * ALIGNED column quad (k..k+3) of every leg pairs with the WINDOW (L-k-3 ..
 * L-k) of the partner leg, offset by one column from the plane's 64-B
 * blocks. Per iteration k = 0, 4, .., L/2:
 *   primary : the quad's R legs loaded with their lanes REVERSED (so lane i
 *             is column k+3-i; the twiddle records reversed at create), the
 *             combine;
 *   partner : the window (L-k-3 .. L-k) of every leg, assembled from the
 *             two blocks it straddles (one permute2f128 + one shuffle_pd
 *             per plane), the combine with its own record set;
 *   untangle: leg r of the primary (columns k+3-i) with leg R-1-r of the
 *             partner (columns L-k-3+i): the pairs sum to L. Split lanes,
 *             no shuffle. S~, C~ from a table laid out per (leg, quad) with
 *             the primary's reversed lanes;
 *   stores  : X[f] through the REINT edge with the reversal folded into its
 *             pre-permute (0x27 for 0xD8), X[m] through the plain REINT at
 *             the window's (unaligned) address.
 * Column 0 (k = 0, lane 3) pairs with column 0 of leg R-r, not with the
 * window's "column L" (the next run's column 0 — for leg R-1 the plane's
 * end): the k = 0 iteration blends lane 3 of every partner from the primary
 * combine of leg (R-r) mod R (the DC pair Z[0] with itself comes out as
 * X[0] and X[M]). Column L/2 pairs with itself in leg R-1-r: the last
 * iteration (k = L/2, lane 3) — its other three lanes redo pairs already
 * stored, and IN PLACE they would read X where Z stood, so the last
 * iteration stores lane 3 alone. In place (the pipeline in X): every other
 * iteration reads exactly the columns it overwrites plus block lanes it
 * does not use. L >= 8 (L/2 a multiple of 4): M >= 32 at R = 4, 64 at R = 8.
 *
 * FORWARD ONLY for now (the c2r twin, the fold fused into the backward
 * ingest, is the next arm). The chain and the tile are PLAN INPUT (the c2c
 * cell's ZTT verdict at M, or the race). */
#ifndef VFFT_ZTTR_H
#define VFFT_ZTTR_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#include "ztt.h"
#include "common/math/pi.h"

typedef struct {
    int N, M, R;              /* N real, M = N/2, R = the last radix */
    long L;                   /* the terminator's run length, M / R */
    vfft_ztt_plan_t *zt;      /* the ZTT plan at M: ingest, mids, streams, rb */
    double *twr;              /* the terminator's records, lanes reversed per quad */
    double *twp;              /* the partner window's records: quad q = columns
                               * (L-4q-3 .. L-4q), (R-1) records of 8 doubles */
    double *affS, *affC;      /* S~, C~ per (leg, quad) with reversed lanes, M each */
    double *twr3, *twp3;      /* tlfhc: the primary's records in NATURAL lane order, the
                               * partner's for columns (L-k .. L-k-3) in lanes 0..3 */
    double *affS3, *affC3;    /* tlfhc: S~(f), C~(f) indexed by f, M each */
    double *bS, *bC;          /* the c2r ingest's tables, lane-expanded per column pair:
                               * bS[2n], bS[2n+1] = sin(2 pi n/N); bC = -cos; 2M doubles each */
    int blocked;              /* the terminator form: 2 = tlfhc (natural lanes), 1 = tlfhb, 0 = monolithic */
    int stk;                  /* the kernels' stack state 0..3: rsp = 64k - 32 - 16 stk at their call */
    int mt, mt_t;             /* the threaded arm (zttr_mt.h): 0 serial, 1 BLOCKS, 2 TILES; bound for mt_t threads */
    double *scratch;          /* N+2 doubles, IN-PLACE placements only (vfft_zttr_set_inplace): the
                               * ingest and the mids run here, the last stage lands in the caller's buffer */
} vfft_zttr_plan_t;

static inline void vfft_zttr_destroy(vfft_zttr_plan_t *p)
{
    if (!p) return;
    vfft_ztt_destroy(p->zt);
    vfft_aligned_free(p->twr);
    vfft_aligned_free(p->twp);
    vfft_aligned_free(p->affS);
    vfft_aligned_free(p->twr3);
    vfft_aligned_free(p->twp3);
    vfft_aligned_free(p->affS3);
    vfft_aligned_free(p->bS);
    vfft_aligned_free(p->scratch);
    free(p);
}

/* an in-place plan: the pipeline needs a plane beside the caller's buffer */
static inline int vfft_zttr_set_inplace(vfft_zttr_plan_t *p)
{
    if (p->scratch) return 1;
    p->scratch = (double *)vfft_aligned_alloc((size_t)(p->N + 2) * sizeof(double));
    return p->scratch != NULL;
}

/* every ZTT-r chain of m: the ends in {4, 8} (the fused ingest and the fused
 * terminator are lane lattices), the mids in {4, 8, 3, 5, 7, 9, 15} (the
 * ZTT's odd band), product m, 2..7 stages, (m / r0) % 4 == 0 -- the
 * candidates a per-cell calibration sweeps; the ZTT's create is the law on
 * legality (a run not in whole blocks, a pow2 chain without a driver) and
 * vfft_zttr_create on the terminator's run length */
#define VFFT_ZTTR_MAX_CHAINS 256
static inline void _zttr_chains_rec(int rem, int pos, int ch[8], int out[][8], int *nfs, int cap, int *n)
{
    static const int mids[7] = { 4, 8, 3, 5, 7, 9, 15 };
    if (*n >= cap) return;
    if (rem == 4 || rem == 8)
    {   /* the remaining product is the last radix */
        ch[pos] = rem;
        memcpy(out[*n], ch, sizeof(int) * (size_t)(pos + 1));
        nfs[*n] = pos + 1;
        (*n)++;
    }
    if (pos >= 6) return;
    for (int i = 0; i < 7 && *n < cap; i++)
        if (rem % mids[i] == 0 && rem / mids[i] >= 4)
        {
            ch[pos] = mids[i];
            _zttr_chains_rec(rem / mids[i], pos + 1, ch, out, nfs, cap, n);
        }
}
static inline int vfft_zttr_chains(int m, int out[][8], int *nfs, int cap)
{
    int n = 0, ch[8];
    for (int r0 = 4; r0 <= 8 && n < cap; r0 += 4)
    {
        if (m % r0 || (m / r0) % 4) continue;
        ch[0] = r0;
        _zttr_chains_rec(m / r0, 1, ch, out, nfs, cap, &n);
    }
    return n;
}

/* one terminator record set for the columns b[0..3], legs 1..R-1, fwd:
 * [c x4][-sin x4] of w_M^(r b) — the layout _ztt_fill_stage writes */
static inline void _zttr_records(double *rec, int R, long M, const long b[4])
{
    for (int r = 1; r < R; r++)
        for (int lane = 0; lane < 4; lane++)
        {
            long pw = ((long)r * b[lane]) % M;
            double c, sn;
            if (2 * pw > M) pw -= M;
            vfft_cs2pi_exact((long long)pw, (long long)M, &c, &sn);
            rec[(r - 1) * 8 + lane] = c;
            rec[(r - 1) * 8 + 4 + lane] = -sn;
        }
}

static inline vfft_zttr_plan_t *vfft_zttr_create(int N, const int *chain, int nf, size_t tile)
{
    if (N < 64 || (N & 1)) return NULL;
    const int M = N / 2;
    vfft_ztt_plan_t *zt = vfft_ztt_create_chain(M, chain, nf);
    if (!zt) return NULL;
    if (tile && !vfft_ztt_set_tile(zt, tile)) { vfft_ztt_destroy(zt); return NULL; }
    const int R = chain[nf - 1];
    const long L = zt->L[nf - 1];
    if ((R != 4 && R != 8) || L < 8 || (L % 8)) { vfft_ztt_destroy(zt); return NULL; }
    vfft_zttr_plan_t *p = (vfft_zttr_plan_t *)calloc(1, sizeof *p);
    if (!p) { vfft_ztt_destroy(zt); return NULL; }
    p->N = N; p->M = M; p->R = R; p->L = L; p->zt = zt;
    p->blocked = 2;                                  /* tlfhc, at stack state 3 */
    p->stk = 3;
    const long nq = L / 8 + 1;                       /* quads k = 0, 4, .., L/2 */
    const size_t recs = (size_t)nq * (size_t)(R - 1) * 8u;
    p->twr = (double *)vfft_aligned_alloc(recs * sizeof(double));
    p->twp = (double *)vfft_aligned_alloc(recs * sizeof(double));
    p->affS = (double *)vfft_aligned_alloc(2u * (size_t)M * sizeof(double));
    p->twr3 = (double *)vfft_aligned_alloc(recs * sizeof(double));
    p->twp3 = (double *)vfft_aligned_alloc(recs * sizeof(double));
    p->affS3 = (double *)vfft_aligned_alloc(2u * (size_t)M * sizeof(double));
    if (!p->twr || !p->twp || !p->affS || !p->twr3 || !p->twp3 || !p->affS3) { vfft_zttr_destroy(p); return NULL; }
    p->affC = p->affS + M;
    p->affC3 = p->affS3 + M;
    p->bS = (double *)vfft_aligned_alloc(4u * (size_t)M * sizeof(double));
    if (!p->bS) { vfft_zttr_destroy(p); return NULL; }
    p->bC = p->bS + 2 * (size_t)M;
    for (long n = 0; n < M; n++)
    {
        const double th = 2.0 * VFFT_PI * (double)n / (double)N;
        p->bS[2 * n] = p->bS[2 * n + 1] = sin(th);
        p->bC[2 * n] = p->bC[2 * n + 1] = -cos(th);
    }
    for (long q = 0; q < nq; q++)
    {
        const long k = 4 * q;
        long b[4];
        for (int i = 0; i < 4; i++) b[i] = k + 3 - i;          /* the primary, reversed */
        _zttr_records(p->twr + (size_t)q * (size_t)(R - 1) * 8u, R, M, b);
        for (int i = 0; i < 4; i++) b[i] = (L - k - 3 + i) % M; /* the window; column L -> 0 */
        _zttr_records(p->twp + (size_t)q * (size_t)(R - 1) * 8u, R, M, b);
        for (int i = 0; i < 4; i++) b[i] = k + i;                /* tlfhc: natural */
        _zttr_records(p->twr3 + (size_t)q * (size_t)(R - 1) * 8u, R, M, b);
        for (int i = 0; i < 4; i++) b[i] = L - k - i;            /* tlfhc: the window reversed; column L is a run's column 0 */
        _zttr_records(p->twp3 + (size_t)q * (size_t)(R - 1) * 8u, R, M, b);
    }
    for (long f = 0; f < M; f++)
    {
        const double th = 2.0 * VFFT_PI * (double)f / (double)N;
        p->affS3[f] = 0.5 - 0.5 * sin(th);
        p->affC3[f] = 0.5 * cos(th);
    }
    /* S~(f), C~(f) at f = rL + k + 3 - i for lane i: the table holds each
     * quad's four values in that order */
    for (long f0 = 0; f0 < M; f0 += 4)
        for (int i = 0; i < 4; i++)
        {
            const long f = f0 + 3 - i;
            const double th = 2.0 * VFFT_PI * (double)f / (double)N;
            p->affS[f0 + i] = 0.5 - 0.5 * sin(th);
            p->affC[f0 + i] = 0.5 * cos(th);
        }
    return p;
}

/* ── the split radix-R combines, transcribed from the emitted tlf bodies
 * (codelets/zil/avx2/ztt/radixR_z_tlf_avx2.c): pre-twiddle legs 1..R-1
 * from the record set `tw` ([c x4][s x4] per leg, 8 doubles), then the
 * DIT butterfly; outputs leg-major in (or, oi). ── */
#if defined(__AVX2__)
static inline __attribute__((always_inline)) void _zttr_comb4(
    const __m256d *lr, const __m256d *li, const double *tw, __m256d *orr, __m256d *oi)
{
    const __m256d t1 = _mm256_loadu_pd(tw + 16), t3 = _mm256_loadu_pd(tw + 20);
    const __m256d t4 = _mm256_fnmadd_pd(li[3], t3, _mm256_mul_pd(lr[3], t1));
    const __m256d t5 = _mm256_fmadd_pd(lr[3], t3, _mm256_mul_pd(li[3], t1));
    const __m256d t10 = _mm256_loadu_pd(tw + 0), t12 = _mm256_loadu_pd(tw + 4);
    const __m256d t13 = _mm256_fnmadd_pd(li[1], t12, _mm256_mul_pd(lr[1], t10));
    const __m256d t14 = _mm256_fmadd_pd(lr[1], t12, _mm256_mul_pd(li[1], t10));
    const __m256d t16 = _mm256_sub_pd(t14, t5);
    const __m256d t18 = _mm256_sub_pd(t13, t4);
    const __m256d t35 = _mm256_add_pd(t5, t14);
    const __m256d t36 = _mm256_add_pd(t4, t13);
    const __m256d t22 = _mm256_loadu_pd(tw + 8), t24 = _mm256_loadu_pd(tw + 12);
    const __m256d t25 = _mm256_fnmadd_pd(li[2], t24, _mm256_mul_pd(lr[2], t22));
    const __m256d t26 = _mm256_fmadd_pd(lr[2], t24, _mm256_mul_pd(li[2], t22));
    const __m256d t32 = _mm256_sub_pd(lr[0], t25);
    const __m256d t39 = _mm256_add_pd(t25, lr[0]);
    const __m256d t30 = _mm256_sub_pd(li[0], t26);
    const __m256d t38 = _mm256_add_pd(t26, li[0]);
    orr[0] = _mm256_add_pd(t36, t39); oi[0] = _mm256_add_pd(t35, t38);
    orr[1] = _mm256_add_pd(t16, t32); oi[1] = _mm256_sub_pd(t30, t18);
    orr[2] = _mm256_sub_pd(t39, t36); oi[2] = _mm256_sub_pd(t38, t35);
    orr[3] = _mm256_sub_pd(t32, t16); oi[3] = _mm256_add_pd(t18, t30);
}

static inline __attribute__((always_inline)) void _zttr_comb8(
    const __m256d *lr, const __m256d *li, const double *tw, __m256d *orr, __m256d *oi)
{
    const __m256d t40 = _mm256_set1_pd(0.70710678118654757);
    const __m256d t1 = _mm256_loadu_pd(tw + 48), t3 = _mm256_loadu_pd(tw + 52);
    const __m256d t4 = _mm256_fnmadd_pd(li[7], t3, _mm256_mul_pd(lr[7], t1));
    const __m256d t5 = _mm256_fmadd_pd(lr[7], t3, _mm256_mul_pd(li[7], t1));
    const __m256d t10 = _mm256_loadu_pd(tw + 16), t12 = _mm256_loadu_pd(tw + 20);
    const __m256d t13 = _mm256_fnmadd_pd(li[3], t12, _mm256_mul_pd(lr[3], t10));
    const __m256d t14 = _mm256_fmadd_pd(lr[3], t12, _mm256_mul_pd(li[3], t10));
    const __m256d t16 = _mm256_sub_pd(t14, t5);
    const __m256d t18 = _mm256_sub_pd(t13, t4);
    const __m256d t81 = _mm256_add_pd(t5, t14);
    const __m256d t82 = _mm256_add_pd(t4, t13);
    const __m256d t22 = _mm256_loadu_pd(tw + 32), t24 = _mm256_loadu_pd(tw + 36);
    const __m256d t25 = _mm256_fnmadd_pd(li[5], t24, _mm256_mul_pd(lr[5], t22));
    const __m256d t26 = _mm256_fmadd_pd(lr[5], t24, _mm256_mul_pd(li[5], t22));
    const __m256d t29 = _mm256_loadu_pd(tw + 0), t31 = _mm256_loadu_pd(tw + 4);
    const __m256d t32 = _mm256_fnmadd_pd(li[1], t31, _mm256_mul_pd(lr[1], t29));
    const __m256d t33 = _mm256_fmadd_pd(lr[1], t31, _mm256_mul_pd(li[1], t29));
    const __m256d t34 = _mm256_sub_pd(t33, t26);
    const __m256d t36 = _mm256_sub_pd(t32, t25);
    const __m256d t84 = _mm256_add_pd(t26, t33);
    const __m256d t85 = _mm256_add_pd(t25, t32);
    const __m256d t37 = _mm256_add_pd(t18, t34);
    const __m256d t38 = _mm256_sub_pd(t36, t16);
    const __m256d t101 = _mm256_sub_pd(t34, t18);
    const __m256d t103 = _mm256_add_pd(t16, t36);
    const __m256d t86 = _mm256_sub_pd(t84, t81);
    const __m256d t88 = _mm256_sub_pd(t85, t82);
    const __m256d t115 = _mm256_add_pd(t81, t84);
    const __m256d t116 = _mm256_add_pd(t82, t85);
    const __m256d t39 = _mm256_sub_pd(t38, t37);
    const __m256d t44 = _mm256_add_pd(t37, t38);
    const __m256d t104 = _mm256_add_pd(t101, t103);
    const __m256d t107 = _mm256_sub_pd(t101, t103);
    const __m256d t48 = _mm256_loadu_pd(tw + 40), t50 = _mm256_loadu_pd(tw + 44);
    const __m256d t51 = _mm256_fnmadd_pd(li[6], t50, _mm256_mul_pd(lr[6], t48));
    const __m256d t52 = _mm256_fmadd_pd(lr[6], t50, _mm256_mul_pd(li[6], t48));
    const __m256d t55 = _mm256_loadu_pd(tw + 8), t57 = _mm256_loadu_pd(tw + 12);
    const __m256d t58 = _mm256_fnmadd_pd(li[2], t57, _mm256_mul_pd(lr[2], t55));
    const __m256d t59 = _mm256_fmadd_pd(lr[2], t57, _mm256_mul_pd(li[2], t55));
    const __m256d t60 = _mm256_sub_pd(t59, t52);
    const __m256d t62 = _mm256_sub_pd(t58, t51);
    const __m256d t91 = _mm256_add_pd(t52, t59);
    const __m256d t92 = _mm256_add_pd(t51, t58);
    const __m256d t66 = _mm256_loadu_pd(tw + 24), t68 = _mm256_loadu_pd(tw + 28);
    const __m256d t69 = _mm256_fnmadd_pd(li[4], t68, _mm256_mul_pd(lr[4], t66));
    const __m256d t70 = _mm256_fmadd_pd(lr[4], t68, _mm256_mul_pd(li[4], t66));
    const __m256d t76 = _mm256_sub_pd(lr[0], t69);
    const __m256d t95 = _mm256_add_pd(t69, lr[0]);
    const __m256d t78 = _mm256_sub_pd(t76, t60);
    const __m256d t131 = _mm256_fmadd_pd(t39, t40, t78);
    const __m256d t135 = _mm256_fnmadd_pd(t39, t40, t78);
    const __m256d t98 = _mm256_sub_pd(t95, t92);
    const __m256d t99 = _mm256_sub_pd(t98, t86);
    const __m256d t125 = _mm256_add_pd(t86, t98);
    const __m256d t111 = _mm256_add_pd(t60, t76);
    const __m256d t133 = _mm256_fnmadd_pd(t40, t104, t111);
    const __m256d t137 = _mm256_fmadd_pd(t40, t104, t111);
    const __m256d t119 = _mm256_add_pd(t92, t95);
    const __m256d t120 = _mm256_sub_pd(t119, t116);
    const __m256d t129 = _mm256_add_pd(t116, t119);
    const __m256d t74 = _mm256_sub_pd(li[0], t70);
    const __m256d t94 = _mm256_add_pd(t70, li[0]);
    const __m256d t77 = _mm256_add_pd(t62, t74);
    const __m256d t132 = _mm256_fmadd_pd(t40, t44, t77);
    const __m256d t136 = _mm256_fnmadd_pd(t40, t44, t77);
    const __m256d t96 = _mm256_sub_pd(t94, t91);
    const __m256d t100 = _mm256_add_pd(t88, t96);
    const __m256d t126 = _mm256_sub_pd(t96, t88);
    const __m256d t110 = _mm256_sub_pd(t74, t62);
    const __m256d t134 = _mm256_fnmadd_pd(t40, t107, t110);
    const __m256d t138 = _mm256_fmadd_pd(t40, t107, t110);
    const __m256d t118 = _mm256_add_pd(t91, t94);
    const __m256d t122 = _mm256_sub_pd(t118, t115);
    const __m256d t130 = _mm256_add_pd(t115, t118);
    orr[0] = t129; oi[0] = t130;
    orr[1] = t137; oi[1] = t138;
    orr[2] = t125; oi[2] = t126;
    orr[3] = t135; oi[3] = t136;
    orr[4] = t120; oi[4] = t122;
    orr[5] = t133; oi[5] = t134;
    orr[6] = t99;  oi[6] = t100;
    orr[7] = t131; oi[7] = t132;
}

/* the Hermitian terminator over the plane W (the last stage's R runs of L
 * complex, 64-B [re x4][im x4] blocks) into the CCE X[0..M]; W may be X
 * itself (the pipeline in the destination). ONE BODY PER RADIX: R is a
 * literal inside the macro so every leg loop unrolls and the vectors stay
 * in registers; the partner combine's outputs are parked on the stack
 * (PW[], L1-hot) and the untangle streams them leg by leg against the
 * primary combine held in registers. */
#define ZTTR_UNROLL _Pragma("GCC unroll 8")
#define ZTTR_TLFH_BODY(RR, COMB)                                                                          \
static inline void _zttr_tlfh##RR(const vfft_zttr_plan_t *p, const double *W, double *X)                 \
{                                                                                                         \
    enum { R = RR };                                                                                      \
    const long L = p->L;                                                                                  \
    const double *affS = p->affS, *affC = p->affC;                                                        \
    double cen[2 * R];                                                                                    \
    double PW[4 * R * 4] __attribute__((aligned(64)));  /* [re x4][im x4] per partner leg */              \
    __m256d cr[R], ci[R];                                                                                 \
    {   /* the centre column first (lane 3 of the quad k = L/2), stored last */                           \
        const double *twr = p->twr + (size_t)(L / 8) * (size_t)(R - 1) * 8u;                              \
        __m256d pr[R], pi[R], Pr[R], Pi[R];                                                               \
        ZTTR_UNROLL for (int r = 0; r < R; r++)                                                           \
        {                                                                                                 \
            const double *a = W + 2 * ((size_t)r * (size_t)L + (size_t)(L / 2));                          \
            pr[r] = _mm256_permute4x64_pd(_mm256_loadu_pd(a), 0x1B);                                      \
            pi[r] = _mm256_permute4x64_pd(_mm256_loadu_pd(a + 4), 0x1B);                                  \
        }                                                                                                 \
        COMB(pr, pi, twr, Pr, Pi);                                                                        \
        ZTTR_UNROLL for (int r = 0; r < R; r++)                                                           \
        {                                                                                                 \
            const int rp = R - 1 - r;                                                                     \
            const size_t f0 = (size_t)r * (size_t)L + (size_t)(L / 2);                                    \
            const double S = affS[f0 + 3], C = affC[f0 + 3];                                              \
            const __m128d ah = _mm256_extractf128_pd(Pr[r], 1), aih = _mm256_extractf128_pd(Pi[r], 1);    \
            const __m128d bh = _mm256_extractf128_pd(Pr[rp], 1), bih = _mm256_extractf128_pd(Pi[rp], 1);  \
            const double Ar = _mm_cvtsd_f64(_mm_unpackhi_pd(ah, ah)), Ai = _mm_cvtsd_f64(_mm_unpackhi_pd(aih, aih)); \
            const double Br = _mm_cvtsd_f64(_mm_unpackhi_pd(bh, bh)), Bi = _mm_cvtsd_f64(_mm_unpackhi_pd(bih, bih)); \
            const double t1 = Ar - Br, t2 = Ai + Bi;                                                      \
            const double xr = S * t1 + C * t2, xi = S * t2 - C * t1;                                      \
            cen[2 * r] = Br + xr;                                                                         \
            cen[2 * r + 1] = xi - Bi;                                                                     \
        }                                                                                                 \
    }                                                                                                     \
    for (long k = 0; k < L / 2; k += 4)                                                                   \
    {                                                                                                     \
        const long q = k / 4;                                                                             \
        const double *twr = p->twr + (size_t)q * (size_t)(R - 1) * 8u;                                    \
        const double *twp = p->twp + (size_t)q * (size_t)(R - 1) * 8u;                                    \
        {   /* the partner window (L-k-3 .. L-k): block A's lanes 1..3 and the                            \
             * carried block's lane 0; combined and PARKED */                                             \
            __m256d wr[R], wi[R], Wr[R], Wi[R];                                                           \
            ZTTR_UNROLL for (int r = 0; r < R; r++)                                                       \
            {                                                                                             \
                const double *a = W + 2 * ((size_t)r * (size_t)L + (size_t)(L - k - 4));                  \
                const __m256d ar = _mm256_loadu_pd(a), ai = _mm256_loadu_pd(a + 4);                       \
                const __m256d br = k ? cr[r] : _mm256_setzero_pd(), bi = k ? ci[r] : _mm256_setzero_pd(); \
                const __m256d ur = _mm256_permute2f128_pd(ar, br, 0x21), ui = _mm256_permute2f128_pd(ai, bi, 0x21); \
                wr[r] = _mm256_shuffle_pd(ar, ur, 0x5);                                                   \
                wi[r] = _mm256_shuffle_pd(ai, ui, 0x5);                                                   \
                cr[r] = ar; ci[r] = ai;                                                                   \
            }                                                                                             \
            COMB(wr, wi, twp, Wr, Wi);                                                                    \
            ZTTR_UNROLL for (int r = 0; r < R; r++)                                                       \
            {                                                                                             \
                _mm256_store_pd(PW + 8 * r, Wr[r]);                                                       \
                _mm256_store_pd(PW + 8 * r + 4, Wi[r]);                                                   \
            }                                                                                             \
        }                                                                                                 \
        {   /* the primary quad, lanes reversed (lane i = column k+3-i), in registers */                  \
            __m256d pr[R], pi[R], Pr[R], Pi[R];                                                           \
            ZTTR_UNROLL for (int r = 0; r < R; r++)                                                       \
            {                                                                                             \
                const double *a = W + 2 * ((size_t)r * (size_t)L + (size_t)k);                            \
                pr[r] = _mm256_permute4x64_pd(_mm256_loadu_pd(a), 0x1B);                                  \
                pi[r] = _mm256_permute4x64_pd(_mm256_loadu_pd(a + 4), 0x1B);                              \
            }                                                                                             \
            COMB(pr, pi, twr, Pr, Pi);                                                                    \
            if (k == 0)                                                                                   \
                /* column 0's partner is column 0 of leg R-r: lane 3 of the primary combine of that leg */\
                ZTTR_UNROLL for (int r = 0; r < R; r++)                                                   \
                {                                                                                         \
                    const int src = (R - r) % R, rp = R - 1 - r;                                          \
                    _mm256_store_pd(PW + 8 * rp, _mm256_blend_pd(_mm256_load_pd(PW + 8 * rp), Pr[src], 0x8));       \
                    _mm256_store_pd(PW + 8 * rp + 4, _mm256_blend_pd(_mm256_load_pd(PW + 8 * rp + 4), Pi[src], 0x8)); \
                }                                                                                         \
            ZTTR_UNROLL for (int r = 0; r < R; r++)                                                       \
            {                                                                                             \
                const int rp = R - 1 - r;                                                                 \
                const size_t f0 = (size_t)r * (size_t)L + (size_t)k;                                      \
                const __m256d S = _mm256_loadu_pd(affS + f0), C = _mm256_loadu_pd(affC + f0);             \
                const __m256d Ar = Pr[r], Ai = Pi[r];                                                     \
                const __m256d Br = _mm256_load_pd(PW + 8 * rp), Bi = _mm256_load_pd(PW + 8 * rp + 4);     \
                const __m256d t1 = _mm256_sub_pd(Ar, Br), t2 = _mm256_add_pd(Ai, Bi);                     \
                const __m256d xr = _mm256_fmadd_pd(S, t1, _mm256_mul_pd(C, t2));                          \
                const __m256d xi = _mm256_fmsub_pd(S, t2, _mm256_mul_pd(C, t1));                          \
                const __m256d Xfr = _mm256_add_pd(Br, xr), Xfi = _mm256_sub_pd(xi, Bi);                   \
                const __m256d Xmr = _mm256_sub_pd(Ar, xr), Xmi = _mm256_sub_pd(xi, Ai);                   \
                double *of = X + 2 * f0;                                                                  \
                double *om = X + 2 * ((size_t)rp * (size_t)L + (size_t)(L - k - 3));                      \
                const __m256d fr = _mm256_permute4x64_pd(Xfr, 0x27), fi = _mm256_permute4x64_pd(Xfi, 0x27); \
                _mm256_storeu_pd(of, _mm256_unpacklo_pd(fr, fi));                                         \
                _mm256_storeu_pd(of + 4, _mm256_unpackhi_pd(fr, fi));                                     \
                const __m256d mr = _mm256_permute4x64_pd(Xmr, 0xD8), mi = _mm256_permute4x64_pd(Xmi, 0xD8); \
                _mm256_storeu_pd(om, _mm256_unpacklo_pd(mr, mi));                                         \
                _mm256_storeu_pd(om + 4, _mm256_unpackhi_pd(mr, mi));                                     \
            }                                                                                             \
        }                                                                                                 \
    }                                                                                                     \
    ZTTR_UNROLL for (int r = 0; r < R; r++)                                                               \
    {                                                                                                     \
        double *o = X + 2 * ((size_t)r * (size_t)L + (size_t)(L / 2));                                    \
        o[0] = cen[2 * r];                                                                                \
        o[1] = cen[2 * r + 1];                                                                            \
    }                                                                                                     \
}
ZTTR_TLFH_BODY(4, _zttr_comb4)
ZTTR_TLFH_BODY(8, _zttr_comb8)
#undef ZTTR_TLFH_BODY

/* ── the BLOCKED terminator (tlfhb): the emitter's pass discipline ──
 * pass 1: both windows' legs pre-twiddled, the even legs' and the odd legs'
 *         half-size DFTs (DFT2 at R = 4, DFT4 at R = 8), PARKED as E[m], O[m]
 *         (re, im) per window in SP[] / SW[] on the stack (L1-hot);
 * pass 2: for m = 0..R/2-1, the radix-2 combine P_m = E[m] + W_R^m O[m],
 *         P_{m+R/2} = E[m] - W_R^m O[m] of the primary and the SAME of the
 *         partner pair mp = R/2-1-m -- the two partner legs (mp, mp+R/2) are
 *         exactly (R-1-(m+R/2), R-1-m), the mirrors of the primary pair --
 *         then the two untangles and their stores. About twelve vectors live.
 * k = 0: lane 3 of each partner leg rp = R-1-r must be lane 3 of the PRIMARY
 *         leg (R-r) mod R (column 0 pairs across legs); those R lane-3 values
 *         come from a pre-pass over the primary pairs (z0[]), blended in. */
static inline __attribute__((always_inline)) void _zttr_twl(
    const __m256d xr, const __m256d xi, const double *rec, __m256d *yr, __m256d *yi)
{
    const __m256d c = _mm256_loadu_pd(rec), sn = _mm256_loadu_pd(rec + 4);
    *yr = _mm256_fnmadd_pd(xi, sn, _mm256_mul_pd(xr, c));
    *yi = _mm256_fmadd_pd(xr, sn, _mm256_mul_pd(xi, c));
}
/* the split DFT4 of (a0, a1, a2, a3): X0 = (a0+a2)+(a1+a3), X2 = (a0+a2)-(a1+a3),
 * X1 = (a0-a2) + (-i)(a1-a3), X3 = (a0-a2) - (-i)(a1-a3) */
static inline __attribute__((always_inline)) void _zttr_dft4(
    const __m256d *ar, const __m256d *ai, __m256d *xr, __m256d *xi)
{
    const __m256d s02r = _mm256_add_pd(ar[0], ar[2]), s02i = _mm256_add_pd(ai[0], ai[2]);
    const __m256d d02r = _mm256_sub_pd(ar[0], ar[2]), d02i = _mm256_sub_pd(ai[0], ai[2]);
    const __m256d s13r = _mm256_add_pd(ar[1], ar[3]), s13i = _mm256_add_pd(ai[1], ai[3]);
    const __m256d d13r = _mm256_sub_pd(ar[1], ar[3]), d13i = _mm256_sub_pd(ai[1], ai[3]);
    xr[0] = _mm256_add_pd(s02r, s13r); xi[0] = _mm256_add_pd(s02i, s13i);
    xr[2] = _mm256_sub_pd(s02r, s13r); xi[2] = _mm256_sub_pd(s02i, s13i);
    xr[1] = _mm256_add_pd(d02r, d13i); xi[1] = _mm256_sub_pd(d02i, d13r);
    xr[3] = _mm256_sub_pd(d02r, d13i); xi[3] = _mm256_add_pd(d02i, d13r);
}
/* W_R^m * (or + i oi) for R = 4, 8 (m < R/2), split form */
static inline __attribute__((always_inline)) void _zttr_wm(
    const int R, const int m, const __m256d orr, const __m256d oi, __m256d *tr, __m256d *ti)
{
    const __m256d h = _mm256_set1_pd(0.70710678118654757);
    if (m == 0) { *tr = orr; *ti = oi; return; }
    if (R == 4 || m == 2) { *tr = oi; *ti = _mm256_sub_pd(_mm256_setzero_pd(), orr); return; }   /* -i */
    if (m == 1) { *tr = _mm256_mul_pd(_mm256_add_pd(orr, oi), h); *ti = _mm256_mul_pd(_mm256_sub_pd(oi, orr), h); return; }
    /* m == 3: (-1 - i)/sqrt2 */
    *tr = _mm256_mul_pd(_mm256_sub_pd(oi, orr), h);
    *ti = _mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(), _mm256_add_pd(orr, oi)), h);
}
#define ZTTR_TLFHB_BODY(RR)                                                                               \
static __attribute__((noinline)) void _zttr_tlfhb##RR(const vfft_zttr_plan_t *p, const double *W, double *X) \
{                                                                                                         \
    enum { R = RR, H = RR / 2 };                                                                          \
    const long L = p->L;                                                                                  \
    const double *affS = p->affS, *affC = p->affC;                                                        \
    double cen[2 * R];                                                                                    \
    /* the parking (E[m] re,im then O[m] re,im, 8 doubles each, per window) and the carry     \
     * (block A per leg, [re x4][im x4]) in a region aligned BY HAND: mingw gcc never realigns  \
     * a frame, so an aligned(64) local sits on a 16-B slot and half its ymm traffic would     \
     * split cache lines by the caller's rsp */                                                \
    double raw[40 * R + 8];                                                                               \
    double *const SP = (double *)(((uintptr_t)raw + 63u) & ~(uintptr_t)63u);                              \
    double *const SW = SP + 16 * R;                                                                       \
    double *const CR = SW + 16 * R;                                                                       \
    double z0[2 * R];                                                                                     \
    {   /* the centre column: the monolithic combine of quad k = L/2, lane 3 */                           \
        const double *twr = p->twr + (size_t)(L / 8) * (size_t)(R - 1) * 8u;                              \
        __m256d pr[R], pi[R], Pr[R], Pi[R];                                                               \
        ZTTR_UNROLL for (int r = 0; r < R; r++)                                                           \
        {                                                                                                 \
            const double *a = W + 2 * ((size_t)r * (size_t)L + (size_t)(L / 2));                          \
            pr[r] = _mm256_permute4x64_pd(_mm256_loadu_pd(a), 0x1B);                                      \
            pi[r] = _mm256_permute4x64_pd(_mm256_loadu_pd(a + 4), 0x1B);                                  \
        }                                                                                                 \
        if (R == 4) _zttr_comb4(pr, pi, twr, Pr, Pi); else _zttr_comb8(pr, pi, twr, Pr, Pi);              \
        ZTTR_UNROLL for (int r = 0; r < R; r++)                                                           \
        {                                                                                                 \
            const int rp = R - 1 - r;                                                                     \
            const size_t f0 = (size_t)r * (size_t)L + (size_t)(L / 2);                                    \
            const double S = affS[f0 + 3], C = affC[f0 + 3];                                              \
            double A[4], Ai[4], B[4], Bi[4];                                                              \
            _mm256_storeu_pd(A, Pr[r]); _mm256_storeu_pd(Ai, Pi[r]);                                      \
            _mm256_storeu_pd(B, Pr[rp]); _mm256_storeu_pd(Bi, Pi[rp]);                                    \
            const double t1 = A[3] - B[3], t2 = Ai[3] + Bi[3];                                            \
            const double xr = S * t1 + C * t2, xi = S * t2 - C * t1;                                      \
            cen[2 * r] = B[3] + xr;                                                                       \
            cen[2 * r + 1] = xi - Bi[3];                                                                  \
        }                                                                                                 \
    }                                                                                                     \
    ZTTR_UNROLL for (int r = 0; r < R; r++)                                                               \
    {                                                                                                     \
        _mm256_storeu_pd(CR + 8 * r, _mm256_setzero_pd());                                                \
        _mm256_storeu_pd(CR + 8 * r + 4, _mm256_setzero_pd());                                            \
    }                                                                                                     \
    for (long k = 0; k < L / 2; k += 4)                                                                   \
    {                                                                                                     \
        const long q = k / 4;                                                                             \
        const double *twr = p->twr + (size_t)q * (size_t)(R - 1) * 8u;                                    \
        const double *twp = p->twp + (size_t)q * (size_t)(R - 1) * 8u;                                    \
        /* ── pass 1, the partner window: assembled, twiddled; the even legs' half DFT parked, then the odd legs' ── */\
        ZTTR_UNROLL for (int par = 0; par < 2; par++)                                                    \
        {                                                                                                \
            __m256d xr[H], xi[H], Yr[H], Yi[H];                                                          \
            ZTTR_UNROLL for (int j = 0; j < H; j++)                                                      \
            {                                                                                            \
                const int r = 2 * j + par;                                                               \
                const double *a = W + 2 * ((size_t)r * (size_t)L + (size_t)(L - k - 4));                 \
                const __m256d ar = _mm256_loadu_pd(a), ai = _mm256_loadu_pd(a + 4);                      \
                const __m256d br = _mm256_loadu_pd(CR + 8 * r), bi = _mm256_loadu_pd(CR + 8 * r + 4);    \
                const __m256d ur = _mm256_permute2f128_pd(ar, br, 0x21), ui = _mm256_permute2f128_pd(ai, bi, 0x21);\
                __m256d wr = _mm256_shuffle_pd(ar, ur, 0x5), wi = _mm256_shuffle_pd(ai, ui, 0x5);        \
                _mm256_storeu_pd(CR + 8 * r, ar); _mm256_storeu_pd(CR + 8 * r + 4, ai);                  \
                if (r) _zttr_twl(wr, wi, twp + (r - 1) * 8, &wr, &wi);                                   \
                xr[j] = wr; xi[j] = wi;                                                                  \
            }                                                                                            \
            if (R == 4)                                                                                  \
            {                                                                                            \
                Yr[0] = _mm256_add_pd(xr[0], xr[1]); Yi[0] = _mm256_add_pd(xi[0], xi[1]);                \
                Yr[1] = _mm256_sub_pd(xr[0], xr[1]); Yi[1] = _mm256_sub_pd(xi[0], xi[1]);                \
            }                                                                                            \
            else _zttr_dft4(xr, xi, Yr, Yi);                                                             \
            ZTTR_UNROLL for (int m = 0; m < H; m++)                                                      \
            {                                                                                            \
                _mm256_storeu_pd(SW + 16 * m + 8 * par, Yr[m]);                                          \
                _mm256_storeu_pd(SW + 16 * m + 8 * par + 4, Yi[m]);                                      \
            }                                                                                            \
        }                                                                                                \
        /* ── pass 1, the primary quad (lanes reversed): the even legs' half DFT parked, then the odd legs' ── */\
        ZTTR_UNROLL for (int par = 0; par < 2; par++)                                                    \
        {                                                                                                \
            __m256d xr[H], xi[H], Yr[H], Yi[H];                                                          \
            ZTTR_UNROLL for (int j = 0; j < H; j++)                                                      \
            {                                                                                            \
                const int r = 2 * j + par;                                                               \
                const double *a = W + 2 * ((size_t)r * (size_t)L + (size_t)k);                           \
                __m256d wr = _mm256_permute4x64_pd(_mm256_loadu_pd(a), 0x1B);                            \
                __m256d wi = _mm256_permute4x64_pd(_mm256_loadu_pd(a + 4), 0x1B);                        \
                if (r) _zttr_twl(wr, wi, twr + (r - 1) * 8, &wr, &wi);                                   \
                xr[j] = wr; xi[j] = wi;                                                                  \
            }                                                                                            \
            if (R == 4)                                                                                  \
            {                                                                                            \
                Yr[0] = _mm256_add_pd(xr[0], xr[1]); Yi[0] = _mm256_add_pd(xi[0], xi[1]);                \
                Yr[1] = _mm256_sub_pd(xr[0], xr[1]); Yi[1] = _mm256_sub_pd(xi[0], xi[1]);                \
            }                                                                                            \
            else _zttr_dft4(xr, xi, Yr, Yi);                                                             \
            ZTTR_UNROLL for (int m = 0; m < H; m++)                                                      \
            {                                                                                            \
                _mm256_storeu_pd(SP + 16 * m + 8 * par, Yr[m]);                                          \
                _mm256_storeu_pd(SP + 16 * m + 8 * par + 4, Yi[m]);                                      \
            }                                                                                            \
        }                                                                                                \
        /* ── k = 0: the primary legs' lane 3 (column 0), a pre-pass over the pairs ── */                 \
        if (k == 0)                                                                                       \
            ZTTR_UNROLL for (int m = 0; m < H; m++)                                                       \
            {                                                                                             \
                __m256d tr, ti;                                                                           \
                _zttr_wm(R, m, _mm256_loadu_pd(SP + 16 * m + 8), _mm256_loadu_pd(SP + 16 * m + 12), &tr, &ti); \
                const __m256d Er = _mm256_loadu_pd(SP + 16 * m), Ei = _mm256_loadu_pd(SP + 16 * m + 4);   \
                double a[4], b[4], c[4], d[4];                                                            \
                _mm256_storeu_pd(a, _mm256_add_pd(Er, tr)); _mm256_storeu_pd(b, _mm256_add_pd(Ei, ti));   \
                _mm256_storeu_pd(c, _mm256_sub_pd(Er, tr)); _mm256_storeu_pd(d, _mm256_sub_pd(Ei, ti));   \
                z0[2 * m] = a[3]; z0[2 * m + 1] = b[3];                                                   \
                z0[2 * (m + H)] = c[3]; z0[2 * (m + H) + 1] = d[3];                                       \
            }                                                                                             \
        /* ── pass 2: pair m of the primary with pair R/2-1-m of the partner ── */                        \
        ZTTR_UNROLL for (int m = 0; m < H; m++)                                                           \
        {                                                                                                 \
            const int mp = H - 1 - m;                                                                     \
            __m256d tr, ti, ur, ui;                                                                       \
            _zttr_wm(R, m, _mm256_loadu_pd(SP + 16 * m + 8), _mm256_loadu_pd(SP + 16 * m + 12), &tr, &ti); \
            _zttr_wm(R, mp, _mm256_loadu_pd(SW + 16 * mp + 8), _mm256_loadu_pd(SW + 16 * mp + 12), &ur, &ui); \
            const __m256d EPr = _mm256_loadu_pd(SP + 16 * m), EPi = _mm256_loadu_pd(SP + 16 * m + 4);     \
            const __m256d EWr = _mm256_loadu_pd(SW + 16 * mp), EWi = _mm256_loadu_pd(SW + 16 * mp + 4);   \
            const __m256d P0r = _mm256_add_pd(EPr, tr), P0i = _mm256_add_pd(EPi, ti);   /* leg m       */ \
            const __m256d P1r = _mm256_sub_pd(EPr, tr), P1i = _mm256_sub_pd(EPi, ti);   /* leg m + R/2 */ \
            __m256d W0r = _mm256_add_pd(EWr, ur), W0i = _mm256_add_pd(EWi, ui);         /* leg mp      */ \
            __m256d W1r = _mm256_sub_pd(EWr, ur), W1i = _mm256_sub_pd(EWi, ui);         /* leg mp+R/2  */ \
            if (k == 0)                                                                                   \
            {   /* the partner of leg r's column 0 is column 0 of leg (R - r) mod R */                    \
                const int s1 = (R - m) % R;          /* for primary leg m: partner leg mp+R/2 */          \
                const int s0 = (R - (m + H)) % R;    /* for primary leg m+R/2: partner leg mp */          \
                W1r = _mm256_blend_pd(W1r, _mm256_set1_pd(z0[2 * s1]), 0x8);                              \
                W1i = _mm256_blend_pd(W1i, _mm256_set1_pd(z0[2 * s1 + 1]), 0x8);                          \
                W0r = _mm256_blend_pd(W0r, _mm256_set1_pd(z0[2 * s0]), 0x8);                              \
                W0i = _mm256_blend_pd(W0i, _mm256_set1_pd(z0[2 * s0 + 1]), 0x8);                          \
            }                                                                                             \
            ZTTR_UNROLL for (int half = 0; half < 2; half++)                                              \
            {                                                                                             \
                const int r = half ? m + H : m, rp = R - 1 - r;                                           \
                const __m256d Ar = half ? P1r : P0r, Ai = half ? P1i : P0i;                               \
                const __m256d Br = half ? W0r : W1r, Bi = half ? W0i : W1i;                               \
                const size_t f0 = (size_t)r * (size_t)L + (size_t)k;                                      \
                const __m256d S = _mm256_loadu_pd(affS + f0), C = _mm256_loadu_pd(affC + f0);             \
                const __m256d t1 = _mm256_sub_pd(Ar, Br), t2 = _mm256_add_pd(Ai, Bi);                     \
                const __m256d xr = _mm256_fmadd_pd(S, t1, _mm256_mul_pd(C, t2));                          \
                const __m256d xi = _mm256_fmsub_pd(S, t2, _mm256_mul_pd(C, t1));                          \
                const __m256d Xfr = _mm256_add_pd(Br, xr), Xfi = _mm256_sub_pd(xi, Bi);                   \
                const __m256d Xmr = _mm256_sub_pd(Ar, xr), Xmi = _mm256_sub_pd(xi, Ai);                   \
                double *of = X + 2 * f0;                                                                  \
                double *om = X + 2 * ((size_t)rp * (size_t)L + (size_t)(L - k - 3));                      \
                const __m256d fr = _mm256_permute4x64_pd(Xfr, 0x27), fi = _mm256_permute4x64_pd(Xfi, 0x27); \
                _mm256_storeu_pd(of, _mm256_unpacklo_pd(fr, fi));                                         \
                _mm256_storeu_pd(of + 4, _mm256_unpackhi_pd(fr, fi));                                     \
                const __m256d mr = _mm256_permute4x64_pd(Xmr, 0xD8), mi = _mm256_permute4x64_pd(Xmi, 0xD8); \
                _mm256_storeu_pd(om, _mm256_unpacklo_pd(mr, mi));                                         \
                _mm256_storeu_pd(om + 4, _mm256_unpackhi_pd(mr, mi));                                     \
            }                                                                                             \
        }                                                                                                 \
    }                                                                                                     \
    ZTTR_UNROLL for (int r = 0; r < R; r++)                                                               \
    {                                                                                                     \
        double *o = X + 2 * ((size_t)r * (size_t)L + (size_t)(L / 2));                                    \
        o[0] = cen[2 * r];                                                                                \
        o[1] = cen[2 * r + 1];                                                                            \
    }                                                                                                     \
}
ZTTR_TLFHB_BODY(4)
ZTTR_TLFHB_BODY(8)
#undef ZTTR_TLFHB_BODY

/* One call's work for the two fused kernels (tlfhc, t0h): the plan and a range
 * of columns. The serial run is one job over everything; the threaded run
 * (zttr_mt.h) gives every worker its own range. mode bit 0: compute the
 * peeled centre column into cen (tlfhc); bit 1: store it (tlfhc) / run the
 * centre column (t0h). */
typedef struct
{
    const vfft_zttr_plan_t *p;
    long lo, hi;
    int mode;
    double *cen;              /* tlfhc: 2R doubles, the caller's */
    const double *carry;      /* tlfhc, a threaded range: the RAW block at its first mirror column, 8 doubles a leg */
    const double *edge;       /* tlfhc, a threaded range: the RAW block its last partner load reads, 8 doubles a leg.
                               * The run is in the destination plane: the stage's input is split blocks and its
                               * output interleaved pairs, so the next range's first store overwrites two values of
                               * this block and this range's last store overwrites the rest -- both are read from a
                               * snapshot taken before the workers start (zttr_mt.h). NULL = read the plane. */
} _zttr_job_t;
typedef void (*_zttr_job_fn)(const _zttr_job_t *, const double *, double *);

/* ── the NATURAL-ORDER blocked terminator (tlfhc) ──
 * The primary quad keeps its lane order (lane i = column k+i, no reversal);
 * the partner window is REVERSED as it is assembled: lanes 0..3 = columns
 * L-k, L-k-1, L-k-2, L-k-3 = [B0 A3 A2 A1] from block A = (L-k-4 .. L-k-1)
 * and the carried block B = one permute4x64 (0x6C) and one blend per plane,
 * against the two shuffles plus the primary's reversal of tlfhb. Column 0's
 * partner is "column L" of the same input run, which by the run's
 * L-periodicity is its column 0: the carry starts as block (r, 0), the
 * window's record for column L is w_M^(r L), and the k = 0 iteration needs
 * no pre-pass and no output blend. Stores: X[f] through the plain REINT
 * (0xD8), X[m] with the reversal folded into its pre-permute (0x27) at the
 * window's (unaligned) address. Otherwise the tlfhb pass discipline. */
#define ZTTR_TLFHC_BODY(RR)                                                                               \
static __attribute__((noinline)) void _zttr_tlfhc##RR(const _zttr_job_t *jb, const double *W, double *X)     \
{                                                                                                         \
    enum { R = RR, H = RR / 2 };                                                                          \
    const vfft_zttr_plan_t *const p = jb->p;                                                              \
    const long L = p->L;                                                                                  \
    const long c0 = jb->lo ? L - jb->lo : 0;            /* the range's first mirror column (column L is column 0) */\
    const double *const ein = jb->edge;                   /* a range's last partner block, from the snapshot */    \
    const size_t M = (size_t)p->M;                                                                        \
    const double *affS = p->affS3, *affC = p->affC3;                                                      \
    double *const cen = jb->cen;                                                                            \
    double raw[40 * R + 8];                              /* the parking and the carry, aligned by hand */ \
    double *const SP = (double *)(((uintptr_t)raw + 63u) & ~(uintptr_t)63u);                              \
    double *const SW = SP + 16 * R;                                                                       \
    double *const CR = SW + 16 * R;                                                                       \
    const int pf = (W != X);        /* the plane mode: X went cold under W, prefetch its lines */        \
    if (jb->mode & 1) { /* the centre column L/2: lane 0 of the quad k = L/2, its partner lane 0 of leg R-1-r */         \
        const double *twr = p->twr3 + (size_t)(L / 8) * (size_t)(R - 1) * 8u;                             \
        __m256d pr[R], pi[R], Pr[R], Pi[R];                                                               \
        ZTTR_UNROLL for (int r = 0; r < R; r++)                                                           \
        {                                                                                                 \
            const double *a = W + 2 * ((size_t)r * (size_t)L + (size_t)(L / 2));                          \
            pr[r] = _mm256_loadu_pd(a);                                                                   \
            pi[r] = _mm256_loadu_pd(a + 4);                                                               \
        }                                                                                                 \
        if (R == 4) _zttr_comb4(pr, pi, twr, Pr, Pi); else _zttr_comb8(pr, pi, twr, Pr, Pi);              \
        ZTTR_UNROLL for (int r = 0; r < R; r++)                                                           \
        {                                                                                                 \
            const int rp = R - 1 - r;                                                                     \
            const size_t f = (size_t)r * (size_t)L + (size_t)(L / 2);                                     \
            const double S = affS[f], C = affC[f];                                                        \
            const double Ar = _mm256_cvtsd_f64(Pr[r]), Ai = _mm256_cvtsd_f64(Pi[r]);                      \
            const double Br = _mm256_cvtsd_f64(Pr[rp]), Bi = _mm256_cvtsd_f64(Pi[rp]);                    \
            const double t1 = Ar - Br, t2 = Ai + Bi;                                                      \
            const double xr = S * t1 + C * t2, xi = S * t2 - C * t1;                                      \
            cen[2 * r] = Br + xr;                                                                         \
            cen[2 * r + 1] = xi - Bi;                                                                     \
        }                                                                                                 \
    }                                                                                                     \
    ZTTR_UNROLL for (int r = 0; r < R; r++)                                                               \
    {   /* the carry's start: lane 0 of the block at the first mirror column */                                 \
        _mm256_storeu_pd(CR + 8 * r, jb->carry ? _mm256_loadu_pd(jb->carry + 8 * r) : _mm256_castpd128_pd256(_mm_load_sd(W + 2 * ((size_t)r * (size_t)L + (size_t)c0)))); \
        _mm256_storeu_pd(CR + 8 * r + 4, jb->carry ? _mm256_loadu_pd(jb->carry + 8 * r + 4) : _mm256_castpd128_pd256(_mm_load_sd(W + 2 * ((size_t)r * (size_t)L + (size_t)c0) + 4))); \
    }                                                                                                     \
    for (long k = jb->lo; k < jb->hi; k += 4)                                                             \
    {                                                                                                     \
        const long q = k / 4;                                                                             \
        const int edge = (k + 4 == jb->hi) && ein;                                                        \
        const double *twr = p->twr3 + (size_t)q * (size_t)(R - 1) * 8u;                                   \
        const double *twp = p->twp3 + (size_t)q * (size_t)(R - 1) * 8u;                                   \
        if (pf)                                                                                           \
            ZTTR_UNROLL for (int r = 0; r < R; r++)                                                       \
            {   /* four windows ahead: the primary's line, the partner's new block (tlfi's distance) */   \
                _mm_prefetch((const char *)(X + 2 * ((size_t)r * (size_t)L + (size_t)(k + 16))), _MM_HINT_T0); \
                _mm_prefetch((const char *)(X + 2 * ((size_t)r * (size_t)L + (size_t)(L - k - 20))), _MM_HINT_T0); \
            }                                                                                             \
        /* pass 1, the partner window reversed [B0 A3 A2 A1]: the even legs' half DFT parked, then the odd legs' */\
        ZTTR_UNROLL for (int par = 0; par < 2; par++)                                                    \
        {                                                                                                \
            __m256d xr[H], xi[H], Yr[H], Yi[H];                                                          \
            ZTTR_UNROLL for (int j = 0; j < H; j++)                                                      \
            {                                                                                            \
                const int r = 2 * j + par;                                                               \
                const double *a = W + 2 * ((size_t)r * (size_t)L + (size_t)(L - k - 4));                 \
                const double *as = edge ? ein + 8 * r : a; const __m256d ar = _mm256_loadu_pd(as);               \
                const __m256d ai = _mm256_loadu_pd(as + 4);                                              \
                __m256d wr = _mm256_blend_pd(_mm256_permute4x64_pd(ar, 0x6C), _mm256_loadu_pd(CR + 8 * r), 0x1);\
                __m256d wi = _mm256_blend_pd(_mm256_permute4x64_pd(ai, 0x6C), _mm256_loadu_pd(CR + 8 * r + 4), 0x1);\
                _mm256_storeu_pd(CR + 8 * r, ar); _mm256_storeu_pd(CR + 8 * r + 4, ai);                  \
                if (r) _zttr_twl(wr, wi, twp + (r - 1) * 8, &wr, &wi);                                   \
                xr[j] = wr; xi[j] = wi;                                                                  \
            }                                                                                            \
            if (R == 4)                                                                                  \
            {                                                                                            \
                Yr[0] = _mm256_add_pd(xr[0], xr[1]); Yi[0] = _mm256_add_pd(xi[0], xi[1]);                \
                Yr[1] = _mm256_sub_pd(xr[0], xr[1]); Yi[1] = _mm256_sub_pd(xi[0], xi[1]);                \
            }                                                                                            \
            else _zttr_dft4(xr, xi, Yr, Yi);                                                             \
            ZTTR_UNROLL for (int m = 0; m < H; m++)                                                      \
            {                                                                                            \
                _mm256_storeu_pd(SW + 16 * m + 8 * par, Yr[m]);                                          \
                _mm256_storeu_pd(SW + 16 * m + 8 * par + 4, Yi[m]);                                      \
            }                                                                                            \
        }                                                                                                \
        /* pass 1, the primary quad, natural lanes: the even legs' half DFT parked, then the odd legs' */\
        ZTTR_UNROLL for (int par = 0; par < 2; par++)                                                    \
        {                                                                                                \
            __m256d xr[H], xi[H], Yr[H], Yi[H];                                                          \
            ZTTR_UNROLL for (int j = 0; j < H; j++)                                                      \
            {                                                                                            \
                const int r = 2 * j + par;                                                               \
                const double *a = W + 2 * ((size_t)r * (size_t)L + (size_t)k);                           \
                __m256d wr = _mm256_loadu_pd(a), wi = _mm256_loadu_pd(a + 4);                            \
                if (r) _zttr_twl(wr, wi, twr + (r - 1) * 8, &wr, &wi);                                   \
                xr[j] = wr; xi[j] = wi;                                                                  \
            }                                                                                            \
            if (R == 4)                                                                                  \
            {                                                                                            \
                Yr[0] = _mm256_add_pd(xr[0], xr[1]); Yi[0] = _mm256_add_pd(xi[0], xi[1]);                \
                Yr[1] = _mm256_sub_pd(xr[0], xr[1]); Yi[1] = _mm256_sub_pd(xi[0], xi[1]);                \
            }                                                                                            \
            else _zttr_dft4(xr, xi, Yr, Yi);                                                             \
            ZTTR_UNROLL for (int m = 0; m < H; m++)                                                      \
            {                                                                                            \
                _mm256_storeu_pd(SP + 16 * m + 8 * par, Yr[m]);                                          \
                _mm256_storeu_pd(SP + 16 * m + 8 * par + 4, Yi[m]);                                      \
            }                                                                                            \
        }                                                                                                \
        /* pass 2: pair m of the primary with pair R/2-1-m of the partner */                              \
        ZTTR_UNROLL for (int m = 0; m < H; m++)                                                           \
        {                                                                                                 \
            const int mp = H - 1 - m;                                                                     \
            __m256d tr, ti, ur, ui;                                                                       \
            _zttr_wm(R, m, _mm256_loadu_pd(SP + 16 * m + 8), _mm256_loadu_pd(SP + 16 * m + 12), &tr, &ti); \
            _zttr_wm(R, mp, _mm256_loadu_pd(SW + 16 * mp + 8), _mm256_loadu_pd(SW + 16 * mp + 12), &ur, &ui); \
            const __m256d EPr = _mm256_loadu_pd(SP + 16 * m), EPi = _mm256_loadu_pd(SP + 16 * m + 4);     \
            const __m256d EWr = _mm256_loadu_pd(SW + 16 * mp), EWi = _mm256_loadu_pd(SW + 16 * mp + 4);   \
            const __m256d P0r = _mm256_add_pd(EPr, tr), P0i = _mm256_add_pd(EPi, ti);   /* leg m       */ \
            const __m256d P1r = _mm256_sub_pd(EPr, tr), P1i = _mm256_sub_pd(EPi, ti);   /* leg m + R/2 */ \
            const __m256d W0r = _mm256_add_pd(EWr, ur), W0i = _mm256_add_pd(EWi, ui);   /* leg mp      */ \
            const __m256d W1r = _mm256_sub_pd(EWr, ur), W1i = _mm256_sub_pd(EWi, ui);   /* leg mp+R/2  */ \
            ZTTR_UNROLL for (int half = 0; half < 2; half++)                                              \
            {                                                                                             \
                const int r = half ? m + H : m;                                                           \
                const __m256d Ar = half ? P1r : P0r, Ai = half ? P1i : P0i;                               \
                const __m256d Br = half ? W0r : W1r, Bi = half ? W0i : W1i;                               \
                const size_t f0 = (size_t)r * (size_t)L + (size_t)k;                                      \
                const __m256d S = _mm256_loadu_pd(affS + f0), C = _mm256_loadu_pd(affC + f0);             \
                const __m256d t1 = _mm256_sub_pd(Ar, Br), t2 = _mm256_add_pd(Ai, Bi);                     \
                const __m256d xr = _mm256_fmadd_pd(S, t1, _mm256_mul_pd(C, t2));                          \
                const __m256d xi = _mm256_fmsub_pd(S, t2, _mm256_mul_pd(C, t1));                          \
                const __m256d Xfr = _mm256_add_pd(Br, xr), Xfi = _mm256_sub_pd(xi, Bi);                   \
                const __m256d Xmr = _mm256_sub_pd(Ar, xr), Xmi = _mm256_sub_pd(xi, Ai);                   \
                double *of = X + 2 * f0;                                                                  \
                double *om = X + 2 * (M - f0 - 3);                                                        \
                const __m256d fr = _mm256_permute4x64_pd(Xfr, 0xD8), fi = _mm256_permute4x64_pd(Xfi, 0xD8); \
                _mm256_storeu_pd(of, _mm256_unpacklo_pd(fr, fi));                                         \
                _mm256_storeu_pd(of + 4, _mm256_unpackhi_pd(fr, fi));                                     \
                const __m256d mr = _mm256_permute4x64_pd(Xmr, 0x27), mi = _mm256_permute4x64_pd(Xmi, 0x27); \
                _mm256_storeu_pd(om, _mm256_unpacklo_pd(mr, mi));                                         \
                _mm256_storeu_pd(om + 4, _mm256_unpackhi_pd(mr, mi));                                     \
            }                                                                                             \
        }                                                                                                 \
    }                                                                                                     \
    if (jb->mode & 2)                                                                                   \
    ZTTR_UNROLL for (int r = 0; r < R; r++)                                                               \
    {                                                                                                     \
        double *o = X + 2 * ((size_t)r * (size_t)L + (size_t)(L / 2));                                    \
        o[0] = cen[2 * r];                                                                                \
        o[1] = cen[2 * r + 1];                                                                            \
    }                                                                                                     \
}
ZTTR_TLFHC_BODY(4)
ZTTR_TLFHC_BODY(8)
#undef ZTTR_TLFHC_BODY

/* THE STACK-ALIGNING ENTRY. Win64 gives a callee 16-B stack alignment and
 * mingw gcc never realigns a frame, so a spilling kernel's ymm slots split
 * cache lines or not by the CALLER's rsp (up to 30% either way -- the
 * alignment lottery). The blocked terminators are called through this
 * trampoline: rsp is set to a chosen residue mod 64 before the call, so the
 * kernel's frame sits at one state whatever the caller does, and the state
 * (p->stk) is a plan knob the calibration can race. Win64 call: args in
 * rcx, rdx, r8; 32 B of shadow space below the return address; rcx, rdx,
 * r8-r11, rax and every vector register volatile (ymm6-15's upper halves
 * are not preserved either), r12 callee-saved.
 * The entry exists for the Win64 ABI under GCC-style inline asm (mingw gcc,
 * clang, icx). Elsewhere -- the System V ABI passes rdi, rsi, rdx and keeps
 * a red zone below rsp that the entry's call would overwrite -- the
 * terminator is entered directly and p->stk binds nothing. */
typedef void (*_zttr_term_fn)(const vfft_zttr_plan_t *, const double *, double *);
#if defined(_WIN64) && defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
static inline void _zttr_call_aligned(_zttr_term_fn fn, const vfft_zttr_plan_t *p, const double *W, double *X)
{
    register const vfft_zttr_plan_t *a0 __asm__("rcx") = p;
    register const double *a1 __asm__("rdx") = W;
    register double *a2 __asm__("r8") = X;
    register _zttr_term_fn f __asm__("r10") = fn;
    register long ad __asm__("r11") = 32 + 16 * (long)(p->stk & 3);
    __asm__ volatile(
        "movq %%rsp, %%r12\n\t"
        "andq $-64, %%rsp\n\t"
        "subq %%r11, %%rsp\n\t"
        "call *%%r10\n\t"
        "movq %%r12, %%rsp\n\t"
        : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(f), "+r"(ad)
        :
        : "rax", "r9", "r12", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7",
          "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15", "memory", "cc");
}

/* the same aligned entry for a job (tlfhc, t0h): the stack state is the job's plan's */
static inline void _zttr_call_job(_zttr_job_fn fn, const _zttr_job_t *jb, const double *W, double *X)
{
    register const _zttr_job_t *a0 __asm__("rcx") = jb;
    register const double *a1 __asm__("rdx") = W;
    register double *a2 __asm__("r8") = X;
    register _zttr_job_fn f __asm__("r10") = fn;
    register long ad __asm__("r11") = 32 + 16 * (long)(jb->p->stk & 3);
    __asm__ volatile(
        "movq %%rsp, %%r12\n\t"
        "andq $-64, %%rsp\n\t"
        "subq %%r11, %%rsp\n\t"
        "call *%%r10\n\t"
        "movq %%r12, %%rsp\n\t"
        : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(f), "+r"(ad)
        :
        : "rax", "r9", "r12", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7",
          "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15", "memory", "cc");
}
#else
static inline void _zttr_call_aligned(_zttr_term_fn fn, const vfft_zttr_plan_t *p, const double *W, double *X)
{
    fn(p, W, X);
}

static inline void _zttr_call_job(_zttr_job_fn fn, const _zttr_job_t *jb, const double *W, double *X)
{
    fn(jb, W, X);
}
#endif

static inline void _zttr_tlfh(const vfft_zttr_plan_t *p, const double *W, double *X)
{
    if (p->blocked == 2)
    {   /* the whole run as one job: every column quad, the centre computed and stored */
        double cen[16];
        const _zttr_job_t jb = { p, 0, p->L / 2, 3, cen };
        _zttr_call_job(p->R == 4 ? _zttr_tlfhc4 : _zttr_tlfhc8, &jb, W, X);
        return;
    }
    if (p->blocked) { _zttr_call_aligned(p->R == 4 ? _zttr_tlfhb4 : _zttr_tlfhb8, p, W, X); return; }
    if (p->R == 4) _zttr_tlfh4(p, W, X); else _zttr_tlfh8(p, W, X);
}
#endif /* __AVX2__ */

/* r2c: x[N] -> X[0..M] CCE (N+2 doubles). The pipeline runs in the plane W:
 * the ingest scatters x's packed view into W's first 2M doubles, the mids
 * run there, the terminator untangles W into X (W == X out of place; the
 * plan's scratch in place, where x == X). x must not alias W. */
static inline void _zttr_run_fwd(const vfft_zttr_plan_t *p, const double *x, double *W, double *X)
{
    const vfft_ztt_plan_t *zt = p->zt;
    const int nf = zt->nf;
    const size_t M = (size_t)p->M, tile = zt->tile;
    const double *tw = zt->tw;
    zt->st_fwd[0](x, 0, W, 0, 0, (const double *)zt->rb, (size_t)zt->ncol, 0, 0, 0, (size_t)zt->ncol);
    if (tile)
        for (size_t t = 0; t < M / tile; t++)
        {
            double *B = W + t * tile * 2;
            for (int s = 1; s < nf - 1; s++)
            {
                const size_t RL = (size_t)zt->L[s] * (size_t)zt->chain[s];
                if (tile % RL == 0)
                    zt->st_fwd[s](B, 0, B, 0, tw + zt->twoff[s], 0, (size_t)zt->L[s], tile / RL, 0, 0, (size_t)zt->L[s]);
            }
        }
    for (int s = 1; s < nf - 1; s++)
    {
        const size_t RL = (size_t)zt->L[s] * (size_t)zt->chain[s];
        if (!tile || tile % RL)
            zt->st_fwd[s](W, 0, W, 0, tw + zt->twoff[s], 0, (size_t)zt->L[s], (size_t)zt->Gs[s], 0, 0, (size_t)zt->L[s]);
    }
#if defined(__AVX2__)
    _zttr_tlfh(p, W, X);
#else
    (void)W; (void)X;
#endif
}
static inline void vfft_zttr_execute_fwd(const vfft_zttr_plan_t *p, const double *x, double *X)
{
    _zttr_run_fwd(p, x, X, X);
}
static inline void vfft_zttr_execute_fwd_ip(const vfft_zttr_plan_t *p, double *xX)
{
    _zttr_run_fwd(p, xX, p->scratch, xX);
}

/* ── THE c2r TWIN: the backward fold fused into the backward ingest ──
 * The c2r is Zhat = fold_bwd(X) (zr2c.h: for each pair n, M-n:
 *   t = X[n] - conj(X[M-n]),  e = X[n] + conj(X[M-n]),  cy = t * (sin, -cos)(2 pi n/N),
 *   Zhat[n] = e - cy,  Zhat[M-n] = conj(e + cy))
 * followed by the c2c backward at M; the ZTT's backward ingest (t0tp bwd)
 * reads Zhat[j Ls + c] for legs j = 0..R0-1 (Ls = M/R0), two columns per
 * vector, and writes each column's radix-R0 run to the plane at rb[c].
 * Fused, the ingest reads X instead: the untangle is POINTWISE on the pair,
 * so it sits at the load edge before any butterfly -- nothing to park at
 * R0 = 4, the mirror columns' eight legs at R0 = 8. The mirror of (leg j,
 * column c) is (leg R0-1-j, column Ls-c): each iteration takes the column
 * pair (k, k+1) of every leg and the pair (Ls-k, Ls-k-1) of the mirror leg
 * -- one 32-B load each, the mirror's lanes swapped once -- untangles them
 * into both pairs' Zhat, and runs FOUR butterflies: columns k, k+1 and the
 * mirror columns Ls-k, Ls-k-1 (their legs in reversed order). Column 0's
 * partners are column 0 of the other legs (and X[M] for leg 0), which is
 * exactly what the mirror load [X[j'Ls + Ls - 1] | X[(j'+1) Ls]] holds; the
 * mirror "column Ls" it produces does not exist and is not stored. Column
 * Ls/2 pairs with itself across legs and is peeled. X[0], X[M] are taken
 * real (their imaginary lanes zeroed), as the fold does. */
static inline __attribute__((always_inline)) void _zttr_unt_bwd(
    const __m256d F, const __m256d Mv, const __m256d wr, const __m256d wi, __m256d *zf, __m256d *zm)
{
    const __m256d CONJ = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0);
    const __m256d cm = _mm256_xor_pd(Mv, CONJ);
    const __m256d t = _mm256_sub_pd(F, cm), e = _mm256_add_pd(F, cm);
    const __m256d ts = _mm256_permute_pd(t, 0x5);
    const __m256d cy = _mm256_fmaddsub_pd(wr, t, _mm256_mul_pd(wi, ts));
    *zf = _mm256_sub_pd(e, cy);
    *zm = _mm256_xor_pd(_mm256_add_pd(e, cy), CONJ);
}
/* the backward radix-4 / radix-8 butterflies on interleaved [c | c+1]
 * vectors, transcribed from the emitted t0tp bwd bodies (conjugate roots:
 * the rotation is +i); outputs in natural order */
static inline __attribute__((always_inline)) void _zttr_bf4b(const __m256d *a, __m256d *Y)
{
    const __m256d pim = _mm256_setr_pd(-0.0, 0.0, -0.0, 0.0);
    const __m256d t0 = _mm256_add_pd(a[0], a[2]), t1 = _mm256_sub_pd(a[0], a[2]);
    const __m256d t2 = _mm256_add_pd(a[1], a[3]), t3 = _mm256_sub_pd(a[1], a[3]);
    const __m256d r = _mm256_xor_pd(_mm256_permute_pd(t3, 0x5), pim);
    Y[0] = _mm256_add_pd(t0, t2); Y[2] = _mm256_sub_pd(t0, t2);
    Y[1] = _mm256_add_pd(t1, r);  Y[3] = _mm256_sub_pd(t1, r);
}
static inline __attribute__((always_inline)) void _zttr_bf8b(const __m256d *a, __m256d *Y)
{
    const __m256d pim = _mm256_setr_pd(-0.0, 0.0, -0.0, 0.0);
    const __m256d rh = _mm256_set1_pd(0.70710678118654752440);
    const __m256d b0 = _mm256_add_pd(a[0], a[4]), d0 = _mm256_sub_pd(a[0], a[4]);
    const __m256d b1 = _mm256_add_pd(a[1], a[5]), d1 = _mm256_sub_pd(a[1], a[5]);
    const __m256d b2 = _mm256_add_pd(a[2], a[6]), d2 = _mm256_sub_pd(a[2], a[6]);
    const __m256d b3 = _mm256_add_pd(a[3], a[7]), d3 = _mm256_sub_pd(a[3], a[7]);
    const __m256d c0 = d0;
    const __m256d d1i = _mm256_xor_pd(_mm256_permute_pd(d1, 0x5), pim);
    const __m256d c1 = _mm256_mul_pd(_mm256_add_pd(d1, d1i), rh);
    const __m256d c2 = _mm256_xor_pd(_mm256_permute_pd(d2, 0x5), pim);
    const __m256d d3i = _mm256_xor_pd(_mm256_permute_pd(d3, 0x5), pim);
    const __m256d c3 = _mm256_mul_pd(_mm256_sub_pd(d3i, d3), rh);
    const __m256d e0 = _mm256_add_pd(b0, b2), e1 = _mm256_sub_pd(b0, b2);
    const __m256d e2 = _mm256_add_pd(b1, b3), e3 = _mm256_sub_pd(b1, b3);
    const __m256d er = _mm256_xor_pd(_mm256_permute_pd(e3, 0x5), pim);
    const __m256d o0 = _mm256_add_pd(c0, c2), o1 = _mm256_sub_pd(c0, c2);
    const __m256d o2 = _mm256_add_pd(c1, c3), o3 = _mm256_sub_pd(c1, c3);
    const __m256d orr = _mm256_xor_pd(_mm256_permute_pd(o3, 0x5), pim);
    Y[0] = _mm256_add_pd(e0, e2); Y[4] = _mm256_sub_pd(e0, e2);
    Y[2] = _mm256_add_pd(e1, er); Y[6] = _mm256_sub_pd(e1, er);
    Y[1] = _mm256_add_pd(o0, o2); Y[5] = _mm256_sub_pd(o0, o2);
    Y[3] = _mm256_add_pd(o1, orr); Y[7] = _mm256_sub_pd(o1, orr);
}
/* the run store: four natural-order outputs Y[q..q+3] of the column pair ->
 * one [re x4][im x4] block per column (pl = the low lanes' column, ph = the
 * high lanes'); NULL skips a column */
static inline __attribute__((always_inline)) void _zttr_blk_store(
    const __m256d *Y, double *pl, double *ph)
{
    const __m256d u0 = _mm256_unpacklo_pd(Y[0], Y[1]), u1 = _mm256_unpackhi_pd(Y[0], Y[1]);
    const __m256d u2 = _mm256_unpacklo_pd(Y[2], Y[3]), u3 = _mm256_unpackhi_pd(Y[2], Y[3]);
    if (pl)
    {
        _mm256_storeu_pd(pl, _mm256_permute2f128_pd(u0, u2, 0x20));
        _mm256_storeu_pd(pl + 4, _mm256_permute2f128_pd(u1, u3, 0x20));
    }
    if (ph)
    {
        _mm256_storeu_pd(ph, _mm256_permute2f128_pd(u0, u2, 0x31));
        _mm256_storeu_pd(ph + 4, _mm256_permute2f128_pd(u1, u3, 0x31));
    }
}
#define ZTTR_T0H_BODY(RR)                                                                                 \
static __attribute__((noinline)) void _zttr_t0h##RR(const _zttr_job_t *jb, const double *X, double *W)     \
{                                                                                                         \
    enum { R = RR, BLK = 2 * RR };                    /* doubles per column run */                        \
    const vfft_zttr_plan_t *const p = jb->p;                                                              \
    const size_t Ls = (size_t)p->zt->ncol;                                                                \
    const size_t *rb = p->zt->rb;                                                                         \
    const double *SB = p->bS, *CB = p->bC;                                                                \
    const __m256d ZLO = _mm256_setzero_pd();                                                              \
    for (size_t k = (size_t)jb->lo; k < (size_t)jb->hi; k += 2)                                         \
    {                                                                                                     \
        __m256d z[R], m[R], Y[R];                                                                         \
        ZTTR_UNROLL for (int j = 0; j < R; j++)                                                           \
        {                                                                                                 \
            const size_t n = (size_t)j * Ls + k, jp = (size_t)(R - 1 - j);                                \
            __m256d F = _mm256_loadu_pd(X + 2 * n);                                                       \
            __m256d Mv = _mm256_loadu_pd(X + 2 * (jp * Ls + Ls - k - 1));                                 \
            Mv = _mm256_permute2f128_pd(Mv, Mv, 0x01);           /* [X[M-n] | X[M-n-1]] */                 \
            if (k == 0 && j == 0) { F = _mm256_blend_pd(F, ZLO, 0x2); Mv = _mm256_blend_pd(Mv, ZLO, 0x2); } \
            _zttr_unt_bwd(F, Mv, _mm256_loadu_pd(SB + 2 * n), _mm256_loadu_pd(CB + 2 * n), &z[j], &m[R - 1 - j]); \
        }                                                                                                 \
        if (R == 4) _zttr_bf4b(z, Y); else _zttr_bf8b(z, Y);                                              \
        _zttr_blk_store(Y, W + BLK * rb[k], W + BLK * rb[k + 1]);                                         \
        if (R == 8) _zttr_blk_store(Y + 4, W + BLK * rb[k] + 8, W + BLK * rb[k + 1] + 8);                  \
        if (R == 4) _zttr_bf4b(m, Y); else _zttr_bf8b(m, Y);                                              \
        _zttr_blk_store(Y, k ? W + BLK * rb[Ls - k] : NULL, W + BLK * rb[Ls - k - 1]);                    \
        if (R == 8) _zttr_blk_store(Y + 4, k ? W + BLK * rb[Ls - k] + 8 : NULL, W + BLK * rb[Ls - k - 1] + 8); \
    }                                                                                                     \
    if (jb->mode & 2) { /* the centre column Ls/2: its partner is column Ls/2 of the mirror leg; the high lane          \
         * (column Ls/2 + 1) was stored by the loop as a mirror and is left alone */                    \
        __m256d z[R], m[R], Y[R];                                                                         \
        const size_t k = Ls / 2;                                                                          \
        ZTTR_UNROLL for (int j = 0; j < R; j++)                                                           \
        {                                                                                                 \
            const size_t n = (size_t)j * Ls + k, jp = (size_t)(R - 1 - j);                                \
            const __m256d F = _mm256_loadu_pd(X + 2 * n);                                                 \
            __m256d Mv = _mm256_loadu_pd(X + 2 * (jp * Ls + Ls - k - 1));                                 \
            Mv = _mm256_permute2f128_pd(Mv, Mv, 0x01);                                                    \
            _zttr_unt_bwd(F, Mv, _mm256_loadu_pd(SB + 2 * n), _mm256_loadu_pd(CB + 2 * n), &z[j], &m[R - 1 - j]); \
        }                                                                                                 \
        if (R == 4) _zttr_bf4b(z, Y); else _zttr_bf8b(z, Y);                                              \
        _zttr_blk_store(Y, W + BLK * rb[k], NULL);                                                        \
        if (R == 8) _zttr_blk_store(Y + 4, W + BLK * rb[k] + 8, NULL);                                    \
    }                                                                                                     \
}
ZTTR_T0H_BODY(4)
ZTTR_T0H_BODY(8)
#undef ZTTR_T0H_BODY

/* c2r: X[0..M] CCE (N+2 doubles) -> x[N]. The fused ingest untangles X into
 * the plane W, the backward mids run there, the last stage lands in x
 * (W == x out of place; the plan's scratch in place, where X == x).
 * Unnormalised, as the door's c2r. */
static inline void _zttr_run_bwd(const vfft_zttr_plan_t *p, const double *X, double *W, double *x)
{
    const vfft_ztt_plan_t *zt = p->zt;
    const int nf = zt->nf;
    const size_t M = (size_t)p->M, tile = zt->tile;
    const double *tw = zt->twb;
    {   /* the whole fused ingest as one job: every column pair, then the centre column */
        const _zttr_job_t jb = { p, 0, (long)(zt->ncol / 2), 2, NULL };
        _zttr_call_job(zt->chain[0] == 4 ? _zttr_t0h4 : _zttr_t0h8, &jb, X, W);
    }
    if (tile)
        for (size_t t = 0; t < M / tile; t++)
        {
            double *B = W + t * tile * 2;
            for (int s = 1; s < nf - 1; s++)
            {
                const size_t RL = (size_t)zt->L[s] * (size_t)zt->chain[s];
                if (tile % RL == 0)
                    zt->st_bwd[s](B, 0, B, 0, tw + zt->twoff[s], 0, (size_t)zt->L[s], tile / RL, 0, 0, (size_t)zt->L[s]);
            }
        }
    for (int s = 1; s < nf - 1; s++)
    {
        const size_t RL = (size_t)zt->L[s] * (size_t)zt->chain[s];
        if (!tile || tile % RL)
            zt->st_bwd[s](W, 0, W, 0, tw + zt->twoff[s], 0, (size_t)zt->L[s], (size_t)zt->Gs[s], 0, 0, (size_t)zt->L[s]);
    }
    {   /* the plane mode (W != x): the ZTT's prefetching in-place terminator,
         * bitwise tlf -- x went cold under the plane */
        const size_t L = (size_t)zt->L[nf - 1];
        const vfft_ztt_kfn last = (W != x && zt->tl_bwd_plane) ? zt->tl_bwd_plane : zt->st_bwd[nf - 1];
        last(W, 0, x, 0, tw + zt->twoff[nf - 1], 0, L, 1, L, 0, L);
    }
}
static inline void vfft_zttr_execute_bwd(const vfft_zttr_plan_t *p, const double *X, double *x)
{
    _zttr_run_bwd(p, X, x, x);
}
static inline void vfft_zttr_execute_bwd_ip(const vfft_zttr_plan_t *p, double *Xx)
{
    _zttr_run_bwd(p, Xx, p->scratch, Xx);
}

#endif /* VFFT_ZTTR_H */
