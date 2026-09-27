/* vtw_avx512.h — every AVX-512 twiddle table of the IL runtime, and nothing else.
 *
 * Included only when the build's ISA is avx512 (VFFT_IL_VW == 8, il_isa.h). Each
 * AVX2 table builder in the runtime has its AVX-512 counterpart here, taking the
 * same inputs; the call site keeps its AVX2 code as it was and calls the
 * counterpart under #if VFFT_IL_VW == 8. The AVX2 builders never see this file
 * and nothing here reads an AVX2 table.
 *
 * THE RECORDS (the contract the avx512 codelets state in their headers). A
 * record is 16 doubles, one zmm of cosines then one zmm of sines. For a
 * twiddle w, c = Re w and t = Im w:
 *   PAIR   (VTW2: t2, t2t, t2tg, t2cs, t2csg T1, the pair and chain3 mids)
 *          4 complex columns per record, column j in slot j:
 *          [c0 c0 c1 c1 c2 c2 c3 c3][-t0 +t0 -t1 +t1 -t2 +t2 -t3 +t3]
 *   BCAST  (t2c, t2cp, t2csg T2: one digit per record) the same twiddle in
 *          all four slots.
 *   SPLAT  (msz) [c x8][t x8].
 *   LANE   (ZTURN-T stage streams) 8 columns per record, column i in lane i:
 *          [c0 .. c7][t0 .. t7].
 * A table of `cols` columns has ceil(cols / 4) pair records per leg; the
 * slots past the last column carry the next column's angle (valid values,
 * never read), as the AVX2 tables do. The odd-count tails read their columns
 * at the lane offset (il_odd_count_tail.md; docs/design/avx512_tail_handling.md).
 *
 * Values: every angle goes through vfft_cs2pi_exact with the SAME integer
 * arguments the AVX2 builder passes, so a column's twiddle is bitwise the one
 * its AVX2 twin table carries; only the placement differs. The ZTURN-T pow2
 * streams expand from the baked quarter wave (ztt_qw16384.h), as ztt.h does. */
#ifndef VFFT_VTW_AVX512_H
#define VFFT_VTW_AVX512_H

#include <stddef.h>
#include "tw_exact.h"
#include "ztt_qw16384.h"

#define VFFT_VTW512_VW  8    /* doubles per zmm */
#define VFFT_VTW512_PER 4    /* complex columns per pair record */
#define VFFT_VTW512_REC 16   /* doubles per record */

/* the block -> natural index map of a table's sets (NULL = identity) */
typedef size_t (*vfft_vtw512_q_fn)(const void *ctx, size_t b);

/* ── records ──────────────────────────────────────────────────────────── */

/* w = exp(-2*pi*i*p/n): *c = Re w, *t = Im w; conj gives the conjugate */
static inline void _vtw512_w(long long p, long long n, int conj, double *c, double *t)
{
    double s;
    vfft_cs2pi_exact(p, n, c, &s);
    *t = conj ? s : -s;
}
static inline void vfft_vtw512_pair_put(double *rec, int j, double c, double t)
{
    rec[2 * j] = c;
    rec[2 * j + 1] = c;
    rec[VFFT_VTW512_VW + 2 * j] = -t;
    rec[VFFT_VTW512_VW + 2 * j + 1] = t;
}
static inline void vfft_vtw512_bcast(double *rec, double c, double t)
{
    int j;
    for (j = 0; j < VFFT_VTW512_PER; j++) vfft_vtw512_pair_put(rec, j, c, t);
}
static inline void vfft_vtw512_splat(double *rec, double c, double t)
{
    int i;
    for (i = 0; i < VFFT_VTW512_VW; i++) { rec[i] = c; rec[VFFT_VTW512_VW + i] = t; }
}
static inline size_t vfft_vtw512_groups(size_t cols)
{
    return (cols + VFFT_VTW512_PER - 1) / VFFT_VTW512_PER;
}

/* ── the pair engine (il2p.h create): the t2 mid's stream ─────────────────
 * column k in [0, R2), leg l in [1, R1): w = exp(-2*pi*i*l*k/N); record
 * (group k/4, leg l) at ((k/4)*(R1-1) + l-1)*16. */
static inline size_t vfft_vtw512_pair2p_doubles(int R1, int R2)
{
    return vfft_vtw512_groups((size_t)R2) * (size_t)(R1 - 1) * VFFT_VTW512_REC;
}
static inline void vfft_vtw512_pair2p_fill(double *tw, int R1, int R2, int N, int conj)
{
    const size_t ng = vfft_vtw512_groups((size_t)R2);
    size_t g;
    int l, j;
    for (g = 0; g < ng; g++)
        for (l = 1; l < R1; l++) {
            double *rec = tw + (g * (size_t)(R1 - 1) + (size_t)(l - 1)) * VFFT_VTW512_REC;
            for (j = 0; j < VFFT_VTW512_PER; j++) {
                double c, t;
                _vtw512_w((long long)l * (long long)(VFFT_VTW512_PER * g + (size_t)j),
                          (long long)N, conj, &c, &t);
                vfft_vtw512_pair_put(rec, j, c, t);
            }
        }
}

/* ── the 3-stage chain (il2p.h, _vfft_il3p_vtw2): per block, cols columns ─
 * column = the GLOBAL index blk*cols + local; w = exp(-2*pi*i*l*k/modulus).
 * Laid per block with a ceiling group count, so a block's odd tail has its
 * record (the kernel pairs columns from its own base). */
static inline size_t vfft_vtw512_il3p_recs(int cols)
{
    return vfft_vtw512_groups((size_t)cols);
}
static inline size_t vfft_vtw512_il3p_doubles(int legs, int blocks, int cols)
{
    return (size_t)blocks * vfft_vtw512_il3p_recs(cols) * (size_t)(legs - 1) * VFFT_VTW512_REC;
}
static inline void vfft_vtw512_il3p_fill(double *tw, int legs, int blocks, int cols,
                                         int modulus, int conj)
{
    const size_t ng = vfft_vtw512_il3p_recs(cols);
    size_t g;
    int blk, l, j;
    for (blk = 0; blk < blocks; blk++)
        for (g = 0; g < ng; g++)
            for (l = 1; l < legs; l++) {
                double *rec = tw + (((size_t)blk * ng + g) * (size_t)(legs - 1)
                                    + (size_t)(l - 1)) * VFFT_VTW512_REC;
                for (j = 0; j < VFFT_VTW512_PER; j++) {
                    double c, t;
                    _vtw512_w((long long)l * ((long long)blk * cols
                                              + (long long)(VFFT_VTW512_PER * g + (size_t)j)),
                              (long long)modulus, conj, &c, &t);
                    vfft_vtw512_pair_put(rec, j, c, t);
                }
            }
}

/* ── one twiddle per (block, leg): the per-digit tables ───────────────────
 * Block b's natural index Q = q(ctx, b) (identity when q is NULL); record
 * (b, leg l) at (b*(R-1) + l-1)*16 carries w = exp(-2*pi*i*(l*Q mod L)/L).
 * BCAST: the flat DIT's t2cp stages, the t2csg T2 base (R = 2, one record
 * per group) and the 2D column stages (d-major, Q = d).
 * SPLAT: the flat DIT's msz stages. */
static inline size_t vfft_vtw512_blocks_doubles(size_t nb, int R)
{
    return nb * (size_t)(R - 1) * VFFT_VTW512_REC;
}
static inline void _vtw512_blocks(double *tw, size_t nb, int R, size_t L,
                                  vfft_vtw512_q_fn q, const void *ctx, int conj, int splat)
{
    size_t b;
    int l;
    for (b = 0; b < nb; b++) {
        const size_t Q = q ? q(ctx, b) : b;
        for (l = 1; l < R; l++) {
            double *rec = tw + (b * (size_t)(R - 1) + (size_t)(l - 1)) * VFFT_VTW512_REC;
            double c, t;
            _vtw512_w((long long)((size_t)l * Q % L), (long long)L, conj, &c, &t);
            if (splat) vfft_vtw512_splat(rec, c, t);
            else vfft_vtw512_bcast(rec, c, t);
        }
    }
}
static inline void vfft_vtw512_blocks_bcast(double *tw, size_t nb, int R, size_t L,
                                            vfft_vtw512_q_fn q, const void *ctx, int conj)
{
    _vtw512_blocks(tw, nb, R, L, q, ctx, conj, 0);
}
static inline void vfft_vtw512_blocks_splat(double *tw, size_t nb, int R, size_t L,
                                            vfft_vtw512_q_fn q, const void *ctx, int conj)
{
    _vtw512_blocks(tw, nb, R, L, q, ctx, conj, 1);
}

/* ── the flat DIT's t2csg T1: one pair record per 4 blocks of a group ─────
 * column jj in [0, G): w = exp(-2*pi*i*(W*jj mod L)/L) (the group-internal
 * step; the kernel forms W^1 = T1 x T2 and derives the higher legs). */
static inline size_t vfft_vtw512_step_doubles(size_t G)
{
    return vfft_vtw512_groups(G) * VFFT_VTW512_REC;
}
static inline void vfft_vtw512_step_fill(double *tw, size_t G, size_t W, size_t L, int conj)
{
    const size_t ng = vfft_vtw512_groups(G);
    size_t g;
    int j;
    for (g = 0; g < ng; g++) {
        double *rec = tw + g * VFFT_VTW512_REC;
        for (j = 0; j < VFFT_VTW512_PER; j++) {
            const size_t jj = VFFT_VTW512_PER * g + (size_t)j;
            double c, t;
            _vtw512_w((long long)((W * jj) % L), (long long)L, conj, &c, &t);
            vfft_vtw512_pair_put(rec, j, c, t);
        }
    }
}

/* ── the flat DIT's t2cs: columns = the G blocks of a group ───────────────
 * Per group g: ceil(G/4) pair records per leg, slot j of record pp = block
 * b2 = g*G + 4*pp + j (Q = q(b2), 0 past the last block);
 * w = exp(-2*pi*i*(l*Q mod L)/L). Forward only (t2cs has no backward twin). */
static inline size_t vfft_vtw512_groups_doubles(size_t ngrp, size_t G, int R)
{
    return ngrp * vfft_vtw512_groups(G) * (size_t)(R - 1) * VFFT_VTW512_REC;
}
static inline void vfft_vtw512_groups_fill(double *tw, size_t ngrp, size_t G, size_t nb,
                                           int R, size_t L, vfft_vtw512_q_fn q, const void *ctx)
{
    const size_t np = vfft_vtw512_groups(G), recs_blk = (size_t)(R - 1);
    size_t g, pp;
    int l, j;
    for (g = 0; g < ngrp; g++)
        for (pp = 0; pp < np; pp++)
            for (l = 1; l < R; l++) {
                double *rec = tw + ((g * np + pp) * recs_blk + (size_t)(l - 1)) * VFFT_VTW512_REC;
                for (j = 0; j < VFFT_VTW512_PER; j++) {
                    const size_t b2 = g * G + VFFT_VTW512_PER * pp + (size_t)j;
                    const size_t Q = (b2 < nb) ? (q ? q(ctx, b2) : b2) : 0;
                    double c, t;
                    _vtw512_w((long long)((size_t)l * Q % L), (long long)L, 0, &c, &t);
                    vfft_vtw512_pair_put(rec, j, c, t);
                }
            }
}

/* ── ZTURN-T: one stage's stream (ztt.h, _ztt_fill_stage) ─────────────────
 * Record (column block k/8, leg r >= 1) at ((k/8)*(R-1) + r-1)*16 =
 * [c(k..k+7)][s(k..k+7)] of w_RL^(r*b), b = k + lane; fwd s = -sin, bwd s =
 * +sin (plain sine, lane layout). L is a whole number of 8-column blocks (the
 * ZTURN-T law at this width). The values are ztt.h's: a 2^a*odd modulus
 * through vfft_cs2pi_exact, a pow2 modulus from the baked quarter wave (and
 * its fine table above the octave). Returns the doubles written, 2*(R-1)*L. */
#define _VTW512_QW_M   16384L   /* the quarter wave's full circle, ztt_qw16384.h */
#define _VTW512_QW_LGM 14
#define _VTW512_QW_Q   4096L
#define _VTW512_QW_LGQ 12
static inline double _vtw512_qsin(long u)
{
    const long q = u >> _VTW512_QW_LGQ;
    const long rem = u & (_VTW512_QW_Q - 1);
    const double v = (q & 1) ? VFFT_ZTT_QW16384_SIN[_VTW512_QW_Q - rem]
                             : VFFT_ZTT_QW16384_SIN[rem];
    return (q & 2) ? -v : v;
}
static inline size_t vfft_vtw512_ztt_stage(double *tw, long L, int R, long RL, int bwd)
{
    long k;
    int r, lane;
    if (RL & (RL - 1)) {
        for (k = 0; k < L; k += VFFT_VTW512_VW)
            for (r = 1; r < R; r++) {
                double *rec = tw + ((size_t)(k / VFFT_VTW512_VW) * (size_t)(R - 1)
                                    + (size_t)(r - 1)) * VFFT_VTW512_REC;
                for (lane = 0; lane < VFFT_VTW512_VW; lane++) {
                    const long b = k + lane;
                    long pw = ((long)r * b) % RL;
                    double c, sn;   /* a = 2*pi*pw/RL */
                    if (2 * pw > RL) pw -= RL;
                    vfft_cs2pi_exact((long long)pw, (long long)RL, &c, &sn);
                    rec[lane] = c;
                    rec[VFFT_VTW512_VW + lane] = bwd ? sn : -sn;
                }
            }
        return (size_t)2 * (size_t)(R - 1) * (size_t)L;
    }
    {
        int lg = 0, up, sh;
        double fc[64], fs[64];
        long bf;
        while ((1L << lg) < RL) lg++;
        up = lg > _VTW512_QW_LGM ? lg - _VTW512_QW_LGM : 0;
        sh = up ? 0 : _VTW512_QW_LGM - lg;
        if (up)
            for (bf = 0; bf < (1L << up); bf++)
                vfft_cs2pi_exact((long long)bf, (long long)RL, &fc[bf], &fs[bf]);
        for (k = 0; k < L; k += VFFT_VTW512_VW)
            for (r = 1; r < R; r++) {
                double *rec = tw + ((size_t)(k / VFFT_VTW512_VW) * (size_t)(R - 1)
                                    + (size_t)(r - 1)) * VFFT_VTW512_REC;
                for (lane = 0; lane < VFFT_VTW512_VW; lane++) {
                    const long b = k + lane;
                    const long pw = ((long)r * b) % RL;
                    double c, s;
                    if (!up) {
                        const long idx = pw << sh;
                        s = _vtw512_qsin(idx);
                        c = _vtw512_qsin((idx + _VTW512_QW_Q) & (_VTW512_QW_M - 1));
                    } else {   /* pw = a*2^up + bb: w = table(a) * fine(bb) */
                        const long a = pw >> up, bb = pw & ((1L << up) - 1);
                        const double sa = _vtw512_qsin(a);
                        const double ca = _vtw512_qsin((a + _VTW512_QW_Q) & (_VTW512_QW_M - 1));
                        c = ca * fc[bb] - sa * fs[bb];
                        s = sa * fc[bb] + ca * fs[bb];
                    }
                    rec[lane] = c;
                    rec[VFFT_VTW512_VW + lane] = bwd ? s : -s;
                }
            }
    }
    return (size_t)2 * (size_t)(R - 1) * (size_t)L;
}

#endif /* VFFT_VTW_AVX512_H */
