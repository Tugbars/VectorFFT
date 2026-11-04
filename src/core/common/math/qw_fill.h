/* qw_fill.h — the ZTURN-T twiddle streams expanded from the BAKED QUARTER-WAVE
 * (ztt_qw16384.h), for either library (common/: split cannot include il/).
 *
 * A stage of the run-contiguous DIT combines R adjacent runs of length L with
 * the column-varying pre-twiddle w_{RL}^{r*b}, b the column inside the run,
 * r = 1..R-1 the leg. Two record shapes carry it, both 2*(R-1)*L doubles per
 * stage, both the SAME angles:
 *   quad : for (column block k, leg r) the record [c x LANES][s x LANES] at
 *          tw + ((k/LANES)*(R-1) + (r-1)) * 2*LANES — ZTURN-T's own stream
 *          (il/rank1/ztt.h) and the split K=1 engine's (split/rank1/zts.h);
 *   pair : for (column b, leg r) the pair [c, s] at tw + 2*((R-1)*b + (r-1)),
 *          broadcast by the lane-major kernels (one twiddle serves every lane).
 * fwd s = -sin (c + i s = exp(-i th)), bwd s = +sin. Up to the table's octave
 * (RL <= 16384) every value is a table entry (index shift + quadrant
 * reflection + sign flip, no trig); above it the two-level product
 * table(a) * fine(b), one small cos/sin table per stage; a 2^a*odd modulus
 * goes through vfft_cs2pi_exact. The -0.0 at r*b == 0 falls out of the sign
 * flip, as ZTURN-T's record contract wants. These are ztt.h's own builders,
 * shared; ztt.h keeps its copies until the IL side adopts this header. */
#ifndef VFFT_QW_FILL_H
#define VFFT_QW_FILL_H

#include <stddef.h>
#include "ztt_qw16384.h"   /* VFFT_ZTT_QW16384_SIN[0..4096]: sin(2*pi*k/16384) */
#include "tw_exact.h"      /* vfft_cs2pi_exact: once-rounded cos/sin(2*pi*p/n) */

#define VFFT_QW_M   16384L   /* the full circle of the baked table            */
#define VFFT_QW_LGM 14
#define VFFT_QW_Q   4096L    /* a quarter                                     */
#define VFFT_QW_LGQ 12

/* sin(2*pi*u/M) from the quarter wave, u in [0, M): quadrant q = u / Q,
 * rem = u % Q: q=0 T[rem], q=1 T[Q-rem] (mirror about pi/2), q>=2 negated */
static inline double vfft_qw_sin(long u)
{
    const long q = u >> VFFT_QW_LGQ;
    const long rem = u & (VFFT_QW_Q - 1);
    const double v = (q & 1) ? VFFT_ZTT_QW16384_SIN[VFFT_QW_Q - rem]
                             : VFFT_ZTT_QW16384_SIN[rem];
    return (q & 2) ? -v : v;
}

static inline int _vfft_qw_log2(long v)
{
    int k = 0;
    while ((1L << k) < v) k++;
    return k;
}

/* cos and sin of 2*pi*pw/RL for one record entry: the octave path, the
 * two-level product above it, vfft_cs2pi_exact for a 2^a*odd modulus (whose
 * fine tables fc/fs the caller built when `up` > 0 and RL is a power of two) */
static inline void _vfft_qw_cs(long pw, long RL, int pow2, int up, int sh,
                               const double *fc, const double *fs, double *c, double *s)
{
    if (!pow2)
    {
        double cc, ss;
        if (2 * pw > RL) pw -= RL;
        vfft_cs2pi_exact((long long)pw, (long long)RL, &cc, &ss);
        *c = cc;
        *s = ss;
    }
    else if (!up)
    {
        const long idx = pw << sh;
        *s = vfft_qw_sin(idx);
        *c = vfft_qw_sin((idx + VFFT_QW_Q) & (VFFT_QW_M - 1));
    }
    else
    {   /* pw = a*2^up + bb: w = table(a) * fine(bb) */
        const long a = pw >> up, bb = pw & ((1L << up) - 1);
        const double sa = vfft_qw_sin(a);
        const double ca = vfft_qw_sin((a + VFFT_QW_Q) & (VFFT_QW_M - 1));
        *c = ca * fc[bb] - sa * fs[bb];
        *s = sa * fc[bb] + ca * fs[bb];
    }
}

/* the per-stage preamble shared by both shapes: pow2 / octave / fine table */
#define _VFFT_QW_STAGE_SETUP(RL, pow2, up, sh, fc, fs)                                  \
    const int pow2 = ((RL) & ((RL) - 1)) == 0;                                          \
    const int lg_ = pow2 ? _vfft_qw_log2(RL) : 0;                                       \
    const int up = (pow2 && lg_ > VFFT_QW_LGM) ? lg_ - VFFT_QW_LGM : 0;                 \
    const int sh = (pow2 && !up) ? VFFT_QW_LGM - lg_ : 0;                               \
    double fc[64], fs[64];                                                              \
    if (up)                                                                             \
        for (long bf_ = 0; bf_ < (1L << up); bf_++)                                     \
            vfft_cs2pi_exact((long long)bf_, (long long)(RL), &fc[bf_], &fs[bf_]);

/* quad records, LANES columns per block (4 at AVX2, 8 at AVX-512); returns the
 * doubles written = 2*(R-1)*L */
static inline size_t vfft_qw_fill_stage_quad(double *tw, long L, int R, long RL, int bwd, int lanes)
{
    _VFFT_QW_STAGE_SETUP(RL, pow2, up, sh, fc, fs)
    long k;
    int r, lane;
    for (k = 0; k < L; k += lanes)
        for (r = 1; r < R; r++)
        {
            double *rec = tw + ((size_t)(k / lanes) * (size_t)(R - 1) + (size_t)(r - 1)) * (size_t)(2 * lanes);
            for (lane = 0; lane < lanes; lane++)
            {
                const long b = k + lane;
                double c, s;
                _vfft_qw_cs(((long)r * b) % RL, RL, pow2, up, sh, fc, fs, &c, &s);
                rec[lane] = c;
                rec[lanes + lane] = bwd ? s : -s;
            }
        }
    return (size_t)2 * (size_t)(R - 1) * (size_t)L;
}

/* (c, s) pairs per (column, leg); returns the doubles written = 2*(R-1)*L */
static inline size_t vfft_qw_fill_stage_pair(double *tw, long L, int R, long RL, int bwd)
{
    _VFFT_QW_STAGE_SETUP(RL, pow2, up, sh, fc, fs)
    long b;
    int r;
    for (b = 0; b < L; b++)
        for (r = 1; r < R; r++)
        {
            double *rec = tw + 2 * ((size_t)(R - 1) * (size_t)b + (size_t)(r - 1));
            double c, s;
            _vfft_qw_cs(((long)r * b) % RL, RL, pow2, up, sh, fc, fs, &c, &s);
            rec[0] = c;
            rec[1] = bwd ? s : -s;
        }
    return (size_t)2 * (size_t)(R - 1) * (size_t)L;
}

#undef _VFFT_QW_STAGE_SETUP

/* mixed-radix digit reversal over R[0..nR-1]: n = d0 + R0*(d1 + ...), d0
 * fastest; digit d_s carries weight prod_{u>s} R[u] (the run-base table:
 * rb[digitrev(j, chain+1, nf-1)] = j) */
static inline long vfft_qw_digitrev(long n, const int *R, int nR)
{
    long out = 0;
    int s, u;
    for (s = 0; s < nR; s++)
    {
        const int d = (int)(n % R[s]);
        long wt = 1;
        n /= R[s];
        for (u = s + 1; u < nR; u++) wt *= R[u];
        out += (long)d * wt;
    }
    return out;
}

#endif /* VFFT_QW_FILL_H */
