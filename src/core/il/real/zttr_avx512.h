/* zttr_avx512.h — ZTT-r's two fused kernels at 8 lanes (the avx512 build).
 *
 * Included by zttr.h at the avx512 ISA (VFFT_ZTTR_VW == 8), in place of its 4-lane kernels:
 * the plan, the job, the stack entry and the two runs are zttr.h's, and the
 * method is the one its header states. Here the plane is 128-byte
 * [re x8][im x8] blocks (the 8-lane ZTT's), a step takes eight columns, and
 * the twiddle records are the LANE records of isa/avx512/vtw_avx512.h,
 * [c x8][s x8] per leg.
 *
 *   tlfhc  the Hermitian terminator (r2c), R = 4 and 8: the primary oct
 *          (columns k .. k+7) in natural lanes; the partner window REVERSED
 *          as it is assembled, lanes 0..7 = columns L-k, L-k-1, .., L-k-7 =
 *          [B0 A7 A6 .. A1] from block A = (L-k-8 .. L-k-1) and the carried
 *          block B: one two-source permute per plane. The pass discipline is
 *          the 4-lane tlfhc's (pass 1 parks the even and the odd legs' half
 *          DFTs of both windows, pass 2 pairs leg pair m of the primary with
 *          pair R/2-1-m of the partner and untangles). Stores: X[f] and X[m]
 *          each through one two-source permute per zmm, re and im interleaved
 *          (X[m] with the reversal in its index). The run length L of the
 *          last stage is a multiple of 16.
 *   t0h    the fused backward ingest (c2r), R0 = 8 (the 8-lane ZTT's only
 *          natural ingest): four columns (k .. k+3) of every leg and their
 *          mirrors (Ls-k .. Ls-k-3) of the mirror leg per step, the mirror
 *          vector reversed whole by one 128-bit-lane shuffle; the untangle
 *          pointwise at the load edge, its conjugations inside the
 *          alternating add/subtract of a fused multiply by 1; two radix-8
 *          butterflies and the turned block stores of the emitted t0tp bwd
 *          (codelets/zil/avx512/ztt/radix8_z_t0tp_bwd_avx512.c). Sixteen legs
 *          stay in registers: nothing is parked. */
#ifndef VFFT_ZTTR_AVX512_H
#define VFFT_ZTTR_AVX512_H

/* a leg times its record [c x8][s x8] */
static inline __attribute__((always_inline)) void _zttr_twl(
    const __m512d xr, const __m512d xi, const double *rec, __m512d *yr, __m512d *yi)
{
    const __m512d c = _mm512_loadu_pd(rec), sn = _mm512_loadu_pd(rec + 8);
    *yr = _mm512_fnmadd_pd(xi, sn, _mm512_mul_pd(xr, c));
    *yi = _mm512_fmadd_pd(xr, sn, _mm512_mul_pd(xi, c));
}
/* the split DFT4 of (a0, a1, a2, a3): X0 = (a0+a2)+(a1+a3), X2 = (a0+a2)-(a1+a3),
 * X1 = (a0-a2) + (-i)(a1-a3), X3 = (a0-a2) - (-i)(a1-a3) */
static inline __attribute__((always_inline)) void _zttr_dft4(
    const __m512d *ar, const __m512d *ai, __m512d *xr, __m512d *xi)
{
    const __m512d s02r = _mm512_add_pd(ar[0], ar[2]), s02i = _mm512_add_pd(ai[0], ai[2]);
    const __m512d d02r = _mm512_sub_pd(ar[0], ar[2]), d02i = _mm512_sub_pd(ai[0], ai[2]);
    const __m512d s13r = _mm512_add_pd(ar[1], ar[3]), s13i = _mm512_add_pd(ai[1], ai[3]);
    const __m512d d13r = _mm512_sub_pd(ar[1], ar[3]), d13i = _mm512_sub_pd(ai[1], ai[3]);
    xr[0] = _mm512_add_pd(s02r, s13r); xi[0] = _mm512_add_pd(s02i, s13i);
    xr[2] = _mm512_sub_pd(s02r, s13r); xi[2] = _mm512_sub_pd(s02i, s13i);
    xr[1] = _mm512_add_pd(d02r, d13i); xi[1] = _mm512_sub_pd(d02i, d13r);
    xr[3] = _mm512_sub_pd(d02r, d13i); xi[3] = _mm512_add_pd(d02i, d13r);
}
/* W_R^m * (or + i oi) for R = 4, 8 (m < R/2), split form */
static inline __attribute__((always_inline)) void _zttr_wm(
    const int R, const int m, const __m512d orr, const __m512d oi, __m512d *tr, __m512d *ti)
{
    const __m512d h = _mm512_set1_pd(0.70710678118654757);
    if (m == 0) { *tr = orr; *ti = oi; return; }
    if (R == 4 || m == 2) { *tr = oi; *ti = _mm512_sub_pd(_mm512_setzero_pd(), orr); return; }   /* -i */
    if (m == 1) { *tr = _mm512_mul_pd(_mm512_add_pd(orr, oi), h); *ti = _mm512_mul_pd(_mm512_sub_pd(oi, orr), h); return; }
    /* m == 3: (-1 - i)/sqrt2 */
    *tr = _mm512_mul_pd(_mm512_sub_pd(oi, orr), h);
    *ti = _mm512_mul_pd(_mm512_sub_pd(_mm512_setzero_pd(), _mm512_add_pd(orr, oi)), h);
}
/* the half DFTs of one parity's legs x[0 .. R/2-1] */
static inline __attribute__((always_inline)) void _zttr_half(
    const int R, const __m512d *xr, const __m512d *xi, __m512d *Yr, __m512d *Yi)
{
    if (R == 4)
    {
        Yr[0] = _mm512_add_pd(xr[0], xr[1]); Yi[0] = _mm512_add_pd(xi[0], xi[1]);
        Yr[1] = _mm512_sub_pd(xr[0], xr[1]); Yi[1] = _mm512_sub_pd(xi[0], xi[1]);
    }
    else _zttr_dft4(xr, xi, Yr, Yi);
}

/* ── the terminator (tlfhc), one body per radix ── */
#define ZTTR_TLFHC_BODY(RR)                                                                               \
static __attribute__((noinline)) void _zttr_tlfhc##RR(const _zttr_job_t *jb, const double *W, double *X)  \
{                                                                                                         \
    enum { R = RR, H = RR / 2 };                                                                          \
    const vfft_zttr_plan_t *const p = jb->p;                                                              \
    const long L = p->L;                                                                                  \
    const long c0 = jb->lo ? L - jb->lo : 0;            /* the range's first mirror column (column L is column 0) */\
    const double *const ein = jb->edge;                 /* a range's last partner block, from the snapshot */\
    const size_t M = (size_t)p->M;                                                                        \
    const double *affS = p->affS3, *affC = p->affC3;                                                      \
    double *const cen = jb->cen;                                                                          \
    double raw[48 * R + 8];                             /* the parking and the carry, aligned by hand */  \
    double *const SP = (double *)(((uintptr_t)raw + 63u) & ~(uintptr_t)63u);                              \
    double *const SW = SP + 16 * R;                                                                       \
    double *const CR = SW + 16 * R;                                                                       \
    const __m512i WIN = _mm512_setr_epi64(8, 7, 6, 5, 4, 3, 2, 1);          /* [B0 A7 A6 A5 A4 A3 A2 A1] */\
    const __m512i FLO = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);        /* (re, im) of the lanes 0..3 */\
    const __m512i FHI = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);      /*             the lanes 4..7 */\
    const __m512i MLO = _mm512_setr_epi64(7, 15, 6, 14, 5, 13, 4, 12);      /* the lanes 7..4, reversed   */\
    const __m512i MHI = _mm512_setr_epi64(3, 11, 2, 10, 1, 9, 0, 8);        /* the lanes 3..0             */\
    const int pf = (W != X);        /* the plane mode: X went cold under W, prefetch its lines */        \
    if (jb->mode & 1) { /* the centre column L/2: lane 0 of the oct k = L/2, its partner lane 0 of leg R-1-r */\
        const double *twr = p->twr3 + (size_t)(L / 16) * (size_t)(R - 1) * 16u;                           \
        __m512d Er[H], Ei[H], Or[H], Oi[H], Pr[R], Pi[R];                                                 \
        ZTTR_UNROLL for (int par = 0; par < 2; par++)                                                     \
        {                                                                                                 \
            __m512d xr[H], xi[H];                                                                         \
            ZTTR_UNROLL for (int j = 0; j < H; j++)                                                       \
            {                                                                                             \
                const int r = 2 * j + par;                                                                \
                const double *a = W + 2 * ((size_t)r * (size_t)L + (size_t)(L / 2));                      \
                __m512d wr = _mm512_loadu_pd(a), wi = _mm512_loadu_pd(a + 8);                             \
                if (r) _zttr_twl(wr, wi, twr + (r - 1) * 16, &wr, &wi);                                   \
                xr[j] = wr; xi[j] = wi;                                                                   \
            }                                                                                             \
            if (par) _zttr_half(R, xr, xi, Or, Oi); else _zttr_half(R, xr, xi, Er, Ei);                   \
        }                                                                                                 \
        ZTTR_UNROLL for (int m = 0; m < H; m++)                                                           \
        {                                                                                                 \
            __m512d tr, ti;                                                                               \
            _zttr_wm(R, m, Or[m], Oi[m], &tr, &ti);                                                       \
            Pr[m] = _mm512_add_pd(Er[m], tr);     Pi[m] = _mm512_add_pd(Ei[m], ti);                       \
            Pr[m + H] = _mm512_sub_pd(Er[m], tr); Pi[m + H] = _mm512_sub_pd(Ei[m], ti);                   \
        }                                                                                                 \
        ZTTR_UNROLL for (int r = 0; r < R; r++)                                                           \
        {                                                                                                 \
            const int rp = R - 1 - r;                                                                     \
            const size_t f = (size_t)r * (size_t)L + (size_t)(L / 2);                                     \
            const double S = affS[f], C = affC[f];                                                        \
            const double Ar = _mm512_cvtsd_f64(Pr[r]), Ai = _mm512_cvtsd_f64(Pi[r]);                      \
            const double Br = _mm512_cvtsd_f64(Pr[rp]), Bi = _mm512_cvtsd_f64(Pi[rp]);                    \
            const double t1 = Ar - Br, t2 = Ai + Bi;                                                      \
            const double xr = S * t1 + C * t2, xi = S * t2 - C * t1;                                      \
            cen[2 * r] = Br + xr;                                                                         \
            cen[2 * r + 1] = xi - Bi;                                                                     \
        }                                                                                                 \
    }                                                                                                     \
    ZTTR_UNROLL for (int r = 0; r < R; r++)                                                               \
    {   /* the carry's start: lane 0 of the block at the first mirror column */                           \
        if (jb->carry)                                                                                    \
        {                                                                                                 \
            _mm512_store_pd(CR + 16 * r, _mm512_loadu_pd(jb->carry + 16 * r));                            \
            _mm512_store_pd(CR + 16 * r + 8, _mm512_loadu_pd(jb->carry + 16 * r + 8));                    \
        }                                                                                                 \
        else                                                                                              \
        {                                                                                                 \
            const double *a = W + 2 * ((size_t)r * (size_t)L + (size_t)c0);                               \
            _mm512_store_pd(CR + 16 * r, _mm512_zextpd128_pd512(_mm_load_sd(a)));                         \
            _mm512_store_pd(CR + 16 * r + 8, _mm512_zextpd128_pd512(_mm_load_sd(a + 8)));                 \
        }                                                                                                 \
    }                                                                                                     \
    for (long k = jb->lo; k < jb->hi; k += 8)                                                             \
    {                                                                                                     \
        const long q = k / 8;                                                                             \
        const int edge = (k + 8 == jb->hi) && ein;                                                        \
        const double *twr = p->twr3 + (size_t)q * (size_t)(R - 1) * 16u;                                  \
        const double *twp = p->twp3 + (size_t)q * (size_t)(R - 1) * 16u;                                  \
        if (pf)                                                                                           \
            ZTTR_UNROLL for (int r = 0; r < R; r++)                                                       \
            {   /* four windows ahead: the primary's two lines, the partner's two */                      \
                const double *xf = X + 2 * ((ptrdiff_t)r * L + (k + 32));                                 \
                const double *xm = X + 2 * ((ptrdiff_t)r * L + (L - k - 40));                             \
                _mm_prefetch((const char *)xf, _MM_HINT_T0);                                              \
                _mm_prefetch((const char *)(xf + 8), _MM_HINT_T0);                                        \
                _mm_prefetch((const char *)xm, _MM_HINT_T0);                                              \
                _mm_prefetch((const char *)(xm + 8), _MM_HINT_T0);                                        \
            }                                                                                             \
        /* pass 1, the partner window reversed [B0 A7 .. A1]: the even legs' half DFT parked, then the odd legs' */\
        ZTTR_UNROLL for (int par = 0; par < 2; par++)                                                     \
        {                                                                                                 \
            __m512d xr[H], xi[H], Yr[H], Yi[H];                                                           \
            ZTTR_UNROLL for (int j = 0; j < H; j++)                                                       \
            {                                                                                             \
                const int r = 2 * j + par;                                                                \
                const double *a = W + 2 * ((size_t)r * (size_t)L + (size_t)(L - k - 8));                  \
                const double *as = edge ? ein + 16 * r : a;                                               \
                const __m512d ar = _mm512_loadu_pd(as), ai = _mm512_loadu_pd(as + 8);                     \
                __m512d wr = _mm512_permutex2var_pd(ar, WIN, _mm512_load_pd(CR + 16 * r));                \
                __m512d wi = _mm512_permutex2var_pd(ai, WIN, _mm512_load_pd(CR + 16 * r + 8));            \
                _mm512_store_pd(CR + 16 * r, ar); _mm512_store_pd(CR + 16 * r + 8, ai);                   \
                if (r) _zttr_twl(wr, wi, twp + (r - 1) * 16, &wr, &wi);                                   \
                xr[j] = wr; xi[j] = wi;                                                                   \
            }                                                                                             \
            _zttr_half(R, xr, xi, Yr, Yi);                                                                \
            ZTTR_UNROLL for (int m = 0; m < H; m++)                                                       \
            {                                                                                             \
                _mm512_store_pd(SW + 32 * m + 16 * par, Yr[m]);                                           \
                _mm512_store_pd(SW + 32 * m + 16 * par + 8, Yi[m]);                                       \
            }                                                                                             \
        }                                                                                                 \
        /* pass 1, the primary oct, natural lanes: the even legs' half DFT parked, then the odd legs' */ \
        ZTTR_UNROLL for (int par = 0; par < 2; par++)                                                     \
        {                                                                                                 \
            __m512d xr[H], xi[H], Yr[H], Yi[H];                                                           \
            ZTTR_UNROLL for (int j = 0; j < H; j++)                                                       \
            {                                                                                             \
                const int r = 2 * j + par;                                                                \
                const double *a = W + 2 * ((size_t)r * (size_t)L + (size_t)k);                            \
                __m512d wr = _mm512_loadu_pd(a), wi = _mm512_loadu_pd(a + 8);                             \
                if (r) _zttr_twl(wr, wi, twr + (r - 1) * 16, &wr, &wi);                                   \
                xr[j] = wr; xi[j] = wi;                                                                   \
            }                                                                                             \
            _zttr_half(R, xr, xi, Yr, Yi);                                                                \
            ZTTR_UNROLL for (int m = 0; m < H; m++)                                                       \
            {                                                                                             \
                _mm512_store_pd(SP + 32 * m + 16 * par, Yr[m]);                                           \
                _mm512_store_pd(SP + 32 * m + 16 * par + 8, Yi[m]);                                       \
            }                                                                                             \
        }                                                                                                 \
        /* pass 2: pair m of the primary with pair R/2-1-m of the partner */                              \
        ZTTR_UNROLL for (int m = 0; m < H; m++)                                                           \
        {                                                                                                 \
            const int mp = H - 1 - m;                                                                     \
            __m512d tr, ti, ur, ui;                                                                       \
            _zttr_wm(R, m, _mm512_load_pd(SP + 32 * m + 16), _mm512_load_pd(SP + 32 * m + 24), &tr, &ti); \
            _zttr_wm(R, mp, _mm512_load_pd(SW + 32 * mp + 16), _mm512_load_pd(SW + 32 * mp + 24), &ur, &ui); \
            const __m512d EPr = _mm512_load_pd(SP + 32 * m), EPi = _mm512_load_pd(SP + 32 * m + 8);       \
            const __m512d EWr = _mm512_load_pd(SW + 32 * mp), EWi = _mm512_load_pd(SW + 32 * mp + 8);     \
            const __m512d P0r = _mm512_add_pd(EPr, tr), P0i = _mm512_add_pd(EPi, ti);   /* leg m       */ \
            const __m512d P1r = _mm512_sub_pd(EPr, tr), P1i = _mm512_sub_pd(EPi, ti);   /* leg m + R/2 */ \
            const __m512d W0r = _mm512_add_pd(EWr, ur), W0i = _mm512_add_pd(EWi, ui);   /* leg mp      */ \
            const __m512d W1r = _mm512_sub_pd(EWr, ur), W1i = _mm512_sub_pd(EWi, ui);   /* leg mp+R/2  */ \
            ZTTR_UNROLL for (int half = 0; half < 2; half++)                                              \
            {                                                                                             \
                const int r = half ? m + H : m;                                                           \
                const __m512d Ar = half ? P1r : P0r, Ai = half ? P1i : P0i;                               \
                const __m512d Br = half ? W0r : W1r, Bi = half ? W0i : W1i;                               \
                const size_t f0 = (size_t)r * (size_t)L + (size_t)k;                                      \
                const __m512d S = _mm512_loadu_pd(affS + f0), C = _mm512_loadu_pd(affC + f0);             \
                const __m512d t1 = _mm512_sub_pd(Ar, Br), t2 = _mm512_add_pd(Ai, Bi);                     \
                const __m512d xr = _mm512_fmadd_pd(S, t1, _mm512_mul_pd(C, t2));                          \
                const __m512d xi = _mm512_fmsub_pd(S, t2, _mm512_mul_pd(C, t1));                          \
                const __m512d Xfr = _mm512_add_pd(Br, xr), Xfi = _mm512_sub_pd(xi, Bi);                   \
                const __m512d Xmr = _mm512_sub_pd(Ar, xr), Xmi = _mm512_sub_pd(xi, Ai);                   \
                double *of = X + 2 * f0;                                                                  \
                double *om = X + 2 * (M - f0 - 7);                                                        \
                _mm512_storeu_pd(of, _mm512_permutex2var_pd(Xfr, FLO, Xfi));                              \
                _mm512_storeu_pd(of + 8, _mm512_permutex2var_pd(Xfr, FHI, Xfi));                          \
                _mm512_storeu_pd(om, _mm512_permutex2var_pd(Xmr, MLO, Xmi));                              \
                _mm512_storeu_pd(om + 8, _mm512_permutex2var_pd(Xmr, MHI, Xmi));                          \
            }                                                                                             \
        }                                                                                                 \
    }                                                                                                     \
    if (jb->mode & 2)                                                                                     \
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

/* ── the fused backward ingest (t0h), R0 = 8 ── */

/* the untangle of four pairs (X[n+i], X[M-n-i]): F = X[n ..], Mv = the mirrors in F's order, unconjugated;
 * t = F - conj(M), e = F + conj(M), cy = t (sin, -cos); Zhat[n+i] = e - cy, Zhat[M-n-i] = conj(e + cy) */
static inline __attribute__((always_inline)) void _zttr_unt_bwd(
    const __m512d F, const __m512d Mv, const __m512d wr, const __m512d wi, __m512d *zf, __m512d *zm)
{
    const __m512d ONE = _mm512_set1_pd(1.0);
    const __m512d PM = _mm512_setr_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
    const __m512d t = _mm512_fmaddsub_pd(F, ONE, Mv), e = _mm512_fmsubadd_pd(F, ONE, Mv);
    const __m512d cy = _mm512_fmaddsub_pd(wr, t, _mm512_mul_pd(wi, _mm512_permute_pd(t, 0x55)));
    *zf = _mm512_sub_pd(e, cy);
    *zm = _mm512_fmsubadd_pd(e, PM, cy);
}
/* the backward radix-8 butterfly on interleaved four-column vectors, as the emitted t0tp bwd (conjugate
 * roots: the rotation is +i); E[m] = output 2m, O[m] = output 2m+1 */
static inline __attribute__((always_inline)) void _zttr_bf8b(const __m512d *a, __m512d *E, __m512d *O)
{
    const __m512d pim = _mm512_setr_pd(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0);
    const __m512d rh = _mm512_set1_pd(0.70710678118654752440);
    const __m512d b0 = _mm512_add_pd(a[0], a[4]), d0 = _mm512_sub_pd(a[0], a[4]);
    const __m512d b1 = _mm512_add_pd(a[1], a[5]), d1 = _mm512_sub_pd(a[1], a[5]);
    const __m512d b2 = _mm512_add_pd(a[2], a[6]), d2 = _mm512_sub_pd(a[2], a[6]);
    const __m512d b3 = _mm512_add_pd(a[3], a[7]), d3 = _mm512_sub_pd(a[3], a[7]);
    const __m512d c0 = d0;
    const __m512d d1i = _mm512_xor_pd(_mm512_permute_pd(d1, 0x55), pim);
    const __m512d c1 = _mm512_mul_pd(_mm512_add_pd(d1, d1i), rh);
    const __m512d c2 = _mm512_xor_pd(_mm512_permute_pd(d2, 0x55), pim);
    const __m512d d3i = _mm512_xor_pd(_mm512_permute_pd(d3, 0x55), pim);
    const __m512d c3 = _mm512_mul_pd(_mm512_sub_pd(d3i, d3), rh);
    const __m512d e0 = _mm512_add_pd(b0, b2), e1 = _mm512_sub_pd(b0, b2);
    const __m512d e2 = _mm512_add_pd(b1, b3), e3 = _mm512_sub_pd(b1, b3);
    const __m512d er = _mm512_xor_pd(_mm512_permute_pd(e3, 0x55), pim);
    const __m512d o0 = _mm512_add_pd(c0, c2), o1 = _mm512_sub_pd(c0, c2);
    const __m512d o2 = _mm512_add_pd(c1, c3), o3 = _mm512_sub_pd(c1, c3);
    const __m512d orr = _mm512_xor_pd(_mm512_permute_pd(o3, 0x55), pim);
    E[0] = _mm512_add_pd(e0, e2); E[2] = _mm512_sub_pd(e0, e2);
    E[1] = _mm512_add_pd(e1, er); E[3] = _mm512_sub_pd(e1, er);
    O[0] = _mm512_add_pd(o0, o2); O[2] = _mm512_sub_pd(o0, o2);
    O[1] = _mm512_add_pd(o1, orr); O[3] = _mm512_sub_pd(o1, orr);
}
/* the run stores: the eight outputs of the four columns in the vectors' slots 0..3 -> one [re x8][im x8]
 * block per column (p0..p3; NULL skips a column) */
static inline __attribute__((always_inline)) void _zttr_blk_store(
    const __m512d *E, const __m512d *O, double *p0, double *p1, double *p2, double *p3)
{
    const __m512i tlo = _mm512_set_epi64(13, 12, 5, 4, 9, 8, 1, 0);
    const __m512i thi = _mm512_set_epi64(15, 14, 7, 6, 11, 10, 3, 2);
    const __m512d t0 = _mm512_unpacklo_pd(E[0], O[0]), t1 = _mm512_unpackhi_pd(E[0], O[0]);
    const __m512d t2 = _mm512_unpacklo_pd(E[1], O[1]), t3 = _mm512_unpackhi_pd(E[1], O[1]);
    const __m512d t4 = _mm512_unpacklo_pd(E[2], O[2]), t5 = _mm512_unpackhi_pd(E[2], O[2]);
    const __m512d t6 = _mm512_unpacklo_pd(E[3], O[3]), t7 = _mm512_unpackhi_pd(E[3], O[3]);
    const __m512d x0 = _mm512_permutex2var_pd(t0, tlo, t2), x1 = _mm512_permutex2var_pd(t1, tlo, t3);
    const __m512d x2 = _mm512_permutex2var_pd(t0, thi, t2), x3 = _mm512_permutex2var_pd(t1, thi, t3);
    const __m512d x4 = _mm512_permutex2var_pd(t4, tlo, t6), x5 = _mm512_permutex2var_pd(t5, tlo, t7);
    const __m512d x6 = _mm512_permutex2var_pd(t4, thi, t6), x7 = _mm512_permutex2var_pd(t5, thi, t7);
    if (p0) { _mm512_storeu_pd(p0, _mm512_shuffle_f64x2(x0, x4, 0x44)); _mm512_storeu_pd(p0 + 8, _mm512_shuffle_f64x2(x1, x5, 0x44)); }
    if (p1) { _mm512_storeu_pd(p1, _mm512_shuffle_f64x2(x2, x6, 0x44)); _mm512_storeu_pd(p1 + 8, _mm512_shuffle_f64x2(x3, x7, 0x44)); }
    if (p2) { _mm512_storeu_pd(p2, _mm512_shuffle_f64x2(x0, x4, 0xEE)); _mm512_storeu_pd(p2 + 8, _mm512_shuffle_f64x2(x1, x5, 0xEE)); }
    if (p3) { _mm512_storeu_pd(p3, _mm512_shuffle_f64x2(x2, x6, 0xEE)); _mm512_storeu_pd(p3 + 8, _mm512_shuffle_f64x2(x3, x7, 0xEE)); }
}
/* the legs of the column quad at k: z[j] = Zhat of (leg j, columns k .. k+3), m[j] = Zhat of (leg j, columns
 * Ls-k, Ls-k-1, Ls-k-2, Ls-k-3) */
#define ZTTR_T0H_LEGS(k)                                                                                  \
    ZTTR_UNROLL for (int j = 0; j < R; j++)                                                               \
    {                                                                                                     \
        const size_t n = (size_t)j * Ls + (k), jp = (size_t)(R - 1 - j);                                  \
        __m512d F = _mm512_loadu_pd(X + 2 * n);                                                           \
        __m512d Mv = _mm512_loadu_pd(X + 2 * (jp * Ls + Ls - (k) - 3));                                   \
        Mv = _mm512_shuffle_f64x2(Mv, Mv, 0x1B);             /* [X[M-n] X[M-n-1] X[M-n-2] X[M-n-3]] */     \
        if ((k) == 0 && j == 0) { F = _mm512_maskz_mov_pd(0xFD, F); Mv = _mm512_maskz_mov_pd(0xFD, Mv); } /* X[0], X[M] real */\
        _zttr_unt_bwd(F, Mv, _mm512_loadu_pd(SB + 2 * n), _mm512_loadu_pd(CB + 2 * n), &z[j], &m[R - 1 - j]); \
    }
static __attribute__((noinline)) void _zttr_t0h8(const _zttr_job_t *jb, const double *X, double *W)
{
    enum { R = 8, BLK = 16 };                         /* doubles per column run */
    const vfft_zttr_plan_t *const p = jb->p;
    const size_t Ls = (size_t)p->zt->ncol;
    const size_t *rb = p->zt->rb;
    const double *SB = p->bS, *CB = p->bC;
    for (size_t k = (size_t)jb->lo; k < (size_t)jb->hi; k += 4)
    {
        __m512d z[R], m[R], E[4], O[4];
        ZTTR_T0H_LEGS(k)
        _zttr_bf8b(z, E, O);
        _zttr_blk_store(E, O, W + BLK * rb[k], W + BLK * rb[k + 1], W + BLK * rb[k + 2], W + BLK * rb[k + 3]);
        _zttr_bf8b(m, E, O);
        _zttr_blk_store(E, O, k ? W + BLK * rb[Ls - k] : NULL, W + BLK * rb[Ls - k - 1], W + BLK * rb[Ls - k - 2],
                        W + BLK * rb[Ls - k - 3]);
    }
    if (jb->mode & 2)
    {   /* the centre column Ls/2: its partner is column Ls/2 of the mirror leg; the columns after it in the
         * vector were stored by the loop as mirrors and are left alone */
        __m512d z[R], m[R], E[4], O[4];
        const size_t k = Ls / 2;
        ZTTR_T0H_LEGS(k)
        _zttr_bf8b(z, E, O);
        _zttr_blk_store(E, O, W + BLK * rb[k], NULL, NULL, NULL);
        (void)m;
    }
}
#undef ZTTR_T0H_LEGS

#endif /* VFFT_ZTTR_AVX512_H */
