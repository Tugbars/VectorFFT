/**
 * @file  fft_radix5_scalar.h
 * @brief Radix-5 WFTA butterfly — scalar (portable C) implementation
 *
 * Provides the same 4-function API as AVX2/AVX-512 butterflies:
 *   radix5_wfta_fwd_scalar      — forward, twiddled (BLOCKED2)
 *   radix5_wfta_bwd_scalar      — backward, twiddled
 *   radix5_wfta_fwd_scalar_N1   — forward, no twiddles
 *   radix5_wfta_bwd_scalar_N1   — backward, no twiddles
 *
 * Used as the vtable fallback when SIMD is unavailable.
 */

#ifndef FFT_RADIX5_SCALAR_H
#define FFT_RADIX5_SCALAR_H

#include <math.h>

#ifndef R5_BUTTERFLY_API
#define R5_BUTTERFLY_API static inline __attribute__((always_inline))
#endif

/* ================================================================== */
/*  WFTA constants                                                     */
/* ================================================================== */
#ifndef R5_QA
#define R5_QA   (-0.25)
#define R5_QB    0.55901699437494742410
#define R5_SIN1  0.95105651629515357212
#define R5_SIN2  0.58778525229247312917
#endif

/* ================================================================== */
/*  Scalar complex multiply helpers                                    */
/* ================================================================== */

static inline void r5s_cmul(double ar, double ai, double wr, double wi,
                             double *tr, double *ti) {
    *tr = ar * wr - ai * wi;
    *ti = ar * wi + ai * wr;
}

static inline void r5s_cmulj(double ar, double ai, double wr, double wi,
                              double *tr, double *ti) {
    *tr = ar * wr + ai * wi;
    *ti = ai * wr - ar * wi;
}

/* ================================================================== */
/*  Scalar DFT-5 core — forward (-i rotation)                         */
/* ================================================================== */

static inline void r5s_core_fwd(
    double ar, double ai, double br, double bi,
    double cr, double ci, double dr, double di,
    double er, double ei,
    double *y0r, double *y0i, double *y1r, double *y1i,
    double *y2r, double *y2i, double *y3r, double *y3i,
    double *y4r, double *y4i)
{
    double s1r = br + er, s1i = bi + ei;
    double d1r = br - er, d1i = bi - ei;
    double s2r = cr + dr, s2i = ci + di;
    double d2r = cr - dr, d2i = ci - di;
    double Ar = s1r + s2r, Ai = s1i + s2i;
    *y0r = ar + Ar;  *y0i = ai + Ai;
    double Br = s1r - s2r, Bi = s1i - s2i;
    double comr = ar + R5_QA * Ar, comi = ai + R5_QA * Ai;
    double mbr  = R5_QB * Br,      mbi  = R5_QB * Bi;
    double t1r = comr + mbr, t1i = comi + mbi;
    double t2r = comr - mbr, t2i = comi - mbi;
    double v1r = R5_SIN1 * d1r + R5_SIN2 * d2r;
    double v1i = R5_SIN1 * d1i + R5_SIN2 * d2i;
    double v2r = R5_SIN2 * d1r - R5_SIN1 * d2r;
    double v2i = R5_SIN2 * d1i - R5_SIN1 * d2i;
    *y1r = t1r + v1i;  *y1i = t1i - v1r;
    *y4r = t1r - v1i;  *y4i = t1i + v1r;
    *y2r = t2r + v2i;  *y2i = t2i - v2r;
    *y3r = t2r - v2i;  *y3i = t2i + v2r;
}

/* ================================================================== */
/*  Scalar DFT-5 core — backward (+i rotation)                        */
/* ================================================================== */

static inline void r5s_core_bwd(
    double ar, double ai, double br, double bi,
    double cr, double ci, double dr, double di,
    double er, double ei,
    double *y0r, double *y0i, double *y1r, double *y1i,
    double *y2r, double *y2i, double *y3r, double *y3i,
    double *y4r, double *y4i)
{
    double s1r = br + er, s1i = bi + ei;
    double d1r = br - er, d1i = bi - ei;
    double s2r = cr + dr, s2i = ci + di;
    double d2r = cr - dr, d2i = ci - di;
    double Ar = s1r + s2r, Ai = s1i + s2i;
    *y0r = ar + Ar;  *y0i = ai + Ai;
    double Br = s1r - s2r, Bi = s1i - s2i;
    double comr = ar + R5_QA * Ar, comi = ai + R5_QA * Ai;
    double mbr  = R5_QB * Br,      mbi  = R5_QB * Bi;
    double t1r = comr + mbr, t1i = comi + mbi;
    double t2r = comr - mbr, t2i = comi - mbi;
    double v1r = R5_SIN1 * d1r + R5_SIN2 * d2r;
    double v1i = R5_SIN1 * d1i + R5_SIN2 * d2i;
    double v2r = R5_SIN2 * d1r - R5_SIN1 * d2r;
    double v2i = R5_SIN2 * d1i - R5_SIN1 * d2i;
    *y1r = t1r - v1i;  *y1i = t1i + v1r;
    *y4r = t1r + v1i;  *y4i = t1i - v1r;
    *y2r = t2r - v2i;  *y2i = t2i + v2r;
    *y3r = t2r + v2i;  *y3i = t2i - v2r;
}

/* ================================================================== */
/*  Forward N1 — scalar                                                */
/* ================================================================== */

R5_BUTTERFLY_API
void radix5_wfta_fwd_scalar_N1(
    const double *restrict a_re, const double *restrict a_im,
    const double *restrict b_re, const double *restrict b_im,
    const double *restrict c_re, const double *restrict c_im,
    const double *restrict d_re, const double *restrict d_im,
    const double *restrict e_re, const double *restrict e_im,
    double *restrict y0_re, double *restrict y0_im,
    double *restrict y1_re, double *restrict y1_im,
    double *restrict y2_re, double *restrict y2_im,
    double *restrict y3_re, double *restrict y3_im,
    double *restrict y4_re, double *restrict y4_im,
    int K)
{
    for (int k = 0; k < K; k++) {
        r5s_core_fwd(a_re[k], a_im[k], b_re[k], b_im[k],
                     c_re[k], c_im[k], d_re[k], d_im[k],
                     e_re[k], e_im[k],
                     &y0_re[k], &y0_im[k], &y1_re[k], &y1_im[k],
                     &y2_re[k], &y2_im[k], &y3_re[k], &y3_im[k],
                     &y4_re[k], &y4_im[k]);
    }
}

/* ================================================================== */
/*  Backward N1 — scalar                                               */
/* ================================================================== */

R5_BUTTERFLY_API
void radix5_wfta_bwd_scalar_N1(
    const double *restrict a_re, const double *restrict a_im,
    const double *restrict b_re, const double *restrict b_im,
    const double *restrict c_re, const double *restrict c_im,
    const double *restrict d_re, const double *restrict d_im,
    const double *restrict e_re, const double *restrict e_im,
    double *restrict y0_re, double *restrict y0_im,
    double *restrict y1_re, double *restrict y1_im,
    double *restrict y2_re, double *restrict y2_im,
    double *restrict y3_re, double *restrict y3_im,
    double *restrict y4_re, double *restrict y4_im,
    int K)
{
    for (int k = 0; k < K; k++) {
        r5s_core_bwd(a_re[k], a_im[k], b_re[k], b_im[k],
                     c_re[k], c_im[k], d_re[k], d_im[k],
                     e_re[k], e_im[k],
                     &y0_re[k], &y0_im[k], &y1_re[k], &y1_im[k],
                     &y2_re[k], &y2_im[k], &y3_re[k], &y3_im[k],
                     &y4_re[k], &y4_im[k]);
    }
}

/* ================================================================== */
/*  Forward twiddled — scalar, BLOCKED2                                */
/* ================================================================== */

R5_BUTTERFLY_API
void radix5_wfta_fwd_scalar(
    const double *restrict a_re, const double *restrict a_im,
    const double *restrict b_re, const double *restrict b_im,
    const double *restrict c_re, const double *restrict c_im,
    const double *restrict d_re, const double *restrict d_im,
    const double *restrict e_re, const double *restrict e_im,
    double *restrict y0_re, double *restrict y0_im,
    double *restrict y1_re, double *restrict y1_im,
    double *restrict y2_re, double *restrict y2_im,
    double *restrict y3_re, double *restrict y3_im,
    double *restrict y4_re, double *restrict y4_im,
    const double *restrict tw1_re, const double *restrict tw1_im,
    const double *restrict tw2_re, const double *restrict tw2_im,
    int K)
{
    for (int k = 0; k < K; k++) {
        double w3r, w3i, w4r, w4i;
        r5s_cmul(tw1_re[k], tw1_im[k], tw2_re[k], tw2_im[k], &w3r, &w3i);
        r5s_cmul(tw2_re[k], tw2_im[k], tw2_re[k], tw2_im[k], &w4r, &w4i);

        double tbr, tbi, tcr, tci, tdr, tdi, ter, tei;
        r5s_cmul(b_re[k], b_im[k], tw1_re[k], tw1_im[k], &tbr, &tbi);
        r5s_cmul(c_re[k], c_im[k], tw2_re[k], tw2_im[k], &tcr, &tci);
        r5s_cmul(d_re[k], d_im[k], w3r, w3i, &tdr, &tdi);
        r5s_cmul(e_re[k], e_im[k], w4r, w4i, &ter, &tei);

        r5s_core_fwd(a_re[k], a_im[k], tbr, tbi, tcr, tci,
                     tdr, tdi, ter, tei,
                     &y0_re[k], &y0_im[k], &y1_re[k], &y1_im[k],
                     &y2_re[k], &y2_im[k], &y3_re[k], &y3_im[k],
                     &y4_re[k], &y4_im[k]);
    }
}

/* ================================================================== */
/*  Backward twiddled — scalar, IDFT-5 → post-conj-twiddle             */
/* ================================================================== */

R5_BUTTERFLY_API
void radix5_wfta_bwd_scalar(
    const double *restrict a_re, const double *restrict a_im,
    const double *restrict b_re, const double *restrict b_im,
    const double *restrict c_re, const double *restrict c_im,
    const double *restrict d_re, const double *restrict d_im,
    const double *restrict e_re, const double *restrict e_im,
    double *restrict y0_re, double *restrict y0_im,
    double *restrict y1_re, double *restrict y1_im,
    double *restrict y2_re, double *restrict y2_im,
    double *restrict y3_re, double *restrict y3_im,
    double *restrict y4_re, double *restrict y4_im,
    const double *restrict tw1_re, const double *restrict tw1_im,
    const double *restrict tw2_re, const double *restrict tw2_im,
    int K)
{
    for (int k = 0; k < K; k++) {
        double r0r, r0i, r1r, r1i, r2r, r2i, r3r, r3i, r4r, r4i;
        r5s_core_bwd(a_re[k], a_im[k], b_re[k], b_im[k],
                     c_re[k], c_im[k], d_re[k], d_im[k],
                     e_re[k], e_im[k],
                     &r0r, &r0i, &r1r, &r1i, &r2r, &r2i,
                     &r3r, &r3i, &r4r, &r4i);

        double w3r, w3i, w4r, w4i;
        r5s_cmul(tw1_re[k], tw1_im[k], tw2_re[k], tw2_im[k], &w3r, &w3i);
        r5s_cmul(tw2_re[k], tw2_im[k], tw2_re[k], tw2_im[k], &w4r, &w4i);

        y0_re[k] = r0r;  y0_im[k] = r0i;
        r5s_cmulj(r1r, r1i, tw1_re[k], tw1_im[k], &y1_re[k], &y1_im[k]);
        r5s_cmulj(r2r, r2i, tw2_re[k], tw2_im[k], &y2_re[k], &y2_im[k]);
        r5s_cmulj(r3r, r3i, w3r, w3i, &y3_re[k], &y3_im[k]);
        r5s_cmulj(r4r, r4i, w4r, w4i, &y4_re[k], &y4_im[k]);
    }
}

#endif /* FFT_RADIX5_SCALAR_H */
