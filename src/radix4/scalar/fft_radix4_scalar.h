/**
 * @file fft_radix4_scalar_optimized.h
 * @brief Production-Grade Scalar Radix-4 Implementation
 *
 * @author VectorFFT Team
 * @version 2.1 (Scalar Optimized)
 * @date 2025
 */

#ifndef FFT_RADIX4_SCALAR_OPTIMIZED_H
#define FFT_RADIX4_SCALAR_OPTIMIZED_H

#include "fft_radix4.h"
/* simd_math.h — not needed for scalar path */
#include <stdint.h>

#ifdef _MSC_VER
#define FORCE_INLINE_SC static __forceinline
#define RESTRICT_SC __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE_SC static inline __attribute__((always_inline))
#define RESTRICT_SC __restrict__
#else
#define FORCE_INLINE_SC static inline
#define RESTRICT_SC
#endif

#ifndef RADIX4_PREFETCH_DISTANCE_SCALAR
#define RADIX4_PREFETCH_DISTANCE_SCALAR 32
#endif

#ifndef RADIX4_DERIVE_W3_SCALAR
#define RADIX4_DERIVE_W3_SCALAR 0
#endif

#if defined(__GNUC__) || defined(__clang__)
#define PREFETCH_SCALAR(ptr) __builtin_prefetch((const void *)(ptr), 0, 0)
#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
#include <emmintrin.h>
#define PREFETCH_SCALAR(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T0)
#else
#define PREFETCH_SCALAR(ptr) ((void)0)
#endif

FORCE_INLINE_SC void cmul_scalar(
    double ar, double ai, double wr, double wi,
    double *RESTRICT_SC tr, double *RESTRICT_SC ti)
{
    *tr = ar * wr - ai * wi;
    *ti = ar * wi + ai * wr;
}

FORCE_INLINE_SC void radix4_butterfly_core_fv_scalar(
    double a_re, double a_im,
    double tB_re, double tB_im, double tC_re, double tC_im, double tD_re, double tD_im,
    double *RESTRICT_SC y0_re, double *RESTRICT_SC y0_im,
    double *RESTRICT_SC y1_re, double *RESTRICT_SC y1_im,
    double *RESTRICT_SC y2_re, double *RESTRICT_SC y2_im,
    double *RESTRICT_SC y3_re, double *RESTRICT_SC y3_im)
{
    double sumBD_re = tB_re + tD_re, sumBD_im = tB_im + tD_im;
    double difBD_re = tB_re - tD_re, difBD_im = tB_im - tD_im;
    double sumAC_re = a_re + tC_re, sumAC_im = a_im + tC_im;
    double difAC_re = a_re - tC_re, difAC_im = a_im - tC_im;

    /* Forward: rot = (+i) * difBD = (-difBD_im, +difBD_re) */
    double rot_re = -difBD_im, rot_im = difBD_re;

    *y0_re = sumAC_re + sumBD_re; *y0_im = sumAC_im + sumBD_im;
    *y1_re = difAC_re - rot_re;   *y1_im = difAC_im - rot_im;
    *y2_re = sumAC_re - sumBD_re; *y2_im = sumAC_im - sumBD_im;
    *y3_re = difAC_re + rot_re;   *y3_im = difAC_im + rot_im;
}

FORCE_INLINE_SC void radix4_butterfly_core_bv_scalar(
    double a_re, double a_im,
    double tB_re, double tB_im, double tC_re, double tC_im, double tD_re, double tD_im,
    double *RESTRICT_SC y0_re, double *RESTRICT_SC y0_im,
    double *RESTRICT_SC y1_re, double *RESTRICT_SC y1_im,
    double *RESTRICT_SC y2_re, double *RESTRICT_SC y2_im,
    double *RESTRICT_SC y3_re, double *RESTRICT_SC y3_im)
{
    double sumBD_re = tB_re + tD_re, sumBD_im = tB_im + tD_im;
    double difBD_re = tB_re - tD_re, difBD_im = tB_im - tD_im;
    double sumAC_re = a_re + tC_re, sumAC_im = a_im + tC_im;
    double difAC_re = a_re - tC_re, difAC_im = a_im - tC_im;

    /* Backward: rot = (-i) * difBD = (+difBD_im, -difBD_re) */
    double rot_re = difBD_im, rot_im = -difBD_re;

    *y0_re = sumAC_re + sumBD_re; *y0_im = sumAC_im + sumBD_im;
    *y1_re = difAC_re - rot_re;   *y1_im = difAC_im - rot_im;
    *y2_re = sumAC_re - sumBD_re; *y2_im = sumAC_im - sumBD_im;
    *y3_re = difAC_re + rot_re;   *y3_im = difAC_im + rot_im;
}

FORCE_INLINE_SC void radix4_stage_scalar_bv(
    size_t K,
    const double *RESTRICT_SC a_re, const double *RESTRICT_SC a_im,
    const double *RESTRICT_SC b_re, const double *RESTRICT_SC b_im,
    const double *RESTRICT_SC c_re, const double *RESTRICT_SC c_im,
    const double *RESTRICT_SC d_re, const double *RESTRICT_SC d_im,
    double *RESTRICT_SC y0_re, double *RESTRICT_SC y0_im,
    double *RESTRICT_SC y1_re, double *RESTRICT_SC y1_im,
    double *RESTRICT_SC y2_re, double *RESTRICT_SC y2_im,
    double *RESTRICT_SC y3_re, double *RESTRICT_SC y3_im,
    const double *RESTRICT_SC w1r, const double *RESTRICT_SC w1i,
    const double *RESTRICT_SC w2r, const double *RESTRICT_SC w2i,
    const double *RESTRICT_SC w3r, const double *RESTRICT_SC w3i)
{
    for (size_t k = 0; k < K; k++)
    {
        if (k + RADIX4_PREFETCH_DISTANCE_SCALAR < K)
        {
            PREFETCH_SCALAR(&a_re[k + RADIX4_PREFETCH_DISTANCE_SCALAR]);
            PREFETCH_SCALAR(&b_re[k + RADIX4_PREFETCH_DISTANCE_SCALAR]);
            PREFETCH_SCALAR(&c_re[k + RADIX4_PREFETCH_DISTANCE_SCALAR]);
            PREFETCH_SCALAR(&d_re[k + RADIX4_PREFETCH_DISTANCE_SCALAR]);
        }

        double a_r = a_re[k], a_i = a_im[k];
        double b_r = b_re[k], b_i = b_im[k];
        double c_r = c_re[k], c_i = c_im[k];
        double d_r = d_re[k], d_i = d_im[k];

        double w1_r = w1r[k], w1_i = w1i[k];
        double w2_r = w2r[k], w2_i = w2i[k];

#if RADIX4_DERIVE_W3_SCALAR
        double w3_r, w3_i;
        cmul_scalar(w1_r, w1_i, w2_r, w2_i, &w3_r, &w3_i);
#else
        double w3_r = w3r[k], w3_i = w3i[k];
#endif

        double tB_r, tB_i, tC_r, tC_i, tD_r, tD_i;
        cmul_scalar(b_r, b_i, w1_r, w1_i, &tB_r, &tB_i);
        cmul_scalar(c_r, c_i, w2_r, w2_i, &tC_r, &tC_i);
        cmul_scalar(d_r, d_i, w3_r, w3_i, &tD_r, &tD_i);

        radix4_butterfly_core_bv_scalar(a_r, a_i, tB_r, tB_i, tC_r, tC_i, tD_r, tD_i,
                                        &y0_re[k], &y0_im[k],
                                        &y1_re[k], &y1_im[k],
                                        &y2_re[k], &y2_im[k],
                                        &y3_re[k], &y3_im[k]);
    }
}

/**
 * @brief Stage wrapper - Backward FFT (Scalar)
 *
 * Twiddle layout: blocked SoA
 *   tw->re[0..K-1] = W1_re, tw->re[K..2K-1] = W2_re, tw->re[2K..3K-1] = W3_re
 */
FORCE_INLINE_SC void radix4_stage_baseptr_bv_scalar(
    size_t N,
    size_t K,
    const double *RESTRICT_SC in_re,
    const double *RESTRICT_SC in_im,
    double *RESTRICT_SC out_re,
    double *RESTRICT_SC out_im,
    const fft_twiddles_soa *RESTRICT_SC tw)
{
    (void)N;

    const double *RESTRICT_SC tw_re = tw->re;
    const double *RESTRICT_SC tw_im = tw->im;

    const double *a_re = in_re,        *a_im = in_im;
    const double *b_re = in_re + K,    *b_im = in_im + K;
    const double *c_re = in_re + 2*K,  *c_im = in_im + 2*K;
    const double *d_re = in_re + 3*K,  *d_im = in_im + 3*K;

    double *y0_re = out_re,        *y0_im = out_im;
    double *y1_re = out_re + K,    *y1_im = out_im + K;
    double *y2_re = out_re + 2*K,  *y2_im = out_im + 2*K;
    double *y3_re = out_re + 3*K,  *y3_im = out_im + 3*K;

    const double *w1r = tw_re,         *w1i = tw_im;
    const double *w2r = tw_re + K,     *w2i = tw_im + K;
    const double *w3r = tw_re + 2*K,   *w3i = tw_im + 2*K;

    radix4_stage_scalar_bv(K, a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                           y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im,
                           w1r, w1i, w2r, w2i, w3r, w3i);
}

FORCE_INLINE_SC void fft_radix4_backward_stage_scalar(
    size_t N, size_t K,
    const double *RESTRICT_SC in_re, const double *RESTRICT_SC in_im,
    double *RESTRICT_SC out_re, double *RESTRICT_SC out_im,
    const fft_twiddles_soa *RESTRICT_SC tw)
{
    radix4_stage_baseptr_bv_scalar(N, K, in_re, in_im, out_re, out_im, tw);
}

/*==========================================================================
 * FORWARD SCALAR STAGE
 *========================================================================*/

FORCE_INLINE_SC void radix4_stage_scalar_fv(
    size_t K,
    const double *RESTRICT_SC a_re, const double *RESTRICT_SC a_im,
    const double *RESTRICT_SC b_re, const double *RESTRICT_SC b_im,
    const double *RESTRICT_SC c_re, const double *RESTRICT_SC c_im,
    const double *RESTRICT_SC d_re, const double *RESTRICT_SC d_im,
    double *RESTRICT_SC y0_re, double *RESTRICT_SC y0_im,
    double *RESTRICT_SC y1_re, double *RESTRICT_SC y1_im,
    double *RESTRICT_SC y2_re, double *RESTRICT_SC y2_im,
    double *RESTRICT_SC y3_re, double *RESTRICT_SC y3_im,
    const double *RESTRICT_SC w1r, const double *RESTRICT_SC w1i,
    const double *RESTRICT_SC w2r, const double *RESTRICT_SC w2i,
    const double *RESTRICT_SC w3r, const double *RESTRICT_SC w3i)
{
    for (size_t k = 0; k < K; k++)
    {
        if (k + RADIX4_PREFETCH_DISTANCE_SCALAR < K)
        {
            PREFETCH_SCALAR(&a_re[k + RADIX4_PREFETCH_DISTANCE_SCALAR]);
            PREFETCH_SCALAR(&b_re[k + RADIX4_PREFETCH_DISTANCE_SCALAR]);
            PREFETCH_SCALAR(&c_re[k + RADIX4_PREFETCH_DISTANCE_SCALAR]);
            PREFETCH_SCALAR(&d_re[k + RADIX4_PREFETCH_DISTANCE_SCALAR]);
        }

        double a_r = a_re[k], a_i = a_im[k];
        double b_r = b_re[k], b_i = b_im[k];
        double c_r = c_re[k], c_i = c_im[k];
        double d_r = d_re[k], d_i = d_im[k];

        double w1_r = w1r[k], w1_i = w1i[k];
        double w2_r = w2r[k], w2_i = w2i[k];

#if RADIX4_DERIVE_W3_SCALAR
        double w3_r, w3_i;
        cmul_scalar(w1_r, w1_i, w2_r, w2_i, &w3_r, &w3_i);
#else
        double w3_r = w3r[k], w3_i = w3i[k];
#endif

        double tB_r, tB_i, tC_r, tC_i, tD_r, tD_i;
        cmul_scalar(b_r, b_i, w1_r, w1_i, &tB_r, &tB_i);
        cmul_scalar(c_r, c_i, w2_r, w2_i, &tC_r, &tC_i);
        cmul_scalar(d_r, d_i, w3_r, w3_i, &tD_r, &tD_i);

        radix4_butterfly_core_fv_scalar(a_r, a_i, tB_r, tB_i, tC_r, tC_i, tD_r, tD_i,
                                        &y0_re[k], &y0_im[k],
                                        &y1_re[k], &y1_im[k],
                                        &y2_re[k], &y2_im[k],
                                        &y3_re[k], &y3_im[k]);
    }
}

FORCE_INLINE_SC void radix4_stage_baseptr_fv_scalar(
    size_t N, size_t K,
    const double *RESTRICT_SC in_re, const double *RESTRICT_SC in_im,
    double *RESTRICT_SC out_re, double *RESTRICT_SC out_im,
    const fft_twiddles_soa *RESTRICT_SC tw)
{
    (void)N;
    const double *RESTRICT_SC tw_re = tw->re;
    const double *RESTRICT_SC tw_im = tw->im;

    const double *a_re = in_re,        *a_im = in_im;
    const double *b_re = in_re + K,    *b_im = in_im + K;
    const double *c_re = in_re + 2*K,  *c_im = in_im + 2*K;
    const double *d_re = in_re + 3*K,  *d_im = in_im + 3*K;

    double *y0_re = out_re,        *y0_im = out_im;
    double *y1_re = out_re + K,    *y1_im = out_im + K;
    double *y2_re = out_re + 2*K,  *y2_im = out_im + 2*K;
    double *y3_re = out_re + 3*K,  *y3_im = out_im + 3*K;

    const double *w1r = tw_re,         *w1i = tw_im;
    const double *w2r = tw_re + K,     *w2i = tw_im + K;
    const double *w3r = tw_re + 2*K,   *w3i = tw_im + 2*K;

    radix4_stage_scalar_fv(K, a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                           y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im,
                           w1r, w1i, w2r, w2i, w3r, w3i);
}

FORCE_INLINE_SC void fft_radix4_forward_stage_scalar(
    size_t N, size_t K,
    const double *RESTRICT_SC in_re, const double *RESTRICT_SC in_im,
    double *RESTRICT_SC out_re, double *RESTRICT_SC out_im,
    const fft_twiddles_soa *RESTRICT_SC tw)
{
    radix4_stage_baseptr_fv_scalar(N, K, in_re, in_im, out_re, out_im, tw);
}

#endif // FFT_RADIX4_SCALAR_OPTIMIZED_H
