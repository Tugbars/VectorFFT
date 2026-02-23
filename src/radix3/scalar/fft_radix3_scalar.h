/**
 * @file fft_radix3_scalar.h
 * @brief Scalar Radix-3 DIF Stage Kernels (FMA)
 *
 * SoA split-complex radix-3 butterfly with FMA-optimised arithmetic.
 * Provides forward/backward × full-twiddle variants.
 *
 * TWIDDLE LAYOUT (SoA contiguous, matches radix-8 convention):
 *   tw->re[0   .. K-1]  = W1_re(k)   k=0..K-1
 *   tw->re[K   .. 2K-1] = W2_re(k)
 *   tw->im[0   .. K-1]  = W1_im(k)
 *   tw->im[K   .. 2K-1] = W2_im(k)
 *
 * DATA LAYOUT:
 *   in_re[0..K-1]   = row 0 (a)      in_im same
 *   in_re[K..2K-1]  = row 1 (b)
 *   in_re[2K..3K-1] = row 2 (c)
 *
 * BUTTERFLY (forward, DIF sign = -2π):
 *   tB = b · W1,   tC = c · W2          (complex multiply)
 *   sum = tB + tC,  dif = tB - tC
 *   rot_re =  (√3/2)·dif_im
 *   rot_im = -(√3/2)·dif_re             (via fnmadd)
 *   common = a + (-½)·sum
 *   y0 = a + sum
 *   y1 = common + rot
 *   y2 = common - rot
 *
 * BACKWARD: rot signs flipped (rot_re negated, rot_im positive).
 *
 * @version 1.0
 * @date 2025
 */

#ifndef FFT_RADIX3_SCALAR_H
#define FFT_RADIX3_SCALAR_H

#include <stddef.h>
#include <math.h>

/*============================================================================
 * COMPILER PORTABILITY
 *============================================================================*/
#ifdef _MSC_VER
  #define R3S_INLINE static __forceinline
  #define R3S_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
  #define R3S_INLINE static inline __attribute__((always_inline))
  #define R3S_RESTRICT __restrict__
#else
  #define R3S_INLINE static inline
  #define R3S_RESTRICT
#endif

/*============================================================================
 * TWIDDLE TYPES (guarded — shared with other headers)
 *============================================================================*/
#ifndef RADIX3_TWIDDLE_TYPES_DEFINED
#define RADIX3_TWIDDLE_TYPES_DEFINED

typedef struct {
    const double *R3S_RESTRICT re;   /**< [2*K]: W1_re[K], W2_re[K] contiguous */
    const double *R3S_RESTRICT im;   /**< [2*K]: W1_im[K], W2_im[K] contiguous */
} radix3_stage_twiddles_t;

#endif /* RADIX3_TWIDDLE_TYPES_DEFINED */

/*============================================================================
 * CONSTANTS
 *============================================================================*/
#define R3_C_HALF      (-0.5)
#define R3_C_SQRT3_2   0.86602540378443864676372317075293618347140262690519

/*============================================================================
 * CORE: SCALAR FORWARD (with twiddles)
 *============================================================================*/

/**
 * @brief Scalar radix-3 forward stage — full twiddles.
 *
 * Processes K butterflies.  Works for any K ≥ 1.
 * FMA where the compiler supports it (e.g. -mfma).
 */
R3S_INLINE void
radix3_stage_forward_scalar(
    size_t K,
    const double *R3S_RESTRICT in_re,
    const double *R3S_RESTRICT in_im,
    double       *R3S_RESTRICT out_re,
    double       *R3S_RESTRICT out_im,
    const radix3_stage_twiddles_t *R3S_RESTRICT tw)
{
    /* Hoist row pointers (AGU optimisation) */
    const double *R3S_RESTRICT a_re = in_re;
    const double *R3S_RESTRICT a_im = in_im;
    const double *R3S_RESTRICT b_re = in_re + K;
    const double *R3S_RESTRICT b_im = in_im + K;
    const double *R3S_RESTRICT c_re = in_re + 2*K;
    const double *R3S_RESTRICT c_im = in_im + 2*K;

    double *R3S_RESTRICT o0_re = out_re;
    double *R3S_RESTRICT o0_im = out_im;
    double *R3S_RESTRICT o1_re = out_re + K;
    double *R3S_RESTRICT o1_im = out_im + K;
    double *R3S_RESTRICT o2_re = out_re + 2*K;
    double *R3S_RESTRICT o2_im = out_im + 2*K;

    /* Twiddle pointers */
    const double *R3S_RESTRICT w1r = tw->re;          /* W1 real [0..K-1] */
    const double *R3S_RESTRICT w1i = tw->im;
    const double *R3S_RESTRICT w2r = tw->re + K;      /* W2 real [K..2K-1] */
    const double *R3S_RESTRICT w2i = tw->im + K;

    const double half    = R3_C_HALF;
    const double sqrt3_2 = R3_C_SQRT3_2;

    for (size_t k = 0; k < K; k++) {
        /* Load inputs */
        double ar = a_re[k], ai = a_im[k];
        double br = b_re[k], bi = b_im[k];
        double cr = c_re[k], ci = c_im[k];

        /* Complex multiply: tB = b · W1 */
        double tBr = br * w1r[k] - bi * w1i[k];
        double tBi = br * w1i[k] + bi * w1r[k];

        /* Complex multiply: tC = c · W2 */
        double tCr = cr * w2r[k] - ci * w2i[k];
        double tCi = cr * w2i[k] + ci * w2r[k];

        /* Butterfly */
        double sum_r = tBr + tCr;
        double sum_i = tBi + tCi;
        double dif_r = tBr - tCr;
        double dif_i = tBi - tCi;

        double rot_r =  sqrt3_2 * dif_i;       /* forward: +√3/2 · dif_im */
        double rot_i = -sqrt3_2 * dif_r;       /* forward: -√3/2 · dif_re */

        double com_r = ar + half * sum_r;       /* a + (-½)·sum */
        double com_i = ai + half * sum_i;

        o0_re[k] = ar + sum_r;
        o0_im[k] = ai + sum_i;
        o1_re[k] = com_r + rot_r;
        o1_im[k] = com_i + rot_i;
        o2_re[k] = com_r - rot_r;
        o2_im[k] = com_i - rot_i;
    }
}

/*============================================================================
 * CORE: SCALAR BACKWARD (with twiddles)
 *============================================================================*/

R3S_INLINE void
radix3_stage_backward_scalar(
    size_t K,
    const double *R3S_RESTRICT in_re,
    const double *R3S_RESTRICT in_im,
    double       *R3S_RESTRICT out_re,
    double       *R3S_RESTRICT out_im,
    const radix3_stage_twiddles_t *R3S_RESTRICT tw)
{
    const double *R3S_RESTRICT a_re = in_re;
    const double *R3S_RESTRICT a_im = in_im;
    const double *R3S_RESTRICT b_re = in_re + K;
    const double *R3S_RESTRICT b_im = in_im + K;
    const double *R3S_RESTRICT c_re = in_re + 2*K;
    const double *R3S_RESTRICT c_im = in_im + 2*K;

    double *R3S_RESTRICT o0_re = out_re;
    double *R3S_RESTRICT o0_im = out_im;
    double *R3S_RESTRICT o1_re = out_re + K;
    double *R3S_RESTRICT o1_im = out_im + K;
    double *R3S_RESTRICT o2_re = out_re + 2*K;
    double *R3S_RESTRICT o2_im = out_im + 2*K;

    const double *R3S_RESTRICT w1r = tw->re;
    const double *R3S_RESTRICT w1i = tw->im;
    const double *R3S_RESTRICT w2r = tw->re + K;
    const double *R3S_RESTRICT w2i = tw->im + K;

    const double half    = R3_C_HALF;
    const double sqrt3_2 = R3_C_SQRT3_2;

    for (size_t k = 0; k < K; k++) {
        double ar = a_re[k], ai = a_im[k];
        double br = b_re[k], bi = b_im[k];
        double cr = c_re[k], ci = c_im[k];

        double tBr = br * w1r[k] - bi * w1i[k];
        double tBi = br * w1i[k] + bi * w1r[k];
        double tCr = cr * w2r[k] - ci * w2i[k];
        double tCi = cr * w2i[k] + ci * w2r[k];

        double sum_r = tBr + tCr;
        double sum_i = tBi + tCi;
        double dif_r = tBr - tCr;
        double dif_i = tBi - tCi;

        double rot_r = -sqrt3_2 * dif_i;       /* backward: -√3/2 · dif_im */
        double rot_i =  sqrt3_2 * dif_r;       /* backward: +√3/2 · dif_re */

        double com_r = ar + half * sum_r;
        double com_i = ai + half * sum_i;

        o0_re[k] = ar + sum_r;
        o0_im[k] = ai + sum_i;
        o1_re[k] = com_r + rot_r;
        o1_im[k] = com_i + rot_i;
        o2_re[k] = com_r - rot_r;
        o2_im[k] = com_i - rot_i;
    }
}

#endif /* FFT_RADIX3_SCALAR_H */
