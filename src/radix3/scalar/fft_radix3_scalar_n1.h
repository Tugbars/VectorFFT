/**
 * @file fft_radix3_scalar_n1.h
 * @brief Scalar Radix-3 N1 (Twiddle-less) Stage Kernels
 *
 * For stages where all twiddle factors are unity (W^0 = 1).
 * Pure butterfly only — no complex multiply step.
 *
 * y0 = a + b + c
 * y1 = a + ω·b + ω²·c  =  a + (-½)(b+c) ± j(√3/2)(b-c)
 * y2 = a + ω²·b + ω·c
 *
 * @version 1.0
 * @date 2025
 */

#ifndef FFT_RADIX3_SCALAR_N1_H
#define FFT_RADIX3_SCALAR_N1_H

#include "fft_radix3_scalar.h"   /* picks up types, RESTRICT, constants */

/*============================================================================
 * FORWARD N1
 *============================================================================*/

R3S_INLINE void
radix3_stage_n1_forward_scalar(
    size_t K,
    const double *R3S_RESTRICT in_re,
    const double *R3S_RESTRICT in_im,
    double       *R3S_RESTRICT out_re,
    double       *R3S_RESTRICT out_im)
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

    const double half    = R3_C_HALF;
    const double sqrt3_2 = R3_C_SQRT3_2;

    for (size_t k = 0; k < K; k++) {
        double ar = a_re[k], ai = a_im[k];
        double br = b_re[k], bi = b_im[k];
        double cr = c_re[k], ci = c_im[k];

        double sum_r = br + cr,  sum_i = bi + ci;
        double dif_r = br - cr,  dif_i = bi - ci;

        double rot_r =  sqrt3_2 * dif_i;
        double rot_i = -sqrt3_2 * dif_r;
        double com_r = ar + half * sum_r;
        double com_i = ai + half * sum_i;

        o0_re[k] = ar + sum_r;  o0_im[k] = ai + sum_i;
        o1_re[k] = com_r + rot_r;  o1_im[k] = com_i + rot_i;
        o2_re[k] = com_r - rot_r;  o2_im[k] = com_i - rot_i;
    }
}

/*============================================================================
 * BACKWARD N1
 *============================================================================*/

R3S_INLINE void
radix3_stage_n1_backward_scalar(
    size_t K,
    const double *R3S_RESTRICT in_re,
    const double *R3S_RESTRICT in_im,
    double       *R3S_RESTRICT out_re,
    double       *R3S_RESTRICT out_im)
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

    const double half    = R3_C_HALF;
    const double sqrt3_2 = R3_C_SQRT3_2;

    for (size_t k = 0; k < K; k++) {
        double ar = a_re[k], ai = a_im[k];
        double br = b_re[k], bi = b_im[k];
        double cr = c_re[k], ci = c_im[k];

        double sum_r = br + cr,  sum_i = bi + ci;
        double dif_r = br - cr,  dif_i = bi - ci;

        double rot_r = -sqrt3_2 * dif_i;        /* backward: sign flip */
        double rot_i =  sqrt3_2 * dif_r;
        double com_r = ar + half * sum_r;
        double com_i = ai + half * sum_i;

        o0_re[k] = ar + sum_r;  o0_im[k] = ai + sum_i;
        o1_re[k] = com_r + rot_r;  o1_im[k] = com_i + rot_i;
        o2_re[k] = com_r - rot_r;  o2_im[k] = com_i - rot_i;
    }
}

#endif /* FFT_RADIX3_SCALAR_N1_H */