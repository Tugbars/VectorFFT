/**
 * @file fft_radix64_scalar_n1.h
 * @brief Radix-64 N1 (Twiddle-less) Scalar Reference Implementation
 *
 * @details
 * SCALAR FALLBACK for radix-64 N1 transforms.
 * - Pure C, no SIMD intrinsics
 * - Serves as reference for correctness testing
 * - Portable to all platforms
 *
 * ARCHITECTURE: 8×8 COOLEY-TUKEY
 * ================================
 * 1. Eight radix-8 N1 butterflies (r=0..7, 8..15, ..., 56..63)
 * 2. Apply W₆₄ merge twiddles to outputs 1-7 (output 0 unchanged)
 * 3. Radix-8 final combine (2 radix-4 + W₈ structure)
 *
 * N1 OPTIMIZATION:
 * - NO stage twiddles (all W₆₄ stage twiddles = 1+0i)
 * - Only internal W₆₄ geometric merge twiddles remain
 * - 40-50% faster than standard radix-64
 *
 * @author VectorFFT Team
 * @version 1.0 (Scalar Reference)
 * @date 2025
 */

#ifndef FFT_RADIX64_SCALAR_N1_H
#define FFT_RADIX64_SCALAR_N1_H

#include <math.h>
#include <stddef.h>
#include <stdbool.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//==============================================================================
// COMPILER HINTS
//==============================================================================

#ifdef _MSC_VER
#define FORCE_INLINE static __forceinline
#define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE static inline __attribute__((always_inline))
#define RESTRICT __restrict__
#else
#define FORCE_INLINE static inline
#define RESTRICT
#endif

//==============================================================================
// W₈ GEOMETRIC CONSTANTS (REUSE FROM SIMD VERSIONS)
//==============================================================================

#ifndef W8_FV_1_RE
#define W8_FV_1_RE  0.7071067811865475  // cos(π/4)
#define W8_FV_1_IM -0.7071067811865475  // -sin(π/4)
#define W8_FV_3_RE -0.7071067811865475  // cos(3π/4)
#define W8_FV_3_IM -0.7071067811865475  // -sin(3π/4)
#endif

#ifndef W8_BV_1_RE
#define W8_BV_1_RE  0.7071067811865475  // cos(π/4)
#define W8_BV_1_IM  0.7071067811865475  // sin(π/4)
#define W8_BV_3_RE -0.7071067811865475  // cos(3π/4)
#define W8_BV_3_IM  0.7071067811865475  // sin(3π/4)
#endif

//==============================================================================
// W₆₄ GEOMETRIC CONSTANTS (FORWARD)
//==============================================================================

#ifndef W64_FV_1_RE
#define W64_FV_1_RE  0.9951847266721969   // cos(π/32)
#define W64_FV_1_IM -0.09801714032956060  // -sin(π/32)
#define W64_FV_2_RE  0.9807852804032304   // cos(2π/32) = cos(π/16)
#define W64_FV_2_IM -0.19509032201612825  // -sin(π/16)
#define W64_FV_3_RE  0.9569403357322089   // cos(3π/32)
#define W64_FV_3_IM -0.29028467725446233  // -sin(3π/32)
#define W64_FV_4_RE  0.9238795325112867   // cos(4π/32) = cos(π/8)
#define W64_FV_4_IM -0.3826834323650898   // -sin(π/8)
#define W64_FV_5_RE  0.8819212643483549   // cos(5π/32)
#define W64_FV_5_IM -0.47139673682599764  // -sin(5π/32)
#define W64_FV_6_RE  0.8314696123025452   // cos(6π/32) = cos(3π/16)
#define W64_FV_6_IM -0.5555702330196022   // -sin(3π/16)
#define W64_FV_7_RE  0.773010453362737    // cos(7π/32)
#define W64_FV_7_IM -0.6343932841636455   // -sin(7π/32)
#endif

//==============================================================================
// W₆₄ GEOMETRIC CONSTANTS (BACKWARD)
//==============================================================================

#ifndef W64_BV_1_RE
#define W64_BV_1_RE  0.9951847266721969   // cos(π/32)
#define W64_BV_1_IM  0.09801714032956060  // sin(π/32)
#define W64_BV_2_RE  0.9807852804032304   // cos(π/16)
#define W64_BV_2_IM  0.19509032201612825  // sin(π/16)
#define W64_BV_3_RE  0.9569403357322089   // cos(3π/32)
#define W64_BV_3_IM  0.29028467725446233  // sin(3π/32)
#define W64_BV_4_RE  0.9238795325112867   // cos(π/8)
#define W64_BV_4_IM  0.3826834323650898   // sin(π/8)
#define W64_BV_5_RE  0.8819212643483549   // cos(5π/32)
#define W64_BV_5_IM  0.47139673682599764  // sin(5π/32)
#define W64_BV_6_RE  0.8314696123025452   // cos(3π/16)
#define W64_BV_6_IM  0.5555702330196022   // sin(3π/16)
#define W64_BV_7_RE  0.773010453362737    // cos(7π/32)
#define W64_BV_7_IM  0.6343932841636455   // sin(7π/32)
#endif

#ifndef C8_CONSTANT
#define C8_CONSTANT 0.7071067811865475  // √2/2
#endif

//==============================================================================
// SCALAR COMPLEX MULTIPLY
//==============================================================================

/**
 * @brief Scalar complex multiply: (ar + j*ai) * (br + j*bi)
 */
FORCE_INLINE void
cmul_scalar(double ar, double ai, double br, double bi,
            double *RESTRICT tr, double *RESTRICT ti)
{
    *tr = ar * br - ai * bi;
    *ti = ar * bi + ai * br;
}

//==============================================================================
// RADIX-4 BUTTERFLY (SCALAR)
//==============================================================================

/**
 * @brief Radix-4 DIT butterfly (scalar)
 * 
 * @details
 * Standard Cooley-Tukey radix-4 with sign_flip for forward/backward:
 * - sign_flip = -1.0 (forward): multiply by -j
 * - sign_flip = +1.0 (backward): multiply by +j
 */
FORCE_INLINE void
radix4_core_scalar(
    double x0_re, double x0_im,
    double x1_re, double x1_im,
    double x2_re, double x2_im,
    double x3_re, double x3_im,
    double *RESTRICT y0_re, double *RESTRICT y0_im,
    double *RESTRICT y1_re, double *RESTRICT y1_im,
    double *RESTRICT y2_re, double *RESTRICT y2_im,
    double *RESTRICT y3_re, double *RESTRICT y3_im,
    double sign_flip)
{
    // First stage: x0+x2, x1+x3, x0-x2, x1-x3
    double a_re = x0_re + x2_re;
    double a_im = x0_im + x2_im;
    double b_re = x1_re + x3_re;
    double b_im = x1_im + x3_im;
    double c_re = x0_re - x2_re;
    double c_im = x0_im - x2_im;
    double d_re = x1_re - x3_re;
    double d_im = x1_im - x3_im;

    // Second stage: combine with ±j multiplication
    *y0_re = a_re + b_re;
    *y0_im = a_im + b_im;
    *y1_re = c_re + sign_flip * d_im;  // c + (±j)*d
    *y1_im = c_im - sign_flip * d_re;
    *y2_re = a_re - b_re;
    *y2_im = a_im - b_im;
    *y3_re = c_re - sign_flip * d_im;  // c - (±j)*d
    *y3_im = c_im + sign_flip * d_re;
}

//==============================================================================
// W₈ TWIDDLE APPLICATION (SCALAR)
//==============================================================================

/**
 * @brief Apply W₈ twiddles (forward, scalar)
 * 
 * @details
 * OPTIMIZED: Uses specialized paths for W₈^1 and W₈^3 (√2/2 constants)
 */
FORCE_INLINE void
apply_w8_twiddles_forward_scalar(
    double *RESTRICT o1_re, double *RESTRICT o1_im,
    double *RESTRICT o2_re, double *RESTRICT o2_im,
    double *RESTRICT o3_re, double *RESTRICT o3_im)
{
    const double C8 = C8_CONSTANT;
    const double NEG_C8 = -C8_CONSTANT;

    // W₈^1 = (C8, -C8) - OPTIMIZED
    {
        double r1 = *o1_re;
        double i1 = *o1_im;
        double sum = r1 + i1;
        double diff = i1 - r1;
        *o1_re = sum * C8;
        *o1_im = diff * C8;
    }

    // W₈^2 = (0, -1) - OPTIMIZED (swap + negate)
    {
        double r2 = *o2_re;
        *o2_re = *o2_im;
        *o2_im = -r2;
    }

    // W₈^3 = (-C8, -C8) - OPTIMIZED
    {
        double r3 = *o3_re;
        double i3 = *o3_im;
        double diff = r3 - i3;
        double sum = r3 + i3;
        *o3_re = diff * NEG_C8;
        *o3_im = sum * NEG_C8;
    }
}

/**
 * @brief Apply W₈ twiddles (backward, scalar)
 */
FORCE_INLINE void
apply_w8_twiddles_backward_scalar(
    double *RESTRICT o1_re, double *RESTRICT o1_im,
    double *RESTRICT o2_re, double *RESTRICT o2_im,
    double *RESTRICT o3_re, double *RESTRICT o3_im)
{
    const double C8 = C8_CONSTANT;
    const double NEG_C8 = -C8_CONSTANT;

    // W₈^(-1) = (C8, +C8) - OPTIMIZED
    {
        double r1 = *o1_re;
        double i1 = *o1_im;
        double diff = r1 - i1;
        double sum = r1 + i1;
        *o1_re = diff * C8;
        *o1_im = sum * C8;
    }

    // W₈^(-2) = (0, +1) - OPTIMIZED
    {
        double r2 = *o2_re;
        *o2_re = -(*o2_im);
        *o2_im = r2;
    }

    // W₈^(-3) = (-C8, +C8) - OPTIMIZED
    {
        double r3 = *o3_re;
        double i3 = *o3_im;
        double sum = r3 + i3;
        double diff = i3 - r3;
        *o3_re = sum * NEG_C8;
        *o3_im = diff * C8;
    }
}

//==============================================================================
// RADIX-8 N1 BUTTERFLY (SCALAR)
//==============================================================================

/**
 * @brief Radix-8 N1 butterfly (scalar, forward)
 * 
 * @details
 * Standard 8-point FFT = 2 radix-4 + W₈ twiddles
 */
FORCE_INLINE void
radix8_n1_butterfly_forward_scalar(
    double x_re[8], double x_im[8])
{
    // First radix-4: even indices
    double e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_scalar(x_re[0], x_im[0], x_re[2], x_im[2],
                       x_re[4], x_im[4], x_re[6], x_im[6],
                       &e0_re, &e0_im, &e1_re, &e1_im,
                       &e2_re, &e2_im, &e3_re, &e3_im,
                       -1.0);

    // Second radix-4: odd indices
    double o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_scalar(x_re[1], x_im[1], x_re[3], x_im[3],
                       x_re[5], x_im[5], x_re[7], x_im[7],
                       &o0_re, &o0_im, &o1_re, &o1_im,
                       &o2_re, &o2_im, &o3_re, &o3_im,
                       -1.0);

    // Apply W₈ twiddles to odd outputs
    apply_w8_twiddles_forward_scalar(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im);

    // Final combination
    x_re[0] = e0_re + o0_re;
    x_im[0] = e0_im + o0_im;
    x_re[1] = e1_re + o1_re;
    x_im[1] = e1_im + o1_im;
    x_re[2] = e2_re + o2_re;
    x_im[2] = e2_im + o2_im;
    x_re[3] = e3_re + o3_re;
    x_im[3] = e3_im + o3_im;
    x_re[4] = e0_re - o0_re;
    x_im[4] = e0_im - o0_im;
    x_re[5] = e1_re - o1_re;
    x_im[5] = e1_im - o1_im;
    x_re[6] = e2_re - o2_re;
    x_im[6] = e2_im - o2_im;
    x_re[7] = e3_re - o3_re;
    x_im[7] = e3_im - o3_im;
}

/**
 * @brief Radix-8 N1 butterfly (scalar, backward)
 */
FORCE_INLINE void
radix8_n1_butterfly_backward_scalar(
    double x_re[8], double x_im[8])
{
    // First radix-4: even indices
    double e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_scalar(x_re[0], x_im[0], x_re[2], x_im[2],
                       x_re[4], x_im[4], x_re[6], x_im[6],
                       &e0_re, &e0_im, &e1_re, &e1_im,
                       &e2_re, &e2_im, &e3_re, &e3_im,
                       +1.0);

    // Second radix-4: odd indices
    double o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_scalar(x_re[1], x_im[1], x_re[3], x_im[3],
                       x_re[5], x_im[5], x_re[7], x_im[7],
                       &o0_re, &o0_im, &o1_re, &o1_im,
                       &o2_re, &o2_im, &o3_re, &o3_im,
                       +1.0);

    // Apply W₈ conjugate twiddles
    apply_w8_twiddles_backward_scalar(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im);

    // Final combination
    x_re[0] = e0_re + o0_re;
    x_im[0] = e0_im + o0_im;
    x_re[1] = e1_re + o1_re;
    x_im[1] = e1_im + o1_im;
    x_re[2] = e2_re + o2_re;
    x_im[2] = e2_im + o2_im;
    x_re[3] = e3_re + o3_re;
    x_im[3] = e3_im + o3_im;
    x_re[4] = e0_re - o0_re;
    x_im[4] = e0_im - o0_im;
    x_re[5] = e1_re - o1_re;
    x_im[5] = e1_im - o1_im;
    x_re[6] = e2_re - o2_re;
    x_im[6] = e2_im - o2_im;
    x_re[7] = e3_re - o3_re;
    x_im[7] = e3_im - o3_im;
}

//==============================================================================
// W₆₄ MERGE TWIDDLE APPLICATION (SCALAR)
//==============================================================================

/**
 * @brief Apply W₆₄ merge twiddles (forward, scalar)
 * 
 * @details
 * OPTIMIZED: W64^2 = W₈^1 uses optimized path
 */
FORCE_INLINE void
apply_w64_merge_twiddles_forward_scalar(
    double x1_re[8], double x1_im[8],
    double x2_re[8], double x2_im[8],
    double x3_re[8], double x3_im[8],
    double x4_re[8], double x4_im[8],
    double x5_re[8], double x5_im[8],
    double x6_re[8], double x6_im[8],
    double x7_re[8], double x7_im[8])
{
    const double C8 = C8_CONSTANT;
    
    // x1 *= W64^1 (generic)
    for (int i = 0; i < 8; i++)
    {
        double tmp_re, tmp_im;
        cmul_scalar(x1_re[i], x1_im[i], W64_FV_1_RE, W64_FV_1_IM, &tmp_re, &tmp_im);
        x1_re[i] = tmp_re;
        x1_im[i] = tmp_im;
    }

    // x2 *= W64^2 = W₈^1 (OPTIMIZED)
    for (int i = 0; i < 8; i++)
    {
        double sum = x2_re[i] + x2_im[i];
        double diff = x2_im[i] - x2_re[i];
        x2_re[i] = sum * C8;
        x2_im[i] = diff * C8;
    }

    // x3 *= W64^3 (generic)
    for (int i = 0; i < 8; i++)
    {
        double tmp_re, tmp_im;
        cmul_scalar(x3_re[i], x3_im[i], W64_FV_3_RE, W64_FV_3_IM, &tmp_re, &tmp_im);
        x3_re[i] = tmp_re;
        x3_im[i] = tmp_im;
    }

    // x4 *= W64^4 (generic)
    for (int i = 0; i < 8; i++)
    {
        double tmp_re, tmp_im;
        cmul_scalar(x4_re[i], x4_im[i], W64_FV_4_RE, W64_FV_4_IM, &tmp_re, &tmp_im);
        x4_re[i] = tmp_re;
        x4_im[i] = tmp_im;
    }

    // x5 *= W64^5 (generic)
    for (int i = 0; i < 8; i++)
    {
        double tmp_re, tmp_im;
        cmul_scalar(x5_re[i], x5_im[i], W64_FV_5_RE, W64_FV_5_IM, &tmp_re, &tmp_im);
        x5_re[i] = tmp_re;
        x5_im[i] = tmp_im;
    }

    // x6 *= W64^6 (generic)
    for (int i = 0; i < 8; i++)
    {
        double tmp_re, tmp_im;
        cmul_scalar(x6_re[i], x6_im[i], W64_FV_6_RE, W64_FV_6_IM, &tmp_re, &tmp_im);
        x6_re[i] = tmp_re;
        x6_im[i] = tmp_im;
    }

    // x7 *= W64^7 (generic)
    for (int i = 0; i < 8; i++)
    {
        double tmp_re, tmp_im;
        cmul_scalar(x7_re[i], x7_im[i], W64_FV_7_RE, W64_FV_7_IM, &tmp_re, &tmp_im);
        x7_re[i] = tmp_re;
        x7_im[i] = tmp_im;
    }
}

/**
 * @brief Apply W₆₄ merge twiddles (backward, scalar)
 */
FORCE_INLINE void
apply_w64_merge_twiddles_backward_scalar(
    double x1_re[8], double x1_im[8],
    double x2_re[8], double x2_im[8],
    double x3_re[8], double x3_im[8],
    double x4_re[8], double x4_im[8],
    double x5_re[8], double x5_im[8],
    double x6_re[8], double x6_im[8],
    double x7_re[8], double x7_im[8])
{
    const double C8 = C8_CONSTANT;

    // x1 *= W64^(-1) (generic)
    for (int i = 0; i < 8; i++)
    {
        double tmp_re, tmp_im;
        cmul_scalar(x1_re[i], x1_im[i], W64_BV_1_RE, W64_BV_1_IM, &tmp_re, &tmp_im);
        x1_re[i] = tmp_re;
        x1_im[i] = tmp_im;
    }

    // x2 *= W64^(-2) = W₈^(-1) (OPTIMIZED)
    for (int i = 0; i < 8; i++)
    {
        double diff = x2_re[i] - x2_im[i];
        double sum = x2_re[i] + x2_im[i];
        x2_re[i] = diff * C8;
        x2_im[i] = sum * C8;
    }

    // x3 *= W64^(-3) (generic)
    for (int i = 0; i < 8; i++)
    {
        double tmp_re, tmp_im;
        cmul_scalar(x3_re[i], x3_im[i], W64_BV_3_RE, W64_BV_3_IM, &tmp_re, &tmp_im);
        x3_re[i] = tmp_re;
        x3_im[i] = tmp_im;
    }

    // x4 *= W64^(-4) (generic)
    for (int i = 0; i < 8; i++)
    {
        double tmp_re, tmp_im;
        cmul_scalar(x4_re[i], x4_im[i], W64_BV_4_RE, W64_BV_4_IM, &tmp_re, &tmp_im);
        x4_re[i] = tmp_re;
        x4_im[i] = tmp_im;
    }

    // x5 *= W64^(-5) (generic)
    for (int i = 0; i < 8; i++)
    {
        double tmp_re, tmp_im;
        cmul_scalar(x5_re[i], x5_im[i], W64_BV_5_RE, W64_BV_5_IM, &tmp_re, &tmp_im);
        x5_re[i] = tmp_re;
        x5_im[i] = tmp_im;
    }

    // x6 *= W64^(-6) (generic)
    for (int i = 0; i < 8; i++)
    {
        double tmp_re, tmp_im;
        cmul_scalar(x6_re[i], x6_im[i], W64_BV_6_RE, W64_BV_6_IM, &tmp_re, &tmp_im);
        x6_re[i] = tmp_re;
        x6_im[i] = tmp_im;
    }

    // x7 *= W64^(-7) (generic)
    for (int i = 0; i < 8; i++)
    {
        double tmp_re, tmp_im;
        cmul_scalar(x7_re[i], x7_im[i], W64_BV_7_RE, W64_BV_7_IM, &tmp_re, &tmp_im);
        x7_re[i] = tmp_re;
        x7_im[i] = tmp_im;
    }
}

//==============================================================================
// RADIX-8 FINAL COMBINE (SCALAR)
//==============================================================================

/**
 * @brief Radix-8 final combine (forward, scalar)
 */
FORCE_INLINE void
radix8_final_combine_forward_scalar(
    const double x0_re[8], const double x0_im[8],
    const double x1_re[8], const double x1_im[8],
    const double x2_re[8], const double x2_im[8],
    const double x3_re[8], const double x3_im[8],
    const double x4_re[8], const double x4_im[8],
    const double x5_re[8], const double x5_im[8],
    const double x6_re[8], const double x6_im[8],
    const double x7_re[8], const double x7_im[8],
    double y_re[64], double y_im[64])
{
    for (int m = 0; m < 8; m++)
    {
        double inputs_re[8] = {x0_re[m], x1_re[m], x2_re[m], x3_re[m],
                               x4_re[m], x5_re[m], x6_re[m], x7_re[m]};
        double inputs_im[8] = {x0_im[m], x1_im[m], x2_im[m], x3_im[m],
                               x4_im[m], x5_im[m], x6_im[m], x7_im[m]};

        radix8_n1_butterfly_forward_scalar(inputs_re, inputs_im);

        for (int r = 0; r < 8; r++)
        {
            y_re[m + r * 8] = inputs_re[r];
            y_im[m + r * 8] = inputs_im[r];
        }
    }
}

/**
 * @brief Radix-8 final combine (backward, scalar)
 */
FORCE_INLINE void
radix8_final_combine_backward_scalar(
    const double x0_re[8], const double x0_im[8],
    const double x1_re[8], const double x1_im[8],
    const double x2_re[8], const double x2_im[8],
    const double x3_re[8], const double x3_im[8],
    const double x4_re[8], const double x4_im[8],
    const double x5_re[8], const double x5_im[8],
    const double x6_re[8], const double x6_im[8],
    const double x7_re[8], const double x7_im[8],
    double y_re[64], double y_im[64])
{
    for (int m = 0; m < 8; m++)
    {
        double inputs_re[8] = {x0_re[m], x1_re[m], x2_re[m], x3_re[m],
                               x4_re[m], x5_re[m], x6_re[m], x7_re[m]};
        double inputs_im[8] = {x0_im[m], x1_im[m], x2_im[m], x3_im[m],
                               x4_im[m], x5_im[m], x6_im[m], x7_im[m]};

        radix8_n1_butterfly_backward_scalar(inputs_re, inputs_im);

        for (int r = 0; r < 8; r++)
        {
            y_re[m + r * 8] = inputs_re[r];
            y_im[m + r * 8] = inputs_im[r];
        }
    }
}

//==============================================================================
// MAIN DRIVER: FORWARD N1 (SCALAR)
//==============================================================================

/**
 * @brief Radix-64 DIT Forward Stage - N1 (NO TWIDDLES) - SCALAR
 *
 * @details
 * Scalar reference implementation for all platforms.
 * Serves as correctness baseline for SIMD versions.
 *
 * @param[in] K Transform 64th-size (N/64)
 * @param[in] in_re Input real array (64K elements, SoA)
 * @param[in] in_im Input imaginary array (64K elements, SoA)
 * @param[out] out_re Output real array (64K elements, SoA)
 * @param[out] out_im Output imaginary array (64K elements, SoA)
 */
void radix64_stage_dit_forward_n1_soa_scalar(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im)
{
    // Process each k-index
    for (size_t k = 0; k < K; k++)
    {
        // Load 64 input lanes
        double x_re[64], x_im[64];
        for (int r = 0; r < 64; r++)
        {
            x_re[r] = in_re[k + r * K];
            x_im[r] = in_im[k + r * K];
        }

        // Eight radix-8 N1 butterflies
        double x0_re[8], x0_im[8], x1_re[8], x1_im[8];
        double x2_re[8], x2_im[8], x3_re[8], x3_im[8];
        double x4_re[8], x4_im[8], x5_re[8], x5_im[8];
        double x6_re[8], x6_im[8], x7_re[8], x7_im[8];

        for (int i = 0; i < 8; i++)
        {
            x0_re[i] = x_re[i + 0 * 8];
            x0_im[i] = x_im[i + 0 * 8];
            x1_re[i] = x_re[i + 1 * 8];
            x1_im[i] = x_im[i + 1 * 8];
            x2_re[i] = x_re[i + 2 * 8];
            x2_im[i] = x_im[i + 2 * 8];
            x3_re[i] = x_re[i + 3 * 8];
            x3_im[i] = x_im[i + 3 * 8];
            x4_re[i] = x_re[i + 4 * 8];
            x4_im[i] = x_im[i + 4 * 8];
            x5_re[i] = x_re[i + 5 * 8];
            x5_im[i] = x_im[i + 5 * 8];
            x6_re[i] = x_re[i + 6 * 8];
            x6_im[i] = x_im[i + 6 * 8];
            x7_re[i] = x_re[i + 7 * 8];
            x7_im[i] = x_im[i + 7 * 8];
        }

        radix8_n1_butterfly_forward_scalar(x0_re, x0_im);
        radix8_n1_butterfly_forward_scalar(x1_re, x1_im);
        radix8_n1_butterfly_forward_scalar(x2_re, x2_im);
        radix8_n1_butterfly_forward_scalar(x3_re, x3_im);
        radix8_n1_butterfly_forward_scalar(x4_re, x4_im);
        radix8_n1_butterfly_forward_scalar(x5_re, x5_im);
        radix8_n1_butterfly_forward_scalar(x6_re, x6_im);
        radix8_n1_butterfly_forward_scalar(x7_re, x7_im);

        // Apply W₆₄ merge twiddles
        apply_w64_merge_twiddles_forward_scalar(
            x1_re, x1_im, x2_re, x2_im, x3_re, x3_im, x4_re, x4_im,
            x5_re, x5_im, x6_re, x6_im, x7_re, x7_im);

        // Radix-8 final combine
        double y_re[64], y_im[64];
        radix8_final_combine_forward_scalar(
            x0_re, x0_im, x1_re, x1_im, x2_re, x2_im, x3_re, x3_im,
            x4_re, x4_im, x5_re, x5_im, x6_re, x6_im, x7_re, x7_im,
            y_re, y_im);

        // Store 64 output lanes
        for (int r = 0; r < 64; r++)
        {
            out_re[k + r * K] = y_re[r];
            out_im[k + r * K] = y_im[r];
        }
    }
}

//==============================================================================
// MAIN DRIVER: BACKWARD N1 (SCALAR)
//==============================================================================

/**
 * @brief Radix-64 DIT Backward Stage - N1 (NO TWIDDLES) - SCALAR
 */
void radix64_stage_dit_backward_n1_soa_scalar(
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im)
{
    for (size_t k = 0; k < K; k++)
    {
        // Load 64 input lanes
        double x_re[64], x_im[64];
        for (int r = 0; r < 64; r++)
        {
            x_re[r] = in_re[k + r * K];
            x_im[r] = in_im[k + r * K];
        }

        // Eight radix-8 N1 butterflies
        double x0_re[8], x0_im[8], x1_re[8], x1_im[8];
        double x2_re[8], x2_im[8], x3_re[8], x3_im[8];
        double x4_re[8], x4_im[8], x5_re[8], x5_im[8];
        double x6_re[8], x6_im[8], x7_re[8], x7_im[8];

        for (int i = 0; i < 8; i++)
        {
            x0_re[i] = x_re[i + 0 * 8];
            x0_im[i] = x_im[i + 0 * 8];
            x1_re[i] = x_re[i + 1 * 8];
            x1_im[i] = x_im[i + 1 * 8];
            x2_re[i] = x_re[i + 2 * 8];
            x2_im[i] = x_im[i + 2 * 8];
            x3_re[i] = x_re[i + 3 * 8];
            x3_im[i] = x_im[i + 3 * 8];
            x4_re[i] = x_re[i + 4 * 8];
            x4_im[i] = x_im[i + 4 * 8];
            x5_re[i] = x_re[i + 5 * 8];
            x5_im[i] = x_im[i + 5 * 8];
            x6_re[i] = x_re[i + 6 * 8];
            x6_im[i] = x_im[i + 6 * 8];
            x7_re[i] = x_re[i + 7 * 8];
            x7_im[i] = x_im[i + 7 * 8];
        }

        radix8_n1_butterfly_backward_scalar(x0_re, x0_im);
        radix8_n1_butterfly_backward_scalar(x1_re, x1_im);
        radix8_n1_butterfly_backward_scalar(x2_re, x2_im);
        radix8_n1_butterfly_backward_scalar(x3_re, x3_im);
        radix8_n1_butterfly_backward_scalar(x4_re, x4_im);
        radix8_n1_butterfly_backward_scalar(x5_re, x5_im);
        radix8_n1_butterfly_backward_scalar(x6_re, x6_im);
        radix8_n1_butterfly_backward_scalar(x7_re, x7_im);

        // Apply W₆₄ merge twiddles (backward)
        apply_w64_merge_twiddles_backward_scalar(
            x1_re, x1_im, x2_re, x2_im, x3_re, x3_im, x4_re, x4_im,
            x5_re, x5_im, x6_re, x6_im, x7_re, x7_im);

        // Radix-8 final combine (backward)
        double y_re[64], y_im[64];
        radix8_final_combine_backward_scalar(
            x0_re, x0_im, x1_re, x1_im, x2_re, x2_im, x3_re, x3_im,
            x4_re, x4_im, x5_re, x5_im, x6_re, x6_im, x7_re, x7_im,
            y_re, y_im);

        // Store 64 output lanes
        for (int r = 0; r < 64; r++)
        {
            out_re[k + r * K] = y_re[r];
            out_im[k + r * K] = y_im[r];
        }
    }
}

#endif // FFT_RADIX64_SCALAR_N1_H