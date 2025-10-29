/**
 * @file fft_radix8_scalar_first_stage.h
 * @brief Radix-8 SCALAR First Stage (Twiddle-less) - K=1 Optimized
 *
 * @details
 * FIRST STAGE OPTIMIZATION:
 * =========================
 * - No stage twiddles (all W_N^(k*m) = W_N^0 = 1 when k=0)
 * - Only W_8 geometric constants needed
 * - Eliminates 7 complex multiplications per butterfly
 * - Maximum performance for N=8 base case
 *
 * USE CASES:
 * ==========
 * 1. First stage of mixed-radix FFT (N=8*K where K>1)
 * 2. Standalone N=8 FFT
 * 3. Base case in recursive decomposition
 *
 * OPTIMIZATIONS INCLUDED:
 * =======================
 * ✅ Zero stage twiddle overhead (identity multiplications skipped)
 * ✅ Hoisted W_8 geometric constants
 * ✅ Optimized radix-4 core
 * ✅ Pure C - maximum portability
 * ✅ Minimal memory bandwidth
 *
 * @author FFT Optimization Team
 * @version 1.0-SCALAR-FIRST-STAGE
 * @date 2025
 */

#ifndef FFT_RADIX8_SCALAR_FIRST_STAGE_H
#define FFT_RADIX8_SCALAR_FIRST_STAGE_H

#include <stddef.h>
#include <stdint.h>

//==============================================================================
// COMPILER PORTABILITY
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
// W_8 GEOMETRIC CONSTANTS
//==============================================================================

#define C8_CONSTANT 0.7071067811865475244008443621048490392848359376887

#define W8_FV_1_RE C8_CONSTANT
#define W8_FV_1_IM (-C8_CONSTANT)
#define W8_FV_3_RE (-C8_CONSTANT)
#define W8_FV_3_IM (-C8_CONSTANT)

#define W8_BV_1_RE C8_CONSTANT
#define W8_BV_1_IM C8_CONSTANT
#define W8_BV_3_RE (-C8_CONSTANT)
#define W8_BV_3_IM C8_CONSTANT

//==============================================================================
// RADIX-4 CORE (SCALAR)
//==============================================================================

FORCE_INLINE void
radix4_core_scalar(
    double x0_re, double x0_im, double x1_re, double x1_im,
    double x2_re, double x2_im, double x3_re, double x3_im,
    double *RESTRICT y0_re, double *RESTRICT y0_im,
    double *RESTRICT y1_re, double *RESTRICT y1_im,
    double *RESTRICT y2_re, double *RESTRICT y2_im,
    double *RESTRICT y3_re, double *RESTRICT y3_im,
    int forward)
{
    double t0_re = x0_re + x2_re;
    double t0_im = x0_im + x2_im;
    double t1_re = x0_re - x2_re;
    double t1_im = x0_im - x2_im;
    double t2_re = x1_re + x3_re;
    double t2_im = x1_im + x3_im;
    double t3_re = x1_re - x3_re;
    double t3_im = x1_im - x3_im;

    *y0_re = t0_re + t2_re;
    *y0_im = t0_im + t2_im;
    *y2_re = t0_re - t2_re;
    *y2_im = t0_im - t2_im;

    if (forward)
    {
        *y1_re = t1_re + t3_im;
        *y1_im = t1_im - t3_re;
        *y3_re = t1_re - t3_im;
        *y3_im = t1_im + t3_re;
    }
    else
    {
        *y1_re = t1_re - t3_im;
        *y1_im = t1_im + t3_re;
        *y3_re = t1_re + t3_im;
        *y3_im = t1_im - t3_re;
    }
}

//==============================================================================
// W_8 TWIDDLE APPLICATION
//==============================================================================

FORCE_INLINE void
apply_w8_twiddles_forward_scalar(
    double *RESTRICT o1_re, double *RESTRICT o1_im,
    double *RESTRICT o2_re, double *RESTRICT o2_im,
    double *RESTRICT o3_re, double *RESTRICT o3_im,
    double W8_1_re, double W8_1_im,
    double W8_3_re, double W8_3_im)
{
    // W_8^1 multiplication
    double r1 = *o1_re, i1 = *o1_im;
    *o1_re = r1 * W8_1_re - i1 * W8_1_im;
    *o1_im = r1 * W8_1_im + i1 * W8_1_re;

    // W_8^2 = (0, -1) - optimized as swap + negate
    double r2 = *o2_re;
    *o2_re = *o2_im;
    *o2_im = -r2;

    // W_8^3 multiplication
    double r3 = *o3_re, i3 = *o3_im;
    *o3_re = r3 * W8_3_re - i3 * W8_3_im;
    *o3_im = r3 * W8_3_im + i3 * W8_3_re;
}

FORCE_INLINE void
apply_w8_twiddles_backward_scalar(
    double *RESTRICT o1_re, double *RESTRICT o1_im,
    double *RESTRICT o2_re, double *RESTRICT o2_im,
    double *RESTRICT o3_re, double *RESTRICT o3_im,
    double W8_1_re, double W8_1_im,
    double W8_3_re, double W8_3_im)
{
    // W_8^(-1) multiplication
    double r1 = *o1_re, i1 = *o1_im;
    *o1_re = r1 * W8_1_re - i1 * W8_1_im;
    *o1_im = r1 * W8_1_im + i1 * W8_1_re;

    // W_8^(-2) = (0, 1) - optimized as negate + swap
    double r2 = *o2_re;
    *o2_re = -(*o2_im);
    *o2_im = r2;

    // W_8^(-3) multiplication
    double r3 = *o3_re, i3 = *o3_im;
    *o3_re = r3 * W8_3_re - i3 * W8_3_im;
    *o3_im = r3 * W8_3_im + i3 * W8_3_re;
}

//==============================================================================
// RADIX-8 BUTTERFLY - FIRST STAGE (NO STAGE TWIDDLES)
//==============================================================================

/**
 * @brief Radix-8 butterfly for first stage (k=0, all stage twiddles = 1)
 * @param in_re Input real array (length 8)
 * @param in_im Input imaginary array (length 8)
 * @param out_re Output real array (length 8)
 * @param out_im Output imaginary array (length 8)
 * @param forward 1 for forward FFT, 0 for inverse FFT
 */
FORCE_INLINE void
radix8_butterfly_first_stage_forward_scalar(
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    // Load all 8 inputs (K=1, so indices are just 0..7)
    double x0_re = in_re[0];
    double x0_im = in_im[0];
    double x1_re = in_re[1];
    double x1_im = in_im[1];
    double x2_re = in_re[2];
    double x2_im = in_im[2];
    double x3_re = in_re[3];
    double x3_im = in_im[3];
    double x4_re = in_re[4];
    double x4_im = in_im[4];
    double x5_re = in_re[5];
    double x5_im = in_im[5];
    double x6_re = in_re[6];
    double x6_im = in_im[6];
    double x7_re = in_re[7];
    double x7_im = in_im[7];

    // NO STAGE TWIDDLES - x1..x7 remain unchanged

    // Even radix-4: DFT of [x0, x2, x4, x6]
    double e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_scalar(x0_re, x0_im, x2_re, x2_im, x4_re, x4_im, x6_re, x6_im,
                       &e0_re, &e0_im, &e1_re, &e1_im, &e2_re, &e2_im, &e3_re, &e3_im, 1);

    // Odd radix-4: DFT of [x1, x3, x5, x7]
    double o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_scalar(x1_re, x1_im, x3_re, x3_im, x5_re, x5_im, x7_re, x7_im,
                       &o0_re, &o0_im, &o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im, 1);

    // Apply W_8 geometric twiddles to odd outputs
    apply_w8_twiddles_forward_scalar(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                     W8_FV_1_RE, W8_FV_1_IM, W8_FV_3_RE, W8_FV_3_IM);

    // Combine even + odd
    out_re[0] = e0_re + o0_re;
    out_im[0] = e0_im + o0_im;
    out_re[1] = e1_re + o1_re;
    out_im[1] = e1_im + o1_im;
    out_re[2] = e2_re + o2_re;
    out_im[2] = e2_im + o2_im;
    out_re[3] = e3_re + o3_re;
    out_im[3] = e3_im + o3_im;
    out_re[4] = e0_re - o0_re;
    out_im[4] = e0_im - o0_im;
    out_re[5] = e1_re - o1_re;
    out_im[5] = e1_im - o1_im;
    out_re[6] = e2_re - o2_re;
    out_im[6] = e2_im - o2_im;
    out_re[7] = e3_re - o3_re;
    out_im[7] = e3_im - o3_im;
}

/**
 * @brief Radix-8 butterfly for first stage - BACKWARD/INVERSE
 */
FORCE_INLINE void
radix8_butterfly_first_stage_backward_scalar(
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    // Load all 8 inputs
    double x0_re = in_re[0];
    double x0_im = in_im[0];
    double x1_re = in_re[1];
    double x1_im = in_im[1];
    double x2_re = in_re[2];
    double x2_im = in_im[2];
    double x3_re = in_re[3];
    double x3_im = in_im[3];
    double x4_re = in_re[4];
    double x4_im = in_im[4];
    double x5_re = in_re[5];
    double x5_im = in_im[5];
    double x6_re = in_re[6];
    double x6_im = in_im[6];
    double x7_re = in_re[7];
    double x7_im = in_im[7];

    // NO STAGE TWIDDLES

    // Even radix-4: IDFT of [x0, x2, x4, x6]
    double e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_scalar(x0_re, x0_im, x2_re, x2_im, x4_re, x4_im, x6_re, x6_im,
                       &e0_re, &e0_im, &e1_re, &e1_im, &e2_re, &e2_im, &e3_re, &e3_im, 0);

    // Odd radix-4: IDFT of [x1, x3, x5, x7]
    double o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_scalar(x1_re, x1_im, x3_re, x3_im, x5_re, x5_im, x7_re, x7_im,
                       &o0_re, &o0_im, &o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im, 0);

    // Apply W_8^(-1) geometric twiddles
    apply_w8_twiddles_backward_scalar(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im,
                                      W8_BV_1_RE, W8_BV_1_IM, W8_BV_3_RE, W8_BV_3_IM);

    // Combine
    out_re[0] = e0_re + o0_re;
    out_im[0] = e0_im + o0_im;
    out_re[1] = e1_re + o1_re;
    out_im[1] = e1_im + o1_im;
    out_re[2] = e2_re + o2_re;
    out_im[2] = e2_im + o2_im;
    out_re[3] = e3_re + o3_re;
    out_im[3] = e3_im + o3_im;
    out_re[4] = e0_re - o0_re;
    out_im[4] = e0_im - o0_im;
    out_re[5] = e1_re - o1_re;
    out_im[5] = e1_im - o1_im;
    out_re[6] = e2_re - o2_re;
    out_im[6] = e2_im - o2_im;
    out_re[7] = e3_re - o3_re;
    out_im[7] = e3_im - o3_im;
}

//==============================================================================
// BATCH PROCESSING (MULTIPLE RADIX-8 BUTTERFLIES)
//==============================================================================

/**
 * @brief Process multiple first-stage radix-8 butterflies
 * @param num_butterflies Number of independent radix-8 butterflies
 * @param in_re Input real array (length = 8 * num_butterflies)
 * @param in_im Input imaginary array
 * @param out_re Output real array
 * @param out_im Output imaginary array
 * @param stride Stride between butterfly groups (typically 8)
 */
FORCE_INLINE void
radix8_first_stage_batch_forward_scalar(
    size_t num_butterflies,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    size_t stride)
{
    for (size_t i = 0; i < num_butterflies; i++)
    {
        size_t offset = i * stride;
        radix8_butterfly_first_stage_forward_scalar(
            in_re + offset, in_im + offset,
            out_re + offset, out_im + offset);
    }
}

/**
 * @brief Process multiple first-stage radix-8 butterflies - BACKWARD
 */
FORCE_INLINE void
radix8_first_stage_batch_backward_scalar(
    size_t num_butterflies,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im,
    size_t stride)
{
    for (size_t i = 0; i < num_butterflies; i++)
    {
        size_t offset = i * stride;
        radix8_butterfly_first_stage_backward_scalar(
            in_re + offset, in_im + offset,
            out_re + offset, out_im + offset);
    }
}

#endif // FFT_RADIX8_SCALAR_FIRST_STAGE_H