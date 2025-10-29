/**
 * @file fft_radix2_scalar.h
 * @brief Scalar Radix-2 FFT Butterflies and Special Cases
 *
 * @details
 * Scalar implementations for:
 * - General butterfly processing (cleanup/fallback)
 * - Special cases with optimized twiddle factors:
 *   * k=0:    W[0] = 1 (no multiply)
 *   * k=N/4:  W[N/4] = -i (swap and negate)
 *   * k=N/8:  W[N/8] = (√2/2, -√2/2) (2 muls instead of 4)
 *   * k=3N/8: W[3N/8] = (-√2/2, -√2/2) (2 muls instead of 4)
 *
 * @author Tugbars
 * @version 3.0 (Separated architecture)
 * @date 2025
 */

#ifndef FFT_RADIX2_SCALAR_H
#define FFT_RADIX2_SCALAR_H

#include "fft_radix2_uniform.h"

//==============================================================================
// SCALAR CONFIGURATION
//==============================================================================

/// √2/2 constant for N/8 and 3N/8 twiddle optimizations
#define SQRT1_2 0.70710678118654752440

//==============================================================================
// GENERAL SCALAR BUTTERFLY
//==============================================================================

/**
 * @brief Process single butterfly - scalar fallback
 *
 * @details
 * General-purpose scalar butterfly for cleanup iterations or when SIMD
 * is not available. Processes exactly one butterfly using scalar
 * floating-point operations.
 *
 * Algorithm:
 * @code
 *   prod = odd * twiddle
 *   y[k] = even + prod
 *   y[k+half] = even - prod
 * @endcode
 *
 * @param[in] k Butterfly index
 * @param[in] in_re Input real array
 * @param[in] in_im Input imaginary array
 * @param[out] out_re Output real array
 * @param[out] out_im Output imaginary array
 * @param[in] stage_tw Stage twiddle factors (SoA)
 * @param[in] half Transform half-size
 */
static inline __attribute__((always_inline))
void radix2_pipeline_1_scalar(
    int k,
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
    const fft_twiddles_soa *restrict stage_tw,
    int half)
{
    // Load even and odd samples
    const double e_re = in_re[k];
    const double e_im = in_im[k];
    const double o_re = in_re[k + half];
    const double o_im = in_im[k + half];
    
    // Load twiddle factor
    const double w_re = stage_tw->re[k];
    const double w_im = stage_tw->im[k];
    
    // Complex multiply: prod = odd * twiddle
    const double prod_re = o_re * w_re - o_im * w_im;
    const double prod_im = o_re * w_im + o_im * w_re;
    
    // Butterfly: y0 = even + prod, y1 = even - prod
    out_re[k] = e_re + prod_re;
    out_im[k] = e_im + prod_im;
    out_re[k + half] = e_re - prod_re;
    out_im[k + half] = e_im - prod_im;
}

//==============================================================================
// SPECIAL CASE: k=0 (W[0] = 1, NO MULTIPLY!)
//==============================================================================

/**
 * @brief Special case for k=0 where twiddle = 1
 *
 * @details
 * ⚠️  CRITICAL: k=0 applies to SINGLE INDEX ONLY (not a range!)
 * 
 * At k=0, W[0] = exp(-2πi·0/N) = 1, so no twiddle multiply needed.
 * This is ONLY for the single butterfly at index k=0.
 *
 * Algorithm:
 * @code
 *   y[0] = x[0] + x[half]
 *   y[half] = x[0] - x[half]
 * @endcode
 *
 * Performance:
 * - 4 adds/subs vs 8 muls + 4 adds/subs
 * - ~3× faster than general butterfly
 * - Always worth special-casing
 *
 * @param[in] in_re Input real array
 * @param[in] in_im Input imaginary array
 * @param[out] out_re Output real array
 * @param[out] out_im Output imaginary array
 * @param[in] half Transform half-size
 */
static inline __attribute__((always_inline))
void radix2_k0_scalar(
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
    int half)
{
    const double e_re = in_re[0];
    const double e_im = in_im[0];
    const double o_re = in_re[half];
    const double o_im = in_im[half];
    
    // W[0] = 1, so just add/subtract (no multiply)
    out_re[0] = e_re + o_re;
    out_im[0] = e_im + o_im;
    out_re[half] = e_re - o_re;
    out_im[half] = e_im - o_im;
}

//==============================================================================
// SPECIAL CASE: k=N/4 (W[N/4] = -i, SWAP AND NEGATE!)
//==============================================================================

/**
 * @brief Special case for k=N/4 where twiddle = -i
 *
 * @details
 * ⚠️  CRITICAL: k=N/4 applies to SINGLE INDEX ONLY (not a range!)
 * 
 * At k=N/4 (for power-of-2 N), W[N/4] = exp(-2πi·(1/4)) = exp(-πi/2) = -i
 * 
 * Multiplying by -i = (0, -1):
 *   (a + bi) * (-i) = (a + bi) * (0 - i)
 *                   = a*0 - b*(-1) + i*(a*(-1) + b*0)
 *                   = b - ai
 *
 * So: multiply by -i = swap real and imaginary, then negate new imaginary
 *     prod_re = o_im
 *     prod_im = -o_re
 *
 * This is ONLY for the single butterfly at index k=N/4.
 *
 * Performance:
 * - 4 adds/subs + 1 negation vs 8 muls + 4 adds/subs
 * - ~2.5× faster than general butterfly
 *
 * @param[in] in_re Input real array
 * @param[in] in_im Input imaginary array
 * @param[out] out_re Output real array
 * @param[out] out_im Output imaginary array
 * @param[in] k_quarter Index k = N/4
 * @param[in] half Transform half-size
 */
static inline __attribute__((always_inline))
void radix2_k_quarter_scalar(
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
    int k_quarter,
    int half)
{
    const double e_re = in_re[k_quarter];
    const double e_im = in_im[k_quarter];
    const double o_re = in_re[k_quarter + half];
    const double o_im = in_im[k_quarter + half];
    
    // W[N/4] = -i: multiply by -i = swap and negate real
    // prod = o * (-i) = (o_re, o_im) * (0, -1) = (o_im, -o_re)
    const double prod_re = o_im;
    const double prod_im = -o_re;
    
    // Butterfly
    out_re[k_quarter] = e_re + prod_re;
    out_im[k_quarter] = e_im + prod_im;
    out_re[k_quarter + half] = e_re - prod_re;
    out_im[k_quarter + half] = e_im - prod_im;
}

//==============================================================================
// SPECIAL CASE: k=N/8 and k=3N/8 (W = ±√2/2 - i√2/2, 2 MULS INSTEAD OF 4!)
//==============================================================================

/**
 * @brief Special case for k=N/8 and k=3N/8 where twiddles are ±√2/2 - i√2/2
 *
 * @details
 * ⚠️  CRITICAL: Applies to TWO SINGLE INDICES ONLY (not ranges!)
 * 
 * For N divisible by 8 (i.e., half divisible by 4):
 * 
 * k = N/8:  W = exp(-2πi/8) = exp(-πi/4) = cos(-π/4) + i·sin(-π/4)
 *                             = √2/2 - i·√2/2
 *                             = (+√2/2, -√2/2)
 * 
 * k = 3N/8: W = exp(-6πi/8) = exp(-3πi/4) = cos(-3π/4) + i·sin(-3π/4)
 *                             = -√2/2 - i·√2/2
 *                             = (-√2/2, -√2/2)
 *
 * Complex multiply optimization:
 * For W = (sign_re·c, -c) where c = √2/2:
 *   prod_re = o_re·(sign_re·c) - o_im·(-c)
 *           = c·(sign_re·o_re + o_im)        [N/8: sign_re = +1]
 *           = c·(-o_re + o_im)               [3N/8: sign_re = -1]
 * 
 *   prod_im = o_re·(-c) + o_im·(sign_re·c)
 *           = c·(sign_re·o_im - o_re)        [N/8: sign_re = +1]
 *           = c·(-o_im - o_re)               [3N/8: sign_re = -1]
 *
 * Reduces from 4 muls + 2 adds to 2 muls + 4 adds/subs.
 *
 * Performance:
 * - 2 muls + 8 adds/subs vs 8 muls + 4 adds/subs
 * - ~2× faster than general butterfly
 *
 * @param[in] in_re Input real array
 * @param[in] in_im Input imaginary array
 * @param[out] out_re Output real array
 * @param[out] out_im Output imaginary array
 * @param[in] k_eighth Index (N/8 or 3N/8)
 * @param[in] half Transform half-size
 * @param[in] sign_re +1 for N/8, -1 for 3N/8
 */
static inline __attribute__((always_inline))
void radix2_k_eighth_scalar(
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
    int k_eighth,
    int half,
    int sign_re)
{
    const double c = SQRT1_2;  // √2/2
    
    const double e_re = in_re[k_eighth];
    const double e_im = in_im[k_eighth];
    const double o_re = in_re[k_eighth + half];
    const double o_im = in_im[k_eighth + half];
    
    // W = (sign_re·c, -c)
    // Optimized complex multiply using only 2 muls
    double sum, diff;
    
    if (sign_re > 0) {
        // N/8 case: W = (+√2/2, -√2/2)
        sum  = o_re + o_im;   // o_re + o_im
        diff = o_im - o_re;   // o_im - o_re
    } else {
        // 3N/8 case: W = (-√2/2, -√2/2)
        sum  = -o_re + o_im;  // -o_re + o_im
        diff = -o_im - o_re;  // -o_im - o_re
    }
    
    const double prod_re = c * sum;
    const double prod_im = c * diff;
    
    // Butterfly
    out_re[k_eighth] = e_re + prod_re;
    out_im[k_eighth] = e_im + prod_im;
    out_re[k_eighth + half] = e_re - prod_re;
    out_im[k_eighth + half] = e_im - prod_im;
}

//==============================================================================
// CONVENIENCE WRAPPERS FOR N/8 AND 3N/8
//==============================================================================

/**
 * @brief Special case for k=N/8 specifically
 * 
 * @details Convenience wrapper that calls radix2_k_eighth_scalar with sign_re=+1
 */
static inline __attribute__((always_inline))
void radix2_k_n8_scalar(
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
    int k_eighth,
    int half)
{
    radix2_k_eighth_scalar(in_re, in_im, out_re, out_im, k_eighth, half, +1);
}

/**
 * @brief Special case for k=3N/8 specifically
 * 
 * @details Convenience wrapper that calls radix2_k_eighth_scalar with sign_re=-1
 */
static inline __attribute__((always_inline))
void radix2_k_3n8_scalar(
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
    int k_3eighth,
    int half)
{
    radix2_k_eighth_scalar(in_re, in_im, out_re, out_im, k_3eighth, half, -1);
}

#endif // FFT_RADIX2_SCALAR_H
