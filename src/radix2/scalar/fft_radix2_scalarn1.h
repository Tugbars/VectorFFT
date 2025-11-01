/**
 * @file fft_radix2_scalarn1.h
 * @brief Scalar Twiddle-Less Radix-2 FFT Butterflies (W=1 Optimized)
 *
 * @details
 * Scalar twiddle-less butterfly implementations for cases where W=1.
 * These are the building blocks for first-stage optimizations and provide
 * fallback for architectures without SIMD support.
 *
 * Performance benefit: ~3× faster than general butterfly
 * - General butterfly: 4 muls + 4 adds/subs = ~8 scalar ops
 * - Twiddle-less:     0 muls + 4 adds/subs = ~4 scalar ops
 *
 * Use cases:
 * - First stage of multi-stage FFT (most twiddles = 1)
 * - Stockham auto-sort first pass (no twiddles)
 * - Cleanup/fallback when SIMD not available
 * - Testing and validation reference
 *
 * @author Tugbars
 * @version 3.0 (Separated architecture, twiddle-less variant)
 * @date 2025
 */

#ifndef FFT_RADIX2_SCALARN1_H
#define FFT_RADIX2_SCALARN1_H

#include "fft_radix2_uniform.h"

//==============================================================================
// TWIDDLE-LESS BUTTERFLY - SCALAR (W=1)
//==============================================================================

/**
 * @brief Process single butterfly without twiddle multiply (W=1) - scalar
 *
 * @details
 * ⚡⚡⚡ ULTRA-FAST VERSION - No complex multiply!
 *
 * Computes simplified FFT butterfly with W=1:
 * @code
 *   y[k] = x[k] + x[k+half]
 *   y[k+half] = x[k] - x[k+half]
 * @endcode
 *
 * Performance:
 * - 4 scalar add/sub operations (vs 4 muls + 4 adds/subs)
 * - ~3× faster than general butterfly
 * - ~4 cycles latency on modern x86-64
 *
 * Typical use case:
 * @code
 *   // First stage: butterflies with W=1
 *   for (k = 0; k < threshold; k++) {
 *       radix2_pipeline_1_scalar_n1(...);
 *   }
 *   // Remaining butterflies: use general butterfly with twiddles
 *   for (k = threshold; k < half; k++) {
 *       radix2_pipeline_1_scalar(...);
 *   }
 * @endcode
 *
 * @param[in] k Butterfly index
 * @param[in] in_re Input real array
 * @param[in] in_im Input imaginary array
 * @param[out] out_re Output real array
 * @param[out] out_im Output imaginary array
 * @param[in] half Transform half-size
 *
 * @note NO stage_tw parameter - no twiddles used!
 * @note Total operations: 4 scalar add/sub
 * @note Compare to general butterfly: 4 muls + 4 adds/subs = 8 ops
 */
static inline __attribute__((always_inline))
void radix2_pipeline_1_scalar_n1(
    int k,
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
    int half)
{
    // Load even and odd samples
    const double e_re = in_re[k];
    const double e_im = in_im[k];
    const double o_re = in_re[k + half];
    const double o_im = in_im[k + half];
    
    // W=1, so prod = odd (no multiply needed!)
    // Butterfly: y0 = even + odd, y1 = even - odd
    out_re[k] = e_re + o_re;
    out_im[k] = e_im + o_im;
    out_re[k + half] = e_re - o_re;
    out_im[k + half] = e_im - o_im;
}

//==============================================================================
// RANGE PROCESSING - TWIDDLE-LESS SCALAR
//==============================================================================

/**
 * @brief Process range of butterflies WITHOUT twiddles - scalar loop
 *
 * @details
 * Processes all butterflies in range [k_start, k_end) with W=1.
 * Used for first stage optimization or when all twiddles in range are 1.
 *
 * This is a convenience function that wraps the single-butterfly version
 * in a loop, useful for integrating into higher-level FFT stages.
 *
 * @param[in] k_start Starting butterfly index
 * @param[in] k_end Ending butterfly index (exclusive)
 * @param[in] in_re Input real array
 * @param[in] in_im Input imaginary array
 * @param[out] out_re Output real array
 * @param[out] out_im Output imaginary array
 * @param[in] half Transform half-size
 *
 * @note Processes k_end - k_start butterflies
 * @note All butterflies in range must have W=1 for correctness
 */
static inline __attribute__((always_inline))
void radix2_range_scalar_n1(
    int k_start,
    int k_end,
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
    int half)
{
    for (int k = k_start; k < k_end; k++)
    {
        radix2_pipeline_1_scalar_n1(k, in_re, in_im, out_re, out_im, half);
    }
}

//==============================================================================
// SPECIAL CASE: ENTIRE STAGE WITH W=1 (STOCKHAM AUTO-SORT)
//==============================================================================

/**
 * @brief Process entire stage WITHOUT twiddles (all W=1)
 *
 * @details
 * Special case for Stockham auto-sort or first FFT stage where ALL
 * butterflies have W=1. This is the fastest possible radix-2 stage.
 *
 * Use cases:
 * - Stockham auto-sort first pass (bit-reversal equivalent)
 * - First stage of DIF FFT (all twiddles = 1)
 * - Testing and validation
 *
 * Algorithm:
 * @code
 *   for k in [0, half):
 *       y[k] = x[k] + x[k+half]
 *       y[k+half] = x[k] - x[k+half]
 * @endcode
 *
 * Performance:
 * - 4N scalar operations for N-point FFT stage
 * - ~3× faster than general radix-2 stage with twiddles
 * - Ideal for small N or validation testing
 *
 * @param[out] out_re Output real array (N elements)
 * @param[out] out_im Output imaginary array (N elements)
 * @param[in] in_re Input real array (N elements)
 * @param[in] in_im Input imaginary array (N elements)
 * @param[in] half Transform half-size (N/2)
 *
 * @note All butterflies processed with W=1 (no twiddle multiply)
 * @note Out-of-place only (in != out)
 * @note Total operations: 4*half scalar add/sub = 2N operations
 */
static inline void radix2_stage_scalar_n1_full(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    int half)
{
    radix2_range_scalar_n1(0, half, in_re, in_im, out_re, out_im, half);
}

//==============================================================================
// HYBRID: TWIDDLE-LESS + GENERAL (FIRST STAGE OPTIMIZATION)
//==============================================================================

/**
 * @brief Hybrid radix-2 stage: twiddle-less for small k, general for rest
 *
 * @details
 * Optimized for first FFT stage where many twiddles W[k] ≈ 1 for small k.
 * Automatically switches from twiddle-less to general butterfly at threshold.
 *
 * Algorithm:
 * @code
 *   // For k in [0, threshold): Use twiddle-less (W=1 assumed)
 *   for (k = 0; k < threshold; k++) {
 *       y0 = even + odd
 *       y1 = even - odd
 *   }
 *   // For k in [threshold, half): Use general butterfly with twiddles
 *   for (k = threshold; k < half; k++) {
 *       prod = odd * W[k]
 *       y0 = even + prod
 *       y1 = even - prod
 *   }
 * @endcode
 *
 * Typical threshold: 1-4 butterflies depending on N
 *
 * @param[out] out_re Output real array (N elements)
 * @param[out] out_im Output imaginary array (N elements)
 * @param[in] in_re Input real array (N elements)
 * @param[in] in_im Input imaginary array (N elements)
 * @param[in] stage_tw Stage twiddle factors (SoA, used only after threshold)
 * @param[in] half Transform half-size (N/2)
 * @param[in] n1_threshold Number of butterflies to process without twiddles
 *
 * @note Threshold=0 means use twiddles for all butterflies (normal behavior)
 * @note Threshold=half means use twiddle-less for ALL butterflies (W=1 everywhere)
 * @note Typical threshold: 1, 2, or 4 for first stage optimization
 * @note For threshold < k < half, caller must have stage_tw precomputed
 */
static inline void radix2_stage_scalar_n1_hybrid(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw,
    int half,
    int n1_threshold)
{
    int k = 0;
    
    // Phase 1: Twiddle-less butterflies (W=1 for small k)
    if (n1_threshold > 0)
    {
        radix2_range_scalar_n1(0, n1_threshold, in_re, in_im,
                               out_re, out_im, half);
        k = n1_threshold;
    }
    
    // Phase 2: General butterflies with twiddles (remaining k)
    while (k < half)
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
        
        k++;
    }
}

#endif // FFT_RADIX2_SCALARN1_H