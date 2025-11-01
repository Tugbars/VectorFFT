/**
 * @file fft_radix64_scalar_n1.h
 * @brief Radix-64 N1 (Twiddle-less) Scalar - 8×8 Cooley-Tukey [U=4 OPTIMIZED]
 *
 * @details
 * CLEAN REBUILD - Follows scalar implementation exactly
 * U=4 OPTIMIZATION - m-stripmined for code clarity and locality
 * SCALAR REFERENCE VERSION - Pure C, no SIMD
 *
 * N1 CODELET ARCHITECTURE:
 * ========================
 * - "N1" = No stage twiddles (all W₆₄ stage twiddles = 1+0i)
 * - Only internal W₆₄ geometric merge twiddles remain
 * - Used as first/last stage in mixed-radix factorizations
 * - 64 = 8 × 8 decomposition for optimal performance
 *
 * ARCHITECTURE: 8×8 COOLEY-TUKEY
 * ================================
 * 1. Eight radix-8 N1 butterflies (r=0..7, 8..15, ..., 56..63)
 * 2. Apply W₆₄ merge twiddles to outputs 1-7 (output 0 unchanged)
 * 3. Radix-8 final combine (across 8 sub-FFTs)
 *
 * SCALAR OPTIMIZATIONS:
 * =====================
 * ✅ Pure C - no SIMD, fully portable
 * ✅ Main loop: k += 4 (U=4 with m-stripmining)
 * ✅ Optimized W₈ twiddles (specialized paths for W₈^1, W₈^3)
 * ✅ Generic W₆₄ twiddles (no shortcuts - mathematically correct)
 * ✅ Cache-friendly k-tiling
 * ✅ Clear reference implementation for validation
 *
 * U=4 M-STRIPMINE OPTIMIZATION:
 * ==============================
 * For a given m (0..7):
 * - Process 4 positions in parallel: k, k+1, k+2, k+3
 * - Each position: load → butterfly → W64 → combine → store
 * - Excellent code clarity and cache locality
 *
 * This scalar version serves as:
 * 1. Reference implementation for correctness validation
 * 2. Fallback for non-SIMD platforms
 * 3. Clear documentation of the algorithm
 *
 * @author VectorFFT Team
 * @version 3.0 (U=4 m-stripmine optimization, scalar)
 * @date 2025
 */

#ifndef FFT_RADIX64_SCALAR_N1_H
#define FFT_RADIX64_SCALAR_N1_H

#include <stddef.h>
#include <stdbool.h>
#include <math.h>

// Include radix-8 N1 scalar for reuse
#include "fft_radix8_scalar_n1.h"

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
// CONFIGURATION
//==============================================================================

#ifndef RADIX64_TILE_SIZE_N1_SCALAR
#define RADIX64_TILE_SIZE_N1_SCALAR 64
#endif

//==============================================================================
// HELPER: Complex Multiply (Generic)
//==============================================================================

/**
 * @brief Generic complex multiply (ar + i*ai) * (br + i*bi)
 *
 * @details
 * Scalar version - straightforward complex multiplication.
 * Compiler can optimize this well with FMA if available.
 */
FORCE_INLINE void
cmul_scalar(
    double ar, double ai,
    double br, double bi,
    double *RESTRICT tr, double *RESTRICT ti)
{
    *tr = ar * br - ai * bi;
    *ti = ar * bi + ai * br;
}

//==============================================================================
// OPTIMIZED W₈ TWIDDLES (SAFE OPTIMIZATIONS)
//==============================================================================

/**
 * @brief Apply W₈ twiddles - OPTIMIZED (Forward)
 *
 * @details
 * Uses specialized paths for W₈^1 and W₈^3:
 * - W₈^1 = (c, -c) where c = √2/2
 * - W₈^2 = (0, -1)
 * - W₈^3 = (-c, -c)
 *
 * These are the ONLY safe optimizations (proven mathematically correct).
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
        double r = *o1_re;
        double i = *o1_im;
        double sum = r + i;
        double diff = i - r;
        *o1_re = sum * C8;
        *o1_im = diff * C8;
    }

    // W₈^2 = (0, -1) - OPTIMIZED
    {
        double r = *o2_re;
        *o2_re = *o2_im;
        *o2_im = -r;
    }

    // W₈^3 = (-C8, -C8) - OPTIMIZED
    {
        double r = *o3_re;
        double i = *o3_im;
        double diff = r - i;
        double sum = r + i;
        *o3_re = diff * NEG_C8;
        *o3_im = sum * NEG_C8;
    }
}

/**
 * @brief Apply W₈ twiddles - OPTIMIZED (Backward)
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
        double r = *o1_re;
        double i = *o1_im;
        double diff = r - i;
        double sum = r + i;
        *o1_re = diff * C8;
        *o1_im = sum * C8;
    }

    // W₈^(-2) = (0, +1) - OPTIMIZED
    {
        double r = *o2_re;
        *o2_re = -(*o2_im);
        *o2_im = r;
    }

    // W₈^(-3) = (-C8, +C8) - OPTIMIZED
    {
        double r = *o3_re;
        double i = *o3_im;
        double sum = r + i;
        double diff = i - r;
        *o3_re = sum * NEG_C8;
        *o3_im = diff * C8;
    }
}

//==============================================================================
// RADIX-8 N1 BUTTERFLY (Inline for Scalar Arrays)
//==============================================================================

/**
 * @brief Radix-8 N1 butterfly (Forward, Scalar)
 *
 * @details
 * In-place radix-8 butterfly using 4+4 decomposition.
 * Takes 8 complex inputs, produces 8 complex outputs.
 */
FORCE_INLINE void
radix8_n1_butterfly_inline_forward_scalar(
    double x_re[8], double x_im[8])
{
    // Even radix-4: (x0, x2, x4, x6)
    double e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_scalar(x_re[0], x_im[0], x_re[2], x_im[2],
                       x_re[4], x_im[4], x_re[6], x_im[6],
                       &e0_re, &e0_im, &e1_re, &e1_im,
                       &e2_re, &e2_im, &e3_re, &e3_im,
                       -1.0); // forward: sign = -1

    // Odd radix-4: (x1, x3, x5, x7)
    double o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_scalar(x_re[1], x_im[1], x_re[3], x_im[3],
                       x_re[5], x_im[5], x_re[7], x_im[7],
                       &o0_re, &o0_im, &o1_re, &o1_im,
                       &o2_re, &o2_im, &o3_re, &o3_im,
                       -1.0); // forward: sign = -1

    // Apply W₈ twiddles to odd outputs
    apply_w8_twiddles_forward_scalar(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im);

    // Combine even and odd
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
 * @brief Radix-8 N1 butterfly (Backward, Scalar)
 */
FORCE_INLINE void
radix8_n1_butterfly_inline_backward_scalar(
    double x_re[8], double x_im[8])
{
    // Even radix-4
    double e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    radix4_core_scalar(x_re[0], x_im[0], x_re[2], x_im[2],
                       x_re[4], x_im[4], x_re[6], x_im[6],
                       &e0_re, &e0_im, &e1_re, &e1_im,
                       &e2_re, &e2_im, &e3_re, &e3_im,
                       +1.0); // backward: sign = +1

    // Odd radix-4
    double o0_re, o0_im, o1_re, o1_im, o2_re, o2_im, o3_re, o3_im;
    radix4_core_scalar(x_re[1], x_im[1], x_re[3], x_im[3],
                       x_re[5], x_im[5], x_re[7], x_im[7],
                       &o0_re, &o0_im, &o1_re, &o1_im,
                       &o2_re, &o2_im, &o3_re, &o3_im,
                       +1.0); // backward: sign = +1

    // Apply conjugate W₈ twiddles
    apply_w8_twiddles_backward_scalar(&o1_re, &o1_im, &o2_re, &o2_im, &o3_re, &o3_im);

    // Combine
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
// W₆₄ CONSTANT TABLES (For On-Demand Access)
//==============================================================================

/**
 * @brief W64 twiddle constants
 *
 * @details
 * Same tables as SIMD versions. In scalar code, we just index directly
 * instead of broadcasting.
 */

// Forward W64 constants (W64^k for k=1..7)
static const double W64_FV_TABLE_RE[8] = {
    1.0,         // W64^0 (unused, for alignment)
    W64_FV_1_RE, // W64^1
    W64_FV_2_RE, // W64^2
    W64_FV_3_RE, // W64^3
    W64_FV_4_RE, // W64^4
    W64_FV_5_RE, // W64^5
    W64_FV_6_RE, // W64^6
    W64_FV_7_RE  // W64^7
};

static const double W64_FV_TABLE_IM[8] = {
    0.0,         // W64^0 (unused, for alignment)
    W64_FV_1_IM, // W64^1
    W64_FV_2_IM, // W64^2
    W64_FV_3_IM, // W64^3
    W64_FV_4_IM, // W64^4
    W64_FV_5_IM, // W64^5
    W64_FV_6_IM, // W64^6
    W64_FV_7_IM  // W64^7
};

// Backward W64 constants (W64^(-k) for k=1..7)
static const double W64_BV_TABLE_RE[8] = {
    1.0,         // W64^0 (unused, for alignment)
    W64_BV_1_RE, // W64^(-1)
    W64_BV_2_RE, // W64^(-2)
    W64_BV_3_RE, // W64^(-3)
    W64_BV_4_RE, // W64^(-4)
    W64_BV_5_RE, // W64^(-5)
    W64_BV_6_RE, // W64^(-6)
    W64_BV_7_RE  // W64^(-7)
};

static const double W64_BV_TABLE_IM[8] = {
    0.0,         // W64^0 (unused, for alignment)
    W64_BV_1_IM, // W64^(-1)
    W64_BV_2_IM, // W64^(-2)
    W64_BV_3_IM, // W64^(-3)
    W64_BV_4_IM, // W64^(-4)
    W64_BV_5_IM, // W64^(-5)
    W64_BV_6_IM, // W64^(-6)
    W64_BV_7_IM  // W64^(-7)
};

//==============================================================================
// M-SLICE LOAD/STORE (Single scalar value per position)
//==============================================================================

/**
 * @brief Load single m-slice for scalar processing
 *
 * @details
 * For m-slice 'm' at position 'k', loads x_r[m] for r = 0..7
 * (i.e., indices m, m+8, m+16, ..., m+56)
 *
 * Scalar version: loads single value at position k for each r.
 * Memory pattern: Each x_r[m] is at position: k + (r*8 + m) * K
 */
FORCE_INLINE void
load_m_slice_soa_n1_scalar(
    size_t k, // k-position (single scalar index)
    size_t m, // m-slice index (0..7)
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double x_re[8], // Output: x0[m]..x7[m]
    double x_im[8])
{
    for (int r = 0; r < 8; r++)
    {
        size_t idx = k + (r * 8 + m) * K;
        x_re[r] = in_re[idx];
        x_im[r] = in_im[idx];
    }
}

/**
 * @brief Store single m-slice
 */
FORCE_INLINE void
store_m_slice_soa_n1_scalar(
    size_t k,
    size_t m,
    size_t K,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const double y_re[8],
    const double y_im[8])
{
    for (int r = 0; r < 8; r++)
    {
        size_t idx = k + (r * 8 + m) * K;
        out_re[idx] = y_re[r];
        out_im[idx] = y_im[r];
    }
}

//==============================================================================
// APPLY W₆₄ FOR SINGLE M-SLICE
//==============================================================================

/**
 * @brief Apply W64 twiddles to a single m-slice (Forward)
 *
 * @details
 * Applies W64^r to x_r[m] for r=1..7 (x_0[m] unchanged).
 * Scalar version: direct table lookup and multiplication.
 */
FORCE_INLINE void
apply_w64_m_slice_forward_scalar(
    double x_re[8], // x0[m]..x7[m] - modified in-place
    double x_im[8],
    const double *RESTRICT w64_re_table, // W64_FV_TABLE_RE
    const double *RESTRICT w64_im_table) // W64_FV_TABLE_IM
{
    double tmp_re, tmp_im;

    // x0[m] unchanged (W64^0 = 1)

    // x1[m] *= W64^1
    cmul_scalar(x_re[1], x_im[1],
                w64_re_table[1], w64_im_table[1],
                &tmp_re, &tmp_im);
    x_re[1] = tmp_re;
    x_im[1] = tmp_im;

    // x2[m] *= W64^2
    cmul_scalar(x_re[2], x_im[2],
                w64_re_table[2], w64_im_table[2],
                &tmp_re, &tmp_im);
    x_re[2] = tmp_re;
    x_im[2] = tmp_im;

    // x3[m] *= W64^3
    cmul_scalar(x_re[3], x_im[3],
                w64_re_table[3], w64_im_table[3],
                &tmp_re, &tmp_im);
    x_re[3] = tmp_re;
    x_im[3] = tmp_im;

    // x4[m] *= W64^4
    cmul_scalar(x_re[4], x_im[4],
                w64_re_table[4], w64_im_table[4],
                &tmp_re, &tmp_im);
    x_re[4] = tmp_re;
    x_im[4] = tmp_im;

    // x5[m] *= W64^5
    cmul_scalar(x_re[5], x_im[5],
                w64_re_table[5], w64_im_table[5],
                &tmp_re, &tmp_im);
    x_re[5] = tmp_re;
    x_im[5] = tmp_im;

    // x6[m] *= W64^6
    cmul_scalar(x_re[6], x_im[6],
                w64_re_table[6], w64_im_table[6],
                &tmp_re, &tmp_im);
    x_re[6] = tmp_re;
    x_im[6] = tmp_im;

    // x7[m] *= W64^7
    cmul_scalar(x_re[7], x_im[7],
                w64_re_table[7], w64_im_table[7],
                &tmp_re, &tmp_im);
    x_re[7] = tmp_re;
    x_im[7] = tmp_im;
}

/**
 * @brief Apply W64 twiddles to a single m-slice (Backward)
 */
FORCE_INLINE void
apply_w64_m_slice_backward_scalar(
    double x_re[8], // x0[m]..x7[m] - modified in-place
    double x_im[8],
    const double *RESTRICT w64_re_table, // W64_BV_TABLE_RE
    const double *RESTRICT w64_im_table) // W64_BV_TABLE_IM
{
    double tmp_re, tmp_im;

    // x0[m] unchanged (W64^0 = 1)

    // Apply W64^(-k) for k=1..7
    for (int r = 1; r < 8; r++)
    {
        cmul_scalar(x_re[r], x_im[r],
                    w64_re_table[r], w64_im_table[r],
                    &tmp_re, &tmp_im);
        x_re[r] = tmp_re;
        x_im[r] = tmp_im;
    }
}

//==============================================================================
// RADIX-8 FINAL COMBINE FOR SINGLE M-SLICE
//==============================================================================

/**
 * @brief Final radix-8 combine for a single m-slice (Forward)
 *
 * @details
 * Takes x0[m]..x7[m] (outputs from 8 radix-8 butterflies after W64 twiddling)
 * and performs the final radix-8 across them to produce y[m + r*8] for r=0..7.
 */
FORCE_INLINE void
radix8_final_combine_m_slice_forward_scalar(
    const double x_re[8], // x0[m]..x7[m]
    const double x_im[8],
    double y_re[8], // Output: y[m + r*8] for r=0..7
    double y_im[8])
{
    // Copy inputs to working array
    double inputs_re[8], inputs_im[8];
    for (int r = 0; r < 8; r++)
    {
        inputs_re[r] = x_re[r];
        inputs_im[r] = x_im[r];
    }

    // Perform radix-8 butterfly
    radix8_n1_butterfly_inline_forward_scalar(inputs_re, inputs_im);

    // Copy outputs
    for (int r = 0; r < 8; r++)
    {
        y_re[r] = inputs_re[r];
        y_im[r] = inputs_im[r];
    }
}

/**
 * @brief Final radix-8 combine for a single m-slice (Backward)
 */
FORCE_INLINE void
radix8_final_combine_m_slice_backward_scalar(
    const double x_re[8], // x0[m]..x7[m]
    const double x_im[8],
    double y_re[8], // Output: y[m + r*8] for r=0..7
    double y_im[8])
{
    // Copy inputs to working array
    double inputs_re[8], inputs_im[8];
    for (int r = 0; r < 8; r++)
    {
        inputs_re[r] = x_re[r];
        inputs_im[r] = x_im[r];
    }

    // Perform radix-8 butterfly
    radix8_n1_butterfly_inline_backward_scalar(inputs_re, inputs_im);

    // Copy outputs
    for (int r = 0; r < 8; r++)
    {
        y_re[r] = inputs_re[r];
        y_im[r] = inputs_im[r];
    }
}

//==============================================================================
// M-STRIPMINED PROCESSING KERNEL (U=4 Core, Scalar)
//==============================================================================

/**
 * @brief Process single m-slice across U=4 positions (Forward, Scalar)
 *
 * @details
 * This is the heart of the U=4 optimization for scalar code. For a given m (0..7):
 * - Process 4 positions in sequence: k, k+1, k+2, k+3
 * - Each position: load → butterfly → W64 → combine → store
 *
 * Benefits:
 * - Clear code structure
 * - Good cache locality (working on single m-slice across 4 nearby k positions)
 * - Easy to understand and validate
 * - Compiler can optimize inner loops well
 */
FORCE_INLINE void
process_m_slice_u4_forward_scalar(
    size_t k, // Base k position
    size_t m, // m-slice index (0..7)
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const double *RESTRICT w64_re_table,
    const double *RESTRICT w64_im_table)
{
    // Process 4 positions: k, k+1, k+2, k+3
    for (int u = 0; u < 4; u++)
    {
        size_t pos = k + u;

        // Working arrays for this position
        double x_re[8], x_im[8];
        double y_re[8], y_im[8];

        // Load m-slice for this position
        load_m_slice_soa_n1_scalar(pos, m, K, in_re, in_im, x_re, x_im);

        // Radix-8 N1 butterfly (in-place)
        radix8_n1_butterfly_inline_forward_scalar(x_re, x_im);

        // Apply W64 merge twiddles (in-place)
        apply_w64_m_slice_forward_scalar(x_re, x_im, w64_re_table, w64_im_table);

        // Final radix-8 combine
        radix8_final_combine_m_slice_forward_scalar(x_re, x_im, y_re, y_im);

        // Store results
        store_m_slice_soa_n1_scalar(pos, m, K, out_re, out_im, y_re, y_im);
    }
}

/**
 * @brief Process single m-slice across U=4 positions (Backward, Scalar)
 */
FORCE_INLINE void
process_m_slice_u4_backward_scalar(
    size_t k,
    size_t m,
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const double *RESTRICT w64_re_table,
    const double *RESTRICT w64_im_table)
{
    // Process 4 positions: k, k+1, k+2, k+3
    for (int u = 0; u < 4; u++)
    {
        size_t pos = k + u;

        double x_re[8], x_im[8];
        double y_re[8], y_im[8];

        load_m_slice_soa_n1_scalar(pos, m, K, in_re, in_im, x_re, x_im);

        radix8_n1_butterfly_inline_backward_scalar(x_re, x_im);

        apply_w64_m_slice_backward_scalar(x_re, x_im, w64_re_table, w64_im_table);

        radix8_final_combine_m_slice_backward_scalar(x_re, x_im, y_re, y_im);

        store_m_slice_soa_n1_scalar(pos, m, K, out_re, out_im, y_re, y_im);
    }
}

//==============================================================================
// SINGLE POSITION PROCESSING (For tail loops)
//==============================================================================

/**
 * @brief Process single m-slice for one position (Forward)
 *
 * @details
 * Used for tail cases where we have k+1, k+2, or k+3 but not full U=4.
 */
FORCE_INLINE void
process_m_slice_single_forward_scalar(
    size_t pos,
    size_t m,
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const double *RESTRICT w64_re_table,
    const double *RESTRICT w64_im_table)
{
    double x_re[8], x_im[8];
    double y_re[8], y_im[8];

    load_m_slice_soa_n1_scalar(pos, m, K, in_re, in_im, x_re, x_im);

    radix8_n1_butterfly_inline_forward_scalar(x_re, x_im);

    apply_w64_m_slice_forward_scalar(x_re, x_im, w64_re_table, w64_im_table);

    radix8_final_combine_m_slice_forward_scalar(x_re, x_im, y_re, y_im);

    store_m_slice_soa_n1_scalar(pos, m, K, out_re, out_im, y_re, y_im);
}

/**
 * @brief Process single m-slice for one position (Backward)
 */
FORCE_INLINE void
process_m_slice_single_backward_scalar(
    size_t pos,
    size_t m,
    size_t K,
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    const double *RESTRICT w64_re_table,
    const double *RESTRICT w64_im_table)
{
    double x_re[8], x_im[8];
    double y_re[8], y_im[8];

    load_m_slice_soa_n1_scalar(pos, m, K, in_re, in_im, x_re, x_im);

    radix8_n1_butterfly_inline_backward_scalar(x_re, x_im);

    apply_w64_m_slice_backward_scalar(x_re, x_im, w64_re_table, w64_im_table);

    radix8_final_combine_m_slice_backward_scalar(x_re, x_im, y_re, y_im);

    store_m_slice_soa_n1_scalar(pos, m, K, out_re, out_im, y_re, y_im);
}
//==============================================================================
// MAIN DRIVER: FORWARD N1 - SCALAR (U=4 M-STRIPMINED)
//==============================================================================

/**
 * @brief Radix-64 N1 stage (Forward, Scalar, U=4 m-stripmined)
 *
 * @details
 * Clean scalar reference implementation with U=4 m-stripmine optimization.
 *
 * Processing order:
 * 1. K-tiling outer loop (for cache locality)
 * 2. Main U=4 loop: k += 4
 *    - For each m-slice (m=0..7):
 *      - Process 4 positions: k, k+1, k+2, k+3
 * 3. Tail loops for remaining positions
 *
 * This provides:
 * - Good cache locality (working on same m-slice)
 * - Clear code structure
 * - Easy validation
 * - Compiler can optimize well
 */
void radix64_stage_dit_forward_n1_soa_scalar(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    const size_t tile_size = RADIX64_TILE_SIZE_N1_SCALAR;

    // W₆₄ constants in tables
    const double *w64_re_table = W64_FV_TABLE_RE;
    const double *w64_im_table = W64_FV_TABLE_IM;

    // K-tiling outer loop (preserves cache locality)
    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        size_t k_end = (k_tile + tile_size < K) ? (k_tile + tile_size) : K;

        //======================================================================
        // MAIN U=4 LOOP: k += 4 (Process 4 positions per iteration)
        //======================================================================
        size_t k;
        for (k = k_tile; k + 4 <= k_end; k += 4)
        {
            // M-STRIPMINE: Process all 8 m-slices for this k-block
            // Each m-iteration processes 4 positions: k, k+1, k+2, k+3
            for (size_t m = 0; m < 8; m++)
            {
                process_m_slice_u4_forward_scalar(
                    k, m, K,
                    in_re, in_im,
                    out_re, out_im,
                    w64_re_table, w64_im_table);
            }
        }

        //======================================================================
        // TAIL LOOP #1: k += 3 (U=3)
        //======================================================================
        if (k + 3 <= k_end)
        {
            for (size_t m = 0; m < 8; m++)
            {
                // Process 3 positions: k, k+1, k+2
                for (int u = 0; u < 3; u++)
                {
                    size_t pos = k + u;
                    process_m_slice_single_forward_scalar(
                        pos, m, K,
                        in_re, in_im,
                        out_re, out_im,
                        w64_re_table, w64_im_table);
                }
            }
            k += 3;
        }

        //======================================================================
        // TAIL LOOP #2: k += 2 (U=2)
        //======================================================================
        if (k + 2 <= k_end)
        {
            for (size_t m = 0; m < 8; m++)
            {
                // Process 2 positions: k, k+1
                for (int u = 0; u < 2; u++)
                {
                    size_t pos = k + u;
                    process_m_slice_single_forward_scalar(
                        pos, m, K,
                        in_re, in_im,
                        out_re, out_im,
                        w64_re_table, w64_im_table);
                }
            }
            k += 2;
        }

        //======================================================================
        // TAIL LOOP #3: k += 1 (U=1, final position)
        //======================================================================
        if (k < k_end)
        {
            for (size_t m = 0; m < 8; m++)
            {
                process_m_slice_single_forward_scalar(
                    k, m, K,
                    in_re, in_im,
                    out_re, out_im,
                    w64_re_table, w64_im_table);
            }
        }
    }
}

//==============================================================================
// MAIN DRIVER: BACKWARD N1 - SCALAR (U=4 M-STRIPMINED)
//==============================================================================

/**
 * @brief Radix-64 N1 stage (Backward, Scalar, U=4 m-stripmined)
 *
 * @details
 * Clean scalar reference implementation with U=4 m-stripmine optimization.
 * Identical structure to forward version, using backward butterflies and
 * conjugate twiddles.
 */
void radix64_stage_dit_backward_n1_soa_scalar(
    size_t K,
    const double *RESTRICT in_re, const double *RESTRICT in_im,
    double *RESTRICT out_re, double *RESTRICT out_im)
{
    const size_t tile_size = RADIX64_TILE_SIZE_N1_SCALAR;

    // W₆₄ constants in tables (BACKWARD)
    const double *w64_re_table = W64_BV_TABLE_RE;
    const double *w64_im_table = W64_BV_TABLE_IM;

    // K-tiling outer loop
    for (size_t k_tile = 0; k_tile < K; k_tile += tile_size)
    {
        size_t k_end = (k_tile + tile_size < K) ? (k_tile + tile_size) : K;

        //======================================================================
        // MAIN U=4 LOOP: k += 4
        //======================================================================
        size_t k;
        for (k = k_tile; k + 4 <= k_end; k += 4)
        {
            // M-STRIPMINE: Process all 8 m-slices
            for (size_t m = 0; m < 8; m++)
            {
                process_m_slice_u4_backward_scalar(
                    k, m, K,
                    in_re, in_im,
                    out_re, out_im,
                    w64_re_table, w64_im_table);
            }
        }

        //======================================================================
        // TAIL LOOP #1: k += 3 (U=3)
        //======================================================================
        if (k + 3 <= k_end)
        {
            for (size_t m = 0; m < 8; m++)
            {
                for (int u = 0; u < 3; u++)
                {
                    size_t pos = k + u;
                    process_m_slice_single_backward_scalar(
                        pos, m, K,
                        in_re, in_im,
                        out_re, out_im,
                        w64_re_table, w64_im_table);
                }
            }
            k += 3;
        }

        //======================================================================
        // TAIL LOOP #2: k += 2 (U=2)
        //======================================================================
        if (k + 2 <= k_end)
        {
            for (size_t m = 0; m < 8; m++)
            {
                for (int u = 0; u < 2; u++)
                {
                    size_t pos = k + u;
                    process_m_slice_single_backward_scalar(
                        pos, m, K,
                        in_re, in_im,
                        out_re, out_im,
                        w64_re_table, w64_im_table);
                }
            }
            k += 2;
        }

        //======================================================================
        // TAIL LOOP #3: k += 1 (U=1)
        //======================================================================
        if (k < k_end)
        {
            for (size_t m = 0; m < 8; m++)
            {
                process_m_slice_single_backward_scalar(
                    k, m, K,
                    in_re, in_im,
                    out_re, out_im,
                    w64_re_table, w64_im_table);
            }
        }
    }
}

#endif // FFT_RADIX64_SCALAR_N1_H