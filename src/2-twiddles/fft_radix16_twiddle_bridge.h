/**
 * @file fft_radix16_twiddle_bridge.h
 * @brief Bridge between twiddle reorganization and radix-16 butterfly kernels
 *
 * @details
 * This header provides convenience functions to extract twiddle data from
 * materialized handles and pack them into the specific structures expected
 * by radix-16 AVX-512/AVX2/scalar butterflies.
 *
 * ARCHITECTURE:
 * =============
 * The bridge is a THIN ADAPTER that reads from the reorganization system
 * and packages data for butterfly consumption. It handles:
 * 
 * 1. Threshold-based layout selection (BLOCKED8 vs BLOCKED4)
 * 2. Pointer extraction from materialized arrays
 * 3. Delta_w extraction from layout_specific_data (for recurrence)
 * 4. Architecture-specific struct packaging (AVX-512/AVX2/scalar)
 *
 * USAGE PATTERN:
 * ==============
 * ```c
 * // Planning phase:
 * twiddle_handle_t *tw = get_stage_twiddles(N, 16, FFT_FORWARD);
 * twiddle_materialize_auto(tw, SIMD_ARCH_AVX512);
 *
 * // Execution phase (AVX-512):
 * radix16_stage_twiddles_blocked8_t tw_b8;
 * radix16_stage_twiddles_blocked4_avx512_t tw_b4;
 * 
 * int mode = radix16_prepare_twiddles_avx512(tw, &tw_b8, &tw_b4);
 * 
 * if (mode == RADIX16_TW_BLOCKED8) {
 *     radix16_stage_dit_forward_avx512(..., &tw_b8, mode);
 * } else {
 *     radix16_stage_dit_forward_avx512(..., &tw_b4, mode);
 * }
 *
 * // Cleanup:
 * twiddle_destroy(tw);
 * ```
 *
 * @author VectorFFT Team
 * @version 2.0 (Updated for reorganization system)
 * @date 2025
 */

#ifndef FFT_RADIX16_TWIDDLE_BRIDGE_H
#define FFT_RADIX16_TWIDDLE_BRIDGE_H

#include "fft_twiddles_reorganization.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

//==============================================================================
// THRESHOLDS (Must match reorganization.c)
//==============================================================================

#define RADIX16_BLOCKED8_THRESHOLD 512      // K ≤ 512: use BLOCKED8
#define RADIX16_RECURRENCE_THRESHOLD 4096   // K > 4096: enable recurrence

//==============================================================================
// BUTTERFLY STRUCTURE DEFINITIONS
//==============================================================================

/**
 * @brief BLOCKED8 twiddles for radix-16 (K ≤ 512)
 * 
 * Memory layout: [W1..W8[0..K-1]]
 * Simple flat pointers - butterflies use direct indexing
 */
typedef struct
{
    const double *re; ///< [8 * K] - all 8 twiddle factors
    const double *im; ///< [8 * K]
} radix16_stage_twiddles_blocked8_t;

/**
 * @brief BLOCKED4 twiddles for radix-16 AVX-512 (K > 512)
 * 
 * Memory layout: [W1..W4[0..K-1]]
 * Includes optional delta_w for recurrence-based twiddle computation
 */
#ifdef __AVX512F__
typedef struct
{
    const double *re;        ///< [4 * K] - 4 base twiddle factors
    const double *im;        ///< [4 * K]
    __m512d delta_w_re[15];  ///< Phase increments (if recurrence enabled)
    __m512d delta_w_im[15];  ///< Phase increments (if recurrence enabled)
    size_t K;                ///< Butterflies per stage
    bool recurrence_enabled; ///< Whether to use twiddle walking
} radix16_stage_twiddles_blocked4_avx512_t;
#endif

/**
 * @brief BLOCKED4 twiddles for radix-16 AVX2 (K > 512)
 */
#ifdef __AVX2__
typedef struct
{
    const double *re;
    const double *im;
    __m256d delta_w_re[15];
    __m256d delta_w_im[15];
    size_t K;
    bool recurrence_enabled;
} radix16_stage_twiddles_blocked4_avx2_t;
#endif

/**
 * @brief BLOCKED4 twiddles for radix-16 scalar (K > 512)
 */
typedef struct
{
    const double *re;
    const double *im;
    double delta_w_re[15];
    double delta_w_im[15];
    size_t K;
    bool recurrence_enabled;
} radix16_stage_twiddles_blocked4_scalar_t;

/**
 * @brief Twiddle mode enum (returned by bridge functions)
 */
typedef enum
{
    RADIX16_TW_BLOCKED8,  ///< 8 twiddles stored, derive 7 at runtime
    RADIX16_TW_BLOCKED4   ///< 4 twiddles stored, derive 11 at runtime
} radix16_twiddle_mode_t;

//==============================================================================
// BLOCKED4 LAYOUT-SPECIFIC DATA STRUCTURE
//==============================================================================

/**
 * @brief Layout-specific data for BLOCKED4 with recurrence
 * 
 * @details
 * This structure is stored in handle->layout_specific_data when
 * BLOCKED4 layout with recurrence is materialized. The bridge extracts
 * this data and copies it into butterfly-specific structs.
 */
#ifdef __AVX512F__
typedef struct
{
    __m512d delta_w_re[15];  ///< exp(-2πi × s / N) for s=1..15 (AVX-512)
    __m512d delta_w_im[15];
} radix16_blocked4_recurrence_data_avx512_t;
#endif

#ifdef __AVX2__
typedef struct
{
    __m256d delta_w_re[15];  ///< exp(-2πi × s / N) for s=1..15 (AVX2)
    __m256d delta_w_im[15];
} radix16_blocked4_recurrence_data_avx2_t;
#endif

typedef struct
{
    double delta_w_re[15];  ///< exp(-2πi × s / N) for s=1..15 (scalar)
    double delta_w_im[15];
} radix16_blocked4_recurrence_data_scalar_t;

//==============================================================================
// AVX-512 BRIDGE FUNCTIONS
//==============================================================================

#ifdef __AVX512F__

/**
 * @brief Prepare radix-16 twiddles for AVX-512 execution
 *
 * @param handle Materialized twiddle handle (from planner)
 * @param tw_b8 Output: BLOCKED8 structure (populated if K ≤ 512)
 * @param tw_b4 Output: BLOCKED4 structure (populated if K > 512)
 *
 * @return Twiddle mode (RADIX16_TW_BLOCKED8 or RADIX16_TW_BLOCKED4), or -1 on error
 *
 * @note Both output pointers should be provided; function populates the relevant one
 */
static inline int radix16_prepare_twiddles_avx512(
    const twiddle_handle_t *handle,
    radix16_stage_twiddles_blocked8_t *tw_b8,
    radix16_stage_twiddles_blocked4_avx512_t *tw_b4)
{
    // Validate inputs
    if (!handle || handle->radix != 16)
        return -1;
    if (!handle->materialized_re || !handle->materialized_im)
        return -1;
    if (!tw_b8 || !tw_b4)
        return -1;

    int K = handle->layout_desc.butterflies_per_stage;

    // ──────────────────────────────────────────────────────────────────
    // BLOCKED8: K ≤ 512 (stores W1..W8, derive W9..W15 at runtime)
    // ──────────────────────────────────────────────────────────────────
    if (K <= RADIX16_BLOCKED8_THRESHOLD)
    {
        tw_b8->re = handle->materialized_re;
        tw_b8->im = handle->materialized_im;

        return RADIX16_TW_BLOCKED8;
    }

    // ──────────────────────────────────────────────────────────────────
    // BLOCKED4: K > 512 (stores W1..W4, derive W5..W15 at runtime)
    // ──────────────────────────────────────────────────────────────────
    else
    {
        tw_b4->re = handle->materialized_re;
        tw_b4->im = handle->materialized_im;
        tw_b4->K = K;

        // Check if recurrence is enabled (K > 4096)
        if (K > RADIX16_RECURRENCE_THRESHOLD && handle->layout_specific_data)
        {
            // Extract delta_w from layout_specific_data
            const radix16_blocked4_recurrence_data_avx512_t *recur_data =
                (const radix16_blocked4_recurrence_data_avx512_t *)handle->layout_specific_data;

            tw_b4->recurrence_enabled = true;

            for (int i = 0; i < 15; i++)
            {
                tw_b4->delta_w_re[i] = recur_data->delta_w_re[i];
                tw_b4->delta_w_im[i] = recur_data->delta_w_im[i];
            }
        }
        else
        {
            // No recurrence - zero out delta_w
            tw_b4->recurrence_enabled = false;

            for (int i = 0; i < 15; i++)
            {
                tw_b4->delta_w_re[i] = _mm512_setzero_pd();
                tw_b4->delta_w_im[i] = _mm512_setzero_pd();
            }
        }

        return RADIX16_TW_BLOCKED4;
    }
}

#endif // __AVX512F__

//==============================================================================
// AVX2 BRIDGE FUNCTIONS
//==============================================================================

#ifdef __AVX2__

/**
 * @brief Prepare radix-16 twiddles for AVX2 execution
 */
static inline int radix16_prepare_twiddles_avx2(
    const twiddle_handle_t *handle,
    radix16_stage_twiddles_blocked8_t *tw_b8,
    radix16_stage_twiddles_blocked4_avx2_t *tw_b4)
{
    if (!handle || handle->radix != 16)
        return -1;
    if (!handle->materialized_re || !handle->materialized_im)
        return -1;
    if (!tw_b8 || !tw_b4)
        return -1;

    int K = handle->layout_desc.butterflies_per_stage;

    if (K <= RADIX16_BLOCKED8_THRESHOLD)
    {
        tw_b8->re = handle->materialized_re;
        tw_b8->im = handle->materialized_im;

        return RADIX16_TW_BLOCKED8;
    }
    else
    {
        tw_b4->re = handle->materialized_re;
        tw_b4->im = handle->materialized_im;
        tw_b4->K = K;

        if (K > RADIX16_RECURRENCE_THRESHOLD && handle->layout_specific_data)
        {
            const radix16_blocked4_recurrence_data_avx2_t *recur_data =
                (const radix16_blocked4_recurrence_data_avx2_t *)handle->layout_specific_data;

            tw_b4->recurrence_enabled = true;

            for (int i = 0; i < 15; i++)
            {
                tw_b4->delta_w_re[i] = recur_data->delta_w_re[i];
                tw_b4->delta_w_im[i] = recur_data->delta_w_im[i];
            }
        }
        else
        {
            tw_b4->recurrence_enabled = false;

            for (int i = 0; i < 15; i++)
            {
                tw_b4->delta_w_re[i] = _mm256_setzero_pd();
                tw_b4->delta_w_im[i] = _mm256_setzero_pd();
            }
        }

        return RADIX16_TW_BLOCKED4;
    }
}

#endif // __AVX2__

//==============================================================================
// SCALAR BRIDGE FUNCTIONS
//==============================================================================

/**
 * @brief Prepare radix-16 twiddles for scalar execution
 */
static inline int radix16_prepare_twiddles_scalar(
    const twiddle_handle_t *handle,
    radix16_stage_twiddles_blocked8_t *tw_b8,
    radix16_stage_twiddles_blocked4_scalar_t *tw_b4)
{
    if (!handle || handle->radix != 16)
        return -1;
    if (!handle->materialized_re || !handle->materialized_im)
        return -1;
    if (!tw_b8 || !tw_b4)
        return -1;

    int K = handle->layout_desc.butterflies_per_stage;

    if (K <= RADIX16_BLOCKED8_THRESHOLD)
    {
        tw_b8->re = handle->materialized_re;
        tw_b8->im = handle->materialized_im;

        return RADIX16_TW_BLOCKED8;
    }
    else
    {
        tw_b4->re = handle->materialized_re;
        tw_b4->im = handle->materialized_im;
        tw_b4->K = K;

        if (K > RADIX16_RECURRENCE_THRESHOLD && handle->layout_specific_data)
        {
            const radix16_blocked4_recurrence_data_scalar_t *recur_data =
                (const radix16_blocked4_recurrence_data_scalar_t *)handle->layout_specific_data;

            tw_b4->recurrence_enabled = true;

            for (int i = 0; i < 15; i++)
            {
                tw_b4->delta_w_re[i] = recur_data->delta_w_re[i];
                tw_b4->delta_w_im[i] = recur_data->delta_w_im[i];
            }
        }
        else
        {
            tw_b4->recurrence_enabled = false;

            for (int i = 0; i < 15; i++)
            {
                tw_b4->delta_w_re[i] = 0.0;
                tw_b4->delta_w_im[i] = 0.0;
            }
        }

        return RADIX16_TW_BLOCKED4;
    }
}

//==============================================================================
// GENERIC WRAPPER (Runtime dispatch)
//==============================================================================

/**
 * @brief Prepare twiddles with runtime architecture detection
 *
 * @details
 * Convenience wrapper that detects SIMD capabilities at compile time
 * and calls the appropriate architecture-specific function.
 *
 * @note Less type-safe than direct calls - use architecture-specific
 *       functions when possible for better compile-time checking
 */
static inline int radix16_prepare_twiddles_auto(
    const twiddle_handle_t *handle,
    void *tw_b8,
    void *tw_b4)
{
#ifdef __AVX512F__
    return radix16_prepare_twiddles_avx512(
        handle,
        (radix16_stage_twiddles_blocked8_t *)tw_b8,
        (radix16_stage_twiddles_blocked4_avx512_t *)tw_b4);
#elif defined(__AVX2__)
    return radix16_prepare_twiddles_avx2(
        handle,
        (radix16_stage_twiddles_blocked8_t *)tw_b8,
        (radix16_stage_twiddles_blocked4_avx2_t *)tw_b4);
#else
    return radix16_prepare_twiddles_scalar(
        handle,
        (radix16_stage_twiddles_blocked8_t *)tw_b8,
        (radix16_stage_twiddles_blocked4_scalar_t *)tw_b4);
#endif
}

#endif // FFT_RADIX16_TWIDDLE_BRIDGE_H