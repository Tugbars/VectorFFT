/**
 * @file fft_radix16_uniform_optimized.h
 * @brief Unified interface for radix-16 FFT operations
 *
 * @details
 * Provides a consistent API for radix-16 transforms across all SIMD levels.
 * Supports hybrid blocked twiddle system and twiddle-less (N1) variants.
 *
 * @author VectorFFT Team
 * @version 5.0
 * @date 2025
 */

#ifndef FFT_RADIX16_UNIFORM_OPTIMIZED_H
#define FFT_RADIX16_UNIFORM_OPTIMIZED_H

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

//==============================================================================
// CONFIGURATION
//==============================================================================

/**
 * @def RADIX16_BLOCKED8_THRESHOLD
 * @brief K threshold for BLOCKED8 vs BLOCKED4
 */
#ifndef RADIX16_BLOCKED8_THRESHOLD
#define RADIX16_BLOCKED8_THRESHOLD 512
#endif

//==============================================================================
// TWIDDLE STRUCTURES
//==============================================================================

/**
 * @brief BLOCKED8 twiddle structure (for K > threshold)
 *
 * Stores 8 twiddle factor blocks (W1, W2, ..., W8).
 * W9=-W1, ..., W15=-W7 are derived via sign flips.
 */
typedef struct
{
    const double *restrict re; ///< Real components (8*K elements)
    const double *restrict im; ///< Imaginary components (8*K elements)
} radix16_stage_twiddles_blocked8_t;

/**
 * @brief BLOCKED4 twiddle structure (for K ≤ threshold)
 *
 * Stores 4 twiddle factor blocks (W1, W2, W3, W4).
 * W5=W1×W4, W6=W2×W4, W7=W3×W4, W8=W4² are derived via FMA.
 * W9=-W1, ..., W15=-W7 are derived via sign flips.
 */
typedef struct
{
    const double *restrict re; ///< Real components (4*K elements)
    const double *restrict im; ///< Imaginary components (4*K elements)
} radix16_stage_twiddles_blocked4_t;

/**
 * @brief Twiddle mode selector
 */
typedef enum
{
    RADIX16_TW_BLOCKED4, ///< Use BLOCKED4 mode (K ≤ threshold)
    RADIX16_TW_BLOCKED8  ///< Use BLOCKED8 mode (K > threshold)
} radix16_twiddle_mode_t;

//==============================================================================
// PLANNING HELPER
//==============================================================================

/**
 * @brief Choose twiddle mode based on K
 *
 * @param K Transform sixteenth-size
 * @return RADIX16_TW_BLOCKED4 if K ≤ 512, else RADIX16_TW_BLOCKED8
 */
static inline radix16_twiddle_mode_t
radix16_choose_twiddle_mode(size_t K)
{
    return (K <= RADIX16_BLOCKED8_THRESHOLD) ? RADIX16_TW_BLOCKED4 : RADIX16_TW_BLOCKED8;
}

//==============================================================================
// FORWARD FFT (WITH TWIDDLES)
//==============================================================================

/**
 * @brief Radix-16 forward FFT with hybrid blocked twiddles
 *
 * @param[out] out_re Output real array (16*K elements, SoA)
 * @param[out] out_im Output imaginary array (16*K elements, SoA)
 * @param[in] in_re Input real array (16*K elements, SoA)
 * @param[in] in_im Input imaginary array (16*K elements, SoA)
 * @param[in] stage_tw_blocked8 BLOCKED8 twiddles (pass NULL if K ≤ threshold)
 * @param[in] stage_tw_blocked4 BLOCKED4 twiddles (pass NULL if K > threshold)
 * @param[in] K Transform sixteenth-size
 */
void fft_radix16_fv(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const radix16_stage_twiddles_blocked8_t *restrict stage_tw_blocked8,
    const radix16_stage_twiddles_blocked4_t *restrict stage_tw_blocked4,
    int K);

//==============================================================================
// FORWARD FFT (NO TWIDDLES - N1)
//==============================================================================

/**
 * @brief Radix-16 forward FFT without twiddles (first stage optimization)
 *
 * @param[out] out_re Output real array (16*K elements, SoA)
 * @param[out] out_im Output imaginary array (16*K elements, SoA)
 * @param[in] in_re Input real array (16*K elements, SoA)
 * @param[in] in_im Input imaginary array (16*K elements, SoA)
 * @param[in] K Transform sixteenth-size
 */
void fft_radix16_fv_n1(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    int K);

//==============================================================================
// BACKWARD FFT (WITH TWIDDLES)
//==============================================================================

/**
 * @brief Radix-16 backward FFT with hybrid blocked twiddles
 *
 * @param[out] out_re Output real array (16*K elements, SoA)
 * @param[out] out_im Output imaginary array (16*K elements, SoA)
 * @param[in] in_re Input real array (16*K elements, SoA)
 * @param[in] in_im Input imaginary array (16*K elements, SoA)
 * @param[in] stage_tw_blocked8 BLOCKED8 twiddles (pass NULL if K ≤ threshold)
 * @param[in] stage_tw_blocked4 BLOCKED4 twiddles (pass NULL if K > threshold)
 * @param[in] K Transform sixteenth-size
 */
void fft_radix16_bv(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const radix16_stage_twiddles_blocked8_t *restrict stage_tw_blocked8,
    const radix16_stage_twiddles_blocked4_t *restrict stage_tw_blocked4,
    int K);

//==============================================================================
// BACKWARD FFT (NO TWIDDLES - N1)
//==============================================================================

/**
 * @brief Radix-16 backward FFT without twiddles (first stage optimization)
 *
 * @param[out] out_re Output real array (16*K elements, SoA)
 * @param[out] out_im Output imaginary array (16*K elements, SoA)
 * @param[in] in_re Input real array (16*K elements, SoA)
 * @param[in] in_im Input imaginary array (16*K elements, SoA)
 * @param[in] K Transform sixteenth-size
 */
void fft_radix16_bv_n1(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    int K);

#ifdef __cplusplus
}
#endif

#endif // FFT_RADIX16_UNIFORM_OPTIMIZED_H