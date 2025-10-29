/**
 * @file fft_radix8_uniform.h
 * @brief Unified interface for radix-8 FFT operations
 *
 * @details
 * Provides a consistent API for radix-8 transforms across all SIMD levels.
 * Supports both hybrid blocked twiddle system and twiddle-less (N1) variants.
 *
 * @author Tugbars
 * @version 3.1
 * @date 2025
 */

#ifndef FFT_RADIX8_UNIFORM_H
#define FFT_RADIX8_UNIFORM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

    //==============================================================================
    // TWIDDLE STRUCTURES
    //==============================================================================

    /**
     * @brief BLOCKED4 twiddle structure (for K ≤ 256)
     *
     * Stores 4 twiddle factor blocks (W1, W2, W3, W4).
     * W5=-W1, W6=-W2, W7=-W3 are derived via sign flips.
     */
    typedef struct
    {
        const double *restrict re; ///< Real components (4*K elements)
        const double *restrict im; ///< Imaginary components (4*K elements)
    } radix8_stage_twiddles_blocked4_t;

    /**
     * @brief BLOCKED2 twiddle structure (for K > 256)
     *
     * Stores 2 twiddle factor blocks (W1, W2).
     * W3=W1×W2, W4=W2² are derived via FMA operations.
     * W5=-W1, W6=-W2, W7=-W3 are derived via sign flips.
     */
    typedef struct
    {
        const double *restrict re; ///< Real components (2*K elements)
        const double *restrict im; ///< Imaginary components (2*K elements)
    } radix8_stage_twiddles_blocked2_t;

    /**
     * @brief Twiddle mode selector
     */
    typedef enum
    {
        RADIX8_TW_BLOCKED4, ///< Use BLOCKED4 mode (K ≤ 256)
        RADIX8_TW_BLOCKED2  ///< Use BLOCKED2 mode (K > 256)
    } radix8_twiddle_mode_t;

    //==============================================================================
    // FORWARD FFT (WITH TWIDDLES)
    //==============================================================================

    /**
     * @brief Radix-8 forward FFT with hybrid blocked twiddles
     *
     * @param[out] out_re Output real array (8*K elements, SoA)
     * @param[out] out_im Output imaginary array (8*K elements, SoA)
     * @param[in] in_re Input real array (8*K elements, SoA)
     * @param[in] in_im Input imaginary array (8*K elements, SoA)
     * @param[in] stage_tw_blocked4 BLOCKED4 twiddles (pass NULL if K > 256)
     * @param[in] stage_tw_blocked2 BLOCKED2 twiddles (pass NULL if K ≤ 256)
     * @param[in] K Transform eighth-size
     */
    void fft_radix8_fv(
        double *restrict out_re,
        double *restrict out_im,
        const double *restrict in_re,
        const double *restrict in_im,
        const radix8_stage_twiddles_blocked4_t *restrict stage_tw_blocked4,
        const radix8_stage_twiddles_blocked2_t *restrict stage_tw_blocked2,
        int K);

    //==============================================================================
    // FORWARD FFT (NO TWIDDLES - N1)
    //==============================================================================

    /**
     * @brief Radix-8 forward FFT without twiddles (first stage optimization)
     *
     * @param[out] out_re Output real array (8*K elements, SoA)
     * @param[out] out_im Output imaginary array (8*K elements, SoA)
     * @param[in] in_re Input real array (8*K elements, SoA)
     * @param[in] in_im Input imaginary array (8*K elements, SoA)
     * @param[in] K Transform eighth-size
     */
    void fft_radix8_fv_n1(
        double *restrict out_re,
        double *restrict out_im,
        const double *restrict in_re,
        const double *restrict in_im,
        int K);

    //==============================================================================
    // BACKWARD FFT (WITH TWIDDLES)
    //==============================================================================

    /**
     * @brief Radix-8 backward FFT with hybrid blocked twiddles
     *
     * @param[out] out_re Output real array (8*K elements, SoA)
     * @param[out] out_im Output imaginary array (8*K elements, SoA)
     * @param[in] in_re Input real array (8*K elements, SoA)
     * @param[in] in_im Input imaginary array (8*K elements, SoA)
     * @param[in] stage_tw_blocked4 BLOCKED4 twiddles (pass NULL if K > 256)
     * @param[in] stage_tw_blocked2 BLOCKED2 twiddles (pass NULL if K ≤ 256)
     * @param[in] K Transform eighth-size
     */
    void fft_radix8_bv(
        double *restrict out_re,
        double *restrict out_im,
        const double *restrict in_re,
        const double *restrict in_im,
        const radix8_stage_twiddles_blocked4_t *restrict stage_tw_blocked4,
        const radix8_stage_twiddles_blocked2_t *restrict stage_tw_blocked2,
        int K);

    //==============================================================================
    // BACKWARD FFT (NO TWIDDLES - N1)
    //==============================================================================

    /**
     * @brief Radix-8 backward FFT without twiddles (first stage optimization)
     *
     * @param[out] out_re Output real array (8*K elements, SoA)
     * @param[out] out_im Output imaginary array (8*K elements, SoA)
     * @param[in] in_re Input real array (8*K elements, SoA)
     * @param[in] in_im Input imaginary array (8*K elements, SoA)
     * @param[in] K Transform eighth-size
     */
    void fft_radix8_bv_n1(
        double *restrict out_re,
        double *restrict out_im,
        const double *restrict in_re,
        const double *restrict in_im,
        int K);

#ifdef __cplusplus
}
#endif

#endif // FFT_RADIX8_UNIFORM_H