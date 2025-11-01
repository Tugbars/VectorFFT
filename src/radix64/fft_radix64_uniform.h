/**
 * @file fft_radix64_uniform.h
 * @brief Radix-64 FFT Uniform Public API
 * 
 * @details
 * Public API for radix-64 FFT transforms (forward and backward).
 * 
 * VARIANTS:
 * - Standard: fft_radix64_fv() / fft_radix64_bv() (WITH stage twiddles)
 * - N1 (twiddle-less): fft_radix64_fv_n1() / fft_radix64_bv_n1() (NO stage twiddles)
 * 
 * DATA LAYOUT:
 * - Native SoA (Structure of Arrays)
 * - Input/output arrays must be aligned (64-byte for AVX-512, 32-byte for AVX-2)
 * 
 * @author VectorFFT Team
 * @version 1.0
 * @date 2025
 */

#ifndef FFT_RADIX64_UNIFORM_H
#define FFT_RADIX64_UNIFORM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// RADIX-64 FORWARD TRANSFORMS
//==============================================================================

/**
 * @brief Radix-64 forward FFT - WITH stage twiddles
 * 
 * @param[out] out_re Output real array (N=64K elements, SoA)
 * @param[out] out_im Output imaginary array (N=64K elements, SoA)
 * @param[in] in_re Input real array (N=64K elements, SoA)
 * @param[in] in_im Input imaginary array (N=64K elements, SoA)
 * @param[in] stage_tw_blocked4 Blocked4 stage twiddles (for K ≤ threshold)
 * @param[in] stage_tw_blocked2 Blocked2 stage twiddles (for K > threshold)
 * @param[in] K Transform 64th-size (N/64)
 */
void fft_radix64_fv(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const void *restrict stage_tw_blocked4,  // TBD: proper struct type
    const void *restrict stage_tw_blocked2,  // TBD: proper struct type
    int K);

/**
 * @brief Radix-64 forward FFT - NO stage twiddles (N1)
 * 
 * @details
 * Twiddle-less variant: 40-50% faster than standard version.
 * Use for first stage or when all stage twiddles are unity.
 * 
 * @param[out] out_re Output real array (N=64K elements, SoA)
 * @param[out] out_im Output imaginary array (N=64K elements, SoA)
 * @param[in] in_re Input real array (N=64K elements, SoA)
 * @param[in] in_im Input imaginary array (N=64K elements, SoA)
 * @param[in] K Transform 64th-size (N/64)
 */
void fft_radix64_fv_n1(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    int K);

//==============================================================================
// RADIX-64 BACKWARD TRANSFORMS
//==============================================================================

/**
 * @brief Radix-64 backward FFT - WITH stage twiddles
 * 
 * @param[out] out_re Output real array (N=64K elements, SoA)
 * @param[out] out_im Output imaginary array (N=64K elements, SoA)
 * @param[in] in_re Input real array (N=64K elements, SoA)
 * @param[in] in_im Input imaginary array (N=64K elements, SoA)
 * @param[in] stage_tw_blocked4 Blocked4 stage twiddles (for K ≤ threshold)
 * @param[in] stage_tw_blocked2 Blocked2 stage twiddles (for K > threshold)
 * @param[in] K Transform 64th-size (N/64)
 */
void fft_radix64_bv(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const void *restrict stage_tw_blocked4,  // TBD: proper struct type
    const void *restrict stage_tw_blocked2,  // TBD: proper struct type
    int K);

/**
 * @brief Radix-64 backward FFT - NO stage twiddles (N1)
 * 
 * @details
 * Twiddle-less variant: 40-50% faster than standard version.
 * Use for first stage (IFFT) or when all stage twiddles are unity.
 * 
 * @param[out] out_re Output real array (N=64K elements, SoA)
 * @param[out] out_im Output imaginary array (N=64K elements, SoA)
 * @param[in] in_re Input real array (N=64K elements, SoA)
 * @param[in] in_im Input imaginary array (N=64K elements, SoA)
 * @param[in] K Transform 64th-size (N/64)
 */
void fft_radix64_bv_n1(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    int K);

#ifdef __cplusplus
}
#endif

#endif // FFT_RADIX64_UNIFORM_H