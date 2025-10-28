//==============================================================================
// fft_rader_plans.h - Rader Algorithm Global Cache (Pure SoA)
//==============================================================================

/**
 * @file fft_rader_plans.h
 * @brief SINGLE SOURCE OF TRUTH for Rader convolution twiddles (SoA format)
 * 
 * Global cache: One plan per prime, shared across all stages.
 * Pure SoA layout eliminates shuffle overhead in butterflies.
 */

#ifndef FFT_RADER_PLANS_H
#define FFT_RADER_PLANS_H

#include "fft_planning_types.h"

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// API - Rader Manager (SoA)
//==============================================================================

/**
 * @brief Get Rader convolution twiddles in SoA format (thread-safe, lazy)
 * 
 * Returns pre-computed convolution twiddles for Rader's algorithm in
 * pure Structure-of-Arrays format. Uses global cache - same plan shared
 * by all stages with same prime.
 * 
 * **Formula:**
 * ```
 * For prime p:
 *   FORWARD:  tw->re[q] = cos(-2π × perm_out[q] / p)
 *             tw->im[q] = sin(-2π × perm_out[q] / p)
 *   INVERSE:  tw->re[q] = cos(+2π × perm_out[q] / p)
 *             tw->im[q] = sin(+2π × perm_out[q] / p)
 * where q = 0..(p-2)
 * ```
 * 
 * **SoA Benefits:**
 * - Zero shuffle overhead: Direct vector loads
 * - 5-10% faster Rader butterflies
 * - Example: `__m256d w_re = _mm256_load_pd(&tw->re[k]);`
 * 
 * @param prime Prime radix (7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67)
 * @param direction FFT_FORWARD or FFT_INVERSE
 * @return Pointer to SoA twiddles [prime-1 elements] from global cache, or NULL on failure
 * 
 * @note Returned pointer BORROWED from cache - do NOT free!
 * @note Thread-safe with lazy initialization
 */
const fft_twiddles_soa* get_rader_twiddles_soa(int prime, fft_direction_t direction);

/**
 * @brief Initialize Rader cache with common primes (optional, thread-safe)
 * 
 * Pre-populates cache with primes 7, 11, 13. Other primes lazy-initialized.
 * Optional call - fft_init() calls automatically on first Rader use.
 * Useful for avoiding first-use latency in time-critical code.
 */
void init_rader_cache(void);

/**
 * @brief Cleanup Rader cache and free all resources (thread-safe)
 * 
 * Call at program exit after all FFT plans freed.
 * 
 * @warning NOT safe while FFT plans exist or operations in progress!
 */
void cleanup_rader_cache(void);

//==============================================================================
// Optional - Permutation Access (For Debugging)
//==============================================================================

/**
 * @brief Get input permutation: [g^0, g^1, ..., g^(P-2)] mod P
 * 
 * @param prime Prime radix
 * @return Pointer to permutation [prime-1 elements], or NULL on failure
 * 
 * @note For debugging/verification only - most butterflies hardcode permutations
 */
const int* get_rader_input_perm(int prime);

/**
 * @brief Get output permutation: inverse of input permutation
 * 
 * @param prime Prime radix
 * @return Pointer to permutation [prime-1 elements], or NULL on failure
 * 
 * @note For debugging/verification only
 */
const int* get_rader_output_perm(int prime);

#ifdef __cplusplus
}
#endif

#endif // FFT_RADER_PLANS_H