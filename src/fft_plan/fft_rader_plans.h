// fft_rader_plans.c
// SINGLE SOURCE OF TRUTH for Rader convolution twiddles
// Global cache: One plan per prime, shared across all stages

#ifndef FFT_RADER_PLANS_H
#define FFT_RADER_PLANS_H

#include "fft_planning_types.h"

//==============================================================================
// API - Rader Manager
//==============================================================================

/**
 * @brief Get Rader convolution twiddles (SINGLE SOURCE OF TRUTH)
 * 
 * Returns pre-computed convolution twiddles for Rader's algorithm.
 * Uses global cache - same plan shared by all stages with same prime.
 * 
 * For prime p, computes:
 *   FORWARD:  conv_tw[q] = exp(-2πi * out_perm[q] / p)
 *   INVERSE:  conv_tw[q] = exp(+2πi * out_perm[q] / p)
 * 
 * @param prime Prime radix (7, 11, 13, etc.)
 * @param direction FORWARD or INVERSE
 * @return Pointer to (prime-1) convolution twiddles (from global cache)
 */
const fft_data* get_rader_twiddles(int prime, fft_direction_t direction);

/**
 * @brief Initialize Rader cache (called once at program startup)
 */
void init_rader_cache(void);

/**
 * @brief Cleanup Rader cache (called at program exit)
 */
void cleanup_rader_cache(void);

#endif // FFT_RADER_PLANS_H