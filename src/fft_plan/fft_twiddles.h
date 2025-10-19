// fft_twiddles.h
// TIER 1 TWIDDLES: Cooley-Tukey stage rotations
// (TIER 2 twiddles for Rader are in fft_rader_plans.h)

#ifndef FFT_TWIDDLES_H
#define FFT_TWIDDLES_H

#include "fft_planning_types.h"

//==============================================================================
// TIER 1: COOLEY-TUKEY STAGE TWIDDLES
//==============================================================================

/**
 * @brief Compute Cooley-Tukey stage twiddles (SINGLE SOURCE OF TRUTH)
 * 
 * **Purpose:** Inter-stage rotation twiddles for mixed-radix FFT
 * 
 * **Formula:**
 * tw[k*(radix-1) + (r-1)] = W_{N_stage}^(r×k)
 *                         = exp(sign × 2πi × r × k / N_stage)
 * 
 * where:
 * - sign = -1 for FFT_FORWARD, +1 for FFT_INVERSE
 * - N_stage = transform size at this stage (NOT radix!)
 * - r = 1..radix-1 (radix multiplier)
 * - k = 0..sub_len-1 (butterfly position)
 * - sub_len = N_stage / radix
 * 
 * **Layout:** Interleaved by butterfly position:
 * [W^(1×0), W^(2×0), ..., W^((R-1)×0),
 *  W^(1×1), W^(2×1), ..., W^((R-1)×1),
 *  ...
 *  W^(1×K), W^(2×K), ..., W^((R-1)×K)]
 * 
 * **Memory:** 32-byte aligned for AVX2, OWNED by stage descriptor
 * 
 * **Optimization:** Uses AVX2 + FMA when sub_len > 8
 * 
 * **Note:** For Rader stages (prime radix ≥7), this function computes
 * the CT twiddles. The Rader convolution twiddles come from a separate
 * global cache (see fft_rader_plans.h).
 * 
 * @param N_stage Transform size at this stage (e.g., 14 for N=14, stage 0)
 * @param radix Decomposition radix (2, 3, 4, 5, 7, 8, 11, 13, 16, 32)
 * @param direction FFT_FORWARD or FFT_INVERSE
 * @return Allocated array of (radix-1) × sub_len complex twiddles, or NULL on error
 */
fft_data* compute_stage_twiddles(
    int N_stage,
    int radix,
    fft_direction_t direction
);

/**
 * @brief Free Cooley-Tukey stage twiddles
 * 
 * Safe to call with NULL. Uses platform-specific aligned free.
 * 
 * @param twiddles Twiddle array from compute_stage_twiddles()
 */
void free_stage_twiddles(fft_data *twiddles);

#endif // FFT_TWIDDLES_H