// fft_twiddles.c
// SINGLE SOURCE OF TRUTH for all twiddle computation
// Ultra-optimized: AVX2, FMA, loop unrolling, prefetching

#ifndef FFT_TWIDDLES_H
#define FFT_TWIDDLES_H

#include "fft_planning_types.h"

//==============================================================================
// API - Twiddle Manager
//==============================================================================

/**
 * @brief Compute Cooley-Tukey stage twiddles (SINGLE SOURCE OF TRUTH)
 * 
 * Computes: W^(r*k) = exp(sign * 2πi * r * k / N_stage)
 *   where sign = -1 for FORWARD, +1 for INVERSE
 *         r = 1..radix-1, k = 0..sub_len-1
 * 
 * Layout: Interleaved [W^(1*0), W^(2*0), ..., W^(R-1*0),
 *                      W^(1*1), W^(2*1), ..., W^(R-1*1), ...]
 * 
 * @param N_stage Current stage size
 * @param radix Radix of decomposition
 * @param direction FORWARD or INVERSE
 * @return Allocated array of (radix-1) * sub_len twiddles (32-byte aligned)
 */
fft_data* compute_stage_twiddles(
    int N_stage,
    int radix,
    fft_direction_t direction
);

/**
 * @brief Free stage twiddles
 */
void free_stage_twiddles(fft_data *twiddles);

#endif // FFT_TWIDDLES_H