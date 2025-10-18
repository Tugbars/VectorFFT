// fft_planning.c
// MAIN PLANNING ORCHESTRATOR - The "fft_init" umbrella
// Coordinates: Factorization → Twiddle Manager → Rader Manager → Plan

#ifndef FFT_PLANNING_H
#define FFT_PLANNING_H

#include "fft_planning_types.h"

//==============================================================================
// MAIN API
//==============================================================================

/**
 * @brief Create FFT plan (FFTW-style planning)
 * 
 * This is the UMBRELLA function that orchestrates:
 *   1. Factorization
 *   2. Twiddle Manager (compute stage twiddles)
 *   3. Rader Manager (get Rader plans)
 *   4. Scratch allocation
 * 
 * @param N Signal length
 * @param direction FORWARD or INVERSE
 * @return Complete FFT plan (or NULL on error)
 * 
 * Note: If N cannot be factorized into supported radices, Bluestein's
 *       algorithm will be used (pads to next power-of-2 >= 2N-1).
 */
fft_object fft_init(int N, fft_direction_t direction);

/**
 * @brief Free FFT plan
 */
void free_fft(fft_object plan);

#endif // FFT_PLANNING_H