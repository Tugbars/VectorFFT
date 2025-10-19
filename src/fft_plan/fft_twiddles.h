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

//==============================================================================
// TIER 1B: DFT KERNEL TWIDDLES (NEW!)
//==============================================================================

/**
 * @brief Compute DFT kernel twiddles for radix-r DFT
 * 
 * **Purpose:** Root-of-unity twiddles for the radix-r DFT itself
 * (NOT the Cooley-Tukey inter-stage twiddles)
 * 
 * **Formula:**
 * W_r[m] = exp(sign × 2πi × m / radix)
 * 
 * where:
 * - sign = -1 for FFT_FORWARD, +1 for FFT_INVERSE
 * - m = 0..radix-1
 * 
 * **Usage:** General radix butterflies use these to compute the
 * radix-r DFT kernel via matrix multiplication:
 * 
 * output[m] = Σ_{j=0}^{r-1} twiddled_input[j] × W_r[m]^j
 * 
 * **When Needed:**
 * - General radix fallback (radix ∉ {2,3,4,5,7,8,11,13})
 * - Future specialized radices (e.g., radix-6, radix-9, radix-15)
 * 
 * **Memory:**
 * - 32-byte aligned for AVX2
 * - Size: radix complex values (e.g., 17 complex = 272 bytes)
 * - OWNED by stage descriptor
 * - Negligible compared to stage twiddles
 * 
 * **Optimization:**
 * - Computed once at planning time
 * - Eliminates 10-64 sincos() calls per butterfly execution
 * - 20× speedup for general radix
 * 
 * **Example:**
 * For radix=17, forward FFT:
 * - W_r[0]  = exp(-2πi×0/17) = 1.0
 * - W_r[1]  = exp(-2πi×1/17) = 0.932 - 0.362i
 * - W_r[2]  = exp(-2πi×2/17) = 0.739 - 0.674i
 * - ...
 * - W_r[16] = exp(-2πi×16/17) = 0.932 + 0.362i (conjugate of W_r[1])
 * 
 * @param radix DFT kernel size (2..64 typically)
 * @param direction FFT_FORWARD or FFT_INVERSE
 * @return Allocated array of radix complex twiddles, or NULL on error
 */
fft_data* compute_dft_kernel_twiddles(
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