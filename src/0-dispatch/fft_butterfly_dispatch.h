/**
 * @file fft_butterfly_dispatch.h
 * @brief Unified butterfly dispatch for both twiddle and twiddle-less butterflies
 * 
 * @details
 * Supports two butterfly types (like FFTW):
 * 1. n1 butterflies: Twiddle-less (base case, W^0 = 1)
 * 2. Twiddle butterflies: Full Cooley-Tukey with twiddle factors
 */

#ifndef FFT_BUTTERFLY_DISPATCH_H
#define FFT_BUTTERFLY_DISPATCH_H

#include "fft_planning_types.h"
#include "fft_twiddles_planner_api.h"

//==============================================================================
// BUTTERFLY FUNCTION TYPES
//==============================================================================

/**
 * @brief Twiddle-less butterfly (n1 codelet)
 * 
 * Used for:
 * - Base cases (small FFTs as building blocks)
 * - First stage in recursive decomposition (twiddles are all 1)
 * 
 * @param output Output buffer
 * @param input Input buffer
 * @param sub_len Number of butterflies (for consistency, often 1)
 */
typedef void (*butterfly_n1_func_t)(
    fft_data *restrict output,
    const fft_data *restrict input,
    int sub_len
);

/**
 * @brief Twiddle butterfly (standard Cooley-Tukey)
 * 
 * Used for:
 * - Mixed-radix stages where twiddles are needed
 * - All recursive combination stages
 * 
 * @param output Output buffer
 * @param input Input buffer
 * @param twiddles Twiddle factors (SoA format)
 * @param sub_len Number of butterflies to process
 */
typedef void (*butterfly_twiddle_func_t)(
    fft_data *restrict output,
    const fft_data *restrict input,
    const fft_twiddles_soa_view *restrict twiddles,
    int sub_len
);

//==============================================================================
// BUTTERFLY DESCRIPTOR
//==============================================================================

/**
 * @brief Butterfly function pair
 * 
 * Contains both n1 (twiddle-less) and twiddle versions of a butterfly.
 * Some radices may only have one type.
 */
typedef struct {
    butterfly_n1_func_t n1;           // Twiddle-less version (may be NULL)
    butterfly_twiddle_func_t twiddle; // Twiddle version (may be NULL)
} butterfly_pair_t;

//==============================================================================
// DISPATCH API
//==============================================================================

/**
 * @brief Get butterfly function pair for given radix and direction
 * 
 * @param radix Radix (2, 3, 4, 5, 7, 8, 11, 13, 16, 32)
 * @param is_forward 1 for forward FFT, 0 for inverse FFT
 * @return Butterfly pair with n1 and twiddle functions
 * 
 * @note Either .n1 or .twiddle may be NULL if not implemented
 * 
 * @example
 *   // For base case (no twiddles needed):
 *   butterfly_pair_t bf = get_butterfly_pair(16, 1);
 *   if (bf.n1) {
 *       bf.n1(output, input, 1);
 *   }
 * 
 *   // For mixed-radix stage (twiddles needed):
 *   butterfly_pair_t bf = get_butterfly_pair(16, 1);
 *   if (bf.twiddle) {
 *       bf.twiddle(output, input, &tw_view, sub_len);
 *   }
 */
butterfly_pair_t get_butterfly_pair(int radix, int is_forward);

/**
 * @brief Get twiddle-less (n1) butterfly only
 * 
 * Convenience function for base cases.
 */
static inline butterfly_n1_func_t get_butterfly_n1(int radix, int is_forward)
{
    butterfly_pair_t pair = get_butterfly_pair(radix, is_forward);
    return pair.n1;
}

/**
 * @brief Get twiddle butterfly only
 * 
 * Convenience function for mixed-radix stages.
 */
static inline butterfly_twiddle_func_t get_butterfly_twiddle(int radix, int is_forward)
{
    butterfly_pair_t pair = get_butterfly_pair(radix, is_forward);
    return pair.twiddle;
}

#endif // FFT_BUTTERFLY_DISPATCH_H