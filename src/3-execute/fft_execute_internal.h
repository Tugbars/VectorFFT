/**
 * @file fft_execute_internal.h
 * @brief Internal execution strategy interfaces
 * 
 * This header defines the internal API between fft_execute.c (dispatcher)
 * and the individual execution strategy implementations.
 */

#ifndef FFT_EXECUTE_INTERNAL_H
#define FFT_EXECUTE_INTERNAL_H

#include "fft_planning_types.h"

//==============================================================================
// STRATEGY EXECUTION FUNCTIONS
//==============================================================================

/**
 * @brief Execute FFT using bit-reversal (small power-of-2 only)
 * 
 * @param plan FFT plan (must have strategy = FFT_EXEC_INPLACE_BITREV)
 * @param input Input data (N elements)
 * @param output Output buffer (N elements, may alias input)
 * @return 0 on success, -1 on error
 * 
 * @note This is true in-place, no workspace needed
 * @note Only supports N ≤ 64, power-of-2
 */
int fft_exec_bitrev_strategy(
    fft_object plan,
    const fft_data *input,
    fft_data *output);

/**
 * @brief Execute FFT using cache-oblivious recursion
 * 
 * @param plan FFT plan (must have strategy = FFT_EXEC_RECURSIVE_CT)
 * @param input Input data (N elements)
 * @param output Output buffer (N elements)
 * @param workspace Working buffer (from fft_get_workspace_size())
 * @return 0 on success, -1 on error
 * 
 * @note Requires workspace (2×N elements)
 * @note Automatically adapts to cache hierarchy
 */
int fft_exec_recursive_strategy(
    fft_object plan,
    const fft_data *input,
    fft_data *output,
    fft_data *workspace);

/**
 * @brief Execute FFT using four-step algorithm
 * 
 * @param plan FFT plan (must have strategy = FFT_EXEC_FOURSTEP)
 * @param input Input data (N elements)
 * @param output Output buffer (N elements)
 * @param workspace Working buffer (from fft_get_workspace_size())
 * @return 0 on success, -1 on error
 * 
 * @note Requires workspace (N + 3*max_dim elements)
 * @note Best for large N (≥ 64K) with explicit cache blocking
 * 
 * @see fft_fourstep.c for implementation
 */
int fft_exec_fourstep(
    fft_object plan,
    const fft_data *input,
    fft_data *output,
    fft_data *workspace);

//==============================================================================
// HELPER FUNCTIONS (SHARED ACROSS STRATEGIES)
//==============================================================================

/**
 * @brief Get optimal base case size for cache-oblivious recursion
 * 
 * @return Base case size (typically 16-64 depending on L1 cache)
 */
int fft_get_optimal_base_case(void);

/**
 * @brief Get L1 cache size (architecture-specific heuristic)
 * 
 * @return L1 data cache size in bytes
 */
size_t fft_get_l1_cache_size(void);

#endif // FFT_EXECUTE_INTERNAL_H