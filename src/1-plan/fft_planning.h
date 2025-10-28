//==============================================================================
// fft_planning.h
// MAIN PLANNING ORCHESTRATOR - The "fft_init" umbrella
//==============================================================================

/**
 * @file fft_planning.h
 * @brief Public interface for FFT plan creation and management
 * 
 * **Architecture Overview:**
 * This module provides the top-level API for creating FFT execution plans.
 * Planning uses a sophisticated multi-phase approach:
 * 
 * 1. **Prime Factorization** - Decompose N into prime factors
 * 2. **Dynamic Programming Packing** - Combine primes into optimal radix sequence
 * 3. **Twiddle Computation** - Pre-compute all rotation factors
 * 4. **Rader Integration** - Fetch convolution twiddles for prime radices
 * 5. **Strategy Selection** - Choose execution algorithm (in-place, Stockham, Bluestein)
 * 
 * **Key Features:**
 * - Optimal factorization using DP (not greedy)
 * - Cost-aware radix selection
 * - Intelligent fallbacks for unfactorizable sizes
 * - Zero-copy workspace model (user provides at execution time)
 * - Thread-safe execution with shared plans
 * 
 * **Usage Pattern:**
 * ```c
 * // Create plan once (expensive, ~1-10ms)
 * fft_object plan = fft_init(1024, FFT_FORWARD);
 * 
 * // Query workspace requirement
 * size_t ws_size = fft_get_workspace_size(plan);
 * fft_data *workspace = ws_size ? malloc(ws_size * sizeof(fft_data)) : NULL;
 * 
 * // Execute many times (fast, ~10-100μs)
 * for (int i = 0; i < 1000; i++) {
 *     fft_exec_dft(plan, input, output, workspace);
 * }
 * 
 * // Cleanup
 * free(workspace);
 * free_fft(plan);
 * ```
 */

#ifndef FFT_PLANNING_H
#define FFT_PLANNING_H

#include "fft_planning_types.h"

fft_object fft_init(int N, fft_direction_t direction);

void free_fft(fft_object plan);

int fft_can_execute_inplace(fft_object plan);

size_t fft_get_workspace_size(fft_object plan);

fft_exec_strategy_t fft_get_strategy(fft_object plan);

void init_rader_cache(void);

void cleanup_rader_cache(void);


#endif // FFT_PLANNING_H