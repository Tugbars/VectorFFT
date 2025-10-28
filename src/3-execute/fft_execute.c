/**
 * @file fft_execute.c
 * @brief FFT execution dispatcher and public API
 * 
 * @details
 * This file provides the public execution API and dispatches to
 * appropriate strategy implementations:
 * - fft_bitrev.c: Small power-of-2 (N ≤ 64)
 * - fft_recursive.c: Cache-oblivious (64 < N < 256K)
 * - fft_fourstep.c: Explicit blocking (N ≥ 256K)
 * - bluestein/: Arbitrary sizes via chirp-z
 * 
 * **Normalization Convention:**
 * - fft_exec_dft(): NO normalization (raw DFT)
 * - fft_exec_normalized(): 1/N on inverse only (FFTW-compatible)
 */

#include "fft_execute_internal.h"
#include "fft_planning_types.h"
#include "fft_planning.h"
#include "fft_normalize.h"
#include "../bluestein/bluestein.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

//==============================================================================
// LEGACY COMPATIBILITY (NO WORKSPACE)
//==============================================================================

/**
 * @brief Execute FFT without workspace (limited use)
 * 
 * @deprecated Use fft_exec_dft() with workspace for better performance
 */
int fft_exec(fft_object plan, const fft_data *input, fft_data *output)
{
    if (!plan || !input || !output) {
        return -1;
    }
    
    switch (plan->strategy)
    {
    case FFT_EXEC_INPLACE_BITREV:
        // No workspace needed
        return fft_exec_bitrev_strategy(plan, input, output);
        
    case FFT_EXEC_RECURSIVE_CT:
    {
        // Allocate workspace on-the-fly
        size_t ws_size = fft_get_workspace_size(plan);
        fft_data *workspace = (fft_data *)malloc(ws_size * sizeof(fft_data));
        if (!workspace) return -1;
        
        int result = fft_exec_recursive_strategy(plan, input, output, workspace);
        free(workspace);
        return result;
    }
    
    case FFT_EXEC_BLUESTEIN:
    case FFT_EXEC_FOURSTEP:
        fprintf(stderr, "ERROR: Strategy requires workspace, use fft_exec_dft()\n");
        return -1;
        
    default:
        return -1;
    }
}

/**
 * @brief Execute in-place FFT (small power-of-2 only)
 */
int fft_exec_inplace(fft_object plan, fft_data *data)
{
    if (!plan || !data) {
        return -1;
    }
    
    if (plan->strategy != FFT_EXEC_INPLACE_BITREV) {
        fprintf(stderr, "ERROR: Plan does not support in-place execution\n");
        return -1;
    }
    
    return fft_exec_bitrev_strategy(plan, data, data);
}

//==============================================================================
// PRIMARY EXECUTION API
//==============================================================================

/**
 * @brief Execute FFT without normalization (main API)
 */
int fft_exec_dft(
    fft_object plan,
    const fft_data *input,
    fft_data *output,
    fft_data *workspace)
{
    if (!plan || !input || !output) {
        return -1;
    }
    
    switch (plan->strategy)
    {
    case FFT_EXEC_INPLACE_BITREV:
        // Bit-reversal: No workspace needed
        return fft_exec_bitrev_strategy(plan, input, output);
        
    case FFT_EXEC_RECURSIVE_CT:
        // Cache-oblivious recursion
        if (!workspace) {
            fprintf(stderr, "ERROR: Recursive strategy requires workspace\n");
            return -1;
        }
        return fft_exec_recursive_strategy(plan, input, output, workspace);
        
    case FFT_EXEC_FOURSTEP:
        // Four-step with explicit blocking
        if (!workspace) {
            fprintf(stderr, "ERROR: Four-step requires workspace\n");
            return -1;
        }
        return fft_exec_fourstep(plan, input, output, workspace);
        
    case FFT_EXEC_BLUESTEIN:
        // Bluestein for arbitrary sizes
        if (!workspace) {
            fprintf(stderr, "ERROR: Bluestein requires workspace\n");
            return -1;
        }
        
        size_t scratch_size = bluestein_get_scratch_size(plan->n_input);
        
        if (plan->direction == FFT_FORWARD) {
            return bluestein_exec_forward(
                plan->bluestein_fwd, input, output, workspace, scratch_size);
        } else {
            return bluestein_exec_inverse(
                plan->bluestein_inv, input, output, workspace, scratch_size);
        }
        
    default:
        fprintf(stderr, "ERROR: Unknown execution strategy: %d\n", plan->strategy);
        return -1;
    }
}

/**
 * @brief Execute FFT with automatic normalization
 */
int fft_exec_normalized(
    fft_object plan,
    const fft_data *input,
    fft_data *output,
    fft_data *workspace)
{
    int result = fft_exec_dft(plan, input, output, workspace);
    if (result != 0) return result;
    
    // Apply 1/N normalization on inverse only
    if (plan->direction == FFT_INVERSE) {
        FFT_NORMALIZE_INVERSE(output, plan->n_fft);
    }
    
    return 0;
}

/**
 * @brief Round-trip test (forward + inverse with normalization)
 */
int fft_roundtrip_normalized(
    fft_object fwd_plan,
    fft_object inv_plan,
    const fft_data *input,
    fft_data *output,
    fft_data *workspace)
{
    if (!fwd_plan || !inv_plan || !input || !output) {
        return -1;
    }
    
    if (fwd_plan->n_fft != inv_plan->n_fft) {
        fprintf(stderr, "ERROR: Plan size mismatch\n");
        return -1;
    }
    
    const int N = fwd_plan->n_fft;
    
    fft_data *freq = (fft_data *)malloc(N * sizeof(fft_data));
    if (!freq) return -1;
    
    // Forward FFT
    int result = fft_exec_dft(fwd_plan, input, freq, workspace);
    if (result != 0) {
        free(freq);
        return result;
    }
    
    // Inverse FFT
    result = fft_exec_dft(inv_plan, freq, output, workspace);
    if (result != 0) {
        free(freq);
        return result;
    }
    
    // Normalize
    FFT_NORMALIZE_INVERSE(output, N);
    
    free(freq);
    return 0;
}