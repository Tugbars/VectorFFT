/**

fft_exec_strategy_t choose_strategy(int N, int stride) {
    //==========================================================================
    // STEP 1: Check if N is highly composite (product of small primes ≤ 13)
    //==========================================================================
    bool is_composite = can_factor_into_small_primes(N);
    
    if (is_composite) {
        // ──────────────────────────────────────────────────────────
        // PATH A: Composite N → Use direct methods
        // ──────────────────────────────────────────────────────────
        
        if (N <= 64 && is_power_of_2(N)) {
            return FFT_EXEC_INPLACE_BITREV;  // Fastest for tiny N
        }
        
        if (N < 262144) {  // N < 256K
            return FFT_EXEC_RECURSIVE_CT;    // Cache-oblivious (FFTW-style)
        }
        
        if (fft_should_use_fourstep(N, stride)) {
            return FFT_EXEC_FOURSTEP;        // Explicit blocking for large N
        }
        
        return FFT_EXEC_RECURSIVE_CT;        // Default
    }
    else {
        // ──────────────────────────────────────────────────────────
        // PATH B: Prime or large prime factors → Must use Bluestein
        // ──────────────────────────────────────────────────────────
        return FFT_EXEC_BLUESTEIN;
    }
}



bool can_factor_into_small_primes(int N) {
    int remaining = N;
    
    // Try dividing by radices we support: 2,3,4,5,7,8,11,13,16,32
    int radices[] = {2, 3, 5, 7, 11, 13};
    
    for (int i = 0; i < 6; i++) {
        while (remaining % radices[i] == 0) {
            remaining /= radices[i];
        }
    }
    
    // If remaining == 1, N is fully factorizable
    // If remaining > 13, has large prime factor → need Bluestein
    return (remaining == 1);
}
```

## Empirical Performance Chart
```
Performance (lower is better) vs N:
═══════════════════════════════════════════════════════════════════

Time (µs)
    │
10000├─────────────────────────────────────────────────────────────
     │                                           ╱ Bluestein (prime N)
     │                                      ╱╱╱╱
     │                                 ╱╱╱╱
 1000├──────────────────────────╱╱╱╱╱─────────────────────────────
     │                     ╱╱╱╱    ╲ Four-step
     │                ╱╱╱╱          ╲ (composite N ≥ 256K)
     │           ╱╱╱╱                ╲
  100├──────╱╱╱╱─────────────────────╲────────────────────────────
     │  ╱╱╱╱  ╲ Recursive                ╲
     │╱╱     ╲ (composite 64 < N < 256K)  ╲
   10├╲──────╲─────────────────────────────╲──────────────────────
     │ ╲ Bit-reversal                       ╲
     │  ╲ (composite N ≤ 64)                 ╲
    1├───╲──────────────────────────────────────────────────────────
     └────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────► N
         64   256   1K   4K  16K  64K 256K   1M   4M  16M  64M


─── Composite N (optimal methods)
╱╱╱ Prime N (Bluestein fallback, 3-5× slower)
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
    if (!plan || !input || !output)
    {
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
        if (!workspace)
            return -1;

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
    if (!plan || !data)
    {
        return -1;
    }

    if (plan->strategy != FFT_EXEC_INPLACE_BITREV)
    {
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
    if (!plan || !input || !output)
    {
        return -1;
    }

    switch (plan->strategy)
    {
    case FFT_EXEC_INPLACE_BITREV:
        // Bit-reversal: No workspace needed
        return fft_exec_bitrev_strategy(plan, input, output);

    case FFT_EXEC_RECURSIVE_CT:
        // Cache-oblivious recursion
        if (!workspace)
        {
            fprintf(stderr, "ERROR: Recursive strategy requires workspace\n");
            return -1;
        }
        return fft_exec_recursive_strategy(plan, input, output, workspace);

    case FFT_EXEC_FOURSTEP:
        // Four-step with explicit blocking
        if (!workspace)
        {
            fprintf(stderr, "ERROR: Four-step requires workspace\n");
            return -1;
        }
        return fft_exec_fourstep(plan, input, output, workspace);

    case FFT_EXEC_BLUESTEIN:
        // Bluestein for arbitrary sizes
        if (!workspace)
        {
            fprintf(stderr, "ERROR: Bluestein requires workspace\n");
            return -1;
        }

        size_t scratch_size = bluestein_get_scratch_size(plan->n_input);

        if (plan->direction == FFT_FORWARD)
        {
            return bluestein_exec_forward(
                plan->bluestein_fwd, input, output, workspace, scratch_size);
        }
        else
        {
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
    if (result != 0)
        return result;

    // Apply 1/N normalization on inverse only
    if (plan->direction == FFT_INVERSE)
    {
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
    if (!fwd_plan || !inv_plan || !input || !output)
    {
        return -1;
    }

    if (fwd_plan->n_fft != inv_plan->n_fft)
    {
        fprintf(stderr, "ERROR: Plan size mismatch\n");
        return -1;
    }

    const int N = fwd_plan->n_fft;

    fft_data *freq = (fft_data *)malloc(N * sizeof(fft_data));
    if (!freq)
        return -1;

    // Forward FFT
    int result = fft_exec_dft(fwd_plan, input, freq, workspace);
    if (result != 0)
    {
        free(freq);
        return result;
    }

    // Inverse FFT
    result = fft_exec_dft(inv_plan, freq, output, workspace);
    if (result != 0)
    {
        free(freq);
        return result;
    }

    // Normalize
    FFT_NORMALIZE_INVERSE(output, N);

    free(freq);
    return 0;
}