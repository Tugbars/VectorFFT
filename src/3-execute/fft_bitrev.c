/**
 * @file fft_bitrev.c
 * @brief Bit-reversal FFT strategy for small power-of-2 sizes
 *
 * @details
 * This implements the classic Cooley-Tukey FFT with:
 * - Bit-reversal permutation (in-place)
 * - Iterative radix-2 butterflies
 * - True in-place operation (no workspace)
 *
 * **Use Cases:**
 * - N ≤ 64, power-of-2
 * - Embedded systems (minimal memory)
 * - Real-time audio (deterministic, no allocation)
 *
 * **Performance:**
 * - Optimal for N ≤ 16 (fits entirely in L1)
 * - Competitive up to N=64
 * - Loses to recursive strategies for N > 64
 */

#include "fft_execute_internal.h"
#include "fft_planning_types.h"
#include "fft_twiddles_planner_api.h"
#include <string.h>

//==============================================================================
// BIT REVERSAL
//==============================================================================

/**
 * @brief Reverse bits of x using log2(N) bits
 */
static inline unsigned int bit_reverse(unsigned int x, int bits)
{
    unsigned int result = 0;
    for (int i = 0; i < bits; i++)
    {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

/**
 * @brief Apply bit-reversal permutation in-place
 *
 * @param data Array to permute (modified in-place)
 * @param N Size (must be power-of-2)
 */
static void bit_reverse_permutation(fft_data *data, int N)
{
    int bits = __builtin_ctz(N); // log2(N) for power-of-2

    for (unsigned int i = 0; i < (unsigned int)N; i++)
    {
        unsigned int j = bit_reverse(i, bits);

        if (i < j)
        {
            fft_data temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }
}

//==============================================================================
// MAIN EXECUTION
//==============================================================================

/**
 * @brief Execute FFT using bit-reversal strategy
 *
 * **Algorithm:**
 * 1. Bit-reverse permutation (in-place)
 * 2. Iterative Cooley-Tukey stages
 *    - Stage s: Process groups of size 2^(s+1)
 *    - Apply radix-2 butterflies with twiddles
 *
 * **Complexity:**
 * - Time: O(N log N)
 * - Space: O(1) auxiliary (true in-place)
 * - Cache: Optimal for N ≤ 64 (fits in L1)
 */
int fft_exec_bitrev_strategy(
    fft_object plan,
    const fft_data *input,
    fft_data *output)
{
    if (!plan || !input || !output)
    {
        return -1;
    }

    if (plan->strategy != FFT_EXEC_INPLACE_BITREV)
    {
        return -1;
    }

    const int N = plan->n_fft;

    // Copy input to output (if not aliased)
    if (input != output)
    {
        memcpy(output, input, N * sizeof(fft_data));
    }

    // Step 1: Bit-reverse permutation
    bit_reverse_permutation(output, N);

    // Step 2: Cooley-Tukey stages
    int distance = 1;

    for (int stage = 0; stage < plan->num_stages; stage++)
    {
        stage_descriptor *s = &plan->stages[stage];

        if (s->radix != 2)
        {
            return -1; // Bit-reversal only supports radix-2
        }

        // Get twiddle view
        fft_twiddles_soa_view tw_view;
        if (twiddle_get_soa_view(s->stage_tw, &tw_view) != 0)
        {
            return -1;
        }

        const int num_groups = N / (2 * distance);

        // Process all butterflies at this stage
        for (int group = 0; group < num_groups; group++)
        {
            for (int k = 0; k < distance; k++)
            {
                int idx = group * 2 * distance + k;

                // Construct twiddle from SoA storage
                fft_data W;
                W.re = tw_view.re[k];
                W.im = tw_view.im[k];

                butterfly_radix2_inplace(output, idx, distance, W);
            }
        }

        distance *= 2;
    }

    return 0;
}