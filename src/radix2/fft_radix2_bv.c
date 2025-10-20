//==============================================================================
// fft_radix2_bv.c - Inverse Radix-2 Butterfly (OPTIMIZED)
//==============================================================================
//
// ARCHITECTURE:
// - Separate from _fv for single source of truth on stage twiddles
// - Direction encoded in function name, not runtime parameter
// - Shared implementation via inline helpers + macros
//
// ONLY DIFFERENCE FROM FORWARD: k=N/4 uses +i instead of -i
//

#include "fft_radix2.h"
#include "simd_math.h"
#include "fft_radix2_macros.h"

void fft_radix2_bv(
    fft_data *restrict output_buffer,
    const fft_data *restrict sub_outputs,
    const fft_data *restrict stage_tw,
    int sub_len)
{
    // Alignment hints (data comes pre-aligned from planner)
    output_buffer = __builtin_assume_aligned(output_buffer, 32);
    sub_outputs = __builtin_assume_aligned(sub_outputs, 32);
    stage_tw = __builtin_assume_aligned(stage_tw, 32);

    const int half = sub_len;

    //==========================================================================
    // STAGE 0: k=0 (W^0 = 1) - Inline helper (IDENTICAL to forward)
    //==========================================================================
    radix2_butterfly_k0(output_buffer, sub_outputs, half);

    //==========================================================================
    // STAGE 1: k=N/4 (W^(N/4) = +i for inverse) ⚡ ONLY DIFFERENCE
    //==========================================================================
    int k_quarter = 0;
    if ((half & 1) == 0)
    {
        k_quarter = half >> 1;
        radix2_butterfly_k_quarter(output_buffer, sub_outputs, half, k_quarter, true);
    }

    //==========================================================================
    // STAGE 2: General case - Unified loop helper (IDENTICAL to forward)
    //==========================================================================
    radix2_process_main_loop(output_buffer, sub_outputs, stage_tw, half, k_quarter);
}

//==============================================================================
// FORWARD vs INVERSE
//==============================================================================

/**
 * IDENTICAL CODE (99.9%):
 * - All SIMD paths (via macros)
 * - All loop structures (via inline helper)
 * - k=0 handling (via inline helper)
 * - Butterfly arithmetic (same macro)
 *
 * DIFFERENCE (1 parameter):
 * - Line 40: radix2_butterfly_k_quarter(..., true) vs (..., false)
 *            ^^^^
 *            is_inverse flag
 *
 * TWIDDLE DIFFERENCE (handled externally by planner):
 * - Forward:  stage_tw contains exp(-2πik/N) - planner computes this
 * - Inverse:  stage_tw contains exp(+2πik/N) - planner computes this
 *
 * WHY SEPARATE FUNCTIONS:
 * - Single source of truth for stage twiddles
 * - No runtime direction checks
 * - Critical for mixed-radix (prevents sign confusion)
 * - Planner responsibility: compute twiddles with correct sign
 * - Radix responsibility: apply butterflies (direction-agnostic)
 */
