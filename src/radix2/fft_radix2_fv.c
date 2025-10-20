//==============================================================================
// fft_radix2_fv.c - Forward Radix-2 Butterfly (OPTIMIZED)
//==============================================================================
//
// ARCHITECTURE:
// - Separate from _bv for single source of truth on stage twiddles
// - Direction encoded in function name, not runtime parameter
// - Shared implementation via inline helpers + macros
//

#include "fft_radix2.h"
#include "simd_math.h"
#include "fft_radix2_macros.h"

void fft_radix2_fv(
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
    // STAGE 0: k=0 (W^0 = 1) - Inline helper
    //==========================================================================
    radix2_butterfly_k0(output_buffer, sub_outputs, half);
    
    //==========================================================================
    // STAGE 1: k=N/4 (W^(N/4) = -i for forward) - Inline helper
    //==========================================================================
    int k_quarter = 0;
    if ((half & 1) == 0) {
        k_quarter = half >> 1;
        radix2_butterfly_k_quarter(output_buffer, sub_outputs, half, k_quarter, false);
    }
    
    //==========================================================================
    // STAGE 2: General case - Unified loop helper
    //==========================================================================
    radix2_process_main_loop(output_buffer, sub_outputs, stage_tw, half, k_quarter);
}
