//==============================================================================
// fft_radix2_fv.c - Forward Radix-2 Butterfly (SOA VERSION)
//==============================================================================

#include "fft_radix2.h"
#include "simd_math.h"
#include "fft_radix2_macros.h"

void fft_radix2_fv(
    fft_data *restrict output_buffer,
    const fft_data *restrict sub_outputs,
    const fft_twiddles_soa *restrict stage_tw, // ✅ ONLY CHANGE
    int sub_len)
{
    // Alignment hints (data comes pre-aligned from planner)
#ifdef __AVX512F__
    output_buffer = __builtin_assume_aligned(output_buffer, 64);
    sub_outputs = __builtin_assume_aligned(sub_outputs, 64);
#else
    output_buffer = __builtin_assume_aligned(output_buffer, 32);
    sub_outputs = __builtin_assume_aligned(sub_outputs, 32);
#endif

    const int half = sub_len;

    //==========================================================================
    // STAGE 0: k=0 (W^0 = 1) - Inline helper
    //==========================================================================
    radix2_butterfly_k0(output_buffer, sub_outputs, half);

    //==========================================================================
    // STAGE 1: k=N/4 (W^(N/4) = -i for forward) - Inline helper
    //==========================================================================
    int k_quarter = (half & 1) ? 0 : (half >> 1);

    if (k_quarter)
    {
        radix2_butterfly_k_quarter(output_buffer, sub_outputs, half, k_quarter, false);
    }

    //==========================================================================
    // STAGE 2: General case - Unified loop helper (split loops, no branches)
    //==========================================================================
    radix2_process_main_loop_soa(output_buffer, sub_outputs, stage_tw, half, k_quarter);
}