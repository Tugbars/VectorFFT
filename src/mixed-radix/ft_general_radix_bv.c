//==============================================================================
// fft_general_radix_bv.c - Inverse General Radix Butterfly
//==============================================================================

#include "fft_general_radix.h"
#include "fft_general_radix_shared.h"
#include <stdio.h>

void fft_general_radix_bv(
    fft_data *restrict output_buffer,
    const fft_data *restrict sub_outputs,
    const fft_data *restrict stage_tw,
    const fft_data *restrict dft_kernel_tw,  // ⚡ ADDED: Same as forward!
    int radix,
    int sub_len)
{
    const int r = radix;
    const int K = sub_len;
    
    // Validate
    if (r > 64 || !dft_kernel_tw) {  // ⚡ Check dft_kernel_tw too!
        fprintf(stderr, "Error: Invalid radix %d or NULL dft_kernel_tw\n", r);
        return;
    }
    
    //==========================================================================
    // Use precomputed DFT kernel twiddles (⚡ IDENTICAL to forward!)
    //==========================================================================
    
    const fft_data *W_r = dft_kernel_tw;  // ⚡ Just use precomputed!
    
    //==========================================================================
    // Main Transform Loop (⚡ IDENTICAL to forward!)
    //==========================================================================
    
    int k = 0;
    
#ifdef __AVX2__
    // AVX2: Process 4 k-indices at once
    GENERAL_RADIX_AVX2_LOOP(k, K, r, sub_outputs, stage_tw, W_r, output_buffer);
#endif
    
    // Scalar tail: Process remaining k-indices
    GENERAL_RADIX_SCALAR_LOOP(k, K, r, sub_outputs, stage_tw, W_r, output_buffer);
}

// 300