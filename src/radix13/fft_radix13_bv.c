//==============================================================================
// fft_radix13_bv.c - Radix-13 Butterfly (INVERSE Transform, Rader's Algorithm)
//==============================================================================

#include "fft_radix13_uniform.h"
#include "fft_radix13_macros.h"
#include "highSpeedFFT.h"

void fft_radix13_bv(
    fft_data *output_buffer,
    fft_data *sub_outputs,
    const fft_data *stage_tw,
    int sub_len)
{
    const int K = sub_len;
    int k = 0;

#ifdef __AVX2__
    //==========================================================================
    // PRECOMPUTE CONVOLUTION TWIDDLES (INVERSE) - Point 3: Hoist once
    //==========================================================================
    __m256d tw_brd[12];
    PRECOMPUTE_RADER13_TWIDDLES_BV(tw_brd);  // Only difference from FV!

    //==========================================================================
    // AVX2 PATH: Process 2 butterflies per iteration - Point 5: Full AVX2
    //==========================================================================
    for (; k + 1 < K; k += 2)
    {
        // Prefetch
        if (k + 8 < K) {
            _mm_prefetch((const char *)&sub_outputs[k + 8].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_outputs[k + 8 + K].re, _MM_HINT_T0);
        }

        // Step 1: Load 13 lanes
        __m256d x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12;
        LOAD_13_LANES_AVX2_2X(k, K, sub_outputs, x0, x1, x2, x3, x4, x5, x6, 
                               x7, x8, x9, x10, x11, x12);

        // Step 2: Apply stage twiddles
        APPLY_STAGE_TWIDDLES_R13_AVX2_2X(k, K, x1, x2, x3, x4, x5, x6, x7, x8, 
                                          x9, x10, x11, x12, stage_tw);

        // Step 3: Compute Y0
        __m256d y0;
        COMPUTE_Y0_AVX2(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, y0);

        // Step 4: Apply Rader input permutation
        __m256d tx0, tx1, tx2, tx3, tx4, tx5, tx6, tx7, tx8, tx9, tx10, tx11;
        RADER13_INPUT_PERMUTE_AVX2(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12,
                                    tx0, tx1, tx2, tx3, tx4, tx5, tx6, tx7, tx8, tx9, tx10, tx11);

        // Step 5: Full 12-point cyclic convolution (Point 1 & 4: Fully unrolled)
        __m256d v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11;
        CONV12_FULL_AVX2(tx0, tx1, tx2, tx3, tx4, tx5, tx6, tx7, tx8, tx9, tx10, tx11, tw_brd,
                         v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11);

        // Step 6: Map convolution outputs to final DFT outputs
        __m256d y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12;
        MAP_RADER13_OUTPUTS_AVX2(x0, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11,
                                  y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12);

        // Step 7: Store (Point 6: Using optimized stores)
        STORE_13_LANES_AVX2_2X(k, K, output_buffer, y0, y1, y2, y3, y4, y5, y6,
                                y7, y8, y9, y10, y11, y12);
    }
#endif // __AVX2__

    //==========================================================================
    // SCALAR TAIL: Handle remaining 0-1 elements
    //==========================================================================
    fft_data tw_scalar[12];
    PRECOMPUTE_RADER13_TWIDDLES_SCALAR_BV(tw_scalar);  // Only difference from FV!

    for (; k < K; k++)
    {
        RADIX13_BUTTERFLY_SCALAR(k, K, sub_outputs, stage_tw, tw_scalar, output_buffer);
    }
}