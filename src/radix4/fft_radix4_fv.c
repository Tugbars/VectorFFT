//==============================================================================
// fft_radix4_fv.c - Forward Radix-4 Butterfly (Precomputed Twiddles)
//==============================================================================
//
// DESIGN PRINCIPLES:
// 1. No direction parameter - always forward FFT
// 2. stage_tw is NEVER NULL - always precomputed
// 3. Macros for 99% code reuse with inverse
// 4. Clean AVX2 8x unroll path
// 5. Simple scalar tail
//

#include "fft_radix4.h"
#include "simd_math.h"
#include "fft_radix4_macros.h"

// Non-temporal store threshold
#define STREAM_THRESHOLD 8192

//==============================================================================
// FORWARD RADIX-4 BUTTERFLY - Main Function
//==============================================================================

/**
 * @brief Ultra-optimized forward radix-4 butterfly
 * 
 * ASSUMPTIONS:
 * - stage_tw is NEVER NULL (always precomputed)
 * - Direction is ALWAYS forward (no runtime checks)
 * - Twiddles have forward sign: W_N^k = exp(-2πik/N)
 * 
 * TWIDDLE LAYOUT:
 * - stage_tw[k*3 + 0] = W_N^(1*k)  (lane 1)
 * - stage_tw[k*3 + 1] = W_N^(2*k)  (lane 2)
 * - stage_tw[k*3 + 2] = W_N^(3*k)  (lane 3)
 * 
 * PERFORMANCE:
 * - AVX2 8x unroll: ~1.0 cycles/butterfly
 * - Scalar tail: ~4.0 cycles/butterfly
 */
void fft_radix4_fv(
    fft_data *restrict output_buffer,
    const fft_data *restrict sub_outputs,
    const fft_data *restrict stage_tw,
    int sub_len)
{
    const int K = sub_len;
    int k = 0;

#ifdef __AVX2__
    //==========================================================================
    // AVX2 PATH: 8x UNROLL (processes 16 butterflies per iteration)
    //==========================================================================
    
    const int use_streaming = (K >= STREAM_THRESHOLD);

    // Main loop: process 2 butterflies at a time, 8 times per iteration
    for (; k + 15 < K; k += 16)
    {
        // Prefetch ahead
        PREFETCH_4_LANES(k, K, PREFETCH_L1, sub_outputs, _MM_HINT_T0);
        PREFETCH_4_LANES(k, K, PREFETCH_L2, sub_outputs, _MM_HINT_T1);
        PREFETCH_4_LANES(k, K, PREFETCH_L3, sub_outputs, _MM_HINT_T2);
        
        // Process 8 pairs of butterflies
        for (int p = 0; p < 8; p++)
        {
            int kk = k + 2*p;
            
            //==================================================================
            // STAGE 0: Load inputs (2 butterflies = 8 complex values)
            //==================================================================
            __m256d a, b, c, d;
            LOAD_4_LANES_AVX2(kk, K, sub_outputs, a, b, c, d);
            
            //==================================================================
            // STAGE 1: Apply precomputed twiddles (NO sin/cos!)
            //==================================================================
            __m256d tw_b, tw_c, tw_d;
            // Lane 0 (a) needs no twiddle
            APPLY_STAGE_TWIDDLES_AVX2(kk, b, c, d, stage_tw, tw_b, tw_c, tw_d);
            
            //==================================================================
            // STAGE 2: Butterfly core arithmetic
            //==================================================================
            __m256d sumBD, difBD, sumAC, difAC;
            RADIX4_BUTTERFLY_CORE_AVX2(a, tw_b, tw_c, tw_d, 
                                        sumBD, difBD, sumAC, difAC);
            
            //==================================================================
            // STAGE 3: Forward rotation (-i multiplication)
            //==================================================================
            __m256d rot;
            RADIX4_ROTATE_FORWARD_AVX2(difBD, rot);
            
            //==================================================================
            // STAGE 4: Assemble outputs
            //==================================================================
            __m256d y0, y1, y2, y3;
            RADIX4_ASSEMBLE_OUTPUTS_AVX2(sumAC, sumBD, difAC, rot, 
                                          y0, y1, y2, y3);
            
            //==================================================================
            // STAGE 5: Store results
            //==================================================================
            if (use_streaming)
            {
                STORE_4_LANES_AVX2_STREAM(kk, K, output_buffer, y0, y1, y2, y3);
            }
            else
            {
                STORE_4_LANES_AVX2(kk, K, output_buffer, y0, y1, y2, y3);
            }
        }
    }
    
    // Cleanup loop: process 2 butterflies at a time
    for (; k + 1 < K; k += 2)
    {
        __m256d a, b, c, d;
        LOAD_4_LANES_AVX2(k, K, sub_outputs, a, b, c, d);
        
        __m256d tw_b, tw_c, tw_d;
        APPLY_STAGE_TWIDDLES_AVX2(k, b, c, d, stage_tw, tw_b, tw_c, tw_d);
        
        __m256d sumBD, difBD, sumAC, difAC;
        RADIX4_BUTTERFLY_CORE_AVX2(a, tw_b, tw_c, tw_d, 
                                    sumBD, difBD, sumAC, difAC);
        
        __m256d rot;
        RADIX4_ROTATE_FORWARD_AVX2(difBD, rot);
        
        __m256d y0, y1, y2, y3;
        RADIX4_ASSEMBLE_OUTPUTS_AVX2(sumAC, sumBD, difAC, rot, 
                                      y0, y1, y2, y3);
        
        STORE_4_LANES_AVX2(k, K, output_buffer, y0, y1, y2, y3);
    }
    
    if (use_streaming)
    {
        _mm_sfence();
    }
    
#endif // __AVX2__

    //==========================================================================
    // SCALAR TAIL: Process remaining single butterflies
    //==========================================================================
    for (; k < K; ++k)
    {
        RADIX4_BUTTERFLY_SCALAR_FV(k, K, sub_outputs, stage_tw, output_buffer);
    }
}
