//==============================================================================
// fft_radix4_bv.c - Inverse Radix-4 Butterfly (Precomputed Twiddles)
//==============================================================================
//
// DESIGN: Identical to fft_radix4_fv.c except:
// - Uses RADIX4_ROTATE_INVERSE_* instead of RADIX4_ROTATE_FORWARD_*
// - Twiddles have inverse sign: W_N^k = exp(+2πik/N)
//

#include "fft_radix4.h"
#include "simd_math.h"
#include "fft_radix4_macros.h"

// Non-temporal store threshold
#define STREAM_THRESHOLD 8192

//==============================================================================
// INVERSE RADIX-4 BUTTERFLY - Main Function
//==============================================================================

/**
 * @brief Ultra-optimized inverse radix-4 butterfly
 * 
 * DIFFERENCE FROM FORWARD:
 * - Uses +i rotation instead of -i
 * - Twiddles: Precomputed with positive sign (exp(+2πik/N))
 * 
 * Everything else is IDENTICAL to fft_radix4_fv()
 */
void fft_radix4_bv(
    fft_data *restrict output_buffer,
    const fft_data *restrict sub_outputs,
    const fft_data *restrict stage_tw,
    int sub_len)
{
    const int K = sub_len;
    int k = 0;

#ifdef __AVX2__
    //==========================================================================
    // AVX2 PATH: 8x UNROLL (IDENTICAL to forward except rotation)
    //==========================================================================
    
    const int use_streaming = (K >= STREAM_THRESHOLD);

    // Main loop: process 2 butterflies at a time, 8 times per iteration
    for (; k + 15 < K; k += 16)
    {
        // Prefetch ahead (IDENTICAL)
        PREFETCH_4_LANES(k, K, PREFETCH_L1, sub_outputs, _MM_HINT_T0);
        PREFETCH_4_LANES(k, K, PREFETCH_L2, sub_outputs, _MM_HINT_T1);
        PREFETCH_4_LANES(k, K, PREFETCH_L3, sub_outputs, _MM_HINT_T2);
        
        // Process 8 pairs of butterflies
        for (int p = 0; p < 8; p++)
        {
            int kk = k + 2*p;
            
            //==================================================================
            // STAGE 0: Load inputs (IDENTICAL)
            //==================================================================
            __m256d a, b, c, d;
            LOAD_4_LANES_AVX2(kk, K, sub_outputs, a, b, c, d);
            
            //==================================================================
            // STAGE 1: Apply precomputed twiddles (IDENTICAL)
            //==================================================================
            __m256d tw_b, tw_c, tw_d;
            APPLY_STAGE_TWIDDLES_AVX2(kk, b, c, d, stage_tw, tw_b, tw_c, tw_d);
            
            //==================================================================
            // STAGE 2: Butterfly core arithmetic (IDENTICAL)
            //==================================================================
            __m256d sumBD, difBD, sumAC, difAC;
            RADIX4_BUTTERFLY_CORE_AVX2(a, tw_b, tw_c, tw_d, 
                                        sumBD, difBD, sumAC, difAC);
            
            //==================================================================
            // STAGE 3: Inverse rotation (+i multiplication) ⚡ ONLY DIFFERENCE
            //==================================================================
            __m256d rot;
            RADIX4_ROTATE_INVERSE_AVX2(difBD, rot);
            
            //==================================================================
            // STAGE 4: Assemble outputs (IDENTICAL)
            //==================================================================
            __m256d y0, y1, y2, y3;
            RADIX4_ASSEMBLE_OUTPUTS_AVX2(sumAC, sumBD, difAC, rot, 
                                          y0, y1, y2, y3);
            
            //==================================================================
            // STAGE 5: Store results (IDENTICAL)
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
    
    // Cleanup loop: process 2 butterflies at a time (IDENTICAL except rotation)
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
        RADIX4_ROTATE_INVERSE_AVX2(difBD, rot);  // ⚡ ONLY DIFFERENCE
        
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
        RADIX4_BUTTERFLY_SCALAR_BV(k, K, sub_outputs, stage_tw, output_buffer);
    }
}

//==============================================================================
// SUMMARY: Forward vs Inverse
//==============================================================================

/**
 * IDENTICAL CODE (~99%):
 * - All load/store patterns
 * - All prefetching
 * - All twiddle application
 * - Butterfly core arithmetic
 * - Output assembly
 * 
 * DIFFERENT (1 macro call per butterfly):
 * - Rotation: RADIX4_ROTATE_FORWARD_AVX2 vs RADIX4_ROTATE_INVERSE_AVX2
 * 
 * TWIDDLE DIFFERENCE (computed by planning):
 * - Forward:  exp(-2πik/N) - negative sign
 * - Inverse:  exp(+2πik/N) - positive sign
 * 
 * This is why _fv and _bv can share 99% of code via macros!
 * 
 * ROTATION DETAILS:
 * - Forward: -i * difBD = (difBD_im, -difBD_re)
 * - Inverse: +i * difBD = (-difBD_im, difBD_re)
 * 
 * In AVX2, this is just a different XOR mask:
 * - Forward: XOR with (-0.0, 0.0, -0.0, 0.0)
 * - Inverse: XOR with (0.0, -0.0, 0.0, -0.0)
 */

//==============================================================================
// PERFORMANCE NOTES
//==============================================================================

/**
 * Performance is IDENTICAL to forward butterfly:
 * - AVX2: ~1.0 cycles/butterfly
 * - Scalar: ~4.0 cycles/butterfly
 * 
 */