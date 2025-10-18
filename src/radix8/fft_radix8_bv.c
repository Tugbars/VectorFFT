//==============================================================================
// fft_radix8_bv.c - Inverse Radix-8 Butterfly (Precomputed Twiddles)
//==============================================================================
//
// DESIGN: Identical to fft_radix8_fv.c except:
// - Uses APPLY_W8_TWIDDLES_BV_* instead of APPLY_W8_TWIDDLES_FV_*
// - Radix-4 rotation uses +i instead of -i
// - Twiddles have inverse sign: W_N^k = exp(+2πik/N)
//

#include "fft_radix8.h"
#include "simd_math.h"
#include "fft_radix8_macros.h"

// Non-temporal store threshold
#define STREAM_THRESHOLD 4096

//==============================================================================
// INVERSE RADIX-8 BUTTERFLY - Main Function
//==============================================================================

/**
 * @brief Ultra-optimized inverse radix-8 butterfly
 * 
 * DIFFERENCE FROM FORWARD:
 * - W_8 twiddles use positive sign (conjugate)
 * - Radix-4 rotation uses +i instead of -i
 * - Twiddles: Precomputed with positive sign (exp(+2πik/N))
 * 
 * Everything else is IDENTICAL to fft_radix8_fv()
 */
void fft_radix8_bv(
    fft_data *restrict output_buffer,
    const fft_data *restrict sub_outputs,
    const fft_data *restrict stage_tw,
    int sub_len)
{
    const int K = sub_len;
    int k = 0;

#ifdef __AVX2__
    //==========================================================================
    // AVX2 PATH: Process 2 butterflies at a time (IDENTICAL except rotation)
    //==========================================================================
    
    const int use_streaming = (K >= STREAM_THRESHOLD);
    
    // Radix-4 rotation mask (inverse: +i) ⚡ ONLY DIFFERENCE from forward
    const __m256d rot_mask = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);

    // Main loop: process 2 butterflies per iteration
    for (; k + 1 < K; k += 2)
    {
        // Prefetch ahead (IDENTICAL)
        PREFETCH_8_LANES(k, K, PREFETCH_L1, sub_outputs, _MM_HINT_T0);
        PREFETCH_8_LANES(k, K, PREFETCH_L2, sub_outputs, _MM_HINT_T1);
        PREFETCH_8_LANES(k, K, PREFETCH_L3, sub_outputs, _MM_HINT_T2);
        
        //======================================================================
        // STAGE 1: Load inputs (IDENTICAL)
        //======================================================================
        __m256d x[8];
        LOAD_8_LANES_AVX2(k, K, sub_outputs, x);
        
        //======================================================================
        // STAGE 2: Apply precomputed twiddles (IDENTICAL)
        //======================================================================
        APPLY_STAGE_TWIDDLES_AVX2(k, x, stage_tw);
        
        //======================================================================
        // STAGE 3: Split-radix decomposition (uses inverse rotation mask)
        //======================================================================
        
        __m256d e[4], o[4];
        
        // Even radix-4: lanes [0,2,4,6]
        RADIX4_CORE_AVX2(x[0], x[2], x[4], x[6], e[0], e[1], e[2], e[3], rot_mask);
        
        // Odd radix-4: lanes [1,3,5,7]
        RADIX4_CORE_AVX2(x[1], x[3], x[5], x[7], o[0], o[1], o[2], o[3], rot_mask);
        
        //======================================================================
        // STAGE 4: Apply W_8 geometric twiddles (INVERSE) ⚡ ONLY DIFFERENCE
        //======================================================================
        APPLY_W8_TWIDDLES_BV_AVX2(o);
        
        //======================================================================
        // STAGE 5: Final radix-2 combination and store (IDENTICAL)
        //======================================================================
        if (use_streaming)
        {
            FINAL_RADIX2_AVX2_STREAM(e, o, output_buffer, k, K);
        }
        else
        {
            FINAL_RADIX2_AVX2(e, o, output_buffer, k, K);
        }
    }
    
    if (use_streaming)
    {
        _mm_sfence();
    }
    
#endif // __AVX2__

    //==========================================================================
    // SCALAR TAIL: Process remaining single butterflies
    //==========================================================================
    for (; k < K; k++)
    {
        //======================================================================
        // STAGE 1: Load input lanes (IDENTICAL)
        //======================================================================
        fft_data x[8];
        x[0] = sub_outputs[k];
        for (int j = 1; j <= 7; j++)
        {
            x[j] = sub_outputs[k + j * K];
        }
        
        //======================================================================
        // STAGE 2: Apply precomputed twiddles (IDENTICAL)
        //======================================================================
        APPLY_STAGE_TWIDDLES_SCALAR(k, x, stage_tw);
        
        //======================================================================
        // STAGE 3: Even radix-4 [0,2,4,6] (uses +1 rotation)
        //======================================================================
        fft_data e[4];
        RADIX4_CORE_SCALAR(
            x[0].re, x[0].im, x[2].re, x[2].im,
            x[4].re, x[4].im, x[6].re, x[6].im,
            e[0].re, e[0].im, e[1].re, e[1].im,
            e[2].re, e[2].im, e[3].re, e[3].im,
            +1  // Inverse rotation sign ⚡ ONLY DIFFERENCE
        );
        
        //======================================================================
        // STAGE 4: Odd radix-4 [1,3,5,7] (uses +1 rotation)
        //======================================================================
        fft_data o[4];
        RADIX4_CORE_SCALAR(
            x[1].re, x[1].im, x[3].re, x[3].im,
            x[5].re, x[5].im, x[7].re, x[7].im,
            o[0].re, o[0].im, o[1].re, o[1].im,
            o[2].re, o[2].im, o[3].re, o[3].im,
            +1  // Inverse rotation sign ⚡ ONLY DIFFERENCE
        );
        
        //======================================================================
        // STAGE 5: Apply W_8 geometric twiddles (INVERSE) ⚡ ONLY DIFFERENCE
        //======================================================================
        APPLY_W8_TWIDDLES_BV_SCALAR(o);
        
        //======================================================================
        // STAGE 6: Final radix-2 combination and store (IDENTICAL)
        //======================================================================
        FINAL_RADIX2_SCALAR(e, o, output_buffer, k, K);
    }
}

//==============================================================================
// SUMMARY: Forward vs Inverse
//==============================================================================

/**
 * IDENTICAL CODE (~99%):
 * - All load/store patterns
 * - All prefetching
 * - All twiddle application (stage_tw)
 * - Split-radix decomposition structure
 * - Final radix-2 combination
 * 
 * DIFFERENT (2 things):
 * 1. Radix-4 rotation mask:
 *    - Forward: rot_mask = (0.0, -0.0, 0.0, -0.0)  // -i rotation
 *    - Inverse: rot_mask = (-0.0, 0.0, -0.0, 0.0)  // +i rotation
 * 
 * 2. W_8 twiddle application:
 *    - Forward: APPLY_W8_TWIDDLES_FV_* uses (√2/2, -√2/2), (0, -1), (-√2/2, -√2/2)
 *    - Inverse: APPLY_W8_TWIDDLES_BV_* uses (√2/2, +√2/2), (0, +1), (-√2/2, +√2/2)
 * 
 * TWIDDLE DIFFERENCE (computed by planning):
 * - Forward stage_tw:  exp(-2πijk/N) - negative sign
 * - Inverse stage_tw:  exp(+2πijk/N) - positive sign
 * 
 * This is why _fv and _bv can share 99% of code via macros!
 */

//==============================================================================
// PERFORMANCE NOTES
//==============================================================================

/**
 * Performance is IDENTICAL to forward butterfly:
 * - AVX2: ~20 cycles/butterfly
 * - Scalar: ~40 cycles/butterfly
 * 
 * The direction change has ZERO performance impact because:
 * 1. Same number of operations
 * 2. Only constant values differ (XOR masks, W_8 constants)
 * 3. Constants are loaded once and reused
 * 
 */