//==============================================================================
// fft_radix3_bv.c - Inverse Radix-3 Butterfly (Precomputed Twiddles)
//==============================================================================
//
// DESIGN: Identical to fft_radix3_fv.c except:
// - Uses RADIX3_ROTATE_INVERSE_* instead of RADIX3_ROTATE_FORWARD_*
// - Twiddles have inverse sign: W_N^k = exp(+2πik/N)
//

#include "fft_radix3.h"
#include "simd_math.h"
#include "fft_radix3_macros.h"

// Non-temporal store threshold
#define STREAM_THRESHOLD 8192

//==============================================================================
// INVERSE RADIX-3 BUTTERFLY - Main Function
//==============================================================================

/**
 * @brief Ultra-optimized inverse radix-3 butterfly
 * 
 * DIFFERENCE FROM FORWARD:
 * - Uses +i rotation instead of -i
 * - Twiddles: Precomputed with positive sign (exp(+2πik/N))
 * 
 * Everything else is IDENTICAL to fft_radix3_fv()
 */
void fft_radix3_bv(
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
    const __m256d v_sqrt3_2 = _mm256_set1_pd(S_SQRT3_2);

    // Main loop: process 2 butterflies per iteration
    for (; k + 1 < K; k += 2)
    {
        // Prefetch ahead (IDENTICAL)
        PREFETCH_3_LANES(k, K, PREFETCH_L1, sub_outputs, _MM_HINT_T0);
        PREFETCH_3_LANES(k, K, PREFETCH_L2, sub_outputs, _MM_HINT_T1);
        PREFETCH_3_LANES(k, K, PREFETCH_L3, sub_outputs, _MM_HINT_T2);
        
        //======================================================================
        // STAGE 1: Load inputs (IDENTICAL)
        //======================================================================
        __m256d a, b, c;
        LOAD_3_LANES_AVX2(k, K, sub_outputs, a, b, c);
        
        //======================================================================
        // STAGE 2: Apply precomputed twiddles (IDENTICAL)
        //======================================================================
        __m256d tw_b, tw_c;
        APPLY_STAGE_TWIDDLES_AVX2(k, b, c, stage_tw, tw_b, tw_c);
        
        //======================================================================
        // STAGE 3: Butterfly core arithmetic (IDENTICAL)
        //======================================================================
        __m256d sum, dif, common;
        RADIX3_BUTTERFLY_CORE_AVX2(a, tw_b, tw_c, sum, dif, common);
        
        //======================================================================
        // STAGE 4: Inverse rotation (+i * sqrt(3)/2) ⚡ ONLY DIFFERENCE
        //======================================================================
        __m256d scaled_rot;
        RADIX3_ROTATE_INVERSE_AVX2(dif, scaled_rot, v_sqrt3_2);
        
        //======================================================================
        // STAGE 5: Assemble outputs (IDENTICAL)
        //======================================================================
        __m256d y0, y1, y2;
        RADIX3_ASSEMBLE_OUTPUTS_AVX2(a, sum, common, scaled_rot, y0, y1, y2);
        
        //======================================================================
        // STAGE 6: Store results (IDENTICAL)
        //======================================================================
        if (use_streaming)
        {
            STORE_3_LANES_AVX2_STREAM(k, K, output_buffer, y0, y1, y2);
        }
        else
        {
            STORE_3_LANES_AVX2(k, K, output_buffer, y0, y1, y2);
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
        RADIX3_BUTTERFLY_SCALAR_BV(k, K, sub_outputs, stage_tw, output_buffer);
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
 * - Butterfly core (sum, dif, common computation)
 * - Output assembly
 * 
 * DIFFERENT (1 macro call):
 * - Rotation:
 *   Forward:  RADIX3_ROTATE_FORWARD_AVX2  (-i * dif * sqrt(3)/2)
 *   Inverse:  RADIX3_ROTATE_INVERSE_AVX2  (+i * dif * sqrt(3)/2)
 * 
 * TWIDDLE DIFFERENCE (computed by planning):
 * - Forward stage_tw:  exp(-2πijk/N) - negative sign
 * - Inverse stage_tw:  exp(+2πijk/N) - positive sign
 * 
 * This is why _fv and _bv can share 99% of code via macros!
 * 
 * ROTATION DETAILS:
 * - Forward: (-i * dif) * sqrt(3)/2 = (dif_im, -dif_re) * sqrt(3)/2
 * - Inverse: (+i * dif) * sqrt(3)/2 = (-dif_im, dif_re) * sqrt(3)/2
 * 
 * In AVX2, this is just a different XOR mask:
 * - Forward: XOR with (0.0, -0.0, 0.0, -0.0)
 * - Inverse: XOR with (-0.0, 0.0, -0.0, 0.0)
 */

//==============================================================================
// PERFORMANCE NOTES
//==============================================================================

/**
 * Performance is IDENTICAL to forward butterfly:
 * - AVX2: ~6 cycles/butterfly
 * - Scalar: ~12 cycles/butterfly
 * 
 * The rotation direction change has ZERO performance impact because:
 * 1. Both use the same number of operations (1 XOR + 1 PERMUTE + 1 MUL)
 * 2. Only the XOR mask constant differs
 * 3. Constant is loaded once and reused
 * 
 */