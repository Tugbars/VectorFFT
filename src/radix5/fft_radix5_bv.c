//==============================================================================
// fft_radix5_bv.c - Inverse Radix-5 Butterfly (Precomputed Twiddles)
//==============================================================================
//
// DESIGN: Identical to fft_radix5_fv.c except:
// - Uses RADIX5_BUTTERFLY_BV_* instead of RADIX5_BUTTERFLY_FV_*
// - Rotation uses +i instead of -i
// - Twiddles have inverse sign: exp(+2πik/N)
//

#include "fft_radix5.h"
#include "simd_math.h"
#include "fft_radix5_macros.h"

// Non-temporal store threshold
#define STREAM_THRESHOLD 8192

//==============================================================================
// INVERSE RADIX-5 BUTTERFLY - Main Function
//==============================================================================

/**
 * @brief Ultra-optimized inverse radix-5 butterfly
 * 
 * DIFFERENCE FROM FORWARD:
 * - Uses +i rotation instead of -i
 * - Twiddles: Precomputed with positive sign (exp(+2πik/N))
 * 
 * Everything else is IDENTICAL to fft_radix5_fv()
 */
void fft_radix5_bv(
    fft_data *restrict output_buffer,
    const fft_data *restrict sub_outputs,
    const fft_data *restrict stage_tw,
    int sub_len)
{
    const int K = sub_len;
    int k = 0;

#ifdef __AVX2__
    //==========================================================================
    // AVX2 PATH (IDENTICAL except rotation)
    //==========================================================================
    
    const int use_streaming = (K >= STREAM_THRESHOLD);

    //==========================================================================
    // MAIN LOOP: 2X UNROLL (IDENTICAL except rotation)
    //==========================================================================
    for (; k + 1 < K; k += 2)
    {
        // Prefetch ahead (IDENTICAL)
        PREFETCH_5_LANES_R5(k, K, PREFETCH_L1_R5, sub_outputs, _MM_HINT_T0);
        PREFETCH_5_LANES_R5(k, K, PREFETCH_L2_R5, sub_outputs, _MM_HINT_T1);
        PREFETCH_5_LANES_R5(k, K, PREFETCH_L3_R5, sub_outputs, _MM_HINT_T2);
        
        //======================================================================
        // STAGE 1: Load inputs (IDENTICAL)
        //======================================================================
        __m256d a, b, c, d, e;
        LOAD_5_LANES_AVX2(k, K, sub_outputs, a, b, c, d, e);
        
        //======================================================================
        // STAGE 2: Apply precomputed stage twiddles (IDENTICAL)
        //======================================================================
        __m256d b2, c2, d2, e2;
        APPLY_STAGE_TWIDDLES_R5_AVX2(k, b, c, d, e, stage_tw, b2, c2, d2, e2);
        
        //======================================================================
        // STAGE 3: Radix-5 butterfly (inverse) ⚡ ONLY DIFFERENCE
        //======================================================================
        __m256d y0, y1, y2, y3, y4;
        RADIX5_BUTTERFLY_BV_AVX2(a, b2, c2, d2, e2, y0, y1, y2, y3, y4);
        
        //======================================================================
        // STAGE 4: Store results (IDENTICAL)
        //======================================================================
        if (use_streaming)
        {
            STORE_5_LANES_AVX2_STREAM(k, K, output_buffer, y0, y1, y2, y3, y4);
        }
        else
        {
            STORE_5_LANES_AVX2(k, K, output_buffer, y0, y1, y2, y3, y4);
        }
    }
    
    if (use_streaming)
    {
        _mm_sfence();
    }
    
#endif // __AVX2__

    //==========================================================================
    // SCALAR TAIL: ⚡ ONLY DIFFERENCE: INVERSE ROTATION
    //==========================================================================
    for (; k < K; ++k)
    {
        //======================================================================
        // STAGE 1: Load inputs (IDENTICAL)
        //======================================================================
        fft_data a = sub_outputs[k];
        fft_data b = sub_outputs[k + K];
        fft_data c = sub_outputs[k + 2*K];
        fft_data d = sub_outputs[k + 3*K];
        fft_data e = sub_outputs[k + 4*K];
        
        //======================================================================
        // STAGE 2: Apply precomputed stage twiddles (IDENTICAL)
        //======================================================================
        const fft_data *tw = &stage_tw[k * 4];
        
        double b2r = b.re * tw[0].re - b.im * tw[0].im;
        double b2i = b.re * tw[0].im + b.im * tw[0].re;
        
        double c2r = c.re * tw[1].re - c.im * tw[1].im;
        double c2i = c.re * tw[1].im + c.im * tw[1].re;
        
        double d2r = d.re * tw[2].re - d.im * tw[2].im;
        double d2i = d.re * tw[2].im + d.im * tw[2].re;
        
        double e2r = e.re * tw[3].re - e.im * tw[3].im;
        double e2i = e.re * tw[3].im + e.im * tw[3].re;
        
        //======================================================================
        // STAGE 3: Radix-5 butterfly (inverse) ⚡ ONLY DIFFERENCE
        //======================================================================
        fft_data y0, y1, y2, y3, y4;
        RADIX5_BUTTERFLY_BV_SCALAR(a, b2r, b2i, c2r, c2i, d2r, d2i, e2r, e2i,
                                    y0, y1, y2, y3, y4);
        
        //======================================================================
        // STAGE 4: Store results (IDENTICAL)
        //======================================================================
        output_buffer[k] = y0;
        output_buffer[k + K] = y1;
        output_buffer[k + 2*K] = y2;
        output_buffer[k + 3*K] = y3;
        output_buffer[k + 4*K] = y4;
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
 * - Butterfly core arithmetic (sums/diffs, geometric constants)
 * - Output assembly
 * 
 * DIFFERENT (1 thing):
 * - Rotation direction (±i multiplication):
 *   - Forward: r1 = (base1i, -base1r), r2 = (-base2i, base2r)
 *   - Inverse: r1 = (-base1i, base1r), r2 = (base2i, -base2r)
 * 
 * TWIDDLE DIFFERENCE (computed by planning):
 * - Forward stage_tw:  exp(-2πik/N) - negative sign
 * - Inverse stage_tw:  exp(+2πik/N) - positive sign
 * 
 * GEOMETRIC CONSTANTS (identical for both):
 * - C5_1, C5_2, S5_1, S5_2 are direction-independent
 * - Only the rotation (±i) changes based on direction
 * 
 * This is why _fv and _bv can share 99% of code via macros!
 */