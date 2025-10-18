//==============================================================================
// fft_radix7_bv.c - INVERSE Radix-7 Rader Butterfly (Precomputed Twiddles)
//==============================================================================
//
// DESIGN PRINCIPLES:
// 1. No direction parameter - always INVERSE FFT
// 2. stage_tw precomputed (from Twiddle Manager) - CONJUGATED
// 3. rader_tw precomputed (from Rader Manager) with INVERSE sign
// ...

/**
 * @brief Ultra-optimized INVERSE radix-7 Rader butterfly
 * 
 * ASSUMPTIONS:
 * - stage_tw is NEVER NULL (precomputed, K×6 values, CONJUGATED)
 * - rader_tw is NEVER NULL (precomputed, 6 values with INVERSE sign)
 * - Direction is ALWAYS INVERSE
 * 
 * TWIDDLE LAYOUT:
 * - stage_tw[k*6 + (r-1)] = W_N^(-r*k) = conj(W_N^(r*k)) for r=1..6
 * - rader_tw[q] = exp(+2πi * out_perm[q] / 7) for q=0..5  [NOTE: positive sign]
 */

#include "fft_radix7.h"
#include "simd_math.h"
#include "fft_radix7_macros.h"

//==============================================================================
// FORWARD RADIX-7 RADER BUTTERFLY - Main Function
//==============================================================================

/**
 * @brief Ultra-optimized forward radix-7 Rader butterfly
 * 
 * ASSUMPTIONS:
 * - stage_tw is NEVER NULL (precomputed, K×6 values)
 * - rader_tw is NEVER NULL (precomputed, 6 values with forward sign)
 * - Direction is ALWAYS forward
 * 
 * TWIDDLE LAYOUT:
 * - stage_tw[k*6 + (r-1)] = W_N^(r*k) for r=1..6
 * - rader_tw[q] = exp(-2πi * out_perm[q] / 7) for q=0..5
 *   where out_perm = [1,5,4,6,2,3]
 * 
 * RADER ALGORITHM:
 * - Prime 7, generator g=3
 * - perm_in = [1,3,2,6,4,5]
 * - out_perm = [1,5,4,6,2,3]
 * - 6-point cyclic convolution
 */
void fft_radix7_bv(
    fft_data *restrict output_buffer,
    const fft_data *restrict sub_outputs,
    const fft_data *restrict stage_tw,
    const fft_data *restrict rader_tw,
    int sub_len)
{
    const int K = sub_len;
    int k = 0;

#ifdef __AVX2__
    //==========================================================================
    // AVX2 PATH
    //==========================================================================
    
    // Broadcast Rader twiddles for AVX2
    __m256d tw_brd[6];
    BROADCAST_RADER_TWIDDLES_R7(rader_tw, tw_brd);

    //==========================================================================
    // MAIN LOOP: 8X UNROLL (processes 8 butterflies per iteration)
    //==========================================================================
    for (; k + 7 < K; k += 8)
    {
        // Prefetch ahead
        PREFETCH_7_LANES_R7(k, K, PREFETCH_L1_R7, sub_outputs, _MM_HINT_T0);
        
        //======================================================================
        // Process 4 pairs of butterflies (8 total)
        //======================================================================
        for (int p = 0; p < 4; p++)
        {
            int kk = k + 2*p;
            
            //==================================================================
            // STAGE 1: Load inputs
            //==================================================================
            __m256d x0, x1, x2, x3, x4, x5, x6;
            LOAD_7_LANES_AVX2(kk, K, sub_outputs, x0, x1, x2, x3, x4, x5, x6);
            
            //==================================================================
            // STAGE 2: Apply precomputed stage twiddles
            //==================================================================
            APPLY_STAGE_TWIDDLES_R7_AVX2(kk, x1, x2, x3, x4, x5, x6, stage_tw);
            
            //==================================================================
            // STAGE 3: Compute y0 (DC component)
            //==================================================================
            __m256d y0;
            COMPUTE_Y0_R7_AVX2(x0, x1, x2, x3, x4, x5, x6, y0);
            
            //==================================================================
            // STAGE 4: Rader input permutation
            //==================================================================
            __m256d tx0, tx1, tx2, tx3, tx4, tx5;
            PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5);
            
            //==================================================================
            // STAGE 5: Cyclic convolution (using precomputed rader_tw)
            //==================================================================
            __m256d v0, v1, v2, v3, v4, v5;
            RADER_CONVOLUTION_R7_AVX2(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd,
                                       v0, v1, v2, v3, v4, v5);
            
            //==================================================================
            // STAGE 6: Assemble outputs
            //==================================================================
            __m256d y1, y2, y3, y4, y5, y6;
            ASSEMBLE_OUTPUTS_R7_AVX2(x0, v0, v1, v2, v3, v4, v5,
                                      y0, y1, y2, y3, y4, y5, y6);
            
            //==================================================================
            // STAGE 7: Store results
            //==================================================================
            STORE_7_LANES_AVX2(kk, K, output_buffer, y0, y1, y2, y3, y4, y5, y6);
        }
    }
    
    //==========================================================================
    // CLEANUP: 2X UNROLL
    //==========================================================================
    for (; k + 1 < K; k += 2)
    {
        //======================================================================
        // STAGE 1: Load inputs
        //======================================================================
        __m256d x0, x1, x2, x3, x4, x5, x6;
        LOAD_7_LANES_AVX2(k, K, sub_outputs, x0, x1, x2, x3, x4, x5, x6);
        
        //======================================================================
        // STAGE 2: Apply precomputed stage twiddles
        //======================================================================
        APPLY_STAGE_TWIDDLES_R7_AVX2(k, x1, x2, x3, x4, x5, x6, stage_tw);
        
        //======================================================================
        // STAGE 3: Compute y0 + Rader convolution + Output assembly
        //======================================================================
        __m256d y0;
        COMPUTE_Y0_R7_AVX2(x0, x1, x2, x3, x4, x5, x6, y0);
        
        __m256d tx0, tx1, tx2, tx3, tx4, tx5;
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, tx0, tx1, tx2, tx3, tx4, tx5);
        
        __m256d v0, v1, v2, v3, v4, v5;
        RADER_CONVOLUTION_R7_AVX2(tx0, tx1, tx2, tx3, tx4, tx5, tw_brd,
                                   v0, v1, v2, v3, v4, v5);
        
        __m256d y1, y2, y3, y4, y5, y6;
        ASSEMBLE_OUTPUTS_R7_AVX2(x0, v0, v1, v2, v3, v4, v5,
                                  y0, y1, y2, y3, y4, y5, y6);
        
        //======================================================================
        // STAGE 4: Store results
        //======================================================================
        STORE_7_LANES_AVX2(k, K, output_buffer, y0, y1, y2, y3, y4, y5, y6);
    }
    
#endif // __AVX2__

    //==========================================================================
    // SCALAR TAIL: Process remaining single butterflies
    //==========================================================================
    for (; k < K; ++k)
    {
        //======================================================================
        // STAGE 1: Load inputs
        //======================================================================
        fft_data x0 = sub_outputs[k + 0*K];
        fft_data x1 = sub_outputs[k + 1*K];
        fft_data x2 = sub_outputs[k + 2*K];
        fft_data x3 = sub_outputs[k + 3*K];
        fft_data x4 = sub_outputs[k + 4*K];
        fft_data x5 = sub_outputs[k + 5*K];
        fft_data x6 = sub_outputs[k + 6*K];
        
        //======================================================================
        // STAGE 2: Apply precomputed stage twiddles
        //======================================================================
        if (sub_len > 1)
        {
            const fft_data *tw = &stage_tw[6*k];
            
            fft_data t;
            t = x1; x1.re = t.re * tw[0].re - t.im * tw[0].im; x1.im = t.re * tw[0].im + t.im * tw[0].re;
            t = x2; x2.re = t.re * tw[1].re - t.im * tw[1].im; x2.im = t.re * tw[1].im + t.im * tw[1].re;
            t = x3; x3.re = t.re * tw[2].re - t.im * tw[2].im; x3.im = t.re * tw[2].im + t.im * tw[2].re;
            t = x4; x4.re = t.re * tw[3].re - t.im * tw[3].im; x4.im = t.re * tw[3].im + t.im * tw[3].re;
            t = x5; x5.re = t.re * tw[4].re - t.im * tw[4].im; x5.im = t.re * tw[4].im + t.im * tw[4].re;
            t = x6; x6.re = t.re * tw[5].re - t.im * tw[5].im; x6.im = t.re * tw[5].im + t.im * tw[5].re;
        }
        
        //======================================================================
        // STAGE 3: Compute y0 (DC component)
        //======================================================================
        fft_data y0;
        y0.re = x0.re + x1.re + x2.re + x3.re + x4.re + x5.re + x6.re;
        y0.im = x0.im + x1.im + x2.im + x3.im + x4.im + x5.im + x6.im;
        
        //======================================================================
        // STAGE 4: Rader input permutation (perm_in = [1,3,2,6,4,5])
        //======================================================================
        fft_data tx[6];
        PERMUTE_INPUTS_R7(x1, x2, x3, x4, x5, x6, 
                          tx[0], tx[1], tx[2], tx[3], tx[4], tx[5]);
        
        //======================================================================
        // STAGE 5: Cyclic convolution (using precomputed rader_tw)
        //======================================================================
        fft_data v[6];
        RADER_CONVOLUTION_R7_SCALAR(tx, rader_tw, v);
        
        //======================================================================
        // STAGE 6: Assemble outputs (out_perm = [1,5,4,6,2,3])
        //======================================================================
        fft_data y1 = {x0.re + v[0].re, x0.im + v[0].im};
        fft_data y5 = {x0.re + v[1].re, x0.im + v[1].im};
        fft_data y4 = {x0.re + v[2].re, x0.im + v[2].im};
        fft_data y6 = {x0.re + v[3].re, x0.im + v[3].im};
        fft_data y2 = {x0.re + v[4].re, x0.im + v[4].im};
        fft_data y3 = {x0.re + v[5].re, x0.im + v[5].im};
        
        //======================================================================
        // STAGE 7: Store results
        //======================================================================
        output_buffer[k + 0*K] = y0;
        output_buffer[k + 1*K] = y1;
        output_buffer[k + 2*K] = y2;
        output_buffer[k + 3*K] = y3;
        output_buffer[k + 4*K] = y4;
        output_buffer[k + 5*K] = y5;
        output_buffer[k + 6*K] = y6;
    }
}