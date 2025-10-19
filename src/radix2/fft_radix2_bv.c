//==============================================================================
// fft_radix2_bv.c - Inverse Radix-2 Butterfly (Precomputed Twiddles)
//==============================================================================
// 
// DESIGN: Identical to fft_radix2_fv.c except:
// - k=N/4 rotation uses +i instead of -i
// - Twiddles have inverse sign: W_N^k = exp(+2πik/N)
//

#include "fft_radix2.h"
#include "simd_math.h"
#include "fft_radix2_macros.h"

//==============================================================================
// INVERSE RADIX-2 BUTTERFLY - Main Function
//==============================================================================

/**
 * @brief Ultra-optimized inverse radix-2 butterfly
 * 
 * DIFFERENCE FROM FORWARD:
 * - k=N/4: Uses +i rotation instead of -i
 * - Twiddles: Precomputed with positive sign (exp(+2πik/N))
 * 
 * Everything else is IDENTICAL to fft_radix2_fv()
 */
void fft_radix2_bv(
    fft_data *restrict output_buffer,
    const fft_data *restrict sub_outputs,
    const fft_data *restrict stage_tw,
    int sub_len)
{
    const int half = sub_len;
    
    //==========================================================================
    // STAGE 0: k=0 (W^0 = 1, IDENTICAL to forward)
    //==========================================================================
    {
        fft_data even_0 = sub_outputs[0];
        fft_data odd_0 = sub_outputs[half];
        
        output_buffer[0].re = even_0.re + odd_0.re;
        output_buffer[0].im = even_0.im + odd_0.im;
        output_buffer[half].re = even_0.re - odd_0.re;
        output_buffer[half].im = even_0.im - odd_0.im;
    }
    
    //==========================================================================
    // STAGE 1: k=N/4 (W^(N/4) = +i for INVERSE FFT) ⚡ ONLY DIFFERENCE
    //==========================================================================
    int k_quarter = 0;
    if ((half & 1) == 0) {
        k_quarter = half >> 1;
        
        fft_data even_q = sub_outputs[k_quarter];
        fft_data odd_q = sub_outputs[half + k_quarter];
        
        // ⚡ Inverse: W^(N/4) = exp(+2πi/4) = exp(+πi/2) = +i
        // Multiply by +i: (a + bi) * i = -b + ai
        double rotated_re = -odd_q.im;  // -b
        double rotated_im = odd_q.re;   // a
        
        output_buffer[k_quarter].re = even_q.re + rotated_re;
        output_buffer[k_quarter].im = even_q.im + rotated_im;
        output_buffer[half + k_quarter].re = even_q.re - rotated_re;
        output_buffer[half + k_quarter].im = even_q.im - rotated_im;
    }
    
    //==========================================================================
    // STAGE 2: General case (IDENTICAL to forward - twiddles differ)
    //==========================================================================
    
    int k = 1;
    int range1_end = k_quarter ? k_quarter : half;
    
#ifdef __AVX512F__
    //==========================================================================
    // AVX-512 PATH (IDENTICAL to forward)
    //==========================================================================
    
    for (; k + 15 < range1_end; k += 16) {
        PREFETCH_NEXT_AVX2(k, 32, sub_outputs, stage_tw, half);
        
        __m512d e0 = load4_aos(&sub_outputs[k+0]);
        __m512d e1 = load4_aos(&sub_outputs[k+4]);
        __m512d e2 = load4_aos(&sub_outputs[k+8]);
        __m512d e3 = load4_aos(&sub_outputs[k+12]);
        
        __m512d o0 = load4_aos(&sub_outputs[k+0+half]);
        __m512d o1 = load4_aos(&sub_outputs[k+4+half]);
        __m512d o2 = load4_aos(&sub_outputs[k+8+half]);
        __m512d o3 = load4_aos(&sub_outputs[k+12+half]);
        
        __m512d w0 = load4_aos(&stage_tw[k+0]);
        __m512d w1 = load4_aos(&stage_tw[k+4]);
        __m512d w2 = load4_aos(&stage_tw[k+8]);
        __m512d w3 = load4_aos(&stage_tw[k+12]);
        
        __m512d x00, x10, x01, x11, x02, x12, x03, x13;
        RADIX2_BUTTERFLY_AVX512(e0, o0, w0, x00, x10);
        RADIX2_BUTTERFLY_AVX512(e1, o1, w1, x01, x11);
        RADIX2_BUTTERFLY_AVX512(e2, o2, w2, x02, x12);
        RADIX2_BUTTERFLY_AVX512(e3, o3, w3, x03, x13);
        
        STOREU_PD512(&output_buffer[k+0].re, x00);
        STOREU_PD512(&output_buffer[k+4].re, x01);
        STOREU_PD512(&output_buffer[k+8].re, x02);
        STOREU_PD512(&output_buffer[k+12].re, x03);
        
        STOREU_PD512(&output_buffer[k+0+half].re, x10);
        STOREU_PD512(&output_buffer[k+4+half].re, x11);
        STOREU_PD512(&output_buffer[k+8+half].re, x12);
        STOREU_PD512(&output_buffer[k+12+half].re, x13);
    }
#endif // __AVX512F__
    
#ifdef __AVX2__
    //==========================================================================
    // AVX2 PATH (IDENTICAL to forward)
    //==========================================================================
    
    for (; k + 7 < range1_end; k += 8) {
        PREFETCH_NEXT_AVX2(k, 16, sub_outputs, stage_tw, half);
        
        __m256d e0, e1, e2, e3, o0, o1, o2, o3, w0, w1, w2, w3;
        LOAD_STAGE_INPUTS_AVX2(k, sub_outputs, stage_tw, half,
                              e0, e1, e2, e3, o0, o1, o2, o3, w0, w1, w2, w3);
        
        __m256d x00, x10, x01, x11, x02, x12, x03, x13;
        RADIX2_BUTTERFLY_AVX2(e0, o0, w0, x00, x10);
        RADIX2_BUTTERFLY_AVX2(e1, o1, w1, x01, x11);
        RADIX2_BUTTERFLY_AVX2(e2, o2, w2, x02, x12);
        RADIX2_BUTTERFLY_AVX2(e3, o3, w3, x03, x13);
        
        STORE_BUTTERFLY_OUTPUTS_AVX2(k, output_buffer, half,
                                     x00, x01, x02, x03, x10, x11, x12, x13);
    }
    
    for (; k + 1 < range1_end; k += 2) {
        __m256d even = load2_aos(&sub_outputs[k], &sub_outputs[k+1]);
        __m256d odd = load2_aos(&sub_outputs[k+half], &sub_outputs[k+half+1]);
        __m256d w = load2_aos(&stage_tw[k], &stage_tw[k+1]);
        
        __m256d x0, x1;
        RADIX2_BUTTERFLY_AVX2(even, odd, w, x0, x1);
        
        STOREU_PD(&output_buffer[k].re, x0);
        STOREU_PD(&output_buffer[k+half].re, x1);
    }
#endif // __AVX2__
    
    //==========================================================================
    // SSE2 TAIL (IDENTICAL to forward)
    //==========================================================================
    for (; k < range1_end; k++) {
        __m128d even = LOADU_SSE2(&sub_outputs[k].re);
        __m128d odd = LOADU_SSE2(&sub_outputs[k+half].re);
        __m128d w = LOADU_SSE2(&stage_tw[k].re);
        
        __m128d x0, x1;
        RADIX2_BUTTERFLY_SSE2(even, odd, w, x0, x1);
        
        STOREU_SSE2(&output_buffer[k].re, x0);
        STOREU_SSE2(&output_buffer[k+half].re, x1);
    }
    
    //==========================================================================
    // RANGE 2 (IDENTICAL to forward)
    //==========================================================================
    if (k_quarter) {
        k = k_quarter + 1;
        
#ifdef __AVX512F__
        for (; k + 15 < half; k += 16) {
            PREFETCH_NEXT_AVX2(k, 32, sub_outputs, stage_tw, half);
            
            __m512d e0 = load4_aos(&sub_outputs[k+0]);
            __m512d e1 = load4_aos(&sub_outputs[k+4]);
            __m512d e2 = load4_aos(&sub_outputs[k+8]);
            __m512d e3 = load4_aos(&sub_outputs[k+12]);
            
            __m512d o0 = load4_aos(&sub_outputs[k+0+half]);
            __m512d o1 = load4_aos(&sub_outputs[k+4+half]);
            __m512d o2 = load4_aos(&sub_outputs[k+8+half]);
            __m512d o3 = load4_aos(&sub_outputs[k+12+half]);
            
            __m512d w0 = load4_aos(&stage_tw[k+0]);
            __m512d w1 = load4_aos(&stage_tw[k+4]);
            __m512d w2 = load4_aos(&stage_tw[k+8]);
            __m512d w3 = load4_aos(&stage_tw[k+12]);
            
            __m512d x00, x10, x01, x11, x02, x12, x03, x13;
            RADIX2_BUTTERFLY_AVX512(e0, o0, w0, x00, x10);
            RADIX2_BUTTERFLY_AVX512(e1, o1, w1, x01, x11);
            RADIX2_BUTTERFLY_AVX512(e2, o2, w2, x02, x12);
            RADIX2_BUTTERFLY_AVX512(e3, o3, w3, x03, x13);
            
            STOREU_PD512(&output_buffer[k+0].re, x00);
            STOREU_PD512(&output_buffer[k+4].re, x01);
            STOREU_PD512(&output_buffer[k+8].re, x02);
            STOREU_PD512(&output_buffer[k+12].re, x03);
            STOREU_PD512(&output_buffer[k+0+half].re, x10);
            STOREU_PD512(&output_buffer[k+4+half].re, x11);
            STOREU_PD512(&output_buffer[k+8+half].re, x12);
            STOREU_PD512(&output_buffer[k+12+half].re, x13);
        }
#endif
        
#ifdef __AVX2__
        for (; k + 7 < half; k += 8) {
            PREFETCH_NEXT_AVX2(k, 16, sub_outputs, stage_tw, half);
            
            __m256d e0, e1, e2, e3, o0, o1, o2, o3, w0, w1, w2, w3;
            LOAD_STAGE_INPUTS_AVX2(k, sub_outputs, stage_tw, half,
                                  e0, e1, e2, e3, o0, o1, o2, o3, w0, w1, w2, w3);
            
            __m256d x00, x10, x01, x11, x02, x12, x03, x13;
            RADIX2_BUTTERFLY_AVX2(e0, o0, w0, x00, x10);
            RADIX2_BUTTERFLY_AVX2(e1, o1, w1, x01, x11);
            RADIX2_BUTTERFLY_AVX2(e2, o2, w2, x02, x12);
            RADIX2_BUTTERFLY_AVX2(e3, o3, w3, x03, x13);
            
            STORE_BUTTERFLY_OUTPUTS_AVX2(k, output_buffer, half,
                                         x00, x01, x02, x03, x10, x11, x12, x13);
        }
        
        for (; k + 1 < half; k += 2) {
            __m256d even = load2_aos(&sub_outputs[k], &sub_outputs[k+1]);
            __m256d odd = load2_aos(&sub_outputs[k+half], &sub_outputs[k+half+1]);
            __m256d w = load2_aos(&stage_tw[k], &stage_tw[k+1]);
            
            __m256d x0, x1;
            RADIX2_BUTTERFLY_AVX2(even, odd, w, x0, x1);
            
            STOREU_PD(&output_buffer[k].re, x0);
            STOREU_PD(&output_buffer[k+half].re, x1);
        }
#endif
        
        for (; k < half; k++) {
            __m128d even = LOADU_SSE2(&sub_outputs[k].re);
            __m128d odd = LOADU_SSE2(&sub_outputs[k+half].re);
            __m128d w = LOADU_SSE2(&stage_tw[k].re);
            
            __m128d x0, x1;
            RADIX2_BUTTERFLY_SSE2(even, odd, w, x0, x1);
            
            STOREU_SSE2(&output_buffer[k].re, x0);
            STOREU_SSE2(&output_buffer[k+half].re, x1);
        }
    }
}

//==============================================================================
// SUMMARY: Forward vs Inverse
//==============================================================================

/**
 * IDENTICAL CODE (~99%):
 * - All SIMD paths
 * - All macros
 * - All loop structures
 * - Butterfly arithmetic
 * 
 * DIFFERENT (3 lines total):
 * - k=N/4 rotation: -i (forward) vs +i (inverse)
 * 
 * TWIDDLE DIFFERENCE (computed by planning):
 * - Forward:  exp(-2πik/N) - negative sign
 * - Inverse:  exp(+2πik/N) - positive sign
 * 
 * This is why _fv and _bv can share macros!
 */