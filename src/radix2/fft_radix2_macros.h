//==============================================================================
// fft_radix2_macros.h - Shared Macros for Radix-2 Butterflies
//==============================================================================
//
// USAGE:
//   #include "fft_radix2_macros.h" in both fft_radix2_fv.c and fft_radix2_bv.c
//
// BENEFITS:
//   - Single source of truth for butterfly patterns
//   - Easy to add optimizations (just update macros)
//   - Consistent across forward/inverse
//

#ifndef FFT_RADIX2_MACROS_H
#define FFT_RADIX2_MACROS_H

#include "simd_math.h"

//==============================================================================
// BUTTERFLY ARITHMETIC - Core Pattern
//==============================================================================

/**
 * @brief Radix-2 butterfly: y0 = even + tw_odd, y1 = even - tw_odd
 * 
 * The butterfly operation is IDENTICAL for forward and inverse FFTs.
 * Only the twiddles differ (precomputed with opposite signs).
 */

#ifdef __AVX512F__
#define RADIX2_BUTTERFLY_AVX512(even, odd, twiddle, y0_out, y1_out) \
    do { \
        __m512d tw_odd = cmul_avx512_aos(odd, twiddle); \
        y0_out = _mm512_add_pd(even, tw_odd); \
        y1_out = _mm512_sub_pd(even, tw_odd); \
    } while (0)
#endif

#ifdef __AVX2__
#define RADIX2_BUTTERFLY_AVX2(even, odd, twiddle, y0_out, y1_out) \
    do { \
        __m256d tw_odd = cmul_avx2_aos(odd, twiddle); \
        y0_out = _mm256_add_pd(even, tw_odd); \
        y1_out = _mm256_sub_pd(even, tw_odd); \
    } while (0)
#endif

#define RADIX2_BUTTERFLY_SSE2(even, odd, twiddle, y0_out, y1_out) \
    do { \
        __m128d tw_odd = cmul_sse2_aos(odd, twiddle); \
        y0_out = _mm_add_pd(even, tw_odd); \
        y1_out = _mm_sub_pd(even, tw_odd); \
    } while (0)

//==============================================================================
// DATA MOVEMENT - Load/Store Patterns
//==============================================================================

/**
 * @brief Load 8 butterflies worth of data (AVX2 path)
 * 
 * Loads:
 * - 8 even inputs (sub_outputs[k+0..7])
 * - 8 odd inputs (sub_outputs[k+half+0..7])
 * - 8 twiddles (stage_tw[k+0..7])
 * 
 * Into 12 AVX2 registers (e0-e3, o0-o3, w0-w3)
 */
#ifdef __AVX2__
#define LOAD_STAGE_INPUTS_AVX2(k, sub_outputs, stage_tw, half, \
                                e0, e1, e2, e3, o0, o1, o2, o3, w0, w1, w2, w3) \
    do { \
        e0 = load2_aos(&sub_outputs[(k)+0], &sub_outputs[(k)+1]); \
        e1 = load2_aos(&sub_outputs[(k)+2], &sub_outputs[(k)+3]); \
        e2 = load2_aos(&sub_outputs[(k)+4], &sub_outputs[(k)+5]); \
        e3 = load2_aos(&sub_outputs[(k)+6], &sub_outputs[(k)+7]); \
        \
        o0 = load2_aos(&sub_outputs[(k)+(half)], &sub_outputs[(k)+(half)+1]); \
        o1 = load2_aos(&sub_outputs[(k)+(half)+2], &sub_outputs[(k)+(half)+3]); \
        o2 = load2_aos(&sub_outputs[(k)+(half)+4], &sub_outputs[(k)+(half)+5]); \
        o3 = load2_aos(&sub_outputs[(k)+(half)+6], &sub_outputs[(k)+(half)+7]); \
        \
        w0 = load2_aos(&stage_tw[(k)+0], &stage_tw[(k)+1]); \
        w1 = load2_aos(&stage_tw[(k)+2], &stage_tw[(k)+3]); \
        w2 = load2_aos(&stage_tw[(k)+4], &stage_tw[(k)+5]); \
        w3 = load2_aos(&stage_tw[(k)+6], &stage_tw[(k)+7]); \
    } while (0)
#endif

/**
 * @brief Store 8 butterfly outputs (AVX2 path)
 */
#ifdef __AVX2__
#define STORE_BUTTERFLY_OUTPUTS_AVX2(k, output_buffer, half, \
                                     x00, x01, x02, x03, x10, x11, x12, x13) \
    do { \
        STOREU_PD(&output_buffer[(k)+0].re, x00); \
        STOREU_PD(&output_buffer[(k)+2].re, x01); \
        STOREU_PD(&output_buffer[(k)+4].re, x02); \
        STOREU_PD(&output_buffer[(k)+6].re, x03); \
        \
        STOREU_PD(&output_buffer[(k)+(half)].re, x10); \
        STOREU_PD(&output_buffer[(k)+(half)+2].re, x11); \
        STOREU_PD(&output_buffer[(k)+(half)+4].re, x12); \
        STOREU_PD(&output_buffer[(k)+(half)+6].re, x13); \
    } while (0)
#endif

//==============================================================================
// PREFETCHING - Cache Optimization
//==============================================================================

#ifdef __AVX512F__
/**
 * @brief Prefetch next iteration's data into L1 cache (AVX-512 - 16 butterflies)
 * 
 * Prefetches twice as much data as AVX2 version
 */
#define PREFETCH_NEXT_AVX512(k, distance, sub_outputs, stage_tw, half) \
    do { \
        if ((k) + (distance) < range1_end) { \
            _mm_prefetch((const char *)&sub_outputs[(k)+(distance)], _MM_HINT_T0); \
            _mm_prefetch((const char *)&sub_outputs[(k)+(distance)+8], _MM_HINT_T0); \
            _mm_prefetch((const char *)&sub_outputs[(k)+(distance)+(half)], _MM_HINT_T0); \
            _mm_prefetch((const char *)&sub_outputs[(k)+(distance)+(half)+8], _MM_HINT_T0); \
            _mm_prefetch((const char *)&stage_tw[(k)+(distance)], _MM_HINT_T0); \
            _mm_prefetch((const char *)&stage_tw[(k)+(distance)+8], _MM_HINT_T0); \
        } \
    } while (0)
#endif

/**
 * @brief Prefetch next iteration's data into L1 cache
 * 
 * Tuned for typical cache line size (64 bytes) and L1 latency (~4 cycles)
 */
#define PREFETCH_NEXT_AVX2(k, distance, sub_outputs, stage_tw, half) \
    do { \
        if ((k) + (distance) < range1_end) { \
            _mm_prefetch((const char *)&sub_outputs[(k)+(distance)], _MM_HINT_T0); \
            _mm_prefetch((const char *)&sub_outputs[(k)+(distance)+(half)], _MM_HINT_T0); \
            _mm_prefetch((const char *)&stage_tw[(k)+(distance)], _MM_HINT_T0); \
        } \
    } while (0)

//==============================================================================
// AVX-512 PATTERNS (16x butterflies)
//==============================================================================

#ifdef __AVX512F__
/**
 * @brief Load 16 butterflies worth of data (AVX-512 path)
 */
#define LOAD_STAGE_INPUTS_AVX512(k, sub_outputs, stage_tw, half, \
                                  e0, e1, e2, e3, o0, o1, o2, o3, w0, w1, w2, w3) \
    do { \
        e0 = load4_aos(&sub_outputs[(k)+0]); \
        e1 = load4_aos(&sub_outputs[(k)+4]); \
        e2 = load4_aos(&sub_outputs[(k)+8]); \
        e3 = load4_aos(&sub_outputs[(k)+12]); \
        \
        o0 = load4_aos(&sub_outputs[(k)+(half)]); \
        o1 = load4_aos(&sub_outputs[(k)+(half)+4]); \
        o2 = load4_aos(&sub_outputs[(k)+(half)+8]); \
        o3 = load4_aos(&sub_outputs[(k)+(half)+12]); \
        \
        w0 = load4_aos(&stage_tw[(k)+0]); \
        w1 = load4_aos(&stage_tw[(k)+4]); \
        w2 = load4_aos(&stage_tw[(k)+8]); \
        w3 = load4_aos(&stage_tw[(k)+12]); \
    } while (0)

/**
 * @brief Store 16 butterfly outputs (AVX-512 path)
 */
#define STORE_BUTTERFLY_OUTPUTS_AVX512(k, output_buffer, half, \
                                       x00, x01, x02, x03, x10, x11, x12, x13) \
    do { \
        STOREU_PD512(&output_buffer[(k)+0].re, x00); \
        STOREU_PD512(&output_buffer[(k)+4].re, x01); \
        STOREU_PD512(&output_buffer[(k)+8].re, x02); \
        STOREU_PD512(&output_buffer[(k)+12].re, x03); \
        \
        STOREU_PD512(&output_buffer[(k)+(half)].re, x10); \
        STOREU_PD512(&output_buffer[(k)+(half)+4].re, x11); \
        STOREU_PD512(&output_buffer[(k)+(half)+8].re, x12); \
        STOREU_PD512(&output_buffer[(k)+(half)+12].re, x13); \
    } while (0)
#endif

//==============================================================================
// SPECIAL CASES - k=0 and k=N/4
//==============================================================================

/**
 * @brief k=0 butterfly (W^0 = 1, no twiddle multiplication)
 * 
 * IDENTICAL for forward and inverse
 */
#define RADIX2_BUTTERFLY_K0(sub_outputs, output_buffer, half) \
    do { \
        fft_data even_0 = sub_outputs[0]; \
        fft_data odd_0 = sub_outputs[half]; \
        output_buffer[0].re = even_0.re + odd_0.re; \
        output_buffer[0].im = even_0.im + odd_0.im; \
        output_buffer[half].re = even_0.re - odd_0.re; \
        output_buffer[half].im = even_0.im - odd_0.im; \
    } while (0)

/**
 * @brief k=N/4 butterfly for FORWARD FFT (W^(N/4) = -i)
 * 
 * Multiply by -i: (a + bi) * (-i) = b - ai
 */
#define RADIX2_BUTTERFLY_K_QUARTER_FORWARD(sub_outputs, output_buffer, half, k_quarter) \
    do { \
        fft_data even_q = sub_outputs[k_quarter]; \
        fft_data odd_q = sub_outputs[half + k_quarter]; \
        \
        double rotated_re = odd_q.im;   /* b */ \
        double rotated_im = -odd_q.re;  /* -a */ \
        \
        output_buffer[k_quarter].re = even_q.re + rotated_re; \
        output_buffer[k_quarter].im = even_q.im + rotated_im; \
        output_buffer[half + k_quarter].re = even_q.re - rotated_re; \
        output_buffer[half + k_quarter].im = even_q.im - rotated_im; \
    } while (0)

/**
 * @brief k=N/4 butterfly for INVERSE FFT (W^(N/4) = +i)
 * 
 * Multiply by +i: (a + bi) * i = -b + ai
 */
#define RADIX2_BUTTERFLY_K_QUARTER_INVERSE(sub_outputs, output_buffer, half, k_quarter) \
    do { \
        fft_data even_q = sub_outputs[k_quarter]; \
        fft_data odd_q = sub_outputs[half + k_quarter]; \
        \
        double rotated_re = -odd_q.im;  /* -b */ \
        double rotated_im = odd_q.re;   /* a */ \
        \
        output_buffer[k_quarter].re = even_q.re + rotated_re; \
        output_buffer[k_quarter].im = even_q.im + rotated_im; \
        output_buffer[half + k_quarter].re = even_q.re - rotated_re; \
        output_buffer[half + k_quarter].im = even_q.im - rotated_im; \
    } while (0)

//==============================================================================
// COMPLETE BUTTERFLY PIPELINE MACROS (NEW)
//==============================================================================

#ifdef __AVX512F__
/**
 * @brief Complete 16-butterfly pipeline: load -> compute -> store
 * 
 * Processes 16 butterflies in one shot with optimal register reuse
 */
#define RADIX2_PIPELINE_16_AVX512(k, sub_outputs, stage_tw, output_buffer, half) \
    do { \
        __m512d e0, e1, e2, e3, o0, o1, o2, o3, w0, w1, w2, w3; \
        LOAD_STAGE_INPUTS_AVX512(k, sub_outputs, stage_tw, half, \
                                  e0, e1, e2, e3, o0, o1, o2, o3, w0, w1, w2, w3); \
        \
        __m512d x00, x10, x01, x11, x02, x12, x03, x13; \
        RADIX2_BUTTERFLY_AVX512(e0, o0, w0, x00, x10); \
        RADIX2_BUTTERFLY_AVX512(e1, o1, w1, x01, x11); \
        RADIX2_BUTTERFLY_AVX512(e2, o2, w2, x02, x12); \
        RADIX2_BUTTERFLY_AVX512(e3, o3, w3, x03, x13); \
        \
        STORE_BUTTERFLY_OUTPUTS_AVX512(k, output_buffer, half, \
                                       x00, x01, x02, x03, x10, x11, x12, x13); \
    } while (0)
#endif

#ifdef __AVX2__
/**
 * @brief Complete 8-butterfly pipeline: load -> compute -> store
 * 
 * Processes 8 butterflies in one shot with optimal register reuse
 */
#define RADIX2_PIPELINE_8_AVX2(k, sub_outputs, stage_tw, output_buffer, half) \
    do { \
        __m256d e0, e1, e2, e3, o0, o1, o2, o3, w0, w1, w2, w3; \
        LOAD_STAGE_INPUTS_AVX2(k, sub_outputs, stage_tw, half, \
                              e0, e1, e2, e3, o0, o1, o2, o3, w0, w1, w2, w3); \
        \
        __m256d x00, x10, x01, x11, x02, x12, x03, x13; \
        RADIX2_BUTTERFLY_AVX2(e0, o0, w0, x00, x10); \
        RADIX2_BUTTERFLY_AVX2(e1, o1, w1, x01, x11); \
        RADIX2_BUTTERFLY_AVX2(e2, o2, w2, x02, x12); \
        RADIX2_BUTTERFLY_AVX2(e3, o3, w3, x03, x13); \
        \
        STORE_BUTTERFLY_OUTPUTS_AVX2(k, output_buffer, half, \
                                     x00, x01, x02, x03, x10, x11, x12, x13); \
    } while (0)
#endif

/**
 * @brief Complete 2-butterfly pipeline: load -> compute -> store (AVX2)
 * 
 * Useful for cleanup loops
 */
#ifdef __AVX2__
#define RADIX2_PIPELINE_2_AVX2(k, sub_outputs, stage_tw, output_buffer, half) \
    do { \
        __m256d even = load2_aos(&sub_outputs[k], &sub_outputs[(k)+1]); \
        __m256d odd = load2_aos(&sub_outputs[(k)+(half)], &sub_outputs[(k)+(half)+1]); \
        __m256d w = load2_aos(&stage_tw[k], &stage_tw[(k)+1]); \
        \
        __m256d x0, x1; \
        RADIX2_BUTTERFLY_AVX2(even, odd, w, x0, x1); \
        \
        STOREU_PD(&output_buffer[k].re, x0); \
        STOREU_PD(&output_buffer[(k)+(half)].re, x1); \
    } while (0)
#endif

/**
 * @brief Complete 1-butterfly pipeline: load -> compute -> store (SSE2)
 * 
 * Scalar tail
 */
#define RADIX2_PIPELINE_1_SSE2(k, sub_outputs, stage_tw, output_buffer, half) \
    do { \
        __m128d even = LOADU_SSE2(&sub_outputs[k].re); \
        __m128d odd = LOADU_SSE2(&sub_outputs[(k)+(half)].re); \
        __m128d w = LOADU_SSE2(&stage_tw[k].re); \
        \
        __m128d x0, x1; \
        RADIX2_BUTTERFLY_SSE2(even, odd, w, x0, x1); \
        \
        STOREU_SSE2(&output_buffer[k].re, x0); \
        STOREU_SSE2(&output_buffer[(k)+(half)].re, x1); \
    } while (0)

//==============================================================================
// USAGE NOTES
//==============================================================================

/**
 * These macros enable:
 * 1. Code reuse between forward/inverse butterflies
 * 2. Easy optimization updates (change macro, both functions benefit)
 * 3. Consistent style across radices (radix-4, radix-8 can use similar patterns)
 * 
 * Example: Adding prefetch for L2 cache
 * 
 * Just add to PREFETCH_NEXT_AVX2:
 *   if ((k) + (distance*2) < range1_end) {
 *       _mm_prefetch(..., _MM_HINT_T1);  // L2 prefetch
 *   }
 * 
 * Both fft_radix2_fv and fft_radix2_bv get the optimization automatically!
 */

#endif // FFT_RADIX2_MACROS_H