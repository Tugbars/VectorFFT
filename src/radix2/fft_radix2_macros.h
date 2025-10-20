//==============================================================================
// fft_radix2_macros.h - Shared Macros + Inline Helpers for Radix-2 Butterflies
//==============================================================================
//
// DESIGN:
// - Small helpers: static __always_inline (type safety)
// - Large SIMD blocks: Macros (performance)
// - Direction stays in function names (_fv vs _bv)
//
// OPTIMIZATIONS:
// - Interleaved load/compute/store to reduce register pressure
// - Split loops to eliminate k_quarter branches
// - 4x SSE2 pipeline for efficient tail processing
// - Portable alignment (64-byte for AVX-512)
//

#ifndef FFT_RADIX2_MACROS_H
#define FFT_RADIX2_MACROS_H

#include "fft_radix2.h"
#include "simd_math.h"

//==============================================================================
// BUTTERFLY ARITHMETIC - Core Pattern (MACROS - used in hot loops)
//==============================================================================

#ifdef __AVX512F__
#define RADIX2_BUTTERFLY_AVX512(even, odd, twiddle, y0_out, y1_out) \
    do                                                              \
    {                                                               \
        __m512d tw_odd = cmul_avx512_aos(odd, twiddle);             \
        y0_out = _mm512_add_pd(even, tw_odd);                       \
        y1_out = _mm512_sub_pd(even, tw_odd);                       \
    } while (0)
#endif

#ifdef __AVX2__
#define RADIX2_BUTTERFLY_AVX2(even, odd, twiddle, y0_out, y1_out) \
    do                                                            \
    {                                                             \
        __m256d tw_odd = cmul_avx2_aos(odd, twiddle);             \
        y0_out = _mm256_add_pd(even, tw_odd);                     \
        y1_out = _mm256_sub_pd(even, tw_odd);                     \
    } while (0)
#endif

#define RADIX2_BUTTERFLY_SSE2(even, odd, twiddle, y0_out, y1_out) \
    do                                                            \
    {                                                             \
        __m128d tw_odd = cmul_sse2_aos(odd, twiddle);             \
        y0_out = _mm_add_pd(even, tw_odd);                        \
        y1_out = _mm_sub_pd(even, tw_odd);                        \
    } while (0)

//==============================================================================
// PREFETCHING - Cache Optimization
//==============================================================================

#ifdef __AVX512F__
#define PREFETCH_NEXT_AVX512(k, distance, sub_outputs, stage_tw, half, end)                       \
    do                                                                                            \
    {                                                                                             \
        if ((k) + (distance) < (end))                                                             \
        {                                                                                         \
            /* Prefetch even values (2 cache lines for 16 butterflies) */                         \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance)], _MM_HINT_T0);              \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + 8], _MM_HINT_T0);          \
            /* Prefetch odd values (2 cache lines) */                                             \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + (half)], _MM_HINT_T0);     \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + (half) + 8], _MM_HINT_T0); \
            /* Prefetch twiddle factors (2 cache lines) */                                        \
            _mm_prefetch((const char *)&stage_tw[(k) + (distance)], _MM_HINT_T0);                 \
            _mm_prefetch((const char *)&stage_tw[(k) + (distance) + 8], _MM_HINT_T0);             \
        }                                                                                         \
    } while (0)
#endif

#define PREFETCH_NEXT_AVX2(k, distance, sub_outputs, stage_tw, half, end)                     \
    do                                                                                        \
    {                                                                                         \
        if ((k) + (distance) < (end))                                                         \
        {                                                                                     \
            /* Prefetch even values (1 cache line for 8 butterflies) */                       \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance)], _MM_HINT_T0);          \
            /* Prefetch odd values (1 cache line) */                                          \
            _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + (half)], _MM_HINT_T0); \
            /* Prefetch twiddle factors (1 cache line) */                                     \
            _mm_prefetch((const char *)&stage_tw[(k) + (distance)], _MM_HINT_T0);             \
        }                                                                                     \
    } while (0)

//==============================================================================
// SPECIAL CASES - Small Inline Helpers (TYPE SAFE)
//==============================================================================

/**
 * @brief k=0 butterfly (W^0 = 1, no twiddle)
 * IDENTICAL for forward and inverse
 */
static __always_inline void radix2_butterfly_k0(
    fft_data *restrict output_buffer,
    const fft_data *restrict sub_outputs,
    int half)
{
    fft_data even_0 = sub_outputs[0];
    fft_data odd_0 = sub_outputs[half];

    output_buffer[0].re = even_0.re + odd_0.re;
    output_buffer[0].im = even_0.im + odd_0.im;
    output_buffer[half].re = even_0.re - odd_0.re;
    output_buffer[half].im = even_0.im - odd_0.im;
}

/**
 * @brief k=N/4 butterfly - parameterized by direction
 *
 * @param is_inverse: false = forward (-i rotation), true = inverse (+i rotation)
 *
 * Forward:  W^(N/4) = -i → (a+bi)*(-i) = b-ai
 * Inverse:  W^(N/4) = +i → (a+bi)*(i)  = -b+ai
 */
static __always_inline void radix2_butterfly_k_quarter(
    fft_data *restrict output_buffer,
    const fft_data *restrict sub_outputs,
    int half,
    int k_quarter,
    bool is_inverse)
{
    fft_data even_q = sub_outputs[k_quarter];
    fft_data odd_q = sub_outputs[half + k_quarter];

    double rotated_re, rotated_im;

    if (is_inverse)
    {
        // Inverse: multiply by +i
        rotated_re = -odd_q.im; // -b
        rotated_im = odd_q.re;  // a
    }
    else
    {
        // Forward: multiply by -i
        rotated_re = odd_q.im;  // b
        rotated_im = -odd_q.re; // -a
    }

    output_buffer[k_quarter].re = even_q.re + rotated_re;
    output_buffer[k_quarter].im = even_q.im + rotated_im;
    output_buffer[half + k_quarter].re = even_q.re - rotated_re;
    output_buffer[half + k_quarter].im = even_q.im - rotated_im;
}

//==============================================================================
// COMPLETE BUTTERFLY PIPELINES (MACROS - large hot paths)
// OPTIMIZED: Interleaved load/compute/store to reduce register pressure
//==============================================================================

#ifdef __AVX512F__
/**
 * @brief Complete 16-butterfly pipeline with interleaved load/compute/store
 * Processes 4 batches of 4 butterflies each to minimize live register count
 */
#define RADIX2_PIPELINE_16_AVX512(k, sub_outputs, stage_tw, output_buffer, half, end) \
    do                                                                                \
    {                                                                                 \
        PREFETCH_NEXT_AVX512(k, 32, sub_outputs, stage_tw, half, end);                \
                                                                                      \
        /* Batch 0: butterflies 0-3 */                                                \
        __m512d e0 = load4_aos(&sub_outputs[(k) + 0]);                                \
        __m512d o0 = load4_aos(&sub_outputs[(k) + (half)]);                           \
        __m512d w0 = load4_aos(&stage_tw[(k) + 0]);                                   \
        __m512d x00, x10;                                                             \
        RADIX2_BUTTERFLY_AVX512(e0, o0, w0, x00, x10);                                \
        STOREU_PD512(&output_buffer[(k) + 0].re, x00);                                \
        STOREU_PD512(&output_buffer[(k) + (half)].re, x10);                           \
                                                                                      \
        /* Batch 1: butterflies 4-7 */                                                \
        __m512d e1 = load4_aos(&sub_outputs[(k) + 4]);                                \
        __m512d o1 = load4_aos(&sub_outputs[(k) + (half) + 4]);                       \
        __m512d w1 = load4_aos(&stage_tw[(k) + 4]);                                   \
        __m512d x01, x11;                                                             \
        RADIX2_BUTTERFLY_AVX512(e1, o1, w1, x01, x11);                                \
        STOREU_PD512(&output_buffer[(k) + 4].re, x01);                                \
        STOREU_PD512(&output_buffer[(k) + (half) + 4].re, x11);                       \
                                                                                      \
        /* Batch 2: butterflies 8-11 */                                               \
        __m512d e2 = load4_aos(&sub_outputs[(k) + 8]);                                \
        __m512d o2 = load4_aos(&sub_outputs[(k) + (half) + 8]);                       \
        __m512d w2 = load4_aos(&stage_tw[(k) + 8]);                                   \
        __m512d x02, x12;                                                             \
        RADIX2_BUTTERFLY_AVX512(e2, o2, w2, x02, x12);                                \
        STOREU_PD512(&output_buffer[(k) + 8].re, x02);                                \
        STOREU_PD512(&output_buffer[(k) + (half) + 8].re, x12);                       \
                                                                                      \
        /* Batch 3: butterflies 12-15 */                                              \
        __m512d e3 = load4_aos(&sub_outputs[(k) + 12]);                               \
        __m512d o3 = load4_aos(&sub_outputs[(k) + (half) + 12]);                      \
        __m512d w3 = load4_aos(&stage_tw[(k) + 12]);                                  \
        __m512d x03, x13;                                                             \
        RADIX2_BUTTERFLY_AVX512(e3, o3, w3, x03, x13);                                \
        STOREU_PD512(&output_buffer[(k) + 12].re, x03);                               \
        STOREU_PD512(&output_buffer[(k) + (half) + 12].re, x13);                      \
    } while (0)
#endif

#ifdef __AVX2__
/**
 * @brief Complete 8-butterfly pipeline with interleaved load/compute/store
 * Processes 4 batches of 2 butterflies each
 */
#define RADIX2_PIPELINE_8_AVX2(k, sub_outputs, stage_tw, output_buffer, half, end)              \
    do                                                                                          \
    {                                                                                           \
        PREFETCH_NEXT_AVX2(k, 16, sub_outputs, stage_tw, half, end);                            \
                                                                                                \
        /* Batch 0: butterflies 0-1 */                                                          \
        __m256d e0 = load2_aos(&sub_outputs[(k) + 0], &sub_outputs[(k) + 1]);                   \
        __m256d o0 = load2_aos(&sub_outputs[(k) + (half)], &sub_outputs[(k) + (half) + 1]);     \
        __m256d w0 = load2_aos(&stage_tw[(k) + 0], &stage_tw[(k) + 1]);                         \
        __m256d x00, x10;                                                                       \
        RADIX2_BUTTERFLY_AVX2(e0, o0, w0, x00, x10);                                            \
        STOREU_PD(&output_buffer[(k) + 0].re, x00);                                             \
        STOREU_PD(&output_buffer[(k) + (half)].re, x10);                                        \
                                                                                                \
        /* Batch 1: butterflies 2-3 */                                                          \
        __m256d e1 = load2_aos(&sub_outputs[(k) + 2], &sub_outputs[(k) + 3]);                   \
        __m256d o1 = load2_aos(&sub_outputs[(k) + (half) + 2], &sub_outputs[(k) + (half) + 3]); \
        __m256d w1 = load2_aos(&stage_tw[(k) + 2], &stage_tw[(k) + 3]);                         \
        __m256d x01, x11;                                                                       \
        RADIX2_BUTTERFLY_AVX2(e1, o1, w1, x01, x11);                                            \
        STOREU_PD(&output_buffer[(k) + 2].re, x01);                                             \
        STOREU_PD(&output_buffer[(k) + (half) + 2].re, x11);                                    \
                                                                                                \
        /* Batch 2: butterflies 4-5 */                                                          \
        __m256d e2 = load2_aos(&sub_outputs[(k) + 4], &sub_outputs[(k) + 5]);                   \
        __m256d o2 = load2_aos(&sub_outputs[(k) + (half) + 4], &sub_outputs[(k) + (half) + 5]); \
        __m256d w2 = load2_aos(&stage_tw[(k) + 4], &stage_tw[(k) + 5]);                         \
        __m256d x02, x12;                                                                       \
        RADIX2_BUTTERFLY_AVX2(e2, o2, w2, x02, x12);                                            \
        STOREU_PD(&output_buffer[(k) + 4].re, x02);                                             \
        STOREU_PD(&output_buffer[(k) + (half) + 4].re, x12);                                    \
                                                                                                \
        /* Batch 3: butterflies 6-7 */                                                          \
        __m256d e3 = load2_aos(&sub_outputs[(k) + 6], &sub_outputs[(k) + 7]);                   \
        __m256d o3 = load2_aos(&sub_outputs[(k) + (half) + 6], &sub_outputs[(k) + (half) + 7]); \
        __m256d w3 = load2_aos(&stage_tw[(k) + 6], &stage_tw[(k) + 7]);                         \
        __m256d x03, x13;                                                                       \
        RADIX2_BUTTERFLY_AVX2(e3, o3, w3, x03, x13);                                            \
        STOREU_PD(&output_buffer[(k) + 6].re, x03);                                             \
        STOREU_PD(&output_buffer[(k) + (half) + 6].re, x13);                                    \
    } while (0)

/**
 * @brief Complete 2-butterfly pipeline (AVX2)
 */
#define RADIX2_PIPELINE_2_AVX2(k, sub_outputs, stage_tw, output_buffer, half)                \
    do                                                                                       \
    {                                                                                        \
        __m256d even = load2_aos(&sub_outputs[k], &sub_outputs[(k) + 1]);                    \
        __m256d odd = load2_aos(&sub_outputs[(k) + (half)], &sub_outputs[(k) + (half) + 1]); \
        __m256d w = load2_aos(&stage_tw[k], &stage_tw[(k) + 1]);                             \
                                                                                             \
        __m256d x0, x1;                                                                      \
        RADIX2_BUTTERFLY_AVX2(even, odd, w, x0, x1);                                         \
                                                                                             \
        STOREU_PD(&output_buffer[k].re, x0);                                                 \
        STOREU_PD(&output_buffer[(k) + (half)].re, x1);                                      \
    } while (0)
#endif

/**
 * @brief Complete 4-butterfly pipeline (SSE2) - for efficient tail processing
 * Processes 4 butterflies at once instead of 1 at a time
 */
#define RADIX2_PIPELINE_4_SSE2(k, sub_outputs, stage_tw, output_buffer, half) \
    do                                                                        \
    {                                                                         \
        /* Butterfly 0 */                                                     \
        __m128d e0 = LOADU_SSE2(&sub_outputs[(k) + 0].re);                    \
        __m128d o0 = LOADU_SSE2(&sub_outputs[(k) + (half) + 0].re);           \
        __m128d w0 = LOADU_SSE2(&stage_tw[(k) + 0].re);                       \
        __m128d x00, x10;                                                     \
        RADIX2_BUTTERFLY_SSE2(e0, o0, w0, x00, x10);                          \
        STOREU_SSE2(&output_buffer[(k) + 0].re, x00);                         \
        STOREU_SSE2(&output_buffer[(k) + (half) + 0].re, x10);                \
                                                                              \
        /* Butterfly 1 */                                                     \
        __m128d e1 = LOADU_SSE2(&sub_outputs[(k) + 1].re);                    \
        __m128d o1 = LOADU_SSE2(&sub_outputs[(k) + (half) + 1].re);           \
        __m128d w1 = LOADU_SSE2(&stage_tw[(k) + 1].re);                       \
        __m128d x01, x11;                                                     \
        RADIX2_BUTTERFLY_SSE2(e1, o1, w1, x01, x11);                          \
        STOREU_SSE2(&output_buffer[(k) + 1].re, x01);                         \
        STOREU_SSE2(&output_buffer[(k) + (half) + 1].re, x11);                \
                                                                              \
        /* Butterfly 2 */                                                     \
        __m128d e2 = LOADU_SSE2(&sub_outputs[(k) + 2].re);                    \
        __m128d o2 = LOADU_SSE2(&sub_outputs[(k) + (half) + 2].re);           \
        __m128d w2 = LOADU_SSE2(&stage_tw[(k) + 2].re);                       \
        __m128d x02, x12;                                                     \
        RADIX2_BUTTERFLY_SSE2(e2, o2, w2, x02, x12);                          \
        STOREU_SSE2(&output_buffer[(k) + 2].re, x02);                         \
        STOREU_SSE2(&output_buffer[(k) + (half) + 2].re, x12);                \
                                                                              \
        /* Butterfly 3 */                                                     \
        __m128d e3 = LOADU_SSE2(&sub_outputs[(k) + 3].re);                    \
        __m128d o3 = LOADU_SSE2(&sub_outputs[(k) + (half) + 3].re);           \
        __m128d w3 = LOADU_SSE2(&stage_tw[(k) + 3].re);                       \
        __m128d x03, x13;                                                     \
        RADIX2_BUTTERFLY_SSE2(e3, o3, w3, x03, x13);                          \
        STOREU_SSE2(&output_buffer[(k) + 3].re, x03);                         \
        STOREU_SSE2(&output_buffer[(k) + (half) + 3].re, x13);                \
    } while (0)

/**
 * @brief Complete 1-butterfly pipeline (SSE2)
 */
#define RADIX2_PIPELINE_1_SSE2(k, sub_outputs, stage_tw, output_buffer, half) \
    do                                                                        \
    {                                                                         \
        __m128d even = LOADU_SSE2(&sub_outputs[k].re);                        \
        __m128d odd = LOADU_SSE2(&sub_outputs[(k) + (half)].re);              \
        __m128d w = LOADU_SSE2(&stage_tw[k].re);                              \
                                                                              \
        __m128d x0, x1;                                                       \
        RADIX2_BUTTERFLY_SSE2(even, odd, w, x0, x1);                          \
                                                                              \
        STOREU_SSE2(&output_buffer[k].re, x0);                                \
        STOREU_SSE2(&output_buffer[(k) + (half)].re, x1);                     \
    } while (0)

//==============================================================================
// UNIFIED LOOP HELPER (INLINE - complex control flow)
// OPTIMIZED: Split loops to eliminate k_quarter branch from hot path
//==============================================================================

/**
 * @brief Process main loop with k=N/4 handled separately
 *
 * Split into three segments:
 * 1. [1, k_quarter) - fully vectorized, no branches
 * 2. k_quarter - already handled by caller
 * 3. (k_quarter, half) - fully vectorized, no branches
 *
 * This eliminates ALL conditional logic from the hot SIMD loops
 */
static __always_inline void radix2_process_main_loop(
    fft_data *restrict output_buffer,
    const fft_data *restrict sub_outputs,
    const fft_data *restrict stage_tw,
    int half,
    int k_quarter)
{
    int k = 1;
    int end_first = k_quarter ? k_quarter : half;

    //==========================================================================
    // SEGMENT 1: Process [1, k_quarter) with NO branches
    //==========================================================================

#ifdef __AVX512F__
    // AVX-512: 16x butterflies
    while (k + 15 < end_first)
    {
        RADIX2_PIPELINE_16_AVX512(k, sub_outputs, stage_tw, output_buffer, half, end_first);
        k += 16;
    }
#endif

#ifdef __AVX2__
    // AVX2: 8x butterflies
    while (k + 7 < end_first)
    {
        RADIX2_PIPELINE_8_AVX2(k, sub_outputs, stage_tw, output_buffer, half, end_first);
        k += 8;
    }

    // AVX2: 2x butterflies
    while (k + 1 < end_first)
    {
        RADIX2_PIPELINE_2_AVX2(k, sub_outputs, stage_tw, output_buffer, half);
        k += 2;
    }
#endif

    // SSE2: 4x butterflies (efficient tail)
    while (k + 3 < end_first)
    {
        RADIX2_PIPELINE_4_SSE2(k, sub_outputs, stage_tw, output_buffer, half);
        k += 4;
    }

    // SSE2: 1x butterfly final tail
    while (k < end_first)
    {
        RADIX2_PIPELINE_1_SSE2(k, sub_outputs, stage_tw, output_buffer, half);
        k++;
    }

    //==========================================================================
    // SEGMENT 2: Skip k_quarter (already handled by caller)
    //==========================================================================
    if (k_quarter)
    {
        k = k_quarter + 1;
    }

    //==========================================================================
    // SEGMENT 3: Process (k_quarter, half) with NO branches
    //==========================================================================

#ifdef __AVX512F__
    // AVX-512: 16x butterflies
    while (k + 15 < half)
    {
        RADIX2_PIPELINE_16_AVX512(k, sub_outputs, stage_tw, output_buffer, half, half);
        k += 16;
    }
#endif

#ifdef __AVX2__
    // AVX2: 8x butterflies
    while (k + 7 < half)
    {
        RADIX2_PIPELINE_8_AVX2(k, sub_outputs, stage_tw, output_buffer, half, half);
        k += 8;
    }

    // AVX2: 2x butterflies
    while (k + 1 < half)
    {
        RADIX2_PIPELINE_2_AVX2(k, sub_outputs, stage_tw, output_buffer, half);
        k += 2;
    }
#endif

    // SSE2: 4x butterflies (efficient tail)
    while (k + 3 < half)
    {
        RADIX2_PIPELINE_4_SSE2(k, sub_outputs, stage_tw, output_buffer, half);
        k += 4;
    }

    // SSE2: 1x butterfly final tail
    while (k < half)
    {
        RADIX2_PIPELINE_1_SSE2(k, sub_outputs, stage_tw, output_buffer, half);
        k++;
    }
}

#endif // FFT_RADIX2_MACROS_H