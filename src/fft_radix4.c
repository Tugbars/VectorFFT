#include "fft_radix4.h"
#include "simd_math.h"

// Compiler-specific alignment hints
#if defined(_MSC_VER)
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#elif defined(__GNUC__) || defined(__clang__)
#define ASSUME_ALIGNED(ptr, alignment) __builtin_assume_aligned(ptr, alignment)
#else
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#endif

// OpenMP detection
#ifdef HAVE_OPENMP
#include <omp.h>
#define USE_OPENMP 1
#else
#define USE_OPENMP 0
#endif

#undef USE_OPENMP      // ← ADD THIS LINE TO FORCE DISABLE
#define USE_OPENMP 0   // ← AND THIS

// Configuration constants
#define RADIX4_PREFETCH_DISTANCE 128
//#define RADIX4_PARALLEL_THRESHOLD 512 // Lower threshold than radix-2 (more work per butterfly)
#define RADIX4_PARALLEL_THRESHOLD 100000  // Set very high to disable

/**
 * @brief Ultra-optimized radix-4 butterfly with advanced optimizations
 *
 * OPTIMIZATIONS:
 * 1. Software pipelining - overlaps load/compute/store
 * 2. Multi-threading - OpenMP parallelization
 * 3. Non-temporal stores - for large FFTs
 * 4. Alignment hints - compiler optimizations
 * 5. Multi-level prefetching
 */

void fft_radix4_butterfly(
    fft_data *output_buffer,
    fft_data *sub_outputs,
    const fft_data *stage_tw,
    int sub_len,
    int transform_sign)
{
    const int quarter = sub_len;
    int k = 0;

    // Alignment hints
    output_buffer = (fft_data *)ASSUME_ALIGNED(output_buffer, 64);
    sub_outputs = (fft_data *)ASSUME_ALIGNED(sub_outputs, 64);
    stage_tw = (const fft_data *)ASSUME_ALIGNED(stage_tw, 64);

    // Decide on parallelization
#if USE_OPENMP
    const int use_parallel = quarter >= RADIX4_PARALLEL_THRESHOLD;
#else
    const int use_parallel = 0;
    (void)use_parallel;
#endif

#ifdef HAS_AVX512
    //======================================================================
    // AVX-512: SOFTWARE PIPELINED + PARALLEL
    //======================================================================

    const __m512d rot_mask_512 = (transform_sign == 1)
                                     ? _mm512_castsi512_pd(_mm512_set_epi64(
                                           0x8000000000000000, 0x0000000000000000,
                                           0x8000000000000000, 0x0000000000000000,
                                           0x8000000000000000, 0x0000000000000000,
                                           0x8000000000000000, 0x0000000000000000))
                                     : _mm512_castsi512_pd(_mm512_set_epi64(
                                           0x0000000000000000, 0x8000000000000000,
                                           0x0000000000000000, 0x8000000000000000,
                                           0x0000000000000000, 0x8000000000000000,
                                           0x0000000000000000, 0x8000000000000000));

#define RADIX4_BUTTERFLY_AVX512(a, b2, c2, d2, y0, y1, y2, y3)    \
    {                                                             \
        __m512d sumBD = _mm512_add_pd(b2, d2);                    \
        __m512d difBD = _mm512_sub_pd(b2, d2);                    \
        __m512d a_pc = _mm512_add_pd(a, c2);                      \
        __m512d a_mc = _mm512_sub_pd(a, c2);                      \
        y0 = _mm512_add_pd(a_pc, sumBD);                          \
        y2 = _mm512_sub_pd(a_pc, sumBD);                          \
        __m512d difBD_swp = _mm512_permute_pd(difBD, 0b01010101); \
        __m512d rot = _mm512_xor_pd(difBD_swp, rot_mask_512);     \
        y1 = _mm512_sub_pd(a_mc, rot);                            \
        y3 = _mm512_add_pd(a_mc, rot);                            \
    }

    // Software pipelined main loop
    if (k + 31 < quarter)
    {
        // Prologue: preload first iteration
        __m512d next_a0 = load4_aos(&sub_outputs[k + 0]);
        __m512d next_a1 = load4_aos(&sub_outputs[k + 4]);
        __m512d next_a2 = load4_aos(&sub_outputs[k + 8]);
        __m512d next_a3 = load4_aos(&sub_outputs[k + 12]);

        __m512d next_b0 = load4_aos(&sub_outputs[k + 0 + quarter]);
        __m512d next_b1 = load4_aos(&sub_outputs[k + 4 + quarter]);
        __m512d next_b2 = load4_aos(&sub_outputs[k + 8 + quarter]);
        __m512d next_b3 = load4_aos(&sub_outputs[k + 12 + quarter]);

        __m512d next_c0 = load4_aos(&sub_outputs[k + 0 + 2 * quarter]);
        __m512d next_c1 = load4_aos(&sub_outputs[k + 4 + 2 * quarter]);
        __m512d next_c2 = load4_aos(&sub_outputs[k + 8 + 2 * quarter]);
        __m512d next_c3 = load4_aos(&sub_outputs[k + 12 + 2 * quarter]);

        __m512d next_d0 = load4_aos(&sub_outputs[k + 0 + 3 * quarter]);
        __m512d next_d1 = load4_aos(&sub_outputs[k + 4 + 3 * quarter]);
        __m512d next_d2 = load4_aos(&sub_outputs[k + 8 + 3 * quarter]);
        __m512d next_d3 = load4_aos(&sub_outputs[k + 12 + 3 * quarter]);

        __m512d next_w1_0 = load4_aos(&stage_tw[3 * (k + 0)]);
        __m512d next_w1_1 = load4_aos(&stage_tw[3 * (k + 4)]);
        __m512d next_w1_2 = load4_aos(&stage_tw[3 * (k + 8)]);
        __m512d next_w1_3 = load4_aos(&stage_tw[3 * (k + 12)]);

        __m512d next_w2_0 = load4_aos(&stage_tw[3 * (k + 0) + 1]);
        __m512d next_w2_1 = load4_aos(&stage_tw[3 * (k + 4) + 1]);
        __m512d next_w2_2 = load4_aos(&stage_tw[3 * (k + 8) + 1]);
        __m512d next_w2_3 = load4_aos(&stage_tw[3 * (k + 12) + 1]);

        __m512d next_w3_0 = load4_aos(&stage_tw[3 * (k + 0) + 2]);
        __m512d next_w3_1 = load4_aos(&stage_tw[3 * (k + 4) + 2]);
        __m512d next_w3_2 = load4_aos(&stage_tw[3 * (k + 8) + 2]);
        __m512d next_w3_3 = load4_aos(&stage_tw[3 * (k + 12) + 2]);

        // Pipelined loop
        for (; k + 31 < quarter; k += 16)
        {
            // Use preloaded values
            __m512d a0 = next_a0, a1 = next_a1, a2 = next_a2, a3 = next_a3;
            __m512d b0 = next_b0, b1 = next_b1, b2 = next_b2, b3 = next_b3;
            __m512d c0 = next_c0, c1 = next_c1, c2 = next_c2, c3 = next_c3;
            __m512d d0 = next_d0, d1 = next_d1, d2 = next_d2, d3 = next_d3;
            __m512d w1_0 = next_w1_0, w1_1 = next_w1_1, w1_2 = next_w1_2, w1_3 = next_w1_3;
            __m512d w2_0 = next_w2_0, w2_1 = next_w2_1, w2_2 = next_w2_2, w2_3 = next_w2_3;
            __m512d w3_0 = next_w3_0, w3_1 = next_w3_1, w3_2 = next_w3_2, w3_3 = next_w3_3;

            // Load next iteration while computing current
            if (k + 47 < quarter)
            {
                if (k + RADIX4_PREFETCH_DISTANCE < quarter)
                {
                    _mm_prefetch((const char *)&sub_outputs[k + RADIX4_PREFETCH_DISTANCE].re, _MM_HINT_T0);
                    _mm_prefetch((const char *)&stage_tw[3 * (k + RADIX4_PREFETCH_DISTANCE)].re, _MM_HINT_T0);
                }

                next_a0 = load4_aos(&sub_outputs[k + 16]);
                next_a1 = load4_aos(&sub_outputs[k + 20]);
                next_a2 = load4_aos(&sub_outputs[k + 24]);
                next_a3 = load4_aos(&sub_outputs[k + 28]);

                next_b0 = load4_aos(&sub_outputs[k + 16 + quarter]);
                next_b1 = load4_aos(&sub_outputs[k + 20 + quarter]);
                next_b2 = load4_aos(&sub_outputs[k + 24 + quarter]);
                next_b3 = load4_aos(&sub_outputs[k + 28 + quarter]);

                next_c0 = load4_aos(&sub_outputs[k + 16 + 2 * quarter]);
                next_c1 = load4_aos(&sub_outputs[k + 20 + 2 * quarter]);
                next_c2 = load4_aos(&sub_outputs[k + 24 + 2 * quarter]);
                next_c3 = load4_aos(&sub_outputs[k + 28 + 2 * quarter]);

                next_d0 = load4_aos(&sub_outputs[k + 16 + 3 * quarter]);
                next_d1 = load4_aos(&sub_outputs[k + 20 + 3 * quarter]);
                next_d2 = load4_aos(&sub_outputs[k + 24 + 3 * quarter]);
                next_d3 = load4_aos(&sub_outputs[k + 28 + 3 * quarter]);

                next_w1_0 = load4_aos(&stage_tw[3 * (k + 16)]);
                next_w1_1 = load4_aos(&stage_tw[3 * (k + 20)]);
                next_w1_2 = load4_aos(&stage_tw[3 * (k + 24)]);
                next_w1_3 = load4_aos(&stage_tw[3 * (k + 28)]);

                next_w2_0 = load4_aos(&stage_tw[3 * (k + 16) + 1]);
                next_w2_1 = load4_aos(&stage_tw[3 * (k + 20) + 1]);
                next_w2_2 = load4_aos(&stage_tw[3 * (k + 24) + 1]);
                next_w2_3 = load4_aos(&stage_tw[3 * (k + 28) + 1]);

                next_w3_0 = load4_aos(&stage_tw[3 * (k + 16) + 2]);
                next_w3_1 = load4_aos(&stage_tw[3 * (k + 20) + 2]);
                next_w3_2 = load4_aos(&stage_tw[3 * (k + 24) + 2]);
                next_w3_3 = load4_aos(&stage_tw[3 * (k + 28) + 2]);
            }

            // Compute: twiddle multiply
            __m512d b2_0 = cmul_avx512_aos(b0, w1_0);
            __m512d b2_1 = cmul_avx512_aos(b1, w1_1);
            __m512d b2_2 = cmul_avx512_aos(b2, w1_2);
            __m512d b2_3 = cmul_avx512_aos(b3, w1_3);

            __m512d c2_0 = cmul_avx512_aos(c0, w2_0);
            __m512d c2_1 = cmul_avx512_aos(c1, w2_1);
            __m512d c2_2 = cmul_avx512_aos(c2, w2_2);
            __m512d c2_3 = cmul_avx512_aos(c3, w2_3);

            __m512d d2_0 = cmul_avx512_aos(d0, w3_0);
            __m512d d2_1 = cmul_avx512_aos(d1, w3_1);
            __m512d d2_2 = cmul_avx512_aos(d2, w3_2);
            __m512d d2_3 = cmul_avx512_aos(d3, w3_3);

            // Butterfly
            __m512d y0_0, y1_0, y2_0, y3_0;
            __m512d y0_1, y1_1, y2_1, y3_1;
            __m512d y0_2, y1_2, y2_2, y3_2;
            __m512d y0_3, y1_3, y2_3, y3_3;

            RADIX4_BUTTERFLY_AVX512(a0, b2_0, c2_0, d2_0, y0_0, y1_0, y2_0, y3_0);
            RADIX4_BUTTERFLY_AVX512(a1, b2_1, c2_1, d2_1, y0_1, y1_1, y2_1, y3_1);
            RADIX4_BUTTERFLY_AVX512(a2, b2_2, c2_2, d2_2, y0_2, y1_2, y2_2, y3_2);
            RADIX4_BUTTERFLY_AVX512(a3, b2_3, c2_3, d2_3, y0_3, y1_3, y2_3, y3_3);

            // Store (non-temporal for large FFTs)
            if (quarter >= 4096)
            {
                _mm512_stream_pd((double *)&output_buffer[k + 0].re, y0_0);
                _mm512_stream_pd((double *)&output_buffer[k + 4].re, y0_1);
                _mm512_stream_pd((double *)&output_buffer[k + 8].re, y0_2);
                _mm512_stream_pd((double *)&output_buffer[k + 12].re, y0_3);

                _mm512_stream_pd((double *)&output_buffer[k + 0 + quarter].re, y1_0);
                _mm512_stream_pd((double *)&output_buffer[k + 4 + quarter].re, y1_1);
                _mm512_stream_pd((double *)&output_buffer[k + 8 + quarter].re, y1_2);
                _mm512_stream_pd((double *)&output_buffer[k + 12 + quarter].re, y1_3);

                _mm512_stream_pd((double *)&output_buffer[k + 0 + 2 * quarter].re, y2_0);
                _mm512_stream_pd((double *)&output_buffer[k + 4 + 2 * quarter].re, y2_1);
                _mm512_stream_pd((double *)&output_buffer[k + 8 + 2 * quarter].re, y2_2);
                _mm512_stream_pd((double *)&output_buffer[k + 12 + 2 * quarter].re, y2_3);

                _mm512_stream_pd((double *)&output_buffer[k + 0 + 3 * quarter].re, y3_0);
                _mm512_stream_pd((double *)&output_buffer[k + 4 + 3 * quarter].re, y3_1);
                _mm512_stream_pd((double *)&output_buffer[k + 8 + 3 * quarter].re, y3_2);
                _mm512_stream_pd((double *)&output_buffer[k + 12 + 3 * quarter].re, y3_3);
            }
            else
            {
                STOREU_PD512(&output_buffer[k + 0].re, y0_0);
                STOREU_PD512(&output_buffer[k + 4].re, y0_1);
                STOREU_PD512(&output_buffer[k + 8].re, y0_2);
                STOREU_PD512(&output_buffer[k + 12].re, y0_3);

                STOREU_PD512(&output_buffer[k + 0 + quarter].re, y1_0);
                STOREU_PD512(&output_buffer[k + 4 + quarter].re, y1_1);
                STOREU_PD512(&output_buffer[k + 8 + quarter].re, y1_2);
                STOREU_PD512(&output_buffer[k + 12 + quarter].re, y1_3);

                STOREU_PD512(&output_buffer[k + 0 + 2 * quarter].re, y2_0);
                STOREU_PD512(&output_buffer[k + 4 + 2 * quarter].re, y2_1);
                STOREU_PD512(&output_buffer[k + 8 + 2 * quarter].re, y2_2);
                STOREU_PD512(&output_buffer[k + 12 + 2 * quarter].re, y2_3);

                STOREU_PD512(&output_buffer[k + 0 + 3 * quarter].re, y3_0);
                STOREU_PD512(&output_buffer[k + 4 + 3 * quarter].re, y3_1);
                STOREU_PD512(&output_buffer[k + 8 + 3 * quarter].re, y3_2);
                STOREU_PD512(&output_buffer[k + 12 + 3 * quarter].re, y3_3);
            }
        }
    }

    // Standard cleanup
    for (; k + 15 < quarter; k += 16)
    {
        if (k + 32 < quarter)
        {
            _mm_prefetch((const char *)&sub_outputs[k + 32].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&stage_tw[3 * (k + 32)].re, _MM_HINT_T0);
        }

        __m512d a0 = load4_aos(&sub_outputs[k + 0]);
        __m512d a1 = load4_aos(&sub_outputs[k + 4]);
        __m512d a2 = load4_aos(&sub_outputs[k + 8]);
        __m512d a3 = load4_aos(&sub_outputs[k + 12]);

        __m512d b0 = load4_aos(&sub_outputs[k + 0 + quarter]);
        __m512d b1 = load4_aos(&sub_outputs[k + 4 + quarter]);
        __m512d b2 = load4_aos(&sub_outputs[k + 8 + quarter]);
        __m512d b3 = load4_aos(&sub_outputs[k + 12 + quarter]);

        __m512d c0 = load4_aos(&sub_outputs[k + 0 + 2 * quarter]);
        __m512d c1 = load4_aos(&sub_outputs[k + 4 + 2 * quarter]);
        __m512d c2 = load4_aos(&sub_outputs[k + 8 + 2 * quarter]);
        __m512d c3 = load4_aos(&sub_outputs[k + 12 + 2 * quarter]);

        __m512d d0 = load4_aos(&sub_outputs[k + 0 + 3 * quarter]);
        __m512d d1 = load4_aos(&sub_outputs[k + 4 + 3 * quarter]);
        __m512d d2 = load4_aos(&sub_outputs[k + 8 + 3 * quarter]);
        __m512d d3 = load4_aos(&sub_outputs[k + 12 + 3 * quarter]);

        __m512d w1_0 = load4_aos(&stage_tw[3 * (k + 0)]);
        __m512d w1_1 = load4_aos(&stage_tw[3 * (k + 4)]);
        __m512d w1_2 = load4_aos(&stage_tw[3 * (k + 8)]);
        __m512d w1_3 = load4_aos(&stage_tw[3 * (k + 12)]);

        __m512d w2_0 = load4_aos(&stage_tw[3 * (k + 0) + 1]);
        __m512d w2_1 = load4_aos(&stage_tw[3 * (k + 4) + 1]);
        __m512d w2_2 = load4_aos(&stage_tw[3 * (k + 8) + 1]);
        __m512d w2_3 = load4_aos(&stage_tw[3 * (k + 12) + 1]);

        __m512d w3_0 = load4_aos(&stage_tw[3 * (k + 0) + 2]);
        __m512d w3_1 = load4_aos(&stage_tw[3 * (k + 4) + 2]);
        __m512d w3_2 = load4_aos(&stage_tw[3 * (k + 8) + 2]);
        __m512d w3_3 = load4_aos(&stage_tw[3 * (k + 12) + 2]);

        __m512d b2_0 = cmul_avx512_aos(b0, w1_0);
        __m512d b2_1 = cmul_avx512_aos(b1, w1_1);
        __m512d b2_2 = cmul_avx512_aos(b2, w1_2);
        __m512d b2_3 = cmul_avx512_aos(b3, w1_3);

        __m512d c2_0 = cmul_avx512_aos(c0, w2_0);
        __m512d c2_1 = cmul_avx512_aos(c1, w2_1);
        __m512d c2_2 = cmul_avx512_aos(c2, w2_2);
        __m512d c2_3 = cmul_avx512_aos(c3, w2_3);

        __m512d d2_0 = cmul_avx512_aos(d0, w3_0);
        __m512d d2_1 = cmul_avx512_aos(d1, w3_1);
        __m512d d2_2 = cmul_avx512_aos(d2, w3_2);
        __m512d d2_3 = cmul_avx512_aos(d3, w3_3);

        __m512d y0_0, y1_0, y2_0, y3_0;
        __m512d y0_1, y1_1, y2_1, y3_1;
        __m512d y0_2, y1_2, y2_2, y3_2;
        __m512d y0_3, y1_3, y2_3, y3_3;

        RADIX4_BUTTERFLY_AVX512(a0, b2_0, c2_0, d2_0, y0_0, y1_0, y2_0, y3_0);
        RADIX4_BUTTERFLY_AVX512(a1, b2_1, c2_1, d2_1, y0_1, y1_1, y2_1, y3_1);
        RADIX4_BUTTERFLY_AVX512(a2, b2_2, c2_2, d2_2, y0_2, y1_2, y2_2, y3_2);
        RADIX4_BUTTERFLY_AVX512(a3, b2_3, c2_3, d2_3, y0_3, y1_3, y2_3, y3_3);

        STOREU_PD512(&output_buffer[k + 0].re, y0_0);
        STOREU_PD512(&output_buffer[k + 4].re, y0_1);
        STOREU_PD512(&output_buffer[k + 8].re, y0_2);
        STOREU_PD512(&output_buffer[k + 12].re, y0_3);

        STOREU_PD512(&output_buffer[k + 0 + quarter].re, y1_0);
        STOREU_PD512(&output_buffer[k + 4 + quarter].re, y1_1);
        STOREU_PD512(&output_buffer[k + 8 + quarter].re, y1_2);
        STOREU_PD512(&output_buffer[k + 12 + quarter].re, y1_3);

        STOREU_PD512(&output_buffer[k + 0 + 2 * quarter].re, y2_0);
        STOREU_PD512(&output_buffer[k + 4 + 2 * quarter].re, y2_1);
        STOREU_PD512(&output_buffer[k + 8 + 2 * quarter].re, y2_2);
        STOREU_PD512(&output_buffer[k + 12 + 2 * quarter].re, y2_3);

        STOREU_PD512(&output_buffer[k + 0 + 3 * quarter].re, y3_0);
        STOREU_PD512(&output_buffer[k + 4 + 3 * quarter].re, y3_1);
        STOREU_PD512(&output_buffer[k + 8 + 3 * quarter].re, y3_2);
        STOREU_PD512(&output_buffer[k + 12 + 3 * quarter].re, y3_3);
    }

    // 8x cleanup
    for (; k + 7 < quarter; k += 8)
    {
        __m512d a0 = load4_aos(&sub_outputs[k + 0]);
        __m512d a1 = load4_aos(&sub_outputs[k + 4]);

        __m512d b0 = load4_aos(&sub_outputs[k + 0 + quarter]);
        __m512d b1 = load4_aos(&sub_outputs[k + 4 + quarter]);

        __m512d c0 = load4_aos(&sub_outputs[k + 0 + 2 * quarter]);
        __m512d c1 = load4_aos(&sub_outputs[k + 4 + 2 * quarter]);

        __m512d d0 = load4_aos(&sub_outputs[k + 0 + 3 * quarter]);
        __m512d d1 = load4_aos(&sub_outputs[k + 4 + 3 * quarter]);

        __m512d w1_0 = load4_aos(&stage_tw[3 * (k + 0)]);
        __m512d w1_1 = load4_aos(&stage_tw[3 * (k + 4)]);

        __m512d w2_0 = load4_aos(&stage_tw[3 * (k + 0) + 1]);
        __m512d w2_1 = load4_aos(&stage_tw[3 * (k + 4) + 1]);

        __m512d w3_0 = load4_aos(&stage_tw[3 * (k + 0) + 2]);
        __m512d w3_1 = load4_aos(&stage_tw[3 * (k + 4) + 2]);

        __m512d b2_0 = cmul_avx512_aos(b0, w1_0);
        __m512d b2_1 = cmul_avx512_aos(b1, w1_1);

        __m512d c2_0 = cmul_avx512_aos(c0, w2_0);
        __m512d c2_1 = cmul_avx512_aos(c1, w2_1);

        __m512d d2_0 = cmul_avx512_aos(d0, w3_0);
        __m512d d2_1 = cmul_avx512_aos(d1, w3_1);

        __m512d y0_0, y1_0, y2_0, y3_0;
        __m512d y0_1, y1_1, y2_1, y3_1;

        RADIX4_BUTTERFLY_AVX512(a0, b2_0, c2_0, d2_0, y0_0, y1_0, y2_0, y3_0);
        RADIX4_BUTTERFLY_AVX512(a1, b2_1, c2_1, d2_1, y0_1, y1_1, y2_1, y3_1);

        STOREU_PD512(&output_buffer[k + 0].re, y0_0);
        STOREU_PD512(&output_buffer[k + 4].re, y0_1);

        STOREU_PD512(&output_buffer[k + 0 + quarter].re, y1_0);
        STOREU_PD512(&output_buffer[k + 4 + quarter].re, y1_1);

        STOREU_PD512(&output_buffer[k + 0 + 2 * quarter].re, y2_0);
        STOREU_PD512(&output_buffer[k + 4 + 2 * quarter].re, y2_1);

        STOREU_PD512(&output_buffer[k + 0 + 3 * quarter].re, y3_0);
        STOREU_PD512(&output_buffer[k + 4 + 3 * quarter].re, y3_1);
    }

    // 4x cleanup
    for (; k + 3 < quarter; k += 4)
    {
        __m512d a = load4_aos(&sub_outputs[k]);
        __m512d b = load4_aos(&sub_outputs[k + quarter]);
        __m512d c = load4_aos(&sub_outputs[k + 2 * quarter]);
        __m512d d = load4_aos(&sub_outputs[k + 3 * quarter]);

        __m512d w1 = load4_aos(&stage_tw[3 * k]);
        __m512d w2 = load4_aos(&stage_tw[3 * k + 1]);
        __m512d w3 = load4_aos(&stage_tw[3 * k + 2]);

        __m512d b2 = cmul_avx512_aos(b, w1);
        __m512d c2 = cmul_avx512_aos(c, w2);
        __m512d d2 = cmul_avx512_aos(d, w3);

        __m512d y0, y1, y2, y3;
        RADIX4_BUTTERFLY_AVX512(a, b2, c2, d2, y0, y1, y2, y3);

        STOREU_PD512(&output_buffer[k].re, y0);
        STOREU_PD512(&output_buffer[k + quarter].re, y1);
        STOREU_PD512(&output_buffer[k + 2 * quarter].re, y2);
        STOREU_PD512(&output_buffer[k + 3 * quarter].re, y3);
    }

#undef RADIX4_BUTTERFLY_AVX512
#endif // HAS_AVX512

#ifdef __AVX2__
    //======================================================================
    // AVX2 PATH WITH OPTIONAL PARALLELIZATION
    //======================================================================

    const __m256d rot_mask = (transform_sign == 1)
                                 ? _mm256_set_pd(-0.0, 0.0, -0.0, 0.0)
                                 : _mm256_set_pd(0.0, -0.0, 0.0, -0.0);

#define RADIX4_BUTTERFLY_AVX2(a, b2, c2, d2, y0, y1, y2, y3)  \
    {                                                         \
        __m256d sumBD = _mm256_add_pd(b2, d2);                \
        __m256d difBD = _mm256_sub_pd(b2, d2);                \
        __m256d a_pc = _mm256_add_pd(a, c2);                  \
        __m256d a_mc = _mm256_sub_pd(a, c2);                  \
        y0 = _mm256_add_pd(a_pc, sumBD);                      \
        y2 = _mm256_sub_pd(a_pc, sumBD);                      \
        __m256d difBD_swp = _mm256_permute_pd(difBD, 0b0101); \
        __m256d rot = _mm256_xor_pd(difBD_swp, rot_mask);     \
        y1 = _mm256_sub_pd(a_mc, rot);                        \
        y3 = _mm256_add_pd(a_mc, rot);                        \
    }

#if USE_OPENMP
    if (use_parallel && quarter - k >= 8)  // Need at least 8 elements
    {
        const int num_blocks = (quarter - k) / 8;  // Number of 8-element blocks
        
        #pragma omp parallel for schedule(static)
        for (int block_idx = 0; block_idx < num_blocks; block_idx++)
        {
            int kk = k + block_idx * 8;  // Calculate position for this block
            
            // NO inner loop - just process this one 8-element block directly
            if (kk + 16 < quarter)
            {
                _mm_prefetch((const char *)&sub_outputs[kk + 16].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&stage_tw[3 * (kk + 16)].re, _MM_HINT_T0);
            }

            // Load 8 butterflies (2 complex per AVX2 register)
            __m256d a0 = load2_aos(&sub_outputs[kk + 0], &sub_outputs[kk + 1]);
            __m256d a1 = load2_aos(&sub_outputs[kk + 2], &sub_outputs[kk + 3]);
            __m256d a2 = load2_aos(&sub_outputs[kk + 4], &sub_outputs[kk + 5]);
            __m256d a3 = load2_aos(&sub_outputs[kk + 6], &sub_outputs[kk + 7]);

            __m256d b0 = load2_aos(&sub_outputs[kk + 0 + quarter], &sub_outputs[kk + 1 + quarter]);
            __m256d b1 = load2_aos(&sub_outputs[kk + 2 + quarter], &sub_outputs[kk + 3 + quarter]);
            __m256d b2 = load2_aos(&sub_outputs[kk + 4 + quarter], &sub_outputs[kk + 5 + quarter]);
            __m256d b3 = load2_aos(&sub_outputs[kk + 6 + quarter], &sub_outputs[kk + 7 + quarter]);

            __m256d c0 = load2_aos(&sub_outputs[kk + 0 + 2 * quarter], &sub_outputs[kk + 1 + 2 * quarter]);
            __m256d c1 = load2_aos(&sub_outputs[kk + 2 + 2 * quarter], &sub_outputs[kk + 3 + 2 * quarter]);
            __m256d c2 = load2_aos(&sub_outputs[kk + 4 + 2 * quarter], &sub_outputs[kk + 5 + 2 * quarter]);
            __m256d c3 = load2_aos(&sub_outputs[kk + 6 + 2 * quarter], &sub_outputs[kk + 7 + 2 * quarter]);

            __m256d d0 = load2_aos(&sub_outputs[kk + 0 + 3 * quarter], &sub_outputs[kk + 1 + 3 * quarter]);
            __m256d d1 = load2_aos(&sub_outputs[kk + 2 + 3 * quarter], &sub_outputs[kk + 3 + 3 * quarter]);
            __m256d d2 = load2_aos(&sub_outputs[kk + 4 + 3 * quarter], &sub_outputs[kk + 5 + 3 * quarter]);
            __m256d d3 = load2_aos(&sub_outputs[kk + 6 + 3 * quarter], &sub_outputs[kk + 7 + 3 * quarter]);

            __m256d w1_0 = load2_aos(&stage_tw[3 * (kk + 0)], &stage_tw[3 * (kk + 1)]);
            __m256d w1_1 = load2_aos(&stage_tw[3 * (kk + 2)], &stage_tw[3 * (kk + 3)]);
            __m256d w1_2 = load2_aos(&stage_tw[3 * (kk + 4)], &stage_tw[3 * (kk + 5)]);
            __m256d w1_3 = load2_aos(&stage_tw[3 * (kk + 6)], &stage_tw[3 * (kk + 7)]);

            __m256d w2_0 = load2_aos(&stage_tw[3 * (kk + 0) + 1], &stage_tw[3 * (kk + 1) + 1]);
            __m256d w2_1 = load2_aos(&stage_tw[3 * (kk + 2) + 1], &stage_tw[3 * (kk + 3) + 1]);
            __m256d w2_2 = load2_aos(&stage_tw[3 * (kk + 4) + 1], &stage_tw[3 * (kk + 5) + 1]);
            __m256d w2_3 = load2_aos(&stage_tw[3 * (kk + 6) + 1], &stage_tw[3 * (kk + 7) + 1]);

            __m256d w3_0 = load2_aos(&stage_tw[3 * (kk + 0) + 2], &stage_tw[3 * (kk + 1) + 2]);
            __m256d w3_1 = load2_aos(&stage_tw[3 * (kk + 2) + 2], &stage_tw[3 * (kk + 3) + 2]);
            __m256d w3_2 = load2_aos(&stage_tw[3 * (kk + 4) + 2], &stage_tw[3 * (kk + 5) + 2]);
            __m256d w3_3 = load2_aos(&stage_tw[3 * (kk + 6) + 2], &stage_tw[3 * (kk + 7) + 2]);

            __m256d b2_0 = cmul_avx2_aos(b0, w1_0);
            __m256d b2_1 = cmul_avx2_aos(b1, w1_1);
            __m256d b2_2 = cmul_avx2_aos(b2, w1_2);
            __m256d b2_3 = cmul_avx2_aos(b3, w1_3);

            __m256d c2_0 = cmul_avx2_aos(c0, w2_0);
            __m256d c2_1 = cmul_avx2_aos(c1, w2_1);
            __m256d c2_2 = cmul_avx2_aos(c2, w2_2);
            __m256d c2_3 = cmul_avx2_aos(c3, w2_3);

            __m256d d2_0 = cmul_avx2_aos(d0, w3_0);
            __m256d d2_1 = cmul_avx2_aos(d1, w3_1);
            __m256d d2_2 = cmul_avx2_aos(d2, w3_2);
            __m256d d2_3 = cmul_avx2_aos(d3, w3_3);

            __m256d y0_0, y1_0, y2_0, y3_0;
            __m256d y0_1, y1_1, y2_1, y3_1;
            __m256d y0_2, y1_2, y2_2, y3_2;
            __m256d y0_3, y1_3, y2_3, y3_3;

            RADIX4_BUTTERFLY_AVX2(a0, b2_0, c2_0, d2_0, y0_0, y1_0, y2_0, y3_0);
            RADIX4_BUTTERFLY_AVX2(a1, b2_1, c2_1, d2_1, y0_1, y1_1, y2_1, y3_1);
            RADIX4_BUTTERFLY_AVX2(a2, b2_2, c2_2, d2_2, y0_2, y1_2, y2_2, y3_2);
            RADIX4_BUTTERFLY_AVX2(a3, b2_3, c2_3, d2_3, y0_3, y1_3, y2_3, y3_3);

            STOREU_PD(&output_buffer[kk + 0].re, y0_0);
            STOREU_PD(&output_buffer[kk + 2].re, y0_1);
            STOREU_PD(&output_buffer[kk + 4].re, y0_2);
            STOREU_PD(&output_buffer[kk + 6].re, y0_3);

            STOREU_PD(&output_buffer[kk + 0 + quarter].re, y1_0);
            STOREU_PD(&output_buffer[kk + 2 + quarter].re, y1_1);
            STOREU_PD(&output_buffer[kk + 4 + quarter].re, y1_2);
            STOREU_PD(&output_buffer[kk + 6 + quarter].re, y1_3);

            STOREU_PD(&output_buffer[kk + 0 + 2 * quarter].re, y2_0);
            STOREU_PD(&output_buffer[kk + 2 + 2 * quarter].re, y2_1);
            STOREU_PD(&output_buffer[kk + 4 + 2 * quarter].re, y2_2);
            STOREU_PD(&output_buffer[kk + 6 + 2 * quarter].re, y2_3);

            STOREU_PD(&output_buffer[kk + 0 + 3 * quarter].re, y3_0);
            STOREU_PD(&output_buffer[kk + 2 + 3 * quarter].re, y3_1);
            STOREU_PD(&output_buffer[kk + 4 + 3 * quarter].re, y3_2);
            STOREU_PD(&output_buffer[kk + 6 + 3 * quarter].re, y3_3);
        }
        k += num_blocks * 8;  // Advance k by all blocks processed
    }
#endif

    // Serial AVX2
    for (; k + 7 < quarter; k += 8)
    {
        if (k + 16 < quarter)
        {
            _mm_prefetch((const char *)&sub_outputs[k + 16].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_outputs[k + 16 + quarter].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_outputs[k + 16 + 2 * quarter].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_outputs[k + 16 + 3 * quarter].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&stage_tw[3 * (k + 16)].re, _MM_HINT_T0);
        }

        __m256d a0 = load2_aos(&sub_outputs[k + 0], &sub_outputs[k + 1]);
        __m256d a1 = load2_aos(&sub_outputs[k + 2], &sub_outputs[k + 3]);
        __m256d a2 = load2_aos(&sub_outputs[k + 4], &sub_outputs[k + 5]);
        __m256d a3 = load2_aos(&sub_outputs[k + 6], &sub_outputs[k + 7]);

        __m256d b0 = load2_aos(&sub_outputs[k + 0 + quarter], &sub_outputs[k + 1 + quarter]);
        __m256d b1 = load2_aos(&sub_outputs[k + 2 + quarter], &sub_outputs[k + 3 + quarter]);
        __m256d b2 = load2_aos(&sub_outputs[k + 4 + quarter], &sub_outputs[k + 5 + quarter]);
        __m256d b3 = load2_aos(&sub_outputs[k + 6 + quarter], &sub_outputs[k + 7 + quarter]);

        __m256d c0 = load2_aos(&sub_outputs[k + 0 + 2 * quarter], &sub_outputs[k + 1 + 2 * quarter]);
        __m256d c1 = load2_aos(&sub_outputs[k + 2 + 2 * quarter], &sub_outputs[k + 3 + 2 * quarter]);
        __m256d c2 = load2_aos(&sub_outputs[k + 4 + 2 * quarter], &sub_outputs[k + 5 + 2 * quarter]);
        __m256d c3 = load2_aos(&sub_outputs[k + 6 + 2 * quarter], &sub_outputs[k + 7 + 2 * quarter]);

        __m256d d0 = load2_aos(&sub_outputs[k + 0 + 3 * quarter], &sub_outputs[k + 1 + 3 * quarter]);
        __m256d d1 = load2_aos(&sub_outputs[k + 2 + 3 * quarter], &sub_outputs[k + 3 + 3 * quarter]);
        __m256d d2 = load2_aos(&sub_outputs[k + 4 + 3 * quarter], &sub_outputs[k + 5 + 3 * quarter]);
        __m256d d3 = load2_aos(&sub_outputs[k + 6 + 3 * quarter], &sub_outputs[k + 7 + 3 * quarter]);

        __m256d w1_0 = load2_aos(&stage_tw[3 * (k + 0)], &stage_tw[3 * (k + 1)]);
        __m256d w1_1 = load2_aos(&stage_tw[3 * (k + 2)], &stage_tw[3 * (k + 3)]);
        __m256d w1_2 = load2_aos(&stage_tw[3 * (k + 4)], &stage_tw[3 * (k + 5)]);
        __m256d w1_3 = load2_aos(&stage_tw[3 * (k + 6)], &stage_tw[3 * (k + 7)]);

        __m256d w2_0 = load2_aos(&stage_tw[3 * (k + 0) + 1], &stage_tw[3 * (k + 1) + 1]);
        __m256d w2_1 = load2_aos(&stage_tw[3 * (k + 2) + 1], &stage_tw[3 * (k + 3) + 1]);
        __m256d w2_2 = load2_aos(&stage_tw[3 * (k + 4) + 1], &stage_tw[3 * (k + 5) + 1]);
        __m256d w2_3 = load2_aos(&stage_tw[3 * (k + 6) + 1], &stage_tw[3 * (k + 7) + 1]);

        __m256d w3_0 = load2_aos(&stage_tw[3 * (k + 0) + 2], &stage_tw[3 * (k + 1) + 2]);
        __m256d w3_1 = load2_aos(&stage_tw[3 * (k + 2) + 2], &stage_tw[3 * (k + 3) + 2]);
        __m256d w3_2 = load2_aos(&stage_tw[3 * (k + 4) + 2], &stage_tw[3 * (k + 5) + 2]);
        __m256d w3_3 = load2_aos(&stage_tw[3 * (k + 6) + 2], &stage_tw[3 * (k + 7) + 2]);

        __m256d b2_0 = cmul_avx2_aos(b0, w1_0);
        __m256d b2_1 = cmul_avx2_aos(b1, w1_1);
        __m256d b2_2 = cmul_avx2_aos(b2, w1_2);
        __m256d b2_3 = cmul_avx2_aos(b3, w1_3);

        __m256d c2_0 = cmul_avx2_aos(c0, w2_0);
        __m256d c2_1 = cmul_avx2_aos(c1, w2_1);
        __m256d c2_2 = cmul_avx2_aos(c2, w2_2);
        __m256d c2_3 = cmul_avx2_aos(c3, w2_3);

        __m256d d2_0 = cmul_avx2_aos(d0, w3_0);
        __m256d d2_1 = cmul_avx2_aos(d1, w3_1);
        __m256d d2_2 = cmul_avx2_aos(d2, w3_2);
        __m256d d2_3 = cmul_avx2_aos(d3, w3_3);

        __m256d y0_0, y1_0, y2_0, y3_0;
        __m256d y0_1, y1_1, y2_1, y3_1;
        __m256d y0_2, y1_2, y2_2, y3_2;
        __m256d y0_3, y1_3, y2_3, y3_3;

        RADIX4_BUTTERFLY_AVX2(a0, b2_0, c2_0, d2_0, y0_0, y1_0, y2_0, y3_0);
        RADIX4_BUTTERFLY_AVX2(a1, b2_1, c2_1, d2_1, y0_1, y1_1, y2_1, y3_1);
        RADIX4_BUTTERFLY_AVX2(a2, b2_2, c2_2, d2_2, y0_2, y1_2, y2_2, y3_2);
        RADIX4_BUTTERFLY_AVX2(a3, b2_3, c2_3, d2_3, y0_3, y1_3, y2_3, y3_3);

        STOREU_PD(&output_buffer[k + 0].re, y0_0);
        STOREU_PD(&output_buffer[k + 2].re, y0_1);
        STOREU_PD(&output_buffer[k + 4].re, y0_2);
        STOREU_PD(&output_buffer[k + 6].re, y0_3);

        STOREU_PD(&output_buffer[k + 0 + quarter].re, y1_0);
        STOREU_PD(&output_buffer[k + 2 + quarter].re, y1_1);
        STOREU_PD(&output_buffer[k + 4 + quarter].re, y1_2);
        STOREU_PD(&output_buffer[k + 6 + quarter].re, y1_3);

        STOREU_PD(&output_buffer[k + 0 + 2 * quarter].re, y2_0);
        STOREU_PD(&output_buffer[k + 2 + 2 * quarter].re, y2_1);
        STOREU_PD(&output_buffer[k + 4 + 2 * quarter].re, y2_2);
        STOREU_PD(&output_buffer[k + 6 + 2 * quarter].re, y2_3);

        STOREU_PD(&output_buffer[k + 0 + 3 * quarter].re, y3_0);
        STOREU_PD(&output_buffer[k + 2 + 3 * quarter].re, y3_1);
        STOREU_PD(&output_buffer[k + 4 + 3 * quarter].re, y3_2);
        STOREU_PD(&output_buffer[k + 6 + 3 * quarter].re, y3_3);
    }

    // 2x cleanup
    const __m256d rot_mask_final = (transform_sign == 1)
                                       ? _mm256_set_pd(-0.0, 0.0, -0.0, 0.0)
                                       : _mm256_set_pd(0.0, -0.0, 0.0, -0.0);

    for (; k + 1 < quarter; k += 2)
    {
        if (k + 8 < quarter)
        {
            _mm_prefetch((const char *)&sub_outputs[k + 8].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_outputs[k + 8 + quarter].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_outputs[k + 8 + 2 * quarter].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_outputs[k + 8 + 3 * quarter].re, _MM_HINT_T0);
        }

        __m256d a = load2_aos(&sub_outputs[k], &sub_outputs[k + 1]);
        __m256d b = load2_aos(&sub_outputs[k + quarter], &sub_outputs[k + quarter + 1]);
        __m256d c = load2_aos(&sub_outputs[k + 2 * quarter], &sub_outputs[k + 2 * quarter + 1]);
        __m256d d = load2_aos(&sub_outputs[k + 3 * quarter], &sub_outputs[k + 3 * quarter + 1]);

        __m256d w1 = load2_aos(&stage_tw[3 * k], &stage_tw[3 * (k + 1)]);
        __m256d w2 = load2_aos(&stage_tw[3 * k + 1], &stage_tw[3 * (k + 1) + 1]);
        __m256d w3 = load2_aos(&stage_tw[3 * k + 2], &stage_tw[3 * (k + 1) + 2]);

        __m256d b2 = cmul_avx2_aos(b, w1);
        __m256d c2 = cmul_avx2_aos(c, w2);
        __m256d d2 = cmul_avx2_aos(d, w3);

        __m256d sumBD = _mm256_add_pd(b2, d2);
        __m256d difBD = _mm256_sub_pd(b2, d2);
        __m256d a_pc = _mm256_add_pd(a, c2);
        __m256d a_mc = _mm256_sub_pd(a, c2);

        __m256d y0 = _mm256_add_pd(a_pc, sumBD);
        __m256d y2 = _mm256_sub_pd(a_pc, sumBD);

        __m256d difBD_swp = _mm256_permute_pd(difBD, 0b0101);
        __m256d rot = _mm256_xor_pd(difBD_swp, rot_mask_final);

        __m256d y1 = _mm256_sub_pd(a_mc, rot);
        __m256d y3 = _mm256_add_pd(a_mc, rot);

        STOREU_PD(&output_buffer[k].re, y0);
        STOREU_PD(&output_buffer[k + quarter].re, y1);
        STOREU_PD(&output_buffer[k + 2 * quarter].re, y2);
        STOREU_PD(&output_buffer[k + 3 * quarter].re, y3);
    }
#undef RADIX4_BUTTERFLY_AVX2
#endif // __AVX2__

    //======================================================================
    // SSE2 TAIL
    //======================================================================
    for (; k < quarter; ++k)
    {
        __m128d a = LOADU_SSE2(&sub_outputs[k].re);
        __m128d b = LOADU_SSE2(&sub_outputs[k + quarter].re);
        __m128d c = LOADU_SSE2(&sub_outputs[k + 2 * quarter].re);
        __m128d d = LOADU_SSE2(&sub_outputs[k + 3 * quarter].re);

        __m128d w1 = LOADU_SSE2(&stage_tw[3 * k].re);
        __m128d w2 = LOADU_SSE2(&stage_tw[3 * k + 1].re);
        __m128d w3 = LOADU_SSE2(&stage_tw[3 * k + 2].re);

        __m128d b2 = cmul_sse2_aos(b, w1);
        __m128d c2 = cmul_sse2_aos(c, w2);
        __m128d d2 = cmul_sse2_aos(d, w3);

        __m128d sumBD = _mm_add_pd(b2, d2);
        __m128d difBD = _mm_sub_pd(b2, d2);
        __m128d a_pc = _mm_add_pd(a, c2);
        __m128d a_mc = _mm_sub_pd(a, c2);

        __m128d y0 = _mm_add_pd(a_pc, sumBD);
        __m128d y2 = _mm_sub_pd(a_pc, sumBD);

        __m128d swp = _mm_shuffle_pd(difBD, difBD, 0b01);
        __m128d rot = (transform_sign == 1)
                          ? _mm_xor_pd(swp, _mm_set_pd(0.0, -0.0))
                          : _mm_xor_pd(swp, _mm_set_pd(-0.0, 0.0));

        __m128d y1 = _mm_sub_pd(a_mc, rot);
        __m128d y3 = _mm_add_pd(a_mc, rot);

        STOREU_SSE2(&output_buffer[k].re, y0);
        STOREU_SSE2(&output_buffer[k + quarter].re, y1);
        STOREU_SSE2(&output_buffer[k + 2 * quarter].re, y2);
        STOREU_SSE2(&output_buffer[k + 3 * quarter].re, y3);
    }

    // Memory fence for non-temporal stores
    if (quarter >= 4096)
    {
        _mm_sfence();
    }
}