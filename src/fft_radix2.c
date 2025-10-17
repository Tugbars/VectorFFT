#include "fft_radix2.h"
#include "simd_math.h"

// Compiler-specific alignment hints
#if defined(_MSC_VER)
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#elif defined(__GNUC__) || defined(__clang__)
#define ASSUME_ALIGNED(ptr, alignment) __builtin_assume_aligned(ptr, alignment)
#else
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#endif

// OpenMP detection via CMake flag
#ifdef HAVE_OPENMP
#include <omp.h>
#define USE_OPENMP 1
#else
#define USE_OPENMP 0
#endif

// Configuration constants
#define RADIX2_PREFETCH_DISTANCE 128   // L1 prefetch distance
#define RADIX2_PARALLEL_THRESHOLD 1024 // Minimum size for threading

/**
 * @brief Ultra-optimized radix-2 butterfly with advanced micro-architectural tuning
 *
 * ADVANCED OPTIMIZATIONS IMPLEMENTED:
 * 1. Software pipelining - overlaps load/compute/store across iterations
 * 2. Non-temporal stores - bypasses cache for write-only data
 * 3. Explicit alignment hints - enables aligned vector operations
 * 4. Multi-threading - parallelizes independent butterflies
 * 5. Prefetch distance tuning - matched to cache hierarchy
 *
 * PERFORMANCE TARGETS:
 * - AVX-512: ~0.5 cycles/butterfly (theoretical min: 0.33)
 * - Memory bandwidth: <0.5 of theoretical peak
 * - Cache miss rate: <1%
 */

void fft_radix2_butterfly(
    fft_data *output_buffer,
    fft_data *sub_outputs,
    const fft_data *stage_tw,
    int sub_len,
    int transform_sign)
{
    const int half = sub_len;

    // Portable alignment hints
    output_buffer = (fft_data *)ASSUME_ALIGNED(output_buffer, 64);
    sub_outputs = (fft_data *)ASSUME_ALIGNED(sub_outputs, 64);
    stage_tw = (const fft_data *)ASSUME_ALIGNED(stage_tw, 64);

    //======================================================================
    // STAGE 0: SPECIAL CASE k=0 (W^0 = 1, NO TWIDDLE)
    //======================================================================
    {
        fft_data even_0 = sub_outputs[0];
        fft_data odd_0 = sub_outputs[half];
        output_buffer[0].re = even_0.re + odd_0.re;
        output_buffer[0].im = even_0.im + odd_0.im;
        output_buffer[half].re = even_0.re - odd_0.re;
        output_buffer[half].im = even_0.im - odd_0.im;
    }

    //======================================================================
    // STAGE 1: SPECIAL CASE k=N/4 (W^(N/4) = ±i, 90° ROTATION)
    //======================================================================
    int k_quarter = 0;
    if ((half & 1) == 0)
    {
        k_quarter = half >> 1;

        fft_data even_q = sub_outputs[k_quarter];
        fft_data odd_q = sub_outputs[half + k_quarter];

        double rotated_re = transform_sign > 0 ? odd_q.im : -odd_q.im;
        double rotated_im = transform_sign > 0 ? -odd_q.re : odd_q.re;

        output_buffer[k_quarter].re = even_q.re + rotated_re;
        output_buffer[k_quarter].im = even_q.im + rotated_im;
        output_buffer[half + k_quarter].re = even_q.re - rotated_re;
        output_buffer[half + k_quarter].im = even_q.im - rotated_im;
    }

    //======================================================================
    // STAGE 2: GENERAL CASE WITH ADVANCED OPTIMIZATIONS
    //======================================================================

    int k = 1;
    int range1_end = k_quarter ? k_quarter : half;

    // Decide whether to parallelize
    const int work_size = half - 1 - (k_quarter ? 1 : 0);
#if USE_OPENMP
    const int use_parallel = work_size >= RADIX2_PARALLEL_THRESHOLD;
#else
    const int use_parallel = 0;
    (void)use_parallel; // Suppress unused variable warning
#endif

#ifdef __AVX512F__
    //======================================================================
    // AVX-512 PATH WITH SOFTWARE PIPELINING
    //======================================================================

    // Software pipelined loop
    if (k + 31 < range1_end)
    {
        // Prologue: pre-load first iteration
        __m512d next_e0 = load4_aos(&sub_outputs[k + 0]);
        __m512d next_e1 = load4_aos(&sub_outputs[k + 4]);
        __m512d next_e2 = load4_aos(&sub_outputs[k + 8]);
        __m512d next_e3 = load4_aos(&sub_outputs[k + 12]);

        __m512d next_o0 = load4_aos(&sub_outputs[k + 0 + half]);
        __m512d next_o1 = load4_aos(&sub_outputs[k + 4 + half]);
        __m512d next_o2 = load4_aos(&sub_outputs[k + 8 + half]);
        __m512d next_o3 = load4_aos(&sub_outputs[k + 12 + half]);

        __m512d next_w0 = load4_aos(&stage_tw[k + 0]);
        __m512d next_w1 = load4_aos(&stage_tw[k + 4]);
        __m512d next_w2 = load4_aos(&stage_tw[k + 8]);
        __m512d next_w3 = load4_aos(&stage_tw[k + 12]);

        // Pipelined main loop
        for (; k + 31 < range1_end; k += 16)
        {
            // Use pre-loaded values
            __m512d e0 = next_e0, e1 = next_e1, e2 = next_e2, e3 = next_e3;
            __m512d o0 = next_o0, o1 = next_o1, o2 = next_o2, o3 = next_o3;
            __m512d w0 = next_w0, w1 = next_w1, w2 = next_w2, w3 = next_w3;

            // Load next iteration
            if (k + 47 < range1_end)
            {
                if (k + RADIX2_PREFETCH_DISTANCE < range1_end)
                {
                    _mm_prefetch((const char *)&sub_outputs[k + RADIX2_PREFETCH_DISTANCE], _MM_HINT_T0);
                    _mm_prefetch((const char *)&sub_outputs[k + RADIX2_PREFETCH_DISTANCE + half], _MM_HINT_T0);
                    _mm_prefetch((const char *)&stage_tw[k + RADIX2_PREFETCH_DISTANCE], _MM_HINT_T0);
                }

                next_e0 = load4_aos(&sub_outputs[k + 16]);
                next_e1 = load4_aos(&sub_outputs[k + 20]);
                next_e2 = load4_aos(&sub_outputs[k + 24]);
                next_e3 = load4_aos(&sub_outputs[k + 28]);

                next_o0 = load4_aos(&sub_outputs[k + 16 + half]);
                next_o1 = load4_aos(&sub_outputs[k + 20 + half]);
                next_o2 = load4_aos(&sub_outputs[k + 24 + half]);
                next_o3 = load4_aos(&sub_outputs[k + 28 + half]);

                next_w0 = load4_aos(&stage_tw[k + 16]);
                next_w1 = load4_aos(&stage_tw[k + 20]);
                next_w2 = load4_aos(&stage_tw[k + 24]);
                next_w3 = load4_aos(&stage_tw[k + 28]);
            }

            // Compute
            __m512d tw0 = cmul_avx512_aos(o0, w0);
            __m512d tw1 = cmul_avx512_aos(o1, w1);
            __m512d tw2 = cmul_avx512_aos(o2, w2);
            __m512d tw3 = cmul_avx512_aos(o3, w3);

            __m512d x00 = _mm512_add_pd(e0, tw0);
            __m512d x10 = _mm512_sub_pd(e0, tw0);
            __m512d x01 = _mm512_add_pd(e1, tw1);
            __m512d x11 = _mm512_sub_pd(e1, tw1);
            __m512d x02 = _mm512_add_pd(e2, tw2);
            __m512d x12 = _mm512_sub_pd(e2, tw2);
            __m512d x03 = _mm512_add_pd(e3, tw3);
            __m512d x13 = _mm512_sub_pd(e3, tw3);

            // Store (non-temporal for large FFTs)
            if (half >= 4096)
            {
                _mm512_stream_pd((double *)&output_buffer[k + 0].re, x00);
                _mm512_stream_pd((double *)&output_buffer[k + 4].re, x01);
                _mm512_stream_pd((double *)&output_buffer[k + 8].re, x02);
                _mm512_stream_pd((double *)&output_buffer[k + 12].re, x03);
                _mm512_stream_pd((double *)&output_buffer[k + 0 + half].re, x10);
                _mm512_stream_pd((double *)&output_buffer[k + 4 + half].re, x11);
                _mm512_stream_pd((double *)&output_buffer[k + 8 + half].re, x12);
                _mm512_stream_pd((double *)&output_buffer[k + 12 + half].re, x13);
            }
            else
            {
                STOREU_PD512(&output_buffer[k + 0].re, x00);
                STOREU_PD512(&output_buffer[k + 4].re, x01);
                STOREU_PD512(&output_buffer[k + 8].re, x02);
                STOREU_PD512(&output_buffer[k + 12].re, x03);
                STOREU_PD512(&output_buffer[k + 0 + half].re, x10);
                STOREU_PD512(&output_buffer[k + 4 + half].re, x11);
                STOREU_PD512(&output_buffer[k + 8 + half].re, x12);
                STOREU_PD512(&output_buffer[k + 12 + half].re, x13);
            }
        }
    }

    // Cleanup
    for (; k + 15 < range1_end; k += 16)
    {
        if (k + 64 < range1_end)
        {
            _mm_prefetch((const char *)&sub_outputs[k + 64], _MM_HINT_T1);
            _mm_prefetch((const char *)&sub_outputs[k + 64 + half], _MM_HINT_T1);
        }
        if (k + 32 < range1_end)
        {
            _mm_prefetch((const char *)&sub_outputs[k + 32], _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_outputs[k + 32 + half], _MM_HINT_T0);
            _mm_prefetch((const char *)&stage_tw[k + 32], _MM_HINT_T0);
        }

        __m512d e0 = load4_aos(&sub_outputs[k + 0]);
        __m512d e1 = load4_aos(&sub_outputs[k + 4]);
        __m512d e2 = load4_aos(&sub_outputs[k + 8]);
        __m512d e3 = load4_aos(&sub_outputs[k + 12]);

        __m512d o0 = load4_aos(&sub_outputs[k + 0 + half]);
        __m512d o1 = load4_aos(&sub_outputs[k + 4 + half]);
        __m512d o2 = load4_aos(&sub_outputs[k + 8 + half]);
        __m512d o3 = load4_aos(&sub_outputs[k + 12 + half]);

        __m512d w0 = load4_aos(&stage_tw[k + 0]);
        __m512d w1 = load4_aos(&stage_tw[k + 4]);
        __m512d w2 = load4_aos(&stage_tw[k + 8]);
        __m512d w3 = load4_aos(&stage_tw[k + 12]);

        __m512d tw0 = cmul_avx512_aos(o0, w0);
        __m512d tw1 = cmul_avx512_aos(o1, w1);
        __m512d tw2 = cmul_avx512_aos(o2, w2);
        __m512d tw3 = cmul_avx512_aos(o3, w3);

        __m512d x00 = _mm512_add_pd(e0, tw0);
        __m512d x10 = _mm512_sub_pd(e0, tw0);
        __m512d x01 = _mm512_add_pd(e1, tw1);
        __m512d x11 = _mm512_sub_pd(e1, tw1);
        __m512d x02 = _mm512_add_pd(e2, tw2);
        __m512d x12 = _mm512_sub_pd(e2, tw2);
        __m512d x03 = _mm512_add_pd(e3, tw3);
        __m512d x13 = _mm512_sub_pd(e3, tw3);

        STOREU_PD512(&output_buffer[k + 0].re, x00);
        STOREU_PD512(&output_buffer[k + 4].re, x01);
        STOREU_PD512(&output_buffer[k + 8].re, x02);
        STOREU_PD512(&output_buffer[k + 12].re, x03);
        STOREU_PD512(&output_buffer[k + 0 + half].re, x10);
        STOREU_PD512(&output_buffer[k + 4 + half].re, x11);
        STOREU_PD512(&output_buffer[k + 8 + half].re, x12);
        STOREU_PD512(&output_buffer[k + 12 + half].re, x13);
    }
#endif // HAS_AVX512

#ifdef __AVX2__
    //======================================================================
    // AVX2 PATH WITH OPTIONAL PARALLELIZATION
    //======================================================================

#if USE_OPENMP
    if (use_parallel && range1_end - k >= 16)
    {
        const int num_blocks = (range1_end - k) / 16;

#pragma omp parallel for schedule(static)
        for (int block_idx = 0; block_idx < num_blocks; block_idx++)
        {
            int kb = k + block_idx * 16;

            for (int kk = kb; kk < kb + 16 && kk + 7 < range1_end; kk += 8)
            {
                if (kk + 32 < range1_end)
                {
                    _mm_prefetch((const char *)&sub_outputs[kk + 32], _MM_HINT_T0);
                    _mm_prefetch((const char *)&sub_outputs[kk + 32 + half], _MM_HINT_T0);
                    _mm_prefetch((const char *)&stage_tw[kk + 32], _MM_HINT_T0);
                }

                __m256d e0 = load2_aos(&sub_outputs[kk + 0], &sub_outputs[kk + 1]);
                __m256d e1 = load2_aos(&sub_outputs[kk + 2], &sub_outputs[kk + 3]);
                __m256d e2 = load2_aos(&sub_outputs[kk + 4], &sub_outputs[kk + 5]);
                __m256d e3 = load2_aos(&sub_outputs[kk + 6], &sub_outputs[kk + 7]);

                __m256d o0 = load2_aos(&sub_outputs[kk + 0 + half], &sub_outputs[kk + 1 + half]);
                __m256d o1 = load2_aos(&sub_outputs[kk + 2 + half], &sub_outputs[kk + 3 + half]);
                __m256d o2 = load2_aos(&sub_outputs[kk + 4 + half], &sub_outputs[kk + 5 + half]);
                __m256d o3 = load2_aos(&sub_outputs[kk + 6 + half], &sub_outputs[kk + 7 + half]);

                __m256d w0 = load2_aos(&stage_tw[kk + 0], &stage_tw[kk + 1]);
                __m256d w1 = load2_aos(&stage_tw[kk + 2], &stage_tw[kk + 3]);
                __m256d w2 = load2_aos(&stage_tw[kk + 4], &stage_tw[kk + 5]);
                __m256d w3 = load2_aos(&stage_tw[kk + 6], &stage_tw[kk + 7]);

                __m256d tw0 = cmul_avx2_aos(o0, w0);
                __m256d tw1 = cmul_avx2_aos(o1, w1);
                __m256d tw2 = cmul_avx2_aos(o2, w2);
                __m256d tw3 = cmul_avx2_aos(o3, w3);

                __m256d x00 = _mm256_add_pd(e0, tw0);
                __m256d x10 = _mm256_sub_pd(e0, tw0);
                __m256d x01 = _mm256_add_pd(e1, tw1);
                __m256d x11 = _mm256_sub_pd(e1, tw1);
                __m256d x02 = _mm256_add_pd(e2, tw2);
                __m256d x12 = _mm256_sub_pd(e2, tw2);
                __m256d x03 = _mm256_add_pd(e3, tw3);
                __m256d x13 = _mm256_sub_pd(e3, tw3);

                STOREU_PD(&output_buffer[kk + 0].re, x00);
                STOREU_PD(&output_buffer[kk + 2].re, x01);
                STOREU_PD(&output_buffer[kk + 4].re, x02);
                STOREU_PD(&output_buffer[kk + 6].re, x03);
                STOREU_PD(&output_buffer[kk + 0 + half].re, x10);
                STOREU_PD(&output_buffer[kk + 2 + half].re, x11);
                STOREU_PD(&output_buffer[kk + 4 + half].re, x12);
                STOREU_PD(&output_buffer[kk + 6 + half].re, x13);
            }
        }
        k += num_blocks * 16; // FIXED: Update k after parallel section
    }
#endif

    // Serial AVX2
    for (; k + 7 < range1_end; k += 8)
    {
        if (k + 64 < range1_end)
        {
            _mm_prefetch((const char *)&sub_outputs[k + 64], _MM_HINT_T1);
        }
        if (k + 16 < range1_end)
        {
            _mm_prefetch((const char *)&sub_outputs[k + 16], _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_outputs[k + 16 + half], _MM_HINT_T0);
            _mm_prefetch((const char *)&stage_tw[k + 16], _MM_HINT_T0);
        }

        __m256d e0 = load2_aos(&sub_outputs[k + 0], &sub_outputs[k + 1]);
        __m256d e1 = load2_aos(&sub_outputs[k + 2], &sub_outputs[k + 3]);
        __m256d e2 = load2_aos(&sub_outputs[k + 4], &sub_outputs[k + 5]);
        __m256d e3 = load2_aos(&sub_outputs[k + 6], &sub_outputs[k + 7]);

        __m256d o0 = load2_aos(&sub_outputs[k + 0 + half], &sub_outputs[k + 1 + half]);
        __m256d o1 = load2_aos(&sub_outputs[k + 2 + half], &sub_outputs[k + 3 + half]);
        __m256d o2 = load2_aos(&sub_outputs[k + 4 + half], &sub_outputs[k + 5 + half]);
        __m256d o3 = load2_aos(&sub_outputs[k + 6 + half], &sub_outputs[k + 7 + half]);

        __m256d w0 = load2_aos(&stage_tw[k + 0], &stage_tw[k + 1]);
        __m256d w1 = load2_aos(&stage_tw[k + 2], &stage_tw[k + 3]);
        __m256d w2 = load2_aos(&stage_tw[k + 4], &stage_tw[k + 5]);
        __m256d w3 = load2_aos(&stage_tw[k + 6], &stage_tw[k + 7]);

        __m256d tw0 = cmul_avx2_aos(o0, w0);
        __m256d tw1 = cmul_avx2_aos(o1, w1);
        __m256d tw2 = cmul_avx2_aos(o2, w2);
        __m256d tw3 = cmul_avx2_aos(o3, w3);

        __m256d x00 = _mm256_add_pd(e0, tw0);
        __m256d x10 = _mm256_sub_pd(e0, tw0);
        __m256d x01 = _mm256_add_pd(e1, tw1);
        __m256d x11 = _mm256_sub_pd(e1, tw1);
        __m256d x02 = _mm256_add_pd(e2, tw2);
        __m256d x12 = _mm256_sub_pd(e2, tw2);
        __m256d x03 = _mm256_add_pd(e3, tw3);
        __m256d x13 = _mm256_sub_pd(e3, tw3);

        STOREU_PD(&output_buffer[k + 0].re, x00);
        STOREU_PD(&output_buffer[k + 2].re, x01);
        STOREU_PD(&output_buffer[k + 4].re, x02);
        STOREU_PD(&output_buffer[k + 6].re, x03);
        STOREU_PD(&output_buffer[k + 0 + half].re, x10);
        STOREU_PD(&output_buffer[k + 2 + half].re, x11);
        STOREU_PD(&output_buffer[k + 4 + half].re, x12);
        STOREU_PD(&output_buffer[k + 6 + half].re, x13);
    }

    for (; k + 1 < range1_end; k += 2)
    {
        __m256d even = load2_aos(&sub_outputs[k], &sub_outputs[k + 1]);
        __m256d odd = load2_aos(&sub_outputs[k + half], &sub_outputs[k + half + 1]);
        __m256d w = load2_aos(&stage_tw[k], &stage_tw[k + 1]);

        __m256d tw = cmul_avx2_aos(odd, w);
        __m256d x0 = _mm256_add_pd(even, tw);
        __m256d x1 = _mm256_sub_pd(even, tw);

        STOREU_PD(&output_buffer[k].re, x0);
        STOREU_PD(&output_buffer[k + half].re, x1);
    }
#endif // __AVX2__

    //======================================================================
    // SSE2 TAIL
    //======================================================================
    for (; k < range1_end; ++k)
    {
        __m128d even = LOADU_SSE2(&sub_outputs[k].re);
        __m128d odd = LOADU_SSE2(&sub_outputs[k + half].re);
        __m128d w = LOADU_SSE2(&stage_tw[k].re);
        __m128d tw = cmul_sse2_aos(odd, w);
        STOREU_SSE2(&output_buffer[k].re, _mm_add_pd(even, tw));
        STOREU_SSE2(&output_buffer[k + half].re, _mm_sub_pd(even, tw));
    }

    //======================================================================
    // SECOND RANGE: (k_quarter, half)
    //======================================================================
    if (k_quarter)
    {
        k = k_quarter + 1;

#ifdef HAS_AVX512
        for (; k + 15 < half; k += 16)
        {
            if (k + 32 < half)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 32], _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 32 + half], _MM_HINT_T0);
                _mm_prefetch((const char *)&stage_tw[k + 32], _MM_HINT_T0);
            }

            __m512d e0 = load4_aos(&sub_outputs[k + 0]);
            __m512d e1 = load4_aos(&sub_outputs[k + 4]);
            __m512d e2 = load4_aos(&sub_outputs[k + 8]);
            __m512d e3 = load4_aos(&sub_outputs[k + 12]);

            __m512d o0 = load4_aos(&sub_outputs[k + 0 + half]);
            __m512d o1 = load4_aos(&sub_outputs[k + 4 + half]);
            __m512d o2 = load4_aos(&sub_outputs[k + 8 + half]);
            __m512d o3 = load4_aos(&sub_outputs[k + 12 + half]);

            __m512d w0 = load4_aos(&stage_tw[k + 0]);
            __m512d w1 = load4_aos(&stage_tw[k + 4]);
            __m512d w2 = load4_aos(&stage_tw[k + 8]);
            __m512d w3 = load4_aos(&stage_tw[k + 12]);

            __m512d tw0 = cmul_avx512_aos(o0, w0);
            __m512d tw1 = cmul_avx512_aos(o1, w1);
            __m512d tw2 = cmul_avx512_aos(o2, w2);
            __m512d tw3 = cmul_avx512_aos(o3, w3);

            __m512d x00 = _mm512_add_pd(e0, tw0);
            __m512d x10 = _mm512_sub_pd(e0, tw0);
            __m512d x01 = _mm512_add_pd(e1, tw1);
            __m512d x11 = _mm512_sub_pd(e1, tw1);
            __m512d x02 = _mm512_add_pd(e2, tw2);
            __m512d x12 = _mm512_sub_pd(e2, tw2);
            __m512d x03 = _mm512_add_pd(e3, tw3);
            __m512d x13 = _mm512_sub_pd(e3, tw3);

            STOREU_PD512(&output_buffer[k + 0].re, x00);
            STOREU_PD512(&output_buffer[k + 4].re, x01);
            STOREU_PD512(&output_buffer[k + 8].re, x02);
            STOREU_PD512(&output_buffer[k + 12].re, x03);
            STOREU_PD512(&output_buffer[k + 0 + half].re, x10);
            STOREU_PD512(&output_buffer[k + 4 + half].re, x11);
            STOREU_PD512(&output_buffer[k + 8 + half].re, x12);
            STOREU_PD512(&output_buffer[k + 12 + half].re, x13);
        }
#endif

#ifdef __AVX2__
        for (; k + 7 < half; k += 8)
        {
            if (k + 16 < half)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 16], _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 16 + half], _MM_HINT_T0);
                _mm_prefetch((const char *)&stage_tw[k + 16], _MM_HINT_T0);
            }

            __m256d e0 = load2_aos(&sub_outputs[k + 0], &sub_outputs[k + 1]);
            __m256d e1 = load2_aos(&sub_outputs[k + 2], &sub_outputs[k + 3]);
            __m256d e2 = load2_aos(&sub_outputs[k + 4], &sub_outputs[k + 5]);
            __m256d e3 = load2_aos(&sub_outputs[k + 6], &sub_outputs[k + 7]);

            __m256d o0 = load2_aos(&sub_outputs[k + 0 + half], &sub_outputs[k + 1 + half]);
            __m256d o1 = load2_aos(&sub_outputs[k + 2 + half], &sub_outputs[k + 3 + half]);
            __m256d o2 = load2_aos(&sub_outputs[k + 4 + half], &sub_outputs[k + 5 + half]);
            __m256d o3 = load2_aos(&sub_outputs[k + 6 + half], &sub_outputs[k + 7 + half]);

            __m256d w0 = load2_aos(&stage_tw[k + 0], &stage_tw[k + 1]);
            __m256d w1 = load2_aos(&stage_tw[k + 2], &stage_tw[k + 3]);
            __m256d w2 = load2_aos(&stage_tw[k + 4], &stage_tw[k + 5]);
            __m256d w3 = load2_aos(&stage_tw[k + 6], &stage_tw[k + 7]);

            __m256d tw0 = cmul_avx2_aos(o0, w0);
            __m256d tw1 = cmul_avx2_aos(o1, w1);
            __m256d tw2 = cmul_avx2_aos(o2, w2);
            __m256d tw3 = cmul_avx2_aos(o3, w3);

            __m256d x00 = _mm256_add_pd(e0, tw0);
            __m256d x10 = _mm256_sub_pd(e0, tw0);
            __m256d x01 = _mm256_add_pd(e1, tw1);
            __m256d x11 = _mm256_sub_pd(e1, tw1);
            __m256d x02 = _mm256_add_pd(e2, tw2);
            __m256d x12 = _mm256_sub_pd(e2, tw2);
            __m256d x03 = _mm256_add_pd(e3, tw3);
            __m256d x13 = _mm256_sub_pd(e3, tw3);

            STOREU_PD(&output_buffer[k + 0].re, x00);
            STOREU_PD(&output_buffer[k + 2].re, x01);
            STOREU_PD(&output_buffer[k + 4].re, x02);
            STOREU_PD(&output_buffer[k + 6].re, x03);
            STOREU_PD(&output_buffer[k + 0 + half].re, x10);
            STOREU_PD(&output_buffer[k + 2 + half].re, x11);
            STOREU_PD(&output_buffer[k + 4 + half].re, x12);
            STOREU_PD(&output_buffer[k + 6 + half].re, x13);
        }

        for (; k + 1 < half; k += 2)
        {
            __m256d even = load2_aos(&sub_outputs[k], &sub_outputs[k + 1]);
            __m256d odd = load2_aos(&sub_outputs[k + half], &sub_outputs[k + half + 1]);
            __m256d w = load2_aos(&stage_tw[k], &stage_tw[k + 1]);

            __m256d tw = cmul_avx2_aos(odd, w);
            __m256d x0 = _mm256_add_pd(even, tw);
            __m256d x1 = _mm256_sub_pd(even, tw);

            STOREU_PD(&output_buffer[k].re, x0);
            STOREU_PD(&output_buffer[k + half].re, x1);
        }
#endif

        for (; k < half; ++k)
        {
            __m128d even = LOADU_SSE2(&sub_outputs[k].re);
            __m128d odd = LOADU_SSE2(&sub_outputs[k + half].re);
            __m128d w = LOADU_SSE2(&stage_tw[k].re);
            __m128d tw = cmul_sse2_aos(odd, w);
            STOREU_SSE2(&output_buffer[k].re, _mm_add_pd(even, tw));
            STOREU_SSE2(&output_buffer[k + half].re, _mm_sub_pd(even, tw));
        }
    }

    // Memory fence for non-temporal stores
    if (half >= 4096)
    {
        _mm_sfence();
    }
}