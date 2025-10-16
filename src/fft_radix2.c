#include "fft_radix2.h"   // ✅ Gets highSpeedFFT.h → fft_types.h
#include "simd_math.h"    // ✅ Gets complex math operations


void fft_radix2_butterfly(
    fft_data *output_buffer,
    fft_data *sub_outputs,
    const fft_data *stage_tw,
    int sub_len,
    int transform_sign)
{
    const int half = sub_len; // K = N/2

    //======================================================================
    // STAGE 0: SPECIAL CASE k=0 (W^0 = 1, NO TWIDDLE)
    //======================================================================
    {
        fft_data even_0 = sub_outputs[0];           // X[0*k]
        fft_data odd_0 = sub_outputs[half];         // X[1*k]
        output_buffer[0].re = even_0.re + odd_0.re; // Y[0*k] = X[0*k] + X[1*k]
        output_buffer[0].im = even_0.im + odd_0.im;
        output_buffer[half].re = even_0.re - odd_0.re; // Y[1*k] = X[0*k] - X[1*k]
        output_buffer[half].im = even_0.im - odd_0.im;
    }

    //======================================================================
    // STAGE 1: SPECIAL CASE k=N/4 (W^(N/4) = ±i, 90° ROTATION)
    // Only if N divisible by 4
    //======================================================================
    int k_quarter = 0;
    if ((half & 1) == 0) // N/4 exists
    {
        k_quarter = half >> 1; // k = N/4

        fft_data even_q = sub_outputs[k_quarter];       // X[0*(N/4)]
        fft_data odd_q = sub_outputs[half + k_quarter]; // X[1*(N/4)]

        // Rotate odd by ±90°: odd * (±i) = ∓im + i*re
        double rotated_re = transform_sign > 0 ? odd_q.im : -odd_q.im;
        double rotated_im = transform_sign > 0 ? -odd_q.re : odd_q.re;

        output_buffer[k_quarter].re = even_q.re + rotated_re; // Y[0*(N/4)]
        output_buffer[k_quarter].im = even_q.im + rotated_im;
        output_buffer[half + k_quarter].re = even_q.re - rotated_re; // Y[1*(N/4)]
        output_buffer[half + k_quarter].im = even_q.im - rotated_im;
    }

    //======================================================================
    // STAGE 2: GENERAL CASE k=1...N/2-1 (TWIDDLE MULTIPLY REQUIRED)
    // Split into 2 ranges to skip k=N/4 if it exists
    //======================================================================
    int k = 1;                                     // Start after k=0
    int range1_end = k_quarter ? k_quarter : half; // Range 1: [1, N/4)

#ifdef HAS_AVX512
                                                   //======================================================================
    // AVX-512: First range [1, k_quarter) or [1, half) if no k_quarter
    //======================================================================
    for (; k + 15 < range1_end; k += 16)
    {
        // Prefetch
        if (k + 32 < range1_end)
        {
            _mm_prefetch((const char *)&sub_outputs[k + 32], _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_outputs[k + 32 + half], _MM_HINT_T0);
            _mm_prefetch((const char *)&stage_tw[k + 32], _MM_HINT_T0);
        }

        // Load even/odd samples
        __m512d e0 = load4_aos(&sub_outputs[k + 0]);
        __m512d e1 = load4_aos(&sub_outputs[k + 4]);
        __m512d e2 = load4_aos(&sub_outputs[k + 8]);
        __m512d e3 = load4_aos(&sub_outputs[k + 12]);

        __m512d o0 = load4_aos(&sub_outputs[k + 0 + half]);
        __m512d o1 = load4_aos(&sub_outputs[k + 4 + half]);
        __m512d o2 = load4_aos(&sub_outputs[k + 8 + half]);
        __m512d o3 = load4_aos(&sub_outputs[k + 12 + half]);

        // Load twiddles - FIX: stage_tw[k] contains W^k
        __m512d w0 = load4_aos(&stage_tw[k + 0]);
        __m512d w1 = load4_aos(&stage_tw[k + 4]);
        __m512d w2 = load4_aos(&stage_tw[k + 8]);
        __m512d w3 = load4_aos(&stage_tw[k + 12]);

        // Twiddle multiply
        __m512d tw0 = cmul_avx512_aos(o0, w0);
        __m512d tw1 = cmul_avx512_aos(o1, w1);
        __m512d tw2 = cmul_avx512_aos(o2, w2);
        __m512d tw3 = cmul_avx512_aos(o3, w3);

        // Butterfly
        __m512d x00 = _mm512_add_pd(e0, tw0);
        __m512d x10 = _mm512_sub_pd(e0, tw0);
        __m512d x01 = _mm512_add_pd(e1, tw1);
        __m512d x11 = _mm512_sub_pd(e1, tw1);
        __m512d x02 = _mm512_add_pd(e2, tw2);
        __m512d x12 = _mm512_sub_pd(e2, tw2);
        __m512d x03 = _mm512_add_pd(e3, tw3);
        __m512d x13 = _mm512_sub_pd(e3, tw3);

        // Store
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
    // AVX2: First range [k, range1_end)
    //======================================================================
    for (; k + 7 < range1_end; k += 8)
    {
        // Prefetch
        if (k + 16 < range1_end)
        {
            _mm_prefetch((const char *)&sub_outputs[k + 16], _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_outputs[k + 16 + half], _MM_HINT_T0);
            _mm_prefetch((const char *)&stage_tw[k + 16], _MM_HINT_T0);
        }

        // Load 8 even pairs
        __m256d e0 = load2_aos(&sub_outputs[k + 0], &sub_outputs[k + 1]);
        __m256d e1 = load2_aos(&sub_outputs[k + 2], &sub_outputs[k + 3]);
        __m256d e2 = load2_aos(&sub_outputs[k + 4], &sub_outputs[k + 5]);
        __m256d e3 = load2_aos(&sub_outputs[k + 6], &sub_outputs[k + 7]);

        // Load 8 odd pairs
        __m256d o0 = load2_aos(&sub_outputs[k + 0 + half], &sub_outputs[k + 1 + half]);
        __m256d o1 = load2_aos(&sub_outputs[k + 2 + half], &sub_outputs[k + 3 + half]);
        __m256d o2 = load2_aos(&sub_outputs[k + 4 + half], &sub_outputs[k + 5 + half]);
        __m256d o3 = load2_aos(&sub_outputs[k + 6 + half], &sub_outputs[k + 7 + half]);

        // Load twiddles - FIX: stage_tw[k] contains W^k
        __m256d w0 = load2_aos(&stage_tw[k + 0], &stage_tw[k + 1]);
        __m256d w1 = load2_aos(&stage_tw[k + 2], &stage_tw[k + 3]);
        __m256d w2 = load2_aos(&stage_tw[k + 4], &stage_tw[k + 5]);
        __m256d w3 = load2_aos(&stage_tw[k + 6], &stage_tw[k + 7]);

        // Twiddle multiply
        __m256d tw0 = cmul_avx2_aos(o0, w0);
        __m256d tw1 = cmul_avx2_aos(o1, w1);
        __m256d tw2 = cmul_avx2_aos(o2, w2);
        __m256d tw3 = cmul_avx2_aos(o3, w3);

        // Butterfly
        __m256d x00 = _mm256_add_pd(e0, tw0);
        __m256d x10 = _mm256_sub_pd(e0, tw0);
        __m256d x01 = _mm256_add_pd(e1, tw1);
        __m256d x11 = _mm256_sub_pd(e1, tw1);
        __m256d x02 = _mm256_add_pd(e2, tw2);
        __m256d x12 = _mm256_sub_pd(e2, tw2);
        __m256d x03 = _mm256_add_pd(e3, tw3);
        __m256d x13 = _mm256_sub_pd(e3, tw3);

        // Store results
        STOREU_PD(&output_buffer[k + 0].re, x00);
        STOREU_PD(&output_buffer[k + 2].re, x01);
        STOREU_PD(&output_buffer[k + 4].re, x02);
        STOREU_PD(&output_buffer[k + 6].re, x03);
        STOREU_PD(&output_buffer[k + 0 + half].re, x10);
        STOREU_PD(&output_buffer[k + 2 + half].re, x11);
        STOREU_PD(&output_buffer[k + 4 + half].re, x12);
        STOREU_PD(&output_buffer[k + 6 + half].re, x13);
    }

    // Cleanup: 2x unrolling for first range
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
    // SSE2 TAIL for first range
    //======================================================================
    for (; k < range1_end; ++k)
    {
        __m128d even = LOADU_SSE2(&sub_outputs[k].re);
        __m128d odd = LOADU_SSE2(&sub_outputs[k + half].re);
        __m128d w = LOADU_SSE2(&stage_tw[k].re); // FIX: stage_tw[k] contains W^k
        __m128d tw = cmul_sse2_aos(odd, w);
        STOREU_SSE2(&output_buffer[k].re, _mm_add_pd(even, tw));
        STOREU_SSE2(&output_buffer[k + half].re, _mm_sub_pd(even, tw));
    }

    //======================================================================
    // Second range: (k_quarter, half) if k_quarter exists
    //======================================================================
    if (k_quarter)
    {
        k = k_quarter + 1; // Skip k_quarter since we handled it

#ifdef HAS_AVX512
        // AVX-512 for second range
        for (; k + 15 < half; k += 16)
        {
            // Prefetch
            if (k + 32 < half)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 32], _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 32 + half], _MM_HINT_T0);
                _mm_prefetch((const char *)&stage_tw[k + 32], _MM_HINT_T0);
            }

            // Load even/odd samples
            __m512d e0 = load4_aos(&sub_outputs[k + 0]);
            __m512d e1 = load4_aos(&sub_outputs[k + 4]);
            __m512d e2 = load4_aos(&sub_outputs[k + 8]);
            __m512d e3 = load4_aos(&sub_outputs[k + 12]);

            __m512d o0 = load4_aos(&sub_outputs[k + 0 + half]);
            __m512d o1 = load4_aos(&sub_outputs[k + 4 + half]);
            __m512d o2 = load4_aos(&sub_outputs[k + 8 + half]);
            __m512d o3 = load4_aos(&sub_outputs[k + 12 + half]);

            // Load twiddles
            __m512d w0 = load4_aos(&stage_tw[k + 0]);
            __m512d w1 = load4_aos(&stage_tw[k + 4]);
            __m512d w2 = load4_aos(&stage_tw[k + 8]);
            __m512d w3 = load4_aos(&stage_tw[k + 12]);

            // Twiddle multiply
            __m512d tw0 = cmul_avx512_aos(o0, w0);
            __m512d tw1 = cmul_avx512_aos(o1, w1);
            __m512d tw2 = cmul_avx512_aos(o2, w2);
            __m512d tw3 = cmul_avx512_aos(o3, w3);

            // Butterfly
            __m512d x00 = _mm512_add_pd(e0, tw0);
            __m512d x10 = _mm512_sub_pd(e0, tw0);
            __m512d x01 = _mm512_add_pd(e1, tw1);
            __m512d x11 = _mm512_sub_pd(e1, tw1);
            __m512d x02 = _mm512_add_pd(e2, tw2);
            __m512d x12 = _mm512_sub_pd(e2, tw2);
            __m512d x03 = _mm512_add_pd(e3, tw3);
            __m512d x13 = _mm512_sub_pd(e3, tw3);

            // Store
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
        // AVX2 for second range
        for (; k + 7 < half; k += 8)
        {
            // Prefetch
            if (k + 16 < half)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 16], _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 16 + half], _MM_HINT_T0);
                _mm_prefetch((const char *)&stage_tw[k + 16], _MM_HINT_T0);
            }

            // Load 8 even pairs
            __m256d e0 = load2_aos(&sub_outputs[k + 0], &sub_outputs[k + 1]);
            __m256d e1 = load2_aos(&sub_outputs[k + 2], &sub_outputs[k + 3]);
            __m256d e2 = load2_aos(&sub_outputs[k + 4], &sub_outputs[k + 5]);
            __m256d e3 = load2_aos(&sub_outputs[k + 6], &sub_outputs[k + 7]);

            // Load 8 odd pairs
            __m256d o0 = load2_aos(&sub_outputs[k + 0 + half], &sub_outputs[k + 1 + half]);
            __m256d o1 = load2_aos(&sub_outputs[k + 2 + half], &sub_outputs[k + 3 + half]);
            __m256d o2 = load2_aos(&sub_outputs[k + 4 + half], &sub_outputs[k + 5 + half]);
            __m256d o3 = load2_aos(&sub_outputs[k + 6 + half], &sub_outputs[k + 7 + half]);

            // Load twiddles
            __m256d w0 = load2_aos(&stage_tw[k + 0], &stage_tw[k + 1]);
            __m256d w1 = load2_aos(&stage_tw[k + 2], &stage_tw[k + 3]);
            __m256d w2 = load2_aos(&stage_tw[k + 4], &stage_tw[k + 5]);
            __m256d w3 = load2_aos(&stage_tw[k + 6], &stage_tw[k + 7]);

            // Twiddle multiply
            __m256d tw0 = cmul_avx2_aos(o0, w0);
            __m256d tw1 = cmul_avx2_aos(o1, w1);
            __m256d tw2 = cmul_avx2_aos(o2, w2);
            __m256d tw3 = cmul_avx2_aos(o3, w3);

            // Butterfly
            __m256d x00 = _mm256_add_pd(e0, tw0);
            __m256d x10 = _mm256_sub_pd(e0, tw0);
            __m256d x01 = _mm256_add_pd(e1, tw1);
            __m256d x11 = _mm256_sub_pd(e1, tw1);
            __m256d x02 = _mm256_add_pd(e2, tw2);
            __m256d x12 = _mm256_sub_pd(e2, tw2);
            __m256d x03 = _mm256_add_pd(e3, tw3);
            __m256d x13 = _mm256_sub_pd(e3, tw3);

            // Store results
            STOREU_PD(&output_buffer[k + 0].re, x00);
            STOREU_PD(&output_buffer[k + 2].re, x01);
            STOREU_PD(&output_buffer[k + 4].re, x02);
            STOREU_PD(&output_buffer[k + 6].re, x03);
            STOREU_PD(&output_buffer[k + 0 + half].re, x10);
            STOREU_PD(&output_buffer[k + 2 + half].re, x11);
            STOREU_PD(&output_buffer[k + 4 + half].re, x12);
            STOREU_PD(&output_buffer[k + 6 + half].re, x13);
        }

        // 2x cleanup for second range
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

        // SSE2 tail for second range
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
}