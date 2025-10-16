#include "fft_radix4.h" // ✅ Gets highSpeedFFT.h → fft_types.h
#include "simd_math.h"  // ✅ Gets complex math operations

void fft_radix4_butterfly(
    fft_data *output_buffer,
    fft_data *sub_outputs,
    const fft_data *stage_tw,
    int sub_len,
    int transform_sign)
{
    //======================================================================
    // RADIX-4 BUTTERFLY (DIT) - FFTW-STYLE OPTIMIZED
    //
    // Pure AoS, no conversions, heavy unrolling for maximum performance.
    //======================================================================

    const int quarter = sub_len;
    int k = 0;

#ifdef HAS_AVX512
    //------------------------------------------------------------------
    // AVX-512 PATH: 16x unrolling (4 registers × 4 complex = 16 butterflies)
    //------------------------------------------------------------------

    // Precompute rotation masks - FIXED
    const __m512d rot_mask_512 = (transform_sign == 1)
        ? _mm512_castsi512_pd(_mm512_set_epi64(
            0x0000000000000000, 0x8000000000000000,  // -0.0, 0.0 (lane 3)
            0x0000000000000000, 0x8000000000000000,  // -0.0, 0.0 (lane 2)
            0x0000000000000000, 0x8000000000000000,  // -0.0, 0.0 (lane 1)
            0x0000000000000000, 0x8000000000000000)) // -0.0, 0.0 (lane 0)
        : _mm512_castsi512_pd(_mm512_set_epi64(
            0x8000000000000000, 0x0000000000000000,  // 0.0, -0.0 (lane 3)
            0x8000000000000000, 0x0000000000000000,  // 0.0, -0.0 (lane 2)
            0x8000000000000000, 0x0000000000000000,  // 0.0, -0.0 (lane 1)
            0x8000000000000000, 0x0000000000000000)); // 0.0, -0.0 (lane 0)

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

    for (; k + 15 < quarter; k += 16)
    {
        // Prefetch ahead
        if (k + 32 < quarter)
        {
            _mm_prefetch((const char *)&sub_outputs[k + 32].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&stage_tw[3 * (k + 32)].re, _MM_HINT_T0);
        }

        //==================================================================
        // Load inputs (16 butterflies = 4 loads per lane × 4 lanes)
        //==================================================================
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

        //==================================================================
        // Load twiddles W^k, W^{2k}, W^{3k} (k-major: 3 per butterfly)
        //==================================================================
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

        //==================================================================
        // Twiddle multiply
        //==================================================================
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

        //==================================================================
        // Store results
        //==================================================================
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

    //==========================================================================
    // Cleanup: 8x unrolling (process 8 butterflies at once)
    //==========================================================================
    for (; k + 7 < quarter; k += 8)
    {
        // Load inputs (8 butterflies = 2 AVX-512 registers per lane)
        __m512d a0 = load4_aos(&sub_outputs[k + 0]);
        __m512d a1 = load4_aos(&sub_outputs[k + 4]);

        __m512d b0 = load4_aos(&sub_outputs[k + 0 + quarter]);
        __m512d b1 = load4_aos(&sub_outputs[k + 4 + quarter]);

        __m512d c0 = load4_aos(&sub_outputs[k + 0 + 2 * quarter]);
        __m512d c1 = load4_aos(&sub_outputs[k + 4 + 2 * quarter]);

        __m512d d0 = load4_aos(&sub_outputs[k + 0 + 3 * quarter]);
        __m512d d1 = load4_aos(&sub_outputs[k + 4 + 3 * quarter]);

        // Load twiddles
        __m512d w1_0 = load4_aos(&stage_tw[3 * (k + 0)]);
        __m512d w1_1 = load4_aos(&stage_tw[3 * (k + 4)]);

        __m512d w2_0 = load4_aos(&stage_tw[3 * (k + 0) + 1]);
        __m512d w2_1 = load4_aos(&stage_tw[3 * (k + 4) + 1]);

        __m512d w3_0 = load4_aos(&stage_tw[3 * (k + 0) + 2]);
        __m512d w3_1 = load4_aos(&stage_tw[3 * (k + 4) + 2]);

        // Twiddle multiply
        __m512d b2_0 = cmul_avx512_aos(b0, w1_0);
        __m512d b2_1 = cmul_avx512_aos(b1, w1_1);

        __m512d c2_0 = cmul_avx512_aos(c0, w2_0);
        __m512d c2_1 = cmul_avx512_aos(c1, w2_1);

        __m512d d2_0 = cmul_avx512_aos(d0, w3_0);
        __m512d d2_1 = cmul_avx512_aos(d1, w3_1);

        // Radix-4 butterfly (using the macro you defined earlier)
        __m512d y0_0, y1_0, y2_0, y3_0;
        __m512d y0_1, y1_1, y2_1, y3_1;

        RADIX4_BUTTERFLY_AVX512(a0, b2_0, c2_0, d2_0, y0_0, y1_0, y2_0, y3_0);
        RADIX4_BUTTERFLY_AVX512(a1, b2_1, c2_1, d2_1, y0_1, y1_1, y2_1, y3_1);

        // Store results
        STOREU_PD512(&output_buffer[k + 0].re, y0_0);
        STOREU_PD512(&output_buffer[k + 4].re, y0_1);

        STOREU_PD512(&output_buffer[k + 0 + quarter].re, y1_0);
        STOREU_PD512(&output_buffer[k + 4 + quarter].re, y1_1);

        STOREU_PD512(&output_buffer[k + 0 + 2 * quarter].re, y2_0);
        STOREU_PD512(&output_buffer[k + 4 + 2 * quarter].re, y2_1);

        STOREU_PD512(&output_buffer[k + 0 + 3 * quarter].re, y3_0);
        STOREU_PD512(&output_buffer[k + 4 + 3 * quarter].re, y3_1);
    }

    //==========================================================================
    // Cleanup: 4x unrolling (process 4 butterflies at once)
    //==========================================================================
    for (; k + 3 < quarter; k += 4)
    {
        // Load inputs (4 butterflies = 1 AVX-512 register per lane)
        __m512d a = load4_aos(&sub_outputs[k]);
        __m512d b = load4_aos(&sub_outputs[k + quarter]);
        __m512d c = load4_aos(&sub_outputs[k + 2 * quarter]);
        __m512d d = load4_aos(&sub_outputs[k + 3 * quarter]);

        // Load twiddles
        __m512d w1 = load4_aos(&stage_tw[3 * k]);
        __m512d w2 = load4_aos(&stage_tw[3 * k + 1]);
        __m512d w3 = load4_aos(&stage_tw[3 * k + 2]);

        // Twiddle multiply
        __m512d b2 = cmul_avx512_aos(b, w1);
        __m512d c2 = cmul_avx512_aos(c, w2);
        __m512d d2 = cmul_avx512_aos(d, w3);

        // Radix-4 butterfly
        __m512d y0, y1, y2, y3;
        RADIX4_BUTTERFLY_AVX512(a, b2, c2, d2, y0, y1, y2, y3);

        // Store results
        STOREU_PD512(&output_buffer[k].re, y0);
        STOREU_PD512(&output_buffer[k + quarter].re, y1);
        STOREU_PD512(&output_buffer[k + 2 * quarter].re, y2);
        STOREU_PD512(&output_buffer[k + 3 * quarter].re, y3);
    }

#undef RADIX4_BUTTERFLY_AVX512
#endif // HAS_AVX512

#ifdef __AVX2__
    //------------------------------------------------------------------
    // AVX2 PATH: 8x unrolled, pure AoS
    //------------------------------------------------------------------

    // Precompute rotation masks
    const __m256d rot_mask = (transform_sign == 1)
                             ? _mm256_set_pd(-0.0, 0.0, -0.0, 0.0)  // This is actually +i
                             : _mm256_set_pd(0.0, -0.0, 0.0, -0.0); // This is actually -i

    // DEFINE AVX2 MACRO
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

    for (; k + 7 < quarter; k += 8)
    {
        // Prefetch ahead
        if (k + 16 < quarter)
        {
            _mm_prefetch((const char *)&sub_outputs[k + 16].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_outputs[k + 16 + quarter].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_outputs[k + 16 + 2 * quarter].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_outputs[k + 16 + 3 * quarter].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&stage_tw[3 * (k + 16)].re, _MM_HINT_T0);
        }

        //==================================================================
        // Load inputs (8 butterflies = 4 AVX2 loads per lane)
        //==================================================================
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

        //==================================================================
        // Load twiddles W^k, W^{2k}, W^{3k} (k-major: 3 per butterfly)
        //==================================================================
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

        //==================================================================
        // Twiddle multiply
        //==================================================================
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

        //==================================================================
        // Store results (pure AoS, no conversions!)
        //==================================================================
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

    //------------------------------------------------------------------
    // Cleanup: 2x unrolling
    //------------------------------------------------------------------
    const __m256d rot_mask_final = (transform_sign == 1)
                             ? _mm256_set_pd(-0.0, 0.0, -0.0, 0.0)  // This is actually +i
                             : _mm256_set_pd(0.0, -0.0, 0.0, -0.0); // This is actually -i

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

    //------------------------------------------------------------------
    // SSE2 TAIL: Handle remaining 0..1 elements
    //------------------------------------------------------------------
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
                          ? _mm_xor_pd(swp, _mm_set_pd(0.0, -0.0))  // +i (was -0.0, 0.0)
                          : _mm_xor_pd(swp, _mm_set_pd(-0.0, 0.0)); // -i (was 0.0, -0.0)

        __m128d y1 = _mm_sub_pd(a_mc, rot);
        __m128d y3 = _mm_add_pd(a_mc, rot);

        STOREU_SSE2(&output_buffer[k].re, y0);
        STOREU_SSE2(&output_buffer[k + quarter].re, y1);
        STOREU_SSE2(&output_buffer[k + 2 * quarter].re, y2);
        STOREU_SSE2(&output_buffer[k + 3 * quarter].re, y3);
    }
}