#include "fft_radix5.h" // ✅ Gets highSpeedFFT.h → fft_types.h
#include "simd_math.h"  // ✅ Gets complex math operations

// --- Radix-5 constants ---
static const double C5_1 = 0.30901699437;  // cos(72°)
static const double C5_2 = -0.80901699437; // cos(144°)
static const double S5_1 = 0.95105651629;  // sin(72°)  ← BACK TO POSITIVE
static const double S5_2 = 0.58778525229;  // sin(144°) ← BACK TO POSITIVE


void fft_radix5_butterfly(
    fft_data *output_buffer,
    fft_data *sub_outputs,
    const fft_data *stage_tw,
    int sub_len,
    int transform_sign)
{
    //======================================================================
    // RADIX-5 BUTTERFLY (Rader DIT) - FFTW-STYLE OPTIMIZED
    //
    // Pure AoS, no conversions, 8x unrolling for maximum performance.
    //======================================================================


    const int fifth = sub_len;
    int k = 0;

#ifdef __AVX2__
    //------------------------------------------------------------------
    // AVX2 PATH: 8x unrolled, pure AoS
    //------------------------------------------------------------------
    const __m256d vc1 = _mm256_set1_pd(C5_1); // cos(2π/5)
    const __m256d vc2 = _mm256_set1_pd(C5_2); // cos(4π/5)
    const __m256d vs1 = _mm256_set1_pd(S5_1); // sin(2π/5)
    const __m256d vs2 = _mm256_set1_pd(S5_2); // sin(4π/5)

    // Precompute rotation mask for (-i*sgn) multiplication
    // After permute: [im0, re0, im1, re1]
    // Forward (sgn=+1): multiply by -i → negate lanes 1,3 (imaginary after swap)
    // Inverse (sgn=-1): multiply by +i → negate lanes 0,2 (real after swap)
    const __m256d rot_mask = (transform_sign == 1)
                                 ? _mm256_set_pd(0.0, -0.0, 0.0, -0.0)  // -i
                                 : _mm256_set_pd(-0.0, 0.0, -0.0, 0.0); // +i

    for (; k + 7 < fifth; k += 8)
    {
        // Prefetch ahead
        if (k + 16 < fifth)
        {
            _mm_prefetch((const char *)&sub_outputs[k + 16].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_outputs[k + 16 + fifth].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_outputs[k + 16 + 2 * fifth].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_outputs[k + 16 + 3 * fifth].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_outputs[k + 16 + 4 * fifth].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&stage_tw[4 * (k + 16)].re, _MM_HINT_T0);
        }

        //==================================================================
        // Load inputs (8 butterflies = 4 AVX2 loads per lane)
        //==================================================================
        __m256d a0 = load2_aos(&sub_outputs[k + 0], &sub_outputs[k + 1]);
        __m256d a1 = load2_aos(&sub_outputs[k + 2], &sub_outputs[k + 3]);
        __m256d a2 = load2_aos(&sub_outputs[k + 4], &sub_outputs[k + 5]);
        __m256d a3 = load2_aos(&sub_outputs[k + 6], &sub_outputs[k + 7]);

        __m256d b0 = load2_aos(&sub_outputs[k + 0 + fifth], &sub_outputs[k + 1 + fifth]);
        __m256d b1 = load2_aos(&sub_outputs[k + 2 + fifth], &sub_outputs[k + 3 + fifth]);
        __m256d b2 = load2_aos(&sub_outputs[k + 4 + fifth], &sub_outputs[k + 5 + fifth]);
        __m256d b3 = load2_aos(&sub_outputs[k + 6 + fifth], &sub_outputs[k + 7 + fifth]);

        __m256d c0 = load2_aos(&sub_outputs[k + 0 + 2 * fifth], &sub_outputs[k + 1 + 2 * fifth]);
        __m256d c1 = load2_aos(&sub_outputs[k + 2 + 2 * fifth], &sub_outputs[k + 3 + 2 * fifth]);
        __m256d c2 = load2_aos(&sub_outputs[k + 4 + 2 * fifth], &sub_outputs[k + 5 + 2 * fifth]);
        __m256d c3 = load2_aos(&sub_outputs[k + 6 + 2 * fifth], &sub_outputs[k + 7 + 2 * fifth]);

        __m256d d0 = load2_aos(&sub_outputs[k + 0 + 3 * fifth], &sub_outputs[k + 1 + 3 * fifth]);
        __m256d d1 = load2_aos(&sub_outputs[k + 2 + 3 * fifth], &sub_outputs[k + 3 + 3 * fifth]);
        __m256d d2 = load2_aos(&sub_outputs[k + 4 + 3 * fifth], &sub_outputs[k + 5 + 3 * fifth]);
        __m256d d3 = load2_aos(&sub_outputs[k + 6 + 3 * fifth], &sub_outputs[k + 7 + 3 * fifth]);

        __m256d e0 = load2_aos(&sub_outputs[k + 0 + 4 * fifth], &sub_outputs[k + 1 + 4 * fifth]);
        __m256d e1 = load2_aos(&sub_outputs[k + 2 + 4 * fifth], &sub_outputs[k + 3 + 4 * fifth]);
        __m256d e2 = load2_aos(&sub_outputs[k + 4 + 4 * fifth], &sub_outputs[k + 5 + 4 * fifth]);
        __m256d e3 = load2_aos(&sub_outputs[k + 6 + 4 * fifth], &sub_outputs[k + 7 + 4 * fifth]);

        //==================================================================
        // Load twiddles W^k, W^{2k}, W^{3k}, W^{4k} (k-major: 4 per butterfly)
        //==================================================================
        __m256d w1_0 = load2_aos(&stage_tw[4 * (k + 0)], &stage_tw[4 * (k + 1)]);
        __m256d w1_1 = load2_aos(&stage_tw[4 * (k + 2)], &stage_tw[4 * (k + 3)]);
        __m256d w1_2 = load2_aos(&stage_tw[4 * (k + 4)], &stage_tw[4 * (k + 5)]);
        __m256d w1_3 = load2_aos(&stage_tw[4 * (k + 6)], &stage_tw[4 * (k + 7)]);

        __m256d w2_0 = load2_aos(&stage_tw[4 * (k + 0) + 1], &stage_tw[4 * (k + 1) + 1]);
        __m256d w2_1 = load2_aos(&stage_tw[4 * (k + 2) + 1], &stage_tw[4 * (k + 3) + 1]);
        __m256d w2_2 = load2_aos(&stage_tw[4 * (k + 4) + 1], &stage_tw[4 * (k + 5) + 1]);
        __m256d w2_3 = load2_aos(&stage_tw[4 * (k + 6) + 1], &stage_tw[4 * (k + 7) + 1]);

        __m256d w3_0 = load2_aos(&stage_tw[4 * (k + 0) + 2], &stage_tw[4 * (k + 1) + 2]);
        __m256d w3_1 = load2_aos(&stage_tw[4 * (k + 2) + 2], &stage_tw[4 * (k + 3) + 2]);
        __m256d w3_2 = load2_aos(&stage_tw[4 * (k + 4) + 2], &stage_tw[4 * (k + 5) + 2]);
        __m256d w3_3 = load2_aos(&stage_tw[4 * (k + 6) + 2], &stage_tw[4 * (k + 7) + 2]);

        __m256d w4_0 = load2_aos(&stage_tw[4 * (k + 0) + 3], &stage_tw[4 * (k + 1) + 3]);
        __m256d w4_1 = load2_aos(&stage_tw[4 * (k + 2) + 3], &stage_tw[4 * (k + 3) + 3]);
        __m256d w4_2 = load2_aos(&stage_tw[4 * (k + 4) + 3], &stage_tw[4 * (k + 5) + 3]);
        __m256d w4_3 = load2_aos(&stage_tw[4 * (k + 6) + 3], &stage_tw[4 * (k + 7) + 3]);

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

        __m256d e2_0 = cmul_avx2_aos(e0, w4_0);
        __m256d e2_1 = cmul_avx2_aos(e1, w4_1);
        __m256d e2_2 = cmul_avx2_aos(e2, w4_2);
        __m256d e2_3 = cmul_avx2_aos(e3, w4_3);

        //==================================================================
        // Radix-5 butterfly (8 butterflies in parallel)
        //==================================================================
#define RADIX5_BUTTERFLY_AVX2(a, b2, c2, d2, e2, y0, y1, y2, y3, y4) \
    do                                                               \
    {                                                                \
        __m256d t0 = _mm256_add_pd(b2, e2);                          \
        __m256d t1 = _mm256_add_pd(c2, d2);                          \
        __m256d t2 = _mm256_sub_pd(b2, e2);                          \
        __m256d t3 = _mm256_sub_pd(c2, d2);                          \
        y0 = _mm256_add_pd(a, _mm256_add_pd(t0, t1));                \
        __m256d base1 = FMADD(vs1, t2, _mm256_mul_pd(vs2, t3));      \
        __m256d tmp1 = FMADD(vc1, t0, _mm256_mul_pd(vc2, t1));       \
        __m256d base1_swp = _mm256_permute_pd(base1, 0b0101);        \
        __m256d r1 = _mm256_xor_pd(base1_swp, rot_mask);             \
        __m256d a1 = _mm256_add_pd(a, tmp1);                         \
        y1 = _mm256_add_pd(a1, r1);                                  \
        y4 = _mm256_sub_pd(a1, r1);                                  \
        __m256d base2 = FMSUB(vs2, t2, _mm256_mul_pd(vs1, t3));      \
        __m256d tmp2 = FMADD(vc2, t0, _mm256_mul_pd(vc1, t1));       \
        __m256d base2_swp = _mm256_permute_pd(base2, 0b0101);        \
        __m256d r2 = _mm256_xor_pd(base2_swp, rot_mask);             \
        __m256d a2 = _mm256_add_pd(a, tmp2);                         \
        y3 = _mm256_add_pd(a2, r2);                                  \
        y2 = _mm256_sub_pd(a2, r2);                                  \
    } while (0)

        __m256d y0_0, y1_0, y2_0, y3_0, y4_0;
        __m256d y0_1, y1_1, y2_1, y3_1, y4_1;
        __m256d y0_2, y1_2, y2_2, y3_2, y4_2;
        __m256d y0_3, y1_3, y2_3, y3_3, y4_3;

        RADIX5_BUTTERFLY_AVX2(a0, b2_0, c2_0, d2_0, e2_0, y0_0, y1_0, y2_0, y3_0, y4_0);
        RADIX5_BUTTERFLY_AVX2(a1, b2_1, c2_1, d2_1, e2_1, y0_1, y1_1, y2_1, y3_1, y4_1);
        RADIX5_BUTTERFLY_AVX2(a2, b2_2, c2_2, d2_2, e2_2, y0_2, y1_2, y2_2, y3_2, y4_2);
        RADIX5_BUTTERFLY_AVX2(a3, b2_3, c2_3, d2_3, e2_3, y0_3, y1_3, y2_3, y3_3, y4_3);

#undef RADIX5_BUTTERFLY_AVX2

        //==================================================================
        // Store results (pure AoS!)
        //==================================================================
        STOREU_PD(&output_buffer[k + 0].re, y0_0);
        STOREU_PD(&output_buffer[k + 2].re, y0_1);
        STOREU_PD(&output_buffer[k + 4].re, y0_2);
        STOREU_PD(&output_buffer[k + 6].re, y0_3);

        STOREU_PD(&output_buffer[k + 0 + fifth].re, y1_0);
        STOREU_PD(&output_buffer[k + 2 + fifth].re, y1_1);
        STOREU_PD(&output_buffer[k + 4 + fifth].re, y1_2);
        STOREU_PD(&output_buffer[k + 6 + fifth].re, y1_3);

        STOREU_PD(&output_buffer[k + 0 + 2 * fifth].re, y2_0);
        STOREU_PD(&output_buffer[k + 2 + 2 * fifth].re, y2_1);
        STOREU_PD(&output_buffer[k + 4 + 2 * fifth].re, y2_2);
        STOREU_PD(&output_buffer[k + 6 + 2 * fifth].re, y2_3);

        STOREU_PD(&output_buffer[k + 0 + 3 * fifth].re, y3_0);
        STOREU_PD(&output_buffer[k + 2 + 3 * fifth].re, y3_1);
        STOREU_PD(&output_buffer[k + 4 + 3 * fifth].re, y3_2);
        STOREU_PD(&output_buffer[k + 6 + 3 * fifth].re, y3_3);

        STOREU_PD(&output_buffer[k + 0 + 4 * fifth].re, y4_0);
        STOREU_PD(&output_buffer[k + 2 + 4 * fifth].re, y4_1);
        STOREU_PD(&output_buffer[k + 4 + 4 * fifth].re, y4_2);
        STOREU_PD(&output_buffer[k + 6 + 4 * fifth].re, y4_3);
    }

    //------------------------------------------------------------------
    // Cleanup: 2x unrolling (reuse rot_mask from above)
    //------------------------------------------------------------------
    for (; k + 1 < fifth; k += 2)
    {
        if (k + 8 < fifth)
        {
            _mm_prefetch((const char *)&sub_outputs[k + 8].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_outputs[k + 8 + fifth].re, _MM_HINT_T0);
        }

        __m256d a = load2_aos(&sub_outputs[k], &sub_outputs[k + 1]);
        __m256d b = load2_aos(&sub_outputs[k + fifth], &sub_outputs[k + fifth + 1]);
        __m256d c = load2_aos(&sub_outputs[k + 2 * fifth], &sub_outputs[k + 2 * fifth + 1]);
        __m256d d = load2_aos(&sub_outputs[k + 3 * fifth], &sub_outputs[k + 3 * fifth + 1]);
        __m256d e = load2_aos(&sub_outputs[k + 4 * fifth], &sub_outputs[k + 4 * fifth + 1]);

        __m256d w1 = load2_aos(&stage_tw[4 * k], &stage_tw[4 * (k + 1)]);
        __m256d w2 = load2_aos(&stage_tw[4 * k + 1], &stage_tw[4 * (k + 1) + 1]);
        __m256d w3 = load2_aos(&stage_tw[4 * k + 2], &stage_tw[4 * (k + 1) + 2]);
        __m256d w4 = load2_aos(&stage_tw[4 * k + 3], &stage_tw[4 * (k + 1) + 3]);

        __m256d b2 = cmul_avx2_aos(b, w1);
        __m256d c2 = cmul_avx2_aos(c, w2);
        __m256d d2 = cmul_avx2_aos(d, w3);
        __m256d e2 = cmul_avx2_aos(e, w4);

        __m256d t0 = _mm256_add_pd(b2, e2);
        __m256d t1 = _mm256_add_pd(c2, d2);
        __m256d t2 = _mm256_sub_pd(b2, e2);
        __m256d t3 = _mm256_sub_pd(c2, d2);

        __m256d y0 = _mm256_add_pd(a, _mm256_add_pd(t0, t1));

        __m256d base1 = FMADD(vs1, t2, _mm256_mul_pd(vs2, t3));
        __m256d tmp1 = FMADD(vc1, t0, _mm256_mul_pd(vc2, t1));
        __m256d base1_swp = _mm256_permute_pd(base1, 0b0101);
        __m256d r1 = _mm256_xor_pd(base1_swp, rot_mask);
        __m256d a1 = _mm256_add_pd(a, tmp1);
        __m256d y1 = _mm256_add_pd(a1, r1);
        __m256d y4 = _mm256_sub_pd(a1, r1);

        __m256d base2 = FMSUB(vs2, t2, _mm256_mul_pd(vs1, t3));
        __m256d tmp2 = FMADD(vc2, t0, _mm256_mul_pd(vc1, t1));
        __m256d base2_swp = _mm256_permute_pd(base2, 0b0101);
        __m256d r2 = _mm256_xor_pd(base2_swp, rot_mask);
        __m256d a2 = _mm256_add_pd(a, tmp2);
        __m256d y3 = _mm256_add_pd(a2, r2);  // ← SWAPPED
        __m256d y2 = _mm256_sub_pd(a2, r2);  // ← SWAPPED

        STOREU_PD(&output_buffer[k].re, y0);
        STOREU_PD(&output_buffer[k + fifth].re, y1);
        STOREU_PD(&output_buffer[k + 2 * fifth].re, y2);
        STOREU_PD(&output_buffer[k + 3 * fifth].re, y3);
        STOREU_PD(&output_buffer[k + 4 * fifth].re, y4);
    }
#endif // __AVX2__

    //------------------------------------------------------------------
    // Scalar tail: Handle remaining 0..1 elements
    //------------------------------------------------------------------
    for (; k < fifth; ++k)
    {
        __m128d a = LOADU_SSE2(&sub_outputs[k].re);
        __m128d b = LOADU_SSE2(&sub_outputs[k + fifth].re);
        __m128d c = LOADU_SSE2(&sub_outputs[k + 2 * fifth].re);
        __m128d d = LOADU_SSE2(&sub_outputs[k + 3 * fifth].re);
        __m128d e = LOADU_SSE2(&sub_outputs[k + 4 * fifth].re);

        __m128d w1 = LOADU_SSE2(&stage_tw[4 * k].re);
        __m128d w2 = LOADU_SSE2(&stage_tw[4 * k + 1].re);
        __m128d w3 = LOADU_SSE2(&stage_tw[4 * k + 2].re);
        __m128d w4 = LOADU_SSE2(&stage_tw[4 * k + 3].re);

        __m128d b2 = cmul_sse2_aos(b, w1);
        __m128d c2 = cmul_sse2_aos(c, w2);
        __m128d d2 = cmul_sse2_aos(d, w3);
        __m128d e2 = cmul_sse2_aos(e, w4);

        __m128d t0 = _mm_add_pd(b2, e2);
        __m128d t1 = _mm_add_pd(c2, d2);
        __m128d t2 = _mm_sub_pd(b2, e2);
        __m128d t3 = _mm_sub_pd(c2, d2);

        __m128d y0 = _mm_add_pd(a, _mm_add_pd(t0, t1));
        STOREU_SSE2(&output_buffer[k].re, y0);

        const __m128d vc1_128 = _mm_set1_pd(C5_1);
        const __m128d vc2_128 = _mm_set1_pd(C5_2);
        const __m128d vs1_128 = _mm_set1_pd(S5_1);
        const __m128d vs2_128 = _mm_set1_pd(S5_2);

        __m128d base1 = FMADD_SSE2(vs1_128, t2, _mm_mul_pd(vs2_128, t3));
        __m128d tmp1 = FMADD_SSE2(vc1_128, t0, _mm_mul_pd(vc2_128, t1));
        __m128d base1_swp = _mm_shuffle_pd(base1, base1, 0b01);
         // ✅ FIX: SWAP THE ROTATION MASK CONDITIONS
        __m128d r1 = (transform_sign == 1)
                     ? _mm_xor_pd(base1_swp, _mm_set_pd(-0.0, 0.0))  // +i (was -i)
                     : _mm_xor_pd(base1_swp, _mm_set_pd(0.0, -0.0)); // -i (was +i)
                 
        __m128d a1 = _mm_add_pd(a, tmp1);
        __m128d y1 = _mm_add_pd(a1, r1);
        __m128d y4 = _mm_sub_pd(a1, r1);

        __m128d base2 = FMSUB_SSE2(vs2_128, t2, _mm_mul_pd(vs1_128, t3));
        __m128d tmp2 = FMADD_SSE2(vc2_128, t0, _mm_mul_pd(vc1_128, t1));
        __m128d base2_swp = _mm_shuffle_pd(base2, base2, 0b01);
         __m128d r2 = (transform_sign == 1)
                     ? _mm_xor_pd(base2_swp, _mm_set_pd(0.0, -0.0))  // -i (was +i) 
                     : _mm_xor_pd(base2_swp, _mm_set_pd(-0.0, 0.0)); // +i (was -i)
                     
        __m128d a2 = _mm_add_pd(a, tmp2);
        __m128d y3 = _mm_add_pd(a2, r2);  // ← SWAPPED
        __m128d y2 = _mm_sub_pd(a2, r2);  // ← SWAPPED

        STOREU_SSE2(&output_buffer[k + fifth].re, y1);
        STOREU_SSE2(&output_buffer[k + 2 * fifth].re, y2);
        STOREU_SSE2(&output_buffer[k + 3 * fifth].re, y3);
        STOREU_SSE2(&output_buffer[k + 4 * fifth].re, y4);
    }
}