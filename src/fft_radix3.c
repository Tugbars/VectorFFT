#include "fft_radix3.h"   // ✅ Gets highSpeedFFT.h → fft_types.h
#include "simd_math.h"    // ✅ Gets complex math operations


void fft_radix3_butterfly(
    fft_data *output_buffer,
    fft_data *sub_outputs,
    const fft_data *stage_tw,
    int sub_len,
    int transform_sign)
{
      //======================================================================
        // RADIX-3 BUTTERFLY (DIT) - FFTW-STYLE OPTIMIZED
        //
        // Uses Rader's algorithm with symmetry exploitation:
        // Y_0 = a + b + c
        // Y_1 = a + (b+c)*C1 + (-i*sgn)*(b-c)*S1
        // Y_2 = a + (b+c)*C2 + (-i*sgn)*(b-c)*S2
        //
        // Where C1 = cos(2π/3) = -0.5, S1 = sin(2π/3) = √3/2
        //       C2 = cos(4π/3) = -0.5, S2 = sin(4π/3) = -√3/2
        //
        // Optimization: C1 = C2 = -0.5, so factor out common term
        //======================================================================

        const int third = sub_len;
        int k = 0;

        // Constants
        const double C_HALF = -0.5;                  // cos(2π/3) = cos(4π/3)
        const double S_SQRT3_2 = 0.8660254037844386; // √3/2

#ifdef __AVX2__
        //----------------------------------------------------------------------
        // AVX2 PATH: 8x unrolling, pure AoS
        //----------------------------------------------------------------------
        const __m256d v_half = _mm256_set1_pd(C_HALF);
        const __m256d v_sqrt3_2 = _mm256_set1_pd(S_SQRT3_2);

        // Precompute rotation mask for (-i*sgn) multiplication
        // After permute: [im0, re0, im1, re1]
        // Forward (sgn=+1): multiply by -i → negate lanes 1,3 (imaginary after swap)
        // Inverse (sgn=-1): multiply by +i → negate lanes 0,2 (real after swap)
        const __m256d rot_mask = (transform_sign == 1)
                                     ? _mm256_set_pd(0.0, -0.0, 0.0, -0.0)  // -i: negate lanes [3]=0,[2]=-0,[1]=0,[0]=-0
                                     : _mm256_set_pd(-0.0, 0.0, -0.0, 0.0); // +i: negate lanes [3]=-0,[2]=0,[1]=-0,[0]=0

        for (; k + 7 < third; k += 8)
        {
            // Prefetch ahead
            if (k + 16 < third)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 16].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 16 + third].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 16 + 2 * third].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&stage_tw[2 * (k + 16)].re, _MM_HINT_T0);
            }

            //==================================================================
            // Load inputs (8 butterflies = 4 AVX2 loads per lane)
            //==================================================================
            __m256d a0 = load2_aos(&sub_outputs[k + 0], &sub_outputs[k + 1]);
            __m256d a1 = load2_aos(&sub_outputs[k + 2], &sub_outputs[k + 3]);
            __m256d a2 = load2_aos(&sub_outputs[k + 4], &sub_outputs[k + 5]);
            __m256d a3 = load2_aos(&sub_outputs[k + 6], &sub_outputs[k + 7]);

            __m256d b0 = load2_aos(&sub_outputs[k + 0 + third], &sub_outputs[k + 1 + third]);
            __m256d b1 = load2_aos(&sub_outputs[k + 2 + third], &sub_outputs[k + 3 + third]);
            __m256d b2 = load2_aos(&sub_outputs[k + 4 + third], &sub_outputs[k + 5 + third]);
            __m256d b3 = load2_aos(&sub_outputs[k + 6 + third], &sub_outputs[k + 7 + third]);

            __m256d c0 = load2_aos(&sub_outputs[k + 0 + 2 * third], &sub_outputs[k + 1 + 2 * third]);
            __m256d c1 = load2_aos(&sub_outputs[k + 2 + 2 * third], &sub_outputs[k + 3 + 2 * third]);
            __m256d c2 = load2_aos(&sub_outputs[k + 4 + 2 * third], &sub_outputs[k + 5 + 2 * third]);
            __m256d c3 = load2_aos(&sub_outputs[k + 6 + 2 * third], &sub_outputs[k + 7 + 2 * third]);

            //==================================================================
            // Load twiddles W^k and W^{2k} (k-major: stage_tw[2k], stage_tw[2k+1])
            //==================================================================
            __m256d w1_0 = load2_aos(&stage_tw[2 * (k + 0)], &stage_tw[2 * (k + 1)]);
            __m256d w1_1 = load2_aos(&stage_tw[2 * (k + 2)], &stage_tw[2 * (k + 3)]);
            __m256d w1_2 = load2_aos(&stage_tw[2 * (k + 4)], &stage_tw[2 * (k + 5)]);
            __m256d w1_3 = load2_aos(&stage_tw[2 * (k + 6)], &stage_tw[2 * (k + 7)]);

            __m256d w2_0 = load2_aos(&stage_tw[2 * (k + 0) + 1], &stage_tw[2 * (k + 1) + 1]);
            __m256d w2_1 = load2_aos(&stage_tw[2 * (k + 2) + 1], &stage_tw[2 * (k + 3) + 1]);
            __m256d w2_2 = load2_aos(&stage_tw[2 * (k + 4) + 1], &stage_tw[2 * (k + 5) + 1]);
            __m256d w2_3 = load2_aos(&stage_tw[2 * (k + 6) + 1], &stage_tw[2 * (k + 7) + 1]);

            //==================================================================
            // Twiddle multiply: b2 = b * W^k, c2 = c * W^{2k}
            //==================================================================
            __m256d b2_0 = cmul_avx2_aos(b0, w1_0);
            __m256d b2_1 = cmul_avx2_aos(b1, w1_1);
            __m256d b2_2 = cmul_avx2_aos(b2, w1_2);
            __m256d b2_3 = cmul_avx2_aos(b3, w1_3);

            __m256d c2_0 = cmul_avx2_aos(c0, w2_0);
            __m256d c2_1 = cmul_avx2_aos(c1, w2_1);
            __m256d c2_2 = cmul_avx2_aos(c2, w2_2);
            __m256d c2_3 = cmul_avx2_aos(c3, w2_3);

            //==================================================================
            // Radix-3 butterfly computation (8 butterflies in parallel)
            //
            // sum = b2 + c2
            // dif = b2 - c2
            // Y_0 = a + sum
            // common = a - 0.5 * sum
            // Y_1 = common + (-i*sgn) * √3/2 * dif
            // Y_2 = common - (-i*sgn) * √3/2 * dif
            //==================================================================

#define RADIX3_BUTTERFLY_AVX2(a, b2, c2, y0, y1, y2)          \
    {                                                         \
        __m256d sum = _mm256_add_pd(b2, c2);                  \
        __m256d dif = _mm256_sub_pd(b2, c2);                  \
        y0 = _mm256_add_pd(a, sum);                           \
        __m256d common = FMADD(v_half, sum, a);               \
        __m256d dif_swp = _mm256_permute_pd(dif, 0b0101);     \
        __m256d rot90 = _mm256_xor_pd(dif_swp, rot_mask);     \
        __m256d scaled_rot = _mm256_mul_pd(rot90, v_sqrt3_2); \
        y1 = _mm256_add_pd(common, scaled_rot);               \
        y2 = _mm256_sub_pd(common, scaled_rot);               \
    }

            __m256d y0_0, y1_0, y2_0;
            __m256d y0_1, y1_1, y2_1;
            __m256d y0_2, y1_2, y2_2;
            __m256d y0_3, y1_3, y2_3;

            RADIX3_BUTTERFLY_AVX2(a0, b2_0, c2_0, y0_0, y1_0, y2_0);
            RADIX3_BUTTERFLY_AVX2(a1, b2_1, c2_1, y0_1, y1_1, y2_1);
            RADIX3_BUTTERFLY_AVX2(a2, b2_2, c2_2, y0_2, y1_2, y2_2);
            RADIX3_BUTTERFLY_AVX2(a3, b2_3, c2_3, y0_3, y1_3, y2_3);

#undef RADIX3_BUTTERFLY_AVX2

            //==================================================================
            // Store results (pure AoS)
            //==================================================================
            STOREU_PD(&output_buffer[k + 0].re, y0_0);
            STOREU_PD(&output_buffer[k + 2].re, y0_1);
            STOREU_PD(&output_buffer[k + 4].re, y0_2);
            STOREU_PD(&output_buffer[k + 6].re, y0_3);

            STOREU_PD(&output_buffer[k + 0 + third].re, y1_0);
            STOREU_PD(&output_buffer[k + 2 + third].re, y1_1);
            STOREU_PD(&output_buffer[k + 4 + third].re, y1_2);
            STOREU_PD(&output_buffer[k + 6 + third].re, y1_3);

            STOREU_PD(&output_buffer[k + 0 + 2 * third].re, y2_0);
            STOREU_PD(&output_buffer[k + 2 + 2 * third].re, y2_1);
            STOREU_PD(&output_buffer[k + 4 + 2 * third].re, y2_2);
            STOREU_PD(&output_buffer[k + 6 + 2 * third].re, y2_3);
        }

        //----------------------------------------------------------------------
        // Cleanup: 2x unrolling
        //----------------------------------------------------------------------
        for (; k + 1 < third; k += 2)
        {
            if (k + 8 < third)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 8].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 8 + third].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 8 + 2 * third].re, _MM_HINT_T0);
            }

            __m256d a = load2_aos(&sub_outputs[k], &sub_outputs[k + 1]);
            __m256d b = load2_aos(&sub_outputs[k + third], &sub_outputs[k + third + 1]);
            __m256d c = load2_aos(&sub_outputs[k + 2 * third], &sub_outputs[k + 2 * third + 1]);

            __m256d w1 = load2_aos(&stage_tw[2 * k], &stage_tw[2 * (k + 1)]);
            __m256d w2 = load2_aos(&stage_tw[2 * k + 1], &stage_tw[2 * (k + 1) + 1]);

            __m256d b2 = cmul_avx2_aos(b, w1);
            __m256d c2 = cmul_avx2_aos(c, w2);

            __m256d sum = _mm256_add_pd(b2, c2);
            __m256d dif = _mm256_sub_pd(b2, c2);

            __m256d y0 = _mm256_add_pd(a, sum);
            __m256d common = FMADD(v_half, sum, a);

            __m256d dif_swp = _mm256_permute_pd(dif, 0b0101);
            __m256d rot90 = _mm256_xor_pd(dif_swp, rot_mask);
            __m256d scaled_rot = _mm256_mul_pd(rot90, v_sqrt3_2);

            __m256d y1 = _mm256_add_pd(common, scaled_rot);
            __m256d y2 = _mm256_sub_pd(common, scaled_rot);

            STOREU_PD(&output_buffer[k].re, y0);
            STOREU_PD(&output_buffer[k + third].re, y1);
            STOREU_PD(&output_buffer[k + 2 * third].re, y2);
        }
#endif // __AVX2__

        //----------------------------------------------------------------------
        // Scalar tail: Handle remaining 0..1 elements
        //----------------------------------------------------------------------
        for (; k < third; ++k)
        {
            // Load inputs
            fft_data a = sub_outputs[k];
            fft_data b = sub_outputs[k + third];
            fft_data c = sub_outputs[k + 2 * third];

            // Load twiddles
            fft_data w1 = stage_tw[2 * k];
            fft_data w2 = stage_tw[2 * k + 1];

            // Twiddle multiply
            double b2r = b.re * w1.re - b.im * w1.im;
            double b2i = b.re * w1.im + b.im * w1.re;

            double c2r = c.re * w2.re - c.im * w2.im;
            double c2i = c.re * w2.im + c.im * w2.re;

            // Radix-3 butterfly
            double sumr = b2r + c2r;
            double sumi = b2i + c2i;
            double difr = b2r - c2r;
            double difi = b2i - c2i;

            // Y_0 = a + sum
            output_buffer[k].re = a.re + sumr;
            output_buffer[k].im = a.im + sumi;

            // common = a - 0.5 * sum
            double commonr = a.re + C_HALF * sumr;
            double commoni = a.im + C_HALF * sumi;

            // scaled_rot = (-i*sgn) * √3/2 * dif
            double scaled_rotr, scaled_roti;
            if (transform_sign == 1)
            {
                // Forward: multiply by -i
                scaled_rotr = S_SQRT3_2 * difi;
                scaled_roti = -S_SQRT3_2 * difr;
            }
            else
            {
                // Inverse: multiply by +i
                scaled_rotr = -S_SQRT3_2 * difi;
                scaled_roti = S_SQRT3_2 * difr;
            }

            // Y_1 = common + scaled_rot
            output_buffer[k + third].re = commonr + scaled_rotr;
            output_buffer[k + third].im = commoni + scaled_roti;

            // Y_2 = common - scaled_rot
            output_buffer[k + 2 * third].re = commonr - scaled_rotr;
            output_buffer[k + 2 * third].im = commoni - scaled_roti;
        }
}