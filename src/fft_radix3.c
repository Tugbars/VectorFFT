#include "fft_radix3.h"
#include "simd_math.h"

void fft_radix3_butterfly(
    fft_data * restrict output_buffer,
    fft_data * restrict sub_outputs,
    const fft_data * restrict stage_tw,
    int sub_len,
    int transform_sign)
{
    const int third = sub_len;
    int k = 0;

    const double C_HALF = -0.5;
    const double S_SQRT3_2 = 0.8660254037844386;

#ifdef __AVX2__
    const __m256d v_half = _mm256_set1_pd(C_HALF);
    const __m256d v_sqrt3_2 = _mm256_set1_pd(S_SQRT3_2);

    // Aligned with radix-5 convention
    const __m256d rot_mask = (transform_sign == 1)
                             ? _mm256_set_pd(0.0, -0.0, 0.0, -0.0)
                             : _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);

    // Pointer arithmetic optimization - compute base addresses once
    fft_data * restrict out0 = output_buffer;
    fft_data * restrict out1 = output_buffer + third;
    fft_data * restrict out2 = output_buffer + 2 * third;
    
    const fft_data * restrict in0 = sub_outputs;
    const fft_data * restrict in1 = sub_outputs + third;
    const fft_data * restrict in2 = sub_outputs + 2 * third;

    //======================================================================
    // Main loop: 16x unrolling for better instruction-level parallelism
    //======================================================================
    for (; k + 15 < third; k += 16)
    {
        // Prefetch ahead (keep as requested)
        if (k + 32 < third)
        {
            _mm_prefetch((const char *)&in0[k + 32].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&in1[k + 32].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&in2[k + 32].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&stage_tw[2 * (k + 32)].re, _MM_HINT_T0);
        }

        //==================================================================
        // Software pipelining: Load all inputs first (16 butterflies)
        //==================================================================
        __m256d a0 = load2_aos(&in0[k + 0], &in0[k + 1]);
        __m256d a1 = load2_aos(&in0[k + 2], &in0[k + 3]);
        __m256d a2 = load2_aos(&in0[k + 4], &in0[k + 5]);
        __m256d a3 = load2_aos(&in0[k + 6], &in0[k + 7]);
        __m256d a4 = load2_aos(&in0[k + 8], &in0[k + 9]);
        __m256d a5 = load2_aos(&in0[k + 10], &in0[k + 11]);
        __m256d a6 = load2_aos(&in0[k + 12], &in0[k + 13]);
        __m256d a7 = load2_aos(&in0[k + 14], &in0[k + 15]);

        __m256d b0 = load2_aos(&in1[k + 0], &in1[k + 1]);
        __m256d b1 = load2_aos(&in1[k + 2], &in1[k + 3]);
        __m256d b2 = load2_aos(&in1[k + 4], &in1[k + 5]);
        __m256d b3 = load2_aos(&in1[k + 6], &in1[k + 7]);
        __m256d b4 = load2_aos(&in1[k + 8], &in1[k + 9]);
        __m256d b5 = load2_aos(&in1[k + 10], &in1[k + 11]);
        __m256d b6 = load2_aos(&in1[k + 12], &in1[k + 13]);
        __m256d b7 = load2_aos(&in1[k + 14], &in1[k + 15]);

        __m256d c0 = load2_aos(&in2[k + 0], &in2[k + 1]);
        __m256d c1 = load2_aos(&in2[k + 2], &in2[k + 3]);
        __m256d c2 = load2_aos(&in2[k + 4], &in2[k + 5]);
        __m256d c3 = load2_aos(&in2[k + 6], &in2[k + 7]);
        __m256d c4 = load2_aos(&in2[k + 8], &in2[k + 9]);
        __m256d c5 = load2_aos(&in2[k + 10], &in2[k + 11]);
        __m256d c6 = load2_aos(&in2[k + 12], &in2[k + 13]);
        __m256d c7 = load2_aos(&in2[k + 14], &in2[k + 15]);

        //==================================================================
        // Load twiddles (interleaved to improve scheduling)
        //==================================================================
        __m256d w1_0 = load2_aos(&stage_tw[2 * (k + 0)], &stage_tw[2 * (k + 1)]);
        __m256d w2_0 = load2_aos(&stage_tw[2 * (k + 0) + 1], &stage_tw[2 * (k + 1) + 1]);
        
        __m256d w1_1 = load2_aos(&stage_tw[2 * (k + 2)], &stage_tw[2 * (k + 3)]);
        __m256d w2_1 = load2_aos(&stage_tw[2 * (k + 2) + 1], &stage_tw[2 * (k + 3) + 1]);
        
        __m256d w1_2 = load2_aos(&stage_tw[2 * (k + 4)], &stage_tw[2 * (k + 5)]);
        __m256d w2_2 = load2_aos(&stage_tw[2 * (k + 4) + 1], &stage_tw[2 * (k + 5) + 1]);
        
        __m256d w1_3 = load2_aos(&stage_tw[2 * (k + 6)], &stage_tw[2 * (k + 7)]);
        __m256d w2_3 = load2_aos(&stage_tw[2 * (k + 6) + 1], &stage_tw[2 * (k + 7) + 1]);

        __m256d w1_4 = load2_aos(&stage_tw[2 * (k + 8)], &stage_tw[2 * (k + 9)]);
        __m256d w2_4 = load2_aos(&stage_tw[2 * (k + 8) + 1], &stage_tw[2 * (k + 9) + 1]);
        
        __m256d w1_5 = load2_aos(&stage_tw[2 * (k + 10)], &stage_tw[2 * (k + 11)]);
        __m256d w2_5 = load2_aos(&stage_tw[2 * (k + 10) + 1], &stage_tw[2 * (k + 11) + 1]);
        
        __m256d w1_6 = load2_aos(&stage_tw[2 * (k + 12)], &stage_tw[2 * (k + 13)]);
        __m256d w2_6 = load2_aos(&stage_tw[2 * (k + 12) + 1], &stage_tw[2 * (k + 13) + 1]);
        
        __m256d w1_7 = load2_aos(&stage_tw[2 * (k + 14)], &stage_tw[2 * (k + 15)]);
        __m256d w2_7 = load2_aos(&stage_tw[2 * (k + 14) + 1], &stage_tw[2 * (k + 15) + 1]);

        //==================================================================
        // Twiddle multiply (interleaved for better pipeline utilization)
        //==================================================================
        __m256d b2_0 = cmul_avx2_aos(b0, w1_0);
        __m256d c2_0 = cmul_avx2_aos(c0, w2_0);
        
        __m256d b2_1 = cmul_avx2_aos(b1, w1_1);
        __m256d c2_1 = cmul_avx2_aos(c1, w2_1);
        
        __m256d b2_2 = cmul_avx2_aos(b2, w1_2);
        __m256d c2_2 = cmul_avx2_aos(c2, w2_2);
        
        __m256d b2_3 = cmul_avx2_aos(b3, w1_3);
        __m256d c2_3 = cmul_avx2_aos(c3, w2_3);

        __m256d b2_4 = cmul_avx2_aos(b4, w1_4);
        __m256d c2_4 = cmul_avx2_aos(c4, w2_4);
        
        __m256d b2_5 = cmul_avx2_aos(b5, w1_5);
        __m256d c2_5 = cmul_avx2_aos(c5, w2_5);
        
        __m256d b2_6 = cmul_avx2_aos(b6, w1_6);
        __m256d c2_6 = cmul_avx2_aos(c6, w2_6);
        
        __m256d b2_7 = cmul_avx2_aos(b7, w1_7);
        __m256d c2_7 = cmul_avx2_aos(c7, w2_7);

        //==================================================================
        // Butterfly computations (macro for clarity and compiler optimization)
        //==================================================================
#define RADIX3_BUTTERFLY_AVX2(a, b2, c2, y0, y1, y2)          \
    do {                                                      \
        __m256d sum = _mm256_add_pd(b2, c2);                  \
        __m256d dif = _mm256_sub_pd(b2, c2);                  \
        y0 = _mm256_add_pd(a, sum);                           \
        __m256d common = FMADD(v_half, sum, a);               \
        __m256d dif_swp = _mm256_permute_pd(dif, 0b0101);     \
        __m256d rot90 = _mm256_xor_pd(dif_swp, rot_mask);     \
        __m256d scaled_rot = _mm256_mul_pd(rot90, v_sqrt3_2); \
        y1 = _mm256_add_pd(common, scaled_rot);               \
        y2 = _mm256_sub_pd(common, scaled_rot);               \
    } while(0)

        __m256d y0_0, y1_0, y2_0;
        __m256d y0_1, y1_1, y2_1;
        __m256d y0_2, y1_2, y2_2;
        __m256d y0_3, y1_3, y2_3;
        __m256d y0_4, y1_4, y2_4;
        __m256d y0_5, y1_5, y2_5;
        __m256d y0_6, y1_6, y2_6;
        __m256d y0_7, y1_7, y2_7;

        RADIX3_BUTTERFLY_AVX2(a0, b2_0, c2_0, y0_0, y1_0, y2_0);
        RADIX3_BUTTERFLY_AVX2(a1, b2_1, c2_1, y0_1, y1_1, y2_1);
        RADIX3_BUTTERFLY_AVX2(a2, b2_2, c2_2, y0_2, y1_2, y2_2);
        RADIX3_BUTTERFLY_AVX2(a3, b2_3, c2_3, y0_3, y1_3, y2_3);
        RADIX3_BUTTERFLY_AVX2(a4, b2_4, c2_4, y0_4, y1_4, y2_4);
        RADIX3_BUTTERFLY_AVX2(a5, b2_5, c2_5, y0_5, y1_5, y2_5);
        RADIX3_BUTTERFLY_AVX2(a6, b2_6, c2_6, y0_6, y1_6, y2_6);
        RADIX3_BUTTERFLY_AVX2(a7, b2_7, c2_7, y0_7, y1_7, y2_7);

#undef RADIX3_BUTTERFLY_AVX2

        //==================================================================
        // Store results (grouped by output lane for better cache usage)
        //==================================================================
        STOREU_PD(&out0[k + 0].re, y0_0);
        STOREU_PD(&out0[k + 2].re, y0_1);
        STOREU_PD(&out0[k + 4].re, y0_2);
        STOREU_PD(&out0[k + 6].re, y0_3);
        STOREU_PD(&out0[k + 8].re, y0_4);
        STOREU_PD(&out0[k + 10].re, y0_5);
        STOREU_PD(&out0[k + 12].re, y0_6);
        STOREU_PD(&out0[k + 14].re, y0_7);

        STOREU_PD(&out1[k + 0].re, y1_0);
        STOREU_PD(&out1[k + 2].re, y1_1);
        STOREU_PD(&out1[k + 4].re, y1_2);
        STOREU_PD(&out1[k + 6].re, y1_3);
        STOREU_PD(&out1[k + 8].re, y1_4);
        STOREU_PD(&out1[k + 10].re, y1_5);
        STOREU_PD(&out1[k + 12].re, y1_6);
        STOREU_PD(&out1[k + 14].re, y1_7);

        STOREU_PD(&out2[k + 0].re, y2_0);
        STOREU_PD(&out2[k + 2].re, y2_1);
        STOREU_PD(&out2[k + 4].re, y2_2);
        STOREU_PD(&out2[k + 6].re, y2_3);
        STOREU_PD(&out2[k + 8].re, y2_4);
        STOREU_PD(&out2[k + 10].re, y2_5);
        STOREU_PD(&out2[k + 12].re, y2_6);
        STOREU_PD(&out2[k + 14].re, y2_7);
    }

    //======================================================================
    // Cleanup: 8x unrolling
    //======================================================================
    for (; k + 7 < third; k += 8)
    {
        if (k + 16 < third)
        {
            _mm_prefetch((const char *)&in0[k + 16].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&in1[k + 16].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&in2[k + 16].re, _MM_HINT_T0);
        }

        __m256d a0 = load2_aos(&in0[k + 0], &in0[k + 1]);
        __m256d a1 = load2_aos(&in0[k + 2], &in0[k + 3]);
        __m256d a2 = load2_aos(&in0[k + 4], &in0[k + 5]);
        __m256d a3 = load2_aos(&in0[k + 6], &in0[k + 7]);

        __m256d b0 = load2_aos(&in1[k + 0], &in1[k + 1]);
        __m256d b1 = load2_aos(&in1[k + 2], &in1[k + 3]);
        __m256d b2 = load2_aos(&in1[k + 4], &in1[k + 5]);
        __m256d b3 = load2_aos(&in1[k + 6], &in1[k + 7]);

        __m256d c0 = load2_aos(&in2[k + 0], &in2[k + 1]);
        __m256d c1 = load2_aos(&in2[k + 2], &in2[k + 3]);
        __m256d c2 = load2_aos(&in2[k + 4], &in2[k + 5]);
        __m256d c3 = load2_aos(&in2[k + 6], &in2[k + 7]);

        __m256d w1_0 = load2_aos(&stage_tw[2 * (k + 0)], &stage_tw[2 * (k + 1)]);
        __m256d w1_1 = load2_aos(&stage_tw[2 * (k + 2)], &stage_tw[2 * (k + 3)]);
        __m256d w1_2 = load2_aos(&stage_tw[2 * (k + 4)], &stage_tw[2 * (k + 5)]);
        __m256d w1_3 = load2_aos(&stage_tw[2 * (k + 6)], &stage_tw[2 * (k + 7)]);

        __m256d w2_0 = load2_aos(&stage_tw[2 * (k + 0) + 1], &stage_tw[2 * (k + 1) + 1]);
        __m256d w2_1 = load2_aos(&stage_tw[2 * (k + 2) + 1], &stage_tw[2 * (k + 3) + 1]);
        __m256d w2_2 = load2_aos(&stage_tw[2 * (k + 4) + 1], &stage_tw[2 * (k + 5) + 1]);
        __m256d w2_3 = load2_aos(&stage_tw[2 * (k + 6) + 1], &stage_tw[2 * (k + 7) + 1]);

        __m256d b2_0 = cmul_avx2_aos(b0, w1_0);
        __m256d b2_1 = cmul_avx2_aos(b1, w1_1);
        __m256d b2_2 = cmul_avx2_aos(b2, w1_2);
        __m256d b2_3 = cmul_avx2_aos(b3, w1_3);

        __m256d c2_0 = cmul_avx2_aos(c0, w2_0);
        __m256d c2_1 = cmul_avx2_aos(c1, w2_1);
        __m256d c2_2 = cmul_avx2_aos(c2, w2_2);
        __m256d c2_3 = cmul_avx2_aos(c3, w2_3);

#define RADIX3_BUTTERFLY_AVX2(a, b2, c2, y0, y1, y2)          \
    do {                                                      \
        __m256d sum = _mm256_add_pd(b2, c2);                  \
        __m256d dif = _mm256_sub_pd(b2, c2);                  \
        y0 = _mm256_add_pd(a, sum);                           \
        __m256d common = FMADD(v_half, sum, a);               \
        __m256d dif_swp = _mm256_permute_pd(dif, 0b0101);     \
        __m256d rot90 = _mm256_xor_pd(dif_swp, rot_mask);     \
        __m256d scaled_rot = _mm256_mul_pd(rot90, v_sqrt3_2); \
        y1 = _mm256_add_pd(common, scaled_rot);               \
        y2 = _mm256_sub_pd(common, scaled_rot);               \
    } while(0)

        __m256d y0_0, y1_0, y2_0;
        __m256d y0_1, y1_1, y2_1;
        __m256d y0_2, y1_2, y2_2;
        __m256d y0_3, y1_3, y2_3;

        RADIX3_BUTTERFLY_AVX2(a0, b2_0, c2_0, y0_0, y1_0, y2_0);
        RADIX3_BUTTERFLY_AVX2(a1, b2_1, c2_1, y0_1, y1_1, y2_1);
        RADIX3_BUTTERFLY_AVX2(a2, b2_2, c2_2, y0_2, y1_2, y2_2);
        RADIX3_BUTTERFLY_AVX2(a3, b2_3, c2_3, y0_3, y1_3, y2_3);

#undef RADIX3_BUTTERFLY_AVX2

        STOREU_PD(&out0[k + 0].re, y0_0);
        STOREU_PD(&out0[k + 2].re, y0_1);
        STOREU_PD(&out0[k + 4].re, y0_2);
        STOREU_PD(&out0[k + 6].re, y0_3);

        STOREU_PD(&out1[k + 0].re, y1_0);
        STOREU_PD(&out1[k + 2].re, y1_1);
        STOREU_PD(&out1[k + 4].re, y1_2);
        STOREU_PD(&out1[k + 6].re, y1_3);

        STOREU_PD(&out2[k + 0].re, y2_0);
        STOREU_PD(&out2[k + 2].re, y2_1);
        STOREU_PD(&out2[k + 4].re, y2_2);
        STOREU_PD(&out2[k + 6].re, y2_3);
    }

    //======================================================================
    // Cleanup: 2x unrolling
    //======================================================================
    for (; k + 1 < third; k += 2)
    {
        __m256d a = load2_aos(&in0[k], &in0[k + 1]);
        __m256d b = load2_aos(&in1[k], &in1[k + 1]);
        __m256d c = load2_aos(&in2[k], &in2[k + 1]);

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

        STOREU_PD(&out0[k].re, y0);
        STOREU_PD(&out1[k].re, y1);
        STOREU_PD(&out2[k].re, y2);
    }
#endif // __AVX2__

    //======================================================================
    // Scalar tail: Handle remaining 0..1 elements
    //======================================================================
    for (; k < third; ++k)
    {
        fft_data a = sub_outputs[k];
        fft_data b = sub_outputs[k + third];
        fft_data c = sub_outputs[k + 2 * third];

        fft_data w1 = stage_tw[2 * k];
        fft_data w2 = stage_tw[2 * k + 1];

        double b2r = b.re * w1.re - b.im * w1.im;
        double b2i = b.re * w1.im + b.im * w1.re;

        double c2r = c.re * w2.re - c.im * w2.im;
        double c2i = c.re * w2.im + c.im * w2.re;

        double sumr = b2r + c2r;
        double sumi = b2i + c2i;
        double difr = b2r - c2r;
        double difi = b2i - c2i;

        output_buffer[k].re = a.re + sumr;
        output_buffer[k].im = a.im + sumi;

        double commonr = a.re + C_HALF * sumr;
        double commoni = a.im + C_HALF * sumi;

        double scaled_rotr, scaled_roti;
        if (transform_sign == 1) {
            // Forward: multiply by -i
            scaled_rotr = S_SQRT3_2 * difi;
            scaled_roti = -S_SQRT3_2 * difr;
        }
        else {
            // Inverse: multiply by +i
            scaled_rotr = -S_SQRT3_2 * difi;
            scaled_roti = S_SQRT3_2 * difr;
        }

        output_buffer[k + third].re = commonr + scaled_rotr;
        output_buffer[k + third].im = commoni + scaled_roti;

        output_buffer[k + 2 * third].re = commonr - scaled_rotr;
        output_buffer[k + 2 * third].im = commoni - scaled_roti;
    }
}