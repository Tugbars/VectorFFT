#include "fft_radix5.h"
#include "simd_math.h"

static const double C5_1 = 0.30901699437;
static const double C5_2 = -0.80901699437;
static const double S5_1 = 0.95105651629;
static const double S5_2 = 0.58778525229;

void fft_radix5_butterfly(
    fft_data * restrict output_buffer,
    fft_data * restrict sub_outputs,
    const fft_data * restrict stage_tw,
    int sub_len,
    int transform_sign)
{
    const int fifth = sub_len;
    int k = 0;

#ifdef __AVX2__
    const __m256d vc1 = _mm256_set1_pd(C5_1);
    const __m256d vc2 = _mm256_set1_pd(C5_2);
    const __m256d vs1 = _mm256_set1_pd(S5_1);
    const __m256d vs2 = _mm256_set1_pd(S5_2);

    const __m256d rot_mask = (transform_sign == 1)
                                 ? _mm256_set_pd(0.0, -0.0, 0.0, -0.0)
                                 : _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);

    // Pointer arithmetic optimization
    fft_data * restrict out0 = output_buffer;
    fft_data * restrict out1 = output_buffer + fifth;
    fft_data * restrict out2 = output_buffer + 2 * fifth;
    fft_data * restrict out3 = output_buffer + 3 * fifth;
    fft_data * restrict out4 = output_buffer + 4 * fifth;
    
    const fft_data * restrict in0 = sub_outputs;
    const fft_data * restrict in1 = sub_outputs + fifth;
    const fft_data * restrict in2 = sub_outputs + 2 * fifth;
    const fft_data * restrict in3 = sub_outputs + 3 * fifth;
    const fft_data * restrict in4 = sub_outputs + 4 * fifth;

    //======================================================================
    // Main loop: 16x unrolling
    //======================================================================
    for (; k + 15 < fifth; k += 16)
    {
        if (k + 32 < fifth)
        {
            _mm_prefetch((const char *)&in0[k + 32].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&in1[k + 32].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&in2[k + 32].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&in3[k + 32].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&in4[k + 32].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&stage_tw[4 * (k + 32)].re, _MM_HINT_T0);
        }

        // Load inputs
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

        __m256d d0 = load2_aos(&in3[k + 0], &in3[k + 1]);
        __m256d d1 = load2_aos(&in3[k + 2], &in3[k + 3]);
        __m256d d2 = load2_aos(&in3[k + 4], &in3[k + 5]);
        __m256d d3 = load2_aos(&in3[k + 6], &in3[k + 7]);
        __m256d d4 = load2_aos(&in3[k + 8], &in3[k + 9]);
        __m256d d5 = load2_aos(&in3[k + 10], &in3[k + 11]);
        __m256d d6 = load2_aos(&in3[k + 12], &in3[k + 13]);
        __m256d d7 = load2_aos(&in3[k + 14], &in3[k + 15]);

        __m256d e0 = load2_aos(&in4[k + 0], &in4[k + 1]);
        __m256d e1 = load2_aos(&in4[k + 2], &in4[k + 3]);
        __m256d e2 = load2_aos(&in4[k + 4], &in4[k + 5]);
        __m256d e3 = load2_aos(&in4[k + 6], &in4[k + 7]);
        __m256d e4 = load2_aos(&in4[k + 8], &in4[k + 9]);
        __m256d e5 = load2_aos(&in4[k + 10], &in4[k + 11]);
        __m256d e6 = load2_aos(&in4[k + 12], &in4[k + 13]);
        __m256d e7 = load2_aos(&in4[k + 14], &in4[k + 15]);

        // Load twiddles (interleaved for better scheduling)
        __m256d w1_0 = load2_aos(&stage_tw[4 * (k + 0)], &stage_tw[4 * (k + 1)]);
        __m256d w2_0 = load2_aos(&stage_tw[4 * (k + 0) + 1], &stage_tw[4 * (k + 1) + 1]);
        __m256d w3_0 = load2_aos(&stage_tw[4 * (k + 0) + 2], &stage_tw[4 * (k + 1) + 2]);
        __m256d w4_0 = load2_aos(&stage_tw[4 * (k + 0) + 3], &stage_tw[4 * (k + 1) + 3]);

        __m256d w1_1 = load2_aos(&stage_tw[4 * (k + 2)], &stage_tw[4 * (k + 3)]);
        __m256d w2_1 = load2_aos(&stage_tw[4 * (k + 2) + 1], &stage_tw[4 * (k + 3) + 1]);
        __m256d w3_1 = load2_aos(&stage_tw[4 * (k + 2) + 2], &stage_tw[4 * (k + 3) + 2]);
        __m256d w4_1 = load2_aos(&stage_tw[4 * (k + 2) + 3], &stage_tw[4 * (k + 3) + 3]);

        __m256d w1_2 = load2_aos(&stage_tw[4 * (k + 4)], &stage_tw[4 * (k + 5)]);
        __m256d w2_2 = load2_aos(&stage_tw[4 * (k + 4) + 1], &stage_tw[4 * (k + 5) + 1]);
        __m256d w3_2 = load2_aos(&stage_tw[4 * (k + 4) + 2], &stage_tw[4 * (k + 5) + 2]);
        __m256d w4_2 = load2_aos(&stage_tw[4 * (k + 4) + 3], &stage_tw[4 * (k + 5) + 3]);

        __m256d w1_3 = load2_aos(&stage_tw[4 * (k + 6)], &stage_tw[4 * (k + 7)]);
        __m256d w2_3 = load2_aos(&stage_tw[4 * (k + 6) + 1], &stage_tw[4 * (k + 7) + 1]);
        __m256d w3_3 = load2_aos(&stage_tw[4 * (k + 6) + 2], &stage_tw[4 * (k + 7) + 2]);
        __m256d w4_3 = load2_aos(&stage_tw[4 * (k + 6) + 3], &stage_tw[4 * (k + 7) + 3]);

        __m256d w1_4 = load2_aos(&stage_tw[4 * (k + 8)], &stage_tw[4 * (k + 9)]);
        __m256d w2_4 = load2_aos(&stage_tw[4 * (k + 8) + 1], &stage_tw[4 * (k + 9) + 1]);
        __m256d w3_4 = load2_aos(&stage_tw[4 * (k + 8) + 2], &stage_tw[4 * (k + 9) + 2]);
        __m256d w4_4 = load2_aos(&stage_tw[4 * (k + 8) + 3], &stage_tw[4 * (k + 9) + 3]);

        __m256d w1_5 = load2_aos(&stage_tw[4 * (k + 10)], &stage_tw[4 * (k + 11)]);
        __m256d w2_5 = load2_aos(&stage_tw[4 * (k + 10) + 1], &stage_tw[4 * (k + 11) + 1]);
        __m256d w3_5 = load2_aos(&stage_tw[4 * (k + 10) + 2], &stage_tw[4 * (k + 11) + 2]);
        __m256d w4_5 = load2_aos(&stage_tw[4 * (k + 10) + 3], &stage_tw[4 * (k + 11) + 3]);

        __m256d w1_6 = load2_aos(&stage_tw[4 * (k + 12)], &stage_tw[4 * (k + 13)]);
        __m256d w2_6 = load2_aos(&stage_tw[4 * (k + 12) + 1], &stage_tw[4 * (k + 13) + 1]);
        __m256d w3_6 = load2_aos(&stage_tw[4 * (k + 12) + 2], &stage_tw[4 * (k + 13) + 2]);
        __m256d w4_6 = load2_aos(&stage_tw[4 * (k + 12) + 3], &stage_tw[4 * (k + 13) + 3]);

        __m256d w1_7 = load2_aos(&stage_tw[4 * (k + 14)], &stage_tw[4 * (k + 15)]);
        __m256d w2_7 = load2_aos(&stage_tw[4 * (k + 14) + 1], &stage_tw[4 * (k + 15) + 1]);
        __m256d w3_7 = load2_aos(&stage_tw[4 * (k + 14) + 2], &stage_tw[4 * (k + 15) + 2]);
        __m256d w4_7 = load2_aos(&stage_tw[4 * (k + 14) + 3], &stage_tw[4 * (k + 15) + 3]);

        // Twiddle multiply
        __m256d b2_0 = cmul_avx2_aos(b0, w1_0);
        __m256d c2_0 = cmul_avx2_aos(c0, w2_0);
        __m256d d2_0 = cmul_avx2_aos(d0, w3_0);
        __m256d e2_0 = cmul_avx2_aos(e0, w4_0);

        __m256d b2_1 = cmul_avx2_aos(b1, w1_1);
        __m256d c2_1 = cmul_avx2_aos(c1, w2_1);
        __m256d d2_1 = cmul_avx2_aos(d1, w3_1);
        __m256d e2_1 = cmul_avx2_aos(e1, w4_1);

        __m256d b2_2 = cmul_avx2_aos(b2, w1_2);
        __m256d c2_2 = cmul_avx2_aos(c2, w2_2);
        __m256d d2_2 = cmul_avx2_aos(d2, w3_2);
        __m256d e2_2 = cmul_avx2_aos(e2, w4_2);

        __m256d b2_3 = cmul_avx2_aos(b3, w1_3);
        __m256d c2_3 = cmul_avx2_aos(c3, w2_3);
        __m256d d2_3 = cmul_avx2_aos(d3, w3_3);
        __m256d e2_3 = cmul_avx2_aos(e3, w4_3);

        __m256d b2_4 = cmul_avx2_aos(b4, w1_4);
        __m256d c2_4 = cmul_avx2_aos(c4, w2_4);
        __m256d d2_4 = cmul_avx2_aos(d4, w3_4);
        __m256d e2_4 = cmul_avx2_aos(e4, w4_4);

        __m256d b2_5 = cmul_avx2_aos(b5, w1_5);
        __m256d c2_5 = cmul_avx2_aos(c5, w2_5);
        __m256d d2_5 = cmul_avx2_aos(d5, w3_5);
        __m256d e2_5 = cmul_avx2_aos(e5, w4_5);

        __m256d b2_6 = cmul_avx2_aos(b6, w1_6);
        __m256d c2_6 = cmul_avx2_aos(c6, w2_6);
        __m256d d2_6 = cmul_avx2_aos(d6, w3_6);
        __m256d e2_6 = cmul_avx2_aos(e6, w4_6);

        __m256d b2_7 = cmul_avx2_aos(b7, w1_7);
        __m256d c2_7 = cmul_avx2_aos(c7, w2_7);
        __m256d d2_7 = cmul_avx2_aos(d7, w3_7);
        __m256d e2_7 = cmul_avx2_aos(e7, w4_7);

        // Butterfly computation
#define RADIX5_BUTTERFLY_AVX2(a, b2, c2, d2, e2, y0, y1, y2, y3, y4) \
    do {                                                             \
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
    } while(0)

        __m256d y0_0, y1_0, y2_0, y3_0, y4_0;
        __m256d y0_1, y1_1, y2_1, y3_1, y4_1;
        __m256d y0_2, y1_2, y2_2, y3_2, y4_2;
        __m256d y0_3, y1_3, y2_3, y3_3, y4_3;
        __m256d y0_4, y1_4, y2_4, y3_4, y4_4;
        __m256d y0_5, y1_5, y2_5, y3_5, y4_5;
        __m256d y0_6, y1_6, y2_6, y3_6, y4_6;
        __m256d y0_7, y1_7, y2_7, y3_7, y4_7;

        RADIX5_BUTTERFLY_AVX2(a0, b2_0, c2_0, d2_0, e2_0, y0_0, y1_0, y2_0, y3_0, y4_0);
        RADIX5_BUTTERFLY_AVX2(a1, b2_1, c2_1, d2_1, e2_1, y0_1, y1_1, y2_1, y3_1, y4_1);
        RADIX5_BUTTERFLY_AVX2(a2, b2_2, c2_2, d2_2, e2_2, y0_2, y1_2, y2_2, y3_2, y4_2);
        RADIX5_BUTTERFLY_AVX2(a3, b2_3, c2_3, d2_3, e2_3, y0_3, y1_3, y2_3, y3_3, y4_3);
        RADIX5_BUTTERFLY_AVX2(a4, b2_4, c2_4, d2_4, e2_4, y0_4, y1_4, y2_4, y3_4, y4_4);
        RADIX5_BUTTERFLY_AVX2(a5, b2_5, c2_5, d2_5, e2_5, y0_5, y1_5, y2_5, y3_5, y4_5);
        RADIX5_BUTTERFLY_AVX2(a6, b2_6, c2_6, d2_6, e2_6, y0_6, y1_6, y2_6, y3_6, y4_6);
        RADIX5_BUTTERFLY_AVX2(a7, b2_7, c2_7, d2_7, e2_7, y0_7, y1_7, y2_7, y3_7, y4_7);

#undef RADIX5_BUTTERFLY_AVX2

        // Store results
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

        STOREU_PD(&out3[k + 0].re, y3_0);
        STOREU_PD(&out3[k + 2].re, y3_1);
        STOREU_PD(&out3[k + 4].re, y3_2);
        STOREU_PD(&out3[k + 6].re, y3_3);
        STOREU_PD(&out3[k + 8].re, y3_4);
        STOREU_PD(&out3[k + 10].re, y3_5);
        STOREU_PD(&out3[k + 12].re, y3_6);
        STOREU_PD(&out3[k + 14].re, y3_7);

        STOREU_PD(&out4[k + 0].re, y4_0);
        STOREU_PD(&out4[k + 2].re, y4_1);
        STOREU_PD(&out4[k + 4].re, y4_2);
        STOREU_PD(&out4[k + 6].re, y4_3);
        STOREU_PD(&out4[k + 8].re, y4_4);
        STOREU_PD(&out4[k + 10].re, y4_5);
        STOREU_PD(&out4[k + 12].re, y4_6);
        STOREU_PD(&out4[k + 14].re, y4_7);
    }

    //======================================================================
    // Cleanup: 8x unrolling
    //======================================================================
    for (; k + 7 < fifth; k += 8)
    {
        if (k + 16 < fifth)
        {
            _mm_prefetch((const char *)&in0[k + 16].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&in1[k + 16].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&in2[k + 16].re, _MM_HINT_T0);
        }

        __m256d a0 = load2_aos(&in0[k], &in0[k + 1]);
        __m256d a1 = load2_aos(&in0[k + 2], &in0[k + 3]);
        __m256d a2 = load2_aos(&in0[k + 4], &in0[k + 5]);
        __m256d a3 = load2_aos(&in0[k + 6], &in0[k + 7]);

        __m256d b0 = load2_aos(&in1[k], &in1[k + 1]);
        __m256d b1 = load2_aos(&in1[k + 2], &in1[k + 3]);
        __m256d b2 = load2_aos(&in1[k + 4], &in1[k + 5]);
        __m256d b3 = load2_aos(&in1[k + 6], &in1[k + 7]);

        __m256d c0 = load2_aos(&in2[k], &in2[k + 1]);
        __m256d c1 = load2_aos(&in2[k + 2], &in2[k + 3]);
        __m256d c2 = load2_aos(&in2[k + 4], &in2[k + 5]);
        __m256d c3 = load2_aos(&in2[k + 6], &in2[k + 7]);

        __m256d d0 = load2_aos(&in3[k], &in3[k + 1]);
        __m256d d1 = load2_aos(&in3[k + 2], &in3[k + 3]);
        __m256d d2 = load2_aos(&in3[k + 4], &in3[k + 5]);
        __m256d d3 = load2_aos(&in3[k + 6], &in3[k + 7]);

        __m256d e0 = load2_aos(&in4[k], &in4[k + 1]);
        __m256d e1 = load2_aos(&in4[k + 2], &in4[k + 3]);
        __m256d e2 = load2_aos(&in4[k + 4], &in4[k + 5]);
        __m256d e3 = load2_aos(&in4[k + 6], &in4[k + 7]);

        __m256d w1_0 = load2_aos(&stage_tw[4 * k], &stage_tw[4 * (k + 1)]);
        __m256d w2_0 = load2_aos(&stage_tw[4 * k + 1], &stage_tw[4 * (k + 1) + 1]);
        __m256d w3_0 = load2_aos(&stage_tw[4 * k + 2], &stage_tw[4 * (k + 1) + 2]);
        __m256d w4_0 = load2_aos(&stage_tw[4 * k + 3], &stage_tw[4 * (k + 1) + 3]);

        __m256d w1_1 = load2_aos(&stage_tw[4 * (k + 2)], &stage_tw[4 * (k + 3)]);
        __m256d w2_1 = load2_aos(&stage_tw[4 * (k + 2) + 1], &stage_tw[4 * (k + 3) + 1]);
        __m256d w3_1 = load2_aos(&stage_tw[4 * (k + 2) + 2], &stage_tw[4 * (k + 3) + 2]);
        __m256d w4_1 = load2_aos(&stage_tw[4 * (k + 2) + 3], &stage_tw[4 * (k + 3) + 3]);

        __m256d w1_2 = load2_aos(&stage_tw[4 * (k + 4)], &stage_tw[4 * (k + 5)]);
        __m256d w2_2 = load2_aos(&stage_tw[4 * (k + 4) + 1], &stage_tw[4 * (k + 5) + 1]);
        __m256d w3_2 = load2_aos(&stage_tw[4 * (k + 4) + 2], &stage_tw[4 * (k + 5) + 2]);
        __m256d w4_2 = load2_aos(&stage_tw[4 * (k + 4) + 3], &stage_tw[4 * (k + 5) + 3]);

        __m256d w1_3 = load2_aos(&stage_tw[4 * (k + 6)], &stage_tw[4 * (k + 7)]);
        __m256d w2_3 = load2_aos(&stage_tw[4 * (k + 6) + 1], &stage_tw[4 * (k + 7) + 1]);
        __m256d w3_3 = load2_aos(&stage_tw[4 * (k + 6) + 2], &stage_tw[4 * (k + 7) + 2]);
        __m256d w4_3 = load2_aos(&stage_tw[4 * (k + 6) + 3], &stage_tw[4 * (k + 7) + 3]);

        __m256d b2_0 = cmul_avx2_aos(b0, w1_0);
        __m256d c2_0 = cmul_avx2_aos(c0, w2_0);
        __m256d d2_0 = cmul_avx2_aos(d0, w3_0);
        __m256d e2_0 = cmul_avx2_aos(e0, w4_0);

        __m256d b2_1 = cmul_avx2_aos(b1, w1_1);
        __m256d c2_1 = cmul_avx2_aos(c1, w2_1);
        __m256d d2_1 = cmul_avx2_aos(d1, w3_1);
        __m256d e2_1 = cmul_avx2_aos(e1, w4_1);

        __m256d b2_2 = cmul_avx2_aos(b2, w1_2);
        __m256d c2_2 = cmul_avx2_aos(c2, w2_2);
        __m256d d2_2 = cmul_avx2_aos(d2, w3_2);
        __m256d e2_2 = cmul_avx2_aos(e2, w4_2);

        __m256d b2_3 = cmul_avx2_aos(b3, w1_3);
        __m256d c2_3 = cmul_avx2_aos(c3, w2_3);
        __m256d d2_3 = cmul_avx2_aos(d3, w3_3);
        __m256d e2_3 = cmul_avx2_aos(e3, w4_3);

#define RADIX5_BUTTERFLY_AVX2(a, b2, c2, d2, e2, y0, y1, y2, y3, y4) \
    do {                                                             \
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
    } while(0)

        __m256d y0_0, y1_0, y2_0, y3_0, y4_0;
        __m256d y0_1, y1_1, y2_1, y3_1, y4_1;
        __m256d y0_2, y1_2, y2_2, y3_2, y4_2;
        __m256d y0_3, y1_3, y2_3, y3_3, y4_3;

        RADIX5_BUTTERFLY_AVX2(a0, b2_0, c2_0, d2_0, e2_0, y0_0, y1_0, y2_0, y3_0, y4_0);
        RADIX5_BUTTERFLY_AVX2(a1, b2_1, c2_1, d2_1, e2_1, y0_1, y1_1, y2_1, y3_1, y4_1);
        RADIX5_BUTTERFLY_AVX2(a2, b2_2, c2_2, d2_2, e2_2, y0_2, y1_2, y2_2, y3_2, y4_2);
        RADIX5_BUTTERFLY_AVX2(a3, b2_3, c2_3, d2_3, e2_3, y0_3, y1_3, y2_3, y3_3, y4_3);

#undef RADIX5_BUTTERFLY_AVX2

        STOREU_PD(&out0[k].re, y0_0);
        STOREU_PD(&out0[k + 2].re, y0_1);
        STOREU_PD(&out0[k + 4].re, y0_2);
        STOREU_PD(&out0[k + 6].re, y0_3);

        STOREU_PD(&out1[k].re, y1_0);
        STOREU_PD(&out1[k + 2].re, y1_1);
        STOREU_PD(&out1[k + 4].re, y1_2);
        STOREU_PD(&out1[k + 6].re, y1_3);

        STOREU_PD(&out2[k].re, y2_0);
        STOREU_PD(&out2[k + 2].re, y2_1);
        STOREU_PD(&out2[k + 4].re, y2_2);
        STOREU_PD(&out2[k + 6].re, y2_3);

        STOREU_PD(&out3[k].re, y3_0);
        STOREU_PD(&out3[k + 2].re, y3_1);
        STOREU_PD(&out3[k + 4].re, y3_2);
        STOREU_PD(&out3[k + 6].re, y3_3);

        STOREU_PD(&out4[k].re, y4_0);
        STOREU_PD(&out4[k + 2].re, y4_1);
        STOREU_PD(&out4[k + 4].re, y4_2);
        STOREU_PD(&out4[k + 6].re, y4_3);
    }

    //======================================================================
    // Cleanup: 2x unrolling
    //======================================================================
    for (; k + 1 < fifth; k += 2)
    {
        __m256d a = load2_aos(&in0[k], &in0[k + 1]);
        __m256d b = load2_aos(&in1[k], &in1[k + 1]);
        __m256d c = load2_aos(&in2[k], &in2[k + 1]);
        __m256d d = load2_aos(&in3[k], &in3[k + 1]);
        __m256d e = load2_aos(&in4[k], &in4[k + 1]);

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
        __m256d y3 = _mm256_add_pd(a2, r2);
        __m256d y2 = _mm256_sub_pd(a2, r2);

        STOREU_PD(&out0[k].re, y0);
        STOREU_PD(&out1[k].re, y1);
        STOREU_PD(&out2[k].re, y2);
        STOREU_PD(&out3[k].re, y3);
        STOREU_PD(&out4[k].re, y4);
    }
#endif // __AVX2__

    //======================================================================
    // Scalar tail
    //======================================================================
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
        // Exactly as in original document 2 (with the "FIX" applied)
        __m128d r1 = (transform_sign == 1)
                     ? _mm_xor_pd(base1_swp, _mm_set_pd(-0.0, 0.0))  // +i (SWAPPED)
                     : _mm_xor_pd(base1_swp, _mm_set_pd(0.0, -0.0)); // -i (SWAPPED)
                 
        __m128d a1 = _mm_add_pd(a, tmp1);
        __m128d y1 = _mm_add_pd(a1, r1);
        __m128d y4 = _mm_sub_pd(a1, r1);

        __m128d base2 = FMSUB_SSE2(vs2_128, t2, _mm_mul_pd(vs1_128, t3));
        __m128d tmp2 = FMADD_SSE2(vc2_128, t0, _mm_mul_pd(vc1_128, t1));
        __m128d base2_swp = _mm_shuffle_pd(base2, base2, 0b01);
        __m128d r2 = (transform_sign == 1)
                     ? _mm_xor_pd(base2_swp, _mm_set_pd(0.0, -0.0))  // -i (SWAPPED) 
                     : _mm_xor_pd(base2_swp, _mm_set_pd(-0.0, 0.0)); // +i (SWAPPED)
                     
        __m128d a2 = _mm_add_pd(a, tmp2);
        __m128d y3 = _mm_add_pd(a2, r2);
        __m128d y2 = _mm_sub_pd(a2, r2);

        STOREU_SSE2(&output_buffer[k + fifth].re, y1);
        STOREU_SSE2(&output_buffer[k + 2 * fifth].re, y2);
        STOREU_SSE2(&output_buffer[k + 3 * fifth].re, y3);
        STOREU_SSE2(&output_buffer[k + 4 * fifth].re, y4);
    }
}