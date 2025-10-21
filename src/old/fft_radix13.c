#include "fft_radix13.h" // ✅ Gets highSpeedFFT.h → fft_types.h
#include "simd_math.h"   // ✅ Gets complex math operations

void fft_radix13_butterfly(
    fft_data *output_buffer,
    fft_data *sub_outputs,
    const fft_data *stage_tw,
    int sub_len,
    int transform_sign)
{
    //======================================================================
    // RADIX-13 BUTTERFLY (Rader DIT), FFTW-style optimized, pure AoS
    //  - AVX2: 8x unrolled main loop, 2x cleanup
    //  - scalar tail for 0..1 leftover
    //  - per-stage twiddles: stage_tw[12*k + 0..11] for x1..x12 (DIT)
    //======================================================================
    const int thirteenth = sub_len;
    int k = 0;

    // Rader permutations for N=13 (generator g=2, inverse 7):
    //   perm_in  = [1,2,4,8,3,6,12,11,9,5,10,7]  (reorder inputs x1..x12)
    //   out_perm = [1,12,10,3,9,4,7,5,2,6,11,8]  (where each conv[q] lands)

    const double base_angle = (transform_sign == 1 ? -2.0 : +2.0) * M_PI / 13.0;

#ifdef __AVX2__
    //----------------------------------------------------------------------
    // AVX2 PATH
    //----------------------------------------------------------------------
    __m256d tw_brd[12];
    {
        const int op[12] = {1, 12, 10, 3, 9, 4, 7, 5, 2, 6, 11, 8};
        for (int q = 0; q < 12; ++q)
        {
            double a = op[q] * base_angle;
            double wr, wi;
#ifdef __GNUC__
            sincos(a, &wi, &wr);
#else
            wr = cos(a);
            wi = sin(a);
#endif
            tw_brd[q] = _mm256_set_pd(wi, wr, wi, wr);
        }
    }

// Macro for 12-point cyclic convolution
#define CONV12_Q(q, v, tx0, tx1, tx2, tx3, tx4, tx5, tx6, tx7, tx8, tx9, tx10, tx11, tw_brd) \
    do                                                                                       \
    {                                                                                        \
        v##_0 = cmul_avx2_aos(tx0##_0, tw_brd[(q + 0) % 12]);                                \
        v##_1 = cmul_avx2_aos(tx0##_1, tw_brd[(q + 0) % 12]);                                \
        v##_2 = cmul_avx2_aos(tx0##_2, tw_brd[(q + 0) % 12]);                                \
        v##_3 = cmul_avx2_aos(tx0##_3, tw_brd[(q + 0) % 12]);                                \
        v##_0 = _mm256_add_pd(v##_0, cmul_avx2_aos(tx1##_0, tw_brd[(q + 11) % 12]));         \
        v##_1 = _mm256_add_pd(v##_1, cmul_avx2_aos(tx1##_1, tw_brd[(q + 11) % 12]));         \
        v##_2 = _mm256_add_pd(v##_2, cmul_avx2_aos(tx1##_2, tw_brd[(q + 11) % 12]));         \
        v##_3 = _mm256_add_pd(v##_3, cmul_avx2_aos(tx1##_3, tw_brd[(q + 11) % 12]));         \
        v##_0 = _mm256_add_pd(v##_0, cmul_avx2_aos(tx2##_0, tw_brd[(q + 10) % 12]));         \
        v##_1 = _mm256_add_pd(v##_1, cmul_avx2_aos(tx2##_1, tw_brd[(q + 10) % 12]));         \
        v##_2 = _mm256_add_pd(v##_2, cmul_avx2_aos(tx2##_2, tw_brd[(q + 10) % 12]));         \
        v##_3 = _mm256_add_pd(v##_3, cmul_avx2_aos(tx2##_3, tw_brd[(q + 10) % 12]));         \
        v##_0 = _mm256_add_pd(v##_0, cmul_avx2_aos(tx3##_0, tw_brd[(q + 9) % 12]));          \
        v##_1 = _mm256_add_pd(v##_1, cmul_avx2_aos(tx3##_1, tw_brd[(q + 9) % 12]));          \
        v##_2 = _mm256_add_pd(v##_2, cmul_avx2_aos(tx3##_2, tw_brd[(q + 9) % 12]));          \
        v##_3 = _mm256_add_pd(v##_3, cmul_avx2_aos(tx3##_3, tw_brd[(q + 9) % 12]));          \
        v##_0 = _mm256_add_pd(v##_0, cmul_avx2_aos(tx4##_0, tw_brd[(q + 8) % 12]));          \
        v##_1 = _mm256_add_pd(v##_1, cmul_avx2_aos(tx4##_1, tw_brd[(q + 8) % 12]));          \
        v##_2 = _mm256_add_pd(v##_2, cmul_avx2_aos(tx4##_2, tw_brd[(q + 8) % 12]));          \
        v##_3 = _mm256_add_pd(v##_3, cmul_avx2_aos(tx4##_3, tw_brd[(q + 8) % 12]));          \
        v##_0 = _mm256_add_pd(v##_0, cmul_avx2_aos(tx5##_0, tw_brd[(q + 7) % 12]));          \
        v##_1 = _mm256_add_pd(v##_1, cmul_avx2_aos(tx5##_1, tw_brd[(q + 7) % 12]));          \
        v##_2 = _mm256_add_pd(v##_2, cmul_avx2_aos(tx5##_2, tw_brd[(q + 7) % 12]));          \
        v##_3 = _mm256_add_pd(v##_3, cmul_avx2_aos(tx5##_3, tw_brd[(q + 7) % 12]));          \
        v##_0 = _mm256_add_pd(v##_0, cmul_avx2_aos(tx6##_0, tw_brd[(q + 6) % 12]));          \
        v##_1 = _mm256_add_pd(v##_1, cmul_avx2_aos(tx6##_1, tw_brd[(q + 6) % 12]));          \
        v##_2 = _mm256_add_pd(v##_2, cmul_avx2_aos(tx6##_2, tw_brd[(q + 6) % 12]));          \
        v##_3 = _mm256_add_pd(v##_3, cmul_avx2_aos(tx6##_3, tw_brd[(q + 6) % 12]));          \
        v##_0 = _mm256_add_pd(v##_0, cmul_avx2_aos(tx7##_0, tw_brd[(q + 5) % 12]));          \
        v##_1 = _mm256_add_pd(v##_1, cmul_avx2_aos(tx7##_1, tw_brd[(q + 5) % 12]));          \
        v##_2 = _mm256_add_pd(v##_2, cmul_avx2_aos(tx7##_2, tw_brd[(q + 5) % 12]));          \
        v##_3 = _mm256_add_pd(v##_3, cmul_avx2_aos(tx7##_3, tw_brd[(q + 5) % 12]));          \
        v##_0 = _mm256_add_pd(v##_0, cmul_avx2_aos(tx8##_0, tw_brd[(q + 4) % 12]));          \
        v##_1 = _mm256_add_pd(v##_1, cmul_avx2_aos(tx8##_1, tw_brd[(q + 4) % 12]));          \
        v##_2 = _mm256_add_pd(v##_2, cmul_avx2_aos(tx8##_2, tw_brd[(q + 4) % 12]));          \
        v##_3 = _mm256_add_pd(v##_3, cmul_avx2_aos(tx8##_3, tw_brd[(q + 4) % 12]));          \
        v##_0 = _mm256_add_pd(v##_0, cmul_avx2_aos(tx9##_0, tw_brd[(q + 3) % 12]));          \
        v##_1 = _mm256_add_pd(v##_1, cmul_avx2_aos(tx9##_1, tw_brd[(q + 3) % 12]));          \
        v##_2 = _mm256_add_pd(v##_2, cmul_avx2_aos(tx9##_2, tw_brd[(q + 3) % 12]));          \
        v##_3 = _mm256_add_pd(v##_3, cmul_avx2_aos(tx9##_3, tw_brd[(q + 3) % 12]));          \
        v##_0 = _mm256_add_pd(v##_0, cmul_avx2_aos(tx10##_0, tw_brd[(q + 2) % 12]));         \
        v##_1 = _mm256_add_pd(v##_1, cmul_avx2_aos(tx10##_1, tw_brd[(q + 2) % 12]));         \
        v##_2 = _mm256_add_pd(v##_2, cmul_avx2_aos(tx10##_2, tw_brd[(q + 2) % 12]));         \
        v##_3 = _mm256_add_pd(v##_3, cmul_avx2_aos(tx10##_3, tw_brd[(q + 2) % 12]));         \
        v##_0 = _mm256_add_pd(v##_0, cmul_avx2_aos(tx11##_0, tw_brd[(q + 1) % 12]));         \
        v##_1 = _mm256_add_pd(v##_1, cmul_avx2_aos(tx11##_1, tw_brd[(q + 1) % 12]));         \
        v##_2 = _mm256_add_pd(v##_2, cmul_avx2_aos(tx11##_2, tw_brd[(q + 1) % 12]));         \
        v##_3 = _mm256_add_pd(v##_3, cmul_avx2_aos(tx11##_3, tw_brd[(q + 1) % 12]));         \
    } while (0)

// Macro for mapping convolution outputs (out_perm)
#define MAP_CONV_TO_OUTPUT_13(x0, v, y, suffix)                   \
    do                                                            \
    {                                                             \
        y##1##suffix = _mm256_add_pd(x0##suffix, v##0##suffix);   \
        y##12##suffix = _mm256_add_pd(x0##suffix, v##1##suffix);  \
        y##10##suffix = _mm256_add_pd(x0##suffix, v##2##suffix);  \
        y##3##suffix = _mm256_add_pd(x0##suffix, v##3##suffix);   \
        y##9##suffix = _mm256_add_pd(x0##suffix, v##4##suffix);   \
        y##4##suffix = _mm256_add_pd(x0##suffix, v##5##suffix);   \
        y##7##suffix = _mm256_add_pd(x0##suffix, v##6##suffix);   \
        y##5##suffix = _mm256_add_pd(x0##suffix, v##7##suffix);   \
        y##2##suffix = _mm256_add_pd(x0##suffix, v##8##suffix);   \
        y##6##suffix = _mm256_add_pd(x0##suffix, v##9##suffix);   \
        y##11##suffix = _mm256_add_pd(x0##suffix, v##10##suffix); \
        y##8##suffix = _mm256_add_pd(x0##suffix, v##11##suffix);  \
    } while (0)

    // -----------------------------
    // 8x unrolled main loop
    // -----------------------------
    for (; k + 7 < thirteenth; k += 8)
    {
        if (k + 16 < thirteenth)
        {
            _mm_prefetch((const char *)&sub_outputs[k + 16].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_outputs[k + 16 + thirteenth].re, _MM_HINT_T0);
            if (thirteenth > 1)
                _mm_prefetch((const char *)&stage_tw[12 * (k + 16)].re, _MM_HINT_T0);
        }

        // Load x0..x12 for 8 butterflies
        __m256d x0_0 = load2_aos(&sub_outputs[k + 0 * thirteenth + 0], &sub_outputs[k + 0 * thirteenth + 1]);
        __m256d x0_1 = load2_aos(&sub_outputs[k + 0 * thirteenth + 2], &sub_outputs[k + 0 * thirteenth + 3]);
        __m256d x0_2 = load2_aos(&sub_outputs[k + 0 * thirteenth + 4], &sub_outputs[k + 0 * thirteenth + 5]);
        __m256d x0_3 = load2_aos(&sub_outputs[k + 0 * thirteenth + 6], &sub_outputs[k + 0 * thirteenth + 7]);

        __m256d x1_0 = load2_aos(&sub_outputs[k + 1 * thirteenth + 0], &sub_outputs[k + 1 * thirteenth + 1]);
        __m256d x1_1 = load2_aos(&sub_outputs[k + 1 * thirteenth + 2], &sub_outputs[k + 1 * thirteenth + 3]);
        __m256d x1_2 = load2_aos(&sub_outputs[k + 1 * thirteenth + 4], &sub_outputs[k + 1 * thirteenth + 5]);
        __m256d x1_3 = load2_aos(&sub_outputs[k + 1 * thirteenth + 6], &sub_outputs[k + 1 * thirteenth + 7]);

        __m256d x2_0 = load2_aos(&sub_outputs[k + 2 * thirteenth + 0], &sub_outputs[k + 2 * thirteenth + 1]);
        __m256d x2_1 = load2_aos(&sub_outputs[k + 2 * thirteenth + 2], &sub_outputs[k + 2 * thirteenth + 3]);
        __m256d x2_2 = load2_aos(&sub_outputs[k + 2 * thirteenth + 4], &sub_outputs[k + 2 * thirteenth + 5]);
        __m256d x2_3 = load2_aos(&sub_outputs[k + 2 * thirteenth + 6], &sub_outputs[k + 2 * thirteenth + 7]);

        __m256d x3_0 = load2_aos(&sub_outputs[k + 3 * thirteenth + 0], &sub_outputs[k + 3 * thirteenth + 1]);
        __m256d x3_1 = load2_aos(&sub_outputs[k + 3 * thirteenth + 2], &sub_outputs[k + 3 * thirteenth + 3]);
        __m256d x3_2 = load2_aos(&sub_outputs[k + 3 * thirteenth + 4], &sub_outputs[k + 3 * thirteenth + 5]);
        __m256d x3_3 = load2_aos(&sub_outputs[k + 3 * thirteenth + 6], &sub_outputs[k + 3 * thirteenth + 7]);

        __m256d x4_0 = load2_aos(&sub_outputs[k + 4 * thirteenth + 0], &sub_outputs[k + 4 * thirteenth + 1]);
        __m256d x4_1 = load2_aos(&sub_outputs[k + 4 * thirteenth + 2], &sub_outputs[k + 4 * thirteenth + 3]);
        __m256d x4_2 = load2_aos(&sub_outputs[k + 4 * thirteenth + 4], &sub_outputs[k + 4 * thirteenth + 5]);
        __m256d x4_3 = load2_aos(&sub_outputs[k + 4 * thirteenth + 6], &sub_outputs[k + 4 * thirteenth + 7]);

        __m256d x5_0 = load2_aos(&sub_outputs[k + 5 * thirteenth + 0], &sub_outputs[k + 5 * thirteenth + 1]);
        __m256d x5_1 = load2_aos(&sub_outputs[k + 5 * thirteenth + 2], &sub_outputs[k + 5 * thirteenth + 3]);
        __m256d x5_2 = load2_aos(&sub_outputs[k + 5 * thirteenth + 4], &sub_outputs[k + 5 * thirteenth + 5]);
        __m256d x5_3 = load2_aos(&sub_outputs[k + 5 * thirteenth + 6], &sub_outputs[k + 5 * thirteenth + 7]);

        __m256d x6_0 = load2_aos(&sub_outputs[k + 6 * thirteenth + 0], &sub_outputs[k + 6 * thirteenth + 1]);
        __m256d x6_1 = load2_aos(&sub_outputs[k + 6 * thirteenth + 2], &sub_outputs[k + 6 * thirteenth + 3]);
        __m256d x6_2 = load2_aos(&sub_outputs[k + 6 * thirteenth + 4], &sub_outputs[k + 6 * thirteenth + 5]);
        __m256d x6_3 = load2_aos(&sub_outputs[k + 6 * thirteenth + 6], &sub_outputs[k + 6 * thirteenth + 7]);

        __m256d x7_0 = load2_aos(&sub_outputs[k + 7 * thirteenth + 0], &sub_outputs[k + 7 * thirteenth + 1]);
        __m256d x7_1 = load2_aos(&sub_outputs[k + 7 * thirteenth + 2], &sub_outputs[k + 7 * thirteenth + 3]);
        __m256d x7_2 = load2_aos(&sub_outputs[k + 7 * thirteenth + 4], &sub_outputs[k + 7 * thirteenth + 5]);
        __m256d x7_3 = load2_aos(&sub_outputs[k + 7 * thirteenth + 6], &sub_outputs[k + 7 * thirteenth + 7]);

        __m256d x8_0 = load2_aos(&sub_outputs[k + 8 * thirteenth + 0], &sub_outputs[k + 8 * thirteenth + 1]);
        __m256d x8_1 = load2_aos(&sub_outputs[k + 8 * thirteenth + 2], &sub_outputs[k + 8 * thirteenth + 3]);
        __m256d x8_2 = load2_aos(&sub_outputs[k + 8 * thirteenth + 4], &sub_outputs[k + 8 * thirteenth + 5]);
        __m256d x8_3 = load2_aos(&sub_outputs[k + 8 * thirteenth + 6], &sub_outputs[k + 8 * thirteenth + 7]);

        __m256d x9_0 = load2_aos(&sub_outputs[k + 9 * thirteenth + 0], &sub_outputs[k + 9 * thirteenth + 1]);
        __m256d x9_1 = load2_aos(&sub_outputs[k + 9 * thirteenth + 2], &sub_outputs[k + 9 * thirteenth + 3]);
        __m256d x9_2 = load2_aos(&sub_outputs[k + 9 * thirteenth + 4], &sub_outputs[k + 9 * thirteenth + 5]);
        __m256d x9_3 = load2_aos(&sub_outputs[k + 9 * thirteenth + 6], &sub_outputs[k + 9 * thirteenth + 7]);

        __m256d x10_0 = load2_aos(&sub_outputs[k + 10 * thirteenth + 0], &sub_outputs[k + 10 * thirteenth + 1]);
        __m256d x10_1 = load2_aos(&sub_outputs[k + 10 * thirteenth + 2], &sub_outputs[k + 10 * thirteenth + 3]);
        __m256d x10_2 = load2_aos(&sub_outputs[k + 10 * thirteenth + 4], &sub_outputs[k + 10 * thirteenth + 5]);
        __m256d x10_3 = load2_aos(&sub_outputs[k + 10 * thirteenth + 6], &sub_outputs[k + 10 * thirteenth + 7]);

        __m256d x11_0 = load2_aos(&sub_outputs[k + 11 * thirteenth + 0], &sub_outputs[k + 11 * thirteenth + 1]);
        __m256d x11_1 = load2_aos(&sub_outputs[k + 11 * thirteenth + 2], &sub_outputs[k + 11 * thirteenth + 3]);
        __m256d x11_2 = load2_aos(&sub_outputs[k + 11 * thirteenth + 4], &sub_outputs[k + 11 * thirteenth + 5]);
        __m256d x11_3 = load2_aos(&sub_outputs[k + 11 * thirteenth + 6], &sub_outputs[k + 11 * thirteenth + 7]);

        __m256d x12_0 = load2_aos(&sub_outputs[k + 12 * thirteenth + 0], &sub_outputs[k + 12 * thirteenth + 1]);
        __m256d x12_1 = load2_aos(&sub_outputs[k + 12 * thirteenth + 2], &sub_outputs[k + 12 * thirteenth + 3]);
        __m256d x12_2 = load2_aos(&sub_outputs[k + 12 * thirteenth + 4], &sub_outputs[k + 12 * thirteenth + 5]);
        __m256d x12_3 = load2_aos(&sub_outputs[k + 12 * thirteenth + 6], &sub_outputs[k + 12 * thirteenth + 7]);

        // Apply per-stage DIT twiddles (if multi-stage)
        if (thirteenth > 1)
        {
            __m256d w1_0 = load2_aos(&stage_tw[12 * (k + 0) + 0], &stage_tw[12 * (k + 1) + 0]);
            __m256d w1_1 = load2_aos(&stage_tw[12 * (k + 2) + 0], &stage_tw[12 * (k + 3) + 0]);
            __m256d w1_2 = load2_aos(&stage_tw[12 * (k + 4) + 0], &stage_tw[12 * (k + 5) + 0]);
            __m256d w1_3 = load2_aos(&stage_tw[12 * (k + 6) + 0], &stage_tw[12 * (k + 7) + 0]);

            __m256d w2_0 = load2_aos(&stage_tw[12 * (k + 0) + 1], &stage_tw[12 * (k + 1) + 1]);
            __m256d w2_1 = load2_aos(&stage_tw[12 * (k + 2) + 1], &stage_tw[12 * (k + 3) + 1]);
            __m256d w2_2 = load2_aos(&stage_tw[12 * (k + 4) + 1], &stage_tw[12 * (k + 5) + 1]);
            __m256d w2_3 = load2_aos(&stage_tw[12 * (k + 6) + 1], &stage_tw[12 * (k + 7) + 1]);

            __m256d w3_0 = load2_aos(&stage_tw[12 * (k + 0) + 2], &stage_tw[12 * (k + 1) + 2]);
            __m256d w3_1 = load2_aos(&stage_tw[12 * (k + 2) + 2], &stage_tw[12 * (k + 3) + 2]);
            __m256d w3_2 = load2_aos(&stage_tw[12 * (k + 4) + 2], &stage_tw[12 * (k + 5) + 2]);
            __m256d w3_3 = load2_aos(&stage_tw[12 * (k + 6) + 2], &stage_tw[12 * (k + 7) + 2]);

            __m256d w4_0 = load2_aos(&stage_tw[12 * (k + 0) + 3], &stage_tw[12 * (k + 1) + 3]);
            __m256d w4_1 = load2_aos(&stage_tw[12 * (k + 2) + 3], &stage_tw[12 * (k + 3) + 3]);
            __m256d w4_2 = load2_aos(&stage_tw[12 * (k + 4) + 3], &stage_tw[12 * (k + 5) + 3]);
            __m256d w4_3 = load2_aos(&stage_tw[12 * (k + 6) + 3], &stage_tw[12 * (k + 7) + 3]);

            __m256d w5_0 = load2_aos(&stage_tw[12 * (k + 0) + 4], &stage_tw[12 * (k + 1) + 4]);
            __m256d w5_1 = load2_aos(&stage_tw[12 * (k + 2) + 4], &stage_tw[12 * (k + 3) + 4]);
            __m256d w5_2 = load2_aos(&stage_tw[12 * (k + 4) + 4], &stage_tw[12 * (k + 5) + 4]);
            __m256d w5_3 = load2_aos(&stage_tw[12 * (k + 6) + 4], &stage_tw[12 * (k + 7) + 4]);

            __m256d w6_0 = load2_aos(&stage_tw[12 * (k + 0) + 5], &stage_tw[12 * (k + 1) + 5]);
            __m256d w6_1 = load2_aos(&stage_tw[12 * (k + 2) + 5], &stage_tw[12 * (k + 3) + 5]);
            __m256d w6_2 = load2_aos(&stage_tw[12 * (k + 4) + 5], &stage_tw[12 * (k + 5) + 5]);
            __m256d w6_3 = load2_aos(&stage_tw[12 * (k + 6) + 5], &stage_tw[12 * (k + 7) + 5]);

            __m256d w7_0 = load2_aos(&stage_tw[12 * (k + 0) + 6], &stage_tw[12 * (k + 1) + 6]);
            __m256d w7_1 = load2_aos(&stage_tw[12 * (k + 2) + 6], &stage_tw[12 * (k + 3) + 6]);
            __m256d w7_2 = load2_aos(&stage_tw[12 * (k + 4) + 6], &stage_tw[12 * (k + 5) + 6]);
            __m256d w7_3 = load2_aos(&stage_tw[12 * (k + 6) + 6], &stage_tw[12 * (k + 7) + 6]);

            __m256d w8_0 = load2_aos(&stage_tw[12 * (k + 0) + 7], &stage_tw[12 * (k + 1) + 7]);
            __m256d w8_1 = load2_aos(&stage_tw[12 * (k + 2) + 7], &stage_tw[12 * (k + 3) + 7]);
            __m256d w8_2 = load2_aos(&stage_tw[12 * (k + 4) + 7], &stage_tw[12 * (k + 5) + 7]);
            __m256d w8_3 = load2_aos(&stage_tw[12 * (k + 6) + 7], &stage_tw[12 * (k + 7) + 7]);

            __m256d w9_0 = load2_aos(&stage_tw[12 * (k + 0) + 8], &stage_tw[12 * (k + 1) + 8]);
            __m256d w9_1 = load2_aos(&stage_tw[12 * (k + 2) + 8], &stage_tw[12 * (k + 3) + 8]);
            __m256d w9_2 = load2_aos(&stage_tw[12 * (k + 4) + 8], &stage_tw[12 * (k + 5) + 8]);
            __m256d w9_3 = load2_aos(&stage_tw[12 * (k + 6) + 8], &stage_tw[12 * (k + 7) + 8]);

            __m256d w10_0 = load2_aos(&stage_tw[12 * (k + 0) + 9], &stage_tw[12 * (k + 1) + 9]);
            __m256d w10_1 = load2_aos(&stage_tw[12 * (k + 2) + 9], &stage_tw[12 * (k + 3) + 9]);
            __m256d w10_2 = load2_aos(&stage_tw[12 * (k + 4) + 9], &stage_tw[12 * (k + 5) + 9]);
            __m256d w10_3 = load2_aos(&stage_tw[12 * (k + 6) + 9], &stage_tw[12 * (k + 7) + 9]);

            __m256d w11_0 = load2_aos(&stage_tw[12 * (k + 0) + 10], &stage_tw[12 * (k + 1) + 10]);
            __m256d w11_1 = load2_aos(&stage_tw[12 * (k + 2) + 10], &stage_tw[12 * (k + 3) + 10]);
            __m256d w11_2 = load2_aos(&stage_tw[12 * (k + 4) + 10], &stage_tw[12 * (k + 5) + 10]);
            __m256d w11_3 = load2_aos(&stage_tw[12 * (k + 6) + 10], &stage_tw[12 * (k + 7) + 10]);

            __m256d w12_0 = load2_aos(&stage_tw[12 * (k + 0) + 11], &stage_tw[12 * (k + 1) + 11]);
            __m256d w12_1 = load2_aos(&stage_tw[12 * (k + 2) + 11], &stage_tw[12 * (k + 3) + 11]);
            __m256d w12_2 = load2_aos(&stage_tw[12 * (k + 4) + 11], &stage_tw[12 * (k + 5) + 11]);
            __m256d w12_3 = load2_aos(&stage_tw[12 * (k + 6) + 11], &stage_tw[12 * (k + 7) + 11]);

            x1_0 = cmul_avx2_aos(x1_0, w1_0);
            x1_1 = cmul_avx2_aos(x1_1, w1_1);
            x1_2 = cmul_avx2_aos(x1_2, w1_2);
            x1_3 = cmul_avx2_aos(x1_3, w1_3);

            x2_0 = cmul_avx2_aos(x2_0, w2_0);
            x2_1 = cmul_avx2_aos(x2_1, w2_1);
            x2_2 = cmul_avx2_aos(x2_2, w2_2);
            x2_3 = cmul_avx2_aos(x2_3, w2_3);

            x3_0 = cmul_avx2_aos(x3_0, w3_0);
            x3_1 = cmul_avx2_aos(x3_1, w3_1);
            x3_2 = cmul_avx2_aos(x3_2, w3_2);
            x3_3 = cmul_avx2_aos(x3_3, w3_3);

            x4_0 = cmul_avx2_aos(x4_0, w4_0);
            x4_1 = cmul_avx2_aos(x4_1, w4_1);
            x4_2 = cmul_avx2_aos(x4_2, w4_2);
            x4_3 = cmul_avx2_aos(x4_3, w4_3);

            x5_0 = cmul_avx2_aos(x5_0, w5_0);
            x5_1 = cmul_avx2_aos(x5_1, w5_1);
            x5_2 = cmul_avx2_aos(x5_2, w5_2);
            x5_3 = cmul_avx2_aos(x5_3, w5_3);

            x6_0 = cmul_avx2_aos(x6_0, w6_0);
            x6_1 = cmul_avx2_aos(x6_1, w6_1);
            x6_2 = cmul_avx2_aos(x6_2, w6_2);
            x6_3 = cmul_avx2_aos(x6_3, w6_3);

            x7_0 = cmul_avx2_aos(x7_0, w7_0);
            x7_1 = cmul_avx2_aos(x7_1, w7_1);
            x7_2 = cmul_avx2_aos(x7_2, w7_2);
            x7_3 = cmul_avx2_aos(x7_3, w7_3);

            x8_0 = cmul_avx2_aos(x8_0, w8_0);
            x8_1 = cmul_avx2_aos(x8_1, w8_1);
            x8_2 = cmul_avx2_aos(x8_2, w8_2);
            x8_3 = cmul_avx2_aos(x8_3, w8_3);

            x9_0 = cmul_avx2_aos(x9_0, w9_0);
            x9_1 = cmul_avx2_aos(x9_1, w9_1);
            x9_2 = cmul_avx2_aos(x9_2, w9_2);
            x9_3 = cmul_avx2_aos(x9_3, w9_3);

            x10_0 = cmul_avx2_aos(x10_0, w10_0);
            x10_1 = cmul_avx2_aos(x10_1, w10_1);
            x10_2 = cmul_avx2_aos(x10_2, w10_2);
            x10_3 = cmul_avx2_aos(x10_3, w10_3);

            x11_0 = cmul_avx2_aos(x11_0, w11_0);
            x11_1 = cmul_avx2_aos(x11_1, w11_1);
            x11_2 = cmul_avx2_aos(x11_2, w11_2);
            x11_3 = cmul_avx2_aos(x11_3, w11_3);

            x12_0 = cmul_avx2_aos(x12_0, w12_0);
            x12_1 = cmul_avx2_aos(x12_1, w12_1);
            x12_2 = cmul_avx2_aos(x12_2, w12_2);
            x12_3 = cmul_avx2_aos(x12_3, w12_3);
        }

        // y0 = sum(x0..x12)
        __m256d y0_0 = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(x0_0, x1_0), _mm256_add_pd(x2_0, x3_0)),
                                                   _mm256_add_pd(_mm256_add_pd(x4_0, x5_0), _mm256_add_pd(x6_0, x7_0))),
                                     _mm256_add_pd(_mm256_add_pd(x8_0, x9_0), _mm256_add_pd(_mm256_add_pd(x10_0, x11_0), x12_0)));
        __m256d y0_1 = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(x0_1, x1_1), _mm256_add_pd(x2_1, x3_1)),
                                                   _mm256_add_pd(_mm256_add_pd(x4_1, x5_1), _mm256_add_pd(x6_1, x7_1))),
                                     _mm256_add_pd(_mm256_add_pd(x8_1, x9_1), _mm256_add_pd(_mm256_add_pd(x10_1, x11_1), x12_1)));
        __m256d y0_2 = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(x0_2, x1_2), _mm256_add_pd(x2_2, x3_2)),
                                                   _mm256_add_pd(_mm256_add_pd(x4_2, x5_2), _mm256_add_pd(x6_2, x7_2))),
                                     _mm256_add_pd(_mm256_add_pd(x8_2, x9_2), _mm256_add_pd(_mm256_add_pd(x10_2, x11_2), x12_2)));
        __m256d y0_3 = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(x0_3, x1_3), _mm256_add_pd(x2_3, x3_3)),
                                                   _mm256_add_pd(_mm256_add_pd(x4_3, x5_3), _mm256_add_pd(x6_3, x7_3))),
                                     _mm256_add_pd(_mm256_add_pd(x8_3, x9_3), _mm256_add_pd(_mm256_add_pd(x10_3, x11_3), x12_3)));

        // Rader input permute: tx = [x1, x2, x4, x8, x3, x6, x12, x11, x9, x5, x10, x7]
        __m256d tx0_0 = x1_0, tx1_0 = x2_0, tx2_0 = x4_0, tx3_0 = x8_0, tx4_0 = x3_0, tx5_0 = x6_0;
        __m256d tx6_0 = x12_0, tx7_0 = x11_0, tx8_0 = x9_0, tx9_0 = x5_0, tx10_0 = x10_0, tx11_0 = x7_0;

        __m256d tx0_1 = x1_1, tx1_1 = x2_1, tx2_1 = x4_1, tx3_1 = x8_1, tx4_1 = x3_1, tx5_1 = x6_1;
        __m256d tx6_1 = x12_1, tx7_1 = x11_1, tx8_1 = x9_1, tx9_1 = x5_1, tx10_1 = x10_1, tx11_1 = x7_1;

        __m256d tx0_2 = x1_2, tx1_2 = x2_2, tx2_2 = x4_2, tx3_2 = x8_2, tx4_2 = x3_2, tx5_2 = x6_2;
        __m256d tx6_2 = x12_2, tx7_2 = x11_2, tx8_2 = x9_2, tx9_2 = x5_2, tx10_2 = x10_2, tx11_2 = x7_2;

        __m256d tx0_3 = x1_3, tx1_3 = x2_3, tx2_3 = x4_3, tx3_3 = x8_3, tx4_3 = x3_3, tx5_3 = x6_3;
        __m256d tx6_3 = x12_3, tx7_3 = x11_3, tx8_3 = x9_3, tx9_3 = x5_3, tx10_3 = x10_3, tx11_3 = x7_3;

        // 12-pt cyclic convolution
        __m256d v0_0, v0_1, v0_2, v0_3;
        __m256d v1_0, v1_1, v1_2, v1_3;
        __m256d v2_0, v2_1, v2_2, v2_3;
        __m256d v3_0, v3_1, v3_2, v3_3;
        __m256d v4_0, v4_1, v4_2, v4_3;
        __m256d v5_0, v5_1, v5_2, v5_3;
        __m256d v6_0, v6_1, v6_2, v6_3;
        __m256d v7_0, v7_1, v7_2, v7_3;
        __m256d v8_0, v8_1, v8_2, v8_3;
        __m256d v9_0, v9_1, v9_2, v9_3;
        __m256d v10_0, v10_1, v10_2, v10_3;
        __m256d v11_0, v11_1, v11_2, v11_3;

        CONV12_Q(0, v0, tx0, tx1, tx2, tx3, tx4, tx5, tx6, tx7, tx8, tx9, tx10, tx11, tw_brd);
        CONV12_Q(1, v1, tx0, tx1, tx2, tx3, tx4, tx5, tx6, tx7, tx8, tx9, tx10, tx11, tw_brd);
        CONV12_Q(2, v2, tx0, tx1, tx2, tx3, tx4, tx5, tx6, tx7, tx8, tx9, tx10, tx11, tw_brd);
        CONV12_Q(3, v3, tx0, tx1, tx2, tx3, tx4, tx5, tx6, tx7, tx8, tx9, tx10, tx11, tw_brd);
        CONV12_Q(4, v4, tx0, tx1, tx2, tx3, tx4, tx5, tx6, tx7, tx8, tx9, tx10, tx11, tw_brd);
        CONV12_Q(5, v5, tx0, tx1, tx2, tx3, tx4, tx5, tx6, tx7, tx8, tx9, tx10, tx11, tw_brd);
        CONV12_Q(6, v6, tx0, tx1, tx2, tx3, tx4, tx5, tx6, tx7, tx8, tx9, tx10, tx11, tw_brd);
        CONV12_Q(7, v7, tx0, tx1, tx2, tx3, tx4, tx5, tx6, tx7, tx8, tx9, tx10, tx11, tw_brd);
        CONV12_Q(8, v8, tx0, tx1, tx2, tx3, tx4, tx5, tx6, tx7, tx8, tx9, tx10, tx11, tw_brd);
        CONV12_Q(9, v9, tx0, tx1, tx2, tx3, tx4, tx5, tx6, tx7, tx8, tx9, tx10, tx11, tw_brd);
        CONV12_Q(10, v10, tx0, tx1, tx2, tx3, tx4, tx5, tx6, tx7, tx8, tx9, tx10, tx11, tw_brd);
        CONV12_Q(11, v11, tx0, tx1, tx2, tx3, tx4, tx5, tx6, tx7, tx8, tx9, tx10, tx11, tw_brd);

        // Map to outputs
        __m256d y1_0, y1_1, y1_2, y1_3;
        __m256d y2_0, y2_1, y2_2, y2_3;
        __m256d y3_0, y3_1, y3_2, y3_3;
        __m256d y4_0, y4_1, y4_2, y4_3;
        __m256d y5_0, y5_1, y5_2, y5_3;
        __m256d y6_0, y6_1, y6_2, y6_3;
        __m256d y7_0, y7_1, y7_2, y7_3;
        __m256d y8_0, y8_1, y8_2, y8_3;
        __m256d y9_0, y9_1, y9_2, y9_3;
        __m256d y10_0, y10_1, y10_2, y10_3;
        __m256d y11_0, y11_1, y11_2, y11_3;
        __m256d y12_0, y12_1, y12_2, y12_3;

        MAP_CONV_TO_OUTPUT_13(x0, v, y, _0);
        MAP_CONV_TO_OUTPUT_13(x0, v, y, _1);
        MAP_CONV_TO_OUTPUT_13(x0, v, y, _2);
        MAP_CONV_TO_OUTPUT_13(x0, v, y, _3);

        // Store
        STOREU_PD(&output_buffer[k + 0 * thirteenth + 0].re, y0_0);
        STOREU_PD(&output_buffer[k + 0 * thirteenth + 2].re, y0_1);
        STOREU_PD(&output_buffer[k + 0 * thirteenth + 4].re, y0_2);
        STOREU_PD(&output_buffer[k + 0 * thirteenth + 6].re, y0_3);

        STOREU_PD(&output_buffer[k + 1 * thirteenth + 0].re, y1_0);
        STOREU_PD(&output_buffer[k + 1 * thirteenth + 2].re, y1_1);
        STOREU_PD(&output_buffer[k + 1 * thirteenth + 4].re, y1_2);
        STOREU_PD(&output_buffer[k + 1 * thirteenth + 6].re, y1_3);

        STOREU_PD(&output_buffer[k + 2 * thirteenth + 0].re, y2_0);
        STOREU_PD(&output_buffer[k + 2 * thirteenth + 2].re, y2_1);
        STOREU_PD(&output_buffer[k + 2 * thirteenth + 4].re, y2_2);
        STOREU_PD(&output_buffer[k + 2 * thirteenth + 6].re, y2_3);

        STOREU_PD(&output_buffer[k + 3 * thirteenth + 0].re, y3_0);
        STOREU_PD(&output_buffer[k + 3 * thirteenth + 2].re, y3_1);
        STOREU_PD(&output_buffer[k + 3 * thirteenth + 4].re, y3_2);
        STOREU_PD(&output_buffer[k + 3 * thirteenth + 6].re, y3_3);

        STOREU_PD(&output_buffer[k + 4 * thirteenth + 0].re, y4_0);
        STOREU_PD(&output_buffer[k + 4 * thirteenth + 2].re, y4_1);
        STOREU_PD(&output_buffer[k + 4 * thirteenth + 4].re, y4_2);
        STOREU_PD(&output_buffer[k + 4 * thirteenth + 6].re, y4_3);

        STOREU_PD(&output_buffer[k + 5 * thirteenth + 0].re, y5_0);
        STOREU_PD(&output_buffer[k + 5 * thirteenth + 2].re, y5_1);
        STOREU_PD(&output_buffer[k + 5 * thirteenth + 4].re, y5_2);
        STOREU_PD(&output_buffer[k + 5 * thirteenth + 6].re, y5_3);

        STOREU_PD(&output_buffer[k + 6 * thirteenth + 0].re, y6_0);
        STOREU_PD(&output_buffer[k + 6 * thirteenth + 2].re, y6_1);
        STOREU_PD(&output_buffer[k + 6 * thirteenth + 4].re, y6_2);
        STOREU_PD(&output_buffer[k + 6 * thirteenth + 6].re, y6_3);

        STOREU_PD(&output_buffer[k + 7 * thirteenth + 0].re, y7_0);
        STOREU_PD(&output_buffer[k + 7 * thirteenth + 2].re, y7_1);
        STOREU_PD(&output_buffer[k + 7 * thirteenth + 4].re, y7_2);
        STOREU_PD(&output_buffer[k + 7 * thirteenth + 6].re, y7_3);

        STOREU_PD(&output_buffer[k + 8 * thirteenth + 0].re, y8_0);
        STOREU_PD(&output_buffer[k + 8 * thirteenth + 2].re, y8_1);
        STOREU_PD(&output_buffer[k + 8 * thirteenth + 4].re, y8_2);
        STOREU_PD(&output_buffer[k + 8 * thirteenth + 6].re, y8_3);

        STOREU_PD(&output_buffer[k + 9 * thirteenth + 0].re, y9_0);
        STOREU_PD(&output_buffer[k + 9 * thirteenth + 2].re, y9_1);
        STOREU_PD(&output_buffer[k + 9 * thirteenth + 4].re, y9_2);
        STOREU_PD(&output_buffer[k + 9 * thirteenth + 6].re, y9_3);

        STOREU_PD(&output_buffer[k + 10 * thirteenth + 0].re, y10_0);
        STOREU_PD(&output_buffer[k + 10 * thirteenth + 2].re, y10_1);
        STOREU_PD(&output_buffer[k + 10 * thirteenth + 4].re, y10_2);
        STOREU_PD(&output_buffer[k + 10 * thirteenth + 6].re, y10_3);

        STOREU_PD(&output_buffer[k + 11 * thirteenth + 0].re, y11_0);
        STOREU_PD(&output_buffer[k + 11 * thirteenth + 2].re, y11_1);
        STOREU_PD(&output_buffer[k + 11 * thirteenth + 4].re, y11_2);
        STOREU_PD(&output_buffer[k + 11 * thirteenth + 6].re, y11_3);

        STOREU_PD(&output_buffer[k + 12 * thirteenth + 0].re, y12_0);
        STOREU_PD(&output_buffer[k + 12 * thirteenth + 2].re, y12_1);
        STOREU_PD(&output_buffer[k + 12 * thirteenth + 4].re, y12_2);
        STOREU_PD(&output_buffer[k + 12 * thirteenth + 6].re, y12_3);
    }

    // -----------------------------
    // 2x AVX2 cleanup
    // -----------------------------
    for (; k + 1 < thirteenth; k += 2)
    {
        if (k + 8 < thirteenth)
        {
            _mm_prefetch((const char *)&sub_outputs[k + 8].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_outputs[k + 8 + thirteenth].re, _MM_HINT_T0);
        }

        // Load x0..x12 (2 butterflies)
        __m256d x0 = load2_aos(&sub_outputs[k], &sub_outputs[k + 1]);
        __m256d x1 = load2_aos(&sub_outputs[k + thirteenth], &sub_outputs[k + thirteenth + 1]);
        __m256d x2 = load2_aos(&sub_outputs[k + 2 * thirteenth], &sub_outputs[k + 2 * thirteenth + 1]);
        __m256d x3 = load2_aos(&sub_outputs[k + 3 * thirteenth], &sub_outputs[k + 3 * thirteenth + 1]);
        __m256d x4 = load2_aos(&sub_outputs[k + 4 * thirteenth], &sub_outputs[k + 4 * thirteenth + 1]);
        __m256d x5 = load2_aos(&sub_outputs[k + 5 * thirteenth], &sub_outputs[k + 5 * thirteenth + 1]);
        __m256d x6 = load2_aos(&sub_outputs[k + 6 * thirteenth], &sub_outputs[k + 6 * thirteenth + 1]);
        __m256d x7 = load2_aos(&sub_outputs[k + 7 * thirteenth], &sub_outputs[k + 7 * thirteenth + 1]);
        __m256d x8 = load2_aos(&sub_outputs[k + 8 * thirteenth], &sub_outputs[k + 8 * thirteenth + 1]);
        __m256d x9 = load2_aos(&sub_outputs[k + 9 * thirteenth], &sub_outputs[k + 9 * thirteenth + 1]);
        __m256d x10 = load2_aos(&sub_outputs[k + 10 * thirteenth], &sub_outputs[k + 10 * thirteenth + 1]);
        __m256d x11 = load2_aos(&sub_outputs[k + 11 * thirteenth], &sub_outputs[k + 11 * thirteenth + 1]);
        __m256d x12 = load2_aos(&sub_outputs[k + 12 * thirteenth], &sub_outputs[k + 12 * thirteenth + 1]);

        if (thirteenth > 1)
        {
            __m256d w1 = load2_aos(&stage_tw[12 * k + 0], &stage_tw[12 * (k + 1) + 0]);
            __m256d w2 = load2_aos(&stage_tw[12 * k + 1], &stage_tw[12 * (k + 1) + 1]);
            __m256d w3 = load2_aos(&stage_tw[12 * k + 2], &stage_tw[12 * (k + 1) + 2]);
            __m256d w4 = load2_aos(&stage_tw[12 * k + 3], &stage_tw[12 * (k + 1) + 3]);
            __m256d w5 = load2_aos(&stage_tw[12 * k + 4], &stage_tw[12 * (k + 1) + 4]);
            __m256d w6 = load2_aos(&stage_tw[12 * k + 5], &stage_tw[12 * (k + 1) + 5]);
            __m256d w7 = load2_aos(&stage_tw[12 * k + 6], &stage_tw[12 * (k + 1) + 6]);
            __m256d w8 = load2_aos(&stage_tw[12 * k + 7], &stage_tw[12 * (k + 1) + 7]);
            __m256d w9 = load2_aos(&stage_tw[12 * k + 8], &stage_tw[12 * (k + 1) + 8]);
            __m256d w10 = load2_aos(&stage_tw[12 * k + 9], &stage_tw[12 * (k + 1) + 9]);
            __m256d w11 = load2_aos(&stage_tw[12 * k + 10], &stage_tw[12 * (k + 1) + 10]);
            __m256d w12 = load2_aos(&stage_tw[12 * k + 11], &stage_tw[12 * (k + 1) + 11]);

            x1 = cmul_avx2_aos(x1, w1);
            x2 = cmul_avx2_aos(x2, w2);
            x3 = cmul_avx2_aos(x3, w3);
            x4 = cmul_avx2_aos(x4, w4);
            x5 = cmul_avx2_aos(x5, w5);
            x6 = cmul_avx2_aos(x6, w6);
            x7 = cmul_avx2_aos(x7, w7);
            x8 = cmul_avx2_aos(x8, w8);
            x9 = cmul_avx2_aos(x9, w9);
            x10 = cmul_avx2_aos(x10, w10);
            x11 = cmul_avx2_aos(x11, w11);
            x12 = cmul_avx2_aos(x12, w12);
        }

        // y0
        __m256d y0 = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(x0, x1), _mm256_add_pd(x2, x3)),
                                                 _mm256_add_pd(_mm256_add_pd(x4, x5), _mm256_add_pd(x6, x7))),
                                   _mm256_add_pd(_mm256_add_pd(x8, x9), _mm256_add_pd(_mm256_add_pd(x10, x11), x12)));

        // Rader permute
        __m256d t0 = x1, t1 = x2, t2 = x4, t3 = x8, t4 = x3, t5 = x6;
        __m256d t6 = x12, t7 = x11, t8 = x9, t9 = x5, t10 = x10, t11 = x7;

        // 12-pt cyclic convolution (simplified for 2x)
        __m256d u0 = cmul_avx2_aos(t0, tw_brd[0]);
        u0 = _mm256_add_pd(u0, cmul_avx2_aos(t1, tw_brd[11]));
        u0 = _mm256_add_pd(u0, cmul_avx2_aos(t2, tw_brd[10]));
        u0 = _mm256_add_pd(u0, cmul_avx2_aos(t3, tw_brd[9]));
        u0 = _mm256_add_pd(u0, cmul_avx2_aos(t4, tw_brd[8]));
        u0 = _mm256_add_pd(u0, cmul_avx2_aos(t5, tw_brd[7]));
        u0 = _mm256_add_pd(u0, cmul_avx2_aos(t6, tw_brd[6]));
        u0 = _mm256_add_pd(u0, cmul_avx2_aos(t7, tw_brd[5]));
        u0 = _mm256_add_pd(u0, cmul_avx2_aos(t8, tw_brd[4]));
        u0 = _mm256_add_pd(u0, cmul_avx2_aos(t9, tw_brd[3]));
        u0 = _mm256_add_pd(u0, cmul_avx2_aos(t10, tw_brd[2]));
        u0 = _mm256_add_pd(u0, cmul_avx2_aos(t11, tw_brd[1]));

        __m256d u1 = cmul_avx2_aos(t0, tw_brd[1]);
        u1 = _mm256_add_pd(u1, cmul_avx2_aos(t1, tw_brd[0]));
        u1 = _mm256_add_pd(u1, cmul_avx2_aos(t2, tw_brd[11]));
        u1 = _mm256_add_pd(u1, cmul_avx2_aos(t3, tw_brd[10]));
        u1 = _mm256_add_pd(u1, cmul_avx2_aos(t4, tw_brd[9]));
        u1 = _mm256_add_pd(u1, cmul_avx2_aos(t5, tw_brd[8]));
        u1 = _mm256_add_pd(u1, cmul_avx2_aos(t6, tw_brd[7]));
        u1 = _mm256_add_pd(u1, cmul_avx2_aos(t7, tw_brd[6]));
        u1 = _mm256_add_pd(u1, cmul_avx2_aos(t8, tw_brd[5]));
        u1 = _mm256_add_pd(u1, cmul_avx2_aos(t9, tw_brd[4]));
        u1 = _mm256_add_pd(u1, cmul_avx2_aos(t10, tw_brd[3]));
        u1 = _mm256_add_pd(u1, cmul_avx2_aos(t11, tw_brd[2]));

        __m256d u2 = cmul_avx2_aos(t0, tw_brd[2]);
        u2 = _mm256_add_pd(u2, cmul_avx2_aos(t1, tw_brd[1]));
        u2 = _mm256_add_pd(u2, cmul_avx2_aos(t2, tw_brd[0]));
        u2 = _mm256_add_pd(u2, cmul_avx2_aos(t3, tw_brd[11]));
        u2 = _mm256_add_pd(u2, cmul_avx2_aos(t4, tw_brd[10]));
        u2 = _mm256_add_pd(u2, cmul_avx2_aos(t5, tw_brd[9]));
        u2 = _mm256_add_pd(u2, cmul_avx2_aos(t6, tw_brd[8]));
        u2 = _mm256_add_pd(u2, cmul_avx2_aos(t7, tw_brd[7]));
        u2 = _mm256_add_pd(u2, cmul_avx2_aos(t8, tw_brd[6]));
        u2 = _mm256_add_pd(u2, cmul_avx2_aos(t9, tw_brd[5]));
        u2 = _mm256_add_pd(u2, cmul_avx2_aos(t10, tw_brd[4]));
        u2 = _mm256_add_pd(u2, cmul_avx2_aos(t11, tw_brd[3]));

        __m256d u3 = cmul_avx2_aos(t0, tw_brd[3]);
        u3 = _mm256_add_pd(u3, cmul_avx2_aos(t1, tw_brd[2]));
        u3 = _mm256_add_pd(u3, cmul_avx2_aos(t2, tw_brd[1]));
        u3 = _mm256_add_pd(u3, cmul_avx2_aos(t3, tw_brd[0]));
        u3 = _mm256_add_pd(u3, cmul_avx2_aos(t4, tw_brd[11]));
        u3 = _mm256_add_pd(u3, cmul_avx2_aos(t5, tw_brd[10]));
        u3 = _mm256_add_pd(u3, cmul_avx2_aos(t6, tw_brd[9]));
        u3 = _mm256_add_pd(u3, cmul_avx2_aos(t7, tw_brd[8]));
        u3 = _mm256_add_pd(u3, cmul_avx2_aos(t8, tw_brd[7]));
        u3 = _mm256_add_pd(u3, cmul_avx2_aos(t9, tw_brd[6]));
        u3 = _mm256_add_pd(u3, cmul_avx2_aos(t10, tw_brd[5]));
        u3 = _mm256_add_pd(u3, cmul_avx2_aos(t11, tw_brd[4]));

        __m256d u4 = cmul_avx2_aos(t0, tw_brd[4]);
        u4 = _mm256_add_pd(u4, cmul_avx2_aos(t1, tw_brd[3]));
        u4 = _mm256_add_pd(u4, cmul_avx2_aos(t2, tw_brd[2]));
        u4 = _mm256_add_pd(u4, cmul_avx2_aos(t3, tw_brd[1]));
        u4 = _mm256_add_pd(u4, cmul_avx2_aos(t4, tw_brd[0]));
        u4 = _mm256_add_pd(u4, cmul_avx2_aos(t5, tw_brd[11]));
        u4 = _mm256_add_pd(u4, cmul_avx2_aos(t6, tw_brd[10]));
        u4 = _mm256_add_pd(u4, cmul_avx2_aos(t7, tw_brd[9]));
        u4 = _mm256_add_pd(u4, cmul_avx2_aos(t8, tw_brd[8]));
        u4 = _mm256_add_pd(u4, cmul_avx2_aos(t9, tw_brd[7]));
        u4 = _mm256_add_pd(u4, cmul_avx2_aos(t10, tw_brd[6]));
        u4 = _mm256_add_pd(u4, cmul_avx2_aos(t11, tw_brd[5]));

        __m256d u5 = cmul_avx2_aos(t0, tw_brd[5]);
        u5 = _mm256_add_pd(u5, cmul_avx2_aos(t1, tw_brd[4]));
        u5 = _mm256_add_pd(u5, cmul_avx2_aos(t2, tw_brd[3]));
        u5 = _mm256_add_pd(u5, cmul_avx2_aos(t3, tw_brd[2]));
        u5 = _mm256_add_pd(u5, cmul_avx2_aos(t4, tw_brd[1]));
        u5 = _mm256_add_pd(u5, cmul_avx2_aos(t5, tw_brd[0]));
        u5 = _mm256_add_pd(u5, cmul_avx2_aos(t6, tw_brd[11]));
        u5 = _mm256_add_pd(u5, cmul_avx2_aos(t7, tw_brd[10]));
        u5 = _mm256_add_pd(u5, cmul_avx2_aos(t8, tw_brd[9]));
        u5 = _mm256_add_pd(u5, cmul_avx2_aos(t9, tw_brd[8]));
        u5 = _mm256_add_pd(u5, cmul_avx2_aos(t10, tw_brd[7]));
        u5 = _mm256_add_pd(u5, cmul_avx2_aos(t11, tw_brd[6]));

        __m256d u6 = cmul_avx2_aos(t0, tw_brd[6]);
        u6 = _mm256_add_pd(u6, cmul_avx2_aos(t1, tw_brd[5]));
        u6 = _mm256_add_pd(u6, cmul_avx2_aos(t2, tw_brd[4]));
        u6 = _mm256_add_pd(u6, cmul_avx2_aos(t3, tw_brd[3]));
        u6 = _mm256_add_pd(u6, cmul_avx2_aos(t4, tw_brd[2]));
        u6 = _mm256_add_pd(u6, cmul_avx2_aos(t5, tw_brd[1]));
        u6 = _mm256_add_pd(u6, cmul_avx2_aos(t6, tw_brd[0]));
        u6 = _mm256_add_pd(u6, cmul_avx2_aos(t7, tw_brd[11]));
        u6 = _mm256_add_pd(u6, cmul_avx2_aos(t8, tw_brd[10]));
        u6 = _mm256_add_pd(u6, cmul_avx2_aos(t9, tw_brd[9]));
        u6 = _mm256_add_pd(u6, cmul_avx2_aos(t10, tw_brd[8]));
        u6 = _mm256_add_pd(u6, cmul_avx2_aos(t11, tw_brd[7]));

        __m256d u7 = cmul_avx2_aos(t0, tw_brd[7]);
        u7 = _mm256_add_pd(u7, cmul_avx2_aos(t1, tw_brd[6]));
        u7 = _mm256_add_pd(u7, cmul_avx2_aos(t2, tw_brd[5]));
        u7 = _mm256_add_pd(u7, cmul_avx2_aos(t3, tw_brd[4]));
        u7 = _mm256_add_pd(u7, cmul_avx2_aos(t4, tw_brd[3]));
        u7 = _mm256_add_pd(u7, cmul_avx2_aos(t5, tw_brd[2]));
        u7 = _mm256_add_pd(u7, cmul_avx2_aos(t6, tw_brd[1]));
        u7 = _mm256_add_pd(u7, cmul_avx2_aos(t7, tw_brd[0]));
        u7 = _mm256_add_pd(u7, cmul_avx2_aos(t8, tw_brd[11]));
        u7 = _mm256_add_pd(u7, cmul_avx2_aos(t9, tw_brd[10]));
        u7 = _mm256_add_pd(u7, cmul_avx2_aos(t10, tw_brd[9]));
        u7 = _mm256_add_pd(u7, cmul_avx2_aos(t11, tw_brd[8]));

        __m256d u8 = cmul_avx2_aos(t0, tw_brd[8]);
        u8 = _mm256_add_pd(u8, cmul_avx2_aos(t1, tw_brd[7]));
        u8 = _mm256_add_pd(u8, cmul_avx2_aos(t2, tw_brd[6]));
        u8 = _mm256_add_pd(u8, cmul_avx2_aos(t3, tw_brd[5]));
        u8 = _mm256_add_pd(u8, cmul_avx2_aos(t4, tw_brd[4]));
        u8 = _mm256_add_pd(u8, cmul_avx2_aos(t5, tw_brd[3]));
        u8 = _mm256_add_pd(u8, cmul_avx2_aos(t6, tw_brd[2]));
        u8 = _mm256_add_pd(u8, cmul_avx2_aos(t7, tw_brd[1]));
        u8 = _mm256_add_pd(u8, cmul_avx2_aos(t8, tw_brd[0]));
        u8 = _mm256_add_pd(u8, cmul_avx2_aos(t9, tw_brd[11]));
        u8 = _mm256_add_pd(u8, cmul_avx2_aos(t10, tw_brd[10]));
        u8 = _mm256_add_pd(u8, cmul_avx2_aos(t11, tw_brd[9]));

        __m256d u9 = cmul_avx2_aos(t0, tw_brd[9]);
        u9 = _mm256_add_pd(u9, cmul_avx2_aos(t1, tw_brd[8]));
        u9 = _mm256_add_pd(u9, cmul_avx2_aos(t2, tw_brd[7]));
        u9 = _mm256_add_pd(u9, cmul_avx2_aos(t3, tw_brd[6]));
        u9 = _mm256_add_pd(u9, cmul_avx2_aos(t4, tw_brd[5]));
        u9 = _mm256_add_pd(u9, cmul_avx2_aos(t5, tw_brd[4]));
        u9 = _mm256_add_pd(u9, cmul_avx2_aos(t6, tw_brd[3]));
        u9 = _mm256_add_pd(u9, cmul_avx2_aos(t7, tw_brd[2]));
        u9 = _mm256_add_pd(u9, cmul_avx2_aos(t8, tw_brd[1]));
        u9 = _mm256_add_pd(u9, cmul_avx2_aos(t9, tw_brd[0]));
        u9 = _mm256_add_pd(u9, cmul_avx2_aos(t10, tw_brd[11]));
        u9 = _mm256_add_pd(u9, cmul_avx2_aos(t11, tw_brd[10]));

        __m256d u10 = cmul_avx2_aos(t0, tw_brd[10]);
        u10 = _mm256_add_pd(u10, cmul_avx2_aos(t1, tw_brd[9]));
        u10 = _mm256_add_pd(u10, cmul_avx2_aos(t2, tw_brd[8]));
        u10 = _mm256_add_pd(u10, cmul_avx2_aos(t3, tw_brd[7]));
        u10 = _mm256_add_pd(u10, cmul_avx2_aos(t4, tw_brd[6]));
        u10 = _mm256_add_pd(u10, cmul_avx2_aos(t5, tw_brd[5]));
        u10 = _mm256_add_pd(u10, cmul_avx2_aos(t6, tw_brd[4]));
        u10 = _mm256_add_pd(u10, cmul_avx2_aos(t7, tw_brd[3]));
        u10 = _mm256_add_pd(u10, cmul_avx2_aos(t8, tw_brd[2]));
        u10 = _mm256_add_pd(u10, cmul_avx2_aos(t9, tw_brd[1]));
        u10 = _mm256_add_pd(u10, cmul_avx2_aos(t10, tw_brd[0]));
        u10 = _mm256_add_pd(u10, cmul_avx2_aos(t11, tw_brd[11]));

        __m256d u11 = cmul_avx2_aos(t0, tw_brd[11]);
        u11 = _mm256_add_pd(u11, cmul_avx2_aos(t1, tw_brd[10]));
        u11 = _mm256_add_pd(u11, cmul_avx2_aos(t2, tw_brd[9]));
        u11 = _mm256_add_pd(u11, cmul_avx2_aos(t3, tw_brd[8]));
        u11 = _mm256_add_pd(u11, cmul_avx2_aos(t4, tw_brd[7]));
        u11 = _mm256_add_pd(u11, cmul_avx2_aos(t5, tw_brd[6]));
        u11 = _mm256_add_pd(u11, cmul_avx2_aos(t6, tw_brd[5]));
        u11 = _mm256_add_pd(u11, cmul_avx2_aos(t7, tw_brd[4]));
        u11 = _mm256_add_pd(u11, cmul_avx2_aos(t8, tw_brd[3]));
        u11 = _mm256_add_pd(u11, cmul_avx2_aos(t9, tw_brd[2]));
        u11 = _mm256_add_pd(u11, cmul_avx2_aos(t10, tw_brd[1]));
        u11 = _mm256_add_pd(u11, cmul_avx2_aos(t11, tw_brd[0]));

        // Map to outputs (out_perm = [1,12,10,3,9,4,7,5,2,6,11,8])
        __m256d y1 = _mm256_add_pd(x0, u0);   // m = 1
        __m256d y12 = _mm256_add_pd(x0, u1);  // m = 12
        __m256d y10 = _mm256_add_pd(x0, u2);  // m = 10
        __m256d y3 = _mm256_add_pd(x0, u3);   // m = 3
        __m256d y9 = _mm256_add_pd(x0, u4);   // m = 9
        __m256d y4 = _mm256_add_pd(x0, u5);   // m = 4
        __m256d y7 = _mm256_add_pd(x0, u6);   // m = 7
        __m256d y5 = _mm256_add_pd(x0, u7);   // m = 5
        __m256d y2 = _mm256_add_pd(x0, u8);   // m = 2
        __m256d y6 = _mm256_add_pd(x0, u9);   // m = 6
        __m256d y11 = _mm256_add_pd(x0, u10); // m = 11
        __m256d y8 = _mm256_add_pd(x0, u11);  // m = 8

        // Store
        STOREU_PD(&output_buffer[k + 0 * thirteenth].re, y0);
        STOREU_PD(&output_buffer[k + 1 * thirteenth].re, y1);
        STOREU_PD(&output_buffer[k + 2 * thirteenth].re, y2);
        STOREU_PD(&output_buffer[k + 3 * thirteenth].re, y3);
        STOREU_PD(&output_buffer[k + 4 * thirteenth].re, y4);
        STOREU_PD(&output_buffer[k + 5 * thirteenth].re, y5);
        STOREU_PD(&output_buffer[k + 6 * thirteenth].re, y6);
        STOREU_PD(&output_buffer[k + 7 * thirteenth].re, y7);
        STOREU_PD(&output_buffer[k + 8 * thirteenth].re, y8);
        STOREU_PD(&output_buffer[k + 9 * thirteenth].re, y9);
        STOREU_PD(&output_buffer[k + 10 * thirteenth].re, y10);
        STOREU_PD(&output_buffer[k + 11 * thirteenth].re, y11);
        STOREU_PD(&output_buffer[k + 12 * thirteenth].re, y12);
    }

#undef CONV12_Q
#undef MAP_CONV_TO_OUTPUT_13
#endif // __AVX2__
       // -----------------------------
       // Scalar tail (0..1 leftover)
       // -----------------------------
    for (; k < thirteenth; ++k)
    {
        // Load 13 lanes
        const fft_data x0 = sub_outputs[k];
        const fft_data x1 = sub_outputs[k + thirteenth];
        const fft_data x2 = sub_outputs[k + 2 * thirteenth];
        const fft_data x3 = sub_outputs[k + 3 * thirteenth];
        const fft_data x4 = sub_outputs[k + 4 * thirteenth];
        const fft_data x5 = sub_outputs[k + 5 * thirteenth];
        const fft_data x6 = sub_outputs[k + 6 * thirteenth];
        const fft_data x7 = sub_outputs[k + 7 * thirteenth];
        const fft_data x8 = sub_outputs[k + 8 * thirteenth];
        const fft_data x9 = sub_outputs[k + 9 * thirteenth];
        const fft_data x10 = sub_outputs[k + 10 * thirteenth];
        const fft_data x11 = sub_outputs[k + 11 * thirteenth];
        const fft_data x12 = sub_outputs[k + 12 * thirteenth];

        // Load twiddles (k-major)
        const fft_data w1 = stage_tw[12 * k];
        const fft_data w2 = stage_tw[12 * k + 1];
        const fft_data w3 = stage_tw[12 * k + 2];
        const fft_data w4 = stage_tw[12 * k + 3];
        const fft_data w5 = stage_tw[12 * k + 4];
        const fft_data w6 = stage_tw[12 * k + 5];
        const fft_data w7 = stage_tw[12 * k + 6];
        const fft_data w8 = stage_tw[12 * k + 7];
        const fft_data w9 = stage_tw[12 * k + 8];
        const fft_data w10 = stage_tw[12 * k + 9];
        const fft_data w11 = stage_tw[12 * k + 10];
        const fft_data w12 = stage_tw[12 * k + 11];

        // Twiddle multiply
        double x1r = x1.re * w1.re - x1.im * w1.im, x1i = x1.re * w1.im + x1.im * w1.re;
        double x2r = x2.re * w2.re - x2.im * w2.im, x2i = x2.re * w2.im + x2.im * w2.re;
        double x3r = x3.re * w3.re - x3.im * w3.im, x3i = x3.re * w3.im + x3.im * w3.re;
        double x4r = x4.re * w4.re - x4.im * w4.im, x4i = x4.re * w4.im + x4.im * w4.re;
        double x5r = x5.re * w5.re - x5.im * w5.im, x5i = x5.re * w5.im + x5.im * w5.re;
        double x6r = x6.re * w6.re - x6.im * w6.im, x6i = x6.re * w6.im + x6.im * w6.re;
        double x7r = x7.re * w7.re - x7.im * w7.im, x7i = x7.re * w7.im + x7.im * w7.re;
        double x8r = x8.re * w8.re - x8.im * w8.im, x8i = x8.re * w8.im + x8.im * w8.re;
        double x9r = x9.re * w9.re - x9.im * w9.im, x9i = x9.re * w9.im + x9.im * w9.re;
        double x10r = x10.re * w10.re - x10.im * w10.im, x10i = x10.re * w10.im + x10.im * w10.re;
        double x11r = x11.re * w11.re - x11.im * w11.im, x11i = x11.re * w11.im + x11.im * w11.re;
        double x12r = x12.re * w12.re - x12.im * w12.im, x12i = x12.re * w12.im + x12.im * w12.re;

        // Y_0
        fft_data y0 = {
            x0.re + (x1r + x2r + x3r + x4r + x5r + x6r + x7r + x8r + x9r + x10r + x11r + x12r),
            x0.im + (x1i + x2i + x3i + x4i + x5i + x6i + x7i + x8i + x9i + x10i + x11i + x12i)};

        // Rader input permute: tx = [x1, x2, x4, x8, x3, x6, x12, x11, x9, x5, x10, x7]
        fft_data tx[12];
        tx[0].re = x1r;
        tx[0].im = x1i;
        tx[1].re = x2r;
        tx[1].im = x2i;
        tx[2].re = x4r;
        tx[2].im = x4i;
        tx[3].re = x8r;
        tx[3].im = x8i;
        tx[4].re = x3r;
        tx[4].im = x3i;
        tx[5].re = x6r;
        tx[5].im = x6i;
        tx[6].re = x12r;
        tx[6].im = x12i;
        tx[7].re = x11r;
        tx[7].im = x11i;
        tx[8].re = x9r;
        tx[8].im = x9i;
        tx[9].re = x5r;
        tx[9].im = x5i;
        tx[10].re = x10r;
        tx[10].im = x10i;
        tx[11].re = x7r;
        tx[11].im = x7i;

        // Build convolution twiddles
        fft_data tw[12];
        {
            const int op[12] = {1, 12, 10, 3, 9, 4, 7, 5, 2, 6, 11, 8};
            for (int q = 0; q < 12; ++q)
            {
                double a = op[q] * base_angle;
#ifdef __GNUC__
                sincos(a, &tw[q].im, &tw[q].re);
#else
                tw[q].re = cos(a);
                tw[q].im = sin(a);
#endif
            }
        }

        // 12-pt cyclic convolution
        fft_data v[12];
        for (int q = 0; q < 12; ++q)
        {
            v[q].re = 0.0;
            v[q].im = 0.0;
            for (int l = 0; l < 12; ++l)
            {
                int idx = (q - l + 12) % 12;
                v[q].re += tx[l].re * tw[idx].re - tx[l].im * tw[idx].im;
                v[q].im += tx[l].re * tw[idx].im + tx[l].im * tw[idx].re;
            }
        }

        // Map to outputs (out_perm = [1,12,10,3,9,4,7,5,2,6,11,8])
        fft_data y1 = {x0.re + v[0].re, x0.im + v[0].im};    // m = 1
        fft_data y12 = {x0.re + v[1].re, x0.im + v[1].im};   // m = 12
        fft_data y10 = {x0.re + v[2].re, x0.im + v[2].im};   // m = 10
        fft_data y3 = {x0.re + v[3].re, x0.im + v[3].im};    // m = 3
        fft_data y9 = {x0.re + v[4].re, x0.im + v[4].im};    // m = 9
        fft_data y4 = {x0.re + v[5].re, x0.im + v[5].im};    // m = 4
        fft_data y7 = {x0.re + v[6].re, x0.im + v[6].im};    // m = 7
        fft_data y5 = {x0.re + v[7].re, x0.im + v[7].im};    // m = 5
        fft_data y2 = {x0.re + v[8].re, x0.im + v[8].im};    // m = 2
        fft_data y6 = {x0.re + v[9].re, x0.im + v[9].im};    // m = 6
        fft_data y11 = {x0.re + v[10].re, x0.im + v[10].im}; // m = 11
        fft_data y8 = {x0.re + v[11].re, x0.im + v[11].im};  // m = 8

        // Store
        output_buffer[k] = y0;
        output_buffer[k + thirteenth] = y1;
        output_buffer[k + 2 * thirteenth] = y2;
        output_buffer[k + 3 * thirteenth] = y3;
        output_buffer[k + 4 * thirteenth] = y4;
        output_buffer[k + 5 * thirteenth] = y5;
        output_buffer[k + 6 * thirteenth] = y6;
        output_buffer[k + 7 * thirteenth] = y7;
        output_buffer[k + 8 * thirteenth] = y8;
        output_buffer[k + 9 * thirteenth] = y9;
        output_buffer[k + 10 * thirteenth] = y10;
        output_buffer[k + 11 * thirteenth] = y11;
        output_buffer[k + 12 * thirteenth] = y12;
    }
}