#include "fft_radix4.h"
#include "simd_math.h"


static inline __m512d load4_aos_stride3(const fft_data* base3k_plus_off) {
    // base3k_plus_off = &stage_tw[3*k + off]
    __m256d v01 = load2_aos(&base3k_plus_off[3*0], &base3k_plus_off[3*1]); // k, k+1
    __m256d v23 = load2_aos(&base3k_plus_off[3*2], &base3k_plus_off[3*3]); // k+2, k+3
    __m512d r   = _mm512_castpd256_pd512(v01);
    return _mm512_insertf64x4(r, v23, 1);
}

//==============================================================================
// MAIN BUTTERFLY FUNCTION WITH MULTI-LEVEL PREFETCHING
//==============================================================================

void fft_radix4_butterfly(
    fft_data *restrict output_buffer,
    fft_data *restrict sub_outputs,
    const fft_data *restrict stage_tw,
    int sub_len,
    int transform_sign)
{
    const int quarter = sub_len;


    //==========================================================================
    // MULTI-LEVEL PREFETCHING CONFIGURATION (Optimization #2)
    //==========================================================================
    
    // Adaptive prefetch distances based on FFT size
    int PREFETCH_L1, PREFETCH_L2;
    
    if (quarter <= 256) {
        // Small FFTs: aggressive L1 prefetch only
        PREFETCH_L1 = 16;
        PREFETCH_L2 = 0; // Disable L2 prefetch
    } else if (quarter <= 4096) {
        // Medium FFTs: L1 + L2 prefetch
        PREFETCH_L1 = 32;
        PREFETCH_L2 = 128;
    } else {
        // Large FFTs: aggressive multi-level prefetch
        PREFETCH_L1 = 64;
        PREFETCH_L2 = 256;
    }

    int k = 0;

#ifdef HAS_AVX512
    //------------------------------------------------------------------
    // AVX-512 PATH: Split 16x unrolling with multi-level prefetch
    //------------------------------------------------------------------
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

    for (; k + 15 < quarter; k += 16)
    {
        //======================================================================
        // MULTI-LEVEL PREFETCHING (Optimization #2)
        //======================================================================
        if (k + PREFETCH_L1 < quarter)
        {
            _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L1], _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L1 + quarter], _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L1 + 2 * quarter], _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L1 + 3 * quarter], _MM_HINT_T0);
            _mm_prefetch((const char *)&stage_tw[3 * (k + PREFETCH_L1)], _MM_HINT_T0);
        }
        
        if (PREFETCH_L2 > 0 && k + PREFETCH_L2 < quarter)
        {
            _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L2], _MM_HINT_T1);
            _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L2 + quarter], _MM_HINT_T1);
            _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L2 + 2 * quarter], _MM_HINT_T1);
            _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L2 + 3 * quarter], _MM_HINT_T1);
            _mm_prefetch((const char *)&stage_tw[3 * (k + PREFETCH_L2)], _MM_HINT_T1);
        }

        //==================================================================
        // PHASE 1: Process butterflies 0-7 (reduced register pressure)
        //==================================================================
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

        //==================================================================
        // PHASE 2: Process butterflies 8-15
        //==================================================================
        {
            __m512d a2 = load4_aos(&sub_outputs[k + 8]);
            __m512d a3 = load4_aos(&sub_outputs[k + 12]);

            __m512d b2 = load4_aos(&sub_outputs[k + 8 + quarter]);
            __m512d b3 = load4_aos(&sub_outputs[k + 12 + quarter]);

            __m512d c2 = load4_aos(&sub_outputs[k + 8 + 2 * quarter]);
            __m512d c3 = load4_aos(&sub_outputs[k + 12 + 2 * quarter]);

            __m512d d2 = load4_aos(&sub_outputs[k + 8 + 3 * quarter]);
            __m512d d3 = load4_aos(&sub_outputs[k + 12 + 3 * quarter]);

            __m512d w1_2 = load4_aos(&stage_tw[3 * (k + 8)]);
            __m512d w1_3 = load4_aos(&stage_tw[3 * (k + 12)]);

            __m512d w2_2 = load4_aos(&stage_tw[3 * (k + 8) + 1]);
            __m512d w2_3 = load4_aos(&stage_tw[3 * (k + 12) + 1]);

            __m512d w3_2 = load4_aos(&stage_tw[3 * (k + 8) + 2]);
            __m512d w3_3 = load4_aos(&stage_tw[3 * (k + 12) + 2]);

            __m512d b2_2 = cmul_avx512_aos(b2, w1_2);
            __m512d b2_3 = cmul_avx512_aos(b3, w1_3);

            __m512d c2_2 = cmul_avx512_aos(c2, w2_2);
            __m512d c2_3 = cmul_avx512_aos(c3, w2_3);

            __m512d d2_2 = cmul_avx512_aos(d2, w3_2);
            __m512d d2_3 = cmul_avx512_aos(d3, w3_3);

            __m512d y0_2, y1_2, y2_2, y3_2;
            __m512d y0_3, y1_3, y2_3, y3_3;

            RADIX4_BUTTERFLY_AVX512(a2, b2_2, c2_2, d2_2, y0_2, y1_2, y2_2, y3_2);
            RADIX4_BUTTERFLY_AVX512(a3, b2_3, c2_3, d2_3, y0_3, y1_3, y2_3, y3_3);

            STOREU_PD512(&output_buffer[k + 8].re, y0_2);
            STOREU_PD512(&output_buffer[k + 12].re, y0_3);

            STOREU_PD512(&output_buffer[k + 8 + quarter].re, y1_2);
            STOREU_PD512(&output_buffer[k + 12 + quarter].re, y1_3);

            STOREU_PD512(&output_buffer[k + 8 + 2 * quarter].re, y2_2);
            STOREU_PD512(&output_buffer[k + 12 + 2 * quarter].re, y2_3);

            STOREU_PD512(&output_buffer[k + 8 + 3 * quarter].re, y3_2);
            STOREU_PD512(&output_buffer[k + 12 + 3 * quarter].re, y3_3);
        }
    }

    //==========================================================================
    // Cleanup: 8x unrolling (process 8 butterflies at once)
    //==========================================================================
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

    //==========================================================================
    // Cleanup: 4x unrolling (process 4 butterflies at once)
    //==========================================================================
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
    //------------------------------------------------------------------
    // AVX2 PATH: Software pipelined 8x unroll with interleaved loads
    //------------------------------------------------------------------
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
        __m256d rot = _mm256_xor_pd(_mm256_permute_pd(difBD, 0b0101), rot_mask); \
        y1 = _mm256_sub_pd(a_mc, rot);                        \
        y3 = _mm256_add_pd(a_mc, rot);                        \
    }

    for (; k + 7 < quarter; k += 8)
    {
        //======================================================================
        // MULTI-LEVEL PREFETCHING (keep existing)
        //======================================================================
        if (k + PREFETCH_L1 < quarter)
        {
            _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L1], _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L1 + quarter], _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L1 + 2 * quarter], _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L1 + 3 * quarter], _MM_HINT_T0);
            _mm_prefetch((const char *)&stage_tw[3 * (k + PREFETCH_L1)], _MM_HINT_T0);
        }
        
        if (PREFETCH_L2 > 0 && k + PREFETCH_L2 < quarter)
        {
            _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L2], _MM_HINT_T1);
            _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L2 + quarter], _MM_HINT_T1);
            _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L2 + 2 * quarter], _MM_HINT_T1);
            _mm_prefetch((const char *)&sub_outputs[k + PREFETCH_L2 + 3 * quarter], _MM_HINT_T1);
            _mm_prefetch((const char *)&stage_tw[3 * (k + PREFETCH_L2)], _MM_HINT_T1);
        }

        //==================================================================
        // BUTTERFLY 0-1: Load and compute, then start loading butterfly 2-3
        //==================================================================
        __m256d a0 = load2_aos(&sub_outputs[k + 0], &sub_outputs[k + 1]);
        __m256d b0 = load2_aos(&sub_outputs[k + 0 + quarter], &sub_outputs[k + 1 + quarter]);
        __m256d c0 = load2_aos(&sub_outputs[k + 0 + 2 * quarter], &sub_outputs[k + 1 + 2 * quarter]);
        __m256d d0 = load2_aos(&sub_outputs[k + 0 + 3 * quarter], &sub_outputs[k + 1 + 3 * quarter]);

        __m256d w1_0 = load2_aos(&stage_tw[3 * (k + 0)], &stage_tw[3 * (k + 1)]);
        __m256d w2_0 = load2_aos(&stage_tw[3 * (k + 0) + 1], &stage_tw[3 * (k + 1) + 1]);
        __m256d w3_0 = load2_aos(&stage_tw[3 * (k + 0) + 2], &stage_tw[3 * (k + 1) + 2]);

        __m256d b2_0 = cmul_avx2_aos(b0, w1_0);
        __m256d c2_0 = cmul_avx2_aos(c0, w2_0);
        __m256d d2_0 = cmul_avx2_aos(d0, w3_0);

        // ⭐ SOFTWARE PIPELINE: Start loading next butterfly while computing current
        __m256d a1 = load2_aos(&sub_outputs[k + 2], &sub_outputs[k + 3]);
        __m256d b1 = load2_aos(&sub_outputs[k + 2 + quarter], &sub_outputs[k + 3 + quarter]);

        __m256d y0_0, y1_0, y2_0, y3_0;
        RADIX4_BUTTERFLY_AVX2(a0, b2_0, c2_0, d2_0, y0_0, y1_0, y2_0, y3_0);

        //==================================================================
        // BUTTERFLY 2-3: Continue loading while butterfly 0-1 completes
        //==================================================================
        __m256d c1 = load2_aos(&sub_outputs[k + 2 + 2 * quarter], &sub_outputs[k + 3 + 2 * quarter]);
        __m256d d1 = load2_aos(&sub_outputs[k + 2 + 3 * quarter], &sub_outputs[k + 3 + 3 * quarter]);

        __m256d w1_1 = load2_aos(&stage_tw[3 * (k + 2)], &stage_tw[3 * (k + 3)]);
        __m256d w2_1 = load2_aos(&stage_tw[3 * (k + 2) + 1], &stage_tw[3 * (k + 3) + 1]);
        __m256d w3_1 = load2_aos(&stage_tw[3 * (k + 2) + 2], &stage_tw[3 * (k + 3) + 2]);

        __m256d b2_1 = cmul_avx2_aos(b1, w1_1);
        __m256d c2_1 = cmul_avx2_aos(c1, w2_1);
        __m256d d2_1 = cmul_avx2_aos(d1, w3_1);

        // ⭐ Start loading butterfly 4-5
        __m256d a2 = load2_aos(&sub_outputs[k + 4], &sub_outputs[k + 5]);
        __m256d b2 = load2_aos(&sub_outputs[k + 4 + quarter], &sub_outputs[k + 5 + quarter]);

        __m256d y0_1, y1_1, y2_1, y3_1;
        RADIX4_BUTTERFLY_AVX2(a1, b2_1, c2_1, d2_1, y0_1, y1_1, y2_1, y3_1);

        //==================================================================
        // BUTTERFLY 4-5: Continue pattern
        //==================================================================
        __m256d c2 = load2_aos(&sub_outputs[k + 4 + 2 * quarter], &sub_outputs[k + 5 + 2 * quarter]);
        __m256d d2 = load2_aos(&sub_outputs[k + 4 + 3 * quarter], &sub_outputs[k + 5 + 3 * quarter]);

        __m256d w1_2 = load2_aos(&stage_tw[3 * (k + 4)], &stage_tw[3 * (k + 5)]);
        __m256d w2_2 = load2_aos(&stage_tw[3 * (k + 4) + 1], &stage_tw[3 * (k + 5) + 1]);
        __m256d w3_2 = load2_aos(&stage_tw[3 * (k + 4) + 2], &stage_tw[3 * (k + 5) + 2]);

        __m256d b2_2 = cmul_avx2_aos(b2, w1_2);
        __m256d c2_2 = cmul_avx2_aos(c2, w2_2);
        __m256d d2_2 = cmul_avx2_aos(d2, w3_2);

        // ⭐ Start loading butterfly 6-7
        __m256d a3 = load2_aos(&sub_outputs[k + 6], &sub_outputs[k + 7]);
        __m256d b3 = load2_aos(&sub_outputs[k + 6 + quarter], &sub_outputs[k + 7 + quarter]);

        __m256d y0_2, y1_2, y2_2, y3_2;
        RADIX4_BUTTERFLY_AVX2(a2, b2_2, c2_2, d2_2, y0_2, y1_2, y2_2, y3_2);

        //==================================================================
        // BUTTERFLY 6-7: Final butterfly in this iteration
        //==================================================================
        __m256d c3 = load2_aos(&sub_outputs[k + 6 + 2 * quarter], &sub_outputs[k + 7 + 2 * quarter]);
        __m256d d3 = load2_aos(&sub_outputs[k + 6 + 3 * quarter], &sub_outputs[k + 7 + 3 * quarter]);

        __m256d w1_3 = load2_aos(&stage_tw[3 * (k + 6)], &stage_tw[3 * (k + 7)]);
        __m256d w2_3 = load2_aos(&stage_tw[3 * (k + 6) + 1], &stage_tw[3 * (k + 7) + 1]);
        __m256d w3_3 = load2_aos(&stage_tw[3 * (k + 6) + 2], &stage_tw[3 * (k + 7) + 2]);

        __m256d b2_3 = cmul_avx2_aos(b3, w1_3);
        __m256d c2_3 = cmul_avx2_aos(c3, w2_3);
        __m256d d2_3 = cmul_avx2_aos(d3, w3_3);

        __m256d y0_3, y1_3, y2_3, y3_3;
        RADIX4_BUTTERFLY_AVX2(a3, b2_3, c2_3, d2_3, y0_3, y1_3, y2_3, y3_3);

        //==================================================================
        // Store all results (order preserved for correctness)
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
    // Cleanup: 2x unrolling (keep as-is for safety)
    //------------------------------------------------------------------
    for (; k + 1 < quarter; k += 2)
    {
        if (k + 8 < quarter)
        {
            _mm_prefetch((const char *)&sub_outputs[k + 8], _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_outputs[k + 8 + quarter], _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_outputs[k + 8 + 2 * quarter], _MM_HINT_T0);
            _mm_prefetch((const char *)&sub_outputs[k + 8 + 3 * quarter], _MM_HINT_T0);
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

        __m256d y0, y1, y2, y3;
        RADIX4_BUTTERFLY_AVX2(a, b2, c2, d2, y0, y1, y2, y3);

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
#ifdef __GNUC__
#pragma GCC unroll 1
#endif
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
}

/*

__m512d w1_0 = load4_aos_stride3(&stage_tw[3*(k+0) + 0]);
__m512d w2_0 = load4_aos_stride3(&stage_tw[3*(k+0) + 1]);
__m512d w3_0 = load4_aos_stride3(&stage_tw[3*(k+0) + 2]);

__m512d w1_1 = load4_aos_stride3(&stage_tw[3*(k+4) + 0]);
__m512d w2_1 = load4_aos_stride3(&stage_tw[3*(k+4) + 1]);
__m512d w3_1 = load4_aos_stride3(&stage_tw[3*(k+4) + 2]);

// Phase 2
__m512d w1_2 = load4_aos_stride3(&stage_tw[3*(k+8)  + 0]);
__m512d w2_2 = load4_aos_stride3(&stage_tw[3*(k+8)  + 1]);
__m512d w3_2 = load4_aos_stride3(&stage_tw[3*(k+8)  + 2]);

__m512d w1_3 = load4_aos_stride3(&stage_tw[3*(k+12) + 0]);
__m512d w2_3 = load4_aos_stride3(&stage_tw[3*(k+12) + 1]);
__m512d w3_3 = load4_aos_stride3(&stage_tw[3*(k+12) + 2]);

// 8× cleanup
__m512d w1_0 = load4_aos_stride3(&stage_tw[3*(k+0) + 0]);
__m512d w2_0 = load4_aos_stride3(&stage_tw[3*(k+0) + 1]);
__m512d w3_0 = load4_aos_stride3(&stage_tw[3*(k+0) + 2]);

__m512d w1_1 = load4_aos_stride3(&stage_tw[3*(k+4) + 0]);
__m512d w2_1 = load4_aos_stride3(&stage_tw[3*(k+4) + 1]);
__m512d w3_1 = load4_aos_stride3(&stage_tw[3*(k+4) + 2]);

// 4× cleanup
__m512d w1 = load4_aos_stride3(&stage_tw[3*k + 0]);
__m512d w2 = load4_aos_stride3(&stage_tw[3*k + 1]);
__m512d w3 = load4_aos_stride3(&stage_tw[3*k + 2]);
*/