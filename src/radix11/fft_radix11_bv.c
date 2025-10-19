//==============================================================================
// fft_radix11_bv.c - Radix-11 Butterfly (BACKWARD/INVERSE Transform)
//==============================================================================

#include "fft_radix11_macros.h"
#include "highSpeedFFT.h"

void fft_radix11_bv(
    fft_data *output_buffer,
    fft_data *sub_outputs,
    const fft_data *stage_tw,
    int sub_len)
{
    const int K = sub_len;
    int k = 0;

#ifdef __AVX2__
    //==========================================================================
    // AVX2 PATH: Process 2 butterflies per iteration
    //==========================================================================
    for (; k + 1 < K; k += 2)
    {
        // Prefetch ahead
        PREFETCH_11_LANES_R11(k, K, PREFETCH_L1_R11, sub_outputs);

        // Step 1: Load 11 lanes for 2 butterflies
        __m256d a, b, c, d, e, f, g, h, i, j, xk;
        LOAD_11_LANES_AVX2(k, K, sub_outputs, a, b, c, d, e, f, g, h, i, j, xk);

        // Step 2: Apply precomputed stage twiddles (W^(1*k)...W^(10*k))
        APPLY_STAGE_TWIDDLES_R11_AVX2(k, b, c, d, e, f, g, h, i, j, xk, stage_tw);

        // Step 3: Compute symmetric pairs and Y0
        __m256d t0, t1, t2, t3, t4;  // Sums: (b+k), (c+j), (d+i), (e+h), (f+g)
        __m256d s0, s1, s2, s3, s4;  // Diffs: (b-k), (c-j), (d-i), (e-h), (f-g)
        __m256d y0;
        RADIX11_BUTTERFLY_CORE_AVX2(a, b, c, d, e, f, g, h, i, j, xk,
                                    t0, t1, t2, t3, t4,
                                    s0, s1, s2, s3, s4, y0);

        // Step 4-8: Compute 5 conjugate pairs

        // Pair 1: Y_1, Y_10
        __m256d real1, rot1;
        RADIX11_REAL_PAIR1_AVX2(a, t0, t1, t2, t3, t4, real1);
        RADIX11_IMAG_PAIR1_BV_AVX2(s0, s1, s2, s3, s4, rot1);  // BV uses +i rotation
        __m256d y1, y10;
        RADIX11_ASSEMBLE_PAIR_AVX2(real1, rot1, y1, y10);

        // Pair 2: Y_2, Y_9
        __m256d real2, rot2;
        RADIX11_REAL_PAIR2_AVX2(a, t0, t1, t2, t3, t4, real2);
        RADIX11_IMAG_PAIR2_BV_AVX2(s0, s1, s2, s3, s4, rot2);  // BV uses +i rotation
        __m256d y2, y9;
        RADIX11_ASSEMBLE_PAIR_AVX2(real2, rot2, y2, y9);

        // Pair 3: Y_3, Y_8
        __m256d real3, rot3;
        RADIX11_REAL_PAIR3_AVX2(a, t0, t1, t2, t3, t4, real3);
        RADIX11_IMAG_PAIR3_BV_AVX2(s0, s1, s2, s3, s4, rot3);  // BV uses +i rotation
        __m256d y3, y8;
        RADIX11_ASSEMBLE_PAIR_AVX2(real3, rot3, y3, y8);

        // Pair 4: Y_4, Y_7
        __m256d real4, rot4;
        RADIX11_REAL_PAIR4_AVX2(a, t0, t1, t2, t3, t4, real4);
        RADIX11_IMAG_PAIR4_BV_AVX2(s0, s1, s2, s3, s4, rot4);  // BV uses +i rotation
        __m256d y4, y7;
        RADIX11_ASSEMBLE_PAIR_AVX2(real4, rot4, y4, y7);

        // Pair 5: Y_5, Y_6
        __m256d real5, rot5;
        RADIX11_REAL_PAIR5_AVX2(a, t0, t1, t2, t3, t4, real5);
        RADIX11_IMAG_PAIR5_BV_AVX2(s0, s1, s2, s3, s4, rot5);  // BV uses +i rotation
        __m256d y5, y6;
        RADIX11_ASSEMBLE_PAIR_AVX2(real5, rot5, y5, y6);

        // Step 9: Store 11 outputs for 2 butterflies
        STORE_11_LANES_AVX2(k, K, output_buffer, 
                            y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10);
    }
#endif // __AVX2__

    //==========================================================================
    // SCALAR TAIL: Handle remaining 0-1 elements
    //==========================================================================
    for (; k < K; k++)
    {
        RADIX11_BUTTERFLY_SCALAR_BV(k, K, sub_outputs, stage_tw, output_buffer);
    }
}