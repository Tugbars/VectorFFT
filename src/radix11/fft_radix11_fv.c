//==============================================================================
// fft_radix11_fv.c - Radix-11 FORWARD Butterfly (AVX-512 + AVX2 + Scalar)
//==============================================================================

#include "fft_radix11_uniform.h"
#include "fft_radix11_macros.h"
#include "highSpeedFFT.h"

/**
 * @brief Radix-11 forward DFT butterfly with multi-tier SIMD optimization
 * 
 * Architecture:
 *   Tier 1: AVX-512 (4 butterflies per iteration) - Ice Lake+, Zen 4+
 *   Tier 2: AVX2 (2 butterflies per iteration) - Haswell+, Zen+
 *   Tier 3: Scalar (1 butterfly per iteration) - All CPUs
 * 
 * @param output_buffer Output array (size: 11 * sub_len)
 * @param sub_outputs Input array from previous stage (size: 11 * sub_len)
 * @param stage_tw Stage twiddle factors [W^(k), W^(2k), ..., W^(10k)]
 * @param sub_len Number of sub-transforms (butterfly count)
 */
void fft_radix11_fv(
    fft_data *output_buffer,
    fft_data *sub_outputs,
    const fft_data *stage_tw,
    int sub_len)
{
    const int K = sub_len;
    int k = 0;

#ifdef __AVX512F__
    //==========================================================================
    // TIER 1: AVX-512 PATH - Process 4 butterflies per iteration
    //==========================================================================
    
    // ⚡ CRITICAL: Broadcast constants ONCE for entire AVX-512 section
    radix11_consts_avx512 K512 = broadcast_radix11_consts_avx512();
    
    for (; k + 3 < K; k += 4)
    {
        // Step 1: Load 11 lanes for 4 butterflies (44 complex values)
        __m512d a, b, c, d, e, f, g, h, i, j, xk;
        LOAD_11_LANES_AVX512(k, K, sub_outputs, a, b, c, d, e, f, g, h, i, j, xk);
        
        // Step 2: Apply precomputed stage twiddles (W^(1*k)...W^(10*k))
        APPLY_STAGE_TWIDDLES_R11_AVX512(k, b, c, d, e, f, g, h, i, j, xk, stage_tw);
        
        // Step 3: Compute symmetric pairs and Y0 (DC component)
        __m512d t0, t1, t2, t3, t4;  // Sums: (b+k), (c+j), (d+i), (e+h), (f+g)
        __m512d s0, s1, s2, s3, s4;  // Diffs: (b-k), (c-j), (d-i), (e-h), (f-g)
        __m512d y0;
        RADIX11_BUTTERFLY_CORE_AVX512(a, b, c, d, e, f, g, h, i, j, xk,
                                      t0, t1, t2, t3, t4,
                                      s0, s1, s2, s3, s4, y0);
        
        // Step 4-8: Compute 5 conjugate pairs (FORWARD uses -i rotation)
        
        // Pair 1: Y_1, Y_10
        __m512d real1, rot1;
        RADIX11_REAL_PAIR1_AVX512(a, t0, t1, t2, t3, t4, K512, real1);
        RADIX11_IMAG_PAIR1_FV_AVX512(s0, s1, s2, s3, s4, K512, rot1);  // FV = -i
        __m512d y1, y10;
        RADIX11_ASSEMBLE_PAIR_AVX512(real1, rot1, y1, y10);
        
        // Pair 2: Y_2, Y_9
        __m512d real2, rot2;
        RADIX11_REAL_PAIR2_AVX512(a, t0, t1, t2, t3, t4, K512, real2);
        RADIX11_IMAG_PAIR2_FV_AVX512(s0, s1, s2, s3, s4, K512, rot2);  // FV = -i
        __m512d y2, y9;
        RADIX11_ASSEMBLE_PAIR_AVX512(real2, rot2, y2, y9);
        
        // Pair 3: Y_3, Y_8
        __m512d real3, rot3;
        RADIX11_REAL_PAIR3_AVX512(a, t0, t1, t2, t3, t4, K512, real3);
        RADIX11_IMAG_PAIR3_FV_AVX512(s0, s1, s2, s3, s4, K512, rot3);  // FV = -i
        __m512d y3, y8;
        RADIX11_ASSEMBLE_PAIR_AVX512(real3, rot3, y3, y8);
        
        // Pair 4: Y_4, Y_7
        __m512d real4, rot4;
        RADIX11_REAL_PAIR4_AVX512(a, t0, t1, t2, t3, t4, K512, real4);
        RADIX11_IMAG_PAIR4_FV_AVX512(s0, s1, s2, s3, s4, K512, rot4);  // FV = -i
        __m512d y4, y7;
        RADIX11_ASSEMBLE_PAIR_AVX512(real4, rot4, y4, y7);
        
        // Pair 5: Y_5, Y_6
        __m512d real5, rot5;
        RADIX11_REAL_PAIR5_AVX512(a, t0, t1, t2, t3, t4, K512, real5);
        RADIX11_IMAG_PAIR5_FV_AVX512(s0, s1, s2, s3, s4, K512, rot5);  // FV = -i
        __m512d y5, y6;
        RADIX11_ASSEMBLE_PAIR_AVX512(real5, rot5, y5, y6);
        
        // Step 9: Store 11 outputs for 4 butterflies (44 complex values)
        STORE_11_LANES_AVX512(k, K, output_buffer, 
                              y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10);
    }
#endif // __AVX512F__

#ifdef __AVX2__
    //==========================================================================
    // TIER 2: AVX2 PATH - Process 2 butterflies per iteration
    //==========================================================================
    
    // ⚡ CRITICAL: Broadcast constants ONCE for entire AVX2 section
    radix11_consts_avx2 K2 = broadcast_radix11_consts_avx2();
    
    for (; k + 1 < K; k += 2)
    {
        // Prefetch ahead for better cache utilization
        PREFETCH_11_LANES_R11(k, K, PREFETCH_L1_R11, sub_outputs);
        
        // Step 1: Load 11 lanes for 2 butterflies (22 complex values)
        __m256d a, b, c, d, e, f, g, h, i, j, kval;
        LOAD_11_LANES_AVX2(k, K, sub_outputs, a, b, c, d, e, f, g, h, i, j, kval);
        
        // Step 2: Apply precomputed stage twiddles (W^(1*k)...W^(10*k))
        APPLY_STAGE_TWIDDLES_R11_AVX2(k, b, c, d, e, f, g, h, i, j, kval, stage_tw);
        
        // Step 3: Compute symmetric pairs and Y0 (DC component)
        __m256d t0, t1, t2, t3, t4;  // Sums: (b+k), (c+j), (d+i), (e+h), (f+g)
        __m256d s0, s1, s2, s3, s4;  // Diffs: (b-k), (c-j), (d-i), (e-h), (f-g)
        __m256d y0;
        RADIX11_BUTTERFLY_CORE_AVX2(a, b, c, d, e, f, g, h, i, j, kval,
                                    t0, t1, t2, t3, t4,
                                    s0, s1, s2, s3, s4, y0);
        
        // Step 4-8: Compute 5 conjugate pairs (FORWARD uses -i rotation)
        
        // Pair 1: Y_1, Y_10
        __m256d real1, rot1;
        RADIX11_REAL_PAIR1_AVX2(a, t0, t1, t2, t3, t4, K2, real1);
        RADIX11_IMAG_PAIR1_FV_AVX2(s0, s1, s2, s3, s4, K2, rot1);  // FV = -i
        __m256d y1, y10;
        RADIX11_ASSEMBLE_PAIR_AVX2(real1, rot1, y1, y10);
        
        // Pair 2: Y_2, Y_9
        __m256d real2, rot2;
        RADIX11_REAL_PAIR2_AVX2(a, t0, t1, t2, t3, t4, K2, real2);
        RADIX11_IMAG_PAIR2_FV_AVX2(s0, s1, s2, s3, s4, K2, rot2);  // FV = -i
        __m256d y2, y9;
        RADIX11_ASSEMBLE_PAIR_AVX2(real2, rot2, y2, y9);
        
        // Pair 3: Y_3, Y_8
        __m256d real3, rot3;
        RADIX11_REAL_PAIR3_AVX2(a, t0, t1, t2, t3, t4, K2, real3);
        RADIX11_IMAG_PAIR3_FV_AVX2(s0, s1, s2, s3, s4, K2, rot3);  // FV = -i
        __m256d y3, y8;
        RADIX11_ASSEMBLE_PAIR_AVX2(real3, rot3, y3, y8);
        
        // Pair 4: Y_4, Y_7
        __m256d real4, rot4;
        RADIX11_REAL_PAIR4_AVX2(a, t0, t1, t2, t3, t4, K2, real4);
        RADIX11_IMAG_PAIR4_FV_AVX2(s0, s1, s2, s3, s4, K2, rot4);  // FV = -i
        __m256d y4, y7;
        RADIX11_ASSEMBLE_PAIR_AVX2(real4, rot4, y4, y7);
        
        // Pair 5: Y_5, Y_6
        __m256d real5, rot5;
        RADIX11_REAL_PAIR5_AVX2(a, t0, t1, t2, t3, t4, K2, real5);
        RADIX11_IMAG_PAIR5_FV_AVX2(s0, s1, s2, s3, s4, K2, rot5);  // FV = -i
        __m256d y5, y6;
        RADIX11_ASSEMBLE_PAIR_AVX2(real5, rot5, y5, y6);
        
        // Step 9: Store 11 outputs for 2 butterflies (22 complex values)
        STORE_11_LANES_AVX2(k, K, output_buffer, 
                            y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10);
    }
#endif // __AVX2__

    //==========================================================================
    // TIER 3: SCALAR TAIL - Handle remaining 0-3 butterflies
    //==========================================================================
    for (; k < K; k++)
    {
        RADIX11_BUTTERFLY_SCALAR_FV(k, K, sub_outputs, stage_tw, output_buffer);
    }
}
