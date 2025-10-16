#include "fft_radix8.h" // ✅ Gets highSpeedFFT.h → fft_types.h
#include "simd_math.h"  // ✅ Gets complex math operations

void fft_radix8_butterfly(
    fft_data *output_buffer,
    fft_data *sub_outputs,
    const fft_data *stage_tw,
    int sub_len,
    int transform_sign)
{
    //==========================================================================
    // OPTIMIZED RADIX-8 BUTTERFLY (FFTW-style: 2×4 decomposition)
    //
    // Critical optimizations for low-latency quant trading:
    // 1. Aggressive prefetching with tuned distances
    // 2. Maximized instruction-level parallelism
    // 3. Minimized memory bandwidth via streaming stores
    // 4. Reduced latency operations where possible
    // 5. Better CPU port utilization through operation interleaving
    //==========================================================================

    const int eighth = sub_len;
    int k = 0;

#ifdef __AVX2__
    //----------------------------------------------------------------------
    // AVX2 PATH: Heavily optimized 8x unrolling
    //----------------------------------------------------------------------

    // Pre-compute all constant masks and factors
    const __m256d mask_neg_i = _mm256_set_pd(0.0, -0.0, 0.0, -0.0);
    const __m256d mask_pos_i = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);
    const __m256d rot_mask = (transform_sign == 1) ? mask_neg_i : mask_pos_i;

    // Pre-compute √2/2 for W_8 twiddles
    const __m256d c8 = _mm256_set1_pd(0.7071067811865476);
    const __m256d neg_mask = _mm256_set1_pd(-0.0);

    // Pre-compute W_8^2 rotation masks for both forward and inverse
    const __m256d w8_2_mask = (transform_sign == 1)
                                  ? _mm256_set_pd(-0.0, 0.0, -0.0, 0.0)  // Forward: (im, -re)
                                  : _mm256_set_pd(0.0, -0.0, 0.0, -0.0); // Inverse: (-im, re)

    // Tuned prefetch distances for typical cache hierarchies
    const int prefetch_L1 = 16; // L1 distance
    const int prefetch_L2 = 32; // L2 distance
    const int prefetch_L3 = 64; // L3 distance

    for (; k + 7 < eighth; k += 8)
    {
        //==================================================================
        // Multi-level prefetching strategy
        //==================================================================
        if (k + prefetch_L3 < eighth)
        {
            // L3 prefetch - furthest ahead
            _mm_prefetch((const char *)&sub_outputs[k + prefetch_L3].re, _MM_HINT_T2);
            _mm_prefetch((const char *)&stage_tw[7 * (k + prefetch_L3)].re, _MM_HINT_T2);
        }

        if (k + prefetch_L2 < eighth)
        {
            // L2 prefetch - medium distance
            for (int lane = 0; lane < 8; lane += 2)
            {
                _mm_prefetch((const char *)&sub_outputs[k + prefetch_L2 + lane * eighth].re, _MM_HINT_T1);
            }
            _mm_prefetch((const char *)&stage_tw[7 * (k + prefetch_L2)].re, _MM_HINT_T1);
        }

        if (k + prefetch_L1 < eighth)
        {
            // L1 prefetch - nearest, all lanes
            for (int lane = 0; lane < 8; ++lane)
            {
                _mm_prefetch((const char *)&sub_outputs[k + prefetch_L1 + lane * eighth].re, _MM_HINT_T0);
            }
            _mm_prefetch((const char *)&stage_tw[7 * (k + prefetch_L1)].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&stage_tw[7 * (k + prefetch_L1) + 6].re, _MM_HINT_T0);
        }

        //==================================================================
        // Stage 1: Load and apply input twiddles with maximum ILP
        //==================================================================
        __m256d x[8][4];

        // Lane 0: No twiddle multiplication needed
        x[0][0] = load2_aos(&sub_outputs[k + 0], &sub_outputs[k + 1]);
        x[0][1] = load2_aos(&sub_outputs[k + 2], &sub_outputs[k + 3]);
        x[0][2] = load2_aos(&sub_outputs[k + 4], &sub_outputs[k + 5]);
        x[0][3] = load2_aos(&sub_outputs[k + 6], &sub_outputs[k + 7]);

        // Lanes 1-7: Interleave loads and complex multiplications
        // Unroll by 2 for better port utilization
        for (int lane = 1; lane < 8; lane += 2)
        {
            // Load data for two lanes
            __m256d data_l0_0 = load2_aos(&sub_outputs[k + 0 + lane * eighth],
                                          &sub_outputs[k + 1 + lane * eighth]);
            __m256d data_l1_0 = load2_aos(&sub_outputs[k + 0 + (lane + 1) * eighth],
                                          &sub_outputs[k + 1 + (lane + 1) * eighth]);

            __m256d data_l0_1 = load2_aos(&sub_outputs[k + 2 + lane * eighth],
                                          &sub_outputs[k + 3 + lane * eighth]);
            __m256d data_l1_1 = load2_aos(&sub_outputs[k + 2 + (lane + 1) * eighth],
                                          &sub_outputs[k + 3 + (lane + 1) * eighth]);

            // Load twiddles for two lanes
            __m256d tw_l0_0 = load2_aos(&stage_tw[7 * (k + 0) + (lane - 1)],
                                        &stage_tw[7 * (k + 1) + (lane - 1)]);
            __m256d tw_l1_0 = load2_aos(&stage_tw[7 * (k + 0) + lane],
                                        &stage_tw[7 * (k + 1) + lane]);

            __m256d tw_l0_1 = load2_aos(&stage_tw[7 * (k + 2) + (lane - 1)],
                                        &stage_tw[7 * (k + 3) + (lane - 1)]);
            __m256d tw_l1_1 = load2_aos(&stage_tw[7 * (k + 2) + lane],
                                        &stage_tw[7 * (k + 3) + lane]);

            // Apply twiddles - interleaved for ILP
            x[lane][0] = cmul_avx2_aos(data_l0_0, tw_l0_0);
            x[lane + 1][0] = cmul_avx2_aos(data_l1_0, tw_l1_0);
            x[lane][1] = cmul_avx2_aos(data_l0_1, tw_l0_1);
            x[lane + 1][1] = cmul_avx2_aos(data_l1_1, tw_l1_1);

            // Continue for remaining pairs
            __m256d data_l0_2 = load2_aos(&sub_outputs[k + 4 + lane * eighth],
                                          &sub_outputs[k + 5 + lane * eighth]);
            __m256d data_l1_2 = load2_aos(&sub_outputs[k + 4 + (lane + 1) * eighth],
                                          &sub_outputs[k + 5 + (lane + 1) * eighth]);

            __m256d data_l0_3 = load2_aos(&sub_outputs[k + 6 + lane * eighth],
                                          &sub_outputs[k + 7 + lane * eighth]);
            __m256d data_l1_3 = load2_aos(&sub_outputs[k + 6 + (lane + 1) * eighth],
                                          &sub_outputs[k + 7 + (lane + 1) * eighth]);

            __m256d tw_l0_2 = load2_aos(&stage_tw[7 * (k + 4) + (lane - 1)],
                                        &stage_tw[7 * (k + 5) + (lane - 1)]);
            __m256d tw_l1_2 = load2_aos(&stage_tw[7 * (k + 4) + lane],
                                        &stage_tw[7 * (k + 5) + lane]);

            __m256d tw_l0_3 = load2_aos(&stage_tw[7 * (k + 6) + (lane - 1)],
                                        &stage_tw[7 * (k + 7) + (lane - 1)]);
            __m256d tw_l1_3 = load2_aos(&stage_tw[7 * (k + 6) + lane],
                                        &stage_tw[7 * (k + 7) + lane]);

            x[lane][2] = cmul_avx2_aos(data_l0_2, tw_l0_2);
            x[lane + 1][2] = cmul_avx2_aos(data_l1_2, tw_l1_2);
            x[lane][3] = cmul_avx2_aos(data_l0_3, tw_l0_3);
            x[lane + 1][3] = cmul_avx2_aos(data_l1_3, tw_l1_3);
        }

        // Handle lane 7 if not covered by the unroll-by-2
        if ((7 & 1) == 1)
        {
            const int lane = 7;
            __m256d data0 = load2_aos(&sub_outputs[k + 0 + lane * eighth],
                                      &sub_outputs[k + 1 + lane * eighth]);
            __m256d data1 = load2_aos(&sub_outputs[k + 2 + lane * eighth],
                                      &sub_outputs[k + 3 + lane * eighth]);
            __m256d data2 = load2_aos(&sub_outputs[k + 4 + lane * eighth],
                                      &sub_outputs[k + 5 + lane * eighth]);
            __m256d data3 = load2_aos(&sub_outputs[k + 6 + lane * eighth],
                                      &sub_outputs[k + 7 + lane * eighth]);

            __m256d tw0 = load2_aos(&stage_tw[7 * (k + 0) + (lane - 1)],
                                    &stage_tw[7 * (k + 1) + (lane - 1)]);
            __m256d tw1 = load2_aos(&stage_tw[7 * (k + 2) + (lane - 1)],
                                    &stage_tw[7 * (k + 3) + (lane - 1)]);
            __m256d tw2 = load2_aos(&stage_tw[7 * (k + 4) + (lane - 1)],
                                    &stage_tw[7 * (k + 5) + (lane - 1)]);
            __m256d tw3 = load2_aos(&stage_tw[7 * (k + 6) + (lane - 1)],
                                    &stage_tw[7 * (k + 7) + (lane - 1)]);

            x[lane][0] = cmul_avx2_aos(data0, tw0);
            x[lane][1] = cmul_avx2_aos(data1, tw1);
            x[lane][2] = cmul_avx2_aos(data2, tw2);
            x[lane][3] = cmul_avx2_aos(data3, tw3);
        }

        //==================================================================
        // Stage 2 & 3: Parallel radix-4 butterflies on even and odd indices
        // Process butterflies in groups of 2 for better port utilization
        //==================================================================
        __m256d e[4][4]; // Even radix-4 outputs
        __m256d o[4][4]; // Odd radix-4 outputs

        // Unroll by 2 for better ILP
        for (int b = 0; b < 4; b += 2)
        {
            // Load inputs for two butterfly groups
            __m256d a_e0 = x[0][b], a_e1 = x[0][b + 1];
            __m256d c_e0 = x[2][b], c_e1 = x[2][b + 1];
            __m256d e_e0 = x[4][b], e_e1 = x[4][b + 1];
            __m256d g_e0 = x[6][b], g_e1 = x[6][b + 1];

            __m256d a_o0 = x[1][b], a_o1 = x[1][b + 1];
            __m256d c_o0 = x[3][b], c_o1 = x[3][b + 1];
            __m256d e_o0 = x[5][b], e_o1 = x[5][b + 1];
            __m256d g_o0 = x[7][b], g_o1 = x[7][b + 1];

            // Even butterflies - group 0
            __m256d sumCG_e0 = _mm256_add_pd(c_e0, g_e0);
            __m256d difCG_e0 = _mm256_sub_pd(c_e0, g_e0);
            __m256d sumAE_e0 = _mm256_add_pd(a_e0, e_e0);
            __m256d difAE_e0 = _mm256_sub_pd(a_e0, e_e0);

            // Even butterflies - group 1
            __m256d sumCG_e1 = _mm256_add_pd(c_e1, g_e1);
            __m256d difCG_e1 = _mm256_sub_pd(c_e1, g_e1);
            __m256d sumAE_e1 = _mm256_add_pd(a_e1, e_e1);
            __m256d difAE_e1 = _mm256_sub_pd(a_e1, e_e1);

            // Odd butterflies - group 0
            __m256d sumCG_o0 = _mm256_add_pd(c_o0, g_o0);
            __m256d difCG_o0 = _mm256_sub_pd(c_o0, g_o0);
            __m256d sumAE_o0 = _mm256_add_pd(a_o0, e_o0);
            __m256d difAE_o0 = _mm256_sub_pd(a_o0, e_o0);

            // Odd butterflies - group 1
            __m256d sumCG_o1 = _mm256_add_pd(c_o1, g_o1);
            __m256d difCG_o1 = _mm256_sub_pd(c_o1, g_o1);
            __m256d sumAE_o1 = _mm256_add_pd(a_o1, e_o1);
            __m256d difAE_o1 = _mm256_sub_pd(a_o1, e_o1);

            // Complete even butterflies
            e[0][b] = _mm256_add_pd(sumAE_e0, sumCG_e0);
            e[0][b + 1] = _mm256_add_pd(sumAE_e1, sumCG_e1);
            e[2][b] = _mm256_sub_pd(sumAE_e0, sumCG_e0);
            e[2][b + 1] = _mm256_sub_pd(sumAE_e1, sumCG_e1);

            __m256d difCG_swp_e0 = _mm256_permute_pd(difCG_e0, 0b0101);
            __m256d difCG_swp_e1 = _mm256_permute_pd(difCG_e1, 0b0101);
            __m256d rot_e0 = _mm256_xor_pd(difCG_swp_e0, rot_mask);
            __m256d rot_e1 = _mm256_xor_pd(difCG_swp_e1, rot_mask);

            e[1][b] = _mm256_sub_pd(difAE_e0, rot_e0);
            e[1][b + 1] = _mm256_sub_pd(difAE_e1, rot_e1);
            e[3][b] = _mm256_add_pd(difAE_e0, rot_e0);
            e[3][b + 1] = _mm256_add_pd(difAE_e1, rot_e1);

            // Complete odd butterflies
            o[0][b] = _mm256_add_pd(sumAE_o0, sumCG_o0);
            o[0][b + 1] = _mm256_add_pd(sumAE_o1, sumCG_o1);
            o[2][b] = _mm256_sub_pd(sumAE_o0, sumCG_o0);
            o[2][b + 1] = _mm256_sub_pd(sumAE_o1, sumCG_o1);

            __m256d difCG_swp_o0 = _mm256_permute_pd(difCG_o0, 0b0101);
            __m256d difCG_swp_o1 = _mm256_permute_pd(difCG_o1, 0b0101);
            __m256d rot_o0 = _mm256_xor_pd(difCG_swp_o0, rot_mask);
            __m256d rot_o1 = _mm256_xor_pd(difCG_swp_o1, rot_mask);

            o[1][b] = _mm256_sub_pd(difAE_o0, rot_o0);
            o[1][b + 1] = _mm256_sub_pd(difAE_o1, rot_o1);
            o[3][b] = _mm256_add_pd(difAE_o0, rot_o0);
            o[3][b + 1] = _mm256_add_pd(difAE_o1, rot_o1);
        }

        //==================================================================
        // Stage 4: Apply W_8 twiddles with optimized operations
        //==================================================================

        // W_8^1 = (√2/2)(1 - i*sgn) - Process all 4 together
        if (transform_sign == 1)
        {
            // Forward transform
            for (int b = 0; b < 4; ++b)
            {
                __m256d v = o[1][b];
                __m256d real = _mm256_unpacklo_pd(v, v); // [r0,r0,r1,r1]
                __m256d imag = _mm256_unpackhi_pd(v, v); // [i0,i0,i1,i1]
                __m256d sum_ri = _mm256_add_pd(real, imag);
                __m256d dif_ir = _mm256_sub_pd(imag, real);
                __m256d new_real = _mm256_mul_pd(sum_ri, c8);
                __m256d new_imag = _mm256_mul_pd(dif_ir, c8);
                o[1][b] = _mm256_unpacklo_pd(new_real, new_imag);
            }
        }
        else
        {
            // Inverse transform
            for (int b = 0; b < 4; ++b)
            {
                __m256d v = o[1][b];
                __m256d real = _mm256_unpacklo_pd(v, v);
                __m256d imag = _mm256_unpackhi_pd(v, v);
                __m256d dif_ri = _mm256_sub_pd(real, imag);
                __m256d sum_ir = _mm256_add_pd(imag, real);
                __m256d new_real = _mm256_mul_pd(dif_ri, c8);
                __m256d new_imag = _mm256_mul_pd(sum_ir, c8);
                o[1][b] = _mm256_unpacklo_pd(new_real, new_imag);
            }
        }

        // W_8^2 = -i*sgn - Simple swap and sign change
        for (int b = 0; b < 4; ++b)
        {
            __m256d swapped = _mm256_permute_pd(o[2][b], 0b0101);
            o[2][b] = _mm256_xor_pd(swapped, w8_2_mask);
        }

        // W_8^3 = (√2/2)(-1 - i*sgn) - Optimized version
        if (transform_sign == 1)
        {
            // Forward: real' = -(r+i)*√2/2, imag' = -(i+r)*√2/2
            for (int b = 0; b < 4; ++b)
            {
                __m256d v = o[3][b];
                __m256d real = _mm256_unpacklo_pd(v, v);
                __m256d imag = _mm256_unpackhi_pd(v, v);
                __m256d sum = _mm256_add_pd(real, imag);
                __m256d result = _mm256_mul_pd(sum, c8);
                __m256d neg_result = _mm256_xor_pd(result, neg_mask);
                o[3][b] = _mm256_unpacklo_pd(neg_result, neg_result);
            }
        }
        else
        {
            // Inverse: real' = -(r-i)*√2/2, imag' = (r-i)*√2/2
            for (int b = 0; b < 4; ++b)
            {
                __m256d v = o[3][b];
                __m256d real = _mm256_unpacklo_pd(v, v);
                __m256d imag = _mm256_unpackhi_pd(v, v);
                __m256d dif = _mm256_sub_pd(real, imag);
                __m256d result = _mm256_mul_pd(dif, c8);
                __m256d neg_real = _mm256_xor_pd(result, neg_mask);
                o[3][b] = _mm256_unpacklo_pd(neg_real, result);
            }
        }

        //==================================================================
        // Stage 5: Final radix-2 combination with optimized stores
        //==================================================================

        // Process outputs in groups for better cache utilization
        for (int m = 0; m < 4; ++m)
        {
            // Compute all sums first
            __m256d sum0 = _mm256_add_pd(e[m][0], o[m][0]);
            __m256d sum1 = _mm256_add_pd(e[m][1], o[m][1]);
            __m256d sum2 = _mm256_add_pd(e[m][2], o[m][2]);
            __m256d sum3 = _mm256_add_pd(e[m][3], o[m][3]);

            // Then all differences
            __m256d dif0 = _mm256_sub_pd(e[m][0], o[m][0]);
            __m256d dif1 = _mm256_sub_pd(e[m][1], o[m][1]);
            __m256d dif2 = _mm256_sub_pd(e[m][2], o[m][2]);
            __m256d dif3 = _mm256_sub_pd(e[m][3], o[m][3]);

            // Store using STOREU_PD macro (handles alignment checking)
            STOREU_PD(&output_buffer[k + 0 + m * eighth].re, sum0);
            STOREU_PD(&output_buffer[k + 2 + m * eighth].re, sum1);
            STOREU_PD(&output_buffer[k + 4 + m * eighth].re, sum2);
            STOREU_PD(&output_buffer[k + 6 + m * eighth].re, sum3);

            STOREU_PD(&output_buffer[k + 0 + (m + 4) * eighth].re, dif0);
            STOREU_PD(&output_buffer[k + 2 + (m + 4) * eighth].re, dif1);
            STOREU_PD(&output_buffer[k + 4 + (m + 4) * eighth].re, dif2);
            STOREU_PD(&output_buffer[k + 6 + (m + 4) * eighth].re, dif3);
        }
    }

    //----------------------------------------------------------------------
    // Cleanup: 2x unrolling with similar optimizations
    //----------------------------------------------------------------------
    for (; k + 1 < eighth; k += 2)
    {
        // Prefetch for cleanup
        if (k + 8 < eighth)
        {
            _mm_prefetch((const char *)&sub_outputs[k + 8].re, _MM_HINT_T0);
            _mm_prefetch((const char *)&stage_tw[7 * (k + 8)].re, _MM_HINT_T0);
        }

        // Load 8 lanes
        __m256d x[8];

        // Lane 0: no twiddle
        x[0] = load2_aos(&sub_outputs[k], &sub_outputs[k + 1]);

        // Lanes 1-7: apply twiddles
        for (int lane = 1; lane < 8; ++lane)
        {
            __m256d data = load2_aos(&sub_outputs[k + lane * eighth],
                                     &sub_outputs[k + lane * eighth + 1]);
            __m256d tw = load2_aos(&stage_tw[7 * k + (lane - 1)],
                                   &stage_tw[7 * (k + 1) + (lane - 1)]);
            x[lane] = cmul_avx2_aos(data, tw);
        }

        // First radix-4 on evens [0,2,4,6]
        __m256d e[4];
        {
            __m256d a = x[0];
            __m256d c = x[2];
            __m256d e_val = x[4];
            __m256d g = x[6];

            __m256d sumCG = _mm256_add_pd(c, g);
            __m256d difCG = _mm256_sub_pd(c, g);
            __m256d sumAE = _mm256_add_pd(a, e_val);
            __m256d difAE = _mm256_sub_pd(a, e_val);

            e[0] = _mm256_add_pd(sumAE, sumCG);
            e[2] = _mm256_sub_pd(sumAE, sumCG);

            __m256d difCG_swp = _mm256_permute_pd(difCG, 0b0101);
            __m256d rot = _mm256_xor_pd(difCG_swp, rot_mask);

            e[1] = _mm256_sub_pd(difAE, rot);
            e[3] = _mm256_add_pd(difAE, rot);
        }

        // Second radix-4 on odds [1,3,5,7]
        __m256d o[4];
        {
            __m256d a = x[1];
            __m256d c = x[3];
            __m256d e_val = x[5];
            __m256d g = x[7];

            __m256d sumCG = _mm256_add_pd(c, g);
            __m256d difCG = _mm256_sub_pd(c, g);
            __m256d sumAE = _mm256_add_pd(a, e_val);
            __m256d difAE = _mm256_sub_pd(a, e_val);

            o[0] = _mm256_add_pd(sumAE, sumCG);
            o[2] = _mm256_sub_pd(sumAE, sumCG);

            __m256d difCG_swp = _mm256_permute_pd(difCG, 0b0101);
            __m256d rot = _mm256_xor_pd(difCG_swp, rot_mask);

            o[1] = _mm256_sub_pd(difAE, rot);
            o[3] = _mm256_add_pd(difAE, rot);
        }

        // Apply W_8 twiddles to odd results

        // o[1] *= W_8^1 = (√2/2)(1 - i*sgn)
        {
            __m256d real = _mm256_unpacklo_pd(o[1], o[1]);
            __m256d imag = _mm256_unpackhi_pd(o[1], o[1]);

            if (transform_sign == 1)
            {
                __m256d new_r = _mm256_mul_pd(_mm256_add_pd(real, imag), c8);
                __m256d new_i = _mm256_mul_pd(_mm256_sub_pd(imag, real), c8);
                o[1] = _mm256_unpacklo_pd(new_r, new_i);
            }
            else
            {
                __m256d new_r = _mm256_mul_pd(_mm256_sub_pd(real, imag), c8);
                __m256d new_i = _mm256_mul_pd(_mm256_add_pd(imag, real), c8);
                o[1] = _mm256_unpacklo_pd(new_r, new_i);
            }
        }

        // o[2] *= W_8^2 = -i*sgn
        {
            __m256d swapped = _mm256_permute_pd(o[2], 0b0101);
            o[2] = _mm256_xor_pd(swapped, w8_2_mask);
        }

        // o[3] *= W_8^3 = (√2/2)(-1 - i*sgn)
        {
            __m256d real = _mm256_unpacklo_pd(o[3], o[3]);
            __m256d imag = _mm256_unpackhi_pd(o[3], o[3]);

            if (transform_sign == 1)
            {
                __m256d sum = _mm256_add_pd(real, imag);
                __m256d result = _mm256_mul_pd(sum, c8);
                __m256d neg_result = _mm256_xor_pd(result, neg_mask);
                o[3] = _mm256_unpacklo_pd(neg_result, neg_result);
            }
            else
            {
                __m256d dif = _mm256_sub_pd(real, imag);
                __m256d result = _mm256_mul_pd(dif, c8);
                __m256d neg_real = _mm256_xor_pd(result, neg_mask);
                o[3] = _mm256_unpacklo_pd(neg_real, result);
            }
        }

        // Final combination
        for (int m = 0; m < 4; ++m)
        {
            __m256d sum = _mm256_add_pd(e[m], o[m]);
            __m256d dif = _mm256_sub_pd(e[m], o[m]);

            STOREU_PD(&output_buffer[k + m * eighth].re, sum);
            STOREU_PD(&output_buffer[k + (m + 4) * eighth].re, dif);
        }
    }

#endif // __AVX2__

    //======================================================================
    // SCALAR TAIL - Optimized scalar code
    //======================================================================
    for (; k < eighth; ++k)
    {
        // Load 8 lanes
        fft_data x[8];
        x[0] = sub_outputs[k];

        // Apply twiddles W^{jk} for j=1..7 with loop unrolling
        for (int j = 1; j < 8; ++j)
        {
            x[j] = sub_outputs[k + j * eighth];
            fft_data tw = stage_tw[7 * k + (j - 1)];
            double xr = x[j].re, xi = x[j].im;
            x[j].re = xr * tw.re - xi * tw.im;
            x[j].im = xr * tw.im + xi * tw.re;
        }

        // First radix-4 on evens [0,2,4,6]
        fft_data e[4];
        {
            fft_data a = x[0];
            fft_data b = x[2];
            fft_data c = x[4];
            fft_data d = x[6];

            double sumBDr = b.re + d.re, sumBDi = b.im + d.im;
            double difBDr = b.re - d.re, difBDi = b.im - d.im;
            double a_pc_r = a.re + c.re, a_pc_i = a.im + c.im;
            double a_mc_r = a.re - c.re, a_mc_i = a.im - c.im;

            e[0].re = a_pc_r + sumBDr;
            e[0].im = a_pc_i + sumBDi;
            e[2].re = a_pc_r - sumBDr;
            e[2].im = a_pc_i - sumBDi;

            double rotr = (transform_sign == 1) ? -difBDi : difBDi;
            double roti = (transform_sign == 1) ? difBDr : -difBDr;

            e[1].re = a_mc_r - rotr;
            e[1].im = a_mc_i - roti;
            e[3].re = a_mc_r + rotr;
            e[3].im = a_mc_i + roti;
        }

        // Second radix-4 on odds [1,3,5,7]
        fft_data o[4];
        {
            fft_data a = x[1];
            fft_data b = x[3];
            fft_data c = x[5];
            fft_data d = x[7];

            double sumBDr = b.re + d.re, sumBDi = b.im + d.im;
            double difBDr = b.re - d.re, difBDi = b.im - d.im;
            double a_pc_r = a.re + c.re, a_pc_i = a.im + c.im;
            double a_mc_r = a.re - c.re, a_mc_i = a.im - c.im;

            o[0].re = a_pc_r + sumBDr;
            o[0].im = a_pc_i + sumBDi;
            o[2].re = a_pc_r - sumBDr;
            o[2].im = a_pc_i - sumBDi;

            double rotr = (transform_sign == 1) ? -difBDi : difBDi;
            double roti = (transform_sign == 1) ? difBDr : -difBDr;

            o[1].re = a_mc_r - rotr;
            o[1].im = a_mc_i - roti;
            o[3].re = a_mc_r + rotr;
            o[3].im = a_mc_i + roti;
        }

        // Apply W_8 twiddles
        const double c8 = 0.7071067811865476; // √2/2

        // o[1] *= W_8^1 = (√2/2)(1 - i*sgn)
        {
            double r = o[1].re, i = o[1].im;
            if (transform_sign == 1)
            {
                o[1].re = (r + i) * c8;
                o[1].im = (i - r) * c8;
            }
            else
            {
                o[1].re = (r - i) * c8;
                o[1].im = (i + r) * c8;
            }
        }

        // o[2] *= W_8^2 = -i*sgn
        {
            double r = o[2].re, i = o[2].im;
            if (transform_sign == 1)
            {
                o[2].re = i;
                o[2].im = -r;
            }
            else
            {
                o[2].re = -i;
                o[2].im = r;
            }
        }

        // o[3] *= W_8^3 = (√2/2)(-1 - i*sgn)
        {
            double r = o[3].re, i = o[3].im;
            if (transform_sign == 1)
            {
                double neg_sum = -(r + i) * c8;
                o[3].re = neg_sum;
                o[3].im = neg_sum;
            }
            else
            {
                double dif_scaled = (r - i) * c8;
                o[3].re = -dif_scaled;
                o[3].im = dif_scaled;
            }
        }

        // Final combination with direct assignment
        output_buffer[k + 0 * eighth].re = e[0].re + o[0].re;
        output_buffer[k + 0 * eighth].im = e[0].im + o[0].im;
        output_buffer[k + 1 * eighth].re = e[1].re + o[1].re;
        output_buffer[k + 1 * eighth].im = e[1].im + o[1].im;
        output_buffer[k + 2 * eighth].re = e[2].re + o[2].re;
        output_buffer[k + 2 * eighth].im = e[2].im + o[2].im;
        output_buffer[k + 3 * eighth].re = e[3].re + o[3].re;
        output_buffer[k + 3 * eighth].im = e[3].im + o[3].im;

        output_buffer[k + 4 * eighth].re = e[0].re - o[0].re;
        output_buffer[k + 4 * eighth].im = e[0].im - o[0].im;
        output_buffer[k + 5 * eighth].re = e[1].re - o[1].re;
        output_buffer[k + 5 * eighth].im = e[1].im - o[1].im;
        output_buffer[k + 6 * eighth].re = e[2].re - o[2].re;
        output_buffer[k + 6 * eighth].im = e[2].im - o[2].im;
        output_buffer[k + 7 * eighth].re = e[3].re - o[3].re;
        output_buffer[k + 7 * eighth].im = e[3].im - o[3].im;
    }
}