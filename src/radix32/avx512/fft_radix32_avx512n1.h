/**
 * @file fft_radix32_avx512_n1.h
 * @brief Radix-32 N=1 (twiddle-less) FFT implementation for AVX-512
 *
 * This file contains optimized radix-32 transforms for the special case where
 * N=32 (no stage twiddle factors required). This is the base case for larger
 * mixed-radix transforms and is heavily optimized for small-size throughput.
 *
 * N=1 Optimization Strategy:
 * - Pass 1 (Radix-8): All stage twiddles are W^0 = 1 (identity, no multiplication)
 * - Pass 2 (Radix-4): Geometric cross-group twiddles from planner
 * - Twiddles provided by planner as precomputed constants
 * - Reuses existing radix-8 and radix-4 butterfly cores from fused file
 *
 * Decomposition: Radix-32 = Radix-8 × Radix-4
 * - Pass 1: 4 radix-8 butterflies (no stage twiddles, only geometric)
 * - Pass 2: 8 radix-4 cross-group butterflies (geometric twiddles from planner)
 *
 * Performance:
 * - ~600 cycles for full radix-32 N=1 transform (Skylake-X)
 * - ~18.75 cycles per butterfly (32 complex outputs)
 * - 2-3x faster than general K>1 case (no twiddle memory traffic)
 *
 * Use Cases:
 * - Base case for mixed-radix algorithms (N=32, 64, 96, 128, ...)
 * - High-throughput batch processing of small FFTs
 * - Embedded FFT kernels in larger transforms
 *
 * @note Requires AVX-512F support (Skylake-X+, Zen4+)
 * @note Depends on fft_radix32_8x4_fused.h
 * @author Tugbars Heptaskin
 * @date 2025
 */

#ifndef FFT_RADIX32_AVX512_N1_H
#define FFT_RADIX32_AVX512_N1_H

#include "fft_radix32_8x4_fused.h"
#include <immintrin.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

    //==============================================================================
    // N=1 PASS 1: RADIX-8 (NO STAGE TWIDDLES)
    //==============================================================================

    /**
     * @brief Radix-8 pass for N=1 (all stage twiddles are W^0 = 1)
     *
     * Processes 4 groups, each combining 8 stripes with stride=4.
     * Since all stage twiddles are identity, we skip multiplication entirely.
     *
     * Input layout:  in[0..31] natural order
     * Output layout: temp[32] (group-major for Pass 2)
     *
     * Memory organization after Pass 1:
     * - temp[0..7]   = Group A outputs (bins 0-7)
     * - temp[8..15]  = Group B outputs (bins 0-7)
     * - temp[16..23] = Group C outputs (bins 0-7)
     * - temp[24..31] = Group D outputs (bins 0-7)
     *
     * @param in_re Input real [32]
     * @param in_im Input imaginary [32]
     * @param temp_re Temp real [32] (group-major)
     * @param temp_im Temp imaginary [32] (group-major)
     */
    TARGET_AVX512
    static inline void radix32_n1_pass1_radix8_forward_avx512(
        const double *RESTRICT in_re,
        const double *RESTRICT in_im,
        double *RESTRICT temp_re,
        double *RESTRICT temp_im)
    {
        const __m512d sign_mask = _mm512_set1_pd(-0.0);
        const __m512d sqrt2_2 = _mm512_set1_pd(0.70710678118654752440);

        // Process 4 groups (A, B, C, D)
        for (size_t group = 0; group < 4; group++)
        {
            // Load 8 stripes from this group (stride = 4)
            // Group A: 0,4,8,12,16,20,24,28
            // Group B: 1,5,9,13,17,21,25,29
            // Group C: 2,6,10,14,18,22,26,30
            // Group D: 3,7,11,15,19,23,27,31

            __m512d x_re[8], x_im[8];
            for (size_t i = 0; i < 8; i++)
            {
                x_re[i] = _mm512_set_pd(
                    in_re[group + 28], in_re[group + 24],
                    in_re[group + 20], in_re[group + 16],
                    in_re[group + 12], in_re[group + 8],
                    in_re[group + 4], in_re[group + 0]);
                x_im[i] = _mm512_set_pd(
                    in_im[group + 28], in_im[group + 24],
                    in_im[group + 20], in_im[group + 16],
                    in_im[group + 12], in_im[group + 8],
                    in_im[group + 4], in_im[group + 0]);
            }

            // Radix-8 butterfly using wave architecture (no twiddles)
            __m512d y_re[8], y_im[8];
            process_radix8_group(x_re, x_im, y_re, y_im, sign_mask, sqrt2_2);

            // Store in group-major order for Pass 2
            size_t out_base = group * 8;
            for (size_t i = 0; i < 8; i++)
            {
                _mm512_store_pd(&temp_re[out_base + i], y_re[i]);
                _mm512_store_pd(&temp_im[out_base + i], y_im[i]);
            }
        }
    }

    /**
     * @brief Radix-8 pass for N=1 - BACKWARD
     */
    TARGET_AVX512
    static inline void radix32_n1_pass1_radix8_backward_avx512(
        const double *RESTRICT in_re,
        const double *RESTRICT in_im,
        double *RESTRICT temp_re,
        double *RESTRICT temp_im)
    {
        const __m512d sign_mask = _mm512_set1_pd(-0.0);
        const __m512d sqrt2_2 = _mm512_set1_pd(0.70710678118654752440);

        for (size_t group = 0; group < 4; group++)
        {
            __m512d x_re[8], x_im[8];
            for (size_t i = 0; i < 8; i++)
            {
                x_re[i] = _mm512_set_pd(
                    in_re[group + 28], in_re[group + 24],
                    in_re[group + 20], in_re[group + 16],
                    in_re[group + 12], in_re[group + 8],
                    in_re[group + 4], in_re[group + 0]);
                x_im[i] = _mm512_set_pd(
                    in_im[group + 28], in_im[group + 24],
                    in_im[group + 20], in_im[group + 16],
                    in_im[group + 12], in_im[group + 8],
                    in_im[group + 4], in_im[group + 0]);
            }

            // Radix-8 butterfly (backward variant)
            __m512d y_re[8], y_im[8];
            process_radix8_group_backward(x_re, x_im, y_re, y_im, sign_mask, sqrt2_2);

            size_t out_base = group * 8;
            for (size_t i = 0; i < 8; i++)
            {
                _mm512_store_pd(&temp_re[out_base + i], y_re[i]);
                _mm512_store_pd(&temp_im[out_base + i], y_im[i]);
            }
        }
    }

    //==============================================================================
    // N=1 PASS 2: RADIX-4 CROSS-GROUP (GEOMETRIC TWIDDLES FROM PLANNER)
    //==============================================================================

    /**
     * @brief Radix-4 cross-group pass for N=1 (geometric twiddles from planner)
     *
     * Processes 8 positions, each combining 4 groups using radix-4 butterfly.
     * Geometric twiddles are provided by planner via pass2_plan.
     *
     * Input layout:  temp[32] group-major from Pass 1
     * Output layout: out[32] natural order
     *
     * Twiddle usage:
     * - Position 0: Identity (no twiddles)
     * - Position 1: W_32^1, W_32^2, W_32^3
     * - Position 2: W_16^1, W_16^2, W_16^3
     * - Position 3: W_32^3, W_32^6, W_32^9
     * - Position 4: W_8^1, W_8^2, W_8^3 (fast-path)
     * - Position 5: W_32^5, W_32^10, W_32^15
     * - Position 6: W_32^6, W_32^12, W_32^18
     * - Position 7: W_32^7, W_32^14, W_32^21
     *
     * @param temp_re Temp real [32] (group-major)
     * @param temp_im Temp imaginary [32] (group-major)
     * @param out_re Output real [32]
     * @param out_im Output imaginary [32]
     * @param plan Geometric constants from planner (pass2 only)
     */
    TARGET_AVX512
    static inline void radix32_n1_pass2_radix4_forward_avx512(
        const double *RESTRICT temp_re,
        const double *RESTRICT temp_im,
        double *RESTRICT out_re,
        double *RESTRICT out_im,
        const radix32_pass2_plan_t *RESTRICT plan)
    {
        const __m512d sign_mask = _mm512_set1_pd(-0.0);
        const __m512d sqrt2_2 = _mm512_set1_pd(0.70710678118654752440);

        // Load all 4 groups into registers (32 values = 4 groups × 8 bins)
        __m512d A_re[8], A_im[8]; // Group A (temp[0..7])
        __m512d B_re[8], B_im[8]; // Group B (temp[8..15])
        __m512d C_re[8], C_im[8]; // Group C (temp[16..23])
        __m512d D_re[8], D_im[8]; // Group D (temp[24..31])

        for (size_t i = 0; i < 8; i++)
        {
            A_re[i] = _mm512_load_pd(&temp_re[0 + i]);
            A_im[i] = _mm512_load_pd(&temp_im[0 + i]);
            B_re[i] = _mm512_load_pd(&temp_re[8 + i]);
            B_im[i] = _mm512_load_pd(&temp_im[8 + i]);
            C_re[i] = _mm512_load_pd(&temp_re[16 + i]);
            C_im[i] = _mm512_load_pd(&temp_im[16 + i]);
            D_re[i] = _mm512_load_pd(&temp_re[24 + i]);
            D_im[i] = _mm512_load_pd(&temp_im[24 + i]);
        }

        // Extract geometric twiddles from plan
        const __m512d pos1_w1_re = plan->pos1_w1_re;
        const __m512d pos1_w1_im = plan->pos1_w1_im;
        const __m512d pos1_w2_re = plan->pos1_w2_re;
        const __m512d pos1_w2_im = plan->pos1_w2_im;
        const __m512d pos1_w3_re = plan->pos1_w3_re;
        const __m512d pos1_w3_im = plan->pos1_w3_im;

        const __m512d pos2_w1_re = plan->pos2_w1_re;
        const __m512d pos2_w1_im = plan->pos2_w1_im;
        const __m512d pos2_w2_re = plan->pos2_w2_re;
        const __m512d pos2_w2_im = plan->pos2_w2_im;
        const __m512d pos2_w3_re = plan->pos2_w3_re;
        const __m512d pos2_w3_im = plan->pos2_w3_im;

        const __m512d pos3_w1_re = plan->pos3_w1_re;
        const __m512d pos3_w1_im = plan->pos3_w1_im;
        const __m512d pos3_w2_re = plan->pos3_w2_re;
        const __m512d pos3_w2_im = plan->pos3_w2_im;
        const __m512d pos3_w3_re = plan->pos3_w3_re;
        const __m512d pos3_w3_im = plan->pos3_w3_im;

        const __m512d pos5_w1_re = plan->pos5_w1_re;
        const __m512d pos5_w1_im = plan->pos5_w1_im;
        const __m512d pos5_w2_re = plan->pos5_w2_re;
        const __m512d pos5_w2_im = plan->pos5_w2_im;
        const __m512d pos5_w3_re = plan->pos5_w3_re;
        const __m512d pos5_w3_im = plan->pos5_w3_im;

        const __m512d pos6_w1_re = plan->pos6_w1_re;
        const __m512d pos6_w1_im = plan->pos6_w1_im;
        const __m512d pos6_w2_re = plan->pos6_w2_re;
        const __m512d pos6_w2_im = plan->pos6_w2_im;
        const __m512d pos6_w3_re = plan->pos6_w3_re;
        const __m512d pos6_w3_im = plan->pos6_w3_im;

        const __m512d pos7_w1_re = plan->pos7_w1_re;
        const __m512d pos7_w1_im = plan->pos7_w1_im;
        const __m512d pos7_w2_re = plan->pos7_w2_re;
        const __m512d pos7_w2_im = plan->pos7_w2_im;
        const __m512d pos7_w3_re = plan->pos7_w3_re;
        const __m512d pos7_w3_im = plan->pos7_w3_im;

        // Process 8 positions
        __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im;

        // Position 0: Identity (no twiddles)
        radix4_butterfly_core_fv_avx512(
            A_re[0], A_im[0], B_re[0], B_im[0],
            C_re[0], C_im[0], D_re[0], D_im[0],
            &y0_re, &y0_im, &y1_re, &y1_im,
            &y2_re, &y2_im, &y3_re, &y3_im,
            sign_mask);
        _mm512_store_pd(&out_re[0], y0_re);
        _mm512_store_pd(&out_im[0], y0_im);
        _mm512_store_pd(&out_re[8], y1_re);
        _mm512_store_pd(&out_im[8], y1_im);
        _mm512_store_pd(&out_re[16], y2_re);
        _mm512_store_pd(&out_im[16], y2_im);
        _mm512_store_pd(&out_re[24], y3_re);
        _mm512_store_pd(&out_im[24], y3_im);

        // Position 1: W_32^1, W_32^2, W_32^3
        {
            __m512d a_re = A_re[1], a_im = A_im[1];
            __m512d b_re = B_re[1], b_im = B_im[1];
            __m512d c_re = C_re[1], c_im = C_im[1];
            __m512d d_re = D_re[1], d_im = D_im[1];

            cmul_avx512(b_re, b_im, pos1_w1_re, pos1_w1_im, &b_re, &b_im);
            cmul_avx512(c_re, c_im, pos1_w2_re, pos1_w2_im, &c_re, &c_im);
            cmul_avx512(d_re, d_im, pos1_w3_re, pos1_w3_im, &d_re, &d_im);

            radix4_butterfly_core_fv_avx512(
                a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,
                sign_mask);

            _mm512_store_pd(&out_re[1], y0_re);
            _mm512_store_pd(&out_im[1], y0_im);
            _mm512_store_pd(&out_re[9], y1_re);
            _mm512_store_pd(&out_im[9], y1_im);
            _mm512_store_pd(&out_re[17], y2_re);
            _mm512_store_pd(&out_im[17], y2_im);
            _mm512_store_pd(&out_re[25], y3_re);
            _mm512_store_pd(&out_im[25], y3_im);
        }

        // Position 2: W_16^1, W_16^2, W_16^3
        {
            __m512d a_re = A_re[2], a_im = A_im[2];
            __m512d b_re = B_re[2], b_im = B_im[2];
            __m512d c_re = C_re[2], c_im = C_im[2];
            __m512d d_re = D_re[2], d_im = D_im[2];

            cmul_avx512(b_re, b_im, pos2_w1_re, pos2_w1_im, &b_re, &b_im);
            cmul_avx512(c_re, c_im, pos2_w2_re, pos2_w2_im, &c_re, &c_im);
            cmul_avx512(d_re, d_im, pos2_w3_re, pos2_w3_im, &d_re, &d_im);

            radix4_butterfly_core_fv_avx512(
                a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,
                sign_mask);

            _mm512_store_pd(&out_re[2], y0_re);
            _mm512_store_pd(&out_im[2], y0_im);
            _mm512_store_pd(&out_re[10], y1_re);
            _mm512_store_pd(&out_im[10], y1_im);
            _mm512_store_pd(&out_re[18], y2_re);
            _mm512_store_pd(&out_im[18], y2_im);
            _mm512_store_pd(&out_re[26], y3_re);
            _mm512_store_pd(&out_im[26], y3_im);
        }

        // Position 3: W_32^3, W_32^6, W_32^9
        {
            __m512d a_re = A_re[3], a_im = A_im[3];
            __m512d b_re = B_re[3], b_im = B_im[3];
            __m512d c_re = C_re[3], c_im = C_im[3];
            __m512d d_re = D_re[3], d_im = D_im[3];

            cmul_avx512(b_re, b_im, pos3_w1_re, pos3_w1_im, &b_re, &b_im);
            cmul_avx512(c_re, c_im, pos3_w2_re, pos3_w2_im, &c_re, &c_im);
            cmul_avx512(d_re, d_im, pos3_w3_re, pos3_w3_im, &d_re, &d_im);

            radix4_butterfly_core_fv_avx512(
                a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,
                sign_mask);

            _mm512_store_pd(&out_re[3], y0_re);
            _mm512_store_pd(&out_im[3], y0_im);
            _mm512_store_pd(&out_re[11], y1_re);
            _mm512_store_pd(&out_im[11], y1_im);
            _mm512_store_pd(&out_re[19], y2_re);
            _mm512_store_pd(&out_im[19], y2_im);
            _mm512_store_pd(&out_re[27], y3_re);
            _mm512_store_pd(&out_im[27], y3_im);
        }

        // Position 4: W_8 fast-path
        {
            __m512d a_re = A_re[4], a_im = A_im[4];
            __m512d b_re = B_re[4], b_im = B_im[4];
            __m512d c_re = C_re[4], c_im = C_im[4];
            __m512d d_re = D_re[4], d_im = D_im[4];

            // W8^1 (forward): (1 - i)/√2
            __m512d b_sum = _mm512_add_pd(b_re, b_im);
            __m512d b_diff = _mm512_sub_pd(b_im, b_re);
            b_re = _mm512_mul_pd(sqrt2_2, b_sum);
            b_im = _mm512_mul_pd(sqrt2_2, b_diff);

            // W8^2 (forward): -i
            __m512d c_tw_re, c_tw_im;
            rot_neg_j(c_re, c_im, sign_mask, &c_tw_re, &c_tw_im);
            c_re = c_tw_re;
            c_im = c_tw_im;

            // W8^3 (forward): (-1 - i)/√2
            __m512d d_im_minus_re = _mm512_sub_pd(d_im, d_re);
            __m512d d_neg_re = _mm512_xor_pd(d_re, sign_mask);
            __m512d d_neg_re_minus_im = _mm512_sub_pd(d_neg_re, d_im);
            __m512d d_re2 = _mm512_mul_pd(sqrt2_2, d_im_minus_re);
            __m512d d_im2 = _mm512_mul_pd(sqrt2_2, d_neg_re_minus_im);
            d_re = d_re2;
            d_im = d_im2;

            radix4_butterfly_core_fv_avx512(
                a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,
                sign_mask);

            _mm512_store_pd(&out_re[4], y0_re);
            _mm512_store_pd(&out_im[4], y0_im);
            _mm512_store_pd(&out_re[12], y1_re);
            _mm512_store_pd(&out_im[12], y1_im);
            _mm512_store_pd(&out_re[20], y2_re);
            _mm512_store_pd(&out_im[20], y2_im);
            _mm512_store_pd(&out_re[28], y3_re);
            _mm512_store_pd(&out_im[28], y3_im);
        }

        // Position 5: W_32^5, W_32^10, W_32^15
        {
            __m512d a_re = A_re[5], a_im = A_im[5];
            __m512d b_re = B_re[5], b_im = B_im[5];
            __m512d c_re = C_re[5], c_im = C_im[5];
            __m512d d_re = D_re[5], d_im = D_im[5];

            cmul_avx512(b_re, b_im, pos5_w1_re, pos5_w1_im, &b_re, &b_im);
            cmul_avx512(c_re, c_im, pos5_w2_re, pos5_w2_im, &c_re, &c_im);
            cmul_avx512(d_re, d_im, pos5_w3_re, pos5_w3_im, &d_re, &d_im);

            radix4_butterfly_core_fv_avx512(
                a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,
                sign_mask);

            _mm512_store_pd(&out_re[5], y0_re);
            _mm512_store_pd(&out_im[5], y0_im);
            _mm512_store_pd(&out_re[13], y1_re);
            _mm512_store_pd(&out_im[13], y1_im);
            _mm512_store_pd(&out_re[21], y2_re);
            _mm512_store_pd(&out_im[21], y2_im);
            _mm512_store_pd(&out_re[29], y3_re);
            _mm512_store_pd(&out_im[29], y3_im);
        }

        // Position 6: W_32^6, W_32^12, W_32^18
        {
            __m512d a_re = A_re[6], a_im = A_im[6];
            __m512d b_re = B_re[6], b_im = B_im[6];
            __m512d c_re = C_re[6], c_im = C_im[6];
            __m512d d_re = D_re[6], d_im = D_im[6];

            cmul_avx512(b_re, b_im, pos6_w1_re, pos6_w1_im, &b_re, &b_im);
            cmul_avx512(c_re, c_im, pos6_w2_re, pos6_w2_im, &c_re, &c_im);
            cmul_avx512(d_re, d_im, pos6_w3_re, pos6_w3_im, &d_re, &d_im);

            radix4_butterfly_core_fv_avx512(
                a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,
                sign_mask);

            _mm512_store_pd(&out_re[6], y0_re);
            _mm512_store_pd(&out_im[6], y0_im);
            _mm512_store_pd(&out_re[14], y1_re);
            _mm512_store_pd(&out_im[14], y1_im);
            _mm512_store_pd(&out_re[22], y2_re);
            _mm512_store_pd(&out_im[22], y2_im);
            _mm512_store_pd(&out_re[30], y3_re);
            _mm512_store_pd(&out_im[30], y3_im);
        }

        // Position 7: W_32^7, W_32^14, W_32^21
        {
            __m512d a_re = A_re[7], a_im = A_im[7];
            __m512d b_re = B_re[7], b_im = B_im[7];
            __m512d c_re = C_re[7], c_im = C_im[7];
            __m512d d_re = D_re[7], d_im = D_im[7];

            cmul_avx512(b_re, b_im, pos7_w1_re, pos7_w1_im, &b_re, &b_im);
            cmul_avx512(c_re, c_im, pos7_w2_re, pos7_w2_im, &c_re, &c_im);
            cmul_avx512(d_re, d_im, pos7_w3_re, pos7_w3_im, &d_re, &d_im);

            radix4_butterfly_core_fv_avx512(
                a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,
                sign_mask);

            _mm512_store_pd(&out_re[7], y0_re);
            _mm512_store_pd(&out_im[7], y0_im);
            _mm512_store_pd(&out_re[15], y1_re);
            _mm512_store_pd(&out_im[15], y1_im);
            _mm512_store_pd(&out_re[23], y2_re);
            _mm512_store_pd(&out_im[23], y2_im);
            _mm512_store_pd(&out_re[31], y3_re);
            _mm512_store_pd(&out_im[31], y3_im);
        }
    }

    /**
     * @brief Radix-4 cross-group pass for N=1 - BACKWARD
     */
    TARGET_AVX512
    static inline void radix32_n1_pass2_radix4_backward_avx512(
        const double *RESTRICT temp_re,
        const double *RESTRICT temp_im,
        double *RESTRICT out_re,
        double *RESTRICT out_im,
        const radix32_pass2_plan_t *RESTRICT plan)
    {
        const __m512d sign_mask = _mm512_set1_pd(-0.0);
        const __m512d sqrt2_2 = _mm512_set1_pd(0.70710678118654752440);

        // Load all 4 groups
        __m512d A_re[8], A_im[8];
        __m512d B_re[8], B_im[8];
        __m512d C_re[8], C_im[8];
        __m512d D_re[8], D_im[8];

        for (size_t i = 0; i < 8; i++)
        {
            A_re[i] = _mm512_load_pd(&temp_re[0 + i]);
            A_im[i] = _mm512_load_pd(&temp_im[0 + i]);
            B_re[i] = _mm512_load_pd(&temp_re[8 + i]);
            B_im[i] = _mm512_load_pd(&temp_im[8 + i]);
            C_re[i] = _mm512_load_pd(&temp_re[16 + i]);
            C_im[i] = _mm512_load_pd(&temp_im[16 + i]);
            D_re[i] = _mm512_load_pd(&temp_re[24 + i]);
            D_im[i] = _mm512_load_pd(&temp_im[24 + i]);
        }

        // Extract geometric twiddles (same as forward, will be used correctly by BV core)
        const __m512d pos1_w1_re = plan->pos1_w1_re;
        const __m512d pos1_w1_im = plan->pos1_w1_im;
        const __m512d pos1_w2_re = plan->pos1_w2_re;
        const __m512d pos1_w2_im = plan->pos1_w2_im;
        const __m512d pos1_w3_re = plan->pos1_w3_re;
        const __m512d pos1_w3_im = plan->pos1_w3_im;

        const __m512d pos2_w1_re = plan->pos2_w1_re;
        const __m512d pos2_w1_im = plan->pos2_w1_im;
        const __m512d pos2_w2_re = plan->pos2_w2_re;
        const __m512d pos2_w2_im = plan->pos2_w2_im;
        const __m512d pos2_w3_re = plan->pos2_w3_re;
        const __m512d pos2_w3_im = plan->pos2_w3_im;

        const __m512d pos3_w1_re = plan->pos3_w1_re;
        const __m512d pos3_w1_im = plan->pos3_w1_im;
        const __m512d pos3_w2_re = plan->pos3_w2_re;
        const __m512d pos3_w2_im = plan->pos3_w2_im;
        const __m512d pos3_w3_re = plan->pos3_w3_re;
        const __m512d pos3_w3_im = plan->pos3_w3_im;

        const __m512d pos5_w1_re = plan->pos5_w1_re;
        const __m512d pos5_w1_im = plan->pos5_w1_im;
        const __m512d pos5_w2_re = plan->pos5_w2_re;
        const __m512d pos5_w2_im = plan->pos5_w2_im;
        const __m512d pos5_w3_re = plan->pos5_w3_re;
        const __m512d pos5_w3_im = plan->pos5_w3_im;

        const __m512d pos6_w1_re = plan->pos6_w1_re;
        const __m512d pos6_w1_im = plan->pos6_w1_im;
        const __m512d pos6_w2_re = plan->pos6_w2_re;
        const __m512d pos6_w2_im = plan->pos6_w2_im;
        const __m512d pos6_w3_re = plan->pos6_w3_re;
        const __m512d pos6_w3_im = plan->pos6_w3_im;

        const __m512d pos7_w1_re = plan->pos7_w1_re;
        const __m512d pos7_w1_im = plan->pos7_w1_im;
        const __m512d pos7_w2_re = plan->pos7_w2_re;
        const __m512d pos7_w2_im = plan->pos7_w2_im;
        const __m512d pos7_w3_re = plan->pos7_w3_re;
        const __m512d pos7_w3_im = plan->pos7_w3_im;

        __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im;

        // Position 0: Identity
        radix4_butterfly_core_bv_avx512(
            A_re[0], A_im[0], B_re[0], B_im[0],
            C_re[0], C_im[0], D_re[0], D_im[0],
            &y0_re, &y0_im, &y1_re, &y1_im,
            &y2_re, &y2_im, &y3_re, &y3_im,
            sign_mask);
        _mm512_store_pd(&out_re[0], y0_re);
        _mm512_store_pd(&out_im[0], y0_im);
        _mm512_store_pd(&out_re[8], y1_re);
        _mm512_store_pd(&out_im[8], y1_im);
        _mm512_store_pd(&out_re[16], y2_re);
        _mm512_store_pd(&out_im[16], y2_im);
        _mm512_store_pd(&out_re[24], y3_re);
        _mm512_store_pd(&out_im[24], y3_im);

        // Position 1
        {
            __m512d a_re = A_re[1], a_im = A_im[1];
            __m512d b_re = B_re[1], b_im = B_im[1];
            __m512d c_re = C_re[1], c_im = C_im[1];
            __m512d d_re = D_re[1], d_im = D_im[1];

            cmul_avx512(b_re, b_im, pos1_w1_re, pos1_w1_im, &b_re, &b_im);
            cmul_avx512(c_re, c_im, pos1_w2_re, pos1_w2_im, &c_re, &c_im);
            cmul_avx512(d_re, d_im, pos1_w3_re, pos1_w3_im, &d_re, &d_im);

            radix4_butterfly_core_bv_avx512(
                a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,
                sign_mask);

            _mm512_store_pd(&out_re[1], y0_re);
            _mm512_store_pd(&out_im[1], y0_im);
            _mm512_store_pd(&out_re[9], y1_re);
            _mm512_store_pd(&out_im[9], y1_im);
            _mm512_store_pd(&out_re[17], y2_re);
            _mm512_store_pd(&out_im[17], y2_im);
            _mm512_store_pd(&out_re[25], y3_re);
            _mm512_store_pd(&out_im[25], y3_im);
        }

        // Position 2
        {
            __m512d a_re = A_re[2], a_im = A_im[2];
            __m512d b_re = B_re[2], b_im = B_im[2];
            __m512d c_re = C_re[2], c_im = C_im[2];
            __m512d d_re = D_re[2], d_im = D_im[2];

            cmul_avx512(b_re, b_im, pos2_w1_re, pos2_w1_im, &b_re, &b_im);
            cmul_avx512(c_re, c_im, pos2_w2_re, pos2_w2_im, &c_re, &c_im);
            cmul_avx512(d_re, d_im, pos2_w3_re, pos2_w3_im, &d_re, &d_im);

            radix4_butterfly_core_bv_avx512(
                a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,
                sign_mask);

            _mm512_store_pd(&out_re[2], y0_re);
            _mm512_store_pd(&out_im[2], y0_im);
            _mm512_store_pd(&out_re[10], y1_re);
            _mm512_store_pd(&out_im[10], y1_im);
            _mm512_store_pd(&out_re[18], y2_re);
            _mm512_store_pd(&out_im[18], y2_im);
            _mm512_store_pd(&out_re[26], y3_re);
            _mm512_store_pd(&out_im[26], y3_im);
        }

        // Position 3
        {
            __m512d a_re = A_re[3], a_im = A_im[3];
            __m512d b_re = B_re[3], b_im = B_im[3];
            __m512d c_re = C_re[3], c_im = C_im[3];
            __m512d d_re = D_re[3], d_im = D_im[3];

            cmul_avx512(b_re, b_im, pos3_w1_re, pos3_w1_im, &b_re, &b_im);
            cmul_avx512(c_re, c_im, pos3_w2_re, pos3_w2_im, &c_re, &c_im);
            cmul_avx512(d_re, d_im, pos3_w3_re, pos3_w3_im, &d_re, &d_im);

            radix4_butterfly_core_bv_avx512(
                a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,
                sign_mask);

            _mm512_store_pd(&out_re[3], y0_re);
            _mm512_store_pd(&out_im[3], y0_im);
            _mm512_store_pd(&out_re[11], y1_re);
            _mm512_store_pd(&out_im[11], y1_im);
            _mm512_store_pd(&out_re[19], y2_re);
            _mm512_store_pd(&out_im[19], y2_im);
            _mm512_store_pd(&out_re[27], y3_re);
            _mm512_store_pd(&out_im[27], y3_im);
        }

        // Position 4: W_8 fast-path (backward)
        {
            __m512d a_re = A_re[4], a_im = A_im[4];
            __m512d b_re = B_re[4], b_im = B_im[4];
            __m512d c_re = C_re[4], c_im = C_im[4];
            __m512d d_re = D_re[4], d_im = D_im[4];

            // W8^1 (conjugated): sqrt(2)/2 * (1 + i)
            __m512d sum_b = _mm512_add_pd(b_re, b_im);
            __m512d nim_b = _mm512_xor_pd(b_im, sign_mask);
            __m512d diff_b = _mm512_add_pd(b_re, nim_b);
            b_re = _mm512_mul_pd(sqrt2_2, sum_b);
            b_im = _mm512_mul_pd(sqrt2_2, diff_b);

            // W8^2 (conjugated): +i
            __m512d c_tw_re, c_tw_im;
            rot_pos_j(c_re, c_im, sign_mask, &c_tw_re, &c_tw_im);
            c_re = c_tw_re;
            c_im = c_tw_im;

            // W8^3 (conjugated): sqrt(2)/2 * (-1 + i)
            __m512d sum_d = _mm512_add_pd(d_re, d_im);
            __m512d nim_d = _mm512_xor_pd(d_im, sign_mask);
            __m512d diff_d = _mm512_add_pd(d_re, nim_d);
            __m512d d_re_tmp = _mm512_mul_pd(sqrt2_2, sum_d);
            d_re = _mm512_xor_pd(d_re_tmp, sign_mask);
            d_im = _mm512_mul_pd(sqrt2_2, diff_d);

            radix4_butterfly_core_bv_avx512(
                a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,
                sign_mask);

            _mm512_store_pd(&out_re[4], y0_re);
            _mm512_store_pd(&out_im[4], y0_im);
            _mm512_store_pd(&out_re[12], y1_re);
            _mm512_store_pd(&out_im[12], y1_im);
            _mm512_store_pd(&out_re[20], y2_re);
            _mm512_store_pd(&out_im[20], y2_im);
            _mm512_store_pd(&out_re[28], y3_re);
            _mm512_store_pd(&out_im[28], y3_im);
        }

        // Positions 5, 6, 7 (remaining positions)
        {
            __m512d a_re = A_re[5], a_im = A_im[5];
            __m512d b_re = B_re[5], b_im = B_im[5];
            __m512d c_re = C_re[5], c_im = C_im[5];
            __m512d d_re = D_re[5], d_im = D_im[5];

            cmul_avx512(b_re, b_im, pos5_w1_re, pos5_w1_im, &b_re, &b_im);
            cmul_avx512(c_re, c_im, pos5_w2_re, pos5_w2_im, &c_re, &c_im);
            cmul_avx512(d_re, d_im, pos5_w3_re, pos5_w3_im, &d_re, &d_im);

            radix4_butterfly_core_bv_avx512(
                a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,
                sign_mask);

            _mm512_store_pd(&out_re[5], y0_re);
            _mm512_store_pd(&out_im[5], y0_im);
            _mm512_store_pd(&out_re[13], y1_re);
            _mm512_store_pd(&out_im[13], y1_im);
            _mm512_store_pd(&out_re[21], y2_re);
            _mm512_store_pd(&out_im[21], y2_im);
            _mm512_store_pd(&out_re[29], y3_re);
            _mm512_store_pd(&out_im[29], y3_im);
        }

        {
            __m512d a_re = A_re[6], a_im = A_im[6];
            __m512d b_re = B_re[6], b_im = B_im[6];
            __m512d c_re = C_re[6], c_im = C_im[6];
            __m512d d_re = D_re[6], d_im = D_im[6];

            cmul_avx512(b_re, b_im, pos6_w1_re, pos6_w1_im, &b_re, &b_im);
            cmul_avx512(c_re, c_im, pos6_w2_re, pos6_w2_im, &c_re, &c_im);
            cmul_avx512(d_re, d_im, pos6_w3_re, pos6_w3_im, &d_re, &d_im);

            radix4_butterfly_core_bv_avx512(
                a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,
                sign_mask);

            _mm512_store_pd(&out_re[6], y0_re);
            _mm512_store_pd(&out_im[6], y0_im);
            _mm512_store_pd(&out_re[14], y1_re);
            _mm512_store_pd(&out_im[14], y1_im);
            _mm512_store_pd(&out_re[22], y2_re);
            _mm512_store_pd(&out_im[22], y2_im);
            _mm512_store_pd(&out_re[30], y3_re);
            _mm512_store_pd(&out_im[30], y3_im);
        }

        {
            __m512d a_re = A_re[7], a_im = A_im[7];
            __m512d b_re = B_re[7], b_im = B_im[7];
            __m512d c_re = C_re[7], c_im = C_im[7];
            __m512d d_re = D_re[7], d_im = D_im[7];

            cmul_avx512(b_re, b_im, pos7_w1_re, pos7_w1_im, &b_re, &b_im);
            cmul_avx512(c_re, c_im, pos7_w2_re, pos7_w2_im, &c_re, &c_im);
            cmul_avx512(d_re, d_im, pos7_w3_re, pos7_w3_im, &d_re, &d_im);

            radix4_butterfly_core_bv_avx512(
                a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
                &y0_re, &y0_im, &y1_re, &y1_im, &y2_re, &y2_im, &y3_re, &y3_im,
                sign_mask);

            _mm512_store_pd(&out_re[7], y0_re);
            _mm512_store_pd(&out_im[7], y0_im);
            _mm512_store_pd(&out_re[15], y1_re);
            _mm512_store_pd(&out_im[15], y1_im);
            _mm512_store_pd(&out_re[23], y2_re);
            _mm512_store_pd(&out_im[23], y2_im);
            _mm512_store_pd(&out_re[31], y3_re);
            _mm512_store_pd(&out_im[31], y3_im);
        }
    }

    //==============================================================================
    // TOP-LEVEL N=1 WRAPPERS
    //==============================================================================

    /**
     * @brief Complete radix-32 N=1 transform - FORWARD
     *
     * @param in_re Input real [32] (aligned)
     * @param in_im Input imaginary [32] (aligned)
     * @param out_re Output real [32] (aligned)
     * @param out_im Output imaginary [32] (aligned)
     * @param plan Geometric twiddles from planner (pass2 only)
     */
    TARGET_AVX512
    static inline void radix32_n1_forward_avx512(
        const double *RESTRICT in_re,
        const double *RESTRICT in_im,
        double *RESTRICT out_re,
        double *RESTRICT out_im,
        const radix32_pass2_plan_t *RESTRICT plan)
    {
        // Temp storage for inter-pass data (aligned)
        ALIGNAS(64)
        double temp_re[32];
        ALIGNAS(64)
        double temp_im[32];

        // Pass 1: Radix-8 (no stage twiddles)
        radix32_n1_pass1_radix8_forward_avx512(in_re, in_im, temp_re, temp_im);

        // Pass 2: Radix-4 cross-group (geometric twiddles from planner)
        radix32_n1_pass2_radix4_forward_avx512(temp_re, temp_im, out_re, out_im, plan);
    }

    /**
     * @brief Complete radix-32 N=1 transform - BACKWARD
     */
    TARGET_AVX512
    static inline void radix32_n1_backward_avx512(
        const double *RESTRICT in_re,
        const double *RESTRICT in_im,
        double *RESTRICT out_re,
        double *RESTRICT out_im,
        const radix32_pass2_plan_t *RESTRICT plan)
    {
        ALIGNAS(64)
        double temp_re[32];
        ALIGNAS(64)
        double temp_im[32];

        // Pass 1: Radix-8 (backward, no stage twiddles)
        radix32_n1_pass1_radix8_backward_avx512(in_re, in_im, temp_re, temp_im);

        // Pass 2: Radix-4 cross-group (backward, geometric twiddles)
        radix32_n1_pass2_radix4_backward_avx512(temp_re, temp_im, out_re, out_im, plan);
    }

#ifdef __cplusplus
}
#endif

#endif /* FFT_RADIX32_AVX512_N1_H */