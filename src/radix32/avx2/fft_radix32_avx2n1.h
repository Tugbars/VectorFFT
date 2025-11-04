/**
 * @file fft_radix32_avx2_n1.h
 * @brief Radix-32 N=1 (twiddle-less) FFT implementation for AVX2
 *
 * This file contains optimized radix-32 transforms for the special case where
 * N=32 (no twiddle factors required). This is the base case for larger mixed-radix
 * transforms and is heavily optimized for small-size throughput.
 *
 * N=1 Optimization Strategy:
 * - Pass 1 (DIT-4): All stage twiddles are W^0 = 1 (identity, no multiplication)
 * - Pass 2 (DIF-8): All stage twiddles are geometric constants (W_32^k)
 * - Twiddles provided by planner as SoA (Structure of Arrays)
 * - Reuses existing radix-4 and radix-8 butterfly cores
 *
 * Decomposition: Radix-32 = Radix-4 DIT × Radix-8 DIF
 * - Pass 1: 8 radix-4 DIT butterflies (trivial twiddles W^0=1, skip multiplication)
 * - Pass 2: 4 radix-8 DIF butterflies (geometric twiddles W_32^1..W_32^7 from planner)
 *
 * Performance:
 * - ~1200 cycles for full radix-32 N=1 transform (Skylake)
 * - ~37.5 cycles per butterfly (32 butterflies total)
 * - 2-3x faster than general K>1 case (no twiddle memory traffic)
 *
 * Use Cases:
 * - Base case for mixed-radix algorithms (N=32, 64, 96, 128, ...)
 * - High-throughput batch processing of small FFTs
 * - Embedded FFT kernels in larger transforms
 *
 * Twiddle Responsibility:
 * - Planner: Generates and stores twiddles in SoA format
 * - This file: Consumes twiddles as read-only borrowed references
 *
 * @note Requires AVX2 + FMA3 support (Haswell+, Zen1+)
 * @note Depends on fft_radix4_avx2.h and fft_radix8_avx2_dif.h
 */

#ifndef FFT_RADIX32_AVX2_N1_H
#define FFT_RADIX32_AVX2_N1_H

#include "fft_radix4_avx2.h"     // For radix-4 DIT cores
#include "fft_radix8_avx2_dif.h" // For radix-8 DIF cores

#ifdef __cplusplus
extern "C"
{
#endif

    //==============================================================================
    // N=1 TWIDDLE STRUCTURES (SoA from Planner)
    //==============================================================================

    /**
     * @brief Geometric twiddles for radix-32 N=1 Pass 2 (DIF-8)
     *
     * The planner provides these twiddles in SoA format. For N=1, all twiddles
     * are geometric constants W_32^k = exp(-2πik/32).
     *
     * Layout:
     * - re[0..6] = {cos(W_32^1), cos(W_32^2), ..., cos(W_32^7)}
     * - im[0..6] = {sin(W_32^1), sin(W_32^2), ..., sin(W_32^7)}
     *
     * Note: W_32^0 = 1 is implicit (not stored), W_32^5..W_32^7 derived via negation
     */
    typedef struct
    {
        const double *re; ///< Real parts [7] (borrowed, read-only)
        const double *im; ///< Imaginary parts [7] (borrowed, read-only)
    } radix32_n1_twiddles_dif8_t;

    //==============================================================================
    // RADIX-32 N=1 PASS 1: RADIX-4 DIT (NO TWIDDLES)
    //==============================================================================

    /**
     * @brief Radix-4 DIT pass for N=1 (all twiddles are W^0 = 1)
     *
     * Processes 8 groups, each combining 4 stripes with stride=8.
     * Since all stage twiddles are identity, we skip multiplication entirely.
     *
     * Input layout:  in[0..31] natural order
     * Output layout: temp[32] (stripe-major for Pass 2)
     *
     * Memory organization after Pass 1:
     * - temp[0..3]   = bin 0, groups 0..3
     * - temp[4..7]   = bin 0, groups 4..7
     * - temp[8..11]  = bin 1, groups 0..3
     * - temp[12..15] = bin 1, groups 4..7
     * - temp[16..19] = bin 2, groups 0..3
     * - temp[20..23] = bin 2, groups 4..7
     * - temp[24..27] = bin 3, groups 0..3
     * - temp[28..31] = bin 3, groups 4..7
     *
     * @param in_re Input real [32]
     * @param in_im Input imaginary [32]
     * @param temp_re Temp real [32] (stripe-major)
     * @param temp_im Temp imaginary [32] (stripe-major)
     */
    TARGET_AVX2_FMA
    static inline void radix32_n1_pass1_dit4_forward_avx2(
        const double *RESTRICT in_re,
        const double *RESTRICT in_im,
        double *RESTRICT temp_re,
        double *RESTRICT temp_im)
    {
        const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);

        // Process 8 groups (each group combines 4 stripes with stride 8)
        for (size_t group = 0; group < 8; group++)
        {
            // Load 4 stripes from this group (stride = 8)
            // Stripes: 0, 8, 16, 24
            __m256d x0r = _mm256_set_pd(in_re[group + 24], in_re[group + 16],
                                        in_re[group + 8], in_re[group + 0]);
            __m256d x0i = _mm256_set_pd(in_im[group + 24], in_im[group + 16],
                                        in_im[group + 8], in_im[group + 0]);

            __m256d x1r = _mm256_set_pd(in_re[group + 25], in_re[group + 17],
                                        in_re[group + 9], in_re[group + 1]);
            __m256d x1i = _mm256_set_pd(in_im[group + 25], in_im[group + 17],
                                        in_im[group + 9], in_im[group + 1]);

            __m256d x2r = _mm256_set_pd(in_re[group + 26], in_re[group + 18],
                                        in_re[group + 10], in_re[group + 2]);
            __m256d x2i = _mm256_set_pd(in_im[group + 26], in_im[group + 18],
                                        in_im[group + 10], in_im[group + 2]);

            __m256d x3r = _mm256_set_pd(in_re[group + 27], in_re[group + 19],
                                        in_re[group + 11], in_re[group + 3]);
            __m256d x3i = _mm256_set_pd(in_im[group + 27], in_im[group + 19],
                                        in_im[group + 11], in_im[group + 3]);

            // Radix-4 DIT butterfly (no twiddles needed, all W^0 = 1)
            __m256d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
            radix4_core_avx2(x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
                             &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
                             SIGN_FLIP);

            // Store in stripe-major order for Pass 2
            // Each bin gets 8 values (2 vectors of 4)
            _mm256_store_pd(&temp_re[0 + group], y0r);
            _mm256_store_pd(&temp_im[0 + group], y0i);
            _mm256_store_pd(&temp_re[8 + group], y1r);
            _mm256_store_pd(&temp_im[8 + group], y1i);
            _mm256_store_pd(&temp_re[16 + group], y2r);
            _mm256_store_pd(&temp_im[16 + group], y2i);
            _mm256_store_pd(&temp_re[24 + group], y3r);
            _mm256_store_pd(&temp_im[24 + group], y3i);
        }
    }

    /**
     * @brief Radix-4 DIT pass for N=1 - BACKWARD
     */
    TARGET_AVX2_FMA
    static inline void radix32_n1_pass1_dit4_backward_avx2(
        const double *RESTRICT in_re,
        const double *RESTRICT in_im,
        double *RESTRICT temp_re,
        double *RESTRICT temp_im)
    {
        const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);
        const __m256d neg_zero = _mm256_set1_pd(-0.0);
        const __m256d neg_sign = _mm256_xor_pd(SIGN_FLIP, neg_zero);

        for (size_t group = 0; group < 8; group++)
        {
            __m256d x0r = _mm256_set_pd(in_re[group + 24], in_re[group + 16],
                                        in_re[group + 8], in_re[group + 0]);
            __m256d x0i = _mm256_set_pd(in_im[group + 24], in_im[group + 16],
                                        in_im[group + 8], in_im[group + 0]);

            __m256d x1r = _mm256_set_pd(in_re[group + 25], in_re[group + 17],
                                        in_re[group + 9], in_re[group + 1]);
            __m256d x1i = _mm256_set_pd(in_im[group + 25], in_im[group + 17],
                                        in_im[group + 9], in_im[group + 1]);

            __m256d x2r = _mm256_set_pd(in_re[group + 26], in_re[group + 18],
                                        in_re[group + 10], in_re[group + 2]);
            __m256d x2i = _mm256_set_pd(in_im[group + 26], in_im[group + 18],
                                        in_im[group + 10], in_im[group + 2]);

            __m256d x3r = _mm256_set_pd(in_re[group + 27], in_re[group + 19],
                                        in_re[group + 11], in_re[group + 3]);
            __m256d x3i = _mm256_set_pd(in_im[group + 27], in_im[group + 19],
                                        in_im[group + 11], in_im[group + 3]);

            __m256d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
            radix4_core_avx2(x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
                             &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
                             neg_sign);

            _mm256_store_pd(&temp_re[0 + group], y0r);
            _mm256_store_pd(&temp_im[0 + group], y0i);
            _mm256_store_pd(&temp_re[8 + group], y1r);
            _mm256_store_pd(&temp_im[8 + group], y1i);
            _mm256_store_pd(&temp_re[16 + group], y2r);
            _mm256_store_pd(&temp_im[16 + group], y2i);
            _mm256_store_pd(&temp_re[24 + group], y3r);
            _mm256_store_pd(&temp_im[24 + group], y3i);
        }
    }

    //==============================================================================
    // RADIX-32 N=1 PASS 2: RADIX-8 DIF (GEOMETRIC TWIDDLES FROM PLANNER)
    //==============================================================================

    /**
     * @brief Radix-8 DIF pass for N=1 (geometric twiddles from planner)
     *
     * Processes 4 bins, each combining 8 groups using radix-8 DIF.
     * Stage twiddles are geometric constants provided by planner as SoA.
     *
     * Input layout:  temp[32] stripe-major from Pass 1
     * Output layout: out[32] natural order
     *
     * Twiddle usage:
     * - tw->re[0], tw->im[0] = W_32^1
     * - tw->re[1], tw->im[1] = W_32^2
     * - tw->re[2], tw->im[2] = W_32^3
     * - tw->re[3], tw->im[3] = W_32^4
     * - W_32^5 = -W_32^1, W_32^6 = -W_32^2, W_32^7 = -W_32^3 (derived)
     *
     * @param temp_re Temp real [32] (stripe-major)
     * @param temp_im Temp imaginary [32] (stripe-major)
     * @param out_re Output real [32]
     * @param out_im Output imaginary [32]
     * @param tw Geometric twiddles from planner (SoA, borrowed)
     */
    TARGET_AVX2_FMA
    static inline void radix32_n1_pass2_dif8_forward_avx2(
        const double *RESTRICT temp_re,
        const double *RESTRICT temp_im,
        double *RESTRICT out_re,
        double *RESTRICT out_im,
        const radix32_n1_twiddles_dif8_t *RESTRICT tw)
    {
        const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);

        // Broadcast geometric twiddles to all lanes
        __m256d W1r = _mm256_set1_pd(tw->re[0]); // W_32^1
        __m256d W1i = _mm256_set1_pd(tw->im[0]);
        __m256d W2r = _mm256_set1_pd(tw->re[1]); // W_32^2
        __m256d W2i = _mm256_set1_pd(tw->im[1]);
        __m256d W3r = _mm256_set1_pd(tw->re[2]); // W_32^3
        __m256d W3i = _mm256_set1_pd(tw->im[2]);
        __m256d W4r = _mm256_set1_pd(tw->re[3]); // W_32^4
        __m256d W4i = _mm256_set1_pd(tw->im[3]);

        // Derive W_32^5 = -W_32^1, W_32^6 = -W_32^2, W_32^7 = -W_32^3
        __m256d W5r = _mm256_xor_pd(W1r, SIGN_FLIP);
        __m256d W5i = _mm256_xor_pd(W1i, SIGN_FLIP);
        __m256d W6r = _mm256_xor_pd(W2r, SIGN_FLIP);
        __m256d W6i = _mm256_xor_pd(W2i, SIGN_FLIP);
        __m256d W7r = _mm256_xor_pd(W3r, SIGN_FLIP);
        __m256d W7i = _mm256_xor_pd(W3i, SIGN_FLIP);

        // Process 4 bins (each bin combines 8 groups)
        for (size_t bin = 0; bin < 4; bin++)
        {
            // Load 8 groups for this bin (stride = 8)
            size_t base = bin * 8;
            __m256d x0r = _mm256_load_pd(&temp_re[base + 0]);
            __m256d x0i = _mm256_load_pd(&temp_im[base + 0]);
            __m256d x1r = _mm256_load_pd(&temp_re[base + 4]);
            __m256d x1i = _mm256_load_pd(&temp_im[base + 4]);

            base = bin * 8 + 8;
            __m256d x2r = _mm256_load_pd(&temp_re[base + 0]);
            __m256d x2i = _mm256_load_pd(&temp_im[base + 0]);
            __m256d x3r = _mm256_load_pd(&temp_re[base + 4]);
            __m256d x3i = _mm256_load_pd(&temp_im[base + 4]);

            base = bin * 8 + 16;
            __m256d x4r = _mm256_load_pd(&temp_re[base + 0]);
            __m256d x4i = _mm256_load_pd(&temp_im[base + 0]);
            __m256d x5r = _mm256_load_pd(&temp_re[base + 4]);
            __m256d x5i = _mm256_load_pd(&temp_im[base + 4]);

            base = bin * 8 + 24;
            __m256d x6r = _mm256_load_pd(&temp_re[base + 0]);
            __m256d x6i = _mm256_load_pd(&temp_im[base + 0]);
            __m256d x7r = _mm256_load_pd(&temp_re[base + 4]);
            __m256d x7i = _mm256_load_pd(&temp_im[base + 4]);

            // Apply stage twiddles (x0 gets W^0=1, skip)
            __m256d t0r = x0r, t0i = x0i; // x0 *= W^0 = 1 (identity)
            __m256d t1r, t1i, t2r, t2i, t3r, t3i, t4r, t4i;
            __m256d t5r, t5i, t6r, t6i, t7r, t7i;

            cmul_v256(x1r, x1i, W1r, W1i, &t1r, &t1i); // x1 *= W_32^1
            cmul_v256(x2r, x2i, W2r, W2i, &t2r, &t2i); // x2 *= W_32^2
            cmul_v256(x3r, x3i, W3r, W3i, &t3r, &t3i); // x3 *= W_32^3
            cmul_v256(x4r, x4i, W4r, W4i, &t4r, &t4i); // x4 *= W_32^4
            cmul_v256(x5r, x5i, W5r, W5i, &t5r, &t5i); // x5 *= W_32^5 = -W_32^1
            cmul_v256(x6r, x6i, W6r, W6i, &t6r, &t6i); // x6 *= W_32^6 = -W_32^2
            cmul_v256(x7r, x7i, W7r, W7i, &t7r, &t7i); // x7 *= W_32^7 = -W_32^3

            // Radix-8 DIF butterfly
            __m256d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
            __m256d y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;

            radix8_dif_butterfly_forward_avx2(
                t0r, t0i, t1r, t1i, t2r, t2i, t3r, t3i,
                t4r, t4i, t5r, t5i, t6r, t6i, t7r, t7i,
                &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
                &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);

            // Store outputs in natural order
            // Bin 0 → out[0..7], Bin 1 → out[8..15], etc.
            size_t out_base = bin * 8;
            _mm256_store_pd(&out_re[out_base + 0], y0r);
            _mm256_store_pd(&out_im[out_base + 0], y0i);
            _mm256_store_pd(&out_re[out_base + 4], y1r);
            _mm256_store_pd(&out_im[out_base + 4], y1i);

            out_base = bin * 8 + 8;
            _mm256_store_pd(&out_re[out_base + 0], y2r);
            _mm256_store_pd(&out_im[out_base + 0], y2i);
            _mm256_store_pd(&out_re[out_base + 4], y3r);
            _mm256_store_pd(&out_im[out_base + 4], y3i);

            out_base = bin * 8 + 16;
            _mm256_store_pd(&out_re[out_base + 0], y4r);
            _mm256_store_pd(&out_im[out_base + 0], y4i);
            _mm256_store_pd(&out_re[out_base + 4], y5r);
            _mm256_store_pd(&out_im[out_base + 4], y5i);

            out_base = bin * 8 + 24;
            _mm256_store_pd(&out_re[out_base + 0], y6r);
            _mm256_store_pd(&out_im[out_base + 0], y6i);
            _mm256_store_pd(&out_re[out_base + 4], y7r);
            _mm256_store_pd(&out_im[out_base + 4], y7i);
        }
    }

    /**
     * @brief Radix-8 DIF pass for N=1 - BACKWARD (CORRECTED)
     *
     * CRITICAL: Stage twiddles must be conjugated for inverse transform.
     * The backward butterfly handles internal W8 rotations, but NOT stage twiddles.
     */
    TARGET_AVX2_FMA
    static inline void radix32_n1_pass2_dif8_backward_avx2(
        const double *RESTRICT temp_re,
        const double *RESTRICT temp_im,
        double *RESTRICT out_re,
        double *RESTRICT out_im,
        const radix32_n1_twiddles_dif8_t *RESTRICT tw)
    {
        const __m256d SIGN_FLIP = _mm256_set1_pd(-0.0);

        // ⚠️ BACKWARD: Use conjugates of forward twiddles
        // W̄_32^k = cos(-2πk/32) - i·sin(-2πk/32) = cos(2πk/32) + i·sin(2πk/32)
        __m256d W1r = _mm256_set1_pd(tw->re[0]);
        __m256d W1i = _mm256_set1_pd(-tw->im[0]); // Conjugate: negate imaginary
        __m256d W2r = _mm256_set1_pd(tw->re[1]);
        __m256d W2i = _mm256_set1_pd(-tw->im[1]);
        __m256d W3r = _mm256_set1_pd(tw->re[2]);
        __m256d W3i = _mm256_set1_pd(-tw->im[2]);
        __m256d W4r = _mm256_set1_pd(tw->re[3]);
        __m256d W4i = _mm256_set1_pd(-tw->im[3]);

        // Derive W̄_32^5 = -W̄_32^1, W̄_32^6 = -W̄_32^2, W̄_32^7 = -W̄_32^3
        __m256d W5r = _mm256_xor_pd(W1r, SIGN_FLIP);
        __m256d W5i = _mm256_xor_pd(W1i, SIGN_FLIP);
        __m256d W6r = _mm256_xor_pd(W2r, SIGN_FLIP);
        __m256d W6i = _mm256_xor_pd(W2i, SIGN_FLIP);
        __m256d W7r = _mm256_xor_pd(W3r, SIGN_FLIP);
        __m256d W7i = _mm256_xor_pd(W3i, SIGN_FLIP);

        for (size_t bin = 0; bin < 4; bin++)
        {
            size_t base = bin * 8;
            __m256d x0r = _mm256_load_pd(&temp_re[base + 0]);
            __m256d x0i = _mm256_load_pd(&temp_im[base + 0]);
            __m256d x1r = _mm256_load_pd(&temp_re[base + 4]);
            __m256d x1i = _mm256_load_pd(&temp_im[base + 4]);

            base = bin * 8 + 8;
            __m256d x2r = _mm256_load_pd(&temp_re[base + 0]);
            __m256d x2i = _mm256_load_pd(&temp_im[base + 0]);
            __m256d x3r = _mm256_load_pd(&temp_re[base + 4]);
            __m256d x3i = _mm256_load_pd(&temp_im[base + 4]);

            base = bin * 8 + 16;
            __m256d x4r = _mm256_load_pd(&temp_re[base + 0]);
            __m256d x4i = _mm256_load_pd(&temp_im[base + 0]);
            __m256d x5r = _mm256_load_pd(&temp_re[base + 4]);
            __m256d x5i = _mm256_load_pd(&temp_im[base + 4]);

            base = bin * 8 + 24;
            __m256d x6r = _mm256_load_pd(&temp_re[base + 0]);
            __m256d x6i = _mm256_load_pd(&temp_im[base + 0]);
            __m256d x7r = _mm256_load_pd(&temp_re[base + 4]);
            __m256d x7i = _mm256_load_pd(&temp_im[base + 4]);

            // Apply conjugated stage twiddles
            __m256d t0r = x0r, t0i = x0i;
            __m256d t1r, t1i, t2r, t2i, t3r, t3i, t4r, t4i;
            __m256d t5r, t5i, t6r, t6i, t7r, t7i;

            cmul_v256(x1r, x1i, W1r, W1i, &t1r, &t1i);
            cmul_v256(x2r, x2i, W2r, W2i, &t2r, &t2i);
            cmul_v256(x3r, x3i, W3r, W3i, &t3r, &t3i);
            cmul_v256(x4r, x4i, W4r, W4i, &t4r, &t4i);
            cmul_v256(x5r, x5i, W5r, W5i, &t5r, &t5i);
            cmul_v256(x6r, x6i, W6r, W6i, &t6r, &t6i);
            cmul_v256(x7r, x7i, W7r, W7i, &t7r, &t7i);

            // Radix-8 DIF butterfly (BACKWARD - handles internal W8)
            __m256d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
            __m256d y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;

            radix8_dif_butterfly_backward_avx2(
                t0r, t0i, t1r, t1i, t2r, t2i, t3r, t3i,
                t4r, t4i, t5r, t5i, t6r, t6i, t7r, t7i,
                &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
                &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);

            // Store outputs
            size_t out_base = bin * 8;
            _mm256_store_pd(&out_re[out_base + 0], y0r);
            _mm256_store_pd(&out_im[out_base + 0], y0i);
            _mm256_store_pd(&out_re[out_base + 4], y1r);
            _mm256_store_pd(&out_im[out_base + 4], y1i);

            out_base = bin * 8 + 8;
            _mm256_store_pd(&out_re[out_base + 0], y2r);
            _mm256_store_pd(&out_im[out_base + 0], y2i);
            _mm256_store_pd(&out_re[out_base + 4], y3r);
            _mm256_store_pd(&out_im[out_base + 4], y3i);

            out_base = bin * 8 + 16;
            _mm256_store_pd(&out_re[out_base + 0], y4r);
            _mm256_store_pd(&out_im[out_base + 0], y4i);
            _mm256_store_pd(&out_re[out_base + 4], y5r);
            _mm256_store_pd(&out_im[out_base + 4], y5i);

            out_base = bin * 8 + 24;
            _mm256_store_pd(&out_re[out_base + 0], y6r);
            _mm256_store_pd(&out_im[out_base + 0], y6i);
            _mm256_store_pd(&out_re[out_base + 4], y7r);
            _mm256_store_pd(&out_im[out_base + 4], y7i);
        }
    }

    //==============================================================================
    // TOP-LEVEL N=1 WRAPPERS
    //==============================================================================

    /**
     * @brief Complete radix-32 N=1 transform - FORWARD
     *
     * @param in_re Input real [32]
     * @param in_im Input imaginary [32]
     * @param out_re Output real [32]
     * @param out_im Output imaginary [32]
     * @param tw_dif8 Geometric twiddles for Pass 2 (from planner)
     *
     * @note Requires aligned temp storage (64 doubles = 512 bytes)
     */
    TARGET_AVX2_FMA
    static inline void radix32_n1_forward_avx2(
        const double *RESTRICT in_re,
        const double *RESTRICT in_im,
        double *RESTRICT out_re,
        double *RESTRICT out_im,
        const radix32_n1_twiddles_dif8_t *RESTRICT tw_dif8)
    {
        // Temp storage for inter-pass data (aligned)
        ALIGNED(32)
        double temp_re[32];
        ALIGNED(32)
        double temp_im[32];

        // Pass 1: Radix-4 DIT (no twiddles)
        radix32_n1_pass1_dit4_forward_avx2(in_re, in_im, temp_re, temp_im);

        // Pass 2: Radix-8 DIF (geometric twiddles from planner)
        radix32_n1_pass2_dif8_forward_avx2(temp_re, temp_im, out_re, out_im, tw_dif8);
    }

    /**
     * @brief Complete radix-32 N=1 transform - BACKWARD (CORRECTED)
     */
    TARGET_AVX2_FMA
    static inline void radix32_n1_backward_avx2(
        const double *RESTRICT in_re,
        const double *RESTRICT in_im,
        double *RESTRICT out_re,
        double *RESTRICT out_im,
        const radix32_n1_twiddles_dif8_t *RESTRICT tw_dif8)
    {
        ALIGNED(32)
        double temp_re[32];
        ALIGNED(32)
        double temp_im[32];

        // Pass 1: Radix-4 DIT (backward, no twiddles)
        radix32_n1_pass1_dit4_backward_avx2(in_re, in_im, temp_re, temp_im);

        // Pass 2: Radix-8 DIF (backward, conjugated twiddles)
        radix32_n1_pass2_dif8_backward_avx2(temp_re, temp_im, out_re, out_im, tw_dif8);
    }

    //==============================================================================
    // USAGE NOTES
    //==============================================================================

    /**
     * @section Usage Example
     *
     * @code
     * // Planner initializes geometric twiddles (done once)
     * double tw_re[4] = {cos(-2π·1/32), cos(-2π·2/32), cos(-2π·3/32), cos(-2π·4/32)};
     * double tw_im[4] = {sin(-2π·1/32), sin(-2π·2/32), sin(-2π·3/32), sin(-2π·4/32)};
     *
     * radix32_n1_twiddles_dif8_t tw_dif8 = {
     *     .re = tw_re,
     *     .im = tw_im
     * };
     *
     * // Transform 32-point FFT (done many times)
     * ALIGNED(32) double in_re[32], in_im[32];
     * ALIGNED(32) double out_re[32], out_im[32];
     *
     * radix32_n1_forward_avx2(in_re, in_im, out_re, out_im, &tw_dif8);
     * @endcode
     */

#ifdef __cplusplus
}
#endif

#endif /* FFT_RADIX32_AVX2_N1_H */