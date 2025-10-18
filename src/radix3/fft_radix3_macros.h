//==============================================================================
// fft_radix3_macros.h - Shared Macros for Radix-3 Butterflies
//==============================================================================
//
// ALGORITHM: Radix-3 DFT with geometric decomposition
//   1. Apply input twiddles W_N^(j*k) to lanes 1-2
//   2. Compute sum and difference of twiddled inputs
//   3. Apply geometric rotation (±i * sqrt(3)/2)
//   4. Assemble outputs
//

#ifndef FFT_RADIX3_MACROS_H
#define FFT_RADIX3_MACROS_H

#include "simd_math.h"

//==============================================================================
// GEOMETRIC CONSTANTS - IDENTICAL for forward/inverse
//==============================================================================

#define C_HALF      (-0.5)
#define S_SQRT3_2   0.8660254037844386467618  // sqrt(3)/2

//==============================================================================
// AVX-512 SUPPORT - 4X throughput vs AVX2 (processes 4 butterflies)
//==============================================================================

#ifdef __AVX512F__

//==============================================================================
// COMPLEX MULTIPLICATION - AVX-512
//==============================================================================

/**
 * @brief Optimized complex multiply for AVX-512: out = a * w (4 complex values)
 *
 * Uses FMA and handles 4 complex numbers (8 doubles) per operation.
 */
#define CMUL_FMA_AOS_AVX512(out, a, w)                                    \
    do                                                                    \
    {                                                                     \
        __m512d ar = _mm512_unpacklo_pd(a, a);                            \
        __m512d ai = _mm512_unpackhi_pd(a, a);                            \
        __m512d wr = _mm512_unpacklo_pd(w, w);                            \
        __m512d wi = _mm512_unpackhi_pd(w, w);                            \
        __m512d re = _mm512_fmsub_pd(ar, wr, _mm512_mul_pd(ai, wi));     \
        __m512d im = _mm512_fmadd_pd(ar, wi, _mm512_mul_pd(ai, wr));     \
        (out) = _mm512_unpacklo_pd(re, im);                               \
    } while (0)

//==============================================================================
// RADIX-3 BUTTERFLY CORE - AVX-512
//==============================================================================

/**
 * @brief Compute sum, difference, and common term (AVX-512, 4 butterflies)
 * 
 * sum = tw_b + tw_c
 * dif = tw_b - tw_c
 * common = a + C_HALF * sum
 */
#define RADIX3_BUTTERFLY_CORE_AVX512(a, tw_b, tw_c, sum, dif, common)    \
    do {                                                                  \
        const __m512d v_half = _mm512_set1_pd(C_HALF);                    \
        sum = _mm512_add_pd(tw_b, tw_c);                                  \
        dif = _mm512_sub_pd(tw_b, tw_c);                                  \
        common = _mm512_fmadd_pd(v_half, sum, a);                         \
    } while (0)

//==============================================================================
// ROTATION AND SCALING - AVX-512
//==============================================================================

/**
 * @brief FORWARD rotation: -i * dif * sqrt(3)/2 (AVX-512, 4 butterflies)
 * 
 * (a + bi) * (-i) = b - ai
 * scaled_rot = (b - ai) * sqrt(3)/2
 */
#define RADIX3_ROTATE_FORWARD_AVX512(dif, scaled_rot)                         \
    do {                                                                      \
        const __m512d v_sqrt3_2 = _mm512_set1_pd(S_SQRT3_2);                  \
        const __m512d rot_mask = _mm512_set_pd(0.0, -0.0, 0.0, -0.0,         \
                                                0.0, -0.0, 0.0, -0.0);        \
        __m512d dif_swp = _mm512_permute_pd(dif, 0b01010101);                 \
        __m512d rot90 = _mm512_xor_pd(dif_swp, rot_mask);                     \
        scaled_rot = _mm512_mul_pd(rot90, v_sqrt3_2);                         \
    } while (0)

/**
 * @brief INVERSE rotation: +i * dif * sqrt(3)/2 (AVX-512, 4 butterflies)
 * 
 * (a + bi) * (+i) = -b + ai
 * scaled_rot = (-b + ai) * sqrt(3)/2
 */
#define RADIX3_ROTATE_INVERSE_AVX512(dif, scaled_rot)                         \
    do {                                                                      \
        const __m512d v_sqrt3_2 = _mm512_set1_pd(S_SQRT3_2);                  \
        const __m512d rot_mask = _mm512_set_pd(-0.0, 0.0, -0.0, 0.0,         \
                                                -0.0, 0.0, -0.0, 0.0);        \
        __m512d dif_swp = _mm512_permute_pd(dif, 0b01010101);                 \
        __m512d rot90 = _mm512_xor_pd(dif_swp, rot_mask);                     \
        scaled_rot = _mm512_mul_pd(rot90, v_sqrt3_2);                         \
    } while (0)

//==============================================================================
// OUTPUT ASSEMBLY - AVX-512
//==============================================================================

/**
 * @brief Assemble final outputs (AVX-512, 4 butterflies)
 * 
 * y0 = a + sum
 * y1 = common + scaled_rot
 * y2 = common - scaled_rot
 */
#define RADIX3_ASSEMBLE_OUTPUTS_AVX512(a, sum, common, scaled_rot, y0, y1, y2) \
    do {                                                                        \
        y0 = _mm512_add_pd(a, sum);                                             \
        y1 = _mm512_add_pd(common, scaled_rot);                                 \
        y2 = _mm512_sub_pd(common, scaled_rot);                                 \
    } while (0)

//==============================================================================
// APPLY PRECOMPUTED TWIDDLES - AVX-512
//==============================================================================

/**
 * @brief AVX-512: Apply stage twiddles for 4 butterflies (kk through kk+3)
 *
 * stage_tw layout: [W^(1*k), W^(2*k)] for each k
 * Loads twiddles for 4 butterflies simultaneously.
 */
#define APPLY_STAGE_TWIDDLES_AVX512(kk, b, c, stage_tw, tw_b, tw_c)      \
    do {                                                                  \
        __m512d w1 = load4_aos(&stage_tw[(kk)*2 + 0],                    \
                               &stage_tw[(kk+1)*2 + 0],                   \
                               &stage_tw[(kk+2)*2 + 0],                   \
                               &stage_tw[(kk+3)*2 + 0]);                  \
        __m512d w2 = load4_aos(&stage_tw[(kk)*2 + 1],                    \
                               &stage_tw[(kk+1)*2 + 1],                   \
                               &stage_tw[(kk+2)*2 + 1],                   \
                               &stage_tw[(kk+3)*2 + 1]);                  \
                                                                          \
        CMUL_FMA_AOS_AVX512(tw_b, b, w1);                                 \
        CMUL_FMA_AOS_AVX512(tw_c, c, w2);                                 \
    } while (0)

//==============================================================================
// DATA MOVEMENT - AVX-512
//==============================================================================

/**
 * @brief Load 3 lanes for 4 butterflies (kk through kk+3)
 *
 * Loads input data for 3 lanes (0 to 2) from sub_outputs buffer.
 * Each register holds 4 complex values (for 4 butterflies) in AoS layout.
 */
#define LOAD_3_LANES_AVX512(kk, K, sub_outputs, a, b, c)                 \
    do {                                                                  \
        a = load4_aos(&sub_outputs[kk],                                   \
                      &sub_outputs[(kk)+1],                               \
                      &sub_outputs[(kk)+2],                               \
                      &sub_outputs[(kk)+3]);                              \
        b = load4_aos(&sub_outputs[(kk)+K],                               \
                      &sub_outputs[(kk)+1+K],                             \
                      &sub_outputs[(kk)+2+K],                             \
                      &sub_outputs[(kk)+3+K]);                            \
        c = load4_aos(&sub_outputs[(kk)+2*K],                             \
                      &sub_outputs[(kk)+1+2*K],                           \
                      &sub_outputs[(kk)+2+2*K],                           \
                      &sub_outputs[(kk)+3+2*K]);                          \
    } while (0)

/**
 * @brief Store 3 outputs for 4 butterflies (AVX-512)
 *
 * Stores three 512-bit vectors (each with 4 complex values) to output_buffer.
 * Uses unaligned stores for flexibility.
 */
#define STORE_3_LANES_AVX512(kk, K, output_buffer, y0, y1, y2)           \
    do {                                                                  \
        STOREU_PD512(&output_buffer[kk].re, y0);                          \
        STOREU_PD512(&output_buffer[(kk)+K].re, y1);                      \
        STOREU_PD512(&output_buffer[(kk)+2*K].re, y2);                    \
    } while (0)

/**
 * @brief Store with non-temporal hint (streaming) - AVX-512
 *
 * Uses streaming stores to bypass cache for large datasets.
 */
#define STORE_3_LANES_AVX512_STREAM(kk, K, output_buffer, y0, y1, y2)    \
    do {                                                                  \
        _mm512_stream_pd(&output_buffer[kk].re, y0);                      \
        _mm512_stream_pd(&output_buffer[(kk)+K].re, y1);                  \
        _mm512_stream_pd(&output_buffer[(kk)+2*K].re, y2);                \
    } while (0)

//==============================================================================
// PREFETCHING - AVX-512
//==============================================================================

/**
 * @brief Prefetch distances optimized for AVX-512 (larger working set)
 */
#define PREFETCH_L1_AVX512 16  // 1KB ahead
#define PREFETCH_L2_AVX512 64  // 4KB ahead
#define PREFETCH_L3_AVX512 128 // 8KB ahead

/**
 * @brief Prefetch 3 lanes ahead for AVX-512 (4 butterflies)
 *
 * Prefetches more aggressively than AVX2 due to higher throughput.
 */
#define PREFETCH_3_LANES_AVX512(k, K, distance, sub_outputs, stage_tw, hint)     \
    do {                                                                         \
        if ((k) + (distance) < K) {                                              \
            _mm_prefetch((const char *)&sub_outputs[(k)+(distance)], hint);      \
            _mm_prefetch((const char *)&sub_outputs[(k)+(distance)+K], hint);    \
            _mm_prefetch((const char *)&sub_outputs[(k)+(distance)+2*K], hint);  \
            _mm_prefetch((const char *)&stage_tw[((k)+(distance))*2], hint);     \
        }                                                                        \
    } while (0)

//==============================================================================
// COMPLETE BUTTERFLY PIPELINE - AVX-512
//==============================================================================

/**
 * @brief Complete AVX-512 radix-3 butterfly (FORWARD, 4 butterflies)
 *
 * Processes 4 butterflies (12 complex values) in one macro call.
 * 
 * Algorithm:
 * 1. Load 3 lanes for 4 butterflies (12 complex values)
 * 2. Apply input twiddles to lanes 1-2
 * 3. Compute butterfly core (sum, diff, common)
 * 4. Apply forward rotation (-i * sqrt(3)/2)
 * 5. Assemble outputs
 * 6. Store 12 outputs
 */
#define RADIX3_PIPELINE_4_FV_AVX512(kk, K, sub_outputs, stage_tw, output_buffer) \
    do {                                                                         \
        /* Step 1: Load 3 lanes for 4 butterflies */                            \
        __m512d a, b, c;                                                         \
        LOAD_3_LANES_AVX512(kk, K, sub_outputs, a, b, c);                        \
                                                                                 \
        /* Step 2: Apply precomputed stage twiddles */                          \
        __m512d tw_b, tw_c;                                                      \
        APPLY_STAGE_TWIDDLES_AVX512(kk, b, c, stage_tw, tw_b, tw_c);             \
                                                                                 \
        /* Step 3: Compute butterfly core */                                    \
        __m512d sum, dif, common;                                                \
        RADIX3_BUTTERFLY_CORE_AVX512(a, tw_b, tw_c, sum, dif, common);           \
                                                                                 \
        /* Step 4: Apply forward rotation */                                    \
        __m512d scaled_rot;                                                      \
        RADIX3_ROTATE_FORWARD_AVX512(dif, scaled_rot);                           \
                                                                                 \
        /* Step 5: Assemble outputs */                                          \
        __m512d y0, y1, y2;                                                      \
        RADIX3_ASSEMBLE_OUTPUTS_AVX512(a, sum, common, scaled_rot, y0, y1, y2);  \
                                                                                 \
        /* Step 6: Store results */                                             \
        STORE_3_LANES_AVX512(kk, K, output_buffer, y0, y1, y2);                  \
    } while (0)

/**
 * @brief Complete AVX-512 radix-3 butterfly (INVERSE, 4 butterflies)
 *
 * Identical structure to forward but uses inverse rotation.
 */
#define RADIX3_PIPELINE_4_BV_AVX512(kk, K, sub_outputs, stage_tw, output_buffer) \
    do {                                                                         \
        /* Step 1: Load 3 lanes for 4 butterflies */                            \
        __m512d a, b, c;                                                         \
        LOAD_3_LANES_AVX512(kk, K, sub_outputs, a, b, c);                        \
                                                                                 \
        /* Step 2: Apply precomputed stage twiddles */                          \
        __m512d tw_b, tw_c;                                                      \
        APPLY_STAGE_TWIDDLES_AVX512(kk, b, c, stage_tw, tw_b, tw_c);             \
                                                                                 \
        /* Step 3: Compute butterfly core */                                    \
        __m512d sum, dif, common;                                                \
        RADIX3_BUTTERFLY_CORE_AVX512(a, tw_b, tw_c, sum, dif, common);           \
                                                                                 \
        /* Step 4: Apply INVERSE rotation */                                    \
        __m512d scaled_rot;                                                      \
        RADIX3_ROTATE_INVERSE_AVX512(dif, scaled_rot);                           \
                                                                                 \
        /* Step 5: Assemble outputs */                                          \
        __m512d y0, y1, y2;                                                      \
        RADIX3_ASSEMBLE_OUTPUTS_AVX512(a, sum, common, scaled_rot, y0, y1, y2);  \
                                                                                 \
        /* Step 6: Store results */                                             \
        STORE_3_LANES_AVX512(kk, K, output_buffer, y0, y1, y2);                  \
    } while (0)

//==============================================================================
// STREAMING VERSIONS (for very large transforms)
//==============================================================================

/**
 * @brief Streaming version (FORWARD) - uses non-temporal stores
 *
 * Use for transforms larger than L3 cache.
 */
#define RADIX3_PIPELINE_4_FV_AVX512_STREAM(kk, K, sub_outputs, stage_tw, output_buffer) \
    do {                                                                                \
        __m512d a, b, c;                                                                \
        LOAD_3_LANES_AVX512(kk, K, sub_outputs, a, b, c);                               \
                                                                                        \
        __m512d tw_b, tw_c;                                                             \
        APPLY_STAGE_TWIDDLES_AVX512(kk, b, c, stage_tw, tw_b, tw_c);                    \
                                                                                        \
        __m512d sum, dif, common;                                                       \
        RADIX3_BUTTERFLY_CORE_AVX512(a, tw_b, tw_c, sum, dif, common);                  \
                                                                                        \
        __m512d scaled_rot;                                                             \
        RADIX3_ROTATE_FORWARD_AVX512(dif, scaled_rot);                                  \
                                                                                        \
        __m512d y0, y1, y2;                                                             \
        RADIX3_ASSEMBLE_OUTPUTS_AVX512(a, sum, common, scaled_rot, y0, y1, y2);         \
                                                                                        \
        STORE_3_LANES_AVX512_STREAM(kk, K, output_buffer, y0, y1, y2);                  \
    } while (0)

/**
 * @brief Streaming version (INVERSE)
 */
#define RADIX3_PIPELINE_4_BV_AVX512_STREAM(kk, K, sub_outputs, stage_tw, output_buffer) \
    do {                                                                                \
        __m512d a, b, c;                                                                \
        LOAD_3_LANES_AVX512(kk, K, sub_outputs, a, b, c);                               \
                                                                                        \
        __m512d tw_b, tw_c;                                                             \
        APPLY_STAGE_TWIDDLES_AVX512(kk, b, c, stage_tw, tw_b, tw_c);                    \
                                                                                        \
        __m512d sum, dif, common;                                                       \
        RADIX3_BUTTERFLY_CORE_AVX512(a, tw_b, tw_c, sum, dif, common);                  \
                                                                                        \
        __m512d scaled_rot;                                                             \
        RADIX3_ROTATE_INVERSE_AVX512(dif, scaled_rot);                                  \
                                                                                        \
        __m512d y0, y1, y2;                                                             \
        RADIX3_ASSEMBLE_OUTPUTS_AVX512(a, sum, common, scaled_rot, y0, y1, y2);         \
                                                                                        \
        STORE_3_LANES_AVX512_STREAM(kk, K, output_buffer, y0, y1, y2);                  \
    } while (0)

#endif // __AVX512F__

//==============================================================================
// COMPLEX MULTIPLICATION - FMA-optimized (IDENTICAL)
//==============================================================================

#ifdef __AVX2__
#define CMUL_FMA_AOS(out, a, w)                                      \
    do                                                               \
    {                                                                \
        __m256d ar = _mm256_unpacklo_pd(a, a);                       \
        __m256d ai = _mm256_unpackhi_pd(a, a);                       \
        __m256d wr = _mm256_unpacklo_pd(w, w);                       \
        __m256d wi = _mm256_unpackhi_pd(w, w);                       \
        __m256d re = _mm256_fmsub_pd(ar, wr, _mm256_mul_pd(ai, wi)); \
        __m256d im = _mm256_fmadd_pd(ar, wi, _mm256_mul_pd(ai, wr)); \
        (out) = _mm256_unpacklo_pd(re, im);                          \
    } while (0)
#endif

//==============================================================================
// RADIX-3 BUTTERFLY CORE - Direction-agnostic (IDENTICAL)
//==============================================================================

/**
 * @brief Compute sum, difference, and common term (IDENTICAL for both directions)
 * 
 * sum = tw_b + tw_c
 * dif = tw_b - tw_c
 * common = a + C_HALF * sum
 */
#ifdef __AVX2__
#define RADIX3_BUTTERFLY_CORE_AVX2(a, tw_b, tw_c, sum, dif, common) \
    do { \
        const __m256d v_half = _mm256_set1_pd(C_HALF); \
        sum = _mm256_add_pd(tw_b, tw_c); \
        dif = _mm256_sub_pd(tw_b, tw_c); \
        common = _mm256_fmadd_pd(v_half, sum, a); \
    } while (0)
#endif

// Scalar version
#define RADIX3_BUTTERFLY_CORE_SCALAR(a, tw_b, tw_c, \
                                      sum_re, sum_im, dif_re, dif_im, \
                                      common_re, common_im) \
    do { \
        sum_re = tw_b.re + tw_c.re; \
        sum_im = tw_b.im + tw_c.im; \
        dif_re = tw_b.re - tw_c.re; \
        dif_im = tw_b.im - tw_c.im; \
        common_re = a.re + C_HALF * sum_re; \
        common_im = a.im + C_HALF * sum_im; \
    } while (0)

//==============================================================================
// ROTATION AND SCALING - ONLY DIFFERENCE between forward/inverse
//==============================================================================

/**
 * @brief FORWARD rotation: -i * dif * sqrt(3)/2
 * 
 * (a + bi) * (-i) = b - ai
 * scaled_rot = (b - ai) * sqrt(3)/2
 */
#ifdef __AVX2__
#define RADIX3_ROTATE_FORWARD_AVX2(dif, scaled_rot, v_sqrt3_2) \
    do { \
        const __m256d rot_mask = _mm256_set_pd(0.0, -0.0, 0.0, -0.0); \
        __m256d dif_swp = _mm256_permute_pd(dif, 0b0101); \
        __m256d rot90 = _mm256_xor_pd(dif_swp, rot_mask); \
        scaled_rot = _mm256_mul_pd(rot90, v_sqrt3_2); \
    } while (0)
#endif

#define RADIX3_ROTATE_FORWARD_SCALAR(dif_re, dif_im, scaled_rot_re, scaled_rot_im) \
    do { \
        scaled_rot_re = S_SQRT3_2 * dif_im; \
        scaled_rot_im = -S_SQRT3_2 * dif_re; \
    } while (0)

/**
 * @brief INVERSE rotation: +i * dif * sqrt(3)/2
 * 
 * (a + bi) * (+i) = -b + ai
 * scaled_rot = (-b + ai) * sqrt(3)/2
 */
#ifdef __AVX2__
#define RADIX3_ROTATE_INVERSE_AVX2(dif, scaled_rot, v_sqrt3_2) \
    do { \
        const __m256d rot_mask = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0); \
        __m256d dif_swp = _mm256_permute_pd(dif, 0b0101); \
        __m256d rot90 = _mm256_xor_pd(dif_swp, rot_mask); \
        scaled_rot = _mm256_mul_pd(rot90, v_sqrt3_2); \
    } while (0)
#endif

#define RADIX3_ROTATE_INVERSE_SCALAR(dif_re, dif_im, scaled_rot_re, scaled_rot_im) \
    do { \
        scaled_rot_re = -S_SQRT3_2 * dif_im; \
        scaled_rot_im = S_SQRT3_2 * dif_re; \
    } while (0)

//==============================================================================
// OUTPUT ASSEMBLY - IDENTICAL for forward/inverse
//==============================================================================

/**
 * @brief Assemble final outputs (IDENTICAL)
 * 
 * y0 = a + sum
 * y1 = common + scaled_rot
 * y2 = common - scaled_rot
 */
#ifdef __AVX2__
#define RADIX3_ASSEMBLE_OUTPUTS_AVX2(a, sum, common, scaled_rot, y0, y1, y2) \
    do { \
        y0 = _mm256_add_pd(a, sum); \
        y1 = _mm256_add_pd(common, scaled_rot); \
        y2 = _mm256_sub_pd(common, scaled_rot); \
    } while (0)
#endif

#define RADIX3_ASSEMBLE_OUTPUTS_SCALAR(a, sum_re, sum_im, \
                                        common_re, common_im, \
                                        scaled_rot_re, scaled_rot_im, \
                                        y0, y1, y2) \
    do { \
        y0.re = a.re + sum_re; \
        y0.im = a.im + sum_im; \
        y1.re = common_re + scaled_rot_re; \
        y1.im = common_im + scaled_rot_im; \
        y2.re = common_re - scaled_rot_re; \
        y2.im = common_im - scaled_rot_im; \
    } while (0)

//==============================================================================
// APPLY PRECOMPUTED TWIDDLES - IDENTICAL for forward/inverse
//==============================================================================

/**
 * @brief Scalar: Apply stage twiddles to lanes 1-2
 * 
 * stage_tw layout: [W^(1*k), W^(2*k)] for each k
 */
#define APPLY_STAGE_TWIDDLES_SCALAR(k, b, c, stage_tw, tw_b, tw_c) \
    do { \
        const fft_data *w_ptr = &stage_tw[(k)*2]; \
        \
        tw_b.re = b.re * w_ptr[0].re - b.im * w_ptr[0].im; \
        tw_b.im = b.re * w_ptr[0].im + b.im * w_ptr[0].re; \
        \
        tw_c.re = c.re * w_ptr[1].re - c.im * w_ptr[1].im; \
        tw_c.im = c.re * w_ptr[1].im + c.im * w_ptr[1].re; \
    } while (0)

/**
 * @brief AVX2: Apply stage twiddles for 2 butterflies (kk and kk+1)
 */
#ifdef __AVX2__
#define APPLY_STAGE_TWIDDLES_AVX2(kk, b, c, stage_tw, tw_b, tw_c) \
    do { \
        __m256d w1 = load2_aos(&stage_tw[(kk)*2 + 0], &stage_tw[(kk+1)*2 + 0]); \
        __m256d w2 = load2_aos(&stage_tw[(kk)*2 + 1], &stage_tw[(kk+1)*2 + 1]); \
        \
        CMUL_FMA_AOS(tw_b, b, w1); \
        CMUL_FMA_AOS(tw_c, c, w2); \
    } while (0)
#endif

//==============================================================================
// DATA MOVEMENT - IDENTICAL for forward/inverse
//==============================================================================

#ifdef __AVX2__
/**
 * @brief Load 3 lanes for 2 butterflies (kk and kk+1)
 */
#define LOAD_3_LANES_AVX2(kk, K, sub_outputs, a, b, c) \
    do { \
        a = load2_aos(&sub_outputs[kk], &sub_outputs[(kk)+1]); \
        b = load2_aos(&sub_outputs[(kk)+K], &sub_outputs[(kk)+1+K]); \
        c = load2_aos(&sub_outputs[(kk)+2*K], &sub_outputs[(kk)+1+2*K]); \
    } while (0)

/**
 * @brief Store 3 outputs for 2 butterflies
 */
#define STORE_3_LANES_AVX2(kk, K, output_buffer, y0, y1, y2) \
    do { \
        STOREU_PD(&output_buffer[kk].re, y0); \
        STOREU_PD(&output_buffer[(kk)+K].re, y1); \
        STOREU_PD(&output_buffer[(kk)+2*K].re, y2); \
    } while (0)

/**
 * @brief Store with non-temporal hint (streaming)
 */
#define STORE_3_LANES_AVX2_STREAM(kk, K, output_buffer, y0, y1, y2) \
    do { \
        _mm256_stream_pd(&output_buffer[kk].re, y0); \
        _mm256_stream_pd(&output_buffer[(kk)+K].re, y1); \
        _mm256_stream_pd(&output_buffer[(kk)+2*K].re, y2); \
    } while (0)
#endif

//==============================================================================
// PREFETCHING - IDENTICAL for forward/inverse
//==============================================================================

#define PREFETCH_L1 8
#define PREFETCH_L2 32
#define PREFETCH_L3 64

#ifdef __AVX2__
#define PREFETCH_3_LANES(k, K, distance, sub_outputs, hint) \
    do { \
        if ((k) + (distance) < K) { \
            _mm_prefetch((const char *)&sub_outputs[(k)+(distance)], hint); \
            _mm_prefetch((const char *)&sub_outputs[(k)+(distance)+K], hint); \
            _mm_prefetch((const char *)&sub_outputs[(k)+(distance)+2*K], hint); \
        } \
    } while (0)
#endif

//==============================================================================
// COMPLETE BUTTERFLY - Unified macro for each direction
//==============================================================================

/**
 * @brief Complete scalar radix-3 butterfly (forward version)
 */
#define RADIX3_BUTTERFLY_SCALAR_FV(k, K, sub_outputs, stage_tw, output_buffer) \
    do { \
        fft_data a = sub_outputs[k]; \
        fft_data b = sub_outputs[k + K]; \
        fft_data c = sub_outputs[k + 2*K]; \
        \
        fft_data tw_b, tw_c; \
        APPLY_STAGE_TWIDDLES_SCALAR(k, b, c, stage_tw, tw_b, tw_c); \
        \
        double sum_re, sum_im, dif_re, dif_im, common_re, common_im; \
        RADIX3_BUTTERFLY_CORE_SCALAR(a, tw_b, tw_c, \
                                      sum_re, sum_im, dif_re, dif_im, \
                                      common_re, common_im); \
        \
        double scaled_rot_re, scaled_rot_im; \
        RADIX3_ROTATE_FORWARD_SCALAR(dif_re, dif_im, scaled_rot_re, scaled_rot_im); \
        \
        fft_data y0, y1, y2; \
        RADIX3_ASSEMBLE_OUTPUTS_SCALAR(a, sum_re, sum_im, \
                                        common_re, common_im, \
                                        scaled_rot_re, scaled_rot_im, \
                                        y0, y1, y2); \
        \
        output_buffer[k] = y0; \
        output_buffer[k + K] = y1; \
        output_buffer[k + 2*K] = y2; \
    } while (0)

/**
 * @brief Complete scalar radix-3 butterfly (inverse version)
 */
#define RADIX3_BUTTERFLY_SCALAR_BV(k, K, sub_outputs, stage_tw, output_buffer) \
    do { \
        fft_data a = sub_outputs[k]; \
        fft_data b = sub_outputs[k + K]; \
        fft_data c = sub_outputs[k + 2*K]; \
        \
        fft_data tw_b, tw_c; \
        APPLY_STAGE_TWIDDLES_SCALAR(k, b, c, stage_tw, tw_b, tw_c); \
        \
        double sum_re, sum_im, dif_re, dif_im, common_re, common_im; \
        RADIX3_BUTTERFLY_CORE_SCALAR(a, tw_b, tw_c, \
                                      sum_re, sum_im, dif_re, dif_im, \
                                      common_re, common_im); \
        \
        double scaled_rot_re, scaled_rot_im; \
        RADIX3_ROTATE_INVERSE_SCALAR(dif_re, dif_im, scaled_rot_re, scaled_rot_im); \
        \
        fft_data y0, y1, y2; \
        RADIX3_ASSEMBLE_OUTPUTS_SCALAR(a, sum_re, sum_im, \
                                        common_re, common_im, \
                                        scaled_rot_re, scaled_rot_im, \
                                        y0, y1, y2); \
        \
        output_buffer[k] = y0; \
        output_buffer[k + K] = y1; \
        output_buffer[k + 2*K] = y2; \
    } while (0)

#endif // FFT_RADIX3_MACROS_H