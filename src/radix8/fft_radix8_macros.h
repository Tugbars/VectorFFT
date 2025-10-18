//==============================================================================
// fft_radix8_macros.h - Shared Macros for Radix-8 Butterflies
//==============================================================================
//
// ALGORITHM: Split-radix 2×(4,4) decomposition
//   1. Apply input twiddles W_N^(j*k) to lanes 1-7
//   2. Two parallel radix-4 butterflies (even [0,2,4,6], odd [1,3,5,7])
//   3. Apply W_8 geometric twiddles to odd outputs
//   4. Final radix-2 combination
//
// USAGE:
//   #include "fft_radix8_macros.h" in both fft_radix8_fv.c and fft_radix8_bv.c
//

#ifndef FFT_RADIX8_MACROS_H
#define FFT_RADIX8_MACROS_H

#include "simd_math.h"

//==============================================================================
// W_8 CONSTANTS - Direction-dependent
//==============================================================================

// High-precision sqrt(2)/2
#define C8_CONSTANT 0.7071067811865475244008443621048490392848359376887

/**
 * @brief W_8 twiddle constants for FORWARD FFT
 * 
 * W_8 = exp(-2πi/8) = exp(-πi/4)
 * W_8^1 = (√2/2, -√2/2)
 * W_8^2 = (0, -1)
 * W_8^3 = (-√2/2, -√2/2)
 */
#define W8_FV_1_RE  C8_CONSTANT
#define W8_FV_1_IM  (-C8_CONSTANT)
#define W8_FV_2_RE  0.0
#define W8_FV_2_IM  (-1.0)
#define W8_FV_3_RE  (-C8_CONSTANT)
#define W8_FV_3_IM  (-C8_CONSTANT)

/**
 * @brief W_8 twiddle constants for INVERSE FFT
 * 
 * W_8 = exp(+2πi/8) = exp(+πi/4)
 * W_8^1 = (√2/2, +√2/2)
 * W_8^2 = (0, +1)
 * W_8^3 = (-√2/2, +√2/2)
 */
#define W8_BV_1_RE  C8_CONSTANT
#define W8_BV_1_IM  C8_CONSTANT
#define W8_BV_2_RE  0.0
#define W8_BV_2_IM  1.0
#define W8_BV_3_RE  (-C8_CONSTANT)
#define W8_BV_3_IM  C8_CONSTANT

//==============================================================================
// COMPLEX MULTIPLICATION - FMA-optimized (IDENTICAL for both directions)
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
// RADIX-4 SUB-BUTTERFLY - Used for even/odd splits
//==============================================================================

/**
 * @brief Radix-4 butterfly core (IDENTICAL for forward/inverse)
 */
#ifdef __AVX2__
#define RADIX4_CORE_AVX2(a, b, c, d, y0, y1, y2, y3, rot_mask) \
    do { \
        __m256d sum_bd = _mm256_add_pd(b, d); \
        __m256d dif_bd = _mm256_sub_pd(b, d); \
        __m256d sum_ac = _mm256_add_pd(a, c); \
        __m256d dif_ac = _mm256_sub_pd(a, c); \
        \
        y0 = _mm256_add_pd(sum_ac, sum_bd); \
        y2 = _mm256_sub_pd(sum_ac, sum_bd); \
        \
        __m256d dif_bd_swp = _mm256_permute_pd(dif_bd, 0b0101); \
        __m256d dif_bd_rot = _mm256_xor_pd(dif_bd_swp, rot_mask); \
        \
        y1 = _mm256_sub_pd(dif_ac, dif_bd_rot); \
        y3 = _mm256_add_pd(dif_ac, dif_bd_rot); \
    } while (0)
#endif

// Scalar version
#define RADIX4_CORE_SCALAR(a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, \
                           y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, \
                           rot_sign) \
    do { \
        double sum_bd_re = b_re + d_re; \
        double sum_bd_im = b_im + d_im; \
        double dif_bd_re = b_re - d_re; \
        double dif_bd_im = b_im - d_im; \
        double sum_ac_re = a_re + c_re; \
        double sum_ac_im = a_im + c_im; \
        double dif_ac_re = a_re - c_re; \
        double dif_ac_im = a_im - c_im; \
        \
        y0_re = sum_ac_re + sum_bd_re; \
        y0_im = sum_ac_im + sum_bd_im; \
        y2_re = sum_ac_re - sum_bd_re; \
        y2_im = sum_ac_im - sum_bd_im; \
        \
        double rot_re = (rot_sign) * dif_bd_im; \
        double rot_im = (rot_sign) * (-dif_bd_re); \
        \
        y1_re = dif_ac_re - rot_re; \
        y1_im = dif_ac_im - rot_im; \
        y3_re = dif_ac_re + rot_re; \
        y3_im = dif_ac_im + rot_im; \
    } while (0)

//==============================================================================
// W_8 TWIDDLE APPLICATION - Direction-specific
//==============================================================================

/**
 * @brief Apply W_8 twiddles for FORWARD FFT (AVX2)
 * 
 * o[1] *= W_8^1 = (√2/2, -√2/2)
 * o[2] *= W_8^2 = (0, -1)
 * o[3] *= W_8^3 = (-√2/2, -√2/2)
 */
#ifdef __AVX2__
#define APPLY_W8_TWIDDLES_FV_AVX2(o) \
    do { \
        const __m256d vw81_re = _mm256_set1_pd(W8_FV_1_RE); \
        const __m256d vw81_im = _mm256_set1_pd(W8_FV_1_IM); \
        const __m256d vw83_re = _mm256_set1_pd(W8_FV_3_RE); \
        const __m256d vw83_im = _mm256_set1_pd(W8_FV_3_IM); \
        \
        /* o[1] *= W_8^1 */ \
        { \
            __m256d o1_re = _mm256_movedup_pd(o[1]); \
            __m256d o1_im = _mm256_permute_pd(o[1], 0xF); \
            __m256d new_re = _mm256_fmsub_pd(o1_re, vw81_re, _mm256_mul_pd(o1_im, vw81_im)); \
            __m256d new_im = _mm256_fmadd_pd(o1_re, vw81_im, _mm256_mul_pd(o1_im, vw81_re)); \
            o[1] = _mm256_unpacklo_pd(new_re, new_im); \
        } \
        \
        /* o[2] *= W_8^2 = (0, -1) - optimized: swap and negate */ \
        { \
            __m256d o2_rotated = _mm256_permute_pd(o[2], 0x5); \
            const __m256d rot90_mask = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0); \
            o[2] = _mm256_xor_pd(o2_rotated, rot90_mask); \
        } \
        \
        /* o[3] *= W_8^3 */ \
        { \
            __m256d o3_re = _mm256_movedup_pd(o[3]); \
            __m256d o3_im = _mm256_permute_pd(o[3], 0xF); \
            __m256d new_re = _mm256_fmsub_pd(o3_re, vw83_re, _mm256_mul_pd(o3_im, vw83_im)); \
            __m256d new_im = _mm256_fmadd_pd(o3_re, vw83_im, _mm256_mul_pd(o3_im, vw83_re)); \
            o[3] = _mm256_unpacklo_pd(new_re, new_im); \
        } \
    } while (0)
#endif

/**
 * @brief Apply W_8 twiddles for INVERSE FFT (AVX2)
 * 
 * o[1] *= W_8^1 = (√2/2, +√2/2)
 * o[2] *= W_8^2 = (0, +1)
 * o[3] *= W_8^3 = (-√2/2, +√2/2)
 */
#ifdef __AVX2__
#define APPLY_W8_TWIDDLES_BV_AVX2(o) \
    do { \
        const __m256d vw81_re = _mm256_set1_pd(W8_BV_1_RE); \
        const __m256d vw81_im = _mm256_set1_pd(W8_BV_1_IM); \
        const __m256d vw83_re = _mm256_set1_pd(W8_BV_3_RE); \
        const __m256d vw83_im = _mm256_set1_pd(W8_BV_3_IM); \
        \
        /* o[1] *= W_8^1 */ \
        { \
            __m256d o1_re = _mm256_movedup_pd(o[1]); \
            __m256d o1_im = _mm256_permute_pd(o[1], 0xF); \
            __m256d new_re = _mm256_fmsub_pd(o1_re, vw81_re, _mm256_mul_pd(o1_im, vw81_im)); \
            __m256d new_im = _mm256_fmadd_pd(o1_re, vw81_im, _mm256_mul_pd(o1_im, vw81_re)); \
            o[1] = _mm256_unpacklo_pd(new_re, new_im); \
        } \
        \
        /* o[2] *= W_8^2 = (0, +1) - optimized: swap and negate */ \
        { \
            __m256d o2_rotated = _mm256_permute_pd(o[2], 0x5); \
            const __m256d rot90_mask = _mm256_set_pd(0.0, -0.0, 0.0, -0.0); \
            o[2] = _mm256_xor_pd(o2_rotated, rot90_mask); \
        } \
        \
        /* o[3] *= W_8^3 */ \
        { \
            __m256d o3_re = _mm256_movedup_pd(o[3]); \
            __m256d o3_im = _mm256_permute_pd(o[3], 0xF); \
            __m256d new_re = _mm256_fmsub_pd(o3_re, vw83_re, _mm256_mul_pd(o3_im, vw83_im)); \
            __m256d new_im = _mm256_fmadd_pd(o3_re, vw83_im, _mm256_mul_pd(o3_im, vw83_re)); \
            o[3] = _mm256_unpacklo_pd(new_re, new_im); \
        } \
    } while (0)
#endif

// Scalar versions
#define APPLY_W8_TWIDDLES_FV_SCALAR(o) \
    do { \
        /* o[1] *= W_8^1 */ \
        { \
            double r = o[1].re, i = o[1].im; \
            o[1].re = r * W8_FV_1_RE - i * W8_FV_1_IM; \
            o[1].im = r * W8_FV_1_IM + i * W8_FV_1_RE; \
        } \
        /* o[2] *= W_8^2 = (0, -1) */ \
        { \
            double r = o[2].re, i = o[2].im; \
            o[2].re = -i * W8_FV_2_IM; \
            o[2].im = r * W8_FV_2_IM; \
        } \
        /* o[3] *= W_8^3 */ \
        { \
            double r = o[3].re, i = o[3].im; \
            o[3].re = r * W8_FV_3_RE - i * W8_FV_3_IM; \
            o[3].im = r * W8_FV_3_IM + i * W8_FV_3_RE; \
        } \
    } while (0)

#define APPLY_W8_TWIDDLES_BV_SCALAR(o) \
    do { \
        /* o[1] *= W_8^1 */ \
        { \
            double r = o[1].re, i = o[1].im; \
            o[1].re = r * W8_BV_1_RE - i * W8_BV_1_IM; \
            o[1].im = r * W8_BV_1_IM + i * W8_BV_1_RE; \
        } \
        /* o[2] *= W_8^2 = (0, +1) */ \
        { \
            double r = o[2].re, i = o[2].im; \
            o[2].re = -i * W8_BV_2_IM; \
            o[2].im = r * W8_BV_2_IM; \
        } \
        /* o[3] *= W_8^3 */ \
        { \
            double r = o[3].re, i = o[3].im; \
            o[3].re = r * W8_BV_3_RE - i * W8_BV_3_IM; \
            o[3].im = r * W8_BV_3_IM + i * W8_BV_3_RE; \
        } \
    } while (0)

//==============================================================================
// FINAL RADIX-2 COMBINATION - IDENTICAL for forward/inverse
//==============================================================================

#ifdef __AVX2__
#define FINAL_RADIX2_AVX2(e, o, output_buffer, k, K) \
    do { \
        for (int m = 0; m < 4; m++) { \
            __m256d sum = _mm256_add_pd(e[m], o[m]); \
            __m256d dif = _mm256_sub_pd(e[m], o[m]); \
            STOREU_PD(&output_buffer[k + m * K].re, sum); \
            STOREU_PD(&output_buffer[k + (m + 4) * K].re, dif); \
        } \
    } while (0)

#define FINAL_RADIX2_AVX2_STREAM(e, o, output_buffer, k, K) \
    do { \
        for (int m = 0; m < 4; m++) { \
            __m256d sum = _mm256_add_pd(e[m], o[m]); \
            __m256d dif = _mm256_sub_pd(e[m], o[m]); \
            _mm256_stream_pd(&output_buffer[k + m * K].re, sum); \
            _mm256_stream_pd(&output_buffer[k + (m + 4) * K].re, dif); \
        } \
    } while (0)
#endif

#define FINAL_RADIX2_SCALAR(e, o, output_buffer, k, K) \
    do { \
        for (int m = 0; m < 4; m++) { \
            output_buffer[k + m * K].re = e[m].re + o[m].re; \
            output_buffer[k + m * K].im = e[m].im + o[m].im; \
            output_buffer[k + (m + 4) * K].re = e[m].re - o[m].re; \
            output_buffer[k + (m + 4) * K].im = e[m].im - o[m].im; \
        } \
    } while (0)

//==============================================================================
// APPLY PRECOMPUTED TWIDDLES - IDENTICAL for forward/inverse
//==============================================================================

/**
 * @brief Scalar: Apply stage twiddles to lanes 1-7
 * 
 * stage_tw layout: [W^(1*k), W^(2*k), ..., W^(7*k)] for each k
 */
#define APPLY_STAGE_TWIDDLES_SCALAR(k, x, stage_tw) \
    do { \
        const fft_data *w_ptr = &stage_tw[(k)*7]; \
        for (int j = 1; j <= 7; j++) { \
            fft_data a = x[j]; \
            x[j].re = a.re * w_ptr[j-1].re - a.im * w_ptr[j-1].im; \
            x[j].im = a.re * w_ptr[j-1].im + a.im * w_ptr[j-1].re; \
        } \
    } while (0)

/**
 * @brief AVX2: Apply stage twiddles for 2 butterflies (kk and kk+1)
 */
#ifdef __AVX2__
#define APPLY_STAGE_TWIDDLES_AVX2(kk, x, stage_tw) \
    do { \
        for (int j = 1; j <= 7; j++) { \
            __m256d w = load2_aos(&stage_tw[(kk)*7 + (j-1)], &stage_tw[(kk+1)*7 + (j-1)]); \
            CMUL_FMA_AOS(x[j], x[j], w); \
        } \
    } while (0)
#endif

//==============================================================================
// DATA MOVEMENT - IDENTICAL for forward/inverse
//==============================================================================

#ifdef __AVX2__
#define LOAD_8_LANES_AVX2(kk, K, sub_outputs, x) \
    do { \
        x[0] = load2_aos(&sub_outputs[kk], &sub_outputs[(kk)+1]); \
        for (int j = 1; j <= 7; j++) { \
            x[j] = load2_aos(&sub_outputs[(kk)+j*K], &sub_outputs[(kk)+1+j*K]); \
        } \
    } while (0)
#endif

//==============================================================================
// PREFETCHING - IDENTICAL for forward/inverse
//==============================================================================

#define PREFETCH_L1 16
#define PREFETCH_L2 32
#define PREFETCH_L3 64

#ifdef __AVX2__
#define PREFETCH_8_LANES(k, K, distance, sub_outputs, hint) \
    do { \
        if ((k) + (distance) < K) { \
            _mm_prefetch((const char *)&sub_outputs[(k)+(distance)], hint); \
            for (int j = 1; j < 8; j++) { \
                _mm_prefetch((const char *)&sub_outputs[(k)+(distance)+j*K], hint); \
            } \
        } \
    } while (0)
#endif

#endif // FFT_RADIX8_MACROS_H