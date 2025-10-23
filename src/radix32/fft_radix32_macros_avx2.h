/**
 * @file fft_radix32_macros_native_soa_avx2_corrected.h
 * @brief FULLY CORRECTED Native SoA Radix-32 FFT - AVX2 with All Optimizations
 *
 * @details
 * ALL CORRECTIONS AND OPTIMIZATIONS FROM AVX-512 VERSION:
 * =========================================================
 * ✅ Reduced register pressure (8 lanes per chunk, fits in 16 ymm)
 * ✅ Fully unrolled twiddles (7 explicit statements)
 * ✅ Four W32 octave-specific macros (J0, J1, J2, J3)
 * ✅ Tail protection with masked loads/stores
 * ✅ Correct prefetch bounds checking
 * ✅ All constants hoisted and passed
 * ✅ Fixed twiddle indexing with chunk_offset
 * ✅ Added sfence after streaming stores
 * ✅ Direction-aware W32 application
 * ✅ FMA and non-FMA code paths
 */

#ifndef FFT_RADIX32_MACROS_NATIVE_SOA_AVX2_CORRECTED_H
#define FFT_RADIX32_MACROS_NATIVE_SOA_AVX2_CORRECTED_H

#include "simd_math.h"
#include "fft_twiddles.h"

#ifdef __AVX2__

//==============================================================================
// CONFIGURATION
//==============================================================================

#ifndef FFT_STREAMING_THRESHOLD_K
#define FFT_STREAMING_THRESHOLD_K 256
#endif

#ifndef FFT_PREFETCH_DISTANCE_NEAR
#define FFT_PREFETCH_DISTANCE_NEAR 64
#endif

#ifndef FFT_PREFETCH_DISTANCE_FAR
#define FFT_PREFETCH_DISTANCE_FAR 192
#endif

//==============================================================================
// HOISTED CONSTANT TABLES
//==============================================================================

static const double __attribute__((aligned(32))) W32_COS_TABLE_AVX2[8] = {
    1.0,                    // cos(0)
    0.98078528040323044912, // cos(2π/32)
    0.92387953251128675612, // cos(4π/32)
    0.83146961230254523708, // cos(6π/32)
    0.70710678118654752440, // cos(8π/32) = √2/2
    0.55557023301960222474, // cos(10π/32)
    0.38268343236508977172, // cos(12π/32)
    0.19509032201612826784  // cos(14π/32)
};

static const double __attribute__((aligned(32))) W32_SIN_TABLE_AVX2[8] = {
    0.0,                    // sin(0)
    0.19509032201612826784, // sin(2π/32)
    0.38268343236508977172, // sin(4π/32)
    0.55557023301960222474, // sin(6π/32)
    0.70710678118654752440, // sin(8π/32) = √2/2
    0.83146961230254523708, // sin(10π/32)
    0.92387953251128675612, // sin(12π/32)
    0.98078528040323044912  // sin(14π/32)
};

//==============================================================================
// COMPLEX MULTIPLICATION WITH P0/P1 OPTIMIZATION
//==============================================================================

#ifdef __FMA__
#define CMUL_NATIVE_SOA_AVX2_P0P1(ar, ai, wr, wi, tr, ti) \
    do                                                    \
    {                                                     \
        __m256d ai_wi = _mm256_mul_pd(ai, wi);            \
        __m256d ai_wr = _mm256_mul_pd(ai, wr);            \
        tr = _mm256_fmsub_pd(ar, wr, ai_wi);              \
        ti = _mm256_fmadd_pd(ar, wi, ai_wr);              \
    } while (0)
#else
#define CMUL_NATIVE_SOA_AVX2_P0P1(ar, ai, wr, wi, tr, ti) \
    do                                                    \
    {                                                     \
        __m256d ar_wr = _mm256_mul_pd(ar, wr);            \
        __m256d ai_wi = _mm256_mul_pd(ai, wi);            \
        __m256d ar_wi = _mm256_mul_pd(ar, wi);            \
        __m256d ai_wr = _mm256_mul_pd(ai, wr);            \
        tr = _mm256_sub_pd(ar_wr, ai_wi);                 \
        ti = _mm256_add_pd(ar_wi, ai_wr);                 \
    } while (0)
#endif

//==============================================================================
// RADIX-4 BUTTERFLY
//==============================================================================

#define RADIX4_BUTTERFLY_NATIVE_SOA_AVX2(a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, sign_mask) \
    do                                                                                              \
    {                                                                                               \
        __m256d sumBD_re = _mm256_add_pd(b_re, d_re);                                               \
        __m256d sumBD_im = _mm256_add_pd(b_im, d_im);                                               \
        __m256d difBD_re = _mm256_sub_pd(b_re, d_re);                                               \
        __m256d difBD_im = _mm256_sub_pd(b_im, d_im);                                               \
        __m256d sumAC_re = _mm256_add_pd(a_re, c_re);                                               \
        __m256d sumAC_im = _mm256_add_pd(a_im, c_im);                                               \
        __m256d difAC_re = _mm256_sub_pd(a_re, c_re);                                               \
        __m256d difAC_im = _mm256_sub_pd(a_im, c_im);                                               \
        __m256d rot_re = _mm256_xor_pd(difBD_im, sign_mask);                                        \
        __m256d rot_im = difBD_re;                                                                  \
        __m256d y0_re = _mm256_add_pd(sumAC_re, sumBD_re);                                          \
        __m256d y0_im = _mm256_add_pd(sumAC_im, sumBD_im);                                          \
        __m256d y1_re = _mm256_add_pd(difAC_re, rot_re);                                            \
        __m256d y1_im = _mm256_add_pd(difAC_im, rot_im);                                            \
        __m256d y2_re = _mm256_sub_pd(sumAC_re, sumBD_re);                                          \
        __m256d y2_im = _mm256_sub_pd(sumAC_im, sumBD_im);                                          \
        __m256d y3_re = _mm256_sub_pd(difAC_re, rot_re);                                            \
        __m256d y3_im = _mm256_sub_pd(difAC_im, rot_im);                                            \
        a_re = y0_re;                                                                               \
        a_im = y0_im;                                                                               \
        b_re = y1_re;                                                                               \
        b_im = y1_im;                                                                               \
        c_re = y2_re;                                                                               \
        c_im = y2_im;                                                                               \
        d_re = y3_re;                                                                               \
        d_im = y3_im;                                                                               \
    } while (0)

//==============================================================================
// W_8 TWIDDLES WITH PASSED CONSTANTS
//==============================================================================

#define APPLY_W8_TWIDDLES_HOISTED_AVX2(o1_re, o1_im, o3_re, o3_im, o5_re, o5_im, sqrt2_2, neg_zero)   \
    do                                                                                                \
    {                                                                                                 \
        __m256d t1_re = _mm256_mul_pd(_mm256_sub_pd(o1_re, o1_im), sqrt2_2);                          \
        __m256d t1_im = _mm256_mul_pd(_mm256_add_pd(o1_re, o1_im), sqrt2_2);                          \
        o1_re = t1_re;                                                                                \
        o1_im = t1_im;                                                                                \
        __m256d t3_re = o3_im;                                                                        \
        __m256d t3_im = _mm256_xor_pd(o3_re, neg_zero);                                               \
        o3_re = t3_re;                                                                                \
        o3_im = t3_im;                                                                                \
        __m256d t5_re = _mm256_mul_pd(_mm256_xor_pd(_mm256_add_pd(o5_re, o5_im), neg_zero), sqrt2_2); \
        __m256d t5_im = _mm256_mul_pd(_mm256_sub_pd(o5_im, o5_re), sqrt2_2);                          \
        o5_re = t5_re;                                                                                \
        o5_im = t5_im;                                                                                \
    } while (0)

//==============================================================================
// RADIX-8 COMBINE WITH IMPROVED SCHEDULING
//==============================================================================

#define RADIX8_COMBINE_SCHEDULED_AVX2(                      \
    e0_re, e0_im, e2_re, e2_im, e4_re, e4_im, e6_re, e6_im, \
    o1_re, o1_im, o3_re, o3_im, o5_re, o5_im, o7_re, o7_im, \
    y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, \
    y4_re, y4_im, y5_re, y5_im, y6_re, y6_im, y7_re, y7_im) \
    do                                                      \
    {                                                       \
        y0_re = _mm256_add_pd(e0_re, o1_re);                \
        y0_im = _mm256_add_pd(e0_im, o1_im);                \
        y4_re = _mm256_sub_pd(e0_re, o1_re);                \
        y4_im = _mm256_sub_pd(e0_im, o1_im);                \
        y1_re = _mm256_add_pd(e2_re, o3_re);                \
        y1_im = _mm256_add_pd(e2_im, o3_im);                \
        y5_re = _mm256_sub_pd(e2_re, o3_re);                \
        y5_im = _mm256_sub_pd(e2_im, o3_im);                \
        y2_re = _mm256_add_pd(e4_re, o5_re);                \
        y2_im = _mm256_add_pd(e4_im, o5_im);                \
        y6_re = _mm256_sub_pd(e4_re, o5_re);                \
        y6_im = _mm256_sub_pd(e4_im, o5_im);                \
        y3_re = _mm256_add_pd(e6_re, o7_re);                \
        y3_im = _mm256_add_pd(e6_im, o7_im);                \
        y7_re = _mm256_sub_pd(e6_re, o7_re);                \
        y7_im = _mm256_sub_pd(e6_im, o7_im);                \
    } while (0)

//==============================================================================
// FULLY UNROLLED 8-LANE TWIDDLE APPLICATION
//==============================================================================

#define APPLY_TWIDDLES_8LANE_FULLY_UNROLLED_AVX2(kk_base, x_re, x_im, tw, K_stride, chunk_offset) \
    do                                                                                            \
    {                                                                                             \
        __m256d tw_re, tw_im, res_re, res_im;                                                     \
        /* Lane 1 */ {                                                                            \
            const int gl = (chunk_offset) + 1;                                                    \
            tw_re = _mm256_load_pd(&(tw)->re[(kk_base) + gl * (K_stride)]);                       \
            tw_im = _mm256_load_pd(&(tw)->im[(kk_base) + gl * (K_stride)]);                       \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[1], x_im[1], tw_re, tw_im, res_re, res_im);            \
            x_re[1] = res_re;                                                                     \
            x_im[1] = res_im;                                                                     \
        }                                                                                         \
        /* Lane 2 */ {                                                                            \
            const int gl = (chunk_offset) + 2;                                                    \
            tw_re = _mm256_load_pd(&(tw)->re[(kk_base) + gl * (K_stride)]);                       \
            tw_im = _mm256_load_pd(&(tw)->im[(kk_base) + gl * (K_stride)]);                       \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[2], x_im[2], tw_re, tw_im, res_re, res_im);            \
            x_re[2] = res_re;                                                                     \
            x_im[2] = res_im;                                                                     \
        }                                                                                         \
        /* Lane 3 */ {                                                                            \
            const int gl = (chunk_offset) + 3;                                                    \
            tw_re = _mm256_load_pd(&(tw)->re[(kk_base) + gl * (K_stride)]);                       \
            tw_im = _mm256_load_pd(&(tw)->im[(kk_base) + gl * (K_stride)]);                       \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[3], x_im[3], tw_re, tw_im, res_re, res_im);            \
            x_re[3] = res_re;                                                                     \
            x_im[3] = res_im;                                                                     \
        }                                                                                         \
        /* Lane 4 */ {                                                                            \
            const int gl = (chunk_offset) + 4;                                                    \
            tw_re = _mm256_load_pd(&(tw)->re[(kk_base) + gl * (K_stride)]);                       \
            tw_im = _mm256_load_pd(&(tw)->im[(kk_base) + gl * (K_stride)]);                       \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[4], x_im[4], tw_re, tw_im, res_re, res_im);            \
            x_re[4] = res_re;                                                                     \
            x_im[4] = res_im;                                                                     \
        }                                                                                         \
        /* Lane 5 */ {                                                                            \
            const int gl = (chunk_offset) + 5;                                                    \
            tw_re = _mm256_load_pd(&(tw)->re[(kk_base) + gl * (K_stride)]);                       \
            tw_im = _mm256_load_pd(&(tw)->im[(kk_base) + gl * (K_stride)]);                       \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[5], x_im[5], tw_re, tw_im, res_re, res_im);            \
            x_re[5] = res_re;                                                                     \
            x_im[5] = res_im;                                                                     \
        }                                                                                         \
        /* Lane 6 */ {                                                                            \
            const int gl = (chunk_offset) + 6;                                                    \
            tw_re = _mm256_load_pd(&(tw)->re[(kk_base) + gl * (K_stride)]);                       \
            tw_im = _mm256_load_pd(&(tw)->im[(kk_base) + gl * (K_stride)]);                       \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[6], x_im[6], tw_re, tw_im, res_re, res_im);            \
            x_re[6] = res_re;                                                                     \
            x_im[6] = res_im;                                                                     \
        }                                                                                         \
        /* Lane 7 */ {                                                                            \
            const int gl = (chunk_offset) + 7;                                                    \
            tw_re = _mm256_load_pd(&(tw)->re[(kk_base) + gl * (K_stride)]);                       \
            tw_im = _mm256_load_pd(&(tw)->im[(kk_base) + gl * (K_stride)]);                       \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[7], x_im[7], tw_re, tw_im, res_re, res_im);            \
            x_re[7] = res_re;                                                                     \
            x_im[7] = res_im;                                                                     \
        }                                                                                         \
    } while (0)

//==============================================================================
// FOUR W32 OCTAVE-SPECIFIC MACROS (J0, J1, J2, J3) - AVX2
//==============================================================================

#define APPLY_W32_8LANE_EXPLICIT_J0_AVX2(x_re, x_im, w_cos, w_sin, sqrt2_2, neg_zero, sign_for_conj) \
    do                                                                                               \
    {                                                                                                \
        __m256d t_re, t_im;                                                                          \
        /* Lane 0: W^0 = 1 */                                                                        \
        /* Lane 1 */ {                                                                               \
            __m256d sin_val = _mm256_xor_pd(w_sin[1], sign_for_conj);                                \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[1], x_im[1], w_cos[1], sin_val, t_re, t_im);              \
            x_re[1] = t_re;                                                                          \
            x_im[1] = t_im;                                                                          \
        }                                                                                            \
        /* Lane 2 */ {                                                                               \
            __m256d sin_val = _mm256_xor_pd(w_sin[2], sign_for_conj);                                \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[2], x_im[2], w_cos[2], sin_val, t_re, t_im);              \
            x_re[2] = t_re;                                                                          \
            x_im[2] = t_im;                                                                          \
        }                                                                                            \
        /* Lane 3 */ {                                                                               \
            __m256d sin_val = _mm256_xor_pd(w_sin[3], sign_for_conj);                                \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[3], x_im[3], w_cos[3], sin_val, t_re, t_im);              \
            x_re[3] = t_re;                                                                          \
            x_im[3] = t_im;                                                                          \
        }                                                                                            \
        /* Lane 4: √2/2 special */ {                                                                 \
            t_re = _mm256_mul_pd(_mm256_sub_pd(x_re[4], x_im[4]), sqrt2_2);                          \
            t_im = _mm256_mul_pd(_mm256_add_pd(x_re[4], x_im[4]), sqrt2_2);                          \
            x_re[4] = t_re;                                                                          \
            x_im[4] = t_im;                                                                          \
        }                                                                                            \
        /* Lane 5 */ {                                                                               \
            __m256d sin_val = _mm256_xor_pd(w_cos[3], sign_for_conj);                                \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[5], x_im[5], w_sin[3], sin_val, t_re, t_im);              \
            x_re[5] = t_re;                                                                          \
            x_im[5] = t_im;                                                                          \
        }                                                                                            \
        /* Lane 6 */ {                                                                               \
            __m256d sin_val = _mm256_xor_pd(w_cos[2], sign_for_conj);                                \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[6], x_im[6], w_sin[2], sin_val, t_re, t_im);              \
            x_re[6] = t_re;                                                                          \
            x_im[6] = t_im;                                                                          \
        }                                                                                            \
        /* Lane 7 */ {                                                                               \
            __m256d sin_val = _mm256_xor_pd(w_cos[1], sign_for_conj);                                \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[7], x_im[7], w_sin[1], sin_val, t_re, t_im);              \
            x_re[7] = t_re;                                                                          \
            x_im[7] = t_im;                                                                          \
        }                                                                                            \
    } while (0)

#define APPLY_W32_8LANE_EXPLICIT_J1_AVX2(x_re, x_im, w_cos, w_sin, sqrt2_2, neg_zero, sign_for_conj) \
    do                                                                                               \
    {                                                                                                \
        __m256d t_re, t_im;                                                                          \
        /* Lane 0 (global 8): -i */ {                                                                \
            t_re = x_im[0];                                                                          \
            t_im = _mm256_xor_pd(x_re[0], neg_zero);                                                 \
            x_re[0] = t_re;                                                                          \
            x_im[0] = t_im;                                                                          \
        }                                                                                            \
        /* Lane 1 (global 9): (-1-i)/√2 */ {                                                         \
            __m256d sum = _mm256_add_pd(x_re[1], x_im[1]);                                           \
            __m256d dif = _mm256_sub_pd(x_re[1], x_im[1]);                                           \
            t_re = _mm256_mul_pd(_mm256_xor_pd(sum, neg_zero), sqrt2_2);                             \
            t_im = _mm256_mul_pd(dif, sqrt2_2);                                                      \
            x_re[1] = t_re;                                                                          \
            x_im[1] = t_im;                                                                          \
        }                                                                                            \
        /* Lane 2 (global 10) */ {                                                                   \
            __m256d cos_val = _mm256_xor_pd(w_sin[2], neg_zero);                                     \
            __m256d sin_val = _mm256_xor_pd(w_cos[2], sign_for_conj);                                \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[2], x_im[2], cos_val, sin_val, t_re, t_im);               \
            x_re[2] = t_re;                                                                          \
            x_im[2] = t_im;                                                                          \
        }                                                                                            \
        /* Lane 3 (global 11) */ {                                                                   \
            __m256d cos_val = _mm256_xor_pd(w_sin[1], neg_zero);                                     \
            __m256d sin_val = _mm256_xor_pd(w_cos[1], sign_for_conj);                                \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[3], x_im[3], cos_val, sin_val, t_re, t_im);               \
            x_re[3] = t_re;                                                                          \
            x_im[3] = t_im;                                                                          \
        }                                                                                            \
        /* Lane 4 (global 12): -1 */ {                                                               \
            t_re = _mm256_xor_pd(x_re[4], neg_zero);                                                 \
            t_im = _mm256_xor_pd(x_im[4], neg_zero);                                                 \
            x_re[4] = t_re;                                                                          \
            x_im[4] = t_im;                                                                          \
        }                                                                                            \
        /* Lane 5 (global 13) */ {                                                                   \
            __m256d cos_val = _mm256_xor_pd(w_cos[1], neg_zero);                                     \
            __m256d sin_val = _mm256_xor_pd(w_sin[1], sign_for_conj);                                \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[5], x_im[5], cos_val, sin_val, t_re, t_im);               \
            x_re[5] = t_re;                                                                          \
            x_im[5] = t_im;                                                                          \
        }                                                                                            \
        /* Lane 6 (global 14) */ {                                                                   \
            __m256d cos_val = _mm256_xor_pd(w_cos[2], neg_zero);                                     \
            __m256d sin_val = _mm256_xor_pd(w_sin[2], sign_for_conj);                                \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[6], x_im[6], cos_val, sin_val, t_re, t_im);               \
            x_re[6] = t_re;                                                                          \
            x_im[6] = t_im;                                                                          \
        }                                                                                            \
        /* Lane 7 (global 15) */ {                                                                   \
            __m256d cos_val = _mm256_xor_pd(w_cos[3], neg_zero);                                     \
            __m256d sin_val = _mm256_xor_pd(w_sin[3], sign_for_conj);                                \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[7], x_im[7], cos_val, sin_val, t_re, t_im);               \
            x_re[7] = t_re;                                                                          \
            x_im[7] = t_im;                                                                          \
        }                                                                                            \
    } while (0)

#define APPLY_W32_8LANE_EXPLICIT_J2_AVX2(x_re, x_im, w_cos, w_sin, sqrt2_2, neg_zero, sign_for_conj) \
    do                                                                                               \
    {                                                                                                \
        __m256d t_re, t_im;                                                                          \
        /* Lane 0 (global 16): +i */ {                                                               \
            t_re = _mm256_xor_pd(x_im[0], neg_zero);                                                 \
            t_im = x_re[0];                                                                          \
            x_re[0] = t_re;                                                                          \
            x_im[0] = t_im;                                                                          \
        }                                                                                            \
        /* Lane 1 (global 17) */ {                                                                   \
            __m256d dif = _mm256_sub_pd(x_re[1], x_im[1]);                                           \
            __m256d sum = _mm256_add_pd(x_re[1], x_im[1]);                                           \
            t_re = _mm256_mul_pd(dif, sqrt2_2);                                                      \
            t_im = _mm256_mul_pd(_mm256_xor_pd(sum, neg_zero), sqrt2_2);                             \
            x_re[1] = t_re;                                                                          \
            x_im[1] = t_im;                                                                          \
        }                                                                                            \
        /* Lane 2 (global 18) */ {                                                                   \
            __m256d cos_val = _mm256_xor_pd(w_sin[3], neg_zero);                                     \
            __m256d sin_val = _mm256_xor_pd(_mm256_xor_pd(w_cos[3], neg_zero), sign_for_conj);       \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[2], x_im[2], cos_val, sin_val, t_re, t_im);               \
            x_re[2] = t_re;                                                                          \
            x_im[2] = t_im;                                                                          \
        }                                                                                            \
        /* Lane 3 (global 19) */ {                                                                   \
            __m256d cos_val = _mm256_xor_pd(w_sin[2], neg_zero);                                     \
            __m256d sin_val = _mm256_xor_pd(_mm256_xor_pd(w_cos[2], neg_zero), sign_for_conj);       \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[3], x_im[3], cos_val, sin_val, t_re, t_im);               \
            x_re[3] = t_re;                                                                          \
            x_im[3] = t_im;                                                                          \
        }                                                                                            \
        /* Lane 4 (global 20) */ {                                                                   \
            t_re = _mm256_mul_pd(_mm256_xor_pd(_mm256_add_pd(x_re[4], x_im[4]), neg_zero), sqrt2_2); \
            t_im = _mm256_mul_pd(_mm256_sub_pd(x_im[4], x_re[4]), sqrt2_2);                          \
            x_re[4] = t_re;                                                                          \
            x_im[4] = t_im;                                                                          \
        }                                                                                            \
        /* Lane 5 (global 21) */ {                                                                   \
            __m256d cos_val = _mm256_xor_pd(w_sin[1], neg_zero);                                     \
            __m256d sin_val = _mm256_xor_pd(_mm256_xor_pd(w_cos[1], neg_zero), sign_for_conj);       \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[5], x_im[5], cos_val, sin_val, t_re, t_im);               \
            x_re[5] = t_re;                                                                          \
            x_im[5] = t_im;                                                                          \
        }                                                                                            \
        /* Lane 6 (global 22) */ {                                                                   \
            __m256d cos_val = _mm256_xor_pd(w_cos[3], neg_zero);                                     \
            __m256d sin_val = _mm256_xor_pd(_mm256_xor_pd(w_sin[3], neg_zero), sign_for_conj);       \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[6], x_im[6], cos_val, sin_val, t_re, t_im);               \
            x_re[6] = t_re;                                                                          \
            x_im[6] = t_im;                                                                          \
        }                                                                                            \
        /* Lane 7 (global 23) */ {                                                                   \
            __m256d cos_val = _mm256_xor_pd(w_cos[2], neg_zero);                                     \
            __m256d sin_val = _mm256_xor_pd(_mm256_xor_pd(w_sin[2], neg_zero), sign_for_conj);       \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[7], x_im[7], cos_val, sin_val, t_re, t_im);               \
            x_re[7] = t_re;                                                                          \
            x_im[7] = t_im;                                                                          \
        }                                                                                            \
    } while (0)

#define APPLY_W32_8LANE_EXPLICIT_J3_AVX2(x_re, x_im, w_cos, w_sin, sqrt2_2, neg_zero, sign_for_conj) \
    do                                                                                               \
    {                                                                                                \
        __m256d t_re, t_im;                                                                          \
        /* Lane 0 (global 24): real/-imag */ {                                                       \
            t_re = x_re[0];                                                                          \
            t_im = _mm256_xor_pd(x_im[0], neg_zero);                                                 \
            x_re[0] = t_re;                                                                          \
            x_im[0] = t_im;                                                                          \
        }                                                                                            \
        /* Lane 1 (global 25) */ {                                                                   \
            __m256d cos_val = w_cos[1];                                                              \
            __m256d sin_val = _mm256_xor_pd(_mm256_xor_pd(w_sin[1], neg_zero), sign_for_conj);       \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[1], x_im[1], cos_val, sin_val, t_re, t_im);               \
            x_re[1] = t_re;                                                                          \
            x_im[1] = t_im;                                                                          \
        }                                                                                            \
        /* Lane 2 (global 26) */ {                                                                   \
            __m256d cos_val = w_cos[2];                                                              \
            __m256d sin_val = _mm256_xor_pd(_mm256_xor_pd(w_sin[2], neg_zero), sign_for_conj);       \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[2], x_im[2], cos_val, sin_val, t_re, t_im);               \
            x_re[2] = t_re;                                                                          \
            x_im[2] = t_im;                                                                          \
        }                                                                                            \
        /* Lane 3 (global 27) */ {                                                                   \
            __m256d cos_val = w_cos[3];                                                              \
            __m256d sin_val = _mm256_xor_pd(_mm256_xor_pd(w_sin[3], neg_zero), sign_for_conj);       \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[3], x_im[3], cos_val, sin_val, t_re, t_im);               \
            x_re[3] = t_re;                                                                          \
            x_im[3] = t_im;                                                                          \
        }                                                                                            \
        /* Lane 4 (global 28): (1+i)/√2 */ {                                                         \
            t_re = _mm256_mul_pd(_mm256_add_pd(x_re[4], x_im[4]), sqrt2_2);                          \
            t_im = _mm256_mul_pd(_mm256_sub_pd(x_im[4], x_re[4]), sqrt2_2);                          \
            x_re[4] = t_re;                                                                          \
            x_im[4] = t_im;                                                                          \
        }                                                                                            \
        /* Lane 5 (global 29) */ {                                                                   \
            __m256d cos_val = w_sin[3];                                                              \
            __m256d sin_val = _mm256_xor_pd(_mm256_xor_pd(w_cos[3], neg_zero), sign_for_conj);       \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[5], x_im[5], cos_val, sin_val, t_re, t_im);               \
            x_re[5] = t_re;                                                                          \
            x_im[5] = t_im;                                                                          \
        }                                                                                            \
        /* Lane 6 (global 30) */ {                                                                   \
            __m256d cos_val = w_sin[2];                                                              \
            __m256d sin_val = _mm256_xor_pd(_mm256_xor_pd(w_cos[2], neg_zero), sign_for_conj);       \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[6], x_im[6], cos_val, sin_val, t_re, t_im);               \
            x_re[6] = t_re;                                                                          \
            x_im[6] = t_im;                                                                          \
        }                                                                                            \
        /* Lane 7 (global 31) */ {                                                                   \
            __m256d cos_val = w_sin[1];                                                              \
            __m256d sin_val = _mm256_xor_pd(_mm256_xor_pd(w_cos[1], neg_zero), sign_for_conj);       \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[7], x_im[7], cos_val, sin_val, t_re, t_im);               \
            x_re[7] = t_re;                                                                          \
            x_im[7] = t_im;                                                                          \
        }                                                                                            \
    } while (0)

//==============================================================================
// TAIL MASKING UTILITIES (AVX2)
//==============================================================================

/**
 * @brief Create mask vector for partial loads/stores
 * AVX2 doesn't have mask registers, use vmaskmovpd with mask vector
 */
static inline __m256d fft_create_tail_mask_avx2(int remainder)
{
    static const double __attribute__((aligned(32))) mask_table[8][4] = {
        {0.0, 0.0, 0.0, 0.0},  // 0 elements
        {-1.0, 0.0, 0.0, 0.0}, // 1 element (all bits set = interpret as mask)
        {-1.0, -1.0, 0.0, 0.0},
        {-1.0, -1.0, -1.0, 0.0},
        {-1.0, -1.0, -1.0, -1.0}, // 4 elements
        {-1.0, -1.0, -1.0, -1.0}, // 5+ not used, clamp to 4
        {-1.0, -1.0, -1.0, -1.0},
        {-1.0, -1.0, -1.0, -1.0}};
    int idx = (remainder < 4) ? remainder : 4;
    return _mm256_load_pd(mask_table[idx]);
}

//==============================================================================
// TWO-LEVEL PREFETCH WITH CORRECT BOUNDS
//==============================================================================

#define PREFETCH_TWO_LEVEL_BOUNDED_AVX2(base_ptr, near_offset, far_offset, array_len) \
    do                                                                                \
    {                                                                                 \
        if ((near_offset) < (array_len))                                              \
        {                                                                             \
            _mm_prefetch((const char *)&(base_ptr)[(near_offset)], _MM_HINT_T0);      \
        }                                                                             \
        if ((far_offset) < (array_len))                                               \
        {                                                                             \
            _mm_prefetch((const char *)&(base_ptr)[(far_offset)], _MM_HINT_T1);       \
        }                                                                             \
    } while (0)

//==============================================================================
// MAIN RADIX-32 STAGE - FULLY CORRECTED AVX2
//==============================================================================

/**
 * @brief Fully corrected radix-32 FFT stage for AVX2
 *
 * AVX2-SPECIFIC OPTIMIZATIONS:
 * ============================
 * - 8 lanes per chunk (4-wide doubles, 16 ymm registers)
 * - FMA and non-FMA code paths
 * - vmaskmovpd for tail handling
 * - All corrections from AVX-512 version ported
 */
#define RADIX32_STAGE_CORRECTED_AVX2(in_re, in_im, out_re, out_im, stage_tw, K, sign_mask)                         \
    do                                                                                                             \
    {                                                                                                              \
        /* ============================================================ */                                         \
        /* HOIST ALL CONSTANTS                                          */                                         \
        /* ============================================================ */                                         \
        const __m256d SIGN_MASK = (sign_mask);                                                                     \
        const __m256d NEG_ZERO = _mm256_set1_pd(-0.0);                                                             \
        const __m256d SQRT2_2 = _mm256_set1_pd(W32_COS_TABLE_AVX2[4]);                                             \
        const __m256d SIGN_FOR_CONJ = (sign_mask);                                                                 \
        const int USE_STREAMING = ((K) >= FFT_STREAMING_THRESHOLD_K);                                              \
        const int K_VAL = (K);                                                                                     \
        const int TOTAL_ELEMENTS = 32 * K_VAL;                                                                     \
                                                                                                                   \
        /* Load W32 constants from tables */                                                                       \
        __m256d W32_COS[8], W32_SIN[8];                                                                            \
        for (int i = 0; i < 8; ++i)                                                                                \
        {                                                                                                          \
            W32_COS[i] = _mm256_set1_pd(W32_COS_TABLE_AVX2[i]);                                                    \
            W32_SIN[i] = _mm256_set1_pd(W32_SIN_TABLE_AVX2[i]);                                                    \
        }                                                                                                          \
                                                                                                                   \
        /* Alignment hints */                                                                                      \
        double *restrict in_re_aligned = (double *)__builtin_assume_aligned((in_re), 32);                          \
        double *restrict in_im_aligned = (double *)__builtin_assume_aligned((in_im), 32);                          \
        double *restrict out_re_aligned = (double *)__builtin_assume_aligned((out_re), 32);                        \
        double *restrict out_im_aligned = (double *)__builtin_assume_aligned((out_im), 32);                        \
        const double *restrict tw_re_aligned = (const double *)__builtin_assume_aligned((stage_tw)->re, 32);       \
        const double *restrict tw_im_aligned = (const double *)__builtin_assume_aligned((stage_tw)->im, 32);       \
                                                                                                                   \
        /* Tail mask for K % 4 != 0 */                                                                             \
        const int K_REMAINDER = K_VAL & 3;                                                                         \
        const __m256d TAIL_MASK = K_REMAINDER ? fft_create_tail_mask_avx2(K_REMAINDER) : _mm256_setzero_pd();      \
        const int K_FULL_VECS = K_VAL & ~3;                                                                        \
                                                                                                                   \
        /* ============================================================ */                                         \
        /* MAIN LOOP: 8 BUTTERFLIES × 8-LANE CHUNKS                    */                                          \
        /* ============================================================ */                                         \
        for (int b = 0; b < 8; ++b)                                                                                \
        {                                                                                                          \
            _Pragma("GCC ivdep") for (int kk = 0; kk < K_FULL_VECS; kk += 4)                                       \
            {                                                                                                      \
                const int kk_base = kk + b * 4;                                                                    \
                                                                                                                   \
                /* Two-level prefetch */                                                                           \
                const int pf_near_offset = kk_base + FFT_PREFETCH_DISTANCE_NEAR;                                   \
                const int pf_far_offset = kk_base + FFT_PREFETCH_DISTANCE_FAR;                                     \
                                                                                                                   \
                for (int pf_lane = 0; pf_lane < 32; pf_lane += 8)                                                  \
                {                                                                                                  \
                    PREFETCH_TWO_LEVEL_BOUNDED_AVX2(in_re_aligned,                                                 \
                                                    pf_near_offset + pf_lane * K_VAL,                              \
                                                    pf_far_offset + pf_lane * K_VAL,                               \
                                                    TOTAL_ELEMENTS);                                               \
                    PREFETCH_TWO_LEVEL_BOUNDED_AVX2(in_im_aligned,                                                 \
                                                    pf_near_offset + pf_lane * K_VAL,                              \
                                                    pf_far_offset + pf_lane * K_VAL,                               \
                                                    TOTAL_ELEMENTS);                                               \
                }                                                                                                  \
                                                                                                                   \
                /* -------------------- PROCESS 8-LANE CHUNKS -------------------- */                              \
                for (int chunk = 0; chunk < 4; ++chunk)                                                            \
                {                                                                                                  \
                    const int chunk_offset = chunk * 8;                                                            \
                                                                                                                   \
                    /* Minimal live set: 8 lanes (16 ymm) */                                                       \
                    __m256d x_re[8], x_im[8];                                                                      \
                                                                                                                   \
                    /* LOAD */                                                                                     \
                    for (int lane = 0; lane < 8; ++lane)                                                           \
                    {                                                                                              \
                        const int global_lane = chunk_offset + lane;                                               \
                        const int idx = kk_base + global_lane * K_VAL;                                             \
                        x_re[lane] = _mm256_load_pd(&in_re_aligned[idx]);                                          \
                        x_im[lane] = _mm256_load_pd(&in_im_aligned[idx]);                                          \
                    }                                                                                              \
                                                                                                                   \
                    /* STAGE TWIDDLES */                                                                           \
                    if (chunk_offset > 0)                                                                          \
                    {                                                                                              \
                        APPLY_TWIDDLES_8LANE_FULLY_UNROLLED_AVX2(kk_base, x_re, x_im,                              \
                                                                 stage_tw, K_VAL, chunk_offset);                   \
                    }                                                                                              \
                    else                                                                                           \
                    {                                                                                              \
                        __m256d tw_re, tw_im, res_re, res_im;                                                      \
                        for (int lane = 1; lane < 8; ++lane)                                                       \
                        {                                                                                          \
                            const int gl = chunk_offset + lane;                                                    \
                            const int tw_idx = kk_base + gl * K_VAL;                                               \
                            tw_re = _mm256_load_pd(&tw_re_aligned[tw_idx]);                                        \
                            tw_im = _mm256_load_pd(&tw_im_aligned[tw_idx]);                                        \
                            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[lane], x_im[lane],                                      \
                                                      tw_re, tw_im, res_re, res_im);                               \
                            x_re[lane] = res_re;                                                                   \
                            x_im[lane] = res_im;                                                                   \
                        }                                                                                          \
                    }                                                                                              \
                                                                                                                   \
                    /* FIRST RADIX-4 */                                                                            \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX2(x_re[0], x_im[0], x_re[1], x_im[1],                           \
                                                     x_re[2], x_im[2], x_re[3], x_im[3],                           \
                                                     SIGN_MASK);                                                   \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX2(x_re[4], x_im[4], x_re[5], x_im[5],                           \
                                                     x_re[6], x_im[6], x_re[7], x_im[7],                           \
                                                     SIGN_MASK);                                                   \
                                                                                                                   \
                    /* W32 TWIDDLES - SELECT CORRECT OCTAVE */                                                     \
                    switch (chunk_offset)                                                                          \
                    {                                                                                              \
                    case 0:                                                                                        \
                        APPLY_W32_8LANE_EXPLICIT_J0_AVX2(x_re, x_im, W32_COS, W32_SIN,                             \
                                                         SQRT2_2, NEG_ZERO, SIGN_FOR_CONJ);                        \
                        break;                                                                                     \
                    case 8:                                                                                        \
                        APPLY_W32_8LANE_EXPLICIT_J1_AVX2(x_re, x_im, W32_COS, W32_SIN,                             \
                                                         SQRT2_2, NEG_ZERO, SIGN_FOR_CONJ);                        \
                        break;                                                                                     \
                    case 16:                                                                                       \
                        APPLY_W32_8LANE_EXPLICIT_J2_AVX2(x_re, x_im, W32_COS, W32_SIN,                             \
                                                         SQRT2_2, NEG_ZERO, SIGN_FOR_CONJ);                        \
                        break;                                                                                     \
                    case 24:                                                                                       \
                        APPLY_W32_8LANE_EXPLICIT_J3_AVX2(x_re, x_im, W32_COS, W32_SIN,                             \
                                                         SQRT2_2, NEG_ZERO, SIGN_FOR_CONJ);                        \
                        break;                                                                                     \
                    }                                                                                              \
                                                                                                                   \
                    /* RADIX-8 DECOMPOSITION */                                                                    \
                    __m256d e0_re = x_re[0], e0_im = x_im[0];                                                      \
                    __m256d e2_re = x_re[2], e2_im = x_im[2];                                                      \
                    __m256d e4_re = x_re[4], e4_im = x_im[4];                                                      \
                    __m256d e6_re = x_re[6], e6_im = x_im[6];                                                      \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX2(e0_re, e0_im, e2_re, e2_im,                                   \
                                                     e4_re, e4_im, e6_re, e6_im, SIGN_MASK);                       \
                                                                                                                   \
                    __m256d o1_re = x_re[1], o1_im = x_im[1];                                                      \
                    __m256d o3_re = x_re[3], o3_im = x_im[3];                                                      \
                    __m256d o5_re = x_re[5], o5_im = x_im[5];                                                      \
                    __m256d o7_re = x_re[7], o7_im = x_im[7];                                                      \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX2(o1_re, o1_im, o3_re, o3_im,                                   \
                                                     o5_re, o5_im, o7_re, o7_im, SIGN_MASK);                       \
                                                                                                                   \
                    APPLY_W8_TWIDDLES_HOISTED_AVX2(o1_re, o1_im, o3_re, o3_im, o5_re, o5_im,                       \
                                                   SQRT2_2, NEG_ZERO);                                             \
                                                                                                                   \
                    RADIX8_COMBINE_SCHEDULED_AVX2(                                                                 \
                        e0_re, e0_im, e2_re, e2_im, e4_re, e4_im, e6_re, e6_im,                                    \
                        o1_re, o1_im, o3_re, o3_im, o5_re, o5_im, o7_re, o7_im,                                    \
                        x_re[0], x_im[0], x_re[1], x_im[1], x_re[2], x_im[2], x_re[3], x_im[3],                    \
                        x_re[4], x_im[4], x_re[5], x_im[5], x_re[6], x_im[6], x_re[7], x_im[7]);                   \
                                                                                                                   \
                    /* STORE */                                                                                    \
                    for (int lane = 0; lane < 8; ++lane)                                                           \
                    {                                                                                              \
                        const int out_lane = chunk_offset + lane;                                                  \
                        const int out_idx = kk_base + out_lane * K_VAL;                                            \
                                                                                                                   \
                        if (kk + 4 < K_FULL_VECS)                                                                  \
                        {                                                                                          \
                            _mm_prefetch((const char *)&out_re_aligned[out_idx + 4], _MM_HINT_T0);                 \
                            _mm_prefetch((const char *)&out_im_aligned[out_idx + 4], _MM_HINT_T0);                 \
                        }                                                                                          \
                                                                                                                   \
                        if (USE_STREAMING)                                                                         \
                        {                                                                                          \
                            _mm256_stream_pd(&out_re_aligned[out_idx], x_re[lane]);                                \
                            _mm256_stream_pd(&out_im_aligned[out_idx], x_im[lane]);                                \
                        }                                                                                          \
                        else                                                                                       \
                        {                                                                                          \
                            _mm256_store_pd(&out_re_aligned[out_idx], x_re[lane]);                                 \
                            _mm256_store_pd(&out_im_aligned[out_idx], x_im[lane]);                                 \
                        }                                                                                          \
                    }                                                                                              \
                } /* End chunk loop */                                                                             \
            } /* End kk loop */                                                                                    \
        } /* End b loop */                                                                                         \
                                                                                                                   \
        /* ============================================================ */                                         \
        /* TAIL HANDLING: K % 4 != 0                                    */                                         \
        /* ============================================================ */                                         \
        if (K_REMAINDER > 0)                                                                                       \
        {                                                                                                          \
            for (int b = 0; b < 8; ++b)                                                                            \
            {                                                                                                      \
                const int kk = K_FULL_VECS;                                                                        \
                const int kk_base = kk + b * 4;                                                                    \
                                                                                                                   \
                for (int chunk = 0; chunk < 4; ++chunk)                                                            \
                {                                                                                                  \
                    const int chunk_offset = chunk * 8;                                                            \
                    __m256d x_re[8], x_im[8];                                                                      \
                                                                                                                   \
                    /* MASKED LOAD */                                                                              \
                    for (int lane = 0; lane < 8; ++lane)                                                           \
                    {                                                                                              \
                        const int global_lane = chunk_offset + lane;                                               \
                        const int idx = kk_base + global_lane * K_VAL;                                             \
                        x_re[lane] = _mm256_maskload_pd(&in_re_aligned[idx], _mm256_castpd_si256(TAIL_MASK));      \
                        x_im[lane] = _mm256_maskload_pd(&in_im_aligned[idx], _mm256_castpd_si256(TAIL_MASK));      \
                    }                                                                                              \
                                                                                                                   \
                    /* Apply transformations (same as main loop) */                                                \
                    if (chunk_offset > 0)                                                                          \
                    {                                                                                              \
                        APPLY_TWIDDLES_8LANE_FULLY_UNROLLED_AVX2(kk_base, x_re, x_im,                              \
                                                                 stage_tw, K_VAL, chunk_offset);                   \
                    }                                                                                              \
                    else                                                                                           \
                    {                                                                                              \
                        __m256d tw_re, tw_im, res_re, res_im;                                                      \
                        for (int lane = 1; lane < 8; ++lane)                                                       \
                        {                                                                                          \
                            const int gl = chunk_offset + lane;                                                    \
                            const int tw_idx = kk_base + gl * K_VAL;                                               \
                            tw_re = _mm256_maskload_pd(&tw_re_aligned[tw_idx], _mm256_castpd_si256(TAIL_MASK));    \
                            tw_im = _mm256_maskload_pd(&tw_im_aligned[tw_idx], _mm256_castpd_si256(TAIL_MASK));    \
                            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[lane], x_im[lane],                                      \
                                                      tw_re, tw_im, res_re, res_im);                               \
                            x_re[lane] = res_re;                                                                   \
                            x_im[lane] = res_im;                                                                   \
                        }                                                                                          \
                    }                                                                                              \
                                                                                                                   \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX2(x_re[0], x_im[0], x_re[1], x_im[1],                           \
                                                     x_re[2], x_im[2], x_re[3], x_im[3],                           \
                                                     SIGN_MASK);                                                   \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX2(x_re[4], x_im[4], x_re[5], x_im[5],                           \
                                                     x_re[6], x_im[6], x_re[7], x_im[7],                           \
                                                     SIGN_MASK);                                                   \
                                                                                                                   \
                    switch (chunk_offset)                                                                          \
                    {                                                                                              \
                    case 0:                                                                                        \
                        APPLY_W32_8LANE_EXPLICIT_J0_AVX2(x_re, x_im, W32_COS, W32_SIN,                             \
                                                         SQRT2_2, NEG_ZERO, SIGN_FOR_CONJ);                        \
                        break;                                                                                     \
                    case 8:                                                                                        \
                        APPLY_W32_8LANE_EXPLICIT_J1_AVX2(x_re, x_im, W32_COS, W32_SIN,                             \
                                                         SQRT2_2, NEG_ZERO, SIGN_FOR_CONJ);                        \
                        break;                                                                                     \
                    case 16:                                                                                       \
                        APPLY_W32_8LANE_EXPLICIT_J2_AVX2(x_re, x_im, W32_COS, W32_SIN,                             \
                                                         SQRT2_2, NEG_ZERO, SIGN_FOR_CONJ);                        \
                        break;                                                                                     \
                    case 24:                                                                                       \
                        APPLY_W32_8LANE_EXPLICIT_J3_AVX2(x_re, x_im, W32_COS, W32_SIN,                             \
                                                         SQRT2_2, NEG_ZERO, SIGN_FOR_CONJ);                        \
                        break;                                                                                     \
                    }                                                                                              \
                                                                                                                   \
                    __m256d e0_re = x_re[0], e0_im = x_im[0];                                                      \
                    __m256d e2_re = x_re[2], e2_im = x_im[2];                                                      \
                    __m256d e4_re = x_re[4], e4_im = x_im[4];                                                      \
                    __m256d e6_re = x_re[6], e6_im = x_im[6];                                                      \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX2(e0_re, e0_im, e2_re, e2_im,                                   \
                                                     e4_re, e4_im, e6_re, e6_im, SIGN_MASK);                       \
                                                                                                                   \
                    __m256d o1_re = x_re[1], o1_im = x_im[1];                                                      \
                    __m256d o3_re = x_re[3], o3_im = x_im[3];                                                      \
                    __m256d o5_re = x_re[5], o5_im = x_im[5];                                                      \
                    __m256d o7_re = x_re[7], o7_im = x_im[7];                                                      \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX2(o1_re, o1_im, o3_re, o3_im,                                   \
                                                     o5_re, o5_im, o7_re, o7_im, SIGN_MASK);                       \
                                                                                                                   \
                    APPLY_W8_TWIDDLES_HOISTED_AVX2(o1_re, o1_im, o3_re, o3_im, o5_re, o5_im,                       \
                                                   SQRT2_2, NEG_ZERO);                                             \
                                                                                                                   \
                    RADIX8_COMBINE_SCHEDULED_AVX2(                                                                 \
                        e0_re, e0_im, e2_re, e2_im, e4_re, e4_im, e6_re, e6_im,                                    \
                        o1_re, o1_im, o3_re, o3_im, o5_re, o5_im, o7_re, o7_im,                                    \
                        x_re[0], x_im[0], x_re[1], x_im[1], x_re[2], x_im[2], x_re[3], x_im[3],                    \
                        x_re[4], x_im[4], x_re[5], x_im[5], x_re[6], x_im[6], x_re[7], x_im[7]);                   \
                                                                                                                   \
                    /* MASKED STORE */                                                                             \
                    for (int lane = 0; lane < 8; ++lane)                                                           \
                    {                                                                                              \
                        const int out_lane = chunk_offset + lane;                                                  \
                        const int out_idx = kk_base + out_lane * K_VAL;                                            \
                        _mm256_maskstore_pd(&out_re_aligned[out_idx], _mm256_castpd_si256(TAIL_MASK), x_re[lane]); \
                        _mm256_maskstore_pd(&out_im_aligned[out_idx], _mm256_castpd_si256(TAIL_MASK), x_im[lane]); \
                    }                                                                                              \
                }                                                                                                  \
            } /* End b loop for tail */                                                                            \
        }                                                                                                          \
                                                                                                                   \
        /* ============================================================ */                                         \
        /* SFENCE AFTER STREAMING STORES                                */                                         \
        /* ============================================================ */                                         \
        if (USE_STREAMING)                                                                                         \
        {                                                                                                          \
            _mm_sfence();                                                                                          \
        }                                                                                                          \
    } while (0)

#endif // __AVX2__
#endif // FFT_RADIX32_MACROS_NATIVE_SOA_AVX2_CORRECTED_H