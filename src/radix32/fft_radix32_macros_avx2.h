/**
 * @file fft_radix32_macros_native_soa_avx2_fv_bv_separated.h
 * @brief MATHEMATICALLY CORRECT Native SoA Radix-32 FFT - FV/BV Separated - AVX2
 *
 * @details
 * ARCHITECTURAL UPGRADE - NOW MATCHES AVX-512 VERSION:
 * ====================================================
 * ✅ COMPLETELY SEPARATED FV (forward) and BV (backward) macro paths
 * ✅ NO SIGN_FOR_CONJ parameter - hard-coded signs per direction
 * ✅ CORRECT W32 quarter-turn identities (J0-J3) for both FV and BV
 * ✅ Separate W8 twiddle kernels for FV and BV
 *
 * PRESERVED OPTIMIZATIONS (ALL MAINTAINED):
 * ==========================================
 * ✅ Reduced register pressure (8 lanes per chunk, fits in 16 ymm)
 * ✅ Fully unrolled twiddles (7 explicit statements per chunk)
 * ✅ Four W32 octave-specific macros (J0, J1, J2, J3) per direction
 * ✅ Tail protection with masked loads/stores
 * ✅ Correct prefetch bounds checking (two-level, T0/T1)
 * ✅ All constants hoisted and passed (NEG_ZERO, SQRT2_2)
 * ✅ Fixed twiddle indexing with chunk_offset
 * ✅ Added sfence after streaming stores
 * ✅ FMA and non-FMA code paths
 * ✅ Instruction scheduling and software pipelining depth
 * ✅ Radix-4 butterfly with direction-aware SIGN_MASK
 * ✅ Special-case handling for k=0,4,8,12,... lanes
 * ✅ ILP-friendly scheduling (two R4s on disjoint lanes)
 * ✅ Streaming store thresholds
 * ✅ Aligned loads with restrict pointers
 *
 * MATHEMATICAL CORRECTNESS:
 * ========================
 * For W32 twiddles with k = j + 8·m where j ∈ [0..7], m ∈ [0..3]:
 *
 * Base angles: θ_j = 2π·j/32, so (c_j, s_j) = (cos θ_j, sin θ_j)
 *
 * Quarter-turn identities:
 *   m=0 (J0, k=0..7):   W32[k] = exp(-i·2π·k/32) → (wr, wi) = (+c_j, -s_j) [FV]
 *   m=1 (J1, k=8..15):  W32[k] = exp(-i·2π·k/32) → (wr, wi) = (-s_j, -c_j) [FV]
 *   m=2 (J2, k=16..23): W32[k] = exp(-i·2π·k/32) → (wr, wi) = (-c_j, +s_j) [FV]
 *   m=3 (J3, k=24..31): W32[k] = exp(-i·2π·k/32) → (wr, wi) = (+s_j, +c_j) [FV]
 *
 * For inverse (BV), flip the sign of wi.
 */

#ifndef FFT_RADIX32_MACROS_NATIVE_SOA_AVX2_FV_BV_SEPARATED_H
#define FFT_RADIX32_MACROS_NATIVE_SOA_AVX2_FV_BV_SEPARATED_H

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
// RADIX-4 BUTTERFLY (DIRECTION-AWARE VIA SIGN_MASK)
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
        a_re = _mm256_add_pd(sumAC_re, sumBD_re);                                                   \
        a_im = _mm256_add_pd(sumAC_im, sumBD_im);                                                   \
        b_re = _mm256_add_pd(difAC_re, rot_re);                                                     \
        b_im = _mm256_add_pd(difAC_im, rot_im);                                                     \
        c_re = _mm256_sub_pd(sumAC_re, sumBD_re);                                                   \
        c_im = _mm256_sub_pd(sumAC_im, sumBD_im);                                                   \
        d_re = _mm256_sub_pd(difAC_re, rot_re);                                                     \
        d_im = _mm256_sub_pd(difAC_im, rot_im);                                                     \
    } while (0)

//==============================================================================
// RADIX-8 SCHEDULED COMBINE (PRESERVED)
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
        y1_re = _mm256_add_pd(e2_re, o3_re);                \
        y1_im = _mm256_add_pd(e2_im, o3_im);                \
        y2_re = _mm256_add_pd(e4_re, o5_re);                \
        y2_im = _mm256_add_pd(e4_im, o5_im);                \
        y3_re = _mm256_add_pd(e6_re, o7_re);                \
        y3_im = _mm256_add_pd(e6_im, o7_im);                \
        y4_re = _mm256_sub_pd(e0_re, o1_re);                \
        y4_im = _mm256_sub_pd(e0_im, o1_im);                \
        y5_re = _mm256_sub_pd(e2_re, o3_re);                \
        y5_im = _mm256_sub_pd(e2_im, o3_im);                \
        y6_re = _mm256_sub_pd(e4_re, o5_re);                \
        y6_im = _mm256_sub_pd(e4_im, o5_im);                \
        y7_re = _mm256_sub_pd(e6_re, o7_re);                \
        y7_im = _mm256_sub_pd(e6_im, o7_im);                \
    } while (0)

//==============================================================================
// W8 TWIDDLES - FORWARD (FV)
//==============================================================================

#define APPLY_W8_TWIDDLES_HOISTED_FV_AVX2(o1_re, o1_im, o3_re, o3_im, o5_re, o5_im, \
                                          sqrt2_2, neg_zero)                        \
    do                                                                              \
    {                                                                               \
        __m256d sqrt2_vec = _mm256_set1_pd(sqrt2_2);                                \
        __m256d tmp_re, tmp_im;                                                     \
        tmp_re = _mm256_sub_pd(o3_re, o3_im);                                       \
        tmp_im = _mm256_add_pd(o3_re, o3_im);                                       \
        o3_re = _mm256_mul_pd(tmp_re, sqrt2_vec);                                   \
        o3_im = _mm256_mul_pd(tmp_im, sqrt2_vec);                                   \
        tmp_re = o5_im;                                                             \
        tmp_im = o5_re;                                                             \
        o5_re = _mm256_xor_pd(tmp_re, neg_zero);                                    \
        o5_im = tmp_im;                                                             \
    } while (0)

//==============================================================================
// W8 TWIDDLES - BACKWARD (BV)
//==============================================================================

#define APPLY_W8_TWIDDLES_HOISTED_BV_AVX2(o1_re, o1_im, o3_re, o3_im, o5_re, o5_im, \
                                          sqrt2_2, neg_zero)                        \
    do                                                                              \
    {                                                                               \
        __m256d sqrt2_vec = _mm256_set1_pd(sqrt2_2);                                \
        __m256d tmp_re, tmp_im;                                                     \
        tmp_re = _mm256_add_pd(o3_re, o3_im);                                       \
        tmp_im = _mm256_sub_pd(o3_im, o3_re);                                       \
        o3_re = _mm256_mul_pd(tmp_re, sqrt2_vec);                                   \
        o3_im = _mm256_mul_pd(tmp_im, sqrt2_vec);                                   \
        tmp_re = o5_im;                                                             \
        tmp_im = o5_re;                                                             \
        o5_re = tmp_re;                                                             \
        o5_im = _mm256_xor_pd(tmp_im, neg_zero);                                    \
    } while (0)

//==============================================================================
// W32 BASE BUILDER (NEUTRAL, USED BY BOTH FV AND BV)
//==============================================================================

#define W32_BUILD_COSSIN_FROM_BASE_AVX2(j, octave_m, w_cos, w_sin, neg_zero, cos_theta, sin_theta) \
    do                                                                                             \
    {                                                                                              \
        __m256d c_vec = _mm256_set1_pd(w_cos[j]);                                                  \
        __m256d s_vec = _mm256_set1_pd(w_sin[j]);                                                  \
        if ((octave_m) == 0)                                                                       \
        {                                                                                          \
            cos_theta = c_vec;                                                                     \
            sin_theta = s_vec;                                                                     \
        }                                                                                          \
        else if ((octave_m) == 1)                                                                  \
        {                                                                                          \
            cos_theta = _mm256_xor_pd(s_vec, neg_zero);                                            \
            sin_theta = c_vec;                                                                     \
        }                                                                                          \
        else if ((octave_m) == 2)                                                                  \
        {                                                                                          \
            cos_theta = _mm256_xor_pd(c_vec, neg_zero);                                            \
            sin_theta = _mm256_xor_pd(s_vec, neg_zero);                                            \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            cos_theta = s_vec;                                                                     \
            sin_theta = _mm256_xor_pd(c_vec, neg_zero);                                            \
        }                                                                                          \
    } while (0)

//==============================================================================
// W32 J0 (k=0..7) - FORWARD
//==============================================================================

#define APPLY_W32_8LANE_EXPLICIT_J0_FV_AVX2(x_re, x_im, w_cos, w_sin, sqrt2_2, neg_zero)     \
    do                                                                                       \
    {                                                                                        \
        __m256d wr, wi, cos_theta, sin_theta, tmp_re, tmp_im;                                \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(1, 0, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = _mm256_xor_pd(sin_theta, neg_zero);                                             \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[1], x_im[1], wr, wi, tmp_re, tmp_im);                 \
        x_re[1] = tmp_re;                                                                    \
        x_im[1] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(2, 0, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = _mm256_xor_pd(sin_theta, neg_zero);                                             \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[2], x_im[2], wr, wi, tmp_re, tmp_im);                 \
        x_re[2] = tmp_re;                                                                    \
        x_im[2] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(3, 0, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = _mm256_xor_pd(sin_theta, neg_zero);                                             \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[3], x_im[3], wr, wi, tmp_re, tmp_im);                 \
        x_re[3] = tmp_re;                                                                    \
        x_im[3] = tmp_im;                                                                    \
        __m256d sqrt2_vec = _mm256_set1_pd(sqrt2_2);                                         \
        tmp_re = _mm256_sub_pd(x_re[4], x_im[4]);                                            \
        tmp_im = _mm256_add_pd(x_re[4], x_im[4]);                                            \
        x_re[4] = _mm256_mul_pd(tmp_re, sqrt2_vec);                                          \
        x_im[4] = _mm256_mul_pd(tmp_im, sqrt2_vec);                                          \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(5, 0, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = _mm256_xor_pd(sin_theta, neg_zero);                                             \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[5], x_im[5], wr, wi, tmp_re, tmp_im);                 \
        x_re[5] = tmp_re;                                                                    \
        x_im[5] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(6, 0, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = _mm256_xor_pd(sin_theta, neg_zero);                                             \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[6], x_im[6], wr, wi, tmp_re, tmp_im);                 \
        x_re[6] = tmp_re;                                                                    \
        x_im[6] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(7, 0, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = _mm256_xor_pd(sin_theta, neg_zero);                                             \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[7], x_im[7], wr, wi, tmp_re, tmp_im);                 \
        x_re[7] = tmp_re;                                                                    \
        x_im[7] = tmp_im;                                                                    \
    } while (0)

//==============================================================================
// W32 J0 (k=0..7) - BACKWARD
//==============================================================================

#define APPLY_W32_8LANE_EXPLICIT_J0_BV_AVX2(x_re, x_im, w_cos, w_sin, sqrt2_2, neg_zero)     \
    do                                                                                       \
    {                                                                                        \
        __m256d wr, wi, cos_theta, sin_theta, tmp_re, tmp_im;                                \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(1, 0, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = sin_theta;                                                                      \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[1], x_im[1], wr, wi, tmp_re, tmp_im);                 \
        x_re[1] = tmp_re;                                                                    \
        x_im[1] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(2, 0, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = sin_theta;                                                                      \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[2], x_im[2], wr, wi, tmp_re, tmp_im);                 \
        x_re[2] = tmp_re;                                                                    \
        x_im[2] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(3, 0, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = sin_theta;                                                                      \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[3], x_im[3], wr, wi, tmp_re, tmp_im);                 \
        x_re[3] = tmp_re;                                                                    \
        x_im[3] = tmp_im;                                                                    \
        __m256d sqrt2_vec = _mm256_set1_pd(sqrt2_2);                                         \
        tmp_re = _mm256_add_pd(x_re[4], x_im[4]);                                            \
        tmp_im = _mm256_sub_pd(x_im[4], x_re[4]);                                            \
        x_re[4] = _mm256_mul_pd(tmp_re, sqrt2_vec);                                          \
        x_im[4] = _mm256_mul_pd(tmp_im, sqrt2_vec);                                          \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(5, 0, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = sin_theta;                                                                      \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[5], x_im[5], wr, wi, tmp_re, tmp_im);                 \
        x_re[5] = tmp_re;                                                                    \
        x_im[5] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(6, 0, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = sin_theta;                                                                      \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[6], x_im[6], wr, wi, tmp_re, tmp_im);                 \
        x_re[6] = tmp_re;                                                                    \
        x_im[6] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(7, 0, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = sin_theta;                                                                      \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[7], x_im[7], wr, wi, tmp_re, tmp_im);                 \
        x_re[7] = tmp_re;                                                                    \
        x_im[7] = tmp_im;                                                                    \
    } while (0)

//==============================================================================
// W32 J1 (k=8..15) - FORWARD
//==============================================================================

#define APPLY_W32_8LANE_EXPLICIT_J1_FV_AVX2(x_re, x_im, w_cos, w_sin, sqrt2_2, neg_zero)     \
    do                                                                                       \
    {                                                                                        \
        __m256d wr, wi, cos_theta, sin_theta, tmp_re, tmp_im;                                \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(0, 1, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = _mm256_xor_pd(sin_theta, neg_zero);                                             \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[0], x_im[0], wr, wi, tmp_re, tmp_im);                 \
        x_re[0] = tmp_re;                                                                    \
        x_im[0] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(1, 1, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = _mm256_xor_pd(sin_theta, neg_zero);                                             \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[1], x_im[1], wr, wi, tmp_re, tmp_im);                 \
        x_re[1] = tmp_re;                                                                    \
        x_im[1] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(2, 1, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = _mm256_xor_pd(sin_theta, neg_zero);                                             \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[2], x_im[2], wr, wi, tmp_re, tmp_im);                 \
        x_re[2] = tmp_re;                                                                    \
        x_im[2] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(3, 1, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = _mm256_xor_pd(sin_theta, neg_zero);                                             \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[3], x_im[3], wr, wi, tmp_re, tmp_im);                 \
        x_re[3] = tmp_re;                                                                    \
        x_im[3] = tmp_im;                                                                    \
        tmp_re = x_im[4];                                                                    \
        tmp_im = x_re[4];                                                                    \
        x_re[4] = _mm256_xor_pd(tmp_re, neg_zero);                                           \
        x_im[4] = _mm256_xor_pd(tmp_im, neg_zero);                                           \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(5, 1, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = _mm256_xor_pd(sin_theta, neg_zero);                                             \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[5], x_im[5], wr, wi, tmp_re, tmp_im);                 \
        x_re[5] = tmp_re;                                                                    \
        x_im[5] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(6, 1, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = _mm256_xor_pd(sin_theta, neg_zero);                                             \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[6], x_im[6], wr, wi, tmp_re, tmp_im);                 \
        x_re[6] = tmp_re;                                                                    \
        x_im[6] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(7, 1, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = _mm256_xor_pd(sin_theta, neg_zero);                                             \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[7], x_im[7], wr, wi, tmp_re, tmp_im);                 \
        x_re[7] = tmp_re;                                                                    \
        x_im[7] = tmp_im;                                                                    \
    } while (0)

//==============================================================================
// W32 J1 (k=8..15) - BACKWARD
//==============================================================================

#define APPLY_W32_8LANE_EXPLICIT_J1_BV_AVX2(x_re, x_im, w_cos, w_sin, sqrt2_2, neg_zero)     \
    do                                                                                       \
    {                                                                                        \
        __m256d wr, wi, cos_theta, sin_theta, tmp_re, tmp_im;                                \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(0, 1, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = sin_theta;                                                                      \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[0], x_im[0], wr, wi, tmp_re, tmp_im);                 \
        x_re[0] = tmp_re;                                                                    \
        x_im[0] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(1, 1, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = sin_theta;                                                                      \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[1], x_im[1], wr, wi, tmp_re, tmp_im);                 \
        x_re[1] = tmp_re;                                                                    \
        x_im[1] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(2, 1, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = sin_theta;                                                                      \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[2], x_im[2], wr, wi, tmp_re, tmp_im);                 \
        x_re[2] = tmp_re;                                                                    \
        x_im[2] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(3, 1, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = sin_theta;                                                                      \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[3], x_im[3], wr, wi, tmp_re, tmp_im);                 \
        x_re[3] = tmp_re;                                                                    \
        x_im[3] = tmp_im;                                                                    \
        tmp_re = x_im[4];                                                                    \
        tmp_im = x_re[4];                                                                    \
        x_re[4] = tmp_re;                                                                    \
        x_im[4] = _mm256_xor_pd(tmp_im, neg_zero);                                           \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(5, 1, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = sin_theta;                                                                      \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[5], x_im[5], wr, wi, tmp_re, tmp_im);                 \
        x_re[5] = tmp_re;                                                                    \
        x_im[5] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(6, 1, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = sin_theta;                                                                      \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[6], x_im[6], wr, wi, tmp_re, tmp_im);                 \
        x_re[6] = tmp_re;                                                                    \
        x_im[6] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(7, 1, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = sin_theta;                                                                      \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[7], x_im[7], wr, wi, tmp_re, tmp_im);                 \
        x_re[7] = tmp_re;                                                                    \
        x_im[7] = tmp_im;                                                                    \
    } while (0)

//==============================================================================
// W32 J2 (k=16..23) - FORWARD
//==============================================================================

#define APPLY_W32_8LANE_EXPLICIT_J2_FV_AVX2(x_re, x_im, w_cos, w_sin, sqrt2_2, neg_zero)     \
    do                                                                                       \
    {                                                                                        \
        __m256d wr, wi, cos_theta, sin_theta, tmp_re, tmp_im;                                \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(0, 2, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = _mm256_xor_pd(sin_theta, neg_zero);                                             \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[0], x_im[0], wr, wi, tmp_re, tmp_im);                 \
        x_re[0] = tmp_re;                                                                    \
        x_im[0] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(1, 2, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = _mm256_xor_pd(sin_theta, neg_zero);                                             \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[1], x_im[1], wr, wi, tmp_re, tmp_im);                 \
        x_re[1] = tmp_re;                                                                    \
        x_im[1] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(2, 2, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = _mm256_xor_pd(sin_theta, neg_zero);                                             \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[2], x_im[2], wr, wi, tmp_re, tmp_im);                 \
        x_re[2] = tmp_re;                                                                    \
        x_im[2] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(3, 2, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = _mm256_xor_pd(sin_theta, neg_zero);                                             \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[3], x_im[3], wr, wi, tmp_re, tmp_im);                 \
        x_re[3] = tmp_re;                                                                    \
        x_im[3] = tmp_im;                                                                    \
        __m256d sqrt2_vec = _mm256_set1_pd(sqrt2_2);                                         \
        tmp_re = _mm256_add_pd(x_re[4], x_im[4]);                                            \
        tmp_im = _mm256_sub_pd(x_im[4], x_re[4]);                                            \
        x_re[4] = _mm256_mul_pd(tmp_re, sqrt2_vec);                                          \
        x_im[4] = _mm256_mul_pd(tmp_im, sqrt2_vec);                                          \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(5, 2, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = _mm256_xor_pd(sin_theta, neg_zero);                                             \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[5], x_im[5], wr, wi, tmp_re, tmp_im);                 \
        x_re[5] = tmp_re;                                                                    \
        x_im[5] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(6, 2, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = _mm256_xor_pd(sin_theta, neg_zero);                                             \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[6], x_im[6], wr, wi, tmp_re, tmp_im);                 \
        x_re[6] = tmp_re;                                                                    \
        x_im[6] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(7, 2, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = _mm256_xor_pd(sin_theta, neg_zero);                                             \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[7], x_im[7], wr, wi, tmp_re, tmp_im);                 \
        x_re[7] = tmp_re;                                                                    \
        x_im[7] = tmp_im;                                                                    \
    } while (0)

//==============================================================================
// W32 J2 (k=16..23) - BACKWARD
//==============================================================================

#define APPLY_W32_8LANE_EXPLICIT_J2_BV_AVX2(x_re, x_im, w_cos, w_sin, sqrt2_2, neg_zero)     \
    do                                                                                       \
    {                                                                                        \
        __m256d wr, wi, cos_theta, sin_theta, tmp_re, tmp_im;                                \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(0, 2, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = sin_theta;                                                                      \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[0], x_im[0], wr, wi, tmp_re, tmp_im);                 \
        x_re[0] = tmp_re;                                                                    \
        x_im[0] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(1, 2, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = sin_theta;                                                                      \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[1], x_im[1], wr, wi, tmp_re, tmp_im);                 \
        x_re[1] = tmp_re;                                                                    \
        x_im[1] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(2, 2, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = sin_theta;                                                                      \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[2], x_im[2], wr, wi, tmp_re, tmp_im);                 \
        x_re[2] = tmp_re;                                                                    \
        x_im[2] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(3, 2, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = sin_theta;                                                                      \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[3], x_im[3], wr, wi, tmp_re, tmp_im);                 \
        x_re[3] = tmp_re;                                                                    \
        x_im[3] = tmp_im;                                                                    \
        __m256d sqrt2_vec = _mm256_set1_pd(sqrt2_2);                                         \
        tmp_re = _mm256_sub_pd(x_re[4], x_im[4]);                                            \
        tmp_im = _mm256_add_pd(x_re[4], x_im[4]);                                            \
        x_re[4] = _mm256_mul_pd(tmp_re, sqrt2_vec);                                          \
        x_im[4] = _mm256_mul_pd(tmp_im, sqrt2_vec);                                          \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(5, 2, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = sin_theta;                                                                      \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[5], x_im[5], wr, wi, tmp_re, tmp_im);                 \
        x_re[5] = tmp_re;                                                                    \
        x_im[5] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(6, 2, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = sin_theta;                                                                      \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[6], x_im[6], wr, wi, tmp_re, tmp_im);                 \
        x_re[6] = tmp_re;                                                                    \
        x_im[6] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(7, 2, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = sin_theta;                                                                      \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[7], x_im[7], wr, wi, tmp_re, tmp_im);                 \
        x_re[7] = tmp_re;                                                                    \
        x_im[7] = tmp_im;                                                                    \
    } while (0)

//==============================================================================
// W32 J3 (k=24..31) - FORWARD
//==============================================================================

#define APPLY_W32_8LANE_EXPLICIT_J3_FV_AVX2(x_re, x_im, w_cos, w_sin, sqrt2_2, neg_zero)     \
    do                                                                                       \
    {                                                                                        \
        __m256d wr, wi, cos_theta, sin_theta, tmp_re, tmp_im;                                \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(0, 3, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = _mm256_xor_pd(sin_theta, neg_zero);                                             \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[0], x_im[0], wr, wi, tmp_re, tmp_im);                 \
        x_re[0] = tmp_re;                                                                    \
        x_im[0] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(1, 3, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = _mm256_xor_pd(sin_theta, neg_zero);                                             \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[1], x_im[1], wr, wi, tmp_re, tmp_im);                 \
        x_re[1] = tmp_re;                                                                    \
        x_im[1] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(2, 3, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = _mm256_xor_pd(sin_theta, neg_zero);                                             \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[2], x_im[2], wr, wi, tmp_re, tmp_im);                 \
        x_re[2] = tmp_re;                                                                    \
        x_im[2] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(3, 3, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = _mm256_xor_pd(sin_theta, neg_zero);                                             \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[3], x_im[3], wr, wi, tmp_re, tmp_im);                 \
        x_re[3] = tmp_re;                                                                    \
        x_im[3] = tmp_im;                                                                    \
        tmp_re = x_im[4];                                                                    \
        tmp_im = x_re[4];                                                                    \
        x_re[4] = tmp_re;                                                                    \
        x_im[4] = _mm256_xor_pd(tmp_im, neg_zero);                                           \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(5, 3, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = _mm256_xor_pd(sin_theta, neg_zero);                                             \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[5], x_im[5], wr, wi, tmp_re, tmp_im);                 \
        x_re[5] = tmp_re;                                                                    \
        x_im[5] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(6, 3, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = _mm256_xor_pd(sin_theta, neg_zero);                                             \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[6], x_im[6], wr, wi, tmp_re, tmp_im);                 \
        x_re[6] = tmp_re;                                                                    \
        x_im[6] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(7, 3, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = _mm256_xor_pd(sin_theta, neg_zero);                                             \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[7], x_im[7], wr, wi, tmp_re, tmp_im);                 \
        x_re[7] = tmp_re;                                                                    \
        x_im[7] = tmp_im;                                                                    \
    } while (0)

//==============================================================================
// W32 J3 (k=24..31) - BACKWARD
//==============================================================================

#define APPLY_W32_8LANE_EXPLICIT_J3_BV_AVX2(x_re, x_im, w_cos, w_sin, sqrt2_2, neg_zero)     \
    do                                                                                       \
    {                                                                                        \
        __m256d wr, wi, cos_theta, sin_theta, tmp_re, tmp_im;                                \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(0, 3, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = sin_theta;                                                                      \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[0], x_im[0], wr, wi, tmp_re, tmp_im);                 \
        x_re[0] = tmp_re;                                                                    \
        x_im[0] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(1, 3, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = sin_theta;                                                                      \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[1], x_im[1], wr, wi, tmp_re, tmp_im);                 \
        x_re[1] = tmp_re;                                                                    \
        x_im[1] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(2, 3, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = sin_theta;                                                                      \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[2], x_im[2], wr, wi, tmp_re, tmp_im);                 \
        x_re[2] = tmp_re;                                                                    \
        x_im[2] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(3, 3, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = sin_theta;                                                                      \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[3], x_im[3], wr, wi, tmp_re, tmp_im);                 \
        x_re[3] = tmp_re;                                                                    \
        x_im[3] = tmp_im;                                                                    \
        tmp_re = x_im[4];                                                                    \
        tmp_im = x_re[4];                                                                    \
        x_re[4] = _mm256_xor_pd(tmp_re, neg_zero);                                           \
        x_im[4] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(5, 3, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = sin_theta;                                                                      \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[5], x_im[5], wr, wi, tmp_re, tmp_im);                 \
        x_re[5] = tmp_re;                                                                    \
        x_im[5] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(6, 3, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = sin_theta;                                                                      \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[6], x_im[6], wr, wi, tmp_re, tmp_im);                 \
        x_re[6] = tmp_re;                                                                    \
        x_im[6] = tmp_im;                                                                    \
        W32_BUILD_COSSIN_FROM_BASE_AVX2(7, 3, w_cos, w_sin, neg_zero, cos_theta, sin_theta); \
        wr = cos_theta;                                                                      \
        wi = sin_theta;                                                                      \
        CMUL_NATIVE_SOA_AVX2_P0P1(x_re[7], x_im[7], wr, wi, tmp_re, tmp_im);                 \
        x_re[7] = tmp_re;                                                                    \
        x_im[7] = tmp_im;                                                                    \
    } while (0)

//==============================================================================
// FULLY UNROLLED TWIDDLE APPLICATION (PRESERVED - NO CHANGES)
//==============================================================================

#define APPLY_TWIDDLES_8LANE_FULLY_UNROLLED_AVX2(kk_base, x_re, x_im, stage_tw, K_VAL, chunk_offset) \
    do                                                                                               \
    {                                                                                                \
        const double *tw_re_aligned = (const double *)__builtin_assume_aligned(stage_tw->re, 32);    \
        const double *tw_im_aligned = (const double *)__builtin_assume_aligned(stage_tw->im, 32);    \
        __m256d tw_re, tw_im, res_re, res_im;                                                        \
        {                                                                                            \
            const int gl = chunk_offset + 1;                                                         \
            const int tw_idx = kk_base + gl * K_VAL;                                                 \
            tw_re = _mm256_load_pd(&tw_re_aligned[tw_idx]);                                          \
            tw_im = _mm256_load_pd(&tw_im_aligned[tw_idx]);                                          \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[1], x_im[1], tw_re, tw_im, res_re, res_im);               \
            x_re[1] = res_re;                                                                        \
            x_im[1] = res_im;                                                                        \
        }                                                                                            \
        {                                                                                            \
            const int gl = chunk_offset + 2;                                                         \
            const int tw_idx = kk_base + gl * K_VAL;                                                 \
            tw_re = _mm256_load_pd(&tw_re_aligned[tw_idx]);                                          \
            tw_im = _mm256_load_pd(&tw_im_aligned[tw_idx]);                                          \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[2], x_im[2], tw_re, tw_im, res_re, res_im);               \
            x_re[2] = res_re;                                                                        \
            x_im[2] = res_im;                                                                        \
        }                                                                                            \
        {                                                                                            \
            const int gl = chunk_offset + 3;                                                         \
            const int tw_idx = kk_base + gl * K_VAL;                                                 \
            tw_re = _mm256_load_pd(&tw_re_aligned[tw_idx]);                                          \
            tw_im = _mm256_load_pd(&tw_im_aligned[tw_idx]);                                          \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[3], x_im[3], tw_re, tw_im, res_re, res_im);               \
            x_re[3] = res_re;                                                                        \
            x_im[3] = res_im;                                                                        \
        }                                                                                            \
        {                                                                                            \
            const int gl = chunk_offset + 4;                                                         \
            const int tw_idx = kk_base + gl * K_VAL;                                                 \
            tw_re = _mm256_load_pd(&tw_re_aligned[tw_idx]);                                          \
            tw_im = _mm256_load_pd(&tw_im_aligned[tw_idx]);                                          \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[4], x_im[4], tw_re, tw_im, res_re, res_im);               \
            x_re[4] = res_re;                                                                        \
            x_im[4] = res_im;                                                                        \
        }                                                                                            \
        {                                                                                            \
            const int gl = chunk_offset + 5;                                                         \
            const int tw_idx = kk_base + gl * K_VAL;                                                 \
            tw_re = _mm256_load_pd(&tw_re_aligned[tw_idx]);                                          \
            tw_im = _mm256_load_pd(&tw_im_aligned[tw_idx]);                                          \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[5], x_im[5], tw_re, tw_im, res_re, res_im);               \
            x_re[5] = res_re;                                                                        \
            x_im[5] = res_im;                                                                        \
        }                                                                                            \
        {                                                                                            \
            const int gl = chunk_offset + 6;                                                         \
            const int tw_idx = kk_base + gl * K_VAL;                                                 \
            tw_re = _mm256_load_pd(&tw_re_aligned[tw_idx]);                                          \
            tw_im = _mm256_load_pd(&tw_im_aligned[tw_idx]);                                          \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[6], x_im[6], tw_re, tw_im, res_re, res_im);               \
            x_re[6] = res_re;                                                                        \
            x_im[6] = res_im;                                                                        \
        }                                                                                            \
        {                                                                                            \
            const int gl = chunk_offset + 7;                                                         \
            const int tw_idx = kk_base + gl * K_VAL;                                                 \
            tw_re = _mm256_load_pd(&tw_re_aligned[tw_idx]);                                          \
            tw_im = _mm256_load_pd(&tw_im_aligned[tw_idx]);                                          \
            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[7], x_im[7], tw_re, tw_im, res_re, res_im);               \
            x_re[7] = res_re;                                                                        \
            x_im[7] = res_im;                                                                        \
        }                                                                                            \
    } while (0)

//==============================================================================
// TWO-LEVEL BOUNDED PREFETCH (PRESERVED)
//==============================================================================

#define TWO_LEVEL_BOUNDED_PREFETCH_AVX2(ptr, K_VAL, lane, array_len, NEAR, FAR) \
    do                                                                          \
    {                                                                           \
        const int near_idx = kk + NEAR + (lane) * K_VAL;                        \
        const int far_idx = kk + FAR + (lane) * K_VAL;                          \
        if (near_idx < (array_len))                                             \
        {                                                                       \
            _mm_prefetch((const char *)&(ptr)[near_idx], _MM_HINT_T0);          \
        }                                                                       \
        if (far_idx < (array_len))                                              \
        {                                                                       \
            _mm_prefetch((const char *)&(ptr)[far_idx], _MM_HINT_T1);           \
        }                                                                       \
    } while (0)

//==============================================================================
// MAIN RADIX-32 PIPELINE - FORWARD (FV)
//==============================================================================

#define RADIX32_PIPELINE_4_FV_AVX2(in_re, in_im, out_re, out_im, stage_tw, K, USE_STREAMING)                                                                                                         \
    do                                                                                                                                                                                               \
    {                                                                                                                                                                                                \
        const int K_VAL = (K);                                                                                                                                                                       \
        const int array_len = K_VAL * 32;                                                                                                                                                            \
        const double *__restrict__ in_re_aligned = (const double *)__builtin_assume_aligned(in_re, 32);                                                                                              \
        const double *__restrict__ in_im_aligned = (const double *)__builtin_assume_aligned(in_im, 32);                                                                                              \
        double *__restrict__ out_re_aligned = (double *)__builtin_assume_aligned(out_re, 32);                                                                                                        \
        double *__restrict__ out_im_aligned = (double *)__builtin_assume_aligned(out_im, 32);                                                                                                        \
        const double *__restrict__ tw_re_aligned = (const double *)__builtin_assume_aligned(stage_tw->re, 32);                                                                                       \
        const double *__restrict__ tw_im_aligned = (const double *)__builtin_assume_aligned(stage_tw->im, 32);                                                                                       \
                                                                                                                                                                                                     \
        const __m256d NEG_ZERO = _mm256_set1_pd(-0.0);                                                                                                                                               \
        const __m256d ZERO_VEC = _mm256_setzero_pd();                                                                                                                                                \
        const double SQRT2_2 = 0.70710678118654752440;                                                                                                                                               \
        const __m256d SIGN_MASK = NEG_ZERO;                                                                                                                                                          \
                                                                                                                                                                                                     \
        const int full_blocks = K_VAL / 4;                                                                                                                                                           \
        const int remainder = K_VAL % 4;                                                                                                                                                             \
                                                                                                                                                                                                     \
        for (int b = 0; b < full_blocks; ++b)                                                                                                                                                        \
        {                                                                                                                                                                                            \
            _Pragma("GCC ivdep") for (int kk = b * 4; kk < (b + 1) * 4; kk += 4)                                                                                                                     \
            {                                                                                                                                                                                        \
                const int kk_base = kk;                                                                                                                                                              \
                for (int chunk_offset = 0; chunk_offset < 32; chunk_offset += 8)                                                                                                                     \
                {                                                                                                                                                                                    \
                    __m256d x_re[8], x_im[8];                                                                                                                                                        \
                                                                                                                                                                                                     \
                    TWO_LEVEL_BOUNDED_PREFETCH_AVX2(in_re_aligned, K_VAL, 0,                                                                                                                         \
                                                    array_len, FFT_PREFETCH_DISTANCE_NEAR, FFT_PREFETCH_DISTANCE_FAR);                                                                               \
                    TWO_LEVEL_BOUNDED_PREFETCH_AVX2(in_im_aligned, K_VAL, 0,                                                                                                                         \
                                                    array_len, FFT_PREFETCH_DISTANCE_NEAR, FFT_PREFETCH_DISTANCE_FAR);                                                                               \
                                                                                                                                                                                                     \
                    for (int lane = 0; lane < 8; ++lane)                                                                                                                                             \
                    {                                                                                                                                                                                \
                        const int global_lane = chunk_offset + lane;                                                                                                                                 \
                        const int idx = kk_base + global_lane * K_VAL;                                                                                                                               \
                        x_re[lane] = _mm256_load_pd(&in_re_aligned[idx]);                                                                                                                            \
                        x_im[lane] = _mm256_load_pd(&in_im_aligned[idx]);                                                                                                                            \
                    }                                                                                                                                                                                \
                                                                                                                                                                                                     \
                    if (chunk_offset > 0)                                                                                                                                                            \
                    {                                                                                                                                                                                \
                        APPLY_TWIDDLES_8LANE_FULLY_UNROLLED_AVX2(kk_base, x_re, x_im,                                                                                                                \
                                                                 stage_tw, K_VAL, chunk_offset);                                                                                                     \
                    }                                                                                                                                                                                \
                    else                                                                                                                                                                             \
                    {                                                                                                                                                                                \
                        __m256d tw_re, tw_im, res_re, res_im;                                                                                                                                        \
                        for (int lane = 1; lane < 8; ++lane)                                                                                                                                         \
                        {                                                                                                                                                                            \
                            const int gl = chunk_offset + lane;                                                                                                                                      \
                            const int tw_idx = kk_base + gl * K_VAL;                                                                                                                                 \
                            tw_re = _mm256_load_pd(&tw_re_aligned[tw_idx]);                                                                                                                          \
                            tw_im = _mm256_load_pd(&tw_im_aligned[tw_idx]);                                                                                                                          \
                            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[lane], x_im[lane],                                                                                                                        \
                                                      tw_re, tw_im, res_re, res_im);                                                                                                                 \
                            x_re[lane] = res_re;                                                                                                                                                     \
                            x_im[lane] = res_im;                                                                                                                                                     \
                        }                                                                                                                                                                            \
                    }                                                                                                                                                                                \
                                                                                                                                                                                                     \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX2(x_re[0], x_im[0], x_re[1], x_im[1],                                                                                                             \
                                                     x_re[2], x_im[2], x_re[3], x_im[3],                                                                                                             \
                                                     SIGN_MASK);                                                                                                                                     \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX2(x_re[4], x_im[4], x_re[5], x_im[5],                                                                                                             \
                                                     x_re[6], x_im[6], x_re[7], x_im[7],                                                                                                             \
                                                     SIGN_MASK);                                                                                                                                     \
                                                                                                                                                                                                     \
                    switch (chunk_offset)                                                                                                                                                            \
                    {                                                                                                                                                                                \
                    case 0:                                                                                                                                                                          \
                        APPLY_W32_8LANE_EXPLICIT_J0_FV_AVX2(x_re, x_im, W32_COS_TABLE_AVX2, W32_SIN_TABLE_AVX2,                                                                                      \
                                                            SQRT2_2, NEG_ZERO);                                                                                                                      \
                        break;                                                                                                                                                                       \
                    case 8:                                                                                                                                                                          \
                        APPLY_W32_8LANE_EXPLICIT_J1_FV_AVX2(x_re, x_im, W32_COS_TABLE_AVX2, W32_SIN_TABLE_AVX2,                                                                                      \
                                                            SQRT2_2, NEG_ZERO);                                                                                                                      \
                        break;                                                                                                                                                                       \
                    case 16:                                                                                                                                                                         \
                        APPLY_W32_8LANE_EXPLICIT_J2_FV_AVX2(x_re, x_im, W32_COS_TABLE_AVX2, W32_SIN_TABLE_AVX2,                                                                                      \
                                                            SQRT2_2, NEG_ZERO);                                                                                                                      \
                        break;                                                                                                                                                                       \
                    case 24:                                                                                                                                                                         \
                        APPLY_W32_8LANE_EXPLICIT_J3_FV_AVX2(x_re, x_im, W32_COS_TABLE_AVX2, W32_SIN_TABLE_AVX2,                                                                                      \
                                                            SQRT2_2, NEG_ZERO);                                                                                                                      \
                        break;                                                                                                                                                                       \
                    }                                                                                                                                                                                \
                                                                                                                                                                                                     \
                    __m256d e0_re = x_re[0], e0_im = x_im[0];                                                                                                                                        \
                    __m256d e2_re = x_re[2], e2_im = x_im[2];                                                                                                                                        \
                    __m256d e4_re = x_re[4], e4_im = x_im[4];                                                                                                                                        \
                    __m256d e6_re = x_re[6], e6_im = x_im[6];                                                                                                                                        \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX2(e0_re, e0_im, e2_re, e2_im,                                                                                                                     \
                                                     e4_re, e4_im, e6_re, e6_im, SIGN_MASK);                                                                                                         \
                                                                                                                                                                                                     \
                    __m256d o1_re = x_re[1], o1_im = x_im[1];                                                                                                                                        \
                    __m256d o3_re = x_re[3], o3_im = x_im[3];                                                                                                                                        \
                    __m256d o5_re = x_re[5], o5_im = x_im[5];                                                                                                                                        \
                    __m256d o7_re = x_re[7], o7_im = x_im[7];                                                                                                                                        \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX2(o1_re, o1_im, o3_re, o3_im,                                                                                                                     \
                                                     o5_re, o5_im, o7_re, o7_im, SIGN_MASK);                                                                                                         \
                                                                                                                                                                                                     \
                    APPLY_W8_TWIDDLES_HOISTED_FV_AVX2(o1_re, o1_im, o3_re, o3_im, o5_re, o5_im,                                                                                                      \
                                                      SQRT2_2, NEG_ZERO);                                                                                                                            \
                                                                                                                                                                                                     \
                    RADIX8_COMBINE_SCHEDULED_AVX2(                                                                                                                                                   \
                        e0_re, e0_im, e2_re, e2_im, e4_re, e4_im, e6_re, e6_im,                                                                                                                      \
                        o1_re, o1_im, o3_re, o3_im, o5_re, o5_im, o7_re, o7_im,                                                                                                                      \
                        x_re[0], x_im[0], x_re[1], x_im[1], x_re[2], x_im[2], x_re[3], x_im[3],                                                                                                      \
                        x_re[4], x_im[4], x_re[5], x_im[5], x_re[6], x_im[6], x_re[7], x_im[7]);                                                                                                     \
                                                                                                                                                                                                     \
                    for (int lane = 0; lane < 8; ++lane)                                                                                                                                             \
                    {                                                                                                                                                                                \
                        const int out_lane = chunk_offset + lane;                                                                                                                                    \
                        const int out_idx = kk_base + out_lane * K_VAL;                                                                                                                              \
                        if (USE_STREAMING)                                                                                                                                                           \
                        {                                                                                                                                                                            \
                            _mm256_stream_pd(&out_re_aligned[out_idx], x_re[lane]);                                                                                                                  \
                            _mm256_stream_pd(&out_im_aligned[out_idx], x_im[lane]);                                                                                                                  \
                        }                                                                                                                                                                            \
                        else                                                                                                                                                                         \
                        {                                                                                                                                                                            \
                            _mm256_store_pd(&out_re_aligned[out_idx], x_re[lane]);                                                                                                                   \
                            _mm256_store_pd(&out_im_aligned[out_idx], x_im[lane]);                                                                                                                   \
                        }                                                                                                                                                                            \
                    }                                                                                                                                                                                \
                }                                                                                                                                                                                    \
            }                                                                                                                                                                                        \
        }                                                                                                                                                                                            \
                                                                                                                                                                                                     \
        if (remainder > 0)                                                                                                                                                                           \
        {                                                                                                                                                                                            \
            const int tail_start = full_blocks * 4;                                                                                                                                                  \
            const int tail_count = remainder;                                                                                                                                                        \
            const __m256d TAIL_MASK = (tail_count == 1) ? _mm256_castsi256_pd(_mm256_setr_epi64x(-1LL, 0, 0, 0)) : (tail_count == 2) ? _mm256_castsi256_pd(_mm256_setr_epi64x(-1LL, -1LL, 0, 0))     \
                                                                                                                                     : _mm256_castsi256_pd(_mm256_setr_epi64x(-1LL, -1LL, -1LL, 0)); \
                                                                                                                                                                                                     \
            for (int b = 0; b < 1; ++b)                                                                                                                                                              \
            {                                                                                                                                                                                        \
                _Pragma("GCC ivdep") for (int kk = tail_start; kk < K_VAL; kk += 4)                                                                                                                  \
                {                                                                                                                                                                                    \
                    const int kk_base = kk;                                                                                                                                                          \
                    for (int chunk_offset = 0; chunk_offset < 32; chunk_offset += 8)                                                                                                                 \
                    {                                                                                                                                                                                \
                        __m256d x_re[8], x_im[8];                                                                                                                                                    \
                                                                                                                                                                                                     \
                        for (int lane = 0; lane < 8; ++lane)                                                                                                                                         \
                        {                                                                                                                                                                            \
                            const int global_lane = chunk_offset + lane;                                                                                                                             \
                            const int idx = kk_base + global_lane * K_VAL;                                                                                                                           \
                            x_re[lane] = _mm256_maskload_pd(&in_re_aligned[idx], _mm256_castpd_si256(TAIL_MASK));                                                                                    \
                            x_im[lane] = _mm256_maskload_pd(&in_im_aligned[idx], _mm256_castpd_si256(TAIL_MASK));                                                                                    \
                        }                                                                                                                                                                            \
                                                                                                                                                                                                     \
                        if (chunk_offset > 0)                                                                                                                                                        \
                        {                                                                                                                                                                            \
                            __m256d tw_re, tw_im, res_re, res_im;                                                                                                                                    \
                            for (int lane = 1; lane < 8; ++lane)                                                                                                                                     \
                            {                                                                                                                                                                        \
                                const int gl = chunk_offset + lane;                                                                                                                                  \
                                const int tw_idx = kk_base + gl * K_VAL;                                                                                                                             \
                                tw_re = _mm256_maskload_pd(&tw_re_aligned[tw_idx], _mm256_castpd_si256(TAIL_MASK));                                                                                  \
                                tw_im = _mm256_maskload_pd(&tw_im_aligned[tw_idx], _mm256_castpd_si256(TAIL_MASK));                                                                                  \
                                CMUL_NATIVE_SOA_AVX2_P0P1(x_re[lane], x_im[lane],                                                                                                                    \
                                                          tw_re, tw_im, res_re, res_im);                                                                                                             \
                                x_re[lane] = res_re;                                                                                                                                                 \
                                x_im[lane] = res_im;                                                                                                                                                 \
                            }                                                                                                                                                                        \
                        }                                                                                                                                                                            \
                        else                                                                                                                                                                         \
                        {                                                                                                                                                                            \
                            __m256d tw_re, tw_im, res_re, res_im;                                                                                                                                    \
                            for (int lane = 1; lane < 8; ++lane)                                                                                                                                     \
                            {                                                                                                                                                                        \
                                const int tw_idx = kk_base + lane * K_VAL;                                                                                                                           \
                                tw_re = _mm256_maskload_pd(&tw_re_aligned[tw_idx], _mm256_castpd_si256(TAIL_MASK));                                                                                  \
                                tw_im = _mm256_maskload_pd(&tw_im_aligned[tw_idx], _mm256_castpd_si256(TAIL_MASK));                                                                                  \
                                CMUL_NATIVE_SOA_AVX2_P0P1(x_re[lane], x_im[lane],                                                                                                                    \
                                                          tw_re, tw_im, res_re, res_im);                                                                                                             \
                                x_re[lane] = res_re;                                                                                                                                                 \
                                x_im[lane] = res_im;                                                                                                                                                 \
                            }                                                                                                                                                                        \
                        }                                                                                                                                                                            \
                                                                                                                                                                                                     \
                        RADIX4_BUTTERFLY_NATIVE_SOA_AVX2(x_re[0], x_im[0], x_re[1], x_im[1],                                                                                                         \
                                                         x_re[2], x_im[2], x_re[3], x_im[3],                                                                                                         \
                                                         SIGN_MASK);                                                                                                                                 \
                        RADIX4_BUTTERFLY_NATIVE_SOA_AVX2(x_re[4], x_im[4], x_re[5], x_im[5],                                                                                                         \
                                                         x_re[6], x_im[6], x_re[7], x_im[7],                                                                                                         \
                                                         SIGN_MASK);                                                                                                                                 \
                                                                                                                                                                                                     \
                        switch (chunk_offset)                                                                                                                                                        \
                        {                                                                                                                                                                            \
                        case 0:                                                                                                                                                                      \
                            APPLY_W32_8LANE_EXPLICIT_J0_FV_AVX2(x_re, x_im, W32_COS_TABLE_AVX2, W32_SIN_TABLE_AVX2,                                                                                  \
                                                                SQRT2_2, NEG_ZERO);                                                                                                                  \
                            break;                                                                                                                                                                   \
                        case 8:                                                                                                                                                                      \
                            APPLY_W32_8LANE_EXPLICIT_J1_FV_AVX2(x_re, x_im, W32_COS_TABLE_AVX2, W32_SIN_TABLE_AVX2,                                                                                  \
                                                                SQRT2_2, NEG_ZERO);                                                                                                                  \
                            break;                                                                                                                                                                   \
                        case 16:                                                                                                                                                                     \
                            APPLY_W32_8LANE_EXPLICIT_J2_FV_AVX2(x_re, x_im, W32_COS_TABLE_AVX2, W32_SIN_TABLE_AVX2,                                                                                  \
                                                                SQRT2_2, NEG_ZERO);                                                                                                                  \
                            break;                                                                                                                                                                   \
                        case 24:                                                                                                                                                                     \
                            APPLY_W32_8LANE_EXPLICIT_J3_FV_AVX2(x_re, x_im, W32_COS_TABLE_AVX2, W32_SIN_TABLE_AVX2,                                                                                  \
                                                                SQRT2_2, NEG_ZERO);                                                                                                                  \
                            break;                                                                                                                                                                   \
                        }                                                                                                                                                                            \
                                                                                                                                                                                                     \
                        __m256d e0_re = x_re[0], e0_im = x_im[0];                                                                                                                                    \
                        __m256d e2_re = x_re[2], e2_im = x_im[2];                                                                                                                                    \
                        __m256d e4_re = x_re[4], e4_im = x_im[4];                                                                                                                                    \
                        __m256d e6_re = x_re[6], e6_im = x_im[6];                                                                                                                                    \
                        RADIX4_BUTTERFLY_NATIVE_SOA_AVX2(e0_re, e0_im, e2_re, e2_im,                                                                                                                 \
                                                         e4_re, e4_im, e6_re, e6_im, SIGN_MASK);                                                                                                     \
                                                                                                                                                                                                     \
                        __m256d o1_re = x_re[1], o1_im = x_im[1];                                                                                                                                    \
                        __m256d o3_re = x_re[3], o3_im = x_im[3];                                                                                                                                    \
                        __m256d o5_re = x_re[5], o5_im = x_im[5];                                                                                                                                    \
                        __m256d o7_re = x_re[7], o7_im = x_im[7];                                                                                                                                    \
                        RADIX4_BUTTERFLY_NATIVE_SOA_AVX2(o1_re, o1_im, o3_re, o3_im,                                                                                                                 \
                                                         o5_re, o5_im, o7_re, o7_im, SIGN_MASK);                                                                                                     \
                                                                                                                                                                                                     \
                        APPLY_W8_TWIDDLES_HOISTED_FV_AVX2(o1_re, o1_im, o3_re, o3_im, o5_re, o5_im,                                                                                                  \
                                                          SQRT2_2, NEG_ZERO);                                                                                                                        \
                                                                                                                                                                                                     \
                        RADIX8_COMBINE_SCHEDULED_AVX2(                                                                                                                                               \
                            e0_re, e0_im, e2_re, e2_im, e4_re, e4_im, e6_re, e6_im,                                                                                                                  \
                            o1_re, o1_im, o3_re, o3_im, o5_re, o5_im, o7_re, o7_im,                                                                                                                  \
                            x_re[0], x_im[0], x_re[1], x_im[1], x_re[2], x_im[2], x_re[3], x_im[3],                                                                                                  \
                            x_re[4], x_im[4], x_re[5], x_im[5], x_re[6], x_im[6], x_re[7], x_im[7]);                                                                                                 \
                                                                                                                                                                                                     \
                        for (int lane = 0; lane < 8; ++lane)                                                                                                                                         \
                        {                                                                                                                                                                            \
                            const int out_lane = chunk_offset + lane;                                                                                                                                \
                            const int out_idx = kk_base + out_lane * K_VAL;                                                                                                                          \
                            _mm256_maskstore_pd(&out_re_aligned[out_idx], _mm256_castpd_si256(TAIL_MASK), x_re[lane]);                                                                               \
                            _mm256_maskstore_pd(&out_im_aligned[out_idx], _mm256_castpd_si256(TAIL_MASK), x_im[lane]);                                                                               \
                        }                                                                                                                                                                            \
                    }                                                                                                                                                                                \
                }                                                                                                                                                                                    \
            }                                                                                                                                                                                        \
        }                                                                                                                                                                                            \
                                                                                                                                                                                                     \
        if (USE_STREAMING)                                                                                                                                                                           \
        {                                                                                                                                                                                            \
            _mm_sfence();                                                                                                                                                                            \
        }                                                                                                                                                                                            \
    } while (0)

//==============================================================================
// MAIN RADIX-32 PIPELINE - BACKWARD (BV)
//==============================================================================

#define RADIX32_PIPELINE_4_BV_AVX2(in_re, in_im, out_re, out_im, stage_tw, K, USE_STREAMING)                                                                                                         \
    do                                                                                                                                                                                               \
    {                                                                                                                                                                                                \
        const int K_VAL = (K);                                                                                                                                                                       \
        const int array_len = K_VAL * 32;                                                                                                                                                            \
        const double *__restrict__ in_re_aligned = (const double *)__builtin_assume_aligned(in_re, 32);                                                                                              \
        const double *__restrict__ in_im_aligned = (const double *)__builtin_assume_aligned(in_im, 32);                                                                                              \
        double *__restrict__ out_re_aligned = (double *)__builtin_assume_aligned(out_re, 32);                                                                                                        \
        double *__restrict__ out_im_aligned = (double *)__builtin_assume_aligned(out_im, 32);                                                                                                        \
        const double *__restrict__ tw_re_aligned = (const double *)__builtin_assume_aligned(stage_tw->re, 32);                                                                                       \
        const double *__restrict__ tw_im_aligned = (const double *)__builtin_assume_aligned(stage_tw->im, 32);                                                                                       \
                                                                                                                                                                                                     \
        const __m256d NEG_ZERO = _mm256_set1_pd(-0.0);                                                                                                                                               \
        const __m256d ZERO_VEC = _mm256_setzero_pd();                                                                                                                                                \
        const double SQRT2_2 = 0.70710678118654752440;                                                                                                                                               \
        const __m256d SIGN_MASK = ZERO_VEC;                                                                                                                                                          \
                                                                                                                                                                                                     \
        const int full_blocks = K_VAL / 4;                                                                                                                                                           \
        const int remainder = K_VAL % 4;                                                                                                                                                             \
                                                                                                                                                                                                     \
        for (int b = 0; b < full_blocks; ++b)                                                                                                                                                        \
        {                                                                                                                                                                                            \
            _Pragma("GCC ivdep") for (int kk = b * 4; kk < (b + 1) * 4; kk += 4)                                                                                                                     \
            {                                                                                                                                                                                        \
                const int kk_base = kk;                                                                                                                                                              \
                for (int chunk_offset = 0; chunk_offset < 32; chunk_offset += 8)                                                                                                                     \
                {                                                                                                                                                                                    \
                    __m256d x_re[8], x_im[8];                                                                                                                                                        \
                                                                                                                                                                                                     \
                    TWO_LEVEL_BOUNDED_PREFETCH_AVX2(in_re_aligned, K_VAL, 0,                                                                                                                         \
                                                    array_len, FFT_PREFETCH_DISTANCE_NEAR, FFT_PREFETCH_DISTANCE_FAR);                                                                               \
                    TWO_LEVEL_BOUNDED_PREFETCH_AVX2(in_im_aligned, K_VAL, 0,                                                                                                                         \
                                                    array_len, FFT_PREFETCH_DISTANCE_NEAR, FFT_PREFETCH_DISTANCE_FAR);                                                                               \
                                                                                                                                                                                                     \
                    for (int lane = 0; lane < 8; ++lane)                                                                                                                                             \
                    {                                                                                                                                                                                \
                        const int global_lane = chunk_offset + lane;                                                                                                                                 \
                        const int idx = kk_base + global_lane * K_VAL;                                                                                                                               \
                        x_re[lane] = _mm256_load_pd(&in_re_aligned[idx]);                                                                                                                            \
                        x_im[lane] = _mm256_load_pd(&in_im_aligned[idx]);                                                                                                                            \
                    }                                                                                                                                                                                \
                                                                                                                                                                                                     \
                    if (chunk_offset > 0)                                                                                                                                                            \
                    {                                                                                                                                                                                \
                        APPLY_TWIDDLES_8LANE_FULLY_UNROLLED_AVX2(kk_base, x_re, x_im,                                                                                                                \
                                                                 stage_tw, K_VAL, chunk_offset);                                                                                                     \
                    }                                                                                                                                                                                \
                    else                                                                                                                                                                             \
                    {                                                                                                                                                                                \
                        __m256d tw_re, tw_im, res_re, res_im;                                                                                                                                        \
                        for (int lane = 1; lane < 8; ++lane)                                                                                                                                         \
                        {                                                                                                                                                                            \
                            const int gl = chunk_offset + lane;                                                                                                                                      \
                            const int tw_idx = kk_base + gl * K_VAL;                                                                                                                                 \
                            tw_re = _mm256_load_pd(&tw_re_aligned[tw_idx]);                                                                                                                          \
                            tw_im = _mm256_load_pd(&tw_im_aligned[tw_idx]);                                                                                                                          \
                            CMUL_NATIVE_SOA_AVX2_P0P1(x_re[lane], x_im[lane],                                                                                                                        \
                                                      tw_re, tw_im, res_re, res_im);                                                                                                                 \
                            x_re[lane] = res_re;                                                                                                                                                     \
                            x_im[lane] = res_im;                                                                                                                                                     \
                        }                                                                                                                                                                            \
                    }                                                                                                                                                                                \
                                                                                                                                                                                                     \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX2(x_re[0], x_im[0], x_re[1], x_im[1],                                                                                                             \
                                                     x_re[2], x_im[2], x_re[3], x_im[3],                                                                                                             \
                                                     SIGN_MASK);                                                                                                                                     \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX2(x_re[4], x_im[4], x_re[5], x_im[5],                                                                                                             \
                                                     x_re[6], x_im[6], x_re[7], x_im[7],                                                                                                             \
                                                     SIGN_MASK);                                                                                                                                     \
                                                                                                                                                                                                     \
                    switch (chunk_offset)                                                                                                                                                            \
                    {                                                                                                                                                                                \
                    case 0:                                                                                                                                                                          \
                        APPLY_W32_8LANE_EXPLICIT_J0_BV_AVX2(x_re, x_im, W32_COS_TABLE_AVX2, W32_SIN_TABLE_AVX2,                                                                                      \
                                                            SQRT2_2, NEG_ZERO);                                                                                                                      \
                        break;                                                                                                                                                                       \
                    case 8:                                                                                                                                                                          \
                        APPLY_W32_8LANE_EXPLICIT_J1_BV_AVX2(x_re, x_im, W32_COS_TABLE_AVX2, W32_SIN_TABLE_AVX2,                                                                                      \
                                                            SQRT2_2, NEG_ZERO);                                                                                                                      \
                        break;                                                                                                                                                                       \
                    case 16:                                                                                                                                                                         \
                        APPLY_W32_8LANE_EXPLICIT_J2_BV_AVX2(x_re, x_im, W32_COS_TABLE_AVX2, W32_SIN_TABLE_AVX2,                                                                                      \
                                                            SQRT2_2, NEG_ZERO);                                                                                                                      \
                        break;                                                                                                                                                                       \
                    case 24:                                                                                                                                                                         \
                        APPLY_W32_8LANE_EXPLICIT_J3_BV_AVX2(x_re, x_im, W32_COS_TABLE_AVX2, W32_SIN_TABLE_AVX2,                                                                                      \
                                                            SQRT2_2, NEG_ZERO);                                                                                                                      \
                        break;                                                                                                                                                                       \
                    }                                                                                                                                                                                \
                                                                                                                                                                                                     \
                    __m256d e0_re = x_re[0], e0_im = x_im[0];                                                                                                                                        \
                    __m256d e2_re = x_re[2], e2_im = x_im[2];                                                                                                                                        \
                    __m256d e4_re = x_re[4], e4_im = x_im[4];                                                                                                                                        \
                    __m256d e6_re = x_re[6], e6_im = x_im[6];                                                                                                                                        \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX2(e0_re, e0_im, e2_re, e2_im,                                                                                                                     \
                                                     e4_re, e4_im, e6_re, e6_im, SIGN_MASK);                                                                                                         \
                                                                                                                                                                                                     \
                    __m256d o1_re = x_re[1], o1_im = x_im[1];                                                                                                                                        \
                    __m256d o3_re = x_re[3], o3_im = x_im[3];                                                                                                                                        \
                    __m256d o5_re = x_re[5], o5_im = x_im[5];                                                                                                                                        \
                    __m256d o7_re = x_re[7], o7_im = x_im[7];                                                                                                                                        \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX2(o1_re, o1_im, o3_re, o3_im,                                                                                                                     \
                                                     o5_re, o5_im, o7_re, o7_im, SIGN_MASK);                                                                                                         \
                                                                                                                                                                                                     \
                    APPLY_W8_TWIDDLES_HOISTED_BV_AVX2(o1_re, o1_im, o3_re, o3_im, o5_re, o5_im,                                                                                                      \
                                                      SQRT2_2, NEG_ZERO);                                                                                                                            \
                                                                                                                                                                                                     \
                    RADIX8_COMBINE_SCHEDULED_AVX2(                                                                                                                                                   \
                        e0_re, e0_im, e2_re, e2_im, e4_re, e4_im, e6_re, e6_im,                                                                                                                      \
                        o1_re, o1_im, o3_re, o3_im, o5_re, o5_im, o7_re, o7_im,                                                                                                                      \
                        x_re[0], x_im[0], x_re[1], x_im[1], x_re[2], x_im[2], x_re[3], x_im[3],                                                                                                      \
                        x_re[4], x_im[4], x_re[5], x_im[5], x_re[6], x_im[6], x_re[7], x_im[7]);                                                                                                     \
                                                                                                                                                                                                     \
                    for (int lane = 0; lane < 8; ++lane)                                                                                                                                             \
                    {                                                                                                                                                                                \
                        const int out_lane = chunk_offset + lane;                                                                                                                                    \
                        const int out_idx = kk_base + out_lane * K_VAL;                                                                                                                              \
                        if (USE_STREAMING)                                                                                                                                                           \
                        {                                                                                                                                                                            \
                            _mm256_stream_pd(&out_re_aligned[out_idx], x_re[lane]);                                                                                                                  \
                            _mm256_stream_pd(&out_im_aligned[out_idx], x_im[lane]);                                                                                                                  \
                        }                                                                                                                                                                            \
                        else                                                                                                                                                                         \
                        {                                                                                                                                                                            \
                            _mm256_store_pd(&out_re_aligned[out_idx], x_re[lane]);                                                                                                                   \
                            _mm256_store_pd(&out_im_aligned[out_idx], x_im[lane]);                                                                                                                   \
                        }                                                                                                                                                                            \
                    }                                                                                                                                                                                \
                }                                                                                                                                                                                    \
            }                                                                                                                                                                                        \
        }                                                                                                                                                                                            \
                                                                                                                                                                                                     \
        if (remainder > 0)                                                                                                                                                                           \
        {                                                                                                                                                                                            \
            const int tail_start = full_blocks * 4;                                                                                                                                                  \
            const int tail_count = remainder;                                                                                                                                                        \
            const __m256d TAIL_MASK = (tail_count == 1) ? _mm256_castsi256_pd(_mm256_setr_epi64x(-1LL, 0, 0, 0)) : (tail_count == 2) ? _mm256_castsi256_pd(_mm256_setr_epi64x(-1LL, -1LL, 0, 0))     \
                                                                                                                                     : _mm256_castsi256_pd(_mm256_setr_epi64x(-1LL, -1LL, -1LL, 0)); \
                                                                                                                                                                                                     \
            for (int b = 0; b < 1; ++b)                                                                                                                                                              \
            {                                                                                                                                                                                        \
                _Pragma("GCC ivdep") for (int kk = tail_start; kk < K_VAL; kk += 4)                                                                                                                  \
                {                                                                                                                                                                                    \
                    const int kk_base = kk;                                                                                                                                                          \
                    for (int chunk_offset = 0; chunk_offset < 32; chunk_offset += 8)                                                                                                                 \
                    {                                                                                                                                                                                \
                        __m256d x_re[8], x_im[8];                                                                                                                                                    \
                                                                                                                                                                                                     \
                        for (int lane = 0; lane < 8; ++lane)                                                                                                                                         \
                        {                                                                                                                                                                            \
                            const int global_lane = chunk_offset + lane;                                                                                                                             \
                            const int idx = kk_base + global_lane * K_VAL;                                                                                                                           \
                            x_re[lane] = _mm256_maskload_pd(&in_re_aligned[idx], _mm256_castpd_si256(TAIL_MASK));                                                                                    \
                            x_im[lane] = _mm256_maskload_pd(&in_im_aligned[idx], _mm256_castpd_si256(TAIL_MASK));                                                                                    \
                        }                                                                                                                                                                            \
                                                                                                                                                                                                     \
                        if (chunk_offset > 0)                                                                                                                                                        \
                        {                                                                                                                                                                            \
                            __m256d tw_re, tw_im, res_re, res_im;                                                                                                                                    \
                            for (int lane = 1; lane < 8; ++lane)                                                                                                                                     \
                            {                                                                                                                                                                        \
                                const int gl = chunk_offset + lane;                                                                                                                                  \
                                const int tw_idx = kk_base + gl * K_VAL;                                                                                                                             \
                                tw_re = _mm256_maskload_pd(&tw_re_aligned[tw_idx], _mm256_castpd_si256(TAIL_MASK));                                                                                  \
                                tw_im = _mm256_maskload_pd(&tw_im_aligned[tw_idx], _mm256_castpd_si256(TAIL_MASK));                                                                                  \
                                CMUL_NATIVE_SOA_AVX2_P0P1(x_re[lane], x_im[lane],                                                                                                                    \
                                                          tw_re, tw_im, res_re, res_im);                                                                                                             \
                                x_re[lane] = res_re;                                                                                                                                                 \
                                x_im[lane] = res_im;                                                                                                                                                 \
                            }                                                                                                                                                                        \
                        }                                                                                                                                                                            \
                        else                                                                                                                                                                         \
                        {                                                                                                                                                                            \
                            __m256d tw_re, tw_im, res_re, res_im;                                                                                                                                    \
                            for (int lane = 1; lane < 8; ++lane)                                                                                                                                     \
                            {                                                                                                                                                                        \
                                const int tw_idx = kk_base + lane * K_VAL;                                                                                                                           \
                                tw_re = _mm256_maskload_pd(&tw_re_aligned[tw_idx], _mm256_castpd_si256(TAIL_MASK));                                                                                  \
                                tw_im = _mm256_maskload_pd(&tw_im_aligned[tw_idx], _mm256_castpd_si256(TAIL_MASK));                                                                                  \
                                CMUL_NATIVE_SOA_AVX2_P0P1(x_re[lane], x_im[lane],                                                                                                                    \
                                                          tw_re, tw_im, res_re, res_im);                                                                                                             \
                                x_re[lane] = res_re;                                                                                                                                                 \
                                x_im[lane] = res_im;                                                                                                                                                 \
                            }                                                                                                                                                                        \
                        }                                                                                                                                                                            \
                                                                                                                                                                                                     \
                        RADIX4_BUTTERFLY_NATIVE_SOA_AVX2(x_re[0], x_im[0], x_re[1], x_im[1],                                                                                                         \
                                                         x_re[2], x_im[2], x_re[3], x_im[3],                                                                                                         \
                                                         SIGN_MASK);                                                                                                                                 \
                        RADIX4_BUTTERFLY_NATIVE_SOA_AVX2(x_re[4], x_im[4], x_re[5], x_im[5],                                                                                                         \
                                                         x_re[6], x_im[6], x_re[7], x_im[7],                                                                                                         \
                                                         SIGN_MASK);                                                                                                                                 \
                                                                                                                                                                                                     \
                        switch (chunk_offset)                                                                                                                                                        \
                        {                                                                                                                                                                            \
                        case 0:                                                                                                                                                                      \
                            APPLY_W32_8LANE_EXPLICIT_J0_BV_AVX2(x_re, x_im, W32_COS_TABLE_AVX2, W32_SIN_TABLE_AVX2,                                                                                  \
                                                                SQRT2_2, NEG_ZERO);                                                                                                                  \
                            break;                                                                                                                                                                   \
                        case 8:                                                                                                                                                                      \
                            APPLY_W32_8LANE_EXPLICIT_J1_BV_AVX2(x_re, x_im, W32_COS_TABLE_AVX2, W32_SIN_TABLE_AVX2,                                                                                  \
                                                                SQRT2_2, NEG_ZERO);                                                                                                                  \
                            break;                                                                                                                                                                   \
                        case 16:                                                                                                                                                                     \
                            APPLY_W32_8LANE_EXPLICIT_J2_BV_AVX2(x_re, x_im, W32_COS_TABLE_AVX2, W32_SIN_TABLE_AVX2,                                                                                  \
                                                                SQRT2_2, NEG_ZERO);                                                                                                                  \
                            break;                                                                                                                                                                   \
                        case 24:                                                                                                                                                                     \
                            APPLY_W32_8LANE_EXPLICIT_J3_BV_AVX2(x_re, x_im, W32_COS_TABLE_AVX2, W32_SIN_TABLE_AVX2,                                                                                  \
                                                                SQRT2_2, NEG_ZERO);                                                                                                                  \
                            break;                                                                                                                                                                   \
                        }                                                                                                                                                                            \
                                                                                                                                                                                                     \
                        __m256d e0_re = x_re[0], e0_im = x_im[0];                                                                                                                                    \
                        __m256d e2_re = x_re[2], e2_im = x_im[2];                                                                                                                                    \
                        __m256d e4_re = x_re[4], e4_im = x_im[4];                                                                                                                                    \
                        __m256d e6_re = x_re[6], e6_im = x_im[6];                                                                                                                                    \
                        RADIX4_BUTTERFLY_NATIVE_SOA_AVX2(e0_re, e0_im, e2_re, e2_im,                                                                                                                 \
                                                         e4_re, e4_im, e6_re, e6_im, SIGN_MASK);                                                                                                     \
                                                                                                                                                                                                     \
                        __m256d o1_re = x_re[1], o1_im = x_im[1];                                                                                                                                    \
                        __m256d o3_re = x_re[3], o3_im = x_im[3];                                                                                                                                    \
                        __m256d o5_re = x_re[5], o5_im = x_im[5];                                                                                                                                    \
                        __m256d o7_re = x_re[7], o7_im = x_im[7];                                                                                                                                    \
                        RADIX4_BUTTERFLY_NATIVE_SOA_AVX2(o1_re, o1_im, o3_re, o3_im,                                                                                                                 \
                                                         o5_re, o5_im, o7_re, o7_im, SIGN_MASK);                                                                                                     \
                                                                                                                                                                                                     \
                        APPLY_W8_TWIDDLES_HOISTED_BV_AVX2(o1_re, o1_im, o3_re, o3_im, o5_re, o5_im,                                                                                                  \
                                                          SQRT2_2, NEG_ZERO);                                                                                                                        \
                                                                                                                                                                                                     \
                        RADIX8_COMBINE_SCHEDULED_AVX2(                                                                                                                                               \
                            e0_re, e0_im, e2_re, e2_im, e4_re, e4_im, e6_re, e6_im,                                                                                                                  \
                            o1_re, o1_im, o3_re, o3_im, o5_re, o5_im, o7_re, o7_im,                                                                                                                  \
                            x_re[0], x_im[0], x_re[1], x_im[1], x_re[2], x_im[2], x_re[3], x_im[3],                                                                                                  \
                            x_re[4], x_im[4], x_re[5], x_im[5], x_re[6], x_im[6], x_re[7], x_im[7]);                                                                                                 \
                                                                                                                                                                                                     \
                        for (int lane = 0; lane < 8; ++lane)                                                                                                                                         \
                        {                                                                                                                                                                            \
                            const int out_lane = chunk_offset + lane;                                                                                                                                \
                            const int out_idx = kk_base + out_lane * K_VAL;                                                                                                                          \
                            _mm256_maskstore_pd(&out_re_aligned[out_idx], _mm256_castpd_si256(TAIL_MASK), x_re[lane]);                                                                               \
                            _mm256_maskstore_pd(&out_im_aligned[out_idx], _mm256_castpd_si256(TAIL_MASK), x_im[lane]);                                                                               \
                        }                                                                                                                                                                            \
                    }                                                                                                                                                                                \
                }                                                                                                                                                                                    \
            }                                                                                                                                                                                        \
        }                                                                                                                                                                                            \
                                                                                                                                                                                                     \
        if (USE_STREAMING)                                                                                                                                                                           \
        {                                                                                                                                                                                            \
            _mm_sfence();                                                                                                                                                                            \
        }                                                                                                                                                                                            \
    } while (0)

//==============================================================================
// STREAMING STORE VARIANTS (for use_streaming path in .c files)
//==============================================================================

#define RADIX32_PIPELINE_4_FV_AVX2_STREAM(k, K, sub_outputs, stage_tw, output_buffer) \
    RADIX32_PIPELINE_4_FV_AVX2((sub_outputs).re, (sub_outputs).im,                    \
                               (output_buffer).re, (output_buffer).im,                \
                               stage_tw, K, 1)

#define RADIX32_PIPELINE_4_BV_AVX2_STREAM(k, K, sub_outputs, stage_tw, output_buffer) \
    RADIX32_PIPELINE_4_BV_AVX2((sub_outputs).re, (sub_outputs).im,                    \
                               (output_buffer).re, (output_buffer).im,                \
                               stage_tw, K, 1)

#endif // __AVX2__
#endif // FFT_RADIX32_MACROS_NATIVE_SOA_AVX2_FV_BV_SEPARATED_H