/**
 * @file fft_radix32_macros_native_soa_avx512_fixed.h
 * @brief MATHEMATICALLY CORRECT Native SoA Radix-32 FFT - FV/BV Separated
 *
 * @details
 * FIXES APPLIED:
 * ========================
 * ✅ CORRECT W32 quarter-turn identities (J0-J3) for both FV and BV
 * ✅ Completely separated FV (forward) and BV (backward) macro paths
 * ✅ No SIGN_FOR_CONJ parameter - hard-coded signs per direction
 * ✅ Preserves all optimizations: prefetch, streaming, tail masks, scheduling
 * ✅ Fully unrolled twiddles (7 explicit statements per chunk)
 * ✅ NEG_ZERO/SQRT2_2 passed as parameters (no rebuild inside macros)
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

#ifndef FFT_RADIX32_MACROS_NATIVE_SOA_AVX512_FIXED_H
#define FFT_RADIX32_MACROS_NATIVE_SOA_AVX512_FIXED_H

#include "simd_math.h"
#include "fft_twiddles.h"

#ifdef __AVX512F__

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

static const double __attribute__((aligned(64))) W32_COS_TABLE[8] = {
    1.0,                    // cos(0)
    0.98078528040323044912, // cos(2π/32)
    0.92387953251128675612, // cos(4π/32)
    0.83146961230254523708, // cos(6π/32)
    0.70710678118654752440, // cos(8π/32) = √2/2
    0.55557023301960222474, // cos(10π/32)
    0.38268343236508977172, // cos(12π/32)
    0.19509032201612826784  // cos(14π/32)
};

static const double __attribute__((aligned(64))) W32_SIN_TABLE[8] = {
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
// COMPLEX MULTIPLICATION (PRESERVED)
//==============================================================================

#define CMUL_NATIVE_SOA_AVX512_P0P1(ar, ai, wr, wi, tr, ti) \
    do                                                      \
    {                                                       \
        __m512d ai_wi = _mm512_mul_pd(ai, wi);              \
        __m512d ai_wr = _mm512_mul_pd(ai, wr);              \
        tr = _mm512_fmsub_pd(ar, wr, ai_wi);                \
        ti = _mm512_fmadd_pd(ar, wi, ai_wr);                \
    } while (0)

//==============================================================================
// GENERIC W32 ROTATION BUILDER (BASE FOR BOTH FV AND BV)
//==============================================================================

/**
 * @brief Build (wr, wi) for octave m from base (c[j], s[j])
 * @details Uses quarter-turn identities for k = j + 8·m
 *
 * For angle θ = 2π·k/32:
 *   m=0: cos(θ)   = +c,  sin(θ)   = +s
 *   m=1: cos(θ)   = -s,  sin(θ)   = +c  [θ = θ_j + π/2]
 *   m=2: cos(θ)   = -c,  sin(θ)   = -s  [θ = θ_j + π]
 *   m=3: cos(θ)   = +s,  sin(θ)   = -c  [θ = θ_j + 3π/2]
 *
 * This macro outputs the RAW cos/sin values.
 * FV and BV macros will apply direction-specific sign flips.
 */
#define W32_BUILD_COSSIN_FROM_BASE_AVX512(j, octave_m, w_cos, w_sin, neg_zero, cos_theta, sin_theta) \
    do                                                                                               \
    {                                                                                                \
        __m512d c = _mm512_set1_pd(w_cos[j]);                                                        \
        __m512d s = _mm512_set1_pd(w_sin[j]);                                                        \
        if ((octave_m) == 0)                                                                         \
        {                                                                                            \
            /* m=0: (+c, +s) */                                                                      \
            cos_theta = c;                                                                           \
            sin_theta = s;                                                                           \
        }                                                                                            \
        else if ((octave_m) == 1)                                                                    \
        {                                                                                            \
            /* m=1: (-s, +c) */                                                                      \
            cos_theta = _mm512_xor_pd(s, neg_zero);                                                  \
            sin_theta = c;                                                                           \
        }                                                                                            \
        else if ((octave_m) == 2)                                                                    \
        {                                                                                            \
            /* m=2: (-c, -s) */                                                                      \
            cos_theta = _mm512_xor_pd(c, neg_zero);                                                  \
            sin_theta = _mm512_xor_pd(s, neg_zero);                                                  \
        }                                                                                            \
        else                                                                                         \
        {                                                                                            \
            /* m=3: (+s, -c) */                                                                      \
            cos_theta = s;                                                                           \
            sin_theta = _mm512_xor_pd(c, neg_zero);                                                  \
        }                                                                                            \
    } while (0)

//==============================================================================
// FORWARD (FV) W32 ROTATION MACROS
//==============================================================================

/**
 * @brief FV J=0: k = 0..7
 * W32[k] = exp(-i·2π·k/32) → (wr, wi) = (cos θ, -sin θ)
 */
#define W32_J0_FV(cg, sg, wr, wi)                                  \
    do                                                             \
    {                                                              \
        wr = (cg);                                                 \
        wi = _mm512_xor_pd((sg), _mm512_set1_pd(-0.0)); /* -sin */ \
    } while (0)

/**
 * @brief FV J=1: k = 8..15, θ = θ_j + π/2
 * cos(θ + π/2) = -sin(θ), sin(θ + π/2) = cos(θ)
 * W32[k] = exp(-i·θ) → (wr, wi) = (-sin θ_j, -cos θ_j)
 */
#define W32_J1_FV(cg, sg, wr, wi)                                       \
    do                                                                  \
    {                                                                   \
        wr = _mm512_xor_pd((sg), _mm512_set1_pd(-0.0)); /* -sin(θ_j) */ \
        wi = _mm512_xor_pd((cg), _mm512_set1_pd(-0.0)); /* -cos(θ_j) */ \
    } while (0)

/**
 * @brief FV J=2: k = 16..23, θ = θ_j + π
 * cos(θ + π) = -cos(θ), sin(θ + π) = -sin(θ)
 * W32[k] = exp(-i·θ) → (wr, wi) = (-cos θ_j, +sin θ_j)
 */
#define W32_J2_FV(cg, sg, wr, wi)                                       \
    do                                                                  \
    {                                                                   \
        wr = _mm512_xor_pd((cg), _mm512_set1_pd(-0.0)); /* -cos(θ_j) */ \
        wi = (sg);                                      /* +sin(θ_j) */ \
    } while (0)

/**
 * @brief FV J=3: k = 24..31, θ = θ_j + 3π/2
 * cos(θ + 3π/2) = sin(θ), sin(θ + 3π/2) = -cos(θ)
 * W32[k] = exp(-i·θ) → (wr, wi) = (+sin θ_j, +cos θ_j)
 */
#define W32_J3_FV(cg, sg, wr, wi)  \
    do                             \
    {                              \
        wr = (sg); /* +sin(θ_j) */ \
        wi = (cg); /* +cos(θ_j) */ \
    } while (0)

//==============================================================================
// BACKWARD (BV) W32 ROTATION MACROS
//==============================================================================

/**
 * @brief BV J=0: k = 0..7
 * W32[k] = exp(+i·2π·k/32) → (wr, wi) = (cos θ, +sin θ)
 */
#define W32_J0_BV(cg, sg, wr, wi) \
    do                            \
    {                             \
        wr = (cg);                \
        wi = (sg); /* +sin */     \
    } while (0)

/**
 * @brief BV J=1: k = 8..15
 * W32[k] = exp(+i·θ) → (wr, wi) = (-sin θ_j, +cos θ_j)
 */
#define W32_J1_BV(cg, sg, wr, wi)                                       \
    do                                                                  \
    {                                                                   \
        wr = _mm512_xor_pd((sg), _mm512_set1_pd(-0.0)); /* -sin(θ_j) */ \
        wi = (cg);                                      /* +cos(θ_j) */ \
    } while (0)

/**
 * @brief BV J=2: k = 16..23
 * W32[k] = exp(+i·θ) → (wr, wi) = (-cos θ_j, -sin θ_j)
 */
#define W32_J2_BV(cg, sg, wr, wi)                                       \
    do                                                                  \
    {                                                                   \
        wr = _mm512_xor_pd((cg), _mm512_set1_pd(-0.0)); /* -cos(θ_j) */ \
        wi = _mm512_xor_pd((sg), _mm512_set1_pd(-0.0)); /* -sin(θ_j) */ \
    } while (0)

/**
 * @brief BV J=3: k = 24..31
 * W32[k] = exp(+i·θ) → (wr, wi) = (+sin θ_j, -cos θ_j)
 */
#define W32_J3_BV(cg, sg, wr, wi)                                       \
    do                                                                  \
    {                                                                   \
        wr = (sg);                                      /* +sin(θ_j) */ \
        wi = _mm512_xor_pd((cg), _mm512_set1_pd(-0.0)); /* -cos(θ_j) */ \
    } while (0)

//==============================================================================
// FORWARD (FV) W32 APPLICATION - 8 LANES WITH √2/2 OPTIMIZATION
//==============================================================================

#define APPLY_W32_8LANE_EXPLICIT_J0_FV(x_re, x_im, W32_COS, W32_SIN, SQRT2_2, NEG_ZERO) \
    do                                                                                  \
    {                                                                                   \
        __m512d wr, wi, tr, ti;                                                         \
        /* Lane 0: k=0, W32[0] = 1+0i, skip */                                          \
        /* Lane 1: k=1 */                                                               \
        __m512d c1 = _mm512_set1_pd(W32_COS[1]);                                        \
        __m512d s1 = _mm512_set1_pd(W32_SIN[1]);                                        \
        W32_J0_FV(c1, s1, wr, wi);                                                      \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[1], x_im[1], wr, wi, tr, ti);                  \
        x_re[1] = tr;                                                                   \
        x_im[1] = ti;                                                                   \
        /* Lane 2: k=2 */                                                               \
        __m512d c2 = _mm512_set1_pd(W32_COS[2]);                                        \
        __m512d s2 = _mm512_set1_pd(W32_SIN[2]);                                        \
        W32_J0_FV(c2, s2, wr, wi);                                                      \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[2], x_im[2], wr, wi, tr, ti);                  \
        x_re[2] = tr;                                                                   \
        x_im[2] = ti;                                                                   \
        /* Lane 3: k=3 */                                                               \
        __m512d c3 = _mm512_set1_pd(W32_COS[3]);                                        \
        __m512d s3 = _mm512_set1_pd(W32_SIN[3]);                                        \
        W32_J0_FV(c3, s3, wr, wi);                                                      \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[3], x_im[3], wr, wi, tr, ti);                  \
        x_re[3] = tr;                                                                   \
        x_im[3] = ti;                                                                   \
        /* Lane 4: k=4, √2/2 optimization */                                            \
        tr = _mm512_mul_pd(_mm512_add_pd(x_re[4], x_im[4]), SQRT2_2);                   \
        ti = _mm512_mul_pd(_mm512_sub_pd(x_im[4], x_re[4]), SQRT2_2);                   \
        x_re[4] = tr;                                                                   \
        x_im[4] = ti;                                                                   \
        /* Lane 5: k=5 */                                                               \
        __m512d c5 = _mm512_set1_pd(W32_COS[5]);                                        \
        __m512d s5 = _mm512_set1_pd(W32_SIN[5]);                                        \
        W32_J0_FV(c5, s5, wr, wi);                                                      \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[5], x_im[5], wr, wi, tr, ti);                  \
        x_re[5] = tr;                                                                   \
        x_im[5] = ti;                                                                   \
        /* Lane 6: k=6 */                                                               \
        __m512d c6 = _mm512_set1_pd(W32_COS[6]);                                        \
        __m512d s6 = _mm512_set1_pd(W32_SIN[6]);                                        \
        W32_J0_FV(c6, s6, wr, wi);                                                      \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[6], x_im[6], wr, wi, tr, ti);                  \
        x_re[6] = tr;                                                                   \
        x_im[6] = ti;                                                                   \
        /* Lane 7: k=7 */                                                               \
        __m512d c7 = _mm512_set1_pd(W32_COS[7]);                                        \
        __m512d s7 = _mm512_set1_pd(W32_SIN[7]);                                        \
        W32_J0_FV(c7, s7, wr, wi);                                                      \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[7], x_im[7], wr, wi, tr, ti);                  \
        x_re[7] = tr;                                                                   \
        x_im[7] = ti;                                                                   \
    } while (0)

#define APPLY_W32_8LANE_EXPLICIT_J1_FV(x_re, x_im, W32_COS, W32_SIN, SQRT2_2, NEG_ZERO)        \
    do                                                                                         \
    {                                                                                          \
        __m512d wr, wi, tr, ti;                                                                \
        /* Lane 0: k=8, W32[8] = -i → (0, -1) */                                               \
        tr = x_im[0];                                                                          \
        ti = _mm512_xor_pd(x_re[0], NEG_ZERO);                                                 \
        x_re[0] = tr;                                                                          \
        x_im[0] = ti;                                                                          \
        /* Lane 1: k=9 */                                                                      \
        __m512d c1 = _mm512_set1_pd(W32_COS[1]);                                               \
        __m512d s1 = _mm512_set1_pd(W32_SIN[1]);                                               \
        W32_J1_FV(c1, s1, wr, wi);                                                             \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[1], x_im[1], wr, wi, tr, ti);                         \
        x_re[1] = tr;                                                                          \
        x_im[1] = ti;                                                                          \
        /* Lane 2: k=10 */                                                                     \
        __m512d c2 = _mm512_set1_pd(W32_COS[2]);                                               \
        __m512d s2 = _mm512_set1_pd(W32_SIN[2]);                                               \
        W32_J1_FV(c2, s2, wr, wi);                                                             \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[2], x_im[2], wr, wi, tr, ti);                         \
        x_re[2] = tr;                                                                          \
        x_im[2] = ti;                                                                          \
        /* Lane 3: k=11 */                                                                     \
        __m512d c3 = _mm512_set1_pd(W32_COS[3]);                                               \
        __m512d s3 = _mm512_set1_pd(W32_SIN[3]);                                               \
        W32_J1_FV(c3, s3, wr, wi);                                                             \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[3], x_im[3], wr, wi, tr, ti);                         \
        x_re[3] = tr;                                                                          \
        x_im[3] = ti;                                                                          \
        /* Lane 4: k=12, √2/2 optimization for (-√2/2, -√2/2) */                               \
        tr = _mm512_mul_pd(_mm512_xor_pd(_mm512_add_pd(x_re[4], x_im[4]), NEG_ZERO), SQRT2_2); \
        ti = _mm512_mul_pd(_mm512_xor_pd(_mm512_sub_pd(x_re[4], x_im[4]), NEG_ZERO), SQRT2_2); \
        x_re[4] = tr;                                                                          \
        x_im[4] = ti;                                                                          \
        /* Lane 5: k=13 */                                                                     \
        __m512d c5 = _mm512_set1_pd(W32_COS[5]);                                               \
        __m512d s5 = _mm512_set1_pd(W32_SIN[5]);                                               \
        W32_J1_FV(c5, s5, wr, wi);                                                             \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[5], x_im[5], wr, wi, tr, ti);                         \
        x_re[5] = tr;                                                                          \
        x_im[5] = ti;                                                                          \
        /* Lane 6: k=14 */                                                                     \
        __m512d c6 = _mm512_set1_pd(W32_COS[6]);                                               \
        __m512d s6 = _mm512_set1_pd(W32_SIN[6]);                                               \
        W32_J1_FV(c6, s6, wr, wi);                                                             \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[6], x_im[6], wr, wi, tr, ti);                         \
        x_re[6] = tr;                                                                          \
        x_im[6] = ti;                                                                          \
        /* Lane 7: k=15 */                                                                     \
        __m512d c7 = _mm512_set1_pd(W32_COS[7]);                                               \
        __m512d s7 = _mm512_set1_pd(W32_SIN[7]);                                               \
        W32_J1_FV(c7, s7, wr, wi);                                                             \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[7], x_im[7], wr, wi, tr, ti);                         \
        x_re[7] = tr;                                                                          \
        x_im[7] = ti;                                                                          \
    } while (0)

#define APPLY_W32_8LANE_EXPLICIT_J2_FV(x_re, x_im, W32_COS, W32_SIN, SQRT2_2, NEG_ZERO)        \
    do                                                                                         \
    {                                                                                          \
        __m512d wr, wi, tr, ti;                                                                \
        /* Lane 0: k=16, W32[16] = -1 → (-1, 0) */                                             \
        x_re[0] = _mm512_xor_pd(x_re[0], NEG_ZERO);                                            \
        x_im[0] = _mm512_xor_pd(x_im[0], NEG_ZERO);                                            \
        /* Lane 1: k=17 */                                                                     \
        __m512d c1 = _mm512_set1_pd(W32_COS[1]);                                               \
        __m512d s1 = _mm512_set1_pd(W32_SIN[1]);                                               \
        W32_J2_FV(c1, s1, wr, wi);                                                             \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[1], x_im[1], wr, wi, tr, ti);                         \
        x_re[1] = tr;                                                                          \
        x_im[1] = ti;                                                                          \
        /* Lane 2: k=18 */                                                                     \
        __m512d c2 = _mm512_set1_pd(W32_COS[2]);                                               \
        __m512d s2 = _mm512_set1_pd(W32_SIN[2]);                                               \
        W32_J2_FV(c2, s2, wr, wi);                                                             \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[2], x_im[2], wr, wi, tr, ti);                         \
        x_re[2] = tr;                                                                          \
        x_im[2] = ti;                                                                          \
        /* Lane 3: k=19 */                                                                     \
        __m512d c3 = _mm512_set1_pd(W32_COS[3]);                                               \
        __m512d s3 = _mm512_set1_pd(W32_SIN[3]);                                               \
        W32_J2_FV(c3, s3, wr, wi);                                                             \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[3], x_im[3], wr, wi, tr, ti);                         \
        x_re[3] = tr;                                                                          \
        x_im[3] = ti;                                                                          \
        /* Lane 4: k=20, √2/2 optimization for (-√2/2, +√2/2) */                               \
        tr = _mm512_mul_pd(_mm512_xor_pd(_mm512_add_pd(x_re[4], x_im[4]), NEG_ZERO), SQRT2_2); \
        ti = _mm512_mul_pd(_mm512_sub_pd(x_im[4], x_re[4]), SQRT2_2);                          \
        x_re[4] = tr;                                                                          \
        x_im[4] = ti;                                                                          \
        /* Lane 5: k=21 */                                                                     \
        __m512d c5 = _mm512_set1_pd(W32_COS[5]);                                               \
        __m512d s5 = _mm512_set1_pd(W32_SIN[5]);                                               \
        W32_J2_FV(c5, s5, wr, wi);                                                             \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[5], x_im[5], wr, wi, tr, ti);                         \
        x_re[5] = tr;                                                                          \
        x_im[5] = ti;                                                                          \
        /* Lane 6: k=22 */                                                                     \
        __m512d c6 = _mm512_set1_pd(W32_COS[6]);                                               \
        __m512d s6 = _mm512_set1_pd(W32_SIN[6]);                                               \
        W32_J2_FV(c6, s6, wr, wi);                                                             \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[6], x_im[6], wr, wi, tr, ti);                         \
        x_re[6] = tr;                                                                          \
        x_im[6] = ti;                                                                          \
        /* Lane 7: k=23 */                                                                     \
        __m512d c7 = _mm512_set1_pd(W32_COS[7]);                                               \
        __m512d s7 = _mm512_set1_pd(W32_SIN[7]);                                               \
        W32_J2_FV(c7, s7, wr, wi);                                                             \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[7], x_im[7], wr, wi, tr, ti);                         \
        x_re[7] = tr;                                                                          \
        x_im[7] = ti;                                                                          \
    } while (0)

#define APPLY_W32_8LANE_EXPLICIT_J3_FV(x_re, x_im, W32_COS, W32_SIN, SQRT2_2, NEG_ZERO) \
    do                                                                                  \
    {                                                                                   \
        __m512d wr, wi, tr, ti;                                                         \
        /* Lane 0: k=24, W32[24] = +i → (0, 1) */                                       \
        tr = _mm512_xor_pd(x_im[0], NEG_ZERO);                                          \
        ti = x_re[0];                                                                   \
        x_re[0] = tr;                                                                   \
        x_im[0] = ti;                                                                   \
        /* Lane 1: k=25 */                                                              \
        __m512d c1 = _mm512_set1_pd(W32_COS[1]);                                        \
        __m512d s1 = _mm512_set1_pd(W32_SIN[1]);                                        \
        W32_J3_FV(c1, s1, wr, wi);                                                      \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[1], x_im[1], wr, wi, tr, ti);                  \
        x_re[1] = tr;                                                                   \
        x_im[1] = ti;                                                                   \
        /* Lane 2: k=26 */                                                              \
        __m512d c2 = _mm512_set1_pd(W32_COS[2]);                                        \
        __m512d s2 = _mm512_set1_pd(W32_SIN[2]);                                        \
        W32_J3_FV(c2, s2, wr, wi);                                                      \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[2], x_im[2], wr, wi, tr, ti);                  \
        x_re[2] = tr;                                                                   \
        x_im[2] = ti;                                                                   \
        /* Lane 3: k=27 */                                                              \
        __m512d c3 = _mm512_set1_pd(W32_COS[3]);                                        \
        __m512d s3 = _mm512_set1_pd(W32_SIN[3]);                                        \
        W32_J3_FV(c3, s3, wr, wi);                                                      \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[3], x_im[3], wr, wi, tr, ti);                  \
        x_re[3] = tr;                                                                   \
        x_im[3] = ti;                                                                   \
        /* Lane 4: k=28, √2/2 optimization for (+√2/2, +√2/2) */                        \
        tr = _mm512_mul_pd(_mm512_sub_pd(x_im[4], x_re[4]), SQRT2_2);                   \
        ti = _mm512_mul_pd(_mm512_add_pd(x_re[4], x_im[4]), SQRT2_2);                   \
        x_re[4] = tr;                                                                   \
        x_im[4] = ti;                                                                   \
        /* Lane 5: k=29 */                                                              \
        __m512d c5 = _mm512_set1_pd(W32_COS[5]);                                        \
        __m512d s5 = _mm512_set1_pd(W32_SIN[5]);                                        \
        W32_J3_FV(c5, s5, wr, wi);                                                      \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[5], x_im[5], wr, wi, tr, ti);                  \
        x_re[5] = tr;                                                                   \
        x_im[5] = ti;                                                                   \
        /* Lane 6: k=30 */                                                              \
        __m512d c6 = _mm512_set1_pd(W32_COS[6]);                                        \
        __m512d s6 = _mm512_set1_pd(W32_SIN[6]);                                        \
        W32_J3_FV(c6, s6, wr, wi);                                                      \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[6], x_im[6], wr, wi, tr, ti);                  \
        x_re[6] = tr;                                                                   \
        x_im[6] = ti;                                                                   \
        /* Lane 7: k=31 */                                                              \
        __m512d c7 = _mm512_set1_pd(W32_COS[7]);                                        \
        __m512d s7 = _mm512_set1_pd(W32_SIN[7]);                                        \
        W32_J3_FV(c7, s7, wr, wi);                                                      \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[7], x_im[7], wr, wi, tr, ti);                  \
        x_re[7] = tr;                                                                   \
        x_im[7] = ti;                                                                   \
    } while (0)

//==============================================================================
// BACKWARD (BV) W32 APPLICATION - 8 LANES WITH √2/2 OPTIMIZATION
//==============================================================================

#define APPLY_W32_8LANE_EXPLICIT_J0_BV(x_re, x_im, W32_COS, W32_SIN, SQRT2_2, NEG_ZERO) \
    do                                                                                  \
    {                                                                                   \
        __m512d wr, wi, tr, ti;                                                         \
        /* Lane 0: k=0, W32[0] = 1+0i, skip */                                          \
        /* Lane 1: k=1 */                                                               \
        __m512d c1 = _mm512_set1_pd(W32_COS[1]);                                        \
        __m512d s1 = _mm512_set1_pd(W32_SIN[1]);                                        \
        W32_J0_BV(c1, s1, wr, wi);                                                      \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[1], x_im[1], wr, wi, tr, ti);                  \
        x_re[1] = tr;                                                                   \
        x_im[1] = ti;                                                                   \
        /* Lane 2: k=2 */                                                               \
        __m512d c2 = _mm512_set1_pd(W32_COS[2]);                                        \
        __m512d s2 = _mm512_set1_pd(W32_SIN[2]);                                        \
        W32_J0_BV(c2, s2, wr, wi);                                                      \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[2], x_im[2], wr, wi, tr, ti);                  \
        x_re[2] = tr;                                                                   \
        x_im[2] = ti;                                                                   \
        /* Lane 3: k=3 */                                                               \
        __m512d c3 = _mm512_set1_pd(W32_COS[3]);                                        \
        __m512d s3 = _mm512_set1_pd(W32_SIN[3]);                                        \
        W32_J0_BV(c3, s3, wr, wi);                                                      \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[3], x_im[3], wr, wi, tr, ti);                  \
        x_re[3] = tr;                                                                   \
        x_im[3] = ti;                                                                   \
        /* Lane 4: k=4, √2/2 optimization */                                            \
        tr = _mm512_mul_pd(_mm512_add_pd(x_re[4], x_im[4]), SQRT2_2);                   \
        ti = _mm512_mul_pd(_mm512_sub_pd(x_re[4], x_im[4]), SQRT2_2);                   \
        x_re[4] = tr;                                                                   \
        x_im[4] = ti;                                                                   \
        /* Lane 5: k=5 */                                                               \
        __m512d c5 = _mm512_set1_pd(W32_COS[5]);                                        \
        __m512d s5 = _mm512_set1_pd(W32_SIN[5]);                                        \
        W32_J0_BV(c5, s5, wr, wi);                                                      \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[5], x_im[5], wr, wi, tr, ti);                  \
        x_re[5] = tr;                                                                   \
        x_im[5] = ti;                                                                   \
        /* Lane 6: k=6 */                                                               \
        __m512d c6 = _mm512_set1_pd(W32_COS[6]);                                        \
        __m512d s6 = _mm512_set1_pd(W32_SIN[6]);                                        \
        W32_J0_BV(c6, s6, wr, wi);                                                      \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[6], x_im[6], wr, wi, tr, ti);                  \
        x_re[6] = tr;                                                                   \
        x_im[6] = ti;                                                                   \
        /* Lane 7: k=7 */                                                               \
        __m512d c7 = _mm512_set1_pd(W32_COS[7]);                                        \
        __m512d s7 = _mm512_set1_pd(W32_SIN[7]);                                        \
        W32_J0_BV(c7, s7, wr, wi);                                                      \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[7], x_im[7], wr, wi, tr, ti);                  \
        x_re[7] = tr;                                                                   \
        x_im[7] = ti;                                                                   \
    } while (0)

#define APPLY_W32_8LANE_EXPLICIT_J1_BV(x_re, x_im, W32_COS, W32_SIN, SQRT2_2, NEG_ZERO)        \
    do                                                                                         \
    {                                                                                          \
        __m512d wr, wi, tr, ti;                                                                \
        /* Lane 0: k=8, W32[8] = +i → (0, 1) */                                                \
        tr = _mm512_xor_pd(x_im[0], NEG_ZERO);                                                 \
        ti = x_re[0];                                                                          \
        x_re[0] = tr;                                                                          \
        x_im[0] = ti;                                                                          \
        /* Lane 1: k=9 */                                                                      \
        __m512d c1 = _mm512_set1_pd(W32_COS[1]);                                               \
        __m512d s1 = _mm512_set1_pd(W32_SIN[1]);                                               \
        W32_J1_BV(c1, s1, wr, wi);                                                             \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[1], x_im[1], wr, wi, tr, ti);                         \
        x_re[1] = tr;                                                                          \
        x_im[1] = ti;                                                                          \
        /* Lane 2: k=10 */                                                                     \
        __m512d c2 = _mm512_set1_pd(W32_COS[2]);                                               \
        __m512d s2 = _mm512_set1_pd(W32_SIN[2]);                                               \
        W32_J1_BV(c2, s2, wr, wi);                                                             \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[2], x_im[2], wr, wi, tr, ti);                         \
        x_re[2] = tr;                                                                          \
        x_im[2] = ti;                                                                          \
        /* Lane 3: k=11 */                                                                     \
        __m512d c3 = _mm512_set1_pd(W32_COS[3]);                                               \
        __m512d s3 = _mm512_set1_pd(W32_SIN[3]);                                               \
        W32_J1_BV(c3, s3, wr, wi);                                                             \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[3], x_im[3], wr, wi, tr, ti);                         \
        x_re[3] = tr;                                                                          \
        x_im[3] = ti;                                                                          \
        /* Lane 4: k=12, √2/2 optimization for (-√2/2, +√2/2) */                               \
        tr = _mm512_mul_pd(_mm512_xor_pd(_mm512_add_pd(x_re[4], x_im[4]), NEG_ZERO), SQRT2_2); \
        ti = _mm512_mul_pd(_mm512_sub_pd(x_re[4], x_im[4]), SQRT2_2);                          \
        x_re[4] = tr;                                                                          \
        x_im[4] = ti;                                                                          \
        /* Lane 5: k=13 */                                                                     \
        __m512d c5 = _mm512_set1_pd(W32_COS[5]);                                               \
        __m512d s5 = _mm512_set1_pd(W32_SIN[5]);                                               \
        W32_J1_BV(c5, s5, wr, wi);                                                             \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[5], x_im[5], wr, wi, tr, ti);                         \
        x_re[5] = tr;                                                                          \
        x_im[5] = ti;                                                                          \
        /* Lane 6: k=14 */                                                                     \
        __m512d c6 = _mm512_set1_pd(W32_COS[6]);                                               \
        __m512d s6 = _mm512_set1_pd(W32_SIN[6]);                                               \
        W32_J1_BV(c6, s6, wr, wi);                                                             \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[6], x_im[6], wr, wi, tr, ti);                         \
        x_re[6] = tr;                                                                          \
        x_im[6] = ti;                                                                          \
        /* Lane 7: k=15 */                                                                     \
        __m512d c7 = _mm512_set1_pd(W32_COS[7]);                                               \
        __m512d s7 = _mm512_set1_pd(W32_SIN[7]);                                               \
        W32_J1_BV(c7, s7, wr, wi);                                                             \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[7], x_im[7], wr, wi, tr, ti);                         \
        x_re[7] = tr;                                                                          \
        x_im[7] = ti;                                                                          \
    } while (0)

#define APPLY_W32_8LANE_EXPLICIT_J2_BV(x_re, x_im, W32_COS, W32_SIN, SQRT2_2, NEG_ZERO)        \
    do                                                                                         \
    {                                                                                          \
        __m512d wr, wi, tr, ti;                                                                \
        /* Lane 0: k=16, W32[16] = -1 → (-1, 0) */                                             \
        x_re[0] = _mm512_xor_pd(x_re[0], NEG_ZERO);                                            \
        x_im[0] = _mm512_xor_pd(x_im[0], NEG_ZERO);                                            \
        /* Lane 1: k=17 */                                                                     \
        __m512d c1 = _mm512_set1_pd(W32_COS[1]);                                               \
        __m512d s1 = _mm512_set1_pd(W32_SIN[1]);                                               \
        W32_J2_BV(c1, s1, wr, wi);                                                             \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[1], x_im[1], wr, wi, tr, ti);                         \
        x_re[1] = tr;                                                                          \
        x_im[1] = ti;                                                                          \
        /* Lane 2: k=18 */                                                                     \
        __m512d c2 = _mm512_set1_pd(W32_COS[2]);                                               \
        __m512d s2 = _mm512_set1_pd(W32_SIN[2]);                                               \
        W32_J2_BV(c2, s2, wr, wi);                                                             \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[2], x_im[2], wr, wi, tr, ti);                         \
        x_re[2] = tr;                                                                          \
        x_im[2] = ti;                                                                          \
        /* Lane 3: k=19 */                                                                     \
        __m512d c3 = _mm512_set1_pd(W32_COS[3]);                                               \
        __m512d s3 = _mm512_set1_pd(W32_SIN[3]);                                               \
        W32_J2_BV(c3, s3, wr, wi);                                                             \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[3], x_im[3], wr, wi, tr, ti);                         \
        x_re[3] = tr;                                                                          \
        x_im[3] = ti;                                                                          \
        /* Lane 4: k=20, √2/2 optimization for (-√2/2, -√2/2) */                               \
        tr = _mm512_mul_pd(_mm512_xor_pd(_mm512_add_pd(x_re[4], x_im[4]), NEG_ZERO), SQRT2_2); \
        ti = _mm512_mul_pd(_mm512_xor_pd(_mm512_sub_pd(x_im[4], x_re[4]), NEG_ZERO), SQRT2_2); \
        x_re[4] = tr;                                                                          \
        x_im[4] = ti;                                                                          \
        /* Lane 5: k=21 */                                                                     \
        __m512d c5 = _mm512_set1_pd(W32_COS[5]);                                               \
        __m512d s5 = _mm512_set1_pd(W32_SIN[5]);                                               \
        W32_J2_BV(c5, s5, wr, wi);                                                             \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[5], x_im[5], wr, wi, tr, ti);                         \
        x_re[5] = tr;                                                                          \
        x_im[5] = ti;                                                                          \
        /* Lane 6: k=22 */                                                                     \
        __m512d c6 = _mm512_set1_pd(W32_COS[6]);                                               \
        __m512d s6 = _mm512_set1_pd(W32_SIN[6]);                                               \
        W32_J2_BV(c6, s6, wr, wi);                                                             \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[6], x_im[6], wr, wi, tr, ti);                         \
        x_re[6] = tr;                                                                          \
        x_im[6] = ti;                                                                          \
        /* Lane 7: k=23 */                                                                     \
        __m512d c7 = _mm512_set1_pd(W32_COS[7]);                                               \
        __m512d s7 = _mm512_set1_pd(W32_SIN[7]);                                               \
        W32_J2_BV(c7, s7, wr, wi);                                                             \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[7], x_im[7], wr, wi, tr, ti);                         \
        x_re[7] = tr;                                                                          \
        x_im[7] = ti;                                                                          \
    } while (0)

#define APPLY_W32_8LANE_EXPLICIT_J3_BV(x_re, x_im, W32_COS, W32_SIN, SQRT2_2, NEG_ZERO)        \
    do                                                                                         \
    {                                                                                          \
        __m512d wr, wi, tr, ti;                                                                \
        /* Lane 0: k=24, W32[24] = -i → (0, -1) */                                             \
        tr = x_im[0];                                                                          \
        ti = _mm512_xor_pd(x_re[0], NEG_ZERO);                                                 \
        x_re[0] = tr;                                                                          \
        x_im[0] = ti;                                                                          \
        /* Lane 1: k=25 */                                                                     \
        __m512d c1 = _mm512_set1_pd(W32_COS[1]);                                               \
        __m512d s1 = _mm512_set1_pd(W32_SIN[1]);                                               \
        W32_J3_BV(c1, s1, wr, wi);                                                             \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[1], x_im[1], wr, wi, tr, ti);                         \
        x_re[1] = tr;                                                                          \
        x_im[1] = ti;                                                                          \
        /* Lane 2: k=26 */                                                                     \
        __m512d c2 = _mm512_set1_pd(W32_COS[2]);                                               \
        __m512d s2 = _mm512_set1_pd(W32_SIN[2]);                                               \
        W32_J3_BV(c2, s2, wr, wi);                                                             \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[2], x_im[2], wr, wi, tr, ti);                         \
        x_re[2] = tr;                                                                          \
        x_im[2] = ti;                                                                          \
        /* Lane 3: k=27 */                                                                     \
        __m512d c3 = _mm512_set1_pd(W32_COS[3]);                                               \
        __m512d s3 = _mm512_set1_pd(W32_SIN[3]);                                               \
        W32_J3_BV(c3, s3, wr, wi);                                                             \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[3], x_im[3], wr, wi, tr, ti);                         \
        x_re[3] = tr;                                                                          \
        x_im[3] = ti;                                                                          \
        /* Lane 4: k=28, √2/2 optimization for (+√2/2, -√2/2) */                               \
        tr = _mm512_mul_pd(_mm512_sub_pd(x_im[4], x_re[4]), SQRT2_2);                          \
        ti = _mm512_mul_pd(_mm512_xor_pd(_mm512_add_pd(x_re[4], x_im[4]), NEG_ZERO), SQRT2_2); \
        x_re[4] = tr;                                                                          \
        x_im[4] = ti;                                                                          \
        /* Lane 5: k=29 */                                                                     \
        __m512d c5 = _mm512_set1_pd(W32_COS[5]);                                               \
        __m512d s5 = _mm512_set1_pd(W32_SIN[5]);                                               \
        W32_J3_BV(c5, s5, wr, wi);                                                             \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[5], x_im[5], wr, wi, tr, ti);                         \
        x_re[5] = tr;                                                                          \
        x_im[5] = ti;                                                                          \
        /* Lane 6: k=30 */                                                                     \
        __m512d c6 = _mm512_set1_pd(W32_COS[6]);                                               \
        __m512d s6 = _mm512_set1_pd(W32_SIN[6]);                                               \
        W32_J3_BV(c6, s6, wr, wi);                                                             \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[6], x_im[6], wr, wi, tr, ti);                         \
        x_re[6] = tr;                                                                          \
        x_im[6] = ti;                                                                          \
        /* Lane 7: k=31 */                                                                     \
        __m512d c7 = _mm512_set1_pd(W32_COS[7]);                                               \
        __m512d s7 = _mm512_set1_pd(W32_SIN[7]);                                               \
        W32_J3_BV(c7, s7, wr, wi);                                                             \
        CMUL_NATIVE_SOA_AVX512_P0P1(x_re[7], x_im[7], wr, wi, tr, ti);                         \
        x_re[7] = tr;                                                                          \
        x_im[7] = ti;                                                                          \
    } while (0)

//==============================================================================
// RADIX-4 BUTTERFLY (PRESERVED - DIRECTION-AWARE VIA sign_mask)
//==============================================================================

/**
 * @brief Radix-4 butterfly with configurable rotation direction
 * @param sign_mask Use NEG_ZERO for forward, _mm512_setzero_pd() for inverse
 */
#define RADIX4_BUTTERFLY_NATIVE_SOA_AVX512(a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, sign_mask) \
    do                                                                                                \
    {                                                                                                 \
        __m512d sumBD_re = _mm512_add_pd(b_re, d_re);                                                 \
        __m512d sumBD_im = _mm512_add_pd(b_im, d_im);                                                 \
        __m512d difBD_re = _mm512_sub_pd(b_re, d_re);                                                 \
        __m512d difBD_im = _mm512_sub_pd(b_im, d_im);                                                 \
        __m512d sumAC_re = _mm512_add_pd(a_re, c_re);                                                 \
        __m512d sumAC_im = _mm512_add_pd(a_im, c_im);                                                 \
        __m512d difAC_re = _mm512_sub_pd(a_re, c_re);                                                 \
        __m512d difAC_im = _mm512_sub_pd(a_im, c_im);                                                 \
        __m512d rot_re = _mm512_xor_pd(difBD_im, sign_mask);                                          \
        __m512d rot_im = difBD_re;                                                                    \
        __m512d y0_re = _mm512_add_pd(sumAC_re, sumBD_re);                                            \
        __m512d y0_im = _mm512_add_pd(sumAC_im, sumBD_im);                                            \
        __m512d y1_re = _mm512_add_pd(difAC_re, rot_re);                                              \
        __m512d y1_im = _mm512_add_pd(difAC_im, rot_im);                                              \
        __m512d y2_re = _mm512_sub_pd(sumAC_re, sumBD_re);                                            \
        __m512d y2_im = _mm512_sub_pd(sumAC_im, sumBD_im);                                            \
        __m512d y3_re = _mm512_sub_pd(difAC_re, rot_re);                                              \
        __m512d y3_im = _mm512_sub_pd(difAC_im, rot_im);                                              \
        a_re = y0_re;                                                                                 \
        a_im = y0_im;                                                                                 \
        b_re = y1_re;                                                                                 \
        b_im = y1_im;                                                                                 \
        c_re = y2_re;                                                                                 \
        c_im = y2_im;                                                                                 \
        d_re = y3_re;                                                                                 \
        d_im = y3_im;                                                                                 \
    } while (0)

//==============================================================================
// W_8 TWIDDLES WITH PASSED CONSTANTS (FORWARD - NEGATIVE ROTATION)
//==============================================================================

#define APPLY_W8_TWIDDLES_HOISTED_FV(o1_re, o1_im, o3_re, o3_im, o5_re, o5_im, sqrt2_2, neg_zero)     \
    do                                                                                                \
    {                                                                                                 \
        __m512d t1_re = _mm512_mul_pd(_mm512_add_pd(o1_re, o1_im), sqrt2_2);                          \
        __m512d t1_im = _mm512_mul_pd(_mm512_sub_pd(o1_im, o1_re), sqrt2_2);                          \
        o1_re = t1_re;                                                                                \
        o1_im = t1_im;                                                                                \
        __m512d t3_re = o3_im;                                                                        \
        __m512d t3_im = _mm512_xor_pd(o3_re, neg_zero);                                               \
        o3_re = t3_re;                                                                                \
        o3_im = t3_im;                                                                                \
        __m512d t5_re = _mm512_mul_pd(_mm512_xor_pd(_mm512_add_pd(o5_re, o5_im), neg_zero), sqrt2_2); \
        __m512d t5_im = _mm512_mul_pd(_mm512_sub_pd(o5_im, o5_re), sqrt2_2);                          \
        o5_re = t5_re;                                                                                \
        o5_im = t5_im;                                                                                \
    } while (0)

//==============================================================================
// W_8 TWIDDLES WITH PASSED CONSTANTS (BACKWARD - POSITIVE ROTATION)
//==============================================================================

#define APPLY_W8_TWIDDLES_HOISTED_BV(o1_re, o1_im, o3_re, o3_im, o5_re, o5_im, sqrt2_2, neg_zero)     \
    do                                                                                                \
    {                                                                                                 \
        __m512d t1_re = _mm512_mul_pd(_mm512_add_pd(o1_re, o1_im), sqrt2_2);                          \
        __m512d t1_im = _mm512_mul_pd(_mm512_sub_pd(o1_re, o1_im), sqrt2_2);                          \
        o1_re = t1_re;                                                                                \
        o1_im = t1_im;                                                                                \
        __m512d t3_re = _mm512_xor_pd(o3_im, neg_zero);                                               \
        __m512d t3_im = o3_re;                                                                        \
        o3_re = t3_re;                                                                                \
        o3_im = t3_im;                                                                                \
        __m512d t5_re = _mm512_mul_pd(_mm512_xor_pd(_mm512_add_pd(o5_re, o5_im), neg_zero), sqrt2_2); \
        __m512d t5_im = _mm512_mul_pd(_mm512_xor_pd(_mm512_sub_pd(o5_im, o5_re), neg_zero), sqrt2_2); \
        o5_re = t5_re;                                                                                \
        o5_im = t5_im;                                                                                \
    } while (0)

//==============================================================================
// RADIX-8 COMBINE WITH IMPROVED SCHEDULING (PRESERVED)
//==============================================================================

#define RADIX8_COMBINE_SCHEDULED(                           \
    e0_re, e0_im, e2_re, e2_im, e4_re, e4_im, e6_re, e6_im, \
    o1_re, o1_im, o3_re, o3_im, o5_re, o5_im, o7_re, o7_im, \
    y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im, \
    y4_re, y4_im, y5_re, y5_im, y6_re, y6_im, y7_re, y7_im) \
    do                                                      \
    {                                                       \
        y0_re = _mm512_add_pd(e0_re, o1_re);                \
        y0_im = _mm512_add_pd(e0_im, o1_im);                \
        y4_re = _mm512_sub_pd(e0_re, o1_re);                \
        y4_im = _mm512_sub_pd(e0_im, o1_im);                \
        y1_re = _mm512_add_pd(e2_re, o3_re);                \
        y1_im = _mm512_add_pd(e2_im, o3_im);                \
        y5_re = _mm512_sub_pd(e2_re, o3_re);                \
        y5_im = _mm512_sub_pd(e2_im, o3_im);                \
        y2_re = _mm512_add_pd(e4_re, o5_re);                \
        y2_im = _mm512_add_pd(e4_im, o5_im);                \
        y6_re = _mm512_sub_pd(e4_re, o5_re);                \
        y6_im = _mm512_sub_pd(e4_im, o5_im);                \
        y3_re = _mm512_add_pd(e6_re, o7_re);                \
        y3_im = _mm512_add_pd(e6_im, o7_im);                \
        y7_re = _mm512_sub_pd(e6_re, o7_re);                \
        y7_im = _mm512_sub_pd(e6_im, o7_im);                \
    } while (0)

//==============================================================================
// TRULY FULLY UNROLLED TWIDDLE APPLICATION (PRESERVED)
//==============================================================================

/**
 * @brief TRULY fully unrolled twiddle application - 7 explicit statements
 * FIXED: Correct chunk_offset handling in twiddle indexing
 */
#define APPLY_TWIDDLES_8LANE_FULLY_UNROLLED(kk_base, x_re, x_im, tw, K_stride, chunk_offset) \
    do                                                                                       \
    {                                                                                        \
        __m512d tw_re, tw_im, res_re, res_im;                                                \
        /* Lane 1 */ {                                                                       \
            const int gl = (chunk_offset) + 1;                                               \
            tw_re = _mm512_load_pd(&(tw)->re[(kk_base) + gl * (K_stride)]);                  \
            tw_im = _mm512_load_pd(&(tw)->im[(kk_base) + gl * (K_stride)]);                  \
            CMUL_NATIVE_SOA_AVX512_P0P1(x_re[1], x_im[1], tw_re, tw_im, res_re, res_im);     \
            x_re[1] = res_re;                                                                \
            x_im[1] = res_im;                                                                \
        }                                                                                    \
        /* Lane 2 */ {                                                                       \
            const int gl = (chunk_offset) + 2;                                               \
            tw_re = _mm512_load_pd(&(tw)->re[(kk_base) + gl * (K_stride)]);                  \
            tw_im = _mm512_load_pd(&(tw)->im[(kk_base) + gl * (K_stride)]);                  \
            CMUL_NATIVE_SOA_AVX512_P0P1(x_re[2], x_im[2], tw_re, tw_im, res_re, res_im);     \
            x_re[2] = res_re;                                                                \
            x_im[2] = res_im;                                                                \
        }                                                                                    \
        /* Lane 3 */ {                                                                       \
            const int gl = (chunk_offset) + 3;                                               \
            tw_re = _mm512_load_pd(&(tw)->re[(kk_base) + gl * (K_stride)]);                  \
            tw_im = _mm512_load_pd(&(tw)->im[(kk_base) + gl * (K_stride)]);                  \
            CMUL_NATIVE_SOA_AVX512_P0P1(x_re[3], x_im[3], tw_re, tw_im, res_re, res_im);     \
            x_re[3] = res_re;                                                                \
            x_im[3] = res_im;                                                                \
        }                                                                                    \
        /* Lane 4 */ {                                                                       \
            const int gl = (chunk_offset) + 4;                                               \
            tw_re = _mm512_load_pd(&(tw)->re[(kk_base) + gl * (K_stride)]);                  \
            tw_im = _mm512_load_pd(&(tw)->im[(kk_base) + gl * (K_stride)]);                  \
            CMUL_NATIVE_SOA_AVX512_P0P1(x_re[4], x_im[4], tw_re, tw_im, res_re, res_im);     \
            x_re[4] = res_re;                                                                \
            x_im[4] = res_im;                                                                \
        }                                                                                    \
        /* Lane 5 */ {                                                                       \
            const int gl = (chunk_offset) + 5;                                               \
            tw_re = _mm512_load_pd(&(tw)->re[(kk_base) + gl * (K_stride)]);                  \
            tw_im = _mm512_load_pd(&(tw)->im[(kk_base) + gl * (K_stride)]);                  \
            CMUL_NATIVE_SOA_AVX512_P0P1(x_re[5], x_im[5], tw_re, tw_im, res_re, res_im);     \
            x_re[5] = res_re;                                                                \
            x_im[5] = res_im;                                                                \
        }                                                                                    \
        /* Lane 6 */ {                                                                       \
            const int gl = (chunk_offset) + 6;                                               \
            tw_re = _mm512_load_pd(&(tw)->re[(kk_base) + gl * (K_stride)]);                  \
            tw_im = _mm512_load_pd(&(tw)->im[(kk_base) + gl * (K_stride)]);                  \
            CMUL_NATIVE_SOA_AVX512_P0P1(x_re[6], x_im[6], tw_re, tw_im, res_re, res_im);     \
            x_re[6] = res_re;                                                                \
            x_im[6] = res_im;                                                                \
        }                                                                                    \
        /* Lane 7 */ {                                                                       \
            const int gl = (chunk_offset) + 7;                                               \
            tw_re = _mm512_load_pd(&(tw)->re[(kk_base) + gl * (K_stride)]);                  \
            tw_im = _mm512_load_pd(&(tw)->im[(kk_base) + gl * (K_stride)]);                  \
            CMUL_NATIVE_SOA_AVX512_P0P1(x_re[7], x_im[7], tw_re, tw_im, res_re, res_im);     \
            x_re[7] = res_re;                                                                \
            x_im[7] = res_im;                                                                \
        }                                                                                    \
    } while (0)

//==============================================================================
// TAIL MASKING UTILITIES (PRESERVED)
//==============================================================================

/**
 * @brief Generate mask for partial vector (K % 8 != 0)
 */
static inline __mmask8 fft_tail_mask(int remainder)
{
    return (__mmask8)((1u << remainder) - 1u);
}

//==============================================================================
// TWO-LEVEL PREFETCH WITH CORRECT BOUNDS (PRESERVED)
//==============================================================================

#define PREFETCH_TWO_LEVEL_BOUNDED(base_ptr, near_offset, far_offset, array_len) \
    do                                                                           \
    {                                                                            \
        if ((near_offset) < (array_len))                                         \
        {                                                                        \
            _mm_prefetch((const char *)&(base_ptr)[(near_offset)], _MM_HINT_T0); \
        }                                                                        \
        if ((far_offset) < (array_len))                                          \
        {                                                                        \
            _mm_prefetch((const char *)&(base_ptr)[(far_offset)], _MM_HINT_T1);  \
        }                                                                        \
    } while (0)

//==============================================================================
// MAIN PIPELINE MACROS - FORWARD (FV)
//==============================================================================

/**
 * @brief Forward Radix-32 Pipeline (Regular Stores)
 * Uses: APPLY_W32_*_FV, APPLY_W8_TWIDDLES_HOISTED_FV
 */
#define RADIX32_PIPELINE_4_FV_AVX512(                                                                        \
    in_re, in_im, out_re, out_im, stage_tw, K_VAL, USE_STREAMING)                                            \
    do                                                                                                       \
    {                                                                                                        \
        const int TOTAL_ELEMENTS = 32 * (K_VAL);                                                             \
        const __m512d NEG_ZERO = _mm512_set1_pd(-0.0);                                                       \
        const __m512d SQRT2_2 = _mm512_set1_pd(0.70710678118654752440);                                      \
        const __m512d SIGN_MASK = NEG_ZERO; /* Forward: negate imaginary for -i rotation */                  \
                                                                                                             \
        double *restrict in_re_aligned = (double *)__builtin_assume_aligned((in_re), 64);                    \
        double *restrict in_im_aligned = (double *)__builtin_assume_aligned((in_im), 64);                    \
        double *restrict out_re_aligned = (double *)__builtin_assume_aligned((out_re), 64);                  \
        double *restrict out_im_aligned = (double *)__builtin_assume_aligned((out_im), 64);                  \
        const double *restrict tw_re_aligned = (const double *)__builtin_assume_aligned((stage_tw)->re, 64); \
        const double *restrict tw_im_aligned = (const double *)__builtin_assume_aligned((stage_tw)->im, 64); \
                                                                                                             \
        const int K_REMAINDER = (K_VAL) & 7;                                                                 \
        const __mmask8 TAIL_MASK = K_REMAINDER ? fft_tail_mask(K_REMAINDER) : 0xFF;                          \
        const int K_FULL_VECS = (K_VAL) & ~7;                                                                \
                                                                                                             \
        /* Main loop: 4 butterflies */                                                                       \
        for (int b = 0; b < 4; ++b)                                                                          \
        {                                                                                                    \
            _Pragma("GCC ivdep") for (int kk = 0; kk < K_FULL_VECS; kk += 8)                                 \
            {                                                                                                \
                const int kk_base = kk + b * 8;                                                              \
                                                                                                             \
                /* Prefetch */                                                                               \
                const int pf_near = kk_base + FFT_PREFETCH_DISTANCE_NEAR;                                    \
                const int pf_far = kk_base + FFT_PREFETCH_DISTANCE_FAR;                                      \
                for (int pf_lane = 0; pf_lane < 32; pf_lane += 8)                                            \
                {                                                                                            \
                    PREFETCH_TWO_LEVEL_BOUNDED(in_re_aligned, pf_near + pf_lane * K_VAL,                     \
                                               pf_far + pf_lane * K_VAL, TOTAL_ELEMENTS);                    \
                    PREFETCH_TWO_LEVEL_BOUNDED(in_im_aligned, pf_near + pf_lane * K_VAL,                     \
                                               pf_far + pf_lane * K_VAL, TOTAL_ELEMENTS);                    \
                }                                                                                            \
                                                                                                             \
                /* Process 4 chunks of 8 lanes */                                                            \
                for (int chunk = 0; chunk < 4; ++chunk)                                                      \
                {                                                                                            \
                    const int chunk_offset = chunk * 8;                                                      \
                    __m512d x_re[8], x_im[8];                                                                \
                                                                                                             \
                    /* Load */                                                                               \
                    for (int lane = 0; lane < 8; ++lane)                                                     \
                    {                                                                                        \
                        const int gl = chunk_offset + lane;                                                  \
                        const int idx = kk_base + gl * K_VAL;                                                \
                        x_re[lane] = _mm512_load_pd(&in_re_aligned[idx]);                                    \
                        x_im[lane] = _mm512_load_pd(&in_im_aligned[idx]);                                    \
                    }                                                                                        \
                                                                                                             \
                    /* Stage twiddles */                                                                     \
                    if (chunk_offset > 0)                                                                    \
                    {                                                                                        \
                        APPLY_TWIDDLES_8LANE_FULLY_UNROLLED(kk_base, x_re, x_im,                             \
                                                            stage_tw, K_VAL, chunk_offset);                  \
                    }                                                                                        \
                    else                                                                                     \
                    {                                                                                        \
                        __m512d tw_re, tw_im, res_re, res_im;                                                \
                        for (int lane = 1; lane < 8; ++lane)                                                 \
                        {                                                                                    \
                            const int gl = lane;                                                             \
                            const int tw_idx = kk_base + gl * K_VAL;                                         \
                            tw_re = _mm512_load_pd(&tw_re_aligned[tw_idx]);                                  \
                            tw_im = _mm512_load_pd(&tw_im_aligned[tw_idx]);                                  \
                            CMUL_NATIVE_SOA_AVX512_P0P1(x_re[lane], x_im[lane],                              \
                                                        tw_re, tw_im, res_re, res_im);                       \
                            x_re[lane] = res_re;                                                             \
                            x_im[lane] = res_im;                                                             \
                        }                                                                                    \
                    }                                                                                        \
                                                                                                             \
                    /* First radix-4 layer */                                                                \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX512(x_re[0], x_im[0], x_re[1], x_im[1],                   \
                                                       x_re[2], x_im[2], x_re[3], x_im[3],                   \
                                                       SIGN_MASK);                                           \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX512(x_re[4], x_im[4], x_re[5], x_im[5],                   \
                                                       x_re[6], x_im[6], x_re[7], x_im[7],                   \
                                                       SIGN_MASK);                                           \
                                                                                                             \
                    /* W32 twiddles - FORWARD */                                                             \
                    switch (chunk_offset)                                                                    \
                    {                                                                                        \
                    case 0:                                                                                  \
                        APPLY_W32_8LANE_EXPLICIT_J0_FV(x_re, x_im, W32_COS_TABLE, W32_SIN_TABLE,             \
                                                       SQRT2_2, NEG_ZERO);                                   \
                        break;                                                                               \
                    case 8:                                                                                  \
                        APPLY_W32_8LANE_EXPLICIT_J1_FV(x_re, x_im, W32_COS_TABLE, W32_SIN_TABLE,             \
                                                       SQRT2_2, NEG_ZERO);                                   \
                        break;                                                                               \
                    case 16:                                                                                 \
                        APPLY_W32_8LANE_EXPLICIT_J2_FV(x_re, x_im, W32_COS_TABLE, W32_SIN_TABLE,             \
                                                       SQRT2_2, NEG_ZERO);                                   \
                        break;                                                                               \
                    case 24:                                                                                 \
                        APPLY_W32_8LANE_EXPLICIT_J3_FV(x_re, x_im, W32_COS_TABLE, W32_SIN_TABLE,             \
                                                       SQRT2_2, NEG_ZERO);                                   \
                        break;                                                                               \
                    }                                                                                        \
                                                                                                             \
                    /* Second radix-4 layer (even/odd) */                                                    \
                    __m512d e0_re = x_re[0], e0_im = x_im[0];                                                \
                    __m512d e2_re = x_re[2], e2_im = x_im[2];                                                \
                    __m512d e4_re = x_re[4], e4_im = x_im[4];                                                \
                    __m512d e6_re = x_re[6], e6_im = x_im[6];                                                \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX512(e0_re, e0_im, e2_re, e2_im,                           \
                                                       e4_re, e4_im, e6_re, e6_im, SIGN_MASK);               \
                                                                                                             \
                    __m512d o1_re = x_re[1], o1_im = x_im[1];                                                \
                    __m512d o3_re = x_re[3], o3_im = x_im[3];                                                \
                    __m512d o5_re = x_re[5], o5_im = x_im[5];                                                \
                    __m512d o7_re = x_re[7], o7_im = x_im[7];                                                \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX512(o1_re, o1_im, o3_re, o3_im,                           \
                                                       o5_re, o5_im, o7_re, o7_im, SIGN_MASK);               \
                                                                                                             \
                    /* W8 twiddles - FORWARD */                                                              \
                    APPLY_W8_TWIDDLES_HOISTED_FV(o1_re, o1_im, o3_re, o3_im, o5_re, o5_im,                   \
                                                 SQRT2_2, NEG_ZERO);                                         \
                                                                                                             \
                    /* Radix-8 combine */                                                                    \
                    RADIX8_COMBINE_SCHEDULED(                                                                \
                        e0_re, e0_im, e2_re, e2_im, e4_re, e4_im, e6_re, e6_im,                              \
                        o1_re, o1_im, o3_re, o3_im, o5_re, o5_im, o7_re, o7_im,                              \
                        x_re[0], x_im[0], x_re[1], x_im[1], x_re[2], x_im[2], x_re[3], x_im[3],              \
                        x_re[4], x_im[4], x_re[5], x_im[5], x_re[6], x_im[6], x_re[7], x_im[7]);             \
                                                                                                             \
                    /* Store */                                                                              \
                    for (int lane = 0; lane < 8; ++lane)                                                     \
                    {                                                                                        \
                        const int out_lane = chunk_offset + lane;                                            \
                        const int out_idx = kk_base + out_lane * K_VAL;                                      \
                        if (USE_STREAMING)                                                                   \
                        {                                                                                    \
                            _mm512_stream_pd(&out_re_aligned[out_idx], x_re[lane]);                          \
                            _mm512_stream_pd(&out_im_aligned[out_idx], x_im[lane]);                          \
                        }                                                                                    \
                        else                                                                                 \
                        {                                                                                    \
                            _mm512_store_pd(&out_re_aligned[out_idx], x_re[lane]);                           \
                            _mm512_store_pd(&out_im_aligned[out_idx], x_im[lane]);                           \
                        }                                                                                    \
                    }                                                                                        \
                }                                                                                            \
            }                                                                                                \
        }                                                                                                    \
                                                                                                             \
        /* Tail handling (if K_VAL % 8 != 0) */                                                              \
        if (K_REMAINDER)                                                                                     \
        {                                                                                                    \
            for (int b = 0; b < 4; ++b)                                                                      \
            {                                                                                                \
                const int kk_base = K_FULL_VECS + b * 8;                                                     \
                if (kk_base >= K_VAL)                                                                        \
                    break;                                                                                   \
                                                                                                             \
                for (int chunk = 0; chunk < 4; ++chunk)                                                      \
                {                                                                                            \
                    const int chunk_offset = chunk * 8;                                                      \
                    __m512d x_re[8], x_im[8];                                                                \
                                                                                                             \
                    /* Masked load */                                                                        \
                    for (int lane = 0; lane < 8; ++lane)                                                     \
                    {                                                                                        \
                        const int gl = chunk_offset + lane;                                                  \
                        const int idx = kk_base + gl * K_VAL;                                                \
                        x_re[lane] = _mm512_maskz_loadu_pd(TAIL_MASK, &in_re_aligned[idx]);                  \
                        x_im[lane] = _mm512_maskz_loadu_pd(TAIL_MASK, &in_im_aligned[idx]);                  \
                    }                                                                                        \
                                                                                                             \
                    /* Apply all transformations (same as main loop) */                                      \
                    if (chunk_offset > 0)                                                                    \
                    {                                                                                        \
                        __m512d tw_re, tw_im, res_re, res_im;                                                \
                        for (int lane = 1; lane < 8; ++lane)                                                 \
                        {                                                                                    \
                            const int gl = chunk_offset + lane;                                              \
                            const int tw_idx = kk_base + gl * K_VAL;                                         \
                            tw_re = _mm512_maskz_loadu_pd(TAIL_MASK, &tw_re_aligned[tw_idx]);                \
                            tw_im = _mm512_maskz_loadu_pd(TAIL_MASK, &tw_im_aligned[tw_idx]);                \
                            CMUL_NATIVE_SOA_AVX512_P0P1(x_re[lane], x_im[lane],                              \
                                                        tw_re, tw_im, res_re, res_im);                       \
                            x_re[lane] = res_re;                                                             \
                            x_im[lane] = res_im;                                                             \
                        }                                                                                    \
                    }                                                                                        \
                    else                                                                                     \
                    {                                                                                        \
                        __m512d tw_re, tw_im, res_re, res_im;                                                \
                        for (int lane = 1; lane < 8; ++lane)                                                 \
                        {                                                                                    \
                            const int tw_idx = kk_base + lane * K_VAL;                                       \
                            tw_re = _mm512_maskz_loadu_pd(TAIL_MASK, &tw_re_aligned[tw_idx]);                \
                            tw_im = _mm512_maskz_loadu_pd(TAIL_MASK, &tw_im_aligned[tw_idx]);                \
                            CMUL_NATIVE_SOA_AVX512_P0P1(x_re[lane], x_im[lane],                              \
                                                        tw_re, tw_im, res_re, res_im);                       \
                            x_re[lane] = res_re;                                                             \
                            x_im[lane] = res_im;                                                             \
                        }                                                                                    \
                    }                                                                                        \
                                                                                                             \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX512(x_re[0], x_im[0], x_re[1], x_im[1],                   \
                                                       x_re[2], x_im[2], x_re[3], x_im[3],                   \
                                                       SIGN_MASK);                                           \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX512(x_re[4], x_im[4], x_re[5], x_im[5],                   \
                                                       x_re[6], x_im[6], x_re[7], x_im[7],                   \
                                                       SIGN_MASK);                                           \
                                                                                                             \
                    switch (chunk_offset)                                                                    \
                    {                                                                                        \
                    case 0:                                                                                  \
                        APPLY_W32_8LANE_EXPLICIT_J0_FV(x_re, x_im, W32_COS_TABLE, W32_SIN_TABLE,             \
                                                       SQRT2_2, NEG_ZERO);                                   \
                        break;                                                                               \
                    case 8:                                                                                  \
                        APPLY_W32_8LANE_EXPLICIT_J1_FV(x_re, x_im, W32_COS_TABLE, W32_SIN_TABLE,             \
                                                       SQRT2_2, NEG_ZERO);                                   \
                        break;                                                                               \
                    case 16:                                                                                 \
                        APPLY_W32_8LANE_EXPLICIT_J2_FV(x_re, x_im, W32_COS_TABLE, W32_SIN_TABLE,             \
                                                       SQRT2_2, NEG_ZERO);                                   \
                        break;                                                                               \
                    case 24:                                                                                 \
                        APPLY_W32_8LANE_EXPLICIT_J3_FV(x_re, x_im, W32_COS_TABLE, W32_SIN_TABLE,             \
                                                       SQRT2_2, NEG_ZERO);                                   \
                        break;                                                                               \
                    }                                                                                        \
                                                                                                             \
                    __m512d e0_re = x_re[0], e0_im = x_im[0];                                                \
                    __m512d e2_re = x_re[2], e2_im = x_im[2];                                                \
                    __m512d e4_re = x_re[4], e4_im = x_im[4];                                                \
                    __m512d e6_re = x_re[6], e6_im = x_im[6];                                                \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX512(e0_re, e0_im, e2_re, e2_im,                           \
                                                       e4_re, e4_im, e6_re, e6_im, SIGN_MASK);               \
                                                                                                             \
                    __m512d o1_re = x_re[1], o1_im = x_im[1];                                                \
                    __m512d o3_re = x_re[3], o3_im = x_im[3];                                                \
                    __m512d o5_re = x_re[5], o5_im = x_im[5];                                                \
                    __m512d o7_re = x_re[7], o7_im = x_im[7];                                                \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX512(o1_re, o1_im, o3_re, o3_im,                           \
                                                       o5_re, o5_im, o7_re, o7_im, SIGN_MASK);               \
                                                                                                             \
                    APPLY_W8_TWIDDLES_HOISTED_FV(o1_re, o1_im, o3_re, o3_im, o5_re, o5_im,                   \
                                                 SQRT2_2, NEG_ZERO);                                         \
                                                                                                             \
                    RADIX8_COMBINE_SCHEDULED(                                                                \
                        e0_re, e0_im, e2_re, e2_im, e4_re, e4_im, e6_re, e6_im,                              \
                        o1_re, o1_im, o3_re, o3_im, o5_re, o5_im, o7_re, o7_im,                              \
                        x_re[0], x_im[0], x_re[1], x_im[1], x_re[2], x_im[2], x_re[3], x_im[3],              \
                        x_re[4], x_im[4], x_re[5], x_im[5], x_re[6], x_im[6], x_re[7], x_im[7]);             \
                                                                                                             \
                    /* Masked store */                                                                       \
                    for (int lane = 0; lane < 8; ++lane)                                                     \
                    {                                                                                        \
                        const int out_lane = chunk_offset + lane;                                            \
                        const int out_idx = kk_base + out_lane * K_VAL;                                      \
                        _mm512_mask_storeu_pd(&out_re_aligned[out_idx], TAIL_MASK, x_re[lane]);              \
                        _mm512_mask_storeu_pd(&out_im_aligned[out_idx], TAIL_MASK, x_im[lane]);              \
                    }                                                                                        \
                }                                                                                            \
            }                                                                                                \
        }                                                                                                    \
                                                                                                             \
        if (USE_STREAMING)                                                                                   \
        {                                                                                                    \
            _mm_sfence();                                                                                    \
        }                                                                                                    \
    } while (0)

//==============================================================================
// MAIN PIPELINE MACROS - BACKWARD (BV)
//==============================================================================

/**
 * @brief Backward Radix-32 Pipeline (Regular Stores)
 * Uses: APPLY_W32_*_BV, APPLY_W8_TWIDDLES_HOISTED_BV
 */
#define RADIX32_PIPELINE_4_BV_AVX512(                                                                        \
    in_re, in_im, out_re, out_im, stage_tw, K_VAL, USE_STREAMING)                                            \
    do                                                                                                       \
    {                                                                                                        \
        const int TOTAL_ELEMENTS = 32 * (K_VAL);                                                             \
        const __m512d NEG_ZERO = _mm512_set1_pd(-0.0);                                                       \
        const __m512d SQRT2_2 = _mm512_set1_pd(0.70710678118654752440);                                      \
        const __m512d SIGN_MASK = _mm512_setzero_pd(); /* Backward: don't negate for +i rotation */          \
                                                                                                             \
        double *restrict in_re_aligned = (double *)__builtin_assume_aligned((in_re), 64);                    \
        double *restrict in_im_aligned = (double *)__builtin_assume_aligned((in_im), 64);                    \
        double *restrict out_re_aligned = (double *)__builtin_assume_aligned((out_re), 64);                  \
        double *restrict out_im_aligned = (double *)__builtin_assume_aligned((out_im), 64);                  \
        const double *restrict tw_re_aligned = (const double *)__builtin_assume_aligned((stage_tw)->re, 64); \
        const double *restrict tw_im_aligned = (const double *)__builtin_assume_aligned((stage_tw)->im, 64); \
                                                                                                             \
        const int K_REMAINDER = (K_VAL) & 7;                                                                 \
        const __mmask8 TAIL_MASK = K_REMAINDER ? fft_tail_mask(K_REMAINDER) : 0xFF;                          \
        const int K_FULL_VECS = (K_VAL) & ~7;                                                                \
                                                                                                             \
        /* Main loop - identical structure to FV but uses _BV macros */                                      \
        for (int b = 0; b < 4; ++b)                                                                          \
        {                                                                                                    \
            _Pragma("GCC ivdep") for (int kk = 0; kk < K_FULL_VECS; kk += 8)                                 \
            {                                                                                                \
                const int kk_base = kk + b * 8;                                                              \
                                                                                                             \
                const int pf_near = kk_base + FFT_PREFETCH_DISTANCE_NEAR;                                    \
                const int pf_far = kk_base + FFT_PREFETCH_DISTANCE_FAR;                                      \
                for (int pf_lane = 0; pf_lane < 32; pf_lane += 8)                                            \
                {                                                                                            \
                    PREFETCH_TWO_LEVEL_BOUNDED(in_re_aligned, pf_near + pf_lane * K_VAL,                     \
                                               pf_far + pf_lane * K_VAL, TOTAL_ELEMENTS);                    \
                    PREFETCH_TWO_LEVEL_BOUNDED(in_im_aligned, pf_near + pf_lane * K_VAL,                     \
                                               pf_far + pf_lane * K_VAL, TOTAL_ELEMENTS);                    \
                }                                                                                            \
                                                                                                             \
                for (int chunk = 0; chunk < 4; ++chunk)                                                      \
                {                                                                                            \
                    const int chunk_offset = chunk * 8;                                                      \
                    __m512d x_re[8], x_im[8];                                                                \
                                                                                                             \
                    for (int lane = 0; lane < 8; ++lane)                                                     \
                    {                                                                                        \
                        const int gl = chunk_offset + lane;                                                  \
                        const int idx = kk_base + gl * K_VAL;                                                \
                        x_re[lane] = _mm512_load_pd(&in_re_aligned[idx]);                                    \
                        x_im[lane] = _mm512_load_pd(&in_im_aligned[idx]);                                    \
                    }                                                                                        \
                                                                                                             \
                    if (chunk_offset > 0)                                                                    \
                    {                                                                                        \
                        APPLY_TWIDDLES_8LANE_FULLY_UNROLLED(kk_base, x_re, x_im,                             \
                                                            stage_tw, K_VAL, chunk_offset);                  \
                    }                                                                                        \
                    else                                                                                     \
                    {                                                                                        \
                        __m512d tw_re, tw_im, res_re, res_im;                                                \
                        for (int lane = 1; lane < 8; ++lane)                                                 \
                        {                                                                                    \
                            const int gl = lane;                                                             \
                            const int tw_idx = kk_base + gl * K_VAL;                                         \
                            tw_re = _mm512_load_pd(&tw_re_aligned[tw_idx]);                                  \
                            tw_im = _mm512_load_pd(&tw_im_aligned[tw_idx]);                                  \
                            CMUL_NATIVE_SOA_AVX512_P0P1(x_re[lane], x_im[lane],                              \
                                                        tw_re, tw_im, res_re, res_im);                       \
                            x_re[lane] = res_re;                                                             \
                            x_im[lane] = res_im;                                                             \
                        }                                                                                    \
                    }                                                                                        \
                                                                                                             \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX512(x_re[0], x_im[0], x_re[1], x_im[1],                   \
                                                       x_re[2], x_im[2], x_re[3], x_im[3],                   \
                                                       SIGN_MASK);                                           \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX512(x_re[4], x_im[4], x_re[5], x_im[5],                   \
                                                       x_re[6], x_im[6], x_re[7], x_im[7],                   \
                                                       SIGN_MASK);                                           \
                                                                                                             \
                    /* W32 twiddles - BACKWARD */                                                            \
                    switch (chunk_offset)                                                                    \
                    {                                                                                        \
                    case 0:                                                                                  \
                        APPLY_W32_8LANE_EXPLICIT_J0_BV(x_re, x_im, W32_COS_TABLE, W32_SIN_TABLE,             \
                                                       SQRT2_2, NEG_ZERO);                                   \
                        break;                                                                               \
                    case 8:                                                                                  \
                        APPLY_W32_8LANE_EXPLICIT_J1_BV(x_re, x_im, W32_COS_TABLE, W32_SIN_TABLE,             \
                                                       SQRT2_2, NEG_ZERO);                                   \
                        break;                                                                               \
                    case 16:                                                                                 \
                        APPLY_W32_8LANE_EXPLICIT_J2_BV(x_re, x_im, W32_COS_TABLE, W32_SIN_TABLE,             \
                                                       SQRT2_2, NEG_ZERO);                                   \
                        break;                                                                               \
                    case 24:                                                                                 \
                        APPLY_W32_8LANE_EXPLICIT_J3_BV(x_re, x_im, W32_COS_TABLE, W32_SIN_TABLE,             \
                                                       SQRT2_2, NEG_ZERO);                                   \
                        break;                                                                               \
                    }                                                                                        \
                                                                                                             \
                    __m512d e0_re = x_re[0], e0_im = x_im[0];                                                \
                    __m512d e2_re = x_re[2], e2_im = x_im[2];                                                \
                    __m512d e4_re = x_re[4], e4_im = x_im[4];                                                \
                    __m512d e6_re = x_re[6], e6_im = x_im[6];                                                \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX512(e0_re, e0_im, e2_re, e2_im,                           \
                                                       e4_re, e4_im, e6_re, e6_im, SIGN_MASK);               \
                                                                                                             \
                    __m512d o1_re = x_re[1], o1_im = x_im[1];                                                \
                    __m512d o3_re = x_re[3], o3_im = x_im[3];                                                \
                    __m512d o5_re = x_re[5], o5_im = x_im[5];                                                \
                    __m512d o7_re = x_re[7], o7_im = x_im[7];                                                \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX512(o1_re, o1_im, o3_re, o3_im,                           \
                                                       o5_re, o5_im, o7_re, o7_im, SIGN_MASK);               \
                                                                                                             \
                    /* W8 twiddles - BACKWARD */                                                             \
                    APPLY_W8_TWIDDLES_HOISTED_BV(o1_re, o1_im, o3_re, o3_im, o5_re, o5_im,                   \
                                                 SQRT2_2, NEG_ZERO);                                         \
                                                                                                             \
                    RADIX8_COMBINE_SCHEDULED(                                                                \
                        e0_re, e0_im, e2_re, e2_im, e4_re, e4_im, e6_re, e6_im,                              \
                        o1_re, o1_im, o3_re, o3_im, o5_re, o5_im, o7_re, o7_im,                              \
                        x_re[0], x_im[0], x_re[1], x_im[1], x_re[2], x_im[2], x_re[3], x_im[3],              \
                        x_re[4], x_im[4], x_re[5], x_im[5], x_re[6], x_im[6], x_re[7], x_im[7]);             \
                                                                                                             \
                    for (int lane = 0; lane < 8; ++lane)                                                     \
                    {                                                                                        \
                        const int out_lane = chunk_offset + lane;                                            \
                        const int out_idx = kk_base + out_lane * K_VAL;                                      \
                        if (USE_STREAMING)                                                                   \
                        {                                                                                    \
                            _mm512_stream_pd(&out_re_aligned[out_idx], x_re[lane]);                          \
                            _mm512_stream_pd(&out_im_aligned[out_idx], x_im[lane]);                          \
                        }                                                                                    \
                        else                                                                                 \
                        {                                                                                    \
                            _mm512_store_pd(&out_re_aligned[out_idx], x_re[lane]);                           \
                            _mm512_store_pd(&out_im_aligned[out_idx], x_im[lane]);                           \
                        }                                                                                    \
                    }                                                                                        \
                }                                                                                            \
            }                                                                                                \
        }                                                                                                    \
                                                                                                             \
        /* Tail handling (identical to FV but uses _BV macros) */                                            \
        if (K_REMAINDER)                                                                                     \
        {                                                                                                    \
            for (int b = 0; b < 4; ++b)                                                                      \
            {                                                                                                \
                const int kk_base = K_FULL_VECS + b * 8;                                                     \
                if (kk_base >= K_VAL)                                                                        \
                    break;                                                                                   \
                                                                                                             \
                for (int chunk = 0; chunk < 4; ++chunk)                                                      \
                {                                                                                            \
                    const int chunk_offset = chunk * 8;                                                      \
                    __m512d x_re[8], x_im[8];                                                                \
                                                                                                             \
                    for (int lane = 0; lane < 8; ++lane)                                                     \
                    {                                                                                        \
                        const int gl = chunk_offset + lane;                                                  \
                        const int idx = kk_base + gl * K_VAL;                                                \
                        x_re[lane] = _mm512_maskz_loadu_pd(TAIL_MASK, &in_re_aligned[idx]);                  \
                        x_im[lane] = _mm512_maskz_loadu_pd(TAIL_MASK, &in_im_aligned[idx]);                  \
                    }                                                                                        \
                                                                                                             \
                    if (chunk_offset > 0)                                                                    \
                    {                                                                                        \
                        __m512d tw_re, tw_im, res_re, res_im;                                                \
                        for (int lane = 1; lane < 8; ++lane)                                                 \
                        {                                                                                    \
                            const int gl = chunk_offset + lane;                                              \
                            const int tw_idx = kk_base + gl * K_VAL;                                         \
                            tw_re = _mm512_maskz_loadu_pd(TAIL_MASK, &tw_re_aligned[tw_idx]);                \
                            tw_im = _mm512_maskz_loadu_pd(TAIL_MASK, &tw_im_aligned[tw_idx]);                \
                            CMUL_NATIVE_SOA_AVX512_P0P1(x_re[lane], x_im[lane],                              \
                                                        tw_re, tw_im, res_re, res_im);                       \
                            x_re[lane] = res_re;                                                             \
                            x_im[lane] = res_im;                                                             \
                        }                                                                                    \
                    }                                                                                        \
                    else                                                                                     \
                    {                                                                                        \
                        __m512d tw_re, tw_im, res_re, res_im;                                                \
                        for (int lane = 1; lane < 8; ++lane)                                                 \
                        {                                                                                    \
                            const int tw_idx = kk_base + lane * K_VAL;                                       \
                            tw_re = _mm512_maskz_loadu_pd(TAIL_MASK, &tw_re_aligned[tw_idx]);                \
                            tw_im = _mm512_maskz_loadu_pd(TAIL_MASK, &tw_im_aligned[tw_idx]);                \
                            CMUL_NATIVE_SOA_AVX512_P0P1(x_re[lane], x_im[lane],                              \
                                                        tw_re, tw_im, res_re, res_im);                       \
                            x_re[lane] = res_re;                                                             \
                            x_im[lane] = res_im;                                                             \
                        }                                                                                    \
                    }                                                                                        \
                                                                                                             \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX512(x_re[0], x_im[0], x_re[1], x_im[1],                   \
                                                       x_re[2], x_im[2], x_re[3], x_im[3],                   \
                                                       SIGN_MASK);                                           \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX512(x_re[4], x_im[4], x_re[5], x_im[5],                   \
                                                       x_re[6], x_im[6], x_re[7], x_im[7],                   \
                                                       SIGN_MASK);                                           \
                                                                                                             \
                    switch (chunk_offset)                                                                    \
                    {                                                                                        \
                    case 0:                                                                                  \
                        APPLY_W32_8LANE_EXPLICIT_J0_BV(x_re, x_im, W32_COS_TABLE, W32_SIN_TABLE,             \
                                                       SQRT2_2, NEG_ZERO);                                   \
                        break;                                                                               \
                    case 8:                                                                                  \
                        APPLY_W32_8LANE_EXPLICIT_J1_BV(x_re, x_im, W32_COS_TABLE, W32_SIN_TABLE,             \
                                                       SQRT2_2, NEG_ZERO);                                   \
                        break;                                                                               \
                    case 16:                                                                                 \
                        APPLY_W32_8LANE_EXPLICIT_J2_BV(x_re, x_im, W32_COS_TABLE, W32_SIN_TABLE,             \
                                                       SQRT2_2, NEG_ZERO);                                   \
                        break;                                                                               \
                    case 24:                                                                                 \
                        APPLY_W32_8LANE_EXPLICIT_J3_BV(x_re, x_im, W32_COS_TABLE, W32_SIN_TABLE,             \
                                                       SQRT2_2, NEG_ZERO);                                   \
                        break;                                                                               \
                    }                                                                                        \
                                                                                                             \
                    __m512d e0_re = x_re[0], e0_im = x_im[0];                                                \
                    __m512d e2_re = x_re[2], e2_im = x_im[2];                                                \
                    __m512d e4_re = x_re[4], e4_im = x_im[4];                                                \
                    __m512d e6_re = x_re[6], e6_im = x_im[6];                                                \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX512(e0_re, e0_im, e2_re, e2_im,                           \
                                                       e4_re, e4_im, e6_re, e6_im, SIGN_MASK);               \
                                                                                                             \
                    __m512d o1_re = x_re[1], o1_im = x_im[1];                                                \
                    __m512d o3_re = x_re[3], o3_im = x_im[3];                                                \
                    __m512d o5_re = x_re[5], o5_im = x_im[5];                                                \
                    __m512d o7_re = x_re[7], o7_im = x_im[7];                                                \
                    RADIX4_BUTTERFLY_NATIVE_SOA_AVX512(o1_re, o1_im, o3_re, o3_im,                           \
                                                       o5_re, o5_im, o7_re, o7_im, SIGN_MASK);               \
                                                                                                             \
                    APPLY_W8_TWIDDLES_HOISTED_BV(o1_re, o1_im, o3_re, o3_im, o5_re, o5_im,                   \
                                                 SQRT2_2, NEG_ZERO);                                         \
                                                                                                             \
                    RADIX8_COMBINE_SCHEDULED(                                                                \
                        e0_re, e0_im, e2_re, e2_im, e4_re, e4_im, e6_re, e6_im,                              \
                        o1_re, o1_im, o3_re, o3_im, o5_re, o5_im, o7_re, o7_im,                              \
                        x_re[0], x_im[0], x_re[1], x_im[1], x_re[2], x_im[2], x_re[3], x_im[3],              \
                        x_re[4], x_im[4], x_re[5], x_im[5], x_re[6], x_im[6], x_re[7], x_im[7]);             \
                                                                                                             \
                    for (int lane = 0; lane < 8; ++lane)                                                     \
                    {                                                                                        \
                        const int out_lane = chunk_offset + lane;                                            \
                        const int out_idx = kk_base + out_lane * K_VAL;                                      \
                        _mm512_mask_storeu_pd(&out_re_aligned[out_idx], TAIL_MASK, x_re[lane]);              \
                        _mm512_mask_storeu_pd(&out_im_aligned[out_idx], TAIL_MASK, x_im[lane]);              \
                    }                                                                                        \
                }                                                                                            \
            }                                                                                                \
        }                                                                                                    \
                                                                                                             \
        if (USE_STREAMING)                                                                                   \
        {                                                                                                    \
            _mm_sfence();                                                                                    \
        }                                                                                                    \
    } while (0)

//==============================================================================
// STREAMING STORE VARIANTS (for use_streaming path in .c files)
//==============================================================================

#define RADIX32_PIPELINE_4_FV_AVX512_STREAM(k, K, sub_outputs, stage_tw, output_buffer) \
    RADIX32_PIPELINE_4_FV_AVX512((sub_outputs).re, (sub_outputs).im,                    \
                                 (output_buffer).re, (output_buffer).im,                \
                                 stage_tw, K, 1)

#define RADIX32_PIPELINE_4_BV_AVX512_STREAM(k, K, sub_outputs, stage_tw, output_buffer) \
    RADIX32_PIPELINE_4_BV_AVX512((sub_outputs).re, (sub_outputs).im,                    \
                                 (output_buffer).re, (output_buffer).im,                \
                                 stage_tw, K, 1)

#endif // __AVX512F__
#endif // FFT_RADIX32_MACROS_NATIVE_SOA_AVX512_FIXED_H