//==============================================================================
// fft_radix32_macros.h - Shared Macros for Radix-32 Butterflies
//==============================================================================
//
// USAGE:
//   #include "fft_radix32_macros.h" in both fft_radix32_fv.c and fft_radix32_bv.c
//
// BENEFITS:
//   - 99% code reuse between forward/inverse
//   - Single source of truth for radix-32 decomposition
//   - Only difference: W_32 and W_8 constants (baked into each file)
//

#ifndef FFT_RADIX32_MACROS_H
#define FFT_RADIX32_MACROS_H

#include "simd_math.h"

//==============================================================================
// COMPLEX MULTIPLICATION - FMA-optimized (IDENTICAL for both directions)
//==============================================================================

#ifdef __AVX2__
/**
 * @brief Optimized complex multiply: out = a * w (6 FMA + 2 UNPACK)
 */
#define CMUL_FMA_R32(out, a, w)                                      \
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
// RADIX-4 BUTTERFLY CORE - Direction-agnostic arithmetic
//==============================================================================

/**
 * @brief Core radix-4 sums/differences (IDENTICAL for forward/inverse)
 */
#ifdef __AVX2__
#define RADIX4_BUTTERFLY_CORE_R32(a, b, c, d, sumBD, difBD, sumAC, difAC) \
    do { \
        sumBD = _mm256_add_pd(b, d); \
        difBD = _mm256_sub_pd(b, d); \
        sumAC = _mm256_add_pd(a, c); \
        difAC = _mm256_sub_pd(a, c); \
    } while (0)
#endif

// Scalar version
#define RADIX4_BUTTERFLY_CORE_R32_SCALAR(a, b, c, d, \
                                          sumBD, difBD, sumAC, difAC) \
    do { \
        sumBD.re = b.re + d.re; \
        sumBD.im = b.im + d.im; \
        difBD.re = b.re - d.re; \
        difBD.im = b.im - d.im; \
        sumAC.re = a.re + c.re; \
        sumAC.im = a.im + c.im; \
        difAC.re = a.re - c.re; \
        difAC.im = a.im - c.im; \
    } while (0)

//==============================================================================
// ROTATION - ONLY DIFFERENCE BETWEEN FORWARD/INVERSE
//==============================================================================

/**
 * @brief FORWARD rotation: -i * difBD
 */
#ifdef __AVX2__
#define RADIX4_ROTATE_FORWARD_R32(difBD, rot) \
    do { \
        __m256d rot_mask_fv = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0); \
        rot = _mm256_xor_pd(_mm256_permute_pd(difBD, 0b0101), rot_mask_fv); \
    } while (0)
#endif

#define RADIX4_ROTATE_FORWARD_R32_SCALAR(difBD, rot) \
    do { \
        rot.re = difBD.im;   \
        rot.im = -difBD.re;  \
    } while (0)

/**
 * @brief INVERSE rotation: +i * difBD
 */
#ifdef __AVX2__
#define RADIX4_ROTATE_INVERSE_R32(difBD, rot) \
    do { \
        __m256d rot_mask_bv = _mm256_set_pd(0.0, -0.0, 0.0, -0.0); \
        rot = _mm256_xor_pd(_mm256_permute_pd(difBD, 0b0101), rot_mask_bv); \
    } while (0)
#endif

#define RADIX4_ROTATE_INVERSE_R32_SCALAR(difBD, rot) \
    do { \
        rot.re = -difBD.im;  \
        rot.im = difBD.re;   \
    } while (0)

//==============================================================================
// OUTPUT ASSEMBLY - IDENTICAL for forward/inverse
//==============================================================================

#ifdef __AVX2__
#define RADIX4_ASSEMBLE_OUTPUTS_R32(sumAC, sumBD, difAC, rot, y0, y1, y2, y3) \
    do { \
        y0 = _mm256_add_pd(sumAC, sumBD); \
        y2 = _mm256_sub_pd(sumAC, sumBD); \
        y1 = _mm256_sub_pd(difAC, rot);   \
        y3 = _mm256_add_pd(difAC, rot);   \
    } while (0)
#endif

#define RADIX4_ASSEMBLE_OUTPUTS_R32_SCALAR(sumAC, sumBD, difAC, rot, y0, y1, y2, y3) \
    do { \
        y0.re = sumAC.re + sumBD.re; \
        y0.im = sumAC.im + sumBD.im; \
        y2.re = sumAC.re - sumBD.re; \
        y2.im = sumAC.im - sumBD.im; \
        y1.re = difAC.re - rot.re;   \
        y1.im = difAC.im - rot.im;   \
        y3.re = difAC.re + rot.re;   \
        y3.im = difAC.im + rot.im;   \
    } while (0)

//==============================================================================
// APPLY PRECOMPUTED STAGE TWIDDLES - IDENTICAL for forward/inverse
//==============================================================================

/**
 * @brief Apply stage twiddles for 2 butterflies (kk and kk+1)
 * 
 * @param kk Current index
 * @param d_vec Input vector (2 complex values)
 * @param stage_tw Precomputed stage twiddles [K * 31]
 * @param lane Lane index (1-31)
 * @param tw_out Output twiddled vector
 */
#ifdef __AVX2__
#define APPLY_STAGE_TWIDDLE_R32(kk, d_vec, stage_tw, lane, tw_out) \
    do { \
        const int tw_idx = (kk) * 31 + (lane) - 1; \
        __m256d tw = _mm256_set_pd( \
            stage_tw[tw_idx + 31].im, stage_tw[tw_idx + 31].re, \
            stage_tw[tw_idx].im, stage_tw[tw_idx].re); \
        CMUL_FMA_R32(tw_out, d_vec, tw); \
    } while (0)
#endif

//==============================================================================
// W_32 TWIDDLES - HARDCODED GEOMETRIC CONSTANTS
//==============================================================================

/**
 * @brief Apply W_32 twiddles (FORWARD: exp(-2πi*j*g/32))
 * 
 * Applies to lanes [8..31] after first radix-4 layer
 * For j=1,2,3 and g=0..7
 */
#ifdef __AVX2__
#define APPLY_W32_TWIDDLES_FV_AVX2(x) \
    do { \
        /* j=1, g=0..7 (lanes 8-15) */ \
        CMUL_FMA_R32(x[8][b], x[8][b], _mm256_set_pd(0.0, 1.0, 0.0, 1.0)); \
        CMUL_FMA_R32(x[9][b], x[9][b], _mm256_set_pd(-0.1950903220, 0.9807852804, -0.1950903220, 0.9807852804)); \
        CMUL_FMA_R32(x[10][b], x[10][b], _mm256_set_pd(-0.3826834324, 0.9238795325, -0.3826834324, 0.9238795325)); \
        CMUL_FMA_R32(x[11][b], x[11][b], _mm256_set_pd(-0.5555702330, 0.8314696123, -0.5555702330, 0.8314696123)); \
        CMUL_FMA_R32(x[12][b], x[12][b], _mm256_set_pd(-0.7071067812, 0.7071067812, -0.7071067812, 0.7071067812)); \
        CMUL_FMA_R32(x[13][b], x[13][b], _mm256_set_pd(-0.8314696123, 0.5555702330, -0.8314696123, 0.5555702330)); \
        CMUL_FMA_R32(x[14][b], x[14][b], _mm256_set_pd(-0.9238795325, 0.3826834324, -0.9238795325, 0.3826834324)); \
        CMUL_FMA_R32(x[15][b], x[15][b], _mm256_set_pd(-0.9807852804, 0.1950903220, -0.9807852804, 0.1950903220)); \
        \
        /* j=2, g=0..7 (lanes 16-23) */ \
        CMUL_FMA_R32(x[16][b], x[16][b], _mm256_set_pd(0.0, 1.0, 0.0, 1.0)); \
        CMUL_FMA_R32(x[17][b], x[17][b], _mm256_set_pd(-0.3826834324, 0.9238795325, -0.3826834324, 0.9238795325)); \
        CMUL_FMA_R32(x[18][b], x[18][b], _mm256_set_pd(-0.7071067812, 0.7071067812, -0.7071067812, 0.7071067812)); \
        CMUL_FMA_R32(x[19][b], x[19][b], _mm256_set_pd(-0.9238795325, 0.3826834324, -0.9238795325, 0.3826834324)); \
        CMUL_FMA_R32(x[20][b], x[20][b], _mm256_set_pd(-1.0, 0.0, -1.0, 0.0)); \
        CMUL_FMA_R32(x[21][b], x[21][b], _mm256_set_pd(-0.9238795325, -0.3826834324, -0.9238795325, -0.3826834324)); \
        CMUL_FMA_R32(x[22][b], x[22][b], _mm256_set_pd(-0.7071067812, -0.7071067812, -0.7071067812, -0.7071067812)); \
        CMUL_FMA_R32(x[23][b], x[23][b], _mm256_set_pd(-0.3826834324, -0.9238795325, -0.3826834324, -0.9238795325)); \
        \
        /* j=3, g=0..7 (lanes 24-31) */ \
        CMUL_FMA_R32(x[24][b], x[24][b], _mm256_set_pd(0.0, 1.0, 0.0, 1.0)); \
        CMUL_FMA_R32(x[25][b], x[25][b], _mm256_set_pd(-0.5555702330, 0.8314696123, -0.5555702330, 0.8314696123)); \
        CMUL_FMA_R32(x[26][b], x[26][b], _mm256_set_pd(-0.9238795325, 0.3826834324, -0.9238795325, 0.3826834324)); \
        CMUL_FMA_R32(x[27][b], x[27][b], _mm256_set_pd(-0.9238795325, -0.3826834324, -0.9238795325, -0.3826834324)); \
        CMUL_FMA_R32(x[28][b], x[28][b], _mm256_set_pd(-0.7071067812, -0.7071067812, -0.7071067812, -0.7071067812)); \
        CMUL_FMA_R32(x[29][b], x[29][b], _mm256_set_pd(-0.1950903220, -0.9807852804, -0.1950903220, -0.9807852804)); \
        CMUL_FMA_R32(x[30][b], x[30][b], _mm256_set_pd(0.3826834324, -0.9238795325, 0.3826834324, -0.9238795325)); \
        CMUL_FMA_R32(x[31][b], x[31][b], _mm256_set_pd(0.8314696123, -0.5555702330, 0.8314696123, -0.5555702330)); \
    } while (0)
#endif

/**
 * @brief Apply W_32 twiddles (INVERSE: exp(+2πi*j*g/32))
 */
#ifdef __AVX2__
#define APPLY_W32_TWIDDLES_BV_AVX2(x) \
    do { \
        /* j=1, g=0..7 (lanes 8-15) - conjugate of forward */ \
        CMUL_FMA_R32(x[8][b], x[8][b], _mm256_set_pd(0.0, 1.0, 0.0, 1.0)); \
        CMUL_FMA_R32(x[9][b], x[9][b], _mm256_set_pd(0.1950903220, 0.9807852804, 0.1950903220, 0.9807852804)); \
        CMUL_FMA_R32(x[10][b], x[10][b], _mm256_set_pd(0.3826834324, 0.9238795325, 0.3826834324, 0.9238795325)); \
        CMUL_FMA_R32(x[11][b], x[11][b], _mm256_set_pd(0.5555702330, 0.8314696123, 0.5555702330, 0.8314696123)); \
        CMUL_FMA_R32(x[12][b], x[12][b], _mm256_set_pd(0.7071067812, 0.7071067812, 0.7071067812, 0.7071067812)); \
        CMUL_FMA_R32(x[13][b], x[13][b], _mm256_set_pd(0.8314696123, 0.5555702330, 0.8314696123, 0.5555702330)); \
        CMUL_FMA_R32(x[14][b], x[14][b], _mm256_set_pd(0.9238795325, 0.3826834324, 0.9238795325, 0.3826834324)); \
        CMUL_FMA_R32(x[15][b], x[15][b], _mm256_set_pd(0.9807852804, 0.1950903220, 0.9807852804, 0.1950903220)); \
        \
        /* j=2, g=0..7 (lanes 16-23) */ \
        CMUL_FMA_R32(x[16][b], x[16][b], _mm256_set_pd(0.0, 1.0, 0.0, 1.0)); \
        CMUL_FMA_R32(x[17][b], x[17][b], _mm256_set_pd(0.3826834324, 0.9238795325, 0.3826834324, 0.9238795325)); \
        CMUL_FMA_R32(x[18][b], x[18][b], _mm256_set_pd(0.7071067812, 0.7071067812, 0.7071067812, 0.7071067812)); \
        CMUL_FMA_R32(x[19][b], x[19][b], _mm256_set_pd(0.9238795325, 0.3826834324, 0.9238795325, 0.3826834324)); \
        CMUL_FMA_R32(x[20][b], x[20][b], _mm256_set_pd(0.0, 1.0, 0.0, 1.0)); \
        CMUL_FMA_R32(x[21][b], x[21][b], _mm256_set_pd(0.9238795325, -0.3826834324, 0.9238795325, -0.3826834324)); \
        CMUL_FMA_R32(x[22][b], x[22][b], _mm256_set_pd(0.7071067812, -0.7071067812, 0.7071067812, -0.7071067812)); \
        CMUL_FMA_R32(x[23][b], x[23][b], _mm256_set_pd(0.3826834324, -0.9238795325, 0.3826834324, -0.9238795325)); \
        \
        /* j=3, g=0..7 (lanes 24-31) */ \
        CMUL_FMA_R32(x[24][b], x[24][b], _mm256_set_pd(0.0, 1.0, 0.0, 1.0)); \
        CMUL_FMA_R32(x[25][b], x[25][b], _mm256_set_pd(0.5555702330, 0.8314696123, 0.5555702330, 0.8314696123)); \
        CMUL_FMA_R32(x[26][b], x[26][b], _mm256_set_pd(0.9238795325, 0.3826834324, 0.9238795325, 0.3826834324)); \
        CMUL_FMA_R32(x[27][b], x[27][b], _mm256_set_pd(0.9238795325, -0.3826834324, 0.9238795325, -0.3826834324)); \
        CMUL_FMA_R32(x[28][b], x[28][b], _mm256_set_pd(0.7071067812, -0.7071067812, 0.7071067812, -0.7071067812)); \
        CMUL_FMA_R32(x[29][b], x[29][b], _mm256_set_pd(0.1950903220, -0.9807852804, 0.1950903220, -0.9807852804)); \
        CMUL_FMA_R32(x[30][b], x[30][b], _mm256_set_pd(-0.3826834324, -0.9238795325, -0.3826834324, -0.9238795325)); \
        CMUL_FMA_R32(x[31][b], x[31][b], _mm256_set_pd(-0.8314696123, -0.5555702330, -0.8314696123, -0.5555702330)); \
    } while (0)
#endif

//==============================================================================
// W_8 TWIDDLES - HARDCODED GEOMETRIC CONSTANTS
//==============================================================================

/**
 * @brief Apply W_8 twiddles (FORWARD)
 * 
 * W_8^1 = (√2/2, -√2/2)
 * W_8^2 = (0, -1) = -i
 * W_8^3 = (-√2/2, -√2/2)
 */
#ifdef __AVX2__
#define APPLY_W8_TWIDDLES_FV_AVX2(o1, o2, o3) \
    do { \
        __m256d w8_1 = _mm256_set_pd(-0.7071067812, 0.7071067812, -0.7071067812, 0.7071067812); \
        CMUL_FMA_R32(o1, o1, w8_1); \
        \
        __m256d w8_2_mask = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0); \
        o2 = _mm256_permute_pd(o2, 0b0101); \
        o2 = _mm256_xor_pd(o2, w8_2_mask); \
        \
        __m256d w8_3 = _mm256_set_pd(-0.7071067812, -0.7071067812, -0.7071067812, -0.7071067812); \
        CMUL_FMA_R32(o3, o3, w8_3); \
    } while (0)
#endif

/**
 * @brief Apply W_8 twiddles (INVERSE)
 * 
 * W_8^1 = (√2/2, +√2/2)
 * W_8^2 = (0, +1) = +i
 * W_8^3 = (-√2/2, +√2/2)
 */
#ifdef __AVX2__
#define APPLY_W8_TWIDDLES_BV_AVX2(o1, o2, o3) \
    do { \
        __m256d w8_1 = _mm256_set_pd(0.7071067812, 0.7071067812, 0.7071067812, 0.7071067812); \
        CMUL_FMA_R32(o1, o1, w8_1); \
        \
        __m256d w8_2_mask = _mm256_set_pd(0.0, -0.0, 0.0, -0.0); \
        o2 = _mm256_permute_pd(o2, 0b0101); \
        o2 = _mm256_xor_pd(o2, w8_2_mask); \
        \
        __m256d w8_3 = _mm256_set_pd(0.7071067812, -0.7071067812, 0.7071067812, -0.7071067812); \
        CMUL_FMA_R32(o3, o3, w8_3); \
    } while (0)
#endif

//==============================================================================
// SCALAR W_8 TWIDDLES
//==============================================================================

/**
 * @brief Apply W_8 twiddles (FORWARD, scalar)
 */
#define APPLY_W8_TWIDDLES_FV_SCALAR(o) \
    do { \
        /* W_8^1 = (√2/2, -√2/2) */ \
        { \
            double r = o[1].re, i = o[1].im; \
            o[1].re = (r + i) * 0.7071067812; \
            o[1].im = (i - r) * 0.7071067812; \
        } \
        \
        /* W_8^2 = -i */ \
        { \
            double r = o[2].re; \
            o[2].re = o[2].im; \
            o[2].im = -r; \
        } \
        \
        /* W_8^3 = (-√2/2, -√2/2) */ \
        { \
            double r = o[3].re, i = o[3].im; \
            o[3].re = (-r + i) * 0.7071067812; \
            o[3].im = (-r - i) * 0.7071067812; \
        } \
    } while (0)

/**
 * @brief Apply W_8 twiddles (INVERSE, scalar)
 */
#define APPLY_W8_TWIDDLES_BV_SCALAR(o) \
    do { \
        /* W_8^1 = (√2/2, +√2/2) */ \
        { \
            double r = o[1].re, i = o[1].im; \
            o[1].re = (r - i) * 0.7071067812; \
            o[1].im = (i + r) * 0.7071067812; \
        } \
        \
        /* W_8^2 = +i */ \
        { \
            double r = o[2].re; \
            o[2].re = -o[2].im; \
            o[2].im = r; \
        } \
        \
        /* W_8^3 = (-√2/2, +√2/2) */ \
        { \
            double r = o[3].re, i = o[3].im; \
            o[3].re = (-r - i) * 0.7071067812; \
            o[3].im = (i - r) * 0.7071067812; \
        } \
    } while (0)

//==============================================================================
// RADIX-8 COMBINE - IDENTICAL for forward/inverse
//==============================================================================

/**
 * @brief Combine even/odd radix-4 results into radix-8 output
 */
#ifdef __AVX2__
#define RADIX8_COMBINE_R32(e0, e1, e2, e3, o0, o1, o2, o3, \
                           x0, x1, x2, x3, x4, x5, x6, x7) \
    do { \
        x0 = _mm256_add_pd(e0, o0); \
        x4 = _mm256_sub_pd(e0, o0); \
        x1 = _mm256_add_pd(e1, o1); \
        x5 = _mm256_sub_pd(e1, o1); \
        x2 = _mm256_add_pd(e2, o2); \
        x6 = _mm256_sub_pd(e2, o2); \
        x3 = _mm256_add_pd(e3, o3); \
        x7 = _mm256_sub_pd(e3, o3); \
    } while (0)
#endif

//==============================================================================
// DATA MOVEMENT - IDENTICAL for forward/inverse
//==============================================================================

#ifdef __AVX2__
/**
 * @brief Load 2 complex values from two locations
 */
#define LOAD_2_COMPLEX_R32(ptr1, ptr2) \
    load2_aos(ptr1, ptr2)

/**
 * @brief Store 2 complex values
 */
#define STORE_2_COMPLEX_R32(ptr, vec) \
    STOREU_PD(&(ptr)->re, vec)

/**
 * @brief Store with streaming
 */
#define STORE_2_COMPLEX_R32_STREAM(ptr, vec) \
    _mm256_stream_pd(&(ptr)->re, vec)
#endif

//==============================================================================
// PREFETCHING - IDENTICAL for forward/inverse
//==============================================================================

#define PREFETCH_L1_R32 8
#define PREFETCH_L2_R32 32
#define PREFETCH_L3_R32 64

#ifdef __AVX2__
#define PREFETCH_32_LANES_R32(k, K, distance, sub_outputs, hint) \
    do { \
        if ((k) + (distance) < K) { \
            for (int _lane = 0; _lane < 32; _lane += 4) { \
                _mm_prefetch((const char *)&sub_outputs[(k)+(distance)+_lane*K], hint); \
            } \
        } \
    } while (0)
#endif

//==============================================================================
// COMPLETE SCALAR RADIX-4 BUTTERFLY
//==============================================================================

/**
 * @brief Complete scalar radix-4 butterfly (forward version)
 */
#define RADIX4_BUTTERFLY_SCALAR_FV_R32(a, b, c, d) \
    do { \
        fft_data sumBD, difBD, sumAC, difAC; \
        RADIX4_BUTTERFLY_CORE_R32_SCALAR(a, b, c, d, sumBD, difBD, sumAC, difAC); \
        \
        fft_data rot; \
        RADIX4_ROTATE_FORWARD_R32_SCALAR(difBD, rot); \
        \
        fft_data y0, y1, y2, y3; \
        RADIX4_ASSEMBLE_OUTPUTS_R32_SCALAR(sumAC, sumBD, difAC, rot, y0, y1, y2, y3); \
        \
        a = y0; \
        b = y1; \
        c = y2; \
        d = y3; \
    } while (0)

/**
 * @brief Complete scalar radix-4 butterfly (inverse version)
 */
#define RADIX4_BUTTERFLY_SCALAR_BV_R32(a, b, c, d) \
    do { \
        fft_data sumBD, difBD, sumAC, difAC; \
        RADIX4_BUTTERFLY_CORE_R32_SCALAR(a, b, c, d, sumBD, difBD, sumAC, difAC); \
        \
        fft_data rot; \
        RADIX4_ROTATE_INVERSE_R32_SCALAR(difBD, rot); \
        \
        fft_data y0, y1, y2, y3; \
        RADIX4_ASSEMBLE_OUTPUTS_R32_SCALAR(sumAC, sumBD, difAC, rot, y0, y1, y2, y3); \
        \
        a = y0; \
        b = y1; \
        c = y2; \
        d = y3; \
    } while (0)

#endif // FFT_RADIX32_MACROS_H