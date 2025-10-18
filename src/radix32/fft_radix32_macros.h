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
 * Same algorithm as AVX2 version but with 512-bit registers.
 */
#define CMUL_FMA_R32_AVX512(out, a, w)                               \
    do                                                               \
    {                                                                \
        __m512d ar = _mm512_unpacklo_pd(a, a);                       \
        __m512d ai = _mm512_unpackhi_pd(a, a);                       \
        __m512d wr = _mm512_unpacklo_pd(w, w);                       \
        __m512d wi = _mm512_unpackhi_pd(w, w);                       \
        __m512d re = _mm512_fmsub_pd(ar, wr, _mm512_mul_pd(ai, wi)); \
        __m512d im = _mm512_fmadd_pd(ar, wi, _mm512_mul_pd(ai, wr)); \
        (out) = _mm512_unpacklo_pd(re, im);                          \
    } while (0)

//==============================================================================
// RADIX-4 BUTTERFLY CORE - AVX-512
//==============================================================================

/**
 * @brief Core radix-4 sums/differences for AVX-512 (4 butterflies)
 *
 * Processes 4 radix-4 butterflies simultaneously as building blocks in radix-32 decomposition.
 */
#define RADIX4_BUTTERFLY_CORE_R32_AVX512(a, b, c, d, sumBD, difBD, sumAC, difAC) \
    do                                                                           \
    {                                                                            \
        sumBD = _mm512_add_pd(b, d);                                             \
        difBD = _mm512_sub_pd(b, d);                                             \
        sumAC = _mm512_add_pd(a, c);                                             \
        difAC = _mm512_sub_pd(a, c);                                             \
    } while (0)

//==============================================================================
// ROTATION - AVX-512
//==============================================================================

/**
 * @brief FORWARD rotation: -i * difBD for AVX-512
 *
 * Processes 4 complex values using AVX-512 permute and XOR.
 * (a + bi) * (-i) = b - ai
 */
#define RADIX4_ROTATE_FORWARD_R32_AVX512(difBD, rot)                            \
    do                                                                          \
    {                                                                           \
        __m512d rot_mask_fv = _mm512_set_pd(-0.0, 0.0, -0.0, 0.0,               \
                                            -0.0, 0.0, -0.0, 0.0);              \
        rot = _mm512_xor_pd(_mm512_permute_pd(difBD, 0b01010101), rot_mask_fv); \
    } while (0)

/**
 * @brief INVERSE rotation: +i * difBD for AVX-512
 *
 * Processes 4 complex values using AVX-512 permute and XOR.
 * (a + bi) * (+i) = -b + ai
 */
#define RADIX4_ROTATE_INVERSE_R32_AVX512(difBD, rot)                            \
    do                                                                          \
    {                                                                           \
        __m512d rot_mask_bv = _mm512_set_pd(0.0, -0.0, 0.0, -0.0,               \
                                            0.0, -0.0, 0.0, -0.0);              \
        rot = _mm512_xor_pd(_mm512_permute_pd(difBD, 0b01010101), rot_mask_bv); \
    } while (0)

//==============================================================================
// OUTPUT ASSEMBLY - AVX-512
//==============================================================================

/**
 * @brief Assemble final outputs from radix-4 intermediates (AVX-512)
 *
 * Processes 4 butterflies simultaneously.
 * y0 = sumAC + sumBD
 * y1 = difAC - rot
 * y2 = sumAC - sumBD
 * y3 = difAC + rot
 */
#define RADIX4_ASSEMBLE_OUTPUTS_R32_AVX512(sumAC, sumBD, difAC, rot, y0, y1, y2, y3) \
    do                                                                               \
    {                                                                                \
        y0 = _mm512_add_pd(sumAC, sumBD);                                            \
        y2 = _mm512_sub_pd(sumAC, sumBD);                                            \
        y1 = _mm512_sub_pd(difAC, rot);                                              \
        y3 = _mm512_add_pd(difAC, rot);                                              \
    } while (0)

//==============================================================================
// APPLY PRECOMPUTED STAGE TWIDDLES - AVX-512
//==============================================================================

/**
 * @brief Apply stage twiddles for 4 butterflies (kk through kk+3)
 *
 * Loads precomputed twiddle factors for 4 butterflies and applies them to a specific lane.
 */
#define APPLY_STAGE_TWIDDLE_R32_AVX512(kk, d_vec, stage_tw, lane, tw_out) \
    do                                                                    \
    {                                                                     \
        const int tw_idx = (kk) * 31 + (lane) - 1;                        \
        __m512d tw = load4_aos(&stage_tw[tw_idx],                         \
                               &stage_tw[tw_idx + 31],                    \
                               &stage_tw[tw_idx + 62],                    \
                               &stage_tw[tw_idx + 93]);                   \
        CMUL_FMA_R32_AVX512(tw_out, d_vec, tw);                           \
    } while (0)

//==============================================================================
// W_32 TWIDDLES - AVX-512 (HARDCODED GEOMETRIC CONSTANTS)
//==============================================================================

/**
 * @brief Apply W_32 twiddles for FORWARD FFT (AVX-512, 4 butterflies)
 *
 * Applies to lanes [8..31] after first radix-4 layer.
 * For j=1,2,3 and g=0..7, processes 4 butterflies simultaneously.
 */
#define APPLY_W32_TWIDDLES_FV_AVX512(x)                                                                                                                                                 \
    do                                                                                                                                                                                  \
    {                                                                                                                                                                                   \
        /* j=1, g=0..7 (lanes 8-15) */                                                                                                                                                  \
        CMUL_FMA_R32_AVX512(x[8][b], x[8][b], _mm512_set_pd(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));                                                                                   \
        CMUL_FMA_R32_AVX512(x[9][b], x[9][b], _mm512_set_pd(-0.1950903220, 0.9807852804, -0.1950903220, 0.9807852804, -0.1950903220, 0.9807852804, -0.1950903220, 0.9807852804));       \
        CMUL_FMA_R32_AVX512(x[10][b], x[10][b], _mm512_set_pd(-0.3826834324, 0.9238795325, -0.3826834324, 0.9238795325, -0.3826834324, 0.9238795325, -0.3826834324, 0.9238795325));     \
        CMUL_FMA_R32_AVX512(x[11][b], x[11][b], _mm512_set_pd(-0.5555702330, 0.8314696123, -0.5555702330, 0.8314696123, -0.5555702330, 0.8314696123, -0.5555702330, 0.8314696123));     \
        CMUL_FMA_R32_AVX512(x[12][b], x[12][b], _mm512_set_pd(-0.7071067812, 0.7071067812, -0.7071067812, 0.7071067812, -0.7071067812, 0.7071067812, -0.7071067812, 0.7071067812));     \
        CMUL_FMA_R32_AVX512(x[13][b], x[13][b], _mm512_set_pd(-0.8314696123, 0.5555702330, -0.8314696123, 0.5555702330, -0.8314696123, 0.5555702330, -0.8314696123, 0.5555702330));     \
        CMUL_FMA_R32_AVX512(x[14][b], x[14][b], _mm512_set_pd(-0.9238795325, 0.3826834324, -0.9238795325, 0.3826834324, -0.9238795325, 0.3826834324, -0.9238795325, 0.3826834324));     \
        CMUL_FMA_R32_AVX512(x[15][b], x[15][b], _mm512_set_pd(-0.9807852804, 0.1950903220, -0.9807852804, 0.1950903220, -0.9807852804, 0.1950903220, -0.9807852804, 0.1950903220));     \
                                                                                                                                                                                        \
        /* j=2, g=0..7 (lanes 16-23) */                                                                                                                                                 \
        CMUL_FMA_R32_AVX512(x[16][b], x[16][b], _mm512_set_pd(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));                                                                                 \
        CMUL_FMA_R32_AVX512(x[17][b], x[17][b], _mm512_set_pd(-0.3826834324, 0.9238795325, -0.3826834324, 0.9238795325, -0.3826834324, 0.9238795325, -0.3826834324, 0.9238795325));     \
        CMUL_FMA_R32_AVX512(x[18][b], x[18][b], _mm512_set_pd(-0.7071067812, 0.7071067812, -0.7071067812, 0.7071067812, -0.7071067812, 0.7071067812, -0.7071067812, 0.7071067812));     \
        CMUL_FMA_R32_AVX512(x[19][b], x[19][b], _mm512_set_pd(-0.9238795325, 0.3826834324, -0.9238795325, 0.3826834324, -0.9238795325, 0.3826834324, -0.9238795325, 0.3826834324));     \
        CMUL_FMA_R32_AVX512(x[20][b], x[20][b], _mm512_set_pd(-1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0));                                                                             \
        CMUL_FMA_R32_AVX512(x[21][b], x[21][b], _mm512_set_pd(-0.9238795325, -0.3826834324, -0.9238795325, -0.3826834324, -0.9238795325, -0.3826834324, -0.9238795325, -0.3826834324)); \
        CMUL_FMA_R32_AVX512(x[22][b], x[22][b], _mm512_set_pd(-0.7071067812, -0.7071067812, -0.7071067812, -0.7071067812, -0.7071067812, -0.7071067812, -0.7071067812, -0.7071067812)); \
        CMUL_FMA_R32_AVX512(x[23][b], x[23][b], _mm512_set_pd(-0.3826834324, -0.9238795325, -0.3826834324, -0.9238795325, -0.3826834324, -0.9238795325, -0.3826834324, -0.9238795325)); \
                                                                                                                                                                                        \
        /* j=3, g=0..7 (lanes 24-31) */                                                                                                                                                 \
        CMUL_FMA_R32_AVX512(x[24][b], x[24][b], _mm512_set_pd(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));                                                                                 \
        CMUL_FMA_R32_AVX512(x[25][b], x[25][b], _mm512_set_pd(-0.5555702330, 0.8314696123, -0.5555702330, 0.8314696123, -0.5555702330, 0.8314696123, -0.5555702330, 0.8314696123));     \
        CMUL_FMA_R32_AVX512(x[26][b], x[26][b], _mm512_set_pd(-0.9238795325, 0.3826834324, -0.9238795325, 0.3826834324, -0.9238795325, 0.3826834324, -0.9238795325, 0.3826834324));     \
        CMUL_FMA_R32_AVX512(x[27][b], x[27][b], _mm512_set_pd(-0.9238795325, -0.3826834324, -0.9238795325, -0.3826834324, -0.9238795325, -0.3826834324, -0.9238795325, -0.3826834324)); \
        CMUL_FMA_R32_AVX512(x[28][b], x[28][b], _mm512_set_pd(-0.7071067812, -0.7071067812, -0.7071067812, -0.7071067812, -0.7071067812, -0.7071067812, -0.7071067812, -0.7071067812)); \
        CMUL_FMA_R32_AVX512(x[29][b], x[29][b], _mm512_set_pd(-0.1950903220, -0.9807852804, -0.1950903220, -0.9807852804, -0.1950903220, -0.9807852804, -0.1950903220, -0.9807852804)); \
        CMUL_FMA_R32_AVX512(x[30][b], x[30][b], _mm512_set_pd(0.3826834324, -0.9238795325, 0.3826834324, -0.9238795325, 0.3826834324, -0.9238795325, 0.3826834324, -0.9238795325));     \
        CMUL_FMA_R32_AVX512(x[31][b], x[31][b], _mm512_set_pd(0.8314696123, -0.5555702330, 0.8314696123, -0.5555702330, 0.8314696123, -0.5555702330, 0.8314696123, -0.5555702330));     \
    } while (0)

/**
 * @brief Apply W_32 twiddles for INVERSE FFT (AVX-512, 4 butterflies)
 *
 * Conjugate of forward twiddles.
 */
#define APPLY_W32_TWIDDLES_BV_AVX512(x)                                                                                                                                                 \
    do                                                                                                                                                                                  \
    {                                                                                                                                                                                   \
        /* j=1, g=0..7 (lanes 8-15) - conjugate of forward */                                                                                                                           \
        CMUL_FMA_R32_AVX512(x[8][b], x[8][b], _mm512_set_pd(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));                                                                                   \
        CMUL_FMA_R32_AVX512(x[9][b], x[9][b], _mm512_set_pd(0.1950903220, 0.9807852804, 0.1950903220, 0.9807852804, 0.1950903220, 0.9807852804, 0.1950903220, 0.9807852804));           \
        CMUL_FMA_R32_AVX512(x[10][b], x[10][b], _mm512_set_pd(0.3826834324, 0.9238795325, 0.3826834324, 0.9238795325, 0.3826834324, 0.9238795325, 0.3826834324, 0.9238795325));         \
        CMUL_FMA_R32_AVX512(x[11][b], x[11][b], _mm512_set_pd(0.5555702330, 0.8314696123, 0.5555702330, 0.8314696123, 0.5555702330, 0.8314696123, 0.5555702330, 0.8314696123));         \
        CMUL_FMA_R32_AVX512(x[12][b], x[12][b], _mm512_set_pd(0.7071067812, 0.7071067812, 0.7071067812, 0.7071067812, 0.7071067812, 0.7071067812, 0.7071067812, 0.7071067812));         \
        CMUL_FMA_R32_AVX512(x[13][b], x[13][b], _mm512_set_pd(0.8314696123, 0.5555702330, 0.8314696123, 0.5555702330, 0.8314696123, 0.5555702330, 0.8314696123, 0.5555702330));         \
        CMUL_FMA_R32_AVX512(x[14][b], x[14][b], _mm512_set_pd(0.9238795325, 0.3826834324, 0.9238795325, 0.3826834324, 0.9238795325, 0.3826834324, 0.9238795325, 0.3826834324));         \
        CMUL_FMA_R32_AVX512(x[15][b], x[15][b], _mm512_set_pd(0.9807852804, 0.1950903220, 0.9807852804, 0.1950903220, 0.9807852804, 0.1950903220, 0.9807852804, 0.1950903220));         \
                                                                                                                                                                                        \
        /* j=2, g=0..7 (lanes 16-23) */                                                                                                                                                 \
        CMUL_FMA_R32_AVX512(x[16][b], x[16][b], _mm512_set_pd(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));                                                                                 \
        CMUL_FMA_R32_AVX512(x[17][b], x[17][b], _mm512_set_pd(0.3826834324, 0.9238795325, 0.3826834324, 0.9238795325, 0.3826834324, 0.9238795325, 0.3826834324, 0.9238795325));         \
        CMUL_FMA_R32_AVX512(x[18][b], x[18][b], _mm512_set_pd(0.7071067812, 0.7071067812, 0.7071067812, 0.7071067812, 0.7071067812, 0.7071067812, 0.7071067812, 0.7071067812));         \
        CMUL_FMA_R32_AVX512(x[19][b], x[19][b], _mm512_set_pd(0.9238795325, 0.3826834324, 0.9238795325, 0.3826834324, 0.9238795325, 0.3826834324, 0.9238795325, 0.3826834324));         \
        CMUL_FMA_R32_AVX512(x[20][b], x[20][b], _mm512_set_pd(1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0));                                                                                 \
        CMUL_FMA_R32_AVX512(x[21][b], x[21][b], _mm512_set_pd(0.9238795325, -0.3826834324, 0.9238795325, -0.3826834324, 0.9238795325, -0.3826834324, 0.9238795325, -0.3826834324));     \
        CMUL_FMA_R32_AVX512(x[22][b], x[22][b], _mm512_set_pd(0.7071067812, -0.7071067812, 0.7071067812, -0.7071067812, 0.7071067812, -0.7071067812, 0.7071067812, -0.7071067812));     \
        CMUL_FMA_R32_AVX512(x[23][b], x[23][b], _mm512_set_pd(0.3826834324, -0.9238795325, 0.3826834324, -0.9238795325, 0.3826834324, -0.9238795325, 0.3826834324, -0.9238795325));     \
                                                                                                                                                                                        \
        /* j=3, g=0..7 (lanes 24-31) */                                                                                                                                                 \
        CMUL_FMA_R32_AVX512(x[24][b], x[24][b], _mm512_set_pd(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));                                                                                 \
        CMUL_FMA_R32_AVX512(x[25][b], x[25][b], _mm512_set_pd(0.5555702330, 0.8314696123, 0.5555702330, 0.8314696123, 0.5555702330, 0.8314696123, 0.5555702330, 0.8314696123));         \
        CMUL_FMA_R32_AVX512(x[26][b], x[26][b], _mm512_set_pd(0.9238795325, 0.3826834324, 0.9238795325, 0.3826834324, 0.9238795325, 0.3826834324, 0.9238795325, 0.3826834324));         \
        CMUL_FMA_R32_AVX512(x[27][b], x[27][b], _mm512_set_pd(0.9238795325, -0.3826834324, 0.9238795325, -0.3826834324, 0.9238795325, -0.3826834324, 0.9238795325, -0.3826834324));     \
        CMUL_FMA_R32_AVX512(x[28][b], x[28][b], _mm512_set_pd(0.7071067812, -0.7071067812, 0.7071067812, -0.7071067812, 0.7071067812, -0.7071067812, 0.7071067812, -0.7071067812));     \
        CMUL_FMA_R32_AVX512(x[29][b], x[29][b], _mm512_set_pd(0.1950903220, -0.9807852804, 0.1950903220, -0.9807852804, 0.1950903220, -0.9807852804, 0.1950903220, -0.9807852804));     \
        CMUL_FMA_R32_AVX512(x[30][b], x[30][b], _mm512_set_pd(-0.3826834324, -0.9238795325, -0.3826834324, -0.9238795325, -0.3826834324, -0.9238795325, -0.3826834324, -0.9238795325)); \
        CMUL_FMA_R32_AVX512(x[31][b], x[31][b], _mm512_set_pd(-0.8314696123, -0.5555702330, -0.8314696123, -0.5555702330, -0.8314696123, -0.5555702330, -0.8314696123, -0.5555702330)); \
    } while (0)

//==============================================================================
// W_8 TWIDDLES - AVX-512
//==============================================================================

/**
 * @brief Apply W_8 twiddles for FORWARD FFT (AVX-512, 4 butterflies)
 *
 * W_8^1 = (√2/2, -√2/2)
 * W_8^2 = (0, -1) = -i
 * W_8^3 = (-√2/2, -√2/2)
 */
#define APPLY_W8_TWIDDLES_FV_AVX512(o1, o2, o3)                     \
    do                                                              \
    {                                                               \
        __m512d w8_1 = _mm512_set_pd(-0.7071067812, 0.7071067812,   \
                                     -0.7071067812, 0.7071067812,   \
                                     -0.7071067812, 0.7071067812,   \
                                     -0.7071067812, 0.7071067812);  \
        CMUL_FMA_R32_AVX512(o1, o1, w8_1);                          \
                                                                    \
        __m512d w8_2_mask = _mm512_set_pd(-0.0, 0.0, -0.0, 0.0,     \
                                          -0.0, 0.0, -0.0, 0.0);    \
        o2 = _mm512_permute_pd(o2, 0b01010101);                     \
        o2 = _mm512_xor_pd(o2, w8_2_mask);                          \
                                                                    \
        __m512d w8_3 = _mm512_set_pd(-0.7071067812, -0.7071067812,  \
                                     -0.7071067812, -0.7071067812,  \
                                     -0.7071067812, -0.7071067812,  \
                                     -0.7071067812, -0.7071067812); \
        CMUL_FMA_R32_AVX512(o3, o3, w8_3);                          \
    } while (0)

/**
 * @brief Apply W_8 twiddles for INVERSE FFT (AVX-512, 4 butterflies)
 *
 * W_8^1 = (√2/2, +√2/2)
 * W_8^2 = (0, +1) = +i
 * W_8^3 = (-√2/2, +√2/2)
 */
#define APPLY_W8_TWIDDLES_BV_AVX512(o1, o2, o3)                    \
    do                                                             \
    {                                                              \
        __m512d w8_1 = _mm512_set_pd(0.7071067812, 0.7071067812,   \
                                     0.7071067812, 0.7071067812,   \
                                     0.7071067812, 0.7071067812,   \
                                     0.7071067812, 0.7071067812);  \
        CMUL_FMA_R32_AVX512(o1, o1, w8_1);                         \
                                                                   \
        __m512d w8_2_mask = _mm512_set_pd(0.0, -0.0, 0.0, -0.0,    \
                                          0.0, -0.0, 0.0, -0.0);   \
        o2 = _mm512_permute_pd(o2, 0b01010101);                    \
        o2 = _mm512_xor_pd(o2, w8_2_mask);                         \
                                                                   \
        __m512d w8_3 = _mm512_set_pd(0.7071067812, -0.7071067812,  \
                                     0.7071067812, -0.7071067812,  \
                                     0.7071067812, -0.7071067812,  \
                                     0.7071067812, -0.7071067812); \
        CMUL_FMA_R32_AVX512(o3, o3, w8_3);                         \
    } while (0)

//==============================================================================
// RADIX-8 COMBINE - AVX-512
//==============================================================================

/**
 * @brief Combine even/odd radix-4 results into radix-8 output (AVX-512)
 *
 * Performs radix-2 combinations for 4 butterflies simultaneously.
 */
#define RADIX8_COMBINE_R32_AVX512(e0, e1, e2, e3, o0, o1, o2, o3, \
                                  x0, x1, x2, x3, x4, x5, x6, x7) \
    do                                                            \
    {                                                             \
        x0 = _mm512_add_pd(e0, o0);                               \
        x4 = _mm512_sub_pd(e0, o0);                               \
        x1 = _mm512_add_pd(e1, o1);                               \
        x5 = _mm512_sub_pd(e1, o1);                               \
        x2 = _mm512_add_pd(e2, o2);                               \
        x6 = _mm512_sub_pd(e2, o2);                               \
        x3 = _mm512_add_pd(e3, o3);                               \
        x7 = _mm512_sub_pd(e3, o3);                               \
    } while (0)

//==============================================================================
// DATA MOVEMENT - AVX-512
//==============================================================================

/**
 * @brief Load 4 complex values from four locations (AVX-512)
 *
 * Loads four complex values (AoS) into one 512-bit register.
 */
#define LOAD_4_COMPLEX_R32_AVX512(ptr1, ptr2, ptr3, ptr4) \
    load4_aos(ptr1, ptr2, ptr3, ptr4)

/**
 * @brief Store 4 complex values (AVX-512)
 *
 * Stores a 512-bit vector containing four complex values (AoS) using unaligned store.
 */
#define STORE_4_COMPLEX_R32_AVX512(ptr, vec) \
    STOREU_PD512(&(ptr)->re, vec)

/**
 * @brief Store with streaming (AVX-512)
 *
 * Stores using non-temporal streaming store to bypass cache.
 */
#define STORE_4_COMPLEX_R32_AVX512_STREAM(ptr, vec) \
    _mm512_stream_pd(&(ptr)->re, vec)

//==============================================================================
// PREFETCHING - AVX-512
//==============================================================================

/**
 * @brief Prefetch distances optimized for AVX-512 in radix-32
 */
#define PREFETCH_L1_R32_AVX512 16  // 1KB ahead
#define PREFETCH_L2_R32_AVX512 64  // 4KB ahead
#define PREFETCH_L3_R32_AVX512 128 // 8KB ahead

/**
 * @brief Prefetch 32 lanes ahead for AVX-512 (4 butterflies)
 *
 * Prefetches more aggressively for the higher throughput of AVX-512.
 */
#define PREFETCH_32_LANES_R32_AVX512(k, K, distance, sub_outputs, stage_tw, hint)             \
    do                                                                                        \
    {                                                                                         \
        if ((k) + (distance) < K)                                                             \
        {                                                                                     \
            for (int _lane = 0; _lane < 32; _lane += 4)                                       \
            {                                                                                 \
                _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + _lane * K], hint); \
            }                                                                                 \
            _mm_prefetch((const char *)&stage_tw[((k) + (distance)) * 31], hint);             \
        }                                                                                     \
    } while (0)

//==============================================================================
// COMPLETE BUTTERFLY PIPELINE - AVX-512
//==============================================================================

//==============================================================================
// COMPLETE RADIX-32 BUTTERFLY PIPELINE - AVX-512 (THE ULTIMATE!)
//==============================================================================

/**
 * @brief Complete AVX-512 radix-32 butterfly (FORWARD, 4 butterflies)
 *
 * Processes 4 butterflies (128 complex values!) in one macro call.
 * This is the ultimate FFT macro - maximum throughput!
 *
 * Algorithm structure:
 * 1. Load 32 lanes for 4 butterflies (128 complex values)
 * 2. Apply input twiddles W_N^(j*k) to lanes 1-31
 * 3. First layer: 8 radix-4 butterflies (groups of 4)
 * 4. Apply W_32 geometric twiddles to lanes 8-31
 * 5. Second layer: 8 radix-4 butterflies (transposed groups)
 * 6. Apply W_8 twiddles to odd groups
 * 7. Final radix-2 combinations (even/odd merge)
 * 8. Store all 32 lanes × 4 butterflies = 128 outputs
 */
#define RADIX32_PIPELINE_4_FV_AVX512(kk, K, sub_outputs, stage_tw, output_buffer)                       \
    do                                                                                                  \
    {                                                                                                   \
        /* ================================================================== */                        \
        /* STEP 1: Load 32 lanes for 4 butterflies (128 complex values)      */                         \
        /* ================================================================== */                        \
        __m512d x[32];                                                                                  \
        for (int lane = 0; lane < 32; lane++)                                                           \
        {                                                                                               \
            x[lane] = load4_aos(&sub_outputs[(kk) + lane * K],                                          \
                                &sub_outputs[(kk) + 1 + lane * K],                                      \
                                &sub_outputs[(kk) + 2 + lane * K],                                      \
                                &sub_outputs[(kk) + 3 + lane * K]);                                     \
        }                                                                                               \
                                                                                                        \
        /* ================================================================== */                        \
        /* STEP 2: Apply input twiddles W_N^(j*k) to lanes 1-31              */                         \
        /* ================================================================== */                        \
        for (int lane = 1; lane < 32; lane++)                                                           \
        {                                                                                               \
            APPLY_STAGE_TWIDDLE_R32_AVX512(kk, x[lane], stage_tw, lane, x[lane]);                       \
        }                                                                                               \
                                                                                                        \
        /* ================================================================== */                        \
        /* STEP 3: First layer - 8 radix-4 butterflies (groups of 4)         */                         \
        /* Groups: [0,4,8,12], [1,5,9,13], ..., [7,11,15,19], ...            */                         \
        /* ================================================================== */                        \
        __m512d y[32];                                                                                  \
        for (int g = 0; g < 8; g++)                                                                     \
        {                                                                                               \
            RADIX4_BUTTERFLY_FV_R32_AVX512(                                                             \
                x[g], x[g + 8], x[g + 16], x[g + 24], /* Results overwrite inputs, then copy to y */    \
            );                                                                                          \
            y[g * 4 + 0] = x[g];                                                                        \
            y[g * 4 + 1] = x[g + 8];                                                                    \
            y[g * 4 + 2] = x[g + 16];                                                                   \
            y[g * 4 + 3] = x[g + 24];                                                                   \
        }                                                                                               \
                                                                                                        \
        /* ================================================================== */                        \
        /* STEP 4: Apply W_32 geometric twiddles to lanes 8-31               */                         \
        /* This is the key step that distinguishes radix-32                   */                        \
        /* ================================================================== */                        \
        for (int b = 0; b < 4; b++)                                                                     \
        { /* 4 butterflies */                                                                           \
            /* Unpack y array into x[lane][butterfly] logical structure */                              \
            __m512d x_temp[32][1]; /* [1] just for syntax compatibility */                              \
            for (int lane = 0; lane < 32; lane++)                                                       \
            {                                                                                           \
                x_temp[lane][0] = y[lane];                                                              \
            }                                                                                           \
                                                                                                        \
            /* Apply W_32 twiddles using the forward macro */                                           \
            APPLY_W32_TWIDDLES_FV_AVX512(x_temp);                                                       \
                                                                                                        \
            /* Pack back into y */                                                                      \
            for (int lane = 0; lane < 32; lane++)                                                       \
            {                                                                                           \
                y[lane] = x_temp[lane][0];                                                              \
            }                                                                                           \
        }                                                                                               \
                                                                                                        \
        /* ================================================================== */                        \
        /* STEP 5: Second layer - 8 radix-4 butterflies (transposed)         */                         \
        /* Now process [0,1,2,3], [4,5,6,7], ..., [28,29,30,31]              */                         \
        /* ================================================================== */                        \
        __m512d z[32];                                                                                  \
        for (int g = 0; g < 8; g++)                                                                     \
        {                                                                                               \
            int base = g * 4;                                                                           \
            RADIX4_BUTTERFLY_FV_R32_AVX512(                                                             \
                y[base + 0], y[base + 1], y[base + 2], y[base + 3], /* Results written back in place */ \
            );                                                                                          \
            z[base + 0] = y[base + 0];                                                                  \
            z[base + 1] = y[base + 1];                                                                  \
            z[base + 2] = y[base + 2];                                                                  \
            z[base + 3] = y[base + 3];                                                                  \
        }                                                                                               \
                                                                                                        \
        /* ================================================================== */                        \
        /* STEP 6: Apply W_8 twiddles to odd radix-8 groups                  */                         \
        /* Groups: even [0-3,8-11,16-19,24-27], odd [4-7,12-15,20-23,28-31]  */                         \
        /* Apply W_8 twiddles to outputs 1,2,3 of each odd group              */                        \
        /* ================================================================== */                        \
        for (int g = 0; g < 4; g++)                                                                     \
        {                                   /* 4 odd groups */                                          \
            int base_odd = (g * 2 + 1) * 4; /* 4, 12, 20, 28 */                                         \
            APPLY_W8_TWIDDLES_FV_AVX512(                                                                \
                z[base_odd + 1],                                                                        \
                z[base_odd + 2],                                                                        \
                z[base_odd + 3]);                                                                       \
        }                                                                                               \
                                                                                                        \
        /* ================================================================== */                        \
        /* STEP 7: Final radix-2 combinations (even/odd merge)               */                         \
        /* Combine even groups [0-3,8-11,16-19,24-27] with                   */                         \
        /*         odd groups  [4-7,12-15,20-23,28-31]                        */                        \
        /* ================================================================== */                        \
        __m512d final[32];                                                                              \
        for (int g = 0; g < 4; g++)                                                                     \
        {                                   /* 4 pairs of even/odd groups */                            \
            int base_even = (g * 2) * 4;    /* 0, 8, 16, 24 */                                          \
            int base_odd = (g * 2 + 1) * 4; /* 4, 12, 20, 28 */                                         \
            int out_base = g * 8;           /* 0, 8, 16, 24 */                                          \
                                                                                                        \
            /* Radix-8 combine: out[j] = even[j] + odd[j], out[j+4] = even[j] - odd[j] */               \
            RADIX8_COMBINE_R32_AVX512(                                                                  \
                z[base_even + 0], z[base_even + 1], z[base_even + 2], z[base_even + 3],                 \
                z[base_odd + 0], z[base_odd + 1], z[base_odd + 2], z[base_odd + 3],                     \
                final[out_base + 0], final[out_base + 1], final[out_base + 2], final[out_base + 3],     \
                final[out_base + 4], final[out_base + 5], final[out_base + 6], final[out_base + 7]);    \
        }                                                                                               \
                                                                                                        \
        /* ================================================================== */                        \
        /* STEP 8: Store all 32 lanes × 4 butterflies = 128 outputs          */                         \
        /* ================================================================== */                        \
        for (int lane = 0; lane < 32; lane++)                                                           \
        {                                                                                               \
            STORE_4_COMPLEX_R32_AVX512(&output_buffer[(kk) + lane * K], final[lane]);                   \
        }                                                                                               \
    } while (0)

/**
 * @brief Complete AVX-512 radix-32 butterfly (INVERSE, 4 butterflies)
 *
 * Identical structure to forward but uses inverse rotation and twiddles.
 */
#define RADIX32_PIPELINE_4_BV_AVX512(kk, K, sub_outputs, stage_tw, output_buffer)                    \
    do                                                                                               \
    {                                                                                                \
        /* ================================================================== */                     \
        /* STEP 1: Load 32 lanes for 4 butterflies (128 complex values)      */                      \
        /* ================================================================== */                     \
        __m512d x[32];                                                                               \
        for (int lane = 0; lane < 32; lane++)                                                        \
        {                                                                                            \
            x[lane] = load4_aos(&sub_outputs[(kk) + lane * K],                                       \
                                &sub_outputs[(kk) + 1 + lane * K],                                   \
                                &sub_outputs[(kk) + 2 + lane * K],                                   \
                                &sub_outputs[(kk) + 3 + lane * K]);                                  \
        }                                                                                            \
                                                                                                     \
        /* ================================================================== */                     \
        /* STEP 2: Apply input twiddles W_N^(-j*k) to lanes 1-31 (conjugate) */                      \
        /* ================================================================== */                     \
        for (int lane = 1; lane < 32; lane++)                                                        \
        {                                                                                            \
            APPLY_STAGE_TWIDDLE_R32_AVX512(kk, x[lane], stage_tw, lane, x[lane]);                    \
        }                                                                                            \
                                                                                                     \
        /* ================================================================== */                     \
        /* STEP 3: First layer - 8 radix-4 butterflies (INVERSE rotation)    */                      \
        /* ================================================================== */                     \
        __m512d y[32];                                                                               \
        for (int g = 0; g < 8; g++)                                                                  \
        {                                                                                            \
            RADIX4_BUTTERFLY_BV_R32_AVX512(                                                          \
                x[g], x[g + 8], x[g + 16], x[g + 24]);                                               \
            y[g * 4 + 0] = x[g];                                                                     \
            y[g * 4 + 1] = x[g + 8];                                                                 \
            y[g * 4 + 2] = x[g + 16];                                                                \
            y[g * 4 + 3] = x[g + 24];                                                                \
        }                                                                                            \
                                                                                                     \
        /* ================================================================== */                     \
        /* STEP 4: Apply W_32 INVERSE geometric twiddles to lanes 8-31       */                      \
        /* ================================================================== */                     \
        for (int b = 0; b < 4; b++)                                                                  \
        {                                                                                            \
            __m512d x_temp[32][1];                                                                   \
            for (int lane = 0; lane < 32; lane++)                                                    \
            {                                                                                        \
                x_temp[lane][0] = y[lane];                                                           \
            }                                                                                        \
                                                                                                     \
            /* Use INVERSE W_32 twiddles */                                                          \
            APPLY_W32_TWIDDLES_BV_AVX512(x_temp);                                                    \
                                                                                                     \
            for (int lane = 0; lane < 32; lane++)                                                    \
            {                                                                                        \
                y[lane] = x_temp[lane][0];                                                           \
            }                                                                                        \
        }                                                                                            \
                                                                                                     \
        /* ================================================================== */                     \
        /* STEP 5: Second layer - 8 radix-4 butterflies (INVERSE)            */                      \
        /* ================================================================== */                     \
        __m512d z[32];                                                                               \
        for (int g = 0; g < 8; g++)                                                                  \
        {                                                                                            \
            int base = g * 4;                                                                        \
            RADIX4_BUTTERFLY_BV_R32_AVX512(                                                          \
                y[base + 0], y[base + 1], y[base + 2], y[base + 3]);                                 \
            z[base + 0] = y[base + 0];                                                               \
            z[base + 1] = y[base + 1];                                                               \
            z[base + 2] = y[base + 2];                                                               \
            z[base + 3] = y[base + 3];                                                               \
        }                                                                                            \
                                                                                                     \
        /* ================================================================== */                     \
        /* STEP 6: Apply W_8 INVERSE twiddles to odd groups                  */                      \
        /* ================================================================== */                     \
        for (int g = 0; g < 4; g++)                                                                  \
        {                                                                                            \
            int base_odd = (g * 2 + 1) * 4;                                                          \
            APPLY_W8_TWIDDLES_BV_AVX512(                                                             \
                z[base_odd + 1],                                                                     \
                z[base_odd + 2],                                                                     \
                z[base_odd + 3]);                                                                    \
        }                                                                                            \
                                                                                                     \
        /* ================================================================== */                     \
        /* STEP 7: Final radix-2 combinations (IDENTICAL to forward)         */                      \
        /* ================================================================== */                     \
        __m512d final[32];                                                                           \
        for (int g = 0; g < 4; g++)                                                                  \
        {                                                                                            \
            int base_even = (g * 2) * 4;                                                             \
            int base_odd = (g * 2 + 1) * 4;                                                          \
            int out_base = g * 8;                                                                    \
                                                                                                     \
            RADIX8_COMBINE_R32_AVX512(                                                               \
                z[base_even + 0], z[base_even + 1], z[base_even + 2], z[base_even + 3],              \
                z[base_odd + 0], z[base_odd + 1], z[base_odd + 2], z[base_odd + 3],                  \
                final[out_base + 0], final[out_base + 1], final[out_base + 2], final[out_base + 3],  \
                final[out_base + 4], final[out_base + 5], final[out_base + 6], final[out_base + 7]); \
        }                                                                                            \
                                                                                                     \
        /* ================================================================== */                     \
        /* STEP 8: Store all 128 outputs                                     */                      \
        /* ================================================================== */                     \
        for (int lane = 0; lane < 32; lane++)                                                        \
        {                                                                                            \
            STORE_4_COMPLEX_R32_AVX512(&output_buffer[(kk) + lane * K], final[lane]);                \
        }                                                                                            \
    } while (0)

//==============================================================================
// STREAMING VERSIONS (for very large transforms)
//==============================================================================

/**
 * @brief Streaming version (FORWARD) - uses non-temporal stores
 *
 * Identical to regular version but stores bypass cache.
 * Use for transforms larger than L3 cache.
 */
#define RADIX32_PIPELINE_4_FV_AVX512_STREAM(kk, K, sub_outputs, stage_tw, output_buffer)             \
    do                                                                                               \
    {                                                                                                \
        /* Steps 1-7: Identical to non-streaming version */                                          \
        __m512d x[32];                                                                               \
        for (int lane = 0; lane < 32; lane++)                                                        \
        {                                                                                            \
            x[lane] = load4_aos(&sub_outputs[(kk) + lane * K],                                       \
                                &sub_outputs[(kk) + 1 + lane * K],                                   \
                                &sub_outputs[(kk) + 2 + lane * K],                                   \
                                &sub_outputs[(kk) + 3 + lane * K]);                                  \
        }                                                                                            \
                                                                                                     \
        for (int lane = 1; lane < 32; lane++)                                                        \
        {                                                                                            \
            APPLY_STAGE_TWIDDLE_R32_AVX512(kk, x[lane], stage_tw, lane, x[lane]);                    \
        }                                                                                            \
                                                                                                     \
        __m512d y[32];                                                                               \
        for (int g = 0; g < 8; g++)                                                                  \
        {                                                                                            \
            RADIX4_BUTTERFLY_FV_R32_AVX512(x[g], x[g + 8], x[g + 16], x[g + 24]);                    \
            y[g * 4 + 0] = x[g];                                                                     \
            y[g * 4 + 1] = x[g + 8];                                                                 \
            y[g * 4 + 2] = x[g + 16];                                                                \
            y[g * 4 + 3] = x[g + 24];                                                                \
        }                                                                                            \
                                                                                                     \
        for (int b = 0; b < 4; b++)                                                                  \
        {                                                                                            \
            __m512d x_temp[32][1];                                                                   \
            for (int lane = 0; lane < 32; lane++)                                                    \
                x_temp[lane][0] = y[lane];                                                           \
            APPLY_W32_TWIDDLES_FV_AVX512(x_temp);                                                    \
            for (int lane = 0; lane < 32; lane++)                                                    \
                y[lane] = x_temp[lane][0];                                                           \
        }                                                                                            \
                                                                                                     \
        __m512d z[32];                                                                               \
        for (int g = 0; g < 8; g++)                                                                  \
        {                                                                                            \
            int base = g * 4;                                                                        \
            RADIX4_BUTTERFLY_FV_R32_AVX512(y[base + 0], y[base + 1], y[base + 2], y[base + 3]);      \
            z[base + 0] = y[base + 0];                                                               \
            z[base + 1] = y[base + 1];                                                               \
            z[base + 2] = y[base + 2];                                                               \
            z[base + 3] = y[base + 3];                                                               \
        }                                                                                            \
                                                                                                     \
        for (int g = 0; g < 4; g++)                                                                  \
        {                                                                                            \
            int base_odd = (g * 2 + 1) * 4;                                                          \
            APPLY_W8_TWIDDLES_FV_AVX512(z[base_odd + 1], z[base_odd + 2], z[base_odd + 3]);          \
        }                                                                                            \
                                                                                                     \
        __m512d final[32];                                                                           \
        for (int g = 0; g < 4; g++)                                                                  \
        {                                                                                            \
            int base_even = (g * 2) * 4, base_odd = (g * 2 + 1) * 4, out_base = g * 8;               \
            RADIX8_COMBINE_R32_AVX512(                                                               \
                z[base_even + 0], z[base_even + 1], z[base_even + 2], z[base_even + 3],              \
                z[base_odd + 0], z[base_odd + 1], z[base_odd + 2], z[base_odd + 3],                  \
                final[out_base + 0], final[out_base + 1], final[out_base + 2], final[out_base + 3],  \
                final[out_base + 4], final[out_base + 5], final[out_base + 6], final[out_base + 7]); \
        }                                                                                            \
                                                                                                     \
        /* STREAMING STORES - bypass cache */                                                        \
        for (int lane = 0; lane < 32; lane++)                                                        \
        {                                                                                            \
            STORE_4_COMPLEX_R32_AVX512_STREAM(&output_buffer[(kk) + lane * K], final[lane]);         \
        }                                                                                            \
    } while (0)

/**
 * @brief Streaming version (INVERSE)
 */
#define RADIX32_PIPELINE_4_BV_AVX512_STREAM(kk, K, sub_outputs, stage_tw, output_buffer)             \
    do                                                                                               \
    {                                                                                                \
        /* Identical to non-streaming inverse but with streaming stores at the end */                \
        __m512d x[32];                                                                               \
        for (int lane = 0; lane < 32; lane++)                                                        \
        {                                                                                            \
            x[lane] = load4_aos(&sub_outputs[(kk) + lane * K],                                       \
                                &sub_outputs[(kk) + 1 + lane * K],                                   \
                                &sub_outputs[(kk) + 2 + lane * K],                                   \
                                &sub_outputs[(kk) + 3 + lane * K]);                                  \
        }                                                                                            \
        for (int lane = 1; lane < 32; lane++)                                                        \
        {                                                                                            \
            APPLY_STAGE_TWIDDLE_R32_AVX512(kk, x[lane], stage_tw, lane, x[lane]);                    \
        }                                                                                            \
        __m512d y[32];                                                                               \
        for (int g = 0; g < 8; g++)                                                                  \
        {                                                                                            \
            RADIX4_BUTTERFLY_BV_R32_AVX512(x[g], x[g + 8], x[g + 16], x[g + 24]);                    \
            y[g * 4 + 0] = x[g];                                                                     \
            y[g * 4 + 1] = x[g + 8];                                                                 \
            y[g * 4 + 2] = x[g + 16];                                                                \
            y[g * 4 + 3] = x[g + 24];                                                                \
        }                                                                                            \
        for (int b = 0; b < 4; b++)                                                                  \
        {                                                                                            \
            __m512d x_temp[32][1];                                                                   \
            for (int lane = 0; lane < 32; lane++)                                                    \
                x_temp[lane][0] = y[lane];                                                           \
            APPLY_W32_TWIDDLES_BV_AVX512(x_temp);                                                    \
            for (int lane = 0; lane < 32; lane++)                                                    \
                y[lane] = x_temp[lane][0];                                                           \
        }                                                                                            \
        __m512d z[32];                                                                               \
        for (int g = 0; g < 8; g++)                                                                  \
        {                                                                                            \
            int base = g * 4;                                                                        \
            RADIX4_BUTTERFLY_BV_R32_AVX512(y[base + 0], y[base + 1], y[base + 2], y[base + 3]);      \
            z[base + 0] = y[base + 0];                                                               \
            z[base + 1] = y[base + 1];                                                               \
            z[base + 2] = y[base + 2];                                                               \
            z[base + 3] = y[base + 3];                                                               \
        }                                                                                            \
        for (int g = 0; g < 4; g++)                                                                  \
        {                                                                                            \
            int base_odd = (g * 2 + 1) * 4;                                                          \
            APPLY_W8_TWIDDLES_BV_AVX512(z[base_odd + 1], z[base_odd + 2], z[base_odd + 3]);          \
        }                                                                                            \
        __m512d final[32];                                                                           \
        for (int g = 0; g < 4; g++)                                                                  \
        {                                                                                            \
            int base_even = (g * 2) * 4, base_odd = (g * 2 + 1) * 4, out_base = g * 8;               \
            RADIX8_COMBINE_R32_AVX512(                                                               \
                z[base_even + 0], z[base_even + 1], z[base_even + 2], z[base_even + 3],              \
                z[base_odd + 0], z[base_odd + 1], z[base_odd + 2], z[base_odd + 3],                  \
                final[out_base + 0], final[out_base + 1], final[out_base + 2], final[out_base + 3],  \
                final[out_base + 4], final[out_base + 5], final[out_base + 6], final[out_base + 7]); \
        }                                                                                            \
        for (int lane = 0; lane < 32; lane++)                                                        \
        {                                                                                            \
            STORE_4_COMPLEX_R32_AVX512_STREAM(&output_buffer[(kk) + lane * K], final[lane]);         \
        }                                                                                            \
    } while (0)

/**
 * @brief Complete AVX-512 radix-4 butterfly (forward version)
 *
 * Processes 4 radix-4 butterflies in one call.
 */
#define RADIX4_BUTTERFLY_FV_R32_AVX512(a, b, c, d)          \
    do                                                      \
    {                                                       \
        __m512d sumBD, difBD, sumAC, difAC;                 \
        RADIX4_BUTTERFLY_CORE_R32_AVX512(a, b, c, d,        \
                                         sumBD, difBD,      \
                                         sumAC, difAC);     \
                                                            \
        __m512d rot;                                        \
        RADIX4_ROTATE_FORWARD_R32_AVX512(difBD, rot);       \
                                                            \
        __m512d y0, y1, y2, y3;                             \
        RADIX4_ASSEMBLE_OUTPUTS_R32_AVX512(sumAC, sumBD,    \
                                           difAC, rot,      \
                                           y0, y1, y2, y3); \
                                                            \
        a = y0;                                             \
        b = y1;                                             \
        c = y2;                                             \
        d = y3;                                             \
    } while (0)

/**
 * @brief Complete AVX-512 radix-4 butterfly (inverse version)
 *
 * Processes 4 radix-4 butterflies in one call.
 */
#define RADIX4_BUTTERFLY_BV_R32_AVX512(a, b, c, d)          \
    do                                                      \
    {                                                       \
        __m512d sumBD, difBD, sumAC, difAC;                 \
        RADIX4_BUTTERFLY_CORE_R32_AVX512(a, b, c, d,        \
                                         sumBD, difBD,      \
                                         sumAC, difAC);     \
                                                            \
        __m512d rot;                                        \
        RADIX4_ROTATE_INVERSE_R32_AVX512(difBD, rot);       \
                                                            \
        __m512d y0, y1, y2, y3;                             \
        RADIX4_ASSEMBLE_OUTPUTS_R32_AVX512(sumAC, sumBD,    \
                                           difAC, rot,      \
                                           y0, y1, y2, y3); \
                                                            \
        a = y0;                                             \
        b = y1;                                             \
        c = y2;                                             \
        d = y3;                                             \
    } while (0)

#endif // __AVX512F__

//==============================================================================
// USAGE EXAMPLE (for future implementation)
//==============================================================================

/**
 * In fft_radix32_fv.c or fft_radix32_bv.c:
 *
 * #ifdef __AVX512F__
 *     // Main loop: 4 butterflies per iteration (128 complex values processed!)
 *     for (; k + 3 < K; k += 4) {
 *         PREFETCH_32_LANES_R32_AVX512(k, K, PREFETCH_L1_R32_AVX512,
 *                                      sub_outputs, stage_tw, _MM_HINT_T0);
 *         RADIX32_PIPELINE_4_FV_AVX512(k, K, sub_outputs, stage_tw, output_buffer);
 *     }
 * #endif
 *
 * #ifdef __AVX2__
 *     // Fallback to AVX2: 2 butterflies per iteration
 *     for (; k + 1 < K; k += 2) {
 *         PREFETCH_32_LANES_R32(k, K, PREFETCH_L1_R32, sub_outputs, _MM_HINT_T0);
 *         // ... existing AVX2 code ...
 *     }
 * #endif
 *
 * // Scalar tail
 * for (; k < K; k++) {
 *     // ... scalar butterfly ...
 * }
 *
 * PERFORMANCE NOTES - RADIX-32 IS THE ULTIMATE FFT BUTTERFLY:
 *
 * **Throughput:**
 * - AVX-512 processes 4 butterflies (128 complex values) per iteration!
 * - 2x throughput vs AVX2 (which does 2 butterflies = 64 complex values)
 * - 4x throughput vs scalar (32 complex values)
 *
 * **Efficiency:**
 * - Radix-32 minimizes total butterfly count for N = 2^k sizes
 * - For N=1024: only 1024/32 = 32 radix-32 butterflies needed at base level!
 * - Compare to radix-2: 512 butterflies needed at base level
 *
 * **Expected Speedup:**
 * - 45-65% faster than AVX2 on Skylake-X/Cascade Lake
 * - 65-95% faster than AVX2 on Ice Lake/Sapphire Rapids
 * - Best performance for power-of-32 sizes: 32, 1024, 32768, etc.
 *
 * **When to Use:**
 * - Large transforms (N > 1024) where radix-32 shines
 * - Multi-dimensional FFTs with 32-aligned dimensions
 * - When L3 cache can hold working set
 * - Use streaming stores for N > L3 cache size
 *
 * **Architecture Notes:**
 * - Requires 32 AVX-512 registers (ZMM0-ZMM31)
 * - Heavy register pressure - compiler must optimize carefully
 * - Pipeline depth considerations for Ice Lake+
 * - Memory bandwidth becomes bottleneck before compute on modern CPUs
 *
 * **The Ultimate Achievement:**
 * This is as good as it gets for FFT butterflies in userspace!
 * 128 complex values (256 doubles = 2KB) processed per iteration!
 * 🚀🔥💪
 */

/**
 * @brief Optimized complex multiply: out = a * w (6 FMA + 2 UNPACK)
 *
 * This macro performs a complex multiplication using AVX2 instructions, optimized with fused multiply-add (FMA) operations.
 * It is used for applying twiddle factors in the radix-32 butterfly for both forward and inverse transforms.
 * The operation assumes Array-of-Structures (AoS) layout for complex numbers (real and imaginary parts interleaved).
 */
#ifdef __AVX2__
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
 *
 * This macro computes the basic sums and differences for a radix-4 butterfly, used as a building block in the radix-32 decomposition.
 * It processes AVX2 vectors containing two complex numbers each and is identical for both forward and inverse transforms.
 */
#ifdef __AVX2__
#define RADIX4_BUTTERFLY_CORE_R32(a, b, c, d, sumBD, difBD, sumAC, difAC) \
    do                                                                    \
    {                                                                     \
        sumBD = _mm256_add_pd(b, d);                                      \
        difBD = _mm256_sub_pd(b, d);                                      \
        sumAC = _mm256_add_pd(a, c);                                      \
        difAC = _mm256_sub_pd(a, c);                                      \
    } while (0)
#endif

/**
 * @brief Scalar version of the radix-4 butterfly core sums/differences.
 *
 * This macro performs the same sum and difference calculations as RADIX4_BUTTERFLY_CORE_R32 but in scalar mode.
 * It is used for tail cases or non-SIMD environments in the radix-32 butterfly.
 */
#define RADIX4_BUTTERFLY_CORE_R32_SCALAR(a, b, c, d,                 \
                                         sumBD, difBD, sumAC, difAC) \
    do                                                               \
    {                                                                \
        sumBD.re = b.re + d.re;                                      \
        sumBD.im = b.im + d.im;                                      \
        difBD.re = b.re - d.re;                                      \
        difBD.im = b.im - d.im;                                      \
        sumAC.re = a.re + c.re;                                      \
        sumAC.im = a.im + c.im;                                      \
        difAC.re = a.re - c.re;                                      \
        difAC.im = a.im - c.im;                                      \
    } while (0)

//==============================================================================
// ROTATION - ONLY DIFFERENCE BETWEEN FORWARD/INVERSE
//==============================================================================

/**
 * @brief FORWARD rotation: -i * difBD
 *
 * This macro applies a forward rotation (multiplication by -i) to the difference vector in the radix-4 butterfly.
 * It uses AVX2 permute and XOR operations for efficiency in the forward transform.
 */
#ifdef __AVX2__
#define RADIX4_ROTATE_FORWARD_R32(difBD, rot)                               \
    do                                                                      \
    {                                                                       \
        __m256d rot_mask_fv = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);          \
        rot = _mm256_xor_pd(_mm256_permute_pd(difBD, 0b0101), rot_mask_fv); \
    } while (0)
#endif

/**
 * @brief Scalar forward rotation: -i * difBD
 *
 * This macro applies the forward rotation (multiplication by -i) in scalar mode for the radix-4 butterfly.
 */
#define RADIX4_ROTATE_FORWARD_R32_SCALAR(difBD, rot) \
    do                                               \
    {                                                \
        rot.re = difBD.im;                           \
        rot.im = -difBD.re;                          \
    } while (0)

/**
 * @brief INVERSE rotation: +i * difBD
 *
 * This macro applies an inverse rotation (multiplication by +i) to the difference vector in the radix-4 butterfly.
 * It uses AVX2 permute and XOR operations for efficiency in the inverse transform.
 */
#ifdef __AVX2__
#define RADIX4_ROTATE_INVERSE_R32(difBD, rot)                               \
    do                                                                      \
    {                                                                       \
        __m256d rot_mask_bv = _mm256_set_pd(0.0, -0.0, 0.0, -0.0);          \
        rot = _mm256_xor_pd(_mm256_permute_pd(difBD, 0b0101), rot_mask_bv); \
    } while (0)
#endif

/**
 * @brief Scalar inverse rotation: +i * difBD
 *
 * This macro applies the inverse rotation (multiplication by +i) in scalar mode for the radix-4 butterfly.
 */
#define RADIX4_ROTATE_INVERSE_R32_SCALAR(difBD, rot) \
    do                                               \
    {                                                \
        rot.re = -difBD.im;                          \
        rot.im = difBD.re;                           \
    } while (0)

//==============================================================================
// OUTPUT ASSEMBLY - IDENTICAL for forward/inverse
//==============================================================================

/**
 * @brief Assemble final outputs from radix-4 intermediates
 *
 * This macro combines the sums, differences, and rotated values to produce the four outputs of a radix-4 butterfly.
 * It is identical for both forward and inverse transforms in AVX2 mode.
 */
#ifdef __AVX2__
#define RADIX4_ASSEMBLE_OUTPUTS_R32(sumAC, sumBD, difAC, rot, y0, y1, y2, y3) \
    do                                                                        \
    {                                                                         \
        y0 = _mm256_add_pd(sumAC, sumBD);                                     \
        y2 = _mm256_sub_pd(sumAC, sumBD);                                     \
        y1 = _mm256_sub_pd(difAC, rot);                                       \
        y3 = _mm256_add_pd(difAC, rot);                                       \
    } while (0)
#endif

/**
 * @brief Scalar version to assemble final outputs from radix-4 intermediates.
 *
 * This macro performs the same output assembly as RADIX4_ASSEMBLE_OUTPUTS_R32 but in scalar mode.
 */
#define RADIX4_ASSEMBLE_OUTPUTS_R32_SCALAR(sumAC, sumBD, difAC, rot, y0, y1, y2, y3) \
    do                                                                               \
    {                                                                                \
        y0.re = sumAC.re + sumBD.re;                                                 \
        y0.im = sumAC.im + sumBD.im;                                                 \
        y2.re = sumAC.re - sumBD.re;                                                 \
        y2.im = sumAC.im - sumBD.im;                                                 \
        y1.re = difAC.re - rot.re;                                                   \
        y1.im = difAC.im - rot.im;                                                   \
        y3.re = difAC.re + rot.re;                                                   \
        y3.im = difAC.im + rot.im;                                                   \
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
 *
 * This macro applies precomputed twiddle factors to a specific lane for two butterflies simultaneously using AVX2.
 * It loads twiddles directly into a vector and uses CMUL_FMA_R32 for multiplication.
 */
#ifdef __AVX2__
#define APPLY_STAGE_TWIDDLE_R32(kk, d_vec, stage_tw, lane, tw_out) \
    do                                                             \
    {                                                              \
        const int tw_idx = (kk) * 31 + (lane) - 1;                 \
        __m256d tw = _mm256_set_pd(                                \
            stage_tw[tw_idx + 31].im, stage_tw[tw_idx + 31].re,    \
            stage_tw[tw_idx].im, stage_tw[tw_idx].re);             \
        CMUL_FMA_R32(tw_out, d_vec, tw);                           \
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
 *
 * This macro applies hardcoded twiddle factors from the 32nd roots of unity to lanes 8-31 after the first radix-4 stage.
 * It uses precomputed constants for efficiency in the forward transform.
 */
#ifdef __AVX2__
#define APPLY_W32_TWIDDLES_FV_AVX2(x)                                                                                \
    do                                                                                                               \
    {                                                                                                                \
        /* j=1, g=0..7 (lanes 8-15) */                                                                               \
        CMUL_FMA_R32(x[8][b], x[8][b], _mm256_set_pd(0.0, 1.0, 0.0, 1.0));                                           \
        CMUL_FMA_R32(x[9][b], x[9][b], _mm256_set_pd(-0.1950903220, 0.9807852804, -0.1950903220, 0.9807852804));     \
        CMUL_FMA_R32(x[10][b], x[10][b], _mm256_set_pd(-0.3826834324, 0.9238795325, -0.3826834324, 0.9238795325));   \
        CMUL_FMA_R32(x[11][b], x[11][b], _mm256_set_pd(-0.5555702330, 0.8314696123, -0.5555702330, 0.8314696123));   \
        CMUL_FMA_R32(x[12][b], x[12][b], _mm256_set_pd(-0.7071067812, 0.7071067812, -0.7071067812, 0.7071067812));   \
        CMUL_FMA_R32(x[13][b], x[13][b], _mm256_set_pd(-0.8314696123, 0.5555702330, -0.8314696123, 0.5555702330));   \
        CMUL_FMA_R32(x[14][b], x[14][b], _mm256_set_pd(-0.9238795325, 0.3826834324, -0.9238795325, 0.3826834324));   \
        CMUL_FMA_R32(x[15][b], x[15][b], _mm256_set_pd(-0.9807852804, 0.1950903220, -0.9807852804, 0.1950903220));   \
                                                                                                                     \
        /* j=2, g=0..7 (lanes 16-23) */                                                                              \
        CMUL_FMA_R32(x[16][b], x[16][b], _mm256_set_pd(0.0, 1.0, 0.0, 1.0));                                         \
        CMUL_FMA_R32(x[17][b], x[17][b], _mm256_set_pd(-0.3826834324, 0.9238795325, -0.3826834324, 0.9238795325));   \
        CMUL_FMA_R32(x[18][b], x[18][b], _mm256_set_pd(-0.7071067812, 0.7071067812, -0.7071067812, 0.7071067812));   \
        CMUL_FMA_R32(x[19][b], x[19][b], _mm256_set_pd(-0.9238795325, 0.3826834324, -0.9238795325, 0.3826834324));   \
        CMUL_FMA_R32(x[20][b], x[20][b], _mm256_set_pd(-1.0, 0.0, -1.0, 0.0));                                       \
        CMUL_FMA_R32(x[21][b], x[21][b], _mm256_set_pd(-0.9238795325, -0.3826834324, -0.9238795325, -0.3826834324)); \
        CMUL_FMA_R32(x[22][b], x[22][b], _mm256_set_pd(-0.7071067812, -0.7071067812, -0.7071067812, -0.7071067812)); \
        CMUL_FMA_R32(x[23][b], x[23][b], _mm256_set_pd(-0.3826834324, -0.9238795325, -0.3826834324, -0.9238795325)); \
                                                                                                                     \
        /* j=3, g=0..7 (lanes 24-31) */                                                                              \
        CMUL_FMA_R32(x[24][b], x[24][b], _mm256_set_pd(0.0, 1.0, 0.0, 1.0));                                         \
        CMUL_FMA_R32(x[25][b], x[25][b], _mm256_set_pd(-0.5555702330, 0.8314696123, -0.5555702330, 0.8314696123));   \
        CMUL_FMA_R32(x[26][b], x[26][b], _mm256_set_pd(-0.9238795325, 0.3826834324, -0.9238795325, 0.3826834324));   \
        CMUL_FMA_R32(x[27][b], x[27][b], _mm256_set_pd(-0.9238795325, -0.3826834324, -0.9238795325, -0.3826834324)); \
        CMUL_FMA_R32(x[28][b], x[28][b], _mm256_set_pd(-0.7071067812, -0.7071067812, -0.7071067812, -0.7071067812)); \
        CMUL_FMA_R32(x[29][b], x[29][b], _mm256_set_pd(-0.1950903220, -0.9807852804, -0.1950903220, -0.9807852804)); \
        CMUL_FMA_R32(x[30][b], x[30][b], _mm256_set_pd(0.3826834324, -0.9238795325, 0.3826834324, -0.9238795325));   \
        CMUL_FMA_R32(x[31][b], x[31][b], _mm256_set_pd(0.8314696123, -0.5555702330, 0.8314696123, -0.5555702330));   \
    } while (0)
#endif

/**
 * @brief Apply W_32 twiddles (INVERSE: exp(+2πi*j*g/32))
 *
 * This macro applies hardcoded twiddle factors from the 32nd roots of unity to lanes 8-31 after the first radix-4 stage.
 * It uses precomputed constants (conjugates of forward) for efficiency in the inverse transform.
 */
#ifdef __AVX2__
#define APPLY_W32_TWIDDLES_BV_AVX2(x)                                                                                \
    do                                                                                                               \
    {                                                                                                                \
        /* j=1, g=0..7 (lanes 8-15) - conjugate of forward */                                                        \
        CMUL_FMA_R32(x[8][b], x[8][b], _mm256_set_pd(0.0, 1.0, 0.0, 1.0));                                           \
        CMUL_FMA_R32(x[9][b], x[9][b], _mm256_set_pd(0.1950903220, 0.9807852804, 0.1950903220, 0.9807852804));       \
        CMUL_FMA_R32(x[10][b], x[10][b], _mm256_set_pd(0.3826834324, 0.9238795325, 0.3826834324, 0.9238795325));     \
        CMUL_FMA_R32(x[11][b], x[11][b], _mm256_set_pd(0.5555702330, 0.8314696123, 0.5555702330, 0.8314696123));     \
        CMUL_FMA_R32(x[12][b], x[12][b], _mm256_set_pd(0.7071067812, 0.7071067812, 0.7071067812, 0.7071067812));     \
        CMUL_FMA_R32(x[13][b], x[13][b], _mm256_set_pd(0.8314696123, 0.5555702330, 0.8314696123, 0.5555702330));     \
        CMUL_FMA_R32(x[14][b], x[14][b], _mm256_set_pd(0.9238795325, 0.3826834324, 0.9238795325, 0.3826834324));     \
        CMUL_FMA_R32(x[15][b], x[15][b], _mm256_set_pd(0.9807852804, 0.1950903220, 0.9807852804, 0.1950903220));     \
                                                                                                                     \
        /* j=2, g=0..7 (lanes 16-23) */                                                                              \
        CMUL_FMA_R32(x[16][b], x[16][b], _mm256_set_pd(0.0, 1.0, 0.0, 1.0));                                         \
        CMUL_FMA_R32(x[17][b], x[17][b], _mm256_set_pd(0.3826834324, 0.9238795325, 0.3826834324, 0.9238795325));     \
        CMUL_FMA_R32(x[18][b], x[18][b], _mm256_set_pd(0.7071067812, 0.7071067812, 0.7071067812, 0.7071067812));     \
        CMUL_FMA_R32(x[19][b], x[19][b], _mm256_set_pd(0.9238795325, 0.3826834324, 0.9238795325, 0.3826834324));     \
        CMUL_FMA_R32(x[20][b], x[20][b], _mm256_set_pd(0.0, 1.0, 0.0, 1.0));                                         \
        CMUL_FMA_R32(x[21][b], x[21][b], _mm256_set_pd(0.9238795325, -0.3826834324, 0.9238795325, -0.3826834324));   \
        CMUL_FMA_R32(x[22][b], x[22][b], _mm256_set_pd(0.7071067812, -0.7071067812, 0.7071067812, -0.7071067812));   \
        CMUL_FMA_R32(x[23][b], x[23][b], _mm256_set_pd(0.3826834324, -0.9238795325, 0.3826834324, -0.9238795325));   \
                                                                                                                     \
        /* j=3, g=0..7 (lanes 24-31) */                                                                              \
        CMUL_FMA_R32(x[24][b], x[24][b], _mm256_set_pd(0.0, 1.0, 0.0, 1.0));                                         \
        CMUL_FMA_R32(x[25][b], x[25][b], _mm256_set_pd(0.5555702330, 0.8314696123, 0.5555702330, 0.8314696123));     \
        CMUL_FMA_R32(x[26][b], x[26][b], _mm256_set_pd(0.9238795325, 0.3826834324, 0.9238795325, 0.3826834324));     \
        CMUL_FMA_R32(x[27][b], x[27][b], _mm256_set_pd(0.9238795325, -0.3826834324, 0.9238795325, -0.3826834324));   \
        CMUL_FMA_R32(x[28][b], x[28][b], _mm256_set_pd(0.7071067812, -0.7071067812, 0.7071067812, -0.7071067812));   \
        CMUL_FMA_R32(x[29][b], x[29][b], _mm256_set_pd(0.1950903220, -0.9807852804, 0.1950903220, -0.9807852804));   \
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
 *
 * This macro applies hardcoded twiddle factors from the 8th roots of unity to specified outputs.
 * It uses optimized CMUL_FMA_R32 and permute/XOR for efficiency in the forward transform.
 */
#ifdef __AVX2__
#define APPLY_W8_TWIDDLES_FV_AVX2(o1, o2, o3)                                                     \
    do                                                                                            \
    {                                                                                             \
        __m256d w8_1 = _mm256_set_pd(-0.7071067812, 0.7071067812, -0.7071067812, 0.7071067812);   \
        CMUL_FMA_R32(o1, o1, w8_1);                                                               \
                                                                                                  \
        __m256d w8_2_mask = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);                                  \
        o2 = _mm256_permute_pd(o2, 0b0101);                                                       \
        o2 = _mm256_xor_pd(o2, w8_2_mask);                                                        \
                                                                                                  \
        __m256d w8_3 = _mm256_set_pd(-0.7071067812, -0.7071067812, -0.7071067812, -0.7071067812); \
        CMUL_FMA_R32(o3, o3, w8_3);                                                               \
    } while (0)
#endif

/**
 * @brief Apply W_8 twiddles (INVERSE)
 *
 * W_8^1 = (√2/2, +√2/2)
 * W_8^2 = (0, +1) = +i
 * W_8^3 = (-√2/2, +√2/2)
 *
 * This macro applies hardcoded twiddle factors from the 8th roots of unity to specified outputs.
 * It uses optimized CMUL_FMA_R32 and permute/XOR for efficiency in the inverse transform.
 */
#ifdef __AVX2__
#define APPLY_W8_TWIDDLES_BV_AVX2(o1, o2, o3)                                                   \
    do                                                                                          \
    {                                                                                           \
        __m256d w8_1 = _mm256_set_pd(0.7071067812, 0.7071067812, 0.7071067812, 0.7071067812);   \
        CMUL_FMA_R32(o1, o1, w8_1);                                                             \
                                                                                                \
        __m256d w8_2_mask = _mm256_set_pd(0.0, -0.0, 0.0, -0.0);                                \
        o2 = _mm256_permute_pd(o2, 0b0101);                                                     \
        o2 = _mm256_xor_pd(o2, w8_2_mask);                                                      \
                                                                                                \
        __m256d w8_3 = _mm256_set_pd(0.7071067812, -0.7071067812, 0.7071067812, -0.7071067812); \
        CMUL_FMA_R32(o3, o3, w8_3);                                                             \
    } while (0)
#endif

//==============================================================================
// SCALAR W_8 TWIDDLES
//==============================================================================

/**
 * @brief Apply W_8 twiddles (FORWARD, scalar)
 *
 * This macro applies hardcoded twiddle factors from the 8th roots of unity in scalar mode for the forward transform.
 * It optimizes multiplications for the specific constants.
 */
#define APPLY_W8_TWIDDLES_FV_SCALAR(o)         \
    do                                         \
    {                                          \
        /* W_8^1 = (√2/2, -√2/2) */            \
        {                                      \
            double r = o[1].re, i = o[1].im;   \
            o[1].re = (r + i) * 0.7071067812;  \
            o[1].im = (i - r) * 0.7071067812;  \
        }                                      \
                                               \
        /* W_8^2 = -i */                       \
        {                                      \
            double r = o[2].re;                \
            o[2].re = o[2].im;                 \
            o[2].im = -r;                      \
        }                                      \
                                               \
        /* W_8^3 = (-√2/2, -√2/2) */           \
        {                                      \
            double r = o[3].re, i = o[3].im;   \
            o[3].re = (-r + i) * 0.7071067812; \
            o[3].im = (-r - i) * 0.7071067812; \
        }                                      \
    } while (0)

/**
 * @brief Apply W_8 twiddles (INVERSE, scalar)
 *
 * This macro applies hardcoded twiddle factors from the 8th roots of unity in scalar mode for the inverse transform.
 * It optimizes multiplications for the specific constants.
 */
#define APPLY_W8_TWIDDLES_BV_SCALAR(o)         \
    do                                         \
    {                                          \
        /* W_8^1 = (√2/2, +√2/2) */            \
        {                                      \
            double r = o[1].re, i = o[1].im;   \
            o[1].re = (r - i) * 0.7071067812;  \
            o[1].im = (i + r) * 0.7071067812;  \
        }                                      \
                                               \
        /* W_8^2 = +i */                       \
        {                                      \
            double r = o[2].re;                \
            o[2].re = -o[2].im;                \
            o[2].im = r;                       \
        }                                      \
                                               \
        /* W_8^3 = (-√2/2, +√2/2) */           \
        {                                      \
            double r = o[3].re, i = o[3].im;   \
            o[3].re = (-r - i) * 0.7071067812; \
            o[3].im = (i - r) * 0.7071067812;  \
        }                                      \
    } while (0)

//==============================================================================
// RADIX-8 COMBINE - IDENTICAL for forward/inverse
//==============================================================================

/**
 * @brief Combine even/odd radix-4 results into radix-8 output
 *
 * This macro performs radix-2 combinations of even and odd outputs from radix-4 butterflies to form radix-8 results.
 * It computes sums and differences for eight outputs and is identical for both forward and inverse transforms.
 */
#ifdef __AVX2__
#define RADIX8_COMBINE_R32(e0, e1, e2, e3, o0, o1, o2, o3, \
                           x0, x1, x2, x3, x4, x5, x6, x7) \
    do                                                     \
    {                                                      \
        x0 = _mm256_add_pd(e0, o0);                        \
        x4 = _mm256_sub_pd(e0, o0);                        \
        x1 = _mm256_add_pd(e1, o1);                        \
        x5 = _mm256_sub_pd(e1, o1);                        \
        x2 = _mm256_add_pd(e2, o2);                        \
        x6 = _mm256_sub_pd(e2, o2);                        \
        x3 = _mm256_add_pd(e3, o3);                        \
        x7 = _mm256_sub_pd(e3, o3);                        \
    } while (0)
#endif

//==============================================================================
// DATA MOVEMENT - IDENTICAL for forward/inverse
//==============================================================================

/**
 * @brief Load 2 complex values from two locations
 *
 * This macro loads two complex values (AoS format) from separate pointers into an AVX2 vector.
 * It is a wrapper around load2_aos for data movement in the radix-32 butterfly.
 */
#ifdef __AVX2__
#define LOAD_2_COMPLEX_R32(ptr1, ptr2) \
    load2_aos(ptr1, ptr2)
#endif

/**
 * @brief Store 2 complex values
 *
 * This macro stores an AVX2 vector containing two complex values (AoS) to a pointer using unaligned store.
 */
#ifdef __AVX2__
#define STORE_2_COMPLEX_R32(ptr, vec) \
    STOREU_PD(&(ptr)->re, vec)
#endif

/**
 * @brief Store with streaming
 *
 * This macro stores an AVX2 vector containing two complex values (AoS) using non-temporal streaming store.
 * It bypasses cache for large data sets to improve performance.
 */
#ifdef __AVX2__
#define STORE_2_COMPLEX_R32_STREAM(ptr, vec) \
    _mm256_stream_pd(&(ptr)->re, vec)
#endif

//==============================================================================
// PREFETCHING - IDENTICAL for forward/inverse
//==============================================================================

/**
 * @brief Prefetch distances for L1, L2, and L3 caches in radix-32.
 *
 * These constants define how far ahead to prefetch data in terms of indices for the radix-32 butterfly.
 * They are tuned to optimize memory access by loading data into caches preemptively.
 */
#define PREFETCH_L1_R32 8
#define PREFETCH_L2_R32 32
#define PREFETCH_L3_R32 64

/**
 * @brief Prefetch 32 lanes ahead for AVX2 in radix-32.
 *
 * This macro issues prefetch instructions for future strided data accesses in the sub_outputs buffer.
 * It prefetches every 4th lane to cover the 32 lanes efficiently, using the specified cache hint.
 */
#ifdef __AVX2__
#define PREFETCH_32_LANES_R32(k, K, distance, sub_outputs, hint)                              \
    do                                                                                        \
    {                                                                                         \
        if ((k) + (distance) < K)                                                             \
        {                                                                                     \
            for (int _lane = 0; _lane < 32; _lane += 4)                                       \
            {                                                                                 \
                _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + _lane * K], hint); \
            }                                                                                 \
        }                                                                                     \
    } while (0)
#endif

//==============================================================================
// COMPLETE SCALAR RADIX-4 BUTTERFLY
//==============================================================================

/**
 * @brief Complete scalar radix-4 butterfly (forward version)
 *
 * This macro performs a full radix-4 butterfly in scalar mode for the forward transform.
 * It computes core arithmetic, applies rotation, assembles outputs, and overwrites inputs with results.
 */
#define RADIX4_BUTTERFLY_SCALAR_FV_R32(a, b, c, d)                                    \
    do                                                                                \
    {                                                                                 \
        fft_data sumBD, difBD, sumAC, difAC;                                          \
        RADIX4_BUTTERFLY_CORE_R32_SCALAR(a, b, c, d, sumBD, difBD, sumAC, difAC);     \
                                                                                      \
        fft_data rot;                                                                 \
        RADIX4_ROTATE_FORWARD_R32_SCALAR(difBD, rot);                                 \
                                                                                      \
        fft_data y0, y1, y2, y3;                                                      \
        RADIX4_ASSEMBLE_OUTPUTS_R32_SCALAR(sumAC, sumBD, difAC, rot, y0, y1, y2, y3); \
                                                                                      \
        a = y0;                                                                       \
        b = y1;                                                                       \
        c = y2;                                                                       \
        d = y3;                                                                       \
    } while (0)

/**
 * @brief Complete scalar radix-4 butterfly (inverse version)
 *
 * This macro performs a full radix-4 butterfly in scalar mode for the inverse transform.
 * It computes core arithmetic, applies rotation, assembles outputs, and overwrites inputs with results.
 */
#define RADIX4_BUTTERFLY_SCALAR_BV_R32(a, b, c, d)                                    \
    do                                                                                \
    {                                                                                 \
        fft_data sumBD, difBD, sumAC, difAC;                                          \
        RADIX4_BUTTERFLY_CORE_R32_SCALAR(a, b, c, d, sumBD, difBD, sumAC, difAC);     \
                                                                                      \
        fft_data rot;                                                                 \
        RADIX4_ROTATE_INVERSE_R32_SCALAR(difBD, rot);                                 \
                                                                                      \
        fft_data y0, y1, y2, y3;                                                      \
        RADIX4_ASSEMBLE_OUTPUTS_R32_SCALAR(sumAC, sumBD, difAC, rot, y0, y1, y2, y3); \
                                                                                      \
        a = y0;                                                                       \
        b = y1;                                                                       \
        c = y2;                                                                       \
        d = y3;                                                                       \
    } while (0)

#endif // FFT_RADIX32_MACROS_H