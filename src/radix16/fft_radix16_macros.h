//==============================================================================
// fft_radix16_macros.h - Shared Macros for Radix-16 Butterflies
//==============================================================================
//
// ALGORITHM: 2-stage radix-4 decomposition
//   1. Apply input twiddles W_N^(j*k) for j=1..15
//   2. First radix-4 stage (4 groups of 4)
//   3. Apply W_4 intermediate twiddles
//   4. Second radix-4 stage (final)
//
// KEY INSIGHT: Reuse radix-4 macros for both stages!
//

#ifndef FFT_RADIX16_MACROS_H
#define FFT_RADIX16_MACROS_H

#include "simd_math.h"

//==============================================================================
// REUSE RADIX-4 INFRASTRUCTURE
//==============================================================================

// We'll manually include the key radix-4 macros we need
// (In real code, could #include "fft_radix4_macros.h")

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
// W_4 INTERMEDIATE TWIDDLES - Direction-dependent
//==============================================================================

/**
 * @brief W_4 twiddle constants for FORWARD FFT
 *
 * W_4 = exp(-2πi/4) = exp(-πi/2) = -i
 * W_4^0 = 1
 * W_4^1 = -i
 * W_4^2 = -1
 * W_4^3 = +i
 */
#define W4_FV_0_RE 1.0
#define W4_FV_0_IM 0.0
#define W4_FV_1_RE 0.0
#define W4_FV_1_IM (-1.0)
#define W4_FV_2_RE (-1.0)
#define W4_FV_2_IM 0.0
#define W4_FV_3_RE 0.0
#define W4_FV_3_IM 1.0

/**
 * @brief W_4 twiddle constants for INVERSE FFT
 *
 * W_4 = exp(+2πi/4) = exp(+πi/2) = +i
 * W_4^0 = 1
 * W_4^1 = +i
 * W_4^2 = -1
 * W_4^3 = -i
 */
#define W4_BV_0_RE 1.0
#define W4_BV_0_IM 0.0
#define W4_BV_1_RE 0.0
#define W4_BV_1_IM 1.0
#define W4_BV_2_RE (-1.0)
#define W4_BV_2_IM 0.0
#define W4_BV_3_RE 0.0
#define W4_BV_3_IM (-1.0)

//==============================================================================
// RADIX-4 BUTTERFLY - Reused from radix-4
//==============================================================================

#ifdef __AVX2__
#define RADIX4_BUTTERFLY_AVX2(a, b, c, d, y0, y1, y2, y3, rot_mask) \
    do                                                              \
    {                                                               \
        __m256d sumBD = _mm256_add_pd(b, d);                        \
        __m256d difBD = _mm256_sub_pd(b, d);                        \
        __m256d sumAC = _mm256_add_pd(a, c);                        \
        __m256d difAC = _mm256_sub_pd(a, c);                        \
                                                                    \
        y0 = _mm256_add_pd(sumAC, sumBD);                           \
        y2 = _mm256_sub_pd(sumAC, sumBD);                           \
                                                                    \
        __m256d dif_bd_swp = _mm256_permute_pd(difBD, 0b0101);      \
        __m256d dif_bd_rot = _mm256_xor_pd(dif_bd_swp, rot_mask);   \
                                                                    \
        y1 = _mm256_sub_pd(difAC, dif_bd_rot);                      \
        y3 = _mm256_add_pd(difAC, dif_bd_rot);                      \
    } while (0)
#endif

//==============================================================================
// APPLY W_4 INTERMEDIATE TWIDDLES - ONLY DIFFERENCE
//==============================================================================

/**
 * @brief Apply W_4 intermediate twiddles for FORWARD FFT
 *
 * Pattern for y[4*m + j] where m=0..3, j=0..3:
 * - y[0..3]:   no twiddle (W_4^0 = 1)
 * - y[4..7]:   W_4^{0*j, 1*j, 2*j, 3*j} = {1, -i, -1, +i}
 * - y[8..11]:  W_4^{0*j, 2*j, 0*j, 2*j} = {1, -1, 1, -1}
 * - y[12..15]: W_4^{0*j, 3*j, 2*j, 1*j} = {1, +i, -1, -i}
 */
#ifdef __AVX2__
#define APPLY_W4_INTERMEDIATE_FV_AVX2(y)                                                \
    do                                                                                  \
    {                                                                                   \
        /* m=1: W_4^j for j=1,2,3 */                                                    \
        {                                                                               \
            __m256d w1 = _mm256_set_pd(W4_FV_1_IM, W4_FV_1_RE, W4_FV_1_IM, W4_FV_1_RE); \
            __m256d w2 = _mm256_set_pd(W4_FV_2_IM, W4_FV_2_RE, W4_FV_2_IM, W4_FV_2_RE); \
            __m256d w3 = _mm256_set_pd(W4_FV_3_IM, W4_FV_3_RE, W4_FV_3_IM, W4_FV_3_RE); \
            CMUL_FMA_AOS(y[5], y[5], w1);                                               \
            CMUL_FMA_AOS(y[6], y[6], w2);                                               \
            CMUL_FMA_AOS(y[7], y[7], w3);                                               \
        }                                                                               \
        /* m=2: W_4^{2j} for j=1,2,3 → {-1, 1, -1} */                                   \
        {                                                                               \
            __m256d w2 = _mm256_set_pd(W4_FV_2_IM, W4_FV_2_RE, W4_FV_2_IM, W4_FV_2_RE); \
            CMUL_FMA_AOS(y[9], y[9], w2);                                               \
            /* y[10] *= W_4^0 = 1 (skip) */                                             \
            CMUL_FMA_AOS(y[11], y[11], w2);                                             \
        }                                                                               \
        /* m=3: W_4^{3j} for j=1,2,3 → {+i, -1, -i} */                                  \
        {                                                                               \
            __m256d w3 = _mm256_set_pd(W4_FV_3_IM, W4_FV_3_RE, W4_FV_3_IM, W4_FV_3_RE); \
            __m256d w2 = _mm256_set_pd(W4_FV_2_IM, W4_FV_2_RE, W4_FV_2_IM, W4_FV_2_RE); \
            __m256d w1 = _mm256_set_pd(W4_FV_1_IM, W4_FV_1_RE, W4_FV_1_IM, W4_FV_1_RE); \
            CMUL_FMA_AOS(y[13], y[13], w3);                                             \
            CMUL_FMA_AOS(y[14], y[14], w2);                                             \
            CMUL_FMA_AOS(y[15], y[15], w1);                                             \
        }                                                                               \
    } while (0)
#endif

/**
 * @brief Apply W_4 intermediate twiddles for INVERSE FFT
 *
 * Same pattern but with conjugated W_4 twiddles
 */
#ifdef __AVX2__
#define APPLY_W4_INTERMEDIATE_BV_AVX2(y)                                                \
    do                                                                                  \
    {                                                                                   \
        /* m=1: W_4^j for j=1,2,3 */                                                    \
        {                                                                               \
            __m256d w1 = _mm256_set_pd(W4_BV_1_IM, W4_BV_1_RE, W4_BV_1_IM, W4_BV_1_RE); \
            __m256d w2 = _mm256_set_pd(W4_BV_2_IM, W4_BV_2_RE, W4_BV_2_IM, W4_BV_2_RE); \
            __m256d w3 = _mm256_set_pd(W4_BV_3_IM, W4_BV_3_RE, W4_BV_3_IM, W4_BV_3_RE); \
            CMUL_FMA_AOS(y[5], y[5], w1);                                               \
            CMUL_FMA_AOS(y[6], y[6], w2);                                               \
            CMUL_FMA_AOS(y[7], y[7], w3);                                               \
        }                                                                               \
        /* m=2: W_4^{2j} for j=1,2,3 → {-1, 1, -1} */                                   \
        {                                                                               \
            __m256d w2 = _mm256_set_pd(W4_BV_2_IM, W4_BV_2_RE, W4_BV_2_IM, W4_BV_2_RE); \
            CMUL_FMA_AOS(y[9], y[9], w2);                                               \
            /* y[10] *= W_4^0 = 1 (skip) */                                             \
            CMUL_FMA_AOS(y[11], y[11], w2);                                             \
        }                                                                               \
        /* m=3: W_4^{3j} for j=1,2,3 → {-i, -1, +i} */                                  \
        {                                                                               \
            __m256d w3 = _mm256_set_pd(W4_BV_3_IM, W4_BV_3_RE, W4_BV_3_IM, W4_BV_3_RE); \
            __m256d w2 = _mm256_set_pd(W4_BV_2_IM, W4_BV_2_RE, W4_BV_2_IM, W4_BV_2_RE); \
            __m256d w1 = _mm256_set_pd(W4_BV_1_IM, W4_BV_1_RE, W4_BV_1_IM, W4_BV_1_RE); \
            CMUL_FMA_AOS(y[13], y[13], w3);                                             \
            CMUL_FMA_AOS(y[14], y[14], w2);                                             \
            CMUL_FMA_AOS(y[15], y[15], w1);                                             \
        }                                                                               \
    } while (0)
#endif

// Scalar versions
#define APPLY_W4_INTERMEDIATE_FV_SCALAR(y)              \
    do                                                  \
    {                                                   \
        /* m=1: j=1,2,3 */                              \
        {                                               \
            double r, i;                                \
            r = y[5].re;                                \
            i = y[5].im;                                \
            y[5].re = r * W4_FV_1_RE - i * W4_FV_1_IM;  \
            y[5].im = r * W4_FV_1_IM + i * W4_FV_1_RE;  \
            r = y[6].re;                                \
            i = y[6].im;                                \
            y[6].re = r * W4_FV_2_RE - i * W4_FV_2_IM;  \
            y[6].im = r * W4_FV_2_IM + i * W4_FV_2_RE;  \
            r = y[7].re;                                \
            i = y[7].im;                                \
            y[7].re = r * W4_FV_3_RE - i * W4_FV_3_IM;  \
            y[7].im = r * W4_FV_3_IM + i * W4_FV_3_RE;  \
        }                                               \
        /* m=2: j=1,3 */                                \
        {                                               \
            y[9].re = -y[9].re;                         \
            y[9].im = -y[9].im;                         \
            y[11].re = -y[11].re;                       \
            y[11].im = -y[11].im;                       \
        }                                               \
        /* m=3: j=1,2,3 */                              \
        {                                               \
            double r, i;                                \
            r = y[13].re;                               \
            i = y[13].im;                               \
            y[13].re = r * W4_FV_3_RE - i * W4_FV_3_IM; \
            y[13].im = r * W4_FV_3_IM + i * W4_FV_3_RE; \
            y[14].re = -y[14].re;                       \
            y[14].im = -y[14].im;                       \
            r = y[15].re;                               \
            i = y[15].im;                               \
            y[15].re = r * W4_FV_1_RE - i * W4_FV_1_IM; \
            y[15].im = r * W4_FV_1_IM + i * W4_FV_1_RE; \
        }                                               \
    } while (0)

#define APPLY_W4_INTERMEDIATE_BV_SCALAR(y)              \
    do                                                  \
    {                                                   \
        /* m=1: j=1,2,3 */                              \
        {                                               \
            double r, i;                                \
            r = y[5].re;                                \
            i = y[5].im;                                \
            y[5].re = r * W4_BV_1_RE - i * W4_BV_1_IM;  \
            y[5].im = r * W4_BV_1_IM + i * W4_BV_1_RE;  \
            r = y[6].re;                                \
            i = y[6].im;                                \
            y[6].re = r * W4_BV_2_RE - i * W4_BV_2_IM;  \
            y[6].im = r * W4_BV_2_IM + i * W4_BV_2_RE;  \
            r = y[7].re;                                \
            i = y[7].im;                                \
            y[7].re = r * W4_BV_3_RE - i * W4_BV_3_IM;  \
            y[7].im = r * W4_BV_3_IM + i * W4_BV_3_RE;  \
        }                                               \
        /* m=2: j=1,3 */                                \
        {                                               \
            y[9].re = -y[9].re;                         \
            y[9].im = -y[9].im;                         \
            y[11].re = -y[11].re;                       \
            y[11].im = -y[11].im;                       \
        }                                               \
        /* m=3: j=1,2,3 */                              \
        {                                               \
            double r, i;                                \
            r = y[13].re;                               \
            i = y[13].im;                               \
            y[13].re = r * W4_BV_3_RE - i * W4_BV_3_IM; \
            y[13].im = r * W4_BV_3_IM + i * W4_BV_3_RE; \
            y[14].re = -y[14].re;                       \
            y[14].im = -y[14].im;                       \
            r = y[15].re;                               \
            i = y[15].im;                               \
            y[15].re = r * W4_BV_1_RE - i * W4_BV_1_IM; \
            y[15].im = r * W4_BV_1_IM + i * W4_BV_1_RE; \
        }                                               \
    } while (0)

//==============================================================================
// APPLY STAGE TWIDDLES - IDENTICAL for forward/inverse
//==============================================================================

/**
 * @brief Scalar: Apply stage twiddles to lanes 1-15
 *
 * stage_tw layout: [W^(1*k), W^(2*k), ..., W^(15*k)] for each k
 */
#define APPLY_STAGE_TWIDDLES_R16_SCALAR(k, x, stage_tw)                \
    do                                                                 \
    {                                                                  \
        const fft_data *w_ptr = &stage_tw[(k) * 15];                   \
        for (int j = 1; j <= 15; j++)                                  \
        {                                                              \
            fft_data a = x[j];                                         \
            x[j].re = a.re * w_ptr[j - 1].re - a.im * w_ptr[j - 1].im; \
            x[j].im = a.re * w_ptr[j - 1].im + a.im * w_ptr[j - 1].re; \
        }                                                              \
    } while (0)

/**
 * @brief AVX2: Apply stage twiddles for 2 butterflies (kk and kk+1)
 */
#ifdef __AVX2__
#define APPLY_STAGE_TWIDDLES_R16_AVX2(kk, x, stage_tw)                                                 \
    do                                                                                                 \
    {                                                                                                  \
        for (int j = 1; j <= 15; j++)                                                                  \
        {                                                                                              \
            __m256d w = load2_aos(&stage_tw[(kk) * 15 + (j - 1)], &stage_tw[(kk + 1) * 15 + (j - 1)]); \
            CMUL_FMA_AOS(x[j], x[j], w);                                                               \
        }                                                                                              \
    } while (0)
#endif

//==============================================================================
// DATA MOVEMENT - IDENTICAL for forward/inverse
//==============================================================================

#ifdef __AVX2__
#define LOAD_16_LANES_AVX2(kk, K, sub_outputs, x)                                                  \
    do                                                                                             \
    {                                                                                              \
        for (int lane = 0; lane < 16; lane++)                                                      \
        {                                                                                          \
            x[lane] = load2_aos(&sub_outputs[(kk) + lane * K], &sub_outputs[(kk) + 1 + lane * K]); \
        }                                                                                          \
    } while (0)

#define STORE_16_LANES_AVX2(kk, K, output_buffer, z, m_offset)            \
    do                                                                    \
    {                                                                     \
        for (int m = 0; m < 4; m++)                                       \
        {                                                                 \
            STOREU_PD(&output_buffer[(kk) + m * K].re, z[m]);             \
            STOREU_PD(&output_buffer[(kk) + (m + 4) * K].re, z[m + 4]);   \
            STOREU_PD(&output_buffer[(kk) + (m + 8) * K].re, z[m + 8]);   \
            STOREU_PD(&output_buffer[(kk) + (m + 12) * K].re, z[m + 12]); \
        }                                                                 \
    } while (0)

#define STORE_16_LANES_AVX2_STREAM(kk, K, output_buffer, z, m_offset)            \
    do                                                                           \
    {                                                                            \
        for (int m = 0; m < 4; m++)                                              \
        {                                                                        \
            _mm256_stream_pd(&output_buffer[(kk) + m * K].re, z[m]);             \
            _mm256_stream_pd(&output_buffer[(kk) + (m + 4) * K].re, z[m + 4]);   \
            _mm256_stream_pd(&output_buffer[(kk) + (m + 8) * K].re, z[m + 8]);   \
            _mm256_stream_pd(&output_buffer[(kk) + (m + 12) * K].re, z[m + 12]); \
        }                                                                        \
    } while (0)
#endif

//==============================================================================
// PREFETCHING
//==============================================================================

#define PREFETCH_L1 8
#define PREFETCH_L2 32
#define PREFETCH_L3 64

#ifdef __AVX2__
#define PREFETCH_16_LANES(k, K, distance, sub_outputs, hint)                                 \
    do                                                                                       \
    {                                                                                        \
        if ((k) + (distance) < K)                                                            \
        {                                                                                    \
            for (int lane = 0; lane < 16; lane++)                                            \
            {                                                                                \
                _mm_prefetch((const char *)&sub_outputs[(k) + (distance) + lane * K], hint); \
            }                                                                                \
        }                                                                                    \
    } while (0)
#endif

#endif // FFT_RADIX16_MACROS_H