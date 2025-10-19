//==============================================================================
// fft_radix13_macros.h - Shared Macros for Radix-13 Butterflies (Rader's Algorithm)
//==============================================================================
//
// ALGORITHM: Radix-13 DFT using Rader's Algorithm
//   Prime 13 → 12-point cyclic convolution
//   - Generator g=2, inverse g^-1=7
//   - Input permutation:  [1,2,4,8,3,6,12,11,9,5,10,7]
//   - Output permutation: [1,12,10,3,9,4,7,5,2,6,11,8]
//
// USAGE:
//   #include "fft_radix13_macros.h" in both fft_radix13_fv.c and fft_radix13_bv.c
//

#ifndef FFT_RADIX13_MACROS_H
#define FFT_RADIX13_MACROS_H

#include "simd_math.h"
#include "math.h"

//==============================================================================
// RADER PERMUTATIONS - IDENTICAL for forward/inverse
//==============================================================================

// Input permutation indices (generator powers mod 13)
#define RADER13_INPUT_PERM {1, 2, 4, 8, 3, 6, 12, 11, 9, 5, 10, 7}

// Output permutation indices (where convolution results map)
#define RADER13_OUTPUT_PERM {1, 12, 10, 3, 9, 4, 7, 5, 2, 6, 11, 8}

//==============================================================================
// PRECOMPUTE CONVOLUTION TWIDDLES - DIRECTION-SPECIFIC
//==============================================================================

#ifdef __AVX2__
#define PRECOMPUTE_RADER13_TWIDDLES_FV(tw_brd)         \
    do                                                 \
    {                                                  \
        const int op[12] = RADER13_OUTPUT_PERM;        \
        const double base_angle = -2.0 * M_PI / 13.0;  \
        for (int q = 0; q < 12; ++q)                   \
        {                                              \
            double a = op[q] * base_angle;             \
            double wr, wi;                             \
            sincos(a, &wi, &wr);                       \
            tw_brd[q] = _mm256_set_pd(wi, wr, wi, wr); \
        }                                              \
    } while (0)

#define PRECOMPUTE_RADER13_TWIDDLES_BV(tw_brd)         \
    do                                                 \
    {                                                  \
        const int op[12] = RADER13_OUTPUT_PERM;        \
        const double base_angle = +2.0 * M_PI / 13.0;  \
        for (int q = 0; q < 12; ++q)                   \
        {                                              \
            double a = op[q] * base_angle;             \
            double wr, wi;                             \
            sincos(a, &wi, &wr);                       \
            tw_brd[q] = _mm256_set_pd(wi, wr, wi, wr); \
        }                                              \
    } while (0)
#endif

//==============================================================================
// OPTIMIZED COMPLEX MULTIPLICATION - Using fmaddsub (Point 2)
//==============================================================================

#ifdef __AVX2__
/**
 * @brief FMA-optimized complex multiply: out = a * w
 * Uses fmaddsub for maximum performance
 */
#define CMUL_FMA_R13(out, a, w)                          \
    do {                                                 \
        __m256d wr = _mm256_unpacklo_pd(w, w);           \
        __m256d wi = _mm256_unpackhi_pd(w, w);           \
        __m256d as = _mm256_permute_pd(a, 0x5);          \
        __m256d t  = _mm256_mul_pd(a, wr);               \
        out = _mm256_fmaddsub_pd(as, wi, t);             \
    } while (0)
#endif

//==============================================================================
// FULL AVX2 12-POINT CYCLIC CONVOLUTION (Point 1 & 4)
//==============================================================================

#ifdef __AVX2__
/**
 * @brief Fully unrolled 12-point cyclic convolution for 2 butterflies
 *
 * Computes all v0..v11 with 12 accumulators, no % arithmetic
 * Uses precomputed index table for direct addressing
 * 144 complex multiplies fully unrolled with FMA
 */
#define CONV12_FULL_AVX2(tx0, tx1, tx2, tx3, tx4, tx5, tx6, tx7, tx8, tx9, tx10, tx11, tw_brd, \
                         v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11)                     \
    do                                                                                         \
    {                                                                                          \
        __m256d tmp;                                                                           \
        /* Initialize accumulators with l=0 term */                                            \
        CMUL_FMA_R13(v0, tx0, tw_brd[0]);                                                      \
        CMUL_FMA_R13(v1, tx0, tw_brd[1]);                                                      \
        CMUL_FMA_R13(v2, tx0, tw_brd[2]);                                                      \
        CMUL_FMA_R13(v3, tx0, tw_brd[3]);                                                      \
        CMUL_FMA_R13(v4, tx0, tw_brd[4]);                                                      \
        CMUL_FMA_R13(v5, tx0, tw_brd[5]);                                                      \
        CMUL_FMA_R13(v6, tx0, tw_brd[6]);                                                      \
        CMUL_FMA_R13(v7, tx0, tw_brd[7]);                                                      \
        CMUL_FMA_R13(v8, tx0, tw_brd[8]);                                                      \
        CMUL_FMA_R13(v9, tx0, tw_brd[9]);                                                      \
        CMUL_FMA_R13(v10, tx0, tw_brd[10]);                                                    \
        CMUL_FMA_R13(v11, tx0, tw_brd[11]);                                                    \
                                                                                               \
        /* Accumulate l=1..11 using precomputed indices */                                     \
        CMUL_FMA_R13(tmp, tx1, tw_brd[11]);                                                    \
        v0 = _mm256_add_pd(v0, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx1, tw_brd[0]);                                                     \
        v1 = _mm256_add_pd(v1, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx1, tw_brd[1]);                                                     \
        v2 = _mm256_add_pd(v2, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx1, tw_brd[2]);                                                     \
        v3 = _mm256_add_pd(v3, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx1, tw_brd[3]);                                                     \
        v4 = _mm256_add_pd(v4, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx1, tw_brd[4]);                                                     \
        v5 = _mm256_add_pd(v5, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx1, tw_brd[5]);                                                     \
        v6 = _mm256_add_pd(v6, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx1, tw_brd[6]);                                                     \
        v7 = _mm256_add_pd(v7, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx1, tw_brd[7]);                                                     \
        v8 = _mm256_add_pd(v8, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx1, tw_brd[8]);                                                     \
        v9 = _mm256_add_pd(v9, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx1, tw_brd[9]);                                                     \
        v10 = _mm256_add_pd(v10, tmp);                                                         \
        CMUL_FMA_R13(tmp, tx1, tw_brd[10]);                                                    \
        v11 = _mm256_add_pd(v11, tmp);                                                         \
                                                                                               \
        CMUL_FMA_R13(tmp, tx2, tw_brd[10]);                                                    \
        v0 = _mm256_add_pd(v0, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx2, tw_brd[11]);                                                    \
        v1 = _mm256_add_pd(v1, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx2, tw_brd[0]);                                                     \
        v2 = _mm256_add_pd(v2, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx2, tw_brd[1]);                                                     \
        v3 = _mm256_add_pd(v3, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx2, tw_brd[2]);                                                     \
        v4 = _mm256_add_pd(v4, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx2, tw_brd[3]);                                                     \
        v5 = _mm256_add_pd(v5, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx2, tw_brd[4]);                                                     \
        v6 = _mm256_add_pd(v6, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx2, tw_brd[5]);                                                     \
        v7 = _mm256_add_pd(v7, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx2, tw_brd[6]);                                                     \
        v8 = _mm256_add_pd(v8, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx2, tw_brd[7]);                                                     \
        v9 = _mm256_add_pd(v9, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx2, tw_brd[8]);                                                     \
        v10 = _mm256_add_pd(v10, tmp);                                                         \
        CMUL_FMA_R13(tmp, tx2, tw_brd[9]);                                                     \
        v11 = _mm256_add_pd(v11, tmp);                                                         \
                                                                                               \
        CMUL_FMA_R13(tmp, tx3, tw_brd[9]);                                                     \
        v0 = _mm256_add_pd(v0, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx3, tw_brd[10]);                                                    \
        v1 = _mm256_add_pd(v1, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx3, tw_brd[11]);                                                    \
        v2 = _mm256_add_pd(v2, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx3, tw_brd[0]);                                                     \
        v3 = _mm256_add_pd(v3, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx3, tw_brd[1]);                                                     \
        v4 = _mm256_add_pd(v4, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx3, tw_brd[2]);                                                     \
        v5 = _mm256_add_pd(v5, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx3, tw_brd[3]);                                                     \
        v6 = _mm256_add_pd(v6, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx3, tw_brd[4]);                                                     \
        v7 = _mm256_add_pd(v7, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx3, tw_brd[5]);                                                     \
        v8 = _mm256_add_pd(v8, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx3, tw_brd[6]);                                                     \
        v9 = _mm256_add_pd(v9, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx3, tw_brd[7]);                                                     \
        v10 = _mm256_add_pd(v10, tmp);                                                         \
        CMUL_FMA_R13(tmp, tx3, tw_brd[8]);                                                     \
        v11 = _mm256_add_pd(v11, tmp);                                                         \
                                                                                               \
        CMUL_FMA_R13(tmp, tx4, tw_brd[8]);                                                     \
        v0 = _mm256_add_pd(v0, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx4, tw_brd[9]);                                                     \
        v1 = _mm256_add_pd(v1, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx4, tw_brd[10]);                                                    \
        v2 = _mm256_add_pd(v2, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx4, tw_brd[11]);                                                    \
        v3 = _mm256_add_pd(v3, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx4, tw_brd[0]);                                                     \
        v4 = _mm256_add_pd(v4, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx4, tw_brd[1]);                                                     \
        v5 = _mm256_add_pd(v5, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx4, tw_brd[2]);                                                     \
        v6 = _mm256_add_pd(v6, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx4, tw_brd[3]);                                                     \
        v7 = _mm256_add_pd(v7, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx4, tw_brd[4]);                                                     \
        v8 = _mm256_add_pd(v8, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx4, tw_brd[5]);                                                     \
        v9 = _mm256_add_pd(v9, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx4, tw_brd[6]);                                                     \
        v10 = _mm256_add_pd(v10, tmp);                                                         \
        CMUL_FMA_R13(tmp, tx4, tw_brd[7]);                                                     \
        v11 = _mm256_add_pd(v11, tmp);                                                         \
                                                                                               \
        CMUL_FMA_R13(tmp, tx5, tw_brd[7]);                                                     \
        v0 = _mm256_add_pd(v0, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx5, tw_brd[8]);                                                     \
        v1 = _mm256_add_pd(v1, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx5, tw_brd[9]);                                                     \
        v2 = _mm256_add_pd(v2, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx5, tw_brd[10]);                                                    \
        v3 = _mm256_add_pd(v3, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx5, tw_brd[11]);                                                    \
        v4 = _mm256_add_pd(v4, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx5, tw_brd[0]);                                                     \
        v5 = _mm256_add_pd(v5, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx5, tw_brd[1]);                                                     \
        v6 = _mm256_add_pd(v6, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx5, tw_brd[2]);                                                     \
        v7 = _mm256_add_pd(v7, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx5, tw_brd[3]);                                                     \
        v8 = _mm256_add_pd(v8, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx5, tw_brd[4]);                                                     \
        v9 = _mm256_add_pd(v9, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx5, tw_brd[5]);                                                     \
        v10 = _mm256_add_pd(v10, tmp);                                                         \
        CMUL_FMA_R13(tmp, tx5, tw_brd[6]);                                                     \
        v11 = _mm256_add_pd(v11, tmp);                                                         \
                                                                                               \
        CMUL_FMA_R13(tmp, tx6, tw_brd[6]);                                                     \
        v0 = _mm256_add_pd(v0, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx6, tw_brd[7]);                                                     \
        v1 = _mm256_add_pd(v1, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx6, tw_brd[8]);                                                     \
        v2 = _mm256_add_pd(v2, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx6, tw_brd[9]);                                                     \
        v3 = _mm256_add_pd(v3, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx6, tw_brd[10]);                                                    \
        v4 = _mm256_add_pd(v4, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx6, tw_brd[11]);                                                    \
        v5 = _mm256_add_pd(v5, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx6, tw_brd[0]);                                                     \
        v6 = _mm256_add_pd(v6, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx6, tw_brd[1]);                                                     \
        v7 = _mm256_add_pd(v7, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx6, tw_brd[2]);                                                     \
        v8 = _mm256_add_pd(v8, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx6, tw_brd[3]);                                                     \
        v9 = _mm256_add_pd(v9, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx6, tw_brd[4]);                                                     \
        v10 = _mm256_add_pd(v10, tmp);                                                         \
        CMUL_FMA_R13(tmp, tx6, tw_brd[5]);                                                     \
        v11 = _mm256_add_pd(v11, tmp);                                                         \
                                                                                               \
        CMUL_FMA_R13(tmp, tx7, tw_brd[5]);                                                     \
        v0 = _mm256_add_pd(v0, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx7, tw_brd[6]);                                                     \
        v1 = _mm256_add_pd(v1, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx7, tw_brd[7]);                                                     \
        v2 = _mm256_add_pd(v2, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx7, tw_brd[8]);                                                     \
        v3 = _mm256_add_pd(v3, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx7, tw_brd[9]);                                                     \
        v4 = _mm256_add_pd(v4, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx7, tw_brd[10]);                                                    \
        v5 = _mm256_add_pd(v5, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx7, tw_brd[11]);                                                    \
        v6 = _mm256_add_pd(v6, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx7, tw_brd[0]);                                                     \
        v7 = _mm256_add_pd(v7, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx7, tw_brd[1]);                                                     \
        v8 = _mm256_add_pd(v8, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx7, tw_brd[2]);                                                     \
        v9 = _mm256_add_pd(v9, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx7, tw_brd[3]);                                                     \
        v10 = _mm256_add_pd(v10, tmp);                                                         \
        CMUL_FMA_R13(tmp, tx7, tw_brd[4]);                                                     \
        v11 = _mm256_add_pd(v11, tmp);                                                         \
                                                                                               \
        CMUL_FMA_R13(tmp, tx8, tw_brd[4]);                                                     \
        v0 = _mm256_add_pd(v0, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx8, tw_brd[5]);                                                     \
        v1 = _mm256_add_pd(v1, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx8, tw_brd[6]);                                                     \
        v2 = _mm256_add_pd(v2, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx8, tw_brd[7]);                                                     \
        v3 = _mm256_add_pd(v3, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx8, tw_brd[8]);                                                     \
        v4 = _mm256_add_pd(v4, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx8, tw_brd[9]);                                                     \
        v5 = _mm256_add_pd(v5, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx8, tw_brd[10]);                                                    \
        v6 = _mm256_add_pd(v6, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx8, tw_brd[11]);                                                    \
        v7 = _mm256_add_pd(v7, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx8, tw_brd[0]);                                                     \
        v8 = _mm256_add_pd(v8, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx8, tw_brd[1]);                                                     \
        v9 = _mm256_add_pd(v9, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx8, tw_brd[2]);                                                     \
        v10 = _mm256_add_pd(v10, tmp);                                                         \
        CMUL_FMA_R13(tmp, tx8, tw_brd[3]);                                                     \
        v11 = _mm256_add_pd(v11, tmp);                                                         \
                                                                                               \
        CMUL_FMA_R13(tmp, tx9, tw_brd[3]);                                                     \
        v0 = _mm256_add_pd(v0, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx9, tw_brd[4]);                                                     \
        v1 = _mm256_add_pd(v1, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx9, tw_brd[5]);                                                     \
        v2 = _mm256_add_pd(v2, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx9, tw_brd[6]);                                                     \
        v3 = _mm256_add_pd(v3, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx9, tw_brd[7]);                                                     \
        v4 = _mm256_add_pd(v4, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx9, tw_brd[8]);                                                     \
        v5 = _mm256_add_pd(v5, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx9, tw_brd[9]);                                                     \
        v6 = _mm256_add_pd(v6, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx9, tw_brd[10]);                                                    \
        v7 = _mm256_add_pd(v7, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx9, tw_brd[11]);                                                    \
        v8 = _mm256_add_pd(v8, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx9, tw_brd[0]);                                                     \
        v9 = _mm256_add_pd(v9, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx9, tw_brd[1]);                                                     \
        v10 = _mm256_add_pd(v10, tmp);                                                         \
        CMUL_FMA_R13(tmp, tx9, tw_brd[2]);                                                     \
        v11 = _mm256_add_pd(v11, tmp);                                                         \
                                                                                               \
        CMUL_FMA_R13(tmp, tx10, tw_brd[2]);                                                    \
        v0 = _mm256_add_pd(v0, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx10, tw_brd[3]);                                                    \
        v1 = _mm256_add_pd(v1, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx10, tw_brd[4]);                                                    \
        v2 = _mm256_add_pd(v2, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx10, tw_brd[5]);                                                    \
        v3 = _mm256_add_pd(v3, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx10, tw_brd[6]);                                                    \
        v4 = _mm256_add_pd(v4, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx10, tw_brd[7]);                                                    \
        v5 = _mm256_add_pd(v5, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx10, tw_brd[8]);                                                    \
        v6 = _mm256_add_pd(v6, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx10, tw_brd[9]);                                                    \
        v7 = _mm256_add_pd(v7, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx10, tw_brd[10]);                                                   \
        v8 = _mm256_add_pd(v8, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx10, tw_brd[11]);                                                   \
        v9 = _mm256_add_pd(v9, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx10, tw_brd[0]);                                                    \
        v10 = _mm256_add_pd(v10, tmp);                                                         \
        CMUL_FMA_R13(tmp, tx10, tw_brd[1]);                                                    \
        v11 = _mm256_add_pd(v11, tmp);                                                         \
                                                                                               \
        CMUL_FMA_R13(tmp, tx11, tw_brd[1]);                                                    \
        v0 = _mm256_add_pd(v0, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx11, tw_brd[2]);                                                    \
        v1 = _mm256_add_pd(v1, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx11, tw_brd[3]);                                                    \
        v2 = _mm256_add_pd(v2, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx11, tw_brd[4]);                                                    \
        v3 = _mm256_add_pd(v3, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx11, tw_brd[5]);                                                    \
        v4 = _mm256_add_pd(v4, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx11, tw_brd[6]);                                                    \
        v5 = _mm256_add_pd(v5, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx11, tw_brd[7]);                                                    \
        v6 = _mm256_add_pd(v6, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx11, tw_brd[8]);                                                    \
        v7 = _mm256_add_pd(v7, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx11, tw_brd[9]);                                                    \
        v8 = _mm256_add_pd(v8, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx11, tw_brd[10]);                                                   \
        v9 = _mm256_add_pd(v9, tmp);                                                           \
        CMUL_FMA_R13(tmp, tx11, tw_brd[11]);                                                   \
        v10 = _mm256_add_pd(v10, tmp);                                                         \
        CMUL_FMA_R13(tmp, tx11, tw_brd[0]);                                                    \
        v11 = _mm256_add_pd(v11, tmp);                                                         \
    } while (0)

/**
 * @brief Map 12 convolution outputs to final DFT outputs using output permutation
 */
#define MAP_RADER13_OUTPUTS_AVX2(x0, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, \
                                 y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12)    \
    do                                                                                 \
    {                                                                                  \
        y1 = _mm256_add_pd(x0, v0);   /* out_perm[0] = 1 */                            \
        y12 = _mm256_add_pd(x0, v1);  /* out_perm[1] = 12 */                           \
        y10 = _mm256_add_pd(x0, v2);  /* out_perm[2] = 10 */                           \
        y3 = _mm256_add_pd(x0, v3);   /* out_perm[3] = 3 */                            \
        y9 = _mm256_add_pd(x0, v4);   /* out_perm[4] = 9 */                            \
        y4 = _mm256_add_pd(x0, v5);   /* out_perm[5] = 4 */                            \
        y7 = _mm256_add_pd(x0, v6);   /* out_perm[6] = 7 */                            \
        y5 = _mm256_add_pd(x0, v7);   /* out_perm[7] = 5 */                            \
        y2 = _mm256_add_pd(x0, v8);   /* out_perm[8] = 2 */                            \
        y6 = _mm256_add_pd(x0, v9);   /* out_perm[9] = 6 */                            \
        y11 = _mm256_add_pd(x0, v10); /* out_perm[10] = 11 */                          \
        y8 = _mm256_add_pd(x0, v11);  /* out_perm[11] = 8 */                           \
    } while (0)
#endif

//==============================================================================
// SCALAR TWIDDLE PRECOMPUTATION
//==============================================================================

#define PRECOMPUTE_RADER13_TWIDDLES_SCALAR_FV(tw)                   \
    do                                                              \
    {                                                               \
        const int op[12] = RADER13_OUTPUT_PERM;                     \
        const double base_angle = -2.0 * M_PI / 13.0; /* FORWARD */ \
        for (int q = 0; q < 12; ++q)                                \
        {                                                           \
            double a = op[q] * base_angle;                          \
            sincos(a, &tw[q].im, &tw[q].re);                        \
        }                                                           \
    } while (0)

#define PRECOMPUTE_RADER13_TWIDDLES_SCALAR_BV(tw)                   \
    do                                                              \
    {                                                               \
        const int op[12] = RADER13_OUTPUT_PERM;                     \
        const double base_angle = +2.0 * M_PI / 13.0; /* INVERSE */ \
        for (int q = 0; q < 12; ++q)                                \
        {                                                           \
            double a = op[q] * base_angle;                          \
            sincos(a, &tw[q].im, &tw[q].re);                        \
        }                                                           \
    } while (0)
//==============================================================================
// LOAD/STORE HELPERS - Addressing Point 6 (alignment)
//==============================================================================

#ifdef __AVX2__
/**
 * @brief Load 13 lanes for 2 butterflies (AVX2 2x mode)
 */
#define LOAD_13_LANES_AVX2_2X(k, K, sub_outputs, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12) \
    do                                                                                                  \
    {                                                                                                   \
        x0 = load2_aos(&sub_outputs[k + 0 * K], &sub_outputs[k + 0 * K + 1]);                           \
        x1 = load2_aos(&sub_outputs[k + 1 * K], &sub_outputs[k + 1 * K + 1]);                           \
        x2 = load2_aos(&sub_outputs[k + 2 * K], &sub_outputs[k + 2 * K + 1]);                           \
        x3 = load2_aos(&sub_outputs[k + 3 * K], &sub_outputs[k + 3 * K + 1]);                           \
        x4 = load2_aos(&sub_outputs[k + 4 * K], &sub_outputs[k + 4 * K + 1]);                           \
        x5 = load2_aos(&sub_outputs[k + 5 * K], &sub_outputs[k + 5 * K + 1]);                           \
        x6 = load2_aos(&sub_outputs[k + 6 * K], &sub_outputs[k + 6 * K + 1]);                           \
        x7 = load2_aos(&sub_outputs[k + 7 * K], &sub_outputs[k + 7 * K + 1]);                           \
        x8 = load2_aos(&sub_outputs[k + 8 * K], &sub_outputs[k + 8 * K + 1]);                           \
        x9 = load2_aos(&sub_outputs[k + 9 * K], &sub_outputs[k + 9 * K + 1]);                           \
        x10 = load2_aos(&sub_outputs[k + 10 * K], &sub_outputs[k + 10 * K + 1]);                        \
        x11 = load2_aos(&sub_outputs[k + 11 * K], &sub_outputs[k + 11 * K + 1]);                        \
        x12 = load2_aos(&sub_outputs[k + 12 * K], &sub_outputs[k + 12 * K + 1]);                        \
    } while (0)

/**
 * @brief Store 13 lanes for 2 butterflies (Point 6: using STOREU_PD from simd_math.h)
 * Assumes simd_math.h handles alignment checks internally
 */
#define STORE_13_LANES_AVX2_2X(k, K, output, y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12) \
    do                                                                                              \
    {                                                                                               \
        STOREU_PD(&output[k + 0 * K].re, y0);                                                       \
        STOREU_PD(&output[k + 1 * K].re, y1);                                                       \
        STOREU_PD(&output[k + 2 * K].re, y2);                                                       \
        STOREU_PD(&output[k + 3 * K].re, y3);                                                       \
        STOREU_PD(&output[k + 4 * K].re, y4);                                                       \
        STOREU_PD(&output[k + 5 * K].re, y5);                                                       \
        STOREU_PD(&output[k + 6 * K].re, y6);                                                       \
        STOREU_PD(&output[k + 7 * K].re, y7);                                                       \
        STOREU_PD(&output[k + 8 * K].re, y8);                                                       \
        STOREU_PD(&output[k + 9 * K].re, y9);                                                       \
        STOREU_PD(&output[k + 10 * K].re, y10);                                                     \
        STOREU_PD(&output[k + 11 * K].re, y11);                                                     \
        STOREU_PD(&output[k + 12 * K].re, y12);                                                     \
    } while (0)

/**
 * @brief Apply stage twiddles for 2 butterflies (AVX2) - Point 7: Branchless
 *
 * stage_tw layout: [W^(1*k), W^(2*k), ..., W^(12*k)] for each k
 * Unconditionally loads and multiplies when K > 1; compiler will optimize out dead code for K=1
 */
#define APPLY_STAGE_TWIDDLES_R13_AVX2_2X(k, K, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, stage_tw) \
    do                                                                                                      \
    {                                                                                                       \
        /* Branchless: Always load, use conditional move semantics */                                       \
        const int do_twiddle = (K > 1);                                                                     \
        if (do_twiddle)                                                                                     \
        {                                                                                                   \
            __m256d w1 = load2_aos(&stage_tw[12 * k + 0], &stage_tw[12 * (k + 1) + 0]);                     \
            __m256d w2 = load2_aos(&stage_tw[12 * k + 1], &stage_tw[12 * (k + 1) + 1]);                     \
            __m256d w3 = load2_aos(&stage_tw[12 * k + 2], &stage_tw[12 * (k + 1) + 2]);                     \
            __m256d w4 = load2_aos(&stage_tw[12 * k + 3], &stage_tw[12 * (k + 1) + 3]);                     \
            __m256d w5 = load2_aos(&stage_tw[12 * k + 4], &stage_tw[12 * (k + 1) + 4]);                     \
            __m256d w6 = load2_aos(&stage_tw[12 * k + 5], &stage_tw[12 * (k + 1) + 5]);                     \
            __m256d w7 = load2_aos(&stage_tw[12 * k + 6], &stage_tw[12 * (k + 1) + 6]);                     \
            __m256d w8 = load2_aos(&stage_tw[12 * k + 7], &stage_tw[12 * (k + 1) + 7]);                     \
            __m256d w9 = load2_aos(&stage_tw[12 * k + 8], &stage_tw[12 * (k + 1) + 8]);                     \
            __m256d w10 = load2_aos(&stage_tw[12 * k + 9], &stage_tw[12 * (k + 1) + 9]);                    \
            __m256d w11 = load2_aos(&stage_tw[12 * k + 10], &stage_tw[12 * (k + 1) + 10]);                  \
            __m256d w12 = load2_aos(&stage_tw[12 * k + 11], &stage_tw[12 * (k + 1) + 11]);                  \
            CMUL_FMA_R13(x1, x1, w1);                                                                       \
            CMUL_FMA_R13(x2, x2, w2);                                                                       \
            CMUL_FMA_R13(x3, x3, w3);                                                                       \
            CMUL_FMA_R13(x4, x4, w4);                                                                       \
            CMUL_FMA_R13(x5, x5, w5);                                                                       \
            CMUL_FMA_R13(x6, x6, w6);                                                                       \
            CMUL_FMA_R13(x7, x7, w7);                                                                       \
            CMUL_FMA_R13(x8, x8, w8);                                                                       \
            CMUL_FMA_R13(x9, x9, w9);                                                                       \
            CMUL_FMA_R13(x10, x10, w10);                                                                    \
            CMUL_FMA_R13(x11, x11, w11);                                                                    \
            CMUL_FMA_R13(x12, x12, w12);                                                                    \
        }                                                                                                   \
    } while (0)

/**
 * @brief Compute Y0 = sum of all 13 inputs
 */
#define COMPUTE_Y0_AVX2(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, y0)      \
    do                                                                                  \
    {                                                                                   \
        y0 = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(x0, x1),           \
                                                       _mm256_add_pd(x2, x3)),          \
                                         _mm256_add_pd(_mm256_add_pd(x4, x5),           \
                                                       _mm256_add_pd(x6, x7))),         \
                           _mm256_add_pd(_mm256_add_pd(x8, x9),                         \
                                         _mm256_add_pd(_mm256_add_pd(x10, x11), x12))); \
    } while (0)

/**
 * @brief Apply Rader input permutation [1,2,4,8,3,6,12,11,9,5,10,7]
 */
#define RADER13_INPUT_PERMUTE_AVX2(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12,            \
                                   tx0, tx1, tx2, tx3, tx4, tx5, tx6, tx7, tx8, tx9, tx10, tx11) \
    do                                                                                           \
    {                                                                                            \
        tx0 = x1;   /* perm[0] = 1 */                                                            \
        tx1 = x2;   /* perm[1] = 2 */                                                            \
        tx2 = x4;   /* perm[2] = 4 */                                                            \
        tx3 = x8;   /* perm[3] = 8 */                                                            \
        tx4 = x3;   /* perm[4] = 3 */                                                            \
        tx5 = x6;   /* perm[5] = 6 */                                                            \
        tx6 = x12;  /* perm[6] = 12 */                                                           \
        tx7 = x11;  /* perm[7] = 11 */                                                           \
        tx8 = x9;   /* perm[8] = 9 */                                                            \
        tx9 = x5;   /* perm[9] = 5 */                                                            \
        tx10 = x10; /* perm[10] = 10 */                                                          \
        tx11 = x7;  /* perm[11] = 7 */                                                           \
    } while (0)
#endif

//==============================================================================
// SCALAR BUTTERFLY - COMPLETE IMPLEMENTATION
//==============================================================================

/**
 * @brief Complete scalar radix-13 butterfly using Rader's algorithm
 * Direction-agnostic - twiddles precomputed with correct sign
 */
#define RADIX13_BUTTERFLY_SCALAR(k, K, sub_outputs, stage_tw, tw_scalar, output_buffer)          \
    do                                                                                           \
    {                                                                                            \
        /* Load 13 lanes */                                                                      \
        const fft_data x0 = sub_outputs[k + 0 * K];                                              \
        const fft_data x1 = sub_outputs[k + 1 * K];                                              \
        const fft_data x2 = sub_outputs[k + 2 * K];                                              \
        const fft_data x3 = sub_outputs[k + 3 * K];                                              \
        const fft_data x4 = sub_outputs[k + 4 * K];                                              \
        const fft_data x5 = sub_outputs[k + 5 * K];                                              \
        const fft_data x6 = sub_outputs[k + 6 * K];                                              \
        const fft_data x7 = sub_outputs[k + 7 * K];                                              \
        const fft_data x8 = sub_outputs[k + 8 * K];                                              \
        const fft_data x9 = sub_outputs[k + 9 * K];                                              \
        const fft_data x10 = sub_outputs[k + 10 * K];                                            \
        const fft_data x11 = sub_outputs[k + 11 * K];                                            \
        const fft_data x12 = sub_outputs[k + 12 * K];                                            \
                                                                                                 \
        /* Apply stage twiddles */                                                               \
        double x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i, x5r, x5i, x6r, x6i;                       \
        double x7r, x7i, x8r, x8i, x9r, x9i, x10r, x10i, x11r, x11i, x12r, x12i;                 \
        if (K > 1)                                                                               \
        {                                                                                        \
            const fft_data w1 = stage_tw[12 * k + 0];                                            \
            const fft_data w2 = stage_tw[12 * k + 1];                                            \
            const fft_data w3 = stage_tw[12 * k + 2];                                            \
            const fft_data w4 = stage_tw[12 * k + 3];                                            \
            const fft_data w5 = stage_tw[12 * k + 4];                                            \
            const fft_data w6 = stage_tw[12 * k + 5];                                            \
            const fft_data w7 = stage_tw[12 * k + 6];                                            \
            const fft_data w8 = stage_tw[12 * k + 7];                                            \
            const fft_data w9 = stage_tw[12 * k + 8];                                            \
            const fft_data w10 = stage_tw[12 * k + 9];                                           \
            const fft_data w11 = stage_tw[12 * k + 10];                                          \
            const fft_data w12 = stage_tw[12 * k + 11];                                          \
            x1r = x1.re * w1.re - x1.im * w1.im;                                                 \
            x1i = x1.re * w1.im + x1.im * w1.re;                                                 \
            x2r = x2.re * w2.re - x2.im * w2.im;                                                 \
            x2i = x2.re * w2.im + x2.im * w2.re;                                                 \
            x3r = x3.re * w3.re - x3.im * w3.im;                                                 \
            x3i = x3.re * w3.im + x3.im * w3.re;                                                 \
            x4r = x4.re * w4.re - x4.im * w4.im;                                                 \
            x4i = x4.re * w4.im + x4.im * w4.re;                                                 \
            x5r = x5.re * w5.re - x5.im * w5.im;                                                 \
            x5i = x5.re * w5.im + x5.im * w5.re;                                                 \
            x6r = x6.re * w6.re - x6.im * w6.im;                                                 \
            x6i = x6.re * w6.im + x6.im * w6.re;                                                 \
            x7r = x7.re * w7.re - x7.im * w7.im;                                                 \
            x7i = x7.re * w7.im + x7.im * w7.re;                                                 \
            x8r = x8.re * w8.re - x8.im * w8.im;                                                 \
            x8i = x8.re * w8.im + x8.im * w8.re;                                                 \
            x9r = x9.re * w9.re - x9.im * w9.im;                                                 \
            x9i = x9.re * w9.im + x9.im * w9.re;                                                 \
            x10r = x10.re * w10.re - x10.im * w10.im;                                            \
            x10i = x10.re * w10.im + x10.im * w10.re;                                            \
            x11r = x11.re * w11.re - x11.im * w11.im;                                            \
            x11i = x11.re * w11.im + x11.im * w11.re;                                            \
            x12r = x12.re * w12.re - x12.im * w12.im;                                            \
            x12i = x12.re * w12.im + x12.im * w12.re;                                            \
        }                                                                                        \
        else                                                                                     \
        {                                                                                        \
            x1r = x1.re;                                                                         \
            x1i = x1.im;                                                                         \
            x2r = x2.re;                                                                         \
            x2i = x2.im;                                                                         \
            x3r = x3.re;                                                                         \
            x3i = x3.im;                                                                         \
            x4r = x4.re;                                                                         \
            x4i = x4.im;                                                                         \
            x5r = x5.re;                                                                         \
            x5i = x5.im;                                                                         \
            x6r = x6.re;                                                                         \
            x6i = x6.im;                                                                         \
            x7r = x7.re;                                                                         \
            x7i = x7.im;                                                                         \
            x8r = x8.re;                                                                         \
            x8i = x8.im;                                                                         \
            x9r = x9.re;                                                                         \
            x9i = x9.im;                                                                         \
            x10r = x10.re;                                                                       \
            x10i = x10.im;                                                                       \
            x11r = x11.re;                                                                       \
            x11i = x11.im;                                                                       \
            x12r = x12.re;                                                                       \
            x12i = x12.im;                                                                       \
        }                                                                                        \
                                                                                                 \
        /* Y0 */                                                                                 \
        fft_data y0 = {                                                                          \
            x0.re + (x1r + x2r + x3r + x4r + x5r + x6r + x7r + x8r + x9r + x10r + x11r + x12r),  \
            x0.im + (x1i + x2i + x3i + x4i + x5i + x6i + x7i + x8i + x9i + x10i + x11i + x12i)}; \
                                                                                                 \
        /* Rader input permutation */                                                            \
        fft_data tx[12];                                                                         \
        tx[0].re = x1r;                                                                          \
        tx[0].im = x1i;                                                                          \
        tx[1].re = x2r;                                                                          \
        tx[1].im = x2i;                                                                          \
        tx[2].re = x4r;                                                                          \
        tx[2].im = x4i;                                                                          \
        tx[3].re = x8r;                                                                          \
        tx[3].im = x8i;                                                                          \
        tx[4].re = x3r;                                                                          \
        tx[4].im = x3i;                                                                          \
        tx[5].re = x6r;                                                                          \
        tx[5].im = x6i;                                                                          \
        tx[6].re = x12r;                                                                         \
        tx[6].im = x12i;                                                                         \
        tx[7].re = x11r;                                                                         \
        tx[7].im = x11i;                                                                         \
        tx[8].re = x9r;                                                                          \
        tx[8].im = x9i;                                                                          \
        tx[9].re = x5r;                                                                          \
        tx[9].im = x5i;                                                                          \
        tx[10].re = x10r;                                                                        \
        tx[10].im = x10i;                                                                        \
        tx[11].re = x7r;                                                                         \
        tx[11].im = x7i;                                                                         \
                                                                                                 \
        /* 12-point cyclic convolution */                                                        \
        fft_data v[12];                                                                          \
        for (int q = 0; q < 12; ++q)                                                             \
        {                                                                                        \
            v[q].re = 0.0;                                                                       \
            v[q].im = 0.0;                                                                       \
            for (int l = 0; l < 12; ++l)                                                         \
            {                                                                                    \
                int idx = (q - l + 12) % 12;                                                     \
                v[q].re += tx[l].re * tw_scalar[idx].re - tx[l].im * tw_scalar[idx].im;          \
                v[q].im += tx[l].re * tw_scalar[idx].im + tx[l].im * tw_scalar[idx].re;          \
            }                                                                                    \
        }                                                                                        \
                                                                                                 \
        /* Map to outputs */                                                                     \
        output_buffer[k + 0 * K] = y0;                                                           \
        output_buffer[k + 1 * K] = (fft_data){x0.re + v[0].re, x0.im + v[0].im};                 \
        output_buffer[k + 2 * K] = (fft_data){x0.re + v[8].re, x0.im + v[8].im};                 \
        output_buffer[k + 3 * K] = (fft_data){x0.re + v[3].re, x0.im + v[3].im};                 \
        output_buffer[k + 4 * K] = (fft_data){x0.re + v[5].re, x0.im + v[5].im};                 \
        output_buffer[k + 5 * K] = (fft_data){x0.re + v[7].re, x0.im + v[7].im};                 \
        output_buffer[k + 6 * K] = (fft_data){x0.re + v[9].re, x0.im + v[9].im};                 \
        output_buffer[k + 7 * K] = (fft_data){x0.re + v[6].re, x0.im + v[6].im};                 \
        output_buffer[k + 8 * K] = (fft_data){x0.re + v[11].re, x0.im + v[11].im};               \
        output_buffer[k + 9 * K] = (fft_data){x0.re + v[4].re, x0.im + v[4].im};                 \
        output_buffer[k + 10 * K] = (fft_data){x0.re + v[2].re, x0.im + v[2].im};                \
        output_buffer[k + 11 * K] = (fft_data){x0.re + v[10].re, x0.im + v[10].im};              \
        output_buffer[k + 12 * K] = (fft_data){x0.re + v[1].re, x0.im + v[1].im};                \
    } while (0)

#endif // FFT_RADIX13_MACROS_H