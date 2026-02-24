/**
 * @file  avx2_tail_mask.h
 * @brief AVX2 masked load/store utilities for tail handling (1..3 elements)
 *
 * Eliminates scalar tail code by using _mm256_maskload_pd / _mm256_maskstore_pd
 * with a compile-time mask lookup table.
 *
 * AVX2 mask convention: sign bit of each 64-bit lane selects active (−1) or
 * inactive (0). Inactive lanes read as zero, writes are suppressed.
 *
 * USAGE:
 *   int tail = K - k;                        // 1, 2, or 3
 *   __m256i mask = avx2_tail_mask(tail);
 *   __m256d v = _mm256_maskload_pd(ptr, mask);  // zeros upper lanes
 *   _mm256_maskstore_pd(ptr, mask, v);           // writes only valid lanes
 *
 * REUSABLE: This file has no radix-specific dependencies.
 *           Include from radix-3, radix-5, radix-7, etc.
 */

#ifndef AVX2_TAIL_MASK_H
#define AVX2_TAIL_MASK_H

#include <immintrin.h>
#include <stdint.h>

/* ================================================================== */
/*  Mask lookup table — 4 entries, 32-byte aligned for aligned load    */
/*                                                                     */
/*  Index 0: unused (tail=0 means no remainder)                       */
/*  Index 1: lane 0 active                                            */
/*  Index 2: lanes 0-1 active                                         */
/*  Index 3: lanes 0-2 active                                         */
/* ================================================================== */
static const int64_t avx2_tail_mask_table[4][4]
    __attribute__((aligned(32))) = {
        {0, 0, 0, 0},    /* tail=0: no-op                  */
        {-1, 0, 0, 0},   /* tail=1: lane 0 only            */
        {-1, -1, 0, 0},  /* tail=2: lanes 0-1              */
        {-1, -1, -1, 0}, /* tail=3: lanes 0-2              */
};

/**
 * @brief Get AVX2 mask for given tail count (1..3)
 *
 * Returns __m256i with sign-bit masks suitable for
 * _mm256_maskload_pd / _mm256_maskstore_pd.
 *
 * @param tail  Number of valid elements (1, 2, or 3)
 * @return      __m256i mask
 */
static inline __attribute__((always_inline))
__m256i
avx2_tail_mask(int tail)
{
    return _mm256_load_si256((const __m256i *)avx2_tail_mask_table[tail]);
}

/* ================================================================== */
/*  Convenience macros                                                 */
/* ================================================================== */

/** Masked load: returns __m256d with inactive lanes zeroed */
#define AVX2_MASKLOAD_PD(ptr, mask) \
    _mm256_maskload_pd((const double *)(ptr), (mask))

/** Masked store: only writes active lanes, others untouched */
#define AVX2_MASKSTORE_PD(ptr, mask, val) \
    _mm256_maskstore_pd((double *)(ptr), (mask), (val))

#endif /* AVX2_TAIL_MASK_H */
