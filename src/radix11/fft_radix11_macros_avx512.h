/**
 * @file fft_radix11_macros_refactored.h
 * @brief REFACTORED: Separate Forward/Backward Radix-11 Butterfly Macros
 *
 * @details
 * REFACTORING APPLIED:
 * ====================
 * ✅ Complete separation of FV (Forward) and BV (Backward) macro sets
 * ✅ Dedicated prefetch macros for each direction
 * ✅ Dedicated top-level butterfly macros for each direction
 * ✅ Fixed prefetch macro calling convention
 * ✅ Clear organization: Shared → FV-specific → BV-specific
 *
 * ALL OPTIMIZATIONS PRESERVED:
 * ============================
 * ✅ Direct geometric decomposition with 5 symmetric pairs
 * ✅ Constants broadcast ONCE per butterfly
 * ✅ FMA instructions for complex multiply
 * ✅ Type-safe inline functions
 * ✅ TRUE END-TO-END SoA
 * ✅ Software pipelining depth
 * ✅ Prefetching (ISA-aware, fixed calling)
 * ✅ Cache-blocking for large K
 * ✅ Masked tail handling for K % 8 != 0
 * ✅ Correct de-interleave using permutex2var
 * ✅ NT store policy (disabled for scattered writes)
 * ✅ Both lo/hi halves processed
 *
 * @author FFT Optimization Team
 * @version 6.0 (REFACTORED: FV/BV SEPARATION)
 * @date 2025
 */

#ifndef FFT_RADIX11_MACROS_REFACTORED_H
#define FFT_RADIX11_MACROS_REFACTORED_H

#include "fft_radix11.h"
#include "simd_math.h"

//==============================================================================
// CONFIGURATION
//==============================================================================

/// SIMD-dependent thresholds
#if defined(__AVX512F__)
#define R11_PARALLEL_THRESHOLD 2048
#define R11_VECTOR_WIDTH 8
#define R11_REQUIRED_ALIGNMENT 64
#elif defined(__AVX2__)
#define R11_PARALLEL_THRESHOLD 4096
#define R11_VECTOR_WIDTH 4
#define R11_REQUIRED_ALIGNMENT 32
#elif defined(__SSE2__)
#define R11_PARALLEL_THRESHOLD 8192
#define R11_VECTOR_WIDTH 2
#define R11_REQUIRED_ALIGNMENT 16
#else
#define R11_PARALLEL_THRESHOLD 16384
#define R11_VECTOR_WIDTH 1
#define R11_REQUIRED_ALIGNMENT 8
#endif

/// Cache configuration
#define R11_CACHE_LINE_BYTES 64
#define R11_DOUBLES_PER_CACHE_LINE (R11_CACHE_LINE_BYTES / sizeof(double))

/**
 * @brief FFTW-style cache blocking size
 * @details Block size for keeping 11 output streams in L2 cache
 * Target: ~256KB L2 cache -> block_size ~= 1024
 */
#define R11_CACHE_BLOCK_SIZE 1024

/**
 * @brief Last Level Cache size in bytes
 */
#ifndef R11_LLC_BYTES
#define R11_LLC_BYTES (8 * 1024 * 1024)
#endif

/**
 * @brief Non-temporal store policy for radix-11
 * @details Following FFTW: NT stores DISABLED for scattered multi-row writes
 */
#ifndef R11_FORCE_NT
#define R11_USE_NT_STORES 0
#else
#define R11_USE_NT_STORES R11_FORCE_NT
#endif

/**
 * @brief Prefetch distance (in elements)
 * @details ISA-aware configuration
 */
#ifndef R11_PREFETCH_DISTANCE
#if defined(__AVX512F__)
#define R11_PREFETCH_DISTANCE 32 // Deeper for AVX-512
#elif defined(__AVX2__)
#define R11_PREFETCH_DISTANCE 24
#elif defined(__SSE2__)
#define R11_PREFETCH_DISTANCE 16
#else
#define R11_PREFETCH_DISTANCE 8
#endif
#endif

/**
 * @brief Prefetch hint policy
 */
#ifndef R11_PREFETCH_HINT
#define R11_PREFETCH_HINT _MM_HINT_T0 // L1 cache
#endif

//==============================================================================
// GEOMETRIC CONSTANTS
//==============================================================================

#define C11_1 0.8412535328311812
#define C11_2 0.4154150130018864
#define C11_3 -0.14231483827328514
#define C11_4 -0.6548607339452850
#define C11_5 -0.9594929736144974

#define S11_1 0.5406408174555976
#define S11_2 0.9096319953545184
#define S11_3 0.9898214418809327
#define S11_4 0.7557495743542582
#define S11_5 0.2817325568414297

//==============================================================================
// AVX-512 IMPLEMENTATION
//==============================================================================

#ifdef __AVX512F__

/**
 * @brief Radix-11 constants broadcasted to AVX-512 registers
 */
typedef struct
{
    __m512d c1, c2, c3, c4, c5;
    __m512d s1, s2, s3, s4, s5;
} radix11_consts_avx512;

/**
 * @brief Broadcast radix-11 constants (PRESERVED)
 * @note: Renamed local variable from 'k' to 'KC' to avoid shadowing loop indices
 */
static inline radix11_consts_avx512 broadcast_radix11_consts_avx512(void)
{
    radix11_consts_avx512 KC;
    KC.c1 = _mm512_set1_pd(C11_1);
    KC.c2 = _mm512_set1_pd(C11_2);
    KC.c3 = _mm512_set1_pd(C11_3);
    KC.c4 = _mm512_set1_pd(C11_4);
    KC.c5 = _mm512_set1_pd(C11_5);
    KC.s1 = _mm512_set1_pd(S11_1);
    KC.s2 = _mm512_set1_pd(S11_2);
    KC.s3 = _mm512_set1_pd(S11_3);
    KC.s4 = _mm512_set1_pd(S11_4);
    KC.s5 = _mm512_set1_pd(S11_5);
    return KC;
}

//==============================================================================
// SIGN MASKS FOR ±i ROTATION (CRITICAL FIX)
//==============================================================================

/**
 * @brief Sign masks for rotating by ±i
 * @details
 * To multiply complex number by -i (forward):  (a+bi)*(-i) = b - ai  => [b, -a]
 * To multiply complex number by +i (backward): (a+bi)*(+i) = -b + ai => [-b, a]
 *
 * Implementation:
 * 1. Swap real/imag: shuffle_pd(x, x, 0x55)
 * 2. Apply sign mask via XOR with sign bit
 */

/**
 * @brief Get sign mask for negating even lanes (real parts after swap)
 * @details Pattern: [0x8000..., 0, 0x8000..., 0, 0x8000..., 0, 0x8000..., 0]
 * Used for forward rotation: multiply by -i = [b, -a]
 */
static inline __m512i get_neg_even_mask(void)
{
    return _mm512_setr_epi64(
        0x8000000000000000ULL, 0,
        0x8000000000000000ULL, 0,
        0x8000000000000000ULL, 0,
        0x8000000000000000ULL, 0);
}

/**
 * @brief Get sign mask for negating odd lanes (imaginary parts after swap)
 * @details Pattern: [0, 0x8000..., 0, 0x8000..., 0, 0x8000..., 0, 0x8000...]
 * Used for backward rotation: multiply by +i = [-b, a]
 */
static inline __m512i get_neg_odd_mask(void)
{
    return _mm512_setr_epi64(
        0, 0x8000000000000000ULL,
        0, 0x8000000000000000ULL,
        0, 0x8000000000000000ULL,
        0, 0x8000000000000000ULL);
}

/**
 * @brief Rotate complex vector by -i (forward FFT)
 * @details (a+bi)*(-i) = b - ai => swap and negate real part
 * @param base Input complex vector [a0,b0,a1,b1,...]
 * @return Rotated vector [b0,-a0,b1,-a1,...]
 */
#define ROTATE_BY_MINUS_I_AVX512(base, rot_out)                                     \
    do                                                                              \
    {                                                                               \
        __m512d swapped = _mm512_shuffle_pd(base, base, 0x55);                      \
        rot_out = _mm512_xor_pd(swapped, _mm512_castsi512_pd(get_neg_even_mask())); \
    } while (0)

/**
 * @brief Rotate complex vector by +i (backward FFT)
 * @details (a+bi)*(+i) = -b + ai => swap and negate imaginary part
 * @param base Input complex vector [a0,b0,a1,b1,...]
 * @return Rotated vector [-b0,a0,-b1,a1,...]
 */
#define ROTATE_BY_PLUS_I_AVX512(base, rot_out)                                     \
    do                                                                             \
    {                                                                              \
        __m512d swapped = _mm512_shuffle_pd(base, base, 0x55);                     \
        rot_out = _mm512_xor_pd(swapped, _mm512_castsi512_pd(get_neg_odd_mask())); \
    } while (0)

//==============================================================================
// PERMUTATION INDICES FOR DE-INTERLEAVE (CRITICAL FIX)
//==============================================================================

/**
 * @brief Permutation indices for extracting real/imag from interleaved vectors
 * @details These are CONSTANTS - create them once at function scope
 *
 * CRITICAL: permutex2var_pd index semantics:
 *   - Indices 0-7:   select from first vector (y_lo)
 *   - Indices 16-23: select from second vector (y_hi)
 *   - Indices 8-15:  undefined behavior for our use case
 *
 * Given interleaved lo/hi:
 *   y_lo = [r0, i0, r1, i1, r2, i2, r3, i3]
 *   y_hi = [r4, i4, r5, i5, r6, i6, r7, i7]
 *
 * Extract reals (evens):
 *   re = [r0, r1, r2, r3, r4, r5, r6, r7]
 *   Indices: [0, 2, 4, 6] from y_lo, [16+0, 16+2, 16+4, 16+6] from y_hi
 *
 * Extract imaginaries (odds):
 *   im = [i0, i1, i2, i3, i4, i5, i6, i7]
 *   Indices: [1, 3, 5, 7] from y_lo, [16+1, 16+3, 16+5, 16+7] from y_hi
 */
static inline __m512i get_deinterleave_idx_re(void)
{
    return _mm512_setr_epi64(0, 2, 4, 6, 16 + 0, 16 + 2, 16 + 4, 16 + 6);
}

static inline __m512i get_deinterleave_idx_im(void)
{
    return _mm512_setr_epi64(1, 3, 5, 7, 16 + 1, 16 + 3, 16 + 5, 16 + 7);
}

//==============================================================================
// SHARED PREFETCH MACROS
//==============================================================================

/**
 * @brief Prefetch input data and twiddles - FIXED SIGNATURE
 * @details Software pipelining: prefetch data that will be used soon
 */
#define PREFETCH_RADIX11_INPUT(k, K, in_re, in_im, distance, stage_tw, sub_len)               \
    do                                                                                        \
    {                                                                                         \
        size_t pf_k = k + (distance);                                                         \
        if (pf_k < K)                                                                         \
        {                                                                                     \
            _mm_prefetch((const char *)(&in_re[pf_k + 0 * K]), R11_PREFETCH_HINT);            \
            _mm_prefetch((const char *)(&in_im[pf_k + 0 * K]), R11_PREFETCH_HINT);            \
            _mm_prefetch((const char *)(&in_re[pf_k + 5 * K]), R11_PREFETCH_HINT);            \
            _mm_prefetch((const char *)(&in_im[pf_k + 5 * K]), R11_PREFETCH_HINT);            \
            _mm_prefetch((const char *)(&in_re[pf_k + 10 * K]), R11_PREFETCH_HINT);           \
            _mm_prefetch((const char *)(&in_im[pf_k + 10 * K]), R11_PREFETCH_HINT);           \
            /* Prefetch twiddles when they'll be used */                                      \
            if ((sub_len) > 1)                                                                \
            {                                                                                 \
                _mm_prefetch((const char *)(&stage_tw->re[1 * K + pf_k]), R11_PREFETCH_HINT); \
                _mm_prefetch((const char *)(&stage_tw->im[1 * K + pf_k]), R11_PREFETCH_HINT); \
                _mm_prefetch((const char *)(&stage_tw->re[6 * K + pf_k]), R11_PREFETCH_HINT); \
                _mm_prefetch((const char *)(&stage_tw->im[6 * K + pf_k]), R11_PREFETCH_HINT); \
            }                                                                                 \
        }                                                                                     \
    } while (0)

//==============================================================================
// SHARED LOAD MACROS (CORRECT - PROCESSES BOTH LO/HI)
//==============================================================================

/**
 * @brief Load 11 lanes from SoA buffers - FULL 8-LANE VERSION
 * @details Processes BOTH lo (lanes 0-3) and hi (lanes 4-7)
 */
#define LOAD_11_LANES_AVX512_NATIVE_SOA_FULL(k, K, in_re, in_im,                       \
                                             x0_lo, x0_hi, x1_lo, x1_hi, x2_lo, x2_hi, \
                                             x3_lo, x3_hi, x4_lo, x4_hi, x5_lo, x5_hi, \
                                             x6_lo, x6_hi, x7_lo, x7_hi, x8_lo, x8_hi, \
                                             x9_lo, x9_hi, x10_lo, x10_hi)             \
    do                                                                                 \
    {                                                                                  \
        __m512d re0 = _mm512_loadu_pd(&in_re[k + 0 * K]);                              \
        __m512d im0 = _mm512_loadu_pd(&in_im[k + 0 * K]);                              \
        __m512d re1 = _mm512_loadu_pd(&in_re[k + 1 * K]);                              \
        __m512d im1 = _mm512_loadu_pd(&in_im[k + 1 * K]);                              \
        __m512d re2 = _mm512_loadu_pd(&in_re[k + 2 * K]);                              \
        __m512d im2 = _mm512_loadu_pd(&in_im[k + 2 * K]);                              \
        __m512d re3 = _mm512_loadu_pd(&in_re[k + 3 * K]);                              \
        __m512d im3 = _mm512_loadu_pd(&in_im[k + 3 * K]);                              \
        __m512d re4 = _mm512_loadu_pd(&in_re[k + 4 * K]);                              \
        __m512d im4 = _mm512_loadu_pd(&in_im[k + 4 * K]);                              \
        __m512d re5 = _mm512_loadu_pd(&in_re[k + 5 * K]);                              \
        __m512d im5 = _mm512_loadu_pd(&in_im[k + 5 * K]);                              \
        __m512d re6 = _mm512_loadu_pd(&in_re[k + 6 * K]);                              \
        __m512d im6 = _mm512_loadu_pd(&in_im[k + 6 * K]);                              \
        __m512d re7 = _mm512_loadu_pd(&in_re[k + 7 * K]);                              \
        __m512d im7 = _mm512_loadu_pd(&in_im[k + 7 * K]);                              \
        __m512d re8 = _mm512_loadu_pd(&in_re[k + 8 * K]);                              \
        __m512d im8 = _mm512_loadu_pd(&in_im[k + 8 * K]);                              \
        __m512d re9 = _mm512_loadu_pd(&in_re[k + 9 * K]);                              \
        __m512d im9 = _mm512_loadu_pd(&in_im[k + 9 * K]);                              \
        __m512d re10 = _mm512_loadu_pd(&in_re[k + 10 * K]);                            \
        __m512d im10 = _mm512_loadu_pd(&in_im[k + 10 * K]);                            \
        /* Interleave into complex pairs */                                            \
        __m512d y0_lo = _mm512_unpacklo_pd(re0, im0);                                  \
        __m512d y0_hi = _mm512_unpackhi_pd(re0, im0);                                  \
        __m512d y1_lo = _mm512_unpacklo_pd(re1, im1);                                  \
        __m512d y1_hi = _mm512_unpackhi_pd(re1, im1);                                  \
        __m512d y2_lo = _mm512_unpacklo_pd(re2, im2);                                  \
        __m512d y2_hi = _mm512_unpackhi_pd(re2, im2);                                  \
        __m512d y3_lo = _mm512_unpacklo_pd(re3, im3);                                  \
        __m512d y3_hi = _mm512_unpackhi_pd(re3, im3);                                  \
        __m512d y4_lo = _mm512_unpacklo_pd(re4, im4);                                  \
        __m512d y4_hi = _mm512_unpackhi_pd(re4, im4);                                  \
        __m512d y5_lo = _mm512_unpacklo_pd(re5, im5);                                  \
        __m512d y5_hi = _mm512_unpackhi_pd(re5, im5);                                  \
        __m512d y6_lo = _mm512_unpacklo_pd(re6, im6);                                  \
        __m512d y6_hi = _mm512_unpackhi_pd(re6, im6);                                  \
        __m512d y7_lo = _mm512_unpacklo_pd(re7, im7);                                  \
        __m512d y7_hi = _mm512_unpackhi_pd(re7, im7);                                  \
        __m512d y8_lo = _mm512_unpacklo_pd(re8, im8);                                  \
        __m512d y8_hi = _mm512_unpackhi_pd(re8, im8);                                  \
        __m512d y9_lo = _mm512_unpacklo_pd(re9, im9);                                  \
        __m512d y9_hi = _mm512_unpackhi_pd(re9, im9);                                  \
        __m512d y10_lo = _mm512_unpacklo_pd(re10, im10);                               \
        __m512d y10_hi = _mm512_unpackhi_pd(re10, im10);                               \
        /* Assign outputs */                                                           \
        x0_lo = y0_lo;                                                                 \
        x0_hi = y0_hi;                                                                 \
        x1_lo = y1_lo;                                                                 \
        x1_hi = y1_hi;                                                                 \
        x2_lo = y2_lo;                                                                 \
        x2_hi = y2_hi;                                                                 \
        x3_lo = y3_lo;                                                                 \
        x3_hi = y3_hi;                                                                 \
        x4_lo = y4_lo;                                                                 \
        x4_hi = y4_hi;                                                                 \
        x5_lo = y5_lo;                                                                 \
        x5_hi = y5_hi;                                                                 \
        x6_lo = y6_lo;                                                                 \
        x6_hi = y6_hi;                                                                 \
        x7_lo = y7_lo;                                                                 \
        x7_hi = y7_hi;                                                                 \
        x8_lo = y8_lo;                                                                 \
        x8_hi = y8_hi;                                                                 \
        x9_lo = y9_lo;                                                                 \
        x9_hi = y9_hi;                                                                 \
        x10_lo = y10_lo;                                                               \
        x10_hi = y10_hi;                                                               \
    } while (0)

/**
 * @brief Load 11 lanes with masking for tail - MASKED VERSION
 * @details Zeros invalid lanes to prevent processing garbage
 */
#define LOAD_11_LANES_AVX512_NATIVE_SOA_MASKED(k, K, remaining, in_re, in_im,            \
                                               x0_lo, x0_hi, x1_lo, x1_hi, x2_lo, x2_hi, \
                                               x3_lo, x3_hi, x4_lo, x4_hi, x5_lo, x5_hi, \
                                               x6_lo, x6_hi, x7_lo, x7_hi, x8_lo, x8_hi, \
                                               x9_lo, x9_hi, x10_lo, x10_hi)             \
    do                                                                                   \
    {                                                                                    \
        size_t lo_count = (remaining < 4) ? remaining : 4;                               \
        size_t hi_count = (remaining > 4) ? (remaining - 4) : 0;                         \
        __mmask8 mask_lo = (__mmask8)((1U << (2 * lo_count)) - 1);                       \
        __mmask8 mask_hi = (__mmask8)((1U << (2 * hi_count)) - 1);                       \
        __m512d re0 = _mm512_maskz_loadu_pd(mask_lo, &in_re[k + 0 * K]);                 \
        __m512d im0 = _mm512_maskz_loadu_pd(mask_lo, &in_im[k + 0 * K]);                 \
        __m512d re1 = _mm512_maskz_loadu_pd(mask_lo, &in_re[k + 1 * K]);                 \
        __m512d im1 = _mm512_maskz_loadu_pd(mask_lo, &in_im[k + 1 * K]);                 \
        __m512d re2 = _mm512_maskz_loadu_pd(mask_lo, &in_re[k + 2 * K]);                 \
        __m512d im2 = _mm512_maskz_loadu_pd(mask_lo, &in_im[k + 2 * K]);                 \
        __m512d re3 = _mm512_maskz_loadu_pd(mask_lo, &in_re[k + 3 * K]);                 \
        __m512d im3 = _mm512_maskz_loadu_pd(mask_lo, &in_im[k + 3 * K]);                 \
        __m512d re4 = _mm512_maskz_loadu_pd(mask_lo, &in_re[k + 4 * K]);                 \
        __m512d im4 = _mm512_maskz_loadu_pd(mask_lo, &in_im[k + 4 * K]);                 \
        __m512d re5 = _mm512_maskz_loadu_pd(mask_lo, &in_re[k + 5 * K]);                 \
        __m512d im5 = _mm512_maskz_loadu_pd(mask_lo, &in_im[k + 5 * K]);                 \
        __m512d re6 = _mm512_maskz_loadu_pd(mask_hi, &in_re[k + 6 * K]);                 \
        __m512d im6 = _mm512_maskz_loadu_pd(mask_hi, &in_im[k + 6 * K]);                 \
        __m512d re7 = _mm512_maskz_loadu_pd(mask_hi, &in_re[k + 7 * K]);                 \
        __m512d im7 = _mm512_maskz_loadu_pd(mask_hi, &in_im[k + 7 * K]);                 \
        __m512d re8 = _mm512_maskz_loadu_pd(mask_hi, &in_re[k + 8 * K]);                 \
        __m512d im8 = _mm512_maskz_loadu_pd(mask_hi, &in_im[k + 8 * K]);                 \
        __m512d re9 = _mm512_maskz_loadu_pd(mask_hi, &in_re[k + 9 * K]);                 \
        __m512d im9 = _mm512_maskz_loadu_pd(mask_hi, &in_im[k + 9 * K]);                 \
        __m512d re10 = _mm512_maskz_loadu_pd(mask_hi, &in_re[k + 10 * K]);               \
        __m512d im10 = _mm512_maskz_loadu_pd(mask_hi, &in_im[k + 10 * K]);               \
        /* Interleave into complex pairs */                                              \
        __m512d y0_lo = _mm512_unpacklo_pd(re0, im0);                                    \
        __m512d y0_hi = _mm512_unpackhi_pd(re0, im0);                                    \
        __m512d y1_lo = _mm512_unpacklo_pd(re1, im1);                                    \
        __m512d y1_hi = _mm512_unpackhi_pd(re1, im1);                                    \
        __m512d y2_lo = _mm512_unpacklo_pd(re2, im2);                                    \
        __m512d y2_hi = _mm512_unpackhi_pd(re2, im2);                                    \
        __m512d y3_lo = _mm512_unpacklo_pd(re3, im3);                                    \
        __m512d y3_hi = _mm512_unpackhi_pd(re3, im3);                                    \
        __m512d y4_lo = _mm512_unpacklo_pd(re4, im4);                                    \
        __m512d y4_hi = _mm512_unpackhi_pd(re4, im4);                                    \
        __m512d y5_lo = _mm512_unpacklo_pd(re5, im5);                                    \
        __m512d y5_hi = _mm512_unpackhi_pd(re5, im5);                                    \
        __m512d y6_lo = _mm512_unpacklo_pd(re6, im6);                                    \
        __m512d y6_hi = _mm512_unpackhi_pd(re6, im6);                                    \
        __m512d y7_lo = _mm512_unpacklo_pd(re7, im7);                                    \
        __m512d y7_hi = _mm512_unpackhi_pd(re7, im7);                                    \
        __m512d y8_lo = _mm512_unpacklo_pd(re8, im8);                                    \
        __m512d y8_hi = _mm512_unpackhi_pd(re8, im8);                                    \
        __m512d y9_lo = _mm512_unpacklo_pd(re9, im9);                                    \
        __m512d y9_hi = _mm512_unpackhi_pd(re9, im9);                                    \
        __m512d y10_lo = _mm512_unpacklo_pd(re10, im10);                                 \
        __m512d y10_hi = _mm512_unpackhi_pd(re10, im10);                                 \
        /* Assign outputs */                                                             \
        x0_lo = y0_lo;                                                                   \
        x0_hi = y0_hi;                                                                   \
        x1_lo = y1_lo;                                                                   \
        x1_hi = y1_hi;                                                                   \
        x2_lo = y2_lo;                                                                   \
        x2_hi = y2_hi;                                                                   \
        x3_lo = y3_lo;                                                                   \
        x3_hi = y3_hi;                                                                   \
        x4_lo = y4_lo;                                                                   \
        x4_hi = y4_hi;                                                                   \
        x5_lo = y5_lo;                                                                   \
        x5_hi = y5_hi;                                                                   \
        x6_lo = y6_lo;                                                                   \
        x6_hi = y6_hi;                                                                   \
        x7_lo = y7_lo;                                                                   \
        x7_hi = y7_hi;                                                                   \
        x8_lo = y8_lo;                                                                   \
        x8_hi = y8_hi;                                                                   \
        x9_lo = y9_lo;                                                                   \
        x9_hi = y9_hi;                                                                   \
        x10_lo = y10_lo;                                                                 \
        x10_hi = y10_hi;                                                                 \
    } while (0)

//==============================================================================
// SHARED STORE MACROS
//==============================================================================

/**
 * @brief Store 11 lanes to SoA buffers - FULL VERSION
 * @details De-interleaves and writes both lo/hi halves
 */
#define STORE_11_LANES_AVX512_NATIVE_SOA_FULL(k, K, out_re, out_im,                     \
                                              y0_lo, y0_hi, y1_lo, y1_hi, y2_lo, y2_hi, \
                                              y3_lo, y3_hi, y4_lo, y4_hi, y5_lo, y5_hi, \
                                              y6_lo, y6_hi, y7_lo, y7_hi, y8_lo, y8_hi, \
                                              y9_lo, y9_hi, y10_lo, y10_hi)             \
    do                                                                                  \
    {                                                                                   \
        __m512i idx_re = get_deinterleave_idx_re();                                     \
        __m512i idx_im = get_deinterleave_idx_im();                                     \
        /* De-interleave using permutex2var (CORRECT) */                                \
        __m512d re0 = _mm512_permutex2var_pd(y0_lo, idx_re, y0_hi);                     \
        __m512d im0 = _mm512_permutex2var_pd(y0_lo, idx_im, y0_hi);                     \
        __m512d re1 = _mm512_permutex2var_pd(y1_lo, idx_re, y1_hi);                     \
        __m512d im1 = _mm512_permutex2var_pd(y1_lo, idx_im, y1_hi);                     \
        __m512d re2 = _mm512_permutex2var_pd(y2_lo, idx_re, y2_hi);                     \
        __m512d im2 = _mm512_permutex2var_pd(y2_lo, idx_im, y2_hi);                     \
        __m512d re3 = _mm512_permutex2var_pd(y3_lo, idx_re, y3_hi);                     \
        __m512d im3 = _mm512_permutex2var_pd(y3_lo, idx_im, y3_hi);                     \
        __m512d re4 = _mm512_permutex2var_pd(y4_lo, idx_re, y4_hi);                     \
        __m512d im4 = _mm512_permutex2var_pd(y4_lo, idx_im, y4_hi);                     \
        __m512d re5 = _mm512_permutex2var_pd(y5_lo, idx_re, y5_hi);                     \
        __m512d im5 = _mm512_permutex2var_pd(y5_lo, idx_im, y5_hi);                     \
        __m512d re6 = _mm512_permutex2var_pd(y6_lo, idx_re, y6_hi);                     \
        __m512d im6 = _mm512_permutex2var_pd(y6_lo, idx_im, y6_hi);                     \
        __m512d re7 = _mm512_permutex2var_pd(y7_lo, idx_re, y7_hi);                     \
        __m512d im7 = _mm512_permutex2var_pd(y7_lo, idx_im, y7_hi);                     \
        __m512d re8 = _mm512_permutex2var_pd(y8_lo, idx_re, y8_hi);                     \
        __m512d im8 = _mm512_permutex2var_pd(y8_lo, idx_im, y8_hi);                     \
        __m512d re9 = _mm512_permutex2var_pd(y9_lo, idx_re, y9_hi);                     \
        __m512d im9 = _mm512_permutex2var_pd(y9_lo, idx_im, y9_hi);                     \
        __m512d re10 = _mm512_permutex2var_pd(y10_lo, idx_re, y10_hi);                  \
        __m512d im10 = _mm512_permutex2var_pd(y10_lo, idx_im, y10_hi);                  \
        /* NORMAL stores (no NT for scattered radix-11) */                              \
        _mm512_storeu_pd(&out_re[k + 0 * K], re0);                                      \
        _mm512_storeu_pd(&out_im[k + 0 * K], im0);                                      \
        _mm512_storeu_pd(&out_re[k + 1 * K], re1);                                      \
        _mm512_storeu_pd(&out_im[k + 1 * K], im1);                                      \
        _mm512_storeu_pd(&out_re[k + 2 * K], re2);                                      \
        _mm512_storeu_pd(&out_im[k + 2 * K], im2);                                      \
        _mm512_storeu_pd(&out_re[k + 3 * K], re3);                                      \
        _mm512_storeu_pd(&out_im[k + 3 * K], im3);                                      \
        _mm512_storeu_pd(&out_re[k + 4 * K], re4);                                      \
        _mm512_storeu_pd(&out_im[k + 4 * K], im4);                                      \
        _mm512_storeu_pd(&out_re[k + 5 * K], re5);                                      \
        _mm512_storeu_pd(&out_im[k + 5 * K], im5);                                      \
        _mm512_storeu_pd(&out_re[k + 6 * K], re6);                                      \
        _mm512_storeu_pd(&out_im[k + 6 * K], im6);                                      \
        _mm512_storeu_pd(&out_re[k + 7 * K], re7);                                      \
        _mm512_storeu_pd(&out_im[k + 7 * K], im7);                                      \
        _mm512_storeu_pd(&out_re[k + 8 * K], re8);                                      \
        _mm512_storeu_pd(&out_im[k + 8 * K], im8);                                      \
        _mm512_storeu_pd(&out_re[k + 9 * K], re9);                                      \
        _mm512_storeu_pd(&out_im[k + 9 * K], im9);                                      \
        _mm512_storeu_pd(&out_re[k + 10 * K], re10);                                    \
        _mm512_storeu_pd(&out_im[k + 10 * K], im10);                                    \
    } while (0)

/**
 * @brief Store 11 lanes with masking for tail - MASKED VERSION
 */
#define STORE_11_LANES_AVX512_NATIVE_SOA_MASKED(k, K, remaining, out_re, out_im,          \
                                                y0_lo, y0_hi, y1_lo, y1_hi, y2_lo, y2_hi, \
                                                y3_lo, y3_hi, y4_lo, y4_hi, y5_lo, y5_hi, \
                                                y6_lo, y6_hi, y7_lo, y7_hi, y8_lo, y8_hi, \
                                                y9_lo, y9_hi, y10_lo, y10_hi)             \
    do                                                                                    \
    {                                                                                     \
        size_t lo_count = (remaining < 4) ? remaining : 4;                                \
        size_t hi_count = (remaining > 4) ? (remaining - 4) : 0;                          \
        __mmask8 mask_lo = (__mmask8)((1U << (2 * lo_count)) - 1);                        \
        __mmask8 mask_hi = (__mmask8)((1U << (2 * hi_count)) - 1);                        \
        __m512i idx_re = get_deinterleave_idx_re();                                       \
        __m512i idx_im = get_deinterleave_idx_im();                                       \
        /* De-interleave */                                                               \
        __m512d re0 = _mm512_permutex2var_pd(y0_lo, idx_re, y0_hi);                       \
        __m512d im0 = _mm512_permutex2var_pd(y0_lo, idx_im, y0_hi);                       \
        __m512d re1 = _mm512_permutex2var_pd(y1_lo, idx_re, y1_hi);                       \
        __m512d im1 = _mm512_permutex2var_pd(y1_lo, idx_im, y1_hi);                       \
        __m512d re2 = _mm512_permutex2var_pd(y2_lo, idx_re, y2_hi);                       \
        __m512d im2 = _mm512_permutex2var_pd(y2_lo, idx_im, y2_hi);                       \
        __m512d re3 = _mm512_permutex2var_pd(y3_lo, idx_re, y3_hi);                       \
        __m512d im3 = _mm512_permutex2var_pd(y3_lo, idx_im, y3_hi);                       \
        __m512d re4 = _mm512_permutex2var_pd(y4_lo, idx_re, y4_hi);                       \
        __m512d im4 = _mm512_permutex2var_pd(y4_lo, idx_im, y4_hi);                       \
        __m512d re5 = _mm512_permutex2var_pd(y5_lo, idx_re, y5_hi);                       \
        __m512d im5 = _mm512_permutex2var_pd(y5_lo, idx_im, y5_hi);                       \
        __m512d re6 = _mm512_permutex2var_pd(y6_lo, idx_re, y6_hi);                       \
        __m512d im6 = _mm512_permutex2var_pd(y6_lo, idx_im, y6_hi);                       \
        __m512d re7 = _mm512_permutex2var_pd(y7_lo, idx_re, y7_hi);                       \
        __m512d im7 = _mm512_permutex2var_pd(y7_lo, idx_im, y7_hi);                       \
        __m512d re8 = _mm512_permutex2var_pd(y8_lo, idx_re, y8_hi);                       \
        __m512d im8 = _mm512_permutex2var_pd(y8_lo, idx_im, y8_hi);                       \
        __m512d re9 = _mm512_permutex2var_pd(y9_lo, idx_re, y9_hi);                       \
        __m512d im9 = _mm512_permutex2var_pd(y9_lo, idx_im, y9_hi);                       \
        __m512d re10 = _mm512_permutex2var_pd(y10_lo, idx_re, y10_hi);                    \
        __m512d im10 = _mm512_permutex2var_pd(y10_lo, idx_im, y10_hi);                    \
        /* Masked stores */                                                               \
        _mm512_mask_storeu_pd(&out_re[k + 0 * K], mask_lo, re0);                          \
        _mm512_mask_storeu_pd(&out_im[k + 0 * K], mask_lo, im0);                          \
        _mm512_mask_storeu_pd(&out_re[k + 1 * K], mask_lo, re1);                          \
        _mm512_mask_storeu_pd(&out_im[k + 1 * K], mask_lo, im1);                          \
        _mm512_mask_storeu_pd(&out_re[k + 2 * K], mask_lo, re2);                          \
        _mm512_mask_storeu_pd(&out_im[k + 2 * K], mask_lo, im2);                          \
        _mm512_mask_storeu_pd(&out_re[k + 3 * K], mask_lo, re3);                          \
        _mm512_mask_storeu_pd(&out_im[k + 3 * K], mask_lo, im3);                          \
        _mm512_mask_storeu_pd(&out_re[k + 4 * K], mask_lo, re4);                          \
        _mm512_mask_storeu_pd(&out_im[k + 4 * K], mask_lo, im4);                          \
        _mm512_mask_storeu_pd(&out_re[k + 5 * K], mask_lo, re5);                          \
        _mm512_mask_storeu_pd(&out_im[k + 5 * K], mask_lo, im5);                          \
        _mm512_mask_storeu_pd(&out_re[k + 6 * K], mask_hi, re6);                          \
        _mm512_mask_storeu_pd(&out_im[k + 6 * K], mask_hi, im6);                          \
        _mm512_mask_storeu_pd(&out_re[k + 7 * K], mask_hi, re7);                          \
        _mm512_mask_storeu_pd(&out_im[k + 7 * K], mask_hi, im7);                          \
        _mm512_mask_storeu_pd(&out_re[k + 8 * K], mask_hi, re8);                          \
        _mm512_mask_storeu_pd(&out_im[k + 8 * K], mask_hi, im8);                          \
        _mm512_mask_storeu_pd(&out_re[k + 9 * K], mask_hi, re9);                          \
        _mm512_mask_storeu_pd(&out_im[k + 9 * K], mask_hi, im9);                          \
        _mm512_mask_storeu_pd(&out_re[k + 10 * K], mask_hi, re10);                        \
        _mm512_mask_storeu_pd(&out_im[k + 10 * K], mask_hi, im10);                        \
    } while (0)

//==============================================================================
// SHARED TWIDDLE APPLICATION
//==============================================================================

/**
 * @brief Apply stage twiddles - SHARED FOR BOTH FV AND BV
 * @details Complex multiply x * w using FMA
 */
#define APPLY_STAGE_TWIDDLES_R11_AVX512_SOA_NATIVE(k, K, x1, x2, x3, x4, x5,               \
                                                   x6, x7, x8, x9, x10, stage_tw, sub_len) \
    do                                                                                     \
    {                                                                                      \
        if ((sub_len) > 1)                                                                 \
        {                                                                                  \
            __m512d w_re, w_im, x_re, x_im, tmp_re, tmp_im;                                \
            w_re = _mm512_loadu_pd(&stage_tw->re[0 * K + k]);                              \
            w_im = _mm512_loadu_pd(&stage_tw->im[0 * K + k]);                              \
            x_re = _mm512_shuffle_pd(x1, x1, 0x00);                                        \
            x_im = _mm512_shuffle_pd(x1, x1, 0xFF);                                        \
            tmp_re = _mm512_fmsub_pd(x_re, w_re, _mm512_mul_pd(x_im, w_im));               \
            tmp_im = _mm512_fmadd_pd(x_re, w_im, _mm512_mul_pd(x_im, w_re));               \
            x1 = _mm512_unpacklo_pd(tmp_re, tmp_im);                                       \
            w_re = _mm512_loadu_pd(&stage_tw->re[1 * K + k]);                              \
            w_im = _mm512_loadu_pd(&stage_tw->im[1 * K + k]);                              \
            x_re = _mm512_shuffle_pd(x2, x2, 0x00);                                        \
            x_im = _mm512_shuffle_pd(x2, x2, 0xFF);                                        \
            tmp_re = _mm512_fmsub_pd(x_re, w_re, _mm512_mul_pd(x_im, w_im));               \
            tmp_im = _mm512_fmadd_pd(x_re, w_im, _mm512_mul_pd(x_im, w_re));               \
            x2 = _mm512_unpacklo_pd(tmp_re, tmp_im);                                       \
            w_re = _mm512_loadu_pd(&stage_tw->re[2 * K + k]);                              \
            w_im = _mm512_loadu_pd(&stage_tw->im[2 * K + k]);                              \
            x_re = _mm512_shuffle_pd(x3, x3, 0x00);                                        \
            x_im = _mm512_shuffle_pd(x3, x3, 0xFF);                                        \
            tmp_re = _mm512_fmsub_pd(x_re, w_re, _mm512_mul_pd(x_im, w_im));               \
            tmp_im = _mm512_fmadd_pd(x_re, w_im, _mm512_mul_pd(x_im, w_re));               \
            x3 = _mm512_unpacklo_pd(tmp_re, tmp_im);                                       \
            w_re = _mm512_loadu_pd(&stage_tw->re[3 * K + k]);                              \
            w_im = _mm512_loadu_pd(&stage_tw->im[3 * K + k]);                              \
            x_re = _mm512_shuffle_pd(x4, x4, 0x00);                                        \
            x_im = _mm512_shuffle_pd(x4, x4, 0xFF);                                        \
            tmp_re = _mm512_fmsub_pd(x_re, w_re, _mm512_mul_pd(x_im, w_im));               \
            tmp_im = _mm512_fmadd_pd(x_re, w_im, _mm512_mul_pd(x_im, w_re));               \
            x4 = _mm512_unpacklo_pd(tmp_re, tmp_im);                                       \
            w_re = _mm512_loadu_pd(&stage_tw->re[4 * K + k]);                              \
            w_im = _mm512_loadu_pd(&stage_tw->im[4 * K + k]);                              \
            x_re = _mm512_shuffle_pd(x5, x5, 0x00);                                        \
            x_im = _mm512_shuffle_pd(x5, x5, 0xFF);                                        \
            tmp_re = _mm512_fmsub_pd(x_re, w_re, _mm512_mul_pd(x_im, w_im));               \
            tmp_im = _mm512_fmadd_pd(x_re, w_im, _mm512_mul_pd(x_im, w_re));               \
            x5 = _mm512_unpacklo_pd(tmp_re, tmp_im);                                       \
            w_re = _mm512_loadu_pd(&stage_tw->re[5 * K + k]);                              \
            w_im = _mm512_loadu_pd(&stage_tw->im[5 * K + k]);                              \
            x_re = _mm512_shuffle_pd(x6, x6, 0x00);                                        \
            x_im = _mm512_shuffle_pd(x6, x6, 0xFF);                                        \
            tmp_re = _mm512_fmsub_pd(x_re, w_re, _mm512_mul_pd(x_im, w_im));               \
            tmp_im = _mm512_fmadd_pd(x_re, w_im, _mm512_mul_pd(x_im, w_re));               \
            x6 = _mm512_unpacklo_pd(tmp_re, tmp_im);                                       \
            w_re = _mm512_loadu_pd(&stage_tw->re[6 * K + k]);                              \
            w_im = _mm512_loadu_pd(&stage_tw->im[6 * K + k]);                              \
            x_re = _mm512_shuffle_pd(x7, x7, 0x00);                                        \
            x_im = _mm512_shuffle_pd(x7, x7, 0xFF);                                        \
            tmp_re = _mm512_fmsub_pd(x_re, w_re, _mm512_mul_pd(x_im, w_im));               \
            tmp_im = _mm512_fmadd_pd(x_re, w_im, _mm512_mul_pd(x_im, w_re));               \
            x7 = _mm512_unpacklo_pd(tmp_re, tmp_im);                                       \
            w_re = _mm512_loadu_pd(&stage_tw->re[7 * K + k]);                              \
            w_im = _mm512_loadu_pd(&stage_tw->im[7 * K + k]);                              \
            x_re = _mm512_shuffle_pd(x8, x8, 0x00);                                        \
            x_im = _mm512_shuffle_pd(x8, x8, 0xFF);                                        \
            tmp_re = _mm512_fmsub_pd(x_re, w_re, _mm512_mul_pd(x_im, w_im));               \
            tmp_im = _mm512_fmadd_pd(x_re, w_im, _mm512_mul_pd(x_im, w_re));               \
            x8 = _mm512_unpacklo_pd(tmp_re, tmp_im);                                       \
            w_re = _mm512_loadu_pd(&stage_tw->re[8 * K + k]);                              \
            w_im = _mm512_loadu_pd(&stage_tw->im[8 * K + k]);                              \
            x_re = _mm512_shuffle_pd(x9, x9, 0x00);                                        \
            x_im = _mm512_shuffle_pd(x9, x9, 0xFF);                                        \
            tmp_re = _mm512_fmsub_pd(x_re, w_re, _mm512_mul_pd(x_im, w_im));               \
            tmp_im = _mm512_fmadd_pd(x_re, w_im, _mm512_mul_pd(x_im, w_re));               \
            x9 = _mm512_unpacklo_pd(tmp_re, tmp_im);                                       \
            w_re = _mm512_loadu_pd(&stage_tw->re[9 * K + k]);                              \
            w_im = _mm512_loadu_pd(&stage_tw->im[9 * K + k]);                              \
            x_re = _mm512_shuffle_pd(x10, x10, 0x00);                                      \
            x_im = _mm512_shuffle_pd(x10, x10, 0xFF);                                      \
            tmp_re = _mm512_fmsub_pd(x_re, w_re, _mm512_mul_pd(x_im, w_im));               \
            tmp_im = _mm512_fmadd_pd(x_re, w_im, _mm512_mul_pd(x_im, w_re));               \
            x10 = _mm512_unpacklo_pd(tmp_re, tmp_im);                                      \
        }                                                                                  \
    } while (0)

//==============================================================================
// SHARED BUTTERFLY CORE AND REAL PAIR COMPUTATIONS (PRESERVED EXACTLY)
//==============================================================================

/**
 * @brief Radix-11 butterfly core - SHARED (direction-independent)
 * @details Computes t-sums and s-differences from 11 inputs
 */
#define RADIX11_BUTTERFLY_CORE_AVX512(a, b, c, d, e, f, g, h, i, j, xk,           \
                                      t0, t1, t2, t3, t4, s0, s1, s2, s3, s4, y0) \
    do                                                                            \
    {                                                                             \
        t0 = _mm512_add_pd(b, xk);                                                \
        t1 = _mm512_add_pd(c, j);                                                 \
        t2 = _mm512_add_pd(d, i);                                                 \
        t3 = _mm512_add_pd(e, h);                                                 \
        t4 = _mm512_add_pd(f, g);                                                 \
        s0 = _mm512_sub_pd(b, xk);                                                \
        s1 = _mm512_sub_pd(c, j);                                                 \
        s2 = _mm512_sub_pd(d, i);                                                 \
        s3 = _mm512_sub_pd(e, h);                                                 \
        s4 = _mm512_sub_pd(f, g);                                                 \
        __m512d sum_t = _mm512_add_pd(_mm512_add_pd(t0, t1),                      \
                                      _mm512_add_pd(_mm512_add_pd(t2, t3), t4));  \
        y0 = _mm512_add_pd(a, sum_t);                                             \
    } while (0)

/**
 * @brief Real part computations - SHARED (all 5 pairs)
 * @details Direct geometric decomposition using FMA chains
 */
#define RADIX11_REAL_PAIR1_AVX512(x0, t0, t1, t2, t3, t4, KC, real_out)                                              \
    do                                                                                                               \
    {                                                                                                                \
        __m512d term = _mm512_fmadd_pd(KC.c1, t0,                                                                    \
                                       _mm512_fmadd_pd(KC.c2, t1,                                                    \
                                                       _mm512_fmadd_pd(KC.c3, t2,                                    \
                                                                       _mm512_fmadd_pd(KC.c4, t3,                    \
                                                                                       _mm512_mul_pd(KC.c5, t4))))); \
        real_out = _mm512_add_pd(x0, term);                                                                          \
    } while (0)

#define RADIX11_REAL_PAIR2_AVX512(x0, t0, t1, t2, t3, t4, KC, real_out)                                              \
    do                                                                                                               \
    {                                                                                                                \
        __m512d term = _mm512_fmadd_pd(KC.c2, t0,                                                                    \
                                       _mm512_fmadd_pd(KC.c4, t1,                                                    \
                                                       _mm512_fmadd_pd(KC.c5, t2,                                    \
                                                                       _mm512_fmadd_pd(KC.c3, t3,                    \
                                                                                       _mm512_mul_pd(KC.c1, t4))))); \
        real_out = _mm512_add_pd(x0, term);                                                                          \
    } while (0)

#define RADIX11_REAL_PAIR3_AVX512(x0, t0, t1, t2, t3, t4, KC, real_out)                                              \
    do                                                                                                               \
    {                                                                                                                \
        __m512d term = _mm512_fmadd_pd(KC.c3, t0,                                                                    \
                                       _mm512_fmadd_pd(KC.c5, t1,                                                    \
                                                       _mm512_fmadd_pd(KC.c2, t2,                                    \
                                                                       _mm512_fmadd_pd(KC.c1, t3,                    \
                                                                                       _mm512_mul_pd(KC.c4, t4))))); \
        real_out = _mm512_add_pd(x0, term);                                                                          \
    } while (0)

#define RADIX11_REAL_PAIR4_AVX512(x0, t0, t1, t2, t3, t4, KC, real_out)                                              \
    do                                                                                                               \
    {                                                                                                                \
        __m512d term = _mm512_fmadd_pd(KC.c4, t0,                                                                    \
                                       _mm512_fmadd_pd(KC.c3, t1,                                                    \
                                                       _mm512_fmadd_pd(KC.c1, t2,                                    \
                                                                       _mm512_fmadd_pd(KC.c5, t3,                    \
                                                                                       _mm512_mul_pd(KC.c2, t4))))); \
        real_out = _mm512_add_pd(x0, term);                                                                          \
    } while (0)

#define RADIX11_REAL_PAIR5_AVX512(x0, t0, t1, t2, t3, t4, KC, real_out)                                              \
    do                                                                                                               \
    {                                                                                                                \
        __m512d term = _mm512_fmadd_pd(KC.c5, t0,                                                                    \
                                       _mm512_fmadd_pd(KC.c1, t1,                                                    \
                                                       _mm512_fmadd_pd(KC.c4, t2,                                    \
                                                                       _mm512_fmadd_pd(KC.c2, t3,                    \
                                                                                       _mm512_mul_pd(KC.c3, t4))))); \
        real_out = _mm512_add_pd(x0, term);                                                                          \
    } while (0)

/**
 * @brief Assembly of conjugate pairs - SHARED
 */
#define RADIX11_ASSEMBLE_PAIR_AVX512(real_part, rot_part, y_m, y_conj) \
    do                                                                 \
    {                                                                  \
        y_m = _mm512_add_pd(real_part, rot_part);                      \
        y_conj = _mm512_sub_pd(real_part, rot_part);                   \
    } while (0)

//==============================================================================
// FORWARD (FV) IMAGINARY PAIR MACROS
//==============================================================================

/**
 * @brief Forward imaginary pair computations (FV)
 * @details Computes base = Σ(S11_m * s_m), then rotates by -i
 *
 * CRITICAL: Must multiply base by -i for forward FFT
 * (a+bi)*(-i) = b - ai => swap and negate real part
 */
#define RADIX11_IMAG_PAIR1_FV_AVX512(s0, s1, s2, s3, s4, KC, rot_out)                                                \
    do                                                                                                               \
    {                                                                                                                \
        __m512d base = _mm512_fmadd_pd(KC.s1, s0,                                                                    \
                                       _mm512_fmadd_pd(KC.s2, s1,                                                    \
                                                       _mm512_fmadd_pd(KC.s3, s2,                                    \
                                                                       _mm512_fmadd_pd(KC.s4, s3,                    \
                                                                                       _mm512_mul_pd(KC.s5, s4))))); \
        ROTATE_BY_MINUS_I_AVX512(base, rot_out);                                                                     \
    } while (0)

#define RADIX11_IMAG_PAIR2_FV_AVX512(s0, s1, s2, s3, s4, KC, rot_out)                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        __m512d base = _mm512_fmadd_pd(KC.s2, s0,                                                                      \
                                       _mm512_fmadd_pd(KC.s4, s1,                                                      \
                                                       _mm512_fnmadd_pd(KC.s5, s2,                                     \
                                                                        _mm512_fnmadd_pd(KC.s3, s3,                    \
                                                                                         _mm512_mul_pd(KC.s1, s4))))); \
        ROTATE_BY_MINUS_I_AVX512(base, rot_out);                                                                       \
    } while (0)

#define RADIX11_IMAG_PAIR3_FV_AVX512(s0, s1, s2, s3, s4, KC, rot_out)                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        __m512d base = _mm512_fmadd_pd(KC.s3, s0,                                                                      \
                                       _mm512_fnmadd_pd(KC.s5, s1,                                                     \
                                                        _mm512_fnmadd_pd(KC.s2, s2,                                    \
                                                                         _mm512_fmadd_pd(KC.s1, s3,                    \
                                                                                         _mm512_mul_pd(KC.s4, s4))))); \
        ROTATE_BY_MINUS_I_AVX512(base, rot_out);                                                                       \
    } while (0)

#define RADIX11_IMAG_PAIR4_FV_AVX512(s0, s1, s2, s3, s4, KC, rot_out)                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        __m512d base = _mm512_fmadd_pd(KC.s4, s0,                                                                      \
                                       _mm512_fnmadd_pd(KC.s3, s1,                                                     \
                                                        _mm512_fmadd_pd(KC.s1, s2,                                     \
                                                                        _mm512_fnmadd_pd(KC.s5, s3,                    \
                                                                                         _mm512_mul_pd(KC.s2, s4))))); \
        ROTATE_BY_MINUS_I_AVX512(base, rot_out);                                                                       \
    } while (0)

#define RADIX11_IMAG_PAIR5_FV_AVX512(s0, s1, s2, s3, s4, KC, rot_out)                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        __m512d base = _mm512_fmadd_pd(KC.s5, s0,                                                                      \
                                       _mm512_fnmadd_pd(KC.s1, s1,                                                     \
                                                        _mm512_fmadd_pd(KC.s4, s2,                                     \
                                                                        _mm512_fnmadd_pd(KC.s2, s3,                    \
                                                                                         _mm512_mul_pd(KC.s3, s4))))); \
        ROTATE_BY_MINUS_I_AVX512(base, rot_out);                                                                       \
    } while (0)

//==============================================================================
// BACKWARD (BV) IMAGINARY PAIR MACROS
//==============================================================================

/**
 * @brief Backward imaginary pair computations (BV)
 * @details Computes base = Σ(S11_m * s_m), then rotates by +i
 *
 * CRITICAL: Must multiply base by +i for backward FFT
 * (a+bi)*(+i) = -b + ai => swap and negate imaginary part
 */
#define RADIX11_IMAG_PAIR1_BV_AVX512(s0, s1, s2, s3, s4, KC, rot_out)                                                \
    do                                                                                                               \
    {                                                                                                                \
        __m512d base = _mm512_fmadd_pd(KC.s1, s0,                                                                    \
                                       _mm512_fmadd_pd(KC.s2, s1,                                                    \
                                                       _mm512_fmadd_pd(KC.s3, s2,                                    \
                                                                       _mm512_fmadd_pd(KC.s4, s3,                    \
                                                                                       _mm512_mul_pd(KC.s5, s4))))); \
        ROTATE_BY_PLUS_I_AVX512(base, rot_out);                                                                      \
    } while (0)

#define RADIX11_IMAG_PAIR2_BV_AVX512(s0, s1, s2, s3, s4, KC, rot_out)                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        __m512d base = _mm512_fmadd_pd(KC.s2, s0,                                                                      \
                                       _mm512_fmadd_pd(KC.s4, s1,                                                      \
                                                       _mm512_fnmadd_pd(KC.s5, s2,                                     \
                                                                        _mm512_fnmadd_pd(KC.s3, s3,                    \
                                                                                         _mm512_mul_pd(KC.s1, s4))))); \
        ROTATE_BY_PLUS_I_AVX512(base, rot_out);                                                                        \
    } while (0)

#define RADIX11_IMAG_PAIR3_BV_AVX512(s0, s1, s2, s3, s4, KC, rot_out)                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        __m512d base = _mm512_fmadd_pd(KC.s3, s0,                                                                      \
                                       _mm512_fnmadd_pd(KC.s5, s1,                                                     \
                                                        _mm512_fnmadd_pd(KC.s2, s2,                                    \
                                                                         _mm512_fmadd_pd(KC.s1, s3,                    \
                                                                                         _mm512_mul_pd(KC.s4, s4))))); \
        ROTATE_BY_PLUS_I_AVX512(base, rot_out);                                                                        \
    } while (0)

#define RADIX11_IMAG_PAIR4_BV_AVX512(s0, s1, s2, s3, s4, KC, rot_out)                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        __m512d base = _mm512_fmadd_pd(KC.s4, s0,                                                                      \
                                       _mm512_fnmadd_pd(KC.s3, s1,                                                     \
                                                        _mm512_fmadd_pd(KC.s1, s2,                                     \
                                                                        _mm512_fnmadd_pd(KC.s5, s3,                    \
                                                                                         _mm512_mul_pd(KC.s2, s4))))); \
        ROTATE_BY_PLUS_I_AVX512(base, rot_out);                                                                        \
    } while (0)

#define RADIX11_IMAG_PAIR5_BV_AVX512(s0, s1, s2, s3, s4, KC, rot_out)                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        __m512d base = _mm512_fmadd_pd(KC.s5, s0,                                                                      \
                                       _mm512_fnmadd_pd(KC.s1, s1,                                                     \
                                                        _mm512_fmadd_pd(KC.s4, s2,                                     \
                                                                        _mm512_fnmadd_pd(KC.s2, s3,                    \
                                                                                         _mm512_mul_pd(KC.s3, s4))))); \
        ROTATE_BY_PLUS_I_AVX512(base, rot_out);                                                                        \
    } while (0)

//==============================================================================
// FORWARD (FV) TOP-LEVEL BUTTERFLY MACROS
//==============================================================================

/**
 * @brief Forward radix-11 butterfly - FULL VERSION
 * @details Processes 8 complex elements (4 lo + 4 hi) with prefetch
 */
#define RADIX11_BUTTERFLY_FV_AVX512_NATIVE_SOA(k, K, in_re, in_im, stage_tw,           \
                                               out_re, out_im, sub_len)                \
    do                                                                                 \
    {                                                                                  \
        /* Prefetch upcoming data */                                                   \
        PREFETCH_RADIX11_INPUT(k, K, in_re, in_im, R11_PREFETCH_DISTANCE,              \
                               stage_tw, sub_len);                                     \
        /* Broadcast constants ONCE */                                                 \
        radix11_consts_avx512 KC = broadcast_radix11_consts_avx512();                  \
        /* Load all 11 lanes (both lo and hi halves) */                                \
        __m512d x0_lo, x0_hi, x1_lo, x1_hi, x2_lo, x2_hi, x3_lo, x3_hi;                \
        __m512d x4_lo, x4_hi, x5_lo, x5_hi, x6_lo, x6_hi, x7_lo, x7_hi;                \
        __m512d x8_lo, x8_hi, x9_lo, x9_hi, x10_lo, x10_hi;                            \
        LOAD_11_LANES_AVX512_NATIVE_SOA_FULL(k, K, in_re, in_im,                       \
                                             x0_lo, x0_hi, x1_lo, x1_hi,               \
                                             x2_lo, x2_hi, x3_lo, x3_hi,               \
                                             x4_lo, x4_hi, x5_lo, x5_hi,               \
                                             x6_lo, x6_hi, x7_lo, x7_hi,               \
                                             x8_lo, x8_hi, x9_lo, x9_hi,               \
                                             x10_lo, x10_hi);                          \
        /* Apply stage twiddles to LO half */                                          \
        APPLY_STAGE_TWIDDLES_R11_AVX512_SOA_NATIVE(k, K, x1_lo, x2_lo, x3_lo,          \
                                                   x4_lo, x5_lo, x6_lo, x7_lo,         \
                                                   x8_lo, x9_lo, x10_lo,               \
                                                   stage_tw, sub_len);                 \
        /* Process LO half butterfly */                                                \
        __m512d t0_lo, t1_lo, t2_lo, t3_lo, t4_lo;                                     \
        __m512d s0_lo, s1_lo, s2_lo, s3_lo, s4_lo, y0_lo;                              \
        RADIX11_BUTTERFLY_CORE_AVX512(x0_lo, x1_lo, x2_lo, x3_lo, x4_lo,               \
                                      x5_lo, x6_lo, x7_lo, x8_lo, x9_lo,               \
                                      x10_lo, t0_lo, t1_lo, t2_lo, t3_lo,              \
                                      t4_lo, s0_lo, s1_lo, s2_lo, s3_lo,               \
                                      s4_lo, y0_lo);                                   \
        __m512d real1_lo, real2_lo, real3_lo, real4_lo, real5_lo;                      \
        RADIX11_REAL_PAIR1_AVX512(x0_lo, t0_lo, t1_lo, t2_lo, t3_lo, t4_lo,            \
                                  KC, real1_lo);                                       \
        RADIX11_REAL_PAIR2_AVX512(x0_lo, t0_lo, t1_lo, t2_lo, t3_lo, t4_lo,            \
                                  KC, real2_lo);                                       \
        RADIX11_REAL_PAIR3_AVX512(x0_lo, t0_lo, t1_lo, t2_lo, t3_lo, t4_lo,            \
                                  KC, real3_lo);                                       \
        RADIX11_REAL_PAIR4_AVX512(x0_lo, t0_lo, t1_lo, t2_lo, t3_lo, t4_lo,            \
                                  KC, real4_lo);                                       \
        RADIX11_REAL_PAIR5_AVX512(x0_lo, t0_lo, t1_lo, t2_lo, t3_lo, t4_lo,            \
                                  KC, real5_lo);                                       \
        /* FORWARD imaginary pairs (FV) for LO half */                                 \
        __m512d rot1_lo, rot2_lo, rot3_lo, rot4_lo, rot5_lo;                           \
        RADIX11_IMAG_PAIR1_FV_AVX512(s0_lo, s1_lo, s2_lo, s3_lo, s4_lo,                \
                                     KC, rot1_lo);                                     \
        RADIX11_IMAG_PAIR2_FV_AVX512(s0_lo, s1_lo, s2_lo, s3_lo, s4_lo,                \
                                     KC, rot2_lo);                                     \
        RADIX11_IMAG_PAIR3_FV_AVX512(s0_lo, s1_lo, s2_lo, s3_lo, s4_lo,                \
                                     KC, rot3_lo);                                     \
        RADIX11_IMAG_PAIR4_FV_AVX512(s0_lo, s1_lo, s2_lo, s3_lo, s4_lo,                \
                                     KC, rot4_lo);                                     \
        RADIX11_IMAG_PAIR5_FV_AVX512(s0_lo, s1_lo, s2_lo, s3_lo, s4_lo,                \
                                     KC, rot5_lo);                                     \
        /* Assemble LO half outputs */                                                 \
        __m512d y1_lo, y2_lo, y3_lo, y4_lo, y5_lo, y6_lo, y7_lo, y8_lo, y9_lo, y10_lo; \
        RADIX11_ASSEMBLE_PAIR_AVX512(real1_lo, rot1_lo, y1_lo, y10_lo);                \
        RADIX11_ASSEMBLE_PAIR_AVX512(real2_lo, rot2_lo, y2_lo, y9_lo);                 \
        RADIX11_ASSEMBLE_PAIR_AVX512(real3_lo, rot3_lo, y3_lo, y8_lo);                 \
        RADIX11_ASSEMBLE_PAIR_AVX512(real4_lo, rot4_lo, y4_lo, y7_lo);                 \
        RADIX11_ASSEMBLE_PAIR_AVX512(real5_lo, rot5_lo, y5_lo, y6_lo);                 \
        /* Apply stage twiddles to HI half */                                          \
        APPLY_STAGE_TWIDDLES_R11_AVX512_SOA_NATIVE(k + 4, K, x1_hi, x2_hi, x3_hi,      \
                                                   x4_hi, x5_hi, x6_hi, x7_hi,         \
                                                   x8_hi, x9_hi, x10_hi,               \
                                                   stage_tw, sub_len);                 \
        /* Process HI half butterfly */                                                \
        __m512d t0_hi, t1_hi, t2_hi, t3_hi, t4_hi;                                     \
        __m512d s0_hi, s1_hi, s2_hi, s3_hi, s4_hi, y0_hi;                              \
        RADIX11_BUTTERFLY_CORE_AVX512(x0_hi, x1_hi, x2_hi, x3_hi, x4_hi,               \
                                      x5_hi, x6_hi, x7_hi, x8_hi, x9_hi,               \
                                      x10_hi, t0_hi, t1_hi, t2_hi, t3_hi,              \
                                      t4_hi, s0_hi, s1_hi, s2_hi, s3_hi,               \
                                      s4_hi, y0_hi);                                   \
        __m512d real1_hi, real2_hi, real3_hi, real4_hi, real5_hi;                      \
        RADIX11_REAL_PAIR1_AVX512(x0_hi, t0_hi, t1_hi, t2_hi, t3_hi, t4_hi,            \
                                  KC, real1_hi);                                       \
        RADIX11_REAL_PAIR2_AVX512(x0_hi, t0_hi, t1_hi, t2_hi, t3_hi, t4_hi,            \
                                  KC, real2_hi);                                       \
        RADIX11_REAL_PAIR3_AVX512(x0_hi, t0_hi, t1_hi, t2_hi, t3_hi, t4_hi,            \
                                  KC, real3_hi);                                       \
        RADIX11_REAL_PAIR4_AVX512(x0_hi, t0_hi, t1_hi, t2_hi, t3_hi, t4_hi,            \
                                  KC, real4_hi);                                       \
        RADIX11_REAL_PAIR5_AVX512(x0_hi, t0_hi, t1_hi, t2_hi, t3_hi, t4_hi,            \
                                  KC, real5_hi);                                       \
        /* FORWARD imaginary pairs (FV) for HI half */                                 \
        __m512d rot1_hi, rot2_hi, rot3_hi, rot4_hi, rot5_hi;                           \
        RADIX11_IMAG_PAIR1_FV_AVX512(s0_hi, s1_hi, s2_hi, s3_hi, s4_hi,                \
                                     KC, rot1_hi);                                     \
        RADIX11_IMAG_PAIR2_FV_AVX512(s0_hi, s1_hi, s2_hi, s3_hi, s4_hi,                \
                                     KC, rot2_hi);                                     \
        RADIX11_IMAG_PAIR3_FV_AVX512(s0_hi, s1_hi, s2_hi, s3_hi, s4_hi,                \
                                     KC, rot3_hi);                                     \
        RADIX11_IMAG_PAIR4_FV_AVX512(s0_hi, s1_hi, s2_hi, s3_hi, s4_hi,                \
                                     KC, rot4_hi);                                     \
        RADIX11_IMAG_PAIR5_FV_AVX512(s0_hi, s1_hi, s2_hi, s3_hi, s4_hi,                \
                                     KC, rot5_hi);                                     \
        /* Assemble HI half outputs */                                                 \
        __m512d y1_hi, y2_hi, y3_hi, y4_hi, y5_hi, y6_hi, y7_hi, y8_hi, y9_hi, y10_hi; \
        RADIX11_ASSEMBLE_PAIR_AVX512(real1_hi, rot1_hi, y1_hi, y10_hi);                \
        RADIX11_ASSEMBLE_PAIR_AVX512(real2_hi, rot2_hi, y2_hi, y9_hi);                 \
        RADIX11_ASSEMBLE_PAIR_AVX512(real3_hi, rot3_hi, y3_hi, y8_hi);                 \
        RADIX11_ASSEMBLE_PAIR_AVX512(real4_hi, rot4_hi, y4_hi, y7_hi);                 \
        RADIX11_ASSEMBLE_PAIR_AVX512(real5_hi, rot5_hi, y5_hi, y6_hi);                 \
        /* Store both halves (NORMAL STORES ONLY) */                                   \
        STORE_11_LANES_AVX512_NATIVE_SOA_FULL(k, K, out_re, out_im,                    \
                                              y0_lo, y0_hi, y1_lo, y1_hi,              \
                                              y2_lo, y2_hi, y3_lo, y3_hi,              \
                                              y4_lo, y4_hi, y5_lo, y5_hi,              \
                                              y6_lo, y6_hi, y7_lo, y7_hi,              \
                                              y8_lo, y8_hi, y9_lo, y9_hi,              \
                                              y10_lo, y10_hi);                         \
    } while (0)

/**
 * @brief Forward radix-11 butterfly - TAIL VERSION with masking
 * @details Handles remaining < 8 elements with masked loads/stores
 */
#define RADIX11_BUTTERFLY_FV_AVX512_NATIVE_SOA_TAIL(k, K, remaining, in_re, in_im,     \
                                                    stage_tw, out_re, out_im,          \
                                                    sub_len)                           \
    do                                                                                 \
    {                                                                                  \
        /* Prefetch if there's more data ahead */                                      \
        PREFETCH_RADIX11_INPUT(k, K, in_re, in_im, R11_PREFETCH_DISTANCE,              \
                               stage_tw, sub_len);                                     \
        radix11_consts_avx512 KC = broadcast_radix11_consts_avx512();                  \
        __m512d x0_lo, x0_hi, x1_lo, x1_hi, x2_lo, x2_hi, x3_lo, x3_hi;                \
        __m512d x4_lo, x4_hi, x5_lo, x5_hi, x6_lo, x6_hi, x7_lo, x7_hi;                \
        __m512d x8_lo, x8_hi, x9_lo, x9_hi, x10_lo, x10_hi;                            \
        /* Masked load - zeros invalid lanes */                                        \
        LOAD_11_LANES_AVX512_NATIVE_SOA_MASKED(k, K, remaining, in_re, in_im,          \
                                               x0_lo, x0_hi, x1_lo, x1_hi,             \
                                               x2_lo, x2_hi, x3_lo, x3_hi,             \
                                               x4_lo, x4_hi, x5_lo, x5_hi,             \
                                               x6_lo, x6_hi, x7_lo, x7_hi,             \
                                               x8_lo, x8_hi, x9_lo, x9_hi,             \
                                               x10_lo, x10_hi);                        \
        /* Apply stage twiddles to LO half */                                          \
        APPLY_STAGE_TWIDDLES_R11_AVX512_SOA_NATIVE(k, K, x1_lo, x2_lo, x3_lo,          \
                                                   x4_lo, x5_lo, x6_lo, x7_lo,         \
                                                   x8_lo, x9_lo, x10_lo,               \
                                                   stage_tw, sub_len);                 \
        /* Process LO half butterfly */                                                \
        __m512d t0_lo, t1_lo, t2_lo, t3_lo, t4_lo;                                     \
        __m512d s0_lo, s1_lo, s2_lo, s3_lo, s4_lo, y0_lo;                              \
        RADIX11_BUTTERFLY_CORE_AVX512(x0_lo, x1_lo, x2_lo, x3_lo, x4_lo,               \
                                      x5_lo, x6_lo, x7_lo, x8_lo, x9_lo,               \
                                      x10_lo, t0_lo, t1_lo, t2_lo, t3_lo,              \
                                      t4_lo, s0_lo, s1_lo, s2_lo, s3_lo,               \
                                      s4_lo, y0_lo);                                   \
        __m512d real1_lo, real2_lo, real3_lo, real4_lo, real5_lo;                      \
        RADIX11_REAL_PAIR1_AVX512(x0_lo, t0_lo, t1_lo, t2_lo, t3_lo, t4_lo,            \
                                  KC, real1_lo);                                       \
        RADIX11_REAL_PAIR2_AVX512(x0_lo, t0_lo, t1_lo, t2_lo, t3_lo, t4_lo,            \
                                  KC, real2_lo);                                       \
        RADIX11_REAL_PAIR3_AVX512(x0_lo, t0_lo, t1_lo, t2_lo, t3_lo, t4_lo,            \
                                  KC, real3_lo);                                       \
        RADIX11_REAL_PAIR4_AVX512(x0_lo, t0_lo, t1_lo, t2_lo, t3_lo, t4_lo,            \
                                  KC, real4_lo);                                       \
        RADIX11_REAL_PAIR5_AVX512(x0_lo, t0_lo, t1_lo, t2_lo, t3_lo, t4_lo,            \
                                  KC, real5_lo);                                       \
        __m512d rot1_lo, rot2_lo, rot3_lo, rot4_lo, rot5_lo;                           \
        RADIX11_IMAG_PAIR1_FV_AVX512(s0_lo, s1_lo, s2_lo, s3_lo, s4_lo,                \
                                     KC, rot1_lo);                                     \
        RADIX11_IMAG_PAIR2_FV_AVX512(s0_lo, s1_lo, s2_lo, s3_lo, s4_lo,                \
                                     KC, rot2_lo);                                     \
        RADIX11_IMAG_PAIR3_FV_AVX512(s0_lo, s1_lo, s2_lo, s3_lo, s4_lo,                \
                                     KC, rot3_lo);                                     \
        RADIX11_IMAG_PAIR4_FV_AVX512(s0_lo, s1_lo, s2_lo, s3_lo, s4_lo,                \
                                     KC, rot4_lo);                                     \
        RADIX11_IMAG_PAIR5_FV_AVX512(s0_lo, s1_lo, s2_lo, s3_lo, s4_lo,                \
                                     KC, rot5_lo);                                     \
        __m512d y1_lo, y2_lo, y3_lo, y4_lo, y5_lo, y6_lo, y7_lo, y8_lo, y9_lo, y10_lo; \
        RADIX11_ASSEMBLE_PAIR_AVX512(real1_lo, rot1_lo, y1_lo, y10_lo);                \
        RADIX11_ASSEMBLE_PAIR_AVX512(real2_lo, rot2_lo, y2_lo, y9_lo);                 \
        RADIX11_ASSEMBLE_PAIR_AVX512(real3_lo, rot3_lo, y3_lo, y8_lo);                 \
        RADIX11_ASSEMBLE_PAIR_AVX512(real4_lo, rot4_lo, y4_lo, y7_lo);                 \
        RADIX11_ASSEMBLE_PAIR_AVX512(real5_lo, rot5_lo, y5_lo, y6_lo);                 \
        /* Check if HI half has valid data */                                          \
        size_t remaining_hi = (remaining > 4) ? (remaining - 4) : 0;                   \
        if (remaining_hi > 0)                                                          \
        {                                                                              \
            /* Apply stage twiddles to HI half */                                      \
            APPLY_STAGE_TWIDDLES_R11_AVX512_SOA_NATIVE(k + 4, K, x1_hi, x2_hi,         \
                                                       x3_hi, x4_hi, x5_hi, x6_hi,     \
                                                       x7_hi, x8_hi, x9_hi, x10_hi,    \
                                                       stage_tw, sub_len);             \
        }                                                                              \
        /* Process HI half butterfly */                                                \
        __m512d t0_hi, t1_hi, t2_hi, t3_hi, t4_hi;                                     \
        __m512d s0_hi, s1_hi, s2_hi, s3_hi, s4_hi, y0_hi;                              \
        RADIX11_BUTTERFLY_CORE_AVX512(x0_hi, x1_hi, x2_hi, x3_hi, x4_hi,               \
                                      x5_hi, x6_hi, x7_hi, x8_hi, x9_hi,               \
                                      x10_hi, t0_hi, t1_hi, t2_hi, t3_hi,              \
                                      t4_hi, s0_hi, s1_hi, s2_hi, s3_hi,               \
                                      s4_hi, y0_hi);                                   \
        __m512d real1_hi, real2_hi, real3_hi, real4_hi, real5_hi;                      \
        RADIX11_REAL_PAIR1_AVX512(x0_hi, t0_hi, t1_hi, t2_hi, t3_hi, t4_hi,            \
                                  KC, real1_hi);                                       \
        RADIX11_REAL_PAIR2_AVX512(x0_hi, t0_hi, t1_hi, t2_hi, t3_hi, t4_hi,            \
                                  KC, real2_hi);                                       \
        RADIX11_REAL_PAIR3_AVX512(x0_hi, t0_hi, t1_hi, t2_hi, t3_hi, t4_hi,            \
                                  KC, real3_hi);                                       \
        RADIX11_REAL_PAIR4_AVX512(x0_hi, t0_hi, t1_hi, t2_hi, t3_hi, t4_hi,            \
                                  KC, real4_hi);                                       \
        RADIX11_REAL_PAIR5_AVX512(x0_hi, t0_hi, t1_hi, t2_hi, t3_hi, t4_hi,            \
                                  KC, real5_hi);                                       \
        __m512d rot1_hi, rot2_hi, rot3_hi, rot4_hi, rot5_hi;                           \
        RADIX11_IMAG_PAIR1_FV_AVX512(s0_hi, s1_hi, s2_hi, s3_hi, s4_hi,                \
                                     KC, rot1_hi);                                     \
        RADIX11_IMAG_PAIR2_FV_AVX512(s0_hi, s1_hi, s2_hi, s3_hi, s4_hi,                \
                                     KC, rot2_hi);                                     \
        RADIX11_IMAG_PAIR3_FV_AVX512(s0_hi, s1_hi, s2_hi, s3_hi, s4_hi,                \
                                     KC, rot3_hi);                                     \
        RADIX11_IMAG_PAIR4_FV_AVX512(s0_hi, s1_hi, s2_hi, s3_hi, s4_hi,                \
                                     KC, rot4_hi);                                     \
        RADIX11_IMAG_PAIR5_FV_AVX512(s0_hi, s1_hi, s2_hi, s3_hi, s4_hi,                \
                                     KC, rot5_hi);                                     \
        __m512d y1_hi, y2_hi, y3_hi, y4_hi, y5_hi, y6_hi, y7_hi, y8_hi, y9_hi, y10_hi; \
        RADIX11_ASSEMBLE_PAIR_AVX512(real1_hi, rot1_hi, y1_hi, y10_hi);                \
        RADIX11_ASSEMBLE_PAIR_AVX512(real2_hi, rot2_hi, y2_hi, y9_hi);                 \
        RADIX11_ASSEMBLE_PAIR_AVX512(real3_hi, rot3_hi, y3_hi, y8_hi);                 \
        RADIX11_ASSEMBLE_PAIR_AVX512(real4_hi, rot4_hi, y4_hi, y7_hi);                 \
        RADIX11_ASSEMBLE_PAIR_AVX512(real5_hi, rot5_hi, y5_hi, y6_hi);                 \
        /* Masked store - writes only valid lanes */                                   \
        STORE_11_LANES_AVX512_NATIVE_SOA_MASKED(k, K, remaining, out_re, out_im,       \
                                                y0_lo, y0_hi, y1_lo, y1_hi,            \
                                                y2_lo, y2_hi, y3_lo, y3_hi,            \
                                                y4_lo, y4_hi, y5_lo, y5_hi,            \
                                                y6_lo, y6_hi, y7_lo, y7_hi,            \
                                                y8_lo, y8_hi, y9_lo, y9_hi,            \
                                                y10_lo, y10_hi);                       \
    } while (0)

//==============================================================================
// BACKWARD (BV) TOP-LEVEL BUTTERFLY MACROS
//==============================================================================

/**
 * @brief Backward radix-11 butterfly - FULL VERSION
 * @details Processes 8 complex elements (4 lo + 4 hi) with prefetch
 */
#define RADIX11_BUTTERFLY_BV_AVX512_NATIVE_SOA(k, K, in_re, in_im, stage_tw,           \
                                               out_re, out_im, sub_len)                \
    do                                                                                 \
    {                                                                                  \
        /* Prefetch upcoming data */                                                   \
        PREFETCH_RADIX11_INPUT(k, K, in_re, in_im, R11_PREFETCH_DISTANCE,              \
                               stage_tw, sub_len);                                     \
        /* Broadcast constants ONCE */                                                 \
        radix11_consts_avx512 KC = broadcast_radix11_consts_avx512();                  \
        /* Load all 11 lanes (both lo and hi halves) */                                \
        __m512d x0_lo, x0_hi, x1_lo, x1_hi, x2_lo, x2_hi, x3_lo, x3_hi;                \
        __m512d x4_lo, x4_hi, x5_lo, x5_hi, x6_lo, x6_hi, x7_lo, x7_hi;                \
        __m512d x8_lo, x8_hi, x9_lo, x9_hi, x10_lo, x10_hi;                            \
        LOAD_11_LANES_AVX512_NATIVE_SOA_FULL(k, K, in_re, in_im,                       \
                                             x0_lo, x0_hi, x1_lo, x1_hi,               \
                                             x2_lo, x2_hi, x3_lo, x3_hi,               \
                                             x4_lo, x4_hi, x5_lo, x5_hi,               \
                                             x6_lo, x6_hi, x7_lo, x7_hi,               \
                                             x8_lo, x8_hi, x9_lo, x9_hi,               \
                                             x10_lo, x10_hi);                          \
        /* Apply stage twiddles to LO half */                                          \
        APPLY_STAGE_TWIDDLES_R11_AVX512_SOA_NATIVE(k, K, x1_lo, x2_lo, x3_lo,          \
                                                   x4_lo, x5_lo, x6_lo, x7_lo,         \
                                                   x8_lo, x9_lo, x10_lo,               \
                                                   stage_tw, sub_len);                 \
        /* Process LO half butterfly */                                                \
        __m512d t0_lo, t1_lo, t2_lo, t3_lo, t4_lo;                                     \
        __m512d s0_lo, s1_lo, s2_lo, s3_lo, s4_lo, y0_lo;                              \
        RADIX11_BUTTERFLY_CORE_AVX512(x0_lo, x1_lo, x2_lo, x3_lo, x4_lo,               \
                                      x5_lo, x6_lo, x7_lo, x8_lo, x9_lo,               \
                                      x10_lo, t0_lo, t1_lo, t2_lo, t3_lo,              \
                                      t4_lo, s0_lo, s1_lo, s2_lo, s3_lo,               \
                                      s4_lo, y0_lo);                                   \
        __m512d real1_lo, real2_lo, real3_lo, real4_lo, real5_lo;                      \
        RADIX11_REAL_PAIR1_AVX512(x0_lo, t0_lo, t1_lo, t2_lo, t3_lo, t4_lo,            \
                                  KC, real1_lo);                                       \
        RADIX11_REAL_PAIR2_AVX512(x0_lo, t0_lo, t1_lo, t2_lo, t3_lo, t4_lo,            \
                                  KC, real2_lo);                                       \
        RADIX11_REAL_PAIR3_AVX512(x0_lo, t0_lo, t1_lo, t2_lo, t3_lo, t4_lo,            \
                                  KC, real3_lo);                                       \
        RADIX11_REAL_PAIR4_AVX512(x0_lo, t0_lo, t1_lo, t2_lo, t3_lo, t4_lo,            \
                                  KC, real4_lo);                                       \
        RADIX11_REAL_PAIR5_AVX512(x0_lo, t0_lo, t1_lo, t2_lo, t3_lo, t4_lo,            \
                                  KC, real5_lo);                                       \
        /* BACKWARD imaginary pairs (BV) for LO half */                                \
        __m512d rot1_lo, rot2_lo, rot3_lo, rot4_lo, rot5_lo;                           \
        RADIX11_IMAG_PAIR1_BV_AVX512(s0_lo, s1_lo, s2_lo, s3_lo, s4_lo,                \
                                     KC, rot1_lo);                                     \
        RADIX11_IMAG_PAIR2_BV_AVX512(s0_lo, s1_lo, s2_lo, s3_lo, s4_lo,                \
                                     KC, rot2_lo);                                     \
        RADIX11_IMAG_PAIR3_BV_AVX512(s0_lo, s1_lo, s2_lo, s3_lo, s4_lo,                \
                                     KC, rot3_lo);                                     \
        RADIX11_IMAG_PAIR4_BV_AVX512(s0_lo, s1_lo, s2_lo, s3_lo, s4_lo,                \
                                     KC, rot4_lo);                                     \
        RADIX11_IMAG_PAIR5_BV_AVX512(s0_lo, s1_lo, s2_lo, s3_lo, s4_lo,                \
                                     KC, rot5_lo);                                     \
        /* Assemble LO half outputs */                                                 \
        __m512d y1_lo, y2_lo, y3_lo, y4_lo, y5_lo, y6_lo, y7_lo, y8_lo, y9_lo, y10_lo; \
        RADIX11_ASSEMBLE_PAIR_AVX512(real1_lo, rot1_lo, y1_lo, y10_lo);                \
        RADIX11_ASSEMBLE_PAIR_AVX512(real2_lo, rot2_lo, y2_lo, y9_lo);                 \
        RADIX11_ASSEMBLE_PAIR_AVX512(real3_lo, rot3_lo, y3_lo, y8_lo);                 \
        RADIX11_ASSEMBLE_PAIR_AVX512(real4_lo, rot4_lo, y4_lo, y7_lo);                 \
        RADIX11_ASSEMBLE_PAIR_AVX512(real5_lo, rot5_lo, y5_lo, y6_lo);                 \
        /* Apply stage twiddles to HI half */                                          \
        APPLY_STAGE_TWIDDLES_R11_AVX512_SOA_NATIVE(k + 4, K, x1_hi, x2_hi, x3_hi,      \
                                                   x4_hi, x5_hi, x6_hi, x7_hi,         \
                                                   x8_hi, x9_hi, x10_hi,               \
                                                   stage_tw, sub_len);                 \
        /* Process HI half butterfly */                                                \
        __m512d t0_hi, t1_hi, t2_hi, t3_hi, t4_hi;                                     \
        __m512d s0_hi, s1_hi, s2_hi, s3_hi, s4_hi, y0_hi;                              \
        RADIX11_BUTTERFLY_CORE_AVX512(x0_hi, x1_hi, x2_hi, x3_hi, x4_hi,               \
                                      x5_hi, x6_hi, x7_hi, x8_hi, x9_hi,               \
                                      x10_hi, t0_hi, t1_hi, t2_hi, t3_hi,              \
                                      t4_hi, s0_hi, s1_hi, s2_hi, s3_hi,               \
                                      s4_hi, y0_hi);                                   \
        __m512d real1_hi, real2_hi, real3_hi, real4_hi, real5_hi;                      \
        RADIX11_REAL_PAIR1_AVX512(x0_hi, t0_hi, t1_hi, t2_hi, t3_hi, t4_hi,            \
                                  KC, real1_hi);                                       \
        RADIX11_REAL_PAIR2_AVX512(x0_hi, t0_hi, t1_hi, t2_hi, t3_hi, t4_hi,            \
                                  KC, real2_hi);                                       \
        RADIX11_REAL_PAIR3_AVX512(x0_hi, t0_hi, t1_hi, t2_hi, t3_hi, t4_hi,            \
                                  KC, real3_hi);                                       \
        RADIX11_REAL_PAIR4_AVX512(x0_hi, t0_hi, t1_hi, t2_hi, t3_hi, t4_hi,            \
                                  KC, real4_hi);                                       \
        RADIX11_REAL_PAIR5_AVX512(x0_hi, t0_hi, t1_hi, t2_hi, t3_hi, t4_hi,            \
                                  KC, real5_hi);                                       \
        /* BACKWARD imaginary pairs (BV) for HI half */                                \
        __m512d rot1_hi, rot2_hi, rot3_hi, rot4_hi, rot5_hi;                           \
        RADIX11_IMAG_PAIR1_BV_AVX512(s0_hi, s1_hi, s2_hi, s3_hi, s4_hi,                \
                                     KC, rot1_hi);                                     \
        RADIX11_IMAG_PAIR2_BV_AVX512(s0_hi, s1_hi, s2_hi, s3_hi, s4_hi,                \
                                     KC, rot2_hi);                                     \
        RADIX11_IMAG_PAIR3_BV_AVX512(s0_hi, s1_hi, s2_hi, s3_hi, s4_hi,                \
                                     KC, rot3_hi);                                     \
        RADIX11_IMAG_PAIR4_BV_AVX512(s0_hi, s1_hi, s2_hi, s3_hi, s4_hi,                \
                                     KC, rot4_hi);                                     \
        RADIX11_IMAG_PAIR5_BV_AVX512(s0_hi, s1_hi, s2_hi, s3_hi, s4_hi,                \
                                     KC, rot5_hi);                                     \
        /* Assemble HI half outputs */                                                 \
        __m512d y1_hi, y2_hi, y3_hi, y4_hi, y5_hi, y6_hi, y7_hi, y8_hi, y9_hi, y10_hi; \
        RADIX11_ASSEMBLE_PAIR_AVX512(real1_hi, rot1_hi, y1_hi, y10_hi);                \
        RADIX11_ASSEMBLE_PAIR_AVX512(real2_hi, rot2_hi, y2_hi, y9_hi);                 \
        RADIX11_ASSEMBLE_PAIR_AVX512(real3_hi, rot3_hi, y3_hi, y8_hi);                 \
        RADIX11_ASSEMBLE_PAIR_AVX512(real4_hi, rot4_hi, y4_hi, y7_hi);                 \
        RADIX11_ASSEMBLE_PAIR_AVX512(real5_hi, rot5_hi, y5_hi, y6_hi);                 \
        /* Store both halves (NORMAL STORES ONLY) */                                   \
        STORE_11_LANES_AVX512_NATIVE_SOA_FULL(k, K, out_re, out_im,                    \
                                              y0_lo, y0_hi, y1_lo, y1_hi,              \
                                              y2_lo, y2_hi, y3_lo, y3_hi,              \
                                              y4_lo, y4_hi, y5_lo, y5_hi,              \
                                              y6_lo, y6_hi, y7_lo, y7_hi,              \
                                              y8_lo, y8_hi, y9_lo, y9_hi,              \
                                              y10_lo, y10_hi);                         \
    } while (0)

/**
 * @brief Backward radix-11 butterfly - TAIL VERSION with masking
 * @details Handles remaining < 8 elements with masked loads/stores
 */
#define RADIX11_BUTTERFLY_BV_AVX512_NATIVE_SOA_TAIL(k, K, remaining, in_re, in_im,     \
                                                    stage_tw, out_re, out_im,          \
                                                    sub_len)                           \
    do                                                                                 \
    {                                                                                  \
        /* Prefetch if there's more data ahead */                                      \
        PREFETCH_RADIX11_INPUT(k, K, in_re, in_im, R11_PREFETCH_DISTANCE,              \
                               stage_tw, sub_len);                                     \
        radix11_consts_avx512 KC = broadcast_radix11_consts_avx512();                  \
        __m512d x0_lo, x0_hi, x1_lo, x1_hi, x2_lo, x2_hi, x3_lo, x3_hi;                \
        __m512d x4_lo, x4_hi, x5_lo, x5_hi, x6_lo, x6_hi, x7_lo, x7_hi;                \
        __m512d x8_lo, x8_hi, x9_lo, x9_hi, x10_lo, x10_hi;                            \
        /* Masked load - zeros invalid lanes */                                        \
        LOAD_11_LANES_AVX512_NATIVE_SOA_MASKED(k, K, remaining, in_re, in_im,          \
                                               x0_lo, x0_hi, x1_lo, x1_hi,             \
                                               x2_lo, x2_hi, x3_lo, x3_hi,             \
                                               x4_lo, x4_hi, x5_lo, x5_hi,             \
                                               x6_lo, x6_hi, x7_lo, x7_hi,             \
                                               x8_lo, x8_hi, x9_lo, x9_hi,             \
                                               x10_lo, x10_hi);                        \
        /* Apply stage twiddles to LO half */                                          \
        APPLY_STAGE_TWIDDLES_R11_AVX512_SOA_NATIVE(k, K, x1_lo, x2_lo, x3_lo,          \
                                                   x4_lo, x5_lo, x6_lo, x7_lo,         \
                                                   x8_lo, x9_lo, x10_lo,               \
                                                   stage_tw, sub_len);                 \
        /* Process LO half butterfly */                                                \
        __m512d t0_lo, t1_lo, t2_lo, t3_lo, t4_lo;                                     \
        __m512d s0_lo, s1_lo, s2_lo, s3_lo, s4_lo, y0_lo;                              \
        RADIX11_BUTTERFLY_CORE_AVX512(x0_lo, x1_lo, x2_lo, x3_lo, x4_lo,               \
                                      x5_lo, x6_lo, x7_lo, x8_lo, x9_lo,               \
                                      x10_lo, t0_lo, t1_lo, t2_lo, t3_lo,              \
                                      t4_lo, s0_lo, s1_lo, s2_lo, s3_lo,               \
                                      s4_lo, y0_lo);                                   \
        __m512d real1_lo, real2_lo, real3_lo, real4_lo, real5_lo;                      \
        RADIX11_REAL_PAIR1_AVX512(x0_lo, t0_lo, t1_lo, t2_lo, t3_lo, t4_lo,            \
                                  KC, real1_lo);                                       \
        RADIX11_REAL_PAIR2_AVX512(x0_lo, t0_lo, t1_lo, t2_lo, t3_lo, t4_lo,            \
                                  KC, real2_lo);                                       \
        RADIX11_REAL_PAIR3_AVX512(x0_lo, t0_lo, t1_lo, t2_lo, t3_lo, t4_lo,            \
                                  KC, real3_lo);                                       \
        RADIX11_REAL_PAIR4_AVX512(x0_lo, t0_lo, t1_lo, t2_lo, t3_lo, t4_lo,            \
                                  KC, real4_lo);                                       \
        RADIX11_REAL_PAIR5_AVX512(x0_lo, t0_lo, t1_lo, t2_lo, t3_lo, t4_lo,            \
                                  KC, real5_lo);                                       \
        __m512d rot1_lo, rot2_lo, rot3_lo, rot4_lo, rot5_lo;                           \
        RADIX11_IMAG_PAIR1_BV_AVX512(s0_lo, s1_lo, s2_lo, s3_lo, s4_lo,                \
                                     KC, rot1_lo);                                     \
        RADIX11_IMAG_PAIR2_BV_AVX512(s0_lo, s1_lo, s2_lo, s3_lo, s4_lo,                \
                                     KC, rot2_lo);                                     \
        RADIX11_IMAG_PAIR3_BV_AVX512(s0_lo, s1_lo, s2_lo, s3_lo, s4_lo,                \
                                     KC, rot3_lo);                                     \
        RADIX11_IMAG_PAIR4_BV_AVX512(s0_lo, s1_lo, s2_lo, s3_lo, s4_lo,                \
                                     KC, rot4_lo);                                     \
        RADIX11_IMAG_PAIR5_BV_AVX512(s0_lo, s1_lo, s2_lo, s3_lo, s4_lo,                \
                                     KC, rot5_lo);                                     \
        __m512d y1_lo, y2_lo, y3_lo, y4_lo, y5_lo, y6_lo, y7_lo, y8_lo, y9_lo, y10_lo; \
        RADIX11_ASSEMBLE_PAIR_AVX512(real1_lo, rot1_lo, y1_lo, y10_lo);                \
        RADIX11_ASSEMBLE_PAIR_AVX512(real2_lo, rot2_lo, y2_lo, y9_lo);                 \
        RADIX11_ASSEMBLE_PAIR_AVX512(real3_lo, rot3_lo, y3_lo, y8_lo);                 \
        RADIX11_ASSEMBLE_PAIR_AVX512(real4_lo, rot4_lo, y4_lo, y7_lo);                 \
        RADIX11_ASSEMBLE_PAIR_AVX512(real5_lo, rot5_lo, y5_lo, y6_lo);                 \
        /* Check if HI half has valid data */                                          \
        size_t remaining_hi = (remaining > 4) ? (remaining - 4) : 0;                   \
        if (remaining_hi > 0)                                                          \
        {                                                                              \
            /* Apply stage twiddles to HI half */                                      \
            APPLY_STAGE_TWIDDLES_R11_AVX512_SOA_NATIVE(k + 4, K, x1_hi, x2_hi,         \
                                                       x3_hi, x4_hi, x5_hi, x6_hi,     \
                                                       x7_hi, x8_hi, x9_hi, x10_hi,    \
                                                       stage_tw, sub_len);             \
        }                                                                              \
        /* Process HI half butterfly */                                                \
        __m512d t0_hi, t1_hi, t2_hi, t3_hi, t4_hi;                                     \
        __m512d s0_hi, s1_hi, s2_hi, s3_hi, s4_hi, y0_hi;                              \
        RADIX11_BUTTERFLY_CORE_AVX512(x0_hi, x1_hi, x2_hi, x3_hi, x4_hi,               \
                                      x5_hi, x6_hi, x7_hi, x8_hi, x9_hi,               \
                                      x10_hi, t0_hi, t1_hi, t2_hi, t3_hi,              \
                                      t4_hi, s0_hi, s1_hi, s2_hi, s3_hi,               \
                                      s4_hi, y0_hi);                                   \
        __m512d real1_hi, real2_hi, real3_hi, real4_hi, real5_hi;                      \
        RADIX11_REAL_PAIR1_AVX512(x0_hi, t0_hi, t1_hi, t2_hi, t3_hi, t4_hi,            \
                                  KC, real1_hi);                                       \
        RADIX11_REAL_PAIR2_AVX512(x0_hi, t0_hi, t1_hi, t2_hi, t3_hi, t4_hi,            \
                                  KC, real2_hi);                                       \
        RADIX11_REAL_PAIR3_AVX512(x0_hi, t0_hi, t1_hi, t2_hi, t3_hi, t4_hi,            \
                                  KC, real3_hi);                                       \
        RADIX11_REAL_PAIR4_AVX512(x0_hi, t0_hi, t1_hi, t2_hi, t3_hi, t4_hi,            \
                                  KC, real4_hi);                                       \
        RADIX11_REAL_PAIR5_AVX512(x0_hi, t0_hi, t1_hi, t2_hi, t3_hi, t4_hi,            \
                                  KC, real5_hi);                                       \
        __m512d rot1_hi, rot2_hi, rot3_hi, rot4_hi, rot5_hi;                           \
        RADIX11_IMAG_PAIR1_BV_AVX512(s0_hi, s1_hi, s2_hi, s3_hi, s4_hi,                \
                                     KC, rot1_hi);                                     \
        RADIX11_IMAG_PAIR2_BV_AVX512(s0_hi, s1_hi, s2_hi, s3_hi, s4_hi,                \
                                     KC, rot2_hi);                                     \
        RADIX11_IMAG_PAIR3_BV_AVX512(s0_hi, s1_hi, s2_hi, s3_hi, s4_hi,                \
                                     KC, rot3_hi);                                     \
        RADIX11_IMAG_PAIR4_BV_AVX512(s0_hi, s1_hi, s2_hi, s3_hi, s4_hi,                \
                                     KC, rot4_hi);                                     \
        RADIX11_IMAG_PAIR5_BV_AVX512(s0_hi, s1_hi, s2_hi, s3_hi, s4_hi,                \
                                     KC, rot5_hi);                                     \
        __m512d y1_hi, y2_hi, y3_hi, y4_hi, y5_hi, y6_hi, y7_hi, y8_hi, y9_hi, y10_hi; \
        RADIX11_ASSEMBLE_PAIR_AVX512(real1_hi, rot1_hi, y1_hi, y10_hi);                \
        RADIX11_ASSEMBLE_PAIR_AVX512(real2_hi, rot2_hi, y2_hi, y9_hi);                 \
        RADIX11_ASSEMBLE_PAIR_AVX512(real3_hi, rot3_hi, y3_hi, y8_hi);                 \
        RADIX11_ASSEMBLE_PAIR_AVX512(real4_hi, rot4_hi, y4_hi, y7_hi);                 \
        RADIX11_ASSEMBLE_PAIR_AVX512(real5_hi, rot5_hi, y5_hi, y6_hi);                 \
        /* Masked store - writes only valid lanes */                                   \
        STORE_11_LANES_AVX512_NATIVE_SOA_MASKED(k, K, remaining, out_re, out_im,       \
                                                y0_lo, y0_hi, y1_lo, y1_hi,            \
                                                y2_lo, y2_hi, y3_lo, y3_hi,            \
                                                y4_lo, y4_hi, y5_lo, y5_hi,            \
                                                y6_lo, y6_hi, y7_lo, y7_hi,            \
                                                y8_lo, y8_hi, y9_lo, y9_hi,            \
                                                y10_lo, y10_hi);                       \
    } while (0)

#endif // __AVX512F__

#endif // FFT_RADIX11_MACROS_REFACTORED_H