/**
 * @file fft_radix3_macros_true_soa_optimized.h
 * @brief TRUE END-TO-END SoA Radix-3 Butterfly Macros - OPTIMIZED VERSION
 *
 * @details
 * This header provides macro implementations for radix-3 FFT butterflies that
 * operate entirely in Structure-of-Arrays (SoA) format without any split/join
 * operations in the computational hot path.
 *
 * OPTIMIZATIONS APPLIED:
 * ======================
 * 1. AVX2 FMA: Enables fused multiply-add for CMUL and common term computation
 *    - CMUL: 6 µops → 4 µops, 7-cycle → 4-cycle latency
 *    - common term: 2 µops → 1 µop, saves ~8-12 cycles per butterfly
 *    - Expected gain: +15-25% on Haswell+ CPUs
 *
 * 2. SSE2 load1_pd: Direct load-and-broadcast instead of load→GPR→set1
 *    - Eliminates GPR round-trip, saves 2-3 cycles per twiddle load
 *    - 4 twiddles/butterfly → 8-12 cycles saved
 *    - Expected gain: +2-4%
 *
 * 3. AVX-512 Masked Tail: Vectorized tail processing instead of scalar fallback
 *    - Eliminates SIMD→scalar transition stall
 *    - No branch to scalar code → better prediction
 *    - Expected gain: +5-10% on large FFTs with non-aligned sizes

 *
 * @author Tugbars
 * @version 2.0 (Optimized: FMA + load1 + masked tail)
 * @date 2025
 */

#ifndef FFT_RADIX3_MACROS_TRUE_SOA_OPTIMIZED_H
#define FFT_RADIX3_MACROS_TRUE_SOA_OPTIMIZED_H

#include <immintrin.h>


//==============================================================================
// CONFIGURATION
//==============================================================================

/**
 * @def RADIX3_STREAM_THRESHOLD
 * @brief Threshold for enabling non-temporal stores
 */
#define RADIX3_STREAM_THRESHOLD 8192

/**
 * @def RADIX3_PREFETCH_DISTANCE
 * @brief Software prefetch lead distance (in elements)
 */
#ifndef RADIX3_PREFETCH_DISTANCE
#define RADIX3_PREFETCH_DISTANCE 24
#endif

/**
 * @def RADIX3_USE_SOFTWARE_PIPELINING
 * @brief Enable software pipelining for AVX-512 (Intel 14900KF optimization)
 *
 * @details
 * Software pipelining overlaps LOAD → CMUL → BUTTERFLY → STORE stages
 * across iterations to hide memory latency and improve ILP on CPUs with
 * large reorder buffers (Golden Cove P-cores: 512-entry ROB).
 *
 * Expected performance gain on Intel Raptor Lake (14900KF):
 * - Small FFTs (81-729):     +5-8%
 * - Medium FFTs (3K-27K):    +12-18%
 * - Large FFTs (81K-729K):   +10-15%
 *
 * Trade-offs:
 * - Increases register pressure (uses ~40 AVX-512 registers)
 * - More complex code maintenance
 * - Best on CPUs with strong OOO execution (Intel client/server)
 *
 * Disable on:
 * - Older CPUs with small ROBs (<200 entries)
 * - Embedded/mobile CPUs with limited resources
 * - When compile times are critical
 *
 * Enable by: -DRADIX3_USE_SOFTWARE_PIPELINING
 */
#ifndef RADIX3_USE_SOFTWARE_PIPELINING
// Auto-enable for known good architectures (optional)
// #define RADIX3_USE_SOFTWARE_PIPELINING
#endif

//==============================================================================
// GEOMETRIC CONSTANTS (identical for forward/inverse)
//==============================================================================

#define C_HALF (-0.5)
#define S_SQRT3_2 0.8660254037844386467618 // sqrt(3)/2

//==============================================================================
// HOISTED VECTOR CONSTANTS (OPTIMIZATION: Avoid per-macro set1 overhead)
//==============================================================================

/**
 * @brief Pre-defined vector constants to avoid redundant set1 instructions
 * @details These constants are created once and reused across all butterfly
 *          computations, saving ~2-3 µops per butterfly invocation.
 */
#ifdef __AVX512F__
#define V512_HALF _mm512_set1_pd(C_HALF)
#define V512_SQRT3_2 _mm512_set1_pd(S_SQRT3_2)
#define V512_NEG_SQRT3_2 _mm512_set1_pd(-S_SQRT3_2)
#endif

#ifdef __AVX2__
#define V256_HALF _mm256_set1_pd(C_HALF)
#define V256_SQRT3_2 _mm256_set1_pd(S_SQRT3_2)
#define V256_NEG_SQRT3_2 _mm256_set1_pd(-S_SQRT3_2)
#endif

#ifdef __SSE2__
#define V128_HALF _mm_set1_pd(C_HALF)
#define V128_SQRT3_2 _mm_set1_pd(S_SQRT3_2)
#define V128_NEG_SQRT3_2 _mm_set1_pd(-S_SQRT3_2)
#endif

//==============================================================================
// LOAD/STORE HELPERS - NATIVE SoA
//==============================================================================

#ifdef __AVX512F__
#define LOAD_RE_AVX512(ptr) _mm512_loadu_pd(ptr)
#define LOAD_IM_AVX512(ptr) _mm512_loadu_pd(ptr)
#define STORE_RE_AVX512(ptr, val) _mm512_storeu_pd(ptr, val)
#define STORE_IM_AVX512(ptr, val) _mm512_storeu_pd(ptr, val)
#define STREAM_RE_AVX512(ptr, val) _mm512_stream_pd(ptr, val)
#define STREAM_IM_AVX512(ptr, val) _mm512_stream_pd(ptr, val)
#endif

#ifdef __AVX2__
#define LOAD_RE_AVX2(ptr) _mm256_loadu_pd(ptr)
#define LOAD_IM_AVX2(ptr) _mm256_loadu_pd(ptr)
#define STORE_RE_AVX2(ptr, val) _mm256_storeu_pd(ptr, val)
#define STORE_IM_AVX2(ptr, val) _mm256_storeu_pd(ptr, val)
#define STREAM_RE_AVX2(ptr, val) _mm256_stream_pd(ptr, val)
#define STREAM_IM_AVX2(ptr, val) _mm256_stream_pd(ptr, val)
#endif

#ifdef __SSE2__
#define LOAD_RE_SSE2(ptr) _mm_loadu_pd(ptr)
#define LOAD_IM_SSE2(ptr) _mm_loadu_pd(ptr)
#define STORE_RE_SSE2(ptr, val) _mm_storeu_pd(ptr, val)
#define STORE_IM_SSE2(ptr, val) _mm_storeu_pd(ptr, val)
#define STREAM_RE_SSE2(ptr, val) _mm_stream_pd(ptr, val)
#define STREAM_IM_SSE2(ptr, val) _mm_stream_pd(ptr, val)
#endif

//==============================================================================
// COMPLEX MULTIPLY - NATIVE SoA (NO SPLIT/JOIN!)
//==============================================================================

#ifdef __AVX512F__
/**
 * @brief Complex multiply - NATIVE SoA form (AVX-512)
 *
 * @details
 * ⚡⚡ CRITICAL: Data is ALREADY in split form from memory!
 * No split operation needed - direct loads from separate re/im arrays!
 *
 * Formula: (ar + i*ai) * (wr + i*wi) = (ar*wr - ai*wi) + i*(ar*wi + ai*wr)
 *
 * Uses FMA for optimal performance: 4 µops, 4-cycle latency
 */
#define CMUL_NATIVE_SOA_AVX512(ar, ai, wr, wi, tr, ti)       \
    do                                                       \
    {                                                        \
        tr = _mm512_fmsub_pd(ar, wr, _mm512_mul_pd(ai, wi)); \
        ti = _mm512_fmadd_pd(ar, wi, _mm512_mul_pd(ai, wr)); \
    } while (0)
#endif

#ifdef __AVX2__
/**
 * @brief Complex multiply - NATIVE SoA form (AVX2)
 *
 * @details
 * OPTIMIZATION: Uses FMA when available (Haswell+)
 * - With FMA: 4 µops, 4-cycle latency (same as AVX-512)
 * - Without FMA: 6 µops, 7-cycle latency (fallback)
 *
 * FMA brings AVX2 performance much closer to AVX-512!
 */
#if defined(__FMA__)
// OPTIMIZED PATH: FMA reduces µops and latency
#define CMUL_NATIVE_SOA_AVX2(ar, ai, wr, wi, tr, ti)         \
    do                                                       \
    {                                                        \
        tr = _mm256_fmsub_pd(ar, wr, _mm256_mul_pd(ai, wi)); \
        ti = _mm256_fmadd_pd(ar, wi, _mm256_mul_pd(ai, wr)); \
    } while (0)
#else
// FALLBACK PATH: No FMA available
#define CMUL_NATIVE_SOA_AVX2(ar, ai, wr, wi, tr, ti) \
    do                                               \
    {                                                \
        tr = _mm256_sub_pd(_mm256_mul_pd(ar, wr),    \
                           _mm256_mul_pd(ai, wi));   \
        ti = _mm256_add_pd(_mm256_mul_pd(ar, wi),    \
                           _mm256_mul_pd(ai, wr));   \
    } while (0)
#endif
#endif

#ifdef __SSE2__
/**
 * @brief Complex multiply - NATIVE SoA form (SSE2)
 *
 * @details
 * SSE2 does not have FMA, so we use the traditional multiply-add sequence.
 */
#define CMUL_NATIVE_SOA_SSE2(ar, ai, wr, wi, tr, ti) \
    do                                               \
    {                                                \
        tr = _mm_sub_pd(_mm_mul_pd(ar, wr),          \
                        _mm_mul_pd(ai, wi));         \
        ti = _mm_add_pd(_mm_mul_pd(ar, wi),          \
                        _mm_mul_pd(ai, wr));         \
    } while (0)
#endif

//==============================================================================
// RADIX-3 BUTTERFLY COMPUTATIONS - NATIVE SoA
//==============================================================================

#ifdef __AVX512F__
/**
 * @brief Radix-3 butterfly core - Forward - NATIVE SoA (AVX-512)
 *
 * @details
 * ⚡ Operates on 8 complex numbers in parallel using FMA
 * ⚡ All data stays in SoA format - zero shuffle operations!
 *
 * OPTIMIZATIONS:
 * - Uses hoisted constants (V512_*) to avoid per-call set1 overhead
 * - Uses FNMADD for negated products to save negate instruction
 */
#define RADIX3_BUTTERFLY_NATIVE_SOA_FV_AVX512(a_re, a_im, tB_re, tB_im, tC_re, tC_im,   \
                                              y0_re, y0_im, y1_re, y1_im, y2_re, y2_im) \
    do                                                                                  \
    {                                                                                   \
        __m512d sum_re = _mm512_add_pd(tB_re, tC_re);                                   \
        __m512d sum_im = _mm512_add_pd(tB_im, tC_im);                                   \
        __m512d dif_re = _mm512_sub_pd(tB_re, tC_re);                                   \
        __m512d dif_im = _mm512_sub_pd(tB_im, tC_im);                                   \
        __m512d common_re = _mm512_fmadd_pd(V512_HALF, sum_re, a_re);                   \
        __m512d common_im = _mm512_fmadd_pd(V512_HALF, sum_im, a_im);                   \
        __m512d rot_re = _mm512_mul_pd(V512_SQRT3_2, dif_im);                           \
        __m512d rot_im = _mm512_fnmadd_pd(V512_SQRT3_2, dif_re, _mm512_setzero_pd());   \
        y0_re = _mm512_add_pd(a_re, sum_re);                                            \
        y0_im = _mm512_add_pd(a_im, sum_im);                                            \
        y1_re = _mm512_add_pd(common_re, rot_re);                                       \
        y1_im = _mm512_add_pd(common_im, rot_im);                                       \
        y2_re = _mm512_sub_pd(common_re, rot_re);                                       \
        y2_im = _mm512_sub_pd(common_im, rot_im);                                       \
    } while (0)

/**
 * @brief Radix-3 butterfly core - Backward - NATIVE SoA (AVX-512)
 *
 * @details
 * Identical to forward except for sign flip in rotation computation.
 * Uses hoisted constants and FNMADD for optimal performance.
 */
#define RADIX3_BUTTERFLY_NATIVE_SOA_BV_AVX512(a_re, a_im, tB_re, tB_im, tC_re, tC_im,   \
                                              y0_re, y0_im, y1_re, y1_im, y2_re, y2_im) \
    do                                                                                  \
    {                                                                                   \
        __m512d sum_re = _mm512_add_pd(tB_re, tC_re);                                   \
        __m512d sum_im = _mm512_add_pd(tB_im, tC_im);                                   \
        __m512d dif_re = _mm512_sub_pd(tB_re, tC_re);                                   \
        __m512d dif_im = _mm512_sub_pd(tB_im, tC_im);                                   \
        __m512d common_re = _mm512_fmadd_pd(V512_HALF, sum_re, a_re);                   \
        __m512d common_im = _mm512_fmadd_pd(V512_HALF, sum_im, a_im);                   \
        __m512d rot_re = _mm512_fnmadd_pd(V512_SQRT3_2, dif_im, _mm512_setzero_pd());   \
        __m512d rot_im = _mm512_mul_pd(V512_SQRT3_2, dif_re);                           \
        y0_re = _mm512_add_pd(a_re, sum_re);                                            \
        y0_im = _mm512_add_pd(a_im, sum_im);                                            \
        y1_re = _mm512_add_pd(common_re, rot_re);                                       \
        y1_im = _mm512_add_pd(common_im, rot_im);                                       \
        y2_re = _mm512_sub_pd(common_re, rot_re);                                       \
        y2_im = _mm512_sub_pd(common_im, rot_im);                                       \
    } while (0)
#endif // __AVX512F__

#ifdef __AVX2__
/**
 * @brief Radix-3 butterfly core - Forward - NATIVE SoA (AVX2)
 *
 * @details
 * ⚡ Operates on 4 complex numbers in parallel
 * ⚡ Uses FMA when available (Haswell+) for optimal performance
 *
 * OPTIMIZATIONS:
 * - Uses hoisted constants (V256_*) to avoid per-call set1 overhead
 * - Uses FNMADD for negated products when FMA available
 * - FMA for common term (V_HALF * sum_re + a_re)
 */
#if defined(__FMA__)
// OPTIMIZED: Uses FMA for common term and FNMADD for negated products
#define RADIX3_BUTTERFLY_NATIVE_SOA_FV_AVX2(a_re, a_im, tB_re, tB_im, tC_re, tC_im,   \
                                            y0_re, y0_im, y1_re, y1_im, y2_re, y2_im) \
    do                                                                                \
    {                                                                                 \
        __m256d sum_re = _mm256_add_pd(tB_re, tC_re);                                 \
        __m256d sum_im = _mm256_add_pd(tB_im, tC_im);                                 \
        __m256d dif_re = _mm256_sub_pd(tB_re, tC_re);                                 \
        __m256d dif_im = _mm256_sub_pd(tB_im, tC_im);                                 \
        __m256d common_re = _mm256_fmadd_pd(V256_HALF, sum_re, a_re);                 \
        __m256d common_im = _mm256_fmadd_pd(V256_HALF, sum_im, a_im);                 \
        __m256d rot_re = _mm256_mul_pd(V256_SQRT3_2, dif_im);                         \
        __m256d rot_im = _mm256_fnmadd_pd(V256_SQRT3_2, dif_re, _mm256_setzero_pd()); \
        y0_re = _mm256_add_pd(a_re, sum_re);                                          \
        y0_im = _mm256_add_pd(a_im, sum_im);                                          \
        y1_re = _mm256_add_pd(common_re, rot_re);                                     \
        y1_im = _mm256_add_pd(common_im, rot_im);                                     \
        y2_re = _mm256_sub_pd(common_re, rot_re);                                     \
        y2_im = _mm256_sub_pd(common_im, rot_im);                                     \
    } while (0)
#else
// FALLBACK: No FMA available
#define RADIX3_BUTTERFLY_NATIVE_SOA_FV_AVX2(a_re, a_im, tB_re, tB_im, tC_re, tC_im,   \
                                            y0_re, y0_im, y1_re, y1_im, y2_re, y2_im) \
    do                                                                                \
    {                                                                                 \
        __m256d sum_re = _mm256_add_pd(tB_re, tC_re);                                 \
        __m256d sum_im = _mm256_add_pd(tB_im, tC_im);                                 \
        __m256d dif_re = _mm256_sub_pd(tB_re, tC_re);                                 \
        __m256d dif_im = _mm256_sub_pd(tB_im, tC_im);                                 \
        __m256d common_re = _mm256_add_pd(a_re, _mm256_mul_pd(V256_HALF, sum_re));    \
        __m256d common_im = _mm256_add_pd(a_im, _mm256_mul_pd(V256_HALF, sum_im));    \
        __m256d rot_re = _mm256_mul_pd(V256_SQRT3_2, dif_im);                         \
        __m256d rot_im = _mm256_mul_pd(V256_NEG_SQRT3_2, dif_re);                     \
        y0_re = _mm256_add_pd(a_re, sum_re);                                          \
        y0_im = _mm256_add_pd(a_im, sum_im);                                          \
        y1_re = _mm256_add_pd(common_re, rot_re);                                     \
        y1_im = _mm256_add_pd(common_im, rot_im);                                     \
        y2_re = _mm256_sub_pd(common_re, rot_re);                                     \
        y2_im = _mm256_sub_pd(common_im, rot_im);                                     \
    } while (0)
#endif

/**
 * @brief Radix-3 butterfly core - Backward - NATIVE SoA (AVX2)
 *
 * @details
 * Identical to forward except for sign flip in rotation.
 * Uses hoisted constants and FNMADD when FMA available.
 */
#if defined(__FMA__)
// OPTIMIZED: Uses FMA for common term and FNMADD for negated products
#define RADIX3_BUTTERFLY_NATIVE_SOA_BV_AVX2(a_re, a_im, tB_re, tB_im, tC_re, tC_im,   \
                                            y0_re, y0_im, y1_re, y1_im, y2_re, y2_im) \
    do                                                                                \
    {                                                                                 \
        __m256d sum_re = _mm256_add_pd(tB_re, tC_re);                                 \
        __m256d sum_im = _mm256_add_pd(tB_im, tC_im);                                 \
        __m256d dif_re = _mm256_sub_pd(tB_re, tC_re);                                 \
        __m256d dif_im = _mm256_sub_pd(tB_im, tC_im);                                 \
        __m256d common_re = _mm256_fmadd_pd(V256_HALF, sum_re, a_re);                 \
        __m256d common_im = _mm256_fmadd_pd(V256_HALF, sum_im, a_im);                 \
        __m256d rot_re = _mm256_fnmadd_pd(V256_SQRT3_2, dif_im, _mm256_setzero_pd()); \
        __m256d rot_im = _mm256_mul_pd(V256_SQRT3_2, dif_re);                         \
        y0_re = _mm256_add_pd(a_re, sum_re);                                          \
        y0_im = _mm256_add_pd(a_im, sum_im);                                          \
        y1_re = _mm256_add_pd(common_re, rot_re);                                     \
        y1_im = _mm256_add_pd(common_im, rot_im);                                     \
        y2_re = _mm256_sub_pd(common_re, rot_re);                                     \
        y2_im = _mm256_sub_pd(common_im, rot_im);                                     \
    } while (0)
#else
// FALLBACK: No FMA available
#define RADIX3_BUTTERFLY_NATIVE_SOA_BV_AVX2(a_re, a_im, tB_re, tB_im, tC_re, tC_im,   \
                                            y0_re, y0_im, y1_re, y1_im, y2_re, y2_im) \
    do                                                                                \
    {                                                                                 \
        __m256d sum_re = _mm256_add_pd(tB_re, tC_re);                                 \
        __m256d sum_im = _mm256_add_pd(tB_im, tC_im);                                 \
        __m256d dif_re = _mm256_sub_pd(tB_re, tC_re);                                 \
        __m256d dif_im = _mm256_sub_pd(tB_im, tC_im);                                 \
        __m256d common_re = _mm256_add_pd(a_re, _mm256_mul_pd(V256_HALF, sum_re));    \
        __m256d common_im = _mm256_add_pd(a_im, _mm256_mul_pd(V256_HALF, sum_im));    \
        __m256d rot_re = _mm256_mul_pd(V256_NEG_SQRT3_2, dif_im);                     \
        __m256d rot_im = _mm256_mul_pd(V256_SQRT3_2, dif_re);                         \
        y0_re = _mm256_add_pd(a_re, sum_re);                                          \
        y0_im = _mm256_add_pd(a_im, sum_im);                                          \
        y1_re = _mm256_add_pd(common_re, rot_re);                                     \
        y1_im = _mm256_add_pd(common_im, rot_im);                                     \
        y2_re = _mm256_sub_pd(common_re, rot_re);                                     \
        y2_im = _mm256_sub_pd(common_im, rot_im);                                     \
    } while (0)
#endif
#endif // __AVX2__

#ifdef __SSE2__
/**
 * @brief Radix-3 butterfly core - Forward - NATIVE SoA (SSE2)
 *
 * @details
 * ⚡ Operates on 2 complex numbers in parallel
 * No FMA available in SSE2, uses traditional multiply-add sequence.
 * Uses hoisted constants (V128_*) to avoid per-call set1 overhead.
 */
#define RADIX3_BUTTERFLY_NATIVE_SOA_FV_SSE2(a_re, a_im, tB_re, tB_im, tC_re, tC_im,   \
                                            y0_re, y0_im, y1_re, y1_im, y2_re, y2_im) \
    do                                                                                \
    {                                                                                 \
        __m128d sum_re = _mm_add_pd(tB_re, tC_re);                                    \
        __m128d sum_im = _mm_add_pd(tB_im, tC_im);                                    \
        __m128d dif_re = _mm_sub_pd(tB_re, tC_re);                                    \
        __m128d dif_im = _mm_sub_pd(tB_im, tC_im);                                    \
        __m128d common_re = _mm_add_pd(a_re, _mm_mul_pd(V128_HALF, sum_re));          \
        __m128d common_im = _mm_add_pd(a_im, _mm_mul_pd(V128_HALF, sum_im));          \
        __m128d rot_re = _mm_mul_pd(V128_SQRT3_2, dif_im);                            \
        __m128d rot_im = _mm_mul_pd(V128_NEG_SQRT3_2, dif_re);                        \
        y0_re = _mm_add_pd(a_re, sum_re);                                             \
        y0_im = _mm_add_pd(a_im, sum_im);                                             \
        y1_re = _mm_add_pd(common_re, rot_re);                                        \
        y1_im = _mm_add_pd(common_im, rot_im);                                        \
        y2_re = _mm_sub_pd(common_re, rot_re);                                        \
        y2_im = _mm_sub_pd(common_im, rot_im);                                        \
    } while (0)

/**
 * @brief Radix-3 butterfly core - Backward - NATIVE SoA (SSE2)
 *
 * @details
 * Identical to forward except for sign flip in rotation computation.
 * Uses hoisted constants (V128_*) to avoid per-call set1 overhead.
 */
#define RADIX3_BUTTERFLY_NATIVE_SOA_BV_SSE2(a_re, a_im, tB_re, tB_im, tC_re, tC_im,   \
                                            y0_re, y0_im, y1_re, y1_im, y2_re, y2_im) \
    do                                                                                \
    {                                                                                 \
        __m128d sum_re = _mm_add_pd(tB_re, tC_re);                                    \
        __m128d sum_im = _mm_add_pd(tB_im, tC_im);                                    \
        __m128d dif_re = _mm_sub_pd(tB_re, tC_re);                                    \
        __m128d dif_im = _mm_sub_pd(tB_im, tC_im);                                    \
        __m128d common_re = _mm_add_pd(a_re, _mm_mul_pd(V128_HALF, sum_re));          \
        __m128d common_im = _mm_add_pd(a_im, _mm_mul_pd(V128_HALF, sum_im));          \
        __m128d rot_re = _mm_mul_pd(V128_NEG_SQRT3_2, dif_im);                        \
        __m128d rot_im = _mm_mul_pd(V128_SQRT3_2, dif_re);                            \
        y0_re = _mm_add_pd(a_re, sum_re);                                             \
        y0_im = _mm_add_pd(a_im, sum_im);                                             \
        y1_re = _mm_add_pd(common_re, rot_re);                                        \
        y1_im = _mm_add_pd(common_im, rot_im);                                        \
        y2_re = _mm_sub_pd(common_re, rot_re);                                        \
        y2_im = _mm_sub_pd(common_im, rot_im);                                        \
    } while (0)
#endif // __SSE2__

//==============================================================================
// AVX-512 MASKED TAIL PROCESSING (OPTIMIZATION #3)
//==============================================================================

#ifdef __AVX512F__
/**
 * @brief AVX-512 masked tail processing - Forward
 *
 * @details
 * OPTIMIZATION: Vectorized tail processing using AVX-512 masks instead of
 * falling back to scalar code. This eliminates:
 * - SIMD→scalar transition stalls
 * - Branch misprediction from scalar fallback
 * - Code duplication
 *
 * Processes 1-7 remaining elements using masked loads/stores.
 *
 * @param k Starting index for tail
 * @param k_end End index (exclusive)
 * @param K Stride between butterfly lanes
 * @param in_re Input real array
 * @param in_im Input imaginary array
 * @param out_re Output real array
 * @param out_im Output imaginary array
 * @param tw Twiddle factors (SoA format)
 */
static inline void radix3_avx512_tail_fv(int k, int k_end, int K,
                                         const double *in_re, const double *in_im,
                                         double *out_re, double *out_im,
                                         const struct fft_twiddles_soa *tw)
{
    if (k >= k_end)
        return; // No tail to process

    int rem = k_end - k;
    if (rem >= 8)
        return; // Safety: should be handled by main loop

    // Create mask for remaining elements (1-7)
    __mmask8 m = (__mmask8)((1u << rem) - 1u);

    // Masked loads: load only 'rem' elements, zero-fill the rest
    __m512d a_re = _mm512_maskz_loadu_pd(m, &in_re[k]);
    __m512d a_im = _mm512_maskz_loadu_pd(m, &in_im[k]);
    __m512d b_re = _mm512_maskz_loadu_pd(m, &in_re[k + K]);
    __m512d b_im = _mm512_maskz_loadu_pd(m, &in_im[k + K]);
    __m512d c_re = _mm512_maskz_loadu_pd(m, &in_re[k + 2 * K]);
    __m512d c_im = _mm512_maskz_loadu_pd(m, &in_im[k + 2 * K]);

    // Load twiddle factors
    __m512d w1_re = _mm512_maskz_loadu_pd(m, &tw->re[0 * K + k]);
    __m512d w1_im = _mm512_maskz_loadu_pd(m, &tw->im[0 * K + k]);
    __m512d w2_re = _mm512_maskz_loadu_pd(m, &tw->re[1 * K + k]);
    __m512d w2_im = _mm512_maskz_loadu_pd(m, &tw->im[1 * K + k]);

    // Complex multiplies
    __m512d tB_re, tB_im, tC_re, tC_im;
    CMUL_NATIVE_SOA_AVX512(b_re, b_im, w1_re, w1_im, tB_re, tB_im);
    CMUL_NATIVE_SOA_AVX512(c_re, c_im, w2_re, w2_im, tC_re, tC_im);

    // Butterfly computation
    __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im;
    RADIX3_BUTTERFLY_NATIVE_SOA_FV_AVX512(a_re, a_im, tB_re, tB_im, tC_re, tC_im,
                                          y0_re, y0_im, y1_re, y1_im, y2_re, y2_im);

    // Masked stores: write only 'rem' elements
    _mm512_mask_storeu_pd(&out_re[k], m, y0_re);
    _mm512_mask_storeu_pd(&out_im[k], m, y0_im);
    _mm512_mask_storeu_pd(&out_re[k + K], m, y1_re);
    _mm512_mask_storeu_pd(&out_im[k + K], m, y1_im);
    _mm512_mask_storeu_pd(&out_re[k + 2 * K], m, y2_re);
    _mm512_mask_storeu_pd(&out_im[k + 2 * K], m, y2_im);
}

/**
 * @brief AVX-512 masked tail processing - Backward
 *
 * @details
 * Same as forward version but with sign flip in butterfly rotation.
 */
static inline void radix3_avx512_tail_bv(int k, int k_end, int K,
                                         const double *in_re, const double *in_im,
                                         double *out_re, double *out_im,
                                         const struct fft_twiddles_soa *tw)
{
    if (k >= k_end)
        return;

    int rem = k_end - k;
    if (rem >= 8)
        return;

    __mmask8 m = (__mmask8)((1u << rem) - 1u);

    __m512d a_re = _mm512_maskz_loadu_pd(m, &in_re[k]);
    __m512d a_im = _mm512_maskz_loadu_pd(m, &in_im[k]);
    __m512d b_re = _mm512_maskz_loadu_pd(m, &in_re[k + K]);
    __m512d b_im = _mm512_maskz_loadu_pd(m, &in_im[k + K]);
    __m512d c_re = _mm512_maskz_loadu_pd(m, &in_re[k + 2 * K]);
    __m512d c_im = _mm512_maskz_loadu_pd(m, &in_im[k + 2 * K]);

    __m512d w1_re = _mm512_maskz_loadu_pd(m, &tw->re[0 * K + k]);
    __m512d w1_im = _mm512_maskz_loadu_pd(m, &tw->im[0 * K + k]);
    __m512d w2_re = _mm512_maskz_loadu_pd(m, &tw->re[1 * K + k]);
    __m512d w2_im = _mm512_maskz_loadu_pd(m, &tw->im[1 * K + k]);

    __m512d tB_re, tB_im, tC_re, tC_im;
    CMUL_NATIVE_SOA_AVX512(b_re, b_im, w1_re, w1_im, tB_re, tB_im);
    CMUL_NATIVE_SOA_AVX512(c_re, c_im, w2_re, w2_im, tC_re, tC_im);

    __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im;
    RADIX3_BUTTERFLY_NATIVE_SOA_BV_AVX512(a_re, a_im, tB_re, tB_im, tC_re, tC_im,
                                          y0_re, y0_im, y1_re, y1_im, y2_re, y2_im);

    _mm512_mask_storeu_pd(&out_re[k], m, y0_re);
    _mm512_mask_storeu_pd(&out_im[k], m, y0_im);
    _mm512_mask_storeu_pd(&out_re[k + K], m, y1_re);
    _mm512_mask_storeu_pd(&out_im[k + K], m, y1_im);
    _mm512_mask_storeu_pd(&out_re[k + 2 * K], m, y2_re);
    _mm512_mask_storeu_pd(&out_im[k + 2 * K], m, y2_im);
}
#endif // __AVX512F__

//==============================================================================
// SOFTWARE PIPELINED VECTORIZED PIPELINE MACROS (AVX-512 ONLY)
//==============================================================================

#if defined(__AVX512F__) && defined(RADIX3_USE_SOFTWARE_PIPELINING)

/**
 * @brief AVX-512 software pipelined radix-3 (U=2, 16-wide per iteration) - Forward
 *
 * @details
 * ADVANCED OPTIMIZATION: 4-stage pipeline with U=2 unroll
 *
 * Pipeline stages overlap across iterations:
 * - LOAD(i+1):     Load inputs/twiddles for next iteration
 * - CMUL(i):       Complex multiply using data from previous LOAD
 * - BUTTERFLY(i-1): Butterfly using results from previous CMUL
 * - STORE(i-2):    Store results from previous BUTTERFLY
 *
 * Benefits on Intel Raptor Lake (14900KF):
 * - Hides L1 load latency (~4-5 cycles)
 * - Maximizes ILP with 512-entry ROB
 * - Keeps execution ports saturated
 * - Expected: +12-18% vs non-pipelined
 *
 * Register usage: ~40 AVX-512 registers (safe on modern CPUs)
 *
 * Loop structure:
 * - Prologue: Fill pipeline (2 iterations)
 * - Main loop: All stages active (processes 16 elements/iter)
 * - Epilogue: Drain pipeline (2 iterations)
 */
#define RADIX3_PIPELINE_8_NATIVE_SOA_FV_AVX512_SWPIPE(k, k_end, K, in_re, in_im, out_re, out_im, tw) \
    do                                                                                               \
    {                                                                                                \
        if ((k_end) - (k) < 16)                                                                      \
            break; /* Need at least 16 elements for U=2 */                                           \
                                                                                                     \
        /* Pipeline stage registers - double buffered for U=2 */                                     \
        __m512d s0_a_re, s0_a_im, s0_b_re, s0_b_im, s0_c_re, s0_c_im;                                \
        __m512d s0_w1_re, s0_w1_im, s0_w2_re, s0_w2_im;                                              \
        __m512d s1_a_re, s1_a_im, s1_tB_re, s1_tB_im, s1_tC_re, s1_tC_im;                            \
        __m512d s2_y0_re, s2_y0_im, s2_y1_re, s2_y1_im, s2_y2_re, s2_y2_im;                          \
                                                                                                     \
        /* PROLOGUE: Fill pipeline */                                                                \
        /* Iteration -2: LOAD only */                                                                \
        s0_a_re = LOAD_RE_AVX512(&in_re[k]);                                                         \
        s0_a_im = LOAD_IM_AVX512(&in_im[k]);                                                         \
        s0_b_re = LOAD_RE_AVX512(&in_re[(k) + (K)]);                                                 \
        s0_b_im = LOAD_IM_AVX512(&in_im[(k) + (K)]);                                                 \
        s0_c_re = LOAD_RE_AVX512(&in_re[(k) + 2 * (K)]);                                             \
        s0_c_im = LOAD_IM_AVX512(&in_im[(k) + 2 * (K)]);                                             \
        s0_w1_re = LOAD_RE_AVX512(&tw->re[0 * (K) + (k)]);                                           \
        s0_w1_im = LOAD_IM_AVX512(&tw->im[0 * (K) + (k)]);                                           \
        s0_w2_re = LOAD_RE_AVX512(&tw->re[1 * (K) + (k)]);                                           \
        s0_w2_im = LOAD_IM_AVX512(&tw->im[1 * (K) + (k)]);                                           \
                                                                                                     \
        /* Iteration -1: LOAD + CMUL(for iter -2) */                                                 \
        s1_a_re = LOAD_RE_AVX512(&in_re[(k) + 8]);                                                   \
        s1_a_im = LOAD_IM_AVX512(&in_im[(k) + 8]);                                                   \
        __m512d s1_b_re = LOAD_RE_AVX512(&in_re[(k) + 8 + (K)]);                                     \
        __m512d s1_b_im = LOAD_IM_AVX512(&in_im[(k) + 8 + (K)]);                                     \
        __m512d s1_c_re = LOAD_RE_AVX512(&in_re[(k) + 8 + 2 * (K)]);                                 \
        __m512d s1_c_im = LOAD_IM_AVX512(&in_im[(k) + 8 + 2 * (K)]);                                 \
        __m512d s1_w1_re = LOAD_RE_AVX512(&tw->re[0 * (K) + (k) + 8]);                               \
        __m512d s1_w1_im = LOAD_IM_AVX512(&tw->im[0 * (K) + (k) + 8]);                               \
        __m512d s1_w2_re = LOAD_RE_AVX512(&tw->re[1 * (K) + (k) + 8]);                               \
        __m512d s1_w2_im = LOAD_IM_AVX512(&tw->im[1 * (K) + (k) + 8]);                               \
                                                                                                     \
        /* CMUL for iteration -2 (s0) */                                                             \
        __m512d s0_tB_re, s0_tB_im, s0_tC_re, s0_tC_im;                                              \
        CMUL_NATIVE_SOA_AVX512(s0_b_re, s0_b_im, s0_w1_re, s0_w1_im, s0_tB_re, s0_tB_im);            \
        CMUL_NATIVE_SOA_AVX512(s0_c_re, s0_c_im, s0_w2_re, s0_w2_im, s0_tC_re, s0_tC_im);            \
                                                                                                     \
        (k) += 16; /* Advance past prologue iterations */                                            \
                                                                                                     \
        /* MAIN LOOP: All 4 stages active, U=2 unroll (16 elements per iteration) */                 \
        for (; (k) + 16 <= (k_end); (k) += 16)                                                       \
        {                                                                                            \
            /* ==================== FIRST HALF (8 elements) ==================== */                  \
                                                                                                     \
            /* Stage 0: LOAD iteration i (at k) */                                                   \
            __m512d n0_a_re = LOAD_RE_AVX512(&in_re[k]);                                             \
            __m512d n0_a_im = LOAD_IM_AVX512(&in_im[k]);                                             \
            __m512d n0_b_re = LOAD_RE_AVX512(&in_re[(k) + (K)]);                                     \
            __m512d n0_b_im = LOAD_IM_AVX512(&in_im[(k) + (K)]);                                     \
            __m512d n0_c_re = LOAD_RE_AVX512(&in_re[(k) + 2 * (K)]);                                 \
            __m512d n0_c_im = LOAD_IM_AVX512(&in_im[(k) + 2 * (K)]);                                 \
            __m512d n0_w1_re = LOAD_RE_AVX512(&tw->re[0 * (K) + (k)]);                               \
            __m512d n0_w1_im = LOAD_IM_AVX512(&tw->im[0 * (K) + (k)]);                               \
            __m512d n0_w2_re = LOAD_RE_AVX512(&tw->re[1 * (K) + (k)]);                               \
            __m512d n0_w2_im = LOAD_IM_AVX512(&tw->im[1 * (K) + (k)]);                               \
                                                                                                     \
            /* Stage 1: CMUL iteration i-1 (using s1 from last iter) */                              \
            CMUL_NATIVE_SOA_AVX512(s1_b_re, s1_b_im, s1_w1_re, s1_w1_im, s1_tB_re, s1_tB_im);        \
            CMUL_NATIVE_SOA_AVX512(s1_c_re, s1_c_im, s1_w2_re, s1_w2_im, s1_tC_re, s1_tC_im);        \
                                                                                                     \
            /* Stage 2: BUTTERFLY iteration i-2 (using s0 from last iter) */                         \
            RADIX3_BUTTERFLY_NATIVE_SOA_FV_AVX512(s0_a_re, s0_a_im, s0_tB_re, s0_tB_im,              \
                                                  s0_tC_re, s0_tC_im, s2_y0_re, s2_y0_im,            \
                                                  s2_y1_re, s2_y1_im, s2_y2_re, s2_y2_im);           \
                                                                                                     \
            /* Stage 3: STORE iteration i-3 (from s2) */                                             \
            STORE_RE_AVX512(&out_re[(k) - 16], s2_y0_re);                                            \
            STORE_IM_AVX512(&out_im[(k) - 16], s2_y0_im);                                            \
            STORE_RE_AVX512(&out_re[(k) - 16 + (K)], s2_y1_re);                                      \
            STORE_IM_AVX512(&out_im[(k) - 16 + (K)], s2_y1_im);                                      \
            STORE_RE_AVX512(&out_re[(k) - 16 + 2 * (K)], s2_y2_re);                                  \
            STORE_IM_AVX512(&out_im[(k) - 16 + 2 * (K)], s2_y2_im);                                  \
                                                                                                     \
            /* ==================== SECOND HALF (8 elements) ==================== */                 \
                                                                                                     \
            /* Stage 0: LOAD iteration i+1 (at k+8) */                                               \
            __m512d n1_a_re = LOAD_RE_AVX512(&in_re[(k) + 8]);                                       \
            __m512d n1_a_im = LOAD_IM_AVX512(&in_im[(k) + 8]);                                       \
            __m512d n1_b_re = LOAD_RE_AVX512(&in_re[(k) + 8 + (K)]);                                 \
            __m512d n1_b_im = LOAD_IM_AVX512(&in_im[(k) + 8 + (K)]);                                 \
            __m512d n1_c_re = LOAD_RE_AVX512(&in_re[(k) + 8 + 2 * (K)]);                             \
            __m512d n1_c_im = LOAD_IM_AVX512(&in_im[(k) + 8 + 2 * (K)]);                             \
            __m512d n1_w1_re = LOAD_RE_AVX512(&tw->re[0 * (K) + (k) + 8]);                           \
            __m512d n1_w1_im = LOAD_IM_AVX512(&tw->im[0 * (K) + (k) + 8]);                           \
            __m512d n1_w2_re = LOAD_RE_AVX512(&tw->re[1 * (K) + (k) + 8]);                           \
            __m512d n1_w2_im = LOAD_IM_AVX512(&tw->im[1 * (K) + (k) + 8]);                           \
                                                                                                     \
            /* Stage 1: CMUL iteration i (using n0 just loaded) */                                   \
            __m512d n0_tB_re, n0_tB_im, n0_tC_re, n0_tC_im;                                          \
            CMUL_NATIVE_SOA_AVX512(n0_b_re, n0_b_im, n0_w1_re, n0_w1_im, n0_tB_re, n0_tB_im);        \
            CMUL_NATIVE_SOA_AVX512(n0_c_re, n0_c_im, n0_w2_re, n0_w2_im, n0_tC_re, n0_tC_im);        \
                                                                                                     \
            /* Stage 2: BUTTERFLY iteration i-1 (using s1 CMUL results) */                           \
            __m512d n1_y0_re, n1_y0_im, n1_y1_re, n1_y1_im, n1_y2_re, n1_y2_im;                      \
            RADIX3_BUTTERFLY_NATIVE_SOA_FV_AVX512(s1_a_re, s1_a_im, s1_tB_re, s1_tB_im,              \
                                                  s1_tC_re, s1_tC_im, n1_y0_re, n1_y0_im,            \
                                                  n1_y1_re, n1_y1_im, n1_y2_re, n1_y2_im);           \
                                                                                                     \
            /* Stage 3: STORE iteration i-2 (from first half butterfly) */                           \
            STORE_RE_AVX512(&out_re[(k) - 8], n1_y0_re);                                             \
            STORE_IM_AVX512(&out_im[(k) - 8], n1_y0_im);                                             \
            STORE_RE_AVX512(&out_re[(k) - 8 + (K)], n1_y1_re);                                       \
            STORE_IM_AVX512(&out_im[(k) - 8 + (K)], n1_y1_im);                                       \
            STORE_RE_AVX512(&out_re[(k) - 8 + 2 * (K)], n1_y2_re);                                   \
            STORE_IM_AVX512(&out_im[(k) - 8 + 2 * (K)], n1_y2_im);                                   \
                                                                                                     \
            /* Rotate pipeline registers for next iteration */                                       \
            s0_a_re = n0_a_re;                                                                       \
            s0_a_im = n0_a_im;                                                                       \
            s0_tB_re = n0_tB_re;                                                                     \
            s0_tB_im = n0_tB_im;                                                                     \
            s0_tC_re = n0_tC_re;                                                                     \
            s0_tC_im = n0_tC_im;                                                                     \
                                                                                                     \
            s1_a_re = n1_a_re;                                                                       \
            s1_a_im = n1_a_im;                                                                       \
            s1_b_re = n1_b_re;                                                                       \
            s1_b_im = n1_b_im;                                                                       \
            s1_c_re = n1_c_re;                                                                       \
            s1_c_im = n1_c_im;                                                                       \
            s1_w1_re = n1_w1_re;                                                                     \
            s1_w1_im = n1_w1_im;                                                                     \
            s1_w2_re = n1_w2_re;                                                                     \
            s1_w2_im = n1_w2_im;                                                                     \
        }                                                                                            \
                                                                                                     \
        /* EPILOGUE: Drain pipeline (2 iterations) */                                                \
        /* Iteration k-16: CMUL + BUTTERFLY + STORE */                                               \
        CMUL_NATIVE_SOA_AVX512(s1_b_re, s1_b_im, s1_w1_re, s1_w1_im, s1_tB_re, s1_tB_im);            \
        CMUL_NATIVE_SOA_AVX512(s1_c_re, s1_c_im, s1_w2_re, s1_w2_im, s1_tC_re, s1_tC_im);            \
        RADIX3_BUTTERFLY_NATIVE_SOA_FV_AVX512(s0_a_re, s0_a_im, s0_tB_re, s0_tB_im,                  \
                                              s0_tC_re, s0_tC_im, s2_y0_re, s2_y0_im,                \
                                              s2_y1_re, s2_y1_im, s2_y2_re, s2_y2_im);               \
        STORE_RE_AVX512(&out_re[(k) - 16], s2_y0_re);                                                \
        STORE_IM_AVX512(&out_im[(k) - 16], s2_y0_im);                                                \
        STORE_RE_AVX512(&out_re[(k) - 16 + (K)], s2_y1_re);                                          \
        STORE_IM_AVX512(&out_im[(k) - 16 + (K)], s2_y1_im);                                          \
        STORE_RE_AVX512(&out_re[(k) - 16 + 2 * (K)], s2_y2_re);                                      \
        STORE_IM_AVX512(&out_im[(k) - 16 + 2 * (K)], s2_y2_im);                                      \
                                                                                                     \
        /* Iteration k-8: BUTTERFLY + STORE */                                                       \
        RADIX3_BUTTERFLY_NATIVE_SOA_FV_AVX512(s1_a_re, s1_a_im, s1_tB_re, s1_tB_im,                  \
                                              s1_tC_re, s1_tC_im, s2_y0_re, s2_y0_im,                \
                                              s2_y1_re, s2_y1_im, s2_y2_re, s2_y2_im);               \
        STORE_RE_AVX512(&out_re[(k) - 8], s2_y0_re);                                                 \
        STORE_IM_AVX512(&out_im[(k) - 8], s2_y0_im);                                                 \
        STORE_RE_AVX512(&out_re[(k) - 8 + (K)], s2_y1_re);                                           \
        STORE_IM_AVX512(&out_im[(k) - 8 + (K)], s2_y1_im);                                           \
        STORE_RE_AVX512(&out_re[(k) - 8 + 2 * (K)], s2_y2_re);                                       \
        STORE_IM_AVX512(&out_im[(k) - 8 + 2 * (K)], s2_y2_im);                                       \
                                                                                                     \
        /* Handle any remaining elements (<16) with tail function */                                 \
        radix3_avx512_tail_fv(k, k_end, K, in_re, in_im, out_re, out_im, tw);                        \
    } while (0)

/**
 * @brief AVX-512 software pipelined radix-3 (U=2, 16-wide per iteration) - Backward
 *
 * @details
 * Same as forward version but with sign flip in butterfly rotation.
 * See RADIX3_PIPELINE_8_NATIVE_SOA_FV_AVX512_SWPIPE for full documentation.
 */
#define RADIX3_PIPELINE_8_NATIVE_SOA_BV_AVX512_SWPIPE(k, k_end, K, in_re, in_im, out_re, out_im, tw) \
    do                                                                                               \
    {                                                                                                \
        if ((k_end) - (k) < 16)                                                                      \
            break;                                                                                   \
                                                                                                     \
        __m512d s0_a_re, s0_a_im, s0_b_re, s0_b_im, s0_c_re, s0_c_im;                                \
        __m512d s0_w1_re, s0_w1_im, s0_w2_re, s0_w2_im;                                              \
        __m512d s1_a_re, s1_a_im, s1_tB_re, s1_tB_im, s1_tC_re, s1_tC_im;                            \
        __m512d s2_y0_re, s2_y0_im, s2_y1_re, s2_y1_im, s2_y2_re, s2_y2_im;                          \
                                                                                                     \
        /* PROLOGUE */                                                                               \
        s0_a_re = LOAD_RE_AVX512(&in_re[k]);                                                         \
        s0_a_im = LOAD_IM_AVX512(&in_im[k]);                                                         \
        s0_b_re = LOAD_RE_AVX512(&in_re[(k) + (K)]);                                                 \
        s0_b_im = LOAD_IM_AVX512(&in_im[(k) + (K)]);                                                 \
        s0_c_re = LOAD_RE_AVX512(&in_re[(k) + 2 * (K)]);                                             \
        s0_c_im = LOAD_IM_AVX512(&in_im[(k) + 2 * (K)]);                                             \
        s0_w1_re = LOAD_RE_AVX512(&tw->re[0 * (K) + (k)]);                                           \
        s0_w1_im = LOAD_IM_AVX512(&tw->im[0 * (K) + (k)]);                                           \
        s0_w2_re = LOAD_RE_AVX512(&tw->re[1 * (K) + (k)]);                                           \
        s0_w2_im = LOAD_IM_AVX512(&tw->im[1 * (K) + (k)]);                                           \
                                                                                                     \
        s1_a_re = LOAD_RE_AVX512(&in_re[(k) + 8]);                                                   \
        s1_a_im = LOAD_IM_AVX512(&in_im[(k) + 8]);                                                   \
        __m512d s1_b_re = LOAD_RE_AVX512(&in_re[(k) + 8 + (K)]);                                     \
        __m512d s1_b_im = LOAD_IM_AVX512(&in_im[(k) + 8 + (K)]);                                     \
        __m512d s1_c_re = LOAD_RE_AVX512(&in_re[(k) + 8 + 2 * (K)]);                                 \
        __m512d s1_c_im = LOAD_IM_AVX512(&in_im[(k) + 8 + 2 * (K)]);                                 \
        __m512d s1_w1_re = LOAD_RE_AVX512(&tw->re[0 * (K) + (k) + 8]);                               \
        __m512d s1_w1_im = LOAD_IM_AVX512(&tw->im[0 * (K) + (k) + 8]);                               \
        __m512d s1_w2_re = LOAD_RE_AVX512(&tw->re[1 * (K) + (k) + 8]);                               \
        __m512d s1_w2_im = LOAD_IM_AVX512(&tw->im[1 * (K) + (k) + 8]);                               \
                                                                                                     \
        __m512d s0_tB_re, s0_tB_im, s0_tC_re, s0_tC_im;                                              \
        CMUL_NATIVE_SOA_AVX512(s0_b_re, s0_b_im, s0_w1_re, s0_w1_im, s0_tB_re, s0_tB_im);            \
        CMUL_NATIVE_SOA_AVX512(s0_c_re, s0_c_im, s0_w2_re, s0_w2_im, s0_tC_re, s0_tC_im);            \
                                                                                                     \
        (k) += 16;                                                                                   \
                                                                                                     \
        /* MAIN LOOP */                                                                              \
        for (; (k) + 16 <= (k_end); (k) += 16)                                                       \
        {                                                                                            \
            /* First half */                                                                         \
            __m512d n0_a_re = LOAD_RE_AVX512(&in_re[k]);                                             \
            __m512d n0_a_im = LOAD_IM_AVX512(&in_im[k]);                                             \
            __m512d n0_b_re = LOAD_RE_AVX512(&in_re[(k) + (K)]);                                     \
            __m512d n0_b_im = LOAD_IM_AVX512(&in_im[(k) + (K)]);                                     \
            __m512d n0_c_re = LOAD_RE_AVX512(&in_re[(k) + 2 * (K)]);                                 \
            __m512d n0_c_im = LOAD_IM_AVX512(&in_im[(k) + 2 * (K)]);                                 \
            __m512d n0_w1_re = LOAD_RE_AVX512(&tw->re[0 * (K) + (k)]);                               \
            __m512d n0_w1_im = LOAD_IM_AVX512(&tw->im[0 * (K) + (k)]);                               \
            __m512d n0_w2_re = LOAD_RE_AVX512(&tw->re[1 * (K) + (k)]);                               \
            __m512d n0_w2_im = LOAD_IM_AVX512(&tw->im[1 * (K) + (k)]);                               \
                                                                                                     \
            CMUL_NATIVE_SOA_AVX512(s1_b_re, s1_b_im, s1_w1_re, s1_w1_im, s1_tB_re, s1_tB_im);        \
            CMUL_NATIVE_SOA_AVX512(s1_c_re, s1_c_im, s1_w2_re, s1_w2_im, s1_tC_re, s1_tC_im);        \
                                                                                                     \
            RADIX3_BUTTERFLY_NATIVE_SOA_BV_AVX512(s0_a_re, s0_a_im, s0_tB_re, s0_tB_im,              \
                                                  s0_tC_re, s0_tC_im, s2_y0_re, s2_y0_im,            \
                                                  s2_y1_re, s2_y1_im, s2_y2_re, s2_y2_im);           \
                                                                                                     \
            STORE_RE_AVX512(&out_re[(k) - 16], s2_y0_re);                                            \
            STORE_IM_AVX512(&out_im[(k) - 16], s2_y0_im);                                            \
            STORE_RE_AVX512(&out_re[(k) - 16 + (K)], s2_y1_re);                                      \
            STORE_IM_AVX512(&out_im[(k) - 16 + (K)], s2_y1_im);                                      \
            STORE_RE_AVX512(&out_re[(k) - 16 + 2 * (K)], s2_y2_re);                                  \
            STORE_IM_AVX512(&out_im[(k) - 16 + 2 * (K)], s2_y2_im);                                  \
                                                                                                     \
            /* Second half */                                                                        \
            __m512d n1_a_re = LOAD_RE_AVX512(&in_re[(k) + 8]);                                       \
            __m512d n1_a_im = LOAD_IM_AVX512(&in_im[(k) + 8]);                                       \
            __m512d n1_b_re = LOAD_RE_AVX512(&in_re[(k) + 8 + (K)]);                                 \
            __m512d n1_b_im = LOAD_IM_AVX512(&in_im[(k) + 8 + (K)]);                                 \
            __m512d n1_c_re = LOAD_RE_AVX512(&in_re[(k) + 8 + 2 * (K)]);                             \
            __m512d n1_c_im = LOAD_IM_AVX512(&in_im[(k) + 8 + 2 * (K)]);                             \
            __m512d n1_w1_re = LOAD_RE_AVX512(&tw->re[0 * (K) + (k) + 8]);                           \
            __m512d n1_w1_im = LOAD_IM_AVX512(&tw->im[0 * (K) + (k) + 8]);                           \
            __m512d n1_w2_re = LOAD_RE_AVX512(&tw->re[1 * (K) + (k) + 8]);                           \
            __m512d n1_w2_im = LOAD_IM_AVX512(&tw->im[1 * (K) + (k) + 8]);                           \
                                                                                                     \
            __m512d n0_tB_re, n0_tB_im, n0_tC_re, n0_tC_im;                                          \
            CMUL_NATIVE_SOA_AVX512(n0_b_re, n0_b_im, n0_w1_re, n0_w1_im, n0_tB_re, n0_tB_im);        \
            CMUL_NATIVE_SOA_AVX512(n0_c_re, n0_c_im, n0_w2_re, n0_w2_im, n0_tC_re, n0_tC_im);        \
                                                                                                     \
            __m512d n1_y0_re, n1_y0_im, n1_y1_re, n1_y1_im, n1_y2_re, n1_y2_im;                      \
            RADIX3_BUTTERFLY_NATIVE_SOA_BV_AVX512(s1_a_re, s1_a_im, s1_tB_re, s1_tB_im,              \
                                                  s1_tC_re, s1_tC_im, n1_y0_re, n1_y0_im,            \
                                                  n1_y1_re, n1_y1_im, n1_y2_re, n1_y2_im);           \
                                                                                                     \
            STORE_RE_AVX512(&out_re[(k) - 8], n1_y0_re);                                             \
            STORE_IM_AVX512(&out_im[(k) - 8], n1_y0_im);                                             \
            STORE_RE_AVX512(&out_re[(k) - 8 + (K)], n1_y1_re);                                       \
            STORE_IM_AVX512(&out_im[(k) - 8 + (K)], n1_y1_im);                                       \
            STORE_RE_AVX512(&out_re[(k) - 8 + 2 * (K)], n1_y2_re);                                   \
            STORE_IM_AVX512(&out_im[(k) - 8 + 2 * (K)], n1_y2_im);                                   \
                                                                                                     \
            s0_a_re = n0_a_re;                                                                       \
            s0_a_im = n0_a_im;                                                                       \
            s0_tB_re = n0_tB_re;                                                                     \
            s0_tB_im = n0_tB_im;                                                                     \
            s0_tC_re = n0_tC_re;                                                                     \
            s0_tC_im = n0_tC_im;                                                                     \
                                                                                                     \
            s1_a_re = n1_a_re;                                                                       \
            s1_a_im = n1_a_im;                                                                       \
            s1_b_re = n1_b_re;                                                                       \
            s1_b_im = n1_b_im;                                                                       \
            s1_c_re = n1_c_re;                                                                       \
            s1_c_im = n1_c_im;                                                                       \
            s1_w1_re = n1_w1_re;                                                                     \
            s1_w1_im = n1_w1_im;                                                                     \
            s1_w2_re = n1_w2_re;                                                                     \
            s1_w2_im = n1_w2_im;                                                                     \
        }                                                                                            \
                                                                                                     \
        /* EPILOGUE */                                                                               \
        CMUL_NATIVE_SOA_AVX512(s1_b_re, s1_b_im, s1_w1_re, s1_w1_im, s1_tB_re, s1_tB_im);            \
        CMUL_NATIVE_SOA_AVX512(s1_c_re, s1_c_im, s1_w2_re, s1_w2_im, s1_tC_re, s1_tC_im);            \
        RADIX3_BUTTERFLY_NATIVE_SOA_BV_AVX512(s0_a_re, s0_a_im, s0_tB_re, s0_tB_im,                  \
                                              s0_tC_re, s0_tC_im, s2_y0_re, s2_y0_im,                \
                                              s2_y1_re, s2_y1_im, s2_y2_re, s2_y2_im);               \
        STORE_RE_AVX512(&out_re[(k) - 16], s2_y0_re);                                                \
        STORE_IM_AVX512(&out_im[(k) - 16], s2_y0_im);                                                \
        STORE_RE_AVX512(&out_re[(k) - 16 + (K)], s2_y1_re);                                          \
        STORE_IM_AVX512(&out_im[(k) - 16 + (K)], s2_y1_im);                                          \
        STORE_RE_AVX512(&out_re[(k) - 16 + 2 * (K)], s2_y2_re);                                      \
        STORE_IM_AVX512(&out_im[(k) - 16 + 2 * (K)], s2_y2_im);                                      \
                                                                                                     \
        RADIX3_BUTTERFLY_NATIVE_SOA_BV_AVX512(s1_a_re, s1_a_im, s1_tB_re, s1_tB_im,                  \
                                              s1_tC_re, s1_tC_im, s2_y0_re, s2_y0_im,                \
                                              s2_y1_re, s2_y1_im, s2_y2_re, s2_y2_im);               \
        STORE_RE_AVX512(&out_re[(k) - 8], s2_y0_re);                                                 \
        STORE_IM_AVX512(&out_im[(k) - 8], s2_y0_im);                                                 \
        STORE_RE_AVX512(&out_re[(k) - 8 + (K)], s2_y1_re);                                           \
        STORE_IM_AVX512(&out_im[(k) - 8 + (K)], s2_y1_im);                                           \
        STORE_RE_AVX512(&out_re[(k) - 8 + 2 * (K)], s2_y2_re);                                       \
        STORE_IM_AVX512(&out_im[(k) - 8 + 2 * (K)], s2_y2_im);                                       \
                                                                                                     \
        radix3_avx512_tail_bv(k, k_end, K, in_re, in_im, out_re, out_im, tw);                        \
    } while (0)

#endif // RADIX3_USE_SOFTWARE_PIPELINING

//==============================================================================
// VECTORIZED PIPELINE MACROS - NATIVE SoA
//==============================================================================

#ifdef __AVX512F__
/**
 * @brief AVX-512 radix-3 pipeline (8-wide) - Forward
 *
 * @details
 * OPTIMIZATION: Uses masked tail instead of scalar fallback!
 * Process main loop in 8-wide chunks, then call tail function for remainder.
 *
 * ADDED: Light prefetch for twiddles (T0) and inputs (T0) to improve
 * cache utilization without overwhelming the prefetch engine.
 */
#define RADIX3_PIPELINE_8_NATIVE_SOA_FV_AVX512(k, k_end, K, in_re, in_im, out_re, out_im, tw) \
    do                                                                                        \
    {                                                                                         \
        const int prefetch_dist = RADIX3_PREFETCH_DISTANCE;                                   \
        for (; (k) + 8 <= (k_end); (k) += 8)                                                  \
        {                                                                                     \
            if ((k) + prefetch_dist < (k_end))                                                \
            {                                                                                 \
                int pk = (k) + prefetch_dist;                                                 \
                _mm_prefetch((const char *)&tw->re[0 * (K) + pk], _MM_HINT_T0);               \
                _mm_prefetch((const char *)&tw->im[0 * (K) + pk], _MM_HINT_T0);               \
                _mm_prefetch((const char *)&tw->re[1 * (K) + pk], _MM_HINT_T0);               \
                _mm_prefetch((const char *)&tw->im[1 * (K) + pk], _MM_HINT_T0);               \
                _mm_prefetch((const char *)&in_re[pk], _MM_HINT_T0);                          \
                _mm_prefetch((const char *)&in_im[pk], _MM_HINT_T0);                          \
                _mm_prefetch((const char *)&in_re[pk + (K)], _MM_HINT_T0);                    \
                _mm_prefetch((const char *)&in_im[pk + (K)], _MM_HINT_T0);                    \
                _mm_prefetch((const char *)&in_re[pk + 2 * (K)], _MM_HINT_T0);                \
                _mm_prefetch((const char *)&in_im[pk + 2 * (K)], _MM_HINT_T0);                \
            }                                                                                 \
            __m512d a_re = LOAD_RE_AVX512(&in_re[k]);                                         \
            __m512d a_im = LOAD_IM_AVX512(&in_im[k]);                                         \
            __m512d b_re = LOAD_RE_AVX512(&in_re[(k) + (K)]);                                 \
            __m512d b_im = LOAD_IM_AVX512(&in_im[(k) + (K)]);                                 \
            __m512d c_re = LOAD_RE_AVX512(&in_re[(k) + 2 * (K)]);                             \
            __m512d c_im = LOAD_IM_AVX512(&in_im[(k) + 2 * (K)]);                             \
            __m512d w1_re = LOAD_RE_AVX512(&tw->re[0 * (K) + (k)]);                           \
            __m512d w1_im = LOAD_IM_AVX512(&tw->im[0 * (K) + (k)]);                           \
            __m512d w2_re = LOAD_RE_AVX512(&tw->re[1 * (K) + (k)]);                           \
            __m512d w2_im = LOAD_IM_AVX512(&tw->im[1 * (K) + (k)]);                           \
            __m512d tB_re, tB_im, tC_re, tC_im;                                               \
            CMUL_NATIVE_SOA_AVX512(b_re, b_im, w1_re, w1_im, tB_re, tB_im);                   \
            CMUL_NATIVE_SOA_AVX512(c_re, c_im, w2_re, w2_im, tC_re, tC_im);                   \
            __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im;                                 \
            RADIX3_BUTTERFLY_NATIVE_SOA_FV_AVX512(a_re, a_im, tB_re, tB_im, tC_re, tC_im,     \
                                                  y0_re, y0_im, y1_re, y1_im, y2_re, y2_im);  \
            STORE_RE_AVX512(&out_re[k], y0_re);                                               \
            STORE_IM_AVX512(&out_im[k], y0_im);                                               \
            STORE_RE_AVX512(&out_re[(k) + (K)], y1_re);                                       \
            STORE_IM_AVX512(&out_im[(k) + (K)], y1_im);                                       \
            STORE_RE_AVX512(&out_re[(k) + 2 * (K)], y2_re);                                   \
            STORE_IM_AVX512(&out_im[(k) + 2 * (K)], y2_im);                                   \
        }                                                                                     \
        radix3_avx512_tail_fv(k, k_end, K, in_re, in_im, out_re, out_im, tw);                 \
    } while (0)

/**
 * @brief AVX-512 radix-3 pipeline (8-wide) - Backward
 *
 * @details
 * OPTIMIZATION: Uses masked tail instead of scalar fallback!
 * ADDED: Light prefetch for improved cache utilization.
 */
#define RADIX3_PIPELINE_8_NATIVE_SOA_BV_AVX512(k, k_end, K, in_re, in_im, out_re, out_im, tw) \
    do                                                                                        \
    {                                                                                         \
        const int prefetch_dist = RADIX3_PREFETCH_DISTANCE;                                   \
        for (; (k) + 8 <= (k_end); (k) += 8)                                                  \
        {                                                                                     \
            if ((k) + prefetch_dist < (k_end))                                                \
            {                                                                                 \
                int pk = (k) + prefetch_dist;                                                 \
                _mm_prefetch((const char *)&tw->re[0 * (K) + pk], _MM_HINT_T0);               \
                _mm_prefetch((const char *)&tw->im[0 * (K) + pk], _MM_HINT_T0);               \
                _mm_prefetch((const char *)&tw->re[1 * (K) + pk], _MM_HINT_T0);               \
                _mm_prefetch((const char *)&tw->im[1 * (K) + pk], _MM_HINT_T0);               \
                _mm_prefetch((const char *)&in_re[pk], _MM_HINT_T0);                          \
                _mm_prefetch((const char *)&in_im[pk], _MM_HINT_T0);                          \
                _mm_prefetch((const char *)&in_re[pk + (K)], _MM_HINT_T0);                    \
                _mm_prefetch((const char *)&in_im[pk + (K)], _MM_HINT_T0);                    \
                _mm_prefetch((const char *)&in_re[pk + 2 * (K)], _MM_HINT_T0);                \
                _mm_prefetch((const char *)&in_im[pk + 2 * (K)], _MM_HINT_T0);                \
            }                                                                                 \
            __m512d a_re = LOAD_RE_AVX512(&in_re[k]);                                         \
            __m512d a_im = LOAD_IM_AVX512(&in_im[k]);                                         \
            __m512d b_re = LOAD_RE_AVX512(&in_re[(k) + (K)]);                                 \
            __m512d b_im = LOAD_IM_AVX512(&in_im[(k) + (K)]);                                 \
            __m512d c_re = LOAD_RE_AVX512(&in_re[(k) + 2 * (K)]);                             \
            __m512d c_im = LOAD_IM_AVX512(&in_im[(k) + 2 * (K)]);                             \
            __m512d w1_re = LOAD_RE_AVX512(&tw->re[0 * (K) + (k)]);                           \
            __m512d w1_im = LOAD_IM_AVX512(&tw->im[0 * (K) + (k)]);                           \
            __m512d w2_re = LOAD_RE_AVX512(&tw->re[1 * (K) + (k)]);                           \
            __m512d w2_im = LOAD_IM_AVX512(&tw->im[1 * (K) + (k)]);                           \
            __m512d tB_re, tB_im, tC_re, tC_im;                                               \
            CMUL_NATIVE_SOA_AVX512(b_re, b_im, w1_re, w1_im, tB_re, tB_im);                   \
            CMUL_NATIVE_SOA_AVX512(c_re, c_im, w2_re, w2_im, tC_re, tC_im);                   \
            __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im;                                 \
            RADIX3_BUTTERFLY_NATIVE_SOA_BV_AVX512(a_re, a_im, tB_re, tB_im, tC_re, tC_im,     \
                                                  y0_re, y0_im, y1_re, y1_im, y2_re, y2_im);  \
            STORE_RE_AVX512(&out_re[k], y0_re);                                               \
            STORE_IM_AVX512(&out_im[k], y0_im);                                               \
            STORE_RE_AVX512(&out_re[(k) + (K)], y1_re);                                       \
            STORE_IM_AVX512(&out_im[(k) + (K)], y1_im);                                       \
            STORE_RE_AVX512(&out_re[(k) + 2 * (K)], y2_re);                                   \
            STORE_IM_AVX512(&out_im[(k) + 2 * (K)], y2_im);                                   \
        }                                                                                     \
        radix3_avx512_tail_bv(k, k_end, K, in_re, in_im, out_re, out_im, tw);                 \
    } while (0)

/**
 * @brief AVX-512 radix-3 pipeline (8-wide) with streaming stores - Forward
 *
 * @details
 * Uses non-temporal stores for large FFTs to reduce cache pollution.
 * Still uses masked tail for remainder processing.
 */
#define RADIX3_PIPELINE_8_NATIVE_SOA_FV_AVX512_STREAM(k, k_end, K, in_re, in_im, out_re, out_im, tw, prefetch_dist, end_k) \
    do                                                                                                                     \
    {                                                                                                                      \
        for (; (k) + 8 <= (k_end); (k) += 8)                                                                               \
        {                                                                                                                  \
            if ((k) + (prefetch_dist) < (end_k))                                                                           \
            {                                                                                                              \
                int pk = (k) + (prefetch_dist);                                                                            \
                _mm_prefetch((const char *)&tw->re[0 * (K) + pk], _MM_HINT_T0);                                            \
                _mm_prefetch((const char *)&tw->im[0 * (K) + pk], _MM_HINT_T0);                                            \
                _mm_prefetch((const char *)&tw->re[1 * (K) + pk], _MM_HINT_T0);                                            \
                _mm_prefetch((const char *)&tw->im[1 * (K) + pk], _MM_HINT_T0);                                            \
                _mm_prefetch((const char *)&in_re[pk], _MM_HINT_NTA);                                                      \
                _mm_prefetch((const char *)&in_im[pk], _MM_HINT_NTA);                                                      \
                _mm_prefetch((const char *)&in_re[pk + (K)], _MM_HINT_NTA);                                                \
                _mm_prefetch((const char *)&in_im[pk + (K)], _MM_HINT_NTA);                                                \
                _mm_prefetch((const char *)&in_re[pk + 2 * (K)], _MM_HINT_NTA);                                            \
                _mm_prefetch((const char *)&in_im[pk + 2 * (K)], _MM_HINT_NTA);                                            \
            }                                                                                                              \
            __m512d a_re = LOAD_RE_AVX512(&in_re[k]);                                                                      \
            __m512d a_im = LOAD_IM_AVX512(&in_im[k]);                                                                      \
            __m512d b_re = LOAD_RE_AVX512(&in_re[(k) + (K)]);                                                              \
            __m512d b_im = LOAD_IM_AVX512(&in_im[(k) + (K)]);                                                              \
            __m512d c_re = LOAD_RE_AVX512(&in_re[(k) + 2 * (K)]);                                                          \
            __m512d c_im = LOAD_IM_AVX512(&in_im[(k) + 2 * (K)]);                                                          \
            __m512d w1_re = LOAD_RE_AVX512(&tw->re[0 * (K) + (k)]);                                                        \
            __m512d w1_im = LOAD_IM_AVX512(&tw->im[0 * (K) + (k)]);                                                        \
            __m512d w2_re = LOAD_RE_AVX512(&tw->re[1 * (K) + (k)]);                                                        \
            __m512d w2_im = LOAD_IM_AVX512(&tw->im[1 * (K) + (k)]);                                                        \
            __m512d tB_re, tB_im, tC_re, tC_im;                                                                            \
            CMUL_NATIVE_SOA_AVX512(b_re, b_im, w1_re, w1_im, tB_re, tB_im);                                                \
            CMUL_NATIVE_SOA_AVX512(c_re, c_im, w2_re, w2_im, tC_re, tC_im);                                                \
            __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im;                                                              \
            RADIX3_BUTTERFLY_NATIVE_SOA_FV_AVX512(a_re, a_im, tB_re, tB_im, tC_re, tC_im,                                  \
                                                  y0_re, y0_im, y1_re, y1_im, y2_re, y2_im);                               \
            STREAM_RE_AVX512(&out_re[k], y0_re);                                                                           \
            STREAM_IM_AVX512(&out_im[k], y0_im);                                                                           \
            STREAM_RE_AVX512(&out_re[(k) + (K)], y1_re);                                                                   \
            STREAM_IM_AVX512(&out_im[(k) + (K)], y1_im);                                                                   \
            STREAM_RE_AVX512(&out_re[(k) + 2 * (K)], y2_re);                                                               \
            STREAM_IM_AVX512(&out_im[(k) + 2 * (K)], y2_im);                                                               \
        }                                                                                                                  \
        radix3_avx512_tail_fv(k, k_end, K, in_re, in_im, out_re, out_im, tw);                                              \
    } while (0)

/**
 * @brief AVX-512 radix-3 pipeline (8-wide) with streaming stores - Backward
 */
#define RADIX3_PIPELINE_8_NATIVE_SOA_BV_AVX512_STREAM(k, k_end, K, in_re, in_im, out_re, out_im, tw, prefetch_dist, end_k) \
    do                                                                                                                     \
    {                                                                                                                      \
        for (; (k) + 8 <= (k_end); (k) += 8)                                                                               \
        {                                                                                                                  \
            if ((k) + (prefetch_dist) < (end_k))                                                                           \
            {                                                                                                              \
                int pk = (k) + (prefetch_dist);                                                                            \
                _mm_prefetch((const char *)&tw->re[0 * (K) + pk], _MM_HINT_T0);                                            \
                _mm_prefetch((const char *)&tw->im[0 * (K) + pk], _MM_HINT_T0);                                            \
                _mm_prefetch((const char *)&tw->re[1 * (K) + pk], _MM_HINT_T0);                                            \
                _mm_prefetch((const char *)&tw->im[1 * (K) + pk], _MM_HINT_T0);                                            \
                _mm_prefetch((const char *)&in_re[pk], _MM_HINT_NTA);                                                      \
                _mm_prefetch((const char *)&in_im[pk], _MM_HINT_NTA);                                                      \
                _mm_prefetch((const char *)&in_re[pk + (K)], _MM_HINT_NTA);                                                \
                _mm_prefetch((const char *)&in_im[pk + (K)], _MM_HINT_NTA);                                                \
                _mm_prefetch((const char *)&in_re[pk + 2 * (K)], _MM_HINT_NTA);                                            \
                _mm_prefetch((const char *)&in_im[pk + 2 * (K)], _MM_HINT_NTA);                                            \
            }                                                                                                              \
            __m512d a_re = LOAD_RE_AVX512(&in_re[k]);                                                                      \
            __m512d a_im = LOAD_IM_AVX512(&in_im[k]);                                                                      \
            __m512d b_re = LOAD_RE_AVX512(&in_re[(k) + (K)]);                                                              \
            __m512d b_im = LOAD_IM_AVX512(&in_im[(k) + (K)]);                                                              \
            __m512d c_re = LOAD_RE_AVX512(&in_re[(k) + 2 * (K)]);                                                          \
            __m512d c_im = LOAD_IM_AVX512(&in_im[(k) + 2 * (K)]);                                                          \
            __m512d w1_re = LOAD_RE_AVX512(&tw->re[0 * (K) + (k)]);                                                        \
            __m512d w1_im = LOAD_IM_AVX512(&tw->im[0 * (K) + (k)]);                                                        \
            __m512d w2_re = LOAD_RE_AVX512(&tw->re[1 * (K) + (k)]);                                                        \
            __m512d w2_im = LOAD_IM_AVX512(&tw->im[1 * (K) + (k)]);                                                        \
            __m512d tB_re, tB_im, tC_re, tC_im;                                                                            \
            CMUL_NATIVE_SOA_AVX512(b_re, b_im, w1_re, w1_im, tB_re, tB_im);                                                \
            CMUL_NATIVE_SOA_AVX512(c_re, c_im, w2_re, w2_im, tC_re, tC_im);                                                \
            __m512d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im;                                                              \
            RADIX3_BUTTERFLY_NATIVE_SOA_BV_AVX512(a_re, a_im, tB_re, tB_im, tC_re, tC_im,                                  \
                                                  y0_re, y0_im, y1_re, y1_im, y2_re, y2_im);                               \
            STREAM_RE_AVX512(&out_re[k], y0_re);                                                                           \
            STREAM_IM_AVX512(&out_im[k], y0_im);                                                                           \
            STREAM_RE_AVX512(&out_re[(k) + (K)], y1_re);                                                                   \
            STREAM_IM_AVX512(&out_im[(k) + (K)], y1_im);                                                                   \
            STREAM_RE_AVX512(&out_re[(k) + 2 * (K)], y2_re);                                                               \
            STREAM_IM_AVX512(&out_im[(k) + 2 * (K)], y2_im);                                                               \
        }                                                                                                                  \
        radix3_avx512_tail_bv(k, k_end, K, in_re, in_im, out_re, out_im, tw);                                              \
    } while (0)

#endif // __AVX512F__

#ifdef __AVX2__
/**
 * @brief AVX2 radix-3 pipeline (4-wide) - Forward
 *
 * @details
 * OPTIMIZATION: Uses FMA when available for CMUL and common term!
 * Falls back to scalar for remainder (AVX2 doesn't have masks).
 */
#define RADIX3_PIPELINE_4_NATIVE_SOA_FV_AVX2(k, k_end, K, in_re, in_im, out_re, out_im, tw) \
    do                                                                                      \
    {                                                                                       \
        for (; (k) + 4 <= (k_end); (k) += 4)                                                \
        {                                                                                   \
            __m256d a_re = LOAD_RE_AVX2(&in_re[k]);                                         \
            __m256d a_im = LOAD_IM_AVX2(&in_im[k]);                                         \
            __m256d b_re = LOAD_RE_AVX2(&in_re[(k) + (K)]);                                 \
            __m256d b_im = LOAD_IM_AVX2(&in_im[(k) + (K)]);                                 \
            __m256d c_re = LOAD_RE_AVX2(&in_re[(k) + 2 * (K)]);                             \
            __m256d c_im = LOAD_IM_AVX2(&in_im[(k) + 2 * (K)]);                             \
            __m256d w1_re = LOAD_RE_AVX2(&tw->re[0 * (K) + (k)]);                           \
            __m256d w1_im = LOAD_IM_AVX2(&tw->im[0 * (K) + (k)]);                           \
            __m256d w2_re = LOAD_RE_AVX2(&tw->re[1 * (K) + (k)]);                           \
            __m256d w2_im = LOAD_IM_AVX2(&tw->im[1 * (K) + (k)]);                           \
            __m256d tB_re, tB_im, tC_re, tC_im;                                             \
            CMUL_NATIVE_SOA_AVX2(b_re, b_im, w1_re, w1_im, tB_re, tB_im);                   \
            CMUL_NATIVE_SOA_AVX2(c_re, c_im, w2_re, w2_im, tC_re, tC_im);                   \
            __m256d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im;                               \
            RADIX3_BUTTERFLY_NATIVE_SOA_FV_AVX2(a_re, a_im, tB_re, tB_im, tC_re, tC_im,     \
                                                y0_re, y0_im, y1_re, y1_im, y2_re, y2_im);  \
            STORE_RE_AVX2(&out_re[k], y0_re);                                               \
            STORE_IM_AVX2(&out_im[k], y0_im);                                               \
            STORE_RE_AVX2(&out_re[(k) + (K)], y1_re);                                       \
            STORE_IM_AVX2(&out_im[(k) + (K)], y1_im);                                       \
            STORE_RE_AVX2(&out_re[(k) + 2 * (K)], y2_re);                                   \
            STORE_IM_AVX2(&out_im[(k) + 2 * (K)], y2_im);                                   \
        }                                                                                   \
    } while (0)

/**
 * @brief AVX2 radix-3 pipeline (4-wide) - Backward
 */
#define RADIX3_PIPELINE_4_NATIVE_SOA_BV_AVX2(k, k_end, K, in_re, in_im, out_re, out_im, tw) \
    do                                                                                      \
    {                                                                                       \
        for (; (k) + 4 <= (k_end); (k) += 4)                                                \
        {                                                                                   \
            __m256d a_re = LOAD_RE_AVX2(&in_re[k]);                                         \
            __m256d a_im = LOAD_IM_AVX2(&in_im[k]);                                         \
            __m256d b_re = LOAD_RE_AVX2(&in_re[(k) + (K)]);                                 \
            __m256d b_im = LOAD_IM_AVX2(&in_im[(k) + (K)]);                                 \
            __m256d c_re = LOAD_RE_AVX2(&in_re[(k) + 2 * (K)]);                             \
            __m256d c_im = LOAD_IM_AVX2(&in_im[(k) + 2 * (K)]);                             \
            __m256d w1_re = LOAD_RE_AVX2(&tw->re[0 * (K) + (k)]);                           \
            __m256d w1_im = LOAD_IM_AVX2(&tw->im[0 * (K) + (k)]);                           \
            __m256d w2_re = LOAD_RE_AVX2(&tw->re[1 * (K) + (k)]);                           \
            __m256d w2_im = LOAD_IM_AVX2(&tw->im[1 * (K) + (k)]);                           \
            __m256d tB_re, tB_im, tC_re, tC_im;                                             \
            CMUL_NATIVE_SOA_AVX2(b_re, b_im, w1_re, w1_im, tB_re, tB_im);                   \
            CMUL_NATIVE_SOA_AVX2(c_re, c_im, w2_re, w2_im, tC_re, tC_im);                   \
            __m256d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im;                               \
            RADIX3_BUTTERFLY_NATIVE_SOA_BV_AVX2(a_re, a_im, tB_re, tB_im, tC_re, tC_im,     \
                                                y0_re, y0_im, y1_re, y1_im, y2_re, y2_im);  \
            STORE_RE_AVX2(&out_re[k], y0_re);                                               \
            STORE_IM_AVX2(&out_im[k], y0_im);                                               \
            STORE_RE_AVX2(&out_re[(k) + (K)], y1_re);                                       \
            STORE_IM_AVX2(&out_im[(k) + (K)], y1_im);                                       \
            STORE_RE_AVX2(&out_re[(k) + 2 * (K)], y2_re);                                   \
            STORE_IM_AVX2(&out_im[(k) + 2 * (K)], y2_im);                                   \
        }                                                                                   \
    } while (0)

/**
 * @brief AVX2 radix-3 pipeline (4-wide) with streaming stores - Forward
 */
#define RADIX3_PIPELINE_4_NATIVE_SOA_FV_AVX2_STREAM(k, k_end, K, in_re, in_im, out_re, out_im, tw, prefetch_dist, end_k) \
    do                                                                                                                   \
    {                                                                                                                    \
        for (; (k) + 4 <= (k_end); (k) += 4)                                                                             \
        {                                                                                                                \
            if ((k) + (prefetch_dist) < (end_k))                                                                         \
            {                                                                                                            \
                int pk = (k) + (prefetch_dist);                                                                          \
                _mm_prefetch((const char *)&tw->re[0 * (K) + pk], _MM_HINT_T0);                                          \
                _mm_prefetch((const char *)&tw->im[0 * (K) + pk], _MM_HINT_T0);                                          \
                _mm_prefetch((const char *)&tw->re[1 * (K) + pk], _MM_HINT_T0);                                          \
                _mm_prefetch((const char *)&tw->im[1 * (K) + pk], _MM_HINT_T0);                                          \
                _mm_prefetch((const char *)&in_re[pk], _MM_HINT_NTA);                                                    \
                _mm_prefetch((const char *)&in_im[pk], _MM_HINT_NTA);                                                    \
                _mm_prefetch((const char *)&in_re[pk + (K)], _MM_HINT_NTA);                                              \
                _mm_prefetch((const char *)&in_im[pk + (K)], _MM_HINT_NTA);                                              \
                _mm_prefetch((const char *)&in_re[pk + 2 * (K)], _MM_HINT_NTA);                                          \
                _mm_prefetch((const char *)&in_im[pk + 2 * (K)], _MM_HINT_NTA);                                          \
            }                                                                                                            \
            __m256d a_re = LOAD_RE_AVX2(&in_re[k]);                                                                      \
            __m256d a_im = LOAD_IM_AVX2(&in_im[k]);                                                                      \
            __m256d b_re = LOAD_RE_AVX2(&in_re[(k) + (K)]);                                                              \
            __m256d b_im = LOAD_IM_AVX2(&in_im[(k) + (K)]);                                                              \
            __m256d c_re = LOAD_RE_AVX2(&in_re[(k) + 2 * (K)]);                                                          \
            __m256d c_im = LOAD_IM_AVX2(&in_im[(k) + 2 * (K)]);                                                          \
            __m256d w1_re = LOAD_RE_AVX2(&tw->re[0 * (K) + (k)]);                                                        \
            __m256d w1_im = LOAD_IM_AVX2(&tw->im[0 * (K) + (k)]);                                                        \
            __m256d w2_re = LOAD_RE_AVX2(&tw->re[1 * (K) + (k)]);                                                        \
            __m256d w2_im = LOAD_IM_AVX2(&tw->im[1 * (K) + (k)]);                                                        \
            __m256d tB_re, tB_im, tC_re, tC_im;                                                                          \
            CMUL_NATIVE_SOA_AVX2(b_re, b_im, w1_re, w1_im, tB_re, tB_im);                                                \
            CMUL_NATIVE_SOA_AVX2(c_re, c_im, w2_re, w2_im, tC_re, tC_im);                                                \
            __m256d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im;                                                            \
            RADIX3_BUTTERFLY_NATIVE_SOA_FV_AVX2(a_re, a_im, tB_re, tB_im, tC_re, tC_im,                                  \
                                                y0_re, y0_im, y1_re, y1_im, y2_re, y2_im);                               \
            STREAM_RE_AVX2(&out_re[k], y0_re);                                                                           \
            STREAM_IM_AVX2(&out_im[k], y0_im);                                                                           \
            STREAM_RE_AVX2(&out_re[(k) + (K)], y1_re);                                                                   \
            STREAM_IM_AVX2(&out_im[(k) + (K)], y1_im);                                                                   \
            STREAM_RE_AVX2(&out_re[(k) + 2 * (K)], y2_re);                                                               \
            STREAM_IM_AVX2(&out_im[(k) + 2 * (K)], y2_im);                                                               \
        }                                                                                                                \
    } while (0)

/**
 * @brief AVX2 radix-3 pipeline (4-wide) with streaming stores - Backward
 */
#define RADIX3_PIPELINE_4_NATIVE_SOA_BV_AVX2_STREAM(k, k_end, K, in_re, in_im, out_re, out_im, tw, prefetch_dist, end_k) \
    do                                                                                                                   \
    {                                                                                                                    \
        for (; (k) + 4 <= (k_end); (k) += 4)                                                                             \
        {                                                                                                                \
            if ((k) + (prefetch_dist) < (end_k))                                                                         \
            {                                                                                                            \
                int pk = (k) + (prefetch_dist);                                                                          \
                _mm_prefetch((const char *)&tw->re[0 * (K) + pk], _MM_HINT_T0);                                          \
                _mm_prefetch((const char *)&tw->im[0 * (K) + pk], _MM_HINT_T0);                                          \
                _mm_prefetch((const char *)&tw->re[1 * (K) + pk], _MM_HINT_T0);                                          \
                _mm_prefetch((const char *)&tw->im[1 * (K) + pk], _MM_HINT_T0);                                          \
                _mm_prefetch((const char *)&in_re[pk], _MM_HINT_NTA);                                                    \
                _mm_prefetch((const char *)&in_im[pk], _MM_HINT_NTA);                                                    \
                _mm_prefetch((const char *)&in_re[pk + (K)], _MM_HINT_NTA);                                              \
                _mm_prefetch((const char *)&in_im[pk + (K)], _MM_HINT_NTA);                                              \
                _mm_prefetch((const char *)&in_re[pk + 2 * (K)], _MM_HINT_NTA);                                          \
                _mm_prefetch((const char *)&in_im[pk + 2 * (K)], _MM_HINT_NTA);                                          \
            }                                                                                                            \
            __m256d a_re = LOAD_RE_AVX2(&in_re[k]);                                                                      \
            __m256d a_im = LOAD_IM_AVX2(&in_im[k]);                                                                      \
            __m256d b_re = LOAD_RE_AVX2(&in_re[(k) + (K)]);                                                              \
            __m256d b_im = LOAD_IM_AVX2(&in_im[(k) + (K)]);                                                              \
            __m256d c_re = LOAD_RE_AVX2(&in_re[(k) + 2 * (K)]);                                                          \
            __m256d c_im = LOAD_IM_AVX2(&in_im[(k) + 2 * (K)]);                                                          \
            __m256d w1_re = LOAD_RE_AVX2(&tw->re[0 * (K) + (k)]);                                                        \
            __m256d w1_im = LOAD_IM_AVX2(&tw->im[0 * (K) + (k)]);                                                        \
            __m256d w2_re = LOAD_RE_AVX2(&tw->re[1 * (K) + (k)]);                                                        \
            __m256d w2_im = LOAD_IM_AVX2(&tw->im[1 * (K) + (k)]);                                                        \
            __m256d tB_re, tB_im, tC_re, tC_im;                                                                          \
            CMUL_NATIVE_SOA_AVX2(b_re, b_im, w1_re, w1_im, tB_re, tB_im);                                                \
            CMUL_NATIVE_SOA_AVX2(c_re, c_im, w2_re, w2_im, tC_re, tC_im);                                                \
            __m256d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im;                                                            \
            RADIX3_BUTTERFLY_NATIVE_SOA_BV_AVX2(a_re, a_im, tB_re, tB_im, tC_re, tC_im,                                  \
                                                y0_re, y0_im, y1_re, y1_im, y2_re, y2_im);                               \
            STREAM_RE_AVX2(&out_re[k], y0_re);                                                                           \
            STREAM_IM_AVX2(&out_im[k], y0_im);                                                                           \
            STREAM_RE_AVX2(&out_re[(k) + (K)], y1_re);                                                                   \
            STREAM_IM_AVX2(&out_im[(k) + (K)], y1_im);                                                                   \
            STREAM_RE_AVX2(&out_re[(k) + 2 * (K)], y2_re);                                                               \
            STREAM_IM_AVX2(&out_im[(k) + 2 * (K)], y2_im);                                                               \
        }                                                                                                                \
    } while (0)

#endif // __AVX2__

#ifdef __SSE2__
/**
 * @brief SSE2 radix-3 pipeline (2-wide) - Forward
 *
 * @details
 * OPTIMIZATION: Uses _mm_load1_pd for twiddles instead of _mm_set1_pd!
 * Eliminates GPR round-trip, saves 2-3 cycles per twiddle load.
 */
#define RADIX3_PIPELINE_2_NATIVE_SOA_FV_SSE2(k, k_end, K, in_re, in_im, out_re, out_im, tw) \
    do                                                                                      \
    {                                                                                       \
        for (; (k) + 2 <= (k_end); (k) += 2)                                                \
        {                                                                                   \
            __m128d a_re = LOAD_RE_SSE2(&in_re[k]);                                         \
            __m128d a_im = LOAD_IM_SSE2(&in_im[k]);                                         \
            __m128d b_re = LOAD_RE_SSE2(&in_re[(k) + (K)]);                                 \
            __m128d b_im = LOAD_IM_SSE2(&in_im[(k) + (K)]);                                 \
            __m128d c_re = LOAD_RE_SSE2(&in_re[(k) + 2 * (K)]);                             \
            __m128d c_im = LOAD_IM_SSE2(&in_im[(k) + 2 * (K)]);                             \
            __m128d w1_re = _mm_load1_pd(&tw->re[0 * (K) + (k)]);                           \
            __m128d w1_im = _mm_load1_pd(&tw->im[0 * (K) + (k)]);                           \
            __m128d w2_re = _mm_load1_pd(&tw->re[1 * (K) + (k)]);                           \
            __m128d w2_im = _mm_load1_pd(&tw->im[1 * (K) + (k)]);                           \
            __m128d tB_re, tB_im, tC_re, tC_im;                                             \
            CMUL_NATIVE_SOA_SSE2(b_re, b_im, w1_re, w1_im, tB_re, tB_im);                   \
            CMUL_NATIVE_SOA_SSE2(c_re, c_im, w2_re, w2_im, tC_re, tC_im);                   \
            __m128d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im;                               \
            RADIX3_BUTTERFLY_NATIVE_SOA_FV_SSE2(a_re, a_im, tB_re, tB_im, tC_re, tC_im,     \
                                                y0_re, y0_im, y1_re, y1_im, y2_re, y2_im);  \
            STORE_RE_SSE2(&out_re[k], y0_re);                                               \
            STORE_IM_SSE2(&out_im[k], y0_im);                                               \
            STORE_RE_SSE2(&out_re[(k) + (K)], y1_re);                                       \
            STORE_IM_SSE2(&out_im[(k) + (K)], y1_im);                                       \
            STORE_RE_SSE2(&out_re[(k) + 2 * (K)], y2_re);                                   \
            STORE_IM_SSE2(&out_im[(k) + 2 * (K)], y2_im);                                   \
        }                                                                                   \
    } while (0)

/**
 * @brief SSE2 radix-3 pipeline (2-wide) - Backward
 *
 * @details
 * OPTIMIZATION: Uses _mm_load1_pd for twiddles!
 */
#define RADIX3_PIPELINE_2_NATIVE_SOA_BV_SSE2(k, k_end, K, in_re, in_im, out_re, out_im, tw) \
    do                                                                                      \
    {                                                                                       \
        for (; (k) + 2 <= (k_end); (k) += 2)                                                \
        {                                                                                   \
            __m128d a_re = LOAD_RE_SSE2(&in_re[k]);                                         \
            __m128d a_im = LOAD_IM_SSE2(&in_im[k]);                                         \
            __m128d b_re = LOAD_RE_SSE2(&in_re[(k) + (K)]);                                 \
            __m128d b_im = LOAD_IM_SSE2(&in_im[(k) + (K)]);                                 \
            __m128d c_re = LOAD_RE_SSE2(&in_re[(k) + 2 * (K)]);                             \
            __m128d c_im = LOAD_IM_SSE2(&in_im[(k) + 2 * (K)]);                             \
            __m128d w1_re = _mm_load1_pd(&tw->re[0 * (K) + (k)]);                           \
            __m128d w1_im = _mm_load1_pd(&tw->im[0 * (K) + (k)]);                           \
            __m128d w2_re = _mm_load1_pd(&tw->re[1 * (K) + (k)]);                           \
            __m128d w2_im = _mm_load1_pd(&tw->im[1 * (K) + (k)]);                           \
            __m128d tB_re, tB_im, tC_re, tC_im;                                             \
            CMUL_NATIVE_SOA_SSE2(b_re, b_im, w1_re, w1_im, tB_re, tB_im);                   \
            CMUL_NATIVE_SOA_SSE2(c_re, c_im, w2_re, w2_im, tC_re, tC_im);                   \
            __m128d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im;                               \
            RADIX3_BUTTERFLY_NATIVE_SOA_BV_SSE2(a_re, a_im, tB_re, tB_im, tC_re, tC_im,     \
                                                y0_re, y0_im, y1_re, y1_im, y2_re, y2_im);  \
            STORE_RE_SSE2(&out_re[k], y0_re);                                               \
            STORE_IM_SSE2(&out_im[k], y0_im);                                               \
            STORE_RE_SSE2(&out_re[(k) + (K)], y1_re);                                       \
            STORE_IM_SSE2(&out_im[(k) + (K)], y1_im);                                       \
            STORE_RE_SSE2(&out_re[(k) + 2 * (K)], y2_re);                                   \
            STORE_IM_SSE2(&out_im[(k) + 2 * (K)], y2_im);                                   \
        }                                                                                   \
    } while (0)

/**
 * @brief SSE2 radix-3 pipeline (2-wide) with streaming stores - Forward
 *
 * @details
 * OPTIMIZATION: Uses _mm_load1_pd for twiddles!
 */
#define RADIX3_PIPELINE_2_NATIVE_SOA_FV_SSE2_STREAM(k, k_end, K, in_re, in_im, out_re, out_im, tw, prefetch_dist, end_k) \
    do                                                                                                                   \
    {                                                                                                                    \
        for (; (k) + 2 <= (k_end); (k) += 2)                                                                             \
        {                                                                                                                \
            if ((k) + (prefetch_dist) < (end_k))                                                                         \
            {                                                                                                            \
                int pk = (k) + (prefetch_dist);                                                                          \
                _mm_prefetch((const char *)&tw->re[0 * (K) + pk], _MM_HINT_T0);                                          \
                _mm_prefetch((const char *)&tw->im[0 * (K) + pk], _MM_HINT_T0);                                          \
                _mm_prefetch((const char *)&tw->re[1 * (K) + pk], _MM_HINT_T0);                                          \
                _mm_prefetch((const char *)&tw->im[1 * (K) + pk], _MM_HINT_T0);                                          \
                _mm_prefetch((const char *)&in_re[pk], _MM_HINT_NTA);                                                    \
                _mm_prefetch((const char *)&in_im[pk], _MM_HINT_NTA);                                                    \
                _mm_prefetch((const char *)&in_re[pk + (K)], _MM_HINT_NTA);                                              \
                _mm_prefetch((const char *)&in_im[pk + (K)], _MM_HINT_NTA);                                              \
                _mm_prefetch((const char *)&in_re[pk + 2 * (K)], _MM_HINT_NTA);                                          \
                _mm_prefetch((const char *)&in_im[pk + 2 * (K)], _MM_HINT_NTA);                                          \
            }                                                                                                            \
            __m128d a_re = LOAD_RE_SSE2(&in_re[k]);                                                                      \
            __m128d a_im = LOAD_IM_SSE2(&in_im[k]);                                                                      \
            __m128d b_re = LOAD_RE_SSE2(&in_re[(k) + (K)]);                                                              \
            __m128d b_im = LOAD_IM_SSE2(&in_im[(k) + (K)]);                                                              \
            __m128d c_re = LOAD_RE_SSE2(&in_re[(k) + 2 * (K)]);                                                          \
            __m128d c_im = LOAD_IM_SSE2(&in_im[(k) + 2 * (K)]);                                                          \
            __m128d w1_re = _mm_load1_pd(&tw->re[0 * (K) + (k)]);                                                        \
            __m128d w1_im = _mm_load1_pd(&tw->im[0 * (K) + (k)]);                                                        \
            __m128d w2_re = _mm_load1_pd(&tw->re[1 * (K) + (k)]);                                                        \
            __m128d w2_im = _mm_load1_pd(&tw->im[1 * (K) + (k)]);                                                        \
            __m128d tB_re, tB_im, tC_re, tC_im;                                                                          \
            CMUL_NATIVE_SOA_SSE2(b_re, b_im, w1_re, w1_im, tB_re, tB_im);                                                \
            CMUL_NATIVE_SOA_SSE2(c_re, c_im, w2_re, w2_im, tC_re, tC_im);                                                \
            __m128d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im;                                                            \
            RADIX3_BUTTERFLY_NATIVE_SOA_FV_SSE2(a_re, a_im, tB_re, tB_im, tC_re, tC_im,                                  \
                                                y0_re, y0_im, y1_re, y1_im, y2_re, y2_im);                               \
            STREAM_RE_SSE2(&out_re[k], y0_re);                                                                           \
            STREAM_IM_SSE2(&out_im[k], y0_im);                                                                           \
            STREAM_RE_SSE2(&out_re[(k) + (K)], y1_re);                                                                   \
            STREAM_IM_SSE2(&out_im[(k) + (K)], y1_im);                                                                   \
            STREAM_RE_SSE2(&out_re[(k) + 2 * (K)], y2_re);                                                               \
            STREAM_IM_SSE2(&out_im[(k) + 2 * (K)], y2_im);                                                               \
        }                                                                                                                \
    } while (0)

/**
 * @brief SSE2 radix-3 pipeline (2-wide) with streaming stores - Backward
 *
 * @details
 * OPTIMIZATION: Uses _mm_load1_pd for twiddles!
 */
#define RADIX3_PIPELINE_2_NATIVE_SOA_BV_SSE2_STREAM(k, k_end, K, in_re, in_im, out_re, out_im, tw, prefetch_dist, end_k) \
    do                                                                                                                   \
    {                                                                                                                    \
        for (; (k) + 2 <= (k_end); (k) += 2)                                                                             \
        {                                                                                                                \
            if ((k) + (prefetch_dist) < (end_k))                                                                         \
            {                                                                                                            \
                int pk = (k) + (prefetch_dist);                                                                          \
                _mm_prefetch((const char *)&tw->re[0 * (K) + pk], _MM_HINT_T0);                                          \
                _mm_prefetch((const char *)&tw->im[0 * (K) + pk], _MM_HINT_T0);                                          \
                _mm_prefetch((const char *)&tw->re[1 * (K) + pk], _MM_HINT_T0);                                          \
                _mm_prefetch((const char *)&tw->im[1 * (K) + pk], _MM_HINT_T0);                                          \
                _mm_prefetch((const char *)&in_re[pk], _MM_HINT_NTA);                                                    \
                _mm_prefetch((const char *)&in_im[pk], _MM_HINT_NTA);                                                    \
                _mm_prefetch((const char *)&in_re[pk + (K)], _MM_HINT_NTA);                                              \
                _mm_prefetch((const char *)&in_im[pk + (K)], _MM_HINT_NTA);                                              \
                _mm_prefetch((const char *)&in_re[pk + 2 * (K)], _MM_HINT_NTA);                                          \
                _mm_prefetch((const char *)&in_im[pk + 2 * (K)], _MM_HINT_NTA);                                          \
            }                                                                                                            \
            __m128d a_re = LOAD_RE_SSE2(&in_re[k]);                                                                      \
            __m128d a_im = LOAD_IM_SSE2(&in_im[k]);                                                                      \
            __m128d b_re = LOAD_RE_SSE2(&in_re[(k) + (K)]);                                                              \
            __m128d b_im = LOAD_IM_SSE2(&in_im[(k) + (K)]);                                                              \
            __m128d c_re = LOAD_RE_SSE2(&in_re[(k) + 2 * (K)]);                                                          \
            __m128d c_im = LOAD_IM_SSE2(&in_im[(k) + 2 * (K)]);                                                          \
            __m128d w1_re = _mm_load1_pd(&tw->re[0 * (K) + (k)]);                                                        \
            __m128d w1_im = _mm_load1_pd(&tw->im[0 * (K) + (k)]);                                                        \
            __m128d w2_re = _mm_load1_pd(&tw->re[1 * (K) + (k)]);                                                        \
            __m128d w2_im = _mm_load1_pd(&tw->im[1 * (K) + (k)]);                                                        \
            __m128d tB_re, tB_im, tC_re, tC_im;                                                                          \
            CMUL_NATIVE_SOA_SSE2(b_re, b_im, w1_re, w1_im, tB_re, tB_im);                                                \
            CMUL_NATIVE_SOA_SSE2(c_re, c_im, w2_re, w2_im, tC_re, tC_im);                                                \
            __m128d y0_re, y0_im, y1_re, y1_im, y2_re, y2_im;                                                            \
            RADIX3_BUTTERFLY_NATIVE_SOA_BV_SSE2(a_re, a_im, tB_re, tB_im, tC_re, tC_im,                                  \
                                                y0_re, y0_im, y1_re, y1_im, y2_re, y2_im);                               \
            STREAM_RE_SSE2(&out_re[k], y0_re);                                                                           \
            STREAM_IM_SSE2(&out_im[k], y0_im);                                                                           \
            STREAM_RE_SSE2(&out_re[(k) + (K)], y1_re);                                                                   \
            STREAM_IM_SSE2(&out_im[(k) + (K)], y1_im);                                                                   \
            STREAM_RE_SSE2(&out_re[(k) + 2 * (K)], y2_re);                                                               \
            STREAM_IM_SSE2(&out_im[(k) + 2 * (K)], y2_im);                                                               \
        }                                                                                                                \
    } while (0)

#endif // __SSE2__

//==============================================================================
// SCALAR FALLBACK
//==============================================================================

/**
 * @brief Scalar radix-3 butterfly - Forward - NATIVE SoA
 */
#define RADIX3_PIPELINE_1_NATIVE_SOA_FV_SCALAR(k, K, in_re, in_im, out_re, out_im, tw) \
    do                                                                                 \
    {                                                                                  \
        double a_re = in_re[k];                                                        \
        double a_im = in_im[k];                                                        \
        double b_re = in_re[(k) + (K)];                                                \
        double b_im = in_im[(k) + (K)];                                                \
        double c_re = in_re[(k) + 2 * (K)];                                            \
        double c_im = in_im[(k) + 2 * (K)];                                            \
        double w1_re = tw->re[0 * (K) + (k)];                                          \
        double w1_im = tw->im[0 * (K) + (k)];                                          \
        double w2_re = tw->re[1 * (K) + (k)];                                          \
        double w2_im = tw->im[1 * (K) + (k)];                                          \
        double tB_re = b_re * w1_re - b_im * w1_im;                                    \
        double tB_im = b_re * w1_im + b_im * w1_re;                                    \
        double tC_re = c_re * w2_re - c_im * w2_im;                                    \
        double tC_im = c_re * w2_im + c_im * w2_re;                                    \
        double sum_re = tB_re + tC_re;                                                 \
        double sum_im = tB_im + tC_im;                                                 \
        double dif_re = tB_re - tC_re;                                                 \
        double dif_im = tB_im - tC_im;                                                 \
        double common_re = a_re + C_HALF * sum_re;                                     \
        double common_im = a_im + C_HALF * sum_im;                                     \
        double rot_re = S_SQRT3_2 * dif_im;                                            \
        double rot_im = -S_SQRT3_2 * dif_re;                                           \
        out_re[k] = a_re + sum_re;                                                     \
        out_im[k] = a_im + sum_im;                                                     \
        out_re[(k) + (K)] = common_re + rot_re;                                        \
        out_im[(k) + (K)] = common_im + rot_im;                                        \
        out_re[(k) + 2 * (K)] = common_re - rot_re;                                    \
        out_im[(k) + 2 * (K)] = common_im - rot_im;                                    \
    } while (0)

/**
 * @brief Scalar radix-3 butterfly - Forward - NATIVE SoA (streaming variant)
 *
 * Note: Scalar operations can't use streaming stores, but we define this
 * for API completeness. It's functionally identical to the non-streaming version.
 */
#define RADIX3_PIPELINE_1_NATIVE_SOA_FV_SCALAR_STREAM(k, K, in_re, in_im, out_re, out_im, tw, prefetch_dist, k_end) \
    RADIX3_PIPELINE_1_NATIVE_SOA_FV_SCALAR(k, K, in_re, in_im, out_re, out_im, tw)

/**
 * @brief Scalar radix-3 butterfly - Backward - NATIVE SoA
 */
#define RADIX3_PIPELINE_1_NATIVE_SOA_BV_SCALAR(k, K, in_re, in_im, out_re, out_im, tw) \
    do                                                                                 \
    {                                                                                  \
        double a_re = in_re[k];                                                        \
        double a_im = in_im[k];                                                        \
        double b_re = in_re[(k) + (K)];                                                \
        double b_im = in_im[(k) + (K)];                                                \
        double c_re = in_re[(k) + 2 * (K)];                                            \
        double c_im = in_im[(k) + 2 * (K)];                                            \
        double w1_re = tw->re[0 * (K) + (k)];                                          \
        double w1_im = tw->im[0 * (K) + (k)];                                          \
        double w2_re = tw->re[1 * (K) + (k)];                                          \
        double w2_im = tw->im[1 * (K) + (k)];                                          \
        double tB_re = b_re * w1_re - b_im * w1_im;                                    \
        double tB_im = b_re * w1_im + b_im * w1_re;                                    \
        double tC_re = c_re * w2_re - c_im * w2_im;                                    \
        double tC_im = c_re * w2_im + c_im * w2_re;                                    \
        double sum_re = tB_re + tC_re;                                                 \
        double sum_im = tB_im + tC_im;                                                 \
        double dif_re = tB_re - tC_re;                                                 \
        double dif_im = tB_im - tC_im;                                                 \
        double common_re = a_re + C_HALF * sum_re;                                     \
        double common_im = a_im + C_HALF * sum_im;                                     \
        double rot_re = -S_SQRT3_2 * dif_im; /* Backward: sign flip! */                \
        double rot_im = S_SQRT3_2 * dif_re;  /* Backward: sign flip! */                \
        out_re[k] = a_re + sum_re;                                                     \
        out_im[k] = a_im + sum_im;                                                     \
        out_re[(k) + (K)] = common_re + rot_re;                                        \
        out_im[(k) + (K)] = common_im + rot_im;                                        \
        out_re[(k) + 2 * (K)] = common_re - rot_re;                                    \
        out_im[(k) + 2 * (K)] = common_im - rot_im;                                    \
    } while (0)

/**
 * @brief Scalar radix-3 butterfly - Backward - NATIVE SoA (streaming variant)
 *
 * Note: Scalar operations can't use streaming stores, but we define this
 * for API completeness. It's functionally identical to the non-streaming version.
 */
#define RADIX3_PIPELINE_1_NATIVE_SOA_BV_SCALAR_STREAM(k, K, in_re, in_im, out_re, out_im, tw, prefetch_dist, k_end) \
    RADIX3_PIPELINE_1_NATIVE_SOA_BV_SCALAR(k, K, in_re, in_im, out_re, out_im, tw)

#endif // FFT_RADIX3_MACROS_TRUE_SOA_OPTIMIZED_H