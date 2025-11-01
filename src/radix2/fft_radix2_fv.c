/**
 * @file fft_radix2_bv.c
 * @brief Unified Radix-2 FFT Butterfly Interface - Multi-Architecture with N1 Support
 *
 * @details
 * This header provides a unified interface to all radix-2 FFT butterfly
 * implementations across different SIMD architectures, including twiddle-less
 * (N1) variants for optimal first-stage performance.
 *
 * Architecture Support:
 * - AVX-512: 8 doubles/vector, 2×/4× unroll, streaming stores, masked cleanup
 * - AVX2:    4 doubles/vector, 2× unroll, FMA support
 * - SSE2:    2 doubles/vector, 2× unroll, baseline x86-64
 * - Scalar:  Single butterfly, special cases (k=0, N/4, N/8, 3N/8)
 *
 * N1 (Twiddle-Less) Support:
 * - ~3× faster than general butterfly for W=1 cases
 * - Automatic hybrid mode: N1 for [0, threshold), general for [threshold, half)
 * - All SIMD architectures supported
 *
 * @author FFT Optimization Team
 * @version 3.1 (Added N1 integration and hybrid mode)
 * @date 2025
 */

#ifndef FFT_RADIX2_OPTIMIZED_H
#define FFT_RADIX2_OPTIMIZED_H

#include <stdint.h> // For uintptr_t (alignment checks)
#include "fft_radix2_uniform.h"

//==============================================================================
// ARCHITECTURE DETECTION AND CONFIGURATION
//==============================================================================

// Include architecture-specific headers based on compiler flags
#ifdef __AVX512F__
#include "fft_radix2_avx512.h"
#include "fft_radix2_avx512n1.h"  // N1 (twiddle-less) variant
#define RADIX2_HAS_AVX512 1
#define RADIX2_VECTOR_WIDTH AVX512_VECTOR_WIDTH
#define RADIX2_ALIGNMENT AVX512_ALIGNMENT
#define RADIX2_PREFETCH_DISTANCE AVX512_PREFETCH_DISTANCE
#endif

#ifdef __AVX2__
#include "fft_radix2_avx2.h"
#include "fft_radix2_avx2n1.h"  // N1 (twiddle-less) variant
#define RADIX2_HAS_AVX2 1
#ifndef RADIX2_VECTOR_WIDTH // Don't override if AVX-512 already defined
#define RADIX2_VECTOR_WIDTH AVX2_VECTOR_WIDTH
#define RADIX2_ALIGNMENT AVX2_ALIGNMENT
#define RADIX2_PREFETCH_DISTANCE AVX2_PREFETCH_DISTANCE
#endif
#endif

#include "fft_radix2_sse2.h" // Always available on x86-64
#include "fft_radix2_sse2n1.h"  // N1 (twiddle-less) variant
#define RADIX2_HAS_SSE2 1
#ifndef RADIX2_VECTOR_WIDTH
#define RADIX2_VECTOR_WIDTH SSE2_VECTOR_WIDTH
#define RADIX2_ALIGNMENT SSE2_ALIGNMENT
#define RADIX2_PREFETCH_DISTANCE SSE2_PREFETCH_DISTANCE
#endif

#include "fft_radix2_scalar.h" // Always available
#include "fft_radix2_scalarn1.h"  // N1 (twiddle-less) variant
#define RADIX2_HAS_SCALAR 1

//==============================================================================
// OPTIMAL UNROLL SELECTION
//==============================================================================

/**
 * @brief Optimal unroll depth for current architecture
 * @details
 * - AVX-512: 4× unroll (32 butterflies) for high-end Intel
 * - AVX2:    2× unroll (8 butterflies)
 * - SSE2:    2× unroll (4 butterflies)
 */
#ifdef __AVX512F__
#define RADIX2_OPTIMAL_UNROLL 4
#define RADIX2_OPTIMAL_BUTTERFLIES_PER_ITER 32
#elif defined(__AVX2__)
#define RADIX2_OPTIMAL_UNROLL 2
#define RADIX2_OPTIMAL_BUTTERFLIES_PER_ITER 8
#else
#define RADIX2_OPTIMAL_UNROLL 2
#define RADIX2_OPTIMAL_BUTTERFLIES_PER_ITER 4
#endif

//==============================================================================
// NON-TEMPORAL STORE FENCE
//==============================================================================

/**
 * @brief Memory fence for non-temporal stores
 *
 * @details
 * CRITICAL: Must be called after using *_stream() functions to ensure all
 * non-temporal stores complete before subsequent memory operations.
 *
 * Usage pattern:
 * @code
 *   // Process stage with streaming stores
 *   for (each stage) {
 *       radix2_process_butterflies_optimal(..., use_streaming=1, ...);
 *       if (use_streaming) {
 *           RADIX2_NT_STAGE_FENCE();  // Fence once per stage
 *       }
 *   }
 * @endcode
 *
 * Performance note: Fence is expensive (~20-30 cycles), so amortize by calling
 * once per stage, not per butterfly. The *_stream() functions deliberately do
 * NOT include fences internally for this reason.
 */
#define RADIX2_NT_STAGE_FENCE() _mm_sfence()

//==============================================================================
// INTERNAL HELPER: N1 (TWIDDLE-LESS) RANGE PROCESSING
//==============================================================================

/**
 * @brief Process range with N1 (twiddle-less) variants - internal helper
 *
 * @details
 * Processes butterflies in range [k_start, k_end) using twiddle-less (W=1)
 * variants. This is ~3× faster than general butterfly for first-stage
 * optimization where many twiddles are 1 or very close to 1.
 *
 * @param[in] k_start Starting butterfly index
 * @param[in] k_end Ending butterfly index (exclusive)
 * @param[in] in_re Input real array
 * @param[in] in_im Input imaginary array
 * @param[out] out_re Output real array
 * @param[out] out_im Output imaginary array
 * @param[in] half Transform half-size
 * @param[in] use_streaming Use non-temporal stores for large N
 * @param[in] prefetch_dist Prefetch distance in elements
 */
static inline __attribute__((always_inline)) void _process_range_simd_n1(
    int k_start,
    int k_end,
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
    int half,
    int use_streaming,
    int prefetch_dist)
{
    // Alignment hints for compiler optimization
#if defined(__GNUC__) || defined(__clang__)
    in_re = (const double *)__builtin_assume_aligned(in_re, RADIX2_ALIGNMENT);
    in_im = (const double *)__builtin_assume_aligned(in_im, RADIX2_ALIGNMENT);
    out_re = (double *)__builtin_assume_aligned(out_re, RADIX2_ALIGNMENT);
    out_im = (double *)__builtin_assume_aligned(out_im, RADIX2_ALIGNMENT);
#endif

    int k = k_start;

    // Alignment peeling for streaming stores
    if (use_streaming)
    {
        const size_t vec_align = RADIX2_VECTOR_WIDTH * sizeof(double);

        // Peel until ALL addresses are aligned (both k and k+half)
        while (k < k_end &&
               ((((uintptr_t)&in_re[k]) % vec_align) != 0 ||
                (((uintptr_t)&in_im[k]) % vec_align) != 0 ||
                (((uintptr_t)&in_re[k + half]) % vec_align) != 0 ||
                (((uintptr_t)&in_im[k + half]) % vec_align) != 0 ||
                (((uintptr_t)&out_re[k]) % vec_align) != 0 ||
                (((uintptr_t)&out_im[k]) % vec_align) != 0 ||
                (((uintptr_t)&out_re[k + half]) % vec_align) != 0 ||
                (((uintptr_t)&out_im[k + half]) % vec_align) != 0))
        {
            radix2_pipeline_1_scalar_n1(k, in_re, in_im, out_re, out_im, half);
            k++;
        }
    }

// Main SIMD loop with optimal unrolling - N1 variants
#ifdef __AVX512F__
    // AVX-512: 4× unroll (32 butterflies/iteration)
    if (use_streaming)
    {
        for (; k + 31 < k_end; k += 32)
        {
            radix2_pipeline_32_avx512_n1_unroll4_stream(
                k, in_re, in_im, out_re, out_im, half, prefetch_dist);
        }
    }
    else
    {
        for (; k + 31 < k_end; k += 32)
        {
            radix2_pipeline_32_avx512_n1_unroll4_aligned(
                k, in_re, in_im, out_re, out_im, half, prefetch_dist);
        }
    }

    // Cleanup: 16 butterflies
    if (k + 15 < k_end)
    {
        if (use_streaming)
        {
            radix2_pipeline_16_avx512_n1_unroll2_stream(
                k, in_re, in_im, out_re, out_im, half, prefetch_dist);
        }
        else
        {
            radix2_pipeline_16_avx512_n1_unroll2_aligned(
                k, in_re, in_im, out_re, out_im, half, prefetch_dist);
        }
        k += 16;
    }

    // Cleanup: 8 butterflies
    if (k + 7 < k_end)
    {
        if (use_streaming)
        {
            radix2_pipeline_8_avx512_n1_stream(
                k, in_re, in_im, out_re, out_im, half, prefetch_dist);
        }
        else
        {
            radix2_pipeline_8_avx512_n1_aligned(
                k, in_re, in_im, out_re, out_im, half, prefetch_dist);
        }
        k += 8;
    }

    // Note: No masked N1 variant - use scalar cleanup

#elif defined(__AVX2__)
    // AVX2: 2× unroll (8 butterflies/iteration)
    if (use_streaming)
    {
        for (; k + 7 < k_end; k += 8)
        {
            radix2_pipeline_8_avx2_n1_unroll2_stream(
                k, in_re, in_im, out_re, out_im, half, prefetch_dist);
        }
    }
    else
    {
        for (; k + 7 < k_end; k += 8)
        {
            radix2_pipeline_8_avx2_n1_unroll2_aligned(
                k, in_re, in_im, out_re, out_im, half, prefetch_dist);
        }
    }

    // Cleanup: 4 butterflies
    if (k + 3 < k_end)
    {
        if (use_streaming)
        {
            radix2_pipeline_4_avx2_n1_stream(
                k, in_re, in_im, out_re, out_im, half, prefetch_dist);
        }
        else
        {
            radix2_pipeline_4_avx2_n1_aligned(
                k, in_re, in_im, out_re, out_im, half, prefetch_dist);
        }
        k += 4;
    }

#else // SSE2
    // SSE2: 2× unroll (4 butterflies/iteration)
    if (use_streaming)
    {
        for (; k + 3 < k_end; k += 4)
        {
            radix2_pipeline_4_sse2_n1_unroll2_stream(
                k, in_re, in_im, out_re, out_im, half, prefetch_dist);
        }
    }
    else
    {
        for (; k + 3 < k_end; k += 4)
        {
            radix2_pipeline_4_sse2_n1_unroll2_aligned(
                k, in_re, in_im, out_re, out_im, half, prefetch_dist);
        }
    }

    // Cleanup: 2 butterflies
    if (k + 1 < k_end)
    {
        if (use_streaming)
        {
            radix2_pipeline_2_sse2_n1_stream(
                k, in_re, in_im, out_re, out_im, half, prefetch_dist);
        }
        else
        {
            radix2_pipeline_2_sse2_n1_aligned(
                k, in_re, in_im, out_re, out_im, half, prefetch_dist);
        }
        k += 2;
    }
#endif

    // Scalar cleanup for final remaining butterflies
    while (k < k_end)
    {
        radix2_pipeline_1_scalar_n1(k, in_re, in_im, out_re, out_im, half);
        k++;
    }
}

//==============================================================================
// INTERNAL HELPER: GENERAL (WITH TWIDDLES) RANGE PROCESSING
//==============================================================================

/**
 * @brief Process range with optimal SIMD - internal helper
 *
 * @details
 * This function processes a contiguous range [k_start, k_end) using the
 * fastest SIMD implementation available. It handles:
 * - Alignment peeling for streaming stores
 * - Optimal unrolling (4× for AVX-512, 2× otherwise)
 * - SIMD cleanup (masked for AVX-512, smaller vectors otherwise)
 * - Scalar fallback for final elements
 */
static inline __attribute__((always_inline)) void _process_range_simd(
    int k_start,
    int k_end,
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw,
    int half,
    int use_streaming,
    int prefetch_dist)
{
    // Alignment hints for compiler optimization
#if defined(__GNUC__) || defined(__clang__)
    in_re = (const double *)__builtin_assume_aligned(in_re, RADIX2_ALIGNMENT);
    in_im = (const double *)__builtin_assume_aligned(in_im, RADIX2_ALIGNMENT);
    out_re = (double *)__builtin_assume_aligned(out_re, RADIX2_ALIGNMENT);
    out_im = (double *)__builtin_assume_aligned(out_im, RADIX2_ALIGNMENT);
    stage_tw->re = (const double *)__builtin_assume_aligned(stage_tw->re, RADIX2_ALIGNMENT);
    stage_tw->im = (const double *)__builtin_assume_aligned(stage_tw->im, RADIX2_ALIGNMENT);
#endif

    int k = k_start;

    // Alignment peeling for streaming stores
    if (use_streaming)
    {
        const size_t vec_align = RADIX2_VECTOR_WIDTH * sizeof(double);

        // Peel until ALL addresses are aligned (both k and k+half)
        while (k < k_end &&
               ((((uintptr_t)&in_re[k]) % vec_align) != 0 ||
                (((uintptr_t)&in_im[k]) % vec_align) != 0 ||
                (((uintptr_t)&in_re[k + half]) % vec_align) != 0 ||
                (((uintptr_t)&in_im[k + half]) % vec_align) != 0 ||
                (((uintptr_t)&out_re[k]) % vec_align) != 0 ||
                (((uintptr_t)&out_im[k]) % vec_align) != 0 ||
                (((uintptr_t)&out_re[k + half]) % vec_align) != 0 ||
                (((uintptr_t)&out_im[k + half]) % vec_align) != 0))
        {
            radix2_pipeline_1_scalar(k, in_re, in_im, out_re, out_im,
                                     stage_tw, half);
            k++;
        }
    }

// Main SIMD loop with optimal unrolling
#ifdef __AVX512F__
    // AVX-512: 4× unroll (32 butterflies/iteration)
    if (use_streaming)
    {
        for (; k + 31 < k_end; k += 32)
        {
            radix2_pipeline_32_avx512_unroll4_stream(
                k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
        }
    }
    else
    {
        for (; k + 31 < k_end; k += 32)
        {
            radix2_pipeline_32_avx512_unroll4_aligned(
                k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
        }
    }

    // Cleanup: 16 butterflies
    if (k + 15 < k_end)
    {
        if (use_streaming)
        {
            radix2_pipeline_16_avx512_unroll2_stream(
                k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
        }
        else
        {
            radix2_pipeline_16_avx512_unroll2_aligned(
                k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
        }
        k += 16;
    }

    // Cleanup: 8 butterflies
    if (k + 7 < k_end)
    {
        if (use_streaming)
        {
            radix2_pipeline_8_avx512_stream(
                k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
        }
        else
        {
            radix2_pipeline_8_avx512_aligned(
                k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
        }
        k += 8;
    }

    // Masked cleanup for remaining 1-7 butterflies (branchless!)
    if (k < k_end)
    {
        int remaining = k_end - k;
        if (remaining > 0)
        {
            radix2_pipeline_masked_avx512(k, remaining, in_re, in_im,
                                          out_re, out_im, stage_tw, half);
        }
        k = k_end;
    }

#elif defined(__AVX2__)
    // AVX2: 2× unroll (8 butterflies/iteration)
    if (use_streaming)
    {
        for (; k + 7 < k_end; k += 8)
        {
            radix2_pipeline_8_avx2_unroll2_stream(
                k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
        }
    }
    else
    {
        for (; k + 7 < k_end; k += 8)
        {
            radix2_pipeline_8_avx2_unroll2_aligned(
                k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
        }
    }

    // Cleanup: 4 butterflies
    if (k + 3 < k_end)
    {
        if (use_streaming)
        {
            radix2_pipeline_4_avx2_stream(
                k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
        }
        else
        {
            radix2_pipeline_4_avx2_aligned(
                k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
        }
        k += 4;
    }

#else // SSE2
    // SSE2: 2× unroll (4 butterflies/iteration)
    if (use_streaming)
    {
        for (; k + 3 < k_end; k += 4)
        {
            radix2_pipeline_4_sse2_unroll2_stream(
                k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
        }
    }
    else
    {
        for (; k + 3 < k_end; k += 4)
        {
            radix2_pipeline_4_sse2_unroll2_aligned(
                k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
        }
    }

    // Cleanup: 2 butterflies
    if (k + 1 < k_end)
    {
        if (use_streaming)
        {
            radix2_pipeline_2_sse2_stream(
                k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
        }
        else
        {
            radix2_pipeline_2_sse2_aligned(
                k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
        }
        k += 2;
    }
#endif

    // Scalar cleanup for final remaining butterflies
    while (k < k_end)
    {
        radix2_pipeline_1_scalar(k, in_re, in_im, out_re, out_im,
                                 stage_tw, half);
        k++;
    }
}

//==============================================================================
// UNIFIED PIPELINE INTERFACE
//==============================================================================

/**
 * @brief Process butterflies using optimal SIMD path for current architecture
 *
 * @details
 * This is the main entry point for butterfly processing. It automatically
 * dispatches to the fastest implementation available on the current CPU.
 *
 * The function processes butterflies in phases:
 * 1. N1 (twiddle-less) range [0, n1_threshold) - ~3× faster
 * 2. Special cases (k=N/4, N/8, 3N/8) - optimized scalar
 * 3. Bulk processing with optimal SIMD and unrolling
 * 4. Cleanup remaining butterflies
 *
 * @param[out] out_re Output real array (N elements)
 * @param[out] out_im Output imaginary array (N elements)
 * @param[in] in_re Input real array (N elements)
 * @param[in] in_im Input imaginary array (N elements)
 * @param[in] stage_tw Stage twiddle factors (SoA format)
 * @param[in] half Transform half-size (N/2)
 * @param[in] use_streaming Use non-temporal stores for large N
 * @param[in] n1_threshold Number of butterflies to process without twiddles
 *                         (0 = no N1 optimization, typical: 4-16)
 *
 * @note Arrays must be aligned to RADIX2_ALIGNMENT bytes
 * @note Out-of-place only (in != out)
 * @note If use_streaming=1, caller MUST call RADIX2_NT_STAGE_FENCE() after
 *       this function to ensure all non-temporal stores complete
 * @note n1_threshold should be chosen based on how many twiddles are close to 1
 *       (typically 4-16 for first stage, 0 for other stages)
 */
static inline void radix2_process_butterflies_optimal(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw,
    int half,
    int use_streaming,
    int n1_threshold)
{
    const int prefetch_dist = RADIX2_PREFETCH_DISTANCE;
    int k = 0;

    // Phase 1: N1 (twiddle-less) butterflies for [0, n1_threshold)
    if (n1_threshold > 0 && n1_threshold <= half)
    {
        // Special case: k=0 is always W=1, handle it first
        radix2_k0_scalar(in_re, in_im, out_re, out_im, half);
        k = 1;
        
        // Process remaining N1 range
        if (k < n1_threshold)
        {
            _process_range_simd_n1(k, n1_threshold, in_re, in_im,
                                   out_re, out_im, half, use_streaming, prefetch_dist);
            k = n1_threshold;
        }
    }
    else
    {
        // No N1 optimization, start with k=0 as special case
        radix2_k0_scalar(in_re, in_im, out_re, out_im, half);
        k = 1;
    }

    // Determine special case indices (only process if beyond N1 threshold)
    int k_quarter = 0;
    int k_eighth = 0;
    int k_3eighth = 0;

    if ((half & (half - 1)) == 0)
    {                         // Power of 2
        k_quarter = half / 2; // N/4

        if ((half & 3) == 0)
        {                                // Divisible by 4
            k_eighth = half >> 2;        // N/8
            k_3eighth = (3 * half) >> 2; // 3N/8
        }
    }

// Phase 2: General butterflies with twiddles, handling special cases
// We'll process in ranges, excluding special case indices

// Helper macro to process a contiguous range
#define PROCESS_RANGE(start, end)                                          \
    do                                                                     \
    {                                                                      \
        int kk = (start);                                                  \
        _process_range_simd(kk, (end), out_re, out_im, in_re, in_im,      \
                            stage_tw, half, use_streaming, prefetch_dist); \
    } while (0)

    // Range before k_eighth (check bounds and ensure it's beyond current k)
    if (k_eighth > 0 && k_eighth < half && k_eighth > k)
    {
        PROCESS_RANGE(k, k_eighth);
        radix2_k_n8_scalar(in_re, in_im, out_re, out_im, k_eighth, half);
        k = k_eighth + 1;
    }

    // Range before k_quarter (check bounds and ensure it's beyond current k)
    if (k_quarter > 0 && k_quarter < half && k_quarter > k)
    {
        PROCESS_RANGE(k, k_quarter);
        radix2_k_quarter_scalar(in_re, in_im, out_re, out_im, k_quarter, half);
        k = k_quarter + 1;
    }

    // Range before k_3eighth (check bounds and ensure it's beyond current k)
    if (k_3eighth > 0 && k_3eighth < half && k_3eighth > k)
    {
        PROCESS_RANGE(k, k_3eighth);
        radix2_k_3n8_scalar(in_re, in_im, out_re, out_im, k_3eighth, half);
        k = k_3eighth + 1;
    }

    // Final range to half
    if (k < half)
    {
        PROCESS_RANGE(k, half);
    }

#undef PROCESS_RANGE
}

//==============================================================================
// ARCHITECTURE CAPABILITY QUERY
//==============================================================================

/**
 * @brief Query which SIMD architectures are available
 *
 * @return String describing available SIMD support
 */
static inline const char *radix2_get_simd_capabilities(void)
{
#ifdef __AVX512F__
    return "AVX-512F (8×double, 4× unroll, N1 support)";
#elif defined(__AVX2__)
    return "AVX2 (4×double, 2× unroll, FMA, N1 support)";
#else
    return "SSE2 (2×double, 2× unroll, N1 support)";
#endif
}

/**
 * @brief Get optimal alignment requirement for current architecture
 *
 * @return Required alignment in bytes
 */
static inline size_t radix2_get_alignment_requirement(void)
{
    return RADIX2_ALIGNMENT;
}

/**
 * @brief Get vector width for current architecture
 *
 * @return Number of doubles per SIMD vector
 */
static inline int radix2_get_vector_width(void)
{
    return RADIX2_VECTOR_WIDTH;
}

#endif // FFT_RADIX2_OPTIMIZED_H