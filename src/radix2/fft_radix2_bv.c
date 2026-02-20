/**
 * @file fft_radix2_bv.c
 * @brief Unified Radix-2 FFT Butterfly Implementation
 *
 * FIXES from original:
 *  1. Removed #ifndef include guard (this is a .c TU, not a header)
 *  2. Query functions: external linkage (not static inline) to match fft_radix2.h
 *  3. Naming: _n1 convention consistent with scalarn1 and avx2n1 headers
 */

#include <stdint.h>
#include <immintrin.h>
#include "fft_radix2_uniform.h"

/* Architecture-specific headers */
#ifdef __AVX512F__
#include "fft_radix2_avx512.h"
#include "fft_radix2_avx512n1.h"
#define RADIX2_HAS_AVX512 1
#define RADIX2_VECTOR_WIDTH AVX512_VECTOR_WIDTH
#define RADIX2_ALIGNMENT AVX512_ALIGNMENT
#define RADIX2_PREFETCH_DISTANCE AVX512_PREFETCH_DISTANCE
#endif

#ifdef __AVX2__
#include "fft_radix2_avx2.h"
#include "fft_radix2_avx2n1.h"
#define RADIX2_HAS_AVX2 1
#ifndef RADIX2_VECTOR_WIDTH
#define RADIX2_VECTOR_WIDTH AVX2_VECTOR_WIDTH
#define RADIX2_ALIGNMENT AVX2_ALIGNMENT
#define RADIX2_PREFETCH_DISTANCE AVX2_PREFETCH_DISTANCE
#endif
#endif

#include "fft_radix2_sse2.h"
#include "fft_radix2_sse2n1.h"
#define RADIX2_HAS_SSE2 1
#ifndef RADIX2_VECTOR_WIDTH
#define RADIX2_VECTOR_WIDTH SSE2_VECTOR_WIDTH
#define RADIX2_ALIGNMENT SSE2_ALIGNMENT
#define RADIX2_PREFETCH_DISTANCE SSE2_PREFETCH_DISTANCE
#endif

#include "fft_radix2_scalar.h"
#include "fft_radix2_scalarn1.h"
#define RADIX2_HAS_SCALAR 1

/* Unroll config */
#ifdef __AVX512F__
#define RADIX2_OPTIMAL_UNROLL 4
#elif defined(__AVX2__)
#define RADIX2_OPTIMAL_UNROLL 2
#else
#define RADIX2_OPTIMAL_UNROLL 2
#endif

#define RADIX2_STREAMING_THRESHOLD (8 * 1024 * 1024)

/*==========================================================================
 * INTERNAL: N1 (twiddle-less) range processing
 *==========================================================================*/

static inline __attribute__((always_inline)) void _process_range_simd_n1(
    int k_start, int k_end,
    const double *restrict in_re, const double *restrict in_im,
    double *restrict out_re, double *restrict out_im,
    int half, int use_streaming, int prefetch_dist)
{
#if defined(__GNUC__) || defined(__clang__)
    in_re  = (const double *)__builtin_assume_aligned(in_re,  RADIX2_ALIGNMENT);
    in_im  = (const double *)__builtin_assume_aligned(in_im,  RADIX2_ALIGNMENT);
    out_re = (double *)__builtin_assume_aligned(out_re, RADIX2_ALIGNMENT);
    out_im = (double *)__builtin_assume_aligned(out_im, RADIX2_ALIGNMENT);
#endif

    int k = k_start;

    if (use_streaming) {
        const size_t va = RADIX2_VECTOR_WIDTH * sizeof(double);
        while (k < k_end &&
               ((((uintptr_t)&in_re[k])        % va) != 0 ||
                (((uintptr_t)&in_im[k])        % va) != 0 ||
                (((uintptr_t)&in_re[k + half]) % va) != 0 ||
                (((uintptr_t)&in_im[k + half]) % va) != 0 ||
                (((uintptr_t)&out_re[k])       % va) != 0 ||
                (((uintptr_t)&out_im[k])       % va) != 0 ||
                (((uintptr_t)&out_re[k + half])% va) != 0 ||
                (((uintptr_t)&out_im[k + half])% va) != 0))
        {
            radix2_pipeline_1_scalar_n1(k, in_re, in_im, out_re, out_im, half);
            k++;
        }
    }

#ifdef __AVX512F__
    /* AVX-512: 4× unroll (32 bf), 2× (16), 1× (8)
     * Non-streaming: use unaligned variants (half may not be multiple of 8)
     */
    if (use_streaming) {
        for (; k + 31 < k_end; k += 32)
            radix2_pipeline_32_avx512_n1_unroll4_stream(k, in_re, in_im, out_re, out_im, half, prefetch_dist);
    } else {
        for (; k + 31 < k_end; k += 32)
            radix2_pipeline_32_avx512_n1_unroll4(k, in_re, in_im, out_re, out_im, half, prefetch_dist);
    }
    if (k + 15 < k_end) {
        if (use_streaming) radix2_pipeline_16_avx512_n1_unroll2_stream(k, in_re, in_im, out_re, out_im, half, prefetch_dist);
        else               radix2_pipeline_16_avx512_n1_unroll2(k, in_re, in_im, out_re, out_im, half, prefetch_dist);
        k += 16;
    }
    if (k + 7 < k_end) {
        if (use_streaming) radix2_pipeline_8_avx512_n1_stream(k, in_re, in_im, out_re, out_im, half, prefetch_dist);
        else               radix2_pipeline_8_avx512_n1(k, in_re, in_im, out_re, out_im, half, prefetch_dist);
        k += 8;
    }

#elif defined(__AVX2__)
    /* AVX2: 2× unroll (8 bf), 1× (4) cleanup
     * Streaming path: alignment peeling above guarantees aligned addresses → use _aligned/_stream
     * Non-streaming:  no peeling → must use unaligned variants to handle arbitrary 'half'
     */
    if (use_streaming) {
        for (; k + 7 < k_end; k += 8)
            radix2_pipeline_8_avx2_n1_unroll2_stream(k, in_re, in_im, out_re, out_im, half, prefetch_dist);
    } else {
        for (; k + 7 < k_end; k += 8)
            radix2_pipeline_8_avx2_n1_unroll2(k, in_re, in_im, out_re, out_im, half, prefetch_dist);
    }
    if (k + 3 < k_end) {
        if (use_streaming) radix2_pipeline_4_avx2_n1_stream(k, in_re, in_im, out_re, out_im, half, prefetch_dist);
        else               radix2_pipeline_4_avx2_n1(k, in_re, in_im, out_re, out_im, half, prefetch_dist);
        k += 4;
    }

#else /* SSE2 */
    /* SSE2: 2× unroll (4 bf), 1× (2) cleanup
     * Non-streaming: use unaligned variants (half may not be multiple of 2)
     */
    if (use_streaming) {
        for (; k + 3 < k_end; k += 4)
            radix2_pipeline_4_sse2_n1_unroll2_stream(k, in_re, in_im, out_re, out_im, half, prefetch_dist);
    } else {
        for (; k + 3 < k_end; k += 4)
            radix2_pipeline_4_sse2_n1_unroll2(k, in_re, in_im, out_re, out_im, half, prefetch_dist);
    }
    if (k + 1 < k_end) {
        if (use_streaming) radix2_pipeline_2_sse2_n1_stream(k, in_re, in_im, out_re, out_im, half, prefetch_dist);
        else               radix2_pipeline_2_sse2_n1(k, in_re, in_im, out_re, out_im, half, prefetch_dist);
        k += 2;
    }
#endif

    while (k < k_end) {
        radix2_pipeline_1_scalar_n1(k, in_re, in_im, out_re, out_im, half);
        k++;
    }
}

/*==========================================================================
 * INTERNAL: General (with twiddles) range processing
 *==========================================================================*/

static inline __attribute__((always_inline)) void _process_range_simd(
    int k_start, int k_end,
    double *restrict out_re, double *restrict out_im,
    const double *restrict in_re, const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw,
    int half, int use_streaming, int prefetch_dist)
{
#if defined(__GNUC__) || defined(__clang__)
    in_re  = (const double *)__builtin_assume_aligned(in_re,  RADIX2_ALIGNMENT);
    in_im  = (const double *)__builtin_assume_aligned(in_im,  RADIX2_ALIGNMENT);
    out_re = (double *)__builtin_assume_aligned(out_re, RADIX2_ALIGNMENT);
    out_im = (double *)__builtin_assume_aligned(out_im, RADIX2_ALIGNMENT);
    /* Twiddle alignment guaranteed by planner allocation */
#endif

    int k = k_start;

    if (use_streaming) {
        const size_t va = RADIX2_VECTOR_WIDTH * sizeof(double);
        while (k < k_end &&
               ((((uintptr_t)&in_re[k])        % va) != 0 ||
                (((uintptr_t)&in_im[k])        % va) != 0 ||
                (((uintptr_t)&in_re[k + half]) % va) != 0 ||
                (((uintptr_t)&in_im[k + half]) % va) != 0 ||
                (((uintptr_t)&out_re[k])       % va) != 0 ||
                (((uintptr_t)&out_im[k])       % va) != 0 ||
                (((uintptr_t)&out_re[k + half])% va) != 0 ||
                (((uintptr_t)&out_im[k + half])% va) != 0))
        {
            radix2_pipeline_1_scalar(k, in_re, in_im, out_re, out_im, stage_tw, half);
            k++;
        }
    }

#ifdef __AVX512F__
    /* AVX-512: 4× unroll (32 bf), 2× (16), 1× (8), masked tail
     * Non-streaming: use unaligned variants (half may not be multiple of 8)
     */
    if (use_streaming) {
        for (; k + 31 < k_end; k += 32)
            radix2_pipeline_32_avx512_unroll4_stream(k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
    } else {
        for (; k + 31 < k_end; k += 32)
            radix2_pipeline_32_avx512_unroll4(k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
    }
    if (k + 15 < k_end) {
        if (use_streaming) radix2_pipeline_16_avx512_unroll2_stream(k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
        else               radix2_pipeline_16_avx512_unroll2(k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
        k += 16;
    }
    if (k + 7 < k_end) {
        if (use_streaming) radix2_pipeline_8_avx512_stream(k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
        else               radix2_pipeline_8_avx512(k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
        k += 8;
    }
    if (k < k_end) {
        radix2_pipeline_masked_avx512(k, k_end - k, in_re, in_im, out_re, out_im, stage_tw, half);
        k = k_end;
    }

#elif defined(__AVX2__)
    /* AVX2: 2× unroll (8 bf), 1× (4) cleanup
     * Non-streaming: use unaligned variants (half may not be multiple of 4)
     */
    if (use_streaming) {
        for (; k + 7 < k_end; k += 8)
            radix2_pipeline_8_avx2_unroll2_stream(k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
    } else {
        for (; k + 7 < k_end; k += 8)
            radix2_pipeline_8_avx2_unroll2(k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
    }
    if (k + 3 < k_end) {
        if (use_streaming) radix2_pipeline_4_avx2_stream(k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
        else               radix2_pipeline_4_avx2(k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
        k += 4;
    }

#else /* SSE2 */
    /* SSE2: 2× unroll (4 bf), 1× (2) cleanup
     * Non-streaming: use unaligned variants (half may not be even-aligned)
     */
    if (use_streaming) {
        for (; k + 3 < k_end; k += 4)
            radix2_pipeline_4_sse2_unroll2_stream(k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
    } else {
        for (; k + 3 < k_end; k += 4)
            radix2_pipeline_4_sse2_unroll2(k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
    }
    if (k + 1 < k_end) {
        if (use_streaming) radix2_pipeline_2_sse2_stream(k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
        else               radix2_pipeline_2_sse2(k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
        k += 2;
    }
#endif

    while (k < k_end) {
        radix2_pipeline_1_scalar(k, in_re, in_im, out_re, out_im, stage_tw, half);
        k++;
    }
}

/*==========================================================================
 * PUBLIC API: fft_radix2_bv() - WITH TWIDDLES
 *==========================================================================*/

void fft_radix2_bv(
    double *restrict out_re, double *restrict out_im,
    const double *restrict in_re, const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw, int half)
{
    const int prefetch_dist = RADIX2_PREFETCH_DISTANCE;
    const int N = half * 2;
    const size_t ws = (size_t)N * 4 * sizeof(double);
    const int use_streaming = (ws > RADIX2_STREAMING_THRESHOLD);

    int k = 0;

    /* k=0: W=1 */
    radix2_k0_scalar(in_re, in_im, out_re, out_im, half);
    k = 1;

    int k_quarter = 0, k_eighth = 0, k_3eighth = 0;
    if ((half & (half - 1)) == 0) {
        k_quarter = half / 2;
        if ((half & 3) == 0) {
            k_eighth  = half >> 2;
            k_3eighth = (3 * half) >> 2;
        }
    }

#define PROCESS_RANGE(s, e) \
    _process_range_simd((s), (e), out_re, out_im, in_re, in_im, stage_tw, half, use_streaming, prefetch_dist)

    if (k_eighth > 0 && k_eighth < half && k_eighth > k) {
        PROCESS_RANGE(k, k_eighth);
        radix2_k_n8_scalar(in_re, in_im, out_re, out_im, stage_tw, k_eighth, half);
        k = k_eighth + 1;
    }
    if (k_quarter > 0 && k_quarter < half && k_quarter > k) {
        PROCESS_RANGE(k, k_quarter);
        radix2_k_quarter_scalar(in_re, in_im, out_re, out_im, stage_tw, k_quarter, half);
        k = k_quarter + 1;
    }
    if (k_3eighth > 0 && k_3eighth < half && k_3eighth > k) {
        PROCESS_RANGE(k, k_3eighth);
        radix2_k_3n8_scalar(in_re, in_im, out_re, out_im, stage_tw, k_3eighth, half);
        k = k_3eighth + 1;
    }
    if (k < half)
        PROCESS_RANGE(k, half);

#undef PROCESS_RANGE

    if (use_streaming) _mm_sfence();
}

/*==========================================================================
 * PUBLIC API: fft_radix2_bv_n1() - TWIDDLE-LESS
 *==========================================================================*/

void fft_radix2_bv_n1(
    double *restrict out_re, double *restrict out_im,
    const double *restrict in_re, const double *restrict in_im,
    int half)
{
    const int prefetch_dist = RADIX2_PREFETCH_DISTANCE;
    const int N = half * 2;
    const size_t ws = (size_t)N * 4 * sizeof(double);
    const int use_streaming = (ws > RADIX2_STREAMING_THRESHOLD);

    _process_range_simd_n1(0, half, in_re, in_im, out_re, out_im,
                           half, use_streaming, prefetch_dist);

    if (use_streaming) _mm_sfence();
}

/*==========================================================================
 * PUBLIC API: Capability queries (external linkage!)
 *==========================================================================*/

const char *radix2_get_simd_capabilities(void)
{
#ifdef __AVX512F__
    return "AVX-512F (8x double, 4x unroll, N1 support)";
#elif defined(__AVX2__)
    return "AVX2 (4x double, 2x unroll, FMA, N1 support)";
#else
    return "SSE2 (2x double, 2x unroll, N1 support)";
#endif
}

size_t radix2_get_alignment_requirement(void) { return RADIX2_ALIGNMENT; }
int    radix2_get_vector_width(void)          { return RADIX2_VECTOR_WIDTH; }
