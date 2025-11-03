/**
 * @file fft_radix32_avx512_tile_io.h
 * @brief Tile I/O helpers for Radix-32 AVX-512 FFT (stripe ↔ tile layout)
 *
 * @details
 * These functions handle data movement between the global stripe layout
 * used by the planner and the contiguous, tile-local layout expected by
 * the fused radix-32 butterfly kernel.
 *
 * Layouts:
 *   - **Stripe layout:**  in[stripe][K]  — 32 interleaved FFT stripes
 *   - **Tile layout:**    tile[stripe][tile_size] — contiguous per stripe
 *
 * Purpose:
 *   - Convert strided global buffers into contiguous tiles for high-throughput
 *     vectorized computation (no gather/scatter penalties).
 *   - Write processed tiles back to global layout efficiently.
 *
 * Variants:
 *   - ::radix32_gather_stripes_to_tile
 *       → Simple gather, cached loads.
 *   - ::radix32_gather_stripes_to_tile_prefetch
 *       → Adds prefetch hints for next tile (hide memory latency).
 *   - ::radix32_scatter_tile_to_stripes
 *       → Writes tile back with normal (cached) stores; use when results
 *         are reused soon by next stage.
 *   - ::radix32_scatter_tile_to_stripes_nt
 *       → Uses non-temporal (streaming) stores to avoid cache pollution;
 *         requires 64-byte alignment; use for large-K or final output.
 *
 * Typical flow per tile:
 *   1. Gather stripes → tile.
 *   2. Compute fused radix-32 butterfly on tile.
 *   3. Scatter tile → stripes (cached or NT depending on reuse).
 *
 * Data volume:
 *   32 stripes × tile_size × 16 B (re+im) ≈ 32 KB @ tile_size=64
 *   → Fits cleanly in L1 for full tile processing.
 */

#include <immintrin.h>   // AVX-512 intrinsics (_mm512_*, _mm_prefetch, etc.)
#include <stddef.h>      // size_t
#include <stdint.h>      // uintptr_t (for alignment check)
#include <stdbool.h>     // bool type
#include <string.h>      // optional: for memset/memcpy if used elsewhere


// Alignment helper if not globally defined
#ifndef ALIGNAS
#  include <stdalign.h>  // for alignas() in C11
#  define ALIGNAS(x) alignas(x)
#endif

// Restrict keyword fallback for portability
#ifndef RESTRICT
#  if defined(_MSC_VER)
#    define RESTRICT __restrict
#  else
#    define RESTRICT __restrict__
#  endif
#endif

#ifndef TARGET_AVX512
  // Force compiler to generate AVX-512 instructions for these functions
  #if defined(__GNUC__) || defined(__clang__)
    #define TARGET_AVX512 __attribute__((target("avx512f,avx512dq,avx512cd,avx512bw,avx512vl")))
  #elif defined(_MSC_VER)
    #define TARGET_AVX512
  #else
    #define TARGET_AVX512
  #endif
#endif

#ifndef FORCE_INLINE
  // Hint compiler to always inline
  #if defined(_MSC_VER)
    #define FORCE_INLINE __forceinline
  #elif defined(__GNUC__) || defined(__clang__)
    #define FORCE_INLINE inline __attribute__((always_inline))
  #else
    #define FORCE_INLINE inline
  #endif
#endif
 
//==============================================================================
// GATHER/SCATTER: STRIPE LAYOUT ↔ TILE-LOCAL LAYOUT
//==============================================================================


#define RADIX32_NUM_STRIPES 32  // Radix-32 produces 32 output stripes

/**
 * @brief Gather input data from 32 stripes into contiguous tile
 *
 * Converts from blocked stripe layout to tile-local contiguous layout:
 *   Source: in[stripe][k_offset..k_offset+tile_size-1]  (32 stripes, stride K)
 *   Dest:   tile[stripe][0..tile_size-1]                (contiguous per stripe)
 *
 * Memory pattern:
 *   - 32 separate reads (one per stripe), each contiguous
 *   - Compiler vectorizes inner loop effectively
 *   - Total: tile_size × 32 × 16 bytes (re+im) = 32KB @ tile_size=64
 *
 * @param in_re Source real [32 stripes][K]
 * @param in_im Source imag [32 stripes][K]
 * @param K Total butterflies per stage
 * @param k_offset Starting position in K dimension
 * @param tile_size Number of samples to gather (≤ RADIX32_TILE_SIZE)
 * @param tile_re Destination real [32][tile_size]
 * @param tile_im Destination imag [32][tile_size]
 */
TARGET_AVX512
FORCE_INLINE void radix32_gather_stripes_to_tile(
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    size_t K,
    size_t k_offset,
    size_t tile_size,
    double *RESTRICT tile_re,
    double *RESTRICT tile_im)
{
    // Gather each of 32 stripes into contiguous tile-local buffer
    for (int stripe = 0; stripe < RADIX32_NUM_STRIPES; stripe++)
    {
        const double *src_re = &in_re[stripe * K + k_offset];
        const double *src_im = &in_im[stripe * K + k_offset];
        double *dst_re = &tile_re[stripe * tile_size];
        double *dst_im = &tile_im[stripe * tile_size];

        // Vectorized copy (compiler auto-vectorizes, or use explicit SIMD)
        size_t i = 0;

        // Main loop: process 8 doubles at a time
        for (; i + 8 <= tile_size; i += 8)
        {
            __m512d vr = _mm512_loadu_pd(&src_re[i]);
            __m512d vi = _mm512_loadu_pd(&src_im[i]);
            _mm512_store_pd(&dst_re[i], vr);
            _mm512_store_pd(&dst_im[i], vi);
        }

        // Tail: handle remaining samples (scalar fallback)
        for (; i < tile_size; i++)
        {
            dst_re[i] = src_re[i];
            dst_im[i] = src_im[i];
        }
    }
}

/**
 * @brief Gather with software prefetch for large K
 *
 * Identical to radix32_gather_stripes_to_tile but adds prefetch hints
 * for the next tile to hide memory latency.
 *
 * @param in_re Source real [32 stripes][K]
 * @param in_im Source imag [32 stripes][K]
 * @param K Total butterflies per stage
 * @param k_offset Starting position in K dimension
 * @param tile_size Number of samples to gather
 * @param prefetch_offset Offset for next tile prefetch (in samples)
 * @param tile_re Destination real [32][tile_size]
 * @param tile_im Destination imag [32][tile_size]
 */
TARGET_AVX512
FORCE_INLINE void radix32_gather_stripes_to_tile_prefetch(
    const double *RESTRICT in_re,
    const double *RESTRICT in_im,
    size_t K,
    size_t k_offset,
    size_t tile_size,
    size_t prefetch_offset,
    double *RESTRICT tile_re,
    double *RESTRICT tile_im)
{
    for (int stripe = 0; stripe < RADIX32_NUM_STRIPES; stripe++)
    {
        const double *src_re = &in_re[stripe * K + k_offset];
        const double *src_im = &in_im[stripe * K + k_offset];
        double *dst_re = &tile_re[stripe * tile_size];
        double *dst_im = &tile_im[stripe * tile_size];

        // Prefetch next tile if within bounds
        if (k_offset + prefetch_offset < K)
        {
            _mm_prefetch((const char *)&in_re[stripe * K + k_offset + prefetch_offset], _MM_HINT_T0);
            _mm_prefetch((const char *)&in_im[stripe * K + k_offset + prefetch_offset], _MM_HINT_T0);
        }

        // Vectorized copy
        size_t i = 0;
        for (; i + 8 <= tile_size; i += 8)
        {
            __m512d vr = _mm512_loadu_pd(&src_re[i]);
            __m512d vi = _mm512_loadu_pd(&src_im[i]);
            _mm512_store_pd(&dst_re[i], vr);
            _mm512_store_pd(&dst_im[i], vi);
        }

        for (; i < tile_size; i++)
        {
            dst_re[i] = src_re[i];
            dst_im[i] = src_im[i];
        }
    }
}

/**
 * @brief Scatter tile-local data back to 32 stripes (normal stores)
 *
 * Converts from tile-local contiguous layout back to blocked stripes:
 *   Source: tile[stripe][0..tile_size-1]                (contiguous per stripe)
 *   Dest:   out[stripe][k_offset..k_offset+tile_size-1] (32 stripes, stride K)
 *
 * @param tile_re Source real [32][tile_size]
 * @param tile_im Source imag [32][tile_size]
 * @param tile_size Number of samples to scatter
 * @param k_offset Starting position in K dimension
 * @param K Total butterflies per stage
 * @param out_re Destination real [32 stripes][K]
 * @param out_im Destination imag [32 stripes][K]
 */
TARGET_AVX512
FORCE_INLINE void radix32_scatter_tile_to_stripes(
    const double *RESTRICT tile_re,
    const double *RESTRICT tile_im,
    size_t tile_size,
    size_t k_offset,
    size_t K,
    double *RESTRICT out_re,
    double *RESTRICT out_im)
{
    for (int stripe = 0; stripe < RADIX32_NUM_STRIPES; stripe++)
    {
        const double *src_re = &tile_re[stripe * tile_size];
        const double *src_im = &tile_im[stripe * tile_size];
        double *dst_re = &out_re[stripe * K + k_offset];
        double *dst_im = &out_im[stripe * K + k_offset];

        // Vectorized copy
        size_t i = 0;
        for (; i + 8 <= tile_size; i += 8)
        {
            __m512d vr = _mm512_load_pd(&src_re[i]);
            __m512d vi = _mm512_load_pd(&src_im[i]);
            _mm512_storeu_pd(&dst_re[i], vr);
            _mm512_storeu_pd(&dst_im[i], vi);
        }

        for (; i < tile_size; i++)
        {
            dst_re[i] = src_re[i];
            dst_im[i] = src_im[i];
        }
    }
}

/**
 * @brief Scatter with non-temporal stores for large K
 *
 * Uses streaming stores (_mm512_stream_pd) to avoid cache pollution
 * when output data won't be read back soon.
 *
 * CRITICAL: Requires 64-byte alignment of destination addresses.
 * Use only when K > RADIX32_NT_THRESHOLD and outputs are aligned.
 *
 * @param tile_re Source real [32][tile_size]
 * @param tile_im Source imag [32][tile_size]
 * @param tile_size Number of samples to scatter
 * @param k_offset Starting position in K dimension
 * @param K Total butterflies per stage
 * @param out_re Destination real [32 stripes][K] (64-byte aligned)
 * @param out_im Destination imag [32 stripes][K] (64-byte aligned)
 */
TARGET_AVX512
FORCE_INLINE void radix32_scatter_tile_to_stripes_nt(
    const double *RESTRICT tile_re,
    const double *RESTRICT tile_im,
    size_t tile_size,
    size_t k_offset,
    size_t K,
    double *RESTRICT out_re,
    double *RESTRICT out_im)
{
    for (int stripe = 0; stripe < RADIX32_NUM_STRIPES; stripe++)
    {
        const double *src_re = &tile_re[stripe * tile_size];
        const double *src_im = &tile_im[stripe * tile_size];
        double *dst_re = &out_re[stripe * K + k_offset];
        double *dst_im = &out_im[stripe * K + k_offset];

        // Non-temporal streaming stores (requires alignment)
        size_t i = 0;

        // Ensure alignment: k_offset must be 8-aligned for NT stores
        // If not aligned, use regular stores for first few elements
        size_t align_offset = ((uintptr_t)dst_re & 63) / 8; // Doubles to alignment
        if (align_offset > 0)
        {
            size_t align_count = (8 - align_offset) & 7;
            for (i = 0; i < align_count && i < tile_size; i++)
            {
                dst_re[i] = src_re[i];
                dst_im[i] = src_im[i];
            }
        }

        // Main loop: NT stores (8 doubles at a time)
        for (; i + 8 <= tile_size; i += 8)
        {
            __m512d vr = _mm512_load_pd(&src_re[i]);
            __m512d vi = _mm512_load_pd(&src_im[i]);
            _mm512_stream_pd(&dst_re[i], vr);
            _mm512_stream_pd(&dst_im[i], vi);
        }

        // Tail: regular stores
        for (; i < tile_size; i++)
        {
            dst_re[i] = src_re[i];
            dst_im[i] = src_im[i];
        }
    }

    // Ensure NT stores are visible before function returns
    _mm_sfence();
}

/**
 * @brief Adaptive scatter dispatcher
 *
 * Selects between normal and non-temporal stores based on plan configuration.
 *
 * @param tile_re Source real [32][tile_size]
 * @param tile_im Source imag [32][tile_size]
 * @param tile_size Number of samples to scatter
 * @param k_offset Starting position in K dimension
 * @param K Total butterflies per stage
 * @param out_re Destination real [32 stripes][K]
 * @param out_im Destination imag [32 stripes][K]
 * @param use_nt True to use non-temporal stores
 */
TARGET_AVX512
FORCE_INLINE void radix32_scatter_tile_to_stripes_auto(
    const double *RESTRICT tile_re,
    const double *RESTRICT tile_im,
    size_t tile_size,
    size_t k_offset,
    size_t K,
    double *RESTRICT out_re,
    double *RESTRICT out_im,
    bool use_nt)
{
    if (use_nt)
    {
        radix32_scatter_tile_to_stripes_nt(tile_re, tile_im, tile_size,
                                           k_offset, K, out_re, out_im);
    }
    else
    {
        radix32_scatter_tile_to_stripes(tile_re, tile_im, tile_size,
                                        k_offset, K, out_re, out_im);
    }
}