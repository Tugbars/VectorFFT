/**
 * @file fft_radix2_fv.c
 * @brief TRUE END-TO-END SoA Radix-2 FFT Implementation
 * 
 * @details
 * This module implements a native Structure-of-Arrays (SoA) radix-2 FFT that
 * eliminates split/join operations at stage boundaries, achieving significant
 * performance improvements over traditional Array-of-Structures approaches.
 * 
 * @author Tugbars
 * @version 2.2 (Optimized with N/8 paths and aligned loads)
 * @date 2025
 */

#include "fft_radix2_uniform.h"
#include "fft_radix2_macros.h"

#include <immintrin.h>  // For SIMD intrinsics
#include <assert.h>     // For assertions
#include <stdint.h>     // For uintptr_t (alignment checks)
#include <stdlib.h>     // For getenv (environment variable)

//==============================================================================
// CONFIGURATION
//==============================================================================

/// Cache line size in bytes (typical for x86-64)
#define CACHE_LINE_BYTES 64

/**
 * @brief Required alignment based on SIMD instruction set
 * @details Alignment must match the widest SIMD vector being used:
 *          - AVX-512: 64 bytes (512 bits)
 *          - AVX/AVX2: 32 bytes (256 bits)
 *          - SSE2: 16 bytes (128 bits)
 *          - Scalar: 8 bytes (natural double alignment)
 */
#if defined(__AVX512F__)
    #define REQUIRED_ALIGNMENT 64
    #define VECTOR_WIDTH 8  ///< Doubles per SIMD vector (AVX-512)
#elif defined(__AVX2__) || defined(__AVX__)
    #define REQUIRED_ALIGNMENT 32
    #define VECTOR_WIDTH 4  ///< Doubles per SIMD vector (AVX2)
#elif defined(__SSE2__)
    #define REQUIRED_ALIGNMENT 16
    #define VECTOR_WIDTH 2  ///< Doubles per SIMD vector (SSE2)
#else
    #define REQUIRED_ALIGNMENT 8
    #define VECTOR_WIDTH 1  ///< Scalar (no SIMD)
#endif

/**
 * @brief Last Level Cache size in bytes
 * @details Conservative default: 8 MB. For better performance, detect at 
 *          runtime or compile with -DLLC_BYTES=...
 */
#ifndef LLC_BYTES
    #define LLC_BYTES (8 * 1024 * 1024)
#endif

/**
 * @brief Non-temporal store threshold as fraction of LLC
 * @details Enable NT when write footprint exceeds this fraction of LLC
 */
#define NT_THRESHOLD 0.7

/**
 * @brief Minimum half-size for enabling non-temporal stores
 * @details Avoid NT overhead for very small writes (< ~32 KB per array)
 */
#define NT_MIN_HALF 4096

//==============================================================================
// HELPER: Environment Variable Parsing
//==============================================================================

/**
 * @brief Check for FFT_NT environment variable override
 * 
 * @details
 * Allows runtime control of non-temporal stores:
 *   - FFT_NT=0: Force NT stores OFF (always use normal stores)
 *   - FFT_NT=1: Force NT stores ON (if alignment allows)
 *   - Not set or other: Use automatic heuristic
 * 
 * @return 0 = force off, 1 = force on, -1 = use heuristic
 */
static inline int check_nt_env_override(void)
{
    static int cached_value = -2;  // -2 = not yet checked
    
    if (cached_value == -2)
    {
        const char *env = getenv("FFT_NT");
        if (env == NULL)
        {
            cached_value = -1;  // Use heuristic
        }
        else if (env[0] == '0')
        {
            cached_value = 0;   // Force off
        }
        else if (env[0] == '1')
        {
            cached_value = 1;   // Force on
        }
        else
        {
            cached_value = -1;  // Unknown value, use heuristic
        }
    }
    
    return cached_value;
}

//==============================================================================
// HELPER: Process a Range of Butterflies (Native SoA)
//==============================================================================

/**
 * @brief Process radix-2 butterflies in range [k_start, k_end) - NATIVE SoA
 * 
 * Data flow:
 *   - Load: in_re[k], in_im[k] (direct, no conversion!)
 *   - Compute: butterfly in split form
 *   - Store: out_re[k], out_im[k] (direct, no conversion!)
 * 
 * This function processes a contiguous range of butterfly indices, applying
 * the optimal SIMD path for the target architecture. It automatically falls
 * back through AVX-512 → AVX2 → SSE2 → Scalar as needed.
 * 
 * @param[out] out_re Output real array
 * @param[out] out_im Output imaginary array
 * @param[in] in_re Input real array
 * @param[in] in_im Input imaginary array
 * @param[in] stage_tw SoA twiddle factors for this stage
 * @param[in] half Transform half-size (N/2 for this stage)
 * @param[in] k_start Starting butterfly index (inclusive)
 * @param[in] k_end Ending butterfly index (exclusive)
 * @param[in] use_streaming Use streaming stores for large N
 * 
 * @pre k_start >= 0
 * @pre k_end <= half
 * @pre k_start < k_end
 * @pre All pointers non-NULL
 * @pre out_re != in_re && out_im != in_im (out-of-place required)
 */
static void radix2_process_range_native_soa(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw,
    int half,
    int k_start,
    int k_end,
    int use_streaming)
{
    // Prefetch distance for software prefetching
    const int prefetch_dist = RADIX2_PREFETCH_DISTANCE;
    
    int k = k_start;
    
    //==========================================================================
    // ALIGNMENT PEELING (for non-temporal stores)
    //==========================================================================
    // If NT stores are enabled, peel scalar iterations until both output AND
    // input addresses are vector-aligned. This ensures:
    // 1. Every _mm*_stream_pd hits naturally aligned addresses
    // 2. All aligned input loads in STREAM macros work correctly
    // 3. No cache-line straddling for loads or stores
    
    if (use_streaming)
    {
        const size_t vec_align = VECTOR_WIDTH * sizeof(double);
        
        // Peel until ALL of {in_re[k], in_im[k], out_re[k], out_im[k]} are aligned
        while (k < k_end && 
               ((((uintptr_t)&in_re[k]) % vec_align) != 0 ||
                (((uintptr_t)&in_im[k]) % vec_align) != 0 ||
                (((uintptr_t)&out_re[k]) % vec_align) != 0 ||
                (((uintptr_t)&out_im[k]) % vec_align) != 0))
        {
            RADIX2_PIPELINE_1_NATIVE_SOA_SCALAR(k, in_re, in_im, out_re, out_im,
                                                stage_tw, half);
            k++;
        }
    }
    
#ifdef __AVX512F__
    // ========================================================================
    // AVX-512: UNROLLED 2× + DOUBLE-PUMPED (process 16 butterflies/iteration)
    // ========================================================================
    if (use_streaming) {
        // Process 16 butterflies per iteration (2× unroll)
        for (; k + 15 < k_end; k += 16) {
            // Pipeline 0: butterflies [k, k+7]
            RADIX2_PIPELINE_8_NATIVE_SOA_AVX512_STREAM(
                k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
            
            // Pipeline 1: butterflies [k+8, k+15] (independent, can overlap)
            RADIX2_PIPELINE_8_NATIVE_SOA_AVX512_STREAM(
                k + 8, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
        }
        
        // Cleanup: process remaining 8 if any
        if (k + 7 < k_end) {
            RADIX2_PIPELINE_8_NATIVE_SOA_AVX512_STREAM(
                k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
            k += 8;
        }
    } else {
        // Non-streaming: same unrolling pattern
        for (; k + 15 < k_end; k += 16) {
            RADIX2_PIPELINE_8_NATIVE_SOA_AVX512(
                k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
            RADIX2_PIPELINE_8_NATIVE_SOA_AVX512(
                k + 8, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
        }
        
        if (k + 7 < k_end) {
            RADIX2_PIPELINE_8_NATIVE_SOA_AVX512(
                k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
            k += 8;
        }
    }
#endif
    
#ifdef __AVX2__
    // ========================================================================
    // AVX2: UNROLLED 2× + DOUBLE-PUMPED (process 8 butterflies/iteration)
    // ========================================================================
    if (use_streaming) {
        for (; k + 7 < k_end; k += 8) {
            // Pipeline 0: butterflies [k, k+3]
            RADIX2_PIPELINE_4_NATIVE_SOA_AVX2_STREAM(
                k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
            
            // Pipeline 1: butterflies [k+4, k+7]
            RADIX2_PIPELINE_4_NATIVE_SOA_AVX2_STREAM(
                k + 4, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
        }
        
        if (k + 3 < k_end) {
            RADIX2_PIPELINE_4_NATIVE_SOA_AVX2_STREAM(
                k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
            k += 4;
        }
    } else {
        for (; k + 7 < k_end; k += 8) {
            RADIX2_PIPELINE_4_NATIVE_SOA_AVX2(
                k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
            RADIX2_PIPELINE_4_NATIVE_SOA_AVX2(
                k + 4, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
        }
        
        if (k + 3 < k_end) {
            RADIX2_PIPELINE_4_NATIVE_SOA_AVX2(
                k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
            k += 4;
        }
    }
#endif

#ifdef __SSE2__
    // ========================================================================
    // SSE2: UNROLLED 2× + DOUBLE-PUMPED (process 4 butterflies/iteration)
    // ========================================================================
    if (use_streaming) {
        for (; k + 3 < k_end; k += 4) {
            RADIX2_PIPELINE_2_NATIVE_SOA_SSE2_STREAM(
                k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
            RADIX2_PIPELINE_2_NATIVE_SOA_SSE2_STREAM(
                k + 2, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
        }
        
        if (k + 1 < k_end) {
            RADIX2_PIPELINE_2_NATIVE_SOA_SSE2_STREAM(
                k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
            k += 2;
        }
    } else {
        for (; k + 3 < k_end; k += 4) {
            RADIX2_PIPELINE_2_NATIVE_SOA_SSE2(
                k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
            RADIX2_PIPELINE_2_NATIVE_SOA_SSE2(
                k + 2, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
        }
        
        if (k + 1 < k_end) {
            RADIX2_PIPELINE_2_NATIVE_SOA_SSE2(
                k, in_re, in_im, out_re, out_im, stage_tw, half, prefetch_dist);
            k += 2;
        }
    }
#endif

    //==========================================================================
    // Scalar Cleanup (remaining butterflies)
    //==========================================================================
    
    while (k < k_end)
    {
        RADIX2_PIPELINE_1_NATIVE_SOA_SCALAR(k, in_re, in_im, out_re, out_im,
                                            stage_tw, half);
        k++;
    }
}

//==============================================================================
// MAIN FUNCTION: Radix-2 DIF Butterfly - NATIVE SoA
//==============================================================================

/**
 * @brief Radix-2 DIF butterfly - NATIVE SoA (ZERO SHUFFLE IN HOT PATH!)
 * 
 * ALGORITHM:
 * @code
 *   For k = 0 to half-1:
 *     W[k] = exp(-2πi·k/N)  (from stage_tw, already SoA)
 *     y_re[k]      = x_re[k] + W_re[k]·x_re[k+half] - W_im[k]·x_im[k+half]
 *     y_im[k]      = x_im[k] + W_re[k]·x_im[k+half] + W_im[k]·x_re[k+half]
 *     y_re[k+half] = x_re[k] - W_re[k]·x_re[k+half] + W_im[k]·x_im[k+half]
 *     y_im[k+half] = x_im[k] - W_re[k]·x_im[k+half] - W_im[k]·x_re[k+half]
 * @endcode
 * 
 * MEMORY LAYOUT:
 *   - Input:  in_re[0..N-1], in_im[0..N-1]   (separate arrays)
 *   - Output: out_re[0..N-1], out_im[0..N-1] (separate arrays)
 *   - Twiddles: stage_tw->re[0..half-1], stage_tw->im[0..half-1] (SoA)
 * 
 * @param[out] out_re Output real array (N elements)
 * @param[out] out_im Output imaginary array (N elements)
 * @param[in] in_re Input real array (N elements)
 * @param[in] in_im Input imaginary array (N elements)
 * @param[in] stage_tw Stage twiddles (SoA format)
 * @param[in] half Transform half-size (N/2)
 * 
 * @pre Output buffers aligned to REQUIRED_ALIGNMENT (SIMD-dependent):
 *      - AVX-512: 64-byte alignment
 *      - AVX2: 32-byte alignment
 *      - SSE2: 16-byte alignment
 * @pre Twiddle factors aligned to REQUIRED_ALIGNMENT
 * 
 * 
 */
void fft_radix2_fv(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw,
    int half)
{
    //==========================================================================
    // ALIGNMENT HINTS (GCC/Clang optimization)
    //==========================================================================
    // Tell the compiler that pointers are aligned, enabling better code generation
    
#if defined(__GNUC__) || defined(__clang__)
    in_re  = (const double*)__builtin_assume_aligned(in_re,  REQUIRED_ALIGNMENT);
    in_im  = (const double*)__builtin_assume_aligned(in_im,  REQUIRED_ALIGNMENT);
    out_re = (double*)__builtin_assume_aligned(out_re, REQUIRED_ALIGNMENT);
    out_im = (double*)__builtin_assume_aligned(out_im, REQUIRED_ALIGNMENT);
    stage_tw->re = (const double*)__builtin_assume_aligned(stage_tw->re, REQUIRED_ALIGNMENT);
    stage_tw->im = (const double*)__builtin_assume_aligned(stage_tw->im, REQUIRED_ALIGNMENT);
#endif

    //==========================================================================
    // NON-TEMPORAL STORE HEURISTIC (safe with fallback)
    //==========================================================================
    // Enable NT stores based on:
    //   1. Environment variable FFT_NT (0=off, 1=on, unset=heuristic)
    //   2. Execution is out-of-place (in != out)
    //   3. Per-stage write footprint > 70% of LLC
    //   4. half >= 4096 (avoid NT overhead for small writes)
    //   5. Output buffers are properly aligned (runtime check with fallback)
    // 
    // Per-stage writes: 2 arrays (re, im) × half elements × sizeof(double)
    // No cross-call hysteresis needed: half shrinks monotonically in radix-2,
    // so we cross threshold once and stay disabled.
    
    int nt_env_override = check_nt_env_override();
    
    const size_t write_footprint = 2ull * half * sizeof(double);
    const int is_out_of_place = (in_re != out_re) && (in_im != out_im);
    
    int use_streaming = 0;
    
    if (nt_env_override == 0)
    {
        // FFT_NT=0: Force off
        use_streaming = 0;
    }
    else if (nt_env_override == 1)
    {
        // FFT_NT=1: Force on (if out-of-place and aligned)
        use_streaming = is_out_of_place;
    }
    else
    {
        // FFT_NT not set or invalid: Use automatic heuristic
        use_streaming = is_out_of_place && 
                       (half >= NT_MIN_HALF) &&
                       (write_footprint > (size_t)(NT_THRESHOLD * LLC_BYTES));
    }
    
    // ⚡ CRITICAL: Runtime alignment check with fallback
    // Don't rely on assert (it's a no-op in release builds)
    // If misaligned, downgrade to normal stores instead of crashing
    if (use_streaming)
    {
        uintptr_t r0 = (uintptr_t)&out_re[0];
        uintptr_t i0 = (uintptr_t)&out_im[0];
        
        if ((r0 % REQUIRED_ALIGNMENT) != 0 || (i0 % REQUIRED_ALIGNMENT) != 0)
        {
            // Misaligned: fallback to normal stores
            use_streaming = 0;
        }
    }
    
    // Verify twiddle alignment (these should always be aligned)
    // Keep assertions for twiddles since user confirmed they're always aligned
    assert(((uintptr_t)stage_tw->re % REQUIRED_ALIGNMENT) == 0 && 
           "stage_tw->re must be properly aligned for SIMD");
    assert(((uintptr_t)stage_tw->im % REQUIRED_ALIGNMENT) == 0 && 
           "stage_tw->im must be properly aligned for SIMD");
    
    //==========================================================================
    // DETERMINE k=N/4, k=N/8, k=3N/8 SPECIAL CASES
    //==========================================================================
    
    int k_quarter = 0;
    int k_eighth = 0;
    int k_3eighth = 0;
    
    if ((half & (half - 1)) == 0)  // Power of 2
    {
        k_quarter = half / 2;  // N/4
        
        // N/8 and 3N/8 only exist when N is divisible by 8 (i.e., half divisible by 4)
        if ((half & 3) == 0)
        {
            k_eighth = half >> 2;       // N/8
            k_3eighth = (3 * half) >> 2;  // 3N/8
        }
    }

    //==========================================================================
    // SPECIAL CASE: k=0 (W[0] = 1, no multiply needed)
    //==========================================================================
    // ⚠️  Only ONE butterfly at k=0 - scalar is fine
    
    radix2_k0_native_soa_scalar(in_re, in_im, out_re, out_im, half);

    //==========================================================================
    // SPECIAL CASE: k=N/8 (W = √2/2 - i√2/2, optimized multiply)
    //==========================================================================
    // ⚠️  Only ONE butterfly at k=N/8 - scalar is fine
    
    if (k_eighth > 0 && k_eighth < half)
    {
        radix2_k_eighth_native_soa_scalar(in_re, in_im, out_re, out_im, 
                                          k_eighth, half, +1);
    }

    //==========================================================================
    // SPECIAL CASE: k=N/4 (W[N/4] = -i, specialized rotation)
    //==========================================================================
    // ⚠️  Only ONE butterfly at k=N/4 - scalar is fine
    
    if (k_quarter > 0 && k_quarter < half)
    {
        radix2_k_quarter_native_soa_scalar(in_re, in_im, out_re, out_im, 
                                           k_quarter, half);
    }

    //==========================================================================
    // SPECIAL CASE: k=3N/8 (W = -√2/2 - i√2/2, optimized multiply)
    //==========================================================================
    // ⚠️  Only ONE butterfly at k=3N/8 - scalar is fine
    
    if (k_3eighth > 0 && k_3eighth < half)
    {
        radix2_k_eighth_native_soa_scalar(in_re, in_im, out_re, out_im, 
                                          k_3eighth, half, -1);
    }

    //==========================================================================
    // GENERAL CASE: All other k values
    //==========================================================================
    // Build list of ranges to process, excluding special cases
    
    typedef struct {
        int start;
        int end;
    } range_t;
    
    range_t ranges[4];  // Maximum 4 ranges possible
    int num_ranges = 0;
    
    // Build ranges: split around k=0, k_eighth, k_quarter, k_3eighth
    int current = 1;  // Start after k=0
    
    // Range before k_eighth
    if (k_eighth > 0 && k_eighth > current)
    {
        ranges[num_ranges].start = current;
        ranges[num_ranges].end = k_eighth;
        num_ranges++;
        current = k_eighth + 1;
    }
    
    // Range before k_quarter
    if (k_quarter > 0 && k_quarter > current)
    {
        ranges[num_ranges].start = current;
        ranges[num_ranges].end = k_quarter;
        num_ranges++;
        current = k_quarter + 1;
    }
    
    // Range before k_3eighth
    if (k_3eighth > 0 && k_3eighth > current)
    {
        ranges[num_ranges].start = current;
        ranges[num_ranges].end = k_3eighth;
        num_ranges++;
        current = k_3eighth + 1;
    }
    
    // Final range to half
    if (current < half)
    {
        ranges[num_ranges].start = current;
        ranges[num_ranges].end = half;
        num_ranges++;
    }
    
    //==========================================================================
    // PROCESS RANGES SEQUENTIALLY
    //==========================================================================
    // Threading module should call radix2_process_range_native_soa() directly
    // for parallel execution. This function is the single-threaded core.
    
    for (int i = 0; i < num_ranges; i++)
    {
        radix2_process_range_native_soa(out_re, out_im, in_re, in_im,
                                       stage_tw, half, 
                                       ranges[i].start, ranges[i].end,
                                       use_streaming);
    }
}