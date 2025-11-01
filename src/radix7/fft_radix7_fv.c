/**
 * @file fft_radix7_fv.c
 * @brief TRUE END-TO-END SoA Radix-7 FFT Implementation - Forward
 *
 * @details
 * This module implements native SoA radix-7 forward FFT using Rader's algorithm:
 * - Standard twiddle version (fft_radix7_fv) for general stages
 * - Twiddle-less n1 version (fft_radix7_fv_n1) for first stage
 *
 * Architecture support (cascading cleanup):
 * - AVX-512: 8-wide, U2 pipeline (16 butterflies/iter)
 * - AVX2:    4-wide, U2 pipeline (8 butterflies/iter)
 * - SSE2:    2-wide, U2 pipeline (4 butterflies/iter)
 * - Scalar:  1-wide, fallback for remainder
 *
 * @author Tugbars
 * @version 4.0 (Generation 3 SoA + U2)
 * @date 2025
 */

#include "fft_radix7_uniform.h"

// Standard twiddle implementations (with Rader)
#ifdef __AVX512F__
#include "fft_radix7_avx512.h"
#endif

#ifdef __AVX2__
#include "fft_radix7_avx2.h"
#endif

#include "fft_radix7_sse2.h"   // Always available on x86-64
#include "fft_radix7_scalar.h" // Universal fallback

// N1 (twiddle-less) implementations - TODO: Add when ready
// #ifdef __AVX512F__
//     #include "fft_radix7_avx512_n1.h"
// #endif
// #ifdef __AVX2__
//     #include "fft_radix7_avx2_n1.h"
// #endif
// #include "fft_radix7_sse2_n1.h"
// #include "fft_radix7_scalar_n1.h"

#include <immintrin.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

//==============================================================================
// CONFIGURATION
//==============================================================================

#define CACHE_LINE_BYTES 64

#if defined(__AVX512F__)
#define REQUIRED_ALIGNMENT 64
#define VECTOR_WIDTH 8
#define U2_WIDTH 16
#elif defined(__AVX2__)
#define REQUIRED_ALIGNMENT 32
#define VECTOR_WIDTH 4
#define U2_WIDTH 8
#elif defined(__SSE2__)
#define REQUIRED_ALIGNMENT 16
#define VECTOR_WIDTH 2
#define U2_WIDTH 4
#else
#define REQUIRED_ALIGNMENT 8
#define VECTOR_WIDTH 1
#define U2_WIDTH 1
#endif

#ifndef LLC_BYTES
#define LLC_BYTES (8 * 1024 * 1024)
#endif

#define NT_THRESHOLD 0.7
#define NT_MIN_K 4096
#define LARGE_STAGE_K 2048

//==============================================================================
// HELPER: Environment Variable Parsing
//==============================================================================

static inline int check_nt_env_override(void)
{
    static int cached_value = -2;

    if (cached_value == -2)
    {
        const char *env = getenv("FFT_R7_NT");
        if (env == NULL)
        {
            cached_value = -1; // No override
        }
        else if (env[0] == '0')
        {
            cached_value = 0; // Force disable
        }
        else if (env[0] == '1')
        {
            cached_value = 1; // Force enable
        }
        else
        {
            cached_value = -1; // Invalid value, no override
        }
    }

    return cached_value;
}

//==============================================================================
// HELPER: Process Range (Native SoA) - WITH TWIDDLES - FORWARD
//==============================================================================

/**
 * @brief Process range [k_start, k_end) with cascading SIMD - FORWARD
 *
 * @details
 * Cascading pattern:
 * 1. AVX-512 (if compiled): U2 loop (16/iter) → single (8/iter) → scalar
 * 2. AVX2 (if compiled):    U2 loop (8/iter)  → single (4/iter) → scalar
 * 3. SSE2 (always):         U2 loop (4/iter)  → single (2/iter) → scalar
 *
 * Forward FFT: Uses conjugated twiddles or different algorithm
 *
 * @param k_start Starting butterfly index
 * @param k_end Ending butterfly index (exclusive)
 * @param in_re Input real array (7K elements)
 * @param in_im Input imaginary array (7K elements)
 * @param out_re Output real array (7K elements)
 * @param out_im Output imaginary array (7K elements)
 * @param stage_tw Stage twiddle factors (6 complex, blocked SoA)
 * @param rader_tw_re Broadcast Rader twiddle real components (6 vectors)
 * @param rader_tw_im Broadcast Rader twiddle imaginary components (6 vectors)
 * @param K Transform size (N/7)
 * @param sub_len Sub-transform length
 * @param use_nt Use non-temporal stores
 * @param large_stage Use T1 prefetch hint for twiddles
 */
static void radix7_process_range_soa_fv(
    int k_start,
    int k_end,
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
    const fft_twiddle_soa *restrict stage_tw,
#ifdef __AVX512F__
    const __m512d rader_tw_re[6],
    const __m512d rader_tw_im[6],
#elif defined(__AVX2__)
    const __m256d rader_tw_re[6],
    const __m256d rader_tw_im[6],
#else
    const __m128d rader_tw_re[6],
    const __m128d rader_tw_im[6],
#endif
    int K,
    int sub_len,
    bool use_nt,
    bool large_stage)
{
    int k = k_start;

#ifdef __AVX512F__
    // AVX-512: Process in chunks of 16 (U2: two 8-wide butterflies)
    for (; k + 15 < k_end; k += 16)
    {
        // Prefetch ahead
        prefetch_7_lanes_avx512_soa(k, K, in_re, in_im, stage_tw, sub_len, large_stage);

        // Process two butterflies simultaneously: k and k+8
        radix7_butterfly_dual_avx512_soa(k, k + 8, K,
                                         in_re, in_im, stage_tw,
                                         rader_tw_re, rader_tw_im,
                                         out_re, out_im, sub_len, use_nt);
    }

    // Cleanup: Single 8-wide butterfly
    if (k + 7 < k_end)
    {
        radix7_butterfly_single_avx512_soa(k, K, in_re, in_im, stage_tw,
                                           rader_tw_re, rader_tw_im,
                                           out_re, out_im, sub_len, use_nt);
        k += 8;
    }

#elif defined(__AVX2__)
    // AVX2: Process in chunks of 8 (U2: two 4-wide butterflies)
    for (; k + 7 < k_end; k += 8)
    {
        // Prefetch ahead
        prefetch_7_lanes_avx2_soa(k, K, in_re, in_im, stage_tw, sub_len, large_stage);

        // Process two butterflies simultaneously: k and k+4
        radix7_butterfly_dual_avx2_soa(k, k + 4, K,
                                       in_re, in_im, stage_tw,
                                       rader_tw_re, rader_tw_im,
                                       out_re, out_im, sub_len, use_nt);
    }

    // Cleanup: Single 4-wide butterfly
    if (k + 3 < k_end)
    {
        radix7_butterfly_single_avx2_soa(k, K, in_re, in_im, stage_tw,
                                         rader_tw_re, rader_tw_im,
                                         out_re, out_im, sub_len, use_nt);
        k += 4;
    }

#else // SSE2
    // SSE2: Process in chunks of 4 (U2: two 2-wide butterflies)
    for (; k + 3 < k_end; k += 4)
    {
        // Prefetch ahead
        prefetch_7_lanes_sse2_soa(k, K, in_re, in_im, stage_tw, sub_len, large_stage);

        // Process two butterflies simultaneously: k and k+2
        radix7_butterfly_dual_sse2_soa(k, k + 2, K,
                                       in_re, in_im, stage_tw,
                                       rader_tw_re, rader_tw_im,
                                       out_re, out_im, sub_len, use_nt);
    }

    // Cleanup: Single 2-wide butterfly
    if (k + 1 < k_end)
    {
        radix7_butterfly_single_sse2_soa(k, K, in_re, in_im, stage_tw,
                                         rader_tw_re, rader_tw_im,
                                         out_re, out_im, sub_len, use_nt);
        k += 2;
    }
#endif

    // Scalar cleanup for final remaining butterflies
    while (k < k_end)
    {
        radix7_butterfly_scalar(k, K, in_re, in_im, stage_tw, rader_tw,
                                out_re, out_im, sub_len);
        k++;
    }
}

//==============================================================================
// MAIN FUNCTION: Radix-7 FFT Butterfly - NATIVE SoA - FORWARD (WITH TWIDDLES)
//==============================================================================

/**
 * @brief Radix-7 FFT butterfly - NATIVE SoA - Forward FFT
 *
 * @details
 * Standard radix-7 DIT butterfly using Rader's algorithm for prime radix.
 * Forward variant uses different twiddle signs or algorithm than backward.
 *
 * Algorithm (Rader + Cooley-Tukey):
 * 1. Load 7 lanes from input
 * 2. Apply stage twiddles (x0 unchanged, x1-x6 multiplied)
 * 3. Compute DC component (y0 = sum of all inputs, tree reduction)
 * 4. Permute x1-x6 for Rader [1,3,2,6,4,5]
 * 5. 6-point cyclic convolution (round-robin schedule)
 * 6. Assemble outputs with permutation [1,5,4,6,2,3]
 * 7. Store 7 lanes
 *
 * @param[out] out_re Output real array (7K elements)
 * @param[out] out_im Output imaginary array (7K elements)
 * @param[in] in_re Input real array (7K elements)
 * @param[in] in_im Input imaginary array (7K elements)
 * @param[in] stage_tw Stage twiddle factors W^k (6 complex, blocked SoA)
 * @param[in] rader_tw Rader twiddle factors (6 complex, SoA)
 * @param[in] K Transform size (N/7)
 * @param[in] sub_len Sub-transform length
 *
 * @note For first stage where all twiddles ≈ 1, use fft_radix7_fv_n1() instead
 */
void fft_radix7_fv(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddle_soa *restrict stage_tw,
    const fft_twiddle_soa *restrict rader_tw,
    int K,
    int sub_len)
{
    const int N = 7 * K;

    // Alignment hints for compiler optimization
#if defined(__GNUC__) || defined(__clang__)
    in_re = (const double *)__builtin_assume_aligned(in_re, REQUIRED_ALIGNMENT);
    in_im = (const double *)__builtin_assume_aligned(in_im, REQUIRED_ALIGNMENT);
    out_re = (double *)__builtin_assume_aligned(out_re, REQUIRED_ALIGNMENT);
    out_im = (double *)__builtin_assume_aligned(out_im, REQUIRED_ALIGNMENT);
    stage_tw->re = (const double *)__builtin_assume_aligned(stage_tw->re, REQUIRED_ALIGNMENT);
    stage_tw->im = (const double *)__builtin_assume_aligned(stage_tw->im, REQUIRED_ALIGNMENT);
    rader_tw->re = (const double *)__builtin_assume_aligned(rader_tw->re, REQUIRED_ALIGNMENT);
    rader_tw->im = (const double *)__builtin_assume_aligned(rader_tw->im, REQUIRED_ALIGNMENT);
#endif

    // Verify alignment (debug builds)
    assert(((uintptr_t)in_re % REQUIRED_ALIGNMENT) == 0 &&
           "in_re must be properly aligned for SIMD");
    assert(((uintptr_t)in_im % REQUIRED_ALIGNMENT) == 0 &&
           "in_im must be properly aligned for SIMD");
    assert(((uintptr_t)out_re % REQUIRED_ALIGNMENT) == 0 &&
           "out_re must be properly aligned for SIMD");
    assert(((uintptr_t)out_im % REQUIRED_ALIGNMENT) == 0 &&
           "out_im must be properly aligned for SIMD");
    assert(((uintptr_t)stage_tw->re % REQUIRED_ALIGNMENT) == 0 &&
           "stage_tw->re must be properly aligned for SIMD");
    assert(((uintptr_t)stage_tw->im % REQUIRED_ALIGNMENT) == 0 &&
           "stage_tw->im must be properly aligned for SIMD");

    // Broadcast Rader twiddles ONCE for entire stage (P0 optimization!)
#ifdef __AVX512F__
    __m512d rader_tw_re[6], rader_tw_im[6];
    broadcast_rader_twiddles_avx512_soa(rader_tw, rader_tw_re, rader_tw_im);
#elif defined(__AVX2__)
    __m256d rader_tw_re[6], rader_tw_im[6];
    broadcast_rader_twiddles_avx2_soa(rader_tw, rader_tw_re, rader_tw_im);
#else
    __m128d rader_tw_re[6], rader_tw_im[6];
    broadcast_rader_twiddles_sse2_soa(rader_tw, rader_tw_re, rader_tw_im);
#endif

    // Decide on non-temporal stores (LLC-aware heuristic)
    int nt_env_override = check_nt_env_override();

    const size_t bytes_per_stage = (size_t)K * 7 * 2 * sizeof(double); // 7 lanes, 2 components
    bool use_nt = false;

    if (nt_env_override == 0)
    {
        use_nt = false; // Force disable
    }
    else if (nt_env_override == 1)
    {
        use_nt = true; // Force enable
    }
    else
    {
        // Automatic decision based on working set
        use_nt = (bytes_per_stage > (size_t)(NT_THRESHOLD * LLC_BYTES)) &&
                 (K >= NT_MIN_K);
    }

    // Determine if stage is "large" for prefetch hint selection
    bool large_stage = (K >= LARGE_STAGE_K);

    // Process entire range [0, K)
    radix7_process_range_soa_fv(0, K, in_re, in_im, out_re, out_im,
                                stage_tw, rader_tw_re, rader_tw_im,
                                K, sub_len, use_nt, large_stage);

    // Memory fence if non-temporal stores were used
    if (use_nt)
    {
        _mm_sfence();
    }
}

//==============================================================================
// HELPER: Process Range (Native SoA) - NO TWIDDLES (N1) - FORWARD
//==============================================================================

/**
 * @brief Process range [k_start, k_end) with cascading SIMD - N1 FORWARD
 *
 * @details
 * N1 (twiddle-less) variant - NO stage twiddles applied!
 * Saves 6 complex multiplies per butterfly (~20-30% faster).
 *
 * Forward FFT: Uses forward-FFT Rader twiddles
 *
 * Cascading pattern:
 * 1. AVX-512 (if compiled): U2 loop (16/iter) → single (8/iter) → scalar
 * 2. AVX2 (if compiled):    U2 loop (8/iter)  → single (4/iter) → scalar
 * 3. SSE2 (always):         U2 loop (4/iter)  → single (2/iter) → scalar
 *
 * @param k_start Starting butterfly index
 * @param k_end Ending butterfly index (exclusive)
 * @param in_re Input real array (7K elements)
 * @param in_im Input imaginary array (7K elements)
 * @param out_re Output real array (7K elements)
 * @param out_im Output imaginary array (7K elements)
 * @param rader_tw_re Broadcast Rader twiddle real components (6 vectors)
 * @param rader_tw_im Broadcast Rader twiddle imaginary components (6 vectors)
 * @param K Transform size (N/7)
 * @param use_nt Use non-temporal stores
 */
static void radix7_process_range_soa_fv_n1(
    int k_start,
    int k_end,
    const double *restrict in_re,
    const double *restrict in_im,
    double *restrict out_re,
    double *restrict out_im,
#ifdef __AVX512F__
    const __m512d rader_tw_re[6],
    const __m512d rader_tw_im[6],
#elif defined(__AVX2__)
    const __m256d rader_tw_re[6],
    const __m256d rader_tw_im[6],
#else
    const __m128d rader_tw_re[6],
    const __m128d rader_tw_im[6],
#endif
    int K,
    bool use_nt)
{
    int k = k_start;

#ifdef __AVX512F__
    // AVX-512: Process in chunks of 16 (U2: two 8-wide butterflies)
    for (; k + 15 < k_end; k += 16)
    {
        // Prefetch ahead (N1 simplified - no stage twiddles!)
        prefetch_7_lanes_avx512_n1_soa(k, K, in_re, in_im);

        // Process two butterflies simultaneously: k and k+8
        radix7_butterfly_dual_avx512_n1_soa(k, k + 8, K,
                                            in_re, in_im,
                                            rader_tw_re, rader_tw_im,
                                            out_re, out_im, use_nt);
    }

    // Cleanup: Single 8-wide butterfly
    if (k + 7 < k_end)
    {
        radix7_butterfly_single_avx512_n1_soa(k, K, in_re, in_im,
                                              rader_tw_re, rader_tw_im,
                                              out_re, out_im, use_nt);
        k += 8;
    }

#elif defined(__AVX2__)
    // AVX2: Process in chunks of 8 (U2: two 4-wide butterflies)
    for (; k + 7 < k_end; k += 8)
    {
        // Prefetch ahead (N1 simplified - no stage twiddles!)
        prefetch_7_lanes_avx2_n1_soa(k, K, in_re, in_im);

        // Process two butterflies simultaneously: k and k+4
        radix7_butterfly_dual_avx2_n1_soa(k, k + 4, K,
                                          in_re, in_im,
                                          rader_tw_re, rader_tw_im,
                                          out_re, out_im, use_nt);
    }

    // Cleanup: Single 4-wide butterfly
    if (k + 3 < k_end)
    {
        radix7_butterfly_single_avx2_n1_soa(k, K, in_re, in_im,
                                            rader_tw_re, rader_tw_im,
                                            out_re, out_im, use_nt);
        k += 4;
    }

#else // SSE2
    // SSE2: Process in chunks of 4 (U2: two 2-wide butterflies)
    for (; k + 3 < k_end; k += 4)
    {
        // Prefetch ahead (N1 simplified - no stage twiddles!)
        prefetch_7_lanes_sse2_n1_soa(k, K, in_re, in_im);

        // Process two butterflies simultaneously: k and k+2
        radix7_butterfly_dual_sse2_n1_soa(k, k + 2, K,
                                          in_re, in_im,
                                          rader_tw_re, rader_tw_im,
                                          out_re, out_im, use_nt);
    }

    // Cleanup: Single 2-wide butterfly
    if (k + 1 < k_end)
    {
        radix7_butterfly_single_sse2_n1_soa(k, K, in_re, in_im,
                                            rader_tw_re, rader_tw_im,
                                            out_re, out_im, use_nt);
        k += 2;
    }
#endif

    // Scalar cleanup for final remaining butterflies
    while (k < k_end)
    {
        radix7_butterfly_scalar_n1(k, K, in_re, in_im, rader_tw,
                                   out_re, out_im);
        k++;
    }
}

//==============================================================================
// MAIN FUNCTION: Radix-7 FFT Butterfly - NATIVE SoA - FORWARD (NO TWIDDLES)
//==============================================================================

/**
 * @brief Radix-7 FFT butterfly - NATIVE SoA - Forward FFT - NO TWIDDLES (n1)
 *
 * @details
 * Twiddle-less variant for first radix-7 stage (forward) or when all stage
 * twiddles ≈ 1. Still uses Rader's algorithm but skips stage twiddle multiply.
 * ~20-30% faster than standard version.
 *
 * Algorithm (simplified):
 * 1. Load 7 lanes (no stage twiddles!)
 * 2. Compute DC component y0
 * 3. Permute + 6-point convolution (with Rader twiddles)
 * 4. Assemble + store
 *
 * @param[out] out_re Output real array (7K elements)
 * @param[out] out_im Output imaginary array (7K elements)
 * @param[in] in_re Input real array (7K elements)
 * @param[in] in_im Input imaginary array (7K elements)
 * @param[in] rader_tw Rader twiddle factors (6 complex, SoA)
 * @param[in] K Transform size (N/7)
 *
 * @note NO stage_tw parameter - assumes all W^(k*m) = 1
 * @note Still requires Rader twiddles for convolution
 */
void fft_radix7_fv_n1(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddle_soa *restrict rader_tw,
    int K)
{
    const int N = 7 * K;

    // Alignment hints for compiler optimization
#if defined(__GNUC__) || defined(__clang__)
    in_re = (const double *)__builtin_assume_aligned(in_re, REQUIRED_ALIGNMENT);
    in_im = (const double *)__builtin_assume_aligned(in_im, REQUIRED_ALIGNMENT);
    out_re = (double *)__builtin_assume_aligned(out_re, REQUIRED_ALIGNMENT);
    out_im = (double *)__builtin_assume_aligned(out_im, REQUIRED_ALIGNMENT);
    rader_tw->re = (const double *)__builtin_assume_aligned(rader_tw->re, REQUIRED_ALIGNMENT);
    rader_tw->im = (const double *)__builtin_assume_aligned(rader_tw->im, REQUIRED_ALIGNMENT);
#endif

    // Verify alignment (debug builds)
    assert(((uintptr_t)in_re % REQUIRED_ALIGNMENT) == 0 &&
           "in_re must be properly aligned for SIMD");
    assert(((uintptr_t)in_im % REQUIRED_ALIGNMENT) == 0 &&
           "in_im must be properly aligned for SIMD");
    assert(((uintptr_t)out_re % REQUIRED_ALIGNMENT) == 0 &&
           "out_re must be properly aligned for SIMD");
    assert(((uintptr_t)out_im % REQUIRED_ALIGNMENT) == 0 &&
           "out_im must be properly aligned for SIMD");

    // Broadcast Rader twiddles ONCE for entire stage (P0 optimization!)
#ifdef __AVX512F__
    __m512d rader_tw_re[6], rader_tw_im[6];
    broadcast_rader_twiddles_avx512_soa(rader_tw, rader_tw_re, rader_tw_im);
#elif defined(__AVX2__)
    __m256d rader_tw_re[6], rader_tw_im[6];
    broadcast_rader_twiddles_avx2_soa(rader_tw, rader_tw_re, rader_tw_im);
#else
    __m128d rader_tw_re[6], rader_tw_im[6];
    broadcast_rader_twiddles_sse2_soa(rader_tw, rader_tw_re, rader_tw_im);
#endif

    // Decide on non-temporal stores (LLC-aware heuristic)
    int nt_env_override = check_nt_env_override();

    const size_t bytes_per_stage = (size_t)K * 7 * 2 * sizeof(double); // 7 lanes, 2 components
    bool use_nt = false;

    if (nt_env_override == 0)
    {
        use_nt = false; // Force disable
    }
    else if (nt_env_override == 1)
    {
        use_nt = true; // Force enable
    }
    else
    {
        // Automatic decision based on working set
        use_nt = (bytes_per_stage > (size_t)(NT_THRESHOLD * LLC_BYTES)) &&
                 (K >= NT_MIN_K);
    }

    // Process entire range [0, K) with N1 variant
    radix7_process_range_soa_fv_n1(0, K, in_re, in_im, out_re, out_im,
                                   rader_tw_re, rader_tw_im,
                                   K, use_nt);

    // Memory fence if non-temporal stores were used
    if (use_nt)
    {
        _mm_sfence();
    }
}