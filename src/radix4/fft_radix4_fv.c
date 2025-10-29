/**
 * @file fft_radix4_fv.c
 * @brief TRUE END-TO-END SoA Radix-4 FFT Implementation - Forward
 * 
 * @details
 * This module implements a native Structure-of-Arrays (SoA) radix-4 forward FFT
 * with automatic SIMD dispatch and optimized memory access patterns.
 * 
 * ARCHITECTURE:
 * - Native SoA throughout (no split/join)
 * - Automatic SIMD selection: AVX-512 → AVX2 → SSE2 → Scalar
 * - U=2 software pipelining for SIMD paths
 * - Non-temporal stores for large N
 * - Blocked twiddle layout: [W1[K], W2[K], W3[K]]
 * 
 * @author VectorFFT Team
 * @version 2.1
 * @date 2025
 */

#include "fft_radix4_uniform.h"

// Include all SIMD implementations
#ifdef __AVX512F__
    #include "fft_radix4_avx512_corrected_part1.h"
    #include "fft_radix4_avx512_corrected_part2.h"
#endif

#ifdef __AVX2__
    #include "fft_radix4_avx2_u2_pipelined.h"
#endif

#ifdef __SSE2__
    #include "fft_radix4_sse2_u2_pipelined.h"
#endif

#include "fft_radix4_scalar_optimized.h"

#include <immintrin.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

//==============================================================================
// CONFIGURATION
//==============================================================================

#define CACHE_LINE_BYTES 64

/**
 * @brief Required alignment based on SIMD instruction set
 */
#if defined(__AVX512F__)
    #define REQUIRED_ALIGNMENT 64
    #define VECTOR_WIDTH 8
#elif defined(__AVX2__) || defined(__AVX__)
    #define REQUIRED_ALIGNMENT 32
    #define VECTOR_WIDTH 4
#elif defined(__SSE2__)
    #define REQUIRED_ALIGNMENT 16
    #define VECTOR_WIDTH 2
#else
    #define REQUIRED_ALIGNMENT 8
    #define VECTOR_WIDTH 1
#endif

#ifndef LLC_BYTES
    #define LLC_BYTES (8 * 1024 * 1024)
#endif

#define NT_THRESHOLD 0.7
#define NT_MIN_K 4096

//==============================================================================
// HELPER: Environment Variable Parsing
//==============================================================================

static inline int check_nt_env_override(void)
{
    static int cached_value = -2;
    
    if (cached_value == -2)
    {
        const char *env = getenv("FFT_NT");
        if (env == NULL)
        {
            cached_value = -1;
        }
        else if (env[0] == '0')
        {
            cached_value = 0;
        }
        else if (env[0] == '1')
        {
            cached_value = 1;
        }
        else
        {
            cached_value = -1;
        }
    }
    
    return cached_value;
}

//==============================================================================
// HELPER: Process Range (Native SoA)
//==============================================================================

/**
 * @brief Process radix-4 butterflies in range [k_start, k_end) - NATIVE SoA
 * 
 * Data flow:
 *   - Load: in_re[k], in_im[k] (4 input blocks: A, B, C, D)
 *   - Compute: radix-4 butterfly with twiddles
 *   - Store: out_re[k], out_im[k] (4 output blocks: Y0, Y1, Y2, Y3)
 * 
 * @param[out] out_re Output real array (N elements)
 * @param[out] out_im Output imaginary array (N elements)
 * @param[in] in_re Input real array (N elements)
 * @param[in] in_im Input imaginary array (N elements)
 * @param[in] stage_tw SoA twiddle factors (blocked: [W1[K], W2[K], W3[K]])
 * @param[in] K Transform quarter-size (N/4 for this stage)
 * @param[in] N Full transform size
 * @param[in] k_start Starting index (inclusive)
 * @param[in] k_end Ending index (exclusive)
 * @param[in] use_streaming Use streaming stores for large N
 */
static void radix4_process_range_native_soa_fv(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw,
    int K,
    int N,
    int k_start,
    int k_end,
    int use_streaming)
{
    // Base pointers for input blocks [A, B, C, D]
    const double *restrict a_re = in_re;
    const double *restrict b_re = in_re + K;
    const double *restrict c_re = in_re + 2 * K;
    const double *restrict d_re = in_re + 3 * K;
    
    const double *restrict a_im = in_im;
    const double *restrict b_im = in_im + K;
    const double *restrict c_im = in_im + 2 * K;
    const double *restrict d_im = in_im + 3 * K;
    
    // Base pointers for output blocks [Y0, Y1, Y2, Y3]
    double *restrict y0_re = out_re;
    double *restrict y1_re = out_re + K;
    double *restrict y2_re = out_re + 2 * K;
    double *restrict y3_re = out_re + 3 * K;
    
    double *restrict y0_im = out_im;
    double *restrict y1_im = out_im + K;
    double *restrict y2_im = out_im + 2 * K;
    double *restrict y3_im = out_im + 3 * K;
    
    // Twiddle base pointers (blocked SoA)
    const double *restrict tw_re = stage_tw->re;
    const double *restrict tw_im = stage_tw->im;
    
    const double *restrict w1r = tw_re + 0 * K;
    const double *restrict w1i = tw_im + 0 * K;
    const double *restrict w2r = tw_re + 1 * K;
    const double *restrict w2i = tw_im + 1 * K;
    const double *restrict w3r = tw_re + 2 * K;
    const double *restrict w3i = tw_im + 2 * K;
    
    // Compute actual range to process within [k_start, k_end)
    int range_K = k_end - k_start;
    
    // Adjust pointers to start at k_start
    a_re += k_start;
    a_im += k_start;
    b_re += k_start;
    b_im += k_start;
    c_re += k_start;
    c_im += k_start;
    d_re += k_start;
    d_im += k_start;
    
    y0_re += k_start;
    y0_im += k_start;
    y1_re += k_start;
    y1_im += k_start;
    y2_re += k_start;
    y2_im += k_start;
    y3_re += k_start;
    y3_im += k_start;
    
    w1r += k_start;
    w1i += k_start;
    w2r += k_start;
    w2i += k_start;
    w3r += k_start;
    w3i += k_start;
    
    //==========================================================================
    // SIMD DISPATCH
    //==========================================================================
    
#ifdef __AVX512F__
    // AVX-512: Process using U=2 pipelined kernel
    {
        const bool is_write_only = true;
        const bool is_cold_out = (N >= 4096);
        
        // Temporary twiddle structure for stage wrapper
        fft_twiddles_soa tw_local;
        tw_local.re = tw_re + k_start;
        tw_local.im = tw_im + k_start;
        
        radix4_stage_baseptr_fv_avx512(N, range_K,
                                       a_re, a_im,
                                       out_re + k_start, out_im + k_start,
                                       &tw_local,
                                       is_write_only, is_cold_out);
        return;
    }
#endif
    
#ifdef __AVX2__
    // AVX2: Process using U=2 pipelined kernel
    {
        const bool is_write_only = true;
        const bool is_cold_out = (N >= 4096);
        
        fft_twiddles_soa tw_local;
        tw_local.re = tw_re + k_start;
        tw_local.im = tw_im + k_start;
        
        radix4_stage_baseptr_fv_avx2(N, range_K,
                                     a_re, a_im,
                                     out_re + k_start, out_im + k_start,
                                     &tw_local,
                                     is_write_only, is_cold_out);
        return;
    }
#endif
    
#ifdef __SSE2__
    // SSE2: Process using U=2 pipelined kernel
    {
        const bool is_write_only = true;
        const bool is_cold_out = (N >= 4096);
        
        fft_twiddles_soa tw_local;
        tw_local.re = tw_re + k_start;
        tw_local.im = tw_im + k_start;
        
        radix4_stage_baseptr_fv_sse2(N, range_K,
                                     a_re, a_im,
                                     out_re + k_start, out_im + k_start,
                                     &tw_local,
                                     is_write_only, is_cold_out);
        return;
    }
#endif
    
    // Scalar fallback
    {
        fft_twiddles_soa tw_local;
        tw_local.re = tw_re + k_start;
        tw_local.im = tw_im + k_start;
        
        radix4_stage_baseptr_fv_scalar(N, range_K,
                                       a_re, a_im,
                                       out_re + k_start, out_im + k_start,
                                       &tw_local);
    }
}

//==============================================================================
// MAIN FUNCTION: Radix-4 DIF Butterfly - NATIVE SoA - FORWARD
//==============================================================================

/**
 * @brief Radix-4 DIF butterfly - NATIVE SoA - Forward FFT
 * 
 * ALGORITHM:
 * @code
 *   For k = 0 to K-1:
 *     W1[k] = exp(-2πi·k/N)
 *     W2[k] = exp(-4πi·k/N) = W1[k]^2
 *     W3[k] = exp(-6πi·k/N) = W1[k]^3
 *     
 *     tB = B[k] * W1[k]
 *     tC = C[k] * W2[k]
 *     tD = D[k] * W3[k]
 *     
 *     sumAC = A[k] + tC
 *     difAC = A[k] - tC
 *     sumBD = tB + tD
 *     difBD = tB - tD
 *     
 *     rot = (+i) * difBD  (forward)
 *     
 *     Y0[k] = sumAC + sumBD
 *     Y1[k] = difAC - rot
 *     Y2[k] = sumAC - sumBD
 *     Y3[k] = difAC + rot
 * @endcode
 * 
 * MEMORY LAYOUT:
 *   - Input:  in_re[0..N-1], in_im[0..N-1] (4 blocks of K each)
 *   - Output: out_re[0..N-1], out_im[0..N-1] (4 blocks of K each)
 *   - Twiddles: stage_tw->re[0..3K-1], stage_tw->im[0..3K-1] (blocked SoA)
 * 
 * @param[out] out_re Output real array (N elements)
 * @param[out] out_im Output imaginary array (N elements)
 * @param[in] in_re Input real array (N elements)
 * @param[in] in_im Input imaginary array (N elements)
 * @param[in] stage_tw Stage twiddles (blocked SoA format: [W1, W2, W3])
 * @param[in] K Transform quarter-size (N/4)
 */
void fft_radix4_fv(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw,
    int K)
{
    const int N = 4 * K;
    
    //==========================================================================
    // ALIGNMENT HINTS
    //==========================================================================
    
#if defined(__GNUC__) || defined(__clang__)
    in_re  = (const double*)__builtin_assume_aligned(in_re,  REQUIRED_ALIGNMENT);
    in_im  = (const double*)__builtin_assume_aligned(in_im,  REQUIRED_ALIGNMENT);
    out_re = (double*)__builtin_assume_aligned(out_re, REQUIRED_ALIGNMENT);
    out_im = (double*)__builtin_assume_aligned(out_im, REQUIRED_ALIGNMENT);
    stage_tw->re = (const double*)__builtin_assume_aligned(stage_tw->re, REQUIRED_ALIGNMENT);
    stage_tw->im = (const double*)__builtin_assume_aligned(stage_tw->im, REQUIRED_ALIGNMENT);
#endif
    
    //==========================================================================
    // NON-TEMPORAL STORE HEURISTIC
    //==========================================================================
    
    int nt_env_override = check_nt_env_override();
    
    const size_t write_footprint = 4ull * K * sizeof(double);  // 4 output arrays
    const int is_out_of_place = (in_re != out_re) && (in_im != out_im);
    
    int use_streaming = 0;
    
    if (nt_env_override == 0)
    {
        use_streaming = 0;
    }
    else if (nt_env_override == 1)
    {
        use_streaming = is_out_of_place;
    }
    else
    {
        use_streaming = is_out_of_place &&
                       (K >= NT_MIN_K) &&
                       (write_footprint > (size_t)(NT_THRESHOLD * LLC_BYTES));
    }
    
    // Runtime alignment check with fallback
    if (use_streaming)
    {
        uintptr_t r0 = (uintptr_t)&out_re[0];
        uintptr_t i0 = (uintptr_t)&out_im[0];
        
        if ((r0 % REQUIRED_ALIGNMENT) != 0 || (i0 % REQUIRED_ALIGNMENT) != 0)
        {
            use_streaming = 0;
        }
    }
    
    // Verify twiddle alignment
    assert(((uintptr_t)stage_tw->re % REQUIRED_ALIGNMENT) == 0 &&
           "stage_tw->re must be properly aligned for SIMD");
    assert(((uintptr_t)stage_tw->im % REQUIRED_ALIGNMENT) == 0 &&
           "stage_tw->im must be properly aligned for SIMD");
    
    //==========================================================================
    // PROCESS FULL RANGE
    //==========================================================================
    // Unlike radix-2, radix-4 has no special geometric cases (no k=0, k=N/4)
    // All k values are processed uniformly
    
    radix4_process_range_native_soa_fv(out_re, out_im, in_re, in_im,
                                       stage_tw, K, N,
                                       0, K,  // Full range [0, K)
                                       use_streaming);
}
