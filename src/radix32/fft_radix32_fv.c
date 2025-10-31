/**
 * @file fft_radix32_fv.c
 * @brief Native SoA Radix-32 FFT Implementation - Forward (fv)
 * 
 * @details
 * This module implements native SoA radix-32 forward FFT with:
 * - Standard twiddle version (fft_radix32_fv)
 * - Twiddle-less n1 version (fft_radix32_fv_n1) for first stage [FUTURE]
 * 
 * Architecture: 2×16 Cooley-Tukey decomposition with merge
 * 
 * @author VectorFFT Team
 * @version 1.0 (2×16 Cooley-Tukey)
 * @date 2025
 */

#include "fft_radix32_uniform.h"
#include <immintrin.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

// Architecture-specific implementations
#ifdef __AVX512F__
    #include "fft_radix32_avx512_native_soa.h"
#endif

#ifdef __AVX2__
    #include "fft_radix32_avx2_native_soa.h"
#endif

#include "fft_radix32_scalar_native_soa.h"

// N1 implementations (FUTURE)
#ifdef __AVX512F__
    // #include "fft_radix32_avx512_n1.h"  // TODO
#endif

#ifdef __AVX2__
    // #include "fft_radix32_avx2_n1.h"    // TODO
#endif

// #include "fft_radix32_scalar_n1.h"      // TODO

//==============================================================================
// CONFIGURATION
//==============================================================================

#define CACHE_LINE_BYTES 64

#if defined(__AVX512F__)
    #define REQUIRED_ALIGNMENT 64
    #define VECTOR_WIDTH 8
#elif defined(__AVX2__) || defined(__AVX__)
    #define REQUIRED_ALIGNMENT 32
    #define VECTOR_WIDTH 4
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
// HELPER: Process Range (Native SoA) - WITH TWIDDLES - FORWARD
//==============================================================================

static void radix32_process_range_native_soa_fv(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const void *restrict stage_tw_opaque,
    int K,
    int N,
    int k_start,
    int k_end,
    int use_streaming)
{
    (void)use_streaming; // Handled internally by architecture-specific code
    (void)N;             // Used for heuristics
    
    int range_K = k_end - k_start;
    
    // Adjust pointers to range
    const double *in_re_range = in_re + k_start;
    const double *in_im_range = in_im + k_start;
    double *out_re_range = out_re + k_start;
    double *out_im_range = out_im + k_start;
    
#ifdef __AVX512F__
    radix32_stage_dit_forward_soa_avx512(range_K,
                                         in_re_range, in_im_range,
                                         out_re_range, out_im_range,
                                         stage_tw_opaque);
    return;
#endif
    
#ifdef __AVX2__
    radix32_stage_dit_forward_soa_avx2(range_K,
                                       in_re_range, in_im_range,
                                       out_re_range, out_im_range,
                                       stage_tw_opaque);
    return;
#endif
    
    // Scalar fallback
    radix32_stage_dit_forward_soa_scalar(range_K,
                                         in_re_range, in_im_range,
                                         out_re_range, out_im_range,
                                         stage_tw_opaque);
}

//==============================================================================
// HELPER: Process Range (Native SoA) - NO TWIDDLES (N1) - FORWARD [STUB]
//==============================================================================

static void radix32_process_range_native_soa_fv_n1(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    int K,
    int N,
    int k_start,
    int k_end)
{
    (void)out_re;
    (void)out_im;
    (void)in_re;
    (void)in_im;
    (void)K;
    (void)N;
    (void)k_start;
    (void)k_end;
    
    // TODO: Implement N1 variant (no twiddles)
    // This is for first-stage optimization where all twiddles = 1+0i
    assert(0 && "fft_radix32_fv_n1 not yet implemented");
}

//==============================================================================
// MAIN FUNCTION: Radix-32 DIT Forward - NATIVE SoA - WITH TWIDDLES
//==============================================================================

/**
 * @brief Radix-32 DIT forward butterfly - NATIVE SoA - WITH TWIDDLES
 * 
 * @details
 * Implements 2×16 Cooley-Tukey decomposition:
 * 1. Two radix-16 sub-FFTs (even/odd indices)
 * 2. Apply merge twiddles W₃₂^m to odd half
 * 3. Radix-2 combine
 * 
 * Supports BLOCKED8/BLOCKED4 twiddle modes with optional recurrence.
 * 
 * @param[out] out_re Output real array (N elements, N=32K)
 * @param[out] out_im Output imaginary array (N elements)
 * @param[in] in_re Input real array (N elements)
 * @param[in] in_im Input imaginary array (N elements)
 * @param[in] stage_tw_opaque Opaque pointer to radix32_stage_twiddles_*
 * @param[in] K Transform size / 32
 */
void fft_radix32_fv(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const void *restrict stage_tw_opaque,
    int K)
{
    const int N = 32 * K;
    
    // Apply alignment hints
#if defined(__GNUC__) || defined(__clang__)
    in_re  = (const double*)__builtin_assume_aligned(in_re,  REQUIRED_ALIGNMENT);
    in_im  = (const double*)__builtin_assume_aligned(in_im,  REQUIRED_ALIGNMENT);
    out_re = (double*)__builtin_assume_aligned(out_re, REQUIRED_ALIGNMENT);
    out_im = (double*)__builtin_assume_aligned(out_im, REQUIRED_ALIGNMENT);
#endif
    
    // Check environment override for NT stores
    int nt_env_override = check_nt_env_override();
    
    // Calculate working set size
    const size_t write_footprint = 32ull * K * sizeof(double);
    const int is_out_of_place = (in_re != out_re) && (in_im != out_im);
    
    // Determine if we should use NT stores
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
    
    // Verify alignment for NT stores
    if (use_streaming)
    {
        uintptr_t r0 = (uintptr_t)&out_re[0];
        uintptr_t i0 = (uintptr_t)&out_im[0];
        
        if ((r0 % REQUIRED_ALIGNMENT) != 0 || (i0 % REQUIRED_ALIGNMENT) != 0)
        {
            use_streaming = 0;
        }
    }
    
    // Verify twiddle alignment (critical for SIMD)
    // Note: Actual validation depends on architecture-specific structure
    // This is a placeholder - real validation happens in implementation
    (void)stage_tw_opaque; // Suppress unused warning if no validation
    
    // Process full range
    radix32_process_range_native_soa_fv(out_re, out_im, in_re, in_im,
                                        stage_tw_opaque, K, N, 0, K,
                                        use_streaming);
}

//==============================================================================
// MAIN FUNCTION: Radix-32 DIT Forward - NATIVE SoA - NO TWIDDLES (N1) [STUB]
//==============================================================================

/**
 * @brief Radix-32 DIT forward butterfly - NATIVE SoA - NO TWIDDLES (n1)
 * 
 * @details
 * Twiddle-less variant for first radix-32 stage or when all twiddles = 1+0i.
 * Expected to be 40-60% faster than standard version.
 * 
 * NOT YET IMPLEMENTED - this is a placeholder for future optimization.
 * 
 * @param[out] out_re Output real array (N elements, N=32K)
 * @param[out] out_im Output imaginary array (N elements)
 * @param[in] in_re Input real array (N elements)
 * @param[in] in_im Input imaginary array (N elements)
 * @param[in] K Transform size / 32
 */
void fft_radix32_fv_n1(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    int K)
{
    const int N = 32 * K;
    
    // Apply alignment hints
#if defined(__GNUC__) || defined(__clang__)
    in_re  = (const double*)__builtin_assume_aligned(in_re,  REQUIRED_ALIGNMENT);
    in_im  = (const double*)__builtin_assume_aligned(in_im,  REQUIRED_ALIGNMENT);
    out_re = (double*)__builtin_assume_aligned(out_re, REQUIRED_ALIGNMENT);
    out_im = (double*)__builtin_assume_aligned(out_im, REQUIRED_ALIGNMENT);
#endif
    
    // Process full range (no twiddles)
    radix32_process_range_native_soa_fv_n1(out_re, out_im, in_re, in_im,
                                           K, N, 0, K);
}
