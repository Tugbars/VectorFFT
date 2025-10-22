/**
 * @file fft_radix3_bv_native_soa.c
 * @brief TRUE END-TO-END SoA Radix-3 FFT Implementation - BACKWARD (INVERSE)
 *
 * @details
 * Native SoA radix-3 IFFT that eliminates split/join at stage boundaries.
 *
 * @author FFT Optimization Team
 * @version 1.0 (Native SoA)
 * @date 2025
 */

#include "fft_radix3_uniform.h"
#include "fft_radix3_macros.h"

#include <immintrin.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

//==============================================================================
// CONFIGURATION
//==============================================================================

#if defined(__AVX512F__)
#define REQUIRED_ALIGNMENT 64
#elif defined(__AVX2__) || defined(__AVX__)
#define REQUIRED_ALIGNMENT 32
#elif defined(__SSE2__)
#define REQUIRED_ALIGNMENT 16
#else
#define REQUIRED_ALIGNMENT 8
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
            cached_value = -1;
        else if (env[0] == '0')
            cached_value = 0;
        else if (env[0] == '1')
            cached_value = 1;
        else
            cached_value = -1;
    }

    return cached_value;
}

//==============================================================================
// HELPER: Process a Range of Butterflies - BACKWARD
//==============================================================================

static void radix3_process_range_native_soa_bv(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw,
    int K,
    int k_start,
    int k_end,
    int use_streaming)
{
    int k = k_start;
    const int prefetch_dist = RADIX3_PREFETCH_DISTANCE;

#ifdef __AVX512F__
    // ⚡ AVX-512: Process 8 butterflies per iteration (DOUBLE-PUMPED!)
    if (use_streaming)
    {
        for (; k + 7 < k_end; k += 8)
        {
            RADIX3_PIPELINE_4_NATIVE_SOA_BV_AVX512_STREAM(k, K, in_re, in_im, out_re, out_im, stage_tw, prefetch_dist, k_end);
            RADIX3_PIPELINE_4_NATIVE_SOA_BV_AVX512_STREAM(k + 4, K, in_re, in_im, out_re, out_im, stage_tw, prefetch_dist, k_end);
        }
    }
    else
    {
        for (; k + 7 < k_end; k += 8)
        {
            RADIX3_PIPELINE_4_NATIVE_SOA_BV_AVX512(k, K, in_re, in_im, out_re, out_im, stage_tw, prefetch_dist, k_end);
            RADIX3_PIPELINE_4_NATIVE_SOA_BV_AVX512(k + 4, K, in_re, in_im, out_re, out_im, stage_tw, prefetch_dist, k_end);
        }
    }

    // Cleanup: 4-butterfly group
    if (use_streaming)
    {
        for (; k + 3 < k_end; k += 4)
        {
            RADIX3_PIPELINE_4_NATIVE_SOA_BV_AVX512_STREAM(k, K, in_re, in_im, out_re, out_im, stage_tw, 0, k_end);
        }
    }
    else
    {
        for (; k + 3 < k_end; k += 4)
        {
            RADIX3_PIPELINE_4_NATIVE_SOA_BV_AVX512(k, K, in_re, in_im, out_re, out_im, stage_tw, 0, k_end);
        }
    }
#endif

#ifdef __AVX2__
    // ⚡ AVX2: Process 4 butterflies per iteration (P1 DOUBLE-PUMPED for ILP!)
    if (use_streaming)
    {
        for (; k + 3 < k_end; k += 4)
        {
            RADIX3_PIPELINE_2_NATIVE_SOA_BV_AVX2_STREAM(k, K, in_re, in_im, out_re, out_im, stage_tw, prefetch_dist, k_end);
            RADIX3_PIPELINE_2_NATIVE_SOA_BV_AVX2_STREAM(k + 2, K, in_re, in_im, out_re, out_im, stage_tw, prefetch_dist, k_end);
        }
    }
    else
    {
        for (; k + 3 < k_end; k += 4)
        {
            RADIX3_PIPELINE_2_NATIVE_SOA_BV_AVX2(k, K, in_re, in_im, out_re, out_im, stage_tw, prefetch_dist, k_end);
            RADIX3_PIPELINE_2_NATIVE_SOA_BV_AVX2(k + 2, K, in_re, in_im, out_re, out_im, stage_tw, prefetch_dist, k_end);
        }
    }

    // Cleanup: 2-butterfly group
    if (use_streaming)
    {
        for (; k + 1 < k_end; k += 2)
        {
            RADIX3_PIPELINE_2_NATIVE_SOA_BV_AVX2_STREAM(k, K, in_re, in_im, out_re, out_im, stage_tw, 0, k_end);
        }
    }
    else
    {
        for (; k + 1 < k_end; k += 2)
        {
            RADIX3_PIPELINE_2_NATIVE_SOA_BV_AVX2(k, K, in_re, in_im, out_re, out_im, stage_tw, 0, k_end);
        }
    }
#endif

#ifdef __SSE2__
    // ⚡ SSE2: Process 2 butterflies per iteration (P1 DOUBLE-PUMPED for ILP!)
    if (use_streaming)
    {
        for (; k + 1 < k_end; k += 2)
        {
            RADIX3_PIPELINE_1_NATIVE_SOA_BV_SSE2_STREAM(k, K, in_re, in_im, out_re, out_im, stage_tw, prefetch_dist, k_end);
            RADIX3_PIPELINE_1_NATIVE_SOA_BV_SSE2_STREAM(k + 1, K, in_re, in_im, out_re, out_im, stage_tw, prefetch_dist, k_end);
        }
    }
    else
    {
        for (; k + 1 < k_end; k += 2)
        {
            RADIX3_PIPELINE_1_NATIVE_SOA_BV_SSE2(k, K, in_re, in_im, out_re, out_im, stage_tw, prefetch_dist, k_end);
            RADIX3_PIPELINE_1_NATIVE_SOA_BV_SSE2(k + 1, K, in_re, in_im, out_re, out_im, stage_tw, prefetch_dist, k_end);
        }
    }

    // Cleanup: 1-butterfly tail
    if (use_streaming)
    {
        for (; k < k_end; k++)
        {
            RADIX3_PIPELINE_1_NATIVE_SOA_BV_SSE2_STREAM(k, K, in_re, in_im, out_re, out_im, stage_tw, 0, k_end);
        }
    }
    else
    {
        for (; k < k_end; k++)
        {
            RADIX3_PIPELINE_1_NATIVE_SOA_BV_SSE2(k, K, in_re, in_im, out_re, out_im, stage_tw, 0, k_end);
        }
    }
#endif

    // Scalar fallback
    for (; k < k_end; k++)
    {
        RADIX3_PIPELINE_1_NATIVE_SOA_BV_SCALAR(k, K, in_re, in_im, out_re, out_im, stage_tw);
    }

    if (use_streaming)
    {
        _mm_sfence();
    }
}

//==============================================================================
// MAIN FUNCTION: Radix-3 Backward Transform - NATIVE SoA
//==============================================================================

/**
 * @brief Execute one stage of radix-3 FFT - NATIVE SoA - BACKWARD (INVERSE)
 *
 * @details
 * ⚡⚡⚡ ZERO SHUFFLE VERSION!
 *
 * This is the inverse transform. The only difference from forward is the
 * rotation direction in the butterfly (sign flip in rotation calculation).
 *
 * @param[out] out_re Output real array (SoA)
 * @param[out] out_im Output imaginary array (SoA)
 * @param[in] in_re Input real array (SoA)
 * @param[in] in_im Input imaginary array (SoA)
 * @param[in] stage_tw Stage twiddle factors (SoA format)
 * @param[in] K Sub-transform length (N/3 for this stage)
 */
void fft_radix3_bv_native_soa(
    double *restrict out_re,
    double *restrict out_im,
    const double *restrict in_re,
    const double *restrict in_im,
    const fft_twiddles_soa *restrict stage_tw,
    int K)
{
    // Sanity checks
    if (!out_re || !out_im || !in_re || !in_im || !stage_tw || K <= 0)
        return;

    // In-place not supported
    if (in_re == out_re || in_im == out_im)
    {
        assert(0 && "In-place execution not supported!");
        return;
    }

    // Non-temporal store heuristic
    int nt_env_override = check_nt_env_override();
    const size_t write_footprint = 3ull * 2ull * K * sizeof(double);
    const int is_out_of_place = (in_re != out_re) && (in_im != out_im);

    int use_streaming = 0;

    if (nt_env_override == 0)
        use_streaming = 0;
    else if (nt_env_override == 1)
        use_streaming = is_out_of_place;
    else
        use_streaming = is_out_of_place &&
                        (K >= NT_MIN_K) &&
                        (write_footprint > (size_t)(NT_THRESHOLD * LLC_BYTES));

    // Runtime alignment check with fallback
    if (use_streaming)
    {
        uintptr_t r0 = (uintptr_t)&out_re[0];
        uintptr_t i0 = (uintptr_t)&out_im[0];

        if ((r0 % REQUIRED_ALIGNMENT) != 0 || (i0 % REQUIRED_ALIGNMENT) != 0)
            use_streaming = 0;
    }

    // Verify twiddle alignment
    assert(((uintptr_t)stage_tw->re % REQUIRED_ALIGNMENT) == 0);
    assert(((uintptr_t)stage_tw->im % REQUIRED_ALIGNMENT) == 0);

    // Process all butterflies
    radix3_process_range_native_soa_bv(
        out_re, out_im,
        in_re, in_im,
        stage_tw,
        K,
        0, // k_start
        K, // k_end
        use_streaming);
}