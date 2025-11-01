/**
 * @file fft_twiddles_reorganization.c
 * @brief Functions that reorder twiddles from strided to blocked layouts
 *
 * @details
 * These functions are the "transposition" layer that transforms twiddles from
 * canonical storage (strided) into SIMD-optimized layouts (blocked).
 *
 * CRITICAL CONCEPT:
 * - Input: twiddle_get(handle, r, k) provides twiddles in any format
 * - Output: Blocked structures with sequential SIMD-friendly access
 * - Cost: Paid ONCE during planning, zero cost during execution
 *
 *
 * ┌─────────────────────────────────────────────────────────────┐
│ CANONICAL STORAGE (Hybrid System)                          │
├─────────────────────────────────────────────────────────────┤
│ fft_twiddles_hybrid.h                                       │
│ fft_twiddles_hybrid.c                                       │
│   - twiddle_create()                                        │
│   - twiddle_get()                                           │
│   - twiddle_destroy()                                       │
│   - SIMPLE/FACTORED strategies                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ REORGANIZATION SYSTEM (Layout Transformation)               │
├─────────────────────────────────────────────────────────────┤
│ fft_twiddles_reorganization.h                               │
│   - Type definitions (enums, structs)                       │
│   - Function declarations                                   │
│   - Accessor functions (inline)                             │
│   - Block structures (radix-16)                             │
│                                                             │
│ fft_twiddles_reorganization.c                               │
│   - twiddle_materialize_auto()                              │
│   - twiddle_materialize_with_layout()                       │
│   - choose_optimal_layout()                                 │
│   - materialize_radix2/4/5/8/16_blocked_*()                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ PLANNER API (Optional Convenience Wrappers)                 │
├─────────────────────────────────────────────────────────────┤
│ fft_twiddles_planner_api.h                                  │
│ fft_twiddles_planner_api.c                                  │
│   - get_stage_twiddles()                                    │
│   - twiddle_get_soa_view()                                  │
│   - Thin wrappers over reorganization system                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ BRIDGES (Optional, Radix-8/16 Only)                         │
├─────────────────────────────────────────────────────────────┤
│ fft_twiddle_bridges.h                                       │
│   - radix8_prepare_twiddles_avx512()                        │
│   - radix16_prepare_twiddles_avx512()                       │
│   - Package for butterfly consumption                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ BUTTERFLIES (Execution)                                     │
├─────────────────────────────────────────────────────────────┤
│ fft_radix2_avx512.h    - Direct pointer access             │
│ fft_radix4_avx512.h    - Direct pointer access             │
│ fft_radix5_avx512.h    - Direct pointer access             │
│ fft_radix8_avx512.h    - Via bridge (optional)             │
│ fft_radix16_avx512.h   - Via bridge + block accessors      │
└─────────────────────────────────────────────────────────────┘
 *
 * @author VectorFFT Team
 * @date 2025
 */

#include "fft_twiddles_layout_extensions.h"
#include "fft_twiddles_hybrid.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef _WIN32
#include <malloc.h>
#define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_free(ptr) free(ptr)
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288419716939937510
#endif

//==============================================================================
// THRESHOLDS (from butterfly headers)
//==============================================================================

#define RADIX8_BLOCKED4_THRESHOLD 256
#define RADIX16_BLOCKED8_THRESHOLD 512
#define RADIX16_RECURRENCE_THRESHOLD 4096

#define RADIX4_BLOCKED3_THRESHOLD 512
#define RADIX5_BLOCKED4_THRESHOLD 512 // NEW: Same strategy as radix-4

//==============================================================================
// LAYOUT SELECTION
//==============================================================================

twiddle_layout_type_t choose_optimal_layout(
    int radix,
    simd_arch_t arch,
    int butterflies_per_stage,
    int prefer_precomputed)
{
    // Small radices (2, 3, 4, 5, 7) - strided is fine
    if (radix <= 7)
    {
        return TWIDDLE_LAYOUT_STRIDED;
    }

    int simd_width = get_simd_width(arch);

    // Very small stages - overhead dominates
    if (butterflies_per_stage < simd_width * 4)
    {
        return TWIDDLE_LAYOUT_STRIDED;
    }

    // Radix-16/32 with precomputed preference
    if (prefer_precomputed && (radix == 16 || radix == 32))
    {
        if (butterflies_per_stage <= 16384)
        {
            return TWIDDLE_LAYOUT_PRECOMPUTED;
        }
    }

    // Default: blocked layout for large radices
    return TWIDDLE_LAYOUT_BLOCKED;
}

//==============================================================================
// RADIX-2 BLOCKED LAYOUT (ALL K VALUES) - AVX-512
//==============================================================================

/**
 * @brief Materialize radix-2 twiddles in blocked (SoA) format for AVX-512
 *
 * @details
 * Radix-2 has only ONE twiddle factor per butterfly: W_N^k
 *
 * Memory layout: [W[0..K-1]] (just one block!)
 * No BLOCKED2/BLOCKED4 variants needed - there's only 1 twiddle factor
 *
 * This is the simplest radix - just materialize W_N^k in SoA format
 */
static int materialize_radix2_blocked_avx512(twiddle_handle_t *handle)
{
    const int RADIX = 2;
    int K = handle->n / RADIX;

    size_t total_size = K * sizeof(double);

    double *re_data = (double *)aligned_alloc(64, total_size);
    double *im_data = (double *)aligned_alloc(64, total_size);

    if (!re_data || !im_data)
    {
        aligned_free(re_data);
        aligned_free(im_data);
        return -1;
    }

    // Read from canonical storage
    for (int k = 0; k < K; k++)
    {
        double tw_re, tw_im;
        twiddle_get(handle, 1, k, &tw_re, &tw_im); // W^1 only

        re_data[k] = tw_re;
        im_data[k] = tw_im;
    }

    handle->materialized_re = re_data;
    handle->materialized_im = im_data;
    handle->materialized_count = K;
    handle->owns_materialized = 1;

    handle->layout_desc.type = TWIDDLE_LAYOUT_BLOCKED;
    handle->layout_desc.simd_arch = SIMD_ARCH_AVX512;
    handle->layout_desc.simd_width = 8;
    handle->layout_desc.block_size = K * sizeof(double);
    handle->layout_desc.num_blocks = 1;
    handle->layout_desc.total_size = total_size * 2;
    handle->layout_desc.radix = 2;
    handle->layout_desc.num_twiddle_factors = 1;
    handle->layout_desc.butterflies_per_stage = K;
    handle->layout_desc.has_precomputed = 0;

    handle->layout_specific_data = NULL;

    return 0;
}

//==============================================================================
// RADIX-2 BLOCKED LAYOUT - AVX2
//==============================================================================

static int materialize_radix2_blocked_avx2(twiddle_handle_t *handle)
{
    const int RADIX = 2;
    int K = handle->n / RADIX;

    size_t total_size = K * sizeof(double);

    double *re_data = (double *)aligned_alloc(64, total_size);
    double *im_data = (double *)aligned_alloc(64, total_size);

    if (!re_data || !im_data)
    {
        aligned_free(re_data);
        aligned_free(im_data);
        return -1;
    }

    double sign = (handle->direction == FFT_FORWARD) ? -1.0 : +1.0;
    double base_angle = sign * 2.0 * M_PI / (double)handle->n;

    extern void sincos_octant(double angle, double *s, double *c);

    for (int k = 0; k < K; k++)
    {
        double angle = base_angle * (double)k;
        double s_val, c_val;
        sincos_octant(angle, &s_val, &c_val);

        re_data[k] = c_val;
        im_data[k] = s_val;
    }

    handle->materialized_re = re_data;
    handle->materialized_im = im_data;
    handle->materialized_count = K;
    handle->owns_materialized = 1;

    handle->layout_desc.type = TWIDDLE_LAYOUT_BLOCKED;
    handle->layout_desc.simd_arch = SIMD_ARCH_AVX2;
    handle->layout_desc.simd_width = 4;
    handle->layout_desc.block_size = K * sizeof(double);
    handle->layout_desc.num_blocks = 1;
    handle->layout_desc.total_size = total_size * 2;
    handle->layout_desc.radix = 2;
    handle->layout_desc.num_twiddle_factors = 1;
    handle->layout_desc.butterflies_per_stage = K;
    handle->layout_desc.has_precomputed = 0;

    handle->layout_specific_data = NULL;

    return 0;
}

//==============================================================================
// RADIX-3 BLOCKED LAYOUT - AVX-512
//==============================================================================

/**
 * @brief Materialize radix-3 twiddles in blocked format for AVX-512
 *
 * Memory layout: [W1[0..K-1], W2[0..K-1]]
 * No BLOCKED2 variant needed - only 2 twiddles total
 */
static int materialize_radix3_blocked_avx512(twiddle_handle_t *handle)
{
    const int RADIX = 3;
    int K = handle->n / RADIX;

    // Allocate: 2 blocks × K × sizeof(double) × 2 (re/im)
    size_t total_size = 2 * K * sizeof(double);

    double *re_data = (double *)aligned_alloc(64, total_size);
    double *im_data = (double *)aligned_alloc(64, total_size);

    if (!re_data || !im_data)
    {
        aligned_free(re_data);
        aligned_free(im_data);
        return -1;
    }

    // Read W^1, W^2 from canonical storage
    for (int s = 1; s <= 2; s++)
    {
        int block_offset = (s - 1) * K;

        for (int k = 0; k < K; k++)
        {
            double tw_re, tw_im;
            twiddle_get(handle, s, k, &tw_re, &tw_im);

            re_data[block_offset + k] = tw_re;
            im_data[block_offset + k] = tw_im;
        }
    }

    handle->materialized_re = re_data;
    handle->materialized_im = im_data;
    handle->materialized_count = 2 * K;
    handle->owns_materialized = 1;

    handle->layout_desc.type = TWIDDLE_LAYOUT_BLOCKED;
    handle->layout_desc.simd_arch = SIMD_ARCH_AVX512;
    handle->layout_desc.simd_width = 8;
    handle->layout_desc.block_size = K * sizeof(double);
    handle->layout_desc.num_blocks = 2;
    handle->layout_desc.total_size = total_size * 2;
    handle->layout_desc.radix = 3;
    handle->layout_desc.num_twiddle_factors = 2;
    handle->layout_desc.butterflies_per_stage = K;
    handle->layout_desc.has_precomputed = 0;

    handle->layout_specific_data = NULL;

    return 0;
}

//==============================================================================
// RADIX-3 BLOCKED LAYOUT - AVX2
//==============================================================================

static int materialize_radix3_blocked_avx2(twiddle_handle_t *handle)
{
    const int RADIX = 3;
    int K = handle->n / RADIX;

    size_t total_size = 2 * K * sizeof(double);

    double *re_data = (double *)aligned_alloc(64, total_size);
    double *im_data = (double *)aligned_alloc(64, total_size);

    if (!re_data || !im_data)
    {
        aligned_free(re_data);
        aligned_free(im_data);
        return -1;
    }

    // Read W^1, W^2 from canonical storage
    for (int s = 1; s <= 2; s++)
    {
        int block_offset = (s - 1) * K;

        for (int k = 0; k < K; k++)
        {
            double tw_re, tw_im;
            twiddle_get(handle, s, k, &tw_re, &tw_im);

            re_data[block_offset + k] = tw_re;
            im_data[block_offset + k] = tw_im;
        }
    }

    handle->materialized_re = re_data;
    handle->materialized_im = im_data;
    handle->materialized_count = 2 * K;
    handle->owns_materialized = 1;

    handle->layout_desc.type = TWIDDLE_LAYOUT_BLOCKED;
    handle->layout_desc.simd_arch = SIMD_ARCH_AVX2;
    handle->layout_desc.simd_width = 4;
    handle->layout_desc.block_size = K * sizeof(double);
    handle->layout_desc.num_blocks = 2;
    handle->layout_desc.total_size = total_size * 2;
    handle->layout_desc.radix = 3;
    handle->layout_desc.num_twiddle_factors = 2;
    handle->layout_desc.butterflies_per_stage = K;
    handle->layout_desc.has_precomputed = 0;

    handle->layout_specific_data = NULL;

    return 0;
}

//==============================================================================
// RADIX-4 BLOCKED3 LAYOUT (K ≤ 512) - AVX-512
//==============================================================================

/**
 * @brief Materialize radix-4 twiddles in BLOCKED3 format for AVX-512
 *
 * Memory layout: [W1[0..K-1], W2[0..K-1], W3[0..K-1]]
 * All 3 twiddle factors stored explicitly (no derivation)
 *
 * Use when: K ≤ 512 (memory bandwidth available, avoid computation)
 */
static int materialize_radix4_blocked3_avx512(twiddle_handle_t *handle)
{
    const int RADIX = 4;
    int K = handle->n / RADIX;

    size_t total_size = 3 * K * sizeof(double);

    double *re_data = (double *)aligned_alloc(64, total_size);
    double *im_data = (double *)aligned_alloc(64, total_size);

    if (!re_data || !im_data)
    {
        aligned_free(re_data);
        aligned_free(im_data);
        return -1;
    }

    // Read W^1, W^2, W^3 from canonical storage
    for (int s = 1; s <= 3; s++)
    {
        int block_offset = (s - 1) * K;

        for (int k = 0; k < K; k++)
        {
            double tw_re, tw_im;
            twiddle_get(handle, s, k, &tw_re, &tw_im);

            re_data[block_offset + k] = tw_re;
            im_data[block_offset + k] = tw_im;
        }
    }

    handle->materialized_re = re_data;
    handle->materialized_im = im_data;
    handle->materialized_count = 3 * K;
    handle->owns_materialized = 1;

    handle->layout_desc.type = TWIDDLE_LAYOUT_BLOCKED;
    handle->layout_desc.simd_arch = SIMD_ARCH_AVX512;
    handle->layout_desc.simd_width = 8;
    handle->layout_desc.block_size = K * sizeof(double);
    handle->layout_desc.num_blocks = 3;
    handle->layout_desc.total_size = total_size * 2;
    handle->layout_desc.radix = 4;
    handle->layout_desc.num_twiddle_factors = 3;
    handle->layout_desc.butterflies_per_stage = K;
    handle->layout_desc.has_precomputed = 0;

    handle->layout_specific_data = NULL;

    return 0;
}

//==============================================================================
// RADIX-4 BLOCKED2 LAYOUT (K > 512) - AVX-512
//==============================================================================

/**
 * @brief Materialize radix-4 twiddles in BLOCKED2 format for AVX-512
 *
 * Memory layout: [W1[0..K-1], W2[0..K-1]]
 * Runtime derives: W3 = W1 × W2 (1 complex multiply per butterfly)
 *
 * Bandwidth savings: 33% (store 2 blocks instead of 3)
 * Cost: 1 FMA operation per butterfly (hidden by butterfly computation)
 *
 * Use when: K > 512 (memory bandwidth constrained)
 */
static int materialize_radix4_blocked2_avx512(twiddle_handle_t *handle)
{
    const int RADIX = 4;
    int K = handle->n / RADIX;

    size_t total_size = 2 * K * sizeof(double);

    double *re_data = (double *)aligned_alloc(64, total_size);
    double *im_data = (double *)aligned_alloc(64, total_size);

    if (!re_data || !im_data)
    {
        aligned_free(re_data);
        aligned_free(im_data);
        return -1;
    }

    // Read W^1, W^2 only (W^3 will be derived at runtime)
    for (int s = 1; s <= 2; s++)
    {
        int block_offset = (s - 1) * K;

        for (int k = 0; k < K; k++)
        {
            double tw_re, tw_im;
            twiddle_get(handle, s, k, &tw_re, &tw_im);

            re_data[block_offset + k] = tw_re;
            im_data[block_offset + k] = tw_im;
        }
    }

    handle->materialized_re = re_data;
    handle->materialized_im = im_data;
    handle->materialized_count = 2 * K;
    handle->owns_materialized = 1;

    handle->layout_desc.type = TWIDDLE_LAYOUT_BLOCKED;
    handle->layout_desc.simd_arch = SIMD_ARCH_AVX512;
    handle->layout_desc.simd_width = 8;
    handle->layout_desc.block_size = K * sizeof(double);
    handle->layout_desc.num_blocks = 2;
    handle->layout_desc.total_size = total_size * 2;
    handle->layout_desc.radix = 4;
    handle->layout_desc.num_twiddle_factors = 3;
    handle->layout_desc.butterflies_per_stage = K;
    handle->layout_desc.has_precomputed = 0;

    handle->layout_specific_data = NULL;

    return 0;
}

//==============================================================================
// RADIX-4 BLOCKED3 LAYOUT (K ≤ 512) - AVX2
//==============================================================================

static int materialize_radix4_blocked3_avx2(twiddle_handle_t *handle)
{
    const int RADIX = 4;
    int K = handle->n / RADIX;

    assert(K <= RADIX4_BLOCKED3_THRESHOLD);

    size_t total_size = 3 * K * sizeof(double);

    double *re_data = (double *)aligned_alloc(64, total_size);
    double *im_data = (double *)aligned_alloc(64, total_size);

    if (!re_data || !im_data)
    {
        aligned_free(re_data);
        aligned_free(im_data);
        return -1;
    }

    double sign = (handle->direction == FFT_FORWARD) ? -1.0 : +1.0;
    double base_angle = sign * 2.0 * M_PI / (double)handle->n;

    extern void sincos_octant(double angle, double *s, double *c);

    for (int s = 1; s <= 3; s++)
    {
        int block_offset = (s - 1) * K;

        for (int k = 0; k < K; k++)
        {
            double angle = base_angle * (double)s * (double)k;
            double s_val, c_val;
            sincos_octant(angle, &s_val, &c_val);

            re_data[block_offset + k] = c_val;
            im_data[block_offset + k] = s_val;
        }
    }

    handle->materialized_re = re_data;
    handle->materialized_im = im_data;
    handle->materialized_count = 3 * K;
    handle->owns_materialized = 1;

    handle->layout_desc.type = TWIDDLE_LAYOUT_BLOCKED;
    handle->layout_desc.simd_arch = SIMD_ARCH_AVX2;
    handle->layout_desc.simd_width = 4;
    handle->layout_desc.block_size = K * sizeof(double);
    handle->layout_desc.num_blocks = 3;
    handle->layout_desc.total_size = total_size * 2;
    handle->layout_desc.radix = 4;
    handle->layout_desc.num_twiddle_factors = 3;
    handle->layout_desc.butterflies_per_stage = K;
    handle->layout_desc.has_precomputed = 0;

    handle->layout_specific_data = NULL;

    return 0;
}

//==============================================================================
// RADIX-4 BLOCKED2 LAYOUT (K > 512) - AVX2
//==============================================================================

static int materialize_radix4_blocked2_avx2(twiddle_handle_t *handle)
{
    const int RADIX = 4;
    int K = handle->n / RADIX;

    assert(K > RADIX4_BLOCKED3_THRESHOLD);

    size_t total_size = 2 * K * sizeof(double);

    double *re_data = (double *)aligned_alloc(64, total_size);
    double *im_data = (double *)aligned_alloc(64, total_size);

    if (!re_data || !im_data)
    {
        aligned_free(re_data);
        aligned_free(im_data);
        return -1;
    }

    double sign = (handle->direction == FFT_FORWARD) ? -1.0 : +1.0;
    double base_angle = sign * 2.0 * M_PI / (double)handle->n;

    extern void sincos_octant(double angle, double *s, double *c);

    for (int s = 1; s <= 2; s++)
    {
        int block_offset = (s - 1) * K;

        for (int k = 0; k < K; k++)
        {
            double angle = base_angle * (double)s * (double)k;
            double s_val, c_val;
            sincos_octant(angle, &s_val, &c_val);

            re_data[block_offset + k] = c_val;
            im_data[block_offset + k] = s_val;
        }
    }

    handle->materialized_re = re_data;
    handle->materialized_im = im_data;
    handle->materialized_count = 2 * K;
    handle->owns_materialized = 1;

    handle->layout_desc.type = TWIDDLE_LAYOUT_BLOCKED;
    handle->layout_desc.simd_arch = SIMD_ARCH_AVX2;
    handle->layout_desc.simd_width = 4;
    handle->layout_desc.block_size = K * sizeof(double);
    handle->layout_desc.num_blocks = 2;
    handle->layout_desc.total_size = total_size * 2;
    handle->layout_desc.radix = 4;
    handle->layout_desc.num_twiddle_factors = 3;
    handle->layout_desc.butterflies_per_stage = K;
    handle->layout_desc.has_precomputed = 0;

    handle->layout_specific_data = NULL;

    return 0;
}

//==============================================================================
// RADIX-5 BLOCKED4 LAYOUT (K ≤ 512) - AVX-512
//==============================================================================

static int materialize_radix5_blocked4_avx512(twiddle_handle_t *handle)
{
    const int RADIX = 5;
    int K = handle->n / RADIX;

    assert(K <= RADIX5_BLOCKED4_THRESHOLD);

    size_t total_size = 4 * K * sizeof(double);

    double *re_data = (double *)aligned_alloc(64, total_size);
    double *im_data = (double *)aligned_alloc(64, total_size);

    if (!re_data || !im_data)
    {
        aligned_free(re_data);
        aligned_free(im_data);
        return -1;
    }

    // Read W^1, W^2, W^3, W^4 from canonical storage
    for (int s = 1; s <= 4; s++)
    {
        int block_offset = (s - 1) * K;

        for (int k = 0; k < K; k++)
        {
            double tw_re, tw_im;
            twiddle_get(handle, s, k, &tw_re, &tw_im);

            re_data[block_offset + k] = tw_re;
            im_data[block_offset + k] = tw_im;
        }
    }

    handle->materialized_re = re_data;
    handle->materialized_im = im_data;
    handle->materialized_count = 4 * K;
    handle->owns_materialized = 1;

    handle->layout_desc.type = TWIDDLE_LAYOUT_BLOCKED;
    handle->layout_desc.simd_arch = SIMD_ARCH_AVX512;
    handle->layout_desc.simd_width = 8;
    handle->layout_desc.block_size = K * sizeof(double);
    handle->layout_desc.num_blocks = 4;
    handle->layout_desc.total_size = total_size * 2;
    handle->layout_desc.radix = 5;
    handle->layout_desc.num_twiddle_factors = 4;
    handle->layout_desc.butterflies_per_stage = K;
    handle->layout_desc.has_precomputed = 0;

    handle->layout_specific_data = NULL;

    return 0;
}

//==============================================================================
// RADIX-5 BLOCKED2 LAYOUT (K > 512) - AVX-512
//==============================================================================

static int materialize_radix5_blocked2_avx512(twiddle_handle_t *handle)
{
    const int RADIX = 5;
    int K = handle->n / RADIX;

    assert(K > RADIX5_BLOCKED4_THRESHOLD);

    size_t total_size = 2 * K * sizeof(double);

    double *re_data = (double *)aligned_alloc(64, total_size);
    double *im_data = (double *)aligned_alloc(64, total_size);

    if (!re_data || !im_data)
    {
        aligned_free(re_data);
        aligned_free(im_data);
        return -1;
    }

    // Read W^1, W^2 only (W^3, W^4 will be derived at runtime)
    for (int s = 1; s <= 2; s++)
    {
        int block_offset = (s - 1) * K;

        for (int k = 0; k < K; k++)
        {
            double tw_re, tw_im;
            twiddle_get(handle, s, k, &tw_re, &tw_im);

            re_data[block_offset + k] = tw_re;
            im_data[block_offset + k] = tw_im;
        }
    }

    handle->materialized_re = re_data;
    handle->materialized_im = im_data;
    handle->materialized_count = 2 * K;
    handle->owns_materialized = 1;

    handle->layout_desc.type = TWIDDLE_LAYOUT_BLOCKED;
    handle->layout_desc.simd_arch = SIMD_ARCH_AVX512;
    handle->layout_desc.simd_width = 8;
    handle->layout_desc.block_size = K * sizeof(double);
    handle->layout_desc.num_blocks = 2;
    handle->layout_desc.total_size = total_size * 2;
    handle->layout_desc.radix = 5;
    handle->layout_desc.num_twiddle_factors = 4; // Total (even though storing 2)
    handle->layout_desc.butterflies_per_stage = K;
    handle->layout_desc.has_precomputed = 0;

    handle->layout_specific_data = NULL;

    return 0;
}

//==============================================================================
// RADIX-5 BLOCKED4 LAYOUT (K ≤ 512) - AVX-2
//==============================================================================

static int materialize_radix5_blocked4_avx2(twiddle_handle_t *handle)
{
    const int RADIX = 5;
    int K = handle->n / RADIX;

    assert(K <= RADIX5_BLOCKED4_THRESHOLD);

    size_t total_size = 4 * K * sizeof(double);

    double *re_data = (double *)aligned_alloc(64, total_size);
    double *im_data = (double *)aligned_alloc(64, total_size);

    if (!re_data || !im_data)
    {
        aligned_free(re_data);
        aligned_free(im_data);
        return -1;
    }

    // Read W^1, W^2, W^3, W^4 from canonical storage
    for (int s = 1; s <= 4; s++)
    {
        int block_offset = (s - 1) * K;

        for (int k = 0; k < K; k++)
        {
            double tw_re, tw_im;
            twiddle_get(handle, s, k, &tw_re, &tw_im);

            re_data[block_offset + k] = tw_re;
            im_data[block_offset + k] = tw_im;
        }
    }

    handle->materialized_re = re_data;
    handle->materialized_im = im_data;
    handle->materialized_count = 4 * K;
    handle->owns_materialized = 1;

    handle->layout_desc.type = TWIDDLE_LAYOUT_BLOCKED;
    handle->layout_desc.simd_arch = SIMD_ARCH_AVX2;
    handle->layout_desc.simd_width = 4;
    handle->layout_desc.block_size = K * sizeof(double);
    handle->layout_desc.num_blocks = 4;
    handle->layout_desc.total_size = total_size * 2;
    handle->layout_desc.radix = 5;
    handle->layout_desc.num_twiddle_factors = 4;
    handle->layout_desc.butterflies_per_stage = K;
    handle->layout_desc.has_precomputed = 0;

    handle->layout_specific_data = NULL;

    return 0;
}

//==============================================================================
// RADIX-5 BLOCKED2 LAYOUT (K > 512) - AVX-2
//==============================================================================

static int materialize_radix5_blocked2_avx2(twiddle_handle_t *handle)
{
    const int RADIX = 5;
    int K = handle->n / RADIX;

    assert(K > RADIX5_BLOCKED4_THRESHOLD);

    size_t total_size = 2 * K * sizeof(double);

    double *re_data = (double *)aligned_alloc(64, total_size);
    double *im_data = (double *)aligned_alloc(64, total_size);

    if (!re_data || !im_data)
    {
        aligned_free(re_data);
        aligned_free(im_data);
        return -1;
    }

    // Read W^1, W^2 only (W^3, W^4 will be derived at runtime)
    for (int s = 1; s <= 2; s++)
    {
        int block_offset = (s - 1) * K;

        for (int k = 0; k < K; k++)
        {
            double tw_re, tw_im;
            twiddle_get(handle, s, k, &tw_re, &tw_im);

            re_data[block_offset + k] = tw_re;
            im_data[block_offset + k] = tw_im;
        }
    }

    handle->materialized_re = re_data;
    handle->materialized_im = im_data;
    handle->materialized_count = 2 * K;
    handle->owns_materialized = 1;

    handle->layout_desc.type = TWIDDLE_LAYOUT_BLOCKED;
    handle->layout_desc.simd_arch = SIMD_ARCH_AVX2;
    handle->layout_desc.simd_width = 4;
    handle->layout_desc.block_size = K * sizeof(double);
    handle->layout_desc.num_blocks = 2;
    handle->layout_desc.total_size = total_size * 2;
    handle->layout_desc.radix = 5;
    handle->layout_desc.num_twiddle_factors = 4; // Total (even though storing 2)
    handle->layout_desc.butterflies_per_stage = K;
    handle->layout_desc.has_precomputed = 0;

    handle->layout_specific_data = NULL;

    return 0;
}

//==============================================================================
// RADIX-8 BLOCKED4 LAYOUT (K ≤ 256) - AVX-512
//==============================================================================

/**
 * @brief Materialize radix-8 twiddles in BLOCKED4 format for AVX-512
 *
 * Memory layout: [W1[0..K-1], W2[0..K-1], W3[0..K-1], W4[0..K-1]]
 * Runtime derives: W5=-W1, W6=-W2, W7=-W3 (sign flips only)
 *
 * Bandwidth savings: 43% (store 4 blocks instead of 7)
 */
static int materialize_radix8_blocked4_avx512(twiddle_handle_t *handle)
{
    const int RADIX = 8;
    int K = handle->n / RADIX;

    size_t total_size = 4 * K * sizeof(double);

    double *re_data = (double *)aligned_alloc(64, total_size);
    double *im_data = (double *)aligned_alloc(64, total_size);

    if (!re_data || !im_data)
    {
        aligned_free(re_data);
        aligned_free(im_data);
        return -1;
    }

    // Read W^1..W^4 from canonical storage
    for (int s = 1; s <= 4; s++)
    {
        int block_offset = (s - 1) * K;

        for (int k = 0; k < K; k++)
        {
            double tw_re, tw_im;
            twiddle_get(handle, s, k, &tw_re, &tw_im);

            re_data[block_offset + k] = tw_re;
            im_data[block_offset + k] = tw_im;
        }
    }

    handle->materialized_re = re_data;
    handle->materialized_im = im_data;
    handle->materialized_count = 4 * K;
    handle->owns_materialized = 1;

    handle->layout_desc.type = TWIDDLE_LAYOUT_BLOCKED;
    handle->layout_desc.simd_arch = SIMD_ARCH_AVX512;
    handle->layout_desc.simd_width = 8;
    handle->layout_desc.block_size = K * sizeof(double);
    handle->layout_desc.num_blocks = 4;
    handle->layout_desc.total_size = total_size * 2;
    handle->layout_desc.radix = 8;
    handle->layout_desc.num_twiddle_factors = 7;
    handle->layout_desc.butterflies_per_stage = K;
    handle->layout_desc.has_precomputed = 0;

    handle->layout_specific_data = NULL;

    return 0;
}

//==============================================================================
// RADIX-8 BLOCKED2 LAYOUT (K > 256) - AVX-512
//==============================================================================

/**
 * @brief Materialize radix-8 twiddles in BLOCKED2 format for AVX-512
 *
 * Memory layout: [W1[0..K-1], W2[0..K-1]]
 * Runtime derives: W3=W1×W2, W4=W2², W5=-W1, W6=-W2, W7=-W3
 *
 * Bandwidth savings: 71% (store 2 blocks instead of 7)
 * Cost: 2 FMA operations per butterfly (hidden by computation)
 */

static int materialize_radix8_blocked2_avx512(twiddle_handle_t *handle)
{
    const int RADIX = 8;
    int K = handle->n / RADIX;

    size_t total_size = 2 * K * sizeof(double);

    double *re_data = (double *)aligned_alloc(64, total_size);
    double *im_data = (double *)aligned_alloc(64, total_size);

    if (!re_data || !im_data)
    {
        aligned_free(re_data);
        aligned_free(im_data);
        return -1;
    }

    // Read W^1, W^2 only (W^3..W^7 will be derived at runtime)
    for (int s = 1; s <= 2; s++)
    {
        int block_offset = (s - 1) * K;

        for (int k = 0; k < K; k++)
        {
            double tw_re, tw_im;
            twiddle_get(handle, s, k, &tw_re, &tw_im);

            re_data[block_offset + k] = tw_re;
            im_data[block_offset + k] = tw_im;
        }
    }

    handle->materialized_re = re_data;
    handle->materialized_im = im_data;
    handle->materialized_count = 2 * K;
    handle->owns_materialized = 1;

    handle->layout_desc.type = TWIDDLE_LAYOUT_BLOCKED;
    handle->layout_desc.simd_arch = SIMD_ARCH_AVX512;
    handle->layout_desc.simd_width = 8;
    handle->layout_desc.block_size = K * sizeof(double);
    handle->layout_desc.num_blocks = 2;
    handle->layout_desc.total_size = total_size * 2;
    handle->layout_desc.radix = 8;
    handle->layout_desc.num_twiddle_factors = 7;
    handle->layout_desc.butterflies_per_stage = K;
    handle->layout_desc.has_precomputed = 0;

    handle->layout_specific_data = NULL;

    return 0;
}

//==============================================================================
// RADIX-8 BLOCKED4 LAYOUT (K ≤ 256) - AVX2
//==============================================================================

/**
 * @brief Materialize radix-8 twiddles in BLOCKED4 format for AVX2
 */
/**
 * @brief Materialize radix-8 twiddles in BLOCKED4 format for AVX2
 *
 * Memory layout: [W1[0..K-1], W2[0..K-1], W3[0..K-1], W4[0..K-1]]
 * Runtime derives: W5=-W1, W6=-W2, W7=-W3 (sign flips only)
 *
 * Bandwidth savings: 43% (store 4 blocks instead of 7)
 */
static int materialize_radix8_blocked4_avx2(twiddle_handle_t *handle)
{
    const int RADIX = 8;
    int K = handle->n / RADIX;

    assert(K <= RADIX8_BLOCKED4_THRESHOLD);

    // Allocate: 4 blocks × K × sizeof(double) × 2 (re/im)
    size_t total_size = 4 * K * sizeof(double);

    double *re_data = (double *)aligned_alloc(64, total_size);
    double *im_data = (double *)aligned_alloc(64, total_size);

    if (!re_data || !im_data)
    {
        aligned_free(re_data);
        aligned_free(im_data);
        return -1;
    }

    // ──────────────────────────────────────────────────────────────────
    // REORGANIZATION: Read W^1..W^4 from canonical storage
    // ──────────────────────────────────────────────────────────────────

    for (int s = 1; s <= 4; s++)
    {
        int block_offset = (s - 1) * K;

        for (int k = 0; k < K; k++)
        {
            double tw_re, tw_im;
            twiddle_get(handle, s, k, &tw_re, &tw_im);

            re_data[block_offset + k] = tw_re;
            im_data[block_offset + k] = tw_im;
        }
    }

    // Store in handle
    handle->materialized_re = re_data;
    handle->materialized_im = im_data;
    handle->materialized_count = 4 * K;
    handle->owns_materialized = 1;

    handle->layout_desc.type = TWIDDLE_LAYOUT_BLOCKED;
    handle->layout_desc.simd_arch = SIMD_ARCH_AVX2;
    handle->layout_desc.simd_width = 4;
    handle->layout_desc.block_size = K * sizeof(double);
    handle->layout_desc.num_blocks = 4;
    handle->layout_desc.total_size = total_size * 2;
    handle->layout_desc.radix = 8;
    handle->layout_desc.num_twiddle_factors = 7;
    handle->layout_desc.butterflies_per_stage = K;
    handle->layout_desc.has_precomputed = 0;
    handle->layout_desc.precompute_offset = 0;

    handle->layout_specific_data = NULL;

    return 0;
}

//==============================================================================
// RADIX-8 BLOCKED2 LAYOUT (K > 256) - AVX2
//==============================================================================

/**
 * @brief Materialize radix-8 twiddles in BLOCKED2 format for AVX2
 */
/**
 * @brief Materialize radix-8 twiddles in BLOCKED2 format for AVX2
 *
 * Memory layout: [W1[0..K-1], W2[0..K-1]]
 * Runtime derives: W3=W1×W2, W4=W2², W5=-W1, W6=-W2, W7=-W3
 *
 * Bandwidth savings: 71% (store 2 blocks instead of 7)
 * Cost: 2 FMA operations per butterfly (hidden by computation)
 */
static int materialize_radix8_blocked2_avx2(twiddle_handle_t *handle)
{
    const int RADIX = 8;
    int K = handle->n / RADIX;

    assert(K > RADIX8_BLOCKED4_THRESHOLD);

    // Allocate: 2 blocks × K × sizeof(double) × 2 (re/im)
    size_t total_size = 2 * K * sizeof(double);

    double *re_data = (double *)aligned_alloc(64, total_size);
    double *im_data = (double *)aligned_alloc(64, total_size);

    if (!re_data || !im_data)
    {
        aligned_free(re_data);
        aligned_free(im_data);
        return -1;
    }

    // ──────────────────────────────────────────────────────────────────
    // REORGANIZATION: Read W^1, W^2 from canonical storage
    // ──────────────────────────────────────────────────────────────────

    for (int s = 1; s <= 2; s++)
    {
        int block_offset = (s - 1) * K;

        for (int k = 0; k < K; k++)
        {
            double tw_re, tw_im;
            twiddle_get(handle, s, k, &tw_re, &tw_im);

            re_data[block_offset + k] = tw_re;
            im_data[block_offset + k] = tw_im;
        }
    }

    // Store in handle
    handle->materialized_re = re_data;
    handle->materialized_im = im_data;
    handle->materialized_count = 2 * K;
    handle->owns_materialized = 1;

    handle->layout_desc.type = TWIDDLE_LAYOUT_BLOCKED;
    handle->layout_desc.simd_arch = SIMD_ARCH_AVX2;
    handle->layout_desc.simd_width = 4;
    handle->layout_desc.block_size = K * sizeof(double);
    handle->layout_desc.num_blocks = 2;
    handle->layout_desc.total_size = total_size * 2;
    handle->layout_desc.radix = 8;
    handle->layout_desc.num_twiddle_factors = 7;
    handle->layout_desc.butterflies_per_stage = K;
    handle->layout_desc.has_precomputed = 0;
    handle->layout_desc.precompute_offset = 0;

    handle->layout_specific_data = NULL;

    return 0;
}

//==============================================================================
// RADIX-16 BLOCKED LAYOUT (AVX2)
//==============================================================================

/**
 * @brief Reorganize radix-16 twiddles into AVX2 blocked layout
 *
 * @details
 * Radix-16 has 15 twiddle factors per butterfly (W^1 through W^15).
 * This is the HIGH-VALUE optimization: 15 strided accesses → 4 cache lines.
 *
 * STRIDED layout (input):
 *   tw_re: [W1[0..K-1], W2[0..K-1], ..., W15[0..K-1]]
 *   For butterfly k, need: tw_re[k], tw_re[k+K], ..., tw_re[k+14K]
 *   Result: 15 cache line fetches (scattered memory)
 *
 * BLOCKED layout (output):
 *   blocks[i]: Contains ALL 15 factors for 4 butterflies [4i, 4i+1, 4i+2, 4i+3]
 *   Size: 15 factors × 2 components × 4 doubles = 240 bytes = 3.75 cache lines
 *   Result: Sequential access, perfect prefetch
 *
 * Expected speedup: +15-20% for large FFTs
 */
static int materialize_radix16_blocked_avx2(twiddle_handle_t *handle)
{
    const int RADIX = 16;
    const int SIMD_WIDTH = 4;

    int K = handle->n / RADIX;
    int num_blocks = (K + SIMD_WIDTH - 1) / SIMD_WIDTH;

    // Allocate
    size_t alloc_size = num_blocks * sizeof(radix16_twiddle_block_avx2_t);
    radix16_twiddle_block_avx2_t *blocks =
        (radix16_twiddle_block_avx2_t *)aligned_alloc(64, alloc_size);

    if (!blocks)
    {
        return -1;
    }

    memset(blocks, 0, alloc_size);

    // REORGANIZATION: Transform from strided to blocked
    for (int block_idx = 0; block_idx < num_blocks; block_idx++)
    {
        int k_base = block_idx * SIMD_WIDTH;

        // For each of 15 twiddle factors
        for (int s = 1; s <= 15; s++)
        {
            // Gather 4 consecutive twiddles
            for (int lane = 0; lane < SIMD_WIDTH; lane++)
            {
                int k = k_base + lane;

                if (k < K)
                {
                    double tw_re, tw_im;
                    twiddle_get(handle, s, k, &tw_re, &tw_im);

                    blocks[block_idx].tw_data[s - 1][0][lane] = tw_re;
                    blocks[block_idx].tw_data[s - 1][1][lane] = tw_im;
                }
                else
                {
                    blocks[block_idx].tw_data[s - 1][0][lane] = 0.0;
                    blocks[block_idx].tw_data[s - 1][1][lane] = 0.0;
                }
            }
        }
    }

    // Store
    handle->layout_specific_data = blocks;
    handle->owns_materialized = 1;

    handle->layout_desc.type = TWIDDLE_LAYOUT_BLOCKED;
    handle->layout_desc.simd_arch = SIMD_ARCH_AVX2;
    handle->layout_desc.simd_width = SIMD_WIDTH;
    handle->layout_desc.block_size = sizeof(radix16_twiddle_block_avx2_t);
    handle->layout_desc.num_blocks = num_blocks;
    handle->layout_desc.total_size = alloc_size;
    handle->layout_desc.radix = RADIX;
    handle->layout_desc.num_twiddle_factors = 15;
    handle->layout_desc.butterflies_per_stage = K;
    handle->layout_desc.has_precomputed = 0;
    handle->layout_desc.precompute_offset = 0;

    handle->materialized_re = NULL;
    handle->materialized_im = NULL;
    handle->materialized_count = 0;

    return 0;
}

//==============================================================================
// RADIX-16 BLOCKED LAYOUT (AVX-512)
//==============================================================================

/**
 * @brief Reorganize radix-16 twiddles into AVX-512 blocked layout
 *
 * @details
 * Same principle as AVX2, but processes 8 butterflies per block.
 * Block size: 15 × 2 × 8 = 240 doubles = 1920 bytes = 30 cache lines
 * Still much better than strided: 15 × 8 = 120 separate accesses
 */
#ifdef __AVX512F__
static int materialize_radix16_blocked_avx512(twiddle_handle_t *handle)
{
    const int RADIX = 16;
    const int SIMD_WIDTH = 8;

    int K = handle->n / RADIX;
    int num_blocks = (K + SIMD_WIDTH - 1) / SIMD_WIDTH;

    size_t alloc_size = num_blocks * sizeof(radix16_twiddle_block_avx512_t);
    radix16_twiddle_block_avx512_t *blocks =
        (radix16_twiddle_block_avx512_t *)aligned_alloc(64, alloc_size);

    if (!blocks)
    {
        return -1;
    }

    memset(blocks, 0, alloc_size);

    // Reorganization loop
    for (int block_idx = 0; block_idx < num_blocks; block_idx++)
    {
        int k_base = block_idx * SIMD_WIDTH;

        for (int s = 1; s <= 15; s++)
        {
            for (int lane = 0; lane < SIMD_WIDTH; lane++)
            {
                int k = k_base + lane;

                if (k < K)
                {
                    double tw_re, tw_im;
                    twiddle_get(handle, s, k, &tw_re, &tw_im);

                    blocks[block_idx].tw_data[s - 1][0][lane] = tw_re;
                    blocks[block_idx].tw_data[s - 1][1][lane] = tw_im;
                }
                else
                {
                    blocks[block_idx].tw_data[s - 1][0][lane] = 0.0;
                    blocks[block_idx].tw_data[s - 1][1][lane] = 0.0;
                }
            }
        }
    }

    handle->layout_specific_data = blocks;
    handle->owns_materialized = 1;

    handle->layout_desc.type = TWIDDLE_LAYOUT_BLOCKED;
    handle->layout_desc.simd_arch = SIMD_ARCH_AVX512;
    handle->layout_desc.simd_width = SIMD_WIDTH;
    handle->layout_desc.block_size = sizeof(radix16_twiddle_block_avx512_t);
    handle->layout_desc.num_blocks = num_blocks;
    handle->layout_desc.total_size = alloc_size;
    handle->layout_desc.radix = RADIX;
    handle->layout_desc.num_twiddle_factors = 15;
    handle->layout_desc.butterflies_per_stage = K;
    handle->layout_desc.has_precomputed = 0;

    handle->materialized_re = NULL;
    handle->materialized_im = NULL;
    handle->materialized_count = 0;

    return 0;
}
#endif // __AVX512F__


static int materialize_radix16_blocked_avx2(twiddle_handle_t *handle)
{
    const int RADIX = 16;
    const int SIMD_WIDTH = 4;
    int K = handle->n / RADIX;
    int num_blocks = (K + SIMD_WIDTH - 1) / SIMD_WIDTH;

    // ═══════════════════════════════════════════════════════════════
    // 1. CREATE BLOCKED LAYOUT (for optimized butterfly access)
    // ═══════════════════════════════════════════════════════════════
    
    size_t block_alloc_size = num_blocks * sizeof(radix16_twiddle_block_avx2_t);
    radix16_twiddle_block_avx2_t *blocks =
        (radix16_twiddle_block_avx2_t *)aligned_alloc(64, block_alloc_size);

    if (!blocks) {
        return -1;
    }
    memset(blocks, 0, block_alloc_size);

    // Reorganization: Transform from strided to blocked
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        int k_base = block_idx * SIMD_WIDTH;
        for (int s = 1; s <= 15; s++) {
            for (int lane = 0; lane < SIMD_WIDTH; lane++) {
                int k = k_base + lane;
                if (k < K) {
                    double tw_re, tw_im;
                    twiddle_get(handle, s, k, &tw_re, &tw_im);
                    blocks[block_idx].tw_data[s - 1][0][lane] = tw_re;
                    blocks[block_idx].tw_data[s - 1][1][lane] = tw_im;
                } else {
                    blocks[block_idx].tw_data[s - 1][0][lane] = 0.0;
                    blocks[block_idx].tw_data[s - 1][1][lane] = 0.0;
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // 2. ALSO CREATE FLAT SoA VIEW (for generic API access)
    // ═══════════════════════════════════════════════════════════════
    
    int total_twiddles = 15 * K;  // 15 twiddle factors, K butterflies each
    size_t soa_size = total_twiddles * sizeof(double);
    
    double *re_data = (double *)aligned_alloc(64, soa_size);
    double *im_data = (double *)aligned_alloc(64, soa_size);
    
    if (!re_data || !im_data) {
        aligned_free(re_data);
        aligned_free(im_data);
        aligned_free(blocks);
        return -1;
    }
    
    // Populate flat SoA arrays: [W1[0..K-1], W2[0..K-1], ..., W15[0..K-1]]
    for (int s = 1; s <= 15; s++) {
        int block_offset = (s - 1) * K;
        for (int k = 0; k < K; k++) {
            double tw_re, tw_im;
            twiddle_get(handle, s, k, &tw_re, &tw_im);
            re_data[block_offset + k] = tw_re;
            im_data[block_offset + k] = tw_im;
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // 3. STORE BOTH REPRESENTATIONS
    // ═══════════════════════════════════════════════════════════════
    
    handle->layout_specific_data = blocks;  // Optimized blocked layout
    handle->materialized_re = re_data;      // Generic SoA view
    handle->materialized_im = im_data;      // Generic SoA view
    handle->materialized_count = total_twiddles;
    handle->owns_materialized = 1;

    handle->layout_desc.type = TWIDDLE_LAYOUT_BLOCKED;
    handle->layout_desc.simd_arch = SIMD_ARCH_AVX2;
    handle->layout_desc.simd_width = SIMD_WIDTH;
    handle->layout_desc.block_size = sizeof(radix16_twiddle_block_avx2_t);
    handle->layout_desc.num_blocks = num_blocks;
    handle->layout_desc.total_size = block_alloc_size + (soa_size * 2);  // Both layouts
    handle->layout_desc.radix = RADIX;
    handle->layout_desc.num_twiddle_factors = 15;
    handle->layout_desc.butterflies_per_stage = K;
    handle->layout_desc.has_precomputed = 0;
    handle->layout_desc.precompute_offset = 0;

    return 0;
}

//==============================================================================
// RADIX-16 PRECOMPUTED LAYOUT (AVX2)
//==============================================================================

/**
 * @brief Reorganize radix-16 twiddles with precomputed W_4 products
 *
 * @details
 * Radix-16 uses 2-stage radix-4 decomposition with W_4 intermediate twiddles.
 * This layout precomputes the W_4 × stage_twiddle products to eliminate
 * 12 complex multiplies per butterfly.
 *
 * Stage 1: Apply W_N^(s*k) for s=1..15
 * Stage 2: Apply W_4^j for j=1,2,3 to certain outputs
 *
 * We precompute:
 *   W_4^1 × W_N^(4*k), W_4^1 × W_N^(8*k), W_4^1 × W_N^(12*k)
 *   W_4^2 × W_N^(4*k), W_4^2 × W_N^(8*k), W_4^2 × W_N^(12*k)
 *   W_4^3 × W_N^(4*k), W_4^3 × W_N^(8*k), W_4^3 × W_N^(12*k)
 *
 * Result: 9 fewer complex multiplies per butterfly
 * Cost: ~400 bytes per block vs 240 bytes (blocked layout)
 * Use when: K < 16384 (memory bandwidth available)
 */
static int materialize_radix16_precomputed_avx2(twiddle_handle_t *handle)
{
    const int RADIX = 16;
    const int SIMD_WIDTH = 4;

    int K = handle->n / RADIX;
    int num_blocks = (K + SIMD_WIDTH - 1) / SIMD_WIDTH;

    size_t alloc_size = num_blocks * sizeof(radix16_precomputed_block_avx2_t);
    radix16_precomputed_block_avx2_t *blocks =
        (radix16_precomputed_block_avx2_t *)aligned_alloc(64, alloc_size);

    if (!blocks)
    {
        return -1;
    }

    memset(blocks, 0, alloc_size);

    // W_4 constants
    double sign = (handle->direction == FFT_FORWARD) ? -1.0 : 1.0;
    double w4_1_re = 0.0, w4_1_im = sign * 1.0;  // ±i
    double w4_2_re = -1.0, w4_2_im = 0.0;        // -1
    double w4_3_re = 0.0, w4_3_im = sign * -1.0; // ∓i

    // Reorganization with precomputation
    for (int block_idx = 0; block_idx < num_blocks; block_idx++)
    {
        int k_base = block_idx * SIMD_WIDTH;

        // Stage 1: Regular twiddles W_N^(s*k)
        for (int s = 1; s <= 15; s++)
        {
            for (int lane = 0; lane < SIMD_WIDTH; lane++)
            {
                int k = k_base + lane;

                if (k < K)
                {
                    double tw_re, tw_im;
                    twiddle_get(handle, s, k, &tw_re, &tw_im);

                    blocks[block_idx].stage1[s - 1][0][lane] = tw_re;
                    blocks[block_idx].stage1[s - 1][1][lane] = tw_im;
                }
                else
                {
                    blocks[block_idx].stage1[s - 1][0][lane] = 0.0;
                    blocks[block_idx].stage1[s - 1][1][lane] = 0.0;
                }
            }
        }

        // Precompute W_4 products for s=4,8,12 (indices 3,7,11)
        for (int lane = 0; lane < SIMD_WIDTH; lane++)
        {
            int k = k_base + lane;

            if (k < K)
            {
                // Get W_N^(4k), W_N^(8k), W_N^(12k)
                double w4k_re, w4k_im, w8k_re, w8k_im, w12k_re, w12k_im;
                twiddle_get(handle, 4, k, &w4k_re, &w4k_im);
                twiddle_get(handle, 8, k, &w8k_re, &w8k_im);
                twiddle_get(handle, 12, k, &w12k_re, &w12k_im);

                // Precompute W_4^1 × [W4k, W8k, W12k]
                blocks[block_idx].w4_products[0][0][0][lane] =
                    w4_1_re * w4k_re - w4_1_im * w4k_im;
                blocks[block_idx].w4_products[0][0][1][lane] =
                    w4_1_re * w4k_im + w4_1_im * w4k_re;

                blocks[block_idx].w4_products[0][1][0][lane] =
                    w4_1_re * w8k_re - w4_1_im * w8k_im;
                blocks[block_idx].w4_products[0][1][1][lane] =
                    w4_1_re * w8k_im + w4_1_im * w8k_re;

                blocks[block_idx].w4_products[0][2][0][lane] =
                    w4_1_re * w12k_re - w4_1_im * w12k_im;
                blocks[block_idx].w4_products[0][2][1][lane] =
                    w4_1_re * w12k_im + w4_1_im * w12k_re;

                // Precompute W_4^2 × [W4k, W8k, W12k]
                blocks[block_idx].w4_products[1][0][0][lane] =
                    w4_2_re * w4k_re - w4_2_im * w4k_im;
                blocks[block_idx].w4_products[1][0][1][lane] =
                    w4_2_re * w4k_im + w4_2_im * w4k_re;

                blocks[block_idx].w4_products[1][1][0][lane] =
                    w4_2_re * w8k_re - w4_2_im * w8k_im;
                blocks[block_idx].w4_products[1][1][1][lane] =
                    w4_2_re * w8k_im + w4_2_im * w8k_re;

                blocks[block_idx].w4_products[1][2][0][lane] =
                    w4_2_re * w12k_re - w4_2_im * w12k_im;
                blocks[block_idx].w4_products[1][2][1][lane] =
                    w4_2_re * w12k_im + w4_2_im * w12k_re;

                // Precompute W_4^3 × [W4k, W8k, W12k]
                blocks[block_idx].w4_products[2][0][0][lane] =
                    w4_3_re * w4k_re - w4_3_im * w4k_im;
                blocks[block_idx].w4_products[2][0][1][lane] =
                    w4_3_re * w4k_im + w4_3_im * w4k_re;

                blocks[block_idx].w4_products[2][1][0][lane] =
                    w4_3_re * w8k_re - w4_3_im * w8k_im;
                blocks[block_idx].w4_products[2][1][1][lane] =
                    w4_3_re * w8k_im + w4_3_im * w8k_re;

                blocks[block_idx].w4_products[2][2][0][lane] =
                    w4_3_re * w12k_re - w4_3_im * w12k_im;
                blocks[block_idx].w4_products[2][2][1][lane] =
                    w4_3_re * w12k_im + w4_3_im * w12k_re;
            }
        }
    }

    handle->layout_specific_data = blocks;
    handle->owns_materialized = 1;

    handle->layout_desc.type = TWIDDLE_LAYOUT_PRECOMPUTED;
    handle->layout_desc.simd_arch = SIMD_ARCH_AVX2;
    handle->layout_desc.simd_width = SIMD_WIDTH;
    handle->layout_desc.block_size = sizeof(radix16_precomputed_block_avx2_t);
    handle->layout_desc.num_blocks = num_blocks;
    handle->layout_desc.total_size = alloc_size;
    handle->layout_desc.radix = RADIX;
    handle->layout_desc.num_twiddle_factors = 15;
    handle->layout_desc.butterflies_per_stage = K;
    handle->layout_desc.has_precomputed = 1;
    handle->layout_desc.precompute_offset = offsetof(radix16_precomputed_block_avx2_t, w4_products);

    handle->materialized_re = NULL;
    handle->materialized_im = NULL;
    handle->materialized_count = 0;

    return 0;
}

//==============================================================================
// PUBLIC API: UNIFIED MATERIALIZATION WITH LAYOUT
//==============================================================================

int twiddle_materialize_with_layout(
    twiddle_handle_t *handle,
    twiddle_layout_type_t layout,
    simd_arch_t arch)
{
    if (!handle)
    {
        return -1;
    }

    // Already materialized with this layout?
    if (handle->layout_desc.type == layout &&
        handle->layout_desc.simd_arch == arch &&
        (handle->layout_specific_data != NULL || handle->materialized_re != NULL))
    {
        return 0;
    }

     // ═══════════════════════════════════════════════════════════════
    // CLEAN UP ALL PREVIOUS MATERIALIZATIONS
    // ═══════════════════════════════════════════════════════════════
    
    // Type 1: SoA arrays (radix-2/3/4/5/8 use this)
    if (handle->owns_materialized) {
        if (handle->materialized_re) {
            aligned_free(handle->materialized_re);
            handle->materialized_re = NULL;
        }
        if (handle->materialized_im) {
            aligned_free(handle->materialized_im);
            handle->materialized_im = NULL;
        }
        handle->materialized_count = 0;
    }
    
    // Type 2: Block structures (radix-16 uses this)
    if (handle->layout_specific_data) {
        aligned_free(handle->layout_specific_data);
        handle->layout_specific_data = NULL;
    }

    // ═══════════════════════════════════════════════════════════════
    
    // Reset ownership flag
    handle->owns_materialized = 0;
    int result = -1;

    if (layout == TWIDDLE_LAYOUT_BLOCKED)
    {
        //==================================================================
        // RADIX-2 DISPATCH (NEW)
        //==================================================================
        if (handle->radix == 2)
        {
            // No threshold decision needed - only 1 twiddle factor!
            if (arch == SIMD_ARCH_AVX512)
            {
                result = materialize_radix2_blocked_avx512(handle);
            }
            else if (arch == SIMD_ARCH_AVX2)
            {
                result = materialize_radix2_blocked_avx2(handle);
            }
            else
            {
                result = 0; // Scalar fallback
            }
        }

        //==================================================================
        // RADIX-3 DISPATCH (NEW)
        //==================================================================
        else if (handle->radix == 3)
        {
            // No threshold decision - only 2 twiddles (simple)
            if (arch == SIMD_ARCH_AVX512)
            {
                result = materialize_radix3_blocked_avx512(handle);
            }
            else if (arch == SIMD_ARCH_AVX2)
            {
                result = materialize_radix3_blocked_avx2(handle);
            }
            else
            {
                result = 0; // Scalar fallback
            }
        }

        //==================================================================
        // RADIX-4 DISPATCH (NEW)
        //==================================================================
        else if (handle->radix == 4)
        {
            int K = handle->n / handle->radix;

            if (arch == SIMD_ARCH_AVX512)
            {
                if (K <= RADIX4_BLOCKED3_THRESHOLD)
                {
                    result = materialize_radix4_blocked3_avx512(handle);
                }
                else
                {
                    result = materialize_radix4_blocked2_avx512(handle);
                }
            }
            else if (arch == SIMD_ARCH_AVX2)
            {
                if (K <= RADIX4_BLOCKED3_THRESHOLD)
                {
                    result = materialize_radix4_blocked3_avx2(handle);
                }
                else
                {
                    result = materialize_radix4_blocked2_avx2(handle);
                }
            }
            else
            {
                // Scalar fallback
                result = 0;
            }
        }

        //==================================================================
        // RADIX-5 DISPATCH (NEW)
        //==================================================================
        else if (handle->radix == 5)
        {
            int K = handle->n / handle->radix;

            if (arch == SIMD_ARCH_AVX512)
            {
                if (K <= RADIX5_BLOCKED4_THRESHOLD)
                {
                    result = materialize_radix5_blocked4_avx512(handle);
                }
                else
                {
                    result = materialize_radix5_blocked2_avx512(handle);
                }
            }
            else if (arch == SIMD_ARCH_AVX2)
            {
                if (K <= RADIX5_BLOCKED4_THRESHOLD)
                {
                    result = materialize_radix5_blocked4_avx2(handle);
                }
                else
                {
                    result = materialize_radix5_blocked2_avx2(handle);
                }
            }
            else
            {
                result = 0; // Scalar fallback
            }
        }

        //==================================================================
        // RADIX-8 DISPATCH (EXISTING)
        //==================================================================
        else if (handle->radix == 8)
        {
            int K = handle->n / handle->radix;

            if (arch == SIMD_ARCH_AVX512)
            {
                if (K <= RADIX8_BLOCKED4_THRESHOLD)
                {
                    result = materialize_radix8_blocked4_avx512(handle);
                }
                else
                {
                    result = materialize_radix8_blocked2_avx512(handle);
                }
            }
            else if (arch == SIMD_ARCH_AVX2)
            {
                if (K <= RADIX8_BLOCKED4_THRESHOLD)
                {
                    result = materialize_radix8_blocked4_avx2(handle);
                }
                else
                {
                    result = materialize_radix8_blocked2_avx2(handle);
                }
            }
            else
            {
                result = 0;
            }
        }
        //==================================================================
        // RADIX-16 DISPATCH (EXISTING)
        //==================================================================
        else if (handle->radix == 16 && arch == SIMD_ARCH_AVX2)
        {
            result = materialize_radix16_blocked_avx2(handle);
        }
#ifdef __AVX512F__
        else if (handle->radix == 16 && arch == SIMD_ARCH_AVX512)
        {
            result = materialize_radix16_blocked_avx512(handle);
        }
#endif
        else
        {
            // Radix/arch combination not implemented
            result = 0;
        }
    }
    else if (layout == TWIDDLE_LAYOUT_PRECOMPUTED)
    {
        if (handle->radix == 16 && arch == SIMD_ARCH_AVX2)
        {
            result = materialize_radix16_precomputed_avx2(handle);
        }
        else
        {
            result = 0;
        }
    }
    else
    {
        // STRIDED or INTERLEAVED
        result = 0;
    }

    return result;
}
/**
 * @brief Materialize with automatic layout selection
 */
int twiddle_materialize_auto(twiddle_handle_t *handle, simd_arch_t arch)
{
    if (!handle)
    {
        return -1;
    }

    int K = handle->n / handle->radix;

    // Choose optimal layout
    twiddle_layout_type_t layout = choose_optimal_layout(
        handle->radix, arch, K, 1 // prefer_precomputed = 1
    );

    // Materialize with chosen layout
    return twiddle_materialize_with_layout(handle, layout, arch);
}