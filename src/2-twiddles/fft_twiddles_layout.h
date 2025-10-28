/**
 * @file fft_twiddles_layout_extensions.h
 * @brief SIMD-friendly twiddle layout extensions for planner API
 *
 * @details
 * This header extends the existing twiddle planner to support multiple
 * memory layouts optimized for different SIMD architectures and radix sizes.
 *
 * ARCHITECTURAL PRINCIPLE:
 * ========================
 * - Layout decisions happen at PLANNING time (pay once)
 * - Execution kernels get optimal memory access patterns (pay zero)
 * - Different radices can use different layouts
 * - SIMD width (AVX2 vs AVX-512) affects optimal layout
 *
 * @version 1.0
 * @date 2025
 */

#ifndef FFT_TWIDDLES_LAYOUT_EXTENSIONS_H
#define FFT_TWIDDLES_LAYOUT_EXTENSIONS_H

#include <stddef.h>
#include <stdint.h>

//==============================================================================
// LAYOUT TYPES
//==============================================================================

/**
 * @brief Twiddle memory layout strategies
 */
typedef enum
{
    /**
     * @brief STRIDED layout (default, current implementation)
     *
     * Memory: [W1[0..K-1], W2[0..K-1], ..., W(R-1)[0..K-1]]
     *
     * Access pattern for butterfly k:
     *   W_s(k) = tw_re[(s-1)*K + k]
     *
     * Pros: Simple, works for all radices
     * Cons: Poor cache locality (stride = K), bad prefetch
     *
     * Use: Legacy compatibility only
     */
    TWIDDLE_LAYOUT_STRIDED,

    /**
     * @brief BLOCKED layout (SIMD-width blocks)
     *
     * Memory: [W1[0..W-1], W2[0..W-1], ..., W(R-1)[0..W-1],
     *          W1[W..2W-1], W2[W..2W-1], ...]
     * where W = SIMD width (4 for AVX2 double, 8 for AVX-512 double)
     *
     * Access pattern for butterfly block k/W:
     *   W_s[k..k+W-1] = tw_re[block_idx * block_size + (s-1) * W]
     *
     * Pros: All twiddles for SIMD_WIDTH butterflies in sequential memory
     * Cons: Requires block-aware indexing
     *
     * Use: Radix-8, radix-16, radix-32 (high twiddle count)
     */
    TWIDDLE_LAYOUT_BLOCKED,

    /**
     * @brief INTERLEAVED layout (re/im pairs, then by factor)
     *
     * Memory: [W1_re[0..W-1], W1_im[0..W-1], W2_re[0..W-1], W2_im[0..W-1], ...]
     *
     * Pros: Better for some non-FMA architectures
     * Cons: Complicates SoA access
     *
     * Use: Special cases only (not recommended for modern CPUs)
     */
    TWIDDLE_LAYOUT_INTERLEAVED,

    /**
     * @brief PRECOMPUTED layout (with intermediate factors)
     *
     * Memory: [stage_twiddles, precomputed_products, ...]
     * Example for radix-16:
     *   [W_N^(s*k) for s=1..15, W_4 * W_N^(s*k) for s=4..15]
     *
     * Pros: Eliminates intermediate twiddle multiply (e.g., W_4 in radix-16)
     * Cons: Larger memory footprint
     *
     * Use: Radix-16 and radix-32 where intermediate multiplies matter
     */
    TWIDDLE_LAYOUT_PRECOMPUTED

} twiddle_layout_type_t;

/**
 * @brief SIMD architecture target
 */
typedef enum
{
    SIMD_ARCH_SCALAR, // No SIMD (width = 1)
    SIMD_ARCH_SSE2,   // width = 2 (double)
    SIMD_ARCH_AVX2,   // width = 4 (double)
    SIMD_ARCH_AVX512  // width = 8 (double)
} simd_arch_t;

/**
 * @brief Layout descriptor for materialized twiddles
 */
typedef struct
{
    twiddle_layout_type_t type; // Layout strategy
    simd_arch_t simd_arch;      // Target SIMD architecture
    int simd_width;             // Elements per SIMD register (computed)
    int block_size;             // Size of one block (bytes)
    int num_blocks;             // Total number of blocks
    size_t total_size;          // Total allocation size (bytes)

    // Radix-specific metadata
    int radix;                 // FFT radix
    int num_twiddle_factors;   // radix - 1
    int butterflies_per_stage; // K = N_stage / radix

    // Precomputed layout metadata
    int has_precomputed;   // 1 if includes precomputed products
    int precompute_offset; // Offset to precomputed section

} twiddle_layout_desc_t;

//==============================================================================
// EXTENDED TWIDDLE HANDLE (ADD TO fft_twiddles_hybrid.h)
//==============================================================================

/**
 * @brief Extensions to add to twiddle_handle_t structure
 *
 * Add these fields to your existing twiddle_handle_t:
 *
 * typedef struct twiddle_handle {
 *     // ... existing fields (strategy, direction, n, radix, etc.) ...
 *
 *     // Materialized SoA arrays
 *     double *materialized_re;
 *     double *materialized_im;
 *     int materialized_count;
 *     int owns_materialized;
 *
 *     // *** ADD THESE NEW FIELDS: ***
 *     twiddle_layout_desc_t layout_desc;    // Layout metadata
 *     void *layout_specific_data;           // Opaque pointer for layout-specific structs
 *
 * } twiddle_handle_t;
 */

//==============================================================================
// RADIX-16 BLOCKED LAYOUT STRUCTURES
//==============================================================================

/**
 * @brief Blocked twiddle storage for radix-16, AVX2
 *
 * Each block stores all 15 twiddle factors for SIMD_WIDTH=4 butterflies.
 * Total: 15 factors × 2 components × 4 doubles = 240 bytes = 3.75 cache lines
 */
typedef struct
{
    // Storage: [W1_re, W1_im, W2_re, W2_im, ..., W15_re, W15_im]
    // where each is a __m256d (4 doubles)
    double tw_data[15][2][4]; // [factor_idx][re=0,im=1][lane]
} __attribute__((aligned(64))) radix16_twiddle_block_avx2_t;

/**
 * @brief Blocked twiddle storage for radix-16, AVX-512
 *
 * Each block stores all 15 twiddle factors for SIMD_WIDTH=8 butterflies.
 * Total: 15 factors × 2 components × 8 doubles = 480 bytes = 7.5 cache lines
 */
typedef struct
{
    double tw_data[15][2][8]; // [factor_idx][re=0,im=1][lane]
} __attribute__((aligned(64))) radix16_twiddle_block_avx512_t;

/**
 * @brief Precomputed layout for radix-16, AVX2
 *
 * Includes both stage twiddles AND W_4 intermediate products
 */
typedef struct
{
    // Stage twiddles: W_N^(s*k) for s=1..15
    double stage1[15][2][4]; // [factor_idx][re=0,im=1][lane]

    // Precomputed products: W_4^j × W_N^(s*k) for j=1,2,3 and s=4,8,12
    // (reduces 12 complex multiplies to loads)
    double w4_products[3][3][2][4]; // [w4_idx][factor_group][re=0,im=1][lane]

} __attribute__((aligned(64))) radix16_precomputed_block_avx2_t;

//==============================================================================
// LAYOUT CONFIGURATION API
//==============================================================================

/**
 * @brief Choose optimal layout for given radix and architecture
 *
 * @param radix FFT radix (8, 16, 32, etc.)
 * @param arch SIMD architecture
 * @param butterflies_per_stage K = N_stage / radix
 * @param prefer_precomputed Allow precomputed layouts (higher memory)
 *
 * @return Optimal layout type
 */
twiddle_layout_type_t choose_optimal_layout(
    int radix,
    simd_arch_t arch,
    int butterflies_per_stage,
    int prefer_precomputed);

/**
 * @brief Get SIMD width for architecture
 */
static inline int get_simd_width(simd_arch_t arch)
{
    switch (arch)
    {
    case SIMD_ARCH_SCALAR:
        return 1;
    case SIMD_ARCH_SSE2:
        return 2;
    case SIMD_ARCH_AVX2:
        return 4;
    case SIMD_ARCH_AVX512:
        return 8;
    default:
        return 1;
    }
}

//==============================================================================
// MATERIALIZATION API (EXTENDED)
//==============================================================================

/**
 * @brief Materialize twiddles with specific layout
 *
 * @param handle Twiddle handle (must be in factored or simple form)
 * @param layout Desired layout type
 * @param arch Target SIMD architecture
 *
 * @return 0 on success, -1 on error
 *
 * This replaces the generic twiddle_materialize() with layout control.
 */
int twiddle_materialize_with_layout(
    twiddle_handle_t *handle,
    twiddle_layout_type_t layout,
    simd_arch_t arch);

/**
 * @brief Materialize twiddles with automatic layout selection
 *
 * This is the smart version that chooses the best layout for the radix/arch.
 */
int twiddle_materialize_auto(
    twiddle_handle_t *handle,
    simd_arch_t arch);

//==============================================================================
// ACCESSOR FUNCTIONS FOR BLOCKED LAYOUTS
//==============================================================================

/**
 * @brief Get pointer to twiddle block for radix-16 AVX2
 *
 * @param handle Materialized twiddle handle (must use BLOCKED layout)
 * @param block_idx Block index (k / simd_width)
 *
 * @return Pointer to block, or NULL on error
 */
static inline const radix16_twiddle_block_avx2_t *
get_radix16_block_avx2(const twiddle_handle_t *handle, int block_idx)
{
    if (!handle || !handle->materialized_re)
        return NULL;
    if (handle->layout_desc.type != TWIDDLE_LAYOUT_BLOCKED)
        return NULL;
    if (handle->layout_desc.radix != 16)
        return NULL;
    if (handle->layout_desc.simd_arch != SIMD_ARCH_AVX2)
        return NULL;

    return ((const radix16_twiddle_block_avx2_t *)handle->layout_specific_data) + block_idx;
}

/**
 * @brief Get pointer to twiddle block for radix-16 AVX-512
 */
static inline const radix16_twiddle_block_avx512_t *
get_radix16_block_avx512(const twiddle_handle_t *handle, int block_idx)
{
    if (!handle || !handle->materialized_re)
        return NULL;
    if (handle->layout_desc.type != TWIDDLE_LAYOUT_BLOCKED)
        return NULL;
    if (handle->layout_desc.radix != 16)
        return NULL;
    if (handle->layout_desc.simd_arch != SIMD_ARCH_AVX512)
        return NULL;

    return ((const radix16_twiddle_block_avx512_t *)handle->layout_specific_data) + block_idx;
}

/**
 * @brief Get pointer to precomputed block for radix-16 AVX2
 */
static inline const radix16_precomputed_block_avx2_t *
get_radix16_precomputed_avx2(const twiddle_handle_t *handle, int block_idx)
{
    if (!handle || !handle->materialized_re)
        return NULL;
    if (handle->layout_desc.type != TWIDDLE_LAYOUT_PRECOMPUTED)
        return NULL;
    if (handle->layout_desc.radix != 16)
        return NULL;
    if (handle->layout_desc.simd_arch != SIMD_ARCH_AVX2)
        return NULL;

    return ((const radix16_precomputed_block_avx2_t *)handle->layout_specific_data) + block_idx;
}

#endif // FFT_TWIDDLES_LAYOUT_EXTENSIONS_H