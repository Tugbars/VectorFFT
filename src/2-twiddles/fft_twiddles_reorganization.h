/**
 * @file fft_twiddles_layout_extensions.h
 * @brief SIMD-optimized twiddle layout system for VectorFFT
 *
 * @details
 * This header defines the layout transformation system that converts canonical
 * twiddle storage (from fft_twiddles_hybrid.c) into SIMD-friendly memory layouts
 * optimized for different radices and architectures.
 *
 * ARCHITECTURE OVERVIEW:
 * ======================
 * 
 * Canonical Storage (hybrid.c) → Reorganization (reorganization.c) → Butterflies
 *        ↓                              ↓                                  ↓
 *   SIMPLE/FACTORED              BLOCKED layouts              AVX2/AVX-512 kernels
 *   O(√N) memory                 Sequential access            Zero shuffle overhead
 * 
 * LAYOUT PHILOSOPHY:
 * ==================
 * - Layout decisions happen at PLANNING time (pay once, amortize across executions)
 * - Butterflies get optimal memory access patterns (sequential loads, perfect prefetch)
 * - Different radices use different strategies based on twiddle count
 * - SIMD width affects layout granularity (4 for AVX2, 8 for AVX-512)
 * 
 * RADIX-SPECIFIC STRATEGIES:
 * ==========================
 * 
 * **Simple Radices (2, 4, 5, 8):** FLAT POINTER ARITHMETIC
 *   - Memory: [W1[0..K-1] | W2[0..K-1] | W3[0..K-1] | ...]
 *   - Access: tw->materialized_re[block * K + k]
 *   - Variants: BLOCKED2/3/4 (based on memory bandwidth vs derivation cost)
 *   - Example: Radix-8 stores W1,W2,W3,W4 when K≤256, only W1,W2 when K>256
 * 
 * **Complex Radices (16, 32):** STRUCTURED BLOCKS
 *   - Memory: Array of block structures, each containing all factors for SIMD_WIDTH butterflies
 *   - Access: get_radix16_block_avx2(handle, block_idx)->tw_data[factor][component][lane]
 *   - Variants: BLOCKED8, PRECOMPUTED (with intermediate twiddle products)
 *   - Example: Radix-16 BLOCKED8 stores 15 factors × 2 components × 8 lanes = 240 doubles/block
 * 
 * MEMORY TRADE-OFF EXAMPLE (N=1048576, radix=8):
 * ===============================================
 * ```
 * Canonical (FACTORED): ~2KB base twiddles (O(√N))
 * BLOCKED4:             3MB (4 blocks × 131072 × sizeof(double) × 2)
 * BLOCKED2:             1.5MB (2 blocks, derive other 5 at runtime)
 * 
 * Runtime cost: BLOCKED2 adds 2 FMA ops per butterfly, but saves 50% bandwidth
 * Decision: Use BLOCKED4 when K≤256 (computation-bound), BLOCKED2 when K>256 (bandwidth-bound)
 * ```
 *
 * @version 2.0 (Unified architecture for radix-2/4/5/8/16)
 * @date 2025
 */

#ifndef FFT_TWIDDLES_LAYOUT_EXTENSIONS_H
#define FFT_TWIDDLES_LAYOUT_EXTENSIONS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// LAYOUT TYPE ENUMERATION
//==============================================================================

/**
 * @brief Twiddle memory layout strategies
 */
typedef enum
{
    /**
     * @brief STRIDED layout (canonical, not materialized)
     *
     * Memory: [W1[0..K-1], W2[0..K-1], ..., W(R-1)[0..K-1]]
     * Access: W_s(k) = twiddle_get(handle, s, k, &re, &im)
     *
     * Pros: Minimal code, works for all radices
     * Cons: Poor cache locality (stride = K), reconstruction overhead
     *
     * Use: Unmaterialized handles only (fallback)
     */
    TWIDDLE_LAYOUT_STRIDED,

    /**
     * @brief BLOCKED layout (SIMD-optimized, sequential access)
     *
     * For radix-2/4/5/8 (flat pointers):
     *   Memory: [W1[0..K-1] | W2[0..K-1] | ... | Wn[0..K-1]]
     *   Access: tw->materialized_re[block_offset + k]
     *
     * For radix-16/32 (structured blocks):
     *   Memory: Array of blocks, each with all factors for SIMD_WIDTH butterflies
     *   Access: get_radix16_block_avx2(handle, k/SIMD_WIDTH)
     *
     * Pros: Sequential loads, perfect prefetch, cache-friendly
     * Cons: Requires planning-time reorganization
     *
     * Use: All radices, all architectures (default for execution)
     */
    TWIDDLE_LAYOUT_BLOCKED,

    /**
     * @brief INTERLEAVED layout (re/im pairs, deprecated)
     *
     * Memory: [W1_re[0], W1_im[0], W1_re[1], W1_im[1], ...]
     *
     * Pros: Better for non-FMA scalar code (legacy)
     * Cons: Complicates SoA SIMD access
     *
     * Use: Not recommended for modern CPUs (reserved for future use)
     */
    TWIDDLE_LAYOUT_INTERLEAVED,

    /**
     * @brief PRECOMPUTED layout (with intermediate twiddle products)
     *
     * Memory: [stage_twiddles | precomputed_products]
     * Example for radix-16:
     *   stage1[15 factors] + w4_products[W_4^j × W_N^(s*k)]
     *
     * Pros: Eliminates intermediate complex multiplies (saves 12 ops for radix-16)
     * Cons: 1.5-2× memory footprint vs BLOCKED
     *
     * Use: Radix-16/32 with K ≤ 16384 (when memory bandwidth available)
     */
    TWIDDLE_LAYOUT_PRECOMPUTED

} twiddle_layout_type_t;

//==============================================================================
// SIMD ARCHITECTURE ENUMERATION
//==============================================================================

/**
 * @brief Target SIMD architecture for layout optimization
 */
typedef enum
{
    SIMD_ARCH_AUTO,   ///< Auto-detect at runtime (queries CPUID)
    SIMD_ARCH_SCALAR, ///< No SIMD (width = 1, testing/fallback)
    SIMD_ARCH_SSE2,   ///< SSE2 (width = 2 doubles, legacy x86-64)
    SIMD_ARCH_AVX2,   ///< AVX2 + FMA (width = 4 doubles, mainstream)
    SIMD_ARCH_AVX512  ///< AVX-512F (width = 8 doubles, high-end Intel)
} simd_arch_t;

//==============================================================================
// LAYOUT DESCRIPTOR
//==============================================================================

/**
 * @brief Metadata describing a materialized twiddle layout
 * 
 * @details
 * Stored in twiddle_handle_t.layout_desc after materialization.
 * Contains all information needed to interpret materialized memory.
 */
typedef struct
{
    // Layout identification
    twiddle_layout_type_t type; ///< Layout strategy used
    simd_arch_t simd_arch;      ///< Target SIMD architecture
    int simd_width;             ///< Elements per SIMD vector (1/2/4/8)

    // Memory layout parameters
    int block_size;  ///< Size of one block in bytes
    int num_blocks;  ///< Total number of blocks
    size_t total_size; ///< Total allocation size in bytes (re + im combined)

    // FFT parameters
    int radix;                 ///< FFT radix (2/4/5/8/16/32)
    int num_twiddle_factors;   ///< Total factors (radix - 1)
    int butterflies_per_stage; ///< K = N_stage / radix

    // Precomputation metadata (for PRECOMPUTED layout only)
    int has_precomputed;   ///< 1 if includes precomputed products, 0 otherwise
    int precompute_offset; ///< Byte offset to precomputed section

} twiddle_layout_desc_t;

//==============================================================================
// HANDLE EXTENSIONS
//==============================================================================

/**
 * @brief Extensions to twiddle_handle_t (already defined in fft_twiddles_hybrid.h)
 *
 * The reorganization system uses these fields:
 *
 * ```c
 * typedef struct twiddle_handle {
 *     // ... existing fields (strategy, direction, n, radix, data union) ...
 *
 *     // Materialized arrays (flat SoA for radix-2/4/5/8)
 *     double *materialized_re;        // Real parts
 *     double *materialized_im;        // Imaginary parts
 *     int materialized_count;         // Total count
 *     int owns_materialized;          // 1=free on destroy, 0=borrowed
 *
 *     // Layout metadata
 *     twiddle_layout_desc_t layout_desc;  // Describes materialized layout
 *
 *     // Layout-specific data (for radix-16/32 block structures)
 *     void *layout_specific_data;         // Opaque pointer (radix16_twiddle_block_avx2_t*, etc.)
 *
 * } twiddle_handle_t;
 * ```
 *
 * @note These fields are already present - no modification needed!
 */

//==============================================================================
// RADIX-16 BLOCK STRUCTURES (Complex Radices Only)
//==============================================================================

/**
 * @brief Blocked twiddle storage for radix-16, AVX2
 *
 * @details
 * Each block stores all 15 twiddle factors for SIMD_WIDTH=4 butterflies.
 * Total: 15 factors × 2 components (re/im) × 4 doubles = 240 bytes
 *
 * Memory layout:
 * ```
 * tw_data[0][0][0..3] = W1_re for butterflies [k, k+1, k+2, k+3]
 * tw_data[0][1][0..3] = W1_im for butterflies [k, k+1, k+2, k+3]
 * tw_data[1][0][0..3] = W2_re for butterflies [k, k+1, k+2, k+3]
 * ...
 * tw_data[14][1][0..3] = W15_im for butterflies [k, k+1, k+2, k+3]
 * ```
 *
 * Access pattern:
 * ```c
 * const radix16_twiddle_block_avx2_t *block = get_radix16_block_avx2(handle, k/4);
 * __m256d W1_re = _mm256_load_pd(block->tw_data[0][0]);  // W1 real
 * __m256d W1_im = _mm256_load_pd(block->tw_data[0][1]);  // W1 imag
 * ```
 */
typedef struct
{
    double tw_data[15][2][4]; ///< [factor_idx 0..14][re=0,im=1][lane 0..3]
} __attribute__((aligned(64))) radix16_twiddle_block_avx2_t;

/**
 * @brief Blocked twiddle storage for radix-16, AVX-512
 *
 * @details
 * Each block stores all 15 twiddle factors for SIMD_WIDTH=8 butterflies.
 * Total: 15 factors × 2 components × 8 doubles = 480 bytes
 */
typedef struct
{
    double tw_data[15][2][8]; ///< [factor_idx 0..14][re=0,im=1][lane 0..7]
} __attribute__((aligned(64))) radix16_twiddle_block_avx512_t;

/**
 * @brief Precomputed layout for radix-16, AVX2
 *
 * @details
 * Includes both stage twiddles AND precomputed W_4 intermediate products.
 * 
 * Radix-16 decomposes as 4×4, requiring W_4 intermediate twiddles.
 * This layout precomputes W_4^j × W_N^(s*k) to eliminate 12 complex
 * multiplies per butterfly (saves ~8% execution time, costs ~1.7× memory).
 *
 * Memory layout:
 * ```
 * stage1[15][2][4]        = W_N^(s*k) for s=1..15
 * w4_products[3][3][2][4] = W_4^j × W_N^(s*k) for j=1,2,3 and s∈{4,8,12}
 * ```
 */
typedef struct
{
    // Stage twiddles: W_N^(s*k) for s=1..15
    double stage1[15][2][4]; ///< [factor_idx][re=0,im=1][lane]

    // Precomputed products: W_4^j × W_N^(s*k)
    // [w4_power-1][factor_group][re=0,im=1][lane]
    // w4_power ∈ {1,2,3}, factor_group ∈ {4,8,12}
    double w4_products[3][3][2][4];

} __attribute__((aligned(64))) radix16_precomputed_block_avx2_t;

//==============================================================================
// UTILITY FUNCTIONS
//==============================================================================

/**
 * @brief Get SIMD width for architecture
 *
 * @param arch SIMD architecture
 * @return Number of doubles per SIMD vector (1/2/4/8)
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
    case SIMD_ARCH_AUTO:
        // Should be resolved before calling this
        return 4; // Safe default (AVX2)
    default:
        return 1;
    }
}

//==============================================================================
// LAYOUT SELECTION API
//==============================================================================

/**
 * @brief Choose optimal layout for given radix and architecture
 *
 * @details
 * Decision tree:
 * - Radix ≤ 7: STRIDED (no materialization needed)
 * - Radix 8/16/32 with small K: PRECOMPUTED (if prefer_precomputed=1)
 * - Radix 8/16/32: BLOCKED (default for execution)
 *
 * @param radix FFT radix (2, 4, 5, 8, 16, 32)
 * @param arch SIMD architecture (affects block granularity)
 * @param butterflies_per_stage K = N_stage / radix
 * @param prefer_precomputed Allow precomputed layouts (higher memory, faster execution)
 *
 * @return Optimal layout type for this configuration
 */
twiddle_layout_type_t choose_optimal_layout(
    int radix,
    simd_arch_t arch,
    int butterflies_per_stage,
    int prefer_precomputed);

//==============================================================================
// MATERIALIZATION API
//==============================================================================

/**
 * @brief Materialize twiddles with specific layout
 *
 * @details
 * Transforms canonical storage (SIMPLE/FACTORED) into the requested layout.
 * For BLOCKED layouts:
 * - Radix-2/4/5/8: Populates materialized_re/im with flat arrays
 * - Radix-16/32: Populates layout_specific_data with block structures
 *
 * @param handle Twiddle handle (canonical storage)
 * @param layout Desired layout type
 * @param arch Target SIMD architecture
 *
 * @return 0 on success, -1 on error
 *
 * @note Idempotent: safe to call multiple times (does nothing if already materialized)
 */
int twiddle_materialize_with_layout(
    twiddle_handle_t *handle,
    twiddle_layout_type_t layout,
    simd_arch_t arch);

/**
 * @brief Materialize twiddles with automatic layout selection
 *
 * @details
 * Smart wrapper that calls choose_optimal_layout() and then materializes.
 * This is the recommended API for most use cases.
 *
 * @param handle Twiddle handle (canonical storage)
 * @param arch Target SIMD architecture (use SIMD_ARCH_AUTO for runtime detection)
 *
 * @return 0 on success, -1 on error
 *
 * Example:
 * ```c
 * twiddle_handle_t *tw = twiddle_create(N, radix, FFT_FORWARD, TWID_AUTO);
 * twiddle_materialize_auto(tw, SIMD_ARCH_AVX512);
 * // Now ready for butterfly execution
 * ```
 */
int twiddle_materialize_auto(
    twiddle_handle_t *handle,
    simd_arch_t arch);

//==============================================================================
// ACCESSOR FUNCTIONS (Radix-16 Block Structures Only)
//==============================================================================

/**
 * @brief Get pointer to twiddle block for radix-16, AVX2
 *
 * @param handle Materialized twiddle handle (must use BLOCKED layout)
 * @param block_idx Block index (k / 4 for AVX2)
 *
 * @return Pointer to block, or NULL on error
 *
 * Example:
 * ```c
 * for (int k = 0; k < K; k += 4) {
 *     const radix16_twiddle_block_avx2_t *block = 
 *         get_radix16_block_avx2(handle, k / 4);
 *     __m256d W1_re = _mm256_load_pd(block->tw_data[0][0]);
 *     // ... butterfly computation ...
 * }
 * ```
 */
static inline const radix16_twiddle_block_avx2_t *
get_radix16_block_avx2(const twiddle_handle_t *handle, int block_idx)
{
    if (!handle || !handle->layout_specific_data)
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
 * @brief Get pointer to twiddle block for radix-16, AVX-512
 *
 * @param handle Materialized twiddle handle
 * @param block_idx Block index (k / 8 for AVX-512)
 *
 * @return Pointer to block, or NULL on error
 */
static inline const radix16_twiddle_block_avx512_t *
get_radix16_block_avx512(const twiddle_handle_t *handle, int block_idx)
{
    if (!handle || !handle->layout_specific_data)
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
 * @brief Get pointer to precomputed block for radix-16, AVX2
 *
 * @param handle Materialized twiddle handle (must use PRECOMPUTED layout)
 * @param block_idx Block index (k / 4)
 *
 * @return Pointer to precomputed block, or NULL on error
 */
static inline const radix16_precomputed_block_avx2_t *
get_radix16_precomputed_avx2(const twiddle_handle_t *handle, int block_idx)
{
    if (!handle || !handle->layout_specific_data)
        return NULL;
    if (handle->layout_desc.type != TWIDDLE_LAYOUT_PRECOMPUTED)
        return NULL;
    if (handle->layout_desc.radix != 16)
        return NULL;
    if (handle->layout_desc.simd_arch != SIMD_ARCH_AVX2)
        return NULL;

    return ((const radix16_precomputed_block_avx2_t *)handle->layout_specific_data) + block_idx;
}

//==============================================================================
// SIMPLIFIED ACCESS FOR FLAT LAYOUTS (Radix-2/4/5/8)
//==============================================================================

/**
 * @brief Example: Get pointer to W1 block for radix-8 (flat layout)
 *
 * @details
 * For simple radices, just use pointer arithmetic on materialized_re/im:
 *
 * ```c
 * // Radix-8 BLOCKED4 (stores W1, W2, W3, W4)
 * int K = handle->n / 8;
 * const double *w1_re = handle->materialized_re + 0 * K;
 * const double *w2_re = handle->materialized_re + 1 * K;
 * const double *w3_re = handle->materialized_re + 2 * K;
 * const double *w4_re = handle->materialized_re + 3 * K;
 * // W5..W7 derived at runtime via sign flips and FMA
 * ```
 *
 * No accessor functions needed - direct pointer arithmetic is fast and clear.
 */

#ifdef __cplusplus
}
#endif

#endif // FFT_TWIDDLES_LAYOUT_EXTENSIONS_H