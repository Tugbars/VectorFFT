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
 * @author VectorFFT Team
 * @date 2025
 */

#include "fft_twiddles_layout_extensions.h"
#include "fft_twiddles_hybrid.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
// LAYOUT SELECTION
//==============================================================================

twiddle_layout_type_t choose_optimal_layout(
    int radix,
    simd_arch_t arch,
    int butterflies_per_stage,
    int prefer_precomputed)
{
    // Small radices (2, 3, 4, 5, 7) - strided is fine
    if (radix <= 7) {
        return TWIDDLE_LAYOUT_STRIDED;
    }
    
    int simd_width = get_simd_width(arch);
    
    // Very small stages - overhead dominates
    if (butterflies_per_stage < simd_width * 4) {
        return TWIDDLE_LAYOUT_STRIDED;
    }
    
    // Radix-16/32 with precomputed preference
    if (prefer_precomputed && (radix == 16 || radix == 32)) {
        if (butterflies_per_stage <= 16384) {
            return TWIDDLE_LAYOUT_PRECOMPUTED;
        }
    }
    
    // Default: blocked layout for large radices
    return TWIDDLE_LAYOUT_BLOCKED;
}

//==============================================================================
// RADIX-8 BLOCKED LAYOUT (AVX2) - Simple Example
//==============================================================================

/**
 * @brief Reorganize radix-8 twiddles into AVX2 blocked layout
 *
 * @details
 * Radix-8 has 7 twiddle factors per butterfly (W^1 through W^7).
 * 
 * STRIDED layout (input):
 *   tw_re: [W1[0..K-1], W2[0..K-1], ..., W7[0..K-1]]
 *   Access for butterfly k: tw_re[r*K + k] (stride K between factors)
 * 
 * BLOCKED layout (output):
 *   blocks[i]: [W1[4i..4i+3], W2[4i..4i+3], ..., W7[4i..4i+3]]
 *   Access for butterfly 4i: blocks[i].tw_data[r-1][re/im][lane]
 * 
 * @param handle Twiddle handle (any strategy - uses twiddle_get)
 * @return 0 on success, -1 on error
 */
static int materialize_radix8_blocked_avx2(twiddle_handle_t *handle)
{
    const int RADIX = 8;
    const int SIMD_WIDTH = 4;  // AVX2 doubles
    
    int K = handle->n / RADIX;  // Butterflies per stage
    int num_blocks = (K + SIMD_WIDTH - 1) / SIMD_WIDTH;  // Round up
    
    // ──────────────────────────────────────────────────────────────────
    // Allocate blocked storage
    // ──────────────────────────────────────────────────────────────────
    
    typedef struct {
        double tw_data[7][2][4];  // [factor_idx][re=0,im=1][lane]
    } __attribute__((aligned(64))) radix8_block_avx2_t;
    
    size_t alloc_size = num_blocks * sizeof(radix8_block_avx2_t);
    radix8_block_avx2_t *blocks = 
        (radix8_block_avx2_t *)aligned_alloc(64, alloc_size);
    
    if (!blocks) {
        return -1;
    }
    
    memset(blocks, 0, alloc_size);
    
    // ──────────────────────────────────────────────────────────────────
    // REORGANIZATION: Read strided, write blocked
    // ──────────────────────────────────────────────────────────────────
    
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        int k_base = block_idx * SIMD_WIDTH;
        
        // For each twiddle factor W^r (r = 1..7)
        for (int r = 1; r <= 7; r++) {
            // Gather 4 consecutive twiddles for this factor
            for (int lane = 0; lane < SIMD_WIDTH; lane++) {
                int k = k_base + lane;
                
                if (k < K) {
                    double tw_re, tw_im;
                    // Read from canonical storage (works for SIMPLE or FACTORED)
                    twiddle_get(handle, r, k, &tw_re, &tw_im);
                    
                    // Write to blocked storage
                    blocks[block_idx].tw_data[r-1][0][lane] = tw_re;
                    blocks[block_idx].tw_data[r-1][1][lane] = tw_im;
                } else {
                    // Padding for non-multiple-of-4
                    blocks[block_idx].tw_data[r-1][0][lane] = 0.0;
                    blocks[block_idx].tw_data[r-1][1][lane] = 0.0;
                }
            }
        }
    }
    
    // ──────────────────────────────────────────────────────────────────
    // Store in handle
    // ──────────────────────────────────────────────────────────────────
    
    handle->layout_specific_data = blocks;
    handle->owns_materialized = 1;
    
    // Populate layout descriptor
    handle->layout_desc.type = TWIDDLE_LAYOUT_BLOCKED;
    handle->layout_desc.simd_arch = SIMD_ARCH_AVX2;
    handle->layout_desc.simd_width = SIMD_WIDTH;
    handle->layout_desc.block_size = sizeof(radix8_block_avx2_t);
    handle->layout_desc.num_blocks = num_blocks;
    handle->layout_desc.total_size = alloc_size;
    handle->layout_desc.radix = RADIX;
    handle->layout_desc.num_twiddle_factors = 7;
    handle->layout_desc.butterflies_per_stage = K;
    handle->layout_desc.has_precomputed = 0;
    handle->layout_desc.precompute_offset = 0;
    
    // Don't populate materialized_re/im for blocked layout (use layout_specific_data)
    handle->materialized_re = NULL;
    handle->materialized_im = NULL;
    handle->materialized_count = 0;
    
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
    
    // ──────────────────────────────────────────────────────────────────
    // Allocate
    // ──────────────────────────────────────────────────────────────────
    
    size_t alloc_size = num_blocks * sizeof(radix16_twiddle_block_avx2_t);
    radix16_twiddle_block_avx2_t *blocks = 
        (radix16_twiddle_block_avx2_t *)aligned_alloc(64, alloc_size);
    
    if (!blocks) {
        return -1;
    }
    
    memset(blocks, 0, alloc_size);
    
    // ──────────────────────────────────────────────────────────────────
    // REORGANIZATION: Transform from strided to blocked
    // ──────────────────────────────────────────────────────────────────
    
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        int k_base = block_idx * SIMD_WIDTH;
        
        // For each of 15 twiddle factors
        for (int s = 1; s <= 15; s++) {
            // Gather 4 consecutive twiddles
            for (int lane = 0; lane < SIMD_WIDTH; lane++) {
                int k = k_base + lane;
                
                if (k < K) {
                    double tw_re, tw_im;
                    twiddle_get(handle, s, k, &tw_re, &tw_im);
                    
                    blocks[block_idx].tw_data[s-1][0][lane] = tw_re;
                    blocks[block_idx].tw_data[s-1][1][lane] = tw_im;
                } else {
                    blocks[block_idx].tw_data[s-1][0][lane] = 0.0;
                    blocks[block_idx].tw_data[s-1][1][lane] = 0.0;
                }
            }
        }
    }
    
    // ──────────────────────────────────────────────────────────────────
    // Store
    // ──────────────────────────────────────────────────────────────────
    
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
    
    if (!blocks) {
        return -1;
    }
    
    memset(blocks, 0, alloc_size);
    
    // Reorganization loop
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        int k_base = block_idx * SIMD_WIDTH;
        
        for (int s = 1; s <= 15; s++) {
            for (int lane = 0; lane < SIMD_WIDTH; lane++) {
                int k = k_base + lane;
                
                if (k < K) {
                    double tw_re, tw_im;
                    twiddle_get(handle, s, k, &tw_re, &tw_im);
                    
                    blocks[block_idx].tw_data[s-1][0][lane] = tw_re;
                    blocks[block_idx].tw_data[s-1][1][lane] = tw_im;
                } else {
                    blocks[block_idx].tw_data[s-1][0][lane] = 0.0;
                    blocks[block_idx].tw_data[s-1][1][lane] = 0.0;
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
    
    if (!blocks) {
        return -1;
    }
    
    memset(blocks, 0, alloc_size);
    
    // W_4 constants
    double sign = (handle->direction == FFT_FORWARD) ? -1.0 : 1.0;
    double w4_1_re = 0.0,     w4_1_im = sign * 1.0;   // ±i
    double w4_2_re = -1.0,    w4_2_im = 0.0;          // -1
    double w4_3_re = 0.0,     w4_3_im = sign * -1.0;  // ∓i
    
    // Reorganization with precomputation
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        int k_base = block_idx * SIMD_WIDTH;
        
        // Stage 1: Regular twiddles W_N^(s*k)
        for (int s = 1; s <= 15; s++) {
            for (int lane = 0; lane < SIMD_WIDTH; lane++) {
                int k = k_base + lane;
                
                if (k < K) {
                    double tw_re, tw_im;
                    twiddle_get(handle, s, k, &tw_re, &tw_im);
                    
                    blocks[block_idx].stage1[s-1][0][lane] = tw_re;
                    blocks[block_idx].stage1[s-1][1][lane] = tw_im;
                } else {
                    blocks[block_idx].stage1[s-1][0][lane] = 0.0;
                    blocks[block_idx].stage1[s-1][1][lane] = 0.0;
                }
            }
        }
        
        // Precompute W_4 products for s=4,8,12 (indices 3,7,11)
        for (int lane = 0; lane < SIMD_WIDTH; lane++) {
            int k = k_base + lane;
            
            if (k < K) {
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
    if (!handle) {
        return -1;
    }
    
    // Already materialized with this layout?
    if (handle->layout_desc.type == layout &&
        handle->layout_desc.simd_arch == arch &&
        handle->layout_specific_data != NULL) {
        return 0;  // Nothing to do
    }
    
    // Clean up old layout if present
    if (handle->layout_specific_data) {
        aligned_free(handle->layout_specific_data);
        handle->layout_specific_data = NULL;
    }
    
    // Dispatch to appropriate reorganization function
    int result = -1;
    
    if (layout == TWIDDLE_LAYOUT_BLOCKED) {
        if (handle->radix == 8 && arch == SIMD_ARCH_AVX2) {
            result = materialize_radix8_blocked_avx2(handle);
        }
        else if (handle->radix == 16 && arch == SIMD_ARCH_AVX2) {
            result = materialize_radix16_blocked_avx2(handle);
        }
#ifdef __AVX512F__
        else if (handle->radix == 16 && arch == SIMD_ARCH_AVX512) {
            result = materialize_radix16_blocked_avx512(handle);
        }
#endif
        else {
            // Radix/arch combination not implemented - fall back to strided
            result = 0;  // Success (use canonical format)
        }
    }
    else if (layout == TWIDDLE_LAYOUT_PRECOMPUTED) {
        if (handle->radix == 16 && arch == SIMD_ARCH_AVX2) {
            result = materialize_radix16_precomputed_avx2(handle);
        }
        else {
            result = 0;  // Fall back to regular blocked or strided
        }
    }
    else {
        // STRIDED or INTERLEAVED not implemented yet
        result = 0;
    }
    
    return result;
}

/**
 * @brief Materialize with automatic layout selection
 */
int twiddle_materialize_auto(twiddle_handle_t *handle, simd_arch_t arch)
{
    if (!handle) {
        return -1;
    }
    
    int K = handle->n / handle->radix;
    
    // Choose optimal layout
    twiddle_layout_type_t layout = choose_optimal_layout(
        handle->radix, arch, K, 1  // prefer_precomputed = 1
    );
    
    // Materialize with chosen layout
    return twiddle_materialize_with_layout(handle, layout, arch);
}
