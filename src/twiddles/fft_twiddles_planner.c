/**
 * @file fft_twiddles_planner_api.c
 * @brief Implementation of planner API with lazy materialization
 */

#include "fft_twiddles_planner_api.h"
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <malloc.h>
#define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_free(ptr) free(ptr)
#endif

//==============================================================================
// MATERIALIZATION SUPPORT
//==============================================================================

/**
 * @brief Add materialized SoA arrays to twiddle_handle_t
 * 
 * NOTE: This requires modifying fft_twiddles_hybrid.h to add these fields:
 * 
 * typedef struct twiddle_handle {
 *     // ... existing fields ...
 *     
 *     // Materialized SoA for execution (NULL if not materialized)
 *     double *materialized_re;
 *     double *materialized_im;
 *     int materialized_count;
 * } twiddle_handle_t;
 */

int twiddle_materialize(twiddle_handle_t *handle)
{
    if (!handle) return -1;
    
    // Already materialized?
    if (handle->materialized_re != NULL) {
        return 0;  // Nothing to do
    }
    
    // ──────────────────────────────────────────────────────────────────
    // Calculate twiddle count based on handle type
    // ──────────────────────────────────────────────────────────────────
    
    int count;
    
    if (handle->strategy == TWID_SIMPLE) {
        // Simple mode: already in SoA format, just reference it
        // (Could skip allocation and point directly, but for consistency...)
        count = (handle->radix - 1) * (handle->n / handle->radix);
        
        // For simple mode, we can just point to existing data (zero-copy)
        handle->materialized_re = handle->data.simple.re;
        handle->materialized_im = handle->data.simple.im;
        handle->materialized_count = count;
        handle->owns_materialized = 0;  //  - borrowed pointers, don't free

        return 0;
    }
    else if (handle->strategy == TWID_FACTORED) {
        // Factored mode: need to reconstruct all twiddles
        count = (handle->radix - 1) * (handle->n / handle->radix);
    }
    else {
        return -1;  // Unknown strategy
    }
    
    // ──────────────────────────────────────────────────────────────────
    // Allocate SoA arrays for materialized twiddles
    // ──────────────────────────────────────────────────────────────────
    
    double *re = (double *)aligned_alloc(64, count * sizeof(double));
    double *im = (double *)aligned_alloc(64, count * sizeof(double));
    
    if (!re || !im) {
        aligned_free(re);
        aligned_free(im);
        return -1;
    }
    
    // ──────────────────────────────────────────────────────────────────
    // Reconstruct all twiddles from factored representation
    // ──────────────────────────────────────────────────────────────────
    
    int sub_len = handle->n / handle->radix;
    
    for (int r = 1; r < handle->radix; r++) {
        for (int k = 0; k < sub_len; k++) {
            int idx = (r - 1) * sub_len + k;
            
            // Use hybrid system's getter (handles factored reconstruction)
            twiddle_get(handle, r, k, &re[idx], &im[idx]);
        }
    }
    
    // ──────────────────────────────────────────────────────────────────
    // Store materialized arrays in handle
    // ──────────────────────────────────────────────────────────────────
    
    handle->materialized_re = re;
    handle->materialized_im = im;
    handle->materialized_count = count;
    handle->owns_materialized = 1;  //  allocated, must free

    return 0;
}

int twiddle_is_materialized(const twiddle_handle_t *handle)
{
    if (!handle) return 0;
    return (handle->materialized_re != NULL);
}

//==============================================================================
// PLANNER API IMPLEMENTATION
//==============================================================================

twiddle_handle_t *get_stage_twiddles(
    int N_stage,
    int radix,
    fft_direction_t direction)
{
    // ──────────────────────────────────────────────────────────────────
    // Create or get cached handle (uses hybrid system)
    // ──────────────────────────────────────────────────────────────────
    
    twiddle_handle_t *handle = twiddle_create(N_stage, radix, direction);
    if (!handle) {
        return NULL;
    }
    
    // ──────────────────────────────────────────────────────────────────
    // Ensure handle is materialized for execution
    // ──────────────────────────────────────────────────────────────────
    
    if (twiddle_materialize(handle) != 0) {
        twiddle_destroy(handle);  // Release reference on failure
        return NULL;
    }
    
    return handle;  // Borrowed reference (caller must call twiddle_destroy)
}

twiddle_handle_t *get_dft_kernel_twiddles(
    int radix,
    fft_direction_t direction)
{
    // ──────────────────────────────────────────────────────────────────
    // For DFT kernel, we need the full N×N matrix
    // ──────────────────────────────────────────────────────────────────
    
    // Use radix as both n and radix (special case for DFT kernel)
    twiddle_handle_t *handle = twiddle_create(radix * radix, radix, direction);
    if (!handle) {
        return NULL;
    }
    
    // DFT kernels are small, always materialize
    if (twiddle_materialize(handle) != 0) {
        twiddle_destroy(handle);
        return NULL;
    }
    
    return handle;
}

//==============================================================================
// SOA VIEW EXTRACTION
//==============================================================================

int twiddle_get_soa_view(
    const twiddle_handle_t *handle,
    fft_twiddles_soa_view *view)
{
    // ──────────────────────────────────────────────────────────────────
    // Validate inputs
    // ──────────────────────────────────────────────────────────────────
    
    if (!handle || !view) {
        return -1;
    }
    
    if (!twiddle_is_materialized(handle)) {
        return -1;  // Handle not materialized (shouldn't happen with get_stage_twiddles)
    }
    
    // ──────────────────────────────────────────────────────────────────
    // Create view (zero-overhead pointer assignment)
    // ──────────────────────────────────────────────────────────────────
    
    view->re = handle->materialized_re;
    view->im = handle->materialized_im;
    view->count = handle->materialized_count;
    
    return 0;
}