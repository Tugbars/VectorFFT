/**
 * @file fft_twiddles_planner_api.c
 * @brief Thin convenience wrappers over reorganization system
 */

#include "fft_twiddles_planner_api.h"
#include "fft_twiddles_reorganization.h"

//==============================================================================
// CONVENIENCE WRAPPERS
//==============================================================================

twiddle_handle_t *get_stage_twiddles(
    int N_stage,
    int radix,
    fft_direction_t direction)
{
    // Use 3-parameter version (auto-selects SIMPLE/FACTORED)
    twiddle_handle_t *handle = twiddle_create(N_stage, radix, direction);
    if (!handle) {
        return NULL;
    }
    
    // Materialize with auto layout selection
    if (twiddle_materialize_auto(handle, SIMD_ARCH_AUTO) != 0) {
        twiddle_destroy(handle);
        return NULL;
    }
    
    return handle;
}

twiddle_handle_t *get_dft_kernel_twiddles(
    int radix,
    fft_direction_t direction)
{
    // For small DFT kernels, explicitly request SIMPLE strategy
    twiddle_handle_t *handle = twiddle_create_explicit(
        radix * radix, radix, direction, TWID_SIMPLE);
    if (!handle) {
        return NULL;
    }
    
    if (twiddle_materialize_auto(handle, SIMD_ARCH_AUTO) != 0) {
        twiddle_destroy(handle);
        return NULL;
    }
    
    return handle;
}

int twiddle_get_soa_view(
    const twiddle_handle_t *handle,
    fft_twiddles_soa_view *view)
{
    if (!handle || !view) return -1;
    if (!handle->materialized_re) return -1;
    
    view->re = handle->materialized_re;
    view->im = handle->materialized_im;
    view->count = handle->materialized_count;
    
    return 0;
}