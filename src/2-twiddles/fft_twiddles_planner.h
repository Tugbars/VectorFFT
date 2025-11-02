/**
 * @file fft_twiddles_planner_api.h
 * @brief Convenience wrappers for twiddle reorganization system
 * 
 * **Architecture:**
 * 
 * This is a THIN WRAPPER over the twiddle reorganization system.
 * It provides FFTW-style convenience functions for common operations.
 * 
 * Real work happens in:
 * - fft_twiddles_hybrid.c     (canonical storage)
 * - fft_twiddles_reorganization.c (blocked layout materialization)
 */

#ifndef FFT_TWIDDLES_PLANNER_API_H
#define FFT_TWIDDLES_PLANNER_API_H

#include "fft_twiddles_hybrid.h"
#include "fft_twiddles_reorganization.h"

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// SOA VIEW (For radix-2 convenience)
//==============================================================================

/**
 * @brief Lightweight view into materialized SoA twiddles
 * @details Stack-allocated, just holds pointers (borrowed references)
 */
typedef struct {
    const double *re;    ///< Real parts (borrowed, read-only)
    const double *im;    ///< Imaginary parts (borrowed, read-only)
    int count;           ///< Number of twiddles in view
} fft_twiddles_soa_view;

//==============================================================================
// CONVENIENCE WRAPPERS
//==============================================================================

/**
 * @brief Get twiddles for a Cooley-Tukey stage (auto-materializes)
 * 
 * @param N_stage Stage size (N/radix for CT decomposition)
 * @param radix Butterfly radix (2, 4, 5, 8, 16, etc.)
 * @param direction FFT_FORWARD or FFT_BACKWARD
 * @return Handle with materialized blocked layout (NULL on error)
 * 
 * @note Automatically selects optimal layout (BLOCKED2/4/8 based on K)
 * @note Caller must call twiddle_destroy() when done
 */
twiddle_handle_t *get_stage_twiddles(
    int N_stage,
    int radix,
    fft_direction_t direction
);

/**
 * @brief Get twiddles for a small DFT kernel
 * 
 * @param radix Kernel radix
 * @param direction FFT_FORWARD or FFT_BACKWARD
 * @return Handle with materialized twiddles (NULL on error)
 * 
 * @note DFT kernels are small, always use SIMPLE strategy
 */
twiddle_handle_t *get_dft_kernel_twiddles(
    int radix,
    fft_direction_t direction
);

/**
 * @brief Extract SoA view from materialized handle
 * 
 * @param handle Materialized twiddle handle
 * @param view Output view (stack-allocated by caller)
 * @return 0 on success, -1 on error
 * 
 * @note Primarily for radix-2, which expects fft_twiddles_soa struct
 * @note View is valid as long as handle is alive
 */
int twiddle_get_soa_view(
    const twiddle_handle_t *handle,
    fft_twiddles_soa_view *view
);

/**
 * @brief Check if handle has been materialized
 * 
 * @param handle Twiddle handle
 * @return 1 if materialized, 0 otherwise
 * 
 * @note Mostly for debugging/testing
 */
int twiddle_is_materialized(const twiddle_handle_t *handle);

#ifdef __cplusplus
}
#endif

#endif // FFT_TWIDDLES_PLANNER_API_H