/**
 * @file fft_fourstep.h
 * @brief Cache-aware four-step FFT with proper twiddle system integration
 */

#ifndef FFT_FOURSTEP_H
#define FFT_FOURSTEP_H

#include "fft_planning_types.h"
#include "fft_twiddles_hybrid.h"
#include <stddef.h>
#include <stdbool.h>

//==============================================================================
// CONFIGURATION
//==============================================================================

#ifndef FFT_FOURSTEP_MIN_SIZE
#define FFT_FOURSTEP_MIN_SIZE 2048
#endif

#ifndef FFT_FOURSTEP_MAX_RATIO
#define FFT_FOURSTEP_MAX_RATIO 4.0
#endif

//==============================================================================
// FOUR-STEP DATA STRUCTURE
//==============================================================================

/**
 * @brief Four-step FFT plan data (embedded in main fft_plan)
 * 
 * @details
 * CRITICAL: All resources created during PLANNING phase, reused during execution.
 * 
 * Memory ownership:
 * - plan_N1, plan_N2: Owned, freed by fft_fourstep_free_plan()
 * - twiddles_2d: Reference-counted handle, freed by twiddle_destroy()
 */
typedef struct {
    int N1;                          ///< First dimension (row length)
    int N2;                          ///< Second dimension (column height)
    
    fft_object plan_N1;              ///< Pre-computed plan for N1-sized FFTs
    fft_object plan_N2;              ///< Pre-computed plan for N2-sized FFTs
    
    twiddle_handle_t *twiddles_2d;   ///< Pre-computed 2D twiddles W_N^(k1*k2)
    
    double aspect_ratio;             ///< N2/N1 (diagnostics)
    
} fft_fourstep_data_t;

//==============================================================================
// PUBLIC API
//==============================================================================

/**
 * @brief Check if four-step FFT should be used
 */
bool fft_should_use_fourstep(int N, int stride);

/**
 * @brief Initialize four-step plan (called during planning phase)
 * 
 * @details
 * Creates sub-plans and twiddle handles. Must be called ONCE when
 * strategy is selected as FFT_EXEC_FOURSTEP.
 * 
 * @param[in,out] plan Main FFT plan (plan->fourstep populated)
 * @return 0 on success, -1 on error
 */
int fft_fourstep_init_plan(fft_object plan);

/**
 * @brief Free four-step resources
 * 
 * @param[in,out] plan FFT plan with fourstep data
 */
void fft_fourstep_free_plan(fft_object plan);

/**
 * @brief Calculate workspace size
 * 
 * @param[in] plan FFT plan with initialized fourstep data
 * @return Size in fft_data elements
 */
size_t fft_fourstep_workspace_size(const fft_object plan);

/**
 * @brief Execute four-step FFT
 * 
 * @param[in] plan FFT plan with strategy = FFT_EXEC_FOURSTEP
 * @param[in] input Input array (N elements)
 * @param[out] output Output array (N elements)
 * @param[in,out] workspace Scratch buffer
 * @return 0 on success, -1 on error
 */
int fft_exec_fourstep(
    fft_object plan,
    const fft_data* input,
    fft_data* output,
    fft_data* workspace
);

#endif // FFT_FOURSTEP_H
