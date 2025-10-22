//==============================================================================
// bluestein.h - Bluestein's Algorithm (Fully Separated Forward/Inverse)
//==============================================================================

#ifndef BLUESTEIN_H
#define BLUESTEIN_H

#include "../fft_plan/fft_planning_types.h"

//==============================================================================
// OPAQUE TYPES (completely separate structures)
//==============================================================================

typedef struct bluestein_plan_forward_s bluestein_plan_forward;
typedef struct bluestein_plan_inverse_s bluestein_plan_inverse;

//==============================================================================
// FORWARD FFT API
//==============================================================================

/**
 * @brief Create forward Bluestein plan for arbitrary N
 * 
 * Precomputes:
 * - Forward chirp: exp(+πin²/N)
 * - Forward kernel FFT
 * - Internal FFT plans for size M
 * 
 * @param N Signal length (arbitrary)
 * @return Forward plan, or NULL on failure
 */
bluestein_plan_forward* bluestein_plan_create_forward(int N);

/**
 * @brief Execute forward Bluestein transform
 * 
 * @param plan Forward plan
 * @param input Input signal (length N)
 * @param output Output signal (length N)
 * @param scratch Working buffer (3*M elements)
 * @param scratch_size Size of scratch for safety check
 * @return 0 on success, -1 on error
 */
int bluestein_exec_forward(
    bluestein_plan_forward *plan,
    const fft_data *input,
    fft_data *output,
    fft_data *scratch,
    size_t scratch_size
);

/**
 * @brief Free forward plan
 * 
 * @param plan Forward plan (safe to pass NULL)
 */
void bluestein_plan_free_forward(bluestein_plan_forward *plan);

//==============================================================================
// INVERSE FFT API
//==============================================================================

/**
 * @brief Create inverse Bluestein plan for arbitrary N
 * 
 * Precomputes:
 * - Inverse chirp: exp(-πin²/N)
 * - Inverse kernel FFT
 * - Internal FFT plans for size M
 * 
 * @param N Signal length (arbitrary)
 * @return Inverse plan, or NULL on failure
 */
bluestein_plan_inverse* bluestein_plan_create_inverse(int N);

/**
 * @brief Execute inverse Bluestein transform
 * 
 * @param plan Inverse plan
 * @param input Input signal (length N)
 * @param output Output signal (length N)
 * @param scratch Working buffer (3*M elements)
 * @param scratch_size Size of scratch for safety check
 * @return 0 on success, -1 on error
 */
int bluestein_exec_inverse(
    bluestein_plan_inverse *plan,
    const fft_data *input,
    fft_data *output,
    fft_data *scratch,
    size_t scratch_size
);

/**
 * @brief Free inverse plan
 * 
 * @param plan Inverse plan (safe to pass NULL)
 */
void bluestein_plan_free_inverse(bluestein_plan_inverse *plan);

//==============================================================================
// UTILITY FUNCTIONS (shared)
//==============================================================================

/**
 * @brief Calculate required scratch buffer size
 * 
 * @param N Signal length
 * @return Required scratch size in elements
 */
size_t bluestein_get_scratch_size(int N);

/**
 * @brief Get padded convolution size M
 * 
 * @param N Signal length
 * @return M = next power of 2 >= 2*N - 1
 */
int bluestein_get_padded_size(int N);

#endif // BLUESTEIN_H

// 1500