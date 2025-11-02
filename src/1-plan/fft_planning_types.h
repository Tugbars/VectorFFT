#ifndef FFT_PLANNING_TYPES_H
#define FFT_PLANNING_TYPES_H

#include <stddef.h>
#include <stdint.h>
#include "fft_twiddles_hybrid.h"
#include "fft_twiddles_planner_api.h"  // Provides: fft_twiddles_soa_view, twiddle_handle_t

#ifndef FFT_LOG_ERROR
#define FFT_LOG_ERROR(fmt, ...) fprintf(stderr, "[FFT ERROR] " fmt "\n", ##__VA_ARGS__)
#endif

//==============================================================================
// CONFIGURATION
//==============================================================================

#define MAX_FFT_STAGES 32  ///< Maximum factorization depth (supports up to 2^32)

//==============================================================================
// FORWARD DECLARATIONS
//==============================================================================

/**
 * @brief Opaque type for forward Bluestein transform plans
 * 
 * Separate types for forward/inverse enforce type safety and allow
 * direction-specific optimizations (different chirp signs).
 */
typedef struct bluestein_plan_forward_s bluestein_plan_forward;

/**
 * @brief Opaque type for inverse Bluestein transform plans
 * 
 * Implementation lives in bluestein.c to hide internal complexity.
 */
typedef struct bluestein_plan_inverse_s bluestein_plan_inverse;

typedef struct twiddle_handle twiddle_handle_t;  // From fft_twiddles_hybrid.h


//==============================================================================
// BASIC TYPES
//==============================================================================

/**
 * @brief Complex number in interleaved format (AoS) - for DATA, not twiddles
 * 
 * **Usage:**
 * - FFT input/output data (user-facing)
 * - Scratch buffers
 * - Legacy compatibility
 * 
 * **NOT used for twiddles** (twiddles use pure SoA for SIMD efficiency)
 */
typedef struct { double re, im; } fft_data;


typedef struct {
    double *re;    ///< Real components (aligned, contiguous)
    double *im;    ///< Imaginary components (aligned, contiguous)
    int count;     ///< Number of twiddle factors
} fft_twiddles_soa;

/**
 * @brief FFT direction: forward (analysis) or inverse (synthesis)
 * 
 * Values chosen to match sign in twiddle factor: exp(direction × 2πi/N)
 * This eliminates runtime branching in twiddle computation.
 */
typedef enum { 
    FFT_FORWARD = 1,   ///< Forward FFT (time → frequency), exp(-2πi/N)
    FFT_INVERSE = -1   ///< Inverse FFT (frequency → time), exp(+2πi/N)
} fft_direction_t;

//==============================================================================
// EXECUTION STRATEGY
//==============================================================================

/**
 * @brief Algorithm selected by planner based on transform size
 * 
 * Strategy determines memory requirements and execution path:
 * 
 * **Design rationale:**
 * - Power-of-2 sizes use bit-reversal (true in-place, zero workspace)
 * - Composite sizes use Stockham (cache-friendly, needs 1× workspace)
 * - Prime sizes use Bluestein (arbitrary N support, needs 3× workspace)
 * 
 * Strategy is fixed at plan time for optimal performance.
 */
typedef enum {
    FFT_EXEC_INPLACE_BITREV,    ///< Bit-reversal Cooley-Tukey (N = 2^k only, 0 workspace)
    FFT_EXEC_RECURSIVE_CT,      ///< Recursive Cooley-Tukey (any factorizable N, 2N workspace)
     FFT_EXEC_FOURSTEP,         
    FFT_EXEC_BLUESTEIN,         ///< Bluestein chirp-z (arbitrary N, 3M workspace)
    FFT_EXEC_OUT_OF_PLACE       ///< Generic out-of-place (reserved)
} fft_exec_strategy_t;

//==============================================================================
// STAGE DESCRIPTOR
//==============================================================================

typedef struct {
    int radix;
    int N_stage;
    int sub_len;
    
    twiddle_handle_t *stage_tw;
    twiddle_handle_t *rader_tw;
    twiddle_handle_t *dft_kernel_tw;
    
    // ✅ NEW: Pre-resolved butterfly function pointers
    butterfly_n1_func_t butterfly_n1;           // Twiddle-less variant
    butterfly_twiddle_func_t butterfly_twiddle; // Standard variant
    
} stage_descriptor;

typedef struct fft_fourstep_data fft_fourstep_data_t;


//==============================================================================
// FFT PLAN - Everything pre-computed at planning time
//==============================================================================

typedef struct fft_plan_struct {
    int n_input;               ///< Original input size requested by user
    int n_fft;                 ///< Actual FFT size (may differ for Bluestein: M = 2^⌈log₂(2N-1)⌉)
    fft_direction_t direction; ///< Transform direction (fixed at plan time)
    
    fft_exec_strategy_t strategy; ///< Execution algorithm selected by planner

    simd_arch_t simd_arch;
    
    // ─────────────────────────────────────────────────────────────────────
    // Cooley-Tukey decomposition data (used if strategy != BLUESTEIN)
    // ─────────────────────────────────────────────────────────────────────
    
    int num_stages;                      ///< Number of decomposition stages (0 for Bluestein)
    int factors[MAX_FFT_STAGES];         ///< Radix sequence [r₀, r₁, ..., rₖ] where N = r₀×r₁×...×rₖ
    stage_descriptor stages[MAX_FFT_STAGES]; ///< Per-stage twiddles and metadata

     fft_fourstep_data_t *fourstep;
    
    // ─────────────────────────────────────────────────────────────────────
    // Bluestein algorithm data (used if strategy == BLUESTEIN)
    // ─────────────────────────────────────────────────────────────────────
    union {
        bluestein_plan_forward *bluestein_fwd;  ///< Forward Bluestein plan (active if direction == FFT_FORWARD)
        bluestein_plan_inverse *bluestein_inv;  ///< Inverse Bluestein plan (active if direction == FFT_INVERSE)
        void *bluestein_generic;                ///< Generic pointer for NULL checks
    };
    
    
} fft_plan;

/**
 * @brief Opaque plan handle for public API
 * 
 * Users interact with fft_object pointers; internal structure is hidden.
 */
typedef fft_plan* fft_object;

typedef struct {
    int prime;                 ///< Prime radix (7, 11, 13, 17, 19, 23, ..., 67)
    
    
    /**
     * @brief Forward convolution twiddles in SoA format
     * 
     */
    fft_twiddles_soa *conv_tw_fwd;  // ⚡ CHANGED from fft_data*
    
    /**
     * @brief Inverse convolution twiddles in SoA format
     * 
     */
    fft_twiddles_soa *conv_tw_inv;  // ⚡ CHANGED from fft_data*
    
    int *perm_in;              ///< Input permutation [P-1 elements]
    int *perm_out;             ///< Output permutation [P-1 elements]
    
    int primitive_root;        ///< Generator g (smallest primitive root mod P)
    
} rader_plan_cache_entry;


typedef void (*radix_fv_func)(
    fft_data *restrict output,
    fft_data *restrict input,
    const fft_twiddles_soa_view *restrict stage_tw,  // ⚡ CHANGED
    const fft_twiddles_soa_view *restrict rader_tw,  // ⚡ CHANGED
    int sub_len
);

/**
 * @brief Signature for inverse radix-N butterfly implementations
 * 
 * Same signature change as radix_fv_func for consistency.
 */
typedef void (*radix_bv_func)(
    fft_data *restrict output,
    fft_data *restrict input,
    const fft_twiddles_soa_view *restrict stage_tw,  // ⚡ CHANGED
    const fft_twiddles_soa_view *restrict rader_tw,  // ⚡ CHANGED
    int sub_len
);

#endif // FFT_PLANNING_TYPES_H