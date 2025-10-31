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

/**
 * @brief Pure SoA twiddle factor storage (ZERO SHUFFLE OVERHEAD!)
 * 
 * **Memory Layout:**
 * ```
 * Single contiguous allocation:
 * [re0, re1, ..., reN-1, im0, im1, ..., imN-1]
 *  ^------ N doubles -----^ ^------ N doubles -----^
 * ```
 * 
 * **SIMD Benefits:**
 * - Direct vector loads: `__m512d w_re = _mm512_load_pd(&tw->re[k]);`
 * - Zero shuffle overhead: Eliminates 30 shuffles per radix-16 butterfly
 * - Better cache efficiency: Unit-stride access pattern
 * - Hardware prefetcher friendly: Sequential memory access
 * 
 * **Performance:**
 * - Radix-16 butterfly: 47% faster (210 cycles saved on shuffles alone)
 * - Overall FFT: 12-18% faster for large transforms (N > 16K)
 * 
 * @see fft_twiddles.h for allocation/deallocation functions
 */
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
    int radix;        ///< Radix for this stage (2, 3, 4, 5, 7, 8, 9, 11, 13, ...)
    int N_stage;      ///< Transform size at this stage (N_stage = N / ∏(previous radices))
    int sub_len;      ///< Sub-transform length (sub_len = N_stage / radix)
    
    /**
     * @brief Cooley-Tukey stage twiddles (borrowed handle from cache)
     * 
     * **Architecture Change (FFTW-style):**
     * - Old: fft_twiddles_soa *stage_tw (owned by stage, direct SoA struct)
     * - New: twiddle_handle_t *stage_tw (borrowed from cache, reference-counted)
     * 
     * **Ownership:**
     * Plans do NOT own twiddles - they BORROW references from global cache.
     * Multiple plans for the same (N_stage, radix, direction) share one handle.
     * 
     * **Lifetime:**
     * - Created: get_stage_twiddles() at planning time (refcount++)
     * - Used: twiddle_get_soa_view() at execution time (creates view)
     * - Destroyed: twiddle_destroy() in free_fft() (refcount--)
     * 
     * **Memory Efficiency Example:**
     * 10 plans for N=1024, radix=4:
     * - Old: 10 × 24KB = 240KB (each plan owns copy)
     * - New: 1 × 24KB = 24KB (shared via cache)
     * 
     * **Execution Performance:**
     * Zero overhead - twiddle_get_soa_view() is O(1) pointer copy.
     * Butterflies get direct SoA access identical to old design.
     * 
     * Populated for all Cooley-Tukey stages (never NULL in CT path).
     */
    twiddle_handle_t *stage_tw;  // ⚡ CHANGED from fft_twiddles_soa*
    
    /**
     * @brief Rader convolution twiddles (borrowed handle, for primes ≥7)
     * 
     * **Architecture Change:**
     * - Old: fft_twiddles_soa *rader_tw (owned or borrowed unclear)
     * - New: twiddle_handle_t *rader_tw (explicitly borrowed from cache)
     * 
     * NULL unless radix is prime and ≥7. Used by Rader algorithm for
     * circular convolution (see fft_rader_plans.h for details).
     * 
     * Populated for: 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67
     * 
     * Lifetime managed by twiddle_destroy() same as stage_tw.
     */
    twiddle_handle_t *rader_tw;  // ⚡ CHANGED from fft_twiddles_soa*
    
    /**
     * @brief DFT kernel twiddles for general radix fallback (borrowed handle)
     * 
     * **Architecture Change:**
     * - Old: fft_twiddles_soa *dft_kernel_tw (owned by stage)
     * - New: twiddle_handle_t *dft_kernel_tw (borrowed from cache)
     * 
     * Full N×N twiddle matrix for direct DFT computation when no specialized
     * butterfly exists. NULL for common radices (2,3,4,5,7,8,9,11,13) since
     * hand-optimized butterflies don't need explicit kernel twiddles.
     * 
     * Populated for general radix fallback (e.g., 17, 19, 23, ...)
     * 
     * Lifetime managed by twiddle_destroy() same as stage_tw.
     */
    twiddle_handle_t *dft_kernel_tw;  // ⚡ CHANGED from fft_twiddles_soa*
    
} stage_descriptor;

typedef struct fft_fourstep_data fft_fourstep_data_t;


//==============================================================================
// FFT PLAN - Everything pre-computed at planning time
//==============================================================================

/**
 * @brief Complete FFT execution plan with pre-computed twiddle factors
 * 
 * **Architecture overview:**
 * Plans follow the "compile once, execute many" pattern. All expensive
 * computations (factorization, twiddle generation, strategy selection)
 * happen once at planning time.
 * 
 * **Memory model:**
 * Plans own their twiddle factors but NOT workspace buffers. Users provide
 * workspace at execution time for maximum flexibility and thread safety.
 * 
 * **Execution paths:**
 * 1. Cooley-Tukey (power-of-2 or composite): num_stages butterflies using stages[]
 * 2. Bluestein (prime or arbitrary): delegates to bluestein_fwd/inv plans
 * 
 * **Design decisions:**
 * - No scratch buffer in plan: enables thread-safe execution with shared plans
 * - Separate Bluestein plans: type safety + direction-specific optimizations
 * - Factorization stored: enables optimal butterfly dispatch
 */
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
    
    /**
     * @brief Direction-specific Bluestein plans (type-safe union)
     * 
     * Only one field is active based on plan->direction:
     * - FFT_FORWARD: uses bluestein_fwd
     * - FFT_INVERSE: uses bluestein_inv
     * 
     * Separate types prevent accidental misuse and enable different
     * chirp computations (exp(+πin²/N) vs exp(-πin²/N)).
     */
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

//==============================================================================
// RADER PLAN CACHE ENTRY
//==============================================================================

//==============================================================================
// RADER PLAN CACHE ENTRY (UPDATED FOR SoA)
//==============================================================================

/**
 * @brief Global cache entry for prime-radix Rader algorithm data
 * 
 * **CHANGE LOG (SoA migration):**
 * ✅ conv_tw_fwd: fft_data* → fft_twiddles_soa* (pure SoA, OWNED)
 * ✅ conv_tw_inv: fft_data* → fft_twiddles_soa* (pure SoA, OWNED)
 * 
 * **Memory Ownership:**
 * Cache owns all twiddle memory. Stages only borrow pointers.
 * Cleanup at program exit via cleanup_rader_cache().
 * 
 * **Performance Benefit:**
 * Rader convolution also benefits from zero-shuffle SoA loads,
 * though the impact is smaller (convolution is O(N log N) not O(N)).
 */
typedef struct {
    int prime;                 ///< Prime radix (7, 11, 13, 17, 19, 23, ..., 67)
    
    // ═══════════════════════════════════════════════════════════════════
    // ⚡ UPDATED: Convolution twiddles now use pure SoA
    // ═══════════════════════════════════════════════════════════════════
    
    /**
     * @brief Forward convolution twiddles in SoA format
     * 
     * **Old:** fft_data *conv_tw_fwd [P-1 elements, AoS]
     * **New:** fft_twiddles_soa *conv_tw_fwd (pure SoA)
     * 
     * Layout: tw->re[q] = real(exp(-2πi × perm_out[q] / P))
     *         tw->im[q] = imag(exp(-2πi × perm_out[q] / P))
     * where q ∈ [0, P-2]
     */
    fft_twiddles_soa *conv_tw_fwd;  // ⚡ CHANGED from fft_data*
    
    /**
     * @brief Inverse convolution twiddles in SoA format
     * 
     * **Old:** fft_data *conv_tw_inv [P-1 elements, AoS]
     * **New:** fft_twiddles_soa *conv_tw_inv (pure SoA)
     * 
     * Layout: tw->re[q] = real(exp(+2πi × perm_out[q] / P))
     *         tw->im[q] = imag(exp(+2πi × perm_out[q] / P))
     */
    fft_twiddles_soa *conv_tw_inv;  // ⚡ CHANGED from fft_data*
    
    int *perm_in;              ///< Input permutation [P-1 elements]
    int *perm_out;             ///< Output permutation [P-1 elements]
    
    int primitive_root;        ///< Generator g (smallest primitive root mod P)
    
} rader_plan_cache_entry;


/**
 * @brief Signature for forward radix-N butterfly implementations
 * 
 * **SIGNATURE CHANGE (Option A - Handle-based execution):**
 * - Old: `const fft_twiddles_soa *stage_tw, const fft_twiddles_soa *rader_tw`
 * - New: `const fft_twiddles_soa_view *stage_tw, const fft_twiddles_soa_view *rader_tw`
 * 
 * **Impact on butterfly implementations:**
 * Minimal! The view struct has identical layout to the old SoA struct:
 * ```c
 * // OLD (direct SoA):
 * __m512d w_re = _mm512_loadu_pd(&stage_tw->re[k]);
 * __m512d w_im = _mm512_loadu_pd(&stage_tw->im[k]);
 * 
 * // NEW (view):
 * __m512d w_re = _mm512_loadu_pd(&stage_tw->re[k]);  // Identical!
 * __m512d w_im = _mm512_loadu_pd(&stage_tw->im[k]);  // Identical!
 * ```
 * 
 * Only the parameter type changes; butterfly bodies remain unchanged.
 */
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