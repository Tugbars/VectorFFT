/**
 * @file fft_twiddles_planner_api.h
 * @brief FFTW-style planner API for hybrid twiddle system with execution optimization
 * 
 * **Architecture Philosophy:**
 * 
 * Plans store handles (borrowed references from cache), but for EXECUTION,
 * we need direct SoA access for butterfly performance. This creates a tension:
 * 
 * - Cache efficiency: Use factored mode for large N (O(√N) memory)
 * - Execution speed: Butterflies need direct re/im array access (no reconstruction)
 * 
 * **Solution: Lazy Materialization**
 * 
 * Handles can exist in two states:
 * 
 * 1. **Unmaterialized** (factored): Minimal memory, requires reconstruction
 *    - Used for: One-off computations, rarely-used sizes
 *    - Access via: twiddle_get(handle, r, k, &re, &im)
 * 
 * 2. **Materialized** (SoA arrays): Full memory, direct access
 *    - Used for: Execution-critical paths (butterfly kernels)
 *    - Access via: twiddle_get_soa_view(handle, &view)
 * 
 * Materialization happens on-demand when planner requests it, then cached
 * alongside the handle. Multiple plans can share the same materialized data.
 * 
 * **Memory Trade-off Example (N=1048576, radix=4):**
 * ```
 * Factored:     2KB base twiddles (O(√N))
 * Materialized: 24MB full SoA arrays (3 × 262144 × sizeof(double))
 * 
 * If 10 plans need this: 2KB + 24MB (shared) vs 240MB (without cache)
 * ```
 * 
 * **Performance Characteristics:**
 * ```
 * Planning:    O(N) to materialize (done once, amortized)
 * Execution:   O(1) to get view (pointer assignment)
 * Memory:      O(N) per unique (N, radix, direction) tuple
 * Cache hits:  O(1) to increment refcount
 * ```
 */

#ifndef FFT_TWIDDLES_PLANNER_API_H
#define FFT_TWIDDLES_PLANNER_API_H

#include "fft_twiddles_hybrid.h"

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// SOA VIEW (Stack-allocated, lightweight)
//==============================================================================

/**
 * @brief Lightweight SoA view into twiddle data
 * 
 * This struct does NOT own the data - it's just a window into the handle's
 * materialized arrays. Safe to stack-allocate and pass to butterflies.
 * 
 * **Lifetime:**
 * - Created: At execution time via twiddle_get_soa_view()
 * - Used: Passed to butterfly kernels
 * - Destroyed: Automatic (stack cleanup, no explicit free needed)
 * 
 * **Thread Safety:**
 * Multiple threads can safely create views from the same handle and use
 * them concurrently (read-only access to shared materialized data).
 * 
 * **Memory Layout:**
 * ```
 * re[0..count-1]: Real parts (aligned, handle-owned)
 * im[0..count-1]: Imaginary parts (aligned, handle-owned)
 * ```
 */
typedef struct {
    const double *re;    ///< Real parts (borrowed, read-only)
    const double *im;    ///< Imaginary parts (borrowed, read-only)
    int count;           ///< Number of twiddles in view
} fft_twiddles_soa_view;

//==============================================================================
// STAGE TWIDDLES (Cooley-Tukey)
//==============================================================================

/**
 * @brief Get or create stage twiddles for Cooley-Tukey execution
 * 
 * **What This Returns:**
 * A handle containing materialized twiddles W^(r*k) for:
 * - r ∈ [1, radix-1]
 * - k ∈ [0, sub_len-1]
 * - sub_len = N_stage / radix
 * 
 * **Materialization Strategy:**
 * This function ALWAYS materializes the twiddles into full SoA arrays,
 * even for large N. This is necessary because butterflies need direct
 * array access for performance. The materialized data is cached, so
 * subsequent calls for the same (N_stage, radix, direction) are free.
 * 
 * **Cache Behavior:**
 * ```
 * First call for (1024, 4, FWD):
 *   1. Check cache → miss
 *   2. Create factored handle → O(√N) memory
 *   3. Materialize SoA arrays → O(N) time, O(N) memory
 *   4. Insert into cache → refcount = 1
 *   5. Return handle
 * 
 * Second call for (1024, 4, FWD):
 *   1. Check cache → hit!
 *   2. Increment refcount → refcount = 2
 *   3. Return same handle (SoA already materialized)
 * ```
 * 
 * **Memory Ownership:**
 * - Handle: BORROWED (caller does not own, must call twiddle_destroy)
 * - SoA arrays: OWNED by cache (multiple plans can share)
 * - View: Stack-allocated at execution time (no allocation)
 * 
 * **Thread Safety:**
 * Safe to call from multiple threads. Cache uses internal locking.
 * Returned handle is immutable after creation (safe concurrent reads).
 * 
 * **Usage Pattern:**
 * ```c
 * // Planning phase (once per plan)
 * stage->tw = get_stage_twiddles(1024, 4, FFT_FORWARD);
 * 
 * // Execution phase (many times)
 * fft_twiddles_soa_view view;
 * twiddle_get_soa_view(stage->tw, &view);
 * radix4_butterfly(..., &view, ...);
 * 
 * // Cleanup (once per plan)
 * twiddle_destroy(stage->tw);  // Decrements refcount
 * ```
 * 
 * @param N_stage Transform size for this stage (must be divisible by radix)
 * @param radix Radix for this Cooley-Tukey stage (2, 3, 4, 5, 7, 8, etc.)
 * @param direction FFT_FORWARD or FFT_INVERSE
 * @return Handle with materialized SoA (borrowed), or NULL on failure
 * 
 * @note Must call twiddle_destroy() later to release reference
 */
twiddle_handle_t *get_stage_twiddles(
    int N_stage,
    int radix,
    fft_direction_t direction
);

/**
 * @brief Get DFT kernel twiddles for general radix fallback
 * 
 * **What This Returns:**
 * A handle containing full N×N DFT matrix W^(j*k) for j,k ∈ [0, N-1].
 * Used only for radices without specialized butterfly implementations.
 * 
 * **When to Use:**
 * ```c
 * if (!has_specialized_butterfly(radix)) {
 *     stage->dft_kernel = get_dft_kernel_twiddles(radix, direction);
 * } else {
 *     stage->dft_kernel = NULL;  // Not needed
 * }
 * ```
 * 
 * **Memory Cost:**
 * For radix N: 2 × N² × sizeof(double) bytes
 * Examples:
 * - Radix-17: ~4.6KB (manageable)
 * - Radix-31: ~15KB (still reasonable)
 * - Radix-97: ~150KB (getting large, but rare)
 * 
 * **Materialization:**
 * Always materialized (small enough that factorization doesn't help).
 * 
 * @param radix DFT size (must match radix in execution)
 * @param direction FFT_FORWARD or FFT_INVERSE
 * @return Handle with materialized SoA (borrowed), or NULL on failure
 * 
 * @note Must call twiddle_destroy() later to release reference
 */
twiddle_handle_t *get_dft_kernel_twiddles(
    int radix,
    fft_direction_t direction
);

//==============================================================================
// SOA VIEW EXTRACTION (Execution-time, zero overhead)
//==============================================================================

/**
 * @brief Extract SoA view from materialized handle (execution path)
 * 
 * **What This Does:**
 * Creates a lightweight view struct pointing to the handle's materialized
 * SoA arrays. This is a simple pointer assignment - no allocation, no copy.
 * 
 * **Preconditions:**
 * - Handle must be materialized (true for handles from get_stage_twiddles)
 * - Handle must be valid (non-NULL, refcount > 0)
 * 
 * **Performance:**
 * O(1) constant time - just loads pointers from handle struct.
 * 
 * **Thread Safety:**
 * Safe to call from multiple threads with same handle (read-only operation).
 * Multiple views can point to the same materialized data.
 * 
 * **Usage Pattern:**
 * ```c
 * void execute_stage(const fft_stage_t *stage, ...) {
 *     // Create view (stack-allocated, no cleanup needed)
 *     fft_twiddles_soa_view tw_view;
 *     twiddle_get_soa_view(stage->tw, &tw_view);
 *     
 *     // Pass to butterfly (expects const fft_twiddles_soa_view*)
 *     radix8_butterfly(..., &tw_view, ...);
 *     
 *     // View goes out of scope automatically (no free needed)
 * }
 * ```
 * 
 * **Layout Guarantee:**
 * For stage twiddles:
 * ```
 * view.re[(r-1) * sub_len + k] = real(W^(r*k))
 * view.im[(r-1) * sub_len + k] = imag(W^(r*k))
 * where r ∈ [1, radix-1], k ∈ [0, sub_len-1]
 * ```
 * 
 * @param handle Materialized twiddle handle (from get_stage_twiddles)
 * @param view Output view struct (caller-allocated, typically stack)
 * @return 0 on success, -1 if handle not materialized or invalid
 */
int twiddle_get_soa_view(
    const twiddle_handle_t *handle,
    fft_twiddles_soa_view *view
);

//==============================================================================
// MATERIALIZATION CONTROL (Advanced)
//==============================================================================

/**
 * @brief Check if handle has materialized SoA arrays
 * 
 * Useful for debugging and profiling. All handles from get_stage_twiddles()
 * should return true.
 * 
 * @param handle Twiddle handle to check
 * @return 1 if materialized, 0 otherwise
 */
int twiddle_is_materialized(const twiddle_handle_t *handle);

/**
 * @brief Force materialization of handle (if not already materialized)
 * 
 * Normally not needed (get_stage_twiddles auto-materializes), but useful for:
 * - Pre-warming cache before timing-critical code
 * - Converting factored handles to execution-ready form
 * 
 * **Cost:**
 * - Already materialized: O(1) no-op
 * - Needs materialization: O(N) time and memory
 * 
 * @param handle Handle to materialize (modified in-place)
 * @return 0 on success, -1 on failure (allocation error)
 */
int twiddle_materialize(twiddle_handle_t *handle);

#ifdef __cplusplus
}
#endif

#endif // FFT_TWIDDLES_PLANNER_API_H