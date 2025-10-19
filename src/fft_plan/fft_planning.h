//==============================================================================
// fft_planning.h
// MAIN PLANNING ORCHESTRATOR - The "fft_init" umbrella
//==============================================================================

/**
 * @file fft_planning.h
 * @brief Public interface for FFT plan creation and management
 * 
 * **Architecture Overview:**
 * This module provides the top-level API for creating FFT execution plans.
 * Planning uses a sophisticated multi-phase approach:
 * 
 * 1. **Prime Factorization** - Decompose N into prime factors
 * 2. **Dynamic Programming Packing** - Combine primes into optimal radix sequence
 * 3. **Twiddle Computation** - Pre-compute all rotation factors
 * 4. **Rader Integration** - Fetch convolution twiddles for prime radices
 * 5. **Strategy Selection** - Choose execution algorithm (in-place, Stockham, Bluestein)
 * 
 * **Key Features:**
 * - Optimal factorization using DP (not greedy)
 * - Cost-aware radix selection
 * - Intelligent fallbacks for unfactorizable sizes
 * - Zero-copy workspace model (user provides at execution time)
 * - Thread-safe execution with shared plans
 * 
 * **Usage Pattern:**
 * ```c
 * // Create plan once (expensive, ~1-10ms)
 * fft_object plan = fft_init(1024, FFT_FORWARD);
 * 
 * // Query workspace requirement
 * size_t ws_size = fft_get_workspace_size(plan);
 * fft_data *workspace = ws_size ? malloc(ws_size * sizeof(fft_data)) : NULL;
 * 
 * // Execute many times (fast, ~10-100μs)
 * for (int i = 0; i < 1000; i++) {
 *     fft_exec_dft(plan, input, output, workspace);
 * }
 * 
 * // Cleanup
 * free(workspace);
 * free_fft(plan);
 * ```
 */

#ifndef FFT_PLANNING_H
#define FFT_PLANNING_H

#include "fft_planning_types.h"

//==============================================================================
// PLAN CREATION & DESTRUCTION
//==============================================================================

/**
 * @brief Create optimized FFT plan for given size and direction
 * 
 * **Planning Algorithm:**
 * 
 * Phase 1: Prime Factorization
 * - Decompose N into [p₁, p₂, ..., pₖ] via trial division
 * - Always succeeds (every integer has unique prime factorization)
 * 
 * Phase 2: Dynamic Programming Radix Packing
 * - Combine consecutive primes into implemented radices
 * - Minimize cost function (execution time proxy)
 * - Example: [2,2,2,3,3] → [8,9] (not [2,2,2,3,3])
 * 
 * Phase 3: Strategy Selection
 * - Power-of-2 → In-place bit-reversal (zero workspace)
 * - Composite → Stockham auto-sort (N workspace)
 * - Unfactorizable → Bluestein (3M workspace)
 * 
 * Phase 4: Stage Construction
 * - Compute Cooley-Tukey twiddles (Twiddle Manager)
 * - Fetch Rader convolution twiddles (Rader Manager, if prime radix)
 * - Store in immutable plan structure
 * 
 * **Supported Sizes:**
 * - Power-of-2: 2, 4, 8, ..., 2³⁰ (true in-place)
 * - Composite: Products of {2,3,4,5,7,8,9,11,13} (efficient Cooley-Tukey)
 * - Prime: 1021, 2053, ... (Bluestein fallback)
 * - Arbitrary: Any N > 0 (via Bluestein if needed)
 * 
 * **Performance:**
 * - Planning time: 0.1-10ms (O(N) for twiddle computation)
 * - Amortized over many executions (plan reuse)
 * - Plans are immutable → thread-safe execution
 * 
 * **Memory:**
 * - Plan size: O(N) for twiddles + O(log N) for metadata
 * - No workspace allocated (user provides at execution time)
 * - Typical: 16 KB for N=1024, 160 KB for N=10000
 * 
 * **Thread Safety:**
 * - Planning: NOT thread-safe (uses global Rader cache with locks)
 * - Multiple threads can plan different sizes safely
 * - Don't plan same size from multiple threads without external lock
 * 
 * **Error Handling:**
 * Returns NULL on failure:
 * - N ≤ 0 (invalid size)
 * - Invalid direction
 * - Memory allocation failure
 * - Internal planner error (very rare)
 * 
 * @param N Transform size (must be positive integer)
 * @param direction FFT_FORWARD for analysis (time→frequency), 
 *                  FFT_INVERSE for synthesis (frequency→time)
 * @return Opaque plan handle, or NULL on failure
 * 
 * @see free_fft() to destroy plan
 * @see fft_get_workspace_size() to query workspace requirement
 * @see fft_can_execute_inplace() to check if in-place supported
 */
fft_object fft_init(int N, fft_direction_t direction);

/**
 * @brief Destroy FFT plan and free all associated resources
 * 
 * **Cleanup Responsibilities:**
 * - Stage twiddles: Freed via Twiddle Manager
 * - Rader twiddles: NOT freed (borrowed from global cache)
 * - Bluestein sub-plans: Freed recursively
 * - Plan structure: Freed
 * 
 * **Memory Guarantees:**
 * - No leaks: All owned memory freed
 * - No double-free: Borrowed pointers (Rader cache) not freed
 * - NULL-safe: Safe to pass NULL (no-op)
 * - Idempotent: Safe to call multiple times (if you track NULL)
 * 
 * **Thread Safety:**
 * - Safe to free from any thread
 * - Don't free while another thread is executing with this plan!
 * - Undefined behavior (race condition) if freed during execution
 * 
 * **Usage Pattern:**
 * ```c
 * fft_object plan = fft_init(1024, FFT_FORWARD);
 * // ... use plan ...
 * free_fft(plan);
 * plan = NULL;  // Good practice to prevent double-free
 * ```
 * 
 * @param plan Plan to free (NULL is safe, no-op)
 */
void free_fft(fft_object plan);

//==============================================================================
// PLAN QUERY FUNCTIONS
//==============================================================================

/**
 * @brief Check if plan supports true in-place execution
 * 
 * **In-place Support:**
 * - ✅ Power-of-2 sizes: Bit-reversal algorithm, zero workspace
 * - ❌ Mixed-radix: Stockham algorithm, needs N workspace
 * - ❌ Bluestein: Chirp-z algorithm, needs 3M workspace
 * 
 * **Why Power-of-2 Only?**
 * In-place execution requires careful butterfly ordering to avoid
 * overwriting data before it's read. Bit-reversal permutation enables
 * this for radix-2 decompositions. Mixed-radix requires reordering
 * between stages → can't do in-place without temporary buffer.
 * 
 * **Usage:**
 * ```c
 * fft_object plan = fft_init(1024, FFT_FORWARD);
 * 
 * if (fft_can_execute_inplace(plan)) {
 *     // Power-of-2: use in-place API (zero extra memory)
 *     fft_data data[1024];
 *     fft_exec_inplace(plan, data);
 * } else {
 *     // Non-power-of-2: need workspace
 *     fft_data input[1024], output[1024];
 *     size_t ws = fft_get_workspace_size(plan);
 *     fft_data *workspace = malloc(ws * sizeof(fft_data));
 *     fft_exec_dft(plan, input, output, workspace);
 *     free(workspace);
 * }
 * ```
 * 
 * @param plan FFT plan to query
 * @return 1 if fft_exec_inplace() will succeed, 0 otherwise
 */
int fft_can_execute_inplace(fft_object plan);

/**
 * @brief Query workspace buffer size required for execution
 * 
 * **Workspace Requirements:**
 * 
 * | Strategy        | Size              | Example (N=1024) | Notes                    |
 * |-----------------|-------------------|------------------|--------------------------|
 * | In-place        | 0                 | 0 bytes          | Power-of-2 only          |
 * | Stockham        | N                 | 16 KB            | Mixed-radix              |
 * | Bluestein       | 3M (M≈2N)         | ~48 KB           | Primes, arbitrary N      |
 * 
 * **Memory Allocation Strategies:**
 * 
 * 1. **Heap (General):**
 * ```c
 * size_t ws = fft_get_workspace_size(plan);
 * fft_data *workspace = ws ? malloc(ws * sizeof(fft_data)) : NULL;
 * fft_exec_dft(plan, input, output, workspace);
 * free(workspace);
 * ```
 * 
 * 2. **Stack (Small Transforms):**
 * ```c
 * size_t ws = fft_get_workspace_size(plan);
 * if (ws > 0 && ws < 1024) {
 *     fft_data workspace[ws];  // C99 VLA
 *     fft_exec_dft(plan, input, output, workspace);
 * }
 * ```
 * 
 * 3. **Aligned (SIMD Optimization):**
 * ```c
 * size_t ws = fft_get_workspace_size(plan);
 * fft_data *workspace = ws ? aligned_alloc(32, ws * sizeof(fft_data)) : NULL;
 * fft_exec_dft(plan, input, output, workspace);
 * aligned_free(workspace);
 * ```
 * 
 * 4. **Thread-Local Pool (High-Performance):**
 * ```c
 * static _Thread_local fft_data *thread_workspace = NULL;
 * static _Thread_local size_t thread_ws_capacity = 0;
 * 
 * size_t ws = fft_get_workspace_size(plan);
 * if (ws > thread_ws_capacity) {
 *     free(thread_workspace);
 *     thread_workspace = malloc(ws * sizeof(fft_data));
 *     thread_ws_capacity = ws;
 * }
 * fft_exec_dft(plan, input, output, thread_workspace);
 * // Don't free - reuse for next transform
 * ```
 * 
 * **Performance Tip:**
 * Allocate workspace once and reuse for multiple transforms with same size.
 * Allocation overhead (malloc/free) can be 10-50% of transform time for small N.
 * 
 * @param plan FFT plan to query
 * @return Number of fft_data elements needed, or 0 if no workspace required
 */
size_t fft_get_workspace_size(fft_object plan);

/**
 * @brief Get execution strategy selected by planner
 * 
 * **Strategy Types:**
 * - FFT_EXEC_INPLACE_BITREV: Power-of-2 in-place (fastest, zero workspace)
 * - FFT_EXEC_STOCKHAM: Mixed-radix Stockham (good cache, N workspace)
 * - FFT_EXEC_BLUESTEIN: Chirp-z algorithm (arbitrary N, 3M workspace)
 * 
 * **Use Cases:**
 * 
 * 1. **Performance Analysis:**
 * ```c
 * fft_exec_strategy_t strat = fft_get_strategy(plan);
 * printf("Strategy: %s\n", 
 *        strat == FFT_EXEC_INPLACE_BITREV ? "In-place" :
 *        strat == FFT_EXEC_STOCKHAM ? "Stockham" : "Bluestein");
 * ```
 * 
 * 2. **Algorithm Statistics:**
 * ```c
 * int count_inplace = 0, count_stockham = 0, count_bluestein = 0;
 * for (int N = 1; N <= 1024; N++) {
 *     fft_object plan = fft_init(N, FFT_FORWARD);
 *     switch (fft_get_strategy(plan)) {
 *         case FFT_EXEC_INPLACE_BITREV: count_inplace++; break;
 *         case FFT_EXEC_STOCKHAM: count_stockham++; break;
 *         case FFT_EXEC_BLUESTEIN: count_bluestein++; break;
 *     }
 *     free_fft(plan);
 * }
 * printf("Distribution: %d in-place, %d Stockham, %d Bluestein\n",
 *        count_inplace, count_stockham, count_bluestein);
 * ```
 * 
 * 3. **Debugging:**
 * ```c
 * if (fft_get_strategy(plan) == FFT_EXEC_BLUESTEIN) {
 *     printf("Warning: Large prime detected, performance may be suboptimal\n");
 * }
 * ```
 * 
 * @param plan FFT plan to query
 * @return Execution strategy enum, or FFT_EXEC_OUT_OF_PLACE if plan is NULL
 */
fft_exec_strategy_t fft_get_strategy(fft_object plan);

//==============================================================================
// GLOBAL CACHE MANAGEMENT (Optional)
//==============================================================================

/**
 * @brief Pre-initialize Rader convolution cache for common primes
 * 
 * **Purpose:**
 * Rader cache is normally lazy-initialized on first use. Calling this
 * function explicitly initializes it with common primes (7, 11, 13).
 * 
 * **When to Use:**
 * - Startup initialization (avoid first-use latency)
 * - Benchmarking (exclude cache initialization from timing)
 * - Testing (ensure deterministic behavior)
 * 
 * **Thread Safety:**
 * - Safe to call from any thread
 * - Multiple calls are safe (idempotent)
 * - Uses internal mutex
 * 
 * **Performance:**
 * - One-time cost: ~100 μs
 * - Pre-populates: 7, 11, 13 (most common prime radices)
 * - Larger primes (17, 19, ...) still lazy-initialized on demand
 * 
 * @note This is optional; fft_init() will initialize on-demand if needed
 */
void init_rader_cache(void);

/**
 * @brief Free global Rader convolution cache (call at program exit)
 * 
 * **Purpose:**
 * Releases all memory held by the global Rader cache. Should be called
 * once at program exit for clean shutdown (prevents leak detection false
 * positives).
 * 
 * **When to Use:**
 * - Program exit / cleanup
 * - After all FFT plans freed
 * - Before library unload (dynamic libraries)
 * 
 * **Thread Safety:**
 * - NOT safe during concurrent FFT operations
 * - Call only when no FFT plans exist or are being used
 * - Acquires internal mutex
 * 
 * **Effect:**
 * - Frees Rader twiddle arrays for all cached primes
 * - Frees permutation arrays
 * - Resets cache to empty state
 * - After cleanup, fft_init() will reinitialize cache if needed
 * 
 * **Typical Usage:**
 * ```c
 * void cleanup_application(void) {
 *     // Free all FFT plans first
 *     for (int i = 0; i < num_plans; i++) {
 *         free_fft(plans[i]);
 *     }
 *     
 *     // Then cleanup global cache
 *     cleanup_rader_cache();
 * }
 * ```
 * 
 * @warning Don't call while FFT plans exist or operations are in progress!
 */
void cleanup_rader_cache(void);

#endif // FFT_PLANNING_H