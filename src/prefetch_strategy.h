/**
 * @brief Hill Climbing Algorithm - Visual Explanation
 * 
 * Imagine searching for the highest peak in a mountain range while blindfolded:
 * 
 * @code
 *   Performance (higher is better)
 *        ^
 *        |
 *        |     Global Maximum (best solution)
 *        |          /\
 *        |         /  \
 *        |   /\   /    \
 *        |  /  \ /      \    Local Maximum (trap!)
 *        | /    X        \  /\
 *        |/              \/  \
 *        +-------------------------> Parameter Space
 *               (e.g., prefetch distance: 2, 4, 6, 8, 10, 12...)
 * 
 * Strategy: Take small steps, always move uphill
 * Problem: Can get stuck at local maxima (X marks the trap!)
 * @endcode
 */

/**
 * @brief Simple Hill Climbing Pseudocode
 * 
 * @algorithm
 * @code
 * 1. Start at initial position (e.g., prefetch_distance = 8)
 * 2. Measure current performance
 * 3. Try neighbor solutions:
 *    - Option A: distance = 8 + step (e.g., 12)
 *    - Option B: distance = 8 - step (e.g., 4)
 * 4. If neighbor is better:
 *       Move to neighbor
 *       Goto step 2
 *    Else:
 *       Stop (converged)
 * @endcode
 * 
 * @par Example Trace
 * @code
 * Iteration 1: distance=8,  performance=100 cycles/elem
 * Iteration 2: Try distance=12, performance=95  ← Better! Move here
 * Iteration 3: Try distance=16, performance=93  ← Better! Move here
 * Iteration 4: Try distance=20, performance=97  ← Worse! Stop
 * Result: Best distance = 16
 * @endcode
 */

/**
 * @brief Hill Climbing State Machine
 * 
 * @dot
 * digraph hill_climb {
 *   rankdir=LR;
 *   node [shape=box, style=rounded];
 *   
 *   init [label="Phase 0\nInitialization"];
 *   search [label="Phase 1\nActive Search"];
 *   converged [label="Phase 2\nConverged"];
 *   
 *   init -> search [label="First\nmeasurement"];
 *   search -> search [label="Found improvement:\nAccelerate step size"];
 *   search -> search [label="No improvement:\nReverse direction\nReduce step size"];
 *   search -> converged [label="Max iterations\nwithout improvement"];
 *   converged -> search [label="Performance degrades\n>10%: Re-tune"];
 * }
 * @enddot
 */

#ifndef PREFETCH_STRATEGY_H
#define PREFETCH_STRATEGY_H

#include <stdbool.h>
#include <stdlib.h>
#include <immintrin.h>

// Forward declarations to avoid circular dependency with highspeedFFT.h
// These types are defined in highspeedFFT.h
typedef struct fft_set *fft_object;
typedef struct fft_t fft_data;

//==============================================================================
// PREFETCH CONFIGURATION TYPES
//==============================================================================

/**
 * @brief Prefetch scheduling strategy
 */
typedef enum {
    PREFETCH_NONE = 0,    // No prefetch (tiny transforms)
    PREFETCH_SINGLE,      // Single stream (sequential access)
    PREFETCH_DUAL,        // Two streams (read/write)
    PREFETCH_MULTI,       // Multiple streams (mixed-radix stages)
    PREFETCH_STRIDED      // Strided access (transpose-like patterns)
} prefetch_strategy_t;

/**
 * @brief Per-stage prefetch configuration
 */
typedef struct {
    int distance_input;        // Prefetch distance for input reads (elements)
    int distance_output;       // Prefetch distance for output writes
    int distance_twiddle;      // Prefetch distance for twiddle factors
    int hint_input;            // Hint for input data (_MM_HINT_T0, etc.)
    int hint_output;           // Hint for output data
    int hint_twiddle;          // Hint for twiddles
    prefetch_strategy_t strategy; // Overall strategy
    int block_size;            // Blocking factor (for cache reuse)
    bool enable;               // Master enable/disable
} stage_prefetch_t;

/**
 * @brief Dynamic prefetch configuration (per-transform tuning)
 */
typedef struct {
    stage_prefetch_t *stages;     // Per-stage configs (malloc'd array)
    int num_stages;               // Number of stages in transform
    int l1_size;                  // L1 data cache size (bytes)
    int l2_size;                  // L2 cache size (bytes)
    int l3_size;                  // L3 cache size (bytes)
    int cache_line_size;          // Cache line size (typically 64B)
    bool enable_runtime_tuning;   // Enable adaptive tuning during execution
} prefetch_config_t;

/**
 * @brief CPU architecture profile for optimal prefetch tuning
 */
typedef struct {
    const char *name;
    int prefetch_buffers;         // Number of HW prefetch buffers
    int prefetch_latency;         // Cycles to issue prefetch
    int l1_latency;               // L1 cache latency in cycles
    int l2_latency;               // L2 cache latency in cycles
    int l3_latency;               // L3 cache latency in cycles
    bool has_write_prefetch;      // Supports prefetchw instruction
    bool has_strong_hwpf;         // Strong hardware prefetcher
    int optimal_distance[8];      // Optimal distances for different working sets
} cpu_profile_t;

/**
 * @brief Wisdom database entry (for exhaustive search results)
 */
typedef struct {
    int n_fft;                    // Transform size
    int radix;                    // Radix used
    int distance_input;           // Optimal input distance
    int distance_twiddle;         // Optimal twiddle distance
    int hint;                     // Optimal hint
    prefetch_strategy_t strategy; // Optimal strategy
    double cycles_per_element;    // Measured performance
    long timestamp;               // When measured (time_t)
} wisdom_entry_t;

//==============================================================================
// PUBLIC API - INITIALIZATION AND CLEANUP
//==============================================================================

//==============================================================================
// PUBLIC API - INITIALIZATION AND CLEANUP
//==============================================================================

/**
 * @brief Initialize the entire prefetch system
 * 
 * Performs CPU detection, cache detection, wisdom loading, and per-stage
 * configuration. Call once at FFT object creation.
 * 
 * @param fft_obj FFT object containing factorization info
 */
void init_prefetch_system(fft_object fft_obj);

/**
 * @brief Initialize per-stage prefetch configuration (internal)
 * 
 * Called by init_prefetch_system. Can also be called separately for
 * re-initialization without full system init.
 * 
 * @param fft_obj FFT object containing factorization info
 */
void init_stage_prefetch(fft_object fft_obj);

/**
 * @brief Detect CPU cache sizes (internal)
 * 
 * Uses CPUID to detect L1/L2/L3 cache sizes. Called by init_prefetch_system.
 */
void detect_cache_sizes(void);

/**
 * @brief Cleanup prefetch system resources
 * 
 * Frees allocated stage configurations. Call at FFT object destruction.
 */
void cleanup_prefetch_system(void);

/**
 * @brief Get prefetch config for a specific stage
 * 
 * @param factor_index Stage index in recursion (0 = first stage)
 * @return Pointer to stage config, or NULL if invalid index
 */
stage_prefetch_t* get_stage_config(int factor_index);

//==============================================================================
// PUBLIC API - BASIC PREFETCH OPERATIONS
//==============================================================================

/**
 * @brief Stage-aware prefetch for input data
 * 
 * @param input Input data array
 * @param idx Current element index
 * @param cfg Stage configuration (from get_stage_config)
 */
void prefetch_input(const fft_data *input, int idx, stage_prefetch_t *cfg);

/**
 * @brief Stage-aware prefetch for twiddle factors
 * 
 * @param twiddle Twiddle factor array
 * @param idx Current element index
 * @param cfg Stage configuration
 */
void prefetch_twiddle(const fft_data *twiddle, int idx, stage_prefetch_t *cfg);

/**
 * @brief Multi-stream prefetch for recursive stages
 * 
 * Prefetches multiple lanes of input data and twiddle factors
 * for mixed-radix recursive algorithms.
 * 
 * @param input_base Base pointer to input data
 * @param twiddle_base Base pointer to twiddles (can be NULL)
 * @param idx Current butterfly index
 * @param stride Stride between lanes
 * @param radix Current radix
 * @param cfg Stage configuration
 */
void prefetch_stage_recursive(
    const fft_data *input_base,
    const fft_data *twiddle_base,
    int idx,
    int stride,
    int radix,
    stage_prefetch_t *cfg
);

/**
 * @brief Prefetch for tight butterfly loops
 * 
 * Simplified prefetch for inner loop bodies where only input
 * and twiddles need to be prefetched.
 * 
 * @param input Input data array
 * @param twiddle Twiddle array (can be NULL)
 * @param idx Current element index
 * @param cfg Stage configuration
 */
void prefetch_butterfly_loop(
    const fft_data *input,
    const fft_data *twiddle,
    int idx,
    stage_prefetch_t *cfg
);

//==============================================================================
// ADVANCED API - WISDOM DATABASE
//==============================================================================

/**
 * @brief Load wisdom database from file
 * 
 * @param filename Path to wisdom file
 */
void load_wisdom(const char *filename);

/**
 * @brief Save wisdom database to file
 * 
 * @param filename Path to wisdom file
 */
void save_wisdom(const char *filename);

/**
 * @brief Find wisdom entry for given transform size and radix
 * 
 * @param n_fft Transform size
 * @param radix Radix being used
 * @return Pointer to wisdom entry, or NULL if not found
 */
wisdom_entry_t* find_wisdom(int n_fft, int radix);

/**
 * @brief Add new wisdom entry (called by exhaustive search)
 * 
 * @param n_fft Transform size
 * @param radix Radix used
 * @param distance_input Optimal input distance
 * @param distance_twiddle Optimal twiddle distance
 * @param hint Optimal hint
 * @param strategy Optimal strategy
 * @param cycles_per_element Measured performance
 */
void add_wisdom(
    int n_fft,
    int radix,
    int distance_input,
    int distance_twiddle,
    int hint,
    prefetch_strategy_t strategy,
    double cycles_per_element
);

//==============================================================================
// ADVANCED API - EXHAUSTIVE SEARCH
//==============================================================================

//==============================================================================
// ADVANCED API - EXHAUSTIVE SEARCH
//==============================================================================

/**
 * @brief Perform exhaustive search for optimal prefetch configuration
 * 
 * Tests multiple combinations of distance, hint, and strategy to find
 * the fastest configuration. Results are stored in wisdom database.
 * 
 * @param fft_obj FFT object to benchmark
 * @param factor_index Stage index to optimize
 * @param best_config Output: optimal configuration found
 */
void search_optimal_prefetch(
    fft_object fft_obj,
    int factor_index,
    stage_prefetch_t *best_config
);

//==============================================================================
// ADVANCED API - RUNTIME PROFILING
//==============================================================================

/**
 * @brief Start profiling an FFT execution
 * 
 * Call before fft_exec() to begin cycle counting for adaptive tuning.
 */
void profile_start(void);

/**
 * @brief End profiling and update adaptive parameters
 * 
 * Call after fft_exec() to complete timing and adjust prefetch distances.
 * 
 * @param n_elements Number of elements in the FFT
 */
void profile_end(int n_elements);

//==============================================================================
// ADVANCED API - SPECIALIZED PREFETCH OPERATIONS
//==============================================================================

/**
 * @brief Adjust prefetch distance based on loop unrolling factor
 * 
 * @param base_distance Base prefetch distance
 * @param unroll_factor Loop unrolling factor (2, 4, 8, 16, etc.)
 * @return Adjusted distance accounting for larger loop bodies
 */
int adjust_distance_for_unroll(int base_distance, int unroll_factor);

/**
 * @brief Strided prefetch for transpose-like access patterns
 * 
 * Prefetches multiple streams with large stride, handling TLB and cache line issues.
 * 
 * @param base Base pointer to data
 * @param idx Current index
 * @param stride Stride between elements
 * @param num_streams Number of streams to prefetch
 * @param cfg Stage configuration
 */
void prefetch_strided(
    const fft_data *base,
    int idx,
    int stride,
    int num_streams,
    stage_prefetch_t *cfg
);

/**
 * @brief Blocking-aware prefetch for large radices
 * 
 * Prefetches the start of the next block while processing current block.
 * 
 * @param base Base pointer to data
 * @param block_start Current block start index
 * @param block_size Size of each block
 * @param cfg Stage configuration
 */
void prefetch_blocked(
    const fft_data *base,
    int block_start,
    int block_size,
    stage_prefetch_t *cfg
);

/**
 * @brief Group prefetch for radix-based access patterns
 * 
 * Prefetches all lanes of the next radix group for better pipelining.
 * 
 * @param base Base pointer to data
 * @param group_idx Current group index
 * @param group_size Size of each group
 * @param radix Radix of butterfly
 * @param cfg Stage configuration
 */
void prefetch_radix_group(
    const fft_data *base,
    int group_idx,
    int group_size,
    int radix,
    stage_prefetch_t *cfg
);

//==============================================================================
// UTILITY FUNCTIONS
//==============================================================================

/**
 * @brief Get global prefetch configuration (read-only)
 * 
 * @return Pointer to global config
 */
const prefetch_config_t* get_prefetch_config(void);

/**
 * @brief Get detected CPU profile (read-only)
 * 
 * @return Pointer to current CPU profile
 */
const cpu_profile_t* get_cpu_profile(void);

/**
 * @brief Enable/disable prefetch system at runtime
 * 
 * @param enable True to enable, false to disable
 */
void set_prefetch_enable(bool enable);

//==============================================================================
// LOW-LEVEL MACROS (for inline use in hot paths)
//==============================================================================

/**
 * @brief Direct prefetch with hint (low-level)
 * Use this in performance-critical code where function call overhead matters
 */
#define PREFETCH_HINT(addr, hint) _mm_prefetch((const char *)(addr), (hint))

/**
 * @brief Check if prefetch is enabled for a stage
 */
#define PREFETCH_ENABLED(cfg) ((cfg) && (cfg)->enable)

//==============================================================================
// DEBUG SUPPORT
//==============================================================================

#ifdef FFT_DEBUG_PREFETCH

/**
 * @brief Print complete prefetch configuration for debugging
 */
void print_prefetch_config(void);

/**
 * @brief Print per-stage prefetch settings
 * 
 * @param stage_idx Stage index to print
 */
void print_stage_config(int stage_idx);

#endif // FFT_DEBUG_PREFETCH

#endif // PREFETCH_STRATEGY_H


/**
 * @brief Hill Climbing for Prefetch Distance Optimization
 * 
 * @par What is "Prefetch Distance"?
 * @code
 * Prefetch distance = How many iterations ahead to prefetch
 * 
 * Example with distance=4:
 *   Iteration 0: Process data[0],  prefetch data[4]
 *   Iteration 1: Process data[1],  prefetch data[5]
 *   Iteration 2: Process data[2],  prefetch data[6]
 *   Iteration 3: Process data[3],  prefetch data[7]
 *   Iteration 4: Process data[4],  prefetch data[8]  ← data[4] now in cache!
 *                     ↑
 *                prefetched 4 iterations ago
 * 
 * Too small (distance=1):  Data arrives too late (cache miss)
 * Too large (distance=64): Wastes prefetch buffers, evicts useful data
 * Just right (distance=?): Data arrives exactly when needed
 * @endcode
 * 
 * @par The Search Space
 * @code
 *   Performance (lower is better: cycles/element)
 *        ^
 *    150 |                            Too far ahead:
 *        |                            • Wastes buffers
 *    125 |\                           • Pollutes cache
 *        | \                    /
 *    100 |  \                  /
 *        |   \                /       Too close:
 *     75 |    \      SWEET   /        • Prefetch arrives late
 *        |     \     SPOT   /         • Cache misses
 *     50 |      \    ↓     /
 *        |       \  /\    /
 *     25 |        \/  \  /
 *        |         ↑   \/             Local maxima (traps!)
 *      0 +----+----+----+----+----+----+----+----+----+---> distance
 *         2   4   8   12  16  20  24  28  32  36  40
 *                      ↑
 *                   optimal = 16
 * 
 * Goal: Find the valley (minimum cycles/element)
 * @endcode
 */

/**
 * @brief Complete Hill Climbing Execution Trace
 * 
 * @par Scenario: Finding Optimal Prefetch Distance
 * @code
 * ═══════════════════════════════════════════════════════════════
 * PHASE 0: INITIALIZATION
 * ═══════════════════════════════════════════════════════════════
 * 
 * Initial configuration (from heuristics):
 *   distance = 8
 *   step_size = 4
 *   direction = +1 (search upward first)
 *   
 * Measure baseline:
 *   Run FFT with distance=8
 *   Result: 95 cycles/element
 *   
 * State transition: Phase 0 → Phase 1 (Active Search)
 * 
 * 
 * ═══════════════════════════════════════════════════════════════
 * PHASE 1: ACTIVE SEARCH (Hill Climbing)
 * ═══════════════════════════════════════════════════════════════
 * 
 * ITERATION 1:
 * ───────────
 *   Current best: distance=8, throughput=95 cycles/elem
 *   
 *   Try: distance = 8 + (+1 * 4) = 12
 *   Measure: 87 cycles/elem
 *   
 *   Improvement: (95 - 87) / 95 = 8.4% ✓
 *   
 *   Action: ✓ Accept move to 12
 *           ✓ Accelerate step_size: 4 → 6
 *           ✓ Keep direction: +1
 *   
 *   New state:
 *     best_distance = 12
 *     best_throughput = 87
 *     iterations_without_improvement = 0
 * 
 * 
 * ITERATION 2:
 * ───────────
 *   Current best: distance=12, throughput=87 cycles/elem
 *   
 *   Try: distance = 12 + (+1 * 6) = 18
 *   Measure: 92 cycles/elem
 *   
 *   Improvement: (87 - 92) / 87 = -5.7% ✗ (worse!)
 *   
 *   Action: ✗ Reject move (stay at 12)
 *           ✗ Reverse direction: +1 → -1
 *           ✗ Reduce step: 6 → 3
 *   
 *   New state:
 *     best_distance = 12 (unchanged)
 *     best_throughput = 87 (unchanged)
 *     iterations_without_improvement = 1
 *     direction = -1
 * 
 * 
 * ITERATION 3:
 * ───────────
 *   Current best: distance=12, throughput=87 cycles/elem
 *   
 *   Try: distance = 12 + (-1 * 3) = 9
 *   Measure: 90 cycles/elem
 *   
 *   Improvement: (87 - 90) / 87 = -3.4% ✗ (worse!)
 *   
 *   Action: ✗ Reject move
 *           ✗ Reverse direction: -1 → +1
 *           ✗ Reduce step: 3 → 1
 *   
 *   New state:
 *     best_distance = 12 (unchanged)
 *     best_throughput = 87 (unchanged)  
 *     iterations_without_improvement = 2
 *     direction = +1
 * 
 * 
 * ITERATION 4:
 * ───────────
 *   Current best: distance=12, throughput=87 cycles/elem
 *   
 *   Try: distance = 12 + (+1 * 1) = 13
 *   Measure: 86 cycles/elem
 *   
 *   Improvement: (87 - 86) / 87 = 1.1% ✗ (below threshold of 2%)
 *   
 *   Action: ✗ Reject move (improvement too small)
 *           ✗ Reverse direction: +1 → -1
 *           ✗ Reduce step: 1 → 1 (minimum)
 *   
 *   New state:
 *     best_distance = 12 (unchanged)
 *     best_throughput = 87 (unchanged)
 *     iterations_without_improvement = 3
 *     direction = -1
 * 
 * 
 * ITERATION 5:
 * ───────────
 *   Current best: distance=12, throughput=87 cycles/elem
 *   
 *   Try: distance = 12 + (-1 * 1) = 11
 *   Measure: 88 cycles/elem
 *   
 *   Improvement: (87 - 88) / 87 = -1.1% ✗ (worse!)
 *   
 *   Action: ✗ Reject move
 *   
 *   New state:
 *     iterations_without_improvement = 4
 * 
 * 
 * [... continues trying small steps around distance=12 ...]
 * 
 * 
 * ITERATION 20:
 * ────────────
 *   iterations_without_improvement = 20
 *   Reached max_search_iterations limit
 *   
 *   Action: Declare convergence!
 *   State transition: Phase 1 → Phase 2 (Converged)
 * 
 * 
 * ═══════════════════════════════════════════════════════════════
 * PHASE 2: CONVERGED (Monitoring)
 * ═══════════════════════════════════════════════════════════════
 * 
 * Locked in: distance = 12, throughput = 87 cycles/elem
 * 
 * Every 10,000 FFT calls:
 *   Measure current performance
 *   
 *   Example at call 10,000:
 *     Current throughput: 88 cycles/elem
 *     Degradation: (88 - 87) / 87 = 1.1% (< 10% threshold)
 *     Action: Continue with distance=12
 *   
 *   Example at call 20,000:
 *     Current throughput: 97 cycles/elem
 *     Degradation: (97 - 87) / 87 = 11.5% (> 10% threshold)
 *     Action: ⚠ Performance degraded! Restart search
 *     State transition: Phase 2 → Phase 1
 * 
 * @endcode
 */

/**
 * @brief Visual: Convergence to Local Maximum
 * 
 * @code
 *   Performance (lower = better)
 *        ^
 *    100 |                                Legend:
 *        |                                ────
 *     95 |  ①                             ① Start (distance=8, 95 cycles)
 *        |     ╲                           ② Try distance=12 (87 cycles) ✓
 *     90 |        ╲         ③              ③ Try distance=18 (92 cycles) ✗
 *        |           ╲   ╱                 ④ Try distance=9  (90 cycles) ✗
 *     87 |              ②  ←  LOCAL        ⑤ Try distance=13 (86 cycles) ✗ (too small)
 *        |           ╱   ╲  MAXIMUM        
 *     85 |        ④       ╲                Result: Converged at ②
 *        |                  ╲              
 *     80 |                     ╲           Note: Might miss global optimum
 *        |                        ╲        at distance=16 (if it exists),
 *     75 |                           ╲     but 87 is still very good!
 *        +----+----+----+----+----+----+----+----+---> distance
 *         4   6   8   10  12  14  16  18  20  22
 *              ↑       ↑               ↑
 *            start   found           rejected
 * 
 * Search path: 8 → 12 → 18(reject) → 9(reject) → 13(reject) → STOP
 * @endcode
 */

/**
 * @brief Why It Stops at Local Maxima
 * 
 * @par The Hill Climbing Limitation
 * @code
 * Imagine the actual performance landscape:
 * 
 *   Cycles/elem
 *        ^
 *    100 |     
 *        |  A          C (better, but unreachable!)
 *     90 |  |\        /|
 *        |  | \      / |
 *     80 |  |  \  B /  |
 *        |  |   \/\/   |
 *     70 |  |    /\    |
 *        |  |   /  \   |
 *     60 +--+--+----+--+-----> distance
 *         4  8  12  16  20
 *         ↑     ↑      ↑
 *       Start  Stuck  Miss!
 * 
 * If we start at A (distance=8):
 *   • Can only see neighbors: 7, 9
 *   • Both are downhill (worse)
 *   • Algorithm says "stop here" (local max at B)
 *   • Never discovers C (distance=20) which is better!
 * 
 * Why we can't reach C from B:
 *   • Would need to go "uphill" through the valley
 *   • Hill climbing never accepts worse solutions
 *   • Stuck at local maximum B
 * 
 * Real-world: This is usually OK because:
 *   1. Performance landscapes are often smooth
 *   2. Local maxima are still good (87 vs 75 cycles)
 *   3. Periodic re-tuning gives second chances
 *   4. Multiple FFT sizes explore different regions
 * @endcode
 */

/**
 * @brief Step Size Adaptation Visualization
 * 
 * @code
 * How step size changes during search:
 * 
 * Distance
 *    20 |                    
 *       |          ▲ (step=6)
 *    18 |         /
 *       |        /
 *    16 |       /
 *       |      /
 *    14 |     /
 *       | ▲  / (step=4, accelerated to 6)
 *    12 | |\/
 *       | |/\  ▼ (step=3, reversed)
 *    10 | |  \/
 *       | |   /\ ▼ (step=1, further reduced)
 *     8 | |  /  \|
 *       +-|--+---+---+---+----> iteration
 *         1  2   3   4   5
 *         
 * Iteration 1: distance=8→12  (step=4, success, accelerate)
 * Iteration 2: distance=12→18 (step=6, failure, reverse+reduce)
 * Iteration 3: distance=12→9  (step=3, failure, reverse+reduce)
 * Iteration 4: distance=12→13 (step=1, failure, reverse)
 * Iteration 5: distance=12→11 (step=1, failure, STOP)
 * 
 * Result: Converged at distance=12 with progressively
 *         smaller steps to "home in" on the optimum
 * @endcode
 */

/**
 * @brief Contrast: What if We Used Exhaustive Search?
 * 
 * @code
 * Exhaustive search (like FFTW planner):
 * ─────────────────────────────────────
 * distances = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
 * 
 * FOR EACH distance:
 *   Run 100 FFTs
 *   Measure average throughput
 * 
 * Results:
 *   distance=2:  120 cycles/elem
 *   distance=4:  105 cycles/elem
 *   distance=6:   95 cycles/elem
 *   distance=8:   90 cycles/elem
 *   distance=10:  88 cycles/elem
 *   distance=12:  85 cycles/elem ← BEST
 *   distance=14:  87 cycles/elem
 *   distance=16:  90 cycles/elem
 *   distance=18:  95 cycles/elem
 *   distance=20: 100 cycles/elem
 *   distance=22: 105 cycles/elem
 *   distance=24: 110 cycles/elem
 * 
 * Total: 12 distances × 100 FFTs = 1,200 FFT executions
 * Result: GUARANTEED global optimum (distance=12)
 * 
 * 
 * Hill climbing (this code):
 * ──────────────────────────
 * Start: distance=8
 * 
 * Iteration 1: Try 12 → Better!    (100 FFTs)
 * Iteration 2: Try 18 → Worse      (100 FFTs)
 * Iteration 3: Try 9  → Worse      (100 FFTs)
 * Iteration 4: Try 13 → Same       (100 FFTs)
 * Iteration 5: Try 11 → Worse      (100 FFTs)
 * 
 * Total: ~5 iterations × 100 FFTs = 500 FFT executions
 * Result: Found local optimum (distance=12)
 * 
 * Savings: 58% fewer measurements!
 * Trade-off: Might miss global optimum (but didn't in this case)
 * @endcode
 */
