
/**
 * @brief EWMA (Exponentially Weighted Moving Average) Smoothing
 * 
 * @par Problem: Noisy Measurements
 * @code
 *   Raw measurements fluctuate due to:
 *     - OS scheduling
 *     - Cache state
 *     - Other threads
 *   
 *   Example raw data:
 *     95, 103, 97, 101, 96, 105, 98 cycles/elem
 *   
 *   Question: Is 103 better than 95, or just noise?
 * @endcode
 * 
 * @par Solution: EWMA Filter
 * @code
 *   Formula: filtered = α * new_sample + (1-α) * old_filtered
 *   
 *   With α=0.2:
 *     Sample 1: 95  → filtered=95
 *     Sample 2: 103 → filtered=0.2*103 + 0.8*95 = 96.6
 *     Sample 3: 97  → filtered=0.2*97  + 0.8*96.6 = 96.7
 *     Sample 4: 101 → filtered=0.2*101 + 0.8*96.7 = 97.6
 *   
 *   Effect: Smooth out noise, reveal true trend
 * @endcode
 * 
 * @par Visualization
 * @code
 *   Performance
 *        ^
 *    105 |     *           *
 *    100 |   *   *   *   *
 *     95 | *       *
 *        +-----------------> Time
 *         Raw (noisy)
 *        
 *        ^
 *    100 |       /---
 *     95 |  /---/
 *        +-----------------> Time
 *         EWMA (smooth)
 * @endcode
 */

/**
 * @brief Adaptive Step Size Strategy
 * 
 * @par Accelerate on Success
 * @code
 *   Found improvement?
 *     step_size = step_size * 3/2
 *   
 *   Rationale: If we're heading the right direction,
 *              take bigger steps to converge faster
 *   
 *   Example:
 *     Initial: step=4
 *     Success: step=6
 *     Success: step=9
 *     Success: step=13
 *     Failure: step=6 (reduce)
 * @endcode
 * 
 * @par Decelerate on Failure
 * @code
 *   No improvement?
 *     Reverse direction
 *     step_size = step_size / 2
 *   
 *   Rationale: We overshot the optimum,
 *              take smaller steps to find it
 *   
 *   Example:
 *     distance=16, step=8, direction=+1
 *     Try 24 → Worse
 *     → Reverse: direction=-1, step=4
 *     Try 12 → Worse
 *     → Reverse: direction=+1, step=2
 *     Try 18 → Better!
 * @endcode
 * 
 * @par Visual: Converging to Optimum
 * @code
 *   Performance
 *        ^
 *        |        optimal
 *        |          ↓
 *    100 |         *
 *     95 |       *   *
 *     90 |     *       *
 *     85 |   *           *
 *        +--2--4--6--8-10-12-14-16-18-20-> distance
 *   
 *   Search trace:
 *   Start:    8 (step=4)
 *   Step +4: 12 (better!)
 *   Step +6: 18 (worse)
 *   Step -4: 14 (worse)  
 *   Step +2: 16 (better!) ← Converged
 * @endcode
 */

/**
 * @brief Handling Multiple Local Maxima
 * 
 * @par The Problem
 * @code
 *   Performance
 *        ^
 *        |    B(local)      A(global)
 *        |     /\            /\
 *        |    /  \          /  \
 *        |   /    \   /\   /    \
 *        |  /      \ /  \ /      \
 *        +-------------------------> distance
 *         2  4  6  8 10 12 14 16 18 20
 *   
 *   If we start at 4:
 *     Hill climb → converge at B (distance=6)
 *     Miss A (distance=16) which is better!
 * @endcode
 * 
 * @par Solution 1: Random Restarts (This Code)
 * @code
 *   Every 10,000 iterations:
 *     Check if performance degraded
 *     If yes: Restart search from current position
 *   
 *   Rationale: System conditions change over time,
 *              different optimum may emerge
 * @endcode
 * 
 * @par Solution 2: Multi-Start (Not Implemented)
 * @code
 *   Run hill climbing from multiple starting points:
 *     Start 1: distance=4  → converges to 6
 *     Start 2: distance=12 → converges to 16
 *     Start 3: distance=20 → converges to 16
 *   
 *   Pick best: distance=16
 *   
 *   Pros: Better global coverage
 *   Cons: More expensive
 * @endcode
 */


#ifndef ADAPTIVE_TUNING_H
#define ADAPTIVE_TUNING_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Tuning modes
 */
typedef enum {
    TUNING_MODE_DISABLED,        // No adaptive tuning
    TUNING_MODE_SIMPLE,          // Original simple hill-climb
    TUNING_MODE_EWMA,            // EWMA-filtered hill-climb
    TUNING_MODE_PER_STAGE        // Per-stage independent tuning
} tuning_mode_t;

//==============================================================================
// INITIALIZATION & CLEANUP
//==============================================================================

/**
 * @brief Initialize adaptive tuning system
 * @param num_stages Number of FFT stages
 * @param initial_distances Initial prefetch distances for each stage
 * @param working_set_sizes Working set sizes for each stage (bytes)
 * @param radixes Radix of each stage
 */
void init_adaptive_tuning(
    int num_stages,
    const int *initial_distances,
    const int *working_set_sizes,
    const int *radixes
);

/**
 * @brief Cleanup adaptive tuning system
 */
void cleanup_adaptive_tuning(void);

//==============================================================================
// CONFIGURATION
//==============================================================================

/**
 * @brief Set tuning mode
 * @param mode Tuning mode to use
 */
void set_tuning_mode(tuning_mode_t mode);

/**
 * @brief Configure tuning parameters
 * @param ewma_alpha EWMA smoothing factor (0 < alpha <= 1)
 * @param ewma_warmup_samples Number of samples before trusting EWMA
 * @param improvement_threshold Minimum improvement to consider (e.g., 0.02 = 2%)
 * @param max_search_iterations Max iterations without improvement before converging
 * @param initial_step_size Initial search step size
 */
void configure_tuning(
    double ewma_alpha,
    int ewma_warmup_samples,
    double improvement_threshold,
    int max_search_iterations,
    int initial_step_size
);

/**
 * @brief Enable/disable tuning logging
 * @param enable True to enable, false to disable
 */
void set_tuning_logging(bool enable);

/**
 * @brief Enable/disable periodic re-tuning
 * @param enable True to enable periodic re-tuning
 * @param interval Re-tune every N FFT calls (if enable is true)
 */
void set_periodic_retune(bool enable, uint64_t interval);

//==============================================================================
// PROFILING OPERATIONS
//==============================================================================

/**
 * @brief Start profiling an FFT execution
 * @return Timestamp (TSC) at start
 */
uint64_t profile_fft_start(void);

/**
 * @brief End profiling and update tuning
 * @param start_cycles Timestamp from profile_fft_start()
 * @param n_elements Number of elements processed
 * @param stage_idx Stage index (-1 for global mode)
 */
void profile_fft_end(uint64_t start_cycles, int n_elements, int stage_idx);

/**
 * @brief Get current tuned prefetch distance for a stage
 * @param stage_idx Stage index
 * @return Current optimal distance, or -1 if tuning disabled
 */
int get_tuned_distance(int stage_idx);

//==============================================================================
// STATISTICS & MONITORING
//==============================================================================

/**
 * @brief Get tuning statistics
 * @param total_calls Total FFT calls profiled
 * @param total_changes Total parameter changes made
 * @param avg_improvement Average improvement per change (ratio)
 * @param best_distances_out Array to receive best distances (can be NULL)
 * @param max_stages Size of best_distances_out array
 */
void get_tuning_stats(
    uint64_t *total_calls,
    uint64_t *total_changes,
    double *avg_improvement,
    int *best_distances_out,
    int max_stages
);

/**
 * @brief Print tuning report to stdout
 */
void print_tuning_report(void);

//==============================================================================
// WISDOM EXPORT
//==============================================================================

/**
 * @brief Export tuning results to wisdom database
 * @param n_fft FFT size
 * @param radixes Array of radixes for each stage
 * @param add_wisdom_fn Callback function to add wisdom entries
 * 
 * Callback signature:
 * void add_wisdom(int n_fft, int radix, int distance_input, 
 *                 int distance_twiddle, int hint, int strategy, 
 *                 double cycles_per_element)
 */
void export_tuning_to_wisdom(
    int n_fft,
    const int *radixes,
    void (*add_wisdom_fn)(int, int, int, int, int, int, double)
);

//==============================================================================
// BENCHMARKING (for comparing modes)
//==============================================================================

/**
 * @brief Benchmark result structure
 */
typedef struct {
    tuning_mode_t mode;
    double avg_throughput;
    double std_dev;
    uint64_t total_cycles;
    int num_runs;
} tuning_benchmark_t;

/**
 * @brief Run comparison benchmark across tuning modes
 * @param fft_function Function that performs FFT
 * @param fft_data Opaque data pointer passed to fft_function
 * @param num_runs_per_mode Number of runs per mode
 * @param results Output array for results (size >= num_modes)
 * @param num_modes Number of modes to test (max 3)
 */
void benchmark_tuning_modes(
    void (*fft_function)(void*),
    void *fft_data,
    int num_runs_per_mode,
    tuning_benchmark_t *results,
    int num_modes
);

//==============================================================================
// INTEGRATION HELPERS
//==============================================================================

/**
 * @brief Apply tuned distances to stage prefetch configurations
 * @param stages Array of stage_prefetch_t structures
 * @param num_stages Number of stages
 * 
 * Call this after tuning converges to update your prefetch configs
 * with the optimized distances.
 */
typedef struct stage_prefetch_t stage_prefetch_t;
void apply_tuned_distances_to_stages(stage_prefetch_t *stages, int num_stages);

#ifdef __cplusplus
}
#endif

#endif // ADAPTIVE_TUNING_H
