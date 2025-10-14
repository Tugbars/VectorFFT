//==============================================================================
// throttle_enhanced.h - Enhanced Throttling System
//==============================================================================

#ifndef THROTTLE_ENHANCED_H
#define THROTTLE_ENHANCED_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct cpu_profile_t cpu_profile_t;

/**
 * @brief Throttling modes
 */
typedef enum {
    THROTTLE_MODE_SIMPLE,      // Original windowed refill
    THROTTLE_MODE_TOKEN_BUCKET // Time-based token bucket
} throttle_mode_t;

/**
 * @brief Prefetch priority levels
 */
typedef enum {
    PREFETCH_PRIO_CRITICAL = 0,
    PREFETCH_PRIO_HIGH = 1,
    PREFETCH_PRIO_MEDIUM = 2,
    PREFETCH_PRIO_LOW = 3
} prefetch_priority_t;

//==============================================================================
// INITIALIZATION & CLEANUP
//==============================================================================

/**
 * @brief Initialize enhanced throttling system
 * @param cpu_profile CPU architecture profile
 */
void init_throttling_enhanced(const cpu_profile_t *cpu_profile);

//==============================================================================
// CONFIGURATION
//==============================================================================

/**
 * @brief Set throttling mode
 * @param mode Throttling mode to use
 */
void set_throttle_mode(throttle_mode_t mode);

/**
 * @brief Configure token bucket parameters
 * @param l1_capacity Token capacity for L1 (T0/T1) prefetches
 * @param l2_capacity Token capacity for L2 (T2) prefetches
 * @param nta_capacity Token capacity for NTA (streaming) prefetches
 * @param refill_cycles Cycles between token refills
 * @param tokens_per_refill Tokens added per refill
 */
void configure_token_bucket(
    int l1_capacity,
    int l2_capacity,
    int nta_capacity,
    uint64_t refill_cycles,
    int tokens_per_refill
);

/**
 * @brief Enable/disable statistics collection
 * @param enable True to enable, false to disable
 */
void set_throttle_statistics(bool enable);

//==============================================================================
// PREFETCH OPERATIONS
//==============================================================================

/**
 * @brief Throttled prefetch with priority
 * @param addr Address to prefetch
 * @param hint Prefetch hint (_MM_HINT_T0, T1, T2, NTA)
 * @param priority Priority level
 * @param do_prefetch_fn Actual prefetch function to call if approved
 * @return True if prefetch was issued, false if throttled
 */
bool prefetch_throttled_enhanced(
    const void *addr,
    int hint,
    prefetch_priority_t priority,
    void (*do_prefetch_fn)(const void*, int)
);

//==============================================================================
// STATISTICS & MONITORING
//==============================================================================

/**
 * @brief Get throttling statistics
 * @param total_requested Total prefetch requests
 * @param total_issued Total prefetches actually issued
 * @param total_throttled Total prefetches throttled
 * @param critical_issued Critical priority prefetches issued
 * @param throttle_rate Throttle rate (0.0 to 1.0)
 */
void get_throttle_stats(
    uint64_t *total_requested,
    uint64_t *total_issued,
    uint64_t *total_throttled,
    uint64_t *critical_issued,
    double *throttle_rate
);

/**
 * @brief Print throttling statistics to stdout
 */
void print_throttle_stats(void);

//==============================================================================
// AUTO-TUNING
//==============================================================================

/**
 * @brief Auto-tune token bucket parameters based on observed behavior
 * 
 * Call periodically (e.g., every 1000 FFT calls) to adjust token bucket
 * capacities based on observed throttle rate. Aims to keep throttle rate
 * between 10-30%.
 */
void autotune_token_bucket(void);

#ifdef __cplusplus
}
#endif

#endif // THROTTLE_ENHANCED_H

//==============================================================================
// adaptive_tuning.h - Adaptive Tuning System
//==============================================================================

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
