//==============================================================================
// PREFETCH_STRATEGY.H - FFTW-Inspired Prefetch System
// Clean, dependency-free header with proper declarations
//==============================================================================

#ifndef PREFETCH_STRATEGY_H
#define PREFETCH_STRATEGY_H

#include <stdbool.h>
#include <stdlib.h>
#include <immintrin.h>

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