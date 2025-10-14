//==============================================================================
// ENHANCED ADAPTIVE TUNING - EWMA Filtering & Per-Stage Optimization
//==============================================================================

#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

/**
 * @brief Tuning modes
 */
typedef enum {
    TUNING_MODE_DISABLED,        // No adaptive tuning
    TUNING_MODE_SIMPLE,          // Original simple hill-climb
    TUNING_MODE_EWMA,            // EWMA-filtered hill-climb
    TUNING_MODE_PER_STAGE        // Per-stage independent tuning
} tuning_mode_t;

/**
 * @brief EWMA filter state
 */
typedef struct {
    double value;           // Current filtered value
    double alpha;           // Smoothing factor (0 < alpha <= 1)
    int sample_count;       // Number of samples seen
    bool initialized;       // Whether we have valid data
} ewma_filter_t;

/**
 * @brief Per-stage tuning state
 */
typedef struct {
    int stage_idx;
    
    // Current parameters
    int distance_input;
    int distance_twiddle;
    
    // Performance tracking
    ewma_filter_t throughput;       // Filtered throughput (cycles/element)
    uint64_t total_cycles;
    uint64_t total_elements;
    
    // Optimization state
    int best_distance;
    double best_throughput;
    int tuning_phase;           // 0=init, 1=search, 2=converged
    int iterations_without_improvement;
    int search_direction;       // +1 or -1
    int search_step_size;
    
    // Convergence detection
    double improvement_threshold;   // Minimum improvement to consider (e.g., 2%)
    int max_iterations_no_improve; // Give up after N iterations
    
    // Working set info (for heuristics)
    int working_set_size;
    int radix;
} stage_tuning_state_t;

/**
 * @brief Global adaptive tuning configuration
 */
typedef struct {
    tuning_mode_t mode;
    bool enable_logging;
    
    // EWMA parameters
    double ewma_alpha;              // Smoothing factor (default 0.2)
    int ewma_warmup_samples;        // Samples before trusting EWMA
    
    // Search parameters
    double improvement_threshold;   // Minimum improvement (default 2%)
    int max_search_iterations;      // Max iterations without improvement
    int initial_step_size;          // Initial search step (default 4)
    int min_distance;               // Minimum prefetch distance (default 2)
    int max_distance;               // Maximum prefetch distance (default 64)
    
    // Per-stage tuning
    bool tune_stages_independently;
    int stages_to_tune;             // How many stages to tune (0=all)
    
    // Periodic re-tuning
    bool enable_periodic_retune;
    uint64_t retune_interval;       // Re-tune every N calls (default 10000)
    
} tuning_config_t;

/**
 * @brief Global tuning state
 */
typedef struct {
    tuning_config_t config;
    
    // Global (simple mode) state
    struct {
        double throughput_ewma;
        int current_distance;
        int best_distance;
        double best_throughput;
        int tuning_phase;
        int iterations_without_improvement;
        uint64_t total_calls;
    } global;
    
    // Per-stage state
    stage_tuning_state_t *stages;
    int num_stages;
    
    // Statistics
    struct {
        uint64_t total_fft_calls;
        uint64_t total_tuning_changes;
        double avg_improvement;
    } stats;
    
} adaptive_tuning_state_t;

static adaptive_tuning_state_t g_tuning = {
    .config = {
        .mode = TUNING_MODE_DISABLED,
        .enable_logging = false,
        .ewma_alpha = 0.2,
        .ewma_warmup_samples = 10,
        .improvement_threshold = 0.02,      // 2%
        .max_search_iterations = 20,
        .initial_step_size = 4,
        .min_distance = 2,
        .max_distance = 64,
        .tune_stages_independently = false,
        .stages_to_tune = 0,
        .enable_periodic_retune = true,
        .retune_interval = 10000
    },
    .stages = NULL,
    .num_stages = 0
};

//==============================================================================
// EWMA FILTER IMPLEMENTATION
//==============================================================================

/**
 * @brief Initialize EWMA filter
 */
static inline void ewma_init(ewma_filter_t *filter, double alpha) {
    filter->value = 0.0;
    filter->alpha = alpha;
    filter->sample_count = 0;
    filter->initialized = false;
}

/**
 * @brief Update EWMA with new sample
 */
static inline void ewma_update(ewma_filter_t *filter, double sample) {
    if (!filter->initialized) {
        // First sample: initialize with raw value
        filter->value = sample;
        filter->initialized = true;
    } else {
        // EWMA update: value = alpha * sample + (1 - alpha) * old_value
        filter->value = filter->alpha * sample + (1.0 - filter->alpha) * filter->value;
    }
    filter->sample_count++;
}

/**
 * @brief Get current EWMA value
 */
static inline double ewma_get(const ewma_filter_t *filter) {
    return filter->value;
}

/**
 * @brief Check if EWMA has enough samples to be trusted
 */
static inline bool ewma_is_warmed_up(const ewma_filter_t *filter, int warmup_samples) {
    return filter->initialized && (filter->sample_count >= warmup_samples);
}

//==============================================================================
// PER-STAGE TUNING IMPLEMENTATION
//==============================================================================

/**
 * @brief Initialize per-stage tuning state
 */
static void init_stage_tuning(
    stage_tuning_state_t *stage,
    int stage_idx,
    int initial_distance,
    int working_set_size,
    int radix
) {
    stage->stage_idx = stage_idx;
    stage->distance_input = initial_distance;
    stage->distance_twiddle = initial_distance / 2;
    
    ewma_init(&stage->throughput, g_tuning.config.ewma_alpha);
    
    stage->total_cycles = 0;
    stage->total_elements = 0;
    stage->best_distance = initial_distance;
    stage->best_throughput = INFINITY;
    stage->tuning_phase = 0; // init
    stage->iterations_without_improvement = 0;
    stage->search_direction = 1;
    stage->search_step_size = g_tuning.config.initial_step_size;
    stage->improvement_threshold = g_tuning.config.improvement_threshold;
    stage->max_iterations_no_improve = g_tuning.config.max_search_iterations;
    stage->working_set_size = working_set_size;
    stage->radix = radix;
}

/**
 * @brief Update stage performance and potentially adjust distance
 */
static void update_stage_tuning(
    stage_tuning_state_t *stage,
    uint64_t cycles,
    int elements
) {
    // Update statistics
    stage->total_cycles += cycles;
    stage->total_elements += elements;
    
    // Calculate throughput (cycles per element)
    double throughput = (double)cycles / (double)elements;
    
    // Update EWMA
    ewma_update(&stage->throughput, throughput);
    
    // Wait for warmup before tuning
    if (!ewma_is_warmed_up(&stage->throughput, g_tuning.config.ewma_warmup_samples)) {
        return;
    }
    
    double filtered_throughput = ewma_get(&stage->throughput);
    
    // State machine for hill-climbing
    switch (stage->tuning_phase) {
        case 0: // Initial measurement
            stage->best_throughput = filtered_throughput;
            stage->best_distance = stage->distance_input;
            stage->tuning_phase = 1;
            stage->iterations_without_improvement = 0;
            
            if (g_tuning.config.enable_logging) {
                printf("Stage %d: Initial throughput = %.2f cycles/elem (distance=%d)\n",
                    stage->stage_idx, filtered_throughput, stage->distance_input);
            }
            break;
            
        case 1: { // Active search
            // Check if this is an improvement
            double improvement = (stage->best_throughput - filtered_throughput) / stage->best_throughput;
            
            if (improvement > stage->improvement_threshold) {
                // Found improvement!
                stage->best_throughput = filtered_throughput;
                stage->best_distance = stage->distance_input;
                stage->iterations_without_improvement = 0;
                
                g_tuning.stats.total_tuning_changes++;
                g_tuning.stats.avg_improvement = 
                    (g_tuning.stats.avg_improvement * (g_tuning.stats.total_tuning_changes - 1) + improvement) / 
                    g_tuning.stats.total_tuning_changes;
                
                if (g_tuning.config.enable_logging) {
                    printf("Stage %d: Improvement! %.2f%% better, new distance=%d\n",
                        stage->stage_idx, improvement * 100.0, stage->best_distance);
                }
                
                // Continue in same direction, increase step size (accelerate)
                stage->search_step_size = (stage->search_step_size * 3) / 2;
                if (stage->search_step_size > 16) stage->search_step_size = 16;
                
            } else {
                // No improvement
                stage->iterations_without_improvement++;
                
                if (stage->iterations_without_improvement >= stage->max_iterations_no_improve) {
                    // Converged - stop tuning
                    stage->tuning_phase = 2;
                    stage->distance_input = stage->best_distance;
                    stage->distance_twiddle = stage->best_distance / 2;
                    
                    if (g_tuning.config.enable_logging) {
                        printf("Stage %d: Converged at distance=%d (%.2f cycles/elem)\n",
                            stage->stage_idx, stage->best_distance, stage->best_throughput);
                    }
                    break;
                }
                
                // Try different direction or smaller step
                if (stage->iterations_without_improvement % 3 == 0) {
                    // Reverse direction
                    stage->search_direction *= -1;
                    stage->search_step_size /= 2;
                    if (stage->search_step_size < 1) stage->search_step_size = 1;
                }
            }
            
            // Compute next distance to try
