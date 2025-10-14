//==============================================================================
// ENHANCED ADAPTIVE TUNING - EWMA Filtering & Per-Stage Optimization
// STANDALONE MODULE - Optional integration with prefetch_strategy.c
//
// Compilation modes:
//   1. Standalone: compile without HFFT_USE_ADAPTIVE_TUNING
//   2. Integrated: compile with -DHFFT_USE_ADAPTIVE_TUNING
//==============================================================================

#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>

// Epsilon for denominator guards
#define EPS 1e-12

//==============================================================================
// PORTABLE TSC IMPLEMENTATION
//==============================================================================

/**
 * @brief Read CPU timestamp counter (portable)
 * Uses rdtsc on x86, clock_gettime elsewhere
 */
static inline uint64_t read_tsc_inline(void) {
#if defined(__x86_64__) || defined(_M_X64)
    uint32_t lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
#else
    // Fallback: use monotonic clock (nanoseconds as "cycles")
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

//==============================================================================
// STANDALONE TYPE DEFINITIONS
// Only used if NOT linking with prefetch_strategy.c
//==============================================================================

#ifndef HFFT_USE_ADAPTIVE_TUNING

/**
 * @brief Standalone stage prefetch config (minimal subset)
 */
typedef struct {
    bool enable;
    int distance_input;
    int distance_output;
    int distance_twiddle;
} stage_prefetch_t;

#endif // !HFFT_USE_ADAPTIVE_TUNING

//==============================================================================
// CORE TYPES (always defined)
//==============================================================================

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
    int tuned_stage_count;  // How many stages actually being tuned
    
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
    .num_stages = 0,
    .tuned_stage_count = 0
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
        filter->value = sample;
        filter->initialized = true;
    } else {
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
    
    // Clamp initial distance to valid range
    int d0 = initial_distance;
    if (d0 < g_tuning.config.min_distance) d0 = g_tuning.config.min_distance;
    if (d0 > g_tuning.config.max_distance) d0 = g_tuning.config.max_distance;
    
    stage->distance_input = d0;
    stage->distance_twiddle = d0 / 2;
    
    ewma_init(&stage->throughput, g_tuning.config.ewma_alpha);
    
    stage->total_cycles = 0;
    stage->total_elements = 0;
    stage->best_distance = d0;
    stage->best_throughput = DBL_MAX;  // Use DBL_MAX instead of INFINITY
    stage->tuning_phase = 0;
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
    if (elements == 0) return; // Avoid division by zero
    
    stage->total_cycles += cycles;
    stage->total_elements += elements;
    
    double throughput = (double)cycles / (double)elements;
    ewma_update(&stage->throughput, throughput);
    
    if (!ewma_is_warmed_up(&stage->throughput, g_tuning.config.ewma_warmup_samples)) {
        return;
    }
    
    double filtered_throughput = ewma_get(&stage->throughput);
    
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
            // Guard against division by near-zero
            double denom = (stage->best_throughput > EPS) ? stage->best_throughput : EPS;
            double improvement = (stage->best_throughput - filtered_throughput) / denom;
            
            if (improvement > stage->improvement_threshold) {
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
                
                // Accelerate step size on success
                stage->search_step_size = (stage->search_step_size * 3) / 2;
                if (stage->search_step_size > 16) stage->search_step_size = 16;
                
            } else {
                stage->iterations_without_improvement++;
                
                if (stage->iterations_without_improvement >= stage->max_iterations_no_improve) {
                    // Converged
                    stage->tuning_phase = 2;
                    stage->distance_input = stage->best_distance;
                    stage->distance_twiddle = stage->best_distance / 2;
                    
                    if (g_tuning.config.enable_logging) {
                        printf("Stage %d: Converged at distance=%d (%.2f cycles/elem)\n",
                            stage->stage_idx, stage->best_distance, stage->best_throughput);
                    }
                    break;
                }
                
                // Reverse direction and reduce step periodically
                if (stage->iterations_without_improvement % 3 == 0) {
                    stage->search_direction *= -1;
                    stage->search_step_size /= 2;
                    if (stage->search_step_size < 1) stage->search_step_size = 1;
                }
            }
            
            // Compute next distance to try
            int next_distance = stage->distance_input + 
                               (stage->search_direction * stage->search_step_size);
            
            // Clamp to valid range and bounce at boundaries
            if (next_distance < g_tuning.config.min_distance) {
                next_distance = g_tuning.config.min_distance;
                stage->search_direction *= -1;
            }
            if (next_distance > g_tuning.config.max_distance) {
                next_distance = g_tuning.config.max_distance;
                stage->search_direction *= -1;
            }
            
            stage->distance_input = next_distance;
            stage->distance_twiddle = next_distance / 2;
            break;
        }
            
        case 2: // Converged
            if (g_tuning.config.enable_periodic_retune &&
                stage->total_elements > 0 &&
                (stage->total_elements % g_tuning.config.retune_interval) < (uint64_t)elements) {
                
                // Retrigger if performance degrades by >10%
                if (filtered_throughput > stage->best_throughput * 1.10) {
                    if (g_tuning.config.enable_logging) {
                        printf("Stage %d: Performance degraded, restarting search\n", 
                            stage->stage_idx);
                    }
                    stage->tuning_phase = 1;
                    stage->iterations_without_improvement = 0;
                    stage->search_step_size = g_tuning.config.initial_step_size;
                }
            }
            break;
    }
}

//==============================================================================
// GLOBAL (SIMPLE) TUNING WITH EWMA
//==============================================================================

static void init_global_tuning(int initial_distance) {
    // Clamp initial distance to valid range
    int d0 = initial_distance;
    if (d0 < g_tuning.config.min_distance) d0 = g_tuning.config.min_distance;
    if (d0 > g_tuning.config.max_distance) d0 = g_tuning.config.max_distance;
    
    g_tuning.global.throughput_ewma = 0.0;
    g_tuning.global.current_distance = d0;
    g_tuning.global.best_distance = d0;
    g_tuning.global.best_throughput = DBL_MAX;  // Use DBL_MAX instead of INFINITY
    g_tuning.global.tuning_phase = 0;
    g_tuning.global.iterations_without_improvement = 0;
    g_tuning.global.total_calls = 0;
}

static void update_global_tuning(uint64_t cycles, int elements) {
    if (elements == 0) return;
    
    g_tuning.global.total_calls++;
    
    double throughput = (double)cycles / (double)elements;
    
    if (g_tuning.global.total_calls == 1) {
        g_tuning.global.throughput_ewma = throughput;
    } else {
        g_tuning.global.throughput_ewma = 
            g_tuning.config.ewma_alpha * throughput + 
            (1.0 - g_tuning.config.ewma_alpha) * g_tuning.global.throughput_ewma;
    }
    
    if (g_tuning.global.total_calls < (uint64_t)g_tuning.config.ewma_warmup_samples) {
        return;
    }
    
    double filtered = g_tuning.global.throughput_ewma;
    
    switch (g_tuning.global.tuning_phase) {
        case 0:
            g_tuning.global.best_throughput = filtered;
            g_tuning.global.best_distance = g_tuning.global.current_distance;
            g_tuning.global.tuning_phase = 1;
            break;
            
        case 1: {
            // Guard against division by near-zero
            double denom = (g_tuning.global.best_throughput > EPS) ? 
                          g_tuning.global.best_throughput : EPS;
            double improvement = (g_tuning.global.best_throughput - filtered) / denom;
            
            if (improvement > g_tuning.config.improvement_threshold) {
                g_tuning.global.best_throughput = filtered;
                g_tuning.global.best_distance = g_tuning.global.current_distance;
                g_tuning.global.iterations_without_improvement = 0;
                
                if (g_tuning.config.enable_logging) {
                    printf("Global: Improvement %.2f%%, distance=%d\n",
                        improvement * 100.0, g_tuning.global.best_distance);
                }
            } else {
                g_tuning.global.iterations_without_improvement++;
                
                if (g_tuning.global.iterations_without_improvement >= 
                    g_tuning.config.max_search_iterations) {
                    g_tuning.global.tuning_phase = 2;
                    g_tuning.global.current_distance = g_tuning.global.best_distance;
                    
                    if (g_tuning.config.enable_logging) {
                        printf("Global: Converged at distance=%d\n", 
                            g_tuning.global.best_distance);
                    }
                }
            }
            
            if (g_tuning.global.tuning_phase == 1) {
                int step = 4;
                if (g_tuning.global.iterations_without_improvement % 2 == 0) {
                    g_tuning.global.current_distance += step;
                } else {
                    g_tuning.global.current_distance -= step;
                }
                
                // Clamp to valid range
                if (g_tuning.global.current_distance < g_tuning.config.min_distance)
                    g_tuning.global.current_distance = g_tuning.config.min_distance;
                if (g_tuning.global.current_distance > g_tuning.config.max_distance)
                    g_tuning.global.current_distance = g_tuning.config.max_distance;
            }
            break;
        }
            
        case 2:
            if (g_tuning.config.enable_periodic_retune &&
                (g_tuning.global.total_calls % g_tuning.config.retune_interval) == 0) {
                
                if (filtered > g_tuning.global.best_throughput * 1.10) {
                    g_tuning.global.tuning_phase = 1;
                    g_tuning.global.iterations_without_improvement = 0;
                }
            }
            break;
    }
}

//==============================================================================
// PUBLIC API
//==============================================================================

/**
 * @brief Initialize adaptive tuning
 * 
 * NOTE: This is NOT thread-safe. Call from a single controller thread.
 */
void init_adaptive_tuning(
    int num_stages,
    const int *initial_distances,
    const int *working_set_sizes,
    const int *radixes
) {
    g_tuning.num_stages = num_stages;
    g_tuning.tuned_stage_count = 0;
    
    g_tuning.stats.total_fft_calls = 0;
    g_tuning.stats.total_tuning_changes = 0;
    g_tuning.stats.avg_improvement = 0.0;
    
    if (g_tuning.config.mode == TUNING_MODE_DISABLED) {
        return;
    }
    
    int global_initial = (num_stages > 0 && initial_distances) ? initial_distances[0] : 8;
    init_global_tuning(global_initial);
    
    if (g_tuning.config.mode == TUNING_MODE_PER_STAGE || 
        g_tuning.config.tune_stages_independently) {
        
        if (g_tuning.stages) {
            free(g_tuning.stages);
        }
        
        g_tuning.stages = (stage_tuning_state_t*)calloc(
            num_stages, sizeof(stage_tuning_state_t)
        );
        
        if (!g_tuning.stages) {
            fprintf(stderr, "Failed to allocate per-stage tuning state\n");
            g_tuning.config.mode = TUNING_MODE_SIMPLE;
            return;
        }
        
        int stages_to_tune = g_tuning.config.stages_to_tune;
        if (stages_to_tune == 0 || stages_to_tune > num_stages) {
            stages_to_tune = num_stages;
        }
        
        for (int i = 0; i < stages_to_tune; i++) {
            init_stage_tuning(
                &g_tuning.stages[i],
                i,
                initial_distances[i],
                working_set_sizes[i],
                radixes[i]
            );
        }
        
        g_tuning.tuned_stage_count = stages_to_tune;
        
        if (g_tuning.config.enable_logging) {
            printf("Initialized per-stage tuning for %d stages\n", stages_to_tune);
        }
    }
}

void cleanup_adaptive_tuning(void) {
    if (g_tuning.stages) {
        free(g_tuning.stages);
        g_tuning.stages = NULL;
    }
    g_tuning.num_stages = 0;
    g_tuning.tuned_stage_count = 0;
}

uint64_t profile_fft_start(void) {
    if (g_tuning.config.mode == TUNING_MODE_DISABLED) {
        return 0;
    }
    return read_tsc_inline();
}

/**
 * @brief End profiling and update tuning
 * 
 * NOTE: This is NOT thread-safe. Call from a single controller thread.
 */
void profile_fft_end(uint64_t start_cycles, int n_elements, int stage_idx) {
    if (g_tuning.config.mode == TUNING_MODE_DISABLED) {
        return;
    }
    
    uint64_t end_cycles = read_tsc_inline();
    uint64_t elapsed = end_cycles - start_cycles;
    
    g_tuning.stats.total_fft_calls++;
    
    switch (g_tuning.config.mode) {
        case TUNING_MODE_SIMPLE:
        case TUNING_MODE_EWMA:
            update_global_tuning(elapsed, n_elements);
            break;
            
        case TUNING_MODE_PER_STAGE:
            if (stage_idx >= 0 && stage_idx < g_tuning.tuned_stage_count && g_tuning.stages) {
                update_stage_tuning(&g_tuning.stages[stage_idx], elapsed, n_elements);
            }
            break;
            
        default:
            break;
    }
}

/**
 * @brief Get tuned distance for a stage
 * Returns -1 if tuning is disabled
 * Falls back to global distance for untuned stages
 */
int get_tuned_distance(int stage_idx) {
    if (g_tuning.config.mode == TUNING_MODE_DISABLED) {
        return -1;
    }
    
    // Return per-stage distance if available and tuned
    if (g_tuning.config.mode == TUNING_MODE_PER_STAGE && 
        g_tuning.stages && 
        stage_idx >= 0 && 
        stage_idx < g_tuning.tuned_stage_count) {
        return g_tuning.stages[stage_idx].distance_input;
    }
    
    // Fall back to global distance
    return g_tuning.global.current_distance;
}

/**
 * @brief Apply tuned distances back to stage prefetch configs
 * 
 * Call this periodically after profile_fft_end() to propagate tuning
 * results back to the FFT engine.
 */
void apply_tuned_distances(stage_prefetch_t *stages, int n) {
    if (!stages || g_tuning.config.mode == TUNING_MODE_DISABLED) return;
    
    if (g_tuning.config.mode == TUNING_MODE_PER_STAGE && g_tuning.stages) {
        // Apply per-stage tuned distances
        int m = (g_tuning.tuned_stage_count < n) ? g_tuning.tuned_stage_count : n;
        for (int i = 0; i < m; ++i) {
            int d = g_tuning.stages[i].distance_input;
            stages[i].distance_input   = d;
            stages[i].distance_output  = d;
            stages[i].distance_twiddle = d / 2;
        }
        // Fall back to global for remaining stages
        if (m < n) {
            int d = g_tuning.global.current_distance;
            for (int i = m; i < n; ++i) {
                stages[i].distance_input   = d;
                stages[i].distance_output  = d;
                stages[i].distance_twiddle = d / 2;
            }
        }
    } else {
        // Apply global tuned distance to all stages
        int d = g_tuning.global.current_distance;
        for (int i = 0; i < n; ++i) {
            stages[i].distance_input   = d;
            stages[i].distance_output  = d;
            stages[i].distance_twiddle = d / 2;
        }
    }
}

void set_tuning_mode(tuning_mode_t mode) {
    g_tuning.config.mode = mode;
}

void configure_tuning(
    double ewma_alpha,
    int ewma_warmup_samples,
    double improvement_threshold,
    int max_search_iterations,
    int initial_step_size
) {
    g_tuning.config.ewma_alpha = ewma_alpha;
    g_tuning.config.ewma_warmup_samples = ewma_warmup_samples;
    g_tuning.config.improvement_threshold = improvement_threshold;
    g_tuning.config.max_search_iterations = max_search_iterations;
    g_tuning.config.initial_step_size = initial_step_size;
}

void set_tuning_logging(bool enable) {
    g_tuning.config.enable_logging = enable;
}

void set_periodic_retune(bool enable, uint64_t interval) {
    g_tuning.config.enable_periodic_retune = enable;
    if (interval > 0) {
        g_tuning.config.retune_interval = interval;
    }
}

void get_tuning_stats(
    uint64_t *total_calls,
    uint64_t *total_changes,
    double *avg_improvement,
    int *best_distances_out,
    int max_stages
) {
    if (total_calls) *total_calls = g_tuning.stats.total_fft_calls;
    if (total_changes) *total_changes = g_tuning.stats.total_tuning_changes;
    if (avg_improvement) *avg_improvement = g_tuning.stats.avg_improvement;
    
    if (best_distances_out && max_stages > 0) {
        if (g_tuning.config.mode == TUNING_MODE_PER_STAGE && g_tuning.stages) {
            int n = (max_stages < g_tuning.tuned_stage_count) ? max_stages : g_tuning.tuned_stage_count;
            for (int i = 0; i < n; i++) {
                best_distances_out[i] = g_tuning.stages[i].best_distance;
            }
            // Fill remaining with global
            for (int i = n; i < max_stages; i++) {
                best_distances_out[i] = g_tuning.global.best_distance;
            }
        } else {
            for (int i = 0; i < max_stages; i++) {
                best_distances_out[i] = g_tuning.global.best_distance;
            }
        }
    }
}

void print_tuning_report(void) {
    printf("\n=== Adaptive Tuning Report ===\n");
    printf("Mode: ");
    switch (g_tuning.config.mode) {
        case TUNING_MODE_DISABLED: printf("Disabled\n"); return;
        case TUNING_MODE_SIMPLE: printf("Simple\n"); break;
        case TUNING_MODE_EWMA: printf("EWMA-Filtered\n"); break;
        case TUNING_MODE_PER_STAGE: printf("Per-Stage\n"); break;
    }
    
    printf("Total FFT calls: %llu\n", (unsigned long long)g_tuning.stats.total_fft_calls);
    printf("Total tuning changes: %llu\n", (unsigned long long)g_tuning.stats.total_tuning_changes);
    printf("Average improvement: %.2f%%\n", g_tuning.stats.avg_improvement * 100.0);
    
    if (g_tuning.config.mode == TUNING_MODE_PER_STAGE && g_tuning.stages) {
        printf("\nPer-Stage Results:\n");
        for (int i = 0; i < g_tuning.tuned_stage_count; i++) {
            stage_tuning_state_t *s = &g_tuning.stages[i];
            printf("  Stage %d: distance=%d, throughput=%.2f cycles/elem, phase=%d\n",
                i, s->best_distance, s->best_throughput, s->tuning_phase);
        }
    } else {
        printf("\nGlobal Result:\n");
        printf("  Best distance: %d\n", g_tuning.global.best_distance);
        printf("  Best throughput: %.2f cycles/elem\n", g_tuning.global.best_throughput);
        printf("  Phase: %d\n", g_tuning.global.tuning_phase);
    }
    
    printf("==============================\n\n");
}
