//==============================================================================
// ADVANCED PREFETCH STRATEGY (FFTW-inspired, deep version)
//==============================================================================

/**
 * @brief Prefetch hint levels (temporal locality)
 * 
 * T0: High temporal locality (will be used soon)
 * T1: Moderate temporal locality (will be used later)
 * T2: Low temporal locality (likely evicted before reuse)
 * NTA: Non-temporal access (streaming, no cache pollution)
 */
typedef enum {
    PREFETCH_HINT_T0  = _MM_HINT_T0,
    PREFETCH_HINT_T1  = _MM_HINT_T1,
    PREFETCH_HINT_T2  = _MM_HINT_T2,
    PREFETCH_HINT_NTA = _MM_HINT_NTA
} prefetch_hint_t;

/**
 * @brief Prefetch scheduling strategy
 */
typedef enum {
    PREFETCH_NONE,        // No prefetch (tiny transforms)
    PREFETCH_SINGLE,      // Single stream (sequential access)
    PREFETCH_DUAL,        // Two streams (read/write or two reads)
    PREFETCH_MULTI,       // Multiple streams (mixed-radix stages)
    PREFETCH_STRIDED      // Strided access (transpose-like patterns)
} prefetch_strategy_t;

/**
 * @brief Per-stage prefetch configuration
 * 
 * FFTW Key Insight: Different stages need different prefetch strategies!
 * - Early stages: large stride, scattered access → aggressive prefetch
 * - Late stages: small stride, sequential → conservative prefetch
 * - Middle stages: mixed patterns → adaptive prefetch
 */
typedef struct {
    int distance_input;     // Prefetch distance for input reads (cache lines)
    int distance_output;    // Prefetch distance for output writes
    int distance_twiddle;   // Prefetch distance for twiddle factors
    int hint_input;         // Hint for input data
    int hint_output;        // Hint for output data
    int hint_twiddle;       // Hint for twiddles
    prefetch_strategy_t strategy; // Overall strategy
    int block_size;         // Blocking factor (for cache reuse)
    int enable;             // Master enable/disable
} stage_prefetch_t;

/**
 * @brief Dynamic prefetch configuration (per-transform tuning)
 */
typedef struct {
    stage_prefetch_t *stages;  // Per-stage configs (malloc'd array)
    int num_stages;            // Number of stages in transform
    int l1_size;               // L1 data cache size (bytes)
    int l2_size;               // L2 cache size (bytes)
    int l3_size;               // L3 cache size (bytes)
    int cache_line_size;       // Cache line size (typically 64B)
    int enable_runtime_tuning; // Enable adaptive tuning during execution
} prefetch_config_t;

/**
 * @brief Global prefetch configuration with sensible defaults
 */
static prefetch_config_t g_prefetch_config = {
    .stages = NULL,
    .num_stages = 0,
    .l1_size = 32 * 1024,      // 32KB (conservative)
    .l2_size = 256 * 1024,     // 256KB (typical)
    .l3_size = 8 * 1024 * 1024, // 8MB (typical)
    .cache_line_size = 64,     // 64B (x86-64 standard)
    .enable_runtime_tuning = 1
};

/**
 * @brief Detect CPU cache sizes (x86-64 CPUID)
 * 
 * FFTW does this to adapt to hardware!
 */
static inline void detect_cache_sizes(void) {
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int eax, ebx, ecx, edx;
    
    // Check for CPUID support
    __asm__ __volatile__ (
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(0x0)
    );
    
    if (eax >= 0x4) {
        // Intel cache topology (leaf 0x4)
        for (int i = 0; i < 10; ++i) {
            __asm__ __volatile__ (
                "cpuid"
                : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                : "a"(0x4), "c"(i)
            );
            
            int cache_type = eax & 0x1F;
            if (cache_type == 0) break; // No more caches
            
            int cache_level = (eax >> 5) & 0x7;
            int cache_size = ((ebx >> 22) + 1) * 
                           ((ebx >> 12 & 0x3FF) + 1) * 
                           ((ebx & 0xFFF) + 1) * 
                           (ecx + 1);
            
            if (cache_type == 1 || cache_type == 3) { // Data or Unified
                if (cache_level == 1) g_prefetch_config.l1_size = cache_size;
                else if (cache_level == 2) g_prefetch_config.l2_size = cache_size;
                else if (cache_level == 3) g_prefetch_config.l3_size = cache_size;
            }
        }
    }
#endif
    
    // Fallback: use conservative defaults (already set)
}

/**
 * @brief Compute working set size for a stage
 * 
 * FFTW Key Insight: Each stage has a different working set!
 * - Stage 0 (N): processes all N elements, WS = N * sizeof(complex)
 * - Stage k: processes N/r^k elements per butterfly, WS decreases
 */
static inline int compute_stage_working_set(int n_fft, int stage_idx, 
                                            const int *factors, int num_factors) {
    int n_stage = n_fft;
    for (int i = 0; i < stage_idx && i < num_factors; ++i) {
        n_stage /= factors[i];
    }
    return n_stage * sizeof(fft_data);
}

/**
 * @brief Compute optimal prefetch distance for a stage
 * 
 * FFTW Strategy:
 * - Small working sets (< L1): short distance (data stays hot)
 * - Medium working sets (L1..L2): moderate distance (avoid misses)
 * - Large working sets (> L2): long distance (hide memory latency)
 * - Huge working sets (> L3): very long + NTA (streaming)
 */
static inline int compute_stage_prefetch_distance(int working_set, 
                                                  int stride,
                                                  int cache_level) {
    const int cl_size = g_prefetch_config.cache_line_size;
    const int l1 = g_prefetch_config.l1_size;
    const int l2 = g_prefetch_config.l2_size;
    const int l3 = g_prefetch_config.l3_size;
    
    // Base distance on working set size
    int base_distance;
    if (working_set < l1 / 2) {
        base_distance = 2;  // 128B ahead (L1-hot)
    } else if (working_set < l1) {
        base_distance = 4;  // 256B ahead (L1-warm)
    } else if (working_set < l2 / 2) {
        base_distance = 8;  // 512B ahead (L2-hot)
    } else if (working_set < l2) {
        base_distance = 12; // 768B ahead (L2-warm)
    } else if (working_set < l3 / 2) {
        base_distance = 16; // 1KB ahead (L3-hot)
    } else if (working_set < l3) {
        base_distance = 24; // 1.5KB ahead (L3-warm)
    } else {
        base_distance = 32; // 2KB ahead (streaming)
    }
    
    // Adjust for stride (scattered access needs more aggressive prefetch)
    if (stride > 1) {
        base_distance += (stride / 2); // More scatter → more prefetch
    }
    
    return base_distance;
}

/**
 * @brief Compute optimal hint for a stage
 */
static inline int compute_stage_hint(int working_set) {
    const int l1 = g_prefetch_config.l1_size;
    const int l2 = g_prefetch_config.l2_size;
    const int l3 = g_prefetch_config.l3_size;
    
    if (working_set < l1) return _MM_HINT_T0;        // Target L1
    else if (working_set < l2) return _MM_HINT_T1;   // Target L2
    else if (working_set < l3) return _MM_HINT_T2;   // Target L3
    else return _MM_HINT_NTA;                         // Non-temporal
}

/**
 * @brief Initialize per-stage prefetch configuration
 * 
 * FFTW Key: Each stage gets a custom prefetch plan!
 */
static inline void init_stage_prefetch(fft_object fft_obj) {
    const int n_fft = fft_obj->n_fft;
    const int num_factors = fft_obj->lf;
    const int *factors = fft_obj->factors;
    
    // Allocate per-stage configs
    if (g_prefetch_config.stages) {
        free(g_prefetch_config.stages);
    }
    g_prefetch_config.stages = (stage_prefetch_t *)malloc(
        num_factors * sizeof(stage_prefetch_t)
    );
    g_prefetch_config.num_stages = num_factors;
    
    if (!g_prefetch_config.stages) return; // Allocation failure, disable prefetch
    
    // Configure each stage
    int stride = 1;
    for (int stage = 0; stage < num_factors; ++stage) {
        const int radix = factors[stage];
        const int working_set = compute_stage_working_set(n_fft, stage, factors, num_factors);
        
        stage_prefetch_t *cfg = &g_prefetch_config.stages[stage];
        
        // Determine strategy based on radix and working set
        if (working_set < 1024) {
            cfg->strategy = PREFETCH_NONE; // Too small, overhead dominates
        } else if (radix <= 4 && working_set < g_prefetch_config.l1_size) {
            cfg->strategy = PREFETCH_SINGLE; // Sequential, fits in L1
        } else if (radix <= 8) {
            cfg->strategy = PREFETCH_DUAL; // Two streams (in/out)
        } else {
            cfg->strategy = PREFETCH_MULTI; // Multiple lanes
        }
        
        // Compute distances
        cfg->distance_input = compute_stage_prefetch_distance(working_set, stride, 0);
        cfg->distance_output = cfg->distance_input; // Same for now
        cfg->distance_twiddle = cfg->distance_input / 2; // Twiddles reused more
        
        // Compute hints
        cfg->hint_input = compute_stage_hint(working_set);
        cfg->hint_output = compute_stage_hint(working_set);
        cfg->hint_twiddle = _MM_HINT_T0; // Twiddles should stay hot
        
        // Blocking for large radices
        if (radix >= 16 && working_set > g_prefetch_config.l2_size) {
            cfg->block_size = g_prefetch_config.l2_size / (radix * sizeof(fft_data));
        } else {
            cfg->block_size = 0; // No blocking
        }
        
        cfg->enable = (cfg->strategy != PREFETCH_NONE);
        
        stride *= radix; // Stride grows exponentially
    }
}

/**
 * @brief Get prefetch config for current recursion depth
 */
static inline stage_prefetch_t* get_stage_config(int factor_index) {
    if (factor_index < 0 || factor_index >= g_prefetch_config.num_stages) {
        return NULL;
    }
    return &g_prefetch_config.stages[factor_index];
}

/**
 * @brief Unified prefetch macro with dynamic configuration
 */
#define PREFETCH(addr, hint) _mm_prefetch((const char *)(addr), (hint))

/**
 * @brief Advanced prefetch for recursive stages (multi-stream)
 * 
 * Prefetches:
 * 1. Input lanes (scattered by stride)
 * 2. Twiddle factors (if present)
 * 3. Output locations (write prefetch)
 */
static inline void prefetch_stage_recursive(
    const fft_data *input_base,
    const fft_data *twiddle_base,
    fft_data *output_base,
    int idx,
    int stride,
    int radix,
    stage_prefetch_t *cfg
) {
    if (!cfg || !cfg->enable) return;
    
    const int d_in = cfg->distance_input;
    const int d_tw = cfg->distance_twiddle;
    const int d_out = cfg->distance_output;
    
    // Prefetch input lanes (multi-stream)
    if (cfg->strategy >= PREFETCH_MULTI) {
        for (int lane = 0; lane < radix && lane < 4; ++lane) {
            PREFETCH(input_base + (idx + d_in) + lane * stride, cfg->hint_input);
        }
    } else {
        PREFETCH(input_base + idx + d_in, cfg->hint_input);
    }
    
    // Prefetch twiddles (if present)
    if (twiddle_base && cfg->strategy >= PREFETCH_DUAL) {
        PREFETCH(twiddle_base + idx + d_tw, cfg->hint_twiddle);
    }
    
    // Write prefetch for output (optional, CPU-dependent)
    // Some CPUs benefit from write prefetch, others don't
    // if (output_base && cfg->strategy >= PREFETCH_DUAL) {
    //     PREFETCH(output_base + idx + d_out, cfg->hint_output);
    // }
}

/**
 * @brief Prefetch for tight butterfly loops
 */
static inline void prefetch_butterfly_loop(
    const fft_data *input,
    const fft_data *twiddle,
    int idx,
    stage_prefetch_t *cfg
) {
    if (!cfg || !cfg->enable) return;
    
    PREFETCH(input + idx + cfg->distance_input, cfg->hint_input);
    
    if (twiddle && cfg->strategy >= PREFETCH_DUAL) {
        PREFETCH(twiddle + idx + cfg->distance_twiddle, cfg->hint_twiddle);
    }
}

/**
 * @brief Initialize global prefetch system
 * 
 * Call this once at program startup (or per fft_init)
 */
static inline void init_prefetch_system(fft_object fft_obj) {
    // Detect cache sizes (once)
    static int cache_detected = 0;
    if (!cache_detected) {
        detect_cache_sizes();
        cache_detected = 1;
    }
    
    // Initialize per-stage configs
    init_stage_prefetch(fft_obj);
}

/**
 * @brief Runtime profiling for adaptive prefetch tuning
 * 
 * FFTW Feature: Learns optimal prefetch during execution!
 * This is a simplified version - full FFTW does cycle counting.
 */
typedef struct {
    unsigned long long total_cycles;    // Accumulated execution cycles
    unsigned long long total_calls;     // Number of FFT executions
    int current_distance;               // Current prefetch distance
    int best_distance;                  // Best distance found so far
    double best_throughput;             // Best cycles/element
    int tuning_phase;                   // 0=measure, 1=search, 2=converged
    int tuning_iterations;              // Iterations in current phase
} prefetch_profile_t;

static prefetch_profile_t g_prefetch_profile = {0};

/**
 * @brief Read CPU timestamp counter (x86-64)
 */
static inline unsigned long long read_tsc(void) {
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)hi << 32) | lo;
#else
    return 0; // Fallback for non-x86
#endif
}

/**
 * @brief Start profiling an FFT execution
 */
static inline void profile_start(void) {
    if (!g_prefetch_config.enable_runtime_tuning) return;
    g_prefetch_profile.total_cycles = read_tsc();
}

/**
 * @brief End profiling and update adaptive parameters
 */
static inline void profile_end(int n_elements) {
    if (!g_prefetch_config.enable_runtime_tuning) return;
    
    unsigned long long end = read_tsc();
    unsigned long long elapsed = end - g_prefetch_profile.total_cycles;
    
    g_prefetch_profile.total_calls++;
    
    // Compute throughput (cycles per complex element)
    double throughput = (double)elapsed / (double)n_elements;
    
    // Adaptive tuning state machine
    switch (g_prefetch_profile.tuning_phase) {
        case 0: // Initial measurement
            g_prefetch_profile.best_throughput = throughput;
            g_prefetch_profile.best_distance = g_prefetch_profile.current_distance;
            g_prefetch_profile.tuning_phase = 1;
            g_prefetch_profile.tuning_iterations = 0;
            break;
            
        case 1: // Search phase (try different distances)
            if (throughput < g_prefetch_profile.best_throughput * 0.98) {
                // Found better configuration (2% improvement threshold)
                g_prefetch_profile.best_throughput = throughput;
                g_prefetch_profile.best_distance = g_prefetch_profile.current_distance;
                g_prefetch_profile.tuning_iterations = 0;
            } else {
                g_prefetch_profile.tuning_iterations++;
            }
            
            // Try different distances (hill climbing)
            if (g_prefetch_profile.tuning_iterations < 5) {
                // Increase distance
                g_prefetch_profile.current_distance += 4;
            } else if (g_prefetch_profile.tuning_iterations < 10) {
                // Decrease distance
                g_prefetch_profile.current_distance -= 2;
            } else {
                // Converged - use best found
                g_prefetch_profile.tuning_phase = 2;
                g_prefetch_profile.current_distance = g_prefetch_profile.best_distance;
                
                // Apply best distance to all stages
                for (int i = 0; i < g_prefetch_config.num_stages; ++i) {
                    g_prefetch_config.stages[i].distance_input = g_prefetch_profile.best_distance;
                }
            }
            break;
            
        case 2: // Converged (occasionally re-check)
            if (g_prefetch_profile.total_calls % 1000 == 0) {
                // Periodic re-tuning (adaptive to changing workload)
                g_prefetch_profile.tuning_phase = 1;
                g_prefetch_profile.tuning_iterations = 0;
            }
            break;
    }
}

/**
 * @brief Software prefetch scheduling (compiler-independent)
 * 
 * FFTW Insight: Compiler auto-prefetch often fails for FFT access patterns!
 * Explicit prefetch gives 10-30% speedup on complex stride patterns.
 */

/**
 * @brief Prefetch with distance adjustment based on loop unrolling
 * 
 * FFTW Key: Prefetch distance must account for loop body size!
 * Large unroll factors (8x, 16x) need proportionally longer prefetch.
 */
static inline int adjust_distance_for_unroll(int base_distance, int unroll_factor) {
    // Rule of thumb: prefetch_distance ≈ base_distance * sqrt(unroll_factor)
    // This accounts for increased instruction count in loop body
    if (unroll_factor <= 2) return base_distance;
    else if (unroll_factor <= 4) return base_distance + 2;
    else if (unroll_factor <= 8) return base_distance + 4;
    else return base_distance + 8; // 16x+ unrolling
}

/**
 * @brief Stride-aware prefetch (for transpose-like access patterns)
 * 
 * FFTW Insight: Large strides trash TLB and cache lines!
 * Prefetch multiple cache lines per iteration for strided access.
 */
static inline void prefetch_strided(
    const fft_data *base,
    int idx,
    int stride,
    int num_streams,
    stage_prefetch_t *cfg
) {
    if (!cfg || !cfg->enable) return;
    
    const int d = cfg->distance_input;
    const int hint = cfg->hint_input;
    const int cl_per_stride = (stride * sizeof(fft_data) + 63) / 64; // Cache lines
    
    // Prefetch multiple streams (scattered by stride)
    for (int s = 0; s < num_streams && s < 4; ++s) {
        const fft_data *ptr = base + (idx + d) + s * stride;
        PREFETCH(ptr, hint);
        
        // For very large strides, prefetch intermediate cache lines
        if (cl_per_stride > 2) {
            for (int cl = 1; cl < cl_per_stride && cl < 4; ++cl) {
                PREFETCH((char*)ptr + cl * 64, hint);
            }
        }
    }
}

/**
 * @brief Blocking-aware prefetch (for large radices)
 * 
 * FFTW Insight: Radix-16, 32 with cache blocking need different prefetch!
 * Prefetch next block while processing current block.
 */
static inline void prefetch_blocked(
    const fft_data *base,
    int block_start,
    int block_size,
    stage_prefetch_t *cfg
) {
    if (!cfg || !cfg->enable || cfg->block_size == 0) return;
    
    // Prefetch start of next block
    const int next_block = block_start + block_size;
    const int hint = cfg->hint_input;
    
    // Touch multiple cache lines at block boundary
    for (int i = 0; i < 8 && i < block_size; i += 4) {
        PREFETCH(base + next_block + i, hint);
    }
}

/**
 * @brief Write-allocate prefetch (for output-heavy stages)
 * 
 * FFTW Insight: Writing cold cache lines causes read-for-ownership!
 * Prefetch write locations to avoid RFO misses.
 * 
 * NOTE: Only beneficial on some CPUs (Intel > Haswell, AMD Zen3+)
 */
static inline void prefetch_write(
    fft_data *output,
    int idx,
    int distance,
    stage_prefetch_t *cfg
) {
    if (!cfg || !cfg->enable) return;
    
    // Use NTA hint for write prefetch (avoid cache pollution)
    // Some CPUs have separate write prefetch instructions (prefetchw)
#if defined(__PRFCHW__) || defined(__3dNOW__)
    __builtin_prefetch(output + idx + distance, 1, 0); // write=1, nta=0
#else
    PREFETCH(output + idx + distance, _MM_HINT_NTA);
#endif
}

/**
 * @brief Group prefetch (for radix-based access patterns)
 * 
 * FFTW Insight: Radix-N butterfly reads N lanes simultaneously!
 * Prefetch all lanes of next group for better pipelining.
 */
static inline void prefetch_radix_group(
    const fft_data *base,
    int group_idx,
    int group_size,
    int radix,
    stage_prefetch_t *cfg
) {
    if (!cfg || !cfg->enable) return;
    
    const int d = cfg->distance_input;
    const int hint = cfg->hint_input;
    
    // Prefetch next radix group (all lanes)
    const int next_group = group_idx + d;
    for (int lane = 0; lane < radix && lane < 8; ++lane) {
        PREFETCH(base + next_group + lane * group_size, hint);
    }
}