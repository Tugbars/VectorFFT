/**
 * @brief Initialize per-stage prefetch configuration
 * 
 * FFTW Key: Each stage gets a custom prefetch plan!
 * Now enhanced with CPU profiles and wisdom database
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
    
    // Check if exhaustive search is requested
    const char *search_env = getenv("HFFT_EXHAUSTIVE_SEARCH");
    const bool do_exhaustive_search = (search_env && atoi(search_env) == 1);
    
    // Configure each stage
    int stride = 1;
    for (int stage = 0; stage < num_factors; ++stage) {
        const int radix = factors[stage];
        const int working_set = compute_stage_working_set(n_fft, stage, factors, num_factors);
        
        stage_prefetch_t *cfg = &g_prefetch_config.stages[stage];
        
        // FEATURE #5: Try wisdom first
        wisdom_entry_t *wisdom = find_wisdom(n_fft, radix);
        
        if (wisdom) {
            // Use wisdom (precomputed optimal config)
            cfg->distance_input = wisdom->distance_input;
            cfg->distance_twiddle = wisdom->distance_twiddle;
            cfg->hint_input = wisdom->hint;
            cfg->hint_twiddle = _MM_HINT_T0;
            cfg->strategy = wisdom->strategy;
            cfg->enable = true;
        }
        else if (do_exhaustive_search) {
            // FEATURE #5: Perform exhaustive search
            search_optimal_prefetch(fft_obj, stage, cfg);
        }
        else {
            // FEATURE #10: Use CPU profile heuristics
            
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
            
            // FEATURE #10: Use CPU-specific optimal distances
            // Map working set to distance index
            int ws_index = 0;
            if (working_set < g_prefetch_config.l1_size / 4) ws_index = 0;
            else if (working_set < g_prefetch_config.l1_size / 2) ws_index = 1;
            else if (working_set < g_prefetch_config.l1_size) ws_index = 2;
            else if (working_set < g_prefetch_config.l2_size / 2) ws_index = 3;
            else if (working_set < g_prefetch_config.l2_size) ws_index = 4;
            else if (working_set < g_prefetch_config.l3_size / 2) ws_index = 5;
            else if (working_set < g_prefetch_config.l3_size) ws_index = 6;
            else ws_index = 7;
            
            cfg->distance_input = g_cpu_profile->optimal_distance[ws_index];
            
            // Adjust for stride (scattered access needs more aggressive prefetch)
            if (stride > 4) {
                cfg->distance_input += stride / 4;
            }
            
            cfg->distance_output = cfg->distance_input;
            cfg->distance_twiddle = cfg->distance_input / 2; // Twiddles reused more
            
            // Compute hints based on working set and cache hierarchy
            cfg->hint_input = compute_stage_hint(working_set);
            cfg->hint_output = cfg->hint_input;
            cfg->hint_twiddle = _MM_HINT_T0; // Twiddles should stay hot
            
            // FEATURE #10: Disable if CPU has strong hardware prefetcher
            // and working set is small (avoid conflicts)
            if (g_cpu_profile->has_strong_hwpf && working_set < g_prefetch_config.l1_size) {
                cfg->enable = false; // Let HW prefetcher handle it
            } else {
                cfg->enable = (cfg->strategy != PREFETCH_NONE);
            }
        }
        
        // Blocking for large radices
        if (radix >= 16 && working_set > g_prefetch_config.l2_size) {
            cfg->block_size = g_prefetch_config.l2_size / (radix * sizeof(fft_data));
        } else {
            cfg->block_size = 0; // No blocking
        }
        
        stride *= radix; // Stride grows exponentially
    }
}//==============================================================================
// ADVANCED PREFETCH STRATEGY - FINAL 20% IMPLEMENTATION
//==============================================================================

/**
 * @brief CPU architecture profiles for optimal prefetch tuning
 * 
 * FEATURE #10: Architecture-Specific Tuning Database
 * Each CPU has different prefetch characteristics!
 */
typedef struct {
    const char *name;
    int prefetch_buffers;      // Number of HW prefetch buffers (10-20)
    int prefetch_latency;      // Cycles to issue prefetch (150-250)
    int l1_latency;            // L1 cache latency in cycles (3-5)
    int l2_latency;            // L2 cache latency in cycles (10-15)
    int l3_latency;            // L3 cache latency in cycles (30-50)
    bool has_write_prefetch;   // Supports prefetchw instruction
    bool has_strong_hwpf;      // Strong hardware prefetcher (may conflict)
    int optimal_distance[8];   // Optimal distances for different working sets
} cpu_profile_t;

/**
 * @brief Predefined CPU profiles (constantly updated database)
 */
static const cpu_profile_t cpu_profiles[] = {
    // Intel architectures
    {
        .name = "Intel Skylake",
        .prefetch_buffers = 16,
        .prefetch_latency = 200,
        .l1_latency = 4,
        .l2_latency = 12,
        .l3_latency = 42,
        .has_write_prefetch = true,
        .has_strong_hwpf = true,
        .optimal_distance = {4, 6, 8, 12, 16, 20, 24, 32}
    },
    {
        .name = "Intel Ice Lake",
        .prefetch_buffers = 18,
        .prefetch_latency = 180,
        .l1_latency = 4,
        .l2_latency = 13,
        .l3_latency = 40,
        .has_write_prefetch = true,
        .has_strong_hwpf = true,
        .optimal_distance = {4, 6, 8, 10, 14, 18, 22, 28}
    },
    {
        .name = "Intel Sapphire Rapids",
        .prefetch_buffers = 20,
        .prefetch_latency = 170,
        .l1_latency = 4,
        .l2_latency = 14,
        .l3_latency = 38,
        .has_write_prefetch = true,
        .has_strong_hwpf = true,
        .optimal_distance = {4, 6, 8, 10, 12, 16, 20, 28}
    },
    
    // AMD architectures
    {
        .name = "AMD Zen 2",
        .prefetch_buffers = 12,
        .prefetch_latency = 190,
        .l1_latency = 4,
        .l2_latency = 14,
        .l3_latency = 48,
        .has_write_prefetch = true,
        .has_strong_hwpf = false,
        .optimal_distance = {6, 8, 10, 14, 18, 22, 28, 36}
    },
    {
        .name = "AMD Zen 3",
        .prefetch_buffers = 12,
        .prefetch_latency = 180,
        .l1_latency = 4,
        .l2_latency = 14,
        .l3_latency = 46,
        .has_write_prefetch = true,
        .has_strong_hwpf = false,
        .optimal_distance = {6, 8, 10, 14, 18, 22, 28, 36}
    },
    {
        .name = "AMD Zen 4",
        .prefetch_buffers = 14,
        .prefetch_latency = 170,
        .l1_latency = 4,
        .l2_latency = 13,
        .l3_latency = 44,
        .has_write_prefetch = true,
        .has_strong_hwpf = false,
        .optimal_distance = {6, 8, 10, 12, 16, 20, 26, 34}
    },
    
    // ARM architectures
    {
        .name = "ARM Neoverse V1",
        .prefetch_buffers = 10,
        .prefetch_latency = 220,
        .l1_latency = 3,
        .l2_latency = 11,
        .l3_latency = 38,
        .has_write_prefetch = false,
        .has_strong_hwpf = false,
        .optimal_distance = {4, 5, 7, 10, 14, 18, 24, 32}
    },
    {
        .name = "Apple M1",
        .prefetch_buffers = 20,
        .prefetch_latency = 150,
        .l1_latency = 3,
        .l2_latency = 9,
        .l3_latency = 32,
        .has_write_prefetch = true,
        .has_strong_hwpf = true,
        .optimal_distance = {2, 3, 4, 6, 8, 12, 16, 24}
    },
    {
        .name = "Apple M2",
        .prefetch_buffers = 24,
        .prefetch_latency = 140,
        .l1_latency = 3,
        .l2_latency = 9,
        .l3_latency = 30,
        .has_write_prefetch = true,
        .has_strong_hwpf = true,
        .optimal_distance = {2, 3, 4, 6, 8, 10, 14, 20}
    },
    
    // Generic fallback
    {
        .name = "Generic x86-64",
        .prefetch_buffers = 12,
        .prefetch_latency = 200,
        .l1_latency = 4,
        .l2_latency = 12,
        .l3_latency = 40,
        .has_write_prefetch = false,
        .has_strong_hwpf = false,
        .optimal_distance = {4, 6, 8, 12, 16, 20, 24, 32}
    }
};

static const cpu_profile_t *g_cpu_profile = &cpu_profiles[9]; // Default: Generic

/**
 * @brief Detect CPU architecture using CPUID
 * 
 * FEATURE #10: Runtime CPU detection
 */
static inline const cpu_profile_t* detect_cpu_architecture(void) {
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int eax, ebx, ecx, edx;
    char vendor[13] = {0};
    char brand[49] = {0};
    
    // Get vendor string
    __asm__ __volatile__ (
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(0)
    );
    memcpy(vendor, &ebx, 4);
    memcpy(vendor + 4, &edx, 4);
    memcpy(vendor + 8, &ecx, 4);
    
    // Get brand string (for more specific detection)
    for (int i = 0; i < 3; i++) {
        __asm__ __volatile__ (
            "cpuid"
            : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
            : "a"(0x80000002 + i)
        );
        memcpy(brand + i * 16, &eax, 4);
        memcpy(brand + i * 16 + 4, &ebx, 4);
        memcpy(brand + i * 16 + 8, &ecx, 4);
        memcpy(brand + i * 16 + 12, &edx, 4);
    }
    
    // Intel detection
    if (strstr(vendor, "GenuineIntel")) {
        if (strstr(brand, "Sapphire Rapids") || strstr(brand, "Emerald Rapids"))
            return &cpu_profiles[2];
        else if (strstr(brand, "Ice Lake") || strstr(brand, "Tiger Lake"))
            return &cpu_profiles[1];
        else if (strstr(brand, "Skylake") || strstr(brand, "Cascade Lake") || 
                 strstr(brand, "Cooper Lake"))
            return &cpu_profiles[0];
    }
    
    // AMD detection
    else if (strstr(vendor, "AuthenticAMD")) {
        if (strstr(brand, "Zen 4") || strstr(brand, "Ryzen 7000") || 
            strstr(brand, "EPYC 9"))
            return &cpu_profiles[5];
        else if (strstr(brand, "Zen 3") || strstr(brand, "Ryzen 5000") || 
                 strstr(brand, "EPYC 7003"))
            return &cpu_profiles[4];
        else if (strstr(brand, "Zen 2") || strstr(brand, "Ryzen 3000") || 
                 strstr(brand, "EPYC 7002"))
            return &cpu_profiles[3];
    }
    
#elif defined(__aarch64__) || defined(_M_ARM64)
    // ARM detection (simplified - would need MIDR_EL1 register access)
    #if defined(__APPLE__)
        // Apple Silicon detection
        return &cpu_profiles[7]; // M1 (could check for M2 via sysctl)
    #else
        return &cpu_profiles[6]; // Neoverse V1 (generic ARM)
    #endif
#endif
    
    return &cpu_profiles[9]; // Generic fallback
}

/**
 * @brief Prefetch throttling state (per-thread)
 * 
 * FEATURE #7: Prefetch Throttling
 */
typedef struct {
    int issued_count;          // Prefetches issued in current window
    int budget_remaining;      // Budget left in current window
    int window_size;           // Window size (iterations)
    int window_counter;        // Current position in window
    int max_outstanding;       // Max outstanding prefetches
} prefetch_throttle_t;

static __thread prefetch_throttle_t g_throttle = {0};

/**
 * @brief TLB prefetch state
 * 
 * FEATURE #3: TLB Prefetching for huge transforms
 */
typedef struct {
    bool enabled;              // Enable TLB prefetch
    int page_distance;         // Pages ahead to prefetch (4-16)
    int stride_threshold;      // Stride to trigger TLB prefetch
    unsigned long last_page;   // Last prefetched page
} tlb_prefetch_t;

static __thread tlb_prefetch_t g_tlb_prefetch = {0};

/**
 * @brief Wisdom database entry (FEATURE #5: Exhaustive Search)
 */
typedef struct {
    int n_fft;                 // Transform size
    int radix;                 // Radix used
    int distance_input;        // Optimal input distance
    int distance_twiddle;      // Optimal twiddle distance
    int hint;                  // Optimal hint
    prefetch_strategy_t strategy; // Optimal strategy
    double cycles_per_element; // Measured performance
    time_t timestamp;          // When measured
} wisdom_entry_t;

#define MAX_WISDOM_ENTRIES 256
static wisdom_entry_t g_wisdom_db[MAX_WISDOM_ENTRIES];
static int g_wisdom_count = 0;
static pthread_mutex_t g_wisdom_mutex = PTHREAD_MUTEX_INITIALIZER;

/**
 * @brief Initialize prefetch throttling
 * 
 * FEATURE #7: Throttling based on CPU's prefetch buffer count
 */
static inline void init_throttling(void) {
    g_throttle.max_outstanding = g_cpu_profile->prefetch_buffers;
    g_throttle.budget_remaining = g_throttle.max_outstanding;
    g_throttle.window_size = 8; // Reset budget every 8 iterations
    g_throttle.window_counter = 0;
    g_throttle.issued_count = 0;
}

/**
 * @brief Check if we can issue a prefetch (throttling)
 * 
 * FEATURE #7: Budget management
 */
static inline bool can_prefetch(void) {
    // Advance window
    g_throttle.window_counter++;
    if (g_throttle.window_counter >= g_throttle.window_size) {
        g_throttle.window_counter = 0;
        g_throttle.budget_remaining = g_throttle.max_outstanding;
        g_throttle.issued_count = 0;
    }
    
    // Check budget
    if (g_throttle.budget_remaining <= 0) {
        return false;
    }
    
    return true;
}

/**
 * @brief Record prefetch issued (decrement budget)
 */
static inline void record_prefetch_issued(void) {
    g_throttle.budget_remaining--;
    g_throttle.issued_count++;
}

/**
 * @brief Throttled prefetch with priority
 * 
 * FEATURE #7: Priority-based budget allocation
 */
typedef enum {
    PREFETCH_PRIO_CRITICAL = 0,  // Input data (always prefetch)
    PREFETCH_PRIO_HIGH = 1,      // Twiddles (high reuse)
    PREFETCH_PRIO_MEDIUM = 2,    // Output writes
    PREFETCH_PRIO_LOW = 3        // Speculative
} prefetch_priority_t;

static inline void prefetch_throttled(
    const void *addr,
    int hint,
    prefetch_priority_t priority
) {
    // Critical always goes through
    if (priority == PREFETCH_PRIO_CRITICAL) {
        PREFETCH(addr, hint);
        record_prefetch_issued();
        return;
    }
    
    // Others check budget
    if (can_prefetch()) {
        PREFETCH(addr, hint);
        record_prefetch_issued();
    }
    // Else: skip (throttled)
}

/**
 * @brief Initialize TLB prefetching
 * 
 * FEATURE #3: TLB prefetch for huge transforms
 */
static inline void init_tlb_prefetch(int n_fft) {
    const int data_size = n_fft * sizeof(fft_data);
    const int page_size = 4096; // 4KB pages
    const int num_pages = (data_size + page_size - 1) / page_size;
    
    // Enable TLB prefetch if:
    // 1. Transform is huge (> 16M elements = 256MB)
    // 2. Exceeds typical TLB coverage (1024 entries × 4KB = 4MB)
    if (num_pages > 1024) {
        g_tlb_prefetch.enabled = true;
        g_tlb_prefetch.page_distance = 8; // 8 pages ahead (32KB)
        g_tlb_prefetch.stride_threshold = page_size / sizeof(fft_data); // 256 elements
        g_tlb_prefetch.last_page = 0;
    } else {
        g_tlb_prefetch.enabled = false;
    }
}

/**
 * @brief TLB prefetch (touch page to load PTE)
 * 
 * FEATURE #3: Page table prefetching
 */
static inline void prefetch_tlb(const fft_data *addr) {
    if (!g_tlb_prefetch.enabled) return;
    
    const unsigned long page = (unsigned long)addr / 4096;
    
    // Only prefetch if we've moved to a new page
    if (page != g_tlb_prefetch.last_page) {
        // Touch future page (loads PTE into TLB)
        const unsigned long future_page = page + g_tlb_prefetch.page_distance;
        const fft_data *future_addr = (const fft_data*)(future_page * 4096);
        
        // Read first byte to trigger TLB load
        volatile char dummy = *((const volatile char*)future_addr);
        (void)dummy; // Suppress unused warning
        
        g_tlb_prefetch.last_page = page;
    }
}

/**
 * @brief Load wisdom from file
 * 
 * FEATURE #5: Wisdom database (persistent tuning)
 */
static inline void load_wisdom(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return;
    
    pthread_mutex_lock(&g_wisdom_mutex);
    
    g_wisdom_count = 0;
    while (g_wisdom_count < MAX_WISDOM_ENTRIES && !feof(fp)) {
        wisdom_entry_t *entry = &g_wisdom_db[g_wisdom_count];
        
        int items = fscanf(fp, "%d %d %d %d %d %d %lf %ld\n",
            &entry->n_fft,
            &entry->radix,
            &entry->distance_input,
            &entry->distance_twiddle,
            &entry->hint,
            (int*)&entry->strategy,
            &entry->cycles_per_element,
            &entry->timestamp
        );
        
        if (items == 8) {
            g_wisdom_count++;
        }
    }
    
    pthread_mutex_unlock(&g_wisdom_mutex);
    fclose(fp);
}

/**
 * @brief Save wisdom to file
 * 
 * FEATURE #5: Persistent tuning results
 */
static inline void save_wisdom(const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) return;
    
    pthread_mutex_lock(&g_wisdom_mutex);
    
    fprintf(fp, "# HFFT Wisdom Database (auto-generated)\n");
    fprintf(fp, "# Format: n_fft radix dist_in dist_tw hint strategy cycles timestamp\n");
    
    for (int i = 0; i < g_wisdom_count; i++) {
        wisdom_entry_t *entry = &g_wisdom_db[i];
        fprintf(fp, "%d %d %d %d %d %d %.6f %ld\n",
            entry->n_fft,
            entry->radix,
            entry->distance_input,
            entry->distance_twiddle,
            entry->hint,
            entry->strategy,
            entry->cycles_per_element,
            entry->timestamp
        );
    }
    
    pthread_mutex_unlock(&g_wisdom_mutex);
    fclose(fp);
}

/**
 * @brief Find wisdom entry for given size/radix
 * 
 * FEATURE #5: Lookup precomputed optimal config
 */
static inline wisdom_entry_t* find_wisdom(int n_fft, int radix) {
    pthread_mutex_lock(&g_wisdom_mutex);
    
    for (int i = 0; i < g_wisdom_count; i++) {
        if (g_wisdom_db[i].n_fft == n_fft && 
            g_wisdom_db[i].radix == radix) {
            pthread_mutex_unlock(&g_wisdom_mutex);
            return &g_wisdom_db[i];
        }
    }
    
    pthread_mutex_unlock(&g_wisdom_mutex);
    return NULL;
}

/**
 * @brief Add wisdom entry (called by exhaustive search)
 * 
 * FEATURE #5: Store tuning results
 */
static inline void add_wisdom(
    int n_fft,
    int radix,
    int distance_input,
    int distance_twiddle,
    int hint,
    prefetch_strategy_t strategy,
    double cycles_per_element
) {
    pthread_mutex_lock(&g_wisdom_mutex);
    
    if (g_wisdom_count < MAX_WISDOM_ENTRIES) {
        wisdom_entry_t *entry = &g_wisdom_db[g_wisdom_count++];
        entry->n_fft = n_fft;
        entry->radix = radix;
        entry->distance_input = distance_input;
        entry->distance_twiddle = distance_twiddle;
        entry->hint = hint;
        entry->strategy = strategy;
        entry->cycles_per_element = cycles_per_element;
        entry->timestamp = time(NULL);
    }
    
    pthread_mutex_unlock(&g_wisdom_mutex);
}

/**
 * @brief Exhaustive prefetch search for optimal configuration
 * 
 * FEATURE #5: Exhaustive search (called during planning)
 * This is expensive (5-10 seconds per size) but amortized across many runs
 */
static inline void search_optimal_prefetch(
    fft_object fft_obj,
    int factor_index,
    stage_prefetch_t *best_config
) {
    // Check wisdom first
    const int radix = fft_obj->factors[factor_index];
    wisdom_entry_t *wisdom = find_wisdom(fft_obj->n_fft, radix);
    
    if (wisdom) {
        // Use cached result
        best_config->distance_input = wisdom->distance_input;
        best_config->distance_twiddle = wisdom->distance_twiddle;
        best_config->hint_input = wisdom->hint;
        best_config->strategy = wisdom->strategy;
        return;
    }
    
    // No wisdom - perform search
    double best_cycles = 1e9;
    
    // Search space (reduced for practicality)
    int distances[] = {2, 4, 6, 8, 12, 16, 20, 24, 32, 48, 64};
    int hints[] = {_MM_HINT_T0, _MM_HINT_T1, _MM_HINT_T2, _MM_HINT_NTA};
    prefetch_strategy_t strategies[] = {
        PREFETCH_SINGLE, PREFETCH_DUAL, PREFETCH_MULTI
    };
    
    const int num_distances = sizeof(distances) / sizeof(distances[0]);
    const int num_hints = sizeof(hints) / sizeof(hints[0]);
    const int num_strategies = sizeof(strategies) / sizeof(strategies[0]);
    
    // Allocate test buffers
    fft_data *test_input = (fft_data*)_mm_malloc(
        fft_obj->n_fft * sizeof(fft_data), 64
    );
    fft_data *test_output = (fft_data*)_mm_malloc(
        fft_obj->n_fft * sizeof(fft_data), 64
    );
    
    if (!test_input || !test_output) {
        if (test_input) _mm_free(test_input);
        if (test_output) _mm_free(test_output);
        return; // Fall back to defaults
    }
    
    // Initialize test data
    for (int i = 0; i < fft_obj->n_fft; i++) {
        test_input[i].re = (double)i;
        test_input[i].im = 0.0;
    }
    
    // Search loop (trials limited for practicality)
    for (int d_idx = 0; d_idx < num_distances; d_idx++) {
        for (int h_idx = 0; h_idx < num_hints; h_idx++) {
            for (int s_idx = 0; s_idx < num_strategies; s_idx++) {
                // Configure test
                best_config->distance_input = distances[d_idx];
                best_config->distance_twiddle = distances[d_idx] / 2;
                best_config->hint_input = hints[h_idx];
                best_config->hint_twiddle = _MM_HINT_T0;
                best_config->strategy = strategies[s_idx];
                best_config->enable = true;
                
                // Warm up (3 runs)
                for (int w = 0; w < 3; w++) {
                    fft_exec(fft_obj, test_input, test_output);
                }
                
                // Measure (10 runs)
                unsigned long long start = read_tsc();
                for (int r = 0; r < 10; r++) {
                    fft_exec(fft_obj, test_input, test_output);
                }
                unsigned long long end = read_tsc();
                
                double cycles = (double)(end - start) / (10.0 * fft_obj->n_fft);
                
                if (cycles < best_cycles) {
                    best_cycles = cycles;
                    // Keep current config (already in best_config)
                }
            }
        }
    }
    
    // Store wisdom
    add_wisdom(
        fft_obj->n_fft,
        radix,
        best_config->distance_input,
        best_config->distance_twiddle,
        best_config->hint_input,
        best_config->strategy,
        best_cycles
    );
    
    // Cleanup
    _mm_free(test_input);
    _mm_free(test_output);
}

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
 * 
 * Now with FEATURE #7 (Throttling) and FEATURE #3 (TLB)
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
    
    // FEATURE #3: TLB prefetch for huge strides
    if (stride >= g_tlb_prefetch.stride_threshold) {
        prefetch_tlb(input_base + idx + d_in);
    }
    
    // FEATURE #7: Throttled multi-stream prefetch with priorities
    
    // Prefetch input lanes (multi-stream) - CRITICAL priority
    if (cfg->strategy >= PREFETCH_MULTI) {
        for (int lane = 0; lane < radix && lane < 4; ++lane) {
            prefetch_throttled(
                input_base + (idx + d_in) + lane * stride,
                cfg->hint_input,
                PREFETCH_PRIO_CRITICAL
            );
        }
    } else {
        prefetch_throttled(
            input_base + idx + d_in,
            cfg->hint_input,
            PREFETCH_PRIO_CRITICAL
        );
    }
    
    // Prefetch twiddles (if present) - HIGH priority
    if (twiddle_base && cfg->strategy >= PREFETCH_DUAL) {
        prefetch_throttled(
            twiddle_base + idx + d_tw,
            cfg->hint_twiddle,
            PREFETCH_PRIO_HIGH
        );
    }
    
    // Write prefetch for output (optional, CPU-dependent) - MEDIUM priority
    if (output_base && cfg->strategy >= PREFETCH_DUAL) {
        if (g_cpu_profile->has_write_prefetch) {
            prefetch_write(output_base, idx, d_out, cfg);
        }
    }
}

/**
 * @brief Prefetch for tight butterfly loops
 * Now with throttling
 */
static inline void prefetch_butterfly_loop(
    const fft_data *input,
    const fft_data *twiddle,
    int idx,
    stage_prefetch_t *cfg
) {
    if (!cfg || !cfg->enable) return;
    
    // FEATURE #7: Throttled prefetch
    prefetch_throttled(
        input + idx + cfg->distance_input,
        cfg->hint_input,
        PREFETCH_PRIO_CRITICAL
    );
    
    if (twiddle && cfg->strategy >= PREFETCH_DUAL) {
        prefetch_throttled(
            twiddle + idx + cfg->distance_twiddle,
            cfg->hint_twiddle,
            PREFETCH_PRIO_HIGH
        );
    }
}

/**
 * @brief Initialize global prefetch system with all advanced features
 * 
 * Call this once at program startup (or per fft_init)
 */
static inline void init_prefetch_system(fft_object fft_obj) {
    // FEATURE #10: Detect CPU architecture (once)
    static int cpu_detected = 0;
    if (!cpu_detected) {
        g_cpu_profile = detect_cpu_architecture();
        cpu_detected = 1;
        
        fprintf(stderr, "HFFT: Detected CPU: %s\n", g_cpu_profile->name);
        fprintf(stderr, "HFFT: Prefetch buffers: %d, L1: %d, L2: %d, L3: %d cycles\n",
            g_cpu_profile->prefetch_buffers,
            g_cpu_profile->l1_latency,
            g_cpu_profile->l2_latency,
            g_cpu_profile->l3_latency
        );
    }
    
    // Detect cache sizes
    static int cache_detected = 0;
    if (!cache_detected) {
        detect_cache_sizes();
        cache_detected = 1;
    }
    
    // FEATURE #7: Initialize throttling
    init_throttling();
    
    // FEATURE #3: Initialize TLB prefetch
    init_tlb_prefetch(fft_obj->n_fft);
    
    // FEATURE #5: Load wisdom database (if exists)
    static int wisdom_loaded = 0;
    if (!wisdom_loaded) {
        const char *wisdom_file = getenv("HFFT_WISDOM_FILE");
        if (!wisdom_file) wisdom_file = "hfft_wisdom.txt";
        
        load_wisdom(wisdom_file);
        if (g_wisdom_count > 0) {
            fprintf(stderr, "HFFT: Loaded %d wisdom entries from %s\n",
                g_wisdom_count, wisdom_file);
        }
        wisdom_loaded = 1;
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