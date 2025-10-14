//==============================================================================
// ADVANCED PREFETCH SYSTEM - FFTW-Inspired Implementation
// Clean, properly ordered implementation
//==============================================================================
/*
┌─────────────────────────────────┐
│  prefetch_strategy.c            │
│  ├─ init_prefetch_system()      │
│  ├─ prefetch_input()            │
│  ├─ prefetch_twiddle()          │
│  └─ get_stage_config()          │
└─────────────────────────────────┘
         │
         │ Calls (optional)
         ▼
┌─────────────────────────────────┐
│  throttle_enhanced.c (NEW)      │
│  ├─ prefetch_throttled_enhanced()│
│  └─ Token bucket logic          │
└─────────────────────────────────┘

         │
         │ Calls (optional)
         ▼
┌─────────────────────────────────┐
│  adaptive_tuning.c (NEW)        │
│  ├─ profile_fft_start/end()     │
│  ├─ get_tuned_distance()        │
│  └─ EWMA/per-stage tuning       │
└─────────────────────────────────┘
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <stdint.h>

#include "highspeedFFT.h"
#include "prefetch_strategy.h"

//==============================================================================
// INTERNAL TYPE DEFINITIONS (not in header)
//==============================================================================

/**
 * @brief Prefetch throttling state (per-thread)
 */
typedef struct {
    int issued_count;
    int budget_remaining;
    int window_size;
    int window_counter;
    int max_outstanding;
} prefetch_throttle_t;

/**
 * @brief TLB prefetch state
 */
typedef struct {
    bool enabled;
    int page_distance;
    int stride_threshold;
    uintptr_t last_page;
} tlb_prefetch_t;

/**
 * @brief Prefetch priority levels
 */
typedef enum {
    PREFETCH_PRIO_CRITICAL = 0,
    PREFETCH_PRIO_HIGH = 1,
    PREFETCH_PRIO_MEDIUM = 2,
    PREFETCH_PRIO_LOW = 3
} prefetch_priority_t;

/**
 * @brief Runtime profiling for adaptive tuning
 */
typedef struct {
    unsigned long long total_cycles;
    unsigned long long total_calls;
    int current_distance;
    int best_distance;
    double best_throughput;
    int tuning_phase;
    int tuning_iterations;
} prefetch_profile_t;

//==============================================================================
// GLOBAL STATE
//==============================================================================

// Global prefetch configuration
static prefetch_config_t g_prefetch_config = {
    .stages = NULL,
    .num_stages = 0,
    .l1_size = 32 * 1024,
    .l2_size = 256 * 1024,
    .l3_size = 8 * 1024 * 1024,
    .cache_line_size = 64,
    .enable_runtime_tuning = false
};

// CPU profiles database
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

// Thread-local state
static __thread prefetch_throttle_t g_throttle = {0};
static __thread tlb_prefetch_t g_tlb_prefetch = {0};

// Wisdom database
#define MAX_WISDOM_ENTRIES 256
static wisdom_entry_t g_wisdom_db[MAX_WISDOM_ENTRIES];
static int g_wisdom_count = 0;
static pthread_mutex_t g_wisdom_mutex = PTHREAD_MUTEX_INITIALIZER;

// Runtime profiling
static prefetch_profile_t g_prefetch_profile = {0};

//==============================================================================
// INTERNAL HELPER FUNCTIONS
//==============================================================================

/**
 * @brief Read CPU timestamp counter
 */
static inline unsigned long long read_tsc(void) {
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)hi << 32) | lo;
#else
    return 0;
#endif
}

/**
 * @brief Detect CPU cache sizes using CPUID
 */
void detect_cache_sizes(void) {
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int eax, ebx, ecx, edx;
    
    __asm__ __volatile__ (
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(0x0)
    );
    
    if (eax >= 0x4) {
        for (int i = 0; i < 10; ++i) {
            __asm__ __volatile__ (
                "cpuid"
                : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                : "a"(0x4), "c"(i)
            );
            
            int cache_type = eax & 0x1F;
            if (cache_type == 0) break;
            
            int cache_level = (eax >> 5) & 0x7;
            int cache_size = ((ebx >> 22) + 1) * 
                           ((ebx >> 12 & 0x3FF) + 1) * 
                           ((ebx & 0xFFF) + 1) * 
                           (ecx + 1);
            
            if (cache_type == 1 || cache_type == 3) {
                if (cache_level == 1) g_prefetch_config.l1_size = cache_size;
                else if (cache_level == 2) g_prefetch_config.l2_size = cache_size;
                else if (cache_level == 3) g_prefetch_config.l3_size = cache_size;
            }
        }
    }
#endif
}

/**
 * @brief Detect CPU architecture
 */
static inline const cpu_profile_t* detect_cpu_architecture(void) {
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int eax, ebx, ecx, edx;
    char vendor[13] = {0};
    char brand[49] = {0};
    
    __asm__ __volatile__ (
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(0)
    );
    memcpy(vendor, &ebx, 4);
    memcpy(vendor + 4, &edx, 4);
    memcpy(vendor + 8, &ecx, 4);
    
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
    
    if (strstr(vendor, "GenuineIntel")) {
        if (strstr(brand, "Sapphire Rapids") || strstr(brand, "Emerald Rapids"))
            return &cpu_profiles[2];
        else if (strstr(brand, "Ice Lake") || strstr(brand, "Tiger Lake"))
            return &cpu_profiles[1];
        else if (strstr(brand, "Skylake") || strstr(brand, "Cascade Lake"))
            return &cpu_profiles[0];
    }
    else if (strstr(vendor, "AuthenticAMD")) {
        if (strstr(brand, "Zen 4") || strstr(brand, "Ryzen 7000"))
            return &cpu_profiles[5];
        else if (strstr(brand, "Zen 3") || strstr(brand, "Ryzen 5000"))
            return &cpu_profiles[4];
        else if (strstr(brand, "Zen 2") || strstr(brand, "Ryzen 3000"))
            return &cpu_profiles[3];
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    #if defined(__APPLE__)
        return &cpu_profiles[7]; // M1/M2
    #else
        return &cpu_profiles[6]; // Neoverse V1
    #endif
#endif
    
    return &cpu_profiles[9]; // Generic fallback
}

/**
 * @brief Compute working set size for a stage
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
 * @brief Compute optimal prefetch distance
 */
static inline int compute_stage_prefetch_distance(int working_set, int stride) {
    const int l1 = g_prefetch_config.l1_size;
    const int l2 = g_prefetch_config.l2_size;
    const int l3 = g_prefetch_config.l3_size;
    
    int base_distance;
    if (working_set < l1 / 2) {
        base_distance = 2;
    } else if (working_set < l1) {
        base_distance = 4;
    } else if (working_set < l2 / 2) {
        base_distance = 8;
    } else if (working_set < l2) {
        base_distance = 12;
    } else if (working_set < l3 / 2) {
        base_distance = 16;
    } else if (working_set < l3) {
        base_distance = 24;
    } else {
        base_distance = 32;
    }
    
    if (stride > 1) {
        base_distance += (stride / 2);
    }
    
    return base_distance;
}

/**
 * @brief Compute optimal hint
 */
static inline int compute_stage_hint(int working_set) {
    const int l1 = g_prefetch_config.l1_size;
    const int l2 = g_prefetch_config.l2_size;
    const int l3 = g_prefetch_config.l3_size;
    
    if (working_set < l1) return _MM_HINT_T0;
    else if (working_set < l2) return _MM_HINT_T1;
    else if (working_set < l3) return _MM_HINT_T2;
    else return _MM_HINT_NTA;
}

/**
 * @brief Initialize throttling
 */
static inline void init_throttling(void) {
    g_throttle.max_outstanding = g_cpu_profile->prefetch_buffers;
    g_throttle.budget_remaining = g_throttle.max_outstanding;
    g_throttle.window_size = 8;
    g_throttle.window_counter = 0;
    g_throttle.issued_count = 0;
}

/**
 * @brief Initialize TLB prefetching
 */
static inline void init_tlb_prefetch(int n_fft) {
    const int data_size = n_fft * sizeof(fft_data);
    const int page_size = 4096;
    const int num_pages = (data_size + page_size - 1) / page_size;
    
    if (num_pages > 1024) {
        g_tlb_prefetch.enabled = true;
        g_tlb_prefetch.page_distance = 8;
        g_tlb_prefetch.stride_threshold = page_size / sizeof(fft_data);
        g_tlb_prefetch.last_page = 0;
    } else {
        g_tlb_prefetch.enabled = false;
    }
}

/**
 * @brief Check if we can issue a prefetch
 */
static inline bool can_prefetch(void) {
    g_throttle.window_counter++;
    if (g_throttle.window_counter >= g_throttle.window_size) {
        g_throttle.window_counter = 0;
        g_throttle.budget_remaining = g_throttle.max_outstanding;
        g_throttle.issued_count = 0;
    }
    
    return (g_throttle.budget_remaining > 0);
}

/**
 * @brief Record prefetch issued
 */
static inline void record_prefetch_issued(void) {
    g_throttle.budget_remaining--;
    g_throttle.issued_count++;
}

/**
 * @brief Prefetch dispatch based on hint value
 * 
 * _mm_prefetch requires compile-time constant hint, so we need to dispatch
 * based on the runtime hint value to the appropriate compile-time constant.
 */
static inline void do_prefetch(const void *addr, int hint) {
    switch (hint) {
        case _MM_HINT_T0:
            _mm_prefetch((const char *)addr, _MM_HINT_T0);
            break;
        case _MM_HINT_T1:
            _mm_prefetch((const char *)addr, _MM_HINT_T1);
            break;
        case _MM_HINT_T2:
            _mm_prefetch((const char *)addr, _MM_HINT_T2);
            break;
        case _MM_HINT_NTA:
            _mm_prefetch((const char *)addr, _MM_HINT_NTA);
            break;
        default:
            _mm_prefetch((const char *)addr, _MM_HINT_T0);
            break;
    }
}

/**
 * @brief Throttled prefetch with priority
 */
static inline void prefetch_throttled(
    const void *addr,
    int hint,
    prefetch_priority_t priority
) {
    if (priority == PREFETCH_PRIO_CRITICAL) {
        do_prefetch(addr, hint);
        record_prefetch_issued();
        return;
    }
    
    if (can_prefetch()) {
        do_prefetch(addr, hint);
        record_prefetch_issued();
    }
}

/**
 * @brief TLB prefetch
 */
static inline void prefetch_tlb(const fft_data *addr) {
    if (!g_tlb_prefetch.enabled) return;
    
    const uintptr_t page = (uintptr_t)addr / 4096;
    
    if (page != g_tlb_prefetch.last_page) {
        const uintptr_t future_page = page + g_tlb_prefetch.page_distance;
        const fft_data *future_addr = (const fft_data*)(future_page * 4096);
        
        volatile char dummy = *((const volatile char*)future_addr);
        (void)dummy;
        
        g_tlb_prefetch.last_page = page;
    }
}

/**
 * @brief Write prefetch
 */
static inline void prefetch_write(
    fft_data *output,
    int idx,
    int distance,
    stage_prefetch_t *cfg
) {
    if (!cfg || !cfg->enable) return;
    
#if defined(__PRFCHW__) || defined(__3dNOW__)
    __builtin_prefetch(output + idx + distance, 1, 0);
#else
    do_prefetch(output + idx + distance, _MM_HINT_NTA);
#endif
}

//==============================================================================
// PUBLIC API IMPLEMENTATION
//==============================================================================

void init_prefetch_system(fft_object fft_obj) {
    // Detect CPU architecture (once)
    static int cpu_detected = 0;
    if (!cpu_detected) {
        g_cpu_profile = detect_cpu_architecture();
        cpu_detected = 1;
        
        fprintf(stderr, "HFFT: Detected CPU: %s\n", g_cpu_profile->name);
    }
    
    // Detect cache sizes (once)
    static int cache_detected = 0;
    if (!cache_detected) {
        detect_cache_sizes();
        cache_detected = 1;
    }
    
    // Initialize throttling
    init_throttling();
    
    // Initialize TLB prefetch
    init_tlb_prefetch(fft_obj->n_fft);
    
    // Load wisdom database (once)
    static int wisdom_loaded = 0;
    if (!wisdom_loaded) {
        const char *wisdom_file = getenv("HFFT_WISDOM_FILE");
        if (!wisdom_file) wisdom_file = "hfft_wisdom.txt";
        load_wisdom(wisdom_file);
        wisdom_loaded = 1;
    }
    
    // Initialize per-stage configs
    const int n_fft = fft_obj->n_fft;
    const int num_factors = fft_obj->lf;
    const int *factors = fft_obj->factors;
    
    if (g_prefetch_config.stages) {
        free(g_prefetch_config.stages);
    }
    
    g_prefetch_config.stages = (stage_prefetch_t *)malloc(
        num_factors * sizeof(stage_prefetch_t)
    );
    g_prefetch_config.num_stages = num_factors;
    
    if (!g_prefetch_config.stages) return;
    
    int stride = 1;
    for (int stage = 0; stage < num_factors; ++stage) {
        const int radix = factors[stage];
        const int working_set = compute_stage_working_set(n_fft, stage, factors, num_factors);
        
        stage_prefetch_t *cfg = &g_prefetch_config.stages[stage];
        
        // Try wisdom first
        wisdom_entry_t *wisdom = find_wisdom(n_fft, radix);
        
        if (wisdom) {
            cfg->distance_input = wisdom->distance_input;
            cfg->distance_twiddle = wisdom->distance_twiddle;
            cfg->hint_input = wisdom->hint;
            cfg->hint_twiddle = _MM_HINT_T0;
            cfg->strategy = wisdom->strategy;
            cfg->enable = true;
        } else {
            // Use heuristics
            if (working_set < 1024) {
                cfg->strategy = PREFETCH_NONE;
            } else if (radix <= 4 && working_set < g_prefetch_config.l1_size) {
                cfg->strategy = PREFETCH_SINGLE;
            } else if (radix <= 8) {
                cfg->strategy = PREFETCH_DUAL;
            } else {
                cfg->strategy = PREFETCH_MULTI;
            }
            
            cfg->distance_input = compute_stage_prefetch_distance(working_set, stride);
            cfg->distance_output = cfg->distance_input;
            cfg->distance_twiddle = cfg->distance_input / 2;
            
            cfg->hint_input = compute_stage_hint(working_set);
            cfg->hint_output = cfg->hint_input;
            cfg->hint_twiddle = _MM_HINT_T0;
            
            if (g_cpu_profile->has_strong_hwpf && working_set < g_prefetch_config.l1_size) {
                cfg->enable = false;
            } else {
                cfg->enable = (cfg->strategy != PREFETCH_NONE);
            }
        }
        
        if (radix >= 16 && working_set > g_prefetch_config.l2_size) {
            cfg->block_size = g_prefetch_config.l2_size / (radix * sizeof(fft_data));
        } else {
            cfg->block_size = 0;
        }
        
        stride *= radix;
    }
}

void cleanup_prefetch_system(void) {
    if (g_prefetch_config.stages) {
        free(g_prefetch_config.stages);
        g_prefetch_config.stages = NULL;
        g_prefetch_config.num_stages = 0;
    }
}

stage_prefetch_t* get_stage_config(int factor_index) {
    if (factor_index < 0 || factor_index >= g_prefetch_config.num_stages) {
        return NULL;
    }
    return &g_prefetch_config.stages[factor_index];
}

void prefetch_input(const fft_data *input, int idx, stage_prefetch_t *cfg) {
    if (!cfg || !cfg->enable) return;
    do_prefetch(input + idx + cfg->distance_input, cfg->hint_input);
}

void prefetch_twiddle(const fft_data *twiddle, int idx, stage_prefetch_t *cfg) {
    if (!cfg || !cfg->enable) return;
    do_prefetch(twiddle + idx + cfg->distance_twiddle, cfg->hint_twiddle);
}

void prefetch_stage_recursive(
    const fft_data *input_base,
    const fft_data *twiddle_base,
    int idx,
    int stride,
    int radix,
    stage_prefetch_t *cfg
) {
    if (!cfg || !cfg->enable) return;
    
    const int d_in = cfg->distance_input;
    const int d_tw = cfg->distance_twiddle;
    
    if (stride >= g_tlb_prefetch.stride_threshold) {
        prefetch_tlb(input_base + idx + d_in);
    }
    
    if (cfg->strategy >= PREFETCH_MULTI && radix > 4) {
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
    
    if (twiddle_base && cfg->strategy >= PREFETCH_DUAL) {
        prefetch_throttled(
            twiddle_base + idx + d_tw,
            cfg->hint_twiddle,
            PREFETCH_PRIO_HIGH
        );
    }
}

void prefetch_butterfly_loop(
    const fft_data *input,
    const fft_data *twiddle,
    int idx,
    stage_prefetch_t *cfg
) {
    if (!cfg || !cfg->enable) return;
    
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

// Wisdom database functions
void load_wisdom(const char *filename) {
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

void save_wisdom(const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) return;
    
    pthread_mutex_lock(&g_wisdom_mutex);
    
    fprintf(fp, "# HFFT Wisdom Database\n");
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

wisdom_entry_t* find_wisdom(int n_fft, int radix) {
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

void add_wisdom(
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

void search_optimal_prefetch(
    fft_object fft_obj,
    int factor_index,
    stage_prefetch_t *best_config
) {
    // Check wisdom first
    const int radix = fft_obj->factors[factor_index];
    wisdom_entry_t *wisdom = find_wisdom(fft_obj->n_fft, radix);
    
    if (wisdom) {
        best_config->distance_input = wisdom->distance_input;
        best_config->distance_twiddle = wisdom->distance_twiddle;
        best_config->hint_input = wisdom->hint;
        best_config->strategy = wisdom->strategy;
        return;
    }
    
    // No wisdom - use heuristics
    // Full exhaustive search would go here
    const int n_fft = fft_obj->n_fft;
    const int working_set = compute_stage_working_set(
        n_fft, factor_index, fft_obj->factors, fft_obj->lf
    );
    
    if (working_set < 1024) {
        best_config->strategy = PREFETCH_NONE;
    } else if (radix <= 4 && working_set < g_prefetch_config.l1_size) {
        best_config->strategy = PREFETCH_SINGLE;
    } else if (radix <= 8) {
        best_config->strategy = PREFETCH_DUAL;
    } else {
        best_config->strategy = PREFETCH_MULTI;
    }
    
    best_config->distance_input = compute_stage_prefetch_distance(working_set, 1);
    best_config->distance_twiddle = best_config->distance_input / 2;
    best_config->hint_input = compute_stage_hint(working_set);
    best_config->hint_twiddle = _MM_HINT_T0;
    best_config->enable = (best_config->strategy != PREFETCH_NONE);
}

const prefetch_config_t* get_prefetch_config(void) {
    return &g_prefetch_config;
}

const cpu_profile_t* get_cpu_profile(void) {
    return g_cpu_profile;
}

void set_prefetch_enable(bool enable) {
    for (int i = 0; i < g_prefetch_config.num_stages; i++) {
        g_prefetch_config.stages[i].enable = enable;
    }
}

//==============================================================================
// RUNTIME PROFILING AND ADAPTIVE TUNING
//==============================================================================

void profile_start(void) {
    if (!g_prefetch_config.enable_runtime_tuning) return;
    g_prefetch_profile.total_cycles = read_tsc();
}

void profile_end(int n_elements) {
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

//==============================================================================
// ADVANCED PREFETCH OPERATIONS
//==============================================================================

int adjust_distance_for_unroll(int base_distance, int unroll_factor) {
    // Rule of thumb: prefetch_distance ≈ base_distance * sqrt(unroll_factor)
    if (unroll_factor <= 2) return base_distance;
    else if (unroll_factor <= 4) return base_distance + 2;
    else if (unroll_factor <= 8) return base_distance + 4;
    else return base_distance + 8; // 16x+ unrolling
}

void prefetch_strided(
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
        do_prefetch(ptr, hint);
        
        // For very large strides, prefetch intermediate cache lines
        if (cl_per_stride > 2) {
            for (int cl = 1; cl < cl_per_stride && cl < 4; ++cl) {
                do_prefetch((char*)ptr + cl * 64, hint);
            }
        }
    }
}

void prefetch_blocked(
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
        do_prefetch(base + next_block + i, hint);
    }
}

void prefetch_radix_group(
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
        do_prefetch(base + next_group + lane * group_size, hint);
    }
}

//==============================================================================
// DEBUG SUPPORT
//==============================================================================

#ifdef FFT_DEBUG_PREFETCH

void print_prefetch_config(void) {
    printf("=== Prefetch Configuration ===\n");
    printf("CPU Profile: %s\n", g_cpu_profile->name);
    printf("L1: %d bytes, L2: %d bytes, L3: %d bytes\n",
        g_prefetch_config.l1_size,
        g_prefetch_config.l2_size,
        g_prefetch_config.l3_size);
    printf("Cache line: %d bytes\n", g_prefetch_config.cache_line_size);
    printf("Number of stages: %d\n", g_prefetch_config.num_stages);
    printf("Runtime tuning: %s\n", 
        g_prefetch_config.enable_runtime_tuning ? "enabled" : "disabled");
    printf("==============================\n");
}

void print_stage_config(int stage_idx) {
    if (stage_idx < 0 || stage_idx >= g_prefetch_config.num_stages) {
        printf("Invalid stage index: %d\n", stage_idx);
        return;
    }
    
    stage_prefetch_t *cfg = &g_prefetch_config.stages[stage_idx];
    
    printf("=== Stage %d Configuration ===\n", stage_idx);
    printf("Enabled: %s\n", cfg->enable ? "yes" : "no");
    printf("Strategy: ");
    switch (cfg->strategy) {
        case PREFETCH_NONE: printf("NONE\n"); break;
        case PREFETCH_SINGLE: printf("SINGLE\n"); break;
        case PREFETCH_DUAL: printf("DUAL\n"); break;
        case PREFETCH_MULTI: printf("MULTI\n"); break;
        case PREFETCH_STRIDED: printf("STRIDED\n"); break;
        default: printf("UNKNOWN\n"); break;
    }
    printf("Distance input: %d\n", cfg->distance_input);
    printf("Distance output: %d\n", cfg->distance_output);
    printf("Distance twiddle: %d\n", cfg->distance_twiddle);
    printf("Hint input: 0x%x\n", cfg->hint_input);
    printf("Hint output: 0x%x\n", cfg->hint_output);
    printf("Hint twiddle: 0x%x\n", cfg->hint_twiddle);
    printf("Block size: %d\n", cfg->block_size);
    printf("=============================\n");
}

#endif // FFT_DEBUG_PREFETCH
