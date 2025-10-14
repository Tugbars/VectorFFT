//==============================================================================
// ADVANCED PREFETCH SYSTEM - FFTW-Inspired Implementation
// Clean, properly ordered implementation with OPTIONAL enhanced modules
//==============================================================================
/**
 * @file prefetch_strategy.c
 * @brief Advanced cache prefetching system for high-performance FFT operations
 * 
 * This module implements a sophisticated prefetch system inspired by FFTW,
 * with CPU-specific optimizations, adaptive tuning, and wisdom database support.
 * 
 * Architecture:
 * - Base system: CPU detection, cache hierarchy analysis, per-stage configuration
 * - Optional: Enhanced throttling with token bucket algorithm (compile with -DHFFT_USE_ENHANCED_THROTTLE)
 * - Optional: Adaptive distance tuning with EWMA (compile with -DHFFT_USE_ADAPTIVE_TUNING)
 * 
 * Compilation modes:
 *   Default: gcc prefetch_strategy.c
 *            Uses built-in simple throttling and profiling
 *
 *   Enhanced: gcc -DHFFT_USE_ENHANCED_THROTTLE prefetch_strategy.c throttle_enhanced.c
 *             Links with advanced token bucket throttling
 *
 *   Adaptive: gcc -DHFFT_USE_ADAPTIVE_TUNING prefetch_strategy.c adaptive_tuning.c
 *             Links with EWMA-based adaptive distance tuning
 *
 *   Full:     gcc -DHFFT_USE_ENHANCED_THROTTLE -DHFFT_USE_ADAPTIVE_TUNING \
 *                 prefetch_strategy.c throttle_enhanced.c adaptive_tuning.c
 *             Uses all advanced features
 * 
 * @author Your Name
 * @date 2025
 */
/*
┌─────────────────────────────────┐
│  prefetch_strategy.c            │
│  ├─ init_prefetch_system()      │ Core initialization and CPU detection
│  ├─ prefetch_input()            │ Prefetch input data
│  ├─ prefetch_twiddle()          │ Prefetch twiddle factors
│  └─ get_stage_config()          │ Get per-stage configuration
└─────────────────────────────────┘
         │
         │ Optional: -DHFFT_USE_ENHANCED_THROTTLE
         ▼
┌─────────────────────────────────┐
│  throttle_enhanced.c            │ Advanced throttling module
│  ├─ prefetch_throttled_enhanced()│ Token bucket rate limiting
│  └─ Token bucket logic          │ Prevents prefetch buffer saturation
└─────────────────────────────────┘
         │
         │ Optional: -DHFFT_USE_ADAPTIVE_TUNING
         ▼
┌─────────────────────────────────┐
│  adaptive_tuning.c              │ Runtime optimization module
│  ├─ profile_fft_start/end()     │ Performance measurement
│  ├─ get_tuned_distance()        │ Optimal distance selection
│  └─ EWMA/per-stage tuning       │ Statistical optimization
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
// OPTIONAL MODULE INTEGRATION
//==============================================================================

#ifdef HFFT_USE_ENHANCED_THROTTLE
/**
 * @defgroup EnhancedThrottle Enhanced Throttling Module
 * @brief Token bucket-based prefetch throttling (optional)
 * @{
 */

/** @brief Initialize enhanced throttling system */
extern void init_throttling_enhanced(const cpu_profile_t *cpu_profile);

/** @brief Issue throttled prefetch with priority-based token bucket */
extern bool prefetch_throttled_enhanced(
    const void *addr, int hint, prefetch_priority_t priority,
    void (*do_prefetch_fn)(const void*, int));

/** @brief Set throttling mode (simple, token bucket, etc.) */
extern void set_throttle_mode(int mode);

/** @brief Configure token bucket parameters */
extern void configure_token_bucket(int l1, int l2, int nta, uint64_t refill, int tokens);

/** @brief Enable/disable throttling statistics collection */
extern void set_throttle_statistics(bool enable);

/** @brief Print accumulated throttling statistics */
extern void print_throttle_stats(void);

/** @brief Automatically tune token bucket parameters */
extern void autotune_token_bucket(void);

#define THROTTLE_MODE_TOKEN_BUCKET 1
/** @} */
#endif

#ifdef HFFT_USE_ADAPTIVE_TUNING
/**
 * @defgroup AdaptiveTuning Adaptive Tuning Module
 * @brief Runtime optimization of prefetch distances (optional)
 * @{
 */

/** @brief Initialize adaptive tuning system */
extern void init_adaptive_tuning(int num_stages, const int *distances, 
                                 const int *working_sets, const int *radixes);

/** @brief Cleanup adaptive tuning resources */
extern void cleanup_adaptive_tuning(void);

/** @brief Start performance profiling for FFT operation */
extern uint64_t profile_fft_start(void);

/** @brief End profiling and update statistics */
extern void profile_fft_end(uint64_t start, int n_elements, int stage_idx);

/** @brief Get optimized prefetch distance for stage */
extern int get_tuned_distance(int stage_idx);

/** @brief Apply tuned distances to stage configurations */
extern void apply_tuned_distances(stage_prefetch_t *stages, int n);

/** @brief Set tuning algorithm mode (EWMA, per-stage, etc.) */
extern void set_tuning_mode(int mode);

/** @brief Enable/disable detailed tuning logs */
extern void set_tuning_logging(bool enable);

/** @brief Print tuning performance report */
extern void print_tuning_report(void);

#define TUNING_MODE_EWMA 2       ///< Exponentially weighted moving average
#define TUNING_MODE_PER_STAGE 3  ///< Individual per-stage optimization
/** @} */
#endif

//==============================================================================
// INTERNAL TYPE DEFINITIONS (not in header)
//==============================================================================

/**
 * @brief Per-thread prefetch throttling state
 * 
 * Tracks issued prefetches within a sliding window to prevent
 * saturating the CPU's prefetch buffers (typically 10-20 entries).
 */
typedef struct {
    int issued_count;      ///< Total prefetches issued in current window
    int budget_remaining;  ///< Remaining budget in current window
    int window_size;       ///< Window size (iterations before reset)
    int window_counter;    ///< Current position in window
    int max_outstanding;   ///< Maximum allowed outstanding prefetches
} prefetch_throttle_t;

/**
 * @brief TLB (Translation Lookaside Buffer) prefetch state
 * 
 * Large-stride memory accesses can cause TLB misses. This mechanism
 * prefetches page table entries ahead of time.
 */
typedef struct {
    bool enabled;          ///< TLB prefetching active
    int page_distance;     ///< Pages to prefetch ahead
    int stride_threshold;  ///< Minimum stride to trigger TLB prefetch
    uintptr_t last_page;   ///< Last page number accessed
    const fft_data *base;  ///< Base pointer of data region
    size_t len_elements;   ///< Length of data region in elements
} tlb_prefetch_t;

/**
 * @brief Prefetch priority levels for throttling decisions
 * 
 * Critical prefetches bypass throttling, while low-priority
 * prefetches are dropped when buffers are saturated.
 */
typedef enum {
    PREFETCH_PRIO_CRITICAL = 0,  ///< Always issue (bypass throttle)
    PREFETCH_PRIO_HIGH = 1,      ///< High priority (rarely drop)
    PREFETCH_PRIO_MEDIUM = 2,    ///< Medium priority (drop under pressure)
    PREFETCH_PRIO_LOW = 3        ///< Low priority (drop aggressively)
} prefetch_priority_t;

/**
 * @brief Runtime profiling for adaptive tuning (built-in simple version)
 * 
 * Tracks performance metrics to automatically adjust prefetch distances.
 * Used only when optional adaptive tuning module is not compiled in.
 */
typedef struct {
    unsigned long long total_cycles;  ///< Accumulated CPU cycles
    unsigned long long total_calls;   ///< Number of FFT operations
    int current_distance;             ///< Current prefetch distance being tested
    int best_distance;                ///< Best distance found so far
    double best_throughput;           ///< Best throughput (cycles/element)
    int tuning_phase;                 ///< Current phase of tuning algorithm
    int tuning_iterations;            ///< Iterations in current phase
} prefetch_profile_t;

//==============================================================================
// GLOBAL STATE
//==============================================================================

/**
 * @brief Global prefetch configuration
 * 
 * Stores per-stage configurations and cache hierarchy information.
 * Initialized once during system startup and reused across FFT operations.
 */
static prefetch_config_t g_prefetch_config = {
    .stages = NULL,
    .num_stages = 0,
    .l1_size = 32 * 1024,      // Default: 32KB L1 data cache
    .l2_size = 256 * 1024,     // Default: 256KB L2 cache
    .l3_size = 8 * 1024 * 1024,// Default: 8MB L3 cache
    .cache_line_size = 64,
    .enable_runtime_tuning = false
};

/**
 * @brief CPU architecture profiles database
 * 
 * Contains empirically determined optimal parameters for various
 * CPU microarchitectures. Each profile includes:
 * - Number of prefetch buffers (hardware limitation)
 * - Latency characteristics of cache hierarchy
 * - Optimal prefetch distances for different scenarios
 * - Hardware prefetcher capabilities
 */
static const cpu_profile_t cpu_profiles[] = {
    // Intel architectures
    {
        .name = "Intel Skylake",
        .prefetch_buffers = 16,    // Skylake has 16 line fill buffers
        .prefetch_latency = 200,   // Cycles to fetch from memory
        .l1_latency = 4,           // L1 hit latency
        .l2_latency = 12,          // L2 hit latency
        .l3_latency = 42,          // L3 hit latency
        .has_write_prefetch = true,
        .has_strong_hwpf = true,   // Strong hardware prefetcher
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
        .prefetch_buffers = 12,    // Zen 2 has fewer buffers than Intel
        .prefetch_latency = 190,
        .l1_latency = 4,
        .l2_latency = 14,
        .l3_latency = 48,
        .has_write_prefetch = true,
        .has_strong_hwpf = false,  // AMD's hardware prefetcher is less aggressive
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
        .prefetch_buffers = 20,    // Apple Silicon has excellent prefetch
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

// Thread-local state (each thread maintains independent prefetch state)
static __thread prefetch_throttle_t g_throttle = {0};
static __thread tlb_prefetch_t g_tlb_prefetch = {0};
static __thread prefetch_profile_t g_prefetch_profile = {0};

#ifdef HFFT_USE_ADAPTIVE_TUNING
static __thread uint64_t g_profile_start_cycles = 0;
#endif

/**
 * @brief Wisdom database for storing optimal configurations
 * 
 * The wisdom database stores empirically determined optimal
 * prefetch parameters for specific FFT sizes and radixes.
 * This avoids re-tuning the same configurations repeatedly.
 */
#define MAX_WISDOM_ENTRIES 256
static wisdom_entry_t g_wisdom_db[MAX_WISDOM_ENTRIES];
static int g_wisdom_count = 0;
static pthread_mutex_t g_wisdom_mutex = PTHREAD_MUTEX_INITIALIZER;

//==============================================================================
// INTERNAL HELPER FUNCTIONS
//==============================================================================

/**
 * @brief Read CPU timestamp counter for high-precision timing
 * 
 * Returns CPU cycle count. On x86-64, uses RDTSC instruction.
 * On other architectures, returns 0 (profiling disabled).
 * 
 * @return Current CPU cycle count
 */
static inline unsigned long long read_tsc(void) {
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)hi << 32) | lo;
#else
    return 0;  // Profiling not supported on non-x86 platforms
#endif
}

/**
 * @brief Detect CPU cache sizes using CPUID instruction
 * 
 * Queries CPU cache topology on x86-64 systems using CPUID leaf 0x4.
 * Updates global cache size configuration with detected values.
 * 
 * Algorithm:
 * 1. Check if CPUID leaf 0x4 is available
 * 2. Iterate through cache descriptors (up to 10)
 * 3. Extract cache type, level, and size
 * 4. Update L1/L2/L3 sizes based on data caches
 */
void detect_cache_sizes(void) {
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int eax, ebx, ecx, edx;
    
    // Check maximum CPUID leaf
    __asm__ __volatile__ (
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(0x0)
    );
    
    if (eax >= 0x4) {
        // Iterate through cache descriptors
        for (int i = 0; i < 10; ++i) {
            __asm__ __volatile__ (
                "cpuid"
                : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                : "a"(0x4), "c"(i)  // Leaf 0x4, subleaf i
            );
            
            int cache_type = eax & 0x1F;
            if (cache_type == 0) break;  // No more caches
            
            int cache_level = (eax >> 5) & 0x7;
            
            // Calculate cache size:
            // size = (ways + 1) * (partitions + 1) * (line_size + 1) * (sets + 1)
            int cache_size = ((ebx >> 22) + 1) *     // Ways
                           ((ebx >> 12 & 0x3FF) + 1) *  // Partitions
                           ((ebx & 0xFFF) + 1) *        // Line size
                           (ecx + 1);                    // Sets
            
            // Type 1 = data cache, Type 3 = unified cache
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
 * @brief Detect CPU architecture and select optimal profile
 * 
 * Uses CPUID to identify CPU vendor and model, then matches against
 * the database of known CPU profiles.
 * 
 * Process:
 * 1. Read vendor string (CPUID leaf 0)
 * 2. Read brand string (CPUID leaves 0x80000002-0x80000004)
 * 3. Match against known Intel/AMD/ARM architectures
 * 4. Return corresponding profile or generic fallback
 * 
 * @return Pointer to matching CPU profile
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
    
    // Get brand string (48 characters across 3 CPUID calls)
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
    
    // Match Intel architectures
    if (strstr(vendor, "GenuineIntel")) {
        if (strstr(brand, "Sapphire Rapids") || strstr(brand, "Emerald Rapids"))
            return &cpu_profiles[2];
        else if (strstr(brand, "Ice Lake") || strstr(brand, "Tiger Lake"))
            return &cpu_profiles[1];
        else if (strstr(brand, "Skylake") || strstr(brand, "Cascade Lake"))
            return &cpu_profiles[0];
    }
    // Match AMD architectures
    else if (strstr(vendor, "AuthenticAMD")) {
        if (strstr(brand, "Zen 4") || strstr(brand, "Ryzen 7000"))
            return &cpu_profiles[5];
        else if (strstr(brand, "Zen 3") || strstr(brand, "Ryzen 5000"))
            return &cpu_profiles[4];
        else if (strstr(brand, "Zen 2") || strstr(brand, "Ryzen 3000"))
            return &cpu_profiles[3];
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    // ARM architecture detection
    #if defined(__APPLE__)
        return &cpu_profiles[7]; // M1/M2
    #else
        return &cpu_profiles[6]; // Neoverse V1
    #endif
#endif
    
    return &cpu_profiles[9]; // Generic fallback
}

/**
 * @brief Compute working set size for a specific FFT stage
 * 
 * The working set is the amount of data actively used during a stage.
 * This determines which cache level the data fits in, which in turn
 * determines optimal prefetch strategy.
 * 
 * @param n_fft Total FFT size
 * @param stage_idx Current stage index (0 = first stage)
 * @param factors Array of radix factors
 * @param num_factors Total number of factors
 * @return Working set size in bytes
 */
static inline int compute_stage_working_set(int n_fft, int stage_idx, 
                                            const int *factors, int num_factors) {
    // Compute size at current stage by dividing by previous radixes
    int n_stage = n_fft;
    for (int i = 0; i < stage_idx && i < num_factors; ++i) {
        n_stage /= factors[i];
    }
    return n_stage * sizeof(fft_data);
}

/**
 * @brief Compute optimal prefetch distance based on working set
 * 
 * Prefetch distance is how many iterations ahead we prefetch.
 * Too small: prefetch arrives too late (cache miss anyway)
 * Too large: evicts useful data, wastes prefetch buffers
 * 
 * Strategy:
 * - Tiny working sets (< L1/2): distance = 2 (hardware prefetcher sufficient)
 * - Small working sets (< L1): distance = 4
 * - Medium working sets (< L2): distance = 8-12
 * - Large working sets (< L3): distance = 16-24
 * - Huge working sets (> L3): distance = 32+ (aggressive prefetching needed)
 * 
 * @param working_set Size of data actively used (bytes)
 * @param stride Access stride (for strided access patterns)
 * @return Optimal prefetch distance (iterations ahead)
 */
static inline int compute_stage_prefetch_distance(int working_set, int stride) {
    const int l1 = g_prefetch_config.l1_size;
    const int l2 = g_prefetch_config.l2_size;
    const int l3 = g_prefetch_config.l3_size;
    
    int base_distance;
    if (working_set < l1 / 2) {
        base_distance = 2;   // Fits comfortably in L1
    } else if (working_set < l1) {
        base_distance = 4;   // Barely fits in L1
    } else if (working_set < l2 / 2) {
        base_distance = 8;   // Fits comfortably in L2
    } else if (working_set < l2) {
        base_distance = 12;  // Barely fits in L2
    } else if (working_set < l3 / 2) {
        base_distance = 16;  // Fits comfortably in L3
    } else if (working_set < l3) {
        base_distance = 24;  // Barely fits in L3
    } else {
        base_distance = 32;  // Spills to main memory
    }
    
    // Increase distance for strided access (more latency to hide)
    if (stride > 1) {
        base_distance += (stride / 2);
    }
    
    return base_distance;
}

/**
 * @brief Compute optimal prefetch hint based on working set
 * 
 * Prefetch hints tell the CPU which cache level to target:
 * - T0: Prefetch to L1 (all levels, temporal locality)
 * - T1: Prefetch to L2 (skip L1)
 * - T2: Prefetch to L3 (skip L1/L2)
 * - NTA: Non-temporal (bypass cache entirely)
 * 
 * @param working_set Size of data actively used (bytes)
 * @return Prefetch hint constant
 */
static inline int compute_stage_hint(int working_set) {
    const int l1 = g_prefetch_config.l1_size;
    const int l2 = g_prefetch_config.l2_size;
    const int l3 = g_prefetch_config.l3_size;
    
    if (working_set < l1) return _MM_HINT_T0;       // Target L1
    else if (working_set < l2) return _MM_HINT_T1;  // Target L2
    else if (working_set < l3) return _MM_HINT_T2;  // Target L3
    else return _MM_HINT_NTA;                       // Non-temporal (streaming)
}

/**
 * @brief Initialize throttling system (built-in simple version)
 * 
 * Sets up per-thread throttling state based on CPU's prefetch buffer count.
 * The throttle prevents issuing more prefetches than the hardware can handle.
 */
static inline void init_throttling(void) {
    g_throttle.max_outstanding = g_cpu_profile->prefetch_buffers;
    g_throttle.budget_remaining = g_throttle.max_outstanding;
    g_throttle.window_size = 8;      // Reset budget every 8 iterations
    g_throttle.window_counter = 0;
    g_throttle.issued_count = 0;
}

/**
 * @brief Initialize TLB prefetching for large data sets
 * 
 * For very large FFTs (> 1024 pages), TLB misses can become significant.
 * This mechanism prefetches page table entries ahead of time.
 * 
 * @param n_fft Total FFT size (number of complex elements)
 */
static inline void init_tlb_prefetch(int n_fft) {
    const int data_size = n_fft * sizeof(fft_data);
    const int page_size = 4096;
    const int num_pages = (data_size + page_size - 1) / page_size;
    
    g_tlb_prefetch.enabled = false;
    g_tlb_prefetch.page_distance = 8;  // Prefetch 8 pages ahead
    g_tlb_prefetch.stride_threshold = page_size / sizeof(fft_data);
    if (g_tlb_prefetch.stride_threshold == 0) {
        g_tlb_prefetch.stride_threshold = 1;
    }
    g_tlb_prefetch.last_page = 0;
    g_tlb_prefetch.base = NULL;
    g_tlb_prefetch.len_elements = 0;
    
    // Only enable for very large FFTs
    if (num_pages > 1024) {
        g_tlb_prefetch.len_elements = n_fft;
    }
}

/**
 * @brief Set TLB prefetch region
 * 
 * Defines the memory region for TLB prefetching.
 * 
 * @param base Base pointer of FFT data array
 * @param len_elems Length in complex elements
 */
void prefetch_set_tlb_region(const fft_data *base, size_t len_elems) {
    if (!base || len_elems == 0 || g_tlb_prefetch.stride_threshold == 0) {
        g_tlb_prefetch.enabled = false;
        return;
    }
    
    g_tlb_prefetch.base = base;
    g_tlb_prefetch.len_elements = len_elems;
    g_tlb_prefetch.enabled = true;
}

/**
 * @brief Check if we can issue a prefetch (built-in simple version)
 * 
 * Implements sliding window budget management. Every window_size iterations,
 * the budget resets to max_outstanding. This prevents buffer saturation.
 * 
 * @return true if prefetch should be issued, false if throttled
 */
static inline bool can_prefetch(void) {
    g_throttle.window_counter++;
    
    // Reset budget at window boundary
    if (g_throttle.window_counter >= g_throttle.window_size) {
        g_throttle.window_counter = 0;
        g_throttle.budget_remaining = g_throttle.max_outstanding;
        g_throttle.issued_count = 0;
    }
    
    return (g_throttle.budget_remaining > 0);
}

/**
 * @brief Record that a prefetch was issued (built-in simple version)
 * 
 * Decrements budget for non-critical prefetches.
 * Critical prefetches bypass the budget system.
 * 
 * @param critical true if this is a critical prefetch
 */
static inline void record_prefetch_issued(bool critical) {
    if (!critical) {
        if (g_throttle.budget_remaining > 0) {
            g_throttle.budget_remaining--;
        }
    }
    g_throttle.issued_count++;
}

/**
 * @brief Prefetch dispatch based on hint value
 * 
 * Issues actual prefetch instruction with appropriate hint.
 * On x86-64: uses _mm_prefetch intrinsic
 * On other platforms: uses __builtin_prefetch
 * 
 * @param addr Memory address to prefetch
 * @param hint Prefetch hint (T0, T1, T2, or NTA)
 */
static inline void do_prefetch(const void *addr, int hint) {
#if defined(__x86_64__) || defined(_M_X64)
    // x86-64: use specific intrinsics for each hint
    switch (hint) {
        case _MM_HINT_T0:  _mm_prefetch((const char*)addr, _MM_HINT_T0);  break;
        case _MM_HINT_T1:  _mm_prefetch((const char*)addr, _MM_HINT_T1);  break;
        case _MM_HINT_T2:  _mm_prefetch((const char*)addr, _MM_HINT_T2);  break;
        case _MM_HINT_NTA: _mm_prefetch((const char*)addr, _MM_HINT_NTA); break;
        default:           _mm_prefetch((const char*)addr, _MM_HINT_T0);  break;
    }
#else
    // Other platforms: map hints to GCC builtin locality levels (0-3)
    int locality = (hint==_MM_HINT_NTA) ? 0 : (hint==_MM_HINT_T2 ? 1 : (hint==_MM_HINT_T1 ? 2 : 3));
    __builtin_prefetch(addr, 0, locality);
#endif
}

/**
 * @brief Throttled prefetch with priority
 * 
 * Automatically uses enhanced throttling if compiled with HFFT_USE_ENHANCED_THROTTLE,
 * otherwise uses built-in simple throttling.
 * 
 * Priority system:
 * - CRITICAL: Always issued (bypass throttle)
 * - HIGH: Rarely dropped (only under severe pressure)
 * - MEDIUM: Dropped when buffers are half full
 * - LOW: Dropped aggressively to save bandwidth
 * 
 * @param addr Memory address to prefetch
 * @param hint Prefetch hint
 * @param priority Priority level for throttling decision
 */
static inline void prefetch_throttled(
    const void *addr,
    int hint,
    prefetch_priority_t priority
) {
#ifdef HFFT_USE_ENHANCED_THROTTLE
    // Use enhanced token bucket throttling
    prefetch_throttled_enhanced(addr, hint, priority, do_prefetch);
#else
    // Use built-in simple throttling
    if (priority == PREFETCH_PRIO_CRITICAL) {
        do_prefetch(addr, hint);
        record_prefetch_issued(true);
        return;
    }
    
    if (can_prefetch()) {
        do_prefetch(addr, hint);
        record_prefetch_issued(false);
    }
#endif
}

/**
 * @brief TLB prefetch for page table entries
 * 
 * When accessing memory with large strides, TLB misses can add significant
 * latency. This function prefetches the page containing future accesses.
 * 
 * @param addr Address being accessed (triggers page prefetch)
 */
static inline void prefetch_tlb(const fft_data *addr) {
    if (!g_tlb_prefetch.enabled) return;
    
    const uintptr_t page_sz = 4096;
    uintptr_t base_u = (uintptr_t)g_tlb_prefetch.base;
    uintptr_t end_u = base_u + g_tlb_prefetch.len_elements * sizeof(fft_data);
    
    // Calculate page number
    const uintptr_t page = (uintptr_t)addr / page_sz;
    
    // Prefetch only when crossing page boundaries
    if (page != g_tlb_prefetch.last_page) {
        // Prefetch N pages ahead
        const uintptr_t future = (page + (uintptr_t)g_tlb_prefetch.page_distance) * page_sz;
        
        // Bounds check
        if (future >= base_u && future < end_u) {
            __builtin_prefetch((const void*)future, 0, 0);
        }
        
        g_tlb_prefetch.last_page = page;
    }
}

/**
 * @brief Write prefetch for output data
 * 
 * Some CPUs support write prefetch (prefetchw/prefetcht0 with write intent).
 * This brings cache lines into exclusive state, avoiding RFO (Read-For-Ownership).
 * 
 * @param output Output array
 * @param idx Current index
 * @param distance How many iterations ahead to prefetch
 * @param cfg Stage configuration
 */
static inline void prefetch_write(
    fft_data *output,
    int idx,
    int distance,
    stage_prefetch_t *cfg
) {
    if (!cfg || !cfg->enable) return;
    
#if defined(__PRFCHW__) || defined(__3dNOW__)
    // AMD 3DNow! or PREFETCHW: explicit write prefetch
    __builtin_prefetch(output + idx + distance, 1, 0);
#else
    // Fallback: use non-temporal prefetch
    do_prefetch(output + idx + distance, _MM_HINT_NTA);
#endif
}

//==============================================================================
// PUBLIC API IMPLEMENTATION
//==============================================================================

/**
 * @brief Initialize prefetch system for an FFT object
 * 
 * This is the main initialization function that must be called before
 * performing any FFT operations. It performs the following tasks:
 * 
 * 1. CPU Detection (once per process):
 *    - Identify CPU vendor and model
 *    - Select optimal CPU profile from database
 *    - Detect cache hierarchy (L1/L2/L3 sizes)
 * 
 * 2. Throttling Initialization (per thread):
 *    - Set up prefetch buffer management
 *    - Configure token bucket (if enhanced throttling enabled)
 *    - Initialize statistics collection
 * 
 * 3. TLB Prefetch Setup:
 *    - Enable for large FFTs (> 1024 pages)
 *    - Configure page prefetch distance
 * 
 * 4. Wisdom Database Loading (once per process):
 *    - Load previously saved optimal configurations
 *    - Apply to matching FFT sizes
 * 
 * 5. Per-Stage Configuration:
 *    - Compute working set for each FFT stage
 *    - Determine optimal prefetch distance and hint
 *    - Select strategy (none, single, dual, multi)
 *    - Apply wisdom if available
 * 
 * 6. Adaptive Tuning Setup (if enabled):
 *    - Initialize per-stage profiling
 *    - Set up EWMA or per-stage optimization
 *    - Enable logging if requested
 * 
 * @param fft_obj FFT object containing size, factors, and decomposition info
 */
void init_prefetch_system(fft_object fft_obj) {
    // STEP 1: Detect CPU architecture (once per process)
    static int cpu_detected = 0;
    if (!cpu_detected) {
        g_cpu_profile = detect_cpu_architecture();
        cpu_detected = 1;
        
        fprintf(stderr, "HFFT: Detected CPU: %s\n", g_cpu_profile->name);
    }
    
    // STEP 2: Detect cache sizes (once per process)
    static int cache_detected = 0;
    if (!cache_detected) {
        detect_cache_sizes();
        cache_detected = 1;
    }
    
    // STEP 3: Initialize throttling (per thread, built-in or enhanced)
#ifdef HFFT_USE_ENHANCED_THROTTLE
    init_throttling_enhanced(g_cpu_profile);
    
    // Check environment variable for enhanced throttle mode
    const char *throttle_mode = getenv("HFFT_THROTTLE_MODE");
    if (throttle_mode && strcmp(throttle_mode, "token") == 0) {
        set_throttle_mode(THROTTLE_MODE_TOKEN_BUCKET);
        fprintf(stderr, "HFFT: Using token bucket throttling\n");
    }
    
    const char *throttle_stats = getenv("HFFT_THROTTLE_STATS");
    if (throttle_stats && atoi(throttle_stats) == 1) {
        set_throttle_statistics(true);
    }
#else
    init_throttling();
#endif
    
    // STEP 4: Initialize TLB prefetch
    init_tlb_prefetch(fft_obj->n_fft);
    
    // STEP 5: Load wisdom database (once per process)
    static int wisdom_loaded = 0;
    if (!wisdom_loaded) {
        const char *wisdom_file = getenv("HFFT_WISDOM_FILE");
        if (!wisdom_file) wisdom_file = "hfft_wisdom.txt";
        load_wisdom(wisdom_file);
        wisdom_loaded = 1;
    }
    
    // STEP 6: Initialize per-stage configurations
    const int n_fft = fft_obj->n_fft;
    const int num_factors = fft_obj->lf;
    const int *factors = fft_obj->factors;
    
    // Free previous configuration if it exists
    if (g_prefetch_config.stages) {
        free(g_prefetch_config.stages);
    }
    
    // Allocate per-stage configurations
    g_prefetch_config.stages = (stage_prefetch_t *)malloc(
        num_factors * sizeof(stage_prefetch_t)
    );
    g_prefetch_config.num_stages = num_factors;
    
    if (!g_prefetch_config.stages) return;
    
    // Prepare arrays for adaptive tuning (if enabled)
    int *initial_distances = NULL;
    int *working_sets = NULL;
    int *radixes = NULL;
    
#ifdef HFFT_USE_ADAPTIVE_TUNING
    initial_distances = (int*)malloc(num_factors * sizeof(int));
    working_sets = (int*)malloc(num_factors * sizeof(int));
    radixes = (int*)malloc(num_factors * sizeof(int));
#endif
    
    // STEP 7: Configure each stage
    int stride = 1;  // Stride increases with each radix factor
    for (int stage = 0; stage < num_factors; ++stage) {
        const int radix = factors[stage];
        const int working_set = compute_stage_working_set(n_fft, stage, factors, num_factors);
        
        stage_prefetch_t *cfg = &g_prefetch_config.stages[stage];
        
        // Try to find wisdom for this configuration
        wisdom_entry_t *wisdom = find_wisdom(n_fft, radix);
        
        if (wisdom) {
            // Use saved optimal configuration
            cfg->distance_input = wisdom->distance_input;
            cfg->distance_twiddle = wisdom->distance_twiddle;
            cfg->hint_input = wisdom->hint;
            cfg->hint_twiddle = _MM_HINT_T0;
            cfg->strategy = wisdom->strategy;
            cfg->enable = true;
        } else {
            // Use heuristics to determine strategy
            if (working_set < 1024) {
                // Tiny working set: hardware prefetcher is sufficient
                cfg->strategy = PREFETCH_NONE;
            } else if (radix <= 4 && working_set < g_prefetch_config.l1_size) {
                // Small radix, fits in L1: simple prefetch
                cfg->strategy = PREFETCH_SINGLE;
            } else if (radix <= 8) {
                // Medium radix: prefetch input and twiddle
                cfg->strategy = PREFETCH_DUAL;
            } else {
                // Large radix: multi-stream prefetch
                cfg->strategy = PREFETCH_MULTI;
            }
            
            // Compute distances and hints
            cfg->distance_input = compute_stage_prefetch_distance(working_set, stride);
            cfg->distance_output = cfg->distance_input;
            cfg->distance_twiddle = cfg->distance_input / 2;  // Twiddles accessed less frequently
            
            cfg->hint_input = compute_stage_hint(working_set);
            cfg->hint_output = cfg->hint_input;
            cfg->hint_twiddle = _MM_HINT_T0;  // Twiddles should stay in L1
            
            // Disable prefetch if hardware prefetcher is strong and data fits in L1
            if (g_cpu_profile->has_strong_hwpf && working_set < g_prefetch_config.l1_size) {
                cfg->enable = false;
            } else {
                cfg->enable = (cfg->strategy != PREFETCH_NONE);
            }
        }
        
        // Configure blocking for large radixes
        if (radix >= 16 && working_set > g_prefetch_config.l2_size) {
            cfg->block_size = g_prefetch_config.l2_size / (radix * sizeof(fft_data));
        } else {
            cfg->block_size = 0;  // No blocking
        }
        
#ifdef HFFT_USE_ADAPTIVE_TUNING
        // Store initial values for adaptive tuning
        if (initial_distances) initial_distances[stage] = cfg->distance_input;
        if (working_sets) working_sets[stage] = working_set;
        if (radixes) radixes[stage] = radix;
#endif
        
        stride *= radix;  // Stride increases by radix factor
    }
    
    // STEP 8: Initialize adaptive tuning if enabled
#ifdef HFFT_USE_ADAPTIVE_TUNING
    if (initial_distances && working_sets && radixes) {
        init_adaptive_tuning(num_factors, initial_distances, working_sets, radixes);
        
        // Check environment variable for tuning mode
        const char *tuning_mode = getenv("HFFT_TUNING_MODE");
        if (tuning_mode) {
            if (strcmp(tuning_mode, "ewma") == 0) {
                set_tuning_mode(TUNING_MODE_EWMA);
                fprintf(stderr, "HFFT: Using EWMA adaptive tuning\n");
            } else if (strcmp(tuning_mode, "per_stage") == 0) {
                set_tuning_mode(TUNING_MODE_PER_STAGE);
                fprintf(stderr, "HFFT: Using per-stage adaptive tuning\n");
            }
        }
        
        const char *tuning_log = getenv("HFFT_TUNING_LOG");
        if (tuning_log && atoi(tuning_log) == 1) {
            set_tuning_logging(true);
        }
    }
    
    free(initial_distances);
    free(working_sets);
    free(radixes);
#endif
}

/**
 * @brief Cleanup prefetch system resources
 * 
 * Frees all allocated memory and cleans up optional modules.
 * Should be called when FFT object is destroyed.
 */
void cleanup_prefetch_system(void) {
    if (g_prefetch_config.stages) {
        free(g_prefetch_config.stages);
        g_prefetch_config.stages = NULL;
        g_prefetch_config.num_stages = 0;
    }
    
#ifdef HFFT_USE_ADAPTIVE_TUNING
    cleanup_adaptive_tuning();
#endif
}

/**
 * @brief Get prefetch configuration for a specific stage
 * 
 * @param factor_index Index of FFT factor/stage (0-based)
 * @return Pointer to stage configuration, or NULL if invalid index
 */
stage_prefetch_t* get_stage_config(int factor_index) {
    if (factor_index < 0 || factor_index >= g_prefetch_config.num_stages) {
        return NULL;
    }
    return &g_prefetch_config.stages[factor_index];
}

/**
 * @brief Prefetch input data for FFT computation
 * 
 * Issues prefetch for input array elements that will be accessed
 * in future iterations. Distance and hint are determined by stage config.
 * 
 * @param input Input data array
 * @param idx Current iteration index
 * @param cfg Stage prefetch configuration
 */
void prefetch_input(const fft_data *input, int idx, stage_prefetch_t *cfg) {
    if (!cfg || !cfg->enable) return;
    
#ifdef HFFT_USE_ADAPTIVE_TUNING
    // Use adaptively tuned distance if available
    int tuned = get_tuned_distance(cfg - g_prefetch_config.stages);
    if (tuned > 0) {
        cfg->distance_input = tuned;
    }
#endif
    
    do_prefetch(input + idx + cfg->distance_input, cfg->hint_input);
}

/**
 * @brief Prefetch twiddle factors for FFT computation
 * 
 * Twiddle factors are complex exponentials used in butterfly operations.
 * They are accessed with different patterns than input data.
 * 
 * @param twiddle Twiddle factor array
 * @param idx Current iteration index
 * @param cfg Stage prefetch configuration
 */
void prefetch_twiddle(const fft_data *twiddle, int idx, stage_prefetch_t *cfg) {
    if (!cfg || !cfg->enable) return;
    do_prefetch(twiddle + idx + cfg->distance_twiddle, cfg->hint_twiddle);
}

/**
 * @brief Prefetch for recursive FFT stage with multiple streams
 * 
 * In recursive FFT algorithms, data is accessed in multiple parallel
 * streams separated by stride. This function prefetches across streams.
 * 
 * Strategy:
 * - For multi-stream strategy and large radix: prefetch multiple lanes
 * - For large strides: enable TLB prefetching
 * - Twiddle prefetch only for DUAL or MULTI strategies
 * 
 * @param input_base Base pointer to input data
 * @param twiddle_base Base pointer to twiddle factors (can be NULL)
 * @param idx Current index within stage
 * @param stride Stride between parallel streams
 * @param radix Number of parallel streams (radix of butterfly)
 * @param cfg Stage prefetch configuration
 */
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
    
    // Enable TLB prefetch for large-stride access
    if (stride >= g_tlb_prefetch.stride_threshold) {
        prefetch_tlb(input_base + idx + d_in);
    }
    
    // Multi-stream prefetch for large radixes
    if (cfg->strategy >= PREFETCH_MULTI && radix > 4) {
        // Prefetch up to 4 lanes (hardware limit on most CPUs)
        for (int lane = 0; lane < radix && lane < 4; ++lane) {
            prefetch_throttled(
                input_base + (idx + d_in) + lane * stride,
                cfg->hint_input,
                PREFETCH_PRIO_CRITICAL
            );
        }
    } else {
        // Single-stream prefetch
        prefetch_throttled(
            input_base + idx + d_in,
            cfg->hint_input,
            PREFETCH_PRIO_CRITICAL
        );
    }
    
    // Prefetch twiddle factors if strategy requires it
    if (twiddle_base && cfg->strategy >= PREFETCH_DUAL) {
        prefetch_throttled(
            twiddle_base + idx + d_tw,
            cfg->hint_twiddle,
            PREFETCH_PRIO_HIGH
        );
    }
}

/**
 * @brief Prefetch for butterfly loop iteration
 * 
 * Simple prefetch for basic butterfly loops. Prefetches input
 * and optionally twiddles based on strategy.
 * 
 * @param input Input data array
 * @param twiddle Twiddle factor array (can be NULL)
 * @param idx Current iteration index
 * @param cfg Stage prefetch configuration
 */
void prefetch_butterfly_loop(
    const fft_data *input,
    const fft_data *twiddle,
    int idx,
    stage_prefetch_t *cfg
) {
    if (!cfg || !cfg->enable) return;
    
    // Always prefetch input
    prefetch_throttled(
        input + idx + cfg->distance_input,
        cfg->hint_input,
        PREFETCH_PRIO_CRITICAL
    );
    
    // Conditionally prefetch twiddle
    if (twiddle && cfg->strategy >= PREFETCH_DUAL) {
        prefetch_throttled(
            twiddle + idx + cfg->distance_twiddle,
            cfg->hint_twiddle,
            PREFETCH_PRIO_HIGH
        );
    }
}

//==============================================================================
// WISDOM DATABASE FUNCTIONS
//==============================================================================

/**
 * @brief Load wisdom database from file
 * 
 * Wisdom file format (plain text):
 * # Comment lines start with #
 * n_fft radix dist_in dist_tw hint strategy cycles timestamp
 * 
 * Example:
 * 1024 4 8 4 0 2 12.345678 1234567890
 * 
 * @param filename Path to wisdom file
 */
void load_wisdom(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return;
    
    pthread_mutex_lock(&g_wisdom_mutex);
    
    g_wisdom_count = 0;
    char line[256];
    
    // Read entries line by line
    while (g_wisdom_count < MAX_WISDOM_ENTRIES && fgets(line, sizeof(line), fp)) {
        // Skip comments and empty lines
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        
        wisdom_entry_t *entry = &g_wisdom_db[g_wisdom_count];
        
        int strategy_int;
        long long timestamp_ll;
        int items = sscanf(line, "%d %d %d %d %d %d %lf %lld",
            &entry->n_fft,
            &entry->radix,
            &entry->distance_input,
            &entry->distance_twiddle,
            &entry->hint,
            &strategy_int,
            &entry->cycles_per_element,
            &timestamp_ll
        );
        
        if (items == 8) {
            entry->strategy = (prefetch_strategy_t)strategy_int;
            entry->timestamp = (time_t)timestamp_ll;
            g_wisdom_count++;
        }
    }
    
    pthread_mutex_unlock(&g_wisdom_mutex);
    fclose(fp);
}

/**
 * @brief Save wisdom database to file
 * 
 * @param filename Path to wisdom file
 */
void save_wisdom(const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) return;
    
    pthread_mutex_lock(&g_wisdom_mutex);
    
    fprintf(fp, "# HFFT Wisdom Database\n");
    fprintf(fp, "# Format: n_fft radix dist_in dist_tw hint strategy cycles timestamp\n");
    
    for (int i = 0; i < g_wisdom_count; i++) {
        wisdom_entry_t *entry = &g_wisdom_db[i];
        fprintf(fp, "%d %d %d %d %d %d %.6f %lld\n",
            entry->n_fft,
            entry->radix,
            entry->distance_input,
            entry->distance_twiddle,
            entry->hint,
            (int)entry->strategy,
            entry->cycles_per_element,
            (long long)entry->timestamp
        );
    }
    
    pthread_mutex_unlock(&g_wisdom_mutex);
    fclose(fp);
}

/**
 * @brief Find wisdom entry for specific FFT size and radix
 * 
 * @param n_fft FFT size
 * @param radix Radix factor
 * @return Pointer to wisdom entry, or NULL if not found
 */
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

/**
 * @brief Add new wisdom entry to database
 * 
 * @param n_fft FFT size
 * @param radix Radix factor
 * @param distance_input Optimal input prefetch distance
 * @param distance_twiddle Optimal twiddle prefetch distance
 * @param hint Optimal prefetch hint
 * @param strategy Optimal prefetch strategy
 * @param cycles_per_element Measured performance (cycles per element)
 */
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

/**
 * @brief Search for optimal prefetch configuration
 * 
 * Tries wisdom database first, falls back to heuristics if not found.
 * 
 * @param fft_obj FFT object
 * @param factor_index Stage/factor index
 * @param best_config Output: optimal configuration
 */
void search_optimal_prefetch(
    fft_object fft_obj,
    int factor_index,
    stage_prefetch_t *best_config
) {
    const int radix = fft_obj->factors[factor_index];
    wisdom_entry_t *wisdom = find_wisdom(fft_obj->n_fft, radix);
    
    if (wisdom) {
        // Use wisdom
        *best_config = (stage_prefetch_t){
            .enable = true,
            .strategy = wisdom->strategy,
            .distance_input = wisdom->distance_input,
            .distance_output = wisdom->distance_input,
            .distance_twiddle = wisdom->distance_twiddle,
            .hint_input = wisdom->hint,
            .hint_output = wisdom->hint,
            .hint_twiddle = _MM_HINT_T0,
            .block_size = 0
        };
        return;
    }
    
    // Use heuristics
    const int n_fft = fft_obj->n_fft;
    const int working_set = compute_stage_working_set(
        n_fft, factor_index, fft_obj->factors, fft_obj->lf
    );
    
    if (working_set < 1024) {
        best_config->strategy = PREFETCH_NONE;
        best_config->enable = false;
    } else if (radix <= 4 && working_set < g_prefetch_config.l1_size) {
        best_config->strategy = PREFETCH_SINGLE;
        best_config->enable = true;
    } else if (radix <= 8) {
        best_config->strategy = PREFETCH_DUAL;
        best_config->enable = true;
    } else {
        best_config->strategy = PREFETCH_MULTI;
        best_config->enable = true;
    }
    
    best_config->distance_input = compute_stage_prefetch_distance(working_set, 1);
    best_config->distance_output = best_config->distance_input;
    best_config->distance_twiddle = best_config->distance_input / 2;
    best_config->hint_input = compute_stage_hint(working_set);
    best_config->hint_output = best_config->hint_input;
    best_config->hint_twiddle = _MM_HINT_T0;
    best_config->block_size = 0;
}

/**
 * @brief Get global prefetch configuration
 * @return Pointer to global configuration structure
 */
const prefetch_config_t* get_prefetch_config(void) {
    return &g_prefetch_config;
}

/**
 * @brief Get detected CPU profile
 * @return Pointer to CPU profile structure
 */
const cpu_profile_t* get_cpu_profile(void) {
    return g_cpu_profile;
}

/**
 * @brief Enable or disable prefetching globally
*/
void set_prefetch_enable(bool enable) {
    for (int i = 0; i < g_prefetch_config.num_stages; i++) {
        g_prefetch_config.stages[i].enable = enable;
    }
}

//==============================================================================
// RUNTIME PROFILING AND ADAPTIVE TUNING
//==============================================================================

void profile_start(void) {
#ifdef HFFT_USE_ADAPTIVE_TUNING
    // Use enhanced adaptive tuning module
    profile_fft_start();
#else
    // Use built-in simple profiling
    if (!g_prefetch_config.enable_runtime_tuning) return;
    g_prefetch_profile.total_cycles = read_tsc();
#endif
}

void profile_end(int n_elements) {
#ifdef HFFT_USE_ADAPTIVE_TUNING
    // Use enhanced adaptive tuning module
    profile_fft_end(0, n_elements, -1);
    
    // Apply tuned distances periodically
    static int call_count = 0;
    if (++call_count % 100 == 0) {
        apply_tuned_distances(g_prefetch_config.stages, g_prefetch_config.num_stages);
    }
#else
    // Use built-in simple profiling
    if (!g_prefetch_config.enable_runtime_tuning) return;
    
    unsigned long long end = read_tsc();
    unsigned long long elapsed = end - g_prefetch_profile.total_cycles;
    
    g_prefetch_profile.total_calls++;
    
    double throughput = (double)elapsed / (double)n_elements;
    
    switch (g_prefetch_profile.tuning_phase) {
        case 0:
            g_prefetch_profile.best_throughput = throughput;
            g_prefetch_profile.best_distance = g_prefetch_profile.current_distance;
            g_prefetch_profile.tuning_phase = 1;
            g_prefetch_profile.tuning_iterations = 0;
            break;
            
        case 1:
            if (throughput < g_prefetch_profile.best_throughput * 0.98) {
                g_prefetch_profile.best_throughput = throughput;
                g_prefetch_profile.best_distance = g_prefetch_profile.current_distance;
                g_prefetch_profile.tuning_iterations = 0;
            } else {
                g_prefetch_profile.tuning_iterations++;
            }
            
            if (g_prefetch_profile.tuning_iterations < 5) {
                g_prefetch_profile.current_distance += 4;
            } else if (g_prefetch_profile.tuning_iterations < 10) {
                g_prefetch_profile.current_distance -= 2;
            } else {
                g_prefetch_profile.tuning_phase = 2;
                g_prefetch_profile.current_distance = g_prefetch_profile.best_distance;
                
                for (int i = 0; i < g_prefetch_config.num_stages; ++i) {
                    g_prefetch_config.stages[i].distance_input = g_prefetch_profile.best_distance;
                }
            }
            break;
            
        case 2:
            if (g_prefetch_profile.total_calls % 1000 == 0) {
                g_prefetch_profile.tuning_phase = 1;
                g_prefetch_profile.tuning_iterations = 0;
            }
            break;
    }
#endif
}

//==============================================================================
// STATISTICS AND REPORTING
//==============================================================================

void print_prefetch_statistics(void) {
#ifdef HFFT_USE_ENHANCED_THROTTLE
    print_throttle_stats();
#endif

#ifdef HFFT_USE_ADAPTIVE_TUNING
    print_tuning_report();
#endif

    printf("\n=== Prefetch Configuration ===\n");
    printf("CPU: %s\n", g_cpu_profile->name);
    printf("Stages: %d\n", g_prefetch_config.num_stages);
    printf("==============================\n\n");
}

//==============================================================================
// ADVANCED PREFETCH OPERATIONS
//==============================================================================

int adjust_distance_for_unroll(int base_distance, int unroll_factor) {
    if (unroll_factor <= 2) return base_distance;
    else if (unroll_factor <= 4) return base_distance + 2;
    else if (unroll_factor <= 8) return base_distance + 4;
    else return base_distance + 8;
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
    const int cl_per_stride = (stride * sizeof(fft_data) + 63) / 64;
    
    for (int s = 0; s < num_streams && s < 4; ++s) {
        const fft_data *ptr = base + (idx + d) + s * stride;
        do_prefetch(ptr, hint);
        
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
    
    const int next_block = block_start + block_size;
    const int hint = cfg->hint_input;
    
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
    
#ifdef HFFT_USE_ENHANCED_THROTTLE
    printf("Enhanced throttling: ENABLED\n");
#else
    printf("Enhanced throttling: disabled\n");
#endif

#ifdef HFFT_USE_ADAPTIVE_TUNING
    printf("Adaptive tuning: ENABLED\n");
#else
    printf("Adaptive tuning: disabled\n");
#endif
    
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
